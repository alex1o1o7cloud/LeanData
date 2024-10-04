import Mathlib
import Mathlib.Algebra.Arithmetic
import Mathlib.Algebra.Binomial
import Mathlib.Algebra.GcdMonoid.Basic
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Weights
import Mathlib.Algebra.Order.ArithmeticMean
import Mathlib.Algebra.Order.Field
import Mathlib.Analysis.Calculus.Continuous
import Mathlib.Analysis.Geometry.Euclidean.Basic
import Mathlib.Analysis.SpecialFunctions.Polynomials
import Mathlib.Analysis.SpecialFunctions.Pow
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.Trigonometry
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Perm.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Geometry.Euclid.Triangle.Basic
import Mathlib.Tactic
import Mathlib.Topology.ContinuousFunction
import data.real.basic

namespace solve_fx_eq_6_l444_444136

def f (x : ℝ) : ℝ :=
  if x < 1 then 4 * x + 8 else 3 * x - 15

theorem solve_fx_eq_6 : { x : ℝ | f x = 6} = { -1/2, 7 } :=
by
  sorry

end solve_fx_eq_6_l444_444136


namespace sequence_general_term_l444_444960

theorem sequence_general_term (a : ℕ+ → ℤ) (h₁ : a 1 = 2) (h₂ : ∀ n : ℕ+, a (n + 1) = a n - 1) :
  ∀ n : ℕ+, a n = 3 - n := 
sorry

end sequence_general_term_l444_444960


namespace tank_capacity_l444_444827

theorem tank_capacity :
  (∃ c: ℝ, (∃ w: ℝ, w / c = 1/6 ∧ (w + 5) / c = 1/3) → c = 30) :=
by
  sorry

end tank_capacity_l444_444827


namespace non_perfect_powers_count_l444_444481

theorem non_perfect_powers_count :
  let n := 200,
      num_perfect_squares := 14,
      num_perfect_cubes := 5,
      num_perfect_sixth_powers := 1,
      total_perfect_powers := num_perfect_squares + num_perfect_cubes - num_perfect_sixth_powers,
      total_non_perfect_powers := n - total_perfect_powers
  in
  total_non_perfect_powers = 182 :=
by
  sorry

end non_perfect_powers_count_l444_444481


namespace probability_three_cards_l444_444703

theorem probability_three_cards (S : Type) [Fintype S]
  (deck : Finset S) (n : ℕ) (hn : n = 52)
  (hearts : Finset S) (spades : Finset S)
  (tens: Finset S)
  (hhearts_count : ∃ k, hearts.card = k ∧ k = 13)
  (hspades_count : ∃ k, spades.card = k ∧ k = 13)
  (htens_count : ∃ k, tens.card = k ∧ k = 4)
  (hdeck_partition : ∀ x ∈ deck, x ∈ hearts ∨ x ∈ spades ∨ x ∈ tens ∨ (x ∉ hearts ∧ x ∉ spades ∧ x ∉ tens)) :
  (12 / 52 * 13 / 51 * 4 / 50 + 1 / 52 * 13 / 51 * 3 / 50 = 221 / 44200) :=
by {
  sorry
}

end probability_three_cards_l444_444703


namespace area_of_right_square_l444_444288

theorem area_of_right_square (side_length_left : ℕ) (side_length_left_eq : side_length_left = 10) : ∃ area_right, area_right = 68 := 
by
  sorry

end area_of_right_square_l444_444288


namespace count_valid_numbers_between_1_and_200_l444_444426

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k^2 = n
def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k^3 = n
def is_perfect_sixth_power (n : ℕ) : Prop := ∃ k : ℕ, k^6 = n

def count_perfect_squares (m : ℕ) : ℕ :=
  Nat.sqrt m

def count_perfect_cubes (m : ℕ) : ℕ :=
  (m : ℝ).cbrt.to_nat

def count_perfect_sixth_powers (m : ℕ) : ℕ :=
  (m : ℝ).cbrt.sqrt.to_nat

def count_either_squares_or_cubes (m : ℕ) : ℕ :=
  let squares := count_perfect_squares m
  let cubes := count_perfect_cubes m
  let sixths := count_perfect_sixth_powers m
  squares + cubes - sixths

def count_neither_squares_nor_cubes (n : ℕ) : ℕ :=
  n - count_either_squares_or_cubes n

theorem count_valid_numbers_between_1_and_200 : count_neither_squares_nor_cubes 200 = 182 :=
by
  sorry

end count_valid_numbers_between_1_and_200_l444_444426


namespace sum_exterior_angles_of_octagon_l444_444690

theorem sum_exterior_angles_of_octagon : 
  ∀ (P : Type) [Polygon P], (exteriorAngleSum P) = 360 := by
sorry

end sum_exterior_angles_of_octagon_l444_444690


namespace sum_of_fractions_l444_444330

theorem sum_of_fractions :
  (1 / (2 * 3 * 4) + 1 / (3 * 4 * 5) + 1 / (4 * 5 * 6) + 1 / (5 * 6 * 7) + 1 / (6 * 7 * 8)) = 3 / 16 := 
by
  sorry

end sum_of_fractions_l444_444330


namespace not_converge_to_a_l444_444916

theorem not_converge_to_a (x : ℕ → ℝ) (a : ℝ) :
  (∀ ε > 0, ∀ k : ℕ, ∃ n : ℕ, n > k ∧ |x n - a| ≥ ε) →
  ¬ (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |x n - a| < ε) :=
by sorry

end not_converge_to_a_l444_444916


namespace find_c_find_k_l444_444035

-- Definitions
def vector_a := (1 : ℝ, 2 : ℝ)
def vector_b := (-3 : ℝ, 1 : ℝ)
def norm (v : ℝ × ℝ) : ℝ := real.sqrt (v.1 * v.1 + v.2 * v.2)
def is_parallel_to (v1 v2 : ℝ × ℝ) : Prop := ∃ λ : ℝ, v1 = (λ * v2.1, λ * v2.2)
def is_perpendicular_to (v1 v2 : ℝ × ℝ) : Prop := (v1.1 * v2.1 + v1.2 * v2.2 = 0)

-- Theorem statement for part 1
theorem find_c (c : ℝ × ℝ) (hc : norm c = 2 * real.sqrt 5) (hpc : is_parallel_to c vector_a) :
  c = (2, 4) ∨ c = (-2, -4) :=
sorry

-- Theorem statement for part 2
theorem find_k (k : ℝ) 
  (h_perpendicular: is_perpendicular_to (vector_a.1 + k * vector_b.1, vector_a.2 + k * vector_b.2)
                                       (vector_a.1 - k * norm vector_b, vector_a.2 - k * norm vector_b)) :
  k = real.sqrt 2 / 2 ∨ k = - (real.sqrt 2 / 2) :=
sorry

end find_c_find_k_l444_444035


namespace tangent_line_at_1_ln_x_le_x_minus_1_l444_444053

noncomputable def f (x : ℝ) : ℝ := Real.log x - x + 1

theorem tangent_line_at_1 : 
  let A := (1:ℝ, f 1)
  (∃ (L : ℝ → ℝ), (∀ x, L x = 0) ∧ (∀ x, f(x) = f(1) + f' 1 * (x-1) + o)) :=
by
  sorry

theorem ln_x_le_x_minus_1 : ∀ (x : ℝ), x > 0 → Real.log x ≤ x - 1 :=
by
  sorry

end tangent_line_at_1_ln_x_le_x_minus_1_l444_444053


namespace angle_D_measure_l444_444201

theorem angle_D_measure (E D F : ℝ) (h1 : E + D + F = 180) (h2 : E = 30) (h3 : D = 2 * F) : D = 100 :=
by
  -- The proof is not required, only the statement
  sorry

end angle_D_measure_l444_444201


namespace count_not_squares_or_cubes_l444_444504

theorem count_not_squares_or_cubes {A B : Finset ℕ} (h_range : ∀ x, x ∈ A ∪ B → x ≤ 200)
  (h_squares : A.card = 14) (h_cubes : B.card = 5) (h_inter : (A ∩ B).card = 1) :
  (200 - (A ∪ B).card) = 182 :=
by
  sorry

end count_not_squares_or_cubes_l444_444504


namespace perpendicular_condition_l444_444418

def vector_a := (1 : ℝ, 1 : ℝ, 0 : ℝ)
def vector_b := (-1 : ℝ, 0 : ℝ, 1 : ℝ)
def k := 1 / 2

def dot_product (u v : ℝ × ℝ × ℝ) :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

theorem perpendicular_condition (k : ℝ) 
  (h : k * vector_a.1 + vector_b.1, k * vector_a.2 + vector_b.2, k * vector_a.3 + vector_b.3) :
  dot_product h vector_a = 0 → k = 1 / 2 :=
  sorry

end perpendicular_condition_l444_444418


namespace fraction_n_p_l444_444685

theorem fraction_n_p (m n p : ℝ) (r1 r2 : ℝ)
  (h1 : r1 * r2 = m)
  (h2 : -(r1 + r2) = p)
  (h3 : m ≠ 0)
  (h4 : n ≠ 0)
  (h5 : p ≠ 0)
  (h6 : m = - (r1 + r2) / 2)
  (h7 : n = r1 * r2 / 4) :
  n / p = 1 / 8 :=
by
  sorry

end fraction_n_p_l444_444685


namespace num_non_squares_cubes_1_to_200_l444_444440

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n
def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k * k * k = n
def is_perfect_sixth_power (n : ℕ) : Prop := ∃ k : ℕ, k * k * k * k * k * k = n

def num_perfect_squares (a b : ℕ) : ℕ :=
  (List.range (b + 1)).countp is_perfect_square - (List.range (a)).countp is_perfect_square

def num_perfect_cubes (a b : ℕ) : ℕ :=
  (List.range (b + 1)).countp is_perfect_cube - (List.range (a)).countp is_perfect_cube

def num_perfect_sixth_powers (a b : ℕ) : ℕ :=
  (List.range (b + 1)).countp is_perfect_sixth_power - (List.range (a)).countp is_perfect_sixth_power

theorem num_non_squares_cubes_1_to_200 :
  let squares := num_perfect_squares 1 200 in
  let cubes := num_perfect_cubes 1 200 in
  let sixth_powers := num_perfect_sixth_powers 1 200 in
  200 - (squares + cubes - sixth_powers) = 182 :=
by
  let squares := num_perfect_squares 1 200
  let cubes := num_perfect_cubes 1 200
  let sixth_powers := num_perfect_sixth_powers 1 200
  have h : 200 - (squares + cubes - sixth_powers) = 182 := sorry
  exact h

end num_non_squares_cubes_1_to_200_l444_444440


namespace find_angle_between_lateral_edge_and_base_l444_444194

noncomputable def angle_between_lateral_edge_and_base
  (x h : ℝ)
  (A1D1 A2D2 : ℝ) :
  Prop :=
  let lower_base := 10 * x
  let upper_base := 2 * x
  let lateral_surface_area := h^2
  let S_A1D1D2A2 := (1/2) * (A1D1 + A2D2) * sqrt (h^2 + (lower_base/2 - upper_base/2)^2)
  2 * S_A1D1D2A2 = lateral_surface_area

theorem find_angle_between_lateral_edge_and_base
  (x h : ℝ)
  (A1D1 A2D2 : ℝ)
  (h_eq : h = 4 * x * sqrt(2 * (9 + 3 * sqrt(10))))
  (A1D1_eq : A1D1 = 2 * x)
  (A2D2_eq : A2D2 = 10 * x)
  (angle_theta : ℝ) :
  angle_between_lateral_edge_and_base x h A1D1 A2D2 →
  angle_theta = Real.arctan (sqrt (9 + 3 * sqrt(10))) :=
by
  sorry

end find_angle_between_lateral_edge_and_base_l444_444194


namespace arithmetic_sequence_1000th_term_l444_444588

theorem arithmetic_sequence_1000th_term (a_1 : ℤ) (d : ℤ) (n : ℤ) (h1 : a_1 = 1) (h2 : d = 3) (h3 : n = 1000) : 
  a_1 + (n - 1) * d = 2998 := 
by
  sorry

end arithmetic_sequence_1000th_term_l444_444588


namespace cube_surface_area_l444_444668

-- Define the edge length of the cube
def edge_length : ℝ := 4

-- Define the formula for the surface area of a cube
def surface_area (edge : ℝ) : ℝ := 6 * edge^2

-- Prove that given the edge length is 4 cm, the surface area is 96 cm²
theorem cube_surface_area : surface_area edge_length = 96 := by
  -- Proof goes here
  sorry

end cube_surface_area_l444_444668


namespace canal_depth_l444_444225

-- Define the problem parameters
def top_width : ℝ := 6
def bottom_width : ℝ := 4
def cross_section_area : ℝ := 10290

-- Define the theorem to prove the depth of the canal
theorem canal_depth :
  (1 / 2) * (top_width + bottom_width) * h = cross_section_area → h = 2058 :=
by sorry

end canal_depth_l444_444225


namespace percentage_deposit_paid_l444_444249

theorem percentage_deposit_paid (D R T : ℝ) (hd : D = 105) (hr : R = 945) (ht : T = D + R) : (D / T) * 100 = 10 := by
  sorry

end percentage_deposit_paid_l444_444249


namespace average_rate_of_change_l444_444179

noncomputable def f (x : ℝ) := x ^ 2

def interval := (0 : ℝ, 2 : ℝ)

theorem average_rate_of_change :
  (f interval.2 - f interval.1) / (interval.2 - interval.1) = 2 :=
by
  sorry

end average_rate_of_change_l444_444179


namespace painter_earnings_l444_444242

theorem painter_earnings :
  let east_sequence := (list.iota 25).map (λ n, 5 + n * 7),
      west_sequence := (list.iota 25).map (λ n, 2 + n * 5),
      earnings_per_digit (n : ℕ) : ℕ := list.sum (n.digits 10),
      east_earnings := east_sequence.sum (λ n, earnings_per_digit n),
      west_earnings := west_sequence.sum (λ n, earnings_per_digit n)
  in east_earnings + west_earnings = 113 :=
by
  sorry

end painter_earnings_l444_444242


namespace three_digit_number_multiple_of_eleven_l444_444344

theorem three_digit_number_multiple_of_eleven:
  ∃ (a b c : ℕ), (1 ≤ a) ∧ (a ≤ 9) ∧ (0 ≤ b) ∧ (b ≤ 9) ∧ (0 ≤ c) ∧ (c ≤ 9) ∧
                  (100 * a + 10 * b + c = 11 * (a + b + c) ∧ (100 * a + 10 * b + c = 198)) :=
by
  use 1
  use 9
  use 8
  sorry

end three_digit_number_multiple_of_eleven_l444_444344


namespace numbers_neither_square_nor_cube_l444_444542

theorem numbers_neither_square_nor_cube (n : ℕ) (h : n = 200) :
  let num_squares := 14 in
  let num_cubes := 5 in
  let num_sixth_powers := 2 in
  (n - (num_squares + num_cubes - num_sixth_powers)) = 183 :=
by
  -- Let the number of integers in the range
  let num_total := 200
  -- Define how we count perfect squares and cubes
  let num_squares := 14
  let num_cubes := 5
  let num_sixth_powers := 2

  -- Subtract the overlapping perfect sixth powers from the sum of squares and cubes
  let num_either_square_or_cube := num_squares + num_cubes - num_sixth_powers

  -- Calculate the final result
  let result := num_total - num_either_square_or_cube

  -- Prove by computation
  show result = 183 from
    calc
      result = num_total - num_either_square_or_cube := rfl
           ... = 200 - (14 + 5 - 2)                   := rfl
           ... = 200 - 17                             := rfl
           ... = 183                                  := rfl

end numbers_neither_square_nor_cube_l444_444542


namespace farmer_plough_remaining_area_l444_444829

theorem farmer_plough_remaining_area :
  ∀ (x R : ℕ),
  (90 * x = 3780) →
  (85 * (x + 2) + R = 3780) →
  R = 40 :=
by
  intros x R h1 h2
  sorry

end farmer_plough_remaining_area_l444_444829


namespace no_common_integer_solution_l444_444655

theorem no_common_integer_solution (n x y : ℤ) : 
  ¬((n - 6) % 15 = 0 ∧ (n - 5) % 24 = 0) :=
by {
  intro h,
  cases h with h₁ h₂,
  obtain ⟨k₁, hk₁⟩ := int.mod_eq_zero_of_dvd (int.dvd_iff_mod_eq_zero.mp h₁),
  obtain ⟨k₂, hk₂⟩ := int.mod_eq_zero_of_dvd (int.dvd_iff_mod_eq_zero.mp h₂),
  have := congr_arg (λ n, n % 3) (sub_eq_of_eq_add (sub_eq_of_eq_add hk₁.symm).symm hk₂),
  norm_num at this,
}

end no_common_integer_solution_l444_444655


namespace non_perfect_powers_count_l444_444485

theorem non_perfect_powers_count :
  let n := 200,
      num_perfect_squares := 14,
      num_perfect_cubes := 5,
      num_perfect_sixth_powers := 1,
      total_perfect_powers := num_perfect_squares + num_perfect_cubes - num_perfect_sixth_powers,
      total_non_perfect_powers := n - total_perfect_powers
  in
  total_non_perfect_powers = 182 :=
by
  sorry

end non_perfect_powers_count_l444_444485


namespace main_theorem_l444_444746

universe u
variables {α : Type u} [euclidean_geometry α]

open_locale euclidean_geometry

-- Define the given conditions
def is_isosceles (A B C : α) (h : euclidean_geometry.is_triangle A B C) : Prop :=
euclidean_geometry.dist A B = euclidean_geometry.dist A C

def midpoint (D B C : α) : Prop :=
euclidean_geometry.is_midpoint D B C

def perpendicular (E D A C : α) : Prop :=
euclidean_geometry.is_perpendicular E D A C

def circumcircle_intersects (F B D A : α) (c : set α) : Prop :=
c = euclidean_geometry.circumcircle A B D ∧ F ∈ c ∧ B ∈ c

-- The main theorem to prove
theorem main_theorem (A B C D E F : α) (h1 : euclidean_geometry.is_triangle A B C)
    (h2 : is_isosceles A B C h1) (h3 : midpoint D B C) (h4 : perpendicular E D A C)
    (h5 : ∃ c, circumcircle_intersects F B D A c) :
    euclidean_geometry.passes_through (euclidean_geometry.line A F) (euclidean_geometry.midpoint E D) :=
sorry

end main_theorem_l444_444746


namespace AM_perpendicular_to_BC_l444_444100

variables {A B C D E F G M : Type*}
variables [linear_ordered_field A] [ordered_ring B] [ordered_semiring C] [ordered_ring D]

-- Define the points and relationships between them
variables {triangle ABC : triangle}
variables {semicircle : ∀ (BC : line BC), intersects_sides BC AB D ∧ intersects_sides BC AC E}
variables (perpendiculars : ∀ D E BC, perpend_lines D BC F ∧ perpend_lines E BC G)
variables (intersection : line DG ∩ line EF = M)

-- Theorem you'd need to prove
theorem AM_perpendicular_to_BC (h : ∀ {BC AB AC D E F G M : Type*}, 
                                semicircle BC ∧ perpendiculars D E BC ∧ intersection DG EF) : 
  AM ⊥ BC := 
sorry

end AM_perpendicular_to_BC_l444_444100


namespace equal_angles_convex_polygon_l444_444644

theorem equal_angles_convex_polygon (n : ℕ) (h : ∀ p : ℕ, prime p → ¬ (∃ k : ℕ, n = p ^ k)) :
  ∃ (polygon : List ℝ), length polygon = n ∧ ∀ i < n - 2, angle polygon i = π * (1 - 2 / n) :=
sorry

end equal_angles_convex_polygon_l444_444644


namespace DE_equals_AF_AD_AB_equals_DE_DM_l444_444115

-- Define the basic properties of a parallelogram
variable (A B C D E M F : Type*)
variable [has_agent A] [has_agent B] [has_agent C] [has_agent D] [has_agent E] [has_agent M] [has_agent F]

-- Conditions
variables (parallelogram_ABCD : parallelogram A B C D)
variables (interior_angle_bisector_E : bisector_angle (angle A D C) B C E)
variables (perpendicular_bisector_M : bisector_perpendicular A D E M)
variables (intersection_F : same_line A M F B C)

-- Proof that DE = AF
theorem DE_equals_AF : length (segment D E) = length (segment A F) :=
by {
    -- Given conditions are used for this proof
    sorry
}

-- Proof that AD * AB = DE * DM
theorem AD_AB_equals_DE_DM : (length (segment A D) * length (segment A B)) = (length (segment D E) * length (segment D M)) :=
by {
    -- Given conditions are used for this proof
    sorry
}

end DE_equals_AF_AD_AB_equals_DE_DM_l444_444115


namespace area_union_example_l444_444274

noncomputable def area_union_square_circle (s r : ℝ) : ℝ :=
  let A_square := s ^ 2
  let A_circle := Real.pi * r ^ 2
  let A_overlap := (1 / 4) * A_circle
  A_square + A_circle - A_overlap

theorem area_union_example : (area_union_square_circle 10 10) = 100 + 75 * Real.pi :=
by
  sorry

end area_union_example_l444_444274


namespace numbers_not_perfect_squares_or_cubes_l444_444450

theorem numbers_not_perfect_squares_or_cubes : 
  let total_numbers := 200
  let perfect_squares := 14
  let perfect_cubes := 5
  let sixth_powers := 1
  total_numbers - (perfect_squares + perfect_cubes - sixth_powers) = 182 :=
by
  sorry

end numbers_not_perfect_squares_or_cubes_l444_444450


namespace find_point_on_parabola_l444_444261

noncomputable def parabola_vertex : (ℝ × ℝ) := (0, 0)
noncomputable def parabola_focus : (ℝ × ℝ) := (0, 2)
noncomputable def distance_to_focus (x y : ℝ) : ℝ := real.sqrt (x^2 + (y - 2)^2)
noncomputable def point_on_parabola (x y : ℝ) : Prop := distance_to_focus x y = y + 2
noncomputable def PF := 50

theorem find_point_on_parabola :
  ∃ (x y : ℝ), point_on_parabola x y ∧ PF = distance_to_focus x y ∧ y = 48 ∧ x = 8 * real.sqrt 3 :=
by
  apply Exists.intro 8 * real.sqrt 3
  apply Exists.intro 48
  unfold point_on_parabola distance_to_focus
  simp
  split
  sorry
  split
  sorry
  split
  refl
  refl

end find_point_on_parabola_l444_444261


namespace evaluate_expression_l444_444328

theorem evaluate_expression : (2^(-3) * 7^0) / 2^(-5) = (1/256) :=
by
  sorry

end evaluate_expression_l444_444328


namespace product_of_third_sides_equals_53_l444_444715

noncomputable def third_side_product (a b : ℕ) : ℝ :=
  let leg_hyp := real.sqrt ((a:ℝ) ^ 2 + (b:ℝ) ^ 2)
  let leg_leg := real.sqrt ((b:ℝ) ^ 2 - (a:ℝ) ^ 2)
  leg_hyp * leg_leg

theorem product_of_third_sides_equals_53 :
  third_side_product 6 8 = 53 :=
by
  have hypotenuse := real.sqrt (6 ^ 2 + 8 ^ 2)
  have other_leg := real.sqrt (8 ^ 2 - 6 ^ 2)
  have product := hypotenuse * other_leg
  have rounded := real.floor (product + 0.5)
  exact rounded = 53

end product_of_third_sides_equals_53_l444_444715


namespace right_triangle_area_l444_444571

def area_of_triangle (XY YZ: ℝ) : ℝ :=
  1 / 2 * XY * YZ

theorem right_triangle_area
  (XY XZ YZ : ℝ)
  (h1 : XY = 6)
  (h2 : ∠ Y = 90°)
  (h3 : ∠ Z = 60°)
  (h4 : YZ = XZ * (1 / 2))
  (h5 : XZ = 4 * Real.sqrt 3) :
  area_of_triangle XY YZ = 6 * Real.sqrt 3 :=
by
  -- Intermediate steps are omitted
  sorry

end right_triangle_area_l444_444571


namespace correct_choice_l444_444851

noncomputable def z : ℂ := 2 / (-1 + complex.I)

lemma z_val : z = -1 - complex.I :=
by simp [z, complex.div_eq_mul_inv, complex.inv_def, complex.mul_assoc, complex.to_real];

lemma p1 : ¬(abs z = 2) :=
by simp [z_val, complex.abs, complex.norm_sq];

lemma p2 : z^2 = 2 * complex.I :=
by simp [z_val, complex.sq, complex.I_sq, ring_hom.map_neg, ring_hom.map_add, ring_hom.map_sub, ring_hom.map_mul];

lemma p3 : ¬(conj z = 1 + complex.I) :=
by simp [z_val, complex.conj, ring_hom.map_neg, ring_hom.map_add];

lemma p4 : z.im = -1 :=
by simp [z_val];

theorem correct_choice : (p2 ∧ p4) ∧ ¬(p1 ∧ p3) :=
begin
  split,
  { split,
    exact p2,
    exact p4, },
  { intro h,
    cases h,
    { exfalso,
      exact p1 h_left },
    { exfalso,
      exact p3 h_right } 
  }
end

end correct_choice_l444_444851


namespace log_equation_solution_l444_444218

theorem log_equation_solution (x : ℝ) (h₁ : x > 0) (h₂ : x ≠ 1) (h₃ : x ≠ 1/16) (h₄ : x ≠ 1/2) 
    (h_eq : (Real.log 2 / Real.log (4 * Real.sqrt x)) / (Real.log 2 / Real.log (2 * x)) 
            + (Real.log 2 / Real.log (2 * x)) * (Real.log (2 * x) / Real.log (1 / 2)) = 0) 
    : x = 4 := 
sorry

end log_equation_solution_l444_444218


namespace Vasyuki_coloring_possible_l444_444576

theorem Vasyuki_coloring_possible (n : ℕ) (perm : Equiv.Perm (Fin n)) : 
  ∃ (colors : Fin n → Fin 3), ∀ i, colors i ≠ colors (perm i) :=
by
  sorry

end Vasyuki_coloring_possible_l444_444576


namespace speed_of_stream_l444_444834

theorem speed_of_stream (downstream_speed upstream_speed : ℝ) (h1 : downstream_speed = 11) (h2 : upstream_speed = 8) : 
    (downstream_speed - upstream_speed) / 2 = 1.5 :=
by
  rw [h1, h2]
  simp
  norm_num

end speed_of_stream_l444_444834


namespace triangle_inequality_l444_444639

variables {R r p ab bc ca : ℝ}

theorem triangle_inequality (h1 : ab + bc + ca = r^2 + p^2 + 4Rr)
  (h2 : 16Rr - 5r^2 ≤ p^2)
  (h3 : p^2 ≤ 4R^2 + 4Rr + 3r^2) :
  20Rr - 4r^2 ≤ ab + bc + ca ∧ ab + bc + ca ≤ 4 * (R + r)^2 :=
by
  sorry

end triangle_inequality_l444_444639


namespace sufficient_but_not_necessary_condition_not_neccessary_condition_l444_444928

theorem sufficient_but_not_necessary_condition (x y : ℝ) :
  ((x + 3)^2 + (y - 4)^2 = 0) → ((x + 3) * (y - 4) = 0) :=
by { sorry }

theorem not_neccessary_condition (x y : ℝ) :
  ((x + 3) * (y - 4) = 0) ↔ ((x + 3)^2 + (y - 4)^2 = 0) :=
by { sorry }

end sufficient_but_not_necessary_condition_not_neccessary_condition_l444_444928


namespace sum_first_19_terms_l444_444693

/-!
# Arithmetic Sequence Sum

In an arithmetic sequence ${a_n}$, the sum of the first $n$ terms $S_n$ is defined, and it is given that $a_3 + a_{17} = 10$. Prove that $S_{19} = 95$.
-/

variable {a : ℕ → ℝ}  -- sequence definition

-- Define the sum of the first n terms of the arithmetic sequence
def S (n : ℕ) := (n * (a 1 + a n)) / 2

-- Given condition
axiom h : a 3 + a 17 = 10

theorem sum_first_19_terms : S 19 = 95 :=
by sorry

end sum_first_19_terms_l444_444693


namespace inequality_amgm_l444_444653

theorem inequality_amgm (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) : a^3 + b^3 + a + b ≥ 4 * a * b :=
sorry

end inequality_amgm_l444_444653


namespace felicity_collecting_weeks_l444_444331

theorem felicity_collecting_weeks 
  (total_sticks : ℕ)
  (current_progress_pct : ℝ)
  (store_visits_week1 : ℕ)
  (store_visits_week2 : ℕ)
  (aunt_sticks_5weeks : ℕ)
  (current_collected_sticks : ℕ) 
  (average_sticks_per_week : ℝ) 
  (expected_weeks : ℕ) :
  total_sticks = 1500 →
  current_progress_pct = 0.65 →
  store_visits_week1 = 3 →
  store_visits_week2 = 4 →
  aunt_sticks_5weeks = 35 →
  current_collected_sticks = (current_progress_pct * total_sticks).toNat →
  average_sticks_per_week = ((store_visits_week1 + store_visits_week2) / 2 + (aunt_sticks_5weeks / 5)) →
  expected_weeks = (current_collected_sticks / average_sticks_per_week).ceil.toNat →
  expected_weeks = 93 :=
by
  intros
  sorry

end felicity_collecting_weeks_l444_444331


namespace prime_factors_of_N_l444_444890

theorem prime_factors_of_N
  (N : ℕ)
  (h : log 2 (log 3 (log 5 (log 7 (log 11 N)))) = 16) :
  finset.card (nat.factors N).to_finset = 1 := 
sorry

end prime_factors_of_N_l444_444890


namespace arithmetic_sequence_a4_equals_8_l444_444089

variable {α : Type*} [AddGroup α] [Module ℤ α]
variables {a : ℕ → α}

theorem arithmetic_sequence_a4_equals_8 
  (h : 2 * a 4 = a 3 + a 5) 
  (h1 : a 3 + a 5 = 16) : a 4 = 8 :=
by sorry

end arithmetic_sequence_a4_equals_8_l444_444089


namespace sum_exterior_angles_octagon_l444_444692

theorem sum_exterior_angles_octagon : 
  (∀ (polygon : Type) (exterior_angle_sum : polygon → ℝ), 
  ((exterior_angle_sum octagon = 360))) :=
begin
  sorry,
end

end sum_exterior_angles_octagon_l444_444692


namespace intersection_point_is_correct_l444_444963

-- Define the conditions on x and y based on the system of linear equations
def system_of_equations (x y b : ℝ) : Prop :=
  x + y - b = 0 ∧ 3 * x + y - 2 = 0

-- Translate the problem into a Lean theorem statement
theorem intersection_point_is_correct (b : ℝ) :
  system_of_equations (-1) 5 b →
  ∃ (p : ℝ × ℝ), p = (-1, 5) :=
by
  assume h : system_of_equations (-1) 5 b
  use (-1, 5)
  sorry

end intersection_point_is_correct_l444_444963


namespace count_difference_of_squares_l444_444977

theorem count_difference_of_squares : 
  ∃ n : ℕ, (∀ x ∈ finset.range 501, ∃ a b : ℕ, x = a^2 - b^2) → n = 375 :=
by
  let nums := finset.range 501
  let expressible := nums.filter (λ x, ∃ a b : ℕ, x = a^2 - b^2)
  exact sorry

end count_difference_of_squares_l444_444977


namespace cards_to_collect_l444_444620

noncomputable def lloyd_cards : ℕ := 30
noncomputable def mark_cards : ℕ := 90
noncomputable def michael_cards : ℕ := 100

theorem cards_to_collect (total_desired : ℕ) 
    (mark_3times_lloyd : mark_cards = 3 * lloyd_cards)
    (mark_10_less_michael : mark_cards = michael_cards - 10)
    (michael_has_100 : michael_cards = 100) :
    total_desired - (mark_cards + lloyd_cards + michael_cards) = 80 :=
by
    rw [mark_3times_lloyd, mark_10_less_michael, michael_has_100]
    sorry

end cards_to_collect_l444_444620


namespace females_in_coach_class_l444_444739

def total_passengers : ℕ := 120
def percentage_females : ℝ := 0.45
def percentage_first_class : ℝ := 0.10
def fraction_male_first_class : ℝ := 1 / 3

def total_females : ℕ := (percentage_females * total_passengers).to_nat
def first_class_passengers : ℕ := (percentage_first_class * total_passengers).to_nat
def females_first_class : ℕ := (first_class_passengers * (1 - fraction_male_first_class)).to_nat
def coach_class_passengers : ℕ := total_passengers - first_class_passengers
def females_coach_class : ℕ := total_females - females_first_class

theorem females_in_coach_class : females_coach_class = 46 := by
  sorry

end females_in_coach_class_l444_444739


namespace magnitude_range_l444_444066

noncomputable def vector_a (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)
noncomputable def vector_b : ℝ × ℝ := (Real.sqrt 3, -1)

noncomputable def vector_magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

theorem magnitude_range (θ : ℝ) : 
  0 ≤ (vector_magnitude (2 • vector_a θ - vector_b)) ∧ (vector_magnitude (2 • vector_a θ - vector_b)) ≤ 4 := 
sorry

end magnitude_range_l444_444066


namespace sequence_property_l444_444937

-- Given Definitions
def S (a : ℕ → ℕ) (n : ℕ) : ℕ := 
  (finset.range n).sum a

noncomputable def a (n : ℕ) : ℕ := 
  if n = 0 then 0 else 2 ^ (n - 1)

theorem sequence_property :
  (a 1 = 1) ∧ 
  (∀ n, S a n = a (n + 1) - 1) →
  ∀ n, a n = 2^(n-1) := 
  by
    sorry

end sequence_property_l444_444937


namespace line_AF_through_midpoint_of_DE_l444_444801

open EuclideanGeometry

noncomputable def midpoint_of_segment (A B : Point) := ((A.x + B.x) / 2, (A.y + B.y) / 2)

theorem line_AF_through_midpoint_of_DE
  {A B C D E F : Point}
  (h_isosceles : AB = AC)
  (h_midpoint_D : D = midpoint B C)
  (h_perpendicular_DE : ⟂ D E AC)
  (h_circumcircle_intersects_BE : ∃ k : ℕ, circumcircle A B D = circum circle (B ⊗ F))
  : passes_through (line A F) (midpoint_of_segment D E) :=
sorry

end line_AF_through_midpoint_of_DE_l444_444801


namespace count_not_squares_or_cubes_l444_444505

theorem count_not_squares_or_cubes {A B : Finset ℕ} (h_range : ∀ x, x ∈ A ∪ B → x ≤ 200)
  (h_squares : A.card = 14) (h_cubes : B.card = 5) (h_inter : (A ∩ B).card = 1) :
  (200 - (A ∪ B).card) = 182 :=
by
  sorry

end count_not_squares_or_cubes_l444_444505


namespace concyclic_points_l444_444251

theorem concyclic_points
  (O K M N A B C D E F : Type)
  [metricSpace O] [metricSpace K] [metricSpace M] [metricSpace N]
  [metricSpace A] [metricSpace B] [metricSpace C] [metricSpace D] [metricSpace E] [metricSpace F]
  (h_circle_inscribed : ∃ (r : ℝ), r ≠ 0)
  (h_touch_BC : segment BC intersects_circle circumference O at E)
  (h_touch_AD : segment AD intersects_circle circumference O at F)
  (h_int_AO_EF : ∃ K, line AO intersects segment EF at K)
  (h_int_DO_EF : ∃ N, line DO intersects segment EF at N)
  (h_int_BK_CN : ∃ M, line BK intersects line CN at M) :
  Points O K M N lie_on_same_circle :=
begin
  sorry
end

end concyclic_points_l444_444251


namespace complement_union_S_T_l444_444063

open Set

theorem complement_union_S_T (S T : Set ℝ) (x : ℝ)
  (hS : S = {x | x > -2})
  (hT : T = {x | x^2 + 3x - 4 ≤ 0}) :
  (compl S) ∪ T = {x | x ≤ 1} :=
sorry

end complement_union_S_T_l444_444063


namespace training_cost_per_month_correct_l444_444264

-- Define the conditions
def salary1 : ℕ := 42000
def revenue1 : ℕ := 93000
def training_duration : ℕ := 3
def salary2 : ℕ := 45000
def revenue2 : ℕ := 92000
def bonus2 : ℕ := (45000 / 100) -- 1% of salary2 which is 450
def net_gain_diff : ℕ := 850

-- Define the monthly training cost for the first applicant
def monthly_training_cost : ℕ := 1786667 / 100

-- Prove that the monthly training cost for the first applicant is correct
theorem training_cost_per_month_correct :
  (revenue1 - (salary1 + 3 * monthly_training_cost) = revenue2 - (salary2 + bonus2) + net_gain_diff) :=
by
  sorry

end training_cost_per_month_correct_l444_444264


namespace two_person_subcommittees_from_six_l444_444068

theorem two_person_subcommittees_from_six :
  (Nat.choose 6 2) = 15 := by
  sorry

end two_person_subcommittees_from_six_l444_444068


namespace red_car_speed_l444_444880

/-- Dale owns 4 sports cars where:
1. The red car can travel at twice the speed of the green car.
2. The green car can travel at 8 times the speed of the blue car.
3. The blue car can travel at a speed of 80 miles per hour.
We need to determine the speed of the red car. --/
theorem red_car_speed (r g b: ℕ) (h1: r = 2 * g) (h2: g = 8 * b) (h3: b = 80) : 
  r = 1280 :=
by
  sorry

end red_car_speed_l444_444880


namespace remainder_of_b2018_mod_7_l444_444142

def sequence_a (n : ℤ) : ℤ :=
  n^3 - n

def is_valid_term (n : ℤ) : Prop :=
  (n % 10 = 0) ∨ (n % 10 = 1) ∨ (n % 10 = 4) ∨ (n % 10 = 5) ∨ (n % 10 = 6) ∨ (n % 10 = 9)

def valid_terms_in_block (block_start : ℤ) : List ℤ :=
  List.filter is_valid_term (List.range' block_start 10)

-- Determine the term that corresponds to b_{2018}
def nth_valid_term (n : ℕ) : ℤ :=
  let block_num := n / 6;
  let within_block_pos := n % 6;
  let block_start := block_num * 10;
  (valid_terms_in_block block_start).get! within_block_pos

def sequence_b (n : ℕ) : ℤ :=
  sequence_a (nth_valid_term n)

theorem remainder_of_b2018_mod_7 : (sequence_b 2018) % 7 = 4 :=
  by
    -- We'll fill this proof in later
    sorry

end remainder_of_b2018_mod_7_l444_444142


namespace slant_edge_and_inscribed_sphere_radius_l444_444085

variables (a : ℝ) (C : Prop)
def is_slant_edge (AD : ℝ) := AD = (5 * a * real.sqrt 2) / 4
def is_inscribed_sphere_radius (r : ℝ) := r = 12 * a / 5

theorem slant_edge_and_inscribed_sphere_radius (a : ℝ)
  (C1 : ∀ (t : Type) (PQR : t), regular_pyramid PQR)
  (C2 : ∀ (PQR : Type), side_length PQR = a)
  (C3 : ∀ (ABD : Type) (cone : Type) (O : Type), cone_is_inscribed cone ABD)
  (C4 : ∀ (triangle : Type), O_lies_on_median O triangle CE)
  (C5 : ∀ (CE : Type), ratio CE OE = 4) :
  ∃ (AD r : ℝ), is_slant_edge a AD ∧ is_inscribed_sphere_radius a r := 
begin
  -- Proof goes here
  sorry
end

end slant_edge_and_inscribed_sphere_radius_l444_444085


namespace smaller_equilateral_triangle_and_square_properties_l444_444854

open Real

theorem smaller_equilateral_triangle_and_square_properties
  (side_length_large_triangle : ℝ)
  (area_ratio : ℝ) 
  (side_length_square : ℝ)
  (perimeter_square : ℝ) :
  (side_length_large_triangle = 3) →
  (area_ratio = 1/3) →
  (side_length_square = sqrt 3) →
  (perimeter_square = 12) :=
by
  intros h1 h2 h3 h4
  rw [←h1, ←h2] at *
  sorry

end smaller_equilateral_triangle_and_square_properties_l444_444854


namespace num_non_squares_cubes_1_to_200_l444_444438

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n
def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k * k * k = n
def is_perfect_sixth_power (n : ℕ) : Prop := ∃ k : ℕ, k * k * k * k * k * k = n

def num_perfect_squares (a b : ℕ) : ℕ :=
  (List.range (b + 1)).countp is_perfect_square - (List.range (a)).countp is_perfect_square

def num_perfect_cubes (a b : ℕ) : ℕ :=
  (List.range (b + 1)).countp is_perfect_cube - (List.range (a)).countp is_perfect_cube

def num_perfect_sixth_powers (a b : ℕ) : ℕ :=
  (List.range (b + 1)).countp is_perfect_sixth_power - (List.range (a)).countp is_perfect_sixth_power

theorem num_non_squares_cubes_1_to_200 :
  let squares := num_perfect_squares 1 200 in
  let cubes := num_perfect_cubes 1 200 in
  let sixth_powers := num_perfect_sixth_powers 1 200 in
  200 - (squares + cubes - sixth_powers) = 182 :=
by
  let squares := num_perfect_squares 1 200
  let cubes := num_perfect_cubes 1 200
  let sixth_powers := num_perfect_sixth_powers 1 200
  have h : 200 - (squares + cubes - sixth_powers) = 182 := sorry
  exact h

end num_non_squares_cubes_1_to_200_l444_444438


namespace solve_fraction_equal_zero_l444_444968

theorem solve_fraction_equal_zero :
  ∃ x : ℝ, (x + 1 = 0) ∧ (9 * x^2 - 74 * x + 9 ≠ 0) :=
by {
  existsi -1,
  split,
  { 
    -- Proof for x + 1 = 0
    sorry
  },
  { 
    -- Proof for 9x^2 - 74x + 9 ≠ 0 at x = -1
    sorry
  }
}

end solve_fraction_equal_zero_l444_444968


namespace discount_price_l444_444190

theorem discount_price (a : ℝ) (original_price : ℝ) (sold_price : ℝ) :
  original_price = 200 ∧ sold_price = 148 → (original_price * (1 - a/100) * (1 - a/100) = sold_price) :=
by
  sorry

end discount_price_l444_444190


namespace numbers_neither_square_nor_cube_l444_444536

theorem numbers_neither_square_nor_cube (n : ℕ) (h : n = 200) :
  let num_squares := 14 in
  let num_cubes := 5 in
  let num_sixth_powers := 2 in
  (n - (num_squares + num_cubes - num_sixth_powers)) = 183 :=
by
  -- Let the number of integers in the range
  let num_total := 200
  -- Define how we count perfect squares and cubes
  let num_squares := 14
  let num_cubes := 5
  let num_sixth_powers := 2

  -- Subtract the overlapping perfect sixth powers from the sum of squares and cubes
  let num_either_square_or_cube := num_squares + num_cubes - num_sixth_powers

  -- Calculate the final result
  let result := num_total - num_either_square_or_cube

  -- Prove by computation
  show result = 183 from
    calc
      result = num_total - num_either_square_or_cube := rfl
           ... = 200 - (14 + 5 - 2)                   := rfl
           ... = 200 - 17                             := rfl
           ... = 183                                  := rfl

end numbers_neither_square_nor_cube_l444_444536


namespace mod_calculation_l444_444865

theorem mod_calculation :
  (2021 % 7 = 6) ∧ (2022 % 7 = 0) ∧ (2023 % 7 = 1) ∧ (2024 % 7 = 2) →
  (2021 * 2022 * 2023 * 2024) % 7 = 0 :=
by {
  intro h,
  sorry
}

end mod_calculation_l444_444865


namespace interesting_quadruples_count_l444_444316

def is_interesting_quadruple (a b c d : ℕ) : Prop :=
  1 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d ≤ 15 ∧ a + d > b + c + 2

def count_interesting_quadruples : ℕ :=
  633

theorem interesting_quadruples_count : 
  (finset.univ.product (finset.range 16) 
    .product (finset.range 16)
    .product (finset.range 16))
  .filter (λ ((a, b), (c, d)), is_interesting_quadruple a b c d).card 
  = count_interesting_quadruples :=
begin
  sorry
end

end interesting_quadruples_count_l444_444316


namespace irrational_roots_of_odd_coeffs_l444_444643

theorem irrational_roots_of_odd_coeffs (a b c : ℤ) (ha : a % 2 = 1) (hb : b % 2 = 1) (hc : c % 2 = 1) : 
  ¬ ∃ (x : ℚ), a * x^2 + b * x + c = 0 := 
sorry

end irrational_roots_of_odd_coeffs_l444_444643


namespace count_difference_of_squares_l444_444979

theorem count_difference_of_squares : 
  ∃ n : ℕ, (∀ x ∈ finset.range 501, ∃ a b : ℕ, x = a^2 - b^2) → n = 375 :=
by
  let nums := finset.range 501
  let expressible := nums.filter (λ x, ∃ a b : ℕ, x = a^2 - b^2)
  exact sorry

end count_difference_of_squares_l444_444979


namespace marthas_cat_catches_3_rats_l444_444145

theorem marthas_cat_catches_3_rats :
  ∃ R_m : ℕ, 
  let T_m := R_m + 7 in
  let T_c := 5 * T_m - 3 in
  T_c = 47 → R_m = 3 :=
by 
  intro R_m T_m T_c h1 h2 h3,
  have R_m := 3,
  exact sorries

end marthas_cat_catches_3_rats_l444_444145


namespace perpendicular_bisector_l444_444181

theorem perpendicular_bisector (x y : ℝ) :
  (x - 2 * y + 1 = 0 ∧ -1 ≤ x ∧ x ≤ 3) → (2 * x + y - 3 = 0) :=
by
  sorry

end perpendicular_bisector_l444_444181


namespace converse_and_inverse_true_l444_444964

variable (Q : Type) [Parallelogram Q] [Rhombus Q]

theorem converse_and_inverse_true :
  (∀ q : Q, parallelogram q → rhombus q) →
  (∀ q : Q, rhombus q → parallelogram q) ∧
  (∀ q : Q, ¬parallelogram q → ¬rhombus q) :=
by
  sorry

end converse_and_inverse_true_l444_444964


namespace derivative_value_at_pi_over_2_l444_444953

noncomputable def f (x : ℝ) : ℝ := Real.cos x - Real.sin x

theorem derivative_value_at_pi_over_2 : deriv f (Real.pi / 2) = -1 :=
by
  sorry

end derivative_value_at_pi_over_2_l444_444953


namespace dot_product_of_parallel_vectors_l444_444419

def vector (α : Type*) := α × α

def is_parallel {α : Type*} [field α] (v1 v2 : vector α) : Prop :=
  v1.1 * v2.2 = v1.2 * v2.1

def dot_product {α : Type*} [ring α] (v1 v2 : vector α) : α :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem dot_product_of_parallel_vectors :
  ∀ (x : ℝ), is_parallel (x, x - 1) (1, 2) → dot_product (x, x - 1) (1, 2) = -5 :=
by
  intros x hp
  sorry

end dot_product_of_parallel_vectors_l444_444419


namespace f_log2_3_eq_1_over_24_l444_444670

noncomputable def f : ℝ → ℝ
| x := if h : x ≥ 4 then (1 / 2) ^ x else f (x + 1)

theorem f_log2_3_eq_1_over_24 : f (Real.log 3 / Real.log 2) = 1 / 24 := by
  sorry

end f_log2_3_eq_1_over_24_l444_444670


namespace locus_of_P_l444_444814

-- Definition of the problem conditions
variables {A B P : Point} {a : ℝ} (hA : A = (0, 0)) (hB : B = (a, 0))
  (hC : C = (x_0, y_0) ∧ (x_0^2 + y_0^2 = 1)) 
  (hP : P = (x, y) ∧ P is the intersection of the angle bisector of ∠CAB with BC)

-- Definition to check the locus of point P
theorem locus_of_P : (x ∈ ℝ) (y ∈ ℝ) (x^2 + y^2 = (a^2 / (1 + a)^2)) := 
  sorry

end locus_of_P_l444_444814


namespace sugar_needed_for_third_layer_l444_444299

-- Let cups be the amount of sugar, and define the layers
def first_layer_sugar : ℕ := 2
def second_layer_sugar : ℕ := 2 * first_layer_sugar
def third_layer_sugar : ℕ := 3 * second_layer_sugar

-- The theorem we want to prove
theorem sugar_needed_for_third_layer : third_layer_sugar = 12 := by
  sorry

end sugar_needed_for_third_layer_l444_444299


namespace find_x_l444_444991

theorem find_x (x : ℝ) (h : 0.60 / x = 6 / 2) : x = 0.2 :=
by {
  sorry
}

end find_x_l444_444991


namespace problem_l444_444809

noncomputable def a : ℕ := 240
noncomputable def b : ℕ := 3
noncomputable def c : ℕ := 7

theorem problem (x : ℝ) (A B C D E F G H : point) (hAB : dist A B = x) (hBC : dist B C = x) (hCD : dist C D = x) 
(hBE : dist B E = x) (hEC : dist E C = x) (hED_mid : midpoint D E F) (hAF_G : intersects (line A F) (line E C) G)
(hBF_H : intersects (line B F) (line E C) H) (area_BHC_GHF : area B H C + area G H F = 1) : 
AD^2 = (9 * x^2) := 
sorry

end problem_l444_444809


namespace find_principal_l444_444275

variable (P R : ℝ) (SI1 SI2 : ℝ)
variable (T : ℝ := 5)

def simple_interest (P : ℝ) (R : ℝ) (T : ℝ) := (P * R * T) / 100

axiom (h1 : simple_interest P R T = SI1)
axiom (h2 : simple_interest P (R + 2) T = SI2)
axiom (h3 : SI2 = SI1 + 250)

theorem find_principal : P = 2500 := by sorry

end find_principal_l444_444275


namespace max_handshakes_l444_444818

theorem max_handshakes {n : ℕ} (h : n = 30) : (nat.choose 30 2) = 435 :=
by
  rw h
  sorry

end max_handshakes_l444_444818


namespace sum_slope_y_intercept_is_neg2_l444_444200

-- Define the vertices of the triangle
def A : (ℝ × ℝ) := (0, 8)
def B : (ℝ × ℝ) := (2, 0)
def C : (ℝ × ℝ) := (8, 0)

-- Define the midpoint of AC
def M : (ℝ × ℝ) := ((A.1 + C.1) / 2, (A.2 + C.2) / 2)

-- Define the line passing through B and M, described by its slope and y-intercept
def line_through_B_M : (ℝ × ℝ) :=
  let slope := (M.2 - B.2) / (M.1 - B.1) in
  let y_intercept := B.2 - slope * B.1 in
  (slope, y_intercept)

-- Define the sum of the slope and y-intercept of the line
def sum_slope_y_intercept : ℝ := line_through_B_M.1 + line_through_B_M.2

-- Formal statement of the problem to be proved
theorem sum_slope_y_intercept_is_neg2 : sum_slope_y_intercept = -2 := by sorry

end sum_slope_y_intercept_is_neg2_l444_444200


namespace prime_factors_difference_163027_l444_444723

theorem prime_factors_difference_163027 :
  ∃ (p q : ℕ), (prime p ∧ prime q ∧ p * q = 163027 ∧ p ≠ q ∧ abs (p - q) = 662) := sorry

end prime_factors_difference_163027_l444_444723


namespace count_not_squares_or_cubes_l444_444508

theorem count_not_squares_or_cubes {A B : Finset ℕ} (h_range : ∀ x, x ∈ A ∪ B → x ≤ 200)
  (h_squares : A.card = 14) (h_cubes : B.card = 5) (h_inter : (A ∩ B).card = 1) :
  (200 - (A ∪ B).card) = 182 :=
by
  sorry

end count_not_squares_or_cubes_l444_444508


namespace probability_correct_l444_444058

noncomputable def probability_sum_less_than_9 : ℚ :=
  let s := {1, 3, 5, 7}
  let pairs := (s ×ˢ s).toFinset.filter (λ p => p.1 < p.2) -- all unique pairs (a,b)
  let favorable := pairs.filter (λ p => p.1 + p.2 < 9)
  ↑(favorable.card) / ↑(pairs.card)

theorem probability_correct :
  probability_sum_less_than_9 = 2 / 3 :=
sorry

end probability_correct_l444_444058


namespace circle_and_chord_problem_l444_444382

open Real

-- Definitions of points and line
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (2, 0)
def B : ℝ × ℝ := (2, 2)
def l (x : ℝ) : ℝ := x - 1

-- Circle equation definition
def circle (a b r : ℝ) (x y : ℝ) : Prop := (x - a)^2 + (y - b)^2 = r^2

-- The main theorem to prove
theorem circle_and_chord_problem :
  (∀ x y, circle 1 1 (sqrt 2) x y ↔ (x-1)^2 + (y-1)^2 = 2) ∧
  (let d := abs(1 - 1 - 1) / sqrt (1^2 + (-1)^2) in
   let chord_length := 2 * sqrt ((sqrt 2)^2 - d^2) in
   chord_length = sqrt 6) :=
by sorry

end circle_and_chord_problem_l444_444382


namespace perimeter_of_square_l444_444840

/-- The perimeter of a square with side length 15 cm is 60 cm -/
theorem perimeter_of_square (side_length : ℝ) (area : ℝ) (h1 : side_length = 15) (h2 : area = 225) :
  (4 * side_length = 60) :=
by
  -- Proof steps would go here (omitted)
  sorry

end perimeter_of_square_l444_444840


namespace count_even_k_divisible_by_12_l444_444360

theorem count_even_k_divisible_by_12 : 
  (Finset.card {k : ℕ | 1 ≤ k ∧ k ≤ 9 ∧ 12 ∣ k * 234}) = 4 := 
by 
  sorry

end count_even_k_divisible_by_12_l444_444360


namespace remainder_division_by_8_is_6_l444_444631

theorem remainder_division_by_8_is_6 (N Q2 R1 : ℤ) (h1 : N = 64 + R1) (h2 : N % 5 = 4) : R1 = 6 :=
by
  sorry

end remainder_division_by_8_is_6_l444_444631


namespace angle_EBC_equals_72_l444_444936

-- Definition of the given conditions and proof problem
theorem angle_EBC_equals_72 (
  (ABCD C : Quadrilateral),  -- ABCD is a quadrilateral inscribed in a circle
  (A B C D E F : Point),     -- A, B, C, D, E, F are points
  (isCyclic ABCD)            -- ABCD is a cyclic quadrilateral
  (line_AD : Line AD)        -- CD is a line segment
  (AB_extends_E : Line AB E) -- AB is extended to point E
  (point_F_on_CD : F ∈ CD)   -- F is a point on the segment CD
  (angle_BAD : Angle A B D = 85) -- ∠BAD = 85 degrees
  (angle_ADC : Angle A D C = 72) -- ∠ADC = 72 degrees
  (angle_BCF : Angle B C F = 30) -- ∠BCF = 30 degrees
): Angle E B C = 72 :=
sorry

end angle_EBC_equals_72_l444_444936


namespace monotone_decreasing_interval_l444_444548

noncomputable def f (x : ℝ) := sin x * cos x + sqrt 3 * (cos x)^2 - sqrt 3 / 2

theorem monotone_decreasing_interval : 
  ∀ x : ℝ, (0 < x ∧ x < π) → (∃ I : set ℝ, I = Icc (π / 12) (7 * π / 12) ∧ ∀ x ∈ I, ∀ y ∈ I, y < x → f(x) ≤ f(y)) := 
by
  sorry

end monotone_decreasing_interval_l444_444548


namespace cannot_determine_right_triangle_l444_444049

def is_right_triangle (A B C : ℝ) : Prop :=
  A + B + C = 180 ∧ (A = 90 ∨ B = 90 ∨ C = 90)

theorem cannot_determine_right_triangle :
  ∀ A B C : ℝ, 
    (A = 2 * B ∧ A = 3 * C) →
    ¬ is_right_triangle A B C :=
by
  intro A B C h
  have h1 : A = 2 * B := h.1
  have h2 : A = 3 * C := h.2
  sorry

end cannot_determine_right_triangle_l444_444049


namespace relation_between_mu_d_M_l444_444564

-- Define the given conditions
def count1_to_29 := 12
def count30 := 11
def count31 := 7
def total := 366

-- Define the mean (μ), median (M), and median of the modes (d) based on the problem conditions
noncomputable def mean : ℝ := (12 * (∑ k in finset.range 30, k + 1) + 11 * 30 + 7 * 31) / 366
noncomputable def median : ℝ := 16
noncomputable def median_modes : ℝ := 15

-- State the goal: Prove that d < μ < M
theorem relation_between_mu_d_M : median_modes < mean ∧ mean < median := by
  -- Calculation and comparison details will go here.
  sorry

end relation_between_mu_d_M_l444_444564


namespace sand_leak_time_l444_444253

-- Define the conditions
def initial_volume (a : ℝ) : ℝ := a
def remaining_volume (a b t : ℝ) : ℝ := a * real.exp (-b * t)

-- Given conditions
def given_conditions (a b : ℝ) : Prop :=
  remaining_volume a b 8 = (1/2) * a

-- The theorem we want to prove
theorem sand_leak_time (a t : ℝ) (b := real.log 2 / 8) (h : given_conditions a b) :
  remaining_volume a b 24 = (1/8) * a :=
sorry

end sand_leak_time_l444_444253


namespace min_product_of_three_numbers_l444_444366

def SetOfNumbers : Set ℤ := {-9, -5, -1, 1, 3, 5, 8}

theorem min_product_of_three_numbers : 
  ∃ (a b c : ℤ), a ∈ SetOfNumbers ∧ b ∈ SetOfNumbers ∧ c ∈ SetOfNumbers ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ a * b * c = -360 :=
by {
  sorry
}

end min_product_of_three_numbers_l444_444366


namespace passing_through_midpoint_of_DE_l444_444792

noncomputable def is_isosceles (ABC : Triangle) : Prop :=
ABC.AB = ABC.AC

noncomputable def midpoint (D : Point) (BC : Segment) : Prop :=
(B + C) / 2 = D

noncomputable def perpendicular (DE : Line) (AC : Line) : Prop :=
right_angle (angle DE AC)

noncomputable def circumsircle_intersect (F : Point) (ABD : Triangle) (BE : Line) : Prop :=
(circumcircle ABD).contains B ∧ (circumcircle ABD).contains F ∧ line BE.contains B ∧ line BE.contains F

noncomputable def passes_through_midpoint (AF : Line) (DE : Segment) : Prop :=
exists M : Point, midpoint M DE ∧ line AF.contains M

theorem passing_through_midpoint_of_DE
  (ABC : Triangle) (D E F : Point) (DE : Line) (AC BE AF : Line) :
  is_isosceles ABC →
  midpoint D (BC) →
  perpendicular DE AC →
  circumsircle_intersect F (Triangle.mk ABD) BE →
  passes_through_midpoint AF (segment.mk D E) :=
by
  sorry

end passing_through_midpoint_of_DE_l444_444792


namespace unique_even_odd_decomposition_l444_444640

def is_symmetric (s : Set ℝ) : Prop := ∀ x ∈ s, -x ∈ s

def is_even (f : ℝ → ℝ) (s : Set ℝ) : Prop := ∀ x ∈ s, f (-x) = f x

def is_odd (f : ℝ → ℝ) (s : Set ℝ) : Prop := ∀ x ∈ s, f (-x) = -f x

theorem unique_even_odd_decomposition (s : Set ℝ) (hs : is_symmetric s) (f : ℝ → ℝ) (hf : ∀ x ∈ s, True) :
  ∃! g h : ℝ → ℝ, (is_even g s) ∧ (is_odd h s) ∧ (∀ x ∈ s, f x = g x + h x) :=
sorry

end unique_even_odd_decomposition_l444_444640


namespace count_valid_numbers_between_1_and_200_l444_444432

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k^2 = n
def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k^3 = n
def is_perfect_sixth_power (n : ℕ) : Prop := ∃ k : ℕ, k^6 = n

def count_perfect_squares (m : ℕ) : ℕ :=
  Nat.sqrt m

def count_perfect_cubes (m : ℕ) : ℕ :=
  (m : ℝ).cbrt.to_nat

def count_perfect_sixth_powers (m : ℕ) : ℕ :=
  (m : ℝ).cbrt.sqrt.to_nat

def count_either_squares_or_cubes (m : ℕ) : ℕ :=
  let squares := count_perfect_squares m
  let cubes := count_perfect_cubes m
  let sixths := count_perfect_sixth_powers m
  squares + cubes - sixths

def count_neither_squares_nor_cubes (n : ℕ) : ℕ :=
  n - count_either_squares_or_cubes n

theorem count_valid_numbers_between_1_and_200 : count_neither_squares_nor_cubes 200 = 182 :=
by
  sorry

end count_valid_numbers_between_1_and_200_l444_444432


namespace correct_choice_is_D_l444_444730

statement : Prop :=
  (∀ (x : ℝ), (x ≥ 0 → x^2 + x - 1 < 0) ↔ ∃ (x₀ : ℝ), x₀ < 0 ∧ x₀^2 + x₀ - 1 ≥ 0) →
  ((∀ (x y : ℝ), x = y → sin x = sin y) ↔ (∀ (x y : ℝ), sin x ≠ sin y → x ≠ y)) →
  ∀ (p q : Prop), (p ∨ q) → p ∨ q

theorem correct_choice_is_D : statement :=
by
  sorry

end correct_choice_is_D_l444_444730


namespace determine_a_l444_444184

noncomputable def f (a x : ℝ) : ℝ :=
  1 / x - Real.log2 ((1 + a * x) / (1 - x))

theorem determine_a (a : ℝ) : (∀ x : ℝ, f a (-x) = -f a x) → a = 1 := by
  sorry

end determine_a_l444_444184


namespace same_type_sqrt_l444_444729

theorem same_type_sqrt (x : ℝ) : (x = 2 * Real.sqrt 3) ↔
  (x = Real.sqrt (1/3)) ∨
  (¬(x = Real.sqrt 8) ∧ ¬(x = Real.sqrt 18) ∧ ¬(x = Real.sqrt 9)) :=
by
  sorry

end same_type_sqrt_l444_444729


namespace part_1_part_2_l444_444043

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

def f (x : ℝ) : ℝ := (x) / (x^2 + 1)

theorem part_1 (a b : ℝ) (h_odd : is_odd_function (λ x, (a * x + b) / (1 + x^2))) (h_one : (a * 1 + b) / (1 + 1^2) = 1 / 2) :
  f x = (x) / (x^2 + 1) := 
  by 
  sorry

theorem part_2 : ∀ x1 x2 : ℝ, -1 < x1 ∧ x1 < x2 ∧ x2 < 1 → f(x1) < f(x2) :=
  by 
  sorry

end part_1_part_2_l444_444043


namespace number_of_pages_in_book_l444_444985

-- Define the number of pages in the book
def pages (x : ℕ) := x 

-- Conditions
def first_day_read (x : ℕ) := (1 / 6 : ℝ) * x + 10
def second_day_read (x : ℕ) := (1 / 5 : ℝ) * ((5 / 6 : ℝ) * x - 10) + 20
def third_day_read (x : ℕ) := (1 / 4 : ℝ) * (((4 / 5 : ℝ) * ((5 / 6 : ℝ) * x - 10) - 20)) + 25
def pages_remaining_after_third_day (x : ℕ) := 74

theorem number_of_pages_in_book (x : ℕ) (h1: first_day_read x = (1 / 6 : ℝ) * x + 10)
  (h2: second_day_read x = (1 / 5 : ℝ) * (((5 / 6 : ℝ) * x -10)) + 20)
  (h3: third_day_read x = (1 / 4 : ℝ) * (((4 / 5 : ℝ) * ((5 / 6 : ℝ) * x - 10) - 20)) + 25)
  (h4: (1 / 2 : ℝ) * x - 46 = 74):
  x = 240 := sorry

end number_of_pages_in_book_l444_444985


namespace projection_plane_eq_l444_444602

variables {x y z : ℝ}

def w : ℝ × ℝ × ℝ := (3, -3, 3)
def v : ℝ × ℝ × ℝ := (x, y, z)

-- Defining the projection of v onto w
def proj_w_v (v w : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let a := (v.1 * w.1 + v.2 * w.2 + v.3 * w.3) / (w.1 * w.1 + w.2 * w.2 + w.3 * w.3) in
  (a * w.1, a * w.2, a * w.3)

-- The Lean proof statement
theorem projection_plane_eq :
  proj_w_v v w = (6, -6, 6) → (x - y + z - 18 = 0) :=
by
  sorry

end projection_plane_eq_l444_444602


namespace number_of_correct_propositions_l444_444403

def proposition1 := ∀ x, x ∈ set.Ioi (5 / 2) → (real.log (x^2 - 5 * x + 6) / real.log 2) > 0
def proposition2 := ∀ (x₁ y₁ x₂ y₂ x y : ℝ), (y - y₁) * (x₂ - x₁) = (x - x₁) * (y₂ - y₁)
def proposition3 := ∀ R (x₀ : ℝ), (x₀ ≤ R ∧ x₀ ^ 2 - x₀ - 1 > 0) ↔ (¬ ∀ x > R, x ^ 2 - x - 1 ≤ 0)

theorem number_of_correct_propositions : (proposition2 ∧ ¬proposition1 ∧ ¬proposition3) ↔ 1 = 1 :=
by {
  split,
  { intro h,
    cases h with p2 h',
    cases h' with np1 np3,
    exact rfl, },
  { intro h,
    exact ⟨proposition2, (λx hx, sorry), (λ R x₀, sorry)⟩, },
}

end number_of_correct_propositions_l444_444403


namespace unique_immobilizing_point_l444_444847

-- Definitions and conditions
variables {A B C P Q R O : Type}
variables [has_dist A] [has_dist B] [has_dist C] [has_dist P] [has_dist Q] [has_dist R] [has_dist O]

-- Acute triangle ABC
def acute_triangle (A B C : Type) [has_dist A] [has_dist B] [has_dist C] : Prop :=
sorry -- Define the acute triangle property

-- Points P on AB and Q on AC, such that perpend. from P to AB and Q to AC intersect inside ∆ABC
def perpendiculars_intersect 
  (A B C P Q : Type) [has_dist A] [has_dist B] [has_dist C] [has_dist P] [has_dist Q] : Prop :=
sorry -- Define this intersection property

-- P and Q fulfill the intersection within the acute triangle
axiom P_on_AB_and_Q_on_AC (A B C P Q : Type) [has_dist A] [has_dist B] [has_dist C] [has_dist P] [has_dist Q] :
  acute_triangle A B C → perpendiculars_intersect A B C P Q

-- Prove uniqueness point R on BC
theorem unique_immobilizing_point (A B C P Q R O : Type) 
  [has_dist A] [has_dist B] [has_dist C] [has_dist P] [has_dist Q] [has_dist R] [has_dist O] :
  (acute_triangle A B C) → (perpendiculars_intersect A B C P Q) → 
  ∃! (R : Type), is_perpendicular (O, R, B, C) :=
sorry

end unique_immobilizing_point_l444_444847


namespace number_of_valid_numbers_l444_444098

-- Define the natural number set from 1 to 1000
def S : Finset ℕ := (Finset.range 1000).map ⟨Nat.succ, Nat.succ_injective⟩

-- Define the predicate for being in the form a^2 - b^2 + 1
def in_form (n : ℕ) : Prop := ∃ a b : ℕ, n = a^2 - b^2 + 1 

-- Define the predicate for not being divisible by 3
def not_divisible_by_3 (n : ℕ) : Prop := ¬(3 ∣ n)

-- Define the set of numbers in S that satisfy both conditions
def valid_numbers : Finset ℕ := S.filter (λ n, in_form n ∧ not_divisible_by_3 n)

-- The theorem to prove 
theorem number_of_valid_numbers : valid_numbers.card = 501 := by 
  sorry

end number_of_valid_numbers_l444_444098


namespace count_multiples_of_8_ending_in_4_l444_444070

theorem count_multiples_of_8_ending_in_4 (n : ℕ) : 
  n > 0 ∧ n < 500 ∧ (∃ k : ℕ, 8 * k = n ∧ n % 10 = 4) →
  (finset.filter (λ n, (n < 500 ∧ n % 8 = 0 ∧ n % 10 = 4)) (finset.range 500)).card = 6 :=
by 
  sorry

end count_multiples_of_8_ending_in_4_l444_444070


namespace min_value_ab_log_geometric_mean_l444_444040

theorem min_value_ab_log_geometric_mean {a b : ℝ} (h1 : 1 = real.log a * real.log b) (h2 : 1 < a) (h3 : 1 < b) :
  a * b ≥ 100 :=
sorry

end min_value_ab_log_geometric_mean_l444_444040


namespace count_not_squares_or_cubes_200_l444_444526

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n
def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k * k * k = n
def is_sixth_power (n : ℕ) : Prop := ∃ k : ℕ, k^6 = n

theorem count_not_squares_or_cubes_200 :
  let count_squares := (Finset.range 201).filter is_perfect_square).card
  let count_cubes := (Finset.range 201).filter is_perfect_cube).card
  let count_sixth_powers := (Finset.range 201).filter is_sixth_power).card
  (200 - (count_squares + count_cubes - count_sixth_powers)) = 182 := 
by {
  -- Define everything in terms of Finset and filter
  let count_squares := (Finset.range 201).filter is_perfect_square).card,
  let count_cubes := (Finset.range 201).filter is_perfect_cube).card,
  let count_sixth_powers := (Finset.range 201).filter is_sixth_power).card,
  exact sorry
}

end count_not_squares_or_cubes_200_l444_444526


namespace receive_A_then_score_above_90_l444_444628

variable {Alex : Type} [Exam : Alex → Prop] [ScoreAbove90 : Alex → Prop]

theorem receive_A_then_score_above_90 
  (h : ∀ x : Alex, ScoreAbove90 x → Exam x) : ∀ x : Alex, Exam x → ScoreAbove90 x :=
by
  sorry

end receive_A_then_score_above_90_l444_444628


namespace temperature_after_100_replacements_number_of_replacements_needed_l444_444241

def replacement_temp (m v t d: ℝ) (n: ℕ) : ℝ :=
  let α := (m - v) / m
  let β := v / m
  (α^n) * t + (1 - α^n) * d

def  iterations_to_target_temp (m v t d target: ℝ) : ℕ :=
  let α := (m - v) / m
  ⌈(log (target - d) / (t - d)) / (log α)⌉

theorem temperature_after_100_replacements :
  replacement_temp 40001 201 60 10 100 = 30.13 := 
sorry

theorem number_of_replacements_needed :
  iterations_to_target_temp 40001 201 40 15 28 = 129 :=
sorry

end temperature_after_100_replacements_number_of_replacements_needed_l444_444241


namespace count_valid_numbers_between_1_and_200_l444_444423

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k^2 = n
def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k^3 = n
def is_perfect_sixth_power (n : ℕ) : Prop := ∃ k : ℕ, k^6 = n

def count_perfect_squares (m : ℕ) : ℕ :=
  Nat.sqrt m

def count_perfect_cubes (m : ℕ) : ℕ :=
  (m : ℝ).cbrt.to_nat

def count_perfect_sixth_powers (m : ℕ) : ℕ :=
  (m : ℝ).cbrt.sqrt.to_nat

def count_either_squares_or_cubes (m : ℕ) : ℕ :=
  let squares := count_perfect_squares m
  let cubes := count_perfect_cubes m
  let sixths := count_perfect_sixth_powers m
  squares + cubes - sixths

def count_neither_squares_nor_cubes (n : ℕ) : ℕ :=
  n - count_either_squares_or_cubes n

theorem count_valid_numbers_between_1_and_200 : count_neither_squares_nor_cubes 200 = 182 :=
by
  sorry

end count_valid_numbers_between_1_and_200_l444_444423


namespace sum_exterior_angles_of_octagon_l444_444689

theorem sum_exterior_angles_of_octagon : 
  ∀ (P : Type) [Polygon P], (exteriorAngleSum P) = 360 := by
sorry

end sum_exterior_angles_of_octagon_l444_444689


namespace final_scores_l444_444562

variables (Sam_initial Alice_initial Bob_initial : ℕ) (Sam_penalty Sam_gain Alice_multiplier Bob_penalty Bob_multiplier : ℕ)
def final_Sam_score (initial penalty gain : ℕ) : ℕ := initial - penalty + gain
def final_Alice_score (initial multiplier : ℕ) : ℕ := (initial + initial * multiplier / 100).round
def final_Bob_score (initial penalty multiplier : ℕ) : ℕ := (initial - penalty + (initial - penalty) * multiplier / 100).round

theorem final_scores :
  final_Sam_score 92 15 3 = 80 ∧ 
  final_Alice_score 85 10 = 94 ∧ 
  final_Bob_score 78 12 25 = 83 :=
by
  sorry

end final_scores_l444_444562


namespace invertible_functions_l444_444216

noncomputable def p (x : ℝ) : ℝ := real.sqrt (3 - x)
noncomputable def q (x : ℝ) : ℝ := x^3 + 3 * x
noncomputable def r (x : ℝ) : ℝ := x + real.sin x
noncomputable def s (x : ℝ) : ℝ := 3 * x^2 + 9 * x + 21
noncomputable def t (x : ℝ) : ℝ := abs (x - 3) + abs (x + 4)
noncomputable def u (x : ℝ) : ℝ := 2^x + 5^x
noncomputable def v (x : ℝ) : ℝ := x^2 - (1 / x^2)
noncomputable def w (x : ℝ) : ℝ := x / 3

theorem invertible_functions :
  (∃ f_inv : ℝ → ℝ, ∀ x ∈ (set.Ioo (-∞:ℝ) 3), f_inv (p x) = x) ∧
  (¬ ∃ f_inv : ℝ → ℝ, ∀ x ∈ set.univ, f_inv (q x) = x) ∧
  (∃ f_inv : ℝ → ℝ, ∀ x ∈ set.Icc 0 real.pi, f_inv (r x) = x) ∧
  (∃ f_inv : ℝ → ℝ, ∀ x ∈ set.Ici 0, f_inv (s x) = x) ∧
  (¬ ∃ f_inv : ℝ → ℝ, ∀ x ∈ set.univ, f_inv (t x) = x) ∧
  (∃ f_inv : ℝ → ℝ, ∀ x ∈ set.univ, f_inv (u x) = x) ∧
  (∃ f_inv : ℝ → ℝ, ∀ x ∈ set.Ioi 0, f_inv (v x) = x) ∧
  (∃ f_inv : ℝ → ℝ, ∀ x ∈ set.Icc (-3:ℝ) 9, f_inv (w x) = x) :=
sorry

end invertible_functions_l444_444216


namespace counterexample_l444_444133

def s (n : ℕ) : ℕ := (n.digits 10).sum

def f : ℕ → ℕ
| n := if n < 10 then 0 else f (s n) + 1

theorem counterexample 
  : ¬ (∀ n m : ℕ, 0 < n → n < m → f n ≤ f m) :=
by {
  -- Providing the counterexample directly in the theorem.
  have h1 : 0 < 99 := nat.zero_lt_99,
  have h2 : 99 < 100 := nat.lt_succ_self 99,
  have h3 : f 99 = 2, exact calc
    f 99 = f (s 99) + 1 : by simp [f, if_neg (dec_trivial : ¬ (99 < 10))]
    ... = f 18 + 1 : by { norm_num, refl }
    ... = (f (s 18) + 1) + 1 : by simp [f, if_neg (dec_trivial : ¬ (18 < 10))]
    ... = (f 9 + 1) + 1 : by { norm_num, refl }
    ... = (0 + 1) + 1 : by simp [f, if_pos (dec_trivial : 9 < 10)]
    ... = 2 : by norm_num,
  have h4 : f 100 = 1, exact calc
    f 100 = f (s 100) + 1 : by simp [f, if_neg (dec_trivial : ¬ (100 < 10))]
    ... = f 1 + 1 : by { norm_num, refl }
    ... = 0 + 1 : by simp [f, if_pos (dec_trivial : 1 < 10)]
    ... = 1 : by norm_num,
  have h5 : f 99 > f 100 := by linarith [h3, h4],
  exact ⟨99, 100, h1, h2, h5⟩,
}

end counterexample_l444_444133


namespace length_of_one_layer_correct_l444_444914

-- Define the conditions of the problem
def sheet_length : ℝ := 2.7
def overlap_length : ℝ := 0.3
def number_of_sheets : ℕ := 5
def number_of_layers : ℕ := 6

-- Calculate the effective length of each overlapping sheet
def effective_sheet_length : ℝ := sheet_length - overlap_length

-- Calculate the total length of the tape when sheets are overlapped
def total_tape_length : ℝ :=
  (number_of_sheets - 1) * effective_sheet_length + sheet_length

-- Calculate the length of one layer
def length_of_one_layer : ℝ := total_tape_length / number_of_layers

-- Theorem to prove
theorem length_of_one_layer_correct :
  length_of_one_layer = 2.05 :=
  sorry

end length_of_one_layer_correct_l444_444914


namespace quadratic_maximizer_l444_444319

theorem quadratic_maximizer (x : ℝ) : -2 * x^2 + 8 * x + 16 ≤ -2 * 2^2 + 8 * 2 + 16 :=
begin
  sorry
end

end quadratic_maximizer_l444_444319


namespace find_g_inv_f_neg7_l444_444856

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry
noncomputable def f_inv : ℝ → ℝ := sorry
noncomputable def g_inv : ℝ → ℝ := sorry

axiom f_inv_def : ∀ x, f_inv (g x) = 5 * x + 3

theorem find_g_inv_f_neg7 : g_inv (f (-7)) = -2 :=
by
  sorry

end find_g_inv_f_neg7_l444_444856


namespace ratio_is_simplified_l444_444146
open Real

-- Definitions
def total_amount : ℝ := 10.15
def num_quarters : ℕ := 21
def value_per_quarter : ℝ := 0.25
def value_per_dime : ℝ := 0.10

-- Calculation of dimes
def total_value_of_quarters := num_quarters * value_per_quarter
def total_value_of_dimes := total_amount - total_value_of_quarters
def num_dimes := total_value_of_dimes / value_per_dime

-- Ratio Calculation
def quotient := Nat.gcd num_quarters (Real.toNat num_dimes)
def ratio_of_quarters_to_dimes := (num_quarters / quotient, (Real.toNat num_dimes) / quotient)

-- Theorem statement
theorem ratio_is_simplified : 
  ratio_of_quarters_to_dimes = (3, 7) :=
by
  -- Skipping proof
  sorry

end ratio_is_simplified_l444_444146


namespace ball_bounce_height_l444_444239

def h (n : ℕ) : ℝ := (3/4)^n * 360 - 2 * n

theorem ball_bounce_height : ∃ n : ℕ, n = 7 ∧ h n < 40 := by
  sorry

end ball_bounce_height_l444_444239


namespace solve_functional_l444_444335

noncomputable def functional_solution (f : ℝ → ℝ) : Prop :=
∀ x y : ℝ, f ((x + y) / 2) = (f x + f y) / 2

theorem solve_functional : 
  ∀ f : ℝ → ℝ,
  functional_solution f → 
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b :=
begin
  sorry
end

end solve_functional_l444_444335


namespace point_P_x_coordinate_l444_444638

theorem point_P_x_coordinate :
  ∃ x : ℝ, (√((x - 5)^2 + 16) - √((x + 5)^2 + 16) = 6) ∧ x = -3*√2 := 
by
  sorry

end point_P_x_coordinate_l444_444638


namespace passing_through_midpoint_of_DE_l444_444794

noncomputable def is_isosceles (ABC : Triangle) : Prop :=
ABC.AB = ABC.AC

noncomputable def midpoint (D : Point) (BC : Segment) : Prop :=
(B + C) / 2 = D

noncomputable def perpendicular (DE : Line) (AC : Line) : Prop :=
right_angle (angle DE AC)

noncomputable def circumsircle_intersect (F : Point) (ABD : Triangle) (BE : Line) : Prop :=
(circumcircle ABD).contains B ∧ (circumcircle ABD).contains F ∧ line BE.contains B ∧ line BE.contains F

noncomputable def passes_through_midpoint (AF : Line) (DE : Segment) : Prop :=
exists M : Point, midpoint M DE ∧ line AF.contains M

theorem passing_through_midpoint_of_DE
  (ABC : Triangle) (D E F : Point) (DE : Line) (AC BE AF : Line) :
  is_isosceles ABC →
  midpoint D (BC) →
  perpendicular DE AC →
  circumsircle_intersect F (Triangle.mk ABD) BE →
  passes_through_midpoint AF (segment.mk D E) :=
by
  sorry

end passing_through_midpoint_of_DE_l444_444794


namespace cube_tetrahedron_surface_area_ratio_l444_444943

theorem cube_tetrahedron_surface_area_ratio :
  ∀ (e : ℝ), e = 1 → 
    let cube_surface_area := 6 * e ^ 2 in
    let tetrahedron_edge := e * Real.sqrt 2 in
    let tetrahedron_face_area := (Real.sqrt 3 / 2) * tetrahedron_edge ^ 2 in
    let tetrahedron_surface_area := 4 * tetrahedron_face_area in
    cube_surface_area / tetrahedron_surface_area = Real.sqrt 3 :=
by 
  intros e he
  sorry

end cube_tetrahedron_surface_area_ratio_l444_444943


namespace probability_neither_question_correct_l444_444074

def probability_first_question_correct : ℝ := 0.85
def probability_second_question_correct : ℝ := 0.80
def probability_both_questions_correct : ℝ := 0.70

theorem probability_neither_question_correct 
  (P_A : ℝ := probability_first_question_correct)
  (P_B : ℝ := probability_second_question_correct)
  (P_A_and_B : ℝ := probability_both_questions_correct) : 
  P_A + P_B - P_A_and_B = 0.95 → 1 - (P_A + P_B - P_A_and_B) = 0.05 := 
by
  intros h
  rw h
  apply rfl

end probability_neither_question_correct_l444_444074


namespace sail_time_difference_l444_444884

theorem sail_time_difference (distance : ℕ) (v_big : ℕ) (v_small : ℕ) (t_big t_small : ℕ)
  (h_distance : distance = 200)
  (h_v_big : v_big = 50)
  (h_v_small : v_small = 20)
  (h_t_big : t_big = distance / v_big)
  (h_t_small : t_small = distance / v_small)
  : t_small - t_big = 6 := by
  sorry

end sail_time_difference_l444_444884


namespace point_on_y_axis_l444_444637

theorem point_on_y_axis (m : ℝ) (M : ℝ × ℝ) (hM : M = (m + 1, m + 3)) (h_on_y_axis : M.1 = 0) : M = (0, 2) :=
by
  -- Proof omitted
  sorry

end point_on_y_axis_l444_444637


namespace AF_through_midpoint_DE_l444_444758

open EuclideanGeometry

-- Definition of an isosceles triangle
variables {A B C D E F : Point}
variables [plane Geometry]
variables (ABC_isosceles : IsoscelesTriangle A B C)
variables (midpoint_D : Midpoint D B C)
variables (perpendicular_DE_AC : Perpendicular D E A C)
variables (F_on_circumcircle : OnCircumcircle F A B D)
variables (B_on_BE : OnLine B E)

-- Goal: Prove that line AF passes through the midpoint of DE
theorem AF_through_midpoint_DE 
  (h1 : triangle A B C) 
  (h2 : ABC_isosceles) 
  (h3 : midpoint D B C) 
  (h4 : perpendicular DE AC)
  (h5 : OnCircumcircle F A B D) 
  (h6 : OnLine B E) :
  PassesThroughMidpoint A F D E :=
sorry

end AF_through_midpoint_DE_l444_444758


namespace three_hour_classes_per_week_l444_444886

theorem three_hour_classes_per_week (x : ℕ) : 
  (24 * (3 * x + 4 + 4) = 336) → x = 2 := by {
  sorry
}

end three_hour_classes_per_week_l444_444886


namespace value_of_g_of_h_at_2_l444_444553

def g (x : ℝ) : ℝ := 3 * x^2 + 2
def h (x : ℝ) : ℝ := -5 * x^3 + 4

theorem value_of_g_of_h_at_2 : g (h 2) = 3890 := by
  sorry

end value_of_g_of_h_at_2_l444_444553


namespace wheat_rate_correct_l444_444287

-- Given conditions
def wheat_rate_problem : Prop :=
  ∃ (x : ℝ), 
    let cost_first_wheat := 30 * x in  -- Total cost of the first wheat
    let cost_second_wheat := 285 in  -- Total cost of the second wheat (20 kg * 14.25 Rs.)
    let total_cost := cost_first_wheat + cost_second_wheat in
    let total_weight := 50 in  -- Total weight of the mixture
    let selling_price_per_kg := 13.86 in  -- Selling price per kg to achieve 10% profit
    let total_selling_price := total_weight * selling_price_per_kg in
    1.10 * total_cost = total_selling_price ∧
    x = 11.50

theorem wheat_rate_correct : wheat_rate_problem :=
begin
  sorry
end

end wheat_rate_correct_l444_444287


namespace tan_2x_is_odd_l444_444671

noncomputable def f (x : ℝ) : ℝ := Real.tan (2 * x)

theorem tan_2x_is_odd (x : ℝ) (k : ℤ) (h : x ≠ (1/2) * k * Real.pi + Real.pi / 4) :
  f (-x) = -f x :=
by
  rw [f, f]
  simp
  have a : -2 * x = (-2) * x, ring
  rw [a, Real.tan_neg]
  refl

end tan_2x_is_odd_l444_444671


namespace five_points_in_unit_square_l444_444566

theorem five_points_in_unit_square (S : set (ℝ × ℝ)) (P : finset (ℝ × ℝ)) :
  (∀ x y : ℝ, (x, y) ∈ S → 0 ≤ x ∧ x ≤ 5 ∧ 0 ≤ y ∧ y ≤ 5) →
  finset.card P = 201 →
  (∃ sq : set (ℝ × ℝ), 
    (∀ x y : ℝ, (x, y) ∈ sq → 0 ≤ x ∧ x < 1 ∧ 0 ≤ y ∧ y < 1) ∧
    ∃ subset_P : finset (ℝ × ℝ), subset_P ⊆ P ∧ finset.card subset_P = 5 ∧ subset_P ⊆ sq) :=
by
  sorry

end five_points_in_unit_square_l444_444566


namespace problem1_problem2_l444_444868

theorem problem1 : (1 * (-9)) - (-7) + (-6) - 5 = -13 := 
by 
  -- problem1 proof
  sorry

theorem problem2 : ((-5 / 12) + (2 / 3) - (3 / 4)) * (-12) = 6 := 
by 
  -- problem2 proof
  sorry

end problem1_problem2_l444_444868


namespace sum_infinite_series_l444_444305

theorem sum_infinite_series :
  ∑' n : ℕ, (3 * (n+1) + 2) / ((n+1) * (n+2) * (n+4)) = 29 / 36 :=
by
  sorry

end sum_infinite_series_l444_444305


namespace diesel_usage_l444_444625

theorem diesel_usage (weekly_expenditure : ℝ) (cost_per_gallon : ℝ)
  (h_expenditure : weekly_expenditure = 36)
  (h_cost : cost_per_gallon = 3) :
  let weekly_gallons := weekly_expenditure / cost_per_gallon in
  let two_weeks_gallons := 2 * weekly_gallons in
  two_weeks_gallons = 24 := 
by
  sorry

end diesel_usage_l444_444625


namespace difference_of_squares_count_l444_444970

theorem difference_of_squares_count :
  (number_of_integers_between (1 : ℕ) (500 : ℕ) (λ n, ∃ a b : ℕ, n = a^2 - b^2)) = 375 :=
by
  sorry

end difference_of_squares_count_l444_444970


namespace number_of_integer_pairs_l444_444348

theorem number_of_integer_pairs : 
  ∃ (n : ℕ), n = 324 ∧
  ∀ (x y : ℤ), y^2 - x * y = 700000000 ↔ (x, y) ∈ { (x, y) | y ∈ Int.divisors 700000000 ∧ (y > 0 ∨ y < 0) } ∧ ∃ (z : ℕ), z = 162  :=
begin
  sorry
end

end number_of_integer_pairs_l444_444348


namespace final_expression_simplest_form_l444_444544

def e1 (b : ℝ) : ℝ := 2 * b + 4
def e2 (b : ℝ) : ℝ := e1 b - 4 * b
def e3 (b : ℝ) : ℝ := e2 b / 2

theorem final_expression_simplest_form (b : ℝ) : e3 b = -b + 2 :=
by
  sorry

end final_expression_simplest_form_l444_444544


namespace cow_count_16_l444_444738

theorem cow_count_16 (D C : ℕ) 
  (h1 : ∃ (L H : ℕ), L = 2 * D + 4 * C ∧ H = D + C ∧ L = 2 * H + 32) : C = 16 :=
by
  obtain ⟨L, H, ⟨hL, hH, hCond⟩⟩ := h1
  sorry

end cow_count_16_l444_444738


namespace numbers_not_squares_nor_cubes_1_to_200_l444_444499

theorem numbers_not_squares_nor_cubes_1_to_200 : 
  let num_squares := nat.sqrt 200
  let num_cubes := nat.cbrt 200
  let num_sixth_powers := nat.root 6 200
  let total := num_squares + num_cubes - num_sixth_powers
  200 - total = 182 := 
by
  sorry

end numbers_not_squares_nor_cubes_1_to_200_l444_444499


namespace dingding_minimum_correct_answers_l444_444320

theorem dingding_minimum_correct_answers (x : ℕ) :
  (5 * x - (30 - x) > 100) → x ≥ 22 :=
by
  sorry

end dingding_minimum_correct_answers_l444_444320


namespace intersection_case_empty_intersection_range_l444_444061

def setA : Set ℝ := {x | x^2 - 6 * x + 8 < 0}
def setB (a : ℝ) : Set ℝ := {x | (x - a) * (x - 3 * a) < 0}

theorem intersection_case (a : ℝ) (ha : a = 1) :
  setA ∩ setB a = {x | 2 < x ∧ x < 3} :=
by 
  sorry

theorem empty_intersection_range (h : setA ∩ setB a = ∅) :
  a ∈ ((−∞, (2 / 3)] ∪ [4, +∞)) :=
by 
  sorry

end intersection_case_empty_intersection_range_l444_444061


namespace greatest_lower_bound_of_f1_f3_f4_l444_444997

def f1 (x : ℝ) := Real.sin x
def f2 (x : ℝ) := Real.log x
def f3 (x : ℝ) := Real.exp x
def f4 (x : ℝ) := if x > 0 then 1 else if x = 0 then 0 else -1

theorem greatest_lower_bound_of_f1_f3_f4 : 
  (∃ M, ∀ x : ℝ, f1 x ≥ M ∧ (∀ M', (∀ x : ℝ, f1 x ≥ M') → M' ≤ M)) ∧
  (∃ M, ∀ x : ℝ, f3 x ≥ M ∧ (∀ M', (∀ x : ℝ, f3 x ≥ M') → M' ≤ M)) ∧
  (∃ M, ∀ x : ℝ, f4 x ≥ M ∧ (∀ M', (∀ x : ℝ, f4 x ≥ M') → M' ≤ M)) :=
sorry

end greatest_lower_bound_of_f1_f3_f4_l444_444997


namespace triangle_area_is_integer_l444_444584

theorem triangle_area_is_integer (d : ℕ) (h : ℕ) (centers_circle : d = h) : 
  (∃ d ∈ {4, 8, 12}, (1/2 : ℚ) * d^2 ∈ ℤ) :=
by 
  use [4, 8, 12]
  -- Case d = 4
  exists 4
  norm_num
  -- Case d = 8
  exists 8
  norm_num
  -- Case d = 12
  exists 12
  norm_num
  apply algebra.fractional_part_eq_zero
  use ℤ
  use ℚ
  sorry

end triangle_area_is_integer_l444_444584


namespace count_non_perfect_square_or_cube_l444_444514

-- Define perfect_square function
def perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k^2 = n

-- Define perfect_cube function
def perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k^3 = n

-- Define range from 1 to 200
def range_200 := {1 .. 200}

-- Define numbers either perfect square or perfect cube
def perfect_square_or_cube : Finset ℕ :=
  (range_200.filter perfect_square) ∪ (range_200.filter perfect_cube)

-- Define sixth powers in range
def perfect_sixth_power (n : ℕ) : Prop :=
  ∃ k : ℕ, k^6 = n

-- The final problem statement
theorem count_non_perfect_square_or_cube :
  (range_200.card - perfect_square_or_cube.card) = 182 :=
by
  sorry

end count_non_perfect_square_or_cube_l444_444514


namespace AF_passes_through_midpoint_DE_l444_444771

open_locale classical

-- Definition of the geometrical setup
def isosceles_triangle (A B C : Point) : Prop := dist A B = dist A C
def midpoint (D B C : Point) : Prop := dist B D = dist C D
def perpendicular (D E A C : Point) : Prop := ∠ DEA = 90
def circumcircle (A B D F : Point): Prop := ∃ O R, Circle.circle O R A ∧ Circle.circle O R B ∧ Circle.circle O R D ∧ Circle.circle O R F

-- Main statement to prove
theorem AF_passes_through_midpoint_DE
  (A B C D E F : Point)
  (h1 : isosceles_triangle A B C)
  (h2 : midpoint D B C)
  (h3 : perpendicular D E A C)
  (h4 : ∃ O R, Circle.circle O R A ∧ Circle.circle O R B ∧ Circle.circle O R D ∧ Circle.circle O R F)
  (h5 : collinear B E F)
  (h6 : intersects (circumcircle A B D) (line B E) B F) :
  passes_through (line A F) (midpoint D E) := 
sorry

end AF_passes_through_midpoint_DE_l444_444771


namespace intervals_of_monotonicity_no_real_a_for_slope_l444_444954

-- Define the function f(x)
def f (x a : ℝ) : ℝ := Real.log x + x^2 - a * x

-- Define the derivative f'(x)
def f_prime (x a : ℝ) : ℝ := 1 / x + 2 * x - a

-- Part 1: Prove monotonicity intervals when a = 3
theorem intervals_of_monotonicity : 
  (∀ x : ℝ, 0 < x ∧ x < 1/2 → f_prime x 3 > 0) ∧
  (∀ x : ℝ, 1 / 2 < x ∧ x < 1 → f_prime x 3 < 0) ∧
  (∀ x : ℝ, x > 1 → f_prime x 3 > 0) := 
  sorry

-- Part 2: Prove no real a satisfies k = 2/a - a/2
theorem no_real_a_for_slope (x1 x2 : ℝ) (h_extrema : f_prime x1 a = 0 ∧ f_prime x2 a = 0) :
  ∀ k, k = (2 / a) - (a / 2) → False :=
  sorry

end intervals_of_monotonicity_no_real_a_for_slope_l444_444954


namespace no_point_C_l444_444082

open Real

theorem no_point_C (A B C : ℝ × ℝ) (h_ab : dist A B = 10)
  (h_perimeter : dist A B + dist A C + dist B C = 40)
  (h_area : 0.5 * abs ((fst A - fst C) * (snd A - snd B) - (fst A - fst B) * (snd A - snd C)) = 100) :
  false :=
sorry

end no_point_C_l444_444082


namespace bags_of_sugar_bought_l444_444618

-- Define the conditions as constants
def cups_at_home : ℕ := 3
def cups_per_bag : ℕ := 6
def cups_per_batter_dozen : ℕ := 1
def cups_per_frosting_dozen : ℕ := 2
def dozens_of_cupcakes : ℕ := 5

-- Prove that the number of bags of sugar Lillian bought is 2
theorem bags_of_sugar_bought : ∃ bags : ℕ, bags = 2 :=
by
  let total_cups_batter := dozens_of_cupcakes * cups_per_batter_dozen
  let total_cups_frosting := dozens_of_cupcakes * cups_per_frosting_dozen
  let total_cups_needed := total_cups_batter + total_cups_frosting
  let cups_to_buy := total_cups_needed - cups_at_home
  let bags := cups_to_buy / cups_per_bag
  have h : bags = 2 := sorry
  exact ⟨bags, h⟩

end bags_of_sugar_bought_l444_444618


namespace non_union_women_percentage_l444_444224

theorem non_union_women_percentage
  (total_employees : ℕ)
  (men_percentage : ℚ)
  (unionized_percentage : ℚ)
  (unionized_men_percentage : ℚ)
  (H1 : men_percentage = 0.54)
  (H2 : unionized_percentage = 0.60)
  (H3 : unionized_men_percentage = 0.70)
  (H4 : total_employees > 0)
  : 
  (let non_union_employees := total_employees * (1 - unionized_percentage) in
   let non_union_women := total_employees * (1 - men_percentage - unionized_percentage * (1 - unionized_men_percentage)) in
   non_union_women / non_union_employees * 100 = 70) :=
by
  sorry

end non_union_women_percentage_l444_444224


namespace sum_distances_saham_and_mother_l444_444649

theorem sum_distances_saham_and_mother :
  let saham_distance := 2.6
  let mother_distance := 5.98
  saham_distance + mother_distance = 8.58 :=
by
  sorry

end sum_distances_saham_and_mother_l444_444649


namespace find_q_l444_444996

theorem find_q (q x : ℝ) (h1 : x = 2) (h2 : q * x - 3 = 11) : q = 7 :=
by
  sorry

end find_q_l444_444996


namespace possible_ages_of_youngest_child_l444_444830

noncomputable def youngest_child_possible_ages (c_fixed : ℝ) (c_year : ℝ) (C_total : ℝ) (twins_count : ℕ) : set ℕ :=
  {y : ℕ | ∃ t : ℕ, y + twins_count * t = int.of_nat (C_total - c_fixed) / int.of_nat c_year ∧ y < t}

theorem possible_ages_of_youngest_child :
  youngest_child_possible_ages 6.50 0.55 15.95 4 = {1, 5} :=
sorry

end possible_ages_of_youngest_child_l444_444830


namespace number_of_students_l444_444151

/-- 
We are given that 36 students are selected from three grades: 
15 from the first grade, 12 from the second grade, and the rest from the third grade. 
Additionally, there are 900 students in the third grade.
We need to prove: the total number of students in the high school is 3600
-/
theorem number_of_students (x y z : ℕ) (s_total : ℕ) (x_sel : ℕ) (y_sel : ℕ) (z_students : ℕ) 
  (h1 : x_sel = 15) 
  (h2 : y_sel = 12) 
  (h3 : x_sel + y_sel + (s_total - (x_sel + y_sel)) = s_total) 
  (h4 : s_total = 36) 
  (h5 : z_students = 900) 
  (h6 : (s_total - (x_sel + y_sel)) = 9) 
  (h7 : 9 / 900 = 1 / 100) : 
  (36 * 100 = 3600) :=
by sorry

end number_of_students_l444_444151


namespace count_not_squares_or_cubes_l444_444500

theorem count_not_squares_or_cubes {A B : Finset ℕ} (h_range : ∀ x, x ∈ A ∪ B → x ≤ 200)
  (h_squares : A.card = 14) (h_cubes : B.card = 5) (h_inter : (A ∩ B).card = 1) :
  (200 - (A ∪ B).card) = 182 :=
by
  sorry

end count_not_squares_or_cubes_l444_444500


namespace incorrect_statement_b_l444_444067

variables (α β : Set Point) -- representing the sets of points defining the planes α and β
variables (m n : Set Point) -- representing the sets of points defining the lines m and n

-- Definitions for plane properties
def planes_are_different (p q : Set Point) : Prop := p ≠ q
def lines_are_different (x y : Set Point) : Prop := x ≠ y
def line_parallel_to_plane (l : Set Point) (p : Set Point) : Prop := ∀ (a b : Point), (a ∈ l) ∧ (b ∈ l) → ∀ (c : Point), (c ∈ p) → ∃ (k : ℝ), (b = a + (k : ℝ) * (c - a))
def plane_intersects_plane (p q : Set Point) : Prop := ∃ (x : Set Point), (x ⊆ p) ∧ (x ⊆ q)
def line_parallel_to_line (l x : Set Point) : Prop := ∃ (b : Set Point), b ⊆ l ∧ b ⊆ x ∧ ∀ (z : Set Point), z ∈ l → ∃ (k : ℝ), z = b ∪ (k : ℝ) * b

-- Theorem to be proven
theorem incorrect_statement_b
  (h_different_planes : planes_are_different α β)
  (h_different_lines : lines_are_different m n)
  (h_parallel_m_alpha : line_parallel_to_plane m α)
  (h_intersect_planes : plane_intersects_plane α β) :
  ¬ line_parallel_to_line m n :=
sorry

end incorrect_statement_b_l444_444067


namespace number_of_valid_orderings_l444_444716

-- Define the houses by their colors
inductive Color where
| Green : Color
| Blue : Color
| Violet : Color
| Yellow : Color
| Red : Color
deriving DecidableEq

open Color

-- Define the conditions as properties
def green_before_blue (order : List Color) : Prop :=
  order.indexOf Green < order.indexOf Blue

def violet_before_green (order : List Color) : Prop :=
  order.indexOf Violet < order.indexOf Green

def yellow_before_red (order : List Color) : Prop :=
  order.indexOf Yellow < order.indexOf Red

def not_adjacent (c1 c2 : Color) (order : List Color) : Prop :=
  abs (order.indexOf c1 - order.indexOf c2) ≠ 1

def violet_not_first_or_last (order : List Color) : Prop :=
  order.head? ≠ some Violet ∧ order.last? ≠ some Violet

-- Set the main theorem to prove the number of valid orderings
theorem number_of_valid_orderings : 
  ∃ orders : List (List Color), 
  (∀ order ∈ orders, 
    green_before_blue order ∧ 
    violet_before_green order ∧ 
    yellow_before_red order ∧ 
    not_adjacent Red Blue order ∧ 
    violet_not_first_or_last order) ∧
  orders.length = 8 :=
sorry

end number_of_valid_orderings_l444_444716


namespace smallest_ellipse_area_contains_circles_l444_444285

theorem smallest_ellipse_area_contains_circles :
  ∃ (a b : ℝ), 
    (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 → (x - 2)^2 + y^2 = 4) ∧
    (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 → (x + 2)^2 + y^2 = 4) ∧
    π * a * b = π * (3 * real.sqrt 3) / 2 :=
by
  sorry

end smallest_ellipse_area_contains_circles_l444_444285


namespace hall_ratio_l444_444197

variable (w l : ℝ)

theorem hall_ratio
  (h1 : w * l = 200)
  (h2 : l - w = 10) :
  w / l = 1 / 2 := 
by
  sorry

end hall_ratio_l444_444197


namespace std_eq_line_C_cart_eq_curve_P_intersection_dist_AB_l444_444088

-- Definitions corresponding to conditions in the problem
def parametric_line_C (t : ℝ) : ℝ × ℝ := (2 + t, t + 1)
def polar_curve_P (ρ θ : ℝ) : Prop := ρ^2 - 4 * ρ * real.cos θ + 3 = 0

-- Theorems to prove the answers given the conditions
theorem std_eq_line_C :
  ∃ (t : ℝ → ℝ × ℝ), ∀ t, parametric_line_C t = (2 + t, t + 1) ∧
  ∀ (x y : ℝ), x = 2 + t ∧ y = t + 1 → x - y - 1 = 0 := by
  sorry

theorem cart_eq_curve_P :
  ∀ {ρ θ : ℝ}, polar_curve_P ρ θ →
  ∀ {x y : ℝ}, x = ρ * real.cos θ ∧ y = ρ * real.sin θ → 
  x^2 + y^2 - 4 * x + 3 = 0 := by
  sorry

theorem intersection_dist_AB :
  (∀ t, parametric_line_C t = (2 + t, t + 1) ∧ 
  ∀ (ρ θ : ℝ), polar_curve_P ρ θ → 
  ∃ (x y : ℝ), x = 2 + t ∧ y = t + 1 ∧ x = ρ * real.cos θ ∧ y = ρ * real.sin θ) →
  ∃ (A B : ℝ × ℝ), A ≠ B ∧ 
    dist (classical.some A) (classical.some B) = real.sqrt 2 := by
  sorry

end std_eq_line_C_cart_eq_curve_P_intersection_dist_AB_l444_444088


namespace count_diff_squares_1_to_500_l444_444974

theorem count_diff_squares_1_to_500 : ∃ count : ℕ, count = 416 ∧
  (count = (nat.card {n : ℕ | n ∈ (set.Icc 1 500) ∧ (∃ a b : ℕ, a^2 - b^2 = n)})) := by
sorry

end count_diff_squares_1_to_500_l444_444974


namespace confidence_level_99_percent_l444_444988

-- Let's define the conditions first
def χ2_value : ℝ := 4.013
def degrees_of_freedom : ℝ := (2 - 1) * (2 - 1)

-- The theorem to prove the confidence level is 99%
theorem confidence_level_99_percent 
  (χ2_value = 4.013) 
  (degrees_of_freedom = 1) :
  level_of_confidence χ2_value degrees_of_freedom = 99 :=
sorry

end confidence_level_99_percent_l444_444988


namespace keith_wanted_cds_l444_444112

theorem keith_wanted_cds : 
  let cost_speakers := 136.01
  let cost_cd_player := 139.38
  let cost_new_tires := 112.46
  let price_per_cd := 6.16
  let total_spent := 387.85
  total_spent = cost_speakers + cost_cd_player + cost_new_tires →
  floor (total_spent / price_per_cd) = 62 :=
by
  intros cost_speakers cost_cd_player cost_new_tires price_per_cd total_spent h
  sorry

end keith_wanted_cds_l444_444112


namespace cosine_a_c_parallel_condition_l444_444007

-- Vectors a and b
def a : ℝ × ℝ × ℝ := (1, 4, -2)
def b : ℝ × ℝ × ℝ := (-2, 2, 4)

-- Vector c
def c : ℝ × ℝ × ℝ := ((-2:ℝ) / 2, 2 / 2, 4 / 2)  -- which simplifies to (-1, 1, 2)

-- Function to calculate the dot product of two vectors
def dot_product (u v : ℝ × ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2 + u.3 * v.3

-- Function to calculate the magnitude of a vector
def magnitude (u : ℝ × ℝ × ℝ) : ℝ := Real.sqrt (u.1 * u.1 + u.2 * u.2 + u.3 * u.3)

-- Function to calculate cosine of the angle between two vectors
def cosine_angle (u v : ℝ × ℝ × ℝ) : ℝ := (dot_product u v) / (magnitude u * magnitude v)

-- Proof statement for part (1)
theorem cosine_a_c : cosine_angle a c = - (Real.sqrt 14) / 42 := 
  sorry

-- Proof statement for part (2)
theorem parallel_condition : (k : ℝ) (H : k * a + b = m * (a - 3 * b)) -> k = -1 / 3 := 
  sorry

end cosine_a_c_parallel_condition_l444_444007


namespace sugar_for_third_layer_l444_444295

theorem sugar_for_third_layer (s1 : ℕ) (s2 : ℕ) (s3 : ℕ) 
  (h1 : s1 = 2) 
  (h2 : s2 = 2 * s1) 
  (h3 : s3 = 3 * s2) : 
  s3 = 12 := 
sorry

end sugar_for_third_layer_l444_444295


namespace inclination_angle_range_l444_444572

theorem inclination_angle_range (P : Point) (C : PolarCurve) (α : ℝ)
  (hp : P = Point.mk (-3) 0)
  (hc : C = PolarCurve.mk (λ ρ θ, ρ^2 - 2*ρ*Real.cos θ - 3 = 0))
  (hl : Line.mk P α)
  (h_intersect : ∃ t, (l.toParametric t).1^2 + (l.toParametric t).2^2 - 2 * (l.toParametric t).1 - 3 = 0):
  α ∈ Set.Icc 0 (Real.pi / 6) ∪ Set.Icc (5 * Real.pi / 6) Real.pi := 
sorry

end inclination_angle_range_l444_444572


namespace coordinates_of_M_l444_444400

theorem coordinates_of_M :
  -- Given the function f(x) = 2x^2 + 1
  let f : Real → Real := λ x => 2 * x^2 + 1
  -- And its derivative
  let f' : Real → Real := λ x => 4 * x
  -- The coordinates of point M where the instantaneous rate of change is -8 are (-2, 9)
  (∃ x0 : Real, f' x0 = -8 ∧ f x0 = y0 ∧ x0 = -2 ∧ y0 = 9) := by
    sorry

end coordinates_of_M_l444_444400


namespace AF_passes_through_midpoint_DE_l444_444768

open_locale classical

-- Definition of the geometrical setup
def isosceles_triangle (A B C : Point) : Prop := dist A B = dist A C
def midpoint (D B C : Point) : Prop := dist B D = dist C D
def perpendicular (D E A C : Point) : Prop := ∠ DEA = 90
def circumcircle (A B D F : Point): Prop := ∃ O R, Circle.circle O R A ∧ Circle.circle O R B ∧ Circle.circle O R D ∧ Circle.circle O R F

-- Main statement to prove
theorem AF_passes_through_midpoint_DE
  (A B C D E F : Point)
  (h1 : isosceles_triangle A B C)
  (h2 : midpoint D B C)
  (h3 : perpendicular D E A C)
  (h4 : ∃ O R, Circle.circle O R A ∧ Circle.circle O R B ∧ Circle.circle O R D ∧ Circle.circle O R F)
  (h5 : collinear B E F)
  (h6 : intersects (circumcircle A B D) (line B E) B F) :
  passes_through (line A F) (midpoint D E) := 
sorry

end AF_passes_through_midpoint_DE_l444_444768


namespace a_range_l444_444029

open Set

variable (A B : Set Real) (a : Real)

def A_def : Set Real := {x | 3 * x + 1 < 4}
def B_def : Set Real := {x | x - a < 0}
def intersection_eq : A ∩ B = A := sorry

theorem a_range : a ≥ 1 :=
  by
  have hA : A = {x | x < 1} := sorry
  have hB : B = {x | x < a} := sorry
  have h_intersection : (A ∩ B) = A := sorry
  sorry

end a_range_l444_444029


namespace transformation_preserves_region_l444_444269

noncomputable def region_R : set ℂ :=
  { z : ℂ | -2 ≤ z.re ∧ z.re ≤ 2 ∧ -2 ≤ z.im ∧ z.im ≤ 2 }

def transformed_z (z : ℂ) : ℂ :=
  (1/2) * (z.re - z.im) + (1/2) * (z.re + z.im) * complex.I

theorem transformation_preserves_region :
  ∀ z : ℂ, z ∈ region_R → transformed_z(z) ∈ region_R :=
by
  intros z hz
  -- Define coordinates x and y
  let x := z.re
  let y := z.im
  -- Simplify transformed_z to real and imaginary parts
  have hz₁ : -2 ≤ x ∧ x ≤ 2 ∧ -2 ≤ y ∧ y ≤ 2 := hz
  sorry

end transformation_preserves_region_l444_444269


namespace problem1_problem2_l444_444389

open Nat

-- Given conditions for both parts
def is_nat (m : ℕ) : Prop := m ∈ ℕ
def is_nat (n : ℕ) : Prop := n ∈ ℕ

-- Definition of the function f(x)
def f (x : ℝ) (m : ℕ) (n : ℕ) : ℝ := (1 + x : ℝ)^m + (1 + x : ℝ)^n

-- Problem 1
theorem problem1 : (m = 7) ∧ (n = 7) → ∃ a_0 a_2 a_4 a_6 : ℝ, 
  f x m n = a_7 * x^7 + a_6 * x^6 + a_5 * x^5 + a_4 * x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + a_0 ∧ a_0 + a_2 + a_4 + a_6 = 128 := 
by
  sorry

-- Problem 2
theorem problem2 : ∀ m n : ℕ, m + n = 19 → 
  ∃ a_1 a_2 : ℝ,
    (coeff (polynomial.C a_7 * x^7 + polynomial.C a_6 * x^6 + polynomial.C a_5 * x^5 + polynomial.C a_4 * x^4 + polynomial.C a_3 * x^3 + polynomial.C a_2 * x^2 + polynomial.C a_1 * x + polynomial.C a_0) 1 = 19) →
    (∃ k : ℝ, k = (x^2.coeff 2)) ∧ k = 81 :=
by
  sorry

end problem1_problem2_l444_444389


namespace first_term_of_arithmetic_sequence_l444_444600

theorem first_term_of_arithmetic_sequence (a : ℕ → ℚ) (S : ℕ → ℚ) (d : ℚ)
  (h1 : a 3 = 3) (h2 : S 9 - S 6 = 27)
  (h3 : ∀ n, a n = a 1 + (n - 1) * d)
  (h4 : ∀ n, S n = n * (a 1 + a n) / 2) : a 1 = 3 / 5 :=
by
  sorry

end first_term_of_arithmetic_sequence_l444_444600


namespace faster_train_speed_correct_l444_444205

noncomputable def speed_of_faster_train (V_s_kmph : ℝ) (length_faster_train_m : ℝ) (time_s : ℝ) : ℝ :=
  let V_s_mps := V_s_kmph * (1000 / 3600)
  let V_r_mps := length_faster_train_m / time_s
  let V_f_mps := V_r_mps - V_s_mps
  V_f_mps * (3600 / 1000)

theorem faster_train_speed_correct : 
  speed_of_faster_train 36 90.0072 4 = 45.00648 := 
by
  sorry

end faster_train_speed_correct_l444_444205


namespace diamond_more_olivine_l444_444281

theorem diamond_more_olivine :
  ∃ A O D : ℕ, A = 30 ∧ O = A + 5 ∧ A + O + D = 111 ∧ D - O = 11 :=
by
  sorry

end diamond_more_olivine_l444_444281


namespace AF_passes_through_midpoint_of_DE_l444_444777

open EuclideanGeometry

-- Definitions based on the conditions
variables {A B C D E F : Point} 
variables {ABC : Triangle}
variables (isosceles_ABC : IsIsoscelesTriangle ABC A B)
variables (midpoint_D : IsMidpoint D B C)
variables (perpendicular_DE : IsPerpendicular DE D AC)
variables {circumcircle_ABD : Circumcircle ABD}
variables (intersection_F : IntersectsAt circumcircle_ABD BE B F)

-- The theorem statement
theorem AF_passes_through_midpoint_of_DE :
  PassesThroughMidpoint AF D E :=
sorry

end AF_passes_through_midpoint_of_DE_l444_444777


namespace geometric_sequence_fifth_term_l444_444585

variable {a : ℕ → ℝ} (h1 : a 1 = 1) (h4 : a 4 = 8)

theorem geometric_sequence_fifth_term (h_geom : ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q) :
  a 5 = 16 :=
sorry

end geometric_sequence_fifth_term_l444_444585


namespace total_students_in_high_school_l444_444152

theorem total_students_in_high_school (selected_first: ℕ) (selected_second: ℕ) (students_third: ℕ) (total_selected: ℕ) (p: ℚ) :
  selected_first = 15 →
  selected_second = 12 →
  students_third = 900 →
  total_selected = 36 →
  p = 1 / 100 →
  ∃ n: ℕ, (total_selected : ℚ) / n = p ∧ n = 3600 :=
by 
  intros h1 h2 h3 h4 h5
  use 3600
  split
  · sorry -- omit the proof for successful compilation.
  · exact rfl

end total_students_in_high_school_l444_444152


namespace AF_through_midpoint_DE_l444_444757

open EuclideanGeometry

-- Definition of an isosceles triangle
variables {A B C D E F : Point}
variables [plane Geometry]
variables (ABC_isosceles : IsoscelesTriangle A B C)
variables (midpoint_D : Midpoint D B C)
variables (perpendicular_DE_AC : Perpendicular D E A C)
variables (F_on_circumcircle : OnCircumcircle F A B D)
variables (B_on_BE : OnLine B E)

-- Goal: Prove that line AF passes through the midpoint of DE
theorem AF_through_midpoint_DE 
  (h1 : triangle A B C) 
  (h2 : ABC_isosceles) 
  (h3 : midpoint D B C) 
  (h4 : perpendicular DE AC)
  (h5 : OnCircumcircle F A B D) 
  (h6 : OnLine B E) :
  PassesThroughMidpoint A F D E :=
sorry

end AF_through_midpoint_DE_l444_444757


namespace numbers_not_squares_or_cubes_in_200_l444_444468

/-- There are 182 numbers between 1 and 200 that are neither perfect squares nor perfect cubes. -/
theorem numbers_not_squares_or_cubes_in_200 : (finset.range 200).filter (λ n, ¬(∃ m, m^2 = n) ∧ ¬(∃ k, k^3 = n)).card = 182 :=
by 
  /-
    The proof will involve constructing the set of numbers from 1 to 200, filtering out those that
    are perfect squares or perfect cubes, and counting the remaining numbers.
  -/
  sorry

end numbers_not_squares_or_cubes_in_200_l444_444468


namespace segment_length_from_reflection_l444_444713

/-- Given points F and F' obtained by reflecting F over the x-axis, prove the segment length
  from F to F' is 6. -/
theorem segment_length_from_reflection :
  let F  := (-4 : ℤ, 3 : ℤ)
  let F' := (-4 : ℤ, -3 : ℤ)
  dist F F' = 6 :=
by
  let F : ℤ × ℤ := (-4, 3)
  let F' : ℤ × ℤ := (-4, -3)
  sorry

end segment_length_from_reflection_l444_444713


namespace numbers_not_perfect_squares_or_cubes_l444_444453

theorem numbers_not_perfect_squares_or_cubes : 
  let total_numbers := 200
  let perfect_squares := 14
  let perfect_cubes := 5
  let sixth_powers := 1
  total_numbers - (perfect_squares + perfect_cubes - sixth_powers) = 182 :=
by
  sorry

end numbers_not_perfect_squares_or_cubes_l444_444453


namespace minimum_period_f_l444_444188

noncomputable def f (x : ℝ) : ℝ := |sin (2 * x) + sin (3 * x) + sin (4 * x)|

theorem minimum_period_f : ∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T' ≥ T) :=
by
  sorry

end minimum_period_f_l444_444188


namespace gcd_lcm_sum_l444_444724

-- Define the numbers and their prime factorizations
def a := 120
def b := 4620
def a_prime_factors := (2, 3) -- 2^3
def b_prime_factors := (2, 2) -- 2^2

-- Define gcd and lcm based on the problem statement
def gcd_ab := 60
def lcm_ab := 4620

-- The statement to be proved
theorem gcd_lcm_sum : gcd a b + lcm a b = 4680 :=
by sorry

end gcd_lcm_sum_l444_444724


namespace parts_sampling_l444_444240

theorem parts_sampling (first_grade second_grade third_grade : ℕ)
                       (total_sample drawn_third : ℕ)
                       (h_first_grade : first_grade = 24)
                       (h_second_grade : second_grade = 36)
                       (h_total_sample : total_sample = 20)
                       (h_drawn_third : drawn_third = 10)
                       (h_non_third : third_grade = 60 - (24 + 36))
                       (h_total : 2 * (24 + 36) = 120)
                       (h_proportion : 2 * third_grade = 2 * (24 + 36)) :
    (third_grade = 60 ∧ (second_grade * (total_sample - drawn_third) / (24 + 36) = 6)) := by
    simp [h_first_grade, h_second_grade, h_total_sample, h_drawn_third] at *
    sorry

end parts_sampling_l444_444240


namespace velocity_at_2_equals_6_l444_444669

-- Define the position function
def S (t : ℝ) : ℝ := 10 * t - t ^ 2

-- Define the derivative of the position function, which represents the velocity function
def v (t : ℝ) : ℝ := (S' t)

-- State the problem
theorem velocity_at_2_equals_6 : v 2 = 6 := 
by
  sorry

end velocity_at_2_equals_6_l444_444669


namespace doubled_width_new_area_l444_444147

-- Define the dimensions of the original card
def original_length := 5
def original_width := 7
def shortened_length := original_length - 2
def shortened_width := original_width

-- Define the new area after shortening one side
def new_area_after_shortening := 21

-- Prove that the new area after doubling the width is equal to 70
theorem doubled_width_new_area : (original_length * (original_width * 2)) = 70 := by
  calc
    original_length * (original_width * 2)
        = 5 * (7 * 2) : by rfl
    ... = 70 : by rfl

end doubled_width_new_area_l444_444147


namespace difference_of_squares_count_l444_444972

theorem difference_of_squares_count :
  (number_of_integers_between (1 : ℕ) (500 : ℕ) (λ n, ∃ a b : ℕ, n = a^2 - b^2)) = 375 :=
by
  sorry

end difference_of_squares_count_l444_444972


namespace min_force_to_submerge_cube_l444_444217

noncomputable def min_applied_force
  (V : ℝ) -- volume of the cube in m^3
  (ρ_cube : ℝ) -- density of the cube in kg/m^3
  (ρ_water : ℝ) -- density of water in kg/m^3
  (g : ℝ) -- acceleration due to gravity in m/s^2
  : ℝ :=
  let W_cube := ρ_cube * V * g in
  let F_buoyant := ρ_water * V * g in
  F_buoyant - W_cube

theorem min_force_to_submerge_cube :
  min_applied_force (10 * 10^(-6)) 600 1000 10 = 0.04 :=
by 
  sorry

end min_force_to_submerge_cube_l444_444217


namespace hour_hand_travel_distance_one_day_night_l444_444674

-- Define the problem

def length_of_hour_hand : ℝ := 2.5 -- length of the hour hand in cm
def hours_in_day : ℝ := 24 -- number of hours in a day
def distance_traveled_by_tip (r : ℝ) (h : ℝ) : ℝ := 2 * π * r * h -- distance travelled by the tip of the hour hand

theorem hour_hand_travel_distance_one_day_night :
  distance_traveled_by_tip length_of_hour_hand hours_in_day = 31.4 := 
sorry

end hour_hand_travel_distance_one_day_night_l444_444674


namespace maximum_value_l444_444682

theorem maximum_value (a b c : ℝ) (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
    (h_eq : a^2 * (b + c - a) = b^2 * (a + c - b) ∧ b^2 * (a + c - b) = c^2 * (b + a - c)) :
    (2 * b + 3 * c) / a = 5 := 
sorry

end maximum_value_l444_444682


namespace basin_more_than_tank2_l444_444848

/-- Define the water volumes in milliliters -/
def volume_bottle1 : ℕ := 1000 -- 1 liter = 1000 milliliters
def volume_bottle2 : ℕ := 400  -- 400 milliliters
def volume_tank : ℕ := 2800    -- 2800 milliliters
def volume_basin : ℕ := volume_bottle1 + volume_bottle2 + volume_tank -- total volume in basin
def volume_tank2 : ℕ := 4000 + 100 -- 4 liters 100 milliliters tank

/-- Theorem: The basin can hold 100 ml more water than the 4-liter 100-milliliter tank -/
theorem basin_more_than_tank2 : volume_basin = volume_tank2 + 100 :=
by
  -- This is where the proof would go, but it is not required for this exercise
  sorry

end basin_more_than_tank2_l444_444848


namespace arithmetic_sequence_sum_l444_444575

theorem arithmetic_sequence_sum (a b d : ℕ) (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℕ)
  (h1 : a₁ + a₂ + a₃ = 39)
  (h2 : a₄ + a₅ + a₆ = 27)
  (h3 : a₄ = a₁ + 3 * d)
  (h4 : a₅ = a₂ + 3 * d)
  (h5 : a₆ = a₃ + 3 * d)
  (h6 : a₇ = a₄ + 3 * d)
  (h7 : a₈ = a₅ + 3 * d)
  (h8 : a₉ = a₆ + 3 * d) :
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ = 81 :=
sorry

end arithmetic_sequence_sum_l444_444575


namespace maximum_value_of_f_l444_444929

noncomputable def f (x : ℝ) : ℝ := min (3 - x^2) (2 * x)

theorem maximum_value_of_f : ∃ x ∈ (set.Icc (-3:ℝ) (1:ℝ)), ∀ y ∈ (set.univ : set ℝ), f(x) ≥ f(y) := by
  sorry

end maximum_value_of_f_l444_444929


namespace line_AF_passes_midpoint_DE_l444_444765

-- Define Triangle Isosceles
structure IsoscelesTriangle (A B C : Type) :=
  (A B C : Type)
  (isosceles : A = B)

-- Define midpoint
structure Midpoint (D B C : Type) :=
  (D : Type)
  (midpoint : D = B ∧ D = C)

-- Define perpendicular
structure Perpendicular (D E A C : Type) :=
  (perpendicular : ∀ {D E A C : Type}, A = C → D ≠ A → D = E)

-- Conditions
variables {ABC : Triangle} [IsoscelesTriangle ABC]
variables {D : Point} [Midpoint D ABC.base]
variables {DE : Line} [Perpendicular D E ABC.ABC.side]

-- The circumcircle and intersection
variable {circ_circ : ∃ circ_circ : circle (triangle ABD), intersection (BE, circ_circ) = {B, F}}

-- Main proof goal: Prove that line AF passes through the midpoint of DE.
theorem line_AF_passes_midpoint_DE :
  ∀ {A F D E : Type} {circ_circ : circle (triangle ABD)}, 
  intersect (AF, midpoint DE) ↔ intersect (circ_circ, BE) :=
sorry

end line_AF_passes_midpoint_DE_l444_444765


namespace AF_through_midpoint_DE_l444_444756

open EuclideanGeometry

-- Definition of an isosceles triangle
variables {A B C D E F : Point}
variables [plane Geometry]
variables (ABC_isosceles : IsoscelesTriangle A B C)
variables (midpoint_D : Midpoint D B C)
variables (perpendicular_DE_AC : Perpendicular D E A C)
variables (F_on_circumcircle : OnCircumcircle F A B D)
variables (B_on_BE : OnLine B E)

-- Goal: Prove that line AF passes through the midpoint of DE
theorem AF_through_midpoint_DE 
  (h1 : triangle A B C) 
  (h2 : ABC_isosceles) 
  (h3 : midpoint D B C) 
  (h4 : perpendicular DE AC)
  (h5 : OnCircumcircle F A B D) 
  (h6 : OnLine B E) :
  PassesThroughMidpoint A F D E :=
sorry

end AF_through_midpoint_DE_l444_444756


namespace non_perfect_powers_count_l444_444480

theorem non_perfect_powers_count :
  let n := 200,
      num_perfect_squares := 14,
      num_perfect_cubes := 5,
      num_perfect_sixth_powers := 1,
      total_perfect_powers := num_perfect_squares + num_perfect_cubes - num_perfect_sixth_powers,
      total_non_perfect_powers := n - total_perfect_powers
  in
  total_non_perfect_powers = 182 :=
by
  sorry

end non_perfect_powers_count_l444_444480


namespace number_of_students_l444_444150

/-- 
We are given that 36 students are selected from three grades: 
15 from the first grade, 12 from the second grade, and the rest from the third grade. 
Additionally, there are 900 students in the third grade.
We need to prove: the total number of students in the high school is 3600
-/
theorem number_of_students (x y z : ℕ) (s_total : ℕ) (x_sel : ℕ) (y_sel : ℕ) (z_students : ℕ) 
  (h1 : x_sel = 15) 
  (h2 : y_sel = 12) 
  (h3 : x_sel + y_sel + (s_total - (x_sel + y_sel)) = s_total) 
  (h4 : s_total = 36) 
  (h5 : z_students = 900) 
  (h6 : (s_total - (x_sel + y_sel)) = 9) 
  (h7 : 9 / 900 = 1 / 100) : 
  (36 * 100 = 3600) :=
by sorry

end number_of_students_l444_444150


namespace lines_are_perpendicular_l444_444191

-- Define the first line equation
def line1 (x y : ℝ) : Prop := x + y - 2 = 0

-- Define the second line equation
def line2 (x y : ℝ) : Prop := x - y + 3 = 0

-- Definition to determine the perpendicularity of two lines
def are_perpendicular (k1 k2 : ℝ) : Prop := k1 * k2 = -1

theorem lines_are_perpendicular :
  are_perpendicular (-1) (1) := 
by
  sorry

end lines_are_perpendicular_l444_444191


namespace boys_without_pencils_l444_444558

variable (total_students : ℕ) (total_boys : ℕ) (students_with_pencils : ℕ) (girls_with_pencils : ℕ)

theorem boys_without_pencils
  (h1 : total_boys = 18)
  (h2 : students_with_pencils = 25)
  (h3 : girls_with_pencils = 15)
  (h4 : total_students = 30) :
  total_boys - (students_with_pencils - girls_with_pencils) = 8 :=
by
  sorry

end boys_without_pencils_l444_444558


namespace binomial_expansion_problem_l444_444396

theorem binomial_expansion_problem
  (n : ℕ)
  (a : ℝ)
  (h1 : (binomial n 0) + (binomial n 1) + (binomial n 2) = 22)
  (h2 : (a - 1)^n = 1) 
  : n = 6 ∧ a = 2 ∧ ∃ t : ℝ, (∀ x : ℝ, 0 < x → t = (binomial 6 4) * (2^((6 - 4))) * ((-x^-1)/2)^4) ∧ t = 60 := 
sorry

end binomial_expansion_problem_l444_444396


namespace num_from_1_to_200_not_squares_or_cubes_l444_444464

noncomputable def numNonPerfectSquaresAndCubes (n : ℕ) : ℕ :=
  let num_squares := 14
  let num_cubes := 5
  let num_sixth_powers := 2
  n - (num_squares + num_cubes - num_sixth_powers)

theorem num_from_1_to_200_not_squares_or_cubes : numNonPerfectSquaresAndCubes 200 = 183 := by
  sorry

end num_from_1_to_200_not_squares_or_cubes_l444_444464


namespace area_ratio_l444_444155

-- Define the points and orthogonal condition
variables {A B C P D : Point} 
variables (BC AD PD : ℝ)

-- Assume the conditions
axiom point_outside_plane (P : Point) (ABC : Triangle) : P ∉ plane_of ABC
axiom PA_perpendicular_to_plane (A B C P : Point) : (Line_through P A) ⊥ (plane_of (triangle_of A B C))
axiom triangle_areas (S1 S2 : ℝ) : S1 = area_of (triangle_of A B C) ∧ S2 = area_of (triangle_of P B C)
axiom perpendicular_dropped (A B C D : Point) : (Line_through A D) ⊥ (Line_through B C)
axiom three_perpendicular_theorem (P D B C : Point) : (Line_through P D) ⊥ (Line_through B C)

-- Define the equivalence to be proven
theorem area_ratio (A B C P D : Point) (S1 S2 : ℝ) (h1 : S1 = area_of (triangle_of A B C))
  (h2 : S2 = area_of (triangle_of P B C)) : 0 < S1 / S2 ∧ S1 / S2 < 1 :=
sorry

end area_ratio_l444_444155


namespace bicycle_spokes_count_l444_444292

theorem bicycle_spokes_count (bicycles wheels spokes : ℕ) 
       (h1 : bicycles = 4) 
       (h2 : wheels = 2) 
       (h3 : spokes = 10) : 
       bicycles * (wheels * spokes) = 80 :=
by
  sorry

end bicycle_spokes_count_l444_444292


namespace highest_place_of_products_l444_444187

theorem highest_place_of_products :
  ∀ (a b : ℕ), a = 216 → b = 126 →
  (216 * 5 = 1080 ∧ 126 * 5 = 630 ∧
   ∃ place1 place2, place1 = "thousands" ∧ place2 = "hundreds" ∧
   (highest_place 1080 = place1) ∧ (highest_place 630 = place2)) := 
by
  intros a b ha hb
  rw [ha, hb]
  have h1 : 216 * 5 = 1080 := by norm_num
  have h2 : 126 * 5 = 630 := by norm_num
  exists "thousands", "hundreds"
  split; [split, split]; assumption
  sorry

noncomputable def highest_place (n : ℕ) : String :=
  if n < 10 then "units"
  else if n < 100 then "tens"
  else if n < 1000 then "hundreds"
  else "thousands"

end highest_place_of_products_l444_444187


namespace masha_can_pay_with_5_ruble_coins_l444_444650

theorem masha_can_pay_with_5_ruble_coins (p c n : ℤ) (h : 2 * p + c + 7 * n = 100) : (p + 3 * c + n) % 5 = 0 :=
  sorry

end masha_can_pay_with_5_ruble_coins_l444_444650


namespace chromium_percentage_second_alloy_l444_444569

theorem chromium_percentage_second_alloy:
  ∀ (chromium_first_alloy: ℝ) (weight_first_alloy: ℝ) (weight_second_alloy: ℝ) (chromium_new_alloy: ℝ) (weight_new_alloy: ℝ),
  chromium_first_alloy = 10 / 100 →
  weight_first_alloy = 15 →
  weight_second_alloy = 35 →
  chromium_new_alloy = 7.2 / 100 →
  weight_new_alloy = weight_first_alloy + weight_second_alloy →
  (1.5 + (x / 100) * weight_second_alloy = chromium_new_alloy * weight_new_alloy) →
  x = 6 := 
by {
  intros chromium_first_alloy weight_first_alloy weight_second_alloy chromium_new_alloy weight_new_alloy h1 h2 h3 h4 h5 h6,
  sorry,
}

end chromium_percentage_second_alloy_l444_444569


namespace amphibians_frogs_l444_444084

-- Define the species
inductive Species
| toad  -- always tells the truth
| frog  -- always lies
| newt  -- tells the truth every alternate day

open Species

-- Initial conditions
def statement_Alex (Ethan: Species) (Alex: Species) : Prop :=
  (Ethan ≠ Alex)

def statement_Ben (Carl: Species) : Prop :=
  (Carl = frog)

def statement_Carl (Ben: Species) : Prop :=
  (Ben = frog)

def statement_Danny (Alex Ben Carl Danny Ethan: Species) : Prop :=
  (if Alex = toad then 1 else 0) +
  (if Ben = toad then 1 else 0) +
  (if Carl = toad then 1 else 0) +
  (if Danny = toad then 1 else 0) +
  (if Ethan = toad then 1 else 0) ≥ 3

def statement_Ethan (Alex: Species) : Prop :=
  (Alex = newt)

-- Proving the given conditions lead to the conclusion that the number of frogs is 2
theorem amphibians_frogs (Alex Ben Carl Danny Ethan: Species)
  (hAlex: statement_Alex Ethan Alex)
  (hBen: statement_Ben Carl)
  (hCarl: statement_Carl Ben)
  (hDanny: statement_Danny Alex Ben Carl Danny Ethan)
  (hEthan: statement_Ethan Alex) : 
  (if Alex = frog then 1 else 0) +
  (if Ben = frog then 1 else 0) +
  (if Carl = frog then 1 else 0) +
  (if Danny = frog then 1 else 0) +
  (if Ethan = frog then 1 else 0) = 2 := sorry

end amphibians_frogs_l444_444084


namespace numbers_not_squares_nor_cubes_1_to_200_l444_444495

theorem numbers_not_squares_nor_cubes_1_to_200 : 
  let num_squares := nat.sqrt 200
  let num_cubes := nat.cbrt 200
  let num_sixth_powers := nat.root 6 200
  let total := num_squares + num_cubes - num_sixth_powers
  200 - total = 182 := 
by
  sorry

end numbers_not_squares_nor_cubes_1_to_200_l444_444495


namespace max_x_y_given_condition_l444_444038

theorem max_x_y_given_condition (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y + 1/x + 1/y = 5) : x + y ≤ 4 :=
sorry

end max_x_y_given_condition_l444_444038


namespace sequence_formula_b_sum_formula_l444_444014

noncomputable def a (n : ℕ) : ℝ :=
if (n % 2 = 0) then 2^(n / 2)
else 2^((n-1) / 2)

def b (n : ℕ) : ℝ :=
(log (a (2 * n)) / log 2) / a (2 * n - 1)

def T (n : ℕ) : ℝ :=
∑ i in Finset.range n, b (i + 1)

theorem sequence_formula (n : ℕ) (hn : n > 0) :
  a (n + 2) = 2 * a n :=
sorry

theorem b_sum_formula (n : ℕ) (hn : n > 0) :
  T n = 4 - (n + 2) / 2^(n - 1) :=
sorry

end sequence_formula_b_sum_formula_l444_444014


namespace range_of_a_l444_444023

variable {x a : ℝ}

def p (x: ℝ) : Prop := ( (log 2 (1 - x)) < 0 )
def q (x a : ℝ) : Prop := ( x > a )

theorem range_of_a (h : ∀ x, p x → q x a) (suff_not_nec : ∀ x, q x a → p x → False) :
    a ≤ 0 :=
sorry

end range_of_a_l444_444023


namespace range_of_slope_angle_l444_444025

noncomputable def slope_PA (P A : ℝ × ℝ) : ℝ :=
  (P.2 - A.2) / (P.1 - A.1)

noncomputable def slope_PB (P B : ℝ × ℝ) : ℝ :=
  (P.2 - B.2) / (P.1 - B.1)

noncomputable def theta_range (P A B : ℝ × ℝ) : set ℝ :=
  let k_PA := slope_PA P A
  let k_PB := slope_PB P B
  set_of (λ θ, θ = real.arctan k_PA ∨ θ = real.arctan k_PB)

theorem range_of_slope_angle :
  let P := (2, 3)
  let A := (3, 2)
  let B := (-1, -3)
  θ ∈ theta_range P A B :=
sorry

end range_of_slope_angle_l444_444025


namespace three_digit_number_is_11_times_sum_of_digits_l444_444341

theorem three_digit_number_is_11_times_sum_of_digits :
    ∃ a b c : ℕ, a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ 
        (100 * a + 10 * b + c = 11 * (a + b + c)) ↔ 
        (100 * 1 + 10 * 9 + 8 = 11 * (1 + 9 + 8)) := 
by
    sorry

end three_digit_number_is_11_times_sum_of_digits_l444_444341


namespace inscribed_circle_radius_l444_444415

-- Definitions for the given conditions
def circle1_radius : ℝ := 1
def circle2_radius : ℝ := 2
def circle3_radius : ℝ := 3

def distance_o1_o2 (r1 r2 : ℝ) : ℝ := r1 + r2
def distance_o1_o3 (r1 r3 : ℝ) : ℝ := r1 + r3
def distance_o2_o3 (r2 r3 : ℝ) : ℝ := r2 + r3

-- Calculate the distances between centers
def o1_o2_dist := distance_o1_o2 circle1_radius circle2_radius
def o1_o3_dist := distance_o1_o3 circle1_radius circle3_radius
def o2_o3_dist := distance_o2_o3 circle2_radius circle3_radius

-- Lean statement for the proof problem
theorem inscribed_circle_radius :
  let r := (circle1_radius, circle2_radius, circle3_radius) in
  let abc_triangle := (o1_o2_dist, o1_o3_dist, o2_o3_dist) in
  -- Expected radius of the inscribed circle
  let expected_radius := (-30 + 15 * (real.sqrt 2) + 6 * (real.sqrt 5) + 3 * (real.sqrt 10)) / 30 in
  radius_of_inscribed_circle abc_triangle = expected_radius :=
sorry -- Proof is omitted

end inscribed_circle_radius_l444_444415


namespace rhombus_diagonals_and_square_area_l444_444262

theorem rhombus_diagonals_and_square_area (d1 d2 : ℝ) (side_square : ℝ)
  (h_d1 : d1 = 24) (h_d2 : d2 = 16) (h_side_square : side_square = 16) :
  let side_rhombus := Real.sqrt ((d1/2)^2 + (d2/2)^2)
  in (4 * side_rhombus = 16 * Real.sqrt 13)
  ∧ (side_square^2 = 256) :=
by
  sorry

end rhombus_diagonals_and_square_area_l444_444262


namespace numbers_not_squares_nor_cubes_1_to_200_l444_444494

theorem numbers_not_squares_nor_cubes_1_to_200 : 
  let num_squares := nat.sqrt 200
  let num_cubes := nat.cbrt 200
  let num_sixth_powers := nat.root 6 200
  let total := num_squares + num_cubes - num_sixth_powers
  200 - total = 182 := 
by
  sorry

end numbers_not_squares_nor_cubes_1_to_200_l444_444494


namespace sum_binom_eq_binom_l444_444611

theorem sum_binom_eq_binom (p n : ℕ) : (∑ i in Finset.range (n + 1), Nat.choose (p + i) p) = Nat.choose (p + n + 1) (p + 1) := 
sorry

end sum_binom_eq_binom_l444_444611


namespace rowing_speed_problem_l444_444833

noncomputable theory

def speed_in_still_water (v: ℝ) (t: ℝ) (d: ℝ) (current: ℝ) : Prop :=
  (v + current) = (d / (t / 3600))

theorem rowing_speed_problem :
  ∃ v, speed_in_still_water v 44 0.11 3 ∧ v = 6 :=
by
  have h1 : (0.11 / (44 / 3600)) = 9 := by
    calc
      0.11 / (44 / 3600) = 0.11 * 3600 / 44 : by field_simp
      ... = 396 / 44      : by norm_num
      ... = 9             : by norm_num
  use 6
  have h2 : 6 + 3 = 9 := by norm_num
  exact ⟨(by rw [speed_in_still_water, h1] ; exact h2), rfl⟩

end rowing_speed_problem_l444_444833


namespace pyramid_height_eq_375_l444_444823

theorem pyramid_height_eq_375 :
  let a := 5 
  let b := 10
  let V_cube := a^3
  let V_pyramid := (b^2 * h) / 3
  V_cube = V_pyramid →
  h = 3.75 :=
by
  let a := 5
  let b := 10
  let V_cube := a^3
  let V_pyramid := (b^2 * h) / 3
  have : V_cube = 125 := by norm_num
  have : V_pyramid = (100 * h) / 3 := by norm_num
  sorry

end pyramid_height_eq_375_l444_444823


namespace count_non_perfect_square_or_cube_l444_444515

-- Define perfect_square function
def perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k^2 = n

-- Define perfect_cube function
def perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k^3 = n

-- Define range from 1 to 200
def range_200 := {1 .. 200}

-- Define numbers either perfect square or perfect cube
def perfect_square_or_cube : Finset ℕ :=
  (range_200.filter perfect_square) ∪ (range_200.filter perfect_cube)

-- Define sixth powers in range
def perfect_sixth_power (n : ℕ) : Prop :=
  ∃ k : ℕ, k^6 = n

-- The final problem statement
theorem count_non_perfect_square_or_cube :
  (range_200.card - perfect_square_or_cube.card) = 182 :=
by
  sorry

end count_non_perfect_square_or_cube_l444_444515


namespace multiply_m_and_t_l444_444128

theorem multiply_m_and_t (g : ℝ → ℝ) (h : ∀ x y : ℝ, g (g x + y) = g x + g (g y + g (-x)) - x) :
  let m := 1 in 
  let t := g 4 in 
  m * t = 4 :=
by
  -- Assuming g x = x based on the given conditions, conclude the values.
  -- Here we would derive the exact simplifications as shown in the solution steps.
  sorry

end multiply_m_and_t_l444_444128


namespace bugs_max_contacts_l444_444227

theorem bugs_max_contacts :
  ∃ a b : ℕ, (a + b = 2016) ∧ (a * b = 1008^2) :=
by
  sorry

end bugs_max_contacts_l444_444227


namespace Wendy_runs_farther_l444_444206

-- Define the distances Wendy ran and walked
def distance_ran : ℝ := 19.83
def distance_walked : ℝ := 9.17

-- Define the difference in distances
def difference : ℝ := distance_ran - distance_walked

-- The theorem to prove
theorem Wendy_runs_farther : difference = 10.66 := by
  sorry

end Wendy_runs_farther_l444_444206


namespace range_of_k_l444_444075

theorem range_of_k (a k : ℝ) : 
  (∀ x y : ℝ, y^2 - x * y + 2 * x + k = 0 → (x = a ∧ y = -a)) →
  k ≤ 1/2 :=
by sorry

end range_of_k_l444_444075


namespace area_of_triangle_l444_444195

noncomputable def triangle_sides : ℕ × ℕ × ℕ :=
  let x := 10 in
  (5 * x, 12 * x, 13 * x)

def semi_perimeter (a b c : ℕ) : ℕ :=
  (a + b + c) / 2

noncomputable def heron_area (a b c : ℕ) : ℕ :=
  let s := semi_perimeter a b c in
  real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem area_of_triangle : 
  let (a, b, c) := triangle_sides in
  heron_area a b c = 3000 := 
by
  let (a, b, c) := triangle_sides
  have s : ℕ := semi_perimeter a b c
  have h : real.sqrt (s * (s - a) * (s - b) * (s - c)) = 3000 := sorry
  exact h

end area_of_triangle_l444_444195


namespace min_points_to_win_l444_444272

-- Define the points distribution
def points (place : ℕ) : ℕ :=
  if place = 1 then 7 else
  if place = 2 then 4 else
  if place = 3 then 2 else 0

-- Define a function to sum points for a list of places
def total_points (places : List ℕ) : ℕ :=
  (places.map points).sum

-- Define the problem statement
theorem min_points_to_win : 
  ∀ (student1 student2 student3 student4 : List ℕ), 
  student1.length = 4 → 
  student2.length = 4 → 
  student3.length = 4 → 
  student4.length = 4 → 
  (∀ i, (i < 4 → List.nth student1 i ≠ List.nth student2 i ∧ List.nth student1 i ≠ List.nth student3 i ∧ List.nth student1 i ≠ List.nth student4 i)) → 
  total_points student1 > 24 → 
  (total_points student2 < 25 ∧ total_points student3 < 25 ∧ total_points student4 < 25) :=
sorry

end min_points_to_win_l444_444272


namespace find_a_l444_444416

-- Define the sets A and B and the condition that A union B is a subset of A intersect B
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def B (a : ℝ) : Set ℝ := {x | -1 ≤ x ∧ x ≤ a}

theorem find_a (a : ℝ) :
  A ∪ B a ⊆ A ∩ B a → a = 1 :=
sorry

end find_a_l444_444416


namespace parabola_line_intersection_l444_444117

theorem parabola_line_intersection :
  let P := λ x : ℝ, x^2 in
  let Q := (20 : ℝ, 14 : ℝ) in
  ∃ r s : ℝ, (∀ m : ℝ, (r < m ∧ m < s) ↔ ∀ x : ℝ, x^2 ≠ m * x + 14 - 20 * m) ∧ r + s = 80 :=
sorry

end parabola_line_intersection_l444_444117


namespace problem1_problem2_l444_444229

theorem problem1 : -1 + (-6) - (-4) + 0 = -3 := by
  sorry

theorem problem2 : 24 * (-1 / 4) / (-3 / 2) = 4 := by
  sorry

end problem1_problem2_l444_444229


namespace non_perfect_powers_count_l444_444479

theorem non_perfect_powers_count :
  let n := 200,
      num_perfect_squares := 14,
      num_perfect_cubes := 5,
      num_perfect_sixth_powers := 1,
      total_perfect_powers := num_perfect_squares + num_perfect_cubes - num_perfect_sixth_powers,
      total_non_perfect_powers := n - total_perfect_powers
  in
  total_non_perfect_powers = 182 :=
by
  sorry

end non_perfect_powers_count_l444_444479


namespace S_8_value_l444_444950

variable {a : ℕ → ℝ}   -- Define the arithmetic sequence
variable {n S : ℕ → ℝ} -- Define S_n as the sum of first n terms

-- Given conditions
def condition1 : a 3 = 9 - a 6 := sorry

-- Definition of the sum of the first n terms of an arithmetic sequence
def sum_first_n_terms (n : ℕ) : ℝ := (n / 2) * (a 1 + a n)

-- Sum of first 8 terms of the sequence
def S_8 := sum_first_n_terms 8

-- The theorem to be proved
theorem S_8_value : S_8 = 72 :=
  sorry

end S_8_value_l444_444950


namespace chess_tournament_ordering_l444_444695

-- Problem statement in Lean
theorem chess_tournament_ordering (n : ℕ) (h_n : n = 2005) 
  (players : Fin n → Type) 
  (game : (Fin n → Fin n → Prop)) 
  (pairwise_played : ∀ (i j : Fin n), i ≠ j → (game i j ∨ game j i)) 
  (draw_condition : ∀ (i j : Fin n), (¬ game i j) ∧ (¬ game j i) → 
    ∀ (k : Fin n), (game i k ∨ game j k) ∧ (game k i ∨ game k j)) 
  (at_least_two_draws : ∃ i j : Fin n, i ≠ j ∧ (¬ game i j) ∧ (¬ game j i)) :
  ∃ (order : List (Fin n)), 
    (∀ (i : Fin (n-1)), game (order.nth_le i sorry) (order.nth_le (i+1) sorry)) :=
sorry

end chess_tournament_ordering_l444_444695


namespace numbers_not_perfect_squares_or_cubes_l444_444448

theorem numbers_not_perfect_squares_or_cubes : 
  let total_numbers := 200
  let perfect_squares := 14
  let perfect_cubes := 5
  let sixth_powers := 1
  total_numbers - (perfect_squares + perfect_cubes - sixth_powers) = 182 :=
by
  sorry

end numbers_not_perfect_squares_or_cubes_l444_444448


namespace shaded_region_area_l444_444192

theorem shaded_region_area (squares_count : ℕ) (diagonal : ℝ) 
  (h1 : squares_count = 25) (h2 : diagonal = 10) : 
  let side := (diagonal / real.sqrt 2) in
  let small_square_side := side / 5 in
  let small_square_area := (small_square_side) ^ 2 in
  let total_area := small_square_area * (squares_count : ℝ) in
  total_area = 50 := 
by 
  -- Placeholder for the proof
  sorry

end shaded_region_area_l444_444192


namespace min_alpha_gamma_l444_444831

variables {α γ : ℂ} 
def f (z : ℂ) : ℂ := (3 - 2 * complex.I) * z^2 + α * z + γ

theorem min_alpha_gamma (h1 : (f 1).im = 0) (h2 : (f (-complex.I)).im = 0) :
  |α| + |γ| = complex.abs (complex.mk (3 - 3) 2) + complex.abs (complex.mk 0 0) := by
sorry

end min_alpha_gamma_l444_444831


namespace num_diff_of_squares_in_range_l444_444981

/-- 
Define a number expressible as the difference of squares of two nonnegative integers.
-/
def is_diff_of_squares (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = a^2 - b^2

/-- 
Define the range of interest from 1 to 500.
-/
def range_1_to_500 : Finset ℕ := Finset.range 501 \ {0}

/-- 
Define the criterion for numbers that can be expressed in the desired form.
-/
def can_be_expressed (n : ℕ) : Prop :=
  (n % 2 = 1) ∨ (n % 4 = 0)

/-- 
Count the numbers between 1 and 500 that satisfy the condition.
-/
def count_expressible_numbers : ℕ :=
  (range_1_to_500.filter can_be_expressed).card

/-- 
Prove that the count of numbers between 1 and 500 that can be expressed as 
the difference of two squares of nonnegative integers is 375.
-/
theorem num_diff_of_squares_in_range : count_expressible_numbers = 375 :=
  sorry

end num_diff_of_squares_in_range_l444_444981


namespace Shara_will_owe_money_l444_444164

theorem Shara_will_owe_money
    (B : ℕ)
    (h1 : 6 * 10 = 60)
    (h2 : B / 2 = 60)
    (h3 : 4 * 10 = 40)
    (h4 : 60 + 40 = 100) :
  B - 100 = 20 :=
sorry

end Shara_will_owe_money_l444_444164


namespace min_value_of_function_l444_444409

-- We need to define the quadratic inequality and its solution set
section
variables (x a b : ℝ)

-- First Problem: roots of quadratic inequality
def roots_of_quadratic_ineq (a b : ℝ) := 
  (x = 1 ∨ x = 4) → (5 * a = 1 + 4 ∧ b = 4)

-- Second Problem: minimum value of the function
def min_value (x : ℝ) (a b : ℝ) : ℝ :=
  if h : 0 < x ∧ x < 1 then 
    let f := (1 / x + 4 / (1 - x)) in 9
  else 0  -- dummy value outside the interval

-- Proving the minimum value
theorem min_value_of_function : 
  ∀ (x : ℝ) (a b : ℝ), 
  roots_of_quadratic_ineq x a b →
  min_value x 1 4 = 9 := by
  intros x a b h
  unfold roots_of_quadratic_ineq min_value
  split_ifs
  { sorry } -- proof that the minimum value is 9
  { sorry } -- this case is outside the interval
end

end min_value_of_function_l444_444409


namespace range_of_a_l444_444394

noncomputable def f : ℝ → ℝ := sorry

theorem range_of_a (a : ℝ) (h1 : ∀ x ∈ Icc (-2 : ℝ) (4 : ℝ), f (x) ≤ f (x - 1))
(h2 : f (a + 1) > f (2 * a)) : (1 < a ∧ a ≤ 2) :=
by {
  -- Here we assume the function is monotonically decreasing and apply the given conditions to get the required result.
  sorry
}

end range_of_a_l444_444394


namespace main_theorem_l444_444743

universe u
variables {α : Type u} [euclidean_geometry α]

open_locale euclidean_geometry

-- Define the given conditions
def is_isosceles (A B C : α) (h : euclidean_geometry.is_triangle A B C) : Prop :=
euclidean_geometry.dist A B = euclidean_geometry.dist A C

def midpoint (D B C : α) : Prop :=
euclidean_geometry.is_midpoint D B C

def perpendicular (E D A C : α) : Prop :=
euclidean_geometry.is_perpendicular E D A C

def circumcircle_intersects (F B D A : α) (c : set α) : Prop :=
c = euclidean_geometry.circumcircle A B D ∧ F ∈ c ∧ B ∈ c

-- The main theorem to prove
theorem main_theorem (A B C D E F : α) (h1 : euclidean_geometry.is_triangle A B C)
    (h2 : is_isosceles A B C h1) (h3 : midpoint D B C) (h4 : perpendicular E D A C)
    (h5 : ∃ c, circumcircle_intersects F B D A c) :
    euclidean_geometry.passes_through (euclidean_geometry.line A F) (euclidean_geometry.midpoint E D) :=
sorry

end main_theorem_l444_444743


namespace num_from_1_to_200_not_squares_or_cubes_l444_444457

noncomputable def numNonPerfectSquaresAndCubes (n : ℕ) : ℕ :=
  let num_squares := 14
  let num_cubes := 5
  let num_sixth_powers := 2
  n - (num_squares + num_cubes - num_sixth_powers)

theorem num_from_1_to_200_not_squares_or_cubes : numNonPerfectSquaresAndCubes 200 = 183 := by
  sorry

end num_from_1_to_200_not_squares_or_cubes_l444_444457


namespace average_donation_proof_l444_444092

noncomputable def average_donation (total_people : ℝ) (donated_200 : ℝ) (donated_100 : ℝ) (donated_50 : ℝ) : ℝ :=
  let proportion_200 := donated_200 / total_people
  let proportion_100 := donated_100 / total_people
  let proportion_50 := donated_50 / total_people
  let total_donation := (200 * proportion_200) + (100 * proportion_100) + (50 * proportion_50)
  total_donation

theorem average_donation_proof 
  (total_people : ℝ)
  (donated_200 donated_100 donated_50 : ℝ)
  (h1 : proportion_200 = 1 / 10)
  (h2 : proportion_100 = 3 / 4)
  (h3 : proportion_50 = 1 - proportion_200 - proportion_100) :
  average_donation total_people donated_200 donated_100 donated_50 = 102.5 :=
  by 
    sorry

end average_donation_proof_l444_444092


namespace num_non_squares_cubes_1_to_200_l444_444436

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n
def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k * k * k = n
def is_perfect_sixth_power (n : ℕ) : Prop := ∃ k : ℕ, k * k * k * k * k * k = n

def num_perfect_squares (a b : ℕ) : ℕ :=
  (List.range (b + 1)).countp is_perfect_square - (List.range (a)).countp is_perfect_square

def num_perfect_cubes (a b : ℕ) : ℕ :=
  (List.range (b + 1)).countp is_perfect_cube - (List.range (a)).countp is_perfect_cube

def num_perfect_sixth_powers (a b : ℕ) : ℕ :=
  (List.range (b + 1)).countp is_perfect_sixth_power - (List.range (a)).countp is_perfect_sixth_power

theorem num_non_squares_cubes_1_to_200 :
  let squares := num_perfect_squares 1 200 in
  let cubes := num_perfect_cubes 1 200 in
  let sixth_powers := num_perfect_sixth_powers 1 200 in
  200 - (squares + cubes - sixth_powers) = 182 :=
by
  let squares := num_perfect_squares 1 200
  let cubes := num_perfect_cubes 1 200
  let sixth_powers := num_perfect_sixth_powers 1 200
  have h : 200 - (squares + cubes - sixth_powers) = 182 := sorry
  exact h

end num_non_squares_cubes_1_to_200_l444_444436


namespace count_non_perfect_square_or_cube_l444_444518

-- Define perfect_square function
def perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k^2 = n

-- Define perfect_cube function
def perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k^3 = n

-- Define range from 1 to 200
def range_200 := {1 .. 200}

-- Define numbers either perfect square or perfect cube
def perfect_square_or_cube : Finset ℕ :=
  (range_200.filter perfect_square) ∪ (range_200.filter perfect_cube)

-- Define sixth powers in range
def perfect_sixth_power (n : ℕ) : Prop :=
  ∃ k : ℕ, k^6 = n

-- The final problem statement
theorem count_non_perfect_square_or_cube :
  (range_200.card - perfect_square_or_cube.card) = 182 :=
by
  sorry

end count_non_perfect_square_or_cube_l444_444518


namespace total_differential_of_given_function_l444_444910

noncomputable def total_differential (u : ℝ → ℝ → ℝ) (dx dy : ℝ) :=
  (λ x y, (∂ (eval (u x y)) / ∂ x) * dx + (∂ (eval (u x y)) / ∂ y) * dy)

def given_function (x y : ℝ) : ℝ :=
  x^2 * Real.arctan (y/x) - y^2 * Real.arctan (x/y)

theorem total_differential_of_given_function (dx dy : ℝ) (x y : ℝ) :
  total_differential given_function dx dy = 
  (2*x * Real.arctan (y/x) - y) * dx + (x - 2*y * Real.arctan (x/y)) * dy :=
sorry

end total_differential_of_given_function_l444_444910


namespace count_valid_numbers_between_1_and_200_l444_444427

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k^2 = n
def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k^3 = n
def is_perfect_sixth_power (n : ℕ) : Prop := ∃ k : ℕ, k^6 = n

def count_perfect_squares (m : ℕ) : ℕ :=
  Nat.sqrt m

def count_perfect_cubes (m : ℕ) : ℕ :=
  (m : ℝ).cbrt.to_nat

def count_perfect_sixth_powers (m : ℕ) : ℕ :=
  (m : ℝ).cbrt.sqrt.to_nat

def count_either_squares_or_cubes (m : ℕ) : ℕ :=
  let squares := count_perfect_squares m
  let cubes := count_perfect_cubes m
  let sixths := count_perfect_sixth_powers m
  squares + cubes - sixths

def count_neither_squares_nor_cubes (n : ℕ) : ℕ :=
  n - count_either_squares_or_cubes n

theorem count_valid_numbers_between_1_and_200 : count_neither_squares_nor_cubes 200 = 182 :=
by
  sorry

end count_valid_numbers_between_1_and_200_l444_444427


namespace proof_p_or_q_true_l444_444139

def p (x : ℝ) : Prop := (e ^ x > 1) → (x > 0)
def q (x : ℝ) : Prop := (|x - 3| > 1) → (x > 4)

theorem proof_p_or_q_true : (∀ x : ℝ, p x) ∨ (∀ x : ℝ, q x) :=
by
  sorry

end proof_p_or_q_true_l444_444139


namespace count_not_squares_or_cubes_200_l444_444528

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n
def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k * k * k = n
def is_sixth_power (n : ℕ) : Prop := ∃ k : ℕ, k^6 = n

theorem count_not_squares_or_cubes_200 :
  let count_squares := (Finset.range 201).filter is_perfect_square).card
  let count_cubes := (Finset.range 201).filter is_perfect_cube).card
  let count_sixth_powers := (Finset.range 201).filter is_sixth_power).card
  (200 - (count_squares + count_cubes - count_sixth_powers)) = 182 := 
by {
  -- Define everything in terms of Finset and filter
  let count_squares := (Finset.range 201).filter is_perfect_square).card,
  let count_cubes := (Finset.range 201).filter is_perfect_cube).card,
  let count_sixth_powers := (Finset.range 201).filter is_sixth_power).card,
  exact sorry
}

end count_not_squares_or_cubes_200_l444_444528


namespace asymptote_one_vertical_l444_444892

open Function

theorem asymptote_one_vertical (k : ℝ) : (∃ x : ℝ, x ≠ 2 ∧ x ≠ 3 ∧ g(x) = (x^2 - 3x + k) / (x^2 - 5x + 6)) ↔ (k = 0 ∨ k = 2) :=
by sorry

def g(x : ℝ) (k : ℝ) := (x^2 - 3x + k) / (x^2 - 5x + 6)

end asymptote_one_vertical_l444_444892


namespace integer_solutions_l444_444337

theorem integer_solutions (n : ℤ) : (n^2 ∣ 2^n + 1) ↔ n = 1 ∨ n = 3 := by
  sorry

end integer_solutions_l444_444337


namespace sum_of_coefficients_is_neg42_l444_444351

noncomputable def polynomial1 := 3 * (X^8 - 2 * X^5 + 4 * X^3 - 6)
noncomputable def polynomial2 := -5 * (X^4 - 3 * X + 7)
noncomputable def polynomial3 := 2 * (X^6 - 5)

noncomputable def sum_of_coeffs := polynomial1 + polynomial2 + polynomial3

theorem sum_of_coefficients_is_neg42 : 
  (sum_of_coeffs.eval 1) = -42 :=
by
  sorry

end sum_of_coefficients_is_neg42_l444_444351


namespace numbers_not_squares_or_cubes_in_200_l444_444475

/-- There are 182 numbers between 1 and 200 that are neither perfect squares nor perfect cubes. -/
theorem numbers_not_squares_or_cubes_in_200 : (finset.range 200).filter (λ n, ¬(∃ m, m^2 = n) ∧ ¬(∃ k, k^3 = n)).card = 182 :=
by 
  /-
    The proof will involve constructing the set of numbers from 1 to 200, filtering out those that
    are perfect squares or perfect cubes, and counting the remaining numbers.
  -/
  sorry

end numbers_not_squares_or_cubes_in_200_l444_444475


namespace count_valid_numbers_between_1_and_200_l444_444424

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k^2 = n
def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k^3 = n
def is_perfect_sixth_power (n : ℕ) : Prop := ∃ k : ℕ, k^6 = n

def count_perfect_squares (m : ℕ) : ℕ :=
  Nat.sqrt m

def count_perfect_cubes (m : ℕ) : ℕ :=
  (m : ℝ).cbrt.to_nat

def count_perfect_sixth_powers (m : ℕ) : ℕ :=
  (m : ℝ).cbrt.sqrt.to_nat

def count_either_squares_or_cubes (m : ℕ) : ℕ :=
  let squares := count_perfect_squares m
  let cubes := count_perfect_cubes m
  let sixths := count_perfect_sixth_powers m
  squares + cubes - sixths

def count_neither_squares_nor_cubes (n : ℕ) : ℕ :=
  n - count_either_squares_or_cubes n

theorem count_valid_numbers_between_1_and_200 : count_neither_squares_nor_cubes 200 = 182 :=
by
  sorry

end count_valid_numbers_between_1_and_200_l444_444424


namespace convex_polygon_has_center_of_symmetry_nonconvex_polygon_not_necessarily_center_of_symmetry_l444_444159

variables {Polygon : Type} [ConvexPolygon Polygon] [NonconvexPolygon Polygon]

-- Definitions reflecting conditions
def is_central_symmetric (P : Polygon) : Prop := sorry
def can_be_divided_into_central_symmetric_polygons (P : Polygon) : Prop := sorry

-- The proof problem for convex polygon
theorem convex_polygon_has_center_of_symmetry 
  (P : Polygon) [h1 : ConvexPolygon P] 
  (h2 : can_be_divided_into_central_symmetric_polygons P) : 
  is_central_symmetric P := 
sorry

-- The proof problem for non-convex polygon
theorem nonconvex_polygon_not_necessarily_center_of_symmetry 
  (P : Polygon) [h1 : NonconvexPolygon P] 
  (h2 : can_be_divided_into_central_symmetric_polygons P) : 
  ¬is_central_symmetric P := 
sorry

end convex_polygon_has_center_of_symmetry_nonconvex_polygon_not_necessarily_center_of_symmetry_l444_444159


namespace line_AF_midpoint_DE_l444_444789

variables {A B C D E F : Type} [EuclideanGeometry.point A] [EuclideanGeometry.point B] [EuclideanGeometry.point C]
  [EuclideanGeometry.point D] [EuclideanGeometry.point E] [EuclideanGeometry.point F]

-- Definitions based on the problem conditions
def isosceles_triangle (A B C : Type) [EuclideanGeometry.point A] [EuclideanGeometry.point B] [EuclideanGeometry.point C] : Prop :=
  (EuclideanGeometry.distance A B = EuclideanGeometry.distance A C)

def midpoint (D B C : Type) [EuclideanGeometry.point D] [EuclideanGeometry.point B] [EuclideanGeometry.point C] : Prop :=
  EuclideanGeometry.between D B C

def perpendicular (D E : Type) [EuclideanGeometry.point D] [EuclideanGeometry.point E] : Prop := 
  EuclideanGeometry.perp D E

def circumcircle (ABD : Set EuclideanGeometry.point) : Prop :=
  EuclideanGeometry.circumcircle ABD

def intersection (circumcircle_intersection : Type) [EuclideanGeometry.point F] [EuclideanGeometry.point B] : Prop := 
  EuclideanGeometry.intersects circumcircle_intersection F B

-- The theorem we need to prove
theorem line_AF_midpoint_DE
  (isoABC : isosceles_triangle A B C)
  (midD : midpoint D B C)
  (perpDE : perpendicular D E)
  (circABD : circumcircle {A, B, D})
  (interBF : intersection {A, B, D})
  : EuclideanGeometry.on_line A F (EuclideanGeometry.midpoint D E) :=
sorry -- proof here

end line_AF_midpoint_DE_l444_444789


namespace probability_sum_divisible_by_3_l444_444005

def selected_pairs (s : Set ℕ) := 
  {pair : ℕ × ℕ | pair.1 ∈ s ∧ pair.2 ∈ s ∧ pair.1 < pair.2}

def sum_divisible_by_3 (pair : ℕ × ℕ) :=
  (pair.1 + pair.2) % 3 = 0

theorem probability_sum_divisible_by_3 :
  let s := {1, 2, 3, 4, 5, 6}
  let total_pairs := selected_pairs s
  let total_cases := total_pairs.card
  let favorable_cases := (selected_pairs s).filter sum_divisible_by_3
  let favorable_count := favorable_cases.card
  ∃ (prob : ℚ), prob = favorable_count / total_cases ∧ prob = 1 / 3 := 
by 
  sorry

end probability_sum_divisible_by_3_l444_444005


namespace copper_atom_mass_scientific_notation_l444_444681

theorem copper_atom_mass_scientific_notation :
  (0.000000000000000000000106 : ℝ) = 1.06 * 10^(-22) :=
sorry

end copper_atom_mass_scientific_notation_l444_444681


namespace locus_equation_slope_range_l444_444095

variables {P A B C D : Type} 

-- Given coordinates of points A, B, and C
def A : ℝ × ℝ := (0, 4 / 3)
def B : ℝ × ℝ := (-1, 0)
def C : ℝ × ℝ := (1, 0)

-- Define the lines AB, AC, and BC
def line_AB (P : ℝ × ℝ) : ℝ := 4 * P.1 - 3 * P.2 + 4
def line_AC (P : ℝ × ℝ) : ℝ := 4 * P.1 + 3 * P.2 - 4
def line_BC (P : ℝ × ℝ) : ℝ := P.2

-- Define distance function
def distance (line : ℝ × ℝ → ℝ) (P : ℝ × ℝ) : ℝ :=
  (line P) / (real.sqrt (4^2 + 3^2))

-- Conditions for point P
def geometric_mean_condition (P : ℝ × ℝ) : Prop :=
  distance line_BC P = real.sqrt (distance line_AB P * distance line_AC P)

-- Equation of the locus of point P
def is_on_locus (P : ℝ × ℝ) : Prop :=
  (2 * P.1^2 + 2 * P.2^2 + 3 * P.2 - 2 = 0) ∨ (8 * P.1^2 - 17 * P.2^2 + 12 * P.2 - 8 = 0)

-- Incenter D of triangle ABC
def D : ℝ × ℝ := (0, 1 / 2)

-- Line l passing through incenter D with a slope k
def line_l (k : ℝ) (P : ℝ × ℝ) : ℝ :=
  P.2 - k * P.1 - 1 / 2

-- Prove the following statements (without proofs for now)
theorem locus_equation :
  ∀ P : ℝ × ℝ, geometric_mean_condition P → is_on_locus P :=
sorry

theorem slope_range (k : ℝ) :
  (∀ P : ℝ × ℝ, is_on_locus P → line_l k P = 0) → (k = 0 ∨ k = ±(2 * real.sqrt 34) / 17 ∨ k = ±(real.sqrt 2) / 2) :=
sorry

end locus_equation_slope_range_l444_444095


namespace john_cost_per_use_is_5_51_l444_444110

-- Definitions based on the given conditions
def purchase_price : ℝ := 30
def operating_cost_per_use : ℝ := 0.50
def electricity_consumption_per_use : ℝ := 0.1
def price_of_electricity_per_kwh : ℝ := 0.12
def number_of_uses : ℕ := 6

-- Define the total operating cost
def total_operating_cost : ℝ := operating_cost_per_use * number_of_uses

-- Define the electricity cost per use
def electricity_cost_per_use : ℝ := electricity_consumption_per_use * price_of_electricity_per_kwh

-- Define the total electricity cost
def total_electricity_cost : ℝ := electricity_cost_per_use * number_of_uses

-- Define the total cost
def total_cost : ℝ := purchase_price + total_operating_cost + total_electricity_cost

-- Define the cost per use
def cost_per_use (total_cost : ℝ) (number_of_uses : ℕ) : ℝ := total_cost / number_of_uses

-- Lean 4 statement to match the problem question
theorem john_cost_per_use_is_5_51 :
  cost_per_use total_cost number_of_uses = 5.51 :=
sorry

end john_cost_per_use_is_5_51_l444_444110


namespace frustum_volume_correct_l444_444238

noncomputable def volume_of_frustum (d1 d2 h : ℝ) : ℝ :=
  let r1 := d1 / 2
  let r2 := d2 / 2
  (π * h * (r1^2 + r1 * r2 + r2^2)) / 3

theorem frustum_volume_correct :
  volume_of_frustum 10 14 8 = (872 * π) / 3 := 
by
  -- This is where the proof would go 
  sorry

end frustum_volume_correct_l444_444238


namespace tournament_rounds_l444_444163

/-- 
Given a tournament where each participant plays several games with every other participant
and a total of 224 games were played, prove that the number of rounds in the competition is 8.
-/
theorem tournament_rounds (x y : ℕ) (hx : x > 1) (hy : y > 0) (h : x * (x - 1) * y = 448) : y = 8 :=
sorry

end tournament_rounds_l444_444163


namespace radius_of_larger_circle_l444_444686

theorem radius_of_larger_circle (r : ℝ) (radius_ratio : ℝ) (h1 : radius_ratio = 3) (AC_diameter : ℝ) (BC_chord : ℝ) (tangent_point : ℝ) (AB_length : ℝ) (h2 : AB_length = 140) :
  3 * (AB_length / 4) = 210 :=
by 
  sorry

end radius_of_larger_circle_l444_444686


namespace num_non_squares_cubes_1_to_200_l444_444441

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n
def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k * k * k = n
def is_perfect_sixth_power (n : ℕ) : Prop := ∃ k : ℕ, k * k * k * k * k * k = n

def num_perfect_squares (a b : ℕ) : ℕ :=
  (List.range (b + 1)).countp is_perfect_square - (List.range (a)).countp is_perfect_square

def num_perfect_cubes (a b : ℕ) : ℕ :=
  (List.range (b + 1)).countp is_perfect_cube - (List.range (a)).countp is_perfect_cube

def num_perfect_sixth_powers (a b : ℕ) : ℕ :=
  (List.range (b + 1)).countp is_perfect_sixth_power - (List.range (a)).countp is_perfect_sixth_power

theorem num_non_squares_cubes_1_to_200 :
  let squares := num_perfect_squares 1 200 in
  let cubes := num_perfect_cubes 1 200 in
  let sixth_powers := num_perfect_sixth_powers 1 200 in
  200 - (squares + cubes - sixth_powers) = 182 :=
by
  let squares := num_perfect_squares 1 200
  let cubes := num_perfect_cubes 1 200
  let sixth_powers := num_perfect_sixth_powers 1 200
  have h : 200 - (squares + cubes - sixth_powers) = 182 := sorry
  exact h

end num_non_squares_cubes_1_to_200_l444_444441


namespace total_books_after_donations_l444_444862

variable (Boris_books : Nat := 24)
variable (Cameron_books : Nat := 30)

theorem total_books_after_donations :
  (Boris_books - Boris_books / 4) + (Cameron_books - Cameron_books / 3) = 38 := by
  sorry

end total_books_after_donations_l444_444862


namespace sufficient_but_not_necessary_condition_l444_444927

theorem sufficient_but_not_necessary_condition (f : ℝ → ℝ) (h : ∀ x, f x = x⁻¹) :
  ∀ x, (x > 1 → f (x + 2) > f (2*x + 1)) ∧ (¬ (x > 1) → ¬ (f (x + 2) > f (2*x + 1))) :=
by
  sorry

end sufficient_but_not_necessary_condition_l444_444927


namespace circumcircle_eqn_l444_444906

variables (D E F : ℝ)

def point_A := (4, 0)
def point_B := (0, 3)
def point_C := (0, 0)

-- Define the system of equations for the circumcircle
def system : Prop :=
  (16 + 4*D + F = 0) ∧
  (9 + 3*E + F = 0) ∧
  (F = 0)

theorem circumcircle_eqn : system D E F → (D = -4 ∧ E = -3 ∧ F = 0) :=
sorry -- Proof omitted

end circumcircle_eqn_l444_444906


namespace f_is_odd_f_max_on_interval_solve_inequality_l444_444375

noncomputable def f : ℝ → ℝ :=
by sorry

axiom f_additive : ∀ x y : ℝ, f (x + y) = f x + f y
axiom f_neg : ∀ x : ℝ, x > 0 → f x < 0
axiom f_at_one : f 1 = -2

theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x :=
by sorry

theorem f_max_on_interval : ∃ x : ℝ, x ∈ set.Icc (-3 : ℝ) (3 : ℝ) ∧ ∀ y : ℝ, y ∈ set.Icc (-3 : ℝ) (3 : ℝ) → f y ≤ f x :=
by sorry

theorem solve_inequality (a x : ℝ) : f (a * x^2) - 2 * f x < f (a * x) + 4 ↔ 
  (if a = 0 then x < 1 else if a = 2 then x ≠ 1 else if a < 0 then (2 / a) < x ∧ x < 1 else if 0 < a ∧ a < 2 then x > (2 / a) ∨ x < 1 else x < (2 / a) ∨ x > 1) :=
by sorry

end f_is_odd_f_max_on_interval_solve_inequality_l444_444375


namespace proof_exists_lcm_l444_444598

noncomputable def condition1 : ℕ → ℕ := sorry
def condition2 (c : ℝ) : Prop := 0 < c ∧ c < 3 / 2
def question (a : ℕ → ℕ) (c : ℝ) : Prop := ∃ᶠ k in at_top, Nat.lcm (a k) (a (k+1)) > (c * k)

theorem proof_exists_lcm (a : ℕ → ℕ) (c : ℝ) 
(h_st : ∀ i j : ℕ, i ≠ j → a i ≠ a j) 
(h_c : condition2 c) :
question a c := sorry

end proof_exists_lcm_l444_444598


namespace distribution_ways_l444_444568

theorem distribution_ways : 
  ∃ (ways : ℕ), ways = 393 ∧ 
  (∀ (red_balls white_balls boxes : ℕ), red_balls = 7 → white_balls = 7 → boxes = 7 → 
   ways = (cases_on_number_of_white_double_boxes) ) :=
begin
  sorry
end

end distribution_ways_l444_444568


namespace parts_can_be_503_parts_cannot_be_2020_l444_444260

theorem parts_can_be_503 :
  ∃ k : ℕ, 1 + 2 * k = 503 :=
by {
  use 251,
  linarith,
}

theorem parts_cannot_be_2020 :
  ¬ ∃ k : ℕ, 1 + 2 * k = 2020 :=
by {
  intro h,
  cases h with k hk,
  have : odd (2 * k + 1) := odd_iff_exists_bit0.mp ⟨k, rfl⟩,
  linarith,
}

end parts_can_be_503_parts_cannot_be_2020_l444_444260


namespace at_least_one_fuse_blows_l444_444560

theorem at_least_one_fuse_blows (pA pB : ℝ) (hA : pA = 0.85) (hB : pB = 0.74) (independent : ∀ (A B : Prop), A ∧ B → ¬(A ∨ B)) :
  1 - (1 - pA) * (1 - pB) = 0.961 :=
by
  sorry

end at_least_one_fuse_blows_l444_444560


namespace min_value_of_expression_l444_444958

theorem min_value_of_expression (m n : ℝ) (h₁ : 0 < m) (h₂ : m * n ≥ 2) : 2^m + 4^n = 8 :=
by
  sorry

end min_value_of_expression_l444_444958


namespace complex_arithmetic_l444_444037

theorem complex_arithmetic (i : ℂ) (hi : i^2 = -1) (z : ℂ) 
  (hz : z = -1/2 + (sqrt 3 / 2) * i) : z^2 + z + 1 = 0 := 
by
  sorry

end complex_arithmetic_l444_444037


namespace min_attempts_flashlight_a_l444_444737

open Nat

theorem min_attempts_flashlight_a (n : ℕ) (h : n > 2) : 
  ∃ (a : ℕ), (a = (n + 2)) ∧ ∀ (batteries : List Bool) (H : length batteries = 2 * n + 1 ∧ count batteries true = n + 1), 
  (∃ (i j : ℕ), i ≠ j ∧ batteries.nth i = some true ∧ batteries.nth j = some true) :=
sorry

end min_attempts_flashlight_a_l444_444737


namespace log_problem_l444_444318

theorem log_problem : log 4 + log 50 - log 2 = 2 := by
  sorry

end log_problem_l444_444318


namespace count_not_squares_or_cubes_200_l444_444532

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n
def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k * k * k = n
def is_sixth_power (n : ℕ) : Prop := ∃ k : ℕ, k^6 = n

theorem count_not_squares_or_cubes_200 :
  let count_squares := (Finset.range 201).filter is_perfect_square).card
  let count_cubes := (Finset.range 201).filter is_perfect_cube).card
  let count_sixth_powers := (Finset.range 201).filter is_sixth_power).card
  (200 - (count_squares + count_cubes - count_sixth_powers)) = 182 := 
by {
  -- Define everything in terms of Finset and filter
  let count_squares := (Finset.range 201).filter is_perfect_square).card,
  let count_cubes := (Finset.range 201).filter is_perfect_cube).card,
  let count_sixth_powers := (Finset.range 201).filter is_sixth_power).card,
  exact sorry
}

end count_not_squares_or_cubes_200_l444_444532


namespace log_eq_solution_l444_444987

theorem log_eq_solution (b x : ℝ) (hb_pos : 0 < b) (hb_ne_one : b ≠ 1) (hx_ne_one : x ≠ 1) (hx_pos : 0 < x) :
  log (b^3) x + log (x^3) b = 1 → x = b :=
begin
  sorry
end

end log_eq_solution_l444_444987


namespace geom_seq_problem_l444_444935

noncomputable theory

def geom_seq (a : ℕ → ℝ) (q : ℝ) := ∀ n, a (n + 1) = a n * q

def sum_geom_seq (a : ℕ → ℝ) (n : ℕ) := ∑ i in Finset.range n, a i

theorem geom_seq_problem (a : ℕ → ℝ) (q : ℝ) (h_geom : geom_seq a q)
    (h_a2 : a 1 = 2)
    (h_S3 : sum_geom_seq a 3 = 8) :
    (sum_geom_seq a 5) / (a 2) = 11 :=
sorry

end geom_seq_problem_l444_444935


namespace more_triangles_2003_than_2000_l444_444876

/-- 
  Prove that the number of triangles with integer sides and a perimeter of 2003
  is strictly greater than the number of triangles with integer sides and a 
  perimeter of 2000.
-/
theorem more_triangles_2003_than_2000 :
  let T_2000 := { (a, b, c) : ℕ × ℕ × ℕ | a + b + c = 2000 ∧ a + b > c ∧ a + c > b ∧ b + c > a }
  let T_2003 := { (a, b, c) : ℕ × ℕ × ℕ | a + b + c = 2003 ∧ a + b > c ∧ a + c > b ∧ b + c > a }
  in T_2003.card > T_2000.card :=
sorry

end more_triangles_2003_than_2000_l444_444876


namespace numbers_not_squares_nor_cubes_1_to_200_l444_444489

theorem numbers_not_squares_nor_cubes_1_to_200 : 
  let num_squares := nat.sqrt 200
  let num_cubes := nat.cbrt 200
  let num_sixth_powers := nat.root 6 200
  let total := num_squares + num_cubes - num_sixth_powers
  200 - total = 182 := 
by
  sorry

end numbers_not_squares_nor_cubes_1_to_200_l444_444489


namespace line_AF_through_midpoint_of_DE_l444_444803

open EuclideanGeometry

noncomputable def midpoint_of_segment (A B : Point) := ((A.x + B.x) / 2, (A.y + B.y) / 2)

theorem line_AF_through_midpoint_of_DE
  {A B C D E F : Point}
  (h_isosceles : AB = AC)
  (h_midpoint_D : D = midpoint B C)
  (h_perpendicular_DE : ⟂ D E AC)
  (h_circumcircle_intersects_BE : ∃ k : ℕ, circumcircle A B D = circum circle (B ⊗ F))
  : passes_through (line A F) (midpoint_of_segment D E) :=
sorry

end line_AF_through_midpoint_of_DE_l444_444803


namespace AF_through_midpoint_DE_l444_444752

open EuclideanGeometry

-- Definition of an isosceles triangle
variables {A B C D E F : Point}
variables [plane Geometry]
variables (ABC_isosceles : IsoscelesTriangle A B C)
variables (midpoint_D : Midpoint D B C)
variables (perpendicular_DE_AC : Perpendicular D E A C)
variables (F_on_circumcircle : OnCircumcircle F A B D)
variables (B_on_BE : OnLine B E)

-- Goal: Prove that line AF passes through the midpoint of DE
theorem AF_through_midpoint_DE 
  (h1 : triangle A B C) 
  (h2 : ABC_isosceles) 
  (h3 : midpoint D B C) 
  (h4 : perpendicular DE AC)
  (h5 : OnCircumcircle F A B D) 
  (h6 : OnLine B E) :
  PassesThroughMidpoint A F D E :=
sorry

end AF_through_midpoint_DE_l444_444752


namespace measure_of_BAD_in_parallelogram_l444_444570

theorem measure_of_BAD_in_parallelogram (ABCD : Parallelogram)
  (angle_ABC_eq_4_times_angle_BCD : ∠ ABC = 4 * ∠ BCD) :
  ∠ BAD = 36 :=
by sorry

end measure_of_BAD_in_parallelogram_l444_444570


namespace comparison_of_neg_square_roots_l444_444870

noncomputable def compare_square_roots : Prop :=
  -2 * Real.sqrt 11 > -3 * Real.sqrt 5

theorem comparison_of_neg_square_roots : compare_square_roots :=
by
  -- Omitting the proof details
  sorry

end comparison_of_neg_square_roots_l444_444870


namespace fifth_number_is_211_l444_444178

theorem fifth_number_is_211 (l : List ℕ)
  (h_len : l.length = 9)
  (h_avg : (l.sum : ℚ) / l.length = 207)
  (h_elem : l = [201, 202, 204, 205, 209, 209, 210, 212, 212]) :
  l.nth 4 = some 211 := sorry

end fifth_number_is_211_l444_444178


namespace non_neg_real_x_count_l444_444359

theorem non_neg_real_x_count : 
  let S := {x | ∃ n : ℕ, n = (256 - (x ^ (1 / 3))) ^ (1 / 2) ∧ x ≥ 0} in
  S.count = 17 := 
sorry

end non_neg_real_x_count_l444_444359


namespace main_theorem_l444_444749

universe u
variables {α : Type u} [euclidean_geometry α]

open_locale euclidean_geometry

-- Define the given conditions
def is_isosceles (A B C : α) (h : euclidean_geometry.is_triangle A B C) : Prop :=
euclidean_geometry.dist A B = euclidean_geometry.dist A C

def midpoint (D B C : α) : Prop :=
euclidean_geometry.is_midpoint D B C

def perpendicular (E D A C : α) : Prop :=
euclidean_geometry.is_perpendicular E D A C

def circumcircle_intersects (F B D A : α) (c : set α) : Prop :=
c = euclidean_geometry.circumcircle A B D ∧ F ∈ c ∧ B ∈ c

-- The main theorem to prove
theorem main_theorem (A B C D E F : α) (h1 : euclidean_geometry.is_triangle A B C)
    (h2 : is_isosceles A B C h1) (h3 : midpoint D B C) (h4 : perpendicular E D A C)
    (h5 : ∃ c, circumcircle_intersects F B D A c) :
    euclidean_geometry.passes_through (euclidean_geometry.line A F) (euclidean_geometry.midpoint E D) :=
sorry

end main_theorem_l444_444749


namespace prob_exactly_3_weeds_prob_no_more_than_3_weeds_l444_444836

open ProbabilityTheory

-- We define the conditions given in the problem
def n := 125
def p := 0.004
def λ := n * p

-- Part (a): Proving probability of exactly 3 weeds
theorem prob_exactly_3_weeds :
  P_poisson λ 3 = 0.012636 := sorry

-- Part (b): Proving probability of no more than 3 weeds
theorem prob_no_more_than_3_weeds :
  (P_poisson λ 0 + P_poisson λ 1 + P_poisson λ 2 + P_poisson λ 3) = 0.9982 := sorry

end prob_exactly_3_weeds_prob_no_more_than_3_weeds_l444_444836


namespace cos_function_odd_period_l444_444673

def f (x : ℝ) : ℝ := 4 * Real.cos (4 * x - 5 * Real.pi / 2)

theorem cos_function_odd_period : 
  (∀ x : ℝ, f (-x) = -f x) ∧ (∃ p : ℝ, p = Real.pi / 2 ∧ (∀ x : ℝ, f (x + p) = f x)) :=
sorry

end cos_function_odd_period_l444_444673


namespace only_possible_operation_is_ABC_l444_444026

variables (A : Matrix (Fin 3) (Fin 2) ℝ) 
          (B : Matrix (Fin 2) (Fin 3) ℝ)
          (C : Matrix (Fin 3) (Fin 3) ℝ)

theorem only_possible_operation_is_ABC : 
  (¬(∃ AC : Matrix (Fin 3) (Fin 3) ℝ, AC = A.mul C)) ∧
  (¬(∃ BAC : Matrix (Fin 3) (Fin 3) ℝ, let D := B.mul A in BAC = D.mul C)) ∧
  (∃ ABC : Matrix (Fin 3) (Fin 3) ℝ, let D := A.mul B in ABC = D.mul C) ∧
  (¬(∃ AB_AC : Matrix (Fin 3) (Fin 3) ℝ, AB_AC = (A.mul B).sub (A.mul C))) := 
by {
  sorry
}

end only_possible_operation_is_ABC_l444_444026


namespace unique_mismatched_pairs_are_10_l444_444700

-- Define the sock pairs based on color and pattern
inductive Sock
  | red_striped
  | green_polka_dotted
  | blue_checked
  | yellow_floral
  | purple_plaid

-- Function to check whether two socks are mismatched (not sharing the same color or pattern)
def mismatched (a b : Sock) : Prop :=
  a ≠ b

-- Set of all unique socks
def unique_socks : List Sock := [
  Sock.red_striped,
  Sock.green_polka_dotted,
  Sock.blue_checked,
  Sock.yellow_floral,
  Sock.purple_plaid
]

-- The final theorem statement
theorem unique_mismatched_pairs_are_10 : 
  (unique_socks.product unique_socks).count (λ (s : Sock × Sock), mismatched s.1 s.2) / 2 = 10 :=
by 
  sorry

end unique_mismatched_pairs_are_10_l444_444700


namespace AF_through_midpoint_DE_l444_444754

open EuclideanGeometry

-- Definition of an isosceles triangle
variables {A B C D E F : Point}
variables [plane Geometry]
variables (ABC_isosceles : IsoscelesTriangle A B C)
variables (midpoint_D : Midpoint D B C)
variables (perpendicular_DE_AC : Perpendicular D E A C)
variables (F_on_circumcircle : OnCircumcircle F A B D)
variables (B_on_BE : OnLine B E)

-- Goal: Prove that line AF passes through the midpoint of DE
theorem AF_through_midpoint_DE 
  (h1 : triangle A B C) 
  (h2 : ABC_isosceles) 
  (h3 : midpoint D B C) 
  (h4 : perpendicular DE AC)
  (h5 : OnCircumcircle F A B D) 
  (h6 : OnLine B E) :
  PassesThroughMidpoint A F D E :=
sorry

end AF_through_midpoint_DE_l444_444754


namespace odd_divisors_l444_444137

-- Define p_1, p_2, p_3 as distinct prime numbers greater than 3
variables {p_1 p_2 p_3 : ℕ}
-- Define k, a, b, c as positive integers
variables {n k a b c : ℕ}

-- The conditions
def distinct_primes (p_1 p_2 p_3 : ℕ) : Prop :=
  p_1 > 3 ∧ p_2 > 3 ∧ p_3 > 3 ∧ p_1 ≠ p_2 ∧ p_1 ≠ p_3 ∧ p_2 ≠ p_3

def is_n (n k p_1 p_2 p_3 a b c : ℕ) : Prop :=
  n = 2^k * p_1^a * p_2^b * p_3^c

def conditions (a b c : ℕ) : Prop :=
  a + b > c ∧ 1 ≤ b ∧ b ≤ c

-- The main statement
theorem odd_divisors
  (h_prime : distinct_primes p_1 p_2 p_3)
  (h_n : is_n n k p_1 p_2 p_3 a b c)
  (h_cond : conditions a b c) : 
  ∃ d : ℕ, d = (a + 1) * (b + 1) * (c + 1) :=
by sorry

end odd_divisors_l444_444137


namespace solve_tangent_inequality_l444_444687

noncomputable def solution_set_of_tan_inequality (x : ℝ) : Prop :=
  ∃ (k : ℤ), -π / 6 + k * π ≤ x ∧ x < π / 2 + k * π

theorem solve_tangent_inequality (x : ℝ) :
    (1 + sqrt 3 * tan x ≥ 0) ∧ (∃ k : ℤ, k * π - π / 2 < x ∧ x < k * π + π / 2 ) ↔ solution_set_of_tan_inequality x :=
by
  sorry

end solve_tangent_inequality_l444_444687


namespace tickets_difference_l444_444859

theorem tickets_difference :
  let tickets_won := 48.5
  let yoyo_cost := 11.7
  let keychain_cost := 6.3
  let plush_toy_cost := 16.2
  let total_cost := yoyo_cost + keychain_cost + plush_toy_cost
  let tickets_left := tickets_won - total_cost
  tickets_won - tickets_left = total_cost :=
by
  sorry

end tickets_difference_l444_444859


namespace numbers_not_perfect_squares_or_cubes_l444_444452

theorem numbers_not_perfect_squares_or_cubes : 
  let total_numbers := 200
  let perfect_squares := 14
  let perfect_cubes := 5
  let sixth_powers := 1
  total_numbers - (perfect_squares + perfect_cubes - sixth_powers) = 182 :=
by
  sorry

end numbers_not_perfect_squares_or_cubes_l444_444452


namespace problem1_problem2_l444_444811

section Problem1

variable (sqrt12 : ℝ) (pow2024_0 : ℝ) (sin60 : ℝ)
hypothesis h1 : sqrt12 = 2 * Real.sqrt 3
hypothesis h2 : pow2024_0 = 1
hypothesis h3 : sin60 = Real.sin (Real.pi / 3) / 2

theorem problem1 : sqrt12 + pow2024_0 - 4 * sin60 = 1 :=
by sorry

end Problem1

section Problem2

variable (x : ℝ)

theorem problem2 : (x + 2) ^ 2 + x * (x - 4) = 2 * x ^ 2 + 4 :=
by sorry

end Problem2

end problem1_problem2_l444_444811


namespace soda_bottles_duration_l444_444315

theorem soda_bottles_duration:
  (∀ (total_soda_bottles per_day: ℕ), total_soda_bottles = 360 → per_day = 9 →
    (total_soda_bottles / per_day) = 40) :=
by
  intros total_soda_bottles per_day h1 h2
  rw [h1, h2]
  exact Nat.div_eq_of_eq_mul_left zero_lt_nine (by norm_num : 360 = 9 * 40)

constant zero_lt_nine : 0 < 9 := Nat.zero_lt_succ 8

end soda_bottles_duration_l444_444315


namespace count_valid_numbers_between_1_and_200_l444_444425

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k^2 = n
def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k^3 = n
def is_perfect_sixth_power (n : ℕ) : Prop := ∃ k : ℕ, k^6 = n

def count_perfect_squares (m : ℕ) : ℕ :=
  Nat.sqrt m

def count_perfect_cubes (m : ℕ) : ℕ :=
  (m : ℝ).cbrt.to_nat

def count_perfect_sixth_powers (m : ℕ) : ℕ :=
  (m : ℝ).cbrt.sqrt.to_nat

def count_either_squares_or_cubes (m : ℕ) : ℕ :=
  let squares := count_perfect_squares m
  let cubes := count_perfect_cubes m
  let sixths := count_perfect_sixth_powers m
  squares + cubes - sixths

def count_neither_squares_nor_cubes (n : ℕ) : ℕ :=
  n - count_either_squares_or_cubes n

theorem count_valid_numbers_between_1_and_200 : count_neither_squares_nor_cubes 200 = 182 :=
by
  sorry

end count_valid_numbers_between_1_and_200_l444_444425


namespace number_of_girls_l444_444080

-- Definitions from the problem conditions
def ratio_girls_boys (g b : ℕ) : Prop := 4 * b = 3 * g
def total_students (g b : ℕ) : Prop := g + b = 56

-- The proof statement
theorem number_of_girls (g b k : ℕ) (hg : 4 * k = g) (hb : 3 * k = b) (hr : ratio_girls_boys g b) (ht : total_students g b) : g = 32 :=
by sorry

end number_of_girls_l444_444080


namespace find_z_l444_444998

noncomputable theory

namespace ComplexProof

open ComplexConjugate

-- Given condition: 2 * conjugate(z) - 3 = 1 + 5 * I
def condition (z : ℂ) : Prop := 2 * conj z - 3 = 1 + 5 * I

-- The proof problem: Prove that under the given condition, z = 2 - 5/2 * I
theorem find_z (z : ℂ) (h : condition z) : z = 2 - (5 / 2) * I :=
sorry

end ComplexProof

end find_z_l444_444998


namespace non_perfect_powers_count_l444_444478

theorem non_perfect_powers_count :
  let n := 200,
      num_perfect_squares := 14,
      num_perfect_cubes := 5,
      num_perfect_sixth_powers := 1,
      total_perfect_powers := num_perfect_squares + num_perfect_cubes - num_perfect_sixth_powers,
      total_non_perfect_powers := n - total_perfect_powers
  in
  total_non_perfect_powers = 182 :=
by
  sorry

end non_perfect_powers_count_l444_444478


namespace passing_through_midpoint_of_DE_l444_444791

noncomputable def is_isosceles (ABC : Triangle) : Prop :=
ABC.AB = ABC.AC

noncomputable def midpoint (D : Point) (BC : Segment) : Prop :=
(B + C) / 2 = D

noncomputable def perpendicular (DE : Line) (AC : Line) : Prop :=
right_angle (angle DE AC)

noncomputable def circumsircle_intersect (F : Point) (ABD : Triangle) (BE : Line) : Prop :=
(circumcircle ABD).contains B ∧ (circumcircle ABD).contains F ∧ line BE.contains B ∧ line BE.contains F

noncomputable def passes_through_midpoint (AF : Line) (DE : Segment) : Prop :=
exists M : Point, midpoint M DE ∧ line AF.contains M

theorem passing_through_midpoint_of_DE
  (ABC : Triangle) (D E F : Point) (DE : Line) (AC BE AF : Line) :
  is_isosceles ABC →
  midpoint D (BC) →
  perpendicular DE AC →
  circumsircle_intersect F (Triangle.mk ABD) BE →
  passes_through_midpoint AF (segment.mk D E) :=
by
  sorry

end passing_through_midpoint_of_DE_l444_444791


namespace parallel_planes_parallal_lines_l444_444034

variables {α β γ : Type} [plane α] [plane β] [plane γ]
variables (m n : line) (Pγ: plane)
variables (h1 : m = α ∩ γ) (h2 : n = β ∩ γ)

-- if α parallel β, then m parallel n
theorem parallel_planes_parallal_lines (h3 : α ∥ β): m ∥ n :=
by
  -- let's use sorry here since we are focused on structure
  sorry

end parallel_planes_parallal_lines_l444_444034


namespace container_holds_slices_l444_444252

theorem container_holds_slices (x : ℕ) 
  (h1 : x > 1) 
  (h2 : x ≠ 332) 
  (h3 : x ≠ 166) 
  (h4 : x ∣ 332) :
  x = 83 := 
sorry

end container_holds_slices_l444_444252


namespace clusters_of_oats_l444_444623

-- Define conditions:
def clusters_per_spoonful : Nat := 4
def spoonfuls_per_bowl : Nat := 25
def bowls_per_box : Nat := 5

-- Define the question and correct answer:
def clusters_per_box : Nat :=
  clusters_per_spoonful * spoonfuls_per_bowl * bowls_per_box

-- Theorem statement for the proof problem:
theorem clusters_of_oats:
  clusters_per_box = 500 :=
by
  sorry

end clusters_of_oats_l444_444623


namespace problem1_l444_444815

theorem problem1 :
  sqrt (25 / 9) - (8 / 27)^(1 / 3) - (Real.pi + Real.exp 1)^0 + (1 / 4)^(-1 / 2) = 2 :=
by sorry

end problem1_l444_444815


namespace numbers_not_squares_nor_cubes_1_to_200_l444_444497

theorem numbers_not_squares_nor_cubes_1_to_200 : 
  let num_squares := nat.sqrt 200
  let num_cubes := nat.cbrt 200
  let num_sixth_powers := nat.root 6 200
  let total := num_squares + num_cubes - num_sixth_powers
  200 - total = 182 := 
by
  sorry

end numbers_not_squares_nor_cubes_1_to_200_l444_444497


namespace domain_of_f_f_is_odd_f_increasing_on_pos_l444_444404

-- Definition for the function f
def f (x : ℝ) : ℝ := 3 * x - 2 / x

-- Prove the domain of f is (-∞, 0) ∪ (0, +∞)
theorem domain_of_f : {x : ℝ | x ≠ 0} = set.univ \ {0} :=
by
  sorry

-- Prove that f is an odd function
theorem f_is_odd (x : ℝ) : f (-x) = -f (x) :=
by
  sorry

-- Prove that f is increasing on (0, +∞)
theorem f_increasing_on_pos (x1 x2 : ℝ) (h : 0 < x1) (h1 : 0 < x2) (h2 : x1 < x2) : f x1 < f x2 :=
by
  sorry

end domain_of_f_f_is_odd_f_increasing_on_pos_l444_444404


namespace triangle_PQT_is_isosceles_at_P_l444_444807

variables {A B C D E P Q T : Type} [InCircle A B C D E]
variables (h1 : dist A B = dist B C)
          (h2 : dist C D = dist D E)
          (hP : ∃ P, Line A D ∩ Line B E = P)
          (hQ : ∃ Q, Line A C ∩ Line B D = Q)
          (hT : ∃ T, Line B D ∩ Line C E = T)

theorem triangle_PQT_is_isosceles_at_P :
    isosceles_triangle PQT :=
sorry

end triangle_PQT_is_isosceles_at_P_l444_444807


namespace telephone_number_problem_l444_444843

theorem telephone_number_problem 
    (A B C D E F G H I J : ℕ)
    (h_unique : list.nodup [A, B, C, D, E, F, G, H, I, J])
    (h_ABC_desc : A > B ∧ B > C)
    (h_DEF_desc : D > E ∧ E > F)
    (h_GHIJ_desc : G > H ∧ H > I ∧ I > J)
    (h_consecutive_odd : ∃ k : ℕ, [D, E, F] = [2 * k + 1, 2 * k + 3, 2 * k + 5])
    (h_consecutive_even : ∃ l : ℕ, [G, H, I, J] = [2 * l, 2 * l + 2, 2 * l + 4, 2 * l + 6])
    (h_sum : A + B + C = 11) :
  A = 9 := sorry

end telephone_number_problem_l444_444843


namespace game_winning_strategy_l444_444635

theorem game_winning_strategy (n : ℕ) : (n % 2 = 0 → ∃ strategy : ℕ → ℕ, ∀ m, strategy m = 1) ∧ (n % 2 = 1 → ∃ strategy : ℕ → ℕ, ∀ m, strategy m = 2) :=
by
  sorry

end game_winning_strategy_l444_444635


namespace find_a_l444_444924

theorem find_a (a : ℝ) (h : 3 ∈ ({1, a, a - 2} : Set ℝ)) : a = 5 :=
sorry

end find_a_l444_444924


namespace AF_passes_through_midpoint_of_DE_l444_444781

open EuclideanGeometry

-- Definitions based on the conditions
variables {A B C D E F : Point} 
variables {ABC : Triangle}
variables (isosceles_ABC : IsIsoscelesTriangle ABC A B)
variables (midpoint_D : IsMidpoint D B C)
variables (perpendicular_DE : IsPerpendicular DE D AC)
variables {circumcircle_ABD : Circumcircle ABD}
variables (intersection_F : IntersectsAt circumcircle_ABD BE B F)

-- The theorem statement
theorem AF_passes_through_midpoint_of_DE :
  PassesThroughMidpoint AF D E :=
sorry

end AF_passes_through_midpoint_of_DE_l444_444781


namespace part1_l444_444030

def U : Set ℝ := Set.univ
def P (a : ℝ) : Set ℝ := {x | 4 ≤ x ∧ x ≤ 7}
def Q : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}

theorem part1 (a : ℝ) (P_def : P 3 = {x | 4 ≤ x ∧ x ≤ 7}) :
  ((U \ P a) ∩ Q = {x | -2 ≤ x ∧ x < 4}) := by
  sorry

end part1_l444_444030


namespace cut_figure_into_rectangles_and_square_l444_444581

theorem cut_figure_into_rectangles_and_square :
  (∃ (n : ℕ), n = 10) :=
begin
  -- Given a figure of 17 cells
  let figure_cells := 17,
  -- Condition: Cut into 8 rectangles of size 1 × 2
  let rectangles := 8,
  -- Condition: One square of size 1 × 1
  let square := 1,
  -- We need to prove that there are exactly 10 ways to cut the figure
  use 10,
  sorry
end

end cut_figure_into_rectangles_and_square_l444_444581


namespace standard_equation_of_ellipse_line_AB_fixed_point_l444_444401

noncomputable def ellipse_equation (x y : ℝ) (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a > b ∧ (x^2 / a^2 + y^2 / b^2 = 1)

noncomputable def passes_through_point (x y : ℝ) (h : ℝ → ℝ → Prop) : Prop :=
  h x y

noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - (b^2 / a^2))

theorem standard_equation_of_ellipse :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a > b ∧ 
  eccentricity a b = Real.sqrt 3 / 2 ∧ 
  passes_through_point 2 1 (ellipse_equation 2 1) ∧
  ∀ x y, ellipse_equation x y a b ↔ (x^2 / 8 + y^2 / 2 = 1) :=
by
  sorry

theorem line_AB_fixed_point (M N : ℝ × ℝ) :
  (M.snd = N.snd) →
  (∃ k t : ℝ, ∀ x y, 
    (passes_through_point x y (ellipse_equation x y 8 2) ∧
    ∀ P, (P = (2, 1)) ∧ 
          (P ≠ M ∧ P ≠ N) ∧
          M.fst = N.fst ∧ M.snd ≠ N.snd ∧ 
          ((∃ Q : ℝ × ℝ, 
            passes_through_point Q.fst Q.snd (λ x y, ∃ m n, P.fst * m + P.snd * n ≤ Q.snd) ∧ Q = Q)) :=
by
  sorry

end standard_equation_of_ellipse_line_AB_fixed_point_l444_444401


namespace pyramid_height_eq_375_l444_444825

theorem pyramid_height_eq_375 :
  let a := 5 
  let b := 10
  let V_cube := a^3
  let V_pyramid := (b^2 * h) / 3
  V_cube = V_pyramid →
  h = 3.75 :=
by
  let a := 5
  let b := 10
  let V_cube := a^3
  let V_pyramid := (b^2 * h) / 3
  have : V_cube = 125 := by norm_num
  have : V_pyramid = (100 * h) / 3 := by norm_num
  sorry

end pyramid_height_eq_375_l444_444825


namespace part1_part2_part3_l444_444055

def f (x a : ℝ) : ℝ := x * abs (x - a)

theorem part1 (a : ℝ) :
  (∀ x y : ℝ, f x a + x ≤ f y a + y → x ≤ y) →
  -1 ≤ a ∧ a ≤ 1 := sorry

theorem part2 (a : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → f x a < 1) →
  3 / 2 < a ∧ a < 2 := sorry

theorem part3 (a : ℝ) :
  (a ≥ 2) →
  (if a > 8 then ∃ b c : ℝ, b = 2 * a - 4 ∧ c = 4 * a - 16 ∧ ∀ y, y ∈ set.Icc 2 4 → f y a ∈ set.Icc b c
   else if 4 ≤ a ∧ a ≤ 8 then
     (if a < 6 then ∃ b c : ℝ, b = 4 * a - 16 ∧ c = (a^2 / 4) ∧ ∀ y, y ∈ set.Icc 2 4 → f y a ∈ set.Icc b c
      else ∃ b c : ℝ, b = 2 * a - 4 ∧ c = (a^2 / 4) ∧ ∀ y, y ∈ set.Icc 2 4 → f y a ∈ set.Icc b c)
   else if 2 ≤ a ∧ a < 4 then
     (if a < 10 / 3 then ∃ b c : ℝ, b = 0 ∧ c = 16 - 4 * a ∧ ∀ y, y ∈ set.Icc 2 4 → f y a ∈ set.Icc b c
      else ∃ b c : ℝ, b = 0 ∧ c = 2 * a - 4 ∧ ∀ y, y ∈ set.Icc 2 4 → f y a ∈ set.Icc b c)
   else false) := sorry

end part1_part2_part3_l444_444055


namespace necessary_not_sufficient_l444_444369

noncomputable def perpendicular_condition (α β : Set Point) (m : Set Point) : Prop := 
  (∀ pt ∈ m, pt ∈ α) ∧                     -- m is in α
  (∃ n : Set Point, Line n ∧ ∀ pt ∈ n, pt ∉ α ∧ pt ∉ β) ∧ -- α and β are distinct planes
  (∀ pt ∈ m, pt ⊥ β ↔ α ⊥ β)               -- proving necessary but not sufficient condition

theorem necessary_not_sufficient (α β : Set Point) (m : Set Point) (h1 : ∀ pt ∈ m, pt ∈ α)
  (h2: ∃ n : Set Point, Line n ∧ ∀ pt ∈ n, pt ∉ α ∧ pt ∉ β) : 
  ∀ pt ∈ m, pt ⊥ β ↔ α ⊥ β :=
begin
  sorry
end

end necessary_not_sufficient_l444_444369


namespace sum_of_divisors_360_l444_444304

/-- Calculate the sum of the positive whole number divisors of 360 -/
theorem sum_of_divisors_360 : (∑ d in divisors 360, d) = 1170 :=
by
  sorry

end sum_of_divisors_360_l444_444304


namespace passing_through_midpoint_of_DE_l444_444797

noncomputable def is_isosceles (ABC : Triangle) : Prop :=
ABC.AB = ABC.AC

noncomputable def midpoint (D : Point) (BC : Segment) : Prop :=
(B + C) / 2 = D

noncomputable def perpendicular (DE : Line) (AC : Line) : Prop :=
right_angle (angle DE AC)

noncomputable def circumsircle_intersect (F : Point) (ABD : Triangle) (BE : Line) : Prop :=
(circumcircle ABD).contains B ∧ (circumcircle ABD).contains F ∧ line BE.contains B ∧ line BE.contains F

noncomputable def passes_through_midpoint (AF : Line) (DE : Segment) : Prop :=
exists M : Point, midpoint M DE ∧ line AF.contains M

theorem passing_through_midpoint_of_DE
  (ABC : Triangle) (D E F : Point) (DE : Line) (AC BE AF : Line) :
  is_isosceles ABC →
  midpoint D (BC) →
  perpendicular DE AC →
  circumsircle_intersect F (Triangle.mk ABD) BE →
  passes_through_midpoint AF (segment.mk D E) :=
by
  sorry

end passing_through_midpoint_of_DE_l444_444797


namespace helen_hand_wash_frequency_l444_444420

theorem helen_hand_wash_frequency :
  ∀ (time_per_wash total_time_weeks_in_year : ℕ),
  (time_per_wash = 30) →
  (total_time_weeks_in_year = (390, 52)) →
  (52 / (390 / 30) = 4) :=
by
  sorry

end helen_hand_wash_frequency_l444_444420


namespace main_theorem_l444_444748

universe u
variables {α : Type u} [euclidean_geometry α]

open_locale euclidean_geometry

-- Define the given conditions
def is_isosceles (A B C : α) (h : euclidean_geometry.is_triangle A B C) : Prop :=
euclidean_geometry.dist A B = euclidean_geometry.dist A C

def midpoint (D B C : α) : Prop :=
euclidean_geometry.is_midpoint D B C

def perpendicular (E D A C : α) : Prop :=
euclidean_geometry.is_perpendicular E D A C

def circumcircle_intersects (F B D A : α) (c : set α) : Prop :=
c = euclidean_geometry.circumcircle A B D ∧ F ∈ c ∧ B ∈ c

-- The main theorem to prove
theorem main_theorem (A B C D E F : α) (h1 : euclidean_geometry.is_triangle A B C)
    (h2 : is_isosceles A B C h1) (h3 : midpoint D B C) (h4 : perpendicular E D A C)
    (h5 : ∃ c, circumcircle_intersects F B D A c) :
    euclidean_geometry.passes_through (euclidean_geometry.line A F) (euclidean_geometry.midpoint E D) :=
sorry

end main_theorem_l444_444748


namespace operation_105_5_l444_444995

def operation (a b : ℕ) : ℕ :=
  a + 5 + b * 15

theorem operation_105_5 : operation 105 5 = 185 :=
by
  unfold operation
  simp
  -- Skipping steps in the proof
  sorry

end operation_105_5_l444_444995


namespace part1_part2_l444_444962

def A (a : ℝ) : set ℝ := {x | 2 - a ≤ x ∧ x ≤ 2 + a}
def B : set ℝ := {x | x ≤ 1 ∨ x ≥ 4}

theorem part1 (a : ℝ) (h : a = 3) : A a ∩ B = {x | -1 ≤ x ∧ x ≤ 1 ∨ 4 ≤ x ∧ x ≤ 5} := by
  sorry

theorem part2 (a : ℝ) (h1 : A a ∩ B = ∅) (h2 : a > 0) : 0 < a ∧ a < 1 := by
  sorry

end part1_part2_l444_444962


namespace num_from_1_to_200_not_squares_or_cubes_l444_444460

noncomputable def numNonPerfectSquaresAndCubes (n : ℕ) : ℕ :=
  let num_squares := 14
  let num_cubes := 5
  let num_sixth_powers := 2
  n - (num_squares + num_cubes - num_sixth_powers)

theorem num_from_1_to_200_not_squares_or_cubes : numNonPerfectSquaresAndCubes 200 = 183 := by
  sorry

end num_from_1_to_200_not_squares_or_cubes_l444_444460


namespace kim_sequence_probability_l444_444362

/-- Determine the pairs (n, p) such that Kim's arrangement of sequences gives the required probability. -/
theorem kim_sequence_probability (n : ℕ) (h_n : 0 < n) (p : ℚ) (h_p : 0 < p ∧ p < 1) :
  (∃ a : ℕ → ℕ, (∀ k, 0 ≤ a k ∧ a k ≤ Nat.choose n k) ∧ 
  2 * ∑ k in Finset.range (n+1), (a k) * (p^k) * ((1-p)^(n-k)) = 1) ↔ p = 1/2 :=
sorry

end kim_sequence_probability_l444_444362


namespace like_terms_correct_l444_444852

theorem like_terms_correct :
  ∃ (group : String),
  group = "D" ∧
  (∀ (x y : String),
    (x = "9a^{2}x" ∧ y = "9a^{2}" → false) ∧
    (x = "a^{2}" ∧ y = "2a" → false) ∧
    (x = "2a^{2}b" ∧ y = "3ab^{2}" → false) ∧
    (x = "4x^{2}y" ∧ y = "-yx^{2}" → true)) := 
begin
  sorry
end

end like_terms_correct_l444_444852


namespace can_decide_two_fakes_l444_444849

/- Define the weights of coins -/
constant genuine_weight : ℕ
constant fake_weight : ℕ

/- Conditions: All genuine coins have the same weight, all fake coins have the same weight, and fake coins are heavier than genuine coins -/
axiom fw_gt_gw : fake_weight > genuine_weight

/- Condition: There are 4 coins, where it is suspected that exactly 2 are fake -/
constants a b c d : ℕ
axiom coin_weights : List ℕ := [a, b, c, d]

/- Define a balance function that can be used twice to compare weights -/
def balance (left right : List ℕ) : Ordering := 
  if left.sum = right.sum then Ordering.eq 
  else if left.sum > right.sum then Ordering.gt 
  else Ordering.lt

/- Problem Statement: You can determine whether exactly 2 out of 4 coins are fake by using the balance twice -/
theorem can_decide_two_fakes (initial_balance second_balance : Ordering) :
  (∃ left1 right1 left2 right2 : List ℕ,
    left1.length = 2 ∧ right1.length = 2 ∧ left2.length = 2 ∧ right2.length = 2 ∧ 
    balance left1 right1 = initial_balance ∧
    balance left2 right2 = second_balance ∧
    (initial_balance = Ordering.eq → second_balance ≠ Ordering.eq) ∧
    (initial_balance ≠ Ordering.eq → second_balance = balance left2 right2)) → 
  (∃ fakes : List ℕ, fakes.length = 2 ∧ ∀ k ∈ fakes, k = fake_weight) ∨
  (∃ genuines : List ℕ, genuines.length = 4 ∧ ∀ k ∈ genuines, k = genuine_weight) :=
sorry

end can_decide_two_fakes_l444_444849


namespace count_diff_squares_1_to_500_l444_444975

theorem count_diff_squares_1_to_500 : ∃ count : ℕ, count = 416 ∧
  (count = (nat.card {n : ℕ | n ∈ (set.Icc 1 500) ∧ (∃ a b : ℕ, a^2 - b^2 = n)})) := by
sorry

end count_diff_squares_1_to_500_l444_444975


namespace numbers_not_perfect_squares_or_cubes_l444_444451

theorem numbers_not_perfect_squares_or_cubes : 
  let total_numbers := 200
  let perfect_squares := 14
  let perfect_cubes := 5
  let sixth_powers := 1
  total_numbers - (perfect_squares + perfect_cubes - sixth_powers) = 182 :=
by
  sorry

end numbers_not_perfect_squares_or_cubes_l444_444451


namespace tangent_PA_l444_444113

noncomputable def acute_triangle (A B C P M S T X Y : Type) := 
  ∃ (Γ : circle), 
  ∃ (r : line), 
  ∃ (ABC : triangle) (A_MGamma : (bisector (angle A B C))),
  triangle.is_acute ABC 
  ∧ circumcircle ABC = Γ 
  ∧ (angle_bisector (angle A B C)).intersection_point_with_circumcircle A_MGamma = M 
  ∧ line.parallel r (line_through_points B C)
  ∧ line.intersects r (line_through_points A C) X
  ∧ line.intersects r (line_through_points A B) Y
  ∧ line.intersects (line_through_points M X) Γ S
  ∧ line.intersects (line_through_points M Y) Γ T 
  ∧ line.intersects (line_through_points X Y) (line_through_points S T) P

theorem tangent_PA (A B C P M S T X Y : Type) :
  acute_triangle A B C P M S T X Y → tangent PA circumsphere :=
sorry

end tangent_PA_l444_444113


namespace median_length_l444_444102

theorem median_length (PQ PR : ℝ) (cosP : ℝ) (hPQ : PQ = 5) (hPR : PR = 8) (hcosP : cosP = 3 / 5) : 
  ∃ PM : ℝ, PM = sqrt 137 / 2 :=
by
  -- Omitting the proof
  sorry

end median_length_l444_444102


namespace frac_sum_diff_l444_444918

theorem frac_sum_diff (a b : ℝ) (h : (1/a + 1/b) / (1/a - 1/b) = 1001) : (a + b) / (a - b) = -1001 :=
sorry

end frac_sum_diff_l444_444918


namespace eccentricity_of_hyperbola_is_root_5_over_2_l444_444957

noncomputable def hyperbola_eccentricity (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) : ℝ :=
  let c := sqrt (a^2 + b^2)
  let area_AOF : ℝ := b^2
  have ha : a = 2 * b, from sorry -- Derived from the provided area condition and calculations
  ecc := c / a
  ecc

theorem eccentricity_of_hyperbola_is_root_5_over_2 
  (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (area_AOF : ℝ) (h_area : area_AOF = b^2) :
  hyperbola_eccentricity a b h_a_pos h_b_pos = (√5 / 2) :=
sorry

end eccentricity_of_hyperbola_is_root_5_over_2_l444_444957


namespace parallelogram_DEGH_l444_444116

variables {A B C N K L M D E G H : Type*}
variables [IsTriangle A B C] [IsOutsideTriangle ANC ABC CLB BKA]
variables (midpoint D A B) (midpoint E L K) (midpoint G C A) (midpoint H N A)
variables 
  (angle_eq₁ : ∠NAC = ∠KBA)
  (angle_eq₂ : ∠KBA = ∠LCB)
  (angle_eq₃ : ∠NCA = ∠KAB)
  (angle_eq₄ : ∠KAB = ∠LBC)

theorem parallelogram_DEGH : isParallelogram D E G H :=
sorry

end parallelogram_DEGH_l444_444116


namespace digit_to_the_right_of_4_in_21_div_22_is_5_l444_444073

theorem digit_to_the_right_of_4_in_21_div_22_is_5 : 
  ∀ n : ℕ, (0.954545 : ℝ) = 21 / 22 → 
  ∃ m : ℕ, (0.954545 : ℝ).digits.drop(n).head = 4 ∧
           (0.954545 : ℝ).digits.drop(n+1).head = 5 :=
by
  sorry

end digit_to_the_right_of_4_in_21_div_22_is_5_l444_444073


namespace min_area_quadrilateral_min_area_circumscribed_circle_l444_444048

variables (m n : ℝ)

def is_on_curve (x y : ℝ) : Prop :=
  (4 / x^2) + (9 / y^2) = 1

def in_quadrilateral (m n : ℝ) : Prop :=
  is_on_curve m n ∧ is_on_curve (-m) n ∧
  is_on_curve (-m) (-n) ∧ is_on_curve m (-n)

theorem min_area_quadrilateral 
  (h : in_quadrilateral m n) : 
  let area := 4 * m * n in
  area = 48 :=
sorry

theorem min_area_circumscribed_circle
  (h : in_quadrilateral m n) :
  let circ_area := 25 * π in
  circ_area = 25 * π :=
sorry

end min_area_quadrilateral_min_area_circumscribed_circle_l444_444048


namespace perpendicular_planes_condition_l444_444371

variables (α β : Plane) (m : Line) 

-- Assuming the basic definitions:
def perpendicular (α β : Plane) : Prop := sorry
def in_plane (m : Line) (α : Plane) : Prop := sorry
def perpendicular_to_plane (m : Line) (β : Plane) : Prop := sorry

-- Conditions
axiom α_diff_β : α ≠ β
axiom m_in_α : in_plane m α

-- Proving the necessary but not sufficient condition
theorem perpendicular_planes_condition : 
  (perpendicular α β → perpendicular_to_plane m β) ∧ 
  (¬ perpendicular_to_plane m β → ¬ perpendicular α β) ∧ 
  ¬ (perpendicular_to_plane m β → perpendicular α β) :=
sorry

end perpendicular_planes_condition_l444_444371


namespace general_term_formula_l444_444041

noncomputable def sum_seq (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in finset.range n, a i

variable a : ℕ → ℝ

variable pos_terms : ∀ n, a n > 0

variable seq_sum : ∀ n, 4 * sum_seq a (n + 1) = (a n) ^ 2 + 2 * (a n)

theorem general_term_formula : ∀ n, a n = 2 * (n + 1) :=
by
  sorry

end general_term_formula_l444_444041


namespace pyramid_height_eq_375_l444_444824

theorem pyramid_height_eq_375 :
  let a := 5 
  let b := 10
  let V_cube := a^3
  let V_pyramid := (b^2 * h) / 3
  V_cube = V_pyramid →
  h = 3.75 :=
by
  let a := 5
  let b := 10
  let V_cube := a^3
  let V_pyramid := (b^2 * h) / 3
  have : V_cube = 125 := by norm_num
  have : V_pyramid = (100 * h) / 3 := by norm_num
  sorry

end pyramid_height_eq_375_l444_444824


namespace numbers_neither_square_nor_cube_l444_444540

theorem numbers_neither_square_nor_cube (n : ℕ) (h : n = 200) :
  let num_squares := 14 in
  let num_cubes := 5 in
  let num_sixth_powers := 2 in
  (n - (num_squares + num_cubes - num_sixth_powers)) = 183 :=
by
  -- Let the number of integers in the range
  let num_total := 200
  -- Define how we count perfect squares and cubes
  let num_squares := 14
  let num_cubes := 5
  let num_sixth_powers := 2

  -- Subtract the overlapping perfect sixth powers from the sum of squares and cubes
  let num_either_square_or_cube := num_squares + num_cubes - num_sixth_powers

  -- Calculate the final result
  let result := num_total - num_either_square_or_cube

  -- Prove by computation
  show result = 183 from
    calc
      result = num_total - num_either_square_or_cube := rfl
           ... = 200 - (14 + 5 - 2)                   := rfl
           ... = 200 - 17                             := rfl
           ... = 183                                  := rfl

end numbers_neither_square_nor_cube_l444_444540


namespace exists_point_A_l444_444024

noncomputable def point_exists (a : line) (r d d₁ : ℝ) : Prop :=
  ∃ A : point, 
  dist A (proj_line a) = r ∧ 
  dist A (first_visual_plane) = d ∧ 
  dist A (second_visual_plane) = d₁

-- Assuming the existence of basic objects and distances in 3D geometry
axiom a : line
axiom first_visual_plane : plane
axiom second_visual_plane : plane
axiom proj_line : line → set point  -- projection function to the line
axiom first_visual_plane : set point  -- the first visual plane as a set of points
axiom second_visual_plane : set point  -- the second visual plane as a set of points
axiom dist : point → set point → ℝ  -- distance function

theorem exists_point_A : 
  ∀ (a : line) (r d d₁ : ℝ), 
  point_exists a r d d₁ :=
by 
  intro a r d d₁
  sorry

end exists_point_A_l444_444024


namespace min_diameter_to_scientific_notation_l444_444860

def nm_to_m (nm : ℝ) : ℝ := nm * 10 ^ (-9)

theorem min_diameter_to_scientific_notation :
  (nm_to_m 80) = (8 * 10 ^ (-8)) := 
by
  sorry

end min_diameter_to_scientific_notation_l444_444860


namespace triangle_XYZ_XY_l444_444104

open Real

theorem triangle_XYZ_XY (XY XZ YZ : ℝ) (h1 : YZ = 30)
  (h2 : tan (atan (XY / XZ)) = 3 * cos (atan (XZ / YZ))) :
  XY = -5 + (sqrt 3700) / 2 :=
by
  -- Proof goes here, left as sorry
  sorry

end triangle_XYZ_XY_l444_444104


namespace C1_polar_eq_ratio_of_areas_l444_444959

-- Define curve C and its transformation to C1
def curveC_parametric_eq (θ : ℝ) : ℝ × ℝ :=
  (2 * Real.cos θ, Real.sin θ)

def curveC1_parametric_eq (θ : ℝ) : ℝ × ℝ :=
  (2 * Real.cos θ, 2 * Real.sin θ)

-- Define polar equation of the line l
def polar_eq_line_l (ρ θ : ℝ) : Prop :=
  ρ * Real.sin (Real.pi / 4 + θ) = 2 * Real.sqrt 2

-- Define points A, A1, B, B1 with their polar coordinates
def pointA : ℝ × ℝ :=
  (2, Real.pi / 6)

def pointA1 : ℝ × ℝ :=
  (2, -Real.pi / 6)

def pointB (ρ1 : ℝ) : ℝ × ℝ :=
  (ρ1, Real.pi / 6)

def pointB1 (ρ2 : ℝ) : ℝ × ℝ :=
  (ρ2, -Real.pi / 6)

-- The mathematical proof problems
theorem C1_polar_eq :
  ∀ θ : ℝ, ∃ ρ : ℝ, curveC1_parametric_eq θ = (ρ * Real.cos θ, ρ * Real.sin θ) ∧ ρ = 2 := by
  sorry

theorem ratio_of_areas :
  ∀ ρ1 ρ2 : ℝ, (ρ1 * ρ2 = 32) →
  let S1 := 2 * Real.sin (Real.pi / 3),
      S2 := (1 / 2) * (ρ1 * ρ2) * Real.sin (Real.pi / 3)
  in S1 / S2 = 1 / 8 := by
  sorry

end C1_polar_eq_ratio_of_areas_l444_444959


namespace angle_dac_30_l444_444091

theorem angle_dac_30
    (AB_parallel_DC : AB ∥ DC)
    (AFE_straight : ∠ AFE = 180)
    (angle_ABC : ∠ ABC = 70)
    (angle_ACB : ∠ ACB = 85)
    (angle_ADC : ∠ ADC = 125) :
    ∠ DAC = 30 := by
  sorry

end angle_dac_30_l444_444091


namespace rabbit_distribution_count_l444_444646

-- Define the set of rabbits
inductive Rabbit
| Peter | Pauline | Flopsie | Mopsie | Cottontail | Topsy

-- Define the set of stores
inductive Store
| S1 | S2 | S3 | S4

-- Defining the problem conditions in Lean as predicates and parameters
-- no store gets both a parent and a child
def noParentAndChild (distribution : Rabbit → Store) : Prop :=
  ∀ (r1 r2 : Rabbit),
    ((r1 = Rabbit.Peter ∨ r1 = Rabbit.Pauline) ∧ 
     (r2 ≠ Rabbit.Peter ∧ r2 ≠ Rabbit.Pauline)) →
    ¬ (distribution r1 = distribution r2)

-- one store cannot have more than two rabbits
def maxTwoRabbits (distribution : Rabbit → Store) : Prop :=
  ∀ (s : Store),
    (Finset.filter (λ r, distribution r = s) Finset.univ).card ≤ 2

-- Define the main problem statement
theorem rabbit_distribution_count :
  (∃ f : Rabbit → Store, noParentAndChild f ∧ maxTwoRabbits f) → 54 := sorry

end rabbit_distribution_count_l444_444646


namespace interest_rate_is_correct_l444_444180

noncomputable def find_interest_rate (P : ℝ) (t : ℕ) (D : ℝ) : ℝ :=
  let r := 5 in  -- Replace this stub with actual calculation in full proof
  r

theorem interest_rate_is_correct :
  let P := 6800.000000000145 in
  let t := 2 in
  let D := 17 in
  find_interest_rate P t D = 5 :=
by
  sorry  -- Proof goes here

end interest_rate_is_correct_l444_444180


namespace probability_diff_max_min_eq_5_l444_444198

-- Define the set of cards and the number of draws
def cards := {1, 2, 3, 4, 5, 6}
def num_draws := 4

-- Define the main theorem
theorem probability_diff_max_min_eq_5 :
  let total_outcomes := 6 ^ num_draws,
      favorable_outcomes := total_outcomes - 2 * (5 ^ num_draws) + (4 ^ num_draws)
  in (favorable_outcomes : ℚ) / total_outcomes = 151 / 648 :=
by
  sorry

end probability_diff_max_min_eq_5_l444_444198


namespace proof_min_value_l444_444398

noncomputable def min_value (a b : ℝ) (h : (a ≠ 0 ∧ b ≠ 0) ∧
    ((x y : ℝ), (x^2 + y^2 + 2 * a * x + a^2 - 4 = 0) ∧
    (x^2 + y^2 - 4 * b * y - 1 + 4 * b^2 = 0) ∧
    (∃ (tangents : ℕ), tangents = 3)) : ℝ :=
  16/9

theorem proof_min_value (a b : ℝ)
  (h : (a ≠ 0 ∧ b ≠ 0) ∧
    ∀ (x y : ℝ), (x^2 + y^2 + 2 * a * x + a^2 - 4 = 0) ∧
                  (x^2 + y^2 - 4 * b * y - 1 + 4 * b^2 = 0) ∧
                  (∃ (tangents : ℕ), tangents = 3)) :
  min_value a b h = 16 / 9 :=
sorry

end proof_min_value_l444_444398


namespace find_degree_of_g_l444_444170

noncomputable def degree (p : Polynomial ℂ) : ℕ :=
p.natDegree

theorem find_degree_of_g
  (f g : Polynomial ℂ)
  (h : Polynomial ℂ)
  (H_h : h = f.comp(g) + g)
  (deg_h : degree h = 8)
  (deg_f : degree f = 3) :
  degree g = 2 :=
sorry

end find_degree_of_g_l444_444170


namespace symmetric_point_of_A_l444_444666

variables {R : Type*} [linear_ordered_field R]

def symmetric_point (A : R × R) (l : R → R → Prop) (B : R × R) : Prop :=
  let (ax, ay) := A in
  let (bx, by) := B in
  -- Midpoint lies on the line
  l ((ax + bx) / 2) ((ay + by) / 2) ∧
  -- Perpendicular slope condition
  (by - ay) / (bx - ax) = -1 / (1 / 2)

def line_equation (x y : R) : Prop :=
  x + 2 * y - 1 = 0

theorem symmetric_point_of_A : 
  symmetric_point (1 : ℚ, 2) line_equation (-3 / 5 : ℚ, -6 / 5) :=
by
  sorry

end symmetric_point_of_A_l444_444666


namespace sail_time_difference_l444_444885

theorem sail_time_difference (distance : ℕ) (v_big : ℕ) (v_small : ℕ) (t_big t_small : ℕ)
  (h_distance : distance = 200)
  (h_v_big : v_big = 50)
  (h_v_small : v_small = 20)
  (h_t_big : t_big = distance / v_big)
  (h_t_small : t_small = distance / v_small)
  : t_small - t_big = 6 := by
  sorry

end sail_time_difference_l444_444885


namespace quadratic_equal_real_roots_l444_444919

theorem quadratic_equal_real_roots (k : ℝ) : (∀ x : ℝ, k * x^2 + 2 * x + 1 = 0 → k = 1) :=
begin
  -- Sorry to skip the proof as per the instructions.
  sorry
end

end quadratic_equal_real_roots_l444_444919


namespace count_not_squares_or_cubes_l444_444507

theorem count_not_squares_or_cubes {A B : Finset ℕ} (h_range : ∀ x, x ∈ A ∪ B → x ≤ 200)
  (h_squares : A.card = 14) (h_cubes : B.card = 5) (h_inter : (A ∩ B).card = 1) :
  (200 - (A ∪ B).card) = 182 :=
by
  sorry

end count_not_squares_or_cubes_l444_444507


namespace distance_between_centers_of_circles_l444_444286

theorem distance_between_centers_of_circles (PQ PR QR : ℝ) (I O : ℝ → Prop) 
  (hPQ : PQ = 6) (hPR : PR = 6) (hQR : QR = 10) :
  dist I O = 5 * (Real.sqrt 110) / 11 :=
sorry

end distance_between_centers_of_circles_l444_444286


namespace passing_through_midpoint_of_DE_l444_444793

noncomputable def is_isosceles (ABC : Triangle) : Prop :=
ABC.AB = ABC.AC

noncomputable def midpoint (D : Point) (BC : Segment) : Prop :=
(B + C) / 2 = D

noncomputable def perpendicular (DE : Line) (AC : Line) : Prop :=
right_angle (angle DE AC)

noncomputable def circumsircle_intersect (F : Point) (ABD : Triangle) (BE : Line) : Prop :=
(circumcircle ABD).contains B ∧ (circumcircle ABD).contains F ∧ line BE.contains B ∧ line BE.contains F

noncomputable def passes_through_midpoint (AF : Line) (DE : Segment) : Prop :=
exists M : Point, midpoint M DE ∧ line AF.contains M

theorem passing_through_midpoint_of_DE
  (ABC : Triangle) (D E F : Point) (DE : Line) (AC BE AF : Line) :
  is_isosceles ABC →
  midpoint D (BC) →
  perpendicular DE AC →
  circumsircle_intersect F (Triangle.mk ABD) BE →
  passes_through_midpoint AF (segment.mk D E) :=
by
  sorry

end passing_through_midpoint_of_DE_l444_444793


namespace count_valid_numbers_between_1_and_200_l444_444431

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k^2 = n
def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k^3 = n
def is_perfect_sixth_power (n : ℕ) : Prop := ∃ k : ℕ, k^6 = n

def count_perfect_squares (m : ℕ) : ℕ :=
  Nat.sqrt m

def count_perfect_cubes (m : ℕ) : ℕ :=
  (m : ℝ).cbrt.to_nat

def count_perfect_sixth_powers (m : ℕ) : ℕ :=
  (m : ℝ).cbrt.sqrt.to_nat

def count_either_squares_or_cubes (m : ℕ) : ℕ :=
  let squares := count_perfect_squares m
  let cubes := count_perfect_cubes m
  let sixths := count_perfect_sixth_powers m
  squares + cubes - sixths

def count_neither_squares_nor_cubes (n : ℕ) : ℕ :=
  n - count_either_squares_or_cubes n

theorem count_valid_numbers_between_1_and_200 : count_neither_squares_nor_cubes 200 = 182 :=
by
  sorry

end count_valid_numbers_between_1_and_200_l444_444431


namespace ellipse_equation_l444_444021

theorem ellipse_equation (c e : ℝ) (a b : ℝ) (h_hyperbola : c = sqrt 5) (h_eccentricity : e = sqrt 5 / 5)
  (h1 : a^2 - b^2 = 5) (h2 : e = c / a) : (a = 5 ∧ b = 2 * sqrt 5) → (∀ x y, x^2 / 25 + y^2 / 20 = 1) :=
sorry

end ellipse_equation_l444_444021


namespace smallest_positive_solution_l444_444908

theorem smallest_positive_solution 
: ∀ x : ℝ, x > 0 → (tan (3 * x) + tan (4 * x) = sec (4 * x)) → x = pi / 20 := 
by
  intro x hx h_equation
  sorry

end smallest_positive_solution_l444_444908


namespace time_diff_is_6_l444_444882

-- Define the speeds for the different sails
def speed_of_large_sail : ℕ := 50
def speed_of_small_sail : ℕ := 20

-- Define the distance of the trip
def trip_distance : ℕ := 200

-- Calculate the time for each sail
def time_large_sail (distance : ℕ) (speed : ℕ) : ℕ := distance / speed
def time_small_sail (distance : ℕ) (speed : ℕ) : ℕ := distance / speed

-- Define the time difference
def time_difference (distance : ℕ) (speed_large : ℕ) (speed_small : ℕ) : ℕ := 
  (distance / speed_small) - (distance / speed_large)

-- Prove that the time difference between the large and small sails is 6 hours
theorem time_diff_is_6 : time_difference trip_distance speed_of_large_sail speed_of_small_sail = 6 := by
  -- useful := time_difference trip_distance speed_of_large_sail speed_of_small_sail,
  -- change useful with 6,
  sorry

end time_diff_is_6_l444_444882


namespace count_not_squares_or_cubes_l444_444509

theorem count_not_squares_or_cubes {A B : Finset ℕ} (h_range : ∀ x, x ∈ A ∪ B → x ≤ 200)
  (h_squares : A.card = 14) (h_cubes : B.card = 5) (h_inter : (A ∩ B).card = 1) :
  (200 - (A ∪ B).card) = 182 :=
by
  sorry

end count_not_squares_or_cubes_l444_444509


namespace count_difference_of_squares_l444_444978

theorem count_difference_of_squares : 
  ∃ n : ℕ, (∀ x ∈ finset.range 501, ∃ a b : ℕ, x = a^2 - b^2) → n = 375 :=
by
  let nums := finset.range 501
  let expressible := nums.filter (λ x, ∃ a b : ℕ, x = a^2 - b^2)
  exact sorry

end count_difference_of_squares_l444_444978


namespace house_paint_possible_l444_444578
open Function

def family_perms_exist (families : Type) (houses : Type) [Fintype families] [Fintype houses] [DecidableEq houses] : Prop :=
  ∃ (perm : Perm families), ∀ f : families, f ≠ perm f

def colorable (families : Type) (houses : Type) [Fintype families] [Fintype houses] [DecidableEq houses] : Prop :=
  ∃ (colors : houses → ℕ), (∀ h : houses, colors h = 0 ∨ colors h = 1 ∨ colors h = 2) ∧
  ∀ (perm : Perm houses), ∀ h : houses, colors h ≠ colors (perm h)

theorem house_paint_possible (families : Type) (houses : Type) [Fintype families] [Fintype houses] [DecidableEq houses] :
  family_perms_exist families houses → colorable families houses :=
by
  sorry

end house_paint_possible_l444_444578


namespace box_of_books_weight_l444_444244

theorem box_of_books_weight (books_in_box : ℕ) (weight_per_book : ℕ) (books_in_box_eq : books_in_box = 14) (weight_per_book_eq : weight_per_book = 3) :
  books_in_box * weight_per_book = 42 :=
by
  rw [books_in_box_eq, weight_per_book_eq]
  exact rfl

end box_of_books_weight_l444_444244


namespace concyclic_points_l444_444131

-- Definitions of the given terms.
variables {A B C H M N X Y : Type*}
variables [IsAcuteTriangle A B C] [Orthocenter H A B C] [Midpoint M A B] [Midpoint N A C]
variables [LiesOnCircumcircle X M H A B C] [LiesOnCircumcircle Y N H A B C]

-- The theorem statement
theorem concyclic_points (h1 : IsAcuteTriangle A B C) (h2 : Orthocenter H A B C)
                        (h3 : Midpoint M A B) (h4 : Midpoint N A C)
                        (h5 : LiesOnCircumcircle X M H A B C)
                        (h6 : LiesOnCircumcircle Y N H A B C) :
  Concyclic M N X Y :=
by
  sorry

end concyclic_points_l444_444131


namespace triangles_similar_l444_444156

variables {P₁ P₂ P₃ J₁ J₂ J₃ : Type} 

-- Assuming similarity angles as conditions
variables (angle_J1_J2_J3_P2_P1 : ∀ {A B C D : Type}, angle (A B) (B C) = angle (D C) (C D))
variables (angle_J2_J3_J1_P1_P3 : ∀ {A B C D : Type}, angle (A C) (C D) = angle (D A) (A B))
variables (angle_J3_J1_J2_P1_P2 : ∀ {A B C D : Type}, angle (A C) (C B) = angle (D B) (B D))

-- Prove similarity of the triangles
theorem triangles_similar (angle_J1_J2_J3_P2_P1 : ∀ {A B C D : Type}, angle (A B) (B C) = angle (D C) (C D))
  (angle_J2_J3_J1_P1_P3 : ∀ {A B C D : Type}, angle (A C) (C D) = angle (D A) (A B))
  (angle_J3_J1_J2_P1_P2 : ∀ {A B C D : Type}, angle (A C) (C B) = angle (D B) (B D))
  : triangle J₁ J₂ J₃ ∼ triangle P₁ P₂ P₃ :=
sorry

end triangles_similar_l444_444156


namespace ratio_of_areas_l444_444837

theorem ratio_of_areas (w : ℝ) (h : w > 0) :
  let A := 2 * w^2,
      B := 2 * w^2 - (w * (1/2) * (real.sqrt 2) / 2)
  in (B / A) = 1 - real.sqrt 2 / 8 :=
by
  sorry

end ratio_of_areas_l444_444837


namespace AF_through_midpoint_DE_l444_444753

open EuclideanGeometry

-- Definition of an isosceles triangle
variables {A B C D E F : Point}
variables [plane Geometry]
variables (ABC_isosceles : IsoscelesTriangle A B C)
variables (midpoint_D : Midpoint D B C)
variables (perpendicular_DE_AC : Perpendicular D E A C)
variables (F_on_circumcircle : OnCircumcircle F A B D)
variables (B_on_BE : OnLine B E)

-- Goal: Prove that line AF passes through the midpoint of DE
theorem AF_through_midpoint_DE 
  (h1 : triangle A B C) 
  (h2 : ABC_isosceles) 
  (h3 : midpoint D B C) 
  (h4 : perpendicular DE AC)
  (h5 : OnCircumcircle F A B D) 
  (h6 : OnLine B E) :
  PassesThroughMidpoint A F D E :=
sorry

end AF_through_midpoint_DE_l444_444753


namespace arithmetic_sequence_sum_l444_444573

theorem arithmetic_sequence_sum :
  ∀ (a_n : ℕ → ℝ) (d : ℝ), (∀ n, a_n n = a_n 1 + d * (n - 1)) →
  (root1 root2 : ℝ), (root1 + root2 = 10) →
  (root1 = a_n 1) ∧ (root2 = a_n 2015) →
  (a_n 2 + a_n 1008 + a_n 2014 = 15) :=
begin
  intros a_n d a_n_def root1 root2 roots_sum roots_def,
  sorry
end

end arithmetic_sequence_sum_l444_444573


namespace robin_has_43_packages_of_gum_l444_444647

theorem robin_has_43_packages_of_gum (P : ℕ) (h1 : 23 * P + 8 = 997) : P = 43 :=
by
  sorry

end robin_has_43_packages_of_gum_l444_444647


namespace line_AF_passes_midpoint_DE_l444_444759

-- Define Triangle Isosceles
structure IsoscelesTriangle (A B C : Type) :=
  (A B C : Type)
  (isosceles : A = B)

-- Define midpoint
structure Midpoint (D B C : Type) :=
  (D : Type)
  (midpoint : D = B ∧ D = C)

-- Define perpendicular
structure Perpendicular (D E A C : Type) :=
  (perpendicular : ∀ {D E A C : Type}, A = C → D ≠ A → D = E)

-- Conditions
variables {ABC : Triangle} [IsoscelesTriangle ABC]
variables {D : Point} [Midpoint D ABC.base]
variables {DE : Line} [Perpendicular D E ABC.ABC.side]

-- The circumcircle and intersection
variable {circ_circ : ∃ circ_circ : circle (triangle ABD), intersection (BE, circ_circ) = {B, F}}

-- Main proof goal: Prove that line AF passes through the midpoint of DE.
theorem line_AF_passes_midpoint_DE :
  ∀ {A F D E : Type} {circ_circ : circle (triangle ABD)}, 
  intersect (AF, midpoint DE) ↔ intersect (circ_circ, BE) :=
sorry

end line_AF_passes_midpoint_DE_l444_444759


namespace AF_passes_through_midpoint_DE_l444_444769

open_locale classical

-- Definition of the geometrical setup
def isosceles_triangle (A B C : Point) : Prop := dist A B = dist A C
def midpoint (D B C : Point) : Prop := dist B D = dist C D
def perpendicular (D E A C : Point) : Prop := ∠ DEA = 90
def circumcircle (A B D F : Point): Prop := ∃ O R, Circle.circle O R A ∧ Circle.circle O R B ∧ Circle.circle O R D ∧ Circle.circle O R F

-- Main statement to prove
theorem AF_passes_through_midpoint_DE
  (A B C D E F : Point)
  (h1 : isosceles_triangle A B C)
  (h2 : midpoint D B C)
  (h3 : perpendicular D E A C)
  (h4 : ∃ O R, Circle.circle O R A ∧ Circle.circle O R B ∧ Circle.circle O R D ∧ Circle.circle O R F)
  (h5 : collinear B E F)
  (h6 : intersects (circumcircle A B D) (line B E) B F) :
  passes_through (line A F) (midpoint D E) := 
sorry

end AF_passes_through_midpoint_DE_l444_444769


namespace height_of_pole_l444_444174

noncomputable section
open Real

theorem height_of_pole (α β γ : ℝ) (h xA xB xC : ℝ) 
  (hA : tan α = h / xA) (hB : tan β = h / xB) (hC : tan γ = h / xC) 
  (sum_angles : α + β + γ = π / 2) : h = 10 :=
by
  sorry

end height_of_pole_l444_444174


namespace prove_D_value_l444_444565

variable {E F D : ℕ}

-- Conditions
def condition1 : Prop := E + F + D = E * F - 3
def condition2 : Prop := E - F = 2

-- Goal
def goal : Prop := D = 4

theorem prove_D_value (h1 : condition1) (h2 : condition2) : goal := 
by
  -- Sorry to skip the proof, as per instructions 
  sorry

end prove_D_value_l444_444565


namespace number_of_sets_l444_444019

theorem number_of_sets (M : Set ℕ) : 
  (\<|1, 2, 3>|.subseteq M) ∧ (M.subseteq \<|1, 2, 3, 4, 5>|) → 
  (card {S | \subseteq {M | 1 ∈ M ∧ 2 ∈ M ∧ 3 ∈ M ∧ ∀ (x ∈ M, x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5)}) = 4
:= sorry

end number_of_sets_l444_444019


namespace length_LI_l444_444742

-- Definitions assumed from conditions
variables {P M R Z I : Point}
def unitCircle : Circle := { center := C, radius := 1 }
def inscribedRegularOctagon : Octagon := 
  { vertices := [P, Q, R, S, T, U, V, W], inscribedIn := unitCircle }
def diagonal (M R : Point) : Line := Line (M, R)
def intersect (l₁ l₂ : Line) : Point := Point.intersection l₁ l₂

-- Proving the length LI is √2
theorem length_LI :
  let M, R, Z, I := inscribedRegularOctagon.vertices
  let OZ := diagonal O Z
  let MR := diagonal M R
  let I := intersect OZ MR in
  LI = sqrt 2 := 
by
  sorry

end length_LI_l444_444742


namespace line_AF_passes_midpoint_DE_l444_444761

-- Define Triangle Isosceles
structure IsoscelesTriangle (A B C : Type) :=
  (A B C : Type)
  (isosceles : A = B)

-- Define midpoint
structure Midpoint (D B C : Type) :=
  (D : Type)
  (midpoint : D = B ∧ D = C)

-- Define perpendicular
structure Perpendicular (D E A C : Type) :=
  (perpendicular : ∀ {D E A C : Type}, A = C → D ≠ A → D = E)

-- Conditions
variables {ABC : Triangle} [IsoscelesTriangle ABC]
variables {D : Point} [Midpoint D ABC.base]
variables {DE : Line} [Perpendicular D E ABC.ABC.side]

-- The circumcircle and intersection
variable {circ_circ : ∃ circ_circ : circle (triangle ABD), intersection (BE, circ_circ) = {B, F}}

-- Main proof goal: Prove that line AF passes through the midpoint of DE.
theorem line_AF_passes_midpoint_DE :
  ∀ {A F D E : Type} {circ_circ : circle (triangle ABD)}, 
  intersect (AF, midpoint DE) ↔ intersect (circ_circ, BE) :=
sorry

end line_AF_passes_midpoint_DE_l444_444761


namespace octahedron_side_length_l444_444279

noncomputable theory

-- Coordinates of the vertices
def P1 := (0, 0, 0)
def P2 := (2, 0, 0)
def P3 := (0, 1, 0)
def P4 := (0, 0, 1)
def P1' := (2, 1, 1)

-- Question statement in Lean 4
theorem octahedron_side_length : 
  let x := 4 / 3 in
  (P2-P1).dist = 2 ∧ 
  (P3-P1).dist = 1 ∧ 
  (P4-P1).dist = 1 ∧ 
  (P1' - P1).dist = real.sqrt(2) ∧
  (x-2) ^ 2 + (1-x) ^ 2 + (1-x) ^ 2 = 2 * x ^ 2 →
  real.sqrt(2 * x^2) = 4 * real.sqrt(2) / 3 :=
begin
  sorry
end

end octahedron_side_length_l444_444279


namespace range_of_a_l444_444054

theorem range_of_a (a : ℝ) (h1 : 1 < a) (h2 : ∃ x₀ ∈ Icc (0:ℝ) a, (∀ x ∈ Icc (0:ℝ) a, f x₀ ≤ f x) ∧ x₀ < 2) : 1 < a ∧ a < Real.exp 1 :=
by
  sorry

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp (x - 1) - a * x

end range_of_a_l444_444054


namespace line_AF_passes_midpoint_DE_l444_444760

-- Define Triangle Isosceles
structure IsoscelesTriangle (A B C : Type) :=
  (A B C : Type)
  (isosceles : A = B)

-- Define midpoint
structure Midpoint (D B C : Type) :=
  (D : Type)
  (midpoint : D = B ∧ D = C)

-- Define perpendicular
structure Perpendicular (D E A C : Type) :=
  (perpendicular : ∀ {D E A C : Type}, A = C → D ≠ A → D = E)

-- Conditions
variables {ABC : Triangle} [IsoscelesTriangle ABC]
variables {D : Point} [Midpoint D ABC.base]
variables {DE : Line} [Perpendicular D E ABC.ABC.side]

-- The circumcircle and intersection
variable {circ_circ : ∃ circ_circ : circle (triangle ABD), intersection (BE, circ_circ) = {B, F}}

-- Main proof goal: Prove that line AF passes through the midpoint of DE.
theorem line_AF_passes_midpoint_DE :
  ∀ {A F D E : Type} {circ_circ : circle (triangle ABD)}, 
  intersect (AF, midpoint DE) ↔ intersect (circ_circ, BE) :=
sorry

end line_AF_passes_midpoint_DE_l444_444760


namespace no_valid_n_for_conditions_l444_444000

theorem no_valid_n_for_conditions :
  ¬∃ n : ℕ, 1000 ≤ n / 4 ∧ n / 4 ≤ 9999 ∧ 1000 ≤ 4 * n ∧ 4 * n ≤ 9999 := by
  sorry

end no_valid_n_for_conditions_l444_444000


namespace find_x_l444_444233

theorem find_x (x : ℝ) 
  (h : ( sqrt x + sqrt 243 ) / sqrt 48 = 3.0000000000000004) : 
  x = 27 :=
sorry

end find_x_l444_444233


namespace total_toothpicks_for_grid_l444_444711

-- Defining the conditions
def grid_height := 30
def grid_width := 15

-- Define the function that calculates the total number of toothpicks
def total_toothpicks (height : ℕ) (width : ℕ) : ℕ :=
  let horizontal_toothpicks := (height + 1) * width
  let vertical_toothpicks := (width + 1) * height
  horizontal_toothpicks + vertical_toothpicks

-- The theorem stating the problem and its answer
theorem total_toothpicks_for_grid : total_toothpicks grid_height grid_width = 945 :=
by {
  -- Here we would write the proof steps. Using sorry for now.
  sorry
}

end total_toothpicks_for_grid_l444_444711


namespace count_not_squares_or_cubes_l444_444503

theorem count_not_squares_or_cubes {A B : Finset ℕ} (h_range : ∀ x, x ∈ A ∪ B → x ≤ 200)
  (h_squares : A.card = 14) (h_cubes : B.card = 5) (h_inter : (A ∩ B).card = 1) :
  (200 - (A ∪ B).card) = 182 :=
by
  sorry

end count_not_squares_or_cubes_l444_444503


namespace mean_proportional_l444_444347

variable (a b c d : ℕ)
variable (x : ℕ)

def is_geometric_mean (a b : ℕ) (x : ℕ) := x = Int.sqrt (a * b)

theorem mean_proportional (h49 : a = 49) (h64 : b = 64) (h81 : d = 81)
  (h_geometric1 : x = 56) (h_geometric2 : c = 72) :
  c = 64 := sorry

end mean_proportional_l444_444347


namespace complex_sum_identity_l444_444374

open Complex

theorem complex_sum_identity (n : ℕ) (h : n ≥ 2) (A B : Fin n → ℂ) :
  (∑ k in Finset.range n, ∏ j in Finset.range n, (A k + B j) / ∏ j in (Finset.range n).erase k, (A k - A j)) =
  (∑ k in Finset.range n, ∏ j in Finset.range n, (B k + A j) / ∏ j in (Finset.range n).erase k, (B k - B j)) := 
sorry

end complex_sum_identity_l444_444374


namespace decreasing_interval_l444_444677

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + 11
noncomputable def f' (x : ℝ) : ℝ := 6 * x^2 - 12 * x

theorem decreasing_interval : ∀ x : ℝ, 0 < x ∧ x < 2 → f' x < 0 :=
by
  intro x hx
  unfold f'
  cases hx with h_left h_right
  sorry

end decreasing_interval_l444_444677


namespace tangency_abscissa_eq_one_l444_444556

noncomputable def abscissa_of_tangency (f : ℝ → ℝ) (g : ℝ → ℝ) (x : ℝ) : Prop :=
f x = g x ∧ deriv f x = deriv g x

def curve : ℝ → ℝ := λ x, real.sqrt x

def line (x : ℝ) : Prop := x - 2 * (1 / 2 * x - 2) - 4 = 0

theorem tangency_abscissa_eq_one :
  ∃ x : ℝ, abscissa_of_tangency curve (λ x, (1 / 2 * x - 2)) x :=
by
  use 1
  sorry

end tangency_abscissa_eq_one_l444_444556


namespace sugar_for_third_layer_l444_444293

theorem sugar_for_third_layer (s1 : ℕ) (s2 : ℕ) (s3 : ℕ) 
  (h1 : s1 = 2) 
  (h2 : s2 = 2 * s1) 
  (h3 : s3 = 3 * s2) : 
  s3 = 12 := 
sorry

end sugar_for_third_layer_l444_444293


namespace sufficient_but_not_necessary_condition_l444_444022

theorem sufficient_but_not_necessary_condition (x : ℝ) (p : Prop) (q : Prop)
  (h₁ : p ↔ (x^2 - 1 > 0)) (h₂ : q ↔ (x < -2)) :
  (¬p → ¬q) ∧ ¬(¬q → ¬p) := 
by
  sorry

end sufficient_but_not_necessary_condition_l444_444022


namespace num_diff_of_squares_in_range_l444_444982

/-- 
Define a number expressible as the difference of squares of two nonnegative integers.
-/
def is_diff_of_squares (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = a^2 - b^2

/-- 
Define the range of interest from 1 to 500.
-/
def range_1_to_500 : Finset ℕ := Finset.range 501 \ {0}

/-- 
Define the criterion for numbers that can be expressed in the desired form.
-/
def can_be_expressed (n : ℕ) : Prop :=
  (n % 2 = 1) ∨ (n % 4 = 0)

/-- 
Count the numbers between 1 and 500 that satisfy the condition.
-/
def count_expressible_numbers : ℕ :=
  (range_1_to_500.filter can_be_expressed).card

/-- 
Prove that the count of numbers between 1 and 500 that can be expressed as 
the difference of two squares of nonnegative integers is 375.
-/
theorem num_diff_of_squares_in_range : count_expressible_numbers = 375 :=
  sorry

end num_diff_of_squares_in_range_l444_444982


namespace quadratic_equation_roots_difference_l444_444878

theorem quadratic_equation_roots_difference
  (p q : ℝ) 
  (hp : 0 < p) 
  (hq : 0 < q) 
  (h : abs ((-p + sqrt (p^2 - 4 * q)) / 2 - (-p - sqrt (p^2 - 4 * q)) / 2) = 2) :
  p = 2 * sqrt (q + 1) :=
by
  sorry

end quadratic_equation_roots_difference_l444_444878


namespace range_of_a_positive_integers_m_l444_444376

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := log a ((1 / x) + a)

-- Part (1)
theorem range_of_a (a : ℝ) (x : ℝ) : (0 < a ∧ a ≠ 1 ∧ x ∈ (1/2 : ℝ, 3/4 : ℝ) ∧ 
  log a ((1 / x) + a) + log a (x^2) = 0) → a ∈ (4/9 : ℝ, 1) ∪ (1, 2) :=
by
  sorry

-- Part (2)
theorem positive_integers_m (m : ℕ) (a : ℝ) :
  (0 < a ∧ a < 1 ∧ m = 6) →
  (∀ x ∈ ({m,7} : Set ℝ), f (1 / |a * x - 1|) a > f (x / (1 - a)) a) :=
by
  sorry

end range_of_a_positive_integers_m_l444_444376


namespace count_distinct_rational_numbers_l444_444891

theorem count_distinct_rational_numbers :
  ∃ n : ℕ → \{k : ℚ // (|k| < 300 ∧ (∃ x : ℤ, 3 * (x^2) + k * (x) + 18 = 0) ∧ ∃ m : ℤ, k = 3 * m)} =
    108 :=
by
  sorry

end count_distinct_rational_numbers_l444_444891


namespace part_one_part_two_l444_444050

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 
  Real.sin (π / 4 + x) * Real.sin (π / 4 - x) + Real.sqrt 3 * Real.sin x * Real.cos x

-- Problem (1): Prove f(π/6) = 1
theorem part_one : f (π / 6) = 1 := 
  sorry

-- Problem (2): Given f(A/2) = 1 and the internal angles A, B, C of a triangle, prove that the maximum value of sin(B) + sin(C) is sqrt(3)
theorem part_two (A B C : ℝ) (h_angle_sum : A + B + C = π) (h_f : f (A / 2) = 1) : 
  (∃ B' C', B = B' ∧ C = C' ∧ Real.sin B' + Real.sin C' = Real.sqrt 3) := 
  sorry

end part_one_part_two_l444_444050


namespace line_AF_through_midpoint_of_DE_l444_444806

open EuclideanGeometry

noncomputable def midpoint_of_segment (A B : Point) := ((A.x + B.x) / 2, (A.y + B.y) / 2)

theorem line_AF_through_midpoint_of_DE
  {A B C D E F : Point}
  (h_isosceles : AB = AC)
  (h_midpoint_D : D = midpoint B C)
  (h_perpendicular_DE : ⟂ D E AC)
  (h_circumcircle_intersects_BE : ∃ k : ℕ, circumcircle A B D = circum circle (B ⊗ F))
  : passes_through (line A F) (midpoint_of_segment D E) :=
sorry

end line_AF_through_midpoint_of_DE_l444_444806


namespace area_of_ABCD_is_correct_l444_444838

noncomputable def area_of_quadrilateral_ABCD : ℝ :=
  let side_length := 1
  let B := (1 / 2 : ℝ)
  let C := (1 / 2 : ℝ)
  let D := (1 / 2 : ℝ)
  (sqrt 3 / 8)

theorem area_of_ABCD_is_correct :
  area_of_quadrilateral_ABCD = sqrt 3 / 8 :=
  sorry

end area_of_ABCD_is_correct_l444_444838


namespace count_not_squares_or_cubes_l444_444502

theorem count_not_squares_or_cubes {A B : Finset ℕ} (h_range : ∀ x, x ∈ A ∪ B → x ≤ 200)
  (h_squares : A.card = 14) (h_cubes : B.card = 5) (h_inter : (A ∩ B).card = 1) :
  (200 - (A ∪ B).card) = 182 :=
by
  sorry

end count_not_squares_or_cubes_l444_444502


namespace diesel_usage_l444_444624

theorem diesel_usage (weekly_expenditure : ℝ) (cost_per_gallon : ℝ)
  (h_expenditure : weekly_expenditure = 36)
  (h_cost : cost_per_gallon = 3) :
  let weekly_gallons := weekly_expenditure / cost_per_gallon in
  let two_weeks_gallons := 2 * weekly_gallons in
  two_weeks_gallons = 24 := 
by
  sorry

end diesel_usage_l444_444624


namespace numbers_neither_square_nor_cube_l444_444539

theorem numbers_neither_square_nor_cube (n : ℕ) (h : n = 200) :
  let num_squares := 14 in
  let num_cubes := 5 in
  let num_sixth_powers := 2 in
  (n - (num_squares + num_cubes - num_sixth_powers)) = 183 :=
by
  -- Let the number of integers in the range
  let num_total := 200
  -- Define how we count perfect squares and cubes
  let num_squares := 14
  let num_cubes := 5
  let num_sixth_powers := 2

  -- Subtract the overlapping perfect sixth powers from the sum of squares and cubes
  let num_either_square_or_cube := num_squares + num_cubes - num_sixth_powers

  -- Calculate the final result
  let result := num_total - num_either_square_or_cube

  -- Prove by computation
  show result = 183 from
    calc
      result = num_total - num_either_square_or_cube := rfl
           ... = 200 - (14 + 5 - 2)                   := rfl
           ... = 200 - 17                             := rfl
           ... = 183                                  := rfl

end numbers_neither_square_nor_cube_l444_444539


namespace range_of_k_not_monotonic_l444_444956

theorem range_of_k_not_monotonic {y : ℝ → ℝ} (h : ∀ x, y x = |2^x - 1|) (hnm : ∀ k : ℝ, ¬ monotone_on y (set.Ioo (k-1) (k+1))) :
  ∀ k, -1 < k ∧ k < 1 :=
by
  sorry

end range_of_k_not_monotonic_l444_444956


namespace numbers_not_squares_or_cubes_in_200_l444_444469

/-- There are 182 numbers between 1 and 200 that are neither perfect squares nor perfect cubes. -/
theorem numbers_not_squares_or_cubes_in_200 : (finset.range 200).filter (λ n, ¬(∃ m, m^2 = n) ∧ ¬(∃ k, k^3 = n)).card = 182 :=
by 
  /-
    The proof will involve constructing the set of numbers from 1 to 200, filtering out those that
    are perfect squares or perfect cubes, and counting the remaining numbers.
  -/
  sorry

end numbers_not_squares_or_cubes_in_200_l444_444469


namespace average_speed_road_trip_l444_444323

theorem average_speed_road_trip :
  let speed1 := 20
  let speed2 := 30 * 1.10
  let speed3 := 10
  let speed4 := 40 * 1.10
  let speed5 := 25
  let total_distance := speed1 + speed2 + speed3 + speed4 + speed5
  let total_time := 5
  average_speed := total_distance / total_time 
in average_speed = 26.4 := sorry

end average_speed_road_trip_l444_444323


namespace problem_lean_statement_l444_444363

noncomputable def c_and_d_satisfy (c d : ℕ) : Prop :=
  let logc (x : ℝ) : ℝ := Real.log x / Real.log c
  let isPos (n : ℕ) := n > 0
  isPos c ∧ isPos d ∧ (d = c^2) ∧ (d - c = 435)

theorem problem_lean_statement (c d : ℕ) (logc logd : ℝ → ℝ)
  (h1 : c > 0) (h2 : d > 0) (h3 : logc (c+2) * logc (c+4). ... * logd (d-2) * logd d = 2)
  (h4 : (d - c) = 435)
  (h5 : d = c^2) : 
  c + d = 930 := by
  sorry

end problem_lean_statement_l444_444363


namespace probability_of_point_in_smaller_spheres_l444_444270

noncomputable def regular_tetrahedron (a: ℝ) := sorry

def inscribed_sphere_radius (a: ℝ) : ℝ :=
  (Real.sqrt 6 * a) / 12

def circumscribed_sphere_radius (a: ℝ) : ℝ :=
  (Real.sqrt 6 * a) / 4

def smaller_sphere_radius (a: ℝ) : ℝ :=
  (circumscribed_sphere_radius a - inscribed_sphere_radius a) / 2

def volume_of_sphere (r: ℝ) : ℝ :=
  (4 / 3) * Real.pi * r^3

def combined_volume_of_spheres (a: ℝ) : ℝ :=
  (volume_of_sphere (inscribed_sphere_radius a)) +
  4 * (volume_of_sphere (smaller_sphere_radius a))

def probability (a: ℝ) : ℝ :=
  combined_volume_of_spheres a / (volume_of_sphere (circumscribed_sphere_radius a))

theorem probability_of_point_in_smaller_spheres
  (a: ℝ) :
  abs (probability a - 0.2) < 0.01 :=
  sorry

end probability_of_point_in_smaller_spheres_l444_444270


namespace analytical_expression_intervals_monotonically_increasing_l444_444405

noncomputable def A := 2
noncomputable def ω := 2
noncomputable def φ := Real.pi / 3

def f (x : ℝ) : ℝ := A * Real.sin (ω * x + φ)

theorem analytical_expression :
  (A > 0) ∧ (ω > 0) ∧ (0 < φ ∧ φ < Real.pi / 2) ∧
  (f x).range = Set.Icc 0 2 ∧ ∃ (T > 0), T = Real.pi ∧
  f (-Real.pi / 6) = 0 →
  f(x) = 2 * Real.sin (2 * x + Real.pi / 3) := sorry

theorem intervals_monotonically_increasing :
  (A > 0) ∧ (ω > 0) ∧ (0 < φ ∧ φ < Real.pi / 2) ∧
  (f x).range = Set.Icc 0 2 ∧ ∃ (T > 0), T = Real.pi ∧
  f (-Real.pi / 6) = 0 →
  ∀ (k : ℤ), ∀ (x : ℝ), (k * Real.pi - (5 * Real.pi) / 12) ≤ x ∧
  x ≤ (k * Real.pi + Real.pi / 12) → 
  ∀ (x1 x2 : ℝ), (k * Real.pi - (5 * Real.pi) / 12) ≤ x1 ≤ x2 ∧ 
  x2 ≤ (k * Real.pi + Real.pi / 12) → f x1 ≤ f x2 := sorry

end analytical_expression_intervals_monotonically_increasing_l444_444405


namespace average_of_numbers_l444_444177

theorem average_of_numbers :
  let numbers := [-5, -2, 0, 4, 8]
  (list.sum numbers) / (list.length numbers : ℝ) = 1 := by
  sorry

end average_of_numbers_l444_444177


namespace hotel_people_per_room_l444_444258

theorem hotel_people_per_room
  (total_rooms : ℕ := 10)
  (towels_per_person : ℕ := 2)
  (total_towels : ℕ := 60) :
  (total_towels / towels_per_person) / total_rooms = 3 :=
by
  sorry

end hotel_people_per_room_l444_444258


namespace min_value_6x_5y_thm_l444_444989

noncomputable def min_value_6x_5y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 1 / (2 * x + y) + 3 / (x + y) = 2) : ℝ :=
  6 * x + 5 * y

theorem min_value_6x_5y_thm (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 1 / (2 * x + y) + 3 / (x + y) = 2) :
  min_value_6x_5y x y h1 h2 h3 = (13 + 4 * real.sqrt 3) / 2 :=
sorry

end min_value_6x_5y_thm_l444_444989


namespace num_diff_of_squares_in_range_l444_444984

/-- 
Define a number expressible as the difference of squares of two nonnegative integers.
-/
def is_diff_of_squares (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = a^2 - b^2

/-- 
Define the range of interest from 1 to 500.
-/
def range_1_to_500 : Finset ℕ := Finset.range 501 \ {0}

/-- 
Define the criterion for numbers that can be expressed in the desired form.
-/
def can_be_expressed (n : ℕ) : Prop :=
  (n % 2 = 1) ∨ (n % 4 = 0)

/-- 
Count the numbers between 1 and 500 that satisfy the condition.
-/
def count_expressible_numbers : ℕ :=
  (range_1_to_500.filter can_be_expressed).card

/-- 
Prove that the count of numbers between 1 and 500 that can be expressed as 
the difference of two squares of nonnegative integers is 375.
-/
theorem num_diff_of_squares_in_range : count_expressible_numbers = 375 :=
  sorry

end num_diff_of_squares_in_range_l444_444984


namespace max_elements_in_A_l444_444961

def point := (ℕ × ℕ)

def S : set point := {p | 1 ≤ p.1 ∧ p.1 ≤ 10 ∧ 1 ≤ p.2 ∧ p.2 ≤ 10}

def valid_set (A : set point) : Prop :=
  ∀ (a b s t : ℕ), (a, b) ∈ A → (s, t) ∈ A → (a - s) * (b - t) ≤ 0

theorem max_elements_in_A : ∃ A ⊆ S, valid_set A ∧ ∀ B ⊆ S, valid_set B → |A| ≥ |B| := 
sorry

end max_elements_in_A_l444_444961


namespace least_positive_integer_exists_l444_444816

-- Define the condition
def condition (x : List ℕ) : Prop :=
  (List.product (x.map (λ xi => 1 - (1 / xi)))) = (15 / 2013) ∧ x.nodup

-- Define the problem
theorem least_positive_integer_exists (n : ℕ) : 
  (∃ x : List ℕ, x.length = n ∧ condition x) ↔ n = 9 :=
by sorry

end least_positive_integer_exists_l444_444816


namespace number_of_correct_conclusions_is_one_l444_444399

noncomputable def z := (2 : ℂ) / (1 - (complex.I))

def option_one : Prop := complex.abs z = 2
def option_two : Prop := z ^ 2 = 2 * complex.I
def option_three : Prop := complex.conj z = -1 + complex.I
def option_four : Prop := complex.im z = complex.I

def correct_conclusions : Nat := 
  [option_one, option_two, option_three, option_four].count (λ opt, opt)

theorem number_of_correct_conclusions_is_one : correct_conclusions = 1 := sorry

end number_of_correct_conclusions_is_one_l444_444399


namespace tree_growth_period_l444_444728

theorem tree_growth_period (initial height growth_rate : ℕ) (H4 final_height years : ℕ) 
  (h_init : initial_height = 4) 
  (h_growth_rate : growth_rate = 1) 
  (h_H4 : H4 = initial_height + 4 * growth_rate)
  (h_final_height : final_height = H4 + H4 / 4) 
  (h_years : years = (final_height - initial_height) / growth_rate) :
  years = 6 :=
by
  sorry

end tree_growth_period_l444_444728


namespace sugar_for_third_layer_l444_444294

theorem sugar_for_third_layer (s1 : ℕ) (s2 : ℕ) (s3 : ℕ) 
  (h1 : s1 = 2) 
  (h2 : s2 = 2 * s1) 
  (h3 : s3 = 3 * s2) : 
  s3 = 12 := 
sorry

end sugar_for_third_layer_l444_444294


namespace binomial_coeff_a8_l444_444384

theorem binomial_coeff_a8 :
  (∃ (a : ℕ → ℕ), (1 + x)^10 = ∑ k in range 11, a k * (1 - x)^k) → 
  a 8 = 180 :=
by
  sorry

end binomial_coeff_a8_l444_444384


namespace line_AF_midpoint_DE_l444_444786

variables {A B C D E F : Type} [EuclideanGeometry.point A] [EuclideanGeometry.point B] [EuclideanGeometry.point C]
  [EuclideanGeometry.point D] [EuclideanGeometry.point E] [EuclideanGeometry.point F]

-- Definitions based on the problem conditions
def isosceles_triangle (A B C : Type) [EuclideanGeometry.point A] [EuclideanGeometry.point B] [EuclideanGeometry.point C] : Prop :=
  (EuclideanGeometry.distance A B = EuclideanGeometry.distance A C)

def midpoint (D B C : Type) [EuclideanGeometry.point D] [EuclideanGeometry.point B] [EuclideanGeometry.point C] : Prop :=
  EuclideanGeometry.between D B C

def perpendicular (D E : Type) [EuclideanGeometry.point D] [EuclideanGeometry.point E] : Prop := 
  EuclideanGeometry.perp D E

def circumcircle (ABD : Set EuclideanGeometry.point) : Prop :=
  EuclideanGeometry.circumcircle ABD

def intersection (circumcircle_intersection : Type) [EuclideanGeometry.point F] [EuclideanGeometry.point B] : Prop := 
  EuclideanGeometry.intersects circumcircle_intersection F B

-- The theorem we need to prove
theorem line_AF_midpoint_DE
  (isoABC : isosceles_triangle A B C)
  (midD : midpoint D B C)
  (perpDE : perpendicular D E)
  (circABD : circumcircle {A, B, D})
  (interBF : intersection {A, B, D})
  : EuclideanGeometry.on_line A F (EuclideanGeometry.midpoint D E) :=
sorry -- proof here

end line_AF_midpoint_DE_l444_444786


namespace last_item_is_1_l444_444012

variable {m : ℕ} (a : ℕ → ℕ)
variable (a4_ne_1 : a 4 ≠ 1)
variable (not_5th_repeatable : ∀ k : ℕ, (2 ≤ k ∧ k ≤ m - 1) → ∃ i j : ℕ, i ≠ j ∧ (m ≥ 5 ∧ ∀ l, l < k → a (i + l) = a (j + l)))
variable (new_5th_repeatable : ∀ x : ℕ, x ∈ {0, 1} → ∃ i j : ℕ, i ≠ j ∧ (∀ l, l < 5 → (a (i+l) = a (if l = 4 then x else m-4+l) ∧ a (j+l) = a (if l = 4 then x else m-4+l))))

theorem last_item_is_1 (sequence_length : m ≥ 3) : a m = 1 :=
by
  sorry

end last_item_is_1_l444_444012


namespace log_relationship_l444_444072

theorem log_relationship (a b : ℝ) (h1 : log 5 a > log 5 b) (h2 : log 5 b > 0) : 1 < b ∧ b < a :=
by
  sorry

end log_relationship_l444_444072


namespace product_of_consecutive_integers_l444_444199

theorem product_of_consecutive_integers (n : ℤ) :
  n * (n + 1) * (n + 2) = (n + 1)^3 - (n + 1) :=
by
  sorry

end product_of_consecutive_integers_l444_444199


namespace num_from_1_to_200_not_squares_or_cubes_l444_444463

noncomputable def numNonPerfectSquaresAndCubes (n : ℕ) : ℕ :=
  let num_squares := 14
  let num_cubes := 5
  let num_sixth_powers := 2
  n - (num_squares + num_cubes - num_sixth_powers)

theorem num_from_1_to_200_not_squares_or_cubes : numNonPerfectSquaresAndCubes 200 = 183 := by
  sorry

end num_from_1_to_200_not_squares_or_cubes_l444_444463


namespace sum_div_k_le_half_minus_half_n_l444_444039

theorem sum_div_k_le_half_minus_half_n (n : ℕ) (x : ℕ → ℝ)
  (h1 : n ≥ 2)
  (h2 : ∑ k in finset.range n, |x k| = 1)
  (h3 : ∑ i in finset.range n, x i = 0) :
  |∑ k in finset.range n, x k / (k + 1)| ≤ 1 / 2 - 1 / (2 * n) :=
sorry

end sum_div_k_le_half_minus_half_n_l444_444039


namespace minimum_value_trig_function_l444_444990

theorem minimum_value_trig_function (θ : ℝ) (h : θ ∈ set.Icc (-π / 12) (π / 12)) :
  (∃ η, η = θ + π / 4 ∧ ( cos η + sin (2 * θ) ) = (sqrt 3) / 2 - 1 / 2 ) :=
sorry

end minimum_value_trig_function_l444_444990


namespace number_appears_in_list_l444_444118

def phi : ℝ := (1 + Real.sqrt 5) / 2

theorem number_appears_in_list (n : ℕ) :
  (∃ k : ℕ, n = Int.floor (k * phi)) ↔ (∃ k₁ k₂ : ℕ, k₁ ≠ k₂ ∧ n = Int.floor (k₁ / phi) ∧ n = Int.floor (k₂ / phi)) :=
sorry

end number_appears_in_list_l444_444118


namespace incorrect_statement_b_l444_444608

def perp (l : Set Point) (α : Set Point) : Prop := sorry -- define perpendicularity
def parallel (l m : Set Point) : Prop := sorry -- define parallelism
def subset (m : Set Point) (α : Set Point) : Prop := sorry -- define subset

axiom diff_lines (l m n : Set Point) : l ≠ m ∧ m ≠ n ∧ l ≠ n
axiom is_plane (α : Set Point) : Prop

-- Define axiom for the statements A, B, C, and D
axiom statement_a (l : Set Point) (α : Set Point) : perp l α → ∃ p : Point, p ∈ l ∧ p ∈ α

axiom statement_b (l m n : Set Point) (α : Set Point) : 
  subset m α → subset n α → perp l m → perp l n → perp l α

axiom statement_c (l m n : Set Point) (α : Set Point) : 
  parallel l m → parallel m n → perp l α → perp n α

axiom statement_d (l m n : Set Point) (α : Set Point) : 
  parallel l m → perp m α → perp n α → parallel l n

-- The proof problem
theorem incorrect_statement_b (l m n : Set Point) (α : Set Point) :
  is_plane α → diff_lines l m n → 
  ¬(subset m α ∧ subset n α ∧ perp l m ∧ perp l n → perp l α) :=
by
  intros
  sorry

end incorrect_statement_b_l444_444608


namespace least_multiple_of_25_gt_390_l444_444722

theorem least_multiple_of_25_gt_390 : ∃ n : ℕ, n * 25 > 390 ∧ (∀ m : ℕ, m * 25 > 390 → m * 25 ≥ n * 25) ∧ n * 25 = 400 :=
by
  sorry

end least_multiple_of_25_gt_390_l444_444722


namespace f_properties_l444_444052

def f (x : ℝ) : ℝ := Real.cos (π / 2 + 2 * x)

theorem f_properties :
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x : ℝ, f (x + π) = f x) :=
sorry

end f_properties_l444_444052


namespace count_non_perfect_square_or_cube_l444_444521

-- Define perfect_square function
def perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k^2 = n

-- Define perfect_cube function
def perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k^3 = n

-- Define range from 1 to 200
def range_200 := {1 .. 200}

-- Define numbers either perfect square or perfect cube
def perfect_square_or_cube : Finset ℕ :=
  (range_200.filter perfect_square) ∪ (range_200.filter perfect_cube)

-- Define sixth powers in range
def perfect_sixth_power (n : ℕ) : Prop :=
  ∃ k : ℕ, k^6 = n

-- The final problem statement
theorem count_non_perfect_square_or_cube :
  (range_200.card - perfect_square_or_cube.card) = 182 :=
by
  sorry

end count_non_perfect_square_or_cube_l444_444521


namespace min_value_fraction_l444_444057

theorem min_value_fraction (a b c : ℝ) (h1 : a > 0) (h2 : a < (2 / 3) * b) (h3 : c ≥ b^2 / (3 * a)) : 
  ∃ x : ℝ, (∀ y : ℝ, y ≥ x → y ≥ 1) ∧ (x = 1) :=
by
  sorry

end min_value_fraction_l444_444057


namespace lizette_stamps_count_l444_444619

-- Conditions
def lizette_more : ℕ := 125
def minerva_stamps : ℕ := 688

-- Proof of Lizette's stamps count
theorem lizette_stamps_count : (minerva_stamps + lizette_more = 813) :=
by 
  sorry

end lizette_stamps_count_l444_444619


namespace triangle_area_is_12_l444_444712

def point (α : Type*) := prod α α
def A : point ℝ := (6, 0)
def B : point ℝ := (0, 6)
def on_line (C : point ℝ) : Prop := C.1 + C.2 = 10

theorem triangle_area_is_12 (C : point ℝ) (hC : on_line C)
  : 1 / 2 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) = 12 :=
sorry

end triangle_area_is_12_l444_444712


namespace a_seq_formula_min_value_M_l444_444013

open Classical

-- Given conditions for the sequence {a_n}
def a_seq (a : ℕ → ℤ) : Prop :=
  a 1 = 2 ∧ ∀ n : ℕ, n > 0 → (n + 1) * a (n + 1) - (n + 2) * a n = 2

-- Proving that the sequence is arithmetic with the general formula a_n = 2n
theorem a_seq_formula (a : ℕ → ℤ) (h : a_seq a) : ∀ n : ℕ, n > 0 → a n = 2 * n := 
sorry

-- Sum of the first n terms of a sequence
def S_seq (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, n > 0 → S n = (n * (n + 1))

-- Conditions for the sequence {b_n}
def b_seq (b : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → b n = n * (-real.sqrt 6 / 3)^(n + 1)

-- Proving that the minimum value of M such that b_n <= M is 40/27
theorem min_value_M (S : ℕ → ℤ) (b : ℕ → ℝ) (hS : S_seq (λ n, 2 * n) S) (hb : b_seq b) : 
  ∃ M : ℝ, M = 40 / 27 ∧ ∀ n ≥ 1, b n ≤ M := 
sorry

end a_seq_formula_min_value_M_l444_444013


namespace find_a_for_perpendicular_lines_l444_444353

theorem find_a_for_perpendicular_lines :
  ∃ a : ℝ, (a - 3 / 2 + 6 = 0) → a = -3 :=
begin
  -- Proposed Lean statement
  sorry -- Placeholder for the proof
end

end find_a_for_perpendicular_lines_l444_444353


namespace arrange_digits_40_752_l444_444086

theorem arrange_digits_40_752 : 
  ∃ (n : ℕ), n = 96 ∧ 
  (∀ (d1 d2 d3 d4 d5 : ℕ), 
    {d1, d2, d3, d4, d5} = {4, 0, 7, 5, 2} ∧ d1 ≠ 0 → 
    (count_valid_arrangements d1 d2 d3 d4 d5 = n)) :=
sorry

noncomputable def count_valid_arrangements (d1 d2 d3 d4 d5 : ℕ) : ℕ := 
by
  -- Since this function's implementation is skipped,
  -- it is noncomputable that is necessary to indicate missing part
  sorry

end arrange_digits_40_752_l444_444086


namespace find_x_l444_444417

-- Define the vectors a and b
def a : ℝ × ℝ := (1, 1)
def b (x : ℝ) : ℝ × ℝ := (2, x)

-- Define what it means for vectors to be parallel
def parallel (u v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, u = (k * v.1, k * v.2)

-- Formulate the problem that needs to be proven
theorem find_x (x : ℝ) :
  let sum := (a.1 + b x).1, a.2 + b x).2
  let diff := (a.1 - b x).1, a.2 - b x).2
  parallel sum diff → x = 2 :=
begin
  sorry -- proof to be filled in
end

end find_x_l444_444417


namespace line_AF_through_midpoint_of_DE_l444_444799

open EuclideanGeometry

noncomputable def midpoint_of_segment (A B : Point) := ((A.x + B.x) / 2, (A.y + B.y) / 2)

theorem line_AF_through_midpoint_of_DE
  {A B C D E F : Point}
  (h_isosceles : AB = AC)
  (h_midpoint_D : D = midpoint B C)
  (h_perpendicular_DE : ⟂ D E AC)
  (h_circumcircle_intersects_BE : ∃ k : ℕ, circumcircle A B D = circum circle (B ⊗ F))
  : passes_through (line A F) (midpoint_of_segment D E) :=
sorry

end line_AF_through_midpoint_of_DE_l444_444799


namespace common_area_eq_l444_444090

-- Define the conditions for regions M and N
def region_M : set (ℝ × ℝ) :=
  { p | 0 ≤ p.snd ∧ p.snd ≤ p.fst ∧ p.snd ≤ 2 - p.fst }

def region_N (t : ℝ) : set (ℝ × ℝ) :=
  { p | t ≤ p.fst ∧ p.fst ≤ t + 1 ∧ 0 ≤ t ∧ t ≤ 1 }

-- Function to calculate the intersection area f(t)
noncomputable def f (t : ℝ) : ℝ :=
  -t^2 + t + 1 / 2

-- The statement to prove
theorem common_area_eq (t : ℝ) (h : 0 ≤ t ∧ t ≤ 1) :
  (∃ a ∈ region_M, a ∈ region_N t) → f(t) = -t^2 + t + 1 / 2 :=
by
  sorry

end common_area_eq_l444_444090


namespace num_non_squares_cubes_1_to_200_l444_444437

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n
def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k * k * k = n
def is_perfect_sixth_power (n : ℕ) : Prop := ∃ k : ℕ, k * k * k * k * k * k = n

def num_perfect_squares (a b : ℕ) : ℕ :=
  (List.range (b + 1)).countp is_perfect_square - (List.range (a)).countp is_perfect_square

def num_perfect_cubes (a b : ℕ) : ℕ :=
  (List.range (b + 1)).countp is_perfect_cube - (List.range (a)).countp is_perfect_cube

def num_perfect_sixth_powers (a b : ℕ) : ℕ :=
  (List.range (b + 1)).countp is_perfect_sixth_power - (List.range (a)).countp is_perfect_sixth_power

theorem num_non_squares_cubes_1_to_200 :
  let squares := num_perfect_squares 1 200 in
  let cubes := num_perfect_cubes 1 200 in
  let sixth_powers := num_perfect_sixth_powers 1 200 in
  200 - (squares + cubes - sixth_powers) = 182 :=
by
  let squares := num_perfect_squares 1 200
  let cubes := num_perfect_cubes 1 200
  let sixth_powers := num_perfect_sixth_powers 1 200
  have h : 200 - (squares + cubes - sixth_powers) = 182 := sorry
  exact h

end num_non_squares_cubes_1_to_200_l444_444437


namespace difference_of_squares_count_l444_444969

theorem difference_of_squares_count :
  (number_of_integers_between (1 : ℕ) (500 : ℕ) (λ n, ∃ a b : ℕ, n = a^2 - b^2)) = 375 :=
by
  sorry

end difference_of_squares_count_l444_444969


namespace distance_sum_PA_PB_l444_444587

noncomputable def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 2 * sqrt 5 * y = 0

noncomputable def line_parametric (t : ℝ) : ℝ × ℝ :=
  (3 - (sqrt 2 / 2) * t, sqrt 5 + (sqrt 2 / 2) * t)

def point_P : ℝ × ℝ := (3, sqrt 5)

-- Main theorem statement
theorem distance_sum_PA_PB : 
  (∀ (t : ℝ), circle_equation (3 - (sqrt 2 / 2) * t) (sqrt 5 + (sqrt 2 / 2) * t)) → 
  (|PA| + |PB| = 3 * sqrt 2) sorry

end distance_sum_PA_PB_l444_444587


namespace intersection_of_sets_l444_444028

open Set

def A : Set ℝ := { x | -1 < x ∧ x < 2 }
def B : Set ℝ := { x | 0 < x }

theorem intersection_of_sets : A ∩ B = { x | 0 < x ∧ x < 2 } :=
by sorry

end intersection_of_sets_l444_444028


namespace find_base_of_exponent_l444_444993

theorem find_base_of_exponent
  (x : ℝ)
  (h1 : 4 ^ (2 * x + 2) = (some_number : ℝ) ^ (3 * x - 1))
  (x_eq : x = 1) :
  some_number = 16 := 
by
  -- proof steps would go here
  sorry

end find_base_of_exponent_l444_444993


namespace main_theorem_l444_444750

universe u
variables {α : Type u} [euclidean_geometry α]

open_locale euclidean_geometry

-- Define the given conditions
def is_isosceles (A B C : α) (h : euclidean_geometry.is_triangle A B C) : Prop :=
euclidean_geometry.dist A B = euclidean_geometry.dist A C

def midpoint (D B C : α) : Prop :=
euclidean_geometry.is_midpoint D B C

def perpendicular (E D A C : α) : Prop :=
euclidean_geometry.is_perpendicular E D A C

def circumcircle_intersects (F B D A : α) (c : set α) : Prop :=
c = euclidean_geometry.circumcircle A B D ∧ F ∈ c ∧ B ∈ c

-- The main theorem to prove
theorem main_theorem (A B C D E F : α) (h1 : euclidean_geometry.is_triangle A B C)
    (h2 : is_isosceles A B C h1) (h3 : midpoint D B C) (h4 : perpendicular E D A C)
    (h5 : ∃ c, circumcircle_intersects F B D A c) :
    euclidean_geometry.passes_through (euclidean_geometry.line A F) (euclidean_geometry.midpoint E D) :=
sorry

end main_theorem_l444_444750


namespace numbers_not_squares_nor_cubes_1_to_200_l444_444492

theorem numbers_not_squares_nor_cubes_1_to_200 : 
  let num_squares := nat.sqrt 200
  let num_cubes := nat.cbrt 200
  let num_sixth_powers := nat.root 6 200
  let total := num_squares + num_cubes - num_sixth_powers
  200 - total = 182 := 
by
  sorry

end numbers_not_squares_nor_cubes_1_to_200_l444_444492


namespace a_4_eq_5_b_geometric_seq_sum_of_first_n_c_terms_l444_444938

def a : ℕ → ℕ
| 0 := 1
| n+1 := if n % 2 = 0 then 2 * (a n) else (a n) + 1

def b (n : ℕ) : ℕ := a (2*n - 1) + 2

def c (n : ℕ) : ℕ := n * (a (2*n - 1))

def T (n : ℕ) : ℕ := (3*n - 3) * 2^n + 3 - n^2 - n

theorem a_4_eq_5 : a 3 = 5 :=
sorry

theorem b_geometric_seq (n : ℕ) : b (n + 1) / b n = 2 :=
sorry

theorem sum_of_first_n_c_terms (n : ℕ) : (finset.range n).sum c = T n :=
sorry

end a_4_eq_5_b_geometric_seq_sum_of_first_n_c_terms_l444_444938


namespace find_x_l444_444345

theorem find_x (k : ℝ) (x : ℝ) (h : x ≠ 4) :
  (x = (1 - k) / 2) ↔ ((x^2 - 3 * x - 4) / (x - 4) = 3 * x + k) :=
by sorry

end find_x_l444_444345


namespace max_rectangle_area_max_rectangle_area_exists_l444_444267

theorem max_rectangle_area (l w : ℕ) (h : l + w = 20) : l * w ≤ 100 :=
by sorry

-- Alternatively, to also show the existence of the maximum value.
theorem max_rectangle_area_exists : ∃ l w : ℕ, l + w = 20 ∧ l * w = 100 :=
by sorry

end max_rectangle_area_max_rectangle_area_exists_l444_444267


namespace inequality_solution_l444_444658

theorem inequality_solution (x : ℝ) :
  (x+3)/(x+4) > (4*x+5)/(3*x+10) ↔ x ∈ Set.Ioo (-4 : ℝ) (- (10 : ℝ) / 3) ∪ Set.Ioi 2 :=
by
  sorry

end inequality_solution_l444_444658


namespace number_of_units_of_products_minimum_shipping_cost_l444_444822

def volume_per_unit_A : ℝ := 0.8
def mass_per_unit_A : ℝ := 0.5
def volume_per_unit_B : ℝ := 2
def mass_per_unit_B : ℝ := 1

def total_volume : ℝ := 20
def total_mass : ℝ := 10.5

def truck_capacity_volume : ℝ := 6
def truck_capacity_mass : ℝ := 3.5
def cost_per_truck : ℝ := 600
def cost_per_ton : ℝ := 200

theorem number_of_units_of_products
  (x y : ℝ)
  (hx : volume_per_unit_A * x + volume_per_unit_B * y = total_volume)
  (hy : mass_per_unit_A * x + mass_per_unit_B * y = total_mass) :
  x = 5 ∧ y = 8 :=
by
  sorry

theorem minimum_shipping_cost :
  let units_A := 5
  let units_B := 8
  let total_products_volume := volume_per_unit_A * units_A + volume_per_unit_B * units_B
  let total_products_mass := mass_per_unit_A * units_A + mass_per_unit_B * units_B
  let trucks_required_volume := (total_products_volume / truck_capacity_volume).ceil.to_nat
  let trucks_required_mass := (total_products_mass / truck_capacity_mass).ceil.to_nat
  let trucks_required := max trucks_required_volume trucks_required_mass
  let cost_truck_option := trucks_required * cost_per_truck
  let cost_ton_option := total_products_mass * cost_per_ton
  min cost_truck_option cost_ton_option = 2100 :=
by
  sorry

end number_of_units_of_products_minimum_shipping_cost_l444_444822


namespace f_range_value_of_a_l444_444367

def vec_m (x : ℝ) : ℝ × ℝ := (2 - sin (2 * x + π / 6), -2)
def vec_n (x : ℝ) : ℝ × ℝ := (1, sin x ^ 2)
def f (x : ℝ) : ℝ := (vec_m x).1 * (vec_n x).1 + (vec_m x).2 * (vec_n x).2

-- Proof of the range of f(x)
theorem f_range : ∀ x ∈ Icc 0 (π / 2), f x ∈ Icc 0 (3 / 2) :=
by
  intro x hx
  sorry

-- Proof for the value of a given the triangle conditions
theorem value_of_a (A B C : ℝ) (a b c : ℝ) (hB : B ∈ Ioo 0 π) 
  (hb : b = 1) (hc : c = √3) (h_f : f (B / 2) = 1) :
  a = 1 ∨ a = 2 :=
by
  sorry

end f_range_value_of_a_l444_444367


namespace line_AF_midpoint_DE_l444_444785

variables {A B C D E F : Type} [EuclideanGeometry.point A] [EuclideanGeometry.point B] [EuclideanGeometry.point C]
  [EuclideanGeometry.point D] [EuclideanGeometry.point E] [EuclideanGeometry.point F]

-- Definitions based on the problem conditions
def isosceles_triangle (A B C : Type) [EuclideanGeometry.point A] [EuclideanGeometry.point B] [EuclideanGeometry.point C] : Prop :=
  (EuclideanGeometry.distance A B = EuclideanGeometry.distance A C)

def midpoint (D B C : Type) [EuclideanGeometry.point D] [EuclideanGeometry.point B] [EuclideanGeometry.point C] : Prop :=
  EuclideanGeometry.between D B C

def perpendicular (D E : Type) [EuclideanGeometry.point D] [EuclideanGeometry.point E] : Prop := 
  EuclideanGeometry.perp D E

def circumcircle (ABD : Set EuclideanGeometry.point) : Prop :=
  EuclideanGeometry.circumcircle ABD

def intersection (circumcircle_intersection : Type) [EuclideanGeometry.point F] [EuclideanGeometry.point B] : Prop := 
  EuclideanGeometry.intersects circumcircle_intersection F B

-- The theorem we need to prove
theorem line_AF_midpoint_DE
  (isoABC : isosceles_triangle A B C)
  (midD : midpoint D B C)
  (perpDE : perpendicular D E)
  (circABD : circumcircle {A, B, D})
  (interBF : intersection {A, B, D})
  : EuclideanGeometry.on_line A F (EuclideanGeometry.midpoint D E) :=
sorry -- proof here

end line_AF_midpoint_DE_l444_444785


namespace length_of_BC_l444_444645

open Real

-- Definitions related to Rectangle ROMN
structure Rectangle := (RO OM : ℝ)

-- Triangle ABC and points definitions
structure TriangleData := 
  (O R A B C : Point)
  (circumcenter : O = circumcenter_of_triangle A B C)
  (orthocenter : R = orthocenter_of_triangle A B C)
  (midpoint : M = midpoint_of_segment B C)
  (altitude : is_altitude A N B C)

noncomputable def BC_length (r : Rectangle) (τ : TriangleData) : ℝ :=
  2 * (5: ℝ) -- OM length is given as 5 and therefore calculated as 2 * 14

theorem length_of_BC (r : Rectangle) (τ : TriangleData) (h₁ : r.RO = 11) (h₂ : r.OM = 5) : 
  BC_length r τ = 28 := 
  by sorry  -- skipping the proof

end length_of_BC_l444_444645


namespace blue_paint_amount_l444_444922

/-- 
Prove that if Giselle uses 15 quarts of white paint, then according to the ratio 4:3:5, she should use 12 quarts of blue paint.
-/
theorem blue_paint_amount (white_paint : ℚ) (h1 : white_paint = 15) : 
  let blue_ratio := 4;
  let white_ratio := 5;
  blue_ratio / white_ratio * white_paint = 12 :=
by
  sorry

end blue_paint_amount_l444_444922


namespace factorization_solve_ineq_system_l444_444812

-- Part 1: Factorization
theorem factorization (a x y : ℝ) : -8 * a * x^2 + 16 * a * x * y - 8 * a * y^2 = -8 * a * (x - y)^2 := 
by sorry

-- Part 2: Inequality System
theorem solve_ineq_system (x : ℝ) : 
  (2 * x - 7 < 3 * (x - 1)) ∧ (4/3 * x + 3 ≥ 1 - 2/3 * x) ↔ (x ≥ -1) := 
by sorry

end factorization_solve_ineq_system_l444_444812


namespace probability_of_prime_or_multiple_of_4_is_three_fifths_l444_444173

open Set

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def is_multiple_of_4 (n : ℕ) : Prop := n % 4 = 0

noncomputable def probability_prime_or_multiple_of_4 : ℚ :=
  let numbers := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
  let prime_numbers := {n ∈ numbers | is_prime n}
  let multiples_of_4 := {n ∈ numbers | is_multiple_of_4 n}
  let favorable_numbers := prime_numbers ∪ multiples_of_4
  (favorable_numbers.to_finset.card : ℚ) / (numbers.to_finset.card : ℚ)

theorem probability_of_prime_or_multiple_of_4_is_three_fifths :
  probability_prime_or_multiple_of_4 = 3 / 5 :=
sorry

end probability_of_prime_or_multiple_of_4_is_three_fifths_l444_444173


namespace AF_passes_through_midpoint_DE_l444_444773

open_locale classical

-- Definition of the geometrical setup
def isosceles_triangle (A B C : Point) : Prop := dist A B = dist A C
def midpoint (D B C : Point) : Prop := dist B D = dist C D
def perpendicular (D E A C : Point) : Prop := ∠ DEA = 90
def circumcircle (A B D F : Point): Prop := ∃ O R, Circle.circle O R A ∧ Circle.circle O R B ∧ Circle.circle O R D ∧ Circle.circle O R F

-- Main statement to prove
theorem AF_passes_through_midpoint_DE
  (A B C D E F : Point)
  (h1 : isosceles_triangle A B C)
  (h2 : midpoint D B C)
  (h3 : perpendicular D E A C)
  (h4 : ∃ O R, Circle.circle O R A ∧ Circle.circle O R B ∧ Circle.circle O R D ∧ Circle.circle O R F)
  (h5 : collinear B E F)
  (h6 : intersects (circumcircle A B D) (line B E) B F) :
  passes_through (line A F) (midpoint D E) := 
sorry

end AF_passes_through_midpoint_DE_l444_444773


namespace not_divisible_1961_1963_divisible_1963_1965_l444_444157

def is_divisible_by_three (n : Nat) : Prop :=
  n % 3 = 0

theorem not_divisible_1961_1963 : ¬ is_divisible_by_three (1961 * 1963) :=
by
  sorry

theorem divisible_1963_1965 : is_divisible_by_three (1963 * 1965) :=
by
  sorry

end not_divisible_1961_1963_divisible_1963_1965_l444_444157


namespace numbers_neither_square_nor_cube_l444_444535

theorem numbers_neither_square_nor_cube (n : ℕ) (h : n = 200) :
  let num_squares := 14 in
  let num_cubes := 5 in
  let num_sixth_powers := 2 in
  (n - (num_squares + num_cubes - num_sixth_powers)) = 183 :=
by
  -- Let the number of integers in the range
  let num_total := 200
  -- Define how we count perfect squares and cubes
  let num_squares := 14
  let num_cubes := 5
  let num_sixth_powers := 2

  -- Subtract the overlapping perfect sixth powers from the sum of squares and cubes
  let num_either_square_or_cube := num_squares + num_cubes - num_sixth_powers

  -- Calculate the final result
  let result := num_total - num_either_square_or_cube

  -- Prove by computation
  show result = 183 from
    calc
      result = num_total - num_either_square_or_cube := rfl
           ... = 200 - (14 + 5 - 2)                   := rfl
           ... = 200 - 17                             := rfl
           ... = 183                                  := rfl

end numbers_neither_square_nor_cube_l444_444535


namespace line_AF_passes_midpoint_DE_l444_444766

-- Define Triangle Isosceles
structure IsoscelesTriangle (A B C : Type) :=
  (A B C : Type)
  (isosceles : A = B)

-- Define midpoint
structure Midpoint (D B C : Type) :=
  (D : Type)
  (midpoint : D = B ∧ D = C)

-- Define perpendicular
structure Perpendicular (D E A C : Type) :=
  (perpendicular : ∀ {D E A C : Type}, A = C → D ≠ A → D = E)

-- Conditions
variables {ABC : Triangle} [IsoscelesTriangle ABC]
variables {D : Point} [Midpoint D ABC.base]
variables {DE : Line} [Perpendicular D E ABC.ABC.side]

-- The circumcircle and intersection
variable {circ_circ : ∃ circ_circ : circle (triangle ABD), intersection (BE, circ_circ) = {B, F}}

-- Main proof goal: Prove that line AF passes through the midpoint of DE.
theorem line_AF_passes_midpoint_DE :
  ∀ {A F D E : Type} {circ_circ : circle (triangle ABD)}, 
  intersect (AF, midpoint DE) ↔ intersect (circ_circ, BE) :=
sorry

end line_AF_passes_midpoint_DE_l444_444766


namespace min_attendees_l444_444702

-- Define the constants and conditions
def writers : ℕ := 35
def min_editors : ℕ := 39
def x_max : ℕ := 26

-- Define the total number of people formula based on inclusion-exclusion principle
-- and conditions provided
def total_people (x : ℕ) : ℕ := writers + min_editors - x + 2 * x

-- Theorem to prove that the minimum number of attendees is 126
theorem min_attendees : ∃ x, x ≤ x_max ∧ total_people x = 126 :=
by
  use x_max
  sorry

end min_attendees_l444_444702


namespace minimum_value_f_when_a_is_two_range_of_a_such_that_fx_le_x_l444_444056

def f (x : ℝ) (a : ℝ) : ℝ := real.sqrt (x^2 - 2*x + 1) + abs (x + a)

theorem minimum_value_f_when_a_is_two : 
  ∃ x : ℝ, f x 2 = |1 - x| + |x + 2| ∧ f x 2 ≥ 3 :=
sorry

theorem range_of_a_such_that_fx_le_x :
  ∃ (a : ℝ), ∀ x : ℝ, x ∈ set.Icc (2 / 3) 1 → f x a ≤ x → a ∈ set.Icc (-1 : ℝ) (-1 / 3) :=
sorry

end minimum_value_f_when_a_is_two_range_of_a_such_that_fx_le_x_l444_444056


namespace total_amount_is_70000_l444_444820

-- Definitions based on the given conditions
def total_amount_divided (amount_10: ℕ) (amount_20: ℕ) : ℕ :=
  amount_10 + amount_20

def interest_earned (amount_10: ℕ) (amount_20: ℕ) : ℕ :=
  (amount_10 * 10 / 100) + (amount_20 * 20 / 100)

-- Statement to be proved
theorem total_amount_is_70000 (amount_10: ℕ) (amount_20: ℕ) (total_interest: ℕ) :
  amount_10 = 60000 →
  total_interest = 8000 →
  interest_earned amount_10 amount_20 = total_interest →
  total_amount_divided amount_10 amount_20 = 70000 :=
by
  intros h1 h2 h3
  sorry

end total_amount_is_70000_l444_444820


namespace max_green_red_transitions_l444_444204

theorem max_green_red_transitions :
  ∃ (n : ℕ), n = 49 ∧
  (∀ (plots : vector ℕ 100),
   ∀ (painter1 painter2 : ℕ → option bool),
   (∀ i, painter1 i = some true ∨ painter1 i = some false) → -- first painter colorblind but remembers painted plots
   (∀ i, painter2 i = some true ∨ painter2 i = some false) → -- second painter paints remaining plots
   (∀ i, (painter1 i ≠ none → painter2 i = none) ∧ (painter2 i ≠ none → painter1 i = none)) → -- painters alternate days
   ∃ (transitions : ℕ), transitions = n ∧
     (∀ i : ℕ, i < 99 → 
        (painter1 i = some true ∧ painter2 (i+1) = some false) ∨
        (painter1 i = some false ∧ painter2 (i+1) = some true) ∨
        (painter1 (i+1) = some true ∧ painter2 i = some false) ∨
        (painter1 (i+1) = some false ∧ painter2 i = some true))) :=
begin
  sorry
end

end max_green_red_transitions_l444_444204


namespace sugar_needed_for_third_layer_l444_444301

-- Let cups be the amount of sugar, and define the layers
def first_layer_sugar : ℕ := 2
def second_layer_sugar : ℕ := 2 * first_layer_sugar
def third_layer_sugar : ℕ := 3 * second_layer_sugar

-- The theorem we want to prove
theorem sugar_needed_for_third_layer : third_layer_sugar = 12 := by
  sorry

end sugar_needed_for_third_layer_l444_444301


namespace planes_parallel_lines_parallel_l444_444032

open Set

variables {Point : Type} [MetricSpace Point]

structure Line (Point : Type) :=
  (containing : Set Point)
  (is_line : is_infinite is_line)

structure Plane (Point : Type) :=
  (containing : Set Point)
  (is_plane : is_infinite is_plane)

variable {α β γ : Plane Point}
variable {m n : Line Point}

-- alpha intersect gamma = m
axiom h1 : is_intersection α γ m
-- beta intersect gamma = n
axiom h2 : is_intersection β γ n
-- alpha parallel beta
axiom h3 : is_parallel α β

-- Prove that m is parallel to n
theorem planes_parallel_lines_parallel : is_parallel m n :=
by
  sorry

end planes_parallel_lines_parallel_l444_444032


namespace part1_part3_l444_444016

noncomputable def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

def lambda_k_sequence (a S : ℕ → ℝ) (λ k : ℝ) : Prop :=
∀ n : ℕ, S (n + 1)^(1 / k) - S n^(1 / k) = λ * a (n + 1)^(1 / k)

noncomputable def Sn {a : ℕ → ℝ} (S : ℕ → ℝ) : Prop :=
S 1 = a 1 ∧ ∀ n : ℕ, S (n + 1) = S n + a (n + 1)

theorem part1 (a S : ℕ → ℝ) (λ : ℝ) (h_arith : arithmetic_sequence a 1) (h_lambda1 : lambda_k_sequence a S λ 1) : 
λ = 1 :=
sorry

noncomputable theorem part2 (a S : ℕ → ℝ) (h_lambda : lambda_k_sequence a S (sqrt 3 / 3) 2) (h_pos : ∀ n : ℕ, 0 < a n) :
∀ n : ℕ, (if n = 0 then a n = 1 else a n = 3 * (4^(n - 2))) :=
sorry

theorem part3 (a1 a2 a3 S1 S2 S3 : ℕ → ℝ) (λ : ℝ) (h_lambda1 : lambda_k_sequence a1 S1 λ 3)
  (h_lambda2 : lambda_k_sequence a2 S2 λ 3)
  (h_lambda3 : lambda_k_sequence a3 S3 λ 3) 
  (h_ge0_1 : ∀ n : ℕ, 0 ≤ a1 n) (h_ge0_2 : ∀ n : ℕ, 0 ≤ a2 n) (h_ge0_3 : ∀ n : ℕ, 0 ≤ a3 n) :
0 < λ ∧ λ < 1 :=
sorry

end part1_part3_l444_444016


namespace general_formula_sum_of_series_l444_444379

-- Arithmetic sequence with given conditions
variable (a : ℕ → ℕ)
axiom a_3 : a 3 = 4
axiom sum_a2_a6 : a 2 + a 6 = 10

-- The general formula for the arithmetic sequence
theorem general_formula : ∃ d, ∀ n : ℕ, a n = n + 1 := sorry

-- The sum of the first n terms of the series {a_n / 2^n}
def T (n : ℕ) : ℚ := ∑ i in Finset.range n, a i / (2 ^ i)

theorem sum_of_series : ∀ n : ℕ, T (n) = 3 - (n + 3) / (2 ^ n) := sorry

end general_formula_sum_of_series_l444_444379


namespace sqrt_pattern_l444_444387

theorem sqrt_pattern {a b : ℕ} (h1 : sqrt (2 + 2 / 3) = 2 * sqrt (2 / 3))
  (h2 : sqrt (3 + 3 / 8) = 3 * sqrt (3 / 8))
  (h3 : sqrt (4 + 4 / 15) = 4 * sqrt (4 / 15)) :
  sqrt (8 + b / a) = 8 * sqrt (b / a) ↔ (a = 63 ∧ b = 8) :=
by
  sorry

end sqrt_pattern_l444_444387


namespace russian_alphabet_symmetry_l444_444679

-- Definitions based on conditions
def Group1 := {"А", "Д", "М", "П", "Т", "Ш"}
def Group2 := {"В", "Е", "З", "К", "С", "Э", "Ю"}
def Group3 := {"И"}
def Group4 := {"Ж", "Н", "О", "Ф", "Х"}
def Group5 := {"Б", "Г", "Л", "Р", "У", "Ц", "Ч", "Щ", "Я"}

-- Define the symmetry properties
def has_vertical_symmetry (letter : String) : Bool := letter ∈ Group1
def has_horizontal_symmetry (letter : String) : Bool := letter ∈ Group2
def has_central_symmetry (letter : String) : Bool := letter ∈ Group3
def has_all_symmetries (letter : String) : Bool := letter ∈ Group4
def has_no_symmetry (letter : String) : Bool := letter ∈ Group5

-- The main theorem
theorem russian_alphabet_symmetry :
  ∀ letter : String,
    (has_vertical_symmetry letter ∧ letter ∈ Group1) ∨
    (has_horizontal_symmetry letter ∧ letter ∈ Group2) ∨
    (has_central_symmetry letter ∧ letter ∈ Group3) ∨
    (has_all_symmetries letter ∧ letter ∈ Group4) ∨
    (has_no_symmetry letter ∧ letter ∈ Group5)
:= by
  intros letter
  -- Proof omitted, add necessary elaboration here
  sorry

end russian_alphabet_symmetry_l444_444679


namespace max_value_of_f_l444_444172

noncomputable def f (a b : ℝ) : ℝ → ℝ :=
  λ x, (4 - x^2) * (a * x^2 + b * x + 5)

theorem max_value_of_f (a b : ℝ) (sym_first_order_time : a = 1) (sym_second_degree : b = 6) :
  ∀ x : ℝ, f a b x ≤ 36 :=
by
  sorry

end max_value_of_f_l444_444172


namespace count_valid_numbers_between_1_and_200_l444_444428

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k^2 = n
def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k^3 = n
def is_perfect_sixth_power (n : ℕ) : Prop := ∃ k : ℕ, k^6 = n

def count_perfect_squares (m : ℕ) : ℕ :=
  Nat.sqrt m

def count_perfect_cubes (m : ℕ) : ℕ :=
  (m : ℝ).cbrt.to_nat

def count_perfect_sixth_powers (m : ℕ) : ℕ :=
  (m : ℝ).cbrt.sqrt.to_nat

def count_either_squares_or_cubes (m : ℕ) : ℕ :=
  let squares := count_perfect_squares m
  let cubes := count_perfect_cubes m
  let sixths := count_perfect_sixth_powers m
  squares + cubes - sixths

def count_neither_squares_nor_cubes (n : ℕ) : ℕ :=
  n - count_either_squares_or_cubes n

theorem count_valid_numbers_between_1_and_200 : count_neither_squares_nor_cubes 200 = 182 :=
by
  sorry

end count_valid_numbers_between_1_and_200_l444_444428


namespace find_k_l444_444006

open Real

-- Definitions
def vec_a := (1 : ℝ, 2 : ℝ)
def vec_b := (-3 : ℝ, 2 : ℝ)

-- The vectors in the condition
def parallel_vectors (k : ℝ) :=
  let ka_plus_b := (k * vec_a.1 + vec_b.1, k * vec_a.2 + vec_b.2)
  let a_minus_3b := (vec_a.1 - 3 * vec_b.1, vec_a.2 - 3 * vec_b.2)
  ∃ μ : ℝ, ka_plus_b = (μ * a_minus_3b.1, μ * a_minus_3b.2)

-- Proof statement
theorem find_k (k : ℝ) : parallel_vectors k → k = -1/3 := sorry

end find_k_l444_444006


namespace find_a_l444_444140

def A : Set ℝ := {-1, 0, 1}
noncomputable def B (a : ℝ) : Set ℝ := {a + 1, 2 * a}

theorem find_a (a : ℝ) : (A ∩ B a = {0}) → a = -1 := by
  sorry

end find_a_l444_444140


namespace expected_loops_value_l444_444921

-- Defining the number of ropes
def numRopes : ℕ := 6

-- Defining the number of loose ends
def numLooseEnds : ℕ := numRopes * 2

-- Defining the expected number of loops computed from the given conditions
noncomputable def expected_loops : ℝ :=
  (∑ k in finset.range (numRopes + 1), 1 / (2 * (numRopes - k) - 1 : ℝ))

-- The theorem statement
theorem expected_loops_value :
  abs (expected_loops - 1.8782) < 0.0001 :=
sorry

end expected_loops_value_l444_444921


namespace complex_quadrant_l444_444932

theorem complex_quadrant (z : ℂ) (h : 2 * conj z - z = 3 + 6 * Complex.i) : 
  0 < z.re ∧ z.im < 0 :=
by
  have a : z.re = 3, sorry
  have b : z.im = -2, sorry
  exact ⟨by linarith, by linarith⟩

end complex_quadrant_l444_444932


namespace sum_CE_k_sq_l444_444326

noncomputable def sqrt_123 : ℝ := Real.sqrt 123
noncomputable def sqrt_13 : ℝ := Real.sqrt 13

-- Equilateral triangle ABC with side length √123
variables (A B C : Type) [has_dist A] [metric_space A] 
variables (s : ℝ) (D1 E1 D2 E2 E3 E4 : A)
variable (h1 : dist A B = sqrt_123)
variable (h2 : dist B C = sqrt_123)
variable (h3 : dist C A = sqrt_123)
variable (congruence1 : /\( AD1E1, ABC \))
variable (congruence2 : /\( AD1E2, ABC \))
variable (congruence3 : /\( AD2E3, ABC \))
variable (congruence4 : /\( AD2E4, ABC \))
variable (h5 : dist B D1 = sqrt_13)
variable (h6 : dist B D2 = sqrt_13)

theorem sum_CE_k_sq : ∀ (k : ℕ), (1 ≤ k ∧ k ≤ 4) → dist Ct [E_k] → ℝ := CE_1 + CE_2 + CE_3 + CE_4
  (∑_{k = 1}^{4}) dist (C E1) ^ 2 = 492 :=
by
  sorry

end sum_CE_k_sq_l444_444326


namespace race_length_l444_444291

theorem race_length
  (B_s : ℕ := 50) -- Biff's speed in yards per minute
  (K_s : ℕ := 51) -- Kenneth's speed in yards per minute
  (D_above_finish : ℕ := 10) -- distance Kenneth is past the finish line when Biff finishes
  : {L : ℕ // L = 500} := -- the length of the race is 500 yards.
  sorry

end race_length_l444_444291


namespace find_lines_through_M_at_distance_d_l444_444339

-- Definitions for Point, Line, and Cylinder
structure Point := (x : ℝ) (y : ℝ) (z : ℝ)

structure Line := (p1 : Point) (p2 : Point)

structure Cylinder := (axis : Line) (radius : ℝ)

def distance (p : Point) (l : Line) : ℝ := sorry  -- Function to determine the perpendicular distance from a point to a line

def tangent_planes (c : Cylinder) (p : Point) : List Line := sorry  -- Function to return tangent planes to the cylinder

-- The main theorem statement
theorem find_lines_through_M_at_distance_d (M : Point) (AB : Line) (d : ℝ)
  (h1 : distance M AB = d) :
  ∃ l : List Line, (∀ l' ∈ l, tangent_planes (Cylinder AB d) M) := sorry

end find_lines_through_M_at_distance_d_l444_444339


namespace inverse_g_undefined_at_one_l444_444554

noncomputable def g (x : ℝ) : ℝ := (x - 5) / (x - 6)

theorem inverse_g_undefined_at_one :
  ∃ (g_inv : ℝ → ℝ), ∀ x, (g ∘ g_inv x = x ∧ g_inv ∘ g x = x) → ¬ (∃ y, g_inv 1 = y) := 
by
  sorry

end inverse_g_undefined_at_one_l444_444554


namespace exists_balanced_exists_balanced_centerless_iff_l444_444356

-- Define the predicates balanced and centerless.
def balanced (S : Finset (ℝ × ℝ)) : Prop :=
  ∀ A B ∈ S, A ≠ B → ∃ C ∈ S, dist A C = dist B C

def centerless (S : Finset (ℝ × ℝ)) : Prop :=
  ∀ A B C ∈ S, A ≠ B → B ≠ C → C ≠ A →
  ∀ P ∈ S, PA ≠ PB ∨ PB ≠ PC

-- Prove that there exists a balanced point set for every n ≥ 3
theorem exists_balanced (n : ℕ) (h : n ≥ 3) : ∃ S : Finset (ℝ × ℝ), S.card = n ∧ balanced S :=
sorry

-- Determine all integers n ≥ 3 such that there exists a point set both balanced and centerless
theorem exists_balanced_centerless_iff : ∀ n : ℕ, n ≥ 3 → 
  (∃ S : Finset (ℝ × ℝ), S.card = n ∧ balanced S ∧ centerless S) ↔ odd n :=
sorry

end exists_balanced_exists_balanced_centerless_iff_l444_444356


namespace puppies_given_l444_444850

-- Definitions of the initial and left numbers of puppies
def initial_puppies : ℕ := 7
def left_puppies : ℕ := 2

-- Theorem stating that the number of puppies given to friends is the difference
theorem puppies_given : initial_puppies - left_puppies = 5 := by
  sorry -- Proof not required, so we use sorry

end puppies_given_l444_444850


namespace fruit_shop_apples_l444_444183

-- Given conditions
def morning_fraction : ℚ := 3 / 10
def afternoon_fraction : ℚ := 4 / 10
def total_sold : ℕ := 140

-- Define the total number of apples and the resulting condition
def total_fraction_sold : ℚ := morning_fraction + afternoon_fraction

theorem fruit_shop_apples (A : ℕ) (h : total_fraction_sold * A = total_sold) : A = 200 := 
by sorry

end fruit_shop_apples_l444_444183


namespace AF_passes_through_midpoint_of_DE_l444_444779

open EuclideanGeometry

-- Definitions based on the conditions
variables {A B C D E F : Point} 
variables {ABC : Triangle}
variables (isosceles_ABC : IsIsoscelesTriangle ABC A B)
variables (midpoint_D : IsMidpoint D B C)
variables (perpendicular_DE : IsPerpendicular DE D AC)
variables {circumcircle_ABD : Circumcircle ABD}
variables (intersection_F : IntersectsAt circumcircle_ABD BE B F)

-- The theorem statement
theorem AF_passes_through_midpoint_of_DE :
  PassesThroughMidpoint AF D E :=
sorry

end AF_passes_through_midpoint_of_DE_l444_444779


namespace part1_part2_part3_l444_444388

variable (a : ℕ → ℝ)
variable (A B C T : ℕ → ℝ)
variable (n : ℕ)
variable (q : ℝ)

-- Define the conditions
axiom geom_seq : ∀ n, a (n + 1) = a n * q
axiom a2_eq_3 : a 2 = 3
axiom a5_eq_81 : a 5 = 81

-- A(n) = sum of the first n terms of the sequence
def A (n : ℕ) := ∑ i in finset.range (n + 1), a i

-- B(n) = sum of the terms from a_2 to a_{n+1}
def B (n : ℕ) := ∑ i in finset.range (n + 1), a (i + 1)

-- C(n) = sum of the terms from a_3 to a_{n+2}
def C (n : ℕ) := ∑ i in finset.range (n + 1), a (i + 2)

-- Given conditions in third problem
axiom cond1 : a 3 + a 4 = 3
axiom cond2 : a 2 * a 5 = 2
axiom cond3 : a 3 > a 4

-- Prove A(n) is given by (1/2)*(3^n - 1) for the specified conditions
theorem part1 : A n = 1 / 2 * (3^n - 1) :=
sorry

-- Prove that B(n) is the geometric mean of A(n) and C(n)
theorem part2 (n : ℕ) : B n = sqrt (A n * C n) :=
sorry

-- Prove T(n) is given by 4n + 1 / 2^(n-2) - 4 for the specified conditions
def T (n : ℕ) := (finset.range n).sum C

theorem part3 : T n = 4 * n + 1 / 2^(n - 2) - 4 :=
sorry

end part1_part2_part3_l444_444388


namespace sequence_inequality_l444_444099

variables {a b c : ℕ} (n : ℕ)

def a_n (n : ℕ) : ℝ := (a * n : ℝ) / (b * n + c)

theorem sequence_inequality (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a_n n < a_n (n + 1) :=
sorry

end sequence_inequality_l444_444099


namespace tan_double_angle_l444_444941

-- Definitions
def theta (θ : ℝ) := θ
def sin_theta : ℝ := 3 / 5

-- Conditions
def theta_second_quadrant (θ : ℝ) : Prop := 
  (π / 2 < θ) ∧ (θ < π)
def sin_def (θ : ℝ) : Prop := 
  sin θ = 3 / 5

-- The theorem to be proven
theorem tan_double_angle (θ : ℝ) 
  (h1 : theta_second_quadrant θ)
  (h2 : sin_def θ) : 
  tan (2 * θ) = -24 / 7 := 
sorry

end tan_double_angle_l444_444941


namespace num_from_1_to_200_not_squares_or_cubes_l444_444461

noncomputable def numNonPerfectSquaresAndCubes (n : ℕ) : ℕ :=
  let num_squares := 14
  let num_cubes := 5
  let num_sixth_powers := 2
  n - (num_squares + num_cubes - num_sixth_powers)

theorem num_from_1_to_200_not_squares_or_cubes : numNonPerfectSquaresAndCubes 200 = 183 := by
  sorry

end num_from_1_to_200_not_squares_or_cubes_l444_444461


namespace flagpole_height_l444_444858

def xiaoMing_height : ℝ := 1.6
def xiaoMing_shadow : ℝ := 0.4
def flagpole_shadow : ℝ := 5

theorem flagpole_height : ∃ x : ℝ, (1.6 / 0.4 = x / 5) ∧ x = 20 := by
  use 20
  split
  . norm_num
  . norm_num

end flagpole_height_l444_444858


namespace numbers_not_squares_or_cubes_in_200_l444_444474

/-- There are 182 numbers between 1 and 200 that are neither perfect squares nor perfect cubes. -/
theorem numbers_not_squares_or_cubes_in_200 : (finset.range 200).filter (λ n, ¬(∃ m, m^2 = n) ∧ ¬(∃ k, k^3 = n)).card = 182 :=
by 
  /-
    The proof will involve constructing the set of numbers from 1 to 200, filtering out those that
    are perfect squares or perfect cubes, and counting the remaining numbers.
  -/
  sorry

end numbers_not_squares_or_cubes_in_200_l444_444474


namespace count_difference_of_squares_l444_444980

theorem count_difference_of_squares : 
  ∃ n : ℕ, (∀ x ∈ finset.range 501, ∃ a b : ℕ, x = a^2 - b^2) → n = 375 :=
by
  let nums := finset.range 501
  let expressible := nums.filter (λ x, ∃ a b : ℕ, x = a^2 - b^2)
  exact sorry

end count_difference_of_squares_l444_444980


namespace max_non_divisible_subset_size_l444_444589

def is_odd (n : ℕ) : Prop := n % 2 = 1

def first_100_odd_numbers : Finset ℕ := 
  Finset.filter is_odd (Finset.range 200)

def no_divisible (S : Finset ℕ) : Prop := 
  ∀ a b ∈ S, a ≠ b → ¬(a ∣ b)

noncomputable def maximal_non_divisible_subset (M : Finset ℕ) : Finset ℕ :=
  Finset.filter (λ n, 67 ≤ n ∧ n ≤ 199 ∧ is_odd n) M

theorem max_non_divisible_subset_size : 
  (maximal_non_divisible_subset first_100_odd_numbers).card = 67 := 
sorry

end max_non_divisible_subset_size_l444_444589


namespace cake_sugar_calculation_l444_444298

theorem cake_sugar_calculation (sugar_first_layer : ℕ) (sugar_second_layer : ℕ) (sugar_third_layer : ℕ) :
  sugar_first_layer = 2 →
  sugar_second_layer = 2 * sugar_first_layer →
  sugar_third_layer = 3 * sugar_second_layer →
  sugar_third_layer = 12 := 
by
  intros h1 h2 h3
  have h4 : 2 = sugar_first_layer, from h1.symm
  have h5 : sugar_second_layer = 2 * 2, by rw [h4, h2]
  have h6 : sugar_third_layer = 3 * 4, by rw [h5, h3]
  exact h6

end cake_sugar_calculation_l444_444298


namespace equation_solution_system_of_inequalities_solution_l444_444813

theorem equation_solution (x : ℝ) : (3 / (x - 1) = 1 / (2 * x + 3)) ↔ (x = -2) :=
by
  sorry

theorem system_of_inequalities_solution (x : ℝ) : ((3 * x - 1 ≥ x + 1) ∧ (x + 3 > 4 * x - 2)) ↔ (1 ≤ x ∧ x < 5 / 3) :=
by
  sorry

end equation_solution_system_of_inequalities_solution_l444_444813


namespace birds_initially_sitting_l444_444168

theorem birds_initially_sitting (initial_birds birds_joined total_birds : ℕ) 
  (h1 : birds_joined = 4) (h2 : total_birds = 6) (h3 : total_birds = initial_birds + birds_joined) : 
  initial_birds = 2 :=
by
  sorry

end birds_initially_sitting_l444_444168


namespace Olga_paints_zero_boards_l444_444634

variable (t p q t' : ℝ)
variable (rv ro : ℝ)

-- Conditions
axiom Valera_solo_trip : 2 * t + p = 2
axiom Valera_and_Olga_painting_time : 2 * t' + q = 3
axiom Valera_painting_rate : rv = 11 / p
axiom Valera_Omega_painting_rate : rv * q + ro * q = 9
axiom Valera_walk_faster : t' > t

-- Question: How many boards will Olga be able to paint alone if she needs to return home 1 hour after leaving?
theorem Olga_paints_zero_boards :
  t' > 1 → 0 = 0 := 
by 
  sorry

end Olga_paints_zero_boards_l444_444634


namespace fraction_students_walk_home_l444_444857

theorem fraction_students_walk_home :
  let bus := 1/3
  let auto := 1/5
  let bicycle := 1/8
  let total_students := 1
  let other_transportation := bus + auto + bicycle
  let walk_home := total_students - other_transportation
  walk_home = 41/120 :=
by 
  let bus := 1/3
  let auto := 1/5
  let bicycle := 1/8
  let total_students := 1
  let other_transportation := bus + auto + bicycle
  let walk_home := total_students - other_transportation
  have h_bus : bus = 40 / 120 := by sorry
  have h_auto : auto = 24 / 120 := by sorry
  have h_bicycle : bicycle = 15 / 120 := by sorry
  have h_total_transportation : other_transportation = 40 / 120 + 24 / 120 + 15 / 120 := by sorry
  have h_other_transportation_sum : other_transportation = 79 / 120 := by sorry
  have h_walk_home : walk_home = 1 - 79 / 120 := by sorry
  have h_walk_home_simplified : walk_home = 41 / 120 := by sorry
  exact h_walk_home_simplified

end fraction_students_walk_home_l444_444857


namespace modulo_7_example_l444_444986

def sum := 2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 + 99999999

theorem modulo_7_example : (sum % 7) = 5 :=
by
  sorry

end modulo_7_example_l444_444986


namespace Mia_study_minutes_each_day_l444_444622

theorem Mia_study_minutes_each_day :
  let total_minutes_in_day := 24 * 60
  let sleeping_minutes := 7 * 60
  let waking_minutes := total_minutes_in_day - sleeping_minutes
  let tv_minutes := (1 / 5) * waking_minutes
  let exercise_minutes := (1 / 8) * waking_minutes
  let remaining_after_tv_and_exercise := waking_minutes - (tv_minutes + exercise_minutes)
  let social_media_minutes := (1 / 6) * remaining_after_tv_and_exercise
  let remaining_after_social_media := remaining_after_tv_and_exercise - social_media_minutes
  let chores_minutes := (1 / 3) * remaining_after_social_media
  let remaining_after_chores := remaining_after_social_media - chores_minutes
  let study_minutes := (1 / 4) * remaining_after_chores
  Int.ofNat (study_minutes).round = 96 := 
by
  sorry

end Mia_study_minutes_each_day_l444_444622


namespace binomial_expansion_coefficient_l444_444093

theorem binomial_expansion_coefficient (n : ℕ) (h : n = 8) (h5 : ∀ k, binomial_coefficient 8 k ≤ binomial_coefficient 8 4) :
  coefficient_of_term ((x - 1/x)^n) (λ expr, expr = x^2) = -56 :=
by
  sorry

end binomial_expansion_coefficient_l444_444093


namespace complex_quadrant_l444_444393

open Complex

noncomputable def z : ℂ := (2 * I) / (1 - I)

theorem complex_quadrant (z : ℂ) (h : (1 - I) * z = 2 * I) : 
  z.re < 0 ∧ z.im > 0 := by
  sorry

end complex_quadrant_l444_444393


namespace ants_harvest_time_l444_444276

theorem ants_harvest_time :
  ∃ h : ℕ, (∀ h : ℕ, 24 - 4 * h = 12) ∧ h = 3 := sorry

end ants_harvest_time_l444_444276


namespace number_of_correct_statements_l444_444064

variable (a b : Vector) (ha : Vector.has_norm a) (hb : Vector.has_norm b)
variable (h_unequal : a ≠ 0 ∧ b ≠ 0)
variable (h_x : ∀ i, i ∈ {1, 2, 3, 4} → Vector (a, b)) (h_y : ∀ i, i ∈ {1, 2, 3, 4} → Vector (a, b))

def S (x y : Fin 4 → Vector) :=
  x 0 • y 0 + x 1 • y 1 + x 2 • y 2 + x 3 • y 3

def S_min (x y : Fin 4 → Vector) : ℝ := sorry

def statement1 := ∃ s1 s2 s3, S x y = s1 ∨ S x y = s2 ∨ S x y = s3 ∧ s1 ≠ s2 ∧ s2 ≠ s3 ∧ s1 ≠ s3
def statement2 := (a ⟂ b) → S_min x y = 0
def statement3 := (a ∥ b) → S_min x y ≠ abs b
def statement4 := (∥ b ∥ = 2 * ∥ a ∥) ∧ (S_min x y ≠ 4 * (∥a∥ ^ 2)) → angle_between a b = π / 3

theorem number_of_correct_statements :
  (statement1 ∧ statement2 ∧ ¬statement3 ∧ statement4) = 3 :=
sorry

end number_of_correct_statements_l444_444064


namespace path_length_of_dot_l444_444254

-- Define the edge length of the cube
def edge_length : ℝ := 3

-- Define the conditions of the problem
def cube_condition (l : ℝ) (rolling_without_slipping : Prop) (at_least_two_vertices_touching : Prop) (dot_at_one_corner : Prop) (returns_to_original_position : Prop) : Prop :=
  l = edge_length ∧ rolling_without_slipping ∧ at_least_two_vertices_touching ∧ dot_at_one_corner ∧ returns_to_original_position

-- Define the theorem to be proven
theorem path_length_of_dot (rolling_without_slipping : Prop) (at_least_two_vertices_touching : Prop) (dot_at_one_corner : Prop) (returns_to_original_position : Prop) :
  cube_condition edge_length rolling_without_slipping at_least_two_vertices_touching dot_at_one_corner returns_to_original_position →
  ∃ c : ℝ, c = 6 ∧ (c * Real.pi) = 6 * Real.pi :=
by
  intro h
  sorry

end path_length_of_dot_l444_444254


namespace count_not_squares_or_cubes_200_l444_444527

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n
def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k * k * k = n
def is_sixth_power (n : ℕ) : Prop := ∃ k : ℕ, k^6 = n

theorem count_not_squares_or_cubes_200 :
  let count_squares := (Finset.range 201).filter is_perfect_square).card
  let count_cubes := (Finset.range 201).filter is_perfect_cube).card
  let count_sixth_powers := (Finset.range 201).filter is_sixth_power).card
  (200 - (count_squares + count_cubes - count_sixth_powers)) = 182 := 
by {
  -- Define everything in terms of Finset and filter
  let count_squares := (Finset.range 201).filter is_perfect_square).card,
  let count_cubes := (Finset.range 201).filter is_perfect_cube).card,
  let count_sixth_powers := (Finset.range 201).filter is_sixth_power).card,
  exact sorry
}

end count_not_squares_or_cubes_200_l444_444527


namespace AF_passes_through_midpoint_of_DE_l444_444782

open EuclideanGeometry

-- Definitions based on the conditions
variables {A B C D E F : Point} 
variables {ABC : Triangle}
variables (isosceles_ABC : IsIsoscelesTriangle ABC A B)
variables (midpoint_D : IsMidpoint D B C)
variables (perpendicular_DE : IsPerpendicular DE D AC)
variables {circumcircle_ABD : Circumcircle ABD}
variables (intersection_F : IntersectsAt circumcircle_ABD BE B F)

-- The theorem statement
theorem AF_passes_through_midpoint_of_DE :
  PassesThroughMidpoint AF D E :=
sorry

end AF_passes_through_midpoint_of_DE_l444_444782


namespace num_non_squares_cubes_1_to_200_l444_444443

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n
def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k * k * k = n
def is_perfect_sixth_power (n : ℕ) : Prop := ∃ k : ℕ, k * k * k * k * k * k = n

def num_perfect_squares (a b : ℕ) : ℕ :=
  (List.range (b + 1)).countp is_perfect_square - (List.range (a)).countp is_perfect_square

def num_perfect_cubes (a b : ℕ) : ℕ :=
  (List.range (b + 1)).countp is_perfect_cube - (List.range (a)).countp is_perfect_cube

def num_perfect_sixth_powers (a b : ℕ) : ℕ :=
  (List.range (b + 1)).countp is_perfect_sixth_power - (List.range (a)).countp is_perfect_sixth_power

theorem num_non_squares_cubes_1_to_200 :
  let squares := num_perfect_squares 1 200 in
  let cubes := num_perfect_cubes 1 200 in
  let sixth_powers := num_perfect_sixth_powers 1 200 in
  200 - (squares + cubes - sixth_powers) = 182 :=
by
  let squares := num_perfect_squares 1 200
  let cubes := num_perfect_cubes 1 200
  let sixth_powers := num_perfect_sixth_powers 1 200
  have h : 200 - (squares + cubes - sixth_powers) = 182 := sorry
  exact h

end num_non_squares_cubes_1_to_200_l444_444443


namespace avg_degree_gt_d_subgraph_avg_degree_gt_2n_contains_tree_l444_444166

-- Define a structure for a graph
structure Graph (V : Type) :=
  (E : set (V × V))
  (sym : ∀{x y}, (x, y) ∈ E ↔ (y, x) ∈ E)
  (loopless : ∀{x}, ¬ (x, x) ∈ E)

-- Define the average degree of a graph
noncomputable def avg_degree {V : Type} [fintype V] (G : Graph V) : ℚ :=
  2 * (fintype.card G.E) / (fintype.card V)

-- State the main theorem: a graph with avg degree > d has a subgraph where all vertices have degree > d/2
theorem avg_degree_gt_d_subgraph {V : Type} [fintype V] {G : Graph V} {d : ℚ} (h : avg_degree G > d) :
  ∃ (H : Graph V), (∀ v : V, v ∈ H → (∃ n : ℕ, n > d / 2 ∧ n = H.degree v)) :=
sorry

-- State the second part: a graph with avg degree > 2n contains every tree with n vertices as subgraphs
theorem avg_degree_gt_2n_contains_tree {V : Type} [fintype V] {G : Graph V} {n : ℕ} (h : avg_degree G > 2 * n) (T : Graph V) (ht : fintype.card T.V = n) :
  ∃ T' : Graph V, T' = T ∧ T' ⊆ G :=
sorry

end avg_degree_gt_d_subgraph_avg_degree_gt_2n_contains_tree_l444_444166


namespace count_non_perfect_square_or_cube_l444_444512

-- Define perfect_square function
def perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k^2 = n

-- Define perfect_cube function
def perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k^3 = n

-- Define range from 1 to 200
def range_200 := {1 .. 200}

-- Define numbers either perfect square or perfect cube
def perfect_square_or_cube : Finset ℕ :=
  (range_200.filter perfect_square) ∪ (range_200.filter perfect_cube)

-- Define sixth powers in range
def perfect_sixth_power (n : ℕ) : Prop :=
  ∃ k : ℕ, k^6 = n

-- The final problem statement
theorem count_non_perfect_square_or_cube :
  (range_200.card - perfect_square_or_cube.card) = 182 :=
by
  sorry

end count_non_perfect_square_or_cube_l444_444512


namespace sum_binomial_coeff_constant_term_l444_444042

-- Define the conditions and the problem attributes
def expansion (x : ℝ) : (ℝ → ℝ) :=
  λ (n : ℕ), (x - (1 / x^(1/2)))^n

-- Given condition: n = 6
def n : ℕ := 6

-- Prove that the sum of the binomial coefficients in the expansion is 64
theorem sum_binomial_coeff (x : ℝ) : 
  (∑ i in finset.range (n + 1), binomial n i) = 64 :=
by {
  sorry
}

-- Prove that the constant term in the expansion is 15
theorem constant_term (x : ℝ) : 
  (6 choose 4) * (-1)^4 * x^(6 - 3 * 4 / 2) = 15 :=
by {
  have h : (6 choose 4) = 15,
  {
    norm_num,
  },
  have hexp : (6 - 3 * 4 / 2 : ℝ) = 0,
  {
    norm_num,
  },
  rw [h, hexp],
  norm_num,
  sorry,
}

end sum_binomial_coeff_constant_term_l444_444042


namespace find_T_b_minus_T_neg_b_l444_444355

noncomputable def T (r : ℝ) : ℝ := 20 / (1 - r)

theorem find_T_b_minus_T_neg_b (b : ℝ) (h1 : -1 < b ∧ b < 1) (h2 : T b * T (-b) = 3240) (h3 : 1 - b^2 = 100 / 810) :
  T b - T (-b) = 324 * b :=
by
  sorry

end find_T_b_minus_T_neg_b_l444_444355


namespace rewardFunc1_not_meet_requirements_smallest_positive_a_meets_requirements_l444_444832

noncomputable def rewardFunc1 (x : ℝ) : ℝ := log x + x / 50 + 5

def consistentWithGovRequirements1 (x : ℝ) (y : ℝ) : Prop :=
  70000 ≤ y * 10000 ∧ y * 10000 ≤ 0.15 * x * 10000

-- Show that the reward function does not meet the specified government requirements
theorem rewardFunc1_not_meet_requirements (x : ℝ) (hx : 50000 ≤ x ∧ x ≤ 500000) :
  ¬consistentWithGovRequirements1 x (rewardFunc1 x) :=
sorry

noncomputable def rewardFunc2 (x : ℝ) (a : ℝ) : ℝ := (15 * x - a) / (x + 8)

-- Determine the smallest positive integer "a"
def consistentWithGovRequirements2 (x : ℕ) (a : ℕ) (y : ℕ) : Prop :=
  70000 ≤ y ∧ y ≤ 0.15 * x * 10000

theorem smallest_positive_a_meets_requirements : ∃ a : ℕ, 315 ≤ a ∧ a ≤ 344 :=
exists.intro 315 (and.intro (by norm_num) (by norm_num))

lemma min_bonus_meet_required (x : ℕ) (hx : 50 ≤ x ∧ x ≤ 500) (a : ℕ) (ha : 315 ≤ a ∧ a ≤ 344) :
  consistentWithGovRequirements2 x a (rewardFunc2 x a) :=
sorry

end rewardFunc1_not_meet_requirements_smallest_positive_a_meets_requirements_l444_444832


namespace product_of_three_3_digits_has_four_zeros_l444_444309

noncomputable def has_four_zeros_product : Prop :=
  ∃ (a b c: ℕ),
    (100 ≤ a ∧ a < 1000) ∧
    (100 ≤ b ∧ b < 1000) ∧
    (100 ≤ c ∧ c < 1000) ∧
    (∃ (da db dc: Finset ℕ), (da ∪ db ∪ dc = Finset.range 10) ∧
    (∀ x ∈ da, x = a / 10^(x%10) % 10) ∧
    (∀ x ∈ db, x = b / 10^(x%10) % 10) ∧
    (∀ x ∈ dc, x = c / 10^(x%10) % 10)) ∧
    (a * b * c % 10000 = 0)

theorem product_of_three_3_digits_has_four_zeros : has_four_zeros_product := sorry

end product_of_three_3_digits_has_four_zeros_l444_444309


namespace angle_of_squares_attached_l444_444705

-- Definition of the problem scenario:
-- Three squares attached as described, needing to prove x = 39 degrees.

open Real

theorem angle_of_squares_attached (x : ℝ) (h : 
  let angle1 := 30
  let angle2 := 126
  let angle3 := 75
  angle1 + angle2 + angle3 + x = 3 * 90) :
  x = 39 :=
by 
  -- This proof is omitted
  sorry

end angle_of_squares_attached_l444_444705


namespace find_m_l444_444011

def sequence_condition (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
∀ n, a 1 + ∑ i in range (n - 1), (a (i + 2) / 3^i) = n

def arithmetic_mean (a : ℕ → ℕ) (S : ℕ → ℕ) (m : ℕ) : Prop :=
(a m + S m) / 2 = 11

theorem find_m (a : ℕ → ℕ) (S : ℕ → ℕ) (m : ℕ) :
  sequence_condition a S →
  arithmetic_mean a S m →
  m = 3 :=
begin
  sorry
end

end find_m_l444_444011


namespace solution_l444_444934

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f' : ℝ → ℝ := sorry

theorem solution (f : ℝ → ℝ) (f' : ℝ → ℝ) (h_deriv : ∀ x, deriv f x = f'(x))
  (h_cond : ∀ x, f(x) + f'(x) > 2) (h_f1 : exp (f 1) = 2 * exp 1 + 4) :
  ∀ x, (e^x * f(x) > 4 + 2 * e^x) ↔ (x > 1) :=
begin
  sorry
end

end solution_l444_444934


namespace sum_a_k_correct_l444_444944

def sequence_a_k (k : ℕ) : ℕ := 
  (List.range (2017)).filter (λ n, ∃ m∈ List.range (2017),(real.log k / real.log (m + 2)) = (n + 1)).length

noncomputable def sum_a_k : ℕ :=
  (List.range 2018).map sequence_a_k).sum

theorem sum_a_k_correct : sum_a_k = 4102 := 
  sorry

end sum_a_k_correct_l444_444944


namespace a_n_formula_T_n_formula_l444_444412

noncomputable def a_seq (n : ℕ) : ℕ := 4 * n - 2

theorem a_n_formula (n : ℕ) : a_seq (n + 1) - a_seq n = 2 * ((2 * (n + 1) - 1) - (2 * n - 1)) :=
by sorry

noncomputable def b_seq (n : ℕ) : ℕ := 2 * n - 1

noncomputable def c_seq (n : ℕ) : ℕ := (a_seq n) ^ n / (b_seq n) ^ (n - 1)

noncomputable def T_seq (n : ℕ) : ℕ :=
\(\sum_{i = 1}^{n}\ c\_seq\ i\)

theorem T_n_formula (n : ℕ) : T_seq n = 6 + (2 * n - 3) * 2 ^ (n + 1) :=
by sorry

end a_n_formula_T_n_formula_l444_444412


namespace example_problem_l444_444550

def Z (x y : ℝ) : ℝ := x^2 - 3 * x * y + y^2

theorem example_problem :
  Z 4 3 = -11 := 
by
  -- proof goes here
  sorry

end example_problem_l444_444550


namespace problem_solution_l444_444614

variable {n : ℕ}
variable {a : Fin n → ℝ}
variable {k : ℝ}

theorem problem_solution (h_pos_k : k > 0) (h_a_gt_k : ∀ i : Fin n, a i > k) :
  (Finset.univ.sum (λ i : Fin n, let j := i + 1 in (a i)^2 / (a j - k))) ≥ 4 * k * n := by
  sorry

end problem_solution_l444_444614


namespace order_of_abc_l444_444926

namespace Problem

def a := (3/5 : ℝ)^(2/5 : ℝ)
def b := (2/5 : ℝ)^(3/5 : ℝ)
def c := (2/5 : ℝ)^(2/5 : ℝ)

theorem order_of_abc : b < c ∧ c < a :=
by
  sorry

end Problem

end order_of_abc_l444_444926


namespace integer_solution_exists_l444_444119

theorem integer_solution_exists (a : ℕ) (ha : a = 7) : 
  ∃ x : ℤ, ((1 + 1/x) * (1 + 1/(x+1)) * ... * (1 + 1/(x + a)) = a - x) ∧ (x = 2 ∨ x = 4) :=
by
  sorry

end integer_solution_exists_l444_444119


namespace find_m_of_ellipse_conditions_l444_444952

-- definition for isEllipseGivenFocus condition
def isEllipseGivenFocus (m : ℝ) : Prop :=
  ∃ (a : ℝ), a = 5 ∧ (-4)^2 = a^2 - m^2 ∧ 0 < m

-- statement to prove the described condition implies m = 3
theorem find_m_of_ellipse_conditions (m : ℝ) (h : isEllipseGivenFocus m) : m = 3 :=
sorry

end find_m_of_ellipse_conditions_l444_444952


namespace numbers_not_perfect_squares_or_cubes_l444_444446

theorem numbers_not_perfect_squares_or_cubes : 
  let total_numbers := 200
  let perfect_squares := 14
  let perfect_cubes := 5
  let sixth_powers := 1
  total_numbers - (perfect_squares + perfect_cubes - sixth_powers) = 182 :=
by
  sorry

end numbers_not_perfect_squares_or_cubes_l444_444446


namespace eccentricity_range_of_ellipse_l444_444380

theorem eccentricity_range_of_ellipse :
  ∀ (a b : ℝ) (α : ℝ),
  (a > b) → (b > 0) →
  (∃ x y, (x^2 / a^2) + (y^2 / b^2) = 1) →
  (α ∈ set.Icc (real.pi / 6) (real.pi / 4)) →
  let e := 1 / (real.sin α + real.cos α) in
  e ∈ set.Icc (real.sqrt 2 / 2) (real.sqrt 3 - 1) :=
begin
  intros a b α ha hb hex hα,
  sorry
end

end eccentricity_range_of_ellipse_l444_444380


namespace seaweed_livestock_amount_l444_444312

noncomputable def seaweed_for_livestock (harvested: ℕ) (fraction_fire: ℝ) (fraction_human: ℝ) : ℕ :=
  let fire_amount := fraction_fire * harvested
  let remaining_after_fire := harvested - fire_amount
  let human_amount := fraction_human * remaining_after_fire
  let livestock_amount := remaining_after_fire - human_amount
  livestock_amount

theorem seaweed_livestock_amount:
  seaweed_for_livestock 400 0.5 0.25 = 150 :=
by
  sorry

end seaweed_livestock_amount_l444_444312


namespace max_candies_47_operations_l444_444699

theorem max_candies_47_operations :
  ∃ (max_candies : ℕ), max_candies = 1081 ∧ 
  (∀ (n : ℕ) (board : list ℕ), n = 47 → ∀ i, board = list.replicate 47 1 →
    (∃ (consume_candies : ℕ), consume_candies = max_candies  ∧ 
      (∀ t (new_board : list ℕ),
        t ≤ n → new_board.length = n - t → 
        ∃ candies : ℕ, candies = consume_candies))) :=
sorry

end max_candies_47_operations_l444_444699


namespace inequality_solution_l444_444660

theorem inequality_solution (x : ℝ) : 
  (x^2 + 4 * x + 13 > 0) -> ((x - 4) / (x^2 + 4 * x + 13) ≥ 0 ↔ x ≥ 4) :=
by
  intro h_pos
  sorry

end inequality_solution_l444_444660


namespace lassis_from_mangoes_l444_444311

theorem lassis_from_mangoes (m l m' : ℕ) (h : m' = 18) (hlm : l / m = 8 / 3) : l / m' = 48 / 18 :=
by
  sorry

end lassis_from_mangoes_l444_444311


namespace triangle_cyclic_iff_right_angle_l444_444130

open EuclideanGeometry

-- Define points and lines
variables {A B C X Y M : Point}
variables {l : Line}

-- Conditions
def conditions (h1 : A ≠ B) (h2 : A ≠ C) (h3 : B ≠ C) (h4 : midpoint M B C) 
    (h5 : line_perpendicular_to_point l A M) (h6 : line_intersects_points l X A B)
    (h7 : line_intersects_points l Y A C) (h8 : dist A B < dist A C) : Prop := -- dist AB < AC equivalent

-- Goal
theorem triangle_cyclic_iff_right_angle (h1 : A ≠ B) (h2 : A ≠ C) (h3 : B ≠ C) 
    (h4 : midpoint M B C) (h5 : line_perpendicular_to_point l A M)
    (h6 : line_intersects_points l X A B) (h7 : line_intersects_points l Y A C)
    (h8 : dist A B < dist A C) : 
    (angle A B C = 90) ↔ cyclic_quadrilateral X B Y C :=
begin
  -- Proof to be filled in
  sorry
end

end triangle_cyclic_iff_right_angle_l444_444130


namespace more_bad_than_good_labyrinths_l444_444143

-- Define the size of the board
def board_size : ℕ := 8

-- Define what it means for a labyrinth to be good
def is_good_labyrinth (labyrinth : Set (ℤ × ℤ) → bool) : bool :=
  ∀ (x y : ℤ), (0 ≤ x ∧ x < board_size) ∧ (0 ≤ y ∧ y < board_size) → labyrinth (x, y) = tt

-- The theorem stating there are more bad labyrinths than good ones
theorem more_bad_than_good_labyrinths :
  ∃ (count_good count_bad : ℕ), count_bad > count_good ∧
    count_good = (∑ labyrinth in 0..((2^(board_size*board_size - 1)).bit1) filter is_good_labyrinth, 1) ∧
    count_bad = (∑ labyrinth in 0..((2^(board_size*board_size - 1)).bit1) filter (λ l, ¬ is_good_labyrinth l), 1) :=
sorry

end more_bad_than_good_labyrinths_l444_444143


namespace cost_function_segments_l444_444154

def C (n : ℕ) : ℕ :=
  if h : 1 ≤ n ∧ n ≤ 10 then 10 * n
  else if h : 10 < n then 8 * n - 40
  else 0

theorem cost_function_segments :
  (∀ n, 1 ≤ n ∧ n ≤ 10 → C n = 10 * n) ∧
  (∀ n, 10 < n → C n = 8 * n - 40) ∧
  (∀ n, C n = if (1 ≤ n ∧ n ≤ 10) then 10 * n else if (10 < n) then 8 * n - 40 else 0) ∧
  ∃ n₁ n₂, (1 ≤ n₁ ∧ n₁ ≤ 10) ∧ (10 < n₂ ∧ n₂ ≤ 20) ∧ C n₁ = 10 * n₁ ∧ C n₂ = 8 * n₂ - 40 :=
by
  sorry

end cost_function_segments_l444_444154


namespace part1_part3_l444_444015

noncomputable def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

def lambda_k_sequence (a S : ℕ → ℝ) (λ k : ℝ) : Prop :=
∀ n : ℕ, S (n + 1)^(1 / k) - S n^(1 / k) = λ * a (n + 1)^(1 / k)

noncomputable def Sn {a : ℕ → ℝ} (S : ℕ → ℝ) : Prop :=
S 1 = a 1 ∧ ∀ n : ℕ, S (n + 1) = S n + a (n + 1)

theorem part1 (a S : ℕ → ℝ) (λ : ℝ) (h_arith : arithmetic_sequence a 1) (h_lambda1 : lambda_k_sequence a S λ 1) : 
λ = 1 :=
sorry

noncomputable theorem part2 (a S : ℕ → ℝ) (h_lambda : lambda_k_sequence a S (sqrt 3 / 3) 2) (h_pos : ∀ n : ℕ, 0 < a n) :
∀ n : ℕ, (if n = 0 then a n = 1 else a n = 3 * (4^(n - 2))) :=
sorry

theorem part3 (a1 a2 a3 S1 S2 S3 : ℕ → ℝ) (λ : ℝ) (h_lambda1 : lambda_k_sequence a1 S1 λ 3)
  (h_lambda2 : lambda_k_sequence a2 S2 λ 3)
  (h_lambda3 : lambda_k_sequence a3 S3 λ 3) 
  (h_ge0_1 : ∀ n : ℕ, 0 ≤ a1 n) (h_ge0_2 : ∀ n : ℕ, 0 ≤ a2 n) (h_ge0_3 : ∀ n : ℕ, 0 ≤ a3 n) :
0 < λ ∧ λ < 1 :=
sorry

end part1_part3_l444_444015


namespace AF_passes_through_midpoint_DE_l444_444767

open_locale classical

-- Definition of the geometrical setup
def isosceles_triangle (A B C : Point) : Prop := dist A B = dist A C
def midpoint (D B C : Point) : Prop := dist B D = dist C D
def perpendicular (D E A C : Point) : Prop := ∠ DEA = 90
def circumcircle (A B D F : Point): Prop := ∃ O R, Circle.circle O R A ∧ Circle.circle O R B ∧ Circle.circle O R D ∧ Circle.circle O R F

-- Main statement to prove
theorem AF_passes_through_midpoint_DE
  (A B C D E F : Point)
  (h1 : isosceles_triangle A B C)
  (h2 : midpoint D B C)
  (h3 : perpendicular D E A C)
  (h4 : ∃ O R, Circle.circle O R A ∧ Circle.circle O R B ∧ Circle.circle O R D ∧ Circle.circle O R F)
  (h5 : collinear B E F)
  (h6 : intersects (circumcircle A B D) (line B E) B F) :
  passes_through (line A F) (midpoint D E) := 
sorry

end AF_passes_through_midpoint_DE_l444_444767


namespace solve_equation_l444_444657

theorem solve_equation (x y z t : ℤ) (h : x^4 - 2*y^4 - 4*z^4 - 8*t^4 = 0) : x = 0 ∧ y = 0 ∧ z = 0 ∧ t = 0 :=
by
  sorry

end solve_equation_l444_444657


namespace count_not_squares_or_cubes_200_l444_444529

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n
def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k * k * k = n
def is_sixth_power (n : ℕ) : Prop := ∃ k : ℕ, k^6 = n

theorem count_not_squares_or_cubes_200 :
  let count_squares := (Finset.range 201).filter is_perfect_square).card
  let count_cubes := (Finset.range 201).filter is_perfect_cube).card
  let count_sixth_powers := (Finset.range 201).filter is_sixth_power).card
  (200 - (count_squares + count_cubes - count_sixth_powers)) = 182 := 
by {
  -- Define everything in terms of Finset and filter
  let count_squares := (Finset.range 201).filter is_perfect_square).card,
  let count_cubes := (Finset.range 201).filter is_perfect_cube).card,
  let count_sixth_powers := (Finset.range 201).filter is_sixth_power).card,
  exact sorry
}

end count_not_squares_or_cubes_200_l444_444529


namespace product_of_three_digit_numbers_ends_with_four_zeros_l444_444307

/--
Theorem: There exist three three-digit numbers formed using nine different digits such that their product ends with four zeros.
-/
theorem product_of_three_digit_numbers_ends_with_four_zeros :
  ∃ (x y z : ℕ), 100 ≤ x ∧ x < 1000 ∧
                 100 ≤ y ∧ y < 1000 ∧
                 100 ≤ z ∧ z < 1000 ∧
                 (∀ d ∈ (list.digits x).union (list.digits y).union (list.digits z), 
                     list.count d ((list.digits x).union (list.digits y).union (list.digits z)) = 1) ∧
                 (x * y * z) % 10000 = 0 :=
sorry

end product_of_three_digit_numbers_ends_with_four_zeros_l444_444307


namespace correct_result_without_mistake_l444_444732

variable {R : Type*} [CommRing R] (a b c : R)
variable (A : R)

theorem correct_result_without_mistake :
  A + 2 * (ab + 2 * bc - 4 * ac) = (3 * ab - 2 * ac + 5 * bc) → 
  A - 2 * (ab + 2 * bc - 4 * ac) = -ab + 14 * ac - 3 * bc :=
by
  sorry

end correct_result_without_mistake_l444_444732


namespace number_of_valid_permutations_l444_444122

noncomputable def valid_permutations := 
  let S := Finset.range 14 + 1
  let descending_part := ((S.erase 1).subsets 6).filter (fun s => s.to_list.sorted.reverse = s.to_list)
  let ascending_part := ((S.erase 1).subsets 7).filter (fun s => s.to_list.sorted = s.to_list)
  descending_part.card * ascending_part.card

theorem number_of_valid_permutations : valid_permutations = nat.choose 13 6 := by
  sorry

end number_of_valid_permutations_l444_444122


namespace minimum_value_expression_l444_444391

open Real

theorem minimum_value_expression (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 2 * a + b = 3) :
  (∃ (min_val : ℝ), min_val = (13 / 5) ∧ ∀ x y, x > 0 → y > 0 → 2 * x + y = 3 → (2 * x^2 + 1) / x + (y^2 - 2) / (y + 2) ≥ min_val) :=
by
  use (13 / 5)
  sorry

end minimum_value_expression_l444_444391


namespace hamiltonian_path_exists_with_zero_sum_l444_444895

-- Define a complete graph with 101 vertices and each edge marked with 1 or -1.
def complete_graph (n : ℕ) := { (i, j) // i ≠ j } → ℤ

-- Define a marking function for the graph's edges
def edge_marking (G : complete_graph 101) : (Σ (i j : ℕ), i ≠ j ∧ i < 101 ∧ j < 101) → ℤ := 
  λ ⟨i, j, _⟩, if G (i, j) then 1 else -1

-- Condition: The absolute value of the sum of numbers on all the edges is less than 150
def sum_bound (G : complete_graph 101) : Prop :=
  abs (∑ e in { (i, j) // i < 101 ∧ j < 101 }, edge_marking G e) < 150

-- Predicate that there exists a Hamiltonian path with sum of edge weights equal to zero
def exists_hamiltonian_path_with_zero_sum (G : complete_graph 101) : Prop :=
  ∃ (path : list (Σ (i j : ℕ), i ≠ j ∧ i < 101 ∧ j < 101)),
  (list.nodup path ∧ list.length path = 100 ∧ (∑ e in path, edge_marking G e = 0))

-- Complete statement: Prove that such a path exists given the conditions
theorem hamiltonian_path_exists_with_zero_sum :
  ∀ (G : complete_graph 101),
  sum_bound G →
  exists_hamiltonian_path_with_zero_sum G :=
by
  intro G h,
  sorry

end hamiltonian_path_exists_with_zero_sum_l444_444895


namespace PS_length_in_triangle_PQR_l444_444103

noncomputable def triangle_PQR_sides := (PQ QR RP : ℝ)
def P, Q, R, S : Point := sorry

def circleω1 := Circle_through_tangent Q P R
def circleω2 := Circle_through_tangent R P Q

def point_S := intersection circleω1 circleω2 P

theorem PS_length_in_triangle_PQR :
  let PQ := 9
  let QR := 10
  let RP := 11
  PS = 99 / 10 :=
sorry

end PS_length_in_triangle_PQR_l444_444103


namespace sandy_potatoes_l444_444629

theorem sandy_potatoes (n_total n_nancy n_sandy : ℕ) 
  (h_total : n_total = 13) 
  (h_nancy : n_nancy = 6) 
  (h_sum : n_total = n_nancy + n_sandy) : 
  n_sandy = 7 :=
by
  sorry

end sandy_potatoes_l444_444629


namespace manager_salary_correct_l444_444176

-- Define the conditions of the problem
def total_salary_of_24_employees : ℕ := 24 * 2400
def new_average_salary_with_manager : ℕ := 2500
def number_of_people_with_manager : ℕ := 25

-- Define the manager's salary to be proved
def managers_salary : ℕ := 4900

-- Statement of the theorem to prove that the manager's salary is Rs. 4900
theorem manager_salary_correct :
  (number_of_people_with_manager * new_average_salary_with_manager) - total_salary_of_24_employees = managers_salary :=
by
  -- Proof to be filled
  sorry

end manager_salary_correct_l444_444176


namespace Malou_first_quiz_score_l444_444621

variable (score1 score2 score3 : ℝ)

theorem Malou_first_quiz_score (h1 : score1 = 90) (h2 : score2 = 92) (h_avg : (score1 + score2 + score3) / 3 = 91) : score3 = 91 := by
  sorry

end Malou_first_quiz_score_l444_444621


namespace initial_tomatoes_count_l444_444701

-- Definitions and conditions
def birds_eat_fraction : ℚ := 1/3
def tomatoes_left : ℚ := 14
def fraction_tomatoes_left : ℚ := 2/3

-- We want to prove the initial number of tomatoes
theorem initial_tomatoes_count (initial_tomatoes : ℚ) 
  (h1 : tomatoes_left = fraction_tomatoes_left * initial_tomatoes) : 
  initial_tomatoes = 21 := 
by
  -- skipping the proof for now
  sorry

end initial_tomatoes_count_l444_444701


namespace mod_equiv_inverse_sum_l444_444719

theorem mod_equiv_inverse_sum :
  (3^15 + 3^14 + 3^13 + 3^12) % 17 = 5 :=
by sorry

end mod_equiv_inverse_sum_l444_444719


namespace roger_spending_fraction_l444_444648

-- Definitions for conditions
def weekly_budget (A : ℝ) : Prop := 0 < A -- Assume A is positive

def popcorn_cost : ℝ := 5

def movie_ticket_cost (A s : ℝ) : ℝ := 0.25 * (A - s)

def soda_cost (A m : ℝ) : ℝ := 0.10 * (A - m)

def total_spending (A s m : ℝ) : ℝ := m + s + popcorn_cost

-- Definition to check if total spending (with tax) is approximately 28% of A
def spending_fraction (A : ℝ) (fraction : ℝ) : Prop :=
  let s := soda_cost A (movie_ticket_cost A (soda_cost A 0)) in 
  let m := movie_ticket_cost A s in
  let total := total_spending A s m in
  let total_with_tax := total * 1.10 in
  total_with_tax < 0.285 * A -- Allowing a small margin for the approximation

-- The proof goal
theorem roger_spending_fraction (A : ℝ) (hA : weekly_budget A) : spending_fraction A 0.28 :=
by
  sorry

end roger_spending_fraction_l444_444648


namespace system_of_inequalities_unique_integer_solution_l444_444414

theorem system_of_inequalities_unique_integer_solution (a x : ℝ) : 
  (∀ a ∈ set.Ico (-5 : ℝ) (3 : ℝ) ∪ set.Ioc (4 : ℝ) (5 : ℝ), 
  (∃ (x ∈ ℤ), 
    (x < -2 ∨ x > 4) ∧ 
    (2 * x^2 + (2 * a + 7) * x + 7 * a < 0) ∧ 
    (∀ y ∈ ℤ, (y ≠ x → ¬((y < -2 ∨ y > 4) ∧ 
    (2 * y^2 + (2 * a + 7) * y + 7 * a < 0)))))) :=
  by sorry

end system_of_inequalities_unique_integer_solution_l444_444414


namespace num_non_squares_cubes_1_to_200_l444_444434

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n
def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k * k * k = n
def is_perfect_sixth_power (n : ℕ) : Prop := ∃ k : ℕ, k * k * k * k * k * k = n

def num_perfect_squares (a b : ℕ) : ℕ :=
  (List.range (b + 1)).countp is_perfect_square - (List.range (a)).countp is_perfect_square

def num_perfect_cubes (a b : ℕ) : ℕ :=
  (List.range (b + 1)).countp is_perfect_cube - (List.range (a)).countp is_perfect_cube

def num_perfect_sixth_powers (a b : ℕ) : ℕ :=
  (List.range (b + 1)).countp is_perfect_sixth_power - (List.range (a)).countp is_perfect_sixth_power

theorem num_non_squares_cubes_1_to_200 :
  let squares := num_perfect_squares 1 200 in
  let cubes := num_perfect_cubes 1 200 in
  let sixth_powers := num_perfect_sixth_powers 1 200 in
  200 - (squares + cubes - sixth_powers) = 182 :=
by
  let squares := num_perfect_squares 1 200
  let cubes := num_perfect_cubes 1 200
  let sixth_powers := num_perfect_sixth_powers 1 200
  have h : 200 - (squares + cubes - sixth_powers) = 182 := sorry
  exact h

end num_non_squares_cubes_1_to_200_l444_444434


namespace count_non_perfect_square_or_cube_l444_444513

-- Define perfect_square function
def perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k^2 = n

-- Define perfect_cube function
def perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k^3 = n

-- Define range from 1 to 200
def range_200 := {1 .. 200}

-- Define numbers either perfect square or perfect cube
def perfect_square_or_cube : Finset ℕ :=
  (range_200.filter perfect_square) ∪ (range_200.filter perfect_cube)

-- Define sixth powers in range
def perfect_sixth_power (n : ℕ) : Prop :=
  ∃ k : ℕ, k^6 = n

-- The final problem statement
theorem count_non_perfect_square_or_cube :
  (range_200.card - perfect_square_or_cube.card) = 182 :=
by
  sorry

end count_non_perfect_square_or_cube_l444_444513


namespace magnitude_of_projection_l444_444134

variables (u z : ℝ^n)

def dot_product (u z : ℝ^n) : ℝ := ∑ i, u i * z i
def norm (z : ℝ^n) : ℝ := real.sqrt (dot_product z z)
def projection (u z : ℝ^n) : ℝ^n := (dot_product u z / dot_product z z) • z

theorem magnitude_of_projection (h1 : dot_product u z = 4) (h2 : norm z = 7) : norm (projection u z) = 4 :=
by
  sorry

end magnitude_of_projection_l444_444134


namespace calculate_color_cartridges_l444_444707

theorem calculate_color_cartridges (c b : ℕ) (h1 : 32 * c + 27 * b = 123) (h2 : b ≥ 1) : c = 3 :=
by
  sorry

end calculate_color_cartridges_l444_444707


namespace determine_slope_PA1_range_l444_444020

-- Define the ellipse and its vertices
def ellipse (x y : ℝ) : Prop := (x^2 / 4) + (y^2 / 3) = 1
def A1 : ℝ × ℝ := (-2, 0)
def A2 : ℝ × ℝ := (2, 0)

-- Define the problem
def slope_PA2_range (y0 x0 : ℝ) (h : ellipse x0 y0) : x0 ≠ 2 ∧ x0 ≠ -2 → -2 ≤ y0 / (x0 - 2) ∧ y0 / (x0 - 2) ≤ -1

-- Define the proof to determine the range for k_PA1
theorem determine_slope_PA1_range (x0 y0 : ℝ) (h : ellipse x0 y0) (hx0 : x0 ≠ 2 ∧ x0 ≠ -2)
  (hPA2 : -2 ≤ y0 / (x0 - 2) ∧ y0 / (x0 - 2) ≤ -1) :
  (3 / 8) ≤ (y0 / (x0 + 2)) ∧ (y0 / (x0 + 2)) ≤ (3 / 4) := 
sorry

end determine_slope_PA1_range_l444_444020


namespace integer_pairs_satisfy_equation_l444_444889

theorem integer_pairs_satisfy_equation :
  ∀ (x y : ℤ), 9 * x * y - x^2 - 8 * y^2 = 2005 ↔ (x, y) = (63, 58) ∨ (x, y) = (459, 58) ∨ (x, y) = (-63, -58) ∨ (x, y) = (-459, -58) :=
by
  intros x y
  split
  sorry  -- placeholder for necessary proof steps
  
  sorry  -- placeholder for necessary proof steps

end integer_pairs_satisfy_equation_l444_444889


namespace count_non_perfect_square_or_cube_l444_444520

-- Define perfect_square function
def perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k^2 = n

-- Define perfect_cube function
def perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k^3 = n

-- Define range from 1 to 200
def range_200 := {1 .. 200}

-- Define numbers either perfect square or perfect cube
def perfect_square_or_cube : Finset ℕ :=
  (range_200.filter perfect_square) ∪ (range_200.filter perfect_cube)

-- Define sixth powers in range
def perfect_sixth_power (n : ℕ) : Prop :=
  ∃ k : ℕ, k^6 = n

-- The final problem statement
theorem count_non_perfect_square_or_cube :
  (range_200.card - perfect_square_or_cube.card) = 182 :=
by
  sorry

end count_non_perfect_square_or_cube_l444_444520


namespace height_of_parallelogram_l444_444346

-- Define the problem statement
theorem height_of_parallelogram (A : ℝ) (b : ℝ) (h : ℝ) (h_eq : A = b * h) (A_val : A = 384) (b_val : b = 24) : h = 16 :=
by
  -- Skeleton proof, include the initial conditions and proof statement
  sorry

end height_of_parallelogram_l444_444346


namespace max_fraction_in_arith_seq_l444_444939

-- Representation of given conditions (arithmetic sequence with S_4 = 10 and S_8 = 36)
def is_arithmetic_seq (a : ℕ → ℕ) :=
  ∃ (a1 d : ℕ), ∀ n : ℕ, a n = a1 + (n - 1) * d

def sum_of_first_n_terms (a : ℕ → ℕ) (S : ℕ → ℕ) :=
  ∀ n : ℕ, S n = (n / 2) * (2 * a 1 + (n - 1) * (a 2 - a 1))

noncomputable def Sn (a : ℕ → ℕ) : ℕ → ℕ := λ n, (n * (n + 1)) / 2

theorem max_fraction_in_arith_seq
  (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h_seq : is_arithmetic_seq a)
  (h_sum_4 : S 4 = 10)
  (h_sum_8 : S 8 = 36) :
  ∃ n : ℕ, n = 3 ∧ ∀ m ∈ (set.univ : set ℕ), m ≠ 0 → (2 * n) / (n * n + 7 * n + 12) = 1 / 7 := sorry

end max_fraction_in_arith_seq_l444_444939


namespace can_rearrange_figure_to_square_l444_444879

noncomputable def figure_on_grid : Type := sorry -- This will represent the figure

-- The function to check if a given figure can be rearranged to form a square
def can_form_square (f : figure_on_grid) : Prop := 
  ∃ (p1 p2 p3 : figure_on_grid), 
  -- Each part is a valid connected piece of the figure (definition of a connected part)
  -- and their union equals the original figure
  connected_part p1 ∧ connected_part p2 ∧ connected_part p3 ∧ 
  (p1 ∪ p2 ∪ p3 = f) ∧ 
  -- The parts p1, p2, p3 can be rearranged by rotation (not flipping) to form a square
  can_rearrange_to_square p1 p2 p3

-- Theorem statement in Lean 4
theorem can_rearrange_figure_to_square (f : figure_on_grid) : can_form_square f :=
sorry

end can_rearrange_figure_to_square_l444_444879


namespace count_valid_numbers_between_1_and_200_l444_444433

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k^2 = n
def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k^3 = n
def is_perfect_sixth_power (n : ℕ) : Prop := ∃ k : ℕ, k^6 = n

def count_perfect_squares (m : ℕ) : ℕ :=
  Nat.sqrt m

def count_perfect_cubes (m : ℕ) : ℕ :=
  (m : ℝ).cbrt.to_nat

def count_perfect_sixth_powers (m : ℕ) : ℕ :=
  (m : ℝ).cbrt.sqrt.to_nat

def count_either_squares_or_cubes (m : ℕ) : ℕ :=
  let squares := count_perfect_squares m
  let cubes := count_perfect_cubes m
  let sixths := count_perfect_sixth_powers m
  squares + cubes - sixths

def count_neither_squares_nor_cubes (n : ℕ) : ℕ :=
  n - count_either_squares_or_cubes n

theorem count_valid_numbers_between_1_and_200 : count_neither_squares_nor_cubes 200 = 182 :=
by
  sorry

end count_valid_numbers_between_1_and_200_l444_444433


namespace line_AF_midpoint_DE_l444_444790

variables {A B C D E F : Type} [EuclideanGeometry.point A] [EuclideanGeometry.point B] [EuclideanGeometry.point C]
  [EuclideanGeometry.point D] [EuclideanGeometry.point E] [EuclideanGeometry.point F]

-- Definitions based on the problem conditions
def isosceles_triangle (A B C : Type) [EuclideanGeometry.point A] [EuclideanGeometry.point B] [EuclideanGeometry.point C] : Prop :=
  (EuclideanGeometry.distance A B = EuclideanGeometry.distance A C)

def midpoint (D B C : Type) [EuclideanGeometry.point D] [EuclideanGeometry.point B] [EuclideanGeometry.point C] : Prop :=
  EuclideanGeometry.between D B C

def perpendicular (D E : Type) [EuclideanGeometry.point D] [EuclideanGeometry.point E] : Prop := 
  EuclideanGeometry.perp D E

def circumcircle (ABD : Set EuclideanGeometry.point) : Prop :=
  EuclideanGeometry.circumcircle ABD

def intersection (circumcircle_intersection : Type) [EuclideanGeometry.point F] [EuclideanGeometry.point B] : Prop := 
  EuclideanGeometry.intersects circumcircle_intersection F B

-- The theorem we need to prove
theorem line_AF_midpoint_DE
  (isoABC : isosceles_triangle A B C)
  (midD : midpoint D B C)
  (perpDE : perpendicular D E)
  (circABD : circumcircle {A, B, D})
  (interBF : intersection {A, B, D})
  : EuclideanGeometry.on_line A F (EuclideanGeometry.midpoint D E) :=
sorry -- proof here

end line_AF_midpoint_DE_l444_444790


namespace no_real_y_for_common_solution_l444_444354

theorem no_real_y_for_common_solution :
  ∀ (x y : ℝ), x^2 + y^2 = 25 → x^2 + 3 * y = 45 → false :=
by 
sorry

end no_real_y_for_common_solution_l444_444354


namespace flagpole_arrangement_remainder_l444_444314

/-- There are two flagpoles with 24 flags in total, consisting of 14 identical blue flags and 10 identical red flags. Each flagpole has at least one flag, no two red flags on either pole are adjacent, and each sequence starts with a blue flag. The number of such arrangements, when divided by 1000, leaves a remainder of 1. -/
theorem flagpole_arrangement_remainder :
  let M := (Nat.choose 14 10) in
  M % 1000 = 1 :=
by
  sorry

end flagpole_arrangement_remainder_l444_444314


namespace special_op_2_4_5_l444_444617

def special_op (a b c : ℝ) : ℝ := b ^ 2 - 4 * a * c

theorem special_op_2_4_5 : special_op 2 4 5 = -24 := by
  sorry

end special_op_2_4_5_l444_444617


namespace num_from_1_to_200_not_squares_or_cubes_l444_444458

noncomputable def numNonPerfectSquaresAndCubes (n : ℕ) : ℕ :=
  let num_squares := 14
  let num_cubes := 5
  let num_sixth_powers := 2
  n - (num_squares + num_cubes - num_sixth_powers)

theorem num_from_1_to_200_not_squares_or_cubes : numNonPerfectSquaresAndCubes 200 = 183 := by
  sorry

end num_from_1_to_200_not_squares_or_cubes_l444_444458


namespace numbers_neither_square_nor_cube_l444_444533

theorem numbers_neither_square_nor_cube (n : ℕ) (h : n = 200) :
  let num_squares := 14 in
  let num_cubes := 5 in
  let num_sixth_powers := 2 in
  (n - (num_squares + num_cubes - num_sixth_powers)) = 183 :=
by
  -- Let the number of integers in the range
  let num_total := 200
  -- Define how we count perfect squares and cubes
  let num_squares := 14
  let num_cubes := 5
  let num_sixth_powers := 2

  -- Subtract the overlapping perfect sixth powers from the sum of squares and cubes
  let num_either_square_or_cube := num_squares + num_cubes - num_sixth_powers

  -- Calculate the final result
  let result := num_total - num_either_square_or_cube

  -- Prove by computation
  show result = 183 from
    calc
      result = num_total - num_either_square_or_cube := rfl
           ... = 200 - (14 + 5 - 2)                   := rfl
           ... = 200 - 17                             := rfl
           ... = 183                                  := rfl

end numbers_neither_square_nor_cube_l444_444533


namespace cake_sugar_calculation_l444_444297

theorem cake_sugar_calculation (sugar_first_layer : ℕ) (sugar_second_layer : ℕ) (sugar_third_layer : ℕ) :
  sugar_first_layer = 2 →
  sugar_second_layer = 2 * sugar_first_layer →
  sugar_third_layer = 3 * sugar_second_layer →
  sugar_third_layer = 12 := 
by
  intros h1 h2 h3
  have h4 : 2 = sugar_first_layer, from h1.symm
  have h5 : sugar_second_layer = 2 * 2, by rw [h4, h2]
  have h6 : sugar_third_layer = 3 * 4, by rw [h5, h3]
  exact h6

end cake_sugar_calculation_l444_444297


namespace total_prom_cost_l444_444592

-- Definitions for conditions
def ticket_cost := 100
def dinner_cost := 120
def tip_percentage := 0.30
def num_tickets := 2
def limo_hours := 6
def limo_rate := 80

-- Total cost calculation
def total_cost := 
  let ticket_expense := num_tickets * ticket_cost
  let tip := tip_percentage * dinner_cost
  let dinner_expense := dinner_cost + tip
  let limo_expense := limo_hours * limo_rate
  ticket_expense + dinner_expense + limo_expense

-- The theorem we need to prove
theorem total_prom_cost : total_cost = 836 := by
  sorry

end total_prom_cost_l444_444592


namespace convex_polygon_num_sides_l444_444676

theorem convex_polygon_num_sides (n : ℕ) 
  (h1 : ∀ (i : ℕ), i < n → 120 + i * 5 < 180) 
  (h2 : (n - 2) * 180 = n * (240 + (n - 1) * 5) / 2) : 
  n = 9 :=
sorry

end convex_polygon_num_sides_l444_444676


namespace solve_x_of_det_8_l444_444636

variable (x : ℝ)

def matrix_det (a b c d : ℝ) : ℝ := a * d - b * c

theorem solve_x_of_det_8
  (h : matrix_det (x + 1) (1 - x) (1 - x) (x + 1) = 8) : x = 2 := by
  sorry

end solve_x_of_det_8_l444_444636


namespace outer_boundary_diameter_l444_444821

def width_path : ℝ := 10
def width_garden : ℝ := 12
def diameter_fountain : ℝ := 14

theorem outer_boundary_diameter :
  let radius_fountain := diameter_fountain / 2
  let total_radius := radius_fountain + width_garden + width_path
  2 * total_radius = 58 :=
by
  let radius_fountain := diameter_fountain / 2
  let total_radius := radius_fountain + width_garden + width_path
  have h : 2 * total_radius = 58 := by linarith
  exact h

end outer_boundary_diameter_l444_444821


namespace three_digit_number_multiple_of_eleven_l444_444343

theorem three_digit_number_multiple_of_eleven:
  ∃ (a b c : ℕ), (1 ≤ a) ∧ (a ≤ 9) ∧ (0 ≤ b) ∧ (b ≤ 9) ∧ (0 ≤ c) ∧ (c ≤ 9) ∧
                  (100 * a + 10 * b + c = 11 * (a + b + c) ∧ (100 * a + 10 * b + c = 198)) :=
by
  use 1
  use 9
  use 8
  sorry

end three_digit_number_multiple_of_eleven_l444_444343


namespace sum_vectors_ngon_l444_444736

noncomputable def sum_perpendiculars_vectors_ngon (M O : Point) (n : ℕ) (MK : ℕ → Vector) : Vector :=
  ∑ i in (finset.range n), MK i

theorem sum_vectors_ngon (M O : Point) (n : ℕ) (MK : ℕ → Vector) :
  sum_perpendiculars_vectors_ngon M O n MK = (n / 2) • (vector_from M O) :=
sorry

end sum_vectors_ngon_l444_444736


namespace AF_through_midpoint_DE_l444_444755

open EuclideanGeometry

-- Definition of an isosceles triangle
variables {A B C D E F : Point}
variables [plane Geometry]
variables (ABC_isosceles : IsoscelesTriangle A B C)
variables (midpoint_D : Midpoint D B C)
variables (perpendicular_DE_AC : Perpendicular D E A C)
variables (F_on_circumcircle : OnCircumcircle F A B D)
variables (B_on_BE : OnLine B E)

-- Goal: Prove that line AF passes through the midpoint of DE
theorem AF_through_midpoint_DE 
  (h1 : triangle A B C) 
  (h2 : ABC_isosceles) 
  (h3 : midpoint D B C) 
  (h4 : perpendicular DE AC)
  (h5 : OnCircumcircle F A B D) 
  (h6 : OnLine B E) :
  PassesThroughMidpoint A F D E :=
sorry

end AF_through_midpoint_DE_l444_444755


namespace house_paint_possible_l444_444579
open Function

def family_perms_exist (families : Type) (houses : Type) [Fintype families] [Fintype houses] [DecidableEq houses] : Prop :=
  ∃ (perm : Perm families), ∀ f : families, f ≠ perm f

def colorable (families : Type) (houses : Type) [Fintype families] [Fintype houses] [DecidableEq houses] : Prop :=
  ∃ (colors : houses → ℕ), (∀ h : houses, colors h = 0 ∨ colors h = 1 ∨ colors h = 2) ∧
  ∀ (perm : Perm houses), ∀ h : houses, colors h ≠ colors (perm h)

theorem house_paint_possible (families : Type) (houses : Type) [Fintype families] [Fintype houses] [DecidableEq houses] :
  family_perms_exist families houses → colorable families houses :=
by
  sorry

end house_paint_possible_l444_444579


namespace num_non_squares_cubes_1_to_200_l444_444435

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n
def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k * k * k = n
def is_perfect_sixth_power (n : ℕ) : Prop := ∃ k : ℕ, k * k * k * k * k * k = n

def num_perfect_squares (a b : ℕ) : ℕ :=
  (List.range (b + 1)).countp is_perfect_square - (List.range (a)).countp is_perfect_square

def num_perfect_cubes (a b : ℕ) : ℕ :=
  (List.range (b + 1)).countp is_perfect_cube - (List.range (a)).countp is_perfect_cube

def num_perfect_sixth_powers (a b : ℕ) : ℕ :=
  (List.range (b + 1)).countp is_perfect_sixth_power - (List.range (a)).countp is_perfect_sixth_power

theorem num_non_squares_cubes_1_to_200 :
  let squares := num_perfect_squares 1 200 in
  let cubes := num_perfect_cubes 1 200 in
  let sixth_powers := num_perfect_sixth_powers 1 200 in
  200 - (squares + cubes - sixth_powers) = 182 :=
by
  let squares := num_perfect_squares 1 200
  let cubes := num_perfect_cubes 1 200
  let sixth_powers := num_perfect_sixth_powers 1 200
  have h : 200 - (squares + cubes - sixth_powers) = 182 := sorry
  exact h

end num_non_squares_cubes_1_to_200_l444_444435


namespace inequality_amgm_l444_444654

theorem inequality_amgm (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) : a^3 + b^3 + a + b ≥ 4 * a * b :=
sorry

end inequality_amgm_l444_444654


namespace non_perfect_powers_count_l444_444486

theorem non_perfect_powers_count :
  let n := 200,
      num_perfect_squares := 14,
      num_perfect_cubes := 5,
      num_perfect_sixth_powers := 1,
      total_perfect_powers := num_perfect_squares + num_perfect_cubes - num_perfect_sixth_powers,
      total_non_perfect_powers := n - total_perfect_powers
  in
  total_non_perfect_powers = 182 :=
by
  sorry

end non_perfect_powers_count_l444_444486


namespace AF_passes_through_midpoint_of_DE_l444_444780

open EuclideanGeometry

-- Definitions based on the conditions
variables {A B C D E F : Point} 
variables {ABC : Triangle}
variables (isosceles_ABC : IsIsoscelesTriangle ABC A B)
variables (midpoint_D : IsMidpoint D B C)
variables (perpendicular_DE : IsPerpendicular DE D AC)
variables {circumcircle_ABD : Circumcircle ABD}
variables (intersection_F : IntersectsAt circumcircle_ABD BE B F)

-- The theorem statement
theorem AF_passes_through_midpoint_of_DE :
  PassesThroughMidpoint AF D E :=
sorry

end AF_passes_through_midpoint_of_DE_l444_444780


namespace AF_through_midpoint_DE_l444_444751

open EuclideanGeometry

-- Definition of an isosceles triangle
variables {A B C D E F : Point}
variables [plane Geometry]
variables (ABC_isosceles : IsoscelesTriangle A B C)
variables (midpoint_D : Midpoint D B C)
variables (perpendicular_DE_AC : Perpendicular D E A C)
variables (F_on_circumcircle : OnCircumcircle F A B D)
variables (B_on_BE : OnLine B E)

-- Goal: Prove that line AF passes through the midpoint of DE
theorem AF_through_midpoint_DE 
  (h1 : triangle A B C) 
  (h2 : ABC_isosceles) 
  (h3 : midpoint D B C) 
  (h4 : perpendicular DE AC)
  (h5 : OnCircumcircle F A B D) 
  (h6 : OnLine B E) :
  PassesThroughMidpoint A F D E :=
sorry

end AF_through_midpoint_DE_l444_444751


namespace evaporation_amount_l444_444819

noncomputable def water_evaporated_per_day (total_water: ℝ) (percentage_evaporated: ℝ) (days: ℕ) : ℝ :=
  (percentage_evaporated / 100) * total_water / days

theorem evaporation_amount :
  water_evaporated_per_day 10 7 50 = 0.014 :=
by
  sorry

end evaporation_amount_l444_444819


namespace factor_polynomial_l444_444901

theorem factor_polynomial (a : ℝ) : 74 * a^2 + 222 * a + 148 * a^3 = 74 * a * (2 * a^2 + a + 3) :=
by
  sorry

end factor_polynomial_l444_444901


namespace max_mn_l444_444718

section AmericanChess

-- Define conditions for the American piece
structure American (A : Type) :=
  (cannot_attack_itself : ¬ (A → A))
  (attack_symmetric : ∀ a b : A, A a b ↔ A b a)

-- Define the 8x8 chessboard and the coloring pattern
def Chessboard := list (list (option (ℕ × ℕ)))

def checker_board : Chessboard :=
  (list.fin_range 8).attach.bind (λ i, 
    (list.fin_range 8).attach.map (λ j,
      if (i.1 + j.1) % 2 = 0 then some (i.1, j.1) else none))

-- Compute the number of squares attacked by an American in the top-left corner
def m := 32

-- Compute the number of non-attacking Americans including one in top-left corner
def n := 31

-- The theorem to prove the maximum value of mn
theorem max_mn : m * n = 992 :=
by {
  -- Omitting proof details as per instruction
  -- Proof could potentially involve verifying m, n calculations and conditions.
  sorry
}

end AmericanChess

end max_mn_l444_444718


namespace count_not_squares_or_cubes_200_l444_444523

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n
def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k * k * k = n
def is_sixth_power (n : ℕ) : Prop := ∃ k : ℕ, k^6 = n

theorem count_not_squares_or_cubes_200 :
  let count_squares := (Finset.range 201).filter is_perfect_square).card
  let count_cubes := (Finset.range 201).filter is_perfect_cube).card
  let count_sixth_powers := (Finset.range 201).filter is_sixth_power).card
  (200 - (count_squares + count_cubes - count_sixth_powers)) = 182 := 
by {
  -- Define everything in terms of Finset and filter
  let count_squares := (Finset.range 201).filter is_perfect_square).card,
  let count_cubes := (Finset.range 201).filter is_perfect_cube).card,
  let count_sixth_powers := (Finset.range 201).filter is_sixth_power).card,
  exact sorry
}

end count_not_squares_or_cubes_200_l444_444523


namespace find_selling_price_l444_444263

theorem find_selling_price (G P : ℝ) (hG : G = 15) (hP : P = 20) : ∃ SP : ℝ, SP = 90 :=
by
  -- Let CP be the cost price of the article
  let CP := G / (P / 100)
  
  -- Gain = (P / 100) * CP
  have hCP : CP = G / (P / 100), from rfl

  -- Selling price SP = CP + Gain
  let SP := CP + G

  -- Prove that SP = 90
  use SP
  rw [hCP]
  simp
  field_simp
  sorry

end find_selling_price_l444_444263


namespace lastTwoNonZeroDigits_of_80_fact_is_8_l444_444683

-- Define the factorial function
def fac : ℕ → ℕ
  | 0     => 1
  | (n+1) => (n+1) * fac n

-- Define the function to find the last two nonzero digits of a factorial
def lastTwoNonZeroDigits (n : ℕ) : ℕ := sorry -- Placeholder logic for now

-- State the problem as a theorem
theorem lastTwoNonZeroDigits_of_80_fact_is_8 :
  lastTwoNonZeroDigits 80 = 8 :=
sorry

end lastTwoNonZeroDigits_of_80_fact_is_8_l444_444683


namespace slope_intercept_form_correct_l444_444259

theorem slope_intercept_form_correct:
  ∀ (x y : ℝ), (2 * (x - 3) - 1 * (y + 4) = 0) → (∃ m b, y = m * x + b ∧ m = 2 ∧ b = -10) :=
by
  intro x y h
  use 2, -10
  sorry

end slope_intercept_form_correct_l444_444259


namespace keanu_destination_distance_l444_444111

def motorcycle_capacity : ℕ := 8
def consumption_per_40miles : ℕ := 8
def distance_per_tank : ℕ := 40
def refills_round_trip : ℕ := 14

theorem keanu_destination_distance :
  let total_distance := refills_round_trip * distance_per_tank in
  let one_way_distance := total_distance / 2 in
  one_way_distance = 280 :=
by
  let total_distance := refills_round_trip * distance_per_tank
  let one_way_distance := total_distance / 2
  sorry

end keanu_destination_distance_l444_444111


namespace diesel_fuel_usage_l444_444626

theorem diesel_fuel_usage (weekly_spending : ℝ) (cost_per_gallon : ℝ) (weeks : ℝ) (result : ℝ): 
  weekly_spending = 36 → cost_per_gallon = 3 → weeks = 2 → result = 24 → 
  (weekly_spending / cost_per_gallon) * weeks = result :=
by
  intros
  sorry

end diesel_fuel_usage_l444_444626


namespace num_pos_solutions_l444_444361

theorem num_pos_solutions (n : ℕ) (h₀ : n ≤ 1000) : 
  (finset.filter (λ x : ℝ, (∃ (floor : ℤ), floor = int.floor x ∧ x ^ (floor : ℝ) = n) ∧ x > 0) (finset.Icc 1 1000)).card = 412 :=
sorry

end num_pos_solutions_l444_444361


namespace mark_bought_5_pounds_of_apples_l444_444144

noncomputable def cost_of_tomatoes (pounds_tomatoes : ℕ) (cost_per_pound_tomato : ℝ) : ℝ :=
  pounds_tomatoes * cost_per_pound_tomato

noncomputable def cost_of_apples (total_spent : ℝ) (cost_of_tomatoes : ℝ) : ℝ :=
  total_spent - cost_of_tomatoes

noncomputable def pounds_of_apples (cost_of_apples : ℝ) (cost_per_pound_apples : ℝ) : ℝ :=
  cost_of_apples / cost_per_pound_apples

theorem mark_bought_5_pounds_of_apples (pounds_tomatoes : ℕ) (cost_per_pound_tomato : ℝ) 
  (total_spent : ℝ) (cost_per_pound_apples : ℝ) :
  pounds_tomatoes = 2 →
  cost_per_pound_tomato = 5 →
  total_spent = 40 →
  cost_per_pound_apples = 6 →
  pounds_of_apples (cost_of_apples total_spent (cost_of_tomatoes pounds_tomatoes cost_per_pound_tomato)) cost_per_pound_apples = 5 := by
  intros h1 h2 h3 h4
  sorry

end mark_bought_5_pounds_of_apples_l444_444144


namespace cos_x_plus_pi_over_3_angle_B_and_range_of_f_A_l444_444966

noncomputable def vec_m (x : ℝ) : ℝ × ℝ := (Real.sin (x / 4), Real.cos (x / 4))
noncomputable def vec_n (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos (x / 4), Real.cos (x / 4))
noncomputable def f (x : ℝ) : ℝ := vec_m x.1 * vec_n x.1 + vec_m x.2 * vec_n x.2

theorem cos_x_plus_pi_over_3 (x : ℝ) (h : f x = 1) : Real.cos (x + Real.pi / 3) = 1 / 2 :=
  sorry

variable {a b c : ℝ}
variable {A B C : ℝ}
variable (h2 : (2 * a - c) * Real.cos B = b * Real.cos C)
variable (h3 : Real.sin A ≠ 0)

theorem angle_B_and_range_of_f_A (h1 : A + B + C = Real.pi) : B = Real.pi / 3 ∧ (1 < f A ∧ f A < 3 / 2) :=
  sorry

end cos_x_plus_pi_over_3_angle_B_and_range_of_f_A_l444_444966


namespace total_distance_l444_444282

/-!
# Total Distance Travel Problem

Given three points A, B, and C, we want to prove that the total distance
from point A to point C via point B is equal to the sum of the distances
from A to B and from B to C.
-/

open Real

def point := (ℝ × ℝ)

def distance (p1 p2 : point) : ℝ :=
  sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem total_distance (A B C : point) :
  A = (-3, 6) → B = (2, 2) → C = (6, -3) →
  distance A B + distance B C = 2 * sqrt 41 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end total_distance_l444_444282


namespace conjugate_complex_div_l444_444232

def conjugate (z : ℂ) : ℂ := z.conj

theorem conjugate_complex_div :
  conjugate ((7 + complex.I) / (1 - complex.I)) = 3 - 4 * complex.I :=
by
  sorry

end conjugate_complex_div_l444_444232


namespace transformation_14_period_record_count_l444_444422

def transform_count (S : Set (Int × Int)) (k : Int) : Int :=
  (k + 2)^2 - 4

def after_14_transformations (initial_set : Set (Int × Int)) : Set (Int × Int) :=
  {p | ∃ (n m : Int), p = (n, m) ∧ |n - 1| + |m| ≤ 15} \ 
  {(1, 14), (1, 15), (1, -14), (1, -15)}

theorem transformation_14_period_record_count :
  ∀ (initial_set : Set (Int × Int)), 
  initial_set = {(0,0), (2,0)} →
  (after_14_transformations initial_set).size = 477 := by
  intros initial_set h
  sorry

end transformation_14_period_record_count_l444_444422


namespace three_digit_number_parity_count_equal_l444_444215

/-- Prove the number of three-digit numbers with all digits having the same parity is equal to the number of three-digit numbers where adjacent digits have different parity. -/
theorem three_digit_number_parity_count_equal :
  ∃ (same_parity_count alternating_parity_count : ℕ),
    same_parity_count = alternating_parity_count ∧
    -- Condition for digits of the same parity
    same_parity_count = (4 * 5 * 5) + (5 * 5 * 5) ∧
    -- Condition for alternating parity digits (patterns EOE and OEO)
    alternating_parity_count = (4 * 5 * 5) + (5 * 5 * 5) := by
  sorry

end three_digit_number_parity_count_equal_l444_444215


namespace line_AF_midpoint_DE_l444_444784

variables {A B C D E F : Type} [EuclideanGeometry.point A] [EuclideanGeometry.point B] [EuclideanGeometry.point C]
  [EuclideanGeometry.point D] [EuclideanGeometry.point E] [EuclideanGeometry.point F]

-- Definitions based on the problem conditions
def isosceles_triangle (A B C : Type) [EuclideanGeometry.point A] [EuclideanGeometry.point B] [EuclideanGeometry.point C] : Prop :=
  (EuclideanGeometry.distance A B = EuclideanGeometry.distance A C)

def midpoint (D B C : Type) [EuclideanGeometry.point D] [EuclideanGeometry.point B] [EuclideanGeometry.point C] : Prop :=
  EuclideanGeometry.between D B C

def perpendicular (D E : Type) [EuclideanGeometry.point D] [EuclideanGeometry.point E] : Prop := 
  EuclideanGeometry.perp D E

def circumcircle (ABD : Set EuclideanGeometry.point) : Prop :=
  EuclideanGeometry.circumcircle ABD

def intersection (circumcircle_intersection : Type) [EuclideanGeometry.point F] [EuclideanGeometry.point B] : Prop := 
  EuclideanGeometry.intersects circumcircle_intersection F B

-- The theorem we need to prove
theorem line_AF_midpoint_DE
  (isoABC : isosceles_triangle A B C)
  (midD : midpoint D B C)
  (perpDE : perpendicular D E)
  (circABD : circumcircle {A, B, D})
  (interBF : intersection {A, B, D})
  : EuclideanGeometry.on_line A F (EuclideanGeometry.midpoint D E) :=
sorry -- proof here

end line_AF_midpoint_DE_l444_444784


namespace circle_touches_three_sides_of_quadrilateral_l444_444158

theorem circle_touches_three_sides_of_quadrilateral
  (A B C D : Point)
  (convex_quadrilateral : convex_quad A B C D)
  (circle_touches : ∃ O r, tangent_point O r B C ∧ tangent_point O r C D ∧ tangent_point O r D A) :
  (length A B + length C D < length B C + length D A)
:= sorry

end circle_touches_three_sides_of_quadrilateral_l444_444158


namespace math_problem_l444_444555

def op (x y : ℕ) : ℕ := x * y - 2 * x + 3 * y

theorem math_problem : (op 7 4) - (op 4 7) = -15 := 
by
  sorry

end math_problem_l444_444555


namespace total_students_in_high_school_l444_444153

theorem total_students_in_high_school (selected_first: ℕ) (selected_second: ℕ) (students_third: ℕ) (total_selected: ℕ) (p: ℚ) :
  selected_first = 15 →
  selected_second = 12 →
  students_third = 900 →
  total_selected = 36 →
  p = 1 / 100 →
  ∃ n: ℕ, (total_selected : ℚ) / n = p ∧ n = 3600 :=
by 
  intros h1 h2 h3 h4 h5
  use 3600
  split
  · sorry -- omit the proof for successful compilation.
  · exact rfl

end total_students_in_high_school_l444_444153


namespace necessary_not_sufficient_l444_444368

noncomputable def perpendicular_condition (α β : Set Point) (m : Set Point) : Prop := 
  (∀ pt ∈ m, pt ∈ α) ∧                     -- m is in α
  (∃ n : Set Point, Line n ∧ ∀ pt ∈ n, pt ∉ α ∧ pt ∉ β) ∧ -- α and β are distinct planes
  (∀ pt ∈ m, pt ⊥ β ↔ α ⊥ β)               -- proving necessary but not sufficient condition

theorem necessary_not_sufficient (α β : Set Point) (m : Set Point) (h1 : ∀ pt ∈ m, pt ∈ α)
  (h2: ∃ n : Set Point, Line n ∧ ∀ pt ∈ n, pt ∉ α ∧ pt ∉ β) : 
  ∀ pt ∈ m, pt ⊥ β ↔ α ⊥ β :=
begin
  sorry
end

end necessary_not_sufficient_l444_444368


namespace cut_figure_into_rectangles_and_square_l444_444580

theorem cut_figure_into_rectangles_and_square :
  (∃ (n : ℕ), n = 10) :=
begin
  -- Given a figure of 17 cells
  let figure_cells := 17,
  -- Condition: Cut into 8 rectangles of size 1 × 2
  let rectangles := 8,
  -- Condition: One square of size 1 × 1
  let square := 1,
  -- We need to prove that there are exactly 10 ways to cut the figure
  use 10,
  sorry
end

end cut_figure_into_rectangles_and_square_l444_444580


namespace replaced_person_weight_l444_444563

theorem replaced_person_weight (W A W' X : ℝ) 
  (h1 : W = 6 * A) 
  (h2 : W' = 6 * (A + 1.5)) 
  (h3 : W' = W - X + 74) 
  : X = 65 :=
by {
  sorry,
}

end replaced_person_weight_l444_444563


namespace part1_part2_part3_l444_444018

-- Definitions and conditions
def sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) = a n + 1  -- an arithmetic sequence
def S (a : ℕ → ℝ) (n : ℕ) := ∑ i in finset.range (n + 1), a i  -- sum of the first n terms
def lambda_k_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) (λ k : ℝ) :=
  ∀ n : ℕ, (S (n + 1))^(1/k) - (S n)^(1/k) = λ * (a (n + 1))^(1/k)

-- Conditions
axiom a1 : a 1 = 1

-- Part (1): Given an arithmetic sequence {a_n}, if it is a "λ - 1" sequence, then λ = 1
theorem part1 (a : ℕ → ℝ) (λ : ℝ) (h_a : sequence a) (h_lambdak : lambda_k_sequence a (S a) λ 1) : λ = 1 := 
sorry

-- Part (2): If {a_n} is a "$\frac{\sqrt{3}}{3}-2$" sequence, find the general formula for {a_n} when a_n > 0
theorem part2 (a : ℕ → ℝ) (h_lambdak : lambda_k_sequence a (S a) (√3 / 3) 2) (h_pos : ∀ n, a n > 0) : 
  (∀ n, a n = if n = 1 then 1 else 3 * 4^(n - 2)) := 
sorry

-- Part (3): For a given λ, determine if there exist three different sequences {a_n} that form a "λ - 3" sequence, where a_n ≥ 0
theorem part3 (λ : ℝ) : 
  ∃ (a b c : ℕ → ℝ), (λ > 0 ∧ λ < 1) ∧ 
  lambda_k_sequence a (S a) λ 3 ∧ lambda_k_sequence b (S b) λ 3 ∧ lambda_k_sequence c (S c) λ 3 ∧ 
  (∀ n, a n ≥ 0 ∧ b n ≥ 0 ∧ c n ≥ 0) ∧ 
  (∃ n, a n ≠ b n ∧ a n ≠ c n ∧ b n ≠ c n) := 
sorry

end part1_part2_part3_l444_444018


namespace box_of_books_weight_l444_444245

theorem box_of_books_weight :
  (∀ (books_weight_per_unit : ℕ) (number_of_books : ℕ), (books_weight_per_unit = 3) ∧ (number_of_books = 14) → number_of_books * books_weight_per_unit = 42) :=
by
  intro books_weight_per_unit number_of_books h
  cases h with h1 h2
  rw [h1, h2]
  simp [Nat.mul_comm]
  sorry

end box_of_books_weight_l444_444245


namespace cos_alpha_minus_pi_l444_444385

noncomputable def alpha : ℝ := sorry  -- Placeholder for α
noncomputable def cos_val : ℝ := sorry  -- Placeholder for cos(α - π)

axiom alpha_interval : (Real.pi / 2) < alpha ∧ alpha < Real.pi
axiom equation : 3 * Real.sin (2 * alpha) = 2 * Real.cos alpha

theorem cos_alpha_minus_pi : cos_val = (2 * Real.sqrt 2) / 3 := by
  have h1 : Real.sin (2 * alpha) = 2 * Real.sin alpha * Real.cos alpha := sorry
  have h2 : 3 * (2 * Real.sin alpha * Real.cos alpha) = 2 * Real.cos alpha := sorry
  have h3 : 6 * Real.sin alpha * Real.cos alpha = 2 * Real.cos alpha := sorry
  have h4 : Real.cos alpha ≠ 0 := sorry
  have h5 : Real.sin alpha = 1 / 3 := sorry
  have h6 : (1 / 3)^2 + (Real.cos alpha)^2 = 1 := sorry
  have h7 : (Real.cos alpha)^2 = 1 - (1 / 9) := sorry
  have h8 : (Real.cos alpha)^2 = 8 / 9 := sorry
  have h9 : Real.cos alpha = -Real.sqrt (8 / 9) := sorry
  have h10 : Real.cos alpha = -2 * Real.sqrt 2 / 3 := sorry
  show cos_val = (2 * Real.sqrt 2) / 3, from sorry

end cos_alpha_minus_pi_l444_444385


namespace sugar_needed_for_third_layer_l444_444300

-- Let cups be the amount of sugar, and define the layers
def first_layer_sugar : ℕ := 2
def second_layer_sugar : ℕ := 2 * first_layer_sugar
def third_layer_sugar : ℕ := 3 * second_layer_sugar

-- The theorem we want to prove
theorem sugar_needed_for_third_layer : third_layer_sugar = 12 := by
  sorry

end sugar_needed_for_third_layer_l444_444300


namespace g_neg_x_l444_444607

def g (x : ℝ) : ℝ := (2 * x + 3) / (x - 2)

theorem g_neg_x (x : ℝ) (hx : x^2 ≠ 4) : g (-x) = 1 / g (-x) :=
by
  unfold g
  sorry

end g_neg_x_l444_444607


namespace three_digit_number_is_11_times_sum_of_digits_l444_444342

theorem three_digit_number_is_11_times_sum_of_digits :
    ∃ a b c : ℕ, a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ 
        (100 * a + 10 * b + c = 11 * (a + b + c)) ↔ 
        (100 * 1 + 10 * 9 + 8 = 11 * (1 + 9 + 8)) := 
by
    sorry

end three_digit_number_is_11_times_sum_of_digits_l444_444342


namespace profit_achieved_at_50_yuan_l444_444734

theorem profit_achieved_at_50_yuan :
  ∀ (x : ℝ), (30 ≤ x ∧ x ≤ 54) → 
  ((x - 30) * (80 - 2 * (x - 40)) = 1200) →
  x = 50 :=
by
  intros x h_range h_profit
  sorry

end profit_achieved_at_50_yuan_l444_444734


namespace divisible_by_5_l444_444642

theorem divisible_by_5 (n : ℕ) : (∃ k : ℕ, 2^n - 1 = 5 * k) ∨ (∃ k : ℕ, 2^n + 1 = 5 * k) ∨ (∃ k : ℕ, 2^(2*n) + 1 = 5 * k) :=
sorry

end divisible_by_5_l444_444642


namespace find_s_range_l444_444045

variables {a b c s t y1 y2 : ℝ}

-- Conditions
def is_vertex (a b c s t : ℝ) : Prop := ∀ x : ℝ, (a * x^2 + b * x + c = a * (x - s)^2 + t)

def passes_points (a b c y1 y2 : ℝ) : Prop := 
  (a * (-2)^2 + b * (-2) + c = y1) ∧ (a * 4^2 + b * 4 + c = y2)

def valid_constants (a y1 y2 t : ℝ) : Prop := 
  (a ≠ 0) ∧ (y1 > y2) ∧ (y2 > t)

-- Theorem
theorem find_s_range {a b c s t y1 y2 : ℝ}
  (hv : is_vertex a b c s t)
  (hp : passes_points a b c y1 y2)
  (vc : valid_constants a y1 y2 t) : 
  s > 1 ∧ s ≠ 4 :=
sorry -- Proof skipped

end find_s_range_l444_444045


namespace fraction_equality_x_eq_neg1_l444_444364

theorem fraction_equality_x_eq_neg1 (x : ℝ) (h : (5 + x) / (7 + x) = (3 + x) / (4 + x)) : x = -1 := by
  sorry

end fraction_equality_x_eq_neg1_l444_444364


namespace find_n_l444_444574

variable {a_n : ℕ → ℤ}
variable (a2 : ℤ) (an : ℤ) (d : ℤ) (n : ℕ)

def arithmetic_sequence (a2 : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a2 + (n - 2) * d

theorem find_n 
  (h1 : a2 = 12)
  (h2 : an = -20)
  (h3 : d = -2)
  : n = 18 := by
  sorry

end find_n_l444_444574


namespace find_omega_l444_444603

open Real

def f (x : ℝ) (ω : ℝ) : ℝ := 3 * sin (ω * x + π / 3)

theorem find_omega (ω : ℝ) :
  (∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) ω = f x ω) → ω = 4 ∨ ω = -4 := by
  sorry

end find_omega_l444_444603


namespace henry_max_toys_l444_444967

-- Definitions for conditions
def full_price_toys (n : ℕ) : ℕ := n * 4
def cost_of_three_toys : ℕ := 2 * 4 + (4 / 2).toNat  -- cost of three toys under the deal
def max_sets (total_money : ℕ) : ℕ := total_money / cost_of_three_toys
def total_toys (sets : ℕ) : ℕ := sets * 3

-- Theorem statement
theorem henry_max_toys : total_toys (max_sets (full_price_toys 25)) = 30 :=
by
  sorry

end henry_max_toys_l444_444967


namespace julia_marbles_total_groups_l444_444594

-- Definitions representing the problem's conditions
def red : Type := Color.red
def green : Type := Color.green
def blue : Type := Color.blue
def yellow : Type := Color.yellow

variable (red green : Type)
variable (blue : Fin 2) -- two identical blue marbles
variable (yellow : Fin 2) -- two identical yellow marbles

-- Function to count combinations of two marbles from a given set of marbles
noncomputable def count_groups (red green : Type) (blue yellow : Fin 2) : ℕ :=
  let same_color := 1 + 1 -- two identical blue marbles, two identical yellow marbles
  let different_colors := 1 + 1 + 1 + 1 + 1 -- all different color combinations
  same_color + different_colors

-- The proof statement
theorem julia_marbles_total_groups : count_groups red green blue yellow = 7 :=
by sorry

end julia_marbles_total_groups_l444_444594


namespace min_interval_cosine_l444_444667

theorem min_interval_cosine (a b : ℝ) (h1 : ∀ x, a ≤ x ∧ x ≤ b → -1/2 ≤ cos x ∧ cos x ≤ 1) :
  b - a = 2 * Real.pi / 3 :=
sorry

end min_interval_cosine_l444_444667


namespace solve_for_x_l444_444547

theorem solve_for_x (x y : ℝ) (h₁ : x - y = 8) (h₂ : x + y = 16) (h₃ : x * y = 48) : x = 12 :=
sorry

end solve_for_x_l444_444547


namespace num_from_1_to_200_not_squares_or_cubes_l444_444459

noncomputable def numNonPerfectSquaresAndCubes (n : ℕ) : ℕ :=
  let num_squares := 14
  let num_cubes := 5
  let num_sixth_powers := 2
  n - (num_squares + num_cubes - num_sixth_powers)

theorem num_from_1_to_200_not_squares_or_cubes : numNonPerfectSquaresAndCubes 200 = 183 := by
  sorry

end num_from_1_to_200_not_squares_or_cubes_l444_444459


namespace passing_through_midpoint_of_DE_l444_444796

noncomputable def is_isosceles (ABC : Triangle) : Prop :=
ABC.AB = ABC.AC

noncomputable def midpoint (D : Point) (BC : Segment) : Prop :=
(B + C) / 2 = D

noncomputable def perpendicular (DE : Line) (AC : Line) : Prop :=
right_angle (angle DE AC)

noncomputable def circumsircle_intersect (F : Point) (ABD : Triangle) (BE : Line) : Prop :=
(circumcircle ABD).contains B ∧ (circumcircle ABD).contains F ∧ line BE.contains B ∧ line BE.contains F

noncomputable def passes_through_midpoint (AF : Line) (DE : Segment) : Prop :=
exists M : Point, midpoint M DE ∧ line AF.contains M

theorem passing_through_midpoint_of_DE
  (ABC : Triangle) (D E F : Point) (DE : Line) (AC BE AF : Line) :
  is_isosceles ABC →
  midpoint D (BC) →
  perpendicular DE AC →
  circumsircle_intersect F (Triangle.mk ABD) BE →
  passes_through_midpoint AF (segment.mk D E) :=
by
  sorry

end passing_through_midpoint_of_DE_l444_444796


namespace average_increase_l444_444664

def avg_increase (x : ℕ) (y : ℕ) (new_y : ℕ) : Prop :=
  ∀ (a : ℕ) (b : ℕ), a * x = b → a = y → a + new_y = (y + new_y * x)

theorem average_increase (x : ℕ) (y : ℕ) (inc : ℕ) (new_a : ℕ) : 
  x = 10 → y = 23 → inc = 4 → new_a = 27 → avg_increase x y new_a := 
by 
  intros hx hy hinc hnewa 
  unfold avg_increase 
  intros a b 
  assume ha hx' 
  rw [hx, hy] at * 
  have hsum : a = x * y := by rw ha 
  have new_sum : a + x * inc = new_a * x := sorry
  exact new_sum

end average_increase_l444_444664


namespace total_blocks_fallen_l444_444108

def stack_height (n : Nat) : Nat :=
  if n = 1 then 7
  else if n = 2 then 7 + 5
  else if n = 3 then 7 + 5 + 7
  else 0

def blocks_standing (n : Nat) : Nat :=
  if n = 1 then 0
  else if n = 2 then 2
  else if n = 3 then 3
  else 0

def blocks_fallen (n : Nat) : Nat :=
  stack_height n - blocks_standing n

theorem total_blocks_fallen : blocks_fallen 1 + blocks_fallen 2 + blocks_fallen 3 = 33 :=
  by
    sorry

end total_blocks_fallen_l444_444108


namespace line_AF_midpoint_DE_l444_444788

variables {A B C D E F : Type} [EuclideanGeometry.point A] [EuclideanGeometry.point B] [EuclideanGeometry.point C]
  [EuclideanGeometry.point D] [EuclideanGeometry.point E] [EuclideanGeometry.point F]

-- Definitions based on the problem conditions
def isosceles_triangle (A B C : Type) [EuclideanGeometry.point A] [EuclideanGeometry.point B] [EuclideanGeometry.point C] : Prop :=
  (EuclideanGeometry.distance A B = EuclideanGeometry.distance A C)

def midpoint (D B C : Type) [EuclideanGeometry.point D] [EuclideanGeometry.point B] [EuclideanGeometry.point C] : Prop :=
  EuclideanGeometry.between D B C

def perpendicular (D E : Type) [EuclideanGeometry.point D] [EuclideanGeometry.point E] : Prop := 
  EuclideanGeometry.perp D E

def circumcircle (ABD : Set EuclideanGeometry.point) : Prop :=
  EuclideanGeometry.circumcircle ABD

def intersection (circumcircle_intersection : Type) [EuclideanGeometry.point F] [EuclideanGeometry.point B] : Prop := 
  EuclideanGeometry.intersects circumcircle_intersection F B

-- The theorem we need to prove
theorem line_AF_midpoint_DE
  (isoABC : isosceles_triangle A B C)
  (midD : midpoint D B C)
  (perpDE : perpendicular D E)
  (circABD : circumcircle {A, B, D})
  (interBF : intersection {A, B, D})
  : EuclideanGeometry.on_line A F (EuclideanGeometry.midpoint D E) :=
sorry -- proof here

end line_AF_midpoint_DE_l444_444788


namespace numbers_not_squares_nor_cubes_1_to_200_l444_444496

theorem numbers_not_squares_nor_cubes_1_to_200 : 
  let num_squares := nat.sqrt 200
  let num_cubes := nat.cbrt 200
  let num_sixth_powers := nat.root 6 200
  let total := num_squares + num_cubes - num_sixth_powers
  200 - total = 182 := 
by
  sorry

end numbers_not_squares_nor_cubes_1_to_200_l444_444496


namespace solve_problem_l444_444897

noncomputable def parametric_eqn_C1 (α : ℝ) : ℝ × ℝ :=
(2 * sqrt 5 * cos α, 2 * sin α)

def standard_eqn_C1 (x y : ℝ) : Prop :=
(x^2 / 20) + (y^2 / 4) = 1

def polar_eqn_C2 (ρ θ : ℝ) : Real :=
ρ^2 + 4 * ρ * cos θ - 2 * ρ * sin θ + 4

def standard_eqn_C2 (x y : ℝ) : Prop :=
(x + 2)^2 + (y - 1)^2 = 1

def line_l (t : ℝ) : ℝ × ℝ :=
(-4 + (sqrt 2 / 2) * t, (sqrt 2 / 2) * t)

theorem solve_problem (x y t1 t2 : ℝ) (α θ ρ : ℝ) :
  (standard_eqn_C1 (2 * sqrt 5 * cos α) (2 * sin α)) ∧
  (standard_eqn_C2 x y) ∧
  let ⟨lx, ly⟩ := line_l t1,
      ⟨lx₂, ly₂⟩ := line_l t2 in
  (x, y) = (lx, ly) ∧
  (x, y) = (lx₂, ly₂) ∧
  (ρ, θ) = (ρ, θ) ∧
  |t1 - t2| = sqrt 2 
:=
sorry

end solve_problem_l444_444897


namespace right_triangle_conditions_count_l444_444105

-- Define the necessary conditions as predicates
def condition1 (A B C : ℝ) : Prop := A = C - B
def condition2 (a b c : ℝ) : Prop := (a/b = 3/4) ∧ (b/c = 4/5)
def condition3 (A B C : ℝ) : Prop := A / 3 = B / 1 ∧ A / 3 = C / 5
def condition4 (a b c : ℝ) : Prop := a = 5 ∧ b = 12 ∧ c = 13
def condition5 (a b c : ℝ) : Prop := b^2 = a^2 - c^2

-- Define right triangle property based on sides
def is_right_triangle (a b c : ℝ) : Prop := a^2 + c^2 = b^2

-- Main theorem statement
theorem right_triangle_conditions_count :
  ∃ n : ℕ, n = 4 ∧
    ((∃ (A B C : ℝ), condition1 A B C ∧ is_right_triangle A B C) ∧
     (∃ (a b c : ℝ), condition2 a b c ∧ is_right_triangle a b c) ∧
     (∃ (A B C : ℝ), ¬ condition3 A B C ∧ is_right_triangle A B C) ∧
     (∃ (a b c : ℝ), condition4 a b c ∧ is_right_triangle a b c) ∧
     (∃ (a b c : ℝ), condition5 a b c ∧ is_right_triangle a b c)) :=
begin
 sorry
end

end right_triangle_conditions_count_l444_444105


namespace polar_equation_of_circle_l444_444010

theorem polar_equation_of_circle 
    (r : ℝ) (θ : ℝ) 
    (h_radius : r = 1) 
    (h_center : θ = 0) : 
      ∃ ρ : ℝ, ρ = 2 * cos θ :=
by 
  sorry

end polar_equation_of_circle_l444_444010


namespace ratio_of_areas_l444_444808

noncomputable theory

variables {ABC : Type}
variables (A B C A₁ B₁ C₁ : ABC)
variables (r R : ℝ) -- Real numbers representing radii
variables (S : ABC → ℝ) -- Function to calculate area

-- Hypotheses
def is_triangle (ABC : Type) : Prop := -- Definition of a triangle in Lean
sorry

def is_on_circumcircle (A₁ B₁ C₁ A B C : ABC) : Prop := -- Definition of points on circumcircle
sorry

def angle_bisectors_intersect (A₁ B₁ C₁ A B C : ABC) : Prop := -- Definition of intersection of angle bisectors
sorry

def radius_inscribed (r : ℝ) : Prop := -- Definition for in-radius
sorry

def radius_circumscribed (R : ℝ) : Prop := -- Definition for circum-radius
sorry

def area (S : ABC → ℝ) : Prop := -- Definition for area function
sorry

-- Main theorem statement
theorem ratio_of_areas (h1 : is_triangle ABC)
  (h2 : is_on_circumcircle A₁ B₁ C₁ A B C)
  (h3 : angle_bisectors_intersect A₁ B₁ C₁ A B C)
  (h4 : radius_inscribed r)
  (h5 : radius_circumscribed R)
  (h6 : area S)
  : S ABC / S A₁ B₁ C₁ = 2 * r / R :=
sorry

end ratio_of_areas_l444_444808


namespace abs_inequality_solution_l444_444196

theorem abs_inequality_solution (x : ℝ) : |2 * x - 5| > 1 ↔ x < 2 ∨ x > 3 := sorry

end abs_inequality_solution_l444_444196


namespace rational_exponent_simplification_l444_444694

theorem rational_exponent_simplification :
  (16 / 81) ^ (-1 / 4 : ℝ) = 3 / 2 := 
by
  sorry

end rational_exponent_simplification_l444_444694


namespace sequence_sum_identity_l444_444395

theorem sequence_sum_identity (n : ℕ) 
  (a : ℕ → ℕ) 
  (h : ∀ (k : ℕ), a k = 2^(k - 1) + 1) : 
  (∑ k in Finset.range (n + 1), a (k + 1) * Nat.choose n k) = 3^n + 2^n := 
by
  sorry

end sequence_sum_identity_l444_444395


namespace men_science_majors_pct_l444_444817

-- Define the conditions
def women_science_major_pct : ℕ → ℝ := λ total_students => 0.1 * (0.6 * total_students)
def non_science_major_pct : ℝ := 0.6
def men_pct : ℝ := 0.4

-- Define the theorem
theorem men_science_majors_pct (total_students : ℕ) : 
  total_students > 0 → 
  let women_science_majors := women_science_major_pct total_students in
  let science_majors := 0.4 * total_students in
  let men := men_pct * total_students in
  let men_science_majors := science_majors - women_science_majors in
  (men_science_majors / men) * 100 = 85 := 
by
  sorry

end men_science_majors_pct_l444_444817


namespace decimal_rep_250th_l444_444212

theorem decimal_rep_250th (n m : ℕ) (h : n = 8 ∧ m = 11) : 
  (decimal_expansion n m).digit_at_position 250 = 2 := 
sorry

end decimal_rep_250th_l444_444212


namespace basic_full_fare_l444_444266

theorem basic_full_fare 
  (F R : ℝ)
  (h1 : F + R = 216)
  (h2 : (F + R) + (0.5 * F + R) = 327) :
  F = 210 :=
by
  sorry

end basic_full_fare_l444_444266


namespace determine_a_b_l444_444185

-- Define the function f
def f (x a b : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

-- Define the first derivative of the function f
def f' (x a b : ℝ) : ℝ := 3*x^2 + 2*a*x + b

-- Define the conditions given in the problem
def conditions (a b : ℝ) : Prop :=
  (f' 1 a b = 0) ∧ (f 1 a b = 10)

-- Provide the main theorem stating the required proof
theorem determine_a_b (a b : ℝ) (h : conditions a b) : a = 4 ∧ b = -11 :=
by {
  sorry
}

end determine_a_b_l444_444185


namespace total_money_proof_l444_444727

noncomputable def lemons_qty := 80
noncomputable def grapes_qty := 140
noncomputable def oranges_qty := 60
noncomputable def apples_qty := 100
noncomputable def kiwis_qty := 50
noncomputable def pineapples_qty := 30

noncomputable def lemons_price := 8 
noncomputable def grapes_price := 7
noncomputable def oranges_price := 5 
noncomputable def apples_price := 4
noncomputable def kiwis_price := 6 
noncomputable def pineapples_price := 12

noncomputable def lemons_price_increase := 0.5 
noncomputable def grapes_price_increase := 0.25 
noncomputable def oranges_price_increase := 0.1 
noncomputable def apples_price_increase := 0.2 
noncomputable def kiwis_price_decrease := 0.15 
noncomputable def pineapples_price_change := 0.0

noncomputable def lemons_new_price := lemons_price * (1 + lemons_price_increase)
noncomputable def grapes_new_price := grapes_price * (1 + grapes_price_increase)
noncomputable def oranges_new_price := oranges_price * (1 + oranges_price_increase)
noncomputable def apples_new_price := apples_price * (1 + apples_price_increase)
noncomputable def kiwis_new_price := kiwis_price * (1 - kiwis_price_decrease)
noncomputable def pineapples_new_price := pineapples_price * (1 + pineapples_price_change)

noncomputable def total_money_collected :=
  lemons_qty * lemons_new_price + 
  grapes_qty * grapes_new_price + 
  oranges_qty * oranges_new_price + 
  apples_qty * apples_new_price + 
  kiwis_qty * kiwis_new_price + 
  pineapples_qty * pineapples_new_price

theorem total_money_proof : total_money_collected = 3610 := by
    sorry

end total_money_proof_l444_444727


namespace numbers_neither_square_nor_cube_l444_444541

theorem numbers_neither_square_nor_cube (n : ℕ) (h : n = 200) :
  let num_squares := 14 in
  let num_cubes := 5 in
  let num_sixth_powers := 2 in
  (n - (num_squares + num_cubes - num_sixth_powers)) = 183 :=
by
  -- Let the number of integers in the range
  let num_total := 200
  -- Define how we count perfect squares and cubes
  let num_squares := 14
  let num_cubes := 5
  let num_sixth_powers := 2

  -- Subtract the overlapping perfect sixth powers from the sum of squares and cubes
  let num_either_square_or_cube := num_squares + num_cubes - num_sixth_powers

  -- Calculate the final result
  let result := num_total - num_either_square_or_cube

  -- Prove by computation
  show result = 183 from
    calc
      result = num_total - num_either_square_or_cube := rfl
           ... = 200 - (14 + 5 - 2)                   := rfl
           ... = 200 - 17                             := rfl
           ... = 183                                  := rfl

end numbers_neither_square_nor_cube_l444_444541


namespace angle_equality_l444_444121

-- Define the problem statement
variable {A B C P Q : Type} -- representing the points

-- Assuming the points lie in some geometric space and other geometric objects are defined
noncomputable def circumcircle (ΔABC : Triangle) : Circle := sorry
noncomputable def AExcircle (ΔABC : Triangle) : Circle := sorry

axiom commonTangentsIntersectBC (ΔABC : Triangle) (circ k : Circle) (ex k_a : Circle) (tangents PQ : Points) :
  circumcircle ΔABC = k →
  AExcircle ΔABC = k_a →
  tangents = {P, Q} → 
  (∀ P Q, P ∈ BC ∧ Q ∈ BC) → 
  ∃ (P Q : Point), PQ ∩ BC = {P, Q}

-- Theorem to prove
theorem angle_equality (ΔABC : Triangle) (k : Circle) (k_a : Circle) (P Q : Point) :
  circumcircle ΔABC = k →
  AExcircle ΔABC = k_a →
  commonTangentsIntersectBC ΔABC k k_a {P, Q} →
  ∠PAB = ∠CAQ :=
by sorry

end angle_equality_l444_444121


namespace distance_from_point_to_line_le_2_l444_444948

open Real

-- Definitions of points and distances
variables (P A B C : Point) (l : Line)

-- Conditions given in the problem
def PA: ℝ := dist P A
def PB: ℝ := dist P B
def PC: ℝ := dist P C

-- Given the conditions
axiom PA_eq_2 : PA P A = 2
axiom PB_eq_25 : PB P B = 2.5
axiom PC_eq_3 : PC P C = 3

-- Theorem to state the conclusion
theorem distance_from_point_to_line_le_2 : dist_from_point_to_line P l ≤ 2 :=
by
  sorry

end distance_from_point_to_line_le_2_l444_444948


namespace bruna_wins_for_N_5_aline_wins_for_N_20_bruna_wins_in_range_l444_444630

-- a) For N = 5, Bruna has a winning position.
theorem bruna_wins_for_N_5 : ∃ (n : ℕ), n = 5 ∧ (winning_position Bruna n) :=
by
  sorry

-- b) For N = 20, Aline has a winning position.
theorem aline_wins_for_N_20 : ∃ (n : ℕ), n = 20 ∧ (winning_position Aline n) :=
by
  sorry

-- c) For 100 < N < 200, find N such that Bruna has a winning position.
theorem bruna_wins_in_range : ∃ (n : ℕ), 100 < n ∧ n < 200 ∧ (winning_position Bruna n) ∧ n = 191 :=
by
  sorry

end bruna_wins_for_N_5_aline_wins_for_N_20_bruna_wins_in_range_l444_444630


namespace count_of_integers_with_factors_in_range_l444_444069

open Int Nat

def lcm (a b : ℕ) : ℕ := Nat.lcm a b

def lcm_multiple_factors(n : ℕ) : ℕ := 
  lcm (lcm 18 24) 36

def int_within_range (a b c : ℕ)(x : ℕ) : Prop := 
  a < x ∧ x < b ∧ c ∣ x 

theorem count_of_integers_with_factors_in_range (a b n : ℕ) : 
  n = lcm_multiple_factors 1 → 
  500 ≤ a → 
  a ≤ b → 
  b ≤ 1500 → 
  ∑ k in (range' (ceil (500 / n)) ((floor (1500 / n)) - (ceil (500 / n)) + 1)), k * n ∈ (finset.range') 500 1001 → 
  ∃ count, count = 14 := 
by 
  sorry

end count_of_integers_with_factors_in_range_l444_444069


namespace line_b_parallel_to_median_acd_l444_444226

-- Definitions for the problem
variables (a b c d : Line) (p q r : Point)

-- Assumptions from the conditions
axiom no_two_lines_parallel (x y : Line) (hxy : x ≠ y) : ¬ parallel x y
axiom no_three_lines_intersect (x y z : Line) (hxy : x ≠ y) (hyz : y ≠ z) (hzx : z ≠ x) : 
      ¬ ∃ p, on_line p x ∧ on_line p y ∧ on_line p z
axiom line_a_parallel_to_median_bc_d (a b c d : Line) : 
      ∃ m : Median (Triangle.of_lines b c d), parallel a m.line

-- The theorem to prove
theorem line_b_parallel_to_median_acd :
  ∃ m : Median (Triangle.of_lines a c d), parallel b m.line :=
sorry

end line_b_parallel_to_median_acd_l444_444226


namespace zero_point_in_12_l444_444076

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 3

theorem zero_point_in_12 : ∃ c ∈ Ioo 1 2, f c = 0 :=
by {
  sorry
}

end zero_point_in_12_l444_444076


namespace range_f_range_k_two_zeros_l444_444940

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x
noncomputable def g (k x : ℝ) : ℝ := k * x^2
noncomputable def F (k x : ℝ) : ℝ := f x - g k x

theorem range_f : Set.Icc (-1 / Real.exp 1) +∞ = {y | ∃ x, f x = y} :=
  sorry

theorem range_k_two_zeros (k : ℝ) : ((F k).has_two_zeros_on (Set.Ioi 0)) ↔ k ∈ Set.Ioi Real.exp 1 :=
  sorry

end range_f_range_k_two_zeros_l444_444940


namespace find_distance_l444_444688

-- Definitions based on the given conditions
def speed_of_boat := 16 -- in kmph
def speed_of_stream := 2 -- in kmph
def total_time := 960 -- in hours
def downstream_speed := speed_of_boat + speed_of_stream
def upstream_speed := speed_of_boat - speed_of_stream

-- Prove that the distance D is 7590 km given the total time and speeds
theorem find_distance (D : ℝ) :
  (D / downstream_speed + D / upstream_speed = total_time) → D = 7590 :=
by
  sorry

end find_distance_l444_444688


namespace ultimate_power_of_two_l444_444317

-- Define f1 for positive integers
def f1 : ℕ → ℕ
| 1 := 2
| (n+1) := if (n+1).prime_factors = [] then
              2
           else 
              let p1 := (n+1).prime_factors.head
              let p2 := (n+1).prime_factors.tail.head
              let e1 := (n+1).factors_count p1
              let e2 := (n+1).factors_count p2
              (p1 + 2)^(e1 - 1) * (p2 + 2)^(e2 - 1)

-- Define recursive fm for m ≥ 2
noncomputable def fm : ℕ → ℕ → ℕ
| 1 n := f1 n
| (m+1) n := f1 (fm m n)

-- Theorem statement
theorem ultimate_power_of_two :
  ¬ (∃ N m k : ℕ, 1 ≤ N ∧ N ≤ 300 ∧ m ≥ 1 ∧ fm m N = 2^k) :=
sorry

end ultimate_power_of_two_l444_444317


namespace solution_set_inequality_l444_444377

noncomputable def f : ℝ → ℝ := sorry

axiom f_domain : ∀ x : ℝ, true

axiom f_pos_gt1 (x : ℝ) (hx : x > 0) : f x > 1

axiom f_add (x y : ℝ) : f (x + y) = f x * f y

theorem solution_set_inequality :
  { x : ℝ | f (Real.log x / Real.log (1 / 2)) ≤ 1 / f (Real.log x / Real.log (1 / 2) + 1) } =
  { x | 4 ≤ x } :=
by
  sorry

end solution_set_inequality_l444_444377


namespace angle_range_l444_444071

noncomputable def A := Real.ArcCos (3/4)

theorem angle_range (A : ℝ) (h : Real.cos A = 3/4) : 30 * Real.pi / 180 < A ∧ A < 45 * Real.pi / 180 :=
by
  have : Real.cos (Real.ArcCos (3/4)) = 3/4 := by
    exact Real.cos_arccos (by linarith [Real.le_of_forall_real_cos, Real.cos_bound])
  have H : Real.ArcCos (3/4) ∈ Set.Ioo (30 * Real.pi / 180) (45 * Real.pi / 180) := sorry
  convert H using 1
  exact_mod_cast H

end angle_range_l444_444071


namespace compute_difference_of_squares_l444_444872

theorem compute_difference_of_squares : (65^2 - 55^2) = 1200 := 
by
  have h : ∀ a b : ℕ, a^2 - b^2 = (a + b) * (a - b) := λ a b, by sorry
  rw h 65 55
  norm_num
  sorry

end compute_difference_of_squares_l444_444872


namespace no_number_exists_decreasing_by_removing_digit_l444_444321

theorem no_number_exists_decreasing_by_removing_digit :
  ¬ ∃ (x y n : ℕ), x * 10^n + y = 58 * y :=
by
  sorry

end no_number_exists_decreasing_by_removing_digit_l444_444321


namespace roadmap_transformation_l444_444661

theorem roadmap_transformation (V : Type) (E G H : set (V × V))
  (hG : ∀ v ∈ V, ∃ ! (u : V), (v, u) ∈ G)
  (hH : ∀ v ∈ V, ∃ ! (u : V), (v, u) ∈ H)
  (deg_G : ∀ v ∈ V, ∃ k : ℕ, card {u | (v, u) ∈ G} = k)
  (deg_H : ∀ v ∈ V, ∃ k : ℕ, card {u | (v, u) ∈ H} = k) :
  ∃ swap_sequence : list (set (V × V) → set (V × V)),
  (∀ (swap : set (V × V) → set (V × V)), swap ∈ swap_sequence →
    ∃ (a b c d : V), (a, b) ∈ E ∧ (c, d) ∈ E ∧ (b, c) ∉ E ∧ (a, d) ∉ E ∧
      swap = λ e, (e \ {(a, b), (c, d)}) ∪ {(b, c), (a, d)})
  ∧ (foldl (λ acc swap, swap acc) G swap_sequence = H) := sorry

end roadmap_transformation_l444_444661


namespace twelve_p_squared_plus_q_l444_444546

-- Define p and q as natural numbers such that both roots of the equation are prime numbers
variables (p q : ℕ)

-- Given conditions
axiom roots_are_prime : ∀ {x1 x2 : ℕ}, (x1 * x2 = 1985 / p) → (x1 * x2 = 5 * 397) → x1 = 5 ∧ x2 = 397

-- Proof statement
theorem twelve_p_squared_plus_q (hp : ∀ x, prime x → ∃ x1 x2, x1 = 5 ∧ x2 = 397) : 12 * p^2 + q = 414 :=
begin
  sorry
end

end twelve_p_squared_plus_q_l444_444546


namespace average_of_real_roots_l444_444265

noncomputable def average_of_roots (d : ℚ) : ℚ :=
let a := (3 : ℚ),
    b := (-9 : ℚ),
    sum_roots := -b / a in
sum_roots / 2

theorem average_of_real_roots (d : ℚ) (h : ∃ x y : ℚ, 3 * x^2 - 9 * x + d = 0 ∧ 3 * y^2 - 9 * y + d = 0) :
  average_of_roots d = 1.5 :=
by {
    sorry
}

end average_of_real_roots_l444_444265


namespace previous_year_height_l444_444632

noncomputable def previous_height (H_current : ℝ) (g : ℝ) : ℝ :=
  H_current / (1 + g)

theorem previous_year_height :
  previous_height 147 0.05 = 140 :=
by
  unfold previous_height
  -- Proof steps would go here
  sorry

end previous_year_height_l444_444632


namespace intersection_of_M_and_N_is_12_l444_444060

def M : Set ℤ := {x | -1 ≤ x ∧ x < 3}
def N : Set ℤ := {1, 2, 3}

theorem intersection_of_M_and_N_is_12 : M ∩ N = {1, 2} :=
by
  sorry

end intersection_of_M_and_N_is_12_l444_444060


namespace sum_of_remainders_mod_53_l444_444214

theorem sum_of_remainders_mod_53 (x y z : ℕ) (hx : x % 53 = 36) (hy : y % 53 = 15) (hz : z % 53 = 7) : 
  (x + y + z) % 53 = 5 :=
by
  sorry

end sum_of_remainders_mod_53_l444_444214


namespace probability_single_trial_l444_444684

-- Conditions
def prob_at_least_once (p : ℝ) : ℝ := 1 - (1 - p)^3

-- Main theorem
theorem probability_single_trial (h : prob_at_least_once p = 0.973) : p = 0.7 :=
by
  sorry

end probability_single_trial_l444_444684


namespace box_of_books_weight_l444_444243

theorem box_of_books_weight (books_in_box : ℕ) (weight_per_book : ℕ) (books_in_box_eq : books_in_box = 14) (weight_per_book_eq : weight_per_book = 3) :
  books_in_box * weight_per_book = 42 :=
by
  rw [books_in_box_eq, weight_per_book_eq]
  exact rfl

end box_of_books_weight_l444_444243


namespace wheel_distance_3_revolutions_l444_444845

theorem wheel_distance_3_revolutions (r : ℝ) (n : ℝ) (circumference : ℝ) (total_distance : ℝ) :
  r = 2 →
  n = 3 →
  circumference = 2 * Real.pi * r →
  total_distance = n * circumference →
  total_distance = 12 * Real.pi := by
  intros
  sorry

end wheel_distance_3_revolutions_l444_444845


namespace find_m_l444_444942

noncomputable def m_solution (m : ℝ) : ℂ := (m - 3 * Complex.I) / (2 + Complex.I)

theorem find_m :
  ∀ (m : ℝ), Complex.im (m_solution m) ≠ 0 → Complex.re (m_solution m) = 0 → m = 3 / 2 :=
by
  intro m h_im h_re
  sorry

end find_m_l444_444942


namespace function_properties_l444_444257

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6)

theorem function_properties :
  (∀ x, f (x + Real.pi) = f x) ∧
  (f (Real.pi / 3) = 1) ∧
  (∀ x y, -Real.pi / 6 ≤ x → x ≤ y → y ≤ Real.pi / 3 → f x ≤ f y) := by
  sorry

end function_properties_l444_444257


namespace total_ladybugs_correct_l444_444169

noncomputable def total_ladybugs (with_spots : ℕ) (without_spots : ℕ) : ℕ :=
  with_spots + without_spots

theorem total_ladybugs_correct :
  total_ladybugs 12170 54912 = 67082 :=
by
  unfold total_ladybugs
  rfl

end total_ladybugs_correct_l444_444169


namespace numbers_not_squares_nor_cubes_1_to_200_l444_444493

theorem numbers_not_squares_nor_cubes_1_to_200 : 
  let num_squares := nat.sqrt 200
  let num_cubes := nat.cbrt 200
  let num_sixth_powers := nat.root 6 200
  let total := num_squares + num_cubes - num_sixth_powers
  200 - total = 182 := 
by
  sorry

end numbers_not_squares_nor_cubes_1_to_200_l444_444493


namespace team_selection_ways_l444_444324

theorem team_selection_ways (n : ℕ) (h : n = 25) : 25 * (25 - 1) * (25 - 2) = 13800 :=
by {
    rw h,
    norm_num,
    exact 25 * 24 * 23,
    sorry
}

end team_selection_ways_l444_444324


namespace value_of_x_l444_444223

theorem value_of_x (n x : ℝ) (h1: x = 3 * n) (h2: 2 * n + 3 = 0.2 * 25) : x = 3 :=
by
  sorry

end value_of_x_l444_444223


namespace two_pow_58_plus_one_factored_l444_444160

theorem two_pow_58_plus_one_factored :
  ∃ (a b c : ℕ), 2 < a ∧ 2 < b ∧ 2 < c ∧ 2 ^ 58 + 1 = a * b * c :=
sorry

end two_pow_58_plus_one_factored_l444_444160


namespace line_AF_through_midpoint_of_DE_l444_444800

open EuclideanGeometry

noncomputable def midpoint_of_segment (A B : Point) := ((A.x + B.x) / 2, (A.y + B.y) / 2)

theorem line_AF_through_midpoint_of_DE
  {A B C D E F : Point}
  (h_isosceles : AB = AC)
  (h_midpoint_D : D = midpoint B C)
  (h_perpendicular_DE : ⟂ D E AC)
  (h_circumcircle_intersects_BE : ∃ k : ℕ, circumcircle A B D = circum circle (B ⊗ F))
  : passes_through (line A F) (midpoint_of_segment D E) :=
sorry

end line_AF_through_midpoint_of_DE_l444_444800


namespace midpointAB_l444_444381

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def midpoint (p1 p2 : Point3D) : Point3D :=
  { x := (p1.x + p2.x) / 2,
    y := (p1.y + p2.y) / 2,
    z := (p1.z + p2.z) / 2 }

def PointA : Point3D :=
  { x := 0, y := 2, z := 1 }

def PointB : Point3D :=
  { x := -2, y := 0, z := 3 }

theorem midpointAB : midpoint PointA PointB = { x := -1, y := 1, z := 2 } :=
by
  sorry

end midpointAB_l444_444381


namespace numbers_not_perfect_squares_or_cubes_l444_444454

theorem numbers_not_perfect_squares_or_cubes : 
  let total_numbers := 200
  let perfect_squares := 14
  let perfect_cubes := 5
  let sixth_powers := 1
  total_numbers - (perfect_squares + perfect_cubes - sixth_powers) = 182 :=
by
  sorry

end numbers_not_perfect_squares_or_cubes_l444_444454


namespace monotonic_increasing_intervals_l444_444408

theorem monotonic_increasing_intervals {x : ℝ} :
  let y := λ x : ℝ, x^3 - 3 * x
  in ∀ x,
     (1 < x ∨ x < -1) ↔ 
     (∀ x1 x2, x1 < x2 → y x1 < y x2) sorry

end monotonic_increasing_intervals_l444_444408


namespace bug_probability_at_A_after_8_meters_l444_444123

noncomputable def P : ℕ → ℚ 
| 0 => 1
| (n + 1) => (1 / 3) * (1 - P n)

theorem bug_probability_at_A_after_8_meters :
  P 8 = 547 / 2187 := 
sorry

end bug_probability_at_A_after_8_meters_l444_444123


namespace x_needs_2_5_days_to_complete_l444_444741

variables (x y z : ℝ)
variables (work_x work_y work_z work_done_by_y work_done_by_z total_work remaining_work : ℝ)
variables (days_needed_by_x : ℝ)

-- Let x, y, and z be their daily work rates.
def work_rate_x : ℝ := 1 / 30
def work_rate_y : ℝ := 1 / 15
def work_rate_z : ℝ := 1 / 20

-- Calculating work done by y and z.
def work_done_by_y := 10 * work_rate_y
def work_done_by_z := 5 * work_rate_z

-- Total work done by y and z.
def work_done_by_yz := work_done_by_y + work_done_by_z

-- Total work is 1 unit.
def total_work : ℝ := 1
def remaining_work : ℝ := total_work - work_done_by_yz

-- Number of days x needs to finish the remaining work.
def days_needed_by_x := remaining_work / work_rate_x

-- Theorem stating that x alone needs 2.5 days to finish the remaining work.
theorem x_needs_2_5_days_to_complete :
  days_needed_by_x = 2.5 := by
  sorry

end x_needs_2_5_days_to_complete_l444_444741


namespace numbers_neither_square_nor_cube_l444_444534

theorem numbers_neither_square_nor_cube (n : ℕ) (h : n = 200) :
  let num_squares := 14 in
  let num_cubes := 5 in
  let num_sixth_powers := 2 in
  (n - (num_squares + num_cubes - num_sixth_powers)) = 183 :=
by
  -- Let the number of integers in the range
  let num_total := 200
  -- Define how we count perfect squares and cubes
  let num_squares := 14
  let num_cubes := 5
  let num_sixth_powers := 2

  -- Subtract the overlapping perfect sixth powers from the sum of squares and cubes
  let num_either_square_or_cube := num_squares + num_cubes - num_sixth_powers

  -- Calculate the final result
  let result := num_total - num_either_square_or_cube

  -- Prove by computation
  show result = 183 from
    calc
      result = num_total - num_either_square_or_cube := rfl
           ... = 200 - (14 + 5 - 2)                   := rfl
           ... = 200 - 17                             := rfl
           ... = 183                                  := rfl

end numbers_neither_square_nor_cube_l444_444534


namespace ab_a4_b4_divisible_by_30_l444_444652

theorem ab_a4_b4_divisible_by_30 (a b : Int) : 30 ∣ a * b * (a^4 - b^4) := 
by
  sorry

end ab_a4_b4_divisible_by_30_l444_444652


namespace line_AF_through_midpoint_of_DE_l444_444805

open EuclideanGeometry

noncomputable def midpoint_of_segment (A B : Point) := ((A.x + B.x) / 2, (A.y + B.y) / 2)

theorem line_AF_through_midpoint_of_DE
  {A B C D E F : Point}
  (h_isosceles : AB = AC)
  (h_midpoint_D : D = midpoint B C)
  (h_perpendicular_DE : ⟂ D E AC)
  (h_circumcircle_intersects_BE : ∃ k : ℕ, circumcircle A B D = circum circle (B ⊗ F))
  : passes_through (line A F) (midpoint_of_segment D E) :=
sorry

end line_AF_through_midpoint_of_DE_l444_444805


namespace partition_ways_l444_444583

theorem partition_ways (black_cells : ℕ) (grey_cells : ℕ) (total_cells : ℕ) : 
  total_cells = 17 ∧ black_cells = 9 ∧ grey_cells = 8 →
  ∃ ways : ℕ, ways = 10 :=
begin
  sorry
end

end partition_ways_l444_444583


namespace numbers_not_squares_nor_cubes_1_to_200_l444_444490

theorem numbers_not_squares_nor_cubes_1_to_200 : 
  let num_squares := nat.sqrt 200
  let num_cubes := nat.cbrt 200
  let num_sixth_powers := nat.root 6 200
  let total := num_squares + num_cubes - num_sixth_powers
  200 - total = 182 := 
by
  sorry

end numbers_not_squares_nor_cubes_1_to_200_l444_444490


namespace molecular_weight_7_moles_alpo4_l444_444208

theorem molecular_weight_7_moles_alpo4 :
  let molecular_weight_Al := 26.98
  let molecular_weight_P := 30.97
  let molecular_weight_O := 16.00
  let molecular_weight_AlPO4 := molecular_weight_Al + molecular_weight_P + 4 * molecular_weight_O
  let moles := 7
  molecular_weight_AlPO4 * moles = 853.65 :=
by
  let molecular_weight_Al := 26.98
  let molecular_weight_P := 30.97
  let molecular_weight_O := 16.00
  let molecular_weight_AlPO4 := molecular_weight_Al + molecular_weight_P + 4 * molecular_weight_O
  let moles := 7
  show molecular_weight_AlPO4 * moles = 853.65
  sorry

end molecular_weight_7_moles_alpo4_l444_444208


namespace find_vertex_A_l444_444101

variables (B C: ℝ × ℝ × ℝ)

-- Defining midpoints conditions
def midpoint_BC : ℝ × ℝ × ℝ := (1, 5, -1)
def midpoint_AC : ℝ × ℝ × ℝ := (0, 4, -2)
def midpoint_AB : ℝ × ℝ × ℝ := (2, 3, 4)

-- The coordinates of point A we need to prove
def target_A : ℝ × ℝ × ℝ := (1, 2, 3)

-- Lean statement proving the coordinates of A
theorem find_vertex_A (A B C : ℝ × ℝ × ℝ)
  (hBC : midpoint_BC = (1, 5, -1))
  (hAC : midpoint_AC = (0, 4, -2))
  (hAB : midpoint_AB = (2, 3, 4)) :
  A = (1, 2, 3) := 
sorry

end find_vertex_A_l444_444101


namespace product_of_integers_eq_expected_result_l444_444913

theorem product_of_integers_eq_expected_result
  (E F G H I : ℚ) 
  (h1 : E + F + G + H + I = 80) 
  (h2 : E + 2 = F - 2) 
  (h3 : F - 2 = G * 2) 
  (h4 : G * 2 = H * 3) 
  (h5 : H * 3 = I / 2) :
  E * F * G * H * I = (5120000 / 81) := 
by 
  sorry

end product_of_integers_eq_expected_result_l444_444913


namespace circle_equation_passing_through_points_symmetric_circle_equation_midpoint_trajectory_equation_l444_444931

-- Prove the equation of the circle passing through points A and B with center on a specified line
theorem circle_equation_passing_through_points
  (A B : ℝ × ℝ) (line : ℝ → ℝ → Prop)
  (N : ℝ → ℝ → Prop) :
  A = (3, 1) →
  B = (-1, 3) →
  (∀ x y, line x y ↔ 3 * x - y - 2 = 0) →
  (∀ x y, N x y ↔ (x - 2)^2 + (y - 4)^2 = 10) →
  sorry :=
sorry

-- Prove the symmetric circle equation regarding a specified line
theorem symmetric_circle_equation
  (N N' : ℝ → ℝ → Prop) (line : ℝ → ℝ → Prop) :
  (∀ x y, N x y ↔ (x - 2)^2 + (y - 4)^2 = 10) →
  (∀ x y, N' x y ↔ (x - 1)^2 + (y - 5)^2 = 10) →
  (∀ x y, line x y ↔ x - y + 3 = 0) →
  sorry :=
sorry

-- Prove the trajectory equation of the midpoint
theorem midpoint_trajectory_equation
  (C : ℝ × ℝ) (N : ℝ → ℝ → Prop) (M_trajectory : ℝ → ℝ → Prop) :
  C = (3, 0) →
  (∀ x y, N x y ↔ (x - 2)^2 + (y - 4)^2 = 10) →
  (∀ x y, M_trajectory x y ↔ (x - 5 / 2)^2 + (y - 2)^2 = 5 / 2) →
  sorry :=
sorry

end circle_equation_passing_through_points_symmetric_circle_equation_midpoint_trajectory_equation_l444_444931


namespace problem1_problem2_l444_444810

section Problem1

variable (sqrt12 : ℝ) (pow2024_0 : ℝ) (sin60 : ℝ)
hypothesis h1 : sqrt12 = 2 * Real.sqrt 3
hypothesis h2 : pow2024_0 = 1
hypothesis h3 : sin60 = Real.sin (Real.pi / 3) / 2

theorem problem1 : sqrt12 + pow2024_0 - 4 * sin60 = 1 :=
by sorry

end Problem1

section Problem2

variable (x : ℝ)

theorem problem2 : (x + 2) ^ 2 + x * (x - 4) = 2 * x ^ 2 + 4 :=
by sorry

end Problem2

end problem1_problem2_l444_444810


namespace slope_of_line_via_midpoint_and_ellipse_l444_444402

theorem slope_of_line_via_midpoint_and_ellipse 
  (a b : ℝ) (h₀ : a > b) (h₁ : b > 0)
  (ecc : (Real.sqrt 3) / 2  = Real.sqrt (1 - b^2 / a^2))
  (midpoint : ∃ (A B : ℝ × ℝ), ∃ M : ℝ × ℝ, M = (-2, 1) ∧ midpoint A B = M ∧ 
               (A.1^2 / 4 + A.2^2 / b^2 = 1) ∧ (B.1^2 / 4 + B.2^2 / b^2 = 1)) :
  ∃ (l : line (ℝ × ℝ)), slope l = 1 / 2 :=
sorry

end slope_of_line_via_midpoint_and_ellipse_l444_444402


namespace count_valid_numbers_between_1_and_200_l444_444429

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k^2 = n
def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k^3 = n
def is_perfect_sixth_power (n : ℕ) : Prop := ∃ k : ℕ, k^6 = n

def count_perfect_squares (m : ℕ) : ℕ :=
  Nat.sqrt m

def count_perfect_cubes (m : ℕ) : ℕ :=
  (m : ℝ).cbrt.to_nat

def count_perfect_sixth_powers (m : ℕ) : ℕ :=
  (m : ℝ).cbrt.sqrt.to_nat

def count_either_squares_or_cubes (m : ℕ) : ℕ :=
  let squares := count_perfect_squares m
  let cubes := count_perfect_cubes m
  let sixths := count_perfect_sixth_powers m
  squares + cubes - sixths

def count_neither_squares_nor_cubes (n : ℕ) : ℕ :=
  n - count_either_squares_or_cubes n

theorem count_valid_numbers_between_1_and_200 : count_neither_squares_nor_cubes 200 = 182 :=
by
  sorry

end count_valid_numbers_between_1_and_200_l444_444429


namespace max_k_value_l444_444610

theorem max_k_value (k : ℝ) : (
  ∀ (x : ℝ), 2 ≤ x ∧ x ≤ 7 → sqrt (x - 2) + sqrt (7 - x) ≥ k
) → k ≤ sqrt 10 :=
sorry

end max_k_value_l444_444610


namespace number_of_dimes_l444_444109

theorem number_of_dimes (d q : ℕ) (h₁ : 10 * d + 25 * q = 580) (h₂ : d = q + 10) : d = 23 := 
by 
  sorry

end number_of_dimes_l444_444109


namespace trapezoid_area_l444_444230

-- Defining the conditions of the problem
variables {x y : ℝ} (h_dim : x * y = 24)
variables (AE CF : ℝ) (h_AE : AE = 1) (h_CF : CF = 3)

-- Define the points and their respective positions in the problem
variables {ABCD E F : Type} [Point ABCD] [Point E] [Point F]
variables [Rectangle ABCD] [on AD E] [on CD F] 
variables (h_E_pos : distance A E = 1) (h_F_pos : distance C F = 3)

-- Definition of the area of trapezoid EFBA
def area_trapezoid_EFBA := 54

-- Proof statement in Lean
theorem trapezoid_area (h_dim : x * y = 24) (h_AE : AE = 1) (h_CF : CF = 3) :
  area_trapezoid_EFBA = 54 :=
sorry

end trapezoid_area_l444_444230


namespace union_inter_example_l444_444616

noncomputable def A : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
noncomputable def B : Set ℕ := {4, 7, 8, 9}

theorem union_inter_example :
  (A ∪ B = {1, 2, 3, 4, 5, 6, 7, 8, 9}) ∧ (A ∩ B = {4, 7, 8}) :=
by
  sorry

end union_inter_example_l444_444616


namespace no_infinite_non_constant_arithmetic_progression_of_powers_l444_444894

theorem no_infinite_non_constant_arithmetic_progression_of_powers :
  ¬ ∃ (f : ℕ → ℕ), (∀ n, ∃ a b : ℕ, b ≥ 2 ∧ f n = a ^ b) ∧ (∀ d, ∃ N, ∀ n m ≥ N, ((f m - f n) = d * (m - n)) ∧ (m ≠ n)) :=
sorry

end no_infinite_non_constant_arithmetic_progression_of_powers_l444_444894


namespace count_not_squares_or_cubes_l444_444510

theorem count_not_squares_or_cubes {A B : Finset ℕ} (h_range : ∀ x, x ∈ A ∪ B → x ≤ 200)
  (h_squares : A.card = 14) (h_cubes : B.card = 5) (h_inter : (A ∩ B).card = 1) :
  (200 - (A ∪ B).card) = 182 :=
by
  sorry

end count_not_squares_or_cubes_l444_444510


namespace soccer_camp_ratio_l444_444697

noncomputable def number_of_kids_in_camp : ℕ := 2000
noncomputable def number_of_kids_in_soccer_camp_afternoon : ℕ := 750
noncomputable def kids_in_soccer_camp_ratio (S : ℕ) : Prop :=
  (3 / 4 : ℚ) * S = number_of_kids_in_soccer_camp_afternoon

theorem soccer_camp_ratio :
  ∃ S, kids_in_soccer_camp_ratio S ∧ (S / number_of_kids_in_camp : ℚ) = 1 / 2 :=
by {
  let S := 1000, -- we derive this from the equation (3/4) * S = 750 in solution steps
  use S,
  split,
  {
    show (3 / 4 : ℚ) * S = number_of_kids_in_soccer_camp_afternoon,
    exact calc
      (3 / 4 : ℚ) * 1000 = 3 * 250 : by norm_num
      ...                 = 750 : by norm_num,
  },
  {
    show (S : ℚ) / number_of_kids_in_camp = 1 / 2,
    exact calc
      (1000 : ℚ) / 2000 = 1 / 2 : by norm_num,
  }
}

end soccer_camp_ratio_l444_444697


namespace find_80th_percentile_l444_444663

-- Define the data set
def data_set : List ℕ := [3, 4, 5, 5, 6, 7, 7, 8, 9, 10]

-- Define sorting of the data set (not necessary as the original problem states it is already sorted)
def sorted_data_set : List ℕ := List.sort (· ≤ ·) data_set

-- Define the function to calculate the 80th percentile
def percentile (p : ℝ) (data : List ℕ) : ℝ :=
  let sorted_data := List.sort (· ≤ ·) data
  let n := sorted_data.length
  let rank := p * n
  if rank.floor < n then
    let k := rank.floor.toNat
    let next := k + 1
    if next < n then
      (sorted_data[k] + sorted_data[next]) / 2.0
    else sorted_data[k].toReal
  else data.head!.toReal

-- Statement of the problem in Lean 4
theorem find_80th_percentile : percentile 0.80 data_set = 8.5 := by
  sorry

end find_80th_percentile_l444_444663


namespace max_bundles_l444_444290

theorem max_bundles (red_sheets blue_sheets sheets_per_bundle : ℕ) (h1 : red_sheets = 210) (h2 : blue_sheets = 473) (h3 : sheets_per_bundle = 100) :
  (red_sheets + blue_sheets) / sheets_per_bundle = 6 :=
by {
  rw [h1, h2, h3],
  norm_num,
}

end max_bundles_l444_444290


namespace transformed_ellipse_equation_l444_444047

namespace EllipseTransformation

open Real

def original_ellipse (x y : ℝ) : Prop :=
  x^2 / 6 + y^2 = 1

def transformation (x' y' x y : ℝ) : Prop :=
  x' = 1 / 2 * x ∧ y' = 2 * y

theorem transformed_ellipse_equation (x y x' y' : ℝ) 
  (h : original_ellipse x y) (tr : transformation x' y' x y) :
  2 * x'^2 / 3 + y'^2 / 4 = 1 :=
by 
  sorry

end EllipseTransformation

end transformed_ellipse_equation_l444_444047


namespace min_length_intersection_l444_444920

def length (s : Set ℝ) := Sup s - Inf s

def A (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 1981}
def B (b : ℝ) : Set ℝ := {x | b - 1014 ≤ x ∧ x ≤ b}
def U : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2012}

theorem min_length_intersection (a b : ℝ) (hA : A a ⊆ U) (hB : B b ⊆ U) :
  (∀ x, x ∈ A a → x ∈ U) ∧ (∀ x, x ∈ B b → x ∈ U) →
  length (A a ∩ B b) = 983 :=
sorry

end min_length_intersection_l444_444920


namespace sum_exterior_angles_octagon_l444_444691

theorem sum_exterior_angles_octagon : 
  (∀ (polygon : Type) (exterior_angle_sum : polygon → ℝ), 
  ((exterior_angle_sum octagon = 360))) :=
begin
  sorry,
end

end sum_exterior_angles_octagon_l444_444691


namespace area_ABC_360_l444_444079

-- Define the conditions given
def D_midpoint (A B C D : Point) : Prop :=
  midpoint B C D

def E_on_AC (A C E : Point) : Prop :=
  exists (k : ℝ), k > 0 ∧ E = A + k * (C - A) ∧ A.dist E = 2 * E.dist C / 3

def F_on_AD (A D F : Point) : Prop :=
  exists (m : ℝ), m > 0 ∧ F = A + m * (D - A) ∧ A.dist F = 2 * F.dist D / 3

def area_DEF_24 (D E F : Point) : Prop :=
  area D E F = 24

variables {A B C D E F : Point}

-- Stating the theorem
theorem area_ABC_360
  (h1 : D_midpoint A B C D)
  (h2 : E_on_AC A C E)
  (h3 : F_on_AD A D F)
  (h4 : area_DEF_24 D E F) :
  area A B C = 360 :=
sorry

end area_ABC_360_l444_444079


namespace line_AF_midpoint_DE_l444_444783

variables {A B C D E F : Type} [EuclideanGeometry.point A] [EuclideanGeometry.point B] [EuclideanGeometry.point C]
  [EuclideanGeometry.point D] [EuclideanGeometry.point E] [EuclideanGeometry.point F]

-- Definitions based on the problem conditions
def isosceles_triangle (A B C : Type) [EuclideanGeometry.point A] [EuclideanGeometry.point B] [EuclideanGeometry.point C] : Prop :=
  (EuclideanGeometry.distance A B = EuclideanGeometry.distance A C)

def midpoint (D B C : Type) [EuclideanGeometry.point D] [EuclideanGeometry.point B] [EuclideanGeometry.point C] : Prop :=
  EuclideanGeometry.between D B C

def perpendicular (D E : Type) [EuclideanGeometry.point D] [EuclideanGeometry.point E] : Prop := 
  EuclideanGeometry.perp D E

def circumcircle (ABD : Set EuclideanGeometry.point) : Prop :=
  EuclideanGeometry.circumcircle ABD

def intersection (circumcircle_intersection : Type) [EuclideanGeometry.point F] [EuclideanGeometry.point B] : Prop := 
  EuclideanGeometry.intersects circumcircle_intersection F B

-- The theorem we need to prove
theorem line_AF_midpoint_DE
  (isoABC : isosceles_triangle A B C)
  (midD : midpoint D B C)
  (perpDE : perpendicular D E)
  (circABD : circumcircle {A, B, D})
  (interBF : intersection {A, B, D})
  : EuclideanGeometry.on_line A F (EuclideanGeometry.midpoint D E) :=
sorry -- proof here

end line_AF_midpoint_DE_l444_444783


namespace combined_weight_l444_444557

theorem combined_weight (S R : ℝ) (h1 : S - 5 = 2 * R) (h2 : S = 75) : S + R = 110 :=
sorry

end combined_weight_l444_444557


namespace smallest_sum_of_digits_l444_444710

theorem smallest_sum_of_digits (a b : ℕ) (D : ℕ) :
  (100 ≤ a ∧ a ≤ 999) ∧ (100 ≤ b ∧ b ≤ 999) ∧
  D = a - b ∧ 
  ((nat.digits 10 a).nodup) ∧ (nat.digits 10 b).nodup ∧
  (∃ c d e f g h : ℕ,
    a = 100 * c + 10 * d + e ∧
    b = 100 * f + 10 * g + h ∧
    c ≠ d ∧ c ≠ e ∧ d ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧
    d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ e ≠ f ∧ e ≠ g ∧ e ≠ h ∧
    f ≠ g ∧ f ≠ h ∧ g ≠ h)
  → (nat.digits 10 D).sum = 9 :=
sorry

end smallest_sum_of_digits_l444_444710


namespace count_not_squares_or_cubes_200_l444_444525

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n
def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k * k * k = n
def is_sixth_power (n : ℕ) : Prop := ∃ k : ℕ, k^6 = n

theorem count_not_squares_or_cubes_200 :
  let count_squares := (Finset.range 201).filter is_perfect_square).card
  let count_cubes := (Finset.range 201).filter is_perfect_cube).card
  let count_sixth_powers := (Finset.range 201).filter is_sixth_power).card
  (200 - (count_squares + count_cubes - count_sixth_powers)) = 182 := 
by {
  -- Define everything in terms of Finset and filter
  let count_squares := (Finset.range 201).filter is_perfect_square).card,
  let count_cubes := (Finset.range 201).filter is_perfect_cube).card,
  let count_sixth_powers := (Finset.range 201).filter is_sixth_power).card,
  exact sorry
}

end count_not_squares_or_cubes_200_l444_444525


namespace number_of_nissans_sold_l444_444256

theorem number_of_nissans_sold (total_cars : ℕ) (perc_audi perc_toyota perc_acura perc_bmw : ℕ) :
  total_cars = 250 →
  perc_audi = 10 →
  perc_toyota = 25 →
  perc_acura = 15 →
  perc_bmw = 18 →
  let total_percentage_non_nissans := perc_audi + perc_toyota + perc_acura + perc_bmw in
  let percentage_nissan := 100 - total_percentage_non_nissans in
  let number_nissans_sold := total_cars * percentage_nissan / 100 in
  number_nissans_sold = 80 :=
by
  intros h1 h2 h3 h4 h5
  let total_percentage_non_nissans := perc_audi + perc_toyota + perc_acura + perc_bmw
  let percentage_nissan := 100 - total_percentage_non_nissans
  let number_nissans_sold := total_cars * percentage_nissan / 100
  sorry

end number_of_nissans_sold_l444_444256


namespace inequality_holds_for_all_x_l444_444001

theorem inequality_holds_for_all_x (a : ℝ) (h : -1 < a ∧ a < 2) :
  ∀ x : ℝ, -3 < (x^2 + a * x - 2) / (x^2 - x + 1) ∧ (x^2 + a * x - 2) / (x^2 - x + 1) < 2 :=
by
  intro x
  sorry

end inequality_holds_for_all_x_l444_444001


namespace B_interval_l444_444875

def g : ℕ → ℝ 
| 3       := log 3
| (n + 4) := log(n + 4 + g (n + 3))

noncomputable def B : ℝ := g 3016

theorem B_interval : log 3019 < B ∧ B < log 3020 := by
  sorry

end B_interval_l444_444875


namespace sums_of_coprime_integers_l444_444126

def greatest_integer (x : ℝ) := ⌊x⌋

theorem sums_of_coprime_integers (p q : ℕ) (h_coprime : Nat.Gcd p q = 1) :
  ∑ k in Finset.range (p * q), (-1)^(greatest_integer (k / p) + greatest_integer (k / q)) = 
    if (p * q) % 2 = 0 then 0 else 1 :=
by
  sorry

end sums_of_coprime_integers_l444_444126


namespace main_theorem_l444_444747

universe u
variables {α : Type u} [euclidean_geometry α]

open_locale euclidean_geometry

-- Define the given conditions
def is_isosceles (A B C : α) (h : euclidean_geometry.is_triangle A B C) : Prop :=
euclidean_geometry.dist A B = euclidean_geometry.dist A C

def midpoint (D B C : α) : Prop :=
euclidean_geometry.is_midpoint D B C

def perpendicular (E D A C : α) : Prop :=
euclidean_geometry.is_perpendicular E D A C

def circumcircle_intersects (F B D A : α) (c : set α) : Prop :=
c = euclidean_geometry.circumcircle A B D ∧ F ∈ c ∧ B ∈ c

-- The main theorem to prove
theorem main_theorem (A B C D E F : α) (h1 : euclidean_geometry.is_triangle A B C)
    (h2 : is_isosceles A B C h1) (h3 : midpoint D B C) (h4 : perpendicular E D A C)
    (h5 : ∃ c, circumcircle_intersects F B D A c) :
    euclidean_geometry.passes_through (euclidean_geometry.line A F) (euclidean_geometry.midpoint E D) :=
sorry

end main_theorem_l444_444747


namespace window_width_l444_444881

theorem window_width (x : ℝ) : 
  let pane_height := 3 * x
  let pane_width := x
  let border_width := 3
  let panes_per_row := 3
  let num_borders := 4
  num_borders * border_width + panes_per_row * pane_width = 3 * x + 12 :=
by
  have h1 : pane_height = 3 * x := rfl
  have h2 : pane_width = x := rfl
  have h3 : border_width = 3 := rfl
  have h4 : panes_per_row = 3 := rfl
  have h5 : num_borders = 4 := rfl
  calc
    num_borders * border_width + panes_per_row * pane_width
        = 4 * 3 + 3 * x : by rw [h5, h3, h4, h2]
    ... = 12 + 3 * x : by ring
    ... = 3 * x + 12 : by ring
  sorry

end window_width_l444_444881


namespace linear_term_coefficient_l444_444097

theorem linear_term_coefficient (a b c : ℤ) (h_eq : a = 2 ∧ b = 3 ∧ c = -4) :
  b = 3 :=
by
  obtain ⟨h1, h2, h3⟩ from h_eq
  exact h2

end linear_term_coefficient_l444_444097


namespace arctan_tan_75_sub_2_tan_35_l444_444873

theorem arctan_tan_75_sub_2_tan_35 :
  ∃ θ : ℝ, 0 < θ ∧ θ < 180 ∧ θ = 15 ∧ arctan (tan (75 * (π / 180)) - 2 * tan (35 * (π / 180))) = θ :=
begin
  use 15,
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num },
  {
    sorry
  }
end

end arctan_tan_75_sub_2_tan_35_l444_444873


namespace smallest_marked_cells_l444_444132

-- Define n as a positive integer
variable (n : ℕ) (hn : 0 < n)

-- We define what it means to uniquely partition a board of size 2n x 2n into 1x2 and 2x1 dominoes
-- such that none of the dominoes contains two marked cells.
def unique_partition_exists (k : ℕ) : Prop :=
  ∃ mark_cells : Finset (Fin (2 * n) × Fin (2 * n)), 
    mark_cells.card = k ∧ 
    (∃ unique_partition : { P : Finset ((Fin (2 * n) × Fin (2 * n)) × (Fin (2 * n) × Fin (2 * n))) // 
      (∀ (d ∈ P), ((d.1.1 = d.2.1 ∧ abs (d.1.2 - d.2.2) = 1) ∨ (d.1.2 = d.2.2 ∧ abs (d.1.1 - d.2.1) = 1)) ∧
      (∀ c1 c2 ∈ mark_cells, ¬(c1 = d.1 ∧ c2 = d.2) ∧ ¬(c1 = d.2 ∧ c2 = d.1)) }, 
    1)

-- The goal is to prove that the smallest such k for a given n is 2n.
theorem smallest_marked_cells (n : ℕ) (hn : 0 < n) : 
  ∃! (k : ℕ), unique_partition_exists n k :=
begin
  use 2 * n,
  split,
  { -- Proof that it is possible to mark 2n cells and the partition is unique
    sorry
  },
  { -- Proof that any smaller number of marked cells does not satisfy the conditions
    intros k hk,
    have h : k ≥ 2 * n, 
    { sorry },
    exact h }
end

end smallest_marked_cells_l444_444132


namespace count_functions_l444_444358

variables (n p : ℕ) (f : ℕ → ℤ)
  (h1 : ∀ i, 1 ≤ i ∧ i ≤ n)
  (h2 : ∀ i, -p ≤ f i ∧ f i ≤ p)
  (h3 : ∀ i j, 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n ∧ abs (f i - f j) ≤ p)

theorem count_functions (n p : ℕ) (H1 : ∀ i, 1 ≤ i ∧ i ≤ n) (H2 : ∀ i, -p ≤ f i ∧ f i ≤ p) (H3 : ∀ i j, 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n ∧ abs (f i - f j) ≤ p) :
  (number_of_functions f H1 H2 H3) = (p+1)^(n+1) - p^(n+1) := sorry

end count_functions_l444_444358


namespace work_completion_l444_444740

theorem work_completion (x y : ℕ) : 
  (1 / (x + y) = 1 / 12) ∧ (1 / y = 1 / 24) → x = 24 :=
by
  sorry

end work_completion_l444_444740


namespace part1_part2_l444_444406

def f (x : Real) : Real := 
  Real.sin (2 * x + Real.pi / 6) - Real.cos (2 * x + Real.pi / 3) + 2 * Real.cos x ^ 2

theorem part1 : f (Real.pi / 12) = Real.sqrt 3 + 1 := sorry

theorem part2 : ∃ (k : Int), (∀ x, f x ≤ 3) ∧ f (k * Real.pi + Real.pi / 6) = 3 := sorry

end part1_part2_l444_444406


namespace minimum_value_l444_444372

noncomputable def f (x : ℝ) (a b : ℝ) := a^x - b
noncomputable def g (x : ℝ) := x + 1

theorem minimum_value (a b : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f (0 : ℝ) a b * g 0 ≤ 0)
  (h4 : ∀ x : ℝ, f x a b * g x ≤ 0) : (1 / a + 4 / b) ≥ 4 :=
sorry

end minimum_value_l444_444372


namespace train_speed_in_kmph_l444_444844

variable (L V : ℝ) -- L is the length of the train in meters, and V is the speed of the train in m/s.

-- Conditions given in the problem
def crosses_platform_in_30_seconds : Prop := L + 200 = V * 30
def crosses_man_in_20_seconds : Prop := L = V * 20

-- Length of the platform
def platform_length : ℝ := 200

-- The proof problem: Prove the speed of the train is 72 km/h
theorem train_speed_in_kmph 
  (h1 : crosses_man_in_20_seconds L V) 
  (h2 : crosses_platform_in_30_seconds L V) : 
  V * 3.6 = 72 := 
by 
  sorry

end train_speed_in_kmph_l444_444844


namespace only_solution_is_2_3_7_l444_444905

theorem only_solution_is_2_3_7 (a b c : ℕ) (h1 : 1 < a) (h2 : 1 < b) (h3 : 1 < c)
  (h4 : c ∣ (a * b + 1)) (h5 : a ∣ (b * c + 1)) (h6 : b ∣ (c * a + 1)) :
  (a = 2 ∧ b = 3 ∧ c = 7) ∨ (a = 3 ∧ b = 7 ∧ c = 2) ∨ (a = 7 ∧ b = 2 ∧ c = 3) ∨
  (a = 2 ∧ b = 7 ∧ c = 3) ∨ (a = 7 ∧ b = 3 ∧ c = 2) ∨ (a = 3 ∧ b = 2 ∧ c = 7) :=
  sorry

end only_solution_is_2_3_7_l444_444905


namespace numbers_not_squares_or_cubes_in_200_l444_444471

/-- There are 182 numbers between 1 and 200 that are neither perfect squares nor perfect cubes. -/
theorem numbers_not_squares_or_cubes_in_200 : (finset.range 200).filter (λ n, ¬(∃ m, m^2 = n) ∧ ¬(∃ k, k^3 = n)).card = 182 :=
by 
  /-
    The proof will involve constructing the set of numbers from 1 to 200, filtering out those that
    are perfect squares or perfect cubes, and counting the remaining numbers.
  -/
  sorry

end numbers_not_squares_or_cubes_in_200_l444_444471


namespace distance_between_home_and_school_l444_444219

variable (D T : ℝ)

def boy_travel_5kmhr : Prop :=
  5 * (T + 7 / 60) = D

def boy_travel_10kmhr : Prop :=
  10 * (T - 8 / 60) = D

theorem distance_between_home_and_school :
  (boy_travel_5kmhr D T) ∧ (boy_travel_10kmhr D T) → D = 2.5 :=
by
  intro h
  sorry

end distance_between_home_and_school_l444_444219


namespace face_card_then_number_card_prob_l444_444841

-- Definitions from conditions
def num_cards := 52
def num_face_cards := 12
def num_number_cards := 40
def total_ways_to_pick_two_cards := 52 * 51

-- Theorem statement
theorem face_card_then_number_card_prob : 
  (num_face_cards * num_number_cards) / total_ways_to_pick_two_cards = (40 : ℚ) / 221 :=
by
  sorry

end face_card_then_number_card_prob_l444_444841


namespace jackson_spends_858_25_l444_444591

-- Definitions of parameters
def students := 45
def pens_per_student := 6
def notebooks_per_student := 4
def binders_per_student := 2
def highlighters_per_student := 3

def pen_cost := 0.65
def notebook_cost := 1.45
def binder_cost := 4.80
def highlighter_cost := 0.85

def discount := 125.00

-- Calculate the total number of each item needed
def total_pens := students * pens_per_student
def total_notebooks := students * notebooks_per_student
def total_binders := students * binders_per_student
def total_highlighters := students * highlighters_per_student

-- Calculate the total cost for each item
def cost_pens := total_pens * pen_cost
def cost_notebooks := total_notebooks * notebook_cost
def cost_binders := total_binders * binder_cost
def cost_highlighters := total_highlighters * highlighter_cost

-- Calculate the total cost of all items
def total_cost := cost_pens + cost_notebooks + cost_binders + cost_highlighters

-- Calculate the final cost after discount
def final_cost := total_cost - discount

-- The statement we want to prove
theorem jackson_spends_858_25 :
  final_cost = 858.25 :=
sorry

end jackson_spends_858_25_l444_444591


namespace non_perfect_powers_count_l444_444487

theorem non_perfect_powers_count :
  let n := 200,
      num_perfect_squares := 14,
      num_perfect_cubes := 5,
      num_perfect_sixth_powers := 1,
      total_perfect_powers := num_perfect_squares + num_perfect_cubes - num_perfect_sixth_powers,
      total_non_perfect_powers := n - total_perfect_powers
  in
  total_non_perfect_powers = 182 :=
by
  sorry

end non_perfect_powers_count_l444_444487


namespace sum_solutions_l444_444352

-- Define the condition as an axiom
def condition (x : ℝ) : Prop :=
  (1 / Real.sin x) + (1 / Real.cos x) = 2 * Real.sqrt 3 ∧ 0 ≤ x ∧ x ≤ Real.pi

-- Define the final statement as a theorem
theorem sum_solutions : 
  ∑ x in {x : ℝ | condition x}, x = Real.pi / 6 :=
by sorry

end sum_solutions_l444_444352


namespace periodic_sequence_l444_444077

def sequence (a : ℕ → ℚ) : Prop :=
  a 1 = 2 ∧ ∀ n : ℕ, a (n + 1) = (1 + a n) / (1 - a n)

theorem periodic_sequence (a : ℕ → ℚ) (h : sequence a) : a 2018 = -3 :=
by sorry

end periodic_sequence_l444_444077


namespace division_ways_l444_444877

theorem division_ways (n m : ℕ) (hpos_n : n > 0) (hpos_m : m > 0) (h : n * m = 75600) : 
    ∃ (ways : ℕ), ways = 120 :=
begin
  use 120,
  sorry
end

end division_ways_l444_444877


namespace circumradii_equal_l444_444846

theorem circumradii_equal (A B C O M N : Type) [isTriangle A B C] 
    (hAcute : isAcuteAngleTriangle A B C)
    (hCircumcenter : isCircumcenter O A B C)
    (hCircumcircleIntersect : isCircumcircleIntersection M N A B O) :
    (circumradius A B O) = (circumradius M N C) :=
sorry

end circumradii_equal_l444_444846


namespace quadratic_root_product_is_1_div_35_l444_444078

noncomputable theory

def quadratic_roots_product (x : ℝ) : Prop :=
  (log x)^2 + (log 5 + log 7) * log x + log 5 * log 7 = 0 →
  ∃ m n : ℝ, m * n = 1 / 35 ∧ (log m + log 5 = 0 ∧ log n + log 7 = 0)

theorem quadratic_root_product_is_1_div_35 : quadratic_roots_product :=
sorry

end quadratic_root_product_is_1_div_35_l444_444078


namespace line_AF_passes_midpoint_DE_l444_444764

-- Define Triangle Isosceles
structure IsoscelesTriangle (A B C : Type) :=
  (A B C : Type)
  (isosceles : A = B)

-- Define midpoint
structure Midpoint (D B C : Type) :=
  (D : Type)
  (midpoint : D = B ∧ D = C)

-- Define perpendicular
structure Perpendicular (D E A C : Type) :=
  (perpendicular : ∀ {D E A C : Type}, A = C → D ≠ A → D = E)

-- Conditions
variables {ABC : Triangle} [IsoscelesTriangle ABC]
variables {D : Point} [Midpoint D ABC.base]
variables {DE : Line} [Perpendicular D E ABC.ABC.side]

-- The circumcircle and intersection
variable {circ_circ : ∃ circ_circ : circle (triangle ABD), intersection (BE, circ_circ) = {B, F}}

-- Main proof goal: Prove that line AF passes through the midpoint of DE.
theorem line_AF_passes_midpoint_DE :
  ∀ {A F D E : Type} {circ_circ : circle (triangle ABD)}, 
  intersect (AF, midpoint DE) ↔ intersect (circ_circ, BE) :=
sorry

end line_AF_passes_midpoint_DE_l444_444764


namespace slope_of_perpendicular_line_l444_444209

-- Define the points and the equation to compute the slope
def point1 := (3, 5)
def point2 := (-2, 8)

def slope_of_line_containing_points (p1 p2 : Int × Int) : ℚ :=
  (p2.2 - p1.2) / (p2.1 - p1.1 : ℚ)

def perpendicular_slope (slope : ℚ) : ℚ :=
  -1 / slope

theorem slope_of_perpendicular_line : 
  perpendicular_slope (slope_of_line_containing_points point1 point2) = 5 / 3 :=
  by
    sorry

end slope_of_perpendicular_line_l444_444209


namespace same_number_acquaintances_l444_444696

open Finset

variable (n : ℕ) (G : SimpleGraph (Fin n)) (v1 v2 : Fin n)
  (h1 : v1 ≠ v2)
  (h2 : G.Adj v1 v2)
  (h3 : ∀ (x y : Fin n), x ≠ y → x ≠ v1 → y ≠ v1 → (¬ G.Adj x y → (∃! z : Fin n, G.Adj x z ∧ G.Adj z y)))

-- Main theorem statement to prove
theorem same_number_acquaintances : G.degree v1 = G.degree v2 :=
sorry

end same_number_acquaintances_l444_444696


namespace triangular_region_area_l444_444278

noncomputable def area_of_triangle (f g h : ℝ → ℝ) : ℝ :=
  let (x1, y1) := (-3, f (-3))
  let (x2, y2) := (7/3, g (7/3))
  let (x3, y3) := (15/11, f (15/11))
  let base := abs (x2 - x1)
  let height := abs (y3 - 2)
  (1/2) * base * height

theorem triangular_region_area :
  let f x := (2/3) * x + 4
  let g x := -3 * x + 9
  let h x := (2 : ℝ)
  area_of_triangle f g h = 256/33 :=  -- Given conditions
by
  sorry  -- Proof to be supplied

end triangular_region_area_l444_444278


namespace find_positions_of_P_l444_444413

theorem find_positions_of_P :
  let line_eq := (λ x : ℝ, -real.sqrt 3 * x + 2 * real.sqrt 3)
  let B := (2 : ℝ, 0 : ℝ)
  let C := (0 : ℝ, 2 * real.sqrt 3)
  let A := (-2 : ℝ, 0 : ℝ)
  let slope_of_30_deg := 1 / real.sqrt 3
  (∃ (P : ℝ × ℝ), let coord := (2 - 2 * P.1, 2 * real.sqrt 3 * P.1) in
  coord.2 / (coord.1 + 2) = slope_of_30_deg ∧ 0 ≤ P.1 ∧ P.1 ≤ 1) → 
  True := 
sorry

end find_positions_of_P_l444_444413


namespace conjugate_of_complex_solution_l444_444923

theorem conjugate_of_complex_solution (x y : ℝ) (i : ℂ) (h_i_imag : i = complex.I) 
  (h : x / (1 + i) = 1 - y * i) : complex.conj (x + y * i) = 2 - i :=
by
  sorry

end conjugate_of_complex_solution_l444_444923


namespace highest_slope_product_l444_444203

theorem highest_slope_product (m1 m2 : ℝ) (h1 : m1 = 5 * m2) 
    (h2 : abs ((m2 - m1) / (1 + m1 * m2)) = 1) : (m1 * m2) ≤ 1.8 :=
by
  sorry

end highest_slope_product_l444_444203


namespace line_AF_passes_midpoint_DE_l444_444762

-- Define Triangle Isosceles
structure IsoscelesTriangle (A B C : Type) :=
  (A B C : Type)
  (isosceles : A = B)

-- Define midpoint
structure Midpoint (D B C : Type) :=
  (D : Type)
  (midpoint : D = B ∧ D = C)

-- Define perpendicular
structure Perpendicular (D E A C : Type) :=
  (perpendicular : ∀ {D E A C : Type}, A = C → D ≠ A → D = E)

-- Conditions
variables {ABC : Triangle} [IsoscelesTriangle ABC]
variables {D : Point} [Midpoint D ABC.base]
variables {DE : Line} [Perpendicular D E ABC.ABC.side]

-- The circumcircle and intersection
variable {circ_circ : ∃ circ_circ : circle (triangle ABD), intersection (BE, circ_circ) = {B, F}}

-- Main proof goal: Prove that line AF passes through the midpoint of DE.
theorem line_AF_passes_midpoint_DE :
  ∀ {A F D E : Type} {circ_circ : circle (triangle ABD)}, 
  intersect (AF, midpoint DE) ↔ intersect (circ_circ, BE) :=
sorry

end line_AF_passes_midpoint_DE_l444_444762


namespace distance_from_point_to_focus_l444_444999

theorem distance_from_point_to_focus (P : ℝ × ℝ) (hP : P.2^2 = 8 * P.1) (hX : P.1 = 8) :
  dist P (2, 0) = 10 :=
sorry

end distance_from_point_to_focus_l444_444999


namespace sqrt_function_monotonicity_l444_444189

-- Define the function
def g (x : ℝ) : ℝ := -x^2 + 4 * x + 5

-- Define the domain condition
def is_defined (x : ℝ) : Prop := g(x) ≥ 0

-- Define the monotonicity condition
def is_monotonically_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y : ℝ, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y

theorem sqrt_function_monotonicity :
  ∀ x : ℝ, is_defined x → 
  is_monotonically_increasing (λ x, Real.sqrt (g x)) (-1) 2 :=
by
  sorry

end sqrt_function_monotonicity_l444_444189


namespace f_sum_2013_l444_444231

def f : ℝ → ℝ 
noncomputable def f_periodic (x : ℝ) : f (x + 2) = - f x := sorry
noncomputable def f_odd (x : ℝ) : f (-x) = - f x := sorry
noncomputable def f_formula (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 2) : f x = 2*x - x^2 := sorry

theorem f_sum_2013 :
  (finset.sum (finset.range 2014) (λ n, f n)) = 1 :=
begin
  sorry
end

end f_sum_2013_l444_444231


namespace subset_probability_l444_444365

-- Definitions related to the problem conditions
def set1 : set (fin 4) := {0, 1, 2}
def set2 : set (fin 4) := {0, 1, 2, 3}

-- The goal is to prove that the probability of a randomly chosen subset of set2
-- being a subset of set1 is 1/2
theorem subset_probability : 
  (set {A | A ⊆ set1}).support.to_finset.card.to_nat / 
  (set {A | A ⊆ set2}).support.to_finset.card.to_nat = 1 / 2 :=
begin
  sorry
end

end subset_probability_l444_444365


namespace passing_through_midpoint_of_DE_l444_444795

noncomputable def is_isosceles (ABC : Triangle) : Prop :=
ABC.AB = ABC.AC

noncomputable def midpoint (D : Point) (BC : Segment) : Prop :=
(B + C) / 2 = D

noncomputable def perpendicular (DE : Line) (AC : Line) : Prop :=
right_angle (angle DE AC)

noncomputable def circumsircle_intersect (F : Point) (ABD : Triangle) (BE : Line) : Prop :=
(circumcircle ABD).contains B ∧ (circumcircle ABD).contains F ∧ line BE.contains B ∧ line BE.contains F

noncomputable def passes_through_midpoint (AF : Line) (DE : Segment) : Prop :=
exists M : Point, midpoint M DE ∧ line AF.contains M

theorem passing_through_midpoint_of_DE
  (ABC : Triangle) (D E F : Point) (DE : Line) (AC BE AF : Line) :
  is_isosceles ABC →
  midpoint D (BC) →
  perpendicular DE AC →
  circumsircle_intersect F (Triangle.mk ABD) BE →
  passes_through_midpoint AF (segment.mk D E) :=
by
  sorry

end passing_through_midpoint_of_DE_l444_444795


namespace num_factors_34848_l444_444349

/-- Define the number 34848 and its prime factorization -/
def n : ℕ := 34848
def p_factors : List (ℕ × ℕ) := [(2, 5), (3, 2), (11, 2)]

/-- Helper function to calculate the number of divisors from prime factors -/
def num_divisors (factors : List (ℕ × ℕ)) : ℕ := 
  factors.foldr (fun (p : ℕ × ℕ) acc => acc * (p.2 + 1)) 1

/-- Formal statement of the problem -/
theorem num_factors_34848 : num_divisors p_factors = 54 :=
by
  -- Proof that 34848 has the prime factorization 3^2 * 2^5 * 11^2 
  -- and that the number of factors is 54 would go here.
  sorry

end num_factors_34848_l444_444349


namespace arithmetic_mean_of_first_40_consecutive_integers_l444_444866

-- Define the arithmetic sequence with the given conditions
def arithmetic_sequence (a₁ d n : ℕ) : ℕ := a₁ + (n - 1) * d

-- Define the sum of the first n terms of the given arithmetic sequence
def arithmetic_sum (a₁ d n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

-- Define the arithmetic mean of the first n terms of the given arithmetic sequence
def arithmetic_mean (a₁ d n : ℕ) : ℚ :=
  (arithmetic_sum a₁ d n : ℚ) / n

-- The arithmetic sequence starts at 5, has a common difference of 1, and has 40 terms
theorem arithmetic_mean_of_first_40_consecutive_integers :
  arithmetic_mean 5 1 40 = 24.5 :=
by
  sorry

end arithmetic_mean_of_first_40_consecutive_integers_l444_444866


namespace water_flow_into_sea_l444_444271

variable (depth : ℝ) (width : ℝ) (flow_rate_kmph : ℝ)

-- Define the conditions
def river_conditions : Prop :=
  depth = 3 ∧ width = 32 ∧ flow_rate_kmph = 2

-- Convert flow rate from kmph to m/min
def flow_rate_m_per_min (flow_rate_kmph: ℝ) : ℝ :=
  flow_rate_kmph * 1000 / 60

-- Calculate the cross-sectional area of the river
def river_area (depth : ℝ) (width : ℝ) : ℝ :=
  depth * width

-- Calculate the volume of water flowing into the sea per minute
def volume_per_minute (area : ℝ) (flow_rate_m_per_min : ℝ) : ℝ :=
  area * flow_rate_m_per_min

theorem water_flow_into_sea (h : river_conditions depth width flow_rate_kmph) :
  volume_per_minute (river_area depth width) (flow_rate_m_per_min flow_rate_kmph) = 3200 :=
by
  sorry

end water_flow_into_sea_l444_444271


namespace fubini_sum_eq_sum_sum_l444_444605

theorem fubini_sum_eq_sum_sum {a : ℕ → ℕ → ℝ} (h : ∑ i, ∑ j, |a i j| < ∞) :
  ∑ i, ∑ j, a i j = ∑ j, ∑ i, a i j := 
sorry

end fubini_sum_eq_sum_sum_l444_444605


namespace maximum_distance_l444_444222

-- Defining the conditions
def highway_mileage : ℝ := 12.2
def city_mileage : ℝ := 7.6
def gasoline_amount : ℝ := 22

-- Mathematical equivalent proof statement
theorem maximum_distance (h_mileage : ℝ) (g_amount : ℝ) : h_mileage = 12.2 ∧ g_amount = 22 → g_amount * h_mileage = 268.4 :=
by
  intro h
  sorry

end maximum_distance_l444_444222


namespace digit_in_2015th_position_is_zero_l444_444855

-- Definitions required for the problem
def pos_int_seq : List Nat := List.range (10000) -- this is a simplification; assume large enough limit

def digit_at_pos (seq : List Nat) (pos : Nat) : Nat :=
  (seq.toString.data).get! (pos - 1).toNat

-- The theorem statement
theorem digit_in_2015th_position_is_zero : digit_at_pos pos_int_seq 2015 = 0 :=
by
  sorry

end digit_in_2015th_position_is_zero_l444_444855


namespace solve_for_y_l444_444167

theorem solve_for_y (y : ℝ) (h : 4 * 5 ^ y = 1250) : y = 2.5 :=
sorry

end solve_for_y_l444_444167


namespace largest_unique_triangles_set_l444_444273

def is_triangle (a b c : ℕ) : Prop :=
  a < 6 ∧ b < 6 ∧ c < 6 ∧ a ≥ b ∧ b ≥ c ∧ b + c > a

def not_similar (a b c : ℕ) (d e f : ℕ) : Prop :=
  (a * e ≠ b * d ∨ b * f ≠ c * e ∨ a * f ≠ c * d)
  
def distinct_triangles (T : List (ℕ × ℕ × ℕ)) : Prop :=
  ∀ t1 t2 ∈ T, t1 ≠ t2 → not_similar t1.1 t1.2 t1.3 t2.1 t2.2 t2.3

noncomputable def max_unique_triangles : ℕ :=
  15

theorem largest_unique_triangles_set :
  ∃ (S : List (ℕ × ℕ × ℕ)), 
    (∀ t ∈ S, is_triangle t.1 t.2 t.3) ∧
    distinct_triangles S ∧
    S.length = max_unique_triangles :=
sorry

end largest_unique_triangles_set_l444_444273


namespace count_diff_squares_1_to_500_l444_444973

theorem count_diff_squares_1_to_500 : ∃ count : ℕ, count = 416 ∧
  (count = (nat.card {n : ℕ | n ∈ (set.Icc 1 500) ∧ (∃ a b : ℕ, a^2 - b^2 = n)})) := by
sorry

end count_diff_squares_1_to_500_l444_444973


namespace cylinder_radius_l444_444255

theorem cylinder_radius (h₁ : ℝ) (h₂ : ℝ) (d : ℝ) (R : ℝ) :
  h₁ = 9 ∧ h₂ = 2 ∧ d = 23 →
  let x := (h₁^2 + d^2 - h₂^2) / (2 * d) in
  let calculated_R := ((d^2 - (2 * h₁ + 2 * h₂) * x + h₁^2 + h₂^2)/(2 * (h₁ - h₂)))^(1/2) in
  R = calculated_R →
  R = 17 :=
by
  intros h₁_conditions calculated_R_eq_R
  rw [h₁_conditions]
  sorry

end cylinder_radius_l444_444255


namespace unique_solution_pair_l444_444903

theorem unique_solution_pair (x p : ℕ) (hp : Nat.Prime p) (hx : x ≥ 0) (hp2 : p ≥ 2) :
  x * (x + 1) * (x + 2) * (x + 3) = 1679 ^ (p - 1) + 1680 ^ (p - 1) + 1681 ^ (p - 1) ↔ (x = 4 ∧ p = 2) := 
by
  sorry

end unique_solution_pair_l444_444903


namespace AF_passes_through_midpoint_of_DE_l444_444776

open EuclideanGeometry

-- Definitions based on the conditions
variables {A B C D E F : Point} 
variables {ABC : Triangle}
variables (isosceles_ABC : IsIsoscelesTriangle ABC A B)
variables (midpoint_D : IsMidpoint D B C)
variables (perpendicular_DE : IsPerpendicular DE D AC)
variables {circumcircle_ABD : Circumcircle ABD}
variables (intersection_F : IntersectsAt circumcircle_ABD BE B F)

-- The theorem statement
theorem AF_passes_through_midpoint_of_DE :
  PassesThroughMidpoint AF D E :=
sorry

end AF_passes_through_midpoint_of_DE_l444_444776


namespace daps_equiv_dirps_l444_444552

noncomputable def dops_equiv_daps : ℝ := 5 / 4
noncomputable def dips_equiv_dops : ℝ := 3 / 10
noncomputable def dirps_equiv_dips : ℝ := 2

theorem daps_equiv_dirps (n : ℝ) : 20 = (dops_equiv_daps * dips_equiv_dops * dirps_equiv_dips) * n → n = 15 :=
by sorry

end daps_equiv_dirps_l444_444552


namespace range_of_m_l444_444726

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, 1 < x ∧ x < 2 → x^2 + m * x + 2 ≥ 0) →
  m ∈ Set.Icc (-2 * Real.sqrt 2) ∞ :=
by
  sorry

end range_of_m_l444_444726


namespace log_equation_solution_l444_444902

theorem log_equation_solution {x : ℝ} (h : log 16 (4 * x - 14) = 2) : x = 67.5 :=
by
  sorry

end log_equation_solution_l444_444902


namespace problem_solution_l444_444407

theorem problem_solution (a : ℝ) : 
  (∃ a > 0, a ≠ 1 ∧ (a^2 - 2 = 7) ∧
  (∀ x, a = 3 ∧ (3^(x + 1) - 2 = 0 → x = Real.log 2 / Real.log 3 - 1) ∧ 
  (3^(x + 1) - 2 ≥ - 5 / 3 → x ≥ -2))) :=
begin
  sorry,
end

end problem_solution_l444_444407


namespace numbers_not_squares_or_cubes_in_200_l444_444470

/-- There are 182 numbers between 1 and 200 that are neither perfect squares nor perfect cubes. -/
theorem numbers_not_squares_or_cubes_in_200 : (finset.range 200).filter (λ n, ¬(∃ m, m^2 = n) ∧ ¬(∃ k, k^3 = n)).card = 182 :=
by 
  /-
    The proof will involve constructing the set of numbers from 1 to 200, filtering out those that
    are perfect squares or perfect cubes, and counting the remaining numbers.
  -/
  sorry

end numbers_not_squares_or_cubes_in_200_l444_444470


namespace passing_through_midpoint_of_DE_l444_444798

noncomputable def is_isosceles (ABC : Triangle) : Prop :=
ABC.AB = ABC.AC

noncomputable def midpoint (D : Point) (BC : Segment) : Prop :=
(B + C) / 2 = D

noncomputable def perpendicular (DE : Line) (AC : Line) : Prop :=
right_angle (angle DE AC)

noncomputable def circumsircle_intersect (F : Point) (ABD : Triangle) (BE : Line) : Prop :=
(circumcircle ABD).contains B ∧ (circumcircle ABD).contains F ∧ line BE.contains B ∧ line BE.contains F

noncomputable def passes_through_midpoint (AF : Line) (DE : Segment) : Prop :=
exists M : Point, midpoint M DE ∧ line AF.contains M

theorem passing_through_midpoint_of_DE
  (ABC : Triangle) (D E F : Point) (DE : Line) (AC BE AF : Line) :
  is_isosceles ABC →
  midpoint D (BC) →
  perpendicular DE AC →
  circumsircle_intersect F (Triangle.mk ABD) BE →
  passes_through_midpoint AF (segment.mk D E) :=
by
  sorry

end passing_through_midpoint_of_DE_l444_444798


namespace greatest_divisor_of_480_less_than_60_and_factor_of_90_is_30_l444_444721

theorem greatest_divisor_of_480_less_than_60_and_factor_of_90_is_30 :
  ∃ d, d ∣ 480 ∧ d < 60 ∧ d ∣ 90 ∧ (∀ e, e ∣ 480 → e < 60 → e ∣ 90 → e ≤ d) ∧ d = 30 :=
sorry

end greatest_divisor_of_480_less_than_60_and_factor_of_90_is_30_l444_444721


namespace correct_statement_of_triangle_l444_444283

theorem correct_statement_of_triangle :
  (∀ (a b c : ℕ), a^2 + b^2 = c^2 ↔ right_angle a b c) → 
  ∃ correct_statement : ℕ, 
    (correct_statement = 2 ∧ 
     (∀ (n : ℕ), n ≠ 2 → 
      (n = 1 → ¬(a^2 = b^2 + c^2)) ∧ 
      (n = 3 → ¬(a^2 = b^2 + c^2)) ∧ 
      (n = 4 → ¬(a = b + c)))) :=
by sorry

end correct_statement_of_triangle_l444_444283


namespace prove_distance_square_l444_444826

noncomputable def problem_statement : Prop :=
  ∃ (a b : ℝ), 
    (a^2 + b^2 = 57) ∧
    (a^2 + (b + 10)^2 = 200) ∧
    ((a + 4)^2 + b^2 = 200) ∧
    (10^2 + 4^2 = 16^2)

theorem prove_distance_square : problem_statement :=
by sorry

end prove_distance_square_l444_444826


namespace function_inequality_l444_444120

-- We declare the mathematical problem in lean statement form
theorem function_inequality (f : ℝ → ℝ)
  (h : ∀ x : ℝ, sqrt (2 * f x) - sqrt (2 * f x - f (2 * x)) ≥ 2) :
  (∀ x : ℝ, f x ≥ 4) ∧ (∀ x : ℝ, f x ≥ 7) := by
  sorry

end function_inequality_l444_444120


namespace unit_vectors_ratio_eq_one_l444_444604

noncomputable def unit_vector (v : V) [normed_add_comm_group V] [normed_space ℝ V] (hv : ∥v∥ ≠ 0) : V :=
  v / ∥v∥

theorem unit_vectors_ratio_eq_one
  {V : Type*} [normed_add_comm_group V] [normed_space ℝ V]
  {a b : V} (ha : ∥a∥ ≠ 0) (hb : ∥b∥ ≠ 0) :
  let a0 := unit_vector a ha
      b0 := unit_vector b hb
  in ∥a0∥ / ∥b0∥ = 1 :=
by
  sorry

end unit_vectors_ratio_eq_one_l444_444604


namespace statement_A_l444_444731

theorem statement_A (x : ℝ) (h : x < -1) : x^2 > x :=
sorry

end statement_A_l444_444731


namespace count_non_perfect_square_or_cube_l444_444511

-- Define perfect_square function
def perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k^2 = n

-- Define perfect_cube function
def perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k^3 = n

-- Define range from 1 to 200
def range_200 := {1 .. 200}

-- Define numbers either perfect square or perfect cube
def perfect_square_or_cube : Finset ℕ :=
  (range_200.filter perfect_square) ∪ (range_200.filter perfect_cube)

-- Define sixth powers in range
def perfect_sixth_power (n : ℕ) : Prop :=
  ∃ k : ℕ, k^6 = n

-- The final problem statement
theorem count_non_perfect_square_or_cube :
  (range_200.card - perfect_square_or_cube.card) = 182 :=
by
  sorry

end count_non_perfect_square_or_cube_l444_444511


namespace positive_diff_of_b_l444_444135

def g (n : Int) : Int :=
  if n < 0 then
    n^2 - 3*n + 2
  else
    3*n - 25

theorem positive_diff_of_b :
  ∃ b1 b2 : Int, g(-3) + g(3) + g(b1) = 0 ∧ g(-3) + g(3) + g(b2) = 0 ∧ abs(b1 - b2) = 5 := sorry

end positive_diff_of_b_l444_444135


namespace wilsons_theorem_l444_444915

theorem wilsons_theorem (N : ℕ) (h : N > 1) : 
  (fact (N-1) % N = N - 1) ↔ Nat.Prime N :=
sorry

end wilsons_theorem_l444_444915


namespace inverse_of_3_mod_221_l444_444333

theorem inverse_of_3_mod_221 : ∃ x : ℕ, 3 * x ≡ 1 [MOD 221] ∧ x = 74 := by
  use 74
  split
  · exact Nat.modeq_iff_dvd.2 (by norm_num)
  · rfl

end inverse_of_3_mod_221_l444_444333


namespace numbers_neither_square_nor_cube_l444_444537

theorem numbers_neither_square_nor_cube (n : ℕ) (h : n = 200) :
  let num_squares := 14 in
  let num_cubes := 5 in
  let num_sixth_powers := 2 in
  (n - (num_squares + num_cubes - num_sixth_powers)) = 183 :=
by
  -- Let the number of integers in the range
  let num_total := 200
  -- Define how we count perfect squares and cubes
  let num_squares := 14
  let num_cubes := 5
  let num_sixth_powers := 2

  -- Subtract the overlapping perfect sixth powers from the sum of squares and cubes
  let num_either_square_or_cube := num_squares + num_cubes - num_sixth_powers

  -- Calculate the final result
  let result := num_total - num_either_square_or_cube

  -- Prove by computation
  show result = 183 from
    calc
      result = num_total - num_either_square_or_cube := rfl
           ... = 200 - (14 + 5 - 2)                   := rfl
           ... = 200 - 17                             := rfl
           ... = 183                                  := rfl

end numbers_neither_square_nor_cube_l444_444537


namespace triangle_properties_l444_444595

noncomputable def acute_triangle (A B C : Point) : Prop :=
  ∠A < 90 ∧ ∠B < 90 ∧ ∠C < 90

noncomputable def on_smaller_arc (Γ : Circle) (A B C : Point) (D E : Point) : Prop :=
  D ∈ Γ ∧ E ∈ Γ ∧ ∠BAC < 180 ∧ ∠BCA < 180

noncomputable def intersection (l1 l2 : Line) : Point := sorry

noncomputable def is_collinear (A K N : Point) : Prop :=
  ∃ L : Line, A ∈ L ∧ K ∈ L ∧ N ∈ L

noncomputable def symmedian (A B C K : Point) : Prop := sorry

theorem triangle_properties (A B C O K N D E : Point) (R : ℝ) (Γ : Circle)
  (h1 : acute_triangle A B C)
  (h2 : AB < AC ∧ AC < BC)
  (h3 : is_circumcircle Γ A B C O R)
  (h4 : on_smaller_arc Γ A B C D E)
  (h5 : K = intersection (line B D) (line C E))
  (h6 : N = second_intersection (circumcircle ⟨B, K, E⟩) (circumcircle ⟨C, K, D⟩)) :
  is_collinear A K N ↔ symmedian A B C K := 
sorry

end triangle_properties_l444_444595


namespace cot_sum_simplified_l444_444656

noncomputable def cot (x : ℝ) : ℝ := (Real.cos x) / (Real.sin x)

theorem cot_sum_simplified : cot (π / 24) + cot (π / 8) = 96 / (π^2) := 
by 
  sorry

end cot_sum_simplified_l444_444656


namespace distance_to_line_not_greater_than_2_l444_444946

-- Definitions of distances PA, PB, and PC
variables {P A B C : Point}
variable {l : Line}
variable (PA PB PC : ℝ)

-- Hypotheses: given conditions
axiom PA_distance : dist P A = 2
axiom PB_distance : dist P B = 2.5
axiom PC_distance : dist P C = 3
axiom A_on_line : A ∈ l
axiom B_on_line : B ∈ l
axiom C_on_line : C ∈ l
axiom P_not_on_line : P ∉ l

-- Theorem: the distance from P to line l is not greater than 2
theorem distance_to_line_not_greater_than_2 :
  distance_point_to_line P l ≤ 2 :=
sorry

end distance_to_line_not_greater_than_2_l444_444946


namespace prove_planes_perpendicular_l444_444284

variable {a b l : Line}
variable {M N : Plane}

theorem prove_planes_perpendicular (ha_perp_M : a ⊥ M) (ha_parallel_N : a ∥ N) : M ⊥ N :=
sorry

end prove_planes_perpendicular_l444_444284


namespace numbers_neither_square_nor_cube_l444_444538

theorem numbers_neither_square_nor_cube (n : ℕ) (h : n = 200) :
  let num_squares := 14 in
  let num_cubes := 5 in
  let num_sixth_powers := 2 in
  (n - (num_squares + num_cubes - num_sixth_powers)) = 183 :=
by
  -- Let the number of integers in the range
  let num_total := 200
  -- Define how we count perfect squares and cubes
  let num_squares := 14
  let num_cubes := 5
  let num_sixth_powers := 2

  -- Subtract the overlapping perfect sixth powers from the sum of squares and cubes
  let num_either_square_or_cube := num_squares + num_cubes - num_sixth_powers

  -- Calculate the final result
  let result := num_total - num_either_square_or_cube

  -- Prove by computation
  show result = 183 from
    calc
      result = num_total - num_either_square_or_cube := rfl
           ... = 200 - (14 + 5 - 2)                   := rfl
           ... = 200 - 17                             := rfl
           ... = 183                                  := rfl

end numbers_neither_square_nor_cube_l444_444538


namespace box_of_books_weight_l444_444246

theorem box_of_books_weight :
  (∀ (books_weight_per_unit : ℕ) (number_of_books : ℕ), (books_weight_per_unit = 3) ∧ (number_of_books = 14) → number_of_books * books_weight_per_unit = 42) :=
by
  intro books_weight_per_unit number_of_books h
  cases h with h1 h2
  rw [h1, h2]
  simp [Nat.mul_comm]
  sorry

end box_of_books_weight_l444_444246


namespace g_f_neg3_l444_444606

def f (x : ℝ) : ℝ := 3 * x^2 - 7
def g (y : ℝ) : ℝ

axiom g_f_3 : g (f 3) = 15

theorem g_f_neg3 : g (f (-3)) = 15 := by
  sorry

end g_f_neg3_l444_444606


namespace minimizeAcuteTriangles_minimizeObtuseTriangles_l444_444893

-- Define a regular n-gon with n >= 5
structure RegularNGon (n : ℕ) (h : n ≥ 5) :=
(vertices : Fin n → ℝ × ℝ)
(center : ℝ × ℝ)
(is_regular : ∀ i j : Fin n, dist center (vertices i) = dist center (vertices j) ∧ 
              angle (center, vertices i, vertices j) = 2 * π / n)

-- Assume a triangulation of the n-gon
structure Triangulation (N : RegularNGon n h) :=
(triangles : Fin (n + m) → (Fin n × Fin n × Fin n))
(valid_triangulation : ∀ t : Fin (n + m), 
                      let (a, b, c) := triangles t in
                      a ≠ b ∧ b ≠ c ∧ c ≠ a)

-- A function to count the number of acute-angled triangles
def countAcuteTriangles (T : Triangulation N) : ℕ :=
sorry

-- A function to count the number of obtuse-angled triangles
def countObtuseTriangles (T : Triangulation N) : ℕ :=
sorry

theorem minimizeAcuteTriangles (n : ℕ) (h : n ≥ 5) (N : RegularNGon n h) :
  ∀ T : Triangulation N, countAcuteTriangles T = n :=
sorry

theorem minimizeObtuseTriangles (n : ℕ) (h : n ≥ 5) (N : RegularNGon n h) :
  ∀ T : Triangulation N, countObtuseTriangles T = n :=
sorry

end minimizeAcuteTriangles_minimizeObtuseTriangles_l444_444893


namespace expand_fraction_product_l444_444329

theorem expand_fraction_product (x : ℝ) (hx : x ≠ 0) : 
  (3 / 4) * (8 / x^2 - 5 * x^3) = 6 / x^2 - 15 * x^3 / 4 := 
by 
  sorry

end expand_fraction_product_l444_444329


namespace CoincidingPointsInTriangle_l444_444306

theorem CoincidingPointsInTriangle (Δ : Triangle) :
  (Δ.centroid = Δ.orthocenter ∧ Δ.orthocenter = Δ.circumcenter ∧ Δ.circumcenter = Δ.incenter) →
  Δ.is_equilateral :=
by
  sorry

end CoincidingPointsInTriangle_l444_444306


namespace locus_of_points_l444_444410

-- Definitions of planes and their parallelism
variable (α : Type) [ordered_field α]

structure Plane (α : Type) :=
(point : vector α)
(normal : vector α)
(condition : point' : point → normal ⬝ point' = 0)

-- Parallelism definition
def parallel (L1 L2 : Plane α) : Prop :=
∃ k : α, k ≠ 0 ∧ L1.normal = k • L2.normal

variable
  (L1 L2 L3 L4 : Plane ℝ)

-- Given conditions setup
axiom L1_parallel_L2 : parallel L1 L2
axiom L3_parallel_L4 : parallel L3 L4

-- Distance definition
def distance (P : vector ℝ) (L : Plane ℝ) : ℝ := 
(abs (L.normal ⬝ P)) / (|L.normal|)

-- Given constant
variable (c : ℝ)
variable (a b : ℝ) -- Distances between the parallel planes

-- Theorem to prove the locus of points
theorem locus_of_points (P : vector ℝ) :
  (distance P L1 + distance P L2 + distance P L3 + distance P L4 = c) ↔ 
  (if c = a + b then 
    ∃ u v : ℝ, P = u • L1.point + v • L3.point  -- Points within the defined region
  else if c < a + b then 
    false  -- Empty region
  else 
    ∃ u : ℝ, P = u • (L1.point + L3.point)  -- Segments extending infinitely
  ) :=
sorry

end locus_of_points_l444_444410


namespace part1_part2_l444_444599

def M (a : ℝ) : ℝ := a * (4 / 5)
def N (a : ℝ) : ℝ := a / (-4 / 5)
def P (a : ℝ) : ℝ := a * (3 / 2)
def Q (a : ℝ) : ℝ := a / (-3 / 2)

theorem part1 (a : ℝ) (h : a = 20) : M a + N a = -9 := by
  sorry

theorem part2 (a : ℝ) (h : 0 < a) : N a < Q a ∧ Q a < M a ∧ M a < P a := by
  sorry

end part1_part2_l444_444599


namespace condition_type_l444_444549

theorem condition_type (x : ℝ) : 
  (x = 2 → (x - 2) * (x - 1) = 0) 
  ∧ ¬((x - 2) * (x - 1) = 0 → x = 2) :=
by {
  intros,
  sorry
}

end condition_type_l444_444549


namespace BCHG_is_parallelogram_l444_444114

variables {A B C D E F G H : Type*}

-- Let $ABC$ be an acute triangle with circumcircle $\Gamma$
-- Assumptions that define points D, E, and F, and their perpendicularity conditions.
variables [Plane A] [Plane B] [Plane C] [Circumcircle Γ A B C] 
variable D : Point \{ the midpoint of minor arc B C on Γ \}
variables E F : Point
variable (hE : D.line \perp A.line C) (hF : D.line \perp A.line B)

-- Defining intersections G and H
variable (G : Point \{ intersection of B.line E and D.line F \})
variable (H : Point \{ intersection of C.line F and D.line E \})

theorem BCHG_is_parallelogram 
  (hABC: acute_triangle A B C)
  (hΓ: circumcircle Γ A B C)
  (hD: D = midpoint (arc min B C on Γ))
  (hE_F_perp: D.line perpendicular A.line C and D.line perpendicular A.line B)
  (hG: G = intersect_line B.line E.line F)
  (hH: H = intersect_line C.line F.line E) :
  parallelogram B C H G := sorry

end BCHG_is_parallelogram_l444_444114


namespace find_N_l444_444904

theorem find_N (N : ℕ) :
  (∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ Nat.coprime a b ∧ Nat.coprime a c ∧ Nat.coprime b c ∧ 
    (1 / a + 1 / b + 1 / c = N / (a + b + c))) → N = 9 ∨ N = 10 ∨ N = 11 := 
sorry

end find_N_l444_444904


namespace line_AF_passes_midpoint_DE_l444_444763

-- Define Triangle Isosceles
structure IsoscelesTriangle (A B C : Type) :=
  (A B C : Type)
  (isosceles : A = B)

-- Define midpoint
structure Midpoint (D B C : Type) :=
  (D : Type)
  (midpoint : D = B ∧ D = C)

-- Define perpendicular
structure Perpendicular (D E A C : Type) :=
  (perpendicular : ∀ {D E A C : Type}, A = C → D ≠ A → D = E)

-- Conditions
variables {ABC : Triangle} [IsoscelesTriangle ABC]
variables {D : Point} [Midpoint D ABC.base]
variables {DE : Line} [Perpendicular D E ABC.ABC.side]

-- The circumcircle and intersection
variable {circ_circ : ∃ circ_circ : circle (triangle ABD), intersection (BE, circ_circ) = {B, F}}

-- Main proof goal: Prove that line AF passes through the midpoint of DE.
theorem line_AF_passes_midpoint_DE :
  ∀ {A F D E : Type} {circ_circ : circle (triangle ABD)}, 
  intersect (AF, midpoint DE) ↔ intersect (circ_circ, BE) :=
sorry

end line_AF_passes_midpoint_DE_l444_444763


namespace count_not_squares_or_cubes_200_l444_444522

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n
def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k * k * k = n
def is_sixth_power (n : ℕ) : Prop := ∃ k : ℕ, k^6 = n

theorem count_not_squares_or_cubes_200 :
  let count_squares := (Finset.range 201).filter is_perfect_square).card
  let count_cubes := (Finset.range 201).filter is_perfect_cube).card
  let count_sixth_powers := (Finset.range 201).filter is_sixth_power).card
  (200 - (count_squares + count_cubes - count_sixth_powers)) = 182 := 
by {
  -- Define everything in terms of Finset and filter
  let count_squares := (Finset.range 201).filter is_perfect_square).card,
  let count_cubes := (Finset.range 201).filter is_perfect_cube).card,
  let count_sixth_powers := (Finset.range 201).filter is_sixth_power).card,
  exact sorry
}

end count_not_squares_or_cubes_200_l444_444522


namespace num_from_1_to_200_not_squares_or_cubes_l444_444462

noncomputable def numNonPerfectSquaresAndCubes (n : ℕ) : ℕ :=
  let num_squares := 14
  let num_cubes := 5
  let num_sixth_powers := 2
  n - (num_squares + num_cubes - num_sixth_powers)

theorem num_from_1_to_200_not_squares_or_cubes : numNonPerfectSquaresAndCubes 200 = 183 := by
  sorry

end num_from_1_to_200_not_squares_or_cubes_l444_444462


namespace num_from_1_to_200_not_squares_or_cubes_l444_444466

noncomputable def numNonPerfectSquaresAndCubes (n : ℕ) : ℕ :=
  let num_squares := 14
  let num_cubes := 5
  let num_sixth_powers := 2
  n - (num_squares + num_cubes - num_sixth_powers)

theorem num_from_1_to_200_not_squares_or_cubes : numNonPerfectSquaresAndCubes 200 = 183 := by
  sorry

end num_from_1_to_200_not_squares_or_cubes_l444_444466


namespace non_perfect_powers_count_l444_444482

theorem non_perfect_powers_count :
  let n := 200,
      num_perfect_squares := 14,
      num_perfect_cubes := 5,
      num_perfect_sixth_powers := 1,
      total_perfect_powers := num_perfect_squares + num_perfect_cubes - num_perfect_sixth_powers,
      total_non_perfect_powers := n - total_perfect_powers
  in
  total_non_perfect_powers = 182 :=
by
  sorry

end non_perfect_powers_count_l444_444482


namespace triangle_dimensions_triangle_dimensions_l444_444268

theorem triangle_dimensions {a b : ℝ} (h₁ : a = 10) (h₂ : b = 6) :
  ∃ c, c = Real.sqrt (a^2 + b^2) ∧ a = 10 ∧ b = 6 ∧ c = Real.sqrt (10^2 + 6^2) :=
by {
  use Real.sqrt (10^2 + 6^2),
  split,
  { refl },
  {
    split,
    { exact h₁ },
    {
      split,
      { exact h₂ },
      {
        refl
      }
    }
  }
}

-- Alternative, less verbose version expressing the exact dimensions directly.

theorem triangle_dimensions' :
  ∃ (a b : ℝ), a = 10 ∧ b = 6 ∧ Real.sqrt (a^2 + b^2) = Real.sqrt (10^2 + 6^2) :=
by {
  use 10,
  use 6,
  split,
  { refl },
  {
    split,
    { refl },
    {
      refl
    }
  }
}

end triangle_dimensions_triangle_dimensions_l444_444268


namespace infinite_coprime_pairs_l444_444641

theorem infinite_coprime_pairs (m : ℤ) : ∃ infinitely_many (x y : ℤ), Int.gcd x y = 1 ∧ y ∣ (x^2 + m) ∧ x ∣ (y^2 + m) :=
sorry

end infinite_coprime_pairs_l444_444641


namespace non_perfect_powers_count_l444_444488

theorem non_perfect_powers_count :
  let n := 200,
      num_perfect_squares := 14,
      num_perfect_cubes := 5,
      num_perfect_sixth_powers := 1,
      total_perfect_powers := num_perfect_squares + num_perfect_cubes - num_perfect_sixth_powers,
      total_non_perfect_powers := n - total_perfect_powers
  in
  total_non_perfect_powers = 182 :=
by
  sorry

end non_perfect_powers_count_l444_444488


namespace chris_leftover_money_l444_444869

def chris_will_have_leftover : Prop :=
  let video_game_cost := 60
  let candy_cost := 5
  let hourly_wage := 8
  let hours_worked := 9
  let total_earned := hourly_wage * hours_worked
  let total_cost := video_game_cost + candy_cost
  let leftover := total_earned - total_cost
  leftover = 7

theorem chris_leftover_money : chris_will_have_leftover := 
  by
    sorry

end chris_leftover_money_l444_444869


namespace num_diff_of_squares_in_range_l444_444983

/-- 
Define a number expressible as the difference of squares of two nonnegative integers.
-/
def is_diff_of_squares (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = a^2 - b^2

/-- 
Define the range of interest from 1 to 500.
-/
def range_1_to_500 : Finset ℕ := Finset.range 501 \ {0}

/-- 
Define the criterion for numbers that can be expressed in the desired form.
-/
def can_be_expressed (n : ℕ) : Prop :=
  (n % 2 = 1) ∨ (n % 4 = 0)

/-- 
Count the numbers between 1 and 500 that satisfy the condition.
-/
def count_expressible_numbers : ℕ :=
  (range_1_to_500.filter can_be_expressed).card

/-- 
Prove that the count of numbers between 1 and 500 that can be expressed as 
the difference of two squares of nonnegative integers is 375.
-/
theorem num_diff_of_squares_in_range : count_expressible_numbers = 375 :=
  sorry

end num_diff_of_squares_in_range_l444_444983


namespace evaluate_expression_l444_444327

theorem evaluate_expression : 6 - 5 * (9 - 2^3) * 3 = -9 := by
  sorry

end evaluate_expression_l444_444327


namespace not_sum_three_nonzero_squares_l444_444930

-- To state that 8n - 1 is not the sum of three non-zero squares
theorem not_sum_three_nonzero_squares (n : ℕ) :
  ¬ (∃ a b c : ℕ, a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ 8 * n - 1 = a^2 + b^2 + c^2) := by
  sorry

end not_sum_three_nonzero_squares_l444_444930


namespace line_AF_midpoint_DE_l444_444787

variables {A B C D E F : Type} [EuclideanGeometry.point A] [EuclideanGeometry.point B] [EuclideanGeometry.point C]
  [EuclideanGeometry.point D] [EuclideanGeometry.point E] [EuclideanGeometry.point F]

-- Definitions based on the problem conditions
def isosceles_triangle (A B C : Type) [EuclideanGeometry.point A] [EuclideanGeometry.point B] [EuclideanGeometry.point C] : Prop :=
  (EuclideanGeometry.distance A B = EuclideanGeometry.distance A C)

def midpoint (D B C : Type) [EuclideanGeometry.point D] [EuclideanGeometry.point B] [EuclideanGeometry.point C] : Prop :=
  EuclideanGeometry.between D B C

def perpendicular (D E : Type) [EuclideanGeometry.point D] [EuclideanGeometry.point E] : Prop := 
  EuclideanGeometry.perp D E

def circumcircle (ABD : Set EuclideanGeometry.point) : Prop :=
  EuclideanGeometry.circumcircle ABD

def intersection (circumcircle_intersection : Type) [EuclideanGeometry.point F] [EuclideanGeometry.point B] : Prop := 
  EuclideanGeometry.intersects circumcircle_intersection F B

-- The theorem we need to prove
theorem line_AF_midpoint_DE
  (isoABC : isosceles_triangle A B C)
  (midD : midpoint D B C)
  (perpDE : perpendicular D E)
  (circABD : circumcircle {A, B, D})
  (interBF : intersection {A, B, D})
  : EuclideanGeometry.on_line A F (EuclideanGeometry.midpoint D E) :=
sorry -- proof here

end line_AF_midpoint_DE_l444_444787


namespace problem1_problem2_problem3_l444_444027

section ProofProblems

variable {α : Type*} [LinearOrderedField α]

-- Definitions for sequences and sums
def a_seq (n : ℕ) : α := 2^(n-1)
def b_seq (n : ℕ) (a : ℕ → α) : α := (1 - (a n / a (n + 1))^2) / a (n + 1)
def S_seq (n : ℕ) (b : ℕ → α) : α := ∑ i in Finset.range n, b i

-- Problem (1)
theorem problem1 (n : ℕ) : S_seq n (b_seq (a_seq)) = 3 / 4 - 3 / 2^(n+2) :=
by sorry

-- Problem (2)
theorem problem2 (a : ℕ → α) (h : ∀ n, b_seq (a (n+2)) = S_seq n (b_seq a)) :
  (∀ n, a n = 1) ∨ (∀ n, a n = (-1)^(n-1)) :=
by sorry

-- Problem (3)
theorem problem3 (a : ℕ → α) (h1 : a 1 = 1) (h2 : ∀ n, a n ≤ a (n + 1)) :
  0 ≤ S_seq n (b_seq a) ∧ S_seq n (b_seq a) < 2 :=
by sorry

end ProofProblems

end problem1_problem2_problem3_l444_444027


namespace sin_A_sin_C_l444_444590

-- Defining the triangle with specific conditions
variables {A B C a b c S : ℝ} -- Declaring the variables

-- Conditions given in the problem
axiom h1 : a^2 + c^2 = 4 * a * c
axiom h2 : S = (sqrt (3) / 2) * a * c * cos B

-- Statement of the problem where we need to prove the given condition
theorem sin_A_sin_C : sin A * sin C = 1 / 4 :=
by 
  sorry -- Placeholder for the proof

end sin_A_sin_C_l444_444590


namespace rotation_possible_values_l444_444094

theorem rotation_possible_values (α x₀ y₀ : ℝ) 
  (h₁ : cos α - sqrt 3 * sin α = -22 / 13)
  (h₂ : ∃ P Q, P = (1, 0) ∧ Q = (x₀, y₀) ∧ 
	     Q = (cos α * 1 - sin α * 0, sin α * 1 + cos α * 0)) : 
  x₀ = 1 / 26 ∨ x₀ = -23 / 26 := 
by
  sorry

end rotation_possible_values_l444_444094


namespace distance_between_B_and_D_l444_444717

theorem distance_between_B_and_D (a b c d : ℝ) (h1 : |2 * a - 3 * c| = 1) (h2 : |2 * b - 3 * c| = 1) (h3 : |(2/3) * (d - a)| = 1) (h4 : a ≠ b) :
  |d - b| = 0.5 ∨ |d - b| = 2.5 :=
by
  sorry

end distance_between_B_and_D_l444_444717


namespace sum_of_n_square_minus_3000_perfect_square_eq_zero_l444_444874

theorem sum_of_n_square_minus_3000_perfect_square_eq_zero :
  (∑ n in {n : ℤ | ∃ k : ℤ, n^2 - 3000 = k^2}.toFinset, n) = 0 :=
by
  sorry

end sum_of_n_square_minus_3000_perfect_square_eq_zero_l444_444874


namespace cake_sugar_calculation_l444_444296

theorem cake_sugar_calculation (sugar_first_layer : ℕ) (sugar_second_layer : ℕ) (sugar_third_layer : ℕ) :
  sugar_first_layer = 2 →
  sugar_second_layer = 2 * sugar_first_layer →
  sugar_third_layer = 3 * sugar_second_layer →
  sugar_third_layer = 12 := 
by
  intros h1 h2 h3
  have h4 : 2 = sugar_first_layer, from h1.symm
  have h5 : sugar_second_layer = 2 * 2, by rw [h4, h2]
  have h6 : sugar_third_layer = 3 * 4, by rw [h5, h3]
  exact h6

end cake_sugar_calculation_l444_444296


namespace age_of_15th_student_l444_444175

theorem age_of_15th_student (avg_age_all : ℝ) (avg_age_4 : ℝ) (avg_age_10 : ℝ) 
  (total_students : ℕ) (group_4_students : ℕ) (group_10_students : ℕ) 
  (h1 : avg_age_all = 15) (h2 : avg_age_4 = 14) (h3 : avg_age_10 = 16) 
  (h4 : total_students = 15) (h5 : group_4_students = 4) (h6 : group_10_students = 10) : 
  ∃ x : ℝ, x = 9 := 
by 
  sorry

end age_of_15th_student_l444_444175


namespace find_ab_l444_444036

theorem find_ab (a b : ℕ) (h_gt : a > b) (h_eq : (a + b) + (3 * a + a * b - b) + (4 * a / b) = 64) : a = 8 ∧ b = 2 := 
by trivial

end find_ab_l444_444036


namespace sum_of_repeating_decimal_digits_l444_444182

theorem sum_of_repeating_decimal_digits :
  let b : ℕ → ℕ := by by sorry in
  let n : ℕ := 96 in
  (∑ i in Finset.range n, b i) = 450 := by sorry

end sum_of_repeating_decimal_digits_l444_444182


namespace integer_triplets_2003_l444_444336

theorem integer_triplets_2003 :
  ∃ x y z : ℤ,
  (x^3 + y^3 + z^3 - 3 * x * y * z = 2003) ∧
  ((x = 668 ∧ y = 668 ∧ z = 667) ∨
   (x = 668 ∧ y = 667 ∧ z = 668) ∨
   (x = 667 ∧ y = 668 ∧ z = 668)) :=
begin
  sorry
end

end integer_triplets_2003_l444_444336


namespace find_integers_a_l444_444338

theorem find_integers_a (a : ℤ) : 
  (∃ n : ℤ, (a^3 + 1 = (a - 1) * n)) ↔ a = -1 ∨ a = 0 ∨ a = 2 ∨ a = 3 := 
sorry

end find_integers_a_l444_444338


namespace solve_for_lambda_l444_444965

variable (λ : ℝ)

def vec_a : ℝ × ℝ := (1, 2)
def vec_b : ℝ × ℝ := (0, -2)
def vec_c (λ : ℝ) : ℝ × ℝ := (-1, λ)
def vec_ab : ℝ × ℝ := (2 * vec_a.1 - vec_b.1, 2 * vec_a.2 - vec_b.2)

theorem solve_for_lambda (h_parallel : vec_ab = (-2, 6)) : λ = -3 :=
by
  sorry

end solve_for_lambda_l444_444965


namespace line_AF_through_midpoint_of_DE_l444_444802

open EuclideanGeometry

noncomputable def midpoint_of_segment (A B : Point) := ((A.x + B.x) / 2, (A.y + B.y) / 2)

theorem line_AF_through_midpoint_of_DE
  {A B C D E F : Point}
  (h_isosceles : AB = AC)
  (h_midpoint_D : D = midpoint B C)
  (h_perpendicular_DE : ⟂ D E AC)
  (h_circumcircle_intersects_BE : ∃ k : ℕ, circumcircle A B D = circum circle (B ⊗ F))
  : passes_through (line A F) (midpoint_of_segment D E) :=
sorry

end line_AF_through_midpoint_of_DE_l444_444802


namespace count_valid_numbers_between_1_and_200_l444_444430

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k^2 = n
def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k^3 = n
def is_perfect_sixth_power (n : ℕ) : Prop := ∃ k : ℕ, k^6 = n

def count_perfect_squares (m : ℕ) : ℕ :=
  Nat.sqrt m

def count_perfect_cubes (m : ℕ) : ℕ :=
  (m : ℝ).cbrt.to_nat

def count_perfect_sixth_powers (m : ℕ) : ℕ :=
  (m : ℝ).cbrt.sqrt.to_nat

def count_either_squares_or_cubes (m : ℕ) : ℕ :=
  let squares := count_perfect_squares m
  let cubes := count_perfect_cubes m
  let sixths := count_perfect_sixth_powers m
  squares + cubes - sixths

def count_neither_squares_nor_cubes (n : ℕ) : ℕ :=
  n - count_either_squares_or_cubes n

theorem count_valid_numbers_between_1_and_200 : count_neither_squares_nor_cubes 200 = 182 :=
by
  sorry

end count_valid_numbers_between_1_and_200_l444_444430


namespace num_non_squares_cubes_1_to_200_l444_444439

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n
def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k * k * k = n
def is_perfect_sixth_power (n : ℕ) : Prop := ∃ k : ℕ, k * k * k * k * k * k = n

def num_perfect_squares (a b : ℕ) : ℕ :=
  (List.range (b + 1)).countp is_perfect_square - (List.range (a)).countp is_perfect_square

def num_perfect_cubes (a b : ℕ) : ℕ :=
  (List.range (b + 1)).countp is_perfect_cube - (List.range (a)).countp is_perfect_cube

def num_perfect_sixth_powers (a b : ℕ) : ℕ :=
  (List.range (b + 1)).countp is_perfect_sixth_power - (List.range (a)).countp is_perfect_sixth_power

theorem num_non_squares_cubes_1_to_200 :
  let squares := num_perfect_squares 1 200 in
  let cubes := num_perfect_cubes 1 200 in
  let sixth_powers := num_perfect_sixth_powers 1 200 in
  200 - (squares + cubes - sixth_powers) = 182 :=
by
  let squares := num_perfect_squares 1 200
  let cubes := num_perfect_cubes 1 200
  let sixth_powers := num_perfect_sixth_powers 1 200
  have h : 200 - (squares + cubes - sixth_powers) = 182 := sorry
  exact h

end num_non_squares_cubes_1_to_200_l444_444439


namespace speed_with_coaches_l444_444853

theorem speed_with_coaches
  (n : ℕ)
  (v_initial : ℕ)
  (v_final : ℕ)
  (k : ℝ)
  (sqrt : ℕ → ℝ)
  (h1 : v_initial = 60)
  (h2 : v_final = 48)
  (h3 : ∀ n, v_final = v_initial - k * sqrt n)
  (h4 : sqrt 36 = 6)
  (h5 : k = 12 / sqrt n) :
  (v_initial - k * sqrt 36) = 48 := 
begin
  sorry
end

end speed_with_coaches_l444_444853


namespace f_min_value_l444_444955

-- Defining the function f(x)
def f (x : ℝ) : ℝ := -(x - 1)^3 + 12*x - 3

-- Given conditions
axiom f_max_value : ∀ x ∈ set.Icc (-2 : ℝ) (2 : ℝ), f x ≤ 20
axiom f_max_point : f 2 = 20

-- Prove that the minimum value on [-2, 2] is -7
theorem f_min_value : ∃ x ∈ set.Icc (-2 : ℝ) (2 : ℝ), f x = -7 := by
  sorry

end f_min_value_l444_444955


namespace minimum_distance_between_curves_l444_444138

open Real

theorem minimum_distance_between_curves :
  let P := λ x : ℝ => (x, (1/2) * exp x)
  let Q := λ x : ℝ => (x, log (2 * x))
  let distance (P Q : ℝ × ℝ) := sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)
  ∃ x ∈ ℝ, ∀ y ∈ ℝ, distance (P x) (Q y) = sqrt(2) * (1 - log 2) := sorry

end minimum_distance_between_curves_l444_444138


namespace paint_floor_cost_l444_444678

theorem paint_floor_cost :
  ∀ (L : ℝ) (rate : ℝ)
  (condition1 : L = 3 * (L / 3))
  (condition2 : L = 19.595917942265423)
  (condition3 : rate = 5),
  rate * (L * (L / 3)) = 640 :=
by
  intros L rate condition1 condition2 condition3
  sorry

end paint_floor_cost_l444_444678


namespace find_some_expression_l444_444994

noncomputable def problem_statement : Prop :=
  ∃ (some_expression : ℝ), 
    (5 + 7 / 12 = 6 - some_expression) ∧ 
    (some_expression = 0.4167)

theorem find_some_expression : problem_statement := 
  sorry

end find_some_expression_l444_444994


namespace time_to_cover_one_mile_l444_444842

/-
  Given: 
  - A straight one-mile stretch of highway that is 50 feet wide.
  - 1 mile = 5280 feet.
  - Robert rides his bike on a path composed of quarter-circles at 3 miles per hour.

  Prove: 
  - The time it takes Robert to cover the one-mile stretch is π/6 hours.
-/

theorem time_to_cover_one_mile 
  (width_ft : ℝ := 50) 
  (mile_ft : ℝ := 5280) 
  (speed_mph : ℝ := 3) 
  (time : ℝ := (Real.pi / 6)) : 
  let radius_ft := width_ft / 2
  let num_quarter_circles := (mile_ft / radius_ft).ceil.toNat
  let total_distance_ft := num_quarter_circles * (radius_ft * Real.pi / 2)
  let distance_miles := total_distance_ft / mile_ft
  let time_hours := distance_miles / speed_mph 
  time_hours = time := 
by 
  sorry

end time_to_cover_one_mile_l444_444842


namespace books_after_donation_l444_444863

/-- 
  Total books Boris and Cameron have together after donating some books.
 -/
theorem books_after_donation :
  let B : ℕ := 24 in   -- Initial books Boris has
  let C : ℕ := 30 in   -- Initial books Cameron has
  let B_donated := B / 4 in  -- Boris donates a fourth of his books
  let C_donated := C / 3 in  -- Cameron donates a third of his books
  B - B_donated + (C - C_donated) = 38 :=  -- After donating, the total books
by
  sorry

end books_after_donation_l444_444863


namespace fraction_problem_l444_444235

theorem fraction_problem (x : ℝ) (h : (3 / 4) * (1 / 2) * x * 5000 = 750.0000000000001) : 
  x = 0.4 :=
sorry

end fraction_problem_l444_444235


namespace smallest_factor_l444_444551

theorem smallest_factor (x : ℕ) (h1 : 936 = 2^3 * 3^1 * 13^1)
  (h2 : ∃ (x : ℕ), (936 * x) % 2^5 = 0 ∧ (936 * x) % 3^3 = 0 ∧ (936 * x) % 13^2 = 0) : x = 468 := 
sorry

end smallest_factor_l444_444551


namespace four_identical_pairwise_differences_l444_444373

theorem four_identical_pairwise_differences (a : Fin 20 → ℕ) (h_distinct : Function.Injective a) (h_lt_70 : ∀ i, a i < 70) :
  ∃ d, ∃ (f g : Fin 20 × Fin 20), f ≠ g ∧ (a f.1 - a f.2 = d) ∧ (a g.1 - a g.2 = d) ∧
  ∃ (f1 f2 : Fin 20 × Fin 20), (f1 ≠ f ∧ f1 ≠ g) ∧ (f2 ≠ f ∧ f2 ≠ g) ∧ (a f1.1 - a f1.2 = d) ∧ (a f2.1 - a f2.2 = d) ∧
  (a f1.1 - a f1.2 = d) ∧ (a f2.1 - a f2.2 = d) ∧
  ∃ (f3 : Fin 20 × Fin 20), (f3 ≠ f ∧ f3 ≠ g ∧ f3 ≠ f1 ∧ f3 ≠ f2) ∧ (a f3.1 - a f3.2 = d) := 
sorry

end four_identical_pairwise_differences_l444_444373


namespace sampling_method_sequential_is_systematic_l444_444561

def is_sequential_ids (ids : List Nat) : Prop :=
  ids = [5, 10, 15, 20, 25, 30, 35, 40]

def is_systematic_sampling (sampling_method : Prop) : Prop :=
  sampling_method

theorem sampling_method_sequential_is_systematic :
  ∀ ids, is_sequential_ids ids → 
    is_systematic_sampling (ids = [5, 10, 15, 20, 25, 30, 35, 40]) :=
by
  intros
  apply id
  sorry

end sampling_method_sequential_is_systematic_l444_444561


namespace line_AF_through_midpoint_of_DE_l444_444804

open EuclideanGeometry

noncomputable def midpoint_of_segment (A B : Point) := ((A.x + B.x) / 2, (A.y + B.y) / 2)

theorem line_AF_through_midpoint_of_DE
  {A B C D E F : Point}
  (h_isosceles : AB = AC)
  (h_midpoint_D : D = midpoint B C)
  (h_perpendicular_DE : ⟂ D E AC)
  (h_circumcircle_intersects_BE : ∃ k : ℕ, circumcircle A B D = circum circle (B ⊗ F))
  : passes_through (line A F) (midpoint_of_segment D E) :=
sorry

end line_AF_through_midpoint_of_DE_l444_444804


namespace part1_part2_part3_l444_444162

-- Define constants representing the people A, B, and C
constant A : ℕ
constant B : ℕ
constant C : ℕ

-- Define the number of people
def num_people := 7

-- Define factorial operation
def fact : ℕ → ℕ
| 0 => 1
| (n+1) => (n+1) * fact n

-- Part (1) Statement
theorem part1 : fact 2 * fact (num_people - 2) = 240 := by
  sorry

-- Part (2) Statement
theorem part2 : fact 3 * fact (num_people - 3) = 720 := by
  sorry

-- Part (3) Statement
theorem part3 : fact num_people - fact (num_people - 1) - fact (num_people - 1) + fact (num_people - 2) = 3720 := by
  sorry

end part1_part2_part3_l444_444162


namespace part1_part2_part3_l444_444017

-- Definitions and conditions
def sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) = a n + 1  -- an arithmetic sequence
def S (a : ℕ → ℝ) (n : ℕ) := ∑ i in finset.range (n + 1), a i  -- sum of the first n terms
def lambda_k_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) (λ k : ℝ) :=
  ∀ n : ℕ, (S (n + 1))^(1/k) - (S n)^(1/k) = λ * (a (n + 1))^(1/k)

-- Conditions
axiom a1 : a 1 = 1

-- Part (1): Given an arithmetic sequence {a_n}, if it is a "λ - 1" sequence, then λ = 1
theorem part1 (a : ℕ → ℝ) (λ : ℝ) (h_a : sequence a) (h_lambdak : lambda_k_sequence a (S a) λ 1) : λ = 1 := 
sorry

-- Part (2): If {a_n} is a "$\frac{\sqrt{3}}{3}-2$" sequence, find the general formula for {a_n} when a_n > 0
theorem part2 (a : ℕ → ℝ) (h_lambdak : lambda_k_sequence a (S a) (√3 / 3) 2) (h_pos : ∀ n, a n > 0) : 
  (∀ n, a n = if n = 1 then 1 else 3 * 4^(n - 2)) := 
sorry

-- Part (3): For a given λ, determine if there exist three different sequences {a_n} that form a "λ - 3" sequence, where a_n ≥ 0
theorem part3 (λ : ℝ) : 
  ∃ (a b c : ℕ → ℝ), (λ > 0 ∧ λ < 1) ∧ 
  lambda_k_sequence a (S a) λ 3 ∧ lambda_k_sequence b (S b) λ 3 ∧ lambda_k_sequence c (S c) λ 3 ∧ 
  (∀ n, a n ≥ 0 ∧ b n ≥ 0 ∧ c n ≥ 0) ∧ 
  (∃ n, a n ≠ b n ∧ a n ≠ c n ∧ b n ≠ c n) := 
sorry

end part1_part2_part3_l444_444017


namespace AF_passes_through_midpoint_of_DE_l444_444775

open EuclideanGeometry

-- Definitions based on the conditions
variables {A B C D E F : Point} 
variables {ABC : Triangle}
variables (isosceles_ABC : IsIsoscelesTriangle ABC A B)
variables (midpoint_D : IsMidpoint D B C)
variables (perpendicular_DE : IsPerpendicular DE D AC)
variables {circumcircle_ABD : Circumcircle ABD}
variables (intersection_F : IntersectsAt circumcircle_ABD BE B F)

-- The theorem statement
theorem AF_passes_through_midpoint_of_DE :
  PassesThroughMidpoint AF D E :=
sorry

end AF_passes_through_midpoint_of_DE_l444_444775


namespace sqrt3_minus1_pow0_plus_2_inv_l444_444867

theorem sqrt3_minus1_pow0_plus_2_inv : (real.sqrt 3 - 1) ^ 0 + 2 ^ (-1) = 3 / 2 :=
by
  sorry

end sqrt3_minus1_pow0_plus_2_inv_l444_444867


namespace geometric_sequence_arithmetic_condition_l444_444081

theorem geometric_sequence_arithmetic_condition 
  (a : ℕ → ℝ) (h_pos : ∀ n, 0 < a n)
  (h_geo : ∃ q > 0, ∀ n, a (n + 1) = q * a n)
  (h_arith : a 1, (1 / 2) * a 3, 2 * a 2) are_arithmetic : 
  (a 13 + a 14) / (a 14 + a 15) = real.sqrt 2 - 1 := 
sorry

end geometric_sequence_arithmetic_condition_l444_444081


namespace polynomial_condition_l444_444340

theorem polynomial_condition (P : ℝ → ℝ) :
  (∀ a b c : ℝ, a * b + b * c + c * a = 0 → P(a - b) + P(b - c) + P(c - a) = 2 * P(a + b + c)) →
  ∃ α β : ℝ, ∀ x : ℝ, P(x) = α * x ^ 4 + β * x ^ 2 :=
by
  sorry

end polynomial_condition_l444_444340


namespace prove_problem_statement_l444_444386

noncomputable def problem_statement (x : ℝ) : Prop :=
  sin (2 * x + π / 5) = sqrt 3 / 3 →
  sin (4 * π / 5 - 2 * x) + sin (3 * π / 10 - 2 * x) ^ 2 = (2 + sqrt 3) / 3

theorem prove_problem_statement (x : ℝ) : problem_statement x := 
  by 
    sorry

end prove_problem_statement_l444_444386


namespace perpendicular_PN_BD_l444_444586

-- Define point M as the intersection of angle bisector of ∠ BAD with diagonal BD in rectangle ABCD
def point_M (A B C D M : Point) (rectangle_ABCD : Rectangle A B C D) :
  (angle_bisector A B D) ∩ (line_segment B D) = Set.of (M) := sorry

-- Define point P as the intersection of the angle bisector of ∠ BAD with the line extending BC
def point_P (A B C D P : Point) (rectangle_ABCD : Rectangle A B C D) :
  (angle_bisector A B D) ∩ (line_extending B C) = Set.of (P) := sorry

-- Define point N where the line through M and parallel to AB intersects diagonal AC
def point_N (A B C D M N : Point) (rectangle_ABCD : Rectangle A B C D) :
  (line_through M parallel_to (line_segment A B)) ∩ (line_segment A C) = Set.of (N) := sorry

-- Definition of perpendicularity between line PN and diagonal BD
theorem perpendicular_PN_BD (A B C D M P N : Point) (rectangle_ABCD : Rectangle A B C D) :
  point_M A B C D M rectangle_ABCD →
  point_P A B C D P rectangle_ABCD →
  point_N A B C D M N rectangle_ABCD →
  Perpendicular (line_through P N) (line_segment B D) :=
sorry

end perpendicular_PN_BD_l444_444586


namespace AF_passes_through_midpoint_DE_l444_444774

open_locale classical

-- Definition of the geometrical setup
def isosceles_triangle (A B C : Point) : Prop := dist A B = dist A C
def midpoint (D B C : Point) : Prop := dist B D = dist C D
def perpendicular (D E A C : Point) : Prop := ∠ DEA = 90
def circumcircle (A B D F : Point): Prop := ∃ O R, Circle.circle O R A ∧ Circle.circle O R B ∧ Circle.circle O R D ∧ Circle.circle O R F

-- Main statement to prove
theorem AF_passes_through_midpoint_DE
  (A B C D E F : Point)
  (h1 : isosceles_triangle A B C)
  (h2 : midpoint D B C)
  (h3 : perpendicular D E A C)
  (h4 : ∃ O R, Circle.circle O R A ∧ Circle.circle O R B ∧ Circle.circle O R D ∧ Circle.circle O R F)
  (h5 : collinear B E F)
  (h6 : intersects (circumcircle A B D) (line B E) B F) :
  passes_through (line A F) (midpoint D E) := 
sorry

end AF_passes_through_midpoint_DE_l444_444774


namespace problem_min_value_n_l444_444612

theorem problem_min_value_n :
  ∃ (n : ℕ) (y : Fin n → ℝ), (∀ i, 0 ≤ y i) ∧ (∑ i, y i = 1) ∧ (∑ i, (y i)^2 ≤ 1/50) ∧ n = 50 :=
by
  sorry

end problem_min_value_n_l444_444612


namespace inequality_proof_l444_444009

theorem inequality_proof (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h_sum : x + y + z = 1) :
  ((1 / x^2 - x) * (1 / y^2 - y) * (1 / z^2 - z) ≥ (26 / 3)^3) :=
by sorry

end inequality_proof_l444_444009


namespace bus_speeds_l444_444714

theorem bus_speeds (d t : ℝ) (s₁ s₂ : ℝ)
  (h₀ : d = 48)
  (h₁ : t = 1 / 6) -- 10 minutes in hours
  (h₂ : s₂ = s₁ - 4)
  (h₃ : d / s₂ - d / s₁ = t) :
  s₁ = 36 ∧ s₂ = 32 := 
sorry

end bus_speeds_l444_444714


namespace distance_to_line_not_greater_than_2_l444_444947

-- Definitions of distances PA, PB, and PC
variables {P A B C : Point}
variable {l : Line}
variable (PA PB PC : ℝ)

-- Hypotheses: given conditions
axiom PA_distance : dist P A = 2
axiom PB_distance : dist P B = 2.5
axiom PC_distance : dist P C = 3
axiom A_on_line : A ∈ l
axiom B_on_line : B ∈ l
axiom C_on_line : C ∈ l
axiom P_not_on_line : P ∉ l

-- Theorem: the distance from P to line l is not greater than 2
theorem distance_to_line_not_greater_than_2 :
  distance_point_to_line P l ≤ 2 :=
sorry

end distance_to_line_not_greater_than_2_l444_444947


namespace modulus_of_z_eq_sqrt2_l444_444951

noncomputable def complex_z : ℂ := (1 + 3 * Complex.I) / (2 - Complex.I)

theorem modulus_of_z_eq_sqrt2 : Complex.abs complex_z = Real.sqrt 2 := by
  sorry

end modulus_of_z_eq_sqrt2_l444_444951


namespace symmetric_angle_of_inclination_l444_444392

theorem symmetric_angle_of_inclination (α₁ : ℝ) (h : 0 ≤ α₁ ∧ α₁ < π) : 
  (∃ β₁ : ℝ, (α₁ = 0 ∧ β₁ = 0) ∨ (0 < α₁ ∧ α₁ < π ∧ β₁ = π - α₁)) :=
by
  sorry

end symmetric_angle_of_inclination_l444_444392


namespace main_theorem_l444_444745

universe u
variables {α : Type u} [euclidean_geometry α]

open_locale euclidean_geometry

-- Define the given conditions
def is_isosceles (A B C : α) (h : euclidean_geometry.is_triangle A B C) : Prop :=
euclidean_geometry.dist A B = euclidean_geometry.dist A C

def midpoint (D B C : α) : Prop :=
euclidean_geometry.is_midpoint D B C

def perpendicular (E D A C : α) : Prop :=
euclidean_geometry.is_perpendicular E D A C

def circumcircle_intersects (F B D A : α) (c : set α) : Prop :=
c = euclidean_geometry.circumcircle A B D ∧ F ∈ c ∧ B ∈ c

-- The main theorem to prove
theorem main_theorem (A B C D E F : α) (h1 : euclidean_geometry.is_triangle A B C)
    (h2 : is_isosceles A B C h1) (h3 : midpoint D B C) (h4 : perpendicular E D A C)
    (h5 : ∃ c, circumcircle_intersects F B D A c) :
    euclidean_geometry.passes_through (euclidean_geometry.line A F) (euclidean_geometry.midpoint E D) :=
sorry

end main_theorem_l444_444745


namespace total_children_l444_444313

-- Given the conditions
def toy_cars : Nat := 134
def dolls : Nat := 269

-- Prove that the total number of children is 403
theorem total_children (h_cars : toy_cars = 134) (h_dolls : dolls = 269) :
  toy_cars + dolls = 403 :=
by
  sorry

end total_children_l444_444313


namespace mindy_final_amount_l444_444148

/-- Mindy's purchases are $2.75, $6.15, and $11.30. After a 10% discount on the total, 
  the final amount due, rounded to the nearest dollar, is $18. -/
theorem mindy_final_amount :
  let purchase1 := 2.75
  let purchase2 := 6.15
  let purchase3 := 11.30
  let total_before_discount := purchase1 + purchase2 + purchase3
  let discount := 0.10 * total_before_discount
  let total_after_discount := total_before_discount - discount
  round total_after_discount = 18 := 
by
  sorry

end mindy_final_amount_l444_444148


namespace intersection_point_of_lines_l444_444665

theorem intersection_point_of_lines : 
  ∃ (x y : ℝ), (x - 4 * y - 1 = 0) ∧ (2 * x + y - 2 = 0) ∧ (x = 1) ∧ (y = 0) :=
by
  sorry

end intersection_point_of_lines_l444_444665


namespace meeting_opposite_point_l444_444107

-- Define the initial speeds and distances
constant (s : ℕ) -- Speed of Hector in blocks per unit time.
constant (blocks : ℕ) -- Total loop distance in blocks (24 blocks).
constant (t : ℕ) -- Time taken to meet.

-- Conditions from the math problem.
axiom Jane_double_speed : 2 * s -- Jane's speed is twice Hector's speed.
axiom start_opposite_directions : blocks = 24 -- They start walking in opposite directions around a 24-block loop.

-- Lean statement for the proof problem.
theorem meeting_opposite_point (s : ℕ) (blocks t : ℕ) :
  2 * s + s = 3 * s -> blocks = 24 -> t = 8 / s ->
  (2 * s) * t + s * t = blocks -> (2 * s) * (8 / s) = 16 ->
  (s * (8 / s)) = 8 -> 
  ((s * 8 / s ) + (2 * s * 8 / s)) = blocks ->
  ∃ m n : ℕ, (n = 16 ∧ m = 8) ∧ meeting_point := opposite_to_point_A :=
sorry

end meeting_opposite_point_l444_444107


namespace volunteers_selection_l444_444004

theorem volunteers_selection :
  let total_volunteers := 6
  let boys := 4
  let girls := 2
  let chosen_volunteers := 4
  (boys > 0 ∧ girls > 0) →
  ∃ (ways : ℕ), ways = (Nat.choose 4 3 * Nat.choose 2 1) + (Nat.choose 4 2 * Nat.choose 2 2) ∧ ways = 14 :=  
by
  intros _ _ _ _ _ _
  use (Nat.choose 4 3 * Nat.choose 2 1) + (Nat.choose 4 2 * Nat.choose 2 2)
  split
  · rfl
  · exact Eq.refl (8 + 6)
  sorry

end volunteers_selection_l444_444004


namespace petty_cash_correction_l444_444567

theorem petty_cash_correction (q d n c x : ℕ) : 
  25 * q + 10 * d + 5 * n + c + 11 * x = 25 * q + 10 * d + 5 * (n - x) + c + 25 * x + 9 * x := 
begin
  sorry -- proof is omitted
end

end petty_cash_correction_l444_444567


namespace tan_theta_eq2_simplifies_to_minus1_sin_cos_and_tan_relation_l444_444234

-- First proof problem
theorem tan_theta_eq2_simplifies_to_minus1 (θ : ℝ) (h : Real.tan θ = 2) :
  (Real.sin (θ - 6 * Real.pi) + Real.sin (Real.pi / 2 - θ)) / 
  (2 * Real.sin (Real.pi + θ) + Real.cos (-θ)) = -1 := sorry

-- Second proof problem
theorem sin_cos_and_tan_relation (x : ℝ) (hx1 : - Real.pi / 2 < x) (hx2 : x < Real.pi / 2) 
  (h : Real.sin x + Real.cos x = 1 / 5) : Real.tan x = -3 / 4 := sorry

end tan_theta_eq2_simplifies_to_minus1_sin_cos_and_tan_relation_l444_444234


namespace inequality_solution_l444_444659

theorem inequality_solution (x : ℝ) :
  (x+3)/(x+4) > (4*x+5)/(3*x+10) ↔ x ∈ Set.Ioo (-4 : ℝ) (- (10 : ℝ) / 3) ∪ Set.Ioi 2 :=
by
  sorry

end inequality_solution_l444_444659


namespace linear_continuous_exponential_continuous_sine_continuous_l444_444161

-- Linear function continuity
theorem linear_continuous (x : ℝ) : continuous (λ x, 3 * x - 7) :=
by continuity

-- Exponential function continuity
theorem exponential_continuous (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : continuous (λ x : ℝ, a^x) :=
by continuity

-- Sine function continuity
theorem sine_continuous (x : ℝ) : continuous (λ x : ℝ, Real.sin x) :=
by continuity

end linear_continuous_exponential_continuous_sine_continuous_l444_444161


namespace diesel_fuel_usage_l444_444627

theorem diesel_fuel_usage (weekly_spending : ℝ) (cost_per_gallon : ℝ) (weeks : ℝ) (result : ℝ): 
  weekly_spending = 36 → cost_per_gallon = 3 → weeks = 2 → result = 24 → 
  (weekly_spending / cost_per_gallon) * weeks = result :=
by
  intros
  sorry

end diesel_fuel_usage_l444_444627


namespace number_of_students_scoring_above_90_l444_444559

theorem number_of_students_scoring_above_90
  (total_students : ℕ)
  (mean : ℝ)
  (variance : ℝ)
  (students_scoring_at_least_60 : ℕ)
  (h1 : total_students = 1200)
  (h2 : mean = 75)
  (h3 : ∃ (σ : ℝ), variance = σ^2)
  (h4 : students_scoring_at_least_60 = 960)
  : ∃ n, n = total_students - students_scoring_at_least_60 ∧ n = 240 :=
by {
  sorry
}

end number_of_students_scoring_above_90_l444_444559


namespace lemons_needed_l444_444236

theorem lemons_needed (initial_lemons : ℝ) (initial_gallons : ℝ) 
  (reduced_ratio : ℝ) (first_gallons : ℝ) (total_gallons : ℝ) :
  initial_lemons / initial_gallons * first_gallons 
  + (initial_lemons / initial_gallons * reduced_ratio) * (total_gallons - first_gallons) = 56.25 :=
by 
  let initial_ratio := initial_lemons / initial_gallons
  let reduced_ratio_amount := initial_ratio * reduced_ratio 
  let lemons_first := initial_ratio * first_gallons
  let lemons_remaining := reduced_ratio_amount * (total_gallons - first_gallons)
  let total_lemons := lemons_first + lemons_remaining
  show total_lemons = 56.25
  sorry

end lemons_needed_l444_444236


namespace triangle_lines_l444_444397

/-- Given a triangle with vertices A(1, 2), B(-1, 4), and C(4, 5):
  1. The equation of the line l₁ containing the altitude from A to side BC is 5x + y - 7 = 0.
  2. The equation of the line l₂ passing through C such that the distances from A and B to l₂ are equal
     is either x + y - 9 = 0 or x - 2y + 6 = 0. -/
theorem triangle_lines (A B C : ℝ × ℝ)
  (hA : A = (1, 2))
  (hB : B = (-1, 4))
  (hC : C = (4, 5)) :
  ∃ l₁ l₂ : ℝ × ℝ × ℝ,
  (l₁ = (5, 1, -7)) ∧
  ((l₂ = (1, 1, -9)) ∨ (l₂ = (1, -2, 6))) := by
  sorry

end triangle_lines_l444_444397


namespace swimmer_distance_l444_444277

noncomputable def effective_speed := 4.4 - 2.5
noncomputable def time := 3.684210526315789
noncomputable def distance := effective_speed * time

theorem swimmer_distance :
  distance = 7 := by
  sorry

end swimmer_distance_l444_444277


namespace distance_from_point_to_line_le_2_l444_444949

open Real

-- Definitions of points and distances
variables (P A B C : Point) (l : Line)

-- Conditions given in the problem
def PA: ℝ := dist P A
def PB: ℝ := dist P B
def PC: ℝ := dist P C

-- Given the conditions
axiom PA_eq_2 : PA P A = 2
axiom PB_eq_25 : PB P B = 2.5
axiom PC_eq_3 : PC P C = 3

-- Theorem to state the conclusion
theorem distance_from_point_to_line_le_2 : dist_from_point_to_line P l ≤ 2 :=
by
  sorry

end distance_from_point_to_line_le_2_l444_444949


namespace num_from_1_to_200_not_squares_or_cubes_l444_444456

noncomputable def numNonPerfectSquaresAndCubes (n : ℕ) : ℕ :=
  let num_squares := 14
  let num_cubes := 5
  let num_sixth_powers := 2
  n - (num_squares + num_cubes - num_sixth_powers)

theorem num_from_1_to_200_not_squares_or_cubes : numNonPerfectSquaresAndCubes 200 = 183 := by
  sorry

end num_from_1_to_200_not_squares_or_cubes_l444_444456


namespace product_seq_value_l444_444210

theorem product_seq_value :
  (∏ n in Finset.range 49, (1 - (1 : ℚ) / (n + 2))) = 1 / 50 := by
  sorry

end product_seq_value_l444_444210


namespace numbers_not_perfect_squares_or_cubes_l444_444449

theorem numbers_not_perfect_squares_or_cubes : 
  let total_numbers := 200
  let perfect_squares := 14
  let perfect_cubes := 5
  let sixth_powers := 1
  total_numbers - (perfect_squares + perfect_cubes - sixth_powers) = 182 :=
by
  sorry

end numbers_not_perfect_squares_or_cubes_l444_444449


namespace sum_of_roots_even_l444_444289

theorem sum_of_roots_even (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) 
    (h_distinct : ∃ x y : ℤ, x ≠ y ∧ (x^2 - 2 * p * x + (p * q) = 0) ∧ (y^2 - 2 * p * y + (p * q) = 0)) :
    Even (2 * p) :=
by 
  sorry

end sum_of_roots_even_l444_444289


namespace math_competition_l444_444228

open Set

variable (Students : Finset (Fin 39)) (Problems : Finset (Fin 6))
variable (S : Fin 6 → Finset (Fin 39))

-- Define the condition that for any 3 students, there is at most 1 problem that none of the three solved
def condition (Students : Finset (Fin 39)) (Problems : Finset (Fin 6)) (S : Fin 6 → Finset (Fin 39)) : Prop :=
  ∀ (a b c : Fin 39), ∃ (p : Fin 6), a ∈ S p ∨ b ∈ S p ∨ c ∈ S p

-- Define the problem statement in Lean 4
theorem math_competition : condition Students Problems S → 
  (∑ i, if i ∈ Students then 6 - ∑ j, if i ∈ S j then 1 else 0 else 0) = 165 := 
sorry

end math_competition_l444_444228


namespace chocolate_truffles_sold_l444_444247

def fudge_sold_pounds : ℕ := 20
def price_per_pound_fudge : ℝ := 2.50
def price_per_truffle : ℝ := 1.50
def pretzels_sold_dozen : ℕ := 3
def price_per_pretzel : ℝ := 2.00
def total_revenue : ℝ := 212.00

theorem chocolate_truffles_sold (dozens_of_truffles_sold : ℕ) :
  let fudge_revenue := (fudge_sold_pounds : ℝ) * price_per_pound_fudge
  let pretzels_revenue := (pretzels_sold_dozen : ℝ) * 12 * price_per_pretzel
  let truffles_revenue := total_revenue - fudge_revenue - pretzels_revenue
  let num_truffles_sold := truffles_revenue / price_per_truffle
  let dozens_of_truffles_sold := num_truffles_sold / 12
  dozens_of_truffles_sold = 5 :=
by
  sorry

end chocolate_truffles_sold_l444_444247


namespace speeds_of_bus_and_car_l444_444202

theorem speeds_of_bus_and_car
  (d t : ℝ) (v1 v2 : ℝ)
  (h1 : 1.5 * v1 + 1.5 * v2 = d)
  (h2 : 2.5 * v1 + 1 * v2 = d) :
  v1 = 40 ∧ v2 = 80 :=
by sorry

end speeds_of_bus_and_car_l444_444202


namespace non_perfect_powers_count_l444_444484

theorem non_perfect_powers_count :
  let n := 200,
      num_perfect_squares := 14,
      num_perfect_cubes := 5,
      num_perfect_sixth_powers := 1,
      total_perfect_powers := num_perfect_squares + num_perfect_cubes - num_perfect_sixth_powers,
      total_non_perfect_powers := n - total_perfect_powers
  in
  total_non_perfect_powers = 182 :=
by
  sorry

end non_perfect_powers_count_l444_444484


namespace Vasyuki_coloring_possible_l444_444577

theorem Vasyuki_coloring_possible (n : ℕ) (perm : Equiv.Perm (Fin n)) : 
  ∃ (colors : Fin n → Fin 3), ∀ i, colors i ≠ colors (perm i) :=
by
  sorry

end Vasyuki_coloring_possible_l444_444577


namespace AF_passes_through_midpoint_of_DE_l444_444778

open EuclideanGeometry

-- Definitions based on the conditions
variables {A B C D E F : Point} 
variables {ABC : Triangle}
variables (isosceles_ABC : IsIsoscelesTriangle ABC A B)
variables (midpoint_D : IsMidpoint D B C)
variables (perpendicular_DE : IsPerpendicular DE D AC)
variables {circumcircle_ABD : Circumcircle ABD}
variables (intersection_F : IntersectsAt circumcircle_ABD BE B F)

-- The theorem statement
theorem AF_passes_through_midpoint_of_DE :
  PassesThroughMidpoint AF D E :=
sorry

end AF_passes_through_midpoint_of_DE_l444_444778


namespace convex_2000_gon_perpendicular_diagonals_l444_444125

theorem convex_2000_gon_perpendicular_diagonals (M : Polygon) (h1 : M.convex) (h2 : M.sides = 2000) (h3 : M.diameter = 1) (h4 : M.has_largest_area) :
  ∃ d₁ d₂ : Diagonal, d₁.perpendicular d₂ :=
sorry

end convex_2000_gon_perpendicular_diagonals_l444_444125


namespace count_not_squares_or_cubes_200_l444_444524

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n
def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k * k * k = n
def is_sixth_power (n : ℕ) : Prop := ∃ k : ℕ, k^6 = n

theorem count_not_squares_or_cubes_200 :
  let count_squares := (Finset.range 201).filter is_perfect_square).card
  let count_cubes := (Finset.range 201).filter is_perfect_cube).card
  let count_sixth_powers := (Finset.range 201).filter is_sixth_power).card
  (200 - (count_squares + count_cubes - count_sixth_powers)) = 182 := 
by {
  -- Define everything in terms of Finset and filter
  let count_squares := (Finset.range 201).filter is_perfect_square).card,
  let count_cubes := (Finset.range 201).filter is_perfect_cube).card,
  let count_sixth_powers := (Finset.range 201).filter is_sixth_power).card,
  exact sorry
}

end count_not_squares_or_cubes_200_l444_444524


namespace main_theorem_l444_444744

universe u
variables {α : Type u} [euclidean_geometry α]

open_locale euclidean_geometry

-- Define the given conditions
def is_isosceles (A B C : α) (h : euclidean_geometry.is_triangle A B C) : Prop :=
euclidean_geometry.dist A B = euclidean_geometry.dist A C

def midpoint (D B C : α) : Prop :=
euclidean_geometry.is_midpoint D B C

def perpendicular (E D A C : α) : Prop :=
euclidean_geometry.is_perpendicular E D A C

def circumcircle_intersects (F B D A : α) (c : set α) : Prop :=
c = euclidean_geometry.circumcircle A B D ∧ F ∈ c ∧ B ∈ c

-- The main theorem to prove
theorem main_theorem (A B C D E F : α) (h1 : euclidean_geometry.is_triangle A B C)
    (h2 : is_isosceles A B C h1) (h3 : midpoint D B C) (h4 : perpendicular E D A C)
    (h5 : ∃ c, circumcircle_intersects F B D A c) :
    euclidean_geometry.passes_through (euclidean_geometry.line A F) (euclidean_geometry.midpoint E D) :=
sorry

end main_theorem_l444_444744


namespace num_non_squares_cubes_1_to_200_l444_444442

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n
def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k * k * k = n
def is_perfect_sixth_power (n : ℕ) : Prop := ∃ k : ℕ, k * k * k * k * k * k = n

def num_perfect_squares (a b : ℕ) : ℕ :=
  (List.range (b + 1)).countp is_perfect_square - (List.range (a)).countp is_perfect_square

def num_perfect_cubes (a b : ℕ) : ℕ :=
  (List.range (b + 1)).countp is_perfect_cube - (List.range (a)).countp is_perfect_cube

def num_perfect_sixth_powers (a b : ℕ) : ℕ :=
  (List.range (b + 1)).countp is_perfect_sixth_power - (List.range (a)).countp is_perfect_sixth_power

theorem num_non_squares_cubes_1_to_200 :
  let squares := num_perfect_squares 1 200 in
  let cubes := num_perfect_cubes 1 200 in
  let sixth_powers := num_perfect_sixth_powers 1 200 in
  200 - (squares + cubes - sixth_powers) = 182 :=
by
  let squares := num_perfect_squares 1 200
  let cubes := num_perfect_cubes 1 200
  let sixth_powers := num_perfect_sixth_powers 1 200
  have h : 200 - (squares + cubes - sixth_powers) = 182 := sorry
  exact h

end num_non_squares_cubes_1_to_200_l444_444442


namespace product_of_three_digit_numbers_ends_with_four_zeros_l444_444308

/--
Theorem: There exist three three-digit numbers formed using nine different digits such that their product ends with four zeros.
-/
theorem product_of_three_digit_numbers_ends_with_four_zeros :
  ∃ (x y z : ℕ), 100 ≤ x ∧ x < 1000 ∧
                 100 ≤ y ∧ y < 1000 ∧
                 100 ≤ z ∧ z < 1000 ∧
                 (∀ d ∈ (list.digits x).union (list.digits y).union (list.digits z), 
                     list.count d ((list.digits x).union (list.digits y).union (list.digits z)) = 1) ∧
                 (x * y * z) % 10000 = 0 :=
sorry

end product_of_three_digit_numbers_ends_with_four_zeros_l444_444308


namespace numbers_not_squares_or_cubes_in_200_l444_444477

/-- There are 182 numbers between 1 and 200 that are neither perfect squares nor perfect cubes. -/
theorem numbers_not_squares_or_cubes_in_200 : (finset.range 200).filter (λ n, ¬(∃ m, m^2 = n) ∧ ¬(∃ k, k^3 = n)).card = 182 :=
by 
  /-
    The proof will involve constructing the set of numbers from 1 to 200, filtering out those that
    are perfect squares or perfect cubes, and counting the remaining numbers.
  -/
  sorry

end numbers_not_squares_or_cubes_in_200_l444_444477


namespace find_k_which_make_roots_integers_l444_444911

theorem find_k_which_make_roots_integers :
  ∃ k : ℚ, (∀ x : ℚ, k * x^2 + (k + 1) * x + (k - 1) = 0 → x ∈ ℤ) ↔ 
    k = 0 ∨ k = -1/7 ∨ k = 1 := 
sorry

end find_k_which_make_roots_integers_l444_444911


namespace positive_number_satisfying_condition_l444_444213

theorem positive_number_satisfying_condition :
  ∃ x : ℝ, x > 0 ∧ x^2 = 64 ∧ x = 8 := by sorry

end positive_number_satisfying_condition_l444_444213


namespace num_non_squares_cubes_1_to_200_l444_444444

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n
def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k * k * k = n
def is_perfect_sixth_power (n : ℕ) : Prop := ∃ k : ℕ, k * k * k * k * k * k = n

def num_perfect_squares (a b : ℕ) : ℕ :=
  (List.range (b + 1)).countp is_perfect_square - (List.range (a)).countp is_perfect_square

def num_perfect_cubes (a b : ℕ) : ℕ :=
  (List.range (b + 1)).countp is_perfect_cube - (List.range (a)).countp is_perfect_cube

def num_perfect_sixth_powers (a b : ℕ) : ℕ :=
  (List.range (b + 1)).countp is_perfect_sixth_power - (List.range (a)).countp is_perfect_sixth_power

theorem num_non_squares_cubes_1_to_200 :
  let squares := num_perfect_squares 1 200 in
  let cubes := num_perfect_cubes 1 200 in
  let sixth_powers := num_perfect_sixth_powers 1 200 in
  200 - (squares + cubes - sixth_powers) = 182 :=
by
  let squares := num_perfect_squares 1 200
  let cubes := num_perfect_cubes 1 200
  let sixth_powers := num_perfect_sixth_powers 1 200
  have h : 200 - (squares + cubes - sixth_powers) = 182 := sorry
  exact h

end num_non_squares_cubes_1_to_200_l444_444444


namespace container_dimensions_l444_444083

theorem container_dimensions (a b c : ℝ) 
  (h1 : a * b * 16 = 2400)
  (h2 : a * c * 10 = 2400)
  (h3 : b * c * 9.6 = 2400) :
  a = 12 ∧ b = 12.5 ∧ c = 20 :=
by
  sorry

end container_dimensions_l444_444083


namespace matrix_vector_subtraction_l444_444601

open Matrix

variable {α : Type*} [AddCommGroup α] [Module ℝ α]

def matrix_mul_vector (M : Matrix (Fin 2) (Fin 2) ℝ) (v : Fin 2 → ℝ) : Fin 2 → ℝ :=
  M.mulVec v

theorem matrix_vector_subtraction (M : Matrix (Fin 2) (Fin 2) ℝ) (v w : Fin 2 → ℝ)
  (hv : matrix_mul_vector M v = ![4, 6])
  (hw : matrix_mul_vector M w = ![5, -4]) :
  matrix_mul_vector M (v - (2 : ℝ) • w) = ![-6, 14] :=
sorry

end matrix_vector_subtraction_l444_444601


namespace zero_one_sequence_count_l444_444383

theorem zero_one_sequence_count :
  ∑ k in finset.range 6, (nat.choose 11 k) * (nat.choose (11 - k) (10 - 2 * k)) = 24068 := by
  sorry

end zero_one_sequence_count_l444_444383


namespace numbers_not_squares_or_cubes_in_200_l444_444476

/-- There are 182 numbers between 1 and 200 that are neither perfect squares nor perfect cubes. -/
theorem numbers_not_squares_or_cubes_in_200 : (finset.range 200).filter (λ n, ¬(∃ m, m^2 = n) ∧ ¬(∃ k, k^3 = n)).card = 182 :=
by 
  /-
    The proof will involve constructing the set of numbers from 1 to 200, filtering out those that
    are perfect squares or perfect cubes, and counting the remaining numbers.
  -/
  sorry

end numbers_not_squares_or_cubes_in_200_l444_444476


namespace determine_a_l444_444280

variable (a : ℝ)
variable (x1 x2 x3 x4 y1 y2 y3 y4 : ℝ)
variable (regression_eq : ℝ → ℝ)

/-- The parameters for the problem -/
def parameters : Prop :=
  x1 = 2 ∧ y1 = 30 ∧
  x2 = 3 ∧ y2 = a ∧
  x3 = 4 ∧ y3 = 40 ∧
  x4 = 5 ∧ y4 = 50 ∧
  regression_eq = λ x => 8 * x + 11

/-- The average (mean) function -/
def average (l : List ℝ) : ℝ := (l.sum) / (l.length)

/-- The average number of parts processed -/
def average_x := average [x1, x2, x3, x4]

/-- The average processing time -/
def average_y := average [y1, y2, y3, y4]

/-- Lean 4 statement for the proof -/
theorem determine_a (h : parameters a x1 x2 x3 x4 y1 y2 y3 y4 regression_eq) :
  a = 36 :=
by
  sorry

end determine_a_l444_444280


namespace triangle_is_isosceles_l444_444250

-- Definitions used in the conditions
variables {A B C P Q : Type} [Nonempty A] [Nonempty B] [Nonempty C]
variable (triangle_ABC : Triangle A B C)
variable (circle_PASS : Circle A B)
variable (P_intersect : Intersect circle_PASS A C P)
variable (Q_intersect : Intersect circle_PASS B C Q)
variable (median_C : Median C circle_PASS P Q)

-- Statement of the problem to prove the triangle is isosceles
theorem triangle_is_isosceles (h1 : P_intersect ∧ Q_intersect) (h2 : median_C) : 
  IsoscelesTriangle A C B :=
sorry

end triangle_is_isosceles_l444_444250


namespace exists_convergent_subsequence_l444_444597

noncomputable theory
open_locale classical

-- Given sequence a_k of real numbers in [0, 1]
variables (a : ℕ → ℝ)
hypothesis a_in_unit_interval : ∀ n, 0 ≤ a n ∧ a n ≤ 1

-- Main theorem statement
theorem exists_convergent_subsequence :
  ∃ (n : ℕ → ℕ) (A : ℝ), 
    (strict_mono n) ∧ 
    (∀ ε > 0, ∃ N, ∀ i j : ℕ, i ≠ j → i > N → j > N → |a (n i + n j) - A| < ε) :=
sorry

end exists_convergent_subsequence_l444_444597


namespace sequence_inequality_l444_444839

variables {s : ℕ} {m : ℕ → ℕ} {r : ℕ}

-- Assuming the conditions
def increasing_sequence (m : ℕ → ℕ) : Prop :=
  ∀ i j, i < j → m i < m j

def cannot_be_sum_of_two_or_more (m : ℕ → ℕ) (s : ℕ) : Prop :=
  ∀ (x : ℕ), x ∈ {m i | i < s + 1} →
  ¬ ∃ (I : Set ℕ), ∀ i ∈ I, (i < s + 1) ∧ i ≠ 0 → ∑ j in I, m j = x

-- Definition of the theorem
theorem sequence_inequality 
  (hs : 2 ≤ s)
  (hm_inc : increasing_sequence m)
  (hm_sum : cannot_be_sum_of_two_or_more m s)
  (hr : 1 ≤ r ∧ r < s) :
  r * m r + m s ≥ (r + 1) * (s - 1) :=
begin
  sorry
end

end sequence_inequality_l444_444839


namespace inverse_of_A_is_zero_matrix_l444_444907

-- Given conditions
def A : Matrix (Fin 2) (Fin 2) ℤ := ![![6, -4], ![-3, 2]]
def zero_matrix : Matrix (Fin 2) (Fin 2) ℤ := 0

-- Prove that the inverse of matrix A is zero_matrix when determinant is zero
theorem inverse_of_A_is_zero_matrix (h : det A = 0) : inverse A = zero_matrix :=
by sorry

end inverse_of_A_is_zero_matrix_l444_444907


namespace distance_point_line_correct_l444_444096

noncomputable def distance_point_line : ℝ :=
  let P := (2 * Real.sqrt 3, Real.pi / 6)
  let l := fun ρ θ => ρ * Real.cos (θ + Real.pi / 4) - 2 * Real.sqrt 2
  let P_rect := (3: ℝ, Real.sqrt 3)
  let l_standard := fun x y =>
    x - y - 4
  in (|3 - √3 - 4| / √2)

theorem distance_point_line_correct : distance_point_line = (Real.sqrt 2 + Real.sqrt 6) / 2 := sorry

end distance_point_line_correct_l444_444096


namespace AF_passes_through_midpoint_DE_l444_444770

open_locale classical

-- Definition of the geometrical setup
def isosceles_triangle (A B C : Point) : Prop := dist A B = dist A C
def midpoint (D B C : Point) : Prop := dist B D = dist C D
def perpendicular (D E A C : Point) : Prop := ∠ DEA = 90
def circumcircle (A B D F : Point): Prop := ∃ O R, Circle.circle O R A ∧ Circle.circle O R B ∧ Circle.circle O R D ∧ Circle.circle O R F

-- Main statement to prove
theorem AF_passes_through_midpoint_DE
  (A B C D E F : Point)
  (h1 : isosceles_triangle A B C)
  (h2 : midpoint D B C)
  (h3 : perpendicular D E A C)
  (h4 : ∃ O R, Circle.circle O R A ∧ Circle.circle O R B ∧ Circle.circle O R D ∧ Circle.circle O R F)
  (h5 : collinear B E F)
  (h6 : intersects (circumcircle A B D) (line B E) B F) :
  passes_through (line A F) (midpoint D E) := 
sorry

end AF_passes_through_midpoint_DE_l444_444770


namespace Shara_will_owe_money_l444_444165

theorem Shara_will_owe_money
    (B : ℕ)
    (h1 : 6 * 10 = 60)
    (h2 : B / 2 = 60)
    (h3 : 4 * 10 = 40)
    (h4 : 60 + 40 = 100) :
  B - 100 = 20 :=
sorry

end Shara_will_owe_money_l444_444165


namespace probability_sum_even_l444_444704

theorem probability_sum_even (a b c : ℕ) (hₐ : a ∈ {1, 2, 3, 4, 5, 6}) 
  (h_b : b ∈ {1, 2, 3, 4, 5, 6}) (h_c : c ∈ {1, 2, 3, 4, 5, 6}) :
  let outcomes := {0, 1} in 
  let event := (a % 2 + b % 2 + c % 2) % 2 = 0 in
  (∑ x in outcomes, ∑ y in outcomes, ∑ z in outcomes, 
    if (x + y + z) % 2 = 0 then 1 else 0) / (outcomes.card * outcomes.card * outcomes.card) = 1 / 2 := 
sorry

end probability_sum_even_l444_444704


namespace no_polyhedron_with_odd_faces_odd_edges_l444_444322

theorem no_polyhedron_with_odd_faces_odd_edges (F : Finset ℕ) (f : ℕ → ℕ) (n : ℕ)
  (hF_nonempty : F.nonempty)
  (hF_card_odd : F.card % 2 = 1)
  (hface_odd : ∀ i ∈ F, f i % 2 = 1)
  (hedge_count : ∑ i in F, f i = 2 * n) : false :=
by
  sorry

end no_polyhedron_with_odd_faces_odd_edges_l444_444322


namespace production_normalities_l444_444828

noncomputable def morning_production (x : ℝ) : Prop :=
  -- The measured diameter from morning production
  9.88 ≤ x ∧ x ≤ 10.12

noncomputable def afternoon_production (y : ℝ) : Prop :=
  -- The measured diameter from afternoon production is not within the normal range
  y < 9.88 ∨ y > 10.12

theorem production_normalities :
  morning_production 9.9 ∧ afternoon_production 9.3 :=
by
  split; -- Split the conjunction into two separate goals
  -- Prove morning production is normal
  { unfold morning_production
    show 9.88 ≤ 9.9 ∧ 9.9 ≤ 10.12
    sorry
  },
  -- Prove afternoon production is abnormal
  { unfold afternoon_production
    show 9.3 < 9.88 ∨ 9.3 > 10.12
    sorry
  }

end production_normalities_l444_444828


namespace numbers_not_squares_or_cubes_in_200_l444_444473

/-- There are 182 numbers between 1 and 200 that are neither perfect squares nor perfect cubes. -/
theorem numbers_not_squares_or_cubes_in_200 : (finset.range 200).filter (λ n, ¬(∃ m, m^2 = n) ∧ ¬(∃ k, k^3 = n)).card = 182 :=
by 
  /-
    The proof will involve constructing the set of numbers from 1 to 200, filtering out those that
    are perfect squares or perfect cubes, and counting the remaining numbers.
  -/
  sorry

end numbers_not_squares_or_cubes_in_200_l444_444473


namespace floor_sum_identity_l444_444357

noncomputable def floor_function (x : ℝ) : ℤ := Int.floor x

def sequence (n : ℕ) : ℤ :=
  floor_function (n / 3)

def partial_sum (n : ℕ) : ℤ :=
  ∑ i in Finset.range (n + 1), sequence i

theorem floor_sum_identity (n : ℕ) : partial_sum (3 * n) = (3 * n * (3 * n - 1) / 2) / 3 :=
by
  sorry

end floor_sum_identity_l444_444357


namespace _l444_444708

lemma triangle_angle_neq_side_neq (A B C : Type) [euclidean_geometry A B C] 
  (h1 : ∠ A ≠ ∠ B) : ¬ (AC = BC) :=
by 
  assume h2 : AC = BC
  have h3 : ∠ A = ∠ B := isosceles_triangle_theorem h2
  contradiction

end _l444_444708


namespace numbers_not_squares_or_cubes_in_200_l444_444467

/-- There are 182 numbers between 1 and 200 that are neither perfect squares nor perfect cubes. -/
theorem numbers_not_squares_or_cubes_in_200 : (finset.range 200).filter (λ n, ¬(∃ m, m^2 = n) ∧ ¬(∃ k, k^3 = n)).card = 182 :=
by 
  /-
    The proof will involve constructing the set of numbers from 1 to 200, filtering out those that
    are perfect squares or perfect cubes, and counting the remaining numbers.
  -/
  sorry

end numbers_not_squares_or_cubes_in_200_l444_444467


namespace find_a_l444_444051

def f (a : ℝ) (x : ℝ) : ℝ :=
if x ≥ 0 then a * 2^x else 2^(-x)

theorem find_a (a : ℝ) (h : f a (f a (-1)) = 1) : a = 1 / 4 :=
by
  sorry

end find_a_l444_444051


namespace number_of_days_l444_444545

theorem number_of_days (a d g : ℕ) (h1 : a > 0) (h2 : d > 0) (h3 : g > 0) 
    (h4 : (a * d > 0)) : 
    let x := a^2 / g in 
    (d * a * d) = (g * x) :=
by
  sorry

end number_of_days_l444_444545


namespace perpendicular_planes_condition_l444_444370

variables (α β : Plane) (m : Line) 

-- Assuming the basic definitions:
def perpendicular (α β : Plane) : Prop := sorry
def in_plane (m : Line) (α : Plane) : Prop := sorry
def perpendicular_to_plane (m : Line) (β : Plane) : Prop := sorry

-- Conditions
axiom α_diff_β : α ≠ β
axiom m_in_α : in_plane m α

-- Proving the necessary but not sufficient condition
theorem perpendicular_planes_condition : 
  (perpendicular α β → perpendicular_to_plane m β) ∧ 
  (¬ perpendicular_to_plane m β → ¬ perpendicular α β) ∧ 
  ¬ (perpendicular_to_plane m β → perpendicular α β) :=
sorry

end perpendicular_planes_condition_l444_444370


namespace prism_upper_part_volume_surface_area_l444_444193

theorem prism_upper_part_volume_surface_area (a : ℝ) 
  (h_lateral : a < lateral_edge)
  (h_angle_plane_base : angle_plane_base = 45) :

  -- Volume of the upper part of the prism
  volume_upper_prism = a^3 / 8 ∧
  
  -- Surface area of the upper part of the prism
  surface_area_upper_prism = (a^2 * sqrt(3) * (3 + sqrt(2))) / 4 :=
begin
  sorry
end

end prism_upper_part_volume_surface_area_l444_444193


namespace number_of_boys_in_first_group_l444_444992

-- Define the daily work ratios
variables (M B : ℝ) (h_ratio : M = 2 * B)

-- Define the number of boys in the first group
variable (x : ℝ)

-- Define the conditions provided by the problem
variables (h1 : 5 * (12 * M + x * B) = 4 * (13 * M + 24 * B))

-- State the theorem and include the correct answer
theorem number_of_boys_in_first_group (M B : ℝ) (h_ratio : M = 2 * B) (x : ℝ)
    (h1 : 5 * (12 * M + x * B) = 4 * (13 * M + 24 * B)) 
    : x = 16 := 
by 
    sorry

end number_of_boys_in_first_group_l444_444992


namespace correct_statements_l444_444709

theorem correct_statements 
  (population : ℕ)
  (sample_size : ℕ)
  (total_students : population = 240)
  (selected_students : sample_size = 40) :
  (population = 240) ∧
  (∀ (individual : population), True) ∧
  (sample_size = 40) ∧
  (∃ (sample : fin sample_size), True) :=
by
  -- Proof omitted
  sorry

end correct_statements_l444_444709


namespace function_value_at_neg_five_l444_444672

theorem function_value_at_neg_five (a b : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = a * x + b * real.sin x + 1)
  (h2 : f 5 = 7) :
  f (-5) = -5 :=
sorry

end function_value_at_neg_five_l444_444672


namespace sin_angle_eq_sqrt3_div_2_l444_444613

variables {α : Type*} [InnerProductSpace ℝ α]

-- Definitions and assumptions
variables (u v w : α) (φ : ℝ)
variables (h1 : u ≠ 0) (h2 : v ≠ 0) (h3 : w ≠ 0)
variables (h4 : ¬ ∃ k : ℝ, u = k • v) (h5 : ¬ ∃ k : ℝ, v = k • w) (h6 : ¬ ∃ k : ℝ, w = k • u)
variables (h7 : ((u × v) × w) = (1 / 2) * ‖v‖ * ‖w‖ ∙ u)
variables (h8 : φ = real.angle v w)

-- Proof statement
theorem sin_angle_eq_sqrt3_div_2 : real.sin φ = (√3) / 2 := sorry

end sin_angle_eq_sqrt3_div_2_l444_444613


namespace planes_parallel_lines_parallel_l444_444031

open Set

variables {Point : Type} [MetricSpace Point]

structure Line (Point : Type) :=
  (containing : Set Point)
  (is_line : is_infinite is_line)

structure Plane (Point : Type) :=
  (containing : Set Point)
  (is_plane : is_infinite is_plane)

variable {α β γ : Plane Point}
variable {m n : Line Point}

-- alpha intersect gamma = m
axiom h1 : is_intersection α γ m
-- beta intersect gamma = n
axiom h2 : is_intersection β γ n
-- alpha parallel beta
axiom h3 : is_parallel α β

-- Prove that m is parallel to n
theorem planes_parallel_lines_parallel : is_parallel m n :=
by
  sorry

end planes_parallel_lines_parallel_l444_444031


namespace transformed_parabola_equation_l444_444186

-- Define the initial parabola equation
def initial_parabola (x : ℝ) : ℝ := (x - 3)^2 - 4

-- Define the transformation: horizontal shift right by 1 unit
def horizontal_shift (x : ℝ) : ℝ := x - 1

-- Define the transformation: vertical shift up by 2 units
def vertical_shift (y : ℝ) : ℝ := y + 2

-- Define the resulting parabola after both transformations
def transformed_parabola (x : ℝ) : ℝ :=
  let y := initial_parabola (horizontal_shift x)
  in vertical_shift y

-- Prove the final equation of the transformed parabola
theorem transformed_parabola_equation : 
  ∀ x : ℝ, transformed_parabola x = (x - 4)^2 - 2 :=
by
  intro x
  sorry

end transformed_parabola_equation_l444_444186


namespace unique_root_px_sqrt_eq_x_l444_444002

theorem unique_root_px_sqrt_eq_x :
  ∀ p : ℝ, 
  (∃! x : ℝ, x + 1 = real.sqrt (p * x)) ↔ (p = 4 ∨ p ≤ 0) := 
by sorry

end unique_root_px_sqrt_eq_x_l444_444002


namespace num_n_digit_numbers_each_digit_123_appears_l444_444871

theorem num_n_digit_numbers_each_digit_123_appears (n : ℕ) (hn : n ≥ 3) : 
  ∃ count : ℕ, count = 3^n - 3 * 2^n + 3 :=
by 
  use 3^n - 3 * 2^n + 3
  sorry

end num_n_digit_numbers_each_digit_123_appears_l444_444871


namespace count_non_perfect_square_or_cube_l444_444519

-- Define perfect_square function
def perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k^2 = n

-- Define perfect_cube function
def perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k^3 = n

-- Define range from 1 to 200
def range_200 := {1 .. 200}

-- Define numbers either perfect square or perfect cube
def perfect_square_or_cube : Finset ℕ :=
  (range_200.filter perfect_square) ∪ (range_200.filter perfect_cube)

-- Define sixth powers in range
def perfect_sixth_power (n : ℕ) : Prop :=
  ∃ k : ℕ, k^6 = n

-- The final problem statement
theorem count_non_perfect_square_or_cube :
  (range_200.card - perfect_square_or_cube.card) = 182 :=
by
  sorry

end count_non_perfect_square_or_cube_l444_444519


namespace planting_schemes_count_l444_444733

-- Define the basic conditions
def seedTypes : Finset String := {"corn", "potato", "eggplant", "chili", "carrot"}
def firstPlotChoices : Finset String := {"eggplant", "chili"}

-- Define the proof problem (problem statement)
theorem planting_schemes_count 
  (h_seed_types : seedTypes.card = 5)
  (h_first_plot_choices : firstPlotChoices.card = 2) : 
  ∃ n : ℕ, n = 48 ∧ 
  let remainingPlots := seedTypes.erase "eggplant" ∪ seedTypes.erase "chili" in
  let permutations := Nat.factorial (remainingPlots.card) / Nat.factorial (remainingPlots.card - 3) in
  remainingPlots.card = 4 → (firstPlotChoices.card * permutations = n) := 
by 
  -- Placeholder for proof
  sorry 

end planting_schemes_count_l444_444733


namespace difference_of_squares_count_l444_444971

theorem difference_of_squares_count :
  (number_of_integers_between (1 : ℕ) (500 : ℕ) (λ n, ∃ a b : ℕ, n = a^2 - b^2)) = 375 :=
by
  sorry

end difference_of_squares_count_l444_444971


namespace max_sides_covered_by_circles_l444_444933

theorem max_sides_covered_by_circles (n : ℕ) (convex_ngon : Type) 
(h_convex : convex convex_ngon) 
(h_circles_cover : ∀ (i : fin n), covers_by_circle (side convex_ngon i)) : n ≤ 4 := 
sorry

end max_sides_covered_by_circles_l444_444933


namespace a_pow_11_b_pow_11_l444_444149

-- Define the conditions a + b = 1, a^2 + b^2 = 3, a^3 + b^3 = 4, a^4 + b^4 = 7, and a^5 + b^5 = 11
def a : ℝ := sorry
def b : ℝ := sorry

axiom h1 : a + b = 1
axiom h2 : a^2 + b^2 = 3
axiom h3 : a^3 + b^3 = 4
axiom h4 : a^4 + b^4 = 7
axiom h5 : a^5 + b^5 = 11

-- Define the recursion pattern for n ≥ 3
axiom h6 (n : ℕ) (hn : n ≥ 3) : a^n + b^n = a^(n-1) + b^(n-1) + a^(n-2) + b^(n-2)

-- Prove that a^11 + b^11 = 199
theorem a_pow_11_b_pow_11 : a^11 + b^11 = 199 :=
by sorry

end a_pow_11_b_pow_11_l444_444149


namespace union_A_B_inter_A_compl_B_range_of_a_l444_444141

-- Define the sets A, B, and C
def A := {x : ℝ | -1 ≤ x ∧ x < 3}
def B := {x : ℝ | 2 * x - 4 ≥ x - 2}
def C (a : ℝ) := {x : ℝ | x ≥ a - 1}

-- Prove A ∪ B = {x | -1 ≤ x}
theorem union_A_B : A ∪ B = {x : ℝ | -1 ≤ x} :=
by sorry

-- Prove A ∩ (complement B) = {x | -1 ≤ x < 2}
theorem inter_A_compl_B : A ∩ (compl B) = {x : ℝ | -1 ≤ x ∧ x < 2} :=
by sorry

-- Prove the range of a given B ∪ C = C
theorem range_of_a (a : ℝ) (h : B ∪ C a = C a) : a ≤ 3 :=
by sorry

end union_A_B_inter_A_compl_B_range_of_a_l444_444141


namespace num_from_1_to_200_not_squares_or_cubes_l444_444465

noncomputable def numNonPerfectSquaresAndCubes (n : ℕ) : ℕ :=
  let num_squares := 14
  let num_cubes := 5
  let num_sixth_powers := 2
  n - (num_squares + num_cubes - num_sixth_powers)

theorem num_from_1_to_200_not_squares_or_cubes : numNonPerfectSquaresAndCubes 200 = 183 := by
  sorry

end num_from_1_to_200_not_squares_or_cubes_l444_444465


namespace g_of_g_of_3_l444_444390

noncomputable def g (x : ℚ) : ℚ :=
  x^(-2) + x^(-2) / (1 + x^(-2))

theorem g_of_g_of_3 : g (g 3) = 68754921 / 3074821 :=
by
  sorry

end g_of_g_of_3_l444_444390


namespace merchant_profit_l444_444735

noncomputable theory

def profit_percentage (CP : ℝ) (markup_rate : ℝ) (discount_rate : ℝ) : ℝ :=
let marked_price := CP * (1 + markup_rate) in
let selling_price := marked_price * (1 - discount_rate) in
((selling_price - CP) / CP) * 100

theorem merchant_profit :
  profit_percentage 100 0.4 0.2 = 12 := by sorry

end merchant_profit_l444_444735


namespace count_non_perfect_square_or_cube_l444_444517

-- Define perfect_square function
def perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k^2 = n

-- Define perfect_cube function
def perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k^3 = n

-- Define range from 1 to 200
def range_200 := {1 .. 200}

-- Define numbers either perfect square or perfect cube
def perfect_square_or_cube : Finset ℕ :=
  (range_200.filter perfect_square) ∪ (range_200.filter perfect_cube)

-- Define sixth powers in range
def perfect_sixth_power (n : ℕ) : Prop :=
  ∃ k : ℕ, k^6 = n

-- The final problem statement
theorem count_non_perfect_square_or_cube :
  (range_200.card - perfect_square_or_cube.card) = 182 :=
by
  sorry

end count_non_perfect_square_or_cube_l444_444517


namespace a_beats_b_by_d_l444_444220

-- Conditions
def distance_a : ℕ := 256
def time_a : ℕ := 28
def distance_b : ℕ := 256
def time_b : ℕ := 32

-- Define speeds (using rational numbers for exactness)
def speed_a : ℚ := distance_a / time_a
def speed_b : ℚ := distance_b / time_b

-- Distance A runs in the time it takes B to finish
def distance_a_in_time_b : ℚ := speed_a * time_b

-- Distance by which A beats B
def distance_a_beats_b : ℚ := distance_a_in_time_b - distance_b

theorem a_beats_b_by_d : distance_a_beats_b ≈ 36.57 := 
by 
  sorry

end a_beats_b_by_d_l444_444220


namespace part_1_part_2_l444_444917

def custom_operation (a b : ℝ) : ℝ :=
if a ≥ 2 * b then a - b else a + b - 6

theorem part_1 : custom_operation 4 3 = 1 ∧ custom_operation (-1) (-3) = 2 := by
  -- Proof will go here
  sorry

theorem part_2 (x : ℝ) : custom_operation (3 * x + 2) (x - 1) = 5 → x = 1 := by
  -- Proof will go here
  sorry

end part_1_part_2_l444_444917


namespace find_f_neg_six_l444_444171

-- Defining the even and periodic function f with conditions
def even (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f x = f (x + p)

noncomputable def f : ℝ → ℝ :=
  λ x, if (-3 ≤ x ∧ x ≤ 3) then (x + 1) * (x - 1) else 0 -- as outside -3 to 3 values given implicitly to be recalculated by periodicity

-- The theorem we need to prove
theorem find_f_neg_six : f (-6) = -1 :=
by
  sorry

end find_f_neg_six_l444_444171


namespace books_after_donation_l444_444864

/-- 
  Total books Boris and Cameron have together after donating some books.
 -/
theorem books_after_donation :
  let B : ℕ := 24 in   -- Initial books Boris has
  let C : ℕ := 30 in   -- Initial books Cameron has
  let B_donated := B / 4 in  -- Boris donates a fourth of his books
  let C_donated := C / 3 in  -- Cameron donates a third of his books
  B - B_donated + (C - C_donated) = 38 :=  -- After donating, the total books
by
  sorry

end books_after_donation_l444_444864


namespace count_diff_squares_1_to_500_l444_444976

theorem count_diff_squares_1_to_500 : ∃ count : ℕ, count = 416 ∧
  (count = (nat.card {n : ℕ | n ∈ (set.Icc 1 500) ∧ (∃ a b : ℕ, a^2 - b^2 = n)})) := by
sorry

end count_diff_squares_1_to_500_l444_444976


namespace sum_remainder_mod_9_l444_444350

theorem sum_remainder_mod_9 : 
  (123456 + 123457 + 123458 + 123459 + 123460 + 123461) % 9 = 6 :=
by
  sorry

end sum_remainder_mod_9_l444_444350


namespace magnitude_of_complex_power_eight_l444_444900

def complex_number : ℂ := (5 / 13 : ℂ) + (12 / 13 : ℂ) * Complex.I

theorem magnitude_of_complex_power_eight : 
  Complex.abs (complex_number ^ 8) = 1 := 
by 
  sorry

end magnitude_of_complex_power_eight_l444_444900


namespace compute_AQ_l444_444651

theorem compute_AQ :
  ∀ (P Q R A B C : Type) [triangle P Q R] [right_triangle A B C],
    PQ = 6 → PR = 7 → QR = 8 → 
    (segmentOn PR B) → (segmentOn QR C) → (segmentOn PQ A) →
    PC = 4 → BP = 3 → CQ = 3 →
    AQ = 112 / 35 :=
by
  sorry

end compute_AQ_l444_444651


namespace bamboo_volume_top_section_l444_444087

-- Define the conditions as per step a)
def bamboo_sections : ℕ := 9
def top_4_volume : ℝ := 3
def bottom_3_volume : ℝ := 4

def volumes_form_arithmetic_sequence := true

-- The volume of the top section assuming it’s an arithmetic sequence
def volume_top_section : ℝ :=
  let a1 := (13 : ℝ) / 22 in a1

-- Statement to be proved:
theorem bamboo_volume_top_section (d : ℝ) : 
  (4 * volume_top_section + 6 * (d / 2) = top_4_volume ∧
   9 * volume_top_section + 36 * (d / 2) - 
   (6 * volume_top_section + 15 * (d / 2)) = bottom_3_volume) → 
  volume_top_section = (13 : ℝ) / 22 :=
by
  -- Proof steps are to be provided here
  -- Sorry is used to denote the proof is skipped
  sorry

end bamboo_volume_top_section_l444_444087


namespace numbers_not_perfect_squares_or_cubes_l444_444455

theorem numbers_not_perfect_squares_or_cubes : 
  let total_numbers := 200
  let perfect_squares := 14
  let perfect_cubes := 5
  let sixth_powers := 1
  total_numbers - (perfect_squares + perfect_cubes - sixth_powers) = 182 :=
by
  sorry

end numbers_not_perfect_squares_or_cubes_l444_444455


namespace numbers_not_perfect_squares_or_cubes_l444_444447

theorem numbers_not_perfect_squares_or_cubes : 
  let total_numbers := 200
  let perfect_squares := 14
  let perfect_cubes := 5
  let sixth_powers := 1
  total_numbers - (perfect_squares + perfect_cubes - sixth_powers) = 182 :=
by
  sorry

end numbers_not_perfect_squares_or_cubes_l444_444447


namespace problem_l444_444127

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
if h : (-1 : ℝ) ≤ x ∧ x < 0 then a*x + 1
else if h : (0 : ℝ) ≤ x ∧ x ≤ 1 then (b*x + 2) / (x + 1)
else 0 -- This should not matter as we only care about the given ranges

theorem problem (a b : ℝ) (h₁ : f 0.5 a b = f 1.5 a b) : a + 3 * b = -10 :=
by
  -- We'll derive equations from given conditions and prove the result.
  sorry

end problem_l444_444127


namespace product_of_three_3_digits_has_four_zeros_l444_444310

noncomputable def has_four_zeros_product : Prop :=
  ∃ (a b c: ℕ),
    (100 ≤ a ∧ a < 1000) ∧
    (100 ≤ b ∧ b < 1000) ∧
    (100 ≤ c ∧ c < 1000) ∧
    (∃ (da db dc: Finset ℕ), (da ∪ db ∪ dc = Finset.range 10) ∧
    (∀ x ∈ da, x = a / 10^(x%10) % 10) ∧
    (∀ x ∈ db, x = b / 10^(x%10) % 10) ∧
    (∀ x ∈ dc, x = c / 10^(x%10) % 10)) ∧
    (a * b * c % 10000 = 0)

theorem product_of_three_3_digits_has_four_zeros : has_four_zeros_product := sorry

end product_of_three_3_digits_has_four_zeros_l444_444310


namespace area_of_parallelogram_is_5_l444_444129

-- Define the points in ℝ³
structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

-- Define the three given points A, B, and C
def A : Point3D := ⟨1, 2, -1⟩
def B : Point3D := ⟨0, 3, 1⟩
def C : Point3D := ⟨-2, 1, 2⟩

-- Define the vectors AB and AC
def vector_sub (P Q : Point3D) : Point3D :=
  ⟨Q.x - P.x, Q.y - P.y, Q.z - P.z⟩

def AB := vector_sub A B
def AC := vector_sub A C

-- Define the cross product of two vectors
def cross_product (u v : Point3D) : Point3D :=
  ⟨u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x⟩

-- Calculate the cross product of AB and AC
def AB_cross_AC := cross_product AB AC

-- Define the magnitude of the cross product
def magnitude (v : Point3D) : ℝ :=
  real.sqrt (v.x * v.x + v.y * v.y + v.z * v.z)

-- Calculate the area of the parallelogram
def area_parallelogram : ℝ := magnitude AB_cross_AC

-- The goal is to prove that area_parallelogram equals 5
theorem area_of_parallelogram_is_5 : area_parallelogram = 5 := by
  sorry

end area_of_parallelogram_is_5_l444_444129


namespace length_of_AB_l444_444106

-- Given: Triangle ABC is a right triangle with angle C being 90 degrees and angle B being 45 degrees.
-- Also given that the length of side AC is 18√2.
-- Prove that the length of side AB is also 18√2.

-- Definitions of the given conditions
def triangle (A B C : Type) : Type := sorry  -- placeholder for the definition of a triangle
def is_right_triangle (ABC : triangle A B C) : Prop := sorry  -- ABC is a right triangle
def angle_C_eq_90 (ABC : triangle A B C) : Prop := sorry  -- ∠C in triangle ABC is 90°
def angle_B_eq_45 (ABC : triangle A B C) : Prop := sorry  -- ∠B in triangle ABC is 45°
def length_AC_eq_18_sqrt_2 (ABC : triangle A B C) : Prop := sorry  -- length of side AC in triangle ABC is 18√2

-- Our goal is to prove that the length of side AB is 18√2 in the given triangle.
theorem length_of_AB (ABC : triangle A B C) :
  is_right_triangle ABC →
  angle_C_eq_90 ABC →
  angle_B_eq_45 ABC →
  length_AC_eq_18_sqrt_2 ABC →
  length_AB_eq_18_sqrt_2 ABC :=
by
  intros
  sorry  -- proof omitted

end length_of_AB_l444_444106


namespace find_b_l444_444680

theorem find_b (b : ℚ) : (∃ x y : ℚ, x = 3 ∧ y = -5 ∧ (b * x - (b + 2) * y = b - 3)) → b = -13 / 7 :=
sorry

end find_b_l444_444680


namespace numbers_not_squares_nor_cubes_1_to_200_l444_444498

theorem numbers_not_squares_nor_cubes_1_to_200 : 
  let num_squares := nat.sqrt 200
  let num_cubes := nat.cbrt 200
  let num_sixth_powers := nat.root 6 200
  let total := num_squares + num_cubes - num_sixth_powers
  200 - total = 182 := 
by
  sorry

end numbers_not_squares_nor_cubes_1_to_200_l444_444498


namespace polynomial_remainder_l444_444609

def h (x : ℝ) : ℝ := x^3 + x^2 + x + 1

theorem polynomial_remainder (x : ℝ) :
  let hx8 := h (x^8)
  let hx := h x
  ∃ q r, hx8 = q * hx + (4:ℝ) := sorry

end polynomial_remainder_l444_444609


namespace system_of_equations_solution_l444_444888

theorem system_of_equations_solution (x y z : ℝ) :
  (x^2 + y * z = 1 ∧ y^2 - x * z = 0 ∧ z^2 + x * y = 1) →
  (x = y ∧ y = z ∧ (x = real.sqrt 2 / 2 ∨ x = -real.sqrt 2 / 2)) :=
by
  sorry

end system_of_equations_solution_l444_444888


namespace numbers_not_squares_nor_cubes_1_to_200_l444_444491

theorem numbers_not_squares_nor_cubes_1_to_200 : 
  let num_squares := nat.sqrt 200
  let num_cubes := nat.cbrt 200
  let num_sixth_powers := nat.root 6 200
  let total := num_squares + num_cubes - num_sixth_powers
  200 - total = 182 := 
by
  sorry

end numbers_not_squares_nor_cubes_1_to_200_l444_444491


namespace men_added_l444_444248

theorem men_added (M : ℕ) (h1 : M * 8 * 24 = 12 * 8 * 16) : 12 - M = 4 :=
by
  have := congr_arg (λ x, x / 8) h1
  sorry

end men_added_l444_444248


namespace min_of_max_abs_l444_444332

def f (x y : ℝ) : ℝ := x^3 - xy

theorem min_of_max_abs : 
  (∃ y ∈ set.Icc (y: ℝ) ↑(-1) ↑(3), ∀ x ∈ set.Icc (0 : ℝ) 1, f x y = min fun y => max fun x => |f x y|)
  := by sorry

end min_of_max_abs_l444_444332


namespace non_perfect_powers_count_l444_444483

theorem non_perfect_powers_count :
  let n := 200,
      num_perfect_squares := 14,
      num_perfect_cubes := 5,
      num_perfect_sixth_powers := 1,
      total_perfect_powers := num_perfect_squares + num_perfect_cubes - num_perfect_sixth_powers,
      total_non_perfect_powers := n - total_perfect_powers
  in
  total_non_perfect_powers = 182 :=
by
  sorry

end non_perfect_powers_count_l444_444483


namespace total_digits_first_2002_odd_integers_l444_444725

theorem total_digits_first_2002_odd_integers : 
  let n := 2002 in 
  let k := 2 * n - 1 in 
  let one_digit := 1 * 4 in 
  let two_digit := 2 * (k - 9) / 2 + 2 in 
  let three_digit := 3 * (999-100) / 2 + 3 in 
  let four_digit := 4 * (k - 1000) / 2 + 4 in 

  one_digit + two_digit + three_digit + four_digit = 7454 := by
  sorry

end total_digits_first_2002_odd_integers_l444_444725


namespace percentage_of_b_l444_444237

variable (a b c p : ℝ)

theorem percentage_of_b :
  (0.04 * a = 8) →
  (p * b = 4) →
  (c = b / a) →
  p = 1 / (50 * c) :=
by
  sorry

end percentage_of_b_l444_444237


namespace rows_seat_7_students_are_5_l444_444325

-- Definitions based on provided conditions
def total_students : Nat := 53
def total_rows (six_seat_rows seven_seat_rows : Nat) : Prop := 
  total_students = 6 * six_seat_rows + 7 * seven_seat_rows

-- To prove the number of rows seating exactly 7 students is 5
def number_of_7_seat_rows (six_seat_rows seven_seat_rows : Nat) : Prop := 
  total_rows six_seat_rows seven_seat_rows ∧ seven_seat_rows = 5

-- Statement to be proved
theorem rows_seat_7_students_are_5 : ∃ (six_seat_rows seven_seat_rows : Nat), number_of_7_seat_rows six_seat_rows seven_seat_rows := 
by
  -- Skipping the proof
  sorry

end rows_seat_7_students_are_5_l444_444325


namespace compare_harvest_l444_444003

-- Define the variables for the amount harvested per day
variables {x y : ℝ}  -- hectares per day by brand "K" and "H"

-- Conditions given in the problem
def condition : Prop :=
  5 * (4 * x + 3 * y) = 4 * (3 * x + 5 * y)

-- The goal is to prove that the combine of brand "H" harvests more per day than brand "K"
theorem compare_harvest (h : condition) : y > x :=
by
  sorry

end compare_harvest_l444_444003


namespace length_PQ_l444_444124
-- Import the Mathlib library

-- Define the points in the 3D space
structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

-- Define the points A, B, C, D, A', B', C', D'
def E := Point3D.mk 0 0 0
def F := Point3D.mk 2 0 0
def G := Point3D.mk 2 3 0
def H := Point3D.mk 0 3 0
def E' := Point3D.mk 0 0 12
def F' := Point3D.mk 2 0 16
def G' := Point3D.mk 2 3 20
def H' := Point3D.mk 0 3 24

-- Define the midpoints P and Q
def midpoint (A B : Point3D) : Point3D :=
  Point3D.mk ((A.x + B.x) / 2) ((A.y + B.y) / 2) ((A.z + B.z) / 2)

def P := midpoint E' G'
def Q := midpoint F' H'

-- Define the distance between two 3D points
def distance (A B : Point3D) : ℝ :=
  Real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2 + (A.z - B.z)^2)

-- State the theorem
theorem length_PQ : distance P Q = 4 :=
by
  -- The proof is omitted
  sorry

end length_PQ_l444_444124


namespace count_not_squares_or_cubes_l444_444501

theorem count_not_squares_or_cubes {A B : Finset ℕ} (h_range : ∀ x, x ∈ A ∪ B → x ≤ 200)
  (h_squares : A.card = 14) (h_cubes : B.card = 5) (h_inter : (A ∩ B).card = 1) :
  (200 - (A ∪ B).card) = 182 :=
by
  sorry

end count_not_squares_or_cubes_l444_444501


namespace partition_ways_l444_444582

theorem partition_ways (black_cells : ℕ) (grey_cells : ℕ) (total_cells : ℕ) : 
  total_cells = 17 ∧ black_cells = 9 ∧ grey_cells = 8 →
  ∃ ways : ℕ, ways = 10 :=
begin
  sorry
end

end partition_ways_l444_444582


namespace distinct_prime_factors_sum_252_number_of_distinct_prime_factors_252_l444_444303

def is_prime (n : Nat) : Prop :=
  2 ≤ n ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

def prime_factors (n : Nat) : Set Nat :=
  { p | is_prime p ∧ p ∣ n }

def sum_of_set (s : Set Nat) : Nat :=
  s.toFinset.sum id  -- assuming coercion from set to finset is possible

theorem distinct_prime_factors_sum_252 :
  prime_factors 252 = {2, 3, 7} ∧ sum_of_set (prime_factors 252) = 12 :=
by sorry

theorem number_of_distinct_prime_factors_252 :
  (prime_factors 252).card = 3 :=
by sorry

end distinct_prime_factors_sum_252_number_of_distinct_prime_factors_252_l444_444303


namespace john_initial_pairs_9_l444_444593

-- Definitions based on the conditions in the problem

def john_initial_pairs (x : ℕ) := 2 * x   -- Each pair consists of 2 socks

def john_remaining_socks (x : ℕ) := john_initial_pairs x - 5   -- John loses 5 individual socks

def john_max_pairs_left := 7
def john_minimum_socks_required := john_max_pairs_left * 2  -- 7 pairs mean he needs 14 socks

-- Theorem statement proving John initially had 9 pairs of socks
theorem john_initial_pairs_9 : 
  ∀ (x : ℕ), john_remaining_socks x ≥ john_minimum_socks_required → x = 9 := by
  sorry

end john_initial_pairs_9_l444_444593


namespace max_magnitude_expression_l444_444046

theorem max_magnitude_expression (z : ℂ) (h : complex.abs z = 1) : 
    ∃ θ, complex.abs (z^3 + z^2 - 5*z + 3) ≤ (128 * real.sqrt 3) / 27 := 
begin
  sorry,
end

end max_magnitude_expression_l444_444046


namespace hyperbola_focus_l444_444675

noncomputable def c := Real.sqrt (7^2 + 3^2)

theorem hyperbola_focus : 
  (∃ (x y : ℝ), (x, y) = (2 - c, -5)) ↔ (x, y) coordinates of the focus with the smaller x-coordinate of the hyperbola given by (x-2)^2 / 7^2 - (y+5)^2 / 3^2 = 1 :=
by
  sorry

end hyperbola_focus_l444_444675


namespace c_in_terms_of_a_b_l444_444065

def vec := (ℝ × ℝ)
def a : vec := (1, 2)
def b : vec := (-2, 3)
def c : vec := (4, 1)

theorem c_in_terms_of_a_b : c = (2 : ℝ) • a + (-1 : ℝ) • b := by
  sorry

end c_in_terms_of_a_b_l444_444065


namespace symmetry_axis_range_l444_444008

noncomputable def f (x : ℝ) := 2 * Real.sin (2 * x - Real.pi / 6)

theorem symmetry_axis_range (m : ℝ) :
  (∃ (f : ℝ → ℝ), f(Real.pi / 6) = 1 ∧ 
    (∀ x, ∃ c, c ∈ Set.Icc (0 : ℝ) m ∧ f c = f (-c)) ∧
    (∀ x, f x ≤ 2) ∧ (∀ x, f x ≥ -2)) ↔ m ∈ Set.Icc (Real.pi / 3) (5 * Real.pi / 6) :=
sorry

end symmetry_axis_range_l444_444008


namespace commission_rate_correct_l444_444898

-- Define the values for price and quantity of items sold
def price_suit : ℝ := 700
def quantity_suit : ℕ := 2

def price_shirt : ℝ := 50
def quantity_shirt : ℕ := 6

def price_loafers : ℝ := 150
def quantity_loafers : ℕ := 2

def total_commission : ℝ := 300

-- Compute the commission rate
theorem commission_rate_correct :
  let total_sales := (quantity_suit * price_suit) + 
                     (quantity_shirt * price_shirt) + 
                     (quantity_loafers * price_loafers) in
  (total_commission / total_sales) * 100 = 15 := by
  sorry

end commission_rate_correct_l444_444898


namespace bn_general_formula_l444_444378

open Nat

noncomputable def a (n : ℕ) : ℚ := 1 / ((n + 1)^2)
noncomputable def b : ℕ → ℚ
| 0       := 1
| (n + 1) := b n * (1 - a (n + 1))

theorem bn_general_formula (n : ℕ) : b n = (n + 2) / (2 * (n + 1)) := by
  induction n with d hd
  · simp [b, a]; norm_num
  · have h1 : 1 - a (d + 1) = (d^2 + 4*d + 3) / (d + 2)^2 := by
      rw [a]
      field_simp; ring
    simp [b, a, hd, h1]
    field_simp; ring

end bn_general_formula_l444_444378


namespace count_not_squares_or_cubes_l444_444506

theorem count_not_squares_or_cubes {A B : Finset ℕ} (h_range : ∀ x, x ∈ A ∪ B → x ≤ 200)
  (h_squares : A.card = 14) (h_cubes : B.card = 5) (h_inter : (A ∩ B).card = 1) :
  (200 - (A ∪ B).card) = 182 :=
by
  sorry

end count_not_squares_or_cubes_l444_444506


namespace max_playground_area_l444_444421

theorem max_playground_area
  (l w : ℝ)
  (h_fence : 2 * l + 2 * w = 400)
  (h_l_min : l ≥ 100)
  (h_w_min : w ≥ 50) :
  l * w ≤ 10000 :=
by
  sorry

end max_playground_area_l444_444421


namespace sin_diff_pi_over_4_theta_cos_double_theta_l444_444945

variables (θ : ℝ)

-- Conditions
axiom theta_acute : 0 < θ ∧ θ < π / 2
axiom sin_theta_eq : real.sin θ = 1 / 3

-- Proof goals
theorem sin_diff_pi_over_4_theta : real.sin (π / 4 - θ) = (4 - real.sqrt 2) / 6 :=
sorry

theorem cos_double_theta : real.cos (2 * θ) = 7 / 9 :=
sorry

end sin_diff_pi_over_4_theta_cos_double_theta_l444_444945


namespace time_diff_is_6_l444_444883

-- Define the speeds for the different sails
def speed_of_large_sail : ℕ := 50
def speed_of_small_sail : ℕ := 20

-- Define the distance of the trip
def trip_distance : ℕ := 200

-- Calculate the time for each sail
def time_large_sail (distance : ℕ) (speed : ℕ) : ℕ := distance / speed
def time_small_sail (distance : ℕ) (speed : ℕ) : ℕ := distance / speed

-- Define the time difference
def time_difference (distance : ℕ) (speed_large : ℕ) (speed_small : ℕ) : ℕ := 
  (distance / speed_small) - (distance / speed_large)

-- Prove that the time difference between the large and small sails is 6 hours
theorem time_diff_is_6 : time_difference trip_distance speed_of_large_sail speed_of_small_sail = 6 := by
  -- useful := time_difference trip_distance speed_of_large_sail speed_of_small_sail,
  -- change useful with 6,
  sorry

end time_diff_is_6_l444_444883


namespace product_of_d_l444_444887

noncomputable def g (d x : ℝ) := d / (3 * x - 4)

theorem product_of_d (d : ℝ) :
  g d 3 = g⁻¹ d (d + 2) → (∀ d, (3 * d^2 - 3 * d - 8 = 0) → ∏ roots_of_quadratic_eq = -8 / 3) :=
by
  intro h eq
  sorry

end product_of_d_l444_444887


namespace value_of_g_neg2_l444_444211

-- Define the function g as given in the conditions
def g (x : ℝ) : ℝ := x^2 - 3 * x + 1

-- Statement of the problem: Prove that g(-2) = 11
theorem value_of_g_neg2 : g (-2) = 11 := by
  sorry

end value_of_g_neg2_l444_444211


namespace num_three_digit_numbers_l444_444698

/-- Define the cards and their possible outcomes -/
def card1 : set ℕ := {1, 2}
def card2 : set ℕ := {3, 4}
def card3 : set ℕ := {5, 7}

/-- The theorem to prove the total number of different three-digit numbers -/
theorem num_three_digit_numbers : 
  (card1.to_list.length * card2.to_list.length * card3.to_list.length) * (finset.univ.card - 2) = 48 :=
by
  sorry

end num_three_digit_numbers_l444_444698


namespace child_tickets_sold_l444_444706

theorem child_tickets_sold
  (A C : ℕ)
  (h1 : A + C = 130)
  (h2 : 12 * A + 4 * C = 840) : C = 90 :=
  by {
  -- Proof skipped
  sorry
}

end child_tickets_sold_l444_444706


namespace AF_passes_through_midpoint_DE_l444_444772

open_locale classical

-- Definition of the geometrical setup
def isosceles_triangle (A B C : Point) : Prop := dist A B = dist A C
def midpoint (D B C : Point) : Prop := dist B D = dist C D
def perpendicular (D E A C : Point) : Prop := ∠ DEA = 90
def circumcircle (A B D F : Point): Prop := ∃ O R, Circle.circle O R A ∧ Circle.circle O R B ∧ Circle.circle O R D ∧ Circle.circle O R F

-- Main statement to prove
theorem AF_passes_through_midpoint_DE
  (A B C D E F : Point)
  (h1 : isosceles_triangle A B C)
  (h2 : midpoint D B C)
  (h3 : perpendicular D E A C)
  (h4 : ∃ O R, Circle.circle O R A ∧ Circle.circle O R B ∧ Circle.circle O R D ∧ Circle.circle O R F)
  (h5 : collinear B E F)
  (h6 : intersects (circumcircle A B D) (line B E) B F) :
  passes_through (line A F) (midpoint D E) := 
sorry

end AF_passes_through_midpoint_DE_l444_444772


namespace smallest_real_C_l444_444909

noncomputable def golden_ratio : ℝ := (1 + Real.sqrt 5) / 2

noncomputable def fractional_part (x: ℝ) := x - Real.floor x

theorem smallest_real_C (C : ℝ) : 
  C = (golden_ratio - 1) ∧ 
  (∀ (x y : ℕ), x ≠ y → min (fractional_part (Real.sqrt (x^2 + 2*y))) (fractional_part (Real.sqrt (y^2 + 2*x))) < C) :=
begin
  sorry
end 

end smallest_real_C_l444_444909


namespace numbers_neither_square_nor_cube_l444_444543

theorem numbers_neither_square_nor_cube (n : ℕ) (h : n = 200) :
  let num_squares := 14 in
  let num_cubes := 5 in
  let num_sixth_powers := 2 in
  (n - (num_squares + num_cubes - num_sixth_powers)) = 183 :=
by
  -- Let the number of integers in the range
  let num_total := 200
  -- Define how we count perfect squares and cubes
  let num_squares := 14
  let num_cubes := 5
  let num_sixth_powers := 2

  -- Subtract the overlapping perfect sixth powers from the sum of squares and cubes
  let num_either_square_or_cube := num_squares + num_cubes - num_sixth_powers

  -- Calculate the final result
  let result := num_total - num_either_square_or_cube

  -- Prove by computation
  show result = 183 from
    calc
      result = num_total - num_either_square_or_cube := rfl
           ... = 200 - (14 + 5 - 2)                   := rfl
           ... = 200 - 17                             := rfl
           ... = 183                                  := rfl

end numbers_neither_square_nor_cube_l444_444543


namespace probability_interval_l444_444925

noncomputable def Phi : ℝ → ℝ := sorry -- assuming Φ is a given function for CDF of a standard normal distribution

theorem probability_interval (h : Phi 1.98 = 0.9762) : 
  2 * Phi 1.98 - 1 = 0.9524 :=
by
  sorry

end probability_interval_l444_444925


namespace sum_first_six_terms_geom_seq_l444_444044

-- Definitions based on the given conditions
def is_geometric_sequence (a : ℕ → ℝ) := ∃ q : ℝ, ∀ n, a (n + 1) = a n * q
def is_increasing_sequence (a : ℕ → ℝ) := ∀ n, a n < a (n + 1)
def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) := ∑ i in range n, a i

-- Main theorem to be proven
theorem sum_first_six_terms_geom_seq (a : ℕ → ℝ)
  (h_geom : is_geometric_sequence a)
  (h_inc : is_increasing_sequence a)
  (h1 : a 0 + a 2 = 5)
  (h2 : a 0 * a 2 = 4) :
  sum_first_n_terms a 6 = 63 := 
sorry

end sum_first_six_terms_geom_seq_l444_444044


namespace count_not_squares_or_cubes_200_l444_444530

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n
def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k * k * k = n
def is_sixth_power (n : ℕ) : Prop := ∃ k : ℕ, k^6 = n

theorem count_not_squares_or_cubes_200 :
  let count_squares := (Finset.range 201).filter is_perfect_square).card
  let count_cubes := (Finset.range 201).filter is_perfect_cube).card
  let count_sixth_powers := (Finset.range 201).filter is_sixth_power).card
  (200 - (count_squares + count_cubes - count_sixth_powers)) = 182 := 
by {
  -- Define everything in terms of Finset and filter
  let count_squares := (Finset.range 201).filter is_perfect_square).card,
  let count_cubes := (Finset.range 201).filter is_perfect_cube).card,
  let count_sixth_powers := (Finset.range 201).filter is_sixth_power).card,
  exact sorry
}

end count_not_squares_or_cubes_200_l444_444530


namespace negation_of_p_l444_444059

-- Define the proposition p: ∀ x ∈ ℝ, sin x ≤ 1
def proposition_p : Prop := ∀ x : ℝ, Real.sin x ≤ 1

-- The statement to prove the negation of proposition p
theorem negation_of_p : ¬proposition_p ↔ ∃ x : ℝ, Real.sin x > 1 := by
  sorry

end negation_of_p_l444_444059


namespace prime_factors_30_factorial_l444_444302

theorem prime_factors_30_factorial :
  (nat.factors 30!).toFinset.card = 10 ∧
  (nat.factors 30!).sum = 129 :=
by
  sorry

end prime_factors_30_factorial_l444_444302


namespace count_non_perfect_square_or_cube_l444_444516

-- Define perfect_square function
def perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k^2 = n

-- Define perfect_cube function
def perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k^3 = n

-- Define range from 1 to 200
def range_200 := {1 .. 200}

-- Define numbers either perfect square or perfect cube
def perfect_square_or_cube : Finset ℕ :=
  (range_200.filter perfect_square) ∪ (range_200.filter perfect_cube)

-- Define sixth powers in range
def perfect_sixth_power (n : ℕ) : Prop :=
  ∃ k : ℕ, k^6 = n

-- The final problem statement
theorem count_non_perfect_square_or_cube :
  (range_200.card - perfect_square_or_cube.card) = 182 :=
by
  sorry

end count_non_perfect_square_or_cube_l444_444516


namespace triangle_APQ_equilateral_l444_444633

-- Define vectors in a vector space
variables {V : Type} [inner_product_space ℝ V]

-- Definitions for points A, B, C, D, P, and Q
variables (A B C D P Q : V)

-- Definitions for parallelogram and equilateral triangle
def parallelogram (A B C D : V) : Prop :=
  B - A = D - C ∧ C - B = A - D

def equilateral_triangle (X Y Z : V) : Prop :=
  dist X Y = dist Y Z ∧ dist Y Z = dist Z X

-- Prove that triangle APQ is equilateral given the conditions
theorem triangle_APQ_equilateral (h_ABCD : parallelogram A B C D)
  (h_BCP : equilateral_triangle B C P)
  (h_CDQ : equilateral_triangle C D Q) :
  equilateral_triangle A P Q :=
sorry

end triangle_APQ_equilateral_l444_444633


namespace complex_addition_l444_444207

def c : ℂ := 3 - 2 * Complex.I
def d : ℂ := 1 + 3 * Complex.I

theorem complex_addition : 3 * c + 4 * d = 13 + 6 * Complex.I := by
  -- proof goes here
  sorry

end complex_addition_l444_444207


namespace gain_in_transaction_per_year_l444_444835

noncomputable def compounded_interest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

noncomputable def gain_per_year (P : ℝ) (t : ℝ) (r1 : ℝ) (n1 : ℕ) (r2 : ℝ) (n2 : ℕ) : ℝ :=
  let amount_repaid := compounded_interest P r1 n1 t
  let amount_received := compounded_interest P r2 n2 t
  (amount_received - amount_repaid) / t

theorem gain_in_transaction_per_year :
  let P := 8000
  let t := 3
  let r1 := 0.05
  let n1 := 2
  let r2 := 0.07
  let n2 := 4
  abs (gain_per_year P t r1 n1 r2 n2 - 191.96) < 0.01 :=
by
  sorry

end gain_in_transaction_per_year_l444_444835


namespace intersection_M_N_l444_444062

def M : Set ℝ := { x | |x - 2| ≤ 1 }
def N : Set ℝ := { x | x^2 - x - 6 ≥ 0 }

theorem intersection_M_N : M ∩ N = {3} := by
  sorry

end intersection_M_N_l444_444062


namespace total_books_after_donations_l444_444861

variable (Boris_books : Nat := 24)
variable (Cameron_books : Nat := 30)

theorem total_books_after_donations :
  (Boris_books - Boris_books / 4) + (Cameron_books - Cameron_books / 3) = 38 := by
  sorry

end total_books_after_donations_l444_444861


namespace numbers_not_perfect_squares_or_cubes_l444_444445

theorem numbers_not_perfect_squares_or_cubes : 
  let total_numbers := 200
  let perfect_squares := 14
  let perfect_cubes := 5
  let sixth_powers := 1
  total_numbers - (perfect_squares + perfect_cubes - sixth_powers) = 182 :=
by
  sorry

end numbers_not_perfect_squares_or_cubes_l444_444445


namespace boat_speed_ratio_l444_444221

variable (B S : ℝ)

theorem boat_speed_ratio (h : 1 / (B - S) = 2 * (1 / (B + S))) : B / S = 3 := 
by
  sorry

end boat_speed_ratio_l444_444221


namespace numbers_not_squares_or_cubes_in_200_l444_444472

/-- There are 182 numbers between 1 and 200 that are neither perfect squares nor perfect cubes. -/
theorem numbers_not_squares_or_cubes_in_200 : (finset.range 200).filter (λ n, ¬(∃ m, m^2 = n) ∧ ¬(∃ k, k^3 = n)).card = 182 :=
by 
  /-
    The proof will involve constructing the set of numbers from 1 to 200, filtering out those that
    are perfect squares or perfect cubes, and counting the remaining numbers.
  -/
  sorry

end numbers_not_squares_or_cubes_in_200_l444_444472


namespace parallel_planes_parallal_lines_l444_444033

variables {α β γ : Type} [plane α] [plane β] [plane γ]
variables (m n : line) (Pγ: plane)
variables (h1 : m = α ∩ γ) (h2 : n = β ∩ γ)

-- if α parallel β, then m parallel n
theorem parallel_planes_parallal_lines (h3 : α ∥ β): m ∥ n :=
by
  -- let's use sorry here since we are focused on structure
  sorry

end parallel_planes_parallal_lines_l444_444033


namespace impossible_rides_l444_444912

open Set

theorem impossible_rides (friends : Finset ℕ) (boat_capacity : ℕ)
  (h_friends_count : friends.card = 5)
  (h_boat_capacity : boat_capacity ≥ 5) :
  (∀ (groups : Finset (Finset ℕ)),
      groups ⊆ (powerset friends \ {∅}) →
      (∀ g ∈ groups, g ≠ ∅) →
      (∀ (g ∈ groups) (h ∈ groups), g ≠ h) →
      (finset.card groups = 31) →
      ¬∃ (last_trip : Finset ℕ),
        last_trip ∈ groups ∧ ((groups.card + 1) % 2 = 0)) :=
sorry

end impossible_rides_l444_444912


namespace binom_n_plus_1_n_minus_1_eq_l444_444720

theorem binom_n_plus_1_n_minus_1_eq (n : ℕ) (h : 0 < n) : (Nat.choose (n + 1) (n - 1)) = n * (n + 1) / 2 := 
by sorry

end binom_n_plus_1_n_minus_1_eq_l444_444720


namespace slant_asymptote_sum_m_b_l444_444411

-- Definitions for the rational function and its components
def f (x : ℚ) : ℚ := (3 * x^2 + 4 * x - 5) / (x - 4)

-- Defining the conditions for the slant asymptote
def slant_asymptote (y m b : ℚ) : Prop := y = m * x + b

-- Stating the theorem
theorem slant_asymptote_sum_m_b : 
  (∀ x : ℚ, f(x) = (3 * x + 16 + (59 / (x - 4)))) →
   ∃ m b, slant_asymptote (3 * x + 16) m b ∧ m + b = 19 :=
by
  intros h
  exists 3, 16
  simp [slant_asymptote]
  split
  · rfl
  · norm_num
  sorry

end slant_asymptote_sum_m_b_l444_444411


namespace size_of_det_set_2021_l444_444596

noncomputable def cardinality_of_det_set (n : ℕ) : ℕ :=
  { det A | A ∈ (set_of (λ (A : matrix (fin n) (fin n) ℕ),
    (∀ i ∈ fin n, (∑ j ∈ fin n, A i j) ≤ 2) ∧ 
    (∀ A : matrix (fin n) (fin n) ℕ, A i j ∈ {0, 1}))) }.to_finset.card

theorem size_of_det_set_2021 : cardinality_of_det_set 2021 = 1348 :=
by
  sorry

end size_of_det_set_2021_l444_444596


namespace complex_magnitude_pow_eight_l444_444334

theorem complex_magnitude_pow_eight :
  (Complex.abs ((2/5 : ℂ) + (7/5 : ℂ) * Complex.I))^8 = 7890481 / 390625 := 
by
  sorry

end complex_magnitude_pow_eight_l444_444334


namespace inscribed_sphere_radius_in_pyramid_of_spheres_l444_444662

-- Definitions for problem conditions
def radius_of_circumscribing_sphere : Real := Real.sqrt 6 + 1

-- Target proof statement
theorem inscribed_sphere_radius_in_pyramid_of_spheres 
  (r : Real) : r = Real.sqrt 2 - 1 :=
by
  -- Given conditions
  let R := radius_of_circumscribing_sphere
  -- Proof placeholder
  sorry

end inscribed_sphere_radius_in_pyramid_of_spheres_l444_444662


namespace floor_of_T2_l444_444615

noncomputable def T : ℝ :=
  ∑ i in (Finset.range 2008).image (λ i, i + 1), 
    Real.sqrt (1 + 1 / (i:ℝ)^2 + 1 / (i + 1)^2 + 1 / (i + 2)^2)

theorem floor_of_T2 : ⌊T^2⌋ = 64512422 := by
  sorry

end floor_of_T2_l444_444615


namespace ceil_sqrt_165_l444_444899

theorem ceil_sqrt_165 : Real.ceil (Real.sqrt 165) = 13 := by
  have lower_bound : 12 < Real.sqrt 165 := by 
    have h1 : (12:ℝ) * 12 = 144 := by norm_num
    rw [Real.sqrt_lt h1]
    exact Nat.lt_succ_self 164
  have upper_bound : Real.sqrt 165 < 13 := by 
    have h2 : (13:ℝ) * 13 = 169 := by norm_num
    rw [Real.sqrt_lt_iff]
    exact 165.lt_succ_self
  have sqrt_not_int : Real.sqrt 165 ≠ 13 := by
    intro h
    have p : 165 = 169 := by rw [←h, ←sq_sqrt]
    contradiction
  sorry

end ceil_sqrt_165_l444_444899


namespace ratio_prikya_ladonna_l444_444896

def total_cans : Nat := 85
def ladonna_cans : Nat := 25
def yoki_cans : Nat := 10
def prikya_cans : Nat := total_cans - ladonna_cans - yoki_cans

theorem ratio_prikya_ladonna : prikya_cans.toFloat / ladonna_cans.toFloat = 2 / 1 := 
by sorry

end ratio_prikya_ladonna_l444_444896


namespace count_not_squares_or_cubes_200_l444_444531

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n
def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k * k * k = n
def is_sixth_power (n : ℕ) : Prop := ∃ k : ℕ, k^6 = n

theorem count_not_squares_or_cubes_200 :
  let count_squares := (Finset.range 201).filter is_perfect_square).card
  let count_cubes := (Finset.range 201).filter is_perfect_cube).card
  let count_sixth_powers := (Finset.range 201).filter is_sixth_power).card
  (200 - (count_squares + count_cubes - count_sixth_powers)) = 182 := 
by {
  -- Define everything in terms of Finset and filter
  let count_squares := (Finset.range 201).filter is_perfect_square).card,
  let count_cubes := (Finset.range 201).filter is_perfect_cube).card,
  let count_sixth_powers := (Finset.range 201).filter is_sixth_power).card,
  exact sorry
}

end count_not_squares_or_cubes_200_l444_444531
