import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shorter_diagonal_is_sqrt_49_l536_53639

/-- Represents a trapezoid with given side lengths -/
structure Trapezoid where
  ef : ℝ
  gh : ℝ
  eg : ℝ
  fh : ℝ
  ef_parallel_gh : ef > gh
  e_acute : True
  f_acute : True

/-- The length of the shorter diagonal of the trapezoid -/
noncomputable def shorter_diagonal (t : Trapezoid) : ℝ := Real.sqrt 49

/-- Theorem stating that the shorter diagonal of the specified trapezoid is √49 -/
theorem shorter_diagonal_is_sqrt_49 (t : Trapezoid) 
  (h1 : t.ef = 25) 
  (h2 : t.gh = 15) 
  (h3 : t.eg = 13) 
  (h4 : t.fh = 17) : 
  shorter_diagonal t = Real.sqrt 49 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shorter_diagonal_is_sqrt_49_l536_53639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_non_attacking_rooks_l536_53648

/-- Represents a chessboard with a cut-out square -/
structure CutoutBoard where
  size : Nat
  cutout_size : Nat
  cutout_start_row : Nat
  cutout_start_col : Nat

/-- Checks if a position is within the cut-out area -/
def is_in_cutout (board : CutoutBoard) (row col : Nat) : Prop :=
  board.cutout_start_row ≤ row ∧ 
  row < board.cutout_start_row + board.cutout_size ∧
  board.cutout_start_col ≤ col ∧ 
  col < board.cutout_start_col + board.cutout_size

/-- Represents a rook placement on the board -/
structure RookPlacement where
  row : Nat
  col : Nat

/-- Checks if two rook placements are non-attacking -/
def are_non_attacking (board : CutoutBoard) (r1 r2 : RookPlacement) : Prop :=
  (r1.row ≠ r2.row ∧ r1.col ≠ r2.col) ∨
  (is_in_cutout board r1.row r1.col ∧ is_in_cutout board r2.row r2.col)

/-- The main theorem statement -/
theorem max_non_attacking_rooks (board : CutoutBoard) 
  (h_size : board.size = 12)
  (h_cutout_size : board.cutout_size = 4)
  (h_cutout_start_row : board.cutout_start_row = 2)
  (h_cutout_start_col : board.cutout_start_col = 2) :
  ∃ (placements : List RookPlacement),
    placements.length = 14 ∧
    (∀ r, r ∈ placements → r.row < board.size ∧ r.col < board.size) ∧
    (∀ r1 r2, r1 ∈ placements → r2 ∈ placements → r1 ≠ r2 → are_non_attacking board r1 r2) ∧
    (∀ (other_placements : List RookPlacement),
      (∀ r, r ∈ other_placements → r.row < board.size ∧ r.col < board.size) →
      (∀ r1 r2, r1 ∈ other_placements → r2 ∈ other_placements → r1 ≠ r2 → are_non_attacking board r1 r2) →
      other_placements.length ≤ 14) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_non_attacking_rooks_l536_53648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcenter_is_intersection_of_perpendicular_bisectors_l536_53626

/-- A triangle in a 2D plane -/
structure Triangle where
  A : EuclideanSpace ℝ (Fin 2)
  B : EuclideanSpace ℝ (Fin 2)
  C : EuclideanSpace ℝ (Fin 2)

/-- The circumcenter of a triangle -/
noncomputable def circumcenter (t : Triangle) : EuclideanSpace ℝ (Fin 2) := sorry

/-- The perpendicular bisector of a line segment -/
noncomputable def perpendicularBisector (A B : EuclideanSpace ℝ (Fin 2)) : Set (EuclideanSpace ℝ (Fin 2)) := sorry

/-- The angle bisector of an angle -/
noncomputable def angleBisector (A B C : EuclideanSpace ℝ (Fin 2)) : Set (EuclideanSpace ℝ (Fin 2)) := sorry

/-- Theorem: The circumcenter of a triangle is the intersection point of the perpendicular bisectors of its sides, not the angle bisectors -/
theorem circumcenter_is_intersection_of_perpendicular_bisectors (t : Triangle) : 
  circumcenter t ∈ (perpendicularBisector t.A t.B) ∩ (perpendicularBisector t.B t.C) ∧
  circumcenter t ∉ (angleBisector t.B t.A t.C) ∩ (angleBisector t.A t.B t.C) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcenter_is_intersection_of_perpendicular_bisectors_l536_53626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_chord_length_l536_53641

noncomputable section

/-- Curve C₁ in polar coordinates -/
def C₁ (θ : ℝ) : ℝ := 6 * Real.cos θ

/-- Curve C₂ in polar coordinates -/
def C₂ : ℝ := Real.pi / 4

/-- The length of the chord formed by the intersection of C₁ and C₂ -/
def chordLength : ℝ := 3 * Real.sqrt 2

theorem intersection_chord_length :
  let A := (C₁ C₂ * Real.cos C₂, C₁ C₂ * Real.sin C₂)
  let B := (C₁ (-C₂) * Real.cos (-C₂), C₁ (-C₂) * Real.sin (-C₂))
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = chordLength^2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_chord_length_l536_53641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l536_53682

/-- The function f(x) with parameter ω -/
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x - 3 * Real.pi / 4)

/-- Theorem stating the properties of f and the values of ω and tan(α) -/
theorem function_properties (ω : ℝ) (α : ℝ) 
  (h_ω_pos : ω > 0)
  (h_period : ∀ x, f ω (x + Real.pi / ω) = f ω x)
  (h_min_period : ∀ T, T > 0 → (∀ x, f ω (x + T) = f ω x) → T ≥ Real.pi / ω)
  (h_f_value : f ω (α / 2 + 3 * Real.pi / 8) = 24 / 25)
  (h_α_range : α > -Real.pi / 2 ∧ α < Real.pi / 2) :
  ω = 2 ∧ Real.tan α = 24 / 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l536_53682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_approx_092y_l536_53671

-- Define x and y as real numbers
noncomputable def x : ℝ := Real.log 375 / Real.log 16

noncomputable def y : ℝ := Real.log 25 / Real.log 4

-- Define an approximation threshold
def ε : ℝ := 0.01

-- Theorem statement
theorem x_approx_092y : ∃ δ : ℝ, δ < ε ∧ |x - 0.92 * y| < δ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_approx_092y_l536_53671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_Q_less_than_threshold_l536_53607

open BigOperators

def Q (n : ℕ) : ℚ :=
  1 / ((n^2 + 1 : ℚ) * (∏ k in Finset.range (n-1), ((k+1)^2 + 1 : ℚ)))

theorem smallest_n_for_Q_less_than_threshold : 
  (∀ m : ℕ, m < 63 → Q m ≥ 1/4020) ∧ Q 63 < 1/4020 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_Q_less_than_threshold_l536_53607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_product_l536_53661

theorem binomial_product : (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_product_l536_53661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_fraction_l536_53610

/-- The area of a right triangle formed by connecting the bottom-left corner, 
    top-left corner, and mid-point on the right edge of a 6 by 6 grid 
    is 1/4 of the total grid area. -/
theorem triangle_area_fraction (grid_size : ℕ) (h : grid_size = 6) : 
  (((grid_size * (grid_size / 2)) / 2) : ℚ) / (grid_size ^ 2) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_fraction_l536_53610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_derivative_of_y_l536_53645

noncomputable def y (x : ℝ) : ℝ := (4 * x^3 + 5) * Real.exp (2 * x + 1)

theorem fifth_derivative_of_y (x : ℝ) :
  (deriv^[5] y) x = 32 * (4 * x^3 + 30 * x^2 + 60 * x + 35) * Real.exp (2 * x + 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_derivative_of_y_l536_53645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_exact_range_of_f_l536_53623

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.cos x + Real.sin x

-- State the theorem
theorem range_of_f :
  ∀ y ∈ Set.range f,
  -Real.sqrt 3 ≤ y ∧ y ≤ 2 ∧
  ∃ x ∈ Set.Icc (-π/3) π, f x = y :=
by
  sorry

-- Additional theorem to state the exact range
theorem exact_range_of_f :
  Set.range f = Set.Icc (-Real.sqrt 3) 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_exact_range_of_f_l536_53623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_y_l536_53664

noncomputable def y (x : ℝ) : ℝ := Real.sqrt (Real.tan 4) + (Real.sin (21 * x))^2 / (21 * Real.cos (42 * x))

theorem derivative_y (x : ℝ) :
  deriv y x = 2 * Real.tan (42 * x) * (1 / Real.cos (42 * x)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_y_l536_53664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_product_lower_bound_l536_53677

theorem tan_product_lower_bound (α β γ : Real) 
  (h_acute_α : 0 < α ∧ α < Real.pi / 2)
  (h_acute_β : 0 < β ∧ β < Real.pi / 2)
  (h_acute_γ : 0 < γ ∧ γ < Real.pi / 2)
  (h_cos_sum : Real.cos α ^ 2 + Real.cos β ^ 2 + Real.cos γ ^ 2 = 1) :
  Real.tan α * Real.tan β * Real.tan γ ≥ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_product_lower_bound_l536_53677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_equals_one_l536_53678

-- Define the circle
def circleO (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define point M
def M : ℝ × ℝ := (0, 2)

-- Define the arc length
noncomputable def arc_length : ℝ := Real.pi / 2

-- Define angle α
noncomputable def α (N : ℝ × ℝ) : ℝ := Real.arctan (N.2 / N.1)

-- Theorem statement
theorem tan_alpha_equals_one (N : ℝ × ℝ) :
  circleO N.1 N.2 →
  (N.1 - M.1)^2 + (N.2 - M.2)^2 = 2 * (1 - Real.cos arc_length) →
  Real.tan (α N) = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_equals_one_l536_53678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circumradius_l536_53644

/-- The radius of the circumcircle of a triangle with side lengths 8, 10, and 14 -/
theorem triangle_circumradius : (a * b * c) / (4 * area) = 35 * Real.sqrt 2 / 3 :=
  let a : ℝ := 8
  let b : ℝ := 10
  let c : ℝ := 14
  let s : ℝ := (a + b + c) / 2
  let area : ℝ := Real.sqrt (s * (s - a) * (s - b) * (s - c))
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circumradius_l536_53644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l536_53633

theorem circle_equation (r : ℝ) (h1 : r = 6) :
  ∃ (a : ℝ), (∀ (x y : ℝ), 
    ((x - a)^2 + (y - r)^2 = r^2) ∧
    (∃ (x₀ : ℝ), (x₀ - a)^2 + r^2 = r^2) ∧
    (∃ (x₁ y₁ : ℝ), (x₁ - a)^2 + (y₁ - r)^2 = r^2 ∧ x₁^2 + (y₁ - 3)^2 = 1 ∧
      (x₁ - 0)^2 + (y₁ - 3)^2 = (r - 1)^2)) →
    a = 4 ∨ a = -4 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l536_53633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sally_shots_theorem_l536_53605

def successful_shots_in_last_ten (initial_shots : ℕ) (initial_rate : ℚ) 
  (total_shots : ℕ) (new_rate : ℚ) : ℤ :=
  (new_rate * total_shots).floor - (initial_rate * initial_shots).floor

theorem sally_shots_theorem (initial_shots : ℕ) (initial_rate : ℚ) 
  (total_shots : ℕ) (new_rate : ℚ) 
  (h1 : initial_shots = 30)
  (h2 : initial_rate = 3/5)
  (h3 : total_shots = 40)
  (h4 : new_rate = 13/20) :
  successful_shots_in_last_ten initial_shots initial_rate total_shots new_rate = 8 := by
  sorry

#eval successful_shots_in_last_ten 30 (3/5) 40 (13/20)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sally_shots_theorem_l536_53605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_average_after_adding_numbers_l536_53631

theorem new_average_after_adding_numbers 
  (original_list : List ℚ) 
  (h_length : original_list.length = 10) 
  (h_mean : original_list.sum / (original_list.length : ℚ) = 0) : 
  let new_list := original_list ++ [72, -12]
  new_list.sum / (new_list.length : ℚ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_average_after_adding_numbers_l536_53631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_tangent_line_properties_l536_53651

noncomputable section

variable (x : ℝ)

def f (a b : ℝ) (x : ℝ) : ℝ := (a * Real.log x) / (x + 1) + b / x

def tangent_line (x y : ℝ) : Prop := x + 2 * y - 3 = 0

theorem function_and_tangent_line_properties :
  (∃ a b : ℝ, (∀ x > 1, tangent_line x (f a b x))) →
  (∃ a b : ℝ, a = 1 ∧ b = 1) ∧
  (∀ k : ℝ, (∀ x > 1, f 1 1 x > Real.log x / (x - 1) + k / x) → k ≤ 0) :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_tangent_line_properties_l536_53651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_prism_volume_l536_53614

/-- Represents a right triangular prism with a circumscribed sphere -/
structure RightTriangularPrism where
  -- Base legs of the right triangle
  a : ℝ
  b : ℝ
  -- Height of the prism (equal to the hypotenuse of the base)
  h : ℝ
  -- Radius of the circumscribed sphere
  r : ℝ
  -- Conditions
  h_eq_hypotenuse : h^2 = a^2 + b^2
  sphere_volume : (4/3) * Real.pi * r^3 = 32 * Real.pi / 3
  h_eq_2sqrt2 : h = 2 * Real.sqrt 2

/-- The maximum volume of the right triangular prism -/
noncomputable def maxPrismVolume (p : RightTriangularPrism) : ℝ := 4 * Real.sqrt 2

/-- Theorem stating that the volume of the prism is less than or equal to 4√2 -/
theorem max_prism_volume (p : RightTriangularPrism) :
  p.a * p.b * p.h ≤ maxPrismVolume p := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_prism_volume_l536_53614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_greater_than_N_l536_53604

theorem M_greater_than_N (a : ℝ) : 
  (2*a^2 - 4*a) > (a^2 - 2*a - 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_greater_than_N_l536_53604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PQR_l536_53618

-- Define the point P
noncomputable def P : ℝ × ℝ := (2, 5)

-- Define the slopes of the two lines
noncomputable def slope1 : ℝ := 3
noncomputable def slope2 : ℝ := -1

-- Define Q and R as the x-intercepts of the lines
noncomputable def Q : ℝ × ℝ := (1/3, 0)
noncomputable def R : ℝ × ℝ := (7, 0)

-- Define the area of triangle PQR
noncomputable def area_PQR : ℝ := 50/3

-- Theorem statement
theorem area_of_triangle_PQR :
  let line1 := fun x => slope1 * (x - P.1) + P.2
  let line2 := fun x => slope2 * (x - P.1) + P.2
  (line1 Q.1 = Q.2) ∧ (line2 R.1 = R.2) →
  (1/2) * (R.1 - Q.1) * P.2 = area_PQR :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PQR_l536_53618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_inverse_and_ratio_l536_53696

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := (3*x - 2) / (x + 3)

-- Define the inverse function g_inv
noncomputable def g_inv (x : ℝ) : ℝ := (3*x + 2) / (-x + 3)

-- Theorem statement
theorem g_inverse_and_ratio :
  (∀ x, g (g_inv x) = x) ∧ 
  (∀ x, g_inv (g x) = x) ∧
  (3 : ℝ) / (-1 : ℝ) = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_inverse_and_ratio_l536_53696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_sets_with_union_l536_53628

theorem count_sets_with_union (S T : Set ℕ) : 
  S = {1, 3} → T = {1, 3, 5} → 
  (∃! (n : ℕ), ∃ (collection : Finset (Finset ℕ)), 
    (∀ A ∈ collection, S ∪ A = T) ∧ 
    (Finset.card collection = n) ∧ 
    n = 4) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_sets_with_union_l536_53628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_unchanged_by_constant_addition_l536_53647

noncomputable def variance (x : Fin 3 → ℝ) : ℝ :=
  let μ := (x 0 + x 1 + x 2) / 3
  ((x 0 - μ)^2 + (x 1 - μ)^2 + (x 2 - μ)^2) / 3

theorem variance_unchanged_by_constant_addition
  (x : Fin 3 → ℝ) (h : variance x = 5) :
  variance (fun i => x i + 1) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_unchanged_by_constant_addition_l536_53647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_square_l536_53613

theorem functional_equation_square (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (f x - y^2) = (f x)^2 - 2*(f x)*(y^2) + f (f y)) : 
  ∀ x : ℝ, f x = x^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_square_l536_53613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_wing_floors_l536_53665

/-- The number of floors in the first wing of the hotel -/
def F : ℕ := 9

/-- The number of halls per floor in the first wing -/
def halls_first_wing : ℕ := 6

/-- The number of rooms per hall in the first wing -/
def rooms_per_hall_first_wing : ℕ := 32

/-- The number of floors in the second wing -/
def floors_second_wing : ℕ := 7

/-- The number of halls per floor in the second wing -/
def halls_second_wing : ℕ := 9

/-- The number of rooms per hall in the second wing -/
def rooms_per_hall_second_wing : ℕ := 40

/-- The total number of rooms in the hotel -/
def total_rooms : ℕ := 4248

theorem first_wing_floors : 
  F * halls_first_wing * rooms_per_hall_first_wing + 
  floors_second_wing * halls_second_wing * rooms_per_hall_second_wing = total_rooms := by
  -- Proof goes here
  sorry

#eval F

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_wing_floors_l536_53665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l536_53660

noncomputable def g (x : ℝ) : ℝ := (3^x - 2) / (3^x + 2)

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x := by
  intro x
  simp [g]
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l536_53660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_pairs_basis_l536_53634

noncomputable def vector_pair_1 : (ℝ × ℝ) × (ℝ × ℝ) := ((-1, 2), (5, 7))
noncomputable def vector_pair_2 : (ℝ × ℝ) × (ℝ × ℝ) := ((3, 5), (6, 10))
noncomputable def vector_pair_3 : (ℝ × ℝ) × (ℝ × ℝ) := ((2, -3), (1/2, 3/4))

def is_basis (v : (ℝ × ℝ) × (ℝ × ℝ)) : Prop :=
  let (v1, v2) := v
  let (x1, y1) := v1
  let (x2, y2) := v2
  x1 * y2 - x2 * y1 ≠ 0

theorem vector_pairs_basis :
  is_basis vector_pair_1 ∧
  ¬is_basis vector_pair_2 ∧
  is_basis vector_pair_3 := by
  sorry

#check vector_pairs_basis

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_pairs_basis_l536_53634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_one_iff_a_greater_than_four_l536_53669

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then (1/2) * x - 1 else 1/x

-- Theorem statement
theorem f_greater_than_one_iff_a_greater_than_four :
  ∀ a : ℝ, f a > 1 ↔ a > 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_one_iff_a_greater_than_four_l536_53669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_log_range_l536_53612

-- Define the conditions
def is_arithmetic_sequence (x y : ℝ) : Prop := 2 * y = x + (x + y)

def is_geometric_sequence (x y : ℝ) : Prop := y^2 = x * (x * y)

def log_condition (m : ℝ) (x y : ℝ) : Prop := 0 < Real.log (x * y) / Real.log m ∧ Real.log (x * y) / Real.log m < 1

-- State the theorem
theorem sequence_log_range (x y m : ℝ) :
  is_arithmetic_sequence x y →
  is_geometric_sequence x y →
  log_condition m x y →
  m > 8 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_log_range_l536_53612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_angles_l536_53638

open Real

theorem tan_sum_angles (α β : ℝ) :
  sin α = -4/5 →
  α ∈ Set.Icc (3*π/2) (2*π) →
  sin (α + β) / cos β = 2 →
  tan (α + β) = 6/13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_angles_l536_53638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_adjustment_amount_correct_adjustment_results_in_equality_l536_53606

/-- Represents the amount paid by each person -/
structure TripExpenses where
  leroy : ℚ
  bernardo : ℚ
  carlos : ℚ

/-- Calculates the amount LeRoy needs to adjust to equalize expenses -/
noncomputable def adjustmentAmount (expenses : TripExpenses) : ℚ :=
  (expenses.bernardo + expenses.carlos - 2 * expenses.leroy) / 3

/-- Theorem stating that the adjustment amount is correct -/
theorem adjustment_amount_correct (expenses : TripExpenses) :
  let totalExpenses := expenses.leroy + expenses.bernardo + expenses.carlos
  let equalShare := totalExpenses / 3
  adjustmentAmount expenses = equalShare - expenses.leroy := by
  sorry

/-- Theorem proving that the adjustment amount results in equal contributions -/
theorem adjustment_results_in_equality (expenses : TripExpenses) :
  let totalExpenses := expenses.leroy + expenses.bernardo + expenses.carlos
  let equalShare := totalExpenses / 3
  equalShare = expenses.leroy + adjustmentAmount expenses := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_adjustment_amount_correct_adjustment_results_in_equality_l536_53606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_integers_square_sum_l536_53689

theorem consecutive_integers_square_sum (x : ℤ) :
  (x^2 + (x+1)^2 + (x+2)^2 = (x+3)^2 + (x+4)^2) →
  ((x = -2 ∧ 
    List.map (λ i => x + i) (List.range 5) = [-2, -1, 0, 1, 2]) ∨
   (x = 10 ∧ 
    List.map (λ i => x + i) (List.range 5) = [10, 11, 12, 13, 14])) :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_integers_square_sum_l536_53689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cp_length_for_specific_case_lambda_range_when_cp_dot_ab_ge_pa_dot_pb_l536_53691

/-- Represents an equilateral triangle ABC with a point P on AB -/
structure EquilateralTriangleWithPoint where
  a : ℝ  -- Side length of the triangle
  lambda : ℝ  -- Scalar for point P on AB
  h1 : 0 ≤ lambda
  h2 : lambda ≤ 1

/-- The squared length of CP in the equilateral triangle -/
noncomputable def CP_squared (t : EquilateralTriangleWithPoint) : ℝ :=
  t.a^2 - 2 * t.a^2 * t.lambda + (t.a * t.lambda)^2

/-- The dot product of CP and AB -/
noncomputable def CP_dot_AB (t : EquilateralTriangleWithPoint) : ℝ :=
  -1/2 * t.a^2 + t.lambda * t.a^2

/-- The dot product of PA and PB -/
noncomputable def PA_dot_PB (t : EquilateralTriangleWithPoint) : ℝ :=
  -t.lambda * t.a^2 + t.lambda^2 * t.a^2

theorem cp_length_for_specific_case :
  ∀ t : EquilateralTriangleWithPoint, t.a = 6 → t.lambda = 1/3 → CP_squared t = 28 :=
by
  sorry

theorem lambda_range_when_cp_dot_ab_ge_pa_dot_pb :
  ∀ t : EquilateralTriangleWithPoint, 
  CP_dot_AB t ≥ PA_dot_PB t → 
  (2 - Real.sqrt 2) / 2 ≤ t.lambda ∧ t.lambda ≤ 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cp_length_for_specific_case_lambda_range_when_cp_dot_ab_ge_pa_dot_pb_l536_53691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_center_to_point_l536_53625

/-- The circle C in polar coordinates -/
def circle_C (ρ θ : ℝ) : Prop := ρ = 2 * Real.sin θ

/-- The center of circle C -/
def center_C : ℝ × ℝ := (0, 1)

/-- The point we're measuring distance to -/
def point : ℝ × ℝ := (1, 0)

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem distance_from_center_to_point :
  distance center_C point = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_center_to_point_l536_53625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_second_quadrant_l536_53657

noncomputable section

/-- The quadrant of an angle --/
inductive Quadrant
  | I
  | II
  | III
  | IV

/-- Function to determine the quadrant of an angle --/
def QuadrantOf (θ : Real) : Quadrant :=
  if 0 ≤ θ ∧ θ < Real.pi/2 then Quadrant.I
  else if Real.pi/2 ≤ θ ∧ θ < Real.pi then Quadrant.II
  else if Real.pi ≤ θ ∧ θ < 3*Real.pi/2 then Quadrant.III
  else Quadrant.IV

theorem angle_in_second_quadrant :
  ∀ θ : Real, θ = 3 → Real.pi / 2 < θ ∧ θ < Real.pi → QuadrantOf θ = Quadrant.II :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_second_quadrant_l536_53657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_chord_length_l536_53635

noncomputable section

/-- Defines an ellipse E with equation x²/4 + y² = 1 -/
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2 = 1

/-- Defines a point P on the ellipse -/
def point_on_ellipse : Prop := ellipse (-Real.sqrt 3) (1/2)

/-- Defines one focus of the ellipse -/
def focus : Prop := ∃ (c : ℝ), c^2 = 3 ∧ ellipse c 0

/-- Defines a line passing through point M(0, √2) -/
def line (k : ℝ) (x y : ℝ) : Prop := y = k * x + Real.sqrt 2

/-- Defines the length of a chord AB formed by the intersection of the line and the ellipse -/
noncomputable def chord_length (k : ℝ) : ℝ := 
  2 * Real.sqrt (-6 * (1 / (1 + 4 * k^2))^2 + 1 / (1 + 4 * k^2) + 1)

/-- The main theorem stating the maximum chord length -/
theorem max_chord_length : 
  point_on_ellipse → focus → ∃ (max_length : ℝ), 
    (∀ k, chord_length k ≤ max_length) ∧ 
    max_length = 5 * Real.sqrt 6 / 6 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_chord_length_l536_53635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_value_fourth_quadrant_l536_53694

theorem sin_value_fourth_quadrant (θ : Real) (h1 : Real.cos θ = 1/3) 
  (h2 : θ ∈ Set.Icc (3*Real.pi/2) (2*Real.pi)) : 
  Real.sin θ = - (2 * Real.sqrt 2) / 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_value_fourth_quadrant_l536_53694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l536_53686

/-- Given a hyperbola with the equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    left focus F₁(-c, 0), right focus F₂(c, 0), and points M and N on the hyperbola
    such that MN is parallel to F₁F₂ and |F₁F₂| = 3|MN|. Additionally, point Q
    is on the hyperbola and is the midpoint of F₁N. 
    This theorem states that the eccentricity of the hyperbola is 3. -/
theorem hyperbola_eccentricity (a b c : ℝ) (M N Q : ℝ × ℝ) :
  a > 0 → b > 0 →
  (∀ x y, x^2 / a^2 - y^2 / b^2 = 1 ↔ (x, y) ∈ ({M, N, Q} : Set (ℝ × ℝ))) →
  (M.1 - N.1) / (M.2 - N.2) = (-c - c) / 0 →
  (c - (-c)) = 3 * Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) →
  Q = ((N.1 + (-c)) / 2, N.2 / 2) →
  c / a = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l536_53686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l536_53688

/-- A geometric sequence with first term a and common ratio q -/
def geometric_sequence (a q : ℝ) : ℕ → ℝ := λ n ↦ a * q^(n - 1)

/-- The geometric sequence is increasing -/
def is_increasing_sequence (s : ℕ → ℝ) : Prop := ∀ n, s n < s (n + 1)

theorem geometric_sequence_ratio (a q : ℝ) :
  a > 0 →
  is_increasing_sequence (geometric_sequence a q) →
  2 * ((geometric_sequence a q 4) + (geometric_sequence a q 6)) = 5 * (geometric_sequence a q 5) →
  q = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l536_53688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_digit_of_fraction_seventh_nineteenth_587th_digit_l536_53615

theorem decimal_digit_of_fraction (n : ℕ) (a b : ℕ) (h : b ≠ 0) :
  ∃ (d : ℕ) (cycle : List ℕ), 
    (∀ i, i < cycle.length → cycle.get ⟨i, by sorry⟩ < 10) ∧ 
    (∀ k, (a * 10^(k + 1) / b) % 10 = cycle.get ⟨k % cycle.length, by sorry⟩) →
    (a * 10^n / b) % 10 = d :=
by sorry

theorem seventh_nineteenth_587th_digit : 
  ∃ (d : ℕ) (cycle : List ℕ), 
    (∀ i, i < cycle.length → cycle.get ⟨i, by sorry⟩ < 10) ∧
    (∀ k, (7 * 10^(k + 1) / 19) % 10 = cycle.get ⟨k % cycle.length, by sorry⟩) ∧
    (7 * 10^587 / 19) % 10 = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_digit_of_fraction_seventh_nineteenth_587th_digit_l536_53615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_property_l536_53602

/-- Given a triangle PQR with vertices P(3,5), Q(7,-1), and R(9,3),
    if S(x,y) is the centroid of the triangle, then 10x + y = 197/3 -/
theorem centroid_property (x y : ℚ) : 
  let P : ℚ × ℚ := (3, 5)
  let Q : ℚ × ℚ := (7, -1)
  let R : ℚ × ℚ := (9, 3)
  let S : ℚ × ℚ := (x, y)
  (S.1 = (P.1 + Q.1 + R.1) / 3 ∧ S.2 = (P.2 + Q.2 + R.2) / 3) →
  10 * x + y = 197 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_property_l536_53602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_parabola_intersection_l536_53670

-- Define the hyperbola M
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ y^2 / a^2 - x^2 / b^2 = 1

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 16 * y

-- Define the focus of the parabola
def parabola_focus : ℝ × ℝ := (0, 4)

-- Define the focus of the hyperbola
noncomputable def hyperbola_focus (a b : ℝ) : ℝ × ℝ := (0, Real.sqrt (a^2 + b^2))

-- Define the length of the imaginary axis
def imaginary_axis_length (b : ℝ) : ℝ := 2 * b

-- Theorem statement
theorem hyperbola_parabola_intersection (a b : ℝ) :
  (∀ x y, hyperbola a b x y → parabola x y) →
  (hyperbola_focus a b = parabola_focus) →
  (imaginary_axis_length b = 4) →
  (∀ x y, hyperbola a b x y ↔ y^2 / 12 - x^2 / 4 = 1) ∧
  ¬(∃ k m : ℝ, ∃ x₁ y₁ x₂ y₂ : ℝ,
    hyperbola a b x₁ y₁ ∧ hyperbola a b x₂ y₂ ∧
    y₁ = k * x₁ + m ∧ y₂ = k * x₂ + m ∧
    (x₁ + x₂) / 2 = 1 ∧ (y₁ + y₂) / 2 = 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_parabola_intersection_l536_53670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nearest_multiple_of_21_to_2304_l536_53642

theorem nearest_multiple_of_21_to_2304 :
  ∀ n : ℤ, n ≠ 2310 → n % 21 = 0 → |n - 2304| ≥ |2310 - 2304| := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nearest_multiple_of_21_to_2304_l536_53642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_terminal_side_l536_53609

theorem point_on_terminal_side (α : ℝ) (y : ℝ) :
  (∃ P : ℝ × ℝ, P = (3, y) ∧ P.1 = 3 * Real.cos α ∧ P.2 = 3 * Real.sin α) →
  Real.cos α = 3/5 →
  y = 4 ∨ y = -4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_terminal_side_l536_53609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divide_segment_with_parallel_lines_l536_53681

-- Define the basic geometric objects
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

def Segment (A B : Point) : Set Point := {P : Point | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P.x = A.x + t * (B.x - A.x) ∧ P.y = A.y + t * (B.y - A.y)}

-- Define parallel lines
def Parallel (L1 L2 : Line) : Prop := L1.a * L2.b = L1.b * L2.a

-- Define a point lying on a line
def PointOnLine (P : Point) (L : Line) : Prop := L.a * P.x + L.b * P.y + L.c = 0

-- Define the distance between two points
noncomputable def distance (P Q : Point) : ℝ := Real.sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2)

-- Define the concept of dividing a segment into n equal parts
def DivideSegmentEqually (S : Set Point) (n : ℕ) (divisionPoints : List Point) : Prop :=
  (divisionPoints.length = n - 1) ∧
  (∀ i j, i < j → i < n → j ≤ n → 
    (distance (divisionPoints.get? (i-1) |>.getD (Point.mk 0 0)) (divisionPoints.get? i |>.getD (Point.mk 0 0)) = 
     distance (divisionPoints.get? (j-1) |>.getD (Point.mk 0 0)) (divisionPoints.get? j |>.getD (Point.mk 0 0))))

-- State the theorem
theorem divide_segment_with_parallel_lines 
  (L1 L2 : Line) 
  (A B : Point) 
  (n : ℕ) 
  (h1 : Parallel L1 L2) 
  (h2 : PointOnLine A L1) 
  (h3 : PointOnLine B L1) :
  ∃ (divisionPoints : List Point), DivideSegmentEqually (Segment A B) n divisionPoints := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_divide_segment_with_parallel_lines_l536_53681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tin_in_new_alloy_l536_53668

/-- Represents the composition of an alloy -/
structure Alloy where
  total_weight : ℝ
  ratio_1 : ℝ
  ratio_2 : ℝ

/-- Calculates the weight of a component in an alloy given its ratio -/
noncomputable def weight_of_component (a : Alloy) (ratio : ℝ) : ℝ :=
  (ratio / (a.ratio_1 + a.ratio_2)) * a.total_weight

/-- The problem statement -/
theorem tin_in_new_alloy (alloy_a alloy_b : Alloy) 
    (h1 : alloy_a.total_weight = 170)
    (h2 : alloy_b.total_weight = 250)
    (h3 : alloy_a.ratio_1 = 1 ∧ alloy_a.ratio_2 = 3)
    (h4 : alloy_b.ratio_1 = 3 ∧ alloy_b.ratio_2 = 5) :
    weight_of_component alloy_a alloy_a.ratio_2 + weight_of_component alloy_b alloy_b.ratio_1 = 221.25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tin_in_new_alloy_l536_53668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sphere_surface_area_is_18pi_l536_53611

/-- A rectangular solid with volume 4 and one face area 1, inscribed in a sphere -/
structure InscribedSolid where
  /-- The length of the solid -/
  a : ℝ
  /-- The width of the solid -/
  b : ℝ
  /-- The height of the solid -/
  c : ℝ
  /-- The volume of the solid is 4 -/
  volume_eq_4 : a * b * c = 4
  /-- One face has area 1 -/
  face_area_eq_1 : a * b = 1

/-- The minimum surface area of a sphere containing an inscribed solid -/
noncomputable def min_sphere_surface_area (solid : InscribedSolid) : ℝ :=
  18 * Real.pi

/-- Theorem: The minimum surface area of a sphere containing an inscribed solid is 18π -/
theorem min_sphere_surface_area_is_18pi (solid : InscribedSolid) :
  min_sphere_surface_area solid = 18 * Real.pi := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sphere_surface_area_is_18pi_l536_53611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l536_53676

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (x - 1) / Real.sqrt (2 - x)

-- Define the domain A of f
def A : Set ℝ := {x | 1 < x ∧ x < 2}

-- Define the inequality
def inequality (a x : ℝ) : Prop := x^2 - (2*a + 3)*x + a^2 + 3*a ≤ 0

-- Define the solution set B of the inequality
def B (a : ℝ) : Set ℝ := {x | inequality a x}

-- Theorem statement
theorem range_of_a (a : ℝ) : A ∩ B a = A → a ∈ Set.Icc (-1 : ℝ) 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l536_53676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_product_l536_53680

theorem sin_cos_product (α : ℝ) : 
  Real.sin α - Real.cos α = Real.sqrt 2 / 2 → Real.sin α * Real.cos α = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_product_l536_53680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l536_53621

/-- The area of a triangle with vertices at (3, 2), (3, -4), and (9, -1) is 18 square units. -/
theorem triangle_area : ∃ (area : ℝ), area = 18 := by
  -- Define the vertices of the triangle
  let v1 : ℝ × ℝ := (3, 2)
  let v2 : ℝ × ℝ := (3, -4)
  let v3 : ℝ × ℝ := (9, -1)

  -- Calculate the area of the triangle
  let area := (1/2) * |v1.1 - v2.1| * |v3.1 - v1.1|

  -- Prove that the area is equal to 18
  use area
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l536_53621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_students_l536_53695

theorem number_of_students : ℕ :=
  let average_age : ℝ := 15
  let group1_count : ℕ := 8
  let group1_avg : ℝ := 14
  let group2_count : ℕ := 6
  let group2_avg : ℝ := 16
  let last_student_age : ℕ := 17
  15

#check number_of_students

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_students_l536_53695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_hyperbola_triangle_area_l536_53601

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an equilateral hyperbola x^2 - y^2 = 1 -/
def EquilateralHyperbola := {p : Point | p.x^2 - p.y^2 = 1}

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Two lines are perpendicular if their dot product is zero -/
def perpendicular (p1 p2 p3 : Point) : Prop :=
  (p2.x - p1.x) * (p3.x - p1.x) + (p2.y - p1.y) * (p3.y - p1.y) = 0

/-- The area of a triangle given three points -/
noncomputable def triangleArea (p1 p2 p3 : Point) : ℝ :=
  (1/2) * abs ((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y))

/-- The foci of the equilateral hyperbola -/
noncomputable def F1 : Point := ⟨Real.sqrt 2, 0⟩
noncomputable def F2 : Point := ⟨-Real.sqrt 2, 0⟩

theorem equilateral_hyperbola_triangle_area 
  (P : Point) 
  (h1 : P ∈ EquilateralHyperbola) 
  (h2 : perpendicular P F1 F2) : 
  triangleArea P F1 F2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_hyperbola_triangle_area_l536_53601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_junior_score_l536_53685

theorem junior_score (total_students : ℕ) (junior_percent senior_percent : ℚ) 
  (class_average senior_average : ℚ) : 
  junior_percent = 1/5 →
  senior_percent = 4/5 →
  junior_percent + senior_percent = 1 →
  class_average = 85 →
  senior_average = 82 →
  let junior_count := (junior_percent * total_students).floor
  let senior_count := (senior_percent * total_students).floor
  let total_score := class_average * total_students
  let senior_total_score := senior_average * senior_count
  let junior_total_score := total_score - senior_total_score
  junior_total_score / junior_count = 97 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_junior_score_l536_53685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l536_53619

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.exp (abs x) + x^2

-- State the theorem
theorem range_of_a (a : ℝ) : f (3 * a - 2) > f (a - 1) ↔ a < 1/2 ∨ a > 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l536_53619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_f_minimum_f_not_monotonic_f_symmetric_l536_53652

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) + Real.sin x + Real.cos x

-- Statement 1: The period of f is 2π
theorem f_period : ∀ x, f (x + 2 * Real.pi) = f x := by sorry

-- Statement 2: The minimum value of f is -5/4
theorem f_minimum : ∃ x, f x = -5/4 ∧ ∀ y, f y ≥ -5/4 := by sorry

-- Statement 3: f is not monotonic
theorem f_not_monotonic : ¬(∀ x y, x < y → f x < f y) ∧ ¬(∀ x y, x < y → f x > f y) := by sorry

-- Statement 4: f is symmetric about x = π/4
theorem f_symmetric : ∀ x, f (Real.pi/2 - x) = f x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_f_minimum_f_not_monotonic_f_symmetric_l536_53652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_perpendicular_l536_53637

noncomputable section

-- Define the curve
def curve (x : ℝ) : ℝ := x^2 + x - 2

-- Define the tangent line l₁
def l₁ (x : ℝ) : ℝ := 3*x - 3

-- Define the point of tangency for l₁
def tangent_point : ℝ × ℝ := (1, 0)

-- Define the slope of l₂
def slope_l₂ : ℝ := -1/3

-- Define the y-intercept of l₂
def y_intercept_l₂ : ℝ := -2/3

-- Statement to prove
theorem tangent_line_perpendicular :
  ∃ (b : ℝ), 
    (curve b = b^2 + b - 2) ∧ 
    (slope_l₂ = -(1 / (2*tangent_point.1 + 1))) ∧
    (y_intercept_l₂ = curve b - slope_l₂ * b) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_perpendicular_l536_53637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_function_l536_53622

theorem min_value_of_function (x : ℝ) : 
  (2 : ℝ)^x + (2 : ℝ)^(2-x) ≥ 4 ∧ ∃ x₀ : ℝ, (2 : ℝ)^x₀ + (2 : ℝ)^(2-x₀) = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_function_l536_53622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_divisors_450_prime_factors_l536_53655

theorem sum_of_divisors_450_prime_factors :
  let sum_of_divisors := (Nat.divisors 450).sum id
  (Nat.factors sum_of_divisors).toFinset.card = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_divisors_450_prime_factors_l536_53655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_symmetric_and_self_inverse_l536_53673

noncomputable def f (x : ℝ) : ℝ := (x - 3) / (x - 2)

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f (x + a)

theorem g_symmetric_and_self_inverse (a : ℝ) :
  (∀ x, g a x = g a (g a x)) ∧ 
  (∀ x, g a x = x + 1 ↔ g a (x + 1) = x) ↔ 
  a = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_symmetric_and_self_inverse_l536_53673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radical_axis_of_circles_l536_53658

/-- The equation of the first circle -/
def circle1 (x y : ℝ) : ℝ := x^2 + y^2 - 4*x + 6*y

/-- The equation of the second circle -/
def circle2 (x y : ℝ) : ℝ := x^2 + y^2 - 6*x

/-- The equation of the radical axis -/
def radical_axis (x y : ℝ) : Prop := 3*x - y - 9 = 0

/-- Theorem stating that the given equation is the radical axis of the two circles -/
theorem radical_axis_of_circles :
  ∀ x y : ℝ, radical_axis x y ↔ (∃ k : ℝ, k ≠ 0 ∧ circle1 x y = k * circle2 x y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_radical_axis_of_circles_l536_53658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_school_referendum_l536_53603

theorem school_referendum (total_students : ℕ) (voted_A : ℕ) (voted_B : ℕ) (voted_against_both : ℕ) 
  (voted_both : ℕ) :
  total_students = 250 →
  voted_A = 175 →
  voted_B = 145 →
  voted_against_both = 35 →
  total_students - voted_against_both = voted_A + voted_B - voted_both →
  voted_both = 105 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_school_referendum_l536_53603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pi_bounds_l536_53617

noncomputable def π : ℝ := Real.pi

theorem pi_bounds : 3.14 < π ∧ π < 3.142 ∧ 9.86 < π^2 ∧ π^2 < 9.87 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pi_bounds_l536_53617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_y_value_l536_53656

/-- Given two vectors a and b in ℝ³, if they are parallel and have the specified components, then the y-component of b is 7.5. -/
theorem parallel_vectors_y_value (a b : ℝ × ℝ × ℝ) :
  a = (2, 4, 5) →
  (b.1 = 3 ∧ b.2.1 = -6) →
  (∃ (k : ℝ), b = (k * a.1, k * a.2.1, k * a.2.2)) →
  b.2.2 = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_y_value_l536_53656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_one_sufficient_not_necessary_l536_53699

/-- The length of the chord formed by the intersection of a line and a circle -/
noncomputable def chordLength (k : ℝ) : ℝ :=
  2 * Real.sqrt (1 - (1 / Real.sqrt (1 + k^2))^2)

/-- Theorem stating that k = 1 is a sufficient but not necessary condition for chord length √2 -/
theorem k_one_sufficient_not_necessary :
  (∃ k : ℝ, k ≠ 1 ∧ chordLength k = Real.sqrt 2) ∧
  (chordLength 1 = Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_one_sufficient_not_necessary_l536_53699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_27_l536_53616

/-- The angle formed by the hands of a clock at a given time -/
noncomputable def clock_angle (hours : ℕ) (minutes : ℕ) : ℝ :=
  let minute_angle : ℝ := minutes * 6
  let hour_angle : ℝ := (hours % 12) * 30 + minutes * 0.5
  let angle_diff := abs (minute_angle - hour_angle)
  min angle_diff (360 - angle_diff)

/-- The acute angle formed by the hands of a clock at 3:27 is 58.5 degrees -/
theorem clock_angle_at_3_27 : clock_angle 3 27 = 58.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_27_l536_53616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_B_most_circular_l536_53698

/-- Represents an ellipse in standard form (x^2/a^2 + y^2/b^2 = 1) --/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- Calculates the eccentricity of an ellipse --/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (min e.a e.b / max e.a e.b) ^ 2)

/-- The four ellipses given in the problem --/
noncomputable def ellipse_A : Ellipse where
  a := 2
  b := 6
  h_pos_a := by norm_num
  h_pos_b := by norm_num

noncomputable def ellipse_B : Ellipse where
  a := 4
  b := 2 * Real.sqrt 3
  h_pos_a := by norm_num
  h_pos_b := by { apply mul_pos; norm_num; exact Real.sqrt_pos.mpr (by norm_num) }

noncomputable def ellipse_C : Ellipse where
  a := 6
  b := 2
  h_pos_a := by norm_num
  h_pos_b := by norm_num

noncomputable def ellipse_D : Ellipse where
  a := Real.sqrt 6
  b := Real.sqrt 10
  h_pos_a := Real.sqrt_pos.mpr (by norm_num)
  h_pos_b := Real.sqrt_pos.mpr (by norm_num)

/-- Theorem stating that ellipse B is the most circular --/
theorem ellipse_B_most_circular :
  eccentricity ellipse_B < eccentricity ellipse_A ∧
  eccentricity ellipse_B < eccentricity ellipse_C ∧
  eccentricity ellipse_B < eccentricity ellipse_D :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_B_most_circular_l536_53698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_service_time_properties_l536_53624

noncomputable section

variable (v : ℝ) (hv : v > 0)

/-- The distribution function for the service time -/
def F (t : ℝ) : ℝ :=
  if t ≤ 0 then 0 else 1 - Real.exp (-v * t)

/-- The random variable representing service time -/
def T : Type := ℝ

/-- The expected value of T -/
noncomputable def expected_value : ℝ := 1 / v

/-- The variance of T -/
noncomputable def variance : ℝ := 1 / (v^2)

/-- The probability density function (PDF) -/
noncomputable def pdf (t : ℝ) : ℝ :=
  if t ≤ 0 then 0 else v * Real.exp (-v * t)

theorem service_time_properties :
  (∫ (t : ℝ), t * pdf v t = expected_value v) ∧
  (∫ (t : ℝ), (t - expected_value v)^2 * pdf v t = variance v) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_service_time_properties_l536_53624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_2x_plus_sqrt_1_minus_x_squared_l536_53684

open Real MeasureTheory Interval

theorem integral_2x_plus_sqrt_1_minus_x_squared :
  ∫ x in (Set.Icc 0 1), (2 * x + Real.sqrt (1 - x^2)) = 1 + π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_2x_plus_sqrt_1_minus_x_squared_l536_53684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sheep_transaction_gain_l536_53627

theorem sheep_transaction_gain (c : ℝ) (h : c > 0) :
  let total_sheep : ℕ := 800
  let sold_sheep : ℕ := 750
  let remaining_sheep : ℕ := total_sheep - sold_sheep
  let cost_per_sheep : ℝ := c
  let total_cost : ℝ := c * total_sheep
  let revenue_750 : ℝ := total_cost
  let price_per_sheep_750 : ℝ := revenue_750 / sold_sheep
  let price_per_sheep_50 : ℝ := price_per_sheep_750 * 1.1
  let revenue_50 : ℝ := price_per_sheep_50 * remaining_sheep
  let total_revenue : ℝ := revenue_750 + revenue_50
  let profit : ℝ := total_revenue - total_cost
  let percent_gain : ℝ := (profit / total_cost) * 100
  percent_gain = 14 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sheep_transaction_gain_l536_53627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_length_first_group_l536_53663

/-- A sequence of 19 ones and 49 zeros arranged in random order -/
def RandomSequence := List (Fin 2)

/-- A group is a maximal subsequence of identical symbols -/
def GroupOf (s : RandomSequence) := List (Fin 2)

/-- The first group of a sequence -/
def FirstGroup (s : RandomSequence) : GroupOf s :=
  sorry

/-- The length of a group -/
def GroupLength (s : RandomSequence) (g : GroupOf s) : ℕ :=
  sorry

/-- The expected length of the first group -/
noncomputable def ExpectedLengthFirstGroup : ℝ :=
  sorry

/-- Theorem stating that the expected length of the first group is 2.83 -/
theorem expected_length_first_group :
  ExpectedLengthFirstGroup = 2.83 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_length_first_group_l536_53663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_propositions_false_l536_53690

-- Define the propositions
def proposition1 (A B : ℝ) : Prop := ∀ (C : ℝ), Real.sin A > Real.sin B → A > B

def proposition2 (f : ℝ → ℝ) : Prop := 
  (f 1 * f 2 < 0) ↔ ∃ x, x > 1 ∧ x < 2 ∧ f x = 0

def proposition3 (a : ℕ → ℝ) : Prop := 
  a 1 = 1 ∧ a 5 = 16 → a 3 = 4 ∨ a 3 = -4

noncomputable def proposition4 (f g : ℝ → ℝ) : Prop := 
  (∀ x, f x = Real.sin (2 - 2*x)) → 
  (∀ x, g x = f (x - 2)) → 
  (∀ x, g x = Real.sin (4 - 2*x))

-- Theorem stating all propositions are false
theorem all_propositions_false : 
  (¬ ∀ A B : ℝ, proposition1 A B) ∧ 
  (¬ ∀ f : ℝ → ℝ, proposition2 f) ∧ 
  (¬ ∀ a : ℕ → ℝ, proposition3 a) ∧ 
  (¬ ∀ f g : ℝ → ℝ, proposition4 f g) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_propositions_false_l536_53690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l536_53667

def p (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 - 4*x + a > 0

def q (a : ℝ) : Prop := ∀ x : ℝ, x < -1 → 2*x^2 + x > 2 + a*x

theorem range_of_a (a : ℝ) :
  (p a ∨ q a) ∧ ¬(p a ∧ q a) → 1 ≤ a ∧ a ≤ 2 :=
by
  sorry

#check range_of_a

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l536_53667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_circle_centers_l536_53683

/-- The distance between the centers of two circles with polar equations ρ = 2cosθ and ρ = sinθ -/
theorem distance_between_circle_centers : ∃ (d : ℝ), d = Real.sqrt (5/4) ∧
  d = Real.sqrt ((1 - 0)^2 + (0 - 1/2)^2) :=
by
  -- Define d
  let d := Real.sqrt (5/4)
  
  -- Prove existence
  use d
  
  constructor
  · -- First part: d = Real.sqrt (5/4)
    rfl
  
  · -- Second part: d = Real.sqrt ((1 - 0)^2 + (0 - 1/2)^2)
    calc
      d = Real.sqrt (5/4) := rfl
      _ = Real.sqrt ((1 - 0)^2 + (0 - 1/2)^2) := by
        congr
        ring
  
  -- The proof is complete

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_circle_centers_l536_53683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_painting_problem_l536_53674

theorem rectangle_painting_problem :
  let count := Finset.filter (fun p : ℕ × ℕ => 
    let a := p.1
    let b := p.2
    b > a ∧ 
    (a - 4) * (b - 4) = 2 * (a * b) / 3) (Finset.range 100 ×ˢ Finset.range 100)
  Finset.card count = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_painting_problem_l536_53674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_woman_work_days_l536_53600

/-- The number of days required for one woman to complete a work, given that:
  * 10 men and 15 women together can complete the work in 8 days
  * One man alone can complete the work in 100 days -/
noncomputable def days_for_one_woman (total_work : ℝ) : ℝ :=
  let men_rate := total_work / 100
  let combined_rate := total_work / 8
  let women_rate := (combined_rate - 10 * men_rate) / 15
  total_work / women_rate

theorem one_woman_work_days (total_work : ℝ) (total_work_pos : total_work > 0) :
  days_for_one_woman total_work = 600 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_woman_work_days_l536_53600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trihedral_angle_triangle_l536_53643

theorem trihedral_angle_triangle (α β γ : ℝ) 
  (h : α > 0 ∧ β > 0 ∧ γ > 0) : 
  ∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a = Real.sin (α / 2) ∧
    b = Real.sin (β / 2) ∧
    c = Real.sin (γ / 2) ∧
    a + b > c ∧ b + c > a ∧ c + a > b :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trihedral_angle_triangle_l536_53643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l536_53640

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.exp (x * Real.log 2) - 4
  else -(Real.exp (-x * Real.log 2) - 4)

-- State the theorem
theorem solution_set_of_inequality (f : ℝ → ℝ) :
  (∀ x, f (-x) = -f x) →  -- f is odd
  (∀ x > 0, f x = Real.exp (x * Real.log 2) - 4) →  -- definition for x > 0
  {x : ℝ | x * f (x + 1) < 0} = Set.union (Set.Ioo 0 1) (Set.Ioo (-3) (-1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l536_53640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_packing_height_difference_l536_53646

/-- The difference in heights between two packing methods of cylindrical pipes -/
theorem pipe_packing_height_difference :
  let pipe_diameter : ℝ := 8
  let crate_a_pipes : ℕ := 150
  let crate_b_pipes : ℕ := 180
  let crate_a_height : ℝ := 15 * pipe_diameter
  let crate_b_height : ℝ := 4 + 18 * (pipe_diameter * Real.sqrt 3 / 2)
  abs (crate_a_height - crate_b_height) = 116 - 72 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_packing_height_difference_l536_53646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l536_53687

noncomputable def f (x : ℝ) : ℝ := Real.cos x * Real.cos (x - Real.pi/3) - 1/4

theorem triangle_side_length 
  (A B C : ℝ) 
  (a b c : ℝ) 
  (h1 : f A = 1/2) 
  (h2 : c = 2) 
  (h3 : 2 * b * Real.cos A = 3/2) : 
  a = Real.sqrt 7 / 2 := 
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l536_53687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sum_less_than_11_l536_53679

def ball_numbers : Finset ℕ := {1, 2, 3, 4, 5, 6}

def total_outcomes : ℕ := ball_numbers.card * ball_numbers.card

def favorable_outcomes : Finset (ℕ × ℕ) :=
  (ball_numbers.product ball_numbers).filter (λ p => p.1 + p.2 < 11)

theorem probability_sum_less_than_11 :
  (favorable_outcomes.card : ℚ) / total_outcomes = 7 / 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sum_less_than_11_l536_53679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_of_perpendicular_asymptotes_l536_53649

/-- A hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  ha : a > 0
  hb : b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola a b) : ℝ :=
  Real.sqrt (1 + b^2 / a^2)

/-- Condition for perpendicular asymptotes -/
def has_perpendicular_asymptotes (h : Hyperbola a b) : Prop :=
  a = b

/-- Theorem: If a hyperbola has perpendicular asymptotes, its eccentricity is √2 -/
theorem eccentricity_of_perpendicular_asymptotes {a b : ℝ} (h : Hyperbola a b) :
  has_perpendicular_asymptotes h → eccentricity h = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_of_perpendicular_asymptotes_l536_53649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_is_480_l536_53620

-- Define the pyramid QEFGH
structure Pyramid :=
  (E F G H Q : ℝ × ℝ × ℝ)

-- Define the conditions
def pyramid_conditions (p : Pyramid) : Prop :=
  let (Ex, Ey, Ez) := p.E
  let (Fx, Fy, Fz) := p.F
  let (Gx, Gy, Gz) := p.G
  let (Hx, Hy, Hz) := p.H
  let (Qx, Qy, Qz) := p.Q
  -- EF = 10
  ((Ex - Fx)^2 + (Ey - Fy)^2 + (Ez - Fz)^2 = 100) ∧
  -- FG = 6
  ((Fx - Gx)^2 + (Fy - Gy)^2 + (Fz - Gz)^2 = 36) ∧
  -- QE ⟂ EH
  ((Qx - Ex) * (Hx - Ex) + (Qy - Ey) * (Hy - Ey) + (Qz - Ez) * (Hz - Ez) = 0) ∧
  -- QE ⟂ EF
  ((Qx - Ex) * (Fx - Ex) + (Qy - Ey) * (Fy - Ey) + (Qz - Ez) * (Fz - Ez) = 0) ∧
  -- QF = 26
  ((Qx - Fx)^2 + (Qy - Fy)^2 + (Qz - Fz)^2 = 676)

-- Define the volume function
noncomputable def pyramid_volume (p : Pyramid) : ℝ :=
  let (Ex, Ey, Ez) := p.E
  let (Fx, Fy, Fz) := p.F
  let (Gx, Gy, Gz) := p.G
  let (Hx, Hy, Hz) := p.H
  let (Qx, Qy, Qz) := p.Q
  let base_area := Real.sqrt (((Ex - Fx)^2 + (Ey - Fy)^2 + (Ez - Fz)^2) * ((Fx - Gx)^2 + (Fy - Gy)^2 + (Fz - Gz)^2))
  let height := Real.sqrt ((Qx - Ex)^2 + (Qy - Ey)^2 + (Qz - Ez)^2)
  (1/3) * base_area * height

-- Theorem statement
theorem pyramid_volume_is_480 (p : Pyramid) :
  pyramid_conditions p → pyramid_volume p = 480 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_is_480_l536_53620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_intersection_length_l536_53650

/-- Parabola E: x^2 = 4y -/
def parabola_E (x y : ℝ) : Prop := x^2 = 4*y

/-- Circle C: x^2 + y^2 + 8x + ay - 5 = 0 -/
def circle_C (a : ℝ) (x y : ℝ) : Prop := x^2 + y^2 + 8*x + a*y - 5 = 0

/-- Focus of parabola E -/
def focus_E : ℝ × ℝ := (0, 1)

/-- Directrix of parabola E -/
def directrix_E (y : ℝ) : Prop := y = -1

/-- Circle C passes through the focus of parabola E -/
def C_passes_through_focus (a : ℝ) : Prop :=
  circle_C a (focus_E.1) (focus_E.2)

/-- Length of the chord formed by the intersection of directrix_E and circle_C -/
noncomputable def chord_length (a : ℝ) : ℝ := 4 * Real.sqrt 6

theorem chord_intersection_length (a : ℝ) (h : C_passes_through_focus a) :
  chord_length a = 4 * Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_intersection_length_l536_53650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wealthiest_customers_average_income_l536_53693

/-- Given a group of 50 customers where the average income is $45,000 and a subgroup of 40 customers
    with an average income of $42,500, this theorem proves that the average income of the remaining
    10 customers is $55,000. -/
theorem wealthiest_customers_average_income
  (total_customers : ℕ)
  (wealthy_customers : ℕ)
  (other_customers : ℕ)
  (total_average_income : ℚ)
  (other_average_income : ℚ)
  (h1 : total_customers = 50)
  (h2 : wealthy_customers = 10)
  (h3 : other_customers = 40)
  (h4 : total_customers = wealthy_customers + other_customers)
  (h5 : total_average_income = 45000)
  (h6 : other_average_income = 42500) :
  (total_customers * total_average_income - other_customers * other_average_income) / wealthy_customers = 55000 := by
  sorry

-- Remove the #eval line as it's not necessary for building

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wealthiest_customers_average_income_l536_53693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_EFG_is_60_degrees_l536_53654

-- Define the triangle EFG
structure Triangle (E F G : ℝ × ℝ) : Prop where
  right_angle_at_E : (F.1 - E.1) * (G.1 - E.1) + (F.2 - E.2) * (G.2 - E.2) = 0

-- Define point H on FG
noncomputable def H (F G : ℝ × ℝ) : ℝ × ℝ := ((F.1 + G.1) / 2, (F.2 + G.2) / 2)

-- Define the conditions
def Conditions (E F G : ℝ × ℝ) : Prop :=
  ∃ (_t : Triangle E F G),
    let H := H F G
    (F.1 - H.1)^2 + (F.2 - H.2)^2 = (G.1 - H.1)^2 + (G.2 - H.2)^2 ∧
    (F.1 - E.1)^2 + (F.2 - E.2)^2 = 4 * ((H.1 - E.1)^2 + (H.2 - E.2)^2)

-- Define the angle EFG
noncomputable def AngleEFG (E F G : ℝ × ℝ) : ℝ :=
  Real.arccos ((F.1 - E.1) * (G.1 - F.1) + (F.2 - E.2) * (G.2 - F.2)) /
    (Real.sqrt ((F.1 - E.1)^2 + (F.2 - E.2)^2) * Real.sqrt ((G.1 - F.1)^2 + (G.2 - F.2)^2))

-- State the theorem
theorem angle_EFG_is_60_degrees (E F G : ℝ × ℝ) :
  Conditions E F G → AngleEFG E F G = π / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_EFG_is_60_degrees_l536_53654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coprime_sequence_iff_coprime_diff_parity_l536_53692

/-- Given positive integers a and b, and a sequence x_n defined by x_n = a^n + b^n,
    prove the equivalence of the existence of an infinite coprime subsequence and
    the coprimality and parity difference of a and b. -/
theorem coprime_sequence_iff_coprime_diff_parity (a b : ℕ+) :
  (∃ (m : ℕ → ℕ), Monotone m ∧ StrictMono m ∧ 
    ∀ i j, i ≠ j → Nat.Coprime ((a^(m i) + b^(m i)) : ℕ) ((a^(m j) + b^(m j)) : ℕ)) ↔ 
  (Nat.Coprime a.val b.val ∧ a.val % 2 ≠ b.val % 2) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coprime_sequence_iff_coprime_diff_parity_l536_53692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_F_in_triangle_l536_53608

-- Define a triangle DEF
structure Triangle where
  D : ℝ
  E : ℝ
  F : ℝ

-- Define the theorem
theorem cos_F_in_triangle (t : Triangle) 
  (h1 : Real.sin t.D = 4/5)
  (h2 : Real.cos t.E = 12/13) :
  Real.cos t.F = -16/65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_F_in_triangle_l536_53608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_deriv_deriv_l536_53636

-- Define the derivative operation
def my_deriv (x : ℝ) : ℝ := 3 * x - 3

-- State the theorem
theorem fourth_deriv_deriv : my_deriv (my_deriv 4) = 24 := by
  -- Evaluate my_deriv 4
  have h1 : my_deriv 4 = 9 := by
    rw [my_deriv]
    norm_num
  
  -- Evaluate my_deriv 9
  have h2 : my_deriv 9 = 24 := by
    rw [my_deriv]
    norm_num
  
  -- Combine the steps
  rw [h1, h2]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_deriv_deriv_l536_53636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_properties_l536_53629

/-- Polar coordinate -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- Convert polar to Cartesian coordinates -/
noncomputable def polarToCartesian (p : PolarPoint) : ℝ × ℝ :=
  (p.r * Real.cos p.θ, p.r * Real.sin p.θ)

/-- Line passing through a point with given slope angle -/
structure Line where
  point : ℝ × ℝ
  slopeAngle : ℝ

/-- Circle with center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Main theorem -/
theorem intersection_properties
  (P : PolarPoint)
  (M : PolarPoint)
  (l : Line)
  (c : Circle)
  (h1 : P = ⟨1, 3*Real.pi/2⟩)
  (h2 : M = ⟨2, Real.pi/6⟩)
  (h3 : l.point = polarToCartesian P)
  (h4 : l.slopeAngle = Real.pi/3)
  (h5 : c.center = polarToCartesian M)
  (h6 : c.radius = 2)
  (A B : ℝ × ℝ)
  (h7 : A ∈ Set.range (fun t => (l.point.1 + t * Real.cos l.slopeAngle,
                                 l.point.2 + t * Real.sin l.slopeAngle)))
  (h8 : B ∈ Set.range (fun t => (l.point.1 + t * Real.cos l.slopeAngle,
                                 l.point.2 + t * Real.sin l.slopeAngle)))
  (h9 : (A.1 - c.center.1)^2 + (A.2 - c.center.2)^2 = c.radius^2)
  (h10 : (B.1 - c.center.1)^2 + (B.2 - c.center.2)^2 = c.radius^2)
  : (A.1 - B.1)^2 + (A.2 - B.2)^2 = 15 ∧
    ((A.1 - l.point.1)^2 + (A.2 - l.point.2)^2) *
    ((B.1 - l.point.1)^2 + (B.2 - l.point.2)^2) = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_properties_l536_53629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l536_53632

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, Real.sqrt 2 ≤ x ∧ x ≤ 4 → (5/2) * x^2 ≥ m * (x - 1)) ↔ 
  m ∈ Set.Iic 10 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l536_53632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_angle_is_60_degrees_l536_53666

-- Define the line l: √3x - y + 6 = 0
noncomputable def line_l (x y : ℝ) : Prop := Real.sqrt 3 * x - y + 6 = 0

-- Define the angle θ
noncomputable def angle_θ : ℝ := 60 * Real.pi / 180

-- Theorem statement
theorem line_angle_is_60_degrees :
  ∀ x y : ℝ, line_l x y → 
  Real.arctan (Real.sqrt 3) = angle_θ := by
  sorry

#check line_angle_is_60_degrees

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_angle_is_60_degrees_l536_53666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_hexagonal_lattice_triangles_l536_53675

/-- Represents a point in the extended hexagonal lattice -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- Represents the extended hexagonal lattice -/
structure ExtendedHexagonalLattice where
  innerVertices : List LatticePoint
  outerVertices : List LatticePoint
  center : LatticePoint

/-- Represents an equilateral triangle in the lattice -/
structure EquilateralTriangle where
  a : LatticePoint
  b : LatticePoint
  c : LatticePoint

/-- Distance function between two points in the lattice -/
def distance (p1 p2 : LatticePoint) : ℕ :=
  sorry

/-- Predicate to check if three points form an equilateral triangle -/
def isEquilateralTriangle (t : EquilateralTriangle) : Prop :=
  distance t.a t.b = distance t.b t.c ∧ distance t.b t.c = distance t.c t.a

/-- Function to count equilateral triangles in the lattice -/
def countEquilateralTriangles (l : ExtendedHexagonalLattice) : ℕ :=
  sorry

/-- Theorem stating that there are exactly 20 equilateral triangles in the extended hexagonal lattice -/
theorem extended_hexagonal_lattice_triangles
  (l : ExtendedHexagonalLattice)
  (h1 : l.innerVertices.length = 6)
  (h2 : l.outerVertices.length = 6)
  (h3 : ∀ (p1 p2 : LatticePoint), p1 ∈ l.innerVertices → p2 ∈ l.innerVertices → p1 ≠ p2 → distance p1 p2 = 1)
  (h4 : ∀ (p1 p2 : LatticePoint), p1 ∈ l.outerVertices → p2 ∈ l.outerVertices → p1 ≠ p2 → distance p1 p2 = 2)
  (h5 : ∀ (p : LatticePoint), p ∈ l.innerVertices → distance p l.center = 1)
  (h6 : ∀ (p : LatticePoint), p ∈ l.outerVertices → ∃ (q : LatticePoint), q ∈ l.innerVertices ∧ distance p q = 1) :
  countEquilateralTriangles l = 20 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_hexagonal_lattice_triangles_l536_53675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_theorem_smallest_k_minimal_l536_53662

/-- Represents a group of people and their relationships -/
structure PeopleGroup where
  people : Finset Nat
  knows : Nat → Nat → Bool

/-- The smallest k satisfying the condition for given m and n -/
def smallest_k (m n : Nat) : Nat :=
  m + n + max m n - 1

/-- Checks if there are m pairs of people who know each other -/
def has_m_knowing_pairs (g : PeopleGroup) (m : Nat) : Prop :=
  ∃ pairs : Finset (Nat × Nat),
    pairs.card = m ∧
    (∀ p, p ∈ pairs → g.knows p.1 p.2) ∧
    (∀ p q, p ∈ pairs → q ∈ pairs → p ≠ q → p.1 ≠ q.1 ∧ p.1 ≠ q.2 ∧ p.2 ≠ q.1 ∧ p.2 ≠ q.2)

/-- Checks if there are n pairs of people who do not know each other -/
def has_n_not_knowing_pairs (g : PeopleGroup) (n : Nat) : Prop :=
  ∃ pairs : Finset (Nat × Nat),
    pairs.card = n ∧
    (∀ p, p ∈ pairs → ¬g.knows p.1 p.2) ∧
    (∀ p q, p ∈ pairs → q ∈ pairs → p ≠ q → p.1 ≠ q.1 ∧ p.1 ≠ q.2 ∧ p.2 ≠ q.1 ∧ p.2 ≠ q.2)

/-- The main theorem to be proved -/
theorem smallest_k_theorem (m n : Nat) :
  ∀ k : Nat, k ≥ smallest_k m n →
    ∀ g : PeopleGroup, g.people.card = k →
      has_m_knowing_pairs g m ∨ has_n_not_knowing_pairs g n := by
  sorry

/-- The minimality of smallest_k -/
theorem smallest_k_minimal (m n : Nat) :
  ∀ k : Nat, k < smallest_k m n →
    ∃ g : PeopleGroup, g.people.card = k ∧
      ¬(has_m_knowing_pairs g m ∨ has_n_not_knowing_pairs g n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_theorem_smallest_k_minimal_l536_53662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_is_72_l536_53653

/-- A rhombus with the given properties -/
structure Rhombus where
  side_length : ℝ
  diagonal_diff : ℝ
  side_length_eq : side_length = Real.sqrt 113
  diagonal_diff_eq : diagonal_diff = 10

/-- The area of a rhombus given its diagonals -/
noncomputable def rhombus_area (d1 d2 : ℝ) : ℝ := (d1 * d2) / 2

/-- Theorem stating the area of the specific rhombus -/
theorem rhombus_area_is_72 (r : Rhombus) : ∃ d1 d2 : ℝ, 
  d1 > 0 ∧ d2 > 0 ∧ 
  |d1 - d2| = r.diagonal_diff ∧
  d1 * d1 / 4 + d2 * d2 / 4 = r.side_length * r.side_length ∧
  rhombus_area d1 d2 = 72 := by
  sorry

#check rhombus_area_is_72

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_is_72_l536_53653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_simplification_l536_53672

theorem complex_fraction_simplification :
  (((1 : ℂ) + 2*Complex.I)^2 + 3*((1 : ℂ) - Complex.I)) / ((2 : ℂ) + Complex.I) = (1 : ℂ)/5 + (2 : ℂ)/5*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_simplification_l536_53672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_l536_53659

/-- The polynomial x^5 - 5x^4 + 8x^3 - 10x^2 + 4x - 3 -/
def f (x : ℝ) : ℝ := x^5 - 5*x^4 + 8*x^3 - 10*x^2 + 4*x - 3

/-- The divisor x^2 - 3x + k -/
def g (x k : ℝ) : ℝ := x^2 - 3*x + k

/-- The remainder 2x + a -/
def r (x a : ℝ) : ℝ := 2*x + a

/-- The quotient of f divided by g -/
def q (x k : ℝ) : ℝ := x^3 + (2 + k)*x^2 + (k^2 - 3*k + 5)*x + (k^3 - 3*k^2 + 5*k - 4)

theorem polynomial_division : 
  ∀ x k a : ℝ, (∃ c : ℝ, f x = g x k * q x k + r x a) ↔ (k = 4 ∧ a = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_l536_53659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_expression_l536_53697

theorem absolute_value_expression (x : ℝ) : x = 10 → 30 - |-x + 6| = 26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_expression_l536_53697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_winner_l536_53630

/-- Represents a race between two contestants A and B -/
structure Race where
  length : ℚ
  speed_ratio : ℚ
  head_start : ℚ

/-- Calculates the distance by which contestant A wins the race -/
def winning_distance (race : Race) : ℚ :=
  race.length - (race.length - race.head_start) / race.speed_ratio

/-- Theorem stating that in a 600 m race where A's speed is 5/4 times B's speed,
    and A has a 100 m head start, A will win by 200 meters -/
theorem race_winner (race : Race)
  (h1 : race.length = 600)
  (h2 : race.speed_ratio = 5/4)
  (h3 : race.head_start = 100) :
  winning_distance race = 200 := by
  sorry

/-- Example calculation -/
def example_race : Race :=
  { length := 600, speed_ratio := 5/4, head_start := 100 }

#eval winning_distance example_race

end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_winner_l536_53630
