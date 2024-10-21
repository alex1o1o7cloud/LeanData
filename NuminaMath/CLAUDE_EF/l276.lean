import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_planes_with_common_point_may_intersect_l276_27627

/-- A plane in 3D space -/
structure Plane3D where
  -- Define a plane using a point and a normal vector
  point : ℝ × ℝ × ℝ
  normal : ℝ × ℝ × ℝ

/-- Check if a point is on a plane -/
def pointOnPlane (plane : Plane3D) (point : ℝ × ℝ × ℝ) : Prop :=
  let (x, y, z) := point
  let (px, py, pz) := plane.point
  let (nx, ny, nz) := plane.normal
  nx * (x - px) + ny * (y - py) + nz * (z - pz) = 0

/-- Intersection of two planes -/
def planesIntersect (p1 p2 : Plane3D) : Prop :=
  ∃ (x : ℝ × ℝ × ℝ), pointOnPlane p1 x ∧ pointOnPlane p2 x

/-- Common point of two planes -/
def commonPoint (p1 p2 : Plane3D) (point : ℝ × ℝ × ℝ) : Prop :=
  pointOnPlane p1 point ∧ pointOnPlane p2 point

/-- Theorem: If two planes have one common point, they may intersect -/
theorem planes_with_common_point_may_intersect (p1 p2 : Plane3D) (point : ℝ × ℝ × ℝ) :
  commonPoint p1 p2 point → planesIntersect p1 p2 :=
by
  intro h
  use point
  exact h


end NUMINAMATH_CALUDE_ERRORFEEDBACK_planes_with_common_point_may_intersect_l276_27627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_Z_is_one_l276_27678

/-- The magnitude of the complex number Z = (1+2i)/(2-i) is 1. -/
theorem magnitude_of_Z_is_one : Complex.abs ((1 + 2*Complex.I) / (2 - Complex.I)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_Z_is_one_l276_27678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_circles_are_intersecting_l276_27617

-- Define the circles
def circle_M (x y : ℝ) : Prop := x^2 + y^2 = 2
def circle_N (x y : ℝ) : Prop := (x-1)^2 + (y-2)^2 = 3

-- Define the centers and radii
def center_M : ℝ × ℝ := (0, 0)
noncomputable def radius_M : ℝ := Real.sqrt 2
def center_N : ℝ × ℝ := (1, 2)
noncomputable def radius_N : ℝ := Real.sqrt 3

-- Define the distance between centers
noncomputable def distance_between_centers : ℝ := Real.sqrt 5

-- Theorem stating that the circles are intersecting
theorem circles_intersect : 
  (radius_N - radius_M < distance_between_centers) ∧
  (distance_between_centers < radius_M + radius_N) := by
  sorry

-- Additional theorem to explicitly state the circles are intersecting
theorem circles_are_intersecting : 
  ∃ (x y : ℝ), circle_M x y ∧ circle_N x y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_circles_are_intersecting_l276_27617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_square_placement_l276_27646

theorem unit_square_placement (main_square_side : ℝ) (placed_squares : ℕ) : 
  main_square_side = 105 →
  placed_squares = 1999 →
  ∃ (x y : ℝ), 0 ≤ x ∧ x ≤ main_square_side - 1 ∧
                0 ≤ y ∧ y ≤ main_square_side - 1 ∧
                ∀ (i : ℕ), i < placed_squares →
                ∃ (xi yi : ℝ), 0 ≤ xi ∧ xi ≤ main_square_side - 1 ∧
                               0 ≤ yi ∧ yi ≤ main_square_side - 1 ∧
                               |x - xi| ≥ 1 ∨ |y - yi| ≥ 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_square_placement_l276_27646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_point_correct_l276_27693

/-- The meeting point of two celestial bodies moving towards each other -/
noncomputable def meeting_point (d v v₁ : ℝ) : ℝ := (v * d) / (v + v₁)

/-- Theorem stating that the meeting point is correct -/
theorem meeting_point_correct (d v v₁ : ℝ) (h₁ : d > 0) (h₂ : v > 0) (h₃ : v₁ > 0) :
  let x := meeting_point d v v₁
  let t := x / v
  x / v = (d - x) / v₁ ∧ x = (v * d) / (v + v₁) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_point_correct_l276_27693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_rental_theorem_l276_27608

/-- Represents the distribution of people in two categories -/
structure GroupDistribution where
  total : ℕ
  category1 : ℕ
  category2 : ℕ
  h_sum : category1 + category2 = total

/-- Represents the cost associated with each category -/
structure CategoryCosts where
  cost1 : ℚ
  cost2 : ℚ

/-- The probability of selecting at least k people from category1 when choosing n people -/
def prob_at_least (g : GroupDistribution) (n k : ℕ) : ℚ := sorry

/-- The expected value of the sum of costs when randomly selecting n people -/
def expected_cost (g : GroupDistribution) (c : CategoryCosts) (n : ℕ) : ℚ := sorry

/-- Main theorem combining both parts of the problem -/
theorem bicycle_rental_theorem (g : GroupDistribution) (c : CategoryCosts) :
  g.total = 7 ∧ g.category1 = 4 ∧ g.category2 = 3 ∧ c.cost1 = 1 ∧ c.cost2 = 12/10 →
  prob_at_least g 3 2 = 22 / 35 ∧ expected_cost g c 3 = 114 / 35 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_rental_theorem_l276_27608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_expression_l276_27636

noncomputable def arithmetic_sequence (n : ℕ) : ℝ := n

noncomputable def sequence_sum (n : ℕ) : ℝ := n * (1 + n) / 2

theorem min_value_of_expression (n : ℕ) :
  (sequence_sum n + 8) / arithmetic_sequence n ≥ 9/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_expression_l276_27636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_approximation_l276_27622

theorem expression_approximation :
  ∃ k : ℤ, abs (((3^1010 + 7^1011)^2 - (3^1010 - 7^1011)^2 : ℝ) / (10^1010 : ℝ) - k) < 1 ∧ k = 59 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_approximation_l276_27622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_of_five_l276_27697

theorem abs_of_five : |5| = (5 : ℤ) := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_of_five_l276_27697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_function_b_range_l276_27652

-- Define the function f(x)
noncomputable def f (b : ℝ) (x : ℝ) : ℝ := -1/2 * x^2 + b * Real.log (x + 2)

-- State the theorem
theorem decreasing_function_b_range :
  ∀ b : ℝ, (∀ x y : ℝ, x > -1 → y > x → f b y < f b x) ↔ b ≤ -1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_function_b_range_l276_27652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_through_line_parallel_to_line_l276_27685

/-- Given two planes and a line, this theorem proves that a specific plane passes through the 
    intersection of the two planes and is parallel to the given line. -/
theorem plane_through_line_parallel_to_line :
  let plane1 : ℝ → ℝ → ℝ → Prop := λ x y z ↦ 3*x + 2*y + 5*z + 6 = 0
  let plane2 : ℝ → ℝ → ℝ → Prop := λ x y z ↦ x + 4*y + 3*z + 4 = 0
  let line : ℝ → ℝ → ℝ → Prop := λ x y z ↦ ∃ t, x = 3*t + 1 ∧ y = 2*t + 5 ∧ z = -3*t - 1
  let result_plane : ℝ → ℝ → ℝ → Prop := λ x y z ↦ 2*x + 3*y + 4*z + 5 = 0
  (∀ x y z, plane1 x y z ∧ plane2 x y z → result_plane x y z) ∧
  (∀ v : ℝ × ℝ × ℝ, (2 * v.1 + 3 * v.2.1 + 4 * v.2.2 = 0) → (3 * v.1 + 2 * v.2.1 - 3 * v.2.2 = 0)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_through_line_parallel_to_line_l276_27685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_statements_l276_27610

-- Define the basic concepts
def Line : Type := ℝ → ℝ → Prop
def Point : Type := ℝ × ℝ
def Angle : Type := ℝ

-- Define the relationships
def parallel (a b : Line) : Prop := sorry
def perpendicular (a b : Line) : Prop := sorry
def intersect (a b : Line) : Prop := sorry
def angle_bisector (a : Angle) : Line := sorry
def adjacent_angles (a b : Angle) : Prop := sorry
def corresponding_angles (a b : Line) (α β : Angle) : Prop := sorry

-- Define the statements
def statement1 (a b : Line) : Prop := 
  parallel a b → ∀ (α β : Angle), corresponding_angles a b α β → α = β

def statement2 (p : Point) (l : Line) : Prop := 
  ∃! (m : Line), perpendicular m l ∧ (let (x, y) := p; m x y)

def statement3 (l1 l2 : Line) : Prop :=
  intersect l1 l2 → 
  ∃ (α β : Angle), adjacent_angles α β ∧ 
  perpendicular (angle_bisector α) (angle_bisector β)

def statement4 (l1 l2 l3 : Line) : Prop :=
  intersect l1 l2 ∧ intersect l2 l3 ∧ intersect l1 l3 → 
  ∃ (p1 p2 p3 : Point), p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3

def statement5 (a b c : Line) : Prop :=
  parallel a b → parallel b c → parallel a c

def statement6 (a b c : Line) : Prop :=
  perpendicular a b → perpendicular b c → perpendicular a c

-- The theorem to prove
theorem correct_statements :
  (∀ l1 l2, statement3 l1 l2) ∧
  (∀ a b c, statement5 a b c) ∧
  (∃ a b, ¬ statement1 a b) ∧
  (∃ p l, ¬ statement2 p l) ∧
  (∃ l1 l2 l3, ¬ statement4 l1 l2 l3) ∧
  (∃ a b c, ¬ statement6 a b c) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_statements_l276_27610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_diagonal_l276_27625

theorem quadrilateral_diagonal (A B C D : ℝ × ℝ) : 
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let CD := Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2)
  let DA := Real.sqrt ((D.1 - A.1)^2 + (D.2 - A.2)^2)
  let BD := Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2)
  AB = 5 ∧ BC = 17 ∧ CD = 5 ∧ DA = 9 ∧ ∃ n : ℤ, BD = n → BD = 13 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_diagonal_l276_27625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_of_negative_five_l276_27661

theorem opposite_of_negative_five : 
  (-(5 : ℤ)).neg = 5 := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_of_negative_five_l276_27661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_append_to_2014_l276_27680

def is_divisible_by_all_less_than_10 (n : ℕ) : Prop :=
  ∀ k : ℕ, k < 10 → k > 0 → n % k = 0

def append_to_2014 (n : ℕ) : ℕ :=
  2014 * 10^(Nat.log 10 n + 1) + n

theorem smallest_append_to_2014 :
  (is_divisible_by_all_less_than_10 (append_to_2014 506)) ∧
  (∀ m : ℕ, m < 506 → ¬(is_divisible_by_all_less_than_10 (append_to_2014 m))) :=
by sorry

#check smallest_append_to_2014

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_append_to_2014_l276_27680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_Q_difference_l276_27604

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

noncomputable def fractional_part (x : ℝ) : ℝ := x - floor x

noncomputable def Q (x : ℝ) : ℤ := floor x + floor ((fractional_part x) * 10000)

theorem Q_difference : Q 2023 - Q 2022 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_Q_difference_l276_27604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_such_function_exists_l276_27645

theorem no_such_function_exists :
  ¬∃ f : ℝ → ℝ, (∀ x y : ℝ, x > 0 → y > 0 → (f x)^2 ≥ f (x + y) * (f x + y)) ∧ (∀ x : ℝ, x > 0 → f x > 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_such_function_exists_l276_27645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_solution_parallel_solution_l276_27631

/-- Two lines in 2D space -/
structure TwoLines where
  a : ℝ
  l₁ : ℝ → ℝ → Prop := λ x y ↦ a * x + 2 * y + 1 = 0
  l₂ : ℝ → ℝ → Prop := λ x y ↦ x - y + a = 0

/-- Perpendicular lines -/
def perpendicular (lines : TwoLines) : Prop :=
  ∃ x y, lines.l₁ x y ∧ lines.l₂ x y ∧ (lines.a / 2) * 1 = -1

/-- Parallel lines -/
def parallel (lines : TwoLines) : Prop :=
  lines.a / 2 = -1

/-- Distance between parallel lines -/
noncomputable def distance (lines : TwoLines) : ℝ :=
  (3 * Real.sqrt 2) / 4

theorem perpendicular_solution (lines : TwoLines) (h : perpendicular lines) :
    lines.a = 2 ∧ ∃ x y, x = -5/4 ∧ y = 3/4 ∧ lines.l₁ x y ∧ lines.l₂ x y := by
  sorry

theorem parallel_solution (lines : TwoLines) (h : parallel lines) :
    lines.a = -2 ∧ distance lines = (3 * Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_solution_parallel_solution_l276_27631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_when_a_zero_one_zero_iff_a_positive_l276_27613

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - 1 / x - (a + 1) * Real.log x

-- Theorem for the maximum value when a = 0
theorem max_value_when_a_zero :
  ∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → f 0 y ≤ f 0 x ∧ f 0 x = -1 :=
by sorry

-- Theorem for the range of a when f(x) has exactly one zero
theorem one_zero_iff_a_positive :
  ∀ (a : ℝ), (∃! (x : ℝ), x > 0 ∧ f a x = 0) ↔ a > 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_when_a_zero_one_zero_iff_a_positive_l276_27613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_number_is_seven_l276_27682

-- Define the function f based on the graph
def f : ℕ → ℕ
| 3 => 7
| 7 => 3
| _ => 0  -- for other inputs, we'll return 0 (this doesn't affect our proof)

-- Define a function that generates the nth number in the sequence
def larry_sequence (n : ℕ) : ℕ :=
  match n with
  | 0 => 3  -- start with 3
  | n + 1 => f (larry_sequence n)

-- Theorem to prove
theorem tenth_number_is_seven : larry_sequence 9 = 7 := by
  sorry

-- Additional lemma to show the alternating pattern
lemma alternating_pattern (n : ℕ) : 
  larry_sequence n = if n % 2 = 0 then 3 else 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_number_is_seven_l276_27682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_condition_l276_27695

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_iff_slopes_eq {a₁ b₁ a₂ b₂ : ℝ} (h₁ : b₁ ≠ 0) (h₂ : b₂ ≠ 0) :
  (∃ (x y c : ℝ), a₁ * x + b₁ * y + c = 0 ∧ a₂ * x + b₂ * y + (c + 1) = 0) ↔ a₁ / b₁ = a₂ / b₂

/-- The statement to be proved -/
theorem parallel_condition (m : ℝ) :
  (∃ (x y c : ℝ), 2 * x - m * y + c = 0 ∧ (m - 1) * x - y + (c + 1) = 0) ↔ m = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_condition_l276_27695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_division_l276_27651

/-- Represents a division of a square grid into smaller squares -/
structure GridDivision (n : ℕ) where
  squares : List (ℕ × ℕ)  -- List of (size, count) pairs
  valid : ∀ p ∈ squares, p.1 ≤ n ∧ p.2 > 0
  covers_grid : (squares.map (λ p => p.1 * p.1 * p.2)).sum = n * n
  not_all_same_size : squares.length > 1
  equal_count : ∀ i j, i < squares.length → j < squares.length → (squares.get! i).2 = (squares.get! j).2

/-- For any n x n grid, there exists a valid division satisfying the required conditions -/
theorem exists_valid_division (n : ℕ) (h : n > 0) : ∃ (d : GridDivision n), True := by
  sorry

#check exists_valid_division

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_division_l276_27651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_at_two_radians_l276_27630

-- Define the length of the wire
noncomputable def wire_length : ℝ := 40

-- Define the sector
structure Sector where
  radius : ℝ
  arc_length : ℝ
  central_angle : ℝ

-- Define the condition that the wire forms the sector
def forms_sector (s : Sector) : Prop :=
  s.radius * 2 + s.arc_length = wire_length

-- Define the area of the sector
noncomputable def sector_area (s : Sector) : ℝ :=
  1/2 * s.radius * s.arc_length

-- Theorem statement
theorem max_area_at_two_radians :
  ∃ (s : Sector), forms_sector s ∧ 
    (∀ (t : Sector), forms_sector t → sector_area t ≤ sector_area s) ∧
    s.central_angle = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_at_two_radians_l276_27630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conjugate_complex_magnitude_l276_27654

theorem conjugate_complex_magnitude (α β : ℂ) : 
  (∃ (x y : ℝ), α = x + y * I ∧ β = x - y * I) →  -- α and β are conjugates
  (∃ (r : ℝ), α / β^3 = r) →  -- α/β^3 is real
  Complex.abs (α - β) = 6 →
  Complex.abs α = 6 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_conjugate_complex_magnitude_l276_27654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vita_balil_cali_problem_l276_27621

theorem vita_balil_cali_problem (x : ℝ) (h : 1 ≤ x ∧ x ≤ 10) : 
  (x + 5) - (x - 5) = 10 := by
  calc
    (x + 5) - (x - 5) = x + 5 - x + 5 := by ring
    _ = 10 := by ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vita_balil_cali_problem_l276_27621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_freken_bok_weight_l276_27687

/-- The weight of Karlson in kilograms -/
def K : ℝ := sorry

/-- The weight of Freken Bok in kilograms -/
def F : ℝ := sorry

/-- The weight of Malish in kilograms -/
def M : ℝ := sorry

/-- Karlson and Freken Bok together weigh 75 kg more than Malish -/
axiom h1 : K + F = M + 75

/-- Freken Bok and Malish together weigh 45 kg more than Karlson -/
axiom h2 : F + M = K + 45

/-- Freken Bok weighs 60 kg -/
theorem freken_bok_weight : F = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_freken_bok_weight_l276_27687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_skew_median_distances_theorem_l276_27692

/-- A regular tetrahedron with side length a -/
structure RegularTetrahedron (a : ℝ) where
  side_length : a > 0

/-- The distances between skew medians of two faces in a regular tetrahedron -/
noncomputable def skew_median_distances (t : RegularTetrahedron a) : ℝ × ℝ :=
  (a * Real.sqrt (2/35), a / Real.sqrt 10)

/-- Theorem: The distances between the skew medians of two faces of a regular tetrahedron
    with side length a are a * sqrt(2/35) and a / sqrt(10) -/
theorem skew_median_distances_theorem (a : ℝ) (t : RegularTetrahedron a) :
  skew_median_distances t = (a * Real.sqrt (2/35), a / Real.sqrt 10) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_skew_median_distances_theorem_l276_27692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l276_27688

theorem hyperbola_eccentricity (a b x₀ y₀ : ℝ) :
  a > 0 →
  b > 0 →
  x₀^2 / a^2 - y₀^2 / b^2 = 1 →
  a ≤ x₀ →
  x₀ ≤ 2 * a →
  x₀ * 0 / a^2 - y₀ * b / b^2 = 1 →
  (y₀ - b) / x₀ = -2 →
  Real.sqrt (1 + b^2 / a^2) = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l276_27688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_range_of_a_l276_27655

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2 * x^2 - 4 * a * x) * Real.log x

-- Theorem for part (1)
theorem tangent_line_at_one (a : ℝ) :
  a = 1 → ∃ m b : ℝ, ∀ x y : ℝ, y = m * (x - 1) + f 1 1 ↔ 2 * x + y - 2 = 0 :=
by sorry

-- Theorem for part (2)
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x ≥ 1 → f a x + x^2 - a > 0) → a < 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_range_of_a_l276_27655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_union_B_eq_open_one_infinity_l276_27662

open Set

def A : Set ℝ := {x | x ≤ 1 ∨ x > 3}
def B : Set ℝ := {x | x > 2}

theorem complement_A_union_B_eq_open_one_infinity : 
  (Aᶜ ∪ B) = Ioi 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_union_B_eq_open_one_infinity_l276_27662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_minimum_l276_27618

theorem triangle_ratio_minimum (A B C : ℝ) (a b c : ℝ) : 
  0 < B → B < π/2 →
  Real.cos B^2 + (1/2) * Real.sin (2*B) = 1 →
  ((c : ℝ) - a * Real.cos B)^2 + (a * Real.sin B)^2 = 3^2 →
  (∀ a' c' : ℝ, 
    0 < a' → 0 < c' → 
    ((c' : ℝ) - a' * Real.cos B)^2 + (a' * Real.sin B)^2 = 3^2 → 
    16 * b / (a * c) ≤ 16 * b / (a' * c')) →
  16 * b / (a * c) = 16 * (2 - Real.sqrt 2) / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_minimum_l276_27618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_multiples_11_and_8_l276_27643

theorem count_multiples_11_and_8 : 
  Finset.card (Finset.filter (λ n : ℕ => 100 ≤ n ∧ n ≤ 300 ∧ n % 11 = 0 ∧ n % 8 = 0) (Finset.range 301)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_multiples_11_and_8_l276_27643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_theorem_l276_27639

/-- The radius of the sphere inscribed in a right cone -/
noncomputable def sphere_radius (base_radius height : ℝ) : ℝ :=
  (base_radius * height) / (height + base_radius)

/-- Theorem: For a sphere inscribed in a right cone with base radius 15 cm and height 30 cm,
    if the sphere's radius can be expressed as b√d - b cm, then b + d = 12.5 -/
theorem inscribed_sphere_theorem (b d : ℝ) :
  sphere_radius 15 30 = b * (Real.sqrt d - 1) →
  b + d = 12.5 := by
  sorry

#check inscribed_sphere_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_theorem_l276_27639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_power_function_l276_27637

-- Define the exponential function
noncomputable def g (x : ℝ) : ℝ := 2^(x-2) + 7

-- Define the power function
noncomputable def f (α : ℝ) (x : ℝ) : ℝ := x^α

-- Theorem statement
theorem fixed_point_power_function :
  ∃ α : ℝ, (g 2 = 8) ∧ (f α 2 = 8) → f α 3 = 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_power_function_l276_27637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_three_digit_prime_numbers_l276_27699

def is_prime_digit (d : Nat) : Bool :=
  d = 2 || d = 3 || d = 5 || d = 7

def is_three_digit_prime_number (n : Nat) : Bool :=
  100 ≤ n && n ≤ 999 &&
  is_prime_digit (n / 100) &&
  is_prime_digit ((n / 10) % 10) &&
  is_prime_digit (n % 10)

theorem count_three_digit_prime_numbers :
  (Finset.filter (fun n => is_three_digit_prime_number n) (Finset.range 1000)).card = 64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_three_digit_prime_numbers_l276_27699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_and_inequality_range_l276_27619

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (1 - a^x) / Real.log a

theorem domain_and_inequality_range (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x : ℝ, 0 < a ∧ a < 1 → (0 < x → f a x ∈ Set.univ)) ∧
  (∀ x : ℝ, a > 1 → (x < 0 → f a x ∈ Set.univ)) ∧
  (∀ x : ℝ, 0 < a ∧ a < 1 → (0 < x ∧ x < 1 ↔ f a x > f a 1)) ∧
  (∀ x : ℝ, a > 1 → (x < 0 ↔ f a x > f a 1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_and_inequality_range_l276_27619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_values_l276_27666

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 3 * x - 4 else x^2 + 1

-- Theorem statement
theorem f_values : f (-1) = -7 ∧ f 2 = 5 := by
  -- Split the conjunction
  constructor
  
  -- Prove f (-1) = -7
  · simp [f]
    norm_num
  
  -- Prove f 2 = 5
  · simp [f]
    norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_values_l276_27666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_f_implies_a_geq_three_halves_l276_27696

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  (1/2) * (Real.cos x + Real.sin x) * (Real.cos x - Real.sin x - 4*a) + (4*a - 3) * x

theorem monotone_increasing_f_implies_a_geq_three_halves :
  ∀ a : ℝ, (∀ x y : ℝ, 0 ≤ x ∧ x < y ∧ y ≤ π/2 → f a x < f a y) → a ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_f_implies_a_geq_three_halves_l276_27696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_of_region_T_l276_27659

structure Circle where
  radius : ℝ
  center : ℝ × ℝ

def tangentToLine (c : Circle) (ℓ : ℝ → ℝ) : Prop :=
  ∃ (p : ℝ × ℝ), p.2 = ℓ p.1 ∧ (c.center.1 - p.1)^2 + (c.center.2 - p.2)^2 = c.radius^2

def outsideCircle (c1 c2 : Circle) : Prop :=
  (c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2 > (c1.radius + c2.radius)^2

noncomputable def areaInsideExactlyOne (circles : List Circle) : ℝ :=
  sorry

theorem max_area_of_region_T (ℓ : ℝ → ℝ) (B : ℝ × ℝ) (circles : List Circle) :
  circles.length = 4 ∧
  (∀ c, c ∈ circles → tangentToLine c ℓ) ∧
  (∃ c2 c4, c2 ∈ circles ∧ c4 ∈ circles ∧ c2.radius = 2 ∧ c4.radius = 4 ∧ outsideCircle c2 c4) ∧
  (∃ c6 c8, c6 ∈ circles ∧ c8 ∈ circles ∧ c6.radius = 6 ∧ c8.radius = 8) →
  areaInsideExactlyOne circles ≤ 120 * Real.pi :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_of_region_T_l276_27659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_max_value_g_min_value_sum_f_k_pi_180_l276_27649

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 4

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := f x + f (Real.pi / 2 - x)

-- Theorem for the maximum value of g
theorem g_max_value : 
  ∃ x ∈ Set.Icc (Real.pi / 6) (3 * Real.pi / 8), g x = 3/4 ∧ 
  ∀ y ∈ Set.Icc (Real.pi / 6) (3 * Real.pi / 8), g y ≤ 3/4 :=
by sorry

-- Theorem for the minimum value of g
theorem g_min_value : 
  ∃ x ∈ Set.Icc (Real.pi / 6) (3 * Real.pi / 8), g x = 1/2 ∧ 
  ∀ y ∈ Set.Icc (Real.pi / 6) (3 * Real.pi / 8), g y ≥ 1/2 :=
by sorry

-- Theorem for the sum of f(kπ/180)
theorem sum_f_k_pi_180 : 
  (Finset.range 89).sum (λ k => f ((k + 1 : ℕ) * Real.pi / 180)) = 133/4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_max_value_g_min_value_sum_f_k_pi_180_l276_27649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l276_27698

noncomputable def ellipse (a b : ℝ) (x y : ℝ) : Prop := x^2/a^2 + y^2/b^2 = 1

noncomputable def eccentricity (a c : ℝ) : ℝ := c/a

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := ((x2 - x1)^2 + (y2 - y1)^2)^(1/2)

theorem ellipse_properties (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : distance (-a) 0 a 0 = 4) (h4 : eccentricity a (a^2 - b^2)^(1/2) = 1/2) :
  (∃ P_y : ℝ, 
    -- 1. The equation of ellipse C is x²/4 + y²/3 = 1
    (∀ x y : ℝ, ellipse a b x y ↔ x^2/4 + y^2/3 = 1) ∧
    -- 2. There exists a point P(4,±3) such that APQM is a trapezoid
    (∃ M_x M_y : ℝ, 
      ellipse a b M_x M_y ∧
      distance (-a) 0 M_x M_y / distance (-a) 0 4 P_y = 
      distance a 0 4 0 / distance (-a) 0 a 0 ∧
      (P_y = 3 ∨ P_y = -3))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l276_27698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_three_simplest_l276_27642

-- Define the concept of a square root being in simplest form
def is_simplest_form (x : ℝ) : Prop :=
  ∀ y z : ℝ, (y * y = x ∧ z * z = x) → (y = z ∨ y = -z)

-- Define the given square roots
noncomputable def sqrt_half : ℝ := Real.sqrt (1/2)
noncomputable def sqrt_point_two : ℝ := Real.sqrt 0.2
noncomputable def sqrt_three : ℝ := Real.sqrt 3
noncomputable def sqrt_eight : ℝ := Real.sqrt 8

-- Theorem stating that sqrt(3) is in simplest form while others are not
theorem sqrt_three_simplest :
  is_simplest_form sqrt_three ∧
  ¬is_simplest_form sqrt_half ∧
  ¬is_simplest_form sqrt_point_two ∧
  ¬is_simplest_form sqrt_eight :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_three_simplest_l276_27642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_special_point_set_l276_27615

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space defined by two points -/
structure Line3D where
  p1 : Point3D
  p2 : Point3D

/-- Check if two lines are parallel -/
def are_parallel (l1 l2 : Line3D) : Prop := sorry

/-- Check if two lines are distinct -/
def are_distinct (l1 l2 : Line3D) : Prop := sorry

/-- Check if a set of points is coplanar -/
def is_coplanar (s : Set Point3D) : Prop := sorry

/-- The main theorem -/
theorem exists_special_point_set :
  ∃ (M : Set Point3D),
    (Finite M) ∧
    (¬ is_coplanar M) ∧
    (∀ A B : Point3D, A ∈ M → B ∈ M → A ≠ B →
      ∃ C D : Point3D, C ∈ M ∧ D ∈ M ∧
        are_parallel (Line3D.mk A B) (Line3D.mk C D) ∧
        are_distinct (Line3D.mk A B) (Line3D.mk C D)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_special_point_set_l276_27615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_four_vertices_figures_l276_27609

-- Define a cube
structure Cube where
  vertices : Finset (Fin 8)

-- Define a selection of 4 vertices from a cube
def FourVertices (c : Cube) : Finset (Finset (Fin 8)) :=
  c.vertices.powerset.filter (fun s => s.card = 4)

-- Define the possible geometric figures
inductive GeometricFigure
  | Rectangle
  | NonRectangleParallelogram
  | TetrahedronWithIsoscelesRightTriangles
  | TetrahedronWithEquilateralTriangles

-- Helper function to determine if a set of vertices can form a geometric figure
def can_form (vertices : Finset (Fin 8)) (figure : GeometricFigure) : Prop :=
  sorry

-- Theorem stating which geometric figures can be formed
theorem cube_four_vertices_figures (c : Cube) :
  ∀ s ∈ FourVertices c,
    (∃ f : GeometricFigure, f = GeometricFigure.Rectangle ∧ can_form s f) ∧
    (∃ f : GeometricFigure, f = GeometricFigure.TetrahedronWithIsoscelesRightTriangles ∧ can_form s f) ∧
    (∃ f : GeometricFigure, f = GeometricFigure.TetrahedronWithEquilateralTriangles ∧ can_form s f) ∧
    ¬(∀ s ∈ FourVertices c, ∃ f : GeometricFigure, f = GeometricFigure.NonRectangleParallelogram ∧ can_form s f) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_four_vertices_figures_l276_27609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_other_communities_count_l276_27624

theorem other_communities_count (total : ℕ) (muslim_percent : ℚ) (hindu_percent : ℚ) (sikh_percent : ℚ)
  (h_total : total = 650)
  (h_muslim : muslim_percent = 44 / 100)
  (h_hindu : hindu_percent = 28 / 100)
  (h_sikh : sikh_percent = 10 / 100) :
  Int.floor (↑total * (1 - (muslim_percent + hindu_percent + sikh_percent))) = 117 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_other_communities_count_l276_27624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_v_satisfies_conditions_l276_27683

/-- The direction vector of line m -/
noncomputable def m_direction : ℝ × ℝ := (3, -2)

/-- The vector we're looking for -/
noncomputable def v : ℝ × ℝ := (-18/5, 12/5)

/-- Theorem stating that v satisfies the required conditions -/
theorem v_satisfies_conditions :
  -- v is perpendicular to m_direction
  v.1 * m_direction.1 + v.2 * m_direction.2 = 0 ∧
  -- v satisfies the equation 3v₁ + 2v₂ = 6
  3 * v.1 + 2 * v.2 = 6 ∧
  -- v is parallel to m_direction (which implies it's the correct projection vector)
  ∃ (k : ℝ), v = (k * m_direction.1, k * m_direction.2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_v_satisfies_conditions_l276_27683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_arrangement_count_constrained_arrangement_count_task_assignment_count_l276_27653

-- Define the number of boys and girls
def num_boys : ℕ := 3
def num_girls : ℕ := 4
def total_people : ℕ := num_boys + num_girls

-- Part 1: Alternating arrangement
theorem alternating_arrangement_count : 
  (Nat.factorial num_boys) * (Nat.factorial num_girls) = 144 :=
by sorry

-- Part 2: Arrangement with constraints on Boy A and Boy B
theorem constrained_arrangement_count : 
  (Nat.factorial (total_people - 1)) + 
  (total_people - 2) * (total_people - 2) * (Nat.factorial (total_people - 2)) = 3720 :=
by sorry

-- Part 3: Task assignment
theorem task_assignment_count : 
  (Nat.choose num_boys 2) * (Nat.choose num_girls 2) * (Nat.factorial 4) = 432 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_arrangement_count_constrained_arrangement_count_task_assignment_count_l276_27653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slice_area_theorem_l276_27606

noncomputable def cylinder_radius : ℝ := 5
noncomputable def cylinder_height : ℝ := 10

noncomputable def arc_angle : ℝ := Real.pi / 2  -- 90° in radians

-- Theorem statement
theorem slice_area_theorem :
  let sector_area := (1/2) * cylinder_radius^2 * arc_angle
  let stretch_ratio := (cylinder_radius + cylinder_height/2) / cylinder_radius
  stretch_ratio * sector_area = 13.75 * Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slice_area_theorem_l276_27606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_digit_count_l276_27660

def num1 : Nat := 876543210987654
def num2 : Nat := 4321098765

theorem product_digit_count : 
  (Nat.digits 10 (num1 * num2)).length = 24 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_digit_count_l276_27660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficiency_not_necessity_l276_27686

theorem sufficiency_not_necessity : 
  (∃ a b : ℕ, (∃ k : ℕ, a^2 + b^2 = 8 * k) → (∃ m : ℕ, a^3 + b^3 = 16 * m)) ∧ 
  (∃ a b : ℕ, (∃ n : ℕ, a^3 + b^3 = 16 * n) ∧ (∀ k : ℕ, a^2 + b^2 ≠ 8 * k)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficiency_not_necessity_l276_27686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_basic_terms_div_four_l276_27644

/-- A grid filled with +1 or -1 -/
def Grid (n : ℕ) := Fin n → Fin n → Int

/-- A basic term is a product of n cells, with no two cells in the same row or column -/
def BasicTerm (n : ℕ) (grid : Grid n) (perm : Equiv.Perm (Fin n)) : Int :=
  (Finset.univ.prod fun i => grid i (perm i))

/-- The sum of all basic terms -/
def SumOfBasicTerms (n : ℕ) (grid : Grid n) : Int :=
  Finset.sum (Finset.univ : Finset (Equiv.Perm (Fin n))) (fun perm => BasicTerm n grid perm)

/-- The main theorem: the sum of all basic terms is divisible by 4 for n ≥ 4 -/
theorem sum_of_basic_terms_div_four (n : ℕ) (grid : Grid n) (h : n ≥ 4) :
  4 ∣ SumOfBasicTerms n grid := by
  sorry

#check sum_of_basic_terms_div_four

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_basic_terms_div_four_l276_27644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_distances_sum_ge_three_inradius_min_sum_distances_l276_27663

/-- Triangle with sides a, b, c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : a > 0
  pos_b : b > 0
  pos_c : c > 0
  triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b

/-- Altitude of a triangle -/
noncomputable def altitude (t : Triangle) : ℝ → ℝ := sorry

/-- Inradius of a triangle -/
noncomputable def inradius (t : Triangle) : ℝ := sorry

/-- Sum of distances from centroid to sides -/
noncomputable def centroid_distances_sum (t : Triangle) : ℝ :=
  (altitude t t.a + altitude t t.b + altitude t t.c) / 3

/-- Theorem: Sum of distances from centroid to sides is not less than 3 times the inradius -/
theorem centroid_distances_sum_ge_three_inradius (t : Triangle) :
  centroid_distances_sum t ≥ 3 * inradius t := by sorry

/-- Points that minimize the sum of distances to the sides -/
noncomputable def min_distance_points (t : Triangle) : Set ℝ := sorry

/-- Theorem: The minimum sum of distances is min(h_a, h_b, h_c) -/
theorem min_sum_distances (t : Triangle) :
  ∃ p ∈ min_distance_points t, 
    altitude t t.a = altitude t t.b ∧ altitude t t.b = altitude t t.c ∨
    altitude t t.a = altitude t t.b ∧ altitude t t.a < altitude t t.c ∨
    altitude t t.b = altitude t t.c ∧ altitude t t.b < altitude t t.a ∨
    altitude t t.c = altitude t t.a ∧ altitude t t.c < altitude t t.b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_distances_sum_ge_three_inradius_min_sum_distances_l276_27663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_square_perimeter_l276_27634

/-- The perimeter of an equilateral triangle exceeds the perimeter of a square by 108 cm. -/
def perimeter_difference : ℝ := 108

/-- The length of each side of the triangle exceeds the length of each side of the square by d+2 cm. -/
def side_difference (d : ℝ) : ℝ := d + 2

/-- The square has a perimeter greater than 0. -/
def square_perimeter_positive (s : ℝ) : Prop := s > 0

/-- The number of positive integers that are not possible values for d. -/
def impossible_d_count : ℕ := 34

theorem triangle_square_perimeter (d : ℝ) :
  (∃ (s t : ℝ), 
    square_perimeter_positive s ∧
    3 * t - 4 * s = perimeter_difference ∧
    t - s = side_difference d) →
  (∃ (n : ℕ), n = impossible_d_count ∧ 
    ∀ (k : ℕ), k ≤ n → ¬(d = ↑k)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_square_perimeter_l276_27634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_is_correct_l276_27658

-- Define the given parameters
noncomputable def bridge_length : ℝ := 150
noncomputable def crossing_time : ℝ := 12.499
noncomputable def train_speed_kmph : ℝ := 72

-- Convert train speed from km/h to m/s
noncomputable def train_speed_ms : ℝ := train_speed_kmph * 1000 / 3600

-- Calculate the total distance covered
noncomputable def total_distance : ℝ := train_speed_ms * crossing_time

-- Define the train length
noncomputable def train_length : ℝ := total_distance - bridge_length

-- Theorem to prove
theorem train_length_is_correct : 
  ∀ ε > 0, |train_length - 99.98| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_is_correct_l276_27658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_periodicity_implies_zero_l276_27677

theorem sequence_periodicity_implies_zero 
  (a b c d : ℕ → ℝ)
  (ha : ∀ n, a (n + 1) = a n + b n)
  (hb : ∀ n, b (n + 1) = b n + c n)
  (hc : ∀ n, c (n + 1) = c n + d n)
  (hd : ∀ n, d (n + 1) = d n + a n)
  (hperiod : ∃ (k m : ℕ), k ≥ 1 ∧ m ≥ 1 ∧
    (∀ n, a (k + n) = a n ∧ 
          b (k + n) = b n ∧ 
          c (k + n) = c n ∧ 
          d (k + n) = d n)) :
  a 2 = 0 ∧ b 2 = 0 ∧ c 2 = 0 ∧ d 2 = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_periodicity_implies_zero_l276_27677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_properties_l276_27676

/-- Definition of the circle --/
def my_circle (x y : ℝ) : Prop := x^2 + y^2 = 3

/-- Definition of the relation between P and M --/
def relation (x y x_0 y_0 : ℝ) : Prop :=
  my_circle x_0 y_0 ∧ x = x_0 ∧ y_0 = Real.sqrt 3 * y

/-- Definition of the curve C --/
def curve_C (x y : ℝ) : Prop := x^2 / 3 + y^2 = 1

/-- Definition of the line l --/
def line_l (k m : ℝ) : Prop := m^2 = 3/4 * (k^2 + 1)

theorem curve_C_properties :
  (∀ x y, (∃ x_0 y_0, relation x y x_0 y_0) ↔ curve_C x y) ∧
  (∃ e : ℝ, e = Real.sqrt 6 / 3 ∧ 
    ∀ x y, curve_C x y → (x / 3)^2 + y^2 = 1 - e^2 * (x / 3)^2) ∧
  (∃ S : ℝ, S = Real.sqrt 3 / 2 ∧
    ∀ k m : ℝ, ∀ A B : ℝ × ℝ,
      line_l k m →
      curve_C A.1 A.2 →
      curve_C B.1 B.2 →
      (A.2 - B.2) = k * (A.1 - B.1) →
      (A.2 + B.2) / 2 = k * (A.1 + B.1) / 2 + m →
      (A.1 - B.1)^2 + (A.2 - B.2)^2 ≤ 4 ∧
      ((A.1 - B.1)^2 + (A.2 - B.2)^2) * (Real.sqrt 3 / 4) ≤ S) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_properties_l276_27676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_not_divisible_by_three_l276_27602

theorem divisors_not_divisible_by_three (n : ℕ) (h : n = 252) :
  (Finset.filter (λ d ↦ d ∣ n ∧ ¬(3 ∣ d)) (Finset.range (n + 1))).card = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_not_divisible_by_three_l276_27602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cot_30_degrees_l276_27616

theorem cot_30_degrees : Real.tan (π / 6) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cot_30_degrees_l276_27616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_characterization_l276_27612

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function f is monotonically increasing on (0, +∞) if
    for all x, y > 0, x < y implies f(x) < f(y) -/
def MonoIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → x < y → f x < f y

/-- The set of all real numbers a satisfying the given inequality -/
def SolutionSet (f : ℝ → ℝ) : Set ℝ :=
  {a | f (Real.exp (|1/2 * a - 1|)) + f (-Real.sqrt (Real.exp 1)) < 0}

theorem solution_set_characterization (f : ℝ → ℝ) 
    (h_odd : IsOdd f) (h_mono : MonoIncreasing f) :
    SolutionSet f = Set.Ioo 1 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_characterization_l276_27612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_exp_l276_27673

/-- Sequence a_n defined recursively -/
def a : ℕ → ℕ
  | 0 => 1
  | n+1 => a n + 2*n*(a (n-1)) + 9*n*(n-1)*(a (n-2)) + 8*n*(n-1)*(n-2)*(a (n-3))

/-- The sum we want to compute -/
noncomputable def seriesSum : ℝ := ∑' n : ℕ, (10 ^ n * a n : ℝ) / n.factorial

/-- The main theorem -/
theorem series_sum_equals_exp : seriesSum = Real.exp 23110 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_exp_l276_27673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_jar_price_l276_27657

/-- Price of a cylindrical jar of peanut butter given its dimensions and a reference jar -/
noncomputable def price_of_jar (d1 r1 h1 p1 d2 r2 h2 : ℝ) : ℝ :=
  let v1 := Real.pi * (d1 / 2)^2 * h1
  let v2 := Real.pi * (d2 / 2)^2 * h2
  (v2 / v1) * p1

/-- The price of a larger jar given the dimensions and price of a smaller jar -/
theorem larger_jar_price :
  price_of_jar 4 2 5 0.9 12 6 10 = 16.2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_jar_price_l276_27657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_cookie_flour_measurement_l276_27669

theorem alice_cookie_flour_measurement :
  let flour_needed : ℚ := 19/4  -- 4¾ converted to improper fraction
  let cup_capacity : ℚ := 1/3
  let fills : ℕ := (Int.ceil (flour_needed / cup_capacity)).toNat
  fills = 15 := by
    -- Proof goes here
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_cookie_flour_measurement_l276_27669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_mixture_butterfat_percentage_l276_27629

/-- Calculates the butterfat percentage in a mixture of two types of milk -/
noncomputable def butterfat_percentage (vol1 : ℝ) (bf1 : ℝ) (vol2 : ℝ) (bf2 : ℝ) : ℝ :=
  ((vol1 * bf1 + vol2 * bf2) / (vol1 + vol2)) * 100

theorem milk_mixture_butterfat_percentage :
  let initial_volume : ℝ := 8
  let initial_butterfat : ℝ := 35
  let added_volume : ℝ := 4
  let added_butterfat : ℝ := 10
  let total_volume : ℝ := initial_volume + added_volume
  ∃ ε > 0, |butterfat_percentage initial_volume (initial_butterfat / 100) added_volume (added_butterfat / 100) - 26.67| < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_mixture_butterfat_percentage_l276_27629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_center_and_perpendicular_l276_27628

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + 2*x + y^2 = 0

-- Define the given line
def given_line (x y : ℝ) : Prop := x + y = 0

-- Define the resulting line
def result_line (x y : ℝ) : Prop := x - y + 1 = 0

-- Helper function to get y from x in the result line
def result_line_y (x : ℝ) : ℝ := x + 1

-- Theorem statement
theorem line_through_center_and_perpendicular :
  ∃ (cx cy : ℝ),
    (∀ (x y : ℝ), my_circle x y ↔ (x - cx)^2 + (y - cy)^2 = (-cx)^2 + (-cy)^2) ∧
    result_line cx cy ∧
    (∀ (x₁ y₁ x₂ y₂ : ℝ),
      given_line x₁ y₁ ∧ given_line x₂ y₂ ∧ x₁ ≠ x₂ →
      (y₂ - y₁) * (result_line_y x₂ - result_line_y x₁) = -(x₂ - x₁) * (x₂ - x₁)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_center_and_perpendicular_l276_27628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_sum_positive_l276_27691

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function f is monotonically decreasing on a set S if
    for all x, y in S, x < y implies f(x) > f(y) -/
def MonoDecreasing (f : ℝ → ℝ) (S : Set ℝ) : Prop :=
  ∀ x y, x ∈ S → y ∈ S → x < y → f x > f y

theorem odd_function_sum_positive
    (f : ℝ → ℝ)
    (hodd : IsOdd f)
    (hdecr : MonoDecreasing f { x : ℝ | x ≥ 0 })
    (x₁ x₂ : ℝ)
    (hsum : x₁ + x₂ < 0) :
  f x₁ + f x₂ > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_sum_positive_l276_27691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semi_minor_axis_specific_ellipse_l276_27684

/-- The semi-minor axis of an ellipse with given center, focus, and endpoint of semi-major axis -/
noncomputable def semi_minor_axis (center focus endpoint : ℝ × ℝ) : ℝ :=
  let c := Real.sqrt ((center.1 - focus.1)^2 + (center.2 - focus.2)^2)
  let a := Real.sqrt ((center.1 - endpoint.1)^2 + (center.2 - endpoint.2)^2)
  Real.sqrt (a^2 - c^2)

/-- Theorem: The semi-minor axis of the specified ellipse is √5 -/
theorem semi_minor_axis_specific_ellipse :
  semi_minor_axis (2, -1) (2, -3) (2, 2) = Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_semi_minor_axis_specific_ellipse_l276_27684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l276_27648

noncomputable section

-- Define the function f
def f (ω : ℝ) (x : ℝ) : ℝ := 4 * Real.sin (ω * x - Real.pi / 4) * Real.cos (ω * x)

-- Define the function g
def g (x : ℝ) : ℝ := 2 * Real.sin (x - Real.pi / 6) - Real.sqrt 2

-- Define IsExtremeValue
def IsExtremeValue (f : ℝ → ℝ) (x₀ : ℝ) (y : ℝ) : Prop :=
  f x₀ = y ∧ ∀ x, f x ≤ y ∨ f x ≥ y

-- Define IsSmallestPositivePeriod
def IsSmallestPositivePeriod (f : ℝ → ℝ) (T : ℝ) : Prop :=
  T > 0 ∧ (∀ x, f (x + T) = f x) ∧ ∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T'

theorem function_properties 
  (ω : ℝ) 
  (h1 : 0 < ω ∧ ω < 2) 
  (h2 : ∃ (y : ℝ), IsExtremeValue (f ω) (Real.pi / 4) y) :
  (∃ (T : ℝ), T > 0 ∧ IsSmallestPositivePeriod (f ω) T ∧ T = 2 * Real.pi / 3) ∧
  (∀ (α : ℝ), 0 < α ∧ α < Real.pi / 2 → g α = 4 / 3 - Real.sqrt 2 → 
    Real.cos α = (Real.sqrt 15 - 2) / 6) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l276_27648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l276_27603

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (Real.exp x - Real.exp (-x)) / 2

-- State the theorem
theorem range_of_m (m : ℝ) : 
  (∀ θ : ℝ, 0 < θ ∧ θ ≤ Real.pi / 2 → f (Real.sin θ) + f (1 - m) > 0) → 
  m ≤ 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l276_27603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_areas_l276_27635

/-- Represents a convex quadrilateral divided into four triangles by its diagonals -/
structure QuadrilateralWithDiagonals where
  area1 : ℕ
  area2 : ℕ
  area3 : ℕ
  area4 : ℕ

/-- Property that the product of opposite triangle areas are equal -/
def opposite_areas_product_equal (q : QuadrilateralWithDiagonals) : Prop :=
  q.area1 * q.area3 = q.area2 * q.area4

theorem impossible_areas (q : QuadrilateralWithDiagonals) 
  (h : opposite_areas_product_equal q) :
  ¬(∃ (s : Finset ℕ), s.card = 3 ∧ s ⊆ {q.area1, q.area2, q.area3, q.area4} ∧ s = {2001, 2002, 2003}) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_areas_l276_27635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_one_plus_i_l276_27681

/-- The complex number i -/
noncomputable def i : ℂ := Complex.I

/-- The function g(x) = (x^3 - 2x) / (x - i) -/
noncomputable def g (x : ℂ) : ℂ := (x^3 - 2*x) / (x - i)

theorem g_of_one_plus_i : g (1 + i) = -4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_one_plus_i_l276_27681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_three_digit_interesting_arbitrarily_large_interesting_l276_27647

/-- A natural number is interesting if any natural number not exceeding it
    can be represented as the sum of several (possibly one) pairwise distinct positive divisors of it. -/
def interesting (n : ℕ) : Prop :=
  ∀ m ≤ n, ∃ (s : Finset ℕ), (∀ x ∈ s, x ∣ n ∧ x > 0) ∧ s.sum id = m ∧ s.card = (s : Set ℕ).toFinset.card

/-- 992 is the largest three-digit interesting number -/
theorem largest_three_digit_interesting : 
  interesting 992 ∧ ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n > 992 → ¬ interesting n :=
sorry

/-- For all odd n and k ≥ ⌊log₂ n⌋, 2ᵏn is interesting -/
theorem arbitrarily_large_interesting (n : ℕ) (k : ℕ) :
  Odd n → k ≥ Nat.log 2 n → interesting (2^k * n) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_three_digit_interesting_arbitrarily_large_interesting_l276_27647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_n_l276_27671

def alternating_factorial_sum (n : ℕ) : ℤ :=
  List.range n |> List.map (fun i => (-1 : ℤ)^i * (Nat.factorial (n - i) : ℤ)) |> List.sum

def is_valid (n N : ℕ) : Prop :=
  n < N ∧ Nat.Prime (Int.natAbs (alternating_factorial_sum n))

theorem exists_valid_n (N : ℕ) : ∃ n, is_valid n N := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_n_l276_27671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_is_120pi_l276_27665

/-- Represents a circle with a given radius -/
structure Circle where
  radius : ℝ

/-- Represents the configuration of four circles tangent to a line -/
structure CircleConfiguration where
  circles : Fin 4 → Circle
  sumOfSmallerRadii : (circles 0).radius + (circles 1).radius + (circles 2).radius ≥ (circles 3).radius

/-- Calculates the area of a circle -/
noncomputable def circleArea (c : Circle) : ℝ := Real.pi * c.radius * c.radius

/-- Calculates the maximum area covered by points inside exactly one of the circles -/
noncomputable def maxAreaCoveredByOneCircle (config : CircleConfiguration) : ℝ :=
  -- This function should calculate the maximum area as described in the problem
  sorry

/-- The theorem stating the maximum area covered by points inside exactly one circle -/
theorem max_area_is_120pi (config : CircleConfiguration) 
  (h1 : (config.circles 0).radius = 2)
  (h2 : (config.circles 1).radius = 4)
  (h3 : (config.circles 2).radius = 6)
  (h4 : (config.circles 3).radius = 10) :
  maxAreaCoveredByOneCircle config = 120 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_is_120pi_l276_27665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_project_time_B_correct_l276_27605

-- Define the project completion times
noncomputable def project_time_A (x : ℝ) : ℝ := x
noncomputable def project_time_AB (y : ℝ) : ℝ := y

-- Define the function to calculate person B's project time
noncomputable def project_time_B (x y : ℝ) : ℝ := (x * y) / (x - y)

-- Theorem statement
theorem project_time_B_correct (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) :
  project_time_B x y = (x * y) / (x - y) :=
by
  -- Unfold the definition of project_time_B
  unfold project_time_B
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_project_time_B_correct_l276_27605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_η_l276_27623

/-- A random variable distributed as B(n, p) -/
structure BinomialDist (n : ℕ) (p : ℝ) where

/-- The expected value of a binomial distribution -/
def expectedBinomial (n : ℕ) (p : ℝ) : ℝ := n * p

/-- A random variable -/
structure RandomVariable where

/-- The expected value of a random variable -/
noncomputable def expectedValue (X : RandomVariable) : ℝ := 0

variable (ξ : RandomVariable)
variable (η : RandomVariable)

/-- ξ is distributed as B(5, 1/3) -/
axiom ξ_dist : ξ = RandomVariable.mk

/-- η is defined as 2ξ - 1 -/
axiom η_def : η = RandomVariable.mk

/-- The expected value of ξ -/
axiom ξ_expected : expectedValue ξ = 5 * (1/3)

/-- The expected value of a linear transformation of a random variable -/
axiom expectedLinearTransform (a b : ℝ) (X : RandomVariable) :
  expectedValue (RandomVariable.mk) = a * (expectedValue X) + b

theorem expected_value_η : expectedValue η = 7/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_η_l276_27623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_differential_equation_classification_l276_27679

-- Define a structure for differential equations
structure DiffEq where
  lhs : ℝ → ℝ → ℝ → ℝ  -- Left-hand side of the equation (x, y, y')
  rhs : ℝ → ℝ           -- Right-hand side of the equation

-- Define what it means for a differential equation to be first-order linear
def is_first_order_linear (eq : DiffEq) : Prop :=
  ∃ (p q : ℝ → ℝ), ∀ x y y',
    eq.lhs x y y' = y' + p x * y ∧ eq.rhs x = q x

-- Define the three given equations
noncomputable def eq1 : DiffEq :=
  { lhs := λ x y y' => y' + (2 * y) / (x + 1),
    rhs := λ x => (x + 1)^3 }

noncomputable def eq2 : DiffEq :=
  { lhs := λ x y y' => 0,  -- We can't represent y'' directly, so we use 0 as a placeholder
    rhs := λ _ => 0 }

noncomputable def eq3 : DiffEq :=
  { lhs := λ x y y' => y' + x * y^2,
    rhs := λ x => (x - 3)^2 }

-- State the theorem
theorem differential_equation_classification :
  is_first_order_linear eq1 ∧
  ¬is_first_order_linear eq2 ∧
  ¬is_first_order_linear eq3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_differential_equation_classification_l276_27679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_volume_in_pyramid_l276_27690

/-- Represents a pyramid with a square base and equilateral triangle lateral faces -/
structure Pyramid where
  base_side : ℝ
  base_side_positive : 0 < base_side

/-- Represents a cube inside the pyramid -/
structure CubeInPyramid (p : Pyramid) where
  side : ℝ
  side_positive : 0 < side
  fits_in_pyramid : side ≤ p.base_side
  top_center_at_apex : side = Real.sqrt 6 / 2

/-- The volume of the cube inside the pyramid is 3√6/4 -/
theorem cube_volume_in_pyramid (p : Pyramid) (c : CubeInPyramid p) 
    (h : p.base_side = 2) : c.side ^ 3 = 3 * Real.sqrt 6 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_volume_in_pyramid_l276_27690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l276_27626

theorem problem_statement (a b : ℝ) (h : Set.toFinset {a, b/a, 1} = Set.toFinset {a^2, a+b, 0}) : 
  a^2023 + b^2024 = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l276_27626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_final_amount_l276_27656

/-- Calculates the final amount after compound interest --/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (compounds_per_year : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate / compounds_per_year) ^ (compounds_per_year * time)

/-- Theorem: Given the specified conditions, the final amount is $16537.50 --/
theorem investment_final_amount :
  let principal := 15000
  let rate := 0.10
  let compounds_per_year := 2
  let time := 1
  compound_interest principal rate compounds_per_year time = 16537.50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_final_amount_l276_27656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_no_intersection_l276_27633

/-- The function f(x) defined as 3^(-|x-1|) + m -/
noncomputable def f (x m : ℝ) : ℝ := Real.exp (Real.log 3 * (-abs (x - 1))) + m

/-- Theorem stating the condition for f(x) not intersecting the x-axis -/
theorem f_no_intersection (m : ℝ) :
  (∀ x, f x m ≠ 0) ↔ (m ≥ 0 ∨ m < -1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_no_intersection_l276_27633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brokerage_percentage_approx_l276_27650

/-- Given the cash realized on selling a stock and the cash amount before brokerage,
    calculate the brokerage percentage. -/
noncomputable def brokerage_percentage (cash_realized : ℝ) (cash_before_brokerage : ℝ) : ℝ :=
  ((cash_realized - cash_before_brokerage) / cash_before_brokerage) * 100

/-- The brokerage percentage is approximately 0.23% when the cash realized
    on selling a 14% stock is Rs. 109.25 and the cash amount before brokerage is Rs. 109. -/
theorem brokerage_percentage_approx :
  let cash_realized : ℝ := 109.25
  let cash_before_brokerage : ℝ := 109
  abs (brokerage_percentage cash_realized cash_before_brokerage - 0.23) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_brokerage_percentage_approx_l276_27650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_articles_l276_27694

-- Define the number of articles in the cost price
variable (N : ℝ)

-- Define the cost price of one article
variable (C : ℝ)

-- Define the selling price of one article
variable (S : ℝ)

-- Condition 1: The cost price of N articles equals the selling price of 40 articles
axiom cost_price_equals_selling_price : N * C = 40 * S

-- Condition 2: The profit percentage is 49.999999999999986%
axiom profit_percentage : (S - C) / C * 100 = 49.999999999999986

-- Theorem: N equals 60
theorem number_of_articles : N = 60 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_articles_l276_27694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_difference_implies_principal_l276_27640

/-- Calculate compound interest for one year with half-yearly compounding -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) : ℝ :=
  principal * ((1 + rate / 2) ^ 2 - 1)

/-- Calculate simple interest for one year -/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) : ℝ :=
  principal * rate

/-- The problem statement -/
theorem interest_difference_implies_principal :
  ∀ (principal : ℝ),
    principal > 0 →
    compound_interest principal 0.1 - simple_interest principal 0.1 = 4.25 →
    principal = 1700 := by
  intro principal h_pos h_diff
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_difference_implies_principal_l276_27640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circumradius_l276_27632

noncomputable section

def circumradius (A B C : ℝ × ℝ) : ℝ := sorry

def acute_triangle (A B C : ℝ × ℝ) : Prop := sorry
def is_angle_bisector (A D B C : ℝ × ℝ) : Prop := sorry
def bisects (D O H : ℝ × ℝ) : Prop := sorry
def is_circumcenter (O A B C : ℝ × ℝ) : Prop := sorry
def is_orthocenter (H A B C : ℝ × ℝ) : Prop := sorry

theorem triangle_circumradius 
  (A B C : ℝ × ℝ) 
  (D : ℝ × ℝ)
  (O : ℝ × ℝ) 
  (H : ℝ × ℝ) :
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  let AD := Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2)
  acute_triangle A B C →
  is_angle_bisector A D B C →
  bisects D O H →
  is_circumcenter O A B C →
  is_orthocenter H A B C →
  AC = 2 →
  AD = Real.sqrt 3 + Real.sqrt 2 - 1 →
  circumradius A B C = (Real.sqrt 6 - Real.sqrt 2 + 2) / Real.sqrt (2 + Real.sqrt 2) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circumradius_l276_27632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l276_27668

/-- The function f(x) = √(x(40 - x)) + √(x(5 - x)) --/
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x * (40 - x)) + Real.sqrt (x * (5 - x))

/-- The theorem stating the maximum value and where it's attained --/
theorem max_value_of_f :
  ∃ (x₀ : ℝ), 0 ≤ x₀ ∧ x₀ ≤ 5 ∧
  (∀ x, 0 ≤ x ∧ x ≤ 5 → f x ≤ f x₀) ∧
  x₀ = 40 / 9 ∧ f x₀ = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l276_27668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_sqrt_3_l276_27672

open Real

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the conditions
def isAcute (t : Triangle) : Prop :=
  0 < t.A ∧ t.A < Real.pi/2 ∧
  0 < t.B ∧ t.B < Real.pi/2 ∧
  0 < t.C ∧ t.C < Real.pi/2

def angleSum (t : Triangle) : Prop :=
  t.A + t.B + t.C = Real.pi

def givenConditions (t : Triangle) : Prop :=
  t.b = 2 ∧
  t.B = Real.pi/3 ∧
  Real.sin (2 * t.A) + Real.sin (t.A - t.C) - Real.sin t.B = 0

-- Theorem statement
theorem triangle_area_is_sqrt_3 (t : Triangle) 
  (h1 : isAcute t) 
  (h2 : angleSum t) 
  (h3 : givenConditions t) : 
  (Real.sqrt 3 / 4) * t.b^2 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_sqrt_3_l276_27672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_continued_fraction_l276_27675

/-- Represents a digit from 1 to 9 -/
def Digit := Fin 9

/-- The continued fraction expression -/
def continuedFraction (P A B H O : Digit) : ℚ :=
  P.val + 1 / (A.val + 1 / (B.val + 1 / (H.val + 1 / O.val)))

/-- All digits in the fraction are distinct -/
def distinctDigits (P A B H O : Digit) : Prop :=
  P ≠ A ∧ P ≠ B ∧ P ≠ H ∧ P ≠ O ∧
  A ≠ B ∧ A ≠ H ∧ A ≠ O ∧
  B ≠ H ∧ B ≠ O ∧
  H ≠ O

theorem smallest_continued_fraction :
  ∀ P A B H O : Digit,
    distinctDigits P A B H O →
    continuedFraction P A B H O ≥ 555 / 502 ∧
    ∃ P A B H O : Digit, distinctDigits P A B H O ∧ continuedFraction P A B H O = 555 / 502 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_continued_fraction_l276_27675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_of_full_number_l276_27638

def repeated_sequence : ℕ := 123456

def full_number : ℕ := 
  let num_repeats := 200
  let base := 10^6
  (repeated_sequence * (base^num_repeats - 1)) / (base - 1)

theorem remainder_of_full_number : 
  full_number % 789 = 351 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_of_full_number_l276_27638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_shift_l276_27607

noncomputable def f (φ : ℝ) (x : ℝ) : ℝ := Real.cos (2 * (x + φ) + Real.pi / 3)

theorem symmetry_implies_shift (φ : ℝ) :
  (∀ x, f φ x = f φ (-x)) → φ = Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_shift_l276_27607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_equations_l276_27614

/-- Converts polar coordinates (r, θ) to Cartesian coordinates (x, y) -/
noncomputable def polar_to_cartesian (r : ℝ) (θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

/-- Defines the Cartesian equations of the circle and line -/
def circle_and_line_cartesian_equations 
  (C_center : ℝ × ℝ) 
  (C_radius : ℝ) 
  (l_angle : ℝ) : Prop :=
  let (x_c, y_c) := polar_to_cartesian C_center.1 C_center.2
  -- Circle equation
  (∀ x y : ℝ, (x - x_c)^2 + (y - y_c)^2 = C_radius^2 ↔ 
    ∃ r θ : ℝ, polar_to_cartesian r θ = (x, y) ∧ 
    r = C_center.1 ∧ θ = C_center.2 ∧ 
    (x - x_c)^2 + (y - y_c)^2 = C_radius^2)
  -- Line equation
  ∧ (∀ x y : ℝ, y = x ↔ 
    ∃ ρ : ℝ, polar_to_cartesian ρ l_angle = (x, y))

/-- The main theorem stating the Cartesian equations of the circle and line -/
theorem circle_and_line_equations : 
  circle_and_line_cartesian_equations (Real.sqrt 2, Real.pi/4) (Real.sqrt 3) (Real.pi/4) := by
  sorry

#check circle_and_line_equations

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_equations_l276_27614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_proof_l276_27600

/-- A plane in 3D space represented by its equation coefficients -/
structure Plane where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ
  A_pos : A > 0
  gcd_one : Nat.gcd (Nat.gcd (Int.natAbs A) (Int.natAbs B)) (Nat.gcd (Int.natAbs C) (Int.natAbs D)) = 1

/-- A point in 3D space -/
structure Point3D where
  x : ℚ
  y : ℚ
  z : ℚ

/-- Check if a point lies on a plane -/
def Point3D.liesOn (p : Point3D) (plane : Plane) : Prop :=
  plane.A * p.x + plane.B * p.y + plane.C * p.z + plane.D = 0

/-- Check if two planes are perpendicular -/
def Plane.isPerpendicular (p1 p2 : Plane) : Prop :=
  p1.A * p2.A + p1.B * p2.B + p1.C * p2.C = 0

theorem plane_equation_proof (p1 p2 : Point3D) (given_plane : Plane) :
  p1 = ⟨2, -1, 0⟩ →
  p2 = ⟨0, 2, -1⟩ →
  given_plane = { A := 1, B := -1, C := 2, D := -4, A_pos := by norm_num, gcd_one := by sorry } →
  ∃ (result_plane : Plane),
    p1.liesOn result_plane ∧
    p2.liesOn result_plane ∧
    result_plane.isPerpendicular given_plane ∧
    result_plane = { A := 5, B := 5, C := -1, D := -5, A_pos := by norm_num, gcd_one := by sorry } := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_proof_l276_27600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_sqrt_series_converges_terms_for_accuracy_l276_27664

/-- The series ∑_(n=1)^(∞) ((-1)^(n-1))/(√n) -/
noncomputable def alternating_sqrt_series (n : ℕ) : ℝ := (-1)^(n-1) / Real.sqrt (n : ℝ)

/-- The alternating sqrt series converges -/
theorem alternating_sqrt_series_converges :
  Summable alternating_sqrt_series :=
sorry

/-- The number of terms needed for 0.01 accuracy is at least 9999 -/
theorem terms_for_accuracy (n : ℕ) (hn : n ≥ 9999) :
  |alternating_sqrt_series (n + 1)| ≤ 0.01 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_sqrt_series_converges_terms_for_accuracy_l276_27664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_on_small_arc_l276_27689

/-- A type representing points on a circle -/
structure CirclePoint where
  angle : Real

/-- A proposition that states that for any two points, one of the arcs connecting them is smaller than 120° -/
def SmallArcProperty (points : Finset CirclePoint) : Prop :=
  ∀ p q, p ∈ points → q ∈ points → p ≠ q → 
    min (abs (p.angle - q.angle)) (2 * Real.pi - abs (p.angle - q.angle)) < 2 * Real.pi / 3

/-- The main theorem stating that if N points satisfy the SmallArcProperty, then all points lie on an arc of 120° -/
theorem points_on_small_arc {N : ℕ} (points : Finset CirclePoint) (h : points.card = N) 
    (small_arc : SmallArcProperty points) : 
  ∃ (start_angle : Real), ∀ p, p ∈ points → 
    0 ≤ (p.angle - start_angle) % (2 * Real.pi) ∧ (p.angle - start_angle) % (2 * Real.pi) ≤ 2 * Real.pi / 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_on_small_arc_l276_27689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_98516_scientific_notation_l276_27670

/-- Rounds a natural number to the nearest thousand -/
def roundToNearestThousand (n : ℕ) : ℕ :=
  let thousands := n / 1000
  let remainder := n % 1000
  if remainder ≥ 500 then (thousands + 1) * 1000 else thousands * 1000

/-- Converts a natural number to scientific notation (coefficient, exponent) -/
def toScientificNotation (n : ℕ) : ℚ × ℕ :=
  let digits := (String.length (toString n))
  let coefficient : ℚ := n / (10 ^ (digits - 1))
  (coefficient, digits - 1)

theorem round_98516_scientific_notation :
  toScientificNotation (roundToNearestThousand 98516) = (99/10, 4) := by
  sorry

#eval roundToNearestThousand 98516
#eval toScientificNotation (roundToNearestThousand 98516)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_98516_scientific_notation_l276_27670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_return_to_start_l276_27641

/-- Represents a point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the possible moves -/
inductive Move
  | north : Move
  | south : Move
  | east : Move
  | west : Move

/-- Applies a move to a point -/
def applyMove (p : Point) (m : Move) : Point :=
  match m with
  | Move.north => ⟨p.x, p.y + 2 * p.x⟩
  | Move.south => ⟨p.x, p.y - 2 * p.x⟩
  | Move.east => ⟨p.x + 2 * p.y, p.y⟩
  | Move.west => ⟨p.x - 2 * p.y, p.y⟩

/-- Represents a sequence of moves -/
def MovePath := List Move

/-- Applies a sequence of moves to a point -/
def applyPath (p : Point) (path : MovePath) : Point :=
  path.foldl applyMove p

/-- The starting point -/
noncomputable def startPoint : Point := ⟨1, Real.sqrt 2⟩

/-- Theorem stating that it's impossible to return to the starting point -/
theorem no_return_to_start (path : MovePath) : applyPath startPoint path ≠ startPoint := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_return_to_start_l276_27641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_OP_OQ_l276_27667

open Real

-- Define the curves C1 and C2
noncomputable def C1 (α : ℝ) : ℝ × ℝ := (1 + cos α, sin α)
noncomputable def C2 (β : ℝ) : ℝ × ℝ := (cos β, 1 + sin β)

-- Define the rays l1 and l2
noncomputable def l1 (α : ℝ) : ℝ := α
noncomputable def l2 (α : ℝ) : ℝ := α - π/6

-- Define the intersection points P and Q
noncomputable def P (α : ℝ) : ℝ × ℝ := 
  (2 * cos α * cos α, 2 * cos α * sin α)

noncomputable def Q (α : ℝ) : ℝ × ℝ := 
  (2 * sin (α - π/6) * cos (α - π/6), 
   2 * sin (α - π/6) * sin (α - π/6))

-- State the theorem
theorem max_product_OP_OQ : 
  ∃ (max : ℝ), 
    (∀ α, π/6 < α → α < π/2 → 
      sqrt ((P α).1^2 + (P α).2^2) * 
      sqrt ((Q α).1^2 + (Q α).2^2) ≤ max) ∧
    (∃ α, π/6 < α ∧ α < π/2 ∧ 
      sqrt ((P α).1^2 + (P α).2^2) * 
      sqrt ((Q α).1^2 + (Q α).2^2) = max) ∧
    max = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_OP_OQ_l276_27667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l276_27674

noncomputable def f (x : ℝ) := 3 * Real.tan (2 * x + Real.pi / 4)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | ∀ k : ℤ, x ≠ k / 2 * Real.pi + Real.pi / 8} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l276_27674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l276_27620

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x - a^(-x)

-- State the theorem
theorem function_properties (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a 1 = 3/2) :
  (a = 2) ∧
  (∀ t : ℝ, f a (2*t) + f a (t-1) < 0 ↔ t < 1/3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l276_27620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l276_27611

/-- The speed of a train given its length and time to cross a stationary point -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  (length / 1000) / (time / 3600)

/-- Theorem stating that a 180-meter train crossing a point in 6 seconds has a speed of 108 km/h -/
theorem train_speed_calculation :
  train_speed 180 6 = 108 := by
  -- Unfold the definition of train_speed
  unfold train_speed
  -- Simplify the expression
  simp [div_div_eq_mul_div]
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l276_27611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_problem_l276_27601

/-- Compound interest calculation --/
noncomputable def compound_interest_rate (principal : ℝ) (time : ℝ) (interest : ℝ) : ℝ :=
  ((principal + interest) / principal) ^ (1 / time) - 1

/-- Problem statement --/
theorem interest_rate_problem (principal time interest : ℝ) 
  (h_principal : principal = 4000)
  (h_time : time = 2.3333)
  (h_interest : interest = 1554.5) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧ 
  |compound_interest_rate principal time interest - 0.15| < ε := by
  sorry

#eval "Theorem defined successfully."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_problem_l276_27601
