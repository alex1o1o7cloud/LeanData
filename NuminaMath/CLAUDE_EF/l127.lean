import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dodecahedral_die_expected_value_l127_12724

/-- A fair dodecahedral die with faces numbered from 1 to 12 -/
def DodecahedralDie : Finset ℕ := Finset.range 12

/-- The probability of rolling any number on a fair dodecahedral die -/
noncomputable def probability : ℝ := 1 / 12

/-- The expected value of a roll of a fair dodecahedral die -/
noncomputable def expected_value : ℝ := (DodecahedralDie.sum (fun i => (i + 1) * probability))

/-- Theorem: The expected value of a roll of a fair dodecahedral die is 13/2 -/
theorem dodecahedral_die_expected_value : expected_value = 13 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dodecahedral_die_expected_value_l127_12724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_poly_equality_from_large_point_equality_l127_12779

/-- Represents a polynomial with integer coefficients -/
def MyPolynomial := List Int

/-- Returns the maximum absolute value of the coefficients in a polynomial -/
def maxAbsCoeff (p : MyPolynomial) : Int :=
  p.foldl (fun m a => max m (abs a)) 0

/-- Evaluates a polynomial at a given point -/
def evalPoly (p : MyPolynomial) (x : Int) : Int :=
  p.foldl (fun sum a => sum * x + a) 0

theorem poly_equality_from_large_point_equality 
  (f g : MyPolynomial) 
  (t : Int) 
  (h1 : evalPoly f t = evalPoly g t) 
  (h2 : t > 2 * max (maxAbsCoeff f) (maxAbsCoeff g)) :
  ∀ x, evalPoly f x = evalPoly g x := by
  sorry

#check poly_equality_from_large_point_equality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_poly_equality_from_large_point_equality_l127_12779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_saline_solution_water_calculation_l127_12713

/-- Given a saline solution preparation scenario, calculate the amount of water needed. -/
theorem saline_solution_water_calculation 
  (initial_salt : ℝ) 
  (initial_water : ℝ) 
  (total_desired : ℝ) 
  (h1 : initial_salt = 0.05)
  (h2 : initial_water = 0.03)
  (h3 : total_desired = 0.6) :
  (initial_water / (initial_salt + initial_water)) * total_desired = 0.225 := by
  -- Replace all occurrences of the variables with their actual values
  rw [h1, h2, h3]
  -- Simplify the expression
  norm_num
  -- The proof is complete
  done

#check saline_solution_water_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_saline_solution_water_calculation_l127_12713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_negative_x_l127_12716

noncomputable def f (x : ℝ) : ℝ := 4*x + 1/x + 3

theorem max_value_negative_x (f : ℝ → ℝ) :
  (∀ x : ℝ, f (-x) = -f x) →  -- f is odd
  (∀ x > 0, f x = 4*x + 1/x + 3) →  -- definition of f for x > 0
  (∃ x < 0, ∀ y < 0, f y ≤ f x ∧ f x = -7) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_negative_x_l127_12716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_curves_l127_12718

noncomputable def curve1 (x : ℝ) : ℝ := 2 * Real.exp x
noncomputable def curve2 (x : ℝ) : ℝ := Real.log (x / 2)

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem min_distance_between_curves :
  ∃ (p q : ℝ × ℝ),
    (p.2 = curve1 p.1) ∧
    (q.2 = curve2 q.1) ∧
    (∀ (p' q' : ℝ × ℝ),
      p'.2 = curve1 p'.1 →
      q'.2 = curve2 q'.1 →
      distance p q ≤ distance p' q') ∧
    distance p q = Real.sqrt 2 * (1 + Real.log 2) := by
  sorry

#check min_distance_between_curves

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_curves_l127_12718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_logarithm_inequality_l127_12723

-- Define an acute triangle
def AcuteTriangle (A B C : ℝ) : Prop :=
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi ∧ A < Real.pi/2 ∧ B < Real.pi/2 ∧ C < Real.pi/2

-- State the theorem
theorem sine_logarithm_inequality (A B C : ℝ) (h : AcuteTriangle A B C) :
  Real.log (Real.sin B) / Real.log (Real.sin A) +
  Real.log (Real.sin C) / Real.log (Real.sin B) +
  Real.log (Real.sin A) / Real.log (Real.sin C) ≥ 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_logarithm_inequality_l127_12723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_equation_solution_l127_12730

noncomputable def h (x : ℝ) : ℝ := ((2 * x^2 + 3 * x + 1) / 5) ^ (1/3)

theorem h_equation_solution (x : ℝ) : 
  h (3 * x) = 3 * h x ↔ x = -1 + Real.sqrt 10 / 3 ∨ x = -1 - Real.sqrt 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_equation_solution_l127_12730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_A_squared_minus_B_squared_l127_12770

theorem min_value_A_squared_minus_B_squared
  (x y z : ℝ)
  (hx : x ≥ 0)
  (hy : y ≥ 0)
  (hz : z ≥ 0) :
  let A := Real.sqrt (x + 4) + Real.sqrt (y + 7) + Real.sqrt (z + 13)
  let B := Real.sqrt (x + 3) + Real.sqrt (y + 3) + Real.sqrt (z + 3)
  A^2 - B^2 ≥ 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_A_squared_minus_B_squared_l127_12770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l127_12789

/-- Definition of an ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  e : ℝ
  focus_to_vertex : ℝ
  h1 : a > b
  h2 : b > 0
  h3 : e = 1/2
  h4 : focus_to_vertex = 1

/-- The equation of the ellipse -/
def ellipse_equation (E : Ellipse) (x y : ℝ) : Prop :=
  x^2 / E.a^2 + y^2 / E.b^2 = 1

/-- The slope product for a point on the ellipse -/
noncomputable def slope_product (E : Ellipse) (x y : ℝ) : ℝ :=
  (y / (x + E.a)) * (y / (x - E.a))

/-- Main theorem stating the properties of the ellipse -/
theorem ellipse_properties (E : Ellipse) :
  (∀ x y, ellipse_equation E x y ↔ x^2/4 + y^2/3 = 1) ∧
  (∀ x y, ellipse_equation E x y → x ≠ E.a → x ≠ -E.a → slope_product E x y = -3/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l127_12789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_at_10_l127_12768

/-- An arithmetic sequence with its properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  first_negative : a 1 < 0
  positive_difference : d > 0

/-- Sum of first n terms of an arithmetic sequence -/
noncomputable def S (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  (n : ℝ) * (seq.a 1 + seq.a n) / 2

/-- Theorem: For the given arithmetic sequence, S_n is minimized when n = 10 -/
theorem min_sum_at_10 (seq : ArithmeticSequence) 
  (h : S seq 20 / seq.a 10 < 0) :
  ∀ n : ℕ, n ≠ 0 → S seq 10 ≤ S seq n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_at_10_l127_12768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_distance_l127_12788

/-- The first curve: y = x^2 - 3 -/
def curve1 (x y : ℝ) : Prop := y = x^2 - 3

/-- The second curve: x + y = 7 -/
def curve2 (x y : ℝ) : Prop := x + y = 7

/-- An intersection point of the two curves -/
def is_intersection_point (p : ℝ × ℝ) : Prop :=
  curve1 p.1 p.2 ∧ curve2 p.1 p.2

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The main theorem: The distance between intersection points is √490 -/
theorem intersection_points_distance : 
  ∃ (p q : ℝ × ℝ), is_intersection_point p ∧ is_intersection_point q ∧ p ≠ q ∧ distance p q = Real.sqrt 490 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_distance_l127_12788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statements_l127_12758

noncomputable section

open Real

theorem problem_statements :
  (∀ (α β : ℝ), 0 < α ∧ α < π/2 ∧ 0 < β ∧ β < π/2 → (cos α : ℝ) > sin β → α + β < π/2) ∧
  (∀ (f : ℝ → ℝ) (θ : ℝ),
    (∀ x, x ∈ Set.Icc (-1 : ℝ) 1 → f x = f (-x)) ∧
    (∀ x y, x ∈ Set.Icc (-1 : ℝ) 0 ∧ y ∈ Set.Icc (-1 : ℝ) 0 ∧ x < y → f x < f y) ∧
    θ ∈ Set.Ioo (0 : ℝ) (π/4) →
    f (sin θ) > f (cos θ)) ∧
  (∃ (c : ℝ × ℝ), c = (π/6, 0) ∧
    ∀ (x y : ℝ), y = 4 * sin (2*x - π/3) ↔
      y - c.2 = -(((x - c.1) + c.1) - c.1)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statements_l127_12758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_distance_correct_l127_12704

/-- The distance from the top of the hill where two runners meet -/
noncomputable def meeting_distance (total_distance : ℝ) (head_start : ℝ) 
  (alex_up_speed alex_down_speed betty_up_speed betty_down_speed : ℝ) : ℝ :=
  let half_distance := total_distance / 2
  let head_start_hours := head_start / 60
  let alex_up_time := half_distance / alex_up_speed
  let betty_up_time := half_distance / betty_up_speed
  let meeting_time := (half_distance + alex_up_speed * head_start_hours) / (alex_down_speed + betty_up_speed)
  let betty_distance := betty_up_speed * (meeting_time - head_start_hours)
  half_distance - betty_distance

theorem meeting_distance_correct : 
  meeting_distance 16 8 14 18 15 21 = 151 / 77 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_distance_correct_l127_12704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l127_12798

/-- Given two vectors a and b in ℝ², where a = (2, 5) and b = (x, 4),
    if a is parallel to b, then x = 8/5. -/
theorem parallel_vectors_lambda (x : ℝ) :
  let a : Fin 2 → ℝ := ![2, 5]
  let b : Fin 2 → ℝ := ![x, 4]
  (∃ (k : ℝ), a = k • b) →
  x = 8/5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l127_12798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_from_return_times_l127_12721

/-- Represents an ellipse with a given eccentricity -/
structure Ellipse where
  eccentricity : ℝ
  h_ecc_pos : 0 < eccentricity
  h_ecc_lt_one : eccentricity < 1

/-- Represents a particle moving in the ellipse -/
structure Particle (e : Ellipse) where
  velocity : ℝ
  h_vel_pos : 0 < velocity

/-- The time taken for a particle to return to the left focus -/
noncomputable def returnTime (e : Ellipse) (p : Particle e) (path : ℝ) : ℝ :=
  path / p.velocity

/-- Theorem stating the relationship between return times and eccentricity -/
theorem ellipse_eccentricity_from_return_times (e : Ellipse) (p : Particle e) :
  (∃ (max_path min_path : ℝ),
    0 < min_path ∧ min_path < max_path ∧
    returnTime e p max_path = 7 * returnTime e p min_path) →
  e.eccentricity = 5/7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_from_return_times_l127_12721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_f_l127_12739

open Real

noncomputable def f (x : ℝ) := sin (2 * x + 7 * π / 6)

theorem axis_of_symmetry_f :
  (0 ≤ 7 * π / 6 ∧ 7 * π / 6 ≤ 2 * π) →
  (∀ x, sin (2 * (x - π / 3) + 7 * π / 6) = cos (2 * x)) →
  (∀ x, f x = f (π / 3 - x)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_f_l127_12739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_closed_form_l127_12720

def mySequence (n : ℕ) : ℕ :=
  match n with
  | 0 => 2
  | 1 => 3
  | n + 2 => 3 * mySequence (n + 1) - 2 * mySequence n

theorem mySequence_closed_form (n : ℕ) :
  mySequence n = 2^n + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_closed_form_l127_12720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l127_12795

noncomputable def sequence_a (n : ℕ) : ℝ := 2 * n - 1

noncomputable def sum_S (n : ℕ) : ℝ := n ^ 2

noncomputable def sequence_b (n : ℕ) : ℝ := (-1) ^ (n - 1) * (sequence_a (n + 1) / (sum_S n + n))

noncomputable def sum_T (n : ℕ) : ℝ := (n + 1 + (-1) ^ (n + 1)) / (n + 1)

theorem sequence_properties (n : ℕ) :
  (∀ k, sequence_a 1 + (sum_S k).sqrt = k + 1) →
  (∀ k, sum_S k = k ^ 2) ∧
  (∀ k, sequence_a k = 2 * k - 1) ∧
  (∀ k, sum_T k = (k + 1 + (-1) ^ (k + 1)) / (k + 1)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l127_12795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_roots_l127_12737

theorem omega_roots (ω : ℂ) (h1 : ω^8 = 1) (h2 : ω ≠ 1) : 
  ∃ α β : ℂ, α = ω + ω^3 + ω^5 ∧ 
             β = ω^2 + ω^4 + ω^6 + ω^7 ∧
             α^2 + α + 3 = 0 ∧ 
             β^2 + β + 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_roots_l127_12737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_visible_yellow_bus_length_l127_12728

/-- Given the lengths of various vehicles and obstruction by trees, calculate the visible length of a yellow bus. -/
theorem visible_yellow_bus_length 
  (red_bus_length : ℝ)
  (orange_car_length : ℝ)
  (yellow_bus_length : ℝ)
  (green_truck_length : ℝ)
  (tree_coverage_percent : ℝ)
  (h1 : red_bus_length = 4 * orange_car_length)
  (h2 : yellow_bus_length = 3.5 * orange_car_length)
  (h3 : green_truck_length = 2 * orange_car_length)
  (h4 : tree_coverage_percent = 25)
  (h5 : red_bus_length = 48) :
  (1 - tree_coverage_percent / 100) * yellow_bus_length = 31.5 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_visible_yellow_bus_length_l127_12728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chalkboard_ratio_l127_12722

/-- A rectangular chalkboard with given width and area -/
structure Chalkboard where
  width : ℝ
  area : ℝ

/-- The ratio of length to width for a rectangular chalkboard -/
noncomputable def lengthToWidthRatio (c : Chalkboard) : ℝ :=
  c.area / (c.width * c.width)

/-- Theorem stating that a chalkboard with width 3 and area 18 has a length to width ratio of 2 -/
theorem chalkboard_ratio :
  ∀ (c : Chalkboard), c.width = 3 → c.area = 18 → lengthToWidthRatio c = 2 := by
  intro c hw ha
  unfold lengthToWidthRatio
  rw [hw, ha]
  norm_num

#check chalkboard_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chalkboard_ratio_l127_12722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_cos_negative_implies_quadrant_3_or_4_l127_12769

def is_in_quadrant_3_or_4 (α : Real) : Prop :=
  (Real.sin α < 0) ∧ (Real.cos α ≠ 0)

theorem tan_cos_negative_implies_quadrant_3_or_4 (α : Real) 
  (h : Real.tan α * Real.cos α < 0) : is_in_quadrant_3_or_4 α := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_cos_negative_implies_quadrant_3_or_4_l127_12769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l127_12743

noncomputable section

-- Define the unit square
def unit_square : Set (ℝ × ℝ) := {p | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}

-- Define the inscribed circle
def inscribed_circle : Set (ℝ × ℝ) := {p | (p.1 - 1/2)^2 + (p.2 - 1/2)^2 ≤ 1/4}

-- Define points P, L, U, M
def P : ℝ × ℝ := (1, 1)
def L : ℝ × ℝ := (0, 1)
def U : ℝ × ℝ := (0, 0)
def M : ℝ × ℝ := (1, 0)

-- Define the property of I and E being on the circle
def on_circle (p : ℝ × ℝ) : Prop := p ∈ inscribed_circle

-- Define the collinearity of U, I, and E
def collinear (p q r : ℝ × ℝ) : Prop :=
  (q.2 - p.2) * (r.1 - q.1) = (r.2 - q.2) * (q.1 - p.1)

-- Define the area of a triangle given three points
def triangle_area (p q r : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((q.1 - p.1) * (r.2 - p.2) - (r.1 - p.1) * (q.2 - p.2))

-- Theorem statement
theorem max_triangle_area :
  ∀ I E : ℝ × ℝ,
  on_circle I → on_circle E → collinear U I E →
  triangle_area P I E ≤ 1/4 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l127_12743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l127_12749

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3*a - 1)*x + 4*a else a^x - a

theorem range_of_a (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x > f a y) →
  a ∈ Set.Icc (1/7 : ℝ) (1/3 : ℝ) :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l127_12749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cubes_for_closed_structure_l127_12708

/-- Represents a cube with one snap and five receptacle holes --/
structure Cube where
  snap : Fin 6
  receptacles : Finset (Fin 6)
  snap_receptacle_disjoint : snap ∉ receptacles
  receptacle_count : receptacles.card = 5

/-- Represents a configuration of connected cubes --/
structure CubeConfiguration where
  cubes : List Cube
  connections : List (Fin 6 × Fin 6 × Nat × Nat)
  valid_connections : ∀ c ∈ connections,
    let (f1, f2, i, j) := c
    f1 ≠ f2 ∧ i < cubes.length ∧ j < cubes.length ∧ i ≠ j ∧
    f1 ∈ (cubes.get ⟨i, by sorry⟩).receptacles ∧
    f2 = (cubes.get ⟨j, by sorry⟩).snap
  edge_connections : ∀ c ∈ connections,
    let (f1, f2, _, _) := c
    (f1.val % 2 = f2.val % 2)
  all_snaps_covered : ∀ i < cubes.length,
    ∃ c ∈ connections, c.2.2 = i ∧ c.2.1 = (cubes.get ⟨i, by sorry⟩).snap

/-- The main theorem stating that the minimum number of cubes required is 4 --/
theorem min_cubes_for_closed_structure :
  ∀ (config : CubeConfiguration),
    config.cubes.length ≥ 4 := by
  sorry

#check min_cubes_for_closed_structure

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cubes_for_closed_structure_l127_12708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_skew_median_distances_l127_12775

/-- RegularTetrahedron represents a regular tetrahedron with side length a -/
structure RegularTetrahedron where
  a : ℝ
  a_pos : 0 < a

/-- SkewMedianDistance calculates the distances between skew medians of two faces -/
noncomputable def SkewMedianDistance (t : RegularTetrahedron) : ℝ × ℝ :=
  (t.a * Real.sqrt (2/35), t.a / Real.sqrt 10)

/-- Theorem stating the distances between skew medians of two faces in a regular tetrahedron -/
theorem skew_median_distances (t : RegularTetrahedron) :
  SkewMedianDistance t = (t.a * Real.sqrt (2/35), t.a / Real.sqrt 10) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_skew_median_distances_l127_12775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parade_selection_l127_12776

/-- Represents the number of soldiers in a specific age group --/
structure SoldierGroup where
  count : ℕ
  deriving Repr

/-- Represents the unit of soldiers --/
structure SoldierUnit where
  total : ℕ
  young : SoldierGroup
  middle : SoldierGroup
  old : SoldierGroup
  parade_spots : ℕ
  deriving Repr

def SoldierUnit.valid (u : SoldierUnit) : Prop :=
  u.total = u.young.count + u.middle.count + u.old.count

theorem parade_selection (u : SoldierUnit) (h : u.valid) :
  u.old.count * u.parade_spots / u.total = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parade_selection_l127_12776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_multiple_4_5_or_6_l127_12732

def card_count : ℕ := 200

def is_multiple_of_4_5_or_6 (n : ℕ) : Bool :=
  n % 4 = 0 || n % 5 = 0 || n % 6 = 0

def count_multiples : ℕ := (List.range card_count).filter is_multiple_of_4_5_or_6 |>.length

theorem probability_of_multiple_4_5_or_6 :
  (count_multiples : ℚ) / card_count = 47 / 100 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_multiple_4_5_or_6_l127_12732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l127_12750

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (5 - x)

-- Theorem for part (1)
theorem part_one (k : ℝ) :
  (10 : ℝ) ^ (f k) = (10 : ℝ) ^ (f 2) * (10 : ℝ) ^ (f 3) → k = -1 :=
by sorry

-- Theorem for part (2)
theorem part_two (m : ℝ) :
  f (2 * m - 1) < f (m + 1) → 2 < m ∧ m < 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l127_12750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_characterization_l127_12714

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then Real.rpow 2 x - 4 else Real.rpow 2 (-x) - 4

-- State the theorem
theorem solution_set_characterization :
  (∀ x : ℝ, f x = f (-x)) →  -- f is even
  (∀ x : ℝ, x ≥ 0 → f x = Real.rpow 2 x - 4) →  -- f(x) = 2^x - 4 for x ≥ 0
  {x : ℝ | f (x - 2) > 0} = {x : ℝ | x < 0 ∨ x > 4} :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_characterization_l127_12714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_equality_l127_12757

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

noncomputable def fract (x : ℝ) : ℝ := x - Int.floor x

theorem solution_equality (x y z : ℝ) : 
  (floor x + fract y = z) ∧ 
  (floor y + fract z = x) ∧ 
  (floor z + fract x = y) → 
  x = y ∧ y = z :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_equality_l127_12757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_reduction_percentage_l127_12763

/-- Calculates the final price as a percentage of the original price after three consecutive reductions -/
theorem price_reduction_percentage (original_price : ℝ) (reduction1 reduction2 reduction3 : ℝ) :
  reduction1 = 0.09 ∧ reduction2 = 0.10 ∧ reduction3 = 0.15 →
  (original_price * (1 - reduction1) * (1 - reduction2) * (1 - reduction3) / original_price) * 100 = 69.615 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_reduction_percentage_l127_12763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_circle_area_ratio_for_radius_3_l127_12710

/-- The ratio of the area of a star figure to the area of its original circle --/
noncomputable def star_circle_area_ratio (r : ℝ) : ℝ :=
  (81 * Real.sqrt 3 - 36 * Real.pi) / (36 * Real.pi)

/-- Theorem: The ratio of the area of a star figure formed by rearranging 6 congruent arcs
    of a circle with radius 3 within a regular hexagon to the area of the original circle
    is (81√3 - 36π) / (36π) --/
theorem star_circle_area_ratio_for_radius_3 :
  star_circle_area_ratio 3 = (81 * Real.sqrt 3 - 36 * Real.pi) / (36 * Real.pi) := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval star_circle_area_ratio 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_circle_area_ratio_for_radius_3_l127_12710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_crossing_time_l127_12742

/-- The time taken for a train to cross a bridge -/
noncomputable def train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (bridge_length : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  total_distance / train_speed_ms

/-- Theorem: A train 140 m long running at 72 km/hr takes approximately 13.6 seconds to cross a 132 m long bridge -/
theorem train_bridge_crossing_time :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ |train_crossing_time 140 72 132 - 13.6| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_crossing_time_l127_12742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_property_P_implications_l127_12771

/-- Property P for a set of numbers -/
def has_property_P (M : Set ℝ) : Prop :=
  ∀ (i j : ℝ), i ∈ M → j ∈ M → (i + j ∈ M ∨ j - i ∈ M)

/-- The set M satisfies the given conditions -/
def valid_set (M : Set ℝ) (n : ℕ) : Prop :=
  ∃ (a : ℕ → ℝ), 
    (∀ i, i ∈ Finset.range n → a i ∈ M) ∧
    (∀ i, i ∈ Finset.range (n-1) → a i < a (i+1)) ∧
    (n ≥ 2) ∧
    (M.Nonempty) ∧
    (M = (Finset.range n).image (λ i => a i))

theorem property_P_implications {M : Set ℝ} {n : ℕ} (h : valid_set M n) (h_P : has_property_P M) :
  ∃ (a : ℕ → ℝ), 
    (∀ i, i ∈ Finset.range n → a i ∈ M) ∧
    (a 0 = 0) ∧
    (a (n-1) = (2 / n) * (Finset.range n).sum a) ∧
    (n = 5 → ∃ d, ∀ i, i ∈ Finset.range 5 → a i = ↑i * d) := by
  sorry

#check property_P_implications

end NUMINAMATH_CALUDE_ERRORFEEDBACK_property_P_implications_l127_12771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_distance_between_C1_and_C2_l127_12784

-- Define the ellipse C1
def C1 (x y : ℝ) : Prop := x^2 / 3 + y^2 = 1

-- Define the line C2
def C2 (x y : ℝ) : Prop := x + y = 4

-- Define the distance function between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem minimum_distance_between_C1_and_C2 :
  ∃ (x1 y1 x2 y2 : ℝ),
    C1 x1 y1 ∧ C2 x2 y2 ∧
    (∀ (x3 y3 x4 y4 : ℝ),
      C1 x3 y3 → C2 x4 y4 →
      distance x1 y1 x2 y2 ≤ distance x3 y3 x4 y4) ∧
    distance x1 y1 x2 y2 = Real.sqrt 2 ∧
    x1 = 3/2 ∧ y1 = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_distance_between_C1_and_C2_l127_12784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l127_12762

/-- The function f(x) with a constant m -/
noncomputable def f (x m : ℝ) : ℝ := (1/2) * (Real.cos x)^4 + Real.sqrt 3 * Real.sin x * Real.cos x - (1/2) * (Real.sin x)^4 + m

/-- The theorem stating properties of the function f -/
theorem f_properties :
  ∃ (m : ℝ),
    (∀ x, f x m ≤ 3/2) ∧ 
    (m = 1/2) ∧
    (∀ k : ℤ, f (k * Real.pi + Real.pi/6) m = 3/2) ∧
    (∀ k : ℤ, ∀ x ∈ Set.Icc (k * Real.pi - Real.pi/3) (k * Real.pi + Real.pi/6),
      ∀ y ∈ Set.Icc (k * Real.pi - Real.pi/3) (k * Real.pi + Real.pi/6),
      x ≤ y → f x m ≤ f y m) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l127_12762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_ratio_property_l127_12753

-- Define the golden ratio
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

-- Define A as the square of the golden ratio
noncomputable def A : ℝ := φ^2

-- Function to calculate the ceiling of a real number
noncomputable def ceiling (x : ℝ) : ℤ := Int.ceil x

-- Function to find the nearest square integer
def nearestSquare (x : ℤ) : ℤ := sorry

-- Theorem statement
theorem golden_ratio_property :
  ∀ (n : ℕ), n > 0 → 
    (abs ((ceiling (A^n : ℝ)) - nearestSquare (ceiling (A^n : ℝ)))) = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_ratio_property_l127_12753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_half_max_ab_tangent_three_point_inequality_l127_12772

-- Define the natural logarithm function
noncomputable def f (x : ℝ) : ℝ := Real.log x

-- Theorem 1: Tangent line with slope 1/2
theorem tangent_slope_half (m : ℝ) :
  (∃ x₀ : ℝ, x₀ > 0 ∧ f x₀ = 1/2 * x₀ + m) ↔ m = Real.log 2 - 1 := by sorry

-- Theorem 2: Maximum value of ab for tangent line y = ax + b
theorem max_ab_tangent :
  (∃ a b : ℝ, ∃ x₀ : ℝ, x₀ > 0 ∧ f x₀ = a * x₀ + b ∧ ∀ x : ℝ, x > 0 → f x ≤ a * x + b) →
  (∃ a b : ℝ, a * b = Real.exp (-2) ∧
    ∀ a' b' : ℝ, (∃ x₀ : ℝ, x₀ > 0 ∧ f x₀ = a' * x₀ + b' ∧ ∀ x : ℝ, x > 0 → f x ≤ a' * x + b') →
      a' * b' ≤ Real.exp (-2)) := by sorry

-- Theorem 3: Inequality for three points on the curve
theorem three_point_inequality (x₁ x₂ x₃ : ℝ) (h : 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < x₃) :
  (f x₂ - f x₁) / (x₂ - x₁) > (f x₃ - f x₂) / (x₃ - x₂) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_half_max_ab_tangent_three_point_inequality_l127_12772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_with_six_factors_l127_12777

def has_exactly_six_factors (n : ℕ) : Prop :=
  (Finset.filter (· ∣ n) (Finset.range n.succ)).card = 6

theorem smallest_integer_with_six_factors :
  ∃ (n : ℕ), n > 0 ∧ has_exactly_six_factors n ∧
  ∀ (m : ℕ), m > 0 → has_exactly_six_factors m → n ≤ m :=
by
  use 12
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_with_six_factors_l127_12777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_five_theta_l127_12773

theorem cos_five_theta (θ : ℝ) (h : Real.cos θ = 2/5) : Real.cos (5*θ) = 2762/3125 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_five_theta_l127_12773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_of_f_l127_12759

-- Define the function f(x) = ln(2x - x^2)
noncomputable def f (x : ℝ) : ℝ := Real.log (2*x - x^2)

-- State the theorem
theorem monotonic_decreasing_interval_of_f :
  ∃ (a b : ℝ), a = 1 ∧ b = 2 ∧
  (∀ x, x ∈ Set.Ioo a b → 0 < 2*x - x^2) ∧
  (∀ x y, x ∈ Set.Ioo a b → y ∈ Set.Ioo a b → x < y → f y < f x) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_of_f_l127_12759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_polar_points_l127_12791

noncomputable def polar_point := ℝ × ℝ

noncomputable def distance (p₁ p₂ : polar_point) : ℝ :=
  Real.sqrt ((p₁.1 * Real.cos p₁.2 - p₂.1 * Real.cos p₂.2)^2 + 
             (p₁.1 * Real.sin p₁.2 - p₂.1 * Real.sin p₂.2)^2)

theorem distance_between_polar_points 
  (A B : polar_point)
  (h₁ : A.1 = 5)
  (h₂ : B.1 = 7)
  (h₃ : A.2 - B.2 = π / 3) :
  distance A B = Real.sqrt 39 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_polar_points_l127_12791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_eight_rays_l127_12741

/-- Definition of a ray in ℝ² -/
def IsRay (S : Set (ℝ × ℝ)) : Prop :=
  ∃ (a b : ℝ) (c : ℝ), c > 0 ∧ 
    S = {(x, y) | ∃ (t : ℝ), t ≥ 0 ∧ x = a + c * t ∧ y = b + c * t}

/-- The trajectory of a point whose difference in distances to the two coordinate axes is 2 -/
theorem trajectory_eight_rays :
  ∃ (S : Set (ℝ × ℝ)), 
    S = {(x, y) | |x| - |y| = 2 ∨ |y| - |x| = 2} ∧
    (∃ (r₁ r₂ r₃ r₄ r₅ r₆ r₇ r₈ : Set (ℝ × ℝ)), 
      (∀ i ∈ ({r₁, r₂, r₃, r₄, r₅, r₆, r₇, r₈} : Set (Set (ℝ × ℝ))), IsRay i) ∧
      S = r₁ ∪ r₂ ∪ r₃ ∪ r₄ ∪ r₅ ∪ r₆ ∪ r₇ ∪ r₈) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_eight_rays_l127_12741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weekly_social_media_time_l127_12731

noncomputable def daily_phone_time : ℝ := 8
noncomputable def social_media_fraction : ℝ := 1 / 2
def days_in_week : ℕ := 7

theorem weekly_social_media_time :
  daily_phone_time * social_media_fraction * (days_in_week : ℝ) = 28 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weekly_social_media_time_l127_12731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_photo_gallery_total_l127_12787

/-- Given an initial gallery size and photo counts from a two-day trip, 
    calculate the total number of photos after the trip. -/
theorem photo_gallery_total 
  (initial_gallery : ℕ) 
  (first_day : ℕ) 
  (second_day : ℕ) 
  (h1 : initial_gallery = 800)
  (h2 : first_day = initial_gallery * 2 / 3)
  (h3 : second_day = first_day + 180) :
  initial_gallery + first_day + second_day = 2046 := by
  sorry

#eval Nat.add (Nat.add 800 533) 713

end NUMINAMATH_CALUDE_ERRORFEEDBACK_photo_gallery_total_l127_12787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l127_12712

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  positive : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : A + B + C = π

/-- The sine law for a triangle -/
axiom sine_law {t : Triangle} : t.a / Real.sin t.A = t.b / Real.sin t.B

theorem triangle_problem (t : Triangle) 
  (h1 : 4 * t.b * Real.sin t.A = Real.sqrt 7 * t.a)
  (h2 : ∃ (d : ℝ), d > 0 ∧ t.b = t.a + d ∧ t.c = t.b + d) :
  Real.sin t.B = Real.sqrt 7 / 4 ∧ 
  Real.cos t.A - Real.cos t.C = Real.sqrt 7 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l127_12712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l127_12774

theorem problem_statement (a b : ℝ) 
  (h1 : (a + b)^(-2003 : ℤ) = 1)
  (h2 : (-a + b)^2005 = 1) :
  a^2003 + b^2004 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l127_12774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_ratio_l127_12796

/-- The original function f(x) -/
noncomputable def f (x : ℝ) : ℝ := (3*x + 2) / (x - 4)

/-- The inverse function f⁻¹(x) -/
noncomputable def f_inv (a b c d : ℝ) (x : ℝ) : ℝ := (a*x + b) / (c*x + d)

/-- Theorem stating that if f_inv is the inverse of f, then a/c = -4 -/
theorem inverse_function_ratio (a b c d : ℝ) (h : ∀ x, f (f_inv a b c d x) = x) : a / c = -4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_ratio_l127_12796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rebus_unique_solution_l127_12727

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents the rebus equation -/
def RebusEquation (L E T M G U Y A R : Digit) : Prop :=
  250 * (L.val * 100 + E.val * 10 + T.val) + (M.val * 100 + G.val * 10 + U.val) =
  2005 * (Y.val * 1000 + E.val * 100 + A.val * 10 + R.val)

/-- All digits in the rebus are different -/
def AllDifferent (L E T M G U Y A R : Digit) : Prop :=
  L ≠ E ∧ L ≠ T ∧ L ≠ M ∧ L ≠ G ∧ L ≠ U ∧ L ≠ Y ∧ L ≠ A ∧ L ≠ R ∧
  E ≠ T ∧ E ≠ M ∧ E ≠ G ∧ E ≠ U ∧ E ≠ Y ∧ E ≠ A ∧ E ≠ R ∧
  T ≠ M ∧ T ≠ G ∧ T ≠ U ∧ T ≠ Y ∧ T ≠ A ∧ T ≠ R ∧
  M ≠ G ∧ M ≠ U ∧ M ≠ Y ∧ M ≠ A ∧ M ≠ R ∧
  G ≠ U ∧ G ≠ Y ∧ G ≠ A ∧ G ≠ R ∧
  U ≠ Y ∧ U ≠ A ∧ U ≠ R ∧
  Y ≠ A ∧ Y ≠ R ∧
  A ≠ R

theorem rebus_unique_solution :
  ∀ L E T M G U Y A R : Digit,
    RebusEquation L E T M G U Y A R →
    AllDifferent L E T M G U Y A R →
    L = ⟨9, by norm_num⟩ ∧ 
    E = ⟨2, by norm_num⟩ ∧ 
    T = ⟨6, by norm_num⟩ ∧ 
    M = ⟨1, by norm_num⟩ ∧ 
    G = ⟨1, by norm_num⟩ ∧ 
    U = ⟨5, by norm_num⟩ ∧ 
    Y = ⟨1, by norm_num⟩ ∧ 
    A = ⟨2, by norm_num⟩ ∧ 
    R = ⟨3, by norm_num⟩ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rebus_unique_solution_l127_12727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_reflection_distance_C_to_C_l127_12794

/-- The distance between a point and its reflection over the x-axis --/
theorem distance_to_reflection (x y : ℝ) : 
  let C : ℝ × ℝ := (x, y)
  let C' : ℝ × ℝ := (x, -y)
  Real.sqrt ((C'.1 - C.1)^2 + (C'.2 - C.2)^2) = 2 * abs y :=
by sorry

/-- The specific case for point C(-2, 3) --/
theorem distance_C_to_C'_reflection : 
  let C : ℝ × ℝ := (-2, 3)
  let C' : ℝ × ℝ := (-2, -3)
  Real.sqrt ((C'.1 - C.1)^2 + (C'.2 - C.2)^2) = 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_reflection_distance_C_to_C_l127_12794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l127_12782

theorem trigonometric_identities (α : ℝ) 
  (h_acute : 0 < α ∧ α < π / 2) 
  (h_cos : Real.cos (α + π / 6) = 3 / 5) : 
  Real.cos (α - π / 3) = 4 / 5 ∧ Real.cos (2 * α - π / 6) = 24 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l127_12782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_specific_line_l127_12783

/-- The slope angle of a line passing through two given points -/
noncomputable def slope_angle (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.arctan ((y₂ - y₁) / (x₂ - x₁))

/-- Theorem: The slope angle of a line passing through (-1, 3) and (1, 1) is 3π/4 -/
theorem slope_angle_specific_line :
  slope_angle (-1) 3 1 1 = 3 * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_specific_line_l127_12783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_charts_used_l127_12752

/-- Represents the types of charts used for analyzing relationships between categorical variables -/
inductive CategoricalChart
  | ContingencyTable
  | ThreeDimensionalBarChart
  | TwoDimensionalBarChart

/-- Function to check if a given chart is used for categorical variable analysis -/
def isUsedForCategoricalAnalysis (chart : CategoricalChart) : Bool :=
  match chart with
  | CategoricalChart.ContingencyTable => true
  | CategoricalChart.ThreeDimensionalBarChart => true
  | CategoricalChart.TwoDimensionalBarChart => true

/-- Theorem stating that all three chart types are used for categorical variable analysis -/
theorem all_charts_used : 
  (isUsedForCategoricalAnalysis CategoricalChart.ContingencyTable) ∧
  (isUsedForCategoricalAnalysis CategoricalChart.ThreeDimensionalBarChart) ∧
  (isUsedForCategoricalAnalysis CategoricalChart.TwoDimensionalBarChart) := by
  simp [isUsedForCategoricalAnalysis]

#check all_charts_used

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_charts_used_l127_12752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subsequence_of_length_eleven_l127_12705

theorem subsequence_of_length_eleven (π : Fin 101 → Fin 101) : 
  (∃ (s : Finset (Fin 101)), s.card = 11 ∧ 
    (∀ i j, i ∈ s → j ∈ s → i < j → π i < π j)) ∨ 
  (∃ (s : Finset (Fin 101)), s.card = 11 ∧ 
    (∀ i j, i ∈ s → j ∈ s → i < j → π i > π j)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subsequence_of_length_eleven_l127_12705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_exceeds_available_l127_12764

/-- Represents a rectangular field with a square obstacle -/
structure FieldWithObstacle where
  fieldArea : ℝ
  fieldWidth : ℝ
  obstacleSize : ℝ

/-- Calculates the total fencing required for a field with an obstacle -/
noncomputable def totalFencingRequired (f : FieldWithObstacle) : ℝ :=
  let fieldLength := f.fieldArea / f.fieldWidth
  let fieldFencing := fieldLength + 2 * f.fieldWidth
  let obstacleFencing := 4 * f.obstacleSize
  fieldFencing + obstacleFencing

/-- Theorem stating that the fencing required exceeds the available amount -/
theorem fencing_exceeds_available 
  (f : FieldWithObstacle) 
  (h1 : f.fieldArea = 1000)
  (h2 : f.fieldWidth = 25)
  (h3 : f.obstacleSize = 15) : 
  totalFencingRequired f > 100 := by
  sorry

#check fencing_exceeds_available

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_exceeds_available_l127_12764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_simplification_l127_12729

/-- Simplification of a trigonometric expression -/
theorem trig_simplification (α : Real) : 
  (Real.sin α)^2 * (1 + (Real.sin α)⁻¹ + (Real.cos α / Real.sin α)) * 
  (1 - (Real.sin α)⁻¹ + (Real.cos α / Real.sin α)) = Real.sin (2 * α) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_simplification_l127_12729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_ratio_l127_12711

/-- Represents the sum of the first n terms of a geometric sequence -/
noncomputable def S (n : ℕ) : ℝ := sorry

/-- The ratio of S_10 to S_5 is 1:2 -/
axiom ratio_10_5 : S 10 / S 5 = 1 / 2

/-- Theorem: If S_10 : S_5 = 1 : 2, then S_15 : S_5 = 3 : 4 -/
theorem geometric_sum_ratio :
  S 15 / S 5 = 3 / 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_ratio_l127_12711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_coordinates_l127_12717

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a segment in 2D space
structure Segment2D where
  start : Point2D
  stop : Point2D

-- Function to calculate the length of a segment
noncomputable def segmentLength (s : Segment2D) : ℝ :=
  Real.sqrt ((s.stop.x - s.start.x)^2 + (s.stop.y - s.start.y)^2)

-- Function to check if a segment is parallel to x-axis
def isParallelToXAxis (s : Segment2D) : Prop :=
  s.start.y = s.stop.y

-- Theorem statement
theorem segment_coordinates (b : Point2D) (ab : Segment2D) 
  (h1 : b = ab.stop)
  (h2 : b.x = 2 ∧ b.y = 4)
  (h3 : isParallelToXAxis ab)
  (h4 : segmentLength ab = 3) :
  (ab.start = Point2D.mk 5 4) ∨ (ab.start = Point2D.mk (-1) 4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_coordinates_l127_12717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_character_initial_ratio_l127_12748

theorem character_initial_ratio (total : ℕ) (init_a init_c init_d init_e : ℕ) : 
  total = 60 → 
  init_a = total / 2 → 
  init_c = init_a / 2 → 
  init_d + init_e = total - init_a - init_c → 
  init_d = init_e → 
  (init_d : ℚ) / (init_e : ℚ) = 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_character_initial_ratio_l127_12748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_done_equals_64_l127_12785

-- Define the force function
def F (x : ℝ) : ℝ := 3 * x^2

-- State the theorem
theorem work_done_equals_64 :
  ∫ x in (0 : ℝ)..(4 : ℝ), F x = 64 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_done_equals_64_l127_12785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truck_distance_theorem_l127_12706

/-- The distance in yards traveled by a truck in 5 minutes -/
noncomputable def distance_in_yards (b t : ℝ) : ℝ :=
  let feet_per_t_seconds := b / 4
  let seconds_in_5_minutes := 5 * 60
  let feet_in_5_minutes := (feet_per_t_seconds / t) * seconds_in_5_minutes
  let yards_in_5_minutes := feet_in_5_minutes / 3
  yards_in_5_minutes

/-- Theorem stating that the distance traveled by the truck in 5 minutes is 25b/t yards -/
theorem truck_distance_theorem (b t : ℝ) (h : t ≠ 0) : 
  distance_in_yards b t = 25 * b / t := by
  sorry

#check truck_distance_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_truck_distance_theorem_l127_12706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l127_12726

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Check if a point lies on a circle -/
def pointOnCircle (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

/-- Check if two circles are tangent at a point -/
def circleTangent (c1 c2 : Circle) (p : ℝ × ℝ) : Prop :=
  pointOnCircle c1 p ∧ pointOnCircle c2 p ∧
  (c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2 = (c1.radius + c2.radius)^2

/-- The given circle equation -/
noncomputable def givenCircle : Circle :=
  { center := (-1, 1), radius := Real.sqrt 2 }

/-- Theorem: The equation of the circle passing through (0,1) and (0,0),
    and tangent to x^2 + y^2 + 2x - 2y = 0 at the origin -/
theorem circle_equation : ∃ (c : Circle),
  pointOnCircle c (0, 1) ∧
  pointOnCircle c (0, 0) ∧
  circleTangent c givenCircle (0, 0) ∧
  c.center = (-1/2, 1/2) ∧
  c.radius^2 = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l127_12726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_redo_profit_l127_12793

/-- Calculates the profit for Redo's manufacturing company --/
noncomputable def calculate_profit (
  initial_outlay : ℝ)
  (material_cost_first_300 : ℝ)
  (material_cost_beyond_300 : ℝ)
  (exchange_rate : ℝ)
  (import_tax_rate : ℝ)
  (sales_price_first_400 : ℝ)
  (sales_price_beyond_400 : ℝ)
  (export_tax_rate : ℝ)
  (total_sets : ℕ) : ℝ :=
  let material_costs := 
    (min (total_sets : ℝ) 300) * material_cost_first_300 * exchange_rate +
    (max ((total_sets : ℝ) - 300) 0) * material_cost_beyond_300 * exchange_rate
  let import_tax := material_costs * import_tax_rate
  let manufacturing_costs := initial_outlay + material_costs + import_tax
  let sales_revenue := 
    (min (total_sets : ℝ) 400) * sales_price_first_400 +
    (max ((total_sets : ℝ) - 400) 0) * sales_price_beyond_400
  let export_tax := 
    max ((total_sets : ℝ) - 500) 0 * sales_price_beyond_400 * export_tax_rate
  sales_revenue - export_tax - manufacturing_costs

/-- Theorem: The profit for Redo's manufacturing company is $10,990 --/
theorem redo_profit :
  calculate_profit 10000 20 15 1.1 0.1 50 45 0.05 800 = 10990 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_redo_profit_l127_12793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_three_digit_number_l127_12740

def is_valid (n : ℕ) : Prop :=
  n < 1000 ∧ n ≥ 100 ∧ 
  ∃ k : ℕ, n = 6 * k + 2 ∧
  ∃ m : ℕ, n = 7 * m + 4

theorem greatest_three_digit_number : 
  is_valid 998 ∧ ∀ n : ℕ, is_valid n → n ≤ 998 := by
  constructor
  · -- Prove that 998 is valid
    sorry
  · -- Prove that 998 is the greatest
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_three_digit_number_l127_12740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_values_monotone_increasing_condition_l127_12735

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + 2

-- Define the domain
def domain : Set ℝ := Set.Icc (-5) 5

-- Theorem for part (1)
theorem min_max_values :
  (∃ x₀ ∈ domain, f (-1) x₀ = 1 ∧ ∀ y ∈ domain, f (-1) y ≥ f (-1) x₀) ∧
  (∃ x₁ ∈ domain, f (-1) x₁ = 37 ∧ ∀ y ∈ domain, f (-1) y ≤ f (-1) x₁) :=
by sorry

-- Theorem for part (2)
theorem monotone_increasing_condition (a : ℝ) :
  (∀ x y, x ∈ domain → y ∈ domain → x < y → f a x < f a y) ↔ a ≥ 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_values_monotone_increasing_condition_l127_12735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_invertible_functions_product_l127_12786

-- Define the domain of f₃
def domain_f₃ : Set ℤ := {-5, -4, -3, -2, -1, 0, 1, 2}

-- Define the functions
noncomputable def f₂ : ℝ → ℝ := λ x => x^2 - 2*x

noncomputable def f₃ : domain_f₃ → ℝ := sorry

noncomputable def f₄ : ℝ → ℝ := λ x => -Real.arctan x

noncomputable def f₅ : ℝ → ℝ := λ x => 4/x

-- Define invertibility
def is_invertible {α β : Type*} (f : α → β) : Prop :=
  ∃ g : β → α, (∀ x, g (f x) = x) ∧ (∀ y, f (g y) = y)

theorem invertible_functions_product (h₂ : ¬is_invertible f₂)
    (h₃ : is_invertible f₃) (h₄ : is_invertible f₄) (h₅ : is_invertible f₅) :
  3 * 4 * 5 = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_invertible_functions_product_l127_12786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_of_polynomial_power_l127_12765

theorem degree_of_polynomial_power : 
  Polynomial.degree ((Polynomial.X : Polynomial ℝ)^3 - (5 : Polynomial ℝ) * Polynomial.X + (7 : Polynomial ℝ))^8 = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_of_polynomial_power_l127_12765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l127_12746

/-- Given a hyperbola C with equation x²/a² - y²/b² = 1 (a > 0, b > 0),
    a line passing through its right focus with an inclination angle of π/2
    intersects C at points A and B. O is the origin.
    If ∠AOB = ∠OAB, then the eccentricity of C is (√3 + √39)/6. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (C : Set (ℝ × ℝ))
  (hC : C = {p : ℝ × ℝ | p.1^2 / a^2 - p.2^2 / b^2 = 1})
  (A B : ℝ × ℝ)
  (hAB : A ∈ C ∧ B ∈ C)
  (hline : ∃ (m : ℝ), ∀ (p : ℝ × ℝ), p ∈ C → p.2 = m * (p.1 - Real.sqrt (a^2 + b^2)))
  (hangle : (A.1 * B.2 - A.2 * B.1) / (A.1 * B.1 + A.2 * B.2) =
            (A.1 * B.2 - A.2 * B.1) / (A.1^2 + A.2^2)) :
  Real.sqrt (1 + b^2 / a^2) = (Real.sqrt 3 + Real.sqrt 39) / 6 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l127_12746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l127_12719

/-- The distance between two parallel lines in ℝ² --/
noncomputable def distance_between_parallel_lines (a₁ a₂ b₁ b₂ d₁ d₂ : ℝ) : ℝ :=
  let v₁ := b₁ - a₁
  let v₂ := b₂ - a₂
  let d_norm_sq := d₁^2 + d₂^2
  let proj := (v₁ * d₁ + v₂ * d₂) / d_norm_sq
  let perp₁ := v₁ - proj * d₁
  let perp₂ := v₂ - proj * d₂
  Real.sqrt (perp₁^2 + perp₂^2)

/-- Theorem stating the distance between the given parallel lines --/
theorem distance_between_given_lines :
  distance_between_parallel_lines 3 (-2) 5 (-8) 2 (-14) = 26 * Real.sqrt 2 / 25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l127_12719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_digits_10000_l127_12756

/-- The function that calculates the number of digits in a positive integer -/
def num_digits (n : ℕ) : ℕ :=
  if n < 10 then 1
  else if n < 100 then 2
  else if n < 1000 then 3
  else if n < 10000 then 4
  else 5

/-- The function that calculates the sum of digits used to number pages from 1 to n -/
def sum_digits (n : ℕ) : ℕ :=
  (List.range n).map (fun i => num_digits (i + 1)) |>.sum

/-- Theorem stating that the sum of digits used to number pages from 1 to 10,000 is 38,894 -/
theorem sum_digits_10000 : sum_digits 10000 = 38894 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_digits_10000_l127_12756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mark_is_28_l127_12766

/-- Represents a person's age -/
structure Age where
  value : ℕ

/-- Mark's current age -/
def mark_age : Age := ⟨28⟩

/-- Aaron's current age -/
def aaron_age : Age := ⟨11⟩

/-- Theorem stating Mark's current age is 28 -/
theorem mark_is_28 :
  (mark_age.value - 3 = 3 * (aaron_age.value - 3) + 1) →
  (mark_age.value + 4 = 2 * (aaron_age.value + 4) + 2) →
  mark_age.value = 28 := by
  intro h1 h2
  -- The proof goes here
  sorry

#eval mark_age.value
#eval aaron_age.value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mark_is_28_l127_12766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_points_l127_12709

/-- The distance between two points in 3D space -/
noncomputable def distance3D (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  let (x1, y1, z1) := p1
  let (x2, y2, z2) := p2
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)

/-- The theorem stating that the distance between (1, -3, 2) and (4, 4, 6) is √74 -/
theorem distance_between_specific_points :
  distance3D (1, -3, 2) (4, 4, 6) = Real.sqrt 74 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_points_l127_12709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_equality_l127_12767

theorem cos_sin_equality : Real.cos 0 * Real.cos (2 * π / 180) - Real.sin (4 * π / 180) * Real.sin (2 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_equality_l127_12767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_power_rotation_120_l127_12755

noncomputable def rotation_matrix (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![Real.cos θ, -Real.sin θ],
    ![Real.sin θ,  Real.cos θ]]

theorem smallest_power_rotation_120 :
  (∃ (n : ℕ), n > 0 ∧ (rotation_matrix (2 * Real.pi / 3)) ^ n = 1) ∧
  (∀ (m : ℕ), m > 0 ∧ (rotation_matrix (2 * Real.pi / 3)) ^ m = 1 → m ≥ 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_power_rotation_120_l127_12755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_value_theorem_minimum_value_is_eight_minimum_value_l127_12702

theorem minimum_value_theorem (x y : ℝ) (h : 2 * x - y = 4) :
  ∀ a b : ℝ, 2 * a - b = 4 → (4 : ℝ)^x + (1/2 : ℝ)^y ≤ (4 : ℝ)^a + (1/2 : ℝ)^b :=
by sorry

theorem minimum_value_is_eight (x y : ℝ) (h : 2 * x - y = 4) :
  ∃ x₀ y₀ : ℝ, 2 * x₀ - y₀ = 4 ∧ (4 : ℝ)^x₀ + (1/2 : ℝ)^y₀ = 8 :=
by sorry

theorem minimum_value (x y : ℝ) (h : 2 * x - y = 4) :
  (4 : ℝ)^x + (1/2 : ℝ)^y ≥ 8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_value_theorem_minimum_value_is_eight_minimum_value_l127_12702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_bound_l127_12799

noncomputable def a : ℕ → ℝ
| 0 => 2
| n + 1 => a n / 2 + 1 / a n

theorem sequence_bound (n : ℕ) : a n < Real.sqrt 2 + 1 / (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_bound_l127_12799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_X_equals_Y_l127_12707

noncomputable def prob_X_equals_Y (X Y : ℝ) : ℝ :=
  if X = Y then 1 else 0

theorem probability_X_equals_Y :
  ∃ (P : Set (ℝ × ℝ) → ℝ),
    (∀ x y, Real.cos (Real.cos x) = Real.cos (Real.cos y) → -5 * π ≤ x → x ≤ 5 * π → -5 * π ≤ y → y ≤ 5 * π →
      P {(x, y)} = prob_X_equals_Y x y) →
    (∫ x in -5*π..5*π, ∫ y in -5*π..5*π, P {(x, y)} / (10 * π)^2) = 11/100 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_X_equals_Y_l127_12707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_ellipse_properties_l127_12733

/-- An ellipse centered at the origin with specific properties -/
structure SpecialEllipse where
  -- The equation of the ellipse
  equation : ℝ → ℝ → Prop
  -- The left focus is at (-1, 0)
  left_focus : equation (-1) 0
  -- The ratio condition when the line is perpendicular to x-axis
  ratio_condition : ∀ y, equation (-1) y → 
    (4 * (1 + Real.sqrt (1 + y^2))) / (2 * y) = 2 * Real.sqrt 2

/-- The circle passing through specific points related to the ellipse -/
noncomputable def related_circle (e : SpecialEllipse) : ℝ → ℝ → Prop :=
  λ x y ↦ (x + 1/2)^2 + (y - Real.sqrt 2)^2 = 9/4

/-- The dot product of vectors from the right focus to two points on the ellipse -/
def dot_product (e : SpecialEllipse) : ℝ → ℝ → ℝ → ℝ → ℝ :=
  λ x₁ y₁ x₂ y₂ ↦ (x₁ - 1) * (x₂ - 1) + y₁ * y₂

/-- The main theorem stating the properties of the special ellipse -/
theorem special_ellipse_properties (e : SpecialEllipse) :
  (∀ x y, e.equation x y ↔ x^2/2 + y^2 = 1) ∧
  (∀ x y, related_circle e x y ↔ (x + 1/2)^2 + (y - Real.sqrt 2)^2 = 9/4) ∧
  (∃ M, M = 7/2 ∧ ∀ x₁ y₁ x₂ y₂, 
    e.equation x₁ y₁ → e.equation x₂ y₂ → dot_product e x₁ y₁ x₂ y₂ ≤ M) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_ellipse_properties_l127_12733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l127_12792

theorem triangle_inequality (a b c γ : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_pos_γ : 0 < γ ∧ γ < π) (h_triangle : c = Real.sqrt (a^2 + b^2 - 2*a*b*(Real.cos γ))) :
  c ≥ ((a + b) * Real.sin γ) / 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l127_12792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_decreasing_f_implies_a_range_l127_12780

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 2 then (a - 2) * x else (1/2)^x - 1

-- State the theorem
theorem monotone_decreasing_f_implies_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a y < f a x) →
  a ∈ Set.Iic (13/8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_decreasing_f_implies_a_range_l127_12780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_distance_to_line_l127_12734

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the problem conditions
def circleConditions (c : Circle) : Prop :=
  -- The circle passes through (2,1)
  (c.center.1 - 2)^2 + (c.center.2 - 1)^2 = c.radius^2 ∧
  -- The circle is tangent to both coordinate axes
  c.center.1 = c.radius ∧ c.center.2 = c.radius

-- Define the distance function from a point to a line Ax + By + C = 0
noncomputable def distanceToLine (point : ℝ × ℝ) (A B C : ℝ) : ℝ :=
  (|A * point.1 + B * point.2 + C|) / Real.sqrt (A^2 + B^2)

-- The main theorem
theorem circle_distance_to_line :
  ∀ c : Circle, circleConditions c →
    distanceToLine c.center 2 (-1) (-3) = (2 * Real.sqrt 5) / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_distance_to_line_l127_12734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l127_12751

def f (x : ℝ) := -(x - 2)^2 + 1

theorem range_of_f :
  ∀ y, y ∈ Set.range (fun x ↦ f x) ∩ Set.Icc (-8) 1 ↔
    ∃ x ∈ Set.Icc 0 5, f x = y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l127_12751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_travel_time_l127_12700

-- Define the total distance from A to B as 1 unit
noncomputable def total_distance : ℝ := 1

-- Define the cyclist's speed (in units of distance per hour)
variable (x : ℝ)

-- Define the car's speed (3 times the cyclist's speed)
noncomputable def car_speed (x : ℝ) : ℝ := 3 * x

-- Define the time difference between the cyclist and car start times (in hours)
noncomputable def time_difference : ℝ := 1/4

-- Theorem statement
theorem cyclist_travel_time (x : ℝ) :
  -- Condition: Car catches up with cyclist halfway
  (1 / (2 * x)) = (1 / (2 * car_speed x) + time_difference) →
  -- Condition: When car arrives at B, cyclist has 1/3 left
  (1 / x - 1 / (car_speed x)) = 1/3 →
  -- Conclusion: Total travel time for cyclist is 45 minutes (3/4 hour)
  1 / x = 3/4 := by
  sorry

-- Example usage (optional)
#check cyclist_travel_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_travel_time_l127_12700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l127_12761

/-- The function f(x) = (1/2)^(-x^2 + 4x + 1) -/
noncomputable def f (x : ℝ) : ℝ := (1/2) ^ (-x^2 + 4*x + 1)

/-- The domain of x -/
def domain (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 3

/-- The range of f(x) -/
def range (y : ℝ) : Prop := ∃ x, domain x ∧ f x = y

/-- Theorem stating the range of f(x) -/
theorem f_range :
  ∀ y, range y ↔ 1/32 ≤ y ∧ y ≤ 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l127_12761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_current_solution_l127_12701

/-- The complex number representing voltage -/
def V : ℂ := 1 - Complex.I

/-- The complex number representing impedance -/
def Z : ℂ := 1 + 3 * Complex.I

/-- The complex number representing current -/
noncomputable def I : ℂ := V / Z

theorem current_solution :
  I = -1/5 - 2/5 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_current_solution_l127_12701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_is_60_degrees_max_perimeter_max_perimeter_equality_l127_12760

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  S : ℝ

-- Define the given conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.A + t.B + t.C = Real.pi ∧
  t.S > 0 ∧
  Real.sqrt 3 * (t.a * t.b * Real.cos t.C) = 2 * t.S

-- Theorem for part I
theorem angle_C_is_60_degrees (t : Triangle) (h : triangle_conditions t) :
  t.C = Real.pi / 3 := by
  sorry

-- Theorem for part II
theorem max_perimeter (t : Triangle) (h : triangle_conditions t) (h_c : t.c = Real.sqrt 6) :
  t.a + t.b + t.c ≤ 3 * Real.sqrt 6 := by
  sorry

-- Theorem for equality condition in part II
theorem max_perimeter_equality (t : Triangle) (h : triangle_conditions t) (h_c : t.c = Real.sqrt 6) :
  t.a + t.b + t.c = 3 * Real.sqrt 6 ↔ t.a = t.b ∧ t.b = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_is_60_degrees_max_perimeter_max_perimeter_equality_l127_12760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l127_12790

def sequence_a : ℕ → ℚ
  | 0 => 1
  | 1 => 3
  | 2 => 6
  | 3 => 10
  | n + 4 => sequence_a (n + 3) + (n + 4 : ℚ)

theorem sequence_formula (n : ℕ) :
  sequence_a n = (n + 1 : ℚ) * (n + 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l127_12790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_proof_l127_12797

/-- A circle C in the xy-plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The equation of a line in the xy-plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in the xy-plane -/
abbrev Point := ℝ × ℝ

/-- Check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c = 0

/-- Check if a point lies on a circle -/
def Point.liesOnCircle (p : Point) (c : Circle) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

/-- The circle C -/
noncomputable def C : Circle := { center := (2, -3), radius := Real.sqrt 5 }

/-- The line on which the center of C lies -/
def L : Line := { a := 2, b := -1, c := -7 }

/-- Point A on the y-axis -/
def A : Point := (0, -4)

/-- Point B on the y-axis -/
def B : Point := (0, -2)

theorem circle_equation_proof :
  (Point.liesOn C.center L) ∧
  (Point.liesOnCircle A C) ∧
  (Point.liesOnCircle B C) := by
  sorry

#check circle_equation_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_proof_l127_12797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_and_g_range_l127_12745

open Real

/-- The function f(x) defined on (1, +∞) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2 * x^2 - 2*x) * log x + 3/4 * x^2 - (a + 1) * x + 1

/-- The minimum value of f(x) when -1 < a < 1 -/
noncomputable def g (a : ℝ) : ℝ := 
  let m := exp ((a + 3 - 2 * exp 1) / (exp 1 - 2))
  ((1/2 * m^2 - 2*m) * log m + 3/4 * m^2 - (a + 1) * m + 1)

theorem f_increasing_and_g_range :
  (∀ a ≤ -1, StrictMono (f a)) ∧
  (∀ a, -1 < a → a < 1 → -2 * log 2 < g a ∧ g a < 7/4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_and_g_range_l127_12745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scientific_notation_and_rounding_l127_12781

-- Define the original number
def original_number : ℝ := 12480

-- Define the scientific notation representation
def scientific_notation : ℝ := 1.248 * (10 ^ 4)

-- Define the rounded scientific notation
def rounded_scientific_notation : ℝ := 1.25 * (10 ^ 4)

-- Helper function to round to three significant figures
-- This is a placeholder and should be properly implemented
noncomputable def round_to_three_sig_figs (x : ℝ) : ℝ :=
  sorry

-- Theorem to prove the scientific notation and rounding
theorem scientific_notation_and_rounding :
  (original_number = scientific_notation) ∧
  (round_to_three_sig_figs scientific_notation = rounded_scientific_notation) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_scientific_notation_and_rounding_l127_12781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vaccination_rates_and_m_value_l127_12725

-- Define the vaccination rates
noncomputable def vaccination_rate_A : ℝ → ℝ := sorry
noncomputable def vaccination_rate_B : ℝ → ℝ := sorry

-- Define the conditions
axiom rate_ratio : vaccination_rate_A 0 = 1.25 * vaccination_rate_B 0
axiom time_difference : 3000 / vaccination_rate_A 0 = 4000 / vaccination_rate_B 0 - 2
axiom rate_increase_B : vaccination_rate_B 1 = 1.25 * vaccination_rate_B 0
axiom rate_decrease_A (m : ℝ) : vaccination_rate_A 1 = vaccination_rate_A 0 - 5 * m
axiom rate_lower_bound_A (m : ℝ) : vaccination_rate_A 1 ≥ 800
axiom vaccination_difference (m : ℝ) : 
  vaccination_rate_B 1 * (m + 15) = vaccination_rate_A 1 * (2 * m) + 6000

-- State the theorem
theorem vaccination_rates_and_m_value : 
  ∃ m : ℝ, 
    vaccination_rate_A 0 = 1000 ∧ 
    vaccination_rate_B 0 = 800 ∧ 
    m = 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vaccination_rates_and_m_value_l127_12725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_properties_l127_12736

noncomputable def expansion (x : ℝ) (n : ℕ) := (Real.sqrt x + 1 / (2 * Real.rpow x (1/4)))^n

def coeff (n : ℕ) (r : ℕ) : ℚ := (Nat.choose n (r-1) : ℚ) * (2^(1-r))

def arithmetic_sequence (n : ℕ) : Prop :=
  coeff n 1 + coeff n 3 = 2 * coeff n 2

def is_rational_term (n r : ℕ) : Prop :=
  ∃ (k : ℤ), (4 : ℚ) - (3/4) * r = k

theorem expansion_properties :
  ∃ n : ℕ, arithmetic_sequence n ∧
    (n = 8 ∧
     (is_rational_term n 0 ∧ is_rational_term n 4 ∧ is_rational_term n 8) ∧
     (∀ r, r ≠ 0 ∧ r ≠ 4 ∧ r ≠ 8 → ¬is_rational_term n r) ∧
     (coeff n 3 = coeff n 4 ∧
      ∀ r, r ≠ 3 ∧ r ≠ 4 → coeff n r ≤ coeff n 3)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_properties_l127_12736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_l127_12747

noncomputable def original_function (x : ℝ) : ℝ := Real.sin x

noncomputable def transformed_function (x : ℝ) : ℝ := Real.sin ((1/2) * x - Real.pi/10)

theorem function_transformation :
  ∀ x : ℝ, transformed_function x = original_function (2 * (x - Real.pi/10)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_l127_12747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_result_is_correct_l127_12744

-- Define the initial numbers
def a : ℚ := 512.52
def b : ℚ := 256.26
def c : ℚ := 3

-- Define the operation
def result : ℚ := (a - b) * c

-- Define a function to round to the nearest hundredth
def round_to_hundredth (x : ℚ) : ℚ :=
  (x * 100).floor / 100 + if (x * 100 - (x * 100).floor ≥ 1/2) then 1/100 else 0

-- State the theorem
theorem final_result_is_correct :
  round_to_hundredth result = 768.78 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_result_is_correct_l127_12744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_union_equals_four_l127_12778

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Reflect a point about the line y = 1 -/
def reflect (p : Point) : Point :=
  { x := p.x, y := 2 - p.y }

/-- Calculate the area of a triangle using the Shoelace formula -/
noncomputable def triangleArea (t : Triangle) : ℝ :=
  let x1 := t.A.x
  let y1 := t.A.y
  let x2 := t.B.x
  let y2 := t.B.y
  let x3 := t.C.x
  let y3 := t.C.y
  (1/2) * abs (x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))

theorem area_of_union_equals_four :
  let t : Triangle := { A := { x := 6, y := 5 },
                        B := { x := 8, y := 3 },
                        C := { x := 9, y := 1 } }
  let t' : Triangle := { A := reflect t.A,
                         B := reflect t.B,
                         C := reflect t.C }
  (max (triangleArea t) (triangleArea t')) = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_union_equals_four_l127_12778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tulip_percentage_is_ten_percent_l127_12738

/-- Represents the sales data for a flower shop over three days -/
structure FlowerSales where
  tulip_price : ℚ
  rose_price : ℚ
  day1_tulips : ℕ
  day1_roses : ℕ
  day3_roses : ℕ
  total_revenue : ℚ

/-- Calculates the percentage of tulips sold on the third day compared to the second day -/
def tulip_percentage (sales : FlowerSales) : ℚ :=
  let day2_tulips := 2 * sales.day1_tulips
  let day2_roses := 2 * sales.day1_roses
  let day1_revenue := (sales.day1_tulips : ℚ) * sales.tulip_price + (sales.day1_roses : ℚ) * sales.rose_price
  let day2_revenue := 2 * day1_revenue
  let day3_revenue := sales.total_revenue - day1_revenue - day2_revenue
  let day3_tulip_revenue := day3_revenue - (sales.day3_roses : ℚ) * sales.rose_price
  let day3_tulips := day3_tulip_revenue / sales.tulip_price
  100 * day3_tulips / (day2_tulips : ℚ)

/-- Theorem stating that the percentage of tulips sold on the third day is 10% -/
theorem tulip_percentage_is_ten_percent (sales : FlowerSales)
  (h1 : sales.tulip_price = 2)
  (h2 : sales.rose_price = 3)
  (h3 : sales.day1_tulips = 30)
  (h4 : sales.day1_roses = 20)
  (h5 : sales.day3_roses = 16)
  (h6 : sales.total_revenue = 420) :
  tulip_percentage sales = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tulip_percentage_is_ten_percent_l127_12738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_property_l127_12703

noncomputable def f (a c : ℝ) (x : ℝ) : ℝ := a * (x^2 - x) + c

theorem quadratic_function_property (a c : ℝ) (x₁ x₂ x₃ : ℝ) 
  (h_a_nonzero : a ≠ 0)
  (h_distinct : x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃)
  (h_eq₁ : f a c x₁ - a * x₂ = 0)
  (h_eq₂ : f a c x₂ - a * x₃ = 0)
  (h_eq₃ : f a c x₃ - a * x₁ = 0) :
  a^2 > a * c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_property_l127_12703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_distance_l127_12754

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := 4 * x^2 - y^2 + 64 = 0

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Define the foci of the hyperbola
def focus1 (c : ℝ) : ℝ × ℝ := (c, 0)
def focus2 (c : ℝ) : ℝ × ℝ := (-c, 0)

-- Theorem statement
theorem hyperbola_focus_distance (x y c : ℝ) :
  hyperbola x y →
  (∃ f : ℝ × ℝ, (f = focus1 c ∨ f = focus2 c) ∧ distance x y f.1 f.2 = 1) →
  ∃ f' : ℝ × ℝ, (f' = focus1 c ∨ f' = focus2 c) ∧ f' ≠ f ∧ distance x y f'.1 f'.2 = 17 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_distance_l127_12754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_error_calculation_l127_12715

/-- Calculates the percentage error between the correct and incorrect results of a mathematical operation. -/
theorem percentage_error_calculation : 
  let correct_result := (5 * Real.pi / 2) ^ 2
  let incorrect_result := (2 * Real.pi / 5) ^ 2
  let absolute_error := |correct_result - incorrect_result|
  let percentage_error := (absolute_error / correct_result) * 100
  ∃ (ε : ℝ), ε > 0 ∧ |percentage_error - 97.44| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_error_calculation_l127_12715
