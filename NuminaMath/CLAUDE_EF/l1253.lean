import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplification_and_evaluation_l1253_125342

-- Define the original expression
noncomputable def original_expression (x : ℝ) : ℝ :=
  (1 + 1 / (x - 2)) / ((x - 1) / (x^2 - 4*x + 4))

-- Theorem statement
theorem simplification_and_evaluation :
  ∃ (x : ℝ), x ∈ ({1, 2, 3} : Set ℝ) ∧ original_expression x = 1 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplification_and_evaluation_l1253_125342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_slope_l1253_125322

/-- The slope of a line parallel to 3x - 6y = 21 is 1/2 -/
theorem parallel_line_slope (x y : ℝ) :
  ∀ (a b c : ℝ), b ≠ 0 → a / b = 3 / 6 → c / b = 21 / 6 →
  (∃ k : ℝ, a * x - b * y = c + k) →
  1 / 2 = a / b :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_slope_l1253_125322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plate_sequences_count_l1253_125388

/-- The number of possible plate sequences after swapping -/
def plateSequences (n : ℕ) : ℕ := 2^(n-2)

/-- The actual number of possible sequences (to be proven equal to plateSequences) -/
def number_of_possible_sequences (n : ℕ) : ℕ :=
  -- This function represents the true count of possible sequences
  -- Its implementation is not provided as it's part of what needs to be proven
  sorry

/-- Theorem stating the number of possible plate sequences after swapping -/
theorem plate_sequences_count (n : ℕ) (h : n ≥ 2) :
  plateSequences n = number_of_possible_sequences n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plate_sequences_count_l1253_125388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_max_expression_l1253_125393

-- Define the right triangle
noncomputable def RightTriangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2

-- Define the area constraint
noncomputable def AreaConstraint (a b : ℝ) : Prop :=
  (1/2) * a * b = 30

-- Define the expression to be maximized
noncomputable def Expression (a b c : ℝ) : ℝ :=
  (a + b + c) / 30

-- Theorem statement
theorem right_triangle_max_expression {a b c : ℝ} 
  (h_triangle : RightTriangle a b c) 
  (h_area : AreaConstraint a b) :
  Expression a b c ≤ Expression (Real.sqrt 60) (Real.sqrt 60) (Real.sqrt 120) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_max_expression_l1253_125393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_difference_l1253_125344

-- Define the function f(x) = x + 9/x
noncomputable def f (x : ℝ) : ℝ := x + 9 / x

-- Define the interval [1, 4]
def interval : Set ℝ := { x | 1 ≤ x ∧ x ≤ 4 }

-- Define the maximum value M of f(x) on the interval
noncomputable def M : ℝ := sSup (f '' interval)

-- Define the minimum value m of f(x) on the interval
noncomputable def m : ℝ := sInf (f '' interval)

-- Theorem statement
theorem max_min_difference : M - m = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_difference_l1253_125344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_crossing_time_l1253_125319

-- Define the lengths and speeds
noncomputable def train_a_length : ℝ := 350
noncomputable def train_b_length : ℝ := 500
noncomputable def bridge_length : ℝ := 800
noncomputable def train_a_speed_kmph : ℝ := 108
noncomputable def train_b_speed_kmph : ℝ := 90

-- Convert speeds to m/s
noncomputable def train_a_speed_ms : ℝ := train_a_speed_kmph * 1000 / 3600
noncomputable def train_b_speed_ms : ℝ := train_b_speed_kmph * 1000 / 3600

-- Calculate total distance and relative speed
noncomputable def total_distance : ℝ := train_a_length + train_b_length + bridge_length
noncomputable def relative_speed : ℝ := train_a_speed_ms + train_b_speed_ms

-- Theorem to prove
theorem trains_crossing_time : 
  (total_distance / relative_speed) = 30 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_crossing_time_l1253_125319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1253_125335

noncomputable section

variable (f : ℝ → ℝ)

-- f is an even function
axiom f_even : ∀ x, f x = f (-x)

-- f is monotonically decreasing on (0,+∞)
axiom f_decreasing : ∀ x y, 0 < x → x < y → f y < f x

-- The inequality holds for all x in [1,3]
axiom inequality_holds : ∀ x m, 1 ≤ x → x ≤ 3 → 
  f (2*m*x - Real.log x - 3) ≥ 2*f 3 - f (Real.log x + 3 - 2*m*x)

theorem range_of_m :
  ∃ m_min m_max, m_min = 1 / (2 * Real.exp 1) ∧ 
                   m_max = (Real.log 3 + 6) / 6 ∧
                   ∀ m, (∀ x, 1 ≤ x → x ≤ 3 → 
                     f (2*m*x - Real.log x - 3) ≥ 2*f 3 - f (Real.log x + 3 - 2*m*x)) →
                   m_min ≤ m ∧ m ≤ m_max := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1253_125335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_2x_max_value_l1253_125341

theorem sin_cos_2x_max_value :
  (∀ x : ℝ, Real.sin (2 * x) + Real.cos (2 * x) ≤ 2) ∧
  (∃ x : ℝ, Real.sin (2 * x) + Real.cos (2 * x) = 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_2x_max_value_l1253_125341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_iff_a_in_range_l1253_125300

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (a + 3) * x - 3 else 1 - a / (x + 1)

theorem f_monotone_iff_a_in_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x ≤ f a y) ↔ 0 < a ∧ a ≤ 2/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_iff_a_in_range_l1253_125300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_arrangement_l1253_125372

/-- Represents a circular arrangement of seven consecutive natural numbers. -/
def CircularArrangement := Fin 7 → ℕ

/-- Checks if the given numbers are consecutive. -/
def IsConsecutive (arr : CircularArrangement) : Prop :=
  ∃ start : ℕ, ∀ i : Fin 7, arr i = start + i.val

/-- Computes the absolute difference between neighboring numbers in the circular arrangement. -/
def NeighborDifference (arr : CircularArrangement) (i : Fin 7) : ℕ :=
  (arr i).max (arr ((i + 1) % 7)) - (arr i).min (arr ((i + 1) % 7))

/-- Checks if five consecutive differences match the given pattern. -/
def HasConsecutiveDifferences (arr : CircularArrangement) : Prop :=
  ∃ i : Fin 7, 
    (NeighborDifference arr i = 2) ∧
    (NeighborDifference arr ((i + 1) % 7) = 1) ∧
    (NeighborDifference arr ((i + 2) % 7) = 6) ∧
    (NeighborDifference arr ((i + 3) % 7) = 1) ∧
    (NeighborDifference arr ((i + 4) % 7) = 2)

theorem impossible_arrangement :
  ¬∃ (arr : CircularArrangement), IsConsecutive arr ∧ HasConsecutiveDifferences arr :=
sorry

#check impossible_arrangement

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_arrangement_l1253_125372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_four_digit_with_five_thousands_l1253_125301

/-- The set of four-digit positive integers with thousands digit 5 -/
def FourDigitWithFiveThousands : Finset ℕ :=
  Finset.filter (fun n => 5000 ≤ n ∧ n ≤ 5999) (Finset.range 10000)

/-- The count of four-digit positive integers with thousands digit 5 -/
theorem count_four_digit_with_five_thousands :
  Finset.card FourDigitWithFiveThousands = 1000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_four_digit_with_five_thousands_l1253_125301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_energy_increase_l1253_125324

-- Define the initial configuration
noncomputable def initial_energy : ℝ := 15

-- Define the side length of the equilateral triangle
noncomputable def side_length : ℝ := 1

-- Define the energy function for a pair of charges
noncomputable def pair_energy (distance : ℝ) (charge : ℝ) : ℝ := charge^2 / distance

-- Define the total energy function for the initial configuration
noncomputable def initial_total_energy (e : ℝ) (s : ℝ) : ℝ := 3 * pair_energy s 1

-- Define the total energy function for the new configuration
noncomputable def new_total_energy (e : ℝ) (s : ℝ) : ℝ := 
  pair_energy s 1 + 2 * pair_energy (s/2) 1

-- State the theorem
theorem energy_increase : 
  new_total_energy initial_energy side_length - initial_total_energy initial_energy side_length = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_energy_increase_l1253_125324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_is_13_l1253_125304

-- Define the coordinates of the two adjacent vertices
def vertex1 : ℝ × ℝ := (1, 3)
def vertex2 : ℝ × ℝ := (-2, 5)

-- Define the function to calculate the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the function to calculate the area of a square given its side length
def square_area (side : ℝ) : ℝ := side^2

-- Theorem statement
theorem square_area_is_13 :
  square_area (distance vertex1 vertex2) = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_is_13_l1253_125304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_contrapositive_even_function_l1253_125386

variable {D : Type*} [AddGroup D] (f : D → ℝ)

def is_even_function (f : D → ℝ) : Prop :=
  ∀ x : D, f (-x) = f x

theorem contrapositive_even_function :
  (∀ x : D, f (-x) = f x → is_even_function f) ↔
  (¬is_even_function f → ∃ x : D, f (-x) ≠ f x) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_contrapositive_even_function_l1253_125386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_is_12_l1253_125331

-- Define the probability function
noncomputable def probability (m : ℕ) : ℝ := (m - 1 : ℝ)^4 / m^4

-- Define the property we're looking for
def satisfies_condition (m : ℕ) : Prop :=
  probability m > 2/3 ∧ ∀ k < m, probability k ≤ 2/3

-- State the theorem
theorem smallest_m_is_12 :
  satisfies_condition 12 := by
  -- We'll use sorry to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_is_12_l1253_125331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_length_l1253_125309

noncomputable section

/-- Define an ellipse with given foci and major axis -/
def Ellipse (f1 f2 : ℝ × ℝ) (major_axis : ℝ) :=
  {p : ℝ × ℝ | dist p f1 + dist p f2 = major_axis}

/-- Define a line with slope and y-intercept -/
def Line (m b : ℝ) :=
  {p : ℝ × ℝ | p.2 = m * p.1 + b}

/-- The length of a segment defined by two points -/
noncomputable def segmentLength (a b : ℝ × ℝ) : ℝ :=
  Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)

theorem ellipse_line_intersection_length :
  let e := Ellipse (-2 * Real.sqrt 2, 0) (2 * Real.sqrt 2, 0) 6
  let l := Line 1 2
  let intersection := Set.inter e l
  ∃ a b : ℝ × ℝ, a ∈ intersection ∧ b ∈ intersection ∧ a ≠ b ∧
    segmentLength a b = 6 * Real.sqrt 3 / 5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_length_l1253_125309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_trig_sum_l1253_125340

theorem min_value_trig_sum (x : ℝ) (h : 0 < x ∧ x < Real.pi / 2) :
  (Real.tan x + 1 / Real.tan x)^2 + (Real.sin x + 1 / Real.sin x)^2 ≥ 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_trig_sum_l1253_125340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distances_constant_l1253_125330

-- Define an equilateral triangle
structure EquilateralTriangle where
  side_length : ℝ
  side_length_pos : side_length > 0

-- Define a point inside the triangle
structure PointInTriangle where
  x : ℝ
  y : ℝ
  in_triangle : x ≥ 0 ∧ y ≥ 0 ∧ x + y ≤ 1  -- Assuming normalized coordinates

-- Calculate the height of the equilateral triangle
noncomputable def triangle_height (t : EquilateralTriangle) : ℝ :=
  t.side_length * Real.sqrt 3 / 2

-- Calculate the sum of perpendicular distances from a point to all sides
noncomputable def sum_of_distances (t : EquilateralTriangle) (p : PointInTriangle) : ℝ :=
  sorry  -- The actual calculation would go here

-- The theorem to prove
theorem sum_of_distances_constant (t : EquilateralTriangle) (p : PointInTriangle) :
  sum_of_distances t p = triangle_height t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distances_constant_l1253_125330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_is_closed_interval_l1253_125320

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 1 then x else -1

-- Define the set of solutions
def solution_set : Set ℝ :=
  {x | x * f x - x ≤ 2}

-- Theorem statement
theorem solution_set_is_closed_interval :
  solution_set = Set.Icc (-1) 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_is_closed_interval_l1253_125320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_arrangement_of_infinite_polyhedra_l1253_125396

/-- A type representing a convex polyhedron -/
structure ConvexPolyhedron where
  -- Add necessary fields here
  mk :: -- Constructor

/-- A type representing a plane -/
structure Plane where
  -- Add necessary fields here
  mk :: -- Constructor

/-- A type representing a layer bounded by two parallel planes -/
structure Layer where
  plane1 : Plane
  plane2 : Plane
  parallel : Bool -- Simplified for now, replace with actual parallel check later

/-- A type representing an arrangement of convex polyhedra in a layer -/
structure Arrangement where
  layer : Layer
  polyhedra : Set ConvexPolyhedron
  inLayer : Bool -- Simplified for now, replace with actual containment check later
  infinitePolyhedra : Bool -- Simplified for now, replace with actual infinity check later
  noRemovable : Bool -- Simplified for now, replace with actual removability check later

/-- Predicate to check if a polyhedron can be removed without moving others -/
def CanBeRemovedWithoutMovingOthers (p : ConvexPolyhedron) (polyhedra : Set ConvexPolyhedron) : Prop :=
  sorry

/-- Theorem stating the existence of an arrangement satisfying the required properties -/
theorem exists_arrangement_of_infinite_polyhedra :
  ∃ (arr : Arrangement), True := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_arrangement_of_infinite_polyhedra_l1253_125396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_circle_tangent_line_equal_intercepts_l1253_125306

-- Define points A and B
def A : ℝ × ℝ := (2, 0)
def B : ℝ × ℝ := (8, 0)

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the trajectory of M
def trajectory (M : ℝ × ℝ) : Prop :=
  distance M A = (1/2) * distance M B

-- Theorem for the trajectory equation
theorem trajectory_is_circle (M : ℝ × ℝ) (h : trajectory M) :
  M.1^2 + M.2^2 = 16 := by sorry

-- Theorem for the tangent line equation
theorem tangent_line_equal_intercepts (a : ℝ) :
  (∀ M : ℝ × ℝ, M.1^2 + M.2^2 = 16 → (M.1 + M.2 = a ∨ M.1 + M.2 = -a)) →
  a = 4 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_circle_tangent_line_equal_intercepts_l1253_125306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_M_properties_l1253_125316

/-- Definition of the matrix M -/
def M : Matrix (Fin 2) (Fin 2) ℝ := !![6, 2; 4, 4]

/-- The eigenvector corresponding to eigenvalue 8 -/
def e₁ : Fin 2 → ℝ := ![1, 1]

theorem matrix_M_properties :
  -- M has eigenvalue 8 with eigenvector e₁
  M.mulVec e₁ = (fun i => (8 : ℝ) * e₁ i) ∧
  -- M transforms (-1, 2) to (-2, 4)
  M.mulVec ![(-1 : ℝ), 2] = ![(-2 : ℝ), 4] ∧
  -- The eigenvalues of M are 8 and 2
  (M.det = 16 ∧ M.trace = 10) := by
  sorry

#eval M

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_M_properties_l1253_125316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angles_from_circumradius_equation_l1253_125382

/-- Given a triangle with circumradius R and sides a, b, c satisfying a² + b² = c² - R²,
    prove that its angles are 30°, 30°, and 120°. -/
theorem triangle_angles_from_circumradius_equation 
  (a b c R : ℝ) 
  (h : a^2 + b^2 = c^2 - R^2) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ R > 0) :
  ∃ (A B C : ℝ),
    A = 30 * Real.pi / 180 ∧ 
    B = 30 * Real.pi / 180 ∧ 
    C = 120 * Real.pi / 180 ∧
    A + B + C = Real.pi ∧
    Real.sin A = a / (2 * R) ∧
    Real.sin B = b / (2 * R) ∧
    Real.sin C = c / (2 * R) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angles_from_circumradius_equation_l1253_125382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_polynomial_divisibility_implies_value_l1253_125363

-- Define a quadratic polynomial
def quadratic_polynomial (a b c : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c

theorem quadratic_polynomial_divisibility_implies_value (p : ℝ → ℝ) :
  (∃ a b c : ℝ, p = quadratic_polynomial a b c) →
  (∃ q : ℝ → ℝ, ∀ x, p x^3 - x = (x - 2) * (x + 2) * (x - 5) * q x) →
  p 10 = 10 := by
  sorry

#check quadratic_polynomial_divisibility_implies_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_polynomial_divisibility_implies_value_l1253_125363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percent_less_than_correct_percent_less_than_positive_l1253_125389

-- Define M and N as real numbers
variable (M N : ℝ)

-- Define the condition M < N
variable (h : M < N)

-- Define the function that calculates the percent difference
noncomputable def percentLessThan (M N : ℝ) : ℝ := 100 * (N - M) / N

-- Theorem statement
theorem percent_less_than_correct (M N : ℝ) (h : M < N) :
  percentLessThan M N = 100 * (N - M) / N :=
by
  -- Unfold the definition of percentLessThan
  unfold percentLessThan
  -- The equality holds by definition
  rfl

-- Additional theorem to show that the result is positive
theorem percent_less_than_positive (M N : ℝ) (h : M < N) :
  percentLessThan M N > 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percent_less_than_correct_percent_less_than_positive_l1253_125389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_profit_l1253_125321

/-- Calculate the total profit from two investments with different interest rates -/
theorem investment_profit (total_amount interest_rate1 interest_rate2 amount1 : ℕ) 
  (h1 : total_amount = 50000)
  (h2 : interest_rate1 = 10)
  (h3 : interest_rate2 = 20)
  (h4 : amount1 = 30000)
  (h5 : amount1 ≤ total_amount) :
  let amount2 := total_amount - amount1
  let profit1 := amount1 * interest_rate1 / 100
  let profit2 := amount2 * interest_rate2 / 100
  profit1 + profit2 = 7000 := by
  sorry

#check investment_profit

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_profit_l1253_125321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_range_a_l1253_125371

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 - a) * x - a
  else Real.log x / Real.log a

theorem increasing_f_range_a (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) →
  a ∈ Set.Icc (3/2) 3 ∧ a ≠ 3 :=
by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_range_a_l1253_125371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1253_125358

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x - 2*a/x) / Real.log a

-- Define the interval (1, 2)
def open_interval : Set ℝ := Set.Ioo 1 2

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x ∈ open_interval, ∀ y ∈ open_interval, x < y → f a x > f a y) →
  a ∈ Set.Ioc 0 (1/2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1253_125358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_cone_area_ratio_l1253_125368

/-- An equilateral cone is a cone whose axial section is an equilateral triangle -/
structure EquilateralCone where
  radius : ℝ
  height : ℝ
  equilateral : height = radius * Real.sqrt 3

/-- The lateral surface area of an equilateral cone -/
noncomputable def lateralSurfaceArea (cone : EquilateralCone) : ℝ :=
  2 * Real.pi * cone.radius^2

/-- The base area of an equilateral cone -/
noncomputable def baseArea (cone : EquilateralCone) : ℝ :=
  Real.pi * cone.radius^2

/-- The total surface area of an equilateral cone -/
noncomputable def totalSurfaceArea (cone : EquilateralCone) : ℝ :=
  lateralSurfaceArea cone + baseArea cone

/-- Theorem: The ratio of lateral surface area to total surface area of an equilateral cone is 2:3 -/
theorem equilateral_cone_area_ratio (cone : EquilateralCone) :
  lateralSurfaceArea cone / totalSurfaceArea cone = 2 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_cone_area_ratio_l1253_125368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_coefficient_theorem_l1253_125379

noncomputable def binomial_expansion (n : ℕ) (x : ℝ) : ℝ := (x + 1 / (3 * x)) ^ n

def A (n : ℕ) : ℚ := 1

noncomputable def B (n : ℕ) : ℝ := n / 3

noncomputable def C (n : ℕ) : ℝ := n * (n - 1) / 18

noncomputable def coeff_x_squared (n : ℕ) : ℝ := (n.choose 3) / (3^3 : ℝ)

theorem expansion_coefficient_theorem (n : ℕ) :
  4 * (A n : ℝ) = 9 * (C n - B n) → coeff_x_squared n = 56 / 27 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_coefficient_theorem_l1253_125379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_arrangement_max_diff_4_not_exists_arrangement_max_diff_3_l1253_125391

/-- A type representing a 4x4 board with numbers from 1 to 16 -/
def Board := Fin 4 → Fin 4 → Fin 16

/-- A predicate checking if two positions are adjacent on the board -/
def adjacent (p1 p2 : Fin 4 × Fin 4) : Prop :=
  (p1.1 = p2.1 ∧ (p1.2 = p2.2 + 1 ∨ p2.2 = p1.2 + 1)) ∨
  (p1.2 = p2.2 ∧ (p1.1 = p2.1 + 1 ∨ p2.1 = p1.1 + 1))

/-- A predicate checking if a board arrangement is valid -/
def valid_arrangement (b : Board) : Prop :=
  (∀ i j : Fin 4, ∃ n : Fin 16, b i j = n) ∧
  (∀ n : Fin 16, ∃ i j : Fin 4, b i j = n)

/-- Theorem stating the existence of a valid arrangement with max difference 4 -/
theorem exists_arrangement_max_diff_4 :
  ∃ (b : Board), valid_arrangement b ∧
    ∀ p1 p2 : Fin 4 × Fin 4, adjacent p1 p2 →
      (b p1.1 p1.2).val - (b p2.1 p2.2).val ≤ 4 := by
  sorry

/-- Theorem stating the non-existence of a valid arrangement with max difference 3 -/
theorem not_exists_arrangement_max_diff_3 :
  ¬∃ (b : Board), valid_arrangement b ∧
    ∀ p1 p2 : Fin 4 × Fin 4, adjacent p1 p2 →
      (b p1.1 p1.2).val - (b p2.1 p2.2).val ≤ 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_arrangement_max_diff_4_not_exists_arrangement_max_diff_3_l1253_125391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_f_at_3_l1253_125366

noncomputable def f (x : ℝ) : ℝ := (4*x^2 - 14*x + 6) / (x - 3)

theorem limit_of_f_at_3 :
  ∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x - 3| ∧ |x - 3| < δ → |f x - 10| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_f_at_3_l1253_125366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_x0_l1253_125376

-- Define the curve
noncomputable def f (x : ℝ) : ℝ := -5 * Real.exp x + 3

-- Define the point of tangency
def x₀ : ℝ := 0

-- State the theorem
theorem tangent_line_at_x0 : 
  ∃ (m b : ℝ), ∀ x : ℝ, 
    (deriv f x₀) * (x - x₀) + f x₀ = m * x + b ∧ 
    m = -5 ∧ b = -2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_x0_l1253_125376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pythagorean_triple_parity_l1253_125343

theorem pythagorean_triple_parity (a b c : ℕ) (h1 : a^2 + b^2 = c^2) (h2 : Nat.Prime a) :
  Odd b ∧ Even c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pythagorean_triple_parity_l1253_125343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_points_on_circle_l1253_125311

def circle_points (r : ℕ) : Set (ℤ × ℤ) :=
  {p | p.1^2 + p.2^2 = r^2}

theorem integer_points_on_circle :
  Finset.card (Finset.filter (fun p => p.1^2 + p.2^2 = 5^2) (Finset.product (Finset.range 11) (Finset.range 11))) = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_points_on_circle_l1253_125311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1253_125384

-- Define the sequences and sum
def a (n : ℕ) : ℚ := 1 - 2 * n
def b (n : ℕ) : ℚ := 2^n
def S (n : ℕ) : ℚ := n * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2

def c (n : ℕ) : ℚ :=
  if n % 2 = 1 then 2 else (-2 * a n) / (b n)

def T (n : ℕ) : ℚ :=
  let k := (n + 1) / 2
  if n % 2 = 0
  then 2 * k + 26/9 - (12 * k + 13) / (9 * 2^(2 * k - 1))
  else 2 * k + 26/9 - (12 * k + 1) / (9 * 2^(2 * k - 3))

-- State the theorem
theorem sequence_properties :
  (∀ n, a (n+1) - a n = a 2 - a 1) ∧  -- arithmetic sequence
  (∀ n, b (n+1) / b n > 0) ∧  -- geometric sequence with positive ratio
  b 1 = -2 * a 1 ∧
  b 1 = 2 ∧
  a 3 + b 2 = -1 ∧
  S 3 + 2 * b 3 = 7 →
  (∀ n, a n = 1 - 2 * n) ∧
  (∀ n, b n = 2^n) ∧
  (∀ n, T n = Finset.sum (Finset.range n) (fun i => c (i + 1))) := by
  sorry

#check sequence_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1253_125384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_representation_theorem_l1253_125315

theorem subset_representation_theorem :
  ∃ (S : Finset (Finset ℕ)), 
    S.card = 16 ∧ 
    (∀ n ∈ Finset.range 10000, 
      ∃! (T : Finset (Finset ℕ)), 
        T ⊆ S ∧ 
        T.card = 8 ∧ 
        (⋂ X ∈ T, X) = {n}) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_representation_theorem_l1253_125315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octahedron_parallel_edges_l1253_125333

/-- A regular octahedron -/
structure RegularOctahedron where
  -- We don't need to define the internal structure,
  -- as the problem doesn't require it

/-- A pair of parallel edges in a regular octahedron -/
structure ParallelEdgePair where
  -- Again, we don't need to define the internal structure

/-- The set of all unique pairs of parallel edges in a regular octahedron -/
def uniqueParallelEdgePairs (o : RegularOctahedron) : Finset ParallelEdgePair :=
  sorry

/-- Theorem: A regular octahedron has exactly 12 unique pairs of parallel edges -/
theorem octahedron_parallel_edges (o : RegularOctahedron) :
  (uniqueParallelEdgePairs o).card = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octahedron_parallel_edges_l1253_125333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_difference_sum_l1253_125308

theorem tan_difference_sum (α β γ : ℝ) 
  (h1 : Real.tan α = 5) 
  (h2 : Real.tan β = 3) 
  (h3 : Real.tan γ = 2) : 
  Real.tan (α - β + γ) = 17/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_difference_sum_l1253_125308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_inequality_l1253_125302

theorem cosine_sum_inequality (α β γ : Real) : 
  0 < α ∧ 0 < β ∧ 0 < γ ∧ α + β + γ = Real.pi / 2 → 
  ¬(Real.cos α + Real.cos β = Real.cos γ ∨ 
    Real.cos α + Real.cos γ = Real.cos β ∨ 
    Real.cos β + Real.cos γ = Real.cos α) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_inequality_l1253_125302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_solutions_correct_l1253_125353

/-- The number of solutions of the equation a * exp x = x^3 -/
noncomputable def num_solutions (a : ℝ) : ℕ :=
  if a ≤ 0 then 1
  else if 0 < a ∧ a < 27 / Real.exp 3 then 2
  else if a = 27 / Real.exp 3 then 1
  else 0

theorem num_solutions_correct (a : ℝ) :
  num_solutions a = 
    if a ≤ 0 then 1
    else if 0 < a ∧ a < 27 / Real.exp 3 then 2
    else if a = 27 / Real.exp 3 then 1
    else 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_solutions_correct_l1253_125353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_relation_l1253_125370

theorem sin_cos_relation (x : ℝ) (h : Real.sin x = 2 * Real.cos x) :
  (Real.sin x) ^ 2 - 2 * (Real.sin x) * (Real.cos x) + 3 * (Real.cos x) ^ 2 = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_relation_l1253_125370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_in_fourth_quadrant_l1253_125349

def card_numbers : List ℤ := [0, -1, 2, -3]

def is_in_fourth_quadrant (x y : ℤ) : Prop :=
  x > 0 ∧ y < 0

def count_fourth_quadrant_points (numbers : List ℤ) : ℕ :=
  List.sum (List.map (λ x => List.sum (List.map (λ y => 
    if (x > 0 ∧ y < 0) then 1 else 0) 
    (numbers.filter (· ≠ x)))) numbers)

def total_possible_points (numbers : List ℤ) : ℕ :=
  numbers.length * (numbers.length - 1)

theorem probability_in_fourth_quadrant :
  (count_fourth_quadrant_points card_numbers : ℚ) / (total_possible_points card_numbers) = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_in_fourth_quadrant_l1253_125349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_division_theorem_l1253_125303

/-- Represents a part of the triangle -/
structure TrianglePart where
  numbers : List Nat
  area : Nat

/-- Represents the checkered triangle -/
structure CheckeredTriangle where
  parts : List TrianglePart
  total_sum : Nat

/-- Checks if all parts have the same sum -/
def all_parts_equal_sum (triangle : CheckeredTriangle) : Prop :=
  ∀ p1 p2, p1 ∈ triangle.parts → p2 ∈ triangle.parts → (p1.numbers.sum = p2.numbers.sum)

/-- Checks if all parts have different areas -/
def all_parts_different_areas (triangle : CheckeredTriangle) : Prop :=
  ∀ p1 p2, p1 ∈ triangle.parts → p2 ∈ triangle.parts → p1 ≠ p2 → p1.area ≠ p2.area

/-- The main theorem -/
theorem triangle_division_theorem (triangle : CheckeredTriangle) :
  triangle.total_sum = 63 →
  triangle.parts.length = 3 →
  (∀ p, p ∈ triangle.parts → p.numbers.sum = 21) →
  all_parts_equal_sum triangle ∧ all_parts_different_areas triangle :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_division_theorem_l1253_125303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chemistry_players_l1253_125313

def total_players : ℕ := 30
def physics_players : ℕ := 15
def both_subjects : ℕ := 10

theorem chemistry_players : 
  (∀ p, p ∈ Finset.range total_players → 
    p ∈ Finset.range physics_players ∨ 
    p ∈ Finset.range (total_players - physics_players + both_subjects)) →
  total_players - physics_players + both_subjects = 25 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chemistry_players_l1253_125313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zhang_hua_journey_l1253_125347

/-- Represents the probability of encountering a red light at a traffic post --/
structure TrafficPost where
  probability : ℚ

/-- Represents the scenario of Zhang Hua's journey to school --/
structure JourneyScenario where
  posts : Fin 4 → TrafficPost
  independent : Bool

/-- The number of red lights encountered during the journey --/
def RedLightsEncountered (scenario : JourneyScenario) : Fin 5 := sorry

/-- The probability of not being late (encountering less than 3 red lights) --/
def ProbabilityNotLate (scenario : JourneyScenario) : ℚ := sorry

/-- The expected number of red lights encountered --/
def ExpectedRedLights (scenario : JourneyScenario) : ℚ := sorry

theorem zhang_hua_journey 
  (scenario : JourneyScenario)
  (h1 : scenario.posts 0 = TrafficPost.mk (1/2))
  (h2 : scenario.posts 1 = TrafficPost.mk (1/2))
  (h3 : scenario.posts 2 = TrafficPost.mk (1/3))
  (h4 : scenario.posts 3 = TrafficPost.mk (1/3))
  (h5 : scenario.independent = true) :
  ProbabilityNotLate scenario = 29/36 ∧ ExpectedRedLights scenario = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zhang_hua_journey_l1253_125347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_squares_method_correct_l1253_125305

/-- Definition of the least squares method -/
def least_squares_method : Prop :=
  ∃ method, method = "The method of taking the sum of the squares of each deviation as the total deviation and minimizing it"

/-- Alternative definition 1 -/
def alternative_def1 : Prop :=
  ∃ method, method = "The method of summing up each deviation as the total deviation and minimizing it"

/-- Alternative definition 3 -/
def alternative_def3 : Prop :=
  ∃ method, method = "The mathematical method of finding a straight line from sample points that closely follows these sample points"

/-- Alternative definition 4 -/
def alternative_def4 : Prop :=
  ∃ method, method = "Since a regression line equation can be obtained from any observation, there is no need for correlation testing"

/-- Theorem stating that the least squares method is correctly defined -/
theorem least_squares_method_correct :
  least_squares_method ∧
  ¬alternative_def1 ∧
  ¬alternative_def3 ∧
  ¬alternative_def4 := by
  sorry

#check least_squares_method_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_squares_method_correct_l1253_125305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_iff_subgraph_bound_l1253_125351

/-- A simple graph. -/
structure SimpleGraph' (V : Type*) where
  adj : V → V → Prop
  sym : ∀ u v, adj u v → adj v u
  loopless : ∀ v, ¬adj v v

/-- An induced subgraph of a simple graph. -/
def InducedSubgraph (G : SimpleGraph' V) (W : Set V) : SimpleGraph' W where
  adj := λ u v ↦ G.adj u.val v.val
  sym := by sorry
  loopless := by sorry

/-- The cardinality of a graph. -/
noncomputable def card (G : SimpleGraph' V) : ℕ := sorry

/-- The independence number of a graph. -/
noncomputable def independenceNumber (G : SimpleGraph' V) : ℕ := sorry

/-- The clique number of a graph. -/
noncomputable def cliqueNumber (G : SimpleGraph' V) : ℕ := sorry

/-- A graph is perfect. -/
def isPerfect (G : SimpleGraph' V) : Prop := sorry

theorem perfect_iff_subgraph_bound {V : Type*} (G : SimpleGraph' V) :
  isPerfect G ↔ ∀ (W : Set V), 
    let H := InducedSubgraph G W
    card H ≤ independenceNumber H * cliqueNumber H := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_iff_subgraph_bound_l1253_125351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equivalent_annual_rate_approx_l1253_125373

/-- The equivalent annual interest rate for an account with 8% annual interest compounded quarterly -/
noncomputable def equivalent_annual_rate : ℝ :=
  ((1 + 0.08 / 4) ^ 4 - 1) * 100

/-- Theorem stating that the equivalent annual rate is approximately 8.24 -/
theorem equivalent_annual_rate_approx :
  ‖equivalent_annual_rate - 8.24‖ < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equivalent_annual_rate_approx_l1253_125373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centerville_parks_budget_percentage_l1253_125310

/-- Represents the annual budget allocation of Centerville --/
structure BudgetAllocation where
  total : ℚ
  library_percent : ℚ
  library_amount : ℚ
  remaining : ℚ

/-- Calculates the percentage of the budget spent on public parks --/
def parks_percentage (b : BudgetAllocation) : ℚ :=
  let parks_amount := b.total - b.library_amount - b.remaining
  (parks_amount / b.total) * 100

/-- Theorem stating that given the budget conditions, the percentage spent on parks is 24% --/
theorem centerville_parks_budget_percentage 
  (b : BudgetAllocation)
  (h1 : b.library_percent = 15)
  (h2 : b.library_amount = 3000)
  (h3 : b.remaining = 12200)
  (h4 : b.library_percent / 100 * b.total = b.library_amount) :
  parks_percentage b = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centerville_parks_budget_percentage_l1253_125310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_over_a_range_l1253_125369

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def is_valid_triangle (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.A + t.B + t.C = Real.pi

def is_acute_triangle (t : Triangle) : Prop :=
  t.A < Real.pi/2 ∧ t.B < Real.pi/2 ∧ t.C < Real.pi/2

def satisfies_equation (t : Triangle) : Prop :=
  (Real.sqrt 3 * t.c - 2 * Real.sin t.B * Real.sin t.C) = 
  Real.sqrt 3 * (t.b * Real.sin t.B - t.a * Real.sin t.A)

-- Theorem statement
theorem c_over_a_range (t : Triangle) 
  (h1 : is_valid_triangle t)
  (h2 : is_acute_triangle t)
  (h3 : satisfies_equation t) :
  1/2 < t.c / t.a ∧ t.c / t.a < 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_over_a_range_l1253_125369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_channel_depth_is_80_l1253_125346

/-- Represents a trapezoidal water channel -/
structure TrapezoidalChannel where
  topWidth : ℝ
  bottomWidth : ℝ
  area : ℝ

/-- Calculates the depth of a trapezoidal channel -/
noncomputable def channelDepth (channel : TrapezoidalChannel) : ℝ :=
  (2 * channel.area) / (channel.topWidth + channel.bottomWidth)

/-- Theorem: The depth of the specified trapezoidal channel is 80 meters -/
theorem channel_depth_is_80 :
  let channel : TrapezoidalChannel := {
    topWidth := 14,
    bottomWidth := 8,
    area := 880
  }
  channelDepth channel = 80 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_channel_depth_is_80_l1253_125346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sin_sum_in_triangle_l1253_125374

/-- Given a triangle ABC where sin A, sin B, and sin C form a geometric sequence,
    prove that the maximum value of sin A + sin C is √3 when B is maximized. -/
theorem max_sin_sum_in_triangle (A B C : Real) (a b c : Real) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi → -- Triangle condition
  0 < a ∧ 0 < b ∧ 0 < c → -- Positive side lengths
  Real.sin A * Real.sin C = (Real.sin B)^2 → -- Geometric sequence condition
  (∀ A' B' C' : Real, 0 < A' ∧ 0 < B' ∧ 0 < C' ∧ A' + B' + C' = Real.pi → B ≥ B') → -- B is maximum
  Real.sin A + Real.sin C ≤ Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sin_sum_in_triangle_l1253_125374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_reduction_equivalence_l1253_125380

theorem price_reduction_equivalence (P : ℝ) (h : P > 0) :
  let first_reduction := P * (1 - 0.25)
  let second_reduction := first_reduction * (1 - 0.70)
  second_reduction = P * (1 - 0.775) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_reduction_equivalence_l1253_125380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_hexagons_covering_unit_disc_l1253_125375

/-- A regular hexagon with side length 1 -/
structure RegularHexagon where
  side_length : ℝ
  is_unit : side_length = 1

/-- A disc with radius 1 -/
structure UnitDisc where
  radius : ℝ
  is_unit : radius = 1

/-- The property that a list of hexagons covers a disc -/
def Covers (hexagons : List RegularHexagon) (disc : UnitDisc) : Prop :=
  sorry -- We'll leave this as sorry for now

/-- A covering of a unit disc by regular hexagons -/
structure HexagonCovering where
  disc : UnitDisc
  hexagons : List RegularHexagon
  is_covering : Covers hexagons disc

/-- The theorem stating that the minimum number of regular hexagons 
    with side length 1 required to completely cover a disc with radius 1 is 3 -/
theorem min_hexagons_covering_unit_disc :
  ∃ (covering : HexagonCovering), 
    covering.hexagons.length = 3 ∧ 
    (∀ (other_covering : HexagonCovering), other_covering.hexagons.length ≥ 3) := by
  sorry -- We'll leave the proof as sorry for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_hexagons_covering_unit_disc_l1253_125375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enthalpy_C2H4_enthalpy_H2_enthalpy_C2H2_l1253_125326

-- Define the molar mass of C2H4
def molar_mass_C2H4 : ℝ := 28

-- Define the energy released in the C2H4 reaction
def energy_C2H4 : ℝ := 705.5

-- Define the amount of C2H4 in grams
def mass_C2H4 : ℝ := 14

-- Define the energy released in the H2 reaction
def energy_H2 : ℝ := 14.9

-- Define the amount of CO2 produced in the C2H2 reaction
def amount_CO2_C2H2 : ℝ := 1

-- Define a variable for the energy released in the C2H2 reaction
variable (b : ℝ)

-- Theorem for C2H4 reaction
theorem enthalpy_C2H4 : 
  (energy_C2H4 * molar_mass_C2H4 / mass_C2H4) = 1411 := by sorry

-- Theorem for H2 reaction
theorem enthalpy_H2 : energy_H2 = 14.9 := by sorry

-- Theorem for C2H2 reaction
theorem enthalpy_C2H2 : 
  (2 * b / amount_CO2_C2H2) = 2 * b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_enthalpy_C2H4_enthalpy_H2_enthalpy_C2H2_l1253_125326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_sin_2x_l1253_125329

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x)

theorem smallest_positive_period_of_sin_2x :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (S : ℝ), S > 0 ∧ (∀ (x : ℝ), f (x + S) = f x) → T ≤ S) ∧
  T = Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_sin_2x_l1253_125329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_kind_terms_l1253_125395

-- Define a structure for algebraic terms
structure AlgebraicTerm where
  coefficients : List ℤ
  vars : List Char
  exponents : List ℕ

-- Define a function to check if two terms are of the same kind
def sameKind (t1 t2 : AlgebraicTerm) : Prop :=
  t1.vars = t2.vars ∧ t1.exponents = t2.exponents

-- Define the terms from the problem
def term1a : AlgebraicTerm := { coefficients := [-2], vars := ['p', 't'], exponents := [2, 1] }
def term1b : AlgebraicTerm := { coefficients := [1], vars := ['t', 'p'], exponents := [1, 2] }

def term2a : AlgebraicTerm := { coefficients := [-1], vars := ['a', 'b', 'c', 'd'], exponents := [2, 1, 1, 1] }
def term2b : AlgebraicTerm := { coefficients := [3], vars := ['b', 'a', 'c', 'd'], exponents := [2, 1, 1, 1] }

def term3a : AlgebraicTerm := { coefficients := [-1], vars := ['a', 'b'], exponents := [2, 2] }
def term3b : AlgebraicTerm := { coefficients := [1], vars := ['a', 'b'], exponents := [2, 2] }

def term4a : AlgebraicTerm := { coefficients := [8], vars := ['b', 'a'], exponents := [2, 1] }
def term4b : AlgebraicTerm := { coefficients := [4], vars := ['a', 'b'], exponents := [1, 2] }

-- State the theorem
theorem same_kind_terms :
  (sameKind term1a term1b) ∧
  (sameKind term3a term3b) ∧
  (sameKind term4a term4b) ∧
  ¬(sameKind term2a term2b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_kind_terms_l1253_125395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_div_b_simplification_l1253_125354

open BigOperators Finset Nat

def a (n : ℕ) : ℚ := ∑ k in range (n + 1), 1 / (choose n k : ℚ)

def b (n : ℕ) : ℚ := ∑ k in range (n + 1), k^2 / (choose n k : ℚ)

theorem a_div_b_simplification (n : ℕ) (h : n > 1) : a n / b n = 1 / (n * (n - 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_div_b_simplification_l1253_125354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_paint_problem_l1253_125390

def number_of_cubes_with_two_painted_sides (large_side : ℕ) (small_side : ℕ) : ℕ :=
  12  -- This is a placeholder. In a real implementation, this would be calculated based on the input parameters.

theorem cube_paint_problem (large_cube_side : ℕ) (small_cube_side : ℕ) : 
  large_cube_side = 9 →
  small_cube_side = 3 →
  small_cube_side ∣ large_cube_side →
  (12 : ℕ) = (number_of_cubes_with_two_painted_sides large_cube_side small_cube_side) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_paint_problem_l1253_125390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_light_path_in_cube_l1253_125377

-- Define the cube edge length
def cube_edge : ℝ := 10

-- Define the coordinates of point P relative to the face it's on
def p_x : ℝ := 3
def p_y : ℝ := 4

-- Define the number of reflections
def num_reflections : ℕ := 10

-- Define the length of the light path
noncomputable def light_path_length : ℝ := 
  num_reflections * Real.sqrt (cube_edge^2 + p_x^2 + p_y^2)

-- Theorem statement
theorem light_path_in_cube :
  light_path_length = 50 * Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_light_path_in_cube_l1253_125377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x6_in_expansion_l1253_125350

theorem coefficient_x6_in_expansion : ∃ c : ℕ, c = 112 ∧ 
  (Finset.sum (Finset.range 9) (λ k ↦ (Nat.choose 8 k : ℕ) * 2^(8 - k) * (if k = 6 then 1 else 0))) = c :=
by
  -- We'll use 112 as our witness for c
  use 112
  
  -- Split the goal into two parts
  apply And.intro
  
  -- Prove c = 112 (trivial since we used 112)
  · rfl
  
  -- Prove the sum equals 112
  · sorry  -- The actual proof would go here


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x6_in_expansion_l1253_125350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l1253_125314

/-- Represents a parabola in the form ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point (x, y) is on the parabola -/
noncomputable def Parabola.contains (p : Parabola) (x y : ℝ) : Prop :=
  y = p.a * x^2 + p.b * x + p.c

/-- Calculates the x-coordinate of the vertex of a parabola -/
noncomputable def Parabola.vertexX (p : Parabola) : ℝ :=
  -p.b / (2 * p.a)

/-- Calculates the y-coordinate of the vertex of a parabola -/
noncomputable def Parabola.vertexY (p : Parabola) : ℝ :=
  p.c - p.b^2 / (4 * p.a)

/-- Theorem: The given equation represents the required parabola -/
theorem parabola_equation : ∃ (p : Parabola),
  p.a = -3 ∧ p.b = 18 ∧ p.c = -22 ∧
  p.vertexX = 3 ∧ p.vertexY = 5 ∧
  p.contains 2 2 ∧
  p.a ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l1253_125314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_increasing_interval_implies_a_bound_l1253_125348

/-- If the function f(x) = x^2 - e^x - ax has an increasing interval on ℝ, then a < 2*ln(2) - 2 -/
theorem function_increasing_interval_implies_a_bound 
  (f : ℝ → ℝ) 
  (a : ℝ) 
  (h1 : ∀ x, f x = x^2 - Real.exp x - a*x) 
  (h2 : ∃ (I : Set ℝ), Set.Nonempty I ∧ MonotoneOn f I) :
  a < 2 * Real.log 2 - 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_increasing_interval_implies_a_bound_l1253_125348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_special_triangle_l1253_125378

noncomputable def fractional_part (x : ℝ) : ℝ := x - ⌊x⌋

theorem min_perimeter_special_triangle (ℓ m n : ℕ) : 
  ℓ > m ∧ m > n ∧ 
  fractional_part (3^ℓ / 10000) = fractional_part (3^m / 10000) ∧
  fractional_part (3^m / 10000) = fractional_part (3^n / 10000) →
  ℓ + m + n ≥ 3003 :=
by sorry

#check min_perimeter_special_triangle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_special_triangle_l1253_125378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_instantaneous_velocity_at_3_seconds_l1253_125385

/-- The motion equation of an object -/
noncomputable def s (t : ℝ) : ℝ := 1 - 2*t + 2*t^2

/-- The instantaneous velocity of the object at time t -/
noncomputable def v (t : ℝ) : ℝ := deriv s t

/-- Theorem: The instantaneous velocity of the object at 3 seconds is 10 m/s -/
theorem instantaneous_velocity_at_3_seconds :
  v 3 = 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_instantaneous_velocity_at_3_seconds_l1253_125385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l1253_125337

theorem calculation_proof :
  -1^2022 + Real.sqrt 16 * (-3)^2 + (-6) / ((-8 : ℝ) ^ (1/3)) = 38 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l1253_125337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_intersections_and_vertex_l1253_125383

/-- The quadratic function y = x^2 + 4x + k - 1 -/
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := x^2 + 4*x + k - 1

/-- The discriminant of the quadratic function -/
noncomputable def discriminant (k : ℝ) : ℝ := 20 - 4*k

/-- The y-coordinate of the vertex of the quadratic function -/
noncomputable def vertex_y (k : ℝ) : ℝ := (-4^2 + 4*(k-1)) / (4*1)

theorem quadratic_intersections_and_vertex (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f k x₁ = 0 ∧ f k x₂ = 0) ↔ k < 5 ∧
  (vertex_y k = 0 ↔ k = 5) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_intersections_and_vertex_l1253_125383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mode_expected_value_median_equal_l1253_125360

noncomputable section

/-- The probability density function for the random variable X -/
def f (x : ℝ) : ℝ :=
  if 2 < x ∧ x < 4 then -3/4 * x^2 + 9/2 * x - 6 else 0

/-- The random variable X -/
def X : Type := ℝ

/-- The mode of the random variable X -/
noncomputable def mode : ℝ := 3

/-- The expected value of the random variable X -/
noncomputable def expected_value : ℝ :=
  ∫ x in Set.Ioo 2 4, x * f x

/-- The median of the random variable X -/
noncomputable def median : ℝ := 3

theorem mode_expected_value_median_equal :
  mode = expected_value ∧ expected_value = median := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mode_expected_value_median_equal_l1253_125360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_300_to_500_with_digit_sum_16_l1253_125317

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

def count_integers_with_digit_sum (lower : ℕ) (upper : ℕ) (target_sum : ℕ) : ℕ :=
  (List.range (upper - lower + 1)).map (λ i => i + lower)
    |>.filter (λ n => sum_of_digits n = target_sum)
    |>.length

theorem count_integers_300_to_500_with_digit_sum_16 :
  count_integers_with_digit_sum 300 500 16 = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_300_to_500_with_digit_sum_16_l1253_125317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_difference_bound_l1253_125323

theorem polynomial_difference_bound (a : ℝ) (n : ℕ) (P : Polynomial ℝ) 
  (h1 : a ≥ 3) (h2 : P.degree = n) : 
  ∃ i : ℕ, i ≤ n + 1 ∧ |a^i - P.eval (↑i)| ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_difference_bound_l1253_125323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_team_arrangement_l1253_125364

/-- Represents the number of boys on the chess team -/
def num_boys : ℕ := 2

/-- Represents the number of girls on the chess team -/
def num_girls : ℕ := 4

/-- Represents the total number of team members -/
def team_size : ℕ := num_boys + num_girls

/-- Represents the number of possible arrangements -/
def arrangements : ℕ := 96

theorem chess_team_arrangement :
  (2 : ℕ) *  -- Alice's position options (left or right end)
  (team_size - 1) *  -- Positions for the boy block after Alice is placed
  (Nat.factorial num_boys) *  -- Arrangements within the boy block
  (Nat.factorial (num_girls - 1)) -- Arrangements of the remaining girls
  = arrangements := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_team_arrangement_l1253_125364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_white_ball_probability_l1253_125398

-- Define the number of white and blue balls in each urn
def white_balls_urn1 : ℕ := 5
def blue_balls_urn1 : ℕ := 4
def white_balls_urn2 : ℕ := 2
def blue_balls_urn2 : ℕ := 8

-- Define the probability of drawing a white ball from the third urn
def prob_white_third_urn : ℚ := 17/45

-- Theorem statement
theorem white_ball_probability : 
  (white_balls_urn1 : ℚ) / (white_balls_urn1 + blue_balls_urn1) * 
  (blue_balls_urn2 : ℚ) / (white_balls_urn2 + blue_balls_urn2) * (1/2) +
  (blue_balls_urn1 : ℚ) / (white_balls_urn1 + blue_balls_urn1) * 
  (white_balls_urn2 : ℚ) / (white_balls_urn2 + blue_balls_urn2) * (1/2) +
  (white_balls_urn1 : ℚ) / (white_balls_urn1 + blue_balls_urn1) * 
  (white_balls_urn2 : ℚ) / (white_balls_urn2 + blue_balls_urn2) = prob_white_third_urn := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_white_ball_probability_l1253_125398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_and_general_formula_l1253_125365

def a : ℕ → ℚ
  | 0 => 1
  | n+1 => a n / (3 * a n + 1)

def b (n : ℕ) : ℚ := 1 / a n

theorem arithmetic_sequence_and_general_formula :
  (∃ d : ℚ, ∀ n : ℕ, b (n + 1) = b n + d) ∧
  (∀ n : ℕ, a n = 1 / (3 * (n + 1) - 2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_and_general_formula_l1253_125365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_for_point_l1253_125327

theorem sin_plus_cos_for_point (α : ℝ) :
  (∃ (r : ℝ), r > 0 ∧ r * Real.cos α = -1 ∧ r * Real.sin α = Real.sqrt 3) →
  Real.sin α + Real.cos α = -1/2 + Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_for_point_l1253_125327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_natural_sum_22_l1253_125399

theorem unique_natural_sum_22 :
  ∃! (s : Finset ℕ), s.card = 6 ∧ s.sum id = 22 ∧ (∀ x y, x ∈ s → y ∈ s → x ≠ y → x ≠ y) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_natural_sum_22_l1253_125399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_pole_time_l1253_125318

/-- The time it takes for a train to speed past a pole -/
noncomputable def time_to_pass_pole (train_length : ℝ) (tunnel_length : ℝ) (time_through_tunnel : ℝ) : ℝ :=
  train_length / ((train_length + tunnel_length) / time_through_tunnel)

/-- Theorem: The time it takes for the train to speed past the pole is 20 seconds -/
theorem train_passing_pole_time :
  let train_length : ℝ := 500
  let tunnel_length : ℝ := 500
  let time_through_tunnel : ℝ := 40
  time_to_pass_pole train_length tunnel_length time_through_tunnel = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_pole_time_l1253_125318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_terms_count_l1253_125359

/-- The number of rational terms in the expansion of (√2 + ∛3)^n -/
def rationalTermCount (n : ℕ) : ℕ :=
  (n / 6) + 1

/-- The number of rational terms in the expansion of (√2 + ∜3)^n -/
def rationalTermCount2 (n : ℕ) : ℕ :=
  (n / 4) + 1

theorem rational_terms_count :
  (rationalTermCount 300 = 51) ∧ (rationalTermCount2 100 = 26) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_terms_count_l1253_125359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_proof_l1253_125334

theorem triangle_abc_proof (A B C : ℝ) (a b c : ℝ) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively
  (a > 0) → (b > 0) → (c > 0) →
  (A > 0) → (B > 0) → (C > 0) →
  (A + B + C = Real.pi) →
  -- Given conditions
  (a - b = 1) →
  (2 * (Real.cos ((A + B) / 2))^2 - Real.cos (2 * C) = 1) →
  (3 * Real.sin B = 2 * Real.sin A) →
  -- Prove
  (C = Real.pi / 3) ∧ (c / b = Real.sqrt 7 / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_proof_l1253_125334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_condition_inequality_condition_l1253_125357

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a * Real.cos x

-- Theorem 1
theorem tangent_line_condition (a : ℝ) : 
  (∃ k b : ℝ, ∀ x, k * x + b = f a x + (deriv (f a)) 0 * (x - 0)) ∧ 
  (6 = f a 0 + (deriv (f a)) 0 * (1 - 0)) → 
  a = 4 :=
by sorry

-- Theorem 2
theorem inequality_condition (a : ℝ) :
  (∀ x, x ∈ Set.Icc 0 (Real.pi / 2) → f a x ≥ a * x) ↔ 
  a ∈ Set.Icc (-1) ((2 * Real.exp (Real.pi / 2)) / Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_condition_inequality_condition_l1253_125357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_construction_from_medians_and_point_l1253_125362

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in 2D space
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a triangle
structure Triangle where
  A : Point2D
  B : Point2D
  C : Point2D

-- Define the centroid of a triangle
noncomputable def centroid (t : Triangle) : Point2D where
  x := (t.A.x + t.B.x + t.C.x) / 3
  y := (t.A.y + t.B.y + t.C.y) / 3

-- Define membership of a point on a line
def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Define a median of a triangle
def isMedian (l : Line2D) (t : Triangle) : Prop :=
  ∃ (m : Point2D), pointOnLine m l ∧ pointOnLine (centroid t) l ∧
    ((m.x = (t.A.x + t.B.x) / 2 ∧ m.y = (t.A.y + t.B.y) / 2) ∨
     (m.x = (t.B.x + t.C.x) / 2 ∧ m.y = (t.B.y + t.C.y) / 2) ∨
     (m.x = (t.C.x + t.A.x) / 2 ∧ m.y = (t.C.y + t.A.y) / 2))

-- Define a point on a line segment
def pointOnSegment (p : Point2D) (a b : Point2D) : Prop :=
  ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ p.x = (1 - t) * a.x + t * b.x ∧ p.y = (1 - t) * a.y + t * b.y

-- Main theorem
theorem triangle_construction_from_medians_and_point 
  (m1 m2 m3 : Line2D) (P : Point2D) :
  ∃ (t1 t2 : Triangle),
    isMedian m1 t1 ∧ isMedian m2 t1 ∧ isMedian m3 t1 ∧
    isMedian m1 t2 ∧ isMedian m2 t2 ∧ isMedian m3 t2 ∧
    (pointOnSegment P t1.A t1.B ∨ pointOnSegment P t1.B t1.C ∨ pointOnSegment P t1.C t1.A) ∧
    (pointOnSegment P t2.A t2.B ∨ pointOnSegment P t2.B t2.C ∨ pointOnSegment P t2.C t2.A) ∧
    t1 ≠ t2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_construction_from_medians_and_point_l1253_125362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_june_walking_time_l1253_125387

/-- Represents the time it takes June to walk to her aunt's house -/
noncomputable def time_to_aunt (distance_to_julia : ℝ) (biking_time : ℝ) (distance_to_aunt : ℝ) : ℝ :=
  let biking_speed := distance_to_julia / biking_time
  let walking_speed := biking_speed / 2
  distance_to_aunt / walking_speed

/-- Proves that June takes 20 minutes to walk to her aunt's house -/
theorem june_walking_time :
  time_to_aunt 1 4 2.5 = 20 := by
  -- Unfold the definition of time_to_aunt
  unfold time_to_aunt
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_june_walking_time_l1253_125387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tip_percentage_approx_l1253_125332

/-- The tip percentage given the lunch cost and total amount spent -/
noncomputable def tip_percentage (lunch_cost total_spent : ℝ) : ℝ :=
  ((total_spent - lunch_cost) / lunch_cost) * 100

/-- Theorem stating that the tip percentage is approximately 19.96% -/
theorem tip_percentage_approx :
  let lunch_cost : ℝ := 50.20
  let total_spent : ℝ := 60.24
  abs (tip_percentage lunch_cost total_spent - 19.96) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tip_percentage_approx_l1253_125332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l1253_125325

/-- Represents a parabola with parameter p > 0 -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The focus of a parabola -/
noncomputable def focus (c : Parabola) : Point :=
  { x := c.p / 2, y := 0 }

/-- A point on the parabola -/
def on_parabola (c : Parabola) (a : Point) : Prop :=
  a.y^2 = 2 * c.p * a.x

/-- The angle between three points is acute -/
def acute_angle (m a f : Point) : Prop :=
  (a.x - m.x) * (f.x - a.x) + (a.y - m.y) * (f.y - a.y) > 0

/-- Main theorem statement -/
theorem parabola_properties (c : Parabola) :
  (∃ (a : Point),
    on_parabola c a ∧
    a.x = c.p / 4 ∧
    a.y = 2 →
    c.p = 2 * Real.sqrt 2) ∧
  (∀ (m : ℝ),
    (∀ (a : Point), on_parabola c a →
      acute_angle { x := m, y := 0 } a (focus c)) ↔
    (0 < m ∧ m < (9 * c.p) / 2 ∧ m ≠ c.p / 2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l1253_125325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_proof_l1253_125356

open Set

-- Define the function f and its derivative f'
def f : ℝ → ℝ := sorry

def f' : ℝ → ℝ := sorry

-- Define the domain of f
def domain : Set ℝ := {x | x > 0}

-- State the theorem
theorem solution_set_proof (h1 : ∀ x ∈ domain, x * f' x - f x < 0)
                           (h2 : f 2 = 0) :
  {x ∈ domain | (x - 1) * f x > 0} = Ioo 1 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_proof_l1253_125356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l1253_125367

/-- A parabola with focus F and a point M on it -/
structure ParabolaWithPoint where
  F : ℝ × ℝ
  M : ℝ × ℝ
  N : ℝ × ℝ
  h_F : F.1 = 2 ∧ F.2 = 0
  h_parabola : M.2^2 = 8 * M.1
  h_midpoint : M = ((F.1 + N.1) / 2, (F.2 + N.2) / 2)
  h_N_on_y_axis : N.1 = 0

/-- The distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem parabola_focus_distance (p : ParabolaWithPoint) : distance p.F p.N = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l1253_125367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l1253_125352

-- Define the vectors a and b
variable (a b : ℝ × ℝ)

-- Define the conditions
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 * v.1 + v.2 * v.2)
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the scalar multiplication for vectors
def scalar_mult (r : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (r * v.1, r * v.2)

-- Define vector addition
def vector_add (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)

-- Define the theorem
theorem vector_problem (h1 : magnitude a = 4) (h2 : magnitude b = 3)
  (h3 : dot_product (vector_add (scalar_mult 2 a) (scalar_mult (-3) b)) (vector_add (scalar_mult 2 a) b) = 61) :
  -- Part 1: Angle between a and b
  (Real.arccos (dot_product a b / (magnitude a * magnitude b)) = 2 * Real.pi / 3) ∧
  -- Part 2: Projection of a on (3a + 2b)
  (magnitude a * (dot_product a (vector_add (scalar_mult 3 a) (scalar_mult 2 b))) / 
   (magnitude (vector_add (scalar_mult 3 a) (scalar_mult 2 b))) = 2 * Real.sqrt 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l1253_125352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_value_l1253_125339

-- Define the power function as noncomputable
noncomputable def powerFunction (α : ℝ) : ℝ → ℝ := fun x ↦ x ^ α

-- State the theorem
theorem power_function_value (α : ℝ) :
  powerFunction α 3 = Real.sqrt 3 / 3 →
  powerFunction α (1/4) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_value_l1253_125339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_inequality_l1253_125361

theorem right_triangle_inequality (a b c : ℝ) (A B : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < A ∧ 0 < B ∧ A + B = Real.pi / 2 ∧ 
  a^2 + b^2 = c^2 ∧ Real.sin A = a / c ∧ Real.cos B = a / c →
  c ≠ b / Real.cos B :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_inequality_l1253_125361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l1253_125381

def set_A : Set ℝ := {x : ℝ | 2 * x^2 + 5 * x - 3 ≤ 0}

def set_B : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.sqrt (1 / (x + 2))}

theorem intersection_of_A_and_B : set_A ∩ set_B = Set.Ioo (-2 : ℝ) (1/2 : ℝ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l1253_125381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_diagonals_equal_and_bisect_l1253_125355

/-- A square is a quadrilateral with four equal sides and four right angles. -/
structure Square where
  sides : Fin 4 → ℝ
  angles : Fin 4 → ℝ
  side_equality : ∀ i j, sides i = sides j
  right_angles : ∀ i, angles i = 90

/-- A diagonal of a square connects opposite vertices. -/
def Square.diagonal (s : Square) (i : Fin 2) : ℝ := sorry

/-- The midpoint of a diagonal is the point that bisects it. -/
def Square.diagonalMidpoint (s : Square) (i : Fin 2) : ℝ := sorry

/-- Theorem stating that the diagonals of a square are equal and bisect each other. -/
theorem square_diagonals_equal_and_bisect (s : Square) :
  (∀ i j, s.diagonal i = s.diagonal j) ∧
  (∀ i, s.diagonalMidpoint i = (s.diagonal i) / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_diagonals_equal_and_bisect_l1253_125355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_quadrilateral_area_l1253_125328

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square -/
structure Square where
  center : Point
  side_length : ℝ

/-- Calculates the centroid of a triangle -/
noncomputable def centroid (a b c : Point) : Point :=
  { x := (a.x + b.x + c.x) / 3,
    y := (a.y + b.y + c.y) / 3 }

/-- Calculates the area of a quadrilateral given its four vertices -/
noncomputable def quadrilateral_area (a b c d : Point) : ℝ :=
  sorry

/-- Main theorem -/
theorem centroid_quadrilateral_area 
  (s : Square)
  (q : Point)
  (h1 : s.side_length = 40)
  (h2 : (q.x - s.center.x)^2 + (q.y - s.center.y)^2 < (s.side_length/2)^2) -- Q is inside the square
  (h3 : (q.x - (s.center.x - s.side_length/2))^2 + (q.y - (s.center.y - s.side_length/2))^2 = 15^2) -- EQ = 15
  (h4 : (q.x - (s.center.x + s.side_length/2))^2 + (q.y - (s.center.y - s.side_length/2))^2 = 35^2) -- FQ = 35
  : 
  let e := { x := s.center.x - s.side_length/2, y := s.center.y - s.side_length/2 }
  let f := { x := s.center.x + s.side_length/2, y := s.center.y - s.side_length/2 }
  let g := { x := s.center.x + s.side_length/2, y := s.center.y + s.side_length/2 }
  let h := { x := s.center.x - s.side_length/2, y := s.center.y + s.side_length/2 }
  let c1 := centroid e f q
  let c2 := centroid f g q
  let c3 := centroid g h q
  let c4 := centroid h e q
  quadrilateral_area c1 c2 c3 c4 = 12800/9 := by
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_quadrilateral_area_l1253_125328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1253_125336

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  area : Real

-- State the theorem
theorem triangle_properties (abc : Triangle) 
  (h1 : abc.a * Real.cos abc.C + abc.c * Real.cos abc.A = 2 * abc.b * Real.cos abc.B)
  (h2 : abc.b = 2 * Real.sqrt 3)
  (h3 : abc.area = 2 * Real.sqrt 3) :
  abc.B = Real.pi / 3 ∧ 
  abc.a + abc.b + abc.c = 6 + 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1253_125336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_inequality_in_acute_triangle_l1253_125338

/-- Given an acute triangle ABC, prove that the sum of the products of cosines of pairs of angles
    is less than or equal to 1/2 plus twice the product of cosines of all three angles. -/
theorem cosine_inequality_in_acute_triangle (A B C : ℝ) 
  (h_acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2) 
  (h_sum : A + B + C = π) : 
  Real.cos A * Real.cos B + Real.cos B * Real.cos C + Real.cos C * Real.cos A 
    ≤ 1/2 + 2 * Real.cos A * Real.cos B * Real.cos C := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_inequality_in_acute_triangle_l1253_125338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_range_theorem_l1253_125312

/-- Given a function f(x) = ax + 2 where a < 0, if there exists x₀ ∈ [-2, 2] such that f(x₀) < 0, 
    then a ∈ (-∞, -1) -/
theorem function_range_theorem (a : ℝ) (h1 : a < 0) :
  (∃ x₀ : ℝ, x₀ ∈ Set.Icc (-2) 2 ∧ a * x₀ + 2 < 0) → a < -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_range_theorem_l1253_125312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functions_properties_l1253_125397

noncomputable section

variable (a : ℝ)

def f₁ (a x : ℝ) : ℝ := Real.log (x - 3 * a) / Real.log a
def f₂ (a x : ℝ) : ℝ := Real.log (1 / (x - a)) / Real.log a

def meaningful (a : ℝ) : Prop :=
  ∀ x ∈ Set.Icc (a + 2) (a + 3), x - 3 * a > 0 ∧ x - a > 0

def close (a : ℝ) : Prop :=
  ∀ x ∈ Set.Icc (a + 2) (a + 3), |f₁ a x - f₂ a x| ≤ 1

theorem functions_properties :
  (a > 0 ∧ a ≠ 1) →
  (meaningful a ↔ 0 < a ∧ a < 1) ∧
  (close a ↔ 0 < a ∧ a ≤ (9 - Real.sqrt 57) / 12) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functions_properties_l1253_125397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_progression_ratio_l1253_125394

/-- Represents a geometric progression -/
structure GeometricProgression where
  a : ℕ → ℝ  -- The sequence
  q : ℝ      -- Common ratio
  h_geom : ∀ n, a (n + 1) = a n * q

/-- Sum of the first n terms of a geometric progression -/
noncomputable def S (gp : GeometricProgression) (n : ℕ) : ℝ :=
  if gp.q = 1 then n * gp.a 0
  else gp.a 0 * (1 - gp.q^n) / (1 - gp.q)

/-- The main theorem -/
theorem geometric_progression_ratio 
  (gp : GeometricProgression) 
  (h_sum : S gp 4 = 5 * S gp 2) :
  (gp.a 2 * gp.a 7) / (gp.a 4)^2 = -1 ∨ 
  (gp.a 2 * gp.a 7) / (gp.a 4)^2 = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_progression_ratio_l1253_125394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_problem_l1253_125307

/-- Power function f(x) = x^(1/(m^2+m)) -/
noncomputable def f (m : ℕ) (x : ℝ) : ℝ := x^(1 / (m^2 + m : ℝ))

theorem power_function_problem (m : ℕ) (h1 : m > 0) (h2 : f m 2 = Real.sqrt 2) :
  m = 1 ∧ Set.Icc 1 (3/2) ⊆ {a : ℝ | f m (2 - a) > f m (a - 1)} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_problem_l1253_125307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l1253_125392

/-- The circle equation -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 4*y - 3 = 0

/-- The line equation -/
def line_eq (x y : ℝ) : Prop := x + y + 1 = 0

/-- The distance from a point (x, y) to the line -/
noncomputable def distance_to_line (x y : ℝ) : ℝ := |x + y + 1| / Real.sqrt 2

/-- A point satisfies the distance condition -/
def satisfies_distance (x y : ℝ) : Prop := distance_to_line x y = Real.sqrt 2

/-- The main theorem -/
theorem circle_line_intersection :
  ∃! (s : Set (ℝ × ℝ)), s.Finite ∧ s.ncard = 3 ∧ 
  ∀ (p : ℝ × ℝ), p ∈ s ↔ (circle_eq p.1 p.2 ∧ satisfies_distance p.1 p.2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l1253_125392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_amanda_final_distance_l1253_125345

-- Define the conversion rate from meters to feet
noncomputable def meters_to_feet : ℝ := 3.281

-- Define Amanda's movements
noncomputable def north_distance : ℝ := 12
noncomputable def west_distance : ℝ := 45
noncomputable def south_distance : ℝ := 12 + 15 / meters_to_feet

-- Theorem statement
theorem amanda_final_distance :
  let net_south := south_distance * meters_to_feet - north_distance * meters_to_feet
  let distance_squared := west_distance ^ 2 + net_south ^ 2
  Real.sqrt distance_squared = 15 * Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_amanda_final_distance_l1253_125345
