import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_weight_calculation_l356_35667

/-- The weight of one apple in grams -/
def apple_weight : ℕ := 205

/-- The number of apples in each small box -/
def apples_per_small_box : ℕ := 6

/-- The weight of each small box in grams -/
def small_box_weight : ℕ := 220

/-- The number of small boxes in the large box -/
def small_boxes_per_large_box : ℕ := 9

/-- The weight of the large box in grams -/
def large_box_weight : ℕ := 250

/-- The total weight in kilograms -/
def total_weight_kg : ℚ := 13.3

theorem apple_weight_calculation :
  (apple_weight * (apples_per_small_box * small_boxes_per_large_box) +
  small_box_weight * small_boxes_per_large_box +
  large_box_weight : ℕ) = (total_weight_kg * 1000).floor :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_weight_calculation_l356_35667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_size_comparison_l356_35639

-- Define the three numbers
noncomputable def a : ℝ := (7 : ℝ) ^ (3/10 : ℝ)
noncomputable def b : ℝ := (3/10 : ℝ) ^ (7 : ℝ)
noncomputable def c : ℝ := Real.log (3/10 : ℝ)

-- State the theorem
theorem size_comparison : a > b ∧ b > c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_size_comparison_l356_35639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_axis_of_symmetry_f_zero_sum_of_coordinates_l356_35699

-- Define the functions f and g
def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

-- Define the properties of f and g
axiom f_symmetry (x : ℝ) : f (2 - x) = f x
axiom f_value_at_1 : f 1 = 2
axiom f_odd (x : ℝ) : f (3 * x + 2) = -f (-3 * x - 2)
axiom g_symmetry (x : ℝ) : g x = -g (4 - x)

-- Define the intersection points
def intersection_points : Finset (ℝ × ℝ) := sorry
axiom intersection_count : intersection_points.card = 2023

-- Theorem statements
theorem f_axis_of_symmetry : ∀ x, f (2 - x) = f x := by
  intro x
  exact f_symmetry x

theorem f_zero : f 0 = 0 := by sorry

theorem sum_of_coordinates : 
  (intersection_points.sum (fun p ↦ p.1 + p.2)) = 4046 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_axis_of_symmetry_f_zero_sum_of_coordinates_l356_35699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_undefined_value_l356_35635

-- Define the quadratic function in the denominator
def f (x : ℝ) := 4 * x^2 - 81 * x + 49

-- Define the larger root of the quadratic equation
noncomputable def larger_root : ℝ := (81 + Real.sqrt (81^2 - 4 * 4 * 49)) / (2 * 4)

-- Theorem statement
theorem largest_undefined_value :
  ∀ x : ℝ, f x = 0 → x ≤ larger_root :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_undefined_value_l356_35635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_satisfying_inequality_l356_35673

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := -2 * x + Real.sin x

-- State the theorem
theorem range_of_m_satisfying_inequality :
  {m : ℝ | f (2 * m^2 - m + Real.pi - 1) ≥ -2 * Real.pi} = Set.Icc (-1/2) 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_satisfying_inequality_l356_35673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bxd_measure_l356_35631

/-- A rectangle with a specific point inside --/
structure SpecialRectangle where
  /-- The length of side AB --/
  a : ℝ
  /-- The length of side BC --/
  b : ℝ
  /-- The ratio of BC to AB is √2 --/
  ratio : b = a * Real.sqrt 2
  /-- Point X inside the rectangle --/
  x : ℝ × ℝ
  /-- AB = BX = XD --/
  equal_segments : 
    a = Real.sqrt ((x.1 - a)^2 + x.2^2) ∧
    a = Real.sqrt (x.1^2 + (x.2 - b)^2)

/-- The measure of angle BXD in a special rectangle is 2π/3 --/
theorem angle_bxd_measure (rect : SpecialRectangle) : 
  let angle_bxd := Real.arccos (-(1 / 2))
  angle_bxd = 2 * Real.pi / 3 := by
  sorry

#check angle_bxd_measure

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bxd_measure_l356_35631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_even_numbers_in_range_l356_35645

theorem count_even_numbers_in_range : 
  (Finset.filter (fun n => n % 2 = 0 && n > 300 && n ≤ 600) (Finset.range 601)).card = 150 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_even_numbers_in_range_l356_35645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_vector_difference_l356_35648

def vector_a (x y : ℝ) : ℝ × ℝ := (x, y)
def vector_b : ℝ × ℝ := (-1, 2)

theorem magnitude_of_vector_difference (x y : ℝ) 
  (h : vector_a x y + vector_b = (1, 3)) : 
  Real.sqrt ((x + 2)^2 + (y - 4)^2) = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_vector_difference_l356_35648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l356_35682

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  if x ≥ 0 then x^2 - (a + 4)*x + a else -(x^2 - (a + 4)*x + a)

theorem odd_function_properties (a : ℝ) :
  (∀ x, f a x = -f a (-x)) →
  (a = 0 ∧ 
   ∀ x, f a x = if x ≥ 0 then x^2 - 4*x else -x^2 - 4*x) :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l356_35682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_surface_area_l356_35649

-- Define the tetrahedron
structure Tetrahedron where
  A : ℝ  -- Angle A in degrees
  BC : ℝ -- Length of side BC
  PA : ℝ -- Length of PA

-- Define the properties of the tetrahedron
def tetrahedron_properties (t : Tetrahedron) : Prop :=
  t.A = 150 ∧ t.BC = 3 ∧ t.PA = 2

-- Define the function to calculate the surface area of the spherical blank
noncomputable def spherical_blank_surface_area (t : Tetrahedron) : ℝ :=
  40 * Real.pi

-- Theorem statement
theorem minimum_surface_area (t : Tetrahedron) 
  (h : tetrahedron_properties t) : 
  spherical_blank_surface_area t = 40 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_surface_area_l356_35649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l356_35668

noncomputable def geometric_sequence (q : ℝ) (a₁ : ℝ) (n : ℕ) : ℝ := a₁ * q ^ (n - 1)

noncomputable def geometric_sum (q : ℝ) (a₁ : ℝ) (n : ℕ) : ℝ := a₁ * (1 - q^n) / (1 - q)

theorem geometric_sequence_ratio (q : ℝ) (a₁ : ℝ) :
  (∃ n : ℕ, n > 0 ∧ 
    geometric_sequence q a₁ 5 - geometric_sequence q a₁ 3 = 12 ∧
    geometric_sequence q a₁ 6 - geometric_sequence q a₁ 4 = 24) →
  ∀ n : ℕ, n > 0 → geometric_sum q a₁ n / geometric_sequence q a₁ n = 2 - 2^(1-n) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l356_35668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_inequality_l356_35609

open Real

-- Define the functions f and g
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := 4 * log x - (1/2) * m * x^2

noncomputable def g (m : ℝ) (x : ℝ) : ℝ := f m x - (m - 4) * x

-- State the theorem
theorem slope_inequality {m x0 x1 x2 : ℝ} (hm : m > 0) (hx : x1 ≠ x2) :
  let k := (g m x1 - g m x2) / (x1 - x2)
  k = (deriv (g m)) x0 → x1 + x2 > 2 * x0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_inequality_l356_35609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_increase_effect_l356_35601

/-- Represents the percentage change in price, quantity, and revenue -/
structure PercentageChanges where
  price : ℚ
  quantity : ℚ
  revenue : ℚ

/-- Calculates the new revenue based on price and quantity changes -/
def calculateNewRevenue (changes : PercentageChanges) : ℚ :=
  (1 + changes.price / 100) * (1 + changes.quantity / 100) - 1

/-- Theorem stating that a 20% price increase and 20% quantity decrease result in a 4% revenue decrease -/
theorem price_increase_effect (changes : PercentageChanges) :
  changes.price = 20 ∧ changes.quantity = -20 → calculateNewRevenue changes = -4/100 := by
  sorry

#eval calculateNewRevenue { price := 20, quantity := -20, revenue := 0 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_increase_effect_l356_35601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slopes_form_geometric_sequence_l356_35618

-- Define the ellipse C
noncomputable def ellipse (x y : ℝ) (a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the eccentricity
noncomputable def eccentricity (a c : ℝ) : ℝ :=
  c / a

-- Define the focal length
noncomputable def focalLength (c : ℝ) : ℝ :=
  2 * c

-- Define a point on the ellipse
structure PointOnEllipse (a b : ℝ) where
  x : ℝ
  y : ℝ
  on_ellipse : ellipse x y a b

-- Define the slope of a line
noncomputable def lineSlope (x1 y1 x2 y2 : ℝ) : ℝ :=
  (y2 - y1) / (x2 - x1)

-- Main theorem
theorem slopes_form_geometric_sequence 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hab : a > b) 
  (hecc : eccentricity a (Real.sqrt 3) = Real.sqrt 3 / 2) 
  (hfocal : focalLength (Real.sqrt 3) = 2 * Real.sqrt 3) 
  (P Q : PointOnEllipse a b) 
  (hPQ : lineSlope P.x P.y Q.x Q.y = -1/2) 
  (hfirst_quadrant : P.x > 0 ∧ P.y > 0 ∧ Q.x > 0 ∧ Q.y > 0) :
  ∃ (r : ℝ), 
    lineSlope 0 0 P.x P.y * r = lineSlope P.x P.y Q.x Q.y ∧
    lineSlope P.x P.y Q.x Q.y * r = lineSlope 0 0 Q.x Q.y :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slopes_form_geometric_sequence_l356_35618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_two_one_l356_35627

/-- A power function passing through (2,1) -/
noncomputable def f : ℝ → ℝ :=
  fun x => x ^ (Real.log 1 / Real.log 2)

theorem power_function_through_two_one :
  f 2 = 1 ∧ f 4 = 1 := by
  constructor
  · -- Prove f 2 = 1
    sorry
  · -- Prove f 4 = 1
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_two_one_l356_35627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_height_is_18_l356_35686

/-- Represents a line in 2D space --/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Represents a pole --/
structure Pole where
  height : ℝ
  position : ℝ

/-- Calculates the line passing through the top of one pole and the base of another --/
noncomputable def lineFromPoles (p1 p2 : Pole) : Line :=
  { slope := (p2.height - p1.height) / (p2.position - p1.position)
    intercept := p1.height - (p2.height - p1.height) / (p2.position - p1.position) * p1.position }

/-- Finds the intersection point of two lines --/
noncomputable def findIntersection (l1 l2 : Line) : ℝ × ℝ :=
  let x := (l2.intercept - l1.intercept) / (l1.slope - l2.slope)
  let y := l1.slope * x + l1.intercept
  (x, y)

theorem intersection_height_is_18 (pole1 pole2 : Pole)
    (h1 : pole1.height = 30)
    (h2 : pole2.height = 90)
    (d : pole2.position - pole1.position = 120) :
    (findIntersection (lineFromPoles pole1 pole2) (lineFromPoles pole2 pole1)).2 = 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_height_is_18_l356_35686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_l356_35687

theorem power_equality (x : ℝ) (h : (3 : ℝ)^(4*x) = 16) : (81 : ℝ)^(x+1) = 1296 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_l356_35687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_120_terms_l356_35643

noncomputable def a : ℕ → ℝ → ℝ
  | 0, a₀ => a₀
  | n + 1, a₀ => (2 * |Real.sin (n * Real.pi / 2)| - 1) * a n a₀ + 2 * (n + 1)

noncomputable def sum_a (n : ℕ) (a₀ : ℝ) : ℝ :=
  (Finset.range n).sum (λ i => a i a₀)

/-- Theorem stating that the sum of the first 120 terms is 7320 -/
theorem sum_120_terms (a₀ : ℝ) : sum_a 120 a₀ = 7320 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_120_terms_l356_35643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_unit_vector_l356_35616

noncomputable def a : ℝ × ℝ := (1, -Real.sqrt 3)

theorem opposite_unit_vector : 
  let norm_a := Real.sqrt ((a.1)^2 + (a.2)^2)
  let unit_vector := (-a.1 / norm_a, -a.2 / norm_a)
  unit_vector = (-1/2, Real.sqrt 3 / 2) := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_unit_vector_l356_35616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_slope_theorem_l356_35619

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  m : ℝ  -- slope
  c : ℝ  -- y-intercept

noncomputable def Ellipse.focal_distance (e : Ellipse) : ℝ := 
  Real.sqrt (e.a^2 - e.b^2)

def Ellipse.contains (e : Ellipse) (p : Point) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

def parallel_to_x_axis (l : Line) : Prop :=
  l.m = 0

def bisects_angle (l₁ l₂ l₃ : Line) : Prop :=
  sorry  -- Definition of angle bisection

noncomputable def slope_between_points (p₁ p₂ : Point) : ℝ :=
  (p₂.y - p₁.y) / (p₂.x - p₁.x)

theorem ellipse_slope_theorem (e : Ellipse) (P Q A B : Point) :
  e.focal_distance = Real.sqrt 3 / 2 →
  e.contains P →
  P.x = 2 ∧ P.y = -1 →
  e.contains Q →
  parallel_to_x_axis (Line.mk 0 Q.y) →
  ∃ (l_PA l_PB l_PQ : Line),
    e.contains A ∧
    e.contains B ∧
    bisects_angle l_PA l_PB l_PQ →
    slope_between_points A B = -1/2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_slope_theorem_l356_35619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_is_30_l356_35646

/-- The length of the train in meters -/
noncomputable def train_length : ℝ := 120

/-- The time taken to cross the telegraph post in seconds -/
noncomputable def crossing_time : ℝ := 4

/-- The speed of the train in meters per second -/
noncomputable def train_speed : ℝ := train_length / crossing_time

/-- Theorem stating that the train's speed is 30 meters per second -/
theorem train_speed_is_30 : train_speed = 30 := by
  -- Unfold the definitions
  unfold train_speed train_length crossing_time
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_is_30_l356_35646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_2015_l356_35625

/-- Represents a point in a vector space -/
structure Point (V : Type*) [AddCommGroup V] [Module ℝ V] :=
  (coords : V)

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence (α : Type*) [Add α] [Sub α] :=
  (a : ℕ → α)
  (is_arithmetic : ∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n)

/-- Sum of first n terms of an arithmetic sequence -/
noncomputable def sum_n (seq : ArithmeticSequence ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (seq.a 1 + seq.a n) / 2

/-- Collinearity of three points -/
def collinear {V : Type*} [AddCommGroup V] [Module ℝ V] (A B C : Point V) : Prop :=
  ∃ t : ℝ, B.coords - A.coords = t • (C.coords - A.coords)

theorem arithmetic_sequence_sum_2015 
  {V : Type*} [AddCommGroup V] [Module ℝ V]
  (seq : ArithmeticSequence ℝ)
  (O A B C : Point V)
  (h1 : A.coords = seq.a 3 • B.coords + seq.a 2013 • C.coords)
  (h2 : collinear A B C) :
  sum_n seq 2015 = 2015 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_2015_l356_35625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_fixed_point_theorem_l356_35644

/-- Ellipse E centered at the origin with axes of symmetry along x and y axes -/
noncomputable def Ellipse (x y : ℝ) : Prop :=
  x^2 / 3 + y^2 / 4 = 1

/-- Line segment AB -/
noncomputable def LineAB (x y : ℝ) : Prop :=
  y = 2/3 * x - 2

/-- Point P -/
def P : ℝ × ℝ := (1, -2)

/-- Given a point M on the ellipse, find T on AB such that MT is parallel to x-axis -/
noncomputable def findT (M : ℝ × ℝ) : ℝ × ℝ :=
  (3 * M.2 / 2 + 3, M.2)

/-- Given M and T, find H such that MT = TH -/
noncomputable def findH (M T : ℝ × ℝ) : ℝ × ℝ :=
  (2 * T.1 - M.1, T.2)

theorem ellipse_fixed_point_theorem :
  ∀ (M N : ℝ × ℝ),
  Ellipse M.1 M.2 →
  Ellipse N.1 N.2 →
  ∃ (k : ℝ), k * (M.1 - P.1) = M.2 - P.2 ∧ k * (N.1 - P.1) = N.2 - P.2 →
  let T := findT M
  let H := findH M T
  ∃ (t : ℝ), t * (N.1 - H.1) + (1 - t) * N.2 = -2 ∧ t * (N.1 - H.1) + (1 - t) * H.2 = -2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_fixed_point_theorem_l356_35644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_f3_simplest_form_l356_35638

/-- A fraction is in simplest form if it cannot be simplified further. -/
def IsSimplestForm (n d : ℚ) : Prop :=
  ∀ k : ℚ, k ≠ 0 → (n / k).den = 1 ∧ (d / k).den = 1 → k = 1 ∨ k = -1

/-- Given fractions -/
def f1 (x y : ℚ) : ℚ × ℚ := (3 * x * y, x^2)
def f2 (x : ℚ) : ℚ × ℚ := (x - 1, x^2 - 1)
def f3 (x y : ℚ) : ℚ × ℚ := (x + y, 2 * x)
def f4 (x : ℚ) : ℚ × ℚ := (1 - x, x - 1)

/-- Theorem stating that only f3 is in simplest form -/
theorem only_f3_simplest_form (x y : ℚ) (hx : x ≠ 0) :
  ¬IsSimplestForm (f1 x y).1 (f1 x y).2 ∧
  ¬IsSimplestForm (f2 x).1 (f2 x).2 ∧
  IsSimplestForm (f3 x y).1 (f3 x y).2 ∧
  ¬IsSimplestForm (f4 x).1 (f4 x).2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_f3_simplest_form_l356_35638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_theta_value_l356_35614

open Real

-- Define the function f as noncomputable
noncomputable def f (x θ : ℝ) : ℝ := 2 * x * sin (x + θ + π / 3)

-- State the theorem
theorem odd_function_theta_value (θ : ℝ) 
  (h1 : θ > -π/2) (h2 : θ < π/2)
  (h3 : ∀ x, f x θ = -f (-x) θ) : 
  θ = π/6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_theta_value_l356_35614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l356_35605

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus
def focus : ℝ × ℝ := (1, 0)

-- Define point A
def point_A : ℝ × ℝ := (6, 3)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem parabola_properties :
  (∀ x y : ℝ, parabola x y → 
    ∃ (P : ℝ × ℝ), P.1 = x ∧ P.2 = y ∧
      ∀ (Q : ℝ × ℝ), parabola Q.1 Q.2 →
        distance P point_A + distance P focus ≥ 
        distance Q point_A + distance Q focus) ∧
  (∃ (P : ℝ × ℝ), parabola P.1 P.2 ∧
    distance P point_A + distance P focus = 7) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l356_35605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l356_35669

theorem trigonometric_equation_solution (t : Real) 
  (h1 : (1 + Real.sin t) * (1 + Real.cos t) = 9/4)
  (h2 : ∃ (p q r : ℕ+), 
    (1 - Real.sin t) * (1 - Real.cos t) = (p : Real) / (q : Real) - Real.sqrt (r : Real) ∧ 
    Nat.Coprime p q) :
  ∃ (p q r : ℕ+), 
    (1 - Real.sin t) * (1 - Real.cos t) = (p : Real) / (q : Real) - Real.sqrt (r : Real) ∧ 
    Nat.Coprime p q ∧
    p + q + r = 79 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l356_35669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_of_hyperbola_l356_35608

-- Define the hyperbola equation
def hyperbola_eq (x y : ℝ) : Prop :=
  x^2 - 3*y^2 + 6*x - 18*y - 3 = 0

-- Define the focus point
noncomputable def focus : ℝ × ℝ := (-3 - 2*Real.sqrt 7, -3)

-- Theorem statement
theorem focus_of_hyperbola :
  let (fx, fy) := focus
  ∃ c, c > 0 ∧ ∀ x y, hyperbola_eq x y →
    (x - fx)^2 + (y - fy)^2 = (x + 3)^2 + (y + 3)^2 + c^2 := by
  sorry

#check focus_of_hyperbola

end NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_of_hyperbola_l356_35608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_displacement_270_degrees_l356_35659

/-- Calculates the horizontal displacement of a wheel given its diameter and rotation angle. -/
noncomputable def wheel_displacement (diameter : ℝ) (angle : ℝ) : ℝ :=
  (angle / 360) * Real.pi * diameter

/-- Theorem stating that a wheel with diameter 56 cm rotating 270 degrees has a horizontal displacement of 42π cm. -/
theorem wheel_displacement_270_degrees :
  wheel_displacement 56 270 = 42 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_displacement_270_degrees_l356_35659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_qr_length_l356_35694

/-- Two similar triangles with given side lengths and angle -/
structure SimilarTriangles where
  XY : ℝ
  PQ : ℝ
  YZ : ℝ
  PR : ℝ
  XZ : ℝ
  angle : ℝ

/-- The theorem stating that QR = 9 given the conditions -/
theorem qr_length (t : SimilarTriangles)
    (h1 : t.XY = 8)
    (h2 : t.YZ = 18)
    (h3 : t.XZ = 12)
    (h4 : t.PQ = 4)
    (h5 : t.PR = 9)
    (h6 : t.angle = 120) :
    ∃ QR : ℝ, QR = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_qr_length_l356_35694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_surface_area_l356_35685

/-- A regular triangular prism with base edge length √3 and lateral edge length 2 -/
structure RegularTriangularPrism where
  base_edge : ℝ
  lateral_edge : ℝ
  base_edge_eq : base_edge = Real.sqrt 3
  lateral_edge_eq : lateral_edge = 2

/-- The sphere circumscribing the regular triangular prism -/
def circumscribed_sphere (prism : RegularTriangularPrism) (radius : ℝ) : Prop :=
  sorry -- We'll define this later if needed

/-- The surface area of a sphere -/
noncomputable def sphere_surface_area (radius : ℝ) : ℝ := 4 * Real.pi * radius^2

/-- Theorem: The surface area of the sphere circumscribing the regular triangular prism is 8π -/
theorem circumscribed_sphere_surface_area (prism : RegularTriangularPrism) :
  ∃ (radius : ℝ), circumscribed_sphere prism radius ∧ sphere_surface_area radius = 8 * Real.pi := by
  sorry

#check circumscribed_sphere_surface_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_surface_area_l356_35685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l356_35641

/-- Calculates the length of a train given its speed, the speed of a person walking in the same direction, and the time it takes for the train to cross the person. -/
noncomputable def train_length (train_speed : ℝ) (person_speed : ℝ) (crossing_time : ℝ) : ℝ :=
  let relative_speed := train_speed - person_speed
  let relative_speed_mps := relative_speed * (1000 / 3600)
  relative_speed_mps * crossing_time

/-- Theorem stating that given the specified conditions, the length of the train is approximately 1666.67 meters. -/
theorem train_length_calculation :
  let train_speed := 63 -- km/hr
  let person_speed := 3 -- km/hr
  let crossing_time := 29.997600191984642 -- seconds
  abs (train_length train_speed person_speed crossing_time - 1666.67) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l356_35641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l356_35606

-- Define the circles and line
def circle_A : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 4}
def circle_B : Set (ℝ × ℝ) := {p | (p.1 + 2)^2 + (p.2 - 2)^2 = 4}
def symmetry_line : Set (ℝ × ℝ) := {p | p.1 - p.2 + 2 = 0}

-- Define the point P
noncomputable def P : ℝ × ℝ := (Real.sqrt 2, Real.sqrt 2)

-- State the theorem
theorem circle_properties :
  -- 1. Circle A passes through P and is symmetric to B w.r.t. the line
  P ∈ circle_A ∧
  (∀ p : ℝ × ℝ, p ∈ circle_A ↔ 
    ∃ q : ℝ × ℝ, q ∈ circle_B ∧ 
    (p.1 + q.1) / 2 - (p.2 + q.2) / 2 + 2 = 0) →
  -- 2. The length of the common chord is 2√2
  (∃ chord : Set (ℝ × ℝ), 
    chord ⊆ circle_A ∧ 
    chord ⊆ circle_B ∧ 
    (∀ p q : ℝ × ℝ, p ∈ chord ∧ q ∈ chord → 
      Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≤ 2 * Real.sqrt 2)) →
  -- 3. Points with tangent ratio 2 lie on a specific circle
  (∀ Q : ℝ × ℝ, 
    (∃ C D : ℝ × ℝ, C ∈ circle_A ∧ D ∈ circle_B ∧
      ((Q.1 - C.1)^2 + (Q.2 - C.2)^2) * 4 = 
      ((Q.1 - D.1)^2 + (Q.2 - D.2)^2)) →
    ((Q.1 - 2/3)^2 + (Q.2 + 2/3)^2 = (2 * Real.sqrt 17 / 3)^2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l356_35606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_monotonicity_l356_35617

/-- Given a function f(x) = 3x - a/x, where a is a constant, if there exist two points
    (x₀, y₀) and (4+x₀, x₀+y₀) on the graph of f that are symmetric with respect to the origin,
    then f(x) is monotonically increasing on both (-∞, 0) and (0, +∞). -/
theorem function_monotonicity (a : ℝ) (f : ℝ → ℝ) (x₀ y₀ : ℝ) : 
  (∀ x : ℝ, x ≠ 0 → f x = 3 * x - a / x) →
  f x₀ = y₀ ∧ f (4 + x₀) = x₀ + y₀ →
  x₀ + (4 + x₀) = 0 ∧ y₀ + (x₀ + y₀) = 0 →
  (∀ x y : ℝ, x < y ∧ x < 0 ∧ y < 0 → f x < f y) ∧
  (∀ x y : ℝ, 0 < x ∧ x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_monotonicity_l356_35617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_properties_l356_35658

/-- A triangle inscribed in a semicircle with given side lengths -/
structure InscribedTriangle where
  r : ℝ
  AC : ℝ
  BC : ℝ
  h_AC : AC = 3 * r
  h_BC : BC = 4 * r

/-- The diameter of the semicircle and the hypotenuse of the triangle -/
noncomputable def InscribedTriangle.AB (t : InscribedTriangle) : ℝ := Real.sqrt (t.AC ^ 2 + t.BC ^ 2)

/-- The theorem stating the properties of the inscribed triangle -/
theorem inscribed_triangle_properties (t : InscribedTriangle) :
  t.AB = 5 * t.r ∧ t.AC ^ 2 + t.BC ^ 2 = t.AB ^ 2 := by
  sorry

#check inscribed_triangle_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_properties_l356_35658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_l356_35634

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2*x - 2/x - a * Real.log x

theorem f_monotone_decreasing (a : ℝ) :
  (∀ x ∈ Set.Ioo 1 2, StrictMonoOn (f a) (Set.Ioo 1 2)) →
  a > 5 ∧
  ∃ b ≤ 5, ∀ x ∈ Set.Ioo 1 2, StrictMonoOn (f b) (Set.Ioo 1 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_l356_35634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_questions_needed_l356_35651

/-- The number of boxes -/
def num_boxes : Nat := 100

/-- A function that represents a set of yes-or-no questions -/
def Questions := Fin num_boxes → Bool

/-- A function that represents the location of the prize -/
def PrizeLocation := Fin num_boxes

/-- A function that determines if a set of questions can uniquely identify a prize location -/
def can_identify (q : Questions) (p : PrizeLocation) : Prop :=
  ∀ p' : PrizeLocation, (∀ i : Fin num_boxes, q i = q i) → p = p'

/-- The theorem stating that 99 questions are sufficient and necessary -/
theorem min_questions_needed :
  (∃ q : Questions, ∀ p : PrizeLocation, can_identify q p) ∧
  (∀ q : Questions, (∀ p : PrizeLocation, can_identify q p) → num_boxes - 1 ≤ Fintype.card (Fin num_boxes)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_questions_needed_l356_35651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_P_on_curve_C_P_maximizes_area_l356_35683

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (1 + 2*t, 1/2 - t)

-- Define the curve C
noncomputable def curve_C (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ, Real.sin θ)

-- Define the intersection points A and B
def A : ℝ × ℝ := (2, 0)
def B : ℝ × ℝ := (0, 1)

-- Define the distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Statement for the distance between A and B
theorem distance_AB : distance A B = Real.sqrt 5 := by sorry

-- Define the point P that maximizes the area of triangle ABP
noncomputable def P : ℝ × ℝ := (-Real.sqrt 2, -Real.sqrt 2 / 2)

-- Statement that P is on curve C
theorem P_on_curve_C : ∃ θ : ℝ, curve_C θ = P := by sorry

-- Define the area of a triangle given three points
noncomputable def triangle_area (p q r : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((q.1 - p.1) * (r.2 - p.2) - (r.1 - p.1) * (q.2 - p.2))

-- Statement that P maximizes the area of triangle ABP
theorem P_maximizes_area :
  ∀ θ : ℝ, triangle_area A B P ≥ triangle_area A B (curve_C θ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_P_on_curve_C_P_maximizes_area_l356_35683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_dot_product_cosine_l356_35660

theorem vector_dot_product_cosine (x : Real) :
  (Real.sqrt 2 * Real.cos x + Real.sqrt 2 * Real.sin x = 8/5) →
  (π/4 < x) →
  (x < π/2) →
  Real.cos (x + π/4) = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_dot_product_cosine_l356_35660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_f_l356_35640

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2 + Real.cos (Real.pi / 4)

-- State the theorem
theorem derivative_of_f (x : ℝ) (h : x > 0) :
  deriv f x = 1 / (x * Real.log 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_f_l356_35640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_students_from_class_a_l356_35603

/-- Represents a class of students -/
structure StudentClass where
  name : String
  size : ℕ

/-- Represents a sample drawn from two classes -/
structure Sample where
  total : ℕ
  from_class_a : ℕ

/-- Theorem stating the number of students drawn from Class A -/
theorem students_from_class_a 
  (class_a : StudentClass)
  (class_b : StudentClass)
  (sample : Sample)
  (h1 : class_a.name = "A" ∧ class_a.size = 40)
  (h2 : class_b.name = "B" ∧ class_b.size = 50)
  (h3 : sample.total = 18)
  : sample.from_class_a = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_students_from_class_a_l356_35603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_even_units_digit_l356_35615

/-- Represents a five-digit positive integer -/
structure FiveDigitInteger where
  value : Nat
  is_five_digit : 10000 ≤ value ∧ value ≤ 99999

/-- The units digit of a number -/
def units_digit (n : Nat) : Nat :=
  n % 10

/-- Predicate for even numbers -/
def is_even (n : Nat) : Bool :=
  n % 2 = 0

/-- The set of possible units digits -/
def units_digit_set : Finset Nat :=
  Finset.range 10

/-- The set of even units digits -/
def even_units_digit_set : Finset Nat :=
  units_digit_set.filter (fun n => is_even n)

theorem probability_even_units_digit :
  (even_units_digit_set.card : ℚ) / units_digit_set.card = 1 / 2 := by
  sorry

#eval even_units_digit_set
#eval units_digit_set

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_even_units_digit_l356_35615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_P_Q_R_l356_35633

-- Define P, Q, and R
noncomputable def P : ℝ := Real.sqrt 2
noncomputable def Q : ℝ := Real.sqrt 7 - Real.sqrt 3
noncomputable def R : ℝ := Real.sqrt 6 - Real.sqrt 2

-- Theorem to prove the order of P, Q, and R
theorem order_of_P_Q_R : P > R ∧ R > Q := by
  sorry

#check order_of_P_Q_R

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_P_Q_R_l356_35633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_reciprocal_squares_l356_35693

-- Define a right-angled triangle
structure RightTriangle where
  b : ℝ
  c : ℝ
  a : ℝ
  right_angle : a^2 = b^2 + c^2
  positive_sides : b > 0 ∧ c > 0 ∧ a > 0

-- Define the volumes of solids formed by rotation
noncomputable def v (t : RightTriangle) : ℝ := (1/3) * Real.pi * (t.b^2 * t.c^2) / t.a
noncomputable def v₁ (t : RightTriangle) : ℝ := (1/3) * Real.pi * t.c^2 * t.b
noncomputable def v₂ (t : RightTriangle) : ℝ := (1/3) * Real.pi * t.b^2 * t.c

-- State the theorem
theorem volume_reciprocal_squares (t : RightTriangle) :
  (1 / (v t)^2) = (1 / (v₁ t)^2) + (1 / (v₂ t)^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_reciprocal_squares_l356_35693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_form_hyperbola_l356_35670

theorem points_form_hyperbola (s : ℝ) :
  (5 * (Real.exp s + Real.exp (-s)))^2 / 100 - (Real.exp s - Real.exp (-s))^2 / 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_form_hyperbola_l356_35670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lattice_points_on_hyperbola_l356_35656

theorem lattice_points_on_hyperbola : 
  ∃ (solutions : Finset (ℤ × ℤ)), 
    (∀ (x y : ℤ), (x, y) ∈ solutions ↔ x^2 - y^2 = 53) ∧ 
    Finset.card solutions = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lattice_points_on_hyperbola_l356_35656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cherry_lollipops_per_bouquet_is_positive_integer_l356_35662

/-- Represents the number of lollipop bouquets Carmen can create. -/
def num_bouquets : ℕ := 2

/-- Represents the number of orange lollipops in each bouquet. -/
def orange_per_bouquet : ℕ := 6

/-- Represents the number of cherry lollipops in each bouquet. -/
def cherry_per_bouquet : ℕ := 1  -- We assign a default value of 1

/-- Theorem stating that the number of cherry lollipops in each bouquet is a positive integer. -/
theorem cherry_lollipops_per_bouquet_is_positive_integer :
  cherry_per_bouquet > 0 ∧ cherry_per_bouquet ∈ Set.range Nat.succ := by
  constructor
  · -- Prove cherry_per_bouquet > 0
    exact Nat.zero_lt_succ 0
  · -- Prove cherry_per_bouquet ∈ Set.range Nat.succ
    exact ⟨0, rfl⟩

#check cherry_lollipops_per_bouquet_is_positive_integer

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cherry_lollipops_per_bouquet_is_positive_integer_l356_35662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_P_l356_35697

-- Define the circles
noncomputable def C₁ (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 3*y + 3 = 0
noncomputable def C₂ (x y : ℝ) : Prop := x^2 + y^2 + 3*x - 2*y + 2 = 0

-- Define the tangent lengths
noncomputable def PA_length_squared (x y : ℝ) : ℝ := (x - 1)^2 + (y - 3/2)^2 - (5/4)
noncomputable def PB_length_squared (x y : ℝ) : ℝ := (x + 3/2)^2 + (y - 1)^2 - (5/4)

-- State the theorem
theorem locus_of_P (x y : ℝ) :
  PA_length_squared x y = PB_length_squared x y →
  5*x + y - 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_P_l356_35697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_negative_two_l356_35611

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 1 - Real.sqrt x else (2 : ℝ)^x

-- Theorem statement
theorem f_composition_negative_two :
  f (f (-2)) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_negative_two_l356_35611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cookies_needed_is_nine_l356_35637

def brownie_price : ℚ := 3
def lemon_square_price : ℚ := 2
def cookie_price : ℚ := 4
def discount : ℚ := 2
def goal : ℚ := 50
def brownies_sold : ℕ := 4
def lemon_squares_sold : ℕ := 5

def revenue_from_sold_items : ℚ :=
  brownie_price * brownies_sold + lemon_square_price * lemon_squares_sold

def remaining_goal : ℚ := goal - revenue_from_sold_items

def revenue_per_cookie_set : ℚ := cookie_price * 3 - discount

noncomputable def cookies_needed : ℕ := 
  (Int.ceil (remaining_goal / revenue_per_cookie_set) * 3).toNat

theorem cookies_needed_is_nine : cookies_needed = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cookies_needed_is_nine_l356_35637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_pair_l356_35600

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  is_valid : hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≥ 0 ∧ tens ≤ 9 ∧ ones ≥ 0 ∧ ones ≤ 9

/-- Checks if digits are in ascending order -/
def ascending_digits (n : ThreeDigitNumber) : Prop :=
  n.hundreds < n.tens ∧ n.tens < n.ones

/-- Checks if all digits are the same -/
def all_digits_same (n : ThreeDigitNumber) : Prop :=
  n.hundreds = n.tens ∧ n.tens = n.ones

/-- Represents the name of a number in a given language -/
structure NumberName where
  first_word : String
  second_word : String
  third_word : String

/-- Checks if all words in the name start with the same letter -/
def all_words_same_start (name : NumberName) : Prop :=
  name.first_word.front = name.second_word.front ∧ name.second_word.front = name.third_word.front

/-- Checks if all words in the name start with different letters -/
def all_words_different_start (name : NumberName) : Prop :=
  name.first_word.front ≠ name.second_word.front ∧
  name.second_word.front ≠ name.third_word.front ∧
  name.first_word.front ≠ name.third_word.front

/-- Assigns a name to a number in a specific language -/
noncomputable def name_in_language (n : ThreeDigitNumber) : NumberName :=
  sorry

/-- The main theorem stating that 147 and 111 are the only pair satisfying the conditions -/
theorem unique_number_pair :
  ∃! (n₁ n₂ : ThreeDigitNumber),
    ascending_digits n₁ ∧
    all_words_same_start (name_in_language n₁) ∧
    all_digits_same n₂ ∧
    all_words_different_start (name_in_language n₂) ∧
    n₁.hundreds = 1 ∧ n₁.tens = 4 ∧ n₁.ones = 7 ∧
    n₂.hundreds = 1 ∧ n₂.tens = 1 ∧ n₂.ones = 1 :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_pair_l356_35600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_for_50_maxima_l356_35655

theorem min_omega_for_50_maxima (ω : ℝ) : 
  ω > 0 →
  (∀ x ∈ Set.Icc 0 1, ∃ y, y = Real.sin (ω * x)) →
  (∃ n : ℕ, n ≥ 50 ∧ ∀ x ∈ Set.Icc 0 1, ∃ (x₁ x₂ : ℝ), 
    x₁ < x ∧ x < x₂ ∧ 
    (∀ t ∈ Set.Ioo x₁ x₂, Real.sin (ω * x) ≥ Real.sin (ω * t))) →
  ω ≥ 197 * Real.pi / 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_for_50_maxima_l356_35655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_irrational_without_zero_l356_35602

/-- A function that checks if a real number's decimal expansion contains the digit 0 -/
noncomputable def containsZero (x : ℝ) : Prop := sorry

/-- The set of real numbers in (0,1) whose decimal expansion doesn't contain 0 -/
def S : Set ℝ := {x | 0 < x ∧ x < 1 ∧ ¬containsZero x}

/-- The existence of an irrational number c in (0,1) such that neither c nor √c
    contains 0 in their decimal expansions -/
theorem exists_irrational_without_zero : ∃ c ∈ S, Irrational c ∧ Real.sqrt c ∈ S := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_irrational_without_zero_l356_35602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l356_35610

/-- Represents an ellipse in the Cartesian coordinate system -/
structure Ellipse where
  a : ℝ  -- semi-major axis
  b : ℝ  -- semi-minor axis
  c : ℝ  -- focal distance

/-- Defines the eccentricity of an ellipse -/
noncomputable def Ellipse.eccentricity (e : Ellipse) : ℝ := e.c / e.a

/-- Defines the equation of an ellipse -/
def Ellipse.equation (e : Ellipse) : (ℝ → ℝ → Prop) :=
  fun x y => x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- Theorem: Given an ellipse with specific properties, prove its equation -/
theorem ellipse_equation (e : Ellipse) 
    (h_center : e.c < e.a) -- center at origin (implied by c < a)
    (h_foci : e.c > 0) -- foci on x-axis (implied by c > 0)
    (h_ecc : e.eccentricity = Real.sqrt 2 / 2) -- eccentricity is √2/2
    (h_perimeter : 4 * e.a = 16) -- perimeter of triangle ABF1 is 16
    : e.equation = fun x y => x^2 / 16 + y^2 / 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l356_35610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_calculation_l356_35696

theorem triangle_angle_calculation (A B C : Real) (a b c : Real) :
  a = 2 * Real.sqrt 3 →
  b = Real.sqrt 6 →
  A = π / 4 →
  Real.sin B * a = Real.sin A * b →
  B = π / 6 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_calculation_l356_35696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_max_min_sum_l356_35692

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x + Real.log x / Real.log a

theorem function_max_min_sum (a : ℝ) : 
  a > 0 ∧ a ≠ 1 ∧ 
  (∃ (max min : ℝ), 
    (∀ x ∈ Set.Icc 1 2, f a x ≤ max) ∧
    (∃ x ∈ Set.Icc 1 2, f a x = max) ∧
    (∀ x ∈ Set.Icc 1 2, f a x ≥ min) ∧
    (∃ x ∈ Set.Icc 1 2, f a x = min) ∧
    max + min = Real.log 2 / Real.log a + 6) →
  a = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_max_min_sum_l356_35692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l356_35623

noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 5 then (16 / (9 - x)) - 1
  else if 5 < x ∧ x ≤ 16 then 11 - (2 / 45) * x^2
  else 0

noncomputable def y (k : ℝ) (x : ℝ) : ℝ := k * f x

-- Theorem for part (I)
theorem part_one :
  ∀ k : ℝ, 1 ≤ k ∧ k ≤ 4 → y k 3 = 4 → k = 12/5 := by
  sorry

-- Theorem for part (II)
theorem part_two :
  ∃ t : ℝ, t = 14 ∧
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ t → y 4 x ≥ 4) ∧
  (∀ x : ℝ, x > t → y 4 x < 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l356_35623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circle_product_l356_35680

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point is equidistant from two other points -/
def IsEquidistant (P A C : Point) : Prop :=
  (P.x - A.x)^2 + (P.y - A.y)^2 = (P.x - C.x)^2 + (P.y - C.y)^2

/-- Represents an angle between three points -/
noncomputable def Angle (A B C : Point) : ℝ :=
  sorry -- Definition of angle calculation

/-- Checks if a point is on the circumcircle of a triangle -/
def OnCircumcircle (P A B C : Point) : Prop :=
  sorry -- Definition of being on the circumcircle

/-- Checks if three points are collinear -/
def AreCollinear (A B C : Point) : Prop :=
  sorry -- Definition of collinearity

/-- Calculates the distance between two points -/
def Distance (A B : Point) : ℝ :=
  sorry -- Definition of distance calculation

/-- Given a triangle ABC and a point P, proves that AF · CF = 20 under specific conditions -/
theorem triangle_circle_product (A B C P F : Point) 
  (h1 : IsEquidistant P A C)
  (h2 : Angle A P C = 2 * Angle A B C) 
  (h3 : OnCircumcircle F A B C)
  (h4 : AreCollinear A P F) 
  (h5 : Distance P C = 5) 
  (h6 : Distance P F = 4) : 
  Distance A F * Distance C F = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circle_product_l356_35680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_l356_35678

theorem angle_of_inclination (x y : ℝ) : 
  x - y - 1 = 0 → ∃ α : ℝ, α = π / 4 ∧ Real.tan α = (x - (x - 1)) / 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_l356_35678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_quadratic_coefficient_in_expansion_l356_35647

theorem max_quadratic_coefficient_in_expansion :
  ∃ (r : ℕ), r < 7 ∧
  (Nat.choose 7 r) * (2^r) = 2 * (Nat.choose 7 (r-1)) * (2^(r-1)) ∧
  (Nat.choose 7 r) * (2^r) = (5/6) * (Nat.choose 7 (r+1)) * (2^(r+1)) ∧
  ∀ (k : ℕ), k < 7 → (Nat.choose 7 k) * (2^k) * (if k = 2 then 1 else 0) ≤ 560 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_quadratic_coefficient_in_expansion_l356_35647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_to_circle_l356_35612

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 8*x + y^2 - 6*y + 32 = 0

/-- The shortest distance from the origin to the circle -/
noncomputable def shortest_distance : ℝ := 5 - Real.sqrt 2

/-- Theorem stating that the shortest distance from the origin to any point on the circle
    is greater than or equal to the calculated shortest distance -/
theorem shortest_distance_to_circle :
  ∀ (x y : ℝ), circle_equation x y →
  Real.sqrt (x^2 + y^2) ≥ shortest_distance := by
  sorry

#check shortest_distance_to_circle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_to_circle_l356_35612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sample_size_l356_35688

/-- Given a stratified sample from products produced in the ratio 1:3:5,
    prove that if the sample contains 27 items of the second type, then the total sample size is 81. -/
theorem stratified_sample_size (n : ℕ) : 
  (∃ (a b c : ℕ), a + b + c = n ∧ 5 * a = b ∧ 3 * b = 5 * c) → 
  (∃ (b : ℕ), b = 27) → 
  n = 81 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sample_size_l356_35688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l356_35636

noncomputable def f (x y : ℝ) : ℝ :=
  (5 * x^2 - 8 * x * y + 5 * y^2 - 10 * x + 14 * y + 55) /
  (9 - 25 * x^2 + 10 * x * y - y^2)^(5/2)

-- State the theorem
theorem min_value_of_f :
  ∃ (x y : ℝ), ∀ (a b : ℝ), f x y ≤ f a b ∧ f x y = 5/27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l356_35636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_isosceles_or_right_l356_35674

theorem triangle_isosceles_or_right 
  (A B C : ℝ) (a b c : ℝ) 
  (triangle_sum : A + B + C = π) 
  (positive_sides : a > 0 ∧ b > 0 ∧ c > 0) 
  (law_of_sines : a / Real.sin A = b / Real.sin B) 
  (given_condition : a * Real.cos A = b * Real.cos B) : 
  A = B ∨ C = π/2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_isosceles_or_right_l356_35674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_exponential_inequality_l356_35652

theorem negation_of_exponential_inequality :
  (¬ ∀ x : ℝ, x > 0 → Real.exp x > x + 1) ↔ (∃ x : ℝ, x > 0 ∧ Real.exp x ≤ x + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_exponential_inequality_l356_35652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_graph_shift_l356_35664

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6)
noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x)

theorem sine_graph_shift :
  ∀ x : ℝ, f x = g (x - Real.pi / 12) :=
by
  intro x
  simp [f, g]
  -- The actual proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_graph_shift_l356_35664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_bounds_f_attains_bounds_f_max_min_l356_35630

noncomputable def f (x : ℝ) : ℝ := 4 * Real.cos (x - Real.pi / 6) * Real.sin x - 2 * Real.cos (2 * x + Real.pi)

theorem f_bounds : ∀ x : ℝ, -1 ≤ f x ∧ f x ≤ 3 := by sorry

theorem f_attains_bounds : (∃ x : ℝ, f x = -1) ∧ (∃ x : ℝ, f x = 3) := by sorry

theorem f_max_min : (∀ x : ℝ, f x ≤ 3) ∧ (∀ x : ℝ, -1 ≤ f x) ∧ 
  (∃ x₁ x₂ : ℝ, f x₁ = 3 ∧ f x₂ = -1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_bounds_f_attains_bounds_f_max_min_l356_35630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l356_35665

noncomputable def a : ℕ → ℚ
  | 7 => 16/3
  | n + 1 => (3 * a n + 4) / (7 - a n)
  | _ => 0  -- This case is not used in the problem, but needed for completeness

theorem sequence_properties :
  (∃ m : ℕ, m = 8 ∧
    (∀ n, n > m → a n < 2) ∧
    (∀ n, n ≤ m → a n > 2)) ∧
  (∀ n, n ≥ 10 → (a (n-1) + a (n+1)) / 2 < a n) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l356_35665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l356_35657

-- Define the logarithms
noncomputable def a : ℝ := Real.log 6 / Real.log 3
noncomputable def b : ℝ := Real.log 10 / Real.log 5
noncomputable def c : ℝ := Real.log 14 / Real.log 7

-- Theorem statement
theorem log_inequality : a > b ∧ b > c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l356_35657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_purely_imaginary_complex_number_l356_35698

theorem purely_imaginary_complex_number (a : ℝ) :
  let z : ℂ := Complex.mk (a^3 - a) (a / (1 - a))
  (z.re = 0 ∧ z.im ≠ 0) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_purely_imaginary_complex_number_l356_35698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_properties_l356_35628

-- Define the triangle ABC
def triangle_ABC (A B C : ℝ × ℝ) : Prop :=
  -- Area of triangle ABC is √3/2
  abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) / 2 = Real.sqrt 3 / 2 ∧
  -- Dot product of AB and AC is -1
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = -1

-- Define the midpoint M of BC
def midpoint_M (B C M : ℝ × ℝ) : Prop :=
  M.1 = (B.1 + C.1) / 2 ∧ M.2 = (B.2 + C.2) / 2

-- Define the point N on BC
def point_N_on_BC (B C N : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ N.1 = B.1 + t * (C.1 - B.1) ∧ N.2 = B.2 + t * (C.2 - B.2)

-- Define N as the intersection of angle bisector of BAC and BC
def N_angle_bisector (A B C N : ℝ × ℝ) : Prop :=
  point_N_on_BC B C N ∧
  (N.1 - A.1) * (B.1 - A.1) + (N.2 - A.2) * (B.2 - A.2) =
  (N.1 - A.1) * (C.1 - A.1) + (N.2 - A.2) * (C.2 - A.2)

-- Main theorem
theorem triangle_ABC_properties (A B C M N : ℝ × ℝ) :
  triangle_ABC A B C →
  midpoint_M B C M →
  N_angle_bisector A B C N →
  Real.sqrt ((M.1 - A.1)^2 + (M.2 - A.2)^2) = Real.sqrt 3 / 2 →
  (-- Measure of angle A is 2π/3
   Real.arccos ((B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) /
     (Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) * Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2))) = 2 * Real.pi / 3) ∧
  (-- Minimum length of BC is √6
   Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) ≥ Real.sqrt 6) ∧
  (-- Length of segment MN is √7/6
   Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) = Real.sqrt 7 / 6) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_properties_l356_35628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_angle_PF2F1_is_four_fifths_l356_35681

/-- A hyperbola with foci and an intersection point -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  f1 : ℝ × ℝ  -- Left focus
  f2 : ℝ × ℝ  -- Right focus
  p : ℝ × ℝ   -- Intersection point
  h_pos : a > 0 ∧ b > 0
  h_equation : ∀ (x y : ℝ), (x^2 / a^2) - (y^2 / b^2) = 1 → (x, y) ∈ Set.range (λ t => (t, t))
  h_foci : f2.1 > f1.1  -- Right focus is to the right of left focus
  h_p_quad : p.1 < 0 ∧ p.2 > 0  -- P is in the second quadrant
  h_p_on_circle : (p.1 - f1.1)^2 + (p.2 - f1.2)^2 = (f2.1 - f1.1)^2 / 4
  h_p_on_hyperbola : (p.1^2 / a^2) - (p.2^2 / b^2) = 1
  h_eccentricity : (f2.1 - f1.1) / (2 * a) = 5

/-- The cosine of angle PF₂F₁ in the hyperbola configuration -/
noncomputable def cos_angle_PF2F1 (h : Hyperbola) : ℝ :=
  let d := Real.sqrt ((h.p.1 - h.f2.1)^2 + (h.p.2 - h.f2.2)^2)
  let c := (h.f2.1 - h.f1.1) / 2
  ((d^2 + (2*c)^2 - ((h.p.1 - h.f1.1)^2 + (h.p.2 - h.f1.2)^2)) / (2 * d * 2*c))

/-- The main theorem stating that cos ∠PF₂F₁ = 4/5 -/
theorem cos_angle_PF2F1_is_four_fifths (h : Hyperbola) :
  cos_angle_PF2F1 h = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_angle_PF2F1_is_four_fifths_l356_35681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_of_sqrt_l356_35607

/-- Represents the number of repetitions of "44.44" in the input -/
def repetitions_44 : ℕ := 2017

/-- Represents the number of repetitions of "22" in the input -/
def repetitions_22 : ℕ := 2018

/-- Represents the large number under the square root -/
noncomputable def large_number : ℝ := 
  Real.sqrt (44.44 * (10000 ^ repetitions_44) + 22 * (100 ^ repetitions_22) + 5)

/-- Function to calculate the sum of digits of an integer -/
def sum_of_digits (n : ℤ) : ℕ := sorry

/-- The main theorem to be proved -/
theorem sum_of_digits_of_sqrt : 
  sum_of_digits (Int.floor large_number) = 12107 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_of_sqrt_l356_35607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_in_acute_triangle_l356_35622

theorem sin_sum_in_acute_triangle (A B C : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ -- acute triangle
  A + B + C = Real.pi ∧ -- sum of angles in a triangle
  B > Real.pi/6 ∧ 
  Real.sin (A + Real.pi/6) = 3/5 ∧ 
  Real.cos (B - Real.pi/6) = 4/5 →
  Real.sin (A + B) = 24/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_in_acute_triangle_l356_35622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_bed_fraction_is_one_fifth_l356_35621

/-- Represents a rectangular yard with flower beds -/
structure YardWithFlowerBeds where
  /-- Length of the shorter parallel side of the trapezoidal remainder -/
  short_side : ℝ
  /-- Length of the longer parallel side of the trapezoidal remainder -/
  long_side : ℝ

/-- The fraction of the yard occupied by the flower beds -/
noncomputable def flower_bed_fraction (yard : YardWithFlowerBeds) : ℝ :=
  let triangle_leg := (yard.long_side - yard.short_side) / 2
  let triangle_area := triangle_leg ^ 2 / 2
  let total_flower_bed_area := 2 * triangle_area
  let yard_area := yard.long_side * triangle_leg
  total_flower_bed_area / yard_area

theorem flower_bed_fraction_is_one_fifth 
  (yard : YardWithFlowerBeds) 
  (h1 : yard.short_side = 18) 
  (h2 : yard.long_side = 30) : 
  flower_bed_fraction yard = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_bed_fraction_is_one_fifth_l356_35621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_odd_and_decreasing_l356_35666

-- Define the function f(x) = x^(-1)
noncomputable def f (x : ℝ) : ℝ := x⁻¹

-- State the theorem
theorem inverse_function_odd_and_decreasing :
  (∀ x : ℝ, x ≠ 0 → f (-x) = -f x) ∧ 
  (∀ x y : ℝ, 0 < x → 0 < y → x < y → f y < f x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_odd_and_decreasing_l356_35666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_values_l356_35654

/-- Curve C in polar coordinates -/
noncomputable def curve_C (θ : ℝ) : ℝ := 2 * Real.cos θ

/-- Line l in polar coordinates -/
noncomputable def line_l (θ m : ℝ) : ℝ := m / Real.sin (θ + Real.pi / 6)

/-- Condition for C and l to have exactly one common point -/
def one_common_point (m : ℝ) : Prop :=
  ∃! p : ℝ × ℝ, 
    let (r, θ) := p
    r = curve_C θ ∧ r = line_l θ m

/-- Theorem stating the possible values of m -/
theorem m_values : 
  ∀ m : ℝ, one_common_point m → m = -1/2 ∨ m = 3/2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_values_l356_35654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_vector_sum_magnitude_l356_35653

-- Define the points and vectors
noncomputable def O : ℝ × ℝ := (0, 0)
noncomputable def M : ℝ × ℝ := (Real.sqrt 2, Real.sqrt 2)
def P : ℝ × ℝ → Prop := λ p => (p.1 - O.1)^2 + (p.2 - O.2)^2 = 1

-- Define vector operations
def vectorAdd (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)
noncomputable def vectorMagnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

-- State the theorem
theorem max_vector_sum_magnitude :
  ∃ (max : ℝ), max = 3 ∧ 
  ∀ p, P p → vectorMagnitude (vectorAdd (M.1 - O.1, M.2 - O.2) (p.1 - O.1, p.2 - O.2)) ≤ max ∧
  ∃ q, P q ∧ vectorMagnitude (vectorAdd (M.1 - O.1, M.2 - O.2) (q.1 - O.1, q.2 - O.2)) = max := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_vector_sum_magnitude_l356_35653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_range_l356_35672

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a*x^2 - x - 1

-- State the theorem
theorem monotonic_decreasing_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a y < f a x) ↔ a ∈ Set.Icc (-Real.sqrt 3) (Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_range_l356_35672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_price_approx_568_63_l356_35604

/-- The original price of a product given its final price after discounts and tax -/
noncomputable def original_price (final_price : ℝ) (discount1 discount2 discount3 discount4 tax_rate : ℝ) : ℝ :=
  let price_before_tax := final_price / (1 + tax_rate)
  let price_before_discount4 := price_before_tax / (1 - discount4)
  let price_before_discount3 := price_before_discount4 / (1 - discount3)
  let price_before_discount2 := price_before_discount3 / (1 - discount2)
  price_before_discount2 / (1 - discount1)

/-- Theorem stating that the original price of the product is approximately $568.63 -/
theorem original_price_approx_568_63 :
  ∃ ε > 0, |original_price 348.82 0.0825 0.125 0.1975 0.11 0.07 - 568.63| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_price_approx_568_63_l356_35604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l356_35626

/-- Definition of the ellipse C -/
noncomputable def ellipse_C (x y : ℝ) (a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

/-- Eccentricity of the ellipse -/
noncomputable def eccentricity : ℝ := Real.sqrt 2 / 2

/-- Perimeter of triangle ABF₂ -/
noncomputable def triangle_perimeter : ℝ := 8 * Real.sqrt 2

/-- Theorem about the standard equation of ellipse C and the range of |OP|²/|AB| -/
theorem ellipse_properties :
  ∃ (a b : ℝ),
    (∀ x y, ellipse_C x y a b ↔ x^2 / 8 + y^2 / 4 = 1) ∧
    (∀ P : ℝ × ℝ,
      ellipse_C P.1 P.2 a b →
      ∀ A B : ℝ × ℝ,
        ellipse_C A.1 A.2 a b →
        ellipse_C B.1 B.2 a b →
        A ≠ B →
        (∃ k : ℝ, A.2 - B.2 = k * (A.1 - B.1)) →  -- A and B not on x-axis
        (P.1 * (A.1 - B.1) + P.2 * (A.2 - B.2) = 0) →  -- OP ⊥ AB
        Real.sqrt 2 / 2 < (P.1^2 + P.2^2) / Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) ∧
        (P.1^2 + P.2^2) / Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) ≤ 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l356_35626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parallel_to_plane_if_in_parallel_plane_l356_35690

/-- Two planes are parallel if they do not intersect -/
def ParallelPlanes (α β : Set (Fin 3 → ℝ)) : Prop :=
  α ∩ β = ∅

/-- A line is contained in a plane if all points of the line are in the plane -/
def LineInPlane (a : Set (Fin 3 → ℝ)) (α : Set (Fin 3 → ℝ)) : Prop :=
  a ⊆ α

/-- A line is parallel to a plane if they do not intersect -/
def LineParallelToPlane (a : Set (Fin 3 → ℝ)) (β : Set (Fin 3 → ℝ)) : Prop :=
  a ∩ β = ∅

/-- Main theorem: If two planes are parallel and a line is contained in one of them,
    then the line is parallel to the other plane -/
theorem line_parallel_to_plane_if_in_parallel_plane
  (α β : Set (Fin 3 → ℝ)) (a : Set (Fin 3 → ℝ))
  (h1 : ParallelPlanes α β)
  (h2 : LineInPlane a α) :
  LineParallelToPlane a β := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parallel_to_plane_if_in_parallel_plane_l356_35690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l356_35689

/-- Represents the property of one set being tangent to another -/
def IsTangentTo (S T : Set (ℝ × ℝ)) : Prop := sorry

/-- Represents the eccentricity of a conic section -/
noncomputable def Eccentricity (S : Set (ℝ × ℝ)) : ℝ := sorry

/-- The eccentricity of an ellipse with specific properties -/
theorem ellipse_eccentricity (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  let C := {(x, y) : ℝ × ℝ | x^2 / a^2 + y^2 / b^2 = 1}
  let A1A2_circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = a^2}
  let tangent_line := {(x, y) : ℝ × ℝ | b * x - a * y + 2 * a * b = 0}
  IsTangentTo A1A2_circle tangent_line →
  Eccentricity C = Real.sqrt 6 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l356_35689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_distance_is_sqrt_370_minus_15_l356_35676

/-- The distance between the closest points of two circles with centers at (3, 3) and (20, 12), 
    both tangent to the x-axis -/
noncomputable def closest_distance : ℝ :=
  let c1 : ℝ × ℝ := (3, 3)
  let c2 : ℝ × ℝ := (20, 12)
  let r1 : ℝ := c1.2  -- radius of first circle (y-coordinate)
  let r2 : ℝ := c2.2  -- radius of second circle (y-coordinate)
  let center_distance : ℝ := Real.sqrt ((c2.1 - c1.1)^2 + (c2.2 - c1.2)^2)
  center_distance - r1 - r2

theorem closest_distance_is_sqrt_370_minus_15 : 
  closest_distance = Real.sqrt 370 - 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_distance_is_sqrt_370_minus_15_l356_35676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_F_value_l356_35642

-- Define the circle equation
def circle_equation (x y D E F : ℝ) : Prop :=
  x^2 + y^2 + D*x + E*y + F = 0

-- Define the center of a circle
noncomputable def circle_center (D E : ℝ) : ℝ × ℝ :=
  (-D/2, -E/2)

-- Define the radius of a circle
noncomputable def circle_radius (D E F : ℝ) : ℝ :=
  Real.sqrt ((D^2 + E^2) / 4 - F)

-- Theorem statement
theorem circle_F_value (D E F : ℝ) :
  circle_equation 0 0 D E F ∧
  circle_center D E = (2, -4) ∧
  circle_radius D E F = 4 →
  F = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_F_value_l356_35642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_sum_coordinates_equals_nine_l356_35613

def sum_coordinates (p : ℝ × ℝ) : ℝ :=
  p.1 + p.2

theorem midpoint_sum_coordinates_equals_nine :
  let p₁ : ℝ × ℝ := (10, 7)
  let p₂ : ℝ × ℝ := (4, -3)
  sum_coordinates ((p₁.1 + p₂.1) / 2, (p₁.2 + p₂.2) / 2) = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_sum_coordinates_equals_nine_l356_35613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l356_35661

open Real

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := tan (2 * x + π / 4)

-- State the theorem
theorem smallest_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) ∧
  (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') ∧
  T = π / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l356_35661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_sculpture_first_week_cut_l356_35624

/-- Proves that the percentage of marble cut away in the first week is 30% --/
theorem marble_sculpture_first_week_cut (original_weight : ℝ) 
  (second_week_cut : ℝ) (third_week_cut : ℝ) (final_weight : ℝ) :
  original_weight = 300 →
  second_week_cut = 0.3 →
  third_week_cut = 0.15 →
  final_weight = 124.95 →
  let weight_after_second_week := final_weight / (1 - third_week_cut)
  let weight_after_first_week := weight_after_second_week / (1 - second_week_cut)
  (original_weight - weight_after_first_week) / original_weight = 0.3 := by
  sorry

#check marble_sculpture_first_week_cut

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_sculpture_first_week_cut_l356_35624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_second_tangent_l356_35650

-- Define the ellipse structure
structure Ellipse where
  center : ℝ × ℝ
  majorAxis : ℝ
  minorAxis : ℝ

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define a line in 2D space (simplified as two points)
structure Line where
  p1 : Point
  p2 : Point

-- Helper functions (these would need to be defined properly)
def is_major_axis_endpoint (E : Ellipse) (P : Point) : Prop := sorry

def is_tangent_to_ellipse (E : Ellipse) (l : Line) : Prop := sorry

def point_on_line (P : Point) (l : Line) : Prop := sorry

def not_special_case (E : Ellipse) (A B : Point) (e : Line) (P : Point) : Prop := sorry

-- Define the problem statement
theorem ellipse_second_tangent 
  (E : Ellipse) 
  (A B : Point) 
  (e : Line) 
  (P : Point) 
  (h1 : A ≠ B) -- A and B are distinct points
  (h2 : is_major_axis_endpoint E A) -- A is an endpoint of the major axis
  (h3 : is_major_axis_endpoint E B) -- B is an endpoint of the major axis
  (h4 : is_tangent_to_ellipse E e) -- e is tangent to E
  (h5 : point_on_line P e) -- P is on line e
  (h6 : not_special_case E A B e P) -- Not in any of the special cases mentioned
  : ∃ (f : Line), 
    f ≠ e ∧ 
    is_tangent_to_ellipse E f ∧ 
    point_on_line P f :=
by
  sorry -- The proof would go here


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_second_tangent_l356_35650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_sum_l356_35677

def non_decreasing (f : ℝ → ℝ) (D : Set ℝ) : Prop :=
  ∀ x₁ x₂, x₁ ∈ D → x₂ ∈ D → x₁ < x₂ → f x₁ ≤ f x₂

theorem function_sum (f : ℝ → ℝ) :
  non_decreasing f (Set.Icc 0 1) →
  (∀ x, x ∈ Set.Icc 0 1 → f (x/3) = (1/2) * f x) →
  (∀ x, x ∈ Set.Icc 0 1 → f (1-x) = 1 - f x) →
  f 0 = 0 →
  f 1 + f (1/2) + f (1/3) + f (1/6) + f (1/7) + f (1/8) = 11/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_sum_l356_35677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_x_coordinate_l356_35684

-- Define the hyperbola and circle
def my_hyperbola (x y b : ℝ) : Prop := x^2 - y^2/b^2 = 1
def my_circle (x y c : ℝ) : Prop := x^2 + y^2 = c^2

-- Define the foci and their distance
def left_focus : ℝ × ℝ := (-1, 0)
def right_focus : ℝ × ℝ := (1, 0)
def foci_distance (c : ℝ) : Prop := 2*c = 2

-- Define the point P and its properties
def point_P (x y : ℝ) : Prop := x > 0 ∧ y > 0
def distance_F1P (x y c : ℝ) : Prop := (x - 1)^2 + y^2 = c^4

-- Theorem statement
theorem hyperbola_intersection_x_coordinate 
  (b c x y : ℝ) 
  (h1 : my_hyperbola x y b) 
  (h2 : my_circle x y c) 
  (h3 : foci_distance c) 
  (h4 : point_P x y) 
  (h5 : distance_F1P x y c) : 
  x = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_x_coordinate_l356_35684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_distance_for_specific_lines_l356_35620

/-- Two lines intersecting at a point with given slopes -/
structure IntersectingLines where
  x : ℝ
  y : ℝ
  m1 : ℝ
  m2 : ℝ

/-- Calculate the x-intercept of a line given its slope and a point it passes through -/
noncomputable def x_intercept (m : ℝ) (x : ℝ) (y : ℝ) : ℝ :=
  x - y / m

/-- The distance between x-intercepts of two intersecting lines -/
noncomputable def x_intercept_distance (l : IntersectingLines) : ℝ :=
  |x_intercept l.m1 l.x l.y - x_intercept l.m2 l.x l.y|

/-- Theorem statement -/
theorem x_intercept_distance_for_specific_lines :
  let l := IntersectingLines.mk 8 20 4 (-3)
  x_intercept_distance l = 35 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_distance_for_specific_lines_l356_35620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_x_is_zero_l356_35663

/-- A parabola passing through three given points -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  eq : ℝ → ℝ := λ x => a * x^2 + b * x + c
  point1 : eq (-2) = 9
  point2 : eq 2 = 9
  point3 : eq 1 = 6

/-- The x-coordinate of the vertex of a parabola -/
noncomputable def vertex_x (p : Parabola) : ℝ := -p.b / (2 * p.a)

/-- Theorem: The x-coordinate of the vertex of the given parabola is 0 -/
theorem parabola_vertex_x_is_zero (p : Parabola) : vertex_x p = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_x_is_zero_l356_35663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_spheres_radius_l356_35675

/-- Representation of a sphere -/
structure Sphere where
  center : ℝ × ℝ × ℝ
  radius : ℝ

/-- Predicate to check if a tetrahedron is regular with edge length a -/
def is_regular_tetrahedron (a : ℝ) : Prop := sorry

/-- Predicate to check if four spheres touch each other -/
def spheres_touch (s₁ s₂ s₃ s₄ : Sphere) : Prop := sorry

/-- Predicate to check if each sphere touches three faces of the tetrahedron -/
def each_sphere_touches_three_faces (a : ℝ) (s₁ s₂ s₃ s₄ : Sphere) : Prop := sorry

/-- Predicate to check if four spheres are inscribed in a regular tetrahedron -/
def are_inscribed_spheres (a : ℝ) (s₁ s₂ s₃ s₄ : Sphere) : Prop :=
  is_regular_tetrahedron a ∧
  spheres_touch s₁ s₂ s₃ s₄ ∧
  each_sphere_touches_three_faces a s₁ s₂ s₃ s₄

/-- The radius of inscribed spheres in a regular tetrahedron -/
theorem inscribed_spheres_radius (a : ℝ) (h : a > 0) :
  ∃ r : ℝ, r > 0 ∧
  (∀ s₁ s₂ s₃ s₄ : Sphere,
    are_inscribed_spheres a s₁ s₂ s₃ s₄ →
    s₁.radius = r ∧ s₂.radius = r ∧ s₃.radius = r ∧ s₄.radius = r) ∧
  r = (a * Real.sqrt (2/3)) / (4 + 2 * Real.sqrt (2/3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_spheres_radius_l356_35675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_factorial_equation_l356_35632

theorem smallest_n_factorial_equation : ∃ (n : ℕ), n > 0 ∧ 
  (Nat.factorial (n + 1) + Nat.factorial (n + 3) = Nat.factorial n * 482) ∧ 
  ∀ (m : ℕ), m > 0 ∧ m < n → 
    Nat.factorial (m + 1) + Nat.factorial (m + 3) ≠ Nat.factorial m * 482 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_factorial_equation_l356_35632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_of_r_divided_by_15_l356_35691

theorem remainder_of_r_divided_by_15 (r : ℕ) (h : (r : ℝ) / 15 = 8.2) :
  r % 15 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_of_r_divided_by_15_l356_35691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_curve_is_line_l356_35679

open Real

/-- A curve in polar coordinates is defined by the equation θ = π/4. -/
def polar_curve (r : ℝ) (θ : ℝ) : Prop :=
  θ = π / 4

/-- Definition of a line in polar coordinates -/
def is_line (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b : ℝ), ∀ r θ, f r θ → (r * cos θ = a ∧ r * sin θ = b) ∨ (r * cos θ = -a ∧ r * sin θ = -b)

/-- The curve defined by θ = π/4 in polar coordinates is a line -/
theorem polar_curve_is_line : is_line polar_curve := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_curve_is_line_l356_35679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_XX₁_length_l356_35695

/-- Configuration of two right triangles with angle bisectors -/
structure TriangleConfig where
  -- Triangle DEF
  DE : ℝ
  DF : ℝ
  EF : ℝ
  is_right_DEF : DE^2 = DF^2 + EF^2
  DE_eq : DE = 13
  DF_eq : DF = 5
  
  -- D₁ is on EF and is the intersection of angle bisector of D with EF
  D₁F : ℝ
  D₁E : ℝ
  D₁_on_EF : D₁F + D₁E = EF
  D₁_bisects : D₁F / D₁E = DF / EF
  
  -- Triangle XYZ
  XY : ℝ
  XZ : ℝ
  YZ : ℝ
  XY_eq : XY = D₁E
  XZ_eq : XZ = D₁F
  is_right_XYZ : XY^2 = XZ^2 + YZ^2
  
  -- X₁ is on YZ and is the intersection of angle bisector of X with YZ
  X₁Z : ℝ
  X₁Y : ℝ
  X₁_on_YZ : X₁Z + X₁Y = YZ
  X₁_bisects : X₁Z / X₁Y = XZ / XY

/-- The length of XX₁ in the given configuration is 20/17 -/
theorem XX₁_length (config : TriangleConfig) : config.X₁Z = 20 / 17 := by
  sorry

#check XX₁_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_XX₁_length_l356_35695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_difference_quotient_l356_35629

open Real

variable (f : ℝ → ℝ) (a A : ℝ)

-- Define the derivative of f at a
def has_derivative_at (f : ℝ → ℝ) (a A : ℝ) :=
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - a| < δ → |f x - f a - A * (x - a)| ≤ ε * |x - a|

-- State the theorem
theorem limit_difference_quotient 
  (h : has_derivative_at f a A) :
  ∀ ε > (0 : ℝ), ∃ δ > (0 : ℝ), ∀ x : ℝ, 0 < |x| ∧ |x| < δ → 
    |((f (a + x) - f (a - x)) / x) - 2*A| < ε := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_difference_quotient_l356_35629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_bottle_cost_possibilities_l356_35671

theorem water_bottle_cost_possibilities : ∃ (n : ℕ), n = 11 ∧ 
  (∀ x : ℤ, 
    (1250 ≤ 5 * x ∧ 5 * x < 1350) ∧
    (1550 ≤ 6 * x ∧ 6 * x < 1750) →
    259 ≤ x ∧ x ≤ 269) ∧
  (∃ (possibilities : Finset ℤ),
    possibilities.card = n ∧
    (∀ x : ℤ, x ∈ possibilities ↔
      (1250 ≤ 5 * x ∧ 5 * x < 1350) ∧
      (1550 ≤ 6 * x ∧ 6 * x < 1750))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_bottle_cost_possibilities_l356_35671
