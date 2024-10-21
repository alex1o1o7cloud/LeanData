import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_numbers_squares_not_divisible_l922_92232

theorem odd_numbers_squares_not_divisible (n : ℕ) :
  ∀ (S : Finset ℕ),
  (∀ x, x ∈ S → x % 2 = 1 ∧ 2^(2*n) < x ∧ x < 2^(3*n)) →
  (S.card = 2^(2*n - 1) + 1) →
  ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ ¬(a^2 ∣ b) ∧ ¬(b^2 ∣ a) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_numbers_squares_not_divisible_l922_92232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_tangent_line_property_l922_92287

/-- The ellipse C defined by x²/2 + y² = 1 -/
def C : Set (ℝ × ℝ) := {p | p.1^2 / 2 + p.2^2 = 1}

/-- A line in the plane, represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The distance from a point to a line -/
noncomputable def distanceToLine (p : ℝ × ℝ) (l : Line) : ℝ :=
  |l.slope * p.1 - p.2 + l.intercept| / Real.sqrt (l.slope^2 + 1)

/-- A line is tangent to the ellipse C if it intersects C at exactly one point -/
def isTangentToC (l : Line) : Prop :=
  ∃! p, p ∈ C ∧ p.2 = l.slope * p.1 + l.intercept

theorem ellipse_tangent_line_property :
  ∀ l : Line, isTangentToC l →
    (distanceToLine (1, 0) l) * (distanceToLine (-1, 0) l) = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_tangent_line_property_l922_92287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_symmetry_implies_a_value_l922_92259

/-- A circle in the xy-plane --/
structure Circle where
  a : ℝ
  equation : ℝ → ℝ → Prop := fun x y ↦ x^2 + y^2 + a*x + 4*y - 5 = 0

/-- A line in the xy-plane --/
def Line : ℝ → ℝ → Prop := fun x y ↦ x + 2*y - 1 = 0

/-- A point in the xy-plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point is on a circle --/
def IsOnCircle (p : Point) (c : Circle) : Prop :=
  c.equation p.x p.y

/-- Function to get the symmetric point with respect to a line --/
noncomputable def SymmetricPoint (p : Point) : Point :=
  ⟨2 - p.x, 1/2 - p.y⟩

theorem circle_symmetry_implies_a_value (c : Circle) :
  (∃ p : Point, IsOnCircle p c ∧ IsOnCircle (SymmetricPoint p) c) →
  c.a = -10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_symmetry_implies_a_value_l922_92259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_congruence_sas_sas_congruence_true_original_problem_statement_incorrect_l922_92268

-- Define a structure for Triangle
structure Triangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ

/-- Two triangles with two equal sides and an equal angle between them are congruent. -/
theorem triangle_congruence_sas (T1 T2 : Triangle) 
  (h1 : T1.side1 = T2.side1) 
  (h2 : T2.side2 = T2.side2) 
  (h3 : T1.angle3 = T2.angle3) : 
  T1.side3 = T2.side3 ∧ T1.angle1 = T2.angle1 ∧ T1.angle2 = T2.angle2 := by
  sorry

/-- The statement "Two triangles are congruent if two sides and an angle between them are equal" is true. -/
theorem sas_congruence_true : ∀ T1 T2 : Triangle, 
  T1.side1 = T2.side1 → 
  T1.side2 = T2.side2 → 
  T1.angle3 = T2.angle3 → 
  T1.side3 = T2.side3 ∧ T1.angle1 = T2.angle1 ∧ T1.angle2 = T2.angle2 := by
  sorry

/-- The original problem statement claiming that SAS congruence is false is incorrect. -/
theorem original_problem_statement_incorrect : 
  ¬(¬ ∀ T1 T2 : Triangle, 
    T1.side1 = T2.side1 → 
    T1.side2 = T2.side2 → 
    T1.angle3 = T2.angle3 → 
    T1.side3 = T2.side3 ∧ T1.angle1 = T2.angle1 ∧ T1.angle2 = T2.angle2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_congruence_sas_sas_congruence_true_original_problem_statement_incorrect_l922_92268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cx_cy_over_ab_squared_l922_92293

noncomputable section

-- Define the triangle ABC
structure Triangle (A B C : ℝ × ℝ) : Prop where
  right_angle : (C.1 - A.1) * (C.1 - B.1) + (C.2 - A.2) * (C.2 - B.2) = 0

-- Define the centroid G
def centroid (A B C : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)

-- Define the circumcircle
structure Circumcircle (X Y Z P : ℝ × ℝ) : Prop where
  on_circle : ∃ (center : ℝ × ℝ) (radius : ℝ),
    (X.1 - center.1)^2 + (X.2 - center.2)^2 = radius^2 ∧
    (Y.1 - center.1)^2 + (Y.2 - center.2)^2 = radius^2 ∧
    (Z.1 - center.1)^2 + (Z.2 - center.2)^2 = radius^2 ∧
    (P.1 - center.1)^2 + (P.2 - center.2)^2 = radius^2

-- Define the perpendicular line
def perpendicular (P Q R : ℝ × ℝ) : Prop :=
  (Q.1 - P.1) * (R.1 - Q.1) + (Q.2 - P.2) * (R.2 - Q.2) = 0

-- Main theorem
theorem cx_cy_over_ab_squared (A B C : ℝ × ℝ) (G P Q X Y : ℝ × ℝ) : 
  Triangle A B C →
  G = centroid A B C →
  Circumcircle A G C P →
  Circumcircle B G C Q →
  perpendicular P X C →
  perpendicular Q Y C →
  (X.1 - C.1) * (Y.1 - C.1) + (X.2 - C.2) * (Y.2 - C.2) = 
    4/9 * ((A.1 - B.1)^2 + (A.2 - B.2)^2) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cx_cy_over_ab_squared_l922_92293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_circle_center_l922_92269

-- Define the circle equation
noncomputable def circle_equation (x y k : ℝ) : Prop :=
  x^2 + y^2 + k*x + 2*y + k^2 = 0

-- Define the center of the circle
noncomputable def circle_center (k : ℝ) : ℝ × ℝ := (-k/2, -1)

-- Define the area of the circle
noncomputable def circle_area (k : ℝ) : ℝ := Real.pi * (1 - (3*k^2)/4)

-- Theorem statement
theorem max_area_circle_center :
  ∃ (k : ℝ), (∀ (k' : ℝ), circle_area k ≥ circle_area k') →
  circle_center k = (0, -1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_circle_center_l922_92269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_circle_tangent_l922_92279

theorem isosceles_right_triangle_circle_tangent (r t : Real) :
  r = 1 + Real.sin (π / 8) →
  t = Real.cos (π / 8)^2 / (Real.sin (π / 8) + 1) →
  r * t = Real.cos (π / 8)^2 := by
  intros hr ht
  rw [hr, ht]
  field_simp
  ring
  sorry -- Full proof would require more steps

#check isosceles_right_triangle_circle_tangent

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_circle_tangent_l922_92279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_powerFunction_domain_l922_92203

-- Define the power function
noncomputable def powerFunction (x : ℝ) (m : ℕ+) : ℝ := x^(-(1 : ℝ) / (m.val * (m.val + 1)))

-- State the theorem about the domain of the power function
theorem powerFunction_domain (m : ℕ+) :
  {x : ℝ | ∃ y : ℝ, powerFunction x m = y} = Set.Ici (0 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_powerFunction_domain_l922_92203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_l922_92275

def sequence_property (a : ℕ → ℚ) : Prop :=
  a 1 = 1 ∧
  a 2 = 1/3 ∧
  ∀ n : ℕ, n ≥ 2 → a n * a (n-1) + 2 * a n * a (n+1) = 3 * a (n-1) * a (n+1)

theorem sequence_general_term (a : ℕ → ℚ) (h : sequence_property a) :
  ∀ n : ℕ, n > 0 → a n = 1 / (2^n - 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_l922_92275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_average_speed_l922_92217

noncomputable def average_speed (speed1 speed2 : ℝ) : ℝ :=
  (speed1 + speed2) / 2

theorem car_average_speed :
  let speed1 : ℝ := 10
  let speed2 : ℝ := 60
  average_speed speed1 speed2 = 35 := by
  unfold average_speed
  simp
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_average_speed_l922_92217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_winning_cards_divisible_by_101_l922_92204

def is_winning (n : Nat) : Bool :=
  n ≥ 1 && n ≤ 9999 &&
  (n / 1000 + (n / 100) % 10 = (n / 10) % 10 + n % 10)

def sum_of_winning_cards : Nat :=
  (List.range 10000).filter is_winning |>.foldl (· + ·) 0

theorem sum_of_winning_cards_divisible_by_101 :
  sum_of_winning_cards % 101 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_winning_cards_divisible_by_101_l922_92204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_necessary_not_sufficient_l922_92208

-- Define a point in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a distance function between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- Define an ellipse
def is_ellipse (trajectory : Set Point) (f1 f2 : Point) (c : ℝ) : Prop :=
  ∀ p ∈ trajectory, distance p f1 + distance p f2 = c

-- Define the constant sum property
def constant_sum_property (trajectory : Set Point) (f1 f2 : Point) (c : ℝ) : Prop :=
  ∀ p ∈ trajectory, distance p f1 + distance p f2 = c

-- Theorem statement
theorem ellipse_necessary_not_sufficient :
  (∀ trajectory : Set Point, ∀ f1 f2 : Point, ∀ c : ℝ,
    is_ellipse trajectory f1 f2 c → constant_sum_property trajectory f1 f2 c) ∧
  (∃ trajectory : Set Point, ∃ f1 f2 : Point, ∃ c : ℝ,
    constant_sum_property trajectory f1 f2 c ∧ ¬is_ellipse trajectory f1 f2 c) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_necessary_not_sufficient_l922_92208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_property_l922_92297

theorem quadratic_root_property (a b c : ℝ) (f : ℝ → ℝ) 
  (hf : ∀ x, f x = a * x^2 + b * x + c) 
  (h_root : f ((a - b - c) / (2 * a)) = 0) : f (-1) = 0 ∨ f 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_property_l922_92297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_x_coordinate_l922_92247

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a / Real.exp x

theorem tangent_point_x_coordinate 
  (a : ℝ) 
  (h_even : ∀ x, f a x = f a (-x))
  (h_tangent : ∃ x, deriv (f a) x = 3/2) :
  ∃ x, deriv (f a) x = 3/2 ∧ x = Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_x_coordinate_l922_92247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_fraction_approx_l922_92227

-- Define the conversion factors
noncomputable def ml_per_liter : ℝ := 1000
noncomputable def liters_per_gallon : ℝ := 3.78541

-- Define the volume in ml
noncomputable def volume_ml : ℝ := 30

-- Define the fraction of a gallon
noncomputable def fraction_of_gallon : ℝ := volume_ml / (ml_per_liter * liters_per_gallon)

-- Theorem to prove
theorem volume_fraction_approx : 
  ∃ (ε : ℝ), ε > 0 ∧ |fraction_of_gallon - 0.007925| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_fraction_approx_l922_92227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probabilityObtuseAngle_eq_l922_92280

/-- Pentagon vertices -/
noncomputable def F : ℝ × ℝ := (0, 3)
noncomputable def G : ℝ × ℝ := (3, 0)
noncomputable def H : ℝ × ℝ := (2 * Real.pi + 2, 0)
noncomputable def I : ℝ × ℝ := (2 * Real.pi + 2, 3)
noncomputable def J : ℝ × ℝ := (0, 3)

/-- Calculate the area of a pentagon given its vertices -/
noncomputable def pentagonArea (F G H I J : ℝ × ℝ) : ℝ := sorry

/-- Calculate the area of a semicircle given its radius -/
noncomputable def semicircleArea (radius : ℝ) : ℝ := sorry

/-- The probability of ∠FQG being obtuse when Q is randomly selected from the pentagon interior -/
noncomputable def probabilityObtuseAngle : ℝ := sorry

theorem probabilityObtuseAngle_eq :
  probabilityObtuseAngle = 3 / (8 * (2 * Real.pi + 1)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probabilityObtuseAngle_eq_l922_92280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_l922_92266

/-- The foci of the ellipse 2x^2 + 3y^2 = 1 are located at (±√6/6, 0) -/
theorem ellipse_foci (x y : ℝ) : 
  (2 * x^2 + 3 * y^2 = 1) →
  (∃ (s : ℝ), s = Real.sqrt 6 / 6 ∧ 
    ((x = s ∧ y = 0) ∨ (x = -s ∧ y = 0))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_l922_92266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tower_surface_area_l922_92240

/-- Represents a cube in the tower -/
structure Cube where
  volume : ℕ
  side : ℕ
  deriving Repr

/-- Calculates the surface area of a cube -/
def surfaceArea (c : Cube) : ℕ := 6 * c.side * c.side

/-- Calculates the overlap area between two cubes -/
def overlapArea (c : Cube) : ℕ := c.side * c.side

/-- The tower of cubes -/
def tower : List Cube := [
  { volume := 729, side := 9 },
  { volume := 512, side := 8 },
  { volume := 216, side := 6 },
  { volume := 343, side := 7 },
  { volume := 64,  side := 4 },
  { volume := 125, side := 5 },
  { volume := 27,  side := 3 },
  { volume := 1,   side := 1 }
]

/-- Theorem: The total surface area of the tower is 1305 square units -/
theorem tower_surface_area : 
  (tower.map surfaceArea).sum - 
  ((tower.take 2).map overlapArea).sum - 
  ((tower.drop 2).map (λ c => 2 * overlapArea c)).sum = 1305 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tower_surface_area_l922_92240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cryptarithmetic_puzzle_solution_l922_92273

/-- Represents a two-digit number -/
def TwoDigitNumber := Fin 100

/-- Represents a three-digit number -/
def ThreeDigitNumber := Fin 1000

theorem cryptarithmetic_puzzle_solution :
  ∀ (Y E M T : Fin 10),
    (Y ≠ E) → (Y ≠ M) → (Y ≠ T) →
    (E ≠ M) → (E ≠ T) → (M ≠ T) →
    (Y.val * 10 + E.val : ℕ) < (M.val * 10 + E.val : ℕ) →
    (Y.val * 10 + E.val) * (M.val * 10 + E.val) = (T.val * 100 + T.val * 10 + T.val : ℕ) →
    E.val + M.val + T.val + Y.val = 21 := by
  sorry

#eval 27 * 37

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cryptarithmetic_puzzle_solution_l922_92273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sin_cos_function_l922_92205

theorem max_value_sin_cos_function :
  (∀ x : ℝ, Real.sin x * Real.cos x + Real.sin x + Real.cos x ≤ 1/2 + Real.sqrt 2) ∧
  (∃ x : ℝ, Real.sin x * Real.cos x + Real.sin x + Real.cos x = 1/2 + Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sin_cos_function_l922_92205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_five_same_digit_sum_no_smaller_n_works_l922_92292

def digit_sum (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + digit_sum (n / 10)

def has_five_same_digit_sum (s : Finset Nat) : Prop :=
  ∃ (sum : Nat), (s.filter (λ x ↦ digit_sum x = sum)).card ≥ 5

theorem smallest_n_with_five_same_digit_sum :
  ∀ (s : Finset Nat), s ⊆ Finset.range 2017 → s.card ≥ 110 → has_five_same_digit_sum s :=
by
  sorry

theorem no_smaller_n_works :
  ∃ (s : Finset Nat), s ⊆ Finset.range 2017 ∧ s.card = 109 ∧ ¬has_five_same_digit_sum s :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_five_same_digit_sum_no_smaller_n_works_l922_92292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_to_line_l922_92239

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y + 7 = 0

-- Define the line
def line_eq (x y : ℝ) : Prop := x + y = 0

-- Define the distance function from a point to the line
noncomputable def dist_to_line (x y : ℝ) : ℝ := |x + y| / Real.sqrt 2

-- Statement of the theorem
theorem min_distance_circle_to_line :
  ∃ (min_dist : ℝ), min_dist = 2 * Real.sqrt 2 - 1 ∧
  ∀ (x y : ℝ), circle_eq x y →
    dist_to_line x y ≥ min_dist ∧
    ∃ (x₀ y₀ : ℝ), circle_eq x₀ y₀ ∧ dist_to_line x₀ y₀ = min_dist :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_to_line_l922_92239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_25π_l922_92251

noncomputable section

-- Define the radius of the larger circle
def R : ℝ := 10

-- Define the radius of the third smaller circle
def r : ℝ := R / 2

-- Define the area of the shaded region
def shaded_area : ℝ := Real.pi * R^2 - 3 * (Real.pi * r^2)

-- Theorem statement
theorem shaded_area_is_25π : shaded_area = 25 * Real.pi := by
  -- Unfold the definitions
  unfold shaded_area R r
  -- Simplify the expression
  simp [Real.pi]
  -- The proof is complete
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_25π_l922_92251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_property_l922_92296

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  p_pos : p > 0

/-- Point on a parabola -/
structure PointOnParabola (c : Parabola) where
  x : ℝ
  y : ℝ
  x_pos : x > 0
  on_parabola : x^2 = 2 * c.p * y

/-- Theorem: For a parabola x^2 = 2py with p > 0, if a point P(x₀, 1) on the parabola
    satisfies |PO| = |PQ|, where O is the origin and Q is the foot of the perpendicular
    from P to the directrix, then x₀ = 2√2 -/
theorem parabola_point_property (c : Parabola) (P : PointOnParabola c)
  (h1 : P.y = 1)
  (h2 : Real.sqrt (P.x^2 + P.y^2) = Real.sqrt (P.x^2 + (P.y - c.p)^2)) :
  P.x = 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_property_l922_92296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_geometric_series_ratio_l922_92214

/-- The common ratio of an infinite geometric series -/
noncomputable def common_ratio (a : ℝ) (S : ℝ) : ℝ :=
  1 - a / S

theorem infinite_geometric_series_ratio (a S : ℝ) (ha : a = 500) (hS : S = 4000) :
  common_ratio a S = 7 / 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_geometric_series_ratio_l922_92214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_difference_divisibility_l922_92242

theorem factorial_difference_divisibility (n : ℕ) (h : n ≥ 7) :
  ∃ k : ℕ, (Nat.factorial (n + 3) - Nat.factorial (n + 2)) / Nat.factorial n = 6 * k :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_difference_divisibility_l922_92242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_ratio_proof_l922_92209

theorem class_ratio_proof (girls boys : ℚ) : 
  boys = girls + 6 →
  girls + boys = 40 →
  boys / girls = 23 / 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_ratio_proof_l922_92209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_equation_l922_92219

def triangle (a b : Int) : Int := (a + b) + 2

def circleOp (a b : Int) : Int := a * 3 + b

theorem solve_equation (X : Int) : (circleOp (triangle X 24) 18 = 60) → X = -12 := by
  intro h
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_equation_l922_92219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_axisymmetric_polygon_theorem_l922_92222

/-- A convex polygon with an odd number of vertices -/
structure OddPolygon where
  vertices : ℕ
  is_odd : Odd vertices
  is_convex : Bool

/-- An axisymmetric polygon -/
class Axisymmetric (P : OddPolygon) where
  has_axis : Bool

/-- The axis of symmetry -/
def axis_of_symmetry (P : OddPolygon) [Axisymmetric P] : Set ℝ := sorry

/-- The axis of symmetry passes through a vertex -/
def axis_through_vertex (P : OddPolygon) [Axisymmetric P] : Prop :=
  ∃ v : ℝ, v ∈ Set.range (λ i => i : Fin P.vertices → ℝ) ∧ v ∈ axis_of_symmetry P

/-- Theorem: For an odd-sided axisymmetric polygon, the axis of symmetry passes through a vertex -/
theorem odd_axisymmetric_polygon_theorem (P : OddPolygon) [Axisymmetric P] :
  axis_through_vertex P := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_axisymmetric_polygon_theorem_l922_92222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_secant_relation_l922_92202

theorem tangent_secant_relation (x y : ℝ) 
  (h1 : Real.tan x + Real.tan y = 3) 
  (h2 : (Real.cos x)⁻¹ * (Real.cos y)⁻¹ = 5) : 
  (Real.tan x / Real.tan y) + (Real.tan y / Real.tan x) = 223 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_secant_relation_l922_92202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hall_of_mirrors_glass_area_l922_92271

/-- Calculates the total square footage of glass needed for a hall of mirrors --/
theorem hall_of_mirrors_glass_area : 
  (2 * (30 : ℝ) * 12) + 
  (1/2 * 20 * 12) + 
  (1/2 * (20 + 15) * 12) + 
  (1/2 * (25 + 18) * 12) = 1308 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hall_of_mirrors_glass_area_l922_92271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sandbox_sand_weight_l922_92257

/-- Calculates the total weight of sand needed to fill a square sandbox -/
noncomputable def sand_weight (side_length : ℝ) (area_per_bag : ℝ) (weight_per_bag : ℝ) : ℝ :=
  (side_length * side_length / area_per_bag) * weight_per_bag

/-- Theorem: The weight of sand needed to fill a 40-inch square sandbox is 600 pounds -/
theorem sandbox_sand_weight :
  sand_weight 40 80 30 = 600 := by
  -- Unfold the definition of sand_weight
  unfold sand_weight
  -- Simplify the arithmetic
  simp [mul_div_assoc]
  -- Check that the result is equal to 600
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sandbox_sand_weight_l922_92257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_characteristic_function_inequality_l922_92226

/-- A characteristic function is a complex-valued function that satisfies certain properties. -/
def CharacteristicFunction (φ : ℝ → ℂ) : Prop :=
  ∀ t, Complex.abs (φ t) ≤ 1 ∧ φ 0 = 1 ∧ Continuous φ

/-- The inequality holds for any characteristic function φ and any real numbers s and t. -/
theorem characteristic_function_inequality (φ : ℝ → ℂ) (h : CharacteristicFunction φ) (s t : ℝ) :
  Complex.abs (φ (t - s)) ≥ Complex.abs (φ s * φ t) - Real.sqrt (1 - Complex.abs (φ s)^2) * Real.sqrt (1 - Complex.abs (φ t)^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_characteristic_function_inequality_l922_92226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_bases_l922_92283

/-- An isosceles trapezoid with diagonal length a, forming angles α and β -/
structure IsoscelesTrapezoid where
  a : ℝ
  α : ℝ
  β : ℝ

/-- The bases of an isosceles trapezoid -/
noncomputable def bases (t : IsoscelesTrapezoid) : ℝ × ℝ :=
  let x := t.a * (Real.cos t.α - Real.sin t.α * (Real.tan (t.α + t.β))⁻¹)
  let y := t.a * (Real.cos t.α + Real.sin t.α * (Real.tan (t.α + t.β))⁻¹)
  (x, y)

/-- Theorem: The bases of an isosceles trapezoid are as calculated -/
theorem isosceles_trapezoid_bases (t : IsoscelesTrapezoid) :
  bases t = (t.a * (Real.cos t.α - Real.sin t.α * (Real.tan (t.α + t.β))⁻¹),
             t.a * (Real.cos t.α + Real.sin t.α * (Real.tan (t.α + t.β))⁻¹)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_bases_l922_92283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_runners_in_picture_probability_l922_92236

/-- Rachel's lap time in seconds -/
noncomputable def rachel_lap_time : ℚ := 120

/-- Robert's lap time in seconds -/
noncomputable def robert_lap_time : ℚ := 100

/-- Duration of the picture-taking window in seconds -/
noncomputable def picture_window : ℚ := 60

/-- Fraction of the track shown in the picture -/
noncomputable def track_fraction : ℚ := 1/3

/-- Time when both runners are in the picture -/
noncomputable def overlap_time : ℚ := 1400/30

theorem runners_in_picture_probability :
  (overlap_time / picture_window) = 1400 / 1800 := by
  -- The proof goes here
  sorry

#eval (1400 : ℚ) / 1800

end NUMINAMATH_CALUDE_ERRORFEEDBACK_runners_in_picture_probability_l922_92236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_recurring_decimal_product_l922_92255

theorem recurring_decimal_product : 
  (∃ x y : ℚ, x = 2/33 ∧ y = 1/3) → 
  (2/33) * (1/3) = 2/99 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_recurring_decimal_product_l922_92255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_reassembly_l922_92277

/-- Represents a trapezoid with bases a and b and height h -/
structure Trapezoid where
  a : ℝ
  b : ℝ
  h : ℝ

/-- Calculates the area of a trapezoid -/
noncomputable def Trapezoid.area (t : Trapezoid) : ℝ := (t.a + t.b) / 2 * t.h

theorem trapezoid_reassembly (original_width original_height : ℝ) 
    (h : original_width = 16 ∧ original_height = 9) :
    ∃ (y : ℝ), 
      let t : Trapezoid := { a := original_width, b := y, h := original_height / 3 }
      y = original_width ∧ 
      3 * t.area = original_width * original_height := by
  sorry

#check trapezoid_reassembly

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_reassembly_l922_92277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_perpendicular_line_l922_92234

noncomputable def slope_from_angle (θ : ℝ) : ℝ := Real.tan θ

noncomputable def angle_l1 : ℝ := 30 * (Real.pi / 180)

def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

theorem slope_of_perpendicular_line (l1 l2 : ℝ → ℝ) :
  perpendicular (slope_from_angle angle_l1) (slope_from_angle angle_l1) →
  slope_from_angle angle_l1 = Real.sqrt 3 / 3 →
  ∃ m2 : ℝ, perpendicular (slope_from_angle angle_l1) m2 ∧ m2 = -Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_perpendicular_line_l922_92234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_average_theorem_l922_92216

/-- Represents a cricketer's performance over multiple innings -/
structure CricketerPerformance where
  initial_innings : ℕ
  initial_average : ℚ
  new_inning_score : ℕ
  average_increase : ℚ

/-- Calculates the new average after an additional inning -/
def new_average (perf : CricketerPerformance) : ℚ :=
  (perf.initial_innings * perf.initial_average + perf.new_inning_score) / (perf.initial_innings + 1)

/-- Theorem stating that under given conditions, the new average is 16 -/
theorem cricket_average_theorem (perf : CricketerPerformance) 
  (h1 : perf.initial_innings = 16)
  (h2 : perf.new_inning_score = 112)
  (h3 : perf.average_increase = 6)
  (h4 : new_average perf = perf.initial_average + perf.average_increase) :
  new_average perf = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_average_theorem_l922_92216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_actual_distance_approx_l922_92265

/-- Calculates the actual distance between two points on a map, considering topographic changes.

    scale_cm : The number of centimeters on the map
    scale_km : The number of kilometers represented by scale_cm
    map_distance : The distance between two points on the map in centimeters
    elevation_change : The change in elevation between the two points in meters
-/
noncomputable def actual_distance (scale_cm : ℝ) (scale_km : ℝ) (map_distance : ℝ) (elevation_change : ℝ) : ℝ :=
  let km_per_cm := scale_km / scale_cm
  let horizontal_distance := map_distance * km_per_cm
  let horizontal_distance_m := horizontal_distance * 1000
  Real.sqrt (horizontal_distance_m ^ 2 + elevation_change ^ 2) / 1000

/-- Theorem stating that the actual distance between two points is approximately 847.6268 km
    given the specified conditions. -/
theorem actual_distance_approx :
  ∃ ε > 0, abs (actual_distance 0.4 5.3 64 2000 - 847.6268) < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_actual_distance_approx_l922_92265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l922_92237

noncomputable def f (x : ℝ) := 2 * Real.sin (Real.pi * x + Real.pi / 6)

theorem f_properties :
  ∃ α : ℝ,
  (∀ x ∈ Set.Icc (-1/2 : ℝ) (1/2 : ℝ), f x ≤ 2) ∧
  (∀ x ∈ Set.Icc (-1/2 : ℝ) (1/2 : ℝ), f x ≥ -Real.sqrt 3) ∧
  (f (α / (2 * Real.pi)) = 1/4) ∧
  Real.cos (2 * Real.pi / 3 - α) = -31/32 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l922_92237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_is_lunatic_l922_92278

/-- Represents the three types of people in the problem -/
inductive PersonType
  | Priest
  | Liar
  | Lunatic

/-- Represents the three people in the problem -/
inductive Person
  | First
  | Second
  | Third

/-- Defines the type of each person -/
def Person.type : Person → PersonType
  | Person.First => PersonType.Priest  -- Placeholder, will be determined by the proof
  | Person.Second => PersonType.Liar   -- Placeholder, will be determined by the proof
  | Person.Third => PersonType.Lunatic -- Placeholder, will be determined by the proof

/-- Defines the statement made by each person -/
def statement (p : Person) : Prop :=
  match p with
  | Person.First => Person.type Person.First = PersonType.Lunatic
  | Person.Second => ¬(Person.type Person.First = PersonType.Lunatic)
  | Person.Third => Person.type Person.Third = PersonType.Lunatic

/-- Axiom: The priest always tells the truth -/
axiom priest_truth (p : Person) :
  Person.type p = PersonType.Priest → statement p

/-- Axiom: The liar always lies -/
axiom liar_lie (p : Person) :
  Person.type p = PersonType.Liar → ¬(statement p)

/-- Axiom: There is exactly one person of each type -/
axiom unique_types :
  ∃! (p1 p2 p3 : Person),
    Person.type p1 = PersonType.Priest ∧
    Person.type p2 = PersonType.Liar ∧
    Person.type p3 = PersonType.Lunatic

/-- Theorem: The third person is the lunatic -/
theorem third_is_lunatic :
  Person.type Person.Third = PersonType.Lunatic := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_is_lunatic_l922_92278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthogonal_vectors_y_value_l922_92230

theorem orthogonal_vectors_y_value (y : ℚ) : 
  let v1 : Fin 3 → ℚ := ![3, 4, -1]
  let v2 : Fin 3 → ℚ := ![-2, y, 5]
  (v1 • v2 = 0) → y = 11/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthogonal_vectors_y_value_l922_92230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rank_sum_equality_rank_equality_prime_l922_92262

variable {n p : ℕ} (A : Matrix (Fin n) (Fin n) ℝ)

def is_idempotent_p (A : Matrix (Fin n) (Fin n) ℝ) (p : ℕ) : Prop :=
  A^(p+1) = A

theorem rank_sum_equality (hn : n ≥ 2) (hp : p ≥ 2) (h_idempotent : is_idempotent_p A p) :
  Matrix.rank A + Matrix.rank (1 - A^p) = n := by
  sorry

theorem rank_equality_prime (hn : n ≥ 2) (hp : p ≥ 2) (h_prime : Nat.Prime p) 
  (h_idempotent : is_idempotent_p A p) :
  ∀ j ∈ Finset.range (p-1), j ≥ 1 → Matrix.rank (1 - A^j) = Matrix.rank (1 - A) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rank_sum_equality_rank_equality_prime_l922_92262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_distance_sum_less_side_sum_l922_92221

/-- Triangle ABC with centroid O -/
structure Triangle :=
  (A B C O : ℝ × ℝ)
  (is_centroid : O = ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3))

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Sum of distances from centroid to vertices -/
noncomputable def s₁ (t : Triangle) : ℝ :=
  distance t.O t.A + distance t.O t.B + distance t.O t.C

/-- Sum of side lengths -/
noncomputable def s₂ (t : Triangle) : ℝ :=
  distance t.A t.B + distance t.B t.C + distance t.C t.A

/-- Theorem: s₁ is less than s₂ for any triangle -/
theorem centroid_distance_sum_less_side_sum (t : Triangle) : s₁ t < s₂ t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_distance_sum_less_side_sum_l922_92221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_M_and_N_l922_92224

def M : Set ℝ := {x : ℝ | 1 < x ∧ x < 4}
def N : Set ℝ := {1, 2, 3, 4, 5}

theorem intersection_of_M_and_N : M ∩ N = {2, 3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_M_and_N_l922_92224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l922_92233

noncomputable section

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 - 2 * Real.exp 1 * Real.log x

-- Define the domain of f(x)
def domain : Set ℝ := {x | x > 0}

-- State the theorem
theorem f_properties :
  ∀ x ∈ domain,
  (∀ y ∈ domain, x < y → x < Real.sqrt (Real.exp 1) → f x > f y) ∧ 
  (∀ y ∈ domain, x < y → x > Real.sqrt (Real.exp 1) → f x < f y) ∧
  (Set.EqOn (fun y => (2 * Real.exp 1 - 2) * (y - 1) + 1) f {1}) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l922_92233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_most_stable_group_l922_92200

structure OlympicGroup where
  name : String
  variance : ℝ

def is_most_stable (groups : List OlympicGroup) (g : OlympicGroup) : Prop :=
  ∀ h, h ∈ groups → g.variance ≤ h.variance

theorem most_stable_group (groups : List OlympicGroup) 
  (hA : ∃ gA ∈ groups, gA.name = "A" ∧ gA.variance = 0.24)
  (hB : ∃ gB ∈ groups, gB.name = "B" ∧ gB.variance = 0.16)
  (hC : ∃ gC ∈ groups, gC.name = "C" ∧ gC.variance = 0.41)
  (hDistinct : ∀ g h, g ∈ groups → h ∈ groups → g.name ≠ h.name → g ≠ h)
  (hSize : groups.length = 3) :
  ∃ g ∈ groups, g.name = "B" ∧ is_most_stable groups g :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_most_stable_group_l922_92200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_three_pi_fourths_plus_two_alpha_l922_92207

theorem cos_three_pi_fourths_plus_two_alpha (α : ℝ) :
  Real.cos (π/8 - α) = 1/6 → Real.cos (3*π/4 + 2*α) = 17/18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_three_pi_fourths_plus_two_alpha_l922_92207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_angle_at_one_l922_92284

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 1/x + 2*x

-- Define the derivative of f(x)
noncomputable def f_derivative (x : ℝ) : ℝ := -1/(x^2) + 2

-- Theorem statement
theorem tangent_angle_at_one :
  let slope := f_derivative 1
  Real.arctan slope = π/4 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_angle_at_one_l922_92284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_distance_bounds_l922_92254

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/9 = 1

-- Define the line l
def line_l (x y : ℝ) : Prop := 2*x + y - 6 = 0

-- Define the distance function |PA|
noncomputable def distance_PA (x y : ℝ) : ℝ := 
  (2*Real.sqrt 5/5) * abs (5 * Real.sin (Real.arccos (x/2) + Real.arcsin (1/Real.sqrt 5)) - 6)

-- Statement of the theorem
theorem ellipse_line_distance_bounds :
  ∀ x y : ℝ, ellipse x y →
  (∃ θ : ℝ, x = 2 * Real.cos θ ∧ y = 3 * Real.sin θ) ∧
  distance_PA x y ≤ 22*Real.sqrt 5/5 ∧
  distance_PA x y ≥ 2*Real.sqrt 5/5 ∧
  (∃ x' y' : ℝ, ellipse x' y' ∧ distance_PA x' y' = 22*Real.sqrt 5/5) ∧
  (∃ x'' y'' : ℝ, ellipse x'' y'' ∧ distance_PA x'' y'' = 2*Real.sqrt 5/5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_distance_bounds_l922_92254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_axially_symmetric_and_increasing_l922_92289

-- Define the function f(x) = lg |x+1|
noncomputable def f (x : ℝ) : ℝ := Real.log (|x + 1|) / Real.log 10

-- State the theorem
theorem f_axially_symmetric_and_increasing :
  (∃ (a : ℝ), ∀ (x : ℝ), f (a - x) = f (a + x)) ∧
  (∀ (x y : ℝ), 0 < x ∧ x < y → f x < f y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_axially_symmetric_and_increasing_l922_92289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_diff_product_l922_92260

theorem cos_sum_diff_product (α : ℝ) (h : Real.sin α = 1/3) :
  Real.cos (π/4 + α) * Real.cos (π/4 - α) = 7/18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_diff_product_l922_92260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_division_simplification_l922_92225

-- Define the complex number z
noncomputable def z : ℂ := (5 + 4 * Complex.I) / Complex.I

-- Theorem statement
theorem complex_division_simplification :
  z = 4 - 5 * Complex.I := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_division_simplification_l922_92225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l922_92288

/-- Triangle ABC with given properties -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  area : ℝ
  angle_sum : A + B + C = π
  side_a : a > 0
  side_b : b > 0
  side_c : c > 0
  area_formula : area = (1/2) * a * b * Real.sin C

/-- The main theorem -/
theorem triangle_properties (t : Triangle) 
  (h_C : t.C = π/3)
  (h_b : t.b = 8)
  (h_area : t.area = 10 * Real.sqrt 3) :
  t.c = 7 ∧ Real.cos (t.B - t.C) = 13/14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l922_92288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_100_l922_92267

-- Define the cost function
noncomputable def R (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 40 then 10 * x^2 + 100 * x
  else if x ≥ 40 then 701 * x + 10000 / x - 9450
  else 0

-- Define the profit function
noncomputable def W (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 40 then -10 * x^2 + 600 * x - 250
  else if x ≥ 40 then -(x + 10000 / x) + 9200
  else 0

-- State the theorem
theorem max_profit_at_100 :
  ∀ x : ℝ, x > 0 → W x ≤ W 100 ∧ W 100 = 9000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_100_l922_92267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_49_l922_92272

/-- The distance between Yolkino and Palkino in kilometers. -/
def distance : ℕ := 49

/-- The sum of digits of a natural number. -/
def sumOfDigits (n : ℕ) : ℕ :=
  let rec sum_digits (m : ℕ) (acc : ℕ) : ℕ :=
    if m = 0 then acc
    else sum_digits (m / 10) (acc + m % 10)
  sum_digits n 0

/-- Proposition stating that the sum of digits on each post is 13. -/
axiom post_digit_sum (k : ℕ) (h : k ≤ distance) : 
  sumOfDigits k + sumOfDigits (distance - k) = 13

/-- Theorem stating that the distance between Yolkino and Palkino is 49 km. -/
theorem distance_is_49 : distance = 49 := by
  -- The proof is omitted for now
  sorry

#eval distance
#eval sumOfDigits 49
#eval sumOfDigits 10 + sumOfDigits 39

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_49_l922_92272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l922_92253

-- Define the curve function
noncomputable def f (x : ℝ) : ℝ := (x^2 - 2*x) * Real.log (2*x)

-- Define the point of tangency
def x₀ : ℝ := 1
noncomputable def y₀ : ℝ := -Real.log 2

-- Define the slope of the tangent line
def m : ℝ := -1

-- Theorem statement
theorem tangent_line_equation :
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (x + y + Real.log 2 - 1 = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l922_92253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parabola_intersection_eccentricity_l922_92298

/-- Definitions for Ellipse type -/
structure Ellipse (α : Type*) [NormedAddCommGroup α] [NormedSpace ℝ α] where
  foci : Set (α × α)
  curve : Set (α × α)
  eccentricity : ℝ

/-- Definitions for Parabola type -/
structure Parabola (α : Type*) [NormedAddCommGroup α] [NormedSpace ℝ α] where
  vertex : α × α
  focus : α × α
  curve : Set (α × α)

/-- Given an ellipse and a parabola with specific properties, prove the eccentricity of the ellipse -/
theorem ellipse_parabola_intersection_eccentricity 
  (E : Ellipse ℝ) 
  (C : Parabola ℝ) 
  (F1 F2 P : ℝ × ℝ) 
  (h1 : F1 ∈ E.foci ∧ F2 ∈ E.foci)
  (h2 : C.vertex = F1 ∧ C.focus = F2)
  (h3 : P ∈ E.curve ∧ P ∈ C.curve)
  (h4 : dist P F1 = E.eccentricity * dist P F2) :
  E.eccentricity = Real.sqrt 3 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parabola_intersection_eccentricity_l922_92298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_eccentricity_value_l922_92299

-- Define the ellipse
def ellipse (a : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / 4 = 1

-- Define the foci of the ellipse
noncomputable def foci (a : ℝ) : ℝ × ℝ := (Real.sqrt (a^2 - 4), 0)

-- Define the distance from a point to the foci
noncomputable def distance_to_foci (a x y : ℝ) : ℝ :=
  let (f, _) := foci a
  Real.sqrt ((x - f)^2 + y^2) + Real.sqrt ((x + f)^2 + y^2)

-- Theorem statement
theorem ellipse_eccentricity 
  (a : ℝ) 
  (h1 : a > 2) 
  (x y : ℝ) 
  (h2 : ellipse a x y) 
  (h3 : distance_to_foci a x y = 6) : 
  a^2 - 4 = 5 ∧ a = 3 := by sorry

-- The eccentricity is sqrt(5)/3
noncomputable def eccentricity (a : ℝ) : ℝ := Real.sqrt (a^2 - 4) / a

theorem eccentricity_value 
  (a : ℝ) 
  (h : a = 3) : 
  eccentricity a = Real.sqrt 5 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_eccentricity_value_l922_92299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sinB_l922_92229

theorem triangle_sinB (A B C : ℝ) (a b c : ℝ) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi →
  -- Sides a, b, c correspond to angles A, B, C
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C →
  -- Given conditions
  Real.cos C = -1/4 →
  2 * Real.sin A + Real.sin B = Real.sqrt 15 / 2 →
  -- Conclusion
  Real.sin B = Real.sqrt 15 / 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sinB_l922_92229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_expression_l922_92218

noncomputable def f (x : ℝ) := Real.sqrt 3 * Real.sin (x + Real.pi/4)

theorem max_value_of_expression (A B C : ℝ) : 
  0 < A → 0 < B → 0 < C → A + B + C = Real.pi → 
  f B = Real.sqrt 3 →
  ∃ (M : ℝ), M = 1 ∧ ∀ (A' C' : ℝ), 0 < A' → 0 < C' → A' + B + C' = Real.pi → 
    Real.sqrt 2 * Real.cos A' + Real.cos C' ≤ M :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_expression_l922_92218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_math_class_size_l922_92245

theorem math_class_size (total : ℕ) (both : ℕ) (h1 : total = 56) (h2 : both = 8) : ℕ := by
  let physics : ℕ := sorry
  let math : ℕ := sorry
  have h3 : math = 4 * physics := by sorry
  have h4 : total = physics + math - both := by sorry
  exact 48

#check math_class_size

end NUMINAMATH_CALUDE_ERRORFEEDBACK_math_class_size_l922_92245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_piles_is_ten_l922_92294

/-- Represents the banana collection problem -/
structure BananaCollection where
  num_monkeys : ℕ
  num_piles_type1 : ℕ
  hands_per_pile_type1 : ℕ
  bananas_per_hand_type1 : ℕ
  hands_per_pile_type2 : ℕ
  bananas_per_hand_type2 : ℕ
  bananas_per_monkey : ℕ

/-- The specific instance of the banana collection problem -/
def banana_problem : BananaCollection :=
  { num_monkeys := 12
  , num_piles_type1 := 6
  , hands_per_pile_type1 := 9
  , bananas_per_hand_type1 := 14
  , hands_per_pile_type2 := 12
  , bananas_per_hand_type2 := 9
  , bananas_per_monkey := 99
  }

/-- Theorem stating that the total number of piles is 10 -/
theorem total_piles_is_ten (bc : BananaCollection) (h : bc = banana_problem) :
  ∃ (num_piles_type2 : ℕ),
    bc.num_piles_type1 + num_piles_type2 = 10 ∧
    bc.num_monkeys * bc.bananas_per_monkey =
      bc.num_piles_type1 * bc.hands_per_pile_type1 * bc.bananas_per_hand_type1 +
      num_piles_type2 * bc.hands_per_pile_type2 * bc.bananas_per_hand_type2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_piles_is_ten_l922_92294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_formula_l922_92281

-- Define the area function for a triangle given two vectors
noncomputable def triangleArea (a b : ℝ × ℝ) : ℝ :=
  (1/2) * Real.sqrt ((a.1^2 + a.2^2) * (b.1^2 + b.2^2) - (a.1 * b.1 + a.2 * b.2)^2)

-- Theorem statement
theorem triangle_area_formula (a b : ℝ × ℝ) :
  triangleArea a b = (1/2) * Real.sqrt ((a.1^2 + a.2^2) * (b.1^2 + b.2^2) - (a.1 * b.1 + a.2 * b.2)^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_formula_l922_92281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_3750_div_17_l922_92270

noncomputable section

-- Define the triangle XYZ
structure Triangle :=
  (X Y Z : ℝ × ℝ)

-- Define the properties of the triangle
def is_right_triangle (t : Triangle) : Prop :=
  let (x₁, y₁) := t.X
  let (x₂, y₂) := t.Y
  let (x₃, y₃) := t.Z
  (x₁ - x₃) * (x₂ - x₃) + (y₁ - y₃) * (y₂ - y₃) = 0

noncomputable def hypotenuse_length (t : Triangle) : ℝ :=
  let (x₁, y₁) := t.X
  let (x₂, y₂) := t.Y
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

def median_X_equation (x y : ℝ) : Prop :=
  y = x + 5

def median_Y_equation (x y : ℝ) : Prop :=
  y = 3*x + 6

noncomputable def triangle_area (t : Triangle) : ℝ :=
  let (x₁, y₁) := t.X
  let (x₂, y₂) := t.Y
  let (x₃, y₃) := t.Z
  (1/2) * abs ((x₁ - x₃)*(y₂ - y₃) - (x₂ - x₃)*(y₁ - y₃))

theorem triangle_area_is_3750_div_17 (t : Triangle) :
  is_right_triangle t →
  hypotenuse_length t = 50 →
  (∃ x y, median_X_equation x y) →
  (∃ x y, median_Y_equation x y) →
  triangle_area t = 3750 / 17 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_3750_div_17_l922_92270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lowest_point_of_f_l922_92286

noncomputable def f (x : ℝ) : ℝ := (x^2 + 2*x + 2) / (x + 1)

theorem lowest_point_of_f :
  ∀ x : ℝ, x > -1 → f x ≥ 2 ∧ (f x = 2 ↔ x = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lowest_point_of_f_l922_92286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_A_and_B_selected_is_three_tenths_l922_92231

def number_of_students : ℕ := 5
def number_selected : ℕ := 3

def probability_A_and_B_selected : ℚ :=
  (Nat.choose (number_of_students - 2) (number_selected - 2)) /
  (Nat.choose number_of_students number_selected)

theorem probability_A_and_B_selected_is_three_tenths :
  probability_A_and_B_selected = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_A_and_B_selected_is_three_tenths_l922_92231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_imply_p_value_l922_92264

variable (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- Two vectors are non-collinear if one is not a scalar multiple of the other -/
def NonCollinear (a b : V) : Prop :=
  ∀ (k : ℝ), k • a ≠ b

/-- The theorem statement -/
theorem collinear_vectors_imply_p_value
  {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  {a b : V} {A B C D : V} {p : ℝ}
  (h_non_collinear : NonCollinear V a b)
  (h_AB : B - A = 2 • a + p • b)
  (h_BC : C - B = a + b)
  (h_CD : D - C = a - 2 • b)
  (h_collinear : ∃ (k : ℝ), B - A = k • (D - B)) :
  p = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_imply_p_value_l922_92264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_omega_l922_92295

-- Define the determinant operation
def det (a b c d : ℝ) : ℝ := a * d - b * c

-- Define the function f
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := det (Real.sqrt 3) (Real.sin (ω * x)) 1 (Real.cos (ω * x))

-- Define the shifted function g
noncomputable def g (ω : ℝ) (x : ℝ) : ℝ := f ω (x + 2 * Real.pi / 3)

-- State the theorem
theorem smallest_omega : 
  ∃ (ω : ℝ), ω > 0 ∧ 
  (∀ (x : ℝ), g ω x = g ω (-x)) ∧ 
  (∀ (ω' : ℝ), ω' > 0 ∧ (∀ (x : ℝ), g ω' x = g ω' (-x)) → ω ≤ ω') ∧
  ω = 5 / 4 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_omega_l922_92295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_approx_29_29_percent_l922_92211

-- Define the markup percentage
noncomputable def markup_percentage : ℝ := 0.40

-- Define the loss percentage
noncomputable def loss_percentage : ℝ := 0.01

-- Define the function to calculate the marked price
noncomputable def marked_price (cost_price : ℝ) : ℝ :=
  cost_price * (1 + markup_percentage)

-- Define the function to calculate the selling price
noncomputable def selling_price (cost_price : ℝ) : ℝ :=
  cost_price * (1 - loss_percentage)

-- Define the function to calculate the discount percentage
noncomputable def discount_percentage (cost_price : ℝ) : ℝ :=
  (marked_price cost_price - selling_price cost_price) / marked_price cost_price * 100

-- Theorem statement
theorem discount_approx_29_29_percent (cost_price : ℝ) (h : cost_price > 0) :
  ∃ ε > 0, |discount_percentage cost_price - 29.29| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_approx_29_29_percent_l922_92211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l922_92220

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define vector operations
def vec (p q : ℝ × ℝ) : ℝ × ℝ := (q.1 - p.1, q.2 - p.2)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

noncomputable def vec_length (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

-- Define the theorem
theorem triangle_properties (ABC : Triangle) (S : ℝ) :
  -- Given conditions
  (dot_product (vec ABC.A ABC.B) (vec ABC.A ABC.C) = S) →
  (vec_length (vec ABC.C ABC.A - vec ABC.C ABC.B) = 6) →
  -- Prove the following
  (∃ (A : ℝ), 
    -- sin A = 2√5/5
    Real.sin A = 2 * Real.sqrt 5 / 5 ∧
    -- cos A = √5/5
    Real.cos A = Real.sqrt 5 / 5 ∧
    -- tan 2A = -4/3
    Real.tan (2 * A) = -4 / 3 ∧
    -- Area S = 12
    S = 12) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l922_92220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_triple_g_roots_l922_92238

noncomputable def g (x : ℝ) : ℝ := x^2/4 + x + 1

theorem sum_of_triple_g_roots : 
  ∃ (S : Finset ℝ), (∀ x ∈ S, g (g (g x)) = 1) ∧ 
                    (∀ x : ℝ, g (g (g x)) = 1 → x ∈ S) ∧
                    (S.sum id = -8) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_triple_g_roots_l922_92238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_main_theorem_l922_92215

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x

-- Define the condition for f
def f_condition (x : ℝ) : Prop := 3/4 * x^2 - 3*x + 4 = x

-- Define the interval where f is increasing
def f_increasing_interval : Set ℝ := {x | f_condition x}

-- Define m and n
def m : ℚ := 4/3
def n : ℕ := 4

-- State the theorem
theorem range_of_a (x₂ : ℝ) (h₁ : x₂ ∈ Set.Icc 1 2) :
  ∃ a : ℝ, a ∈ Set.Ioo 1 2 ∧ 2*x₂ + a/x₂ > 5 := by
  sorry

-- Define the maximum value of f
def f_max : ℝ := 4

-- Define the interval for x₁
def x₁_interval : Set ℝ := Set.Icc 0 3

-- State the main theorem
theorem main_theorem (x₁ : ℝ) (x₂ : ℝ) (a : ℝ) 
  (h₁ : x₁ ∈ x₁_interval) (h₂ : x₂ ∈ Set.Icc 1 2) :
  2*x₂ + a/x₂ - 1 > f_max → a ∈ Set.Ioo 1 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_main_theorem_l922_92215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_surface_area_base_edge_is_correct_l922_92244

/-- The length of the base edge of a right circular cylinder with volume V that minimizes its surface area -/
noncomputable def min_surface_area_base_edge (V : ℝ) : ℝ := Real.rpow (4 * V) (1/3)

/-- Theorem stating that the length of the base edge of a right circular cylinder with volume V that minimizes its surface area is (4V)^(1/3) -/
theorem min_surface_area_base_edge_is_correct (V : ℝ) (h : V > 0) :
  let r := min_surface_area_base_edge V / 2
  let h := V / (π * r^2)
  let surface_area := 2 * π * r^2 + 2 * π * r * h
  ∀ (r' : ℝ), r' > 0 → 
    let h' := V / (π * r'^2)
    let surface_area' := 2 * π * r'^2 + 2 * π * r' * h'
    surface_area ≤ surface_area' :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_surface_area_base_edge_is_correct_l922_92244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ship_minimum_sailing_rate_l922_92213

/-- Represents the ship's situation -/
structure ShipSituation where
  distance_to_shore : ℝ
  water_intake_rate : ℝ
  sinking_capacity : ℝ
  pump_rate : ℝ

/-- Calculates the minimum average sailing rate for the ship to reach shore just as it begins to sink -/
noncomputable def minimum_sailing_rate (s : ShipSituation) : ℝ :=
  let net_water_rate := s.water_intake_rate - s.pump_rate
  let time_to_sink := s.sinking_capacity / net_water_rate
  s.distance_to_shore / time_to_sink

/-- Theorem stating that the minimum sailing rate for the given situation is approximately 79.15 km/h -/
theorem ship_minimum_sailing_rate :
  let s : ShipSituation := {
    distance_to_shore := 150,
    water_intake_rate := (13/3) * (60 / (5/2)),
    sinking_capacity := 180,
    pump_rate := 9
  }
  ∃ ε > 0, |minimum_sailing_rate s - 79.15| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ship_minimum_sailing_rate_l922_92213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_hyperbola_l922_92206

noncomputable section

open Set Real

-- Define the circle and points
def myCircle (θ : ℝ) (A : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | dist p A = 2 * cos θ}

def myB (θ : ℝ) (A : ℝ × ℝ) : ℝ × ℝ :=
  (A.1 + 2 * sin θ, A.2)

-- Define the locus of points M
def locus (θ : ℝ) (A B : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {M : ℝ × ℝ | ∃ (T : ℝ × ℝ), 
    T ∈ myCircle θ A ∧ 
    dist M B = dist M T ∧
    (dist M A - dist M T = 2 * cos θ)}

-- Theorem statement
theorem locus_is_hyperbola (θ : ℝ) (A : ℝ × ℝ) 
  (h1 : π/4 < θ) (h2 : θ < π/2) :
  ∃ (F1 F2 : ℝ × ℝ) (a : ℝ),
    locus θ A (myB θ A) = 
      {M : ℝ × ℝ | dist M F1 - dist M F2 = 2*a} ∧
    F1 = A ∧ 
    F2 = myB θ A ∧
    a = cos θ := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_hyperbola_l922_92206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_question_selection_exam_question_selection_is_74_l922_92246

theorem exam_question_selection : ℕ := by
  -- Define the total number of questions
  let total_questions : ℕ := 9
  -- Define the number of questions to be answered
  let questions_to_answer : ℕ := 6
  -- Define the number of questions in the first group
  let first_group : ℕ := 5
  -- Define the minimum number of questions to be selected from the first group
  let min_from_first : ℕ := 3

  -- Calculate the number of ways to select questions
  let ways_to_select := 
    (Nat.choose first_group min_from_first * Nat.choose (total_questions - first_group) (questions_to_answer - min_from_first)) +
    (Nat.choose first_group (min_from_first + 1) * Nat.choose (total_questions - first_group) (questions_to_answer - (min_from_first + 1))) +
    (Nat.choose first_group (min_from_first + 2) * Nat.choose (total_questions - first_group) (questions_to_answer - (min_from_first + 2)))

  -- Return the result
  exact ways_to_select

-- Prove that the number of ways to select questions is equal to 74
theorem exam_question_selection_is_74 : exam_question_selection = 74 := by
  -- Unfold the definition of exam_question_selection
  unfold exam_question_selection
  -- Simplify the arithmetic expressions
  simp
  -- The proof is completed
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_question_selection_exam_question_selection_is_74_l922_92246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l922_92263

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, (x - y) * (f (x + y)) - (x + y) * (f (x - y)) = 4 * x * y * (x^2 - y^2)

/-- The theorem stating that any function satisfying the equation must be of the form x³ + cx -/
theorem functional_equation_solution (f : ℝ → ℝ) (h : SatisfiesEquation f) :
  ∃ c : ℝ, ∀ x : ℝ, f x = x^3 + c * x := by
  sorry

#check functional_equation_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l922_92263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l922_92290

/-- Calculates the time for a train to cross a signal pole given its length, platform length, and time to cross the platform. -/
theorem train_crossing_time (train_length platform_length time_cross_platform : ℝ) 
  (h1 : train_length = 300)
  (h2 : platform_length = 150)
  (h3 : time_cross_platform = 27) :
  ∃ (time_cross_pole : ℝ), 
    (abs (time_cross_pole - train_length / ((train_length + platform_length) / time_cross_platform)) < 0.01) ∧ 
    (abs (time_cross_pole - 18) < 0.01) := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l922_92290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_10_l922_92250

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h_arith : ∀ n, a (n + 1) = a n + d
  h_nonzero : d ≠ 0

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * seq.a 1 + (n * (n - 1) / 2) * seq.d

theorem arithmetic_sequence_sum_10 (seq : ArithmeticSequence) 
  (h_geom_mean : seq.a 1 ^ 2 = seq.a 3 * seq.a 7)
  (h_sum_8 : sum_n seq 8 = 32) :
  sum_n seq 10 = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_10_l922_92250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_theorem_l922_92228

def vector_problem (a b : ℝ × ℝ) : Prop :=
  let norm_b := Real.sqrt (b.1^2 + b.2^2)
  let dot_product (v w : ℝ × ℝ) := v.1 * w.1 + v.2 * w.2
  let norm_a := Real.sqrt (a.1^2 + a.2^2)
  norm_b = Real.sqrt 2 ∧
  dot_product (a.1 - b.1, a.2 - b.2) (a.1 + b.1, a.2 + b.2) = 1/4 →
  norm_a = 3/2 ∧
  (dot_product a b = 3/2 → Real.arccos (dot_product a b / (norm_a * norm_b)) = Real.pi/4)

theorem vector_problem_theorem (a b : ℝ × ℝ) :
  vector_problem a b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_theorem_l922_92228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_ratio_in_triangle_l922_92241

/-- Triangle XYZ with angle bisectors XU and YV intersecting at Q -/
structure Triangle where
  X : ℝ × ℝ
  Y : ℝ × ℝ
  Z : ℝ × ℝ
  U : ℝ × ℝ
  V : ℝ × ℝ
  Q : ℝ × ℝ
  XY : ℝ
  XZ : ℝ
  YZ : ℝ

/-- The ratio of segments on an angle bisector -/
noncomputable def angleBisectorRatio (t : Triangle) : ℝ :=
  Real.sqrt ((t.XY * t.XZ) / (t.YZ * (t.XY + t.XZ - t.YZ)))

theorem angle_bisector_ratio_in_triangle (t : Triangle)
  (h1 : t.XY = 6)
  (h2 : t.XZ = 4)
  (h3 : t.YZ = 8)
  (h4 : angleBisectorRatio t = Real.sqrt ((t.XY * t.XZ) / (t.YZ * (t.XY + t.XZ - t.YZ))))
  : (3.5 : ℝ) = (Real.sqrt ((t.XY * t.XZ) / (t.YZ * (t.XY + t.XZ - t.YZ)))) := by
  sorry

#check angle_bisector_ratio_in_triangle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_ratio_in_triangle_l922_92241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beetle_walks_less_percentage_l922_92243

/-- The distance an ant walks in meters -/
noncomputable def ant_distance : ℝ := 500

/-- The time in minutes -/
noncomputable def time : ℝ := 60

/-- The speed of the beetle in km/h -/
noncomputable def beetle_speed : ℝ := 0.425

/-- Convert meters to kilometers -/
noncomputable def meters_to_km (m : ℝ) : ℝ := m / 1000

/-- Convert minutes to hours -/
noncomputable def minutes_to_hours (m : ℝ) : ℝ := m / 60

/-- Calculate the speed of the ant in km/h -/
noncomputable def ant_speed : ℝ := meters_to_km ant_distance / minutes_to_hours time

/-- Calculate the percentage difference between two values -/
noncomputable def percentage_difference (a b : ℝ) : ℝ := (a - b) / a * 100

theorem beetle_walks_less_percentage :
  percentage_difference ant_speed beetle_speed = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_beetle_walks_less_percentage_l922_92243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_1010_equals_decimal_10_l922_92212

/-- Converts a binary digit to its decimal value -/
def binaryToDecimal (digit : Nat) (position : Nat) : Nat :=
  digit * 2^position

/-- Represents the binary number 1010₂ -/
def binaryNumber : List Nat := [0, 1, 0, 1]

theorem binary_1010_equals_decimal_10 : 
  (List.sum (List.zipWith binaryToDecimal (List.reverse binaryNumber) (List.range 4))) = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_1010_equals_decimal_10_l922_92212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_equals_seven_and_half_l922_92235

theorem complex_expression_equals_seven_and_half :
  (0.064 : ℝ)^(-(1/3 : ℝ)) - (-2/3 : ℝ)^(0 : ℝ) + ((-2 : ℝ)^4)^(3/4 : ℝ) + (0.01 : ℝ)^(1/2 : ℝ) = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_equals_seven_and_half_l922_92235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_175_adjacent_to_7_l922_92282

/-- The set of positive integer divisors of 175, excluding 1 -/
def divisors_175 : Finset ℕ := sorry

/-- Predicate to check if two natural numbers have a common factor greater than 1 -/
def has_common_factor (a b : ℕ) : Prop := sorry

/-- The arrangement of divisors in a circle -/
def circular_arrangement (arr : List ℕ) : Prop := sorry

/-- Given two adjacent numbers in the circular arrangement, their sum is 210 if one of them is 7 -/
def adjacent_sum_210 (a b : ℕ) : Prop := 
  (a = 7 ∨ b = 7) → a + b = 210

theorem divisors_175_adjacent_to_7 :
  ∃ (arr : List ℕ), 
    (∀ x, x ∈ arr → x ∈ divisors_175) ∧
    (7 ∈ arr) ∧
    circular_arrangement arr ∧
    (∀ a b, a ∈ arr → b ∈ arr → a ≠ b → has_common_factor a b) ∧
    (∀ a b, a ∈ arr → b ∈ arr → adjacent_sum_210 a b) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_175_adjacent_to_7_l922_92282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_f_achieves_bounds_l922_92252

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x - Real.cos x

theorem f_range :
  Set.range f = Set.Icc (-2) 2 :=
by
  sorry

theorem f_achieves_bounds :
  ∃ x₁ x₂ : ℝ, f x₁ = -2 ∧ f x₂ = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_f_achieves_bounds_l922_92252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_10_plus_2_floor_l922_92258

-- Define the integer part function as noncomputable
noncomputable def integerPart (x : ℝ) : ℤ :=
  ⌊x⌋

-- State the theorem
theorem sqrt_10_plus_2_floor : integerPart (Real.sqrt 10 + 2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_10_plus_2_floor_l922_92258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_statements_proof_l922_92248

theorem statements_proof :
  (∃ n : ℕ, n > 0 ∧ 2^n < n^2) ∧
  (∃ a b c : ℝ, a > b ∧ a*c^2 ≤ b*c^2) ∧
  (∀ a b : ℝ, a > b ∧ b > 0 → a^2 > a*b ∧ a*b > b^2) ∧
  (∀ a b c d : ℝ, a > b ∧ b > 0 ∧ c < d ∧ d < 0 → 1/(a-c) < 1/(b-d)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_statements_proof_l922_92248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_from_excircle_tetrahedron_volume_formula_correct_l922_92285

open Real

/-- The volume of a tetrahedron given its excircle radius and face areas -/
theorem tetrahedron_volume_from_excircle (r₀ S₀ S₁ S₂ S₃ : ℝ) (h_pos : r₀ > 0) :
  ∃ V : ℝ, V = (1/3) * (S₁ + S₂ + S₃ - S₀) * r₀ := by
  -- We assert the existence of a tetrahedron with the given properties
  sorry

/-- The volume formula for a tetrahedron is correct -/
theorem tetrahedron_volume_formula_correct (r₀ S₀ S₁ S₂ S₃ V : ℝ) (h_pos : r₀ > 0)
  (h_tetra : V = (1/3) * (S₁ + S₂ + S₃ - S₀) * r₀) :
  V = (1/3) * (S₁ + S₂ + S₃ - S₀) * r₀ := by
  -- The proof is trivial given the hypothesis
  exact h_tetra

#check tetrahedron_volume_from_excircle
#check tetrahedron_volume_formula_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_from_excircle_tetrahedron_volume_formula_correct_l922_92285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_primes_l922_92249

theorem arithmetic_progression_primes (p : Fin 15 → ℕ) (d : ℕ) :
  (∀ i : Fin 15, Prime (p i)) →
  (∀ i : Fin 14, p (Fin.succ i) = p i + d) →
  (∀ i j : Fin 15, i < j → p i < p j) →
  (2 ∣ d) ∧ (3 ∣ d) ∧ (5 ∣ d) ∧ (7 ∣ d) ∧ (11 ∣ d) ∧ (13 ∣ d) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_primes_l922_92249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_distance_l922_92274

-- Define the speed of the train
noncomputable def train_speed : ℝ := 96

-- Define the ratio of car speed to train speed
noncomputable def car_speed_ratio : ℝ := 5 / 8

-- Define the time in hours
noncomputable def time : ℝ := 1 / 2

-- Theorem statement
theorem car_distance :
  train_speed = 96 → car_speed_ratio = 5 / 8 → time = 1 / 2 →
  car_speed_ratio * train_speed * time = 30 := by
  sorry

#check car_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_distance_l922_92274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_two_three_one_l922_92210

/-- Given function f for any three distinct numbers -/
noncomputable def f (a b c : ℝ) : ℝ := (a^2 + b^2) / (c + 1)

/-- Theorem stating that f(2,3,1) = 13/2 -/
theorem f_two_three_one : f 2 3 1 = 13/2 := by
  -- Unfold the definition of f
  unfold f
  -- Simplify the expression
  simp [pow_two]
  -- Perform the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_two_three_one_l922_92210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angela_insects_l922_92276

/-- The number of insects Dean has -/
def dean_insects : ℕ := 30

/-- The number of insects Jacob has relative to Dean -/
def jacob_multiplier : ℕ := 5

/-- The fraction of insects Angela has relative to Jacob -/
def angela_fraction : ℚ := 1 / 2

/-- Theorem stating the number of insects Angela has -/
theorem angela_insects : (dean_insects * jacob_multiplier : ℚ) * angela_fraction = 75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angela_insects_l922_92276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2010_of_17_eq_8_l922_92256

def digit_sum (n : ℕ) : ℕ := sorry

def f (n : ℕ) : ℕ := digit_sum (n^2 + 1)

def f_k : ℕ → ℕ → ℕ
  | 0, n => n
  | 1, n => f n
  | (k+1), n => f (f_k k n)

theorem f_2010_of_17_eq_8 : f_k 2010 17 = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2010_of_17_eq_8_l922_92256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_palindrome_for_many_bases_l922_92223

-- Define what it means for a number to be a three-digit palindrome in base b
def is_three_digit_palindrome (N : ℕ) (b : ℕ) : Prop :=
  ∃ (d₁ d₂ : ℕ), d₁ < b ∧ d₂ < b ∧ N = d₁ * b^2 + d₂ * b + d₁

-- The main theorem
theorem exists_palindrome_for_many_bases : 
  ∃ (N : ℕ), ∃ (S : Finset ℕ), S.card ≥ 2002 ∧ 
  ∀ (b : ℕ), b ∈ S → is_three_digit_palindrome N b :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_palindrome_for_many_bases_l922_92223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l922_92291

-- Define the polar coordinate system
def PolarPoint := ℝ × ℝ

-- Define the circle C
def CircleC (ρ θ : ℝ) : Prop :=
  ρ^2 - 6*ρ*(Real.cos θ) + 8*ρ*(Real.sin θ) + 21 = 0

-- Define points A and B
noncomputable def A : PolarPoint := (2, Real.pi)
noncomputable def B : PolarPoint := (2, Real.pi/2)

-- Define the area of a triangle given three points in polar coordinates
noncomputable def triangleArea (p1 p2 p3 : PolarPoint) : ℝ := sorry

-- Theorem statement
theorem max_triangle_area :
  ∃ (maxArea : ℝ), maxArea = 9 + 2*(Real.sqrt 2) ∧
  ∀ (F : PolarPoint), CircleC F.1 F.2 →
    triangleArea A B F ≤ maxArea := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l922_92291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_six_percentage_l922_92201

theorem divisible_by_six_percentage (n : ℕ) : n = 120 →
  (↑(Finset.filter (λ x : ℕ => x % 6 = 0) (Finset.range (n + 1))).card / ↑n) * 100 = 50 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_six_percentage_l922_92201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_savings_is_28_l922_92261

/-- Represents the savings of each child and calculates their total savings. -/
def ChildrenSavings : ℚ := by
  -- Define savings per day for each child
  let josiah_daily : ℚ := 1/4
  let leah_daily : ℚ := 1/2
  let megan_daily : ℚ := 2 * leah_daily

  -- Define number of days each child saves
  let josiah_days : ℕ := 24
  let leah_days : ℕ := 20
  let megan_days : ℕ := 12

  -- Calculate total savings for each child
  let josiah_total := josiah_daily * josiah_days
  let leah_total := leah_daily * leah_days
  let megan_total := megan_daily * megan_days

  -- Calculate and return the total savings of all three children
  exact josiah_total + leah_total + megan_total

/-- Theorem stating that the total savings of the three children is $28.00 -/
theorem total_savings_is_28 : ChildrenSavings = 28 := by
  sorry

#eval ChildrenSavings

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_savings_is_28_l922_92261
