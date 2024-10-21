import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l897_89747

/-- Represents a train with its length and time to cross a telegraph post -/
structure Train where
  length : ℝ
  crossTime : ℝ

/-- Calculates the speed of a train -/
noncomputable def trainSpeed (t : Train) : ℝ := t.length / t.crossTime

/-- Represents the problem setup -/
structure TrainProblem where
  train1 : Train
  train2 : Train
  inclineReduction : ℝ

/-- Calculates the time for two trains to cross each other -/
noncomputable def crossingTime (p : TrainProblem) : ℝ :=
  let v1 := trainSpeed p.train1
  let v2 := trainSpeed p.train2 * (1 - p.inclineReduction)
  let totalLength := p.train1.length + p.train2.length
  totalLength / (v1 + v2)

/-- The main theorem stating the crossing time for the given problem -/
theorem train_crossing_time : 
  let problem := TrainProblem.mk 
    (Train.mk 150 12) 
    (Train.mk 180 18) 
    0.1
  ∃ (ε : ℝ), abs (crossingTime problem - 15.35) < ε ∧ ε < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l897_89747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_number_of_irreducible_fractions_l897_89789

theorem even_number_of_irreducible_fractions (n : ℕ) (h : n > 2) :
  Even (Finset.card (Finset.filter (fun k => Nat.gcd k n = 1) (Finset.range (n - 1)))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_number_of_irreducible_fractions_l897_89789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_measure_l897_89719

-- Define the triangle ABC
def triangle_ABC (a b c : ℝ) : Prop :=
  a = 2 ∧ b = 1 ∧ c > 2 * Real.sqrt 2

-- Define the angle C in terms of side lengths using the law of cosines
noncomputable def angle_C (a b c : ℝ) : ℝ :=
  Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))

-- Define the condition for x
def x_condition (x : ℝ) (a b c : ℝ) : Prop :=
  angle_C a b c < x ∧ 
  ∀ y, y < x → angle_C a b c ≥ y

-- Main theorem
theorem smallest_angle_measure :
  ∀ a b c : ℝ, triangle_ABC a b c →
  ∃ x : ℝ, x_condition x a b c ∧ x = 140 * π / 180 := by
  sorry

#check smallest_angle_measure

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_measure_l897_89719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salted_duck_eggs_theorem_l897_89793

/-- Prices of salted duck eggs -/
structure Prices where
  a : ℕ  -- price of brand A
  b : ℕ  -- price of brand B
deriving Repr

/-- Find prices satisfying the given conditions -/
def find_prices : Prices :=
  ⟨30, 20⟩

/-- Calculate the total cost for a given number of boxes -/
def total_cost (p : Prices) (x : ℕ) : ℕ :=
  p.a * x + p.b * (30 - x)

/-- Find the minimum cost satisfying the constraints -/
def min_cost (p : Prices) : ℕ :=
  let costs := [18, 19, 20].map (total_cost p)
  costs.foldl min (costs.head!)

/-- Main theorem -/
theorem salted_duck_eggs_theorem (p : Prices) :
  p = find_prices →
  9 * p.a + 6 * p.b = 390 ∧
  5 * p.a + 8 * p.b = 310 ∧
  min_cost p = 780 := by
  sorry

#eval find_prices
#eval min_cost find_prices

end NUMINAMATH_CALUDE_ERRORFEEDBACK_salted_duck_eggs_theorem_l897_89793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_properties_l897_89745

noncomputable def binomial_expansion (x : ℝ) := (Real.sqrt x - 1 / (2 * x)) ^ 6

theorem binomial_expansion_properties (x : ℝ) (h : x > 0) :
  let expansion := binomial_expansion x
  -- The constant term is 15/4
  ∃ (c : ℝ), c = 15/4 ∧
  -- The sum of the coefficients of all terms is 64
  (binomial_expansion 1 = 64) ∧
  -- The binomial coefficient of the 4th term is the largest
  ∀ (i : Fin 7), i.val + 1 ≠ 4 → Nat.choose 6 i.val ≤ Nat.choose 6 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_properties_l897_89745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_csc_315_degrees_l897_89725

theorem csc_315_degrees : Real.cos (315 * π / 180) / Real.sin (315 * π / 180) = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_csc_315_degrees_l897_89725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l897_89701

-- Part 1
def f1 (x : ℝ) := |2*x - 1| - |x + 2|

def M : Set ℝ := {x | f1 x > 2}

theorem part1 : M = Set.Ioi 5 ∪ Set.Iic (-1) := by sorry

-- Part 2
def f2 (a x : ℝ) := |x - 1| - |x + 2*a^2|

theorem part2 : (∀ x, f2 a x < -3*a) → a ∈ Set.Ioo (-1) (-1/2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l897_89701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_inequality_l897_89787

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - 4*x + a * Real.log x

theorem extreme_points_inequality (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   (∀ x : ℝ, x > 0 → (deriv (f a)) x = 0 → x = x₁ ∨ x = x₂) ∧
   (∀ t : ℝ, f a x₁ + f a x₂ ≥ x₁ + x₂ + t)) →
  (∀ t : ℝ, f a x₁ + f a x₂ ≥ x₁ + x₂ + t → t ≤ -13) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_inequality_l897_89787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_proof_l897_89720

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space defined by two points -/
structure Line3D where
  point1 : Point3D
  point2 : Point3D

/-- A plane in 3D space defined by a normal vector and a point -/
structure Plane3D where
  normal : Point3D
  point : Point3D

/-- The projection of a point onto the xy-plane -/
def project (p : Point3D) : Point3D :=
  { x := p.x, y := p.y, z := 0 }

/-- Check if a point lies on a line -/
noncomputable def pointOnLine (p : Point3D) (l : Line3D) : Prop := sorry

/-- Check if a point lies on a plane -/
noncomputable def pointOnPlane (p : Point3D) (pl : Plane3D) : Prop := sorry

/-- The intersection point of two lines -/
noncomputable def lineIntersection (l1 l2 : Line3D) : Point3D := sorry

/-- The intersection of a line and a plane -/
noncomputable def linePlaneIntersection (l : Line3D) (pl : Plane3D) : Point3D := sorry

theorem intersection_point_proof 
  (M N : Point3D) 
  (M' N' : Point3D)
  (B : Point3D)
  (b : Line3D)
  (l : Line3D)
  (β : Plane3D)
  (h1 : l = Line3D.mk M N)
  (h2 : M' = project M)
  (h3 : N' = project N)
  (h4 : β.point = B)
  (h5 : pointOnPlane (project B) β) :
  let P := lineIntersection (Line3D.mk M' N') b
  linePlaneIntersection l β = P := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_proof_l897_89720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_of_CH_prime_l897_89702

/-- Given two points C and H in a 2D plane, and a translation vector,
    this function returns the midpoint of the translated segment C'H'. -/
noncomputable def midpoint_of_translated_segment (C H : ℝ × ℝ) (translation : ℝ × ℝ) : ℝ × ℝ :=
  let midpoint := ((C.1 + H.1) / 2, (C.2 + H.2) / 2)
  (midpoint.1 - translation.1, midpoint.2 + translation.2)

/-- Theorem stating that for the given points and translation,
    the midpoint of the translated segment C'H' is (0, 5). -/
theorem midpoint_of_CH_prime (C H : ℝ × ℝ) (translation : ℝ × ℝ)
    (hC : C = (2, 2))
    (hH : H = (6, 2))
    (hT : translation = (-4, 3)) :
  midpoint_of_translated_segment C H translation = (0, 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_of_CH_prime_l897_89702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monic_quartic_with_given_roots_l897_89743

noncomputable def p (x : ℂ) : ℂ := x^4 - 10*x^3 + 14*x^2 + 14*x - 2

theorem monic_quartic_with_given_roots :
  (∀ x, p x = 0 → x = 3 + Real.sqrt 7 ∨ x = 2 - Real.sqrt 5 ∨ x = 3 - Real.sqrt 7 ∨ x = 2 + Real.sqrt 5) ∧
  (p (3 + Real.sqrt 7) = 0) ∧
  (p (2 - Real.sqrt 5) = 0) ∧
  (∀ x, p x = x^4 - 10*x^3 + 14*x^2 + 14*x - 2) ∧
  (∀ a b c d e : ℂ, p = (fun x => a*x^4 + b*x^3 + c*x^2 + d*x + e) → a = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monic_quartic_with_given_roots_l897_89743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_specific_l897_89740

/-- The area of a triangle with two sides of length 19 and one side of length 30 is 175. -/
theorem triangle_area_specific : ∃ (D E F : ℝ × ℝ),
  let de := Real.sqrt ((E.1 - D.1)^2 + (E.2 - D.2)^2)
  let ef := Real.sqrt ((F.1 - E.1)^2 + (F.2 - E.2)^2)
  let df := Real.sqrt ((F.1 - D.1)^2 + (F.2 - D.2)^2)
  let s := (de + ef + df) / 2
  let area := Real.sqrt (s * (s - de) * (s - ef) * (s - df))
  de = 19 ∧ ef = 19 ∧ df = 30 ∧ area = 175 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_specific_l897_89740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_set_l897_89770

theorem average_of_set (T : Finset ℕ) (hT : T.Nonempty) : 
  (∃ (m : ℕ), m ∈ T ∧ ∀ x ∈ T, x ≤ m) →
  (∃ (n : ℕ), n ∈ T ∧ ∀ x ∈ T, n ≤ x) →
  (((T.sum (λ x => (x : ℚ)) - m) : ℚ) / ((T.card - 1) : ℚ) = 29) →
  (((T.sum (λ x => (x : ℚ)) - m - n) : ℚ) / ((T.card - 2) : ℚ) = 33) →
  (((T.sum (λ x => (x : ℚ)) - n) : ℚ) / ((T.card - 1) : ℚ) = 38) →
  (m - n = 90) →
  ((T.sum (λ x => (x : ℚ))) / (T.card : ℚ) = 679 / 20) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_set_l897_89770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_m_eq_neg_two_l897_89784

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_iff_equal_slopes {m₁ m₂ : ℝ} : 
  m₁ = m₂ ↔ ∀ (x y₁ y₂ : ℝ), y₁ = m₁ * x + 1 ∧ y₂ = m₂ * x + 1 → y₁ = y₂

/-- The slope of line l₁ -/
noncomputable def slope_l1 (m : ℝ) : ℝ := -(m + 1)

/-- The slope of line l₂ -/
noncomputable def slope_l2 (m : ℝ) : ℝ := -2 / m

/-- The statement to prove -/
theorem parallel_lines_m_eq_neg_two (m : ℝ) :
  m ≠ 0 → (slope_l1 m = slope_l2 m) → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_m_eq_neg_two_l897_89784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l897_89786

-- Define the circle
def my_circle (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1

-- Define the point through which the line passes
def point : ℝ × ℝ := (3, 4)

-- Define the vertical line
def vertical_line (x : ℝ) : Prop := x = 3

-- Define the non-vertical line
def non_vertical_line (x y : ℝ) : Prop := 15 * x - 8 * y - 13 = 0

-- Theorem statement
theorem tangent_line_equation :
  ∃ (x y : ℝ), (vertical_line x ∨ non_vertical_line x y) ∧
  (x, y) = point ∧
  (∀ (a b : ℝ), my_circle a b → (x - a)^2 + (y - b)^2 ≥ 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l897_89786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_on_line_l897_89737

/-- The constraint line -/
def constraint_line (x y : ℝ) : Prop := 6 * x + 8 * y = 48

/-- The distance function from origin to a point (x, y) -/
noncomputable def distance (x y : ℝ) : ℝ := Real.sqrt (x^2 + y^2)

/-- The theorem statement -/
theorem max_distance_on_line :
  ∃ (max : ℝ), max = 8 ∧
  ∀ (x y : ℝ), x ≥ 0 → constraint_line x y →
  distance x y ≤ max := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_on_line_l897_89737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_count_proof_l897_89757

/-- Represents the number of students from School A -/
def x : ℕ := sorry

/-- Represents the number of students from School B -/
def y : ℕ := sorry

/-- The capacity of the first type of bus -/
def bus_capacity_1 : ℕ := 14

/-- The capacity of the second type of bus -/
def bus_capacity_2 : ℕ := 19

/-- The total number of buses needed when using the first type -/
def total_buses_1 : ℕ := 72

/-- The difference in number of buses needed by School B compared to School A when using the second type -/
def bus_difference : ℕ := 7

theorem student_count_proof :
  (x % 10 = 0) ∧ 
  (y % 10 = 0) ∧
  ((x + y) / bus_capacity_1 = total_buses_1) ∧
  (y / bus_capacity_2 = x / bus_capacity_2 + bus_difference) →
  x = 437 ∧ y = 570 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_count_proof_l897_89757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l897_89755

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (Real.sin (Real.pi * x)) / ((x^2 + 1) * (x^2 - 2*x + 2))

-- Theorem statement
theorem f_properties :
  (∃ (M m : ℝ), ∀ x, m ≤ f x ∧ f x ≤ M) ∧ 
  (∀ x : ℝ, x ∈ Set.univ) ∧
  (∃ a : ℝ, ∀ x : ℝ, f (a + x) = f (a - x)) := by
  sorry

#check f_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l897_89755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_divisors_between_2_and_100_l897_89704

theorem no_divisors_between_2_and_100 (n : ℕ+) 
  (h : ∀ k ∈ Finset.range 99, (Finset.sum (Finset.range n) (λ i => (i + 1)^(k + 1))) % n = 0) : 
  ∀ d ∈ Finset.range 99, d > 1 → ¬(d ∣ ↑n) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_divisors_between_2_and_100_l897_89704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_specific_planes_l897_89763

/-- The distance between two parallel planes in 3D space -/
noncomputable def distance_between_planes (a₁ b₁ c₁ d₁ a₂ b₂ c₂ d₂ : ℝ) : ℝ :=
  abs (a₂ * 1 + b₂ * 0 + c₂ * 0 + d₂) / Real.sqrt (a₂^2 + b₂^2 + c₂^2)

/-- Theorem: The distance between the planes 3x - y + 2z - 3 = 0 and 6x - 2y + 4z + 4 = 0 is 5√14 / 14 -/
theorem distance_specific_planes :
  distance_between_planes 3 (-1) 2 (-3) 6 (-2) 4 4 = 5 * Real.sqrt 14 / 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_specific_planes_l897_89763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l897_89715

-- Define the triangle
structure Triangle where
  A : Real
  B : Real
  C : Real
  AB : Real
  BC : Real
  AC : Real
  area : Real

-- Define the conditions
def isValidTriangle (t : Triangle) : Prop :=
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.A + t.B + t.C = Real.pi ∧
  t.AB > 0 ∧ t.BC > 0 ∧ t.AC > 0

def isAcute (t : Triangle) : Prop :=
  t.A < Real.pi/2 ∧ t.B < Real.pi/2 ∧ t.C < Real.pi/2

-- State the theorem
theorem triangle_side_length (t : Triangle) 
  (h1 : isValidTriangle t)
  (h2 : isAcute t)
  (h3 : Real.cos t.A = 1/3)
  (h4 : t.AC = Real.sqrt 3)
  (h5 : t.area = Real.sqrt 2) :
  t.BC = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l897_89715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_v_19_formula_l897_89721

noncomputable def v (n : ℕ) (b : ℝ) : ℝ :=
  match n with
  | 0 => b  -- Add case for 0
  | 1 => b
  | n + 1 => -1 / (v n b + 2)

theorem v_19_formula (b : ℝ) (h : b > 0) : v 19 b = -(b + 2) / (b + 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_v_19_formula_l897_89721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_pyramid_base_length_l897_89761

/-- Represents a right pyramid with a square base -/
structure RightPyramid where
  baseLength : ℝ
  slantHeight : ℝ

/-- Calculates the area of one lateral face of the pyramid -/
noncomputable def lateralFaceArea (p : RightPyramid) : ℝ :=
  (1/2) * p.baseLength * p.slantHeight

theorem right_pyramid_base_length 
  (p : RightPyramid) 
  (h1 : p.slantHeight = 40) 
  (h2 : lateralFaceArea p = 160) : 
  p.baseLength = 8 := by
  sorry

#eval "Compilation successful!"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_pyramid_base_length_l897_89761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_M_N_l897_89774

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x, y = 1 - Real.sin x}
def N : Set ℝ := {x | ∃ y, y = Real.log (2 - x)}

-- State the theorem
theorem union_M_N : M ∪ N = Set.Iic 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_M_N_l897_89774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_m_value_l897_89772

/-- A function f: ℝ → ℝ is a power function if there exists a real number a such that f(x) = x^a for all x > 0 -/
def IsPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, x > 0 → f x = x ^ a

/-- The function y = (m+2)x^(m-1) -/
noncomputable def f (m : ℝ) : ℝ → ℝ := fun x ↦ (m + 2) * (x ^ (m - 1))

theorem power_function_m_value (m : ℝ) :
  IsPowerFunction (f m) → m = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_m_value_l897_89772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_of_circle_line_intersection_l897_89768

noncomputable def circle_center : ℝ × ℝ := (0, 2)
noncomputable def circle_radius : ℝ := 3
noncomputable def line_slope : ℝ := Real.sqrt 3

theorem chord_length_of_circle_line_intersection :
  let circle_eq := fun (x y : ℝ) => x^2 + (y - circle_center.2)^2 = circle_radius^2
  let line_eq := fun (x y : ℝ) => y = line_slope * x
  let d := |circle_center.2| / Real.sqrt (1 + line_slope^2)
  ∃ (A B : ℝ × ℝ),
    circle_eq A.1 A.2 ∧ circle_eq B.1 B.2 ∧
    line_eq A.1 A.2 ∧ line_eq B.1 B.2 ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt (circle_radius^2 - d^2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_of_circle_line_intersection_l897_89768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_challenge_solvable_l897_89708

/-- Represents a figure that can be stuck on a captive's forehead -/
structure Figure where
  id : Nat

/-- Represents a captive in the group -/
structure Captive where
  id : Nat
  figure : Figure

/-- The challenge setup -/
structure Challenge where
  captives : List Captive
  figures : List Figure

def Challenge.isValid (c : Challenge) : Prop :=
  c.captives.length ≥ 3 ∧
  ∀ f₁ f₂ : Figure, f₁ ≠ f₂ →
    (c.captives.filter (λ cap => cap.figure.id = f₁.id)).length ≠
    (c.captives.filter (λ cap => cap.figure.id = f₂.id)).length

def Captive.canSee (c : Captive) (others : List Captive) : List Figure :=
  (others.filter (λ other => other.id ≠ c.id)).map (λ other => other.figure)

def Strategy := Challenge → Captive → Figure

theorem challenge_solvable (c : Challenge) (h : c.isValid) :
  ∃ (s : Strategy), ∃ (cap : Captive), cap ∈ c.captives ∧ s c cap = cap.figure :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_challenge_solvable_l897_89708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_comparison_l897_89730

noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (n : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate / n) ^ (n * time)

noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

theorem loan_comparison : 
  let principal := (12000 : ℝ)
  let compound_rate := (0.08 : ℝ)
  let simple_rate := (0.09 : ℝ)
  let total_time := (15 : ℝ)
  let partial_time := (7 : ℝ)
  let compound_freq := (12 : ℝ)

  let compound_balance_7y := compound_interest principal compound_rate compound_freq partial_time
  let payment_7y := compound_balance_7y / 3
  let remaining_balance := compound_balance_7y - payment_7y
  let final_compound_balance := compound_interest remaining_balance compound_rate compound_freq (total_time - partial_time)
  let total_compound_repayment := payment_7y + final_compound_balance

  let total_simple_repayment := simple_interest principal simple_rate total_time

  let difference := total_simple_repayment - total_compound_repayment

  ∃ ε > 0, |difference - 2335| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_comparison_l897_89730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carriage_problem_equivalence_l897_89739

/-- Represents the number of people -/
def x : ℕ := sorry

/-- Represents the number of carriages -/
def y : ℕ := sorry

/-- Condition: If 3 people ride in one carriage, then 2 carriages are empty -/
axiom condition1 : x = 3 * (y - 2)

/-- Condition: If 2 people ride in one carriage, then 9 people have to walk -/
axiom condition2 : 2 * y = x - 9

/-- Theorem: The system of equations is equivalent to the given conditions -/
theorem carriage_problem_equivalence : 
  (x = 3 * (y - 2) ∧ 2 * y = x - 9) ↔ 
  (∃ (people carriages : ℕ), 
    (people = 3 * (carriages - 2)) ∧ 
    (2 * carriages + 9 = people)) :=
by
  sorry

#check carriage_problem_equivalence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carriage_problem_equivalence_l897_89739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_three_connections_l897_89729

/-- Represents a connection matrix for a system of 7 elements -/
def ConnectionMatrix := Matrix (Fin 7) (Fin 7) Bool

/-- A valid connection matrix is symmetric and has zeros on the diagonal -/
def is_valid_connection_matrix (m : ConnectionMatrix) : Prop :=
  (∀ i j, m i j = m j i) ∧ (∀ i, m i i = false)

/-- Each element is connected to exactly three others -/
def has_three_connections (m : ConnectionMatrix) : Prop :=
  ∀ i, (Finset.univ.filter (λ j => m i j)).card = 3

theorem impossible_three_connections :
  ¬∃ (m : ConnectionMatrix), is_valid_connection_matrix m ∧ has_three_connections m :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_three_connections_l897_89729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_any_figure_has_symmetry_axis_l897_89765

/-- A plane figure --/
structure PlaneFigure where
  -- We don't need to define the internal structure of a plane figure for this problem

/-- Represents a line in a plane --/
structure Line where
  -- We don't need to define the internal structure of a line for this problem

/-- Defines what it means for a line to be an axis of symmetry for a plane figure --/
def has_axis_of_symmetry (f : PlaneFigure) (l : Line) : Prop :=
  -- If the figure is folded along the line, the two sides completely overlap
  sorry -- We leave this undefined for now

/-- The statement that any plane figure has an axis of symmetry --/
def any_figure_has_symmetry_axis : Prop :=
  ∀ f : PlaneFigure, ∃ l : Line, has_axis_of_symmetry f l

/-- Theorem stating that it's false that any plane figure has an axis of symmetry --/
theorem not_any_figure_has_symmetry_axis : ¬any_figure_has_symmetry_axis := by
  sorry

#check not_any_figure_has_symmetry_axis

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_any_figure_has_symmetry_axis_l897_89765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_dist_squared_range_l897_89709

-- Define the circle
def is_on_circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define points A and B
def A : ℝ × ℝ := (-3, 0)
def B : ℝ × ℝ := (4, 2)

-- Define the distance function
def dist_squared (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

-- Define the sum of squared distances function
def sum_dist_squared (x y : ℝ) : ℝ :=
  dist_squared (x, y) A + dist_squared (x, y) B

-- State the theorem
theorem sum_dist_squared_range :
  ∀ x y : ℝ, is_on_circle x y →
    37 - 4 * Real.sqrt 5 ≤ sum_dist_squared x y ∧
    sum_dist_squared x y ≤ 37 + 4 * Real.sqrt 5 :=
by
  sorry

#check sum_dist_squared_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_dist_squared_range_l897_89709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_about_pi_8_l897_89762

noncomputable def f (x : ℝ) := Real.sin (2 * x + Real.pi / 4)

theorem f_symmetry_about_pi_8 : 
  ∀ (x : ℝ), f (π / 8 + x) = f (π / 8 - x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_about_pi_8_l897_89762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solid_angle_dihedral_theorem_solid_angle_polyhedral_theorem_l897_89717

/-- The solid angle of a dihedral angle -/
def solid_angle_dihedral (α : ℝ) : ℝ := 2 * α

/-- The solid angle of a polyhedral angle -/
noncomputable def solid_angle_polyhedral (σ : ℝ) (n : ℕ) : ℝ := σ - (n - 2 : ℝ) * Real.pi

/-- Theorem: The solid angle of a dihedral angle is 2α -/
theorem solid_angle_dihedral_theorem (α : ℝ) :
  solid_angle_dihedral α = 2 * α := by
  -- Proof goes here
  sorry

/-- Theorem: The solid angle of a polyhedral angle is σ - (n-2)π -/
theorem solid_angle_polyhedral_theorem (σ : ℝ) (n : ℕ) :
  solid_angle_polyhedral σ n = σ - (n - 2 : ℝ) * Real.pi := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solid_angle_dihedral_theorem_solid_angle_polyhedral_theorem_l897_89717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_problem_l897_89780

noncomputable section

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define the focus of the parabola
def focus (p : ℝ) : ℝ × ℝ := (p/2, 0)

-- Define a point on the parabola
def point_on_parabola (p : ℝ) (M : ℝ × ℝ) : Prop :=
  parabola p M.1 M.2

-- Define the circumcircle of triangle OFM
def circumcircle (O F M : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {P : ℝ × ℝ | ∃ r, (P.1 - O.1)^2 + (P.2 - O.2)^2 = r^2 ∧
                   (P.1 - F.1)^2 + (P.2 - F.2)^2 = r^2 ∧
                   (P.1 - M.1)^2 + (P.2 - M.2)^2 = r^2}

-- Define the directrix of the parabola
def directrix (p : ℝ) : Set (ℝ × ℝ) := {P : ℝ × ℝ | P.1 = -p/2}

-- Define the tangency condition
def is_tangent (circle : Set (ℝ × ℝ)) (line : Set (ℝ × ℝ)) : Prop :=
  ∃ P, P ∈ circle ∧ P ∈ line ∧ ∀ Q, Q ∈ circle ∧ Q ∈ line → Q = P

-- Define the area of a circle
def circle_area (r : ℝ) : ℝ := Real.pi * r^2

-- The main theorem
theorem parabola_problem (p : ℝ) (M : ℝ × ℝ) :
  parabola p M.1 M.2 →
  let F := focus p
  let O := (0, 0)
  let circle := circumcircle O F M
  is_tangent circle (directrix p) →
  (∃ r, circle_area r = 9*Real.pi ∧ ∀ P ∈ circle, (P.1 - O.1)^2 + (P.2 - O.2)^2 = r^2) →
  p = 4 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_problem_l897_89780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_hyperbola_circle_l897_89733

-- Define the hyperbola
def hyperbola (x y b : ℝ) : Prop := x^2 - y^2/b^2 = 1

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + (y-1)^2 = 1

-- Define the eccentricity
noncomputable def eccentricity : ℝ := Real.sqrt 5 / 2

-- Define the theorem
theorem min_distance_hyperbola_circle :
  ∀ (b : ℝ) (F1 P Q : ℝ × ℝ),
  (∃ (x y : ℝ), P = (x, y) ∧ hyperbola x y b) →  -- P is on the hyperbola
  (∃ (x y : ℝ), Q = (x, y) ∧ circle_eq x y) →    -- Q is on the circle
  (F1.1 < 0) →                                   -- F1 is on the left side
  (P.1 > 0) →                                    -- P is on the right branch
  (∀ (P' Q' : ℝ × ℝ),
    (∃ (x y : ℝ), P' = (x, y) ∧ hyperbola x y b) →
    (∃ (x y : ℝ), Q' = (x, y) ∧ circle_eq x y) →
    (P'.1 > 0) →
    dist P Q + dist P F1 ≤ dist P' Q' + dist P' F1) →
  dist P Q + dist P F1 = 5/2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_hyperbola_circle_l897_89733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_cylinder_sphere_volume_relation_l897_89764

noncomputable def cone_volume (h : ℝ) (r : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

noncomputable def cylinder_volume (h : ℝ) (r : ℝ) : ℝ := Real.pi * r^2 * h

noncomputable def sphere_volume (r : ℝ) : ℝ := (4/3) * Real.pi * r^3

theorem cone_cylinder_sphere_volume_relation (h : ℝ) (h_pos : h > 0) :
  cone_volume h h + cylinder_volume h h = sphere_volume h := by
  -- Expand the definitions
  unfold cone_volume cylinder_volume sphere_volume
  -- Simplify the left-hand side
  simp [Real.pi]
  -- The proof steps would go here, but for now we'll use sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_cylinder_sphere_volume_relation_l897_89764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l897_89728

def sequenceA (n : ℕ) : ℚ :=
  match n with
  | 0 => 1
  | n + 1 => sequenceA n + 1 / ((n + 1) * n)

theorem sequence_formula (n : ℕ) (h : n > 0) : sequenceA n = (2 * n - 1) / n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l897_89728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_height_in_cylinder_l897_89742

/-- Proves that the height of water in a cylinder after transferring from a cone is 2 cm --/
theorem water_height_in_cylinder (r_cone h_cone r_cylinder : ℝ) 
  (hr_cone : r_cone = 15)
  (hh_cone : h_cone = 24)
  (hr_cylinder : r_cylinder = 30) : 
  (1/3) * Real.pi * r_cone^2 * h_cone / (Real.pi * r_cylinder^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_height_in_cylinder_l897_89742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_universal_destination_l897_89751

structure City where
  id : Nat

def Flight := City → City → Prop

structure CityNetwork where
  cities : Set City
  flights : Flight
  flight_unique : ∀ a b : City, flights a b → ¬ flights b a
  exists_common_source : ∀ p q : City, ∃ r : City, 
    (∃ path : List City, path.head? = some r ∧ path.getLast? = some p) ∧
    (∃ path : List City, path.head? = some r ∧ path.getLast? = some q)

theorem exists_universal_destination (network : CityNetwork) :
  ∃ a : City, ∀ b : City, b ∈ network.cities → 
    ∃ path : List City, path.head? = some b ∧ path.getLast? = some a :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_universal_destination_l897_89751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_and_b_for_all_real_roots_l897_89711

/-- The smallest positive real number a for which the polynomial
    x^3 - ax^2 + bx - 2a has all real roots, and the corresponding b value. -/
theorem smallest_a_and_b_for_all_real_roots :
  ∃ (a b : ℝ), a > 0 ∧
  (∀ x : ℝ, x^3 - a*x^2 + b*x - 2*a = 0 → x ∈ Set.univ) ∧
  (∀ a' : ℝ, a' > 0 ∧ a' < a →
    ∃ x : ℂ, x^3 - a'*x^2 + b*x - 2*a' = 0 ∧ x ∉ Set.univ) ∧
  a = 3 * (27 : ℝ)^(1/4) ∧
  b = 3 * (54 : ℝ)^(1/4) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_and_b_for_all_real_roots_l897_89711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_top_block_l897_89705

/-- Represents the modified pyramid structure --/
structure ModifiedPyramid :=
  (base : Fin 10 → ℕ)
  (layer2 : Fin 4 → ℕ)
  (layer3 : Fin 2 → ℕ)
  (top : ℕ)

/-- The base layer contains numbers 1 through 10 in any order --/
def valid_base (p : ModifiedPyramid) : Prop :=
  Finset.univ.image p.base = Finset.range 10

/-- Each block in layer 2 is the sum of numbers from blocks it rests on in layer 1 --/
def valid_layer2 (p : ModifiedPyramid) : Prop :=
  ∀ i : Fin 4, ∃ s : Finset (Fin 10), p.layer2 i = s.sum (λ j => p.base j)

/-- Each block in layer 3 is the sum of numbers from blocks it rests on in layer 2 --/
def valid_layer3 (p : ModifiedPyramid) : Prop :=
  ∀ i : Fin 2, ∃ s : Finset (Fin 4), p.layer3 i = s.sum (λ j => p.layer2 j)

/-- The top block is the sum of numbers from blocks it rests on in layer 3 --/
def valid_top (p : ModifiedPyramid) : Prop :=
  p.top = Finset.univ.sum (λ i => p.layer3 i)

/-- A pyramid is valid if all layers satisfy their respective conditions --/
def valid_pyramid (p : ModifiedPyramid) : Prop :=
  valid_base p ∧ valid_layer2 p ∧ valid_layer3 p ∧ valid_top p

/-- The main theorem: The smallest possible number for the top block is 55 --/
theorem smallest_top_block (p : ModifiedPyramid) (h : valid_pyramid p) : 
  p.top ≥ 55 := by
  sorry

#check smallest_top_block

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_top_block_l897_89705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l897_89738

theorem calculation_proof :
  (Real.sqrt 4 * Real.sqrt 25 - Real.sqrt ((-3)^2) = 7) ∧
  ((-1/2)^2 + (8 : Real)^(1/3) - abs (1 - Real.sqrt 9) = 1/4) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l897_89738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_implies_m_eq_one_second_quadrant_implies_m_range_min_abs_z_in_second_quadrant_l897_89752

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := Complex.mk (m - 1) (2 * m + 1)

-- Theorem 1: If z is a pure imaginary number, then m = 1
theorem pure_imaginary_implies_m_eq_one (m : ℝ) :
  z m = Complex.I * Complex.im (z m) → m = 1 := by sorry

-- Theorem 2a: If z is in the second quadrant, then -1/2 < m < 1
theorem second_quadrant_implies_m_range (m : ℝ) :
  Complex.re (z m) < 0 ∧ Complex.im (z m) > 0 → -1/2 < m ∧ m < 1 := by sorry

-- Theorem 2b: If z is in the second quadrant, the minimum value of |z| is (3√5)/5
theorem min_abs_z_in_second_quadrant :
  ∃ (m : ℝ), -1/2 < m ∧ m < 1 ∧
  ∀ (n : ℝ), -1/2 < n ∧ n < 1 → Complex.abs (z m) ≤ Complex.abs (z n) ∧
  Complex.abs (z m) = 3 * Real.sqrt 5 / 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_implies_m_eq_one_second_quadrant_implies_m_range_min_abs_z_in_second_quadrant_l897_89752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_point_on_y_axis_l897_89779

/-- Given points A and B in the plane, find the point P on the y-axis that minimizes the sum of distances |PA| + |PB| -/
theorem min_distance_point_on_y_axis (A B : ℝ × ℝ) (h1 : A = (2, 5)) (h2 : B = (4, -1)) :
  ∃ P : ℝ × ℝ, P.1 = 0 ∧ (∀ Q : ℝ × ℝ, Q.1 = 0 → 
    dist P A + dist P B ≤ dist Q A + dist Q B) → P = (0, 3) := by
  sorry

/-- The Euclidean distance between two points in the plane -/
noncomputable def dist (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_point_on_y_axis_l897_89779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_airline_route_within_republic_l897_89713

/-- Represents a city in the country -/
structure City where
  id : Nat
  republic : Nat
  routes : Nat

/-- The country with its cities and airline routes -/
structure Country where
  cities : Finset City
  routes : Finset (City × City)

/-- Theorem: In a country with 100 cities divided into three republics, 
    where at least 70 cities have at least 70 airline routes originating from them, 
    there must exist at least one airline route that connects two cities within the same republic -/
theorem airline_route_within_republic (country : Country) : 
  (country.cities.card = 100) →
  (∀ c ∈ country.cities, c.republic ∈ ({1, 2, 3} : Finset Nat)) →
  (country.cities.filter (λ c => c.routes ≥ 70)).card ≥ 70 →
  ∃ (c1 c2 : City), c1 ∈ country.cities ∧ c2 ∈ country.cities ∧ 
                    c1.republic = c2.republic ∧ 
                    (c1, c2) ∈ country.routes :=
by
  intro h_total h_republics h_routes
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_airline_route_within_republic_l897_89713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_arithmetic_sequence_l897_89736

noncomputable def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

noncomputable def S_n (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := n / 2 * (2 * a₁ + (n - 1 : ℝ) * d)

theorem min_sum_arithmetic_sequence :
  ∃ (d : ℝ),
    arithmetic_sequence (-26) d 8 + arithmetic_sequence (-26) d 13 = 5 ∧
    ∀ (n : ℕ), n ≠ 0 → S_n (-26) d n ≥ S_n (-26) d 9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_arithmetic_sequence_l897_89736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_constants_l897_89766

/-- The function y in terms of x, a, and b -/
noncomputable def y (x a b : ℝ) : ℝ := a + b / (x^2 + 1)

/-- Theorem stating that if y(2) = 3 and y(1) = 2, then a + b = 1/3 -/
theorem sum_of_constants (a b : ℝ) : 
  y 2 a b = 3 → y 1 a b = 2 → a + b = 1/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_constants_l897_89766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cartesian_equation_C_max_sum_xy_coord_max_sum_xy_l897_89759

-- Define the circle C in polar coordinates
noncomputable def circle_C (θ : ℝ) : ℝ × ℝ := (6 * Real.cos θ + 8 * Real.sin θ, θ)

-- Define the Cartesian coordinates of a point on circle C
noncomputable def cartesian_point (θ : ℝ) : ℝ × ℝ :=
  let (ρ, _) := circle_C θ
  (ρ * Real.cos θ, ρ * Real.sin θ)

-- Theorem stating the Cartesian equation of circle C
theorem cartesian_equation_C :
  ∀ (x y : ℝ), (∃ θ, cartesian_point θ = (x, y)) ↔ (x - 3)^2 + (y - 4)^2 = 25 := by sorry

-- Theorem for the maximum value of x+y on circle C
theorem max_sum_xy :
  (⨆ (θ : ℝ), let (x, y) := cartesian_point θ; x + y) = 7 + 5 * Real.sqrt 2 := by sorry

-- Theorem for the coordinates of P when x+y is maximum
theorem coord_max_sum_xy :
  ∃ θ, let (x, y) := cartesian_point θ
       x + y = 7 + 5 * Real.sqrt 2 ∧ 
       x = 3 + (5 * Real.sqrt 2) / 2 ∧ 
       y = 4 + (5 * Real.sqrt 2) / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cartesian_equation_C_max_sum_xy_coord_max_sum_xy_l897_89759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_equation_l897_89756

theorem tan_alpha_equation (α : Real) (m : Real) : 
  (Real.tan α = m / 3) → 
  (Real.tan (α + π / 4) = 2 / m) → 
  (m = -6 ∨ m = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_equation_l897_89756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l897_89773

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the angle functions
noncomputable def angleA (t : Triangle) : ℝ := sorry
noncomputable def angleC (t : Triangle) : ℝ := sorry

-- Define the properties of the triangle
def isValidTriangle (t : Triangle) (a b : ℕ) : Prop :=
  let AB := Real.sqrt (a : ℝ) - b
  let AC := 6
  let BC := 8
  angleA t = 2 * angleC t ∧
  -- Other necessary conditions for a valid triangle
  AB > 0 ∧ AC > 0 ∧ BC > 0 ∧
  AB + AC > BC ∧ AB + BC > AC ∧ AC + BC > AB

-- The main theorem
theorem triangle_property (t : Triangle) (a b : ℕ) :
  isValidTriangle t a b → 100 * a + b = 7303 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l897_89773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_value_l897_89754

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - (3/2) * x^2

-- State the theorem
theorem find_a_value (a : ℝ) :
  (∀ x, f a x ≤ 1/6) ∧
  (∀ x ∈ Set.Icc (1/4 : ℝ) (1/2 : ℝ), f a x ≥ 1/8) →
  a = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_value_l897_89754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_digit_integers_with_consecutive_ones_l897_89749

def my_sequence : ℕ → ℕ
  | 0 => 2
  | 1 => 3
  | n + 2 => my_sequence n + my_sequence (n + 1)

theorem nine_digit_integers_with_consecutive_ones : 
  (2^9 : ℕ) - my_sequence 8 = 423 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_digit_integers_with_consecutive_ones_l897_89749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tram_passenger_problem_l897_89700

theorem tram_passenger_problem (b₀ g₀ : ℕ) : 
  let b₁ := b₀ + g₀ / 3
  let g₁ := 2 * g₀ / 3
  let b₂ := 2 * b₁ / 3
  let g₂ := g₁ + b₁ / 3
  (g₂ = b₂ + 2 ∧ b₂ = g₀) → b₀ = 14 ∧ g₀ = 12 := by
  intro h
  sorry

#check tram_passenger_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tram_passenger_problem_l897_89700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pastry_selection_theorem_l897_89795

/-- Represents a pastry type -/
structure PastryType where
  id : Nat
  deriving BEq, Repr

/-- Represents a box of pastries -/
structure Box where
  pastries : List PastryType

/-- Represents a distribution of pastries -/
structure Distribution where
  boxes : List Box
  num_types : Nat
  pastries_per_type : Nat

/-- Main theorem statement -/
theorem pastry_selection_theorem (d : Distribution) 
  (h1 : d.boxes.length = d.num_types)
  (h2 : ∀ b ∈ d.boxes, b.pastries.length = d.pastries_per_type)
  (h3 : ∀ t : PastryType, (d.boxes.map (λ b => b.pastries.count t)).sum = d.pastries_per_type) :
  ∃ (selection : List PastryType), 
    selection.length = d.num_types ∧ 
    selection.Nodup ∧
    ∀ (b : Box), b ∈ d.boxes → ∃ (p : PastryType), p ∈ b.pastries ∧ p ∈ selection := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pastry_selection_theorem_l897_89795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_curves_a_value_l897_89769

-- Define the two curves
noncomputable def curve1 (x : ℝ) : ℝ := x + Real.log x
def curve2 (a x : ℝ) : ℝ := a * x^2 + (a + 2) * x + 1

-- Define the tangent point
def tangent_point : ℝ × ℝ := (1, 1)

-- State the theorem
theorem tangent_curves_a_value :
  ∃ (a : ℝ), 
    (curve1 (tangent_point.fst) = curve2 a (tangent_point.fst)) ∧
    (curve1 (tangent_point.snd) = curve2 a (tangent_point.snd)) ∧
    (∀ (x : ℝ), x ≠ tangent_point.fst → curve1 x ≠ curve2 a x) →
  a = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_curves_a_value_l897_89769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_half_angle_l897_89799

theorem cos_half_angle (α : ℝ) (h1 : Real.sin α = 4/5) (h2 : 0 < α ∧ α < π/2) :
  Real.cos (α/2) = 2*Real.sqrt 5/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_half_angle_l897_89799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_price_after_discounts_l897_89714

theorem final_price_after_discounts (A R1 R2 R3 : ℝ) 
  (h1 : A = 100) (h2 : R1 = 20) (h3 : R2 = 10) (h4 : R3 = 5) :
  A * (1 - R1 / 100) * (1 - R2 / 100) * (1 - R3 / 100) = 68.40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_price_after_discounts_l897_89714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_player_a_winning_strategy_l897_89760

/-- Represents a player in the game -/
inductive Player
| A
| B
deriving Repr, DecidableEq

/-- Represents a move direction -/
inductive MoveDirection
| Horizontal
| Vertical

/-- Represents the game state -/
structure GameState where
  board : Fin 1994 → Fin 1994 → Bool
  current_player : Player
  current_position : Fin 1994 × Fin 1994

/-- Represents a valid move in the game -/
def ValidMove (gs : GameState) (new_pos : Fin 1994 × Fin 1994) : Prop :=
  match gs.current_player with
  | Player.A => new_pos.1 = gs.current_position.1 ∧ new_pos.2 ≠ gs.current_position.2
  | Player.B => new_pos.2 = gs.current_position.2 ∧ new_pos.1 ≠ gs.current_position.1

/-- Represents the game rules -/
def GameRules (gs : GameState) : Prop :=
  ∀ (pos : Fin 1994 × Fin 1994),
    ValidMove gs pos → ¬gs.board pos.1 pos.2

/-- Represents a winning strategy for a player -/
def WinningStrategy (player : Player) : Prop :=
  ∀ (gs : GameState),
    gs.current_player = player →
    GameRules gs →
    ∃ (move : Fin 1994 × Fin 1994),
      ValidMove gs move ∧
      ¬∃ (opponent_move : Fin 1994 × Fin 1994),
        ValidMove { gs with
          current_player := if player = Player.A then Player.B else Player.A,
          current_position := move
        } opponent_move

/-- The main theorem stating that Player A has a winning strategy -/
theorem player_a_winning_strategy :
  WinningStrategy Player.A := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_player_a_winning_strategy_l897_89760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_store_pricing_theorem_l897_89797

/-- Represents the pricing options for suits and ties -/
structure PricingOption where
  suit_price : ℚ
  tie_price : ℚ
  free_ties : ℕ
  discount : ℚ

/-- Calculates the cost for a given pricing option -/
def calculate_cost (option : PricingOption) (suits : ℕ) (ties : ℕ) : ℚ :=
  option.suit_price * suits + option.tie_price * (ties - min suits ties * option.free_ties) * option.discount

/-- Theorem stating the cost comparison and existence of a more cost-effective method -/
theorem store_pricing_theorem (x : ℕ) (h_x : x > 20) :
  let option1 : PricingOption := ⟨400, 80, 1, 1⟩
  let option2 : PricingOption := ⟨400, 80, 0, 9/10⟩
  let cost1 := calculate_cost option1 20 x
  let cost2 := calculate_cost option2 20 x
  (x = 30 → cost1 < cost2) ∧
  ∃ (better_cost : ℚ), x = 30 → better_cost < min cost1 cost2 ∧ better_cost = 8720 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_store_pricing_theorem_l897_89797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_free_fall_velocity_equality_l897_89703

noncomputable section

/-- Motion equation of a ball in free fall -/
def s (g : ℝ) (t : ℝ) : ℝ := (1/2) * g * t^2

/-- Average velocity from t=1 to t=3 -/
def v_bar (g : ℝ) : ℝ := (s g 3 - s g 1) / (3 - 1)

/-- Instantaneous velocity at t=2 -/
def v_2 (g : ℝ) : ℝ := (deriv (s g)) 2

theorem free_fall_velocity_equality (g : ℝ) :
  v_bar g = v_2 g := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_free_fall_velocity_equality_l897_89703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l897_89727

/-- The area of a trapezium with given dimensions -/
noncomputable def trapeziumArea (a b h : ℝ) : ℝ := (a + b) * h / 2

/-- Theorem: The area of a trapezium with parallel sides of lengths 20 cm and 18 cm, 
    and a distance of 15 cm between them, is equal to 285 square centimeters. -/
theorem trapezium_area_example : trapeziumArea 20 18 15 = 285 := by
  -- Unfold the definition of trapeziumArea
  unfold trapeziumArea
  -- Perform the calculation
  simp [mul_add, mul_div_assoc]
  -- The result should now be obvious to Lean
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l897_89727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_construction_condition_l897_89796

/-- Parallel chords in a circle -/
def parallel_chords (a b c : ℝ) : Prop := sorry

/-- b is equidistant from a and c -/
def equidistant (b a c : ℝ) : Prop := sorry

/-- The circle is constructible given chords a, b, and c -/
def constructible_circle (a b c : ℝ) : Prop := sorry

/-- Given parallel chords a, b, and c in a circle, where b is equidistant from a and c,
    the construction of the circle is possible if and only if b² > (a² + c²) / 2 -/
theorem circle_construction_condition (a b c : ℝ) 
  (h_parallel : parallel_chords a b c) 
  (h_equidistant : equidistant b a c) : 
  constructible_circle a b c ↔ b^2 > (a^2 + c^2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_construction_condition_l897_89796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volume_from_unfolded_surface_l897_89734

/-- The volume of a cylinder given its unfolded side surface dimensions -/
theorem cylinder_volume_from_unfolded_surface (length width : ℝ) :
  length > 0 → width > 0 →
  length = 12 ∧ width = 8 →
  ∃ (r h : ℝ), 
    ((2 * π * r = length ∧ h = width) ∨ (2 * π * r = width ∧ h = length)) ∧
    (π * r^2 * h = 288 / π ∨ π * r^2 * h = 192 / π) :=
by
  intro hl hw hlw
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volume_from_unfolded_surface_l897_89734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_number_with_remainder_two_l897_89794

theorem least_number_with_remainder_two : ∃! n : ℕ, 
  n > 1 ∧ 
  (∀ d ∈ ({2, 3, 4, 5, 6} : Finset ℕ), n % d = 2) ∧
  (∀ m : ℕ, m > 1 ∧ (∀ d ∈ ({2, 3, 4, 5, 6} : Finset ℕ), m % d = 2) → n ≤ m) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_number_with_remainder_two_l897_89794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_AMBN_l897_89710

-- Define the coordinate system and necessary components
variable (α : Real) -- Angle between line l and x-axis
variable (θ : Real) -- Polar angle
variable (ρ : Real) -- Polar radius
variable (t : Real) -- Parameter for line l

-- Define line l
noncomputable def line_l (α : Real) (t : Real) : Real × Real :=
  (1 + t * Real.cos α, t * Real.sin α)

-- Define curve C in polar coordinates
def curve_C (ρ θ : Real) : Prop :=
  ρ * Real.sin θ * Real.sin θ - 4 * Real.cos θ = 0

-- Define curve C in Cartesian coordinates
def curve_C_cartesian (x y : Real) : Prop :=
  y * y = 4 * x

-- Define the area of quadrilateral AMBN
noncomputable def area_AMBN (α : Real) : Real :=
  32 / (Real.sin (2 * α))^2

-- State the theorem
theorem min_area_AMBN :
  ∃ (α : Real), ∀ (β : Real), area_AMBN α ≤ area_AMBN β ∧ area_AMBN α = 32 := by
  sorry

#check min_area_AMBN

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_AMBN_l897_89710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_ten_l897_89798

/-- Represents the dimensions of the squares in the figure -/
structure SquareDimensions where
  large : ℝ
  small : ℝ

/-- Calculates the area of the shaded region given the dimensions of the squares -/
noncomputable def shadedArea (d : SquareDimensions) : ℝ :=
  d.small * d.small - (d.small * d.large) / (d.large + d.small) * d.small / 2

/-- Theorem stating that the shaded area is 10 square inches for the given dimensions -/
theorem shaded_area_is_ten (d : SquareDimensions) 
    (h1 : d.large = 12) 
    (h2 : d.small = 4) : 
  shadedArea d = 10 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval shadedArea { large := 12, small := 4 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_ten_l897_89798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_june_turnover_calculation_l897_89771

/-- Calculates the turnover for June given the turnovers for April and May,
    assuming a constant monthly growth rate. -/
noncomputable def june_turnover (april_turnover may_turnover : ℝ) : ℝ :=
  let growth_rate := (may_turnover - april_turnover) / april_turnover
  may_turnover * (1 + growth_rate)

/-- Theorem stating that given the turnover in April is 10 thousand yuan
    and in May is 12 thousand yuan, the turnover in June is 14.4 thousand yuan,
    assuming a constant monthly growth rate. -/
theorem june_turnover_calculation :
  june_turnover 10 12 = 14.4 := by
  -- Unfold the definition of june_turnover
  unfold june_turnover
  -- Simplify the expression
  simp
  -- The proof is completed numerically
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_june_turnover_calculation_l897_89771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_vector_sum_magnitude_l897_89791

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.cos x
noncomputable def g (x : ℝ) : ℝ := Real.tan x

-- Define the interval [0, 2π]
def I : Set ℝ := Set.Icc 0 (2 * Real.pi)

-- Define the intersection points M and N
def M : ℝ × ℝ := sorry
def N : ℝ × ℝ := sorry

-- State the theorem
theorem intersection_vector_sum_magnitude :
  (∀ x ∈ I, f x = g x → (x, f x) = M ∨ (x, f x) = N) →
  M.1 ∈ I ∧ N.1 ∈ I →
  Real.sqrt ((M.1 + N.1 - 0)^2 + (M.2 + N.2 - 0)^2) = Real.pi :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_vector_sum_magnitude_l897_89791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_target_function_coefficient_l897_89776

/-- The target function -/
def f (a : ℝ) (x : ℝ) : ℝ := a * (x + 2)^2 + 2

/-- The logarithmic equation condition -/
def log_condition (x y : ℝ) : Prop :=
  (x - x^2 + 3) ^ (y - 6) = 
    (x - x^2 + 3) ^ ((abs (2*x + 6) - abs (2*x + 3)) / (3*x + 7.5) * Real.sqrt (x^2 + 5*x + 6.25))

/-- The right angle condition for tangents -/
def right_angle_condition (a x₀ y₀ : ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    (deriv (f a) x₁) * (deriv (f a) x₂) = -1 ∧
    y₀ = f a x₀ ∧
    (y₀ - f a x₁) / (x₀ - x₁) = deriv (f a) x₁ ∧
    (y₀ - f a x₂) / (x₀ - x₂) = deriv (f a) x₂

/-- The main theorem -/
theorem target_function_coefficient (x₀ y₀ : ℝ) :
  log_condition x₀ y₀ →
  (∃ a : ℝ, right_angle_condition a x₀ y₀) →
  ∃ a : ℝ, a = -0.15625 ∧ right_angle_condition a x₀ y₀ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_target_function_coefficient_l897_89776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_waitress_income_ratio_l897_89777

/-- Represents the income structure of a waitress -/
structure WaitressIncome where
  salary : ℝ
  tips : ℝ

/-- The fraction of income from tips -/
noncomputable def tipFraction (w : WaitressIncome) : ℝ :=
  w.tips / (w.salary + w.tips)

/-- The ratio of tips to salary -/
noncomputable def tipToSalaryRatio (w : WaitressIncome) : ℝ :=
  w.tips / w.salary

theorem waitress_income_ratio (w : WaitressIncome) 
  (h : tipFraction w = 0.6923076923076923) : 
  tipToSalaryRatio w = 2.25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_waitress_income_ratio_l897_89777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simultaneous_work_time_team_a_participation_days_l897_89718

-- Define the time taken by each team to complete the pipeline
noncomputable def team_a_time : ℝ := 12
noncomputable def team_b_time : ℝ := 24

-- Define the function to calculate time when both teams work simultaneously
noncomputable def simultaneous_time (ta tb : ℝ) : ℝ :=
  1 / (1/ta + 1/tb)

-- Define the function to calculate total days Team A participated
noncomputable def team_a_total_days (a : ℝ) : ℝ :=
  (a + 3) + (4 * a + 2)

-- Theorem for Part 1
theorem simultaneous_work_time :
  simultaneous_time team_a_time team_b_time = 8 := by sorry

-- Theorem for Part 2
theorem team_a_participation_days :
  team_a_total_days 1 = 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simultaneous_work_time_team_a_participation_days_l897_89718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_line_intersection_l897_89767

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Parallelogram ABCD -/
structure Parallelogram where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Check if a point is on a line -/
def on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- The main theorem -/
theorem parallelogram_line_intersection
  (ABCD : Parallelogram)
  (l : Line)
  (A1 B1 C1 : Point)
  (h1 : on_line ABCD.D l)
  (h2 : ¬on_line ABCD.A l ∧ ¬on_line ABCD.B l ∧ ¬on_line ABCD.C l)
  (h3 : on_line A1 l ∧ on_line B1 l ∧ on_line C1 l)
  (h4 : ∃ l1 l2 l3, parallel l1 l ∧ parallel l2 l ∧ parallel l3 l ∧
                    on_line ABCD.A l1 ∧ on_line A1 l1 ∧
                    on_line ABCD.B l2 ∧ on_line B1 l2 ∧
                    on_line ABCD.C l3 ∧ on_line C1 l3) :
  distance ABCD.B B1 = distance ABCD.A A1 + distance ABCD.C C1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_line_intersection_l897_89767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_parallelogram_l897_89712

/-- Definition of the ellipse C -/
noncomputable def ellipse (x y : ℝ) : Prop :=
  x^2 / 6 + y^2 / 2 = 1

/-- The left focus of the ellipse -/
def leftFocus : ℝ × ℝ := (-2, 0)

/-- The eccentricity of the ellipse -/
noncomputable def eccentricity : ℝ := Real.sqrt 6 / 3

/-- Point T on the line x = -3 -/
def T (m : ℝ) : ℝ × ℝ := (-3, m)

/-- Predicate to check if OPTQ forms a parallelogram -/
def isParallelogram (P Q : ℝ × ℝ) (m : ℝ) : Prop :=
  P.1 + Q.1 = -3 ∧ P.2 + Q.2 = m

/-- Predicate to check if PQ is perpendicular to TF -/
def isPerpendicular (P Q : ℝ × ℝ) (m : ℝ) : Prop :=
  (Q.2 - P.2) * (T m).1 - leftFocus.1 = (Q.1 - P.1) * ((T m).2 - leftFocus.2)

/-- The main theorem -/
theorem area_of_parallelogram (P Q : ℝ × ℝ) (m : ℝ) :
  ellipse P.1 P.2 ∧ 
  ellipse Q.1 Q.2 ∧ 
  isParallelogram P Q m ∧ 
  isPerpendicular P Q m →
  |Q.2 - P.2| * 4 = 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_parallelogram_l897_89712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l897_89731

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

def equation (x : ℝ) : Prop := x^2 - 8 * (floor x) + 7 = 0

theorem equation_solutions :
  ∀ x : ℝ, equation x ↔ x = 1 ∨ x = Real.sqrt 33 ∨ x = Real.sqrt 41 ∨ x = 7 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l897_89731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_product_l897_89706

/-- Two lines in 2D space -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Distance between two parallel lines -/
noncomputable def distance_parallel_lines (l1 l2 : Line2D) : ℝ :=
  abs (l1.c - l2.c) / Real.sqrt (l1.a^2 + l1.b^2)

theorem parallel_lines_product (a b : ℝ) :
  let l1 : Line2D := ⟨a, 2, -b⟩
  let l2 : Line2D := ⟨a - 1, 1, -b⟩
  (l1.a / l1.b = l2.a / l2.b) →  -- parallel condition
  (distance_parallel_lines l1 l2 = Real.sqrt 2 / 2) →
  (a * b = 4 ∨ a * b = -4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_product_l897_89706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_five_pairs_l897_89722

def is_valid_pair (a b : ℤ) : Prop :=
  0 ≤ a ∧ a ≤ 10 ∧ 0 ≤ b ∧ b ≤ 10

def triangle_area (x1 y1 x2 y2 x3 y3 : ℤ) : ℚ :=
  (1/2 : ℚ) * |x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2)|

def satisfies_area_condition (a b : ℤ) : Prop :=
  triangle_area 1 1 4 5 a b = 6

def valid_pairs : List (ℤ × ℤ) :=
  [(4,1), (7,5), (10,9), (1,5), (4,9)]

theorem exactly_five_pairs :
  (∀ p ∈ valid_pairs, is_valid_pair p.1 p.2 ∧ satisfies_area_condition p.1 p.2) ∧
  valid_pairs.length = 5 ∧
  (∀ a b, is_valid_pair a b ∧ satisfies_area_condition a b → (a, b) ∈ valid_pairs) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_five_pairs_l897_89722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l897_89785

/-- The equation of a parabola -/
noncomputable def parabola_equation (x : ℝ) : ℝ := (x^2 - 8*x + 16) / 8

/-- The directrix of the parabola -/
def directrix : ℝ := -2

/-- Theorem stating that the directrix of the given parabola is y = -2 -/
theorem parabola_directrix : 
  ∀ x y : ℝ, y = parabola_equation x → 
  ∃ p : ℝ, p > 0 ∧ 
  (∀ point : ℝ × ℝ, point.1 = x ∧ point.2 = y → 
    (point.1 - 4)^2 + (point.2 - 0)^2 = (point.2 - directrix)^2 + p^2) :=
by
  sorry

#check parabola_directrix

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l897_89785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l897_89783

noncomputable def angle_between (a b : ℝ → ℝ → ℝ → ℝ) : ℝ := 
  Real.arccos ((a 0 0 0 * b 0 0 0 + a 0 0 1 * b 0 0 1 + a 0 1 0 * b 0 1 0) / 
    (Real.sqrt (a 0 0 0^2 + a 0 0 1^2 + a 0 1 0^2) * 
     Real.sqrt (b 0 0 0^2 + b 0 0 1^2 + b 0 1 0^2)))

theorem vector_problem (a b : ℝ → ℝ → ℝ → ℝ) 
  (h_angle : angle_between a b = 2 * Real.pi / 3)
  (h_norm_a : Real.sqrt (a 0 0 0^2 + a 0 0 1^2 + a 0 1 0^2) = 1)
  (h_norm_b : Real.sqrt (b 0 0 0^2 + b 0 0 1^2 + b 0 1 0^2) = 2) :
  (a 0 0 0 * b 0 0 0 + a 0 0 1 * b 0 0 1 + a 0 1 0 * b 0 1 0 = -1) ∧ 
  (∃! t : ℝ, ((2 * a 0 0 0 - b 0 0 0) * (t * a 0 0 0 + b 0 0 0) + 
              (2 * a 0 0 1 - b 0 0 1) * (t * a 0 0 1 + b 0 0 1) + 
              (2 * a 0 1 0 - b 0 1 0) * (t * a 0 1 0 + b 0 1 0) = 0) ∧ 
              t = 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l897_89783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_line_l897_89723

/-- The line l in the problem -/
noncomputable def line_l (x y : ℝ) : Prop := x - y - 1 = 0

/-- The circle representing the locus of point P -/
noncomputable def circle_P (x y : ℝ) : Prop := x^2 + (y-1)^2 = 1

/-- The distance from a point (x,y) to the line l -/
noncomputable def distance_to_line (x y : ℝ) : ℝ := |x - y - 1| / Real.sqrt 2

theorem min_distance_to_line :
  ∀ x y : ℝ, circle_P x y → distance_to_line x y ≥ Real.sqrt 2 - 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_line_l897_89723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_primes_in_range_l897_89750

def is_prime_in_range (p : ℕ) : Prop :=
  50 ≤ p ∧ p ≤ 70 ∧ Nat.Prime p

instance : DecidablePred is_prime_in_range :=
  fun p => And.decidable

theorem sum_of_primes_in_range :
  (Finset.filter is_prime_in_range (Finset.range 71)).sum id = 240 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_primes_in_range_l897_89750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_area_l897_89707

-- Define points A and B
def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (3, 1)

-- Define lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := 3 * x - 2 * y + 3 = 0
def l₂ (x y : ℝ) : Prop := 2 * x - y + 2 = 0

-- Define point C as the intersection of l₁ and l₂
def C : ℝ × ℝ := (-1, 0)

-- State the theorem
theorem intersection_and_area :
  (l₁ C.1 C.2 ∧ l₂ C.1 C.2) ∧  -- C is on both lines
  (∃ S : ℝ, S = 5 ∧ 
    S = (1/2) * Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) * 
      (|C.1 + C.2 - 4| / Real.sqrt 2)) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_area_l897_89707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_linear_combination_l897_89746

open Matrix

variable (M : Matrix (Fin 2) (Fin 2) ℝ)
variable (v w : Vec ℝ 2)

theorem matrix_linear_combination 
  (h1 : M.vecMul v = ![2, -3])
  (h2 : M.vecMul w = ![-1, 4]) :
  M.vecMul (-3 • v + 2 • w) = ![-8, 17] := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_linear_combination_l897_89746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_implies_values_h_equals_g_implies_inequality_three_distinct_roots_implies_range_l897_89716

noncomputable section

-- Define the functions
def f (a b x : ℝ) : ℝ := x^2 - 2*a*x + b
def g (a x : ℝ) : ℝ := x - a
def h (a b x : ℝ) : ℝ := (f a b x + g a x - |f a b x - g a x|) / 2

-- Theorem 1
theorem range_implies_values (a b : ℝ) :
  (∀ y ∈ Set.Icc (-3 : ℝ) a, ∃ x ∈ Set.Icc (-3 : ℝ) a, f a b x = y) →
  a = -2 ∧ b = 1 := by
sorry

-- Theorem 2
theorem h_equals_g_implies_inequality (a b : ℝ) :
  (∀ x : ℝ, h a b x = g a x) →
  b - a^2 ≥ 1/4 := by
sorry

-- Theorem 3
theorem three_distinct_roots_implies_range (a : ℝ) :
  (∀ b ∈ Set.Icc (-1 : ℝ) 1, ∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    h a b x = a ∧ h a b y = a ∧ h a b z = a) →
  a < (-1 - Real.sqrt 5) / 2 := by
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_implies_values_h_equals_g_implies_inequality_three_distinct_roots_implies_range_l897_89716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_implies_sides_sine_condition_implies_area_l897_89724

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.c = 2 ∧ t.C = Real.pi/3

-- Theorem 1
theorem area_implies_sides (t : Triangle) : 
  triangle_conditions t → (1/2 * t.a * t.b * Real.sin t.C = Real.sqrt 3) → 
  t.a = 2 ∧ t.b = 2 := by
  sorry

-- Theorem 2
theorem sine_condition_implies_area (t : Triangle) :
  triangle_conditions t → 
  (Real.sin t.C + Real.sin (t.B - t.A) = 2 * Real.sin (2 * t.A)) →
  1/2 * t.a * t.b * Real.sin t.C = 2 * Real.sqrt 3 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_implies_sides_sine_condition_implies_area_l897_89724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_math_olympiad_recipes_l897_89753

/-- Calculates the number of full recipes needed for a school event --/
def recipes_needed (total_students : ℕ) (attendance_rate : ℚ) 
  (cookies_per_student : ℕ) (cookies_per_recipe : ℕ) : ℕ :=
  let expected_attendance := (total_students : ℚ) * attendance_rate
  let total_cookies_needed := expected_attendance * (cookies_per_student : ℚ)
  (total_cookies_needed / (cookies_per_recipe : ℚ)).ceil.toNat

/-- Proves that 15 full recipes are needed for the Math Olympiad discussion --/
theorem math_olympiad_recipes : 
  recipes_needed 150 (3/5) 3 18 = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_math_olympiad_recipes_l897_89753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shifted_function_symmetry_l897_89758

open Real

-- Define the original function f
noncomputable def f (x : ℝ) : ℝ := (sin (5 * π / 12 - x))^2 - (sin (x + π / 12))^2

-- Define the shifted function g
noncomputable def g (φ : ℝ) (x : ℝ) : ℝ := f (x + φ)

-- State the theorem
theorem shifted_function_symmetry (φ : ℝ) (h1 : φ > 0) 
  (h2 : ∀ x, g φ (π/6 - x) = g φ (π/6 + x)) : φ = π/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shifted_function_symmetry_l897_89758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_formed_l897_89726

-- Define the lines
def line1 (x : ℝ) : ℝ := 2 * x + 3
def line2 (x : ℝ) : ℝ := -2 * x + 3
def line3 : ℝ := 1
def line4 (x : ℝ) : ℝ := -2 * x - 2

-- Define the intersection points
def point1 : ℝ × ℝ := (0, 3)
def point2 : ℝ × ℝ := (-1, 1)
def point3 : ℝ × ℝ := (1, 1)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem isosceles_triangle_formed : 
  distance point1 point2 = distance point1 point3 ∧ 
  distance point1 point2 ≠ distance point2 point3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_formed_l897_89726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_monotonicity_intervals_max_value_on_interval_l897_89741

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2

-- Theorem for the tangent line when a = 1
theorem tangent_line_at_one :
  let f₁ := f 1
  let f₁' := λ x => 3*x^2 - 2*x
  (λ x y => x - y - 1 = 0) = (λ x y => y - f₁ 1 = f₁' 1 * (x - 1)) := by sorry

-- Theorem for monotonicity intervals
theorem monotonicity_intervals (a : ℝ) :
  (a = 0 → ∀ x y, x < y → f a x < f a y) ∧
  (a > 0 → (∀ x y, x < y ∧ y < 0 → f a x < f a y) ∧
           (∀ x y, 0 < x ∧ x < y ∧ y < 2*a/3 → f a x > f a y) ∧
           (∀ x y, 2*a/3 < x ∧ x < y → f a x < f a y)) ∧
  (a < 0 → (∀ x y, x < y ∧ y < 2*a/3 → f a x < f a y) ∧
           (∀ x y, 2*a/3 < x ∧ x < y ∧ y < 0 → f a x > f a y) ∧
           (∀ x y, 0 < x ∧ x < y → f a x < f a y)) := by sorry

-- Theorem for maximum value on [1,3]
theorem max_value_on_interval (a : ℝ) :
  let max_value := if a ≥ 13/4 then 1 - a else 27 - 9*a
  ∀ x, x ∈ Set.Icc 1 3 → f a x ≤ max_value := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_monotonicity_intervals_max_value_on_interval_l897_89741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_range_of_a_l897_89781

-- Define set A
def A : Set ℝ := {x | (1/2 : ℝ) ≤ (2 : ℝ)^x ∧ (2 : ℝ)^x ≤ 32}

-- Define set B (domain of y = lg(x^2 - 4))
def B : Set ℝ := {x | x^2 - 4 > 0}

-- Define set C
def C (a : ℝ) : Set ℝ := {x | x ≤ a - 1}

-- Theorem 1: A ∩ B = {x | 2 < x ≤ 5}
theorem intersection_A_B : A ∩ B = {x : ℝ | 2 < x ∧ x ≤ 5} := by sorry

-- Theorem 2: If A ⊆ C(a), then a ≥ 6
theorem range_of_a (a : ℝ) (h : A ⊆ C a) : a ≥ 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_range_of_a_l897_89781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_cream_cone_height_l897_89792

noncomputable def cone_volume (r : ℝ) (h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

noncomputable def cone_height (r : ℝ) (v : ℝ) : ℝ := (3 * v) / (Real.pi * r^2)

theorem ice_cream_cone_height :
  let r : ℝ := 4
  let v : ℝ := 150
  let h : ℝ := cone_height r v
  ⌊h⌋ = 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_cream_cone_height_l897_89792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_problem_l897_89782

/-- The investment problem with four investors: Raghu, Trishul, Vishal, and Swathi. -/
theorem investment_problem (raghu trishul vishal swathi : ℝ) : 
  trishul = 0.9 * raghu →
  vishal = 1.1 * trishul →
  swathi = 1.2 * trishul →
  raghu + trishul + vishal + swathi = 8500 →
  ∃ ε > 0, |raghu - 2141.06| < ε := by
  sorry

#eval Float.toString (8500 / 3.97)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_problem_l897_89782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_sum_l897_89788

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a line passing through the focus
def line (k : ℝ) (x y : ℝ) : Prop := y = k*(x - 1)

-- Define the intersection points of the line and parabola
noncomputable def intersection_points (k : ℝ) : ℝ × ℝ × ℝ × ℝ :=
  let x1 := ((k^2 + 2) - Real.sqrt (k^2 + 1)) / k^2
  let y1 := k*(x1 - 1)
  let x2 := ((k^2 + 2) + Real.sqrt (k^2 + 1)) / k^2
  let y2 := k*(x2 - 1)
  (x1, y1, x2, y2)

-- Define m and n
noncomputable def m_n (k : ℝ) : ℝ × ℝ :=
  let (x1, _, x2, _) := intersection_points k
  (x1 / (1 - x1), x2 / (1 - x2))

-- Theorem statement
theorem parabola_intersection_sum (k : ℝ) :
  let (m, n) := m_n k
  m + n = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_sum_l897_89788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equation_solution_l897_89748

theorem cube_root_equation_solution :
  ∀ x : ℝ, (Real.rpow (10 * x - 2) (1/3) + Real.rpow (8 * x + 2) (1/3) = 3 * Real.rpow x (1/3)) → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equation_solution_l897_89748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l897_89778

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  Real.sqrt (x + (y + z^(1/4))^(1/3)) ≥ (x * y * z)^(1/32) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l897_89778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_f_at_neg_half_l897_89732

noncomputable def f (x : ℝ) : ℝ := (6 * x^2 + x - 1) / (x + 1/2)

theorem limit_of_f_at_neg_half :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x + 1/2| → |x + 1/2| < δ → |f x + 5| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_f_at_neg_half_l897_89732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_C_decreases_l897_89775

/-- The function C(j) representing the given equation -/
noncomputable def C (e R r j : ℝ) : ℝ := (e * j) / (R + j^2 * r)

/-- Theorem stating that C decreases as j increases -/
theorem C_decreases (e R r j : ℝ) (he : e > 0) (hR : R > 0) (hr : r > 0) (hj : j > 0) :
  deriv (fun x => C e R r x) j < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_C_decreases_l897_89775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_real_roots_three_quadratics_l897_89790

/-- Helper function to count the number of real roots of a quadratic function -/
noncomputable def number_of_real_roots (f : ℝ → ℝ) : ℕ := sorry

/-- Given positive real numbers a, b, and c, the maximum total number of real roots
    for the polynomials ax^2 + bx + c, bx^2 + cx + a, and cx^2 + ax + b is 4. -/
theorem max_real_roots_three_quadratics (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ∃ (n : ℕ), n ≤ 4 ∧
  ∀ (m : ℕ), (∃ (x y z : ℕ),
    x = number_of_real_roots (λ t ↦ a*t^2 + b*t + c) ∧
    y = number_of_real_roots (λ t ↦ b*t^2 + c*t + a) ∧
    z = number_of_real_roots (λ t ↦ c*t^2 + a*t + b) ∧
    m = x + y + z) → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_real_roots_three_quadratics_l897_89790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_circle_on_black_squares_l897_89744

/-- Represents a point on a chessboard -/
structure ChessboardPoint where
  x : ℝ
  y : ℝ

/-- Represents a circle on a chessboard -/
structure ChessboardCircle where
  center : ChessboardPoint
  radius : ℝ

/-- Checks if a point is on a black square of the chessboard -/
def is_black_square (p : ChessboardPoint) : Prop :=
  (Int.floor p.x + Int.floor p.y) % 2 = 0

/-- Checks if a circle lies entirely on black squares -/
def circle_on_black_squares (c : ChessboardCircle) : Prop :=
  ∀ p : ChessboardPoint, 
    (p.x - c.center.x)^2 + (p.y - c.center.y)^2 ≤ c.radius^2 → 
    is_black_square p

/-- The theorem stating the largest circle on black squares -/
theorem largest_circle_on_black_squares :
  ∃ (c : ChessboardCircle), 
    circle_on_black_squares c ∧ 
    c.radius = Real.sqrt 2 / 2 ∧
    ∀ (c' : ChessboardCircle), circle_on_black_squares c' → c'.radius ≤ c.radius := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_circle_on_black_squares_l897_89744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l897_89735

/-- Calculates the length of a train given the time to cross a bridge, bridge length, and train speed. -/
noncomputable def train_length (time : ℝ) (bridge_length : ℝ) (speed_kmph : ℝ) : ℝ :=
  let speed_mps := speed_kmph * (1000 / 3600)
  let total_distance := speed_mps * time
  total_distance - bridge_length

/-- The length of the train is approximately 99.98 meters. -/
theorem train_length_calculation :
  let time := 16.665333439991468
  let bridge_length := 150
  let speed_kmph := 54
  |train_length time bridge_length speed_kmph - 99.98| < 0.01 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval train_length 16.665333439991468 150 54

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l897_89735
