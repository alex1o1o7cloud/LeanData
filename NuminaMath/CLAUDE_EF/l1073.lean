import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_regions_area_relation_l1073_107311

-- Define the triangle sides
def a : ℝ := 8
def b : ℝ := 15
def c : ℝ := 17

-- Define the regions
structure Region where
  area : ℝ

-- Define specific regions
def X : Region := ⟨0⟩  -- Initial area set to 0
def Y : Region := ⟨0⟩
def Z : Region := ⟨0⟩

-- State the theorem
theorem circle_regions_area_relation :
  -- The triangle is right-angled
  a^2 + b^2 = c^2 →
  -- Z is the largest region
  (∀ r : Region, r.area ≤ Z.area) →
  -- X is smaller than Y
  X.area < Y.area →
  -- The sum of X and Y equals Z
  X.area + Y.area = Z.area := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_regions_area_relation_l1073_107311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pioneer_hat_assignment_l1073_107323

structure Pioneer where
  id : Nat
  knows : Finset Nat
  hatColor : Nat

def validAssignment (pioneers : Finset Pioneer) : Prop :=
  (∀ p ∈ pioneers, 50 ≤ p.knows.card ∧ p.knows.card ≤ 100) ∧
  (∀ p ∈ pioneers, p.hatColor ≤ 1331) ∧
  (∀ p ∈ pioneers, (pioneers.filter (λ q => q.id ∈ p.knows ∧ q.hatColor ≠ p.hatColor)).card ≥ 20)

theorem pioneer_hat_assignment (pioneers : Finset Pioneer) : 
  ∃ (assignment : Finset Pioneer), 
    validAssignment assignment ∧ 
    assignment.card = pioneers.card := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pioneer_hat_assignment_l1073_107323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_length_l1073_107376

noncomputable section

/-- Parabola defined by y = 2x^2 -/
def parabola (x : ℝ) : ℝ := 2 * x^2

/-- Point on the parabola given x-coordinate -/
def point_on_parabola (x : ℝ) : ℝ × ℝ := (x, parabola x)

/-- Length of a line segment between two x-coordinates on the parabola -/
def segment_length (x1 x2 : ℝ) : ℝ := |x2 - x1|

/-- Area of a triangle given base and height -/
def triangle_area (base height : ℝ) : ℝ := (1/2) * base * height

theorem parabola_triangle_length :
  ∀ a : ℝ,
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := point_on_parabola (-a)
  let C : ℝ × ℝ := point_on_parabola a
  let BC : ℝ := segment_length (-a) a
  let height : ℝ := parabola a
  triangle_area BC height = 128 →
  BC = 8 := by
  intro a
  -- The proof steps would go here
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_length_l1073_107376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l1073_107354

noncomputable section

/-- An ellipse is defined by the equation √((x-4)² + (y+3)²) + √((x+6)² + (y-9)²) = 22 -/
def isOnEllipse (x y : ℝ) : Prop :=
  Real.sqrt ((x - 4)^2 + (y + 3)^2) + Real.sqrt ((x + 6)^2 + (y - 9)^2) = 22

/-- The distance between two points (x₁, y₁) and (x₂, y₂) in ℝ² -/
def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

/-- The foci of the ellipse are (4, -3) and (-6, 9) -/
def foci : (ℝ × ℝ) × (ℝ × ℝ) := ((4, -3), (-6, 9))

theorem ellipse_foci_distance :
  distance (foci.1.1) (foci.1.2) (foci.2.1) (foci.2.2) = 2 * Real.sqrt 61 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l1073_107354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_to_line_l1073_107347

/-- The circle x^2 + y^2 = 1 -/
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- The line x + √3y - 4 = 0 -/
def target_line (x y : ℝ) : Prop := x + Real.sqrt 3 * y - 4 = 0

/-- Distance from a point (x, y) to the line x + √3y - 4 = 0 -/
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  |x + Real.sqrt 3 * y - 4| / Real.sqrt 4

theorem min_distance_circle_to_line :
  ∃ (d : ℝ), d = 1 ∧ 
  (∀ (x y : ℝ), unit_circle x y → distance_to_line x y ≥ d) ∧
  (∃ (x y : ℝ), unit_circle x y ∧ distance_to_line x y = d) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_to_line_l1073_107347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_economic_indicator_calculation_l1073_107303

/-- Given the equation fp - w = 8000, where f = 8 and w = 9 - 75i, 
    prove that p is approximately equal to 1001.13 - 9.38i. -/
theorem economic_indicator_calculation (f w p : ℂ) : 
  f = 8 → w = 9 - 75 * Complex.I → f * p - w = 8000 → 
  Complex.abs (p - (1001.13 - 9.38 * Complex.I)) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_economic_indicator_calculation_l1073_107303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_parallelism_l1073_107314

/-- A structure representing a circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A structure representing a point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate to check if two circles are equal -/
def CirclesEqual (c1 c2 : Circle) : Prop :=
  c1.radius = c2.radius

/-- Predicate to check if a circle touches another circle internally at a point -/
def CircleTouchesInternally (c1 c2 : Circle) (p : Point) : Prop :=
  sorry

/-- Predicate to check if a point lies on a circle -/
def PointOnCircle (p : Point) (c : Circle) : Prop :=
  sorry

/-- Function to get the intersection point of a line and a circle -/
noncomputable def LineCircleIntersection (p1 p2 : Point) (c : Circle) : Point :=
  sorry

/-- Predicate to check if two lines are parallel -/
def LinesParallel (p1 p2 p3 p4 : Point) : Prop :=
  sorry

/-- Theorem statement -/
theorem circle_tangent_parallelism
  (S S₁ S₂ : Circle)
  (A₁ A₂ C : Point)
  (h1 : CirclesEqual S₁ S₂)
  (h2 : CircleTouchesInternally S S₁ A₁)
  (h3 : CircleTouchesInternally S S₂ A₂)
  (h4 : PointOnCircle C S)
  : LinesParallel (LineCircleIntersection A₁ C S₁) (LineCircleIntersection A₂ C S₂) A₁ A₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_parallelism_l1073_107314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_diff_eq_and_initial_condition_l1073_107342

-- Define the differential equation
def diff_eq (x y : ℝ) (y' : ℝ → ℝ) : Prop :=
  (x + y) * (y' x) = 1

-- Define the solution function
def solution (x : ℝ) : ℝ :=
  -(x + 1)

-- Define the derivative of the solution function
def solution_deriv (x : ℝ) : ℝ :=
  -1

-- State the theorem
theorem solution_satisfies_diff_eq_and_initial_condition :
  (∀ x : ℝ, diff_eq x (solution x) solution_deriv) ∧
  solution (-1) = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_diff_eq_and_initial_condition_l1073_107342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_animal_testing_theorem_l1073_107331

/-- Number of animals -/
def total_animals : ℕ := 5

/-- Number of infected animals -/
def infected_animals : ℕ := 1

/-- Scheme A: Test one by one until the infected animal is found -/
def scheme_A : Unit := ()

/-- Scheme B: Test 3 randomly selected animals together, then proceed based on the result -/
def scheme_B : Unit := ()

/-- The probability that Scheme A requires no fewer tests than Scheme B -/
def prob_A_no_fewer_than_B : ℚ := 18/25

/-- The expected number of tests required by Scheme B -/
def expected_tests_B : ℚ := 12/5

/-- Function to calculate the probability that Scheme A requires no fewer tests than Scheme B -/
def probability_A_no_fewer_than_B (schemeA schemeB : Unit) (total infected : ℕ) : ℚ := 
  sorry

/-- Function to calculate the expectation of tests required by Scheme B -/
def expectation_B (schemeB : Unit) (total infected : ℕ) : ℚ := 
  sorry

theorem animal_testing_theorem :
  (total_animals = 5 ∧ infected_animals = 1) →
  (∃ (p : ℚ), p = prob_A_no_fewer_than_B ∧
    p = probability_A_no_fewer_than_B scheme_A scheme_B total_animals infected_animals) ∧
  (∃ (e : ℚ), e = expected_tests_B ∧
    e = expectation_B scheme_B total_animals infected_animals) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_animal_testing_theorem_l1073_107331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_sqrt_difference_l1073_107372

theorem simplify_sqrt_difference : Real.sqrt 20 - Real.sqrt 5 = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_sqrt_difference_l1073_107372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monochromatic_triangle_exists_l1073_107391

/-- A type representing a point in 3D space -/
def Point3D := ℝ × ℝ × ℝ

/-- A type representing an edge between two points -/
inductive Edge
| mk : Point3D → Point3D → Edge

/-- A type representing the color of an edge -/
inductive Color
| Red
| Blue

/-- A predicate that checks if four points are coplanar -/
def are_coplanar (p1 p2 p3 p4 : Point3D) : Prop := sorry

/-- A function that colors an edge -/
def color_edge : Edge → Color := sorry

/-- A predicate that checks if three edges form a triangle -/
def form_triangle (e1 e2 e3 : Edge) : Prop := sorry

/-- The main theorem -/
theorem monochromatic_triangle_exists 
  (points : Finset Point3D)
  (edges : Finset Edge)
  (h_point_count : points.card = 9)
  (h_edge_count : edges.card = 33)
  (h_not_coplanar : ∀ p1 p2 p3 p4, p1 ∈ points → p2 ∈ points → p3 ∈ points → p4 ∈ points →
    p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 → ¬are_coplanar p1 p2 p3 p4)
  (h_edges_between_points : ∀ e ∈ edges, ∃ p1 p2, p1 ∈ points ∧ p2 ∈ points ∧ e = Edge.mk p1 p2)
  : ∃ e1 e2 e3, e1 ∈ edges ∧ e2 ∈ edges ∧ e3 ∈ edges ∧ 
    form_triangle e1 e2 e3 ∧ color_edge e1 = color_edge e2 ∧ color_edge e2 = color_edge e3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monochromatic_triangle_exists_l1073_107391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lg_properties_l1073_107302

-- Define the logarithm function with base 10
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define our function f
noncomputable def f (x : ℝ) : ℝ := lg x

-- State the theorem
theorem lg_properties :
  ∀ (x₁ x₂ : ℝ), x₁ ≠ x₂ → x₁ > 0 → x₂ > 0 →
    (f (x₁ * x₂) = f x₁ + f x₂) ∧
    ((f x₁ - f x₂) / (x₁ - x₂) > 0) ∧
    (f ((x₁ + x₂) / 2) > (f x₁ + f x₂) / 2) :=
by
  -- Introduce variables and assumptions
  intro x₁ x₂ hneq hpos1 hpos2
  
  -- Split the goal into three parts
  constructor
  · -- Prove f (x₁ * x₂) = f x₁ + f x₂
    sorry
  
  constructor
  · -- Prove (f x₁ - f x₂) / (x₁ - x₂) > 0
    sorry
  
  · -- Prove f ((x₁ + x₂) / 2) > (f x₁ + f x₂) / 2
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lg_properties_l1073_107302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_not_coplanar_l1073_107325

/-- Vector a in ℝ³ -/
def a : ℝ × ℝ × ℝ := (4, 1, 1)

/-- Vector b in ℝ³ -/
def b : ℝ × ℝ × ℝ := (-9, -4, -9)

/-- Vector c in ℝ³ -/
def c : ℝ × ℝ × ℝ := (6, 2, 6)

/-- Theorem stating that vectors a, b, and c are not coplanar -/
theorem vectors_not_coplanar : 
  ¬(∃ (x y z : ℝ), x • (a.1, a.2.1, a.2.2) + y • (b.1, b.2.1, b.2.2) + z • (c.1, c.2.1, c.2.2) = (0, 0, 0) ∧ (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_not_coplanar_l1073_107325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l1073_107310

noncomputable def Ellipse (a b : ℝ) := {p : ℝ × ℝ | (p.1 ^ 2 / a ^ 2) + (p.2 ^ 2 / b ^ 2) = 1}

noncomputable def Foci (a b : ℝ) : ℝ × ℝ × ℝ × ℝ :=
  let c := Real.sqrt (a ^ 2 - b ^ 2)
  (-c, 0, c, 0)

noncomputable def Eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - (b ^ 2 / a ^ 2))

theorem ellipse_eccentricity_range (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  ∃ (A B : ℝ × ℝ), A ∈ Ellipse a b ∧ B ∈ Ellipse a b ∧
  let (f1x, f1y, f2x, f2y) := Foci a b
  ‖(A.1 - f1x, A.2 - f1y)‖ = 3 * ‖(B.1 - f2x, B.2 - f2y)‖ →
  (1 / 2 : ℝ) < Eccentricity a b ∧ Eccentricity a b < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l1073_107310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_all_majors_chosen_l1073_107382

/-- The number of students -/
def num_students : ℕ := 4

/-- The number of majors -/
def num_majors : ℕ := 3

/-- The probability that all majors are chosen -/
def prob_all_majors_chosen : ℚ := 4/9

/-- A student type -/
structure Student where
  id : ℕ

/-- A major type -/
structure Major where
  id : ℕ

/-- Theorem stating that the probability of all majors being chosen is 4/9 -/
theorem probability_all_majors_chosen :
  (num_students = 4) →
  (num_majors = 3) →
  (∀ s : Student, ∃! m : Major, s.id ≤ num_students ∧ m.id ≤ num_majors) →
  (∀ s : Student, ∀ m : Major, s.id ≤ num_students ∧ m.id ≤ num_majors → 
    (1 : ℚ) / num_majors = (1 : ℚ) / 3) →
  prob_all_majors_chosen = 4/9 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_all_majors_chosen_l1073_107382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_balloon_arrangements_l1073_107321

def word : String := "BALLOON"

def is_vowel (c : Char) : Bool :=
  c == 'A' || c == 'E' || c == 'I' || c == 'O' || c == 'U'

def vowels : List Char := word.toList.filter is_vowel
def consonants : List Char := word.toList.filter (fun c => !is_vowel c)

def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def count_arrangements (chars : List Char) : Nat :=
  factorial chars.length / (chars.foldl (fun acc c => acc * factorial (chars.filter (· = c)).length) 1)

theorem balloon_arrangements :
  count_arrangements consonants * count_arrangements vowels = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_balloon_arrangements_l1073_107321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cylinder_volume_is_8pi_div_27_l1073_107343

/-- Represents a truncated pyramid with specific properties -/
structure TruncatedPyramid where
  -- Base ABC
  sideABC : ℝ
  -- Base A₁B₁C₁
  sideA₁B₁C₁ : ℝ
  -- Height of the truncated pyramid
  height : ℝ
  -- Assertion that the bases are equilateral triangles
  baseABC_equilateral : sideABC > 0
  baseA₁B₁C₁_equilateral : sideA₁B₁C₁ > 0
  -- Assertion that C₁O is perpendicular to bases
  C₁O_perpendicular : True
  -- Specific dimensions
  sideABC_eq : sideABC = 3
  sideA₁B₁C₁_eq : sideA₁B₁C₁ = 2
  height_eq : height = 3

/-- Calculates the maximum volume of a cylinder inside the truncated pyramid -/
noncomputable def maxCylinderVolume (tp : TruncatedPyramid) : ℝ :=
  8 * Real.pi / 27

/-- Theorem stating that the maximum volume of a cylinder inside the truncated pyramid is 8π/27 -/
theorem max_cylinder_volume_is_8pi_div_27 (tp : TruncatedPyramid) :
  maxCylinderVolume tp = 8 * Real.pi / 27 := by
  -- Unfold the definition of maxCylinderVolume
  unfold maxCylinderVolume
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cylinder_volume_is_8pi_div_27_l1073_107343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_counterfeit_banknotes_probability_l1073_107371

/-- The probability of both banknotes being counterfeit given one is known to be counterfeit -/
theorem counterfeit_banknotes_probability
  (total : ℕ)
  (counterfeits : ℕ)
  (h_total : total = 20)
  (h_counterfeits : counterfeits = 5) :
  (counterfeits * (counterfeits - 1) : ℚ) / (total * (total - 1)) /
  ((counterfeits * (total - counterfeits) + counterfeits * (counterfeits - 1) : ℚ) / (total * (total - 1))) =
  2 / 17 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_counterfeit_banknotes_probability_l1073_107371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jane_nail_polish_drying_time_l1073_107306

/-- Represents the drying times for Jane's nail polish application -/
structure NailPolishDryingTimes where
  base_coat : ℕ
  color_coats : Fin 3 → ℕ
  index_middle_art : Fin 3 → ℕ
  ring_pinky_art : Fin 2 → ℕ
  top_coat : ℕ

/-- Calculates the total drying time for Jane's nail polish application -/
def total_drying_time (times : NailPolishDryingTimes) : ℕ :=
  times.base_coat +
  (Finset.sum (Finset.univ : Finset (Fin 3)) times.color_coats) +
  (Finset.sum (Finset.univ : Finset (Fin 3)) times.index_middle_art) +
  (Finset.sum (Finset.univ : Finset (Fin 2)) times.ring_pinky_art) +
  times.top_coat

/-- Theorem stating that Jane's total nail polish drying time is 86 minutes -/
theorem jane_nail_polish_drying_time :
  ∃ (times : NailPolishDryingTimes),
    times.base_coat = 4 ∧
    times.color_coats 0 = 5 ∧
    times.color_coats 1 = 6 ∧
    times.color_coats 2 = 7 ∧
    times.index_middle_art 0 = 8 ∧
    times.index_middle_art 1 = 10 ∧
    times.index_middle_art 2 = 12 ∧
    times.ring_pinky_art 0 = 11 ∧
    times.ring_pinky_art 1 = 14 ∧
    times.top_coat = 9 ∧
    total_drying_time times = 86 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jane_nail_polish_drying_time_l1073_107306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_prime_pairs_l1073_107355

noncomputable def countPrimePairs : Prop :=
  ∃ (S : Finset (Nat × Nat)), 
    (∀ (p a : Nat), (p, a) ∈ S ↔ 
      (Nat.Prime p ∧ 3 ≤ p ∧ p < 100 ∧ 1 ≤ a ∧ a < p ∧ p ∣ (a^(p-2) - a))) ∧
    (Finset.filter (fun p => Nat.Prime p ∧ 3 ≤ p ∧ p < 100) (Finset.range 100)).card = 24 ∧
    S.card = 48

theorem count_prime_pairs : countPrimePairs := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_prime_pairs_l1073_107355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_decreasing_lambda_bound_l1073_107363

def a (n : ℕ+) (lambda : ℝ) : ℝ := -2 * (n : ℝ)^2 + lambda * (n : ℝ)

theorem sequence_decreasing_lambda_bound (lambda : ℝ) :
  (∀ n : ℕ+, a n lambda > a (n + 1) lambda) → lambda < 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_decreasing_lambda_bound_l1073_107363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_diff_eq_and_initial_conditions_l1073_107330

/-- A function representing the solution to the differential equation -/
noncomputable def y (x : ℝ) : ℝ := 3 * Real.exp x - 2 * Real.exp (2 * x)

/-- The differential equation -/
def diff_eq (f : ℝ → ℝ) : Prop :=
  ∀ x, (deriv (deriv f)) x - 3 * (deriv f x) + 2 * (f x) = 0

theorem solution_satisfies_diff_eq_and_initial_conditions :
  diff_eq y ∧ y 0 = 1 ∧ (deriv y) 0 = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_diff_eq_and_initial_conditions_l1073_107330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_k_value_l1073_107350

theorem quadratic_function_k_value :
  ∀ (a b c k : ℤ) (g : ℝ → ℝ),
  g = (λ x => (a * x^2 + b * x + c : ℝ)) →
  g 1 = 0 →
  10 < g 5 ∧ g 5 < 20 →
  30 < g 6 ∧ g 6 < 40 →
  3000 * k < g 100 ∧ g 100 < 3000 * (k + 1) →
  k = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_k_value_l1073_107350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_min_distances_l1073_107379

-- Define the ellipse
def ellipse (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define a point on the ellipse
def point_on_ellipse (P : ℝ × ℝ) (a b : ℝ) : Prop :=
  ellipse P.1 P.2 a b

-- Define the left focus
def left_focus (F₁ : ℝ × ℝ) (a c : ℝ) : Prop := F₁ = (-c, 0)

-- Define the maximum and minimum distances
def max_distance (m : ℝ) (P F₁ : ℝ × ℝ) : Prop := 
  dist P F₁ ≤ m

def min_distance (n : ℝ) (P F₁ : ℝ × ℝ) : Prop := 
  n ≤ dist P F₁

-- The theorem to prove
theorem sum_of_max_min_distances (a b c : ℝ) (P F₁ : ℝ × ℝ) (m n : ℝ) :
  a = 5 →
  point_on_ellipse P a b →
  left_focus F₁ a c →
  max_distance m P F₁ →
  min_distance n P F₁ →
  m + n = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_min_distances_l1073_107379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_C_is_quadratic_l1073_107362

-- Define the type of functions we're considering
def RealFunction := ℝ → ℝ

-- Define what it means for a function to be quadratic
def IsQuadratic (f : RealFunction) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

-- Define the functions given in the problem
noncomputable def functionA : RealFunction := λ x ↦ 2 + Real.sqrt (1 + x^2)
def functionB : RealFunction := λ x ↦ (x - 1)^2 - x^2
def functionC : RealFunction := λ x ↦ 3 * x^2 - 2
noncomputable def functionD : RealFunction := λ x ↦ Real.sqrt (x^2 + x - 1)

-- Theorem stating that only functionC is quadratic
theorem only_C_is_quadratic :
  IsQuadratic functionC ∧
  ¬IsQuadratic functionA ∧
  ¬IsQuadratic functionB ∧
  ¬IsQuadratic functionD := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_C_is_quadratic_l1073_107362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_with_unit_inscribed_sphere_l1073_107377

/-- Represents a cone with inscribed and circumscribed spheres -/
structure ConeWithSpheres where
  inscribedRadius : ℝ
  circumscribedRadius : ℝ
  height : ℝ
  shareCenter : Bool

/-- Calculate the volume of a cone -/
noncomputable def coneVolume (cone : ConeWithSpheres) : ℝ :=
  (1/3) * Real.pi * cone.inscribedRadius^2 * cone.height

/-- The main theorem to prove -/
theorem cone_volume_with_unit_inscribed_sphere (cone : ConeWithSpheres) 
  (h1 : cone.inscribedRadius = 1)
  (h2 : cone.shareCenter = true) : 
  coneVolume cone = 3 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_with_unit_inscribed_sphere_l1073_107377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_interest_proof_l1073_107309

/-- Calculates the simple interest amount given principal, rate, and time -/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

theorem additional_interest_proof (initial_deposit : ℝ) (initial_amount : ℝ) (higher_amount : ℝ) (time : ℝ) :
  initial_deposit = 8000 →
  initial_amount = 9200 →
  higher_amount = 9440 →
  time = 3 →
  ∃ (initial_rate : ℝ) (additional_rate : ℝ),
    initial_amount = initial_deposit + simple_interest initial_deposit initial_rate time ∧
    higher_amount = initial_deposit + simple_interest initial_deposit (initial_rate + additional_rate) time ∧
    additional_rate = 1 :=
by sorry

#check additional_interest_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_interest_proof_l1073_107309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_calculation_l1073_107396

/-- Given a total investment and the ratio of real estate to mutual funds investment,
    calculate the investment in real estate. -/
noncomputable def real_estate_investment (total : ℝ) (ratio : ℝ) : ℝ :=
  (ratio * total) / (ratio + 1)

/-- Theorem stating that given the specific conditions, the real estate investment
    is approximately $169,230.77. -/
theorem investment_calculation :
  let total := (200000 : ℝ)
  let ratio := (5.5 : ℝ)
  abs (real_estate_investment total ratio - 169230.77) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_calculation_l1073_107396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_implies_a_geq_3_l1073_107393

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 1/x

-- State the theorem
theorem increasing_f_implies_a_geq_3 (a : ℝ) :
  (∀ x₁ x₂ : ℝ, 1/2 < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂) →
  a ≥ 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_implies_a_geq_3_l1073_107393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_arithmetic_sequence_l1073_107359

/-- An arithmetic sequence with first term a₁ and common difference d -/
noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

/-- Sum of the first n terms of an arithmetic sequence -/
noncomputable def sum_arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (a₁ + arithmetic_sequence a₁ d n) / 2

theorem min_sum_arithmetic_sequence (a₁ d : ℝ) :
  a₁ = -11 →
  arithmetic_sequence a₁ d 4 + arithmetic_sequence a₁ d 6 = -6 →
  ∃ (n : ℕ), ∀ (m : ℕ), 
    sum_arithmetic_sequence a₁ d n ≤ sum_arithmetic_sequence a₁ d m ∧
    n = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_arithmetic_sequence_l1073_107359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flowchart_property_l1073_107398

/-- Represents a flowchart --/
structure Flowchart where
  has_start : Bool
  has_end : Bool
  entry_points : Nat
  exit_points : Nat

/-- Predicate for a valid flowchart --/
def is_valid_flowchart (f : Flowchart) : Prop :=
  f.has_start ∧ f.has_end ∧ f.entry_points = 1

/-- Theorem stating the correct property of flowcharts --/
theorem flowchart_property :
  ∀ (f : Flowchart), is_valid_flowchart f →
    (f.has_start ∧ f.has_end) ∧
    ¬(∃ (g : Flowchart), is_valid_flowchart g ∧ g.exit_points = 1) ∧
    ¬(∀ (alg : Nat), ∃! (h : Flowchart), is_valid_flowchart h) :=
  fun f hf => by
    sorry

#check flowchart_property

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flowchart_property_l1073_107398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_curve_and_angle_l1073_107383

-- Define the curve
noncomputable def curve (θ : Real) : Real × Real := (3 * Real.cos θ, 4 * Real.sin θ)

-- Define the point P
noncomputable def P : Real × Real := (12/5, 12/5)

-- Theorem statement
theorem point_on_curve_and_angle :
  ∃ θ : Real, 0 ≤ θ ∧ θ ≤ Real.pi ∧
  curve θ = P ∧
  Real.tan (Real.pi/4) = (P.2 / P.1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_curve_and_angle_l1073_107383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_count_l1073_107352

/-- The number of solutions to the equation -/
def num_solutions : ℕ := 138

/-- The range of possible x values -/
def x_range : Finset ℕ := Finset.range 151 \ {0}

/-- The set of perfect squares up to 12^2 -/
def perfect_squares : Finset ℕ := Finset.filter (λ n => ∃ k : ℕ, k ≤ 12 ∧ n = k^2) x_range

/-- The equation has a solution for x if x is in the range 1 to 150 but not a perfect square up to 12^2 -/
def is_solution (x : ℕ) : Prop :=
  x ∈ x_range ∧ x ∉ perfect_squares

theorem solution_count :
  (x_range \ perfect_squares).card = num_solutions :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_count_l1073_107352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_line_through_1_1_and_2_2_l1073_107381

/-- Theorem: The slope of the line passing through points (1,1) and (2,2) is 1 -/
theorem slope_of_line_through_1_1_and_2_2 :
  (2 - 1) / (2 - 1) = 1 := by
  norm_num

#check slope_of_line_through_1_1_and_2_2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_line_through_1_1_and_2_2_l1073_107381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_odd_and_even_fn_l1073_107301

noncomputable def fn (n : ℕ) : ℤ := ⌊(2 : ℝ)^n * Real.sqrt 69⌋ + ⌊(2 : ℝ)^n * Real.sqrt 96⌋

theorem infinite_odd_and_even_fn :
  (Set.Infinite {n : ℕ | fn n % 2 = 1}) ∧
  (Set.Infinite {n : ℕ | fn n % 2 = 0}) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_odd_and_even_fn_l1073_107301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_conditions_l1073_107364

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Line → Line → Prop)
variable (parallel : Line → Line → Prop)
variable (contains : Plane → Line → Prop)
variable (perpendicularPL : Plane → Line → Prop)
variable (parallelP : Plane → Plane → Prop)
variable (perpendicularP : Plane → Plane → Prop)
variable (parallelLP : Line → Plane → Prop)

-- Define the lines and planes
variable (a b : Line) (α β : Plane)

-- Define the theorem
theorem perpendicular_conditions :
  -- Condition 1 does not necessarily imply a ⊥ b
  (contains α a ∧ parallelLP b β ∧ perpendicularP α β) →
    ¬ (perpendicular a b) ∧
  -- Condition 2 implies a ⊥ b
  (perpendicularPL α a ∧ perpendicularPL β b ∧ perpendicularP α β) →
    perpendicular a b ∧
  -- Condition 3 implies a ⊥ b
  (contains α a ∧ perpendicularPL β b ∧ parallelP α β) →
    perpendicular a b ∧
  -- Condition 4 implies a ⊥ b
  (perpendicularPL α a ∧ parallelLP b β ∧ parallelP α β) →
    perpendicular a b :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_conditions_l1073_107364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_size_relationship_l1073_107345

theorem size_relationship (a b c : ℝ) : 
  a = 20.2 → b = Real.rpow 0.4 0.2 → c = Real.rpow 0.4 0.6 → a > b ∧ b > c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_size_relationship_l1073_107345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rajesh_distance_l1073_107380

theorem rajesh_distance (hiro_distance rajesh_distance : ℝ) 
  (rajesh_condition : rajesh_distance = 4 * hiro_distance - 10)
  (total_distance : hiro_distance + rajesh_distance = 25) :
  rajesh_distance = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rajesh_distance_l1073_107380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_equation_l1073_107339

theorem unique_solution_equation : 
  ∃! p : ℕ × ℕ, 
    let (x, y) := p
    x > 0 ∧ y > 0 ∧ 
    (x^4 * y^4 : ℤ) - 24 * (x^2 * y^2 : ℤ) + 35 = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_equation_l1073_107339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_theta_distribution_l1073_107313

/-- Definition of a random walk -/
def RandomWalk (n : ℕ) := Fin (2 * n + 1) → ℝ

/-- Position of the first maximum in a random walk -/
def θ (S : RandomWalk n) : Fin (2 * n + 1) :=
  sorry

/-- Probability measure on random walks -/
noncomputable def P (n : ℕ) (event : RandomWalk n → Prop) : ℝ :=
  sorry

/-- Definition of u_n (not explicitly given in the problem) -/
noncomputable def u (n : ℕ) : ℝ :=
  sorry

/-- Main theorem about the distribution of θ -/
theorem theta_distribution (n : ℕ) :
  (P n (λ S ↦ θ S = 0) = u (2 * n)) ∧
  (P n (λ S ↦ θ S = 2 * n) = (1 / 2) * u (2 * n)) ∧
  (∀ k, 0 < k → k < n →
    P n (λ S ↦ θ S = 2 * k ∨ θ S = 2 * k + 1) = (1 / 2) * u (2 * k) * u (2 * (n - k))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_theta_distribution_l1073_107313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_convergence_l1073_107389

noncomputable def F : ℕ → ℝ
  | 0 => 1
  | 1 => 4
  | (n + 2) => 3 * F (n + 1) - F n

noncomputable def series_term (n : ℕ) : ℝ := 1 / F (3^n)

theorem series_convergence :
  ∃ (S : ℝ), HasSum series_term S := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_convergence_l1073_107389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_l1073_107315

/-- The function f(x) as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos x + 2 * Real.sqrt 3 * (Real.cos x) ^ 2 - Real.sqrt 3

/-- The smallest positive period of f(x) -/
noncomputable def smallest_positive_period : ℝ := Real.pi

/-- Theorem stating that the smallest positive period of f(x) is π -/
theorem f_period :
  (∀ x : ℝ, f (x + smallest_positive_period) = f x) ∧
  (∀ p : ℝ, 0 < p → p < smallest_positive_period → ∃ y : ℝ, f (y + p) ≠ f y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_l1073_107315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_circle_radius_l1073_107341

/-- Predicate to check if a polygon is a regular octagon with given side length -/
def is_regular_octagon (side_length : ℝ) : Prop := sorry

/-- Predicate to check if a circle is concentric with and outside an octagon -/
def is_concentric_circle_outside (radius : ℝ) (octagon_side : ℝ) : Prop := sorry

/-- Function to calculate the probability of seeing exactly four sides of the octagon
    from a random point on the circle -/
noncomputable def probability_four_sides_visible (radius : ℝ) (octagon_side : ℝ) : ℝ := sorry

/-- The radius of a circle concentric with and outside a regular octagon,
    given specific visibility conditions -/
theorem octagon_circle_radius (side_length : ℝ) (probability : ℝ) : ℝ :=
  let octagon_side := side_length
  let visibility_prob := probability
  let radius := 6
  have h1 : octagon_side = 3 := by sorry
  have h2 : visibility_prob = 1/3 := by sorry
  have h3 : is_regular_octagon octagon_side := by sorry
  have h4 : is_concentric_circle_outside radius octagon_side := by sorry
  have h5 : probability_four_sides_visible radius octagon_side = visibility_prob := by sorry
  radius

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_circle_radius_l1073_107341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_ratio_from_sin_sum_diff_l1073_107349

theorem tan_ratio_from_sin_sum_diff (α β : ℝ) 
  (h1 : Real.sin (α + β) = Real.sqrt 3 / 2)
  (h2 : Real.sin (α - β) = Real.sqrt 2 / 2) :
  Real.tan α / Real.tan β = (5 + 2 * Real.sqrt 6) / (5 - 2 * Real.sqrt 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_ratio_from_sin_sum_diff_l1073_107349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_is_negative_eight_l1073_107328

-- Define the operation □
def box (a b : ℝ) : ℝ := a^2 + 2*a*b - b^2

-- Define the function f
def f (x : ℝ) : ℝ := box x 2

-- Define the logarithm of absolute value function
noncomputable def lg_abs (x : ℝ) : ℝ := Real.log (abs x) / Real.log 10

-- State the theorem
theorem sum_of_roots_is_negative_eight :
  ∃ (x₁ x₂ x₃ x₄ : ℝ),
    (x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄) ∧
    (∀ x ≠ -2, f x = lg_abs (x + 2) ↔ x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄) →
    x₁ + x₂ + x₃ + x₄ = -8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_is_negative_eight_l1073_107328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_company_optimization_l1073_107366

/-- The maximum number of employees that can be reallocated -/
def max_reallocation : ℕ := 500

/-- The upper bound for the parameter a -/
def a_upper_bound : ℝ := 5

theorem company_optimization (x : ℕ) (a : ℝ) :
  (x > 0) →
  (a > 0) →
  (∀ x' : ℕ, x' ≤ max_reallocation → 
    (1000 - x' : ℝ) * (1 + 0.2 * (x' : ℝ) / 100) ≥ 1000) ∧
  (∀ x' : ℕ, x' ≤ max_reallocation → 
    10 * (a - 3 * (x' : ℝ) / 500) * (x' : ℝ) ≤ 
    10 * (1000 - x' : ℝ) * (1 + 0.2 * (x' : ℝ) / 100)) →
  x ≤ max_reallocation ∧ 0 < a ∧ a ≤ a_upper_bound :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_company_optimization_l1073_107366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_O_in_CaO_l1073_107375

/-- The molar mass of calcium in g/mol -/
noncomputable def molar_mass_Ca : ℝ := 40.08

/-- The molar mass of oxygen in g/mol -/
noncomputable def molar_mass_O : ℝ := 16.00

/-- The molar mass of calcium oxide (CaO) in g/mol -/
noncomputable def molar_mass_CaO : ℝ := molar_mass_Ca + molar_mass_O

/-- The mass percentage of oxygen in calcium oxide -/
noncomputable def mass_percentage_O : ℝ := (molar_mass_O / molar_mass_CaO) * 100

/-- Theorem stating that the mass percentage of oxygen in calcium oxide is approximately 28.53% -/
theorem mass_percentage_O_in_CaO :
  ∃ ε > 0, |mass_percentage_O - 28.53| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_O_in_CaO_l1073_107375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_sides_in_pentagon_l1073_107378

-- Define the pentagon
structure Pentagon (V : Type*) [AddCommGroup V] [Module ℝ V] :=
  (A B C D E : V)

-- Define parallelism
def Parallel (V : Type*) [AddCommGroup V] [Module ℝ V] (v w : V) : Prop :=
  ∃ (k : ℝ), v = k • w

-- State the theorem
theorem parallel_sides_in_pentagon
  {V : Type*} [AddCommGroup V] [Module ℝ V]
  (p : Pentagon V)
  (h1 : Parallel V (p.B - p.C) (p.A - p.D))
  (h2 : Parallel V (p.C - p.D) (p.B - p.E))
  (h3 : Parallel V (p.D - p.E) (p.A - p.C))
  (h4 : Parallel V (p.A - p.E) (p.B - p.D)) :
  Parallel V (p.A - p.B) (p.C - p.E) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_sides_in_pentagon_l1073_107378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l1073_107333

/-- A quadratic function satisfying specific conditions -/
def f (x : ℝ) : ℝ := -2 * x^2 + 4 * x

/-- The maximum value function for f on [t, t+1] where t > 0 -/
noncomputable def f_max (t : ℝ) : ℝ :=
  if t < 1 then 2 else -2 * t^2 + 4 * t

theorem quadratic_function_properties :
  (f (-2) = -16) ∧
  (f 4 = -16) ∧
  (∀ x, f x ≤ 2) ∧
  (∃ x, f x = 2) ∧
  (∀ t > 0, ∀ x ∈ Set.Icc t (t + 1), f x ≤ f_max t) ∧
  (∀ t > 0, ∃ x ∈ Set.Icc t (t + 1), f x = f_max t) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l1073_107333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_sequence_computes_target_l1073_107324

/-- The calculator operation -/
def calc_op (x y : ℝ) : ℝ := x * y + x + y + 1

/-- The sequence of polynomials -/
def polynomial_sequence : ℕ → (ℝ → ℝ)
| 0 => fun x => x
| (n+1) => fun x => calc_op (polynomial_sequence n x) x

/-- The target polynomial -/
noncomputable def target_polynomial (x : ℝ) : ℝ := (x^1983 - 1) / (x - 1)

theorem polynomial_sequence_computes_target :
  ∃ n : ℕ, polynomial_sequence n = target_polynomial := by
  sorry

#check polynomial_sequence_computes_target

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_sequence_computes_target_l1073_107324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dress_color_assignment_l1073_107346

-- Define the girls and dress colors
inductive Girl : Type
  | Anya : Girl
  | Valya : Girl
  | Galya : Girl
  | Nina : Girl

inductive DressColor : Type
  | Green : DressColor
  | Blue : DressColor
  | White : DressColor
  | Pink : DressColor

-- Define the assignment of dress colors to girls
variable (dress_assignment : Girl → DressColor)

-- Define the circle arrangement
variable (next_in_circle : Girl → Girl)

-- State the conditions
axiom green_not_anya_or_valya :
  dress_assignment Girl.Anya ≠ DressColor.Green ∧ dress_assignment Girl.Valya ≠ DressColor.Green

axiom green_between_blue_and_nina :
  ∃ (g : Girl), dress_assignment g = DressColor.Green ∧
  ((dress_assignment (next_in_circle g) = DressColor.Blue ∧ next_in_circle (next_in_circle g) = Girl.Nina) ∨
   (dress_assignment (next_in_circle (next_in_circle g)) = DressColor.Blue ∧ next_in_circle g = Girl.Nina))

axiom white_between_pink_and_valya :
  ∃ (g : Girl), dress_assignment g = DressColor.White ∧
  ((dress_assignment (next_in_circle g) = DressColor.Pink ∧ next_in_circle (next_in_circle g) = Girl.Valya) ∨
   (dress_assignment (next_in_circle (next_in_circle g)) = DressColor.Pink ∧ next_in_circle g = Girl.Valya))

-- State the theorem
theorem dress_color_assignment :
  dress_assignment Girl.Anya = DressColor.White ∧
  dress_assignment Girl.Valya = DressColor.Blue ∧
  dress_assignment Girl.Galya = DressColor.Green ∧
  dress_assignment Girl.Nina = DressColor.Pink :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dress_color_assignment_l1073_107346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_from_vector_relation_l1073_107370

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the vectors
variable (O A B C : ℝ × ℝ × ℝ)

-- State the theorem
theorem function_from_vector_relation 
  (h_collinear : ∃ (t : ℝ), A - B = t • (C - B))
  (h_vector_relation : ∀ (x : ℝ), x > 0 → 
    A - O = (f x + 2 * (deriv f 1) * x) • (B - O) - Real.log x • (C - O))
  (h_differentiable : Differentiable ℝ f) :
  f = fun x => Real.log x - (2/3) * x + 1 := by
  sorry

#check function_from_vector_relation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_from_vector_relation_l1073_107370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_EOF_l1073_107332

-- Define the line
def line (x y : ℝ) : Prop := x - 2*y - 3 = 0

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x-2)^2 + (y+3)^2 = 9

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Define the intersection points E and F
noncomputable def E : ℝ × ℝ := sorry
noncomputable def F : ℝ × ℝ := sorry

-- Axioms stating that E and F are on both the line and the circle
axiom E_on_line : line E.1 E.2
axiom E_on_circle : circle_eq E.1 E.2
axiom F_on_line : line F.1 F.2
axiom F_on_circle : circle_eq F.1 F.2

-- Function to calculate the area of a triangle
noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ := sorry

-- Theorem stating the area of triangle EOF
theorem area_of_triangle_EOF :
  let O := origin
  area_triangle O E F = 6 * Real.sqrt 5 / 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_EOF_l1073_107332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_integer_solutions_l1073_107369

theorem two_integer_solutions : 
  ∃! (s : Finset ℤ), (∀ x ∈ s, (x - 3 : ℚ) ^ (30 - x^2) = 1) ∧ s.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_integer_solutions_l1073_107369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_side_equations_l1073_107327

/-- Given a square ABCD with one side equation and diagonal intersection point,
    prove the equations of the other three sides. -/
theorem square_side_equations (A B C D : ℝ × ℝ) (P : ℝ × ℝ) :
  P = (-1, 0) →
  (∀ x y : ℝ, x + 3*y - 5 = 0 ↔ (x, y) ∈ Set.Icc A B) →
  ∃ (f₁ f₂ f₃ : ℝ → ℝ → ℝ),
    (∀ x y : ℝ, f₁ x y = 0 ↔ (x, y) ∈ Set.Icc B C) ∧
    (∀ x y : ℝ, f₂ x y = 0 ↔ (x, y) ∈ Set.Icc C D) ∧
    (∀ x y : ℝ, f₃ x y = 0 ↔ (x, y) ∈ Set.Icc D A) ∧
    (∀ x y : ℝ, f₁ x y = 3*x - y - 3) ∧
    (∀ x y : ℝ, f₂ x y = x + 3*y - 5) ∧
    (∀ x y : ℝ, f₃ x y = x - 3*y + 9) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_side_equations_l1073_107327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_at_two_l1073_107373

theorem derivative_at_two : 
  ∃ (f : ℝ → ℝ), (∀ x, deriv f x = 2*x + 2*(deriv f 2) - 1/x) → deriv f 2 = -7/2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_at_two_l1073_107373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trisha_cookie_count_l1073_107336

/-- Represents the shape and dimensions of a cookie --/
structure Cookie where
  shape : String
  area : ℝ

/-- Calculates the area of a trapezoid --/
noncomputable def trapezoidArea (base1 base2 height : ℝ) : ℝ :=
  (base1 + base2) * height / 2

/-- Calculates the area of a triangle --/
noncomputable def triangleArea (base height : ℝ) : ℝ :=
  base * height / 2

/-- Represents a batch of cookies --/
structure Batch where
  cookie : Cookie
  count : ℕ

theorem trisha_cookie_count (artBatch trishaCookie : Batch) : 
  (artBatch.cookie.shape = "trapezoid" ∧ 
   artBatch.cookie.area = trapezoidArea 3 5 3 ∧
   artBatch.count = 12 ∧
   trishaCookie.cookie.shape = "triangle" ∧
   trishaCookie.cookie.area = triangleArea 3 4) →
  trishaCookie.count = 24 := by
  sorry

#eval "Theorem statement compiled successfully."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trisha_cookie_count_l1073_107336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_c_for_one_in_range_l1073_107337

-- Define the function f
noncomputable def f (x c : ℝ) : ℝ := x^2 + 2*x*(Real.cos x) + c

-- State the theorem
theorem largest_c_for_one_in_range :
  (∃ (c : ℝ), ∀ (c' : ℝ), (∃ (x : ℝ), f x c' = 1) → c' ≤ c) ∧
  (∃ (x : ℝ), f x 2 = 1) ∧
  (∀ (c' : ℝ), c' > 2 → ∀ (x : ℝ), f x c' ≠ 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_c_for_one_in_range_l1073_107337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boxed_product_prices_and_profit_l1073_107388

/-- Represents the purchase and sales data for a boxed product. -/
structure BoxedProduct where
  first_batch_cost : ℚ
  second_batch_cost : ℚ
  selling_price : ℚ
  discount_rate : ℚ
  discounted_units : ℕ

/-- Calculates the purchase prices and profit for the boxed product. -/
noncomputable def calculate_prices_and_profit (product : BoxedProduct) :
  (ℚ × ℚ × ℚ) :=
  let first_price := product.first_batch_cost / 1000
  let second_price := product.second_batch_cost / 2000
  let regular_profit := (product.selling_price - first_price) * 1000 +
                        (product.selling_price - second_price) * (2000 - product.discounted_units)
  let discounted_profit := (product.selling_price * (1 - product.discount_rate) - second_price) *
                           product.discounted_units
  let total_profit := regular_profit + discounted_profit
  (first_price, second_price, total_profit)

/-- Theorem stating the correct purchase prices and profit for the given scenario. -/
theorem boxed_product_prices_and_profit :
  let product := BoxedProduct.mk 40000 88000 56 (1/5) 150
  let (first_price, second_price, profit) := calculate_prices_and_profit product
  first_price = 40 ∧ second_price = 44 ∧ profit = 38320 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_boxed_product_prices_and_profit_l1073_107388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l1073_107353

theorem solve_exponential_equation :
  ∃ x : ℚ, (16 : ℝ) ^ (x : ℝ) * (16 : ℝ) ^ (x : ℝ) * (16 : ℝ) ^ (x : ℝ) = (256 : ℝ) ^ 4 ∧ x = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l1073_107353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_count_approximation_a_2_value_a_3_value_a_4_value_l1073_107312

/-- The number of squares (including tilted ones) that can be formed in an n × n grid --/
def a (n : ℕ) : ℕ := sorry

/-- The approximate lower bound for a(n) --/
def a_lower_bound (n : ℕ) : ℚ :=
  (n * (n + 1) * (2 * n + 1)) / 6

theorem square_count_approximation (n : ℕ) :
  ∃ (k : ℚ), (a n : ℚ) = a_lower_bound n + k ∧ k ≥ 0 := by
  sorry

theorem a_2_value : a 2 = 6 := by
  sorry

theorem a_3_value : a 3 = 20 := by
  sorry

theorem a_4_value : a 4 = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_count_approximation_a_2_value_a_3_value_a_4_value_l1073_107312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_bar_cost_l1073_107351

/-- Given Dan's initial amount, the cost of chocolate, and the remaining amount after purchases,
    prove that the cost of the candy bar is $2. -/
theorem candy_bar_cost (initial : ℕ) (chocolate_cost : ℕ) (remaining : ℕ) : ℕ :=
  by
  -- Define the conditions
  have h1 : initial = 7 := by sorry
  have h2 : chocolate_cost = 3 := by sorry
  have h3 : remaining = 2 := by sorry

  -- Define the total spent
  let total_spent := initial - remaining

  -- Define the candy bar cost
  let candy_bar_cost := total_spent - chocolate_cost

  -- Prove that the candy bar cost is 2
  have h4 : candy_bar_cost = 2 := by
    calc
      candy_bar_cost = (initial - remaining) - chocolate_cost := rfl
      _ = (7 - 2) - 3 := by rw [h1, h2, h3]
      _ = 5 - 3 := by rfl
      _ = 2 := by rfl

  exact 2


end NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_bar_cost_l1073_107351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_of_2023_l1073_107357

theorem reciprocal_of_2023 : (1 : ℝ) / 2023 = 1 / 2023 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_of_2023_l1073_107357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_zero_functions_range_l1073_107358

noncomputable def f (x : ℝ) := Real.exp (x - 1) + x - 2

def g (a x : ℝ) := x^2 - a*x - a + 3

def adjacent_zero_functions (f g : ℝ → ℝ) : Prop :=
  ∃ α β : ℝ, f α = 0 ∧ g β = 0 ∧ |α - β| ≤ 1

theorem adjacent_zero_functions_range (a : ℝ) :
  adjacent_zero_functions f (g a) → a ∈ Set.Icc 2 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_zero_functions_range_l1073_107358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_operations_for_f_l1073_107338

/-- Represents a polynomial of degree 7 with integer coefficients -/
structure MyPolynomial where
  a₇ : ℤ
  a₆ : ℤ
  a₅ : ℤ
  a₄ : ℤ
  a₃ : ℤ
  a₂ : ℤ
  a₁ : ℤ
  a₀ : ℤ

/-- Counts the number of operations in Horner's method for a polynomial of degree n -/
def hornerOperations (n : ℕ) : (ℕ × ℕ) :=
  (n, n - 1)

/-- The specific polynomial f(x) = x^7 + 2x^5 + 3x^4 + 4x^3 + 5x^2 + 6x + 7 -/
def f : MyPolynomial :=
  ⟨1, 0, 2, 3, 4, 5, 6, 7⟩

theorem horner_operations_for_f :
  hornerOperations 7 = (7, 6) := by
  rfl

#eval hornerOperations 7

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_operations_for_f_l1073_107338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1073_107344

/-- The speed of a train in km/hr given its length in meters and time to pass a fixed point in seconds -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  (length / 1000) / (time / 3600)

/-- Theorem: A 280-meter long train passing a fixed point in 14 seconds has a speed of 72 km/hr -/
theorem train_speed_calculation :
  train_speed 280 14 = 72 := by
  -- Unfold the definition of train_speed
  unfold train_speed
  -- Simplify the expression
  simp [div_div]
  -- Perform the calculation
  norm_num
  -- QED


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1073_107344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_pattern_ratio_l1073_107300

/-- Represents a square pattern of tiles -/
structure SquarePattern where
  black_tiles : ℕ
  white_tiles : ℕ

/-- Represents an extended pattern with a border on three sides -/
structure ExtendedPattern where
  original : SquarePattern
  border_tiles : ℕ

/-- The ratio of black to white tiles in the extended pattern -/
def black_to_white_ratio (p : ExtendedPattern) : ℚ :=
  ↑(p.original.black_tiles + p.border_tiles) / ↑p.original.white_tiles

theorem extended_pattern_ratio : 
  ∀ (p : SquarePattern) (e : ExtendedPattern),
    p.black_tiles = 10 →
    p.white_tiles = 26 →
    e.original = p →
    e.border_tiles = 20 →
    black_to_white_ratio e = 30 / 26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_pattern_ratio_l1073_107300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_implies_a_equals_two_l1073_107392

-- Define the line and curve
def line (x : ℝ) : ℝ := x + 1

noncomputable def curve (a : ℝ) (x : ℝ) : ℝ := Real.log (x + a)

-- Define the tangency condition
def is_tangent (a : ℝ) : Prop :=
  ∃ x₀ : ℝ, 
    (line x₀ = curve a x₀) ∧ 
    (deriv line x₀ = deriv (curve a) x₀)

-- Theorem statement
theorem tangent_implies_a_equals_two : 
  ∀ a : ℝ, is_tangent a → a = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_implies_a_equals_two_l1073_107392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_checker_placement_theorem_l1073_107305

/-- Represents a 50x50 chess board with checkers -/
def Board := Fin 50 → Fin 50 → Bool

/-- The maximum number of new checkers that can be added -/
def MaxNewCheckers : Nat := 99

/-- Predicate to check if a row has an even number of checkers -/
def rowEven (b : Board) (row : Fin 50) : Prop :=
  Even (Finset.card (Finset.filter (fun col => b row col) Finset.univ))

/-- Predicate to check if a column has an even number of checkers -/
def colEven (b : Board) (col : Fin 50) : Prop :=
  Even (Finset.card (Finset.filter (fun row => b row col) Finset.univ))

/-- The main theorem -/
theorem checker_placement_theorem (initial : Board) :
  ∃ (final : Board),
    (∀ row col, initial row col → final row col) ∧
    (Finset.card (Finset.filter (fun p => final p.1 p.2 ∧ ¬initial p.1 p.2) (Finset.product Finset.univ Finset.univ)) ≤ MaxNewCheckers) ∧
    (∀ row, rowEven final row) ∧
    (∀ col, colEven final col) :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_checker_placement_theorem_l1073_107305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_k_value_l1073_107335

/-- A polynomial is a perfect square trinomial if it can be expressed as (ax + b)^2 -/
def is_perfect_square_trinomial (a b c : ℝ) : Prop :=
  ∃ (p q : ℝ), ∀ (x : ℝ), a * x^2 + b * x + c = (p * x + q)^2

/-- The main theorem: If 4x^2 + 2kx + 25 is a perfect square trinomial, then k = ±10 -/
theorem perfect_square_k_value (k : ℝ) :
  is_perfect_square_trinomial 4 (2*k) 25 → k = 10 ∨ k = -10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_k_value_l1073_107335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_intervals_f_range_in_interval_l1073_107319

noncomputable def a (x : ℝ) : ℝ × ℝ := (2 * Real.cos x, 1)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sqrt 3 * Real.sin (2 * x))

noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

def is_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

theorem f_decreasing_intervals (k : ℤ) :
  is_decreasing f (k * π + π / 6) (k * π + 2 * π / 3) := by sorry

theorem f_range_in_interval :
  ∀ x, -π/4 ≤ x ∧ x ≤ 0 → -Real.sqrt 3 + 1 ≤ f x ∧ f x ≤ 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_intervals_f_range_in_interval_l1073_107319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_f_analytic_expression_f_greater_than_g_plus_half_exists_a_for_min_value_l1073_107399

noncomputable section

def e : ℝ := Real.exp 1

def f (a : ℝ) : ℝ → ℝ
| x => if x > 0 then a * x + Real.log x else a * x - Real.log (-x)

def g : ℝ → ℝ
| x => Real.log (abs x) / abs x

theorem f_odd (a : ℝ) (x : ℝ) (h : x ≠ 0) : f a (-x) = -(f a x) := by sorry

theorem f_analytic_expression (a : ℝ) (x : ℝ) (h : x ∈ Set.Icc (-e) (-0) ∪ Set.Ioo 0 e) :
  f a x = if x > 0 then a * x + Real.log x else a * x - Real.log (-x) := by sorry

theorem f_greater_than_g_plus_half (x : ℝ) (h : x ∈ Set.Icc (-e) (-0)) :
  f (-1) x > g x + 1/2 := by sorry

theorem exists_a_for_min_value (h : ∃ (a : ℝ), ∀ (x : ℝ), x ∈ Set.Icc (-e) (-0) → f a x ≥ 3) :
  ∃ (a : ℝ), a = -(e^2) ∧ ∀ (x : ℝ), x ∈ Set.Icc (-e) (-0) → f a x ≥ 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_f_analytic_expression_f_greater_than_g_plus_half_exists_a_for_min_value_l1073_107399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_safe_volume_l1073_107329

/-- Represents the relationship between pressure and volume of a gas in a balloon. -/
structure BalloonGas where
  /-- The constant of proportionality for the inverse relationship between pressure and volume. -/
  k : ℝ
  /-- The pressure (in kPa) at which the balloon bursts. -/
  burstPressure : ℝ

/-- The volume (in m³) of gas in the balloon. -/
noncomputable def volume (b : BalloonGas) (p : ℝ) : ℝ := b.k / p

/-- The pressure (in kPa) inside the balloon given a volume of gas. -/
noncomputable def pressure (b : BalloonGas) (v : ℝ) : ℝ := b.k / v

theorem minimum_safe_volume (b : BalloonGas) 
    (h1 : pressure b 0.8 = 112.5)
    (h2 : b.burstPressure = 150) :
  volume b b.burstPressure = 0.6 := by
  sorry

#check minimum_safe_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_safe_volume_l1073_107329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_4_eq_11_div_2_l1073_107394

noncomputable section

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 4 / (3 - x)

-- Define the inverse function of f
noncomputable def f_inv (x : ℝ) : ℝ := (3 * x - 4) / x

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := 1 / (f_inv x) + 5

-- Theorem to prove
theorem g_of_4_eq_11_div_2 : g 4 = 11 / 2 := by
  -- Expand the definition of g
  unfold g
  -- Expand the definition of f_inv
  unfold f_inv
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_4_eq_11_div_2_l1073_107394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_22_point_5_l1073_107384

/-- A line passing through a point with a given slope -/
structure Line where
  point : ℝ × ℝ
  slope : ℝ

/-- The area of a triangle given its three vertices -/
noncomputable def triangleArea (a b c : ℝ × ℝ) : ℝ :=
  let (x₁, y₁) := a
  let (x₂, y₂) := b
  let (x₃, y₃) := c
  (1/2) * abs (x₁*(y₂ - y₃) + x₂*(y₃ - y₁) + x₃*(y₁ - y₂))

/-- Theorem stating that the area of the triangle formed by the intersection of the given lines is 22.5 -/
theorem triangle_area_is_22_point_5 
  (l₁ : Line) 
  (l₂ : Line) 
  (h₁ : l₁.point = (3, 3)) 
  (h₂ : l₁.slope = -1/3) 
  (h₃ : l₂.point = (3, 3)) 
  (h₄ : l₂.slope = 3) :
  ∃ (a b c : ℝ × ℝ), 
    (a ∈ {p : ℝ × ℝ | p.2 = -1/3 * (p.1 - 3) + 3}) ∧ 
    (b ∈ {p : ℝ × ℝ | p.2 = 3 * (p.1 - 3) + 3}) ∧
    (c ∈ {p : ℝ × ℝ | p.1 + p.2 = 12}) ∧
    triangleArea a b c = 22.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_22_point_5_l1073_107384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l1073_107395

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 2^x + 5*x

-- State the theorem
theorem root_in_interval :
  (∀ x y : ℝ, x < y → f x < f y) →  -- f is monotonically increasing
  f (-1) < 0 →
  f 0 > 0 →
  ∃ r : ℝ, r ∈ Set.Ioo (-1) 0 ∧ f r = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l1073_107395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_powers_l1073_107318

-- Define the condition
def condition (x : ℝ) : Prop := x * (Real.log 2 / Real.log 3) = 1

-- State the theorem
theorem sum_of_powers (x : ℝ) (h : condition x) : (2 : ℝ)^x + (2 : ℝ)^(-x) = 10/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_powers_l1073_107318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_einstein_pizza_sales_l1073_107360

/-- Calculates the number of pizza boxes sold given the prices, quantities, and goal -/
def calculate_pizzas_sold (pizza_price : ℚ) (fries_price : ℚ) (soda_price : ℚ)
  (fries_sold : ℕ) (soda_sold : ℕ) (goal : ℚ) (remaining : ℚ) : ℕ :=
  let current_earnings := goal - remaining
  let fries_earnings := fries_price * (fries_sold : ℚ)
  let soda_earnings := soda_price * (soda_sold : ℚ)
  let pizza_earnings := current_earnings - fries_earnings - soda_earnings
  (pizza_earnings / pizza_price).floor.toNat

theorem einstein_pizza_sales :
  calculate_pizzas_sold 12 (3/10) 2 40 25 500 258 = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_einstein_pizza_sales_l1073_107360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dartboard_even_score_probability_l1073_107387

/-- Represents a region on the dartboard -/
structure Region where
  area : ℝ
  value : ℕ

/-- Represents the dartboard -/
structure Dartboard where
  inner_radius : ℝ
  outer_radius : ℝ
  inner_regions : Fin 3 → Region
  outer_regions : Fin 3 → Region

/-- Calculates the probability of hitting a specific region -/
noncomputable def hit_probability (d : Dartboard) (r : Region) : ℝ :=
  r.area / (Real.pi * d.outer_radius^2)

/-- Calculates the probability of getting an even score with a single dart -/
noncomputable def even_score_probability (d : Dartboard) : ℝ :=
  let even_regions := [d.inner_regions 0, d.outer_regions 1, d.outer_regions 2]
  (even_regions.map (hit_probability d)).sum

/-- Calculates the probability of getting an odd score with a single dart -/
noncomputable def odd_score_probability (d : Dartboard) : ℝ :=
  let odd_regions := [d.inner_regions 1, d.inner_regions 2, d.outer_regions 0]
  (odd_regions.map (hit_probability d)).sum

/-- The main theorem to prove -/
theorem dartboard_even_score_probability (d : Dartboard) 
  (h_inner_radius : d.inner_radius = 4)
  (h_outer_radius : d.outer_radius = 9)
  (h_inner_regions : d.inner_regions = ![⟨16*Real.pi/3, 3⟩, ⟨16*Real.pi/3, 5⟩, ⟨16*Real.pi/3, 5⟩])
  (h_outer_regions : d.outer_regions = ![⟨65*Real.pi/3, 5⟩, ⟨65*Real.pi/3, 3⟩, ⟨65*Real.pi/3, 3⟩]) :
  (even_score_probability d)^2 + (odd_score_probability d)^2 = 30725/59049 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dartboard_even_score_probability_l1073_107387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1073_107367

open Real

theorem triangle_properties (a b c A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < π ∧ 
  0 < B ∧ B < π ∧ 
  0 < C ∧ C < π ∧
  A + B + C = π ∧
  c = 2 * b - 2 * a * cos C →
  (A = π / 3) ∧
  (a = 2 → ∃ (S : ℝ), S ≤ Real.sqrt 3 ∧ 
    ∀ (S' : ℝ), (∃ (b' c' : ℝ), S' = 1/2 * 2 * b' * sin A ∧ c' = 2 * b' - 2 * 2 * cos C) 
    → S' ≤ S) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1073_107367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_good_point_l1073_107308

/-- A point on the circle --/
structure Point where
  label : Int

/-- A circle with labeled points --/
structure LabeledCircle where
  points : List Point

/-- Definition of a good point --/
def is_good_point (circle : LabeledCircle) (start : Nat) : Prop :=
  ∀ (direction : Bool) (steps : Nat),
    (List.sum (List.map Point.label (List.take steps (if direction then circle.points.rotate start else (circle.points.rotate start).reverse)))) > 0

/-- The main theorem --/
theorem exists_good_point (circle : LabeledCircle) 
  (h : (circle.points.filter (fun p => p.label = -1)).length < 664)
  (total_points : circle.points.length = 1991) :
  ∃ (p : Nat), p < circle.points.length ∧ is_good_point circle p := by
  sorry

#check exists_good_point

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_good_point_l1073_107308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_negative_necessary_not_sufficient_l1073_107334

-- Define the property of being in the third or fourth quadrant
def in_third_or_fourth_quadrant (α : Real) : Prop :=
  Real.pi < α ∧ α < 2 * Real.pi

-- Define the theorem
theorem sin_negative_necessary_not_sufficient :
  (∀ α, in_third_or_fourth_quadrant α → Real.sin α < 0) ∧
  (∃ α, Real.sin α < 0 ∧ ¬in_third_or_fourth_quadrant α) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_negative_necessary_not_sufficient_l1073_107334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_inverse_l1073_107326

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := 4 * x + 5
noncomputable def g (x : ℝ) : ℝ := 3 * x - 4
noncomputable def h (x : ℝ) : ℝ := f (g x)

-- Define the inverse function
noncomputable def h_inv (x : ℝ) : ℝ := (x + 11) / 12

-- Theorem statement
theorem h_inverse : 
  ∀ x : ℝ, h (h_inv x) = x ∧ h_inv (h x) = x :=
by
  sorry

#check h_inverse

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_inverse_l1073_107326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_special_area_l1073_107348

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if the area S = (√3/4)(a^2 + b^2 - c^2), then C = π/3 -/
theorem triangle_special_area (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_area : (Real.sqrt 3 / 4) * (a^2 + b^2 - c^2) = (1/2) * a * b * Real.sin (Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b)))) :
  Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b)) = π/3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_special_area_l1073_107348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1073_107397

noncomputable def f (x : ℝ) := Real.sin (2 * x + Real.pi / 3) - Real.sqrt 3 * Real.sin (2 * x - Real.pi / 6)

theorem f_properties :
  -- Smallest positive period is π
  (∀ x, f (x + Real.pi) = f x) ∧
  (∀ T, T > 0 → (∀ x, f (x + T) = f x) → T ≥ Real.pi) ∧
  -- Monotonic increase intervals
  (∀ k : ℤ, ∀ x ∈ Set.Icc (-7 * Real.pi / 12 + ↑k * Real.pi) (-Real.pi / 12 + ↑k * Real.pi),
    ∀ y ∈ Set.Icc (-7 * Real.pi / 12 + ↑k * Real.pi) (-Real.pi / 12 + ↑k * Real.pi),
    x ≤ y → f x ≤ f y) ∧
  -- Maximum and minimum on [-π/6, π/3]
  (∀ x ∈ Set.Icc (-Real.pi / 6) (Real.pi / 3), f x ≤ 2) ∧
  (f (-Real.pi / 12) = 2) ∧
  (∀ x ∈ Set.Icc (-Real.pi / 6) (Real.pi / 3), f x ≥ -Real.sqrt 3) ∧
  (f (Real.pi / 3) = -Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1073_107397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_for_twin_prime_diophantine_l1073_107316

/-- Twin primes are prime numbers that differ by 2 -/
def TwinPrimes (p q : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧ q = p + 2

/-- The Diophantine equation n! + pq^2 = (mp)^2 -/
def DiophantineEq (p q m n : ℕ) : Prop :=
  Nat.factorial n + p * q^2 = (m * p)^2

theorem unique_solution_for_twin_prime_diophantine :
  ∃! (p q m n : ℕ), TwinPrimes p q ∧ DiophantineEq p q m n ∧ m ≥ 1 ∧ n ≥ 1 ∧
    p = 3 ∧ q = 5 ∧ m = 3 ∧ n = 3 := by
  sorry

#check unique_solution_for_twin_prime_diophantine

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_for_twin_prime_diophantine_l1073_107316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_walnut_trees_cut_l1073_107365

def logs_per_pine : ℕ := 80
def logs_per_maple : ℕ := 60
def logs_per_walnut : ℕ := 100
def pine_trees_cut : ℕ := 8
def maple_trees_cut : ℕ := 3
def total_logs : ℕ := 1220

theorem walnut_trees_cut :
  (total_logs - (pine_trees_cut * logs_per_pine + maple_trees_cut * logs_per_maple)) / logs_per_walnut = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_walnut_trees_cut_l1073_107365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_curves_l1073_107361

-- Define the curves C₁ and C₂
def C₁ (x y : ℝ) : Prop := x^2/8 + y^2/4 = 1
def C₂ (x y : ℝ) : Prop := x - Real.sqrt 2 * y - 4 = 0

-- Define the distance function between two points
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

-- Theorem statement
theorem min_distance_between_curves :
  ∃ (x₁ y₁ x₂ y₂ : ℝ), C₁ x₁ y₁ ∧ C₂ x₂ y₂ ∧ 
  (∀ (x₃ y₃ x₄ y₄ : ℝ), C₁ x₃ y₃ → C₂ x₄ y₄ → 
    distance x₁ y₁ x₂ y₂ ≤ distance x₃ y₃ x₄ y₄) ∧
  distance x₁ y₁ x₂ y₂ = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_curves_l1073_107361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_property_l1073_107356

theorem tangent_line_property (x₁ x₂ : ℝ) (h₁ : x₁ > 0) (h₂ : x₁ ≠ 1) :
  (∃ y₁ y₂ : ℝ, 
    y₁ = Real.log x₁ ∧ 
    y₂ = Real.exp x₂ ∧
    (λ x => (x - x₁) / x₁ + Real.log x₁) = (λ x => Real.exp x₂ * (x - x₂) + Real.exp x₂)) →
  2 / (x₁ - 1) + x₂ = -1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_property_l1073_107356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bad_carrots_count_l1073_107307

theorem bad_carrots_count (carol_carrots : ℕ) (mom_carrots : ℕ) (good_carrots : ℕ) : ℕ :=
  by
  -- Define the conditions
  have h1 : carol_carrots = 29 := by sorry
  have h2 : mom_carrots = 16 := by sorry
  have h3 : good_carrots = 38 := by sorry

  -- Define the total number of carrots
  let total_carrots := carol_carrots + mom_carrots

  -- Calculate the number of bad carrots
  let bad_carrots := total_carrots - good_carrots

  -- Prove that the number of bad carrots is 7
  have h4 : bad_carrots = 7 := by sorry

  -- Return the result
  exact bad_carrots

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bad_carrots_count_l1073_107307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_of_2023_l1073_107322

theorem opposite_of_2023 : 
  (∀ x : ℤ, x > 0 → -x = -x) → 
  2023 > 0 → 
  -2023 = -2023 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_of_2023_l1073_107322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_and_range_l1073_107368

noncomputable def f (x : ℝ) : ℝ := Real.cos (x + Real.pi / 12) ^ 2
noncomputable def g (x : ℝ) : ℝ := 1 + (1 / 2) * Real.sin (2 * x)
noncomputable def h (x : ℝ) : ℝ := f x + g x

theorem symmetry_and_range (x₀ : ℝ) (m : ℝ) : 
  (∀ x, f (x₀ + x) = f (x₀ - x)) →
  (∃ k : ℤ, g x₀ = (3 + (-1) ^ k) / 4) ∧
  ((∀ x ∈ Set.Icc (-Real.pi / 12) (5 * Real.pi / 12), |h x - m| ≤ 1) → 
   m ∈ Set.Icc 1 (9 / 4)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_and_range_l1073_107368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_small_cone_altitude_is_18_l1073_107386

/-- Represents a frustum of a right circular cone -/
structure Frustum where
  altitude : ℝ
  lower_base_area : ℝ
  upper_base_area : ℝ

/-- Calculates the altitude of the small cone cut off from a right circular cone to form a frustum -/
noncomputable def small_cone_altitude (f : Frustum) : ℝ :=
  f.altitude / 2

/-- Theorem stating that the altitude of the small cone is 18 cm for the given frustum -/
theorem small_cone_altitude_is_18 (f : Frustum)
  (h1 : f.altitude = 36)
  (h2 : f.lower_base_area = 324 * Real.pi)
  (h3 : f.upper_base_area = 36 * Real.pi) :
  small_cone_altitude f = 18 := by
  sorry

#check small_cone_altitude_is_18

end NUMINAMATH_CALUDE_ERRORFEEDBACK_small_cone_altitude_is_18_l1073_107386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_employees_needed_is_245_l1073_107304

/-- Represents the number of employees monitoring different categories --/
structure EmployeeCount where
  seaTurtles : ℕ
  birdMigration : ℕ
  endangeredPlants : ℕ
  seaTurtlesAndBirdMigration : ℕ
  seaTurtlesAndEndangeredPlants : ℕ
  allThree : ℕ

/-- Calculates the minimum number of employees needed given the conditions --/
def minEmployeesNeeded (e : EmployeeCount) : ℕ :=
  let onlySeaTurtles := e.seaTurtles - (e.seaTurtlesAndBirdMigration + e.seaTurtlesAndEndangeredPlants - e.allThree)
  let onlyBirdMigration := e.birdMigration - (e.seaTurtlesAndBirdMigration + e.allThree - e.seaTurtlesAndEndangeredPlants)
  onlySeaTurtles + onlyBirdMigration + e.seaTurtlesAndBirdMigration + e.seaTurtlesAndEndangeredPlants + e.allThree

/-- The theorem stating the minimum number of employees needed --/
theorem min_employees_needed_is_245 (e : EmployeeCount)
  (h1 : e.seaTurtles = 120)
  (h2 : e.birdMigration = 90)
  (h3 : e.seaTurtlesAndBirdMigration = 30)
  (h4 : e.seaTurtlesAndEndangeredPlants = 50)
  (h5 : e.allThree = 15) :
  minEmployeesNeeded e = 245 := by
  sorry

#eval minEmployeesNeeded { 
  seaTurtles := 120, 
  birdMigration := 90, 
  endangeredPlants := 0,
  seaTurtlesAndBirdMigration := 30, 
  seaTurtlesAndEndangeredPlants := 50, 
  allThree := 15 
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_employees_needed_is_245_l1073_107304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eighth_step_shaded_fraction_l1073_107320

/-- The number of triangles added in step n -/
def triangles_added (n : ℕ) : ℕ := 8^n

/-- The total number of triangles up to step n -/
def total_triangles (n : ℕ) : ℕ :=
  Finset.sum (Finset.range n) (fun k => triangles_added (k + 1))

/-- The number of shaded triangles in step n -/
def shaded_triangles (n : ℕ) : ℕ := n.factorial

theorem eighth_step_shaded_fraction :
  (shaded_triangles 8 : ℚ) / (total_triangles 8 : ℚ) = 315 / 4096 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eighth_step_shaded_fraction_l1073_107320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_sum_parity_difference_l1073_107317

/-- The number of positive integer divisors of n, including 1 and n -/
def τ (n : ℕ+) : ℕ := sorry

/-- The sum of τ(k) for k from 1 to n -/
def S (n : ℕ+) : ℕ := sorry

/-- The number of positive integers n ≤ 1000 with S(n) odd -/
def a : ℕ := sorry

/-- The number of positive integers n ≤ 1000 with S(n) even -/
def b : ℕ := sorry

theorem divisor_sum_parity_difference :
  |Int.ofNat a - Int.ofNat b| = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_sum_parity_difference_l1073_107317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_45_minus_cos_135_l1073_107385

theorem cos_45_minus_cos_135 : Real.cos (45 * Real.pi / 180) - Real.cos (135 * Real.pi / 180) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_45_minus_cos_135_l1073_107385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equally_illuminated_points_l1073_107390

/-- Represents a light source with a given intensity -/
structure LightSource where
  intensity : ℝ

/-- Represents a point on an infinite line -/
structure Point where
  distance : ℝ

/-- Theorem: Equally illuminated points on a line between two light sources -/
theorem equally_illuminated_points
  (A B : LightSource)
  (d : ℝ)
  (h_d_pos : d > 0)
  (h_A_pos : A.intensity > 0)
  (h_B_pos : B.intensity > 0) :
  ∃ (x₁ x₂ : Point),
    x₁.distance = d * Real.sqrt A.intensity / (Real.sqrt A.intensity - Real.sqrt B.intensity) ∧
    x₂.distance = d * Real.sqrt A.intensity / (Real.sqrt A.intensity + Real.sqrt B.intensity) ∧
    (A.intensity / x₁.distance^2 = B.intensity / (d - x₁.distance)^2) ∧
    (A.intensity / x₂.distance^2 = B.intensity / (d - x₂.distance)^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equally_illuminated_points_l1073_107390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_B_triangle_height_l1073_107374

noncomputable section

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def satisfiesConditions (t : Triangle) : Prop :=
  2 * t.c = 2 * t.b * Real.cos t.A - t.a ∧
  t.a + t.c = 8 ∧
  t.b = 7 ∧
  t.C > t.A

-- Define the height function
def triangleHeight (t : Triangle) : Real :=
  (t.a * t.c * Real.sin t.B) / t.b

-- Theorem statements
theorem triangle_angle_B (t : Triangle) :
  2 * t.c = 2 * t.b * Real.cos t.A - t.a → t.B = 2 * Real.pi / 3 := by
  sorry

theorem triangle_height (t : Triangle) :
  satisfiesConditions t → triangleHeight t = 15 * Real.sqrt 3 / 14 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_B_triangle_height_l1073_107374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_fourth_quadrant_l1073_107340

noncomputable def z : ℂ := (Complex.I + 2) / Complex.I

theorem z_in_fourth_quadrant :
  Real.sign z.re = 1 ∧ Real.sign z.im = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_fourth_quadrant_l1073_107340
