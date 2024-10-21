import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_chords_l702_70261

-- Define the basic structures
structure Ray where
  start : EuclideanSpace ℝ (Fin 2)
  direction : EuclideanSpace ℝ (Fin 2)

structure Circle where
  center : EuclideanSpace ℝ (Fin 2)
  radius : ℝ

-- Define the given elements
variable (a b f : Ray)
variable (k1 k2 : Circle)
variable (A B : EuclideanSpace ℝ (Fin 2))

-- Define necessary predicates
def IsBisector (f a b : Ray) : Prop := sorry
def IsTangent (c : Circle) (r : Ray) : Prop := sorry
def PointOnRay (p : EuclideanSpace ℝ (Fin 2)) (r : Ray) : Prop := sorry
def TouchPoint (c : Circle) (r : Ray) : EuclideanSpace ℝ (Fin 2) := sorry

-- State the conditions
axiom f_is_bisector : IsBisector f a b
axiom k1_tangent_a_f : IsTangent k1 a ∧ IsTangent k1 f
axiom k2_tangent_b_f : IsTangent k2 b ∧ IsTangent k2 f
axiom A_on_a : PointOnRay A a
axiom B_on_b : PointOnRay B b
axiom k1_touches_a_at_A : TouchPoint k1 a = A
axiom k2_touches_b_at_B : TouchPoint k2 b = B

-- Define necessary geometric constructions
def Line.throughPoints (p q : EuclideanSpace ℝ (Fin 2)) : Set (EuclideanSpace ℝ (Fin 2)) := sorry
def Chord.fromLineIntersection (c : Circle) (l : Set (EuclideanSpace ℝ (Fin 2))) : Set (EuclideanSpace ℝ (Fin 2)) := sorry
def length (s : Set (EuclideanSpace ℝ (Fin 2))) : ℝ := sorry

-- Define the theorem
theorem equal_chords :
  let AB := Line.throughPoints A B
  let chord1 := Chord.fromLineIntersection k1 AB
  let chord2 := Chord.fromLineIntersection k2 AB
  length chord1 = length chord2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_chords_l702_70261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_intersection_theorem_l702_70250

/-- Represents a segment on a line -/
structure Segment where
  color : Bool  -- True for white, False for black
  start : ℝ
  stop : ℝ

/-- Checks if two segments intersect -/
noncomputable def intersect (s1 s2 : Segment) : Bool :=
  (s1.start < s2.stop ∧ s2.start < s1.stop) ∨ (s2.start < s1.stop ∧ s1.start < s2.stop)

/-- Theorem: Given 2k-1 white segments and 2k-1 black segments on a line,
    where each white segment intersects at least k black segments and
    each black segment intersects at least k white segments,
    there exists a black segment intersecting all white segments and
    a white segment intersecting all black segments -/
theorem segment_intersection_theorem (k : ℕ) (segments : List Segment)
    (h1 : segments.length = 4 * k - 2)
    (h2 : (segments.filter (λ s => s.color)).length = 2 * k - 1)
    (h3 : (segments.filter (λ s => ¬s.color)).length = 2 * k - 1)
    (h4 : ∀ s ∈ segments, s.color →
      (segments.filter (λ t => ¬t.color ∧ intersect s t)).length ≥ k)
    (h5 : ∀ s ∈ segments, ¬s.color →
      (segments.filter (λ t => t.color ∧ intersect s t)).length ≥ k) :
    (∃ s ∈ segments, ¬s.color ∧
      ∀ t ∈ segments, t.color → intersect s t) ∧
    (∃ s ∈ segments, s.color ∧
      ∀ t ∈ segments, ¬t.color → intersect s t) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_intersection_theorem_l702_70250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ian_says_1306_ian_number_unique_l702_70251

/-- The number of students participating in the counting game -/
def num_students : ℕ := 9

/-- The upper limit of the counting range -/
def upper_limit : ℕ := 2000

/-- Represents the pattern for each student's count -/
def student_pattern : ℕ → ℕ → ℕ
| 0, k => 4 * k - 2
| n + 1, k => student_pattern n (4 * k - 2)

/-- The number that Ian says -/
def ian_number : ℕ := student_pattern (num_students - 1) 1

theorem ian_says_1306 : ian_number = 1306 := by
  sorry

theorem ian_number_unique : 
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ upper_limit → 
    (k = ian_number ↔ ∀ i : ℕ, i < num_students - 1 → ∃ j : ℕ, student_pattern i j = k) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ian_says_1306_ian_number_unique_l702_70251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farm_animals_l702_70257

theorem farm_animals (total : ℕ) (chicken_percent : ℚ) (goat_cow_ratio : ℕ) :
  total = 500 →
  chicken_percent = 1/10 →
  goat_cow_ratio = 2 →
  ∃ (chickens cows goats : ℕ),
    chickens + cows + goats = total ∧
    chickens = (chicken_percent * ↑total).floor ∧
    goats = goat_cow_ratio * cows ∧
    cows = 150 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_farm_animals_l702_70257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_g_squared_area_between_f_g_l702_70263

noncomputable def f (x : ℝ) := Real.sin (3 * x) + Real.cos x
noncomputable def g (x : ℝ) := Real.cos (3 * x) + Real.sin x

theorem integral_f_g_squared : 
  ∫ x in (0)..(2 * Real.pi), (f x)^2 + (g x)^2 = 4 * Real.pi := by sorry

theorem area_between_f_g : 
  ∫ x in (0)..(Real.pi), |f x - g x| = (8 * Real.sqrt (2 + Real.sqrt 2) - 4) / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_g_squared_area_between_f_g_l702_70263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_minus_beta_eq_neg_one_l702_70237

theorem tan_alpha_minus_beta_eq_neg_one
  (α β : ℝ)
  (h : Real.sin (α + β) + Real.cos (α + β) = 2 * Real.sqrt 2 * Real.cos (α + π / 4) * Real.sin β) :
  Real.tan (α - β) = -1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_minus_beta_eq_neg_one_l702_70237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_phi_l702_70286

/-- The original function f(x) -/
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) + Real.cos (2 * x)

/-- The shifted function g(x) -/
noncomputable def g (x φ : ℝ) : ℝ := f (x + φ)

/-- Condition for symmetry about y-axis -/
def symmetric_about_y_axis (g : ℝ → ℝ) : Prop :=
  ∀ x, g x = g (-x)

theorem smallest_positive_phi :
  ∃ φ : ℝ, φ > 0 ∧
    symmetric_about_y_axis (g · φ) ∧
    (∀ ψ, ψ > 0 → symmetric_about_y_axis (g · ψ) → φ ≤ ψ) ∧
    φ = Real.pi / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_phi_l702_70286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_area_l702_70271

/-- The area of a triangle inscribed in a circle with radius 2 cm, 
    having two angles of π/3 and π/4, is equal to √3 + 3 cm². -/
theorem inscribed_triangle_area (R : ℝ) (angle1 angle2 : ℝ) (h1 : R = 2) 
    (h2 : angle1 = π / 3) (h3 : angle2 = π / 4) : 
  let S := (R^2 * Real.sin angle1 * Real.sin angle2 * Real.sin (π - angle1 - angle2)) / (2 * Real.sin (π / 2))
  S = Real.sqrt 3 + 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_area_l702_70271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_possible_integer_l702_70234

def median (l : List Int) : Int :=
  sorry

def mean (l : List Int) : Int :=
  sorry

def mode (l : List Int) : Int :=
  sorry

def is_valid_list (l : List Int) : Prop :=
  l.length = 5 ∧
  median l = 10 ∧
  median l = mean l + 1 ∧
  mode l = median l + 1

theorem smallest_possible_integer (l : List Int) (h : is_valid_list l) :
  l.minimum ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_possible_integer_l702_70234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tangent_ratio_l702_70291

theorem triangle_tangent_ratio (a b c A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  a / b + b / a = 4 * Real.cos C →
  Real.tan C / Real.tan A + Real.tan C / Real.tan B = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tangent_ratio_l702_70291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cats_bowl_problem_l702_70222

/-- The cat's bowl problem -/
theorem cats_bowl_problem (empty_bowl_weight : ℝ) (daily_food : ℝ) (days : ℝ) (weight_after_eating : ℝ)
  (h1 : empty_bowl_weight = 420)
  (h2 : daily_food = 60)
  (h3 : days = 3)
  (h4 : weight_after_eating = 586) :
  weight_after_eating - empty_bowl_weight - (daily_food * days - 14) = 0 :=
by
  -- Substitute known values
  subst h1 h2 h3 h4
  -- Simplify the expression
  ring
  -- The proof is complete

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cats_bowl_problem_l702_70222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isaac_journey_time_l702_70221

/-- Represents the bicycle journey of Mr. Isaac -/
structure BicycleJourney where
  rate : ℚ  -- Speed in miles per hour
  initialTime : ℚ  -- Initial riding time in minutes
  secondDistance : ℚ  -- Distance of second leg in miles
  restTime : ℚ  -- Rest time in minutes
  finalDistance : ℚ  -- Distance of final leg in miles

/-- Calculates the total time of the journey in minutes -/
def totalJourneyTime (journey : BicycleJourney) : ℚ :=
  journey.initialTime +
  (journey.secondDistance / journey.rate) * 60 +
  journey.restTime +
  (journey.finalDistance / journey.rate) * 60

/-- Theorem stating that Mr. Isaac's journey takes 270 minutes -/
theorem isaac_journey_time :
  let journey : BicycleJourney := {
    rate := 10
    initialTime := 30
    secondDistance := 15
    restTime := 30
    finalDistance := 20
  }
  totalJourneyTime journey = 270 := by
  -- Unfold the definition and simplify
  unfold totalJourneyTime
  simp
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isaac_journey_time_l702_70221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_line_and_axes_l702_70298

/-- Given two lines that intersect on the y-axis, calculate the area enclosed by one line and the coordinate axes -/
theorem area_enclosed_by_line_and_axes 
  (b : ℝ) 
  (h : ∃ (y : ℝ), y = 3 * 0 + b ∧ y = -(0) + 2) : 
  (1 / 2) * |(-b / 3)| * b = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_line_and_axes_l702_70298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_a_range_l702_70273

/-- The range of 'a' for a quadratic function f(x) = ax^2 + bx + 1
    satisfying f(-1) = 1 and f(x) < 2 for all real x -/
theorem quadratic_function_a_range :
  ∀ (a b : ℝ),
    (∀ x : ℝ, a * x^2 + b * x + 1 < 2) →
    (a * (-1)^2 + b * (-1) + 1 = 1) →
    a ∈ Set.Ioc (-4) 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_a_range_l702_70273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_bisectors_l702_70219

-- Define a right triangle with legs 24 and 18
structure RightTriangle where
  leg1 : ℝ
  leg2 : ℝ
  is_right : leg1 = 24 ∧ leg2 = 18

-- Define the angle bisector lengths
noncomputable def angle_bisector1 (t : RightTriangle) : ℝ := 9 * Real.sqrt 5
noncomputable def angle_bisector2 (t : RightTriangle) : ℝ := 8 * Real.sqrt 10

-- Theorem statement
theorem right_triangle_bisectors (t : RightTriangle) :
  (angle_bisector1 t = 9 * Real.sqrt 5) ∧
  (angle_bisector2 t = 8 * Real.sqrt 10) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_bisectors_l702_70219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_in_special_triangle_l702_70215

/-- Given a triangle ABC with side lengths in the ratio 3:5:7, 
    the largest angle is 120 degrees. -/
theorem largest_angle_in_special_triangle :
  ∀ (a b c : ℝ) (k : ℝ),
    a = 3 * k →
    b = 5 * k →
    c = 7 * k →
    k > 0 →
    ∃ (A B C : ℝ),
      A + B + C = π ∧
      Real.cos A = (b^2 + c^2 - a^2) / (2*b*c) ∧
      Real.cos B = (a^2 + c^2 - b^2) / (2*a*c) ∧
      Real.cos C = (a^2 + b^2 - c^2) / (2*a*b) ∧
      max A (max B C) = 2*π/3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_in_special_triangle_l702_70215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sin_value_l702_70231

noncomputable def f (x φ : ℝ) : ℝ := Real.sin (Real.pi * x + φ)

theorem max_sin_value (φ : ℝ) (h1 : 0 < φ) (h2 : φ < 2 * Real.pi) :
  (∀ x, f x φ ≤ f 2 φ) → φ = Real.pi / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sin_value_l702_70231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_of_f_between_one_and_two_l702_70218

-- Define the function f(x) as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2 - 1 / x

-- State the theorem
theorem zero_of_f_between_one_and_two :
  ∃ x₀ : ℝ, f x₀ = 0 ∧ 1 < x₀ ∧ x₀ < 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_of_f_between_one_and_two_l702_70218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_feeding_order_count_l702_70210

/-- Represents a pair of animals in the sanctuary -/
structure AnimalPair :=
  (male : String)
  (female : String)

/-- Represents the feeding order -/
def FeedingOrder := List String

def sanctuary_animals : List AnimalPair := sorry

/-- Check if the feeding order is valid according to the rules -/
def is_valid_feeding (order : FeedingOrder) : Bool := sorry

/-- Count the number of valid feeding orders -/
def count_valid_feedings : Nat := sorry

/-- Helper function to determine if a string represents a male animal -/
def isMale (s : String) : Bool := sorry

theorem feeding_order_count :
  sanctuary_animals.length = 6 ∧
  (∃ pair ∈ sanctuary_animals, pair.male = "lion") ∧
  (∃ pair ∈ sanctuary_animals, pair.male = "giraffe") ∧
  (∀ order : FeedingOrder, is_valid_feeding order →
    order.head? = some "lion" ∧
    (∀ i, i < order.length - 1 →
      (order.get? i = some "giraffe" → order.get? (i+1) ≠ some "giraffe")) ∧
    (∀ i, i < order.length - 1 →
      (isMale (order.get! i)) ≠ (isMale (order.get! (i+1))))) →
  count_valid_feedings = 7200 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_feeding_order_count_l702_70210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_club_student_equality_l702_70275

/-- In a school with clubs and students, where each club has exactly 3 members
    and each student belongs to exactly 3 clubs, the number of clubs equals
    the number of students. -/
theorem club_student_equality (S : ℕ) (C : ℕ) 
  (number_of_members : ℕ → ℕ) (number_of_clubs : ℕ → ℕ) : 
  (∀ c, c ≤ C → (number_of_members c = 3)) →
  (∀ s, s ≤ S → (number_of_clubs s = 3)) →
  C = S := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_club_student_equality_l702_70275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_california_space_per_person_l702_70220

/-- The population of California in 2020 --/
def california_population : ℕ := 39500000

/-- The area of California in square miles --/
def california_area : ℕ := 163696

/-- The number of square feet in one square mile --/
def square_feet_per_mile : ℕ := 5280 * 5280

/-- The average number of square feet per person in California --/
def average_square_feet_per_person : ℚ :=
  (california_area * square_feet_per_mile : ℚ) / california_population

theorem california_space_per_person :
  |average_square_feet_per_person - 115000| < 1000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_california_space_per_person_l702_70220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l702_70233

-- Define the constants based on the given conditions
noncomputable def a : ℝ := Real.log 2 / Real.log 3
noncomputable def b : ℝ := Real.log 2
noncomputable def c : ℝ := 2^(0.3 : ℝ)

-- State the theorem to be proved
theorem relationship_abc : c > b ∧ b > a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l702_70233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equals_two_l702_70248

noncomputable def f (x : ℝ) : ℝ := (4 * Real.sqrt x) / (x^2 * Real.sqrt (x - 1))

theorem integral_equals_two :
  ∫ x in (16/15)..(4/3), f x = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equals_two_l702_70248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_radius_is_two_given_triangle_circumcircle_radius_l702_70225

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  AB : ℝ
  BC : ℝ
  CA : ℝ
  area : ℝ

-- Define the given triangle
noncomputable def given_triangle : Triangle where
  A := Real.pi / 3
  AB := 4
  area := 2 * Real.sqrt 3
  -- Other fields are left undefined as they are not given in the problem
  B := 0  -- placeholder value
  C := 0  -- placeholder value
  BC := 0 -- placeholder value
  CA := 0 -- placeholder value

-- Define the circumcircle radius function
noncomputable def circumcircle_radius (t : Triangle) : ℝ :=
  sorry  -- The actual implementation would depend on how we define triangles and circles

-- Theorem statement
theorem circumcircle_radius_is_two (t : Triangle) 
  (h1 : t.A = Real.pi / 3) 
  (h2 : t.AB = 4) 
  (h3 : t.area = 2 * Real.sqrt 3) : 
  circumcircle_radius t = 2 := by
  sorry

-- Proof that the given triangle's circumcircle radius is 2
theorem given_triangle_circumcircle_radius :
  circumcircle_radius given_triangle = 2 := by
  apply circumcircle_radius_is_two given_triangle
  · rfl  -- proves h1
  · rfl  -- proves h2
  · rfl  -- proves h3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_radius_is_two_given_triangle_circumcircle_radius_l702_70225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_categorization_l702_70255

-- Define the set of given numbers
def given_numbers : Set ℝ := {0.12, 0, -9, -6.8, 2-Real.pi, 13/9, -Real.sqrt 2, 4/5, 9, 0.7373373337, 3}

-- Define the subsets
def irrational_numbers : Set ℝ := {x ∈ given_numbers | ¬∃ (q : ℚ), (x : ℝ) = q}
def integer_numbers : Set ℝ := {x ∈ given_numbers | ∃ (n : ℤ), (n : ℝ) = x}
def fraction_numbers : Set ℝ := {x ∈ given_numbers | ∃ (q : ℚ), (q : ℝ) = x ∧ x ∉ integer_numbers}
def real_numbers : Set ℝ := given_numbers

-- State the theorem
theorem number_categorization :
  irrational_numbers = {2-Real.pi, -Real.sqrt 2, 0.7373373337} ∧
  integer_numbers = {0, -9, 9, 3} ∧
  fraction_numbers = {0.12, -6.8, 13/9, 4/5} ∧
  real_numbers = given_numbers :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_categorization_l702_70255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_range_l702_70249

-- Define the function f
noncomputable def f (ω : ℝ) (φ : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

-- State the theorem
theorem phi_range (ω : ℝ) (φ : ℝ) :
  ω > 0 →
  |φ| ≤ π/2 →
  (∀ x : ℝ, -π/12 < x ∧ x < π/3 → f ω φ x > 1) →
  (∀ x : ℝ, f ω φ (x + π) = f ω φ x) →
  π/4 ≤ φ ∧ φ ≤ π/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_range_l702_70249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l702_70268

/-- Represents an ellipse with given properties -/
structure Ellipse where
  eccentricity : ℝ
  major_axis_length : ℝ
  foci_on_x_axis : Bool

/-- Represents a line with a given slope -/
structure Line where
  slope : ℝ

/-- Represents a point of intersection between an ellipse and a line -/
structure IntersectionPoint where
  x : ℝ
  y : ℝ

/-- Given an ellipse C and a line l, computes the length of the chord AB formed by their intersection -/
noncomputable def chord_length (C : Ellipse) (l : Line) (A B : IntersectionPoint) : ℝ :=
  Real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2)

theorem ellipse_properties (C : Ellipse) (l : Line) (A B : IntersectionPoint) 
  (h1 : C.eccentricity = Real.sqrt 3 / 2)
  (h2 : C.major_axis_length = 4)
  (h3 : C.foci_on_x_axis = true)
  (h4 : l.slope = 1) :
  (∃ (a b : ℝ), a = 2 ∧ b = 1 ∧ ∀ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1 ↔ x^2 / 4 + y^2 = 1)) ∧ 
  (∃ (max_length : ℝ), max_length = 4 * Real.sqrt 10 / 5 ∧ 
    ∀ (A' B' : IntersectionPoint), chord_length C l A' B' ≤ max_length) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l702_70268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_score_and_optimal_time_optimal_time_constraints_l702_70292

/-- Represents the score function for a math test with two parts. -/
noncomputable def score_function (x : ℝ) : ℝ := -(1/5) * x + 2 * Real.sqrt (3 * x) + 125

/-- Theorem stating the maximum score and optimal time allocation for Part II. -/
theorem max_score_and_optimal_time :
  let max_score := 140
  let optimal_time_part_II := 75
  (∀ x ∈ Set.Icc 20 100, score_function x ≤ max_score) ∧
  score_function optimal_time_part_II = max_score := by
  sorry

/-- Verifies that the optimal time allocation satisfies the test constraints. -/
theorem optimal_time_constraints :
  let total_time := 120
  let optimal_time_part_II := 75
  optimal_time_part_II ∈ Set.Icc 20 100 ∧
  (total_time - optimal_time_part_II) ∈ Set.Icc 20 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_score_and_optimal_time_optimal_time_constraints_l702_70292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_decreasing_interval_l702_70224

-- Define the power function
noncomputable def power_function (α : ℝ) (x : ℝ) : ℝ := x ^ α

-- State the theorem
theorem power_function_decreasing_interval 
  (α : ℝ) 
  (h : power_function α 3 = 1/9) :
  ∀ x y, 0 < x ∧ x < y → power_function α x > power_function α y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_decreasing_interval_l702_70224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_interval_f_domain_l702_70247

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (-x^2 + 2*x + 3)

-- State the theorem
theorem f_decreasing_on_interval :
  ∀ x y, x ∈ Set.Icc 1 3 → y ∈ Set.Icc 1 3 → x < y → f x > f y :=
by sorry

-- Define the domain of f
def domain_f : Set ℝ := Set.Icc (-1) 3

-- State that the domain of f is [-1, 3]
theorem f_domain : 
  ∀ x, f x ≠ 0 → x ∈ domain_f :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_interval_f_domain_l702_70247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_properties_verify_k_and_b_verify_a_verify_m_l702_70205

/-- A linear function passing through (-1, -1) and (1, 3) -/
def linear_function (x : ℝ) : ℝ := 2 * x + 1

theorem linear_function_properties :
  let f := linear_function
  -- The function passes through (-1, -1) and (1, 3)
  (f (-1) = -1 ∧ f 1 = 3) ∧
  -- The function passes through (0, 1)
  f 0 = 1 ∧
  -- The function passes through (2, 5)
  f 2 = 5 := by
  sorry

/-- The slope of the linear function -/
def k : ℝ := 2

/-- The y-intercept of the linear function -/
def b : ℝ := 1

/-- The x-coordinate where y = 5 -/
def a : ℝ := 2

/-- The y-coordinate where x = 0 -/
def m : ℝ := 1

theorem verify_k_and_b :
  linear_function = λ x ↦ k * x + b := by
  sorry

theorem verify_a : linear_function a = 5 := by
  sorry

theorem verify_m : linear_function 0 = m := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_properties_verify_k_and_b_verify_a_verify_m_l702_70205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_of_arithmetic_sequence_l702_70245

/-- Definition of arithmetic sequence for three terms -/
def ArithmeticSeq (a b c : ℝ) : Prop := b - a = c - b

/-- If k, -1, and b form an arithmetic sequence, then the line y = kx + b passes through (1, -2) -/
theorem fixed_point_of_arithmetic_sequence (k b : ℝ) : 
  ArithmeticSeq k (-1) b → (1, -2) ∈ {p : ℝ × ℝ | p.2 = k * p.1 + b} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_of_arithmetic_sequence_l702_70245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bouncing_projectile_motion_l702_70299

/-- Represents the parameters of a bouncing projectile motion --/
structure BouncingProjectile where
  v₀ : ℝ             -- Initial velocity
  α : ℝ              -- Angle of projection
  ε : ℝ              -- Coefficient of restitution
  g : ℝ              -- Acceleration due to gravity

/-- Calculates the total time of bouncing for a projectile --/
noncomputable def totalTime (p : BouncingProjectile) : ℝ :=
  (2 * p.v₀ * Real.sin p.α) / (p.g * (1 - p.ε))

/-- Calculates the total horizontal distance covered by a bouncing projectile --/
noncomputable def totalDistance (p : BouncingProjectile) : ℝ :=
  (p.v₀^2 * Real.sin (2 * p.α)) / (p.g * (1 - p.ε))

/-- Theorem stating the correctness of the totalTime and totalDistance functions --/
theorem bouncing_projectile_motion 
  (p : BouncingProjectile) 
  (h1 : p.v₀ > 0) 
  (h2 : p.g > 0) 
  (h3 : 0 < p.ε ∧ p.ε < 1) 
  (h4 : 0 ≤ p.α ∧ p.α ≤ Real.pi / 2) :
  totalTime p = (2 * p.v₀ * Real.sin p.α) / (p.g * (1 - p.ε)) ∧
  totalDistance p = (p.v₀^2 * Real.sin (2 * p.α)) / (p.g * (1 - p.ε)) := by
  sorry

#check bouncing_projectile_motion

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bouncing_projectile_motion_l702_70299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_equation_l702_70246

/-- A line passing through the point (-1,3) and perpendicular to x-2y+3=0 has the equation 2x+y-1=0 -/
theorem perpendicular_line_equation :
  let P : ℝ × ℝ := (-1, 3)
  let perpendicular_line := {(x, y) : ℝ × ℝ | x - 2*y + 3 = 0}
  let target_line := {(x, y) : ℝ × ℝ | 2*x + y - 1 = 0}
  (∀ (x y : ℝ), (x, y) ∈ target_line ↔ 
    ((x, y) ≠ P ∧ 
     (y - P.2) = -(x - P.1) * (1 / 2) ∧
     (x, y) ∉ perpendicular_line)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_equation_l702_70246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_vertex_C_l702_70260

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def dist (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Checks if AD is a median of triangle ABC -/
def is_median (A D B C : Point) : Prop :=
  dist D B = dist D C ∧ D.x = (B.x + C.x) / 2 ∧ D.y = (B.y + C.y) / 2

/-- Angle between AB and AD -/
noncomputable def angle_with_base (A B D : Point) : ℝ :=
  Real.arccos ((B.x - A.x) * (D.x - A.x) + (B.y - A.y) * (D.y - A.y)) / (dist A B * dist A D)

/-- Represents a circle with a center and radius -/
structure Circle where
  center : Point
  radius : ℝ

/-- Set of points on the circle -/
def Circle.points (circle : Circle) : Set Point :=
  {p : Point | dist p circle.center = circle.radius}

/-- Given a triangle ABC with base AB of length 6 and median AD of length 4 at 60 degrees to AB,
    the locus of vertex C is a circle with center A and radius 7 -/
theorem locus_of_vertex_C (A B C D : Point) (h1 : dist A B = 6)
    (h2 : is_median A D B C) (h3 : dist A D = 4) (h4 : angle_with_base A B D = Real.pi / 3) :
  ∃ (circle : Circle), circle.center = A ∧ circle.radius = 7 ∧ C ∈ circle.points :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_vertex_C_l702_70260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_fraction_equality_l702_70284

theorem power_fraction_equality : (2^2 + 2^1 + 2^0 : ℚ) / ((1/2 : ℚ) + (1/4 : ℚ) + (1/8 : ℚ)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_fraction_equality_l702_70284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_hits_per_player_next_six_games_l702_70264

/-- Represents a baseball team with its hitting statistics -/
structure BaseballTeam where
  totalPlayers : ℕ
  gamesPlayed : ℕ
  averageHitsPerGame : ℚ
  bestPlayerTotalHits : ℕ

/-- Calculates the average hits per game for each player excluding the best player -/
noncomputable def averageHitsPerPlayerExcludingBest (team : BaseballTeam) : ℚ :=
  (team.averageHitsPerGame - (team.bestPlayerTotalHits / team.gamesPlayed)) / (team.totalPlayers - 1)

/-- Theorem stating the average hits per player over the next 6 games -/
theorem average_hits_per_player_next_six_games 
  (team : BaseballTeam) 
  (h1 : team.totalPlayers = 11)
  (h2 : team.gamesPlayed = 5)
  (h3 : team.averageHitsPerGame = 15)
  (h4 : team.bestPlayerTotalHits = 25) :
  averageHitsPerPlayerExcludingBest team * 6 = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_hits_per_player_next_six_games_l702_70264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_selling_rate_l702_70212

/-- Proves that given an 8% loss on the initial selling rate and a 45% gain at 11.420689655172414 oranges per rupee, the initial selling rate was approximately 18 oranges per rupee. -/
theorem orange_selling_rate (initial_rate : ℝ) : 
  (initial_rate * 0.92 = 1) →  -- 8% loss on initial rate
  (11.420689655172414 * 1.45 = 1) →  -- 45% gain at new rate
  ∃ ε > 0, |initial_rate - 1 / 18| < ε :=
by
  intro h1 h2
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_selling_rate_l702_70212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_green_chips_count_l702_70208

theorem green_chips_count (total : ℕ) (blue_ratio red_ratio yellow_ratio : ℚ) 
  (h_total : total = 120)
  (h_blue : blue_ratio = 1 / 4)
  (h_red : red_ratio = 1 / 5)
  (h_yellow : yellow_ratio = 1 / 10)
  (h_sum : blue_ratio + red_ratio + yellow_ratio < 1) :
  total - (blue_ratio * ↑total).floor - (red_ratio * ↑total).floor - (yellow_ratio * ↑total).floor = 54 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_green_chips_count_l702_70208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ivan_can_win_l702_70265

/-- Represents the different types of water available -/
inductive Water
  | Plain
  | Magical (n : Nat)

/-- Represents the outcome of drinking water -/
inductive Outcome
  | Survive
  | Die

/-- The rules of the magical water -/
def drinkWater : Option Water → Water → Outcome
  | none, Water.Plain => Outcome.Survive
  | none, Water.Magical _ => Outcome.Die
  | some Water.Plain, _ => Outcome.Die
  | some (Water.Magical m), Water.Plain => Outcome.Survive
  | some (Water.Magical m), Water.Magical n => if n > m then Outcome.Survive else Outcome.Die

/-- Represents a player in the duel -/
structure Player where
  name : String
  canAccessTenth : Bool

/-- The duel between Ivan and Koschei -/
def duel (ivan : Player) (koschei : Player) (ivanPre : Option Water) (ivanGive : Water) (koscheiGive : Water) : Prop :=
  ivan.name = "Ivan" ∧
  koschei.name = "Koschei" ∧
  ivan.canAccessTenth = false ∧
  koschei.canAccessTenth = true ∧
  (∃ (n : Nat), n ≤ 9 ∧ ivanPre = some (Water.Magical n)) ∧
  koscheiGive = Water.Magical 10 ∧
  ivanGive = Water.Plain ∧
  drinkWater ivanPre koscheiGive = Outcome.Survive ∧
  drinkWater none ivanGive = Outcome.Survive ∧
  drinkWater (some ivanGive) (Water.Magical 10) = Outcome.Die

theorem ivan_can_win :
  ∃ (ivan koschei : Player) (ivanPre : Option Water) (ivanGive koscheiGive : Water),
    duel ivan koschei ivanPre ivanGive koscheiGive :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ivan_can_win_l702_70265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_shipping_cost_is_8_33_l702_70200

/-- Represents the manufacturing and sales scenario for an electronic component --/
structure ComponentManufacturing where
  costPerComponent : ℚ
  fixedMonthlyCost : ℚ
  monthlyProduction : ℚ
  sellingPrice : ℚ

/-- Calculates the maximum shipping cost per unit --/
def maxShippingCost (m : ComponentManufacturing) : ℚ :=
  (m.sellingPrice * m.monthlyProduction - m.costPerComponent * m.monthlyProduction - m.fixedMonthlyCost) / m.monthlyProduction

/-- Theorem stating that the maximum shipping cost per unit is $8.33 --/
theorem max_shipping_cost_is_8_33 (m : ComponentManufacturing) 
  (h1 : m.costPerComponent = 80)
  (h2 : m.fixedMonthlyCost = 16500)
  (h3 : m.monthlyProduction = 150)
  (h4 : m.sellingPrice = 198.33) :
  maxShippingCost m = 8.33 := by
  sorry

/-- Computes the result for the given scenario --/
def computeResult : ℚ :=
  maxShippingCost { costPerComponent := 80, fixedMonthlyCost := 16500, monthlyProduction := 150, sellingPrice := 198.33 }

#eval computeResult

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_shipping_cost_is_8_33_l702_70200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_quadratic_function_l702_70216

theorem min_value_quadratic_function (a b : ℝ) (h1 : a > b) (h2 : b > 0)
  (h3 : ∃! x, a * x^2 + 2 * x + b = 0) :
  (∀ x : ℝ, (a^2 + b^2) / (a - b) ≥ 2 * Real.sqrt 2) ∧
  (∃ x : ℝ, (a^2 + b^2) / (a - b) = 2 * Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_quadratic_function_l702_70216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_more_photos_this_year_l702_70235

theorem more_photos_this_year (total_photos : ℕ) 
  (ratio_last_year : ℕ) (ratio_this_year : ℕ) 
  (h1 : total_photos = 2430)
  (h2 : ratio_last_year = 10)
  (h3 : ratio_this_year = 17) : 
  (ratio_this_year - ratio_last_year) * total_photos / (ratio_last_year + ratio_this_year) = 630 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_more_photos_this_year_l702_70235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_transformed_functions_l702_70230

-- Define the functions H and J
def H : ℝ → ℝ := sorry
def J : ℝ → ℝ := sorry

-- Define the given intersection points
axiom intersection1 : H 1 = J 1 ∧ H 1 = 1
axiom intersection2 : H 3 = J 3 ∧ H 3 = 5
axiom intersection3 : H 5 = J 5 ∧ H 5 = 10
axiom intersection4 : H 7 = J 7 ∧ H 7 = 10

-- Theorem to prove
theorem intersection_of_transformed_functions :
  ∃ (x : ℝ), H (3 * x) = 3 * J x ∧ x = 3 ∧ H (3 * x) = 15 := by
  -- The proof goes here
  sorry

#check intersection_of_transformed_functions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_transformed_functions_l702_70230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_base_ratio_l702_70277

/-- An isosceles trapezoid with the given properties -/
structure IsoscelesTrapezoid where
  /-- Length of the smaller base -/
  a : ℝ
  /-- Length of the altitude -/
  h : ℝ
  /-- The larger base is equal to the sum of the smaller base and the altitude -/
  larger_base_eq : a + h > 0
  /-- Each diagonal is equal to the smaller base plus half the altitude -/
  diagonal_eq : a + h / 2 > 0

/-- The ratio of the smaller base to the larger base in the given isosceles trapezoid -/
noncomputable def base_ratio (t : IsoscelesTrapezoid) : ℝ :=
  t.a / (t.a + t.h)

/-- Theorem stating that the ratio of the smaller base to the larger base 
    in the given isosceles trapezoid is (2 - √5) / 2 -/
theorem isosceles_trapezoid_base_ratio (t : IsoscelesTrapezoid) :
  base_ratio t = (2 - Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_base_ratio_l702_70277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_size_bound_l702_70209

def is_valid_subset {n k : ℕ} (P : Set (Fin n → Fin k)) : Prop :=
  ∀ p ∈ P, ∀ i, p i < k

def have_equal_element {n k : ℕ} (p q : Fin n → Fin k) : Prop :=
  ∃ i, p i = q i

theorem subset_size_bound
  {n k : ℕ} (P Q : Set (Fin n → Fin k))
  (h_valid_P : is_valid_subset P)
  (h_valid_Q : is_valid_subset Q)
  (h_equal : ∀ p ∈ P, ∀ q ∈ Q, have_equal_element p q) :
  (Finset.card (P.toFinite.toFinset) ≤ k^(n-1)) ∨ (Finset.card (Q.toFinite.toFinset) ≤ k^(n-1)) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_size_bound_l702_70209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_lines_through_all_points_l702_70276

/-- Represents a point in the grid -/
structure GridPoint where
  x : Fin 10
  y : Fin 10

/-- Represents a line in the grid -/
structure GridLine where
  slope : ℚ
  intercept : ℚ

/-- The set of all marked points in the 10x10 grid -/
def markedPoints : Set GridPoint :=
  { p | true }

/-- Predicate to check if a line is not parallel to the sides of the square -/
def isNotParallelToSides (l : GridLine) : Prop :=
  l.slope ≠ 0 ∧ l.slope.den ≠ 0

/-- Predicate to check if a line passes through a point -/
def passesThrough (l : GridLine) (p : GridPoint) : Prop :=
  (p.y : ℚ) = l.slope * (p.x : ℚ) + l.intercept

/-- The theorem stating the minimum number of lines needed -/
theorem min_lines_through_all_points :
  ∃ (lines : Finset GridLine),
    (∀ l ∈ lines, isNotParallelToSides l) ∧
    (∀ p ∈ markedPoints, ∃ l ∈ lines, passesThrough l p) ∧
    (∀ lines' : Finset GridLine,
      (∀ l ∈ lines', isNotParallelToSides l) →
      (∀ p ∈ markedPoints, ∃ l ∈ lines', passesThrough l p) →
      lines'.card ≥ 18) ∧
    lines.card = 18 :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_lines_through_all_points_l702_70276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_direction_vector_proof_l702_70254

noncomputable def line (x : ℝ) : ℝ := (2 * x + 3) / 5

noncomputable def parameterization (d : ℝ × ℝ) (t : ℝ) : ℝ × ℝ := (2, -1) + t • d

noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem direction_vector_proof : 
  ∃ d : ℝ × ℝ, ∀ x ≥ 2, line x = (parameterization d (distance (x, line x) (2, -1))).2 ∧ 
  d = (5 / Real.sqrt 29, 2 / Real.sqrt 29) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_direction_vector_proof_l702_70254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_exponential_function_l702_70211

noncomputable def f (x : ℝ) : ℝ := Real.exp (x * Real.log 3)

theorem range_of_exponential_function :
  Set.range f = Set.Ioi 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_exponential_function_l702_70211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_sum_problem_l702_70296

open Real

theorem tangent_sum_problem (α : ℝ) 
  (h1 : tan (α + π/4) = -1/2)
  (h2 : π/2 < α)
  (h3 : α < π) :
  (sin (2*α) - 2*(cos α)^2) / sin (α - π/4) = -2*sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_sum_problem_l702_70296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_strawberry_pack_size_is_five_l702_70227

/-- The number of pieces in a pack of strawberry gum -/
def strawberry_pack_size : ℕ := 5

/-- The number of pieces in a pack of blueberry gum -/
def blueberry_pack_size : ℕ := 5

/-- The total number of pieces Linda needs to buy of each flavor -/
def total_pieces : ℕ := 30

/-- The number of packs of blueberry gum Linda needs to buy -/
def blueberry_packs : ℕ := total_pieces / blueberry_pack_size

theorem strawberry_pack_size_is_five :
  (strawberry_pack_size = 5) ∧
  (total_pieces / strawberry_pack_size = blueberry_packs) :=
by
  constructor
  · rfl
  · rfl

#eval strawberry_pack_size
#eval blueberry_packs

end NUMINAMATH_CALUDE_ERRORFEEDBACK_strawberry_pack_size_is_five_l702_70227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_inequality_equality_condition_l702_70217

structure Point where
  x : ℝ
  y : ℝ

noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

def isParallelogram (A B C D : Point) : Prop :=
  distance A B = distance C D ∧ distance B C = distance D A

theorem quadrilateral_inequality (A B C D : Point) :
  (distance A B)^2 + (distance B C)^2 + (distance C D)^2 + (distance D A)^2 ≥
  (distance A C)^2 + (distance B D)^2 := by
  sorry

theorem equality_condition (A B C D : Point) :
  (distance A B)^2 + (distance B C)^2 + (distance C D)^2 + (distance D A)^2 =
  (distance A C)^2 + (distance B D)^2 ↔ isParallelogram A B C D := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_inequality_equality_condition_l702_70217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l702_70253

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := (a * x + b) / (1 + x^2)

theorem function_properties (a b : ℝ) :
  (∀ x, x ∈ Set.Ioo (-1 : ℝ) 1 → f a b x = -f a b (-x)) →
  f a b (1/2) = 2/5 →
  (∃ g : ℝ → ℝ, 
    (∀ x, x ∈ Set.Ioo (-1 : ℝ) 1 → g x = x / (1 + x^2)) ∧
    (∀ x, x ∈ Set.Ioo (-1 : ℝ) 1 → f a b x = g x) ∧
    (∀ x y, x ∈ Set.Ioo (-1 : ℝ) 1 → y ∈ Set.Ioo (-1 : ℝ) 1 → x < y → g x < g y) ∧
    Set.Ioo (0 : ℝ) (1/2) = {t | f a b (t-1) + f a b t < 0}) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l702_70253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_three_l702_70266

/-- The Qin Jiushao formula for calculating the area of a triangle -/
noncomputable def qinJiushaoFormula (a b c : ℝ) : ℝ :=
  Real.sqrt (1/4 * (a^2 * b^2 - ((a^2 + b^2 - c^2)/2)^2))

/-- Theorem stating that the area of a triangle with sides 2, 3, and √13 is 3 -/
theorem triangle_area_is_three :
  qinJiushaoFormula 2 3 (Real.sqrt 13) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_three_l702_70266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l702_70228

noncomputable def f (x : ℝ) : ℝ := (x^5 - 5*x^4 + 10*x^3 - 10*x^2 + 5*x - 1) / (x^3 - 8)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x < 2 ∨ x > 2} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l702_70228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_of_2_7_l702_70287

-- Define the floor function as noncomputable
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

-- State the theorem
theorem floor_of_2_7 : floor 2.7 = 2 := by
  -- The proof is skipped for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_of_2_7_l702_70287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubs_fifth_inning_home_runs_l702_70232

/-- Represents the number of home runs scored by a team in a specific inning. -/
abbrev HomeRuns := Nat

/-- Represents the total number of home runs scored by a team in the game. -/
abbrev TotalHomeRuns := Nat

/-- The Chicago Cubs baseball team -/
structure ChicagoCubs : Type := mk ::

/-- The Cardinals baseball team -/
structure Cardinals : Type := mk ::

/-- The number of home runs scored by the Chicago Cubs in the third inning -/
def cubs_third_inning : HomeRuns := 2

/-- The number of home runs scored by the Chicago Cubs in the eighth inning -/
def cubs_eighth_inning : HomeRuns := 2

/-- The number of home runs scored by the Cardinals in the second inning -/
def cardinals_second_inning : HomeRuns := 1

/-- The number of home runs scored by the Cardinals in the fifth inning -/
def cardinals_fifth_inning : HomeRuns := 1

/-- The difference in total home runs between the Chicago Cubs and the Cardinals -/
def home_run_difference : Nat := 3

/-- The function to calculate the total home runs for the Chicago Cubs -/
def cubs_total_home_runs (fifth_inning : HomeRuns) : TotalHomeRuns :=
  cubs_third_inning + fifth_inning + cubs_eighth_inning

/-- The function to calculate the total home runs for the Cardinals -/
def cardinals_total_home_runs : TotalHomeRuns :=
  cardinals_second_inning + cardinals_fifth_inning

/-- Theorem stating that the Chicago Cubs scored 1 home run in the fifth inning -/
theorem cubs_fifth_inning_home_runs :
    ∃ (fifth_inning : HomeRuns),
      cubs_total_home_runs fifth_inning = cardinals_total_home_runs + home_run_difference ∧
      fifth_inning = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubs_fifth_inning_home_runs_l702_70232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_non_permuting_bijection_l702_70283

-- Define the set of natural numbers (positive integers)
def PositiveNat : Set ℕ := {n : ℕ | n > 0}

-- Define a bijection from ℕ to ℕ
noncomputable def f : ℕ → ℕ := sorry

-- Define what it means for a sequence to be a permutation of (1, 2, ..., n)
def is_permutation (seq : List ℕ) (n : ℕ) : Prop :=
  seq.length = n ∧ (∀ k : ℕ, k ≤ n → k ∈ seq)

-- State the theorem
theorem exists_non_permuting_bijection :
  ∃ (f : ℕ → ℕ), Function.Bijective f ∧
    ∀ n : ℕ, n > 0 → ¬(is_permutation (List.map f (List.range n)) n) :=
by
  sorry

#check exists_non_permuting_bijection

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_non_permuting_bijection_l702_70283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_is_perfect_square_y_relation_l702_70269

def x : ℕ → ℤ
  | 0 => 1  -- Added case for 0
  | 1 => 1
  | 2 => 1
  | (n + 3) => 14 * x (n + 2) - x (n + 1) - 4

def y : ℕ → ℤ
  | 0 => 1  -- Added case for 0
  | 1 => 1
  | 2 => 1
  | (n + 3) => 4 * y (n + 2) - y (n + 1)

theorem x_is_perfect_square :
  ∀ n : ℕ, x n = (y n) ^ 2 := by
  sorry

-- Additional helper theorem
theorem y_relation :
  ∀ n : ℕ, n ≥ 1 → 4 * (y (n + 1)) * (y n) = x (n + 1) + x n + 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_is_perfect_square_y_relation_l702_70269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_not_perfect_square_a_not_square_l702_70206

def a : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => a n * a (n + 1) + 1

theorem a_not_perfect_square (n : ℕ) (h : n ≥ 2) :
  ∃ k : Fin 2, a n % 4 = k.val + 2 :=
by
  sorry

theorem a_not_square (n : ℕ) (h : n ≥ 2) :
  ¬∃ m : ℕ, a n = m ^ 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_not_perfect_square_a_not_square_l702_70206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_identity_proof_l702_70282

theorem identity_proof (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  let x := b / c + c / b
  let y := a / c + c / a
  let z := a / b + b / a
  let w := a / d + d / a
  x^2 + y^2 + z^2 + w^2 - x*y*z*w = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_identity_proof_l702_70282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_for_cake_l702_70207

/-- The amount of milk (in mL) needed for a given amount of flour (in mL) -/
noncomputable def milk_needed (flour : ℝ) : ℝ :=
  (flour / 300) * 75

theorem milk_for_cake (flour : ℝ) (h : flour = 900) :
  milk_needed flour = 225 := by
  rw [milk_needed, h]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_for_cake_l702_70207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_code_exists_l702_70252

def Letter : Type := Char

def Digit : Type := Fin 10

theorem shopkeeper_code_exists : 
  ∃ (f : Letter → Digit), Function.Bijective f ∧ 
    (∀ (x : Letter), x ∈ ['П', 'О', 'Д', 'С', 'В', 'Е', 'Ч', 'Н', 'И', 'К']) ∧
    (((f 'Д').val * 10000 + (f 'Е').val * 1000 + (f 'С').val * 100 + (f 'К').val * 10 + (f 'Ч').val) +
    ((f 'И').val * 10000 + (f 'Н').val * 1000 + (f 'В').val * 100 + (f 'О').val * 10 + (f 'П').val) =
    ((f 'П').val * 100000 + (f 'Д').val * 10000 + (f 'С').val * 1000 + (f 'И').val * 100 + (f 'О').val * 10 + (f 'Н').val)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_code_exists_l702_70252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_is_empty_l702_70226

/-- Given n ≥ 3 is a positive odd number, prove that the intersection of sets A and B is empty -/
theorem intersection_is_empty (n : ℕ) (h_n : n ≥ 3) (h_odd : Odd n) : 
  let z : ℂ := Complex.exp (2 * π * Complex.I / n)
  let A : Set ℂ := {w | ∃ k : ℕ, 1 ≤ k ∧ k < n ∧ w = z ^ k}
  let B : Set ℂ := {w | ∃ k : ℕ, 1 ≤ k ∧ k < n ∧ w = (1 - z^(k+1)) / (1 - z)}
  A ∩ B = ∅ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_is_empty_l702_70226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shifted_function_even_at_5pi_12_l702_70238

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x + Real.pi / 3)

theorem shifted_function_even_at_5pi_12 :
  ∃ (φ : ℝ), 0 < φ ∧ φ < Real.pi / 2 ∧
  (∀ (x : ℝ), f (x - φ) = f (φ - x)) ∧
  φ = 5 * Real.pi / 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shifted_function_even_at_5pi_12_l702_70238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_parabola_focus_coincide_l702_70274

noncomputable section

-- Define the hyperbola
def hyperbola (a : ℝ) (x y : ℝ) : Prop := x^2 / a^2 - y^2 / 4 = 1

-- Define the eccentricity of the hyperbola
def eccentricity (a : ℝ) : ℝ := (3 * Real.sqrt 5) / 5

-- Define the left focus of the hyperbola
def left_focus (a : ℝ) : ℝ × ℝ := (-Real.sqrt (a^2 + 4), 0)

-- Define the parabola
def parabola (m : ℝ) (x y : ℝ) : Prop := y^2 = m * x

-- Define the focus of the parabola
def parabola_focus (m : ℝ) : ℝ × ℝ := (m / 4, 0)

theorem hyperbola_parabola_focus_coincide (a m : ℝ) :
  a > 0 →
  hyperbola a (Real.sqrt a^2) 0 →
  eccentricity a = (3 * Real.sqrt 5) / 5 →
  left_focus a = parabola_focus m →
  m = -12 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_parabola_focus_coincide_l702_70274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_cubic_term_implies_k_eq_9_l702_70267

/-- The first polynomial -/
noncomputable def poly1 (k a b : ℝ) : ℝ := -2*a*b + (1/3)*k*a^2*b + 5*b^2

/-- The second polynomial -/
noncomputable def poly2 (a b : ℝ) : ℝ := b^2 + 3*a^2*b - 5*a*b + 1

/-- The difference between the two polynomials -/
noncomputable def poly_diff (k a b : ℝ) : ℝ := poly1 k a b - poly2 a b

/-- Theorem stating that if the difference between the polynomials does not contain a cubic term, then k = 9 -/
theorem no_cubic_term_implies_k_eq_9 :
  (∀ a b : ℝ, ∃ c d e f : ℝ, poly_diff k a b = c*a^2*b + d*a*b + e*b^2 + f) → k = 9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_cubic_term_implies_k_eq_9_l702_70267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equivalent_pipes_l702_70280

/-- Represents a circular pipe with a given diameter and height -/
structure Pipe where
  diameter : ℝ
  height : ℝ

/-- Calculates the volume of water a pipe can carry -/
noncomputable def volume (p : Pipe) : ℝ :=
  Real.pi * (p.diameter / 2) ^ 2 * p.height

theorem equivalent_pipes : 
  let small_pipe : Pipe := { diameter := 2, height := 3 }
  let large_pipe : Pipe := { diameter := 4, height := 6 }
  ∃ n : ℕ, n * volume small_pipe = volume large_pipe ∧ n = 8 := by
  sorry

#eval "Theorem stated successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equivalent_pipes_l702_70280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thiosulfate_structure_and_formation_l702_70241

/-- Represents an atom -/
structure Atom :=
  (id : Nat)

/-- Represents a bond between two atoms -/
inductive Bond : Type
  | single : Atom → Atom → Bond
  | double : Atom → Atom → Bond
  | triple : Atom → Atom → Bond

/-- Represents the thiosulfate anion -/
structure Thiosulfate :=
  (s1 : Atom)
  (s2 : Atom)
  (o1 o2 o3 : Atom)
  (s1_s2_bond : Bond)
  (s1_o1_bond : Bond)
  (s1_o2_bond : Bond)
  (s1_o3_bond : Bond)

/-- Represents the sulfite anion -/
structure Sulfite :=
  (s : Atom)
  (o1 o2 o3 : Atom)
  (s_o1_bond : Bond)
  (s_o2_bond : Bond)
  (s_o3_bond : Bond)

/-- Represents a molecule -/
structure Molecule :=
  (atoms : List Atom)
  (bonds : List Bond)

/-- Represents an aqueous solution -/
structure AqueousSolution :=
  (water : Molecule)
  (solute : Molecule)

/-- Represents the reaction of sulfite with sulfur in an aqueous solution -/
noncomputable def sulfite_sulfur_reaction (sulfite : Sulfite) (sulfur : Atom) (solution : AqueousSolution) : Thiosulfate :=
  sorry

theorem thiosulfate_structure_and_formation :
  ∀ (thiosulfate : Thiosulfate) (sulfite : Sulfite) (sulfur : Atom) (solution : AqueousSolution),
    (∃ (s1 s2 : Atom), thiosulfate.s1_s2_bond = Bond.single s1 s2) ∧
    (sulfite_sulfur_reaction sulfite sulfur solution = thiosulfate) := by
  sorry

#check thiosulfate_structure_and_formation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_thiosulfate_structure_and_formation_l702_70241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_correct_l702_70297

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^2 - 2 * a * x + 3

-- Define the minimum value function
noncomputable def min_value (a : ℝ) : ℝ :=
  if a < -2 then 5 + 2 * a
  else if a > 2 then 5 - 2 * a
  else 3 - a^2 / 2

-- State the theorem
theorem min_value_correct (a : ℝ) :
  ∀ x ∈ Set.Icc (-1) 1, f a x ≥ min_value a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_correct_l702_70297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_highway_intersection_probability_l702_70281

/-- The probability of passing through both intersections without stopping -/
noncomputable def P (x : ℝ) : ℝ := (x / (x + 30)) * (120 / (x + 120))

/-- The function to be minimized to maximize P(x) -/
noncomputable def f (x : ℝ) : ℝ := x + 3600 / x + 150

theorem highway_intersection_probability :
  (∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → P y ≤ P x) ∧
  (∃ (x : ℝ), x > 0 ∧ P x = 4 / 9) ∧
  (∀ (x : ℝ), x > 0 → P x ≤ 4 / 9) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_highway_intersection_probability_l702_70281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_middle_three_average_l702_70201

def max_diff (a b c d e : ℕ+) : ℕ := 
  max a (max b (max c (max d e))) - min a (min b (min c (min d e)))

theorem middle_three_average (a b c d e : ℕ+) : 
  a < b ∧ b < c ∧ c < d ∧ d < e →  -- five different positive integers
  (a + b + c + d + e : ℚ) / 5 = 5 →  -- average is 5
  e - a = max_diff a b c d e →  -- maximum difference
  (b + c + d : ℚ) / 3 = 3 :=  -- average of middle three is 3
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_middle_three_average_l702_70201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_result_l702_70202

-- Define the rotation matrix for π/2 clockwise rotation
def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  Matrix.of !![0, 1; -1, 0]

-- Define the initial point A
def point_A : Fin 2 → ℝ := ![2, 1]

-- Define the rotated point B
def point_B : Fin 2 → ℝ := Matrix.mulVec rotation_matrix point_A

-- Theorem statement
theorem rotation_result : point_B = ![1, -2] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_result_l702_70202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wrapping_paper_area_theorem_l702_70279

/-- The area of wrapping paper needed for a box -/
noncomputable def wrapping_paper_area (s h : ℝ) : ℝ :=
  3 * s^2 + h^2 + 2 * s * Real.sqrt (2 * (s^2 + h^2))

/-- The theorem stating the area of wrapping paper needed for a box -/
theorem wrapping_paper_area_theorem (s h : ℝ) (hs : s > 0) (hh : h > 0) :
  let box_base := s^2
  let box_height := h
  let diagonal_face := Real.sqrt (s^2 + h^2)
  let box_diagonal := s * Real.sqrt 2
  let paper_side := diagonal_face + box_diagonal
  paper_side^2 = wrapping_paper_area s h :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wrapping_paper_area_theorem_l702_70279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hybrid_car_trip_distance_l702_70294

/-- Represents a hybrid car with specified characteristics -/
structure HybridCar where
  battery_range : ℝ
  gasoline_consumption : ℝ
  average_efficiency : ℝ

/-- Calculates the total trip distance for a given hybrid car -/
noncomputable def trip_distance (car : HybridCar) : ℝ :=
  let total_distance := car.battery_range + (car.average_efficiency * car.gasoline_consumption * (car.average_efficiency - car.battery_range))
  total_distance / (1 + car.gasoline_consumption * car.average_efficiency)

/-- Theorem stating that a hybrid car with given specifications will travel 180 miles -/
theorem hybrid_car_trip_distance :
  let car : HybridCar := {
    battery_range := 60,
    gasoline_consumption := 0.03,
    average_efficiency := 50
  }
  trip_distance car = 180 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hybrid_car_trip_distance_l702_70294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_on_parabola_l702_70213

theorem points_on_parabola (t : ℝ) : 
  let x := (3 : ℝ)^t - 4
  let y := (3 : ℝ)^(2*t) - 7 * (3 : ℝ)^t + 6
  y = x^2 + 2*x - 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_on_parabola_l702_70213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l702_70204

theorem calculation_proof : (1/2)⁻¹ - Real.sqrt 3 * Real.tan (30 * Real.pi / 180) + (Real.pi - 2023)^0 + abs (-2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l702_70204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_PF₁Q_l702_70258

noncomputable section

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 3 - y^2 = 1

-- Define the foci
def F₁ : ℝ × ℝ := (-2, 0)
def F₂ : ℝ × ℝ := (2, 0)

-- Define point P
def P : ℝ × ℝ := (2, Real.sqrt (1/3))

-- Define point Q (y-coordinate is negative of P's y-coordinate due to symmetry)
def Q : ℝ × ℝ := (2, -Real.sqrt (1/3))

-- Define the perimeter of triangle PF₁Q
def perimeter_PF₁Q : ℝ := 
  Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) +
  Real.sqrt ((Q.1 - F₁.1)^2 + (Q.2 - F₁.2)^2) +
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Theorem statement
theorem perimeter_of_PF₁Q : 
  hyperbola P.1 P.2 → hyperbola Q.1 Q.2 → perimeter_PF₁Q = 16 * Real.sqrt 3 / 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_PF₁Q_l702_70258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_subset_size_l702_70289

theorem smallest_subset_size (n : ℕ+) : 
  (∃ k : ℕ+, 
    (∀ S : Finset (Fin (2 * n)),
      S.card = k → 
      ∃ a b c d : Fin (2 * n), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
      a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧
      (a : ℕ) + (b : ℕ) + (c : ℕ) + (d : ℕ) = 4 * n + 1) ∧
    (∀ m : ℕ+, m < k → 
      ∃ T : Finset (Fin (2 * n)),
        T.card = m ∧
        ∀ a b c d : Fin (2 * n), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
        a ∈ T → b ∈ T → c ∈ T → d ∈ T →
        (a : ℕ) + (b : ℕ) + (c : ℕ) + (d : ℕ) ≠ 4 * n + 1)) ∧
  (∀ k : ℕ+, 
    (∀ S : Finset (Fin (2 * n)),
      S.card = k → 
      ∃ a b c d : Fin (2 * n), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
      a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧
      (a : ℕ) + (b : ℕ) + (c : ℕ) + (d : ℕ) = 4 * n + 1) ∧
    (∀ m : ℕ+, m < k → 
      ∃ T : Finset (Fin (2 * n)),
        T.card = m ∧
        ∀ a b c d : Fin (2 * n), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
        a ∈ T → b ∈ T → c ∈ T → d ∈ T →
        (a : ℕ) + (b : ℕ) + (c : ℕ) + (d : ℕ) ≠ 4 * n + 1) →
    k = n + 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_subset_size_l702_70289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sunzi_carriage_problem_l702_70236

/-- Represents the number of carriages -/
def x : ℕ := sorry

/-- Three people share a carriage, leaving two carriages empty -/
def scenario1 : ℕ := 3 * (x - 2)

/-- Two people share a carriage, leaving nine people walking -/
def scenario2 : ℕ := 2 * x + 9

/-- The equation correctly represents the problem -/
theorem sunzi_carriage_problem :
  scenario1 = scenario2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sunzi_carriage_problem_l702_70236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_derivative_parametric_function_l702_70295

noncomputable def x (t : ℝ) : ℝ := Real.sin t
noncomputable def y (t : ℝ) : ℝ := Real.log (Real.cos t)

theorem second_derivative_parametric_function (t : ℝ) :
  let x' := deriv x
  let y' := deriv y
  let y'_x := fun s => y' s / x' s
  let y''_xx := fun s => deriv y'_x s / x' s
  y''_xx t = -(1 + Real.sin t ^ 2) / Real.cos t ^ 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_derivative_parametric_function_l702_70295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_subtraction_l702_70223

theorem factorial_subtraction : Nat.factorial 7 - 6 * Nat.factorial 6 - Nat.factorial 6 = 0 := by
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_subtraction_l702_70223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_l702_70293

noncomputable def f (x : ℝ) : ℝ := 
  Real.log (Real.sin (1/2)) - (1/24) * (Real.cos (12*x))^2 / Real.sin (24*x)

theorem derivative_f (x : ℝ) : 
  deriv f x = 1 / (4 * (Real.sin (12*x))^2) := by
  sorry

#check derivative_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_l702_70293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_m_times_10_to_n_l702_70272

theorem expression_equals_m_times_10_to_n : ∃ (m n : ℕ), 
  (3^202 + 7^203)^2 - (3^202 - 7^203)^2 = m * 10^n ∧ m = 59 ∧ n = 202 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_m_times_10_to_n_l702_70272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_property_l702_70256

/-- Represents a continued fraction of the form [a̅;b] -/
def continuedFraction (a b : ℤ) : ℚ := sorry

/-- Given a quadratic equation ax² + bx + c = 0 with integer coefficients,
    if one root is [a̅;b], then the other root is -1/[b̅;a] -/
theorem quadratic_roots_property (a b c : ℤ) (root1 : ℚ) :
  (∃ x : ℚ, a * x^2 + b * x + c = 0) →
  (root1 = continuedFraction a b) →
  (∃ root2 : ℚ, a * root2^2 + b * root2 + c = 0 ∧ 
                root2 = -(1 / continuedFraction b a)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_property_l702_70256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_train_length_approx_l702_70288

/-- Calculates the length of a train given the speeds of two trains, the time they take to cross each other, and the length of the other train. -/
noncomputable def calculate_train_length (speed1 speed2 : ℝ) (crossing_time : ℝ) (other_train_length : ℝ) : ℝ :=
  let relative_speed := speed1 + speed2
  let relative_speed_mps := relative_speed * (5 / 18)
  let combined_length := relative_speed_mps * crossing_time
  combined_length - other_train_length

/-- Theorem stating that under the given conditions, the length of the first train is approximately 270.04 meters. -/
theorem first_train_length_approx : 
  ∃ ε > 0, |calculate_train_length 120 80 9 230 - 270.04| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_train_length_approx_l702_70288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_barbara_steak_purchase_l702_70259

/-- The price of steak per pound in dollars -/
noncomputable def steak_price : ℝ := 15

/-- The price of chicken breast per pound in dollars -/
noncomputable def chicken_price : ℝ := 8

/-- The amount of chicken breast bought in pounds -/
noncomputable def chicken_amount : ℝ := 1.5

/-- The total amount spent in dollars -/
noncomputable def total_spent : ℝ := 42

/-- The amount of steak bought in pounds -/
noncomputable def steak_amount : ℝ := (total_spent - chicken_price * chicken_amount) / steak_price

theorem barbara_steak_purchase : steak_amount = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_barbara_steak_purchase_l702_70259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_sqrt_3_l702_70242

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The focus of the hyperbola -/
noncomputable def focus (h : Hyperbola) : Point :=
  { x := -Real.sqrt (h.a^2 + h.b^2), y := 0 }

/-- Vector between two points -/
def vector (p q : Point) : Point :=
  { x := q.x - p.x, y := q.y - p.y }

/-- Predicate to check if a vector is perpendicular to an asymptote -/
def IsPerpendicularToAsymptote (h : Hyperbola) (v : Point) : Prop := sorry

/-- Predicate to check if two points intersect the asymptotes -/
def IntersectsAsymptotes (h : Hyperbola) (p q : Point) : Prop := sorry

/-- Scalar multiplication of a vector -/
def scalarMul (c : ℝ) (v : Point) : Point :=
  { x := c * v.x, y := c * v.y }

/-- Theorem: If a perpendicular line from the focus to an asymptote of a hyperbola
    intersects the asymptotes at points P and Q such that FP = 3FQ,
    then the eccentricity of the hyperbola is √3 -/
theorem hyperbola_eccentricity_sqrt_3 (h : Hyperbola) (p q : Point)
  (h_perp : IsPerpendicularToAsymptote h (vector (focus h) p))
  (h_intersect : IntersectsAsymptotes h p q)
  (h_ratio : vector (focus h) p = scalarMul 3 (vector (focus h) q)) :
  eccentricity h = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_sqrt_3_l702_70242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_value_l702_70285

/-- The function f as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * (Real.sqrt 3 * Real.cos x - Real.sin x) + 1

/-- A function is even if f(x) = f(-x) for all x -/
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- The main theorem to be proved -/
theorem phi_value (φ : ℝ) : 
  (is_even (fun x ↦ f (x - φ))) → φ = π / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_value_l702_70285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l702_70244

theorem range_of_a (a : ℝ) : 
  (0 < a ∧ a ≠ 1 ∧ Real.log (2/3) / Real.log a < 1) ↔ 
  (0 < a ∧ a < 2/3) ∨ (1 < a) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l702_70244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_n_for_triangle_property_l702_70214

/-- A function that checks if three numbers can form a triangle with positive area -/
def can_form_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- A function that checks if all triplets in a list of 10 numbers can form triangles -/
def all_triplets_form_triangles (l : List ℕ) : Prop :=
  l.length = 10 ∧ 
  ∀ i j k, i < j → j < k → k < l.length → 
    can_form_triangle (l.get ⟨i, sorry⟩) (l.get ⟨j, sorry⟩) (l.get ⟨k, sorry⟩)

/-- The main theorem stating that 198 is the largest possible n -/
theorem largest_n_for_triangle_property : 
  ∀ n : ℕ, n > 198 → 
    ∃ l : List ℕ, (∀ x, x ∈ l → 3 ≤ x ∧ x ≤ n) ∧ 
                  (∀ x y, x ∈ l → y ∈ l → x ≠ y → x ≠ y) ∧
                  ¬(all_triplets_form_triangles l) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_n_for_triangle_property_l702_70214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_distribution_theorem_l702_70203

def distribute_balls (n : ℕ) : ℕ :=
  (List.range 7).foldl (λ acc k =>
    acc + Nat.choose 7 k * (2^(7-k) - 2)) 0

theorem ball_distribution_theorem :
  distribute_balls 7 = 1932 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_distribution_theorem_l702_70203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l702_70278

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x - 3) + Real.sqrt (12 - 3*x)

-- State the theorem about the range of f
theorem range_of_f :
  ∃ (a b : ℝ), a = 1 ∧ b = 2 ∧
  (∀ y, (∃ x, f x = y) ↔ a ≤ y ∧ y ≤ b) := by
  sorry

#check range_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l702_70278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_12_l702_70290

theorem tan_alpha_plus_pi_12 (α : Real) (h : Real.sin α = 3 * Real.sin (α + π/6)) :
  Real.tan (α + π/12) = 2 * Real.sqrt 3 - 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_12_l702_70290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_laurent_greater_chloe_l702_70239

/-- The probability that a uniformly random number from [0, 4000] is greater than
    a uniformly random number from [0, 3000] -/
theorem prob_laurent_greater_chloe : ℝ := by
  let chloe_range : ℝ := 3000
  let laurent_range : ℝ := 4000
  let total_area := chloe_range * laurent_range
  let favorable_area := total_area - (chloe_range * chloe_range / 2)
  have h : favorable_area / total_area = 5 / 8 := by sorry
  exact 5 / 8


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_laurent_greater_chloe_l702_70239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_m_value_l702_70243

-- Define the hyperbola equation
def hyperbola_equation (m : ℝ) (x y : ℝ) : Prop :=
  x^2 / m - y^2 / 8 = 1

-- Define eccentricity
noncomputable def eccentricity (m : ℝ) : ℝ :=
  Real.sqrt ((8 + m) / m)

-- State the theorem
theorem hyperbola_m_value :
  ∀ m : ℝ, m > 0 → (∀ x y : ℝ, hyperbola_equation m x y) → eccentricity m = Real.sqrt 5 → m = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_m_value_l702_70243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_lines_l702_70240

-- Define the line equation
def line_equation (k m x y : ℚ) : Prop :=
  y = k * x + m

-- Define the ellipse equation
def ellipse_equation (x y : ℚ) : Prop :=
  x^2 / 16 + y^2 / 12 = 1

-- Define the hyperbola equation
def hyperbola_equation (x y : ℚ) : Prop :=
  x^2 / 4 - y^2 / 12 = 1

-- Define the condition for intersection points
def intersection_condition (k m : ℚ) (x y : ℚ) : Prop :=
  line_equation k m x y ∧
  ((ellipse_equation x y) ∨ (hyperbola_equation x y))

-- Define the vector sum condition
def vector_sum_condition (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℚ) : Prop :=
  (x₃ - x₁) + (x₄ - x₂) = 0 ∧ (y₃ - y₁) + (y₄ - y₂) = 0

-- Main theorem
theorem count_valid_lines :
  ∃ (S : Finset (ℚ × ℚ)),
    (∀ (k m : ℚ), (k, m) ∈ S ↔
      ∃ (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℚ),
        intersection_condition k m x₁ y₁ ∧
        intersection_condition k m x₂ y₂ ∧
        intersection_condition k m x₃ y₃ ∧
        intersection_condition k m x₄ y₄ ∧
        ellipse_equation x₁ y₁ ∧
        ellipse_equation x₂ y₂ ∧
        hyperbola_equation x₃ y₃ ∧
        hyperbola_equation x₄ y₄ ∧
        vector_sum_condition x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄) ∧
    S.card = 9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_lines_l702_70240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_productivity_increase_factor_is_four_l702_70229

/-- Represents a worker in the manufacturing process -/
structure Worker where
  setupTime : ℚ
  workTime : ℚ
  partsProcessed : ℚ

/-- Represents the manufacturing scenario -/
structure ManufacturingScenario where
  worker1 : Worker
  worker2 : Worker
  totalParts : ℚ

/-- Calculates the productivity increase factor -/
noncomputable def productivityIncreaseFactor (scenario : ManufacturingScenario) : ℚ :=
  (scenario.worker2.partsProcessed / scenario.worker2.workTime) /
  (scenario.worker1.partsProcessed / scenario.worker1.workTime)

theorem productivity_increase_factor_is_four (scenario : ManufacturingScenario) :
  productivityIncreaseFactor scenario = 4 := by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_productivity_increase_factor_is_four_l702_70229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_balance_coins_in_n_moves_l702_70270

/-- Represents the state of a box, which can be either a coin or contain two smaller boxes -/
inductive Box
  | Coin : Bool → Box
  | Container : Box → Box → Box

/-- Represents the nested box structure with n levels -/
def NestedBoxes (n : ℕ) : Type := Box

/-- Counts the number of heads in a box structure -/
def countHeads : Box → ℕ
  | Box.Coin true => 1
  | Box.Coin false => 0
  | Box.Container b1 b2 => countHeads b1 + countHeads b2

/-- Flips a box and all its contents -/
def flipBox : Box → Box
  | Box.Coin b => Box.Coin (!b)
  | Box.Container b1 b2 => Box.Container (flipBox b1) (flipBox b2)

/-- Represents a sequence of moves, where each move is flipping a specific box -/
def Moves := List (Box → Box)

/-- Applies a sequence of moves to a box structure -/
def applyMoves (b : Box) (moves : Moves) : Box :=
  moves.foldl (fun acc m => m acc) b

/-- Counts the total number of coins in a box structure -/
def countCoins : Box → ℕ
  | Box.Coin _ => 1
  | Box.Container b1 b2 => countCoins b1 + countCoins b2

theorem balance_coins_in_n_moves (n : ℕ) (initial : NestedBoxes n) :
  ∃ (moves : Moves), moves.length ≤ n ∧
    let final := applyMoves initial moves
    2 * countHeads final = countCoins final := by
  sorry

#check balance_coins_in_n_moves

end NUMINAMATH_CALUDE_ERRORFEEDBACK_balance_coins_in_n_moves_l702_70270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combination_permutation_difference_l702_70262

theorem combination_permutation_difference (t : ℕ) : 
  (0 ≤ 11 - 2*t) ∧ (11 - 2*t ≤ 5*t) ∧ (0 ≤ 2*t - 2) ∧ (2*t - 2 ≤ 11 - 3*t) →
  Nat.choose (5*t) (11 - 2*t) - Nat.factorial (11 - 3*t) / Nat.factorial (11 - 3*t - (2*t - 2)) = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combination_permutation_difference_l702_70262
