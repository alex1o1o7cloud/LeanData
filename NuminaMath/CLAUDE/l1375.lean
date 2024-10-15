import Mathlib

namespace NUMINAMATH_CALUDE_tyler_meal_combinations_l1375_137586

/-- The number of types of meat available -/
def num_meats : ℕ := 4

/-- The number of types of vegetables available -/
def num_vegetables : ℕ := 5

/-- The number of types of desserts available -/
def num_desserts : ℕ := 5

/-- The number of types of drinks available -/
def num_drinks : ℕ := 4

/-- The number of vegetables Tyler must choose -/
def vegetables_to_choose : ℕ := 3

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The theorem stating the number of different meal combinations Tyler can choose -/
theorem tyler_meal_combinations : 
  num_meats * choose num_vegetables vegetables_to_choose * num_desserts * num_drinks = 800 := by
  sorry

end NUMINAMATH_CALUDE_tyler_meal_combinations_l1375_137586


namespace NUMINAMATH_CALUDE_max_a6_value_l1375_137527

theorem max_a6_value (a : ℕ → ℝ) 
  (h1 : ∀ n, a (n + 1) ≤ (a (n + 2) + a n) / 2)
  (h2 : a 1 = 1)
  (h3 : a 404 = 2016) :
  ∃ M, a 6 ≤ M ∧ M = 26 :=
by sorry

end NUMINAMATH_CALUDE_max_a6_value_l1375_137527


namespace NUMINAMATH_CALUDE_alices_number_l1375_137548

theorem alices_number (n : ℕ) : 
  (∃ k : ℕ, n = 180 * k) → 
  (∃ m : ℕ, n = 240 * m) → 
  2000 < n → 
  n < 5000 → 
  n = 2160 ∨ n = 2880 ∨ n = 3600 ∨ n = 4320 := by
sorry

end NUMINAMATH_CALUDE_alices_number_l1375_137548


namespace NUMINAMATH_CALUDE_alternative_interest_rate_l1375_137546

/-- Proves that given a principal of 4200 invested for 2 years, if the interest at 18% p.a. 
    is 504 more than the interest at an unknown rate p, then p is equal to 12. -/
theorem alternative_interest_rate (principal : ℝ) (time : ℝ) (known_rate : ℝ) (difference : ℝ) (p : ℝ) : 
  principal = 4200 →
  time = 2 →
  known_rate = 18 →
  difference = 504 →
  principal * known_rate / 100 * time - principal * p / 100 * time = difference →
  p = 12 := by
  sorry

#check alternative_interest_rate

end NUMINAMATH_CALUDE_alternative_interest_rate_l1375_137546


namespace NUMINAMATH_CALUDE_chord_length_is_four_l1375_137502

/-- Given a line and a circle in 2D space, prove that the length of the chord
    intercepted by the line and the circle is equal to 4. -/
theorem chord_length_is_four (x y : ℝ) : 
  (x + 2 * y - 2 = 0) →  -- Line equation
  ((x - 2)^2 + y^2 = 4) →  -- Circle equation
  ∃ (a b c d : ℝ), 
    (a + 2 * b - 2 = 0) ∧  -- Point (a, b) on the line
    ((a - 2)^2 + b^2 = 4) ∧  -- Point (a, b) on the circle
    (c + 2 * d - 2 = 0) ∧  -- Point (c, d) on the line
    ((c - 2)^2 + d^2 = 4) ∧  -- Point (c, d) on the circle
    (a ≠ c ∨ b ≠ d) ∧  -- (a, b) and (c, d) are distinct points
    ((a - c)^2 + (b - d)^2 = 4^2)  -- Distance between points is 4
  := by sorry

end NUMINAMATH_CALUDE_chord_length_is_four_l1375_137502


namespace NUMINAMATH_CALUDE_cubic_roots_problem_l1375_137560

/-- Given a cubic polynomial x³ - 2x² + 5x - 8 with roots p, q, r,
    and another cubic polynomial x³ + ux² + vx + w with roots p+q, q+r, r+p,
    prove that w = 34 -/
theorem cubic_roots_problem (p q r u v w : ℝ) : 
  (∀ x, x^3 - 2*x^2 + 5*x - 8 = 0 ↔ x = p ∨ x = q ∨ x = r) →
  (∀ x, x^3 + u*x^2 + v*x + w = 0 ↔ x = p+q ∨ x = q+r ∨ x = r+p) →
  w = 34 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_problem_l1375_137560


namespace NUMINAMATH_CALUDE_age_sum_proof_l1375_137549

theorem age_sum_proof (younger_age older_age : ℕ) : 
  older_age - younger_age = 2 →
  older_age = 38 →
  younger_age + older_age = 74 :=
by
  sorry

end NUMINAMATH_CALUDE_age_sum_proof_l1375_137549


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1375_137579

theorem negation_of_universal_proposition :
  ¬(∀ x : ℝ, x^2 ≥ Real.log 2) ↔ ∃ x : ℝ, x^2 < Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1375_137579


namespace NUMINAMATH_CALUDE_opposite_signs_and_greater_magnitude_l1375_137537

theorem opposite_signs_and_greater_magnitude (a b : ℝ) : 
  a * b < 0 → a + b < 0 → 
  ((a < 0 ∧ b > 0) ∨ (a > 0 ∧ b < 0)) ∧ 
  (max (abs a) (abs b) > min (abs a) (abs b)) :=
by sorry

end NUMINAMATH_CALUDE_opposite_signs_and_greater_magnitude_l1375_137537


namespace NUMINAMATH_CALUDE_complement_A_in_U_l1375_137585

-- Define the universal set U
def U : Set ℝ := {x | x > 0}

-- Define set A
def A : Set ℝ := {x | x ≥ 2}

-- State the theorem
theorem complement_A_in_U : 
  (U \ A) = {x : ℝ | 0 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_complement_A_in_U_l1375_137585


namespace NUMINAMATH_CALUDE_total_triangles_is_200_l1375_137589

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  sideLength : ℕ

/-- Represents the large equilateral triangle -/
def largeTriangle : EquilateralTriangle :=
  { sideLength := 10 }

/-- Represents a small equilateral triangle -/
def smallTriangle : EquilateralTriangle :=
  { sideLength := 1 }

/-- The number of small triangles that fit in the large triangle -/
def numSmallTriangles : ℕ := 100

/-- Counts the number of equilateral triangles of a given side length -/
def countTriangles (sideLength : ℕ) : ℕ :=
  if sideLength = 1 then numSmallTriangles
  else if sideLength > largeTriangle.sideLength then 0
  else largeTriangle.sideLength - sideLength + 1

/-- The total number of equilateral triangles -/
def totalTriangles : ℕ :=
  (List.range largeTriangle.sideLength).map countTriangles |>.sum

theorem total_triangles_is_200 : totalTriangles = 200 := by
  sorry

end NUMINAMATH_CALUDE_total_triangles_is_200_l1375_137589


namespace NUMINAMATH_CALUDE_total_paintable_area_is_1520_l1375_137542

-- Define the parameters of the problem
def num_bedrooms : ℕ := 4
def room_length : ℝ := 14
def room_width : ℝ := 11
def room_height : ℝ := 9
def unpaintable_area : ℝ := 70

-- Calculate the total wall area of one bedroom
def total_wall_area : ℝ := 2 * (room_length * room_height + room_width * room_height)

-- Calculate the paintable area of one bedroom
def paintable_area_per_room : ℝ := total_wall_area - unpaintable_area

-- Theorem statement
theorem total_paintable_area_is_1520 :
  num_bedrooms * paintable_area_per_room = 1520 := by
  sorry

end NUMINAMATH_CALUDE_total_paintable_area_is_1520_l1375_137542


namespace NUMINAMATH_CALUDE_rational_sum_theorem_l1375_137505

theorem rational_sum_theorem (x y : ℚ) 
  (hx : |x| = 5) 
  (hy : |y| = 2) 
  (hxy : |x - y| = x - y) : 
  x + y = 7 ∨ x + y = 3 := by
sorry

end NUMINAMATH_CALUDE_rational_sum_theorem_l1375_137505


namespace NUMINAMATH_CALUDE_min_value_trig_expression_l1375_137556

theorem min_value_trig_expression (x : Real) (h : 0 < x ∧ x < π / 2) :
  (8 / Real.sin x) + (1 / Real.cos x) ≥ 10 := by
  sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_l1375_137556


namespace NUMINAMATH_CALUDE_roots_of_quadratic_equation_l1375_137599

theorem roots_of_quadratic_equation :
  let f : ℝ → ℝ := λ x ↦ x^2 - 3
  ∃ x₁ x₂ : ℝ, x₁ = Real.sqrt 3 ∧ x₂ = -Real.sqrt 3 ∧ f x₁ = 0 ∧ f x₂ = 0 :=
by sorry

end NUMINAMATH_CALUDE_roots_of_quadratic_equation_l1375_137599


namespace NUMINAMATH_CALUDE_area_between_concentric_circles_l1375_137580

theorem area_between_concentric_circles 
  (r : ℝ)  -- radius of inner circle
  (h1 : r > 0)  -- radius is positive
  (h2 : 3*r - r = 3)  -- width of gray region is 3
  : π * (3*r)^2 - π * r^2 = 18 * π := by
  sorry

end NUMINAMATH_CALUDE_area_between_concentric_circles_l1375_137580


namespace NUMINAMATH_CALUDE_survey_theorem_l1375_137557

/-- Represents the response of a student to a subject --/
inductive Response
| Yes
| No
| Unsure

/-- Represents a subject with its response counts --/
structure Subject where
  yes_count : Nat
  no_count : Nat
  unsure_count : Nat

/-- The survey results --/
structure SurveyResults where
  total_students : Nat
  subject_m : Subject
  subject_r : Subject
  yes_only_m : Nat

def SurveyResults.students_not_yes_either (results : SurveyResults) : Nat :=
  results.total_students - (results.subject_m.yes_count + results.subject_r.yes_count - results.yes_only_m)

theorem survey_theorem (results : SurveyResults) 
  (h1 : results.total_students = 800)
  (h2 : results.subject_m.yes_count = 500)
  (h3 : results.subject_m.no_count = 200)
  (h4 : results.subject_m.unsure_count = 100)
  (h5 : results.subject_r.yes_count = 400)
  (h6 : results.subject_r.no_count = 100)
  (h7 : results.subject_r.unsure_count = 300)
  (h8 : results.yes_only_m = 170) :
  results.students_not_yes_either = 230 := by
  sorry

#eval SurveyResults.students_not_yes_either {
  total_students := 800,
  subject_m := { yes_count := 500, no_count := 200, unsure_count := 100 },
  subject_r := { yes_count := 400, no_count := 100, unsure_count := 300 },
  yes_only_m := 170
}

end NUMINAMATH_CALUDE_survey_theorem_l1375_137557


namespace NUMINAMATH_CALUDE_trigonometric_product_equals_three_fourths_l1375_137569

theorem trigonometric_product_equals_three_fourths :
  let cos30 : ℝ := Real.sqrt 3 / 2
  let sin60 : ℝ := Real.sqrt 3 / 2
  let sin30 : ℝ := 1 / 2
  let cos60 : ℝ := 1 / 2
  (1 - 1 / cos30) * (1 + 1 / sin60) * (1 - 1 / sin30) * (1 + 1 / cos60) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_product_equals_three_fourths_l1375_137569


namespace NUMINAMATH_CALUDE_cost_exceeds_fifty_l1375_137519

/-- Calculates the total cost of items after discount and tax --/
def total_cost (pizza_price : ℝ) (juice_price : ℝ) (chips_price : ℝ) (chocolate_price : ℝ)
  (pizza_count : ℕ) (juice_count : ℕ) (chips_count : ℕ) (chocolate_count : ℕ)
  (pizza_discount : ℝ) (sales_tax : ℝ) : ℝ :=
  let pizza_cost := pizza_price * pizza_count
  let juice_cost := juice_price * juice_count
  let chips_cost := chips_price * chips_count
  let chocolate_cost := chocolate_price * chocolate_count
  let discounted_pizza_cost := pizza_cost * (1 - pizza_discount)
  let subtotal := discounted_pizza_cost + juice_cost + chips_cost + chocolate_cost
  subtotal * (1 + sales_tax)

/-- Theorem: The total cost exceeds $50 --/
theorem cost_exceeds_fifty :
  total_cost 15 4 3.5 1.25 3 4 2 5 0.1 0.05 > 50 := by
  sorry

end NUMINAMATH_CALUDE_cost_exceeds_fifty_l1375_137519


namespace NUMINAMATH_CALUDE_harmonic_progression_logarithm_equality_l1375_137533

/-- Given x, y, z form a harmonic progression, 
    prove that lg (x+z) + lg (x-2y+z) = 2 lg (x-z) -/
theorem harmonic_progression_logarithm_equality 
  (x y z : ℝ) 
  (h : (1/x + 1/z)/2 = 1/y) : 
  Real.log (x+z) + Real.log (x-2*y+z) = 2 * Real.log (x-z) := by
  sorry

end NUMINAMATH_CALUDE_harmonic_progression_logarithm_equality_l1375_137533


namespace NUMINAMATH_CALUDE_min_distance_sum_l1375_137534

noncomputable def circle_C (x y : ℝ) : Prop := (x - 4)^2 + y^2 = 8

noncomputable def circle_D (x y : ℝ) : Prop := x^2 + (y - 4)^2 = 5

def origin : ℝ × ℝ := (0, 0)

theorem min_distance_sum (P Q : ℝ × ℝ) (hP : circle_C P.1 P.2) (hQ : circle_D Q.1 Q.2) :
  ∃ (min_val : ℝ), min_val = Real.sqrt 5 ∧
  ∀ (P' Q' : ℝ × ℝ), circle_C P'.1 P'.2 → circle_D Q'.1 Q'.2 →
    Real.sqrt ((P'.1 - Q'.1)^2 + (P'.2 - Q'.2)^2) + 
    (Real.sqrt 2 / 2) * Real.sqrt (P'.1^2 + P'.2^2) ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_distance_sum_l1375_137534


namespace NUMINAMATH_CALUDE_expression_value_l1375_137507

theorem expression_value (x y : ℝ) (h : x - 2*y = -1) : 6 + 2*x - 4*y = 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1375_137507


namespace NUMINAMATH_CALUDE_simplify_expression_l1375_137582

theorem simplify_expression (y : ℝ) : (5 * y)^3 + (4 * y) * (y^2) = 129 * y^3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1375_137582


namespace NUMINAMATH_CALUDE_largest_additional_plates_is_largest_possible_l1375_137551

/-- Represents the number of choices in each section of a license plate --/
structure LicensePlateChoices where
  section1 : Nat
  section2 : Nat
  section3 : Nat

/-- Calculates the total number of possible license plates --/
def totalPlates (choices : LicensePlateChoices) : Nat :=
  choices.section1 * choices.section2 * choices.section3

/-- The initial number of choices for each section --/
def initialChoices : LicensePlateChoices :=
  { section1 := 5, section2 := 3, section3 := 3 }

/-- The optimal distribution of new letters --/
def optimalChoices : LicensePlateChoices :=
  { section1 := 5, section2 := 5, section3 := 4 }

/-- Theorem stating that the largest number of additional plates is 55 --/
theorem largest_additional_plates :
  totalPlates optimalChoices - totalPlates initialChoices = 55 := by
  sorry

/-- Theorem stating that this is indeed the largest possible number --/
theorem is_largest_possible (newChoices : LicensePlateChoices)
  (h1 : newChoices.section1 + newChoices.section2 + newChoices.section3 = 
        initialChoices.section1 + initialChoices.section2 + initialChoices.section3 + 3) :
  totalPlates newChoices - totalPlates initialChoices ≤ 55 := by
  sorry

end NUMINAMATH_CALUDE_largest_additional_plates_is_largest_possible_l1375_137551


namespace NUMINAMATH_CALUDE_white_dandelions_on_saturday_l1375_137503

/-- Represents the state of dandelions in a meadow on a given day -/
structure DandelionState :=
  (yellow : ℕ)
  (white : ℕ)

/-- The dandelion blooming cycle -/
def dandelionCycle : ℕ := 5

/-- The number of days a dandelion remains yellow -/
def yellowDays : ℕ := 3

/-- The state of dandelions on Monday -/
def mondayState : DandelionState :=
  { yellow := 20, white := 14 }

/-- The state of dandelions on Wednesday -/
def wednesdayState : DandelionState :=
  { yellow := 15, white := 11 }

/-- The number of days between Monday and Saturday -/
def daysToSaturday : ℕ := 5

theorem white_dandelions_on_saturday :
  (wednesdayState.yellow + wednesdayState.white) - mondayState.yellow =
  (mondayState.yellow + mondayState.white + daysToSaturday - dandelionCycle) :=
by sorry

end NUMINAMATH_CALUDE_white_dandelions_on_saturday_l1375_137503


namespace NUMINAMATH_CALUDE_vector_addition_and_scalar_multiplication_l1375_137593

/-- Given vectors a and b, prove that c = 2a + 5b has the specified coordinates -/
theorem vector_addition_and_scalar_multiplication (a b : ℝ × ℝ × ℝ) 
  (h1 : a = (3, -4, 5)) 
  (h2 : b = (-1, 0, -2)) : 
  (2 : ℝ) • a + (5 : ℝ) • b = (1, -8, 0) := by
  sorry

#check vector_addition_and_scalar_multiplication

end NUMINAMATH_CALUDE_vector_addition_and_scalar_multiplication_l1375_137593


namespace NUMINAMATH_CALUDE_sum_equals_negative_power_l1375_137571

open BigOperators

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the sum
def T : ℤ := ∑ k in Finset.range 25, (-1:ℤ)^k * binomial 50 (2*k)

-- Theorem statement
theorem sum_equals_negative_power : T = -2^25 := by sorry

end NUMINAMATH_CALUDE_sum_equals_negative_power_l1375_137571


namespace NUMINAMATH_CALUDE_find_x_l1375_137575

theorem find_x (a b c d x : ℝ) 
  (h1 : a ≠ b) 
  (h2 : b ≠ 0) 
  (h3 : (a + x) / (b + x) = 4 * a / (3 * b)) 
  (h4 : c = 4 * a) 
  (h5 : d = 3 * b) :
  x = a * b / (3 * b - 4 * a) :=
by sorry

end NUMINAMATH_CALUDE_find_x_l1375_137575


namespace NUMINAMATH_CALUDE_total_pencils_l1375_137536

/-- Given an initial number of pencils and a number of pencils added,
    the total number of pencils is equal to the sum of the initial number and the added number. -/
theorem total_pencils (initial : ℕ) (added : ℕ) :
  initial + added = initial + added :=
by sorry

end NUMINAMATH_CALUDE_total_pencils_l1375_137536


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_squared_l1375_137558

noncomputable def a : ℝ := Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 6
noncomputable def b : ℝ := -Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 6
noncomputable def c : ℝ := Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 6
noncomputable def d : ℝ := -Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 6

theorem sum_of_reciprocals_squared :
  (1/a + 1/b + 1/c + 1/d)^2 = 16*(11 + 2*Real.sqrt 30) / (11 + 2*Real.sqrt 30 - 3*Real.sqrt 6)^2 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_squared_l1375_137558


namespace NUMINAMATH_CALUDE_min_value_function_equality_condition_l1375_137552

theorem min_value_function (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x^2 + 12*x + y + 64/(x^2 * y) ≥ 81 :=
by sorry

theorem equality_condition (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x^2 + 12*x + y + 64/(x^2 * y) = 81 ↔ x = 4 ∧ y = 16 :=
by sorry

end NUMINAMATH_CALUDE_min_value_function_equality_condition_l1375_137552


namespace NUMINAMATH_CALUDE_circumradius_of_special_triangle_l1375_137596

/-- Represents a triangle with consecutive natural number side lengths and an inscribed circle radius of 4 -/
structure SpecialTriangle where
  n : ℕ
  side_a : ℕ := n - 1
  side_b : ℕ := n
  side_c : ℕ := n + 1
  inradius : ℝ := 4

/-- The radius of the circumcircle of a SpecialTriangle is 65/8 -/
theorem circumradius_of_special_triangle (t : SpecialTriangle) : 
  (t.side_a : ℝ) * t.side_b * t.side_c / (4 * t.inradius * (t.side_a + t.side_b + t.side_c) / 2) = 65 / 8 := by
  sorry

end NUMINAMATH_CALUDE_circumradius_of_special_triangle_l1375_137596


namespace NUMINAMATH_CALUDE_additional_toothpicks_eq_351_l1375_137578

/-- The number of toothpicks needed for a 3-step staircase -/
def initial_toothpicks : ℕ := 18

/-- The ratio of the geometric progression for toothpick increase -/
def progression_ratio : ℕ := 3

/-- Calculate the total additional toothpicks needed to complete a 6-step staircase -/
def additional_toothpicks : ℕ :=
  let step4_increase := (initial_toothpicks / 2) * progression_ratio
  let step5_increase := step4_increase * progression_ratio
  let step6_increase := step5_increase * progression_ratio
  step4_increase + step5_increase + step6_increase

/-- Theorem stating that the additional toothpicks needed is 351 -/
theorem additional_toothpicks_eq_351 : additional_toothpicks = 351 := by
  sorry

end NUMINAMATH_CALUDE_additional_toothpicks_eq_351_l1375_137578


namespace NUMINAMATH_CALUDE_relay_race_distance_l1375_137523

/-- A runner in the relay race -/
structure Runner where
  name : String
  speed : Real
  time : Real

/-- The relay race -/
def RelayRace (runners : List Runner) (totalTime : Real) : Prop :=
  (List.sum (List.map (fun r => r.speed * r.time) runners) = 17) ∧
  (List.sum (List.map (fun r => r.time) runners) = totalTime)

theorem relay_race_distance :
  let sadie : Runner := ⟨"Sadie", 3, 2⟩
  let ariana : Runner := ⟨"Ariana", 6, 0.5⟩
  let sarah : Runner := ⟨"Sarah", 4, 2⟩
  let runners : List Runner := [sadie, ariana, sarah]
  let totalTime : Real := 4.5
  RelayRace runners totalTime := by sorry

end NUMINAMATH_CALUDE_relay_race_distance_l1375_137523


namespace NUMINAMATH_CALUDE_largest_angle_measure_l1375_137555

/-- A convex heptagon with interior angles as specified -/
structure ConvexHeptagon where
  x : ℝ
  angle1 : ℝ := x + 2
  angle2 : ℝ := 2 * x
  angle3 : ℝ := 3 * x
  angle4 : ℝ := 4 * x
  angle5 : ℝ := 5 * x
  angle6 : ℝ := 6 * x - 2
  angle7 : ℝ := 7 * x - 3

/-- The sum of interior angles of a heptagon is 900 degrees -/
axiom heptagon_angle_sum (h : ConvexHeptagon) : 
  h.angle1 + h.angle2 + h.angle3 + h.angle4 + h.angle5 + h.angle6 + h.angle7 = 900

/-- The measure of the largest angle in the specified convex heptagon is 222.75 degrees -/
theorem largest_angle_measure (h : ConvexHeptagon) : 
  max h.angle1 (max h.angle2 (max h.angle3 (max h.angle4 (max h.angle5 (max h.angle6 h.angle7))))) = 222.75 := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_measure_l1375_137555


namespace NUMINAMATH_CALUDE_rachel_homework_l1375_137539

theorem rachel_homework (reading_pages math_pages : ℕ) : 
  reading_pages = math_pages + 6 →
  reading_pages = 14 →
  math_pages = 8 := by sorry

end NUMINAMATH_CALUDE_rachel_homework_l1375_137539


namespace NUMINAMATH_CALUDE_rebus_solution_l1375_137513

/-- Represents a two-digit number (or single-digit for YA) -/
def TwoDigitNum := {n : ℕ // n < 100}

/-- The rebus equation -/
def rebusEquation (ya oh my : TwoDigitNum) : Prop :=
  ya.val + 8 * oh.val = my.val

/-- All digits in the equation are different -/
def differentDigits (ya oh my : TwoDigitNum) : Prop :=
  ya.val ≠ oh.val ∧ ya.val ≠ my.val ∧ oh.val ≠ my.val

theorem rebus_solution :
  ∃! (ya oh my : TwoDigitNum),
    rebusEquation ya oh my ∧
    differentDigits ya oh my ∧
    ya.val = 0 ∧
    oh.val = 12 ∧
    my.val = 96 :=
sorry

end NUMINAMATH_CALUDE_rebus_solution_l1375_137513


namespace NUMINAMATH_CALUDE_proportion_solution_l1375_137525

theorem proportion_solution (x : ℝ) :
  (0.75 : ℝ) / x = (4.5 : ℝ) / (7/3 : ℝ) →
  x = (0.75 : ℝ) * (7/3 : ℝ) / (4.5 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_proportion_solution_l1375_137525


namespace NUMINAMATH_CALUDE_gain_percent_calculation_l1375_137516

theorem gain_percent_calculation (cost_price selling_price : ℝ) 
  (h1 : cost_price = 100)
  (h2 : selling_price = 115) :
  (selling_price - cost_price) / cost_price * 100 = 15 := by
sorry

end NUMINAMATH_CALUDE_gain_percent_calculation_l1375_137516


namespace NUMINAMATH_CALUDE_original_price_satisfies_conditions_l1375_137591

/-- The original price of a dish that satisfies the given conditions --/
def original_price : ℝ := 42

/-- John's payment given the original price --/
def john_payment (price : ℝ) : ℝ := 0.9 * price + 0.15 * price

/-- Jane's payment given the original price --/
def jane_payment (price : ℝ) : ℝ := 0.9 * price + 0.15 * (0.9 * price)

/-- Theorem stating that the original price satisfies the given conditions --/
theorem original_price_satisfies_conditions :
  john_payment original_price - jane_payment original_price = 0.63 :=
sorry

end NUMINAMATH_CALUDE_original_price_satisfies_conditions_l1375_137591


namespace NUMINAMATH_CALUDE_commute_time_difference_l1375_137540

/-- Given a set of 5 commute times with known average and variance, prove the absolute difference between two unknown times. -/
theorem commute_time_difference (x y : ℝ) : 
  (x + y + 10 + 11 + 9) / 5 = 10 →
  ((x - 10)^2 + (y - 10)^2 + 0^2 + 1^2 + (-1)^2) / 5 = 2 →
  |x - y| = 4 := by
  sorry

end NUMINAMATH_CALUDE_commute_time_difference_l1375_137540


namespace NUMINAMATH_CALUDE_tina_katya_difference_l1375_137544

/-- The number of glasses of lemonade sold by each person -/
structure LemonadeSales where
  katya : ℕ
  ricky : ℕ
  tina : ℕ

/-- The conditions of the lemonade sales problem -/
def lemonade_problem (sales : LemonadeSales) : Prop :=
  sales.katya = 8 ∧
  sales.ricky = 9 ∧
  sales.tina = 2 * (sales.katya + sales.ricky)

/-- The theorem stating the difference between Tina's and Katya's sales -/
theorem tina_katya_difference (sales : LemonadeSales) 
  (h : lemonade_problem sales) : sales.tina - sales.katya = 26 := by
  sorry

end NUMINAMATH_CALUDE_tina_katya_difference_l1375_137544


namespace NUMINAMATH_CALUDE_division_of_decimals_l1375_137541

theorem division_of_decimals : (0.08 / 0.002) / 0.04 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_division_of_decimals_l1375_137541


namespace NUMINAMATH_CALUDE_cabin_rental_security_deposit_l1375_137518

/-- Calculate the security deposit for a cabin rental --/
theorem cabin_rental_security_deposit :
  let rental_period : ℕ := 14 -- 2 weeks
  let daily_rate : ℚ := 125
  let pet_fee : ℚ := 100
  let service_fee_rate : ℚ := 1/5 -- 20%
  let security_deposit_rate : ℚ := 1/2 -- 50%

  let rental_cost := rental_period * daily_rate
  let subtotal := rental_cost + pet_fee
  let service_fee := subtotal * service_fee_rate
  let total_cost := subtotal + service_fee
  let security_deposit := total_cost * security_deposit_rate

  security_deposit = 1110
  := by sorry

end NUMINAMATH_CALUDE_cabin_rental_security_deposit_l1375_137518


namespace NUMINAMATH_CALUDE_infinite_square_root_equals_three_l1375_137573

theorem infinite_square_root_equals_three :
  ∃ x : ℝ, x > 0 ∧ x = Real.sqrt (3 + 2 * x) → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_infinite_square_root_equals_three_l1375_137573


namespace NUMINAMATH_CALUDE_gift_wrapping_problem_l1375_137587

/-- Given three rolls of wrapping paper where the first roll wraps 3 gifts,
    the second roll wraps 5 gifts, and the third roll wraps 4 gifts with no paper leftover,
    prove that the total number of gifts wrapped is 12. -/
theorem gift_wrapping_problem (rolls : Nat) (first_roll : Nat) (second_roll : Nat) (third_roll : Nat)
    (h1 : rolls = 3)
    (h2 : first_roll = 3)
    (h3 : second_roll = 5)
    (h4 : third_roll = 4)
    (h5 : rolls * first_roll ≥ first_roll + second_roll + third_roll) :
    first_roll + second_roll + third_roll = 12 := by
  sorry

end NUMINAMATH_CALUDE_gift_wrapping_problem_l1375_137587


namespace NUMINAMATH_CALUDE_min_value_of_b_l1375_137501

def S (n : ℕ) : ℕ := 2^n - 1

def a : ℕ → ℝ
  | 0 => 1
  | n + 1 => S (n + 1) - S n

def b (n : ℕ) : ℝ := (a n)^2 - 7*(a n) + 6

theorem min_value_of_b :
  ∃ (m : ℝ), ∀ (n : ℕ), b n ≥ m ∧ ∃ (k : ℕ), b k = m ∧ m = -6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_b_l1375_137501


namespace NUMINAMATH_CALUDE_minimum_nickels_needed_l1375_137530

def jacket_cost : ℚ := 45
def ten_dollar_bills : ℕ := 4
def quarters : ℕ := 10
def nickel_value : ℚ := 0.05

theorem minimum_nickels_needed : 
  ∀ n : ℕ, (ten_dollar_bills * 10 + quarters * 0.25 + n * nickel_value ≥ jacket_cost) → n ≥ 50 := by
  sorry

end NUMINAMATH_CALUDE_minimum_nickels_needed_l1375_137530


namespace NUMINAMATH_CALUDE_hotel_flat_fee_l1375_137574

/-- Given a hotel's pricing structure and two customer payments, prove the flat fee for the first night. -/
theorem hotel_flat_fee (f n : ℝ) 
  (ann_payment : f + n = 120)
  (bob_payment : f + 6 * n = 330) :
  f = 78 := by sorry

end NUMINAMATH_CALUDE_hotel_flat_fee_l1375_137574


namespace NUMINAMATH_CALUDE_first_term_of_geometric_series_l1375_137528

/-- The first term of an infinite geometric series with common ratio 1/4 and sum 80 is 60 -/
theorem first_term_of_geometric_series : 
  ∀ (a : ℝ), 
  (a * (1 - (1/4)⁻¹) = 80) → 
  a = 60 := by
sorry

end NUMINAMATH_CALUDE_first_term_of_geometric_series_l1375_137528


namespace NUMINAMATH_CALUDE_acme_savings_at_min_shirts_l1375_137563

/-- Acme T-Shirt Company's pricing function -/
def acme_cost (n : ℕ) : ℚ :=
  if n ≤ 20 then 60 + 10 * n
  else (60 + 10 * n) * (9/10)

/-- Beta T-Shirt Company's pricing function -/
def beta_cost (n : ℕ) : ℚ := 15 * n

/-- The minimum number of shirts for which Acme is cheaper than Beta -/
def min_shirts_for_acme_savings : ℕ := 13

theorem acme_savings_at_min_shirts :
  acme_cost min_shirts_for_acme_savings < beta_cost min_shirts_for_acme_savings ∧
  ∀ k : ℕ, k < min_shirts_for_acme_savings →
    acme_cost k ≥ beta_cost k :=
by sorry

end NUMINAMATH_CALUDE_acme_savings_at_min_shirts_l1375_137563


namespace NUMINAMATH_CALUDE_fuel_spending_reduction_l1375_137508

theorem fuel_spending_reduction 
  (old_efficiency : ℝ) 
  (old_fuel_cost : ℝ) 
  (efficiency_improvement : ℝ) 
  (fuel_cost_increase : ℝ) : 
  let new_efficiency : ℝ := old_efficiency * (1 + efficiency_improvement)
  let new_fuel_cost : ℝ := old_fuel_cost * (1 + fuel_cost_increase)
  let old_trip_cost : ℝ := old_fuel_cost
  let new_trip_cost : ℝ := (1 / (1 + efficiency_improvement)) * new_fuel_cost
  let cost_reduction : ℝ := (old_trip_cost - new_trip_cost) / old_trip_cost
  efficiency_improvement = 0.75 ∧ 
  fuel_cost_increase = 0.30 → 
  cost_reduction = 25 / 28 := by
sorry

end NUMINAMATH_CALUDE_fuel_spending_reduction_l1375_137508


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_24_l1375_137510

/-- An arithmetic sequence where a_n = 2n - 3 -/
def a (n : ℕ) : ℤ := 2 * n - 3

/-- Sum of the first n terms of the arithmetic sequence -/
def S (n : ℕ) : ℤ := (n : ℤ) * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2

theorem arithmetic_sequence_sum_24 :
  ∃ m : ℕ, m > 0 ∧ S m = 24 ∧ ∀ k : ℕ, k > 0 ∧ S k = 24 → k = m :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_24_l1375_137510


namespace NUMINAMATH_CALUDE_system_solution_l1375_137506

theorem system_solution :
  ∃! (x y : ℝ),
    Real.sqrt (2016.5 + x) + Real.sqrt (2016.5 + y) = 114 ∧
    Real.sqrt (2016.5 - x) + Real.sqrt (2016.5 - y) = 56 ∧
    x = 1232.5 ∧ y = 1232.5 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1375_137506


namespace NUMINAMATH_CALUDE_covered_digits_sum_l1375_137577

/-- Represents a five-digit number with some digits possibly covered -/
structure PartialNumber :=
  (d1 d2 d3 d4 d5 : Option Nat)

/-- The sum of the visible digits in a PartialNumber -/
def visibleSum (n : PartialNumber) : Nat :=
  (n.d1.getD 0) * 10000 + (n.d2.getD 0) * 1000 + (n.d3.getD 0) * 100 + (n.d4.getD 0) * 10 + (n.d5.getD 0)

/-- The number of covered digits in a PartialNumber -/
def coveredCount (n : PartialNumber) : Nat :=
  (if n.d1.isNone then 1 else 0) +
  (if n.d2.isNone then 1 else 0) +
  (if n.d3.isNone then 1 else 0) +
  (if n.d4.isNone then 1 else 0) +
  (if n.d5.isNone then 1 else 0)

theorem covered_digits_sum (n1 n2 n3 : PartialNumber) :
  visibleSum n1 + visibleSum n2 + visibleSum n3 = 57263 - 1000 - 200 - 9 ∧
  coveredCount n1 + coveredCount n2 + coveredCount n3 = 3 →
  ∃ (p1 p2 p3 : Nat), 
    (p1 = 1 ∧ p2 = 2 ∧ p3 = 9) ∧
    visibleSum n1 + visibleSum n2 + visibleSum n3 + p1 * 1000 + p2 * 100 + p3 = 57263 :=
by sorry


end NUMINAMATH_CALUDE_covered_digits_sum_l1375_137577


namespace NUMINAMATH_CALUDE_mashed_potatoes_vs_bacon_l1375_137547

theorem mashed_potatoes_vs_bacon (mashed_potatoes bacon : ℕ) 
  (h1 : mashed_potatoes = 408) 
  (h2 : bacon = 42) : 
  mashed_potatoes - bacon = 366 := by
  sorry

end NUMINAMATH_CALUDE_mashed_potatoes_vs_bacon_l1375_137547


namespace NUMINAMATH_CALUDE_fraction_decimal_digits_l1375_137595

/-- The number of digits to the right of the decimal point when a fraction is expressed as a decimal. -/
def decimal_digits (n : ℚ) : ℕ := sorry

/-- The fraction we're considering -/
def fraction : ℚ := 3^6 / (6^4 * 625)

/-- Theorem stating that the number of digits to the right of the decimal point
    in the decimal representation of our fraction is 4 -/
theorem fraction_decimal_digits :
  decimal_digits fraction = 4 := by sorry

end NUMINAMATH_CALUDE_fraction_decimal_digits_l1375_137595


namespace NUMINAMATH_CALUDE_color_drawing_cost_is_240_l1375_137522

/-- The cost of a color drawing given the cost of a black and white drawing and the additional percentage for color. -/
def color_drawing_cost (bw_cost : ℝ) (color_percentage : ℝ) : ℝ :=
  bw_cost * (1 + color_percentage)

/-- Theorem stating that the cost of a color drawing is $240 given the specified conditions. -/
theorem color_drawing_cost_is_240 :
  color_drawing_cost 160 0.5 = 240 := by
  sorry

end NUMINAMATH_CALUDE_color_drawing_cost_is_240_l1375_137522


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l1375_137520

/-- Sum of first n terms of an arithmetic sequence -/
def S (a : ℚ) (n : ℕ) : ℚ := n * (2 * a + (n - 1) * 5) / 2

/-- The theorem statement -/
theorem arithmetic_sequence_first_term (a : ℚ) :
  (∃ c : ℚ, ∀ n : ℕ, n > 0 → S a (2 * n) / S a n = c) →
  a = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l1375_137520


namespace NUMINAMATH_CALUDE_unmarked_trees_l1375_137566

def mark_trees (n : ℕ) : Finset ℕ :=
  Finset.filter (fun i => i % 2 = 1 ∨ i % 3 = 1) (Finset.range n)

theorem unmarked_trees :
  (Finset.range 13 \ mark_trees 13).card = 4 := by
  sorry

end NUMINAMATH_CALUDE_unmarked_trees_l1375_137566


namespace NUMINAMATH_CALUDE_sector_perimeter_l1375_137531

/-- Given a circular sector with central angle 4 radians and area 2 cm², 
    its perimeter is 6 cm -/
theorem sector_perimeter (θ : Real) (A : Real) (r : Real) : 
  θ = 4 → A = 2 → (1/2) * θ * r^2 = A → r + r + θ * r = 6 := by
  sorry

end NUMINAMATH_CALUDE_sector_perimeter_l1375_137531


namespace NUMINAMATH_CALUDE_trapezoid_median_length_l1375_137584

/-- Given a triangle and a trapezoid with the same altitude and equal areas,
    prove that the median of the trapezoid is 24 inches. -/
theorem trapezoid_median_length (h : ℝ) (triangle_area trapezoid_area : ℝ) : 
  triangle_area = (1/2) * 24 * h →
  trapezoid_area = ((15 + 33) / 2) * h →
  triangle_area = trapezoid_area →
  (15 + 33) / 2 = 24 := by
sorry


end NUMINAMATH_CALUDE_trapezoid_median_length_l1375_137584


namespace NUMINAMATH_CALUDE_find_s_l1375_137590

theorem find_s (r s : ℝ) (hr : r > 1) (hs : s > 1) 
  (h1 : 1/r + 1/s = 1) (h2 : r*s = 9) : 
  s = (9 + 3*Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_find_s_l1375_137590


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l1375_137511

/-- Given a line L1 with equation 6x - 3y = 9 and a point P (1, -2),
    prove that the line L2 passing through P and parallel to L1
    has the equation y = 2x - 4 in slope-intercept form. -/
theorem parallel_line_through_point (x y : ℝ) :
  (6 * x - 3 * y = 9) →  -- Equation of L1
  (∃ m b : ℝ, ∀ x y : ℝ, y = m * x + b ↔ 6 * x - 3 * y = 9) →  -- L1 in slope-intercept form
  (∃ m' b' : ℝ, 
    (m' = 2 ∧ b' = -4) ∧  -- Equation of L2: y = 2x - 4
    (m' * 1 + b' = -2) ∧  -- L2 passes through (1, -2)
    (∀ x y : ℝ, y = m' * x + b' ↔ y = 2 * x - 4)) :=
by sorry


end NUMINAMATH_CALUDE_parallel_line_through_point_l1375_137511


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1375_137559

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  (∀ n : ℕ, a (n + 1) = a n * q) →  -- Geometric sequence condition
  q ≠ 1 →  -- Common ratio not equal to 1
  a 1 * a 2 * a 3 = -1/8 →  -- Product of first three terms
  2 * a 4 = a 2 + a 3 →  -- Arithmetic sequence condition
  a 1 + a 2 + a 3 + a 4 = 5/8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1375_137559


namespace NUMINAMATH_CALUDE_polar_bear_daily_fish_consumption_l1375_137543

/-- The amount of trout eaten daily by the polar bear in buckets -/
def trout_amount : ℝ := 0.2

/-- The amount of salmon eaten daily by the polar bear in buckets -/
def salmon_amount : ℝ := 0.4

/-- The total amount of fish eaten daily by the polar bear in buckets -/
def total_fish_amount : ℝ := trout_amount + salmon_amount

theorem polar_bear_daily_fish_consumption :
  total_fish_amount = 0.6 := by sorry

end NUMINAMATH_CALUDE_polar_bear_daily_fish_consumption_l1375_137543


namespace NUMINAMATH_CALUDE_power_comparison_l1375_137504

theorem power_comparison : 1.6^0.3 > 0.9^3.1 := by
  sorry

end NUMINAMATH_CALUDE_power_comparison_l1375_137504


namespace NUMINAMATH_CALUDE_sum_of_even_numbers_l1375_137550

theorem sum_of_even_numbers (n : ℕ) :
  2 * (n * (n + 1) / 2) = n^2 + n :=
sorry

end NUMINAMATH_CALUDE_sum_of_even_numbers_l1375_137550


namespace NUMINAMATH_CALUDE_equation_solution_l1375_137581

theorem equation_solution : 
  ∃! x : ℚ, (x^2 + 3*x + 5) / (x + 6) = x + 7 ∧ x = -37/10 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1375_137581


namespace NUMINAMATH_CALUDE_three_circles_theorem_l1375_137598

/-- Represents a circle with a center and a radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if two circles are externally tangent -/
def are_externally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 = (c1.radius + c2.radius)^2

/-- The area of the shaded region formed by three externally tangent circles -/
noncomputable def shaded_area (c1 c2 c3 : Circle) : ℝ := sorry

/-- The main theorem -/
theorem three_circles_theorem :
  let c1 : Circle := { center := (0, 0), radius := Real.sqrt 3 - 1 }
  let c2 : Circle := { center := (2, 0), radius := 3 - Real.sqrt 3 }
  let c3 : Circle := { center := (0, 2 * Real.sqrt 3), radius := 1 + Real.sqrt 3 }
  are_externally_tangent c1 c2 ∧
  are_externally_tangent c2 c3 ∧
  are_externally_tangent c3 c1 →
  ∃ (a b c : ℚ),
    shaded_area c1 c2 c3 = a * Real.sqrt 3 + b * Real.pi + c * Real.pi * Real.sqrt 3 ∧
    a + b + c = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_three_circles_theorem_l1375_137598


namespace NUMINAMATH_CALUDE_num_cities_with_protests_is_21_l1375_137597

/-- The number of cities experiencing protests given the specified conditions -/
def num_cities_with_protests : ℕ :=
  let protest_days : ℕ := 30
  let arrests_per_day_per_city : ℕ := 10
  let pre_trial_days : ℕ := 4
  let post_trial_days : ℕ := 7  -- half of 2-week sentence
  let total_jail_weeks : ℕ := 9900

  -- Calculate the number of cities
  21

/-- Theorem stating that the number of cities experiencing protests is 21 -/
theorem num_cities_with_protests_is_21 :
  num_cities_with_protests = 21 := by
  sorry

end NUMINAMATH_CALUDE_num_cities_with_protests_is_21_l1375_137597


namespace NUMINAMATH_CALUDE_two_intersecting_circles_common_tangents_l1375_137553

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The number of common external tangents for two intersecting circles -/
def commonExternalTangents (c1 c2 : Circle) : ℕ :=
  sorry

/-- The distance between two points in a 2D plane -/
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  sorry

theorem two_intersecting_circles_common_tangents :
  let c1 : Circle := { center := (1, 2), radius := 1 }
  let c2 : Circle := { center := (2, 5), radius := 3 }
  distance c1.center c2.center < c1.radius + c2.radius →
  commonExternalTangents c1 c2 = 2 :=
sorry

end NUMINAMATH_CALUDE_two_intersecting_circles_common_tangents_l1375_137553


namespace NUMINAMATH_CALUDE_mod_nine_equiv_l1375_137554

theorem mod_nine_equiv : ∃ (n : ℤ), 0 ≤ n ∧ n < 9 ∧ -1234 ≡ n [ZMOD 9] ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_mod_nine_equiv_l1375_137554


namespace NUMINAMATH_CALUDE_units_digit_of_M_M15_l1375_137529

-- Define the Modified Lucas sequence
def M : ℕ → ℕ
  | 0 => 3
  | 1 => 2
  | (n + 2) => M (n + 1) + M n

-- Define a function to get the units digit
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Theorem statement
theorem units_digit_of_M_M15 : unitsDigit (M (M 15)) = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_M_M15_l1375_137529


namespace NUMINAMATH_CALUDE_rice_sale_proof_l1375_137576

/-- Calculates the daily amount of rice to be sold given initial amount, additional amount, and number of days -/
def daily_rice_sale (initial_tons : ℕ) (additional_kg : ℕ) (days : ℕ) : ℕ :=
  (initial_tons * 1000 + additional_kg) / days

/-- Proves that given 4 tons of rice initially, with an additional 4000 kilograms transported in,
    and needing to be sold within 4 days, the amount of rice to be sold each day is 2000 kilograms -/
theorem rice_sale_proof : daily_rice_sale 4 4000 4 = 2000 := by
  sorry

end NUMINAMATH_CALUDE_rice_sale_proof_l1375_137576


namespace NUMINAMATH_CALUDE_cubic_difference_even_iff_sum_even_l1375_137561

theorem cubic_difference_even_iff_sum_even (p q : ℕ) :
  Even (p^3 - q^3) ↔ Even (p + q) := by sorry

end NUMINAMATH_CALUDE_cubic_difference_even_iff_sum_even_l1375_137561


namespace NUMINAMATH_CALUDE_polly_happy_tweets_l1375_137567

/-- Represents the number of tweets Polly makes per minute in different states -/
structure PollyTweets where
  happy : ℕ
  hungry : ℕ
  mirror : ℕ

/-- Represents the duration Polly spends in each state -/
structure Duration where
  happy : ℕ
  hungry : ℕ
  mirror : ℕ

/-- Calculates the total number of tweets given tweet rates and durations -/
def totalTweets (tweets : PollyTweets) (duration : Duration) : ℕ :=
  tweets.happy * duration.happy +
  tweets.hungry * duration.hungry +
  tweets.mirror * duration.mirror

/-- Theorem stating that Polly tweets 18 times per minute when happy -/
theorem polly_happy_tweets (tweets : PollyTweets) (duration : Duration) :
  tweets.hungry = 4 ∧
  tweets.mirror = 45 ∧
  duration.happy = 20 ∧
  duration.hungry = 20 ∧
  duration.mirror = 20 ∧
  totalTweets tweets duration = 1340 →
  tweets.happy = 18 := by
  sorry

end NUMINAMATH_CALUDE_polly_happy_tweets_l1375_137567


namespace NUMINAMATH_CALUDE_arithmetic_sequence_tenth_term_l1375_137526

theorem arithmetic_sequence_tenth_term
  (a₁ a₁₇ : ℚ)
  (h₁ : a₁ = 2 / 3)
  (h₂ : a₁₇ = 3 / 2)
  (h_arith : ∀ n : ℕ, n > 0 → ∃ d : ℚ, a₁₇ = a₁ + (17 - 1) * d ∧ ∀ k : ℕ, k > 0 → a₁ + (k - 1) * d = a₁ + (k - 1) * ((a₁₇ - a₁) / 16)) :
  a₁ + 9 * ((a₁₇ - a₁) / 16) = 109 / 96 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_tenth_term_l1375_137526


namespace NUMINAMATH_CALUDE_ryans_leaf_collection_l1375_137514

/-- Given Ryan's leaf collection scenario, prove the number of remaining leaves. -/
theorem ryans_leaf_collection :
  let initial_leaves : ℕ := 89
  let first_loss : ℕ := 24
  let second_loss : ℕ := 43
  initial_leaves - first_loss - second_loss = 22 :=
by
  sorry

end NUMINAMATH_CALUDE_ryans_leaf_collection_l1375_137514


namespace NUMINAMATH_CALUDE_platform_length_platform_length_proof_l1375_137521

/-- Calculates the length of a platform given train parameters --/
theorem platform_length (train_length : ℝ) (train_speed_kmph : ℝ) (crossing_time : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  let total_distance := train_speed_mps * crossing_time
  total_distance - train_length

/-- Proves that the platform length is 340 m given the specific parameters --/
theorem platform_length_proof :
  platform_length 160 72 25 = 340 := by
  sorry

end NUMINAMATH_CALUDE_platform_length_platform_length_proof_l1375_137521


namespace NUMINAMATH_CALUDE_division_problem_l1375_137568

theorem division_problem (total : ℝ) (a b c : ℝ) (h1 : total = 1080) 
  (h2 : a = (1/3) * (b + c)) (h3 : a = b + 30) (h4 : a + b + c = total) 
  (h5 : ∃ f : ℝ, b = f * (a + c)) : 
  ∃ f : ℝ, b = f * (a + c) ∧ f = 2/7 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1375_137568


namespace NUMINAMATH_CALUDE_stella_toilet_paper_stocking_l1375_137592

/-- Proves that Stella stocks 1 roll per day in each bathroom given the conditions --/
theorem stella_toilet_paper_stocking :
  let num_bathrooms : ℕ := 6
  let days_per_week : ℕ := 7
  let rolls_per_pack : ℕ := 12
  let weeks : ℕ := 4
  let packs_bought : ℕ := 14
  
  let total_rolls : ℕ := packs_bought * rolls_per_pack
  let rolls_per_week : ℕ := total_rolls / weeks
  let rolls_per_day : ℕ := rolls_per_week / days_per_week
  let rolls_per_bathroom_per_day : ℕ := rolls_per_day / num_bathrooms

  rolls_per_bathroom_per_day = 1 :=
by sorry

end NUMINAMATH_CALUDE_stella_toilet_paper_stocking_l1375_137592


namespace NUMINAMATH_CALUDE_factorization_sum_l1375_137532

theorem factorization_sum (a b : ℤ) : 
  (∀ x : ℝ, 25 * x^2 - 160 * x - 336 = (5 * x + a) * (5 * x + b)) → 
  a + 2 * b = 20 := by
  sorry

end NUMINAMATH_CALUDE_factorization_sum_l1375_137532


namespace NUMINAMATH_CALUDE_different_log_differences_l1375_137588

theorem different_log_differences (primes : Finset ℕ) : 
  primes = {3, 5, 7, 11} → 
  Finset.card (Finset.image (λ (p : ℕ × ℕ) => Real.log p.1 - Real.log p.2) 
    (Finset.filter (λ (p : ℕ × ℕ) => p.1 ≠ p.2) (primes.product primes))) = 12 := by
  sorry

end NUMINAMATH_CALUDE_different_log_differences_l1375_137588


namespace NUMINAMATH_CALUDE_coefficient_sum_l1375_137572

theorem coefficient_sum (b₅ b₄ b₃ b₂ b₁ b₀ : ℝ) :
  (∀ x : ℝ, (4*x - 2)^5 = b₅*x^5 + b₄*x^4 + b₃*x^3 + b₂*x^2 + b₁*x + b₀) →
  b₅ + b₄ + b₃ + b₂ + b₁ + b₀ = 32 := by
sorry

end NUMINAMATH_CALUDE_coefficient_sum_l1375_137572


namespace NUMINAMATH_CALUDE_zoe_drank_bottles_l1375_137535

def initial_bottles : ℕ := 42
def bought_bottles : ℕ := 30
def final_bottles : ℕ := 47

theorem zoe_drank_bottles :
  ∃ (drank_bottles : ℕ), initial_bottles - drank_bottles + bought_bottles = final_bottles ∧ drank_bottles = 25 := by
  sorry

end NUMINAMATH_CALUDE_zoe_drank_bottles_l1375_137535


namespace NUMINAMATH_CALUDE_initial_workers_correct_l1375_137570

/-- Represents the initial number of workers employed by the contractor -/
def initial_workers : ℕ := 360

/-- Represents the total number of days to complete the wall -/
def total_days : ℕ := 50

/-- Represents the number of days after which progress is measured -/
def days_passed : ℕ := 25

/-- Represents the percentage of work completed after 'days_passed' -/
def work_completed : ℚ := 2/5

/-- Represents the additional workers needed to complete the work on time -/
def additional_workers : ℕ := 90

/-- Theorem stating that the initial number of workers is correct given the conditions -/
theorem initial_workers_correct :
  initial_workers * (total_days : ℚ) = (initial_workers + additional_workers) * 
    (total_days * work_completed) :=
by sorry

end NUMINAMATH_CALUDE_initial_workers_correct_l1375_137570


namespace NUMINAMATH_CALUDE_y_derivative_l1375_137500

noncomputable def y (x : ℝ) : ℝ := Real.sqrt x + (1/3) * Real.arctan (Real.sqrt x) + (8/3) * Real.arctan (Real.sqrt x / 2)

theorem y_derivative (x : ℝ) (h : x > 0) : 
  deriv y x = (3 * x^2 + 16 * x + 32) / (6 * Real.sqrt x * (x + 1) * (x + 4)) :=
sorry

end NUMINAMATH_CALUDE_y_derivative_l1375_137500


namespace NUMINAMATH_CALUDE_adams_money_from_mother_l1375_137562

/-- Given Adam's initial savings and final total, prove the amount his mother gave him. -/
theorem adams_money_from_mother (initial_savings final_total : ℕ) 
  (h1 : initial_savings = 79)
  (h2 : final_total = 92)
  : final_total - initial_savings = 13 := by
  sorry

end NUMINAMATH_CALUDE_adams_money_from_mother_l1375_137562


namespace NUMINAMATH_CALUDE_art_of_passing_through_walls_l1375_137594

theorem art_of_passing_through_walls (n : ℝ) :
  (8 * Real.sqrt (8 / n) = Real.sqrt (8 * (8 / n))) ↔ n = 63 := by
  sorry

end NUMINAMATH_CALUDE_art_of_passing_through_walls_l1375_137594


namespace NUMINAMATH_CALUDE_rectangle_intersection_sum_l1375_137517

/-- A circle in the xy-plane -/
structure Circle where
  center : ℝ × ℝ
  equation : ℝ → ℝ → ℝ → Prop

/-- A line in the xy-plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Predicate to check if four points form a rectangle -/
def form_rectangle (p1 p2 p3 p4 : ℝ × ℝ) : Prop := sorry

/-- Predicate to check if a point lies on a circle -/
def on_circle (p : ℝ × ℝ) (c : Circle) : Prop := sorry

/-- Predicate to check if a point lies on a line -/
def on_line (p : ℝ × ℝ) (l : Line) : Prop := sorry

theorem rectangle_intersection_sum (m n k : ℝ) :
  let c : Circle := ⟨(-3/2, -1/2), fun x y k => x^2 + y^2 + 3*x + y + k = 0⟩
  let l₁ : Line := ⟨3, m⟩
  let l₂ : Line := ⟨3, n⟩
  (∃ p1 p2 p3 p4 : ℝ × ℝ,
    on_circle p1 c ∧ on_circle p2 c ∧ on_circle p3 c ∧ on_circle p4 c ∧
    ((on_line p1 l₁ ∧ on_line p2 l₁) ∨ (on_line p1 l₁ ∧ on_line p3 l₁) ∨
     (on_line p1 l₁ ∧ on_line p4 l₁) ∨ (on_line p2 l₁ ∧ on_line p3 l₁) ∨
     (on_line p2 l₁ ∧ on_line p4 l₁) ∨ (on_line p3 l₁ ∧ on_line p4 l₁)) ∧
    ((on_line p1 l₂ ∧ on_line p2 l₂) ∨ (on_line p1 l₂ ∧ on_line p3 l₂) ∨
     (on_line p1 l₂ ∧ on_line p4 l₂) ∨ (on_line p2 l₂ ∧ on_line p3 l₂) ∨
     (on_line p2 l₂ ∧ on_line p4 l₂) ∨ (on_line p3 l₂ ∧ on_line p4 l₂)) ∧
    form_rectangle p1 p2 p3 p4) →
  m + n = 8 := by sorry

end NUMINAMATH_CALUDE_rectangle_intersection_sum_l1375_137517


namespace NUMINAMATH_CALUDE_fundraising_ratio_approx_one_third_l1375_137538

/-- The ratio of Miss Rollin's class contribution to the total school fundraising --/
def fundraising_ratio : ℚ :=
  let johnson_amount := 2300
  let sutton_amount := johnson_amount / 2
  let rollin_amount := sutton_amount * 8
  let total_after_fees := 27048
  let total_before_fees := total_after_fees / 0.98
  rollin_amount / total_before_fees

theorem fundraising_ratio_approx_one_third :
  abs (fundraising_ratio - 1/3) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_fundraising_ratio_approx_one_third_l1375_137538


namespace NUMINAMATH_CALUDE_line_slope_thirty_degrees_l1375_137524

theorem line_slope_thirty_degrees (m : ℝ) : 
  (∃ (x y : ℝ), x + m * y - 3 = 0) →
  (Real.tan (30 * π / 180) = -1 / m) →
  m = -Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_line_slope_thirty_degrees_l1375_137524


namespace NUMINAMATH_CALUDE_inequality_solution_range_l1375_137545

theorem inequality_solution_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 3 * a * x + a - 2 < 0) ↔ -8/5 < a ∧ a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l1375_137545


namespace NUMINAMATH_CALUDE_average_time_to_find_waldo_l1375_137509

theorem average_time_to_find_waldo (num_books : ℕ) (puzzles_per_book : ℕ) (total_time : ℕ) :
  num_books = 15 →
  puzzles_per_book = 30 →
  total_time = 1350 →
  (total_time : ℚ) / (num_books * puzzles_per_book : ℚ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_average_time_to_find_waldo_l1375_137509


namespace NUMINAMATH_CALUDE_square_division_exists_l1375_137512

-- Define a rectangle
structure Rectangle where
  width : ℚ
  height : ℚ

-- Define a function to check if all numbers in a list are distinct
def allDistinct (list : List ℚ) : Prop :=
  ∀ i j, i ≠ j → list.get! i ≠ list.get! j

-- State the theorem
theorem square_division_exists : ∃ (rectangles : List Rectangle),
  -- There are 5 rectangles
  rectangles.length = 5 ∧
  -- The sum of areas equals 1
  (rectangles.map (λ r => r.width * r.height)).sum = 1 ∧
  -- All widths and heights are distinct
  allDistinct (rectangles.map (λ r => r.width) ++ rectangles.map (λ r => r.height)) :=
sorry

end NUMINAMATH_CALUDE_square_division_exists_l1375_137512


namespace NUMINAMATH_CALUDE_sally_balloons_l1375_137564

/-- The number of orange balloons Sally has after losing some -/
def remaining_balloons (initial : ℕ) (lost : ℕ) : ℕ := initial - lost

/-- Proof that Sally has 7 orange balloons after losing 2 from her initial 9 -/
theorem sally_balloons : remaining_balloons 9 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_sally_balloons_l1375_137564


namespace NUMINAMATH_CALUDE_problem_solution_l1375_137565

theorem problem_solution (x : ℝ) (h : x^2 - 5*x = 14) :
  (x - 1) * (2*x - 1) - (x + 1)^2 + 1 = 15 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1375_137565


namespace NUMINAMATH_CALUDE_people_at_game_l1375_137583

/-- The number of people who came to a little league game -/
theorem people_at_game (total_seats empty_seats : ℕ) 
  (h1 : total_seats = 92)
  (h2 : empty_seats = 45) :
  total_seats - empty_seats = 47 := by
  sorry

end NUMINAMATH_CALUDE_people_at_game_l1375_137583


namespace NUMINAMATH_CALUDE_stratified_sampling_result_l1375_137515

/-- Calculates the number of high school students selected in a stratified sampling -/
def stratified_sampling (total_students : ℕ) (selected_students : ℕ) (high_school_students : ℕ) : ℕ :=
  (high_school_students * selected_students) / total_students

/-- Theorem: In a stratified sampling of 15 students from 165 students, 
    where 66 are high school students, 6 high school students will be selected -/
theorem stratified_sampling_result : 
  stratified_sampling 165 15 66 = 6 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_result_l1375_137515
