import Mathlib

namespace NUMINAMATH_CALUDE_sine_of_vertex_angle_is_four_fifths_l3496_349614

/-- An isosceles triangle with a special property regarding inscribed rectangles. -/
structure SpecialIsoscelesTriangle where
  -- The vertex angle of the isosceles triangle
  vertex_angle : ℝ
  -- A function that takes two real numbers (representing the sides of a rectangle)
  -- and returns whether that rectangle can be inscribed in the triangle
  is_inscribable : ℝ → ℝ → Prop
  -- The constant perimeter of inscribable rectangles
  constant_perimeter : ℝ
  -- Property: The triangle is isosceles
  is_isosceles : Prop
  -- Property: Any rectangle that can be inscribed has the constant perimeter
  perimeter_is_constant : ∀ x y, is_inscribable x y → x + y = constant_perimeter

/-- 
The main theorem: In a special isosceles triangle where all inscribable rectangles 
have a constant perimeter, the sine of the vertex angle is 4/5.
-/
theorem sine_of_vertex_angle_is_four_fifths (t : SpecialIsoscelesTriangle) : 
  Real.sin t.vertex_angle = 4/5 := by
  sorry


end NUMINAMATH_CALUDE_sine_of_vertex_angle_is_four_fifths_l3496_349614


namespace NUMINAMATH_CALUDE_coffee_shop_lattes_l3496_349677

/-- The number of teas sold -/
def T : ℕ := 6

/-- The number of lattes sold -/
def L : ℕ := 4 * T + 8

theorem coffee_shop_lattes : L = 32 := by
  sorry

end NUMINAMATH_CALUDE_coffee_shop_lattes_l3496_349677


namespace NUMINAMATH_CALUDE_remainder_70_div_17_l3496_349646

theorem remainder_70_div_17 : 70 % 17 = 2 := by sorry

end NUMINAMATH_CALUDE_remainder_70_div_17_l3496_349646


namespace NUMINAMATH_CALUDE_boys_not_adjacent_or_ends_probability_l3496_349603

/-- The number of boys in the lineup -/
def num_boys : ℕ := 2

/-- The number of girls in the lineup -/
def num_girls : ℕ := 4

/-- The total number of people in the lineup -/
def total_people : ℕ := num_boys + num_girls

/-- The number of spaces between girls where boys can be placed -/
def available_spaces : ℕ := num_girls - 1

/-- The probability that the boys are neither adjacent nor at the ends in a lineup of boys and girls -/
theorem boys_not_adjacent_or_ends_probability :
  (num_boys.factorial * num_girls.factorial * available_spaces.choose num_boys) / total_people.factorial = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_boys_not_adjacent_or_ends_probability_l3496_349603


namespace NUMINAMATH_CALUDE_mitchell_has_30_pencils_l3496_349637

/-- The number of pencils Antonio has -/
def antonio_pencils : ℕ := sorry

/-- The number of pencils Mitchell has -/
def mitchell_pencils : ℕ := antonio_pencils + 6

/-- The total number of pencils Mitchell and Antonio have together -/
def total_pencils : ℕ := 54

theorem mitchell_has_30_pencils :
  mitchell_pencils = 30 :=
by
  sorry

#check mitchell_has_30_pencils

end NUMINAMATH_CALUDE_mitchell_has_30_pencils_l3496_349637


namespace NUMINAMATH_CALUDE_sum_of_squares_l3496_349670

theorem sum_of_squares : 17^2 + 19^2 + 23^2 + 29^2 = 2020 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l3496_349670


namespace NUMINAMATH_CALUDE_omelet_problem_l3496_349659

/-- The number of people that can be served omelets given the conditions -/
def number_of_people (eggs_per_dozen : ℕ) (eggs_per_omelet : ℕ) (omelets_per_person : ℕ) : ℕ :=
  let total_eggs := 3 * eggs_per_dozen
  let total_omelets := total_eggs / eggs_per_omelet
  total_omelets / omelets_per_person

/-- Theorem stating that under the given conditions, the number of people is 3 -/
theorem omelet_problem : number_of_people 12 4 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_omelet_problem_l3496_349659


namespace NUMINAMATH_CALUDE_ab_length_is_two_l3496_349633

/-- Represents a point on a line --/
structure Point where
  position : ℝ

/-- Represents the distance between two points --/
def distance (p q : Point) : ℝ := abs (p.position - q.position)

/-- Theorem: Given points A, B, C, D on a line in order, if AC = 5, BD = 6, and CD = 3, then AB = 2 --/
theorem ab_length_is_two 
  (A B C D : Point) 
  (order : A.position < B.position ∧ B.position < C.position ∧ C.position < D.position)
  (ac_length : distance A C = 5)
  (bd_length : distance B D = 6)
  (cd_length : distance C D = 3) :
  distance A B = 2 := by
  sorry

end NUMINAMATH_CALUDE_ab_length_is_two_l3496_349633


namespace NUMINAMATH_CALUDE_even_function_implies_m_eq_neg_one_l3496_349689

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The quadratic function f(x) = (m - 1)x² - (m² - 1)x + m + 2 -/
def f (m : ℝ) (x : ℝ) : ℝ :=
  (m - 1) * x^2 - (m^2 - 1) * x + m + 2

theorem even_function_implies_m_eq_neg_one :
  ∀ m : ℝ, EvenFunction (f m) → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_m_eq_neg_one_l3496_349689


namespace NUMINAMATH_CALUDE_rod_length_for_given_weight_l3496_349610

/-- Represents the properties of a uniform rod -/
structure UniformRod where
  length_kg_ratio : ℝ  -- Ratio of length to weight

/-- Calculates the length of a uniform rod given its weight -/
def rod_length (rod : UniformRod) (weight : ℝ) : ℝ :=
  weight * rod.length_kg_ratio

theorem rod_length_for_given_weight 
  (rod : UniformRod) 
  (h1 : rod_length rod 42.75 = 11.25)
  (h2 : rod.length_kg_ratio = 11.25 / 42.75) : 
  rod_length rod 26.6 = 7 := by
  sorry

#check rod_length_for_given_weight

end NUMINAMATH_CALUDE_rod_length_for_given_weight_l3496_349610


namespace NUMINAMATH_CALUDE_right_triangle_angle_bisector_l3496_349627

theorem right_triangle_angle_bisector (DE DF : ℝ) (h_DE : DE = 13) (h_DF : DF = 5) : ∃ XY₁ : ℝ, XY₁ = (10 * Real.sqrt 6) / 17 := by
  sorry


end NUMINAMATH_CALUDE_right_triangle_angle_bisector_l3496_349627


namespace NUMINAMATH_CALUDE_homework_difference_l3496_349623

theorem homework_difference (math reading history science : ℕ) 
  (h_math : math = 5)
  (h_reading : reading = 7)
  (h_history : history = 3)
  (h_science : science = 6) :
  (reading - math) + (science - history) = 5 :=
by sorry

end NUMINAMATH_CALUDE_homework_difference_l3496_349623


namespace NUMINAMATH_CALUDE_b_reaches_a_in_120_minutes_l3496_349624

/-- Represents the walking scenario of two people A and B -/
structure WalkingScenario where
  speed_B : ℝ  -- B's speed in meters per minute
  initial_distance : ℝ  -- Initial distance between A and B in meters
  meeting_time : ℝ  -- Time when A and B meet in minutes

/-- Calculates the time for B to reach point A after A has reached point B -/
def time_for_B_to_reach_A (scenario : WalkingScenario) : ℝ :=
  -- We'll implement the calculation here
  sorry

/-- Theorem stating that given the conditions, B will take 120 minutes to reach A after A reaches B -/
theorem b_reaches_a_in_120_minutes (scenario : WalkingScenario) 
    (h1 : scenario.meeting_time = 60)
    (h2 : scenario.initial_distance = 4 * scenario.speed_B * scenario.meeting_time) : 
    time_for_B_to_reach_A scenario = 120 :=
  sorry

end NUMINAMATH_CALUDE_b_reaches_a_in_120_minutes_l3496_349624


namespace NUMINAMATH_CALUDE_sin_2alpha_minus_cos_squared_alpha_l3496_349636

theorem sin_2alpha_minus_cos_squared_alpha (α : Real) (h : Real.tan α = 2) :
  Real.sin (2 * α) - Real.cos α ^ 2 = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_minus_cos_squared_alpha_l3496_349636


namespace NUMINAMATH_CALUDE_combined_weight_in_pounds_l3496_349640

-- Define the weight of the elephant in tons
def elephant_weight_tons : ℝ := 3

-- Define the conversion factor from tons to pounds
def tons_to_pounds : ℝ := 2000

-- Define the weight ratio of the donkey compared to the elephant
def donkey_weight_ratio : ℝ := 0.1

-- Theorem statement
theorem combined_weight_in_pounds :
  let elephant_weight_pounds := elephant_weight_tons * tons_to_pounds
  let donkey_weight_pounds := elephant_weight_pounds * donkey_weight_ratio
  elephant_weight_pounds + donkey_weight_pounds = 6600 := by
sorry

end NUMINAMATH_CALUDE_combined_weight_in_pounds_l3496_349640


namespace NUMINAMATH_CALUDE_inverse_function_solution_l3496_349628

/-- Given a function g(x) = 1 / (2ax + b), where a and b are non-zero constants and a ≠ b,
    prove that the solution to g^(-1)(x) = 0 is x = 1/b. -/
theorem inverse_function_solution (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a ≠ b) :
  let g : ℝ → ℝ := λ x ↦ 1 / (2 * a * x + b)
  ∃! x, Function.invFun g x = 0 ∧ x = 1 / b :=
by sorry

end NUMINAMATH_CALUDE_inverse_function_solution_l3496_349628


namespace NUMINAMATH_CALUDE_student_marks_l3496_349602

theorem student_marks (M P C : ℤ) 
  (h1 : C = P + 20) 
  (h2 : (M + C) / 2 = 45) : 
  M + P = 70 := by
  sorry

end NUMINAMATH_CALUDE_student_marks_l3496_349602


namespace NUMINAMATH_CALUDE_no_real_solutions_l3496_349642

theorem no_real_solutions : ¬∃ (x : ℝ), (3*x + 6) / (x^2 + 5*x - 6) = (3 - x) / (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l3496_349642


namespace NUMINAMATH_CALUDE_absolute_value_and_quadratic_equivalence_l3496_349604

theorem absolute_value_and_quadratic_equivalence :
  ∀ (b c : ℝ),
    (∀ x : ℝ, |x - 8| = 3 ↔ x^2 + b*x + c = 0) ↔
    (b = -16 ∧ c = 55) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_and_quadratic_equivalence_l3496_349604


namespace NUMINAMATH_CALUDE_dividend_calculation_l3496_349694

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 21) 
  (h2 : quotient = 14) 
  (h3 : remainder = 7) : 
  divisor * quotient + remainder = 301 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l3496_349694


namespace NUMINAMATH_CALUDE_jason_seashells_l3496_349681

/-- Given that Jason initially had 49 seashells and gave away 13 seashells,
    prove that he now has 36 seashells. -/
theorem jason_seashells (initial : ℕ) (given_away : ℕ) (remaining : ℕ) 
    (h1 : initial = 49)
    (h2 : given_away = 13)
    (h3 : remaining = initial - given_away) :
  remaining = 36 := by
  sorry

end NUMINAMATH_CALUDE_jason_seashells_l3496_349681


namespace NUMINAMATH_CALUDE_unique_x_value_l3496_349609

theorem unique_x_value (x : ℝ) : x^2 ∈ ({0, 1, x} : Set ℝ) → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_unique_x_value_l3496_349609


namespace NUMINAMATH_CALUDE_max_sides_equal_longest_diagonal_l3496_349699

/-- A convex polygon is a polygon where all interior angles are less than or equal to 180 degrees. -/
def ConvexPolygon (P : Set (ℝ × ℝ)) : Prop := sorry

/-- The longest diagonal of a polygon is the longest line segment connecting any two non-adjacent vertices. -/
def LongestDiagonal (P : Set (ℝ × ℝ)) : ℝ := sorry

/-- A side of a polygon is a line segment connecting two adjacent vertices. -/
def Side (P : Set (ℝ × ℝ)) (s : ℝ × ℝ → ℝ × ℝ → Prop) : Prop := sorry

/-- Count of sides equal to the longest diagonal -/
def CountSidesEqualToLongestDiagonal (P : Set (ℝ × ℝ)) : ℕ := sorry

/-- An equilateral triangle is a triangle with all sides of equal length. -/
def EquilateralTriangle (P : Set (ℝ × ℝ)) : Prop := sorry

theorem max_sides_equal_longest_diagonal 
  (P : Set (ℝ × ℝ)) 
  (h_convex : ConvexPolygon P) :
  (CountSidesEqualToLongestDiagonal P ≤ 2) ∨ 
  (CountSidesEqualToLongestDiagonal P = 3 ∧ EquilateralTriangle P) := by
  sorry

end NUMINAMATH_CALUDE_max_sides_equal_longest_diagonal_l3496_349699


namespace NUMINAMATH_CALUDE_ascending_concept_chain_l3496_349621

-- Define the concept hierarchy
def IsNatural (n : ℕ) : Prop := True
def IsInteger (n : ℤ) : Prop := True
def IsRational (q : ℚ) : Prop := True
def IsReal (r : ℝ) : Prop := True
def IsNumber (x : ℝ) : Prop := True

-- Define the chain of ascending concepts
def ConceptChain : Prop :=
  ∃ (n : ℕ) (z : ℤ) (q : ℚ) (r : ℝ),
    n = 3 ∧
    IsNatural n ∧
    (↑n : ℤ) = z ∧
    IsInteger z ∧
    (↑z : ℚ) = q ∧
    IsRational q ∧
    (↑q : ℝ) = r ∧
    IsReal r ∧
    IsNumber r

-- Theorem statement
theorem ascending_concept_chain : ConceptChain :=
  sorry

end NUMINAMATH_CALUDE_ascending_concept_chain_l3496_349621


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3496_349619

theorem sufficient_not_necessary (x y : ℝ) :
  (∀ x y, x + y ≤ 1 → x ≤ 1/2 ∨ y ≤ 1/2) ∧
  (∃ x y, (x ≤ 1/2 ∨ y ≤ 1/2) ∧ x + y > 1) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3496_349619


namespace NUMINAMATH_CALUDE_tomatoes_for_family_of_eight_l3496_349684

/-- The number of tomatoes needed to feed a family for a single meal -/
def tomatoes_needed (slices_per_tomato : ℕ) (slices_per_person : ℕ) (family_size : ℕ) : ℕ :=
  (slices_per_person * family_size) / slices_per_tomato

theorem tomatoes_for_family_of_eight :
  tomatoes_needed 8 20 8 = 20 := by
  sorry

end NUMINAMATH_CALUDE_tomatoes_for_family_of_eight_l3496_349684


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3496_349605

theorem right_triangle_hypotenuse (x y h : ℝ) : 
  x > 0 → 
  y = 2 * x + 2 → 
  (1 / 2) * x * y = 72 → 
  x^2 + y^2 = h^2 → 
  h = Real.sqrt 388 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3496_349605


namespace NUMINAMATH_CALUDE_language_coverage_probability_l3496_349651

def total_students : ℕ := 40
def french_students : ℕ := 30
def spanish_students : ℕ := 32
def german_students : ℕ := 10
def german_and_other : ℕ := 26

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem language_coverage_probability :
  let french_and_spanish : ℕ := french_students + spanish_students - total_students
  let french_only : ℕ := french_students - french_and_spanish
  let spanish_only : ℕ := spanish_students - french_and_spanish
  let remaining : ℕ := total_students - (french_only + spanish_only + french_and_spanish)
  let total_combinations : ℕ := choose total_students 2
  let unfavorable_outcomes : ℕ := choose french_only 2 + choose spanish_only 2 + choose remaining 2
  (total_combinations - unfavorable_outcomes : ℚ) / total_combinations = 353 / 390 :=
sorry

end NUMINAMATH_CALUDE_language_coverage_probability_l3496_349651


namespace NUMINAMATH_CALUDE_card_problem_l3496_349682

theorem card_problem (x y : ℕ) : 
  x - 1 = y + 1 → 
  x + 1 = 2 * (y - 1) → 
  x + y = 12 := by
sorry

end NUMINAMATH_CALUDE_card_problem_l3496_349682


namespace NUMINAMATH_CALUDE_all_equal_cyclic_inequality_l3496_349691

theorem all_equal_cyclic_inequality (a : Fin 100 → ℝ) 
  (h : ∀ i : Fin 100, a i - 3 * a (i + 1) + 2 * a (i + 2) ≥ 0) :
  ∀ i j : Fin 100, a i = a j :=
sorry

end NUMINAMATH_CALUDE_all_equal_cyclic_inequality_l3496_349691


namespace NUMINAMATH_CALUDE_simplify_expression_l3496_349679

theorem simplify_expression (a b : ℝ) : a * b - (a^2 - a * b + b^2) = -a^2 + 2 * a * b - b^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3496_349679


namespace NUMINAMATH_CALUDE_prime_sum_theorem_l3496_349613

theorem prime_sum_theorem : ∃ (A B : ℕ), 
  0 < A ∧ 0 < B ∧
  Nat.Prime A ∧ 
  Nat.Prime B ∧ 
  Nat.Prime (A - B) ∧ 
  Nat.Prime (A - 2*B) ∧
  A + B + (A - B) + (A - 2*B) = 17 := by
sorry

end NUMINAMATH_CALUDE_prime_sum_theorem_l3496_349613


namespace NUMINAMATH_CALUDE_apple_pear_puzzle_l3496_349660

theorem apple_pear_puzzle (apples pears : ℕ) : 
  (apples : ℚ) / 3 = (pears : ℚ) / 2 + 1 →
  (apples : ℚ) / 5 = (pears : ℚ) / 4 - 3 →
  apples = 23 ∧ pears = 16 := by
sorry

end NUMINAMATH_CALUDE_apple_pear_puzzle_l3496_349660


namespace NUMINAMATH_CALUDE_minimum_at_two_implies_a_twelve_l3496_349645

/-- Given a function f(x) = x^3 - ax, prove that if f takes its minimum value at x = 2, then a = 12 -/
theorem minimum_at_two_implies_a_twelve (a : ℝ) : 
  (∀ x : ℝ, x^3 - a*x ≥ 2^3 - a*2) → a = 12 := by
  sorry

end NUMINAMATH_CALUDE_minimum_at_two_implies_a_twelve_l3496_349645


namespace NUMINAMATH_CALUDE_f_properties_l3496_349635

noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x - Real.cos x ^ 2

theorem f_properties :
  (∀ x, -2 ≤ f x ∧ f x ≤ 2) ∧
  (∀ p > 0, (∀ x, f (x + p) = f x) → p ≥ π) ∧
  (∀ x, f (x + π) = f x) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l3496_349635


namespace NUMINAMATH_CALUDE_parking_cost_proof_l3496_349674

-- Define the initial cost for up to 2 hours
def initial_cost : ℝ := 9

-- Define the total parking duration in hours
def total_hours : ℝ := 9

-- Define the average cost per hour for the total duration
def average_cost_per_hour : ℝ := 2.361111111111111

-- Define the cost for each hour in excess of 2 hours
def excess_hour_cost : ℝ := 1.75

-- Theorem statement
theorem parking_cost_proof :
  excess_hour_cost * (total_hours - 2) + initial_cost = average_cost_per_hour * total_hours :=
by sorry

end NUMINAMATH_CALUDE_parking_cost_proof_l3496_349674


namespace NUMINAMATH_CALUDE_astronaut_education_time_l3496_349630

theorem astronaut_education_time (total_time science_time : ℕ) : 
  total_time = 14 ∧ 
  total_time = science_time + 2 * science_time + 2 →
  science_time = 4 := by
sorry

end NUMINAMATH_CALUDE_astronaut_education_time_l3496_349630


namespace NUMINAMATH_CALUDE_solution_set_inequality_l3496_349641

theorem solution_set_inequality (f : ℝ → ℝ) (hf : ∀ x, f x + (deriv f) x > 1) (hf0 : f 0 = 4) :
  {x : ℝ | f x > 3 / Real.exp x + 1} = {x : ℝ | x > 0} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l3496_349641


namespace NUMINAMATH_CALUDE_cucumber_weight_after_evaporation_l3496_349650

/-- Calculates the new weight of cucumbers after water evaporation -/
theorem cucumber_weight_after_evaporation 
  (initial_weight : ℝ) 
  (initial_water_percent : ℝ) 
  (final_water_percent : ℝ) : 
  initial_weight = 100 → 
  initial_water_percent = 99 / 100 → 
  final_water_percent = 96 / 100 → 
  ∃ (new_weight : ℝ), 
    new_weight = 25 ∧ 
    (1 - initial_water_percent) * initial_weight = (1 - final_water_percent) * new_weight :=
by sorry

end NUMINAMATH_CALUDE_cucumber_weight_after_evaporation_l3496_349650


namespace NUMINAMATH_CALUDE_division_problem_l3496_349658

theorem division_problem (a b : ℕ) : 
  a > 0 ∧ b > 0 ∧ 
  a = 13 * b + 6 ∧ 
  a + b + 13 + 6 = 137 → 
  a = 110 ∧ b = 8 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l3496_349658


namespace NUMINAMATH_CALUDE_insurance_cost_over_decade_l3496_349666

/-- The amount spent on car insurance in a year -/
def yearly_insurance_cost : ℕ := 4000

/-- The number of years in a decade -/
def years_in_decade : ℕ := 10

/-- The total cost of car insurance over a decade -/
def decade_insurance_cost : ℕ := yearly_insurance_cost * years_in_decade

theorem insurance_cost_over_decade : 
  decade_insurance_cost = 40000 := by sorry

end NUMINAMATH_CALUDE_insurance_cost_over_decade_l3496_349666


namespace NUMINAMATH_CALUDE_cardinality_of_C_l3496_349661

def A : Finset ℕ := {0, 2, 3, 4, 5, 7}
def B : Finset ℕ := {1, 2, 3, 4, 6}
def C : Finset ℕ := A \ B

theorem cardinality_of_C : Finset.card C = 3 := by
  sorry

end NUMINAMATH_CALUDE_cardinality_of_C_l3496_349661


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3496_349654

def complex_i : ℂ := Complex.I

theorem complex_equation_solution (a : ℝ) :
  (2 : ℂ) / (a + complex_i) = 1 - complex_i → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3496_349654


namespace NUMINAMATH_CALUDE_individual_contribution_proof_l3496_349631

def total_contribution : ℝ := 90
def class_funds : ℝ := 30
def num_students : ℝ := 25

theorem individual_contribution_proof :
  (total_contribution - class_funds) / num_students = 2.40 :=
by sorry

end NUMINAMATH_CALUDE_individual_contribution_proof_l3496_349631


namespace NUMINAMATH_CALUDE_laborer_income_l3496_349606

/-- Represents the monthly income of a laborer -/
def monthly_income : ℝ := 69

/-- Represents the average expenditure for the first 6 months -/
def first_6_months_expenditure : ℝ := 70

/-- Represents the reduced monthly expenditure for the next 4 months -/
def next_4_months_expenditure : ℝ := 60

/-- Represents the amount saved after 10 months -/
def amount_saved : ℝ := 30

/-- Theorem stating that the monthly income is 69 given the problem conditions -/
theorem laborer_income : 
  (6 * first_6_months_expenditure > 6 * monthly_income) ∧ 
  (4 * monthly_income = 4 * next_4_months_expenditure + (6 * first_6_months_expenditure - 6 * monthly_income) + amount_saved) →
  monthly_income = 69 := by
sorry

end NUMINAMATH_CALUDE_laborer_income_l3496_349606


namespace NUMINAMATH_CALUDE_mango_profit_percentage_l3496_349625

theorem mango_profit_percentage 
  (total_crates : ℕ) 
  (total_cost : ℝ) 
  (lost_crates : ℕ) 
  (selling_price : ℝ) : 
  total_crates = 10 → 
  total_cost = 160 → 
  lost_crates = 2 → 
  selling_price = 25 → 
  ((total_crates - lost_crates) * selling_price - total_cost) / total_cost * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_mango_profit_percentage_l3496_349625


namespace NUMINAMATH_CALUDE_extreme_value_implies_b_l3496_349698

/-- Given a function f(x) = x³ + ax² + bx + a² where a and b are real numbers,
    if f(x) has an extreme value of 10 at x = 1, then b = -11 -/
theorem extreme_value_implies_b (a b : ℝ) : 
  let f := fun (x : ℝ) => x^3 + a*x^2 + b*x + a^2
  (∃ (ε : ℝ), ∀ (x : ℝ), x ≠ 1 → |x - 1| < ε → f x ≤ f 1) ∧ 
  (f 1 = 10) →
  b = -11 := by sorry

end NUMINAMATH_CALUDE_extreme_value_implies_b_l3496_349698


namespace NUMINAMATH_CALUDE_intersection_A_B_l3496_349639

-- Define set A
def A : Set ℝ := {x | (x + 1) * (4 - x) > 0}

-- Define set B
def B : Set ℝ := {x | 0 < x ∧ x < 9}

-- Define the open interval (0, 4)
def open_interval_0_4 : Set ℝ := {x | 0 < x ∧ x < 4}

-- Theorem statement
theorem intersection_A_B : A ∩ B = open_interval_0_4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3496_349639


namespace NUMINAMATH_CALUDE_circle_and_symmetry_l3496_349675

-- Define the circle
def Circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}

-- Define the chord line
def ChordLine (x : ℝ) := x + 1

-- Define the variable line
def VariableLine (k : ℝ) (x : ℝ) := k * (x - 1)

-- Define the fixed point N
def N : ℝ × ℝ := (4, 0)

-- Main theorem
theorem circle_and_symmetry :
  -- The chord intercepted by y = x + 1 has length √14
  (∃ (a b : ℝ), (a, ChordLine a) ∈ Circle ∧ (b, ChordLine b) ∈ Circle ∧ (b - a)^2 + (ChordLine b - ChordLine a)^2 = 14) →
  -- The equation of the circle is x² + y² = 4
  (∀ (x y : ℝ), (x, y) ∈ Circle ↔ x^2 + y^2 = 4) ∧
  -- N is the fixed point of symmetry
  (∀ (k : ℝ) (A B : ℝ × ℝ),
    k ≠ 0 →
    A ∈ Circle →
    B ∈ Circle →
    A.2 = VariableLine k A.1 →
    B.2 = VariableLine k B.1 →
    (A.2 / (A.1 - N.1) + B.2 / (B.1 - N.1) = 0)) :=
by sorry


end NUMINAMATH_CALUDE_circle_and_symmetry_l3496_349675


namespace NUMINAMATH_CALUDE_inequality_proof_l3496_349667

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : 0 > c) :
  (c - a < c - b) ∧ (a⁻¹ * c > b⁻¹ * c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3496_349667


namespace NUMINAMATH_CALUDE_rectangle_cover_cost_l3496_349649

/-- Given a rectangle where the length is four times the width and the perimeter is 200 cm,
    the total cost to cover the rectangle at $5 per square centimeter is $8000. -/
theorem rectangle_cover_cost (w l : ℝ) (h1 : l = 4 * w) (h2 : 2 * l + 2 * w = 200) :
  5 * (l * w) = 8000 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_cover_cost_l3496_349649


namespace NUMINAMATH_CALUDE_pens_sold_correct_solve_paul_pens_problem_l3496_349697

/-- Calculates the number of pens sold given the initial and final counts -/
def pens_sold (initial : ℕ) (final : ℕ) : ℕ :=
  initial - final

theorem pens_sold_correct (initial final : ℕ) (h : initial ≥ final) :
  pens_sold initial final = initial - final :=
by sorry

/-- The specific problem instance -/
def paul_pens_problem : Prop :=
  pens_sold 106 14 = 92

theorem solve_paul_pens_problem : paul_pens_problem :=
by sorry

end NUMINAMATH_CALUDE_pens_sold_correct_solve_paul_pens_problem_l3496_349697


namespace NUMINAMATH_CALUDE_rug_coverage_theorem_l3496_349672

/-- The total floor area covered by three overlapping rugs -/
def totalFloorArea (combinedArea twoLayerArea threeLayerArea : ℝ) : ℝ :=
  combinedArea - (twoLayerArea + 2 * threeLayerArea)

/-- Theorem stating that the total floor area covered by the rugs is 140 square meters -/
theorem rug_coverage_theorem :
  totalFloorArea 200 22 19 = 140 := by
  sorry

end NUMINAMATH_CALUDE_rug_coverage_theorem_l3496_349672


namespace NUMINAMATH_CALUDE_f_12345_equals_12345_l3496_349662

/-- The set of all non-zero real-valued functions satisfying the given functional equation. -/
def S : Set (ℝ → ℝ) :=
  {f | ∀ x y z : ℝ, f ≠ 0 ∧ f (x^2 + y * f z) = x * f x + z * f y}

/-- Theorem stating that for any function in S, f(12345) = 12345. -/
theorem f_12345_equals_12345 (f : ℝ → ℝ) (hf : f ∈ S) : f 12345 = 12345 := by
  sorry

end NUMINAMATH_CALUDE_f_12345_equals_12345_l3496_349662


namespace NUMINAMATH_CALUDE_distance_to_line_l3496_349692

/-- Given two perpendicular lines and a point, prove the distance to a third line -/
theorem distance_to_line (m : ℝ) : 
  (∀ x y, 2*x + y - 2 = 0 → x + m*y - 1 = 0 → (2 : ℝ) * (-1/m) = -1) →
  let P := (m, m)
  (abs (P.1 + P.2 + 3) / Real.sqrt 2 : ℝ) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_line_l3496_349692


namespace NUMINAMATH_CALUDE_scarlet_savings_l3496_349668

theorem scarlet_savings (initial_savings : ℕ) (earrings_cost : ℕ) (necklace_cost : ℕ) : 
  initial_savings = 80 → earrings_cost = 23 → necklace_cost = 48 → 
  initial_savings - (earrings_cost + necklace_cost) = 9 := by
sorry

end NUMINAMATH_CALUDE_scarlet_savings_l3496_349668


namespace NUMINAMATH_CALUDE_carpet_square_size_l3496_349601

theorem carpet_square_size (floor_length : ℝ) (floor_width : ℝ) 
  (total_cost : ℝ) (square_cost : ℝ) :
  floor_length = 24 →
  floor_width = 64 →
  total_cost = 576 →
  square_cost = 24 →
  ∃ (square_side : ℝ),
    square_side = 8 ∧
    (floor_length * floor_width) / (square_side * square_side) * square_cost = total_cost :=
by sorry

end NUMINAMATH_CALUDE_carpet_square_size_l3496_349601


namespace NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l3496_349616

theorem condition_sufficient_not_necessary :
  (∀ x : ℝ, x > 1 → (x + 2) / (x - 1) > 0) ∧
  (∃ x : ℝ, x ≤ 1 ∧ (x + 2) / (x - 1) > 0) := by
  sorry

end NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l3496_349616


namespace NUMINAMATH_CALUDE_steps_per_level_l3496_349638

theorem steps_per_level (total_blocks : ℕ) (blocks_per_step : ℕ) (levels : ℕ) : 
  total_blocks = 96 → blocks_per_step = 3 → levels = 4 → 
  (total_blocks / blocks_per_step) / levels = 8 := by
  sorry

end NUMINAMATH_CALUDE_steps_per_level_l3496_349638


namespace NUMINAMATH_CALUDE_additional_triangles_for_hexagon_l3496_349685

/-- The number of vertices in a hexagon -/
def hexagon_vertices : ℕ := 6

/-- The number of sides in a hexagon -/
def hexagon_sides : ℕ := 6

/-- The number of triangles in the original shape -/
def original_triangles : ℕ := 36

/-- The number of additional triangles needed for each vertex -/
def triangles_per_vertex : ℕ := 2

/-- The number of additional triangles needed for each side -/
def triangles_per_side : ℕ := 1

/-- The theorem stating the smallest number of additional triangles needed -/
theorem additional_triangles_for_hexagon :
  hexagon_vertices * triangles_per_vertex + hexagon_sides * triangles_per_side = 18 :=
sorry

end NUMINAMATH_CALUDE_additional_triangles_for_hexagon_l3496_349685


namespace NUMINAMATH_CALUDE_total_harvest_l3496_349608

def tomato_harvest (day1 : ℕ) (extra_day2 : ℕ) : ℕ := 
  let day2 := day1 + extra_day2
  let day3 := 2 * day2
  day1 + day2 + day3

theorem total_harvest : tomato_harvest 120 50 = 630 := by
  sorry

end NUMINAMATH_CALUDE_total_harvest_l3496_349608


namespace NUMINAMATH_CALUDE_min_value_sum_squares_and_reciprocals_l3496_349612

theorem min_value_sum_squares_and_reciprocals (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  a^2 + b^2 + 1/a^2 + 1/b^2 ≥ 4 ∧
  (a^2 + b^2 + 1/a^2 + 1/b^2 = 4 ↔ a = 1 ∧ b = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_squares_and_reciprocals_l3496_349612


namespace NUMINAMATH_CALUDE_holly_throws_five_times_l3496_349665

/-- Represents the Frisbee throwing scenario -/
structure FrisbeeScenario where
  bess_throw_distance : ℕ
  bess_throw_count : ℕ
  holly_throw_distance : ℕ
  total_distance : ℕ

/-- Calculates the number of times Holly throws the Frisbee -/
def holly_throw_count (scenario : FrisbeeScenario) : ℕ :=
  (scenario.total_distance - 2 * scenario.bess_throw_distance * scenario.bess_throw_count) / scenario.holly_throw_distance

/-- Theorem stating that Holly throws the Frisbee 5 times in the given scenario -/
theorem holly_throws_five_times (scenario : FrisbeeScenario) 
  (h1 : scenario.bess_throw_distance = 20)
  (h2 : scenario.bess_throw_count = 4)
  (h3 : scenario.holly_throw_distance = 8)
  (h4 : scenario.total_distance = 200) :
  holly_throw_count scenario = 5 := by
  sorry

#eval holly_throw_count { bess_throw_distance := 20, bess_throw_count := 4, holly_throw_distance := 8, total_distance := 200 }

end NUMINAMATH_CALUDE_holly_throws_five_times_l3496_349665


namespace NUMINAMATH_CALUDE_partnership_investment_l3496_349652

/-- Represents a partnership with three partners -/
structure Partnership where
  investmentA : ℝ
  investmentB : ℝ
  investmentC : ℝ
  totalProfit : ℝ
  cProfit : ℝ

/-- Calculates the total investment of the partnership -/
def totalInvestment (p : Partnership) : ℝ :=
  p.investmentA + p.investmentB + p.investmentC

/-- Theorem stating that if the given conditions are met, 
    then Partner C's investment is 36000 -/
theorem partnership_investment 
  (p : Partnership) 
  (h1 : p.investmentA = 24000)
  (h2 : p.investmentB = 32000)
  (h3 : p.totalProfit = 92000)
  (h4 : p.cProfit = 36000)
  (h5 : p.cProfit / p.totalProfit = p.investmentC / totalInvestment p) :
  p.investmentC = 36000 :=
sorry

end NUMINAMATH_CALUDE_partnership_investment_l3496_349652


namespace NUMINAMATH_CALUDE_cauchy_equation_solution_l3496_349655

theorem cauchy_equation_solution (f : ℝ → ℝ) 
  (h_cauchy : ∀ x y : ℝ, f (x + y) = f x + f y) 
  (h_condition : f 1 ^ 2 = f 1) :
  (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = x) := by
  sorry

end NUMINAMATH_CALUDE_cauchy_equation_solution_l3496_349655


namespace NUMINAMATH_CALUDE_remaining_quantity_average_l3496_349673

theorem remaining_quantity_average (total : ℕ) (avg_all : ℚ) (avg_five : ℚ) (avg_two : ℚ) :
  total = 8 ∧ avg_all = 15 ∧ avg_five = 10 ∧ avg_two = 22 →
  (total * avg_all - 5 * avg_five - 2 * avg_two) = 26 := by
sorry

end NUMINAMATH_CALUDE_remaining_quantity_average_l3496_349673


namespace NUMINAMATH_CALUDE_license_plate_combinations_l3496_349648

def num_consonants : ℕ := 21
def num_vowels : ℕ := 6
def num_digits : ℕ := 10
def num_special_chars : ℕ := 3

theorem license_plate_combinations : 
  num_consonants * num_vowels * num_consonants * num_digits * num_special_chars = 79380 :=
by sorry

end NUMINAMATH_CALUDE_license_plate_combinations_l3496_349648


namespace NUMINAMATH_CALUDE_probability_A1_or_B1_not_both_l3496_349615

def excellent_math : ℕ := 3
def excellent_physics : ℕ := 2
def excellent_chemistry : ℕ := 2

def total_combinations : ℕ := excellent_math * excellent_physics * excellent_chemistry

def favorable_outcomes : ℕ := 
  excellent_physics * excellent_chemistry + 
  (excellent_math - 1) * excellent_chemistry

theorem probability_A1_or_B1_not_both : 
  (favorable_outcomes : ℚ) / total_combinations = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_probability_A1_or_B1_not_both_l3496_349615


namespace NUMINAMATH_CALUDE_same_color_sock_pairs_l3496_349676

def total_socks : ℕ := 12
def red_socks : ℕ := 5
def green_socks : ℕ := 3
def blue_socks : ℕ := 4

theorem same_color_sock_pairs :
  (Nat.choose red_socks 2) + (Nat.choose green_socks 2) + (Nat.choose blue_socks 2) = 19 := by
  sorry

end NUMINAMATH_CALUDE_same_color_sock_pairs_l3496_349676


namespace NUMINAMATH_CALUDE_sixtieth_pair_is_five_seven_l3496_349653

/-- Definition of the sequence of pairs -/
def pair_sequence : ℕ → ℕ × ℕ
| n => let group := (n + 1).sqrt
       let position := n - (group * (group - 1)) / 2
       (position, group + 1 - position)

/-- Theorem stating that the 60th pair is (5, 7) -/
theorem sixtieth_pair_is_five_seven :
  pair_sequence 60 = (5, 7) := by
  sorry


end NUMINAMATH_CALUDE_sixtieth_pair_is_five_seven_l3496_349653


namespace NUMINAMATH_CALUDE_exam_pass_percentage_l3496_349622

theorem exam_pass_percentage
  (failed_hindi : ℚ)
  (failed_english : ℚ)
  (failed_both : ℚ)
  (h1 : failed_hindi = 25 / 100)
  (h2 : failed_english = 48 / 100)
  (h3 : failed_both = 27 / 100) :
  1 - (failed_hindi + failed_english - failed_both) = 54 / 100 :=
by sorry

end NUMINAMATH_CALUDE_exam_pass_percentage_l3496_349622


namespace NUMINAMATH_CALUDE_jellybean_problem_l3496_349671

theorem jellybean_problem (x : ℕ) : 
  x ≥ 150 ∧ 
  x % 15 = 14 ∧ 
  x % 17 = 16 ∧ 
  (∀ y : ℕ, y ≥ 150 ∧ y % 15 = 14 ∧ y % 17 = 16 → x ≤ y) → 
  x = 254 := by
sorry

end NUMINAMATH_CALUDE_jellybean_problem_l3496_349671


namespace NUMINAMATH_CALUDE_isabelle_ticket_cost_l3496_349695

def brothers_ticket_cost : ℕ := 20
def brothers_savings : ℕ := 5
def isabelle_savings : ℕ := 5
def isabelle_earnings : ℕ := 30

def total_amount_needed : ℕ := isabelle_earnings + isabelle_savings + brothers_savings

theorem isabelle_ticket_cost :
  total_amount_needed - brothers_ticket_cost = 15 := by sorry

end NUMINAMATH_CALUDE_isabelle_ticket_cost_l3496_349695


namespace NUMINAMATH_CALUDE_percentage_difference_l3496_349607

theorem percentage_difference (x y : ℝ) (h : x = 5 * y) : 
  (x - y) / x * 100 = 80 := by
sorry

end NUMINAMATH_CALUDE_percentage_difference_l3496_349607


namespace NUMINAMATH_CALUDE_boys_not_in_varsity_clubs_l3496_349680

theorem boys_not_in_varsity_clubs (total_students : ℕ) (girls_percentage : ℚ) (boys_in_clubs_fraction : ℚ) :
  total_students = 150 →
  girls_percentage = 60 / 100 →
  boys_in_clubs_fraction = 1 / 3 →
  (total_students : ℚ) * (1 - girls_percentage) * (1 - boys_in_clubs_fraction) = 40 :=
by sorry

end NUMINAMATH_CALUDE_boys_not_in_varsity_clubs_l3496_349680


namespace NUMINAMATH_CALUDE_triangle_angle_identity_l3496_349663

theorem triangle_angle_identity (α : Real) 
  (h1 : 0 < α ∧ α < Real.pi)  -- α is an internal angle of a triangle
  (h2 : Real.sin α * Real.cos α = 1/8) :  -- given condition
  Real.cos α + Real.sin α = Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_identity_l3496_349663


namespace NUMINAMATH_CALUDE_distinct_paths_6x4_l3496_349644

/-- The number of rows in the grid -/
def rows : ℕ := 4

/-- The number of columns in the grid -/
def cols : ℕ := 6

/-- The total number of steps needed to reach from top-left to bottom-right -/
def total_steps : ℕ := rows + cols - 2

/-- The number of down steps needed -/
def down_steps : ℕ := rows - 1

/-- The number of distinct paths from top-left to bottom-right in a 6x4 grid -/
theorem distinct_paths_6x4 : Nat.choose total_steps down_steps = 56 := by
  sorry

end NUMINAMATH_CALUDE_distinct_paths_6x4_l3496_349644


namespace NUMINAMATH_CALUDE_dagger_operation_result_l3496_349618

-- Define the † operation
def dagger (m n p q : ℚ) : ℚ := m * p * (q / n)

-- Theorem statement
theorem dagger_operation_result :
  let result := dagger (5/9) (6/4)
  (result + 1/6) = 27/2 := by
  sorry

end NUMINAMATH_CALUDE_dagger_operation_result_l3496_349618


namespace NUMINAMATH_CALUDE_dice_prime_probability_l3496_349632

def probability_prime : ℚ := 5 / 12

def number_of_dice : ℕ := 5

def target_prime_count : ℕ := 3

theorem dice_prime_probability :
  let p := probability_prime
  let n := number_of_dice
  let k := target_prime_count
  (n.choose k) * p^k * (1 - p)^(n - k) = 6125 / 24883 := by sorry

end NUMINAMATH_CALUDE_dice_prime_probability_l3496_349632


namespace NUMINAMATH_CALUDE_base_difference_not_divisible_by_three_l3496_349643

/-- Converts a number from base b to base 10 -/
def toBase10 (digits : List Nat) (b : Nat) : Nat :=
  digits.foldr (fun d acc => d + b * acc) 0

/-- The difference of (2021)_b and (221)_b in base 10 -/
def baseDifference (b : Nat) : Nat :=
  toBase10 [2, 0, 2, 1] b - toBase10 [2, 2, 1] b

theorem base_difference_not_divisible_by_three (b : Nat) :
  b > 0 → (baseDifference b % 3 ≠ 0 ↔ b % 3 = 2) := by
  sorry

end NUMINAMATH_CALUDE_base_difference_not_divisible_by_three_l3496_349643


namespace NUMINAMATH_CALUDE_simplify_square_roots_l3496_349600

theorem simplify_square_roots : 81^(1/2) - 144^(1/2) = -63 := by sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l3496_349600


namespace NUMINAMATH_CALUDE_marble_arrangement_theorem_l3496_349693

/-- Represents the colors of marbles --/
inductive Color
| Green
| Red

/-- Represents a circular arrangement of marbles --/
def CircularArrangement := List Color

/-- Counts the number of marbles with same-color neighbors --/
def countSameColorNeighbors (arrangement : CircularArrangement) : Nat :=
  sorry

/-- Counts the number of marbles with different-color neighbors --/
def countDifferentColorNeighbors (arrangement : CircularArrangement) : Nat :=
  sorry

/-- Checks if an arrangement satisfies the neighbor color condition --/
def isValidArrangement (arrangement : CircularArrangement) : Prop :=
  countSameColorNeighbors arrangement = countDifferentColorNeighbors arrangement

/-- Counts the number of valid arrangements --/
def countValidArrangements (greenMarbles redMarbles : Nat) : Nat :=
  sorry

/-- The main theorem --/
theorem marble_arrangement_theorem :
  let greenMarbles : Nat := 7
  let maxRedMarbles : Nat := 14
  (countValidArrangements greenMarbles maxRedMarbles) % 1000 = 432 :=
sorry

end NUMINAMATH_CALUDE_marble_arrangement_theorem_l3496_349693


namespace NUMINAMATH_CALUDE_inequality_proof_l3496_349657

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (a^3 / (a^3 + 2*b^2)) + (b^3 / (b^3 + 2*c^2)) + (c^3 / (c^3 + 2*a^2)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3496_349657


namespace NUMINAMATH_CALUDE_vector_calculation_l3496_349686

def a : Fin 3 → ℝ := ![(-3 : ℝ), 5, 2]
def b : Fin 3 → ℝ := ![(6 : ℝ), -1, -3]
def c : Fin 3 → ℝ := ![(1 : ℝ), 2, 3]

theorem vector_calculation :
  a - (4 • b) + (2 • c) = ![(-25 : ℝ), 13, 20] := by sorry

end NUMINAMATH_CALUDE_vector_calculation_l3496_349686


namespace NUMINAMATH_CALUDE_g_neg_two_equals_neg_seventeen_l3496_349678

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define g in terms of f
def g (x : ℝ) : ℝ := f x + 1

-- State the theorem
theorem g_neg_two_equals_neg_seventeen
  (h1 : ∀ x, f x + 2 * x^2 = -(f (-x) + 2 * (-x)^2)) -- y = f(x) + 2x^2 is odd
  (h2 : f 2 = 2) -- f(2) = 2
  : g f (-2) = -17 := by
  sorry

end NUMINAMATH_CALUDE_g_neg_two_equals_neg_seventeen_l3496_349678


namespace NUMINAMATH_CALUDE_negative_double_less_than_self_l3496_349626

theorem negative_double_less_than_self (a : ℝ) : a < 0 → 2 * a < a := by sorry

end NUMINAMATH_CALUDE_negative_double_less_than_self_l3496_349626


namespace NUMINAMATH_CALUDE_shane_sandwich_problem_l3496_349696

/-- The number of slices in each package of sliced bread -/
def slices_per_bread_package : ℕ := 20

/-- The number of packages of sliced bread Shane buys -/
def bread_packages : ℕ := 2

/-- The number of packages of sliced ham Shane buys -/
def ham_packages : ℕ := 2

/-- The number of ham slices in each package -/
def ham_slices_per_package : ℕ := 8

/-- The number of bread slices needed for each sandwich -/
def bread_slices_per_sandwich : ℕ := 2

/-- The number of bread slices leftover after making sandwiches -/
def leftover_bread_slices : ℕ := 8

theorem shane_sandwich_problem :
  slices_per_bread_package * bread_packages = 
    (ham_packages * ham_slices_per_package * bread_slices_per_sandwich) + leftover_bread_slices := by
  sorry

end NUMINAMATH_CALUDE_shane_sandwich_problem_l3496_349696


namespace NUMINAMATH_CALUDE_runners_meeting_point_l3496_349647

/-- Represents the meeting point on the circular track -/
inductive MeetingPoint
| S -- Boundary of A and D
| A
| B
| C
| D

/-- Represents a runner on the circular track -/
structure Runner where
  startPosition : ℝ -- Position in meters from point S
  direction : Bool -- true for counterclockwise, false for clockwise
  distanceRun : ℝ -- Total distance run in meters

/-- Theorem stating where Alice and Bob meet on the circular track -/
theorem runners_meeting_point 
  (trackCircumference : ℝ)
  (alice : Runner)
  (bob : Runner)
  (h1 : trackCircumference = 60)
  (h2 : alice.startPosition = 0)
  (h3 : alice.direction = true)
  (h4 : alice.distanceRun = 7200)
  (h5 : bob.startPosition = 30)
  (h6 : bob.direction = false)
  (h7 : bob.distanceRun = alice.distanceRun) :
  MeetingPoint.S = 
    (let aliceFinalPosition := alice.startPosition + (alice.distanceRun % trackCircumference)
     let bobFinalPosition := bob.startPosition - (bob.distanceRun % trackCircumference)
     if aliceFinalPosition = bobFinalPosition 
     then MeetingPoint.S 
     else MeetingPoint.A) :=
by sorry

end NUMINAMATH_CALUDE_runners_meeting_point_l3496_349647


namespace NUMINAMATH_CALUDE_difference_of_largest_prime_factors_l3496_349656

theorem difference_of_largest_prime_factors : ∃ (p q : Nat), 
  Nat.Prime p ∧ Nat.Prime q ∧ p * q = 172081 ∧ 
  ∀ (r : Nat), Nat.Prime r ∧ r ∣ 172081 → r ≤ max p q ∧
  max p q - min p q = 13224 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_largest_prime_factors_l3496_349656


namespace NUMINAMATH_CALUDE_a_equals_one_sufficient_not_necessary_l3496_349611

def is_parallel (a : ℝ) : Prop :=
  a^2 = 1

theorem a_equals_one_sufficient_not_necessary :
  (∃ a : ℝ, is_parallel a ∧ a ≠ 1) ∧
  (∀ a : ℝ, a = 1 → is_parallel a) :=
by sorry

end NUMINAMATH_CALUDE_a_equals_one_sufficient_not_necessary_l3496_349611


namespace NUMINAMATH_CALUDE_johns_trip_duration_l3496_349669

/-- The duration of John's trip given his travel conditions -/
def trip_duration (first_country_duration : ℕ) (num_countries : ℕ) : ℕ :=
  first_country_duration + 2 * first_country_duration * (num_countries - 1)

/-- Theorem stating that John's trip duration is 10 weeks -/
theorem johns_trip_duration :
  trip_duration 2 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_johns_trip_duration_l3496_349669


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3496_349683

theorem min_value_sum_reciprocals (p q r s t u : ℝ) 
  (pos_p : p > 0) (pos_q : q > 0) (pos_r : r > 0) 
  (pos_s : s > 0) (pos_t : t > 0) (pos_u : u > 0)
  (sum_eq_10 : p + q + r + s + t + u = 10) : 
  (1/p + 9/q + 4/r + 1/s + 16/t + 25/u) ≥ 25.6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3496_349683


namespace NUMINAMATH_CALUDE_trapezoid_constructible_l3496_349687

/-- A trapezoid with given bases and diagonals -/
structure Trapezoid where
  a : ℝ  -- length of one base
  c : ℝ  -- length of the other base
  e : ℝ  -- length of one diagonal
  f : ℝ  -- length of the other diagonal

/-- Condition for trapezoid constructibility -/
def is_constructible (t : Trapezoid) : Prop :=
  t.e + t.f > t.a + t.c ∧
  t.e + (t.a + t.c) > t.f ∧
  t.f + (t.a + t.c) > t.e

/-- Theorem stating that a trapezoid with bases 8 and 4, and diagonals 9 and 15 is constructible -/
theorem trapezoid_constructible : 
  is_constructible { a := 8, c := 4, e := 9, f := 15 } := by
  sorry


end NUMINAMATH_CALUDE_trapezoid_constructible_l3496_349687


namespace NUMINAMATH_CALUDE_christines_speed_l3496_349688

/-- Given a distance of 20 miles and a time of 5 hours, the speed is 4 miles per hour. -/
theorem christines_speed (distance : ℝ) (time : ℝ) (speed : ℝ) 
  (h1 : distance = 20)
  (h2 : time = 5)
  (h3 : speed = distance / time) :
  speed = 4 := by
  sorry

end NUMINAMATH_CALUDE_christines_speed_l3496_349688


namespace NUMINAMATH_CALUDE_squared_sum_equals_20_75_l3496_349617

theorem squared_sum_equals_20_75 
  (a b c : ℝ) 
  (eq1 : a^2 + 3*b = 9) 
  (eq2 : b^2 + 5*c = -8) 
  (eq3 : c^2 + 7*a = -18) : 
  a^2 + b^2 + c^2 = 20.75 := by
  sorry

end NUMINAMATH_CALUDE_squared_sum_equals_20_75_l3496_349617


namespace NUMINAMATH_CALUDE_grasshopper_visits_all_integers_l3496_349664

def grasshopper_jump (k : ℕ) : ℤ :=
  if k % 2 = 0 then -k else k + 1

def grasshopper_position (n : ℕ) : ℤ :=
  (List.range n).foldl (λ acc k => acc + grasshopper_jump k) 0

theorem grasshopper_visits_all_integers :
  ∀ (z : ℤ), ∃ (n : ℕ), grasshopper_position n = z :=
sorry

end NUMINAMATH_CALUDE_grasshopper_visits_all_integers_l3496_349664


namespace NUMINAMATH_CALUDE_combined_mixture_ratio_l3496_349629

/-- Represents a mixture of milk and water -/
structure Mixture where
  milk : ℚ
  water : ℚ

/-- Combines two mixtures -/
def combineMixtures (m1 m2 : Mixture) : Mixture :=
  { milk := m1.milk + m2.milk, water := m1.water + m2.water }

theorem combined_mixture_ratio :
  let m1 : Mixture := { milk := 4, water := 2 }
  let m2 : Mixture := { milk := 5, water := 1 }
  let combined := combineMixtures m1 m2
  combined.milk / combined.water = 3 := by
  sorry

end NUMINAMATH_CALUDE_combined_mixture_ratio_l3496_349629


namespace NUMINAMATH_CALUDE_blue_paint_cans_l3496_349690

/-- Given a paint mixture with a blue to green ratio of 4:1 and a total of 40 cans,
    prove that 32 cans of blue paint are required. -/
theorem blue_paint_cans (total_cans : ℕ) (blue_ratio green_ratio : ℕ) 
  (h1 : total_cans = 40)
  (h2 : blue_ratio = 4)
  (h3 : green_ratio = 1) :
  (blue_ratio * total_cans) / (blue_ratio + green_ratio) = 32 := by
sorry


end NUMINAMATH_CALUDE_blue_paint_cans_l3496_349690


namespace NUMINAMATH_CALUDE_larger_number_proof_l3496_349634

theorem larger_number_proof (a b : ℕ+) : 
  (Nat.gcd a b = 25) →
  (Nat.lcm a b = 4550) →
  (13 ∣ Nat.lcm a b) →
  (14 ∣ Nat.lcm a b) →
  max a b = 350 := by
sorry

end NUMINAMATH_CALUDE_larger_number_proof_l3496_349634


namespace NUMINAMATH_CALUDE_largest_510_triple_l3496_349620

/-- Converts a base-10 number to its base-5 representation as a list of digits -/
def toBase5 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) : List ℕ :=
    if m = 0 then [] else (m % 5) :: aux (m / 5)
  aux n

/-- Interprets a list of digits as a base-10 number -/
def fromDigits (digits : List ℕ) : ℕ :=
  digits.foldl (fun acc d => 10 * acc + d) 0

/-- Checks if a number is a 5-10 triple -/
def is510Triple (n : ℕ) : Prop :=
  fromDigits (toBase5 n) = 3 * n

theorem largest_510_triple :
  (∀ m : ℕ, m > 115 → ¬ is510Triple m) ∧ is510Triple 115 :=
sorry

end NUMINAMATH_CALUDE_largest_510_triple_l3496_349620
