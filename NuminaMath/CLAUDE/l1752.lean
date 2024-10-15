import Mathlib

namespace NUMINAMATH_CALUDE_inequality_proof_l1752_175273

theorem inequality_proof (a b c d : ℝ) 
  (h1 : b + Real.sin a > d + Real.sin c) 
  (h2 : a + Real.sin b > c + Real.sin d) : 
  a + b > c + d := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1752_175273


namespace NUMINAMATH_CALUDE_remainder_theorem_l1752_175299

theorem remainder_theorem (n : ℤ) : 
  (∃ k : ℤ, 2 * n = 10 * k + 2) → 
  (∃ m : ℤ, n = 20 * m + 1) := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1752_175299


namespace NUMINAMATH_CALUDE_range_of_fraction_l1752_175230

theorem range_of_fraction (x y : ℝ) (h : x^2 + y^2 + 2*x = 0) :
  -1 ≤ (y - x) / (x - 1) ∧ (y - x) / (x - 1) ≤ 1/3 :=
sorry

end NUMINAMATH_CALUDE_range_of_fraction_l1752_175230


namespace NUMINAMATH_CALUDE_reciprocal_difference_square_sum_product_difference_l1752_175281

/-- The difference between the reciprocal of x and y is equal to 1/x - y, where x ≠ 0 -/
theorem reciprocal_difference (x y : ℝ) (h : x ≠ 0) :
  1 / x - y = 1 / x - y := by sorry

/-- The difference between the square of the sum of a and b and the product of a and b
    is equal to (a+b)^2 - ab -/
theorem square_sum_product_difference (a b : ℝ) :
  (a + b)^2 - a * b = (a + b)^2 - a * b := by sorry

end NUMINAMATH_CALUDE_reciprocal_difference_square_sum_product_difference_l1752_175281


namespace NUMINAMATH_CALUDE_product_reciprocals_equals_one_l1752_175297

theorem product_reciprocals_equals_one :
  (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_reciprocals_equals_one_l1752_175297


namespace NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l1752_175288

theorem equilateral_triangle_perimeter (s : ℝ) (h : s > 0) :
  (s^2 * Real.sqrt 3) / 4 = 2 * s → 3 * s = 8 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l1752_175288


namespace NUMINAMATH_CALUDE_ms_cole_total_students_l1752_175234

/-- The number of students in Ms. Cole's sixth-level math class -/
def S6 : ℕ := 40

/-- The number of students in Ms. Cole's fourth-level math class -/
def S4 : ℕ := 4 * S6

/-- The number of students in Ms. Cole's seventh-level math class -/
def S7 : ℕ := 2 * S4

/-- The total number of math students Ms. Cole teaches -/
def total_students : ℕ := S6 + S4 + S7

/-- Theorem stating that Ms. Cole teaches 520 math students in total -/
theorem ms_cole_total_students : total_students = 520 := by
  sorry

end NUMINAMATH_CALUDE_ms_cole_total_students_l1752_175234


namespace NUMINAMATH_CALUDE_sum_of_squared_sums_of_roots_l1752_175255

theorem sum_of_squared_sums_of_roots (p q r : ℝ) : 
  (p^3 - 15*p^2 + 25*p - 10 = 0) → 
  (q^3 - 15*q^2 + 25*q - 10 = 0) → 
  (r^3 - 15*r^2 + 25*r - 10 = 0) → 
  (p+q)^2 + (q+r)^2 + (r+p)^2 = 400 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squared_sums_of_roots_l1752_175255


namespace NUMINAMATH_CALUDE_sum_integers_negative20_to_10_l1752_175260

def sum_integers (a b : Int) : Int :=
  (b - a + 1) * (a + b) / 2

theorem sum_integers_negative20_to_10 :
  sum_integers (-20) 10 = -155 := by
  sorry

end NUMINAMATH_CALUDE_sum_integers_negative20_to_10_l1752_175260


namespace NUMINAMATH_CALUDE_clouddale_rainfall_2008_l1752_175237

def average_monthly_rainfall_2007 : ℝ := 45.2
def rainfall_increase_2008 : ℝ := 3.5
def months_in_year : ℕ := 12

theorem clouddale_rainfall_2008 :
  let average_monthly_rainfall_2008 := average_monthly_rainfall_2007 + rainfall_increase_2008
  let total_rainfall_2008 := average_monthly_rainfall_2008 * months_in_year
  total_rainfall_2008 = 584.4 := by
sorry

end NUMINAMATH_CALUDE_clouddale_rainfall_2008_l1752_175237


namespace NUMINAMATH_CALUDE_regular_16gon_symmetry_sum_l1752_175201

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  -- Add necessary fields here

/-- The number of lines of symmetry in a regular polygon -/
def linesOfSymmetry (p : RegularPolygon n) : ℕ := sorry

/-- The smallest positive angle of rotational symmetry in degrees for a regular polygon -/
def smallestRotationAngle (p : RegularPolygon n) : ℝ := sorry

theorem regular_16gon_symmetry_sum :
  ∀ (p : RegularPolygon 16),
    (linesOfSymmetry p : ℝ) + smallestRotationAngle p = 38.5 := by
  sorry

end NUMINAMATH_CALUDE_regular_16gon_symmetry_sum_l1752_175201


namespace NUMINAMATH_CALUDE_greatest_of_three_consecutive_integers_sum_21_l1752_175243

theorem greatest_of_three_consecutive_integers_sum_21 :
  ∀ x y z : ℤ, 
    (y = x + 1) → 
    (z = y + 1) → 
    (x + y + z = 21) → 
    (max x (max y z) = 8) :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_of_three_consecutive_integers_sum_21_l1752_175243


namespace NUMINAMATH_CALUDE_only_vehicle_green_light_is_random_l1752_175238

-- Define the type for events
inductive Event
  | TriangleInequality
  | SunRise
  | VehicleGreenLight
  | NegativeAbsoluteValue

-- Define a predicate for random events
def isRandomEvent : Event → Prop :=
  fun e => match e with
    | Event.TriangleInequality => false
    | Event.SunRise => false
    | Event.VehicleGreenLight => true
    | Event.NegativeAbsoluteValue => false

-- Theorem statement
theorem only_vehicle_green_light_is_random :
  ∀ e : Event, isRandomEvent e ↔ e = Event.VehicleGreenLight :=
by sorry

end NUMINAMATH_CALUDE_only_vehicle_green_light_is_random_l1752_175238


namespace NUMINAMATH_CALUDE_parabola_equation_l1752_175251

/-- The standard equation of a parabola with vertex (0,0) and focus (3,0) -/
theorem parabola_equation (x y : ℝ) : 
  (∃ (p : ℝ), p > 0 ∧ y^2 = 4 * p * x ∧ 3 = p) → y^2 = 12 * x := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_l1752_175251


namespace NUMINAMATH_CALUDE_annie_diorama_building_time_l1752_175274

/-- The time Annie spent building her diorama -/
def building_time (planning_time : ℕ) : ℕ := 3 * planning_time - 5

/-- The total time Annie spent on her diorama project -/
def total_time (planning_time : ℕ) : ℕ := building_time planning_time + planning_time

theorem annie_diorama_building_time :
  ∃ (planning_time : ℕ), total_time planning_time = 67 ∧ building_time planning_time = 49 := by
sorry

end NUMINAMATH_CALUDE_annie_diorama_building_time_l1752_175274


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l1752_175264

/-- Given two 2D vectors a and b, where a is parallel to b, prove that m = -1 --/
theorem parallel_vectors_m_value (a b : ℝ × ℝ) (m : ℝ) :
  a = (1, 2) →
  b = (2, 3 - m) →
  (∃ (k : ℝ), a = k • b) →
  m = -1 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l1752_175264


namespace NUMINAMATH_CALUDE_age_puzzle_l1752_175252

theorem age_puzzle (A : ℕ) (N : ℚ) (h1 : A = 24) (h2 : (A + 3) * N - (A - 3) * N = A) : N = 4 := by
  sorry

end NUMINAMATH_CALUDE_age_puzzle_l1752_175252


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1752_175268

def A : Set ℝ := {x | |x - 1| < 2}

def B : Set ℝ := {y | ∃ x ∈ Set.Icc 0 2, y = 2^x}

theorem intersection_of_A_and_B : A ∩ B = Set.Ico 1 3 := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1752_175268


namespace NUMINAMATH_CALUDE_units_digit_of_square_l1752_175284

/-- 
Given an integer n, if the tens digit of n^2 is 7, 
then the units digit of n^2 is 6.
-/
theorem units_digit_of_square (n : ℤ) : 
  (n^2 % 100 / 10 = 7) → (n^2 % 10 = 6) := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_square_l1752_175284


namespace NUMINAMATH_CALUDE_point_M_coordinates_l1752_175202

-- Define the points A, B, C, and M
def A : ℝ × ℝ := (2, -4)
def B : ℝ × ℝ := (-1, 3)
def C : ℝ × ℝ := (3, 4)
def M : ℝ × ℝ := (-11, -15)

-- Define vectors
def vec (P Q : ℝ × ℝ) : ℝ × ℝ := (Q.1 - P.1, Q.2 - P.2)

-- State the theorem
theorem point_M_coordinates :
  vec C M = (2 : ℝ) • (vec C A) + (3 : ℝ) • (vec C B) → M = (-11, -15) := by
  sorry


end NUMINAMATH_CALUDE_point_M_coordinates_l1752_175202


namespace NUMINAMATH_CALUDE_min_value_H_negative_reals_l1752_175272

-- Define the concept of an odd function
def OddFunction (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the function H
def H (a b : ℝ) (f g : ℝ → ℝ) (x : ℝ) : ℝ := a * f x + b * g x + 1

-- State the theorem
theorem min_value_H_negative_reals 
  (f g : ℝ → ℝ) (a b : ℝ) 
  (hf : OddFunction f) (hg : OddFunction g)
  (hmax : ∃ M, M = 5 ∧ ∀ x > 0, H a b f g x ≤ M) :
  ∃ m, m = -3 ∧ ∀ x < 0, H a b f g x ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_H_negative_reals_l1752_175272


namespace NUMINAMATH_CALUDE_insufficient_condition_for_similarity_l1752_175278

-- Define the triangles
structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define the angles
def angle (t : Triangle) (v : Fin 3) : ℝ := sorry

-- Define similarity of triangles
def similar (t1 t2 : Triangle) : Prop := sorry

theorem insufficient_condition_for_similarity (ABC A'B'C' : Triangle) :
  angle ABC 1 = 90 ∧ 
  angle A'B'C' 1 = 90 ∧ 
  angle ABC 0 = 30 ∧ 
  angle ABC 2 = 60 →
  ¬ (∀ t1 t2 : Triangle, similar t1 t2) :=
sorry

end NUMINAMATH_CALUDE_insufficient_condition_for_similarity_l1752_175278


namespace NUMINAMATH_CALUDE_min_value_expression_l1752_175214

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 5/3) :
  4 / (a + 2*b) + 9 / (2*a + b) ≥ 5 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l1752_175214


namespace NUMINAMATH_CALUDE_complex_magnitude_l1752_175229

theorem complex_magnitude (z : ℂ) (h : z * (1 + Complex.I) = 2 * Complex.I) : 
  Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l1752_175229


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l1752_175225

/-- Given an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    where √3b = 2c sin B, c = √7, and a + b = 5, prove that:
    1. The angle C is equal to π/3
    2. The area of triangle ABC is (3√3)/2 -/
theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  0 < A → A < π/2 →
  0 < B → B < π/2 →
  0 < C → C < π/2 →
  Real.sqrt 3 * b = 2 * c * Real.sin B →
  c = Real.sqrt 7 →
  a + b = 5 →
  C = π/3 ∧ (1/2 * a * b * Real.sin C = (3 * Real.sqrt 3) / 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l1752_175225


namespace NUMINAMATH_CALUDE_all_three_classes_l1752_175283

/-- Represents the number of students in each class combination --/
structure ClassCombinations where
  yoga : ℕ
  bridge : ℕ
  painting : ℕ
  yogaBridge : ℕ
  yogaPainting : ℕ
  bridgePainting : ℕ
  allThree : ℕ

/-- Represents the given conditions of the problem --/
def problem_conditions (c : ClassCombinations) : Prop :=
  c.yoga + c.bridge + c.painting + c.yogaBridge + c.yogaPainting + c.bridgePainting + c.allThree = 20 ∧
  c.yoga + c.yogaBridge + c.yogaPainting + c.allThree = 10 ∧
  c.bridge + c.yogaBridge + c.bridgePainting + c.allThree = 13 ∧
  c.painting + c.yogaPainting + c.bridgePainting + c.allThree = 9 ∧
  c.yogaBridge + c.yogaPainting + c.bridgePainting + c.allThree = 9

theorem all_three_classes (c : ClassCombinations) :
  problem_conditions c → c.allThree = 3 := by
  sorry

end NUMINAMATH_CALUDE_all_three_classes_l1752_175283


namespace NUMINAMATH_CALUDE_sum_of_solutions_eq_five_l1752_175296

theorem sum_of_solutions_eq_five :
  let f : ℝ → ℝ := λ M => M * (M - 5) + 9
  ∃ M₁ M₂ : ℝ, (f M₁ = 0 ∧ f M₂ = 0 ∧ M₁ ≠ M₂) ∧ M₁ + M₂ = 5 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_eq_five_l1752_175296


namespace NUMINAMATH_CALUDE_tennis_ball_distribution_l1752_175236

theorem tennis_ball_distribution (initial_balls : ℕ) (containers : ℕ) : 
  initial_balls = 100 → 
  containers = 5 → 
  (initial_balls / 2) / containers = 10 := by
  sorry

end NUMINAMATH_CALUDE_tennis_ball_distribution_l1752_175236


namespace NUMINAMATH_CALUDE_solution_set_for_a_equals_two_range_for_empty_solution_set_l1752_175265

def f (a : ℝ) (x : ℝ) : ℝ := |x + 1| - |x - a|

theorem solution_set_for_a_equals_two :
  {x : ℝ | f 2 x > 2} = {x : ℝ | x > 3/2} := by sorry

theorem range_for_empty_solution_set :
  {a : ℝ | a > 0 ∧ ∀ x, f a x < 2*a} = {a : ℝ | a > 1} := by sorry

end NUMINAMATH_CALUDE_solution_set_for_a_equals_two_range_for_empty_solution_set_l1752_175265


namespace NUMINAMATH_CALUDE_linear_function_characterization_l1752_175289

theorem linear_function_characterization (f : ℝ → ℝ) : 
  (∀ x y : ℝ, f (x^2 - y^2) = (x - y) * (f x + f y)) → 
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x :=
by sorry

end NUMINAMATH_CALUDE_linear_function_characterization_l1752_175289


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_arithmetic_sequence_sum_l1752_175212

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of terms with indices that add up to the same value is constant -/
theorem arithmetic_sequence_sum_property (a : ℕ → ℝ) (h : arithmetic_sequence a) :
  ∀ i j k l : ℕ, i + l = j + k → a i + a l = a j + a k :=
sorry

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h : arithmetic_sequence a) 
  (h_sum : a 3 + a 7 = 37) : a 2 + a 4 + a 6 + a 8 = 74 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_arithmetic_sequence_sum_l1752_175212


namespace NUMINAMATH_CALUDE_solution_of_linear_equation_l1752_175291

theorem solution_of_linear_equation (x y a : ℝ) : 
  x = 1 → y = 3 → a * x - 2 * y = 4 → a = 10 := by sorry

end NUMINAMATH_CALUDE_solution_of_linear_equation_l1752_175291


namespace NUMINAMATH_CALUDE_fourth_term_coefficient_implies_a_equals_one_l1752_175269

/-- The binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ := sorry

theorem fourth_term_coefficient_implies_a_equals_one (x a : ℝ) :
  (binomial 9 3 : ℝ) * a^3 = 84 → a = 1 := by sorry

end NUMINAMATH_CALUDE_fourth_term_coefficient_implies_a_equals_one_l1752_175269


namespace NUMINAMATH_CALUDE_recipe_total_cups_l1752_175277

/-- Represents the ratio of ingredients in the recipe -/
structure RecipeRatio :=
  (butter : ℕ)
  (flour : ℕ)
  (sugar : ℕ)
  (eggs : ℕ)

/-- Calculates the total number of cups for all ingredients given a recipe ratio and the amount of sugar -/
def totalCups (ratio : RecipeRatio) (sugarCups : ℕ) : ℕ :=
  let partSize := sugarCups / ratio.sugar
  partSize * (ratio.butter + ratio.flour + ratio.sugar + ratio.eggs)

/-- Theorem stating that for the given recipe ratio and 10 cups of sugar, the total is 30 cups -/
theorem recipe_total_cups : 
  let ratio : RecipeRatio := ⟨2, 7, 5, 1⟩
  totalCups ratio 10 = 30 := by
  sorry

end NUMINAMATH_CALUDE_recipe_total_cups_l1752_175277


namespace NUMINAMATH_CALUDE_triangle_circumscribed_circle_diameter_l1752_175218

theorem triangle_circumscribed_circle_diameter 
  (a : ℝ) (A : ℝ) (D : ℝ) :
  a = 10 ∧ A = π/4 ∧ D = a / Real.sin A → D = 10 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_circumscribed_circle_diameter_l1752_175218


namespace NUMINAMATH_CALUDE_number_problem_l1752_175271

theorem number_problem (x : ℝ) : 0.3 * x = 0.6 * 150 + 120 → x = 700 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1752_175271


namespace NUMINAMATH_CALUDE_digit_counting_theorem_l1752_175263

/-- The set of available digits -/
def availableDigits : Finset ℕ := {0, 1, 2, 3, 5, 9}

/-- Count of four-digit numbers -/
def countFourDigit : ℕ := 300

/-- Count of four-digit odd numbers -/
def countFourDigitOdd : ℕ := 192

/-- Count of four-digit even numbers -/
def countFourDigitEven : ℕ := 108

/-- Total count of natural numbers -/
def countNaturalNumbers : ℕ := 1631

/-- Main theorem stating the counting results -/
theorem digit_counting_theorem :
  (∀ d ∈ availableDigits, d < 10) ∧
  (countFourDigit = 300) ∧
  (countFourDigitOdd = 192) ∧
  (countFourDigitEven = 108) ∧
  (countNaturalNumbers = 1631) := by
  sorry

end NUMINAMATH_CALUDE_digit_counting_theorem_l1752_175263


namespace NUMINAMATH_CALUDE_cucumber_count_l1752_175262

theorem cucumber_count (total : ℕ) (ratio : ℕ) (h1 : total = 420) (h2 : ratio = 4) :
  ∃ (cucumbers : ℕ), cucumbers * (ratio + 1) = total ∧ cucumbers = 84 := by
  sorry

end NUMINAMATH_CALUDE_cucumber_count_l1752_175262


namespace NUMINAMATH_CALUDE_profit_maximization_l1752_175275

/-- Profit function for computer sales --/
def profit_function (x : ℝ) : ℝ := -50 * x + 15000

/-- Constraint on the number of computers --/
def constraint (x : ℝ) : Prop := 100 / 3 ≤ x ∧ x ≤ 100 / 3

theorem profit_maximization (x : ℝ) :
  constraint x →
  ∀ y, constraint y → profit_function y ≤ profit_function x →
  x = 34 :=
sorry

end NUMINAMATH_CALUDE_profit_maximization_l1752_175275


namespace NUMINAMATH_CALUDE_asymptote_sum_l1752_175266

theorem asymptote_sum (A B C : ℤ) : 
  (∀ x : ℝ, x^3 + A*x^2 + B*x + C = (x + 1)*(x - 3)*(x - 4)) → 
  A + B + C = 11 := by
  sorry

end NUMINAMATH_CALUDE_asymptote_sum_l1752_175266


namespace NUMINAMATH_CALUDE_factorization_equality_l1752_175257

theorem factorization_equality (a b c x y : ℝ) :
  -a * (x - y) - b * (y - x) + c * (x - y) = (y - x) * (a - b - c) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1752_175257


namespace NUMINAMATH_CALUDE_intersection_distance_squared_l1752_175227

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Calculates the squared distance between two points in 2D space -/
def squaredDistance (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

/-- Theorem: The squared distance between intersection points of two specific circles -/
theorem intersection_distance_squared (c1 c2 : Circle)
  (h1 : c1 = ⟨(1, -2), 5⟩)
  (h2 : c2 = ⟨(1, 4), 3⟩) :
  ∃ (p1 p2 : ℝ × ℝ),
    squaredDistance p1 c1.center = c1.radius^2 ∧
    squaredDistance p1 c2.center = c2.radius^2 ∧
    squaredDistance p2 c1.center = c1.radius^2 ∧
    squaredDistance p2 c2.center = c2.radius^2 ∧
    squaredDistance p1 p2 = 224/9 := by
  sorry

end NUMINAMATH_CALUDE_intersection_distance_squared_l1752_175227


namespace NUMINAMATH_CALUDE_sphere_volume_radius_3_l1752_175210

/-- The volume of a sphere with radius 3 cm is 36π cm³. -/
theorem sphere_volume_radius_3 :
  let r : ℝ := 3
  let volume := (4 / 3) * Real.pi * r ^ 3
  volume = 36 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_sphere_volume_radius_3_l1752_175210


namespace NUMINAMATH_CALUDE_f_inequality_range_l1752_175293

noncomputable def f (x : ℝ) : ℝ := Real.log (1 + |x|) - 1 / (1 + x^2)

theorem f_inequality_range (x : ℝ) : 
  f x > f (2*x - 1) ↔ x ∈ Set.Ioo (1/3) 1 :=
sorry

end NUMINAMATH_CALUDE_f_inequality_range_l1752_175293


namespace NUMINAMATH_CALUDE_profit_maximum_l1752_175235

/-- The bank's profit function -/
def profit_function (k : ℝ) (x : ℝ) : ℝ := 0.045 * k * x^2 - k * x^3

/-- Theorem stating that the profit function reaches its maximum at x = 0.03 -/
theorem profit_maximum (k : ℝ) (h : k > 0) :
  ∃ (max : ℝ), ∀ (x : ℝ), x > 0 → profit_function k x ≤ profit_function k 0.03 :=
sorry

end NUMINAMATH_CALUDE_profit_maximum_l1752_175235


namespace NUMINAMATH_CALUDE_remainder_sum_powers_mod_seven_l1752_175207

theorem remainder_sum_powers_mod_seven :
  (9^7 + 8^8 + 7^9) % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_powers_mod_seven_l1752_175207


namespace NUMINAMATH_CALUDE_quadratic_not_always_positive_l1752_175222

theorem quadratic_not_always_positive : ¬ (∀ x : ℝ, x^2 + 3*x + 1 > 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_not_always_positive_l1752_175222


namespace NUMINAMATH_CALUDE_coupon_redemption_schedule_l1752_175247

-- Define the days of the week
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

def next_day (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Monday    => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday   => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday  => DayOfWeek.Friday
  | DayOfWeek.Friday    => DayOfWeek.Saturday
  | DayOfWeek.Saturday  => DayOfWeek.Sunday
  | DayOfWeek.Sunday    => DayOfWeek.Monday

def advance_days (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => advance_days (next_day d) n

def is_saturday (d : DayOfWeek) : Prop :=
  d = DayOfWeek.Saturday

theorem coupon_redemption_schedule :
  let start_day := DayOfWeek.Monday
  let days_between_redemptions := 15
  let num_coupons := 7
  ∀ i, i < num_coupons →
    ¬(is_saturday (advance_days start_day (i * days_between_redemptions))) :=
by sorry

end NUMINAMATH_CALUDE_coupon_redemption_schedule_l1752_175247


namespace NUMINAMATH_CALUDE_number_of_preferred_shares_l1752_175295

/-- Represents the number of preferred shares -/
def preferred_shares : ℕ := sorry

/-- Represents the number of common shares -/
def common_shares : ℕ := 3000

/-- Represents the par value of each share in rupees -/
def par_value : ℚ := 50

/-- Represents the annual dividend rate for preferred shares -/
def preferred_dividend_rate : ℚ := 1 / 10

/-- Represents the annual dividend rate for common shares -/
def common_dividend_rate : ℚ := 7 / 100

/-- Represents the total annual dividend received in rupees -/
def total_annual_dividend : ℚ := 16500

/-- Theorem stating that the number of preferred shares is 1200 -/
theorem number_of_preferred_shares : 
  preferred_shares = 1200 :=
by sorry

end NUMINAMATH_CALUDE_number_of_preferred_shares_l1752_175295


namespace NUMINAMATH_CALUDE_robot_gather_time_l1752_175256

/-- The time (in minutes) it takes a robot to create a battery -/
def create_time : ℕ := 9

/-- The number of robots working simultaneously -/
def num_robots : ℕ := 10

/-- The number of batteries manufactured in 5 hours -/
def batteries_produced : ℕ := 200

/-- The time (in hours) taken to manufacture the batteries -/
def production_time : ℕ := 5

/-- The time (in minutes) it takes a robot to gather materials for a battery -/
def gather_time : ℕ := 6

theorem robot_gather_time :
  gather_time = 6 ∧
  create_time = 9 ∧
  num_robots = 10 ∧
  batteries_produced = 200 ∧
  production_time = 5 →
  num_robots * batteries_produced * (gather_time + create_time) = production_time * 60 :=
by sorry

end NUMINAMATH_CALUDE_robot_gather_time_l1752_175256


namespace NUMINAMATH_CALUDE_initial_mean_calculation_l1752_175205

theorem initial_mean_calculation (n : ℕ) (initial_wrong : ℝ) (corrected : ℝ) (new_mean : ℝ) :
  n = 50 →
  initial_wrong = 23 →
  corrected = 48 →
  new_mean = 30.5 →
  (n : ℝ) * new_mean = (n : ℝ) * (n * new_mean - (corrected - initial_wrong)) / n :=
by sorry

end NUMINAMATH_CALUDE_initial_mean_calculation_l1752_175205


namespace NUMINAMATH_CALUDE_sqrt45_same_type_as_sqrt5_l1752_175249

-- Define the property of being "of the same type as √5"
def same_type_as_sqrt5 (x : ℝ) : Prop :=
  ∃ (k : ℝ), x = k * Real.sqrt 5

-- State the theorem
theorem sqrt45_same_type_as_sqrt5 :
  same_type_as_sqrt5 (Real.sqrt 45) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_sqrt45_same_type_as_sqrt5_l1752_175249


namespace NUMINAMATH_CALUDE_tanker_filling_rate_l1752_175285

/-- Proves that the filling rate of 3 barrels per minute is equivalent to 28.62 m³/hour -/
theorem tanker_filling_rate 
  (barrel_rate : ℝ) 
  (liters_per_barrel : ℝ) 
  (h1 : barrel_rate = 3) 
  (h2 : liters_per_barrel = 159) : 
  (barrel_rate * liters_per_barrel * 60) / 1000 = 28.62 := by
  sorry

end NUMINAMATH_CALUDE_tanker_filling_rate_l1752_175285


namespace NUMINAMATH_CALUDE_min_value_theorem_l1752_175298

theorem min_value_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_eq : 2*m + n = 2) :
  (2/m) + (1/n) ≥ 9/2 ∧ ∃ m n, m > 0 ∧ n > 0 ∧ 2*m + n = 2 ∧ (2/m) + (1/n) = 9/2 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1752_175298


namespace NUMINAMATH_CALUDE_first_digit_89_base5_l1752_175294

/-- Converts a natural number to its base-5 representation -/
def toBase5 (n : ℕ) : List ℕ :=
  if n < 5 then [n]
  else (n % 5) :: toBase5 (n / 5)

/-- Returns the first (leftmost) digit of a number in its base-5 representation -/
def firstDigitBase5 (n : ℕ) : ℕ :=
  (toBase5 n).reverse.head!

theorem first_digit_89_base5 :
  firstDigitBase5 89 = 3 := by sorry

end NUMINAMATH_CALUDE_first_digit_89_base5_l1752_175294


namespace NUMINAMATH_CALUDE_olympic_high_school_contest_l1752_175244

theorem olympic_high_school_contest (f s : ℕ) : 
  f > 0 → s > 0 → (2 * f) / 5 = (4 * s) / 5 → f = 2 * s := by
  sorry

#check olympic_high_school_contest

end NUMINAMATH_CALUDE_olympic_high_school_contest_l1752_175244


namespace NUMINAMATH_CALUDE_cosine_sum_equality_l1752_175232

theorem cosine_sum_equality : 
  Real.cos (15 * π / 180) * Real.cos (30 * π / 180) + 
  Real.cos (105 * π / 180) * Real.sin (30 * π / 180) = 
  Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_cosine_sum_equality_l1752_175232


namespace NUMINAMATH_CALUDE_special_sequence_remainder_l1752_175219

def sequence_condition (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, 1 ≤ n ∧ n < 69 → 3 * a n = a (n - 1) + a (n + 1)

theorem special_sequence_remainder :
  ∀ a : ℕ → ℤ,
  sequence_condition a →
  a 0 = 0 →
  a 1 = 1 →
  a 2 = 3 →
  a 3 = 8 →
  a 4 = 21 →
  ∃ k : ℤ, a 69 = 6 * k + 4 :=
by sorry

end NUMINAMATH_CALUDE_special_sequence_remainder_l1752_175219


namespace NUMINAMATH_CALUDE_jindra_dice_count_l1752_175200

/-- Represents the number of dice in half a layer -/
def half_layer : ℕ := 18

/-- Represents the number of complete layers -/
def complete_layers : ℕ := 6

/-- Theorem stating the total number of dice Jindra had yesterday -/
theorem jindra_dice_count : 
  (2 * half_layer * complete_layers) + half_layer = 234 := by
  sorry

end NUMINAMATH_CALUDE_jindra_dice_count_l1752_175200


namespace NUMINAMATH_CALUDE_polynomial_rewrite_l1752_175203

variable (x y : ℝ)

def original_polynomial := x^3 - 3*x^2*y + 3*x*y^2 - y^3

theorem polynomial_rewrite :
  ((x^3 - y^3) - (3*x^2*y - 3*x*y^2) = original_polynomial x y) ∧
  ((x^3 + 3*x*y^2) - (3*x^2*y + y^3) = original_polynomial x y) ∧
  ((3*x*y^2 - 3*x^2*y) - (y^3 - x^3) = original_polynomial x y) ∧
  ¬((x^3 - 3*x^2*y) - (3*x*y^2 + y^3) = original_polynomial x y) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_rewrite_l1752_175203


namespace NUMINAMATH_CALUDE_gym_membership_cost_l1752_175250

/-- Calculates the total cost of gym memberships for the first year -/
theorem gym_membership_cost (cheap_monthly_fee : ℕ) (cheap_signup_fee : ℕ) 
  (expensive_monthly_multiplier : ℕ) (expensive_signup_months : ℕ) (months_per_year : ℕ) : 
  cheap_monthly_fee = 10 →
  cheap_signup_fee = 50 →
  expensive_monthly_multiplier = 3 →
  expensive_signup_months = 4 →
  months_per_year = 12 →
  (cheap_monthly_fee * months_per_year + cheap_signup_fee) + 
  (cheap_monthly_fee * expensive_monthly_multiplier * months_per_year + 
   cheap_monthly_fee * expensive_monthly_multiplier * expensive_signup_months) = 650 := by
  sorry

#check gym_membership_cost

end NUMINAMATH_CALUDE_gym_membership_cost_l1752_175250


namespace NUMINAMATH_CALUDE_x_minus_y_value_l1752_175261

theorem x_minus_y_value (x y : ℚ) 
  (eq1 : 3 * x - 4 * y = 17) 
  (eq2 : x + 3 * y = 1) : 
  x - y = 69 / 13 := by
sorry

end NUMINAMATH_CALUDE_x_minus_y_value_l1752_175261


namespace NUMINAMATH_CALUDE_log_product_theorem_l1752_175223

theorem log_product_theorem (c d : ℕ+) : 
  (d.val - c.val = 435) → 
  (Real.log d.val / Real.log c.val = 2) → 
  (c.val + d.val = 930) := by
sorry

end NUMINAMATH_CALUDE_log_product_theorem_l1752_175223


namespace NUMINAMATH_CALUDE_system_solution_l1752_175221

theorem system_solution : 
  ∃ (x y z : ℝ), 
    (x + y + z = 15 ∧ 
     x^2 + y^2 + z^2 = 81 ∧ 
     x*y + x*z = 3*y*z) ∧
    ((x = 6 ∧ y = 3 ∧ z = 6) ∨ 
     (x = 6 ∧ y = 6 ∧ z = 3)) := by
  sorry

#check system_solution

end NUMINAMATH_CALUDE_system_solution_l1752_175221


namespace NUMINAMATH_CALUDE_cost_price_per_meter_is_58_l1752_175248

/-- Calculates the cost price per meter of cloth given the total length,
    total selling price, and profit per meter. -/
def costPricePerMeter (totalLength : ℕ) (totalSellingPrice : ℕ) (profitPerMeter : ℕ) : ℕ :=
  (totalSellingPrice - totalLength * profitPerMeter) / totalLength

/-- Proves that the cost price per meter of cloth is 58 rupees given the
    specified conditions. -/
theorem cost_price_per_meter_is_58 :
  costPricePerMeter 78 6788 29 = 58 := by sorry

end NUMINAMATH_CALUDE_cost_price_per_meter_is_58_l1752_175248


namespace NUMINAMATH_CALUDE_total_peanuts_l1752_175280

def jose_peanuts : ℕ := 85
def kenya_peanuts : ℕ := jose_peanuts + 48
def malachi_peanuts : ℕ := kenya_peanuts + 35

theorem total_peanuts : jose_peanuts + kenya_peanuts + malachi_peanuts = 386 := by
  sorry

end NUMINAMATH_CALUDE_total_peanuts_l1752_175280


namespace NUMINAMATH_CALUDE_additional_flies_needed_l1752_175220

/-- Represents the number of flies eaten by the frog each day of the week -/
def flies_eaten_per_day : List Nat := [3, 2, 4, 5, 1, 2, 3]

/-- Calculates the total number of flies eaten in a week -/
def total_flies_needed : Nat := flies_eaten_per_day.sum

/-- Number of flies Betty caught in the morning -/
def morning_catch : Nat := 5

/-- Number of flies Betty caught in the afternoon -/
def afternoon_catch : Nat := 6

/-- Number of flies that escaped -/
def escaped_flies : Nat := 1

/-- Calculates the total number of flies Betty successfully caught -/
def total_flies_caught : Nat := morning_catch + afternoon_catch - escaped_flies

/-- Theorem stating the number of additional flies Betty needs -/
theorem additional_flies_needed : 
  total_flies_needed - total_flies_caught = 10 := by sorry

end NUMINAMATH_CALUDE_additional_flies_needed_l1752_175220


namespace NUMINAMATH_CALUDE_card_stack_problem_l1752_175259

theorem card_stack_problem (n : ℕ) : 
  let total_cards := 2 * n
  let pile_A := n
  let pile_B := n
  let card_80_position := 80
  (card_80_position ≤ pile_A) →
  (card_80_position % 2 = 1) →
  (∃ (new_position : ℕ), new_position = card_80_position ∧ 
    new_position = pile_B + (card_80_position + 1) / 2) →
  total_cards = 240 := by
sorry

end NUMINAMATH_CALUDE_card_stack_problem_l1752_175259


namespace NUMINAMATH_CALUDE_system_no_solution_implies_m_equals_two_l1752_175213

/-- Represents a 2x3 augmented matrix -/
structure AugmentedMatrix (α : Type*) :=
  (a11 a12 a13 a21 a22 a23 : α)

/-- Checks if the given augmented matrix represents a system with no real solution -/
def has_no_real_solution (A : AugmentedMatrix ℝ) : Prop :=
  ∀ x y : ℝ, A.a11 * x + A.a12 * y ≠ A.a13 ∨ A.a21 * x + A.a22 * y ≠ A.a23

theorem system_no_solution_implies_m_equals_two :
  ∀ m : ℝ, 
    let A : AugmentedMatrix ℝ := ⟨m, 4, m + 2, 1, m, m⟩
    has_no_real_solution A → m = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_system_no_solution_implies_m_equals_two_l1752_175213


namespace NUMINAMATH_CALUDE_alex_friends_count_l1752_175282

def silk_problem (total_silk : ℕ) (silk_per_dress : ℕ) (dresses_made : ℕ) : ℕ :=
  let silk_used := dresses_made * silk_per_dress
  let silk_given := total_silk - silk_used
  silk_given / silk_per_dress

theorem alex_friends_count :
  silk_problem 600 5 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_alex_friends_count_l1752_175282


namespace NUMINAMATH_CALUDE_sum_of_digits_3_plus_4_pow_17_l1752_175239

/-- The sum of the tens digit and the ones digit of (3+4)^17 in integer form is 7 -/
theorem sum_of_digits_3_plus_4_pow_17 : 
  (((3 + 4)^17 / 10) % 10 + (3 + 4)^17 % 10) = 7 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_3_plus_4_pow_17_l1752_175239


namespace NUMINAMATH_CALUDE_city_g_highest_growth_l1752_175204

structure City where
  name : String
  pop1990 : ℕ
  pop2000 : ℕ

def cities : List City := [
  ⟨"F", 50, 60⟩,
  ⟨"G", 60, 90⟩,
  ⟨"H", 70, 80⟩,
  ⟨"I", 100, 110⟩,
  ⟨"J", 150, 180⟩
]

def growthRate (c : City) : ℚ :=
  (c.pop2000 : ℚ) / (c.pop1990 : ℚ)

def adjustedGrowthRate (c : City) : ℚ :=
  if c.name = "H" then
    growthRate c * (11 / 10)
  else
    growthRate c

theorem city_g_highest_growth :
  ∀ c ∈ cities, c.name ≠ "G" →
    adjustedGrowthRate (cities[1]) ≥ adjustedGrowthRate c := by
  sorry

end NUMINAMATH_CALUDE_city_g_highest_growth_l1752_175204


namespace NUMINAMATH_CALUDE_point_on_line_l1752_175242

theorem point_on_line (m n : ℝ) : 
  (m = n / 6 - 2 / 5) ∧ (m + p = (n + 18) / 6 - 2 / 5) → p = 3 :=
by sorry

end NUMINAMATH_CALUDE_point_on_line_l1752_175242


namespace NUMINAMATH_CALUDE_song_listens_theorem_l1752_175240

def calculate_total_listens (initial_listens : ℕ) (months : ℕ) : ℕ :=
  let doubling_factor := 2 ^ months
  initial_listens * (doubling_factor - 1) + initial_listens

theorem song_listens_theorem (initial_listens : ℕ) (months : ℕ) 
  (h1 : initial_listens = 60000) (h2 : months = 3) :
  calculate_total_listens initial_listens months = 900000 := by
  sorry

#eval calculate_total_listens 60000 3

end NUMINAMATH_CALUDE_song_listens_theorem_l1752_175240


namespace NUMINAMATH_CALUDE_symmetry_proof_l1752_175208

/-- Given two lines in the xy-plane, this function returns true if they are symmetric with respect to the line y = x -/
def are_symmetric_lines (line1 line2 : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, line1 x y ↔ line2 y x

/-- The original line: 2x + 3y + 6 = 0 -/
def original_line (x y : ℝ) : Prop :=
  2 * x + 3 * y + 6 = 0

/-- The symmetric line to be proved: 3x + 2y + 6 = 0 -/
def symmetric_line (x y : ℝ) : Prop :=
  3 * x + 2 * y + 6 = 0

/-- Theorem stating that the symmetric_line is indeed symmetric to the original_line with respect to y = x -/
theorem symmetry_proof : are_symmetric_lines original_line symmetric_line :=
sorry

end NUMINAMATH_CALUDE_symmetry_proof_l1752_175208


namespace NUMINAMATH_CALUDE_factory_weekly_production_l1752_175270

/-- Represents a toy factory with its production characteristics -/
structure ToyFactory where
  daysPerWeek : ℕ
  dailyProduction : ℕ
  constDailyProduction : Bool

/-- Calculates the weekly production of toys for a given factory -/
def weeklyProduction (factory : ToyFactory) : ℕ :=
  factory.daysPerWeek * factory.dailyProduction

/-- Theorem: The weekly production of the given factory is 6500 toys -/
theorem factory_weekly_production :
  ∀ (factory : ToyFactory),
    factory.daysPerWeek = 5 →
    factory.dailyProduction = 1300 →
    factory.constDailyProduction = true →
    weeklyProduction factory = 6500 := by
  sorry

end NUMINAMATH_CALUDE_factory_weekly_production_l1752_175270


namespace NUMINAMATH_CALUDE_unique_solution_condition_l1752_175211

theorem unique_solution_condition (s : ℝ) : 
  (∃! x : ℝ, (s * x - 3) / (x + 1) = x) ↔ (s = 1 + 2 * Real.sqrt 3 ∨ s = 1 - 2 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l1752_175211


namespace NUMINAMATH_CALUDE_female_officers_count_l1752_175290

/-- The total number of officers on duty -/
def total_on_duty : ℕ := 300

/-- The fraction of officers on duty who are female -/
def female_fraction : ℚ := 1/2

/-- The percentage of female officers who were on duty -/
def female_on_duty_percent : ℚ := 15/100

/-- The total number of female officers on the police force -/
def total_female_officers : ℕ := 1000

theorem female_officers_count :
  (total_on_duty : ℚ) * female_fraction / female_on_duty_percent = total_female_officers := by
  sorry

end NUMINAMATH_CALUDE_female_officers_count_l1752_175290


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l1752_175233

def repeating_decimal_3 : ℚ := 1/3
def repeating_decimal_6 : ℚ := 2/3

theorem sum_of_repeating_decimals : 
  repeating_decimal_3 + repeating_decimal_6 = 1 := by sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l1752_175233


namespace NUMINAMATH_CALUDE_product_equivalence_l1752_175209

theorem product_equivalence : 
  (2 + 3) * (2^2 + 3^2) * (2^4 + 3^4) * (2^8 + 3^8) * (2^16 + 3^16) * (2^32 + 3^32) * (2^64 + 3^64) = 3^128 - 2^128 := by
  sorry

end NUMINAMATH_CALUDE_product_equivalence_l1752_175209


namespace NUMINAMATH_CALUDE_school_sections_l1752_175245

theorem school_sections (boys girls : ℕ) (h1 : boys = 408) (h2 : girls = 312) :
  let section_size := Nat.gcd boys girls
  let boy_sections := boys / section_size
  let girl_sections := girls / section_size
  boy_sections + girl_sections = 30 := by
sorry

end NUMINAMATH_CALUDE_school_sections_l1752_175245


namespace NUMINAMATH_CALUDE_celebrity_baby_photo_match_probability_l1752_175216

/-- The number of celebrities and baby photos -/
def n : ℕ := 4

/-- The probability of correctly matching all celebrities with their baby photos -/
def correct_match_probability : ℚ := 1 / (n.factorial : ℚ)

/-- Theorem stating that the probability of correctly matching all celebrities
    with their baby photos when guessing at random is 1/24 -/
theorem celebrity_baby_photo_match_probability :
  correct_match_probability = 1 / 24 := by
  sorry

end NUMINAMATH_CALUDE_celebrity_baby_photo_match_probability_l1752_175216


namespace NUMINAMATH_CALUDE_total_cakes_served_l1752_175224

/-- The number of cakes served on Sunday -/
def sunday_cakes : ℕ := 3

/-- The number of cakes served during lunch on Monday -/
def monday_lunch_cakes : ℕ := 5

/-- The number of cakes served during dinner on Monday -/
def monday_dinner_cakes : ℕ := 6

/-- The number of cakes thrown away on Tuesday -/
def tuesday_thrown_cakes : ℕ := 4

/-- The total number of cakes served on Monday -/
def monday_total_cakes : ℕ := monday_lunch_cakes + monday_dinner_cakes

/-- The number of cakes initially prepared for Tuesday (before throwing away) -/
def tuesday_initial_cakes : ℕ := 2 * monday_total_cakes

/-- The total number of cakes served on Tuesday after throwing away some -/
def tuesday_final_cakes : ℕ := tuesday_initial_cakes - tuesday_thrown_cakes

/-- Theorem stating that the total number of cakes served over three days is 32 -/
theorem total_cakes_served : sunday_cakes + monday_total_cakes + tuesday_final_cakes = 32 := by
  sorry

end NUMINAMATH_CALUDE_total_cakes_served_l1752_175224


namespace NUMINAMATH_CALUDE_calculation_proof_l1752_175228

theorem calculation_proof : 61 + 5 * 12 / (180 / 3) = 62 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1752_175228


namespace NUMINAMATH_CALUDE_bea_earned_more_than_dawn_l1752_175292

/-- The price of Bea's lemonade in cents -/
def bea_price : ℕ := 25

/-- The price of Dawn's lemonade in cents -/
def dawn_price : ℕ := 28

/-- The number of glasses Bea sold -/
def bea_glasses : ℕ := 10

/-- The number of glasses Dawn sold -/
def dawn_glasses : ℕ := 8

/-- Theorem: Bea earned 26 cents more than Dawn -/
theorem bea_earned_more_than_dawn : 
  bea_price * bea_glasses - dawn_price * dawn_glasses = 26 := by
  sorry

end NUMINAMATH_CALUDE_bea_earned_more_than_dawn_l1752_175292


namespace NUMINAMATH_CALUDE_zachary_did_more_pushups_l1752_175217

/-- The number of push-ups Zachary did -/
def zachary_pushups : ℕ := 51

/-- The number of push-ups David did -/
def david_pushups : ℕ := 44

/-- The difference in push-ups between Zachary and David -/
def pushup_difference : ℕ := zachary_pushups - david_pushups

theorem zachary_did_more_pushups : pushup_difference = 7 := by
  sorry

end NUMINAMATH_CALUDE_zachary_did_more_pushups_l1752_175217


namespace NUMINAMATH_CALUDE_largest_multiple_of_seven_below_negative_85_l1752_175215

theorem largest_multiple_of_seven_below_negative_85 :
  ∀ n : ℤ, n % 7 = 0 ∧ n < -85 → n ≤ -91 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_seven_below_negative_85_l1752_175215


namespace NUMINAMATH_CALUDE_ratio_problem_l1752_175286

theorem ratio_problem (a b c P : ℝ) 
  (h1 : b / (a + c) = 1 / 2)
  (h2 : a / (b + c) = 1 / P) :
  (a + b + c) / a = 4 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l1752_175286


namespace NUMINAMATH_CALUDE_equilateral_triangle_side_length_l1752_175258

/-- 
Given an equilateral triangle with a perimeter of 63 cm, 
prove that the length of one side is 21 cm.
-/
theorem equilateral_triangle_side_length 
  (perimeter : ℝ) 
  (is_equilateral : Bool) :
  perimeter = 63 ∧ is_equilateral = true → 
  ∃ (side_length : ℝ), side_length = 21 ∧ perimeter = 3 * side_length :=
by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_side_length_l1752_175258


namespace NUMINAMATH_CALUDE_problem_solution_l1752_175206

def f (m : ℝ) (x : ℝ) : ℝ := |x + 1| + |m - x|

theorem problem_solution :
  (∀ x : ℝ, f 3 x ≥ 6 ↔ (x ≤ -2 ∨ x ≥ 4)) ∧
  (∀ m : ℝ, (∀ x : ℝ, f m x ≥ 8) ↔ (m ≤ -9 ∨ m ≥ 7)) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l1752_175206


namespace NUMINAMATH_CALUDE_sqrt_equation_implies_power_l1752_175241

theorem sqrt_equation_implies_power (x y : ℝ) : 
  Real.sqrt (2 - x) + Real.sqrt (x - 2) + y = 4 → x^y = 16 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_implies_power_l1752_175241


namespace NUMINAMATH_CALUDE_max_value_of_sum_of_squares_l1752_175246

theorem max_value_of_sum_of_squares (x y : ℝ) (h : 3 * x^2 + 2 * y^2 = 6 * x) :
  ∃ (max : ℝ), max = 1 ∧ ∀ (a b : ℝ), 3 * a^2 + 2 * b^2 = 6 * a → a^2 + b^2 ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_sum_of_squares_l1752_175246


namespace NUMINAMATH_CALUDE_two_true_propositions_l1752_175231

theorem two_true_propositions :
  let prop1 := ∀ a : ℝ, a > -1 → a > -2
  let prop2 := ∀ a : ℝ, a > -2 → a > -1
  let prop3 := ∀ a : ℝ, a ≤ -1 → a ≤ -2
  let prop4 := ∀ a : ℝ, a ≤ -2 → a ≤ -1
  (prop1 ∧ ¬prop2 ∧ ¬prop3 ∧ prop4) :=
by
  sorry

end NUMINAMATH_CALUDE_two_true_propositions_l1752_175231


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l1752_175253

theorem fraction_sum_equality : (20 : ℚ) / 24 + (20 : ℚ) / 25 = 49 / 30 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l1752_175253


namespace NUMINAMATH_CALUDE_f_shifted_positive_set_l1752_175287

/-- An odd function f defined on ℝ satisfying f(x) = 2^x - 4 for x > 0 -/
noncomputable def f : ℝ → ℝ :=
  fun x => if x > 0 then 2^x - 4 else -(2^(-x) - 4)

/-- f is an odd function -/
axiom f_odd : ∀ x, f (-x) = -f x

/-- f(x) = 2^x - 4 for x > 0 -/
axiom f_pos : ∀ x, x > 0 → f x = 2^x - 4

theorem f_shifted_positive_set :
  {x : ℝ | f (x - 1) > 0} = {x : ℝ | -1 < x ∧ x < 1 ∨ x > 3} :=
sorry

end NUMINAMATH_CALUDE_f_shifted_positive_set_l1752_175287


namespace NUMINAMATH_CALUDE_quadratic_maximum_l1752_175279

theorem quadratic_maximum : 
  (∀ r : ℝ, -5 * r^2 + 40 * r - 12 ≤ 68) ∧ 
  (∃ r : ℝ, -5 * r^2 + 40 * r - 12 = 68) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_maximum_l1752_175279


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l1752_175226

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_geo : GeometricSequence a)
  (h_positive : ∀ n, a n > 0)
  (h_sum : a 1 + 2 * a 2 = 4)
  (h_product : (a 4) ^ 2 = 4 * a 3 * a 7) :
  a 5 = 1/8 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l1752_175226


namespace NUMINAMATH_CALUDE_oplus_calculation_l1752_175267

def oplus (x y : ℚ) : ℚ := 1 / (x - y) + y

theorem oplus_calculation :
  (oplus 2 (-3) = -2 - 4/5) ∧
  (oplus (oplus (-4) (-1)) (-5) = -4 - 8/11) := by
  sorry

end NUMINAMATH_CALUDE_oplus_calculation_l1752_175267


namespace NUMINAMATH_CALUDE_not_square_or_cube_l1752_175254

theorem not_square_or_cube (n : ℕ+) :
  ¬ ∃ (k m : ℤ), (n * (n + 1) * (n + 2) * (n + 3) : ℤ) = k^2 ∨
                 (n * (n + 1) * (n + 2) * (n + 3) : ℤ) = m^3 :=
by sorry

end NUMINAMATH_CALUDE_not_square_or_cube_l1752_175254


namespace NUMINAMATH_CALUDE_inequality_proof_l1752_175276

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) 
  (h_prod : a * b * c * d = 1 / 4) :
  (16 * a * c + a / (c^2 * b) + 16 * c / (a^2 * d) + 4 / (a * c)) * 
  (b * d + b / (256 * d^2 * c) + d / (b^2 * a) + 1 / (64 * b * d)) ≥ 81 / 4 ∧
  (16 * a * c + a / (c^2 * b) + 16 * c / (a^2 * d) + 4 / (a * c)) * 
  (b * d + b / (256 * d^2 * c) + d / (b^2 * a) + 1 / (64 * b * d)) = 81 / 4 ↔ 
  a = 2 ∧ b = 1 ∧ c = 1 / 2 ∧ d = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1752_175276
