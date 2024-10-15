import Mathlib

namespace NUMINAMATH_CALUDE_sin_cos_derivative_ratio_l332_33239

theorem sin_cos_derivative_ratio (x : ℝ) (f : ℝ → ℝ) 
  (h1 : f = λ x => Real.sin x + Real.cos x)
  (h2 : deriv f = λ x => 3 * f x) :
  (Real.sin x)^2 - 3 / ((Real.cos x)^2 + 1) = -14/9 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_derivative_ratio_l332_33239


namespace NUMINAMATH_CALUDE_negation_of_forall_exp_positive_l332_33258

theorem negation_of_forall_exp_positive :
  (¬ ∀ x : ℝ, Real.exp x > 0) ↔ (∃ x : ℝ, Real.exp x ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_forall_exp_positive_l332_33258


namespace NUMINAMATH_CALUDE_problem_solution_l332_33238

def f (x : ℝ) : ℝ := |2*x - 1|

theorem problem_solution :
  (∀ x : ℝ, f x < |x| + 1 ↔ 0 < x ∧ x < 2) ∧
  (∀ x y : ℝ, |x - y - 1| ≤ 1/3 ∧ |2*y + 1| ≤ 1/6 → f x < 1) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l332_33238


namespace NUMINAMATH_CALUDE_solution_to_system_l332_33285

theorem solution_to_system : 
  ∀ x y z : ℝ, 
  (4*x*y*z = (x+y)*(x*y+2) ∧ 
   4*x*y*z = (x+z)*(x*z+2) ∧ 
   4*x*y*z = (y+z)*(y*z+2)) → 
  ((x = 0 ∧ y = 0 ∧ z = 0) ∨ 
   (x = Real.sqrt 2 ∧ y = Real.sqrt 2 ∧ z = Real.sqrt 2) ∨ 
   (x = -Real.sqrt 2 ∧ y = -Real.sqrt 2 ∧ z = -Real.sqrt 2)) :=
by sorry

end NUMINAMATH_CALUDE_solution_to_system_l332_33285


namespace NUMINAMATH_CALUDE_real_part_of_complex_square_l332_33246

theorem real_part_of_complex_square : Complex.re ((1 + 2 * Complex.I) ^ 2) = -3 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_complex_square_l332_33246


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l332_33280

-- Define set A
def A : Set ℝ := {x | x^2 - 2*x ≤ 0}

-- Define set B
def B : Set ℝ := {x | ∃ y, y = Real.log (1 - x)}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 ≤ x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l332_33280


namespace NUMINAMATH_CALUDE_y_value_proof_l332_33202

/-- Proves that y = 8 on an equally spaced number line from 0 to 32 with 8 steps,
    where y is 2 steps before the midpoint -/
theorem y_value_proof (total_distance : ℝ) (num_steps : ℕ) (y : ℝ) :
  total_distance = 32 →
  num_steps = 8 →
  y = (total_distance / 2) - 2 * (total_distance / num_steps) →
  y = 8 := by
  sorry

end NUMINAMATH_CALUDE_y_value_proof_l332_33202


namespace NUMINAMATH_CALUDE_solve_stamp_problem_l332_33282

def stamp_problem (initial_stamps final_stamps mike_stamps : ℕ) : Prop :=
  let harry_stamps := final_stamps - initial_stamps - mike_stamps
  harry_stamps - 2 * mike_stamps = 10

theorem solve_stamp_problem :
  stamp_problem 3000 3061 17 := by
  sorry

end NUMINAMATH_CALUDE_solve_stamp_problem_l332_33282


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l332_33291

theorem reciprocal_of_negative_2023 :
  ∃ x : ℚ, x * (-2023) = 1 ∧ x = -1/2023 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l332_33291


namespace NUMINAMATH_CALUDE_or_true_implications_l332_33250

theorem or_true_implications (p q : Prop) (h : p ∨ q) :
  ¬(
    ((p ∧ q) = True) ∨
    ((p ∧ q) = False) ∨
    ((¬p ∨ ¬q) = True) ∨
    ((¬p ∨ ¬q) = False)
  ) := by sorry

end NUMINAMATH_CALUDE_or_true_implications_l332_33250


namespace NUMINAMATH_CALUDE_function_zeros_bound_l332_33209

open Real

theorem function_zeros_bound (m : ℝ) (x₁ x₂ : ℝ) :
  let f := fun (x : ℝ) => (sin x) / (exp x) - x^2 + π * x
  (0 ≤ x₁ ∧ x₁ ≤ π) →
  (0 ≤ x₂ ∧ x₂ ≤ π) →
  f x₁ = m →
  f x₂ = m →
  x₁ ≠ x₂ →
  |x₂ - x₁| ≤ π - (2 * m) / (π + 1) :=
by sorry

end NUMINAMATH_CALUDE_function_zeros_bound_l332_33209


namespace NUMINAMATH_CALUDE_monkey_climb_time_l332_33230

/-- Represents the climbing process of a monkey on a tree. -/
structure MonkeyClimb where
  treeHeight : ℕ  -- Height of the tree in feet
  hopDistance : ℕ  -- Distance the monkey hops up each hour
  slipDistance : ℕ  -- Distance the monkey slips back each hour

/-- Calculates the time taken for the monkey to reach the top of the tree. -/
def timeToReachTop (climb : MonkeyClimb) : ℕ :=
  let netClimbPerHour := climb.hopDistance - climb.slipDistance
  let timeToReachNearTop := (climb.treeHeight - 1) / netClimbPerHour
  timeToReachNearTop + 1

/-- Theorem stating that for the given conditions, the monkey takes 19 hours to reach the top. -/
theorem monkey_climb_time :
  let climb : MonkeyClimb := { treeHeight := 19, hopDistance := 3, slipDistance := 2 }
  timeToReachTop climb = 19 := by
  sorry


end NUMINAMATH_CALUDE_monkey_climb_time_l332_33230


namespace NUMINAMATH_CALUDE_inscribed_quadrilateral_angle_measure_l332_33244

-- Define the circle O
variable (O : ℝ × ℝ)

-- Define the quadrilateral ABCD
variable (A B C D : ℝ × ℝ)

-- Define that ABCD is an inscribed quadrilateral of circle O
def is_inscribed_quadrilateral (O A B C D : ℝ × ℝ) : Prop :=
  sorry

-- Define the angle measure function
def angle_measure (P Q R : ℝ × ℝ) : ℝ :=
  sorry

-- Theorem statement
theorem inscribed_quadrilateral_angle_measure 
  (h_inscribed : is_inscribed_quadrilateral O A B C D)
  (h_ratio : angle_measure B A D / angle_measure B C D = 4 / 5) :
  angle_measure B A D = 80 :=
sorry

end NUMINAMATH_CALUDE_inscribed_quadrilateral_angle_measure_l332_33244


namespace NUMINAMATH_CALUDE_power_equality_l332_33235

theorem power_equality (M : ℕ) : 32^4 * 4^6 = 2^M → M = 32 := by sorry

end NUMINAMATH_CALUDE_power_equality_l332_33235


namespace NUMINAMATH_CALUDE_monomial_exponent_difference_l332_33284

theorem monomial_exponent_difference (a b : ℤ) : 
  ((-1 : ℚ) * X^3 * Y^1 = X^a * Y^(b-1)) → (a - b)^2022 = 1 := by
  sorry

end NUMINAMATH_CALUDE_monomial_exponent_difference_l332_33284


namespace NUMINAMATH_CALUDE_forest_logging_time_l332_33219

/-- Represents a logging team with its characteristics -/
structure LoggingTeam where
  loggers : ℕ
  daysPerWeek : ℕ
  treesPerLoggerPerDay : ℕ

/-- Calculates the number of months needed to cut down all trees in the forest -/
def monthsToLogForest (forestWidth : ℕ) (forestLength : ℕ) (treesPerSquareMile : ℕ) 
  (teams : List LoggingTeam) (daysPerMonth : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that it takes 5 months to log the entire forest -/
theorem forest_logging_time : 
  let forestWidth := 4
  let forestLength := 6
  let treesPerSquareMile := 600
  let teamA := LoggingTeam.mk 6 5 5
  let teamB := LoggingTeam.mk 8 4 6
  let teamC := LoggingTeam.mk 10 3 8
  let teamD := LoggingTeam.mk 12 2 10
  let teams := [teamA, teamB, teamC, teamD]
  let daysPerMonth := 30
  monthsToLogForest forestWidth forestLength treesPerSquareMile teams daysPerMonth = 5 :=
  sorry

end NUMINAMATH_CALUDE_forest_logging_time_l332_33219


namespace NUMINAMATH_CALUDE_lending_problem_l332_33262

/-- 
Proves that if A lends P amount to B at 10% per annum, B lends P to C at 14% per annum, 
and B's gain in 3 years is Rs. 420, then P = 3500.
-/
theorem lending_problem (P : ℝ) : 
  (P * (0.14 - 0.10) * 3 = 420) → P = 3500 := by
  sorry

end NUMINAMATH_CALUDE_lending_problem_l332_33262


namespace NUMINAMATH_CALUDE_arithmetic_and_geometric_sequences_l332_33215

-- Define the arithmetic sequence a_n
def a (n : ℕ) : ℤ := 2*n - 12

-- Define the sum of the geometric sequence b_n
def S (n : ℕ) : ℤ := 4*(1 - 3^n)

theorem arithmetic_and_geometric_sequences :
  -- Conditions for a_n
  (a 3 = -6) ∧ (a 6 = 0) ∧
  -- Arithmetic sequence property
  (∀ n : ℕ, a (n+1) - a n = a (n+2) - a (n+1)) ∧
  -- Conditions for b_n
  (∃ b : ℕ → ℤ, b 1 = -8 ∧ b 2 = a 1 + a 2 + a 3 ∧
  -- Geometric sequence property
  (∀ n : ℕ, n ≥ 1 → b (n+1) / b n = b 2 / b 1) ∧
  -- S_n is the sum of the first n terms of b_n
  (∀ n : ℕ, n ≥ 1 → S n = (b 1) * (1 - (b 2 / b 1)^n) / (1 - b 2 / b 1))) := by
  sorry

#check arithmetic_and_geometric_sequences

end NUMINAMATH_CALUDE_arithmetic_and_geometric_sequences_l332_33215


namespace NUMINAMATH_CALUDE_cherry_tomato_jars_l332_33288

theorem cherry_tomato_jars (total_tomatoes : ℕ) (tomatoes_per_jar : ℕ) (h1 : total_tomatoes = 56) (h2 : tomatoes_per_jar = 8) :
  (total_tomatoes / tomatoes_per_jar : ℕ) = 7 := by
  sorry

end NUMINAMATH_CALUDE_cherry_tomato_jars_l332_33288


namespace NUMINAMATH_CALUDE_questions_to_complete_l332_33278

/-- Calculates the number of questions Sasha still needs to complete -/
theorem questions_to_complete 
  (rate : ℕ)        -- Questions completed per hour
  (total : ℕ)       -- Total questions to complete
  (time_worked : ℕ) -- Hours worked
  (h1 : rate = 15)  -- Sasha's rate is 15 questions per hour
  (h2 : total = 60) -- Total questions is 60
  (h3 : time_worked = 2) -- Time worked is 2 hours
  : total - (rate * time_worked) = 30 :=
by sorry

end NUMINAMATH_CALUDE_questions_to_complete_l332_33278


namespace NUMINAMATH_CALUDE_math_exam_problem_l332_33236

theorem math_exam_problem (total : ℕ) (correct : ℕ) (incorrect : ℕ) :
  total = 120 →
  incorrect = 3 * correct →
  total = correct + incorrect →
  correct = 30 := by
sorry

end NUMINAMATH_CALUDE_math_exam_problem_l332_33236


namespace NUMINAMATH_CALUDE_red_highest_probability_l332_33216

def num_red : ℕ := 5
def num_yellow : ℕ := 4
def num_white : ℕ := 1
def num_blue : ℕ := 3

def total_balls : ℕ := num_red + num_yellow + num_white + num_blue

theorem red_highest_probability :
  (num_red : ℚ) / total_balls > max ((num_yellow : ℚ) / total_balls)
                                    (max ((num_white : ℚ) / total_balls)
                                         ((num_blue : ℚ) / total_balls)) :=
by sorry

end NUMINAMATH_CALUDE_red_highest_probability_l332_33216


namespace NUMINAMATH_CALUDE_africa_fraction_proof_l332_33276

def total_passengers : ℕ := 96

def north_america_fraction : ℚ := 1/4
def europe_fraction : ℚ := 1/8
def asia_fraction : ℚ := 1/6
def other_continents : ℕ := 36

theorem africa_fraction_proof :
  ∃ (africa_fraction : ℚ),
    africa_fraction * total_passengers +
    north_america_fraction * total_passengers +
    europe_fraction * total_passengers +
    asia_fraction * total_passengers +
    other_continents = total_passengers ∧
    africa_fraction = 1/12 :=
by sorry

end NUMINAMATH_CALUDE_africa_fraction_proof_l332_33276


namespace NUMINAMATH_CALUDE_z_curve_not_simple_conic_l332_33266

-- Define the complex number z
variable (z : ℂ)

-- Define the condition |z - 1/z| = 1
def condition (z : ℂ) : Prop := Complex.abs (z - z⁻¹) = 1

-- Define the curves we want to exclude
def is_ellipse (curve : ℂ → Prop) : Prop := sorry
def is_parabola (curve : ℂ → Prop) : Prop := sorry
def is_hyperbola (curve : ℂ → Prop) : Prop := sorry

-- Define the curve traced by z
def z_curve (z : ℂ) : Prop := condition z

-- State the theorem
theorem z_curve_not_simple_conic (z : ℂ) :
  condition z →
  ¬(is_ellipse z_curve ∨ is_parabola z_curve ∨ is_hyperbola z_curve) :=
sorry

end NUMINAMATH_CALUDE_z_curve_not_simple_conic_l332_33266


namespace NUMINAMATH_CALUDE_complex_simplification_l332_33257

/-- The imaginary unit -/
def i : ℂ := Complex.I

/-- Theorem stating that the given complex expression simplifies to 5 -/
theorem complex_simplification : 2 * (3 - i) + i * (2 + i) = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_simplification_l332_33257


namespace NUMINAMATH_CALUDE_three_digit_equation_solution_l332_33237

/-- Given that 3 + 6AB = 691 and 6AB is a three-digit number, prove that A = 8 -/
theorem three_digit_equation_solution (A B : ℕ) : 
  (3 + 6 * A * 10 + B = 691) → 
  (100 ≤ 6 * A * 10 + B) →
  (6 * A * 10 + B < 1000) →
  A = 8 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_equation_solution_l332_33237


namespace NUMINAMATH_CALUDE_cubic_sum_of_quadratic_roots_l332_33242

theorem cubic_sum_of_quadratic_roots : 
  ∀ a b : ℝ, 
  (3 * a^2 - 5 * a + 7 = 0) → 
  (3 * b^2 - 5 * b + 7 = 0) → 
  (a ≠ b) →
  (a^3 / b^3 + b^3 / a^3 = -190 / 343) := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_of_quadratic_roots_l332_33242


namespace NUMINAMATH_CALUDE_always_two_real_roots_find_p_l332_33265

-- Define the quadratic equation
def quadratic_equation (x p : ℝ) : Prop :=
  (x - 3) * (x - 2) = p * (p + 1)

-- Theorem 1: The equation always has two real roots
theorem always_two_real_roots (p : ℝ) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation x₁ p ∧ quadratic_equation x₂ p :=
sorry

-- Theorem 2: If the roots satisfy the given condition, then p = -2
theorem find_p (p : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : quadratic_equation x₁ p)
  (h₂ : quadratic_equation x₂ p)
  (h₃ : x₁^2 + x₂^2 - x₁*x₂ = 3*p^2 + 1) :
  p = -2 :=
sorry

end NUMINAMATH_CALUDE_always_two_real_roots_find_p_l332_33265


namespace NUMINAMATH_CALUDE_heels_cost_equals_savings_plus_contribution_l332_33251

/-- The cost of heels Miranda wants to buy -/
def heels_cost : ℕ := 260

/-- The number of months Miranda saved money -/
def months_saved : ℕ := 3

/-- The amount Miranda saved per month -/
def savings_per_month : ℕ := 70

/-- The amount Miranda's sister contributed -/
def sister_contribution : ℕ := 50

/-- Theorem stating that the cost of the heels is equal to Miranda's total savings plus her sister's contribution -/
theorem heels_cost_equals_savings_plus_contribution :
  heels_cost = months_saved * savings_per_month + sister_contribution :=
by sorry

end NUMINAMATH_CALUDE_heels_cost_equals_savings_plus_contribution_l332_33251


namespace NUMINAMATH_CALUDE_inequality_solution_l332_33295

theorem inequality_solution (y : ℝ) : 
  (y^2 + y^3 - 3*y^4) / (y + y^2 - 3*y^3) ≥ -1 ↔ 
  y ∈ Set.Icc (-1) (-4/3) ∪ Set.Ioo (-4/3) 0 ∪ Set.Ioo 0 1 ∪ Set.Ioi 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l332_33295


namespace NUMINAMATH_CALUDE_expand_and_compare_coefficients_l332_33264

theorem expand_and_compare_coefficients (m n : ℤ) : 
  (∀ x : ℤ, (x + 4) * (x - 2) = x^2 + m*x + n) → m = 2 ∧ n = -8 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_compare_coefficients_l332_33264


namespace NUMINAMATH_CALUDE_eighth_term_geometric_sequence_l332_33253

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The 8th term of a geometric sequence given the 4th and 6th terms -/
theorem eighth_term_geometric_sequence (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) 
  (h_4 : a 4 = 7)
  (h_6 : a 6 = 21) : 
  a 8 = 63 := by
sorry

end NUMINAMATH_CALUDE_eighth_term_geometric_sequence_l332_33253


namespace NUMINAMATH_CALUDE_elephant_park_problem_l332_33298

theorem elephant_park_problem (initial_elephants : ℕ) (exodus_duration : ℕ) (exodus_rate : ℕ) 
  (entry_period : ℕ) (final_elephants : ℕ) : 
  initial_elephants = 30000 →
  exodus_duration = 4 →
  exodus_rate = 2880 →
  entry_period = 7 →
  final_elephants = 28980 →
  (final_elephants - (initial_elephants - exodus_duration * exodus_rate)) / entry_period = 1500 := by
sorry

end NUMINAMATH_CALUDE_elephant_park_problem_l332_33298


namespace NUMINAMATH_CALUDE_hot_dogs_leftover_l332_33273

theorem hot_dogs_leftover : 20146130 % 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_hot_dogs_leftover_l332_33273


namespace NUMINAMATH_CALUDE_triangle_side_length_l332_33283

/-- Given a triangle ABC with angle A = 60°, side b = 8, and area = 12√3,
    prove that side a = 2√13 -/
theorem triangle_side_length (A B C : ℝ) (h_angle : A = 60 * π / 180)
    (h_side_b : B = 8) (h_area : (1/2) * B * C * Real.sin A = 12 * Real.sqrt 3) :
    A = 2 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l332_33283


namespace NUMINAMATH_CALUDE_andy_sock_ratio_l332_33248

/-- The ratio of white socks to black socks -/
def sock_ratio (white : ℕ) (black : ℕ) : ℚ := white / black

theorem andy_sock_ratio :
  ∀ white : ℕ,
  let black := 6
  white / 2 = black + 6 →
  sock_ratio white black = 4 / 1 := by
sorry

end NUMINAMATH_CALUDE_andy_sock_ratio_l332_33248


namespace NUMINAMATH_CALUDE_grade_assignment_count_l332_33220

/-- The number of different grades that can be assigned to each student. -/
def numGrades : ℕ := 4

/-- The number of students in the class. -/
def numStudents : ℕ := 15

/-- The theorem stating the number of ways to assign grades to students. -/
theorem grade_assignment_count : numGrades ^ numStudents = 1073741824 := by
  sorry

end NUMINAMATH_CALUDE_grade_assignment_count_l332_33220


namespace NUMINAMATH_CALUDE_roses_in_vase_l332_33290

theorem roses_in_vase (initial_roses : ℕ) (added_roses : ℕ) (total_roses : ℕ) : 
  added_roses = 11 → total_roses = 14 → initial_roses = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_roses_in_vase_l332_33290


namespace NUMINAMATH_CALUDE_joan_makes_ten_ham_sandwiches_l332_33241

/-- Represents the number of slices of cheese required for each type of sandwich -/
structure SandwichRecipe where
  ham_cheese_slices : ℕ
  grilled_cheese_slices : ℕ

/-- Represents the sandwich making scenario -/
structure SandwichScenario where
  recipe : SandwichRecipe
  total_cheese_slices : ℕ
  grilled_cheese_count : ℕ

/-- Calculates the number of ham sandwiches made -/
def ham_sandwiches_made (scenario : SandwichScenario) : ℕ :=
  (scenario.total_cheese_slices - scenario.grilled_cheese_count * scenario.recipe.grilled_cheese_slices) / scenario.recipe.ham_cheese_slices

/-- Theorem stating that Joan makes 10 ham sandwiches -/
theorem joan_makes_ten_ham_sandwiches (scenario : SandwichScenario) 
  (h1 : scenario.recipe.ham_cheese_slices = 2)
  (h2 : scenario.recipe.grilled_cheese_slices = 3)
  (h3 : scenario.total_cheese_slices = 50)
  (h4 : scenario.grilled_cheese_count = 10) :
  ham_sandwiches_made scenario = 10 := by
  sorry

#eval ham_sandwiches_made { 
  recipe := { ham_cheese_slices := 2, grilled_cheese_slices := 3 },
  total_cheese_slices := 50,
  grilled_cheese_count := 10
}

end NUMINAMATH_CALUDE_joan_makes_ten_ham_sandwiches_l332_33241


namespace NUMINAMATH_CALUDE_imaginary_part_of_one_minus_i_squared_l332_33259

theorem imaginary_part_of_one_minus_i_squared :
  Complex.im ((1 - Complex.I) ^ 2) = -2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_one_minus_i_squared_l332_33259


namespace NUMINAMATH_CALUDE_complex_sum_zero_l332_33260

theorem complex_sum_zero (z : ℂ) (h : z = Complex.exp (6 * Real.pi * I / 11)) :
  z^2 / (1 + z^3) + z^4 / (1 + z^6) + z^5 / (1 + z^9) = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_zero_l332_33260


namespace NUMINAMATH_CALUDE_two_books_from_different_genres_l332_33269

/-- Represents the number of books in each genre -/
def booksPerGenre : Nat := 4

/-- Represents the number of genres -/
def numGenres : Nat := 3

/-- Theorem: The number of ways to select two books from different genres 
    given three genres with four books each is 48 -/
theorem two_books_from_different_genres :
  (booksPerGenre * booksPerGenre * (numGenres * (numGenres - 1) / 2)) = 48 := by
  sorry

end NUMINAMATH_CALUDE_two_books_from_different_genres_l332_33269


namespace NUMINAMATH_CALUDE_smallest_n_for_array_formation_l332_33267

theorem smallest_n_for_array_formation : 
  ∃ n k : ℕ+, 
    (∀ m k' : ℕ+, 8 * m = 225 * k' + 3 → n ≤ m) ∧ 
    (8 * n = 225 * k + 3) ∧
    n = 141 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_array_formation_l332_33267


namespace NUMINAMATH_CALUDE_count_ones_in_500_pages_l332_33207

/-- Count the occurrences of digit 1 in a number -/
def countOnesInNumber (n : ℕ) : ℕ := sorry

/-- Count the occurrences of digit 1 in page numbers from 1 to n -/
def countOnesInPages (n : ℕ) : ℕ :=
  (List.range n).map countOnesInNumber |>.sum

theorem count_ones_in_500_pages :
  countOnesInPages 500 = 200 := by sorry

end NUMINAMATH_CALUDE_count_ones_in_500_pages_l332_33207


namespace NUMINAMATH_CALUDE_seed_fertilizer_ratio_is_three_to_one_l332_33274

/-- Given a total amount of seed and fertilizer, and the amount of seed,
    calculate the ratio of seed to fertilizer. -/
def seedFertilizerRatio (total : ℚ) (seed : ℚ) : ℚ :=
  seed / (total - seed)

/-- Theorem stating that given 60 gallons total and 45 gallons of seed,
    the ratio of seed to fertilizer is 3:1. -/
theorem seed_fertilizer_ratio_is_three_to_one :
  seedFertilizerRatio 60 45 = 3 := by
  sorry

#eval seedFertilizerRatio 60 45

end NUMINAMATH_CALUDE_seed_fertilizer_ratio_is_three_to_one_l332_33274


namespace NUMINAMATH_CALUDE_colleen_pays_more_than_joy_l332_33255

/-- Calculates the difference in cost between Colleen's and Joy's pencils -/
def pencil_cost_difference (joy_pencils colleen_pencils pencil_price : ℕ) : ℕ :=
  colleen_pencils * pencil_price - joy_pencils * pencil_price

theorem colleen_pays_more_than_joy :
  pencil_cost_difference 30 50 4 = 80 := by
  sorry

end NUMINAMATH_CALUDE_colleen_pays_more_than_joy_l332_33255


namespace NUMINAMATH_CALUDE_amy_math_problems_l332_33221

/-- The number of math problems Amy had to solve -/
def math_problems : ℕ := sorry

/-- The number of spelling problems Amy had to solve -/
def spelling_problems : ℕ := 6

/-- The number of problems Amy can finish in an hour -/
def problems_per_hour : ℕ := 4

/-- The number of hours it took Amy to finish all problems -/
def total_hours : ℕ := 6

/-- Theorem stating that Amy had 18 math problems -/
theorem amy_math_problems : 
  math_problems = 18 := by sorry

end NUMINAMATH_CALUDE_amy_math_problems_l332_33221


namespace NUMINAMATH_CALUDE_cylinder_minus_cones_volume_l332_33234

/-- The volume of a cylinder minus the volume of three congruent cones -/
theorem cylinder_minus_cones_volume (r h : ℝ) (hr : r = 8) (hh : h = 24) :
  π * r^2 * h - 3 * (1/3 * π * r^2 * (h/3)) = 1024 * π := by
  sorry

end NUMINAMATH_CALUDE_cylinder_minus_cones_volume_l332_33234


namespace NUMINAMATH_CALUDE_newspaper_cost_difference_l332_33211

/-- Calculates the annual cost difference between Juanita's newspaper purchases and Grant's subscription --/
theorem newspaper_cost_difference : 
  let grant_base_cost : ℝ := 200
  let grant_loyalty_discount : ℝ := 0.1
  let grant_summer_discount : ℝ := 0.05
  let juanita_mon_wed_price : ℝ := 0.5
  let juanita_thu_fri_price : ℝ := 0.6
  let juanita_sat_price : ℝ := 0.8
  let juanita_sun_price : ℝ := 3
  let juanita_monthly_coupon : ℝ := 0.25
  let juanita_holiday_surcharge : ℝ := 0.5
  let weeks_per_year : ℕ := 52
  let months_per_year : ℕ := 12
  let summer_months : ℕ := 2

  let grant_annual_cost := grant_base_cost * (1 - grant_loyalty_discount) - 
    (grant_base_cost / months_per_year) * summer_months * grant_summer_discount

  let juanita_weekly_cost := 3 * juanita_mon_wed_price + 2 * juanita_thu_fri_price + 
    juanita_sat_price + juanita_sun_price

  let juanita_annual_cost := juanita_weekly_cost * weeks_per_year - 
    juanita_monthly_coupon * months_per_year + juanita_holiday_surcharge * months_per_year

  juanita_annual_cost - grant_annual_cost = 162.5 := by sorry

end NUMINAMATH_CALUDE_newspaper_cost_difference_l332_33211


namespace NUMINAMATH_CALUDE_inequality_system_solution_l332_33213

theorem inequality_system_solution : 
  {x : ℤ | x > 0 ∧ 
           (1 + 2*x : ℚ)/4 - (1 - 3*x)/10 > -1/5 ∧ 
           (3*x - 1 : ℚ) < 2*(x + 1)} = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l332_33213


namespace NUMINAMATH_CALUDE_sequence_a_formula_l332_33201

def sequence_a (n : ℕ) : ℚ :=
  if n = 1 then -1
  else 2 / (n * (n + 1))

def S (n : ℕ) : ℚ := -2 / (n + 1)

theorem sequence_a_formula (n : ℕ) :
  (n = 1 ∧ sequence_a n = -1) ∨
  (n ≥ 2 ∧ sequence_a n = 2 / (n * (n + 1))) ∧
  (∀ k ≥ 2, (S k)^2 - (sequence_a k) * (S k) = 2 * (sequence_a k)) :=
sorry

end NUMINAMATH_CALUDE_sequence_a_formula_l332_33201


namespace NUMINAMATH_CALUDE_division_into_proportional_parts_l332_33208

theorem division_into_proportional_parts :
  let total : ℚ := 156
  let proportions : List ℚ := [2, 1/2, 1/4, 1/8]
  let parts := proportions.map (λ p => p * (total / proportions.sum))
  parts[2] = 13 + 15/23 := by sorry

end NUMINAMATH_CALUDE_division_into_proportional_parts_l332_33208


namespace NUMINAMATH_CALUDE_odd_function_sum_l332_33227

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_sum (f : ℝ → ℝ) (h_odd : IsOdd f) (h_f_neg_one : f (-1) = 2) :
  f 0 + f 1 = -2 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_sum_l332_33227


namespace NUMINAMATH_CALUDE_even_function_symmetry_l332_33231

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- State the theorem
theorem even_function_symmetry (f : ℝ → ℝ) (h : EvenFunction (fun x ↦ f (x + 1))) :
  ∀ x, f (1 + x) = f (1 - x) := by
  sorry

end NUMINAMATH_CALUDE_even_function_symmetry_l332_33231


namespace NUMINAMATH_CALUDE_smallest_valid_distribution_l332_33206

/-- Represents a distribution of candy pieces to children in a circle. -/
def CandyDistribution := List Nat

/-- Checks if all elements in the list are distinct. -/
def all_distinct (l : List Nat) : Prop :=
  l.Nodup

/-- Checks if all elements in the list are at least 1. -/
def all_at_least_one (l : List Nat) : Prop :=
  ∀ x ∈ l, x ≥ 1

/-- Checks if adjacent elements (including the first and last) have a common factor other than 1. -/
def adjacent_common_factor (l : List Nat) : Prop :=
  ∀ i, ∃ k > 1, k ∣ (l.get! i) ∧ k ∣ (l.get! ((i + 1) % l.length))

/-- Checks if there is no prime that divides all elements in the list. -/
def no_common_prime_divisor (l : List Nat) : Prop :=
  ¬∃ p, Nat.Prime p ∧ ∀ x ∈ l, p ∣ x

/-- Checks if a candy distribution satisfies all conditions. -/
def valid_distribution (d : CandyDistribution) : Prop :=
  d.length = 7 ∧
  all_distinct d ∧
  all_at_least_one d ∧
  adjacent_common_factor d ∧
  no_common_prime_divisor d

/-- The main theorem stating that 44 is the smallest number of candy pieces
    that satisfies all conditions for seven children. -/
theorem smallest_valid_distribution :
  (∃ d : CandyDistribution, valid_distribution d ∧ d.sum = 44) ∧
  (∀ d : CandyDistribution, valid_distribution d → d.sum ≥ 44) :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_distribution_l332_33206


namespace NUMINAMATH_CALUDE_a_work_days_proof_a_work_days_unique_l332_33240

-- Define the work rates and completion times
def total_work : ℝ := 1 -- Normalize total work to 1
def a_completion_time : ℝ := 15
def b_completion_time : ℝ := 26.999999999999996
def b_remaining_time : ℝ := 18

-- Define A's work days as a variable
def a_work_days : ℝ := 5 -- The value we want to prove

-- Theorem statement
theorem a_work_days_proof :
  (total_work / a_completion_time) * a_work_days +
  (total_work / b_completion_time) * b_remaining_time = total_work :=
by
  sorry -- Proof is omitted as per instructions

-- Additional theorem to show that this solution is unique
theorem a_work_days_unique (x : ℝ) :
  (total_work / a_completion_time) * x +
  (total_work / b_completion_time) * b_remaining_time = total_work →
  x = a_work_days :=
by
  sorry -- Proof is omitted as per instructions

end NUMINAMATH_CALUDE_a_work_days_proof_a_work_days_unique_l332_33240


namespace NUMINAMATH_CALUDE_abcd_equality_l332_33243

theorem abcd_equality (a b c d : ℝ) 
  (h1 : a ≥ b ∧ b ≥ c ∧ c ≥ d ∧ d ≥ 0)
  (h2 : a^2 + d^2 = 1)
  (h3 : b^2 + c^2 = 1)
  (h4 : a*c + b*d = 1/3) :
  a*b - c*d = 2*Real.sqrt 2/3 := by
sorry

end NUMINAMATH_CALUDE_abcd_equality_l332_33243


namespace NUMINAMATH_CALUDE_c4h1o_molecular_weight_l332_33200

/-- The atomic weight of carbon in atomic mass units (amu) -/
def carbon_weight : ℝ := 12.01

/-- The atomic weight of hydrogen in atomic mass units (amu) -/
def hydrogen_weight : ℝ := 1.008

/-- The atomic weight of oxygen in atomic mass units (amu) -/
def oxygen_weight : ℝ := 16.00

/-- The number of carbon atoms in the compound -/
def carbon_count : ℕ := 4

/-- The number of hydrogen atoms in the compound -/
def hydrogen_count : ℕ := 1

/-- The number of oxygen atoms in the compound -/
def oxygen_count : ℕ := 1

/-- The molecular weight of C4H1O -/
def molecular_weight : ℝ :=
  carbon_weight * carbon_count + hydrogen_weight * hydrogen_count + oxygen_weight * oxygen_count

theorem c4h1o_molecular_weight :
  molecular_weight = 65.048 := by sorry

end NUMINAMATH_CALUDE_c4h1o_molecular_weight_l332_33200


namespace NUMINAMATH_CALUDE_ace_king_queen_probability_l332_33254

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Nat)
  (aces : Nat)
  (kings : Nat)
  (queens : Nat)

/-- The probability of drawing a specific card from a deck -/
def drawProbability (n : Nat) (total : Nat) : ℚ :=
  n / total

/-- The standard 52-card deck -/
def standardDeck : Deck :=
  { cards := 52, aces := 4, kings := 4, queens := 4 }

theorem ace_king_queen_probability :
  let d := standardDeck
  let p1 := drawProbability d.aces d.cards
  let p2 := drawProbability d.kings (d.cards - 1)
  let p3 := drawProbability d.queens (d.cards - 2)
  p1 * p2 * p3 = 8 / 16575 := by
  sorry

end NUMINAMATH_CALUDE_ace_king_queen_probability_l332_33254


namespace NUMINAMATH_CALUDE_sequence_inequality_l332_33292

theorem sequence_inequality (a : Fin 9 → ℝ) 
  (h_nonneg : ∀ i, 0 ≤ a i) 
  (h_first : a 0 = 0) 
  (h_last : a 8 = 0) 
  (h_nonzero : ∃ i, a i ≠ 0) : 
  (∃ i : Fin 9, 1 < i.val ∧ i.val < 9 ∧ a (i - 1) + a (i + 1) < 2 * a i) ∧
  (∃ i : Fin 9, 1 < i.val ∧ i.val < 9 ∧ a (i - 1) + a (i + 1) < 1.9 * a i) :=
sorry

end NUMINAMATH_CALUDE_sequence_inequality_l332_33292


namespace NUMINAMATH_CALUDE_inscribed_right_triangle_diameter_l332_33281

/-- Given a right triangle inscribed in a circle with legs of lengths 6 and 8,
    the diameter of the circle is 10. -/
theorem inscribed_right_triangle_diameter :
  ∀ (circle : Real → Real → Prop) (triangle : Real → Real → Real → Prop),
    (∃ (x y z : Real), triangle x y z ∧ x^2 + y^2 = z^2) →  -- Right triangle condition
    (∃ (a b : Real), triangle 6 8 a) →  -- Leg lengths condition
    (∀ (p q r : Real), triangle p q r → circle p q) →  -- Triangle inscribed in circle
    (∃ (d : Real), d = 10 ∧ ∀ (p q : Real), circle p q → (p - q)^2 ≤ d^2) :=
by sorry

end NUMINAMATH_CALUDE_inscribed_right_triangle_diameter_l332_33281


namespace NUMINAMATH_CALUDE_taylor_family_reunion_l332_33247

theorem taylor_family_reunion (kids : ℕ) (adults : ℕ) (tables : ℕ) 
  (h1 : kids = 45)
  (h2 : adults = 123)
  (h3 : tables = 14)
  : (kids + adults) / tables = 12 := by
  sorry

end NUMINAMATH_CALUDE_taylor_family_reunion_l332_33247


namespace NUMINAMATH_CALUDE_transformed_quadratic_equation_l332_33218

theorem transformed_quadratic_equation 
  (p q : ℝ) 
  (x₁ x₂ : ℝ) 
  (h1 : x₁^2 + p*x₁ + q = 0) 
  (h2 : x₂^2 + p*x₂ + q = 0) 
  (h3 : x₁ ≠ x₂) :
  ∃ (t : ℝ), q*t^2 + (q+1)*t + 1 = 0 ↔ 
    (t = x₁ + 1/x₁ ∨ t = x₂ + 1/x₂) :=
by sorry

end NUMINAMATH_CALUDE_transformed_quadratic_equation_l332_33218


namespace NUMINAMATH_CALUDE_negation_of_existence_l332_33296

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, x > 0 ∧ P x) ↔ (∀ x : ℝ, x > 0 → ¬ P x) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_l332_33296


namespace NUMINAMATH_CALUDE_bus_students_count_l332_33275

/-- Calculates the number of students on the bus after all stops -/
def final_students (initial : ℕ) (second_on second_off third_on third_off : ℕ) : ℕ :=
  initial + second_on - second_off + third_on - third_off

/-- Theorem stating the final number of students on the bus -/
theorem bus_students_count :
  final_students 39 29 12 35 18 = 73 := by
  sorry

end NUMINAMATH_CALUDE_bus_students_count_l332_33275


namespace NUMINAMATH_CALUDE_difference_greater_than_twice_l332_33224

theorem difference_greater_than_twice (a : ℝ) : 
  (∀ x, x - 5 > 2*x ↔ x = a) ↔ a - 5 > 2*a := by sorry

end NUMINAMATH_CALUDE_difference_greater_than_twice_l332_33224


namespace NUMINAMATH_CALUDE_range_of_m_l332_33287

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x + 5

-- Define the interval [2, 4]
def I : Set ℝ := {x | 2 ≤ x ∧ x ≤ 4}

-- State the theorem
theorem range_of_m (m : ℝ) : 
  (∃ x ∈ I, m - f x > 0) → m > 5 := by sorry

end NUMINAMATH_CALUDE_range_of_m_l332_33287


namespace NUMINAMATH_CALUDE_shortest_chord_length_l332_33245

/-- The circle with equation x^2 + y^2 - 2x - 4y + 1 = 0 -/
def circle1 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 2*p.1 - 4*p.2 + 1 = 0}

/-- The center of circle1 -/
def center1 : ℝ × ℝ := (1, 2)

/-- The line of symmetry for circle1 -/
def l : Set (ℝ × ℝ) := {p : ℝ × ℝ | ∃ t : ℝ, p = t • center1}

/-- The circle with center (0,0) and radius 3 -/
def circle2 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = 9}

/-- The shortest chord length theorem -/
theorem shortest_chord_length :
  ∃ (A B : ℝ × ℝ), A ∈ circle2 ∧ B ∈ circle2 ∧ A ∈ l ∧ B ∈ l ∧
    ∀ (C D : ℝ × ℝ), C ∈ circle2 → D ∈ circle2 → C ∈ l → D ∈ l →
      Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) ≤ Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4 :=
sorry

end NUMINAMATH_CALUDE_shortest_chord_length_l332_33245


namespace NUMINAMATH_CALUDE_range_of_m_l332_33214

def p (x : ℝ) : Prop := |1 - (x - 1) / 3| ≤ 2

def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

theorem range_of_m (m : ℝ) :
  (m > 0) →
  (∀ x, ¬(p x) → ¬(q x m)) →
  (∃ x, ¬(p x) ∧ (q x m)) →
  m ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l332_33214


namespace NUMINAMATH_CALUDE_number_of_book_combinations_l332_33272

-- Define the number of books and the number to choose
def total_books : ℕ := 15
def books_to_choose : ℕ := 3

-- Theorem statement
theorem number_of_book_combinations :
  Nat.choose total_books books_to_choose = 455 := by
  sorry

end NUMINAMATH_CALUDE_number_of_book_combinations_l332_33272


namespace NUMINAMATH_CALUDE_rabbit_average_distance_l332_33299

/-- A square with side length 8 meters -/
def square_side : ℝ := 8

/-- The x-coordinate of the rabbit's final position -/
def rabbit_x : ℝ := 6.4

/-- The y-coordinate of the rabbit's final position -/
def rabbit_y : ℝ := 2.4

/-- The average distance from the rabbit to the sides of the square -/
def average_distance : ℝ := 4

theorem rabbit_average_distance :
  let distances : List ℝ := [
    rabbit_x,  -- distance to left side
    rabbit_y,  -- distance to bottom side
    square_side - rabbit_x,  -- distance to right side
    square_side - rabbit_y   -- distance to top side
  ]
  (distances.sum / distances.length : ℝ) = average_distance := by
  sorry

end NUMINAMATH_CALUDE_rabbit_average_distance_l332_33299


namespace NUMINAMATH_CALUDE_pencil_eraser_cost_problem_l332_33212

theorem pencil_eraser_cost_problem : ∃ (p e : ℕ), 
  13 * p + 3 * e = 100 ∧ 
  p > e ∧ 
  p + e = 10 := by
sorry

end NUMINAMATH_CALUDE_pencil_eraser_cost_problem_l332_33212


namespace NUMINAMATH_CALUDE_strawberry_milk_probability_l332_33286

theorem strawberry_milk_probability : 
  let n : ℕ := 7  -- number of trials
  let k : ℕ := 5  -- number of successes
  let p : ℚ := 3/4  -- probability of success in each trial
  Nat.choose n k * p^k * (1-p)^(n-k) = 5103/16384 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_milk_probability_l332_33286


namespace NUMINAMATH_CALUDE_data_average_is_four_l332_33289

def data : List ℝ := [6, 3, 3, 5, 1]

def isMode (x : ℝ) (l : List ℝ) : Prop :=
  ∀ y ∈ l, (l.count x ≥ l.count y)

theorem data_average_is_four (x : ℝ) (h1 : isMode 3 (x::data)) (h2 : isMode 6 (x::data)) :
  (x::data).sum / (x::data).length = 4 := by
  sorry

end NUMINAMATH_CALUDE_data_average_is_four_l332_33289


namespace NUMINAMATH_CALUDE_total_handshakes_at_gathering_l332_33270

def number_of_couples : ℕ := 15

def men_handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

def men_women_handshakes (n : ℕ) : ℕ := n * (n - 1)

def women_subset_handshakes : ℕ := 3

theorem total_handshakes_at_gathering :
  men_handshakes number_of_couples +
  men_women_handshakes number_of_couples +
  women_subset_handshakes = 318 := by sorry

end NUMINAMATH_CALUDE_total_handshakes_at_gathering_l332_33270


namespace NUMINAMATH_CALUDE_consecutive_even_numbers_sum_l332_33271

theorem consecutive_even_numbers_sum (x : ℤ) : 
  (x % 2 = 0) →                   -- x is even
  (x + (x + 2) + (x + 4) = 1194) →  -- sum of three consecutive even numbers is 1194
  x = 396 :=                      -- the first even number is 396
by
  sorry

end NUMINAMATH_CALUDE_consecutive_even_numbers_sum_l332_33271


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l332_33225

theorem unique_quadratic_solution (a : ℝ) :
  (∃! x : ℝ, a * x^2 + a * x + 1 = 0) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l332_33225


namespace NUMINAMATH_CALUDE_extreme_values_and_tangent_lines_l332_33233

-- Define the function f(x)
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + 1

-- Define the derivative of f(x)
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

-- Theorem statement
theorem extreme_values_and_tangent_lines :
  ∃ (a b : ℝ),
    (f' a b (-2/3) = 0 ∧ f' a b 1 = 0) ∧
    (a = -1/2 ∧ b = -2) ∧
    (∃ (t₁ t₂ : ℝ),
      (f a b t₁ - 1 = (f' a b t₁) * (-t₁) ∧ 2*t₁ + (f a b t₁) - 1 = 0) ∧
      (f a b t₂ - 1 = (f' a b t₂) * (-t₂) ∧ 33*t₂ + 16*(f a b t₂) - 16 = 0)) :=
by sorry

end NUMINAMATH_CALUDE_extreme_values_and_tangent_lines_l332_33233


namespace NUMINAMATH_CALUDE_blue_hat_cost_l332_33203

/-- Proves that the cost of each blue hat is $6 given the conditions of the hat purchase problem. -/
theorem blue_hat_cost (total_hats : ℕ) (green_hat_cost : ℕ) (total_price : ℕ) (green_hats : ℕ) :
  total_hats = 85 →
  green_hat_cost = 7 →
  total_price = 530 →
  green_hats = 20 →
  (total_price - green_hat_cost * green_hats) / (total_hats - green_hats) = 6 := by
sorry

end NUMINAMATH_CALUDE_blue_hat_cost_l332_33203


namespace NUMINAMATH_CALUDE_puppy_price_is_five_l332_33226

/-- The price of a kitten in dollars -/
def kitten_price : ℕ := 6

/-- The number of kittens sold -/
def kittens_sold : ℕ := 2

/-- The total earnings from all pets sold in dollars -/
def total_earnings : ℕ := 17

/-- The price of the puppy in dollars -/
def puppy_price : ℕ := total_earnings - (kitten_price * kittens_sold)

theorem puppy_price_is_five : puppy_price = 5 := by
  sorry

end NUMINAMATH_CALUDE_puppy_price_is_five_l332_33226


namespace NUMINAMATH_CALUDE_rectangular_box_volume_l332_33263

/-- The volume of a rectangular box with face areas 36, 18, and 8 square inches is 72 cubic inches. -/
theorem rectangular_box_volume (l w h : ℝ) 
  (area1 : l * w = 36)
  (area2 : w * h = 18)
  (area3 : l * h = 8) :
  l * w * h = 72 := by sorry

end NUMINAMATH_CALUDE_rectangular_box_volume_l332_33263


namespace NUMINAMATH_CALUDE_dividend_problem_l332_33279

/-- Given a total amount of 585 to be divided among three people (a, b, c) such that
    4 times a's share equals 6 times b's share, which equals 3 times c's share,
    prove that c's share is equal to 135. -/
theorem dividend_problem (total : ℕ) (a b c : ℚ) 
    (h_total : total = 585)
    (h_sum : a + b + c = total)
    (h_prop : (4 * a = 6 * b) ∧ (6 * b = 3 * c)) :
  c = 135 := by
  sorry

end NUMINAMATH_CALUDE_dividend_problem_l332_33279


namespace NUMINAMATH_CALUDE_fraction_calculation_l332_33277

theorem fraction_calculation : (5 / 6 : ℚ) / (9 / 10) - 1 / 15 = 116 / 135 := by sorry

end NUMINAMATH_CALUDE_fraction_calculation_l332_33277


namespace NUMINAMATH_CALUDE_polynomial_factorization_l332_33229

theorem polynomial_factorization (a b : ℝ) :
  -3 * a^3 * b + 6 * a^2 * b^2 - 3 * a * b^3 = -3 * a * b * (a - b)^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l332_33229


namespace NUMINAMATH_CALUDE_isosceles_triangle_exists_l332_33205

/-- A regular polygon with (2n-1) sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin (2*n-1) → ℝ × ℝ

/-- A subset of n vertices from a (2n-1)-gon -/
def VertexSubset (n : ℕ) := Fin n → Fin (2*n-1)

/-- Predicate to check if three vertices form an isosceles triangle -/
def IsIsosceles (p : RegularPolygon n) (a b c : Fin (2*n-1)) : Prop :=
  let va := p.vertices a
  let vb := p.vertices b
  let vc := p.vertices c
  (va.1 - vc.1)^2 + (va.2 - vc.2)^2 = (vb.1 - vc.1)^2 + (vb.2 - vc.2)^2

/-- Main theorem: In any subset of n vertices of a (2n-1)-gon, there exists an isosceles triangle -/
theorem isosceles_triangle_exists (n : ℕ) (h : n ≥ 3) (p : RegularPolygon n) (s : VertexSubset n) :
  ∃ (a b c : Fin n), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ IsIsosceles p (s a) (s b) (s c) :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_exists_l332_33205


namespace NUMINAMATH_CALUDE_factor_of_polynomial_l332_33256

theorem factor_of_polynomial (x : ℝ) : 
  ∃ (q : ℝ → ℝ), (x^6 + 8 : ℝ) = (x^2 + 2) * q x := by
sorry

end NUMINAMATH_CALUDE_factor_of_polynomial_l332_33256


namespace NUMINAMATH_CALUDE_mango_profit_percentage_l332_33249

/-- Represents the rate at which mangoes are bought (number of mangoes per rupee) -/
def buy_rate : ℚ := 6

/-- Represents the rate at which mangoes are sold (number of mangoes per rupee) -/
def sell_rate : ℚ := 3

/-- Calculates the profit percentage given buy and sell rates -/
def profit_percentage (buy : ℚ) (sell : ℚ) : ℚ :=
  ((sell⁻¹ - buy⁻¹) / buy⁻¹) * 100

theorem mango_profit_percentage :
  profit_percentage buy_rate sell_rate = 100 := by
  sorry

end NUMINAMATH_CALUDE_mango_profit_percentage_l332_33249


namespace NUMINAMATH_CALUDE_regular_polygon_exterior_angle_18_deg_has_20_sides_l332_33252

/-- A regular polygon with exterior angles measuring 18 degrees has 20 sides. -/
theorem regular_polygon_exterior_angle_18_deg_has_20_sides :
  ∀ (n : ℕ), 
  n > 0 → 
  (360 : ℝ) / n = 18 → 
  n = 20 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_exterior_angle_18_deg_has_20_sides_l332_33252


namespace NUMINAMATH_CALUDE_first_term_of_sequence_l332_33222

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  -- The sum of the first 30 terms
  sum_30 : ℚ
  -- The sum of terms from 31st to 80th
  sum_31_to_80 : ℚ
  -- Property: The sum of the first 30 terms is 300
  sum_30_eq : sum_30 = 300
  -- Property: The sum of terms from 31st to 80th is 3750
  sum_31_to_80_eq : sum_31_to_80 = 3750

/-- The first term of the arithmetic sequence is -217/16 -/
theorem first_term_of_sequence (seq : ArithmeticSequence) : 
  ∃ (a d : ℚ), a = -217/16 ∧ 
  (∀ n : ℕ, n > 0 → n ≤ 30 → seq.sum_30 = (n/2) * (2*a + (n-1)*d)) ∧
  (seq.sum_31_to_80 = 25 * (2*a + 109*d)) :=
sorry

end NUMINAMATH_CALUDE_first_term_of_sequence_l332_33222


namespace NUMINAMATH_CALUDE_sum_of_w_and_y_is_three_l332_33232

theorem sum_of_w_and_y_is_three :
  ∀ (W X Y Z : ℕ),
    W ∈ ({1, 2, 3, 4} : Set ℕ) →
    X ∈ ({1, 2, 3, 4} : Set ℕ) →
    Y ∈ ({1, 2, 3, 4} : Set ℕ) →
    Z ∈ ({1, 2, 3, 4} : Set ℕ) →
    W ≠ X → W ≠ Y → W ≠ Z → X ≠ Y → X ≠ Z → Y ≠ Z →
    (W : ℚ) / X + (Y : ℚ) / Z = 1 →
    W + Y = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_w_and_y_is_three_l332_33232


namespace NUMINAMATH_CALUDE_unique_solution_for_n_l332_33268

theorem unique_solution_for_n : ∃! (n : ℕ+), ∃ (x : ℕ+),
  n = 2^(2*x.val-1) - 5*x.val - 3 ∧
  n = (2^(x.val-1) - 1) * (2^x.val + 1) ∧
  n = 2015 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_n_l332_33268


namespace NUMINAMATH_CALUDE_separation_of_homologous_chromosomes_unique_l332_33228

-- Define the cell division processes
inductive CellDivisionProcess
  | ChromosomeReplication
  | SeparationOfHomologousChromosomes
  | SeparationOfChromatids
  | Cytokinesis

-- Define the types of cell division
inductive CellDivision
  | Mitosis
  | Meiosis

-- Define a function that determines if a process occurs in a given cell division
def occursIn (process : CellDivisionProcess) (division : CellDivision) : Prop :=
  match division with
  | CellDivision.Mitosis =>
    process ≠ CellDivisionProcess.SeparationOfHomologousChromosomes
  | CellDivision.Meiosis => True

-- Theorem statement
theorem separation_of_homologous_chromosomes_unique :
  ∀ (process : CellDivisionProcess),
    (occursIn process CellDivision.Meiosis ∧ ¬occursIn process CellDivision.Mitosis) →
    process = CellDivisionProcess.SeparationOfHomologousChromosomes :=
by sorry

end NUMINAMATH_CALUDE_separation_of_homologous_chromosomes_unique_l332_33228


namespace NUMINAMATH_CALUDE_seat_difference_is_two_l332_33293

/-- Represents an airplane with first-class and coach class seats. -/
structure Airplane where
  total_seats : ℕ
  coach_seats : ℕ
  first_class_seats : ℕ
  h1 : total_seats = first_class_seats + coach_seats
  h2 : coach_seats > 4 * first_class_seats

/-- The difference between coach seats and 4 times first-class seats. -/
def seat_difference (a : Airplane) : ℕ :=
  a.coach_seats - 4 * a.first_class_seats

/-- Theorem stating the seat difference for a specific airplane configuration. -/
theorem seat_difference_is_two (a : Airplane)
  (h3 : a.total_seats = 387)
  (h4 : a.coach_seats = 310) :
  seat_difference a = 2 := by
  sorry


end NUMINAMATH_CALUDE_seat_difference_is_two_l332_33293


namespace NUMINAMATH_CALUDE_next_coincidence_exact_next_coincidence_l332_33223

def factory_whistle := 18
def train_bell := 24
def fire_alarm := 30

theorem next_coincidence (t : ℕ) : t > 0 ∧ t % factory_whistle = 0 ∧ t % train_bell = 0 ∧ t % fire_alarm = 0 → t ≥ 360 :=
sorry

theorem exact_next_coincidence : ∃ (t : ℕ), t = 360 ∧ t % factory_whistle = 0 ∧ t % train_bell = 0 ∧ t % fire_alarm = 0 :=
sorry

end NUMINAMATH_CALUDE_next_coincidence_exact_next_coincidence_l332_33223


namespace NUMINAMATH_CALUDE_smallest_sum_of_roots_l332_33210

theorem smallest_sum_of_roots (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h1 : ∃ x : ℝ, x^2 + 3*a*x + 3*b = 0)
  (h2 : ∃ x : ℝ, x^2 + 4*b*x + 2*a = 0) :
  a + b ≥ (10/9)^(1/3) := by
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_roots_l332_33210


namespace NUMINAMATH_CALUDE_least_n_for_phi_cube_l332_33297

def phi (n : ℕ) : ℕ := sorry

theorem least_n_for_phi_cube (n : ℕ) : 
  (∀ m < n, phi (phi (phi m)) * phi (phi m) * phi m ≠ 64000) ∧ 
  (phi (phi (phi n)) * phi (phi n) * phi n = 64000) → 
  n = 41 := by sorry

end NUMINAMATH_CALUDE_least_n_for_phi_cube_l332_33297


namespace NUMINAMATH_CALUDE_unique_function_satisfying_equation_l332_33204

theorem unique_function_satisfying_equation :
  ∃! f : ℝ → ℝ, ∀ x y : ℝ, f (x + f y) = x + y :=
by
  sorry

end NUMINAMATH_CALUDE_unique_function_satisfying_equation_l332_33204


namespace NUMINAMATH_CALUDE_greatest_common_multiple_9_15_under_120_l332_33294

theorem greatest_common_multiple_9_15_under_120 :
  ∃ n : ℕ, n = 90 ∧ 
  (∀ m : ℕ, m < 120 → m % 9 = 0 → m % 15 = 0 → m ≤ n) ∧
  90 % 9 = 0 ∧ 90 % 15 = 0 ∧ 90 < 120 :=
by sorry

end NUMINAMATH_CALUDE_greatest_common_multiple_9_15_under_120_l332_33294


namespace NUMINAMATH_CALUDE_profit_per_meter_l332_33217

/-- Profit per meter calculation -/
theorem profit_per_meter
  (length : ℕ)
  (selling_price : ℕ)
  (total_profit : ℕ)
  (h1 : length = 40)
  (h2 : selling_price = 8200)
  (h3 : total_profit = 1000) :
  total_profit / length = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_profit_per_meter_l332_33217


namespace NUMINAMATH_CALUDE_shorter_more_frequent_steps_slower_l332_33261

/-- Represents a tourist's walking characteristics -/
structure Tourist where
  step_length : ℝ
  step_count : ℕ

/-- Calculates the distance covered by a tourist -/
def distance_covered (t : Tourist) : ℝ := t.step_length * t.step_count

/-- Theorem stating that the tourist with shorter and more frequent steps is slower -/
theorem shorter_more_frequent_steps_slower (t1 t2 : Tourist) 
  (h1 : t1.step_length < t2.step_length) 
  (h2 : t1.step_count > t2.step_count) 
  (h3 : t1.step_length * t1.step_count < t2.step_length * t2.step_count) : 
  distance_covered t1 < distance_covered t2 := by
  sorry

#check shorter_more_frequent_steps_slower

end NUMINAMATH_CALUDE_shorter_more_frequent_steps_slower_l332_33261
