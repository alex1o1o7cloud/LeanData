import Mathlib

namespace NUMINAMATH_CALUDE_angle_between_lines_l3353_335392

def line1_direction : ℝ × ℝ := (2, 1)
def line2_direction : ℝ × ℝ := (4, 2)

theorem angle_between_lines (θ : ℝ) : 
  θ = Real.arccos (
    (line1_direction.1 * line2_direction.1 + line1_direction.2 * line2_direction.2) /
    (Real.sqrt (line1_direction.1^2 + line1_direction.2^2) * 
     Real.sqrt (line2_direction.1^2 + line2_direction.2^2))
  ) →
  Real.cos θ = 1 := by
sorry

end NUMINAMATH_CALUDE_angle_between_lines_l3353_335392


namespace NUMINAMATH_CALUDE_student_calculation_difference_l3353_335325

theorem student_calculation_difference : 
  let number : ℝ := 100.00000000000003
  let correct_answer := number * (4/5 : ℝ)
  let student_answer := number / (4/5 : ℝ)
  student_answer - correct_answer = 45.00000000000002 := by
sorry

end NUMINAMATH_CALUDE_student_calculation_difference_l3353_335325


namespace NUMINAMATH_CALUDE_fallen_cakes_ratio_l3353_335338

theorem fallen_cakes_ratio (total_cakes : ℕ) (destroyed_cakes : ℕ) : 
  total_cakes = 12 → 
  destroyed_cakes = 3 → 
  (2 * destroyed_cakes : ℚ) / total_cakes = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fallen_cakes_ratio_l3353_335338


namespace NUMINAMATH_CALUDE_dinner_cost_difference_l3353_335380

theorem dinner_cost_difference (initial_amount : ℝ) (first_course_cost : ℝ) (remaining_amount : ℝ) : 
  initial_amount = 60 →
  first_course_cost = 15 →
  remaining_amount = 20 →
  ∃ (second_course_cost : ℝ),
    initial_amount = first_course_cost + second_course_cost + (0.25 * second_course_cost) + remaining_amount ∧
    second_course_cost - first_course_cost = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_dinner_cost_difference_l3353_335380


namespace NUMINAMATH_CALUDE_parabola_shift_theorem_l3353_335315

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The original parabola y = 2x^2 - 1 -/
def original_parabola : Parabola :=
  { a := 2, b := 0, c := -1 }

/-- Shifts a parabola horizontally and vertically -/
def shift_parabola (p : Parabola) (h_shift : ℝ) (v_shift : ℝ) : Parabola :=
  { a := p.a
    b := -2 * p.a * h_shift + p.b
    c := p.a * h_shift^2 - p.b * h_shift + p.c + v_shift }

theorem parabola_shift_theorem :
  let shifted := shift_parabola original_parabola 1 (-2)
  shifted.a = 2 ∧ shifted.b = 4 ∧ shifted.c = -3 := by sorry

end NUMINAMATH_CALUDE_parabola_shift_theorem_l3353_335315


namespace NUMINAMATH_CALUDE_proportionality_analysis_l3353_335301

/-- Represents a relationship between x and y -/
inductive Relationship
  | DirectlyProportional
  | InverselyProportional
  | Neither

/-- Determines the relationship between x and y given an equation -/
def determineRelationship (equation : ℝ → ℝ → Prop) : Relationship :=
  sorry

/-- Equation A: x + y = 0 -/
def equationA (x y : ℝ) : Prop := x + y = 0

/-- Equation B: 3xy = 10 -/
def equationB (x y : ℝ) : Prop := 3 * x * y = 10

/-- Equation C: x = 5y -/
def equationC (x y : ℝ) : Prop := x = 5 * y

/-- Equation D: x^2 + 3x + y = 10 -/
def equationD (x y : ℝ) : Prop := x^2 + 3*x + y = 10

/-- Equation E: x/y = √3 -/
def equationE (x y : ℝ) : Prop := x / y = Real.sqrt 3

theorem proportionality_analysis :
  (determineRelationship equationA = Relationship.DirectlyProportional) ∧
  (determineRelationship equationB = Relationship.InverselyProportional) ∧
  (determineRelationship equationC = Relationship.DirectlyProportional) ∧
  (determineRelationship equationD = Relationship.Neither) ∧
  (determineRelationship equationE = Relationship.DirectlyProportional) :=
by
  sorry

end NUMINAMATH_CALUDE_proportionality_analysis_l3353_335301


namespace NUMINAMATH_CALUDE_cube_sum_from_conditions_l3353_335395

theorem cube_sum_from_conditions (x y : ℝ) 
  (sum_condition : x + y = 5)
  (sum_squares_condition : x^2 + y^2 = 20) :
  x^3 + y^3 = 87.5 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_from_conditions_l3353_335395


namespace NUMINAMATH_CALUDE_solve_equation_l3353_335331

theorem solve_equation : ∃ x : ℝ, 3 * x + 15 = (1/3) * (6 * x + 45) ∧ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3353_335331


namespace NUMINAMATH_CALUDE_equation_solutions_l3353_335364

def equation (x : ℝ) : Prop :=
  x ≠ 2 ∧ x ≠ 1 ∧ x ≠ -6 ∧ (3*x + 6) / ((x - 1) * (x + 6)) = (3 - x) / (x - 2)

theorem equation_solutions :
  ∀ x : ℝ, equation x ↔ (x = 3 ∨ x = -4) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3353_335364


namespace NUMINAMATH_CALUDE_no_function_satisfies_condition_l3353_335397

theorem no_function_satisfies_condition :
  ¬ ∃ f : ℝ → ℝ, ∀ x y z : ℝ, f (x * y) + f (x * z) - f x * f (y * z) > 1 := by
  sorry

end NUMINAMATH_CALUDE_no_function_satisfies_condition_l3353_335397


namespace NUMINAMATH_CALUDE_division_of_fractions_l3353_335355

theorem division_of_fractions : (7 : ℚ) / (8 / 13) = 91 / 8 := by
  sorry

end NUMINAMATH_CALUDE_division_of_fractions_l3353_335355


namespace NUMINAMATH_CALUDE_inequality_and_range_l3353_335399

-- Define the function representing the left side of the inequality
def f (x : ℝ) : ℝ := |x - 3| + |x + 1|

-- State the theorem
theorem inequality_and_range : 
  (∀ x : ℝ, f x ≥ 4) ∧ 
  (∀ x : ℝ, f x = 4 ↔ -1 ≤ x ∧ x ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_range_l3353_335399


namespace NUMINAMATH_CALUDE_symmetry_and_evenness_l3353_335337

def symmetric_wrt_y_axis (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (|x|) = f (-|x|)

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

theorem symmetry_and_evenness (f : ℝ → ℝ) :
  (even_function f → symmetric_wrt_y_axis f) ∧
  ∃ g : ℝ → ℝ, symmetric_wrt_y_axis g ∧ ¬even_function g :=
sorry

end NUMINAMATH_CALUDE_symmetry_and_evenness_l3353_335337


namespace NUMINAMATH_CALUDE_binomial_18_6_l3353_335390

theorem binomial_18_6 : (Nat.choose 18 6) = 18564 := by
  sorry

end NUMINAMATH_CALUDE_binomial_18_6_l3353_335390


namespace NUMINAMATH_CALUDE_bottle_capacity_proof_l3353_335339

theorem bottle_capacity_proof (total_milk : ℝ) (bottle1_capacity : ℝ) (bottle1_milk : ℝ) (bottle2_capacity : ℝ) :
  total_milk = 8 →
  bottle1_capacity = 8 →
  bottle1_milk = 5.333333333333333 →
  (bottle1_milk / bottle1_capacity) = ((total_milk - bottle1_milk) / bottle2_capacity) →
  bottle2_capacity = 4 := by
sorry

end NUMINAMATH_CALUDE_bottle_capacity_proof_l3353_335339


namespace NUMINAMATH_CALUDE_triangle_not_right_angle_l3353_335386

theorem triangle_not_right_angle (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_ratio : b = (4/3) * a ∧ c = (5/3) * a) (h_sum : a + b + c = 180) :
  ¬ (a = 90 ∨ b = 90 ∨ c = 90) :=
sorry

end NUMINAMATH_CALUDE_triangle_not_right_angle_l3353_335386


namespace NUMINAMATH_CALUDE_trivia_contest_probability_l3353_335308

/-- The number of questions in the trivia contest -/
def num_questions : ℕ := 4

/-- The number of choices for each question -/
def num_choices : ℕ := 4

/-- The minimum number of correct answers needed to win -/
def min_correct : ℕ := 3

/-- The probability of guessing one question correctly -/
def prob_correct : ℚ := 1 / num_choices

/-- The probability of guessing one question incorrectly -/
def prob_incorrect : ℚ := 1 - prob_correct

/-- The probability of winning the trivia contest -/
def prob_winning : ℚ := 13 / 256

theorem trivia_contest_probability :
  (prob_correct ^ num_questions) +
  (num_questions.choose min_correct) * (prob_correct ^ min_correct) * (prob_incorrect ^ (num_questions - min_correct)) =
  prob_winning := by sorry

end NUMINAMATH_CALUDE_trivia_contest_probability_l3353_335308


namespace NUMINAMATH_CALUDE_quadratic_form_minimum_l3353_335347

theorem quadratic_form_minimum : 
  ∀ x y : ℝ, 3 * x^2 + 4 * x * y + 5 * y^2 - 8 * x - 10 * y ≥ -3 ∧
  ∃ x₀ y₀ : ℝ, 3 * x₀^2 + 4 * x₀ * y₀ + 5 * y₀^2 - 8 * x₀ - 10 * y₀ = -3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_form_minimum_l3353_335347


namespace NUMINAMATH_CALUDE_last_digit_of_large_exponentiation_l3353_335329

/-- The last digit of a number -/
def lastDigit (n : ℕ) : ℕ := n % 10

/-- Exponentiation modulo 10 -/
def powMod10 (base exponent : ℕ) : ℕ :=
  (base ^ (exponent % 4)) % 10

theorem last_digit_of_large_exponentiation :
  lastDigit (powMod10 954950230952380948328708 470128749397540235934750230) = 4 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_large_exponentiation_l3353_335329


namespace NUMINAMATH_CALUDE_smallest_group_size_sixty_five_divisible_smallest_group_is_65_l3353_335349

theorem smallest_group_size (n : ℕ) : n > 0 ∧ n % 5 = 0 ∧ n % 13 = 0 → n ≥ 65 := by
  sorry

theorem sixty_five_divisible : 65 % 5 = 0 ∧ 65 % 13 = 0 := by
  sorry

theorem smallest_group_is_65 : ∃ (n : ℕ), n > 0 ∧ n % 5 = 0 ∧ n % 13 = 0 ∧ ∀ (m : ℕ), (m > 0 ∧ m % 5 = 0 ∧ m % 13 = 0) → m ≥ n := by
  sorry

end NUMINAMATH_CALUDE_smallest_group_size_sixty_five_divisible_smallest_group_is_65_l3353_335349


namespace NUMINAMATH_CALUDE_jenny_bus_time_l3353_335376

/-- Represents the schedule of Jenny's day --/
structure Schedule where
  wakeUpTime : Nat  -- in minutes after midnight
  busToSchoolTime : Nat  -- in minutes after midnight
  numClasses : Nat
  classDuration : Nat  -- in minutes
  lunchDuration : Nat  -- in minutes
  extracurricularDuration : Nat  -- in minutes
  busHomeTime : Nat  -- in minutes after midnight

/-- Calculates the time Jenny spent on the bus given her schedule --/
def timeBusSpent (s : Schedule) : Nat :=
  (s.busHomeTime - s.busToSchoolTime) - 
  (s.numClasses * s.classDuration + s.lunchDuration + s.extracurricularDuration)

/-- Jenny's actual schedule --/
def jennySchedule : Schedule :=
  { wakeUpTime := 7 * 60
    busToSchoolTime := 8 * 60
    numClasses := 5
    classDuration := 45
    lunchDuration := 45
    extracurricularDuration := 90
    busHomeTime := 17 * 60 }

theorem jenny_bus_time : timeBusSpent jennySchedule = 180 := by
  sorry

end NUMINAMATH_CALUDE_jenny_bus_time_l3353_335376


namespace NUMINAMATH_CALUDE_sum_cannot_have_all_odd_digits_l3353_335368

/-- A digit is a natural number between 0 and 9. -/
def Digit : Type := {n : ℕ // n ≤ 9}

/-- A sequence of 1001 digits. -/
def DigitSequence : Type := Fin 1001 → Digit

/-- The first number formed by the digit sequence. -/
def firstNumber (a : DigitSequence) : ℕ := sorry

/-- The second number formed by the reversed digit sequence. -/
def secondNumber (a : DigitSequence) : ℕ := sorry

/-- A number has all odd digits if each of its digits is odd. -/
def hasAllOddDigits (n : ℕ) : Prop := sorry

theorem sum_cannot_have_all_odd_digits (a : DigitSequence) :
  ¬(hasAllOddDigits (firstNumber a + secondNumber a)) :=
sorry

end NUMINAMATH_CALUDE_sum_cannot_have_all_odd_digits_l3353_335368


namespace NUMINAMATH_CALUDE_false_proposition_implies_plane_plane_line_l3353_335362

-- Define geometric figures
inductive GeometricFigure
  | Line
  | Plane

-- Define perpendicular and parallel relations
def perpendicular (a b : GeometricFigure) : Prop := sorry
def parallel (a b : GeometricFigure) : Prop := sorry

-- Define the proposition
def proposition (x y z : GeometricFigure) : Prop :=
  perpendicular x y → parallel y z → perpendicular x z

-- Theorem statement
theorem false_proposition_implies_plane_plane_line :
  ∀ x y z : GeometricFigure,
  ¬(proposition x y z) →
  (x = GeometricFigure.Plane ∧ y = GeometricFigure.Plane ∧ z = GeometricFigure.Line) :=
sorry

end NUMINAMATH_CALUDE_false_proposition_implies_plane_plane_line_l3353_335362


namespace NUMINAMATH_CALUDE_linear_function_condition_l3353_335318

/-- A linear function with respect to x of the form y = (m-2)x + 2 -/
def linearFunction (m : ℝ) (x : ℝ) : ℝ := (m - 2) * x + 2

/-- The condition for the function to be linear with respect to x -/
def isLinear (m : ℝ) : Prop := m ≠ 2

theorem linear_function_condition (m : ℝ) :
  (∀ x, ∃ y, y = linearFunction m x) ↔ isLinear m :=
sorry

end NUMINAMATH_CALUDE_linear_function_condition_l3353_335318


namespace NUMINAMATH_CALUDE_maria_alice_ages_sum_l3353_335346

/-- Maria and Alice's ages problem -/
theorem maria_alice_ages_sum : 
  ∀ (maria alice : ℕ), 
    maria = alice + 8 →  -- Maria is eight years older than Alice
    maria + 10 = 3 * (alice - 6) →  -- Ten years from now, Maria will be three times as old as Alice was six years ago
    maria + alice = 44  -- The sum of their current ages is 44
    := by sorry

end NUMINAMATH_CALUDE_maria_alice_ages_sum_l3353_335346


namespace NUMINAMATH_CALUDE_satisfying_function_is_identity_l3353_335367

/-- A function satisfying the given conditions -/
def SatisfyingFunction (f : ℕ → ℕ) : Prop :=
  (∀ k : ℕ, k > 0 → ∃ S : Set ℕ, S.Infinite ∧ ∀ p ∈ S, Prime p ∧ ∃ c : ℕ, c > 0 ∧ f c = p^k) ∧
  (∀ m n : ℕ, m > 0 ∧ n > 0 → (f m + f n) ∣ f (m + n))

/-- The main theorem stating that any function satisfying the conditions is the identity function -/
theorem satisfying_function_is_identity (f : ℕ → ℕ) (h : SatisfyingFunction f) : 
  ∀ n : ℕ, f n = n := by
  sorry

end NUMINAMATH_CALUDE_satisfying_function_is_identity_l3353_335367


namespace NUMINAMATH_CALUDE_cube_of_negative_l3353_335354

theorem cube_of_negative (a : ℝ) : (-a)^3 = -a^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_negative_l3353_335354


namespace NUMINAMATH_CALUDE_power_product_equality_l3353_335312

theorem power_product_equality : 0.25^2015 * 4^2016 = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equality_l3353_335312


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l3353_335394

/-- Two vectors are parallel if and only if one is a scalar multiple of the other -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

/-- The problem statement -/
theorem parallel_vectors_x_value :
  ∀ x : ℝ, are_parallel (1, x) (-2, 1) → x = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l3353_335394


namespace NUMINAMATH_CALUDE_age_squares_sum_l3353_335361

theorem age_squares_sum (d t h : ℕ) : 
  t = 2 * d ∧ 
  h^2 + 4 * d = 5 * t ∧ 
  3 * h^2 = 7 * d^2 + 2 * t^2 →
  d^2 + h^2 + t^2 = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_age_squares_sum_l3353_335361


namespace NUMINAMATH_CALUDE_parabola_y_intercepts_l3353_335348

theorem parabola_y_intercepts (y : ℝ) : ¬ ∃ y, 3 * y^2 - 4 * y + 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_parabola_y_intercepts_l3353_335348


namespace NUMINAMATH_CALUDE_num_dogs_correct_l3353_335341

/-- The number of dogs Ella owns -/
def num_dogs : ℕ := 2

/-- The amount of food each dog eats per day (in scoops) -/
def food_per_dog : ℚ := 1/8

/-- The total amount of food eaten by all dogs per day (in scoops) -/
def total_food : ℚ := 1/4

/-- Theorem stating that the number of dogs is correct given the food consumption -/
theorem num_dogs_correct : (num_dogs : ℚ) * food_per_dog = total_food := by sorry

end NUMINAMATH_CALUDE_num_dogs_correct_l3353_335341


namespace NUMINAMATH_CALUDE_article_price_before_discount_l3353_335375

/-- Given an article with a price that, after a 24% decrease, is 760 rupees,
    prove that its original price was 1000 rupees. -/
theorem article_price_before_discount (price_after_discount : ℝ) 
  (h1 : price_after_discount = 760) 
  (h2 : price_after_discount = 0.76 * price_after_discount / 0.76) : 
  price_after_discount / 0.76 = 1000 := by
  sorry

#check article_price_before_discount

end NUMINAMATH_CALUDE_article_price_before_discount_l3353_335375


namespace NUMINAMATH_CALUDE_amare_dresses_l3353_335324

/-- The number of dresses Amare needs to make -/
def number_of_dresses : ℕ := 4

/-- The amount of fabric required for one dress in yards -/
def fabric_per_dress : ℚ := 5.5

/-- The amount of fabric Amare has in feet -/
def fabric_amare_has : ℕ := 7

/-- The amount of fabric Amare still needs in feet -/
def fabric_amare_needs : ℕ := 59

/-- The number of feet in a yard -/
def feet_per_yard : ℕ := 3

theorem amare_dresses :
  number_of_dresses = 
    (((fabric_amare_has + fabric_amare_needs : ℚ) / feet_per_yard) / fabric_per_dress).floor :=
by sorry

end NUMINAMATH_CALUDE_amare_dresses_l3353_335324


namespace NUMINAMATH_CALUDE_min_sum_with_prime_hcfs_l3353_335319

/-- Given three positive integers with pairwise HCFs being distinct primes, 
    their sum is at least 31 -/
theorem min_sum_with_prime_hcfs (Q R S : ℕ+) 
  (hQR : ∃ (p : ℕ), Nat.Prime p ∧ Nat.gcd Q.val R.val = p)
  (hQS : ∃ (q : ℕ), Nat.Prime q ∧ Nat.gcd Q.val S.val = q)
  (hRS : ∃ (r : ℕ), Nat.Prime r ∧ Nat.gcd R.val S.val = r)
  (h_distinct : ∀ (p q r : ℕ), 
    Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ 
    Nat.gcd Q.val R.val = p ∧ Nat.gcd Q.val S.val = q ∧ Nat.gcd R.val S.val = r →
    p ≠ q ∧ q ≠ r ∧ p ≠ r) :
  Q.val + R.val + S.val ≥ 31 := by
  sorry

#check min_sum_with_prime_hcfs

end NUMINAMATH_CALUDE_min_sum_with_prime_hcfs_l3353_335319


namespace NUMINAMATH_CALUDE_circle_equation_tangent_line_l3353_335383

/-- The equation of a circle with center (3, -1) that is tangent to the line 3x + 4y = 0 is (x-3)² + (y+1)² = 1 -/
theorem circle_equation_tangent_line (x y : ℝ) : 
  let center : ℝ × ℝ := (3, -1)
  let line (x y : ℝ) := 3 * x + 4 * y = 0
  let circle_eq (x y : ℝ) := (x - center.1)^2 + (y - center.2)^2 = 1
  let is_tangent (circle : (ℝ → ℝ → Prop) ) (line : ℝ → ℝ → Prop) := 
    ∃ (x y : ℝ), circle x y ∧ line x y ∧ 
    ∀ (x' y' : ℝ), line x' y' → (x' = x ∧ y' = y) ∨ ¬(circle x' y')
  is_tangent circle_eq line → circle_eq x y := by
sorry


end NUMINAMATH_CALUDE_circle_equation_tangent_line_l3353_335383


namespace NUMINAMATH_CALUDE_problem_solution_l3353_335304

-- Define the set B
def B : Set ℝ := {m | ∀ x ∈ Set.Icc (-1) 2, x^2 - 2*x - m ≤ 0}

-- Define the set A
def A (a : ℝ) : Set ℝ := {x | (x - 2*a) * (x - (a + 1)) ≤ 0}

-- State the theorem
theorem problem_solution :
  (B = Set.Ici 3) ∧
  (∀ a : ℝ, (∀ x : ℝ, x ∈ B → x ∈ A a) ∧ (∃ x : ℝ, x ∈ A a ∧ x ∉ B) → a ≥ 2) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l3353_335304


namespace NUMINAMATH_CALUDE_same_color_prob_is_eleven_thirty_sixths_l3353_335379

/-- A die with 12 sides and specific color distribution -/
structure TwelveSidedDie :=
  (red : ℕ)
  (blue : ℕ)
  (green : ℕ)
  (golden : ℕ)
  (total_sides : red + blue + green + golden = 12)

/-- The probability of two dice showing the same color -/
def same_color_probability (d : TwelveSidedDie) : ℚ :=
  (d.red^2 + d.blue^2 + d.green^2 + d.golden^2) / 144

/-- Theorem stating the probability of two 12-sided dice showing the same color -/
theorem same_color_prob_is_eleven_thirty_sixths :
  ∀ d : TwelveSidedDie,
  d.red = 3 → d.blue = 5 → d.green = 3 → d.golden = 1 →
  same_color_probability d = 11 / 36 :=
sorry

end NUMINAMATH_CALUDE_same_color_prob_is_eleven_thirty_sixths_l3353_335379


namespace NUMINAMATH_CALUDE_length_of_AE_l3353_335356

-- Define the square and points
def Square (A B C D : ℝ × ℝ) : Prop :=
  A = (0, 0) ∧ B = (4, 0) ∧ C = (4, 4) ∧ D = (0, 4)

def PointOnSide (E : ℝ × ℝ) : Prop :=
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ 4 ∧ E = (x, 0)

def ReflectionOverDiagonal (E F : ℝ × ℝ) : Prop :=
  F.1 + F.2 = 4 ∧ F.1 = 4 - E.1

def DistanceCondition (D E F : ℝ × ℝ) : Prop :=
  (F.1 - D.1)^2 + (F.2 - D.2)^2 = 4 * (E.1 - D.1)^2

-- Main theorem
theorem length_of_AE (A B C D E F : ℝ × ℝ) :
  Square A B C D →
  PointOnSide E →
  ReflectionOverDiagonal E F →
  DistanceCondition D E F →
  E.1 = 8/3 :=
sorry

end NUMINAMATH_CALUDE_length_of_AE_l3353_335356


namespace NUMINAMATH_CALUDE_max_boxes_per_delivery_l3353_335391

/-- Represents the maximum capacity of each truck in pounds -/
def truckCapacity : ℕ := 2000

/-- Represents the weight of a light box in pounds -/
def lightBoxWeight : ℕ := 10

/-- Represents the weight of a heavy box in pounds -/
def heavyBoxWeight : ℕ := 40

/-- Represents the number of trucks available for each delivery -/
def numberOfTrucks : ℕ := 3

/-- Theorem stating the maximum number of boxes that can be shipped in each delivery -/
theorem max_boxes_per_delivery :
  ∃ (n : ℕ), n = numberOfTrucks * truckCapacity / (lightBoxWeight + heavyBoxWeight) * 2 ∧ n = 240 := by
  sorry

end NUMINAMATH_CALUDE_max_boxes_per_delivery_l3353_335391


namespace NUMINAMATH_CALUDE_digital_sum_property_l3353_335385

/-- Digital sum of a natural number -/
def digitalSum (n : ℕ) : ℕ := sorry

/-- Proposition: M satisfies S(Mk) = S(M) for all 1 ≤ k ≤ M iff M = 10^l - 1 for some l -/
theorem digital_sum_property (M : ℕ) :
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ M → digitalSum (M * k) = digitalSum M) ↔
  ∃ l : ℕ, l > 0 ∧ M = 10^l - 1 :=
sorry

end NUMINAMATH_CALUDE_digital_sum_property_l3353_335385


namespace NUMINAMATH_CALUDE_trevors_age_a_decade_ago_l3353_335393

theorem trevors_age_a_decade_ago (trevors_brother_age : ℕ) 
  (h1 : trevors_brother_age = 32) 
  (h2 : trevors_brother_age - 20 = 2 * (trevors_brother_age - 30)) : 
  trevors_brother_age - 30 = 16 := by
  sorry

end NUMINAMATH_CALUDE_trevors_age_a_decade_ago_l3353_335393


namespace NUMINAMATH_CALUDE_min_value_of_arithmetic_sequence_l3353_335381

/-- An arithmetic sequence with positive terms -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d ∧ a n > 0

theorem min_value_of_arithmetic_sequence (a : ℕ → ℝ) 
  (h_arith : ArithmeticSequence a) 
  (h_eq : 2 * a 4 + a 3 - 2 * a 2 - a 1 = 8) :
  ∃ m : ℝ, m = 12 * Real.sqrt 3 ∧ ∀ q : ℝ, 2 * a 5 + a 4 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_arithmetic_sequence_l3353_335381


namespace NUMINAMATH_CALUDE_contractor_problem_l3353_335332

-- Define the parameters
def total_days : ℕ := 50
def initial_workers : ℕ := 70
def days_passed : ℕ := 25
def work_completed : ℚ := 40 / 100

-- Define the function to calculate additional workers needed
def additional_workers_needed (total_days : ℕ) (initial_workers : ℕ) (days_passed : ℕ) (work_completed : ℚ) : ℕ :=
  -- The actual calculation will be implemented in the proof
  sorry

-- Theorem statement
theorem contractor_problem :
  additional_workers_needed total_days initial_workers days_passed work_completed = 35 :=
by
  sorry

end NUMINAMATH_CALUDE_contractor_problem_l3353_335332


namespace NUMINAMATH_CALUDE_function_property_l3353_335320

def is_periodic (f : ℕ → ℕ) (period : ℕ) : Prop :=
  ∀ n, f (n + period) = f n

theorem function_property (f : ℕ → ℕ) 
  (h1 : ∀ n, f n ≠ 1)
  (h2 : ∀ n, f (n + 1) + f (n + 3) = f (n + 5) * f (n + 7) - 1375) :
  (is_periodic f 4) ∧ 
  (∀ n k, (f (n + 4 * k + 1) - 1) * (f (n + 4 * k + 3) - 1) = 1376) :=
sorry

end NUMINAMATH_CALUDE_function_property_l3353_335320


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l3353_335327

theorem quadratic_roots_range (m l : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ m * x^2 - 2*x + l = 0 ∧ m * y^2 - 2*y + l = 0) → 
  (0 < m ∧ m < 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l3353_335327


namespace NUMINAMATH_CALUDE_first_caterer_cheaper_at_17_l3353_335359

/-- The least number of people for which the first caterer is cheaper -/
def least_people_first_caterer_cheaper : ℕ := 17

/-- Cost function for the first caterer -/
def cost_first_caterer (people : ℕ) : ℚ := 200 + 18 * people

/-- Cost function for the second caterer -/
def cost_second_caterer (people : ℕ) : ℚ := 250 + 15 * people

/-- Theorem stating that 17 is the least number of people for which the first caterer is cheaper -/
theorem first_caterer_cheaper_at_17 :
  (∀ n : ℕ, n < least_people_first_caterer_cheaper →
    cost_first_caterer n ≥ cost_second_caterer n) ∧
  cost_first_caterer least_people_first_caterer_cheaper < cost_second_caterer least_people_first_caterer_cheaper :=
by sorry

end NUMINAMATH_CALUDE_first_caterer_cheaper_at_17_l3353_335359


namespace NUMINAMATH_CALUDE_shaded_area_is_six_l3353_335352

/-- Represents a quadrilateral divided into four smaller quadrilaterals -/
structure DividedQuadrilateral where
  /-- Areas of the four smaller quadrilaterals -/
  area1 : ℝ
  area2 : ℝ
  area3 : ℝ
  area4 : ℝ
  /-- The sum of areas of two opposite quadrilaterals is 28 -/
  sum_opposite : area1 + area3 = 28
  /-- One of the quadrilaterals has an area of 8 -/
  known_area : area2 = 8

/-- The theorem to be proved -/
theorem shaded_area_is_six (q : DividedQuadrilateral) : q.area4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_is_six_l3353_335352


namespace NUMINAMATH_CALUDE_rare_integer_existence_and_uniqueness_l3353_335398

/-- A function f: ℤ → ℤ satisfying the given functional equation -/
def FunctionalEquation (f : ℤ → ℤ) : Prop :=
  ∀ x y : ℤ, f (f (x + y) + y) = f (f x + y)

/-- An integer v is f-rare if the set {x ∈ ℤ : f(x) = v} is finite and nonempty -/
def IsRare (f : ℤ → ℤ) (v : ℤ) : Prop :=
  let X_v := {x : ℤ | f x = v}
  Set.Finite X_v ∧ Set.Nonempty X_v

theorem rare_integer_existence_and_uniqueness :
  (∃ f : ℤ → ℤ, FunctionalEquation f ∧ ∃ v : ℤ, IsRare f v) ∧
  (∀ f : ℤ → ℤ, FunctionalEquation f → ∀ v w : ℤ, IsRare f v → IsRare f w → v = w) :=
by sorry

end NUMINAMATH_CALUDE_rare_integer_existence_and_uniqueness_l3353_335398


namespace NUMINAMATH_CALUDE_two_star_three_equals_one_l3353_335382

-- Define the ã — operation
def star_op (a b : ℤ) : ℤ := 2 * a - 3 * b + a * b

-- State the theorem
theorem two_star_three_equals_one :
  star_op 2 3 = 1 :=
by sorry

end NUMINAMATH_CALUDE_two_star_three_equals_one_l3353_335382


namespace NUMINAMATH_CALUDE_inequality_implication_l3353_335302

theorem inequality_implication (a b : ℝ) : 
  a^2 - b^2 + 2*a - 4*b - 3 ≠ 0 → a - b ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l3353_335302


namespace NUMINAMATH_CALUDE_solution_set_for_a_5_range_of_a_l3353_335303

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + a| + |x - 2|

-- Part I
theorem solution_set_for_a_5 :
  {x : ℝ | f 5 x > 9} = {x : ℝ | x < -6 ∨ x > 3} := by sorry

-- Part II
def A (a : ℝ) : Set ℝ := {x : ℝ | f a x ≤ |x - 4|}
def B : Set ℝ := {x : ℝ | |2*x - 1| ≤ 3}

theorem range_of_a :
  ∀ a : ℝ, (A a ∪ B = A a) → a ∈ Set.Icc (-1) 0 := by sorry

end NUMINAMATH_CALUDE_solution_set_for_a_5_range_of_a_l3353_335303


namespace NUMINAMATH_CALUDE_point_outside_circle_l3353_335387

/-- A circle with a given radius -/
structure Circle where
  radius : ℝ

/-- A point with a given distance from the center of a circle -/
structure Point where
  distanceFromCenter : ℝ

/-- Determines if a point is outside a circle -/
def isOutside (c : Circle) (p : Point) : Prop :=
  p.distanceFromCenter > c.radius

/-- Theorem: If the radius of a circle is 3 and the distance from a point to the center is 4,
    then the point is outside the circle -/
theorem point_outside_circle (c : Circle) (p : Point)
    (h1 : c.radius = 3)
    (h2 : p.distanceFromCenter = 4) :
    isOutside c p := by
  sorry

end NUMINAMATH_CALUDE_point_outside_circle_l3353_335387


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3353_335316

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x : ℝ, x^2 + a*x + b < 0 ↔ 1 < x ∧ x < 2) → b = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3353_335316


namespace NUMINAMATH_CALUDE_inequality_solution_l3353_335366

theorem inequality_solution (x : ℝ) : 
  (∃ a : ℝ, a ∈ Set.Icc (-1) 2 ∧ (2 - a) * x^3 + (1 - 2*a) * x^2 - 6*x + 5 + 4*a - a^2 < 0) ↔ 
  (x < -2 ∨ (0 < x ∧ x < 1) ∨ 1 < x) := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_l3353_335366


namespace NUMINAMATH_CALUDE_solution_set_correct_l3353_335353

/-- The system of equations --/
def system (x y z : ℝ) : Prop :=
  6 * (x^2 * y^2 + y^2 * z^2 + z^2 * x^2) - 49 * x * y * z = 0 ∧
  6 * y * (x^2 - z^2) + 5 * x * z = 0 ∧
  2 * z * (x^2 - y^2) - 9 * x * y = 0

/-- The solution set --/
def solution_set : Set (ℝ × ℝ × ℝ) :=
  {(0, 0, 0), (2, 1, 3), (2, -1, -3), (-2, 1, -3), (-2, -1, 3)}

/-- Theorem stating that the solution set is correct --/
theorem solution_set_correct :
  ∀ x y z, (x, y, z) ∈ solution_set ↔ system x y z :=
by sorry

end NUMINAMATH_CALUDE_solution_set_correct_l3353_335353


namespace NUMINAMATH_CALUDE_infinitely_many_solutions_l3353_335371

theorem infinitely_many_solutions :
  ∃ f : ℕ → ℕ × ℕ, ∀ n : ℕ,
    let (a, b) := f n
    (a > 0 ∧ b > 0) ∧ a^2 - b^2 = a * b - 1 :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_solutions_l3353_335371


namespace NUMINAMATH_CALUDE_modem_download_time_l3353_335330

theorem modem_download_time (time_a : ℝ) (speed_ratio : ℝ) (time_b : ℝ) : 
  time_a = 25.5 →
  speed_ratio = 0.17 →
  time_b = time_a / speed_ratio →
  time_b = 150 := by
sorry

end NUMINAMATH_CALUDE_modem_download_time_l3353_335330


namespace NUMINAMATH_CALUDE_extreme_points_imply_a_l3353_335370

/-- Given a function f(x) = a * ln(x) + b * x^2 + x, if x = 1 and x = 2 are extreme points,
    then a = -2/3 --/
theorem extreme_points_imply_a (a b : ℝ) :
  let f : ℝ → ℝ := λ x => a * Real.log x + b * x^2 + x
  (∃ (c : ℝ), c ≠ 1 ∧ c ≠ 2 ∧ 
    (deriv f 1 = 0 ∧ deriv f 2 = 0) ∧
    (∀ x ∈ Set.Ioo 1 2, deriv f x ≠ 0)) →
  a = -2/3 := by
sorry

end NUMINAMATH_CALUDE_extreme_points_imply_a_l3353_335370


namespace NUMINAMATH_CALUDE_students_neither_sport_l3353_335307

theorem students_neither_sport (total : ℕ) (football : ℕ) (cricket : ℕ) (both : ℕ)
  (h_total : total = 460)
  (h_football : football = 325)
  (h_cricket : cricket = 175)
  (h_both : both = 90) :
  total - (football + cricket - both) = 50 := by
  sorry

end NUMINAMATH_CALUDE_students_neither_sport_l3353_335307


namespace NUMINAMATH_CALUDE_exponential_equation_solution_l3353_335323

theorem exponential_equation_solution :
  ∃ x : ℝ, (4 : ℝ)^x * (4 : ℝ)^x * (4 : ℝ)^x = (16 : ℝ)^5 ∧ x = 10/3 := by
  sorry

end NUMINAMATH_CALUDE_exponential_equation_solution_l3353_335323


namespace NUMINAMATH_CALUDE_fifteen_initial_points_theorem_l3353_335333

/-- The number of points after n iterations of the marking process -/
def points_after_iteration (initial_points : ℕ) (n : ℕ) : ℕ :=
  match n with
  | 0 => initial_points
  | k + 1 => 2 * (points_after_iteration initial_points k) - 1

/-- The theorem stating that 15 initial points result in 225 points after 4 iterations -/
theorem fifteen_initial_points_theorem :
  ∃ (initial_points : ℕ), 
    initial_points > 0 ∧ 
    points_after_iteration initial_points 4 = 225 ∧ 
    initial_points = 15 := by
  sorry


end NUMINAMATH_CALUDE_fifteen_initial_points_theorem_l3353_335333


namespace NUMINAMATH_CALUDE_angle_ABD_measure_l3353_335311

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : Point)

-- Define the angles in the quadrilateral
def angle_ABC (q : Quadrilateral) : ℝ := 120
def angle_DAB (q : Quadrilateral) : ℝ := 30
def angle_ADB (q : Quadrilateral) : ℝ := 28

-- Define the theorem
theorem angle_ABD_measure (q : Quadrilateral) :
  angle_ABC q = 120 ∧ angle_DAB q = 30 ∧ angle_ADB q = 28 →
  ∃ (angle_ABD : ℝ), angle_ABD = 122 :=
sorry

end NUMINAMATH_CALUDE_angle_ABD_measure_l3353_335311


namespace NUMINAMATH_CALUDE_max_distance_sparkling_points_l3353_335350

theorem max_distance_sparkling_points :
  ∀ (a₁ b₁ a₂ b₂ : ℝ),
    a₁^2 + b₁^2 = 1 →
    a₂^2 + b₂^2 = 1 →
    ∀ (d : ℝ),
      d = Real.sqrt ((a₂ - a₁)^2 + (b₂ - b₁)^2) →
      d ≤ 2 ∧ ∃ (a₁' b₁' a₂' b₂' : ℝ),
        a₁'^2 + b₁'^2 = 1 ∧
        a₂'^2 + b₂'^2 = 1 ∧
        Real.sqrt ((a₂' - a₁')^2 + (b₂' - b₁')^2) = 2 :=
by sorry

end NUMINAMATH_CALUDE_max_distance_sparkling_points_l3353_335350


namespace NUMINAMATH_CALUDE_system_solution_l3353_335378

theorem system_solution : 
  ∃! (x y : ℝ), (3 * x + y = 2 ∧ 2 * x - 3 * y = 27) ∧ x = 3 ∧ y = -7 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3353_335378


namespace NUMINAMATH_CALUDE_range_of_a_l3353_335334

theorem range_of_a (a : ℝ) 
  (h1 : ∀ x > 0, x + 4 / x ≥ a) 
  (h2 : ∃ x : ℝ, x^2 + 2*x + a = 0) : 
  a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3353_335334


namespace NUMINAMATH_CALUDE_kates_remaining_money_is_7_80_l3353_335340

/-- Calculates the amount of money Kate has left after her savings and expenses --/
def kates_remaining_money (march_savings april_savings may_savings june_savings : ℚ)
  (keyboard_cost mouse_cost headset_cost video_game_cost : ℚ)
  (book_cost : ℚ)
  (euro_to_dollar pound_to_dollar : ℚ) : ℚ :=
  let total_savings := march_savings + april_savings + may_savings + june_savings + 2 * april_savings
  let euro_expenses := (keyboard_cost + mouse_cost + headset_cost + video_game_cost) * euro_to_dollar
  let pound_expenses := book_cost * pound_to_dollar
  total_savings - euro_expenses - pound_expenses

/-- Theorem stating that Kate has $7.80 left after her savings and expenses --/
theorem kates_remaining_money_is_7_80 :
  kates_remaining_money 27 13 28 35 42 4 16 25 12 1.2 1.4 = 7.8 := by
  sorry

end NUMINAMATH_CALUDE_kates_remaining_money_is_7_80_l3353_335340


namespace NUMINAMATH_CALUDE_equal_positive_integers_l3353_335328

theorem equal_positive_integers (a b c n : ℕ+) 
  (eq1 : a^2 + b^2 = n * Nat.lcm a b + n^2)
  (eq2 : b^2 + c^2 = n * Nat.lcm b c + n^2)
  (eq3 : c^2 + a^2 = n * Nat.lcm c a + n^2) :
  a = b ∧ b = c := by
  sorry

end NUMINAMATH_CALUDE_equal_positive_integers_l3353_335328


namespace NUMINAMATH_CALUDE_min_wednesday_birthdays_is_eight_l3353_335357

/-- The minimum number of employees with birthdays on Wednesday -/
def min_wednesday_birthdays (total_employees : ℕ) (days_in_week : ℕ) : ℕ :=
  let other_days := days_in_week - 1
  let max_other_day_birthdays := (total_employees - 1) / days_in_week
  max_other_day_birthdays + 1

/-- Prove that given 50 employees, excluding those born in March, and with Wednesday having more 
    birthdays than any other day of the week (which all have an equal number of birthdays), 
    the minimum number of employees having birthdays on Wednesday is 8. -/
theorem min_wednesday_birthdays_is_eight :
  min_wednesday_birthdays 50 7 = 8 := by
  sorry

#eval min_wednesday_birthdays 50 7

end NUMINAMATH_CALUDE_min_wednesday_birthdays_is_eight_l3353_335357


namespace NUMINAMATH_CALUDE_ratio_problem_l3353_335326

theorem ratio_problem (a b c d : ℝ) 
  (h1 : a / b = 5)
  (h2 : b / c = 1 / 2)
  (h3 : c / d = 7) :
  d / a = 2 / 35 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l3353_335326


namespace NUMINAMATH_CALUDE_sin_120_degrees_l3353_335365

theorem sin_120_degrees : Real.sin (2 * Real.pi / 3) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_120_degrees_l3353_335365


namespace NUMINAMATH_CALUDE_complex_expression_equality_l3353_335363

theorem complex_expression_equality : 
  (Real.sqrt 3 - 1)^2 + (Real.sqrt 3 - Real.sqrt 2) * (Real.sqrt 2 + Real.sqrt 3) + 
  (Real.sqrt 2 + 1) / (Real.sqrt 2 - 1) - 3 * Real.sqrt (1/2) = 
  8 - 2 * Real.sqrt 3 + Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_complex_expression_equality_l3353_335363


namespace NUMINAMATH_CALUDE_john_reading_speed_l3353_335373

/-- Calculates the number of pages read per hour given the total pages, reading duration in weeks, and daily reading hours. -/
def pages_per_hour (total_pages : ℕ) (weeks : ℕ) (hours_per_day : ℕ) : ℚ :=
  (total_pages : ℚ) / ((weeks * 7 : ℕ) * hours_per_day)

/-- Theorem stating that under the given conditions, John reads 50 pages per hour. -/
theorem john_reading_speed :
  let total_pages : ℕ := 2800
  let weeks : ℕ := 4
  let hours_per_day : ℕ := 2
  pages_per_hour total_pages weeks hours_per_day = 50 := by
  sorry

end NUMINAMATH_CALUDE_john_reading_speed_l3353_335373


namespace NUMINAMATH_CALUDE_jerry_age_l3353_335345

theorem jerry_age (mickey_age jerry_age : ℕ) : 
  mickey_age = 18 → 
  mickey_age = 2 * jerry_age - 2 → 
  jerry_age = 10 := by
sorry

end NUMINAMATH_CALUDE_jerry_age_l3353_335345


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3353_335384

theorem polynomial_factorization (x y : ℝ) : 3 * x^2 - 3 * y^2 = 3 * (x + y) * (x - y) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3353_335384


namespace NUMINAMATH_CALUDE_min_sum_given_log_sum_l3353_335314

theorem min_sum_given_log_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : Real.log a / Real.log 5 + Real.log b / Real.log 5 = 2) : 
  a + b ≥ 10 ∧ ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ 
    Real.log x / Real.log 5 + Real.log y / Real.log 5 = 2 ∧ x + y = 10 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_given_log_sum_l3353_335314


namespace NUMINAMATH_CALUDE_diet_soda_sales_l3353_335309

theorem diet_soda_sales (total_sodas : ℕ) (regular_ratio diet_ratio : ℕ) (diet_sodas : ℕ) : 
  total_sodas = 64 →
  regular_ratio = 9 →
  diet_ratio = 7 →
  regular_ratio * diet_sodas = diet_ratio * (total_sodas - diet_sodas) →
  diet_sodas = 28 := by
sorry

end NUMINAMATH_CALUDE_diet_soda_sales_l3353_335309


namespace NUMINAMATH_CALUDE_original_jeans_price_l3353_335351

/-- Proves that the original price of jeans is $49.00 given the discount conditions --/
theorem original_jeans_price (x : ℝ) : 
  (0.5 * x - 10 = 14.5) → x = 49 := by
  sorry

end NUMINAMATH_CALUDE_original_jeans_price_l3353_335351


namespace NUMINAMATH_CALUDE_monotonic_increasing_condition_l3353_335369

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 2*x^2 + a*x + 3

-- State the theorem
theorem monotonic_increasing_condition (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, Monotone (f a)) → a ≥ 1 := by sorry

end NUMINAMATH_CALUDE_monotonic_increasing_condition_l3353_335369


namespace NUMINAMATH_CALUDE_parabola_vertex_after_transformation_l3353_335300

/-- The vertex of a parabola after transformation -/
theorem parabola_vertex_after_transformation :
  let f (x : ℝ) := (x - 2)^2 - 2*(x - 2) + 6
  ∃! (h : ℝ × ℝ), (h.1 = 3 ∧ h.2 = 5 ∧ ∀ x, f x ≥ f h.1) :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_after_transformation_l3353_335300


namespace NUMINAMATH_CALUDE_count_ordered_pairs_l3353_335335

theorem count_ordered_pairs : ∃! n : ℕ, n = (Finset.filter (fun p : ℕ × ℕ => 
  p.1 > 0 ∧ p.2 > 0 ∧ p.1 * 4 = 6 * p.2) (Finset.product (Finset.range 25) (Finset.range 25))).card ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_count_ordered_pairs_l3353_335335


namespace NUMINAMATH_CALUDE_kylie_coins_left_l3353_335306

/-- The number of coins Kylie collected and gave away -/
structure CoinCollection where
  piggy_bank : ℕ
  from_brother : ℕ
  from_father : ℕ
  given_away : ℕ

/-- Calculate the number of coins Kylie has left -/
def coins_left (c : CoinCollection) : ℕ :=
  c.piggy_bank + c.from_brother + c.from_father - c.given_away

/-- Theorem stating that Kylie has 15 coins left -/
theorem kylie_coins_left :
  ∀ (c : CoinCollection),
  c.piggy_bank = 15 →
  c.from_brother = 13 →
  c.from_father = 8 →
  c.given_away = 21 →
  coins_left c = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_kylie_coins_left_l3353_335306


namespace NUMINAMATH_CALUDE_product_digit_sum_l3353_335317

def digit_repeat (d₁ d₂ d₃ : ℕ) (n : ℕ) : ℕ :=
  (d₁ * 10^2 + d₂ * 10 + d₃) * (10^(3*n) - 1) / 999

def a : ℕ := digit_repeat 3 0 3 33
def b : ℕ := digit_repeat 5 0 5 33

theorem product_digit_sum :
  (a * b % 10) + ((a * b / 1000) % 10) = 8 := by sorry

end NUMINAMATH_CALUDE_product_digit_sum_l3353_335317


namespace NUMINAMATH_CALUDE_smallest_non_representable_l3353_335342

def representable (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), n = (2^a - 2^b) / (2^c - 2^d)

theorem smallest_non_representable : ∀ k < 11, representable k ∧ ¬ representable 11 :=
sorry

end NUMINAMATH_CALUDE_smallest_non_representable_l3353_335342


namespace NUMINAMATH_CALUDE_ajay_ride_distance_l3353_335310

/-- Given Ajay's speed and travel time, calculate the distance he rides -/
theorem ajay_ride_distance (speed : ℝ) (time : ℝ) (distance : ℝ) : 
  speed = 50 → time = 30 → distance = speed * time → distance = 1500 :=
by
  sorry

#check ajay_ride_distance

end NUMINAMATH_CALUDE_ajay_ride_distance_l3353_335310


namespace NUMINAMATH_CALUDE_max_collisions_l3353_335389

/-- Represents an ant walking on a line -/
structure Ant where
  position : ℝ
  speed : ℝ
  direction : Bool -- true for right, false for left

/-- The state of the system at any given time -/
structure AntSystem where
  n : ℕ
  ants : Fin n → Ant

/-- Predicate to check if the total number of collisions is finite -/
def HasFiniteCollisions (system : AntSystem) : Prop := sorry

/-- The number of collisions that have occurred in the system -/
def NumberOfCollisions (system : AntSystem) : ℕ := sorry

/-- Theorem stating the maximum number of collisions possible -/
theorem max_collisions (n : ℕ) (h : n > 0) :
  ∃ (system : AntSystem),
    system.n = n ∧
    HasFiniteCollisions system ∧
    ∀ (other_system : AntSystem),
      other_system.n = n →
      HasFiniteCollisions other_system →
      NumberOfCollisions other_system ≤ NumberOfCollisions system ∧
      NumberOfCollisions system = n * (n - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_max_collisions_l3353_335389


namespace NUMINAMATH_CALUDE_ratio_in_specific_arithmetic_sequence_l3353_335377

-- Define an arithmetic sequence
def is_arithmetic_sequence (s : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, s (n + 1) - s n = d

-- Define our specific sequence
def our_sequence (s : ℕ → ℝ) (a m b : ℝ) : Prop :=
  s 0 = a ∧ s 1 = m ∧ s 2 = b ∧ s 3 = 3*m

-- State the theorem
theorem ratio_in_specific_arithmetic_sequence (s : ℕ → ℝ) (a m b : ℝ) :
  is_arithmetic_sequence s → our_sequence s a m b → b / a = -2 :=
by sorry

end NUMINAMATH_CALUDE_ratio_in_specific_arithmetic_sequence_l3353_335377


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_square_trisection_l3353_335360

/-- Given an ellipse where the two trisection points on the minor axis and its two foci form a square,
    prove that its eccentricity is √10/10 -/
theorem ellipse_eccentricity_square_trisection (a b c : ℝ) :
  b = 3 * c →                    -- Condition: trisection points and foci form a square
  a ^ 2 = b ^ 2 + c ^ 2 →        -- Definition: relationship between semi-major axis, semi-minor axis, and focal distance
  c / a = (Real.sqrt 10) / 10 := by
sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_square_trisection_l3353_335360


namespace NUMINAMATH_CALUDE_port_distance_l3353_335372

/-- Represents a ship traveling between two ports -/
structure Ship where
  speed : ℝ
  trips : ℕ

/-- Represents the problem setup -/
structure PortProblem where
  blue : Ship
  green : Ship
  first_meeting_distance : ℝ
  total_distance : ℝ

/-- The theorem stating the distance between ports -/
theorem port_distance (p : PortProblem) 
  (h1 : p.blue.trips = 4)
  (h2 : p.green.trips = 3)
  (h3 : p.first_meeting_distance = 20)
  (h4 : p.blue.speed / p.green.speed = p.first_meeting_distance / (p.total_distance - p.first_meeting_distance))
  (h5 : p.blue.speed * p.blue.trips = p.green.speed * p.green.trips) :
  p.total_distance = 35 := by
  sorry

end NUMINAMATH_CALUDE_port_distance_l3353_335372


namespace NUMINAMATH_CALUDE_senate_subcommittee_seating_l3353_335374

/-- The number of ways to arrange senators around a circular table -/
def arrange_senators (num_democrats : ℕ) (num_republicans : ℕ) : ℕ :=
  -- Arrangements of 2 blocks (Democrats and Republicans) in a circle
  1 *
  -- Permutations of Democrats within their block
  (Nat.factorial num_democrats) *
  -- Permutations of Republicans within their block
  (Nat.factorial num_republicans)

/-- Theorem stating the number of arrangements for 6 Democrats and 6 Republicans -/
theorem senate_subcommittee_seating :
  arrange_senators 6 6 = 518400 :=
by sorry

end NUMINAMATH_CALUDE_senate_subcommittee_seating_l3353_335374


namespace NUMINAMATH_CALUDE_odometer_square_sum_l3353_335336

theorem odometer_square_sum : ∃ (a b c : ℕ),
  (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧
  (100 ≤ 100 * a + 10 * b + c) ∧ (100 * a + 10 * b + c < 1000) ∧
  (100 ≤ 100 * b + 10 * c + a) ∧ (100 * b + 10 * c + a < 1000) ∧
  ((100 * b + 10 * c + a) - (100 * a + 10 * b + c)) % 60 = 0 ∧
  a^2 + b^2 + c^2 = 77 := by
sorry

end NUMINAMATH_CALUDE_odometer_square_sum_l3353_335336


namespace NUMINAMATH_CALUDE_elisa_family_women_without_daughters_l3353_335388

/-- Represents a family tree starting from Elisa -/
structure ElisaFamily where
  daughters : Nat
  granddaughters : Nat
  daughters_with_children : Nat

/-- The conditions of Elisa's family -/
def elisa_family : ElisaFamily where
  daughters := 8
  granddaughters := 28
  daughters_with_children := 4

/-- The total number of daughters and granddaughters -/
def total_descendants (f : ElisaFamily) : Nat :=
  f.daughters + f.granddaughters

/-- The number of women (daughters and granddaughters) who have no daughters -/
def women_without_daughters (f : ElisaFamily) : Nat :=
  (f.daughters - f.daughters_with_children) + f.granddaughters

/-- Theorem stating that 32 of Elisa's daughters and granddaughters have no daughters -/
theorem elisa_family_women_without_daughters :
  women_without_daughters elisa_family = 32 := by
  sorry

end NUMINAMATH_CALUDE_elisa_family_women_without_daughters_l3353_335388


namespace NUMINAMATH_CALUDE_triangle_max_area_l3353_335396

theorem triangle_max_area (a b c : ℝ) (h : 2 * a^2 + b^2 + c^2 = 4) :
  let S := (1/2) * a * b * Real.sqrt (1 - ((b^2 + c^2 - a^2) / (2*b*c))^2)
  S ≤ Real.sqrt 5 / 5 := by
sorry

end NUMINAMATH_CALUDE_triangle_max_area_l3353_335396


namespace NUMINAMATH_CALUDE_parallel_perpendicular_plane_l3353_335344

/-- Two lines are parallel -/
def parallel (m n : Line) : Prop := sorry

/-- A line is perpendicular to a plane -/
def perpendicular (l : Line) (α : Plane) : Prop := sorry

/-- Main theorem: If two lines are parallel and one is perpendicular to a plane, 
    then the other is also perpendicular to that plane -/
theorem parallel_perpendicular_plane (m n : Line) (α : Plane) :
  parallel m n → perpendicular m α → perpendicular n α := by sorry

end NUMINAMATH_CALUDE_parallel_perpendicular_plane_l3353_335344


namespace NUMINAMATH_CALUDE_expected_red_pairs_51_17_l3353_335322

/-- The expected number of red adjacent pairs in a circular arrangement of cards -/
def expected_red_pairs (total_cards : ℕ) (red_cards : ℕ) : ℚ :=
  (total_cards : ℚ) * (red_cards : ℚ) / (total_cards : ℚ) * ((red_cards - 1) : ℚ) / ((total_cards - 1) : ℚ)

/-- Theorem: Expected number of red adjacent pairs in a specific card arrangement -/
theorem expected_red_pairs_51_17 :
  expected_red_pairs 51 17 = 464 / 85 := by
  sorry

end NUMINAMATH_CALUDE_expected_red_pairs_51_17_l3353_335322


namespace NUMINAMATH_CALUDE_impossibleAllGood_l3353_335305

/-- A mushroom is either good or bad -/
inductive MushroomType
  | Good
  | Bad

/-- Definition of a mushroom -/
structure Mushroom where
  wormCount : ℕ
  type : MushroomType

/-- A basket of mushrooms -/
structure Basket where
  mushrooms : List Mushroom

/-- Function to determine if a mushroom is good -/
def isGoodMushroom (m : Mushroom) : Prop :=
  m.wormCount < 10

/-- Initial basket setup -/
def initialBasket : Basket :=
  { mushrooms := List.append
      (List.replicate 100 { wormCount := 10, type := MushroomType.Bad })
      (List.replicate 11 { wormCount := 0, type := MushroomType.Good }) }

/-- Theorem: It's impossible for all mushrooms to become good after redistribution -/
theorem impossibleAllGood (b : Basket) : ¬ ∀ m ∈ b.mushrooms, isGoodMushroom m := by
  sorry

#check impossibleAllGood initialBasket

end NUMINAMATH_CALUDE_impossibleAllGood_l3353_335305


namespace NUMINAMATH_CALUDE_coffee_milk_problem_l3353_335343

/-- Represents the liquid mixture in a cup -/
structure Mixture where
  coffee : ℚ
  milk : ℚ

/-- The process of mixing and transferring liquids -/
def mix_and_transfer (coffee_cup milk_cup : Mixture) : Mixture :=
  let transferred_coffee := coffee_cup.coffee / 3
  let mixed_cup := Mixture.mk (milk_cup.coffee + transferred_coffee) milk_cup.milk
  let total_mixed := mixed_cup.coffee + mixed_cup.milk
  let transferred_back := total_mixed / 2
  let coffee_ratio := mixed_cup.coffee / total_mixed
  let milk_ratio := mixed_cup.milk / total_mixed
  Mixture.mk 
    (coffee_cup.coffee - transferred_coffee + transferred_back * coffee_ratio)
    (transferred_back * milk_ratio)

theorem coffee_milk_problem :
  let initial_coffee_cup := Mixture.mk 6 0
  let initial_milk_cup := Mixture.mk 0 3
  let final_coffee_cup := mix_and_transfer initial_coffee_cup initial_milk_cup
  final_coffee_cup.milk / (final_coffee_cup.coffee + final_coffee_cup.milk) = 3 / 13 := by
  sorry

end NUMINAMATH_CALUDE_coffee_milk_problem_l3353_335343


namespace NUMINAMATH_CALUDE_daps_dops_dips_equivalence_l3353_335313

/-- Given that 5 daps are equivalent to 4 dops and 3 dops are equivalent to 11 dips,
    prove that 22.5 daps are equivalent to 66 dips. -/
theorem daps_dops_dips_equivalence 
  (h1 : (5 : ℚ) / 4 = daps_per_dop) 
  (h2 : (3 : ℚ) / 11 = dops_per_dip) : 
  (66 : ℚ) * daps_per_dop * dops_per_dip = (45 : ℚ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_daps_dops_dips_equivalence_l3353_335313


namespace NUMINAMATH_CALUDE_part1_part2_part3_l3353_335358

-- Define the variables and constants
variable (x y : ℝ)  -- x: quantity of vegetable A, y: quantity of vegetable B
def total_weight : ℝ := 40
def total_cost : ℝ := 180
def wholesale_price_A : ℝ := 4.8
def wholesale_price_B : ℝ := 4
def retail_price_A : ℝ := 7.2
def retail_price_B : ℝ := 5.6
def new_total_weight : ℝ := 80
def min_profit : ℝ := 176

-- Part 1
theorem part1 : 
  x + y = total_weight ∧ 
  wholesale_price_A * x + wholesale_price_B * y = total_cost → 
  x = 25 ∧ y = 15 := by sorry

-- Part 2
def m (n : ℝ) : ℝ := wholesale_price_A * n + wholesale_price_B * (new_total_weight - n)

theorem part2 : m n = 0.8 * n + 320 := by sorry

-- Part 3
def profit (n : ℝ) : ℝ := (retail_price_A - wholesale_price_A) * n + 
                           (retail_price_B - wholesale_price_B) * (new_total_weight - n)

theorem part3 : 
  ∀ n : ℝ, profit n ≥ min_profit → n ≥ 60 := by sorry

end NUMINAMATH_CALUDE_part1_part2_part3_l3353_335358


namespace NUMINAMATH_CALUDE_correct_age_difference_l3353_335321

/-- The difference between Priya's father's age and Priya's age -/
def ageDifference (priyaAge fatherAge : ℕ) : ℕ :=
  fatherAge - priyaAge

theorem correct_age_difference :
  let priyaAge : ℕ := 11
  let fatherAge : ℕ := 42
  let futureSum : ℕ := 69
  let yearsLater : ℕ := 8
  (priyaAge + yearsLater) + (fatherAge + yearsLater) = futureSum →
  ageDifference priyaAge fatherAge = 31 := by
  sorry

end NUMINAMATH_CALUDE_correct_age_difference_l3353_335321
