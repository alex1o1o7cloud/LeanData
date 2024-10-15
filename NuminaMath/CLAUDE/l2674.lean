import Mathlib

namespace NUMINAMATH_CALUDE_age_calculation_l2674_267452

-- Define the current ages and time intervals
def luke_current_age : ℕ := 20
def years_to_future : ℕ := 8
def years_to_luke_future : ℕ := 4

-- Define the relationships between ages
def mr_bernard_future_age : ℕ := 3 * luke_current_age
def luke_future_age : ℕ := luke_current_age + years_to_future
def sarah_future_age : ℕ := 2 * (luke_current_age + years_to_luke_future)

-- Calculate the average future age
def average_future_age : ℚ := (mr_bernard_future_age + luke_future_age + sarah_future_age) / 3

-- Define the final result
def result : ℚ := average_future_age - 10

-- Theorem to prove
theorem age_calculation :
  result = 35 + 1/3 :=
sorry

end NUMINAMATH_CALUDE_age_calculation_l2674_267452


namespace NUMINAMATH_CALUDE_eva_math_score_difference_l2674_267407

/-- Represents Eva's scores in a semester -/
structure SemesterScores where
  maths : ℕ
  arts : ℕ
  science : ℕ

/-- Calculates the total score for a semester -/
def totalScore (scores : SemesterScores) : ℕ :=
  scores.maths + scores.arts + scores.science

/-- Represents Eva's scores for the year -/
structure YearScores where
  first : SemesterScores
  second : SemesterScores

/-- The problem statement -/
theorem eva_math_score_difference 
  (year : YearScores)
  (h1 : year.second.maths = 80)
  (h2 : year.second.arts = 90)
  (h3 : year.second.science = 90)
  (h4 : year.first.arts = year.second.arts - 15)
  (h5 : year.first.science = year.second.science - year.second.science / 3)
  (h6 : totalScore year.first + totalScore year.second = 485)
  : year.first.maths = year.second.maths + 10 := by
  sorry

end NUMINAMATH_CALUDE_eva_math_score_difference_l2674_267407


namespace NUMINAMATH_CALUDE_jogging_distance_three_weeks_l2674_267468

/-- Calculates the total miles jogged over a given number of weeks -/
def total_miles_jogged (miles_per_day : ℕ) (days_per_week : ℕ) (num_weeks : ℕ) : ℕ :=
  miles_per_day * days_per_week * num_weeks

/-- Theorem: A person jogging 5 miles per day on weekdays for three weeks covers 75 miles -/
theorem jogging_distance_three_weeks :
  total_miles_jogged 5 5 3 = 75 := by
  sorry

end NUMINAMATH_CALUDE_jogging_distance_three_weeks_l2674_267468


namespace NUMINAMATH_CALUDE_root_product_rational_l2674_267457

-- Define the polynomial f(z)
def f (a b c d e : ℤ) (z : ℂ) : ℂ := a * z^4 + b * z^3 + c * z^2 + d * z + e

-- Define the roots r1, r2, r3, r4
variable (r1 r2 r3 r4 : ℂ)

-- State the theorem
theorem root_product_rational
  (a b c d e : ℤ)
  (h_a_nonzero : a ≠ 0)
  (h_f_factored : ∀ z, f a b c d e z = a * (z - r1) * (z - r2) * (z - r3) * (z - r4))
  (h_sum_rational : ∃ q : ℚ, (r1 + r2 : ℂ) = q)
  (h_sum_distinct : r1 + r2 ≠ r3 + r4) :
  ∃ q : ℚ, (r1 * r2 : ℂ) = q :=
sorry

end NUMINAMATH_CALUDE_root_product_rational_l2674_267457


namespace NUMINAMATH_CALUDE_lucky_larry_coincidence_l2674_267467

theorem lucky_larry_coincidence : ∃ e : ℝ, 
  let a : ℝ := 5
  let b : ℝ := 3
  let c : ℝ := 4
  let d : ℝ := 2
  (a + b - c + d - e) = (a + (b - (c + (d - e)))) := by
  sorry

end NUMINAMATH_CALUDE_lucky_larry_coincidence_l2674_267467


namespace NUMINAMATH_CALUDE_advanced_ticket_price_is_14_50_l2674_267447

/-- The price of an advanced ticket for the Rhapsody Theater -/
def advanced_ticket_price (total_tickets : ℕ) (door_price : ℚ) (total_revenue : ℚ) (door_tickets : ℕ) : ℚ :=
  (total_revenue - door_price * door_tickets) / (total_tickets - door_tickets)

/-- Theorem stating that the advanced ticket price is $14.50 given the specific conditions -/
theorem advanced_ticket_price_is_14_50 :
  advanced_ticket_price 800 22 16640 672 = 14.5 := by
  sorry

#eval advanced_ticket_price 800 22 16640 672

end NUMINAMATH_CALUDE_advanced_ticket_price_is_14_50_l2674_267447


namespace NUMINAMATH_CALUDE_colorings_count_l2674_267406

/-- The number of ways to color the edges of an m × n rectangle with three colors,
    such that each unit square has two sides of one color and two sides of another color. -/
def colorings (m n : ℕ) : ℕ :=
  18 * 2^(m*n - 1) * 3^(m + n - 2)

/-- Theorem stating that the number of valid colorings for an m × n rectangle
    with three colors is equal to 18 × 2^(mn-1) × 3^(m+n-2). -/
theorem colorings_count (m n : ℕ) :
  colorings m n = 18 * 2^(m*n - 1) * 3^(m + n - 2) :=
by sorry

end NUMINAMATH_CALUDE_colorings_count_l2674_267406


namespace NUMINAMATH_CALUDE_negative_one_and_half_equality_l2674_267496

theorem negative_one_and_half_equality : -1 - (1/2 : ℚ) = -(3/2 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_negative_one_and_half_equality_l2674_267496


namespace NUMINAMATH_CALUDE_composite_expression_l2674_267462

theorem composite_expression (a b : ℕ) : 
  ∃ (p q : ℕ), p > 1 ∧ q > 1 ∧ 4*a^2 + 4*a*b + 4*a + 2*b + 1 = p * q :=
by sorry

end NUMINAMATH_CALUDE_composite_expression_l2674_267462


namespace NUMINAMATH_CALUDE_quadratic_root_condition_l2674_267430

theorem quadratic_root_condition (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   x₁^2 + k*x₁ + 4*k^2 - 3 = 0 ∧ 
   x₂^2 + k*x₂ + 4*k^2 - 3 = 0 ∧
   x₁ + x₂ = x₁ * x₂) → 
  k = 3/4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_condition_l2674_267430


namespace NUMINAMATH_CALUDE_ball_selection_probability_l2674_267403

theorem ball_selection_probability (n : ℕ) (h : n = 48) : 
  (Nat.choose (n - 6) 7) / (Nat.choose n 7) = 
  (6 * Nat.choose (n - 7) 6) / (Nat.choose n 7) := by
  sorry

end NUMINAMATH_CALUDE_ball_selection_probability_l2674_267403


namespace NUMINAMATH_CALUDE_interest_period_is_two_years_l2674_267490

/-- Given simple interest rate of 20% per annum and simple interest of $400,
    and compound interest of $440 for the same period and rate,
    prove that the time period is 2 years. -/
theorem interest_period_is_two_years 
  (simple_interest : ℝ) 
  (compound_interest : ℝ) 
  (rate : ℝ) :
  simple_interest = 400 →
  compound_interest = 440 →
  rate = 0.20 →
  ∃ t : ℝ, t = 2 ∧ (1 + rate)^t = (rate * simple_interest * t + simple_interest) / simple_interest :=
by sorry

end NUMINAMATH_CALUDE_interest_period_is_two_years_l2674_267490


namespace NUMINAMATH_CALUDE_multiplicand_difference_l2674_267439

theorem multiplicand_difference (a b : ℕ) : 
  a * b = 100100 → 
  a < b → 
  a % 10 = 2 → 
  b % 10 = 6 → 
  b - a = 564 := by
sorry

end NUMINAMATH_CALUDE_multiplicand_difference_l2674_267439


namespace NUMINAMATH_CALUDE_symmetric_derivative_minimum_value_l2674_267480

-- Define the function f
def f (b c x : ℝ) : ℝ := x^3 + b*x^2 + c*x

-- Define the derivative of f
def f' (b c x : ℝ) : ℝ := 3*x^2 + 2*b*x + c

-- State the theorem
theorem symmetric_derivative_minimum_value (b c : ℝ) :
  (∀ x : ℝ, f' b c (4 - x) = f' b c x) →  -- f' is symmetric about x = 2
  (∃ t : ℝ, ∀ x : ℝ, f b c t ≤ f b c x) →  -- f has a minimum value
  (b = -6) ∧  -- Part 1: value of b
  (∃ g : ℝ → ℝ, (∀ t > 2, g t = f b c t) ∧  -- Part 2: domain of g
                (∀ y : ℝ, (∃ t > 2, g t = y) ↔ y < 8))  -- Part 3: range of g
  := by sorry

end NUMINAMATH_CALUDE_symmetric_derivative_minimum_value_l2674_267480


namespace NUMINAMATH_CALUDE_plane_equation_correct_l2674_267466

/-- Represents a plane in 3D space -/
structure Plane where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ
  A_pos : A > 0
  gcd_one : Nat.gcd (Nat.gcd (Int.natAbs A) (Int.natAbs B)) (Nat.gcd (Int.natAbs C) (Int.natAbs D)) = 1

/-- Checks if a point lies on a plane -/
def Plane.contains (p : Plane) (x y z : ℤ) : Prop :=
  p.A * x + p.B * y + p.C * z + p.D = 0

/-- Checks if a vector is perpendicular to another vector -/
def isPerpendicular (x1 y1 z1 x2 y2 z2 : ℤ) : Prop :=
  x1 * x2 + y1 * y2 + z1 * z2 = 0

theorem plane_equation_correct : ∃ (p : Plane),
  p.contains 10 (-2) 5 ∧
  isPerpendicular p.A p.B p.C 10 (-2) 5 ∧
  p.A = 10 ∧ p.B = -2 ∧ p.C = 5 ∧ p.D = -129 := by
  sorry

end NUMINAMATH_CALUDE_plane_equation_correct_l2674_267466


namespace NUMINAMATH_CALUDE_probability_yellow_ball_l2674_267463

/-- Probability of choosing a yellow ball from a bag -/
theorem probability_yellow_ball (red yellow blue : ℕ) (h : red = 2 ∧ yellow = 5 ∧ blue = 4) :
  (yellow : ℚ) / (red + yellow + blue : ℚ) = 5 / 11 := by
  sorry

end NUMINAMATH_CALUDE_probability_yellow_ball_l2674_267463


namespace NUMINAMATH_CALUDE_prob_divisible_by_3_and_8_prob_divisible_by_3_and_8_value_l2674_267446

/-- The probability of a randomly selected three-digit number being divisible by both 3 and 8 -/
theorem prob_divisible_by_3_and_8 : ℚ :=
  let three_digit_numbers := Finset.Icc 100 999
  let divisible_by_24 := three_digit_numbers.filter (λ n => n % 24 = 0)
  (divisible_by_24.card : ℚ) / three_digit_numbers.card

/-- The probability of a randomly selected three-digit number being divisible by both 3 and 8 is 37/900 -/
theorem prob_divisible_by_3_and_8_value :
  prob_divisible_by_3_and_8 = 37 / 900 := by
  sorry


end NUMINAMATH_CALUDE_prob_divisible_by_3_and_8_prob_divisible_by_3_and_8_value_l2674_267446


namespace NUMINAMATH_CALUDE_decreased_equilateral_angle_l2674_267436

/-- The measure of an angle in an equilateral triangle -/
def equilateral_angle : ℝ := 60

/-- The amount by which angle E is decreased -/
def angle_decrease : ℝ := 15

/-- Theorem: In an equilateral triangle where one angle is decreased by 15 degrees, 
    the measure of the decreased angle is 45 degrees -/
theorem decreased_equilateral_angle :
  equilateral_angle - angle_decrease = 45 := by sorry

end NUMINAMATH_CALUDE_decreased_equilateral_angle_l2674_267436


namespace NUMINAMATH_CALUDE_solution_satisfies_system_l2674_267424

/-- The system of linear equations -/
def system (x : ℝ × ℝ × ℝ × ℝ) : Prop :=
  let (x₁, x₂, x₃, x₄) := x
  x₁ + 2*x₂ + 3*x₃ + x₄ = 1 ∧
  3*x₁ + 13*x₂ + 13*x₃ + 5*x₄ = 3 ∧
  3*x₁ + 7*x₂ + 7*x₃ + 2*x₄ = 12 ∧
  x₁ + 5*x₂ + 3*x₃ + x₄ = 7 ∧
  4*x₁ + 5*x₂ + 6*x₃ + x₄ = 19

/-- The general solution to the system -/
def solution (α : ℝ) : ℝ × ℝ × ℝ × ℝ :=
  (4 - α, 2, α, -7 - 2*α)

/-- Theorem stating that the general solution satisfies the system for any α -/
theorem solution_satisfies_system :
  ∀ α : ℝ, system (solution α) :=
by sorry

end NUMINAMATH_CALUDE_solution_satisfies_system_l2674_267424


namespace NUMINAMATH_CALUDE_repeating_decimal_37_l2674_267409

/-- The repeating decimal 0.373737... expressed as a rational number -/
theorem repeating_decimal_37 : ∃ (x : ℚ), x = 37 / 99 ∧ 
  ∀ (n : ℕ), (100 * x - ⌊100 * x⌋ : ℚ) * 10^n = (37 * 10^n : ℚ) % 100 / 100 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_37_l2674_267409


namespace NUMINAMATH_CALUDE_cubic_trinomial_condition_l2674_267484

/-- 
Given a polynomial of the form 3xy^(|m|) - (1/4)(m-2)xy + 1,
prove that for it to be a cubic trinomial, m must equal -2.
-/
theorem cubic_trinomial_condition (m : ℤ) : 
  (abs m = 2) ∧ ((1/4 : ℚ) * (m - 2) ≠ 0) → m = -2 := by sorry

end NUMINAMATH_CALUDE_cubic_trinomial_condition_l2674_267484


namespace NUMINAMATH_CALUDE_unique_linear_function_l2674_267473

/-- A linear function passing through two given points -/
def linear_function_through_points (k b : ℝ) (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  k ≠ 0 ∧ y₁ = k * x₁ + b ∧ y₂ = k * x₂ + b

theorem unique_linear_function :
  ∃! k b : ℝ, linear_function_through_points k b 1 3 0 (-2) ∧ 
  ∀ x : ℝ, k * x + b = 5 * x - 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_linear_function_l2674_267473


namespace NUMINAMATH_CALUDE_prob_even_sum_is_14_27_l2674_267426

/-- Represents an unfair die where even numbers are twice as likely as odd numbers -/
structure UnfairDie where
  /-- Probability of rolling an odd number -/
  odd_prob : ℝ
  /-- Probability of rolling an even number -/
  even_prob : ℝ
  /-- Ensures probabilities sum to 1 -/
  prob_sum : odd_prob + even_prob = 1
  /-- Ensures even numbers are twice as likely as odd numbers -/
  even_twice_odd : even_prob = 2 * odd_prob

/-- Represents the result of rolling the die three times -/
def ThreeRolls := Fin 3 → Bool

/-- The probability of getting an even sum when rolling the unfair die three times -/
def prob_even_sum (d : UnfairDie) : ℝ :=
  (d.even_prob^3) + 3 * (d.even_prob * d.odd_prob^2)

theorem prob_even_sum_is_14_27 (d : UnfairDie) : prob_even_sum d = 14/27 := by
  sorry

end NUMINAMATH_CALUDE_prob_even_sum_is_14_27_l2674_267426


namespace NUMINAMATH_CALUDE_number_of_valid_arrangements_l2674_267421

-- Define the triangular arrangement
structure TriangularArrangement :=
  (cells : Fin 9 → Nat)

-- Define the condition for valid placement
def ValidPlacement (arr : TriangularArrangement) : Prop :=
  -- Each number from 1 to 9 is used exactly once
  (∀ n : Fin 9, ∃! i : Fin 9, arr.cells i = n.val + 1) ∧
  -- The sum in each four-cell triangle is 23
  (arr.cells 0 + arr.cells 1 + arr.cells 3 + arr.cells 4 = 23) ∧
  (arr.cells 1 + arr.cells 2 + arr.cells 4 + arr.cells 5 = 23) ∧
  (arr.cells 3 + arr.cells 4 + arr.cells 6 + arr.cells 7 = 23) ∧
  -- Specific placements as indicated by arrows
  (arr.cells 3 = 7 ∨ arr.cells 6 = 7) ∧
  (arr.cells 1 = 2 ∨ arr.cells 2 = 2 ∨ arr.cells 4 = 2 ∨ arr.cells 5 = 2)

-- The theorem to be proved
theorem number_of_valid_arrangements :
  ∃! (arrangements : Finset TriangularArrangement),
    (∀ arr ∈ arrangements, ValidPlacement arr) ∧
    arrangements.card = 4 := by
  sorry

end NUMINAMATH_CALUDE_number_of_valid_arrangements_l2674_267421


namespace NUMINAMATH_CALUDE_largest_three_digit_number_with_1_hundreds_l2674_267441

def digits : List Nat := [1, 5, 6, 9]

def isValidNumber (n : Nat) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧
  (n / 100 = 1) ∧
  (∀ d, d ∈ digits → (n / 10 % 10 = d ∨ n % 10 = d))

theorem largest_three_digit_number_with_1_hundreds :
  ∀ n : Nat, isValidNumber n → n ≤ 196 :=
sorry

end NUMINAMATH_CALUDE_largest_three_digit_number_with_1_hundreds_l2674_267441


namespace NUMINAMATH_CALUDE_chocolateProblemSolution_l2674_267419

def chocolateProblem (totalBoxes : Float) (piecesPerBox : Float) (remainingPieces : Nat) : Float :=
  let totalPieces := totalBoxes * piecesPerBox
  let givenPieces := totalPieces - remainingPieces.toFloat
  givenPieces / piecesPerBox

theorem chocolateProblemSolution :
  chocolateProblem 14.0 6.0 42 = 7.0 := by
  sorry

end NUMINAMATH_CALUDE_chocolateProblemSolution_l2674_267419


namespace NUMINAMATH_CALUDE_add_three_tenths_to_57_7_l2674_267482

theorem add_three_tenths_to_57_7 : (57.7 : ℝ) + (3 / 10 : ℝ) = 58 := by
  sorry

end NUMINAMATH_CALUDE_add_three_tenths_to_57_7_l2674_267482


namespace NUMINAMATH_CALUDE_lcm_of_20_45_75_l2674_267491

theorem lcm_of_20_45_75 : Nat.lcm 20 (Nat.lcm 45 75) = 900 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_20_45_75_l2674_267491


namespace NUMINAMATH_CALUDE_inverse_composition_problem_l2674_267422

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the inverse functions
variable (f_inv g_inv : ℝ → ℝ)

-- State the theorem
theorem inverse_composition_problem
  (h : ∀ x, f_inv (g x) = 3 * x + 5)
  (h_inv_f : ∀ x, f_inv (f x) = x)
  (h_inv_g : ∀ x, g_inv (g x) = x)
  (h_f_inv : ∀ x, f (f_inv x) = x)
  (h_g_inv : ∀ x, g (g_inv x) = x) :
  g_inv (f (-8)) = -13/3 :=
sorry

end NUMINAMATH_CALUDE_inverse_composition_problem_l2674_267422


namespace NUMINAMATH_CALUDE_probability_ratio_l2674_267408

def num_balls : ℕ := 25
def num_bins : ℕ := 6

def probability_config_1 : ℚ :=
  (Nat.choose num_bins 2 * (Nat.factorial num_balls / (Nat.factorial 3 * Nat.factorial 3 * (Nat.factorial 5)^4))) /
  (num_bins^num_balls : ℚ)

def probability_config_2 : ℚ :=
  (Nat.choose num_bins 2 * Nat.choose (num_bins - 2) 2 * (Nat.factorial num_balls / (Nat.factorial 3 * Nat.factorial 3 * (Nat.factorial 4)^2 * (Nat.factorial 5)^2))) /
  (num_bins^num_balls : ℚ)

theorem probability_ratio :
  probability_config_1 / probability_config_2 = 625 / 6 := by
  sorry

end NUMINAMATH_CALUDE_probability_ratio_l2674_267408


namespace NUMINAMATH_CALUDE_unique_three_digit_factorial_sum_l2674_267485

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def digit_factorial_sum (n : ℕ) : ℕ :=
  (n.digits 10).map factorial |>.sum

def does_not_contain_five (n : ℕ) : Prop :=
  5 ∉ n.digits 10

theorem unique_three_digit_factorial_sum :
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ does_not_contain_five n ∧ n = digit_factorial_sum n :=
  by sorry

end NUMINAMATH_CALUDE_unique_three_digit_factorial_sum_l2674_267485


namespace NUMINAMATH_CALUDE_hawks_score_l2674_267427

theorem hawks_score (total_score : ℕ) (eagles_margin : ℕ) (eagles_three_pointers : ℕ) 
  (h1 : total_score = 82)
  (h2 : eagles_margin = 18)
  (h3 : eagles_three_pointers = 6) : 
  total_score / 2 - eagles_margin / 2 = 32 := by
  sorry

#check hawks_score

end NUMINAMATH_CALUDE_hawks_score_l2674_267427


namespace NUMINAMATH_CALUDE_parallelogram_base_l2674_267464

/-- 
Given a parallelogram with area 320 cm² and height 16 cm, 
prove that its base is 20 cm.
-/
theorem parallelogram_base (area : ℝ) (height : ℝ) (base : ℝ) :
  area = 320 ∧ height = 16 ∧ area = base * height → base = 20 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_base_l2674_267464


namespace NUMINAMATH_CALUDE_ellipse_intersection_l2674_267404

/-- Definition of the ellipse with given properties -/
def ellipse (P : ℝ × ℝ) : Prop :=
  let F₁ : ℝ × ℝ := (0, 3)
  let F₂ : ℝ × ℝ := (4, 0)
  Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) + 
  Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) = 7

theorem ellipse_intersection :
  ellipse (0, 0) → 
  (∃ x : ℝ, x ≠ 0 ∧ ellipse (x, 0)) → 
  ellipse (56/11, 0) :=
sorry

end NUMINAMATH_CALUDE_ellipse_intersection_l2674_267404


namespace NUMINAMATH_CALUDE_scientific_notation_361000000_l2674_267455

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_361000000 :
  toScientificNotation 361000000 = ScientificNotation.mk 3.61 8 sorry := by sorry

end NUMINAMATH_CALUDE_scientific_notation_361000000_l2674_267455


namespace NUMINAMATH_CALUDE_right_triangle_altitude_relation_l2674_267476

theorem right_triangle_altitude_relation (a b c x : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : x > 0)
  (h5 : c^2 = a^2 + b^2)  -- Pythagorean theorem
  (h6 : a * b = c * x)    -- Area relation
  : 1 / x^2 = 1 / a^2 + 1 / b^2 := by
  sorry

#check right_triangle_altitude_relation

end NUMINAMATH_CALUDE_right_triangle_altitude_relation_l2674_267476


namespace NUMINAMATH_CALUDE_min_tiles_needed_l2674_267456

/-- Represents the dimensions of a rectangular object -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Converts feet to inches -/
def feetToInches (feet : ℕ) : ℕ := feet * 12

/-- Calculates the area of a rectangle in square inches -/
def areaInSquareInches (rect : Rectangle) : ℕ := rect.length * rect.width

/-- Calculates the number of small rectangles needed to cover a larger rectangle -/
def tilesNeeded (smallRect : Rectangle) (largeRect : Rectangle) : ℕ :=
  (areaInSquareInches largeRect) / (areaInSquareInches smallRect)

theorem min_tiles_needed :
  let tile := Rectangle.mk 2 3
  let room := Rectangle.mk (feetToInches 3) (feetToInches 6)
  tilesNeeded tile room = 432 := by sorry

end NUMINAMATH_CALUDE_min_tiles_needed_l2674_267456


namespace NUMINAMATH_CALUDE_exists_nonperiodic_with_repeating_subsequence_l2674_267442

/-- A sequence of natural numbers -/
def Sequence := ℕ → ℕ

/-- Property: For any index k, there exists a t such that the sequence repeats at multiples of t -/
def HasRepeatingSubsequence (a : Sequence) : Prop :=
  ∀ k : ℕ, ∃ t : ℕ, ∀ n : ℕ, a k = a (k + n * t)

/-- Property: A sequence is periodic -/
def IsPeriodic (a : Sequence) : Prop :=
  ∃ T : ℕ, ∀ k : ℕ, a k = a (k + T)

/-- Theorem: There exists a sequence that has repeating subsequences but is not periodic -/
theorem exists_nonperiodic_with_repeating_subsequence :
  ∃ a : Sequence, HasRepeatingSubsequence a ∧ ¬IsPeriodic a :=
sorry

end NUMINAMATH_CALUDE_exists_nonperiodic_with_repeating_subsequence_l2674_267442


namespace NUMINAMATH_CALUDE_rectangular_field_area_l2674_267489

/-- Proves that a rectangular field with sides in ratio 3:4 and fencing cost of 91 rupees at 0.25 rupees per meter has an area of 8112 square meters -/
theorem rectangular_field_area (x : ℝ) (cost_per_meter : ℝ) (total_cost : ℝ) :
  x > 0 →
  cost_per_meter = 0.25 →
  total_cost = 91 →
  (14 * x * cost_per_meter = total_cost) →
  (3 * x) * (4 * x) = 8112 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l2674_267489


namespace NUMINAMATH_CALUDE_negation_of_proposition_l2674_267459

theorem negation_of_proposition (f : ℝ → ℝ) :
  (¬ ∀ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) ≥ 0) ↔ 
  (∃ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l2674_267459


namespace NUMINAMATH_CALUDE_smallest_upper_bound_l2674_267440

theorem smallest_upper_bound (x : ℤ) 
  (h1 : 3 < x ∧ x < 10)
  (h2 : 5 < x ∧ x < 18)
  (h3 : x > -2)
  (h4 : 0 < x ∧ x < 8)
  (h5 : x + 1 < 9) :
  ∃ (upper_bound : ℤ), 
    (∀ y : ℤ, (3 < y ∧ y < 10) → 
               (5 < y ∧ y < 18) → 
               (y > -2) → 
               (0 < y ∧ y < 8) → 
               (y + 1 < 9) → 
               y ≤ upper_bound) ∧
    (upper_bound = 8) :=
sorry

end NUMINAMATH_CALUDE_smallest_upper_bound_l2674_267440


namespace NUMINAMATH_CALUDE_fox_can_catch_mole_l2674_267431

/-- Represents a mound in the line of 100 mounds. -/
def Mound := Fin 100

/-- Represents the state of the game at any given time. -/
structure GameState where
  molePosition : Mound
  foxPosition : Mound

/-- Represents a strategy for the fox. -/
def FoxStrategy := GameState → Mound

/-- Represents the result of a single move in the game. -/
inductive MoveResult
  | Caught
  | Continue (newState : GameState)

/-- Simulates a single move in the game. -/
def makeMove (state : GameState) (strategy : FoxStrategy) : MoveResult :=
  sorry

/-- Simulates the game for a given number of moves. -/
def playGame (initialState : GameState) (strategy : FoxStrategy) (moves : Nat) : Bool :=
  sorry

/-- The main theorem stating that there exists a strategy for the fox to catch the mole. -/
theorem fox_can_catch_mole :
  ∃ (strategy : FoxStrategy), ∀ (initialState : GameState),
    playGame initialState strategy 200 = true :=
  sorry

end NUMINAMATH_CALUDE_fox_can_catch_mole_l2674_267431


namespace NUMINAMATH_CALUDE_exchange_process_duration_l2674_267448

/-- Represents the maximum number of exchanges possible in the described process -/
def max_exchanges (n : ℕ) : ℕ := n - 1

/-- The number of children in the line -/
def total_children : ℕ := 20

/-- The theorem stating that the exchange process cannot continue for more than an hour -/
theorem exchange_process_duration : max_exchanges total_children < 60 := by
  sorry


end NUMINAMATH_CALUDE_exchange_process_duration_l2674_267448


namespace NUMINAMATH_CALUDE_last_passenger_probability_l2674_267495

/-- Represents a bus with n seats and n passengers -/
structure Bus (n : ℕ) where
  seats : Fin n → Passenger
  tickets : Fin n → Passenger

/-- Represents a passenger -/
inductive Passenger
| scientist
| regular (i : ℕ)

/-- The seating strategy for passengers -/
def seatingStrategy (b : Bus n) : Bus n := sorry

/-- The probability that the last passenger sits in their assigned seat -/
def lastPassengerProbability (n : ℕ) : ℚ :=
  if n < 2 then 0 else 1/2

/-- Theorem stating that the probability of the last passenger sitting in their assigned seat is 1/2 for n ≥ 2 -/
theorem last_passenger_probability (n : ℕ) (h : n ≥ 2) :
  lastPassengerProbability n = 1/2 := by sorry

end NUMINAMATH_CALUDE_last_passenger_probability_l2674_267495


namespace NUMINAMATH_CALUDE_investment_income_is_648_l2674_267488

/-- Calculates the annual income from a stock investment given the total investment,
    share face value, quoted price, and dividend rate. -/
def annual_income (total_investment : ℚ) (face_value : ℚ) (quoted_price : ℚ) (dividend_rate : ℚ) : ℚ :=
  let num_shares := total_investment / quoted_price
  let dividend_per_share := (dividend_rate / 100) * face_value
  num_shares * dividend_per_share

/-- Theorem stating that the annual income for the given investment scenario is 648. -/
theorem investment_income_is_648 :
  annual_income 4455 10 8.25 12 = 648 := by
  sorry

end NUMINAMATH_CALUDE_investment_income_is_648_l2674_267488


namespace NUMINAMATH_CALUDE_total_bird_families_l2674_267443

/-- The number of bird families that migrated to Africa -/
def africa : ℕ := 42

/-- The number of bird families that migrated to Asia -/
def asia : ℕ := 31

/-- The difference between the number of families that migrated to Africa and Asia -/
def difference : ℕ := 11

/-- Theorem: The total number of bird families before migration is 73 -/
theorem total_bird_families : africa + asia = 73 ∧ africa = asia + difference := by
  sorry

end NUMINAMATH_CALUDE_total_bird_families_l2674_267443


namespace NUMINAMATH_CALUDE_f_properties_l2674_267450

def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + p.2, p.1 * p.2)

theorem f_properties :
  let f : ℝ × ℝ → ℝ × ℝ := λ p ↦ (p.1 + p.2, p.1 * p.2)
  (f (1, -2) = (-1, -2)) ∧
  (f (2, -1) = (1, -2)) ∧
  (f (-1, 2) = (1, -2)) ∧
  (∀ a b : ℝ, f (a, b) = (1, -2) → (a = 2 ∧ b = -1) ∨ (a = -1 ∧ b = 2)) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2674_267450


namespace NUMINAMATH_CALUDE_point_on_y_axis_l2674_267429

/-- 
If a point P with coordinates (a-1, a²-9) lies on the y-axis, 
then its coordinates are (0, -8).
-/
theorem point_on_y_axis (a : ℝ) : 
  (a - 1 = 0) → (a - 1, a^2 - 9) = (0, -8) := by
  sorry

end NUMINAMATH_CALUDE_point_on_y_axis_l2674_267429


namespace NUMINAMATH_CALUDE_subtraction_of_decimals_l2674_267469

theorem subtraction_of_decimals : 2.5 - 0.32 = 2.18 := by sorry

end NUMINAMATH_CALUDE_subtraction_of_decimals_l2674_267469


namespace NUMINAMATH_CALUDE_complement_union_of_sets_l2674_267449

open Set

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {3, 5}

theorem complement_union_of_sets : 
  (A ∪ B)ᶜ = {2, 4} :=
by sorry

end NUMINAMATH_CALUDE_complement_union_of_sets_l2674_267449


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2674_267460

theorem imaginary_part_of_z (m : ℝ) (z : ℂ) : 
  z = 1 - m * I ∧ z = -2 * I → z.im = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2674_267460


namespace NUMINAMATH_CALUDE_smallest_integer_greater_than_neg_seventeen_thirds_l2674_267405

theorem smallest_integer_greater_than_neg_seventeen_thirds :
  Int.ceil (-17 / 3 : ℚ) = -5 := by sorry

end NUMINAMATH_CALUDE_smallest_integer_greater_than_neg_seventeen_thirds_l2674_267405


namespace NUMINAMATH_CALUDE_union_equality_implies_m_value_l2674_267481

def A (m : ℝ) : Set ℝ := {1, 3, Real.sqrt m}
def B (m : ℝ) : Set ℝ := {1, m}

theorem union_equality_implies_m_value (m : ℝ) :
  A m ∪ B m = A m → m = 0 ∨ m = 3 := by
  sorry

end NUMINAMATH_CALUDE_union_equality_implies_m_value_l2674_267481


namespace NUMINAMATH_CALUDE_unique_symmetry_center_l2674_267434

/-- A point is symmetric to another point with respect to a center -/
def isSymmetric (A B O : ℝ × ℝ) : Prop :=
  A.1 + B.1 = 2 * O.1 ∧ A.2 + B.2 = 2 * O.2

/-- A point is a symmetry center of a set of points -/
def isSymmetryCenter (O : ℝ × ℝ) (H : Set (ℝ × ℝ)) : Prop :=
  ∀ A ∈ H, ∃ B ∈ H, isSymmetric A B O

theorem unique_symmetry_center (H : Set (ℝ × ℝ)) (hfin : Set.Finite H) :
  ∀ O O' : ℝ × ℝ, isSymmetryCenter O H → isSymmetryCenter O' H → O = O' := by
  sorry

#check unique_symmetry_center

end NUMINAMATH_CALUDE_unique_symmetry_center_l2674_267434


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2674_267423

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (3 + Real.sqrt x) = 4 → x = 169 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2674_267423


namespace NUMINAMATH_CALUDE_quadratic_range_at_minus_two_l2674_267414

/-- A quadratic function passing through the origin -/
structure QuadraticThroughOrigin where
  a : ℝ
  b : ℝ
  a_nonzero : a ≠ 0

/-- The quadratic function f(x) = ax² + bx -/
def f (q : QuadraticThroughOrigin) (x : ℝ) : ℝ :=
  q.a * x^2 + q.b * x

/-- Theorem: For a quadratic function f(x) = ax² + bx (a ≠ 0) passing through the origin,
    if 1 ≤ f(-1) ≤ 2 and 2 ≤ f(1) ≤ 4, then 5 ≤ f(-2) ≤ 10 -/
theorem quadratic_range_at_minus_two (q : QuadraticThroughOrigin) 
    (h1 : 1 ≤ f q (-1)) (h2 : f q (-1) ≤ 2)
    (h3 : 2 ≤ f q 1) (h4 : f q 1 ≤ 4) :
    5 ≤ f q (-2) ∧ f q (-2) ≤ 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_range_at_minus_two_l2674_267414


namespace NUMINAMATH_CALUDE_return_probability_is_one_sixth_l2674_267471

/-- Represents a regular tetrahedron -/
structure RegularTetrahedron :=
  (vertices : Finset (Fin 4))
  (edges : Finset (Fin 4 × Fin 4))
  (adjacent : Fin 4 → Finset (Fin 4))
  (adjacent_sym : ∀ v₁ v₂, v₂ ∈ adjacent v₁ ↔ v₁ ∈ adjacent v₂)
  (adjacent_card : ∀ v, (adjacent v).card = 3)

/-- The probability of returning to the starting vertex in two moves -/
def return_probability (t : RegularTetrahedron) : ℚ :=
  1 / 6

/-- Theorem: The probability of returning to the starting vertex in two moves is 1/6 -/
theorem return_probability_is_one_sixth (t : RegularTetrahedron) :
  return_probability t = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_return_probability_is_one_sixth_l2674_267471


namespace NUMINAMATH_CALUDE_change_after_purchase_l2674_267472

/-- Calculates the change after a purchase given initial amount, number of items, and cost per item. -/
def calculate_change (initial_amount : ℕ) (num_items : ℕ) (cost_per_item : ℕ) : ℕ :=
  initial_amount - (num_items * cost_per_item)

/-- Theorem stating that given $20 initially, buying 3 items at $2 each results in $14 change. -/
theorem change_after_purchase :
  calculate_change 20 3 2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_change_after_purchase_l2674_267472


namespace NUMINAMATH_CALUDE_right_triangle_area_l2674_267474

theorem right_triangle_area (a b : ℝ) (h1 : a^2 - 7*a + 12 = 0) (h2 : b^2 - 7*b + 12 = 0) (h3 : a ≠ b) :
  ∃ (area : ℝ), (area = 6 ∨ area = (3 * Real.sqrt 7) / 2) ∧
  ((area = a * b / 2) ∨ (area = a * Real.sqrt (b^2 - a^2) / 2) ∨ (area = b * Real.sqrt (a^2 - b^2) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2674_267474


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l2674_267475

-- Define the given line
def given_line (x y : ℝ) : Prop := x - 2*y + 1 = 0

-- Define the point that the desired line passes through
def point : ℝ × ℝ := (-1, 3)

-- Define the desired line
def desired_line (x y : ℝ) : Prop := y + 2*x - 1 = 0

-- Theorem statement
theorem perpendicular_line_through_point :
  (∀ x y, given_line x y → desired_line x y → (x - point.1) * (x - point.1) + (y - point.2) * (y - point.2) = 0) ∧
  (∀ x₁ y₁ x₂ y₂, given_line x₁ y₁ → given_line x₂ y₂ → desired_line x₁ y₁ → desired_line x₂ y₂ →
    (x₂ - x₁) * (x₂ - x₁) + (y₂ - y₁) * (y₂ - y₁) ≠ 0 →
    ((x₂ - x₁) * (y₂ - y₁)) / ((x₂ - x₁) * (x₂ - x₁) + (y₂ - y₁) * (y₂ - y₁)) = -1/2) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l2674_267475


namespace NUMINAMATH_CALUDE_sum_of_quadratic_solutions_l2674_267444

theorem sum_of_quadratic_solutions (x : ℝ) : 
  (x^2 + 6*x - 22 = 4*x - 18) → 
  (∃ a b : ℝ, (a + b = -2) ∧ (x = a ∨ x = b)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_quadratic_solutions_l2674_267444


namespace NUMINAMATH_CALUDE_triangle_angle_proof_l2674_267438

theorem triangle_angle_proof (A B C a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π → -- Triangle condition
  a * Real.cos B = 3 * b * Real.cos A → -- Given equation
  B = A - π / 6 → -- Given relation between A and B
  B = π / 6 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_proof_l2674_267438


namespace NUMINAMATH_CALUDE_julia_play_difference_l2674_267470

/-- The number of kids Julia played tag with on Monday -/
def monday_tag : ℕ := 28

/-- The number of kids Julia played hide & seek with on Monday -/
def monday_hide_seek : ℕ := 15

/-- The number of kids Julia played tag with on Tuesday -/
def tuesday_tag : ℕ := 33

/-- The number of kids Julia played hide & seek with on Tuesday -/
def tuesday_hide_seek : ℕ := 21

/-- The difference in the total number of kids Julia played with on Tuesday compared to Monday -/
theorem julia_play_difference : 
  (tuesday_tag + tuesday_hide_seek) - (monday_tag + monday_hide_seek) = 11 := by
  sorry

end NUMINAMATH_CALUDE_julia_play_difference_l2674_267470


namespace NUMINAMATH_CALUDE_circle_center_travel_distance_l2674_267494

-- Define the triangle
def triangle_sides : (ℝ × ℝ × ℝ) := (5, 12, 13)

-- Define the circle radius
def circle_radius : ℝ := 2

-- Define the function to calculate the perimeter of the inscribed triangle
def inscribed_triangle_perimeter (sides : ℝ × ℝ × ℝ) (radius : ℝ) : ℝ :=
  let (a, b, c) := sides
  (a - 2 * radius) + (b - 2 * radius) + (c - 2 * radius)

-- Theorem statement
theorem circle_center_travel_distance :
  inscribed_triangle_perimeter triangle_sides circle_radius = 18 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_travel_distance_l2674_267494


namespace NUMINAMATH_CALUDE_line_intercepts_sum_l2674_267478

/-- Given a line with equation y - 7 = -3(x + 2), 
    prove that the sum of its x-intercept and y-intercept is 4/3 -/
theorem line_intercepts_sum (x y : ℝ) :
  (y - 7 = -3 * (x + 2)) →
  ∃ (x_int y_int : ℝ),
    (x_int - 7 = -3 * (x_int + 2)) ∧  -- x-intercept condition
    (0 - 7 = -3 * (x_int + 2)) ∧      -- x-intercept definition
    (y_int - 7 = -3 * (0 + 2)) ∧      -- y-intercept condition
    (x_int + y_int = 4/3) :=
by sorry

end NUMINAMATH_CALUDE_line_intercepts_sum_l2674_267478


namespace NUMINAMATH_CALUDE_tangent_line_equation_l2674_267413

/-- The equation of the tangent line to the curve y = x sin x at the point (π, 0) is y = -πx + π² -/
theorem tangent_line_equation (x y : ℝ) : 
  (y = x * Real.sin x) → -- Curve equation
  (∃ (m b : ℝ), (y = m * x + b) ∧ -- Tangent line equation
                (0 = m * π + b) ∧ -- Point (π, 0) satisfies the tangent line equation
                (m = Real.sin π + π * Real.cos π)) → -- Slope of the tangent line
  (y = -π * x + π^2) -- Resulting tangent line equation
:= by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l2674_267413


namespace NUMINAMATH_CALUDE_proof_by_contradiction_elements_l2674_267418

/-- Elements used as conditions in a proof by contradiction -/
inductive ProofByContradictionElement
  | NegatedConclusion
  | OriginalConditions
  | AxiomsTheoremsDefinitions
  | OriginalConclusion

/-- The set of elements that should be used in a proof by contradiction -/
def ValidProofByContradictionElements : Set ProofByContradictionElement :=
  {ProofByContradictionElement.NegatedConclusion,
   ProofByContradictionElement.OriginalConditions,
   ProofByContradictionElement.AxiomsTheoremsDefinitions}

/-- Theorem stating which elements should be used in a proof by contradiction -/
theorem proof_by_contradiction_elements :
  ValidProofByContradictionElements =
    {ProofByContradictionElement.NegatedConclusion,
     ProofByContradictionElement.OriginalConditions,
     ProofByContradictionElement.AxiomsTheoremsDefinitions} :=
by sorry

end NUMINAMATH_CALUDE_proof_by_contradiction_elements_l2674_267418


namespace NUMINAMATH_CALUDE_power_product_cube_l2674_267412

theorem power_product_cube (a b : ℝ) : (a * b) ^ 3 = a ^ 3 * b ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_power_product_cube_l2674_267412


namespace NUMINAMATH_CALUDE_prime_counting_inequality_characterize_equality_cases_l2674_267400

-- Define π(x) as the prime counting function
def prime_counting (x : ℕ) : ℕ := sorry

-- Define φ(x) as Euler's totient function
def euler_totient (x : ℕ) : ℕ := sorry

theorem prime_counting_inequality (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  prime_counting m - prime_counting n ≤ ((m - 1) * euler_totient n) / n :=
sorry

def equality_cases : List (ℕ × ℕ) :=
  [(1, 1), (2, 1), (3, 1), (3, 2), (5, 2), (7, 2)]

theorem characterize_equality_cases (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  (prime_counting m - prime_counting n = ((m - 1) * euler_totient n) / n) ↔
  (m, n) ∈ equality_cases :=
sorry

end NUMINAMATH_CALUDE_prime_counting_inequality_characterize_equality_cases_l2674_267400


namespace NUMINAMATH_CALUDE_a_plus_b_eighth_power_l2674_267433

theorem a_plus_b_eighth_power (a b : ℝ) 
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7) :
  a^8 + b^8 = 47 := by
  sorry

end NUMINAMATH_CALUDE_a_plus_b_eighth_power_l2674_267433


namespace NUMINAMATH_CALUDE_coloring_book_shelves_l2674_267458

theorem coloring_book_shelves (initial_stock : ℕ) (books_sold : ℕ) (books_per_shelf : ℕ) : 
  initial_stock = 27 → books_sold = 6 → books_per_shelf = 7 → 
  (initial_stock - books_sold) / books_per_shelf = 3 := by
sorry

end NUMINAMATH_CALUDE_coloring_book_shelves_l2674_267458


namespace NUMINAMATH_CALUDE_qin_jiushao_v3_equals_71_l2674_267479

-- Define the polynomial f(x)
def f (x : ℝ) : ℝ := 2*x^6 + 5*x^5 + 6*x^4 + 23*x^3 - 8*x^2 + 10*x - 3

-- Define Qin Jiushao's algorithm for calculating V₃
def qin_jiushao_v3 (x : ℝ) : ℝ :=
  let v0 := 2
  let v1 := v0 * x + 5
  let v2 := v1 * x + 6
  v2 * x + 23

-- Theorem statement
theorem qin_jiushao_v3_equals_71 :
  qin_jiushao_v3 2 = 71 :=
by sorry

end NUMINAMATH_CALUDE_qin_jiushao_v3_equals_71_l2674_267479


namespace NUMINAMATH_CALUDE_subset_sum_equals_A_l2674_267437

theorem subset_sum_equals_A (A : ℕ) (a : List ℕ) : 
  (∀ n ∈ Finset.range 9, A % (n + 1) = 0) →
  (∀ x ∈ a, x < 10) →
  (2 * A = a.sum) →
  ∃ s : List ℕ, s.toFinset ⊆ a.toFinset ∧ s.sum = A := by
  sorry

end NUMINAMATH_CALUDE_subset_sum_equals_A_l2674_267437


namespace NUMINAMATH_CALUDE_unique_solution_trigonometric_equation_l2674_267465

theorem unique_solution_trigonometric_equation :
  ∃! (n k m : ℕ), 1 ≤ n ∧ n ≤ 5 ∧
                  1 ≤ k ∧ k ≤ 5 ∧
                  1 ≤ m ∧ m ≤ 5 ∧
                  (Real.sin (π * n / 12) * Real.sin (π * k / 12) * Real.sin (π * m / 12) = 1 / 8) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_trigonometric_equation_l2674_267465


namespace NUMINAMATH_CALUDE_expression_equivalence_l2674_267454

theorem expression_equivalence (a b : ℝ) 
  (h1 : (a + b) - (a - b) ≠ 0) 
  (h2 : a + b + a - b ≠ 0) : 
  let P := a + b
  let Q := a - b
  ((P + Q)^2 / (P - Q)^2) - ((P - Q)^2 / (P + Q)^2) = (a^2 + b^2) * (a^2 - b^2) / (a^2 * b^2) := by
sorry

end NUMINAMATH_CALUDE_expression_equivalence_l2674_267454


namespace NUMINAMATH_CALUDE_line_parabola_intersection_range_l2674_267453

/-- The range of m for which a line and a parabola have exactly one common point -/
theorem line_parabola_intersection_range (m : ℝ) : 
  (∃! x : ℝ, -1 ≤ x ∧ x ≤ 3 ∧ 
   (2 * x - 2 * m = x^2 + m * x - 1)) ↔ 
  (-3/5 < m ∧ m < 5) :=
by sorry

end NUMINAMATH_CALUDE_line_parabola_intersection_range_l2674_267453


namespace NUMINAMATH_CALUDE_sum_product_ratio_l2674_267416

theorem sum_product_ratio (x y z : ℝ) (h_distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z) (h_sum : x + y + z = 3) :
  (x * y + y * z + z * x) / (x^2 + y^2 + z^2) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_product_ratio_l2674_267416


namespace NUMINAMATH_CALUDE_circle_condition_l2674_267461

/-- A quadratic equation in two variables represents a circle if and only if
    D^2 + E^2 - 4F > 0, where the equation is in the form x^2 + y^2 + Dx + Ey + F = 0 -/
def is_circle (D E F : ℝ) : Prop := D^2 + E^2 - 4*F > 0

/-- The equation x^2 + y^2 + x + y + k = 0 represents a circle -/
def represents_circle (k : ℝ) : Prop := is_circle 1 1 k

/-- If x^2 + y^2 + x + y + k = 0 represents a circle, then k < 1/2 -/
theorem circle_condition (k : ℝ) : represents_circle k → k < 1/2 := by
  sorry

end NUMINAMATH_CALUDE_circle_condition_l2674_267461


namespace NUMINAMATH_CALUDE_share_yield_calculation_l2674_267486

/-- Calculates the effective interest rate (yield) for a share --/
theorem share_yield_calculation (face_value : ℝ) (dividend_rate : ℝ) (market_value : ℝ) :
  face_value = 60 ∧ dividend_rate = 0.09 ∧ market_value = 45 →
  (face_value * dividend_rate) / market_value = 0.12 := by
  sorry

end NUMINAMATH_CALUDE_share_yield_calculation_l2674_267486


namespace NUMINAMATH_CALUDE_continued_fraction_value_l2674_267435

theorem continued_fraction_value : ∃ x : ℝ, 
  x = 3 + 4 / (1 + 4 / (3 + 4 / ((1/2) + x))) ∧ 
  x = (43 + Real.sqrt 4049) / 22 := by
  sorry

end NUMINAMATH_CALUDE_continued_fraction_value_l2674_267435


namespace NUMINAMATH_CALUDE_emily_minimum_grade_to_beat_ahmed_l2674_267497

/-- Represents a student's grade -/
structure StudentGrade where
  current_grade : ℕ
  final_grade : ℕ

/-- Calculates the final average grade given current grade and final assignment grade -/
def finalAverageGrade (s : StudentGrade) : ℚ :=
  (9 * s.current_grade + s.final_grade) / 10

theorem emily_minimum_grade_to_beat_ahmed :
  ∀ (ahmed emily : StudentGrade),
    ahmed.current_grade = 91 →
    emily.current_grade = 92 →
    ahmed.final_grade = 100 →
    (∀ g : ℕ, g < 92 → finalAverageGrade emily < finalAverageGrade ahmed) ∧
    finalAverageGrade { current_grade := 92, final_grade := 92 } > finalAverageGrade ahmed :=
by sorry

end NUMINAMATH_CALUDE_emily_minimum_grade_to_beat_ahmed_l2674_267497


namespace NUMINAMATH_CALUDE_probability_empty_bottle_day14_expected_pills_taken_l2674_267420

/-- Represents the pill-taking scenario with two bottles --/
structure PillScenario where
  totalDays : ℕ
  pillsPerBottle : ℕ
  bottles : ℕ

/-- Calculates the probability of finding an empty bottle on a specific day --/
def probabilityEmptyBottle (scenario : PillScenario) (day : ℕ) : ℚ :=
  sorry

/-- Calculates the expected number of pills taken when discovering an empty bottle --/
def expectedPillsTaken (scenario : PillScenario) : ℚ :=
  sorry

/-- Theorem stating the probability of finding an empty bottle on the 14th day --/
theorem probability_empty_bottle_day14 (scenario : PillScenario) :
  scenario.totalDays = 14 ∧ scenario.pillsPerBottle = 10 ∧ scenario.bottles = 2 →
  probabilityEmptyBottle scenario 14 = 143 / 4096 :=
sorry

/-- Theorem stating the expected number of pills taken when discovering an empty bottle --/
theorem expected_pills_taken (scenario : PillScenario) (ε : ℚ) :
  scenario.pillsPerBottle = 10 ∧ scenario.bottles = 2 →
  ∃ n : ℕ, abs (expectedPillsTaken scenario - 173 / 10) < ε ∧ n > 0 :=
sorry

end NUMINAMATH_CALUDE_probability_empty_bottle_day14_expected_pills_taken_l2674_267420


namespace NUMINAMATH_CALUDE_max_product_of_focal_distances_l2674_267432

/-- The ellipse equation -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 9 = 1

/-- The foci of the ellipse -/
def F1 : ℝ × ℝ := sorry
def F2 : ℝ × ℝ := sorry

/-- Distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- Statement: The maximum value of |PF1| * |PF2| is 25 for any point P on the ellipse -/
theorem max_product_of_focal_distances :
  ∀ P : ℝ × ℝ, is_on_ellipse P.1 P.2 →
  ∃ M : ℝ, M = 25 ∧ ∀ Q : ℝ × ℝ, is_on_ellipse Q.1 Q.2 →
  (distance P F1) * (distance P F2) ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_product_of_focal_distances_l2674_267432


namespace NUMINAMATH_CALUDE_smallest_in_odd_set_l2674_267451

/-- A set of consecutive odd integers -/
def ConsecutiveOddIntegers : Set ℤ := sorry

/-- The median of a set of integers -/
def median (s : Set ℤ) : ℚ := sorry

/-- The greatest integer in a set -/
def greatest (s : Set ℤ) : ℤ := sorry

/-- The smallest integer in a set -/
def smallest (s : Set ℤ) : ℤ := sorry

theorem smallest_in_odd_set (s : Set ℤ) :
  s = ConsecutiveOddIntegers ∧
  median s = 152.5 ∧
  greatest s = 161 →
  smallest s = 138 := by sorry

end NUMINAMATH_CALUDE_smallest_in_odd_set_l2674_267451


namespace NUMINAMATH_CALUDE_pizza_slices_left_l2674_267401

def large_pizza_slices : ℕ := 12
def small_pizza_slices : ℕ := 8
def num_large_pizzas : ℕ := 2
def num_small_pizzas : ℕ := 1

def dean_eaten : ℕ := large_pizza_slices / 2
def frank_eaten : ℕ := 3
def sammy_eaten : ℕ := large_pizza_slices / 3
def nancy_cheese_eaten : ℕ := 2
def nancy_pepperoni_eaten : ℕ := 1
def olivia_eaten : ℕ := 2

def total_slices : ℕ := num_large_pizzas * large_pizza_slices + num_small_pizzas * small_pizza_slices

def total_eaten : ℕ := dean_eaten + frank_eaten + sammy_eaten + nancy_cheese_eaten + nancy_pepperoni_eaten + olivia_eaten

theorem pizza_slices_left : total_slices - total_eaten = 14 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_left_l2674_267401


namespace NUMINAMATH_CALUDE_unique_positive_solution_l2674_267428

theorem unique_positive_solution :
  ∃! x : ℝ, x > 0 ∧ (2 * x^2 - 7)^2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l2674_267428


namespace NUMINAMATH_CALUDE_x_value_when_z_is_64_l2674_267415

/-- Given that x is inversely proportional to y², y is directly proportional to √z,
    and x = 4 when z = 16, prove that x = 1 when z = 64 -/
theorem x_value_when_z_is_64 
  (x y z : ℝ) 
  (h1 : ∃ (k : ℝ), x * y^2 = k) 
  (h2 : ∃ (m : ℝ), y = m * Real.sqrt z) 
  (h3 : x = 4 ∧ z = 16) : 
  z = 64 → x = 1 := by
sorry

end NUMINAMATH_CALUDE_x_value_when_z_is_64_l2674_267415


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l2674_267477

theorem trigonometric_equation_solution (t : ℝ) : 
  (16 * Real.sin (t / 2) - 25 * Real.cos (t / 2) ≠ 0) →
  (40 * (Real.sin (t / 2) ^ 3 - Real.cos (t / 2) ^ 3) / 
   (16 * Real.sin (t / 2) - 25 * Real.cos (t / 2)) = Real.sin t) ↔
  (∃ k : ℤ, t = 2 * Real.arctan (4 / 5) + 2 * Real.pi * ↑k) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l2674_267477


namespace NUMINAMATH_CALUDE_inequality_proof_l2674_267417

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (Real.sqrt (b + c) / a) + (Real.sqrt (c + a) / b) + (Real.sqrt (a + b) / c) ≥
  (4 * (a + b + c)) / Real.sqrt ((a + b) * (b + c) * (c + a)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2674_267417


namespace NUMINAMATH_CALUDE_vaccine_comparison_l2674_267410

/-- Represents a vaccine trial result -/
structure VaccineTrial where
  vaccinated : Nat
  infected : Nat

/-- Determines if a vaccine is considered effective based on trial results and population infection rate -/
def is_effective (trial : VaccineTrial) (population_rate : Real) : Prop :=
  (trial.infected : Real) / trial.vaccinated < population_rate

/-- Compares the effectiveness of two vaccines -/
def more_effective (trial1 trial2 : VaccineTrial) (population_rate : Real) : Prop :=
  is_effective trial1 population_rate ∧ is_effective trial2 population_rate ∧
  (trial1.infected : Real) / trial1.vaccinated < (trial2.infected : Real) / trial2.vaccinated

theorem vaccine_comparison :
  let population_rate : Real := 0.2
  let vaccine_I : VaccineTrial := ⟨8, 0⟩
  let vaccine_II : VaccineTrial := ⟨25, 1⟩
  more_effective vaccine_II vaccine_I population_rate :=
by
  sorry

end NUMINAMATH_CALUDE_vaccine_comparison_l2674_267410


namespace NUMINAMATH_CALUDE_unique_divisible_number_l2674_267411

/-- A function that constructs a five-digit number of the form 6n272 -/
def construct_number (n : Nat) : Nat :=
  60000 + n * 1000 + 272

/-- Proposition: 63272 is the only number of the form 6n272 (where n is a single digit) 
    that is divisible by both 11 and 5 -/
theorem unique_divisible_number : 
  ∃! n : Nat, n < 10 ∧ 
  (construct_number n).mod 11 = 0 ∧ 
  (construct_number n).mod 5 = 0 ∧
  construct_number n = 63272 := by
  sorry

end NUMINAMATH_CALUDE_unique_divisible_number_l2674_267411


namespace NUMINAMATH_CALUDE_certain_number_equation_l2674_267493

theorem certain_number_equation : ∃ x : ℤ, 9548 + 7314 = x + 13500 ∧ x = 3362 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_equation_l2674_267493


namespace NUMINAMATH_CALUDE_baker_sales_difference_l2674_267402

/-- Represents the baker's sales data --/
structure BakerSales where
  usual_pastries : ℕ
  usual_bread : ℕ
  today_pastries : ℕ
  today_bread : ℕ
  pastry_price : ℕ
  bread_price : ℕ

/-- Calculates the difference between today's sales and average daily sales --/
def sales_difference (s : BakerSales) : ℕ :=
  let usual_total := s.usual_pastries * s.pastry_price + s.usual_bread * s.bread_price
  let today_total := s.today_pastries * s.pastry_price + s.today_bread * s.bread_price
  today_total - usual_total

/-- Theorem stating the difference in sales --/
theorem baker_sales_difference :
  ∃ (s : BakerSales),
    s.usual_pastries = 20 ∧
    s.usual_bread = 10 ∧
    s.today_pastries = 14 ∧
    s.today_bread = 25 ∧
    s.pastry_price = 2 ∧
    s.bread_price = 4 ∧
    sales_difference s = 48 := by
  sorry

end NUMINAMATH_CALUDE_baker_sales_difference_l2674_267402


namespace NUMINAMATH_CALUDE_torn_sheets_count_l2674_267492

/-- Represents a book with numbered pages -/
structure Book where
  /-- The last page number in the book -/
  lastPage : ℕ

/-- Represents a range of torn out pages -/
structure TornPages where
  /-- The first torn out page number -/
  first : ℕ
  /-- The last torn out page number -/
  last : ℕ

/-- Check if two numbers have the same digits in any order -/
def sameDigits (a b : ℕ) : Prop := sorry

/-- Calculate the number of sheets torn out -/
def sheetsTornOut (book : Book) (torn : TornPages) : ℕ :=
  (torn.last - torn.first + 1) / 2

/-- The main theorem -/
theorem torn_sheets_count (book : Book) (torn : TornPages) :
  torn.first = 185 →
  sameDigits torn.first torn.last →
  torn.last % 2 = 0 →
  torn.first < torn.last →
  sheetsTornOut book torn = 167 := by sorry

end NUMINAMATH_CALUDE_torn_sheets_count_l2674_267492


namespace NUMINAMATH_CALUDE_reflected_ray_slope_l2674_267483

/-- A light ray is emitted from a point, reflects off the y-axis, and is tangent to a circle. -/
theorem reflected_ray_slope (emissionPoint : ℝ × ℝ) (circleCenter : ℝ × ℝ) (circleRadius : ℝ) :
  emissionPoint = (-2, -3) →
  circleCenter = (-3, 2) →
  circleRadius = 1 →
  ∃ (k : ℝ), (k = -4/3 ∨ k = -3/4) ∧
    (∀ (x y : ℝ), (x + 3)^2 + (y - 2)^2 = 1 →
      (k * x - y - 2 * k - 3 = 0 →
        ((3 * k + 2 + 2 * k + 3)^2 / (k^2 + 1) = 1))) :=
by sorry

end NUMINAMATH_CALUDE_reflected_ray_slope_l2674_267483


namespace NUMINAMATH_CALUDE_system_solution_l2674_267499

theorem system_solution : 
  let x : ℚ := -24/13
  let y : ℚ := 18/13
  let z : ℚ := -23/13
  (3*x + 2*y = z - 1) ∧ 
  (2*x - y = 4*z + 2) ∧ 
  (x + 4*y = 3*z + 9) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l2674_267499


namespace NUMINAMATH_CALUDE_surface_area_unchanged_l2674_267498

/-- Represents the dimensions of a rectangular solid -/
structure RectangularSolid where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents the dimensions of a cube -/
structure Cube where
  side : ℝ

/-- Calculates the surface area of a rectangular solid -/
def surfaceArea (r : RectangularSolid) : ℝ :=
  2 * (r.length * r.width + r.length * r.height + r.width * r.height)

/-- Calculates the exposed surface area of a cube when it touches two faces of the solid -/
def exposedCubeArea (c : Cube) : ℝ :=
  2 * c.side * c.side

/-- Theorem: The surface area remains unchanged after cube removal -/
theorem surface_area_unchanged 
  (original : RectangularSolid)
  (removed : Cube)
  (h1 : original.length = 5)
  (h2 : original.width = 3)
  (h3 : original.height = 4)
  (h4 : removed.side = 2)
  (h5 : exposedCubeArea removed = exposedCubeArea removed) :
  surfaceArea original = surfaceArea original :=
by sorry

end NUMINAMATH_CALUDE_surface_area_unchanged_l2674_267498


namespace NUMINAMATH_CALUDE_dean_has_30_insects_l2674_267445

-- Define the number of insects for each person
def angela_insects : ℕ := 75
def jacob_insects : ℕ := 2 * angela_insects
def dean_insects : ℕ := jacob_insects / 5

-- Theorem to prove
theorem dean_has_30_insects : dean_insects = 30 := by
  sorry

end NUMINAMATH_CALUDE_dean_has_30_insects_l2674_267445


namespace NUMINAMATH_CALUDE_integer_valued_poly_implies_24P_integer_coeffs_l2674_267425

/-- A polynomial of degree 4 that takes integer values for integer inputs -/
def IntegerValuedPolynomial (P : ℝ → ℝ) : Prop :=
  (∃ a b c d e : ℝ, ∀ x, P x = a*x^4 + b*x^3 + c*x^2 + d*x + e) ∧
  (∀ n : ℤ, ∃ m : ℤ, P n = m)

/-- The coefficients of 24P(x) are integers -/
def Coefficients24PAreIntegers (P : ℝ → ℝ) : Prop :=
  ∃ a' b' c' d' e' : ℤ, ∀ x, 24 * P x = a'*x^4 + b'*x^3 + c'*x^2 + d'*x + e'

theorem integer_valued_poly_implies_24P_integer_coeffs
  (P : ℝ → ℝ) (h : IntegerValuedPolynomial P) :
  Coefficients24PAreIntegers P :=
sorry

end NUMINAMATH_CALUDE_integer_valued_poly_implies_24P_integer_coeffs_l2674_267425


namespace NUMINAMATH_CALUDE_multiplication_value_proof_l2674_267487

theorem multiplication_value_proof (n r : ℚ) (hn : n = 9) (hr : r = 18) :
  ∃ x : ℚ, (n / 6) * x = r ∧ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_value_proof_l2674_267487
