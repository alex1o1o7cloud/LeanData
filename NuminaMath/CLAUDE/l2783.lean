import Mathlib

namespace NUMINAMATH_CALUDE_cube_color_probability_l2783_278367

/-- Represents the three possible colors for a cube face -/
inductive Color
  | Red
  | Blue
  | Yellow

/-- Represents a cube with colored faces -/
structure Cube where
  faces : Fin 6 → Color

/-- The probability of each color -/
def colorProb : Color → ℚ
  | _ => 1/3

/-- Checks if a cube configuration satisfies the condition -/
def satisfiesCondition (c : Cube) : Bool :=
  sorry -- Implementation details omitted

/-- Calculates the probability of a cube satisfying the condition -/
noncomputable def probabilityOfSatisfyingCondition : ℚ :=
  sorry -- Implementation details omitted

/-- The main theorem to prove -/
theorem cube_color_probability :
  probabilityOfSatisfyingCondition = 73/243 :=
sorry

end NUMINAMATH_CALUDE_cube_color_probability_l2783_278367


namespace NUMINAMATH_CALUDE_smallest_five_digit_base3_palindrome_is_10001_l2783_278316

/-- Checks if a number is a palindrome in a given base -/
def isPalindrome (n : ℕ) (base : ℕ) : Prop := sorry

/-- Converts a number from one base to another -/
def baseConvert (n : ℕ) (fromBase toBase : ℕ) : ℕ := sorry

/-- Returns the number of digits of a number in a given base -/
def numDigits (n : ℕ) (base : ℕ) : ℕ := sorry

theorem smallest_five_digit_base3_palindrome_is_10001 :
  ∃ (otherBase : ℕ),
    otherBase ≠ 3 ∧
    otherBase > 1 ∧
    isPalindrome 10001 3 ∧
    numDigits 10001 3 = 5 ∧
    isPalindrome (baseConvert 10001 3 otherBase) otherBase ∧
    numDigits (baseConvert 10001 3 otherBase) otherBase = 3 ∧
    ∀ (n : ℕ),
      n < 10001 →
      (isPalindrome n 3 ∧ numDigits n 3 = 5) →
      ¬∃ (b : ℕ), b ≠ 3 ∧ b > 1 ∧
        isPalindrome (baseConvert n 3 b) b ∧
        numDigits (baseConvert n 3 b) b = 3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_five_digit_base3_palindrome_is_10001_l2783_278316


namespace NUMINAMATH_CALUDE_expansion_coefficient_ratio_l2783_278334

theorem expansion_coefficient_ratio (n : ℕ) : 
  (∀ a b : ℝ, (4 : ℝ)^n / (2 : ℝ)^n = 64) → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_expansion_coefficient_ratio_l2783_278334


namespace NUMINAMATH_CALUDE_expression_result_l2783_278324

theorem expression_result : 
  (7899665 : ℝ) - 12 * 3 * 2 + (7^3) / Real.sqrt 144 = 7899621.5833 := by
sorry

end NUMINAMATH_CALUDE_expression_result_l2783_278324


namespace NUMINAMATH_CALUDE_jills_number_satisfies_conditions_l2783_278331

def jills_favorite_number := 98

theorem jills_number_satisfies_conditions :
  -- 98 is even
  Even jills_favorite_number ∧
  -- 98 has repeating prime factors
  ∃ p : Nat, Prime p ∧ (jills_favorite_number % (p * p) = 0) ∧
  -- 7 is a prime factor of 98
  jills_favorite_number % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_jills_number_satisfies_conditions_l2783_278331


namespace NUMINAMATH_CALUDE_gift_wrapping_calculation_l2783_278337

/-- Represents the gift wrapping scenario for Edmund's shop. -/
structure GiftWrapping where
  wrapper_per_day : ℕ        -- inches of gift wrapper per day
  boxes_per_period : ℕ       -- number of gift boxes wrapped in a period
  days_per_period : ℕ        -- number of days in a period
  wrapper_per_box : ℕ        -- inches of gift wrapper per gift box

/-- Theorem stating the relationship between gift wrapper usage and gift boxes wrapped. -/
theorem gift_wrapping_calculation (g : GiftWrapping)
  (h1 : g.wrapper_per_day = 90)
  (h2 : g.boxes_per_period = 15)
  (h3 : g.days_per_period = 3)
  : g.wrapper_per_box = 18 := by
  sorry


end NUMINAMATH_CALUDE_gift_wrapping_calculation_l2783_278337


namespace NUMINAMATH_CALUDE_problem_solution_l2783_278397

theorem problem_solution (x y z : ℚ) 
  (hx : x = 1/3) (hy : y = 1/2) (hz : z = 5/8) :
  x * y * (1 - z) = 1/16 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2783_278397


namespace NUMINAMATH_CALUDE_fixed_point_of_log_function_l2783_278335

-- Define the logarithm function
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define the function f(x) = 1 + logₐ x
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 + log a x

-- Theorem statement
theorem fixed_point_of_log_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_log_function_l2783_278335


namespace NUMINAMATH_CALUDE_probability_two_red_two_blue_l2783_278384

/-- The probability of selecting 2 red and 2 blue marbles from a bag containing 12 red marbles
    and 8 blue marbles, when 4 marbles are selected at random without replacement. -/
theorem probability_two_red_two_blue (total_marbles : ℕ) (red_marbles : ℕ) (blue_marbles : ℕ)
    (selected_marbles : ℕ) :
    total_marbles = red_marbles + blue_marbles →
    total_marbles = 20 →
    red_marbles = 12 →
    blue_marbles = 8 →
    selected_marbles = 4 →
    (Nat.choose red_marbles 2 * Nat.choose blue_marbles 2 : ℚ) /
    (Nat.choose total_marbles selected_marbles) = 56 / 147 :=
by sorry

end NUMINAMATH_CALUDE_probability_two_red_two_blue_l2783_278384


namespace NUMINAMATH_CALUDE_monotonicity_of_g_minimum_a_for_negative_f_l2783_278390

noncomputable section

def f (a x : ℝ) : ℝ := x * Real.log (x + 1) + (1/2 - a) * x + 2 - a

def g (a x : ℝ) : ℝ := f a x + Real.log (x + 1) + (1/2) * x

theorem monotonicity_of_g (a : ℝ) :
  (a ≤ 2 → StrictMono (g a)) ∧
  (a > 2 → StrictAntiOn (g a) (Set.Ioo 0 (Real.exp (a - 2) - 1)) ∧
           StrictMono (g a ∘ (λ x => x + Real.exp (a - 2) - 1))) :=
sorry

theorem minimum_a_for_negative_f :
  (∃ (a : ℤ), ∃ (x : ℝ), x ≥ 0 ∧ f a x < 0) ∧
  (∀ (a : ℤ), a < 3 → ∀ (x : ℝ), x ≥ 0 → f a x ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_monotonicity_of_g_minimum_a_for_negative_f_l2783_278390


namespace NUMINAMATH_CALUDE_quadratic_solution_l2783_278330

theorem quadratic_solution (a b : ℝ) : 
  (1 : ℝ)^2 * a - (1 : ℝ) * b - 5 = 0 → 2023 + a - b = 2028 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_l2783_278330


namespace NUMINAMATH_CALUDE_quadratic_properties_l2783_278341

def f (x : ℝ) := x^2 - 6*x + 8

theorem quadratic_properties :
  (∀ x, f x = (x - 2) * (x - 4)) ∧
  (∀ x, f x ≥ f 3) ∧
  (f 3 = -1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_properties_l2783_278341


namespace NUMINAMATH_CALUDE_cube_sum_divisible_by_nine_l2783_278370

theorem cube_sum_divisible_by_nine (n : ℕ+) :
  ∃ k : ℤ, (n : ℤ)^3 + (n + 1 : ℤ)^3 + (n + 2 : ℤ)^3 = 9 * k :=
by sorry

end NUMINAMATH_CALUDE_cube_sum_divisible_by_nine_l2783_278370


namespace NUMINAMATH_CALUDE_key_lime_requirement_l2783_278391

/-- The number of tablespoons in one cup -/
def tablespoons_per_cup : ℕ := 16

/-- The original amount of key lime juice in cups -/
def original_juice_cups : ℚ := 1/4

/-- The multiplication factor for the juice amount -/
def juice_multiplier : ℕ := 3

/-- The minimum amount of juice (in tablespoons) that a key lime can yield -/
def min_juice_per_lime : ℕ := 1

/-- The maximum amount of juice (in tablespoons) that a key lime can yield -/
def max_juice_per_lime : ℕ := 2

/-- The number of key limes needed to ensure enough juice for the recipe -/
def key_limes_needed : ℕ := 12

theorem key_lime_requirement :
  key_limes_needed * min_juice_per_lime ≥ 
  juice_multiplier * (original_juice_cups * tablespoons_per_cup) ∧
  key_limes_needed * max_juice_per_lime ≥
  juice_multiplier * (original_juice_cups * tablespoons_per_cup) ∧
  ∀ n : ℕ, n < key_limes_needed →
    n * min_juice_per_lime < juice_multiplier * (original_juice_cups * tablespoons_per_cup) :=
by sorry

end NUMINAMATH_CALUDE_key_lime_requirement_l2783_278391


namespace NUMINAMATH_CALUDE_point_on_line_l2783_278352

/-- A point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem point_on_line : 
  let A : Point := ⟨0, 3⟩
  let B : Point := ⟨-8, 0⟩
  let C : Point := ⟨16/3, 5⟩
  collinear A B C := by
  sorry


end NUMINAMATH_CALUDE_point_on_line_l2783_278352


namespace NUMINAMATH_CALUDE_base8_47_equals_39_l2783_278323

/-- Converts a two-digit base-8 number to base-10 --/
def base8_to_base10 (tens : Nat) (ones : Nat) : Nat :=
  tens * 8 + ones

/-- The base-8 number 47 is equal to 39 in base-10 --/
theorem base8_47_equals_39 : base8_to_base10 4 7 = 39 := by
  sorry

end NUMINAMATH_CALUDE_base8_47_equals_39_l2783_278323


namespace NUMINAMATH_CALUDE_pencil_problem_l2783_278388

theorem pencil_problem (red blue green : ℕ) 
  (h_red : red = 15) (h_blue : blue = 13) (h_green : green = 8) :
  let total := red + blue + green
  let min_required := 1 + 2 + 3
  ∃ (n : ℕ), n = 22 ∧ 
    (∀ (k : ℕ), k < n → 
      ∃ (r b g : ℕ), r + b + g = k ∧ r ≤ red ∧ b ≤ blue ∧ g ≤ green ∧ 
        (r < 1 ∨ b < 2 ∨ g < 3)) ∧
    (∀ (r b g : ℕ), r + b + g = n → r ≤ red → b ≤ blue → g ≤ green → 
      r ≥ 1 ∧ b ≥ 2 ∧ g ≥ 3) :=
by sorry

end NUMINAMATH_CALUDE_pencil_problem_l2783_278388


namespace NUMINAMATH_CALUDE_distance_between_cars_l2783_278398

/-- The distance between two cars on a road after they travel towards each other -/
theorem distance_between_cars (initial_distance car1_distance car2_distance : ℝ) :
  initial_distance = 150 ∧ 
  car1_distance = 50 ∧ 
  car2_distance = 35 →
  initial_distance - (car1_distance + car2_distance) = 65 := by
  sorry


end NUMINAMATH_CALUDE_distance_between_cars_l2783_278398


namespace NUMINAMATH_CALUDE_maria_score_l2783_278321

/-- Represents a math contest scoring system -/
structure ScoringSystem where
  correct_points : ℝ
  incorrect_penalty : ℝ

/-- Represents a contestant's performance in the math contest -/
structure ContestPerformance where
  total_questions : ℕ
  correct_answers : ℕ
  incorrect_answers : ℕ
  unanswered_questions : ℕ

/-- Calculates the total score for a contestant given their performance and the scoring system -/
def calculate_score (performance : ContestPerformance) (system : ScoringSystem) : ℝ :=
  (performance.correct_answers : ℝ) * system.correct_points -
  (performance.incorrect_answers : ℝ) * system.incorrect_penalty

/-- Theorem stating that Maria's score in the contest is 12.5 -/
theorem maria_score :
  let system : ScoringSystem := { correct_points := 1, incorrect_penalty := 0.25 }
  let performance : ContestPerformance := {
    total_questions := 30,
    correct_answers := 15,
    incorrect_answers := 10,
    unanswered_questions := 5
  }
  calculate_score performance system = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_maria_score_l2783_278321


namespace NUMINAMATH_CALUDE_maximum_marks_calculation_l2783_278340

theorem maximum_marks_calculation (percentage : ℝ) (received_marks : ℝ) (max_marks : ℝ) : 
  percentage = 80 → received_marks = 240 → percentage / 100 * max_marks = received_marks → max_marks = 300 := by
  sorry

end NUMINAMATH_CALUDE_maximum_marks_calculation_l2783_278340


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l2783_278332

theorem trigonometric_simplification :
  (Real.sin (7 * π / 180) + Real.cos (15 * π / 180) * Real.sin (8 * π / 180)) /
  (Real.cos (7 * π / 180) - Real.sin (15 * π / 180) * Real.sin (8 * π / 180)) = 2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l2783_278332


namespace NUMINAMATH_CALUDE_min_distance_sum_l2783_278314

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 12 = 1

/-- The left focus of the hyperbola -/
def F : ℝ × ℝ := sorry

/-- Point A -/
def A : ℝ × ℝ := (1, 4)

/-- Distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- A point is on the right branch of the hyperbola -/
def on_right_branch (p : ℝ × ℝ) : Prop :=
  hyperbola p.1 p.2 ∧ p.1 > 0

theorem min_distance_sum :
  ∀ P : ℝ × ℝ, on_right_branch P →
    distance P F + distance P A ≥ 9 ∧
    ∃ Q : ℝ × ℝ, on_right_branch Q ∧ distance Q F + distance Q A = 9 :=
sorry

end NUMINAMATH_CALUDE_min_distance_sum_l2783_278314


namespace NUMINAMATH_CALUDE_flagpole_height_correct_l2783_278380

/-- The height of the flagpole in feet -/
def flagpole_height : ℝ := 48

/-- The length of the flagpole's shadow in feet -/
def flagpole_shadow : ℝ := 72

/-- The height of the reference pole in feet -/
def reference_pole_height : ℝ := 18

/-- The length of the reference pole's shadow in feet -/
def reference_pole_shadow : ℝ := 27

/-- Theorem stating that the flagpole height is correct given the shadow lengths -/
theorem flagpole_height_correct :
  flagpole_height * reference_pole_shadow = reference_pole_height * flagpole_shadow :=
by sorry

end NUMINAMATH_CALUDE_flagpole_height_correct_l2783_278380


namespace NUMINAMATH_CALUDE_sum_of_squares_divisible_by_three_l2783_278386

theorem sum_of_squares_divisible_by_three (a b : ℤ) : 
  (3 ∣ a^2 + b^2) → (3 ∣ a) ∧ (3 ∣ b) := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_divisible_by_three_l2783_278386


namespace NUMINAMATH_CALUDE_ellipse_and_intersection_properties_l2783_278306

/-- Definition of the ellipse C -/
def ellipse_C (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

/-- Definition of the intersection line -/
def intersection_line (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x - 2

/-- Theorem stating the properties of the ellipse and the range of k -/
theorem ellipse_and_intersection_properties :
  ∀ (k : ℝ),
  (∃ (x y : ℝ), ellipse_C x y ∧ x = 1 ∧ y = 3/2) →
  (∃ (x₁ y₁ x₂ y₂ : ℝ),
    ellipse_C x₁ y₁ ∧ ellipse_C x₂ y₂ ∧
    intersection_line k x₁ y₁ ∧ intersection_line k x₂ y₂ ∧
    x₁ ≠ x₂) →
  (∀ (x₁ y₁ x₂ y₂ : ℝ),
    ellipse_C x₁ y₁ ∧ ellipse_C x₂ y₂ ∧
    intersection_line k x₁ y₁ ∧ intersection_line k x₂ y₂ ∧
    x₁ ≠ x₂ →
    (1/3 * x₁) * (2/3 * x₂) + (1/3 * y₁) * (2/3 * y₂) < 
    ((1/3 * x₁)^2 + (1/3 * y₁)^2 + (2/3 * x₂)^2 + (2/3 * y₂)^2) / 2) →
  (k > 1/2 ∧ k < 2 * Real.sqrt 3 / 3) ∨ (k < -1/2 ∧ k > -2 * Real.sqrt 3 / 3) :=
sorry

end NUMINAMATH_CALUDE_ellipse_and_intersection_properties_l2783_278306


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l2783_278364

theorem cone_lateral_surface_area (slant_height height : Real) 
  (h1 : slant_height = 15)
  (h2 : height = 9) :
  let radius := Real.sqrt (slant_height^2 - height^2)
  π * radius * slant_height = 180 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l2783_278364


namespace NUMINAMATH_CALUDE_ned_bomb_diffusal_l2783_278366

/-- Represents the problem of Ned racing to deactivate a time bomb -/
def BombDefusalProblem (total_flights : ℕ) (time_per_flight : ℕ) (bomb_timer : ℕ) (time_spent : ℕ) : Prop :=
  let flights_gone := time_spent / time_per_flight
  let flights_left := total_flights - flights_gone
  let time_left := bomb_timer - (flights_left * time_per_flight)
  time_left = 17

/-- Theorem stating that Ned will have 17 seconds to diffuse the bomb -/
theorem ned_bomb_diffusal :
  BombDefusalProblem 20 11 72 165 :=
sorry

end NUMINAMATH_CALUDE_ned_bomb_diffusal_l2783_278366


namespace NUMINAMATH_CALUDE_min_PQ_ratio_approaches_infinity_l2783_278371

-- Define the space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [CompleteSpace V]

-- Define points
variable (X Y M P Q : V)

-- Define conditions
variable (h1 : M = (X + Y) / 2)
variable (h2 : ∃ (t k : ℝ) (d : V), P = Y + t • d ∧ Q = Y - k • d ∧ t > 0 ∧ k > 0)
variable (h3 : ‖X - Q‖ = 2 * ‖M - P‖)
variable (h4 : ‖X - Y‖ / 2 < ‖M - P‖ ∧ ‖M - P‖ < 3 * ‖X - Y‖ / 2)

-- Theorem statement
theorem min_PQ_ratio_approaches_infinity :
  ∀ ε > 0, ∃ δ > 0, ∀ P' Q' : V,
    ‖P' - Q'‖ < ‖P - Q‖ + δ →
    ‖P' - Y‖ / ‖Q' - Y‖ > 1 / ε :=
sorry

end NUMINAMATH_CALUDE_min_PQ_ratio_approaches_infinity_l2783_278371


namespace NUMINAMATH_CALUDE_prob_odd_divisor_15_factorial_l2783_278357

/-- The factorial function -/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

/-- The number of divisors of n -/
def numDivisors (n : ℕ) : ℕ := sorry

/-- The number of odd divisors of n -/
def numOddDivisors (n : ℕ) : ℕ := sorry

/-- The probability of a randomly chosen divisor of n being odd -/
def probOddDivisor (n : ℕ) : ℚ :=
  (numOddDivisors n : ℚ) / (numDivisors n : ℚ)

theorem prob_odd_divisor_15_factorial :
  probOddDivisor (factorial 15) = 1 / 12 := by sorry

end NUMINAMATH_CALUDE_prob_odd_divisor_15_factorial_l2783_278357


namespace NUMINAMATH_CALUDE_smallest_positive_period_dependence_l2783_278311

noncomputable def f (a b x : ℝ) : ℝ := a * (Real.cos x)^2 + b * Real.sin x + Real.tan x

theorem smallest_positive_period_dependence (a b : ℝ) :
  ∃ (p : ℝ), p > 0 ∧ 
  (∀ (x : ℝ), f a b (x + p) = f a b x) ∧
  (∀ (q : ℝ), 0 < q ∧ q < p → ∃ (x : ℝ), f a b (x + q) ≠ f a b x) ∧
  (∀ (a' : ℝ), ∃ (p' : ℝ), p' > 0 ∧ 
    (∀ (x : ℝ), f a' b (x + p') = f a' b x) ∧
    (∀ (q : ℝ), 0 < q ∧ q < p' → ∃ (x : ℝ), f a' b (x + q) ≠ f a' b x) ∧
    p' = p) ∧
  (∃ (b' : ℝ), b' ≠ b → 
    ∀ (p' : ℝ), (∀ (x : ℝ), f a b' (x + p') = f a b' x) →
    (∀ (q : ℝ), 0 < q ∧ q < p' → ∃ (x : ℝ), f a b' (x + q) ≠ f a b' x) →
    p' ≠ p) :=
by sorry


end NUMINAMATH_CALUDE_smallest_positive_period_dependence_l2783_278311


namespace NUMINAMATH_CALUDE_probability_green_then_blue_l2783_278319

def total_marbles : ℕ := 10
def blue_marbles : ℕ := 4
def green_marbles : ℕ := 6

theorem probability_green_then_blue :
  (green_marbles : ℚ) / total_marbles * (blue_marbles : ℚ) / (total_marbles - 1) = 4 / 15 := by
sorry

end NUMINAMATH_CALUDE_probability_green_then_blue_l2783_278319


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2783_278320

/-- Two vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (-2, 1)
  let b : ℝ × ℝ := (x, -2)
  parallel a b → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2783_278320


namespace NUMINAMATH_CALUDE_negation_false_l2783_278394

/-- A multi-digit number ends in 0 -/
def EndsInZero (n : ℕ) : Prop := n % 10 = 0 ∧ n ≥ 10

/-- A number is a multiple of 5 -/
def MultipleOfFive (n : ℕ) : Prop := ∃ k : ℕ, n = 5 * k

theorem negation_false : 
  ¬(∀ n : ℕ, EndsInZero n → MultipleOfFive n) → 
  (∃ n : ℕ, EndsInZero n ∧ ¬MultipleOfFive n) :=
by sorry

end NUMINAMATH_CALUDE_negation_false_l2783_278394


namespace NUMINAMATH_CALUDE_rational_closure_l2783_278310

theorem rational_closure (x y : ℚ) (h : y ≠ 0) :
  (∃ a b : ℤ, (x + y = a / b ∧ b ≠ 0)) ∧
  (∃ c d : ℤ, (x - y = c / d ∧ d ≠ 0)) ∧
  (∃ e f : ℤ, (x * y = e / f ∧ f ≠ 0)) ∧
  (∃ g h : ℤ, (x / y = g / h ∧ h ≠ 0)) :=
by sorry

end NUMINAMATH_CALUDE_rational_closure_l2783_278310


namespace NUMINAMATH_CALUDE_profit_sharing_ratio_l2783_278333

/-- Represents the business partnership between A and B -/
structure Partnership where
  a_initial_investment : ℕ
  b_investment : ℕ
  a_investment_duration : ℕ
  b_investment_duration : ℕ

/-- Calculates the effective capital contribution -/
def effective_capital (investment : ℕ) (duration : ℕ) : ℕ :=
  investment * duration

/-- Simplifies a ratio by dividing both numbers by their GCD -/
def simplify_ratio (a b : ℕ) : ℕ × ℕ :=
  let gcd := Nat.gcd a b
  (a / gcd, b / gcd)

/-- Theorem stating that the profit sharing ratio is 2:3 given the conditions -/
theorem profit_sharing_ratio (p : Partnership) 
  (h1 : p.a_initial_investment = 4500)
  (h2 : p.b_investment = 16200)
  (h3 : p.a_investment_duration = 12)
  (h4 : p.b_investment_duration = 5) :
  simplify_ratio 
    (effective_capital p.a_initial_investment p.a_investment_duration)
    (effective_capital p.b_investment p.b_investment_duration) = (2, 3) := by
  sorry


end NUMINAMATH_CALUDE_profit_sharing_ratio_l2783_278333


namespace NUMINAMATH_CALUDE_parallel_lines_condition_l2783_278313

/-- Two lines are parallel if their slopes are equal and they are not identical. -/
def are_parallel (m n : ℝ) : Prop :=
  (m = 1 ∧ n ≠ -1) ∨ (m = -1 ∧ n ≠ 1)

/-- The theorem states that two lines mx+y-n=0 and x+my+1=0 are parallel
    if and only if (m=1 and n≠-1) or (m=-1 and n≠1). -/
theorem parallel_lines_condition (m n : ℝ) :
  are_parallel m n ↔ ∀ x y : ℝ, (m * x + y - n = 0 ↔ x + m * y + 1 = 0) :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_condition_l2783_278313


namespace NUMINAMATH_CALUDE_perpendicular_intersects_side_l2783_278359

/-- A regular polygon with n sides inscribed in a circle -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  is_regular : sorry
  is_inscribed : sorry

/-- The opposite side of a vertex in a regular polygon -/
def opposite_side (p : RegularPolygon 101) (i : Fin 101) : Set (ℝ × ℝ) :=
  sorry

/-- The perpendicular from a vertex to the line containing the opposite side -/
def perpendicular (p : RegularPolygon 101) (i : Fin 101) : Set (ℝ × ℝ) :=
  sorry

/-- The intersection point of the perpendicular and the line containing the opposite side -/
def intersection_point (p : RegularPolygon 101) (i : Fin 101) : ℝ × ℝ :=
  sorry

/-- Theorem: In a regular 101-gon inscribed in a circle, there exists at least one vertex 
    such that the perpendicular from this vertex to the line containing the opposite side 
    intersects the opposite side itself, not its extension -/
theorem perpendicular_intersects_side (p : RegularPolygon 101) : 
  ∃ i : Fin 101, intersection_point p i ∈ opposite_side p i :=
sorry

end NUMINAMATH_CALUDE_perpendicular_intersects_side_l2783_278359


namespace NUMINAMATH_CALUDE_cos_seven_pi_sixths_l2783_278342

theorem cos_seven_pi_sixths : Real.cos (7 * Real.pi / 6) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_seven_pi_sixths_l2783_278342


namespace NUMINAMATH_CALUDE_tamika_always_wins_l2783_278312

theorem tamika_always_wins : ∀ a b : ℕ, 
  a ∈ ({11, 12, 13} : Set ℕ) → 
  b ∈ ({11, 12, 13} : Set ℕ) → 
  a ≠ b → 
  a * b > (2 + 3 + 4) := by
sorry

end NUMINAMATH_CALUDE_tamika_always_wins_l2783_278312


namespace NUMINAMATH_CALUDE_min_value_theorem_l2783_278308

theorem min_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + 2*y + 3*z = 6) : 
  (1/x + 4/y + 9/z) ≥ 98/3 ∧ 
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ 
    x₀ + 2*y₀ + 3*z₀ = 6 ∧ 1/x₀ + 4/y₀ + 9/z₀ = 98/3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2783_278308


namespace NUMINAMATH_CALUDE_proportion_problem_l2783_278383

theorem proportion_problem (x : ℝ) : 
  (x / 5 = 0.96 / 8) → x = 0.6 := by
sorry

end NUMINAMATH_CALUDE_proportion_problem_l2783_278383


namespace NUMINAMATH_CALUDE_sum_of_exponents_is_eight_l2783_278369

-- Define the expression
def expression (a b c : ℝ) : ℝ := (40 * a^6 * b^8 * c^14) ^ (1/3)

-- Define a function to calculate the sum of exponents outside the radical
def sum_of_exponents_outside_radical (a b c : ℝ) : ℕ :=
  let simplified := expression a b c
  -- This is a placeholder. In a real implementation, we would need to
  -- analyze the simplified expression to determine the exponents.
  8

-- The theorem to prove
theorem sum_of_exponents_is_eight :
  ∀ a b c : ℝ, sum_of_exponents_outside_radical a b c = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_exponents_is_eight_l2783_278369


namespace NUMINAMATH_CALUDE_equation_one_integral_root_l2783_278361

theorem equation_one_integral_root :
  ∃! x : ℤ, x - 9 / (x - 5 : ℚ) = 7 - 9 / (x - 5 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_equation_one_integral_root_l2783_278361


namespace NUMINAMATH_CALUDE_problem_statement_l2783_278302

noncomputable def f₁ (a x : ℝ) : ℝ := Real.exp (abs (x - 2*a + 1))
noncomputable def f₂ (a x : ℝ) : ℝ := Real.exp (abs (x - a) + 1)
noncomputable def f (a x : ℝ) : ℝ := f₁ a x + f₂ a x
noncomputable def g (a x : ℝ) : ℝ := (f₁ a x + f₂ a x) / 2 - abs (f₁ a x - f₂ a x) / 2

theorem problem_statement :
  (∀ x ∈ Set.Icc 2 3, f 2 x ≥ 2 * Real.exp 1) ∧
  (∃ x ∈ Set.Icc 2 3, f 2 x = 2 * Real.exp 1) ∧
  (∀ a, (∀ x ≥ a, f₂ a x ≥ f₁ a x) ↔ 0 ≤ a ∧ a ≤ 2) ∧
  (∀ x ∈ Set.Icc 1 6, 
    g a x ≥ 
      (if 1 ≤ a ∧ a ≤ 7/2 then 1
      else if -2 ≤ a ∧ a ≤ 0 then Real.exp (2 - a)
      else if a < -2 ∨ (0 < a ∧ a < 1) then Real.exp (3 - 2*a)
      else if 7/2 < a ∧ a ≤ 6 then Real.exp 1
      else Real.exp (a - 5))) ∧
  (∃ x ∈ Set.Icc 1 6, 
    g a x = 
      (if 1 ≤ a ∧ a ≤ 7/2 then 1
      else if -2 ≤ a ∧ a ≤ 0 then Real.exp (2 - a)
      else if a < -2 ∨ (0 < a ∧ a < 1) then Real.exp (3 - 2*a)
      else if 7/2 < a ∧ a ≤ 6 then Real.exp 1
      else Real.exp (a - 5))) := by sorry

end NUMINAMATH_CALUDE_problem_statement_l2783_278302


namespace NUMINAMATH_CALUDE_bagel_store_spending_l2783_278304

theorem bagel_store_spending (B D : ℝ) : 
  D = (9/10) * B →
  B = D + 15 →
  B + D = 285 :=
by sorry

end NUMINAMATH_CALUDE_bagel_store_spending_l2783_278304


namespace NUMINAMATH_CALUDE_bus_ride_cost_l2783_278379

/-- The cost of a bus ride from town P to town Q -/
def bus_cost : ℝ := 3.75

/-- The cost of a train ride from town P to town Q -/
def train_cost : ℝ := bus_cost + 2.35

/-- The theorem stating the cost of a bus ride from town P to town Q -/
theorem bus_ride_cost : bus_cost = 3.75 := by sorry

/-- The condition that a train ride costs $2.35 more than a bus ride -/
axiom train_cost_difference : train_cost = bus_cost + 2.35

/-- The condition that the combined cost of one train ride and one bus ride is $9.85 -/
axiom combined_cost : train_cost + bus_cost = 9.85

end NUMINAMATH_CALUDE_bus_ride_cost_l2783_278379


namespace NUMINAMATH_CALUDE_cranberry_juice_ounces_l2783_278389

/-- Given a can of cranberry juice that sells for 84 cents with a cost of 7 cents per ounce,
    prove that the can contains 12 ounces of juice. -/
theorem cranberry_juice_ounces (total_cost : ℕ) (cost_per_ounce : ℕ) (h1 : total_cost = 84) (h2 : cost_per_ounce = 7) :
  total_cost / cost_per_ounce = 12 := by
sorry

end NUMINAMATH_CALUDE_cranberry_juice_ounces_l2783_278389


namespace NUMINAMATH_CALUDE_expected_value_coin_flip_l2783_278347

/-- The expected value of a coin flip game -/
theorem expected_value_coin_flip (p_heads : ℚ) (p_tails : ℚ) 
  (win_heads : ℚ) (lose_tails : ℚ) : 
  p_heads = 1/3 → p_tails = 2/3 → win_heads = 3 → lose_tails = 2 →
  p_heads * win_heads - p_tails * lose_tails = -1/3 := by
  sorry

#check expected_value_coin_flip

end NUMINAMATH_CALUDE_expected_value_coin_flip_l2783_278347


namespace NUMINAMATH_CALUDE_optimal_pen_area_optimal_parallel_side_l2783_278355

/-- The length of the side parallel to the shed that maximizes the rectangular goat pen area -/
def optimal_parallel_side_length : ℝ := 50

/-- The total length of fence available -/
def total_fence_length : ℝ := 100

/-- The length of the shed -/
def shed_length : ℝ := 300

/-- The area of the pen as a function of the perpendicular side length -/
def pen_area (y : ℝ) : ℝ := y * (total_fence_length - 2 * y)

theorem optimal_pen_area :
  ∀ y : ℝ, 0 < y → y < total_fence_length / 2 →
  pen_area y ≤ pen_area (total_fence_length / 4) :=
sorry

theorem optimal_parallel_side :
  optimal_parallel_side_length = total_fence_length / 2 :=
sorry

end NUMINAMATH_CALUDE_optimal_pen_area_optimal_parallel_side_l2783_278355


namespace NUMINAMATH_CALUDE_basketball_league_games_l2783_278348

/-- The number of games played in a basketball league season -/
def total_games (n : ℕ) (games_per_pairing : ℕ) : ℕ :=
  n * (n - 1) * games_per_pairing / 2

/-- Theorem: In a league with 10 teams, where each team plays 4 games with each other team,
    the total number of games played is 180. -/
theorem basketball_league_games :
  total_games 10 4 = 180 := by
  sorry

end NUMINAMATH_CALUDE_basketball_league_games_l2783_278348


namespace NUMINAMATH_CALUDE_salary_problem_l2783_278387

theorem salary_problem (A B : ℝ) 
  (h1 : A + B = 2000)
  (h2 : 0.05 * A = 0.15 * B) :
  A = 1500 := by
sorry

end NUMINAMATH_CALUDE_salary_problem_l2783_278387


namespace NUMINAMATH_CALUDE_percentage_increase_l2783_278375

theorem percentage_increase (t : ℝ) (P : ℝ) : 
  t = 80 →
  (t + (P / 100) * t) - (t - (25 / 100) * t) = 30 →
  P = 12.5 := by
sorry

end NUMINAMATH_CALUDE_percentage_increase_l2783_278375


namespace NUMINAMATH_CALUDE_solution_set_equals_open_interval_l2783_278362

def solution_set : Set ℝ := {x | x^2 - 9*x + 14 < 0 ∧ 2*x + 3 > 0}

theorem solution_set_equals_open_interval :
  solution_set = Set.Ioo 2 7 := by sorry

end NUMINAMATH_CALUDE_solution_set_equals_open_interval_l2783_278362


namespace NUMINAMATH_CALUDE_book_loss_percentage_l2783_278395

/-- Given that the cost price of 5 books equals the selling price of 20 books,
    prove that the loss percentage is 75%. -/
theorem book_loss_percentage : ∀ (C S : ℝ), 
  C > 0 → S > 0 →  -- Ensure positive prices
  5 * C = 20 * S →  -- Given condition
  (C - S) / C * 100 = 75 := by  -- Loss percentage formula
  sorry

end NUMINAMATH_CALUDE_book_loss_percentage_l2783_278395


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2783_278376

theorem inequality_solution_set (x : ℝ) : 
  5 * x^2 + 7 * x > 3 ↔ x < -1 ∨ x > 3/5 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2783_278376


namespace NUMINAMATH_CALUDE_max_value_constraint_max_value_attained_unique_max_value_l2783_278396

theorem max_value_constraint (x y z : ℝ) :
  x^2 + 2*x + (1/5)*y^2 + 7*z^2 = 6 →
  7*x + 10*y + z ≤ 55 :=
by sorry

theorem max_value_attained :
  ∃ x y z : ℝ, x^2 + 2*x + (1/5)*y^2 + 7*z^2 = 6 ∧ 7*x + 10*y + z = 55 :=
by sorry

theorem unique_max_value (x y z : ℝ) :
  x^2 + 2*x + (1/5)*y^2 + 7*z^2 = 6 ∧ 7*x + 10*y + z = 55 →
  x = -13/62 ∧ y = 175/31 ∧ z = 1/62 :=
by sorry

end NUMINAMATH_CALUDE_max_value_constraint_max_value_attained_unique_max_value_l2783_278396


namespace NUMINAMATH_CALUDE_tangent_line_intersection_extreme_values_l2783_278349

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x - 3

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x - 9

-- Theorem for the tangent line intersection points
theorem tangent_line_intersection (x₀ : ℝ) (b : ℝ) :
  f' x₀ = -9 ∧ f x₀ = -9 * x₀ + b → b = -3 ∨ b = -7 :=
sorry

-- Theorem for the extreme values of f(x)
theorem extreme_values :
  (∃ x : ℝ, f x = 2 ∧ ∀ y : ℝ, f y ≤ f x) ∧
  (∃ x : ℝ, f x = -30 ∧ ∀ y : ℝ, f y ≥ f x) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_intersection_extreme_values_l2783_278349


namespace NUMINAMATH_CALUDE_sphere_radius_ratio_l2783_278351

theorem sphere_radius_ratio : 
  ∀ (r R : ℝ), 
    (4 / 3 * π * r^3 = 36 * π) → 
    (4 / 3 * π * R^3 = 450 * π) → 
    r / R = 1 / Real.rpow 12.5 (1/3) := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_ratio_l2783_278351


namespace NUMINAMATH_CALUDE_triangle_angle_solution_l2783_278372

theorem triangle_angle_solution :
  ∀ x : ℝ,
  (40 : ℝ) + 4 * x + 3 * x = 180 →
  x = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_solution_l2783_278372


namespace NUMINAMATH_CALUDE_total_fans_count_l2783_278305

/-- Represents the number of fans for each team -/
structure FanCounts where
  yankees : ℕ
  mets : ℕ
  red_sox : ℕ

/-- Calculates the total number of fans -/
def total_fans (fans : FanCounts) : ℕ :=
  fans.yankees + fans.mets + fans.red_sox

/-- Theorem: Given the ratios and number of Mets fans, prove the total number of fans is 360 -/
theorem total_fans_count (fans : FanCounts) 
  (yankees_mets_ratio : fans.yankees = 3 * fans.mets / 2)
  (mets_redsox_ratio : fans.red_sox = 5 * fans.mets / 4)
  (mets_count : fans.mets = 96) :
  total_fans fans = 360 := by
  sorry

#eval total_fans { yankees := 144, mets := 96, red_sox := 120 }

end NUMINAMATH_CALUDE_total_fans_count_l2783_278305


namespace NUMINAMATH_CALUDE_flower_pots_total_cost_l2783_278303

def flower_pots_cost (n : ℕ) (price_difference : ℚ) (largest_pot_price : ℚ) : ℚ :=
  let smallest_pot_price := largest_pot_price - (n - 1 : ℚ) * price_difference
  (n : ℚ) * smallest_pot_price + ((n - 1) * n / 2 : ℚ) * price_difference

theorem flower_pots_total_cost :
  flower_pots_cost 6 (3/10) (85/40) = 33/4 :=
sorry

end NUMINAMATH_CALUDE_flower_pots_total_cost_l2783_278303


namespace NUMINAMATH_CALUDE_museum_trip_l2783_278377

theorem museum_trip (first_bus : ℕ) (second_bus : ℕ) (third_bus : ℕ) (fourth_bus : ℕ) :
  first_bus = 12 →
  second_bus = 2 * first_bus →
  third_bus = second_bus - 6 →
  first_bus + second_bus + third_bus + fourth_bus = 75 →
  fourth_bus - first_bus = 9 := by
sorry

end NUMINAMATH_CALUDE_museum_trip_l2783_278377


namespace NUMINAMATH_CALUDE_julia_video_games_fraction_l2783_278338

/-- Given the number of video games owned by Theresa, Julia, and Tory,
    prove that Julia has 1/3 as many video games as Tory. -/
theorem julia_video_games_fraction (theresa julia tory : ℕ) : 
  theresa = 3 * julia + 5 →
  tory = 6 →
  theresa = 11 →
  julia * 3 = tory := by
  sorry

end NUMINAMATH_CALUDE_julia_video_games_fraction_l2783_278338


namespace NUMINAMATH_CALUDE_fraction_equality_l2783_278328

theorem fraction_equality (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : (4 * a + b) / (a - 4 * b) = 3) : 
  (a + 4 * b) / (4 * a - b) = 9 / 53 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l2783_278328


namespace NUMINAMATH_CALUDE_cost_per_person_l2783_278326

def total_cost : ℚ := 12100
def num_people : ℕ := 11

theorem cost_per_person :
  total_cost / num_people = 1100 :=
sorry

end NUMINAMATH_CALUDE_cost_per_person_l2783_278326


namespace NUMINAMATH_CALUDE_white_balls_count_l2783_278393

theorem white_balls_count (total : ℕ) (green yellow red purple : ℕ) (prob_not_red_purple : ℚ) :
  total = 60 →
  green = 10 →
  yellow = 7 →
  red = 15 →
  purple = 6 →
  prob_not_red_purple = 13/20 →
  ∃ white : ℕ, white = 22 ∧ total = white + green + yellow + red + purple :=
by sorry

end NUMINAMATH_CALUDE_white_balls_count_l2783_278393


namespace NUMINAMATH_CALUDE_complex_order_multiplication_property_l2783_278385

-- Define the order relation on complex numbers
def complex_order (z1 z2 : ℂ) : Prop :=
  z1.re > z2.re ∨ (z1.re = z2.re ∧ z1.im > z2.im)

-- Define the statement to be proven false
theorem complex_order_multiplication_property (z z1 z2 : ℂ) :
  ¬(complex_order z 0 → complex_order z1 z2 → complex_order (z * z1) (z * z2)) :=
sorry

end NUMINAMATH_CALUDE_complex_order_multiplication_property_l2783_278385


namespace NUMINAMATH_CALUDE_car_speed_problem_l2783_278301

theorem car_speed_problem (speed_second_hour : ℝ) (average_speed : ℝ) :
  speed_second_hour = 80 →
  average_speed = 85 →
  ∃ (speed_first_hour : ℝ),
    speed_first_hour = 90 ∧
    average_speed = (speed_first_hour + speed_second_hour) / 2 :=
by sorry

end NUMINAMATH_CALUDE_car_speed_problem_l2783_278301


namespace NUMINAMATH_CALUDE_smallest_positive_root_floor_l2783_278315

noncomputable def g (x : ℝ) : ℝ := 3 * Real.sin x - Real.cos x + 2 * Real.tan x

def is_smallest_positive_root (s : ℝ) : Prop :=
  s > 0 ∧ g s = 0 ∧ ∀ x, 0 < x ∧ x < s → g x ≠ 0

theorem smallest_positive_root_floor :
  ∃ s, is_smallest_positive_root s ∧ ⌊s⌋ = 3 := by sorry

end NUMINAMATH_CALUDE_smallest_positive_root_floor_l2783_278315


namespace NUMINAMATH_CALUDE_science_club_officer_selection_l2783_278354

def science_club_officers (n : ℕ) (k : ℕ) (special_members : ℕ) : ℕ :=
  (n - special_members).choose k + special_members * (special_members - 1) * (n - special_members)

theorem science_club_officer_selection :
  science_club_officers 25 3 2 = 10764 :=
by sorry

end NUMINAMATH_CALUDE_science_club_officer_selection_l2783_278354


namespace NUMINAMATH_CALUDE_andrew_donut_problem_l2783_278343

/-- The number of donuts Andrew ate on Monday -/
def monday_donuts : ℕ := 14

/-- The number of donuts Andrew ate on Tuesday -/
def tuesday_donuts : ℕ := monday_donuts / 2

/-- The total number of donuts Andrew ate in three days -/
def total_donuts : ℕ := 49

/-- The multiplier for the number of donuts Andrew ate on Wednesday compared to Monday -/
def wednesday_multiplier : ℚ := 2

theorem andrew_donut_problem :
  monday_donuts + tuesday_donuts + (wednesday_multiplier * monday_donuts) = total_donuts :=
sorry

end NUMINAMATH_CALUDE_andrew_donut_problem_l2783_278343


namespace NUMINAMATH_CALUDE_little_d_can_win_l2783_278381

/-- Represents a point in the 3D lattice grid -/
structure LatticePoint where
  x : Int
  y : Int
  z : Int

/-- Represents a plane perpendicular to a coordinate axis -/
inductive Plane
  | X (y z : Int)
  | Y (x z : Int)
  | Z (x y : Int)

/-- Represents the state of the game -/
structure GameState where
  markedPoints : Set LatticePoint
  munchedPlanes : Set Plane

/-- Represents a move by Little D -/
def LittleDMove := LatticePoint

/-- Represents a move by Big Z -/
def BigZMove := Plane

/-- A strategy for Little D is a function that takes the current game state
    and returns the next move -/
def LittleDStrategy := GameState → LittleDMove

/-- Check if n consecutive points are marked on a line parallel to a coordinate axis -/
def hasConsecutiveMarkedPoints (state : GameState) (n : Nat) : Prop :=
  ∃ (start : LatticePoint) (axis : Fin 3),
    ∀ i : Fin n,
      let point : LatticePoint :=
        match axis with
        | 0 => ⟨start.x + i.val, start.y, start.z⟩
        | 1 => ⟨start.x, start.y + i.val, start.z⟩
        | 2 => ⟨start.x, start.y, start.z + i.val⟩
      point ∈ state.markedPoints

/-- The main theorem: Little D can win for any n -/
theorem little_d_can_win (n : Nat) :
  ∃ (strategy : LittleDStrategy),
    ∀ (bigZMoves : Nat → BigZMove),
      ∃ (finalState : GameState),
        hasConsecutiveMarkedPoints finalState n :=
  sorry

end NUMINAMATH_CALUDE_little_d_can_win_l2783_278381


namespace NUMINAMATH_CALUDE_lee_weight_l2783_278329

/-- Given Anna's and Lee's weights satisfying certain conditions, prove Lee's weight is 144 pounds. -/
theorem lee_weight (anna lee : ℝ) 
  (h1 : anna + lee = 240)
  (h2 : lee - anna = lee / 3) : 
  lee = 144 := by sorry

end NUMINAMATH_CALUDE_lee_weight_l2783_278329


namespace NUMINAMATH_CALUDE_max_product_partition_l2783_278392

/-- Given positive integers k and n with k ≥ n, where k = nq + r (0 ≤ r < n),
    F(k) is the maximum product of n positive integers that sum to k. -/
def F (k n : ℕ+) (h : k ≥ n) : ℕ := by sorry

/-- The quotient when k is divided by n -/
def q (k n : ℕ+) : ℕ := k / n

/-- The remainder when k is divided by n -/
def r (k n : ℕ+) : ℕ := k % n

theorem max_product_partition (k n : ℕ+) (h : k ≥ n) :
  F k n h = (q k n) ^ (n - r k n) * ((q k n) + 1) ^ (r k n) := by sorry

end NUMINAMATH_CALUDE_max_product_partition_l2783_278392


namespace NUMINAMATH_CALUDE_system_solution_l2783_278339

theorem system_solution (a b : ℤ) : 
  (∃ x y : ℤ, a * x + 5 * y = 15 ∧ 4 * x - b * y = -2) →
  (4 * (-3) - b * (-1) = -2) →
  (a * 5 + 5 * 4 = 15) →
  a^2023 + (-1/10 * b : ℚ)^2023 = -2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2783_278339


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2783_278322

def A : Set ℤ := {0, 2}
def B : Set ℤ := {-2, -1, 0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {0, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2783_278322


namespace NUMINAMATH_CALUDE_mean_median_difference_l2783_278344

/-- Represents the score distribution of students in a test --/
structure ScoreDistribution where
  score65 : Float
  score75 : Float
  score88 : Float
  score92 : Float
  score100 : Float
  total_percentage : Float
  h_total : total_percentage = score65 + score75 + score88 + score92 + score100

/-- Calculates the median score given a ScoreDistribution --/
def median (sd : ScoreDistribution) : Float :=
  sorry

/-- Calculates the mean score given a ScoreDistribution --/
def mean (sd : ScoreDistribution) : Float :=
  sorry

/-- The main theorem stating the difference between mean and median --/
theorem mean_median_difference (sd : ScoreDistribution) 
  (h1 : sd.score65 = 0.15)
  (h2 : sd.score75 = 0.20)
  (h3 : sd.score88 = 0.25)
  (h4 : sd.score92 = 0.10)
  (h5 : sd.score100 = 0.30)
  (h6 : sd.total_percentage = 1.0) :
  mean sd - median sd = -2 :=
sorry

end NUMINAMATH_CALUDE_mean_median_difference_l2783_278344


namespace NUMINAMATH_CALUDE_power_not_all_ones_l2783_278382

theorem power_not_all_ones (a n : ℕ) : a > 1 → n > 1 → ¬∃ s : ℕ, a^n = 2^s - 1 := by
  sorry

end NUMINAMATH_CALUDE_power_not_all_ones_l2783_278382


namespace NUMINAMATH_CALUDE_volleyball_team_selection_l2783_278374

def total_players : ℕ := 16
def triplets : ℕ := 3
def captain : ℕ := 1
def starters : ℕ := 6

def remaining_players : ℕ := total_players - triplets - captain
def players_to_choose : ℕ := starters - triplets - captain

theorem volleyball_team_selection :
  Nat.choose remaining_players players_to_choose = 66 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_team_selection_l2783_278374


namespace NUMINAMATH_CALUDE_min_n_plus_d_l2783_278363

/-- An arithmetic sequence with positive integer terms -/
structure ArithmeticSequence where
  n : ℕ+  -- number of terms
  d : ℕ+  -- common difference
  first_term : ℕ+ := 1  -- first term
  last_term : ℕ+ := 51  -- last term

/-- The property that the sequence follows the arithmetic sequence formula -/
def is_valid (seq : ArithmeticSequence) : Prop :=
  seq.first_term + (seq.n - 1) * seq.d = seq.last_term

/-- The theorem stating the minimum value of n + d -/
theorem min_n_plus_d (seq : ArithmeticSequence) (h : is_valid seq) : 
  (∀ seq' : ArithmeticSequence, is_valid seq' → seq.n + seq.d ≤ seq'.n + seq'.d) → 
  seq.n + seq.d = 16 := by
  sorry

end NUMINAMATH_CALUDE_min_n_plus_d_l2783_278363


namespace NUMINAMATH_CALUDE_peaches_for_juice_l2783_278356

def total_peaches : ℝ := 7.5

def drying_percentage : ℝ := 0.3

def juice_percentage_of_remainder : ℝ := 0.4

theorem peaches_for_juice :
  let remaining_after_drying := total_peaches * (1 - drying_percentage)
  let juice_amount := remaining_after_drying * juice_percentage_of_remainder
  juice_amount = 2.1 := by sorry

end NUMINAMATH_CALUDE_peaches_for_juice_l2783_278356


namespace NUMINAMATH_CALUDE_years_until_double_age_l2783_278325

/-- Represents the age difference problem between a father and son -/
structure AgeDifference where
  son_age : ℕ
  father_age : ℕ
  years_until_double : ℕ

/-- The age difference scenario satisfies the given conditions -/
def valid_age_difference (ad : AgeDifference) : Prop :=
  ad.son_age = 10 ∧
  ad.father_age = 40 ∧
  ad.father_age = 4 * ad.son_age ∧
  ad.father_age + ad.years_until_double = 2 * (ad.son_age + ad.years_until_double)

/-- Theorem stating that the number of years until the father is twice as old as the son is 20 -/
theorem years_until_double_age : ∀ ad : AgeDifference, valid_age_difference ad → ad.years_until_double = 20 := by
  sorry

end NUMINAMATH_CALUDE_years_until_double_age_l2783_278325


namespace NUMINAMATH_CALUDE_star_1993_1932_l2783_278300

-- Define the * operation
def star (x y : ℤ) : ℤ := x - y

-- State the theorem
theorem star_1993_1932 : star 1993 1932 = 61 :=
  by
  -- Define the properties of the star operation
  have h1 : ∀ x : ℤ, star x x = 0 := by sorry
  have h2 : ∀ x y z : ℤ, star x (star y z) = star x y + z := by sorry
  
  -- Prove the theorem
  sorry

end NUMINAMATH_CALUDE_star_1993_1932_l2783_278300


namespace NUMINAMATH_CALUDE_crate_weight_l2783_278327

/-- Given an empty truck weighing 9600 kg and a total weight of 38000 kg when loaded with 40 identical crates, 
    prove that each crate weighs 710 kg. -/
theorem crate_weight (empty_truck_weight : ℕ) (loaded_truck_weight : ℕ) (num_crates : ℕ) :
  empty_truck_weight = 9600 →
  loaded_truck_weight = 38000 →
  num_crates = 40 →
  (loaded_truck_weight - empty_truck_weight) / num_crates = 710 :=
by sorry

end NUMINAMATH_CALUDE_crate_weight_l2783_278327


namespace NUMINAMATH_CALUDE_unique_solution_is_negation_f_is_bijective_l2783_278365

/-- A function satisfying the given functional equation. -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, (y + 1) * f x + f (x * f y + f (x + y)) = y

/-- The main theorem stating that f(x) = -x is the unique solution. -/
theorem unique_solution_is_negation (f : ℝ → ℝ) 
    (h : SatisfiesFunctionalEquation f) : 
    f = fun x ↦ -x := by
  sorry

/-- f is bijective -/
theorem f_is_bijective (f : ℝ → ℝ) 
    (h : SatisfiesFunctionalEquation f) : 
    Function.Bijective f := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_is_negation_f_is_bijective_l2783_278365


namespace NUMINAMATH_CALUDE_bob_salary_last_year_l2783_278346

/-- Mario's salary this year -/
def mario_salary_this_year : ℝ := 4000

/-- Mario's salary increase percentage -/
def mario_increase_percentage : ℝ := 0.40

/-- Bob's salary last year as a multiple of Mario's salary this year -/
def bob_salary_multiple : ℝ := 3

theorem bob_salary_last_year :
  let mario_salary_last_year := mario_salary_this_year / (1 + mario_increase_percentage)
  let bob_salary_last_year := bob_salary_multiple * mario_salary_this_year
  bob_salary_last_year = 12000 := by sorry

end NUMINAMATH_CALUDE_bob_salary_last_year_l2783_278346


namespace NUMINAMATH_CALUDE_unique_solution_exists_l2783_278360

theorem unique_solution_exists : ∃! x : ℕ, 
  x < 5311735 ∧
  x % 5 = 0 ∧
  x % 715 = 10 ∧
  x % 247 = 140 ∧
  x % 391 = 245 ∧
  x % 187 = 109 ∧
  x = 10020 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_exists_l2783_278360


namespace NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l2783_278358

-- Problem 1
theorem factorization_problem_1 (p q : ℝ) :
  6 * p^3 * q - 10 * p^2 = 2 * p^2 * (3 * p * q - 5) := by sorry

-- Problem 2
theorem factorization_problem_2 (a : ℝ) :
  a^4 - 8 * a^2 + 16 = (a + 2)^2 * (a - 2)^2 := by sorry

end NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l2783_278358


namespace NUMINAMATH_CALUDE_chord_length_concentric_circles_l2783_278399

/-- Given two concentric circles with radii R and r, where R > r, 
    and the area of the ring between them is 16π square inches,
    the length of a chord of the larger circle that is tangent to the smaller circle is 8 inches. -/
theorem chord_length_concentric_circles 
  (R r : ℝ) 
  (h1 : R > r) 
  (h2 : π * R^2 - π * r^2 = 16 * π) : 
  ∃ (c : ℝ), c = 8 ∧ c^2 = 4 * (R^2 - r^2) :=
sorry

end NUMINAMATH_CALUDE_chord_length_concentric_circles_l2783_278399


namespace NUMINAMATH_CALUDE_jia_opened_physical_store_l2783_278309

-- Define the possible shop types
inductive ShopType
| Taobao
| WeChat
| Physical

-- Define the graduates
inductive Graduate
| Jia
| Yi
| Bing

-- Define a function that assigns a shop type to each graduate
def shop : Graduate → ShopType := sorry

-- Define the statements made by each graduate
def jia_statement : Prop :=
  shop Graduate.Jia = ShopType.Taobao ∧ shop Graduate.Yi = ShopType.WeChat

def yi_statement : Prop :=
  shop Graduate.Jia = ShopType.WeChat ∧ shop Graduate.Bing = ShopType.Taobao

def bing_statement : Prop :=
  shop Graduate.Jia = ShopType.Physical ∧ shop Graduate.Yi = ShopType.Taobao

-- Define a function to count the number of true parts in a statement
def true_count (statement : Prop) : Nat := sorry

-- Theorem: Given the conditions, Jia must have opened a physical store
theorem jia_opened_physical_store :
  (true_count jia_statement = 1) →
  (true_count yi_statement = 1) →
  (true_count bing_statement = 1) →
  (shop Graduate.Jia = ShopType.Physical) :=
by sorry

end NUMINAMATH_CALUDE_jia_opened_physical_store_l2783_278309


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2783_278307

def i : ℂ := Complex.I

theorem complex_equation_solution :
  ∀ z : ℂ, (2 - i) * z = i^2021 → z = -1/5 + 2/5*i :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2783_278307


namespace NUMINAMATH_CALUDE_shopkeeper_profit_percentage_l2783_278353

theorem shopkeeper_profit_percentage 
  (theft_percentage : ℝ) 
  (loss_percentage : ℝ) 
  (profit_percentage : ℝ) : 
  theft_percentage = 60 → 
  loss_percentage = 56 → 
  (1 - theft_percentage / 100) * (1 + profit_percentage / 100) = 1 - loss_percentage / 100 → 
  profit_percentage = 10 := by sorry

end NUMINAMATH_CALUDE_shopkeeper_profit_percentage_l2783_278353


namespace NUMINAMATH_CALUDE_february_discount_correct_l2783_278378

/-- Represents the discount percentage applied in February -/
def discount_percentage : ℝ := 7

/-- Represents the initial markup percentage -/
def initial_markup : ℝ := 20

/-- Represents the New Year markup percentage -/
def new_year_markup : ℝ := 25

/-- Represents the profit percentage in February -/
def february_profit : ℝ := 39.5

/-- Theorem stating that the discount percentage in February is correct given the markups and profit -/
theorem february_discount_correct :
  let cost := 100 -- Assuming a base cost of 100 for simplicity
  let initial_price := cost * (1 + initial_markup / 100)
  let new_year_price := initial_price * (1 + new_year_markup / 100)
  let final_price := new_year_price * (1 - discount_percentage / 100)
  final_price - cost = february_profit * cost / 100 :=
sorry


end NUMINAMATH_CALUDE_february_discount_correct_l2783_278378


namespace NUMINAMATH_CALUDE_irrational_expression_l2783_278345

theorem irrational_expression (x : ℝ) : 
  Irrational ((x - 3 * Real.sqrt (x^2 + 4)) / 2) := by sorry

end NUMINAMATH_CALUDE_irrational_expression_l2783_278345


namespace NUMINAMATH_CALUDE_drug_efficacy_rate_l2783_278336

/-- Calculates the efficacy rate of a drug based on a survey --/
def efficacyRate (totalSamples : ℕ) (positiveResponses : ℕ) : ℚ :=
  (positiveResponses : ℚ) / (totalSamples : ℚ)

theorem drug_efficacy_rate :
  let totalSamples : ℕ := 20
  let positiveResponses : ℕ := 16
  efficacyRate totalSamples positiveResponses = 4/5 := by
sorry

end NUMINAMATH_CALUDE_drug_efficacy_rate_l2783_278336


namespace NUMINAMATH_CALUDE_pencil_distribution_ways_l2783_278317

/-- The number of ways to distribute pencils among friends -/
def distribute_pencils (total_pencils : ℕ) (num_friends : ℕ) (min_pencils : ℕ) : ℕ :=
  Nat.choose (total_pencils - num_friends * min_pencils + num_friends - 1) (num_friends - 1)

/-- Theorem: There are 6 ways to distribute 8 pencils among 3 friends with at least 2 pencils each -/
theorem pencil_distribution_ways : distribute_pencils 8 3 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_pencil_distribution_ways_l2783_278317


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2783_278368

theorem trigonometric_identity (α : ℝ) (m : ℝ) (h : Real.sin α - Real.cos α = m) :
  (Real.sin (4 * α) + Real.sin (10 * α) - Real.sin (6 * α)) /
  (Real.cos (2 * α) + 1 - 2 * Real.sin (4 * α) ^ 2) = 2 * (1 - m ^ 2) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2783_278368


namespace NUMINAMATH_CALUDE_total_apples_l2783_278350

theorem total_apples (marin_apples : ℕ) (david_apples : ℕ) (amanda_apples : ℕ) : 
  marin_apples = 6 →
  david_apples = 2 * marin_apples →
  amanda_apples = david_apples + 5 →
  marin_apples + david_apples + amanda_apples = 35 :=
by sorry

end NUMINAMATH_CALUDE_total_apples_l2783_278350


namespace NUMINAMATH_CALUDE_max_valid_sequence_length_l2783_278373

/-- A sequence of integers satisfying the given conditions -/
def ValidSequence (a : ℕ → ℤ) (n : ℕ) : Prop :=
  (∀ i : ℕ, i + 2 < n → a i + a (i + 1) + a (i + 2) > 0) ∧
  (∀ i : ℕ, i + 4 < n → a i + a (i + 1) + a (i + 2) + a (i + 3) + a (i + 4) < 0)

/-- The maximum length of a valid sequence is 6 -/
theorem max_valid_sequence_length :
  (∃ (a : ℕ → ℤ), ValidSequence a 6) ∧
  (∀ n : ℕ, n > 6 → ¬∃ (a : ℕ → ℤ), ValidSequence a n) :=
sorry

end NUMINAMATH_CALUDE_max_valid_sequence_length_l2783_278373


namespace NUMINAMATH_CALUDE_mass_of_man_on_boat_l2783_278318

/-- The mass of a man who causes a boat to sink by a certain depth -/
def mass_of_man (length breadth depth_sunk : ℝ) (water_density : ℝ) : ℝ :=
  length * breadth * depth_sunk * water_density

/-- Theorem stating that the mass of the man is 60 kg -/
theorem mass_of_man_on_boat :
  mass_of_man 3 2 0.01 1000 = 60 := by
  sorry

end NUMINAMATH_CALUDE_mass_of_man_on_boat_l2783_278318
