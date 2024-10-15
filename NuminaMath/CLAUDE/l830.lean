import Mathlib

namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l830_83025

/-- The eccentricity of a hyperbola tangent to a specific circle -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ (x y : ℝ), (x - Real.sqrt 3)^2 + (y - 1)^2 = 3 ∧ 
    (x^2 / a^2 - y^2 / b^2 = 1) ∧ 
    ((Real.sqrt 3 * b - a)^2 = 3 * (b^2 + a^2) ∨ (Real.sqrt 3 * b + a)^2 = 3 * (b^2 + a^2))) →
  Real.sqrt (a^2 + b^2) / a = 2 * Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l830_83025


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l830_83006

-- Define repeating decimals
def repeating_decimal_8 : ℚ := 8/9
def repeating_decimal_2 : ℚ := 2/9

-- Theorem statement
theorem sum_of_repeating_decimals : 
  repeating_decimal_8 + repeating_decimal_2 = 10/9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l830_83006


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l830_83010

theorem complex_number_quadrant : ∀ z : ℂ, 
  (3 - 2*I) * z = 4 + 3*I → 
  (0 < z.re ∧ 0 < z.im) := by
sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l830_83010


namespace NUMINAMATH_CALUDE_base_eight_132_equals_90_l830_83089

def base_eight_to_ten (a b c : Nat) : Nat :=
  a * 8^2 + b * 8^1 + c * 8^0

theorem base_eight_132_equals_90 : base_eight_to_ten 1 3 2 = 90 := by
  sorry

end NUMINAMATH_CALUDE_base_eight_132_equals_90_l830_83089


namespace NUMINAMATH_CALUDE_allan_correct_answers_l830_83079

theorem allan_correct_answers (total_questions : ℕ) 
  (correct_points : ℚ) (incorrect_penalty : ℚ) (final_score : ℚ) :
  total_questions = 120 →
  correct_points = 1 →
  incorrect_penalty = 1/4 →
  final_score = 100 →
  ∃ (correct_answers : ℕ),
    correct_answers ≤ total_questions ∧
    (correct_answers : ℚ) * correct_points + 
    ((total_questions - correct_answers) : ℚ) * (-incorrect_penalty) = final_score ∧
    correct_answers = 104 := by
  sorry

end NUMINAMATH_CALUDE_allan_correct_answers_l830_83079


namespace NUMINAMATH_CALUDE_standard_ellipse_foci_l830_83028

/-- Represents an ellipse with equation (x^2 / 10) + y^2 = 1 -/
structure StandardEllipse where
  equation : ∀ (x y : ℝ), (x^2 / 10) + y^2 = 1

/-- Represents the foci of an ellipse -/
structure EllipseFoci where
  x : ℝ
  y : ℝ

/-- Theorem: The foci of the standard ellipse are at (3, 0) and (-3, 0) -/
theorem standard_ellipse_foci (e : StandardEllipse) : 
  ∃ (f1 f2 : EllipseFoci), f1.x = 3 ∧ f1.y = 0 ∧ f2.x = -3 ∧ f2.y = 0 :=
sorry

end NUMINAMATH_CALUDE_standard_ellipse_foci_l830_83028


namespace NUMINAMATH_CALUDE_initial_distance_is_40_l830_83082

/-- The initial distance between two people walking towards each other -/
def initial_distance (speed : ℝ) (distance_walked : ℝ) : ℝ :=
  2 * distance_walked

/-- Theorem: The initial distance between Fred and Sam is 40 miles -/
theorem initial_distance_is_40 :
  let fred_speed : ℝ := 4
  let sam_speed : ℝ := 4
  let sam_distance : ℝ := 20
  initial_distance fred_speed sam_distance = 40 := by
  sorry


end NUMINAMATH_CALUDE_initial_distance_is_40_l830_83082


namespace NUMINAMATH_CALUDE_polynomial_inequality_l830_83007

theorem polynomial_inequality (n : ℕ) (hn : n > 1) :
  ∀ x : ℝ, x > 0 → x^n - n*x + n - 1 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_inequality_l830_83007


namespace NUMINAMATH_CALUDE_binomial_unique_parameters_l830_83049

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The expectation of a linear transformation of a binomial random variable -/
def expectation (X : BinomialRV) (a b : ℝ) : ℝ := a * X.n * X.p + b

/-- The variance of a linear transformation of a binomial random variable -/
def variance (X : BinomialRV) (a : ℝ) : ℝ := a^2 * X.n * X.p * (1 - X.p)

/-- Theorem: If E(3X + 2) = 9.2 and D(3X + 2) = 12.96 for X ~ B(n, p), then n = 6 and p = 0.4 -/
theorem binomial_unique_parameters (X : BinomialRV) 
  (h2 : expectation X 3 2 = 9.2)
  (h3 : variance X 3 = 12.96) : 
  X.n = 6 ∧ X.p = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_binomial_unique_parameters_l830_83049


namespace NUMINAMATH_CALUDE_labourer_savings_l830_83065

/-- Calculates the amount saved by a labourer after clearing debt -/
def amount_saved (monthly_income : ℕ) (initial_expense : ℕ) (initial_months : ℕ) (reduced_expense : ℕ) (reduced_months : ℕ) : ℕ :=
  let initial_total_expense := initial_expense * initial_months
  let initial_total_income := monthly_income * initial_months
  let debt := if initial_total_expense > initial_total_income then initial_total_expense - initial_total_income else 0
  let reduced_total_expense := reduced_expense * reduced_months
  let reduced_total_income := monthly_income * reduced_months
  reduced_total_income - (reduced_total_expense + debt)

/-- Theorem stating the amount saved by the labourer -/
theorem labourer_savings :
  amount_saved 81 90 6 60 4 = 30 :=
by sorry

end NUMINAMATH_CALUDE_labourer_savings_l830_83065


namespace NUMINAMATH_CALUDE_number_of_fractions_l830_83056

/-- A function that determines if an expression is a fraction in the form a/b -/
def isFraction (expr : String) : Bool :=
  match expr with
  | "5/(a-x)" => true
  | "(m+n)/(mn)" => true
  | "5x^2/x" => true
  | _ => false

/-- The list of expressions given in the problem -/
def expressions : List String :=
  ["1/5(1-x)", "5/(a-x)", "4x/(π-3)", "(m+n)/(mn)", "(x^2-y^2)/2", "5x^2/x"]

/-- Theorem stating that the number of fractions in the given list is 3 -/
theorem number_of_fractions : 
  (expressions.filter isFraction).length = 3 := by sorry

end NUMINAMATH_CALUDE_number_of_fractions_l830_83056


namespace NUMINAMATH_CALUDE_gumdrops_problem_l830_83090

/-- The maximum number of gumdrops that can be bought with a given amount of money and cost per gumdrop. -/
def max_gumdrops (total_money : ℕ) (cost_per_gumdrop : ℕ) : ℕ :=
  total_money / cost_per_gumdrop

/-- Theorem stating that with 80 cents and gumdrops costing 4 cents each, the maximum number of gumdrops that can be bought is 20. -/
theorem gumdrops_problem :
  max_gumdrops 80 4 = 20 := by
  sorry

end NUMINAMATH_CALUDE_gumdrops_problem_l830_83090


namespace NUMINAMATH_CALUDE_dans_remaining_money_l830_83017

def remaining_money (initial_amount spending : ℚ) : ℚ :=
  initial_amount - spending

theorem dans_remaining_money :
  remaining_money 4 3 = 1 :=
by sorry

end NUMINAMATH_CALUDE_dans_remaining_money_l830_83017


namespace NUMINAMATH_CALUDE_min_books_borrowed_l830_83012

/-- Represents the minimum number of books borrowed by the remaining students -/
def min_books_remaining : ℕ := 4

theorem min_books_borrowed (total_students : ℕ) (no_books : ℕ) (one_book : ℕ) (two_books : ℕ) 
  (avg_books : ℚ) (h1 : total_students = 38) (h2 : no_books = 2) (h3 : one_book = 12) 
  (h4 : two_books = 10) (h5 : avg_books = 2) : 
  min_books_remaining = 4 := by
  sorry

#check min_books_borrowed

end NUMINAMATH_CALUDE_min_books_borrowed_l830_83012


namespace NUMINAMATH_CALUDE_grass_field_width_l830_83069

/-- Proves that the width of a rectangular grass field is 192 meters, given specific conditions --/
theorem grass_field_width : 
  ∀ (w : ℝ),
  (82 * (w + 7) - 75 * w = 1918) →
  w = 192 := by
  sorry

end NUMINAMATH_CALUDE_grass_field_width_l830_83069


namespace NUMINAMATH_CALUDE_positive_integer_triple_characterization_l830_83042

theorem positive_integer_triple_characterization :
  ∀ (a b c : ℕ+),
    (a.val^2 = 2^b.val + c.val^4) →
    (a.val % 2 = 1 ∨ b.val % 2 = 1 ∨ c.val % 2 = 1) →
    (a.val % 2 = 0 ∨ b.val % 2 = 0) →
    (a.val % 2 = 0 ∨ c.val % 2 = 0) →
    (b.val % 2 = 0 ∨ c.val % 2 = 0) →
    ∃ (n : ℕ+), a.val = 3 * 2^(2*n.val) ∧ b.val = 4*n.val + 3 ∧ c.val = 2^n.val :=
by sorry

end NUMINAMATH_CALUDE_positive_integer_triple_characterization_l830_83042


namespace NUMINAMATH_CALUDE_smallest_base_for_perfect_square_l830_83052

theorem smallest_base_for_perfect_square : 
  ∀ b : ℕ, b > 4 → (∃ n : ℕ, 3 * b + 4 = n^2) → b ≥ 7 :=
by sorry

end NUMINAMATH_CALUDE_smallest_base_for_perfect_square_l830_83052


namespace NUMINAMATH_CALUDE_labourer_absence_proof_l830_83046

def total_days : ℕ := 25
def daily_wage : ℚ := 2
def daily_fine : ℚ := 1/2
def total_received : ℚ := 75/2

def days_absent : ℕ := 5

theorem labourer_absence_proof :
  ∃ (days_worked : ℕ),
    days_worked + days_absent = total_days ∧
    daily_wage * days_worked - daily_fine * days_absent = total_received :=
by sorry

end NUMINAMATH_CALUDE_labourer_absence_proof_l830_83046


namespace NUMINAMATH_CALUDE_austin_bicycle_weeks_l830_83055

/-- The number of weeks Austin needs to work to buy a bicycle -/
def weeks_to_buy_bicycle (hourly_rate : ℚ) (monday_hours : ℚ) (wednesday_hours : ℚ) (friday_hours : ℚ) (bicycle_cost : ℚ) : ℚ :=
  bicycle_cost / (hourly_rate * (monday_hours + wednesday_hours + friday_hours))

/-- Proof that Austin needs 6 weeks to buy the bicycle -/
theorem austin_bicycle_weeks : 
  weeks_to_buy_bicycle 5 2 1 3 180 = 6 := by
  sorry

end NUMINAMATH_CALUDE_austin_bicycle_weeks_l830_83055


namespace NUMINAMATH_CALUDE_car_price_proof_l830_83026

/-- Calculates the price of a car given loan terms and payments -/
def carPrice (loanYears : ℕ) (downPayment : ℕ) (monthlyPayment : ℕ) : ℕ :=
  downPayment + loanYears * 12 * monthlyPayment

/-- Proves that the price of the car is $20,000 given the specified conditions -/
theorem car_price_proof :
  carPrice 5 5000 250 = 20000 := by
  sorry

#eval carPrice 5 5000 250

end NUMINAMATH_CALUDE_car_price_proof_l830_83026


namespace NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l830_83072

theorem least_positive_integer_with_remainders : ∃ b : ℕ+, 
  (b : ℤ) % 4 = 1 ∧ 
  (b : ℤ) % 5 = 2 ∧ 
  (b : ℤ) % 6 = 3 ∧ 
  (∀ c : ℕ+, c < b → 
    (c : ℤ) % 4 ≠ 1 ∨ 
    (c : ℤ) % 5 ≠ 2 ∨ 
    (c : ℤ) % 6 ≠ 3) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l830_83072


namespace NUMINAMATH_CALUDE_collinear_points_implies_b_value_l830_83064

/-- Three points (x₁, y₁), (x₂, y₂), and (x₃, y₃) are collinear if and only if
    (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁) -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁)

/-- If the points (5, -3), (2b + 4, 5), and (-3b + 6, -1) are collinear, then b = 5/14 -/
theorem collinear_points_implies_b_value :
  ∀ b : ℝ, collinear 5 (-3) (2*b + 4) 5 (-3*b + 6) (-1) → b = 5/14 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_implies_b_value_l830_83064


namespace NUMINAMATH_CALUDE_equal_distribution_l830_83048

theorem equal_distribution (total_amount : ℕ) (num_persons : ℕ) (amount_per_person : ℕ) : 
  total_amount = 42900 →
  num_persons = 22 →
  amount_per_person = total_amount / num_persons →
  amount_per_person = 1950 := by
  sorry

end NUMINAMATH_CALUDE_equal_distribution_l830_83048


namespace NUMINAMATH_CALUDE_no_bounded_sequences_with_property_l830_83011

theorem no_bounded_sequences_with_property :
  ¬ ∃ (a b : ℕ → ℝ),
    (∃ M : ℝ, ∀ n, |a n| ≤ M ∧ |b n| ≤ M) ∧
    (∀ n m : ℕ, m > n → |a m - a n| > 1 / Real.sqrt n ∨ |b m - b n| > 1 / Real.sqrt n) :=
sorry

end NUMINAMATH_CALUDE_no_bounded_sequences_with_property_l830_83011


namespace NUMINAMATH_CALUDE_project_work_time_difference_l830_83014

/-- Given three people working on a project for a total of 140 hours,
    with their working times in the ratio of 3:5:6,
    prove that the difference between the longest and shortest working times is 30 hours. -/
theorem project_work_time_difference (x : ℝ) 
  (h1 : 3 * x + 5 * x + 6 * x = 140) : 6 * x - 3 * x = 30 := by
  sorry

end NUMINAMATH_CALUDE_project_work_time_difference_l830_83014


namespace NUMINAMATH_CALUDE_vector_sum_collinear_points_l830_83013

/-- Given points A, B, C are collinear, O is not on their line, and 
    p⃗OA + q⃗OB + r⃗OC = 0⃗, then p + q + r = 0 -/
theorem vector_sum_collinear_points 
  (O A B C : EuclideanSpace ℝ (Fin 3))
  (p q r : ℝ) :
  Collinear ℝ ({A, B, C} : Set (EuclideanSpace ℝ (Fin 3))) →
  O ∉ affineSpan ℝ {A, B, C} →
  p • (A - O) + q • (B - O) + r • (C - O) = 0 →
  p + q + r = 0 := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_collinear_points_l830_83013


namespace NUMINAMATH_CALUDE_mean_median_difference_l830_83088

-- Define the frequency distribution of days missed
def days_missed : List (Nat × Nat) := [
  (0, 2),  -- 2 students missed 0 days
  (1, 3),  -- 3 students missed 1 day
  (2, 6),  -- 6 students missed 2 days
  (3, 5),  -- 5 students missed 3 days
  (4, 2),  -- 2 students missed 4 days
  (5, 2)   -- 2 students missed 5 days
]

-- Define the total number of students
def total_students : Nat := 20

-- Theorem statement
theorem mean_median_difference :
  let mean := (days_missed.map (λ (d, f) => d * f)).sum / total_students
  let median := 2  -- The median is 2 days (10th and 11th students both missed 2 days)
  mean - median = 2 / 5 := by sorry


end NUMINAMATH_CALUDE_mean_median_difference_l830_83088


namespace NUMINAMATH_CALUDE_quadratic_equation_root_l830_83045

theorem quadratic_equation_root (x : ℝ) : x^2 - 6*x - 4 = 0 ↔ x = Real.sqrt 5 - 3 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_l830_83045


namespace NUMINAMATH_CALUDE_book_reading_fraction_l830_83074

theorem book_reading_fraction (total_pages : ℝ) (pages_read_more : ℝ) : 
  total_pages = 270.00000000000006 →
  pages_read_more = 90 →
  (total_pages / 2 + pages_read_more / 2) / total_pages = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_book_reading_fraction_l830_83074


namespace NUMINAMATH_CALUDE_largest_valid_sample_size_l830_83099

def population : ℕ := 36

def is_valid_sample_size (X : ℕ) : Prop :=
  (population % X = 0) ∧ (population % (X + 1) ≠ 0)

theorem largest_valid_sample_size :
  ∃ (X : ℕ), is_valid_sample_size X ∧ ∀ (Y : ℕ), Y > X → ¬is_valid_sample_size Y :=
by
  sorry

end NUMINAMATH_CALUDE_largest_valid_sample_size_l830_83099


namespace NUMINAMATH_CALUDE_negation_of_proposition_l830_83083

open Real

theorem negation_of_proposition (p : ∀ x : ℝ, x > 0 → (x + 1) * Real.exp x > 1) :
  ∃ x₀ : ℝ, x₀ > 0 ∧ (x₀ + 1) * Real.exp x₀ ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l830_83083


namespace NUMINAMATH_CALUDE_smallest_among_given_numbers_l830_83019

theorem smallest_among_given_numbers : 
  let a := Real.sqrt 3
  let b := -(1/3 : ℝ)
  let c := -2
  let d := 0
  c < b ∧ c < d ∧ c < a :=
by sorry

end NUMINAMATH_CALUDE_smallest_among_given_numbers_l830_83019


namespace NUMINAMATH_CALUDE_impossible_continuous_coverage_l830_83031

/-- Represents a runner on the track -/
structure Runner where
  speed : ℕ
  startPosition : ℝ

/-- Represents the circular track with runners -/
structure Track where
  length : ℝ
  spectatorStandLength : ℝ
  runners : List Runner

/-- Checks if a runner is passing the spectator stands at a given time -/
def isPassingStands (runner : Runner) (track : Track) (time : ℝ) : Prop :=
  let position := (runner.startPosition + runner.speed * time) % track.length
  0 ≤ position ∧ position < track.spectatorStandLength

/-- Main theorem statement -/
theorem impossible_continuous_coverage (track : Track) : 
  track.length = 2000 ∧ 
  track.spectatorStandLength = 100 ∧ 
  track.runners.length = 20 ∧
  (∀ i, i ∈ Finset.range 20 → 
    ∃ r ∈ track.runners, r.speed = i + 10) →
  ¬ (∀ t : ℝ, ∃ r ∈ track.runners, isPassingStands r track t) :=
by sorry

end NUMINAMATH_CALUDE_impossible_continuous_coverage_l830_83031


namespace NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l830_83027

/-- Determinant of a 2x2 matrix -/
def det (a b c d : ℝ) : ℝ := a * d - b * c

/-- Geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_seventh_term
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_third : a 3 = 1)
  (h_det : det (a 6) 8 8 (a 8) = 0) :
  a 7 = 8 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l830_83027


namespace NUMINAMATH_CALUDE_function_inequality_implies_constant_l830_83024

/-- A function f: ℝ → ℝ satisfying f(x+y) ≤ f(x^2+y) for all x, y ∈ ℝ is constant. -/
theorem function_inequality_implies_constant (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x + y) ≤ f (x^2 + y)) : 
  ∃ c : ℝ, ∀ x : ℝ, f x = c := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_implies_constant_l830_83024


namespace NUMINAMATH_CALUDE_distinct_numbers_count_l830_83003

/-- Represents the possible states of a matchstick (present or removed) --/
inductive MatchstickState
| Present
| Removed

/-- Represents the configuration of matchsticks in the symbol --/
structure MatchstickConfiguration :=
(top : MatchstickState)
(bottom : MatchstickState)
(left : MatchstickState)
(right : MatchstickState)

/-- Defines the set of valid number representations --/
def ValidNumberRepresentations : Set MatchstickConfiguration := sorry

/-- Counts the number of distinct valid number representations --/
def CountDistinctNumbers : Nat := sorry

/-- Theorem stating that the number of distinct numbers obtainable is 5 --/
theorem distinct_numbers_count :
  CountDistinctNumbers = 5 := by sorry

end NUMINAMATH_CALUDE_distinct_numbers_count_l830_83003


namespace NUMINAMATH_CALUDE_work_earnings_equation_l830_83036

theorem work_earnings_equation (t : ℚ) : 
  (t + 2) * (3 * t - 2) = (3 * t - 4) * (t + 1) + 5 → t = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_work_earnings_equation_l830_83036


namespace NUMINAMATH_CALUDE_three_digit_divisibility_by_nine_l830_83077

/-- Function to calculate the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- Theorem stating that for all three-digit numbers, if the sum of digits is divisible by 9, then the number is divisible by 9 -/
theorem three_digit_divisibility_by_nine :
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 → (sumOfDigits n % 9 = 0 → n % 9 = 0) :=
by
  sorry

#check three_digit_divisibility_by_nine

end NUMINAMATH_CALUDE_three_digit_divisibility_by_nine_l830_83077


namespace NUMINAMATH_CALUDE_square_of_negative_sqrt_two_l830_83062

theorem square_of_negative_sqrt_two : (-Real.sqrt 2)^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_negative_sqrt_two_l830_83062


namespace NUMINAMATH_CALUDE_relationship_abc_l830_83066

noncomputable def a : ℝ := Real.rpow 0.6 0.6
noncomputable def b : ℝ := Real.rpow 0.6 1.5
noncomputable def c : ℝ := Real.rpow 1.5 0.6

theorem relationship_abc : b < a ∧ a < c := by
  sorry

end NUMINAMATH_CALUDE_relationship_abc_l830_83066


namespace NUMINAMATH_CALUDE_final_amount_calculation_l830_83084

def initial_amount : ℕ := 5
def spent_amount : ℕ := 2
def allowance : ℕ := 26

theorem final_amount_calculation :
  initial_amount - spent_amount + allowance = 29 := by
  sorry

end NUMINAMATH_CALUDE_final_amount_calculation_l830_83084


namespace NUMINAMATH_CALUDE_min_distance_complex_circles_l830_83096

/-- The minimum distance between two complex numbers on specific circles -/
theorem min_distance_complex_circles :
  ∀ (z w : ℂ),
  Complex.abs (z - (2 + Complex.I)) = 2 →
  Complex.abs (w + (3 + 4 * Complex.I)) = 4 →
  (∀ (z' w' : ℂ),
    Complex.abs (z' - (2 + Complex.I)) = 2 →
    Complex.abs (w' + (3 + 4 * Complex.I)) = 4 →
    Complex.abs (z - w) ≤ Complex.abs (z' - w')) →
  Complex.abs (z - w) = 5 * Real.sqrt 2 - 6 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_complex_circles_l830_83096


namespace NUMINAMATH_CALUDE_equation_solution_l830_83086

theorem equation_solution (k : ℤ) : 
  (∃ x : ℤ, x > 0 ∧ 9*x - 3 = k*x + 14) ↔ (k = 8 ∨ k = -8) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l830_83086


namespace NUMINAMATH_CALUDE_parallelogram_height_l830_83051

/-- The height of a parallelogram given its area and base -/
theorem parallelogram_height (area base height : ℝ) : 
  area = base * height → area = 320 → base = 20 → height = 16 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_height_l830_83051


namespace NUMINAMATH_CALUDE_work_rate_problem_l830_83033

theorem work_rate_problem (a b : ℝ) (h1 : a = (1/2) * b) (h2 : (a + b) * 20 = 1) :
  1 / b = 30 := by sorry

end NUMINAMATH_CALUDE_work_rate_problem_l830_83033


namespace NUMINAMATH_CALUDE_jodi_walking_schedule_l830_83076

/-- Represents Jodi's walking schedule over 4 weeks -/
structure WalkingSchedule where
  week1_distance : ℝ
  week2_distance : ℝ
  week3_distance : ℝ
  week4_distance : ℝ
  days_per_week : ℕ
  total_distance : ℝ

/-- Theorem stating that given Jodi's walking schedule, she walked 2 miles per day in the second week -/
theorem jodi_walking_schedule (schedule : WalkingSchedule) 
  (h1 : schedule.week1_distance = 1)
  (h2 : schedule.week3_distance = 3)
  (h3 : schedule.week4_distance = 4)
  (h4 : schedule.days_per_week = 6)
  (h5 : schedule.total_distance = 60)
  : schedule.week2_distance = 2 := by
  sorry

end NUMINAMATH_CALUDE_jodi_walking_schedule_l830_83076


namespace NUMINAMATH_CALUDE_multiple_root_equation_l830_83037

/-- The equation x^4 + p^2*x + q = 0 has a multiple root if and only if p = 2 and q = 3, where p and q are positive prime numbers. -/
theorem multiple_root_equation (p q : ℕ) : 
  (Prime p ∧ Prime q ∧ 0 < p ∧ 0 < q) →
  (∃ (x : ℝ), (x^4 + p^2*x + q = 0 ∧ 
    ∃ (y : ℝ), y ≠ x ∧ y^4 + p^2*y + q = 0 ∧
    (∀ (z : ℝ), z^4 + p^2*z + q = 0 → z = x ∨ z = y))) ↔ 
  (p = 2 ∧ q = 3) :=
by sorry

end NUMINAMATH_CALUDE_multiple_root_equation_l830_83037


namespace NUMINAMATH_CALUDE_line_point_k_value_l830_83080

/-- Given a line containing the points (0, 7), (15, k), and (20, 3), prove that k = 4 -/
theorem line_point_k_value (k : ℝ) : 
  (∀ (x y : ℝ), (x = 0 ∧ y = 7) ∨ (x = 15 ∧ y = k) ∨ (x = 20 ∧ y = 3) → 
    ∃ (m b : ℝ), y = m * x + b) → 
  k = 4 := by
sorry

end NUMINAMATH_CALUDE_line_point_k_value_l830_83080


namespace NUMINAMATH_CALUDE_population_percentage_l830_83058

theorem population_percentage : 
  let total_population : ℕ := 40000
  let part_population : ℕ := 32000
  (part_population : ℚ) / (total_population : ℚ) * 100 = 80 := by
  sorry

end NUMINAMATH_CALUDE_population_percentage_l830_83058


namespace NUMINAMATH_CALUDE_a_lt_b_neither_sufficient_nor_necessary_for_a_sq_lt_b_sq_l830_83067

theorem a_lt_b_neither_sufficient_nor_necessary_for_a_sq_lt_b_sq :
  ∃ (a b c d : ℝ),
    (a < b ∧ ¬(a^2 < b^2)) ∧
    (c^2 < d^2 ∧ ¬(c < d)) :=
sorry

end NUMINAMATH_CALUDE_a_lt_b_neither_sufficient_nor_necessary_for_a_sq_lt_b_sq_l830_83067


namespace NUMINAMATH_CALUDE_relationship_abc_l830_83021

theorem relationship_abc (a b c : ℝ) 
  (ha : a = 2^(1/5))
  (hb : b = 2^(3/10))
  (hc : c = Real.log 2 / Real.log 3) :
  c < a ∧ a < b :=
sorry

end NUMINAMATH_CALUDE_relationship_abc_l830_83021


namespace NUMINAMATH_CALUDE_factorization_equality_l830_83057

theorem factorization_equality (x y : ℝ) : (x + y)^2 - 14*(x + y) + 49 = (x + y - 7)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l830_83057


namespace NUMINAMATH_CALUDE_no_prime_sum_53_l830_83047

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem no_prime_sum_53 : ¬∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p + q = 53 := by
  sorry

end NUMINAMATH_CALUDE_no_prime_sum_53_l830_83047


namespace NUMINAMATH_CALUDE_min_value_fraction_sum_min_value_fraction_sum_achievable_l830_83034

theorem min_value_fraction_sum (a b : ℤ) (h : a > b) :
  (((2 * a + b) : ℚ) / (a - b : ℚ)) + ((a - b : ℚ) / ((2 * a + b) : ℚ)) ≥ 13 / 6 :=
by sorry

theorem min_value_fraction_sum_achievable :
  ∃ (a b : ℤ), a > b ∧ (((2 * a + b) : ℚ) / (a - b : ℚ)) + ((a - b : ℚ) / ((2 * a + b) : ℚ)) = 13 / 6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_fraction_sum_min_value_fraction_sum_achievable_l830_83034


namespace NUMINAMATH_CALUDE_prob_not_at_ends_eight_chairs_l830_83023

/-- The number of chairs in the row -/
def n : ℕ := 8

/-- The probability of two people not sitting at either end when randomly choosing seats in a row of n chairs -/
def prob_not_at_ends (n : ℕ) : ℚ :=
  1 - (2 + 2 * (n - 2)) / (n.choose 2)

theorem prob_not_at_ends_eight_chairs :
  prob_not_at_ends n = 3/7 := by sorry

end NUMINAMATH_CALUDE_prob_not_at_ends_eight_chairs_l830_83023


namespace NUMINAMATH_CALUDE_stratified_sampling_total_components_l830_83098

theorem stratified_sampling_total_components :
  let total_sample_size : ℕ := 45
  let sample_size_A : ℕ := 20
  let sample_size_C : ℕ := 10
  let num_B : ℕ := 300
  let num_C : ℕ := 200
  let num_A : ℕ := (total_sample_size * (num_B + num_C)) / (total_sample_size - sample_size_A - sample_size_C)
  num_A + num_B + num_C = 900 := by
  sorry


end NUMINAMATH_CALUDE_stratified_sampling_total_components_l830_83098


namespace NUMINAMATH_CALUDE_intersection_empty_implies_a_geq_one_l830_83040

theorem intersection_empty_implies_a_geq_one (a : ℝ) : 
  let A : Set ℝ := {0, 1}
  let B : Set ℝ := {x | x > a}
  A ∩ B = ∅ → a ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_intersection_empty_implies_a_geq_one_l830_83040


namespace NUMINAMATH_CALUDE_de_Bruijn_Erdos_l830_83041

/-- A graph is a pair of a vertex set and an edge relation -/
structure Graph (V : Type) :=
  (edge : V → V → Prop)

/-- The chromatic number of a graph is the smallest number of colors needed to color the graph -/
def chromaticNumber {V : Type} (G : Graph V) : ℕ := sorry

/-- A subgraph of G induced by a subset of vertices -/
def inducedSubgraph {V : Type} (G : Graph V) (S : Set V) : Graph S := sorry

/-- A graph is finite if its vertex set is finite -/
def isFinite {V : Type} (G : Graph V) : Prop := sorry

theorem de_Bruijn_Erdos {V : Type} (G : Graph V) (k : ℕ) :
  (∀ (S : Set V), isFinite (inducedSubgraph G S) → chromaticNumber (inducedSubgraph G S) ≤ k) →
  chromaticNumber G ≤ k := by sorry

end NUMINAMATH_CALUDE_de_Bruijn_Erdos_l830_83041


namespace NUMINAMATH_CALUDE_both_hit_target_probability_l830_83032

theorem both_hit_target_probability
  (prob_A : ℝ)
  (prob_B : ℝ)
  (h_A : prob_A = 0.8)
  (h_B : prob_B = 0.6) :
  prob_A * prob_B = 0.48 := by
sorry

end NUMINAMATH_CALUDE_both_hit_target_probability_l830_83032


namespace NUMINAMATH_CALUDE_expression_value_l830_83004

theorem expression_value (a b c : ℝ) (h : a * b + b * c + c * a = 3) :
  (a * (b^2 + 3)) / (a + b) + (b * (c^2 + 3)) / (b + c) + (c * (a^2 + 3)) / (c + a) = 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l830_83004


namespace NUMINAMATH_CALUDE_quadratic_equation_complete_square_l830_83009

theorem quadratic_equation_complete_square (m n : ℝ) : 
  (∀ x, 15 * x^2 - 30 * x - 45 = 0 ↔ (x + m)^2 = n) → m + n = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_complete_square_l830_83009


namespace NUMINAMATH_CALUDE_exactly_one_true_iff_or_and_not_and_l830_83043

theorem exactly_one_true_iff_or_and_not_and (p q : Prop) :
  ((p ∨ q) ∧ ¬(p ∧ q)) ↔ (p ∨ q) ∧ ¬(p ↔ q) := by
  sorry

end NUMINAMATH_CALUDE_exactly_one_true_iff_or_and_not_and_l830_83043


namespace NUMINAMATH_CALUDE_exactly_two_statements_true_l830_83053

def M (x : ℝ) : ℝ := 2 - 4*x
def N (x : ℝ) : ℝ := 4*x + 1

def statement1 : Prop := ¬ ∃ x : ℝ, M x + N x = 0
def statement2 : Prop := ∀ x : ℝ, ¬(M x > 0 ∧ N x > 0)
def statement3 : Prop := ∀ a : ℝ, (∀ x : ℝ, (M x + a) * N x = 1 - 16*x^2) → a = -1
def statement4 : Prop := ∀ x : ℝ, M x * N x = -3 → M x^2 + N x^2 = 11

theorem exactly_two_statements_true : 
  ∃! n : Fin 4, (n.val = 2 ∧ 
    (statement1 ∧ statement3) ∨
    (statement1 ∧ statement2) ∨
    (statement1 ∧ statement4) ∨
    (statement2 ∧ statement3) ∨
    (statement2 ∧ statement4) ∨
    (statement3 ∧ statement4)) :=
by sorry

end NUMINAMATH_CALUDE_exactly_two_statements_true_l830_83053


namespace NUMINAMATH_CALUDE_boxer_win_ratio_is_one_l830_83016

/-- Represents a boxer's career statistics -/
structure BoxerStats where
  wins_before_first_loss : ℕ
  total_losses : ℕ
  win_loss_difference : ℕ

/-- Calculates the ratio of wins after first loss to wins before first loss -/
def win_ratio (stats : BoxerStats) : ℚ :=
  let wins_after_first_loss := stats.win_loss_difference + stats.total_losses - stats.wins_before_first_loss
  wins_after_first_loss / stats.wins_before_first_loss

/-- Theorem stating that for a boxer with given statistics, the win ratio is 1 -/
theorem boxer_win_ratio_is_one (stats : BoxerStats)
  (h1 : stats.wins_before_first_loss = 15)
  (h2 : stats.total_losses = 2)
  (h3 : stats.win_loss_difference = 28) :
  win_ratio stats = 1 := by
  sorry

end NUMINAMATH_CALUDE_boxer_win_ratio_is_one_l830_83016


namespace NUMINAMATH_CALUDE_negation_cube_even_number_l830_83081

theorem negation_cube_even_number (n : ℤ) :
  ¬(∀ n : ℤ, 2 ∣ n → 2 ∣ n^3) ↔ ∃ n : ℤ, 2 ∣ n ∧ ¬(2 ∣ n^3) :=
sorry

end NUMINAMATH_CALUDE_negation_cube_even_number_l830_83081


namespace NUMINAMATH_CALUDE_complex_modulus_l830_83068

theorem complex_modulus (z : ℂ) (h : z^2 = -4) : Complex.abs (1 + z) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l830_83068


namespace NUMINAMATH_CALUDE_alternate_arrangement_probability_l830_83002

/-- The number of male fans -/
def num_male : ℕ := 3

/-- The number of female fans -/
def num_female : ℕ := 3

/-- The total number of fans -/
def total_fans : ℕ := num_male + num_female

/-- The number of ways to arrange fans alternately -/
def alternate_arrangements : ℕ := 2 * (Nat.factorial num_male) * (Nat.factorial num_female)

/-- The total number of possible arrangements -/
def total_arrangements : ℕ := Nat.factorial total_fans

/-- The probability of arranging fans alternately -/
def prob_alternate : ℚ := alternate_arrangements / total_arrangements

theorem alternate_arrangement_probability :
  prob_alternate = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_alternate_arrangement_probability_l830_83002


namespace NUMINAMATH_CALUDE_fraction_equality_l830_83094

theorem fraction_equality (a b : ℝ) (h1 : a ≠ b) (h2 : b ≠ 0) :
  let x := a / b
  (a^2 + b^2) / (a^2 - b^2) = (x^2 + 1) / (x^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l830_83094


namespace NUMINAMATH_CALUDE_modulo_residue_problem_l830_83008

theorem modulo_residue_problem : (392 + 6 * 51 + 8 * 221 + 3^2 * 23) % 17 = 11 := by
  sorry

end NUMINAMATH_CALUDE_modulo_residue_problem_l830_83008


namespace NUMINAMATH_CALUDE_unique_k_for_prime_roots_l830_83029

/-- A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself. -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 0 ∧ m < n → n % m ≠ 0

/-- The roots of a quadratic equation ax^2 + bx + c = 0 are given by (-b ± √(b^2 - 4ac)) / (2a) -/
def isRootOf (x : ℝ) (a b c : ℝ) : Prop := a * x^2 + b * x + c = 0

theorem unique_k_for_prime_roots : ∃! k : ℕ, 
  ∃ p q : ℕ, 
    isPrime p ∧ 
    isPrime q ∧ 
    isRootOf p 1 (-63) k ∧ 
    isRootOf q 1 (-63) k :=
sorry

end NUMINAMATH_CALUDE_unique_k_for_prime_roots_l830_83029


namespace NUMINAMATH_CALUDE_subset_implies_membership_l830_83075

theorem subset_implies_membership {α : Type*} (A B : Set α) (h : A ⊆ B) :
  ∀ x, x ∈ A → x ∈ B := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_membership_l830_83075


namespace NUMINAMATH_CALUDE_smallest_negative_integer_congruence_l830_83070

theorem smallest_negative_integer_congruence :
  ∃ (x : ℤ), x < 0 ∧ (45 * x + 8) % 24 = 5 ∧
  ∀ (y : ℤ), y < 0 ∧ (45 * y + 8) % 24 = 5 → x ≥ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_negative_integer_congruence_l830_83070


namespace NUMINAMATH_CALUDE_largest_integer_problem_l830_83035

theorem largest_integer_problem :
  ∃ (m : ℕ), m < 150 ∧ m > 50 ∧ 
  (∃ (a : ℕ), m = 9 * a - 2) ∧
  (∃ (b : ℕ), m = 6 * b - 4) ∧
  (∀ (n : ℕ), n < 150 ∧ n > 50 ∧ 
    (∃ (c : ℕ), n = 9 * c - 2) ∧
    (∃ (d : ℕ), n = 6 * d - 4) → n ≤ m) ∧
  m = 106 :=
sorry

end NUMINAMATH_CALUDE_largest_integer_problem_l830_83035


namespace NUMINAMATH_CALUDE_concentric_circles_area_ratio_l830_83030

theorem concentric_circles_area_ratio 
  (r R : ℝ) 
  (h_positive : r > 0) 
  (h_ratio : (π * R^2) / (π * r^2) = 4) : 
  R = 2 * r ∧ R - r = r := by
sorry

end NUMINAMATH_CALUDE_concentric_circles_area_ratio_l830_83030


namespace NUMINAMATH_CALUDE_inequality_holds_C_is_maximum_l830_83001

noncomputable def C : ℝ := (Real.sqrt (13 + 16 * Real.sqrt 2) - 1) / 2

theorem inequality_holds (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  x^3 + y^3 + z^3 + C * (x*y^2 + y*z^2 + z*x^2) ≥ (C + 1) * (x^2*y + y^2*z + z^2*x) :=
by sorry

theorem C_is_maximum : 
  ∀ D : ℝ, D > C → ∃ x y z : ℝ, x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧
    x^3 + y^3 + z^3 + D * (x*y^2 + y*z^2 + z*x^2) < (D + 1) * (x^2*y + y^2*z + z^2*x) :=
by sorry

end NUMINAMATH_CALUDE_inequality_holds_C_is_maximum_l830_83001


namespace NUMINAMATH_CALUDE_complex_number_proof_l830_83015

theorem complex_number_proof (z : ℂ) :
  (∃ (z₁ : ℝ), z₁ = (z / (1 + z^2)).re ∧ (z / (1 + z^2)).im = 0) ∧
  (∃ (z₂ : ℝ), z₂ = (z^2 / (1 + z)).re ∧ (z^2 / (1 + z)).im = 0) →
  z = -1/2 + (Complex.I * Real.sqrt 3) / 2 ∨ z = -1/2 - (Complex.I * Real.sqrt 3) / 2 :=
by sorry

end NUMINAMATH_CALUDE_complex_number_proof_l830_83015


namespace NUMINAMATH_CALUDE_curve_composition_l830_83039

-- Define the curve
def curve (x y : ℝ) : Prop := (3*x - y + 1) * (y - Real.sqrt (1 - x^2)) = 0

-- Define a semicircle
def semicircle (x y : ℝ) : Prop := y = Real.sqrt (1 - x^2) ∧ -1 ≤ x ∧ x ≤ 1

-- Define a line segment
def line_segment (x y : ℝ) : Prop := 3*x - y + 1 = 0 ∧ -1 ≤ x ∧ x ≤ 1

-- Theorem statement
theorem curve_composition :
  ∀ x y : ℝ, curve x y ↔ (semicircle x y ∨ line_segment x y) :=
sorry

end NUMINAMATH_CALUDE_curve_composition_l830_83039


namespace NUMINAMATH_CALUDE_banana_count_l830_83000

theorem banana_count (bananas apples oranges : ℕ) : 
  apples = 2 * bananas →
  oranges = 6 →
  bananas + apples + oranges = 12 →
  bananas = 2 := by
sorry

end NUMINAMATH_CALUDE_banana_count_l830_83000


namespace NUMINAMATH_CALUDE_emily_beads_used_l830_83097

/-- The number of beads Emily has used so far -/
def beads_used (total_made : ℕ) (beads_per_necklace : ℕ) (given_away : ℕ) : ℕ :=
  total_made * beads_per_necklace - given_away * beads_per_necklace

/-- Theorem stating that Emily has used 92 beads -/
theorem emily_beads_used :
  beads_used 35 4 12 = 92 := by
  sorry

end NUMINAMATH_CALUDE_emily_beads_used_l830_83097


namespace NUMINAMATH_CALUDE_log_greater_than_square_near_zero_l830_83085

theorem log_greater_than_square_near_zero :
  ∀ ε > 0, ∃ δ > 0, ∀ x, 0 < x → x < δ → Real.log (1 + x) > x^2 := by
  sorry

end NUMINAMATH_CALUDE_log_greater_than_square_near_zero_l830_83085


namespace NUMINAMATH_CALUDE_inverse_variation_l830_83093

/-- Given that p and q vary inversely, prove that when p = 400, q = 1, 
    given that when p = 800, q = 0.5 -/
theorem inverse_variation (p q : ℝ) (h : p * q = 800 * 0.5) :
  p = 400 → q = 1 := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_l830_83093


namespace NUMINAMATH_CALUDE_unique_triple_l830_83060

theorem unique_triple : 
  ∃! (a b c : ℕ), a ≥ b ∧ b ≥ c ∧ a^3 + 9*b^2 + 9*c + 7 = 1997 ∧ 
  a = 10 ∧ b = 10 ∧ c = 10 := by
  sorry

end NUMINAMATH_CALUDE_unique_triple_l830_83060


namespace NUMINAMATH_CALUDE_no_roots_implies_non_integer_l830_83020

theorem no_roots_implies_non_integer (a b : ℝ) : 
  a ≠ b →
  (∀ x : ℝ, (x^2 + 20*a*x + 10*b) * (x^2 + 20*b*x + 10*a) ≠ 0) →
  ¬(∃ n : ℤ, 20*(b-a) = n) :=
by sorry

end NUMINAMATH_CALUDE_no_roots_implies_non_integer_l830_83020


namespace NUMINAMATH_CALUDE_gcd_840_1764_l830_83071

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end NUMINAMATH_CALUDE_gcd_840_1764_l830_83071


namespace NUMINAMATH_CALUDE_min_value_theorem_l830_83073

theorem min_value_theorem (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hf : ∀ x, |x + a| + |x - b| + c ≥ 4) :
  (a + b + c = 4) ∧ 
  (∀ a' b' c' : ℝ, a' > 0 → b' > 0 → c' > 0 → 1/a' + 1/b' + 1/c' ≥ 9/4) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l830_83073


namespace NUMINAMATH_CALUDE_units_digit_of_n_l830_83078

/-- Given two natural numbers m and n, returns true if their product ends with the digit d -/
def product_ends_with (m n d : ℕ) : Prop :=
  (m * n) % 10 = d

/-- Given a natural number x, returns true if its units digit is d -/
def units_digit (x d : ℕ) : Prop :=
  x % 10 = d

theorem units_digit_of_n (m n : ℕ) :
  product_ends_with m n 4 →
  units_digit m 8 →
  units_digit n 3 := by
  sorry

#check units_digit_of_n

end NUMINAMATH_CALUDE_units_digit_of_n_l830_83078


namespace NUMINAMATH_CALUDE_second_point_x_coordinate_l830_83022

/-- Given two points on a line, prove the x-coordinate of the second point -/
theorem second_point_x_coordinate 
  (m n : ℝ) 
  (h1 : m = 2 * n + 5) 
  (h2 : m + 1 = 2 * (n + 0.5) + 5) : 
  m + 1 = 2 * n + 6 := by
  sorry

end NUMINAMATH_CALUDE_second_point_x_coordinate_l830_83022


namespace NUMINAMATH_CALUDE_power_mod_equivalence_l830_83059

theorem power_mod_equivalence (x : ℤ) (h : x^77 % 7 = 6) : x^5 % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_power_mod_equivalence_l830_83059


namespace NUMINAMATH_CALUDE_negation_of_proposition_l830_83038

theorem negation_of_proposition :
  (∀ a b : ℝ, ab > 0 → a > 0) ↔ (∀ a b : ℝ, ab ≤ 0 → a ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l830_83038


namespace NUMINAMATH_CALUDE_second_chapter_pages_l830_83018

theorem second_chapter_pages (total_chapters : ℕ) (total_pages : ℕ) (second_chapter_length : ℕ) :
  total_chapters = 2 →
  total_pages = 81 →
  second_chapter_length = 68 →
  second_chapter_length = 68 := by
  sorry

end NUMINAMATH_CALUDE_second_chapter_pages_l830_83018


namespace NUMINAMATH_CALUDE_intersection_tangent_line_constant_l830_83063

/-- Given two curves f(x) = √x and g(x) = a ln x that intersect and have the same tangent line
    at the point of intersection, prove that a = e/2 -/
theorem intersection_tangent_line_constant (a : ℝ) :
  (∃ x₀ : ℝ, x₀ > 0 ∧ Real.sqrt x₀ = a * Real.log x₀ ∧ 
    (1 / (2 * Real.sqrt x₀) : ℝ) = a / x₀) →
  a = Real.exp 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_intersection_tangent_line_constant_l830_83063


namespace NUMINAMATH_CALUDE_dividend_divisor_remainder_l830_83044

theorem dividend_divisor_remainder (x y : ℕ+) :
  (x : ℝ) / (y : ℝ) = 96.12 →
  (x : ℝ) % (y : ℝ) = 1.44 →
  y = 12 := by
  sorry

end NUMINAMATH_CALUDE_dividend_divisor_remainder_l830_83044


namespace NUMINAMATH_CALUDE_symmetry_implies_values_and_minimum_l830_83092

/-- A function f(x) that is symmetric about the line x = -1 -/
def symmetric_about_neg_one (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-(x + 2)) = f x

/-- The function f(x) = (x^2 - 4)(x^2 + ax + b) -/
def f (a b : ℝ) (x : ℝ) : ℝ :=
  (x^2 - 4) * (x^2 + a*x + b)

theorem symmetry_implies_values_and_minimum (a b : ℝ) :
  symmetric_about_neg_one (f a b) →
  (a = 4 ∧ b = 0) ∧
  (∃ m : ℝ, m = -16 ∧ ∀ x : ℝ, f a b x ≥ m) :=
by sorry

end NUMINAMATH_CALUDE_symmetry_implies_values_and_minimum_l830_83092


namespace NUMINAMATH_CALUDE_john_incentive_amount_l830_83061

/-- Calculates the incentive amount given to an agent based on commission, advance fees, and amount paid. --/
def calculate_incentive (commission : ℕ) (advance_fees : ℕ) (amount_paid : ℕ) : Int :=
  (commission - advance_fees : Int) - amount_paid

/-- Proves that the incentive amount for John is -1780 Rs, indicating an excess payment. --/
theorem john_incentive_amount :
  let commission : ℕ := 25000
  let advance_fees : ℕ := 8280
  let amount_paid : ℕ := 18500
  calculate_incentive commission advance_fees amount_paid = -1780 := by
  sorry

end NUMINAMATH_CALUDE_john_incentive_amount_l830_83061


namespace NUMINAMATH_CALUDE_jony_stops_at_70_l830_83095

/-- Represents the walking scenario of Jony along Sunrise Boulevard -/
structure WalkingScenario where
  start_block : ℕ
  turn_block : ℕ
  block_length : ℕ
  walking_speed : ℕ
  walking_time : ℕ

/-- Calculates the block where Jony stops walking -/
def stop_block (scenario : WalkingScenario) : ℕ :=
  sorry

/-- Theorem stating that Jony stops at block 70 given the specific scenario -/
theorem jony_stops_at_70 : 
  let scenario : WalkingScenario := {
    start_block := 10,
    turn_block := 90,
    block_length := 40,
    walking_speed := 100,
    walking_time := 40
  }
  stop_block scenario = 70 := by
  sorry

end NUMINAMATH_CALUDE_jony_stops_at_70_l830_83095


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_3s_l830_83091

-- Define the displacement function
def h (t : ℝ) : ℝ := 1.5 * t - 0.1 * t^2

-- Define the velocity function as the derivative of the displacement function
def v (t : ℝ) : ℝ := 1.5 - 0.2 * t

-- Theorem statement
theorem instantaneous_velocity_at_3s : v 3 = 0.9 := by
  sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_3s_l830_83091


namespace NUMINAMATH_CALUDE_fraction_simplification_l830_83050

theorem fraction_simplification (b x : ℝ) (h : b^2 + x^2 ≠ 0) :
  (Real.sqrt (b^2 + x^2) - (x^2 - b^2) / Real.sqrt (b^2 + x^2)) / (2 * (b^2 + x^2)^2) =
  b^2 / (b^2 + x^2)^(5/2) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l830_83050


namespace NUMINAMATH_CALUDE_fence_cost_per_foot_l830_83005

theorem fence_cost_per_foot 
  (plot_area : ℝ) 
  (total_cost : ℝ) 
  (h1 : plot_area = 289) 
  (h2 : total_cost = 3808) : 
  total_cost / (4 * Real.sqrt plot_area) = 56 := by
sorry

end NUMINAMATH_CALUDE_fence_cost_per_foot_l830_83005


namespace NUMINAMATH_CALUDE_steve_has_dimes_l830_83087

/-- Represents the types of coins in US currency --/
inductive USCoin
  | Penny
  | Nickel
  | Dime
  | Quarter

/-- Returns the value of a US coin in cents --/
def coin_value (c : USCoin) : ℕ :=
  match c with
  | USCoin.Penny => 1
  | USCoin.Nickel => 5
  | USCoin.Dime => 10
  | USCoin.Quarter => 25

/-- Theorem: Given the conditions, Steve must have 26 dimes --/
theorem steve_has_dimes (total_coins : ℕ) (total_value : ℕ) (majority_coin_count : ℕ)
    (h_total_coins : total_coins = 36)
    (h_total_value : total_value = 310)
    (h_majority_coin_count : majority_coin_count = 26)
    (h_two_types : ∃ (c1 c2 : USCoin), c1 ≠ c2 ∧
      ∃ (n1 n2 : ℕ), n1 + n2 = total_coins ∧
        n1 * coin_value c1 + n2 * coin_value c2 = total_value ∧
        (n1 = majority_coin_count ∨ n2 = majority_coin_count)) :
    ∃ (other_coin : USCoin), other_coin ≠ USCoin.Dime ∧
      majority_coin_count * coin_value USCoin.Dime +
      (total_coins - majority_coin_count) * coin_value other_coin = total_value :=
  sorry

end NUMINAMATH_CALUDE_steve_has_dimes_l830_83087


namespace NUMINAMATH_CALUDE_constant_term_binomial_expansion_l830_83054

/-- The constant term in the expansion of (x^2 - 2/√x)^5 is 80 -/
theorem constant_term_binomial_expansion :
  (∃ (c : ℝ), c = 80 ∧ 
   ∀ (x : ℝ), x > 0 → 
   ∃ (f : ℝ → ℝ), (λ x => (x^2 - 2/Real.sqrt x)^5) = (λ x => f x + c)) := by
sorry

end NUMINAMATH_CALUDE_constant_term_binomial_expansion_l830_83054
