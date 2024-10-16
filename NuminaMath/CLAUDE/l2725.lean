import Mathlib

namespace NUMINAMATH_CALUDE_loan_income_is_135_l2725_272598

/-- Calculates the yearly annual income from two parts of a loan at different interest rates -/
def yearly_income (total : ℚ) (part1 : ℚ) (rate1 : ℚ) (rate2 : ℚ) : ℚ :=
  let part2 := total - part1
  part1 * rate1 + part2 * rate2

/-- Theorem stating that the yearly income from the given loan parts is 135 -/
theorem loan_income_is_135 :
  yearly_income 2500 1500 (5/100) (6/100) = 135 := by
  sorry

end NUMINAMATH_CALUDE_loan_income_is_135_l2725_272598


namespace NUMINAMATH_CALUDE_prob_even_modified_die_l2725_272503

/-- Represents a standard 6-sided die -/
def StandardDie : Finset ℕ := {1, 2, 3, 4, 5, 6}

/-- Total number of dots on a standard die -/
def TotalDots : ℕ := (StandardDie.sum id)

/-- Probability of a specific dot not being removed in one attempt -/
def ProbNotRemoved : ℚ := (TotalDots - 1) / TotalDots

/-- Probability of a specific dot not being removed in two attempts -/
def ProbNotRemovedTwice : ℚ := ProbNotRemoved ^ 2

/-- Probability of a specific dot being removed in two attempts -/
def ProbRemovedTwice : ℚ := 1 - ProbNotRemovedTwice

/-- Probability of losing exactly one dot from a face with n dots -/
def ProbLoseOneDot (n : ℕ) : ℚ := 2 * (n / TotalDots) * ProbNotRemoved

/-- Probability of a face with n dots remaining even after dot removal -/
def ProbRemainsEven (n : ℕ) : ℚ :=
  if n % 2 = 0
  then ProbNotRemovedTwice + (if n ≥ 2 then ProbRemovedTwice else 0)
  else ProbLoseOneDot n

/-- Theorem: The probability of rolling an even number on the modified die -/
theorem prob_even_modified_die :
  (StandardDie.sum (λ n => (1 : ℚ) / 6 * ProbRemainsEven n)) =
  (StandardDie.sum (λ n => if n % 2 = 0 then (1 : ℚ) / 6 * ProbNotRemovedTwice else 0)) +
  (StandardDie.sum (λ n => if n % 2 = 0 ∧ n ≥ 2 then (1 : ℚ) / 6 * ProbRemovedTwice else 0)) +
  (StandardDie.sum (λ n => if n % 2 = 1 then (1 : ℚ) / 6 * ProbLoseOneDot n else 0)) :=
by sorry


end NUMINAMATH_CALUDE_prob_even_modified_die_l2725_272503


namespace NUMINAMATH_CALUDE_ratio_limit_is_27_l2725_272514

/-- The ratio of the largest element to the sum of other elements in the geometric series -/
def ratio (n : ℕ) : ℚ :=
  let a := 3
  let r := 10
  (a * r^n) / (a * (r^n - 1) / (r - 1))

/-- The limit of the ratio as n approaches infinity is 27 -/
theorem ratio_limit_is_27 : ∀ ε > 0, ∃ N, ∀ n ≥ N, |ratio n - 27| < ε :=
sorry

end NUMINAMATH_CALUDE_ratio_limit_is_27_l2725_272514


namespace NUMINAMATH_CALUDE_range_of_a_l2725_272557

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x - a| + |x - 2| ≥ 1) → a ∈ Set.Iic 1 ∪ Set.Ici 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2725_272557


namespace NUMINAMATH_CALUDE_oil_in_engine_l2725_272512

theorem oil_in_engine (oil_per_cylinder : ℕ) (num_cylinders : ℕ) (additional_oil_needed : ℕ) :
  oil_per_cylinder = 8 →
  num_cylinders = 6 →
  additional_oil_needed = 32 →
  oil_per_cylinder * num_cylinders - additional_oil_needed = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_oil_in_engine_l2725_272512


namespace NUMINAMATH_CALUDE_paint_more_expensive_than_wallpaper_l2725_272517

/-- Proves that a can of paint costs more than a roll of wallpaper given specific purchase scenarios -/
theorem paint_more_expensive_than_wallpaper 
  (wallpaper_cost movie_ticket_cost paint_cost : ℝ) 
  (wallpaper_cost_positive : 0 < wallpaper_cost)
  (paint_cost_positive : 0 < paint_cost)
  (movie_ticket_cost_positive : 0 < movie_ticket_cost)
  (equal_spending : 4 * wallpaper_cost + 4 * paint_cost = 7 * wallpaper_cost + 2 * paint_cost + movie_ticket_cost) :
  paint_cost > wallpaper_cost := by
  sorry


end NUMINAMATH_CALUDE_paint_more_expensive_than_wallpaper_l2725_272517


namespace NUMINAMATH_CALUDE_relative_prime_linear_forms_l2725_272585

theorem relative_prime_linear_forms (a b : ℤ) : 
  ∃ c d : ℤ, ∀ n : ℤ, Int.gcd (a * n + c) (b * n + d) = 1 := by
  sorry

end NUMINAMATH_CALUDE_relative_prime_linear_forms_l2725_272585


namespace NUMINAMATH_CALUDE_remaining_balance_calculation_l2725_272587

/-- Calculates the remaining balance for a product purchase with given conditions -/
theorem remaining_balance_calculation (deposit : ℝ) (deposit_rate : ℝ) (tax_rate : ℝ) (discount_rate : ℝ) (service_charge : ℝ) :
  deposit = 110 →
  deposit_rate = 0.10 →
  tax_rate = 0.15 →
  discount_rate = 0.05 →
  service_charge = 50 →
  ∃ (total_price : ℝ),
    total_price = deposit / deposit_rate ∧
    (total_price * (1 + tax_rate) * (1 - discount_rate) + service_charge - deposit) = 1141.75 := by
  sorry

end NUMINAMATH_CALUDE_remaining_balance_calculation_l2725_272587


namespace NUMINAMATH_CALUDE_percentage_problem_l2725_272545

theorem percentage_problem (N : ℝ) (P : ℝ) 
  (h1 : 0.8 * N = 240) 
  (h2 : (P / 100) * N = 60) : 
  P = 20 := by
sorry

end NUMINAMATH_CALUDE_percentage_problem_l2725_272545


namespace NUMINAMATH_CALUDE_count_of_satisfying_integers_l2725_272540

/-- The number of integers satisfying the equation -/
def solution_count : ℕ := 40200

/-- The equation to be satisfied -/
def satisfies_equation (n : ℤ) : Prop :=
  1 + ⌊(200 * n) / 201⌋ = ⌈(198 * n) / 200⌉

theorem count_of_satisfying_integers :
  (∃! (s : Finset ℤ), s.card = solution_count ∧ ∀ n, n ∈ s ↔ satisfies_equation n) :=
sorry

end NUMINAMATH_CALUDE_count_of_satisfying_integers_l2725_272540


namespace NUMINAMATH_CALUDE_quadratic_roots_integrality_l2725_272595

/-- Given two quadratic equations x^2 - px + q = 0 and x^2 - (p+1)x + q = 0,
    this theorem states that when q > 0, both equations can have integer roots,
    but when q < 0, they cannot both have integer roots simultaneously. -/
theorem quadratic_roots_integrality (p q : ℤ) :
  (q > 0 → ∃ (x₁ x₂ x₃ x₄ : ℤ),
    x₁^2 - p*x₁ + q = 0 ∧
    x₂^2 - p*x₂ + q = 0 ∧
    x₃^2 - (p+1)*x₃ + q = 0 ∧
    x₄^2 - (p+1)*x₄ + q = 0) ∧
  (q < 0 → ¬∃ (x₁ x₂ x₃ x₄ : ℤ),
    x₁^2 - p*x₁ + q = 0 ∧
    x₂^2 - p*x₂ + q = 0 ∧
    x₃^2 - (p+1)*x₃ + q = 0 ∧
    x₄^2 - (p+1)*x₄ + q = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_integrality_l2725_272595


namespace NUMINAMATH_CALUDE_day_after_53_from_monday_is_friday_l2725_272533

/-- Days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday
  | DayOfWeek.Sunday => DayOfWeek.Monday

/-- Function to get the day of the week after a given number of days -/
def dayAfter (start : DayOfWeek) (days : Nat) : DayOfWeek :=
  match days with
  | 0 => start
  | n + 1 => nextDay (dayAfter start n)

theorem day_after_53_from_monday_is_friday :
  dayAfter DayOfWeek.Monday 53 = DayOfWeek.Friday := by
  sorry

end NUMINAMATH_CALUDE_day_after_53_from_monday_is_friday_l2725_272533


namespace NUMINAMATH_CALUDE_absent_percentage_l2725_272530

def total_students : ℕ := 100
def present_students : ℕ := 86

theorem absent_percentage : 
  (total_students - present_students) * 100 / total_students = 14 := by
  sorry

end NUMINAMATH_CALUDE_absent_percentage_l2725_272530


namespace NUMINAMATH_CALUDE_main_rectangle_tiled_by_tetraminoes_l2725_272591

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents a tetramino (2 × 3 rectangle with two opposite corners removed) -/
def Tetramino : Rectangle :=
  { width := 2, height := 3 }

/-- The area of a rectangle -/
def area (r : Rectangle) : ℕ :=
  r.width * r.height

/-- The area of a tetramino -/
def tetraminoArea : ℕ :=
  area Tetramino - 2

/-- The main rectangle to be tiled -/
def mainRectangle : Rectangle :=
  { width := 2008, height := 2010 }

/-- Theorem: The main rectangle can be tiled using only tetraminoes -/
theorem main_rectangle_tiled_by_tetraminoes :
  ∃ (n : ℕ), n * tetraminoArea = area mainRectangle :=
sorry

end NUMINAMATH_CALUDE_main_rectangle_tiled_by_tetraminoes_l2725_272591


namespace NUMINAMATH_CALUDE_polynomial_remainder_l2725_272583

/-- Given a polynomial p(x) with the specified division properties, 
    prove that its remainder when divided by (x + 1)(x + 3) is (7/2)x + 13/2 -/
theorem polynomial_remainder (p : Polynomial ℚ) 
  (h1 : (p - 3).eval (-1) = 0)
  (h2 : (p + 4).eval (-3) = 0) :
  ∃ q : Polynomial ℚ, p = q * ((X + 1) * (X + 3)) + (7/2 * X + 13/2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l2725_272583


namespace NUMINAMATH_CALUDE_no_conclusive_deduction_l2725_272531

-- Define the universe of discourse
variable (U : Type)

-- Define predicates for Bars, Fins, and Grips
variable (Bar Fin Grip : U → Prop)

-- Define the given conditions
variable (some_bars_not_fins : ∃ x, Bar x ∧ ¬Fin x)
variable (no_fins_are_grips : ∀ x, Fin x → ¬Grip x)

-- Define the statements to be proved
def some_bars_not_grips := ∃ x, Bar x ∧ ¬Grip x
def some_grips_not_bars := ∃ x, Grip x ∧ ¬Bar x
def no_bar_is_grip := ∀ x, Bar x → ¬Grip x
def some_bars_are_grips := ∃ x, Bar x ∧ Grip x

-- Theorem stating that none of the above statements can be conclusively deduced
theorem no_conclusive_deduction :
  ¬(some_bars_not_grips U Bar Grip ∨
     some_grips_not_bars U Grip Bar ∨
     no_bar_is_grip U Bar Grip ∨
     some_bars_are_grips U Bar Grip) :=
sorry

end NUMINAMATH_CALUDE_no_conclusive_deduction_l2725_272531


namespace NUMINAMATH_CALUDE_bus_speed_excluding_stoppages_l2725_272563

/-- Given a bus that stops for 24 minutes per hour and has a speed of 45 kmph including stoppages,
    its speed excluding stoppages is 75 kmph. -/
theorem bus_speed_excluding_stoppages 
  (stop_time : ℝ) 
  (speed_with_stops : ℝ) 
  (h1 : stop_time = 24)
  (h2 : speed_with_stops = 45) : 
  speed_with_stops * (60 / (60 - stop_time)) = 75 := by
  sorry


end NUMINAMATH_CALUDE_bus_speed_excluding_stoppages_l2725_272563


namespace NUMINAMATH_CALUDE_probability_next_queen_after_first_l2725_272578

/-- Represents a standard deck of 54 playing cards -/
def StandardDeck : ℕ := 54

/-- Number of queens in a standard deck -/
def QueenCount : ℕ := 4

/-- Probability of drawing a queen after the first queen -/
def ProbabilityNextQueenAfterFirst : ℚ := 2 / 27

/-- Theorem stating the probability of drawing a queen after the first queen -/
theorem probability_next_queen_after_first :
  ProbabilityNextQueenAfterFirst = QueenCount / StandardDeck :=
by
  sorry


end NUMINAMATH_CALUDE_probability_next_queen_after_first_l2725_272578


namespace NUMINAMATH_CALUDE_regression_and_probability_theorem_l2725_272581

/-- Data point representing year and sales volume -/
structure DataPoint where
  year : ℕ
  sales : ℕ

/-- Linear regression coefficients -/
structure RegressionCoefficients where
  b : ℚ
  a : ℚ

def data : List DataPoint := [
  ⟨1, 5⟩, ⟨2, 5⟩, ⟨3, 6⟩, ⟨4, 7⟩, ⟨5, 7⟩
]

def calculateRegressionCoefficients (data : List DataPoint) : RegressionCoefficients :=
  sorry

def probabilityConsecutiveYears (data : List DataPoint) : ℚ :=
  sorry

theorem regression_and_probability_theorem :
  let coeffs := calculateRegressionCoefficients data
  coeffs.b = 3/5 ∧ coeffs.a = 21/5 ∧ probabilityConsecutiveYears data = 2/5 := by
  sorry

#check regression_and_probability_theorem

end NUMINAMATH_CALUDE_regression_and_probability_theorem_l2725_272581


namespace NUMINAMATH_CALUDE_quadratic_root_in_interval_l2725_272593

theorem quadratic_root_in_interval
  (a b c : ℝ)
  (h_roots : ∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0)
  (h_ineq : |a*b - a*c| > |b^2 - a*c| + |a*b - c^2|)
  : ∃! x : ℝ, 0 < x ∧ x < 2 ∧ a * x^2 + b * x + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_in_interval_l2725_272593


namespace NUMINAMATH_CALUDE_exponent_properties_l2725_272555

theorem exponent_properties (a x y : ℝ) (h1 : a^x = 3) (h2 : a^y = 2) :
  a^(x - y) = 3/2 ∧ a^(2*x + y) = 18 := by
  sorry

end NUMINAMATH_CALUDE_exponent_properties_l2725_272555


namespace NUMINAMATH_CALUDE_hyperbola_theorem_l2725_272560

/-- The standard form of a hyperbola with center at the origin --/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- Check if a point (x, y) is on the hyperbola --/
def Hyperbola.contains (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / h.a^2 - y^2 / h.b^2 = 1

/-- Check if two hyperbolas have the same asymptotes --/
def same_asymptotes (h1 h2 : Hyperbola) : Prop :=
  h1.a^2 / h1.b^2 = h2.a^2 / h2.b^2

theorem hyperbola_theorem (h1 h2 : Hyperbola) :
  h1.a^2 = 3 ∧ h1.b^2 = 12 ∧
  h2.a^2 = 1 ∧ h2.b^2 = 4 →
  same_asymptotes h1 h2 ∧ h1.contains 2 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_theorem_l2725_272560


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l2725_272509

theorem simplify_and_evaluate (a : ℚ) (h : a = -1/2) :
  (1 + a) * (1 - a) - a * (2 - a) = 2 := by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l2725_272509


namespace NUMINAMATH_CALUDE_average_study_time_difference_l2725_272504

/-- The differences in study times (Mia - Liam) for each day of the week --/
def study_time_differences : List Int := [15, -5, 25, 0, -15, 20, 10]

/-- The number of days in a week --/
def days_in_week : Nat := 7

/-- Theorem: The average difference in study time per day is 7 minutes --/
theorem average_study_time_difference :
  (study_time_differences.sum : ℚ) / days_in_week = 7 := by
  sorry

end NUMINAMATH_CALUDE_average_study_time_difference_l2725_272504


namespace NUMINAMATH_CALUDE_green_peaches_count_l2725_272597

theorem green_peaches_count (red_peaches : ℕ) (green_peaches : ℕ) 
  (h1 : red_peaches = 17)
  (h2 : red_peaches = green_peaches + 1) : 
  green_peaches = 16 := by
  sorry

end NUMINAMATH_CALUDE_green_peaches_count_l2725_272597


namespace NUMINAMATH_CALUDE_coefficients_of_given_equation_l2725_272515

/-- Given a quadratic equation ax^2 + bx + c = 0, this function returns a tuple of its coefficients (a, b, c) -/
def quadraticCoefficients (a b c : ℝ) : ℝ × ℝ × ℝ := (a, b, c)

/-- The quadratic equation 5x^2 + 2x - 1 = 0 -/
def givenEquation : ℝ × ℝ × ℝ := (5, 2, -1)

theorem coefficients_of_given_equation :
  quadraticCoefficients 5 2 (-1) = givenEquation :=
by sorry

end NUMINAMATH_CALUDE_coefficients_of_given_equation_l2725_272515


namespace NUMINAMATH_CALUDE_no_relationship_between_running_and_age_probability_of_one_not_interested_l2725_272535

-- Define the contingency table
def contingency_table : Matrix (Fin 2) (Fin 2) ℕ := !![15, 20; 10, 15]

-- Define the total sample size
def n : ℕ := 60

-- Define the K² formula
def K_squared (a b c d : ℕ) : ℚ :=
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Define the critical value for 90% confidence level
def critical_value : ℚ := 2.706

-- Theorem for part 1
theorem no_relationship_between_running_and_age :
  K_squared 15 20 10 15 < critical_value :=
sorry

-- Theorem for part 2
theorem probability_of_one_not_interested :
  (Nat.choose 5 3 - Nat.choose 3 3) / Nat.choose 5 3 = 3 / 5 :=
sorry

end NUMINAMATH_CALUDE_no_relationship_between_running_and_age_probability_of_one_not_interested_l2725_272535


namespace NUMINAMATH_CALUDE_some_value_is_zero_l2725_272526

theorem some_value_is_zero (x y w : ℝ) (some_value : ℝ) 
  (h1 : some_value + 3 / x = 3 / y)
  (h2 : w * x = y)
  (h3 : (w + x) / 2 = 1 / 2) :
  some_value = 0 := by
sorry

end NUMINAMATH_CALUDE_some_value_is_zero_l2725_272526


namespace NUMINAMATH_CALUDE_sandys_hourly_wage_l2725_272580

theorem sandys_hourly_wage (hours_friday hours_saturday hours_sunday : ℕ) 
  (total_earnings : ℕ) (hourly_wage : ℚ) :
  hours_friday = 10 →
  hours_saturday = 6 →
  hours_sunday = 14 →
  total_earnings = 450 →
  hourly_wage * (hours_friday + hours_saturday + hours_sunday) = total_earnings →
  hourly_wage = 15 := by
sorry

end NUMINAMATH_CALUDE_sandys_hourly_wage_l2725_272580


namespace NUMINAMATH_CALUDE_trajectory_equation_l2725_272594

/-- The trajectory of a point M(x,y) such that its distance to the line x = 4 
    is twice its distance to the point (1,0) -/
def trajectory (x y : ℝ) : Prop :=
  (x - 4)^2 = ((x - 1)^2 + y^2) / 4

/-- The equation of the trajectory -/
theorem trajectory_equation (x y : ℝ) :
  trajectory x y ↔ 3 * x^2 + 30 * x - y^2 - 63 = 0 := by
  sorry

end NUMINAMATH_CALUDE_trajectory_equation_l2725_272594


namespace NUMINAMATH_CALUDE_digits_until_2014_l2725_272541

def odd_sequence (n : ℕ) : ℕ := 2 * n - 1

def digit_count (n : ℕ) : ℕ :=
  if n < 10 then 1
  else if n < 100 then 2
  else if n < 1000 then 3
  else 4

def total_digits (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ i => digit_count (odd_sequence (i + 1)))

theorem digits_until_2014 :
  ∃ n : ℕ, odd_sequence n > 2014 ∧ total_digits (n - 1) = 7850 := by sorry

end NUMINAMATH_CALUDE_digits_until_2014_l2725_272541


namespace NUMINAMATH_CALUDE_ten_player_tournament_matches_l2725_272588

/-- The number of matches in a round-robin tournament. -/
def num_matches (n : ℕ) : ℕ := n.choose 2

/-- Theorem: In a 10-player round-robin tournament, there are 45 matches. -/
theorem ten_player_tournament_matches : num_matches 10 = 45 := by
  sorry

end NUMINAMATH_CALUDE_ten_player_tournament_matches_l2725_272588


namespace NUMINAMATH_CALUDE_volleyball_team_combinations_l2725_272542

theorem volleyball_team_combinations : Nat.choose 16 7 = 11440 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_team_combinations_l2725_272542


namespace NUMINAMATH_CALUDE_min_value_expression_l2725_272520

theorem min_value_expression (x y : ℕ) (hx : x > 0) (hy : y > 0) (hxy : x ≠ y) :
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ a ≠ b ∧
  (a + b^2) * (a^2 - b) / (a * b) = 14 ∧
  ∀ (p q : ℕ), p > 0 → q > 0 → p ≠ q →
    (p + q^2) * (p^2 - q) / (p * q) ≥ 14 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2725_272520


namespace NUMINAMATH_CALUDE_x_varies_as_four_thirds_power_of_z_l2725_272518

/-- If x varies as the fourth power of y, and y varies as the cube root of z,
    then x varies as the (4/3)th power of z. -/
theorem x_varies_as_four_thirds_power_of_z 
  (x y z : ℝ) 
  (hxy : ∃ (a : ℝ), x = a * y^4) 
  (hyz : ∃ (b : ℝ), y = b * z^(1/3)) :
  ∃ (c : ℝ), x = c * z^(4/3) := by
sorry

end NUMINAMATH_CALUDE_x_varies_as_four_thirds_power_of_z_l2725_272518


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2725_272508

-- Define the sets M and N
def M : Set ℝ := {x | x - 2 < 0}
def N : Set ℝ := {x | |x - 1| < 2}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | -1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2725_272508


namespace NUMINAMATH_CALUDE_sequence_average_bound_l2725_272506

theorem sequence_average_bound (n : ℕ) (a : ℕ → ℝ) 
  (h1 : a 1 = 0)
  (h2 : ∀ k ∈ Finset.range n, k > 1 → |a k| = |a (k-1) + 1|) :
  (Finset.sum (Finset.range n) (λ i => a (i+1))) / n ≥ -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sequence_average_bound_l2725_272506


namespace NUMINAMATH_CALUDE_min_projection_value_l2725_272544

theorem min_projection_value (a b : ℝ × ℝ) : 
  let norm_a := Real.sqrt ((a.1 ^ 2) + (a.2 ^ 2))
  let norm_b := Real.sqrt ((b.1 ^ 2) + (b.2 ^ 2))
  let dot_product := a.1 * b.1 + a.2 * b.2
  let cos_theta := dot_product / (norm_a * norm_b)
  norm_a = Real.sqrt 6 ∧ 
  ((a.1 + 2 * b.1) ^ 2 + (a.2 + 2 * b.2) ^ 2) = ((3 * a.1 - 4 * b.1) ^ 2 + (3 * a.2 - 4 * b.2) ^ 2) →
  ∃ (min_value : ℝ), min_value = 12 / 7 ∧ ∀ θ : ℝ, norm_a * |cos_theta| ≥ min_value := by
sorry

end NUMINAMATH_CALUDE_min_projection_value_l2725_272544


namespace NUMINAMATH_CALUDE_savings_after_four_weeks_l2725_272501

/-- Calculates the total savings after a given number of weeks, 
    with an initial saving amount and a fixed weekly increase. -/
def totalSavings (initialSaving : ℕ) (weeklyIncrease : ℕ) (weeks : ℕ) : ℕ :=
  initialSaving + weeklyIncrease * (weeks - 1)

/-- Theorem: Given an initial saving of $20 and a weekly increase of $10,
    the total savings after 4 weeks is $60. -/
theorem savings_after_four_weeks :
  totalSavings 20 10 4 = 60 := by
  sorry

end NUMINAMATH_CALUDE_savings_after_four_weeks_l2725_272501


namespace NUMINAMATH_CALUDE_currency_notes_problem_l2725_272502

theorem currency_notes_problem :
  ∃ (D : ℕ+) (x y : ℕ),
    x + y = 100 ∧
    70 * x + D * y = 5000 :=
by sorry

end NUMINAMATH_CALUDE_currency_notes_problem_l2725_272502


namespace NUMINAMATH_CALUDE_three_non_adjacent_from_ten_l2725_272552

/-- The number of ways to choose 3 non-adjacent items from a set of 10 items. -/
def non_adjacent_choices (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose (n - k + 1) k

/-- Theorem: There are 56 ways to choose 3 non-adjacent items from a set of 10 items. -/
theorem three_non_adjacent_from_ten : non_adjacent_choices 10 3 = 56 := by
  sorry

end NUMINAMATH_CALUDE_three_non_adjacent_from_ten_l2725_272552


namespace NUMINAMATH_CALUDE_teams_bc_work_time_l2725_272592

-- Define the workload of projects
def project_a_workload : ℝ := 1
def project_b_workload : ℝ := 1.25

-- Define the time it takes for each team to complete Project A
def team_a_time : ℝ := 20
def team_b_time : ℝ := 24
def team_c_time : ℝ := 30

-- Define variables for the unknown times
def time_bc_together : ℝ := 15
def time_c_with_a : ℝ := 20 -- This is not given, but we need it for the theorem

theorem teams_bc_work_time :
  (time_bc_together / team_b_time + time_bc_together / team_c_time + time_c_with_a / team_b_time = project_b_workload) ∧
  (time_bc_together / team_a_time + time_c_with_a / team_c_time + time_c_with_a / team_a_time = project_a_workload) :=
by sorry

end NUMINAMATH_CALUDE_teams_bc_work_time_l2725_272592


namespace NUMINAMATH_CALUDE_arman_earnings_l2725_272529

/-- Represents the pay rates and working hours for Arman over two weeks -/
structure PayData :=
  (last_week_rate : ℝ)
  (last_week_hours : ℝ)
  (this_week_rate_increase : ℝ)
  (overtime_multiplier : ℝ)
  (weekend_multiplier : ℝ)
  (night_shift_multiplier : ℝ)
  (monday_hours : ℝ)
  (monday_night_hours : ℝ)
  (tuesday_hours : ℝ)
  (tuesday_night_hours : ℝ)
  (wednesday_hours : ℝ)
  (thursday_hours : ℝ)
  (thursday_night_hours : ℝ)
  (thursday_overtime : ℝ)
  (friday_hours : ℝ)
  (saturday_hours : ℝ)
  (sunday_hours : ℝ)
  (sunday_night_hours : ℝ)

/-- Calculates the total earnings for Arman over two weeks -/
def calculate_earnings (data : PayData) : ℝ :=
  let last_week_earnings := data.last_week_rate * data.last_week_hours
  let this_week_rate := data.last_week_rate + data.this_week_rate_increase
  let this_week_earnings :=
    (data.monday_hours - data.monday_night_hours) * this_week_rate +
    data.monday_night_hours * this_week_rate * data.night_shift_multiplier +
    (data.tuesday_hours - data.tuesday_night_hours) * this_week_rate +
    data.tuesday_night_hours * this_week_rate * data.night_shift_multiplier +
    (data.tuesday_hours - 8) * this_week_rate * data.overtime_multiplier +
    data.wednesday_hours * this_week_rate +
    (data.thursday_hours - data.thursday_night_hours - data.thursday_overtime) * this_week_rate +
    data.thursday_night_hours * this_week_rate * data.night_shift_multiplier +
    data.thursday_overtime * this_week_rate * data.overtime_multiplier +
    data.friday_hours * this_week_rate +
    data.saturday_hours * this_week_rate * data.weekend_multiplier +
    (data.sunday_hours - data.sunday_night_hours) * this_week_rate * data.weekend_multiplier +
    data.sunday_night_hours * this_week_rate * data.weekend_multiplier * data.night_shift_multiplier
  last_week_earnings + this_week_earnings

/-- Theorem stating that Arman's total earnings for the two weeks equal $1055.46 -/
theorem arman_earnings (data : PayData)
  (h1 : data.last_week_rate = 10)
  (h2 : data.last_week_hours = 35)
  (h3 : data.this_week_rate_increase = 0.5)
  (h4 : data.overtime_multiplier = 1.5)
  (h5 : data.weekend_multiplier = 1.7)
  (h6 : data.night_shift_multiplier = 1.3)
  (h7 : data.monday_hours = 8)
  (h8 : data.monday_night_hours = 3)
  (h9 : data.tuesday_hours = 10)
  (h10 : data.tuesday_night_hours = 4)
  (h11 : data.wednesday_hours = 8)
  (h12 : data.thursday_hours = 9)
  (h13 : data.thursday_night_hours = 3)
  (h14 : data.thursday_overtime = 1)
  (h15 : data.friday_hours = 5)
  (h16 : data.saturday_hours = 6)
  (h17 : data.sunday_hours = 4)
  (h18 : data.sunday_night_hours = 2) :
  calculate_earnings data = 1055.46 := by sorry

end NUMINAMATH_CALUDE_arman_earnings_l2725_272529


namespace NUMINAMATH_CALUDE_expression_proof_l2725_272548

theorem expression_proof (x₁ x₂ : ℝ) (E : ℝ → ℝ) :
  (∀ x, (x + 3)^2 / (E x) = 2) →
  x₁ - x₂ = 14 →
  ∃ x, E x = (x + 3)^2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_expression_proof_l2725_272548


namespace NUMINAMATH_CALUDE_polynomial_sum_l2725_272525

-- Define the polynomials
def f (x : ℝ) : ℝ := -3 * x^3 - 3 * x^2 + x - 5
def g (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def h (x : ℝ) : ℝ := 5 * x^2 + 6 * x + 2

-- State the theorem
theorem polynomial_sum (x : ℝ) : f x + g x + h x = -3 * x^3 - 4 * x^2 + 11 * x - 12 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_sum_l2725_272525


namespace NUMINAMATH_CALUDE_square_sum_given_diff_and_product_l2725_272547

theorem square_sum_given_diff_and_product (x y : ℝ) 
  (h1 : x - y = 12) 
  (h2 : x * y = 9) : 
  x^2 + y^2 = 162 := by
sorry

end NUMINAMATH_CALUDE_square_sum_given_diff_and_product_l2725_272547


namespace NUMINAMATH_CALUDE_complement_intersection_equal_set_l2725_272561

def U : Set Int := {0, -1, -2, -3, -4}
def M : Set Int := {0, -1, -2}
def N : Set Int := {0, -3, -4}

theorem complement_intersection_equal_set : (U \ M) ∩ N = {-3, -4} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_equal_set_l2725_272561


namespace NUMINAMATH_CALUDE_expression_evaluation_l2725_272575

theorem expression_evaluation : (3^2 - 3) - (4^2 - 4) + (5^2 - 5) - (6^2 - 6) = -16 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2725_272575


namespace NUMINAMATH_CALUDE_polynomial_remainder_l2725_272527

theorem polynomial_remainder (x : ℝ) : 
  (4*x^3 - 9*x^2 + 12*x - 14) % (2*x - 4) = 6 := by sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l2725_272527


namespace NUMINAMATH_CALUDE_quadratic_roots_theorem_l2725_272556

/-- Represents a continued fraction with repeating terms a and b -/
def RepeatingContinuedFraction (a b : ℤ) : ℝ :=
  sorry

/-- The other root of a quadratic equation with integer coefficients -/
def OtherRoot (a b : ℤ) : ℝ :=
  sorry

theorem quadratic_roots_theorem (a b : ℤ) :
  ∃ (p q r : ℤ), 
    (p * (RepeatingContinuedFraction a b)^2 + q * (RepeatingContinuedFraction a b) + r = 0) →
    (OtherRoot a b = -1 / (RepeatingContinuedFraction b a)) :=
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_theorem_l2725_272556


namespace NUMINAMATH_CALUDE_right_triangle_leg_square_l2725_272550

/-- In a right triangle, if the hypotenuse c is 2 more than one leg a,
    then the square of the other leg b is equal to 4a + 4 -/
theorem right_triangle_leg_square (a c : ℝ) (h1 : c = a + 2) :
  ∃ b : ℝ, a^2 + b^2 = c^2 ∧ b^2 = 4*a + 4 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_leg_square_l2725_272550


namespace NUMINAMATH_CALUDE_complete_square_d_value_l2725_272599

/-- Given a quadratic equation x^2 - 6x + 5 = 0, prove that when converted to the form (x + c)^2 = d, the value of d is 4 -/
theorem complete_square_d_value (x : ℝ) : 
  (x^2 - 6*x + 5 = 0) → 
  (∃ c d : ℝ, (x + c)^2 = d ∧ x^2 - 6*x + 5 = 0) →
  (∃ c : ℝ, (x + c)^2 = 4 ∧ x^2 - 6*x + 5 = 0) :=
by sorry


end NUMINAMATH_CALUDE_complete_square_d_value_l2725_272599


namespace NUMINAMATH_CALUDE_polynomial_roots_product_l2725_272574

theorem polynomial_roots_product (b c : ℤ) : 
  (∀ x : ℝ, x^2 - 2*x - 1 = 0 → x^6 - b*x - c = 0) → 
  b * c = 2030 := by
sorry

end NUMINAMATH_CALUDE_polynomial_roots_product_l2725_272574


namespace NUMINAMATH_CALUDE_triangle_identities_l2725_272576

/-- Given a triangle with sides a, b, c and angles α, β, γ, 
    where the Law of Sines holds, prove the given equations. -/
theorem triangle_identities 
  (a b c α β γ : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_angles : 0 < α ∧ α < π ∧ 0 < β ∧ β < π ∧ 0 < γ ∧ γ < π)
  (h_sum : α + β + γ = π)
  (h_law_of_sines : a / Real.sin α = b / Real.sin β ∧ b / Real.sin β = c / Real.sin γ) :
  (a + b) / c = Real.cos ((α - β) / 2) / Real.sin (γ / 2) ∧ 
  (a - b) / c = Real.sin ((α - β) / 2) / Real.cos (γ / 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_identities_l2725_272576


namespace NUMINAMATH_CALUDE_notebook_cost_l2725_272519

/-- Given the cost of items and the total spent, prove the cost of each notebook -/
theorem notebook_cost 
  (pen_cost : ℕ) 
  (folder_cost : ℕ) 
  (num_pens : ℕ) 
  (num_notebooks : ℕ) 
  (num_folders : ℕ) 
  (total_spent : ℕ) 
  (h1 : pen_cost = 1) 
  (h2 : folder_cost = 5) 
  (h3 : num_pens = 3) 
  (h4 : num_notebooks = 4) 
  (h5 : num_folders = 2) 
  (h6 : total_spent = 25) : 
  (total_spent - num_pens * pen_cost - num_folders * folder_cost) / num_notebooks = 3 := by
  sorry

end NUMINAMATH_CALUDE_notebook_cost_l2725_272519


namespace NUMINAMATH_CALUDE_quadratic_form_constant_l2725_272565

theorem quadratic_form_constant (a h k : ℚ) : 
  (∀ x, x^2 - 7*x = a*(x - h)^2 + k) → k = -49/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_constant_l2725_272565


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l2725_272511

-- Problem 1
theorem simplify_expression_1 (x : ℝ) : 6 * (2 * x - 1) - 3 * (5 + 2 * x) = 6 * x - 21 := by
  sorry

-- Problem 2
theorem simplify_expression_2 (a : ℝ) : (4 * a^2 - 8 * a - 9) + 3 * (2 * a^2 - 2 * a - 5) = 10 * a^2 - 14 * a - 24 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l2725_272511


namespace NUMINAMATH_CALUDE_team_selection_count_l2725_272551

def boys : ℕ := 7
def girls : ℕ := 9
def team_size : ℕ := 7
def boys_in_team : ℕ := 4
def girls_in_team : ℕ := 3

theorem team_selection_count :
  (Nat.choose boys boys_in_team) * (Nat.choose girls girls_in_team) = 2940 := by
  sorry

end NUMINAMATH_CALUDE_team_selection_count_l2725_272551


namespace NUMINAMATH_CALUDE_jerry_weekly_spending_jerry_specific_case_l2725_272528

/-- Given Jerry's earnings and the duration the money lasted, calculate his weekly spending. -/
theorem jerry_weekly_spending (lawn_money weed_money : ℕ) (weeks : ℕ) : 
  (lawn_money + weed_money) / weeks = (lawn_money + weed_money) / weeks :=
by sorry

/-- Jerry's specific case -/
theorem jerry_specific_case : 
  (14 + 31) / 9 = 5 :=
by sorry

end NUMINAMATH_CALUDE_jerry_weekly_spending_jerry_specific_case_l2725_272528


namespace NUMINAMATH_CALUDE_certain_number_exists_and_unique_l2725_272570

theorem certain_number_exists_and_unique : 
  ∃! x : ℕ, 220050 = (x + 445) * (2 * (x - 445)) + 50 :=
sorry

end NUMINAMATH_CALUDE_certain_number_exists_and_unique_l2725_272570


namespace NUMINAMATH_CALUDE_cheaper_call_rate_l2725_272566

/-- China Mobile's promotion factor -/
def china_mobile_promotion : ℚ := 130 / 100

/-- China Telecom's promotion factor -/
def china_telecom_promotion : ℚ := 100 / 40

/-- China Mobile's standard call rate (yuan per minute) -/
def china_mobile_standard_rate : ℚ := 26 / 100

/-- China Telecom's standard call rate (yuan per minute) -/
def china_telecom_standard_rate : ℚ := 30 / 100

/-- China Mobile's actual call rate (yuan per minute) -/
def china_mobile_actual_rate : ℚ := china_mobile_standard_rate / china_mobile_promotion

/-- China Telecom's actual call rate (yuan per minute) -/
def china_telecom_actual_rate : ℚ := china_telecom_standard_rate / china_telecom_promotion

theorem cheaper_call_rate :
  china_telecom_actual_rate < china_mobile_actual_rate ∧
  china_mobile_actual_rate - china_telecom_actual_rate = 8 / 100 := by
  sorry

end NUMINAMATH_CALUDE_cheaper_call_rate_l2725_272566


namespace NUMINAMATH_CALUDE_root_sum_equals_three_l2725_272573

theorem root_sum_equals_three (x₁ x₂ : ℝ) 
  (h₁ : x₁ + Real.log x₁ = 3) 
  (h₂ : x₂ + (10 : ℝ) ^ x₂ = 3) : 
  x₁ + x₂ = 3 := by sorry

end NUMINAMATH_CALUDE_root_sum_equals_three_l2725_272573


namespace NUMINAMATH_CALUDE_expand_polynomial_l2725_272586

theorem expand_polynomial (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := by
  sorry

end NUMINAMATH_CALUDE_expand_polynomial_l2725_272586


namespace NUMINAMATH_CALUDE_arithmetic_harmonic_means_equal_implies_equal_values_l2725_272537

theorem arithmetic_harmonic_means_equal_implies_equal_values (a b : ℝ) 
  (h_arithmetic : (a + b) / 2 = 2) 
  (h_harmonic : 2 / (1/a + 1/b) = 2) : 
  a = 2 ∧ b = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_harmonic_means_equal_implies_equal_values_l2725_272537


namespace NUMINAMATH_CALUDE_length_of_PT_l2725_272584

/-- Given points P, Q, R, S, and T in a coordinate plane where PQ intersects RS at T,
    and the x-coordinate difference between P and Q is 6,
    and the y-coordinate difference between P and Q is 4,
    prove that the length of segment PT is 12√13/11 -/
theorem length_of_PT (P Q R S T : ℝ × ℝ) : 
  (∃ t : ℝ, 0 < t ∧ t < 1 ∧ 
    T = (1 - t) • P + t • Q ∧
    T = (1 - t) • R + t • S) →
  Q.1 - P.1 = 6 →
  Q.2 - P.2 = 4 →
  Real.sqrt ((T.1 - P.1)^2 + (T.2 - P.2)^2) = 12 * Real.sqrt 13 / 11 :=
by sorry

end NUMINAMATH_CALUDE_length_of_PT_l2725_272584


namespace NUMINAMATH_CALUDE_first_fabulous_friday_is_oct31_l2725_272571

/-- Represents a date with a day, month, and day of the week -/
structure Date where
  day : Nat
  month : Nat
  dayOfWeek : Nat
  deriving Repr

/-- Represents a school calendar -/
structure SchoolCalendar where
  startDate : Date
  deriving Repr

/-- Determines if a given date is a Fabulous Friday -/
def isFabulousFriday (d : Date) : Bool :=
  sorry

/-- Finds the first Fabulous Friday after the school start date -/
def firstFabulousFriday (sc : SchoolCalendar) : Date :=
  sorry

/-- Theorem stating that the first Fabulous Friday after school starts on Tuesday, October 3 is October 31 -/
theorem first_fabulous_friday_is_oct31 (sc : SchoolCalendar) :
  sc.startDate = Date.mk 3 10 2 →  -- October 3 is a Tuesday (day 2 of the week)
  firstFabulousFriday sc = Date.mk 31 10 5 :=  -- October 31 is a Friday (day 5 of the week)
  sorry

end NUMINAMATH_CALUDE_first_fabulous_friday_is_oct31_l2725_272571


namespace NUMINAMATH_CALUDE_min_value_expression_l2725_272505

theorem min_value_expression (a b c : ℝ) 
  (h1 : 2 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ 5) :
  (a - 2)^2 + (b/a - 1)^2 + (c/b - 1)^2 + (5/c - 1)^2 ≥ 4 * (5^(1/4) - 1/2)^2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2725_272505


namespace NUMINAMATH_CALUDE_polynomial_with_positive_integer_roots_l2725_272546

theorem polynomial_with_positive_integer_roots :
  ∀ (a b c : ℝ),
  (∃ (p q r s : ℕ+),
    (∀ x : ℝ, x^4 + a*x^3 + b*x^2 + c*x + b = (x - p)*(x - q)*(x - r)*(x - s)) ∧
    p + q + r + s = -a ∧
    p*q + p*r + p*s + q*r + q*s + r*s = b ∧
    p*q*r + p*q*s + p*r*s + q*r*s = -c ∧
    p*q*r*s = b) →
  ((a = -21 ∧ b = 112 ∧ c = -204) ∨ (a = -12 ∧ b = 48 ∧ c = -80)) := by
sorry

end NUMINAMATH_CALUDE_polynomial_with_positive_integer_roots_l2725_272546


namespace NUMINAMATH_CALUDE_equal_costs_l2725_272523

/-- Represents the number of bookcases to be purchased -/
def num_bookcases : ℕ := 20

/-- Represents the number of bookshelves to be purchased -/
def num_bookshelves : ℕ := 40

/-- Cost of a single bookcase in dollars -/
def bookcase_cost : ℕ := 300

/-- Cost of a single bookshelf in dollars -/
def bookshelf_cost : ℕ := 100

/-- Calculates the total cost at supermarket A -/
def cost_A : ℕ := num_bookcases * bookcase_cost + bookshelf_cost * (num_bookshelves - num_bookcases)

/-- Calculates the total cost at supermarket B with 20% discount -/
def cost_B : ℕ := (num_bookcases * bookcase_cost + num_bookshelves * bookshelf_cost) * 8 / 10

/-- Theorem stating that the costs at supermarket A and B are equal -/
theorem equal_costs : cost_A = cost_B := by sorry

end NUMINAMATH_CALUDE_equal_costs_l2725_272523


namespace NUMINAMATH_CALUDE_max_xyz_value_l2725_272532

theorem max_xyz_value (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0)
  (h4 : x * y + 3 * z = (x + 3 * z) * (y + 3 * z)) :
  x * y * z ≤ 1 / 81 :=
sorry

end NUMINAMATH_CALUDE_max_xyz_value_l2725_272532


namespace NUMINAMATH_CALUDE_roof_dimension_difference_l2725_272589

theorem roof_dimension_difference (area : ℝ) (length_width_ratio : ℝ) :
  area = 676 ∧ length_width_ratio = 4 →
  ∃ (length width : ℝ),
    length = length_width_ratio * width ∧
    area = length * width ∧
    length - width = 39 :=
by sorry

end NUMINAMATH_CALUDE_roof_dimension_difference_l2725_272589


namespace NUMINAMATH_CALUDE_binomial_12_3_l2725_272513

theorem binomial_12_3 : Nat.choose 12 3 = 220 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_3_l2725_272513


namespace NUMINAMATH_CALUDE_function_graph_point_l2725_272534

theorem function_graph_point (f : ℝ → ℝ) (h : f 8 = 5) :
  let g := fun x => (f (3 * x) / 3 + 3) / 3
  g (8/3) = 14/9 ∧ 8/3 + 14/9 = 38/9 := by
sorry

end NUMINAMATH_CALUDE_function_graph_point_l2725_272534


namespace NUMINAMATH_CALUDE_max_correct_answers_l2725_272500

theorem max_correct_answers (total_questions : ℕ) (correct_points : ℕ) (incorrect_points : ℕ) (total_score : ℕ) :
  total_questions = 50 →
  correct_points = 5 →
  incorrect_points = 2 →
  total_score = 150 →
  ∃ (correct incorrect unanswered : ℕ),
    correct + incorrect + unanswered = total_questions ∧
    correct * correct_points - incorrect * incorrect_points = total_score ∧
    correct ≤ 35 ∧
    ∀ (c : ℕ), c > correct →
      ¬(∃ (i u : ℕ), c + i + u = total_questions ∧
        c * correct_points - i * incorrect_points = total_score) :=
by sorry

end NUMINAMATH_CALUDE_max_correct_answers_l2725_272500


namespace NUMINAMATH_CALUDE_leading_coefficient_of_p_l2725_272568

/-- The polynomial in question -/
def p (x : ℝ) : ℝ := -2*(x^5 - x^4 + 2*x^3) + 6*(x^5 + x^2 - 1) - 5*(3*x^5 + x^3 + 4)

/-- The leading coefficient of a polynomial -/
def leadingCoefficient (p : ℝ → ℝ) : ℝ :=
  sorry  -- Definition of leading coefficient

theorem leading_coefficient_of_p :
  leadingCoefficient p = -11 := by
  sorry

end NUMINAMATH_CALUDE_leading_coefficient_of_p_l2725_272568


namespace NUMINAMATH_CALUDE_mikes_lawn_mowing_earnings_l2725_272562

/-- Proves that Mike's total earnings from mowing lawns is $101 given the conditions --/
theorem mikes_lawn_mowing_earnings : 
  ∀ (total_earnings : ℕ) 
    (mower_blades_cost : ℕ) 
    (num_games : ℕ) 
    (game_cost : ℕ),
  mower_blades_cost = 47 →
  num_games = 9 →
  game_cost = 6 →
  total_earnings = mower_blades_cost + num_games * game_cost →
  total_earnings = 101 := by
sorry

end NUMINAMATH_CALUDE_mikes_lawn_mowing_earnings_l2725_272562


namespace NUMINAMATH_CALUDE_min_rounds_for_peter_win_l2725_272510

/-- Represents a point on a plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a triangle on a plane -/
structure Triangle :=
  (a : Point)
  (b : Point)
  (c : Point)

/-- Represents a color (red or blue) -/
inductive Color
  | Red
  | Blue

/-- Represents the state of the game after each round -/
structure GameState :=
  (points : List Point)
  (colors : List Color)

/-- Checks if two triangles are similar -/
def areSimilar (t1 t2 : Triangle) : Prop := sorry

/-- Checks if there exists a monochromatic triangle similar to the original -/
def existsMonochromaticSimilarTriangle (original : Triangle) (state : GameState) : Prop := sorry

/-- Represents a strategy for Basil (the coloring player) -/
def BasilStrategy := GameState → Color

/-- Represents a strategy for Peter (the point-choosing player) -/
def PeterStrategy := GameState → Point

/-- Checks if Peter wins given both players' strategies and the number of rounds -/
def peterWins (original : Triangle) (peterStrategy : PeterStrategy) (basilStrategy : BasilStrategy) (rounds : ℕ) : Prop := sorry

theorem min_rounds_for_peter_win (original : Triangle) :
  (∃ (peterStrategy : PeterStrategy), ∀ (basilStrategy : BasilStrategy), peterWins original peterStrategy basilStrategy 5) ∧
  (∀ (rounds : ℕ), rounds < 5 → 
    ∀ (peterStrategy : PeterStrategy), ∃ (basilStrategy : BasilStrategy), ¬peterWins original peterStrategy basilStrategy rounds) :=
sorry

end NUMINAMATH_CALUDE_min_rounds_for_peter_win_l2725_272510


namespace NUMINAMATH_CALUDE_ballpoint_pen_price_relation_l2725_272543

/-- Proves the relationship between price and number of pens for a specific box of ballpoint pens -/
theorem ballpoint_pen_price_relation :
  let box_pens : ℕ := 16
  let box_price : ℚ := 24
  let unit_price : ℚ := box_price / box_pens
  ∀ (x : ℚ) (y : ℚ), y = unit_price * x → y = (3/2 : ℚ) * x := by
  sorry

end NUMINAMATH_CALUDE_ballpoint_pen_price_relation_l2725_272543


namespace NUMINAMATH_CALUDE_max_type_c_test_tubes_l2725_272577

theorem max_type_c_test_tubes 
  (a b c : ℕ) 
  (total_tubes : a + b + c = 1000)
  (solution_percentage : 100 * a + 200 * b + 900 * c = 2017 * (a + b + c)) 
  (non_consecutive : a > 0 ∧ b > 0) : 
  c ≤ 73 :=
sorry

end NUMINAMATH_CALUDE_max_type_c_test_tubes_l2725_272577


namespace NUMINAMATH_CALUDE_mean_visits_between_200_and_300_l2725_272572

def website_visits : List Nat := [300, 400, 300, 200, 200]

theorem mean_visits_between_200_and_300 :
  let mean := (website_visits.sum : ℚ) / website_visits.length
  200 < mean ∧ mean < 300 := by
  sorry

end NUMINAMATH_CALUDE_mean_visits_between_200_and_300_l2725_272572


namespace NUMINAMATH_CALUDE_polynomial_multiplication_simplification_l2725_272553

theorem polynomial_multiplication_simplification :
  ∀ (x : ℝ),
  (3 * x - 2) * (5 * x^12 + 3 * x^11 - 4 * x^9 + x^8) =
  15 * x^13 - x^12 - 6 * x^11 - 12 * x^10 + 11 * x^9 - 2 * x^8 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_multiplication_simplification_l2725_272553


namespace NUMINAMATH_CALUDE_largest_non_sum_30_and_composite_l2725_272538

/-- A function that checks if a number is composite -/
def isComposite (n : ℕ) : Prop :=
  ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

/-- A function that checks if a number can be expressed as the sum of a positive integral multiple of 30 and a positive composite integer -/
def isSum30AndComposite (n : ℕ) : Prop :=
  ∃ k c, 0 < k ∧ isComposite c ∧ n = 30 * k + c

/-- Theorem stating that 210 is the largest positive integer that cannot be expressed as the sum of a positive integral multiple of 30 and a positive composite integer -/
theorem largest_non_sum_30_and_composite :
  (∀ n : ℕ, 210 < n → isSum30AndComposite n) ∧
  ¬isSum30AndComposite 210 :=
sorry

end NUMINAMATH_CALUDE_largest_non_sum_30_and_composite_l2725_272538


namespace NUMINAMATH_CALUDE_vector_equation_vectors_parallel_l2725_272558

/-- Given vectors in R² --/
def a : Fin 2 → ℚ := ![3, 2]
def b : Fin 2 → ℚ := ![-1, 2]
def c : Fin 2 → ℚ := ![4, 1]

/-- Theorem for part 1 --/
theorem vector_equation :
  a = (5/9 : ℚ) • b + (8/9 : ℚ) • c := by sorry

/-- Helper function to check if two vectors are parallel --/
def are_parallel (v w : Fin 2 → ℚ) : Prop :=
  v 0 * w 1 = v 1 * w 0

/-- Theorem for part 2 --/
theorem vectors_parallel :
  are_parallel (a + (-16/3 : ℚ) • c) (2 • b - a) := by sorry

end NUMINAMATH_CALUDE_vector_equation_vectors_parallel_l2725_272558


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_difference_l2725_272536

theorem absolute_value_equation_solution_difference : ∃ x₁ x₂ : ℝ, 
  (|x₁ + 3| = 15) ∧ 
  (|x₂ + 3| = 15) ∧ 
  (x₁ ≠ x₂) ∧ 
  (x₁ - x₂ = 30 ∨ x₂ - x₁ = 30) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_difference_l2725_272536


namespace NUMINAMATH_CALUDE_binomial_coefficient_ratio_l2725_272539

theorem binomial_coefficient_ratio (n : ℕ) : 
  (2^3 * (n.choose 3) = 4 * 2^2 * (n.choose 2)) → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_ratio_l2725_272539


namespace NUMINAMATH_CALUDE_cistern_wet_surface_area_l2725_272596

/-- Calculates the total wet surface area of a rectangular cistern -/
def total_wet_surface_area (length width depth : ℝ) : ℝ :=
  length * width + 2 * (length * depth) + 2 * (width * depth)

/-- Theorem: The total wet surface area of a cistern with given dimensions -/
theorem cistern_wet_surface_area :
  total_wet_surface_area 12 14 1.25 = 233 := by
  sorry

#eval total_wet_surface_area 12 14 1.25

end NUMINAMATH_CALUDE_cistern_wet_surface_area_l2725_272596


namespace NUMINAMATH_CALUDE_weight_after_deliveries_l2725_272521

def initial_load : ℝ := 50000
def first_unloading_percentage : ℝ := 0.1
def second_unloading_percentage : ℝ := 0.2

theorem weight_after_deliveries :
  let remaining_after_first := initial_load * (1 - first_unloading_percentage)
  let final_weight := remaining_after_first * (1 - second_unloading_percentage)
  final_weight = 36000 := by sorry

end NUMINAMATH_CALUDE_weight_after_deliveries_l2725_272521


namespace NUMINAMATH_CALUDE_parallelogram_area_l2725_272564

/-- Given a parallelogram with height 1 and a right triangle within it with legs 40 and (55 - a),
    where a is the length of the shorter side, prove that its area is 200/3. -/
theorem parallelogram_area (a : ℝ) (h : a > 0) :
  let height : ℝ := 1
  let leg1 : ℝ := 40
  let leg2 : ℝ := 55 - a
  let area : ℝ := a * leg1
  (leg1 ^ 2 + leg2 ^ 2 = (height * area) ^ 2) → area = 200 / 3 :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_area_l2725_272564


namespace NUMINAMATH_CALUDE_smallest_integer_cubic_inequality_l2725_272567

theorem smallest_integer_cubic_inequality :
  ∃ n : ℤ, (∀ m : ℤ, m^3 - 12*m^2 + 44*m - 48 ≤ 0 → n ≤ m) ∧ 
  (n^3 - 12*n^2 + 44*n - 48 ≤ 0) ∧ n = 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_cubic_inequality_l2725_272567


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficients_l2725_272590

theorem binomial_expansion_coefficients :
  ∀ (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ),
  (∀ x : ℝ, (1 + 2*x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  (a₄ = 80 ∧ a₁ + a₂ + a₃ = 130) := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficients_l2725_272590


namespace NUMINAMATH_CALUDE_decimal_subtraction_l2725_272524

theorem decimal_subtraction :
  let largest_three_digit := 0.999
  let smallest_four_digit := 0.0001
  largest_three_digit - smallest_four_digit = 0.9989 := by
  sorry

end NUMINAMATH_CALUDE_decimal_subtraction_l2725_272524


namespace NUMINAMATH_CALUDE_greatest_integer_less_than_M_over_100_l2725_272554

def M : ℚ :=
  (1 / (3 * 4 * 5 * 6 * 16 * 17 * 18) +
   1 / (4 * 5 * 6 * 7 * 15 * 16 * 17 * 18) +
   1 / (5 * 6 * 7 * 8 * 14 * 15 * 16 * 17 * 18) +
   1 / (6 * 7 * 8 * 9 * 13 * 14 * 15 * 16 * 17 * 18) +
   1 / (7 * 8 * 9 * 10 * 12 * 13 * 14 * 15 * 16 * 17 * 18) +
   1 / (8 * 9 * 10 * 11 * 11 * 12 * 13 * 14 * 15 * 16 * 17 * 18) +
   1 / (9 * 10 * 11 * 12 * 10 * 11 * 12 * 13 * 14 * 15 * 16 * 17 * 18)) * (2 * 17 * 18)

theorem greatest_integer_less_than_M_over_100 : 
  ∀ n : ℤ, n ≤ ⌊M / 100⌋ ↔ n ≤ 145 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_less_than_M_over_100_l2725_272554


namespace NUMINAMATH_CALUDE_min_value_at_angle_l2725_272582

def minimizing_angle (k : ℤ) : ℝ := 660 + 720 * k

theorem min_value_at_angle (A : ℝ) :
  (∃ k : ℤ, A = minimizing_angle k) ↔
  ∀ B : ℝ, Real.sin (A / 2) - Real.sqrt 3 * Real.cos (A / 2) ≤ 
           Real.sin (B / 2) - Real.sqrt 3 * Real.cos (B / 2) :=
by sorry

#check min_value_at_angle

end NUMINAMATH_CALUDE_min_value_at_angle_l2725_272582


namespace NUMINAMATH_CALUDE_speed_of_sound_in_open_pipe_l2725_272522

-- Define the parameters
variable (l : ℝ) -- Length of the pipe
variable (h : ℝ) -- Fundamental frequency
variable (c : ℝ) -- Speed of sound

-- Define the properties of the pipe
variable (pipe_open : Bool) -- Pipe is open at both ends
variable (pipe_length : l > 0) -- Pipe has positive length

-- Theorem statement
theorem speed_of_sound_in_open_pipe (pipe_open : pipe_open = true) :
  c = 2 * h * l :=
sorry

end NUMINAMATH_CALUDE_speed_of_sound_in_open_pipe_l2725_272522


namespace NUMINAMATH_CALUDE_remainder_theorem_l2725_272559

theorem remainder_theorem : 
  (86592 : ℤ) % 8 = 0 ∧ (8741 : ℤ) % 13 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2725_272559


namespace NUMINAMATH_CALUDE_final_pen_count_l2725_272569

def pen_collection (initial : ℕ) (mike_gives : ℕ) (cindy_multiplier : ℕ) (sharon_takes : ℕ) : ℕ :=
  ((initial + mike_gives) * cindy_multiplier) - sharon_takes

theorem final_pen_count : pen_collection 5 20 2 10 = 40 := by
  sorry

end NUMINAMATH_CALUDE_final_pen_count_l2725_272569


namespace NUMINAMATH_CALUDE_profit_maximized_at_twelve_point_five_l2725_272507

/-- The profit function for the bookstore -/
def P (p : ℝ) : ℝ := 150 * p - 6 * p^2 - 200

/-- The theorem stating that the profit is maximized at p = 12.5 -/
theorem profit_maximized_at_twelve_point_five :
  ∃ (p : ℝ), 0 ≤ p ∧ p ≤ 30 ∧ 
  (∀ (q : ℝ), 0 ≤ q ∧ q ≤ 30 → P p ≥ P q) ∧
  p = 12.5 := by
sorry

end NUMINAMATH_CALUDE_profit_maximized_at_twelve_point_five_l2725_272507


namespace NUMINAMATH_CALUDE_morning_trip_fare_correct_afternoon_trip_fare_formula_afternoon_trip_fare_specific_l2725_272516

/- Define the time periods and corresponding rates -/
def normal_mileage_rate : ℝ := 2.20
def early_morning_mileage_rate : ℝ := 2.80
def peak_mileage_rate : ℝ := 2.75
def normal_time_rate : ℝ := 0.38
def peak_time_rate : ℝ := 0.47

/- Define the fare calculation function -/
def calculate_fare (distance : ℝ) (time : ℝ) (mileage_rate : ℝ) (time_rate : ℝ) : ℝ :=
  distance * mileage_rate + time * time_rate

/- Theorem for the morning trip -/
theorem morning_trip_fare_correct :
  calculate_fare 6 10 early_morning_mileage_rate normal_time_rate = 20.6 := by sorry

/- Theorem for the afternoon trip (general formula) -/
theorem afternoon_trip_fare_formula (x : ℝ) (h : x ≤ 30) :
  calculate_fare x (x / 30 * 60) peak_mileage_rate peak_time_rate = 3.69 * x := by sorry

/- Theorem for the afternoon trip when x = 8 -/
theorem afternoon_trip_fare_specific :
  calculate_fare 8 16 peak_mileage_rate peak_time_rate = 29.52 := by sorry

end NUMINAMATH_CALUDE_morning_trip_fare_correct_afternoon_trip_fare_formula_afternoon_trip_fare_specific_l2725_272516


namespace NUMINAMATH_CALUDE_sequence_fifth_term_is_fifteen_l2725_272579

theorem sequence_fifth_term_is_fifteen (a : ℕ → ℝ) :
  (∀ n : ℕ, n ≠ 0 → a n / n = n - 2) →
  a 5 = 15 := by
sorry

end NUMINAMATH_CALUDE_sequence_fifth_term_is_fifteen_l2725_272579


namespace NUMINAMATH_CALUDE_boat_length_in_steps_l2725_272549

/-- Represents the scenario of Josie jogging alongside a moving boat --/
structure JosieAndBoat where
  josie_speed : ℝ
  boat_speed : ℝ
  boat_length : ℝ
  step_length : ℝ
  steps_forward : ℕ
  steps_backward : ℕ

/-- The conditions of the problem --/
def problem_conditions (scenario : JosieAndBoat) : Prop :=
  scenario.josie_speed > scenario.boat_speed ∧
  scenario.steps_forward = 130 ∧
  scenario.steps_backward = 70 ∧
  scenario.boat_length = scenario.step_length * 91

/-- The theorem to be proved --/
theorem boat_length_in_steps (scenario : JosieAndBoat) 
  (h : problem_conditions scenario) : 
  scenario.boat_length = scenario.step_length * 91 :=
sorry

end NUMINAMATH_CALUDE_boat_length_in_steps_l2725_272549
