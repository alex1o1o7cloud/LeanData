import Mathlib

namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l3457_345774

theorem pure_imaginary_condition (a : ℝ) : 
  (Complex.I + 1) * (Complex.I * a + 2) = Complex.I * (Complex.I * b + c) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l3457_345774


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3457_345724

theorem quadratic_factorization (C D : ℤ) :
  (∀ x : ℝ, 16 * x^2 - 88 * x + 63 = (C * x - 21) * (D * x - 3)) →
  C * D + C = 21 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3457_345724


namespace NUMINAMATH_CALUDE_triangle_rectangle_side_ratio_l3457_345713

/-- Given an equilateral triangle and a rectangle with the same perimeter and a specific length-width ratio for the rectangle, this theorem proves that the ratio of the triangle's side length to the rectangle's length is 1. -/
theorem triangle_rectangle_side_ratio (perimeter : ℝ) (length_width_ratio : ℝ) :
  perimeter > 0 →
  length_width_ratio = 2 →
  let triangle_side := perimeter / 3
  let rectangle_width := perimeter / (2 * (length_width_ratio + 1))
  let rectangle_length := length_width_ratio * rectangle_width
  (triangle_side / rectangle_length) = 1 := by
  sorry

#check triangle_rectangle_side_ratio

end NUMINAMATH_CALUDE_triangle_rectangle_side_ratio_l3457_345713


namespace NUMINAMATH_CALUDE_series_numerator_divisibility_l3457_345796

theorem series_numerator_divisibility (n : ℕ+) (h : Nat.Prime (3 * n + 1)) :
  ∃ k : ℤ, 2 * n - 1 = k * (3 * n + 1) := by
  sorry

end NUMINAMATH_CALUDE_series_numerator_divisibility_l3457_345796


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l3457_345795

theorem least_subtraction_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (x : ℕ), x ≤ d - 1 ∧ (n - x) % d = 0 ∧ ∀ (y : ℕ), y < x → (n - y) % d ≠ 0 :=
sorry

theorem problem_solution :
  ∃ (x : ℕ), x = 8 ∧ (42398 - x) % 15 = 0 ∧ ∀ (y : ℕ), y < x → (42398 - y) % 15 ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l3457_345795


namespace NUMINAMATH_CALUDE_lcm_gcd_product_equals_number_product_l3457_345761

theorem lcm_gcd_product_equals_number_product : 
  let a := 24
  let b := 36
  Nat.lcm a b * Nat.gcd a b = a * b :=
by sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_equals_number_product_l3457_345761


namespace NUMINAMATH_CALUDE_tan_alpha_problem_l3457_345745

theorem tan_alpha_problem (α : Real) 
  (h1 : Real.tan (α + π/4) = -1/2) 
  (h2 : π/2 < α) 
  (h3 : α < π) : 
  (Real.sin (2*α) - 2*(Real.cos α)^2) / Real.sin (α - π/4) = -2*Real.sqrt 5/5 :=
by sorry

end NUMINAMATH_CALUDE_tan_alpha_problem_l3457_345745


namespace NUMINAMATH_CALUDE_two_dice_sum_ten_max_digits_l3457_345751

theorem two_dice_sum_ten_max_digits : ∀ x y : ℕ,
  1 ≤ x ∧ x ≤ 6 ∧ 1 ≤ y ∧ y ≤ 6 → x + y = 10 → x < 10 ∧ y < 10 :=
by sorry

end NUMINAMATH_CALUDE_two_dice_sum_ten_max_digits_l3457_345751


namespace NUMINAMATH_CALUDE_abrahams_a_students_l3457_345708

theorem abrahams_a_students (total_students : ℕ) (total_a_students : ℕ) (abraham_students : ℕ) :
  total_students = 40 →
  total_a_students = 25 →
  abraham_students = 10 →
  (abraham_students : ℚ) / total_students * total_a_students = (abraham_students : ℕ) →
  ∃ (abraham_a_students : ℕ), 
    (abraham_a_students : ℚ) / abraham_students = (total_a_students : ℚ) / total_students ∧
    abraham_a_students = 6 :=
by sorry

end NUMINAMATH_CALUDE_abrahams_a_students_l3457_345708


namespace NUMINAMATH_CALUDE_work_completion_time_l3457_345716

/-- The time needed to complete the work -/
def complete_work (p q : ℝ) (t : ℝ) : Prop :=
  let work_p := t / p
  let work_q := (t - 16) / q
  work_p + work_q = 1

theorem work_completion_time :
  ∀ p q : ℝ,
  p > 0 → q > 0 →
  complete_work p q 40 →
  complete_work q q 24 →
  ∃ t : ℝ, t > 0 ∧ complete_work p q t ∧ t = 25 := by
sorry

end NUMINAMATH_CALUDE_work_completion_time_l3457_345716


namespace NUMINAMATH_CALUDE_problem_solution_l3457_345762

theorem problem_solution (m n : ℝ) 
  (hm : 3 * m^2 + 5 * m - 3 = 0) 
  (hn : 3 * n^2 - 5 * n - 3 = 0) 
  (hmn : m * n ≠ 1) : 
  1 / n^2 + m / n - (5/3) * m = 25/9 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3457_345762


namespace NUMINAMATH_CALUDE_challenge_probability_challenge_probability_value_l3457_345704

/-- The probability of selecting all letters from "CHALLENGE" when choosing 3 letters from "FARM", 
    4 letters from "BENCHES", and 2 letters from "GLOVE" -/
theorem challenge_probability : ℚ := by
  -- Define the number of letters in each word
  let farm_letters : ℕ := 4
  let benches_letters : ℕ := 7
  let glove_letters : ℕ := 5

  -- Define the number of letters to be selected from each word
  let farm_select : ℕ := 3
  let benches_select : ℕ := 4
  let glove_select : ℕ := 2

  -- Define the number of required letters from each word
  let farm_required : ℕ := 2  -- A and L
  let benches_required : ℕ := 3  -- C, H, and E
  let glove_required : ℕ := 2  -- G and E

  -- Calculate the probability
  sorry

-- The theorem states that the probability is 2/350
theorem challenge_probability_value : challenge_probability = 2 / 350 := by sorry

end NUMINAMATH_CALUDE_challenge_probability_challenge_probability_value_l3457_345704


namespace NUMINAMATH_CALUDE_football_draw_count_l3457_345750

/-- Represents the possible outcomes of a football match -/
inductive MatchResult
| Win
| Draw
| Loss

/-- Calculates the points for a given match result -/
def pointsForResult (result : MatchResult) : ℕ :=
  match result with
  | MatchResult.Win => 3
  | MatchResult.Draw => 1
  | MatchResult.Loss => 0

/-- Represents the results of a series of matches -/
structure MatchResults :=
  (wins : ℕ)
  (draws : ℕ)
  (losses : ℕ)

/-- Calculates the total points for a series of match results -/
def totalPoints (results : MatchResults) : ℕ :=
  3 * results.wins + results.draws

theorem football_draw_count 
  (total_matches : ℕ) 
  (total_points : ℕ) 
  (h_matches : total_matches = 5) 
  (h_points : total_points = 7) :
  ∃ (results : MatchResults), 
    results.wins + results.draws + results.losses = total_matches ∧ 
    totalPoints results = total_points ∧ 
    (results.draws = 1 ∨ results.draws = 4) :=
sorry

end NUMINAMATH_CALUDE_football_draw_count_l3457_345750


namespace NUMINAMATH_CALUDE_jamie_min_score_l3457_345730

/-- The minimum average score required on the last two tests to qualify for a geometry class. -/
def min_average_score (score1 score2 score3 : ℚ) (required_average : ℚ) (num_tests : ℕ) : ℚ :=
  ((required_average * num_tests) - (score1 + score2 + score3)) / 2

/-- Theorem stating the minimum average score Jamie must achieve on the next two tests. -/
theorem jamie_min_score : 
  min_average_score 80 90 78 85 5 = 88.5 := by sorry

end NUMINAMATH_CALUDE_jamie_min_score_l3457_345730


namespace NUMINAMATH_CALUDE_suit_cost_ratio_l3457_345753

theorem suit_cost_ratio (off_rack_cost tailoring_cost total_cost : ℝ) 
  (h1 : off_rack_cost = 300)
  (h2 : tailoring_cost = 200)
  (h3 : total_cost = 1400)
  (h4 : ∃ x : ℝ, total_cost = off_rack_cost + (x * off_rack_cost + tailoring_cost)) :
  ∃ x : ℝ, x = 3 ∧ total_cost = off_rack_cost + (x * off_rack_cost + tailoring_cost) :=
by sorry

end NUMINAMATH_CALUDE_suit_cost_ratio_l3457_345753


namespace NUMINAMATH_CALUDE_registration_combinations_l3457_345701

/-- The number of students --/
def num_students : ℕ := 5

/-- The number of activity groups --/
def num_groups : ℕ := 2

/-- The total number of registration methods --/
def total_registrations : ℕ := num_groups ^ num_students

/-- Theorem stating the total number of registration methods --/
theorem registration_combinations :
  total_registrations = 32 := by sorry

end NUMINAMATH_CALUDE_registration_combinations_l3457_345701


namespace NUMINAMATH_CALUDE_stating_first_alloy_amount_l3457_345746

/-- Represents an alloy with a specific ratio of lead to tin -/
structure Alloy where
  lead : ℝ
  tin : ℝ

/-- The first available alloy -/
def alloy1 : Alloy := { lead := 1, tin := 2 }

/-- The second available alloy -/
def alloy2 : Alloy := { lead := 2, tin := 3 }

/-- The desired new alloy -/
def newAlloy : Alloy := { lead := 4, tin := 7 }

/-- The total mass of the new alloy -/
def totalMass : ℝ := 22

/-- 
Theorem stating that 12 grams of the first alloy is needed to create the new alloy
with the desired properties
-/
theorem first_alloy_amount : 
  ∃ (x y : ℝ),
    x * (alloy1.lead + alloy1.tin) + y * (alloy2.lead + alloy2.tin) = totalMass ∧
    (x * alloy1.lead + y * alloy2.lead) / (x * alloy1.tin + y * alloy2.tin) = newAlloy.lead / newAlloy.tin ∧
    x * (alloy1.lead + alloy1.tin) = 12 := by
  sorry


end NUMINAMATH_CALUDE_stating_first_alloy_amount_l3457_345746


namespace NUMINAMATH_CALUDE_greatest_average_speed_l3457_345757

/-- Checks if a number is a palindrome -/
def isPalindrome (n : ℕ) : Prop := sorry

/-- The initial odometer reading -/
def initialReading : ℕ := 12321

/-- The duration of the drive in hours -/
def driveDuration : ℝ := 4

/-- The speed limit in miles per hour -/
def speedLimit : ℝ := 85

/-- The greatest possible average speed in miles per hour -/
def greatestAverageSpeed : ℝ := 75

/-- Theorem stating the greatest possible average speed given the conditions -/
theorem greatest_average_speed :
  isPalindrome initialReading →
  ∃ (finalReading : ℕ),
    isPalindrome finalReading ∧
    finalReading > initialReading ∧
    (finalReading - initialReading : ℝ) / driveDuration ≤ speedLimit ∧
    (finalReading - initialReading : ℝ) / driveDuration = greatestAverageSpeed :=
  sorry

end NUMINAMATH_CALUDE_greatest_average_speed_l3457_345757


namespace NUMINAMATH_CALUDE_solution_pair_l3457_345702

theorem solution_pair : ∃! (x y : ℝ), 
  (x + y = (7 - x) + (7 - y)) ∧ 
  (x - y = (x - 2) + (y - 2)) ∧ 
  x = 3 ∧ y = 4 := by
sorry

end NUMINAMATH_CALUDE_solution_pair_l3457_345702


namespace NUMINAMATH_CALUDE_wind_on_rainy_day_probability_l3457_345703

/-- Given probabilities in a meteorological context -/
structure WeatherProbabilities where
  rain : ℚ
  wind : ℚ
  both : ℚ

/-- The probability of wind on a rainy day -/
def windOnRainyDay (wp : WeatherProbabilities) : ℚ :=
  wp.both / wp.rain

/-- Theorem stating the probability of wind on a rainy day -/
theorem wind_on_rainy_day_probability (wp : WeatherProbabilities) 
  (h1 : wp.rain = 4/15)
  (h2 : wp.wind = 2/15)
  (h3 : wp.both = 1/10) :
  windOnRainyDay wp = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_wind_on_rainy_day_probability_l3457_345703


namespace NUMINAMATH_CALUDE_sum_of_ages_l3457_345709

/-- Represents the ages of family members -/
structure FamilyAges where
  father : ℕ
  son : ℕ
  mother : ℕ
  daughter : ℕ

/-- The conditions of the problem -/
def satisfiesConditions (ages : FamilyAges) : Prop :=
  (ages.father - 20 = 3 * (ages.son - 20)) ∧
  (ages.mother - 20 = 4 * (ages.daughter - 20)) ∧
  (ages.father = 2 * ages.son) ∧
  (ages.mother = 3 * ages.daughter)

/-- The theorem to be proved -/
theorem sum_of_ages (ages : FamilyAges) : 
  satisfiesConditions ages → ages.father + ages.son + ages.mother + ages.daughter = 360 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ages_l3457_345709


namespace NUMINAMATH_CALUDE_total_sheep_l3457_345792

theorem total_sheep (aaron_sheep beth_sheep : ℕ) 
  (h1 : aaron_sheep = 532)
  (h2 : beth_sheep = 76)
  (h3 : aaron_sheep = 7 * beth_sheep) : 
  aaron_sheep + beth_sheep = 608 := by
sorry

end NUMINAMATH_CALUDE_total_sheep_l3457_345792


namespace NUMINAMATH_CALUDE_diagonal_cubes_150_324_375_l3457_345768

/-- The number of unit cubes that a diagonal passes through in a rectangular prism -/
def diagonal_cubes (a b c : ℕ) : ℕ :=
  a + b + c - Nat.gcd a b - Nat.gcd a c - Nat.gcd b c + Nat.gcd (Nat.gcd a b) c

/-- Theorem: In a 150 × 324 × 375 rectangular prism, the diagonal passes through 768 unit cubes -/
theorem diagonal_cubes_150_324_375 :
  diagonal_cubes 150 324 375 = 768 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_cubes_150_324_375_l3457_345768


namespace NUMINAMATH_CALUDE_max_value_expression_l3457_345733

theorem max_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x : ℝ), ∀ (a b : ℝ), a > 0 → b > 0 →
    (|4*a - 10*b| + |2*(a - b*Real.sqrt 3) - 5*(a*Real.sqrt 3 + b)|) / Real.sqrt (a^2 + b^2) ≤ x) ∧
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
    (|4*a - 10*b| + |2*(a - b*Real.sqrt 3) - 5*(a*Real.sqrt 3 + b)|) / Real.sqrt (a^2 + b^2) = 2 * Real.sqrt 87) :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l3457_345733


namespace NUMINAMATH_CALUDE_diagonals_divisible_by_3_count_l3457_345754

/-- A convex polygon with 30 sides -/
structure ConvexPolygon30 where
  sides : ℕ
  convex : Bool
  sides_eq_30 : sides = 30

/-- The number of diagonals in a polygon that are divisible by 3 -/
def diagonals_divisible_by_3 (p : ConvexPolygon30) : ℕ := 17

/-- Theorem stating that the number of diagonals divisible by 3 in a convex 30-sided polygon is 17 -/
theorem diagonals_divisible_by_3_count (p : ConvexPolygon30) : 
  diagonals_divisible_by_3 p = 17 := by sorry

end NUMINAMATH_CALUDE_diagonals_divisible_by_3_count_l3457_345754


namespace NUMINAMATH_CALUDE_special_line_equation_l3457_345755

/-- A line passing through (-10, 10) with x-intercept four times y-intercept -/
structure SpecialLine where
  -- The line passes through (-10, 10)
  passes_through : (x : ℝ) → (y : ℝ) → x = -10 ∧ y = 10 → x * a + y * b + c = 0
  -- The x-intercept is four times the y-intercept
  intercept_relation : (x : ℝ) → (y : ℝ) → x * b = y * a → x = 4 * y
  -- The coefficients of the line equation
  a : ℝ
  b : ℝ
  c : ℝ

/-- The equation of the special line is either x + y = 0 or x + 4y - 30 = 0 -/
theorem special_line_equation (L : SpecialLine) :
  (L.a = 1 ∧ L.b = 1 ∧ L.c = 0) ∨ (L.a = 1 ∧ L.b = 4 ∧ L.c = -30) :=
sorry

end NUMINAMATH_CALUDE_special_line_equation_l3457_345755


namespace NUMINAMATH_CALUDE_jacks_age_problem_l3457_345741

theorem jacks_age_problem (jack_age_2010 : ℕ) (mother_age_multiplier : ℕ) : 
  jack_age_2010 = 12 →
  mother_age_multiplier = 3 →
  ∃ (years_after_2010 : ℕ), 
    (jack_age_2010 + years_after_2010) * 2 = (jack_age_2010 * mother_age_multiplier + years_after_2010) ∧
    years_after_2010 = 12 :=
by sorry

end NUMINAMATH_CALUDE_jacks_age_problem_l3457_345741


namespace NUMINAMATH_CALUDE_expression_evaluation_l3457_345793

theorem expression_evaluation : 
  (4+8-16+32+64-128+256)/(8+16-32+64+128-256+512) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3457_345793


namespace NUMINAMATH_CALUDE_stone_number_150_l3457_345780

/-- Represents the counting pattern for each round -/
def countingPattern : List Nat := [12, 10, 8, 6, 4, 2]

/-- Calculates the sum of a list of natural numbers -/
def sumList (l : List Nat) : Nat :=
  l.foldl (· + ·) 0

/-- Represents the total count in one complete cycle -/
def cycleCount : Nat := sumList countingPattern

/-- Calculates the number of complete cycles before reaching the target count -/
def completeCycles (target : Nat) : Nat :=
  target / cycleCount

/-- Calculates the remaining count after complete cycles -/
def remainingCount (target : Nat) : Nat :=
  target % cycleCount

/-- Finds the original stone number corresponding to the target count -/
def findStoneNumber (target : Nat) : Nat :=
  let remainingCount := remainingCount target
  let rec findInPattern (count : Nat) (pattern : List Nat) : Nat :=
    match pattern with
    | [] => 0  -- Should not happen if the input is valid
    | h :: t =>
      if count <= h then
        12 - (h - count) - (6 - pattern.length) * 2
      else
        findInPattern (count - h) t
  findInPattern remainingCount countingPattern

theorem stone_number_150 :
  findStoneNumber 150 = 4 := by sorry

end NUMINAMATH_CALUDE_stone_number_150_l3457_345780


namespace NUMINAMATH_CALUDE_min_sum_of_squares_l3457_345763

theorem min_sum_of_squares (x y : ℕ) (h : x^2 - y^2 = 121) : 
  ∃ (a b : ℕ), a^2 - b^2 = 121 ∧ a^2 + b^2 ≤ x^2 + y^2 ∧ a^2 + b^2 = 121 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_l3457_345763


namespace NUMINAMATH_CALUDE_symmetry_origin_symmetry_point_l3457_345705

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define symmetry with respect to the origin
def symmetricToOrigin (p : Point2D) : Point2D :=
  { x := -p.x, y := -p.y }

-- Define symmetry with respect to another point
def symmetricToPoint (p : Point2D) (k : Point2D) : Point2D :=
  { x := 2 * k.x - p.x, y := 2 * k.y - p.y }

-- Theorem for symmetry with respect to the origin
theorem symmetry_origin (m : Point2D) :
  symmetricToOrigin m = { x := -m.x, y := -m.y } := by
  sorry

-- Theorem for symmetry with respect to another point
theorem symmetry_point (m k : Point2D) :
  symmetricToPoint m k = { x := 2 * k.x - m.x, y := 2 * k.y - m.y } := by
  sorry

end NUMINAMATH_CALUDE_symmetry_origin_symmetry_point_l3457_345705


namespace NUMINAMATH_CALUDE_even_sum_probability_l3457_345799

/-- Represents a wheel with a given number of even and odd sections -/
structure Wheel where
  total : ℕ
  even : ℕ
  odd : ℕ
  h1 : even + odd = total
  h2 : 0 < total

/-- The probability of getting an even number on a wheel -/
def prob_even (w : Wheel) : ℚ :=
  w.even / w.total

/-- The probability of getting an odd number on a wheel -/
def prob_odd (w : Wheel) : ℚ :=
  w.odd / w.total

/-- Wheel A with 2 even and 3 odd sections -/
def wheel_a : Wheel :=
  { total := 5
  , even := 2
  , odd := 3
  , h1 := by simp
  , h2 := by simp }

/-- Wheel B with 1 even and 1 odd section -/
def wheel_b : Wheel :=
  { total := 2
  , even := 1
  , odd := 1
  , h1 := by simp
  , h2 := by simp }

/-- The probability of getting an even sum when spinning both wheels -/
def prob_even_sum (a b : Wheel) : ℚ :=
  prob_even a * prob_even b + prob_odd a * prob_odd b

theorem even_sum_probability :
  prob_even_sum wheel_a wheel_b = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_even_sum_probability_l3457_345799


namespace NUMINAMATH_CALUDE_perpendicular_lines_m_values_l3457_345767

/-- Two lines are perpendicular if the sum of the products of their coefficients of x and y is zero -/
def are_perpendicular (a1 b1 a2 b2 : ℝ) : Prop := a1 * a2 + b1 * b2 = 0

/-- The first line: mx - (m+2)y + 2 = 0 -/
def line1 (m : ℝ) (x y : ℝ) : Prop := m * x - (m + 2) * y + 2 = 0

/-- The second line: 3x - my - 1 = 0 -/
def line2 (m : ℝ) (x y : ℝ) : Prop := 3 * x - m * y - 1 = 0

theorem perpendicular_lines_m_values :
  ∀ m : ℝ, are_perpendicular m (-(m+2)) 3 (-m) → m = 0 ∨ m = -5 := by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_m_values_l3457_345767


namespace NUMINAMATH_CALUDE_stock_price_increase_probability_l3457_345711

/-- Probability of stock price increase given interest rate conditions -/
theorem stock_price_increase_probability
  (p_increase_when_lowered : ℝ)
  (p_increase_when_unchanged : ℝ)
  (p_increase_when_raised : ℝ)
  (p_rate_reduction : ℝ)
  (p_rate_unchanged : ℝ)
  (h1 : p_increase_when_lowered = 0.7)
  (h2 : p_increase_when_unchanged = 0.2)
  (h3 : p_increase_when_raised = 0.1)
  (h4 : p_rate_reduction = 0.6)
  (h5 : p_rate_unchanged = 0.3)
  (h6 : p_rate_reduction + p_rate_unchanged + (1 - p_rate_reduction - p_rate_unchanged) = 1) :
  p_rate_reduction * p_increase_when_lowered +
  p_rate_unchanged * p_increase_when_unchanged +
  (1 - p_rate_reduction - p_rate_unchanged) * p_increase_when_raised = 0.49 := by
  sorry


end NUMINAMATH_CALUDE_stock_price_increase_probability_l3457_345711


namespace NUMINAMATH_CALUDE_total_lists_is_forty_l3457_345732

/-- The number of elements in the first set (Bin A) -/
def set_A_size : ℕ := 8

/-- The number of elements in the second set (Bin B) -/
def set_B_size : ℕ := 5

/-- The total number of possible lists -/
def total_lists : ℕ := set_A_size * set_B_size

/-- Theorem stating that the total number of possible lists is 40 -/
theorem total_lists_is_forty : total_lists = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_lists_is_forty_l3457_345732


namespace NUMINAMATH_CALUDE_equation_root_l3457_345719

theorem equation_root : ∃ x : ℝ, 
  169 * (157 - 77 * x)^2 + 100 * (201 - 100 * x)^2 = 26 * (77 * x - 157) * (1000 * x - 2010) ∧ 
  x = 31 := by
  sorry

end NUMINAMATH_CALUDE_equation_root_l3457_345719


namespace NUMINAMATH_CALUDE_arccos_neg_one_eq_pi_l3457_345778

theorem arccos_neg_one_eq_pi : Real.arccos (-1) = π := by
  sorry

end NUMINAMATH_CALUDE_arccos_neg_one_eq_pi_l3457_345778


namespace NUMINAMATH_CALUDE_exactly_two_correct_l3457_345764

-- Define a mapping
def Mapping (A B : Type) := A → B

-- Define a function
def Function (α : Type) := α → ℝ

-- Define an odd function
def OddFunction (f : Function ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define propositions
def Proposition1 (A B : Type) : Prop :=
  ∃ (f : Mapping A B), ∃ b : B, ∀ a : A, f a ≠ b

def Proposition2 : Prop :=
  ∀ (f : Function ℝ) (t : ℝ), ∃! x : ℝ, f x = t

def Proposition3 (f : Function ℝ) : Prop :=
  (∀ x y, f (x + y) = f x + f y) → OddFunction f

def Proposition4 (f : Function ℝ) : Prop :=
  (∀ x, 0 ≤ f (2*x - 1) ∧ f (2*x - 1) ≤ 1) →
  (∀ x, -1 ≤ f x ∧ f x ≤ 1)

-- Theorem statement
theorem exactly_two_correct :
  (Proposition1 ℝ ℝ) ∧
  (∃ f : Function ℝ, Proposition3 f) ∧
  ¬(Proposition2) ∧
  ¬(∃ f : Function ℝ, Proposition4 f) :=
sorry

end NUMINAMATH_CALUDE_exactly_two_correct_l3457_345764


namespace NUMINAMATH_CALUDE_chemical_solution_concentration_l3457_345787

/-- Prove that given the conditions of the chemical solution problem, 
    the original solution concentration is 85%. -/
theorem chemical_solution_concentration 
  (x : ℝ) 
  (P : ℝ) 
  (h1 : x = 0.6923076923076923)
  (h2 : (1 - x) * P + x * 20 = 40) : 
  P = 85 := by
  sorry

end NUMINAMATH_CALUDE_chemical_solution_concentration_l3457_345787


namespace NUMINAMATH_CALUDE_f_condition_iff_a_range_l3457_345734

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then 2 - a * x
  else 1/3 * x^3 - 3/2 * a * x^2 + (2 * a^2 + 2) * x - 11/6

theorem f_condition_iff_a_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f a x₁ - f a x₂ < 2 * x₁ - 2 * x₂) ↔ a < -2 := by
  sorry

end NUMINAMATH_CALUDE_f_condition_iff_a_range_l3457_345734


namespace NUMINAMATH_CALUDE_simplify_polynomial_l3457_345756

theorem simplify_polynomial (y : ℝ) : 
  y * (4 * y^2 + 3) - 6 * (y^2 + 3 * y - 8) = 4 * y^3 - 6 * y^2 - 15 * y + 48 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l3457_345756


namespace NUMINAMATH_CALUDE_tangent_slope_angle_range_l3457_345714

theorem tangent_slope_angle_range :
  ∀ (x : ℝ),
  let y := x^3 - x + 2/3
  let slope := (3 * x^2 - 1 : ℝ)
  let α := Real.arctan slope
  α ∈ Set.union (Set.Ico 0 (π/2)) (Set.Icc (3*π/4) π) := by
sorry

end NUMINAMATH_CALUDE_tangent_slope_angle_range_l3457_345714


namespace NUMINAMATH_CALUDE_smallest_product_l3457_345776

def digits : List ℕ := [6, 7, 8, 9]

def is_valid_placement (a b c d : ℕ) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def product (a b c d : ℕ) : ℕ := (10 * a + b) * (10 * c + d)

theorem smallest_product :
  ∀ a b c d : ℕ, is_valid_placement a b c d →
  product a b c d ≥ 5372 :=
sorry

end NUMINAMATH_CALUDE_smallest_product_l3457_345776


namespace NUMINAMATH_CALUDE_photocopy_cost_calculation_l3457_345723

/-- The cost of one photocopy -/
def photocopy_cost : ℝ := sorry

/-- The discount rate for orders over 100 copies -/
def discount_rate : ℝ := 0.25

/-- The number of copies each person needs -/
def copies_per_person : ℕ := 80

/-- The total number of copies when combining orders -/
def total_copies : ℕ := 2 * copies_per_person

/-- The amount saved per person when combining orders -/
def savings_per_person : ℝ := 0.40

theorem photocopy_cost_calculation : 
  photocopy_cost = 0.02 :=
by
  sorry

end NUMINAMATH_CALUDE_photocopy_cost_calculation_l3457_345723


namespace NUMINAMATH_CALUDE_disk_rotation_on_clock_face_l3457_345772

theorem disk_rotation_on_clock_face (clock_radius disk_radius : ℝ) 
  (h1 : clock_radius = 30)
  (h2 : disk_radius = 15)
  (h3 : disk_radius = clock_radius / 2) :
  let initial_position := 0 -- 12 o'clock
  let final_position := π -- 6 o'clock (π radians)
  ∃ (θ : ℝ), 
    θ * disk_radius = final_position * clock_radius ∧ 
    θ % (2 * π) = 0 := by
  sorry

end NUMINAMATH_CALUDE_disk_rotation_on_clock_face_l3457_345772


namespace NUMINAMATH_CALUDE_equation_solution_l3457_345720

theorem equation_solution :
  ∃ x : ℝ, x > 0 ∧ 
  (1 / 3) * (4 * x^2 - 2) = (x^2 - 75 * x - 15) * (x^2 + 40 * x + 8) ∧
  x = (75 + Real.sqrt 5701) / 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3457_345720


namespace NUMINAMATH_CALUDE_simplest_quadratic_radical_l3457_345740

theorem simplest_quadratic_radical : 
  let options : List ℝ := [Real.sqrt (1/2), Real.sqrt 8, Real.sqrt 15, Real.sqrt 20]
  ∀ x ∈ options, x ≠ Real.sqrt 15 → 
    ∃ y z : ℕ, (y > 1 ∧ z > 1 ∧ x = Real.sqrt y * z) ∨ 
              (y > 1 ∧ z > 1 ∧ x = (Real.sqrt y) / z) :=
by sorry

end NUMINAMATH_CALUDE_simplest_quadratic_radical_l3457_345740


namespace NUMINAMATH_CALUDE_triangle_inequality_l3457_345737

theorem triangle_inequality (a b c x y z : ℝ) 
  (triangle_cond : 0 < a ∧ 0 < b ∧ 0 < c ∧ a < b + c ∧ b < a + c ∧ c < a + b)
  (sum_zero : x + y + z = 0) :
  a^2 * y * z + b^2 * z * x + c^2 * x * y ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3457_345737


namespace NUMINAMATH_CALUDE_book_purchase_problem_l3457_345783

/-- Proves that given the conditions of the book purchase problem, the number of math books is 53. -/
theorem book_purchase_problem (total_books : ℕ) (math_cost history_cost total_price : ℚ) 
  (h_total : total_books = 90)
  (h_math_cost : math_cost = 4)
  (h_history_cost : history_cost = 5)
  (h_total_price : total_price = 397) :
  ∃ (math_books : ℕ), 
    math_books = 53 ∧ 
    math_books ≤ total_books ∧
    ∃ (history_books : ℕ),
      history_books = total_books - math_books ∧
      math_cost * math_books + history_cost * history_books = total_price := by
sorry

end NUMINAMATH_CALUDE_book_purchase_problem_l3457_345783


namespace NUMINAMATH_CALUDE_xyz_value_l3457_345725

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 36)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 22) :
  x * y * z = 14 / 3 := by
  sorry

end NUMINAMATH_CALUDE_xyz_value_l3457_345725


namespace NUMINAMATH_CALUDE_houses_in_block_l3457_345717

theorem houses_in_block (junk_mails_per_house : ℕ) (total_junk_mails_per_block : ℕ) 
  (h1 : junk_mails_per_house = 2) 
  (h2 : total_junk_mails_per_block = 14) : 
  total_junk_mails_per_block / junk_mails_per_house = 7 := by
  sorry

end NUMINAMATH_CALUDE_houses_in_block_l3457_345717


namespace NUMINAMATH_CALUDE_fraction_transformation_l3457_345744

theorem fraction_transformation (p q r s x y : ℝ) 
  (h1 : p ≠ q) 
  (h2 : q ≠ 0) 
  (h3 : y ≠ 0) 
  (h4 : s ≠ y * r) 
  (h5 : (p + x) / (q + y * x) = r / s) : 
  x = (q * r - p * s) / (s - y * r) := by
sorry

end NUMINAMATH_CALUDE_fraction_transformation_l3457_345744


namespace NUMINAMATH_CALUDE_breakfast_egg_scramble_time_l3457_345784

/-- Calculates the time to scramble each egg given the breakfast preparation parameters. -/
def time_to_scramble_egg (num_sausages : ℕ) (num_eggs : ℕ) (time_per_sausage : ℕ) (total_time : ℕ) : ℕ :=
  let time_for_sausages := num_sausages * time_per_sausage
  let time_for_eggs := total_time - time_for_sausages
  time_for_eggs / num_eggs

/-- Proves that the time to scramble each egg is 4 minutes given the specific breakfast parameters. -/
theorem breakfast_egg_scramble_time :
  time_to_scramble_egg 3 6 5 39 = 4 := by
  sorry

end NUMINAMATH_CALUDE_breakfast_egg_scramble_time_l3457_345784


namespace NUMINAMATH_CALUDE_choose_three_from_ten_l3457_345718

theorem choose_three_from_ten (n : ℕ) (k : ℕ) : n = 10 → k = 3 → Nat.choose n k = 120 := by
  sorry

end NUMINAMATH_CALUDE_choose_three_from_ten_l3457_345718


namespace NUMINAMATH_CALUDE_other_root_of_complex_quadratic_l3457_345797

theorem other_root_of_complex_quadratic (z : ℂ) :
  z^2 = -39 - 52*I ∧ z = 5 - 7*I → (-z = -5 + 7*I ∧ (-z)^2 = -39 - 52*I) := by
  sorry

end NUMINAMATH_CALUDE_other_root_of_complex_quadratic_l3457_345797


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3457_345742

/-- A sequence of real numbers. -/
def Sequence := ℕ → ℝ

/-- Predicate for an arithmetic sequence. -/
def IsArithmetic (a : Sequence) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : Sequence) 
  (h1 : IsArithmetic a) 
  (h2 : a 1 + a 6 + a 11 = 3) : 
  a 3 + a 9 = 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3457_345742


namespace NUMINAMATH_CALUDE_cycle_selling_price_l3457_345707

/-- Calculates the final selling price of a cycle given initial cost, profit percentage, discount percentage, and sales tax percentage. -/
def finalSellingPrice (costPrice : ℚ) (profitPercentage : ℚ) (discountPercentage : ℚ) (salesTaxPercentage : ℚ) : ℚ :=
  let markedPrice := costPrice * (1 + profitPercentage / 100)
  let discountedPrice := markedPrice * (1 - discountPercentage / 100)
  discountedPrice * (1 + salesTaxPercentage / 100)

/-- Theorem stating that the final selling price of the cycle is 936.32 given the specified conditions. -/
theorem cycle_selling_price :
  finalSellingPrice 800 10 5 12 = 936.32 := by
  sorry


end NUMINAMATH_CALUDE_cycle_selling_price_l3457_345707


namespace NUMINAMATH_CALUDE_contractor_problem_l3457_345728

/-- Represents the number of days originally planned to complete the work -/
def original_days : ℕ := 9

/-- Represents the number of absent laborers -/
def absent_laborers : ℕ := 10

/-- Represents the number of days taken by the remaining laborers to complete the work -/
def actual_days : ℕ := 18

/-- Represents the total number of laborers originally employed -/
def total_laborers : ℕ := 11

theorem contractor_problem :
  (original_days : ℚ) * (total_laborers - absent_laborers) = actual_days * total_laborers :=
by sorry

end NUMINAMATH_CALUDE_contractor_problem_l3457_345728


namespace NUMINAMATH_CALUDE_part1_part2_l3457_345777

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := Complex.mk (m^2 + m - 6) (m^2 + m - 2)

-- Part 1: Prove that if z - 2m is purely imaginary, then m = 3
theorem part1 (m : ℝ) : (z m - 2 * m).re = 0 → m = 3 := by sorry

-- Part 2: Prove that if z is in the second quadrant, then m is in (-3, -2) ∪ (1, 2)
theorem part2 (m : ℝ) : (z m).re < 0 ∧ (z m).im > 0 → m ∈ Set.Ioo (-3) (-2) ∪ Set.Ioo 1 2 := by sorry

end NUMINAMATH_CALUDE_part1_part2_l3457_345777


namespace NUMINAMATH_CALUDE_distribute_seven_balls_three_boxes_l3457_345700

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- Theorem stating that there are 36 ways to distribute 7 indistinguishable balls into 3 distinguishable boxes -/
theorem distribute_seven_balls_three_boxes : 
  distribute_balls 7 3 = 36 := by sorry

end NUMINAMATH_CALUDE_distribute_seven_balls_three_boxes_l3457_345700


namespace NUMINAMATH_CALUDE_absolute_value_equals_sqrt_square_l3457_345790

theorem absolute_value_equals_sqrt_square (x : ℝ) : |x - 3| = Real.sqrt ((x - 3)^2) := by sorry

end NUMINAMATH_CALUDE_absolute_value_equals_sqrt_square_l3457_345790


namespace NUMINAMATH_CALUDE_bus_passengers_l3457_345712

theorem bus_passengers (men women : ℕ) : 
  women = men / 2 → 
  men - 16 = women + 8 → 
  men + women = 72 :=
by sorry

end NUMINAMATH_CALUDE_bus_passengers_l3457_345712


namespace NUMINAMATH_CALUDE_candy_bar_cost_is_one_l3457_345749

/-- The cost of a candy bar given initial and remaining amounts -/
def candy_bar_cost (initial_amount : ℝ) (remaining_amount : ℝ) : ℝ :=
  initial_amount - remaining_amount

/-- Theorem: The candy bar costs $1 given the conditions -/
theorem candy_bar_cost_is_one :
  let initial_amount : ℝ := 4
  let remaining_amount : ℝ := 3
  candy_bar_cost initial_amount remaining_amount = 1 := by
sorry

end NUMINAMATH_CALUDE_candy_bar_cost_is_one_l3457_345749


namespace NUMINAMATH_CALUDE_simplified_inverse_sum_l3457_345758

theorem simplified_inverse_sum (a b x y : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hx : x ≠ 0) (hy : y ≠ 0) :
  (a * x⁻¹ + b * y⁻¹)⁻¹ = (x * y) / (a * y + b * x) := by
  sorry

end NUMINAMATH_CALUDE_simplified_inverse_sum_l3457_345758


namespace NUMINAMATH_CALUDE_event_attendance_l3457_345773

/-- Given an event with a total of 42 people where the number of children is twice the number of adults,
    prove that the number of children is 28. -/
theorem event_attendance (total : ℕ) (adults : ℕ) (children : ℕ)
    (h1 : total = 42)
    (h2 : total = adults + children)
    (h3 : children = 2 * adults) :
    children = 28 := by
  sorry

end NUMINAMATH_CALUDE_event_attendance_l3457_345773


namespace NUMINAMATH_CALUDE_total_guitars_count_l3457_345798

/-- The number of guitars owned by Davey -/
def daveys_guitars : ℕ := 18

/-- The number of guitars owned by Barbeck -/
def barbecks_guitars : ℕ := daveys_guitars / 3

/-- The number of guitars owned by Steve -/
def steves_guitars : ℕ := barbecks_guitars / 2

/-- The total number of guitars -/
def total_guitars : ℕ := daveys_guitars + barbecks_guitars + steves_guitars

theorem total_guitars_count : total_guitars = 27 := by
  sorry

end NUMINAMATH_CALUDE_total_guitars_count_l3457_345798


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3457_345771

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x : ℝ, y = x^2}
def N : Set ℝ := {y | ∃ x : ℝ, x^2 + y^2 = 1}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = Set.Icc 0 1 := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3457_345771


namespace NUMINAMATH_CALUDE_machine_problem_solution_l3457_345779

-- Define the equation
def machine_equation (y : ℝ) : Prop :=
  1 / (y + 4) + 1 / (y + 3) + 1 / (4 * y) = 1 / y

-- Theorem statement
theorem machine_problem_solution :
  ∃ y : ℝ, y > 0 ∧ machine_equation y ∧ y = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_machine_problem_solution_l3457_345779


namespace NUMINAMATH_CALUDE_remaining_pills_l3457_345748

/-- Calculates the total number of pills left after using supplements for a specified number of days. -/
def pillsLeft (largeBottles smallBottles : ℕ) (largePillCount smallPillCount daysUsed : ℕ) : ℕ :=
  (largeBottles * (largePillCount - daysUsed)) + (smallBottles * (smallPillCount - daysUsed))

/-- Theorem stating that given the specific supplement configuration and usage, 350 pills remain. -/
theorem remaining_pills :
  pillsLeft 3 2 120 30 14 = 350 := by
  sorry

end NUMINAMATH_CALUDE_remaining_pills_l3457_345748


namespace NUMINAMATH_CALUDE_power_function_exponent_l3457_345731

/-- A power function passing through (1/4, 1/2) has exponent 1/2 -/
theorem power_function_exponent (m : ℝ) (a : ℝ) :
  m * (1/4 : ℝ)^a = 1/2 → a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_power_function_exponent_l3457_345731


namespace NUMINAMATH_CALUDE_cylinder_volume_with_square_perimeter_l3457_345729

theorem cylinder_volume_with_square_perimeter (h : ℝ) (h_pos : h > 0) :
  let square_area : ℝ := 121
  let square_side : ℝ := Real.sqrt square_area
  let square_perimeter : ℝ := 4 * square_side
  let cylinder_radius : ℝ := square_perimeter / (2 * Real.pi)
  let cylinder_volume : ℝ := Real.pi * cylinder_radius^2 * h
  cylinder_volume = (484 / Real.pi) * h := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_with_square_perimeter_l3457_345729


namespace NUMINAMATH_CALUDE_probability_two_eight_sided_dice_less_than_three_l3457_345770

def roll_two_dice (n : ℕ) : ℕ := n * n

def outcomes_both_greater_equal (n : ℕ) (k : ℕ) : ℕ := (n - k + 1) * (n - k + 1)

def probability_at_least_one_less_than (n : ℕ) (k : ℕ) : ℚ :=
  (roll_two_dice n - outcomes_both_greater_equal n k) / roll_two_dice n

theorem probability_two_eight_sided_dice_less_than_three :
  probability_at_least_one_less_than 8 3 = 7/16 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_eight_sided_dice_less_than_three_l3457_345770


namespace NUMINAMATH_CALUDE_interview_probability_correct_l3457_345760

structure TouristGroup where
  total : ℕ
  outside_fraction : ℚ
  inside_fraction : ℚ
  gold_fraction : ℚ
  silver_fraction : ℚ

def interview_probability (group : TouristGroup) : ℚ × ℚ :=
  let outside := (group.total : ℚ) * group.outside_fraction
  let inside := (group.total : ℚ) * group.inside_fraction
  let gold := outside * group.gold_fraction
  let silver := inside * group.silver_fraction
  let no_card := group.total - (gold + silver)
  let prob_one_silver := (silver * (group.total - silver)) / ((group.total * (group.total - 1)) / 2)
  let prob_equal := (((no_card * (no_card - 1)) / 2) + gold * silver) / ((group.total * (group.total - 1)) / 2)
  (prob_one_silver, prob_equal)

theorem interview_probability_correct (group : TouristGroup) 
  (h1 : group.total = 36)
  (h2 : group.outside_fraction = 3/4)
  (h3 : group.inside_fraction = 1/4)
  (h4 : group.gold_fraction = 1/3)
  (h5 : group.silver_fraction = 2/3) :
  interview_probability group = (2/7, 44/105) := by
  sorry

end NUMINAMATH_CALUDE_interview_probability_correct_l3457_345760


namespace NUMINAMATH_CALUDE_triangle_side_length_l3457_345786

theorem triangle_side_length (a b c : ℝ) (area : ℝ) : 
  a = 1 → b = Real.sqrt 7 → area = Real.sqrt 3 / 2 → 
  (c = 2 ∨ c = 2 * Real.sqrt 3) := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3457_345786


namespace NUMINAMATH_CALUDE_area_ratio_of_nested_squares_l3457_345736

-- Define the squares
structure Square where
  sideLength : ℝ

-- Define the relationship between the squares
structure SquareRelationship where
  outerSquare : Square
  innerSquare : Square
  vertexRatio : ℝ

-- Theorem statement
theorem area_ratio_of_nested_squares (sr : SquareRelationship) 
  (h1 : sr.outerSquare.sideLength = 16)
  (h2 : sr.vertexRatio = 3/4) : 
  (sr.innerSquare.sideLength^2) / (sr.outerSquare.sideLength^2) = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_of_nested_squares_l3457_345736


namespace NUMINAMATH_CALUDE_alcohol_fraction_after_water_increase_l3457_345721

theorem alcohol_fraction_after_water_increase (v : ℝ) (h : v > 0) :
  let initial_alcohol := (2 / 3) * v
  let initial_water := (1 / 3) * v
  let new_water := 3 * initial_water
  let new_total := initial_alcohol + new_water
  initial_alcohol / new_total = 2 / 5 := by
sorry

end NUMINAMATH_CALUDE_alcohol_fraction_after_water_increase_l3457_345721


namespace NUMINAMATH_CALUDE_max_dot_product_l3457_345739

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the center and left focus
def O : ℝ × ℝ := (0, 0)
def F : ℝ × ℝ := (-1, 0)

-- Define the dot product of OF and OP
def dot_product (x y : ℝ) : ℝ := (x + 1) * x + y * y

-- Theorem statement
theorem max_dot_product :
  ∀ x y : ℝ, is_on_ellipse x y →
  ∀ x' y' : ℝ, is_on_ellipse x' y' →
  dot_product x y ≤ 6 :=
sorry

end NUMINAMATH_CALUDE_max_dot_product_l3457_345739


namespace NUMINAMATH_CALUDE_asian_art_pieces_l3457_345759

theorem asian_art_pieces (total : ℕ) (egyptian : ℕ) (asian : ℕ) 
  (h1 : total = 992) 
  (h2 : egyptian = 527) 
  (h3 : total = egyptian + asian) : 
  asian = 465 := by
sorry

end NUMINAMATH_CALUDE_asian_art_pieces_l3457_345759


namespace NUMINAMATH_CALUDE_bird_percentage_problem_l3457_345775

theorem bird_percentage_problem :
  ∀ (total : ℝ) (sparrows pigeons crows parrots : ℝ),
    sparrows = 0.4 * total →
    pigeons = 0.2 * total →
    crows = 0.15 * total →
    parrots = total - (sparrows + pigeons + crows) →
    (crows / (total - pigeons)) * 100 = 18.75 := by
  sorry

end NUMINAMATH_CALUDE_bird_percentage_problem_l3457_345775


namespace NUMINAMATH_CALUDE_urn_probability_l3457_345791

/-- Represents the total number of chips in the urn -/
def total_chips : ℕ := 15

/-- Represents the number of chips of each color -/
def chips_per_color : ℕ := 5

/-- Represents the number of colors -/
def num_colors : ℕ := 3

/-- Represents the number of chips with each number -/
def chips_per_number : ℕ := 3

/-- Represents the number of different numbers on the chips -/
def num_numbers : ℕ := 5

/-- The probability of drawing two chips with either the same color or the same number -/
theorem urn_probability : 
  (num_colors * (chips_per_color.choose 2) + num_numbers * (chips_per_number.choose 2)) / (total_chips.choose 2) = 3 / 7 :=
by sorry

end NUMINAMATH_CALUDE_urn_probability_l3457_345791


namespace NUMINAMATH_CALUDE_problem_solution_l3457_345794

theorem problem_solution : ∃! x : ℝ, (0.8 * x) = ((4 / 5) * 25 + 28) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3457_345794


namespace NUMINAMATH_CALUDE_units_digit_of_17_to_2107_l3457_345781

theorem units_digit_of_17_to_2107 :
  (17^2107 : ℕ) % 10 = 3 :=
sorry

end NUMINAMATH_CALUDE_units_digit_of_17_to_2107_l3457_345781


namespace NUMINAMATH_CALUDE_num_routes_eq_expected_l3457_345710

/-- Represents the number of southern cities -/
def num_southern_cities : ℕ := 4

/-- Represents the number of northern cities -/
def num_northern_cities : ℕ := 5

/-- Calculates the number of different routes for a traveler -/
def num_routes : ℕ := (Nat.factorial (num_southern_cities - 1)) * (num_northern_cities ^ num_southern_cities)

/-- Theorem stating that the number of routes is equal to 3! × 5^4 -/
theorem num_routes_eq_expected : num_routes = 3750 := by
  sorry

end NUMINAMATH_CALUDE_num_routes_eq_expected_l3457_345710


namespace NUMINAMATH_CALUDE_no_solution_quadratic_inequality_l3457_345738

theorem no_solution_quadratic_inequality :
  ¬∃ (x : ℝ), 2 - 3*x + 2*x^2 ≤ 0 := by
sorry

end NUMINAMATH_CALUDE_no_solution_quadratic_inequality_l3457_345738


namespace NUMINAMATH_CALUDE_inverse_function_c_value_l3457_345789

/-- Given a function f and its inverse, prove the value of c -/
theorem inverse_function_c_value 
  (f : ℝ → ℝ) 
  (c : ℝ) 
  (h1 : ∀ x, f x = 1 / (3 * x + c)) 
  (h2 : ∀ x, Function.invFun f x = (2 - 3 * x) / (3 * x)) : 
  c = 1 := by
sorry

end NUMINAMATH_CALUDE_inverse_function_c_value_l3457_345789


namespace NUMINAMATH_CALUDE_island_liars_count_l3457_345766

/-- Represents the types of inhabitants on the island -/
inductive Inhabitant
  | Knight
  | Liar

/-- The total number of inhabitants on the island -/
def total_inhabitants : Nat := 2001

/-- A function that returns true if the statement "more than half of the others are liars" is true -/
def more_than_half_others_are_liars (num_liars : Nat) : Prop :=
  num_liars > (total_inhabitants - 1) / 2

/-- A function that determines if an inhabitant's statement is consistent with their type -/
def consistent_statement (inhabitant : Inhabitant) (num_liars : Nat) : Prop :=
  match inhabitant with
  | Inhabitant.Knight => more_than_half_others_are_liars num_liars
  | Inhabitant.Liar => ¬(more_than_half_others_are_liars num_liars)

theorem island_liars_count :
  ∃ (num_liars : Nat),
    num_liars ≤ total_inhabitants ∧
    (∀ (i : Inhabitant), consistent_statement i num_liars) ∧
    num_liars = 1001 := by
  sorry

end NUMINAMATH_CALUDE_island_liars_count_l3457_345766


namespace NUMINAMATH_CALUDE_candy_problem_l3457_345788

/-- The number of candies left in Shelly's bowl before her friend came over -/
def initial_candies : ℕ := 63

/-- The number of candies Shelly's friend brought -/
def friend_candies : ℕ := 2 * initial_candies

/-- The total number of candies after the friend's contribution -/
def total_candies : ℕ := initial_candies + friend_candies

/-- The number of candies Shelly's friend had after eating 10 -/
def friend_final_candies : ℕ := 85

theorem candy_problem :
  initial_candies = 63 ∧
  friend_candies = 2 * initial_candies ∧
  total_candies = initial_candies + friend_candies ∧
  friend_final_candies + 10 = total_candies / 2 :=
sorry

end NUMINAMATH_CALUDE_candy_problem_l3457_345788


namespace NUMINAMATH_CALUDE_break_even_price_per_lot_l3457_345785

/-- Given a land purchase scenario, calculate the break-even price per lot -/
theorem break_even_price_per_lot (acres : ℕ) (price_per_acre : ℕ) (num_lots : ℕ) :
  acres = 4 →
  price_per_acre = 1863 →
  num_lots = 9 →
  (acres * price_per_acre) / num_lots = 828 := by
  sorry

end NUMINAMATH_CALUDE_break_even_price_per_lot_l3457_345785


namespace NUMINAMATH_CALUDE_square_area_from_rectangle_l3457_345747

theorem square_area_from_rectangle (rectangle_area : ℝ) (rectangle_breadth : ℝ) : 
  rectangle_area = 100 →
  rectangle_breadth = 10 →
  ∃ (circle_radius : ℝ),
    (2 / 5 : ℝ) * circle_radius * rectangle_breadth = rectangle_area →
    circle_radius ^ 2 = 625 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_rectangle_l3457_345747


namespace NUMINAMATH_CALUDE_fuel_station_total_cost_l3457_345735

/-- Calculates the total cost for filling up vehicles at a fuel station -/
def total_cost (service_cost : ℝ) 
                (minivan_price minivan_capacity : ℝ) 
                (pickup_price pickup_capacity : ℝ)
                (semitruck_price : ℝ)
                (minivan_count pickup_count semitruck_count : ℕ) : ℝ :=
  let semitruck_capacity := pickup_capacity * 2.2
  let minivan_total := (service_cost + minivan_price * minivan_capacity) * minivan_count
  let pickup_total := (service_cost + pickup_price * pickup_capacity) * pickup_count
  let semitruck_total := (service_cost + semitruck_price * semitruck_capacity) * semitruck_count
  minivan_total + pickup_total + semitruck_total

/-- The total cost for filling up 4 mini-vans, 2 pick-up trucks, and 3 semi-trucks is $998.80 -/
theorem fuel_station_total_cost : 
  total_cost 2.20 0.70 65 0.85 100 0.95 4 2 3 = 998.80 := by
  sorry

end NUMINAMATH_CALUDE_fuel_station_total_cost_l3457_345735


namespace NUMINAMATH_CALUDE_hyperbola_equation_final_hyperbola_equation_l3457_345743

/-- The standard equation of a hyperbola given specific conditions -/
theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (∃ (C₁ : ℝ → ℝ → Prop) (C₂ : ℝ → ℝ → Prop),
    (∀ x y, C₁ x y ↔ x^2 = 2*y) ∧ 
    (∀ x y, C₂ x y ↔ x^2/a^2 - y^2/b^2 = 1) ∧
    (∃ A : ℝ × ℝ, A.1 = a ∧ A.2 = 0 ∧ C₂ A.1 A.2) ∧
    (a^2 + b^2 = 5*a^2) ∧
    (∃ l : ℝ → ℝ, (∀ x, l x = b/a*(x - a)) ∧
      (∀ x, C₁ x (l x) → (∃! y, C₁ x y ∧ y = l x)))) →
  a = 1 ∧ b = 2 :=
by sorry

/-- The final form of the hyperbola equation -/
theorem final_hyperbola_equation :
  ∃ (C : ℝ → ℝ → Prop), ∀ x y, C x y ↔ x^2 - y^2/4 = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_final_hyperbola_equation_l3457_345743


namespace NUMINAMATH_CALUDE_post_office_distance_l3457_345726

/-- Proves that the distance of a round trip is 20 km given specific speeds and total time -/
theorem post_office_distance (outbound_speed inbound_speed : ℝ) (total_time : ℝ) 
  (h1 : outbound_speed = 25)
  (h2 : inbound_speed = 4)
  (h3 : total_time = 5.8) :
  let distance := (outbound_speed * inbound_speed * total_time) / (outbound_speed + inbound_speed)
  distance = 20 := by
  sorry

end NUMINAMATH_CALUDE_post_office_distance_l3457_345726


namespace NUMINAMATH_CALUDE_only_two_consecutive_primes_l3457_345715

theorem only_two_consecutive_primes : ∀ p : ℕ, 
  (Nat.Prime p ∧ Nat.Prime (p + 1)) → p = 2 := by
  sorry

end NUMINAMATH_CALUDE_only_two_consecutive_primes_l3457_345715


namespace NUMINAMATH_CALUDE_danielles_apartment_rooms_l3457_345722

theorem danielles_apartment_rooms (heidi_rooms danielle_rooms grant_rooms : ℕ) : 
  heidi_rooms = 3 * danielle_rooms →
  grant_rooms = heidi_rooms / 9 →
  grant_rooms = 2 →
  danielle_rooms = 6 := by
sorry

end NUMINAMATH_CALUDE_danielles_apartment_rooms_l3457_345722


namespace NUMINAMATH_CALUDE_product_pure_imaginary_l3457_345782

theorem product_pure_imaginary (b : ℝ) : 
  let Z1 : ℂ := 3 - 4*I
  let Z2 : ℂ := 4 + b*I
  (∃ (y : ℝ), Z1 * Z2 = y*I) → b = -3 := by
sorry

end NUMINAMATH_CALUDE_product_pure_imaginary_l3457_345782


namespace NUMINAMATH_CALUDE_expression_evaluation_l3457_345727

theorem expression_evaluation :
  let a : ℤ := -1
  let b : ℤ := 2
  (7 * a^2 * b + (-4 * a^2 * b + 5 * a * b^2) - (2 * a^2 * b - 3 * a * b^2)) = -30 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3457_345727


namespace NUMINAMATH_CALUDE_abs_sum_inequality_l3457_345765

theorem abs_sum_inequality (k : ℝ) :
  (∀ x : ℝ, |x + 2| + |x + 1| > k) → k < 1 := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_inequality_l3457_345765


namespace NUMINAMATH_CALUDE_anthony_ate_two_bananas_l3457_345706

/-- The number of bananas Anthony bought -/
def initial_bananas : ℕ := 12

/-- The number of bananas Anthony has left -/
def remaining_bananas : ℕ := 10

/-- The number of bananas Anthony ate -/
def eaten_bananas : ℕ := initial_bananas - remaining_bananas

theorem anthony_ate_two_bananas : eaten_bananas = 2 := by
  sorry

end NUMINAMATH_CALUDE_anthony_ate_two_bananas_l3457_345706


namespace NUMINAMATH_CALUDE_spending_difference_is_131_75_l3457_345769

/-- Calculates the difference in spending between Coach A and Coach B -/
def spending_difference : ℝ :=
  let coach_a_basketball_cost : ℝ := 10 * 29
  let coach_a_soccer_ball_cost : ℝ := 5 * 15
  let coach_a_total_before_discount : ℝ := coach_a_basketball_cost + coach_a_soccer_ball_cost
  let coach_a_discount : ℝ := 0.05 * coach_a_total_before_discount
  let coach_a_total : ℝ := coach_a_total_before_discount - coach_a_discount

  let coach_b_baseball_cost : ℝ := 14 * 2.5
  let coach_b_baseball_bat_cost : ℝ := 18
  let coach_b_hockey_stick_cost : ℝ := 4 * 25
  let coach_b_hockey_mask_cost : ℝ := 72
  let coach_b_total_before_discount : ℝ := coach_b_baseball_cost + coach_b_baseball_bat_cost + 
                                           coach_b_hockey_stick_cost + coach_b_hockey_mask_cost
  let coach_b_discount : ℝ := 10
  let coach_b_total : ℝ := coach_b_total_before_discount - coach_b_discount

  coach_a_total - coach_b_total

/-- The theorem states that the difference in spending between Coach A and Coach B is $131.75 -/
theorem spending_difference_is_131_75 : spending_difference = 131.75 := by
  sorry

end NUMINAMATH_CALUDE_spending_difference_is_131_75_l3457_345769


namespace NUMINAMATH_CALUDE_dihedral_angle_segment_length_l3457_345752

/-- Given a dihedral angle of 120°, this theorem calculates the length of the segment
    connecting the ends of two perpendiculars drawn from the ends of a segment on the edge
    of the dihedral angle. -/
theorem dihedral_angle_segment_length 
  (a b c : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : c > 0) : 
  ∃ (length : ℝ), length = Real.sqrt (a^2 + b^2 + a*b + c^2) := by
sorry

end NUMINAMATH_CALUDE_dihedral_angle_segment_length_l3457_345752
