import Mathlib

namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l2493_249326

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℚ) : Prop :=
  ∃ r : ℚ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_first_term
  (a : ℕ → ℚ)
  (h_geometric : IsGeometricSequence a)
  (h_term2 : a 2 = 12)
  (h_term3 : a 3 = 24)
  (h_term6 : a 6 = 384) :
  a 1 = 3/4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l2493_249326


namespace NUMINAMATH_CALUDE_m_range_l2493_249338

theorem m_range (a : ℝ) (m : ℝ) 
  (h1 : 0 < a ∧ a < 1)
  (h2 : ∀ x : ℝ, x^2 - x - 2*a > 0 ↔ -1 < x ∧ x < 2)
  (h3 : ∀ x : ℝ, (1/a)^(x^2 + 2*m*x - m) ≥ 1) :
  -1 ≤ m ∧ m ≤ 0 := by
sorry

end NUMINAMATH_CALUDE_m_range_l2493_249338


namespace NUMINAMATH_CALUDE_figure_100_squares_l2493_249300

def f (n : ℕ) : ℕ := n^3 + 2*n^2 + 2*n + 1

theorem figure_100_squares :
  f 0 = 1 ∧ f 1 = 6 ∧ f 2 = 20 ∧ f 3 = 50 → f 100 = 1020201 := by
  sorry

end NUMINAMATH_CALUDE_figure_100_squares_l2493_249300


namespace NUMINAMATH_CALUDE_original_equals_scientific_l2493_249306

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The original number to be represented -/
def original_number : ℝ := 43050000

/-- The scientific notation representation -/
def scientific_repr : ScientificNotation :=
  { coefficient := 4.305,
    exponent := 7,
    h1 := by sorry }

/-- Theorem stating that the original number equals its scientific notation representation -/
theorem original_equals_scientific :
  original_number = scientific_repr.coefficient * (10 : ℝ) ^ scientific_repr.exponent :=
by sorry

end NUMINAMATH_CALUDE_original_equals_scientific_l2493_249306


namespace NUMINAMATH_CALUDE_chucks_team_leads_l2493_249315

/-- Represents a team's scoring in a single quarter -/
structure QuarterScore where
  fieldGoals : ℕ
  threePointers : ℕ
  freeThrows : ℕ

/-- Calculates the total points for a quarter -/
def quarterPoints (qs : QuarterScore) : ℕ :=
  2 * qs.fieldGoals + 3 * qs.threePointers + qs.freeThrows

/-- Represents a team's scoring for the entire game -/
structure GameScore where
  q1 : QuarterScore
  q2 : QuarterScore
  q3 : QuarterScore
  q4 : QuarterScore
  technicalFouls : ℕ

/-- Calculates the total points for a team in the game -/
def totalPoints (gs : GameScore) : ℕ :=
  quarterPoints gs.q1 + quarterPoints gs.q2 + quarterPoints gs.q3 + quarterPoints gs.q4 + gs.technicalFouls

theorem chucks_team_leads :
  let chucksTeam : GameScore := {
    q1 := { fieldGoals := 9, threePointers := 0, freeThrows := 5 },
    q2 := { fieldGoals := 6, threePointers := 3, freeThrows := 0 },
    q3 := { fieldGoals := 4, threePointers := 2, freeThrows := 6 },
    q4 := { fieldGoals := 8, threePointers := 1, freeThrows := 0 },
    technicalFouls := 3
  }
  let yellowTeam : GameScore := {
    q1 := { fieldGoals := 7, threePointers := 4, freeThrows := 0 },
    q2 := { fieldGoals := 5, threePointers := 2, freeThrows := 3 },
    q3 := { fieldGoals := 6, threePointers := 2, freeThrows := 0 },
    q4 := { fieldGoals := 4, threePointers := 3, freeThrows := 2 },
    technicalFouls := 2
  }
  totalPoints chucksTeam - totalPoints yellowTeam = 2 := by
  sorry

end NUMINAMATH_CALUDE_chucks_team_leads_l2493_249315


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_15_l2493_249341

/-- An arithmetic sequence with common difference d ≠ 0 -/
def arithmetic_sequence (a : ℕ → ℚ) (d : ℚ) : Prop :=
  d ≠ 0 ∧ ∀ n, a (n + 1) = a n + d

/-- The sum of the first n terms of an arithmetic sequence -/
def sum_arithmetic (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (a 1 + a n) / 2

theorem arithmetic_sequence_sum_15 
  (a : ℕ → ℚ) (d : ℚ) 
  (h_arith : arithmetic_sequence a d)
  (h_eq : a 5^2 + a 7^2 + 16*d = a 9^2 + a 11^2) :
  sum_arithmetic a 15 = 15 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_15_l2493_249341


namespace NUMINAMATH_CALUDE_difference_C₁_C₂_l2493_249303

/-- Triangle ABC with given angle measures and altitude from C --/
structure TriangleABC where
  A : ℝ
  B : ℝ
  C : ℝ
  C₁ : ℝ
  C₂ : ℝ
  angleA_eq : A = 30
  angleB_eq : B = 70
  sum_angles : A + B + C = 180
  C_split : C = C₁ + C₂
  right_angle_AC₁ : A + C₁ + 90 = 180
  right_angle_BC₂ : B + C₂ + 90 = 180

/-- Theorem: In the given triangle, C₁ - C₂ = 40° --/
theorem difference_C₁_C₂ (t : TriangleABC) : t.C₁ - t.C₂ = 40 := by
  sorry

end NUMINAMATH_CALUDE_difference_C₁_C₂_l2493_249303


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l2493_249391

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + 8 * y = x * y) :
  x + y ≥ 18 ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x + 8 * y = x * y ∧ x + y = 18 := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l2493_249391


namespace NUMINAMATH_CALUDE_tan_alpha_minus_pi_over_four_l2493_249368

theorem tan_alpha_minus_pi_over_four (α β : ℝ) 
  (h1 : Real.tan (α + β) = 2 / 5)
  (h2 : Real.tan β = 1 / 3) :
  Real.tan (α - π / 4) = - 8 / 9 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_minus_pi_over_four_l2493_249368


namespace NUMINAMATH_CALUDE_number_division_problem_l2493_249317

theorem number_division_problem (x y : ℝ) 
  (h1 : (x - 5) / 7 = 7)
  (h2 : (x - 14) / y = 4) : 
  y = 10 := by
sorry

end NUMINAMATH_CALUDE_number_division_problem_l2493_249317


namespace NUMINAMATH_CALUDE_math_problems_sum_l2493_249369

/-- The sum of four math problem answers given specific conditions -/
theorem math_problems_sum : 
  let answer1 : ℝ := 600
  let answer2 : ℝ := 2 * answer1
  let answer3 : ℝ := answer1 + answer2 - 400
  let answer4 : ℝ := (answer1 + answer2 + answer3) / 3
  (answer1 + answer2 + answer3 + answer4) = 4266.67 := by
  sorry

end NUMINAMATH_CALUDE_math_problems_sum_l2493_249369


namespace NUMINAMATH_CALUDE_square_circle_overlap_ratio_l2493_249361

theorem square_circle_overlap_ratio (r : ℝ) (h : r > 0) :
  let circle_area := π * r^2
  let square_side := 2 * r
  let overlap_area := square_side^2
  overlap_area / circle_area = 4 / π :=
by sorry

end NUMINAMATH_CALUDE_square_circle_overlap_ratio_l2493_249361


namespace NUMINAMATH_CALUDE_product_xy_l2493_249394

theorem product_xy (x y : ℚ) 
  (h1 : 1/x + 1/y = 4) 
  (h2 : 1/x - 1/y = -6) : 
  x * y = -1/5 := by
sorry

end NUMINAMATH_CALUDE_product_xy_l2493_249394


namespace NUMINAMATH_CALUDE_equal_coin_count_l2493_249381

def coin_count (t : ℕ) : ℕ := t / 3

theorem equal_coin_count (t : ℕ) (h : t % 3 = 0) :
  let one_dollar_count := coin_count t
  let two_dollar_count := coin_count t
  one_dollar_count * 1 + two_dollar_count * 2 = t ∧
  one_dollar_count = two_dollar_count :=
by sorry

end NUMINAMATH_CALUDE_equal_coin_count_l2493_249381


namespace NUMINAMATH_CALUDE_det_transformation_l2493_249314

/-- Given a 2x2 matrix with determinant 7, prove that the determinant of a related matrix is also 7 -/
theorem det_transformation (p q r s : ℝ) (h : Matrix.det !![p, q; r, s] = 7) :
  Matrix.det !![p + 2*r, q + 2*s; r, s] = 7 := by
  sorry

end NUMINAMATH_CALUDE_det_transformation_l2493_249314


namespace NUMINAMATH_CALUDE_right_triangle_median_intersection_l2493_249337

theorem right_triangle_median_intersection (a b c d : ℝ) (m : ℝ) : 
  let P := (a, b)
  let Q := (a, b + 2*c)
  let R := (a + 2*d, b)
  let midpoint_x := (a, b + c)
  let midpoint_y := (a + d, b)
  let median_slope_x := c / d
  let median_slope_y := 2 * c / d
  median_slope_x = 3 ∧ 
  median_slope_y = 2*m + 1 →
  ∃! m : ℝ, m = 5/2 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_median_intersection_l2493_249337


namespace NUMINAMATH_CALUDE_shortest_ribbon_length_l2493_249375

theorem shortest_ribbon_length (ribbon_length : ℕ) : 
  (ribbon_length % 2 = 0) ∧ 
  (ribbon_length % 5 = 0) ∧ 
  (ribbon_length % 7 = 0) ∧ 
  (∀ x : ℕ, x < ribbon_length → (x % 2 = 0 ∧ x % 5 = 0 ∧ x % 7 = 0) → False) → 
  ribbon_length = 70 := by
sorry

end NUMINAMATH_CALUDE_shortest_ribbon_length_l2493_249375


namespace NUMINAMATH_CALUDE_currency_notes_count_l2493_249308

theorem currency_notes_count 
  (total_amount : ℕ) 
  (denomination_1 : ℕ) 
  (denomination_2 : ℕ) 
  (count_denomination_2 : ℕ) : 
  total_amount = 5000 ∧ 
  denomination_1 = 95 ∧ 
  denomination_2 = 45 ∧ 
  count_denomination_2 = 71 → 
  ∃ (count_denomination_1 : ℕ), 
    count_denomination_1 * denomination_1 + count_denomination_2 * denomination_2 = total_amount ∧ 
    count_denomination_1 + count_denomination_2 = 90 :=
by sorry

end NUMINAMATH_CALUDE_currency_notes_count_l2493_249308


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l2493_249362

theorem cubic_roots_sum (a b c : ℝ) : 
  (0 < a ∧ a < 1) → 
  (0 < b ∧ b < 1) → 
  (0 < c ∧ c < 1) → 
  a ≠ b → b ≠ c → a ≠ c →
  40 * a^3 - 70 * a^2 + 32 * a - 3 = 0 →
  40 * b^3 - 70 * b^2 + 32 * b - 3 = 0 →
  40 * c^3 - 70 * c^2 + 32 * c - 3 = 0 →
  1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c) = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l2493_249362


namespace NUMINAMATH_CALUDE_largest_digit_sum_l2493_249357

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

theorem largest_digit_sum (a b c y : ℕ) : 
  is_digit a → is_digit b → is_digit c →
  (a * 100 + b * 10 + c : ℚ) / 1000 = 1 / y →
  0 < y → y ≤ 10 →
  ∃ (a' b' c' : ℕ), is_digit a' ∧ is_digit b' ∧ is_digit c' ∧
    (a' * 100 + b' * 10 + c' : ℚ) / 1000 = 1 / y ∧
    a' + b' + c' ≤ 8 ∧
    a + b + c ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_largest_digit_sum_l2493_249357


namespace NUMINAMATH_CALUDE_smallest_m_theorem_l2493_249397

def is_multiple_of_100 (n : ℕ) : Prop := ∃ k : ℕ, n = 100 * k

def count_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

def satisfies_conditions (m : ℕ) : Prop :=
  is_multiple_of_100 m ∧ count_divisors m = 100

theorem smallest_m_theorem :
  ∃! m : ℕ, satisfies_conditions m ∧
    ∀ n : ℕ, satisfies_conditions n → m ≤ n ∧
    m / 100 = 2700 := by sorry

end NUMINAMATH_CALUDE_smallest_m_theorem_l2493_249397


namespace NUMINAMATH_CALUDE_sample_xy_product_l2493_249377

theorem sample_xy_product (x y : ℝ) : 
  (9 + 10 + 11 + x + y) / 5 = 10 →
  ((9 - 10)^2 + (10 - 10)^2 + (11 - 10)^2 + (x - 10)^2 + (y - 10)^2) / 5 = 2 →
  x * y = 96 := by
sorry

end NUMINAMATH_CALUDE_sample_xy_product_l2493_249377


namespace NUMINAMATH_CALUDE_number_plus_sqrt_equals_24_l2493_249393

theorem number_plus_sqrt_equals_24 : ∃ x : ℝ, x + Real.sqrt (-4 + 6 * 4 * 3) = 24 := by
  sorry

end NUMINAMATH_CALUDE_number_plus_sqrt_equals_24_l2493_249393


namespace NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l2493_249390

theorem complex_number_in_fourth_quadrant :
  let z : ℂ := (Complex.I ^ 2016) / (3 + 2 * Complex.I)
  (z.re > 0) ∧ (z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l2493_249390


namespace NUMINAMATH_CALUDE_average_speed_calculation_l2493_249385

/-- Calculate the average speed of a car given its speeds in two consecutive hours -/
theorem average_speed_calculation (speed1 speed2 : ℝ) (h1 : speed1 = 90) (h2 : speed2 = 55) :
  (speed1 + speed2) / 2 = 72.5 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l2493_249385


namespace NUMINAMATH_CALUDE_james_earnings_l2493_249335

/-- James' earnings problem -/
theorem james_earnings (january : ℕ) (february : ℕ) (march : ℕ) 
  (h1 : february = 2 * january)
  (h2 : march = february - 2000)
  (h3 : january + february + march = 18000) :
  january = 4000 := by
  sorry

end NUMINAMATH_CALUDE_james_earnings_l2493_249335


namespace NUMINAMATH_CALUDE_intersection_with_complement_l2493_249395

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {2, 3}

theorem intersection_with_complement : A ∩ (U \ B) = {1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l2493_249395


namespace NUMINAMATH_CALUDE_factorization_proof_l2493_249336

theorem factorization_proof (a : ℝ) : (2*a + 1)*a - 4*a - 2 = (2*a + 1)*(a - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l2493_249336


namespace NUMINAMATH_CALUDE_square_triangle_equal_area_l2493_249358

theorem square_triangle_equal_area (square_perimeter : ℝ) (triangle_height : ℝ) :
  square_perimeter = 80 →
  triangle_height = 40 →
  (square_perimeter / 4)^2 = (1/2) * triangle_height * (square_perimeter / 4) →
  (square_perimeter / 4) = 20 :=
by sorry

end NUMINAMATH_CALUDE_square_triangle_equal_area_l2493_249358


namespace NUMINAMATH_CALUDE_rental_cost_difference_theorem_l2493_249399

/-- Calculates the rental cost difference between a ski boat and a sailboat --/
def rental_cost_difference (
  sailboat_weekday_cost : ℕ)
  (skiboat_weekend_hourly_cost : ℕ)
  (sailboat_fuel_cost_per_hour : ℕ)
  (skiboat_fuel_cost_per_hour : ℕ)
  (rental_hours_per_day : ℕ)
  (rental_days : ℕ)
  (discount_percentage : ℕ) : ℕ :=
  let sailboat_day1_cost := sailboat_weekday_cost + sailboat_fuel_cost_per_hour * rental_hours_per_day
  let sailboat_day2_cost := (sailboat_weekday_cost * (100 - discount_percentage) / 100) + sailboat_fuel_cost_per_hour * rental_hours_per_day
  let sailboat_total_cost := sailboat_day1_cost + sailboat_day2_cost

  let skiboat_day1_cost := skiboat_weekend_hourly_cost * rental_hours_per_day + skiboat_fuel_cost_per_hour * rental_hours_per_day
  let skiboat_day2_cost := (skiboat_weekend_hourly_cost * rental_hours_per_day * (100 - discount_percentage) / 100) + skiboat_fuel_cost_per_hour * rental_hours_per_day
  let skiboat_total_cost := skiboat_day1_cost + skiboat_day2_cost

  skiboat_total_cost - sailboat_total_cost

theorem rental_cost_difference_theorem :
  rental_cost_difference 60 120 10 20 3 2 10 = 630 := by
  sorry

end NUMINAMATH_CALUDE_rental_cost_difference_theorem_l2493_249399


namespace NUMINAMATH_CALUDE_jenna_reading_schedule_l2493_249318

/-- Represents Jenna's reading schedule for September --/
structure ReadingSchedule where
  total_days : Nat
  total_pages : Nat
  busy_days : Nat
  special_day_pages : Nat

/-- Calculates the number of pages Jenna needs to read per day on regular reading days --/
def pages_per_day (schedule : ReadingSchedule) : Nat :=
  let regular_reading_days := schedule.total_days - schedule.busy_days - 1
  let regular_pages := schedule.total_pages - schedule.special_day_pages
  regular_pages / regular_reading_days

/-- Theorem stating that Jenna needs to read 20 pages per day on regular reading days --/
theorem jenna_reading_schedule :
  let schedule := ReadingSchedule.mk 30 600 4 100
  pages_per_day schedule = 20 := by
  sorry

end NUMINAMATH_CALUDE_jenna_reading_schedule_l2493_249318


namespace NUMINAMATH_CALUDE_evaluate_power_l2493_249310

-- Define the problem
theorem evaluate_power : (81 : ℝ) ^ (11/4) = 177147 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_power_l2493_249310


namespace NUMINAMATH_CALUDE_alpha_squared_gt_beta_squared_l2493_249356

theorem alpha_squared_gt_beta_squared 
  (α β : Real) 
  (h1 : α ∈ Set.Icc (-π/2) (π/2)) 
  (h2 : β ∈ Set.Icc (-π/2) (π/2)) 
  (h3 : α * Real.sin α - β * Real.sin β > 0) : 
  α^2 > β^2 :=
sorry

end NUMINAMATH_CALUDE_alpha_squared_gt_beta_squared_l2493_249356


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l2493_249351

/-- The eccentricity of an ellipse with specific geometric properties -/
theorem ellipse_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  let e := Real.sqrt ((a^2 - b^2) / a^2)
  let AF := a - (e * a)
  let AB := Real.sqrt (a^2 + b^2)
  let BF := a
  (∃ (r : ℝ), AF * r = AB ∧ AB * r = 3 * BF) →
  e = (Real.sqrt 5 - 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l2493_249351


namespace NUMINAMATH_CALUDE_marias_first_stop_distance_l2493_249398

def total_distance : ℝ := 560

def distance_before_first_stop : ℝ → Prop := λ x =>
  let remaining_after_first := total_distance - x
  let second_stop_distance := (1/4) * remaining_after_first
  let final_leg := 210
  second_stop_distance + final_leg = remaining_after_first

theorem marias_first_stop_distance :
  ∃ x, distance_before_first_stop x ∧ x = 280 :=
sorry

end NUMINAMATH_CALUDE_marias_first_stop_distance_l2493_249398


namespace NUMINAMATH_CALUDE_nine_point_circle_triangles_l2493_249359

/-- Given 9 points on a circle, this function calculates the number of triangles
    formed by the intersections of chords inside the circle. -/
def triangles_in_circle (n : ℕ) : ℕ :=
  if n = 9 then
    (Nat.choose n 6) * (Nat.choose 6 2) * (Nat.choose 4 2) / 6
  else
    0

/-- Theorem stating that for 9 points on a circle, with chords connecting every pair
    of points and no three chords intersecting at a single point inside the circle,
    the number of triangles formed with all vertices in the interior is 210. -/
theorem nine_point_circle_triangles :
  triangles_in_circle 9 = 210 := by
  sorry

#eval triangles_in_circle 9

end NUMINAMATH_CALUDE_nine_point_circle_triangles_l2493_249359


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2493_249379

/-- A function satisfying the given functional equation -/
def FunctionalEquation (g : ℝ → ℝ) : Prop :=
  ∀ a c : ℝ, c^2 * g a = a^2 * g c

theorem functional_equation_solution (g : ℝ → ℝ) 
  (h1 : FunctionalEquation g) (h2 : g 3 ≠ 0) : 
  (g 6 - g 2) / g 3 = 32 / 9 := by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2493_249379


namespace NUMINAMATH_CALUDE_heesu_has_greatest_sum_l2493_249305

def sora_numbers : Fin 2 → ℕ
| 0 => 4
| 1 => 6

def heesu_numbers : Fin 2 → ℕ
| 0 => 7
| 1 => 5

def jiyeon_numbers : Fin 2 → ℕ
| 0 => 3
| 1 => 8

def sum_numbers (numbers : Fin 2 → ℕ) : ℕ :=
  (numbers 0) + (numbers 1)

theorem heesu_has_greatest_sum :
  sum_numbers heesu_numbers > sum_numbers sora_numbers ∧
  sum_numbers heesu_numbers > sum_numbers jiyeon_numbers :=
by sorry

end NUMINAMATH_CALUDE_heesu_has_greatest_sum_l2493_249305


namespace NUMINAMATH_CALUDE_vaccine_cost_reduction_l2493_249374

/-- The cost reduction for vaccine production over one year -/
def costReduction (initialCost : ℝ) (decreaseRate : ℝ) : ℝ :=
  initialCost * decreaseRate - initialCost * decreaseRate^2

/-- Theorem: The cost reduction for producing 1 set of vaccines this year
    compared to last year, given an initial cost of 5000 yuan two years ago
    and an annual average decrease rate of x, is 5000x - 5000x^2 yuan. -/
theorem vaccine_cost_reduction (x : ℝ) :
  costReduction 5000 x = 5000 * x - 5000 * x^2 := by
  sorry

end NUMINAMATH_CALUDE_vaccine_cost_reduction_l2493_249374


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l2493_249321

theorem quadratic_equation_properties (m : ℝ) :
  let f := fun x => m * x^2 - 4 * x + 1
  (∃ x : ℝ, f x = 0) →
  (f 1 = 0 → m = 3) ∧
  (m ≠ 0 → m ≤ 4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l2493_249321


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2493_249360

/-- A hyperbola with given properties -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b
  h_equation : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1
  h_focus : ∃ c : ℝ, c = 5 -- Right focus coincides with focus of y^2 = 20x
  h_asymptote : ∀ x y : ℝ, y = 4/3 * x ∨ y = -4/3 * x

/-- The theorem stating that the hyperbola with given properties has the equation x^2/9 - y^2/16 = 1 -/
theorem hyperbola_equation (C : Hyperbola) : C.a^2 = 9 ∧ C.b^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2493_249360


namespace NUMINAMATH_CALUDE_decagon_diagonal_intersections_l2493_249384

def regular_decagon : Nat := 10

/-- The number of distinct interior points where two or more diagonals intersect in a regular decagon -/
def intersection_points (n : Nat) : Nat :=
  Nat.choose n 4

theorem decagon_diagonal_intersections :
  intersection_points regular_decagon = 210 := by
  sorry

end NUMINAMATH_CALUDE_decagon_diagonal_intersections_l2493_249384


namespace NUMINAMATH_CALUDE_geometric_sequence_property_not_necessary_condition_l2493_249386

/-- A sequence is geometric if the ratio between any two consecutive terms is constant. -/
def IsGeometricSequence (s : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, s (n + 1) = r * s n

theorem geometric_sequence_property :
  ∀ a b c d : ℝ,
  (∃ s : ℕ → ℝ, IsGeometricSequence s ∧ s 0 = a ∧ s 1 = b ∧ s 2 = c ∧ s 3 = d) →
  a * d = b * c :=
sorry

theorem not_necessary_condition :
  ∃ a b c d : ℝ, a * d = b * c ∧
  ¬(∃ s : ℕ → ℝ, IsGeometricSequence s ∧ s 0 = a ∧ s 1 = b ∧ s 2 = c ∧ s 3 = d) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_not_necessary_condition_l2493_249386


namespace NUMINAMATH_CALUDE_max_F_value_l2493_249367

def is_eternal_number (M : ℕ) : Prop :=
  M ≥ 1000 ∧ M < 10000 ∧
  (M / 100 % 10 + M / 10 % 10 + M % 10 = 12)

def N (M : ℕ) : ℕ :=
  (M / 1000) * 100 + (M / 100 % 10) * 1000 + (M / 10 % 10) + (M % 10) * 10

def F (M : ℕ) : ℚ :=
  (M - N M) / 9

theorem max_F_value (M : ℕ) :
  is_eternal_number M →
  (M / 100 % 10 - M % 10 = M / 1000) →
  (F M / 9).isInt →
  F M ≤ 9 :=
sorry

end NUMINAMATH_CALUDE_max_F_value_l2493_249367


namespace NUMINAMATH_CALUDE_triangle_perimeter_l2493_249363

/-- Proves that a triangle with inradius 2.5 cm and area 45 cm² has a perimeter of 36 cm -/
theorem triangle_perimeter (r : ℝ) (A : ℝ) (p : ℝ) : 
  r = 2.5 → A = 45 → A = r * (p / 2) → p = 36 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l2493_249363


namespace NUMINAMATH_CALUDE_find_number_l2493_249349

theorem find_number : ∃! x : ℚ, (x + 305) / 16 = 31 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l2493_249349


namespace NUMINAMATH_CALUDE_age_proof_l2493_249343

/-- The age of a person whose current age is three times what it was six years ago. -/
def age : ℕ := 9

theorem age_proof : age = 9 := by
  have h : age = 3 * (age - 6) := by sorry
  sorry

end NUMINAMATH_CALUDE_age_proof_l2493_249343


namespace NUMINAMATH_CALUDE_specific_quadrilateral_perimeter_l2493_249324

/-- A convex quadrilateral with a point inside it -/
structure ConvexQuadrilateral where
  W : ℝ × ℝ
  X : ℝ × ℝ
  Y : ℝ × ℝ
  Z : ℝ × ℝ
  Q : ℝ × ℝ
  area : ℝ
  wq : ℝ
  xq : ℝ
  yq : ℝ
  zq : ℝ
  convex : Bool
  inside : Bool

/-- The perimeter of a quadrilateral -/
def perimeter (quad : ConvexQuadrilateral) : ℝ :=
  sorry

/-- Theorem stating the perimeter of the specific quadrilateral -/
theorem specific_quadrilateral_perimeter :
  ∀ (quad : ConvexQuadrilateral),
    quad.area = 2500 ∧
    quad.wq = 30 ∧
    quad.xq = 40 ∧
    quad.yq = 35 ∧
    quad.zq = 50 ∧
    quad.convex = true ∧
    quad.inside = true →
    perimeter quad = 155 + 10 * Real.sqrt 34 + 5 * Real.sqrt 113 :=
  sorry

end NUMINAMATH_CALUDE_specific_quadrilateral_perimeter_l2493_249324


namespace NUMINAMATH_CALUDE_min_n_for_60n_divisible_by_4_and_8_l2493_249309

theorem min_n_for_60n_divisible_by_4_and_8 :
  ∃ (n : ℕ), n > 0 ∧ 4 ∣ 60 * n ∧ 8 ∣ 60 * n ∧
  ∀ (m : ℕ), m > 0 → 4 ∣ 60 * m → 8 ∣ 60 * m → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_min_n_for_60n_divisible_by_4_and_8_l2493_249309


namespace NUMINAMATH_CALUDE_exponent_product_simplification_l2493_249366

theorem exponent_product_simplification :
  (10 ^ 0.5) * (10 ^ 0.25) * (10 ^ 0.15) * (10 ^ 0.05) * (10 ^ 1.05) = 100 := by
  sorry

end NUMINAMATH_CALUDE_exponent_product_simplification_l2493_249366


namespace NUMINAMATH_CALUDE_cube_color_probability_l2493_249345

def cube_face_colors := Fin 3
def num_faces : Nat := 6

-- Probability of each color
def color_prob : ℚ := 1 / 3

-- Total number of possible color arrangements
def total_arrangements : Nat := 3^num_faces

-- Number of arrangements where all faces are the same color
def all_same_color : Nat := 3

-- Number of arrangements where 5 faces are the same color and 1 is different
def five_same_one_different : Nat := 3 * 6 * 2

-- Number of arrangements where 4 faces are the same color and opposite faces are different
def four_same_opposite_different : Nat := 3 * 3 * 6

-- Total number of suitable arrangements
def suitable_arrangements : Nat := all_same_color + five_same_one_different + four_same_opposite_different

-- Probability of suitable arrangements
def prob_suitable_arrangements : ℚ := suitable_arrangements / total_arrangements

theorem cube_color_probability :
  prob_suitable_arrangements = 31 / 243 :=
sorry

end NUMINAMATH_CALUDE_cube_color_probability_l2493_249345


namespace NUMINAMATH_CALUDE_total_jump_rope_time_l2493_249373

/-- The total jump rope time for four girls given their relative jump times -/
theorem total_jump_rope_time (cindy betsy tina sarah : ℕ) : 
  cindy = 12 →
  betsy = cindy / 2 →
  tina = betsy * 3 →
  sarah = cindy + tina →
  cindy + betsy + tina + sarah = 66 := by
  sorry

end NUMINAMATH_CALUDE_total_jump_rope_time_l2493_249373


namespace NUMINAMATH_CALUDE_interior_edges_sum_for_specific_frame_l2493_249312

/-- Represents a rectangular picture frame -/
structure Frame where
  outerLength : ℝ
  outerWidth : ℝ
  frameWidth : ℝ

/-- Calculates the area of the frame -/
def frameArea (f : Frame) : ℝ :=
  f.outerLength * f.outerWidth - (f.outerLength - 2 * f.frameWidth) * (f.outerWidth - 2 * f.frameWidth)

/-- Calculates the sum of the lengths of the four interior edges -/
def interiorEdgesSum (f : Frame) : ℝ :=
  2 * (f.outerLength - 2 * f.frameWidth) + 2 * (f.outerWidth - 2 * f.frameWidth)

/-- Theorem stating the sum of interior edges for a specific frame -/
theorem interior_edges_sum_for_specific_frame :
  ∃ (f : Frame),
    f.outerLength = 7 ∧
    f.frameWidth = 2 ∧
    frameArea f = 30 ∧
    interiorEdgesSum f = 7 := by
  sorry

end NUMINAMATH_CALUDE_interior_edges_sum_for_specific_frame_l2493_249312


namespace NUMINAMATH_CALUDE_initial_insurance_premium_l2493_249316

/-- Proves that the initial insurance premium is $50 given the specified conditions --/
theorem initial_insurance_premium (P : ℝ) : 
  (1.1 * P + 3 * 5 = 70) → P = 50 := by
  sorry

end NUMINAMATH_CALUDE_initial_insurance_premium_l2493_249316


namespace NUMINAMATH_CALUDE_max_leap_years_in_period_l2493_249329

/-- A calendrical system where leap years occur every 5 years -/
structure CalendarSystem where
  leap_year_interval : ℕ
  leap_year_interval_eq : leap_year_interval = 5

/-- The number of years in the period we're considering -/
def period_length : ℕ := 200

/-- The maximum number of leap years in the given period -/
def max_leap_years (c : CalendarSystem) : ℕ := period_length / c.leap_year_interval

/-- Theorem stating that the maximum number of leap years in a 200-year period is 40 -/
theorem max_leap_years_in_period (c : CalendarSystem) : max_leap_years c = 40 := by
  sorry

end NUMINAMATH_CALUDE_max_leap_years_in_period_l2493_249329


namespace NUMINAMATH_CALUDE_f_minimum_l2493_249346

-- Define the function f
def f (x a b : ℝ) : ℝ := |x + a| + |x - b|

-- State the theorem
theorem f_minimum (a b : ℝ) (h : 1 / (2 * a) + 2 / b = 1) :
  ∀ x : ℝ, f x a b ≥ 9/2 :=
by sorry

end NUMINAMATH_CALUDE_f_minimum_l2493_249346


namespace NUMINAMATH_CALUDE_milk_delivery_solution_l2493_249389

/-- Represents the milk delivery problem --/
def MilkDeliveryProblem (jarsPerCarton : ℕ) : Prop :=
  let usualCartons : ℕ := 50
  let actualCartons : ℕ := usualCartons - 20
  let damagedJarsInFiveCartons : ℕ := 5 * 3
  let totalDamagedJars : ℕ := damagedJarsInFiveCartons + jarsPerCarton
  let goodJars : ℕ := 565
  actualCartons * jarsPerCarton - totalDamagedJars = goodJars

/-- Theorem stating that the solution to the milk delivery problem is 20 jars per carton --/
theorem milk_delivery_solution : MilkDeliveryProblem 20 := by
  sorry

end NUMINAMATH_CALUDE_milk_delivery_solution_l2493_249389


namespace NUMINAMATH_CALUDE_smallest_n_square_cube_l2493_249365

theorem smallest_n_square_cube : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (∃ (k : ℕ), 3 * n = k^2) ∧ 
  (∃ (m : ℕ), 5 * n = m^3) ∧
  (∀ (x : ℕ), x > 0 → (∃ (y : ℕ), 3 * x = y^2) → (∃ (z : ℕ), 5 * x = z^3) → x ≥ n) ∧
  n = 675 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_square_cube_l2493_249365


namespace NUMINAMATH_CALUDE_tom_sleep_hours_l2493_249327

/-- Proves that Tom was getting 6 hours of sleep before increasing it by 1/3 to 8 hours --/
theorem tom_sleep_hours : 
  ∀ (x : ℝ), 
  (x + (1/3) * x = 8) → 
  x = 6 := by
sorry

end NUMINAMATH_CALUDE_tom_sleep_hours_l2493_249327


namespace NUMINAMATH_CALUDE_olympic_medal_awards_l2493_249332

/-- The number of ways to award medals in the Olympic 100-meter finals --/
def medal_award_ways (total_sprinters : ℕ) (american_sprinters : ℕ) (medals : ℕ) : ℕ :=
  let non_american_sprinters := total_sprinters - american_sprinters
  let no_american_medal := Nat.descFactorial non_american_sprinters medals
  let one_american_medal := american_sprinters * medals * (Nat.descFactorial non_american_sprinters (medals - 1))
  no_american_medal + one_american_medal

/-- Theorem stating the number of ways to award medals in the given scenario --/
theorem olympic_medal_awards : 
  medal_award_ways 10 4 3 = 480 := by
  sorry

end NUMINAMATH_CALUDE_olympic_medal_awards_l2493_249332


namespace NUMINAMATH_CALUDE_next_adjacent_natural_number_l2493_249311

theorem next_adjacent_natural_number (n a : ℕ) (h : n = a^2) : 
  n + 1 = a^2 + 1 := by sorry

end NUMINAMATH_CALUDE_next_adjacent_natural_number_l2493_249311


namespace NUMINAMATH_CALUDE_cubic_expression_value_l2493_249331

theorem cubic_expression_value (p q : ℝ) : 
  3 * p^2 - 7 * p + 1 = 0 →
  3 * q^2 - 7 * q + 1 = 0 →
  (9 * p^3 - 9 * q^3) / (p - q) = 46 := by
sorry

end NUMINAMATH_CALUDE_cubic_expression_value_l2493_249331


namespace NUMINAMATH_CALUDE_S_finite_iff_power_of_two_l2493_249392

def S (k : ℕ) : Set (ℕ × ℕ × ℕ) :=
  {t | t.1 % 2 = 1 ∧ 
       Nat.gcd t.2.1 t.2.2 = 1 ∧ 
       t.2.1 + t.2.2 = k ∧ 
       t.1 ∣ (t.2.1 ^ t.1 + t.2.2 ^ t.1)}

theorem S_finite_iff_power_of_two (k : ℕ) (h : k > 1) :
  Set.Finite (S k) ↔ ∃ α : ℕ, k = 2^α ∧ α > 0 :=
sorry

end NUMINAMATH_CALUDE_S_finite_iff_power_of_two_l2493_249392


namespace NUMINAMATH_CALUDE_count_seven_to_800_l2493_249376

def count_seven (n : ℕ) : ℕ := 
  let units := n / 10
  let tens := n / 100
  let hundreds := if n ≥ 700 then 100 else 0
  units + tens * 10 + hundreds

theorem count_seven_to_800 : count_seven 800 = 260 := by sorry

end NUMINAMATH_CALUDE_count_seven_to_800_l2493_249376


namespace NUMINAMATH_CALUDE_coffee_blend_cost_calculation_l2493_249380

/-- The cost of the coffee blend given the prices and amounts of two types of coffee. -/
def coffee_blend_cost (price_a price_b : ℝ) (amount_a : ℝ) : ℝ :=
  amount_a * price_a + 2 * amount_a * price_b

/-- Theorem stating the total cost of the coffee blend under given conditions. -/
theorem coffee_blend_cost_calculation :
  coffee_blend_cost 4.60 5.95 67.52 = 1114.08 := by
  sorry

end NUMINAMATH_CALUDE_coffee_blend_cost_calculation_l2493_249380


namespace NUMINAMATH_CALUDE_number_decrease_divide_l2493_249354

theorem number_decrease_divide (x : ℚ) : (x - 4) / 10 = 5 → (x - 5) / 7 = 7 := by
  sorry

end NUMINAMATH_CALUDE_number_decrease_divide_l2493_249354


namespace NUMINAMATH_CALUDE_partnership_gain_l2493_249364

/-- Represents the investment and profit structure of a partnership --/
structure Partnership where
  raman_investment : ℝ
  lakshmi_share : ℝ
  profit_ratio : ℝ → ℝ → ℝ → Prop

/-- Calculates the total annual gain of the partnership --/
def total_annual_gain (p : Partnership) : ℝ :=
  3 * p.lakshmi_share

/-- Theorem stating that the total annual gain of the partnership is 36000 --/
theorem partnership_gain (p : Partnership) 
  (h1 : p.profit_ratio (p.raman_investment * 12) (2 * p.raman_investment * 6) (3 * p.raman_investment * 4))
  (h2 : p.lakshmi_share = 12000) : 
  total_annual_gain p = 36000 := by
  sorry

end NUMINAMATH_CALUDE_partnership_gain_l2493_249364


namespace NUMINAMATH_CALUDE_ellipse_equation_l2493_249330

theorem ellipse_equation (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b)
  (h4 : c / a = 2 / 3) (h5 : a = 3) :
  (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ↔ x^2 / 9 + y^2 / 5 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l2493_249330


namespace NUMINAMATH_CALUDE_rect_to_cylindrical_l2493_249339

/-- Conversion from rectangular to cylindrical coordinates -/
theorem rect_to_cylindrical (x y z : ℝ) (h_x : x = 4) (h_y : y = 4 * Real.sqrt 3) (h_z : z = 5) :
  ∃ (r θ : ℝ), r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧
  r = 8 ∧ θ = Real.pi / 3 ∧ z = 5 ∧
  x = r * Real.cos θ ∧ y = r * Real.sin θ := by
  sorry

end NUMINAMATH_CALUDE_rect_to_cylindrical_l2493_249339


namespace NUMINAMATH_CALUDE_remainder_problem_l2493_249348

theorem remainder_problem (N : ℤ) : 
  N % 899 = 63 → N % 29 = 10 := by sorry

end NUMINAMATH_CALUDE_remainder_problem_l2493_249348


namespace NUMINAMATH_CALUDE_special_curve_hyperbola_range_l2493_249347

/-- A curve defined by the equation x^2 / (m + 2) + y^2 / (m + 1) = 1 --/
def is_special_curve (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (m + 2) + y^2 / (m + 1) = 1

/-- The condition for the curve to be a hyperbola with foci on the x-axis --/
def is_hyperbola_x_foci (m : ℝ) : Prop :=
  (m + 2 > 0) ∧ (m + 1 < 0)

/-- The main theorem stating the range of m for which the curve is a hyperbola with foci on the x-axis --/
theorem special_curve_hyperbola_range (m : ℝ) :
  is_special_curve m ∧ is_hyperbola_x_foci m ↔ -2 < m ∧ m < -1 :=
sorry

end NUMINAMATH_CALUDE_special_curve_hyperbola_range_l2493_249347


namespace NUMINAMATH_CALUDE_dealers_dishonesty_percentage_l2493_249301

/-- The dealer's percentage of dishonesty in terms of weight -/
theorem dealers_dishonesty_percentage
  (standard_weight : ℝ)
  (dealer_weight : ℝ)
  (h1 : standard_weight = 16)
  (h2 : dealer_weight = 14.8) :
  (standard_weight - dealer_weight) / standard_weight * 100 = 7.5 := by
sorry

end NUMINAMATH_CALUDE_dealers_dishonesty_percentage_l2493_249301


namespace NUMINAMATH_CALUDE_grace_total_pennies_l2493_249307

/-- The value of a coin in pennies -/
def coin_value : ℕ := 10

/-- The value of a nickel in pennies -/
def nickel_value : ℕ := 5

/-- The number of coins Grace has -/
def grace_coins : ℕ := 10

/-- The number of nickels Grace has -/
def grace_nickels : ℕ := 10

/-- The total number of pennies Grace will have after exchanging her coins and nickels -/
theorem grace_total_pennies : 
  grace_coins * coin_value + grace_nickels * nickel_value = 150 := by
  sorry

end NUMINAMATH_CALUDE_grace_total_pennies_l2493_249307


namespace NUMINAMATH_CALUDE_caterpillar_climb_days_l2493_249340

/-- The number of days it takes for a caterpillar to climb a pole -/
def climbingDays (poleHeight : ℕ) (dayClimb : ℕ) (nightSlide : ℕ) : ℕ :=
  let netClimbPerDay := dayClimb - nightSlide
  let daysToAlmostTop := (poleHeight - dayClimb) / netClimbPerDay
  daysToAlmostTop + 1

/-- Theorem stating that it takes 16 days for the caterpillar to reach the top -/
theorem caterpillar_climb_days :
  climbingDays 20 5 4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_caterpillar_climb_days_l2493_249340


namespace NUMINAMATH_CALUDE_lock_combination_l2493_249353

-- Define the base
def base : ℕ := 11

-- Define the function to convert from base 11 to decimal
def toDecimal (digits : List ℕ) : ℕ :=
  digits.enum.foldl (λ acc (i, d) => acc + d * base ^ (digits.length - 1 - i)) 0

-- Define the equation in base 11
axiom equation_holds : ∃ (S T A R E B : ℕ),
  S < base ∧ T < base ∧ A < base ∧ R < base ∧ E < base ∧ B < base ∧
  S ≠ T ∧ S ≠ A ∧ S ≠ R ∧ S ≠ E ∧ S ≠ B ∧
  T ≠ A ∧ T ≠ R ∧ T ≠ E ∧ T ≠ B ∧
  A ≠ R ∧ A ≠ E ∧ A ≠ B ∧
  R ≠ E ∧ R ≠ B ∧
  E ≠ B ∧
  (S * base^3 + T * base^2 + A * base + R) +
  (T * base^3 + A * base^2 + R * base + S) +
  (R * base^3 + E * base^2 + S * base + T) +
  (R * base^3 + A * base^2 + R * base + E) +
  (B * base^3 + E * base^2 + A * base + R) =
  (B * base^3 + E * base^2 + S * base + T)

-- Theorem to prove
theorem lock_combination : 
  ∃ (S T A R : ℕ), toDecimal [S, T, A, R] = 7639 ∧
  (∃ (E B : ℕ), 
    S < base ∧ T < base ∧ A < base ∧ R < base ∧ E < base ∧ B < base ∧
    S ≠ T ∧ S ≠ A ∧ S ≠ R ∧ S ≠ E ∧ S ≠ B ∧
    T ≠ A ∧ T ≠ R ∧ T ≠ E ∧ T ≠ B ∧
    A ≠ R ∧ A ≠ E ∧ A ≠ B ∧
    R ≠ E ∧ R ≠ B ∧
    E ≠ B ∧
    (S * base^3 + T * base^2 + A * base + R) +
    (T * base^3 + A * base^2 + R * base + S) +
    (R * base^3 + E * base^2 + S * base + T) +
    (R * base^3 + A * base^2 + R * base + E) +
    (B * base^3 + E * base^2 + A * base + R) =
    (B * base^3 + E * base^2 + S * base + T)) :=
by sorry

end NUMINAMATH_CALUDE_lock_combination_l2493_249353


namespace NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l2493_249344

-- Problem 1
theorem factorization_problem_1 (a b : ℝ) :
  a^2 * (a - b) + 4 * b^2 * (b - a) = (a - b) * (a + 2*b) * (a - 2*b) := by sorry

-- Problem 2
theorem factorization_problem_2 (m : ℝ) :
  m^4 - 1 = (m^2 + 1) * (m + 1) * (m - 1) := by sorry

end NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l2493_249344


namespace NUMINAMATH_CALUDE_negation_of_proposition_negation_of_specific_proposition_l2493_249371

theorem negation_of_proposition (P : ℝ → Prop) : 
  (¬∀ x : ℝ, P x) ↔ (∃ x : ℝ, ¬P x) :=
by sorry

theorem negation_of_specific_proposition : 
  (¬∀ x : ℝ, x^2 + x - 1 < 0) ↔ (∃ x : ℝ, x^2 + x - 1 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_negation_of_specific_proposition_l2493_249371


namespace NUMINAMATH_CALUDE_height_difference_l2493_249342

/-- Given the heights of Anne, her sister, and Bella, prove the height difference between Bella and Anne's sister. -/
theorem height_difference (anne_height : ℝ) (sister_ratio : ℝ) (bella_ratio : ℝ)
  (h1 : anne_height = 80)
  (h2 : sister_ratio = 2)
  (h3 : bella_ratio = 3) :
  bella_ratio * anne_height - anne_height / sister_ratio = 200 := by
  sorry

end NUMINAMATH_CALUDE_height_difference_l2493_249342


namespace NUMINAMATH_CALUDE_apple_picking_problem_l2493_249350

theorem apple_picking_problem (maggie_apples layla_apples average_apples : ℕ) 
  (h1 : maggie_apples = 40)
  (h2 : layla_apples = 22)
  (h3 : average_apples = 30)
  (h4 : (maggie_apples + layla_apples + kelsey_apples) / 3 = average_apples) :
  kelsey_apples = 28 := by
  sorry

end NUMINAMATH_CALUDE_apple_picking_problem_l2493_249350


namespace NUMINAMATH_CALUDE_circle_graph_parts_sum_to_one_l2493_249396

theorem circle_graph_parts_sum_to_one :
  let white : ℚ := 1/2
  let black : ℚ := 1/4
  let gray : ℚ := 1/8
  let blue : ℚ := 1/8
  white + black + gray + blue = 1 := by
sorry

end NUMINAMATH_CALUDE_circle_graph_parts_sum_to_one_l2493_249396


namespace NUMINAMATH_CALUDE_power_multiplication_l2493_249388

theorem power_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l2493_249388


namespace NUMINAMATH_CALUDE_min_value_theorem_l2493_249322

/-- The function f(x) = |x + a| + |x - b| -/
def f (a b x : ℝ) : ℝ := |x + a| + |x - b|

/-- The theorem statement -/
theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (hmin : ∀ x, f a b x ≥ 4) (hmin_exists : ∃ x, f a b x = 4) :
  (a + b = 4) ∧ 
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 4 → 1/4 * x^2 + 1/9 * y^2 ≥ 16/13) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 4 ∧ 1/4 * x^2 + 1/9 * y^2 = 16/13) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2493_249322


namespace NUMINAMATH_CALUDE_balanced_sum_of_palindromes_is_palindrome_l2493_249333

/-- A 4-digit number is balanced if the sum of its first two digits equals the sum of its last two digits -/
def is_balanced (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧
  (n / 1000 + (n / 100 % 10) = (n / 10 % 10) + n % 10)

/-- A 4-digit number is a palindrome if its first digit equals its last digit and its second digit equals its third digit -/
def is_palindrome (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧
  (n / 1000 = n % 10) ∧ ((n / 100) % 10 = (n / 10) % 10)

/-- The sum of two palindrome numbers is divisible by 11 -/
axiom sum_of_palindromes_div_11 (a b : ℕ) :
  is_palindrome a → is_palindrome b → (a + b) % 11 = 0

theorem balanced_sum_of_palindromes_is_palindrome (n : ℕ) :
  is_balanced n → (∃ a b : ℕ, is_palindrome a ∧ is_palindrome b ∧ n = a + b) →
  is_palindrome n :=
by sorry

end NUMINAMATH_CALUDE_balanced_sum_of_palindromes_is_palindrome_l2493_249333


namespace NUMINAMATH_CALUDE_point_placement_on_line_l2493_249328

theorem point_placement_on_line : ∃ (a b c d : ℝ),
  |b - a| = 10 ∧
  |c - a| = 3 ∧
  |d - b| = 5 ∧
  |d - c| = 8 ∧
  a = 0 ∧ b = 10 ∧ c = -3 ∧ d = 5 := by
  sorry

end NUMINAMATH_CALUDE_point_placement_on_line_l2493_249328


namespace NUMINAMATH_CALUDE_function_nonnegative_implies_a_range_l2493_249383

theorem function_nonnegative_implies_a_range 
  (f : ℝ → ℝ) 
  (h : ∀ x ∈ Set.Icc (-2 : ℝ) 2, f x ≥ 0) 
  (h_def : ∀ x, f x = x^2 + a*x + 3 - a) : 
  a ∈ Set.Icc (-7 : ℝ) 2 := by
sorry

end NUMINAMATH_CALUDE_function_nonnegative_implies_a_range_l2493_249383


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l2493_249302

theorem equilateral_triangle_area_perimeter_ratio :
  ∀ s : ℝ,
  s > 0 →
  let area := (s^2 * Real.sqrt 3) / 4
  let perimeter := 3 * s
  s = 6 →
  area / perimeter = Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l2493_249302


namespace NUMINAMATH_CALUDE_even_function_k_value_l2493_249378

/-- A function f is even if f(x) = f(-x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- The function f(x) = kx^2 + (k-1)x + 2 -/
def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 + (k - 1) * x + 2

theorem even_function_k_value :
  ∀ k : ℝ, IsEven (f k) → k = 1 := by sorry

end NUMINAMATH_CALUDE_even_function_k_value_l2493_249378


namespace NUMINAMATH_CALUDE_initial_distance_problem_l2493_249387

theorem initial_distance_problem (enrique_speed jamal_speed meeting_time : ℝ) 
  (h1 : enrique_speed = 16)
  (h2 : jamal_speed = 23)
  (h3 : meeting_time = 8) :
  enrique_speed * meeting_time + jamal_speed * meeting_time = 312 := by
  sorry

end NUMINAMATH_CALUDE_initial_distance_problem_l2493_249387


namespace NUMINAMATH_CALUDE_simplify_expression_l2493_249372

theorem simplify_expression : 18 * (8 / 15) * (1 / 12) = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2493_249372


namespace NUMINAMATH_CALUDE_green_shirts_count_l2493_249325

-- Define the total number of shirts
def total_shirts : ℕ := 23

-- Define the number of blue shirts
def blue_shirts : ℕ := 6

-- Theorem: The number of green shirts is 17
theorem green_shirts_count : total_shirts - blue_shirts = 17 := by
  sorry

end NUMINAMATH_CALUDE_green_shirts_count_l2493_249325


namespace NUMINAMATH_CALUDE_pet_store_parakeets_l2493_249370

/-- Calculates the number of parakeets in a pet store given the number of cages, parrots, and average birds per cage. -/
theorem pet_store_parakeets 
  (num_cages : ℝ) 
  (num_parrots : ℝ) 
  (avg_birds_per_cage : ℝ) 
  (h1 : num_cages = 6)
  (h2 : num_parrots = 6)
  (h3 : avg_birds_per_cage = 1.333333333) :
  num_cages * avg_birds_per_cage - num_parrots = 2 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_parakeets_l2493_249370


namespace NUMINAMATH_CALUDE_geometric_sum_first_7_terms_l2493_249319

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sum_first_7_terms :
  let a : ℚ := 1/3
  let r : ℚ := 1/2
  geometric_sum a r 7 = 127/192 := by sorry

end NUMINAMATH_CALUDE_geometric_sum_first_7_terms_l2493_249319


namespace NUMINAMATH_CALUDE_parallel_sufficient_not_necessary_l2493_249334

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2)

/-- The condition that x = 2 is sufficient but not necessary for parallelism -/
theorem parallel_sufficient_not_necessary (x : ℝ) :
  let a : ℝ × ℝ := (x, 1)
  let b : ℝ × ℝ := (4, x)
  (x = 2 → are_parallel a b) ∧
  ¬(are_parallel a b → x = 2) := by
  sorry

end NUMINAMATH_CALUDE_parallel_sufficient_not_necessary_l2493_249334


namespace NUMINAMATH_CALUDE_largest_integer_prime_abs_quadratic_l2493_249382

theorem largest_integer_prime_abs_quadratic : 
  ∃ (x : ℤ), (∀ y : ℤ, y > x → ¬ Nat.Prime (Int.natAbs (4*y^2 - 39*y + 35))) ∧ 
  Nat.Prime (Int.natAbs (4*x^2 - 39*x + 35)) ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_largest_integer_prime_abs_quadratic_l2493_249382


namespace NUMINAMATH_CALUDE_paving_stone_width_l2493_249352

/-- Represents the dimensions of a rectangular courtyard -/
structure Courtyard where
  length : ℝ
  width : ℝ

/-- Represents the dimensions of a paving stone -/
structure PavingStone where
  length : ℝ
  width : ℝ

/-- Given a courtyard and paving stone specifications, proves that the width of each paving stone is 2 meters -/
theorem paving_stone_width
  (courtyard : Courtyard)
  (stone_count : ℕ)
  (stone_length : ℝ)
  (h1 : courtyard.length = 30)
  (h2 : courtyard.width = 33/2)
  (h3 : stone_count = 99)
  (h4 : stone_length = 5/2) :
  ∃ (stone : PavingStone), stone.length = stone_length ∧ stone.width = 2 :=
sorry

end NUMINAMATH_CALUDE_paving_stone_width_l2493_249352


namespace NUMINAMATH_CALUDE_points_needed_theorem_l2493_249304

/-- Represents the points scored in each game -/
structure GameScores where
  lastHome : ℕ
  firstAway : ℕ
  secondAway : ℕ
  thirdAway : ℕ

/-- Calculates the points needed in the next game -/
def pointsNeededNextGame (scores : GameScores) : ℕ :=
  4 * scores.lastHome - (scores.lastHome + scores.firstAway + scores.secondAway + scores.thirdAway)

/-- Theorem stating the conditions and the result to be proved -/
theorem points_needed_theorem (scores : GameScores) 
  (h1 : scores.lastHome = 2 * scores.firstAway)
  (h2 : scores.secondAway = scores.firstAway + 18)
  (h3 : scores.thirdAway = scores.secondAway + 2)
  (h4 : scores.lastHome = 62) :
  pointsNeededNextGame scores = 55 := by
  sorry

#eval pointsNeededNextGame ⟨62, 31, 49, 51⟩

end NUMINAMATH_CALUDE_points_needed_theorem_l2493_249304


namespace NUMINAMATH_CALUDE_final_movie_length_l2493_249320

def original_length : ℕ := 60
def removed_scenes : List ℕ := [8, 3, 4, 2, 6]

theorem final_movie_length :
  original_length - (removed_scenes.sum) = 37 := by
  sorry

end NUMINAMATH_CALUDE_final_movie_length_l2493_249320


namespace NUMINAMATH_CALUDE_smallest_primer_l2493_249355

/-- A number is primer if it has a prime number of distinct prime factors -/
def isPrimer (n : ℕ) : Prop :=
  Nat.Prime (Finset.card (Nat.factors n).toFinset)

/-- 6 is the smallest primer number -/
theorem smallest_primer : ∀ k : ℕ, k > 0 → k < 6 → ¬ isPrimer k ∧ isPrimer 6 :=
sorry

end NUMINAMATH_CALUDE_smallest_primer_l2493_249355


namespace NUMINAMATH_CALUDE_expression_evaluation_l2493_249323

theorem expression_evaluation : (1/8)^(1/3) - Real.log 2 / Real.log 3 * Real.log 27 / Real.log 4 + 2018^0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2493_249323


namespace NUMINAMATH_CALUDE_max_milk_bags_theorem_l2493_249313

/-- Calculates the maximum number of bags of milk that can be purchased given the cost per bag, 
    the promotion rule, and the total available money. -/
def max_milk_bags (cost_per_bag : ℚ) (promotion_rule : ℕ → ℕ) (total_money : ℚ) : ℕ :=
  sorry

/-- The promotion rule: for every 2 bags purchased, 1 additional bag is given for free -/
def buy_two_get_one_free (n : ℕ) : ℕ :=
  n + n / 2

theorem max_milk_bags_theorem :
  max_milk_bags 2.5 buy_two_get_one_free 30 = 18 := by
  sorry

end NUMINAMATH_CALUDE_max_milk_bags_theorem_l2493_249313
