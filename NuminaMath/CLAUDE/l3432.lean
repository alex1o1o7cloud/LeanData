import Mathlib

namespace NUMINAMATH_CALUDE_rectangle_dimensions_l3432_343251

theorem rectangle_dimensions (w : ℝ) (h1 : w > 0) : 
  (6 * w = 3 * (2 * w^2)) → w = 1 ∧ 2 * w = 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l3432_343251


namespace NUMINAMATH_CALUDE_dartboard_probability_l3432_343233

structure Dartboard :=
  (outer_radius : ℝ)
  (inner_radius : ℝ)
  (sections : ℕ)
  (inner_values : Fin 2 → ℕ)
  (outer_values : Fin 2 → ℕ)

def probability_of_score (db : Dartboard) (score : ℕ) (darts : ℕ) : ℚ :=
  sorry

theorem dartboard_probability (db : Dartboard) :
  db.outer_radius = 8 ∧
  db.inner_radius = 4 ∧
  db.sections = 4 ∧
  db.inner_values 0 = 3 ∧
  db.inner_values 1 = 4 ∧
  db.outer_values 0 = 2 ∧
  db.outer_values 1 = 5 →
  probability_of_score db 12 3 = 9 / 1024 :=
sorry

end NUMINAMATH_CALUDE_dartboard_probability_l3432_343233


namespace NUMINAMATH_CALUDE_orthogonal_vectors_imply_x_equals_two_l3432_343247

/-- Given two vectors a and b in ℝ², prove that if they are orthogonal
    and have the form a = (x-5, 3) and b = (2, x), then x = 2. -/
theorem orthogonal_vectors_imply_x_equals_two :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (x - 5, 3)
  let b : ℝ × ℝ := (2, x)
  (a.1 * b.1 + a.2 * b.2 = 0) → x = 2 := by
sorry

end NUMINAMATH_CALUDE_orthogonal_vectors_imply_x_equals_two_l3432_343247


namespace NUMINAMATH_CALUDE_decrement_calculation_l3432_343207

theorem decrement_calculation (n : ℕ) (original_mean new_mean : ℚ) 
  (h1 : n = 50)
  (h2 : original_mean = 200)
  (h3 : new_mean = 191) :
  (n : ℚ) * original_mean - n * new_mean = n * 9 := by
  sorry

end NUMINAMATH_CALUDE_decrement_calculation_l3432_343207


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l3432_343275

/-- Given vectors a, b, and c in ℝ², prove that if a + 2b is perpendicular to c,
    then the first component of c is -3. -/
theorem perpendicular_vectors (a b c : ℝ × ℝ) :
  a.1 = Real.sqrt 3 →
  a.2 = 1 →
  b.1 = 0 →
  b.2 = 1 →
  c.2 = Real.sqrt 3 →
  (a.1 + 2 * b.1, a.2 + 2 * b.2) • c = 0 →
  c.1 = -3 := by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l3432_343275


namespace NUMINAMATH_CALUDE_situp_ratio_l3432_343209

/-- The number of sit-ups Ken can do -/
def ken_situps : ℕ := 20

/-- The number of sit-ups Nathan can do -/
def nathan_situps : ℕ := 2 * ken_situps

/-- The number of sit-ups Bob can do -/
def bob_situps : ℕ := ken_situps + 10

/-- The combined number of sit-ups Ken and Nathan can do -/
def ken_nathan_combined : ℕ := ken_situps + nathan_situps

theorem situp_ratio : 
  (bob_situps : ℚ) / ken_nathan_combined = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_situp_ratio_l3432_343209


namespace NUMINAMATH_CALUDE_expected_total_rainfall_l3432_343235

/-- Represents the weather conditions for a single day --/
structure WeatherCondition where
  sun_prob : ℝ
  light_rain_prob : ℝ
  heavy_rain_prob : ℝ
  light_rain_amount : ℝ
  heavy_rain_amount : ℝ

/-- Calculates the expected rain amount for a single day --/
def expected_rain_per_day (w : WeatherCondition) : ℝ :=
  w.light_rain_prob * w.light_rain_amount + w.heavy_rain_prob * w.heavy_rain_amount

/-- The number of days in the forecast --/
def forecast_days : ℕ := 6

/-- The weather condition for each day in the forecast --/
def daily_weather : WeatherCondition :=
  { sun_prob := 0.3,
    light_rain_prob := 0.3,
    heavy_rain_prob := 0.4,
    light_rain_amount := 5,
    heavy_rain_amount := 12 }

/-- Theorem: The expected total rainfall over the forecast period is 37.8 inches --/
theorem expected_total_rainfall :
  (forecast_days : ℝ) * expected_rain_per_day daily_weather = 37.8 := by
  sorry

end NUMINAMATH_CALUDE_expected_total_rainfall_l3432_343235


namespace NUMINAMATH_CALUDE_hyperbola_ellipse_foci_coincide_l3432_343230

-- Define the hyperbola equation
def hyperbola (x y m : ℝ) : Prop := y^2 / 2 - x^2 / m = 1

-- Define the ellipse equation
def ellipse (x y : ℝ) : Prop := x^2 / 3 + y^2 / 4 = 1

-- Define the major axis endpoints of the ellipse
def ellipse_major_axis_endpoints : Set (ℝ × ℝ) := {(0, 2), (0, -2)}

-- Define the foci of the hyperbola
def hyperbola_foci (m : ℝ) : Set (ℝ × ℝ) := {(0, 2), (0, -2)}

-- Theorem statement
theorem hyperbola_ellipse_foci_coincide (m : ℝ) :
  (∀ x y, hyperbola x y m → ellipse x y) ∧
  (hyperbola_foci m = ellipse_major_axis_endpoints) →
  m = 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_ellipse_foci_coincide_l3432_343230


namespace NUMINAMATH_CALUDE_retailer_profit_percentage_l3432_343239

theorem retailer_profit_percentage
  (wholesale_price : ℝ)
  (retail_price : ℝ)
  (discount_percentage : ℝ)
  (h1 : wholesale_price = 90)
  (h2 : retail_price = 120)
  (h3 : discount_percentage = 0.1)
  : (((retail_price * (1 - discount_percentage)) - wholesale_price) / wholesale_price) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_retailer_profit_percentage_l3432_343239


namespace NUMINAMATH_CALUDE_disjunction_true_given_p_l3432_343202

theorem disjunction_true_given_p (p q : Prop) (hp : p) (hq : ¬q) : p ∨ q := by sorry

end NUMINAMATH_CALUDE_disjunction_true_given_p_l3432_343202


namespace NUMINAMATH_CALUDE_line_through_point_with_equal_intercepts_l3432_343262

-- Define a line in 2D space
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Function to check if a point is on a line
def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Function to check if a line has equal intercepts
def hasEqualIntercepts (l : Line2D) : Prop :=
  l.a ≠ 0 ∧ l.b ≠ 0 ∧ l.c / l.a = l.c / l.b

-- Theorem statement
theorem line_through_point_with_equal_intercepts :
  ∃ (l : Line2D), pointOnLine ⟨1, 1⟩ l ∧ hasEqualIntercepts l ∧
    ((l.a = 1 ∧ l.b = 1 ∧ l.c = -2) ∨ (l.a = -1 ∧ l.b = 1 ∧ l.c = 0)) :=
sorry

end NUMINAMATH_CALUDE_line_through_point_with_equal_intercepts_l3432_343262


namespace NUMINAMATH_CALUDE_hilt_fountain_distance_l3432_343264

/-- The total distance Mrs. Hilt walks to and from the water fountain -/
def total_distance (distance_to_fountain : ℕ) (number_of_trips : ℕ) : ℕ :=
  2 * distance_to_fountain * number_of_trips

/-- Theorem: Mrs. Hilt walks 240 feet in total -/
theorem hilt_fountain_distance :
  total_distance 30 4 = 240 := by
  sorry

end NUMINAMATH_CALUDE_hilt_fountain_distance_l3432_343264


namespace NUMINAMATH_CALUDE_missy_dog_yells_l3432_343254

/-- The number of times Missy yells at her dogs -/
def total_yells : ℕ := 60

/-- The ratio of yells at the stubborn dog to yells at the obedient dog -/
def stubborn_to_obedient_ratio : ℕ := 4

/-- The number of times Missy yells at the obedient dog -/
def obedient_dog_yells : ℕ := 12

theorem missy_dog_yells :
  obedient_dog_yells * (stubborn_to_obedient_ratio + 1) = total_yells :=
sorry

end NUMINAMATH_CALUDE_missy_dog_yells_l3432_343254


namespace NUMINAMATH_CALUDE_min_sum_of_squares_with_diff_l3432_343231

theorem min_sum_of_squares_with_diff (x y : ℤ) (h : x^2 - y^2 = 165) :
  ∃ (a b : ℤ), a^2 - b^2 = 165 ∧ a^2 + b^2 ≤ x^2 + y^2 ∧ a^2 + b^2 = 173 :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_with_diff_l3432_343231


namespace NUMINAMATH_CALUDE_nina_widget_problem_l3432_343204

theorem nina_widget_problem (x : ℝ) (h1 : 6 * x = 8 * (x - 1)) : 6 * x = 24 := by
  sorry

end NUMINAMATH_CALUDE_nina_widget_problem_l3432_343204


namespace NUMINAMATH_CALUDE_solve_inequality_l3432_343294

theorem solve_inequality (x : ℝ) : 
  (x + 5) / 2 - 1 < (3 * x + 2) / 2 ↔ x > 1 :=
by sorry

end NUMINAMATH_CALUDE_solve_inequality_l3432_343294


namespace NUMINAMATH_CALUDE_number_categorization_l3432_343257

def S : Set ℝ := {-2.5, 0, 8, -2, Real.pi/2, 0.7, -2/3, -1.12112112, 3/4}

theorem number_categorization :
  (∃ P I R : Set ℝ,
    P = {x ∈ S | x > 0} ∧
    I = {x ∈ S | ∃ n : ℤ, x = n} ∧
    R = {x ∈ S | ¬∃ q : ℚ, x = q} ∧
    P = {8, Real.pi/2, 0.7, 3/4} ∧
    I = {0, 8, -2} ∧
    R = {Real.pi/2, -1.12112112}) :=
by sorry

end NUMINAMATH_CALUDE_number_categorization_l3432_343257


namespace NUMINAMATH_CALUDE_greta_is_oldest_l3432_343217

-- Define the set of people
inductive Person : Type
| Ada : Person
| Darwyn : Person
| Max : Person
| Greta : Person
| James : Person

-- Define the age relation
def younger_than (a b : Person) : Prop := sorry

-- Define the conditions
axiom ada_younger_than_darwyn : younger_than Person.Ada Person.Darwyn
axiom max_younger_than_greta : younger_than Person.Max Person.Greta
axiom james_older_than_darwyn : younger_than Person.Darwyn Person.James
axiom max_same_age_as_james : ∀ p, younger_than Person.Max p ↔ younger_than Person.James p

-- Define the oldest person property
def is_oldest (p : Person) : Prop :=
  ∀ q : Person, q ≠ p → younger_than q p

-- Theorem statement
theorem greta_is_oldest : is_oldest Person.Greta := by
  sorry

end NUMINAMATH_CALUDE_greta_is_oldest_l3432_343217


namespace NUMINAMATH_CALUDE_lolita_milk_consumption_l3432_343219

/-- Lolita's weekly milk consumption --/
theorem lolita_milk_consumption :
  let weekday_consumption : ℕ := 3
  let saturday_consumption : ℕ := 2 * weekday_consumption
  let sunday_consumption : ℕ := 3 * weekday_consumption
  let weekdays : ℕ := 5
  weekdays * weekday_consumption + saturday_consumption + sunday_consumption = 30 := by
  sorry

end NUMINAMATH_CALUDE_lolita_milk_consumption_l3432_343219


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_division_l3432_343255

theorem imaginary_part_of_complex_division (z₁ z₂ : ℂ) :
  z₁.re = 1 →
  z₁.im = 1 →
  z₂.re = 0 →
  z₂.im = 1 →
  Complex.im (z₁ / z₂) = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_division_l3432_343255


namespace NUMINAMATH_CALUDE_set_operations_l3432_343220

-- Define the sets A and B
def A : Set ℝ := {x | -5 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | x < -2 ∨ x > 4}

-- State the theorem
theorem set_operations :
  (A ∩ B = {x | -5 ≤ x ∧ x < -2}) ∧
  ((Aᶜ ∪ Bᶜ) = {x | x < -5 ∨ x ≥ -2}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l3432_343220


namespace NUMINAMATH_CALUDE_y_intercept_of_line_l3432_343228

/-- The y-intercept of the line 6x + 10y = 40 is (0, 4) -/
theorem y_intercept_of_line (x y : ℝ) : 6 * x + 10 * y = 40 → x = 0 → y = 4 := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l3432_343228


namespace NUMINAMATH_CALUDE_masks_duration_for_andrew_family_l3432_343298

/-- The number of days a pack of masks lasts for a family -/
def masksDuration (familySize : ℕ) (packSize : ℕ) (daysPerMask : ℕ) : ℕ :=
  let masksUsedPer2Days := familySize
  let fullSets := packSize / masksUsedPer2Days
  let remainingMasks := packSize % masksUsedPer2Days
  let fullDays := fullSets * daysPerMask
  if remainingMasks ≥ familySize then
    fullDays + daysPerMask
  else
    fullDays + 1

/-- Theorem: A pack of 75 masks lasts 21 days for a family of 7, changing masks every 2 days -/
theorem masks_duration_for_andrew_family :
  masksDuration 7 75 2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_masks_duration_for_andrew_family_l3432_343298


namespace NUMINAMATH_CALUDE_square_geq_linear_l3432_343227

theorem square_geq_linear (a b : ℝ) (ha : a > 0) : a^2 ≥ 2*b - a := by sorry

end NUMINAMATH_CALUDE_square_geq_linear_l3432_343227


namespace NUMINAMATH_CALUDE_positive_integer_from_operations_l3432_343201

def integers : Set ℚ := {0, -3, 5, -100, 2008, -1}
def fractions : Set ℚ := {1/2, -1/3, 1/5, -3/2, -1/100}

theorem positive_integer_from_operations : ∃ (a b : ℚ) (c d : ℚ) (op1 op2 : ℚ → ℚ → ℚ),
  a ∈ integers ∧ b ∈ integers ∧ c ∈ fractions ∧ d ∈ fractions ∧
  (op1 = (· + ·) ∨ op1 = (· - ·) ∨ op1 = (· * ·) ∨ op1 = (· / ·)) ∧
  (op2 = (· + ·) ∨ op2 = (· - ·) ∨ op2 = (· * ·) ∨ op2 = (· / ·)) ∧
  ∃ (n : ℕ), (op2 (op1 a b) (op1 c d) : ℚ) = n := by
  sorry

end NUMINAMATH_CALUDE_positive_integer_from_operations_l3432_343201


namespace NUMINAMATH_CALUDE_john_school_year_hours_l3432_343269

/-- Calculates the required working hours per week during school year -/
def school_year_hours_per_week (summer_weeks : ℕ) (summer_hours_per_week : ℕ) (summer_earnings : ℕ) 
  (school_year_weeks : ℕ) (school_year_target : ℕ) : ℚ :=
  let hourly_wage : ℚ := summer_earnings / (summer_weeks * summer_hours_per_week)
  let total_school_year_hours : ℚ := school_year_target / hourly_wage
  total_school_year_hours / school_year_weeks

theorem john_school_year_hours (summer_weeks : ℕ) (summer_hours_per_week : ℕ) (summer_earnings : ℕ) 
  (school_year_weeks : ℕ) (school_year_target : ℕ) 
  (h1 : summer_weeks = 8) 
  (h2 : summer_hours_per_week = 40) 
  (h3 : summer_earnings = 4000) 
  (h4 : school_year_weeks = 25) 
  (h5 : school_year_target = 5000) :
  school_year_hours_per_week summer_weeks summer_hours_per_week summer_earnings school_year_weeks school_year_target = 16 := by
  sorry

end NUMINAMATH_CALUDE_john_school_year_hours_l3432_343269


namespace NUMINAMATH_CALUDE_minimum_point_of_translated_absolute_value_l3432_343203

-- Define the function
def f (x : ℝ) : ℝ := |x - 4| + 1

-- State the theorem
theorem minimum_point_of_translated_absolute_value :
  ∃ (x₀ : ℝ), (∀ (x : ℝ), f x₀ ≤ f x) ∧ (x₀ = 4 ∧ f x₀ = 1) :=
sorry

end NUMINAMATH_CALUDE_minimum_point_of_translated_absolute_value_l3432_343203


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3432_343263

theorem inequality_solution_set (a : ℝ) :
  let S := {x : ℝ | x^2 - (a + 1) * x + a < 0}
  (a > 1 → S = {x : ℝ | 1 < x ∧ x < a}) ∧
  (a = 1 → S = ∅) ∧
  (a < 1 → S = {x : ℝ | a < x ∧ x < 1}) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3432_343263


namespace NUMINAMATH_CALUDE_coins_problem_l3432_343252

theorem coins_problem (A B C D : ℕ) : 
  A = 21 →
  B = A - 9 →
  C = B + 17 →
  A + B + 5 = C + D →
  D = 9 := by
sorry

end NUMINAMATH_CALUDE_coins_problem_l3432_343252


namespace NUMINAMATH_CALUDE_seven_factorial_divisors_l3432_343272

/-- The number of positive divisors of n! -/
def num_divisors_factorial (n : ℕ) : ℕ := sorry

/-- 7! has 60 positive divisors -/
theorem seven_factorial_divisors : num_divisors_factorial 7 = 60 := by sorry

end NUMINAMATH_CALUDE_seven_factorial_divisors_l3432_343272


namespace NUMINAMATH_CALUDE_only_origin_satisfies_l3432_343226

def point_satisfies_inequality (x y : ℝ) : Prop :=
  x + y - 1 < 0

theorem only_origin_satisfies :
  point_satisfies_inequality 0 0 ∧
  ¬point_satisfies_inequality 2 4 ∧
  ¬point_satisfies_inequality (-1) 4 ∧
  ¬point_satisfies_inequality 1 8 :=
by sorry

end NUMINAMATH_CALUDE_only_origin_satisfies_l3432_343226


namespace NUMINAMATH_CALUDE_distinct_integers_with_divisibility_property_l3432_343267

theorem distinct_integers_with_divisibility_property (n : ℕ) (h : n ≥ 2) :
  ∃ (a : Fin n → ℕ+), (∀ i j, i.val < j.val → (a i).val ≠ (a j).val) ∧
    (∀ i j, i.val < j.val → ((a i).val - (a j).val) ∣ ((a i).val + (a j).val)) := by
  sorry

end NUMINAMATH_CALUDE_distinct_integers_with_divisibility_property_l3432_343267


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3432_343236

/-- Given a geometric sequence {aₙ} where all terms are positive, 
    with a₁ = 3 and a₁ + a₂ + a₃ = 21, prove that a₃ + a₄ + a₅ = 84 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →  -- all terms are positive
  a 1 = 3 →  -- first term is 3
  a 1 + a 2 + a 3 = 21 →  -- sum of first three terms is 21
  (∀ n, a (n + 1) = a n * q) →  -- geometric sequence property
  a 3 + a 4 + a 5 = 84 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3432_343236


namespace NUMINAMATH_CALUDE_supplementary_angles_ratio_l3432_343214

/-- Given two supplementary angles in a ratio of 5:3, the measure of the smaller angle is 67.5° -/
theorem supplementary_angles_ratio (angle1 angle2 : ℝ) : 
  angle1 + angle2 = 180 →  -- supplementary angles
  angle1 / angle2 = 5 / 3 →  -- ratio of 5:3
  min angle1 angle2 = 67.5 :=  -- smaller angle is 67.5°
by sorry

end NUMINAMATH_CALUDE_supplementary_angles_ratio_l3432_343214


namespace NUMINAMATH_CALUDE_trigonometric_identities_l3432_343259

open Real

theorem trigonometric_identities (α : ℝ) (h : tan α = 3) :
  (sin α + 3 * cos α) / (2 * sin α + 5 * cos α) = 6 / 11 ∧
  sin α ^ 2 + sin α * cos α + 3 * cos α ^ 2 = 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l3432_343259


namespace NUMINAMATH_CALUDE_gift_distribution_count_l3432_343221

/-- The number of bags of gifts -/
def num_bags : ℕ := 5

/-- The number of elderly people -/
def num_people : ℕ := 4

/-- The number of ways to distribute consecutive pairs -/
def consecutive_pairs : ℕ := 4

/-- The number of ways to arrange the remaining bags -/
def remaining_arrangements : ℕ := 24  -- This is A_4^4

/-- The total number of distribution methods -/
def total_distributions : ℕ := consecutive_pairs * remaining_arrangements

theorem gift_distribution_count :
  total_distributions = 96 :=
sorry

end NUMINAMATH_CALUDE_gift_distribution_count_l3432_343221


namespace NUMINAMATH_CALUDE_h_zero_at_seven_fifths_l3432_343289

/-- The function h(x) = 5x - 7 -/
def h (x : ℝ) : ℝ := 5 * x - 7

/-- Theorem: The value of b that satisfies h(b) = 0 is b = 7/5 -/
theorem h_zero_at_seven_fifths : 
  ∃ b : ℝ, h b = 0 ∧ b = 7/5 := by sorry

end NUMINAMATH_CALUDE_h_zero_at_seven_fifths_l3432_343289


namespace NUMINAMATH_CALUDE_binary_to_quaternary_conversion_l3432_343295

/-- Converts a binary (base 2) number to its decimal (base 10) representation -/
def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

/-- Converts a decimal (base 10) number to its quaternary (base 4) representation -/
def decimal_to_quaternary (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
    aux n []

theorem binary_to_quaternary_conversion :
  let binary : List Bool := [true, false, true, false, true, true, false, true]
  let decimal : ℕ := binary_to_decimal binary
  let quaternary : List ℕ := decimal_to_quaternary decimal
  quaternary = [2, 2, 1, 3] :=
by sorry

end NUMINAMATH_CALUDE_binary_to_quaternary_conversion_l3432_343295


namespace NUMINAMATH_CALUDE_incorrect_average_calculation_l3432_343278

theorem incorrect_average_calculation (n : ℕ) (correct_sum incorrect_sum : ℝ) 
  (h1 : n = 10)
  (h2 : correct_sum / n = 15)
  (h3 : incorrect_sum = correct_sum - 10) :
  incorrect_sum / n = 14 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_average_calculation_l3432_343278


namespace NUMINAMATH_CALUDE_biology_homework_wednesday_l3432_343211

def homework_monday : ℚ := 3/5
def remaining_after_monday : ℚ := 1 - homework_monday
def homework_tuesday : ℚ := (1/3) * remaining_after_monday

theorem biology_homework_wednesday :
  1 - homework_monday - homework_tuesday = 4/15 := by
  sorry

end NUMINAMATH_CALUDE_biology_homework_wednesday_l3432_343211


namespace NUMINAMATH_CALUDE_correct_student_distribution_l3432_343281

/-- Ticket pricing structure -/
def ticket_price (n : ℕ) : ℕ :=
  if n ≤ 50 then 15
  else if n ≤ 100 then 12
  else 10

/-- Total number of students -/
def total_students : ℕ := 105

/-- Total amount paid -/
def total_paid : ℕ := 1401

/-- Number of students in Class (1) -/
def class_1_students : ℕ := 47

/-- Number of students in Class (2) -/
def class_2_students : ℕ := total_students - class_1_students

/-- Theorem: Given the ticket pricing structure and total amount paid, 
    the number of students in Class (1) is 47 and in Class (2) is 58 -/
theorem correct_student_distribution :
  class_1_students > 40 ∧ 
  class_1_students < 50 ∧
  class_2_students = 58 ∧
  class_1_students + class_2_students = total_students ∧
  ticket_price class_1_students * class_1_students + 
  ticket_price class_2_students * class_2_students = total_paid :=
by sorry

end NUMINAMATH_CALUDE_correct_student_distribution_l3432_343281


namespace NUMINAMATH_CALUDE_kelly_egg_income_l3432_343243

/-- Calculates the money made from selling eggs over a given period. -/
def money_from_eggs (num_chickens : ℕ) (eggs_per_day : ℕ) (price_per_dozen : ℕ) (num_weeks : ℕ) : ℕ :=
  let eggs_per_week := num_chickens * eggs_per_day * 7
  let total_eggs := eggs_per_week * num_weeks
  let dozens := total_eggs / 12
  dozens * price_per_dozen

/-- Proves that Kelly makes $280 in 4 weeks from selling eggs. -/
theorem kelly_egg_income : money_from_eggs 8 3 5 4 = 280 := by
  sorry

end NUMINAMATH_CALUDE_kelly_egg_income_l3432_343243


namespace NUMINAMATH_CALUDE_sum_mod_thirteen_equals_zero_l3432_343242

theorem sum_mod_thirteen_equals_zero :
  (7650 + 7651 + 7652 + 7653 + 7654) % 13 = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_mod_thirteen_equals_zero_l3432_343242


namespace NUMINAMATH_CALUDE_divisor_problem_l3432_343253

theorem divisor_problem (D : ℕ) (hD : D > 0) 
  (h1 : 242 % D = 6)
  (h2 : 698 % D = 13)
  (h3 : 940 % D = 5) : 
  D = 14 := by sorry

end NUMINAMATH_CALUDE_divisor_problem_l3432_343253


namespace NUMINAMATH_CALUDE_x_gt_one_sufficient_not_necessary_for_x_squared_gt_one_l3432_343213

theorem x_gt_one_sufficient_not_necessary_for_x_squared_gt_one :
  (∃ x : ℝ, x > 1 → x^2 > 1) ∧
  (∃ x : ℝ, x^2 > 1 ∧ ¬(x > 1)) := by
  sorry

end NUMINAMATH_CALUDE_x_gt_one_sufficient_not_necessary_for_x_squared_gt_one_l3432_343213


namespace NUMINAMATH_CALUDE_room_size_l3432_343282

/-- Given two square carpets in a square room, prove the room's side length is 19 meters. -/
theorem room_size (small_carpet big_carpet room : ℝ) : 
  small_carpet > 0 ∧ 
  big_carpet = 2 * small_carpet ∧
  (room - small_carpet - big_carpet)^2 = 4 ∧
  (room - big_carpet) * (room - small_carpet) = 14 →
  room = 19 := by
  sorry

end NUMINAMATH_CALUDE_room_size_l3432_343282


namespace NUMINAMATH_CALUDE_certain_number_proof_l3432_343293

theorem certain_number_proof : ∃ (n : ℕ), n + 3327 = 13200 ∧ n = 9873 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3432_343293


namespace NUMINAMATH_CALUDE_noah_yearly_call_cost_l3432_343238

/-- The cost of Noah's calls to his Grammy for a year -/
def yearly_call_cost (calls_per_week : ℕ) (minutes_per_call : ℕ) (cost_per_minute : ℚ) (weeks_per_year : ℕ) : ℚ :=
  (calls_per_week * minutes_per_call * cost_per_minute * weeks_per_year : ℚ)

/-- Theorem stating that Noah's yearly call cost to his Grammy is $78 -/
theorem noah_yearly_call_cost :
  yearly_call_cost 1 30 (5/100) 52 = 78 := by
  sorry

end NUMINAMATH_CALUDE_noah_yearly_call_cost_l3432_343238


namespace NUMINAMATH_CALUDE_investment_value_after_six_weeks_l3432_343287

/-- Calculates the final investment value after six weeks of changes and compound interest --/
def calculate_investment (initial_investment : ℝ) (week1_gain : ℝ) (week1_add : ℝ)
  (week2_gain : ℝ) (week2_withdraw : ℝ) (week3_loss : ℝ) (week4_gain : ℝ) (week4_add : ℝ)
  (week5_gain : ℝ) (week6_loss : ℝ) (week6_withdraw : ℝ) (weekly_interest : ℝ) : ℝ :=
  let week1 := (initial_investment * (1 + week1_gain) * (1 + weekly_interest)) + week1_add
  let week2 := (week1 * (1 + week2_gain) * (1 + weekly_interest)) - week2_withdraw
  let week3 := week2 * (1 - week3_loss) * (1 + weekly_interest)
  let week4 := (week3 * (1 + week4_gain) * (1 + weekly_interest)) + week4_add
  let week5 := week4 * (1 + week5_gain) * (1 + weekly_interest)
  let week6 := (week5 * (1 - week6_loss) * (1 + weekly_interest)) - week6_withdraw
  week6

/-- The final investment value after six weeks is approximately $819.74 --/
theorem investment_value_after_six_weeks :
  ∃ ε > 0, |calculate_investment 400 0.25 200 0.50 150 0.10 0.20 100 0.05 0.15 250 0.02 - 819.74| < ε :=
sorry

end NUMINAMATH_CALUDE_investment_value_after_six_weeks_l3432_343287


namespace NUMINAMATH_CALUDE_decimal_127_to_octal_has_three_consecutive_digits_l3432_343299

def decimal_to_octal (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 8) ((m % 8) :: acc)
    aux n []

def is_consecutive_digits (digits : List ℕ) : Bool :=
  match digits with
  | [] => true
  | [_] => true
  | x :: y :: rest => (y = x + 1) && is_consecutive_digits (y :: rest)

theorem decimal_127_to_octal_has_three_consecutive_digits :
  let octal_digits := decimal_to_octal 127
  octal_digits.length = 3 ∧ is_consecutive_digits octal_digits = true :=
sorry

end NUMINAMATH_CALUDE_decimal_127_to_octal_has_three_consecutive_digits_l3432_343299


namespace NUMINAMATH_CALUDE_equation_solutions_l3432_343268

theorem equation_solutions : 
  {x : ℝ | (1 / (x^2 + 13*x - 16) + 1 / (x^2 + 4*x - 16) + 1 / (x^2 - 15*x - 16) = 0) ∧ 
           (x^2 + 13*x - 16 ≠ 0) ∧ (x^2 + 4*x - 16 ≠ 0) ∧ (x^2 - 15*x - 16 ≠ 0)} = 
  {1, -16, 4, -4} := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l3432_343268


namespace NUMINAMATH_CALUDE_concert_ticket_price_l3432_343223

theorem concert_ticket_price 
  (adult_price : ℚ) 
  (child_price : ℚ) 
  (num_adults : ℕ) 
  (num_children : ℕ) 
  (total_revenue : ℚ) :
  child_price = adult_price / 2 →
  num_adults = 183 →
  num_children = 28 →
  total_revenue = 5122 →
  num_adults * adult_price + num_children * child_price = total_revenue →
  adult_price = 26 := by
sorry

end NUMINAMATH_CALUDE_concert_ticket_price_l3432_343223


namespace NUMINAMATH_CALUDE_largest_increase_2003_2004_l3432_343206

def students : ℕ → ℕ
  | 2002 => 70
  | 2003 => 77
  | 2004 => 85
  | 2005 => 89
  | 2006 => 95
  | 2007 => 104
  | 2008 => 112
  | _ => 0

def percentage_increase (year1 year2 : ℕ) : ℚ :=
  (students year2 - students year1 : ℚ) / students year1 * 100

def is_largest_increase (year1 year2 : ℕ) : Prop :=
  ∀ y1 y2, y1 ≥ 2002 ∧ y2 ≤ 2008 ∧ y2 = y1 + 1 →
    percentage_increase year1 year2 ≥ percentage_increase y1 y2

theorem largest_increase_2003_2004 :
  is_largest_increase 2003 2004 :=
sorry

end NUMINAMATH_CALUDE_largest_increase_2003_2004_l3432_343206


namespace NUMINAMATH_CALUDE_ant_collision_theorem_l3432_343218

/-- Represents the possible numbers of ants on the track -/
def PossibleAntCounts : Set ℕ := {10, 11, 14, 25}

/-- Represents a configuration of ants on the track -/
structure AntConfiguration where
  clockwise : ℕ
  counterclockwise : ℕ

/-- Checks if a given configuration is valid -/
def isValidConfiguration (config : AntConfiguration) : Prop :=
  config.clockwise * config.counterclockwise = 24

theorem ant_collision_theorem
  (track_length : ℕ)
  (ant_speed : ℕ)
  (collision_pairs : ℕ)
  (h1 : track_length = 60)
  (h2 : ant_speed = 1)
  (h3 : collision_pairs = 48) :
  ∀ (total_ants : ℕ),
    (∃ (config : AntConfiguration),
      config.clockwise + config.counterclockwise = total_ants ∧
      isValidConfiguration config) →
    total_ants ∈ PossibleAntCounts :=
sorry

end NUMINAMATH_CALUDE_ant_collision_theorem_l3432_343218


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3432_343258

theorem inequality_solution_set (m : ℝ) : 
  (∀ x : ℝ, m * (x - 1) > x^2 - x ↔ 1 < x ∧ x < 2) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3432_343258


namespace NUMINAMATH_CALUDE_motorcycle_journey_avg_speed_l3432_343250

/-- A motorcyclist's journey with specific conditions -/
def motorcycle_journey (distance_AB : ℝ) (speed_BC : ℝ) : Prop :=
  ∃ (time_AB time_BC : ℝ),
    time_AB > 0 ∧ time_BC > 0 ∧
    time_AB = 3 * time_BC ∧
    distance_AB = 120 ∧
    speed_BC = 60 ∧
    (distance_AB / 2) / time_BC = speed_BC ∧
    (distance_AB + distance_AB / 2) / (time_AB + time_BC) = 45

/-- Theorem stating that under the given conditions, the average speed is 45 mph -/
theorem motorcycle_journey_avg_speed :
  motorcycle_journey 120 60 :=
sorry

end NUMINAMATH_CALUDE_motorcycle_journey_avg_speed_l3432_343250


namespace NUMINAMATH_CALUDE_distance_between_points_l3432_343288

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (3, 6)
  let p2 : ℝ × ℝ := (-7, -2)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = 2 * Real.sqrt 41 :=
by sorry

end NUMINAMATH_CALUDE_distance_between_points_l3432_343288


namespace NUMINAMATH_CALUDE_triangle_weights_equal_l3432_343241

/-- Given a triangle ABC with side weights x, y, and z, if the sum of weights on any two sides
    equals the weight on the third side multiplied by a constant k, then all weights are equal. -/
theorem triangle_weights_equal (x y z k : ℝ) 
  (h1 : x + y = k * z) 
  (h2 : y + z = k * x) 
  (h3 : z + x = k * y) : 
  x = y ∧ y = z := by
  sorry

end NUMINAMATH_CALUDE_triangle_weights_equal_l3432_343241


namespace NUMINAMATH_CALUDE_secret_codes_count_l3432_343286

/-- The number of colors available in the game -/
def num_colors : ℕ := 8

/-- The number of slots in the game -/
def num_slots : ℕ := 5

/-- The total number of possible secret codes -/
def total_codes : ℕ := num_colors ^ num_slots

/-- Theorem stating that the total number of possible secret codes is 32768 -/
theorem secret_codes_count : total_codes = 32768 := by
  sorry

end NUMINAMATH_CALUDE_secret_codes_count_l3432_343286


namespace NUMINAMATH_CALUDE_salary_increase_percentage_l3432_343205

/-- Given a salary that increases to $812 with a 16% raise, 
    prove that a 10% raise results in $770.0000000000001 -/
theorem salary_increase_percentage (S : ℝ) 
  (h1 : S + 0.16 * S = 812) 
  (h2 : S + 0.1 * S = 770.0000000000001) : 
  ∃ (P : ℝ), S + P * S = 770.0000000000001 ∧ P = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_salary_increase_percentage_l3432_343205


namespace NUMINAMATH_CALUDE_complex_power_sum_l3432_343224

theorem complex_power_sum (i : ℂ) : i^2 = -1 → i^50 + 3 * i^303 - 2 * i^101 = -1 - 5*i := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l3432_343224


namespace NUMINAMATH_CALUDE_inequality_solution_l3432_343216

theorem inequality_solution (x : ℝ) :
  x ≠ 2 → x ≠ 0 →
  ((x + 1) / (x - 2) + (x + 3) / (3 * x) ≥ 4 ↔ 
   (0 < x ∧ x ≤ 1/2) ∨ (2 < x ∧ x ≤ 11/2)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3432_343216


namespace NUMINAMATH_CALUDE_prob_even_product_two_dice_l3432_343297

/-- A fair six-sided die -/
def SixSidedDie : Finset ℕ := {1, 2, 3, 4, 5, 6}

/-- The probability space for rolling two dice -/
def TwoDiceRoll : Finset (ℕ × ℕ) := SixSidedDie.product SixSidedDie

/-- The event of rolling an even product -/
def EvenProduct : Set (ℕ × ℕ) := {p | p.1 * p.2 % 2 = 0}

theorem prob_even_product_two_dice :
  Finset.card (TwoDiceRoll.filter (λ p => p.1 * p.2 % 2 = 0)) / Finset.card TwoDiceRoll = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_prob_even_product_two_dice_l3432_343297


namespace NUMINAMATH_CALUDE_line_segment_params_sum_of_squares_l3432_343274

/-- Given two points in 2D space, this function returns the parameters of the line segment connecting them. -/
def lineSegmentParams (p1 p2 : ℝ × ℝ) : ℝ × ℝ × ℝ × ℝ := sorry

theorem line_segment_params_sum_of_squares :
  let p1 : ℝ × ℝ := (-3, 6)
  let p2 : ℝ × ℝ := (4, 14)
  let (a, b, c, d) := lineSegmentParams p1 p2
  a^2 + b^2 + c^2 + d^2 = 158 := by sorry

end NUMINAMATH_CALUDE_line_segment_params_sum_of_squares_l3432_343274


namespace NUMINAMATH_CALUDE_floor_abs_negative_real_l3432_343234

theorem floor_abs_negative_real : ⌊|(-45.7 : ℝ)|⌋ = 45 := by sorry

end NUMINAMATH_CALUDE_floor_abs_negative_real_l3432_343234


namespace NUMINAMATH_CALUDE_max_leap_years_150_years_l3432_343279

/-- A calendrical system where leap years occur every four years -/
structure CalendarSystem where
  leap_year_frequency : ℕ
  leap_year_frequency_is_four : leap_year_frequency = 4

/-- The maximum number of leap years in a given period -/
def max_leap_years (c : CalendarSystem) (period : ℕ) : ℕ :=
  (period / c.leap_year_frequency) + min 1 (period % c.leap_year_frequency)

/-- Theorem stating that the maximum number of leap years in a 150-year period is 38 -/
theorem max_leap_years_150_years (c : CalendarSystem) :
  max_leap_years c 150 = 38 := by
  sorry

#eval max_leap_years ⟨4, rfl⟩ 150

end NUMINAMATH_CALUDE_max_leap_years_150_years_l3432_343279


namespace NUMINAMATH_CALUDE_same_color_probability_l3432_343232

/-- The number of pairs of shoes -/
def num_pairs : ℕ := 6

/-- The total number of shoes -/
def total_shoes : ℕ := 2 * num_pairs

/-- The number of shoes to be selected -/
def selected_shoes : ℕ := 2

/-- The probability of selecting two shoes of the same color -/
theorem same_color_probability :
  (num_pairs : ℚ) / (total_shoes.choose selected_shoes) = 1 / 11 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l3432_343232


namespace NUMINAMATH_CALUDE_equation_solution_l3432_343291

theorem equation_solution : ∃ x : ℚ, (2*x - 1)/3 - (x - 2)/6 = 2 ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3432_343291


namespace NUMINAMATH_CALUDE_high_school_baseball_games_l3432_343273

/-- The number of baseball games Benny's high school played is equal to the sum of games he attended and missed -/
theorem high_school_baseball_games 
  (games_attended : ℕ) 
  (games_missed : ℕ) 
  (h1 : games_attended = 14) 
  (h2 : games_missed = 25) : 
  games_attended + games_missed = 39 := by
  sorry

end NUMINAMATH_CALUDE_high_school_baseball_games_l3432_343273


namespace NUMINAMATH_CALUDE_always_positive_l3432_343229

theorem always_positive (x y : ℝ) : x^2 - 4*x + y^2 + 13 > 0 := by
  sorry

end NUMINAMATH_CALUDE_always_positive_l3432_343229


namespace NUMINAMATH_CALUDE_quarter_circle_roll_path_length_l3432_343208

/-- The length of the path traveled by point B when rolling a quarter-circle along another quarter-circle -/
theorem quarter_circle_roll_path_length (r : ℝ) (h : r = 4 / π) :
  let path_length := 2 * π * r
  path_length = 8 := by sorry

end NUMINAMATH_CALUDE_quarter_circle_roll_path_length_l3432_343208


namespace NUMINAMATH_CALUDE_max_pages_for_25_dollars_l3432_343256

/-- The cost in cents to copy one page -/
def cost_per_page : ℕ := 3

/-- The available amount in dollars -/
def available_dollars : ℕ := 25

/-- Convert dollars to cents -/
def dollars_to_cents (dollars : ℕ) : ℕ := dollars * 100

/-- Calculate the maximum number of whole pages that can be copied -/
def max_pages (available : ℕ) (cost : ℕ) : ℕ :=
  (dollars_to_cents available) / cost

theorem max_pages_for_25_dollars :
  max_pages available_dollars cost_per_page = 833 := by
  sorry

end NUMINAMATH_CALUDE_max_pages_for_25_dollars_l3432_343256


namespace NUMINAMATH_CALUDE_norm_scalar_multiple_l3432_343240

theorem norm_scalar_multiple {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V] 
  (v : V) (h : ‖v‖ = 5) : ‖(4 : ℝ) • v‖ = 20 := by
  sorry

end NUMINAMATH_CALUDE_norm_scalar_multiple_l3432_343240


namespace NUMINAMATH_CALUDE_anna_mean_score_l3432_343225

def scores : List ℝ := [88, 90, 92, 95, 96, 98, 100, 102, 105]

def timothy_count : ℕ := 5
def anna_count : ℕ := 4
def timothy_mean : ℝ := 95

theorem anna_mean_score (h1 : scores.length = timothy_count + anna_count)
                        (h2 : timothy_count * timothy_mean = scores.sum - anna_count * anna_mean) :
  anna_mean = 97.75 := by
  sorry

end NUMINAMATH_CALUDE_anna_mean_score_l3432_343225


namespace NUMINAMATH_CALUDE_brandy_trail_mix_chocolate_chips_l3432_343249

/-- The weight of chocolate chips in Brandy's trail mix -/
def weight_chocolate_chips (total_weight peanuts_weight raisins_weight : ℚ) : ℚ :=
  total_weight - (peanuts_weight + raisins_weight)

/-- Theorem stating that the weight of chocolate chips in Brandy's trail mix is 0.17 pounds -/
theorem brandy_trail_mix_chocolate_chips :
  weight_chocolate_chips 0.42 0.17 0.08 = 0.17 := by
  sorry

end NUMINAMATH_CALUDE_brandy_trail_mix_chocolate_chips_l3432_343249


namespace NUMINAMATH_CALUDE_work_completion_time_l3432_343222

/-- The time taken to complete a work given the rates of two workers and their working schedule -/
theorem work_completion_time 
  (p_rate : ℝ) -- Rate at which P completes the work
  (q_rate : ℝ) -- Rate at which Q completes the work
  (p_solo_days : ℕ) -- Days P works alone
  (h1 : p_rate = 1 / 80) -- P can complete the work in 80 days
  (h2 : q_rate = 1 / 48) -- Q can complete the work in 48 days
  (h3 : p_solo_days = 8) -- P works alone for 8 days
  : ∃ (total_days : ℕ), total_days = 35 ∧ 
    p_solo_days * p_rate + (total_days - p_solo_days) * (p_rate + q_rate) = 1 :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l3432_343222


namespace NUMINAMATH_CALUDE_sunset_time_calculation_l3432_343283

/-- Represents time in 24-hour format -/
structure Time24 where
  hours : Nat
  minutes : Nat
  valid : hours < 24 ∧ minutes < 60

/-- Represents a duration in hours and minutes -/
structure Duration where
  hours : Nat
  minutes : Nat
  valid : minutes < 60

/-- Adds a duration to a time, wrapping around 24 hours if necessary -/
def addDuration (t : Time24) (d : Duration) : Time24 := sorry

/-- Converts 24-hour time to 12-hour time string (AM/PM) -/
def to12Hour (t : Time24) : String := sorry

theorem sunset_time_calculation 
  (sunrise : Time24) 
  (daylight : Duration) 
  (h_sunrise : sunrise.hours = 7 ∧ sunrise.minutes = 30)
  (h_daylight : daylight.hours = 11 ∧ daylight.minutes = 10) :
  to12Hour (addDuration sunrise daylight) = "6:40 PM" := by sorry

end NUMINAMATH_CALUDE_sunset_time_calculation_l3432_343283


namespace NUMINAMATH_CALUDE_power_of_power_l3432_343296

theorem power_of_power (x : ℝ) : (x^2)^3 = x^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l3432_343296


namespace NUMINAMATH_CALUDE_point_translation_and_line_l3432_343271

/-- Given a point (5,3) translated 4 units left and 1 unit down,
    if the resulting point lies on y = kx - 2, then k = 4 -/
theorem point_translation_and_line (k : ℝ) : 
  let original_point : ℝ × ℝ := (5, 3)
  let translated_point : ℝ × ℝ := (original_point.1 - 4, original_point.2 - 1)
  (translated_point.2 = k * translated_point.1 - 2) → k = 4 := by
sorry

end NUMINAMATH_CALUDE_point_translation_and_line_l3432_343271


namespace NUMINAMATH_CALUDE_smallest_positive_integer_ending_in_3_divisible_by_5_l3432_343212

theorem smallest_positive_integer_ending_in_3_divisible_by_5 : ∃ n : ℕ,
  (n % 10 = 3) ∧ 
  (n % 5 = 0) ∧ 
  (∀ m : ℕ, m < n → (m % 10 = 3 → m % 5 ≠ 0)) ∧
  n = 53 := by
sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_ending_in_3_divisible_by_5_l3432_343212


namespace NUMINAMATH_CALUDE_triangle_angle_proof_l3432_343246

theorem triangle_angle_proof (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- angles are positive
  a + b + c = 180 →        -- sum of angles in a triangle
  c = 45 →                 -- one angle is 45°
  b = 2 * a →              -- ratio of other two angles is 2:1
  a = 45 :=                -- prove that the smaller angle is also 45°
by sorry

end NUMINAMATH_CALUDE_triangle_angle_proof_l3432_343246


namespace NUMINAMATH_CALUDE_audrey_balls_l3432_343244

theorem audrey_balls (jake_balls : ℕ) (difference : ℕ) : 
  jake_balls = 7 → difference = 34 → jake_balls + difference = 41 :=
by
  sorry

end NUMINAMATH_CALUDE_audrey_balls_l3432_343244


namespace NUMINAMATH_CALUDE_prob_black_ball_is_one_fourth_l3432_343290

/-- Represents the number of black balls in the bag. -/
def black_balls : ℕ := 6

/-- Represents the number of red balls in the bag. -/
def red_balls : ℕ := 18

/-- Represents the total number of balls in the bag. -/
def total_balls : ℕ := black_balls + red_balls

/-- The probability of drawing a black ball from the bag. -/
def prob_black_ball : ℚ := black_balls / total_balls

theorem prob_black_ball_is_one_fourth : prob_black_ball = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_prob_black_ball_is_one_fourth_l3432_343290


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l3432_343260

theorem pure_imaginary_condition (i a : ℂ) : 
  i^2 = -1 →
  (((i^2 + a*i) / (1 + i)).re = 0) →
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l3432_343260


namespace NUMINAMATH_CALUDE_z1_in_first_quadrant_l3432_343292

def z1 (a : ℝ) : ℂ := a + Complex.I
def z2 : ℂ := 1 - Complex.I

theorem z1_in_first_quadrant (a : ℝ) :
  (z1 a / z2).im ≠ 0 ∧ (z1 a / z2).re = 0 →
  0 < (z1 a).re ∧ 0 < (z1 a).im :=
by sorry

end NUMINAMATH_CALUDE_z1_in_first_quadrant_l3432_343292


namespace NUMINAMATH_CALUDE_square_fence_perimeter_36_posts_l3432_343280

/-- Calculates the perimeter of a square fence given the number of posts, post width, and gap between posts. -/
def square_fence_perimeter (total_posts : ℕ) (post_width_inches : ℕ) (gap_feet : ℕ) : ℕ :=
  let posts_per_side : ℕ := (total_posts - 4) / 4 + 1
  let side_length : ℕ := (posts_per_side - 1) * gap_feet
  4 * side_length

/-- Theorem stating that a square fence with 36 posts, 6-inch wide posts, and 6-foot gaps has a perimeter of 192 feet. -/
theorem square_fence_perimeter_36_posts :
  square_fence_perimeter 36 6 6 = 192 := by
  sorry

end NUMINAMATH_CALUDE_square_fence_perimeter_36_posts_l3432_343280


namespace NUMINAMATH_CALUDE_not_sufficient_nor_necessary_l3432_343277

theorem not_sufficient_nor_necessary (a b : ℝ) : 
  (∃ x y : ℝ, x > y ∧ x^2 ≤ y^2) ∧ (∃ u v : ℝ, u^2 > v^2 ∧ u ≤ v) := by
  sorry

end NUMINAMATH_CALUDE_not_sufficient_nor_necessary_l3432_343277


namespace NUMINAMATH_CALUDE_tank_fill_time_l3432_343270

def fill_time_A : ℝ := 60
def fill_time_B : ℝ := 40

theorem tank_fill_time :
  let total_time : ℝ := 30
  let first_half_time : ℝ := total_time / 2
  let second_half_time : ℝ := total_time / 2
  let fill_rate_A : ℝ := 1 / fill_time_A
  let fill_rate_B : ℝ := 1 / fill_time_B
  (fill_rate_B * first_half_time) + ((fill_rate_A + fill_rate_B) * second_half_time) = 1 :=
by sorry

end NUMINAMATH_CALUDE_tank_fill_time_l3432_343270


namespace NUMINAMATH_CALUDE_trig_sum_simplification_l3432_343210

theorem trig_sum_simplification :
  (Real.sin (30 * π / 180) + Real.sin (50 * π / 180) + Real.sin (70 * π / 180) + Real.sin (90 * π / 180) +
   Real.sin (110 * π / 180) + Real.sin (130 * π / 180) + Real.sin (150 * π / 180) + Real.sin (170 * π / 180)) /
  (Real.cos (15 * π / 180) * Real.cos (25 * π / 180) * Real.cos (50 * π / 180)) =
  (8 * Real.sin (80 * π / 180) * Real.cos (40 * π / 180) * Real.cos (20 * π / 180)) /
  (Real.cos (15 * π / 180) * Real.cos (25 * π / 180) * Real.cos (50 * π / 180)) :=
by
  sorry

end NUMINAMATH_CALUDE_trig_sum_simplification_l3432_343210


namespace NUMINAMATH_CALUDE_coin_and_die_probability_l3432_343215

theorem coin_and_die_probability : 
  let coin_prob := 1 / 2  -- Probability of getting heads on a fair coin
  let die_prob := 1 / 6   -- Probability of rolling a multiple of 5 on a 6-sided die
  coin_prob * die_prob = 1 / 12 :=
by sorry

end NUMINAMATH_CALUDE_coin_and_die_probability_l3432_343215


namespace NUMINAMATH_CALUDE_min_product_of_three_exists_min_product_l3432_343284

def S : Set Int := {-10, -8, -5, -3, 0, 4, 6}

theorem min_product_of_three (a b c : Int) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) :
  a * b * c ≥ -240 :=
sorry

theorem exists_min_product :
  ∃ (a b c : Int), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ a * b * c = -240 :=
sorry

end NUMINAMATH_CALUDE_min_product_of_three_exists_min_product_l3432_343284


namespace NUMINAMATH_CALUDE_elephant_pig_equivalence_l3432_343261

variable (P Q : Prop)

theorem elephant_pig_equivalence :
  (P → Q) →
  ((P → Q) ↔ (¬Q → ¬P)) ∧
  ((P → Q) ↔ (¬P ∨ Q)) ∧
  ¬((P → Q) ↔ (Q → P)) :=
by sorry

end NUMINAMATH_CALUDE_elephant_pig_equivalence_l3432_343261


namespace NUMINAMATH_CALUDE_equal_numbers_exist_l3432_343248

/-- Triangle inequality for three sides --/
def is_triangle (x y z : ℝ) : Prop :=
  x ≤ y + z ∧ y ≤ x + z ∧ z ≤ x + y

/-- Main theorem --/
theorem equal_numbers_exist (a b c : ℝ) :
  (∀ n : ℕ, is_triangle (a^n) (b^n) (c^n)) →
  (a = b ∨ b = c ∨ a = c) :=
by sorry

end NUMINAMATH_CALUDE_equal_numbers_exist_l3432_343248


namespace NUMINAMATH_CALUDE_total_volume_calculation_l3432_343276

-- Define the dimensions of the rectangular parallelepiped
def box_length : ℝ := 2
def box_width : ℝ := 3
def box_height : ℝ := 4

-- Define the radius of half-spheres and cylinders
def sphere_radius : ℝ := 1
def cylinder_radius : ℝ := 1

-- Define the number of vertices and edges
def num_vertices : ℕ := 8
def num_edges : ℕ := 12

-- Theorem statement
theorem total_volume_calculation :
  let box_volume := box_length * box_width * box_height
  let half_sphere_volume := (num_vertices : ℝ) * (1/2) * (4/3) * Real.pi * sphere_radius^3
  let cylinder_volume := Real.pi * cylinder_radius^2 * 
    (2 * box_length + 2 * box_width + 2 * box_height)
  let total_volume := box_volume + half_sphere_volume + cylinder_volume
  total_volume = (72 + 112 * Real.pi) / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_total_volume_calculation_l3432_343276


namespace NUMINAMATH_CALUDE_min_rolls_for_two_sixes_l3432_343266

/-- The probability of getting two sixes in a single roll of two dice -/
def p : ℚ := 1 / 36

/-- The probability of not getting two sixes in a single roll of two dice -/
def q : ℚ := 1 - p

/-- The number of rolls -/
def n : ℕ := 25

/-- The theorem stating that n is the minimum number of rolls required -/
theorem min_rolls_for_two_sixes (n : ℕ) : 
  (1 - q ^ n > (1 : ℚ) / 2) ∧ ∀ m < n, (1 - q ^ m ≤ (1 : ℚ) / 2) :=
sorry

end NUMINAMATH_CALUDE_min_rolls_for_two_sixes_l3432_343266


namespace NUMINAMATH_CALUDE_star_composition_l3432_343245

-- Define the star operation
def star (x y : ℝ) : ℝ := x^3 - x*y

-- Theorem statement
theorem star_composition (j : ℝ) : star j (star j j) = 2*j^3 - j^4 := by
  sorry

end NUMINAMATH_CALUDE_star_composition_l3432_343245


namespace NUMINAMATH_CALUDE_p_squared_plus_20_not_prime_l3432_343237

theorem p_squared_plus_20_not_prime (p : ℕ) (h : Prime p) : ¬ Prime (p^2 + 20) := by
  sorry

end NUMINAMATH_CALUDE_p_squared_plus_20_not_prime_l3432_343237


namespace NUMINAMATH_CALUDE_mary_weight_change_ratio_l3432_343265

/-- Represents the sequence of weight changes in Mary's diet journey -/
structure WeightChange where
  initial_weight : ℝ
  initial_loss : ℝ
  second_gain : ℝ
  third_loss : ℝ
  final_gain : ℝ
  final_weight : ℝ

/-- Theorem representing Mary's weight change problem -/
theorem mary_weight_change_ratio (w : WeightChange)
  (h1 : w.initial_weight = 99)
  (h2 : w.initial_loss = 12)
  (h3 : w.second_gain = 2 * w.initial_loss)
  (h4 : w.final_gain = 6)
  (h5 : w.final_weight = 81)
  (h6 : w.initial_weight - w.initial_loss + w.second_gain - w.third_loss + w.final_gain = w.final_weight) :
  w.third_loss / w.initial_loss = 3 := by
  sorry


end NUMINAMATH_CALUDE_mary_weight_change_ratio_l3432_343265


namespace NUMINAMATH_CALUDE_apartment_rent_theorem_l3432_343285

/-- Calculates the total rent paid over a period of time with different monthly rates -/
def totalRent (months1 : ℕ) (rate1 : ℕ) (months2 : ℕ) (rate2 : ℕ) : ℕ :=
  months1 * rate1 + months2 * rate2

theorem apartment_rent_theorem :
  totalRent 36 300 24 350 = 19200 := by
  sorry

end NUMINAMATH_CALUDE_apartment_rent_theorem_l3432_343285


namespace NUMINAMATH_CALUDE_sqrt_50_simplified_l3432_343200

theorem sqrt_50_simplified : Real.sqrt 50 = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_50_simplified_l3432_343200
