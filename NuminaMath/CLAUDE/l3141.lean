import Mathlib

namespace NUMINAMATH_CALUDE_pond_volume_l3141_314177

/-- The volume of a rectangular prism with given dimensions is 1000 cubic meters. -/
theorem pond_volume (length width depth : ℝ) (h1 : length = 20) (h2 : width = 10) (h3 : depth = 5) :
  length * width * depth = 1000 :=
by sorry

end NUMINAMATH_CALUDE_pond_volume_l3141_314177


namespace NUMINAMATH_CALUDE_hot_water_bottle_price_l3141_314186

/-- Proves that the price of a hot-water bottle is 6 dollars given the problem conditions --/
theorem hot_water_bottle_price :
  let thermometer_price : ℚ := 2
  let total_sales : ℚ := 1200
  let thermometer_to_bottle_ratio : ℕ := 7
  let bottles_sold : ℕ := 60
  let thermometers_sold : ℕ := thermometer_to_bottle_ratio * bottles_sold
  let bottle_price : ℚ := (total_sales - (thermometer_price * thermometers_sold)) / bottles_sold
  bottle_price = 6 :=
by sorry

end NUMINAMATH_CALUDE_hot_water_bottle_price_l3141_314186


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l3141_314178

theorem binomial_coefficient_equality (n : ℕ) (h : n ≥ 6) :
  (3^5 : ℚ) * (Nat.choose n 5) = (3^6 : ℚ) * (Nat.choose n 6) ↔ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l3141_314178


namespace NUMINAMATH_CALUDE_slope_intercept_sum_l3141_314129

/-- Given points P, Q, R in a plane, and G as the midpoint of PQ,
    prove that the sum of the slope and y-intercept of line RG is 9/2. -/
theorem slope_intercept_sum (P Q R G : ℝ × ℝ) : 
  P = (0, 10) →
  Q = (0, 0) →
  R = (10, 0) →
  G = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2) →
  let slope := (G.2 - R.2) / (G.1 - R.1)
  let y_intercept := G.2 - slope * G.1
  slope + y_intercept = 9/2 := by
sorry

end NUMINAMATH_CALUDE_slope_intercept_sum_l3141_314129


namespace NUMINAMATH_CALUDE_midway_point_distance_yendor_midway_distance_l3141_314158

/-- Represents an elliptical orbit -/
structure EllipticalOrbit where
  /-- Length of the major axis -/
  major_axis : ℝ
  /-- Distance between the foci -/
  focal_distance : ℝ
  /-- Assumption that the focal distance is less than the major axis -/
  h_focal_lt_major : focal_distance < major_axis

/-- A point on the elliptical orbit -/
structure OrbitPoint (orbit : EllipticalOrbit) where
  /-- Distance from the point to the first focus -/
  dist_focus1 : ℝ
  /-- Distance from the point to the second focus -/
  dist_focus2 : ℝ
  /-- The sum of distances to foci equals the major axis -/
  h_sum_dist : dist_focus1 + dist_focus2 = orbit.major_axis

/-- Theorem: For a point midway along the orbit, its distance to either focus is half the major axis -/
theorem midway_point_distance (orbit : EllipticalOrbit) 
    (point : OrbitPoint orbit) 
    (h_midway : point.dist_focus1 = point.dist_focus2) : 
    point.dist_focus1 = orbit.major_axis / 2 := by sorry

/-- The specific orbit from the problem -/
def yendor_orbit : EllipticalOrbit where
  major_axis := 18
  focal_distance := 12
  h_focal_lt_major := by norm_num

/-- Theorem: In Yendor's orbit, a midway point is 9 AU from each focus -/
theorem yendor_midway_distance (point : OrbitPoint yendor_orbit) 
    (h_midway : point.dist_focus1 = point.dist_focus2) : 
    point.dist_focus1 = 9 ∧ point.dist_focus2 = 9 := by sorry

end NUMINAMATH_CALUDE_midway_point_distance_yendor_midway_distance_l3141_314158


namespace NUMINAMATH_CALUDE_lcm_18_24_l3141_314169

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  sorry

end NUMINAMATH_CALUDE_lcm_18_24_l3141_314169


namespace NUMINAMATH_CALUDE_banana_distribution_l3141_314152

theorem banana_distribution (B N : ℕ) : 
  B = 2 * N ∧ B = 4 * (N - 320) → N = 640 := by
  sorry

end NUMINAMATH_CALUDE_banana_distribution_l3141_314152


namespace NUMINAMATH_CALUDE_incorrect_steps_count_l3141_314168

theorem incorrect_steps_count (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : 
  ∃ (s1 s2 s3 : Prop),
    (s1 ↔ (a * c > b * c ∧ b * c > b * d)) ∧
    (s2 ↔ (a * c > b * c ∧ b * c > b * d → a * c > b * d)) ∧
    (s3 ↔ (a * c > b * d → a / d > b / c)) ∧
    (¬s1 ∧ s2 ∧ ¬s3) :=
by sorry


end NUMINAMATH_CALUDE_incorrect_steps_count_l3141_314168


namespace NUMINAMATH_CALUDE_divisibility_by_240_l3141_314130

theorem divisibility_by_240 (a b c d : ℕ) : 
  240 ∣ (a^(4*b+d) - a^(4*c+d)) := by sorry

end NUMINAMATH_CALUDE_divisibility_by_240_l3141_314130


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l3141_314120

theorem fraction_sum_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = x * y - 1) :
  x / y + y / x = (x^2 * y^2 - 4 * x * y + 1) / (x * y) := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l3141_314120


namespace NUMINAMATH_CALUDE_first_valid_year_is_2028_l3141_314102

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def is_valid_year (year : ℕ) : Prop :=
  year > 2020 ∧ sum_of_digits year = 10

theorem first_valid_year_is_2028 :
  ∀ year : ℕ, year < 2028 → ¬(is_valid_year year) ∧ is_valid_year 2028 :=
sorry

end NUMINAMATH_CALUDE_first_valid_year_is_2028_l3141_314102


namespace NUMINAMATH_CALUDE_det_A_l3141_314136

/-- The matrix A as described in the problem -/
def A (n : ℕ) : Matrix (Fin n) (Fin n) ℚ :=
  λ i j => 1 / (min i.val j.val + 1 : ℚ)

/-- The theorem stating the determinant of matrix A -/
theorem det_A (n : ℕ) : 
  Matrix.det (A n) = (-1 : ℚ)^(n-1) / ((Nat.factorial (n-1)) * (Nat.factorial n)) := by
  sorry

end NUMINAMATH_CALUDE_det_A_l3141_314136


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l3141_314147

theorem gcd_of_three_numbers : Nat.gcd 8885 (Nat.gcd 4514 5246) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l3141_314147


namespace NUMINAMATH_CALUDE_bartender_cheating_l3141_314149

theorem bartender_cheating (total_cost : ℚ) (whiskey_cost pipe_cost : ℕ) : 
  total_cost = 11.80 ∧ whiskey_cost = 3 ∧ pipe_cost = 6 → ¬(∃ n : ℕ, total_cost = n * 3) :=
by sorry

end NUMINAMATH_CALUDE_bartender_cheating_l3141_314149


namespace NUMINAMATH_CALUDE_division_problem_l3141_314197

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) :
  dividend = 271 →
  divisor = 30 →
  remainder = 1 →
  dividend = divisor * quotient + remainder →
  quotient = 9 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l3141_314197


namespace NUMINAMATH_CALUDE_overtime_hours_is_eight_l3141_314139

/-- Represents the payment structure and work hours for a worker --/
structure WorkerPayment where
  ordinary_rate : ℚ  -- Rate for ordinary hours in cents
  overtime_rate : ℚ  -- Rate for overtime hours in cents
  total_hours : ℕ    -- Total hours worked
  total_pay : ℚ      -- Total pay in cents

/-- Calculates the number of overtime hours --/
def calculate_overtime_hours (w : WorkerPayment) : ℚ :=
  (w.total_pay - w.ordinary_rate * w.total_hours) / (w.overtime_rate - w.ordinary_rate)

/-- Theorem stating that under given conditions, the overtime hours are 8 --/
theorem overtime_hours_is_eight :
  let w := WorkerPayment.mk 60 90 50 3240
  calculate_overtime_hours w = 8 := by sorry

end NUMINAMATH_CALUDE_overtime_hours_is_eight_l3141_314139


namespace NUMINAMATH_CALUDE_table_runners_area_l3141_314140

theorem table_runners_area (table_area : ℝ) (covered_percentage : ℝ) 
  (two_layer_area : ℝ) (three_layer_area : ℝ) :
  table_area = 175 →
  covered_percentage = 0.8 →
  two_layer_area = 24 →
  three_layer_area = 24 →
  ∃ (total_area : ℝ), total_area = 188 ∧ 
    total_area = (covered_percentage * table_area - 2 * three_layer_area - two_layer_area) + 
                 2 * two_layer_area + 3 * three_layer_area :=
by sorry

end NUMINAMATH_CALUDE_table_runners_area_l3141_314140


namespace NUMINAMATH_CALUDE_sin_cos_product_l3141_314118

theorem sin_cos_product (θ : Real) (h : Real.tan (θ + Real.pi / 2) = 2) :
  Real.sin θ * Real.cos θ = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_product_l3141_314118


namespace NUMINAMATH_CALUDE_bennys_total_work_hours_l3141_314110

/-- Calculates the total hours worked given the hours per day and number of days -/
def total_hours (hours_per_day : ℕ) (days : ℕ) : ℕ :=
  hours_per_day * days

/-- Theorem: Benny's total work hours -/
theorem bennys_total_work_hours :
  let hours_per_day : ℕ := 3
  let days : ℕ := 6
  total_hours hours_per_day days = 18 := by
  sorry

end NUMINAMATH_CALUDE_bennys_total_work_hours_l3141_314110


namespace NUMINAMATH_CALUDE_factorization_ax2_minus_a_l3141_314119

theorem factorization_ax2_minus_a (a x : ℝ) : a * x^2 - a = a * (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_ax2_minus_a_l3141_314119


namespace NUMINAMATH_CALUDE_weight_loss_difference_l3141_314179

/-- Given Barbi's and Luca's weight loss rates and durations, prove the difference in their total weight losses -/
theorem weight_loss_difference (barbi_monthly_loss : ℝ) (barbi_months : ℕ) 
  (luca_yearly_loss : ℝ) (luca_years : ℕ) : 
  barbi_monthly_loss = 1.5 → 
  barbi_months = 12 → 
  luca_yearly_loss = 9 → 
  luca_years = 11 → 
  luca_yearly_loss * luca_years - barbi_monthly_loss * barbi_months = 81 := by
  sorry

end NUMINAMATH_CALUDE_weight_loss_difference_l3141_314179


namespace NUMINAMATH_CALUDE_quadratic_equation_root_l3141_314155

theorem quadratic_equation_root (x : ℝ) : x^2 + 6*x + 4 = 0 ↔ x = Real.sqrt 5 - 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_l3141_314155


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l3141_314128

theorem simplify_trig_expression (α : Real) (h : 270 * π / 180 < α ∧ α < 360 * π / 180) :
  Real.sqrt (1/2 + 1/2 * Real.sqrt (1/2 + 1/2 * Real.cos (2 * α))) = -Real.cos (α / 2) := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l3141_314128


namespace NUMINAMATH_CALUDE_unique_divisor_with_remainder_l3141_314132

theorem unique_divisor_with_remainder (d : ℕ) : 
  d > 0 ∧ d ≥ 10 ∧ d ≤ 99 ∧ (145 % d = 4) → d = 47 := by
  sorry

end NUMINAMATH_CALUDE_unique_divisor_with_remainder_l3141_314132


namespace NUMINAMATH_CALUDE_tangent_line_cubic_l3141_314135

/-- The equation of the tangent line to y = x³ at (2, 8) is y = 12x - 16 -/
theorem tangent_line_cubic (x y : ℝ) :
  (y = x^3) →  -- curve equation
  (∃ (m b : ℝ), y - 8 = m * (x - 2) ∧ y = m * x + b) →  -- point-slope form of tangent line
  (y = 12 * x - 16) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_cubic_l3141_314135


namespace NUMINAMATH_CALUDE_intersection_implies_b_range_l3141_314174

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p | p.1^2 + 2*p.2^2 = 3}
def N (m b : ℝ) : Set (ℝ × ℝ) := {p | p.2 = m*p.1 + b}

-- State the theorem
theorem intersection_implies_b_range :
  (∀ m : ℝ, ∃ p : ℝ × ℝ, p ∈ M ∩ N m b) →
  b ∈ Set.Icc (-Real.sqrt 6 / 2) (Real.sqrt 6 / 2) := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_b_range_l3141_314174


namespace NUMINAMATH_CALUDE_largest_integer_in_interval_l3141_314123

theorem largest_integer_in_interval : 
  ∃ (y : ℤ), (1/4 : ℚ) < (y : ℚ)/6 ∧ (y : ℚ)/6 < 7/12 ∧ 
  ∀ (z : ℤ), (1/4 : ℚ) < (z : ℚ)/6 → (z : ℚ)/6 < 7/12 → z ≤ y :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_in_interval_l3141_314123


namespace NUMINAMATH_CALUDE_forty_coins_impossible_l3141_314182

/-- Represents the contents of Bethany's purse -/
structure Purse where
  pound_coins : ℕ
  twenty_pence : ℕ
  fifty_pence : ℕ

/-- Calculates the total value of coins in pence -/
def total_value (p : Purse) : ℕ :=
  100 * p.pound_coins + 20 * p.twenty_pence + 50 * p.fifty_pence

/-- Calculates the total number of coins -/
def total_coins (p : Purse) : ℕ :=
  p.pound_coins + p.twenty_pence + p.fifty_pence

/-- Represents Bethany's purse with the given conditions -/
def bethany_purse : Purse :=
  { pound_coins := 11
  , twenty_pence := 0  -- placeholder, actual value unknown
  , fifty_pence := 0 } -- placeholder, actual value unknown

/-- The mean value of coins in pence -/
def mean_value : ℚ := 52

theorem forty_coins_impossible :
  ∀ p : Purse,
    p.pound_coins = 11 →
    (total_value p : ℚ) / (total_coins p : ℚ) = mean_value →
    total_coins p ≠ 40 :=
by sorry

end NUMINAMATH_CALUDE_forty_coins_impossible_l3141_314182


namespace NUMINAMATH_CALUDE_geometric_series_sum_l3141_314153

theorem geometric_series_sum : 
  let a : ℚ := 1/4
  let r : ℚ := 1/4
  let n : ℕ := 5
  let series_sum := (a * (1 - r^n)) / (1 - r)
  series_sum = 341/1024 := by sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l3141_314153


namespace NUMINAMATH_CALUDE_sum_of_medians_is_63_l3141_314170

/-- Represents the scores of a basketball player -/
def Scores := List ℕ

/-- Calculates the median of a list of scores -/
def median (scores : Scores) : ℚ :=
  sorry

/-- Player A's scores -/
def scoresA : Scores :=
  sorry

/-- Player B's scores -/
def scoresB : Scores :=
  sorry

/-- The sum of median scores of players A and B is 63 -/
theorem sum_of_medians_is_63 :
  median scoresA + median scoresB = 63 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_medians_is_63_l3141_314170


namespace NUMINAMATH_CALUDE_union_M_N_complement_M_U_l3141_314164

-- Define the universe set U
def U : Set Nat := {1, 2, 3, 4, 5, 6}

-- Define set M
def M : Set Nat := {2, 3, 4}

-- Define set N
def N : Set Nat := {4, 5}

-- Theorem for the union of M and N
theorem union_M_N : M ∪ N = {2, 3, 4, 5} := by sorry

-- Theorem for the complement of M with respect to U
theorem complement_M_U : (U \ M) = {1, 5, 6} := by sorry

end NUMINAMATH_CALUDE_union_M_N_complement_M_U_l3141_314164


namespace NUMINAMATH_CALUDE_max_d_is_zero_l3141_314125

/-- Represents a 6-digit number of the form 6d6,33e -/
def SixDigitNumber (d e : Nat) : Nat :=
  606330 + d * 1000 + e

theorem max_d_is_zero :
  ∀ d e : Nat,
    d < 10 →
    e < 10 →
    SixDigitNumber d e % 33 = 0 →
    d ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_max_d_is_zero_l3141_314125


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l3141_314146

/-- Given a line y = kx + b that is parallel to y = 2x - 3 and passes through (1, -5),
    prove that its equation is y = 2x - 7 -/
theorem parallel_line_through_point (k b : ℝ) : 
  (∀ x y, y = k * x + b ↔ y = 2 * x - 3) →  -- parallelism condition
  (-5 : ℝ) = k * 1 + b →                   -- point condition
  ∀ x y, y = k * x + b ↔ y = 2 * x - 7 :=   -- conclusion
by sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l3141_314146


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3141_314196

theorem inequality_solution_set : 
  {x : ℝ | (1/2 - x) * (x - 1/3) > 0} = {x : ℝ | 1/3 < x ∧ x < 1/2} :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3141_314196


namespace NUMINAMATH_CALUDE_sum_and_reciprocal_sum_l3141_314183

theorem sum_and_reciprocal_sum (x : ℝ) (h : x ≠ 0) :
  x^2 + (1/x)^2 = 10.25 → x + (1/x) = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_reciprocal_sum_l3141_314183


namespace NUMINAMATH_CALUDE_beacon_population_l3141_314150

def richmond_population : ℕ := 3000
def richmond_victoria_difference : ℕ := 1000
def victoria_beacon_ratio : ℕ := 4

theorem beacon_population : 
  ∃ (victoria_population beacon_population : ℕ),
    richmond_population = victoria_population + richmond_victoria_difference ∧
    victoria_population = victoria_beacon_ratio * beacon_population ∧
    beacon_population = 500 := by
  sorry

end NUMINAMATH_CALUDE_beacon_population_l3141_314150


namespace NUMINAMATH_CALUDE_jane_babysitting_problem_l3141_314180

/-- Represents the problem of determining when Jane stopped babysitting --/
theorem jane_babysitting_problem (jane_start_age : ℕ) (jane_current_age : ℕ) (oldest_babysat_current_age : ℕ) :
  jane_start_age = 20 →
  jane_current_age = 32 →
  oldest_babysat_current_age = 22 →
  (∀ (jane_age : ℕ) (child_age : ℕ),
    jane_start_age ≤ jane_age →
    jane_age ≤ jane_current_age →
    child_age ≤ oldest_babysat_current_age →
    child_age ≤ jane_age / 2) →
  jane_current_age - jane_start_age = 12 :=
by sorry

end NUMINAMATH_CALUDE_jane_babysitting_problem_l3141_314180


namespace NUMINAMATH_CALUDE_quadratic_property_l3141_314142

/-- A quadratic function with a real coefficient b -/
def f (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x + 3

/-- The range of f is [0, +∞) -/
def has_nonnegative_range (b : ℝ) : Prop :=
  ∀ y, (∃ x, f b x = y) → y ≥ 0

/-- The solution set of f(x) < c is an open interval of length 8 -/
def has_solution_interval_of_length_eight (b c : ℝ) : Prop :=
  ∃ m, ∀ x, f b x < c ↔ m - 8 < x ∧ x < m

theorem quadratic_property (b : ℝ) (h1 : has_nonnegative_range b) 
  (h2 : has_solution_interval_of_length_eight b c) : c = 16 :=
sorry

end NUMINAMATH_CALUDE_quadratic_property_l3141_314142


namespace NUMINAMATH_CALUDE_journey_time_difference_l3141_314105

/-- Represents the speed of the bus in miles per hour -/
def speed : ℝ := 60

/-- Represents the distance of the first journey in miles -/
def distance1 : ℝ := 360

/-- Represents the distance of the second journey in miles -/
def distance2 : ℝ := 420

/-- Theorem stating the difference in travel time between the two journeys -/
theorem journey_time_difference : 
  (distance2 - distance1) / speed * 60 = 60 := by
  sorry

end NUMINAMATH_CALUDE_journey_time_difference_l3141_314105


namespace NUMINAMATH_CALUDE_seed_distribution_l3141_314167

theorem seed_distribution (total_seeds : ℕ) (num_pots : ℕ) 
  (h1 : total_seeds = 10) 
  (h2 : num_pots = 4) : 
  ∃ (pot1 pot2 pot3 pot4 : ℕ), 
    pot1 = 2 * pot2 ∧ 
    pot3 = pot2 + 1 ∧ 
    pot1 + pot2 + pot3 + pot4 = total_seeds ∧ 
    pot4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_seed_distribution_l3141_314167


namespace NUMINAMATH_CALUDE_max_value_x_plus_2y_l3141_314151

theorem max_value_x_plus_2y (x y : ℝ) 
  (h1 : 4 * x + 3 * y ≤ 12) 
  (h2 : 3 * x + 6 * y ≤ 9) : 
  ∃ (max : ℝ), max = 3 ∧ x + 2 * y ≤ max ∧ 
  ∀ (z : ℝ), (∃ (a b : ℝ), 4 * a + 3 * b ≤ 12 ∧ 3 * a + 6 * b ≤ 9 ∧ z = a + 2 * b) → z ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_x_plus_2y_l3141_314151


namespace NUMINAMATH_CALUDE_sum_of_solutions_square_equation_l3141_314199

theorem sum_of_solutions_square_equation : 
  ∃ (x₁ x₂ : ℝ), (x₁ - 8)^2 = 49 ∧ (x₂ - 8)^2 = 49 ∧ x₁ + x₂ = 16 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_square_equation_l3141_314199


namespace NUMINAMATH_CALUDE_choose_four_from_six_l3141_314192

theorem choose_four_from_six : Nat.choose 6 4 = 15 := by
  sorry

end NUMINAMATH_CALUDE_choose_four_from_six_l3141_314192


namespace NUMINAMATH_CALUDE_unique_solution_value_l3141_314185

theorem unique_solution_value (p : ℝ) : 
  (∃! x : ℝ, x ≠ 0 ∧ (1 : ℝ) / (3 * x) = (p - x) / 4) ↔ p = 4 / 3 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_value_l3141_314185


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l3141_314109

theorem contrapositive_equivalence (x : ℝ) :
  (x ≠ 3 ∧ x ≠ 4 → x^2 - 7*x + 12 ≠ 0) ↔ (x^2 - 7*x + 12 = 0 → x = 3 ∨ x = 4) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l3141_314109


namespace NUMINAMATH_CALUDE_reading_time_for_18_pages_l3141_314101

-- Define the reading rate (pages per minute)
def reading_rate : ℚ := 4 / 2

-- Define the number of pages to read
def pages_to_read : ℕ := 18

-- Theorem: It takes 9 minutes to read 18 pages at the given rate
theorem reading_time_for_18_pages :
  (pages_to_read : ℚ) / reading_rate = 9 := by
  sorry

end NUMINAMATH_CALUDE_reading_time_for_18_pages_l3141_314101


namespace NUMINAMATH_CALUDE_sqrt_two_nine_two_equals_six_l3141_314106

theorem sqrt_two_nine_two_equals_six : Real.sqrt (2 * 9 * 2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_nine_two_equals_six_l3141_314106


namespace NUMINAMATH_CALUDE_tims_income_percentage_l3141_314113

theorem tims_income_percentage (tim mart juan : ℝ) 
  (h1 : mart = 1.6 * tim) 
  (h2 : mart = 0.8 * juan) : 
  tim = 0.5 * juan := by
  sorry

end NUMINAMATH_CALUDE_tims_income_percentage_l3141_314113


namespace NUMINAMATH_CALUDE_tims_sock_drawer_probability_l3141_314126

/-- Represents the number of socks of each color in the drawer -/
structure SockDrawer where
  gray : ℕ
  white : ℕ
  black : ℕ

/-- Calculates the probability of picking a matching pair of socks -/
def probabilityOfMatchingPair (drawer : SockDrawer) : ℚ :=
  let totalSocks := drawer.gray + drawer.white + drawer.black
  let totalPairs := (totalSocks * (totalSocks - 1)) / 2
  let matchingPairs := (drawer.gray * (drawer.gray - 1) + 
                        drawer.white * (drawer.white - 1) + 
                        drawer.black * (drawer.black - 1)) / 2
  matchingPairs / totalPairs

/-- Theorem stating that the probability of picking a matching pair 
    from Tim's sock drawer is 1/3 -/
theorem tims_sock_drawer_probability : 
  probabilityOfMatchingPair ⟨12, 10, 6⟩ = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_tims_sock_drawer_probability_l3141_314126


namespace NUMINAMATH_CALUDE_triangle_side_length_l3141_314160

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- Ensure positive side lengths
  A = π/3 →  -- 60 degrees in radians
  B = π/4 →  -- 45 degrees in radians
  b = Real.sqrt 6 →
  a + b + c = A + B + C →  -- Triangle angle sum theorem
  a / Real.sin A = b / Real.sin B →  -- Sine rule
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3141_314160


namespace NUMINAMATH_CALUDE_maggie_bouncy_balls_l3141_314187

/-- The number of packs of red bouncy balls Maggie bought -/
def red_packs : ℕ := 4

/-- The number of packs of yellow bouncy balls Maggie bought -/
def yellow_packs : ℕ := 8

/-- The number of packs of green bouncy balls Maggie bought -/
def green_packs : ℕ := 4

/-- The number of bouncy balls in each pack -/
def balls_per_pack : ℕ := 10

/-- The total number of bouncy balls Maggie bought -/
def total_balls : ℕ := red_packs * balls_per_pack + yellow_packs * balls_per_pack + green_packs * balls_per_pack

theorem maggie_bouncy_balls : total_balls = 160 := by
  sorry

end NUMINAMATH_CALUDE_maggie_bouncy_balls_l3141_314187


namespace NUMINAMATH_CALUDE_movie_ticket_theorem_l3141_314103

def movie_ticket_problem (child_ticket_price adult_ticket_price : ℚ) : Prop :=
  let total_spent : ℚ := 30
  let num_child_tickets : ℕ := 4
  let num_adult_tickets : ℕ := 2
  let discount : ℚ := 2
  child_ticket_price = 4.25 ∧
  adult_ticket_price > child_ticket_price ∧
  num_child_tickets + num_adult_tickets > 3 ∧
  num_child_tickets * child_ticket_price + num_adult_tickets * adult_ticket_price - discount = total_spent ∧
  adult_ticket_price - child_ticket_price = 3.25

theorem movie_ticket_theorem :
  ∃ (adult_ticket_price : ℚ), movie_ticket_problem 4.25 adult_ticket_price :=
sorry

end NUMINAMATH_CALUDE_movie_ticket_theorem_l3141_314103


namespace NUMINAMATH_CALUDE_sum_of_powers_of_i_is_zero_l3141_314176

/-- The imaginary unit i -/
def i : ℂ := Complex.I

/-- Theorem stating that i^8621 + i^8622 + i^8623 + i^8624 + i^8625 = 0 -/
theorem sum_of_powers_of_i_is_zero :
  i^8621 + i^8622 + i^8623 + i^8624 + i^8625 = 0 := by sorry

end NUMINAMATH_CALUDE_sum_of_powers_of_i_is_zero_l3141_314176


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l3141_314189

theorem completing_square_equivalence (x : ℝ) :
  x^2 - 6*x - 10 = 0 ↔ (x - 3)^2 = 19 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l3141_314189


namespace NUMINAMATH_CALUDE_e₁_e₂_form_basis_l3141_314143

def e₁ : ℝ × ℝ := (-1, 2)
def e₂ : ℝ × ℝ := (5, 7)

def are_collinear (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

def form_basis (v w : ℝ × ℝ) : Prop :=
  ¬(are_collinear v w)

theorem e₁_e₂_form_basis : form_basis e₁ e₂ := by
  sorry

end NUMINAMATH_CALUDE_e₁_e₂_form_basis_l3141_314143


namespace NUMINAMATH_CALUDE_equation_equality_l3141_314134

theorem equation_equality : 27474 + 3699 + 1985 - 2047 = 31111 := by
  sorry

end NUMINAMATH_CALUDE_equation_equality_l3141_314134


namespace NUMINAMATH_CALUDE_license_plate_difference_l3141_314131

/-- The number of letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of digits -/
def num_digits : ℕ := 10

/-- The number of possible California license plates -/
def california_plates : ℕ := num_letters^4 * num_digits^2

/-- The number of possible New York license plates -/
def new_york_plates : ℕ := num_letters^3 * num_digits^3

/-- The difference in the number of license plates between California and New York -/
theorem license_plate_difference :
  california_plates - new_york_plates = 28121600 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_difference_l3141_314131


namespace NUMINAMATH_CALUDE_toy_store_shelves_l3141_314193

/-- Calculates the number of shelves needed for a given number of items and shelf capacity -/
def shelves_needed (items : ℕ) (capacity : ℕ) : ℕ :=
  (items + capacity - 1) / capacity

/-- Proves that the total number of shelves needed for bears and rabbits is 6 -/
theorem toy_store_shelves : 
  let initial_bears : ℕ := 17
  let initial_rabbits : ℕ := 20
  let new_bears : ℕ := 10
  let new_rabbits : ℕ := 15
  let sold_bears : ℕ := 5
  let sold_rabbits : ℕ := 7
  let bear_shelf_capacity : ℕ := 9
  let rabbit_shelf_capacity : ℕ := 12
  let remaining_bears : ℕ := initial_bears + new_bears - sold_bears
  let remaining_rabbits : ℕ := initial_rabbits + new_rabbits - sold_rabbits
  let bear_shelves : ℕ := shelves_needed remaining_bears bear_shelf_capacity
  let rabbit_shelves : ℕ := shelves_needed remaining_rabbits rabbit_shelf_capacity
  bear_shelves + rabbit_shelves = 6 :=
by sorry

end NUMINAMATH_CALUDE_toy_store_shelves_l3141_314193


namespace NUMINAMATH_CALUDE_part_one_part_two_l3141_314115

-- Define the function f
def f (c : ℝ) (x : ℝ) : ℝ := |x - c|

-- Part I: Prove that f(x) + f(-1/x) ≥ 2 for any real x and c
theorem part_one (c : ℝ) (x : ℝ) : f c x + f c (-1/x) ≥ 2 :=
sorry

-- Part II: Prove that for c = 4, the solution set of |f(1/2x+c) - 1/2f(x)| ≤ 1 is {x | 1 ≤ x ≤ 3}
theorem part_two :
  let c : ℝ := 4
  ∀ x : ℝ, |f c (1/2 * x + c) - 1/2 * f c x| ≤ 1 ↔ 1 ≤ x ∧ x ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3141_314115


namespace NUMINAMATH_CALUDE_fraction_meaningful_l3141_314188

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x - 4)) ↔ x ≠ 4 := by
sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l3141_314188


namespace NUMINAMATH_CALUDE_probability_inside_circle_l3141_314117

def is_inside_circle (x y : ℕ) : Prop := x^2 + y^2 < 9

def favorable_outcomes : ℕ := 4

def total_outcomes : ℕ := 36

theorem probability_inside_circle :
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 9 :=
sorry

end NUMINAMATH_CALUDE_probability_inside_circle_l3141_314117


namespace NUMINAMATH_CALUDE_log_expression_equals_two_l3141_314166

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_expression_equals_two :
  (log10 2)^2 + log10 2 * log10 50 + log10 25 = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_expression_equals_two_l3141_314166


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3141_314138

/-- An arithmetic sequence with given properties -/
def ArithmeticSequence (a₁ : ℝ) (d : ℝ) : ℕ → ℝ :=
  fun n => a₁ + (n - 1) * d

theorem arithmetic_sequence_problem (seq : ℕ → ℝ) :
  (∃ a₁ d : ℝ, seq = ArithmeticSequence a₁ d ∧ 
    seq 3 = 14 ∧ seq 6 = 32) →
  seq 10 = 56 ∧ (∃ d : ℝ, ∀ n : ℕ, seq (n + 1) - seq n = d ∧ d = 6) :=
by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3141_314138


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l3141_314194

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 2) ↔ x ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l3141_314194


namespace NUMINAMATH_CALUDE_expected_steps_is_five_l3141_314175

/-- The coloring process on the unit interval [0,1] --/
structure ColoringProcess where
  /-- The random selection of x in [0,1] --/
  select_x : Unit → Real
  /-- The coloring rule for x ≤ 1/2 --/
  color_left (x : Real) : Set Real := { y | x ≤ y ∧ y ≤ x + 1/2 }
  /-- The coloring rule for x > 1/2 --/
  color_right (x : Real) : Set Real := { y | x ≤ y ∧ y ≤ 1 } ∪ { y | 0 ≤ y ∧ y ≤ x - 1/2 }

/-- The expected number of steps to color the entire interval --/
def expected_steps (process : ColoringProcess) : Real :=
  5  -- The actual value we want to prove

/-- The theorem stating that the expected number of steps is 5 --/
theorem expected_steps_is_five (process : ColoringProcess) :
  expected_steps process = 5 := by sorry

end NUMINAMATH_CALUDE_expected_steps_is_five_l3141_314175


namespace NUMINAMATH_CALUDE_range_of_a_l3141_314124

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x^2 + (a + 1) * x + 1 < 0) → 
  a ∈ Set.Iio (-3) ∪ Set.Ioi 1 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l3141_314124


namespace NUMINAMATH_CALUDE_sum_of_xy_l3141_314107

theorem sum_of_xy (x y : ℕ) 
  (pos_x : x > 0) (pos_y : y > 0)
  (bound_x : x < 30) (bound_y : y < 30)
  (eq : x + y + x * y = 94) : x + y = 22 := by
sorry

end NUMINAMATH_CALUDE_sum_of_xy_l3141_314107


namespace NUMINAMATH_CALUDE_maggie_tractor_hours_l3141_314190

/-- Represents Maggie's work schedule and income for a week. -/
structure WorkWeek where
  tractorHours : ℕ
  officeHours : ℕ
  deliveryHours : ℕ
  totalIncome : ℕ

/-- Checks if a work week satisfies the given conditions. -/
def isValidWorkWeek (w : WorkWeek) : Prop :=
  w.officeHours = 2 * w.tractorHours ∧
  w.deliveryHours = w.officeHours - 3 ∧
  w.totalIncome = 10 * w.officeHours + 12 * w.tractorHours + 15 * w.deliveryHours

/-- Theorem stating that given the conditions, Maggie spent 15 hours driving the tractor. -/
theorem maggie_tractor_hours :
  ∃ (w : WorkWeek), isValidWorkWeek w ∧ w.totalIncome = 820 → w.tractorHours = 15 :=
by sorry


end NUMINAMATH_CALUDE_maggie_tractor_hours_l3141_314190


namespace NUMINAMATH_CALUDE_apple_bag_price_l3141_314165

-- Define the given quantities
def total_harvest : ℕ := 405
def juice_amount : ℕ := 90
def restaurant_amount : ℕ := 60
def bag_size : ℕ := 5
def total_revenue : ℕ := 408

-- Define the selling price of one bag
def selling_price : ℚ := 8

-- Theorem to prove
theorem apple_bag_price :
  (total_harvest - juice_amount - restaurant_amount) / bag_size * selling_price = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_apple_bag_price_l3141_314165


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3141_314181

theorem absolute_value_inequality (a b c : ℝ) :
  |a + c| < b → |a| < |b| - |c| := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3141_314181


namespace NUMINAMATH_CALUDE_interest_rate_is_five_percent_l3141_314173

/-- Calculates the interest rate given the principal, time, and simple interest -/
def calculate_interest_rate (principal time simple_interest : ℚ) : ℚ :=
  (simple_interest * 100) / (principal * time)

/-- Proof that the interest rate is 5% given the specified conditions -/
theorem interest_rate_is_five_percent :
  let principal : ℚ := 16065
  let time : ℚ := 5
  let simple_interest : ℚ := 4016.25
  calculate_interest_rate principal time simple_interest = 5 := by
  sorry

#eval calculate_interest_rate 16065 5 4016.25

end NUMINAMATH_CALUDE_interest_rate_is_five_percent_l3141_314173


namespace NUMINAMATH_CALUDE_cubic_expression_zero_l3141_314163

theorem cubic_expression_zero (x : ℝ) (h : x^2 + 3*x - 3 = 0) : 
  x^3 + 2*x^2 - 6*x + 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expression_zero_l3141_314163


namespace NUMINAMATH_CALUDE_greatest_common_measure_l3141_314154

theorem greatest_common_measure (a b c : ℕ) (ha : a = 729000) (hb : b = 1242500) (hc : c = 32175) :
  Nat.gcd a (Nat.gcd b c) = 225 := by
  sorry

end NUMINAMATH_CALUDE_greatest_common_measure_l3141_314154


namespace NUMINAMATH_CALUDE_root_in_interval_l3141_314133

noncomputable def f (x : ℝ) : ℝ := 4 - 4*x - Real.exp x

theorem root_in_interval :
  ∃ x ∈ Set.Ioo 0 1, f x = 0 :=
by
  have h1 : Continuous f := sorry
  have h2 : StrictMono (fun x => -f x) := sorry
  have h3 : f 0 > 0 := sorry
  have h4 : f 1 < 0 := sorry
  sorry

end NUMINAMATH_CALUDE_root_in_interval_l3141_314133


namespace NUMINAMATH_CALUDE_x_value_l3141_314122

theorem x_value : ∃ x : ℝ, (x = 90 * (1 + 11/100)) ∧ (x = 99.9) := by sorry

end NUMINAMATH_CALUDE_x_value_l3141_314122


namespace NUMINAMATH_CALUDE_largest_number_proof_l3141_314191

def is_hcf (a b h : ℕ) : Prop := h ∣ a ∧ h ∣ b ∧ ∀ k : ℕ, k ∣ a → k ∣ b → k ≤ h

def is_lcm (a b l : ℕ) : Prop := a ∣ l ∧ b ∣ l ∧ ∀ k : ℕ, a ∣ k → b ∣ k → l ∣ k

theorem largest_number_proof (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  is_hcf a b 23 → (∃ l : ℕ, is_lcm a b l ∧ 13 ∣ l ∧ 14 ∣ l) → max a b = 322 :=
by sorry

end NUMINAMATH_CALUDE_largest_number_proof_l3141_314191


namespace NUMINAMATH_CALUDE_max_product_other_sides_l3141_314108

/-- Given a triangle with one side of length 4 and the opposite angle of 60°,
    the maximum product of the lengths of the other two sides is 16. -/
theorem max_product_other_sides (a b c : ℝ) (A B C : ℝ) :
  a = 4 →
  A = π / 3 →
  0 < b ∧ 0 < c →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  b * c ≤ 16 :=
sorry

end NUMINAMATH_CALUDE_max_product_other_sides_l3141_314108


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a3_value_l3141_314116

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a3_value
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_a2 : a 2 = 2 * a 3 + 1)
  (h_a4 : a 4 = 2 * a 3 + 7) :
  a 3 = -4 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a3_value_l3141_314116


namespace NUMINAMATH_CALUDE_lg_45_equals_1_minus_m_plus_2n_l3141_314144

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem lg_45_equals_1_minus_m_plus_2n (m n : ℝ) (h1 : lg 2 = m) (h2 : lg 3 = n) :
  lg 45 = 1 - m + 2 * n := by
  sorry

end NUMINAMATH_CALUDE_lg_45_equals_1_minus_m_plus_2n_l3141_314144


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l3141_314100

theorem complex_modulus_problem (z : ℂ) (h : z^2 = -4) : 
  Complex.abs (1 + z) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l3141_314100


namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_l3141_314156

theorem quadratic_real_roots_condition (k : ℝ) :
  (∃ x : ℝ, k * x^2 + 2 * x + 1 = 0) ↔ (k ≤ 1 ∧ k ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_l3141_314156


namespace NUMINAMATH_CALUDE_fourth_term_of_specific_gp_l3141_314104

def geometric_progression (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r^(n - 1)

theorem fourth_term_of_specific_gp :
  let a₁ := 2
  let a₂ := 2 * Real.sqrt 2
  let a₃ := 4
  let r := a₂ / a₁
  let a₄ := geometric_progression a₁ r 4
  a₄ = 4 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_fourth_term_of_specific_gp_l3141_314104


namespace NUMINAMATH_CALUDE_share_face_value_l3141_314198

/-- Given a share with the following properties:
  * dividend_rate: The dividend rate of the share (9%)
  * desired_return: The desired return on investment (12%)
  * market_value: The market value of the share in Rs. (15)
  
  This theorem proves that the face value of the share is Rs. 20. -/
theorem share_face_value
  (dividend_rate : ℝ)
  (desired_return : ℝ)
  (market_value : ℝ)
  (h1 : dividend_rate = 0.09)
  (h2 : desired_return = 0.12)
  (h3 : market_value = 15) :
  (desired_return * market_value) / dividend_rate = 20 := by
  sorry

#eval (0.12 * 15) / 0.09  -- Expected output: 20

end NUMINAMATH_CALUDE_share_face_value_l3141_314198


namespace NUMINAMATH_CALUDE_ratio_problem_l3141_314184

theorem ratio_problem (w x y z : ℝ) (hw : w ≠ 0) 
  (h1 : w / x = 2 / 3) 
  (h2 : w / y = 6 / 15) 
  (h3 : w / z = 4 / 5) : 
  (x + y) / z = 16 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l3141_314184


namespace NUMINAMATH_CALUDE_cell_division_result_l3141_314157

/-- Represents the cell division process over time -/
def cellDivision (initialOrganisms : ℕ) (initialCellsPerOrganism : ℕ) (divisionRatio : ℕ) (daysBetweenDivisions : ℕ) (totalDays : ℕ) : ℕ :=
  let initialCells := initialOrganisms * initialCellsPerOrganism
  let numDivisions := totalDays / daysBetweenDivisions
  initialCells * divisionRatio ^ numDivisions

/-- Theorem stating the result of the cell division process -/
theorem cell_division_result :
  cellDivision 8 4 3 3 12 = 864 := by
  sorry

end NUMINAMATH_CALUDE_cell_division_result_l3141_314157


namespace NUMINAMATH_CALUDE_rhombus_longer_diagonal_l3141_314159

/-- Given a rhombus with diagonal ratio 2:3 and area 12 cm², prove the longer diagonal is 6 cm -/
theorem rhombus_longer_diagonal (d1 d2 : ℝ) : 
  d1 / d2 = 2 / 3 →  -- ratio of diagonals
  d1 * d2 / 2 = 12 →  -- area of rhombus
  d2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_longer_diagonal_l3141_314159


namespace NUMINAMATH_CALUDE_solve_fraction_equation_l3141_314121

theorem solve_fraction_equation :
  ∀ x : ℚ, (1 / 4 : ℚ) - (1 / 6 : ℚ) = 1 / x → x = 12 := by
  sorry

end NUMINAMATH_CALUDE_solve_fraction_equation_l3141_314121


namespace NUMINAMATH_CALUDE_ladder_slide_l3141_314148

theorem ladder_slide (ladder_length : Real) (initial_distance : Real) (top_slip : Real) (foot_slide : Real) : 
  ladder_length = 30 ∧ 
  initial_distance = 8 ∧ 
  top_slip = 4 ∧ 
  foot_slide = 2 →
  (ladder_length ^ 2 = initial_distance ^ 2 + (Real.sqrt (ladder_length ^ 2 - initial_distance ^ 2)) ^ 2) ∧
  (ladder_length ^ 2 = (initial_distance + foot_slide) ^ 2 + (Real.sqrt (ladder_length ^ 2 - initial_distance ^ 2) - top_slip) ^ 2) :=
by sorry

end NUMINAMATH_CALUDE_ladder_slide_l3141_314148


namespace NUMINAMATH_CALUDE_passion_fruit_crates_l3141_314145

theorem passion_fruit_crates (total_crates grapes_crates mangoes_crates : ℕ) 
  (h1 : total_crates = 50)
  (h2 : grapes_crates = 13)
  (h3 : mangoes_crates = 20) :
  total_crates - (grapes_crates + mangoes_crates) = 17 := by
  sorry

end NUMINAMATH_CALUDE_passion_fruit_crates_l3141_314145


namespace NUMINAMATH_CALUDE_circles_internally_tangent_l3141_314127

/-- Two circles are internally tangent if the distance between their centers
    is equal to the difference of their radii -/
def internally_tangent (r₁ r₂ d : ℝ) : Prop :=
  d = abs (r₁ - r₂)

/-- Given two circles with radii 5 cm and 3 cm, with centers 2 cm apart,
    prove that they are internally tangent -/
theorem circles_internally_tangent :
  let r₁ : ℝ := 5  -- radius of larger circle
  let r₂ : ℝ := 3  -- radius of smaller circle
  let d  : ℝ := 2  -- distance between centers
  internally_tangent r₁ r₂ d := by
  sorry

end NUMINAMATH_CALUDE_circles_internally_tangent_l3141_314127


namespace NUMINAMATH_CALUDE_second_interest_rate_is_ten_percent_l3141_314171

/-- Proves that given specific investment conditions, the second interest rate is 10% -/
theorem second_interest_rate_is_ten_percent 
  (total_investment : ℝ)
  (first_investment : ℝ)
  (first_rate : ℝ)
  (h_total : total_investment = 5400)
  (h_first : first_investment = 3000)
  (h_first_rate : first_rate = 0.08)
  (h_equal_interest : first_investment * first_rate = 
    (total_investment - first_investment) * (10 / 100)) :
  (10 : ℝ) / 100 = (first_investment * first_rate) / (total_investment - first_investment) :=
sorry

end NUMINAMATH_CALUDE_second_interest_rate_is_ten_percent_l3141_314171


namespace NUMINAMATH_CALUDE_limit_a_minus_log_n_eq_zero_l3141_314195

noncomputable def a : ℕ → ℝ
  | 0 => 1
  | n + 1 => a n + Real.exp (-a n)

theorem limit_a_minus_log_n_eq_zero :
  ∃ L : ℝ, L = 0 ∧ ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n - Real.log n - L| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_a_minus_log_n_eq_zero_l3141_314195


namespace NUMINAMATH_CALUDE_joint_purchase_effectiveness_l3141_314141

/-- Represents the benefits of joint purchases -/
structure JointPurchaseBenefits where
  cost_savings : ℝ
  quality_assessment : ℝ
  community_trust : ℝ

/-- Represents the drawbacks of joint purchases -/
structure JointPurchaseDrawbacks where
  transaction_costs : ℝ
  organizational_efforts : ℝ
  convenience_issues : ℝ
  potential_disputes : ℝ

/-- Represents the characteristics of a group making joint purchases -/
structure PurchaseGroup where
  size : ℕ
  is_localized : Bool

/-- Calculates the total benefit of joint purchases for a group -/
def calculate_total_benefit (benefits : JointPurchaseBenefits) (group : PurchaseGroup) : ℝ :=
  benefits.cost_savings + benefits.quality_assessment + benefits.community_trust

/-- Calculates the total drawback of joint purchases for a group -/
def calculate_total_drawback (drawbacks : JointPurchaseDrawbacks) (group : PurchaseGroup) : ℝ :=
  drawbacks.transaction_costs + drawbacks.organizational_efforts + drawbacks.convenience_issues + drawbacks.potential_disputes

/-- Theorem stating that joint purchases are beneficial for large groups but not for small, localized groups -/
theorem joint_purchase_effectiveness (benefits : JointPurchaseBenefits) (drawbacks : JointPurchaseDrawbacks) :
  ∀ (group : PurchaseGroup),
    (group.size > 100 → calculate_total_benefit benefits group > calculate_total_drawback drawbacks group) ∧
    (group.size ≤ 100 ∧ group.is_localized → calculate_total_benefit benefits group ≤ calculate_total_drawback drawbacks group) :=
by sorry

end NUMINAMATH_CALUDE_joint_purchase_effectiveness_l3141_314141


namespace NUMINAMATH_CALUDE_parabola_focus_distance_l3141_314114

/-- Theorem: For a parabola y² = ax (a > 0) with a point P(3/2, y₀) on it,
    if the distance from P to the focus is 2, then a = 2. -/
theorem parabola_focus_distance (a : ℝ) (y₀ : ℝ) :
  a > 0 →
  y₀^2 = a * (3/2) →
  2 = (|3/2 - a/4| + |y₀|) →
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_distance_l3141_314114


namespace NUMINAMATH_CALUDE_gumdrop_replacement_l3141_314111

theorem gumdrop_replacement (blue_percent : Real) (brown_percent : Real) 
  (red_percent : Real) (yellow_percent : Real) (green_count : Nat) :
  blue_percent = 0.3 →
  brown_percent = 0.2 →
  red_percent = 0.15 →
  yellow_percent = 0.1 →
  green_count = 30 →
  let total := green_count / (1 - (blue_percent + brown_percent + red_percent + yellow_percent))
  let blue_count := blue_percent * total
  let brown_count := brown_percent * total
  let new_brown_count := brown_count + blue_count / 2
  new_brown_count = 42 := by
  sorry

end NUMINAMATH_CALUDE_gumdrop_replacement_l3141_314111


namespace NUMINAMATH_CALUDE_square_difference_l3141_314112

theorem square_difference (x : ℤ) (h : x^2 = 1764) : (x + 2) * (x - 2) = 1760 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l3141_314112


namespace NUMINAMATH_CALUDE_new_student_weight_l3141_314172

/-- The weight of the new student given the conditions of the problem -/
theorem new_student_weight (n : ℕ) (initial_weight replaced_weight new_weight : ℝ) 
  (h1 : n = 4)
  (h2 : replaced_weight = 96)
  (h3 : (initial_weight - replaced_weight + new_weight) / n = initial_weight / n - 8) :
  new_weight = 64 := by
  sorry

end NUMINAMATH_CALUDE_new_student_weight_l3141_314172


namespace NUMINAMATH_CALUDE_work_completion_time_l3141_314162

/-- Given that A can do a work in 8 days and A and B together can do the work in 16/3 days,
    prove that B can do the work alone in 16 days. -/
theorem work_completion_time (a b : ℝ) (ha : a = 8) (hab : 1 / a + 1 / b = 3 / 16) :
  b = 16 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l3141_314162


namespace NUMINAMATH_CALUDE_angle_bisector_slope_l3141_314161

/-- The slope of the angle bisector of the acute angle formed at the origin
    by the lines y = x and y = 4x is -5/3 + √2. -/
theorem angle_bisector_slope : ℝ := by
  -- Define the slopes of the two lines
  let m₁ : ℝ := 1
  let m₂ : ℝ := 4

  -- Define the slope of the angle bisector
  let k : ℝ := (m₁ + m₂ + Real.sqrt (1 + m₁^2 + m₂^2)) / (1 - m₁ * m₂)

  -- Prove that k equals -5/3 + √2
  sorry

end NUMINAMATH_CALUDE_angle_bisector_slope_l3141_314161


namespace NUMINAMATH_CALUDE_integral_of_f_equals_seven_sixths_l3141_314137

-- Define the function f
def f (x : ℝ) (f'₁ : ℝ) : ℝ := f'₁ * x^2 + x + 1

-- State the theorem
theorem integral_of_f_equals_seven_sixths :
  ∃ (f'₁ : ℝ), (∫ x in (0:ℝ)..(1:ℝ), f x f'₁) = 7/6 := by
  sorry

end NUMINAMATH_CALUDE_integral_of_f_equals_seven_sixths_l3141_314137
