import Mathlib

namespace NUMINAMATH_CALUDE_factorial_difference_l2389_238984

theorem factorial_difference : Nat.factorial 9 - Nat.factorial 8 = 322560 := by
  sorry

end NUMINAMATH_CALUDE_factorial_difference_l2389_238984


namespace NUMINAMATH_CALUDE_all_methods_applicable_l2389_238983

structure Population where
  total : Nat
  farmers : Nat
  workers : Nat
  sample_size : Nat

inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified

def is_applicable (pop : Population) (method : SamplingMethod) : Prop :=
  match method with
  | SamplingMethod.SimpleRandom => pop.workers > 0
  | SamplingMethod.Systematic => pop.farmers > 0
  | SamplingMethod.Stratified => pop.farmers ≠ pop.workers

theorem all_methods_applicable (pop : Population) 
  (h1 : pop.total = 2004)
  (h2 : pop.farmers = 1600)
  (h3 : pop.workers = 303)
  (h4 : pop.sample_size = 40) :
  (∀ m : SamplingMethod, is_applicable pop m) :=
by sorry

end NUMINAMATH_CALUDE_all_methods_applicable_l2389_238983


namespace NUMINAMATH_CALUDE_smallest_k_no_real_roots_l2389_238920

theorem smallest_k_no_real_roots : 
  ∀ k : ℤ, (∀ x : ℝ, 2 * x * (k * x - 5) - x^2 + 12 ≠ 0) → k ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_no_real_roots_l2389_238920


namespace NUMINAMATH_CALUDE_prob_rain_weekend_l2389_238971

-- Define the probabilities
def prob_rain_sat : ℝ := 0.30
def prob_rain_sun : ℝ := 0.60
def prob_rain_sun_given_rain_sat : ℝ := 0.40

-- Define the theorem
theorem prob_rain_weekend : 
  let prob_no_rain_sat := 1 - prob_rain_sat
  let prob_no_rain_sun := 1 - prob_rain_sun
  let prob_no_rain_sun_given_rain_sat := 1 - prob_rain_sun_given_rain_sat
  let prob_no_rain_both := prob_no_rain_sat * prob_no_rain_sun
  let prob_rain_sat_no_rain_sun := prob_rain_sat * prob_no_rain_sun_given_rain_sat
  let prob_no_rain_all_scenarios := prob_no_rain_both + prob_rain_sat_no_rain_sun
  1 - prob_no_rain_all_scenarios = 0.54 :=
by
  sorry

#check prob_rain_weekend

end NUMINAMATH_CALUDE_prob_rain_weekend_l2389_238971


namespace NUMINAMATH_CALUDE_total_unique_photos_l2389_238956

/-- Represents the number of photographs taken by Octavia -/
def octavia_photos : ℕ := 36

/-- Represents the number of Octavia's photographs framed by Jack -/
def jack_framed_octavia : ℕ := 24

/-- Represents the number of photographs framed by Jack that were taken by other photographers -/
def jack_framed_others : ℕ := 12

/-- Theorem stating the total number of unique photographs either framed by Jack or taken by Octavia -/
theorem total_unique_photos : 
  (octavia_photos + (jack_framed_octavia + jack_framed_others) - jack_framed_octavia) = 48 := by
  sorry


end NUMINAMATH_CALUDE_total_unique_photos_l2389_238956


namespace NUMINAMATH_CALUDE_max_value_expression_l2389_238963

theorem max_value_expression (a b c d : Real) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) (hd : 0 ≤ d ∧ d ≤ 1) : 
  (∃ x y z w, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧ 0 ≤ z ∧ z ≤ 1 ∧ 0 ≤ w ∧ w ≤ 1 ∧ 
    x + y + z + w - x*y - y*z - z*w - w*x = 2) ∧ 
  (∀ a b c d, 0 ≤ a ∧ a ≤ 1 → 0 ≤ b ∧ b ≤ 1 → 0 ≤ c ∧ c ≤ 1 → 0 ≤ d ∧ d ≤ 1 → 
    a + b + c + d - a*b - b*c - c*d - d*a ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l2389_238963


namespace NUMINAMATH_CALUDE_hillarys_reading_assignment_l2389_238959

theorem hillarys_reading_assignment 
  (total_assignment : ℕ) 
  (friday_reading : ℕ) 
  (saturday_reading : ℕ) :
  total_assignment = 60 →
  friday_reading = 16 →
  saturday_reading = 28 →
  total_assignment - (friday_reading + saturday_reading) = 16 :=
by sorry

end NUMINAMATH_CALUDE_hillarys_reading_assignment_l2389_238959


namespace NUMINAMATH_CALUDE_power_of_seven_inverse_l2389_238900

theorem power_of_seven_inverse (x y : ℕ) : 
  (2^x : ℕ) = Nat.gcd 180 (2^Nat.succ x) →
  (3^y : ℕ) = Nat.gcd 180 (3^Nat.succ y) →
  (1/7 : ℚ)^(y - x) = 1 :=
by sorry

end NUMINAMATH_CALUDE_power_of_seven_inverse_l2389_238900


namespace NUMINAMATH_CALUDE_shorter_diagonal_length_l2389_238996

theorem shorter_diagonal_length (a b : ℝ × ℝ) :
  ‖a‖ = 2 →
  ‖b‖ = 4 →
  a • b = 4 →
  ‖a - b‖ = 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_shorter_diagonal_length_l2389_238996


namespace NUMINAMATH_CALUDE_parabola_y_axis_intersection_l2389_238924

/-- The equation of the parabola -/
def parabola (x y : ℝ) : Prop := y = x^2 + 2*x - 3

/-- Definition of a point on the y-axis -/
def on_y_axis (x y : ℝ) : Prop := x = 0

/-- The intersection point of the parabola with the y-axis -/
def intersection_point : ℝ × ℝ := (0, -3)

/-- Theorem stating that the intersection_point is on the parabola and the y-axis -/
theorem parabola_y_axis_intersection :
  let (x, y) := intersection_point
  parabola x y ∧ on_y_axis x y :=
by sorry

end NUMINAMATH_CALUDE_parabola_y_axis_intersection_l2389_238924


namespace NUMINAMATH_CALUDE_perpendicular_and_parallel_properties_l2389_238913

-- Define the necessary structures
structure EuclideanPlane where
  -- Add necessary axioms for Euclidean plane

structure Line where
  -- Add necessary properties for a line

structure Point where
  -- Add necessary properties for a point

-- Define the relationships
def isOn (p : Point) (l : Line) : Prop := sorry

def isPerpendicular (l1 l2 : Line) : Prop := sorry

def isParallel (l1 l2 : Line) : Prop := sorry

-- State the theorem
theorem perpendicular_and_parallel_properties 
  (plane : EuclideanPlane) (l : Line) : 
  (∀ (p : Point), isOn p l → ∃ (perps : Set Line), 
    (∀ (l' : Line), l' ∈ perps ↔ isPerpendicular l' l ∧ isOn p l') ∧ 
    Set.Infinite perps) ∧
  (∀ (p : Point), ¬isOn p l → 
    ∃! (l' : Line), isPerpendicular l' l ∧ isOn p l') ∧
  (∀ (p : Point), ¬isOn p l → 
    ∃! (l' : Line), isParallel l' l ∧ isOn p l') := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_and_parallel_properties_l2389_238913


namespace NUMINAMATH_CALUDE_point_m_locations_l2389_238911

/-- Given a line segment AC with point B on AC such that AB = 2 and BC = 1,
    prove that the only points M on the line AC that satisfy AM + MB = CM
    are at x = 1 and x = -1, where A is at x = 0 and C is at x = 3. -/
theorem point_m_locations (A B C M : ℝ) (h1 : 0 < B) (h2 : B < 3) (h3 : B = 2) :
  (M < 0 ∨ 0 ≤ M ∧ M ≤ 3) →
  (abs (M - 0) + abs (M - 2) = abs (M - 3)) ↔ (M = 1 ∨ M = -1) :=
by sorry

end NUMINAMATH_CALUDE_point_m_locations_l2389_238911


namespace NUMINAMATH_CALUDE_negation_of_exponential_inequality_l2389_238993

theorem negation_of_exponential_inequality :
  (¬ (∀ x : ℝ, Real.exp x ≥ 1)) ↔ (∃ x : ℝ, Real.exp x < 1) := by sorry

end NUMINAMATH_CALUDE_negation_of_exponential_inequality_l2389_238993


namespace NUMINAMATH_CALUDE_zero_not_in_N_star_l2389_238989

-- Define the set of natural numbers
def N : Set ℕ := {n : ℕ | n > 0}

-- Define the set of positive integers (N*)
def N_star : Set ℕ := N

-- Define the set of rational numbers
def Q : Set ℚ := {q : ℚ | ∃ (a b : ℤ), b ≠ 0 ∧ q = a / b}

-- Define the set of real numbers
def R : Set ℝ := Set.univ

-- Theorem statement
theorem zero_not_in_N_star : 0 ∉ N_star := by
  sorry

end NUMINAMATH_CALUDE_zero_not_in_N_star_l2389_238989


namespace NUMINAMATH_CALUDE_last_three_digits_of_7_pow_215_l2389_238986

theorem last_three_digits_of_7_pow_215 : 7^215 ≡ 447 [ZMOD 1000] := by
  sorry

end NUMINAMATH_CALUDE_last_three_digits_of_7_pow_215_l2389_238986


namespace NUMINAMATH_CALUDE_tangent_when_zero_discriminant_l2389_238927

/-- Represents a quadratic function of the form ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the discriminant of a quadratic function -/
def discriminant (f : QuadraticFunction) : ℝ :=
  f.b^2 - 4 * f.a * f.c

/-- Determines if a quadratic function's graph is tangent to the x-axis -/
def is_tangent_to_x_axis (f : QuadraticFunction) : Prop :=
  ∃ x : ℝ, f.a * x^2 + f.b * x + f.c = 0 ∧
    ∀ y : ℝ, y ≠ x → f.a * y^2 + f.b * y + f.c > 0

/-- The main theorem: if the discriminant is zero, the graph is tangent to the x-axis -/
theorem tangent_when_zero_discriminant (k : ℝ) :
  let f : QuadraticFunction := ⟨3, 9, k⟩
  discriminant f = 0 → is_tangent_to_x_axis f :=
by sorry

end NUMINAMATH_CALUDE_tangent_when_zero_discriminant_l2389_238927


namespace NUMINAMATH_CALUDE_people_visited_neither_l2389_238910

theorem people_visited_neither (total : ℕ) (iceland : ℕ) (norway : ℕ) (both : ℕ) :
  total = 100 →
  iceland = 55 →
  norway = 43 →
  both = 61 →
  total - (iceland + norway - both) = 63 :=
by sorry

end NUMINAMATH_CALUDE_people_visited_neither_l2389_238910


namespace NUMINAMATH_CALUDE_jim_gave_away_195_cards_l2389_238953

/-- The number of cards Jim gives away -/
def cards_given_away (initial_cards : ℕ) (cards_per_set : ℕ) (sets_to_brother : ℕ) (sets_to_sister : ℕ) (sets_to_friend : ℕ) : ℕ :=
  (sets_to_brother + sets_to_sister + sets_to_friend) * cards_per_set

/-- Proof that Jim gave away 195 cards -/
theorem jim_gave_away_195_cards :
  cards_given_away 365 13 8 5 2 = 195 := by
  sorry

end NUMINAMATH_CALUDE_jim_gave_away_195_cards_l2389_238953


namespace NUMINAMATH_CALUDE_dart_score_proof_l2389_238958

def bullseye_points : ℕ := 50
def missed_points : ℕ := 0
def third_dart_points : ℕ := bullseye_points / 2

def total_score : ℕ := bullseye_points + missed_points + third_dart_points

theorem dart_score_proof : total_score = 75 := by
  sorry

end NUMINAMATH_CALUDE_dart_score_proof_l2389_238958


namespace NUMINAMATH_CALUDE_bear_discount_calculation_l2389_238955

/-- The discount per bear after the first bear, given the price of the first bear,
    the total number of bears, and the total amount paid. -/
def discount_per_bear (first_bear_price : ℚ) (total_bears : ℕ) (total_paid : ℚ) : ℚ :=
  let full_price := first_bear_price * total_bears
  let discount := full_price - total_paid
  discount / (total_bears - 1)

/-- Theorem stating that under the given conditions, the discount per bear after the first bear is $0.50 -/
theorem bear_discount_calculation :
  let first_bear_price : ℚ := 4
  let total_bears : ℕ := 101
  let total_paid : ℚ := 354
  discount_per_bear first_bear_price total_bears total_paid = 1/2 := by
sorry


end NUMINAMATH_CALUDE_bear_discount_calculation_l2389_238955


namespace NUMINAMATH_CALUDE_abcd_sum_absolute_l2389_238945

theorem abcd_sum_absolute (a b c d : ℤ) 
  (h1 : a * b * c * d = 25)
  (h2 : a > b ∧ b > c ∧ c > d) : 
  |a + b| + |c + d| = 12 := by
sorry

end NUMINAMATH_CALUDE_abcd_sum_absolute_l2389_238945


namespace NUMINAMATH_CALUDE_stool_height_is_30_l2389_238912

/-- The height of the stool Alice needs to reach the light bulb -/
def stool_height : ℝ :=
  let ceiling_height : ℝ := 250  -- in cm
  let light_bulb_below_ceiling : ℝ := 15  -- in cm
  let alice_height : ℝ := 155  -- in cm
  let alice_reach : ℝ := 50  -- in cm
  let light_bulb_height : ℝ := ceiling_height - light_bulb_below_ceiling
  let alice_total_reach : ℝ := alice_height + alice_reach
  light_bulb_height - alice_total_reach

theorem stool_height_is_30 : stool_height = 30 := by
  sorry

end NUMINAMATH_CALUDE_stool_height_is_30_l2389_238912


namespace NUMINAMATH_CALUDE_probability_both_in_photo_l2389_238968

/-- Represents a runner on a circular track -/
structure Runner where
  name : String
  lapTime : ℝ
  direction : Bool  -- true for counterclockwise, false for clockwise

/-- Represents the photography setup -/
structure PhotoSetup where
  trackCoverage : ℝ  -- fraction of track covered by the photo
  minTime : ℝ        -- minimum time after start for taking the photo (in seconds)
  maxTime : ℝ        -- maximum time after start for taking the photo (in seconds)

/-- Calculate the probability of both runners being in the photo -/
def probabilityBothInPhoto (ann : Runner) (ben : Runner) (setup : PhotoSetup) : ℝ :=
  sorry

/-- Theorem statement for the probability problem -/
theorem probability_both_in_photo 
  (ann : Runner) 
  (ben : Runner) 
  (setup : PhotoSetup) 
  (h1 : ann.name = "Ann" ∧ ann.lapTime = 75 ∧ ann.direction = true)
  (h2 : ben.name = "Ben" ∧ ben.lapTime = 60 ∧ ben.direction = false)
  (h3 : setup.trackCoverage = 1/6)
  (h4 : setup.minTime = 12 * 60)
  (h5 : setup.maxTime = 15 * 60) :
  probabilityBothInPhoto ann ben setup = 1/6 :=
sorry

end NUMINAMATH_CALUDE_probability_both_in_photo_l2389_238968


namespace NUMINAMATH_CALUDE_solution_set_inequality_l2389_238925

theorem solution_set_inequality (x : ℝ) :
  (((2 * x - 1) / (x + 2)) > 1) ↔ (x < -2 ∨ x > 3) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l2389_238925


namespace NUMINAMATH_CALUDE_cubic_root_sum_l2389_238969

theorem cubic_root_sum (p q r : ℝ) : 
  (p^3 - 3*p + 1 = 0) → 
  (q^3 - 3*q + 1 = 0) → 
  (r^3 - 3*r + 1 = 0) → 
  p*(q - r)^2 + q*(r - p)^2 + r*(p - q)^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l2389_238969


namespace NUMINAMATH_CALUDE_product_trailing_zeros_l2389_238909

/-- The number of trailing zeros in a natural number -/
def trailingZeros (n : ℕ) : ℕ := sorry

/-- The product of 45 and 800 -/
def product : ℕ := 45 * 800

theorem product_trailing_zeros :
  trailingZeros product = 3 := by sorry

end NUMINAMATH_CALUDE_product_trailing_zeros_l2389_238909


namespace NUMINAMATH_CALUDE_sum_of_tangent_slopes_l2389_238980

/-- The parabola P with equation y = x^2 + 5x -/
def P (x y : ℝ) : Prop := y = x^2 + 5*x

/-- The point Q -/
def Q : ℝ × ℝ := (10, -6)

/-- The equation whose roots are the slopes of lines through Q tangent to P -/
def tangent_slope_equation (m : ℝ) : Prop := m^2 - 50*m + 1 = 0

/-- The sum of the roots of the tangent slope equation is 50 -/
theorem sum_of_tangent_slopes : 
  ∃ r s : ℝ, tangent_slope_equation r ∧ tangent_slope_equation s ∧ r + s = 50 :=
sorry

end NUMINAMATH_CALUDE_sum_of_tangent_slopes_l2389_238980


namespace NUMINAMATH_CALUDE_equation_equivalence_l2389_238994

theorem equation_equivalence (x y : ℝ) 
  (hx1 : x ≠ 0) (hx2 : x ≠ 4) (hy1 : y ≠ 0) (hy2 : y ≠ 6) :
  (2 / x + 3 / y = 1 / 2) ↔ (4 * y / (y - 6) = x) :=
by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l2389_238994


namespace NUMINAMATH_CALUDE_onion_weight_problem_l2389_238928

theorem onion_weight_problem (total_weight : Real) (total_count : Nat) (removed_count : Nat) (removed_avg : Real) (remaining_count : Nat) :
  total_weight = 7.68 →
  total_count = 40 →
  removed_count = 5 →
  removed_avg = 0.206 →
  remaining_count = total_count - removed_count →
  let remaining_weight := total_weight - (removed_count * removed_avg)
  let remaining_avg := remaining_weight / remaining_count
  remaining_avg = 0.190 := by
sorry

end NUMINAMATH_CALUDE_onion_weight_problem_l2389_238928


namespace NUMINAMATH_CALUDE_min_value_of_expression_l2389_238947

theorem min_value_of_expression (x y : ℝ) 
  (h1 : x^2 + y^2 = 2) 
  (h2 : |x| ≠ |y|) : 
  ∃ (m : ℝ), m = 1 ∧ ∀ (a b : ℝ), a^2 + b^2 = 2 → |a| ≠ |b| → 
    (1 / (a + b)^2 + 1 / (a - b)^2) ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l2389_238947


namespace NUMINAMATH_CALUDE_tangent_line_slope_l2389_238923

/-- Given a differentiable function f, prove that its derivative at x = 1 is 2,
    given that the tangent line equation at (1, f(1)) is 2x - y + 2 = 0. -/
theorem tangent_line_slope (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  (∀ x y, x = 1 ∧ y = f 1 → 2 * x - y + 2 = 0) →
  deriv f 1 = 2 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_slope_l2389_238923


namespace NUMINAMATH_CALUDE_remaining_time_is_three_and_half_l2389_238936

/-- The time taken for Cameron and Sandra to complete the remaining task -/
def remaining_time (cameron_rate : ℚ) (combined_rate : ℚ) (cameron_solo_days : ℚ) : ℚ :=
  (1 - cameron_rate * cameron_solo_days) / combined_rate

/-- Theorem stating the remaining time is 3.5 days -/
theorem remaining_time_is_three_and_half :
  remaining_time (1/18) (1/7) 9 = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_remaining_time_is_three_and_half_l2389_238936


namespace NUMINAMATH_CALUDE_shirt_price_calculation_l2389_238917

theorem shirt_price_calculation (P : ℝ) : 
  (P * (1 - 0.33333) * (1 - 0.25) * (1 - 0.2) = 15) → P = 37.50 := by
  sorry

end NUMINAMATH_CALUDE_shirt_price_calculation_l2389_238917


namespace NUMINAMATH_CALUDE_angle_in_first_quadrant_l2389_238914

theorem angle_in_first_quadrant (θ : Real) (h : θ = -5) : 
  ∃ n : ℤ, θ + 2 * π * n ∈ Set.Ioo 0 (π / 2) :=
by sorry

end NUMINAMATH_CALUDE_angle_in_first_quadrant_l2389_238914


namespace NUMINAMATH_CALUDE_total_weight_is_350_l2389_238907

/-- Represents the weight of a single box in kilograms -/
def box_weight : ℕ := 25

/-- Represents the number of columns with 3 boxes -/
def columns_with_3 : ℕ := 1

/-- Represents the number of columns with 2 boxes -/
def columns_with_2 : ℕ := 4

/-- Represents the number of columns with 1 box -/
def columns_with_1 : ℕ := 3

/-- Calculates the total number of boxes in the stack -/
def total_boxes : ℕ := columns_with_3 * 3 + columns_with_2 * 2 + columns_with_1 * 1

/-- Calculates the total weight of all boxes in kilograms -/
def total_weight : ℕ := total_boxes * box_weight

/-- Theorem stating that the total weight of all boxes is 350 kg -/
theorem total_weight_is_350 : total_weight = 350 := by sorry

end NUMINAMATH_CALUDE_total_weight_is_350_l2389_238907


namespace NUMINAMATH_CALUDE_steves_take_home_pay_l2389_238908

/-- Calculates the take-home pay given salary and deduction rates -/
def takeHomePay (salary : ℝ) (taxRate : ℝ) (healthcareRate : ℝ) (unionDues : ℝ) : ℝ :=
  salary - (salary * taxRate) - (salary * healthcareRate) - unionDues

/-- Theorem: Steve's take-home pay is $27,200 -/
theorem steves_take_home_pay :
  takeHomePay 40000 0.20 0.10 800 = 27200 := by
  sorry

#eval takeHomePay 40000 0.20 0.10 800

end NUMINAMATH_CALUDE_steves_take_home_pay_l2389_238908


namespace NUMINAMATH_CALUDE_cubic_inequality_l2389_238929

theorem cubic_inequality (a b c : ℕ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) :
  (a^3 + b^3 + c^3) / 3 ≥ a * b * c + a + b + c ∧
  ((a^3 + b^3 + c^3) / 3 = a * b * c + a + b + c ↔ 
   (b = a + 1 ∧ c = b + 1) ∨ (a = b + 1 ∧ c = a + 1) ∨ (a = c + 1 ∧ b = a + 1)) :=
sorry

end NUMINAMATH_CALUDE_cubic_inequality_l2389_238929


namespace NUMINAMATH_CALUDE_share_difference_l2389_238977

/-- Given a distribution ratio and Vasim's share, calculate the difference between Ranjith's and Faruk's shares -/
theorem share_difference (faruk_ratio vasim_ratio ranjith_ratio vasim_share : ℕ) : 
  faruk_ratio = 3 → 
  vasim_ratio = 3 → 
  ranjith_ratio = 7 → 
  vasim_share = 1500 → 
  (ranjith_ratio * vasim_share / vasim_ratio) - (faruk_ratio * vasim_share / vasim_ratio) = 2000 := by
  sorry

end NUMINAMATH_CALUDE_share_difference_l2389_238977


namespace NUMINAMATH_CALUDE_eight_points_chords_l2389_238997

/-- The number of chords that can be drawn by connecting two points out of n points on the circumference of a circle -/
def num_chords (n : ℕ) : ℕ := n.choose 2

/-- Theorem: The number of chords that can be drawn by connecting two points out of eight points on the circumference of a circle is equal to 28 -/
theorem eight_points_chords : num_chords 8 = 28 := by
  sorry

end NUMINAMATH_CALUDE_eight_points_chords_l2389_238997


namespace NUMINAMATH_CALUDE_keith_attended_four_games_l2389_238931

/-- The number of football games Keith attended -/
def games_attended (total_games missed_games : ℕ) : ℕ :=
  total_games - missed_games

/-- Theorem stating that Keith attended 4 football games -/
theorem keith_attended_four_games :
  ∃ (total_games missed_games : ℕ),
    total_games = 8 ∧
    missed_games = 4 ∧
    games_attended total_games missed_games = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_keith_attended_four_games_l2389_238931


namespace NUMINAMATH_CALUDE_matrix_subtraction_l2389_238978

theorem matrix_subtraction : 
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![4, -3; 6, 5]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![1, -8; 3, 7]
  let C : Matrix (Fin 2) (Fin 2) ℤ := !![3, 5; 3, -2]
  A - B = C := by sorry

end NUMINAMATH_CALUDE_matrix_subtraction_l2389_238978


namespace NUMINAMATH_CALUDE_smallest_positive_angle_2002_l2389_238972

theorem smallest_positive_angle_2002 : 
  ∃ (θ : ℝ), θ > 0 ∧ θ < 360 ∧ ∀ (k : ℤ), -2002 = θ + 360 * k → θ = 158 := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_angle_2002_l2389_238972


namespace NUMINAMATH_CALUDE_convention_handshakes_l2389_238902

/-- The number of handshakes in a convention with multiple companies --/
def number_of_handshakes (num_companies : ℕ) (representatives_per_company : ℕ) : ℕ :=
  let total_people := num_companies * representatives_per_company
  let handshakes_per_person := total_people - representatives_per_company
  (total_people * handshakes_per_person) / 2

/-- Theorem stating that the number of handshakes in the specific convention scenario is 160 --/
theorem convention_handshakes :
  number_of_handshakes 5 4 = 160 := by
  sorry

end NUMINAMATH_CALUDE_convention_handshakes_l2389_238902


namespace NUMINAMATH_CALUDE_repair_center_solution_l2389_238998

/-- Represents a bonus distribution plan -/
structure BonusPlan where
  techBonus : ℕ
  assistBonus : ℕ

/-- Represents the repair center staff and bonus distribution -/
structure RepairCenter where
  techCount : ℕ
  assistCount : ℕ
  totalBonus : ℕ
  bonusPlans : List BonusPlan

/-- The conditions of the repair center problem -/
def repairCenterConditions (rc : RepairCenter) : Prop :=
  rc.techCount + rc.assistCount = 15 ∧
  rc.techCount = 2 * rc.assistCount ∧
  rc.totalBonus = 20000 ∧
  ∀ plan ∈ rc.bonusPlans,
    plan.techBonus ≥ plan.assistBonus ∧
    plan.assistBonus ≥ 800 ∧
    plan.techBonus % 100 = 0 ∧
    plan.assistBonus % 100 = 0 ∧
    rc.techCount * plan.techBonus + rc.assistCount * plan.assistBonus = rc.totalBonus

/-- The theorem stating the solution to the repair center problem -/
theorem repair_center_solution :
  ∃ (rc : RepairCenter),
    repairCenterConditions rc ∧
    rc.techCount = 10 ∧
    rc.assistCount = 5 ∧
    rc.bonusPlans = [
      { techBonus := 1600, assistBonus := 800 },
      { techBonus := 1500, assistBonus := 1000 },
      { techBonus := 1400, assistBonus := 1200 }
    ] :=
  sorry

end NUMINAMATH_CALUDE_repair_center_solution_l2389_238998


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l2389_238961

theorem absolute_value_equation_solution :
  ∀ y : ℝ, (|y - 4| + 3 * y = 11) ↔ (y = 15/4 ∨ y = 7/2) := by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l2389_238961


namespace NUMINAMATH_CALUDE_smallest_cookie_count_l2389_238951

theorem smallest_cookie_count : ∃ (n : ℕ), n > 0 ∧ (∀ m : ℕ, m > 0 → 4*m - 4 = (m^2)/2 → m ≥ n) ∧ 4*n - 4 = (n^2)/2 ∧ n^2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_smallest_cookie_count_l2389_238951


namespace NUMINAMATH_CALUDE_equation_solution_l2389_238965

theorem equation_solution :
  let x : ℚ := 1/2
  2 * x - 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2389_238965


namespace NUMINAMATH_CALUDE_midpoint_distance_after_move_l2389_238946

/-- Given two points A(a,b) and B(c,d) on a Cartesian plane with midpoint M(m,n),
    prove that after moving A 3 units right and 5 units up, and B 5 units left and 3 units down,
    the distance between M and the new midpoint M' is √2. -/
theorem midpoint_distance_after_move (a b c d m n : ℝ) :
  m = (a + c) / 2 →
  n = (b + d) / 2 →
  let m' := (a + 3 + c - 5) / 2
  let n' := (b + 5 + d - 3) / 2
  Real.sqrt ((m' - m)^2 + (n' - n)^2) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_distance_after_move_l2389_238946


namespace NUMINAMATH_CALUDE_system_inequality_solution_range_l2389_238952

theorem system_inequality_solution_range (x y m : ℝ) : 
  x - 2*y = 1 → 
  2*x + y = 4*m → 
  x + 3*y < 6 → 
  m < 7/4 := by
sorry

end NUMINAMATH_CALUDE_system_inequality_solution_range_l2389_238952


namespace NUMINAMATH_CALUDE_like_terms_imply_a_minus_b_eq_two_l2389_238940

/-- Two algebraic expressions are like terms if they have the same variables raised to the same powers. -/
def are_like_terms (expr1 expr2 : ℝ → ℝ → ℝ) : Prop :=
  ∃ (c₁ c₂ : ℝ) (m n : ℕ), 
    (∀ x y, expr1 x y = c₁ * x^m * y^n) ∧ 
    (∀ x y, expr2 x y = c₂ * x^m * y^n)

/-- Given that -2.5x^(a+b)y^(a-1) and 3x^2y are like terms, prove that a - b = 2 -/
theorem like_terms_imply_a_minus_b_eq_two 
  (a b : ℝ) 
  (h : are_like_terms (λ x y => -2.5 * x^(a+b) * y^(a-1)) (λ x y => 3 * x^2 * y)) : 
  a - b = 2 := by
sorry


end NUMINAMATH_CALUDE_like_terms_imply_a_minus_b_eq_two_l2389_238940


namespace NUMINAMATH_CALUDE_max_value_complex_l2389_238960

theorem max_value_complex (z : ℂ) (h : Complex.abs z = 2) :
  Complex.abs ((z - 2)^3 * (z + 1)) ≤ 8 * Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_max_value_complex_l2389_238960


namespace NUMINAMATH_CALUDE_product_and_reciprocal_relation_sum_l2389_238938

theorem product_and_reciprocal_relation_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  a * b = 16 ∧ 1 / a = 3 / b → a + b = 16 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_product_and_reciprocal_relation_sum_l2389_238938


namespace NUMINAMATH_CALUDE_general_term_equals_closed_form_l2389_238944

/-- The general term of the sequence -/
def a (n : ℕ) : ℚ := (2 * n - 1 : ℚ) + n / (2 * n + 1 : ℚ)

/-- The proposed closed form of the general term -/
def a_closed (n : ℕ) : ℚ := (4 * n^2 + n - 1 : ℚ) / (2 * n + 1 : ℚ)

/-- Theorem stating that the general term equals the closed form -/
theorem general_term_equals_closed_form (n : ℕ) : a n = a_closed n := by
  sorry

end NUMINAMATH_CALUDE_general_term_equals_closed_form_l2389_238944


namespace NUMINAMATH_CALUDE_range_of_x_for_positive_f_l2389_238990

-- Define the function f
def f (a x : ℝ) : ℝ := x^2 + (a-4)*x + 4-2*a

-- State the theorem
theorem range_of_x_for_positive_f :
  ∀ a ∈ Set.Icc (-1 : ℝ) 1,
    (∀ x, f a x > 0) ↔ (∀ x, x < 1 ∨ x > 3) := by sorry

end NUMINAMATH_CALUDE_range_of_x_for_positive_f_l2389_238990


namespace NUMINAMATH_CALUDE_solution_set_equation_l2389_238906

theorem solution_set_equation (x : ℝ) : 
  (1 / (x^2 + 12*x - 8) + 1 / (x^2 + 3*x - 8) + 1 / (x^2 - 14*x - 8) = 0) ↔ 
  (x = 2 ∨ x = -4 ∨ x = 1 ∨ x = -8) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_equation_l2389_238906


namespace NUMINAMATH_CALUDE_tulip_arrangement_l2389_238941

/-- The number of red tulips needed for the smile -/
def smile_tulips : ℕ := 18

/-- The number of yellow tulips for the background is 9 times the number of red tulips in the smile -/
def background_tulips : ℕ := 9 * smile_tulips

/-- The total number of tulips needed -/
def total_tulips : ℕ := 196

/-- The number of red tulips needed for each eye -/
def eye_tulips : ℕ := 8

theorem tulip_arrangement : 
  2 * eye_tulips + smile_tulips + background_tulips = total_tulips :=
sorry

end NUMINAMATH_CALUDE_tulip_arrangement_l2389_238941


namespace NUMINAMATH_CALUDE_remainder_problem_l2389_238942

theorem remainder_problem (n : ℕ) 
  (h1 : n^2 % 7 = 2) 
  (h2 : n^3 % 7 = 6) : 
  n % 7 = 3 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l2389_238942


namespace NUMINAMATH_CALUDE_independence_test_problems_l2389_238995

/-- A real-world problem that may or may not be solvable by independence tests. -/
inductive Problem
| DrugCureRate
| DrugRelation
| SmokingLungDisease
| SmokingGenderRelation
| InternetCrimeRate

/-- Determines if a problem involves examining the relationship between two categorical variables. -/
def involves_categorical_relationship (p : Problem) : Prop :=
  match p with
  | Problem.DrugRelation => True
  | Problem.SmokingGenderRelation => True
  | Problem.InternetCrimeRate => True
  | _ => False

/-- The definition of an independence test. -/
def is_independence_test (test : Problem → Prop) : Prop :=
  ∀ p, test p ↔ involves_categorical_relationship p

/-- The theorem stating which problems can be solved using independence tests. -/
theorem independence_test_problems (test : Problem → Prop) 
  (h : is_independence_test test) : 
  (test Problem.DrugRelation ∧ 
   test Problem.SmokingGenderRelation ∧ 
   test Problem.InternetCrimeRate) ∧
  (¬ test Problem.DrugCureRate ∧ 
   ¬ test Problem.SmokingLungDisease) :=
by sorry

end NUMINAMATH_CALUDE_independence_test_problems_l2389_238995


namespace NUMINAMATH_CALUDE_eighteenth_term_is_three_l2389_238915

/-- An equal sum sequence with public sum 5 and a₁ = 2 -/
def EqualSumSequence (a : ℕ → ℕ) : Prop :=
  (∀ n, a n + a (n + 1) = 5) ∧ a 1 = 2

/-- The 18th term of the equal sum sequence is 3 -/
theorem eighteenth_term_is_three (a : ℕ → ℕ) (h : EqualSumSequence a) : a 18 = 3 := by
  sorry

end NUMINAMATH_CALUDE_eighteenth_term_is_three_l2389_238915


namespace NUMINAMATH_CALUDE_woman_birth_year_l2389_238979

theorem woman_birth_year (x : ℕ) (h1 : x > 0) (h2 : x^2 < 1950) 
  (h3 : x^2 + x ≤ 2000) : x^2 = 1936 := by
  sorry

end NUMINAMATH_CALUDE_woman_birth_year_l2389_238979


namespace NUMINAMATH_CALUDE_book_distribution_ways_l2389_238974

theorem book_distribution_ways :
  let n : ℕ := 8  -- Total number of books
  let total_distributions : ℕ := 2^n  -- Total ways to distribute books
  let invalid_distributions : ℕ := 2  -- All books in library or all with students
  (total_distributions - invalid_distributions) = 254 :=
by sorry

end NUMINAMATH_CALUDE_book_distribution_ways_l2389_238974


namespace NUMINAMATH_CALUDE_sufficient_condition_implies_m_range_l2389_238954

theorem sufficient_condition_implies_m_range (m : ℝ) : 
  (∀ x : ℝ, (x - 1) / x ≤ 0 → 4^x + 2^x - m ≤ 0) → m ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_implies_m_range_l2389_238954


namespace NUMINAMATH_CALUDE_polynomial_expansion_theorem_l2389_238930

/-- Given (2x-1)^5 = ax^5 + bx^4 + cx^3 + dx^2 + ex + f, prove the following statements -/
theorem polynomial_expansion_theorem (a b c d e f : ℝ) :
  (∀ x, (2*x - 1)^5 = a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f) →
  (a + b + c + d + e + f = 1) ∧
  (b + c + d + e = -30) ∧
  (a + c + e = 122) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_theorem_l2389_238930


namespace NUMINAMATH_CALUDE_quadratic_one_solution_sum_l2389_238926

theorem quadratic_one_solution_sum (a : ℝ) : 
  (∃ (a₁ a₂ : ℝ), 
    (∀ x : ℝ, 3 * x^2 + a₁ * x + 6 * x + 7 = 0 ↔ x = -((a₁ + 6) / 6)) ∧
    (∀ x : ℝ, 3 * x^2 + a₂ * x + 6 * x + 7 = 0 ↔ x = -((a₂ + 6) / 6)) ∧
    a₁ ≠ a₂ ∧ 
    (∀ a' : ℝ, a' ≠ a₁ ∧ a' ≠ a₂ → 
      ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 3 * x₁^2 + a' * x₁ + 6 * x₁ + 7 = 0 ∧ 
                     3 * x₂^2 + a' * x₂ + 6 * x₂ + 7 = 0)) →
  a₁ + a₂ = -12 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_sum_l2389_238926


namespace NUMINAMATH_CALUDE_equation_roots_iff_q_condition_l2389_238988

/-- The equation x^4 + qx^3 + 2x^2 + qx + 4 = 0 has at least two distinct negative real roots
    if and only if q ≤ 3/√2 -/
theorem equation_roots_iff_q_condition (q : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ < 0 ∧ x₁ ≠ x₂ ∧
    x₁^4 + q*x₁^3 + 2*x₁^2 + q*x₁ + 4 = 0 ∧
    x₂^4 + q*x₂^3 + 2*x₂^2 + q*x₂ + 4 = 0) ↔
  q ≤ 3 / Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_equation_roots_iff_q_condition_l2389_238988


namespace NUMINAMATH_CALUDE_relationship_abc_l2389_238975

theorem relationship_abc : 
  let a : ℝ := (0.2 : ℝ) ^ (1.5 : ℝ)
  let b : ℝ := (2 : ℝ) ^ (0.1 : ℝ)
  let c : ℝ := (0.2 : ℝ) ^ (1.3 : ℝ)
  a < c ∧ c < b := by
sorry

end NUMINAMATH_CALUDE_relationship_abc_l2389_238975


namespace NUMINAMATH_CALUDE_rotated_angle_measure_l2389_238933

/-- 
Given an initial angle of 40 degrees that is rotated 480 degrees clockwise,
the resulting acute angle measures 80 degrees.
-/
theorem rotated_angle_measure (initial_angle rotation : ℝ) : 
  initial_angle = 40 →
  rotation = 480 →
  (rotation % 360 - initial_angle) % 180 = 80 :=
by sorry

end NUMINAMATH_CALUDE_rotated_angle_measure_l2389_238933


namespace NUMINAMATH_CALUDE_count_special_numbers_is_126_l2389_238916

/-- A function that counts 4-digit numbers starting with 1 and having exactly two identical digits, excluding 1 and 5 as the identical digits -/
def count_special_numbers : ℕ :=
  let digits := {2, 3, 4, 6, 7, 8, 9}
  let patterns := 3  -- representing 1xxy, 1xyx, 1yxx
  let choices_for_x := Finset.card digits
  let choices_for_y := 9 - 3  -- total digits minus 1, 5, and x
  patterns * choices_for_x * choices_for_y

/-- The count of special numbers is 126 -/
theorem count_special_numbers_is_126 : count_special_numbers = 126 := by
  sorry

#eval count_special_numbers  -- This line is optional, for verification purposes

end NUMINAMATH_CALUDE_count_special_numbers_is_126_l2389_238916


namespace NUMINAMATH_CALUDE_equation_solution_l2389_238948

theorem equation_solution (x : ℚ) : 
  5 * x - 6 = 15 * x + 21 → 3 * (x + 5)^2 = 2523 / 100 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2389_238948


namespace NUMINAMATH_CALUDE_total_material_bought_l2389_238992

/-- The total amount of material bought by a construction company -/
theorem total_material_bought (gravel sand : ℝ) (h1 : gravel = 5.91) (h2 : sand = 8.11) :
  gravel + sand = 14.02 := by
  sorry

end NUMINAMATH_CALUDE_total_material_bought_l2389_238992


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2389_238999

theorem sufficient_not_necessary_condition (a b : ℝ) : 
  (a = Real.sqrt 3 ∧ b = 1 → Complex.abs ((1 + Complex.I * b) / (a + Complex.I)) = Real.sqrt 2 / 2) ∧
  (∃ (x y : ℝ), (x ≠ Real.sqrt 3 ∨ y ≠ 1) ∧ Complex.abs ((1 + Complex.I * y) / (x + Complex.I)) = Real.sqrt 2 / 2) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2389_238999


namespace NUMINAMATH_CALUDE_fly_distance_from_floor_l2389_238991

theorem fly_distance_from_floor (x y z h : ℝ) :
  x = 2 →
  y = 5 →
  h - z = 7 →
  x^2 + y^2 + z^2 = 11^2 →
  h = Real.sqrt 92 + 7 := by
sorry

end NUMINAMATH_CALUDE_fly_distance_from_floor_l2389_238991


namespace NUMINAMATH_CALUDE_not_all_face_sums_distinct_not_all_face_sums_distinct_l2389_238918

-- Define a cube type
structure Cube where
  vertices : Fin 8 → ℤ
  vertex_values : ∀ v, vertices v = 0 ∨ vertices v = 1

-- Define a function to get the sum of a face
def face_sum (c : Cube) (face : Fin 6) : ℤ :=
  sorry

-- Theorem statement
theorem not_all_face_sums_distinct (c : Cube) :
  ¬ (∀ f₁ f₂ : Fin 6, f₁ ≠ f₂ → face_sum c f₁ ≠ face_sum c f₂) :=
sorry

-- For part b, we can define a similar structure and theorem
structure Cube' where
  vertices : Fin 8 → ℤ
  vertex_values : ∀ v, vertices v = 1 ∨ vertices v = -1

def face_sum' (c : Cube') (face : Fin 6) : ℤ :=
  sorry

theorem not_all_face_sums_distinct' (c : Cube') :
  ¬ (∀ f₁ f₂ : Fin 6, f₁ ≠ f₂ → face_sum' c f₁ ≠ face_sum' c f₂) :=
sorry

end NUMINAMATH_CALUDE_not_all_face_sums_distinct_not_all_face_sums_distinct_l2389_238918


namespace NUMINAMATH_CALUDE_max_product_of_digits_divisible_by_25_l2389_238985

theorem max_product_of_digits_divisible_by_25 (a b : Nat) : 
  a ≤ 9 →
  b ≤ 9 →
  (10 * a + b) % 25 = 0 →
  b * a ≤ 35 := by
  sorry

end NUMINAMATH_CALUDE_max_product_of_digits_divisible_by_25_l2389_238985


namespace NUMINAMATH_CALUDE_tutor_schedule_lcm_l2389_238932

theorem tutor_schedule_lcm : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 9 (Nat.lcm 8 10))) = 360 := by
  sorry

end NUMINAMATH_CALUDE_tutor_schedule_lcm_l2389_238932


namespace NUMINAMATH_CALUDE_janous_inequality_l2389_238962

theorem janous_inequality (α x y z : ℝ) (hα : α > 0) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : x * y + y * z + z * x = α) :
  (1 + α / x^2) * (1 + α / y^2) * (1 + α / z^2) ≥ 16 * (x / z + z / x + 2) := by
  sorry

end NUMINAMATH_CALUDE_janous_inequality_l2389_238962


namespace NUMINAMATH_CALUDE_beaker_volume_difference_l2389_238937

theorem beaker_volume_difference (total_volume : ℝ) (beaker_one_volume : ℝ) 
  (h1 : total_volume = 9.28)
  (h2 : beaker_one_volume = 2.95) : 
  abs (beaker_one_volume - (total_volume - beaker_one_volume)) = 3.38 := by
  sorry

end NUMINAMATH_CALUDE_beaker_volume_difference_l2389_238937


namespace NUMINAMATH_CALUDE_lemonade_percentage_in_second_solution_l2389_238967

/-- Represents a solution mixture --/
structure Solution :=
  (lemonade : ℝ)
  (carbonated_water : ℝ)

/-- Represents the mixture of two solutions --/
structure Mixture :=
  (solution1 : Solution)
  (solution2 : Solution)
  (proportion1 : ℝ)
  (proportion2 : ℝ)
  (total_carbonated_water : ℝ)

/-- The theorem to be proved --/
theorem lemonade_percentage_in_second_solution 
  (mix : Mixture) 
  (h1 : mix.solution1.lemonade = 0.2)
  (h2 : mix.solution1.carbonated_water = 0.8)
  (h3 : mix.solution2.lemonade + mix.solution2.carbonated_water = 1)
  (h4 : mix.proportion1 = 0.4)
  (h5 : mix.proportion2 = 0.6)
  (h6 : mix.total_carbonated_water = 0.65) :
  mix.solution2.lemonade = 0.9945 :=
sorry

end NUMINAMATH_CALUDE_lemonade_percentage_in_second_solution_l2389_238967


namespace NUMINAMATH_CALUDE_inequality_proof_l2389_238973

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (2*a+b)*(2*a-b)*(a-c) + (2*b+c)*(2*b-c)*(b-a) + (2*c+a)*(2*c-a)*(c-b) ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2389_238973


namespace NUMINAMATH_CALUDE_max_value_sine_cosine_l2389_238950

/-- Given a function f(x) = a*sin(x) + 3*cos(x) where its maximum value is 5, 
    prove that a = ±4 -/
theorem max_value_sine_cosine (a : ℝ) :
  (∀ x, a * Real.sin x + 3 * Real.cos x ≤ 5) ∧ 
  (∃ x, a * Real.sin x + 3 * Real.cos x = 5) →
  a = 4 ∨ a = -4 := by
sorry

end NUMINAMATH_CALUDE_max_value_sine_cosine_l2389_238950


namespace NUMINAMATH_CALUDE_ball_distribution_equality_l2389_238970

theorem ball_distribution_equality (k : ℤ) : ∃ (n : ℕ), (19 + 6 * n) % 95 = 0 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ball_distribution_equality_l2389_238970


namespace NUMINAMATH_CALUDE_sum_of_roots_arithmetic_sequence_l2389_238903

theorem sum_of_roots_arithmetic_sequence (a b c d : ℝ) : 
  0 < c ∧ 0 < b ∧ 0 < a ∧ 
  a > b ∧ b > c ∧ 
  b = a - d ∧ c = a - 2*d ∧ 
  0 < d ∧
  (b^2 - 4*a*c > 0) →
  -(b / a) = -1/3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_arithmetic_sequence_l2389_238903


namespace NUMINAMATH_CALUDE_y_value_l2389_238943

theorem y_value : ∀ y : ℚ, (2/3 - 1/4 = 4/y) → y = 48/5 := by
  sorry

end NUMINAMATH_CALUDE_y_value_l2389_238943


namespace NUMINAMATH_CALUDE_sum_of_squares_parity_l2389_238981

theorem sum_of_squares_parity (a b c : ℤ) (h : Odd (a + b + c)) :
  Odd (a^2 + b^2 - c^2 + 2*a*b) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_parity_l2389_238981


namespace NUMINAMATH_CALUDE_train_platform_passage_time_l2389_238919

/-- Given a train of length 2400 meters that crosses a tree in 90 seconds,
    calculate the time it takes to pass a platform of length 1800 meters. -/
theorem train_platform_passage_time 
  (train_length : ℝ) 
  (tree_crossing_time : ℝ) 
  (platform_length : ℝ) 
  (h1 : train_length = 2400) 
  (h2 : tree_crossing_time = 90) 
  (h3 : platform_length = 1800) : 
  (train_length + platform_length) / (train_length / tree_crossing_time) = 157.5 :=
sorry

end NUMINAMATH_CALUDE_train_platform_passage_time_l2389_238919


namespace NUMINAMATH_CALUDE_polygonal_chain_circle_cover_l2389_238905

/-- A planar closed polygonal chain -/
structure ClosedPolygonalChain where
  vertices : Set (ℝ × ℝ)
  is_closed : True  -- This is a placeholder for the closure property
  perimeter : ℝ

/-- Theorem: For any closed polygonal chain with perimeter 1, 
    there exists a point such that all points on the chain 
    are within distance 1/4 from it -/
theorem polygonal_chain_circle_cover 
  (chain : ClosedPolygonalChain) 
  (h_perimeter : chain.perimeter = 1) :
  ∃ (center : ℝ × ℝ), ∀ (p : ℝ × ℝ), p ∈ chain.vertices → 
    Real.sqrt ((p.1 - center.1)^2 + (p.2 - center.2)^2) ≤ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_polygonal_chain_circle_cover_l2389_238905


namespace NUMINAMATH_CALUDE_negative_reciprocal_of_0125_l2389_238976

def negative_reciprocal (a b : ℝ) : Prop := a * b = -1

theorem negative_reciprocal_of_0125 :
  negative_reciprocal 0.125 (-8) := by
  sorry

end NUMINAMATH_CALUDE_negative_reciprocal_of_0125_l2389_238976


namespace NUMINAMATH_CALUDE_maximum_value_inequality_l2389_238904

theorem maximum_value_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : (2 : ℝ) / 5 ≤ z) (h2 : z ≤ min x y) (h3 : x * z ≥ (4 : ℝ) / 15) (h4 : y * z ≥ (1 : ℝ) / 5) :
  (1 : ℝ) / x + 2 / y + 3 / z ≤ 13 ∧
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧
    (2 : ℝ) / 5 ≤ z₀ ∧ z₀ ≤ min x₀ y₀ ∧ x₀ * z₀ ≥ (4 : ℝ) / 15 ∧ y₀ * z₀ ≥ (1 : ℝ) / 5 ∧
    (1 : ℝ) / x₀ + 2 / y₀ + 3 / z₀ = 13 := by
  sorry

end NUMINAMATH_CALUDE_maximum_value_inequality_l2389_238904


namespace NUMINAMATH_CALUDE_planes_parallel_transitive_l2389_238939

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallelism relation for planes
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem planes_parallel_transitive 
  (α β γ : Plane) 
  (h1 : parallel α γ) 
  (h2 : parallel γ β) : 
  parallel α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_transitive_l2389_238939


namespace NUMINAMATH_CALUDE_line_equation_condition1_line_equation_condition2_line_equation_condition3_l2389_238934

-- Define the line l
def line_l (a b c : ℝ) : Prop := ∀ x y : ℝ, a * x + b * y + c = 0

-- Define the point (1, -2) that the line passes through
def point_condition (a b c : ℝ) : Prop := a * 1 + b * (-2) + c = 0

-- Theorem for condition 1
theorem line_equation_condition1 (a b c : ℝ) :
  point_condition a b c →
  (∃ k : ℝ, k = 1 - π / 12 ∧ b / a = -k) →
  line_l a b c ↔ line_l 1 (-Real.sqrt 3) (-2 * Real.sqrt 3 - 1) :=
sorry

-- Theorem for condition 2
theorem line_equation_condition2 (a b c : ℝ) :
  point_condition a b c →
  (b / a = 1) →
  line_l a b c ↔ line_l 1 (-1) (-3) :=
sorry

-- Theorem for condition 3
theorem line_equation_condition3 (a b c : ℝ) :
  point_condition a b c →
  (c / b = -1) →
  line_l a b c ↔ line_l 1 1 1 :=
sorry

end NUMINAMATH_CALUDE_line_equation_condition1_line_equation_condition2_line_equation_condition3_l2389_238934


namespace NUMINAMATH_CALUDE_regression_line_equation_l2389_238982

/-- Given a regression line with slope -1 passing through the point (1, 2),
    prove that its equation is y = -x + 3 -/
theorem regression_line_equation (slope : ℝ) (center : ℝ × ℝ) :
  slope = -1 →
  center = (1, 2) →
  ∀ x y : ℝ, y = slope * (x - center.1) + center.2 ↔ y = -x + 3 :=
by sorry

end NUMINAMATH_CALUDE_regression_line_equation_l2389_238982


namespace NUMINAMATH_CALUDE_homework_problem_distribution_l2389_238901

theorem homework_problem_distribution (total : ℕ) 
  (multiple_choice free_response true_false : ℕ) : 
  total = 45 → 
  multiple_choice = 2 * free_response → 
  free_response = true_false + 7 → 
  total = multiple_choice + free_response + true_false → 
  true_false = 6 := by
  sorry

end NUMINAMATH_CALUDE_homework_problem_distribution_l2389_238901


namespace NUMINAMATH_CALUDE_min_segment_length_for_cyclists_l2389_238922

/-- Represents a cyclist on a circular track -/
structure Cyclist where
  speed : ℝ
  position : ℝ

/-- The circular track -/
def trackLength : ℝ := 300

/-- Theorem stating the minimum length of track segment where all cyclists will eventually appear -/
theorem min_segment_length_for_cyclists (c1 c2 c3 : Cyclist) 
  (h1 : c1.speed ≠ c2.speed)
  (h2 : c2.speed ≠ c3.speed)
  (h3 : c1.speed ≠ c3.speed)
  (h4 : c1.speed > 0 ∧ c2.speed > 0 ∧ c3.speed > 0) :
  ∃ (d : ℝ), d = 75 ∧ 
  (∀ (t : ℝ), ∃ (t' : ℝ), t' ≥ t ∧ 
    (((c1.position + c1.speed * t') % trackLength - 
      (c2.position + c2.speed * t') % trackLength + trackLength) % trackLength ≤ d ∧
     ((c2.position + c2.speed * t') % trackLength - 
      (c3.position + c3.speed * t') % trackLength + trackLength) % trackLength ≤ d ∧
     ((c1.position + c1.speed * t') % trackLength - 
      (c3.position + c3.speed * t') % trackLength + trackLength) % trackLength ≤ d)) :=
sorry

end NUMINAMATH_CALUDE_min_segment_length_for_cyclists_l2389_238922


namespace NUMINAMATH_CALUDE_wendy_run_distance_l2389_238935

/-- The distance Wendy walked in miles -/
def walked_distance : ℝ := 9.166666666666666

/-- The additional distance Wendy ran compared to what she walked in miles -/
def additional_run_distance : ℝ := 10.666666666666666

/-- The total distance Wendy ran in miles -/
def total_run_distance : ℝ := walked_distance + additional_run_distance

theorem wendy_run_distance : total_run_distance = 19.833333333333332 := by
  sorry

end NUMINAMATH_CALUDE_wendy_run_distance_l2389_238935


namespace NUMINAMATH_CALUDE_milk_butter_revenue_l2389_238966

/-- Calculates the total revenue from selling milk and butter --/
def total_revenue (num_cows : ℕ) (milk_per_cow : ℕ) (milk_price : ℚ) (butter_sticks_per_gallon : ℕ) (butter_price : ℚ) : ℚ :=
  let total_milk := num_cows * milk_per_cow
  let milk_revenue := total_milk * milk_price
  milk_revenue

theorem milk_butter_revenue :
  let num_cows : ℕ := 12
  let milk_per_cow : ℕ := 4
  let milk_price : ℚ := 3
  let butter_sticks_per_gallon : ℕ := 2
  let butter_price : ℚ := 3/2
  total_revenue num_cows milk_per_cow milk_price butter_sticks_per_gallon butter_price = 144 := by
  sorry

end NUMINAMATH_CALUDE_milk_butter_revenue_l2389_238966


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2389_238949

/-- Given a polynomial equation, prove the sum of specific coefficients -/
theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℝ) :
  (∀ x : ℝ, (x^2 + 1) * (2*x + 1)^9 = 
    a₀ + a₁*(x+1) + a₂*(x+1)^2 + a₃*(x+1)^3 + a₄*(x+1)^4 + 
    a₅*(x+1)^5 + a₆*(x+1)^6 + a₇*(x+1)^7 + a₈*(x+1)^8 + 
    a₉*(x+1)^9 + a₁₀*(x+1)^10 + a₁₁*(x+1)^11) →
  a₁ + a₂ + a₁₁ = 781 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2389_238949


namespace NUMINAMATH_CALUDE_max_time_sum_of_digits_l2389_238957

/-- Represents a time in 24-hour format -/
structure Time24 where
  hours : Fin 24
  minutes : Fin 60

/-- Calculates the sum of digits for a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Calculates the sum of digits for a Time24 -/
def timeSumOfDigits (t : Time24) : ℕ :=
  sumOfDigits t.hours.val + sumOfDigits t.minutes.val

/-- The maximum sum of digits for any Time24 -/
def maxTimeSumOfDigits : ℕ := 24

theorem max_time_sum_of_digits :
  ∀ t : Time24, timeSumOfDigits t ≤ maxTimeSumOfDigits := by sorry

end NUMINAMATH_CALUDE_max_time_sum_of_digits_l2389_238957


namespace NUMINAMATH_CALUDE_conic_is_hyperbola_l2389_238987

/-- The equation of a conic section -/
def conicEquation (x y : ℝ) : Prop :=
  (x - 3)^2 = 4 * (y + 2)^2 + 25

/-- Definition of a hyperbola -/
def isHyperbola (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b c d e : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ a * b < 0 ∧
    ∀ x y, f x y ↔ a * x^2 + b * y^2 + c * x + d * y + e = 0

/-- Theorem: The given conic equation represents a hyperbola -/
theorem conic_is_hyperbola : isHyperbola conicEquation :=
sorry

end NUMINAMATH_CALUDE_conic_is_hyperbola_l2389_238987


namespace NUMINAMATH_CALUDE_neither_necessary_nor_sufficient_l2389_238964

theorem neither_necessary_nor_sufficient (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ¬(∀ a b, a > b → 1/a < 1/b) ∧ ¬(∀ a b, 1/a < 1/b → a > b) :=
sorry

end NUMINAMATH_CALUDE_neither_necessary_nor_sufficient_l2389_238964


namespace NUMINAMATH_CALUDE_sqrt_eighteen_minus_sqrt_two_l2389_238921

theorem sqrt_eighteen_minus_sqrt_two : Real.sqrt 18 - Real.sqrt 2 = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eighteen_minus_sqrt_two_l2389_238921
