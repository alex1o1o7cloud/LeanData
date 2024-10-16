import Mathlib

namespace NUMINAMATH_CALUDE_exponent_simplification_l2870_287046

theorem exponent_simplification (x : ℝ) : (x^5 * x^3) * x^2 = x^10 := by
  sorry

end NUMINAMATH_CALUDE_exponent_simplification_l2870_287046


namespace NUMINAMATH_CALUDE_shortest_distance_on_specific_cone_l2870_287057

/-- Represents a right circular cone -/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Represents a point on the surface of a cone -/
structure ConePoint where
  distanceFromVertex : ℝ
  angle : ℝ  -- Angle from a reference line on the surface

/-- Calculates the shortest distance between two points on the surface of a cone -/
def shortestDistanceOnCone (c : Cone) (p1 p2 : ConePoint) : ℝ :=
  sorry

/-- The main theorem to prove -/
theorem shortest_distance_on_specific_cone :
  let c : Cone := { baseRadius := 500, height := 400 }
  let p1 : ConePoint := { distanceFromVertex := 150, angle := 0 }
  let p2 : ConePoint := { distanceFromVertex := 400 * Real.sqrt 2, angle := π }
  shortestDistanceOnCone c p1 p2 = 25 * Real.sqrt 741 := by
  sorry

end NUMINAMATH_CALUDE_shortest_distance_on_specific_cone_l2870_287057


namespace NUMINAMATH_CALUDE_bacteria_growth_l2870_287053

/-- Calculates the bacteria population after a given time -/
def bacteria_population (initial_count : ℕ) (doubling_time : ℕ) (elapsed_time : ℕ) : ℕ :=
  initial_count * 2^(elapsed_time / doubling_time)

/-- Theorem: The bacteria population after 20 minutes is 240 -/
theorem bacteria_growth : bacteria_population 15 5 20 = 240 := by
  sorry

end NUMINAMATH_CALUDE_bacteria_growth_l2870_287053


namespace NUMINAMATH_CALUDE_plot_size_in_acres_l2870_287055

-- Define the scale of the map
def map_scale : ℝ := 1

-- Define the dimensions of the plot on the map
def map_length : ℝ := 20
def map_width : ℝ := 25

-- Define the conversion from square miles to acres
def acres_per_square_mile : ℝ := 640

-- State the theorem
theorem plot_size_in_acres :
  let real_area : ℝ := map_length * map_width * map_scale^2
  real_area * acres_per_square_mile = 320000 := by
  sorry

end NUMINAMATH_CALUDE_plot_size_in_acres_l2870_287055


namespace NUMINAMATH_CALUDE_monotonic_decreasing_interval_l2870_287033

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x - 1

theorem monotonic_decreasing_interval :
  ∀ x y : ℝ, x < y → x < 0 → f x > f y :=
by sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_interval_l2870_287033


namespace NUMINAMATH_CALUDE_octagon_square_ratio_l2870_287087

theorem octagon_square_ratio (s r : ℝ) (h : s > 0) (k : r > 0) :
  s^2 = 2 * r^2 * Real.sqrt 2 → r / s = 1 / Real.sqrt (2 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_octagon_square_ratio_l2870_287087


namespace NUMINAMATH_CALUDE_sin_90_degrees_l2870_287075

theorem sin_90_degrees : Real.sin (π / 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_90_degrees_l2870_287075


namespace NUMINAMATH_CALUDE_angle_sum_pi_half_l2870_287065

theorem angle_sum_pi_half (α β : Real) (h1 : 0 < α) (h2 : α < π/2) (h3 : 0 < β) (h4 : β < π/2)
  (h5 : 3 * Real.sin α ^ 2 + 2 * Real.sin β ^ 2 = 1)
  (h6 : 3 * Real.sin (2 * α) - 2 * Real.sin (2 * β) = 0) :
  α + 2 * β = π/2 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_pi_half_l2870_287065


namespace NUMINAMATH_CALUDE_nineteenth_term_is_zero_l2870_287089

/-- A sequence with specific properties -/
def special_sequence (a : ℕ → ℚ) : Prop :=
  a 3 = 2 ∧ 
  a 7 = 1 ∧ 
  ∃ d : ℚ, ∀ n : ℕ, (1 / (a (n + 1) + 1) - 1 / (a n + 1)) = d

/-- The 19th term of the special sequence is 0 -/
theorem nineteenth_term_is_zero (a : ℕ → ℚ) (h : special_sequence a) : 
  a 19 = 0 := by
sorry

end NUMINAMATH_CALUDE_nineteenth_term_is_zero_l2870_287089


namespace NUMINAMATH_CALUDE_sqrt_fraction_simplification_l2870_287072

theorem sqrt_fraction_simplification (x : ℝ) (h : x < -1) :
  Real.sqrt ((x + 1) / (2 - (x + 2) / x)) = Real.sqrt (|x^2 + x| / |x - 2|) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fraction_simplification_l2870_287072


namespace NUMINAMATH_CALUDE_remainder_theorem_l2870_287001

theorem remainder_theorem (x : ℤ) : 
  (x^15 + 3) % (x + 2) = (-2)^15 + 3 :=
by sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2870_287001


namespace NUMINAMATH_CALUDE_division_remainder_proof_l2870_287054

theorem division_remainder_proof (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) :
  dividend = 139 →
  divisor = 19 →
  quotient = 7 →
  dividend = divisor * quotient + remainder →
  remainder = 6 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_proof_l2870_287054


namespace NUMINAMATH_CALUDE_floor_sqrt_sum_equality_l2870_287036

theorem floor_sqrt_sum_equality (n : ℕ) : 
  ⌊Real.sqrt n + Real.sqrt (n + 1)⌋ = ⌊Real.sqrt (n + 1) + Real.sqrt (n + 2)⌋ := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_sum_equality_l2870_287036


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2870_287019

def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_sum
  (a : ℕ → ℤ)
  (h_arith : arithmetic_sequence a (-2))
  (h_sum : (Finset.range 33).sum (fun i => a (3 * i + 1)) = 50) :
  (Finset.range 33).sum (fun i => a (3 * i + 3)) = -82 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2870_287019


namespace NUMINAMATH_CALUDE_simple_interest_rate_calculation_l2870_287026

/-- Given an initial sum of money that amounts to 9800 after 5 years
    and 12005 after 8 years at the same rate of simple interest,
    prove that the rate of interest per annum is 7.5% -/
theorem simple_interest_rate_calculation (P : ℝ) (R : ℝ) :
  P * (1 + 5 * R / 100) = 9800 →
  P * (1 + 8 * R / 100) = 12005 →
  R = 7.5 :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_rate_calculation_l2870_287026


namespace NUMINAMATH_CALUDE_square_root_theorem_l2870_287003

theorem square_root_theorem (x : ℝ) :
  Real.sqrt (x + 3) = 3 → (x + 3)^2 = 81 := by
  sorry

end NUMINAMATH_CALUDE_square_root_theorem_l2870_287003


namespace NUMINAMATH_CALUDE_rectangle_length_l2870_287081

-- Define the radius of the circle
def R : ℝ := 2.5

-- Define pi as an approximation
def π : ℝ := 3.14

-- Define the perimeter of the rectangle
def perimeter : ℝ := 20.7

-- Theorem stating that the length of the rectangle is 7.85 cm
theorem rectangle_length : (π * R) = 7.85 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_l2870_287081


namespace NUMINAMATH_CALUDE_parallel_iff_m_eq_neg_two_l2870_287056

-- Define the lines as functions of x and y
def line1 (m : ℝ) (x y : ℝ) : Prop := 2*x + m*y - 2*m + 4 = 0
def line2 (m : ℝ) (x y : ℝ) : Prop := m*x + 2*y - m + 2 = 0

-- Define what it means for two lines to be parallel
def parallel (m : ℝ) : Prop := ∀ (x y : ℝ), line1 m x y ↔ ∃ (k : ℝ), line2 m (x + k) (y + k)

-- State the theorem
theorem parallel_iff_m_eq_neg_two :
  ∀ m : ℝ, parallel m ↔ m = -2 := by sorry

end NUMINAMATH_CALUDE_parallel_iff_m_eq_neg_two_l2870_287056


namespace NUMINAMATH_CALUDE_square_area_ratio_l2870_287051

theorem square_area_ratio (a b : ℝ) (ha : a > 0) (hb : b = a * Real.sqrt 3) :
  b ^ 2 = 3 * a ^ 2 := by sorry

end NUMINAMATH_CALUDE_square_area_ratio_l2870_287051


namespace NUMINAMATH_CALUDE_vector_inequality_l2870_287032

theorem vector_inequality (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a^2 + b^2) * (c^2 + d^2) ≥ (a*c + b*d)^2 := by
  sorry

end NUMINAMATH_CALUDE_vector_inequality_l2870_287032


namespace NUMINAMATH_CALUDE_x_value_l2870_287060

theorem x_value : ∃ x : ℝ, (49 / 49 = x ^ 4) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l2870_287060


namespace NUMINAMATH_CALUDE_sample_size_is_200_l2870_287052

/-- Represents a statistical survey of students -/
structure StudentSurvey where
  total_students : ℕ
  selected_students : ℕ

/-- Definition of sample size for a student survey -/
def sample_size (survey : StudentSurvey) : ℕ := survey.selected_students

/-- Theorem stating that for the given survey, the sample size is 200 -/
theorem sample_size_is_200 (survey : StudentSurvey) 
  (h1 : survey.total_students = 2000) 
  (h2 : survey.selected_students = 200) : 
  sample_size survey = 200 := by
  sorry

#check sample_size_is_200

end NUMINAMATH_CALUDE_sample_size_is_200_l2870_287052


namespace NUMINAMATH_CALUDE_line_parabola_intersection_l2870_287012

/-- A line with equation y = kx + 2 and a parabola with equation y² = 8x have exactly one point in common if and only if k = 1 or k = 0 -/
theorem line_parabola_intersection (k : ℝ) : 
  (∃! p : ℝ × ℝ, p.2 = k * p.1 + 2 ∧ p.2^2 = 8 * p.1) ↔ (k = 1 ∨ k = 0) :=
sorry

end NUMINAMATH_CALUDE_line_parabola_intersection_l2870_287012


namespace NUMINAMATH_CALUDE_crayons_given_to_friends_l2870_287097

theorem crayons_given_to_friends (initial_crayons : ℕ) (lost_crayons : ℕ) (total_lost_or_given : ℕ) : 
  initial_crayons = 65 → lost_crayons = 16 → total_lost_or_given = 229 →
  total_lost_or_given - lost_crayons = 213 := by
  sorry

end NUMINAMATH_CALUDE_crayons_given_to_friends_l2870_287097


namespace NUMINAMATH_CALUDE_spider_eats_all_flies_l2870_287037

/-- Represents a position on the grid -/
structure Position where
  x : Nat
  y : Nat

/-- Represents the spider's movement strategy -/
structure SpiderStrategy where
  initialPosition : Position
  moveSequence : List Position

/-- Represents the web with flies -/
structure Web where
  size : Nat
  flyPositions : List Position

/-- Theorem stating that the spider can eat all flies in at most 1980 moves -/
theorem spider_eats_all_flies (web : Web) (strategy : SpiderStrategy) : 
  web.size = 100 → 
  web.flyPositions.length = 100 → 
  strategy.initialPosition.x = 0 ∨ strategy.initialPosition.x = 99 → 
  strategy.initialPosition.y = 0 ∨ strategy.initialPosition.y = 99 → 
  ∃ (moves : List Position), 
    moves.length ≤ 1980 ∧ 
    (∀ fly ∈ web.flyPositions, fly ∈ moves) := by
  sorry

end NUMINAMATH_CALUDE_spider_eats_all_flies_l2870_287037


namespace NUMINAMATH_CALUDE_first_half_speed_l2870_287004

/-- Proves that given a 60-mile trip where the average speed on the second half is 16 mph faster
    than the first half, and the average speed for the entire trip is 30 mph,
    the average speed during the first half is 24 mph. -/
theorem first_half_speed (total_distance : ℝ) (speed_increase : ℝ) (total_avg_speed : ℝ)
  (h1 : total_distance = 60)
  (h2 : speed_increase = 16)
  (h3 : total_avg_speed = 30) :
  ∃ (first_half_speed : ℝ),
    first_half_speed > 0 ∧
    (total_distance / 2) / first_half_speed + (total_distance / 2) / (first_half_speed + speed_increase) = total_distance / total_avg_speed ∧
    first_half_speed = 24 :=
by sorry

end NUMINAMATH_CALUDE_first_half_speed_l2870_287004


namespace NUMINAMATH_CALUDE_jerrys_age_l2870_287031

theorem jerrys_age (mickey_age jerry_age : ℝ) : 
  mickey_age = 2.5 * jerry_age - 5 →
  mickey_age = 20 →
  jerry_age = 10 := by
sorry

end NUMINAMATH_CALUDE_jerrys_age_l2870_287031


namespace NUMINAMATH_CALUDE_geometric_sum_four_l2870_287094

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = a n * q

theorem geometric_sum_four (a : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a q →
  a 3 = 4 →
  a 2 + a 4 = -10 →
  |q| > 1 →
  a 1 + a 2 + a 3 + a 4 = -5 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_four_l2870_287094


namespace NUMINAMATH_CALUDE_units_digit_of_product_division_l2870_287071

theorem units_digit_of_product_division : 
  (15 * 16 * 17 * 18 * 19 * 20) / 500 % 10 = 8 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_product_division_l2870_287071


namespace NUMINAMATH_CALUDE_sugar_to_add_l2870_287062

/-- Mary is baking a cake. This theorem proves how many more cups of sugar she needs to add. -/
theorem sugar_to_add (recipe_sugar : ℕ) (added_sugar : ℕ) (h1 : recipe_sugar = 13) (h2 : added_sugar = 2) :
  recipe_sugar - added_sugar = 11 := by
  sorry

end NUMINAMATH_CALUDE_sugar_to_add_l2870_287062


namespace NUMINAMATH_CALUDE_paula_tickets_needed_l2870_287008

/-- Represents the number of times Paula wants to ride each attraction -/
structure RideFrequencies where
  goKarts : Nat
  bumperCars : Nat
  rollerCoaster : Nat
  ferrisWheel : Nat

/-- Represents the ticket cost for each attraction -/
structure TicketCosts where
  goKarts : Nat
  bumperCars : Nat
  rollerCoaster : Nat
  ferrisWheel : Nat

/-- Calculates the total number of tickets needed based on ride frequencies and ticket costs -/
def totalTicketsNeeded (freq : RideFrequencies) (costs : TicketCosts) : Nat :=
  freq.goKarts * costs.goKarts +
  freq.bumperCars * costs.bumperCars +
  freq.rollerCoaster * costs.rollerCoaster +
  freq.ferrisWheel * costs.ferrisWheel

/-- Theorem stating that Paula needs 52 tickets in total -/
theorem paula_tickets_needed :
  let frequencies : RideFrequencies := {
    goKarts := 2,
    bumperCars := 4,
    rollerCoaster := 3,
    ferrisWheel := 1
  }
  let costs : TicketCosts := {
    goKarts := 4,
    bumperCars := 5,
    rollerCoaster := 7,
    ferrisWheel := 3
  }
  totalTicketsNeeded frequencies costs = 52 := by
  sorry

end NUMINAMATH_CALUDE_paula_tickets_needed_l2870_287008


namespace NUMINAMATH_CALUDE_book_distribution_l2870_287068

theorem book_distribution (n : ℕ) (b : ℕ) : 
  (3 * n + 6 = b) →                     -- Condition 1
  (5 * n - 5 ≤ b) →                     -- Condition 2 (lower bound)
  (b < 5 * n - 2) →                     -- Condition 2 (upper bound)
  (n = 5 ∧ b = 21) :=                   -- Conclusion
by sorry

end NUMINAMATH_CALUDE_book_distribution_l2870_287068


namespace NUMINAMATH_CALUDE_house_distance_theorem_l2870_287090

/-- Represents the position of a house on a street -/
structure House where
  position : ℝ

/-- Represents a street with four houses -/
structure Street where
  andrey : House
  borya : House
  vova : House
  gleb : House

/-- The distance between two houses -/
def distance (h1 h2 : House) : ℝ := 
  |h1.position - h2.position|

theorem house_distance_theorem (s : Street) : 
  (distance s.andrey s.borya = 600 ∧ 
   distance s.vova s.gleb = 600 ∧ 
   distance s.andrey s.gleb = 3 * distance s.borya s.vova) → 
  (distance s.andrey s.gleb = 900 ∨ distance s.andrey s.gleb = 1800) :=
sorry

end NUMINAMATH_CALUDE_house_distance_theorem_l2870_287090


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2870_287030

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 4 + a 8 = 16) :
  a 2 + a 10 = 16 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2870_287030


namespace NUMINAMATH_CALUDE_lcd_of_fractions_l2870_287088

theorem lcd_of_fractions (a b c d : ℕ) (ha : a = 2) (hb : b = 4) (hc : c = 5) (hd : d = 6) :
  Nat.lcm a (Nat.lcm b (Nat.lcm c d)) = 60 := by
  sorry

end NUMINAMATH_CALUDE_lcd_of_fractions_l2870_287088


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l2870_287083

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 5*x - 6 ≤ 0}
def B : Set ℝ := {x | x > 7}

-- State the theorem
theorem complement_A_intersect_B :
  (Set.compl A) ∩ B = Set.Ioi 7 := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l2870_287083


namespace NUMINAMATH_CALUDE_spaceship_journey_theorem_l2870_287025

/-- A spaceship's journey to another planet -/
def spaceship_journey (total_journey_time : ℕ) (initial_travel_time : ℕ) (first_break : ℕ) (second_travel_time : ℕ) (second_break : ℕ) (travel_segment : ℕ) (break_duration : ℕ) : Prop :=
  let total_hours : ℕ := total_journey_time * 24
  let initial_breaks : ℕ := first_break + second_break
  let initial_total_time : ℕ := initial_travel_time + second_travel_time + initial_breaks
  let remaining_time : ℕ := total_hours - initial_total_time
  let full_segments : ℕ := remaining_time / (travel_segment + break_duration)
  let total_breaks : ℕ := initial_breaks + full_segments * break_duration
  total_breaks = 8

theorem spaceship_journey_theorem :
  spaceship_journey 3 10 3 10 1 11 1 := by sorry

end NUMINAMATH_CALUDE_spaceship_journey_theorem_l2870_287025


namespace NUMINAMATH_CALUDE_daily_harvest_l2870_287061

/-- Given an orchard with a certain number of sections and a fixed number of sacks harvested per section daily, 
    calculate the total number of sacks harvested every day. -/
theorem daily_harvest (sections : ℕ) (sacks_per_section : ℕ) : 
  sections = 8 → sacks_per_section = 45 → sections * sacks_per_section = 360 := by
  sorry

#check daily_harvest

end NUMINAMATH_CALUDE_daily_harvest_l2870_287061


namespace NUMINAMATH_CALUDE_sequence_a_closed_form_l2870_287038

def sequence_a : ℕ → ℤ
  | 0 => 1
  | 1 => 5
  | (n + 2) => (2 * (sequence_a (n + 1))^2 - 3 * sequence_a (n + 1) - 9) / (2 * sequence_a n)

theorem sequence_a_closed_form (n : ℕ) : sequence_a n = 2^(n + 2) - 3 := by
  sorry

end NUMINAMATH_CALUDE_sequence_a_closed_form_l2870_287038


namespace NUMINAMATH_CALUDE_z_in_second_quadrant_l2870_287048

-- Define the complex number z
def z : ℂ := (1 + Complex.I) * (2 * Complex.I)

-- Theorem statement
theorem z_in_second_quadrant : Real.sign (z.re) = -1 ∧ Real.sign (z.im) = 1 := by
  sorry

end NUMINAMATH_CALUDE_z_in_second_quadrant_l2870_287048


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2870_287095

/-- The sum of the first n terms of an arithmetic sequence -/
def arithmetic_sum (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a₁ + (n - 1) * d) / 2

/-- Theorem: The sum of the first 10 terms of the arithmetic sequence
    with first term -5 and common difference 6 is 220 -/
theorem arithmetic_sequence_sum :
  arithmetic_sum (-5) 6 10 = 220 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2870_287095


namespace NUMINAMATH_CALUDE_pentagon_condition_l2870_287018

/-- Represents the lengths of five segments cut from a wire -/
structure WireSegments where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  sum_eq_two : a + b + c + d + e = 2
  all_positive : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e

/-- Checks if the given segments can form a pentagon -/
def can_form_pentagon (segments : WireSegments) : Prop :=
  segments.a + segments.b + segments.c + segments.d > segments.e ∧
  segments.a + segments.b + segments.c + segments.e > segments.d ∧
  segments.a + segments.b + segments.d + segments.e > segments.c ∧
  segments.a + segments.c + segments.d + segments.e > segments.b ∧
  segments.b + segments.c + segments.d + segments.e > segments.a

/-- Theorem stating the necessary and sufficient condition for forming a pentagon -/
theorem pentagon_condition (segments : WireSegments) :
  can_form_pentagon segments ↔ segments.a < 1 ∧ segments.b < 1 ∧ segments.c < 1 ∧ segments.d < 1 ∧ segments.e < 1 :=
sorry

end NUMINAMATH_CALUDE_pentagon_condition_l2870_287018


namespace NUMINAMATH_CALUDE_sine_amplitude_l2870_287079

theorem sine_amplitude (a b c d : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) 
  (h5 : ∀ x, -3 ≤ a * Real.sin (b * x + c) + d) 
  (h6 : ∀ x, a * Real.sin (b * x + c) + d ≤ 5) : a = 4 := by
sorry

end NUMINAMATH_CALUDE_sine_amplitude_l2870_287079


namespace NUMINAMATH_CALUDE_min_sum_reciprocals_roots_l2870_287022

theorem min_sum_reciprocals_roots (k : ℝ) (x₁ x₂ : ℝ) : 
  x₁ > 0 ∧ x₂ > 0 ∧ 
  x₁^2 - k*x₁ + k + 3 = 0 ∧ 
  x₂^2 - k*x₂ + k + 3 = 0 ∧ 
  x₁ ≠ x₂ →
  (∃ (s : ℝ), s = 1/x₁ + 1/x₂ ∧ s ≥ 2/3 ∧ ∀ (t : ℝ), t = 1/x₁ + 1/x₂ → t ≥ 2/3) :=
sorry

end NUMINAMATH_CALUDE_min_sum_reciprocals_roots_l2870_287022


namespace NUMINAMATH_CALUDE_alex_final_silver_tokens_l2870_287035

/-- Represents the number of tokens Alex has -/
structure TokenCount where
  red : ℕ
  blue : ℕ
  silver : ℕ

/-- Represents a token exchange booth -/
structure Booth where
  red_in : ℕ
  blue_in : ℕ
  red_out : ℕ
  blue_out : ℕ
  silver_out : ℕ

/-- Applies a single exchange at a booth -/
def apply_exchange (tokens : TokenCount) (booth : Booth) : TokenCount :=
  { red := tokens.red - booth.red_in + booth.red_out,
    blue := tokens.blue - booth.blue_in + booth.blue_out,
    silver := tokens.silver + booth.silver_out }

/-- Checks if an exchange is possible -/
def can_exchange (tokens : TokenCount) (booth : Booth) : Prop :=
  tokens.red ≥ booth.red_in ∧ tokens.blue ≥ booth.blue_in

/-- The final state after all possible exchanges -/
def final_state (initial : TokenCount) (booth1 booth2 : Booth) : TokenCount :=
  sorry  -- The implementation would go here

/-- Theorem stating that Alex will end up with 58 silver tokens -/
theorem alex_final_silver_tokens :
  let initial := TokenCount.mk 100 50 0
  let booth1 := Booth.mk 3 0 0 1 2
  let booth2 := Booth.mk 0 4 1 0 1
  (final_state initial booth1 booth2).silver = 58 := by
  sorry


end NUMINAMATH_CALUDE_alex_final_silver_tokens_l2870_287035


namespace NUMINAMATH_CALUDE_area_of_triangle_MOI_l2870_287064

/-- Triangle ABC with given side lengths -/
structure Triangle where
  AB : ℝ
  AC : ℝ
  BC : ℝ

/-- Circumcenter of a triangle -/
def circumcenter (t : Triangle) : ℝ × ℝ := sorry

/-- Incenter of a triangle -/
def incenter (t : Triangle) : ℝ × ℝ := sorry

/-- Point M such that a circle centered at M is tangent to AC, BC, and the circumcircle -/
def tangentPoint (t : Triangle) : ℝ × ℝ := sorry

/-- Area of a triangle given three points -/
def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

/-- Main theorem -/
theorem area_of_triangle_MOI (t : Triangle) 
  (h1 : t.AB = 15) (h2 : t.AC = 8) (h3 : t.BC = 7) : 
  triangleArea (tangentPoint t) (circumcenter t) (incenter t) = 1.765 := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_MOI_l2870_287064


namespace NUMINAMATH_CALUDE_largest_prime_for_primality_check_l2870_287006

theorem largest_prime_for_primality_check : 
  ∀ n : ℕ, 1000 ≤ n ∧ n ≤ 1050 → 
  ∀ p : ℕ, Prime p ∧ p^2 ≤ n → p ≤ 31 :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_for_primality_check_l2870_287006


namespace NUMINAMATH_CALUDE_sand_per_lorry_l2870_287010

/-- Calculates the number of tons of sand per lorry given the following conditions:
  * 500 bags of cement are provided
  * Cement costs $10 per bag
  * 20 lorries of sand are received
  * Sand costs $40 per ton
  * Total cost for all materials is $13000
-/
theorem sand_per_lorry (cement_bags : ℕ) (cement_cost : ℚ) (lorries : ℕ) (sand_cost : ℚ) (total_cost : ℚ) :
  cement_bags = 500 →
  cement_cost = 10 →
  lorries = 20 →
  sand_cost = 40 →
  total_cost = 13000 →
  (total_cost - cement_bags * cement_cost) / sand_cost / lorries = 10 := by
  sorry

#check sand_per_lorry

end NUMINAMATH_CALUDE_sand_per_lorry_l2870_287010


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2870_287069

theorem sufficient_not_necessary : 
  (∃ x : ℝ, x = 3 → x^2 = 9) ∧ 
  (∃ x : ℝ, x^2 = 9 ∧ x ≠ 3) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2870_287069


namespace NUMINAMATH_CALUDE_inequality_proof_l2870_287029

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : x - Real.sqrt x ≤ y - 1/4 ∧ y - 1/4 ≤ x + Real.sqrt x) :
  y - Real.sqrt y ≤ x - 1/4 ∧ x - 1/4 ≤ y + Real.sqrt y := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2870_287029


namespace NUMINAMATH_CALUDE_sum_of_distances_constant_l2870_287013

/-- An equilateral triangle with side length a -/
structure EquilateralTriangle where
  a : ℝ
  a_pos : a > 0

/-- A point on one side of the equilateral triangle -/
structure PointOnSide (triangle : EquilateralTriangle) where
  x : ℝ
  y : ℝ

/-- The sum of perpendicular distances from a point on one side to the other two sides -/
def sumOfDistances (triangle : EquilateralTriangle) (point : PointOnSide triangle) : ℝ := sorry

/-- Theorem: The sum of distances from any point on one side of an equilateral triangle
    to the other two sides is constant and equal to (a√3)/2 -/
theorem sum_of_distances_constant (triangle : EquilateralTriangle) 
  (point : PointOnSide triangle) : 
  sumOfDistances triangle point = (triangle.a * Real.sqrt 3) / 2 := by sorry

end NUMINAMATH_CALUDE_sum_of_distances_constant_l2870_287013


namespace NUMINAMATH_CALUDE_program_sum_equals_expected_sum_l2870_287077

def program_sum (n : ℕ) : ℕ :=
  let rec inner_sum (k : ℕ) : ℕ :=
    match k with
    | 0 => 0
    | k+1 => k+1 + inner_sum k
  let rec outer_sum (i : ℕ) : ℕ :=
    match i with
    | 0 => 0
    | i+1 => inner_sum (i+1) + outer_sum i
  outer_sum n

def expected_sum (n : ℕ) : ℕ :=
  let rec sum_of_sums (k : ℕ) : ℕ :=
    match k with
    | 0 => 0
    | k+1 => (List.range (k+1)).sum + sum_of_sums k
  sum_of_sums n

theorem program_sum_equals_expected_sum (n : ℕ) :
  program_sum n = expected_sum n := by
  sorry

end NUMINAMATH_CALUDE_program_sum_equals_expected_sum_l2870_287077


namespace NUMINAMATH_CALUDE_pages_per_booklet_l2870_287034

theorem pages_per_booklet (total_booklets : ℕ) (total_pages : ℕ) 
  (h1 : total_booklets = 49) 
  (h2 : total_pages = 441) : 
  total_pages / total_booklets = 9 := by
  sorry

end NUMINAMATH_CALUDE_pages_per_booklet_l2870_287034


namespace NUMINAMATH_CALUDE_change_calculation_l2870_287099

def bracelet_price : ℚ := 15
def necklace_price : ℚ := 10
def mug_price : ℚ := 20
def keychain_price : ℚ := 5

def bracelet_quantity : ℕ := 3
def necklace_quantity : ℕ := 2
def mug_quantity : ℕ := 1
def keychain_quantity : ℕ := 4

def discount_rate : ℚ := 12 / 100
def payment : ℚ := 100

def total_before_discount : ℚ :=
  bracelet_price * bracelet_quantity +
  necklace_price * necklace_quantity +
  mug_price * mug_quantity +
  keychain_price * keychain_quantity

def discount_amount : ℚ := total_before_discount * discount_rate
def final_amount : ℚ := total_before_discount - discount_amount

theorem change_calculation :
  payment - final_amount = 760 / 100 := by sorry

end NUMINAMATH_CALUDE_change_calculation_l2870_287099


namespace NUMINAMATH_CALUDE_population_characteristics_changeable_l2870_287045

/-- Represents a population of organisms -/
structure Population where
  species : Type
  individuals : Set species
  space : Type
  time : Type

/-- Characteristics of a population -/
structure PopulationCharacteristics where
  density : ℝ
  birth_rate : ℝ
  death_rate : ℝ
  immigration_rate : ℝ
  age_composition : Set ℕ
  sex_ratio : ℝ

/-- A population has characteristics that can change over time -/
def population_characteristics_can_change (p : Population) : Prop :=
  ∃ (t₁ t₂ : p.time) (c₁ c₂ : PopulationCharacteristics),
    t₁ ≠ t₂ → c₁ ≠ c₂

/-- The main theorem stating that population characteristics can change over time -/
theorem population_characteristics_changeable :
  ∀ (p : Population), population_characteristics_can_change p :=
sorry

end NUMINAMATH_CALUDE_population_characteristics_changeable_l2870_287045


namespace NUMINAMATH_CALUDE_star_commutative_star_not_distributive_star_has_identity_star_identity_is_neg_one_l2870_287041

-- Define the binary operation
def star (x y : ℝ) : ℝ := (x + 2) * (y + 2) - 2

-- Theorem for commutativity
theorem star_commutative : ∀ x y : ℝ, star x y = star y x := by sorry

-- Theorem for non-distributivity
theorem star_not_distributive : ¬(∀ x y z : ℝ, star x (y + z) = star x y + star x z) := by sorry

-- Theorem for existence of identity element
theorem star_has_identity : ∃ e : ℝ, ∀ x : ℝ, star x e = x := by sorry

-- Theorem that -1 is the identity element
theorem star_identity_is_neg_one : ∀ x : ℝ, star x (-1) = x := by sorry

end NUMINAMATH_CALUDE_star_commutative_star_not_distributive_star_has_identity_star_identity_is_neg_one_l2870_287041


namespace NUMINAMATH_CALUDE_quadratic_root_in_unit_interval_l2870_287091

theorem quadratic_root_in_unit_interval (a b : ℝ) : 
  ∃ x : ℝ, 0 < x ∧ x < 1 ∧ 3 * a * x^2 + 2 * b * x - (a + b) = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_in_unit_interval_l2870_287091


namespace NUMINAMATH_CALUDE_square_circle_ratio_l2870_287070

theorem square_circle_ratio (r c d : ℝ) (h : r > 0) (hc : c > 0) (hd : d > c) :
  let s := 2 * r
  s^2 = (c / d) * (s^2 - π * r^2) →
  s / r = Real.sqrt (c * π) / Real.sqrt (d - c) := by
sorry

end NUMINAMATH_CALUDE_square_circle_ratio_l2870_287070


namespace NUMINAMATH_CALUDE_r₂_bound_r₂_bound_tight_l2870_287063

/-- A function f(x) = x² - r₂x + r₃ -/
def f (r₂ r₃ : ℝ) (x : ℝ) : ℝ := x^2 - r₂*x + r₃

/-- Sequence g_n defined recursively -/
def g (r₂ r₃ : ℝ) : ℕ → ℝ
| 0 => 0
| n + 1 => f r₂ r₃ (g r₂ r₃ n)

/-- The statement that needs to be proved -/
theorem r₂_bound (r₂ r₃ : ℝ) :
  (∀ i : ℕ, i ≤ 2011 → g r₂ r₃ (2*i) < g r₂ r₃ (2*i+1) ∧ g r₂ r₃ (2*i+1) > g r₂ r₃ (2*i+2)) →
  (∃ j : ℕ, ∀ i : ℕ, i > j → g r₂ r₃ (i+1) > g r₂ r₃ i) →
  (∀ M : ℝ, ∃ n : ℕ, g r₂ r₃ n > M) →
  abs r₂ ≥ 2 :=
by sorry

/-- The bound is tight -/
theorem r₂_bound_tight : ∀ ε > 0, ∃ r₂ r₃ : ℝ,
  (∀ i : ℕ, i ≤ 2011 → g r₂ r₃ (2*i) < g r₂ r₃ (2*i+1) ∧ g r₂ r₃ (2*i+1) > g r₂ r₃ (2*i+2)) ∧
  (∃ j : ℕ, ∀ i : ℕ, i > j → g r₂ r₃ (i+1) > g r₂ r₃ i) ∧
  (∀ M : ℝ, ∃ n : ℕ, g r₂ r₃ n > M) ∧
  abs r₂ < 2 + ε :=
by sorry

end NUMINAMATH_CALUDE_r₂_bound_r₂_bound_tight_l2870_287063


namespace NUMINAMATH_CALUDE_route_down_length_l2870_287092

/-- Represents the hiking trip up and down a mountain -/
structure MountainHike where
  rate_up : ℝ
  time : ℝ
  rate_down_factor : ℝ

/-- Calculates the distance of a route given rate and time -/
def route_distance (rate : ℝ) (time : ℝ) : ℝ := rate * time

/-- Theorem: The route down the mountain is 18 miles long -/
theorem route_down_length (hike : MountainHike)
  (h1 : hike.rate_up = 6)
  (h2 : hike.time = 2)
  (h3 : hike.rate_down_factor = 1.5) :
  route_distance (hike.rate_up * hike.rate_down_factor) hike.time = 18 := by
  sorry

#check route_down_length

end NUMINAMATH_CALUDE_route_down_length_l2870_287092


namespace NUMINAMATH_CALUDE_great_great_grandmother_age_calculation_l2870_287015

-- Define the ages of family members
def darcie_age : ℚ := 4
def mother_age : ℚ := darcie_age * 6
def grandmother_age : ℚ := mother_age * (5/4)
def great_grandfather_age : ℚ := grandmother_age * (4/3)
def great_great_grandmother_age : ℚ := great_grandfather_age * (10/7)

-- Theorem statement
theorem great_great_grandmother_age_calculation :
  great_great_grandmother_age = 400/7 := by
  sorry

end NUMINAMATH_CALUDE_great_great_grandmother_age_calculation_l2870_287015


namespace NUMINAMATH_CALUDE_tenth_day_is_monday_l2870_287076

/-- Represents the days of the week -/
inductive DayOfWeek
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday
| Sunday

/-- Represents a month with its starting day and number of days -/
structure Month where
  startDay : DayOfWeek
  numDays : Nat

/-- Represents Teacher Zhang's running schedule -/
def runningDays : List DayOfWeek := [DayOfWeek.Monday, DayOfWeek.Saturday, DayOfWeek.Sunday]

/-- Calculate the day of the week for a given day in the month -/
def dayOfWeek (m : Month) (day : Nat) : DayOfWeek :=
  sorry

/-- The total running time in a month in minutes -/
def totalRunningTime : Nat := 5 * 60

/-- The theorem to be proved -/
theorem tenth_day_is_monday (m : Month) 
  (h1 : m.startDay = DayOfWeek.Saturday) 
  (h2 : m.numDays = 31) 
  (h3 : totalRunningTime = 5 * 60) : 
  dayOfWeek m 10 = DayOfWeek.Monday :=
sorry

end NUMINAMATH_CALUDE_tenth_day_is_monday_l2870_287076


namespace NUMINAMATH_CALUDE_parallel_line_k_value_l2870_287058

/-- Given a line passing through (3, -5) and (k, 21) that is parallel to 4x - 5y = 20, prove k = 35.5 -/
theorem parallel_line_k_value (k : ℝ) :
  (∃ (m b : ℝ), (∀ x y : ℝ, y = m * x + b ↔ (x = 3 ∧ y = -5) ∨ (x = k ∧ y = 21)) ∧
                 (∀ x y : ℝ, y = (4/5) * x - 4 ↔ 4*x - 5*y = 20)) →
  k = 35.5 := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_k_value_l2870_287058


namespace NUMINAMATH_CALUDE_glass_volume_l2870_287050

/-- Given a bottle and a glass, proves that the volume of the glass is 0.5 L 
    when water is poured from a full 1.5 L bottle into an empty glass 
    until both are 3/4 full. -/
theorem glass_volume (bottle_initial : ℝ) (glass : ℝ) : 
  bottle_initial = 1.5 →
  (3/4) * bottle_initial + (3/4) * glass = bottle_initial →
  glass = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_glass_volume_l2870_287050


namespace NUMINAMATH_CALUDE_group_size_calculation_l2870_287017

theorem group_size_calculation (average_increase : ℝ) (original_weight : ℝ) (new_weight : ℝ) :
  average_increase = 3.5 →
  original_weight = 75 →
  new_weight = 99.5 →
  (new_weight - original_weight) / average_increase = 7 := by
sorry

end NUMINAMATH_CALUDE_group_size_calculation_l2870_287017


namespace NUMINAMATH_CALUDE_triangle_side_length_l2870_287078

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  (1/2 * a * c * Real.sin B = Real.sqrt 3) →  -- Area of triangle is √3
  (B = Real.pi / 3) →  -- B = 60°
  (a^2 + c^2 = 3 * a * c) →  -- Given condition
  (b^2 = 8) →  -- Equivalent to b = 2√2
  b = 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2870_287078


namespace NUMINAMATH_CALUDE_quadratic_unique_solution_l2870_287016

theorem quadratic_unique_solution (a c : ℝ) : 
  (∃! x, a * x^2 + 18 * x + c = 0) →  -- exactly one solution
  (a + c = 26) →                      -- sum condition
  (a < c) →                           -- order condition
  (a = 13 + 2 * Real.sqrt 22 ∧ c = 13 - 2 * Real.sqrt 22) := by
sorry

end NUMINAMATH_CALUDE_quadratic_unique_solution_l2870_287016


namespace NUMINAMATH_CALUDE_mean_problem_l2870_287049

theorem mean_problem (x : ℝ) : 
  (28 + x + 42 + 78 + 104) / 5 = 90 →
  (128 + 255 + 511 + 1023 + x) / 5 = 423 := by
sorry

end NUMINAMATH_CALUDE_mean_problem_l2870_287049


namespace NUMINAMATH_CALUDE_average_sale_is_3500_l2870_287080

def sales : List ℕ := [3435, 3920, 3855, 4230, 3560, 2000]

theorem average_sale_is_3500 : 
  (sales.sum / sales.length : ℚ) = 3500 := by
  sorry

end NUMINAMATH_CALUDE_average_sale_is_3500_l2870_287080


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2870_287067

def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 0}
def B : Set ℝ := {x | 0 < x ∧ x ≤ 3}

theorem union_of_A_and_B : A ∪ B = {x : ℝ | -2 ≤ x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2870_287067


namespace NUMINAMATH_CALUDE_daughters_age_l2870_287028

theorem daughters_age (father_age : ℕ) (daughter_age : ℕ) : 
  father_age = 40 → 
  father_age = 4 * daughter_age → 
  father_age + 20 = 2 * (daughter_age + 20) → 
  daughter_age = 10 := by
sorry

end NUMINAMATH_CALUDE_daughters_age_l2870_287028


namespace NUMINAMATH_CALUDE_num_boolean_structures_l2870_287086

/-- The transformation group of 3 Boolean variables -/
def TransformationGroup : Type := Fin 6

/-- The state configurations for 3 Boolean variables -/
def StateConfigurations : Type := Fin 8

/-- The number of colors (Boolean states) -/
def NumColors : Nat := 2

/-- A permutation on the state configurations -/
def Permutation : Type := StateConfigurations → StateConfigurations

/-- The group of permutations induced by the transformation group -/
def PermutationGroup : Type := TransformationGroup → Permutation

/-- Count the number of cycles in a permutation -/
def cycleCount (p : Permutation) : Nat :=
  sorry

/-- Pólya's Enumeration Theorem for this specific case -/
def polyaEnumeration (G : PermutationGroup) : Nat :=
  sorry

/-- The main theorem: number of different structures for a Boolean function device with 3 variables -/
theorem num_boolean_structures (G : PermutationGroup) : 
  polyaEnumeration G = 80 :=
sorry

end NUMINAMATH_CALUDE_num_boolean_structures_l2870_287086


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l2870_287085

theorem rectangle_perimeter (a b : ℝ) :
  let area := 3 * a^2 - 3 * a * b + 6 * a
  let side1 := 3 * a
  let side2 := area / side1
  side1 > 0 → side2 > 0 →
  2 * (side1 + side2) = 8 * a - 2 * b + 4 := by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l2870_287085


namespace NUMINAMATH_CALUDE_f_nonnegative_implies_a_bound_f_inequality_l2870_287096

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x + a

theorem f_nonnegative_implies_a_bound (a : ℝ) :
  (∀ x > 0, f a x ≥ 0) → a ≥ 1 / Real.exp 1 := by sorry

theorem f_inequality (a : ℝ) (x₁ x₂ x : ℝ) 
  (h₁ : 0 < x₁) (h₂ : x₁ < x₂) (h : x₁ < x ∧ x < x₂) :
  (f a x - f a x₁) / (x - x₁) < (f a x - f a x₂) / (x - x₂) := by sorry

end NUMINAMATH_CALUDE_f_nonnegative_implies_a_bound_f_inequality_l2870_287096


namespace NUMINAMATH_CALUDE_power_of_power_l2870_287020

theorem power_of_power (a : ℝ) : (a^4)^4 = a^16 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l2870_287020


namespace NUMINAMATH_CALUDE_taxi_fare_calculation_l2870_287014

/-- Represents the taxi fare structure and proves the cost for a 100-mile ride -/
theorem taxi_fare_calculation (base_fare : ℝ) (rate : ℝ) 
  (h1 : base_fare = 10)
  (h2 : base_fare + 80 * rate = 150) :
  base_fare + 100 * rate = 185 := by
  sorry

#check taxi_fare_calculation

end NUMINAMATH_CALUDE_taxi_fare_calculation_l2870_287014


namespace NUMINAMATH_CALUDE_special_function_values_l2870_287011

/-- A function satisfying f(x + y) = 2 f(x) f(y) for all real x and y -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = 2 * f x * f y

/-- Theorem stating the possible values of f(1) for a SpecialFunction -/
theorem special_function_values (f : ℝ → ℝ) (hf : SpecialFunction f) :
  f 1 = 0 ∨ ∃ r : ℝ, f 1 = r :=
by sorry

end NUMINAMATH_CALUDE_special_function_values_l2870_287011


namespace NUMINAMATH_CALUDE_max_value_of_f_l2870_287023

-- Define the function
def f (x : ℝ) : ℝ := -4 * x^2 + 12 * x + 1

-- State the theorem
theorem max_value_of_f :
  ∃ (m : ℝ), (∀ (x : ℝ), f x ≤ m) ∧ (∃ (x : ℝ), f x = m) ∧ m = 10 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2870_287023


namespace NUMINAMATH_CALUDE_parabola_circle_intersection_l2870_287021

/-- Theorem: For a parabola y = x^2 - ax - 3 (a ∈ ℝ) intersecting the x-axis at points A and B,
    and passing through point C(0, -3), if a circle passing through A, B, and C intersects
    the y-axis at point D(0, b), then b = 1. -/
theorem parabola_circle_intersection (a : ℝ) (A B : ℝ × ℝ) (b : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^2 - a*x - 3
  (f A.1 = 0 ∧ f B.1 = 0) →  -- A and B are on the x-axis
  (∃ D E F : ℝ, (D^2 + E^2 - 4*F > 0) ∧  -- Circle equation coefficients
    (∀ x y : ℝ, x^2 + y^2 + D*x + E*y + F = 0 ↔
      ((x = A.1 ∧ y = A.2) ∨ (x = B.1 ∧ y = B.2) ∨ (x = 0 ∧ y = -3) ∨ (x = 0 ∧ y = b)))) →
  b = 1 :=
by sorry

end NUMINAMATH_CALUDE_parabola_circle_intersection_l2870_287021


namespace NUMINAMATH_CALUDE_purse_wallet_cost_difference_l2870_287047

theorem purse_wallet_cost_difference (wallet_cost purse_cost : ℕ) : 
  wallet_cost = 22 →
  purse_cost < 4 * wallet_cost →
  wallet_cost + purse_cost = 107 →
  4 * wallet_cost - purse_cost = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_purse_wallet_cost_difference_l2870_287047


namespace NUMINAMATH_CALUDE_equation_solution_l2870_287084

theorem equation_solution : 
  ∀ x : ℂ, (5 * x^2 - 3 * x + 2) / (x + 2) = 2 * x - 4 ↔ 
  x = (3 + Complex.I * Real.sqrt 111) / 6 ∨ x = (3 - Complex.I * Real.sqrt 111) / 6 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2870_287084


namespace NUMINAMATH_CALUDE_elliptical_cone_theorem_l2870_287066

/-- Given a cone with a 30° aperture and an elliptical base, 
    prove that the square of the minor axis of the ellipse 
    is equal to the product of the shortest and longest slant heights of the cone. -/
theorem elliptical_cone_theorem (b : ℝ) (AC BC : ℝ) : 
  b > 0 → AC > 0 → BC > 0 → (2 * b)^2 = AC * BC := by
  sorry

end NUMINAMATH_CALUDE_elliptical_cone_theorem_l2870_287066


namespace NUMINAMATH_CALUDE_train_cars_distribution_l2870_287002

theorem train_cars_distribution (soldiers_train1 soldiers_train2 soldiers_train3 : ℕ) 
  (h1 : soldiers_train1 = 462)
  (h2 : soldiers_train2 = 546)
  (h3 : soldiers_train3 = 630) :
  let max_soldiers_per_car := Nat.gcd soldiers_train1 (Nat.gcd soldiers_train2 soldiers_train3)
  (cars_train1, cars_train2, cars_train3) = (soldiers_train1 / max_soldiers_per_car,
                                             soldiers_train2 / max_soldiers_per_car,
                                             soldiers_train3 / max_soldiers_per_car) →
  (cars_train1, cars_train2, cars_train3) = (11, 13, 15) := by
sorry

end NUMINAMATH_CALUDE_train_cars_distribution_l2870_287002


namespace NUMINAMATH_CALUDE_angle_B_measure_l2870_287024

-- Define the triangles and angles
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ

-- Define congruence between triangles
def congruent (t1 t2 : Triangle) : Prop :=
  t1.A = t2.A ∧ t1.B = t2.B ∧ t1.C = t2.C

-- Theorem statement
theorem angle_B_measure (ABC DEF : Triangle) :
  congruent ABC DEF →
  ABC.A = 30 →
  DEF.C = 85 →
  ABC.B = 65 :=
by
  sorry

end NUMINAMATH_CALUDE_angle_B_measure_l2870_287024


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_l2870_287098

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (heq : x + 2*y + 2*x*y = 8) :
  ∀ z, z = x + 2*y → z ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_2y_l2870_287098


namespace NUMINAMATH_CALUDE_original_cat_count_l2870_287044

theorem original_cat_count (first_relocation second_relocation final_count : ℕ) 
  (h1 : first_relocation = 600)
  (h2 : second_relocation = (original_count - first_relocation) / 2)
  (h3 : final_count = 600)
  (h4 : final_count = original_count - first_relocation - second_relocation) :
  original_count = 1800 :=
by sorry

#check original_cat_count

end NUMINAMATH_CALUDE_original_cat_count_l2870_287044


namespace NUMINAMATH_CALUDE_polygon_sides_l2870_287000

/-- Theorem: A polygon with 1080° as the sum of its interior angles has 8 sides. -/
theorem polygon_sides (n : ℕ) : (180 * (n - 2) = 1080) → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l2870_287000


namespace NUMINAMATH_CALUDE_blonde_girls_count_l2870_287005

/-- Represents the choir composition -/
structure Choir :=
  (initial_total : ℕ)
  (added_blonde : ℕ)
  (black_haired : ℕ)

/-- Calculates the initial number of blonde-haired girls in the choir -/
def initial_blonde (c : Choir) : ℕ :=
  c.initial_total - c.black_haired

/-- Theorem stating the initial number of blonde-haired girls in the specific choir -/
theorem blonde_girls_count (c : Choir) 
  (h1 : c.initial_total = 80)
  (h2 : c.added_blonde = 10)
  (h3 : c.black_haired = 50) :
  initial_blonde c = 30 := by
  sorry

end NUMINAMATH_CALUDE_blonde_girls_count_l2870_287005


namespace NUMINAMATH_CALUDE_triangle_properties_l2870_287009

/-- Triangle ABC with given properties -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem stating properties of the specific triangle -/
theorem triangle_properties (t : Triangle) 
  (h1 : t.a = 2)
  (h2 : t.B = π/3)
  (h3 : Real.cos t.A = 2 * Real.sqrt 7 / 7) :
  t.c = 3 ∧ (1/2 * t.a * t.c * Real.sin t.B = 3 * Real.sqrt 3 / 2) := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l2870_287009


namespace NUMINAMATH_CALUDE_middle_number_theorem_l2870_287042

theorem middle_number_theorem (x y z : ℤ) 
  (h_order : x < y ∧ y < z)
  (h_sum1 : x + y = 10)
  (h_sum2 : x + z = 21)
  (h_sum3 : y + z = 25) :
  y = 7 := by sorry

end NUMINAMATH_CALUDE_middle_number_theorem_l2870_287042


namespace NUMINAMATH_CALUDE_exists_integer_sqrt_8m_l2870_287043

theorem exists_integer_sqrt_8m : ∃ m : ℕ+, ∃ k : ℕ, (8 * m.val : ℕ) = k^2 := by
  sorry

end NUMINAMATH_CALUDE_exists_integer_sqrt_8m_l2870_287043


namespace NUMINAMATH_CALUDE_susan_money_l2870_287082

theorem susan_money (S : ℝ) : 
  S - S/5 - S/4 - 120 = 540 → S = 1200 := by
  sorry

end NUMINAMATH_CALUDE_susan_money_l2870_287082


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l2870_287039

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a (n + 1) * a m = a n * a (m + 1)

/-- The problem statement -/
theorem geometric_sequence_problem (a : ℕ → ℝ) 
    (h_geo : GeometricSequence a) 
    (h_a5 : a 5 = 2) 
    (h_a7 : a 7 = 8) : 
    a 6 = 4 ∨ a 6 = -4 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_problem_l2870_287039


namespace NUMINAMATH_CALUDE_june_has_greatest_difference_l2870_287074

/-- Sales data for drummers and bugle players for each month -/
def sales_data : List (ℕ × ℕ) := [
  (8, 5),   -- January
  (10, 5),  -- February
  (8, 8),   -- March
  (4, 8),   -- April
  (5, 10),  -- May
  (3, 9)    -- June
]

/-- Calculate the percentage difference between two numbers -/
def percentage_difference (a b : ℕ) : ℚ :=
  (max a b - min a b : ℚ) / (min a b : ℚ) * 100

/-- Find the month with the greatest percentage difference -/
def month_with_greatest_difference : ℕ :=
  let differences := sales_data.map (fun (d, b) => percentage_difference d b)
  differences.indexOf (differences.foldl max 0)

theorem june_has_greatest_difference :
  month_with_greatest_difference = 5 := by
  sorry


end NUMINAMATH_CALUDE_june_has_greatest_difference_l2870_287074


namespace NUMINAMATH_CALUDE_cards_lost_l2870_287059

theorem cards_lost (initial_cards remaining_cards : ℕ) : 
  initial_cards = 88 → remaining_cards = 18 → initial_cards - remaining_cards = 70 := by
  sorry

end NUMINAMATH_CALUDE_cards_lost_l2870_287059


namespace NUMINAMATH_CALUDE_opposite_numbers_sum_l2870_287073

theorem opposite_numbers_sum (a b : ℝ) : a + b = 0 → 3*a + 3*b - 2 = -2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_numbers_sum_l2870_287073


namespace NUMINAMATH_CALUDE_coffee_blend_type_A_quantity_l2870_287093

/-- Represents the cost and quantity of coffee types in Amanda's Coffee Shop blend --/
structure CoffeeBlend where
  typeA_cost : ℝ
  typeB_cost : ℝ
  typeA_quantity : ℝ
  typeB_quantity : ℝ
  total_cost : ℝ

/-- Theorem stating the quantity of type A coffee in the blend --/
theorem coffee_blend_type_A_quantity (blend : CoffeeBlend) 
  (h1 : blend.typeA_cost = 4.60)
  (h2 : blend.typeB_cost = 5.95)
  (h3 : blend.typeB_quantity = 2 * blend.typeA_quantity)
  (h4 : blend.total_cost = 511.50)
  (h5 : blend.total_cost = blend.typeA_cost * blend.typeA_quantity + blend.typeB_cost * blend.typeB_quantity) :
  blend.typeA_quantity = 31 := by
  sorry


end NUMINAMATH_CALUDE_coffee_blend_type_A_quantity_l2870_287093


namespace NUMINAMATH_CALUDE_jessica_purchase_cost_l2870_287027

def chocolate_bars : ℕ := 10
def gummy_bears : ℕ := 10
def chocolate_chips : ℕ := 20

def price_chocolate_bar : ℕ := 3
def price_gummy_bears : ℕ := 2
def price_chocolate_chips : ℕ := 5

def total_cost : ℕ := chocolate_bars * price_chocolate_bar +
                      gummy_bears * price_gummy_bears +
                      chocolate_chips * price_chocolate_chips

theorem jessica_purchase_cost : total_cost = 150 := by
  sorry

end NUMINAMATH_CALUDE_jessica_purchase_cost_l2870_287027


namespace NUMINAMATH_CALUDE_five_Z_three_equals_nineteen_l2870_287007

-- Define the operation Z
def Z (x y : ℝ) : ℝ := x^2 - x*y + y^2

-- Theorem statement
theorem five_Z_three_equals_nineteen : Z 5 3 = 19 := by
  sorry

end NUMINAMATH_CALUDE_five_Z_three_equals_nineteen_l2870_287007


namespace NUMINAMATH_CALUDE_symmetry_point_l2870_287040

/-- A point in a 2D Cartesian coordinate system -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Symmetry with respect to the y-axis -/
def symmetricToYAxis (p q : Point2D) : Prop :=
  q.x = -p.x ∧ q.y = p.y

theorem symmetry_point : 
  let M : Point2D := ⟨3, -4⟩
  let N : Point2D := ⟨-3, -4⟩
  symmetricToYAxis M N → N = ⟨-3, -4⟩ := by
  sorry

end NUMINAMATH_CALUDE_symmetry_point_l2870_287040
