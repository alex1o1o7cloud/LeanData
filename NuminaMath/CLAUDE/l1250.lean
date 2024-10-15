import Mathlib

namespace NUMINAMATH_CALUDE_trapezoid_area_is_correct_l1250_125024

/-- The area of a trapezoid bounded by y = 2x, y = 6, y = 3, and the y-axis -/
def trapezoidArea : ℝ := 6.75

/-- The line y = 2x -/
def line1 (x : ℝ) : ℝ := 2 * x

/-- The line y = 6 -/
def line2 : ℝ := 6

/-- The line y = 3 -/
def line3 : ℝ := 3

/-- The y-axis (x = 0) -/
def yAxis : ℝ := 0

theorem trapezoid_area_is_correct :
  trapezoidArea = 6.75 := by sorry

end NUMINAMATH_CALUDE_trapezoid_area_is_correct_l1250_125024


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1250_125063

/-- A geometric sequence with positive first term and increasing terms -/
def IncreasingGeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  a 1 > 0 ∧ q > 1 ∧ ∀ n, a (n + 1) = q * a n

/-- The relation between consecutive terms in the sequence -/
def SequenceRelation (a : ℕ → ℝ) : Prop :=
  ∀ n, 2 * (a n + a (n + 2)) = 5 * a (n + 1)

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ) (q : ℝ)
  (h_increasing : IncreasingGeometricSequence a q)
  (h_relation : SequenceRelation a) :
  q = 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1250_125063


namespace NUMINAMATH_CALUDE_max_score_theorem_l1250_125007

/-- Represents a pile of stones -/
structure Pile :=
  (stones : ℕ)

/-- Represents the game state -/
structure GameState :=
  (piles : List Pile)
  (score : ℕ)

/-- Defines a move in the game -/
def move (state : GameState) (i j : ℕ) : GameState :=
  sorry

/-- Checks if the game is over (all stones removed) -/
def isGameOver (state : GameState) : Bool :=
  sorry

/-- Calculates the maximum score achievable from a given state -/
def maxScore (state : GameState) : ℕ :=
  sorry

/-- The main theorem stating the maximum achievable score -/
theorem max_score_theorem :
  let initialState : GameState := ⟨List.replicate 100 ⟨400⟩, 0⟩
  maxScore initialState = 3920000 := by
  sorry

end NUMINAMATH_CALUDE_max_score_theorem_l1250_125007


namespace NUMINAMATH_CALUDE_max_product_constraint_l1250_125014

theorem max_product_constraint (a b : ℝ) : 
  a > 0 → b > 0 → a + b = 2 → (∀ x y : ℝ, x > 0 → y > 0 → x + y = 2 → a * b ≥ x * y) → a * b = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_product_constraint_l1250_125014


namespace NUMINAMATH_CALUDE_prob_fewer_tails_eight_coins_l1250_125046

/-- The number of coins flipped -/
def n : ℕ := 8

/-- The probability of getting fewer tails than heads when flipping n coins -/
def prob_fewer_tails (n : ℕ) : ℚ :=
  (1 - (n.choose (n / 2) : ℚ) / 2^n) / 2

theorem prob_fewer_tails_eight_coins : 
  prob_fewer_tails n = 93 / 256 := by
  sorry

end NUMINAMATH_CALUDE_prob_fewer_tails_eight_coins_l1250_125046


namespace NUMINAMATH_CALUDE_interest_rate_problem_l1250_125084

theorem interest_rate_problem (R T : ℝ) : 
  900 * (1 + R * T / 100) = 956 ∧
  900 * (1 + (R + 4) * T / 100) = 1064 →
  T = 3 := by
sorry

end NUMINAMATH_CALUDE_interest_rate_problem_l1250_125084


namespace NUMINAMATH_CALUDE_meaningful_fraction_range_l1250_125022

theorem meaningful_fraction_range (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x - 2)) ↔ x ≠ 2 := by
  sorry

end NUMINAMATH_CALUDE_meaningful_fraction_range_l1250_125022


namespace NUMINAMATH_CALUDE_spadesuit_calculation_l1250_125044

-- Define the spadesuit operation
def spadesuit (a b : ℝ) : ℝ := |a - b|

-- Theorem statement
theorem spadesuit_calculation :
  (spadesuit (spadesuit 5 2) (spadesuit 9 (spadesuit 3 6))) = 3 := by
  sorry

end NUMINAMATH_CALUDE_spadesuit_calculation_l1250_125044


namespace NUMINAMATH_CALUDE_equal_angle_measure_l1250_125041

-- Define the structure of our shape
structure RectangleTriangleConfig where
  -- Rectangle properties
  rect_length : ℝ
  rect_height : ℝ
  rect_height_lt_length : rect_height < rect_length

  -- Triangle properties
  triangle_base : ℝ
  triangle_leg : ℝ

  -- Shared side property
  shared_side_eq_rect_length : triangle_base = rect_length

  -- Isosceles triangle property
  isosceles_triangle : triangle_base = 2 * triangle_leg

-- Theorem statement
theorem equal_angle_measure (config : RectangleTriangleConfig) :
  let angle := Real.arccos (config.triangle_leg / config.triangle_base) * (180 / Real.pi)
  angle = 45 := by sorry

end NUMINAMATH_CALUDE_equal_angle_measure_l1250_125041


namespace NUMINAMATH_CALUDE_graph_single_point_implies_d_eq_21_l1250_125094

/-- The equation of the graph -/
def graph_equation (x y d : ℝ) : Prop :=
  3 * x^2 + y^2 + 12 * x - 6 * y + d = 0

/-- The condition that the graph consists of a single point -/
def is_single_point (d : ℝ) : Prop :=
  ∃! (x y : ℝ), graph_equation x y d

/-- Theorem: If the graph consists of a single point, then d = 21 -/
theorem graph_single_point_implies_d_eq_21 :
  ∀ d : ℝ, is_single_point d → d = 21 := by
  sorry

end NUMINAMATH_CALUDE_graph_single_point_implies_d_eq_21_l1250_125094


namespace NUMINAMATH_CALUDE_water_one_fifth_after_three_pourings_l1250_125040

def water_remaining (n : ℕ) : ℚ :=
  1 / (2 * n - 1)

theorem water_one_fifth_after_three_pourings :
  water_remaining 3 = 1 / 5 := by
  sorry

#check water_one_fifth_after_three_pourings

end NUMINAMATH_CALUDE_water_one_fifth_after_three_pourings_l1250_125040


namespace NUMINAMATH_CALUDE_percent_of_percent_l1250_125034

theorem percent_of_percent (y : ℝ) (hy : y ≠ 0) :
  (0.6 * 0.3 * y) / y = 0.18 := by sorry

end NUMINAMATH_CALUDE_percent_of_percent_l1250_125034


namespace NUMINAMATH_CALUDE_extreme_values_of_f_l1250_125010

def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x + 5

theorem extreme_values_of_f :
  (∃ x : ℝ, f x = 10 ∧ ∀ y : ℝ, f y ≤ f x) ∧
  (∃ x : ℝ, f x = -22 ∧ ∀ y : ℝ, f y ≥ f x) :=
sorry

end NUMINAMATH_CALUDE_extreme_values_of_f_l1250_125010


namespace NUMINAMATH_CALUDE_largest_ball_radius_l1250_125060

/-- The radius of the torus circle -/
def torus_radius : ℝ := 2

/-- The x-coordinate of the torus circle center -/
def torus_center_x : ℝ := 4

/-- The z-coordinate of the torus circle center -/
def torus_center_z : ℝ := 1

/-- The theorem stating that the radius of the largest spherical ball that can sit on top of the center of the torus and touch the horizontal plane is 4 -/
theorem largest_ball_radius : 
  ∃ (r : ℝ), r = 4 ∧ 
  (torus_center_x ^ 2 + (r - torus_center_z) ^ 2 = (r + torus_radius) ^ 2) ∧
  r > 0 :=
sorry

end NUMINAMATH_CALUDE_largest_ball_radius_l1250_125060


namespace NUMINAMATH_CALUDE_prudence_sleep_hours_l1250_125065

/-- The number of hours Prudence sleeps per night from Sunday to Thursday -/
def sleepHoursSundayToThursday : ℝ := 6

/-- The number of hours Prudence sleeps on Friday and Saturday nights -/
def sleepHoursFridaySaturday : ℝ := 9

/-- The number of hours Prudence naps on Saturday and Sunday -/
def napHours : ℝ := 1

/-- The total number of hours Prudence sleeps in 4 weeks -/
def totalSleepHours : ℝ := 200

/-- The number of weeks -/
def numWeeks : ℝ := 4

/-- The number of nights from Sunday to Thursday -/
def nightsSundayToThursday : ℝ := 5

/-- The number of nights for Friday and Saturday -/
def nightsFridaySaturday : ℝ := 2

/-- The number of nap days (Saturday and Sunday) -/
def napDays : ℝ := 2

theorem prudence_sleep_hours :
  sleepHoursSundayToThursday * nightsSundayToThursday +
  sleepHoursFridaySaturday * nightsFridaySaturday +
  napHours * napDays * numWeeks = totalSleepHours :=
by sorry

end NUMINAMATH_CALUDE_prudence_sleep_hours_l1250_125065


namespace NUMINAMATH_CALUDE_polygon_angles_l1250_125021

theorem polygon_angles (n : ℕ) (h : n > 2) :
  (180 * (n - 2) : ℝ) = 3 * 360 →
  n = 8 ∧ (180 * (n - 2) : ℝ) = 1080 := by
  sorry

end NUMINAMATH_CALUDE_polygon_angles_l1250_125021


namespace NUMINAMATH_CALUDE_quadratic_root_reciprocal_l1250_125057

/-- If m is a root of ax² + bx + 1 = 0, then 1/m is a root of x² + bx + a = 0 -/
theorem quadratic_root_reciprocal (a b m : ℝ) (hm : m ≠ 0) 
  (h : a * m^2 + b * m + 1 = 0) : 
  (1/m)^2 + b * (1/m) + a = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_reciprocal_l1250_125057


namespace NUMINAMATH_CALUDE_article_sale_price_l1250_125078

/-- Given an article with unknown cost price, prove that the selling price
    incurring a loss equal to the profit at $852 is $448, given that the
    selling price for a 50% profit is $975. -/
theorem article_sale_price (cost : ℝ) (loss_price : ℝ) : 
  (852 - cost = cost - loss_price) →  -- Profit at $852 equals loss at loss_price
  (cost + 0.5 * cost = 975) →         -- 50% profit price is $975
  loss_price = 448 := by
  sorry

end NUMINAMATH_CALUDE_article_sale_price_l1250_125078


namespace NUMINAMATH_CALUDE_f_simplification_f_value_when_cos_eq_one_fifth_f_value_at_negative_1860_degrees_l1250_125093

noncomputable section

open Real

def f (α : ℝ) : ℝ := (sin (π - α) * cos (2 * π - α) * tan (-α - π)) / (tan (-α) * sin (-π - α))

theorem f_simplification (α : ℝ) (h : π < α ∧ α < 3 * π / 2) : f α = cos α := by
  sorry

theorem f_value_when_cos_eq_one_fifth (α : ℝ) (h1 : π < α ∧ α < 3 * π / 2) (h2 : cos (α - 3 * π / 2) = 1 / 5) :
  f α = -2 * Real.sqrt 6 / 5 := by
  sorry

theorem f_value_at_negative_1860_degrees :
  f (-1860 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_f_simplification_f_value_when_cos_eq_one_fifth_f_value_at_negative_1860_degrees_l1250_125093


namespace NUMINAMATH_CALUDE_mad_hatter_waiting_time_l1250_125035

/-- Represents a clock with a rate different from real time -/
structure AdjustedClock where
  rate : ℚ  -- Rate of the clock compared to real time

/-- The Mad Hatter's clock -/
def madHatterClock : AdjustedClock :=
  { rate := 75 / 60 }

/-- The March Hare's clock -/
def marchHareClock : AdjustedClock :=
  { rate := 50 / 60 }

/-- Calculates the real time passed for a given clock time -/
def realTimePassed (clock : AdjustedClock) (clockTime : ℚ) : ℚ :=
  clockTime / clock.rate

/-- The agreed meeting time on their clocks -/
def meetingTime : ℚ := 5

theorem mad_hatter_waiting_time :
  realTimePassed madHatterClock meetingTime + 2 = realTimePassed marchHareClock meetingTime :=
by sorry

end NUMINAMATH_CALUDE_mad_hatter_waiting_time_l1250_125035


namespace NUMINAMATH_CALUDE_can_calculate_average_if_complete_info_cannot_calculate_camerons_average_l1250_125081

/-- Represents a tour guide's daily work --/
structure TourGuideDay where
  numTours : Nat
  totalQuestions : Nat
  groupSizes : List Nat

/-- Calculates the average number of questions per tourist --/
def averageQuestionsPerTourist (day : TourGuideDay) : Option ℚ :=
  if day.groupSizes.length = day.numTours ∧ day.groupSizes.sum ≠ 0 then
    some ((day.totalQuestions : ℚ) / (day.groupSizes.sum : ℚ))
  else
    none

/-- Theorem: If we have complete information, we can calculate the average questions per tourist --/
theorem can_calculate_average_if_complete_info (day : TourGuideDay) :
    day.groupSizes.length = day.numTours ∧ day.groupSizes.sum ≠ 0 →
    ∃ avg : ℚ, averageQuestionsPerTourist day = some avg :=
  sorry

/-- Cameron's specific day --/
def cameronsDay : TourGuideDay :=
  { numTours := 4
  , totalQuestions := 68
  , groupSizes := [] }  -- Empty list because we don't know the group sizes

/-- Theorem: We cannot calculate the average for Cameron's day due to missing information --/
theorem cannot_calculate_camerons_average :
    averageQuestionsPerTourist cameronsDay = none :=
  sorry

end NUMINAMATH_CALUDE_can_calculate_average_if_complete_info_cannot_calculate_camerons_average_l1250_125081


namespace NUMINAMATH_CALUDE_skittles_pencils_difference_l1250_125099

def number_of_children : ℕ := 17
def pencils_per_child : ℕ := 3
def skittles_per_child : ℕ := 18

theorem skittles_pencils_difference :
  (number_of_children * skittles_per_child) - (number_of_children * pencils_per_child) = 255 := by
  sorry

end NUMINAMATH_CALUDE_skittles_pencils_difference_l1250_125099


namespace NUMINAMATH_CALUDE_full_servings_count_l1250_125055

-- Define the initial amount of peanut butter
def initial_amount : Rat := 34 + 2/3

-- Define the additional amount of peanut butter
def additional_amount : Rat := 15 + 1/3

-- Define the serving size
def serving_size : Rat := 3

-- Theorem to prove
theorem full_servings_count :
  ⌊(initial_amount + additional_amount) / serving_size⌋ = 16 := by
  sorry

end NUMINAMATH_CALUDE_full_servings_count_l1250_125055


namespace NUMINAMATH_CALUDE_circle_area_isosceles_triangle_l1250_125004

/-- The area of a circle circumscribing an isosceles triangle -/
theorem circle_area_isosceles_triangle (a b : ℝ) (h1 : a = 4) (h2 : b = 3) :
  let r := Real.sqrt ((a^2 / 4 + b^2 / 16))
  π * r^2 = 5.6875 * π := by
sorry

end NUMINAMATH_CALUDE_circle_area_isosceles_triangle_l1250_125004


namespace NUMINAMATH_CALUDE_difference_of_two_greatest_values_l1250_125051

def is_three_digit_integer (x : ℕ) : Prop :=
  100 ≤ x ∧ x ≤ 999

def hundreds_digit (x : ℕ) : ℕ :=
  (x / 100) % 10

def tens_digit (x : ℕ) : ℕ :=
  (x / 10) % 10

def units_digit (x : ℕ) : ℕ :=
  x % 10

def satisfies_conditions (x : ℕ) : Prop :=
  let a := hundreds_digit x
  let b := tens_digit x
  let c := units_digit x
  is_three_digit_integer x ∧ 4 * a = 2 * b ∧ 2 * b = c ∧ a > 0

def two_greatest_values (x y : ℕ) : Prop :=
  satisfies_conditions x ∧ satisfies_conditions y ∧
  ∀ z, satisfies_conditions z → z ≤ x ∧ (z ≠ x → z ≤ y)

theorem difference_of_two_greatest_values :
  ∃ x y, two_greatest_values x y ∧ x - y = 124 :=
sorry

end NUMINAMATH_CALUDE_difference_of_two_greatest_values_l1250_125051


namespace NUMINAMATH_CALUDE_real_part_of_complex_fraction_l1250_125028

theorem real_part_of_complex_fraction (i : ℂ) :
  i * i = -1 →
  Complex.re ((2 : ℂ) + i) / i = 1 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_complex_fraction_l1250_125028


namespace NUMINAMATH_CALUDE_complex_real_condition_l1250_125077

theorem complex_real_condition (m : ℝ) : 
  (Complex.I * Complex.I = -1) →
  ((m^2 + Complex.I) * (1 - m * Complex.I)).im = 0 →
  m = 1 := by sorry

end NUMINAMATH_CALUDE_complex_real_condition_l1250_125077


namespace NUMINAMATH_CALUDE_number_in_scientific_notation_l1250_125008

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The number to be expressed in scientific notation -/
def number : ℕ := 21600

/-- Theorem stating that 21,600 in scientific notation is 2.16 × 10^4 -/
theorem number_in_scientific_notation :
  ∃ (sn : ScientificNotation), (sn.coefficient * (10 : ℝ) ^ sn.exponent = number) ∧
    (sn.coefficient = 2.16 ∧ sn.exponent = 4) :=
sorry

end NUMINAMATH_CALUDE_number_in_scientific_notation_l1250_125008


namespace NUMINAMATH_CALUDE_group_size_calculation_l1250_125025

theorem group_size_calculation (n : ℕ) : 
  (n * 15 + 37 = 17 * (n + 1)) → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_group_size_calculation_l1250_125025


namespace NUMINAMATH_CALUDE_horner_method_example_l1250_125032

def f (x : ℝ) : ℝ := x^6 - 5*x^5 + 6*x^4 + x^2 + 3*x + 2

theorem horner_method_example : f (-2) = 320 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_example_l1250_125032


namespace NUMINAMATH_CALUDE_reflection_line_equation_l1250_125002

-- Define the points of the original triangle
def P : ℝ × ℝ := (3, 2)
def Q : ℝ × ℝ := (8, 7)
def R : ℝ × ℝ := (6, -4)

-- Define the points of the reflected triangle
def P' : ℝ × ℝ := (-5, 2)
def Q' : ℝ × ℝ := (-10, 7)
def R' : ℝ × ℝ := (-8, -4)

-- Define the reflection line
def M : ℝ → Prop := λ x => x = -1

theorem reflection_line_equation :
  (∀ (x y : ℝ), (x, y) = P ∨ (x, y) = Q ∨ (x, y) = R →
    ∃ (x' : ℝ), M x' ∧ x' = (x + P'.1) / 2) ∧
  (∀ (x y : ℝ), (x, y) = P' ∨ (x, y) = Q' ∨ (x, y) = R' →
    ∃ (x' : ℝ), M x' ∧ x' = (x + P.1) / 2) :=
sorry

end NUMINAMATH_CALUDE_reflection_line_equation_l1250_125002


namespace NUMINAMATH_CALUDE_expression_evaluation_l1250_125089

theorem expression_evaluation : 
  let a : ℚ := 5
  let b : ℚ := a + 4
  let c : ℚ := b - 12
  (a + 2 ≠ 0) → (b - 3 ≠ 0) → (c + 7 ≠ 0) →
  ((a + 4) / (a + 2)) * ((b + 1) / (b - 3)) * ((c + 10) / (c + 7)) = 3.75 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1250_125089


namespace NUMINAMATH_CALUDE_billboard_dimensions_l1250_125037

/-- Given a rectangular photograph and a billboard, prove the billboard's dimensions -/
theorem billboard_dimensions 
  (photo_width : ℝ) 
  (photo_length : ℝ) 
  (billboard_area : ℝ) 
  (h1 : photo_width = 30) 
  (h2 : photo_length = 40) 
  (h3 : billboard_area = 48) : 
  ∃ (billboard_width billboard_length : ℝ), 
    billboard_width = 6 ∧ 
    billboard_length = 8 ∧ 
    billboard_width * billboard_length = billboard_area :=
by sorry

end NUMINAMATH_CALUDE_billboard_dimensions_l1250_125037


namespace NUMINAMATH_CALUDE_brown_paint_amount_l1250_125053

def total_paint : ℕ := 69
def white_paint : ℕ := 20
def green_paint : ℕ := 15

theorem brown_paint_amount :
  total_paint - (white_paint + green_paint) = 34 := by
  sorry

end NUMINAMATH_CALUDE_brown_paint_amount_l1250_125053


namespace NUMINAMATH_CALUDE_alicia_remaining_masks_l1250_125079

/-- The number of mask sets remaining in Alicia's collection after donation -/
def remaining_masks (initial : ℕ) (donated : ℕ) : ℕ :=
  initial - donated

/-- Theorem stating that Alicia has 39 mask sets left after donating to the museum -/
theorem alicia_remaining_masks :
  remaining_masks 90 51 = 39 := by
  sorry

end NUMINAMATH_CALUDE_alicia_remaining_masks_l1250_125079


namespace NUMINAMATH_CALUDE_coefficient_x_cubed_in_expansion_l1250_125012

/-- The coefficient of x^3 in the expansion of (1-1/x)(1+x)^5 is 5 -/
theorem coefficient_x_cubed_in_expansion : ∃ (f : ℝ → ℝ),
  (∀ x ≠ 0, f x = (1 - 1/x) * (1 + x)^5) ∧
  (∃ a b c d e g : ℝ, ∀ x ≠ 0, f x = a + b*x + c*x^2 + 5*x^3 + d*x^4 + e*x^5 + g/x) :=
by sorry

end NUMINAMATH_CALUDE_coefficient_x_cubed_in_expansion_l1250_125012


namespace NUMINAMATH_CALUDE_tan_sixty_minus_reciprocal_tan_thirty_equals_zero_l1250_125015

theorem tan_sixty_minus_reciprocal_tan_thirty_equals_zero :
  Real.tan (60 * π / 180) - (1 / Real.tan (30 * π / 180)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_tan_sixty_minus_reciprocal_tan_thirty_equals_zero_l1250_125015


namespace NUMINAMATH_CALUDE_fourth_power_congruence_divisibility_l1250_125029

theorem fourth_power_congruence_divisibility (p a b c d : ℕ) (hp : Prime p) 
  (ha : 0 < a) (hab : a < b) (hbc : b < c) (hcd : c < d) (hdp : d < p)
  (hcong : ∃ k : ℕ, a^4 % p = k ∧ b^4 % p = k ∧ c^4 % p = k ∧ d^4 % p = k) :
  (a + b + c + d) ∣ (a^2013 + b^2013 + c^2013 + d^2013) := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_congruence_divisibility_l1250_125029


namespace NUMINAMATH_CALUDE_papi_calot_plants_l1250_125043

/-- The number of plants Papi Calot needs to buy -/
def total_plants (rows : ℕ) (plants_per_row : ℕ) (additional_plants : ℕ) : ℕ :=
  rows * plants_per_row + additional_plants

/-- Theorem stating the total number of plants Papi Calot needs to buy -/
theorem papi_calot_plants : total_plants 7 18 15 = 141 := by
  sorry

end NUMINAMATH_CALUDE_papi_calot_plants_l1250_125043


namespace NUMINAMATH_CALUDE_train_length_calculation_l1250_125011

/-- Given a train crossing a bridge, calculate its length. -/
theorem train_length_calculation (train_speed : ℝ) (crossing_time : ℝ) (bridge_length : ℝ) :
  train_speed = 45 * (1000 / 3600) →
  crossing_time = 30 →
  bridge_length = 255 →
  ∃ (train_length : ℝ), train_length = train_speed * crossing_time - bridge_length ∧ train_length = 120 := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l1250_125011


namespace NUMINAMATH_CALUDE_first_number_100th_group_l1250_125000

/-- The sequence term at position n -/
def sequenceTerm (n : ℕ) : ℕ := 3^(n - 1)

/-- The sum of the first n natural numbers -/
def triangularNumber (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The position of the first number in the nth group -/
def firstNumberPosition (n : ℕ) : ℕ := triangularNumber (n - 1) + 1

/-- The first number in the nth group -/
def firstNumberInGroup (n : ℕ) : ℕ := sequenceTerm (firstNumberPosition n)

theorem first_number_100th_group :
  firstNumberInGroup 100 = 3^4950 := by sorry

end NUMINAMATH_CALUDE_first_number_100th_group_l1250_125000


namespace NUMINAMATH_CALUDE_iphone_price_calculation_l1250_125082

def calculate_final_price (initial_price : ℝ) 
                          (discount1 : ℝ) (tax1 : ℝ) 
                          (discount2 : ℝ) (tax2 : ℝ) : ℝ :=
  let price_after_discount1 := initial_price * (1 - discount1)
  let price_after_tax1 := price_after_discount1 * (1 + tax1)
  let price_after_discount2 := price_after_tax1 * (1 - discount2)
  let final_price := price_after_discount2 * (1 + tax2)
  final_price

theorem iphone_price_calculation :
  let initial_price : ℝ := 1000
  let discount1 : ℝ := 0.1
  let tax1 : ℝ := 0.08
  let discount2 : ℝ := 0.2
  let tax2 : ℝ := 0.06
  let final_price := calculate_final_price initial_price discount1 tax1 discount2 tax2
  ∃ ε > 0, |final_price - 824.26| < ε :=
sorry

end NUMINAMATH_CALUDE_iphone_price_calculation_l1250_125082


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l1250_125059

-- Problem 1
theorem problem_1 : -3 + 8 - 7 - 15 = -17 := by sorry

-- Problem 2
theorem problem_2 : 23 - 6 * (-3) + 2 * (-4) = 33 := by sorry

-- Problem 3
theorem problem_3 : -8 / (4/5) * (-2/3) = 20/3 := by sorry

-- Problem 4
theorem problem_4 : -(2^2) - 9 * ((-1/3)^2) + |(-4)| = -1 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l1250_125059


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l1250_125087

theorem smallest_n_congruence (n : ℕ) : 
  (n > 0 ∧ 23 * n ≡ 789 [ZMOD 11]) → n ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l1250_125087


namespace NUMINAMATH_CALUDE_largest_two_digit_prime_factor_l1250_125083

/-- The binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- Checks if a number is prime -/
def isPrime (n : ℕ) : Prop := Nat.Prime n

/-- Checks if a number is a two-digit number -/
def isTwoDigit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

/-- Checks if a number is a factor of another number -/
def isFactor (a b : ℕ) : Prop := b % a = 0

theorem largest_two_digit_prime_factor :
  ∃ (p : ℕ), isPrime p ∧ 
             isTwoDigit p ∧ 
             isFactor p (binomial 300 150) ∧
             (∀ (q : ℕ), isPrime q → isTwoDigit q → isFactor q (binomial 300 150) → q ≤ p) ∧
             p = 97 := by sorry

end NUMINAMATH_CALUDE_largest_two_digit_prime_factor_l1250_125083


namespace NUMINAMATH_CALUDE_apple_orange_ratio_l1250_125070

/-- Represents the number of fruits each child received -/
structure FruitDistribution where
  mike_oranges : ℕ
  matt_apples : ℕ
  mark_bananas : ℕ

/-- The fruit distribution satisfies the problem conditions -/
def valid_distribution (d : FruitDistribution) : Prop :=
  d.mike_oranges = 3 ∧
  d.mark_bananas = d.mike_oranges + d.matt_apples ∧
  d.mike_oranges + d.matt_apples + d.mark_bananas = 18

theorem apple_orange_ratio (d : FruitDistribution) 
  (h : valid_distribution d) : 
  d.matt_apples / d.mike_oranges = 2 := by
  sorry

#check apple_orange_ratio

end NUMINAMATH_CALUDE_apple_orange_ratio_l1250_125070


namespace NUMINAMATH_CALUDE_area_of_specific_quadrilateral_l1250_125009

/-- Represents a quadrilateral EFGH with specific angle and side length properties -/
structure Quadrilateral :=
  (EF : ℝ)
  (FG : ℝ)
  (GH : ℝ)
  (angle_F : ℝ)
  (angle_G : ℝ)

/-- Calculates the area of the quadrilateral EFGH -/
def area (q : Quadrilateral) : ℝ :=
  sorry

/-- Theorem stating that for a quadrilateral EFGH with given properties, its area is (77√2)/4 -/
theorem area_of_specific_quadrilateral :
  ∀ (q : Quadrilateral),
    q.EF = 5 ∧
    q.FG = 7 ∧
    q.GH = 6 ∧
    q.angle_F = 135 ∧
    q.angle_G = 135 →
    area q = (77 * Real.sqrt 2) / 4 :=
by sorry

end NUMINAMATH_CALUDE_area_of_specific_quadrilateral_l1250_125009


namespace NUMINAMATH_CALUDE_algebraic_simplification_l1250_125001

theorem algebraic_simplification (a b : ℝ) : 
  (a^3 * b^4)^2 / (a * b^2)^3 = a^3 * b^2 ∧ 
  (-a^2)^3 * a^2 + a^8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_simplification_l1250_125001


namespace NUMINAMATH_CALUDE_total_amount_l1250_125074

/-- The ratio of money distribution among w, x, y, and z -/
structure MoneyDistribution where
  w : ℝ
  x : ℝ
  y : ℝ
  z : ℝ
  ratio_condition : x = 0.7 * w ∧ y = 0.5 * w ∧ z = 0.3 * w

/-- The problem statement -/
theorem total_amount (d : MoneyDistribution) (h : d.y = 90) :
  d.w + d.x + d.y + d.z = 450 := by
  sorry


end NUMINAMATH_CALUDE_total_amount_l1250_125074


namespace NUMINAMATH_CALUDE_bricklayer_problem_l1250_125056

theorem bricklayer_problem (x : ℝ) 
  (h1 : (x / 12 + x / 15 - 15) * 6 = x) : x = 900 := by
  sorry

#check bricklayer_problem

end NUMINAMATH_CALUDE_bricklayer_problem_l1250_125056


namespace NUMINAMATH_CALUDE_ticket_distribution_l1250_125045

/-- The number of ways to distribute 5 consecutive tickets to 5 people. -/
def distribute_tickets : ℕ := 5 * 4 * 3 * 2 * 1

/-- The number of ways for A and B to receive consecutive tickets. -/
def consecutive_for_ab : ℕ := 4 * 2

/-- The number of ways to distribute the remaining tickets to 3 people. -/
def distribute_remaining : ℕ := 3 * 2 * 1

/-- 
Theorem: The number of ways to distribute 5 consecutive movie tickets to 5 people, 
including A and B, such that A and B receive consecutive tickets, is equal to 48.
-/
theorem ticket_distribution : 
  consecutive_for_ab * distribute_remaining = 48 := by
  sorry

end NUMINAMATH_CALUDE_ticket_distribution_l1250_125045


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_l1250_125030

theorem quadratic_equation_solutions (x : ℝ) : x^2 = x ↔ x = 0 ∨ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_l1250_125030


namespace NUMINAMATH_CALUDE_exam_maximum_marks_l1250_125013

/-- 
Given an exam where:
1. 40% of the maximum marks are required to pass
2. A student got 40 marks
3. The student failed by 40 marks
Prove that the maximum marks for the exam are 200.
-/
theorem exam_maximum_marks :
  ∀ (max_marks : ℕ) (pass_percentage : ℚ) (student_marks : ℕ) (fail_margin : ℕ),
    pass_percentage = 40 / 100 →
    student_marks = 40 →
    fail_margin = 40 →
    (pass_percentage * max_marks : ℚ) = student_marks + fail_margin →
    max_marks = 200 := by
  sorry

end NUMINAMATH_CALUDE_exam_maximum_marks_l1250_125013


namespace NUMINAMATH_CALUDE_fraction_meaningful_l1250_125058

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = 3 / (x + 5)) ↔ x ≠ -5 := by sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l1250_125058


namespace NUMINAMATH_CALUDE_choose_two_from_four_eq_six_l1250_125097

def choose_two_from_four : ℕ := sorry

theorem choose_two_from_four_eq_six : choose_two_from_four = 6 := by sorry

end NUMINAMATH_CALUDE_choose_two_from_four_eq_six_l1250_125097


namespace NUMINAMATH_CALUDE_chessboard_circle_area_ratio_l1250_125033

/-- Represents a square chessboard -/
structure Chessboard where
  side_length : ℝ
  dimensions : ℕ × ℕ

/-- Represents a circle placed on the chessboard -/
structure PlacedCircle where
  radius : ℝ

/-- Calculates the sum of areas within the circle for intersected squares -/
def S₁ (board : Chessboard) (circle : PlacedCircle) : ℝ := sorry

/-- Calculates the sum of areas outside the circle for intersected squares -/
def S₂ (board : Chessboard) (circle : PlacedCircle) : ℝ := sorry

/-- The main theorem to be proved -/
theorem chessboard_circle_area_ratio
  (board : Chessboard)
  (circle : PlacedCircle)
  (h_board_side : board.side_length = 8)
  (h_board_dim : board.dimensions = (8, 8))
  (h_circle_radius : circle.radius = 4) :
  Int.floor (S₁ board circle / S₂ board circle) = 3 := by sorry

end NUMINAMATH_CALUDE_chessboard_circle_area_ratio_l1250_125033


namespace NUMINAMATH_CALUDE_jelly_bean_probability_l1250_125076

/-- The probability of selecting either a blue or purple jelly bean from a bag -/
theorem jelly_bean_probability :
  let red : ℕ := 8
  let green : ℕ := 9
  let yellow : ℕ := 10
  let blue : ℕ := 12
  let purple : ℕ := 5
  let total : ℕ := red + green + yellow + blue + purple
  let blue_or_purple : ℕ := blue + purple
  (blue_or_purple : ℚ) / total = 17 / 44 :=
by sorry

end NUMINAMATH_CALUDE_jelly_bean_probability_l1250_125076


namespace NUMINAMATH_CALUDE_prob_draw_3_equals_expected_l1250_125054

-- Define the defect rate
def defect_rate : ℝ := 0.03

-- Define the probability of drawing exactly 3 products
def prob_draw_3 (p : ℝ) : ℝ := p^2 * (1 - p) + p^3

-- Theorem statement
theorem prob_draw_3_equals_expected : 
  prob_draw_3 defect_rate = defect_rate^2 * (1 - defect_rate) + defect_rate^3 :=
by sorry

end NUMINAMATH_CALUDE_prob_draw_3_equals_expected_l1250_125054


namespace NUMINAMATH_CALUDE_max_pairs_correct_max_pairs_achievable_l1250_125047

/-- The maximum number of pairs that can be chosen from the set {1, 2, ..., 3009}
    such that no two pairs have a common element and all sums of pairs are distinct
    and less than or equal to 3009 -/
def max_pairs : ℕ := 1003

theorem max_pairs_correct : 
  ∀ (k : ℕ) (pairs : Finset (ℕ × ℕ)),
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 ∈ Finset.range 3009 ∧ p.2 ∈ Finset.range 3009) →
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 < p.2) →
    (∀ (p q : ℕ × ℕ), p ∈ pairs → q ∈ pairs → p ≠ q → p.1 ≠ q.1 ∧ p.1 ≠ q.2 ∧ p.2 ≠ q.1 ∧ p.2 ≠ q.2) →
    (∀ (p q : ℕ × ℕ), p ∈ pairs → q ∈ pairs → p ≠ q → p.1 + p.2 ≠ q.1 + q.2) →
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 + p.2 ≤ 3009) →
    pairs.card ≤ max_pairs :=
by sorry

theorem max_pairs_achievable :
  ∃ (pairs : Finset (ℕ × ℕ)),
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 ∈ Finset.range 3009 ∧ p.2 ∈ Finset.range 3009) ∧
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 < p.2) ∧
    (∀ (p q : ℕ × ℕ), p ∈ pairs → q ∈ pairs → p ≠ q → p.1 ≠ q.1 ∧ p.1 ≠ q.2 ∧ p.2 ≠ q.1 ∧ p.2 ≠ q.2) ∧
    (∀ (p q : ℕ × ℕ), p ∈ pairs → q ∈ pairs → p ≠ q → p.1 + p.2 ≠ q.1 + q.2) ∧
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 + p.2 ≤ 3009) ∧
    pairs.card = max_pairs :=
by sorry

end NUMINAMATH_CALUDE_max_pairs_correct_max_pairs_achievable_l1250_125047


namespace NUMINAMATH_CALUDE_fifteen_times_thirtysix_plus_fifteen_times_three_cubed_l1250_125062

theorem fifteen_times_thirtysix_plus_fifteen_times_three_cubed : 15 * 36 + 15 * 3^3 = 945 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_times_thirtysix_plus_fifteen_times_three_cubed_l1250_125062


namespace NUMINAMATH_CALUDE_overall_loss_percentage_l1250_125016

def purchase_prices : List ℝ := [600, 800, 1000, 1200, 1400]
def selling_prices : List ℝ := [550, 750, 1100, 1000, 1350]

theorem overall_loss_percentage :
  let total_cost_price := purchase_prices.sum
  let total_selling_price := selling_prices.sum
  let loss := total_cost_price - total_selling_price
  let loss_percentage := (loss / total_cost_price) * 100
  loss_percentage = 5 := by sorry

end NUMINAMATH_CALUDE_overall_loss_percentage_l1250_125016


namespace NUMINAMATH_CALUDE_power_sum_of_i_l1250_125026

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem power_sum_of_i : i^23 + i^223 = -2*i := by sorry

end NUMINAMATH_CALUDE_power_sum_of_i_l1250_125026


namespace NUMINAMATH_CALUDE_first_oil_price_l1250_125020

/-- Given two oils mixed together, prove the price of the first oil. -/
theorem first_oil_price 
  (first_oil_volume : ℝ) 
  (second_oil_volume : ℝ) 
  (second_oil_price : ℝ) 
  (mixture_price : ℝ)
  (h1 : first_oil_volume = 10)
  (h2 : second_oil_volume = 5)
  (h3 : second_oil_price = 66)
  (h4 : mixture_price = 58) :
  ∃ (first_oil_price : ℝ), 
    first_oil_price = 54 ∧ 
    first_oil_price * first_oil_volume + second_oil_price * second_oil_volume = 
      mixture_price * (first_oil_volume + second_oil_volume) := by
  sorry

end NUMINAMATH_CALUDE_first_oil_price_l1250_125020


namespace NUMINAMATH_CALUDE_refrigerator_deposit_l1250_125085

/-- Proves the deposit amount for a refrigerator purchase with installments -/
theorem refrigerator_deposit (cash_price : ℕ) (num_installments : ℕ) (installment_amount : ℕ) (savings : ℕ) : 
  cash_price = 8000 →
  num_installments = 30 →
  installment_amount = 300 →
  savings = 4000 →
  cash_price + savings = num_installments * installment_amount + (cash_price + savings - num_installments * installment_amount) :=
by sorry

end NUMINAMATH_CALUDE_refrigerator_deposit_l1250_125085


namespace NUMINAMATH_CALUDE_damage_proportion_l1250_125095

/-- The proportion of a 3x2 rectangle that can be reached by the midpoint of a 2-unit line segment
    rotating freely within the rectangle -/
theorem damage_proportion (rectangle_length : Real) (rectangle_width : Real) (log_length : Real) :
  rectangle_length = 3 ∧ rectangle_width = 2 ∧ log_length = 2 →
  (rectangle_length * rectangle_width - 4 * (Real.pi / 4 * (log_length / 2)^2)) / (rectangle_length * rectangle_width) = 1 - Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_damage_proportion_l1250_125095


namespace NUMINAMATH_CALUDE_clock_rings_count_l1250_125038

/-- Represents the number of times a clock rings in a day -/
def clock_rings (ring_interval : ℕ) (start_hour : ℕ) (day_length : ℕ) : ℕ :=
  (day_length - start_hour) / ring_interval + 1

/-- Theorem stating that a clock ringing every 3 hours starting at 1 A.M. will ring 8 times in a day -/
theorem clock_rings_count : clock_rings 3 1 24 = 8 := by
  sorry

end NUMINAMATH_CALUDE_clock_rings_count_l1250_125038


namespace NUMINAMATH_CALUDE_jerry_feather_ratio_l1250_125066

def jerryFeatherProblem (hawk_feathers eagle_feathers total_feathers feathers_given feathers_left : ℕ) : Prop :=
  hawk_feathers = 6 ∧
  eagle_feathers = 17 * hawk_feathers ∧
  total_feathers = hawk_feathers + eagle_feathers ∧
  feathers_given = 10 ∧
  feathers_left = 49

theorem jerry_feather_ratio 
  (hawk_feathers eagle_feathers total_feathers feathers_given feathers_left : ℕ) 
  (h : jerryFeatherProblem hawk_feathers eagle_feathers total_feathers feathers_given feathers_left) : 
  ∃ (feathers_after_giving feathers_sold : ℕ),
    feathers_after_giving = total_feathers - feathers_given ∧
    feathers_sold = feathers_after_giving - feathers_left ∧
    2 * feathers_sold = feathers_after_giving :=
sorry

end NUMINAMATH_CALUDE_jerry_feather_ratio_l1250_125066


namespace NUMINAMATH_CALUDE_hyperbola_line_intersection_eccentricity_l1250_125050

/-- The eccentricity of a hyperbola that has a common point with the line y = 2x --/
def eccentricity_range (a b : ℝ) : Prop :=
  let e := Real.sqrt (1 + b^2 / a^2)
  (∃ x y : ℝ, y = 2*x ∧ x^2/a^2 - y^2/b^2 = 1) →
  1 < e ∧ e ≤ Real.sqrt 5

theorem hyperbola_line_intersection_eccentricity :
  ∀ a b : ℝ, a > 0 ∧ b > 0 → eccentricity_range a b :=
sorry

end NUMINAMATH_CALUDE_hyperbola_line_intersection_eccentricity_l1250_125050


namespace NUMINAMATH_CALUDE_unique_solution_is_sqrt_two_l1250_125049

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then 2^(-x) else Real.log x / Real.log 4

-- State the theorem
theorem unique_solution_is_sqrt_two :
  ∃! x, x > 1 ∧ f x = 1/4 ∧ x = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_unique_solution_is_sqrt_two_l1250_125049


namespace NUMINAMATH_CALUDE_coin_flip_expected_value_l1250_125023

def penny : ℚ := 1
def fifty_cent : ℚ := 50
def dime : ℚ := 10
def quarter : ℚ := 25

def coin_probability : ℚ := 1 / 2

def expected_value : ℚ := 
  coin_probability * penny + 
  coin_probability * fifty_cent + 
  coin_probability * dime + 
  coin_probability * quarter

theorem coin_flip_expected_value : expected_value = 43 := by
  sorry

end NUMINAMATH_CALUDE_coin_flip_expected_value_l1250_125023


namespace NUMINAMATH_CALUDE_rectangle_area_l1250_125096

/-- The area of a rectangle with diagonal x and length three times its width -/
theorem rectangle_area (x : ℝ) : 
  ∃ (w l : ℝ), w > 0 ∧ l > 0 ∧ l = 3 * w ∧ w ^ 2 + l ^ 2 = x ^ 2 → w * l = (3 / 10) * x ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l1250_125096


namespace NUMINAMATH_CALUDE_field_area_calculation_l1250_125039

theorem field_area_calculation (smaller_area larger_area : ℝ) : 
  smaller_area = 315 →
  larger_area - smaller_area = (1 / 5) * ((smaller_area + larger_area) / 2) →
  smaller_area + larger_area = 700 := by
  sorry

end NUMINAMATH_CALUDE_field_area_calculation_l1250_125039


namespace NUMINAMATH_CALUDE_composition_difference_l1250_125005

/-- Given two functions f and g, prove that their composition difference
    f(g(x)) - g(f(x)) equals 6x^2 - 12x + 9 for all real x. -/
theorem composition_difference (x : ℝ) : 
  let f (x : ℝ) := 3 * x^2 - 6 * x + 1
  let g (x : ℝ) := 2 * x - 1
  f (g x) - g (f x) = 6 * x^2 - 12 * x + 9 := by
  sorry

end NUMINAMATH_CALUDE_composition_difference_l1250_125005


namespace NUMINAMATH_CALUDE_boat_speed_l1250_125075

theorem boat_speed (v_s : ℝ) (t_d t_u : ℝ) (h1 : v_s = 8) (h2 : t_u = 2 * t_d) : 
  ∃ v_b : ℝ, v_b > 0 ∧ (v_b - v_s) * t_u = (v_b + v_s) * t_d ∧ v_b = 24 :=
by sorry

end NUMINAMATH_CALUDE_boat_speed_l1250_125075


namespace NUMINAMATH_CALUDE_fourth_root_equation_solution_l1250_125027

theorem fourth_root_equation_solution (x : ℝ) (h1 : x > 0) 
  (h2 : (1 - x^4)^(1/4) + (1 + x^4)^(1/4) = 1) : x^8 = 35/36 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_equation_solution_l1250_125027


namespace NUMINAMATH_CALUDE_cookies_given_to_friend_l1250_125017

theorem cookies_given_to_friend (initial_cookies : ℕ) (eaten_cookies : ℕ) (remaining_cookies : ℕ) : 
  initial_cookies = 36 →
  eaten_cookies = 10 →
  remaining_cookies = 12 →
  initial_cookies - eaten_cookies - remaining_cookies = 14 := by
sorry

end NUMINAMATH_CALUDE_cookies_given_to_friend_l1250_125017


namespace NUMINAMATH_CALUDE_truck_loading_time_l1250_125073

theorem truck_loading_time (worker1_time worker2_time combined_time : ℝ) 
  (h1 : worker1_time = 6)
  (h2 : combined_time = 2.4)
  (h3 : 1 / worker1_time + 1 / worker2_time = 1 / combined_time) :
  worker2_time = 4 := by
  sorry

end NUMINAMATH_CALUDE_truck_loading_time_l1250_125073


namespace NUMINAMATH_CALUDE_modulus_of_complex_number_l1250_125018

theorem modulus_of_complex_number : 
  let z : ℂ := (1 - I) / (2 * I + 1) * I
  (∃ (k : ℝ), z = k * I) → Complex.abs z = Real.sqrt 10 / 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_number_l1250_125018


namespace NUMINAMATH_CALUDE_rajas_household_expenditure_percentage_l1250_125031

theorem rajas_household_expenditure_percentage 
  (monthly_income : ℝ) 
  (clothes_percentage : ℝ) 
  (medicines_percentage : ℝ) 
  (savings : ℝ) 
  (h1 : monthly_income = 37500) 
  (h2 : clothes_percentage = 20) 
  (h3 : medicines_percentage = 5) 
  (h4 : savings = 15000) : 
  (monthly_income - (monthly_income * clothes_percentage / 100 + 
   monthly_income * medicines_percentage / 100 + savings)) / monthly_income * 100 = 35 := by
sorry

end NUMINAMATH_CALUDE_rajas_household_expenditure_percentage_l1250_125031


namespace NUMINAMATH_CALUDE_allocation_schemes_l1250_125064

def doctors : ℕ := 2
def nurses : ℕ := 4
def hospitals : ℕ := 2
def doctors_per_hospital : ℕ := 1
def nurses_per_hospital : ℕ := 2

theorem allocation_schemes :
  (Nat.choose doctors hospitals) * (Nat.choose nurses (nurses_per_hospital * hospitals)) = 12 :=
sorry

end NUMINAMATH_CALUDE_allocation_schemes_l1250_125064


namespace NUMINAMATH_CALUDE_roots_relation_l1250_125048

/-- The polynomial h(x) -/
def h (x : ℝ) : ℝ := x^3 - 2*x^2 + 4*x - 1

/-- The polynomial j(x) -/
def j (x p q r : ℝ) : ℝ := x^3 + p*x^2 + q*x + r

/-- Theorem stating the relationship between h(x) and j(x) and the values of p, q, and r -/
theorem roots_relation (p q r : ℝ) : 
  (∀ s, h s = 0 → ∃ t, j t p q r = 0 ∧ s = t + 2) → 
  p = 4 ∧ q = 8 ∧ r = 7 := by
  sorry

end NUMINAMATH_CALUDE_roots_relation_l1250_125048


namespace NUMINAMATH_CALUDE_average_of_remaining_numbers_l1250_125091

theorem average_of_remaining_numbers
  (total : ℕ)
  (avg_all : ℚ)
  (avg_first_two : ℚ)
  (avg_next_two : ℚ)
  (h_total : total = 6)
  (h_avg_all : avg_all = 3.95)
  (h_avg_first_two : avg_first_two = 3.6)
  (h_avg_next_two : avg_next_two = 3.85) :
  (total * avg_all - 2 * avg_first_two - 2 * avg_next_two) / 2 = 4.4 := by
  sorry

end NUMINAMATH_CALUDE_average_of_remaining_numbers_l1250_125091


namespace NUMINAMATH_CALUDE_same_color_probability_l1250_125090

def totalBalls : ℕ := 20
def greenBalls : ℕ := 8
def redBalls : ℕ := 5
def blueBalls : ℕ := 7

theorem same_color_probability : 
  (greenBalls : ℚ) ^ 2 / totalBalls ^ 2 + 
  (redBalls : ℚ) ^ 2 / totalBalls ^ 2 + 
  (blueBalls : ℚ) ^ 2 / totalBalls ^ 2 = 345 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l1250_125090


namespace NUMINAMATH_CALUDE_alex_win_probability_l1250_125098

-- Define the game conditions
def standardDie : ℕ := 6
def evenNumbers : Set ℕ := {2, 4, 6}

-- Define the probability of Kelvin winning on first roll
def kelvinFirstWinProb : ℚ := 1 / 2

-- Define the probability of Alex winning on first roll, given Kelvin didn't win
def alexFirstWinProb : ℚ := 2 / 3

-- Define the probability of Kelvin winning on second roll, given Alex didn't win on first
def kelvinSecondWinProb : ℚ := 5 / 6

-- Define the probability of Alex winning on second roll, given Kelvin didn't win on second
def alexSecondWinProb : ℚ := 2 / 3

-- State the theorem
theorem alex_win_probability :
  let totalAlexWinProb := 
    kelvinFirstWinProb * alexFirstWinProb + 
    (1 - kelvinFirstWinProb) * (1 - alexFirstWinProb) * (1 - kelvinSecondWinProb) * alexSecondWinProb
  totalAlexWinProb = 22 / 27 := by
  sorry

end NUMINAMATH_CALUDE_alex_win_probability_l1250_125098


namespace NUMINAMATH_CALUDE_square_cut_parts_l1250_125071

/-- Represents a square grid paper -/
structure GridPaper :=
  (size : ℕ)

/-- Represents a folded square -/
structure FoldedSquare :=
  (original : GridPaper)
  (folded_size : ℕ)

/-- Represents a cut on the folded square -/
inductive Cut
  | Midpoint : Cut

/-- The number of parts resulting from unfolding after the cut -/
def num_parts_after_cut (fs : FoldedSquare) (c : Cut) : ℕ :=
  fs.original.size + 1

theorem square_cut_parts :
  ∀ (gp : GridPaper) (fs : FoldedSquare) (c : Cut),
    gp.size = 8 →
    fs.original = gp →
    fs.folded_size = 1 →
    c = Cut.Midpoint →
    num_parts_after_cut fs c = 9 :=
sorry

end NUMINAMATH_CALUDE_square_cut_parts_l1250_125071


namespace NUMINAMATH_CALUDE_infinite_powers_of_two_l1250_125003

/-- A sequence of natural numbers where each term is the sum of the previous term and its last digit -/
def LastDigitSequence (a₁ : ℕ) : ℕ → ℕ
  | 0 => a₁
  | n + 1 => LastDigitSequence a₁ n + (LastDigitSequence a₁ n % 10)

/-- The theorem stating that the LastDigitSequence contains infinitely many powers of 2 -/
theorem infinite_powers_of_two (a₁ : ℕ) (h : a₁ % 5 ≠ 0) :
  ∀ N : ℕ, ∃ k : ℕ, k > N ∧ ∃ m : ℕ, LastDigitSequence a₁ k = 2^m :=
sorry

end NUMINAMATH_CALUDE_infinite_powers_of_two_l1250_125003


namespace NUMINAMATH_CALUDE_sport_corn_syrup_amount_l1250_125069

/-- Represents the ratios in a flavored drink formulation -/
structure DrinkRatio :=
  (flavoring : ℚ)
  (corn_syrup : ℚ)
  (water : ℚ)

/-- The standard formulation ratio -/
def standard_ratio : DrinkRatio :=
  { flavoring := 1, corn_syrup := 12, water := 30 }

/-- The sport formulation ratio -/
def sport_ratio (r : DrinkRatio) : DrinkRatio :=
  { flavoring := r.flavoring,
    corn_syrup := r.corn_syrup / 3,
    water := r.water * 2 }

/-- Theorem stating the amount of corn syrup in the sport formulation -/
theorem sport_corn_syrup_amount (water_amount : ℚ) :
  let sr := sport_ratio standard_ratio
  let flavoring := water_amount / sr.water
  flavoring * sr.corn_syrup = 7 :=
sorry

end NUMINAMATH_CALUDE_sport_corn_syrup_amount_l1250_125069


namespace NUMINAMATH_CALUDE_absolute_value_equals_sqrt_of_square_l1250_125067

theorem absolute_value_equals_sqrt_of_square (x : ℝ) : |x| = Real.sqrt (x^2) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equals_sqrt_of_square_l1250_125067


namespace NUMINAMATH_CALUDE_find_b_find_perimeter_l1250_125092

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the conditions
def triangle_condition (t : Triangle) : Prop :=
  t.a * Real.cos t.B = (3 * t.c - t.b) * Real.cos t.A

def side_condition (t : Triangle) : Prop :=
  t.a * Real.sin t.B = 2 * Real.sqrt 2

def area_condition (t : Triangle) : Prop :=
  1/2 * t.a * t.b * Real.sin t.C = Real.sqrt 2

-- Theorem 1
theorem find_b (t : Triangle) 
  (h1 : triangle_condition t) 
  (h2 : side_condition t) : 
  t.b = 3 :=
sorry

-- Theorem 2
theorem find_perimeter (t : Triangle) 
  (h1 : t.a = 2 * Real.sqrt 2) 
  (h2 : area_condition t) : 
  t.a + t.b + t.c = 4 + 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_find_b_find_perimeter_l1250_125092


namespace NUMINAMATH_CALUDE_common_tangents_count_l1250_125036

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 16*y - 48 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 8*y - 44 = 0

-- Define the function to count common tangents
def count_common_tangents (c1 c2 : (ℝ → ℝ → Prop)) : ℕ := sorry

-- Theorem statement
theorem common_tangents_count :
  count_common_tangents circle1 circle2 = 2 := by sorry

end NUMINAMATH_CALUDE_common_tangents_count_l1250_125036


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l1250_125019

theorem arithmetic_expression_equality : (2 + 3^2) * 4 - 6 / 3 + 5^2 = 67 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l1250_125019


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l1250_125072

-- Problem 1
theorem problem_1 : -17 - (-6) + 8 - 2 = -5 := by sorry

-- Problem 2
theorem problem_2 : -1^2024 + 16 / (-2)^3 * |(-3) - 1| = -9 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l1250_125072


namespace NUMINAMATH_CALUDE_sphere_radius_range_l1250_125042

/-- Represents the parabola x^2 = 2y where 0 ≤ y ≤ 20 -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 = 2*p.2 ∧ 0 ≤ p.2 ∧ p.2 ≤ 20}

/-- A sphere touching the bottom of the parabola with its center on the y-axis -/
structure Sphere :=
  (center : ℝ)
  (radius : ℝ)
  (touches_bottom : radius = center)
  (inside_parabola : ∀ x y, (x, y) ∈ Parabola → x^2 + (y - center)^2 ≥ radius^2)

/-- The theorem stating the range of the sphere's radius -/
theorem sphere_radius_range (s : Sphere) : 0 < s.radius ∧ s.radius ≤ 1 := by
  sorry


end NUMINAMATH_CALUDE_sphere_radius_range_l1250_125042


namespace NUMINAMATH_CALUDE_percentage_not_sophomores_l1250_125061

theorem percentage_not_sophomores :
  ∀ (total juniors seniors freshmen sophomores : ℕ),
    total = 800 →
    juniors = (22 * total) / 100 →
    seniors = 160 →
    freshmen = sophomores + 48 →
    total = freshmen + sophomores + juniors + seniors →
    (100 * (total - sophomores)) / total = 74 := by
  sorry

end NUMINAMATH_CALUDE_percentage_not_sophomores_l1250_125061


namespace NUMINAMATH_CALUDE_problem_statement_l1250_125068

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (perpendicularLines : Line → Line → Prop)
variable (parallelPlanes : Plane → Plane → Prop)

-- Given two lines a and b, and two planes α and β
variable (a b : Line) (α β : Plane)

-- Given that b is perpendicular to α
variable (h : perpendicular b α)

theorem problem_statement :
  (parallel a α → perpendicularLines a b) ∧
  (perpendicular b β → parallelPlanes α β) := by sorry

end NUMINAMATH_CALUDE_problem_statement_l1250_125068


namespace NUMINAMATH_CALUDE_mia_study_time_l1250_125086

theorem mia_study_time (total_minutes : ℕ) (tv_fraction : ℚ) (study_minutes : ℕ) : 
  total_minutes = 1440 →
  tv_fraction = 1 / 5 →
  study_minutes = 288 →
  (study_minutes : ℚ) / (total_minutes - (tv_fraction * total_minutes : ℚ)) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_mia_study_time_l1250_125086


namespace NUMINAMATH_CALUDE_two_numbers_difference_l1250_125052

theorem two_numbers_difference (x y : ℝ) : 
  x + y = 20 → x^2 - y^2 = 200 → x - y = 10 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l1250_125052


namespace NUMINAMATH_CALUDE_oranges_per_glass_l1250_125006

/-- Proves that the number of oranges per glass is 2, given 12 oranges used for 6 glasses of juice -/
theorem oranges_per_glass (total_oranges : ℕ) (total_glasses : ℕ) 
  (h1 : total_oranges = 12) (h2 : total_glasses = 6) :
  total_oranges / total_glasses = 2 := by
  sorry

end NUMINAMATH_CALUDE_oranges_per_glass_l1250_125006


namespace NUMINAMATH_CALUDE_perpendicular_vectors_m_equals_five_l1250_125080

/-- Given two vectors a and b in ℝ², prove that if a is perpendicular to b, then m = 5 -/
theorem perpendicular_vectors_m_equals_five :
  let a : ℝ × ℝ := (1, -1)
  let b : ℝ × ℝ := (m + 1, 2 * m - 4)
  (a.1 * b.1 + a.2 * b.2 = 0) → m = 5 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_m_equals_five_l1250_125080


namespace NUMINAMATH_CALUDE_expand_polynomial_l1250_125088

theorem expand_polynomial (x : ℝ) : (x + 3) * (4 * x^2 - 5 * x + 6) = 4 * x^3 + 7 * x^2 - 9 * x + 18 := by
  sorry

end NUMINAMATH_CALUDE_expand_polynomial_l1250_125088
