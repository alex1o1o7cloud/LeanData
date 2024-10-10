import Mathlib

namespace coin_collection_values_l1318_131832

/-- Represents a collection of coins -/
structure CoinCollection where
  nickels : ℕ
  quarters : ℕ
  half_dollars : ℕ

/-- Defines the conditions for the coin collection -/
def valid_collection (c : CoinCollection) : Prop :=
  c.quarters = c.nickels / 2 ∧ c.half_dollars = 2 * c.quarters

/-- Calculates the total value of the coin collection in cents -/
def total_value (c : CoinCollection) : ℕ :=
  5 * c.nickels + 25 * c.quarters + 50 * c.half_dollars

/-- Theorem stating that there exist valid collections with total values of $67.50 and $135.00 -/
theorem coin_collection_values : 
  ∃ (c1 c2 : CoinCollection), 
    valid_collection c1 ∧ valid_collection c2 ∧ 
    total_value c1 = 6750 ∧ total_value c2 = 13500 :=
by sorry

end coin_collection_values_l1318_131832


namespace chord_intersection_l1318_131873

theorem chord_intersection (a : ℝ) : 
  let line := {(x, y) : ℝ × ℝ | x + y - a - 1 = 0}
  let circle := {(x, y) : ℝ × ℝ | (x - 2)^2 + (y - 2)^2 = 4}
  let chord_length := 2 * Real.sqrt 2
  (∃ (p q : ℝ × ℝ), p ∈ line ∧ p ∈ circle ∧ q ∈ line ∧ q ∈ circle ∧ 
    Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = chord_length) → 
  (a = 1 ∨ a = 5) :=
by sorry

end chord_intersection_l1318_131873


namespace sin_2alpha_plus_pi_6_l1318_131830

theorem sin_2alpha_plus_pi_6 (α : ℝ) (h : Real.sin (α - π/3) = 2/3 + Real.sin α) :
  Real.sin (2*α + π/6) = -1/9 := by
  sorry

end sin_2alpha_plus_pi_6_l1318_131830


namespace right_triangle_area_thrice_hypotenuse_l1318_131866

theorem right_triangle_area_thrice_hypotenuse : ∃ (a b c : ℝ), 
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- Positive sides
  c^2 = a^2 + b^2 ∧        -- Pythagorean theorem
  (1/2) * a * b = 3 * c    -- Area equals thrice the hypotenuse
  := by sorry

end right_triangle_area_thrice_hypotenuse_l1318_131866


namespace larger_number_problem_l1318_131864

theorem larger_number_problem (a b : ℝ) (h1 : a > b) (h2 : (a + b) + (a - b) = 68) : a = 34 := by
  sorry

end larger_number_problem_l1318_131864


namespace inequality_range_l1318_131858

theorem inequality_range (a : ℝ) :
  (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) ↔ a ∈ Set.Ioc (-2) 2 := by
  sorry

end inequality_range_l1318_131858


namespace parabola_properties_l1318_131886

-- Define the parabola function
def f (x : ℝ) : ℝ := (x - 1)^2 - 2

-- Theorem stating the properties of the parabola
theorem parabola_properties :
  (∀ x y : ℝ, f x = y → ∃ a : ℝ, a > 0 ∧ y = a * (x - 1)^2 - 2) ∧ 
  (∀ x y : ℝ, f x = y → f (2 - x) = y) ∧
  (f 1 = -2 ∧ ∀ x : ℝ, f x ≥ -2) ∧
  (∀ x₁ x₂ : ℝ, x₁ > 1 ∧ x₂ > 1 ∧ x₁ > x₂ → f x₁ > f x₂) :=
by sorry

end parabola_properties_l1318_131886


namespace max_area_rectangle_with_fixed_perimeter_l1318_131813

/-- The maximum area of a rectangle with a perimeter of 16 meters -/
theorem max_area_rectangle_with_fixed_perimeter : 
  ∀ (length width : ℝ), 
  length > 0 → width > 0 → 
  2 * (length + width) = 16 → 
  length * width ≤ 16 := by
sorry

end max_area_rectangle_with_fixed_perimeter_l1318_131813


namespace brets_nap_time_l1318_131841

/-- Represents the duration of Bret's train journey and activities --/
structure TrainJourney where
  totalTime : ℝ
  readingTime : ℝ
  eatingTime : ℝ
  movieTime : ℝ
  chattingTime : ℝ
  browsingTime : ℝ
  waitingTime : ℝ
  workingTime : ℝ

/-- Calculates the remaining time for napping given a TrainJourney --/
def remainingTimeForNap (journey : TrainJourney) : ℝ :=
  journey.totalTime - (journey.readingTime + journey.eatingTime + journey.movieTime + 
    journey.chattingTime + journey.browsingTime + journey.waitingTime + journey.workingTime)

/-- Theorem stating that for Bret's specific journey, the remaining time for napping is 4.75 hours --/
theorem brets_nap_time (journey : TrainJourney) 
  (h1 : journey.totalTime = 15)
  (h2 : journey.readingTime = 2)
  (h3 : journey.eatingTime = 1)
  (h4 : journey.movieTime = 3)
  (h5 : journey.chattingTime = 1)
  (h6 : journey.browsingTime = 0.75)
  (h7 : journey.waitingTime = 0.5)
  (h8 : journey.workingTime = 2) :
  remainingTimeForNap journey = 4.75 := by
  sorry

end brets_nap_time_l1318_131841


namespace crease_length_l1318_131882

/-- A right triangle with sides 5, 12, and 13 inches -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_angle : a^2 + b^2 = c^2
  a_eq_5 : a = 5
  b_eq_12 : b = 12
  c_eq_13 : c = 13

/-- The length of the perpendicular bisector from the right angle to the hypotenuse -/
def perp_bisector_length (t : RightTriangle) : ℝ := t.b

theorem crease_length (t : RightTriangle) :
  perp_bisector_length t = 12 := by sorry

end crease_length_l1318_131882


namespace solve_equation_l1318_131843

theorem solve_equation (x : ℚ) (h : (3/2) * x - 3 = 15) : x = 12 := by
  sorry

end solve_equation_l1318_131843


namespace three_distinct_zeros_l1318_131815

/-- A cubic function with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*a*x + a

/-- The derivative of f with respect to x -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 3*a

/-- Theorem: For f to have three distinct real zeros, a must be in (1/4, +∞) -/
theorem three_distinct_zeros (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a x = 0 ∧ f a y = 0 ∧ f a z = 0) ↔
  a > (1/4 : ℝ) :=
sorry

end three_distinct_zeros_l1318_131815


namespace paint_room_time_l1318_131822

/-- The time required for Doug, Dave, and Diana to paint a room together -/
theorem paint_room_time (t : ℝ) 
  (hDoug : (1 : ℝ) / 5 * t = 1)  -- Doug can paint the room in 5 hours
  (hDave : (1 : ℝ) / 7 * t = 1)  -- Dave can paint the room in 7 hours
  (hDiana : (1 : ℝ) / 6 * t = 1) -- Diana can paint the room in 6 hours
  (hLunch : ℝ) (hLunchTime : hLunch = 2) -- 2-hour lunch break
  : ((1 : ℝ) / 5 + 1 / 7 + 1 / 6) * (t - hLunch) = 1 :=
by sorry

end paint_room_time_l1318_131822


namespace terms_before_five_l1318_131835

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + d * (n - 1)

theorem terms_before_five (a₁ : ℤ) (d : ℤ) :
  a₁ = 105 ∧ d = -5 →
  ∃ n : ℕ, 
    arithmetic_sequence a₁ d n = 5 ∧ 
    (∀ k : ℕ, k < n → arithmetic_sequence a₁ d k > 5) ∧
    n - 1 = 20 := by
  sorry

end terms_before_five_l1318_131835


namespace quadratic_root_condition_l1318_131855

theorem quadratic_root_condition (m : ℝ) : 
  (∃ r₁ r₂ : ℝ, r₁ > 2 ∧ r₂ < 2 ∧ 
    r₁^2 + (2*m - 1)*r₁ + 4 - 2*m = 0 ∧
    r₂^2 + (2*m - 1)*r₂ + 4 - 2*m = 0) →
  m < -3 :=
by sorry

end quadratic_root_condition_l1318_131855


namespace fourth_square_area_l1318_131879

theorem fourth_square_area (PQ PR PS QR RS : ℝ) : 
  PQ^2 = 25 → QR^2 = 64 → RS^2 = 49 → 
  (PQ^2 + QR^2 = PR^2) → (PR^2 + RS^2 = PS^2) → 
  PS^2 = 138 := by sorry

end fourth_square_area_l1318_131879


namespace sheila_tuesday_thursday_hours_l1318_131889

/-- Represents Sheila's work schedule and earnings --/
structure WorkSchedule where
  hours_mwf : ℕ  -- Hours worked on Monday, Wednesday, Friday
  hours_tt : ℕ   -- Hours worked on Tuesday, Thursday
  hourly_rate : ℕ -- Hourly rate in dollars
  weekly_earnings : ℕ -- Total weekly earnings in dollars

/-- Calculates the total hours worked in a week --/
def total_hours (s : WorkSchedule) : ℕ :=
  3 * s.hours_mwf + 2 * s.hours_tt

/-- Calculates the total earnings based on hours worked and hourly rate --/
def calculated_earnings (s : WorkSchedule) : ℕ :=
  s.hourly_rate * (total_hours s)

/-- Theorem stating that Sheila works 6 hours on Tuesday and Thursday --/
theorem sheila_tuesday_thursday_hours (s : WorkSchedule) 
  (h1 : s.hours_mwf = 8)
  (h2 : s.hourly_rate = 11)
  (h3 : s.weekly_earnings = 396)
  (h4 : calculated_earnings s = s.weekly_earnings) :
  s.hours_tt = 6 := by
  sorry

end sheila_tuesday_thursday_hours_l1318_131889


namespace same_terminal_side_angles_l1318_131877

/-- Given an angle α = -51°, this theorem states that all angles with the same terminal side as α
    can be represented as k · 360° - 51°, where k is an integer. -/
theorem same_terminal_side_angles (α : ℝ) (h : α = -51) :
  ∀ θ : ℝ, (∃ k : ℤ, θ = k * 360 - 51) ↔ (∃ n : ℤ, θ = α + n * 360) :=
by sorry

end same_terminal_side_angles_l1318_131877


namespace goat_price_problem_l1318_131852

theorem goat_price_problem (total_cost num_cows num_goats cow_price : ℕ) 
  (h1 : total_cost = 1500)
  (h2 : num_cows = 2)
  (h3 : num_goats = 10)
  (h4 : cow_price = 400) :
  (total_cost - num_cows * cow_price) / num_goats = 70 := by
  sorry

end goat_price_problem_l1318_131852


namespace quadratic_symmetry_implies_ordering_l1318_131816

def f (x : ℝ) (b c : ℝ) : ℝ := x^2 + b*x + c

theorem quadratic_symmetry_implies_ordering (b c : ℝ) 
  (h : ∀ t : ℝ, f (2 + t) b c = f (2 - t) b c) : 
  f 2 b c < f 1 b c ∧ f 1 b c < f 4 b c := by
  sorry

end quadratic_symmetry_implies_ordering_l1318_131816


namespace boat_speed_in_still_water_l1318_131823

/-- The speed of a boat in still water, given its speeds with and against a stream -/
theorem boat_speed_in_still_water (along_stream speed_along_stream : ℝ) 
  (against_stream speed_against_stream : ℝ) :
  along_stream = 9 → against_stream = 5 →
  speed_along_stream = along_stream / 1 →
  speed_against_stream = against_stream / 1 →
  ∃ (boat_speed stream_speed : ℝ),
    boat_speed + stream_speed = speed_along_stream ∧
    boat_speed - stream_speed = speed_against_stream ∧
    boat_speed = 7 := by
  sorry


end boat_speed_in_still_water_l1318_131823


namespace chessboard_decomposition_l1318_131899

/-- Represents a square on a chessboard -/
structure Square where
  side : ℕ
  area : ℕ
  area_eq : area = side * side

/-- Represents a chessboard -/
structure Chessboard where
  side : ℕ
  area : ℕ
  area_eq : area = side * side

/-- Represents a decomposition of a chessboard into squares -/
structure Decomposition (board : Chessboard) where
  squares : List Square
  piece_count : ℕ
  valid : piece_count = 6 ∧ squares.length = 3
  area_sum : (squares.map (·.area)).sum = board.area

/-- The main theorem: A 7x7 chessboard can be decomposed into 6 pieces 
    that form three squares of sizes 6x6, 3x3, and 2x2 -/
theorem chessboard_decomposition :
  ∃ (d : Decomposition ⟨7, 49, rfl⟩),
    d.squares = [⟨6, 36, rfl⟩, ⟨3, 9, rfl⟩, ⟨2, 4, rfl⟩] := by
  sorry

end chessboard_decomposition_l1318_131899


namespace bug_probability_after_seven_steps_l1318_131801

-- Define the probability function
def probability (n : ℕ) : ℚ :=
  match n with
  | 0 => 1
  | m + 1 => 1/3 * (1 - probability m)

-- State the theorem
theorem bug_probability_after_seven_steps :
  probability 7 = 182 / 729 :=
sorry

end bug_probability_after_seven_steps_l1318_131801


namespace smallest_sum_with_same_prob_l1318_131829

/-- Represents a set of symmetrical dice -/
structure DiceSet where
  /-- The number of dice in the set -/
  num_dice : ℕ
  /-- The maximum number of points on each die -/
  max_points : ℕ
  /-- The probability of getting a sum of 2022 -/
  prob_2022 : ℝ
  /-- Assumption that the probability is positive -/
  pos_prob : prob_2022 > 0
  /-- Assumption that 2022 is achievable with these dice -/
  sum_2022 : num_dice * max_points = 2022

/-- 
Theorem: Given a set of symmetrical dice where a sum of 2022 is possible 
with probability p > 0, the smallest sum possible with the same probability p is 337.
-/
theorem smallest_sum_with_same_prob (d : DiceSet) : 
  d.num_dice = 337 := by sorry

end smallest_sum_with_same_prob_l1318_131829


namespace missing_number_proof_l1318_131810

theorem missing_number_proof : ∃ (x : ℤ), |7 - 8 * (x - 12)| - |5 - 11| = 73 ∧ x = 3 := by
  sorry

end missing_number_proof_l1318_131810


namespace train_length_calculation_l1318_131814

/-- The length of a train that crosses a platform of equal length in one minute at 90 km/hr is 750 meters. -/
theorem train_length_calculation (train_length platform_length : ℝ) (speed : ℝ) (time : ℝ) :
  train_length = platform_length →
  speed = 90 →
  time = 1 / 60 →
  train_length = 750 := by
  sorry

end train_length_calculation_l1318_131814


namespace log_comparison_l1318_131826

theorem log_comparison : Real.log 7 / Real.log 5 > Real.log 17 / Real.log 13 := by
  sorry

end log_comparison_l1318_131826


namespace xy_is_zero_l1318_131857

theorem xy_is_zero (x y : ℝ) (h1 : x - y = 6) (h2 : x^3 - y^3 = 162) : x * y = 0 := by
  sorry

end xy_is_zero_l1318_131857


namespace quadratic_discriminant_zero_implies_geometric_progression_l1318_131860

theorem quadratic_discriminant_zero_implies_geometric_progression
  (k a b c : ℝ) (h1 : k ≠ 0) :
  4 * k^2 * (b^2 - a*c) = 0 →
  ∃ r : ℝ, r ≠ 0 ∧ b = a * r ∧ c = b * r :=
by sorry

end quadratic_discriminant_zero_implies_geometric_progression_l1318_131860


namespace hexagon_pentagon_angle_sum_l1318_131880

theorem hexagon_pentagon_angle_sum : 
  let hexagon_angle := 180 * (6 - 2) / 6
  let pentagon_angle := 180 * (5 - 2) / 5
  hexagon_angle + pentagon_angle = 228 := by
  sorry

end hexagon_pentagon_angle_sum_l1318_131880


namespace sin_150_degrees_l1318_131804

theorem sin_150_degrees : Real.sin (150 * π / 180) = 1 / 2 := by
  sorry

end sin_150_degrees_l1318_131804


namespace negation_of_implication_l1318_131825

theorem negation_of_implication (a b : ℝ) :
  ¬(a > b → a + 1 > b) ↔ (a ≤ b → a + 1 ≤ b) := by sorry

end negation_of_implication_l1318_131825


namespace gcd_of_45_135_225_l1318_131854

theorem gcd_of_45_135_225 : Nat.gcd 45 (Nat.gcd 135 225) = 45 := by
  sorry

end gcd_of_45_135_225_l1318_131854


namespace cube_remainder_l1318_131890

theorem cube_remainder (n : ℤ) : n % 6 = 3 → n^3 % 8 = 3 := by
  sorry

end cube_remainder_l1318_131890


namespace circle_area_when_radius_equals_three_times_reciprocal_circumference_l1318_131817

theorem circle_area_when_radius_equals_three_times_reciprocal_circumference :
  ∀ r : ℝ, r > 0 →
  (3 * (1 / (2 * π * r)) = r) →
  (π * r^2 = 3/2) := by
sorry

end circle_area_when_radius_equals_three_times_reciprocal_circumference_l1318_131817


namespace shoe_ratio_problem_l1318_131888

/-- Proof of the shoe ratio problem -/
theorem shoe_ratio_problem (brian_shoes edward_shoes jacob_shoes : ℕ) : 
  (edward_shoes = 3 * brian_shoes) →
  (brian_shoes = 22) →
  (jacob_shoes + edward_shoes + brian_shoes = 121) →
  (jacob_shoes : ℚ) / edward_shoes = 1 / 2 := by
  sorry

end shoe_ratio_problem_l1318_131888


namespace january_first_day_l1318_131851

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a month with specific properties -/
structure Month where
  days : Nat
  tuesdays : Nat
  saturdays : Nat

/-- Returns the day of the week for the first day of the month -/
def firstDayOfMonth (m : Month) : DayOfWeek :=
  sorry

theorem january_first_day (m : Month) 
  (h1 : m.days = 31)
  (h2 : m.tuesdays = 4)
  (h3 : m.saturdays = 4) :
  firstDayOfMonth m = DayOfWeek.Wednesday :=
sorry

end january_first_day_l1318_131851


namespace smallest_prime_divisor_of_sum_l1318_131805

theorem smallest_prime_divisor_of_sum : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ (3^19 + 11^13) ∧ ∀ q, Nat.Prime q → q ∣ (3^19 + 11^13) → p ≤ q :=
by sorry

end smallest_prime_divisor_of_sum_l1318_131805


namespace max_value_constraint_l1318_131838

theorem max_value_constraint (x y z : ℝ) (h : x^2 + y^2 + z^2 = 2) :
  3*x + 4*y + 5*z ≤ 10 :=
by sorry

end max_value_constraint_l1318_131838


namespace range_of_a_l1318_131846

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 + 2*a*x + a > 0) ↔ (0 < a ∧ a < 1) := by
  sorry

end range_of_a_l1318_131846


namespace decimal_to_fraction_l1318_131836

theorem decimal_to_fraction :
  (3.75 : ℚ) = 15 / 4 := by sorry

end decimal_to_fraction_l1318_131836


namespace victors_percentage_l1318_131821

/-- Calculate the percentage of marks obtained given the marks scored and maximum marks -/
def calculatePercentage (marksScored : ℕ) (maxMarks : ℕ) : ℚ :=
  (marksScored : ℚ) / (maxMarks : ℚ) * 100

/-- Theorem stating that Victor's percentage of marks is 95% -/
theorem victors_percentage :
  let marksScored : ℕ := 285
  let maxMarks : ℕ := 300
  calculatePercentage marksScored maxMarks = 95 := by
  sorry


end victors_percentage_l1318_131821


namespace order_of_magnitude_l1318_131898

theorem order_of_magnitude (a b : ℝ) (ha : a > 0) (hb : b < 0) (hab : |a| < |b|) :
  -b > a ∧ a > -a ∧ -a > b := by
  sorry

end order_of_magnitude_l1318_131898


namespace recipe_flour_cups_l1318_131893

/-- The number of cups of sugar required in the recipe -/
def sugar_cups : ℕ := 9

/-- The number of cups of flour Mary has already put in -/
def flour_cups_added : ℕ := 4

/-- The total number of cups of flour required in the recipe -/
def total_flour_cups : ℕ := sugar_cups + 1

theorem recipe_flour_cups : total_flour_cups = 10 := by
  sorry

end recipe_flour_cups_l1318_131893


namespace carter_reads_30_pages_l1318_131849

/-- The number of pages Oliver can read in 1 hour -/
def oliver_pages : ℕ := 40

/-- The number of pages Lucy can read in 1 hour -/
def lucy_pages : ℕ := oliver_pages + 20

/-- The number of pages Carter can read in 1 hour -/
def carter_pages : ℕ := lucy_pages / 2

/-- Theorem stating that Carter can read 30 pages in 1 hour -/
theorem carter_reads_30_pages : carter_pages = 30 := by
  sorry

end carter_reads_30_pages_l1318_131849


namespace number_ordering_l1318_131807

theorem number_ordering : (1 : ℚ) / 5 < (25 : ℚ) / 100 ∧ (25 : ℚ) / 100 < (42 : ℚ) / 100 ∧ (42 : ℚ) / 100 < (1 : ℚ) / 2 ∧ (1 : ℚ) / 2 < (3 : ℚ) / 4 := by
  sorry

end number_ordering_l1318_131807


namespace exponent_division_l1318_131869

theorem exponent_division (a : ℝ) (h : a ≠ 0) : a^12 / a^4 = a^8 := by
  sorry

end exponent_division_l1318_131869


namespace simplify_expression_l1318_131834

theorem simplify_expression (x y : ℝ) (m : ℤ) :
  (x + y) ^ (2 * m + 1) / (x + y) ^ (m - 1) = (x + y) ^ (m + 2) := by
  sorry

end simplify_expression_l1318_131834


namespace perpendicular_line_through_point_l1318_131871

/-- The equation of a line perpendicular to 2x + y - 5 = 0 and passing through (3, 0) -/
theorem perpendicular_line_through_point (x y : ℝ) :
  (2 : ℝ) * x + y - 5 = 0 →  -- Given line
  (∃ c : ℝ, x - 2 * y + c = 0 ∧  -- General form of perpendicular line
            3 - 2 * 0 + c = 0 ∧  -- Passes through (3, 0)
            x - 2 * y - 3 = 0) :=  -- The specific equation we want to prove
by sorry

end perpendicular_line_through_point_l1318_131871


namespace train_speed_calculation_l1318_131844

/-- The speed of the first train -/
def speed_first_train : ℝ := 20

/-- The distance between stations P and Q -/
def distance_PQ : ℝ := 110

/-- The speed of the second train -/
def speed_second_train : ℝ := 25

/-- The time the first train travels before meeting -/
def time_first_train : ℝ := 3

/-- The time the second train travels before meeting -/
def time_second_train : ℝ := 2

theorem train_speed_calculation :
  speed_first_train * time_first_train + speed_second_train * time_second_train = distance_PQ :=
by sorry

end train_speed_calculation_l1318_131844


namespace remainder_sum_mod_seven_l1318_131897

theorem remainder_sum_mod_seven (a b c : ℕ) : 
  a < 7 → b < 7 → c < 7 → a > 0 → b > 0 → c > 0 →
  (a * b * c) % 7 = 2 →
  (3 * c) % 7 = 1 →
  (4 * b) % 7 = (2 + b) % 7 →
  (a + b + c) % 7 = 3 := by
sorry

end remainder_sum_mod_seven_l1318_131897


namespace solve_for_c_l1318_131828

theorem solve_for_c (p q : ℝ → ℝ) (c : ℝ) : 
  (∀ x, p x = 4 * x - 5) →
  (∀ x, q x = 5 * x - c) →
  p (q 3) = 27 →
  c = 7 := by
sorry

end solve_for_c_l1318_131828


namespace train_crossing_time_l1318_131824

/-- Proves that a train with given length and speed takes a specific time to cross a pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 375 →
  train_speed_kmh = 90 →
  crossing_time = 15 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) := by
  sorry

#check train_crossing_time

end train_crossing_time_l1318_131824


namespace ken_snow_days_l1318_131865

/-- Represents the cycling scenario for Ken in a week -/
structure CyclingWeek where
  rain_speed : ℕ  -- miles per 20 minutes when raining
  snow_speed : ℕ  -- miles per 20 minutes when snowing
  rain_days : ℕ   -- number of rainy days
  total_miles : ℕ -- total miles cycled in the week
  hours_per_day : ℕ -- hours cycled per day

/-- Calculates the number of snowy days in a week -/
def snow_days (w : CyclingWeek) : ℕ :=
  ((w.total_miles - w.rain_days * w.rain_speed * 3) / (w.snow_speed * 3))

/-- Theorem stating the number of snowy days in Ken's cycling week -/
theorem ken_snow_days :
  let w : CyclingWeek := {
    rain_speed := 30,
    snow_speed := 10,
    rain_days := 3,
    total_miles := 390,
    hours_per_day := 1
  }
  snow_days w = 4 := by sorry

end ken_snow_days_l1318_131865


namespace simplify_fraction_l1318_131859

theorem simplify_fraction : 45 * (14 / 25) * (1 / 18) * (5 / 11) = 7 / 11 := by
  sorry

end simplify_fraction_l1318_131859


namespace base_n_problem_l1318_131878

theorem base_n_problem (n : ℕ+) (d : ℕ) (h1 : d < 10) 
  (h2 : 4 * n ^ 2 + 2 * n + d = 347)
  (h3 : 4 * n ^ 2 + 2 * n + 9 = 1 * 7 ^ 3 + 2 * 7 ^ 2 + d * 7 + 2) :
  n + d = 11 := by
sorry

end base_n_problem_l1318_131878


namespace sin_theta_value_l1318_131862

theorem sin_theta_value (θ : Real) (h1 : 5 * Real.tan θ = 2 * Real.cos θ) (h2 : 0 < θ) (h3 : θ < Real.pi) :
  Real.sin θ = 1/2 := by
sorry

end sin_theta_value_l1318_131862


namespace negation_of_existential_l1318_131850

theorem negation_of_existential (p : Prop) :
  (¬∃ (x : ℝ), x^2 + 2*x = 3) ↔ (∀ (x : ℝ), x^2 + 2*x ≠ 3) := by
  sorry

end negation_of_existential_l1318_131850


namespace cog_production_90_workers_2_hours_l1318_131811

/-- Represents the production capabilities of workers in a factory --/
structure ProductionRate where
  gears_per_hour : ℝ
  cogs_per_hour : ℝ

/-- Calculates the total production for a given number of workers, hours, and production rate --/
def total_production (workers : ℝ) (hours : ℝ) (rate : ProductionRate) : ProductionRate :=
  { gears_per_hour := workers * hours * rate.gears_per_hour,
    cogs_per_hour := workers * hours * rate.cogs_per_hour }

/-- Theorem stating the production of cogs by 90 workers in 2 hours --/
theorem cog_production_90_workers_2_hours 
  (rate : ProductionRate)
  (h1 : total_production 150 1 rate = { gears_per_hour := 450, cogs_per_hour := 300 })
  (h2 : total_production 100 1.5 rate = { gears_per_hour := 300, cogs_per_hour := 375 })
  (h3 : (total_production 90 2 rate).gears_per_hour = 360) :
  (total_production 90 2 rate).cogs_per_hour = 180 := by
  sorry

#check cog_production_90_workers_2_hours

end cog_production_90_workers_2_hours_l1318_131811


namespace quadratic_intersects_x_axis_l1318_131883

/-- A quadratic function y = kx^2 - 7x - 7 intersects the x-axis if and only if k ≥ -7/4 and k ≠ 0 -/
theorem quadratic_intersects_x_axis (k : ℝ) : 
  (∃ x : ℝ, k * x^2 - 7 * x - 7 = 0) ↔ (k ≥ -7/4 ∧ k ≠ 0) :=
sorry

end quadratic_intersects_x_axis_l1318_131883


namespace miles_driven_equals_365_l1318_131875

/-- Calculates the total miles driven given car efficiencies and gas usage --/
def total_miles_driven (highway_mpg city_mpg : ℚ) (total_gas : ℚ) (highway_city_diff : ℚ) : ℚ :=
  let city_miles := (total_gas * highway_mpg * city_mpg - city_mpg * highway_city_diff) / (highway_mpg + city_mpg)
  let highway_miles := city_miles + highway_city_diff
  city_miles + highway_miles

/-- Theorem stating the total miles driven under given conditions --/
theorem miles_driven_equals_365 :
  total_miles_driven 37 30 11 5 = 365 := by
  sorry

end miles_driven_equals_365_l1318_131875


namespace box_of_balls_l1318_131845

theorem box_of_balls (blue : ℕ) (red : ℕ) (green : ℕ) (yellow : ℕ) : 
  blue = 6 →
  red = 4 →
  green = 3 * blue →
  yellow = 2 * red →
  blue + red + green + yellow = 36 :=
by
  sorry

end box_of_balls_l1318_131845


namespace square_root_of_square_negative_two_l1318_131872

theorem square_root_of_square_negative_two : Real.sqrt ((-2)^2) = 2 := by
  sorry

end square_root_of_square_negative_two_l1318_131872


namespace max_stores_visited_l1318_131861

theorem max_stores_visited (total_stores : ℕ) (total_visits : ℕ) (total_shoppers : ℕ) 
  (double_visitors : ℕ) (h1 : total_stores = 12) (h2 : total_visits = 45) 
  (h3 : total_shoppers = 22) (h4 : double_visitors = 14) 
  (h5 : double_visitors ≤ total_shoppers) 
  (h6 : 2 * double_visitors ≤ total_visits) :
  ∃ (max_visits : ℕ), max_visits ≤ total_stores ∧ 
    (∀ (person_visits : ℕ), person_visits ≤ max_visits) ∧ 
    max_visits = 10 :=
by sorry

end max_stores_visited_l1318_131861


namespace fourth_number_proof_l1318_131808

theorem fourth_number_proof (x : ℝ) (fourth_number : ℝ) : 
  (128 + 255 + 511 + 1023 + x) / 5 = 423 →
  (28 + x + 42 + fourth_number + 104) / 5 = 90 →
  fourth_number = 78 := by
sorry

end fourth_number_proof_l1318_131808


namespace maplefield_population_l1318_131809

/-- The number of towns in the Region of Maplefield -/
def num_towns : ℕ := 25

/-- The lower bound of the average population range -/
def lower_bound : ℕ := 4800

/-- The upper bound of the average population range -/
def upper_bound : ℕ := 5300

/-- The average population of a town in the Region of Maplefield -/
def avg_population : ℚ := (lower_bound + upper_bound) / 2

/-- The total population of all towns in the Region of Maplefield -/
def total_population : ℚ := num_towns * avg_population

theorem maplefield_population : total_population = 126250 := by
  sorry

end maplefield_population_l1318_131809


namespace pizza_pepperoni_count_l1318_131884

theorem pizza_pepperoni_count :
  ∀ (pepperoni ham sausage : ℕ),
    ham = 2 * pepperoni →
    sausage = pepperoni + 12 →
    pepperoni + ham + sausage = 22 * 6 →
    pepperoni = 30 := by
  sorry

end pizza_pepperoni_count_l1318_131884


namespace arithmetic_sequence_fifth_term_l1318_131894

/-- Given an arithmetic sequence where the 20th term is 17 and the 21st term is 20,
    prove that the 5th term is -28. -/
theorem arithmetic_sequence_fifth_term
  (a : ℤ) -- First term of the sequence
  (d : ℤ) -- Common difference
  (h1 : a + 19 * d = 17) -- 20th term is 17
  (h2 : a + 20 * d = 20) -- 21st term is 20
  : a + 4 * d = -28 := by -- 5th term is -28
  sorry

end arithmetic_sequence_fifth_term_l1318_131894


namespace students_6_to_8_hours_l1318_131885

/-- Represents a frequency distribution histogram for study times -/
structure StudyTimeHistogram where
  total_students : ℕ
  freq_6_to_8 : ℕ
  -- Other fields for other time intervals could be added here

/-- Theorem stating that in a given histogram of 100 students, 30 studied for 6 to 8 hours -/
theorem students_6_to_8_hours (h : StudyTimeHistogram) 
  (h_total : h.total_students = 100) : h.freq_6_to_8 = 30 := by
  sorry

end students_6_to_8_hours_l1318_131885


namespace difference_of_31st_terms_l1318_131868

def arithmeticSequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + d * (n - 1)

theorem difference_of_31st_terms : 
  let C := arithmeticSequence 50 12
  let D := arithmeticSequence 50 (-8)
  |C 31 - D 31| = 600 := by sorry

end difference_of_31st_terms_l1318_131868


namespace peanuts_added_l1318_131818

theorem peanuts_added (initial : ℕ) (final : ℕ) (added : ℕ) : 
  initial = 4 → final = 6 → final = initial + added → added = 2 := by
sorry

end peanuts_added_l1318_131818


namespace sweater_shirt_price_difference_l1318_131874

/-- Given the total price and quantity of shirts and sweaters, 
    prove that the average price of a sweater exceeds that of a shirt by $4 -/
theorem sweater_shirt_price_difference 
  (shirt_quantity : ℕ) 
  (shirt_total_price : ℚ)
  (sweater_quantity : ℕ)
  (sweater_total_price : ℚ)
  (h_shirt_quantity : shirt_quantity = 25)
  (h_shirt_price : shirt_total_price = 400)
  (h_sweater_quantity : sweater_quantity = 75)
  (h_sweater_price : sweater_total_price = 1500) :
  sweater_total_price / sweater_quantity - shirt_total_price / shirt_quantity = 4 := by
sorry

end sweater_shirt_price_difference_l1318_131874


namespace flagpole_height_l1318_131806

/-- Given a tree and a flagpole with known measurements, calculate the height of the flagpole -/
theorem flagpole_height
  (tree_height : ℝ)
  (tree_shadow : ℝ)
  (flagpole_shadow : ℝ)
  (h_tree_height : tree_height = 3.6)
  (h_tree_shadow : tree_shadow = 0.6)
  (h_flagpole_shadow : flagpole_shadow = 1.5) :
  ∃ (flagpole_height : ℝ), flagpole_height = 9 ∧
    tree_height / tree_shadow = flagpole_height / flagpole_shadow :=
by sorry

end flagpole_height_l1318_131806


namespace inequality_holds_iff_l1318_131863

theorem inequality_holds_iff (a : ℝ) :
  (∀ x ∈ Set.Icc 2 3, a * x^2 - (a + 2) * x + 2 < 0) ↔ a < 2/3 := by
  sorry

end inequality_holds_iff_l1318_131863


namespace slower_train_speed_calculation_l1318_131856

/-- The speed of the faster train in kilometers per hour -/
def faster_train_speed : ℝ := 162

/-- The length of the faster train in meters -/
def faster_train_length : ℝ := 1320

/-- The time taken by the faster train to cross a man in the slower train, in seconds -/
def crossing_time : ℝ := 33

/-- The speed of the slower train in kilometers per hour -/
def slower_train_speed : ℝ := 18

theorem slower_train_speed_calculation :
  let relative_speed := (faster_train_speed - slower_train_speed) * 1000 / 3600
  faster_train_length = relative_speed * crossing_time →
  slower_train_speed = 18 := by
  sorry

end slower_train_speed_calculation_l1318_131856


namespace smallest_n_for_congruence_l1318_131891

/-- Concatenation of powers of 2 -/
def A (n : ℕ) : ℕ :=
  -- We define A as a placeholder function, as the actual implementation is complex
  sorry

/-- The main theorem -/
theorem smallest_n_for_congruence : 
  (∀ k : ℕ, 3 ≤ k → k < 14 → ¬(A k ≡ 2^(10*k) [MOD 2^170])) ∧ 
  (A 14 ≡ 2^(10*14) [MOD 2^170]) := by
  sorry

end smallest_n_for_congruence_l1318_131891


namespace mika_stickers_total_l1318_131800

/-- The total number of stickers Mika has after receiving stickers from various sources -/
theorem mika_stickers_total : 
  let initial : ℝ := 20.5
  let bought : ℝ := 26.3
  let birthday : ℝ := 19.75
  let sister : ℝ := 6.25
  let mother : ℝ := 57.65
  let cousin : ℝ := 15.8
  initial + bought + birthday + sister + mother + cousin = 146.25 := by
  sorry

end mika_stickers_total_l1318_131800


namespace condition_implies_right_triangle_l1318_131803

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the condition given in the problem
def satisfiesCondition (t : Triangle) : Prop :=
  (t.a + t.b)^2 = t.c^2 + 2*t.a*t.b

-- Define what it means for a triangle to be a right triangle
def isRightTriangle (t : Triangle) : Prop :=
  t.a^2 + t.b^2 = t.c^2

-- State the theorem
theorem condition_implies_right_triangle (t : Triangle) :
  satisfiesCondition t → isRightTriangle t := by
  sorry

end condition_implies_right_triangle_l1318_131803


namespace linear_iff_m_neq_neg_six_l1318_131895

/-- A function f is linear if there exist constants a and b such that f(x) = ax + b for all x, and a ≠ 0 -/
def IsLinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x + b

/-- The function f_m for a given m -/
def f_m (m : ℝ) : ℝ → ℝ := λ x ↦ (m + 2) * x + 4 * x - 5

theorem linear_iff_m_neq_neg_six (m : ℝ) :
  IsLinearFunction (f_m m) ↔ m ≠ -6 := by
  sorry

end linear_iff_m_neq_neg_six_l1318_131895


namespace trigonometric_simplification_l1318_131881

theorem trigonometric_simplification (α : ℝ) :
  (Real.sin (π - α) / Real.cos (π + α)) *
  (Real.cos (-α) * Real.cos (2*π - α)) /
  Real.sin (π/2 + α) = -Real.sin α := by
  sorry

end trigonometric_simplification_l1318_131881


namespace pizza_toppings_combinations_l1318_131876

theorem pizza_toppings_combinations (n m : ℕ) (h1 : n = 8) (h2 : m = 5) : 
  Nat.choose n m = 56 := by
  sorry

end pizza_toppings_combinations_l1318_131876


namespace cylinder_radius_l1318_131848

/-- The original radius of a cylinder satisfying specific conditions -/
theorem cylinder_radius : ∃ (r : ℝ), r > 0 ∧ 
  (∀ (y : ℝ), 
    (2 * π * ((r + 6)^2 - r^2) = y) ∧ 
    (6 * π * r^2 = y)) → 
  r = 6 := by sorry

end cylinder_radius_l1318_131848


namespace area_of_quadrilateral_l1318_131896

/-- Given two orthonormal vectors e₁ and e₂, and vectors AC and BD defined in terms of e₁ and e₂,
    prove that the area of quadrilateral ABCD is 10. -/
theorem area_of_quadrilateral (e₁ e₂ : ℝ × ℝ) 
    (h_orthonormal : e₁ • e₁ = 1 ∧ e₂ • e₂ = 1 ∧ e₁ • e₂ = 0) 
    (AC : ℝ × ℝ) (h_AC : AC = 3 • e₁ - e₂)
    (BD : ℝ × ℝ) (h_BD : BD = 2 • e₁ + 6 • e₂) : 
  Real.sqrt ((AC.1^2 + AC.2^2) * (BD.1^2 + BD.2^2)) / 2 = 10 := by
  sorry

end area_of_quadrilateral_l1318_131896


namespace other_endpoint_of_line_segment_l1318_131827

/-- Given a line segment with midpoint (-1, 4) and one endpoint (3, -1), 
    the other endpoint is (-5, 9). -/
theorem other_endpoint_of_line_segment (m x₁ y₁ x₂ y₂ : ℝ) : 
  m = (-1 : ℝ) ∧ 
  (4 : ℝ) = (y₁ + y₂) / 2 ∧ 
  x₁ = (3 : ℝ) ∧ 
  y₁ = (-1 : ℝ) ∧ 
  m = (x₁ + x₂) / 2 → 
  x₂ = (-5 : ℝ) ∧ y₂ = (9 : ℝ) := by
  sorry

end other_endpoint_of_line_segment_l1318_131827


namespace new_person_weight_l1318_131867

/-- Calculates the weight of a new person given the following conditions:
  * There are 6 people initially
  * Replacing one person weighing 69 kg with a new person increases the average weight by 1.8 kg
-/
theorem new_person_weight (num_people : Nat) (weight_increase : Real) (replaced_weight : Real) :
  num_people = 6 →
  weight_increase = 1.8 →
  replaced_weight = 69 →
  ∃ (new_weight : Real), new_weight = 79.8 ∧
    new_weight = replaced_weight + num_people * weight_increase :=
by sorry

end new_person_weight_l1318_131867


namespace floor_sqrt_120_l1318_131812

theorem floor_sqrt_120 : ⌊Real.sqrt 120⌋ = 10 := by
  sorry

end floor_sqrt_120_l1318_131812


namespace average_score_is_76_point_8_l1318_131837

def class_size : ℕ := 50

def first_group_scores : List ℕ := [90, 85, 88, 92, 80, 94, 89, 91, 84, 87]

def second_group_scores : List ℕ := 
  [85, 80, 83, 87, 75, 89, 84, 86, 79, 82, 77, 74, 81, 78, 70]

def third_group_scores : List ℕ := 
  [40, 62, 58, 70, 72, 68, 64, 66, 74, 76, 60, 78, 80, 82, 84, 86, 88, 61, 63, 65, 67, 69, 71, 73, 75]

def total_score : ℕ := 
  (first_group_scores.sum + second_group_scores.sum + third_group_scores.sum)

theorem average_score_is_76_point_8 :
  (total_score : ℚ) / class_size = 76.8 := by
  sorry

end average_score_is_76_point_8_l1318_131837


namespace cube_of_product_l1318_131842

theorem cube_of_product (a b : ℝ) : (a * b) ^ 3 = a ^ 3 * b ^ 3 := by
  sorry

end cube_of_product_l1318_131842


namespace gcd_7429_12345_is_1_l1318_131839

theorem gcd_7429_12345_is_1 : Nat.gcd 7429 12345 = 1 := by
  sorry

end gcd_7429_12345_is_1_l1318_131839


namespace log_equation_solution_l1318_131870

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  Real.log x / Real.log 3 + Real.log x / Real.log 9 = 5 → x = (3^10)^(1/3) := by
sorry

end log_equation_solution_l1318_131870


namespace inequality_solution_set_l1318_131819

theorem inequality_solution_set (a : ℝ) :
  let S := {x : ℝ | (x - 1) * (x + a) > 0}
  if a < -1 then
    S = {x : ℝ | x > -a ∨ x < 1}
  else if a = -1 then
    S = {x : ℝ | x ≠ 1}
  else
    S = {x : ℝ | x < -a ∨ x > 1} := by
  sorry

end inequality_solution_set_l1318_131819


namespace max_value_of_g_l1318_131853

-- Define the function g(x)
def g (x : ℝ) : ℝ := 4 * x - x^4

-- State the theorem
theorem max_value_of_g :
  ∃ (c : ℝ), c ∈ Set.Icc 0 2 ∧ 
  (∀ x, x ∈ Set.Icc 0 2 → g x ≤ g c) ∧
  g c = 3 :=
sorry

end max_value_of_g_l1318_131853


namespace average_pen_price_l1318_131847

/-- Given the purchase of pens and pencils, prove the average price of a pen. -/
theorem average_pen_price 
  (total_cost : ℝ)
  (num_pens : ℕ)
  (num_pencils : ℕ)
  (avg_pencil_price : ℝ)
  (h1 : total_cost = 570)
  (h2 : num_pens = 30)
  (h3 : num_pencils = 75)
  (h4 : avg_pencil_price = 2) :
  (total_cost - num_pencils * avg_pencil_price) / num_pens = 14 := by
  sorry

end average_pen_price_l1318_131847


namespace boys_meeting_on_circular_track_l1318_131820

/-- The number of times two boys meet on a circular track -/
def number_of_meetings (speed1 speed2 : ℝ) : ℕ :=
  -- We'll define this function later
  sorry

/-- Theorem: Two boys moving in opposite directions on a circular track with speeds
    of 5 ft/s and 9 ft/s will meet 13 times before returning to the starting point -/
theorem boys_meeting_on_circular_track :
  number_of_meetings 5 9 = 13 := by
  sorry

end boys_meeting_on_circular_track_l1318_131820


namespace smallest_three_digit_number_l1318_131802

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  ones : Nat
  h1 : tens ≥ 1 ∧ tens ≤ 9
  h2 : ones ≥ 0 ∧ ones ≤ 9

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  h1 : hundreds ≥ 1 ∧ hundreds ≤ 9
  h2 : tens ≥ 0 ∧ tens ≤ 9
  h3 : ones ≥ 0 ∧ ones ≤ 9

/-- Converts a TwoDigitNumber to its numerical value -/
def twoDigitToNat (n : TwoDigitNumber) : Nat :=
  10 * n.tens + n.ones

/-- Converts a ThreeDigitNumber to its numerical value -/
def threeDigitToNat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

theorem smallest_three_digit_number 
  (ab : TwoDigitNumber) 
  (aab : ThreeDigitNumber) 
  (h1 : ab.tens = aab.hundreds ∧ ab.tens = aab.tens)
  (h2 : ab.ones = aab.ones)
  (h3 : ab.tens ≠ ab.ones)
  (h4 : twoDigitToNat ab = (threeDigitToNat aab) / 9) :
  225 ≤ threeDigitToNat aab :=
by sorry

end smallest_three_digit_number_l1318_131802


namespace parabola_properties_given_parabola_properties_l1318_131892

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ
  D : ℝ × ℝ

/-- Given conditions for the parabola -/
def given_parabola : Parabola where
  p := 2  -- This is derived from the solution, not given directly
  equation := λ x y => y^2 = 2 * 2 * x
  focus := (1, 0)
  D := (2, 0)

/-- Theorem stating the main results -/
theorem parabola_properties (C : Parabola) 
  (h1 : C.p > 0)
  (h2 : C.D = (C.p, 0))
  (h3 : ∃ (M : ℝ × ℝ), C.equation M.1 M.2 ∧ 
        (M.2 - C.D.2) / (M.1 - C.D.1) = 0 ∧ 
        Real.sqrt ((M.1 - C.focus.1)^2 + (M.2 - C.focus.2)^2) = 3) :
  (C.equation = λ x y => y^2 = 4*x) ∧
  (∃ (A B : ℝ × ℝ), 
    C.equation A.1 A.2 ∧ 
    C.equation B.1 B.2 ∧
    (B.2 - A.2) / (B.1 - A.1) = -1/Real.sqrt 2 ∧
    A.1 - Real.sqrt 2 * A.2 - 4 = 0) :=
by sorry

/-- Applying the theorem to the given parabola -/
theorem given_parabola_properties : 
  (given_parabola.equation = λ x y => y^2 = 4*x) ∧
  (∃ (A B : ℝ × ℝ), 
    given_parabola.equation A.1 A.2 ∧ 
    given_parabola.equation B.1 B.2 ∧
    (B.2 - A.2) / (B.1 - A.1) = -1/Real.sqrt 2 ∧
    A.1 - Real.sqrt 2 * A.2 - 4 = 0) :=
by sorry

end parabola_properties_given_parabola_properties_l1318_131892


namespace intersection_A_B_l1318_131887

def A : Set ℤ := {1, 2, 3}

def B : Set ℤ := {x | (x + 1) * (x - 2) < 0}

theorem intersection_A_B : A ∩ B = {1} := by sorry

end intersection_A_B_l1318_131887


namespace xyz_value_l1318_131831

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 45)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 19) :
  x * y * z = 26 / 3 := by
sorry

end xyz_value_l1318_131831


namespace student_weight_l1318_131840

theorem student_weight (student_weight sister_weight : ℝ) 
  (h1 : student_weight + sister_weight = 116)
  (h2 : student_weight - 5 = 2 * sister_weight) : 
  student_weight = 79 := by
sorry

end student_weight_l1318_131840


namespace distribute_volunteers_count_l1318_131833

/-- The number of ways to distribute 5 volunteers into 4 groups -/
def distribute_volunteers : ℕ :=
  Nat.choose 5 2 * Nat.factorial 4

/-- Theorem stating that the number of distribution methods is 240 -/
theorem distribute_volunteers_count : distribute_volunteers = 240 := by
  sorry

end distribute_volunteers_count_l1318_131833
