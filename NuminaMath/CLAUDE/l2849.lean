import Mathlib

namespace NUMINAMATH_CALUDE_max_decimal_places_is_14_complex_expression_decimal_places_l2849_284968

/-- The number of decimal places in 3.456789 -/
def decimal_places_a : ℕ := 6

/-- The number of decimal places in 6.78901234 -/
def decimal_places_b : ℕ := 8

/-- The expression ((10 ^ 5 * 3.456789) ^ 12) / (6.78901234 ^ 4)) ^ 9 -/
noncomputable def complex_expression : ℝ := 
  (((10 ^ 5 * 3.456789) ^ 12) / (6.78901234 ^ 4)) ^ 9

/-- The maximum number of decimal places in the result -/
def max_decimal_places : ℕ := decimal_places_a + decimal_places_b

theorem max_decimal_places_is_14 : 
  max_decimal_places = 14 := by sorry

theorem complex_expression_decimal_places : 
  ∃ (n : ℕ), n ≤ max_decimal_places ∧ 
  complex_expression * (10 ^ n) = ⌊complex_expression * (10 ^ n)⌋ := by sorry

end NUMINAMATH_CALUDE_max_decimal_places_is_14_complex_expression_decimal_places_l2849_284968


namespace NUMINAMATH_CALUDE_sum_equation_l2849_284919

theorem sum_equation (x y z : ℝ) (h1 : y = 3 * x) (h2 : z = 3 * y) : 
  2 * x + 3 * y + z = 20 * x := by
  sorry

end NUMINAMATH_CALUDE_sum_equation_l2849_284919


namespace NUMINAMATH_CALUDE_first_or_third_quadrant_set_l2849_284938

def first_or_third_quadrant (α : ℝ) : Prop :=
  (∃ k : ℤ, 2 * k * Real.pi < α ∧ α < 2 * k * Real.pi + Real.pi / 2) ∨
  (∃ k : ℤ, 2 * k * Real.pi + Real.pi < α ∧ α < 2 * k * Real.pi + 3 * Real.pi / 2)

theorem first_or_third_quadrant_set : 
  {α : ℝ | first_or_third_quadrant α} = 
  {α : ℝ | ∃ k : ℤ, 2 * k * Real.pi < α ∧ α < 2 * k * Real.pi + Real.pi / 2} ∪
  {α : ℝ | ∃ k : ℤ, 2 * k * Real.pi + Real.pi < α ∧ α < 2 * k * Real.pi + 3 * Real.pi / 2} :=
by sorry

end NUMINAMATH_CALUDE_first_or_third_quadrant_set_l2849_284938


namespace NUMINAMATH_CALUDE_sum_of_solutions_is_four_l2849_284987

theorem sum_of_solutions_is_four : ∃ (S : Finset Int), 
  (∀ x : Int, x ∈ S ↔ x^2 = 192 + x) ∧ (S.sum id = 4) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_is_four_l2849_284987


namespace NUMINAMATH_CALUDE_andreas_living_room_area_l2849_284935

theorem andreas_living_room_area :
  ∀ (floor_area carpet_area : ℝ),
    carpet_area = 4 * 9 →
    0.75 * floor_area = carpet_area →
    floor_area = 48 := by
  sorry

end NUMINAMATH_CALUDE_andreas_living_room_area_l2849_284935


namespace NUMINAMATH_CALUDE_right_triangle_side_length_l2849_284992

theorem right_triangle_side_length 
  (Q R S : ℝ × ℝ) 
  (right_angle_Q : (R.1 - Q.1) * (S.1 - Q.1) + (R.2 - Q.2) * (S.2 - Q.2) = 0) 
  (cos_R : (R.1 - Q.1) / Real.sqrt ((R.1 - S.1)^2 + (R.2 - S.2)^2) = 5/13) 
  (RS_length : (R.1 - S.1)^2 + (R.2 - S.2)^2 = 13^2) : 
  (Q.1 - S.1)^2 + (Q.2 - S.2)^2 = 12^2 := by
sorry


end NUMINAMATH_CALUDE_right_triangle_side_length_l2849_284992


namespace NUMINAMATH_CALUDE_solution_implies_a_value_l2849_284963

theorem solution_implies_a_value (a : ℝ) :
  (5 * a - 8 = 10 + 4 * a) → a = 18 := by
  sorry

end NUMINAMATH_CALUDE_solution_implies_a_value_l2849_284963


namespace NUMINAMATH_CALUDE_matching_socks_probability_l2849_284991

def black_socks : ℕ := 12
def white_socks : ℕ := 6
def blue_socks : ℕ := 9

def total_socks : ℕ := black_socks + white_socks + blue_socks

def matching_pairs : ℕ := (black_socks.choose 2) + (white_socks.choose 2) + (blue_socks.choose 2)

def total_pairs : ℕ := total_socks.choose 2

theorem matching_socks_probability :
  (matching_pairs : ℚ) / total_pairs = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_matching_socks_probability_l2849_284991


namespace NUMINAMATH_CALUDE_difference_of_squares_l2849_284960

theorem difference_of_squares (t : ℝ) : t^2 - 144 = (t - 12) * (t + 12) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l2849_284960


namespace NUMINAMATH_CALUDE_snowdrift_depth_change_l2849_284945

/-- Given a snowdrift with certain depth changes over four days, 
    calculate the amount of snow added on the fourth day. -/
theorem snowdrift_depth_change (initial_depth final_depth third_day_addition : ℕ) : 
  initial_depth = 20 →
  final_depth = 34 →
  third_day_addition = 6 →
  final_depth - (initial_depth / 2 + third_day_addition) = 18 := by
  sorry

#check snowdrift_depth_change

end NUMINAMATH_CALUDE_snowdrift_depth_change_l2849_284945


namespace NUMINAMATH_CALUDE_basement_water_pump_time_l2849_284902

/-- Proves that it takes 225 minutes to pump out water from a flooded basement --/
theorem basement_water_pump_time : 
  let basement_length : ℝ := 30
  let basement_width : ℝ := 40
  let water_depth_inches : ℝ := 12
  let num_pumps : ℕ := 4
  let pump_rate : ℝ := 10  -- gallons per minute
  let gallons_per_cubic_foot : ℝ := 7.5
  let inches_per_foot : ℝ := 12

  let water_depth_feet : ℝ := water_depth_inches / inches_per_foot
  let water_volume_cubic_feet : ℝ := basement_length * basement_width * water_depth_feet
  let water_volume_gallons : ℝ := water_volume_cubic_feet * gallons_per_cubic_foot
  let total_pump_rate : ℝ := num_pumps * pump_rate
  let pump_time_minutes : ℝ := water_volume_gallons / total_pump_rate

  pump_time_minutes = 225 := by
  sorry

end NUMINAMATH_CALUDE_basement_water_pump_time_l2849_284902


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l2849_284903

-- Define the polynomial
def P (x : ℝ) : ℝ := 4*x^4 + 4*x^3 - 11*x^2 - 6*x + 9

-- Define the divisor
def D (x : ℝ) : ℝ := (x - 1)^2

-- Define the quotient
def Q (x : ℝ) : ℝ := 4*x^2 + 12*x + 9

-- Theorem statement
theorem polynomial_divisibility :
  ∀ x : ℝ, P x = D x * Q x :=
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l2849_284903


namespace NUMINAMATH_CALUDE_octal_to_decimal_fraction_l2849_284977

theorem octal_to_decimal_fraction (c d : ℕ) : 
  (543 : ℕ) = 5 * 8^2 + 4 * 8^1 + 3 * 8^0 →
  (2 * 10 + c) * 10 + d = 5 * 8^2 + 4 * 8^1 + 3 * 8^0 →
  0 ≤ c ∧ c ≤ 9 →
  0 ≤ d ∧ d ≤ 9 →
  (c * d : ℚ) / 12 = 5 / 4 :=
by sorry

end NUMINAMATH_CALUDE_octal_to_decimal_fraction_l2849_284977


namespace NUMINAMATH_CALUDE_absent_men_calculation_l2849_284923

/-- Represents the number of men who became absent -/
def absentMen (totalMen originalDays actualDays : ℕ) : ℕ :=
  totalMen - (totalMen * originalDays) / actualDays

theorem absent_men_calculation (totalMen originalDays actualDays : ℕ) 
  (h1 : totalMen = 15)
  (h2 : originalDays = 8)
  (h3 : actualDays = 10)
  (h4 : totalMen > 0)
  (h5 : originalDays > 0)
  (h6 : actualDays > 0)
  (h7 : (totalMen * originalDays) % actualDays = 0) :
  absentMen totalMen originalDays actualDays = 3 := by
  sorry

#eval absentMen 15 8 10

end NUMINAMATH_CALUDE_absent_men_calculation_l2849_284923


namespace NUMINAMATH_CALUDE_max_value_inequality_l2849_284904

theorem max_value_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 * b^2 * c^2 * (a + b + c)) / ((a + b)^3 * (b + c)^3) ≤ 1/8 := by
  sorry

end NUMINAMATH_CALUDE_max_value_inequality_l2849_284904


namespace NUMINAMATH_CALUDE_polynomial_equality_sum_of_squares_l2849_284956

theorem polynomial_equality_sum_of_squares :
  ∀ (a b c d e f : ℤ),
  (∀ x : ℝ, 1728 * x^3 + 64 = (a * x^2 + b * x + c) * (d * x^2 + e * x + f)) →
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 23456 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_equality_sum_of_squares_l2849_284956


namespace NUMINAMATH_CALUDE_max_min_on_interval_l2849_284964

def f (x : ℝ) : ℝ := 3 * x^4 + 4 * x^3 + 34

theorem max_min_on_interval :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc (-2 : ℝ) 1, f x ≤ max) ∧
    (∃ x ∈ Set.Icc (-2 : ℝ) 1, f x = max) ∧
    (∀ x ∈ Set.Icc (-2 : ℝ) 1, min ≤ f x) ∧
    (∃ x ∈ Set.Icc (-2 : ℝ) 1, f x = min) ∧
    max = 50 ∧ min = 33 := by
  sorry

end NUMINAMATH_CALUDE_max_min_on_interval_l2849_284964


namespace NUMINAMATH_CALUDE_slips_with_three_l2849_284996

theorem slips_with_three (total_slips : ℕ) (value_a value_b : ℕ) (expected_value : ℚ) : 
  total_slips = 15 →
  value_a = 3 →
  value_b = 8 →
  expected_value = 5 →
  ∃ (slips_with_a : ℕ),
    slips_with_a ≤ total_slips ∧
    (slips_with_a : ℚ) / total_slips * value_a + 
    ((total_slips - slips_with_a) : ℚ) / total_slips * value_b = expected_value ∧
    slips_with_a = 9 := by
sorry

end NUMINAMATH_CALUDE_slips_with_three_l2849_284996


namespace NUMINAMATH_CALUDE_light_configurations_l2849_284912

/-- The number of rows and columns in the grid -/
def gridSize : Nat := 6

/-- The number of possible states for each switch (on or off) -/
def switchStates : Nat := 2

/-- The total number of different configurations of lights in the grid -/
def totalConfigurations : Nat := (switchStates ^ gridSize - 1) * (switchStates ^ gridSize - 1) + 1

/-- Theorem stating that the number of different configurations of lights is 3970 -/
theorem light_configurations :
  totalConfigurations = 3970 := by
  sorry

end NUMINAMATH_CALUDE_light_configurations_l2849_284912


namespace NUMINAMATH_CALUDE_vinegar_left_is_60_l2849_284995

/-- Represents the pickle-making scenario with given supplies and rules. -/
structure PickleScenario where
  jars : ℕ
  cucumbers : ℕ
  initial_vinegar : ℕ
  pickles_per_cucumber : ℕ
  pickles_per_jar : ℕ
  vinegar_per_jar : ℕ

/-- Calculates the amount of vinegar left after making pickles. -/
def vinegar_left (scenario : PickleScenario) : ℕ :=
  let total_pickles := scenario.cucumbers * scenario.pickles_per_cucumber
  let max_jarred_pickles := scenario.jars * scenario.pickles_per_jar
  let actual_jarred_pickles := min total_pickles max_jarred_pickles
  let jars_used := actual_jarred_pickles / scenario.pickles_per_jar
  let vinegar_used := jars_used * scenario.vinegar_per_jar
  scenario.initial_vinegar - vinegar_used

/-- Theorem stating that given the specific scenario, 60 oz of vinegar will be left. -/
theorem vinegar_left_is_60 :
  let scenario : PickleScenario := {
    jars := 4,
    cucumbers := 10,
    initial_vinegar := 100,
    pickles_per_cucumber := 6,
    pickles_per_jar := 12,
    vinegar_per_jar := 10
  }
  vinegar_left scenario = 60 := by sorry

end NUMINAMATH_CALUDE_vinegar_left_is_60_l2849_284995


namespace NUMINAMATH_CALUDE_triangle_angle_value_l2849_284940

theorem triangle_angle_value (A : ℝ) (h : 0 < A ∧ A < π) : 
  Real.sqrt 2 * Real.sin A = Real.sqrt (3 * Real.cos A) → A = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_value_l2849_284940


namespace NUMINAMATH_CALUDE_no_solution_to_inequality_system_l2849_284998

theorem no_solution_to_inequality_system :
  ¬∃ x : ℝ, (x / 6 + 7 / 2 > (3 * x + 29) / 5) ∧
            (x + 9 / 2 > x / 8) ∧
            (11 / 3 - x / 6 < (34 - 3 * x) / 5) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_to_inequality_system_l2849_284998


namespace NUMINAMATH_CALUDE_annes_speed_l2849_284969

/-- Given a distance of 6 miles traveled in 3 hours, prove that the speed is 2 miles per hour. -/
theorem annes_speed (distance : ℝ) (time : ℝ) (speed : ℝ) : 
  distance = 6 → time = 3 → speed = distance / time → speed = 2 := by sorry

end NUMINAMATH_CALUDE_annes_speed_l2849_284969


namespace NUMINAMATH_CALUDE_ribbon_length_difference_equals_side_length_l2849_284929

/-- Represents the dimensions of a box in centimeters -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the ribbon length for the first wrapping method -/
def ribbonLength1 (box : BoxDimensions) (bowLength : ℝ) : ℝ :=
  2 * box.length + 2 * box.width + 4 * box.height + bowLength

/-- Calculates the ribbon length for the second wrapping method -/
def ribbonLength2 (box : BoxDimensions) (bowLength : ℝ) : ℝ :=
  2 * box.length + 4 * box.width + 2 * box.height + bowLength

/-- Theorem stating that the difference in ribbon lengths equals one side of the box -/
theorem ribbon_length_difference_equals_side_length
  (box : BoxDimensions)
  (bowLength : ℝ)
  (h1 : box.length = 22)
  (h2 : box.width = 22)
  (h3 : box.height = 11)
  (h4 : bowLength = 24) :
  ribbonLength2 box bowLength - ribbonLength1 box bowLength = box.length := by
  sorry

end NUMINAMATH_CALUDE_ribbon_length_difference_equals_side_length_l2849_284929


namespace NUMINAMATH_CALUDE_pyarelal_loss_l2849_284932

theorem pyarelal_loss (p a : ℝ) (total_loss : ℝ) : 
  a = (1 / 9) * p → 
  total_loss = 900 → 
  (p / (p + a)) * total_loss = 810 :=
by sorry

end NUMINAMATH_CALUDE_pyarelal_loss_l2849_284932


namespace NUMINAMATH_CALUDE_rotate_180_of_A_l2849_284917

/-- Rotate a point 180 degrees about the origin -/
def rotate_180 (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

/-- The original point A -/
def A : ℝ × ℝ := (-3, 2)

theorem rotate_180_of_A :
  rotate_180 A = (3, -2) := by
  sorry

end NUMINAMATH_CALUDE_rotate_180_of_A_l2849_284917


namespace NUMINAMATH_CALUDE_james_money_calculation_l2849_284972

/-- Given 3 bills of $20 each and $75 already in a wallet, prove that the total amount is $135 -/
theorem james_money_calculation :
  let bills_count : ℕ := 3
  let bill_value : ℕ := 20
  let initial_wallet_amount : ℕ := 75
  bills_count * bill_value + initial_wallet_amount = 135 :=
by sorry

end NUMINAMATH_CALUDE_james_money_calculation_l2849_284972


namespace NUMINAMATH_CALUDE_picnic_bread_slices_l2849_284952

/-- Calculate the total number of bread slices needed for a picnic --/
theorem picnic_bread_slices :
  let total_people : ℕ := 6
  let pb_people : ℕ := 4
  let tuna_people : ℕ := 3
  let turkey_people : ℕ := 2
  let pb_sandwiches_per_person : ℕ := 2
  let tuna_sandwiches_per_person : ℕ := 3
  let turkey_sandwiches_per_person : ℕ := 1
  let pb_slices_per_sandwich : ℕ := 2
  let tuna_slices_per_sandwich : ℕ := 3
  let turkey_slices_per_sandwich : ℚ := 3/2

  let total_pb_sandwiches := pb_people * pb_sandwiches_per_person
  let total_tuna_sandwiches := tuna_people * tuna_sandwiches_per_person
  let total_turkey_sandwiches := turkey_people * turkey_sandwiches_per_person

  let total_pb_slices := total_pb_sandwiches * pb_slices_per_sandwich
  let total_tuna_slices := total_tuna_sandwiches * tuna_slices_per_sandwich
  let total_turkey_slices := (total_turkey_sandwiches : ℚ) * turkey_slices_per_sandwich

  (total_pb_slices : ℚ) + (total_tuna_slices : ℚ) + total_turkey_slices = 46
  := by sorry

end NUMINAMATH_CALUDE_picnic_bread_slices_l2849_284952


namespace NUMINAMATH_CALUDE_a_union_b_iff_c_l2849_284909

-- Define the sets A, B, and C
def A : Set ℝ := {x | x > 2}
def B : Set ℝ := {x | x < 0}
def C : Set ℝ := {x | x * (x - 2) > 0}

-- State the theorem
theorem a_union_b_iff_c : ∀ x : ℝ, x ∈ (A ∪ B) ↔ x ∈ C := by
  sorry

end NUMINAMATH_CALUDE_a_union_b_iff_c_l2849_284909


namespace NUMINAMATH_CALUDE_chris_sick_one_week_l2849_284921

/-- Calculates the number of weeks Chris got sick based on Cathy's work hours -/
def weeks_chris_sick (hours_per_week : ℕ) (total_weeks : ℕ) (cathy_total_hours : ℕ) : ℕ :=
  (cathy_total_hours - (hours_per_week * total_weeks)) / hours_per_week

/-- Proves that Chris got sick for 1 week given the conditions in the problem -/
theorem chris_sick_one_week :
  let hours_per_week : ℕ := 20
  let months : ℕ := 2
  let weeks_per_month : ℕ := 4
  let total_weeks : ℕ := months * weeks_per_month
  let cathy_total_hours : ℕ := 180
  weeks_chris_sick hours_per_week total_weeks cathy_total_hours = 1 := by
  sorry

#eval weeks_chris_sick 20 8 180

end NUMINAMATH_CALUDE_chris_sick_one_week_l2849_284921


namespace NUMINAMATH_CALUDE_ten_books_left_to_read_l2849_284980

/-- The number of books left to read in the 'crazy silly school' series -/
def books_left_to_read (total_books read_books : ℕ) : ℕ :=
  total_books - read_books

/-- Theorem stating that there are 10 books left to read -/
theorem ten_books_left_to_read :
  books_left_to_read 22 12 = 10 := by
  sorry

#eval books_left_to_read 22 12

end NUMINAMATH_CALUDE_ten_books_left_to_read_l2849_284980


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l2849_284931

/-- Lateral surface area of a cone with given base radius and volume -/
theorem cone_lateral_surface_area (r h : ℝ) (hr : r = 3) (hv : (1/3) * π * r^2 * h = 12 * π) :
  π * r * (Real.sqrt (r^2 + h^2)) = 15 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l2849_284931


namespace NUMINAMATH_CALUDE_arctan_equation_solution_l2849_284986

theorem arctan_equation_solution (x : ℝ) :
  Real.arctan (1 / x) + Real.arctan (1 / x^2) = π / 3 →
  x = (1 + Real.sqrt (13 + 4 * Real.sqrt 3)) / (2 * Real.sqrt 3) ∨
  x = (1 - Real.sqrt (13 + 4 * Real.sqrt 3)) / (2 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_arctan_equation_solution_l2849_284986


namespace NUMINAMATH_CALUDE_remaining_balloons_l2849_284965

-- Define the type for balloon labels
inductive BalloonLabel
| A | B | C | D | E | F | G | H | I | J | K | L

-- Define the function to get the next balloon to pop
def nextBalloon (current : BalloonLabel) : BalloonLabel :=
  match current with
  | BalloonLabel.A => BalloonLabel.D
  | BalloonLabel.B => BalloonLabel.E
  | BalloonLabel.C => BalloonLabel.F
  | BalloonLabel.D => BalloonLabel.G
  | BalloonLabel.E => BalloonLabel.H
  | BalloonLabel.F => BalloonLabel.I
  | BalloonLabel.G => BalloonLabel.J
  | BalloonLabel.H => BalloonLabel.K
  | BalloonLabel.I => BalloonLabel.L
  | BalloonLabel.J => BalloonLabel.A
  | BalloonLabel.K => BalloonLabel.B
  | BalloonLabel.L => BalloonLabel.C

-- Define the function to pop balloons
def popBalloons (start : BalloonLabel) (n : Nat) : List BalloonLabel :=
  if n = 0 then []
  else start :: popBalloons (nextBalloon (nextBalloon start)) (n - 1)

-- Theorem statement
theorem remaining_balloons :
  popBalloons BalloonLabel.C 10 = [BalloonLabel.C, BalloonLabel.F, BalloonLabel.I, BalloonLabel.L, BalloonLabel.D, BalloonLabel.H, BalloonLabel.A, BalloonLabel.G, BalloonLabel.B, BalloonLabel.K] ∧
  (∀ b : BalloonLabel, b ∉ popBalloons BalloonLabel.C 10 → b = BalloonLabel.E ∨ b = BalloonLabel.J) :=
by sorry


end NUMINAMATH_CALUDE_remaining_balloons_l2849_284965


namespace NUMINAMATH_CALUDE_no_real_zeros_l2849_284933

theorem no_real_zeros (x : ℝ) : x^6 - x^5 + x^4 - x^3 + x^2 - x + 3/4 ≥ 3/8 := by
  sorry

end NUMINAMATH_CALUDE_no_real_zeros_l2849_284933


namespace NUMINAMATH_CALUDE_songs_downloaded_l2849_284979

def internet_speed : ℝ := 20
def download_time : ℝ := 0.5
def song_size : ℝ := 5

theorem songs_downloaded : 
  ⌊(internet_speed * download_time * 3600) / song_size⌋ = 7200 := by sorry

end NUMINAMATH_CALUDE_songs_downloaded_l2849_284979


namespace NUMINAMATH_CALUDE_min_cost_for_family_trip_l2849_284949

/-- Represents the ticket prices in rubles -/
structure TicketPrices where
  adult_single : ℕ
  child_single : ℕ
  day_pass_single : ℕ
  day_pass_group : ℕ
  three_day_pass_single : ℕ
  three_day_pass_group : ℕ

/-- Calculates the minimum cost for a family's subway tickets -/
def min_family_ticket_cost (prices : TicketPrices) (days : ℕ) (trips_per_day : ℕ) (adults : ℕ) (children : ℕ) : ℕ :=
  sorry

/-- The theorem stating the minimum cost for the given family and conditions -/
theorem min_cost_for_family_trip (prices : TicketPrices) 
  (h1 : prices.adult_single = 40)
  (h2 : prices.child_single = 20)
  (h3 : prices.day_pass_single = 350)
  (h4 : prices.day_pass_group = 1500)
  (h5 : prices.three_day_pass_single = 900)
  (h6 : prices.three_day_pass_group = 3500) :
  min_family_ticket_cost prices 5 10 2 2 = 5200 :=
by sorry

end NUMINAMATH_CALUDE_min_cost_for_family_trip_l2849_284949


namespace NUMINAMATH_CALUDE_perpendicular_slope_l2849_284967

/-- Given a line with equation 5x - 2y = 10, 
    the slope of the perpendicular line is -2/5 -/
theorem perpendicular_slope (x y : ℝ) :
  (5 * x - 2 * y = 10) → 
  (∃ m : ℝ, m = -2/5 ∧ 
    ∀ (x₁ y₁ x₂ y₂ : ℝ), 
      (5 * x₁ - 2 * y₁ = 10) → 
      (5 * x₂ - 2 * y₂ = 10) → 
      x₁ ≠ x₂ → 
      m * ((x₂ - x₁) / (y₂ - y₁)) = -1) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_slope_l2849_284967


namespace NUMINAMATH_CALUDE_symmetry_implies_phi_value_l2849_284962

theorem symmetry_implies_phi_value (φ : Real) :
  φ ∈ Set.Icc 0 Real.pi →
  (∀ x : Real, 3 * Real.cos (x + φ) - 1 = 3 * Real.cos ((2 * Real.pi / 3 - x) + φ) - 1) →
  φ = 2 * Real.pi / 3 := by
sorry

end NUMINAMATH_CALUDE_symmetry_implies_phi_value_l2849_284962


namespace NUMINAMATH_CALUDE_gas_price_increase_l2849_284946

/-- Given two successive price increases in gas, where the second increase is 20%,
    and a driver needs to reduce gas consumption by 35.89743589743589% to keep
    expenditure constant, prove that the first price increase was approximately 30%. -/
theorem gas_price_increase (initial_price : ℝ) (initial_consumption : ℝ) :
  initial_price > 0 →
  initial_consumption > 0 →
  ∃ (first_increase : ℝ),
    (initial_price * initial_consumption =
      initial_price * (1 + first_increase / 100) * 1.20 * initial_consumption * (1 - 35.89743589743589 / 100)) ∧
    (abs (first_increase - 30) < 0.00001) := by
  sorry

end NUMINAMATH_CALUDE_gas_price_increase_l2849_284946


namespace NUMINAMATH_CALUDE_hiking_time_theorem_l2849_284959

/-- Calculates the total time for a hiker to return to the starting point given their hiking rate and distances. -/
def total_hiking_time (rate : ℝ) (initial_distance : ℝ) (total_distance : ℝ) : ℝ :=
  let additional_distance := total_distance - initial_distance
  let time_additional := additional_distance * rate
  let time_return := total_distance * rate
  time_additional + time_return

/-- Theorem stating that under given conditions, the total hiking time is 40 minutes. -/
theorem hiking_time_theorem (rate : ℝ) (initial_distance : ℝ) (total_distance : ℝ) :
  rate = 12 →
  initial_distance = 2.75 →
  total_distance = 3.041666666666667 →
  total_hiking_time rate initial_distance total_distance = 40 := by
  sorry

#eval total_hiking_time 12 2.75 3.041666666666667

end NUMINAMATH_CALUDE_hiking_time_theorem_l2849_284959


namespace NUMINAMATH_CALUDE_geometric_sequence_m_value_l2849_284984

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_m_value
  (a : ℕ → ℝ)
  (q : ℝ)
  (h1 : geometric_sequence a q)
  (h2 : q ≠ 1 ∧ q ≠ -1)
  (h3 : a 1 = -1)
  (h4 : ∃ m : ℕ, a m = a 1 * a 2 * a 3 * a 4 * a 5) :
  ∃ m : ℕ, m = 11 ∧ a m = a 1 * a 2 * a 3 * a 4 * a 5 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_m_value_l2849_284984


namespace NUMINAMATH_CALUDE_x_fifth_minus_ten_x_equals_213_l2849_284913

theorem x_fifth_minus_ten_x_equals_213 (x : ℝ) (h : x = 3) : x^5 - 10*x = 213 := by
  sorry

end NUMINAMATH_CALUDE_x_fifth_minus_ten_x_equals_213_l2849_284913


namespace NUMINAMATH_CALUDE_finite_solutions_egyptian_fraction_l2849_284908

theorem finite_solutions_egyptian_fraction :
  (∃ (S : Set (ℕ+ × ℕ+ × ℕ+)), Finite S ∧
    ∀ (a b c : ℕ+), (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c = (1 : ℚ) / 1983 ↔ (a, b, c) ∈ S) :=
by sorry

end NUMINAMATH_CALUDE_finite_solutions_egyptian_fraction_l2849_284908


namespace NUMINAMATH_CALUDE_circle_area_quadrupled_l2849_284910

theorem circle_area_quadrupled (r n : ℝ) : 
  (r > 0) → (n > 0) → (π * (r + n)^2 = 4 * π * r^2) → r = n / 3 :=
by sorry

end NUMINAMATH_CALUDE_circle_area_quadrupled_l2849_284910


namespace NUMINAMATH_CALUDE_juvy_chives_count_l2849_284970

/-- Calculates the number of chives planted in Juvy's garden. -/
def chives_count (total_rows : ℕ) (plants_per_row : ℕ) (parsley_rows : ℕ) (rosemary_rows : ℕ) : ℕ :=
  (total_rows - (parsley_rows + rosemary_rows)) * plants_per_row

/-- Theorem stating that the number of chives Juvy will plant is 150. -/
theorem juvy_chives_count :
  chives_count 20 10 3 2 = 150 := by
  sorry

end NUMINAMATH_CALUDE_juvy_chives_count_l2849_284970


namespace NUMINAMATH_CALUDE_sum_of_first_six_primes_mod_seventh_prime_l2849_284961

def first_six_primes : List Nat := [2, 3, 5, 7, 11, 13]
def seventh_prime : Nat := 17

theorem sum_of_first_six_primes_mod_seventh_prime : 
  (first_six_primes.sum) % seventh_prime = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_first_six_primes_mod_seventh_prime_l2849_284961


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l2849_284934

theorem polynomial_division_theorem (a b : ℝ) : 
  (∃ (P : ℝ → ℝ), (fun X => a * X^4 + b * X^3 + 1) = fun X => (X - 1)^2 * P X) → 
  a = 3 ∧ b = -4 := by
sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l2849_284934


namespace NUMINAMATH_CALUDE_group_communication_l2849_284983

theorem group_communication (n k : ℕ) : 
  n > 0 → 
  k > 0 → 
  k * (n - 1) * n = 440 → 
  n = 11 := by
sorry

end NUMINAMATH_CALUDE_group_communication_l2849_284983


namespace NUMINAMATH_CALUDE_solve_ice_cubes_problem_l2849_284950

def ice_cubes_problem (x : ℕ) : Prop :=
  let glass_ice := x
  let pitcher_ice := 2 * x
  let total_ice := glass_ice + pitcher_ice
  let tray_capacity := 2 * 12
  total_ice = tray_capacity

theorem solve_ice_cubes_problem :
  ∃ x : ℕ, ice_cubes_problem x ∧ x = 8 := by
  sorry

end NUMINAMATH_CALUDE_solve_ice_cubes_problem_l2849_284950


namespace NUMINAMATH_CALUDE_sally_grew_five_onions_l2849_284939

/-- The number of onions grown by Sally, given the number of onions grown by Sara and Fred, and the total number of onions. -/
def sallys_onions (sara_onions fred_onions total_onions : ℕ) : ℕ :=
  total_onions - (sara_onions + fred_onions)

/-- Theorem stating that Sally grew 5 onions given the conditions in the problem. -/
theorem sally_grew_five_onions :
  sallys_onions 4 9 18 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sally_grew_five_onions_l2849_284939


namespace NUMINAMATH_CALUDE_price_of_49_dozens_l2849_284971

/-- Calculates the price of a given number of dozens of apples at a new price -/
def price_of_apples (initial_price : ℝ) (new_price : ℝ) (dozens : ℕ) : ℝ :=
  dozens * new_price

/-- Theorem: The price of 49 dozens of apples at the new price is 49 times the new price -/
theorem price_of_49_dozens 
  (initial_price : ℝ) 
  (new_price : ℝ) 
  (h1 : initial_price = 1517.25)
  (h2 : new_price = 2499) :
  price_of_apples initial_price new_price 49 = 49 * new_price :=
by sorry

end NUMINAMATH_CALUDE_price_of_49_dozens_l2849_284971


namespace NUMINAMATH_CALUDE_percentage_less_than_l2849_284907

theorem percentage_less_than (x y : ℝ) (h : x = 12 * y) :
  (x - y) / x * 100 = (11 / 12) * 100 :=
sorry

end NUMINAMATH_CALUDE_percentage_less_than_l2849_284907


namespace NUMINAMATH_CALUDE_n_times_n_plus_one_divisible_by_two_l2849_284930

theorem n_times_n_plus_one_divisible_by_two (n : ℕ) (h : 1 ≤ n ∧ n ≤ 99) : 
  2 ∣ (n * (n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_n_times_n_plus_one_divisible_by_two_l2849_284930


namespace NUMINAMATH_CALUDE_james_purchase_cost_l2849_284955

/-- Calculates the total cost of James' purchase --/
def totalCost (bedFramePrice bedPrice bedsideTablePrice bedFrameDiscount bedDiscount bedsideTableDiscount salesTax : ℝ) : ℝ :=
  let discountedBedFramePrice := bedFramePrice * (1 - bedFrameDiscount)
  let discountedBedPrice := bedPrice * (1 - bedDiscount)
  let discountedBedsideTablePrice := bedsideTablePrice * (1 - bedsideTableDiscount)
  let totalDiscountedPrice := discountedBedFramePrice + discountedBedPrice + discountedBedsideTablePrice
  totalDiscountedPrice * (1 + salesTax)

/-- Theorem stating the total cost of James' purchase --/
theorem james_purchase_cost :
  totalCost 75 750 120 0.20 0.20 0.15 0.085 = 826.77 := by
  sorry

end NUMINAMATH_CALUDE_james_purchase_cost_l2849_284955


namespace NUMINAMATH_CALUDE_box_volume_increase_l2849_284976

theorem box_volume_increase (l w h : ℝ) 
  (volume : l * w * h = 5000)
  (surface_area : 2 * (l * w + w * h + h * l) = 1800)
  (edge_sum : 4 * (l + w + h) = 240) :
  (l + 2) * (w + 2) * (h + 2) = 7048 := by
sorry

end NUMINAMATH_CALUDE_box_volume_increase_l2849_284976


namespace NUMINAMATH_CALUDE_greatest_common_divisor_under_30_l2849_284944

theorem greatest_common_divisor_under_30 : ∃ (n : ℕ), n = 18 ∧ 
  n ∣ 540 ∧ n < 30 ∧ n ∣ 180 ∧ 
  ∀ (m : ℕ), m ∣ 540 → m < 30 → m ∣ 180 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_under_30_l2849_284944


namespace NUMINAMATH_CALUDE_total_songs_two_days_l2849_284905

-- Define the number of songs listened to yesterday
def songs_yesterday : ℕ := 9

-- Define the relationship between yesterday's and today's songs
def song_relationship (x : ℕ) : Prop :=
  songs_yesterday = 2 * (x.sqrt : ℕ) - 5

-- Theorem to prove
theorem total_songs_two_days (x : ℕ) 
  (h : song_relationship x) : songs_yesterday + x = 58 := by
  sorry

end NUMINAMATH_CALUDE_total_songs_two_days_l2849_284905


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2849_284916

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + b = 1) :
  (1 / a + 1 / b) ≥ 3 + 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2849_284916


namespace NUMINAMATH_CALUDE_product_sum_relation_l2849_284993

theorem product_sum_relation (a b : ℝ) : 
  (a * b = 2 * (a + b) + 12) → (b = 10) → (b - a = 6) := by
  sorry

end NUMINAMATH_CALUDE_product_sum_relation_l2849_284993


namespace NUMINAMATH_CALUDE_lowest_unique_score_l2849_284957

/-- The scoring function for the modified AHSME -/
def score (c w : ℕ) : ℤ := 30 + 4 * c - 2 * w

/-- Predicate to check if a score uniquely determines c and w -/
def uniquely_determines (s : ℤ) : Prop :=
  ∃! (c w : ℕ), score c w = s ∧ c + w ≤ 30

theorem lowest_unique_score : 
  (∀ s : ℤ, 100 < s → s < 116 → ¬ uniquely_determines s) ∧
  uniquely_determines 116 := by
  sorry

end NUMINAMATH_CALUDE_lowest_unique_score_l2849_284957


namespace NUMINAMATH_CALUDE_reciprocal_equality_implies_equality_l2849_284975

theorem reciprocal_equality_implies_equality (x y : ℝ) (h : x ≠ 0) (k : y ≠ 0) : 
  1 / x = 1 / y → x = y := by
sorry

end NUMINAMATH_CALUDE_reciprocal_equality_implies_equality_l2849_284975


namespace NUMINAMATH_CALUDE_second_hole_depth_l2849_284941

/-- Represents the depth of a hole dug by workers -/
def hole_depth (workers : ℕ) (hours : ℕ) (rate : ℚ) : ℚ :=
  (workers * hours : ℚ) * rate

theorem second_hole_depth :
  let initial_workers : ℕ := 45
  let initial_hours : ℕ := 8
  let initial_depth : ℚ := 30
  let extra_workers : ℕ := 65
  let second_hours : ℕ := 6
  
  let total_workers : ℕ := initial_workers + extra_workers
  let digging_rate : ℚ := initial_depth / (initial_workers * initial_hours)
  
  hole_depth total_workers second_hours digging_rate = 55 := by
  sorry


end NUMINAMATH_CALUDE_second_hole_depth_l2849_284941


namespace NUMINAMATH_CALUDE_equation_solution_l2849_284918

/-- Given the equation P = s / (1 + k + m)^n, prove that n = log(s/P) / log(1 + k + m) -/
theorem equation_solution (P s k m n : ℝ) (h : P = s / (1 + k + m)^n) :
  n = Real.log (s / P) / Real.log (1 + k + m) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2849_284918


namespace NUMINAMATH_CALUDE_remaining_files_l2849_284925

theorem remaining_files (music_files video_files deleted_files : ℕ) 
  (h1 : music_files = 13)
  (h2 : video_files = 30)
  (h3 : deleted_files = 10) :
  music_files + video_files - deleted_files = 33 := by
  sorry

end NUMINAMATH_CALUDE_remaining_files_l2849_284925


namespace NUMINAMATH_CALUDE_cos_sin_sum_equals_half_l2849_284911

theorem cos_sin_sum_equals_half : 
  Real.cos (π / 4) * Real.cos (π / 12) - Real.sin (π / 4) * Real.sin (π / 12) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_sum_equals_half_l2849_284911


namespace NUMINAMATH_CALUDE_inserted_numbers_sum_l2849_284915

theorem inserted_numbers_sum : ∃! (a b : ℝ), 
  0 < a ∧ 0 < b ∧ 
  4 < a ∧ a < b ∧ b < 16 ∧ 
  (∃ r : ℝ, 0 < r ∧ a = 4 * r ∧ b = 4 * r^2) ∧
  (∃ d : ℝ, b = a + d ∧ 16 = b + d) ∧
  a + b = 24 := by
sorry

end NUMINAMATH_CALUDE_inserted_numbers_sum_l2849_284915


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2849_284943

theorem complex_equation_solution (i : ℂ) (m : ℝ) :
  i * i = -1 →
  (1 - m * i) / (i^3) = 1 + i →
  m = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2849_284943


namespace NUMINAMATH_CALUDE_min_beta_delta_sum_l2849_284966

open Complex

/-- A complex-valued function with specific properties -/
def f (β δ : ℂ) (z : ℂ) : ℂ := (3 + 2*I)*z^2 + β*z + δ

/-- The theorem stating the minimum value of |β| + |δ| -/
theorem min_beta_delta_sum :
  ∃ (β δ : ℂ), 
    (∀ (β' δ' : ℂ), (f β' δ' (1 + I)).im = 0 ∧ (f β' δ' (-I)).im = 0 → 
      Complex.abs β + Complex.abs δ ≤ Complex.abs β' + Complex.abs δ') ∧
    Complex.abs β + Complex.abs δ = Real.sqrt 5 + 3 := by
  sorry

end NUMINAMATH_CALUDE_min_beta_delta_sum_l2849_284966


namespace NUMINAMATH_CALUDE_dress_price_l2849_284974

/-- The final price of a dress after discounts and tax -/
def final_price (d : ℝ) : ℝ :=
  let sale_price := d * (1 - 0.25)
  let staff_price := sale_price * (1 - 0.20)
  let coupon_price := staff_price * (1 - 0.10)
  coupon_price * (1 + 0.08)

/-- Theorem stating the final price of the dress -/
theorem dress_price (d : ℝ) :
  final_price d = 0.5832 * d := by
  sorry

end NUMINAMATH_CALUDE_dress_price_l2849_284974


namespace NUMINAMATH_CALUDE_special_gp_common_ratio_l2849_284988

/-- A geometric progression where each term, starting from the third, 
    is equal to the sum of the two preceding terms. -/
def SpecialGeometricProgression (u : ℕ → ℝ) : Prop :=
  ∃ (q : ℝ), ∀ (n : ℕ),
    u (n + 1) = u n * q ∧ 
    u (n + 2) = u (n + 1) + u n

/-- The common ratio of a special geometric progression 
    is either (1 + √5) / 2 or (1 - √5) / 2. -/
theorem special_gp_common_ratio 
  (u : ℕ → ℝ) (h : SpecialGeometricProgression u) : 
  ∃ (q : ℝ), (∀ (n : ℕ), u (n + 1) = u n * q) ∧ 
    (q = (1 + Real.sqrt 5) / 2 ∨ q = (1 - Real.sqrt 5) / 2) := by
  sorry

end NUMINAMATH_CALUDE_special_gp_common_ratio_l2849_284988


namespace NUMINAMATH_CALUDE_M_superset_P_l2849_284924

-- Define the set M
def M : Set ℝ := {y | ∃ x, y = x^2 - 4}

-- Define the set P
def P : Set ℝ := {x | 2 ≤ x ∧ x ≤ 4}

-- Define the transformation function
def f (x : ℝ) : ℝ := x^2 - 4

-- Theorem statement
theorem M_superset_P : M ⊇ f '' P := by sorry

end NUMINAMATH_CALUDE_M_superset_P_l2849_284924


namespace NUMINAMATH_CALUDE_quadratic_roots_theorem_l2849_284982

theorem quadratic_roots_theorem (b c : ℝ) : 
  ({1, 2} : Set ℝ) = {x | x^2 + b*x + c = 0} → b = -3 ∧ c = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_theorem_l2849_284982


namespace NUMINAMATH_CALUDE_power_of_complex_root_of_unity_l2849_284900

open Complex

theorem power_of_complex_root_of_unity : ((1 - I) / (Real.sqrt 2)) ^ 20 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_of_complex_root_of_unity_l2849_284900


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l2849_284926

/-- An arithmetic sequence with first term a₁ and common ratio q -/
structure ArithmeticSequence (α : Type*) [Semiring α] where
  a₁ : α
  q : α

/-- The nth term of an arithmetic sequence -/
def ArithmeticSequence.nthTerm {α : Type*} [Semiring α] (seq : ArithmeticSequence α) (n : ℕ) : α :=
  seq.a₁ * seq.q ^ (n - 1)

/-- Theorem: The general term of an arithmetic sequence -/
theorem arithmetic_sequence_general_term {α : Type*} [Semiring α] (seq : ArithmeticSequence α) (n : ℕ) :
  seq.nthTerm n = seq.a₁ * seq.q ^ (n - 1) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l2849_284926


namespace NUMINAMATH_CALUDE_right_triangle_area_l2849_284958

/-- A right-angled triangle with specific properties -/
structure RightTriangle where
  -- The legs of the triangle
  a : ℝ
  b : ℝ
  -- The hypotenuse of the triangle
  c : ℝ
  -- Conditions
  right_angle : a^2 + b^2 = c^2
  perimeter : a + b + c = 2 + Real.sqrt 6
  hypotenuse : c = 2
  median : (a + b) / 2 = 1

/-- The area of a right-angled triangle with the given properties is 1/2 -/
theorem right_triangle_area (t : RightTriangle) : (t.a * t.b) / 2 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2849_284958


namespace NUMINAMATH_CALUDE_length_of_AB_l2849_284978

-- Define the line l: kx + y - 2 = 0
def line_l (k : ℝ) (x y : ℝ) : Prop := k * x + y - 2 = 0

-- Define the circle C: x² + y² - 6x + 2y + 9 = 0
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 2*y + 9 = 0

-- Define that line l is the axis of symmetry for circle C
def is_axis_of_symmetry (k : ℝ) : Prop := 
  ∀ x y : ℝ, line_l k x y → (∃ x' y' : ℝ, circle_C x' y' ∧ 
    ((x - x')^2 + (y - y')^2 = (x' - 3)^2 + (y' + 1)^2))

-- Define point A
def point_A (k : ℝ) : ℝ × ℝ := (0, k)

-- Define that there exists a tangent line from A to circle C
def exists_tangent (k : ℝ) : Prop :=
  ∃ x y : ℝ, circle_C x y ∧ 
    ((x - 0)^2 + (y - k)^2) * ((x - 3)^2 + (y + 1)^2) = 1

-- Theorem statement
theorem length_of_AB (k : ℝ) : 
  is_axis_of_symmetry k → exists_tangent k → 
  ∃ x y : ℝ, circle_C x y ∧ 
    Real.sqrt ((x - 0)^2 + (y - k)^2) = 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_length_of_AB_l2849_284978


namespace NUMINAMATH_CALUDE_remainder_problem_l2849_284920

theorem remainder_problem (k : ℕ) 
  (h1 : k > 0) 
  (h2 : k % 5 = 2) 
  (h3 : k % 6 = 5) 
  (h4 : k < 42) : 
  k % 7 = 3 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l2849_284920


namespace NUMINAMATH_CALUDE_divisibility_of_expression_l2849_284922

theorem divisibility_of_expression (m : ℕ) 
  (h1 : m > 0) 
  (h2 : Odd m) 
  (h3 : ¬(3 ∣ m)) : 
  112 ∣ (Int.floor (4^m - (2 + Real.sqrt 2)^m)) := by
sorry

end NUMINAMATH_CALUDE_divisibility_of_expression_l2849_284922


namespace NUMINAMATH_CALUDE_min_abs_z_l2849_284914

theorem min_abs_z (z : ℂ) (h : Complex.abs (z - 7) + Complex.abs (z - 6*I) = 15) :
  ∃ (w : ℂ), Complex.abs w = 14/5 ∧ ∀ (v : ℂ), Complex.abs (v - 7) + Complex.abs (v - 6*I) = 15 → Complex.abs v ≥ Complex.abs w :=
sorry

end NUMINAMATH_CALUDE_min_abs_z_l2849_284914


namespace NUMINAMATH_CALUDE_percentage_problem_l2849_284927

theorem percentage_problem (p : ℝ) (x : ℝ) 
  (h1 : (p / 100) * x = 300)
  (h2 : (120 / 100) * x = 1800) : p = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2849_284927


namespace NUMINAMATH_CALUDE_middle_zero_between_zero_and_one_l2849_284954

/-- The cubic function f(x) = x^3 - 4x + a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 4*x + a

/-- Theorem: For 0 < a < 2, if f(x) has three zeros x₁ < x₂ < x₃, then 0 < x₂ < 1 -/
theorem middle_zero_between_zero_and_one (a : ℝ) (x₁ x₂ x₃ : ℝ)
  (ha : 0 < a ∧ a < 2)
  (hzeros : f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0)
  (horder : x₁ < x₂ ∧ x₂ < x₃) :
  0 < x₂ ∧ x₂ < 1 := by
  sorry


end NUMINAMATH_CALUDE_middle_zero_between_zero_and_one_l2849_284954


namespace NUMINAMATH_CALUDE_equation_positive_root_l2849_284981

theorem equation_positive_root (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ (6 / (x - 2) - 1 = a * x / (2 - x))) → a = -3 := by
  sorry

end NUMINAMATH_CALUDE_equation_positive_root_l2849_284981


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l2849_284994

theorem geometric_series_ratio (a : ℝ) (r : ℝ) : 
  (∃ (S : ℝ), S = a / (1 - r) ∧ S = 24) →
  (∃ (S_odd : ℝ), S_odd = a * r / (1 - r^2) ∧ S_odd = 8) →
  r = 1/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_ratio_l2849_284994


namespace NUMINAMATH_CALUDE_div_point_one_eq_mul_ten_l2849_284951

theorem div_point_one_eq_mul_ten (a : ℝ) : a / 0.1 = a * 10 := by sorry

end NUMINAMATH_CALUDE_div_point_one_eq_mul_ten_l2849_284951


namespace NUMINAMATH_CALUDE_smallest_non_square_units_digit_l2849_284948

def is_square_units_digit (d : ℕ) : Prop :=
  ∃ n : ℕ, n^2 % 10 = d

theorem smallest_non_square_units_digit :
  (∀ d < 2, is_square_units_digit d) ∧
  ¬(is_square_units_digit 2) ∧
  (∀ d ≥ 2, ¬(is_square_units_digit d) → d ≥ 2) :=
sorry

end NUMINAMATH_CALUDE_smallest_non_square_units_digit_l2849_284948


namespace NUMINAMATH_CALUDE_ellipse_max_major_axis_l2849_284997

theorem ellipse_max_major_axis 
  (a b : ℝ) 
  (h_ab : a > b ∧ b > 0) 
  (e : ℝ) 
  (h_e : e ∈ Set.Icc (1/2) (Real.sqrt 2 / 2)) 
  (h_perp : ∀ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁^2/a^2 + y₁^2/b^2 = 1 → 
    y₁ = -x₁ + 1 → 
    x₂^2/a^2 + y₂^2/b^2 = 1 → 
    y₂ = -x₂ + 1 → 
    x₁*x₂ + y₁*y₂ = 0) 
  (h_ecc : e^2 = 1 - b^2/a^2) :
  ∃ (max_axis : ℝ), max_axis = Real.sqrt 6 ∧ 
    ∀ (axis : ℝ), axis = 2*a → axis ≤ max_axis :=
sorry

end NUMINAMATH_CALUDE_ellipse_max_major_axis_l2849_284997


namespace NUMINAMATH_CALUDE_smallest_b_undefined_inverse_b_330_satisfies_conditions_smallest_b_is_330_l2849_284947

theorem smallest_b_undefined_inverse (b : ℕ) : b > 0 ∧ 
  (∀ x : ℕ, x * b % 36 ≠ 1) ∧ 
  (∀ y : ℕ, y * b % 55 ≠ 1) → 
  b ≥ 330 := by
  sorry

theorem b_330_satisfies_conditions : 
  (∀ x : ℕ, x * 330 % 36 ≠ 1) ∧ 
  (∀ y : ℕ, y * 330 % 55 ≠ 1) := by
  sorry

theorem smallest_b_is_330 : 
  ∃ b : ℕ, b > 0 ∧ 
  (∀ x : ℕ, x * b % 36 ≠ 1) ∧ 
  (∀ y : ℕ, y * b % 55 ≠ 1) ∧ 
  b = 330 := by
  sorry

end NUMINAMATH_CALUDE_smallest_b_undefined_inverse_b_330_satisfies_conditions_smallest_b_is_330_l2849_284947


namespace NUMINAMATH_CALUDE_garden_length_l2849_284906

/-- Proves that a rectangular garden with perimeter 500 m and breadth 100 m has length 150 m -/
theorem garden_length (perimeter : ℝ) (breadth : ℝ) (length : ℝ) : 
  perimeter = 500 → 
  breadth = 100 → 
  perimeter = 2 * (length + breadth) → 
  length = 150 := by
  sorry

end NUMINAMATH_CALUDE_garden_length_l2849_284906


namespace NUMINAMATH_CALUDE_locus_proof_methods_correctness_l2849_284999

-- Define a type for points in a geometric space
variable {Point : Type}

-- Define a predicate for points satisfying the locus conditions
variable (satisfiesConditions : Point → Prop)

-- Define a predicate for points being on the locus
variable (onLocus : Point → Prop)

-- Define the correctness of each statement
def statementA : Prop :=
  (∀ p : Point, onLocus p → satisfiesConditions p) ∧
  (∀ p : Point, ¬onLocus p → ¬satisfiesConditions p)

def statementB : Prop :=
  (∀ p : Point, ¬satisfiesConditions p → onLocus p) ∧
  (∀ p : Point, onLocus p → satisfiesConditions p)

def statementC : Prop :=
  (∀ p : Point, satisfiesConditions p → onLocus p) ∧
  (∀ p : Point, ¬onLocus p → satisfiesConditions p)

def statementD : Prop :=
  (∀ p : Point, ¬onLocus p → ¬satisfiesConditions p) ∧
  (∀ p : Point, ¬satisfiesConditions p → ¬onLocus p)

def statementE : Prop :=
  (∀ p : Point, satisfiesConditions p → onLocus p) ∧
  (∀ p : Point, ¬satisfiesConditions p → ¬onLocus p)

-- Theorem stating which methods are correct and which are incorrect
theorem locus_proof_methods_correctness :
  (statementA satisfiesConditions onLocus) ∧
  (¬statementB satisfiesConditions onLocus) ∧
  (¬statementC satisfiesConditions onLocus) ∧
  (statementD satisfiesConditions onLocus) ∧
  (statementE satisfiesConditions onLocus) :=
sorry

end NUMINAMATH_CALUDE_locus_proof_methods_correctness_l2849_284999


namespace NUMINAMATH_CALUDE_farm_area_calculation_l2849_284937

/-- Given a farm divided into sections, calculate its total area -/
def farm_total_area (num_sections : ℕ) (section_area : ℕ) : ℕ :=
  num_sections * section_area

/-- Theorem: The total area of a farm with 5 sections of 60 acres each is 300 acres -/
theorem farm_area_calculation : farm_total_area 5 60 = 300 := by
  sorry

end NUMINAMATH_CALUDE_farm_area_calculation_l2849_284937


namespace NUMINAMATH_CALUDE_sequence_length_l2849_284985

/-- The number of terms in the sequence 1, 2³, 2⁶, 2⁹, ..., 2³ⁿ⁺⁶ -/
def num_terms (n : ℕ) : ℕ := n + 3

/-- The exponent of the k-th term in the sequence -/
def exponent (k : ℕ) : ℕ := 3 * (k - 1)

theorem sequence_length (n : ℕ) : 
  (∃ k : ℕ, k > 0 ∧ exponent k = 3 * n + 6) → 
  num_terms n = (Finset.range (n + 3)).card :=
sorry

end NUMINAMATH_CALUDE_sequence_length_l2849_284985


namespace NUMINAMATH_CALUDE_bowling_tournament_prize_orders_l2849_284936

/-- Represents a bowling tournament with 6 players and a specific playoff structure. -/
structure BowlingTournament :=
  (num_players : Nat)
  (playoff_structure : List (Nat × Nat))

/-- Calculates the number of possible prize order combinations in a bowling tournament. -/
def possiblePrizeOrders (tournament : BowlingTournament) : Nat :=
  2^(tournament.num_players - 1)

/-- Theorem stating that the number of possible prize order combinations
    in the given 6-player bowling tournament is 32. -/
theorem bowling_tournament_prize_orders :
  ∃ (t : BowlingTournament),
    t.num_players = 6 ∧
    t.playoff_structure = [(6, 5), (4, 0), (3, 0), (2, 0), (1, 0)] ∧
    possiblePrizeOrders t = 32 :=
by
  sorry


end NUMINAMATH_CALUDE_bowling_tournament_prize_orders_l2849_284936


namespace NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_18_l2849_284953

theorem smallest_four_digit_multiple_of_18 :
  ∀ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ 18 ∣ n → n ≥ 1008 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_18_l2849_284953


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l2849_284990

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_geom : GeometricSequence a)
  (h_pos : ∀ n, a n > 0)
  (h_prod : a 3 * a 7 = 64) :
  a 5 = 8 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l2849_284990


namespace NUMINAMATH_CALUDE_valid_arrangements_count_l2849_284928

/-- The number of ways to arrange n people in a row -/
def totalArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n people in a row where k specific people sit together -/
def arrangementsWithGrouped (n k : ℕ) : ℕ :=
  Nat.factorial (n - k + 1) * Nat.factorial k

/-- The number of valid arrangements for 8 people where 3 specific people cannot sit together -/
def validArrangements : ℕ :=
  totalArrangements 8 - arrangementsWithGrouped 8 3

theorem valid_arrangements_count :
  validArrangements = 36000 := by sorry

end NUMINAMATH_CALUDE_valid_arrangements_count_l2849_284928


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l2849_284989

/-- An isosceles triangle with two sides of length 8 cm and perimeter 30 cm has a base of length 14 cm. -/
theorem isosceles_triangle_base_length :
  ∀ (base_length : ℝ),
    base_length > 0 →
    2 * 8 + base_length = 30 →
    base_length = 14 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l2849_284989


namespace NUMINAMATH_CALUDE_amazing_triangle_exists_l2849_284973

theorem amazing_triangle_exists : ∃ (a b c : ℕ+), 
  (a.val ^ 2 + b.val ^ 2 = c.val ^ 2) ∧ 
  (∃ (d0 d1 d2 d3 d4 d5 d6 d7 d8 : ℕ), 
    d0 < 10 ∧ d1 < 10 ∧ d2 < 10 ∧ d3 < 10 ∧ d4 < 10 ∧ 
    d5 < 10 ∧ d6 < 10 ∧ d7 < 10 ∧ d8 < 10 ∧
    d0 ≠ d1 ∧ d0 ≠ d2 ∧ d0 ≠ d3 ∧ d0 ≠ d4 ∧ d0 ≠ d5 ∧ d0 ≠ d6 ∧ d0 ≠ d7 ∧ d0 ≠ d8 ∧
    d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d1 ≠ d5 ∧ d1 ≠ d6 ∧ d1 ≠ d7 ∧ d1 ≠ d8 ∧
    d2 ≠ d3 ∧ d2 ≠ d4 ∧ d2 ≠ d5 ∧ d2 ≠ d6 ∧ d2 ≠ d7 ∧ d2 ≠ d8 ∧
    d3 ≠ d4 ∧ d3 ≠ d5 ∧ d3 ≠ d6 ∧ d3 ≠ d7 ∧ d3 ≠ d8 ∧
    d4 ≠ d5 ∧ d4 ≠ d6 ∧ d4 ≠ d7 ∧ d4 ≠ d8 ∧
    d5 ≠ d6 ∧ d5 ≠ d7 ∧ d5 ≠ d8 ∧
    d6 ≠ d7 ∧ d6 ≠ d8 ∧
    d7 ≠ d8 ∧
    a.val = d0 * 100 + d1 * 10 + d2 ∧
    b.val = d3 * 100 + d4 * 10 + d5 ∧
    c.val = d6 * 100 + d7 * 10 + d8) :=
by sorry

end NUMINAMATH_CALUDE_amazing_triangle_exists_l2849_284973


namespace NUMINAMATH_CALUDE_binomial_coefficient_problem_l2849_284901

theorem binomial_coefficient_problem (m : ℕ+) 
  (a b : ℕ) 
  (ha : a = Nat.choose (2 * m) m)
  (hb : b = Nat.choose (2 * m + 1) m)
  (h_eq : 13 * a = 7 * b) : 
  m = 6 := by
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_problem_l2849_284901


namespace NUMINAMATH_CALUDE_product_of_g_at_roots_of_f_l2849_284942

theorem product_of_g_at_roots_of_f (x₁ x₂ x₃ x₄ x₅ : ℂ) : 
  (x₁^5 - x₁^3 + 2*x₁^2 + 1 = 0) →
  (x₂^5 - x₂^3 + 2*x₂^2 + 1 = 0) →
  (x₃^5 - x₃^3 + 2*x₃^2 + 1 = 0) →
  (x₄^5 - x₄^3 + 2*x₄^2 + 1 = 0) →
  (x₅^5 - x₅^3 + 2*x₅^2 + 1 = 0) →
  (x₁^2 - 3) * (x₂^2 - 3) * (x₃^2 - 3) * (x₄^2 - 3) * (x₅^2 - 3) = -59 :=
by sorry

end NUMINAMATH_CALUDE_product_of_g_at_roots_of_f_l2849_284942
