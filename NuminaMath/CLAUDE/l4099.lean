import Mathlib

namespace NUMINAMATH_CALUDE_speed_ratio_A_to_B_l4099_409965

-- Define the work completion rates for A and B
def work_rate_B : ℚ := 1 / 12
def work_rate_A_and_B : ℚ := 1 / 4

-- Define A's work rate in terms of B's
def work_rate_A : ℚ := work_rate_A_and_B - work_rate_B

-- Theorem statement
theorem speed_ratio_A_to_B : 
  work_rate_A / work_rate_B = 2 := by
  sorry

end NUMINAMATH_CALUDE_speed_ratio_A_to_B_l4099_409965


namespace NUMINAMATH_CALUDE_friend_distribution_l4099_409910

/-- The number of ways to distribute n distinguishable items among k categories -/
def distribute (n k : ℕ) : ℕ := k ^ n

/-- The number of friends to be distributed -/
def num_friends : ℕ := 8

/-- The number of clubs available -/
def num_clubs : ℕ := 4

theorem friend_distribution :
  distribute num_friends num_clubs = 65536 := by
  sorry

end NUMINAMATH_CALUDE_friend_distribution_l4099_409910


namespace NUMINAMATH_CALUDE_slope_angle_45_implies_a_equals_1_l4099_409963

theorem slope_angle_45_implies_a_equals_1 (a : ℝ) :
  (∃ (y : ℝ → ℝ), y = λ x => a * x - 1) →  -- Line equation y = ax - 1
  (∃ (θ : ℝ), θ = π / 4 ∧ θ = Real.arctan a) →  -- Slope angle is 45° (π/4 radians)
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_slope_angle_45_implies_a_equals_1_l4099_409963


namespace NUMINAMATH_CALUDE_quadratic_function_continuous_l4099_409924

/-- A quadratic function f(x) = ax^2 + bx + c is continuous at any point x ∈ ℝ,
    where a, b, and c are real constants. -/
theorem quadratic_function_continuous (a b c : ℝ) :
  Continuous (fun x : ℝ => a * x^2 + b * x + c) :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_continuous_l4099_409924


namespace NUMINAMATH_CALUDE_population_increase_in_one_day_l4099_409960

/-- Represents the number of seconds in a day -/
def seconds_per_day : ℕ := 24 * 60 * 60

/-- Represents the birth rate (people per 2 seconds) -/
def birth_rate : ℕ := 10

/-- Represents the death rate (people per 2 seconds) -/
def death_rate : ℕ := 2

/-- Calculates the net population increase over one day -/
def net_population_increase : ℕ :=
  (seconds_per_day / 2) * birth_rate - (seconds_per_day / 2) * death_rate

theorem population_increase_in_one_day :
  net_population_increase = 345600 := by sorry

end NUMINAMATH_CALUDE_population_increase_in_one_day_l4099_409960


namespace NUMINAMATH_CALUDE_mindy_income_multiple_l4099_409974

/-- Proves that Mindy earned 3 times more than Mork given their tax rates and combined tax rate -/
theorem mindy_income_multiple (mork_rate mindy_rate combined_rate : ℚ) : 
  mork_rate = 40/100 →
  mindy_rate = 30/100 →
  combined_rate = 325/1000 →
  ∃ k : ℚ, k = 3 ∧ 
    (mork_rate + k * mindy_rate) / (1 + k) = combined_rate :=
by sorry

end NUMINAMATH_CALUDE_mindy_income_multiple_l4099_409974


namespace NUMINAMATH_CALUDE_dexter_card_count_l4099_409918

/-- The number of boxes filled with basketball cards -/
def basketball_boxes : ℕ := 9

/-- The number of cards in each basketball box -/
def cards_per_basketball_box : ℕ := 15

/-- The number of cards in each football box -/
def cards_per_football_box : ℕ := 20

/-- The difference in number of boxes between basketball and football cards -/
def box_difference : ℕ := 3

/-- The total number of cards Dexter has -/
def total_cards : ℕ := 
  (basketball_boxes * cards_per_basketball_box) + 
  ((basketball_boxes - box_difference) * cards_per_football_box)

theorem dexter_card_count : total_cards = 255 := by
  sorry

end NUMINAMATH_CALUDE_dexter_card_count_l4099_409918


namespace NUMINAMATH_CALUDE_download_rate_proof_l4099_409945

/-- Proves that the download rate for the first 60 megabytes is 5 megabytes per second -/
theorem download_rate_proof (file_size : ℝ) (first_part_size : ℝ) (second_part_rate : ℝ) (total_time : ℝ)
  (h1 : file_size = 90)
  (h2 : first_part_size = 60)
  (h3 : second_part_rate = 10)
  (h4 : total_time = 15)
  (h5 : file_size = first_part_size + (file_size - first_part_size))
  (h6 : total_time = first_part_size / R + (file_size - first_part_size) / second_part_rate) :
  R = 5 := by
  sorry

#check download_rate_proof

end NUMINAMATH_CALUDE_download_rate_proof_l4099_409945


namespace NUMINAMATH_CALUDE_expression_evaluation_l4099_409983

theorem expression_evaluation (x : ℝ) (h : x = 2) :
  (x^2 + 2*x + 1) / (x^2 - 1) / ((x / (x - 1)) - 1) = 3 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4099_409983


namespace NUMINAMATH_CALUDE_cheese_needed_for_event_l4099_409980

def meat_for_10_sandwiches : ℝ := 4
def number_of_sandwiches_planned : ℕ := 30
def initial_sandwich_count : ℕ := 10

theorem cheese_needed_for_event :
  let meat_per_sandwich : ℝ := meat_for_10_sandwiches / initial_sandwich_count
  let cheese_per_sandwich : ℝ := meat_per_sandwich / 2
  cheese_per_sandwich * number_of_sandwiches_planned = 6 := by
sorry

end NUMINAMATH_CALUDE_cheese_needed_for_event_l4099_409980


namespace NUMINAMATH_CALUDE_odd_expression_l4099_409928

theorem odd_expression (p q : ℕ) (hp : Odd p) (hq : Odd q) (hp_pos : p > 0) (hq_pos : q > 0) :
  Odd (2 * p^2 - q) := by
  sorry

end NUMINAMATH_CALUDE_odd_expression_l4099_409928


namespace NUMINAMATH_CALUDE_rectangle_area_error_percentage_l4099_409903

/-- Given a rectangle where one side is measured 8% in excess and the other side is measured 5% in deficit, 
    the error percentage in the calculated area is 2.6%. -/
theorem rectangle_area_error_percentage (L W : ℝ) (L' W' : ℝ) (h1 : L' = 1.08 * L) (h2 : W' = 0.95 * W) :
  (L' * W' - L * W) / (L * W) * 100 = 2.6 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_error_percentage_l4099_409903


namespace NUMINAMATH_CALUDE_total_students_l4099_409946

theorem total_students (boys girls : ℕ) (h1 : boys * 5 = girls * 8) (h2 : girls = 120) : 
  boys + girls = 312 := by
sorry

end NUMINAMATH_CALUDE_total_students_l4099_409946


namespace NUMINAMATH_CALUDE_soda_bottle_difference_l4099_409990

theorem soda_bottle_difference (regular_soda : ℕ) (diet_soda : ℕ)
  (h1 : regular_soda = 60)
  (h2 : diet_soda = 19) :
  regular_soda - diet_soda = 41 := by
  sorry

end NUMINAMATH_CALUDE_soda_bottle_difference_l4099_409990


namespace NUMINAMATH_CALUDE_derek_same_color_probability_l4099_409931

/-- Represents the number of marbles of each color -/
structure MarbleDistribution :=
  (red : ℕ)
  (green : ℕ)
  (blue : ℕ)

/-- Represents the number of marbles drawn by each person -/
structure DrawingProcess :=
  (david : ℕ)
  (dana : ℕ)
  (derek : ℕ)

/-- Calculates the probability of Derek getting at least 2 marbles of the same color -/
def probability_same_color (dist : MarbleDistribution) (process : DrawingProcess) : ℚ :=
  sorry

theorem derek_same_color_probability :
  let initial_distribution : MarbleDistribution := ⟨3, 2, 3⟩
  let drawing_process : DrawingProcess := ⟨2, 2, 3⟩
  probability_same_color initial_distribution drawing_process = 19 / 210 :=
sorry

end NUMINAMATH_CALUDE_derek_same_color_probability_l4099_409931


namespace NUMINAMATH_CALUDE_smallest_sum_for_equation_l4099_409949

theorem smallest_sum_for_equation (m n : ℕ+) (h : 3 * m ^ 3 = 5 * n ^ 5) :
  ∀ (x y : ℕ+), 3 * x ^ 3 = 5 * y ^ 5 → m + n ≤ x + y :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_for_equation_l4099_409949


namespace NUMINAMATH_CALUDE_no_exact_table_count_l4099_409951

theorem no_exact_table_count : ¬∃ (t : ℕ), 
  3 * (8 * t) + 4 * (2 * t) + 4 * t = 656 := by
  sorry

end NUMINAMATH_CALUDE_no_exact_table_count_l4099_409951


namespace NUMINAMATH_CALUDE_sienas_initial_bookmarks_l4099_409955

/-- Calculates the number of pages Siena had before March, given her daily bookmarking rate and final page count. -/
theorem sienas_initial_bookmarks (daily_bookmarks : ℕ) (march_days : ℕ) (final_count : ℕ) : 
  daily_bookmarks = 30 → 
  march_days = 31 → 
  final_count = 1330 → 
  final_count - (daily_bookmarks * march_days) = 400 :=
by
  sorry

#check sienas_initial_bookmarks

end NUMINAMATH_CALUDE_sienas_initial_bookmarks_l4099_409955


namespace NUMINAMATH_CALUDE_first_floor_units_count_l4099_409993

/-- A building with a specified number of floors and apartments -/
structure Building where
  floors : ℕ
  firstFloorUnits : ℕ
  otherFloorUnits : ℕ

/-- The total number of apartment units in a building -/
def totalUnits (b : Building) : ℕ :=
  b.firstFloorUnits + (b.floors - 1) * b.otherFloorUnits

theorem first_floor_units_count (b1 b2 : Building) :
  b1 = b2 ∧ 
  b1.floors = 4 ∧ 
  b1.otherFloorUnits = 5 ∧ 
  totalUnits b1 + totalUnits b2 = 34 →
  b1.firstFloorUnits = 2 :=
sorry

end NUMINAMATH_CALUDE_first_floor_units_count_l4099_409993


namespace NUMINAMATH_CALUDE_f_properties_l4099_409975

noncomputable def f (x : ℝ) := (1/3) * x^3 - 2 * x^2 + 3 * x + 2/3

theorem f_properties :
  (∀ x : ℝ, f x ≤ 2) ∧
  (∀ b : ℝ, 
    (b ≤ 0 ∨ b > (9 + Real.sqrt 33) / 6 → 
      (∀ x ∈ Set.Icc b (b + 1), f x ≤ (b^3 / 3) - b^2 + 2) ∧
      ∃ x ∈ Set.Icc b (b + 1), f x = (b^3 / 3) - b^2 + 2) ∧
    (0 < b ∧ b ≤ 1 → 
      (∀ x ∈ Set.Icc b (b + 1), f x ≤ 2) ∧
      ∃ x ∈ Set.Icc b (b + 1), f x = 2) ∧
    (1 < b ∧ b ≤ (9 + Real.sqrt 33) / 6 → 
      (∀ x ∈ Set.Icc b (b + 1), f x ≤ (b^3 / 3) - 2 * b^2 + 3 * b + 2/3) ∧
      ∃ x ∈ Set.Icc b (b + 1), f x = (b^3 / 3) - 2 * b^2 + 3 * b + 2/3)) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l4099_409975


namespace NUMINAMATH_CALUDE_power_sum_zero_l4099_409961

theorem power_sum_zero : (-2 : ℤ)^(3^2) + 2^(3^2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_zero_l4099_409961


namespace NUMINAMATH_CALUDE_equal_to_mac_ratio_l4099_409939

/-- Represents the survey results of computer brand preferences among college students. -/
structure SurveyResults where
  total : ℕ
  mac_preference : ℕ
  no_preference : ℕ
  windows_preference : ℕ

/-- Calculates the number of students who equally preferred both brands. -/
def equal_preference (s : SurveyResults) : ℕ :=
  s.total - (s.mac_preference + s.no_preference + s.windows_preference)

/-- Represents a ratio as a pair of natural numbers. -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- Theorem stating the ratio of students who equally preferred both brands
    to students who preferred Mac to Windows. -/
theorem equal_to_mac_ratio (s : SurveyResults)
  (h_total : s.total = 210)
  (h_mac : s.mac_preference = 60)
  (h_no_pref : s.no_preference = 90)
  (h_windows : s.windows_preference = 40) :
  ∃ (r : Ratio), r.numerator = 1 ∧ r.denominator = 3 ∧
  r.numerator * s.mac_preference = r.denominator * equal_preference s :=
sorry

end NUMINAMATH_CALUDE_equal_to_mac_ratio_l4099_409939


namespace NUMINAMATH_CALUDE_table_tennis_equation_l4099_409957

/-- Represents a table tennis competition -/
structure TableTennisCompetition where
  teams : ℕ
  totalMatches : ℕ
  pairPlaysOneMatch : Bool

/-- The equation for the number of matches in a table tennis competition -/
def matchEquation (c : TableTennisCompetition) : Prop :=
  c.teams * (c.teams - 1) = c.totalMatches * 2

/-- Theorem stating the correct equation for the given competition conditions -/
theorem table_tennis_equation (c : TableTennisCompetition) 
  (h1 : c.pairPlaysOneMatch = true) 
  (h2 : c.totalMatches = 28) : 
  matchEquation c := by
  sorry

end NUMINAMATH_CALUDE_table_tennis_equation_l4099_409957


namespace NUMINAMATH_CALUDE_given_point_in_second_quadrant_l4099_409908

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the second quadrant -/
def is_in_second_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- The given point -/
def given_point : Point :=
  { x := -3, y := 2 }

/-- Theorem: The given point is in the second quadrant -/
theorem given_point_in_second_quadrant :
  is_in_second_quadrant given_point := by
  sorry

end NUMINAMATH_CALUDE_given_point_in_second_quadrant_l4099_409908


namespace NUMINAMATH_CALUDE_rebus_solution_l4099_409930

theorem rebus_solution : ∃! (a b c d : ℕ),
  (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0) ∧
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧
  (1000 * a + 100 * b + 10 * c + a = 182 * (10 * c + d)) ∧
  (a = 2 ∧ b = 9 ∧ c = 1 ∧ d = 6) :=
by sorry

end NUMINAMATH_CALUDE_rebus_solution_l4099_409930


namespace NUMINAMATH_CALUDE_problem_solution_l4099_409925

theorem problem_solution (x y : ℝ) (h1 : x + 2*y = 14) (h2 : y = 3) : 2*x + 3*y = 25 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4099_409925


namespace NUMINAMATH_CALUDE_rainfall_difference_l4099_409962

def rainfall_day1 : ℝ := 26
def rainfall_day2 : ℝ := 34
def rainfall_day3 : ℝ := rainfall_day2 - 12
def average_rainfall : ℝ := 140

theorem rainfall_difference : 
  average_rainfall - (rainfall_day1 + rainfall_day2 + rainfall_day3) = 58 := by
  sorry

end NUMINAMATH_CALUDE_rainfall_difference_l4099_409962


namespace NUMINAMATH_CALUDE_inequality_proof_l4099_409916

theorem inequality_proof (a b c d p q : ℝ) 
  (h1 : a * b + c * d = 2 * p * q) 
  (h2 : a * c ≥ p^2) 
  (h3 : p^2 > 0) : 
  b * d ≤ q^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4099_409916


namespace NUMINAMATH_CALUDE_farm_animals_l4099_409913

theorem farm_animals (cows chickens ducks : ℕ) : 
  (4 * cows + 2 * chickens + 2 * ducks = 2 * (cows + chickens + ducks) + 22) →
  (chickens + ducks = 2 * cows) →
  (cows = 11) := by
  sorry

end NUMINAMATH_CALUDE_farm_animals_l4099_409913


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l4099_409943

universe u

def U : Finset ℕ := {1, 2, 3, 4, 5, 6}
def M : Finset ℕ := {1, 3, 5}

theorem complement_of_M_in_U :
  (U \ M) = {2, 4, 6} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l4099_409943


namespace NUMINAMATH_CALUDE_sum_base8_327_73_l4099_409972

/-- Converts a base-8 number represented as a list of digits to its decimal equivalent. -/
def base8ToDecimal (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => 8 * acc + d) 0

/-- Converts a decimal number to its base-8 representation as a list of digits. -/
def decimalToBase8 (n : Nat) : List Nat :=
  if n < 8 then [n]
  else (n % 8) :: decimalToBase8 (n / 8)

/-- The sum of 327₈ and 73₈ in base 8 is equal to 422₈. -/
theorem sum_base8_327_73 :
  decimalToBase8 (base8ToDecimal [3, 2, 7] + base8ToDecimal [7, 3]) = [4, 2, 2] := by
  sorry

end NUMINAMATH_CALUDE_sum_base8_327_73_l4099_409972


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l4099_409933

theorem sqrt_equation_solution (x : ℝ) :
  Real.sqrt (5 * x + 2 * x^2 + 8) = 12 ↔ x = 8 ∨ x = -17/2 := by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l4099_409933


namespace NUMINAMATH_CALUDE_sara_pumpkins_l4099_409971

/-- The number of pumpkins Sara has now -/
def pumpkins_left : ℕ := 20

/-- The number of pumpkins eaten by rabbits -/
def pumpkins_eaten : ℕ := 23

/-- The initial number of pumpkins Sara grew -/
def initial_pumpkins : ℕ := pumpkins_left + pumpkins_eaten

theorem sara_pumpkins : initial_pumpkins = 43 := by
  sorry

end NUMINAMATH_CALUDE_sara_pumpkins_l4099_409971


namespace NUMINAMATH_CALUDE_minimum_rice_amount_l4099_409976

theorem minimum_rice_amount (o r : ℝ) (ho : o ≥ 8 + r / 3) (ho2 : o ≤ 2 * r) :
  ∃ (min_r : ℕ), min_r = 5 ∧ ∀ (r' : ℕ), r' ≥ min_r → ∃ (o' : ℝ), o' ≥ 8 + r' / 3 ∧ o' ≤ 2 * r' :=
sorry

end NUMINAMATH_CALUDE_minimum_rice_amount_l4099_409976


namespace NUMINAMATH_CALUDE_negative_nine_plus_sixteen_y_squared_equals_seven_y_squared_l4099_409992

theorem negative_nine_plus_sixteen_y_squared_equals_seven_y_squared (y : ℝ) : 
  -9 * y^2 + 16 * y^2 = 7 * y^2 := by
  sorry

end NUMINAMATH_CALUDE_negative_nine_plus_sixteen_y_squared_equals_seven_y_squared_l4099_409992


namespace NUMINAMATH_CALUDE_inverse_graph_point_l4099_409936

-- Define a function f with an inverse
variable (f : ℝ → ℝ)
variable (f_inv : ℝ → ℝ)

-- Define the condition that f_inv is the inverse of f
axiom inverse_relation : ∀ x, f_inv (f x) = x ∧ f (f_inv x) = x

-- Define the condition that the graph of y = x - f(x) passes through (2,5)
axiom graph_condition : 2 - f 2 = 5

-- Theorem to prove
theorem inverse_graph_point :
  (∀ x, f_inv (f x) = x ∧ f (f_inv x) = x) →
  (2 - f 2 = 5) →
  f_inv (-3) + 3 = 5 :=
by sorry

end NUMINAMATH_CALUDE_inverse_graph_point_l4099_409936


namespace NUMINAMATH_CALUDE_paint_usage_proof_l4099_409950

theorem paint_usage_proof (total_paint : ℝ) (second_week_fraction : ℝ) (total_used : ℝ) 
  (h1 : total_paint = 360)
  (h2 : second_week_fraction = 1/6)
  (h3 : total_used = 135) :
  ∃ (first_week_fraction : ℝ),
    first_week_fraction * total_paint + 
    second_week_fraction * (total_paint - first_week_fraction * total_paint) = total_used ∧
    first_week_fraction = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_paint_usage_proof_l4099_409950


namespace NUMINAMATH_CALUDE_min_groups_for_photography_class_l4099_409906

theorem min_groups_for_photography_class (total_students : ℕ) (max_group_size : ℕ) 
  (h1 : total_students = 30) (h2 : max_group_size = 6) : 
  Nat.ceil (total_students / max_group_size) = 5 :=
by sorry

end NUMINAMATH_CALUDE_min_groups_for_photography_class_l4099_409906


namespace NUMINAMATH_CALUDE_equation_solutions_l4099_409984

theorem equation_solutions :
  (∃ (x : ℝ), (1/3) * (x - 3)^2 = 12 ↔ x = 9 ∨ x = -3) ∧
  (∃ (x : ℝ), (2*x - 1)^2 = (1 - x)^2 ↔ x = 2/3 ∨ x = 0) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l4099_409984


namespace NUMINAMATH_CALUDE_smallest_sum_reciprocals_l4099_409967

theorem smallest_sum_reciprocals (x y : ℕ+) (h1 : x ≠ y) (h2 : (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 13) :
  ∃ (a b : ℕ+), a ≠ b ∧ (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 13 ∧ a + b = 196 ∧
  ∀ (c d : ℕ+), c ≠ d → (1 : ℚ) / c + (1 : ℚ) / d = (1 : ℚ) / 13 → c + d ≥ 196 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_reciprocals_l4099_409967


namespace NUMINAMATH_CALUDE_odd_power_divisibility_l4099_409905

theorem odd_power_divisibility (a b : ℕ) (ha : Odd a) (hb : Odd b) :
  ∀ n : ℕ, ∃ m : ℕ, (2^n ∣ a^m * b^2 - 1) ∨ (2^n ∣ b^m * a^2 - 1) :=
by sorry

end NUMINAMATH_CALUDE_odd_power_divisibility_l4099_409905


namespace NUMINAMATH_CALUDE_clock_strikes_ten_l4099_409902

/-- A clock that strikes at regular intervals -/
structure StrikingClock where
  /-- The time it takes to complete a given number of strikes -/
  strike_time : ℕ → ℝ
  /-- The number of strikes at a given hour -/
  strikes_at_hour : ℕ → ℕ

/-- Our specific clock that takes 7 seconds to strike 7 times at 7 o'clock -/
def our_clock : StrikingClock where
  strike_time := fun n => if n = 7 then 7 else 0  -- We only know about 7 strikes
  strikes_at_hour := fun h => if h = 7 then 7 else 0  -- We only know about 7 o'clock

/-- The theorem stating that our clock takes 10.5 seconds to strike 10 times -/
theorem clock_strikes_ten (c : StrikingClock) (h : c.strike_time 7 = 7) :
  c.strike_time 10 = 10.5 := by
  sorry

#check clock_strikes_ten our_clock (by rfl)

end NUMINAMATH_CALUDE_clock_strikes_ten_l4099_409902


namespace NUMINAMATH_CALUDE_average_of_a_and_b_l4099_409989

theorem average_of_a_and_b (a b c : ℝ) (h1 : (b + c) / 2 = 90) (h2 : c - a = 90) : 
  (a + b) / 2 = 45 := by
  sorry

end NUMINAMATH_CALUDE_average_of_a_and_b_l4099_409989


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l4099_409948

def M : Set ℝ := {x | x^2 - 2*x - 3 = 0}
def N : Set ℝ := {x | -2 < x ∧ x ≤ 4}

theorem intersection_of_M_and_N : M ∩ N = {-1, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l4099_409948


namespace NUMINAMATH_CALUDE_polar_coordinate_conversion_l4099_409919

def standard_polar_representation (r : ℝ) (θ : ℝ) : Prop :=
  r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi

theorem polar_coordinate_conversion :
  ∀ (r : ℝ) (θ : ℝ),
    r = -3 ∧ θ = 5 * Real.pi / 6 →
    ∃ (r' : ℝ) (θ' : ℝ),
      standard_polar_representation r' θ' ∧
      r' = 3 ∧ θ' = 11 * Real.pi / 6 :=
by sorry

end NUMINAMATH_CALUDE_polar_coordinate_conversion_l4099_409919


namespace NUMINAMATH_CALUDE_prob_at_least_one_head_in_five_tosses_l4099_409986

/-- The probability of getting at least one head in five coin tosses -/
theorem prob_at_least_one_head_in_five_tosses : 
  let p_head : ℚ := 1/2  -- probability of getting heads on a single toss
  let n : ℕ := 5        -- number of coin tosses
  1 - (1 - p_head)^n = 31/32 :=
by sorry

end NUMINAMATH_CALUDE_prob_at_least_one_head_in_five_tosses_l4099_409986


namespace NUMINAMATH_CALUDE_triangle_inequality_l4099_409977

-- Define the points
variable (A B C P A₁ B₁ C₁ : ℝ × ℝ)

-- Define the equilateral triangle ABC
def is_equilateral (A B C : ℝ × ℝ) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A

-- Define that P is inside triangle ABC
def is_inside_triangle (P A B C : ℝ × ℝ) : Prop :=
  ∃ (α β γ : ℝ), α > 0 ∧ β > 0 ∧ γ > 0 ∧ α + β + γ = 1 ∧
  P = (α * A.1 + β * B.1 + γ * C.1, α * A.2 + β * B.2 + γ * C.2)

-- Define that A₁, B₁, C₁ are on the sides of triangle ABC
def on_side (X Y Z : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), 0 < t ∧ t < 1 ∧ Y = (t * X.1 + (1 - t) * Z.1, t * X.2 + (1 - t) * Z.2)

-- Define the theorem
theorem triangle_inequality (A B C P A₁ B₁ C₁ : ℝ × ℝ) 
  (h1 : is_equilateral A B C)
  (h2 : is_inside_triangle P A B C)
  (h3 : on_side A₁ B C)
  (h4 : on_side B₁ C A)
  (h5 : on_side C₁ A B) :
  dist A₁ B₁ * dist B₁ C₁ * dist C₁ A₁ ≥ dist A₁ B * dist B₁ C * dist C₁ A :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_l4099_409977


namespace NUMINAMATH_CALUDE_equation_solution_l4099_409956

theorem equation_solution : 
  ∃ (x₁ x₂ : ℝ), x₁ = 1 ∧ x₂ = (3/2 : ℝ) ∧ 
  (∀ x : ℝ, 2 * (x - 1)^2 = x - 1 ↔ (x = x₁ ∨ x = x₂)) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l4099_409956


namespace NUMINAMATH_CALUDE_inequality_solution_m_range_l4099_409970

variable (m : ℝ)
def f (x : ℝ) := (m + 1) * x^2 - (m - 1) * x + m - 1

theorem inequality_solution (x : ℝ) :
  (m = -1 ∧ x ≥ 1) ∨
  (m > -1 ∧ (x ≤ (m - 1) / (m + 1) ∨ x ≥ 1)) ∨
  (m < -1 ∧ 1 ≤ x ∧ x ≤ (m - 1) / (m + 1)) ↔
  f m x ≥ (m + 1) * x := by sorry

theorem m_range :
  (∀ x ∈ Set.Icc (-1/2 : ℝ) (1/2 : ℝ), f m x ≥ 0) → m ≥ 1 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_m_range_l4099_409970


namespace NUMINAMATH_CALUDE_unique_four_digit_reverse_multiple_l4099_409941

/-- A function that reverses the digits of a natural number -/
def reverseDigits (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number has exactly four digits -/
def hasFourDigits (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

theorem unique_four_digit_reverse_multiple :
  ∃! (M : ℕ), hasFourDigits M ∧ 
              hasFourDigits (4 * M) ∧ 
              4 * M = reverseDigits M ∧
              M = 2178 := by sorry

end NUMINAMATH_CALUDE_unique_four_digit_reverse_multiple_l4099_409941


namespace NUMINAMATH_CALUDE_ring_toss_total_earnings_l4099_409994

/-- The ring toss game at a carnival earns a certain amount per day for a given number of days. -/
def carnival_earnings (daily_earnings : ℕ) (num_days : ℕ) : ℕ :=
  daily_earnings * num_days

/-- Theorem: The ring toss game earns 3168 dollars in total when it makes 144 dollars per day for 22 days. -/
theorem ring_toss_total_earnings :
  carnival_earnings 144 22 = 3168 := by
  sorry

end NUMINAMATH_CALUDE_ring_toss_total_earnings_l4099_409994


namespace NUMINAMATH_CALUDE_bus_car_speed_problem_l4099_409998

/-- Proves that given the conditions of the problem, the bus speed is 50 km/h and the car speed is 75 km/h -/
theorem bus_car_speed_problem (distance : ℝ) (delay : ℝ) (speed_ratio : ℝ) 
  (h1 : distance = 50)
  (h2 : delay = 1/3)
  (h3 : speed_ratio = 1.5)
  (h4 : ∀ (bus_speed : ℝ), bus_speed > 0 → 
    distance / bus_speed - distance / (speed_ratio * bus_speed) = delay) :
  ∃ (bus_speed car_speed : ℝ),
    bus_speed = 50 ∧ 
    car_speed = 75 ∧
    car_speed = speed_ratio * bus_speed :=
by sorry

end NUMINAMATH_CALUDE_bus_car_speed_problem_l4099_409998


namespace NUMINAMATH_CALUDE_min_value_theorem_l4099_409991

theorem min_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y) / z + (x + z) / y + (y + z) / x + 3 ≥ 9 ∧
  ((x + y) / z + (x + z) / y + (y + z) / x + 3 = 9 ↔ x = y ∧ y = z) :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l4099_409991


namespace NUMINAMATH_CALUDE_largest_mersenne_prime_under_500_l4099_409904

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def is_mersenne_prime (n : ℕ) : Prop := ∃ p : ℕ, is_prime p ∧ n = 2^p - 1

theorem largest_mersenne_prime_under_500 : 
  (∀ m : ℕ, is_mersenne_prime m → m < 500 → m ≤ 127) ∧ 
  is_mersenne_prime 127 ∧ 
  127 < 500 :=
sorry

end NUMINAMATH_CALUDE_largest_mersenne_prime_under_500_l4099_409904


namespace NUMINAMATH_CALUDE_max_yellow_apples_max_total_apples_l4099_409917

/-- Represents the number of apples of each color in the basket -/
structure Basket :=
  (green : ℕ)
  (yellow : ℕ)
  (red : ℕ)

/-- Represents the number of apples Alyona has taken -/
structure TakenApples :=
  (green : ℕ)
  (yellow : ℕ)
  (red : ℕ)

/-- Checks if Alyona should stop taking apples -/
def shouldStop (taken : TakenApples) : Prop :=
  taken.green < taken.yellow ∧ taken.yellow < taken.red

/-- The initial state of the basket -/
def initialBasket : Basket :=
  { green := 10, yellow := 13, red := 18 }

/-- Theorem stating the maximum number of yellow apples Alyona can take -/
theorem max_yellow_apples :
  ∃ (taken : TakenApples),
    taken.yellow = initialBasket.yellow ∧
    taken.yellow ≤ initialBasket.yellow ∧
    ¬(shouldStop taken) ∧
    ∀ (other : TakenApples),
      other.yellow > taken.yellow →
      shouldStop other ∨ other.yellow > initialBasket.yellow :=
sorry

/-- Theorem stating the maximum total number of apples Alyona can take -/
theorem max_total_apples :
  ∃ (taken : TakenApples),
    taken.green + taken.yellow + taken.red = 39 ∧
    taken.green ≤ initialBasket.green ∧
    taken.yellow ≤ initialBasket.yellow ∧
    taken.red ≤ initialBasket.red ∧
    ¬(shouldStop taken) ∧
    ∀ (other : TakenApples),
      other.green + other.yellow + other.red > 39 →
      shouldStop other ∨
      other.green > initialBasket.green ∨
      other.yellow > initialBasket.yellow ∨
      other.red > initialBasket.red :=
sorry

end NUMINAMATH_CALUDE_max_yellow_apples_max_total_apples_l4099_409917


namespace NUMINAMATH_CALUDE_ice_cream_sales_theorem_l4099_409938

/-- Represents the ice cream cone sales scenario -/
structure IceCreamSales where
  free_cone_interval : Nat  -- Every nth customer gets a free cone
  cone_price : Nat          -- Price of each cone in dollars
  free_cones_given : Nat    -- Number of free cones given away

/-- Calculates the total sales amount for the ice cream cones -/
def calculate_sales (sales : IceCreamSales) : Nat :=
  sorry

/-- Theorem stating that given the conditions, the sales amount is $100 -/
theorem ice_cream_sales_theorem (sales : IceCreamSales) 
  (h1 : sales.free_cone_interval = 6)
  (h2 : sales.cone_price = 2)
  (h3 : sales.free_cones_given = 10) : 
  calculate_sales sales = 100 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_sales_theorem_l4099_409938


namespace NUMINAMATH_CALUDE_min_value_theorem_l4099_409964

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_sum : x + 2*y = 4) :
  (2/x + 1/y) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l4099_409964


namespace NUMINAMATH_CALUDE_function_equation_solution_l4099_409923

theorem function_equation_solution (f : ℤ → ℤ) :
  (∀ a b : ℤ, f (2 * a) + 2 * f b = f (f (a + b))) →
  (∃ c : ℤ, ∀ x : ℤ, f x = 0 ∨ f x = 2 * x + c) :=
by sorry

end NUMINAMATH_CALUDE_function_equation_solution_l4099_409923


namespace NUMINAMATH_CALUDE_gum_pack_size_l4099_409995

theorem gum_pack_size : ∃ x : ℕ+, 
  (30 : ℚ) - 2 * x.val = 30 * 40 / (40 + 4 * x.val) ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_gum_pack_size_l4099_409995


namespace NUMINAMATH_CALUDE_parabola_focus_value_hyperbola_standard_equation_l4099_409981

-- Problem 1
theorem parabola_focus_value (p : ℝ) (h1 : p > 0) :
  (∃ x y : ℝ, y^2 = 2*p*x ∧ 2*x - y - 4 = 0 ∧ x = p ∧ y = 0) →
  p = 2 := by sorry

-- Problem 2
theorem hyperbola_standard_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (b / a = 3 / 4) ∧ 
  (a^2 / (a^2 + b^2).sqrt = 16 / 5) →
  ∀ x y : ℝ, x^2 / 16 - y^2 / 9 = 1 ↔ x^2 / a^2 - y^2 / b^2 = 1 := by sorry

end NUMINAMATH_CALUDE_parabola_focus_value_hyperbola_standard_equation_l4099_409981


namespace NUMINAMATH_CALUDE_sequence_problem_l4099_409944

/-- The sequence function F that generates the nth term of the sequence --/
def F : ℕ → ℚ := sorry

/-- The sum of the first n natural numbers --/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The theorem stating that F(16) = 1/6 and F(4952) = 2/99 --/
theorem sequence_problem :
  F 16 = 1 / 6 ∧ F 4952 = 2 / 99 := by sorry

end NUMINAMATH_CALUDE_sequence_problem_l4099_409944


namespace NUMINAMATH_CALUDE_no_integer_solutions_l4099_409907

theorem no_integer_solutions : ¬∃ (m n : ℤ), m^3 + 3*m^2 + 2*m = 8*n^3 + 12*n^2 + 6*n + 1 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l4099_409907


namespace NUMINAMATH_CALUDE_constant_term_expansion_l4099_409958

theorem constant_term_expansion : 
  let f : ℝ → ℝ := λ x => (x^2 + 2) * (1/x^2 - 1)^5
  ∃ c : ℝ, ∀ x : ℝ, x ≠ 0 → f x = c + x * (f x - c) / x ∧ c = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l4099_409958


namespace NUMINAMATH_CALUDE_total_piggy_bank_value_l4099_409911

/-- Represents the capacity of a piggy bank for different coin types -/
structure PiggyBank where
  pennies : Nat
  dimes : Nat
  nickels : Nat
  quarters : Nat

/-- Calculates the total value in a piggy bank -/
def piggyBankValue (pb : PiggyBank) : Rat :=
  pb.pennies * 1 / 100 + pb.dimes * 10 / 100 + pb.nickels * 5 / 100 + pb.quarters * 25 / 100

/-- The first piggy bank -/
def piggyBank1 : PiggyBank := ⟨100, 50, 20, 10⟩

/-- The second piggy bank -/
def piggyBank2 : PiggyBank := ⟨150, 30, 40, 15⟩

/-- The third piggy bank -/
def piggyBank3 : PiggyBank := ⟨200, 60, 10, 20⟩

/-- Theorem stating that the total value in all three piggy banks is $33.25 -/
theorem total_piggy_bank_value :
  piggyBankValue piggyBank1 + piggyBankValue piggyBank2 + piggyBankValue piggyBank3 = 3325 / 100 := by
  sorry

end NUMINAMATH_CALUDE_total_piggy_bank_value_l4099_409911


namespace NUMINAMATH_CALUDE_square_field_area_l4099_409912

/-- The area of a square field with a diagonal of 26 meters is 338 square meters. -/
theorem square_field_area (diagonal : ℝ) (h : diagonal = 26) : 
  (diagonal ^ 2) / 2 = 338 := by
  sorry

end NUMINAMATH_CALUDE_square_field_area_l4099_409912


namespace NUMINAMATH_CALUDE_courtyard_length_l4099_409915

/-- Proves that a courtyard with given dimensions and number of bricks has a specific length -/
theorem courtyard_length 
  (width : ℝ) 
  (brick_length : ℝ) 
  (brick_width : ℝ) 
  (num_bricks : ℕ) : 
  width = 16 →
  brick_length = 0.2 →
  brick_width = 0.1 →
  num_bricks = 24000 →
  (width * (num_bricks * brick_length * brick_width / width)) = 30 := by
sorry

end NUMINAMATH_CALUDE_courtyard_length_l4099_409915


namespace NUMINAMATH_CALUDE_prism_faces_count_l4099_409969

/-- A prism is a polyhedron with two congruent polygonal bases and rectangular lateral faces. -/
structure Prism where
  /-- The number of sides in each base of the prism -/
  base_sides : ℕ
  /-- The number of vertices of the prism -/
  vertices : ℕ
  /-- The number of edges of the prism -/
  edges : ℕ
  /-- The number of faces of the prism -/
  faces : ℕ
  /-- The sum of vertices and edges is 40 -/
  sum_condition : vertices + edges = 40
  /-- The number of vertices is twice the number of base sides -/
  vertices_def : vertices = 2 * base_sides
  /-- The number of edges is thrice the number of base sides -/
  edges_def : edges = 3 * base_sides
  /-- The number of faces is 2 more than the number of base sides -/
  faces_def : faces = base_sides + 2

/-- Theorem: A prism with 40 as the sum of its edges and vertices has 10 faces -/
theorem prism_faces_count (p : Prism) : p.faces = 10 := by
  sorry


end NUMINAMATH_CALUDE_prism_faces_count_l4099_409969


namespace NUMINAMATH_CALUDE_frisbee_sales_receipts_l4099_409929

/-- Represents the total receipts from frisbee sales for a week -/
def total_receipts (x y : ℕ) : ℕ := 3 * x + 4 * y

/-- Theorem stating that the total receipts from frisbee sales for the week is $200 -/
theorem frisbee_sales_receipts :
  ∃ (x y : ℕ), x + y = 60 ∧ y ≥ 20 ∧ total_receipts x y = 200 := by
  sorry

end NUMINAMATH_CALUDE_frisbee_sales_receipts_l4099_409929


namespace NUMINAMATH_CALUDE_circles_configuration_l4099_409937

/-- Two circles with radii r₁ and r₂, and distance d between their centers,
    are in the "one circle inside the other" configuration if d < |r₁ - r₂| -/
def CircleInsideOther (r₁ r₂ d : ℝ) : Prop :=
  d < |r₁ - r₂|

/-- Given two circles with radii 1 and 5, and distance 3 between their centers,
    prove that one circle is inside the other -/
theorem circles_configuration :
  CircleInsideOther 1 5 3 := by
sorry

end NUMINAMATH_CALUDE_circles_configuration_l4099_409937


namespace NUMINAMATH_CALUDE_floor_sqrt_80_l4099_409935

theorem floor_sqrt_80 : ⌊Real.sqrt 80⌋ = 8 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_80_l4099_409935


namespace NUMINAMATH_CALUDE_coefficient_x_seven_l4099_409973

theorem coefficient_x_seven (x : ℝ) :
  ∃ (a₈ a₇ a₆ a₅ a₄ a₃ a₂ a₁ a₀ : ℝ),
    (x + 1)^5 * (2*x - 1)^3 = a₈*x^8 + a₇*x^7 + a₆*x^6 + a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀ ∧
    a₇ = 28 :=
by sorry

end NUMINAMATH_CALUDE_coefficient_x_seven_l4099_409973


namespace NUMINAMATH_CALUDE_division_multiplication_equality_l4099_409934

theorem division_multiplication_equality : (180 / 6) * 3 = 90 := by
  sorry

end NUMINAMATH_CALUDE_division_multiplication_equality_l4099_409934


namespace NUMINAMATH_CALUDE_mean_equality_implies_z_l4099_409988

theorem mean_equality_implies_z (z : ℝ) : 
  (8 + 15 + 24) / 3 = (16 + z) / 2 → z = 15.34 := by
  sorry

end NUMINAMATH_CALUDE_mean_equality_implies_z_l4099_409988


namespace NUMINAMATH_CALUDE_cloth_coloring_problem_l4099_409959

/-- Represents the work done by a group of men coloring cloth -/
structure ClothColoring where
  men : ℕ
  days : ℝ
  length : ℝ

/-- The problem statement -/
theorem cloth_coloring_problem (group1 group2 : ClothColoring) :
  group1.men = 4 ∧
  group1.days = 2 ∧
  group2.men = 5 ∧
  group2.days = 1.2 ∧
  group2.length = 36 ∧
  group1.men * group1.days * group1.length = group2.men * group2.days * group2.length →
  group1.length = 27 := by
  sorry

end NUMINAMATH_CALUDE_cloth_coloring_problem_l4099_409959


namespace NUMINAMATH_CALUDE_cookie_difference_l4099_409909

/-- Proves that the difference between the number of cookies in 8 boxes and 9 bags is 33,
    given that each box contains 12 cookies and each bag contains 7 cookies. -/
theorem cookie_difference :
  let cookies_per_box : ℕ := 12
  let cookies_per_bag : ℕ := 7
  let num_boxes : ℕ := 8
  let num_bags : ℕ := 9
  (num_boxes * cookies_per_box) - (num_bags * cookies_per_bag) = 33 := by
  sorry

end NUMINAMATH_CALUDE_cookie_difference_l4099_409909


namespace NUMINAMATH_CALUDE_quadratic_is_perfect_square_l4099_409997

theorem quadratic_is_perfect_square (a : ℝ) :
  (∃ r s : ℝ, ∀ x : ℝ, a * x^2 - 8 * x + 16 = (r * x + s)^2) →
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_is_perfect_square_l4099_409997


namespace NUMINAMATH_CALUDE_truthful_dwarfs_count_l4099_409996

-- Define the total number of dwarfs
def total_dwarfs : ℕ := 10

-- Define the number of dwarfs who raised hands for each ice cream type
def vanilla_hands : ℕ := total_dwarfs
def chocolate_hands : ℕ := total_dwarfs / 2
def fruit_hands : ℕ := 1

-- Define the total number of hands raised
def total_hands_raised : ℕ := vanilla_hands + chocolate_hands + fruit_hands

-- Theorem to prove
theorem truthful_dwarfs_count : 
  ∃ (truthful : ℕ) (lying : ℕ), 
    truthful + lying = total_dwarfs ∧ 
    lying = total_hands_raised - total_dwarfs ∧
    truthful = 4 := by
  sorry

end NUMINAMATH_CALUDE_truthful_dwarfs_count_l4099_409996


namespace NUMINAMATH_CALUDE_sphere_volume_sphere_surface_area_sphere_surface_eq_cylinder_lateral_l4099_409901

/-- A structure representing a sphere contained in a cylinder -/
structure SphereInCylinder where
  r : ℝ  -- radius of the sphere and base of the cylinder
  h : ℝ  -- height of the cylinder
  sphere_diameter_eq_cylinder : h = 2 * r  -- diameter of sphere equals height of cylinder

/-- The volume of the sphere is (4/3)πr³ -/
theorem sphere_volume (s : SphereInCylinder) : 
  (4 / 3) * Real.pi * s.r ^ 3 = (2 / 3) * Real.pi * s.r ^ 2 * s.h := by sorry

/-- The surface area of the sphere is 4πr² -/
theorem sphere_surface_area (s : SphereInCylinder) :
  4 * Real.pi * s.r ^ 2 = (2 / 3) * (2 * Real.pi * s.r * s.h + 2 * Real.pi * s.r ^ 2) := by sorry

/-- The surface area of the sphere equals the lateral surface area of the cylinder -/
theorem sphere_surface_eq_cylinder_lateral (s : SphereInCylinder) :
  4 * Real.pi * s.r ^ 2 = 2 * Real.pi * s.r * s.h := by sorry

end NUMINAMATH_CALUDE_sphere_volume_sphere_surface_area_sphere_surface_eq_cylinder_lateral_l4099_409901


namespace NUMINAMATH_CALUDE_car_distance_proof_l4099_409987

/-- Calculates the distance traveled by a car given its speed and time -/
def distance_traveled (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- The speed of the car in miles per hour -/
def car_speed : ℝ := 80

/-- The time the car traveled in hours -/
def travel_time : ℝ := 4.5

theorem car_distance_proof : 
  distance_traveled car_speed travel_time = 360 := by sorry

end NUMINAMATH_CALUDE_car_distance_proof_l4099_409987


namespace NUMINAMATH_CALUDE_cuboid_volume_with_margin_eq_l4099_409952

/-- The volume of points inside or within two units of a cuboid with dimensions 5 by 6 by 8 units -/
def cuboid_volume_with_margin : ℝ := sorry

/-- The dimensions of the cuboid -/
def cuboid_dimensions : Fin 3 → ℕ
  | 0 => 5
  | 1 => 6
  | 2 => 8
  | _ => 0

/-- The margin around the cuboid -/
def margin : ℕ := 2

/-- Theorem stating that the volume of points inside or within two units of the cuboid 
    is equal to (2136 + 140π)/3 cubic units -/
theorem cuboid_volume_with_margin_eq : 
  cuboid_volume_with_margin = (2136 + 140 * Real.pi) / 3 := by sorry

end NUMINAMATH_CALUDE_cuboid_volume_with_margin_eq_l4099_409952


namespace NUMINAMATH_CALUDE_puzzle_solution_l4099_409927

def is_valid_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def is_four_digit_number (n : ℕ) : Prop := n ≥ 1000 ∧ n ≤ 9999

def distinct_digits (a b c d : ℕ) : Prop :=
  is_valid_digit a ∧ is_valid_digit b ∧ is_valid_digit c ∧ is_valid_digit d ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def construct_number (a b c d : ℕ) : ℕ := a * 1000 + b * 100 + c * 10 + d

theorem puzzle_solution :
  ∀ t h e a b g m,
    distinct_digits t h e a ∧
    distinct_digits b e t a ∧
    distinct_digits g a m m ∧
    is_four_digit_number (construct_number t h e a) ∧
    is_four_digit_number (construct_number b e t a) ∧
    is_four_digit_number (construct_number g a m m) ∧
    construct_number t h e a + construct_number b e t a = construct_number g a m m →
    t = 4 ∧ h = 9 ∧ e = 4 ∧ a = 0 ∧ b = 5 ∧ g = 1 ∧ m = 8 :=
by sorry

end NUMINAMATH_CALUDE_puzzle_solution_l4099_409927


namespace NUMINAMATH_CALUDE_inequality_solution_set_l4099_409932

theorem inequality_solution_set (m : ℝ) : 
  (∀ x : ℝ, x^2 + x - m > 0 ↔ x < -3 ∨ x > 2) → m = 6 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l4099_409932


namespace NUMINAMATH_CALUDE_cos_pi_half_plus_alpha_l4099_409942

theorem cos_pi_half_plus_alpha (α : Real) 
  (h : (Real.sin (π + α) * Real.cos (-α + 4*π)) / Real.cos α = 1/2) : 
  Real.cos (π/2 + α) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_cos_pi_half_plus_alpha_l4099_409942


namespace NUMINAMATH_CALUDE_remainder_three_to_89_plus_5_mod_7_l4099_409914

theorem remainder_three_to_89_plus_5_mod_7 : (3^89 + 5) % 7 = 3 := by sorry

end NUMINAMATH_CALUDE_remainder_three_to_89_plus_5_mod_7_l4099_409914


namespace NUMINAMATH_CALUDE_earn_twelve_points_l4099_409968

/-- Calculates the points earned in a video game level --/
def points_earned (total_enemies : ℕ) (enemies_not_defeated : ℕ) (points_per_enemy : ℕ) : ℕ :=
  (total_enemies - enemies_not_defeated) * points_per_enemy

/-- Theorem: In the given scenario, the player earns 12 points --/
theorem earn_twelve_points :
  points_earned 6 2 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_earn_twelve_points_l4099_409968


namespace NUMINAMATH_CALUDE_cost_price_calculation_l4099_409926

theorem cost_price_calculation (marked_price : ℝ) (selling_price_percent : ℝ) (profit_percent : ℝ) :
  marked_price = 62.5 →
  selling_price_percent = 0.95 →
  profit_percent = 1.25 →
  ∃ (cost_price : ℝ), cost_price = 47.5 ∧ 
    selling_price_percent * marked_price = profit_percent * cost_price :=
by
  sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l4099_409926


namespace NUMINAMATH_CALUDE_no_hammers_loaded_l4099_409985

theorem no_hammers_loaded (crate_capacity : ℕ) (num_crates : ℕ) (nail_bags : ℕ) (nail_weight : ℕ)
  (plank_bags : ℕ) (plank_weight : ℕ) (leave_out : ℕ) (hammer_weight : ℕ) :
  crate_capacity = 20 →
  num_crates = 15 →
  nail_bags = 4 →
  nail_weight = 5 →
  plank_bags = 10 →
  plank_weight = 30 →
  leave_out = 80 →
  hammer_weight = 5 →
  (∃ (loaded_planks : ℕ), 
    loaded_planks ≤ plank_bags * plank_weight ∧
    crate_capacity * num_crates - leave_out = nail_bags * nail_weight + loaded_planks) →
  (∀ (hammer_bags : ℕ), 
    crate_capacity * num_crates - leave_out < 
      nail_bags * nail_weight + plank_bags * plank_weight - leave_out + hammer_bags * hammer_weight) :=
by sorry

end NUMINAMATH_CALUDE_no_hammers_loaded_l4099_409985


namespace NUMINAMATH_CALUDE_grape_price_l4099_409947

/-- The price of each box of grapes given the following conditions:
  * 60 bundles of asparagus at $3.00 each
  * 40 boxes of grapes
  * 700 apples at $0.50 each
  * Total worth of the produce is $630
-/
theorem grape_price (asparagus_bundles : ℕ) (asparagus_price : ℚ)
                    (grape_boxes : ℕ) (apple_count : ℕ) (apple_price : ℚ)
                    (total_worth : ℚ) :
  asparagus_bundles = 60 →
  asparagus_price = 3 →
  grape_boxes = 40 →
  apple_count = 700 →
  apple_price = 1/2 →
  total_worth = 630 →
  (total_worth - (asparagus_bundles * asparagus_price + apple_count * apple_price)) / grape_boxes = 5/2 :=
by sorry

end NUMINAMATH_CALUDE_grape_price_l4099_409947


namespace NUMINAMATH_CALUDE_rug_area_l4099_409900

theorem rug_area (w l : ℝ) (h1 : l = w + 8) 
  (h2 : (w + 16) * (l + 16) - w * l = 704) : w * l = 180 := by
  sorry

end NUMINAMATH_CALUDE_rug_area_l4099_409900


namespace NUMINAMATH_CALUDE_math_score_difference_l4099_409999

def regression_equation (x : ℝ) : ℝ := 6 + 0.4 * x

theorem math_score_difference (x₁ x₂ : ℝ) (h : x₂ - x₁ = 50) :
  regression_equation x₂ - regression_equation x₁ = 20 := by
  sorry

end NUMINAMATH_CALUDE_math_score_difference_l4099_409999


namespace NUMINAMATH_CALUDE_meeting_time_calculation_l4099_409978

/-- Two people moving towards each other -/
structure TwoPersonMovement where
  v₁ : ℝ  -- Speed of person 1
  v₂ : ℝ  -- Speed of person 2
  t₂ : ℝ  -- Waiting time after turning around

/-- The theorem statement -/
theorem meeting_time_calculation (m : TwoPersonMovement) 
  (h₁ : m.v₁ = 6)   -- Speed of person 1 is 6 m/s
  (h₂ : m.v₂ = 4)   -- Speed of person 2 is 4 m/s
  (h₃ : m.t₂ = 600) -- Waiting time is 10 minutes (600 seconds)
  : ∃ t₁ : ℝ, t₁ = 1200 ∧ (m.v₁ * t₁ + m.v₂ * t₁ = 2 * m.v₂ * t₁ + m.v₂ * m.t₂) := by
  sorry

end NUMINAMATH_CALUDE_meeting_time_calculation_l4099_409978


namespace NUMINAMATH_CALUDE_fraction_equation_implies_sum_of_squares_l4099_409922

theorem fraction_equation_implies_sum_of_squares (A B : ℝ) : 
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ 1 → (2*x - 3) / (x^2 - x) = A / (x - 1) + B / x) →
  A^2 + B^2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_implies_sum_of_squares_l4099_409922


namespace NUMINAMATH_CALUDE_tan_alpha_value_l4099_409920

open Real

theorem tan_alpha_value (α : ℝ) (h_obtuse : π/2 < α ∧ α < π) 
  (h_eq : (sin α - 3 * cos α) / (cos α - sin α) = tan (2 * α)) : 
  tan α = 2 - Real.sqrt 7 := by
sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l4099_409920


namespace NUMINAMATH_CALUDE_gravel_calculation_l4099_409940

/-- The amount of gravel bought by a construction company -/
def gravel_amount : ℝ := 14.02 - 8.11

/-- The total amount of material bought by the construction company -/
def total_material : ℝ := 14.02

/-- The amount of sand bought by the construction company -/
def sand_amount : ℝ := 8.11

theorem gravel_calculation :
  gravel_amount = 5.91 ∧
  total_material = gravel_amount + sand_amount :=
sorry

end NUMINAMATH_CALUDE_gravel_calculation_l4099_409940


namespace NUMINAMATH_CALUDE_sum_divisible_by_1987_l4099_409953

def odd_product : ℕ := (List.range 993).foldl (λ acc i => acc * (2 * i + 1)) 1

def even_product : ℕ := (List.range 993).foldl (λ acc i => acc * (2 * i + 2)) 1

theorem sum_divisible_by_1987 : 
  ∃ k : ℤ, (odd_product : ℤ) + (even_product : ℤ) = 1987 * k := by
  sorry

end NUMINAMATH_CALUDE_sum_divisible_by_1987_l4099_409953


namespace NUMINAMATH_CALUDE_train_passing_time_l4099_409979

/-- Calculates the time for a train to pass a person moving in the opposite direction -/
theorem train_passing_time 
  (train_length : ℝ) 
  (train_speed : ℝ) 
  (person_speed : ℝ) 
  (h1 : train_length = 110) 
  (h2 : train_speed = 65) 
  (h3 : person_speed = 7) : 
  (train_length / ((train_speed + person_speed) * (5/18))) = 5.5 := by
  sorry

#check train_passing_time

end NUMINAMATH_CALUDE_train_passing_time_l4099_409979


namespace NUMINAMATH_CALUDE_shirt_cost_problem_l4099_409921

theorem shirt_cost_problem (total_shirts : ℕ) (known_shirt_count : ℕ) (known_shirt_cost : ℕ) (total_cost : ℕ) :
  total_shirts = 5 →
  known_shirt_count = 3 →
  known_shirt_cost = 15 →
  total_cost = 85 →
  (total_cost - known_shirt_count * known_shirt_cost) / (total_shirts - known_shirt_count) = 20 := by
sorry

end NUMINAMATH_CALUDE_shirt_cost_problem_l4099_409921


namespace NUMINAMATH_CALUDE_find_2a_plus_b_l4099_409966

-- Define the functions
def f (a b x : ℝ) : ℝ := 2 * a * x - 3 * b
def g (x : ℝ) : ℝ := 5 * x + 4
def h (a b x : ℝ) : ℝ := g (f a b x)

-- State the theorem
theorem find_2a_plus_b (a b : ℝ) :
  (∀ x, h a b (2 * x - 9) = x) →
  2 * a + b = 1 / 15 := by
  sorry

end NUMINAMATH_CALUDE_find_2a_plus_b_l4099_409966


namespace NUMINAMATH_CALUDE_divided_triangle_distance_l4099_409982

/-- Represents a triangle with a parallel line dividing its area -/
structure DividedTriangle where
  base : ℝ
  height : ℝ
  divisionRatio : ℝ
  heightAboveDivision : ℝ

/-- The theorem statement -/
theorem divided_triangle_distance (t : DividedTriangle) 
  (h1 : t.height = 3)
  (h2 : t.divisionRatio = 1/3)
  (h3 : t.heightAboveDivision = 2/3 * t.height) :
  t.height - t.heightAboveDivision = 1 := by
  sorry


end NUMINAMATH_CALUDE_divided_triangle_distance_l4099_409982


namespace NUMINAMATH_CALUDE_zoey_holiday_months_l4099_409954

/-- The number of holidays Zoey takes per month -/
def holidays_per_month : ℕ := 2

/-- The total number of holidays Zoey took -/
def total_holidays : ℕ := 24

/-- The number of months Zoey took holidays for -/
def months_of_holidays : ℕ := total_holidays / holidays_per_month

/-- Theorem: The number of months Zoey took holidays for is 12 -/
theorem zoey_holiday_months : months_of_holidays = 12 := by
  sorry

end NUMINAMATH_CALUDE_zoey_holiday_months_l4099_409954
