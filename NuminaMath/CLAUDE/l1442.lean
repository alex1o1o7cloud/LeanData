import Mathlib

namespace pipe_fill_time_l1442_144205

/-- Given a pipe and a tank with a leak, this theorem proves the time taken for the pipe
    to fill the tank alone, based on the time taken to fill with both pipe and leak,
    and the time taken for the leak to empty the tank. -/
theorem pipe_fill_time (fill_time_with_leak : ℝ) (leak_empty_time : ℝ) 
    (h1 : fill_time_with_leak = 18) 
    (h2 : leak_empty_time = 36) : 
    (1 : ℝ) / ((1 : ℝ) / fill_time_with_leak + (1 : ℝ) / leak_empty_time) = 12 := by
  sorry

end pipe_fill_time_l1442_144205


namespace g_of_3_equals_2_l1442_144289

-- Define the function g
def g (x : ℝ) : ℝ := x^2 - 3*x + 2

-- Theorem statement
theorem g_of_3_equals_2 : g 3 = 2 := by sorry

end g_of_3_equals_2_l1442_144289


namespace sequence_properties_l1442_144209

/-- Given a sequence {a_n} with sum of first n terms S_n, satisfying a_n = 2S_n + 1 for n ∈ ℕ* -/
def a (n : ℕ) : ℤ := sorry

/-- Sum of first n terms of sequence {a_n} -/
def S (n : ℕ) : ℤ := sorry

/-- Sequence {b_n} defined as b_n = (2n-1) * a_n -/
def b (n : ℕ) : ℤ := sorry

/-- Sum of first n terms of sequence {b_n} -/
def T (n : ℕ) : ℤ := sorry

theorem sequence_properties :
  (∀ n : ℕ, n > 0 → a n = 2 * S n + 1) →
  (∀ n : ℕ, a n = (-1)^n) ∧
  (∀ n : ℕ, T n = (-1)^n * n) := by sorry

end sequence_properties_l1442_144209


namespace parabola_focus_directrix_l1442_144213

/-- For a parabola y = ax^2 where a > 0, if the distance from the focus to the directrix is 2, then a = 1/4 -/
theorem parabola_focus_directrix (a : ℝ) (h1 : a > 0) : 
  (∃ (f d : ℝ), ∀ (x y : ℝ), y = a * x^2 ∧ |f - d| = 2) → a = 1/4 := by
  sorry

end parabola_focus_directrix_l1442_144213


namespace new_student_weight_l1442_144283

theorem new_student_weight
  (initial_students : ℕ)
  (initial_avg_weight : ℝ)
  (new_avg_weight : ℝ)
  (h1 : initial_students = 19)
  (h2 : initial_avg_weight = 15)
  (h3 : new_avg_weight = 14.8) :
  (initial_students + 1) * new_avg_weight - initial_students * initial_avg_weight = 11 :=
by sorry

end new_student_weight_l1442_144283


namespace complex_exponential_to_rectangular_l1442_144201

theorem complex_exponential_to_rectangular : Complex.exp (13 * Real.pi * Complex.I / 6) = Complex.mk (Real.sqrt 3 / 2) (1 / 2) := by sorry

end complex_exponential_to_rectangular_l1442_144201


namespace parabola_y_axis_intersection_l1442_144268

/-- The parabola is defined by the equation y = x^2 - 3x - 4 -/
def parabola (x y : ℝ) : Prop := y = x^2 - 3*x - 4

/-- The y-axis is defined by x = 0 -/
def y_axis (x : ℝ) : Prop := x = 0

/-- Theorem: The intersection point of the parabola y = x^2 - 3x - 4 with the y-axis has coordinates (0, -4) -/
theorem parabola_y_axis_intersection :
  ∃ (x y : ℝ), parabola x y ∧ y_axis x ∧ x = 0 ∧ y = -4 :=
sorry

end parabola_y_axis_intersection_l1442_144268


namespace complex_number_real_l1442_144297

theorem complex_number_real (a : ℝ) : 
  (∃ (r : ℝ), Complex.mk r 0 = Complex.mk 0 2 - (Complex.I * a) / (1 - Complex.I)) → a = 4 := by
  sorry

end complex_number_real_l1442_144297


namespace inequality_proof_l1442_144295

theorem inequality_proof (a b c d : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (h_eq : a * (c^2 - 1) = b * (b^2 + c^2)) 
  (h_d : d ≤ 1) : 
  d * (a * Real.sqrt (1 - d^2) + b^2 * Real.sqrt (1 + d^2)) ≤ (a + b) * c / 2 := by
  sorry

end inequality_proof_l1442_144295


namespace estimate_expression_range_l1442_144261

theorem estimate_expression_range : 
  5 < (2 * Real.sqrt 5 + 5 * Real.sqrt 2) * Real.sqrt (1/5) ∧
  (2 * Real.sqrt 5 + 5 * Real.sqrt 2) * Real.sqrt (1/5) < 6 := by
  sorry

end estimate_expression_range_l1442_144261


namespace geometric_sequence_fourth_term_l1442_144250

/-- Represents a geometric sequence -/
structure GeometricSequence where
  a : ℕ → ℚ
  q : ℚ
  geom_prop : ∀ n : ℕ, a (n + 1) = a n * q

/-- Sum of the first n terms of a geometric sequence -/
def GeometricSequence.sum (g : GeometricSequence) (n : ℕ) : ℚ :=
  if g.q = 1 then n * g.a 1 else g.a 1 * (1 - g.q ^ n) / (1 - g.q)

theorem geometric_sequence_fourth_term 
  (g : GeometricSequence) 
  (h1 : g.a 1 - g.a 5 = -15/2) 
  (h2 : g.sum 4 = -5) : 
  g.a 4 = 1 := by
  sorry

end geometric_sequence_fourth_term_l1442_144250


namespace theater_construction_cost_ratio_l1442_144254

/-- Proves that the ratio of construction cost to land cost is 2:1 given the theater construction scenario --/
theorem theater_construction_cost_ratio :
  let cost_per_sqft : ℝ := 5
  let space_per_seat : ℝ := 12
  let num_seats : ℕ := 500
  let partner_share : ℝ := 0.4
  let tom_spent : ℝ := 54000

  let total_sqft : ℝ := space_per_seat * num_seats
  let land_cost : ℝ := total_sqft * cost_per_sqft
  let total_cost : ℝ := tom_spent / (1 - partner_share)
  let construction_cost : ℝ := total_cost - land_cost

  construction_cost / land_cost = 2 := by sorry

end theater_construction_cost_ratio_l1442_144254


namespace final_price_after_discounts_arun_paid_price_l1442_144273

/-- Calculates the final price of an article after applying two consecutive discounts -/
theorem final_price_after_discounts (original_price : ℝ) 
  (standard_discount : ℝ) (additional_discount : ℝ) : ℝ :=
  let price_after_standard := original_price * (1 - standard_discount)
  let final_price := price_after_standard * (1 - additional_discount)
  final_price

/-- Proves that the final price of an article originally priced at 2000, 
    after a 30% standard discount and a 20% additional discount, is 1120 -/
theorem arun_paid_price : 
  final_price_after_discounts 2000 0.3 0.2 = 1120 := by
  sorry

end final_price_after_discounts_arun_paid_price_l1442_144273


namespace permutations_with_four_transpositions_l1442_144259

/-- The number of elements in the permutation -/
def n : ℕ := 6

/-- The total number of permutations of n elements -/
def total_permutations : ℕ := n.factorial

/-- The number of even permutations -/
def even_permutations : ℕ := total_permutations / 2

/-- The number of permutations that require i transpositions to become the identity permutation -/
def num_permutations (i : ℕ) : ℕ := sorry

/-- The theorem stating that the number of permutations requiring 4 transpositions is 304 -/
theorem permutations_with_four_transpositions :
  num_permutations 4 = 304 :=
sorry

end permutations_with_four_transpositions_l1442_144259


namespace vector_operation_proof_l1442_144229

def v1 : Fin 3 → ℝ := ![3, -2, 5]
def v2 : Fin 3 → ℝ := ![-1, 6, -3]

theorem vector_operation_proof :
  (2 : ℝ) • (v1 + v2) = ![4, 8, 4] := by sorry

end vector_operation_proof_l1442_144229


namespace line_equation_l1442_144260

/-- The equation of a line passing through the intersection of two given lines and parallel to a third line -/
theorem line_equation (x y : ℝ) : 
  (2 * x - 3 * y - 3 = 0) →   -- First given line
  (x + y + 2 = 0) →           -- Second given line
  (∃ k : ℝ, 3 * x + y - k = 0) →  -- Parallel line condition
  (15 * x + 5 * y + 16 = 0) := by  -- Equation to prove
sorry

end line_equation_l1442_144260


namespace nadia_playing_time_l1442_144246

/-- Represents the number of mistakes Nadia makes per 40 notes -/
def mistakes_per_40_notes : ℚ := 3

/-- Represents the number of notes Nadia can play per minute -/
def notes_per_minute : ℚ := 60

/-- Represents the total number of mistakes Nadia made -/
def total_mistakes : ℚ := 36

/-- Calculates the number of minutes Nadia played -/
def minutes_played : ℚ :=
  total_mistakes / (mistakes_per_40_notes * notes_per_minute / 40)

theorem nadia_playing_time :
  minutes_played = 8 := by sorry

end nadia_playing_time_l1442_144246


namespace line_circle_relationship_l1442_144228

theorem line_circle_relationship (k : ℝ) : 
  ∃ (x y : ℝ), (x - k*y + 1 = 0) ∧ (x^2 + y^2 = 1) :=
by sorry

end line_circle_relationship_l1442_144228


namespace arithmetic_sequence_third_term_l1442_144285

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_third_term 
  (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_sum : a 1 + a 2 + a 3 + a 4 + a 5 = 20) : 
  a 3 = 4 := by
sorry

end arithmetic_sequence_third_term_l1442_144285


namespace rap_song_requests_l1442_144203

/-- Represents the number of song requests for different genres in a night --/
structure SongRequests where
  total : ℕ
  electropop : ℕ
  dance : ℕ
  rock : ℕ
  oldies : ℕ
  dj_choice : ℕ
  rap : ℕ

/-- Theorem stating the number of rap song requests given the conditions --/
theorem rap_song_requests (r : SongRequests) : r.rap = 2 :=
  by
  have h1 : r.total = 30 := by sorry
  have h2 : r.electropop = r.total / 2 := by sorry
  have h3 : r.dance = r.electropop / 3 := by sorry
  have h4 : r.rock = 5 := by sorry
  have h5 : r.oldies = r.rock - 3 := by sorry
  have h6 : r.dj_choice = r.oldies / 2 := by sorry
  have h7 : r.total = r.electropop + r.dance + r.rock + r.oldies + r.dj_choice + r.rap := by sorry
  sorry

#check rap_song_requests

end rap_song_requests_l1442_144203


namespace at_least_one_less_than_two_l1442_144263

theorem at_least_one_less_than_two (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y > 2) :
  min ((1 + y) / x) ((1 + x) / y) < 2 := by
  sorry

end at_least_one_less_than_two_l1442_144263


namespace ellipse_focus_circle_radius_l1442_144232

/-- The radius of a circle centered at a focus of an ellipse and tangent to it -/
theorem ellipse_focus_circle_radius 
  (a b : ℝ) 
  (ha : a = 6) 
  (hb : b = 5) 
  (h_ellipse : a > b) 
  (h_positive : a > 0 ∧ b > 0) :
  let c := Real.sqrt (a^2 - b^2)
  let r := Real.sqrt ((a + c)^2 - a^2)
  r = Real.sqrt 705 / 6 :=
by sorry

end ellipse_focus_circle_radius_l1442_144232


namespace line_chart_best_for_fever_temperature_l1442_144224

/- Define the types of charts -/
inductive ChartType
| Bar
| Line
| Pie

/- Define the properties of data we want to visualize -/
structure TemperatureData where
  showsQuantity : Bool
  showsChanges : Bool
  showsRelationship : Bool

/- Define the characteristics of fever temperature data -/
def feverTemperatureData : TemperatureData :=
  { showsQuantity := true
  , showsChanges := true
  , showsRelationship := false }

/- Define which chart types are suitable for different data properties -/
def suitableChartType (data : TemperatureData) : ChartType :=
  if data.showsChanges then ChartType.Line
  else if data.showsQuantity then ChartType.Bar
  else ChartType.Pie

/- Theorem: Line chart is the best for tracking fever temperature changes -/
theorem line_chart_best_for_fever_temperature : 
  suitableChartType feverTemperatureData = ChartType.Line := by
  sorry

end line_chart_best_for_fever_temperature_l1442_144224


namespace value_range_of_f_l1442_144253

def f (x : ℝ) : ℝ := x^2 - 4*x + 3

theorem value_range_of_f :
  ∀ y ∈ Set.Icc (-1 : ℝ) 3, ∃ x ∈ Set.Icc 0 3, f x = y ∧
  ∀ x ∈ Set.Icc 0 3, f x ∈ Set.Icc (-1 : ℝ) 3 :=
sorry

end value_range_of_f_l1442_144253


namespace sum_of_digits_congruence_l1442_144234

def S (n : ℕ+) : ℕ := sorry

theorem sum_of_digits_congruence (n : ℕ+) (h : S n = 29) : 
  S (n + 1) % 9 = 3 := by sorry

end sum_of_digits_congruence_l1442_144234


namespace tuesday_distance_l1442_144274

/-- Proves that the distance driven on Tuesday is 18 miles -/
theorem tuesday_distance (monday_distance : ℝ) (wednesday_distance : ℝ) (average_distance : ℝ) (num_days : ℕ) :
  monday_distance = 12 →
  wednesday_distance = 21 →
  average_distance = 17 →
  num_days = 3 →
  (monday_distance + wednesday_distance + (num_days * average_distance - monday_distance - wednesday_distance)) / num_days = average_distance →
  num_days * average_distance - monday_distance - wednesday_distance = 18 := by
  sorry

end tuesday_distance_l1442_144274


namespace jake_and_sister_weight_l1442_144272

/-- Given Jake's current weight and the condition about his weight relative to his sister's,
    prove that their combined weight is 212 pounds. -/
theorem jake_and_sister_weight (jake_weight : ℕ) (sister_weight : ℕ) : 
  jake_weight = 152 →
  jake_weight - 32 = 2 * sister_weight →
  jake_weight + sister_weight = 212 := by
  sorry

end jake_and_sister_weight_l1442_144272


namespace michael_has_270_eggs_l1442_144296

/-- Calculates the number of eggs Michael has after buying and giving away crates. -/
def michaels_eggs (initial_crates : ℕ) (given_crates : ℕ) (bought_crates : ℕ) (eggs_per_crate : ℕ) : ℕ :=
  (initial_crates - given_crates + bought_crates) * eggs_per_crate

/-- Proves that Michael has 270 eggs after his transactions. -/
theorem michael_has_270_eggs :
  michaels_eggs 6 2 5 30 = 270 := by
  sorry

end michael_has_270_eggs_l1442_144296


namespace headmaster_retirement_l1442_144269

/-- Represents the months of the year -/
inductive Month
| January | February | March | April | May | June
| July | August | September | October | November | December

/-- Returns the month that is n months after the given month -/
def monthsAfter (start : Month) (n : ℕ) : Month :=
  match n with
  | 0 => start
  | n + 1 => monthsAfter (match start with
    | Month.January => Month.February
    | Month.February => Month.March
    | Month.March => Month.April
    | Month.April => Month.May
    | Month.May => Month.June
    | Month.June => Month.July
    | Month.July => Month.August
    | Month.August => Month.September
    | Month.September => Month.October
    | Month.October => Month.November
    | Month.November => Month.December
    | Month.December => Month.January
  ) n

theorem headmaster_retirement (start_month : Month) (duration : ℕ) :
  start_month = Month.March → duration = 3 →
  monthsAfter start_month duration = Month.May :=
by
  sorry

end headmaster_retirement_l1442_144269


namespace unique_solution_for_equation_l1442_144223

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Main theorem -/
theorem unique_solution_for_equation :
  ∀ (m n : ℕ), n * (n + 1) = 3^m + sum_of_digits n + 1182 → m = 0 ∧ n = 34 := by
  sorry

end unique_solution_for_equation_l1442_144223


namespace product_of_roots_l1442_144233

theorem product_of_roots (x₁ x₂ : ℝ) : 
  x₁^2 - 6*x₁ + 8 = 0 → x₂^2 - 6*x₂ + 8 = 0 → x₁ * x₂ = 8 := by
  sorry

end product_of_roots_l1442_144233


namespace train_crossing_time_l1442_144207

/-- Proves that a train with given length and speed takes a specific time to cross a fixed point -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 140 →
  train_speed_kmh = 31.5 →
  crossing_time = 16 →
  train_length / (train_speed_kmh * 1000 / 3600) = crossing_time :=
by
  sorry

#check train_crossing_time

end train_crossing_time_l1442_144207


namespace work_completed_in_three_days_l1442_144286

-- Define the work rates for A, B, and C
def work_rate_A : ℚ := 1 / 4
def work_rate_B : ℚ := 1 / 14
def work_rate_C : ℚ := 1 / 7

-- Define the total work to be done
def total_work : ℚ := 1

-- Define the work done in the first two days by A and B
def work_done_first_two_days : ℚ := 2 * (work_rate_A + work_rate_B)

-- Define the work done on the third day by A, B, and C
def work_done_third_day : ℚ := work_rate_A + work_rate_B + work_rate_C

-- Theorem to prove
theorem work_completed_in_three_days :
  work_done_first_two_days + work_done_third_day ≥ total_work :=
by sorry

end work_completed_in_three_days_l1442_144286


namespace second_meeting_time_l1442_144238

/-- The time (in seconds) it takes for the racing magic to complete one round -/
def racing_magic_time : ℕ := 60

/-- The time (in seconds) it takes for the charging bull to complete one round -/
def charging_bull_time : ℕ := 90

/-- The time (in minutes) it takes for both objects to meet at the starting point for the second time -/
def meeting_time : ℕ := 3

/-- Theorem stating that the meeting time is correct given the individual round times -/
theorem second_meeting_time (racing_time : ℕ) (bull_time : ℕ) (meet_time : ℕ) 
  (h1 : racing_time = racing_magic_time)
  (h2 : bull_time = charging_bull_time)
  (h3 : meet_time = meeting_time) :
  Nat.lcm racing_time bull_time = meet_time * 60 := by
  sorry

end second_meeting_time_l1442_144238


namespace probability_of_meeting_theorem_l1442_144265

/-- Represents the practice schedule of a person --/
structure PracticeSchedule where
  start_time : ℝ
  duration : ℝ

/-- Represents the practice schedules of two people over multiple days --/
structure PracticeScenario where
  your_schedule : PracticeSchedule
  friend_schedule : PracticeSchedule
  num_days : ℕ

/-- Calculates the probability of meeting given two practice schedules --/
def probability_of_meeting (s : PracticeScenario) : ℝ :=
  sorry

/-- Calculates the probability of meeting on at least k days out of n days --/
def probability_of_meeting_at_least (s : PracticeScenario) (k : ℕ) : ℝ :=
  sorry

theorem probability_of_meeting_theorem :
  let s : PracticeScenario := {
    your_schedule := { start_time := 0, duration := 3 },
    friend_schedule := { start_time := 5, duration := 1 },
    num_days := 5
  }
  probability_of_meeting_at_least s 2 = 232 / 243 := by
  sorry

end probability_of_meeting_theorem_l1442_144265


namespace solve_system_l1442_144257

theorem solve_system (C D : ℚ) 
  (eq1 : 3 * C - 4 * D = 18)
  (eq2 : C = 2 * D - 5) : 
  C = 28 ∧ D = 33 / 2 := by
  sorry

end solve_system_l1442_144257


namespace bill_difference_zero_l1442_144244

theorem bill_difference_zero (anna_tip : ℝ) (anna_percent : ℝ) 
  (ben_tip : ℝ) (ben_percent : ℝ) 
  (h1 : anna_tip = 5) 
  (h2 : anna_percent = 25 / 100)
  (h3 : ben_tip = 3)
  (h4 : ben_percent = 15 / 100)
  (h5 : anna_tip = anna_percent * anna_bill)
  (h6 : ben_tip = ben_percent * ben_bill) :
  anna_bill - ben_bill = 0 :=
by
  sorry

#check bill_difference_zero

end bill_difference_zero_l1442_144244


namespace a_6_equals_8_l1442_144208

def S (n : ℕ+) : ℤ := n^2 - 3*n

theorem a_6_equals_8 : ∃ (a : ℕ+ → ℤ), a 6 = 8 ∧ ∀ n : ℕ+, S n - S (n-1) = a n :=
sorry

end a_6_equals_8_l1442_144208


namespace gcd_7654321_6789012_l1442_144249

theorem gcd_7654321_6789012 : Nat.gcd 7654321 6789012 = 3 := by
  sorry

end gcd_7654321_6789012_l1442_144249


namespace max_cupcakes_eaten_l1442_144294

/-- Given 30 cupcakes shared among three people, where one person eats twice as much as the first
    and the same as the second, the maximum number of cupcakes the first person could have eaten is 6. -/
theorem max_cupcakes_eaten (total : ℕ) (ben charles diana : ℕ) : 
  total = 30 →
  diana = 2 * ben →
  diana = charles →
  total = ben + charles + diana →
  ben ≤ 6 ∧ ∃ ben', ben' = 6 ∧ 
    ∃ charles' diana', 
      diana' = 2 * ben' ∧ 
      diana' = charles' ∧ 
      total = ben' + charles' + diana' :=
by sorry

end max_cupcakes_eaten_l1442_144294


namespace program_schedule_arrangements_l1442_144267

theorem program_schedule_arrangements (n : ℕ) (h : n = 6) : 
  (n + 1).choose 1 * (n + 2).choose 1 = 56 := by
  sorry

end program_schedule_arrangements_l1442_144267


namespace lilith_water_bottles_l1442_144266

/-- The number of water bottles Lilith originally had -/
def num_bottles : ℕ := 60

/-- The original selling price per bottle in dollars -/
def original_price : ℚ := 2

/-- The reduced selling price per bottle in dollars -/
def reduced_price : ℚ := 185/100

theorem lilith_water_bottles :
  (original_price * num_bottles : ℚ) - (reduced_price * num_bottles) = 9 :=
sorry

end lilith_water_bottles_l1442_144266


namespace barry_sotter_magic_barry_sotter_days_l1442_144293

theorem barry_sotter_magic (n : ℕ) : (3/2 : ℝ)^n ≥ 50 ↔ n ≥ 10 := by sorry

theorem barry_sotter_days : ∃ (n : ℕ), (∀ (m : ℕ), (3/2 : ℝ)^m ≥ 50 → n ≤ m) ∧ (3/2 : ℝ)^n ≥ 50 :=
by
  use 10
  sorry

end barry_sotter_magic_barry_sotter_days_l1442_144293


namespace fraction_value_l1442_144248

theorem fraction_value (x y : ℝ) (h : 1 / x - 1 / y = 3) :
  (2 * x + 3 * x * y - 2 * y) / (x - 2 * x * y - y) = 3 / 5 := by
  sorry

end fraction_value_l1442_144248


namespace expression_value_l1442_144284

theorem expression_value (x y : ℝ) (h : x - 2*y = -4) :
  (2*y - x)^2 - 2*x + 4*y - 1 = 23 := by
  sorry

end expression_value_l1442_144284


namespace a_eq_one_sufficient_not_necessary_l1442_144236

def z (a : ℝ) : ℂ := Complex.mk (a^2 - 1) (a - 2)

theorem a_eq_one_sufficient_not_necessary :
  (∀ a : ℝ, a = 1 → z a = Complex.I * Complex.im (z a)) ∧
  (∃ a : ℝ, a ≠ 1 ∧ z a = Complex.I * Complex.im (z a)) :=
by sorry

end a_eq_one_sufficient_not_necessary_l1442_144236


namespace max_revenue_is_50_l1442_144221

def neighborhood_A_homes : ℕ := 10
def neighborhood_A_boxes_per_home : ℕ := 2
def neighborhood_B_homes : ℕ := 5
def neighborhood_B_boxes_per_home : ℕ := 5
def price_per_box : ℕ := 2

def revenue_A : ℕ := neighborhood_A_homes * neighborhood_A_boxes_per_home * price_per_box
def revenue_B : ℕ := neighborhood_B_homes * neighborhood_B_boxes_per_home * price_per_box

theorem max_revenue_is_50 : max revenue_A revenue_B = 50 := by
  sorry

end max_revenue_is_50_l1442_144221


namespace quadratic_non_real_roots_l1442_144270

theorem quadratic_non_real_roots (b : ℝ) : 
  (∀ x : ℂ, x^2 + b*x + 16 = 0 → x.im ≠ 0) ↔ -8 < b ∧ b < 8 := by
  sorry

end quadratic_non_real_roots_l1442_144270


namespace bob_time_improvement_l1442_144243

/-- 
Given Bob's current mile time and his sister's mile time in seconds,
calculate the percentage improvement Bob needs to match his sister's time.
-/
theorem bob_time_improvement (bob_time sister_time : ℕ) :
  bob_time = 640 ∧ sister_time = 608 →
  (bob_time - sister_time : ℚ) / bob_time * 100 = 5 := by
sorry

end bob_time_improvement_l1442_144243


namespace mrs_anderson_pet_food_l1442_144204

/-- Calculates the total ounces of pet food bought by Mrs. Anderson -/
def total_pet_food_ounces (cat_food_bags : ℕ) (cat_food_weight : ℕ) 
  (dog_food_bags : ℕ) (dog_food_extra_weight : ℕ) (ounces_per_pound : ℕ) : ℕ :=
  let total_cat_food := cat_food_bags * cat_food_weight
  let total_dog_food := dog_food_bags * (cat_food_weight + dog_food_extra_weight)
  let total_pounds := total_cat_food + total_dog_food
  total_pounds * ounces_per_pound

/-- Theorem stating that Mrs. Anderson bought 256 ounces of pet food -/
theorem mrs_anderson_pet_food : 
  total_pet_food_ounces 2 3 2 2 16 = 256 := by
  sorry

end mrs_anderson_pet_food_l1442_144204


namespace f_range_l1442_144264

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 2*x

-- Define the domain
def domain : Set ℝ := {x | 0 ≤ x ∧ x ≤ 3}

-- Define the range
def range : Set ℝ := {y | ∃ x ∈ domain, f x = y}

-- Theorem statement
theorem f_range : range = {y | -1 ≤ y ∧ y ≤ 3} := by sorry

end f_range_l1442_144264


namespace angle_C_is_84_l1442_144299

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_180 : A + B + C = 180
  positive : 0 < A ∧ 0 < B ∧ 0 < C

-- Define the ratio condition
def ratio_condition (t : Triangle) : Prop :=
  ∃ (k : ℝ), t.A = 4*k ∧ t.B = 4*k ∧ t.C = 7*k

-- Theorem statement
theorem angle_C_is_84 (t : Triangle) (h : ratio_condition t) : t.C = 84 :=
  sorry

end angle_C_is_84_l1442_144299


namespace diver_min_trips_l1442_144211

/-- The minimum number of trips required to carry all objects to the surface -/
def min_trips (capacity : ℕ) (total_objects : ℕ) : ℕ :=
  (total_objects + capacity - 1) / capacity

/-- Theorem: Given a diver who can carry 3 objects at a time and has found 17 objects,
    the minimum number of trips required to carry all objects to the surface is 6 -/
theorem diver_min_trips :
  min_trips 3 17 = 6 := by
  sorry

end diver_min_trips_l1442_144211


namespace simplify_expression_solve_quadratic_equation_l1442_144298

-- Part 1
theorem simplify_expression :
  Real.sqrt 18 / Real.sqrt 9 - Real.sqrt (1/4) * 2 * Real.sqrt 2 + Real.sqrt 32 = 4 * Real.sqrt 2 := by
  sorry

-- Part 2
theorem solve_quadratic_equation :
  ∀ x : ℝ, x^2 - 2*x = 3 ↔ x = 3 ∨ x = -1 := by
  sorry

end simplify_expression_solve_quadratic_equation_l1442_144298


namespace employed_males_percentage_l1442_144202

/-- In a population where 60% are employed and 30% of the employed are females,
    the percentage of employed males in the total population is 42%. -/
theorem employed_males_percentage
  (total : ℕ) -- Total population
  (employed_ratio : ℚ) -- Ratio of employed people to total population
  (employed_females_ratio : ℚ) -- Ratio of employed females to employed people
  (h1 : employed_ratio = 60 / 100)
  (h2 : employed_females_ratio = 30 / 100)
  : (employed_ratio * (1 - employed_females_ratio)) * 100 = 42 := by
  sorry

end employed_males_percentage_l1442_144202


namespace total_ways_eq_7464_l1442_144280

def num_oreo_flavors : ℕ := 6
def num_milk_flavors : ℕ := 4
def total_products : ℕ := 5

def ways_to_choose (n k : ℕ) : ℕ := Nat.choose n k

def alpha_choices (k : ℕ) : ℕ := ways_to_choose (num_oreo_flavors + num_milk_flavors) k

def beta_choices (k : ℕ) : ℕ :=
  if k = 0 then 1
  else if k = 1 then num_oreo_flavors
  else if k = 2 then ways_to_choose num_oreo_flavors 2 + num_oreo_flavors
  else if k = 3 then ways_to_choose num_oreo_flavors 3 + num_oreo_flavors * (num_oreo_flavors - 1) + num_oreo_flavors
  else if k = 4 then ways_to_choose num_oreo_flavors 4 + num_oreo_flavors * ways_to_choose (num_oreo_flavors - 1) 1 + num_oreo_flavors
  else ways_to_choose num_oreo_flavors 5 + num_oreo_flavors * ways_to_choose (num_oreo_flavors - 1) 1 + 
       num_oreo_flavors * ways_to_choose (num_oreo_flavors - 1) 2 + num_oreo_flavors

def total_ways : ℕ := 
  (Finset.range (total_products + 1)).sum (λ k => alpha_choices k * beta_choices (total_products - k))

theorem total_ways_eq_7464 : total_ways = 7464 := by sorry

end total_ways_eq_7464_l1442_144280


namespace immediate_prepayment_better_l1442_144230

variable (S T r : ℝ)

-- S: initial loan balance
-- T: monthly payment amount
-- r: interest rate for the period

-- Assumption: All variables are positive and r is between 0 and 1
axiom S_pos : S > 0
axiom T_pos : T > 0
axiom r_pos : r > 0
axiom r_lt_one : r < 1

-- Define the final balance for immediate prepayment
def final_balance_immediate (S T r : ℝ) : ℝ :=
  S - 2*T + r*S - 0.5*r*T + (0.5*r*S)^2

-- Define the final balance for waiting until the end of the period
def final_balance_waiting (S T r : ℝ) : ℝ :=
  S - 2*T + r*S

-- Theorem: Immediate prepayment results in a lower final balance
theorem immediate_prepayment_better :
  final_balance_immediate S T r < final_balance_waiting S T r :=
sorry

end immediate_prepayment_better_l1442_144230


namespace four_monotonic_intervals_condition_l1442_144252

/-- A function f(x) defined by a quadratic expression inside an absolute value. -/
def f (m : ℝ) (x : ℝ) : ℝ := |m * x^2 - (2*m + 1) * x + (m + 2)|

/-- The property of having exactly four monotonic intervals. -/
def has_four_monotonic_intervals (g : ℝ → ℝ) : Prop := sorry

/-- The main theorem stating the conditions on m for f to have exactly four monotonic intervals. -/
theorem four_monotonic_intervals_condition (m : ℝ) :
  has_four_monotonic_intervals (f m) ↔ m < (1/4 : ℝ) ∧ m ≠ 0 :=
sorry

end four_monotonic_intervals_condition_l1442_144252


namespace system_solution_conditions_l1442_144290

theorem system_solution_conditions (a b x y z : ℝ) : 
  (x + y + z = a) →
  (x^2 + y^2 + z^2 = b^2) →
  (x * y = z^2) →
  (x > 0 ∧ y > 0 ∧ z > 0 ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z) ↔ 
  (3 * b^2 > a^2 ∧ a^2 > b^2 ∧ a > 0) :=
by sorry

end system_solution_conditions_l1442_144290


namespace correlation_significance_l1442_144255

-- Define r as a real number representing a correlation coefficient
variable (r : ℝ)

-- Define r_0.05 as the critical value for a 5% significance level
variable (r_0_05 : ℝ)

-- Define a function that represents the probability of an event
def event_probability (r : ℝ) (r_0_05 : ℝ) : Prop :=
  ∃ p : ℝ, p < 0.05 ∧ (|r| > r_0_05 ↔ p < 0.05)

-- Theorem stating the equivalence
theorem correlation_significance (r : ℝ) (r_0_05 : ℝ) :
  |r| > r_0_05 ↔ event_probability r r_0_05 :=
sorry

end correlation_significance_l1442_144255


namespace parallel_implies_a_eq_2_perpendicular_implies_a_eq_neg3_or_0_l1442_144215

-- Define the lines l₁ and l₂
def l₁ (a x y : ℝ) : Prop := a * x - y + 2 = 0
def l₂ (a x y : ℝ) : Prop := (a + 2) * x - a * y - 2 = 0

-- Define parallel and perpendicular relations
def parallel (a : ℝ) : Prop := ∀ x y, l₁ a x y ↔ l₂ a x y
def perpendicular (a : ℝ) : Prop := ∀ x₁ y₁ x₂ y₂, l₁ a x₁ y₁ → l₂ a x₂ y₂ → 
  (x₂ - x₁) * (y₂ - y₁) = 0

-- Theorem statements
theorem parallel_implies_a_eq_2 : ∀ a : ℝ, parallel a → a = 2 := by sorry

theorem perpendicular_implies_a_eq_neg3_or_0 : ∀ a : ℝ, perpendicular a → a = -3 ∨ a = 0 := by sorry

end parallel_implies_a_eq_2_perpendicular_implies_a_eq_neg3_or_0_l1442_144215


namespace simplify_fraction_l1442_144239

theorem simplify_fraction (a : ℝ) (ha : a > 0) :
  a^2 / (a^(1/2) * a^(2/3)) = a^(5/6) := by sorry

end simplify_fraction_l1442_144239


namespace cuboids_painted_l1442_144251

theorem cuboids_painted (total_faces : ℕ) (faces_per_cuboid : ℕ) : 
  total_faces = 60 → faces_per_cuboid = 6 → total_faces / faces_per_cuboid = 10 := by
  sorry

end cuboids_painted_l1442_144251


namespace cyclists_speeds_l1442_144256

/-- Represents the scenario of two cyclists riding towards each other -/
structure CyclistsScenario where
  x : ℝ  -- Speed of the first cyclist in km/h
  y : ℝ  -- Speed of the second cyclist in km/h
  AB : ℝ  -- Distance between the two starting points in km

/-- Condition 1: If the first cyclist starts 1 hour earlier and the second one starts half an hour later,
    they meet 18 minutes earlier than normal -/
def condition1 (s : CyclistsScenario) : Prop :=
  (s.AB / (s.x + s.y) + 1 - 18/60) * s.x + (s.AB / (s.x + s.y) - 1/2 - 18/60) * s.y = s.AB

/-- Condition 2: If the first cyclist starts half an hour later and the second one starts 1 hour earlier,
    the meeting point moves by 11.2 km (11200 meters) -/
def condition2 (s : CyclistsScenario) : Prop :=
  (s.AB - 1.5 * s.y) / (s.x + s.y) * s.x + 11.2 = s.AB / (s.x + s.y) * s.x

/-- Theorem stating that given the conditions, the speeds of the cyclists are 16 km/h and 14 km/h -/
theorem cyclists_speeds (s : CyclistsScenario) 
  (h1 : condition1 s) (h2 : condition2 s) : s.x = 16 ∧ s.y = 14 := by
  sorry

end cyclists_speeds_l1442_144256


namespace exists_number_with_digit_sum_decrease_l1442_144219

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem stating the existence of a number satisfying the conditions -/
theorem exists_number_with_digit_sum_decrease : 
  ∃ (n : ℕ), 
    (∃ (m : ℕ), (11 * n = 10 * m)) ∧ 
    (sum_of_digits (11 * n / 10) = (9 * sum_of_digits n) / 10) := by sorry

end exists_number_with_digit_sum_decrease_l1442_144219


namespace gcd_of_2_powers_l1442_144245

theorem gcd_of_2_powers : Nat.gcd (2^2018 - 1) (2^2029 - 1) = 2^11 - 1 := by
  sorry

end gcd_of_2_powers_l1442_144245


namespace intersection_point_x_coordinate_l1442_144226

theorem intersection_point_x_coordinate :
  ∀ (x y : ℝ),
  (y = 3 * x - 15) →
  (3 * x + y = 120) →
  (x = 22.5) := by
sorry

end intersection_point_x_coordinate_l1442_144226


namespace club_membership_l1442_144247

theorem club_membership (total_members attendance : ℕ) 
  (h1 : total_members = 30)
  (h2 : attendance = 20)
  (h3 : ∃ (men women : ℕ), men + women = total_members ∧ men + women / 3 = attendance) :
  ∃ (men : ℕ), men = 15 ∧ 
    ∃ (women : ℕ), men + women = total_members ∧ men + women / 3 = attendance :=
by sorry

end club_membership_l1442_144247


namespace smallest_period_of_sine_function_l1442_144212

/-- Given a function f(x) = √3 * sin(πx/k) whose adjacent maximum and minimum points
    lie on the circle x^2 + y^2 = k^2, prove that its smallest positive period is 4. -/
theorem smallest_period_of_sine_function (k : ℝ) (h : k > 0) :
  let f : ℝ → ℝ := λ x ↦ Real.sqrt 3 * Real.sin (π * x / k)
  (∃ x y : ℝ, x^2 + y^2 = k^2 ∧ 
    (f x = Real.sqrt 3 ∧ f ((x + k/2) % (2*k)) = -Real.sqrt 3)) →
  2 * k = 4 :=
by sorry

end smallest_period_of_sine_function_l1442_144212


namespace triangles_in_decagon_count_l1442_144225

/-- The number of triangles that can be formed from the vertices of a regular decagon -/
def trianglesInDecagon : ℕ := 120

/-- The number of vertices in a regular decagon -/
def decagonVertices : ℕ := 10

/-- The number of vertices needed to form a triangle -/
def triangleVertices : ℕ := 3

/-- Theorem: The number of triangles that can be formed by selecting 3 vertices
    from a 10-vertex polygon is equal to 120 -/
theorem triangles_in_decagon_count :
  Nat.choose decagonVertices triangleVertices = trianglesInDecagon := by
  sorry

end triangles_in_decagon_count_l1442_144225


namespace age_problem_l1442_144216

/-- Proves that 10 years less than the average age of Mr. Bernard and Luke is 26 years. -/
theorem age_problem (luke_age : ℕ) (bernard_age : ℕ) : luke_age = 20 →
  bernard_age + 8 = 3 * luke_age →
  ((bernard_age + luke_age) / 2) - 10 = 26 := by
  sorry

end age_problem_l1442_144216


namespace hcf_of_three_numbers_l1442_144278

theorem hcf_of_three_numbers (a b c : ℕ) (h_lcm : Nat.lcm (Nat.lcm a b) c = 2^4 * 3^2 * 17 * 7)
  (h_a : a = 136) (h_b : b = 144) (h_c : c = 168) : Nat.gcd (Nat.gcd a b) c = 8 := by
  sorry

end hcf_of_three_numbers_l1442_144278


namespace quadratic_function_properties_l1442_144271

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_function_properties 
  (a b c : ℝ) 
  (h1 : f a b c 1 = -3)  -- vertex at (1, -3)
  (h2 : f a b c 2 = -5/2)  -- passes through (2, -5/2)
  (h3 : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a b c x1 = f a b c x2 ∧ |x1 - x2| = 6)  -- intersects y = m at two points 6 units apart
  : 
  (∀ x, f a b c x = 1/2 * x^2 - x - 5/2) ∧  -- Part 1
  (∃ m : ℝ, m = 3/2 ∧ ∀ x, f a b c x = m → (∃ y, f a b c y = m ∧ |x - y| = 6)) ∧  -- Part 2
  (∀ x, -3 < x → x < 3 → -3 ≤ f a b c x ∧ f a b c x < 5)  -- Part 3
  := by sorry

end quadratic_function_properties_l1442_144271


namespace first_discount_percentage_l1442_144235

theorem first_discount_percentage
  (original_price : ℝ)
  (final_price : ℝ)
  (second_discount : ℝ)
  (h1 : original_price = 149.99999999999997)
  (h2 : final_price = 108)
  (h3 : second_discount = 0.2)
  : (original_price - (final_price / (1 - second_discount))) / original_price = 0.1 := by
  sorry

end first_discount_percentage_l1442_144235


namespace max_bottles_from_c_and_d_l1442_144220

/-- Represents a shop selling recyclable bottles -/
structure Shop where
  price : ℕ
  available : ℕ

/-- Calculates the total cost of purchasing a given number of bottles from a shop -/
def totalCost (shop : Shop) (bottles : ℕ) : ℕ :=
  shop.price * bottles

theorem max_bottles_from_c_and_d (budget : ℕ) (shopA shopB shopC shopD : Shop) 
  (bottlesA bottlesB : ℕ) :
  budget = 600 ∧
  shopA = { price := 1, available := 200 } ∧
  shopB = { price := 2, available := 150 } ∧
  shopC = { price := 3, available := 100 } ∧
  shopD = { price := 5, available := 50 } ∧
  bottlesA = 150 ∧
  bottlesB = 180 ∧
  bottlesA ≤ shopA.available ∧
  bottlesB ≤ shopB.available →
  ∃ (bottlesC bottlesD : ℕ),
    bottlesC + bottlesD = 30 ∧
    bottlesC ≤ shopC.available ∧
    bottlesD ≤ shopD.available ∧
    totalCost shopA bottlesA + totalCost shopB bottlesB + totalCost shopC bottlesC + totalCost shopD bottlesD = budget ∧
    ∀ (newBottlesC newBottlesD : ℕ),
      newBottlesC ≤ shopC.available →
      newBottlesD ≤ shopD.available →
      totalCost shopA bottlesA + totalCost shopB bottlesB + totalCost shopC newBottlesC + totalCost shopD newBottlesD ≤ budget →
      newBottlesC + newBottlesD ≤ bottlesC + bottlesD :=
by sorry

end max_bottles_from_c_and_d_l1442_144220


namespace domain_of_function_l1442_144287

/-- The domain of the function f(x) = √(x - 1) + ∛(8 - x) is [1, 8] -/
theorem domain_of_function (f : ℝ → ℝ) (h : f = fun x ↦ Real.sqrt (x - 1) + (8 - x) ^ (1/3)) :
  Set.Icc 1 8 = {x : ℝ | ∃ y, f x = y} := by
  sorry

end domain_of_function_l1442_144287


namespace angle_measure_proof_l1442_144277

theorem angle_measure_proof (C D : ℝ) : 
  C + D = 180 →  -- Angles are supplementary
  C = 9 * D →    -- C is 9 times D
  C = 162 :=     -- The measure of angle C is 162 degrees
by
  sorry

end angle_measure_proof_l1442_144277


namespace inequality_and_equality_condition_l1442_144292

theorem inequality_and_equality_condition 
  (a b x y : ℝ) 
  (ha : a > 0) (hb : b > 0) (hx : x > 0) (hy : y > 0) 
  (hab : a + b = 1) : 
  (1 / (a / x + b / y) ≤ a * x + b * y) ∧ 
  (1 / (a / x + b / y) = a * x + b * y ↔ a * y = b * x) :=
sorry

end inequality_and_equality_condition_l1442_144292


namespace min_coins_for_dollar_l1442_144258

/-- Represents the different types of coins available --/
inductive Coin
  | Penny
  | Nickel
  | Dime
  | Quarter
  | HalfDollar

/-- Returns the value of a coin in cents --/
def coinValue (c : Coin) : ℕ :=
  match c with
  | Coin.Penny => 1
  | Coin.Nickel => 5
  | Coin.Dime => 10
  | Coin.Quarter => 25
  | Coin.HalfDollar => 50

/-- Calculates the total value of a list of coins in cents --/
def totalValue (coins : List Coin) : ℕ :=
  coins.foldl (fun acc c => acc + coinValue c) 0

/-- Theorem: The minimum number of coins to make one dollar is 3 --/
theorem min_coins_for_dollar :
  ∃ (coins : List Coin), totalValue coins = 100 ∧
    (∀ (other_coins : List Coin), totalValue other_coins = 100 →
      coins.length ≤ other_coins.length) ∧
    coins.length = 3 :=
  sorry

#check min_coins_for_dollar

end min_coins_for_dollar_l1442_144258


namespace unique_solution_l1442_144279

/-- Represents the number of children in each family and the house number -/
structure FamilyData where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  N : ℕ

/-- The conditions given in the problem -/
def satisfiesConditions (fd : FamilyData) : Prop :=
  fd.a > fd.b ∧ fd.b > fd.c ∧ fd.c > fd.d ∧
  fd.a + fd.b + fd.c + fd.d < 18 ∧
  fd.a * fd.b * fd.c * fd.d = fd.N

/-- The theorem statement -/
theorem unique_solution :
  ∃! fd : FamilyData, satisfiesConditions fd ∧ fd.N = 120 ∧
    fd.a = 5 ∧ fd.b = 4 ∧ fd.c = 3 ∧ fd.d = 2 := by
  sorry

end unique_solution_l1442_144279


namespace divisibility_implies_multiple_of_three_l1442_144242

theorem divisibility_implies_multiple_of_three (n : ℕ) :
  n ≥ 2 →
  (∃ k : ℕ, 2^n + 1 = k * n) →
  ∃ m : ℕ, n = 3 * m :=
sorry

end divisibility_implies_multiple_of_three_l1442_144242


namespace probability_yellow_chalk_l1442_144276

/-- The number of yellow chalks in the box -/
def yellow_chalks : ℕ := 3

/-- The number of red chalks in the box -/
def red_chalks : ℕ := 2

/-- The total number of chalks in the box -/
def total_chalks : ℕ := yellow_chalks + red_chalks

/-- The probability of selecting a yellow chalk -/
def prob_yellow : ℚ := yellow_chalks / total_chalks

theorem probability_yellow_chalk :
  prob_yellow = 3 / 5 := by sorry

end probability_yellow_chalk_l1442_144276


namespace arithmetic_sequence_count_l1442_144217

theorem arithmetic_sequence_count : 
  ∀ (a₁ : ℕ) (aₙ : ℕ) (d : ℕ),
    a₁ = 2 → aₙ = 2010 → d = 4 →
    ∃ (n : ℕ), n = 503 ∧ aₙ = a₁ + (n - 1) * d :=
by
  sorry

end arithmetic_sequence_count_l1442_144217


namespace original_amount_l1442_144214

theorem original_amount (X : ℝ) : (0.1 * (0.5 * X) = 25) → X = 500 := by
  sorry

end original_amount_l1442_144214


namespace complex_equation_solution_l1442_144237

theorem complex_equation_solution (z : ℂ) (h : (1 + 2*I)*z = 4 + 3*I) : z = 2 - I := by
  sorry

end complex_equation_solution_l1442_144237


namespace multiply_and_simplify_l1442_144275

theorem multiply_and_simplify (y : ℝ) (h : y ≠ 0) :
  (18 * y^3) * (9 * y^2) * (1 / (6*y)^2) = (9/2) * y^3 := by
  sorry

end multiply_and_simplify_l1442_144275


namespace max_production_theorem_l1442_144222

/-- Represents a clothing factory -/
structure Factory where
  production_per_month : ℕ
  top_time_ratio : ℕ
  pant_time_ratio : ℕ

/-- Calculates the maximum number of sets two factories can produce in a month -/
def max_production (factory_a factory_b : Factory) : ℕ :=
  sorry

/-- Theorem stating the maximum production of two specific factories -/
theorem max_production_theorem :
  let factory_a : Factory := ⟨2700, 2, 1⟩
  let factory_b : Factory := ⟨3600, 3, 2⟩
  max_production factory_a factory_b = 6700 := by
  sorry

end max_production_theorem_l1442_144222


namespace negation_equivalence_l1442_144282

theorem negation_equivalence :
  (¬ ∀ (a b : ℝ), ab > 0 → a > 0) ↔ (∀ (a b : ℝ), ab ≤ 0 → a ≤ 0) := by
  sorry

end negation_equivalence_l1442_144282


namespace total_kernels_needed_l1442_144227

/-- Represents a popcorn preference with its kernel-to-popcorn ratio -/
structure PopcornPreference where
  cups_wanted : ℚ
  kernels : ℚ
  cups_produced : ℚ

/-- Calculates the amount of kernels needed for a given preference -/
def kernels_needed (pref : PopcornPreference) : ℚ :=
  pref.kernels * (pref.cups_wanted / pref.cups_produced)

/-- The list of popcorn preferences for the movie night -/
def movie_night_preferences : List PopcornPreference := [
  ⟨3, 3, 6⟩,  -- Joanie
  ⟨4, 2, 4⟩,  -- Mitchell
  ⟨6, 4, 8⟩,  -- Miles and Davis
  ⟨3, 1, 3⟩   -- Cliff
]

/-- Theorem stating that the total amount of kernels needed is 7.5 tablespoons -/
theorem total_kernels_needed :
  (movie_night_preferences.map kernels_needed).sum = 15/2 := by
  sorry


end total_kernels_needed_l1442_144227


namespace mean_proportional_problem_l1442_144218

theorem mean_proportional_problem (x : ℝ) : 
  (156 : ℝ)^2 = x * 104 → x = 234 := by
  sorry

end mean_proportional_problem_l1442_144218


namespace tan_70_cos_10_expression_l1442_144241

theorem tan_70_cos_10_expression : 
  Real.tan (70 * π / 180) * Real.cos (10 * π / 180) * (Real.sqrt 3 * Real.tan (20 * π / 180) - 1) = -1 := by
  sorry

end tan_70_cos_10_expression_l1442_144241


namespace range_of_m_l1442_144210

-- Define propositions p and q
def p (m : ℝ) : Prop := (m - 2) / (m - 3) ≤ 2 / 3

def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 - 4*x + m^2 > 0

-- Define the range of m
def m_range (m : ℝ) : Prop := m < -2 ∨ (0 ≤ m ∧ m ≤ 2) ∨ m ≥ 3

-- Theorem statement
theorem range_of_m :
  ∀ m : ℝ, (p m ∨ q m) ∧ ¬(p m ∧ q m) → m_range m :=
sorry

end range_of_m_l1442_144210


namespace pyramid_cross_section_distance_l1442_144288

/-- Given a right octagonal pyramid with two cross sections parallel to the base,
    this theorem proves the distance of the larger cross section from the apex. -/
theorem pyramid_cross_section_distance
  (area_small area_large : ℝ)
  (height_diff : ℝ)
  (h_area_small : area_small = 256 * Real.sqrt 2)
  (h_area_large : area_large = 576 * Real.sqrt 2)
  (h_height_diff : height_diff = 12) :
  ∃ (h : ℝ), h = 36 ∧ 
    (area_small / area_large = (2/3)^2) ∧
    (h - 2/3 * h = height_diff) := by
  sorry

end pyramid_cross_section_distance_l1442_144288


namespace not_square_product_l1442_144240

theorem not_square_product (a : ℕ) : 
  (∀ n : ℕ, ¬∃ m : ℕ, n * (n + a) = m ^ 2) ↔ a = 1 ∨ a = 2 ∨ a = 4 := by
  sorry

end not_square_product_l1442_144240


namespace three_male_students_probability_l1442_144231

theorem three_male_students_probability 
  (total_male : ℕ) 
  (total_female : ℕ) 
  (selection_size : ℕ) 
  (prob_at_least_one_female : ℚ) : 
  total_male = 4 → 
  total_female = 2 → 
  selection_size = 3 → 
  prob_at_least_one_female = 4/5 → 
  (1 : ℚ) - prob_at_least_one_female = 1/5 := by
sorry

end three_male_students_probability_l1442_144231


namespace quadratic_inequality_solution_set_l1442_144200

theorem quadratic_inequality_solution_set 
  (a b c : ℝ) 
  (h1 : a ≠ 0) 
  (h2 : Set.Ioo (-2 : ℝ) 3 = {x : ℝ | a * x^2 + b * x + c > 0}) : 
  Set.Ioo (-3 : ℝ) 2 = {x : ℝ | a * x^2 - b * x + c > 0} := by
  sorry

end quadratic_inequality_solution_set_l1442_144200


namespace count_valid_pairs_l1442_144291

-- Define Ω as a nonreal root of z^4 = 1
def Ω : ℂ := Complex.I

-- Define the condition for valid pairs
def isValidPair (a b : ℤ) : Prop := Complex.abs (a • Ω + b) = 2

-- Theorem statement
theorem count_valid_pairs : 
  (∃! (n : ℕ), ∃ (s : Finset (ℤ × ℤ)), s.card = n ∧ 
    (∀ (p : ℤ × ℤ), p ∈ s ↔ isValidPair p.1 p.2) ∧ n = 4) :=
sorry

end count_valid_pairs_l1442_144291


namespace pears_amount_correct_l1442_144206

/-- The amount of peaches received by the store in kilograms. -/
def peaches : ℕ := 250

/-- The amount of pears received by the store in kilograms. -/
def pears : ℕ := 100

/-- Theorem stating that the amount of pears is correct given the conditions. -/
theorem pears_amount_correct : peaches = 2 * pears + 50 := by sorry

end pears_amount_correct_l1442_144206


namespace tetrahedron_volume_l1442_144262

/-- The volume of a regular tetrahedron with given base side length and angle between lateral face and base. -/
theorem tetrahedron_volume 
  (base_side : ℝ) 
  (lateral_angle : ℝ) 
  (h : base_side = Real.sqrt 3) 
  (θ : lateral_angle = π / 3) : 
  (1 / 3 : ℝ) * base_side ^ 2 * (base_side / 2) / Real.tan lateral_angle = 1 / 2 := by
  sorry

#check tetrahedron_volume

end tetrahedron_volume_l1442_144262


namespace consecutive_points_distance_l1442_144281

/-- Given 5 consecutive points on a straight line, prove that ac = 11 -/
theorem consecutive_points_distance (a b c d e : ℝ) : 
  (c - b = 3 * (d - c)) →  -- bc = 3 * cd
  (e - d = 8) →            -- de = 8
  (b - a = 5) →            -- ab = 5
  (e - a = 21) →           -- ae = 21
  (c - a = 11) :=          -- ac = 11
by sorry

end consecutive_points_distance_l1442_144281
