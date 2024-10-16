import Mathlib

namespace NUMINAMATH_CALUDE_speeding_ticket_percentage_l2572_257210

theorem speeding_ticket_percentage
  (total_motorists : ℝ)
  (exceed_limit_percentage : ℝ)
  (no_ticket_percentage : ℝ)
  (h1 : exceed_limit_percentage = 0.5)
  (h2 : no_ticket_percentage = 0.2)
  (h3 : total_motorists > 0) :
  let speeding_motorists := total_motorists * exceed_limit_percentage
  let no_ticket_motorists := speeding_motorists * no_ticket_percentage
  let ticket_motorists := speeding_motorists - no_ticket_motorists
  ticket_motorists / total_motorists = 0.4 :=
sorry

end NUMINAMATH_CALUDE_speeding_ticket_percentage_l2572_257210


namespace NUMINAMATH_CALUDE_electronics_not_all_on_sale_l2572_257213

-- Define the universe of discourse
variable (E : Type) [Nonempty E]

-- Define the predicate for "on sale"
variable (on_sale : E → Prop)

-- Define the store
variable (store : Set E)

-- Assume the store is not empty
variable (h_store_nonempty : store.Nonempty)

-- The main theorem
theorem electronics_not_all_on_sale
  (h : ¬∀ (e : E), e ∈ store → on_sale e) :
  (∃ (e : E), e ∈ store ∧ ¬on_sale e) ∧
  (¬∀ (e : E), e ∈ store → on_sale e) :=
by sorry


end NUMINAMATH_CALUDE_electronics_not_all_on_sale_l2572_257213


namespace NUMINAMATH_CALUDE_first_number_is_55_l2572_257293

def problem (x : ℝ) : Prop :=
  let known_numbers : List ℝ := [48, 507, 2, 684, 42]
  let all_numbers : List ℝ := x :: known_numbers
  (List.sum all_numbers) / 6 = 223

theorem first_number_is_55 : 
  ∃ (x : ℝ), problem x ∧ x = 55 :=
sorry

end NUMINAMATH_CALUDE_first_number_is_55_l2572_257293


namespace NUMINAMATH_CALUDE_ramsey_three_three_three_l2572_257252

/-- A coloring of edges in a complete graph with three colors -/
def Coloring (n : ℕ) := Fin n → Fin n → Fin 3

/-- A monochromatic triangle in a coloring -/
def HasMonochromaticTriangle (n : ℕ) (c : Coloring n) : Prop :=
  ∃ (i j k : Fin n), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
    c i j = c j k ∧ c j k = c i k

/-- The Ramsey theorem R(3,3,3) ≤ 17 -/
theorem ramsey_three_three_three :
  ∀ (c : Coloring 17), HasMonochromaticTriangle 17 c :=
sorry

end NUMINAMATH_CALUDE_ramsey_three_three_three_l2572_257252


namespace NUMINAMATH_CALUDE_blue_balls_count_l2572_257221

/-- The number of boxes a person has -/
def num_boxes : ℕ := 2

/-- The number of blue balls in each box -/
def blue_balls_per_box : ℕ := 5

/-- The total number of blue balls a person has -/
def total_blue_balls : ℕ := num_boxes * blue_balls_per_box

theorem blue_balls_count : total_blue_balls = 10 := by
  sorry

end NUMINAMATH_CALUDE_blue_balls_count_l2572_257221


namespace NUMINAMATH_CALUDE_minjin_apples_l2572_257220

theorem minjin_apples : ∃ (initial : ℕ), 
  (initial % 8 = 0) ∧ 
  (6 * ((initial / 8) + 8 - 30) = 12) ∧ 
  (initial = 192) := by
  sorry

end NUMINAMATH_CALUDE_minjin_apples_l2572_257220


namespace NUMINAMATH_CALUDE_jerry_has_36_stickers_l2572_257269

-- Define the number of stickers for each person
def fred_stickers : ℕ := 18
def george_stickers : ℕ := fred_stickers - 6
def jerry_stickers : ℕ := 3 * george_stickers
def carla_stickers : ℕ := jerry_stickers + (jerry_stickers / 4)

-- Theorem to prove
theorem jerry_has_36_stickers : jerry_stickers = 36 := by
  sorry

end NUMINAMATH_CALUDE_jerry_has_36_stickers_l2572_257269


namespace NUMINAMATH_CALUDE_negative_x_squared_times_x_cubed_l2572_257299

theorem negative_x_squared_times_x_cubed (x : ℝ) : (-x^2) * x^3 = -x^5 := by
  sorry

end NUMINAMATH_CALUDE_negative_x_squared_times_x_cubed_l2572_257299


namespace NUMINAMATH_CALUDE_right_triangle_leg_length_l2572_257246

theorem right_triangle_leg_length 
  (north_distance : ℝ) 
  (hypotenuse : ℝ) 
  (h1 : north_distance = 10)
  (h2 : hypotenuse = 14.142135623730951) : 
  ∃ west_distance : ℝ, 
    west_distance ^ 2 + north_distance ^ 2 = hypotenuse ^ 2 ∧ 
    west_distance = 10 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_leg_length_l2572_257246


namespace NUMINAMATH_CALUDE_distance_to_x_axis_l2572_257236

theorem distance_to_x_axis (P : ℝ × ℝ) (h : P = (-3, -2)) : 
  |P.2| = 2 := by sorry

end NUMINAMATH_CALUDE_distance_to_x_axis_l2572_257236


namespace NUMINAMATH_CALUDE_square_root_sum_l2572_257284

theorem square_root_sum (y : ℝ) :
  Real.sqrt (64 - y^2) - Real.sqrt (36 - y^2) = 4 →
  Real.sqrt (64 - y^2) + Real.sqrt (36 - y^2) = 7 := by
sorry

end NUMINAMATH_CALUDE_square_root_sum_l2572_257284


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_negative_three_l2572_257288

theorem sum_of_fractions_equals_negative_three 
  (a b c : ℝ) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hc : c ≠ 0) 
  (h_sum : a + b + c = 3) :
  1 / (b^2 + c^2 - 3*a^2) + 1 / (a^2 + c^2 - 3*b^2) + 1 / (a^2 + b^2 - 3*c^2) = -3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_negative_three_l2572_257288


namespace NUMINAMATH_CALUDE_mcdonald_accounting_error_l2572_257224

theorem mcdonald_accounting_error (x : ℝ) : x = 3.57 ↔ 9 * x = 32.13 := by sorry

end NUMINAMATH_CALUDE_mcdonald_accounting_error_l2572_257224


namespace NUMINAMATH_CALUDE_radio_quiz_win_probability_l2572_257273

/-- Represents a quiz show with multiple-choice questions. -/
structure QuizShow where
  num_questions : ℕ
  num_options : ℕ
  min_correct : ℕ

/-- Calculates the probability of winning a quiz show by random guessing. -/
def win_probability (quiz : QuizShow) : ℚ :=
  sorry

/-- The specific quiz show described in the problem. -/
def radio_quiz : QuizShow :=
  { num_questions := 4
  , num_options := 4
  , min_correct := 2 }

/-- Theorem stating the probability of winning the radio quiz. -/
theorem radio_quiz_win_probability :
  win_probability radio_quiz = 121 / 256 :=
by sorry

end NUMINAMATH_CALUDE_radio_quiz_win_probability_l2572_257273


namespace NUMINAMATH_CALUDE_rectangle_area_l2572_257279

/-- Given a rectangle with perimeter 100 meters and length three times its width, 
    its area is 468.75 square meters. -/
theorem rectangle_area (w : ℝ) (l : ℝ) : 
  (2 * l + 2 * w = 100) → 
  (l = 3 * w) → 
  (l * w = 468.75) :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2572_257279


namespace NUMINAMATH_CALUDE_largest_power_dividing_powProduct_l2572_257285

/-- pow(n) is the largest power of the largest prime that divides n -/
def pow (n : ℕ) : ℕ :=
  sorry

/-- The product of pow(n) from 2 to 2023 -/
def powProduct : ℕ :=
  sorry

theorem largest_power_dividing_powProduct : 
  (∀ m : ℕ, 462^m ∣ powProduct → m ≤ 202) ∧ 462^202 ∣ powProduct :=
sorry

end NUMINAMATH_CALUDE_largest_power_dividing_powProduct_l2572_257285


namespace NUMINAMATH_CALUDE_right_triangle_relation_l2572_257256

theorem right_triangle_relation (a b c h : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hh : h > 0)
  (right_triangle : a^2 + b^2 = c^2) (height_relation : 2 * h * c = a * b) :
  1 / a^2 + 1 / b^2 = 1 / h^2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_relation_l2572_257256


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_derived_inequality_solutions_l2572_257274

def quadratic_inequality (x : ℝ) : Prop := x^2 - 3*x + 2 > 0

def solution_set (x : ℝ) : Prop := x < 1 ∨ x > 2

def derived_inequality (x m : ℝ) : Prop := x^2 - (m + 2)*x + 2*m < 0

theorem quadratic_inequality_solution :
  ∀ x, quadratic_inequality x ↔ solution_set x :=
sorry

theorem derived_inequality_solutions :
  (∀ x, ¬(derived_inequality x 2)) ∧
  (∀ m, m < 2 → ∀ x, derived_inequality x m ↔ m < x ∧ x < 2) ∧
  (∀ m, m > 2 → ∀ x, derived_inequality x m ↔ 2 < x ∧ x < m) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_derived_inequality_solutions_l2572_257274


namespace NUMINAMATH_CALUDE_passing_percentage_is_33_percent_l2572_257272

/-- The passing percentage for an exam -/
def passing_percentage (marks_obtained : ℕ) (marks_failed_by : ℕ) (max_marks : ℕ) : ℚ :=
  ((marks_obtained + marks_failed_by : ℚ) / max_marks) * 100

/-- Theorem: The passing percentage is 33% given the problem conditions -/
theorem passing_percentage_is_33_percent : 
  passing_percentage 59 40 300 = 33 := by sorry

end NUMINAMATH_CALUDE_passing_percentage_is_33_percent_l2572_257272


namespace NUMINAMATH_CALUDE_expression_evaluation_l2572_257201

theorem expression_evaluation :
  ∃ k : ℝ, k > 0 ∧ (3^512 + 7^513)^2 - (3^512 - 7^513)^2 = k * 10^513 ∧ k = 28 * 2.1^512 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2572_257201


namespace NUMINAMATH_CALUDE_tree_height_difference_l2572_257242

theorem tree_height_difference : 
  let pine_height : ℚ := 49/4
  let birch_height : ℚ := 37/2
  birch_height - pine_height = 25/4 := by sorry

#check tree_height_difference

end NUMINAMATH_CALUDE_tree_height_difference_l2572_257242


namespace NUMINAMATH_CALUDE_appetizer_cost_per_person_l2572_257281

/-- Calculates the cost per person for a New Year's Eve appetizer --/
theorem appetizer_cost_per_person 
  (num_guests : ℕ) 
  (num_chip_bags : ℕ) 
  (chip_cost : ℚ) 
  (creme_fraiche_cost : ℚ) 
  (salmon_cost : ℚ) 
  (caviar_cost : ℚ) 
  (h1 : num_guests = 12) 
  (h2 : num_chip_bags = 10) 
  (h3 : chip_cost = 1) 
  (h4 : creme_fraiche_cost = 5) 
  (h5 : salmon_cost = 15) 
  (h6 : caviar_cost = 250) :
  (num_chip_bags * chip_cost + creme_fraiche_cost + salmon_cost + caviar_cost) / num_guests = 280 / 12 :=
by sorry

end NUMINAMATH_CALUDE_appetizer_cost_per_person_l2572_257281


namespace NUMINAMATH_CALUDE_petrol_price_equation_l2572_257241

/-- The original price of petrol in dollars per gallon -/
def P : ℝ := sorry

/-- The equation representing the price reduction scenario -/
theorem petrol_price_equation : (300 / P + 7) * (0.85 * P) = 300 := by sorry

end NUMINAMATH_CALUDE_petrol_price_equation_l2572_257241


namespace NUMINAMATH_CALUDE_dog_weight_ratio_l2572_257219

/-- Given the weights of two dogs, prove the ratio of their weights -/
theorem dog_weight_ratio 
  (evan_dog_weight : ℕ) 
  (total_weight : ℕ) 
  (h1 : evan_dog_weight = 63)
  (h2 : total_weight = 72)
  (h3 : ∃ k : ℕ, k * (total_weight - evan_dog_weight) = evan_dog_weight) :
  evan_dog_weight / (total_weight - evan_dog_weight) = 7 := by
sorry

end NUMINAMATH_CALUDE_dog_weight_ratio_l2572_257219


namespace NUMINAMATH_CALUDE_smallest_base_for_100_in_three_digits_l2572_257251

theorem smallest_base_for_100_in_three_digits : ∃ (b : ℕ), b = 5 ∧ 
  (∀ (n : ℕ), n^2 ≤ 100 ∧ 100 < n^3 → b ≤ n) := by
  sorry

end NUMINAMATH_CALUDE_smallest_base_for_100_in_three_digits_l2572_257251


namespace NUMINAMATH_CALUDE_only_event1_is_random_l2572_257200

-- Define the events
def event1 := "Tossing a coin twice in a row, and both times it lands heads up"
def event2 := "Opposite charges attract each other"
def event3 := "Water freezes at 1℃ under standard atmospheric pressure"

-- Define a predicate for random events
def is_random_event (e : String) : Prop := sorry

-- Theorem statement
theorem only_event1_is_random :
  is_random_event event1 ∧
  ¬is_random_event event2 ∧
  ¬is_random_event event3 := by sorry

end NUMINAMATH_CALUDE_only_event1_is_random_l2572_257200


namespace NUMINAMATH_CALUDE_rectangle_equation_l2572_257263

/-- A rectangle centered at the origin with width 2a and height 2b can be described by the equation
    √x * √(a - x) * √y * √(b - y) = 0, where 0 ≤ x ≤ a and 0 ≤ y ≤ b. -/
theorem rectangle_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  {p : ℝ × ℝ | ∃ (x y : ℝ), p = (x, y) ∧ 
    0 ≤ x ∧ x ≤ a ∧ 0 ≤ y ∧ y ≤ b ∧
    Real.sqrt x * Real.sqrt (a - x) * Real.sqrt y * Real.sqrt (b - y) = 0} =
  {p : ℝ × ℝ | -a ≤ p.1 ∧ p.1 ≤ a ∧ -b ≤ p.2 ∧ p.2 ≤ b} :=
by sorry

end NUMINAMATH_CALUDE_rectangle_equation_l2572_257263


namespace NUMINAMATH_CALUDE_tangent_line_at_zero_two_l2572_257278

noncomputable def f (x : ℝ) : ℝ := Real.exp x + 2 * x + 1

theorem tangent_line_at_zero_two :
  let f : ℝ → ℝ := λ x ↦ Real.exp x + 2 * x + 1
  let x₀ : ℝ := 0
  let y₀ : ℝ := f x₀
  let m : ℝ := (deriv f) x₀
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (3 * x - y + 2 = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_tangent_line_at_zero_two_l2572_257278


namespace NUMINAMATH_CALUDE_train_speed_calculation_l2572_257257

/-- Proves that given a round trip of 240 miles total, where the return trip speed is 38.71 miles per hour, 
and the total travel time is 5.5 hours, the speed of the first leg of the trip is 50 miles per hour. -/
theorem train_speed_calculation (total_distance : ℝ) (return_speed : ℝ) (total_time : ℝ) 
  (h1 : total_distance = 240)
  (h2 : return_speed = 38.71)
  (h3 : total_time = 5.5) :
  ∃ (outbound_speed : ℝ), outbound_speed = 50 ∧ 
  (total_distance / 2) / outbound_speed + (total_distance / 2) / return_speed = total_time :=
by
  sorry

#check train_speed_calculation

end NUMINAMATH_CALUDE_train_speed_calculation_l2572_257257


namespace NUMINAMATH_CALUDE_twelve_team_tournament_matches_l2572_257280

/-- Calculates the total number of matches in a round-robin tournament. -/
def total_matches (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a round-robin tournament with 12 teams, the total number of matches is 66. -/
theorem twelve_team_tournament_matches :
  total_matches 12 = 66 := by
  sorry

#eval total_matches 12  -- This will evaluate to 66

end NUMINAMATH_CALUDE_twelve_team_tournament_matches_l2572_257280


namespace NUMINAMATH_CALUDE_consecutive_odd_integers_problem_l2572_257232

theorem consecutive_odd_integers_problem (n : ℕ) : 
  n ≥ 3 ∧ n ≤ 9 ∧ n % 2 = 1 →
  (n - 2) + n + (n + 2) = ((n - 2) * n * (n + 2)) / 9 →
  n = 5 := by sorry

end NUMINAMATH_CALUDE_consecutive_odd_integers_problem_l2572_257232


namespace NUMINAMATH_CALUDE_fiona_peeled_22_l2572_257289

/-- Represents the potato peeling scenario -/
structure PotatoPeeling where
  total_potatoes : ℕ
  martin_rate : ℕ
  fiona_rate : ℕ
  fiona_join_time : ℕ

/-- Calculates the number of potatoes Fiona peeled -/
def fiona_peeled (scenario : PotatoPeeling) : ℕ :=
  let martin_peeled := scenario.martin_rate * scenario.fiona_join_time
  let remaining := scenario.total_potatoes - martin_peeled
  let combined_rate := scenario.martin_rate + scenario.fiona_rate
  let combined_time := (remaining + combined_rate - 1) / combined_rate -- Ceiling division
  scenario.fiona_rate * combined_time

/-- Theorem stating that Fiona peeled 22 potatoes -/
theorem fiona_peeled_22 (scenario : PotatoPeeling) 
  (h1 : scenario.total_potatoes = 60)
  (h2 : scenario.martin_rate = 4)
  (h3 : scenario.fiona_rate = 6)
  (h4 : scenario.fiona_join_time = 6) :
  fiona_peeled scenario = 22 := by
  sorry

#eval fiona_peeled { total_potatoes := 60, martin_rate := 4, fiona_rate := 6, fiona_join_time := 6 }

end NUMINAMATH_CALUDE_fiona_peeled_22_l2572_257289


namespace NUMINAMATH_CALUDE_ball_distribution_ways_l2572_257202

/-- Represents the number of ways to distribute balls of a given color --/
def distribute_balls (remaining : ℕ) : ℕ := remaining + 1

/-- Represents the total number of ways to distribute balls between two boys --/
def total_distributions (white : ℕ) (black : ℕ) (red : ℕ) : ℕ :=
  distribute_balls (white - 4) * distribute_balls (red - 4)

theorem ball_distribution_ways :
  let white := 6
  let black := 4
  let red := 8
  total_distributions white black red = 15 := by
  sorry

end NUMINAMATH_CALUDE_ball_distribution_ways_l2572_257202


namespace NUMINAMATH_CALUDE_solve_equation_l2572_257254

theorem solve_equation : ∃ y : ℝ, 3 * y - 6 = |-20 + 5| ∧ y = 7 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2572_257254


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l2572_257215

/-- 
Given an isosceles triangle with two sides of length 15 and a perimeter of 40,
prove that the length of the third side (base) is 10.
-/
theorem isosceles_triangle_base_length 
  (a b c : ℝ) 
  (h1 : a = 15) 
  (h2 : b = 15) 
  (h3 : a + b + c = 40) : 
  c = 10 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l2572_257215


namespace NUMINAMATH_CALUDE_dartboard_sector_angle_l2572_257205

theorem dartboard_sector_angle (total_angle : ℝ) (sector_prob : ℝ) : 
  total_angle = 360 → 
  sector_prob = 1/4 → 
  sector_prob * total_angle = 90 :=
by sorry

end NUMINAMATH_CALUDE_dartboard_sector_angle_l2572_257205


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_segment_ratio_l2572_257218

theorem right_triangle_hypotenuse_segment_ratio :
  ∀ (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0),
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  a / b = 2 / 3 →    -- Ratio of legs
  ∃ (x y : ℝ),
    x > 0 ∧ y > 0 ∧
    x + y = c ∧      -- Segments form the hypotenuse
    x / y = 4 / 9    -- Ratio of segments
  := by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_segment_ratio_l2572_257218


namespace NUMINAMATH_CALUDE_kennel_long_furred_dogs_l2572_257225

/-- The number of long-furred dogs in a kennel --/
def long_furred_dogs (total : ℕ) (brown : ℕ) (neither : ℕ) (long_furred_brown : ℕ) : ℕ :=
  total - neither - brown + long_furred_brown

/-- Theorem stating the number of long-furred dogs in the kennel --/
theorem kennel_long_furred_dogs :
  long_furred_dogs 45 17 8 9 = 29 := by
  sorry

#eval long_furred_dogs 45 17 8 9

end NUMINAMATH_CALUDE_kennel_long_furred_dogs_l2572_257225


namespace NUMINAMATH_CALUDE_venue_cost_venue_cost_is_10000_l2572_257239

/-- Calculates the venue cost for John's wedding --/
theorem venue_cost (cost_per_guest : ℕ) (john_guests : ℕ) (wife_extra_percent : ℕ) (total_cost : ℕ) : ℕ :=
  let wife_guests := john_guests + (wife_extra_percent * john_guests) / 100
  let guest_cost := cost_per_guest * wife_guests
  total_cost - guest_cost

/-- Proves that the venue cost is $10,000 given the specified conditions --/
theorem venue_cost_is_10000 :
  venue_cost 500 50 60 50000 = 10000 := by
  sorry

end NUMINAMATH_CALUDE_venue_cost_venue_cost_is_10000_l2572_257239


namespace NUMINAMATH_CALUDE_y_relationship_l2572_257206

-- Define the quadratic function
def f (x : ℝ) : ℝ := -(x + 2)^2 + 4

-- Define the points A, B, C
def A : ℝ × ℝ := (-3, f (-3))
def B : ℝ × ℝ := (0, f 0)
def C : ℝ × ℝ := (3, f 3)

-- Define y₁, y₂, y₃
def y₁ : ℝ := A.2
def y₂ : ℝ := B.2
def y₃ : ℝ := C.2

-- Theorem statement
theorem y_relationship : y₃ < y₂ ∧ y₂ < y₁ := by
  sorry

end NUMINAMATH_CALUDE_y_relationship_l2572_257206


namespace NUMINAMATH_CALUDE_log_157489_between_consecutive_integers_l2572_257258

theorem log_157489_between_consecutive_integers :
  ∃ c d : ℤ, c + 1 = d ∧ (c : ℝ) < Real.log 157489 / Real.log 10 ∧ Real.log 157489 / Real.log 10 < (d : ℝ) ∧ c + d = 11 := by
  sorry

end NUMINAMATH_CALUDE_log_157489_between_consecutive_integers_l2572_257258


namespace NUMINAMATH_CALUDE_special_polynomial_at_seven_l2572_257286

/-- A monic polynomial of degree 7 satisfying specific conditions -/
def special_polynomial (p : ℝ → ℝ) : Prop :=
  (∀ x, ∃ a₀ a₁ a₂ a₃ a₄ a₅ a₆, p x = x^7 + a₆*x^6 + a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) ∧
  p 0 = 0 ∧ p 1 = 1 ∧ p 2 = 2 ∧ p 3 = 3 ∧ p 4 = 4 ∧ p 5 = 5 ∧ p 6 = 6

/-- The theorem stating that any polynomial satisfying the special conditions will have p(7) = 5047 -/
theorem special_polynomial_at_seven (p : ℝ → ℝ) (h : special_polynomial p) : p 7 = 5047 := by
  sorry

end NUMINAMATH_CALUDE_special_polynomial_at_seven_l2572_257286


namespace NUMINAMATH_CALUDE_largest_ball_on_specific_torus_l2572_257233

/-- The radius of the largest spherical ball that can be placed atop a torus -/
def largest_ball_radius (torus_center : ℝ × ℝ × ℝ) (torus_radius : ℝ) : ℝ :=
  let (x, y, z) := torus_center
  2

/-- Theorem: The radius of the largest spherical ball on a specific torus is 2 -/
theorem largest_ball_on_specific_torus :
  largest_ball_radius (4, 0, 2) 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_largest_ball_on_specific_torus_l2572_257233


namespace NUMINAMATH_CALUDE_equation_solution_l2572_257222

theorem equation_solution : ∃ (x₁ x₂ : ℝ), 
  (x₁ * (x₁ + 2) = 3 * x₁ + 6) ∧ 
  (x₂ * (x₂ + 2) = 3 * x₂ + 6) ∧ 
  x₁ = -2 ∧ x₂ = 3 ∧ 
  (∀ x : ℝ, x * (x + 2) = 3 * x + 6 → x = x₁ ∨ x = x₂) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2572_257222


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l2572_257276

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def M : Set Nat := {1, 4}
def N : Set Nat := {1, 3, 5}

theorem intersection_complement_equality : N ∩ (U \ M) = {3, 5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l2572_257276


namespace NUMINAMATH_CALUDE_servings_count_l2572_257282

/-- Represents the number of cups of cereal in a box -/
def total_cups : ℕ := 18

/-- Represents the number of cups per serving -/
def cups_per_serving : ℕ := 2

/-- Calculates the number of servings in a cereal box -/
def servings_in_box : ℕ := total_cups / cups_per_serving

/-- Proves that the number of servings in the cereal box is 9 -/
theorem servings_count : servings_in_box = 9 := by
  sorry

end NUMINAMATH_CALUDE_servings_count_l2572_257282


namespace NUMINAMATH_CALUDE_boys_at_reunion_l2572_257267

/-- The number of handshakes between n boys, where each boy shakes hands
    exactly once with each of the others. -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: There were 11 boys at the reunion given that the total number
    of handshakes was 55 and each boy shook hands exactly once with each
    of the others. -/
theorem boys_at_reunion : ∃ (n : ℕ), n > 0 ∧ handshakes n = 55 ∧ n = 11 := by
  sorry

#eval handshakes 11  -- This should output 55

end NUMINAMATH_CALUDE_boys_at_reunion_l2572_257267


namespace NUMINAMATH_CALUDE_six_customOp_three_l2572_257275

/-- Definition of the custom operation " -/
def customOp (m n : ℕ) : ℕ := n ^ 2 - m

/-- Theorem stating that 6 " 3 = 3 -/
theorem six_customOp_three : customOp 6 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_six_customOp_three_l2572_257275


namespace NUMINAMATH_CALUDE_parabola_properties_l2572_257295

-- Define the parabola
def parabola (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * a * x + 3

-- Theorem statement
theorem parabola_properties (a : ℝ) (h : a ≠ 0) :
  -- 1. Axis of symmetry
  (∀ x : ℝ, parabola a x = parabola a (2 - x)) ∧
  -- 2. Vertex on x-axis after shifting
  ((∃ x : ℝ, parabola a x - 3 * |a| = 0 ∧
             ∀ y : ℝ, parabola a y - 3 * |a| ≥ 0) ↔ (a = 3/4 ∨ a = -3/2)) ∧
  -- 3. Range of a for given points
  (∀ y₁ y₂ : ℝ, y₁ > y₂ → parabola a a = y₁ → parabola a 2 = y₂ → a > 2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l2572_257295


namespace NUMINAMATH_CALUDE_square_division_rectangle_perimeter_l2572_257290

/-- Given a square with perimeter 120 units divided into four congruent rectangles,
    the perimeter of one of these rectangles is 90 units. -/
theorem square_division_rectangle_perimeter :
  ∀ (s : ℝ),
  s > 0 →
  4 * s = 120 →
  2 * (s + s / 2) = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_square_division_rectangle_perimeter_l2572_257290


namespace NUMINAMATH_CALUDE_valid_fraction_pairs_l2572_257260

def is_valid_pair (x y : ℚ) : Prop :=
  ∃ (A B : ℕ+) (r : ℚ),
    x = (A : ℚ) * (1/10 + 1/70) ∧
    y = (B : ℚ) * (1/10 + 1/70) ∧
    x + y = 8 ∧
    r > 1 ∧
    ∃ (C D : ℕ), C > 1 ∧ D > 1 ∧ x = C * r ∧ y = D * r

theorem valid_fraction_pairs :
  (is_valid_pair (16/7) (40/7) ∧
   is_valid_pair (24/7) (32/7) ∧
   is_valid_pair (16/5) (24/5) ∧
   is_valid_pair 4 4) ∧
  ∀ x y, is_valid_pair x y →
    ((x = 16/7 ∧ y = 40/7) ∨
     (x = 24/7 ∧ y = 32/7) ∨
     (x = 16/5 ∧ y = 24/5) ∨
     (x = 4 ∧ y = 4) ∨
     (y = 16/7 ∧ x = 40/7) ∨
     (y = 24/7 ∧ x = 32/7) ∨
     (y = 16/5 ∧ x = 24/5)) :=
by sorry


end NUMINAMATH_CALUDE_valid_fraction_pairs_l2572_257260


namespace NUMINAMATH_CALUDE_smallest_N_eight_works_smallest_N_is_8_l2572_257230

theorem smallest_N : ∀ N : ℕ+, 
  (∃ a b c d : ℕ, 
    a = N.val * 125 / 1000 ∧
    b = N.val * 500 / 1000 ∧
    c = N.val * 250 / 1000 ∧
    d = N.val * 125 / 1000) →
  N.val ≥ 8 :=
by sorry

theorem eight_works : 
  ∃ a b c d : ℕ,
    a = 8 * 125 / 1000 ∧
    b = 8 * 500 / 1000 ∧
    c = 8 * 250 / 1000 ∧
    d = 8 * 125 / 1000 :=
by sorry

theorem smallest_N_is_8 : 
  ∀ N : ℕ+, 
    (∃ a b c d : ℕ, 
      a = N.val * 125 / 1000 ∧
      b = N.val * 500 / 1000 ∧
      c = N.val * 250 / 1000 ∧
      d = N.val * 125 / 1000) ↔
    N.val ≥ 8 :=
by sorry

end NUMINAMATH_CALUDE_smallest_N_eight_works_smallest_N_is_8_l2572_257230


namespace NUMINAMATH_CALUDE_expression_factorization_l2572_257291

theorem expression_factorization (a b c : ℝ) :
  a * (b - c)^2 + b * (c - a)^2 + c * (a - b)^2 = (a - b) * (b - c) * (c - a) * (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l2572_257291


namespace NUMINAMATH_CALUDE_hidden_numbers_average_l2572_257297

/-- Given three cards with visible numbers and hidden consecutive odd numbers,
    if the sum of numbers on each card is equal, then the average of hidden numbers is 18. -/
theorem hidden_numbers_average (v₁ v₂ v₃ h₁ h₂ h₃ : ℕ) : 
  v₁ = 30 ∧ v₂ = 42 ∧ v₃ = 36 →  -- visible numbers
  h₂ = h₁ + 2 ∧ h₃ = h₂ + 2 →    -- hidden numbers are consecutive odd
  v₁ + h₁ = v₂ + h₂ ∧ v₂ + h₂ = v₃ + h₃ →  -- sum on each card is equal
  (h₁ + h₂ + h₃) / 3 = 18 :=
by sorry

end NUMINAMATH_CALUDE_hidden_numbers_average_l2572_257297


namespace NUMINAMATH_CALUDE_smallest_two_digit_prime_with_reversed_composite_l2572_257296

-- Define a function to check if a number is prime
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

-- Define a function to reverse the digits of a two-digit number
def reverseDigits (n : ℕ) : ℕ := 
  let tens := n / 10
  let ones := n % 10
  ones * 10 + tens

-- Define a function to check if a number is a two-digit number with 2 as the tens digit
def isTwoDigitWithTensTwo (n : ℕ) : Prop := n ≥ 20 ∧ n < 30

-- Main theorem
theorem smallest_two_digit_prime_with_reversed_composite :
  ∃ (n : ℕ), 
    isPrime n ∧ 
    isTwoDigitWithTensTwo n ∧ 
    ¬(isPrime (reverseDigits n)) ∧
    (∀ m : ℕ, m < n → ¬(isPrime m ∧ isTwoDigitWithTensTwo m ∧ ¬(isPrime (reverseDigits m)))) ∧
    n = 23 := by
  sorry

end NUMINAMATH_CALUDE_smallest_two_digit_prime_with_reversed_composite_l2572_257296


namespace NUMINAMATH_CALUDE_john_remaining_cards_l2572_257292

def cards_per_deck : ℕ := 52
def half_full_decks : ℕ := 3
def full_decks : ℕ := 3
def discarded_cards : ℕ := 34

theorem john_remaining_cards : 
  cards_per_deck * full_decks + (cards_per_deck / 2) * half_full_decks - discarded_cards = 200 := by
  sorry

end NUMINAMATH_CALUDE_john_remaining_cards_l2572_257292


namespace NUMINAMATH_CALUDE_selection_for_38_classes_6_routes_l2572_257262

/-- The number of ways for a given number of classes to each choose one of a given number of routes. -/
def number_of_selections (num_classes : ℕ) (num_routes : ℕ) : ℕ := num_routes ^ num_classes

/-- Theorem stating that the number of ways for 38 classes to each choose one of 6 routes is 6^38. -/
theorem selection_for_38_classes_6_routes : number_of_selections 38 6 = 6^38 := by
  sorry

#eval number_of_selections 38 6

end NUMINAMATH_CALUDE_selection_for_38_classes_6_routes_l2572_257262


namespace NUMINAMATH_CALUDE_decreasing_multiplicative_to_additive_properties_l2572_257211

/-- A function f: ℝ → ℝ that is decreasing and satisfies f(xy) = f(x) + f(y) for all x, y ∈ ℝ -/
def DecreasingMultiplicativeToAdditive (f : ℝ → ℝ) : Prop :=
  (∀ x y, x < y → f x > f y) ∧ (∀ x y, f (x * y) = f x + f y)

theorem decreasing_multiplicative_to_additive_properties
  (f : ℝ → ℝ) (h : DecreasingMultiplicativeToAdditive f) :
  (f 1 = 0) ∧ (∀ x, f (2 * x - 3) < 0 ↔ x > 2) := by
  sorry

end NUMINAMATH_CALUDE_decreasing_multiplicative_to_additive_properties_l2572_257211


namespace NUMINAMATH_CALUDE_combined_tower_height_l2572_257294

/-- The height of Grace's tower in inches -/
def grace_height : ℕ := 40

/-- The ratio of Grace's tower height to Clyde's tower height -/
def grace_to_clyde_ratio : ℕ := 8

/-- The ratio of Sarah's tower height to Clyde's tower height -/
def sarah_to_clyde_ratio : ℕ := 2

/-- Theorem stating the combined height of all three towers -/
theorem combined_tower_height : 
  grace_height + (grace_height / grace_to_clyde_ratio) * (1 + sarah_to_clyde_ratio) = 55 := by
  sorry

end NUMINAMATH_CALUDE_combined_tower_height_l2572_257294


namespace NUMINAMATH_CALUDE_liar_knight_difference_district_A_l2572_257234

/-- Represents the number of residents in the city -/
def total_residents : ℕ := 50

/-- Represents the number of questions asked -/
def num_questions : ℕ := 4

/-- Represents the number of affirmative answers given by a knight -/
def knight_affirmative : ℕ := 1

/-- Represents the number of affirmative answers given by a liar -/
def liar_affirmative : ℕ := 3

/-- Represents the total number of affirmative answers given -/
def total_affirmative : ℕ := 290

/-- Theorem stating the difference between liars and knights in District A -/
theorem liar_knight_difference_district_A :
  ∃ (knights_A liars_A : ℕ),
    knights_A + liars_A ≤ total_residents ∧
    knights_A * knight_affirmative * num_questions +
    liars_A * liar_affirmative * num_questions ≤ total_affirmative ∧
    liars_A = knights_A + 3 := by
  sorry

end NUMINAMATH_CALUDE_liar_knight_difference_district_A_l2572_257234


namespace NUMINAMATH_CALUDE_divisors_of_60_l2572_257237

/-- The number of positive divisors of 60 is 12. -/
theorem divisors_of_60 : Nat.card (Nat.divisors 60) = 12 := by sorry

end NUMINAMATH_CALUDE_divisors_of_60_l2572_257237


namespace NUMINAMATH_CALUDE_weight_difference_is_19_l2572_257223

/-- The combined weight difference between the lightest and heaviest individual -/
def weightDifference (john roy derek samantha : ℕ) : ℕ :=
  max john (max roy (max derek samantha)) - min john (min roy (min derek samantha))

/-- Theorem: The combined weight difference between the lightest and heaviest individual is 19 pounds -/
theorem weight_difference_is_19 :
  weightDifference 81 79 91 72 = 19 := by
  sorry

end NUMINAMATH_CALUDE_weight_difference_is_19_l2572_257223


namespace NUMINAMATH_CALUDE_prob_red_or_blue_l2572_257228

-- Define the total number of marbles
def total_marbles : ℕ := 120

-- Define the probabilities of each color
def prob_white : ℚ := 1/5
def prob_green : ℚ := 1/10
def prob_orange : ℚ := 1/6
def prob_violet : ℚ := 1/8

-- Theorem statement
theorem prob_red_or_blue :
  let prob_others := prob_white + prob_green + prob_orange + prob_violet
  (1 - prob_others) = 49/120 := by sorry

end NUMINAMATH_CALUDE_prob_red_or_blue_l2572_257228


namespace NUMINAMATH_CALUDE_game_cost_l2572_257216

/-- 
Given:
- Frank's initial money: 11 dollars
- Frank's allowance: 14 dollars
- Frank's final money: 22 dollars

Prove that the cost of the new game is 3 dollars.
-/
theorem game_cost (initial_money : ℕ) (allowance : ℕ) (final_money : ℕ)
  (h1 : initial_money = 11)
  (h2 : allowance = 14)
  (h3 : final_money = 22) :
  initial_money - (final_money - allowance) = 3 :=
by sorry

end NUMINAMATH_CALUDE_game_cost_l2572_257216


namespace NUMINAMATH_CALUDE_polynomial_identity_l2572_257298

theorem polynomial_identity (x : ℝ) : 
  let P (x : ℝ) := (x - 1/2)^2001 + 1/2
  P x + P (1 - x) = 1 := by sorry

end NUMINAMATH_CALUDE_polynomial_identity_l2572_257298


namespace NUMINAMATH_CALUDE_quadratic_sum_has_root_l2572_257264

/-- A quadratic polynomial with a positive leading coefficient -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ
  h : a > 0

/-- The value of a quadratic polynomial at a given point -/
def QuadraticPolynomial.eval (p : QuadraticPolynomial) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- Two polynomials have a common root -/
def has_common_root (p q : QuadraticPolynomial) : Prop :=
  ∃ x, p.eval x = 0 ∧ q.eval x = 0

theorem quadratic_sum_has_root (p₁ p₂ p₃ : QuadraticPolynomial)
  (h₁₂ : has_common_root p₁ p₂)
  (h₂₃ : has_common_root p₂ p₃)
  (h₃₁ : has_common_root p₃ p₁) :
  ∃ x, (p₁.eval x + p₂.eval x + p₃.eval x = 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_sum_has_root_l2572_257264


namespace NUMINAMATH_CALUDE_units_digit_of_2_to_2010_l2572_257287

-- Define a function to get the units digit of a natural number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Define the cycle of units digits for powers of 2
def powerOfTwoCycle : List ℕ := [2, 4, 8, 6]

-- Theorem statement
theorem units_digit_of_2_to_2010 :
  unitsDigit (2^2010) = 4 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_2_to_2010_l2572_257287


namespace NUMINAMATH_CALUDE_stratified_sampling_company_a_l2572_257243

def total_representatives : ℕ := 100
def company_a_representatives : ℕ := 40
def company_b_representatives : ℕ := 60
def total_sample_size : ℕ := 10

theorem stratified_sampling_company_a :
  (company_a_representatives * total_sample_size) / total_representatives = 4 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_company_a_l2572_257243


namespace NUMINAMATH_CALUDE_union_equality_iff_a_in_range_l2572_257249

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ -1 ∨ x ≥ 4}
def B (a : ℝ) : Set ℝ := {x | 2 * a ≤ x ∧ x ≤ a + 3}

-- State the theorem
theorem union_equality_iff_a_in_range (a : ℝ) : 
  A ∪ B a = A ↔ a ∈ Set.Iic (-4) ∪ Set.Ici 2 :=
sorry

end NUMINAMATH_CALUDE_union_equality_iff_a_in_range_l2572_257249


namespace NUMINAMATH_CALUDE_trajectory_equation_of_midpoints_l2572_257235

/-- Given three real numbers forming an arithmetic sequence and equations of a line and parabola,
    prove the trajectory equation of the midpoints of the intercepted chords. -/
theorem trajectory_equation_of_midpoints
  (a b c : ℝ)
  (h_arithmetic : c = 2*b - a) -- arithmetic sequence condition
  (h_line : ∀ x y, b*x + a*y + c = 0 → (x : ℝ) = x ∧ (y : ℝ) = y) -- line equation
  (h_parabola : ∀ x y, y^2 = -1/2*x → (x : ℝ) = x ∧ (y : ℝ) = y) -- parabola equation
  : ∃ (x y : ℝ), x + 1 = -(2*y - 1)^2 ∧ y ≠ 1 :=
sorry

end NUMINAMATH_CALUDE_trajectory_equation_of_midpoints_l2572_257235


namespace NUMINAMATH_CALUDE_scientific_notation_10374_billion_l2572_257250

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation := sorry

/-- Rounds a ScientificNotation to a given number of significant figures -/
def roundToSigFigs (sn : ScientificNotation) (sigFigs : ℕ) : ScientificNotation := sorry

theorem scientific_notation_10374_billion :
  let original_value : ℝ := 10374 * 1000000000
  let scientific_form := toScientificNotation original_value
  let rounded_form := roundToSigFigs scientific_form 3
  rounded_form.coefficient = 1.04 ∧ rounded_form.exponent = 13 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_10374_billion_l2572_257250


namespace NUMINAMATH_CALUDE_stratified_sample_elderly_count_l2572_257217

/-- Represents the number of teachers in a sample -/
structure TeacherSample where
  young : ℕ
  elderly : ℕ

/-- Represents the ratio of young to elderly teachers -/
structure TeacherRatio where
  young : ℕ
  elderly : ℕ

/-- 
Given a stratified sample of teachers where:
- The ratio of young to elderly teachers is 16:9
- There are 320 young teachers in the sample
Prove that there are 180 elderly teachers in the sample
-/
theorem stratified_sample_elderly_count 
  (ratio : TeacherRatio) 
  (sample : TeacherSample) :
  ratio.young = 16 →
  ratio.elderly = 9 →
  sample.young = 320 →
  (ratio.young : ℚ) / ratio.elderly = sample.young / sample.elderly →
  sample.elderly = 180 := by
sorry

end NUMINAMATH_CALUDE_stratified_sample_elderly_count_l2572_257217


namespace NUMINAMATH_CALUDE_stating_largest_cone_in_cube_l2572_257229

/-- Represents the dimensions of a cone carved from a cube. -/
structure ConeDimensions where
  height : ℝ
  baseRadius : ℝ
  volume : ℝ

/-- 
Theorem stating the dimensions of the largest cone that can be carved from a cube.
The cone's axis coincides with one of the cube's body diagonals.
-/
theorem largest_cone_in_cube (a : ℝ) (ha : a > 0) : 
  ∃ (cone : ConeDimensions), 
    cone.height = a * Real.sqrt 3 / 2 ∧
    cone.baseRadius = a * Real.sqrt 3 / (2 * Real.sqrt 2) ∧
    cone.volume = π * a^3 * Real.sqrt 3 / 16 ∧
    ∀ (other : ConeDimensions), other.volume ≤ cone.volume := by
  sorry

end NUMINAMATH_CALUDE_stating_largest_cone_in_cube_l2572_257229


namespace NUMINAMATH_CALUDE_perimeter_difference_is_zero_l2572_257203

/-- A figure composed of unit squares -/
structure UnitSquareFigure where
  squares : ℕ
  perimeter : ℕ

/-- T-shaped figure with 5 unit squares -/
def t_shape : UnitSquareFigure :=
  { squares := 5,
    perimeter := 8 }

/-- Cross-shaped figure with 5 unit squares -/
def cross_shape : UnitSquareFigure :=
  { squares := 5,
    perimeter := 8 }

/-- The positive difference between the perimeters of the T-shape and cross-shape is 0 -/
theorem perimeter_difference_is_zero :
  (t_shape.perimeter : ℤ) - (cross_shape.perimeter : ℤ) = 0 := by
  sorry

#check perimeter_difference_is_zero

end NUMINAMATH_CALUDE_perimeter_difference_is_zero_l2572_257203


namespace NUMINAMATH_CALUDE_floor_fraction_theorem_l2572_257283

theorem floor_fraction_theorem (d : ℝ) : 
  (∃ x : ℤ, x = ⌊d⌋ ∧ 3 * (x : ℝ)^2 + 10 * (x : ℝ) - 40 = 0) ∧ 
  (∃ y : ℝ, y = d - ⌊d⌋ ∧ 4 * y^2 - 20 * y + 19 = 0) →
  d = -9/2 := by
sorry

end NUMINAMATH_CALUDE_floor_fraction_theorem_l2572_257283


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l2572_257271

/-- Given a line L1 with equation 3x - 6y = 9 and a point P (2, -3),
    prove that the line L2 with equation y = -2x + 1 is perpendicular to L1 and passes through P. -/
theorem perpendicular_line_through_point (x y : ℝ) :
  let L1 : ℝ → ℝ → Prop := λ x y => 3 * x - 6 * y = 9
  let L2 : ℝ → ℝ → Prop := λ x y => y = -2 * x + 1
  let P : ℝ × ℝ := (2, -3)
  (∀ x y, L1 x y ↔ y = (1/2) * x - 3/2) →  -- L1 in slope-intercept form
  (L2 P.1 P.2) →  -- L2 passes through P
  (∀ m1 m2, m1 = 1/2 ∧ m2 = -2 → m1 * m2 = -1) →  -- Perpendicular slopes multiply to -1
  ∃ m b, ∀ x y, L2 x y ↔ y = m * x + b ∧ m * (1/2) = -1  -- L2 is perpendicular to L1
  := by sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l2572_257271


namespace NUMINAMATH_CALUDE_binary_to_base4_conversion_l2572_257231

theorem binary_to_base4_conversion : 
  (fun n : ℕ => n.digits 2) 11010011 = (fun n : ℕ => n.digits 4) 3103 := by
  sorry

end NUMINAMATH_CALUDE_binary_to_base4_conversion_l2572_257231


namespace NUMINAMATH_CALUDE_cost_of_one_each_l2572_257209

/-- The cost of goods A, B, and C -/
structure GoodsCost where
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given conditions from the problem -/
def problem_conditions (cost : GoodsCost) : Prop :=
  3 * cost.A + 7 * cost.B + cost.C = 3.15 ∧
  4 * cost.A + 10 * cost.B + cost.C = 4.20

/-- The theorem to prove -/
theorem cost_of_one_each (cost : GoodsCost) :
  problem_conditions cost → cost.A + cost.B + cost.C = 1.05 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_one_each_l2572_257209


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2572_257214

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_sum : a 4 + a 7 = 2)
  (h_product : a 5 * a 6 = -3) :
  a 1 * a 10 = -323 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2572_257214


namespace NUMINAMATH_CALUDE_greatest_number_l2572_257253

theorem greatest_number (p q r s t : ℝ) 
  (h1 : r < s) 
  (h2 : t > q) 
  (h3 : q > p) 
  (h4 : t < r) : 
  s = max p (max q (max r (max s t))) := by
sorry

end NUMINAMATH_CALUDE_greatest_number_l2572_257253


namespace NUMINAMATH_CALUDE_middle_dimension_at_least_six_l2572_257226

/-- Represents the dimensions of a rectangular crate -/
structure CrateDimensions where
  length : Real
  width : Real
  height : Real

/-- Represents a right circular cylinder -/
structure Cylinder where
  radius : Real
  height : Real

/-- Checks if a cylinder fits upright in a crate -/
def cylinderFitsUpright (crate : CrateDimensions) (cylinder : Cylinder) : Prop :=
  (cylinder.radius * 2 ≤ crate.length ∧ cylinder.radius * 2 ≤ crate.width ∧ cylinder.height ≤ crate.height) ∨
  (cylinder.radius * 2 ≤ crate.length ∧ cylinder.radius * 2 ≤ crate.height ∧ cylinder.height ≤ crate.width) ∨
  (cylinder.radius * 2 ≤ crate.width ∧ cylinder.radius * 2 ≤ crate.height ∧ cylinder.height ≤ crate.length)

theorem middle_dimension_at_least_six 
  (crate : CrateDimensions)
  (h1 : crate.length = 3)
  (h2 : crate.height = 12)
  (h3 : cylinderFitsUpright crate { radius := 3, height := 12 }) :
  crate.width ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_middle_dimension_at_least_six_l2572_257226


namespace NUMINAMATH_CALUDE_expression_evaluation_l2572_257247

theorem expression_evaluation (a b c : ℝ) 
  (h : a / (45 - a) + b / (85 - b) + c / (75 - c) = 9) :
  9 / (45 - a) + 17 / (85 - b) + 15 / (75 - c) = 2.4 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2572_257247


namespace NUMINAMATH_CALUDE_phi_equality_l2572_257245

-- Define the set M_φ
def M_phi (φ : ℕ → ℕ) : Set (ℕ → ℤ) :=
  {f | ∀ x, f x > f (φ x)}

-- State the theorem
theorem phi_equality (φ₁ φ₂ : ℕ → ℕ) :
  M_phi φ₁ = M_phi φ₂ → M_phi φ₁ ≠ ∅ → φ₁ = φ₂ := by
  sorry

end NUMINAMATH_CALUDE_phi_equality_l2572_257245


namespace NUMINAMATH_CALUDE_phoebe_peanut_butter_l2572_257207

/-- The number of jars of peanut butter needed for Phoebe and her dog for 30 days -/
def jars_needed (
  phoebe_servings : ℕ)  -- Phoebe's daily servings
  (dog_servings : ℕ)    -- Dog's daily servings
  (days : ℕ)            -- Number of days
  (servings_per_jar : ℕ) -- Servings per jar
  : ℕ :=
  ((phoebe_servings + dog_servings) * days + servings_per_jar - 1) / servings_per_jar

/-- Theorem stating the number of jars needed for Phoebe and her dog for 30 days -/
theorem phoebe_peanut_butter :
  jars_needed 1 1 30 15 = 4 := by
  sorry

end NUMINAMATH_CALUDE_phoebe_peanut_butter_l2572_257207


namespace NUMINAMATH_CALUDE_arithmetic_comparisons_l2572_257255

theorem arithmetic_comparisons : 
  (25 + 45 = 45 + 25) ∧ 
  (56 - 28 < 65 - 28) ∧ 
  (22 * 41 = 41 * 22) ∧ 
  (50 - 32 > 50 - 23) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_comparisons_l2572_257255


namespace NUMINAMATH_CALUDE_runners_meeting_time_l2572_257208

def anna_lap_time : ℕ := 5
def bob_lap_time : ℕ := 8
def carol_lap_time : ℕ := 10

def meeting_time : ℕ := 40

theorem runners_meeting_time :
  Nat.lcm (Nat.lcm anna_lap_time bob_lap_time) carol_lap_time = meeting_time :=
sorry

end NUMINAMATH_CALUDE_runners_meeting_time_l2572_257208


namespace NUMINAMATH_CALUDE_polyhedron_property_l2572_257240

/-- Represents a convex polyhedron with specific properties -/
structure ConvexPolyhedron where
  V : ℕ  -- Number of vertices
  E : ℕ  -- Number of edges
  F : ℕ  -- Number of faces
  t : ℕ  -- Number of triangular faces
  h : ℕ  -- Number of hexagonal faces
  T : ℕ  -- Number of triangular faces meeting at each vertex
  H : ℕ  -- Number of hexagonal faces meeting at each vertex
  euler_formula : V - E + F = 2
  face_count : F = 30
  face_types : t + h = F
  edge_count : E = (3 * t + 6 * h) / 2
  vertex_count : V = 3 * t / T
  triangle_hex_relation : T = 1 ∧ H = 2

/-- Theorem stating the specific property of the polyhedron -/
theorem polyhedron_property (p : ConvexPolyhedron) : 100 * p.H + 10 * p.T + p.V = 270 := by
  sorry

end NUMINAMATH_CALUDE_polyhedron_property_l2572_257240


namespace NUMINAMATH_CALUDE_red_ball_probability_l2572_257238

/-- Represents the number of balls of each color in the bag -/
structure BallCounts where
  red : ℕ
  yellow : ℕ
  white : ℕ

/-- Calculates the total number of balls in the bag -/
def totalBalls (counts : BallCounts) : ℕ :=
  counts.red + counts.yellow + counts.white

/-- Calculates the probability of drawing a ball of a specific color -/
def drawProbability (counts : BallCounts) (color : ℕ) : ℚ :=
  color / (totalBalls counts)

/-- Theorem: The probability of drawing a red ball from a bag with 3 red, 5 yellow, and 2 white balls is 3/10 -/
theorem red_ball_probability :
  let bag := BallCounts.mk 3 5 2
  drawProbability bag bag.red = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_red_ball_probability_l2572_257238


namespace NUMINAMATH_CALUDE_base_conversion_1729_l2572_257268

def base_10_to_base_6 (n : ℕ) : List ℕ :=
  sorry

theorem base_conversion_1729 :
  base_10_to_base_6 1729 = [1, 2, 0, 0, 1] :=
sorry

end NUMINAMATH_CALUDE_base_conversion_1729_l2572_257268


namespace NUMINAMATH_CALUDE_total_books_is_54_l2572_257248

def darla_books : ℕ := 6

def katie_books : ℕ := darla_books / 2

def darla_katie_books : ℕ := darla_books + katie_books

def gary_books : ℕ := 5 * darla_katie_books

def total_books : ℕ := darla_books + katie_books + gary_books

theorem total_books_is_54 : total_books = 54 := by
  sorry

end NUMINAMATH_CALUDE_total_books_is_54_l2572_257248


namespace NUMINAMATH_CALUDE_mary_apples_trees_ratio_l2572_257212

/-- Given that Mary bought 6 apples, ate 2 apples, and planted 4 trees,
    prove that the ratio of trees planted to apples eaten is 2. -/
theorem mary_apples_trees_ratio :
  let total_apples : ℕ := 6
  let eaten_apples : ℕ := 2
  let planted_trees : ℕ := 4
  (planted_trees : ℚ) / eaten_apples = 2 := by
  sorry

end NUMINAMATH_CALUDE_mary_apples_trees_ratio_l2572_257212


namespace NUMINAMATH_CALUDE_graduates_second_degree_l2572_257259

theorem graduates_second_degree (total : ℕ) (job : ℕ) (both : ℕ) (neither : ℕ) : 
  total = 73 → job = 32 → both = 13 → neither = 9 → 
  ∃ (second_degree : ℕ), second_degree = 45 := by
sorry

end NUMINAMATH_CALUDE_graduates_second_degree_l2572_257259


namespace NUMINAMATH_CALUDE_largest_power_dividing_product_l2572_257270

def pow (n : ℕ) : ℕ :=
  sorry

def product_pow : ℕ :=
  sorry

theorem largest_power_dividing_product :
  (∃ m : ℕ, (2310 : ℕ)^m ∣ product_pow ∧ 
    ∀ k : ℕ, (2310 : ℕ)^k ∣ product_pow → k ≤ m) ∧
  (∃ m : ℕ, (2310 : ℕ)^m ∣ product_pow ∧ m = 319) :=
by sorry

end NUMINAMATH_CALUDE_largest_power_dividing_product_l2572_257270


namespace NUMINAMATH_CALUDE_canoe_weight_problem_l2572_257227

theorem canoe_weight_problem (canoe_capacity : ℕ) (dog_weight_ratio : ℚ) (total_weight : ℕ) :
  canoe_capacity = 6 →
  dog_weight_ratio = 1/4 →
  total_weight = 595 →
  ∃ (person_weight : ℕ),
    person_weight = 140 ∧
    (↑(2 * canoe_capacity) / 3 : ℚ).floor * person_weight + 
    (dog_weight_ratio * person_weight).num / (dog_weight_ratio * person_weight).den = total_weight :=
by sorry

end NUMINAMATH_CALUDE_canoe_weight_problem_l2572_257227


namespace NUMINAMATH_CALUDE_gain_amount_proof_l2572_257204

/-- Given an article sold at $180 with a 20% gain, prove that the gain amount is $30. -/
theorem gain_amount_proof (selling_price : ℝ) (gain_percentage : ℝ) 
  (h1 : selling_price = 180)
  (h2 : gain_percentage = 0.20) : 
  let cost_price := selling_price / (1 + gain_percentage)
  selling_price - cost_price = 30 := by
sorry


end NUMINAMATH_CALUDE_gain_amount_proof_l2572_257204


namespace NUMINAMATH_CALUDE_m_range_l2572_257266

-- Define set A
def A : Set ℝ := {x | ∃ y, y = Real.sqrt (x^2 - 4*x + 3)}

-- Define set B
def B (m : ℝ) : Set ℝ := {y | ∃ x ∈ (Set.univ \ A), y = x + m/x ∧ m > 0}

-- Theorem statement
theorem m_range (m : ℝ) : (2 * Real.sqrt m ∈ B m) ↔ (1 < m ∧ m < 9) := by
  sorry

end NUMINAMATH_CALUDE_m_range_l2572_257266


namespace NUMINAMATH_CALUDE_max_value_expression_l2572_257261

/-- For positive real numbers a and b, and angle θ where 0 ≤ θ ≤ π/2,
    the maximum value of 2(a - x)(x + cos(θ)√(x^2 + b^2)) is a^2 + cos^2(θ)b^2 -/
theorem max_value_expression (a b : ℝ) (θ : ℝ) 
    (ha : a > 0) (hb : b > 0) (hθ : 0 ≤ θ ∧ θ ≤ π/2) :
  (⨆ x, 2 * (a - x) * (x + Real.cos θ * Real.sqrt (x^2 + b^2))) = a^2 + Real.cos θ^2 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_expression_l2572_257261


namespace NUMINAMATH_CALUDE_total_edges_theorem_l2572_257265

/-- A graph with the properties described in the problem -/
structure WonderGraph where
  n : ℕ  -- number of cities
  a : ℕ  -- number of roads
  connected : Bool  -- graph is connected
  at_most_one_edge : Bool  -- at most one edge between any two vertices
  indirect_path : Bool  -- indirect path exists between directly connected vertices

/-- The number of subgraphs with even degree vertices -/
def num_even_subgraphs (G : WonderGraph) : ℕ := sorry

/-- The total number of edges in all subgraphs with even degree vertices -/
def total_edges_in_even_subgraphs (G : WonderGraph) : ℕ := sorry

/-- Main theorem: The total number of edges in all subgraphs with even degree vertices is ar/2 -/
theorem total_edges_theorem (G : WonderGraph) :
  total_edges_in_even_subgraphs G = G.a * (num_even_subgraphs G) / 2 :=
sorry

end NUMINAMATH_CALUDE_total_edges_theorem_l2572_257265


namespace NUMINAMATH_CALUDE_juice_remaining_l2572_257244

theorem juice_remaining (initial : ℚ) (given : ℚ) (remaining : ℚ) : 
  initial = 5 → given = 18/7 → remaining = initial - given → remaining = 17/7 := by
  sorry

end NUMINAMATH_CALUDE_juice_remaining_l2572_257244


namespace NUMINAMATH_CALUDE_black_equals_sum_of_whites_l2572_257277

/-- Definition of a white number -/
def is_white_number (x : ℝ) : Prop :=
  ∃ (a b : ℤ), a ≠ 0 ∧ b ≠ 0 ∧ x = Real.sqrt (a + b * Real.sqrt 2)

/-- Definition of a black number -/
def is_black_number (x : ℝ) : Prop :=
  ∃ (c d : ℤ), c ≠ 0 ∧ d ≠ 0 ∧ x = Real.sqrt (c + d * Real.sqrt 7)

/-- Theorem stating that a black number can be equal to the sum of two white numbers -/
theorem black_equals_sum_of_whites :
  ∃ (x y z : ℝ), is_white_number x ∧ is_white_number y ∧ is_black_number z ∧ z = x + y :=
sorry

end NUMINAMATH_CALUDE_black_equals_sum_of_whites_l2572_257277
