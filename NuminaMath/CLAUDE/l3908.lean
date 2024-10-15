import Mathlib

namespace NUMINAMATH_CALUDE_tuesday_temperature_l3908_390885

/-- Given the average temperatures for three consecutive days and the temperature of the last day,
    this theorem proves the temperature of the first day. -/
theorem tuesday_temperature
  (avg_tues_wed_thurs : ℝ)
  (avg_wed_thurs_fri : ℝ)
  (temp_friday : ℝ)
  (h1 : avg_tues_wed_thurs = 32)
  (h2 : avg_wed_thurs_fri = 34)
  (h3 : temp_friday = 44) :
  ∃ (temp_tuesday temp_wednesday temp_thursday : ℝ),
    (temp_tuesday + temp_wednesday + temp_thursday) / 3 = avg_tues_wed_thurs ∧
    (temp_wednesday + temp_thursday + temp_friday) / 3 = avg_wed_thurs_fri ∧
    temp_tuesday = 38 := by
  sorry


end NUMINAMATH_CALUDE_tuesday_temperature_l3908_390885


namespace NUMINAMATH_CALUDE_range_of_a_l3908_390829

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (x ≤ a - 4 ∨ x ≥ a + 4) → (x ≤ 1 ∨ x ≥ 2)) → 
  a ∈ Set.Icc (-2) 5 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3908_390829


namespace NUMINAMATH_CALUDE_apollo_chariot_wheels_l3908_390821

theorem apollo_chariot_wheels (months_in_year : ℕ) (total_cost : ℕ) 
  (initial_rate : ℕ) (h1 : months_in_year = 12) (h2 : total_cost = 54) 
  (h3 : initial_rate = 3) : 
  ∃ (x : ℕ), x * initial_rate + (months_in_year - x) * (2 * initial_rate) = total_cost ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_apollo_chariot_wheels_l3908_390821


namespace NUMINAMATH_CALUDE_digit_sum_possibilities_l3908_390859

/-- Represents a digit from 0 to 9 -/
def Digit := Fin 10

/-- Predicate to check if four digits are all different -/
def all_different (a b c d : Digit) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

/-- The theorem stating the possible sums of four different digits -/
theorem digit_sum_possibilities (a b c d : Digit) 
  (h : all_different a b c d) :
  (a.val + b.val + c.val + d.val = 10) ∨ 
  (a.val + b.val + c.val + d.val = 18) ∨ 
  (a.val + b.val + c.val + d.val = 19) := by
  sorry

end NUMINAMATH_CALUDE_digit_sum_possibilities_l3908_390859


namespace NUMINAMATH_CALUDE_cube_volume_from_paper_area_l3908_390806

/-- Given a rectangular piece of paper with length 48 inches and width 72 inches
    that covers exactly the surface area of a cube, prove that the volume of the cube
    is 8 cubic feet, where 1 foot is 12 inches. -/
theorem cube_volume_from_paper_area (paper_length : ℝ) (paper_width : ℝ) 
    (inches_per_foot : ℝ) (h1 : paper_length = 48) (h2 : paper_width = 72) 
    (h3 : inches_per_foot = 12) : 
    let paper_area := paper_length * paper_width
    let cube_side_length := Real.sqrt (paper_area / 6)
    let cube_side_length_feet := cube_side_length / inches_per_foot
    cube_side_length_feet ^ 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_paper_area_l3908_390806


namespace NUMINAMATH_CALUDE_triangle_and_vector_theorem_l3908_390833

noncomputable section

open Real

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

/-- The given condition for the triangle -/
def triangleCondition (t : Triangle) : Prop :=
  (2 * t.a - t.c) * cos t.B = t.b * cos t.C

/-- The vector m -/
def m (A : ℝ) : ℝ × ℝ := (sin A, cos (2 * A))

/-- The vector n -/
def n : ℝ × ℝ := (6, 1)

/-- The dot product of m and n -/
def dotProduct (A : ℝ) : ℝ := 6 * sin A + cos (2 * A)

/-- The main theorem -/
theorem triangle_and_vector_theorem (t : Triangle) 
  (h : triangleCondition t) : 
  t.B = π / 3 ∧ 
  (∀ A, dotProduct A ≤ 5) ∧ 
  (∃ A, dotProduct A = 5) := by
  sorry

end

end NUMINAMATH_CALUDE_triangle_and_vector_theorem_l3908_390833


namespace NUMINAMATH_CALUDE_f_min_at_4_l3908_390801

/-- The quadratic function f(x) = x^2 - 8x + 15 -/
def f (x : ℝ) : ℝ := x^2 - 8*x + 15

/-- The theorem stating that f(x) attains its minimum at x = 4 -/
theorem f_min_at_4 :
  ∀ x : ℝ, f x ≥ f 4 :=
sorry

end NUMINAMATH_CALUDE_f_min_at_4_l3908_390801


namespace NUMINAMATH_CALUDE_fifth_match_goals_l3908_390851

/-- A football player's goal-scoring record over 5 matches -/
structure FootballRecord where
  total_matches : Nat
  total_goals : Nat
  fifth_match_goals : Nat
  average_increase : Rat

/-- The conditions of the problem -/
def problem_conditions (record : FootballRecord) : Prop :=
  record.total_matches = 5 ∧
  record.total_goals = 8 ∧
  record.average_increase = 1/10 ∧
  (4 * (record.total_goals - record.fifth_match_goals)) / 4 + record.average_increase = 
    record.total_goals / record.total_matches

/-- The theorem stating that under the given conditions, the player scored 2 goals in the fifth match -/
theorem fifth_match_goals (record : FootballRecord) 
  (h : problem_conditions record) : record.fifth_match_goals = 2 := by
  sorry


end NUMINAMATH_CALUDE_fifth_match_goals_l3908_390851


namespace NUMINAMATH_CALUDE_box_volume_increase_l3908_390815

/-- Proves that for a rectangular box with given dimensions, the value of x that
    satisfies the equation for equal volume increase when increasing length or height is 0. -/
theorem box_volume_increase (l w h : ℝ) (x : ℝ) : 
  l = 6 → w = 4 → h = 5 → ((l + x) * w * h = l * w * (h + x)) → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_increase_l3908_390815


namespace NUMINAMATH_CALUDE_yun_lost_paperclips_l3908_390813

theorem yun_lost_paperclips : ∀ (yun_current : ℕ),
  yun_current ≤ 20 →
  (1 + 1/4 : ℚ) * yun_current + 7 = 9 →
  20 - yun_current = 19 :=
by
  sorry

end NUMINAMATH_CALUDE_yun_lost_paperclips_l3908_390813


namespace NUMINAMATH_CALUDE_unique_integer_divisibility_l3908_390852

theorem unique_integer_divisibility : ∃! n : ℕ, n > 1 ∧
  ∀ p : ℕ, Prime p → (p ∣ (n^6 - 1) → p ∣ ((n^3 - 1) * (n^2 - 1))) ∧
  n = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_divisibility_l3908_390852


namespace NUMINAMATH_CALUDE_max_residents_is_475_l3908_390888

/-- Represents the configuration of a section in the block of flats -/
structure SectionConfig where
  floors : Nat
  apartments_per_floor : Nat
  apartment_capacities : List Nat

/-- Calculates the maximum capacity of a single section -/
def section_capacity (config : SectionConfig) : Nat :=
  config.floors * (config.apartment_capacities.sum)

/-- Represents the configuration of the entire block of flats -/
structure BlockConfig where
  lower : SectionConfig
  middle : SectionConfig
  upper : SectionConfig

/-- Calculates the maximum capacity of the entire block -/
def block_capacity (config : BlockConfig) : Nat :=
  section_capacity config.lower + section_capacity config.middle + section_capacity config.upper

/-- The actual configuration of the block as described in the problem -/
def actual_block_config : BlockConfig :=
  { lower := { floors := 4, apartments_per_floor := 5, apartment_capacities := [4, 4, 5, 5, 6] },
    middle := { floors := 5, apartments_per_floor := 6, apartment_capacities := [3, 3, 3, 4, 4, 6] },
    upper := { floors := 6, apartments_per_floor := 5, apartment_capacities := [8, 8, 8, 10, 10] } }

/-- Theorem stating that the maximum capacity of the block is 475 -/
theorem max_residents_is_475 : block_capacity actual_block_config = 475 := by
  sorry

end NUMINAMATH_CALUDE_max_residents_is_475_l3908_390888


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l3908_390830

theorem absolute_value_equation_solution :
  ∃! n : ℚ, |n + 4| = 3 - n :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l3908_390830


namespace NUMINAMATH_CALUDE_sum_consecutive_odd_iff_multiple_of_four_l3908_390810

/-- An even composite number can be represented as the sum of consecutive odd numbers
    if and only if it is a multiple of 4. -/
theorem sum_consecutive_odd_iff_multiple_of_four (m : ℕ) :
  (∃ (n : ℕ) (a : ℤ), m = n * (2 * a + n) ∧ n > 1) ↔ 4 ∣ m ∧ m > 2 :=
sorry

end NUMINAMATH_CALUDE_sum_consecutive_odd_iff_multiple_of_four_l3908_390810


namespace NUMINAMATH_CALUDE_cookie_boxes_theorem_l3908_390843

theorem cookie_boxes_theorem (n : ℕ) : 
  (∃ (mark_sold ann_sold : ℕ),
    mark_sold = n - 8 ∧ 
    ann_sold = n - 2 ∧ 
    mark_sold ≥ 1 ∧ 
    ann_sold ≥ 1 ∧ 
    mark_sold + ann_sold < n) → 
  n = 9 := by
sorry

end NUMINAMATH_CALUDE_cookie_boxes_theorem_l3908_390843


namespace NUMINAMATH_CALUDE_sum_of_squares_l3908_390854

theorem sum_of_squares (x y : ℝ) (h1 : x * y = 120) (h2 : x + y = 23) : x^2 + y^2 = 289 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l3908_390854


namespace NUMINAMATH_CALUDE_leak_empty_time_proof_l3908_390868

/-- Represents the time it takes for a pipe to fill a tank -/
def fill_time_no_leak : ℝ := 8

/-- Represents the time it takes for a pipe to fill a tank with a leak -/
def fill_time_with_leak : ℝ := 12

/-- Represents the time it takes for the leak to empty a full tank -/
def leak_empty_time : ℝ := 24

/-- Theorem stating that given the fill times with and without a leak, 
    the time for the leak to empty the tank is 24 hours -/
theorem leak_empty_time_proof : 
  ∀ (fill_rate : ℝ) (leak_rate : ℝ),
  fill_rate = 1 / fill_time_no_leak →
  fill_rate - leak_rate = 1 / fill_time_with_leak →
  1 / leak_rate = leak_empty_time :=
by sorry

end NUMINAMATH_CALUDE_leak_empty_time_proof_l3908_390868


namespace NUMINAMATH_CALUDE_left_handed_jazz_lovers_l3908_390819

/-- Represents a club with members of different handedness and music preferences -/
structure Club where
  total_members : ℕ
  left_handed : ℕ
  jazz_lovers : ℕ
  right_handed_non_jazz : ℕ

/-- Theorem stating the number of left-handed jazz lovers in the club -/
theorem left_handed_jazz_lovers (c : Club)
  (h1 : c.total_members = 20)
  (h2 : c.left_handed = 8)
  (h3 : c.jazz_lovers = 15)
  (h4 : c.right_handed_non_jazz = 2)
  (h5 : c.left_handed + (c.total_members - c.left_handed) = c.total_members) :
  c.left_handed + c.jazz_lovers - c.total_members + c.right_handed_non_jazz = 5 := by
  sorry

#check left_handed_jazz_lovers

end NUMINAMATH_CALUDE_left_handed_jazz_lovers_l3908_390819


namespace NUMINAMATH_CALUDE_min_product_of_three_numbers_l3908_390874

theorem min_product_of_three_numbers (x y z : ℝ) : 
  x > 0 → y > 0 → z > 0 → 
  x + y + z = 1 → 
  x ≤ 3*y ∧ x ≤ 3*z ∧ y ≤ 3*x ∧ y ≤ 3*z ∧ z ≤ 3*x ∧ z ≤ 3*y →
  x * y * z ≥ 1/36 :=
by sorry

end NUMINAMATH_CALUDE_min_product_of_three_numbers_l3908_390874


namespace NUMINAMATH_CALUDE_ten_year_old_dog_human_years_l3908_390850

/-- Calculates the equivalent human years for a dog's age. -/
def dogYearsToHumanYears (dogAge : ℕ) : ℕ :=
  if dogAge = 0 then 0
  else if dogAge = 1 then 15
  else if dogAge = 2 then 24
  else 24 + 5 * (dogAge - 2)

/-- Theorem: A 10-year-old dog has lived 64 human years. -/
theorem ten_year_old_dog_human_years :
  dogYearsToHumanYears 10 = 64 := by
  sorry

end NUMINAMATH_CALUDE_ten_year_old_dog_human_years_l3908_390850


namespace NUMINAMATH_CALUDE_undergrad_sample_count_l3908_390863

/-- Represents the number of undergraduate students in a stratified sample -/
def undergrad_sample_size (total_population : ℕ) (undergrad_population : ℕ) (sample_size : ℕ) : ℕ :=
  (undergrad_population * sample_size) / total_population

/-- Theorem stating the number of undergraduate students in the stratified sample -/
theorem undergrad_sample_count :
  undergrad_sample_size 5600 3000 280 = 150 := by
  sorry

end NUMINAMATH_CALUDE_undergrad_sample_count_l3908_390863


namespace NUMINAMATH_CALUDE_yan_position_ratio_l3908_390836

/-- Yan's position between home and stadium -/
structure Position where
  home_dist : ℝ     -- distance from home
  stadium_dist : ℝ  -- distance to stadium
  home_dist_nonneg : 0 ≤ home_dist
  stadium_dist_nonneg : 0 ≤ stadium_dist

/-- Yan's walking speed -/
def walking_speed : ℝ := 1

/-- Yan's cycling speed -/
def cycling_speed : ℝ := 4 * walking_speed

theorem yan_position_ratio (pos : Position) : 
  (pos.home_dist / pos.stadium_dist = 3 / 5) ↔ 
  (pos.stadium_dist / walking_speed = 
   pos.home_dist / walking_speed + (pos.home_dist + pos.stadium_dist) / cycling_speed) := by
  sorry

end NUMINAMATH_CALUDE_yan_position_ratio_l3908_390836


namespace NUMINAMATH_CALUDE_adults_on_bus_l3908_390807

theorem adults_on_bus (total_passengers : ℕ) (children_fraction : ℚ) : 
  total_passengers = 360 → children_fraction = 3/7 → 
  (total_passengers : ℚ) * (1 - children_fraction) = 205 := by
sorry

end NUMINAMATH_CALUDE_adults_on_bus_l3908_390807


namespace NUMINAMATH_CALUDE_prob_two_cards_two_suits_l3908_390842

/-- The probability of drawing a card of a specific suit from a standard deck -/
def prob_specific_suit : ℚ := 1 / 4

/-- The number of cards drawn -/
def num_draws : ℕ := 6

/-- The number of suits we're interested in -/
def num_suits : ℕ := 2

/-- The number of cards needed from each suit -/
def cards_per_suit : ℕ := 2

/-- The probability of getting the desired outcome when drawing six cards with replacement -/
def prob_desired_outcome : ℚ := (prob_specific_suit ^ (num_draws : ℕ))

theorem prob_two_cards_two_suits :
  prob_desired_outcome = 1 / 4096 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_cards_two_suits_l3908_390842


namespace NUMINAMATH_CALUDE_initial_peanuts_count_l3908_390870

theorem initial_peanuts_count (initial final added : ℕ) 
  (h1 : added = 4)
  (h2 : final = 8)
  (h3 : final = initial + added) : 
  initial = 4 := by
sorry

end NUMINAMATH_CALUDE_initial_peanuts_count_l3908_390870


namespace NUMINAMATH_CALUDE_operation_result_l3908_390825

-- Define the set of operations
inductive Operation
  | Add
  | Sub
  | Mul
  | Div

-- Define a function to apply an operation
def apply_op (op : Operation) (a b : ℚ) : ℚ :=
  match op with
  | Operation.Add => a + b
  | Operation.Sub => a - b
  | Operation.Mul => a * b
  | Operation.Div => a / b

-- State the theorem
theorem operation_result (star mul : Operation) 
  (h : apply_op star 12 2 / apply_op mul 9 3 = 2) :
  apply_op star 7 3 / apply_op mul 12 6 = 7 / 6 := by
  sorry

end NUMINAMATH_CALUDE_operation_result_l3908_390825


namespace NUMINAMATH_CALUDE_division_result_l3908_390880

theorem division_result : (180 : ℚ) / (12 + 13 * 3 - 5) = 90 / 23 := by
  sorry

end NUMINAMATH_CALUDE_division_result_l3908_390880


namespace NUMINAMATH_CALUDE_triangle_side_length_l3908_390899

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) (S_ABC : ℝ) :
  a = 4 →
  B = π / 3 →
  S_ABC = 6 * Real.sqrt 3 →
  b = 2 * Real.sqrt 7 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3908_390899


namespace NUMINAMATH_CALUDE_shape_division_count_l3908_390834

/-- A shape with 17 cells -/
def Shape : Type := Unit

/-- A rectangle of size 1 × 2 -/
def Rectangle : Type := Unit

/-- A square of size 1 × 1 -/
def Square : Type := Unit

/-- A division of the shape into rectangles and a square -/
def Division : Type := List Rectangle × Square

/-- The number of ways to divide the shape -/
def numDivisions (s : Shape) : ℕ := 10

/-- Theorem: There are 10 ways to divide the shape into 8 rectangles and 1 square -/
theorem shape_division_count (s : Shape) :
  (numDivisions s = 10) ∧
  (∀ d : Division, List.length (d.1) = 8) :=
sorry

end NUMINAMATH_CALUDE_shape_division_count_l3908_390834


namespace NUMINAMATH_CALUDE_lunchroom_students_l3908_390893

theorem lunchroom_students (tables : ℕ) (avg_students : ℚ) : 
  tables = 34 →
  avg_students = 5666666667 / 1000000000 →
  ∃ (total_students : ℕ), total_students = 204 ∧ total_students % tables = 0 := by
  sorry

end NUMINAMATH_CALUDE_lunchroom_students_l3908_390893


namespace NUMINAMATH_CALUDE_trigonometric_expression_evaluation_l3908_390871

theorem trigonometric_expression_evaluation : 3 * Real.cos 0 + 4 * Real.sin (3 * Real.pi / 2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_evaluation_l3908_390871


namespace NUMINAMATH_CALUDE_garden_roller_diameter_l3908_390890

/-- The diameter of a garden roller given its length, area covered, and number of revolutions -/
theorem garden_roller_diameter 
  (length : ℝ) 
  (area_covered : ℝ) 
  (revolutions : ℝ) 
  (h1 : length = 3) 
  (h2 : area_covered = 66) 
  (h3 : revolutions = 5) : 
  ∃ (diameter : ℝ), diameter = 1.4 ∧ 
    area_covered = revolutions * 2 * (22/7) * (diameter/2) * length := by
sorry

end NUMINAMATH_CALUDE_garden_roller_diameter_l3908_390890


namespace NUMINAMATH_CALUDE_propositions_truth_l3908_390879

-- Proposition ①
def proposition_1 : Prop := 
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x₀ : ℝ, x₀^2 ≥ 0)

-- Proposition ②
def proposition_2 : Prop := 
  (¬ ∃ x₀ : ℝ, x₀^2 - x₀ > 0) ↔ (∀ x : ℝ, x^2 - x < 0)

-- Proposition ③
def proposition_3 : Prop := 
  ∀ x : ℝ, x ∈ Set.Icc (-1) 2 → x^2 - 2*x ≤ 3

-- Proposition ④
def proposition_4 : Prop := 
  ∃ x₀ : ℝ, x₀^2 + 1/(x₀^2 + 1) ≤ 1

theorem propositions_truth : 
  ¬ proposition_1 ∧ 
  ¬ proposition_2 ∧ 
  proposition_3 ∧ 
  proposition_4 := by sorry

end NUMINAMATH_CALUDE_propositions_truth_l3908_390879


namespace NUMINAMATH_CALUDE_cupcakes_per_event_l3908_390884

theorem cupcakes_per_event (total_cupcakes : ℕ) (num_events : ℕ) 
  (h1 : total_cupcakes = 768) 
  (h2 : num_events = 8) :
  total_cupcakes / num_events = 96 := by
  sorry

end NUMINAMATH_CALUDE_cupcakes_per_event_l3908_390884


namespace NUMINAMATH_CALUDE_max_frisbee_receipts_l3908_390804

/-- Represents the total receipts from frisbee sales -/
def total_receipts (x y : ℕ) : ℕ := 3 * x + 4 * y

/-- Proves that the maximum total receipts from frisbee sales is $204 -/
theorem max_frisbee_receipts :
  ∀ x y : ℕ,
  x + y = 60 →
  y ≥ 24 →
  total_receipts x y ≤ 204 :=
by
  sorry

#eval total_receipts 36 24  -- Should output 204

end NUMINAMATH_CALUDE_max_frisbee_receipts_l3908_390804


namespace NUMINAMATH_CALUDE_real_gdp_change_omega_l3908_390841

/-- Represents the production and price data for Omega in a given year -/
structure YearData where
  vegetable_production : ℕ
  fruit_production : ℕ
  vegetable_price : ℕ
  fruit_price : ℕ

/-- Calculates the nominal GDP for a given year -/
def nominalGDP (data : YearData) : ℕ :=
  data.vegetable_production * data.vegetable_price + data.fruit_production * data.fruit_price

/-- Calculates the real GDP for a given year using base year prices -/
def realGDP (data : YearData) (base : YearData) : ℕ :=
  data.vegetable_production * base.vegetable_price + data.fruit_production * base.fruit_price

/-- Calculates the percentage change in GDP -/
def percentageChange (old : ℕ) (new : ℕ) : ℚ :=
  100 * (new - old : ℚ) / old

/-- The main theorem stating the percentage change in real GDP -/
theorem real_gdp_change_omega :
  let data2014 : YearData := {
    vegetable_production := 1200,
    fruit_production := 750,
    vegetable_price := 90000,
    fruit_price := 75000
  }
  let data2015 : YearData := {
    vegetable_production := 900,
    fruit_production := 900,
    vegetable_price := 100000,
    fruit_price := 70000
  }
  let nominal2014 := nominalGDP data2014
  let real2015 := realGDP data2015 data2014
  let change := percentageChange nominal2014 real2015
  ∃ ε > 0, |change + 9.59| < ε :=
by sorry

end NUMINAMATH_CALUDE_real_gdp_change_omega_l3908_390841


namespace NUMINAMATH_CALUDE_binomial_20_19_l3908_390872

theorem binomial_20_19 : Nat.choose 20 19 = 20 := by sorry

end NUMINAMATH_CALUDE_binomial_20_19_l3908_390872


namespace NUMINAMATH_CALUDE_parabola_directrix_l3908_390803

/-- Given a parabola y = -3x^2 + 6x - 7, its directrix is y = -47/12 -/
theorem parabola_directrix (x y : ℝ) : 
  y = -3 * x^2 + 6 * x - 7 → 
  ∃ (k : ℝ), k = -47/12 ∧ (∀ (p : ℝ × ℝ), p.1 = x ∧ p.2 = y → 
    (p.1 - 1)^2 + (p.2 - k)^2 = (p.2 + 4)^2 / 9) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l3908_390803


namespace NUMINAMATH_CALUDE_least_exponent_sum_for_500_l3908_390857

def isPowerOfTwo (n : ℕ) : Prop := ∃ k, n = 2^k

def isDistinctPowersOfTwoSum (n : ℕ) (exponents : List ℕ) : Prop :=
  n = (exponents.map (λ e => 2^e)).sum ∧ 
  exponents.length ≥ 2 ∧
  exponents.Nodup

theorem least_exponent_sum_for_500 :
  ∃ (exponents : List ℕ),
    isDistinctPowersOfTwoSum 500 exponents ∧
    exponents.sum = 32 ∧
    ∀ (other_exponents : List ℕ),
      isDistinctPowersOfTwoSum 500 other_exponents →
      other_exponents.sum ≥ 32 :=
by sorry

end NUMINAMATH_CALUDE_least_exponent_sum_for_500_l3908_390857


namespace NUMINAMATH_CALUDE_tangent_line_and_inequality_l3908_390809

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x + 1) * Real.log x - a * (x - 1)

theorem tangent_line_and_inequality (a : ℝ) :
  (∃ x₀ : ℝ, (∀ x : ℝ, f a x ≥ x - 1) ∧ (f a x₀ = x₀ - 1)) →
  (a = 1) ∧
  (∀ x : ℝ, 1 < x → x < 2 → (1 / Real.log x) - (1 / Real.log (x - 1)) < 1 / ((x - 1) * (2 - x))) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_and_inequality_l3908_390809


namespace NUMINAMATH_CALUDE_last_remaining_number_l3908_390877

def josephus_sequence (n : ℕ) : ℕ → ℕ
| 0 => 1
| m + 1 => 2 * josephus_sequence n m

theorem last_remaining_number :
  ∃ k : ℕ, josephus_sequence 200 k = 128 ∧ josephus_sequence 200 (k + 1) > 200 :=
by sorry

end NUMINAMATH_CALUDE_last_remaining_number_l3908_390877


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_three_sqrt_two_over_two_l3908_390826

theorem sqrt_sum_equals_three_sqrt_two_over_two 
  (a b : ℝ) (h1 : a + b = -6) (h2 : a * b = 8) :
  Real.sqrt (b / a) + Real.sqrt (a / b) = 3 * Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_three_sqrt_two_over_two_l3908_390826


namespace NUMINAMATH_CALUDE_commercial_length_l3908_390869

theorem commercial_length (original_length : ℝ) : 
  (original_length * 0.7 = 21) → original_length = 30 := by
  sorry

end NUMINAMATH_CALUDE_commercial_length_l3908_390869


namespace NUMINAMATH_CALUDE_cube_root_of_zero_l3908_390858

theorem cube_root_of_zero (x : ℝ) : x^3 = 0 → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_zero_l3908_390858


namespace NUMINAMATH_CALUDE_triangle_altitude_l3908_390828

/-- Given a rectangle with length 3s and width s, and a triangle inside with one side
    along the diagonal and area half of the rectangle's area, the altitude of the
    triangle to the diagonal base is 3s√10/10 -/
theorem triangle_altitude (s : ℝ) (h : s > 0) :
  let l := 3 * s
  let w := s
  let diagonal := Real.sqrt (l^2 + w^2)
  let rectangle_area := l * w
  let triangle_area := rectangle_area / 2
  triangle_area = (1 / 2) * diagonal * (3 * s * Real.sqrt 10 / 10) :=
by sorry

end NUMINAMATH_CALUDE_triangle_altitude_l3908_390828


namespace NUMINAMATH_CALUDE_cats_dogs_ratio_l3908_390831

/-- Given a ratio of cats to dogs and the number of cats, calculate the number of dogs -/
theorem cats_dogs_ratio (cat_ratio : ℕ) (dog_ratio : ℕ) (num_cats : ℕ) (num_dogs : ℕ) : 
  cat_ratio * num_dogs = dog_ratio * num_cats → 
  cat_ratio = 3 → 
  dog_ratio = 4 → 
  num_cats = 18 → 
  num_dogs = 24 := by
sorry

end NUMINAMATH_CALUDE_cats_dogs_ratio_l3908_390831


namespace NUMINAMATH_CALUDE_rectangle_length_calculation_l3908_390838

theorem rectangle_length_calculation (square_side : ℝ) (rect_width : ℝ) (rect_length : ℝ) :
  square_side = 12 →
  rect_width = 6 →
  square_side * square_side = rect_width * rect_length →
  rect_length = 24 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_length_calculation_l3908_390838


namespace NUMINAMATH_CALUDE_vector_q_in_terms_of_c_and_d_l3908_390820

/-- Given a line segment CD and points P and Q, where P divides CD internally
    in the ratio 3:5 and Q divides DP externally in the ratio 1:2,
    prove that vector Q can be expressed in terms of vectors C and D. -/
theorem vector_q_in_terms_of_c_and_d
  (C D P Q : EuclideanSpace ℝ (Fin 3))
  (h_P_on_CD : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • C + t • D)
  (h_CP_PD : ∃ k : ℝ, k > 0 ∧ dist C P = k * (3 / 8) ∧ dist P D = k * (5 / 8))
  (h_Q_external : ∃ s : ℝ, s < 0 ∧ Q = (1 - s) • D + s • P ∧ abs s = 2) :
  Q = (5 / 8) • C + (-13 / 8) • D :=
by sorry

end NUMINAMATH_CALUDE_vector_q_in_terms_of_c_and_d_l3908_390820


namespace NUMINAMATH_CALUDE_sum_of_exterior_angles_of_triangle_l3908_390878

theorem sum_of_exterior_angles_of_triangle (α β γ : ℝ) (α' β' γ' : ℝ) : 
  α + β + γ = 180 →
  α + α' = 180 →
  β + β' = 180 →
  γ + γ' = 180 →
  α' + β' + γ' = 360 := by
sorry

end NUMINAMATH_CALUDE_sum_of_exterior_angles_of_triangle_l3908_390878


namespace NUMINAMATH_CALUDE_sum_of_naturals_l3908_390847

theorem sum_of_naturals (n : ℕ) : 
  (List.range (n + 1)).sum = n * (n + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_naturals_l3908_390847


namespace NUMINAMATH_CALUDE_constant_theta_is_plane_l3908_390845

-- Define spherical coordinates
structure SphericalCoord where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

-- Define the condition θ = c
def constant_theta (c : ℝ) (p : SphericalCoord) : Prop :=
  p.θ = c

-- Define a plane in 3D space
def is_plane (S : Set SphericalCoord) : Prop :=
  ∃ (a b d : ℝ), ∀ (p : SphericalCoord), p ∈ S ↔ 
    a * (p.ρ * Real.sin p.φ * Real.cos p.θ) + 
    b * (p.ρ * Real.sin p.φ * Real.sin p.θ) + 
    d * (p.ρ * Real.cos p.φ) = 0

-- Theorem statement
theorem constant_theta_is_plane (c : ℝ) :
  is_plane {p : SphericalCoord | constant_theta c p} :=
sorry

end NUMINAMATH_CALUDE_constant_theta_is_plane_l3908_390845


namespace NUMINAMATH_CALUDE_parabola_satisfies_conditions_l3908_390853

-- Define the parabola equation
def parabola_equation (x y : ℝ) : Prop :=
  y^2 - 8*x - 8*y + 16 = 0

-- Define the conditions
def passes_through_point (eq : ℝ → ℝ → Prop) : Prop :=
  eq 2 8

def focus_y_coordinate (eq : ℝ → ℝ → Prop) : Prop :=
  ∃ x, eq x 4 ∧ (∀ y, eq x y → (y - 4)^2 ≤ (8 - 4)^2)

def axis_parallel_to_x (eq : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, eq x y → eq x (-y + 8)

def vertex_on_y_axis (eq : ℝ → ℝ → Prop) : Prop :=
  ∃ y, eq 0 y

def coefficients_are_integers (a b c d e f : ℤ) : Prop :=
  ∀ x y : ℝ, (a:ℝ)*x^2 + (b:ℝ)*x*y + (c:ℝ)*y^2 + (d:ℝ)*x + (e:ℝ)*y + (f:ℝ) = 0 ↔ parabola_equation x y

def c_is_positive (c : ℤ) : Prop :=
  c > 0

def gcd_is_one (a b c d e f : ℤ) : Prop :=
  Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd (a.natAbs) (b.natAbs)) (c.natAbs)) (d.natAbs)) (e.natAbs)) (f.natAbs) = 1

-- State the theorem
theorem parabola_satisfies_conditions :
  ∃ a b c d e f : ℤ,
    passes_through_point parabola_equation ∧
    focus_y_coordinate parabola_equation ∧
    axis_parallel_to_x parabola_equation ∧
    vertex_on_y_axis parabola_equation ∧
    coefficients_are_integers a b c d e f ∧
    c_is_positive c ∧
    gcd_is_one a b c d e f :=
  sorry

end NUMINAMATH_CALUDE_parabola_satisfies_conditions_l3908_390853


namespace NUMINAMATH_CALUDE_propositions_correctness_l3908_390823

theorem propositions_correctness :
  (∃ a b, a > b ∧ b > 0 ∧ (1 / a ≥ 1 / b)) ∧
  (∀ a b, a > b ∧ b > 0 → a^2 - a > b^2 - b) ∧
  (∃ a b, a > b ∧ b > 0 ∧ a^3 ≤ b^3) ∧
  (∀ a b, a > 0 ∧ b > 0 ∧ 2*a + b = 1 → 
    (∀ x y, x > 0 ∧ y > 0 ∧ 2*x + y = 1 → a^2 + b^2 ≤ x^2 + y^2) ∧
    a^2 + b^2 = 1/9) :=
by sorry

end NUMINAMATH_CALUDE_propositions_correctness_l3908_390823


namespace NUMINAMATH_CALUDE_leadership_combinations_l3908_390875

def tribe_size : ℕ := 15
def num_chiefs : ℕ := 1
def num_supporting_chiefs : ℕ := 2
def num_inferior_officers_per_chief : ℕ := 3

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem leadership_combinations : 
  (tribe_size) * 
  (tribe_size - num_chiefs) * 
  (tribe_size - num_chiefs - 1) * 
  (choose (tribe_size - num_chiefs - num_supporting_chiefs) num_inferior_officers_per_chief) *
  (choose (tribe_size - num_chiefs - num_supporting_chiefs - num_inferior_officers_per_chief) num_inferior_officers_per_chief) = 3243240 := by
  sorry

end NUMINAMATH_CALUDE_leadership_combinations_l3908_390875


namespace NUMINAMATH_CALUDE_circle_points_condition_l3908_390856

-- Define the circle equation
def circle_equation (x y a : ℝ) : Prop := x^2 + y^2 + a*x - 1 = 0

-- Define the points A and B
def point_A : ℝ × ℝ := (1, 2)
def point_B : ℝ × ℝ := (2, 1)

-- Define the condition for a point being inside or outside the circle
def is_inside_circle (x y a : ℝ) : Prop := x^2 + y^2 + a*x - 1 < 0
def is_outside_circle (x y a : ℝ) : Prop := x^2 + y^2 + a*x - 1 > 0

-- Theorem statement
theorem circle_points_condition (a : ℝ) :
  (is_inside_circle point_A.1 point_A.2 a ∧ is_outside_circle point_B.1 point_B.2 a) ∨
  (is_outside_circle point_A.1 point_A.2 a ∧ is_inside_circle point_B.1 point_B.2 a) →
  -4 < a ∧ a < -2 := by
  sorry

end NUMINAMATH_CALUDE_circle_points_condition_l3908_390856


namespace NUMINAMATH_CALUDE_magic_card_profit_100_l3908_390849

/-- Calculates the profit from selling a Magic card that has tripled in value --/
def magic_card_profit (purchase_price : ℝ) : ℝ :=
  3 * purchase_price - purchase_price

theorem magic_card_profit_100 :
  magic_card_profit 100 = 200 := by
  sorry

#eval magic_card_profit 100

end NUMINAMATH_CALUDE_magic_card_profit_100_l3908_390849


namespace NUMINAMATH_CALUDE_fallen_piece_theorem_l3908_390860

/-- A function that checks if two numbers have the same digits --/
def same_digits (a b : ℕ) : Prop := sorry

/-- The number of pages in a fallen piece of a book --/
def fallen_piece_pages (first_page last_page : ℕ) : ℕ :=
  last_page - first_page + 1

theorem fallen_piece_theorem :
  ∃ (last_page : ℕ),
    last_page > 328 ∧
    same_digits last_page 328 ∧
    fallen_piece_pages 328 last_page = 496 := by
  sorry

end NUMINAMATH_CALUDE_fallen_piece_theorem_l3908_390860


namespace NUMINAMATH_CALUDE_product_of_successive_numbers_l3908_390894

theorem product_of_successive_numbers : 
  let n : ℝ := 64.4980619863884
  let product := n * (n + 1)
  ∀ ε > 0, |product - 4225| < ε
:= by sorry

end NUMINAMATH_CALUDE_product_of_successive_numbers_l3908_390894


namespace NUMINAMATH_CALUDE_positive_real_inequality_general_real_inequality_l3908_390802

-- Part 1
theorem positive_real_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a^2 / b ≥ 2*a - b := by sorry

-- Part 2
theorem general_real_inequality (a b : ℝ) :
  a^2 + b^2 + 3 ≥ a*b + Real.sqrt 3 * (a + b) := by sorry

end NUMINAMATH_CALUDE_positive_real_inequality_general_real_inequality_l3908_390802


namespace NUMINAMATH_CALUDE_emilys_purchase_cost_l3908_390800

/-- The total cost of Emily's purchase including installation service -/
theorem emilys_purchase_cost : 
  let curtain_pairs : ℕ := 2
  let curtain_price : ℚ := 30
  let wall_prints : ℕ := 9
  let wall_print_price : ℚ := 15
  let installation_fee : ℚ := 50
  (curtain_pairs : ℚ) * curtain_price + 
  (wall_prints : ℚ) * wall_print_price + 
  installation_fee = 245 :=
by sorry

end NUMINAMATH_CALUDE_emilys_purchase_cost_l3908_390800


namespace NUMINAMATH_CALUDE_students_in_both_activities_l3908_390886

theorem students_in_both_activities (total : ℕ) (band : ℕ) (sports : ℕ) (either : ℕ) 
  (h1 : total = 320)
  (h2 : band = 85)
  (h3 : sports = 200)
  (h4 : either = 225) :
  band + sports - either = 60 :=
by sorry

end NUMINAMATH_CALUDE_students_in_both_activities_l3908_390886


namespace NUMINAMATH_CALUDE_more_seventh_graders_l3908_390898

theorem more_seventh_graders (n m : ℕ) 
  (h1 : n > 0) 
  (h2 : m > 0) 
  (h3 : 7 * n = 6 * m) : 
  m > n :=
by
  sorry

end NUMINAMATH_CALUDE_more_seventh_graders_l3908_390898


namespace NUMINAMATH_CALUDE_valid_sequences_of_length_20_l3908_390822

/-- Counts valid binary sequences of given length -/
def countValidSequences (n : ℕ) : ℕ :=
  if n < 3 then 0
  else if n = 3 then 1
  else countValidSequences (n - 4) + 2 * countValidSequences (n - 5) + countValidSequences (n - 6)

/-- Theorem stating the number of valid sequences of length 20 -/
theorem valid_sequences_of_length_20 :
  countValidSequences 20 = 86 := by sorry

end NUMINAMATH_CALUDE_valid_sequences_of_length_20_l3908_390822


namespace NUMINAMATH_CALUDE_min_modulus_of_complex_l3908_390896

theorem min_modulus_of_complex (t : ℝ) : 
  let z : ℂ := (t - 1) + (t + 1) * I
  ∃ (m : ℝ), (∀ t : ℝ, Complex.abs z ≥ m) ∧ (∃ t₀ : ℝ, Complex.abs (((t₀ - 1) : ℂ) + (t₀ + 1) * I) = m) ∧ m = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_modulus_of_complex_l3908_390896


namespace NUMINAMATH_CALUDE_no_zero_root_l3908_390835

-- Define the three equations
def equation1 (x : ℝ) : Prop := 4 * x^2 - 4 = 36
def equation2 (x : ℝ) : Prop := (2*x + 1)^2 = (x + 2)^2
def equation3 (x : ℝ) : Prop := (x^2 - 9 : ℝ) = x + 2

-- Theorem statement
theorem no_zero_root :
  (∀ x : ℝ, equation1 x → x ≠ 0) ∧
  (∀ x : ℝ, equation2 x → x ≠ 0) ∧
  (∀ x : ℝ, equation3 x → x ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_no_zero_root_l3908_390835


namespace NUMINAMATH_CALUDE_constant_shift_invariance_l3908_390864

variable {n : ℕ}
variable (X Y : Fin n → ℝ)
variable (c : ℝ)

def addConstant (X : Fin n → ℝ) (c : ℝ) : Fin n → ℝ :=
  fun i => X i + c

def sampleStandardDeviation (X : Fin n → ℝ) : ℝ :=
  sorry

def sampleRange (X : Fin n → ℝ) : ℝ :=
  sorry

theorem constant_shift_invariance (hc : c ≠ 0) (hY : Y = addConstant X c) :
  sampleStandardDeviation X = sampleStandardDeviation Y ∧
  sampleRange X = sampleRange Y :=
sorry

end NUMINAMATH_CALUDE_constant_shift_invariance_l3908_390864


namespace NUMINAMATH_CALUDE_valid_pairs_count_l3908_390892

def is_valid_pair (x y : ℕ) : Prop :=
  x ≠ y ∧
  x < 10 ∧ y < 10 ∧
  let product := (x * 1111) * (y * 1111)
  product ≥ 1000000 ∧ product < 10000000 ∧
  product % 10 = x ∧ (product / 1000000) % 10 = x

theorem valid_pairs_count :
  ∃ (S : Finset (ℕ × ℕ)), (∀ p ∈ S, is_valid_pair p.1 p.2) ∧ S.card = 3 :=
sorry

end NUMINAMATH_CALUDE_valid_pairs_count_l3908_390892


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3908_390891

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : (2 : ℝ)^(2*x + 2*y) = 2) : 
  ∃ (m : ℝ), m = 3 + 2 * Real.sqrt 2 ∧ ∀ (a b : ℝ), a > 0 → b > 0 → (2 : ℝ)^(2*a + 2*b) = 2 → 1/a + 1/b ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3908_390891


namespace NUMINAMATH_CALUDE_sum_square_free_coefficients_eight_l3908_390812

/-- A function that calculates the sum of coefficients of square-free terms -/
def sumSquareFreeCoefficients (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | k + 2 => sumSquareFreeCoefficients (k + 1) + (k + 1) * sumSquareFreeCoefficients k

/-- The main theorem stating that the sum of coefficients of square-free terms
    in the product of (1 + xᵢxⱼ) for 1 ≤ i < j ≤ 8 is 764 -/
theorem sum_square_free_coefficients_eight :
  sumSquareFreeCoefficients 8 = 764 := by
  sorry

/-- Helper lemma: The recurrence relation for sumSquareFreeCoefficients -/
lemma sum_square_free_coefficients_recurrence (n : ℕ) :
  n ≥ 2 →
  sumSquareFreeCoefficients n = 
    sumSquareFreeCoefficients (n-1) + (n-1) * sumSquareFreeCoefficients (n-2) := by
  sorry

end NUMINAMATH_CALUDE_sum_square_free_coefficients_eight_l3908_390812


namespace NUMINAMATH_CALUDE_runners_visibility_probability_l3908_390866

/-- Represents a runner on a circular track -/
structure Runner where
  name : String
  lapTime : ℕ
  direction : Bool  -- True for counterclockwise, False for clockwise

/-- Represents the circular track -/
structure Track where
  circumference : ℝ
  photoCoverage : ℝ
  shadowInterval : ℕ
  shadowDuration : ℕ

/-- Calculates the probability of both runners being visible in the photo -/
def calculateVisibilityProbability (sarah : Runner) (sam : Runner) (track : Track) : ℚ :=
  sorry

/-- The main theorem to prove -/
theorem runners_visibility_probability :
  let sarah : Runner := ⟨"Sarah", 120, true⟩
  let sam : Runner := ⟨"Sam", 100, false⟩
  let track : Track := ⟨1, 1/3, 45, 15⟩
  calculateVisibilityProbability sarah sam track = 1333/6000 := by
  sorry

end NUMINAMATH_CALUDE_runners_visibility_probability_l3908_390866


namespace NUMINAMATH_CALUDE_max_candy_leftover_l3908_390862

theorem max_candy_leftover (x : ℕ) (h1 : x > 120) : 
  ∃ (q : ℕ), x = 12 * (10 + q) + 11 ∧ 
  ∀ (r : ℕ), r < 11 → ∃ (q' : ℕ), x ≠ 12 * (10 + q') + r :=
by sorry

end NUMINAMATH_CALUDE_max_candy_leftover_l3908_390862


namespace NUMINAMATH_CALUDE_expression_evaluation_l3908_390865

theorem expression_evaluation :
  let x : ℚ := 2
  let y : ℚ := -1/4
  2 * (x - 2*y) - 1/3 * (3*x - 6*y) + 2*x = 13/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3908_390865


namespace NUMINAMATH_CALUDE_sum_of_triangle_perimeters_sum_of_specific_triangle_perimeters_l3908_390883

/-- The sum of perimeters of an infinite series of equilateral triangles -/
theorem sum_of_triangle_perimeters (initial_perimeter : ℝ) : 
  initial_perimeter > 0 →
  (∑' n, initial_perimeter * (1/2)^n) = 2 * initial_perimeter :=
by
  sorry

/-- The specific case where the initial triangle has a perimeter of 180 cm -/
theorem sum_of_specific_triangle_perimeters : 
  (∑' n, 180 * (1/2)^n) = 360 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_triangle_perimeters_sum_of_specific_triangle_perimeters_l3908_390883


namespace NUMINAMATH_CALUDE_unique_solution_system_l3908_390837

theorem unique_solution_system (x y : ℝ) : 
  (x - 2*y = 1 ∧ 3*x + 4*y = 23) ↔ (x = 5 ∧ y = 2) := by
sorry

end NUMINAMATH_CALUDE_unique_solution_system_l3908_390837


namespace NUMINAMATH_CALUDE_equation_solution_l3908_390814

theorem equation_solution :
  ∃ (y₁ y₂ : ℝ), 
    (4 * (-1)^2 + 3 * y₁^2 + 8 * (-1) - 6 * y₁ + 30 = 50) ∧
    (4 * (-1)^2 + 3 * y₂^2 + 8 * (-1) - 6 * y₂ + 30 = 50) ∧
    (y₁ = 1 + Real.sqrt (29/3)) ∧
    (y₂ = 1 - Real.sqrt (29/3)) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3908_390814


namespace NUMINAMATH_CALUDE_min_box_height_l3908_390889

def box_height (x : ℝ) : ℝ := x + 5

def surface_area (x : ℝ) : ℝ := 6 * x^2 + 20 * x

def base_perimeter_plus_height (x : ℝ) : ℝ := 5 * x + 5

theorem min_box_height :
  ∀ x : ℝ,
  x > 0 →
  surface_area x ≥ 150 →
  base_perimeter_plus_height x ≥ 25 →
  box_height x ≥ 9 ∧
  (∃ y : ℝ, y > 0 ∧ surface_area y ≥ 150 ∧ base_perimeter_plus_height y ≥ 25 ∧ box_height y = 9) :=
by sorry

end NUMINAMATH_CALUDE_min_box_height_l3908_390889


namespace NUMINAMATH_CALUDE_circumscribed_circle_area_l3908_390844

theorem circumscribed_circle_area (s : ℝ) (h : s = 12) :
  let triangle_side := s
  let triangle_height := (Real.sqrt 3 / 2) * triangle_side
  let circle_radius := (2 / 3) * triangle_height
  let circle_area := π * circle_radius ^ 2
  circle_area = 48 * π := by sorry

end NUMINAMATH_CALUDE_circumscribed_circle_area_l3908_390844


namespace NUMINAMATH_CALUDE_systematic_sampling_theorem_l3908_390867

theorem systematic_sampling_theorem (total_workers : ℕ) (sample_size : ℕ) (start_num : ℕ) (interval_start : ℕ) (interval_end : ℕ) : 
  total_workers = 840 →
  sample_size = 42 →
  start_num = 21 →
  interval_start = 421 →
  interval_end = 720 →
  (interval_end - interval_start + 1) / (total_workers / sample_size) = 15 :=
by sorry

end NUMINAMATH_CALUDE_systematic_sampling_theorem_l3908_390867


namespace NUMINAMATH_CALUDE_angle_C_is_pi_over_three_max_area_equilateral_l3908_390897

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions of the problem
def satisfiesConditions (t : Triangle) : Prop :=
  t.c * Real.cos t.B + (t.b - 2 * t.a) * Real.cos t.C = 0 ∧ t.c = 2 * Real.sqrt 3

-- Theorem 1: Angle C is π/3
theorem angle_C_is_pi_over_three (t : Triangle) (h : satisfiesConditions t) : 
  t.C = π / 3 := by sorry

-- Theorem 2: Maximum area is 3√3 and occurs when the triangle is equilateral
theorem max_area_equilateral (t : Triangle) (h : satisfiesConditions t) :
  (∃ (area : ℝ), area = 3 * Real.sqrt 3 ∧ 
    area = (1/2) * t.a * t.b * Real.sin t.C ∧
    t.a = t.b ∧ t.b = t.c) := by sorry

end NUMINAMATH_CALUDE_angle_C_is_pi_over_three_max_area_equilateral_l3908_390897


namespace NUMINAMATH_CALUDE_selling_price_is_50_l3908_390816

/-- Represents the manufacturing and sales data for horseshoe sets -/
structure HorseshoeData where
  initial_outlay : ℕ
  cost_per_set : ℕ
  sets_sold : ℕ
  profit : ℕ
  selling_price : ℕ

/-- Calculates the total manufacturing cost -/
def total_manufacturing_cost (data : HorseshoeData) : ℕ :=
  data.initial_outlay + data.cost_per_set * data.sets_sold

/-- Calculates the total revenue -/
def total_revenue (data : HorseshoeData) : ℕ :=
  data.selling_price * data.sets_sold

/-- Theorem stating that the selling price is $50 given the conditions -/
theorem selling_price_is_50 (data : HorseshoeData) 
    (h1 : data.initial_outlay = 10000)
    (h2 : data.cost_per_set = 20)
    (h3 : data.sets_sold = 500)
    (h4 : data.profit = 5000)
    (h5 : data.profit = total_revenue data - total_manufacturing_cost data) :
  data.selling_price = 50 := by
  sorry


end NUMINAMATH_CALUDE_selling_price_is_50_l3908_390816


namespace NUMINAMATH_CALUDE_min_angles_in_circle_l3908_390861

theorem min_angles_in_circle (n : ℕ) (h : n ≥ 3) : ℕ :=
  let S : ℕ → ℕ := fun n =>
    if n % 2 = 1 then
      (n - 1)^2 / 4
    else
      n^2 / 4 - n / 2
  S n

#check min_angles_in_circle

end NUMINAMATH_CALUDE_min_angles_in_circle_l3908_390861


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3908_390846

theorem imaginary_part_of_z (i : ℂ) (h : i^2 = -1) : 
  Complex.im ((1 - i) / (2 * i)) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3908_390846


namespace NUMINAMATH_CALUDE_smallest_solution_quadratic_l3908_390848

theorem smallest_solution_quadratic (x : ℝ) : 
  (3 * x^2 + 18 * x - 90 = x * (x + 10)) → x ≥ -9 :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_quadratic_l3908_390848


namespace NUMINAMATH_CALUDE_evaluate_expression_l3908_390876

theorem evaluate_expression (a : ℝ) : 
  let x := a + 9
  (x - a + 6) = 15 := by
sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3908_390876


namespace NUMINAMATH_CALUDE_tangent_line_at_y_axis_l3908_390818

noncomputable def f (x : ℝ) : ℝ := (x^2 + 4) / (x - 2)

theorem tangent_line_at_y_axis (x y : ℝ) :
  (f 0 = -2) →
  (∀ x, deriv f x = (x^2 - 4*x - 4) / (x - 2)^2) →
  (y = -x - 2) ↔ (y - f 0 = deriv f 0 * (x - 0)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_y_axis_l3908_390818


namespace NUMINAMATH_CALUDE_candy_fraction_of_earnings_l3908_390805

/-- Proves that the fraction of earnings spent on candy is 1/6 -/
theorem candy_fraction_of_earnings : 
  ∀ (candy_bar_price lollipop_price driveway_charge : ℚ)
    (candy_bars lollipops driveways : ℕ),
  candy_bar_price = 3/4 →
  lollipop_price = 1/4 →
  driveway_charge = 3/2 →
  candy_bars = 2 →
  lollipops = 4 →
  driveways = 10 →
  (candy_bar_price * candy_bars + lollipop_price * lollipops) / 
  (driveway_charge * driveways) = 1/6 :=
by sorry

end NUMINAMATH_CALUDE_candy_fraction_of_earnings_l3908_390805


namespace NUMINAMATH_CALUDE_square_of_prime_divisibility_l3908_390840

theorem square_of_prime_divisibility (n p : ℕ) : 
  n > 1 → 
  Nat.Prime p → 
  (n ∣ p - 1) → 
  (p ∣ n^3 - 1) → 
  ∃ k : ℕ, 4*p - 3 = k^2 :=
sorry

end NUMINAMATH_CALUDE_square_of_prime_divisibility_l3908_390840


namespace NUMINAMATH_CALUDE_daughters_age_l3908_390827

/-- Given a mother and daughter whose combined age is 60 years this year,
    and ten years ago the mother's age was seven times the daughter's age,
    prove that the daughter's age this year is 15 years. -/
theorem daughters_age (mother_age daughter_age : ℕ) : 
  mother_age + daughter_age = 60 →
  mother_age - 10 = 7 * (daughter_age - 10) →
  daughter_age = 15 := by
sorry

end NUMINAMATH_CALUDE_daughters_age_l3908_390827


namespace NUMINAMATH_CALUDE_emily_calculation_l3908_390817

def round_to_nearest_ten (x : ℤ) : ℤ :=
  (x + 5) / 10 * 10

theorem emily_calculation : round_to_nearest_ten ((68 + 74 + 59) - 20) = 180 := by
  sorry

end NUMINAMATH_CALUDE_emily_calculation_l3908_390817


namespace NUMINAMATH_CALUDE_min_lcm_a_c_l3908_390881

theorem min_lcm_a_c (a b c : ℕ) (h1 : Nat.lcm a b = 24) (h2 : Nat.lcm b c = 18) :
  ∃ (a' c' : ℕ), Nat.lcm a' c' = 12 ∧ 
    (∀ (x y : ℕ), Nat.lcm x b = 24 → Nat.lcm b y = 18 → Nat.lcm x y ≥ 12) := by
  sorry

end NUMINAMATH_CALUDE_min_lcm_a_c_l3908_390881


namespace NUMINAMATH_CALUDE_boat_speed_theorem_l3908_390855

/-- Represents the speed of a boat in different conditions --/
structure BoatSpeed where
  stillWater : ℝ  -- Speed in still water
  upstream : ℝ    -- Speed against the stream
  downstream : ℝ  -- Speed with the stream

/-- 
Given a man's rowing speed in still water is 4 km/h and 
his speed against the stream is 4 km/h, 
his speed with the stream is also 4 km/h.
-/
theorem boat_speed_theorem (speed : BoatSpeed) 
  (h1 : speed.stillWater = 4)
  (h2 : speed.upstream = 4) :
  speed.downstream = 4 := by
  sorry

#check boat_speed_theorem

end NUMINAMATH_CALUDE_boat_speed_theorem_l3908_390855


namespace NUMINAMATH_CALUDE_cubic_function_property_l3908_390887

/-- Given a cubic function f(x) = mx³ + nx + 1 where mn ≠ 0 and f(-1) = 5, prove that f(1) = 7 -/
theorem cubic_function_property (m n : ℝ) (h1 : m * n ≠ 0) :
  let f := fun x : ℝ => m * x^3 + n * x + 1
  f (-1) = 5 → f 1 = 7 := by
sorry

end NUMINAMATH_CALUDE_cubic_function_property_l3908_390887


namespace NUMINAMATH_CALUDE_isabellas_hair_growth_l3908_390811

/-- Calculates the final length of Isabella's hair after growth -/
def final_hair_length (initial_length growth : ℕ) : ℕ := initial_length + growth

/-- Theorem: Isabella's hair length after growth -/
theorem isabellas_hair_growth (initial_length growth : ℕ) 
  (h1 : initial_length = 18) 
  (h2 : growth = 4) : 
  final_hair_length initial_length growth = 22 := by
  sorry

end NUMINAMATH_CALUDE_isabellas_hair_growth_l3908_390811


namespace NUMINAMATH_CALUDE_crayons_count_l3908_390839

/-- The number of crayons in a box with specific color relationships -/
def total_crayons (blue : ℕ) : ℕ :=
  let red := 4 * blue
  let green := 2 * red
  let yellow := green / 2
  blue + red + green + yellow

/-- Theorem stating that the total number of crayons is 51 when there are 3 blue crayons -/
theorem crayons_count : total_crayons 3 = 51 := by
  sorry

end NUMINAMATH_CALUDE_crayons_count_l3908_390839


namespace NUMINAMATH_CALUDE_divisible_by_35_l3908_390832

theorem divisible_by_35 (n : ℕ) (h : n ≥ 1) :
  ∃ k : ℤ, (3 : ℤ)^(6*n) - (2 : ℤ)^(6*n) = 35 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_35_l3908_390832


namespace NUMINAMATH_CALUDE_ten_player_tournament_decided_in_seven_rounds_l3908_390824

/-- Represents a chess tournament -/
structure ChessTournament where
  num_players : ℕ
  rounds : ℕ

/-- The scoring system for the tournament -/
def score_system : ℕ → ℚ
  | 0 => 0     -- Loss
  | 1 => 1/2   -- Draw
  | _ => 1     -- Win

/-- The maximum possible score for a player after a given number of rounds -/
def max_score (t : ChessTournament) : ℚ := t.rounds

/-- The total points distributed after a given number of rounds -/
def total_points (t : ChessTournament) : ℚ := (t.num_players * t.rounds) / 2

/-- A tournament is decided if the maximum score is greater than the average of the remaining points -/
def is_decided (t : ChessTournament) : Prop :=
  max_score t > (total_points t - max_score t) / (t.num_players - 1)

/-- The main theorem: A 10-player tournament is decided after 7 rounds -/
theorem ten_player_tournament_decided_in_seven_rounds :
  let t : ChessTournament := ⟨10, 7⟩
  is_decided t ∧ ∀ r < 7, ¬is_decided ⟨10, r⟩ := by sorry

end NUMINAMATH_CALUDE_ten_player_tournament_decided_in_seven_rounds_l3908_390824


namespace NUMINAMATH_CALUDE_algebraic_identity_l3908_390808

theorem algebraic_identity (a b : ℝ) : 3 * a^2 * b - 2 * b * a^2 = a^2 * b := by
  sorry

end NUMINAMATH_CALUDE_algebraic_identity_l3908_390808


namespace NUMINAMATH_CALUDE_unique_three_digit_number_l3908_390873

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  h_hundreds : hundreds ≥ 1 ∧ hundreds ≤ 9
  h_tens : tens ≥ 0 ∧ tens ≤ 9
  h_ones : ones ≥ 0 ∧ ones ≤ 9

/-- Sum of all permutations of a three-digit number -/
def sumOfPermutations (n : ThreeDigitNumber) : Nat :=
  100 * (n.hundreds + n.tens + n.ones) +
  10 * (n.hundreds + n.tens + n.ones) +
  (n.hundreds + n.tens + n.ones)

/-- The sum of digits of a three-digit number -/
def sumOfDigits (n : ThreeDigitNumber) : Nat :=
  n.hundreds + n.tens + n.ones

theorem unique_three_digit_number :
  ∀ n : ThreeDigitNumber,
    sumOfPermutations n = 4410 ∧
    Even (sumOfDigits n) →
    n.hundreds = 4 ∧ n.tens = 4 ∧ n.ones = 4 :=
by sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_l3908_390873


namespace NUMINAMATH_CALUDE_line_direction_vector_l3908_390895

def point1 : ℝ × ℝ := (-4, 3)
def point2 : ℝ × ℝ := (2, -2)
def direction_vector (a : ℝ) : ℝ × ℝ := (a, -1)

theorem line_direction_vector (a : ℝ) :
  (∃ k : ℝ, k ≠ 0 ∧ (point2.1 - point1.1, point2.2 - point1.2) = k • direction_vector a) →
  a = 6/5 := by
sorry

end NUMINAMATH_CALUDE_line_direction_vector_l3908_390895


namespace NUMINAMATH_CALUDE_jewelry_thief_l3908_390882

-- Define the set of people
inductive Person : Type
  | A | B | C | D

-- Define the properties
def is_telling_truth (p : Person) : Prop := sorry
def is_thief (p : Person) : Prop := sorry

-- State the theorem
theorem jewelry_thief :
  -- Only one person is telling the truth
  (∃! p : Person, is_telling_truth p) →
  -- Only one person stole the jewelry
  (∃! p : Person, is_thief p) →
  -- A's statement
  (is_telling_truth Person.A ↔ ¬is_thief Person.A) →
  -- B's statement
  (is_telling_truth Person.B ↔ is_thief Person.C) →
  -- C's statement
  (is_telling_truth Person.C ↔ is_thief Person.D) →
  -- D's statement
  (is_telling_truth Person.D ↔ ¬is_thief Person.D) →
  -- Conclusion: A stole the jewelry
  is_thief Person.A :=
by
  sorry


end NUMINAMATH_CALUDE_jewelry_thief_l3908_390882
