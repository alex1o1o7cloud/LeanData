import Mathlib

namespace triangle_side_length_l728_72863

theorem triangle_side_length (A B C : Real) (a b c : Real) :
  A = 2 * Real.pi / 3 →
  b = Real.sqrt 2 →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 →
  a^2 = b^2 + c^2 - 2*b*c*Real.cos A →
  a = Real.sqrt 14 := by
  sorry

end triangle_side_length_l728_72863


namespace first_number_in_ratio_l728_72838

theorem first_number_in_ratio (A B : ℕ+) : 
  (A : ℚ) / (B : ℚ) = 8 / 9 →
  Nat.lcm A B = 432 →
  A = 48 := by
sorry

end first_number_in_ratio_l728_72838


namespace car_trip_average_speed_l728_72895

/-- Calculates the average speed of a car trip given specific conditions -/
theorem car_trip_average_speed
  (total_time : ℝ)
  (initial_time : ℝ)
  (initial_speed : ℝ)
  (remaining_speed : ℝ)
  (h_total_time : total_time = 6)
  (h_initial_time : initial_time = 4)
  (h_initial_speed : initial_speed = 55)
  (h_remaining_speed : remaining_speed = 70) :
  (initial_speed * initial_time + remaining_speed * (total_time - initial_time)) / total_time = 60 :=
by sorry

end car_trip_average_speed_l728_72895


namespace original_price_satisfies_conditions_l728_72805

/-- The original price of a concert ticket -/
def original_price : ℝ := 20

/-- The number of people who received a 40% discount -/
def discount_40_count : ℕ := 10

/-- The number of people who received a 15% discount -/
def discount_15_count : ℕ := 20

/-- The total number of people who bought tickets -/
def total_buyers : ℕ := 45

/-- The total revenue from ticket sales -/
def total_revenue : ℝ := 760

/-- Theorem stating that the original price satisfies the given conditions -/
theorem original_price_satisfies_conditions : 
  discount_40_count * (original_price * 0.6) + 
  discount_15_count * (original_price * 0.85) + 
  (total_buyers - discount_40_count - discount_15_count) * original_price = 
  total_revenue := by sorry

end original_price_satisfies_conditions_l728_72805


namespace triangle_perimeter_l728_72880

theorem triangle_perimeter : ∀ (a b c : ℝ),
  a = 3 ∧ b = 4 ∧ c^2 - 10*c + 16 = 0 ∧
  a + b > c ∧ a + c > b ∧ b + c > a →
  a + b + c = 9 :=
by sorry

end triangle_perimeter_l728_72880


namespace probability_different_suits_pinochle_l728_72829

/-- A pinochle deck of cards -/
structure PinochleDeck :=
  (cards : Nat)
  (suits : Nat)
  (cards_per_suit : Nat)
  (h1 : cards = suits * cards_per_suit)

/-- The probability of drawing three cards of different suits from a pinochle deck -/
def probability_different_suits (deck : PinochleDeck) : Rat :=
  let remaining_after_first := deck.cards - 1
  let suitable_for_second := deck.cards - deck.cards_per_suit
  let remaining_after_second := deck.cards - 2
  let suitable_for_third := deck.cards - 2 * deck.cards_per_suit + 1
  (suitable_for_second : Rat) / remaining_after_first *
  (suitable_for_third : Rat) / remaining_after_second

theorem probability_different_suits_pinochle :
  let deck : PinochleDeck := ⟨48, 4, 12, rfl⟩
  probability_different_suits deck = 414 / 1081 := by
  sorry

end probability_different_suits_pinochle_l728_72829


namespace perpendicular_bisector_covered_l728_72857

def circle_O : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 9}

def perpendicular_bisector (O P : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {q | (q.1 - (O.1 + P.1)/2)^2 + (q.2 - (O.2 + P.2)/2)^2 < ((O.1 - P.1)^2 + (O.2 - P.2)^2) / 4}

def plane_region (m : ℝ) : Set (ℝ × ℝ) := {p | |p.1| + |p.2| ≥ m}

theorem perpendicular_bisector_covered (m : ℝ) :
  (∀ P ∈ circle_O, perpendicular_bisector (0, 0) P ⊆ plane_region m) →
  m ≤ 3/2 := by sorry

end perpendicular_bisector_covered_l728_72857


namespace min_books_proof_l728_72891

def scooter_cost : ℕ := 3000
def earning_per_book : ℕ := 15
def transport_cost_per_book : ℕ := 4

def min_books_to_earn_back : ℕ := 273

theorem min_books_proof :
  min_books_to_earn_back = (
    let profit_per_book := earning_per_book - transport_cost_per_book
    (scooter_cost + profit_per_book - 1) / profit_per_book
  ) :=
by sorry

end min_books_proof_l728_72891


namespace incorrect_conclusions_l728_72806

structure Conclusion where
  correlation : Bool  -- true for positive, false for negative
  coefficient : Real
  constant : Real

def is_correct (c : Conclusion) : Prop :=
  (c.correlation ↔ c.coefficient > 0)

theorem incorrect_conclusions 
  (c1 : Conclusion)
  (c2 : Conclusion)
  (c3 : Conclusion)
  (c4 : Conclusion)
  (h1 : c1 = { correlation := false, coefficient := 2.347, constant := -6.423 })
  (h2 : c2 = { correlation := false, coefficient := -3.476, constant := 5.648 })
  (h3 : c3 = { correlation := true, coefficient := 5.437, constant := 8.493 })
  (h4 : c4 = { correlation := true, coefficient := -4.326, constant := -4.578 }) :
  ¬(is_correct c1) ∧ ¬(is_correct c4) :=
sorry

end incorrect_conclusions_l728_72806


namespace largest_of_five_consecutive_even_integers_l728_72892

/-- Sum of the first n positive integers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Sum of the first n positive even integers -/
def sum_first_n_even (n : ℕ) : ℕ := 2 * sum_first_n n

/-- Sum of five consecutive even integers -/
def sum_five_consecutive_even (n : ℕ) : ℕ := 5 * n - 20

theorem largest_of_five_consecutive_even_integers :
  ∃ (n : ℕ), sum_five_consecutive_even n = sum_first_n_even 30 ∧
             n = 190 ∧
             ∀ (m : ℕ), sum_five_consecutive_even m = sum_first_n_even 30 → m ≤ n :=
by sorry

end largest_of_five_consecutive_even_integers_l728_72892


namespace prime_composite_property_l728_72851

theorem prime_composite_property (n : ℕ) :
  (∀ (a : Fin n → ℕ), Function.Injective a →
    ∃ (i j : Fin n), (a i + a j) / Nat.gcd (a i) (a j) ≥ 2 * n - 1) ∨
  (∃ (a : Fin n → ℕ), Function.Injective a ∧
    ∀ (i j : Fin n), (a i + a j) / Nat.gcd (a i) (a j) < 2 * n - 1) :=
by sorry

end prime_composite_property_l728_72851


namespace divisibility_by_six_l728_72882

theorem divisibility_by_six (n : ℕ) : ∃ k : ℤ, (17 : ℤ)^n - (11 : ℤ)^n = 6 * k := by
  sorry

end divisibility_by_six_l728_72882


namespace polynomial_division_remainder_l728_72827

theorem polynomial_division_remainder : ∃ (q r : Polynomial ℝ),
  (X^4 : Polynomial ℝ) + X^3 + 1 = (X^2 - 2*X + 3) * q + r ∧
  r.degree < (X^2 - 2*X + 3).degree ∧
  r = -3*X - 8 := by
  sorry

end polynomial_division_remainder_l728_72827


namespace work_completion_time_l728_72862

/-- A can do a piece of work in some days. A does the work for 5 days only and leaves the job. 
    B does the remaining work in 3 days. B alone can do the work in 4.5 days. 
    This theorem proves that A alone can do the work in 15 days. -/
theorem work_completion_time (W : ℝ) (A_work_per_day B_work_per_day : ℝ) : 
  (B_work_per_day = W / 4.5) →
  (5 * A_work_per_day + 3 * B_work_per_day = W) →
  (A_work_per_day = W / 15) :=
by sorry

end work_completion_time_l728_72862


namespace launch_vehicle_ratio_l728_72886

/-- Represents a three-stage cylindrical launch vehicle -/
structure LaunchVehicle where
  l₁ : ℝ  -- Length of the first stage
  l₂ : ℝ  -- Length of the second (middle) stage
  l₃ : ℝ  -- Length of the third stage

/-- The conditions for the launch vehicle -/
def LaunchVehicleConditions (v : LaunchVehicle) : Prop :=
  v.l₁ > 0 ∧ v.l₂ > 0 ∧ v.l₃ > 0 ∧
  v.l₂ = (v.l₁ + v.l₃) / 2 ∧
  v.l₂^3 = (6 / 13) * (v.l₁^3 + v.l₃^3)

/-- The theorem stating the ratio of lengths of first and third stages -/
theorem launch_vehicle_ratio (v : LaunchVehicle) 
  (h : LaunchVehicleConditions v) : v.l₁ / v.l₃ = 7 / 5 := by
  sorry

end launch_vehicle_ratio_l728_72886


namespace escalator_walking_speed_l728_72836

theorem escalator_walking_speed 
  (escalator_speed : ℝ) 
  (escalator_length : ℝ) 
  (time_taken : ℝ) 
  (h1 : escalator_speed = 11) 
  (h2 : escalator_length = 140) 
  (h3 : time_taken = 10) : 
  ∃ (walking_speed : ℝ), 
    walking_speed = 3 ∧ 
    escalator_length = (walking_speed + escalator_speed) * time_taken := by
  sorry

end escalator_walking_speed_l728_72836


namespace mary_has_ten_marbles_l728_72802

/-- The number of blue marbles Dan has -/
def dans_marbles : ℕ := 5

/-- The ratio of Mary's marbles to Dan's marbles -/
def mary_to_dan_ratio : ℕ := 2

/-- The number of blue marbles Mary has -/
def marys_marbles : ℕ := mary_to_dan_ratio * dans_marbles

theorem mary_has_ten_marbles : marys_marbles = 10 := by
  sorry

end mary_has_ten_marbles_l728_72802


namespace oxygen_weight_value_l728_72819

/-- The atomic weight of sodium -/
def sodium_weight : ℝ := 22.99

/-- The atomic weight of chlorine -/
def chlorine_weight : ℝ := 35.45

/-- The molecular weight of the compound -/
def compound_weight : ℝ := 74

/-- The atomic weight of oxygen -/
def oxygen_weight : ℝ := compound_weight - (sodium_weight + chlorine_weight)

theorem oxygen_weight_value : oxygen_weight = 15.56 := by
  sorry

end oxygen_weight_value_l728_72819


namespace equation_solution_l728_72849

theorem equation_solution : ∃ x : ℝ, x > 0 ∧ 6 * x^(1/3) - 3 * (x / x^(2/3)) = -1 + 2 * x^(1/3) + 4 ∧ x = 27 := by
  sorry

end equation_solution_l728_72849


namespace stratified_sampling_survey_l728_72881

theorem stratified_sampling_survey (total_households : ℕ) 
                                   (middle_income : ℕ) 
                                   (low_income : ℕ) 
                                   (high_income_selected : ℕ) : 
  total_households = 480 →
  middle_income = 200 →
  low_income = 160 →
  high_income_selected = 6 →
  ∃ (total_selected : ℕ), 
    total_selected * (total_households - middle_income - low_income) = 
    high_income_selected * total_households ∧
    total_selected = 24 :=
by sorry

end stratified_sampling_survey_l728_72881


namespace line_segment_endpoint_l728_72898

/-- The x-coordinate of the end point of the line segment -/
def x : ℝ := 3.4213

/-- The y-coordinate of the end point of the line segment -/
def y : ℝ := 7.8426

/-- The start point of the line segment -/
def start_point : ℝ × ℝ := (2, 2)

/-- The length of the line segment -/
def segment_length : ℝ := 6

theorem line_segment_endpoint :
  x > 0 ∧
  y = 2 * x + 1 ∧
  Real.sqrt ((x - 2)^2 + (y - 2)^2) = segment_length :=
by sorry

end line_segment_endpoint_l728_72898


namespace x_axis_coefficients_l728_72848

/-- A line in 2D space represented by the equation Ax + By + C = 0 -/
structure Line where
  A : ℝ
  B : ℝ
  C : ℝ

/-- Predicate to check if a line is the x-axis -/
def is_x_axis (l : Line) : Prop :=
  ∀ x y : ℝ, l.A * x + l.B * y + l.C = 0 ↔ y = 0

/-- Theorem stating that if a line is the x-axis, then B ≠ 0 and A = C = 0 -/
theorem x_axis_coefficients (l : Line) :
  is_x_axis l → l.B ≠ 0 ∧ l.A = 0 ∧ l.C = 0 :=
sorry

end x_axis_coefficients_l728_72848


namespace cost_for_100km_l728_72854

/-- Represents the cost of a taxi ride in dollars -/
def taxi_cost (distance : ℝ) : ℝ := sorry

/-- The taxi fare is directly proportional to distance traveled -/
axiom fare_proportional (d1 d2 : ℝ) : d1 ≠ 0 → d2 ≠ 0 → 
  taxi_cost d1 / d1 = taxi_cost d2 / d2

/-- Bob's actual ride: 80 km for $160 -/
axiom bob_ride : taxi_cost 80 = 160

/-- The theorem to prove -/
theorem cost_for_100km : taxi_cost 100 = 200 := by sorry

end cost_for_100km_l728_72854


namespace decrease_in_profit_for_given_scenario_l728_72831

/-- Represents the financial data of a textile manufacturing firm -/
structure TextileFirm where
  total_looms : ℕ
  sales_value : ℕ
  manufacturing_expenses : ℕ
  establishment_charges : ℕ

/-- Calculates the decrease in profit when one loom is idle for a month -/
def decrease_in_profit (firm : TextileFirm) : ℕ :=
  let sales_per_loom := firm.sales_value / firm.total_looms
  let expenses_per_loom := firm.manufacturing_expenses / firm.total_looms
  sales_per_loom - expenses_per_loom

/-- Theorem stating the decrease in profit for the given scenario -/
theorem decrease_in_profit_for_given_scenario :
  let firm := TextileFirm.mk 125 500000 150000 75000
  decrease_in_profit firm = 2800 := by
  sorry

#eval decrease_in_profit (TextileFirm.mk 125 500000 150000 75000)

end decrease_in_profit_for_given_scenario_l728_72831


namespace sufficient_not_necessary_condition_m_greater_than_one_sufficient_m_greater_than_one_not_necessary_l728_72830

theorem sufficient_not_necessary_condition (m : ℝ) : 
  (∀ x ≥ 1, 3^(x + m) - 3 * Real.sqrt 3 > 0) ↔ m > 1/2 :=
by sorry

theorem m_greater_than_one_sufficient (m : ℝ) :
  m > 1 → ∀ x ≥ 1, 3^(x + m) - 3 * Real.sqrt 3 > 0 :=
by sorry

theorem m_greater_than_one_not_necessary :
  ∃ m, m ≤ 1 ∧ (∀ x ≥ 1, 3^(x + m) - 3 * Real.sqrt 3 > 0) :=
by sorry

end sufficient_not_necessary_condition_m_greater_than_one_sufficient_m_greater_than_one_not_necessary_l728_72830


namespace four_term_expression_l728_72889

theorem four_term_expression (x : ℝ) : 
  ∃ (a b c d : ℝ), (x^4 - 3)^2 + (x^3 + 3*x)^2 = a*x^8 + b*x^6 + c*x^2 + d ∧ 
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 :=
by sorry

end four_term_expression_l728_72889


namespace corner_sum_l728_72807

/-- Represents a 10x10 array filled with integers from 1 to 100 -/
def CheckerBoard := Fin 10 → Fin 10 → Fin 100

/-- The checkerboard is filled in sequence -/
def is_sequential (board : CheckerBoard) : Prop :=
  ∀ i j, board i j = i.val * 10 + j.val + 1

/-- The sum of the numbers in the four corners of the checkerboard is 202 -/
theorem corner_sum (board : CheckerBoard) (h : is_sequential board) :
  (board 0 0).val + (board 0 9).val + (board 9 0).val + (board 9 9).val = 202 := by
  sorry

end corner_sum_l728_72807


namespace correct_mean_calculation_l728_72800

theorem correct_mean_calculation (n : ℕ) (initial_mean : ℝ) (incorrect_value correct_value : ℝ) :
  n = 25 →
  initial_mean = 190 →
  incorrect_value = 130 →
  correct_value = 165 →
  (n * initial_mean - incorrect_value + correct_value) / n = 191.4 :=
by sorry

end correct_mean_calculation_l728_72800


namespace salary_increase_percentage_l728_72883

theorem salary_increase_percentage 
  (original_salary : ℝ) 
  (current_salary : ℝ) 
  (decrease_percentage : ℝ) 
  (increase_percentage : ℝ) :
  original_salary = 2000 →
  current_salary = 2090 →
  decrease_percentage = 5 →
  current_salary = (1 - decrease_percentage / 100) * (original_salary * (1 + increase_percentage / 100)) →
  increase_percentage = 10 := by
sorry

end salary_increase_percentage_l728_72883


namespace triangle_base_height_proof_l728_72875

theorem triangle_base_height_proof :
  ∀ (base height : ℝ),
    base = height - 4 →
    (1/2) * base * height = 96 →
    base = 12 ∧ height = 16 := by
  sorry

end triangle_base_height_proof_l728_72875


namespace earliest_meeting_time_l728_72876

/-- The time (in minutes) it takes Betty to complete one lap -/
def betty_lap_time : ℕ := 5

/-- The time (in minutes) it takes Charles to complete one lap -/
def charles_lap_time : ℕ := 8

/-- The time (in minutes) it takes Lisa to complete one lap -/
def lisa_lap_time : ℕ := 9

/-- The time (in minutes) Lisa takes as a break after every two laps -/
def lisa_break_time : ℕ := 3

/-- The effective time (in minutes) it takes Lisa to complete one lap, considering her breaks -/
def lisa_effective_lap_time : ℚ := (2 * lisa_lap_time + lisa_break_time) / 2

/-- The start time of the jogging -/
def start_time : String := "6:00 AM"

/-- Proves that the earliest time when all three joggers meet back at the starting point is 1:00 PM -/
theorem earliest_meeting_time : 
  ∃ (t : ℕ), t * betty_lap_time = t * charles_lap_time ∧ 
             t * betty_lap_time = (t : ℚ) * lisa_effective_lap_time ∧
             t * betty_lap_time = 420 := by sorry

end earliest_meeting_time_l728_72876


namespace min_percentage_both_subjects_l728_72855

theorem min_percentage_both_subjects (total : ℝ) (physics_percentage : ℝ) (chemistry_percentage : ℝ)
  (h_physics : physics_percentage = 68)
  (h_chemistry : chemistry_percentage = 72)
  (h_total : total > 0) :
  (physics_percentage + chemistry_percentage - 100 : ℝ) = 40 := by
sorry

end min_percentage_both_subjects_l728_72855


namespace middle_box_label_l728_72842

/-- Represents the possible labels on a box. -/
inductive BoxLabel
  | NoPrize : BoxLabel
  | PrizeInNeighbor : BoxLabel

/-- Represents a row of boxes. -/
structure BoxRow :=
  (size : Nat)
  (labels : Fin size → BoxLabel)
  (prizeLocation : Fin size)

/-- The condition that exactly one statement is true. -/
def exactlyOneTrue (row : BoxRow) : Prop :=
  ∃! i : Fin row.size, 
    (row.labels i = BoxLabel.NoPrize ∧ i ≠ row.prizeLocation) ∨
    (row.labels i = BoxLabel.PrizeInNeighbor ∧ 
      (i.val + 1 = row.prizeLocation.val ∨ i.val = row.prizeLocation.val + 1))

/-- The theorem stating the label on the middle box. -/
theorem middle_box_label (row : BoxRow) 
  (h_size : row.size = 23)
  (h_one_true : exactlyOneTrue row) :
  row.labels ⟨11, by {rw [h_size]; simp}⟩ = BoxLabel.PrizeInNeighbor :=
sorry

end middle_box_label_l728_72842


namespace reciprocal_contraction_l728_72866

open Real

theorem reciprocal_contraction {x₁ x₂ : ℝ} (h₁ : 1 < x₁) (h₂ : x₁ < 2) (h₃ : 1 < x₂) (h₄ : x₂ < 2) (h₅ : x₁ ≠ x₂) :
  |1 / x₁ - 1 / x₂| < |x₂ - x₁| := by
sorry

end reciprocal_contraction_l728_72866


namespace dress_price_is_seven_l728_72815

def total_revenue : ℝ := 69
def num_dresses : ℕ := 7
def num_shirts : ℕ := 4
def price_shirt : ℝ := 5

theorem dress_price_is_seven :
  ∃ (price_dress : ℝ),
    price_dress * num_dresses + price_shirt * num_shirts = total_revenue ∧
    price_dress = 7 := by
  sorry

end dress_price_is_seven_l728_72815


namespace solution_set_when_m_3_range_of_m_for_nonnegative_f_l728_72897

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - m*x + m - 1

-- Statement for Question 1
theorem solution_set_when_m_3 :
  {x : ℝ | f 3 x ≤ 0} = {x : ℝ | 1 ≤ x ∧ x ≤ 2} := by sorry

-- Statement for Question 2
theorem range_of_m_for_nonnegative_f :
  ∀ m : ℝ, (∀ x ∈ Set.Icc 2 4, f m x ≥ -1) ↔ m ≤ 4 := by sorry

end solution_set_when_m_3_range_of_m_for_nonnegative_f_l728_72897


namespace water_consumption_days_l728_72852

/-- Represents the daily water consumption of each sibling -/
structure SiblingWaterConsumption where
  theo : ℕ
  mason : ℕ
  roxy : ℕ

/-- Calculates the number of days it takes for the siblings to drink a given amount of water -/
def calculateDays (consumption : SiblingWaterConsumption) (totalWater : ℕ) : ℕ :=
  totalWater / (consumption.theo + consumption.mason + consumption.roxy)

/-- Theorem stating that it takes 7 days for the siblings to drink 168 cups of water -/
theorem water_consumption_days :
  let consumption : SiblingWaterConsumption := ⟨8, 7, 9⟩
  calculateDays consumption 168 = 7 := by
  sorry

#eval calculateDays ⟨8, 7, 9⟩ 168

end water_consumption_days_l728_72852


namespace not_always_left_to_right_l728_72850

theorem not_always_left_to_right : ∃ (a b c : ℕ), a + b * c ≠ (a + b) * c := by sorry

end not_always_left_to_right_l728_72850


namespace book_price_relationship_l728_72818

/-- Represents a collection of books with linearly increasing prices -/
structure BookCollection where
  basePrice : ℕ
  count : ℕ

/-- Get the price of a book at a specific position -/
def BookCollection.priceAt (bc : BookCollection) (position : ℕ) : ℕ :=
  bc.basePrice + position - 1

/-- The main theorem about the book price relationship -/
theorem book_price_relationship (bc : BookCollection) 
  (h1 : bc.count = 49) : 
  (bc.priceAt 49)^2 = (bc.priceAt 25)^2 + (bc.priceAt 26)^2 := by
  sorry

/-- Helper lemma: The price difference between adjacent books is 1 -/
lemma price_difference (bc : BookCollection) (i : ℕ) 
  (h : i < bc.count) :
  bc.priceAt (i + 1) = bc.priceAt i + 1 := by
  sorry

end book_price_relationship_l728_72818


namespace opposite_areas_equal_l728_72814

/-- Represents a rectangle with an interior point connected to midpoints of its sides --/
structure RectangleWithInteriorPoint where
  /-- The rectangle --/
  rectangle : Set (ℝ × ℝ)
  /-- The interior point --/
  interior_point : ℝ × ℝ
  /-- The midpoints of the rectangle's sides --/
  midpoints : Fin 4 → ℝ × ℝ
  /-- The areas of the four polygons formed --/
  polygon_areas : Fin 4 → ℝ

/-- The sum of opposite polygon areas is equal --/
theorem opposite_areas_equal (r : RectangleWithInteriorPoint) : 
  r.polygon_areas 0 + r.polygon_areas 2 = r.polygon_areas 1 + r.polygon_areas 3 := by
  sorry

end opposite_areas_equal_l728_72814


namespace ratio_of_segments_l728_72890

/-- Given points A, B, C, D, and E on a line in that order, prove the ratio of AC to BD -/
theorem ratio_of_segments (A B C D E : ℝ) : 
  A < B → B < C → C < D → D < E →  -- Points lie on a line in order
  B - A = 3 →                      -- AB = 3
  C - B = 7 →                      -- BC = 7
  E - D = 4 →                      -- DE = 4
  D - A = 17 →                     -- AD = 17
  (C - A) / (D - B) = 5 / 7 := by
sorry

end ratio_of_segments_l728_72890


namespace pencils_left_l728_72867

def initial_pencils : ℕ := 4527
def pencils_to_dorothy : ℕ := 1896
def pencils_to_samuel : ℕ := 754
def pencils_to_alina : ℕ := 307

theorem pencils_left : 
  initial_pencils - (pencils_to_dorothy + pencils_to_samuel + pencils_to_alina) = 1570 := by
  sorry

end pencils_left_l728_72867


namespace largest_coefficient_expansion_l728_72856

theorem largest_coefficient_expansion (x : ℝ) (x_nonzero : x ≠ 0) :
  ∃ (terms : List ℝ), 
    (1/x - 1)^5 = terms.sum ∧ 
    (10/x^3 ∈ terms) ∧
    ∀ (term : ℝ), term ∈ terms → |term| ≤ |10/x^3| :=
by sorry

end largest_coefficient_expansion_l728_72856


namespace composite_sequence_existence_l728_72841

theorem composite_sequence_existence (m : ℕ) (hm : m > 0) :
  ∃ n : ℕ, ∀ k : ℤ, |k| ≤ m → 
    (2^n : ℤ) + k > 0 ∧ ¬(Nat.Prime ((2^n : ℤ) + k).natAbs) :=
by sorry

end composite_sequence_existence_l728_72841


namespace volume_rotational_ellipsoid_l728_72837

/-- The volume of a rotational ellipsoid -/
theorem volume_rotational_ellipsoid (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (∫ y in (-b)..b, π * a^2 * (1 - y^2 / b^2)) = (4 / 3) * π * a^2 * b :=
by sorry

end volume_rotational_ellipsoid_l728_72837


namespace gcd_of_powers_of_97_l728_72871

theorem gcd_of_powers_of_97 :
  Nat.gcd (97^10 + 1) (97^10 + 97^3 + 1) = 1 := by
  sorry

end gcd_of_powers_of_97_l728_72871


namespace sum_of_reciprocals_l728_72843

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x + y = 3 * x * y) : 1 / x + 1 / y = 3 := by
  sorry

end sum_of_reciprocals_l728_72843


namespace rational_sum_zero_l728_72820

theorem rational_sum_zero (x₁ x₂ x₃ x₄ : ℚ) 
  (h₁ : x₁ = x₂ + x₃ + x₄)
  (h₂ : x₂ = x₁ + x₃ + x₄)
  (h₃ : x₃ = x₁ + x₂ + x₄)
  (h₄ : x₄ = x₁ + x₂ + x₃) :
  x₁ = 0 ∧ x₂ = 0 ∧ x₃ = 0 ∧ x₄ = 0 := by
  sorry

end rational_sum_zero_l728_72820


namespace earthquake_aid_calculation_l728_72893

/-- Calculates the total financial aid for a school with high school and junior high students -/
def total_financial_aid (total_students : ℕ) (hs_rate : ℕ) (jhs_rate : ℕ) (hs_exclusion_rate : ℚ) : ℕ :=
  651700

/-- The total financial aid for the given school conditions is 651,700 yuan -/
theorem earthquake_aid_calculation :
  let total_students : ℕ := 1862
  let hs_rate : ℕ := 500
  let jhs_rate : ℕ := 350
  let hs_exclusion_rate : ℚ := 30 / 100
  total_financial_aid total_students hs_rate jhs_rate hs_exclusion_rate = 651700 := by
  sorry

end earthquake_aid_calculation_l728_72893


namespace abs_neg_six_equals_six_l728_72803

theorem abs_neg_six_equals_six : |(-6 : ℤ)| = 6 := by
  sorry

end abs_neg_six_equals_six_l728_72803


namespace solution_set_f_min_m2_n2_l728_72840

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 2| + |x + 1|

-- Theorem for the solution set of f(x) > 7
theorem solution_set_f : 
  {x : ℝ | f x > 7} = {x : ℝ | x > 4 ∨ x < -3} := by sorry

-- Theorem for the minimum value of m^2 + n^2
theorem min_m2_n2 (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (h_min : ∀ x, f x ≥ m + n) (h_eq : m + n = 3) :
  m^2 + n^2 ≥ 9/2 ∧ (m^2 + n^2 = 9/2 ↔ m = 3/2 ∧ n = 3/2) := by sorry

end solution_set_f_min_m2_n2_l728_72840


namespace ball_bounces_to_vertex_l728_72865

/-- The height of the rectangle --/
def rectangle_height : ℕ := 10

/-- The vertical distance covered in one bounce --/
def vertical_distance_per_bounce : ℕ := 2

/-- The number of bounces required to reach the top of the rectangle --/
def number_of_bounces : ℕ := rectangle_height / vertical_distance_per_bounce

theorem ball_bounces_to_vertex :
  number_of_bounces = 5 :=
sorry

end ball_bounces_to_vertex_l728_72865


namespace not_prime_two_pow_plus_one_l728_72874

theorem not_prime_two_pow_plus_one (n m : ℕ) (h1 : m > 1) (h2 : Odd m) (h3 : m ∣ n) :
  ¬ Prime (2^n + 1) := by
sorry

end not_prime_two_pow_plus_one_l728_72874


namespace line_passes_through_fixed_point_l728_72834

/-- The line x + (l-m)y + 3 = 0 always passes through the point (-3, 0) for any real number m -/
theorem line_passes_through_fixed_point :
  ∀ (m : ℝ), (-3 : ℝ) + (1 - m) * (0 : ℝ) + 3 = 0 := by
  sorry

end line_passes_through_fixed_point_l728_72834


namespace hank_lawn_mowing_earnings_l728_72835

/-- Proves that Hank made $50 from mowing lawns given the specified conditions -/
theorem hank_lawn_mowing_earnings :
  let carwash_earnings : ℝ := 100
  let carwash_donation_rate : ℝ := 0.9
  let bake_sale_earnings : ℝ := 80
  let bake_sale_donation_rate : ℝ := 0.75
  let lawn_mowing_donation_rate : ℝ := 1
  let total_donation : ℝ := 200
  let lawn_mowing_earnings : ℝ := 
    total_donation - 
    (carwash_earnings * carwash_donation_rate + 
     bake_sale_earnings * bake_sale_donation_rate)
  lawn_mowing_earnings = 50 := by sorry

end hank_lawn_mowing_earnings_l728_72835


namespace barn_painted_area_l728_72887

/-- Calculates the total area to be painted for a rectangular barn -/
def total_painted_area (width length height : ℝ) : ℝ :=
  2 * (width * height + length * height) + width * length

/-- Theorem stating the total area to be painted for the given barn dimensions -/
theorem barn_painted_area :
  total_painted_area 12 15 6 = 828 := by
  sorry

end barn_painted_area_l728_72887


namespace cubic_curve_triangle_problem_l728_72853

/-- A point on the curve y = x^3 -/
structure CubicPoint where
  x : ℝ
  y : ℝ
  cubic_cond : y = x^3

/-- The problem statement -/
theorem cubic_curve_triangle_problem :
  ∃ (A B C : CubicPoint),
    -- A, B, C are distinct
    A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
    -- BC is parallel to x-axis
    B.y = C.y ∧
    -- Area condition
    |C.x - B.x| * |A.y - B.y| = 2000 ∧
    -- Sum of digits of A's x-coordinate is 1
    (∃ (n : ℕ), A.x = 10 * n + 1 ∧ 0 ≤ n ∧ n < 10) := by
  sorry

end cubic_curve_triangle_problem_l728_72853


namespace range_of_a_l728_72846

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (2*x - 1)/(x - 1) < 0 → x^2 - (2*a + 1)*x + a*(a + 1) ≤ 0) ∧ 
  (∃ x : ℝ, x^2 - (2*a + 1)*x + a*(a + 1) ≤ 0 ∧ (2*x - 1)/(x - 1) ≥ 0) →
  0 ≤ a ∧ a ≤ 1/2 :=
by sorry

end range_of_a_l728_72846


namespace total_shared_amount_l728_72869

/-- Represents the money sharing problem with three people --/
structure MoneySharing where
  ratio1 : ℕ
  ratio2 : ℕ
  ratio3 : ℕ
  share1 : ℕ

/-- Theorem stating that given the conditions, the total shared amount is 195 --/
theorem total_shared_amount (ms : MoneySharing) 
  (h1 : ms.ratio1 = 2)
  (h2 : ms.ratio2 = 3)
  (h3 : ms.ratio3 = 8)
  (h4 : ms.share1 = 30) :
  ms.share1 + (ms.share1 / ms.ratio1 * ms.ratio2) + (ms.share1 / ms.ratio1 * ms.ratio3) = 195 := by
  sorry

#check total_shared_amount

end total_shared_amount_l728_72869


namespace bike_wheel_rotations_l728_72821

theorem bike_wheel_rotations 
  (rotations_per_block : ℕ) 
  (min_blocks : ℕ) 
  (remaining_rotations : ℕ) 
  (h1 : rotations_per_block = 200)
  (h2 : min_blocks = 8)
  (h3 : remaining_rotations = 1000) :
  min_blocks * rotations_per_block - remaining_rotations = 600 :=
by
  sorry

end bike_wheel_rotations_l728_72821


namespace sequence_2017_l728_72873

/-- Property P: If aₚ = aₖ, then aₚ₊₁ = aₖ₊₁ for p, q ∈ ℕ* -/
def PropertyP (a : ℕ → ℕ) : Prop :=
  ∀ p q : ℕ, p ≠ 0 → q ≠ 0 → a p = a q → a (p + 1) = a (q + 1)

/-- The sequence satisfying the given conditions -/
def Sequence (a : ℕ → ℕ) : Prop :=
  PropertyP a ∧
  a 1 = 1 ∧
  a 2 = 2 ∧
  a 3 = 3 ∧
  a 5 = 2 ∧
  a 6 + a 7 + a 8 = 21

theorem sequence_2017 (a : ℕ → ℕ) (h : Sequence a) : a 2017 = 15 := by
  sorry

end sequence_2017_l728_72873


namespace circle_center_correct_l728_72813

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop :=
  9 * x^2 - 18 * x + 9 * y^2 + 36 * y + 44 = 0

/-- The center of a circle -/
def CircleCenter : ℝ × ℝ := (1, -2)

/-- Theorem stating that CircleCenter is the center of the circle defined by CircleEquation -/
theorem circle_center_correct :
  ∀ (x y : ℝ), CircleEquation x y ↔ (x - CircleCenter.1)^2 + (y - CircleCenter.2)^2 = 1/9 :=
by sorry

end circle_center_correct_l728_72813


namespace arithmetic_expression_evaluation_l728_72811

theorem arithmetic_expression_evaluation : 6 + 15 / 3 - 4^2 = -5 := by
  sorry

end arithmetic_expression_evaluation_l728_72811


namespace circulus_vitiosus_characterization_l728_72878

/-- Definition of a logical fallacy --/
def LogicalFallacy : Type := String

/-- Definition of a premise in an argument --/
def Premise : Type := String

/-- Definition of a conclusion in an argument --/
def Conclusion : Type := String

/-- Definition of an argument structure --/
structure Argument where
  premises : List Premise
  conclusion : Conclusion

/-- Definition of circular reasoning (circulus vitiosus) --/
def CirculusVitiosus (arg : Argument) : Prop :=
  arg.conclusion ∈ arg.premises

/-- Theorem stating the characteristic of circulus vitiosus --/
theorem circulus_vitiosus_characterization :
  ∀ (arg : Argument),
    CirculusVitiosus arg ↔
    ∃ (premise : Premise),
      premise ∈ arg.premises ∧ premise = arg.conclusion := by
  sorry

#check circulus_vitiosus_characterization

end circulus_vitiosus_characterization_l728_72878


namespace simplify_fraction_l728_72828

theorem simplify_fraction (x y z : ℚ) (hx : x = 5) (hy : y = 2) (hz : z = 4) :
  (10 * x^2 * y^3 * z) / (15 * x * y^2 * z^2) = 4/3 := by
  sorry

end simplify_fraction_l728_72828


namespace quadratic_roots_equal_roots_condition_l728_72809

-- Part 1: Roots of x^2 - 2x - 8 = 0
theorem quadratic_roots : 
  let f : ℝ → ℝ := λ x => x^2 - 2*x - 8
  ∃ (x₁ x₂ : ℝ), x₁ = 4 ∧ x₂ = -2 ∧ f x₁ = 0 ∧ f x₂ = 0 :=
sorry

-- Part 2: Value of a when roots of x^2 - ax + 1 = 0 are equal
theorem equal_roots_condition :
  let g : ℝ → ℝ → ℝ := λ a x => x^2 - a*x + 1
  ∀ a : ℝ, (∃! x : ℝ, g a x = 0) → (a = 2 ∨ a = -2) :=
sorry

end quadratic_roots_equal_roots_condition_l728_72809


namespace ratio_problem_l728_72896

theorem ratio_problem (second_part : ℝ) (percent : ℝ) (first_part : ℝ) :
  second_part = 5 →
  percent = 180 →
  first_part / second_part = percent / 100 →
  first_part = 9 := by
sorry

end ratio_problem_l728_72896


namespace biff_kenneth_race_l728_72884

/-- Biff and Kenneth's rowboat race problem -/
theorem biff_kenneth_race (race_distance : ℝ) (kenneth_speed : ℝ) (kenneth_extra_distance : ℝ) :
  race_distance = 500 →
  kenneth_speed = 51 →
  kenneth_extra_distance = 10 →
  ∃ (biff_speed : ℝ),
    biff_speed = 50 ∧
    biff_speed * (race_distance + kenneth_extra_distance) / kenneth_speed = race_distance :=
by sorry

end biff_kenneth_race_l728_72884


namespace power_mod_thirteen_l728_72888

theorem power_mod_thirteen : 7^137 % 13 = 11 := by sorry

end power_mod_thirteen_l728_72888


namespace integer_pair_existence_l728_72879

theorem integer_pair_existence : ∃ (x y : ℤ), 
  (x * y + (x + y) = 95) ∧ 
  (x * y - (x + y) = 59) ∧ 
  ((x = 11 ∧ y = 7) ∨ (x = 7 ∧ y = 11)) := by
  sorry

end integer_pair_existence_l728_72879


namespace sqrt_x_div_sqrt_y_l728_72844

theorem sqrt_x_div_sqrt_y (x y : ℝ) (h : (1/3)^2 + (1/4)^2 = ((1/5)^2 + (1/6)^2 + 1/600) * (25*x)/(73*y)) :
  Real.sqrt x / Real.sqrt y = 147/43 := by
  sorry

end sqrt_x_div_sqrt_y_l728_72844


namespace horner_v₃_value_l728_72845

def f (x : ℝ) : ℝ := 7 * x^5 + 5 * x^4 + 3 * x^3 + x^2 + x + 2

def horner_method (x : ℝ) : ℝ × ℝ × ℝ × ℝ := 
  let v₀ : ℝ := 7
  let v₁ : ℝ := v₀ * x + 5
  let v₂ : ℝ := v₁ * x + 3
  let v₃ : ℝ := v₂ * x + 1
  (v₀, v₁, v₂, v₃)

theorem horner_v₃_value :
  (horner_method 2).2.2.2 = 83 := by sorry

end horner_v₃_value_l728_72845


namespace additive_inverse_of_2023_l728_72816

theorem additive_inverse_of_2023 : ∃! x : ℤ, 2023 + x = 0 ∧ x = -2023 := by sorry

end additive_inverse_of_2023_l728_72816


namespace cube_order_l728_72899

theorem cube_order (a b : ℝ) : a > b → a^3 > b^3 := by
  sorry

end cube_order_l728_72899


namespace function_inequality_l728_72823

theorem function_inequality (m n : ℝ) (hm : m < 0) :
  (∃ x : ℝ, x > 0 ∧ Real.log x + m * x + n ≥ 0) →
  n - 1 ≥ Real.log (-m) := by
  sorry

end function_inequality_l728_72823


namespace sum_first_100_triangular_numbers_l728_72812

/-- The n-th triangular number -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The sum of the first n triangular numbers -/
def sum_triangular_numbers (n : ℕ) : ℕ := 
  (Finset.range n).sum (λ i => triangular_number (i + 1))

/-- Theorem: The sum of the first 100 triangular numbers is 171700 -/
theorem sum_first_100_triangular_numbers : 
  sum_triangular_numbers 100 = 171700 := by
  sorry

#eval sum_triangular_numbers 100

end sum_first_100_triangular_numbers_l728_72812


namespace ethan_reading_pages_l728_72808

/-- Represents the number of pages Ethan read on Saturday morning -/
def saturday_morning_pages : ℕ := sorry

/-- Represents the total number of pages in the book -/
def total_pages : ℕ := 360

/-- Represents the number of pages Ethan read on Saturday night -/
def saturday_night_pages : ℕ := 10

/-- Represents the number of pages left to read after Sunday -/
def pages_left : ℕ := 210

/-- The main theorem to prove -/
theorem ethan_reading_pages : 
  saturday_morning_pages = 40 ∧
  (saturday_morning_pages + saturday_night_pages) * 3 = total_pages - pages_left :=
sorry

end ethan_reading_pages_l728_72808


namespace tens_digit_of_23_pow_1987_l728_72861

theorem tens_digit_of_23_pow_1987 : ∃ k : ℕ, 23^1987 ≡ 40 + k [ZMOD 100] :=
sorry

end tens_digit_of_23_pow_1987_l728_72861


namespace square_and_cube_roots_l728_72801

theorem square_and_cube_roots : 
  (∀ x : ℝ, x^2 = 4/9 ↔ x = 2/3 ∨ x = -2/3) ∧ 
  (∀ y : ℝ, y^3 = -64 ↔ y = -4) := by
  sorry

end square_and_cube_roots_l728_72801


namespace stamp_exhibition_problem_l728_72872

theorem stamp_exhibition_problem (x : ℕ) : 
  (∃ (s : ℕ), s = 3 * (s / x) + 24 ∧ s = 4 * (s / x) - 26) → x = 50 := by
  sorry

end stamp_exhibition_problem_l728_72872


namespace union_when_a_eq_2_union_eq_B_iff_l728_72859

-- Define set A
def A : Set ℝ := {x | (x - 1) / (x - 2) ≤ 1 / 2}

-- Define set B parameterized by a
def B (a : ℝ) : Set ℝ := {x | x^2 - (a + 2) * x + 2 * a ≤ 0}

-- Theorem for part (1)
theorem union_when_a_eq_2 : A ∪ B 2 = {x | 0 ≤ x ∧ x ≤ 2} := by sorry

-- Theorem for part (2)
theorem union_eq_B_iff (a : ℝ) : A ∪ B a = B a ↔ a ≤ 0 := by sorry

end union_when_a_eq_2_union_eq_B_iff_l728_72859


namespace parabola_coefficients_l728_72868

/-- A parabola with equation y = ax^2 + bx + c, vertex at (4, 5), and passing through (2, 3) has coefficients (a, b, c) = (-1/2, 4, -3) -/
theorem parabola_coefficients :
  ∀ (a b c : ℝ),
  (∀ x y : ℝ, y = a * x^2 + b * x + c) →
  (5 : ℝ) = a * 4^2 + b * 4 + c →
  (∀ x : ℝ, a * (x - 4)^2 + 5 = a * x^2 + b * x + c) →
  (3 : ℝ) = a * 2^2 + b * 2 + c →
  (a = -1/2 ∧ b = 4 ∧ c = -3) :=
by sorry

end parabola_coefficients_l728_72868


namespace intersection_complement_and_B_l728_72826

def U : Set Int := {-1, 0, 1, 2, 3}
def A : Set Int := {-1, 0}
def B : Set Int := {0, 1, 2}

theorem intersection_complement_and_B : 
  (U \ A) ∩ B = {1, 2} := by sorry

end intersection_complement_and_B_l728_72826


namespace smallest_valid_number_l728_72822

def is_valid (n : ℕ) : Prop :=
  n % 9 = 0 ∧
  n % 2 = 1 ∧
  n % 3 = 1 ∧
  n % 4 = 1 ∧
  n % 5 = 1 ∧
  n % 6 = 1

theorem smallest_valid_number : 
  is_valid 361 ∧ ∀ m : ℕ, m < 361 → ¬is_valid m :=
sorry

end smallest_valid_number_l728_72822


namespace problem_statement_l728_72860

theorem problem_statement (y : ℝ) (hy : y > 0) : 
  ∃ y, ((3/5 * 2500) * (2/7 * ((5/8 * 4000) + (1/4 * 3600) - ((11/20 * 7200) / (3/10 * y))))) = 25000 :=
by sorry

end problem_statement_l728_72860


namespace money_division_l728_72894

theorem money_division (amanda ben carlos total : ℕ) : 
  amanda + ben + carlos = total →
  amanda = 3 * (ben / 5) →
  carlos = 9 * (ben / 5) →
  ben = 50 →
  total = 170 :=
by sorry

end money_division_l728_72894


namespace pizza_fraction_l728_72858

theorem pizza_fraction (pieces_per_day : ℕ) (days : ℕ) (whole_pizzas : ℕ) :
  pieces_per_day = 3 →
  days = 72 →
  whole_pizzas = 27 →
  (1 : ℚ) / (pieces_per_day * days / whole_pizzas) = 1 / 8 := by
  sorry

end pizza_fraction_l728_72858


namespace key_chain_manufacturing_cost_l728_72885

theorem key_chain_manufacturing_cost 
  (P : ℝ) -- Selling price
  (old_profit_percentage : ℝ) -- Old profit percentage
  (new_profit_percentage : ℝ) -- New profit percentage
  (new_manufacturing_cost : ℝ) -- New manufacturing cost
  (h1 : old_profit_percentage = 0.4) -- Old profit was 40%
  (h2 : new_profit_percentage = 0.5) -- New profit is 50%
  (h3 : new_manufacturing_cost = 50) -- New manufacturing cost is $50
  (h4 : P = new_manufacturing_cost / (1 - new_profit_percentage)) -- Selling price calculation
  : (1 - old_profit_percentage) * P = 60 := by
  sorry


end key_chain_manufacturing_cost_l728_72885


namespace money_duration_l728_72817

def mowing_earnings : ℕ := 14
def weed_eating_earnings : ℕ := 26
def weekly_spending : ℕ := 5

theorem money_duration : 
  (mowing_earnings + weed_eating_earnings) / weekly_spending = 8 := by
sorry

end money_duration_l728_72817


namespace triangle_parallel_lines_l728_72870

theorem triangle_parallel_lines (base : ℝ) (h1 : base = 20) : 
  ∀ (line1 line2 : ℝ),
    (line1 / base)^2 = 1/4 →
    (line2 / line1)^2 = 1/3 →
    line2 = 10 * Real.sqrt 3 / 3 :=
by sorry

end triangle_parallel_lines_l728_72870


namespace perpendicular_vectors_x_value_l728_72810

/-- Given two vectors a and b in ℝ², if a is perpendicular to (a - b), then the x-coordinate of b is 9. -/
theorem perpendicular_vectors_x_value (a b : ℝ × ℝ) :
  a = (1, 2) →
  b.2 = -2 →
  (a.1 * (a.1 - b.1) + a.2 * (a.2 - b.2) = 0) →
  b.1 = 9 := by
  sorry

end perpendicular_vectors_x_value_l728_72810


namespace john_kate_penny_difference_l728_72833

theorem john_kate_penny_difference :
  ∀ (john_pennies kate_pennies : ℕ),
    john_pennies = 388 →
    kate_pennies = 223 →
    john_pennies - kate_pennies = 165 := by
  sorry

end john_kate_penny_difference_l728_72833


namespace inequality_solution_minimum_value_and_points_l728_72804

-- Part 1
def solution_set := {x : ℝ | 0 < x ∧ x < 2}

theorem inequality_solution : 
  ∀ x : ℝ, |2*x - 1| < |x| + 1 ↔ x ∈ solution_set :=
sorry

-- Part 2
def constraint (x y z : ℝ) := x^2 + y^2 + z^2 = 4

theorem minimum_value_and_points :
  ∃ (x y z : ℝ), constraint x y z ∧
    (∀ (a b c : ℝ), constraint a b c → x - 2*y + 2*z ≤ a - 2*b + 2*c) ∧
    x - 2*y + 2*z = -6 ∧
    x = -2/3 ∧ y = 4/3 ∧ z = -4/3 :=
sorry

end inequality_solution_minimum_value_and_points_l728_72804


namespace prime_power_triples_l728_72877

theorem prime_power_triples (p : ℕ) (x y : ℕ+) :
  (Nat.Prime p ∧
   ∃ (a b : ℕ), x^(p-1) + y = p^a ∧ x + y^(p-1) = p^b) →
  ((p = 3 ∧ ((x = 2 ∧ y = 5) ∨ (x = 5 ∧ y = 2))) ∨
   (p = 2 ∧ ∃ (k : ℕ), 0 < x.val ∧ x.val < 2^k ∧ y = ⟨2^k - x.val, sorry⟩)) :=
by sorry

end prime_power_triples_l728_72877


namespace hilt_bread_flour_l728_72824

/-- The amount of flour needed for baking bread -/
def flour_for_bread (loaves : ℕ) (flour_per_loaf : ℚ) : ℚ :=
  loaves * flour_per_loaf

/-- Theorem: Mrs. Hilt needs 5 cups of flour to bake 2 loaves of bread -/
theorem hilt_bread_flour :
  flour_for_bread 2 (5/2) = 5 := by
  sorry

end hilt_bread_flour_l728_72824


namespace hyperbola_eccentricity_l728_72832

/-- The eccentricity of a hyperbola with equation x^2/a^2 - y^2/b^2 = 1,
    where one of its asymptotes passes through the point (3, -4), is 5/3. -/
theorem hyperbola_eccentricity (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 → c^2 = a^2 + b^2) →
  (∃ k : ℝ, k * 3 = a ∧ k * (-4) = b) →
  c / a = 5 / 3 := by sorry

end hyperbola_eccentricity_l728_72832


namespace expr_is_symmetrical_l728_72839

/-- Definition of a symmetrical expression -/
def is_symmetrical (f : ℝ → ℝ → ℝ) : Prop :=
  ∀ a b, f a b = f b a

/-- The expression we want to prove is symmetrical -/
def expr (a b : ℝ) : ℝ := 4*a^2 + 4*b^2 - 4*a*b

/-- Theorem: The expression 4a^2 + 4b^2 - 4ab is symmetrical -/
theorem expr_is_symmetrical : is_symmetrical expr := by sorry

end expr_is_symmetrical_l728_72839


namespace investment_rate_proof_l728_72825

def total_investment : ℝ := 12000
def first_investment : ℝ := 5000
def second_investment : ℝ := 4000
def first_rate : ℝ := 0.03
def second_rate : ℝ := 0.045
def desired_income : ℝ := 600

theorem investment_rate_proof :
  let remaining_investment := total_investment - first_investment - second_investment
  let income_from_first := first_investment * first_rate
  let income_from_second := second_investment * second_rate
  let remaining_income := desired_income - income_from_first - income_from_second
  remaining_income / remaining_investment = 0.09 := by sorry

end investment_rate_proof_l728_72825


namespace focus_of_specific_parabola_l728_72864

/-- A parabola with equation y = (x - h)^2 + k, where (h, k) is the vertex. -/
structure Parabola where
  h : ℝ
  k : ℝ

/-- The focus of a parabola. -/
def focus (p : Parabola) : ℝ × ℝ := sorry

/-- Theorem: The focus of the parabola y = (x - 3)^2 is at (3, 1/8). -/
theorem focus_of_specific_parabola :
  let p : Parabola := { h := 3, k := 0 }
  focus p = (3, 1/8) := by sorry

end focus_of_specific_parabola_l728_72864


namespace limit_fraction_sequence_l728_72847

theorem limit_fraction_sequence : 
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |((n : ℝ) + 20) / (3 * n + 13) - 1/3| < ε :=
sorry

end limit_fraction_sequence_l728_72847
