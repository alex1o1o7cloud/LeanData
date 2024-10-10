import Mathlib

namespace correct_average_l641_64161

theorem correct_average (n : ℕ) (incorrect_avg : ℚ) (incorrect_num correct_num : ℚ) :
  n = 10 ∧ 
  incorrect_avg = 46 ∧ 
  incorrect_num = 25 ∧ 
  correct_num = 65 →
  (n : ℚ) * incorrect_avg + (correct_num - incorrect_num) = n * 50 := by
sorry

end correct_average_l641_64161


namespace matrix_equation_holds_l641_64168

def M : Matrix (Fin 2) (Fin 2) ℝ := !![1, 2; 2, 4]

theorem matrix_equation_holds :
  M^3 - 2 • M^2 + (-12) • M = 3 • !![1, 2; 2, 4] := by sorry

end matrix_equation_holds_l641_64168


namespace car_overtake_distance_l641_64135

/-- Proves that the initial distance between two cars is 10 miles given their speeds and overtaking time -/
theorem car_overtake_distance (speed_a speed_b time_to_overtake : ℝ) 
  (h1 : speed_a = 58)
  (h2 : speed_b = 50)
  (h3 : time_to_overtake = 2.25)
  (h4 : (speed_a - speed_b) * time_to_overtake = initial_distance + 8) :
  initial_distance = 10 := by sorry


end car_overtake_distance_l641_64135


namespace smallest_prime_divisor_of_sum_l641_64103

theorem smallest_prime_divisor_of_sum : 
  ∃ (p : ℕ), p.Prime ∧ p ∣ (7^14 + 11^15) ∧ ∀ (q : ℕ), q.Prime → q ∣ (7^14 + 11^15) → p ≤ q :=
by sorry

end smallest_prime_divisor_of_sum_l641_64103


namespace prob_selected_twice_correct_most_likely_selected_correct_l641_64110

/-- Represents the total number of students --/
def total_students : ℕ := 60

/-- Represents the number of students selected in each round --/
def selected_per_round : ℕ := 45

/-- Probability of a student being selected in both rounds --/
def prob_selected_twice : ℚ := 9 / 16

/-- The most likely number of students selected in both rounds --/
def most_likely_selected : ℕ := 34

/-- Function to calculate the probability of being selected in both rounds --/
def calculate_prob_selected_twice : ℚ :=
  (Nat.choose (total_students - 1) (selected_per_round - 1) / Nat.choose total_students selected_per_round) ^ 2

/-- Function to calculate the probability of exactly n students being selected in both rounds --/
def prob_n_selected (n : ℕ) : ℚ :=
  (Nat.choose total_students n * Nat.choose (total_students - n) (selected_per_round - n) * Nat.choose (selected_per_round - n) (selected_per_round - n)) /
  (Nat.choose total_students selected_per_round * Nat.choose total_students selected_per_round)

theorem prob_selected_twice_correct :
  calculate_prob_selected_twice = prob_selected_twice :=
sorry

theorem most_likely_selected_correct :
  ∀ n, 30 ≤ n ∧ n ≤ 45 → prob_n_selected most_likely_selected ≥ prob_n_selected n :=
sorry

end prob_selected_twice_correct_most_likely_selected_correct_l641_64110


namespace smallest_block_volume_l641_64146

theorem smallest_block_volume (a b c : ℕ) (h : (a - 1) * (b - 1) * (c - 1) = 143) :
  a * b * c ≥ 336 :=
by sorry

end smallest_block_volume_l641_64146


namespace savings_calculation_l641_64104

/-- Given a person's income and expenditure ratio, and their total income, calculate their savings. -/
def calculate_savings (income_ratio : ℕ) (expenditure_ratio : ℕ) (total_income : ℕ) : ℕ :=
  let total_ratio := income_ratio + expenditure_ratio
  let expenditure := (expenditure_ratio * total_income) / total_ratio
  total_income - expenditure

/-- Theorem stating that given the specific income-expenditure ratio and total income, 
    the savings amount to 7000. -/
theorem savings_calculation :
  calculate_savings 3 2 21000 = 7000 := by
  sorry

end savings_calculation_l641_64104


namespace louise_yellow_pencils_l641_64170

/-- Proves the number of yellow pencils Louise has --/
theorem louise_yellow_pencils :
  let box_capacity : ℕ := 20
  let red_pencils : ℕ := 20
  let blue_pencils : ℕ := 2 * red_pencils
  let green_pencils : ℕ := red_pencils + blue_pencils
  let total_boxes : ℕ := 8
  let total_capacity : ℕ := total_boxes * box_capacity
  let other_pencils : ℕ := red_pencils + blue_pencils + green_pencils
  let yellow_pencils : ℕ := total_capacity - other_pencils
  yellow_pencils = 40 := by
  sorry

end louise_yellow_pencils_l641_64170


namespace roja_speed_calculation_l641_64185

/-- Roja's speed in km/hr -/
def rojaSpeed : ℝ := 8

/-- Pooja's speed in km/hr -/
def poojaSpeed : ℝ := 3

/-- Time elapsed in hours -/
def timeElapsed : ℝ := 4

/-- Distance between Roja and Pooja after the elapsed time in km -/
def distanceBetween : ℝ := 44

theorem roja_speed_calculation :
  rojaSpeed = 8 ∧
  poojaSpeed = 3 ∧
  timeElapsed = 4 ∧
  distanceBetween = 44 ∧
  distanceBetween = (rojaSpeed + poojaSpeed) * timeElapsed :=
by sorry

end roja_speed_calculation_l641_64185


namespace complex_power_of_four_l641_64156

theorem complex_power_of_four : 
  (3 * (Complex.cos (30 * π / 180) + Complex.I * Complex.sin (30 * π / 180)))^4 = 
  Complex.mk (-40.5) (40.5 * Real.sqrt 3) :=
by sorry

end complex_power_of_four_l641_64156


namespace nested_expression_value_l641_64133

theorem nested_expression_value : 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4)))) = 1364 := by
  sorry

end nested_expression_value_l641_64133


namespace negative_pi_less_than_negative_three_l641_64143

theorem negative_pi_less_than_negative_three :
  π > 3 → -π < -3 := by
  sorry

end negative_pi_less_than_negative_three_l641_64143


namespace park_trees_l641_64199

theorem park_trees (pine_percentage : ℝ) (non_pine_count : ℕ) 
  (h1 : pine_percentage = 0.7)
  (h2 : non_pine_count = 105) : 
  ∃ (total_trees : ℕ), 
    (↑non_pine_count : ℝ) = (1 - pine_percentage) * (total_trees : ℝ) ∧ 
    total_trees = 350 :=
by sorry

end park_trees_l641_64199


namespace value_of_T_l641_64147

-- Define the variables
variable (M A T H E : ℤ)

-- Define the conditions
def condition_H : H = 8 := by sorry
def condition_MATH : M + A + T + H = 47 := by sorry
def condition_MEET : M + E + E + T = 62 := by sorry
def condition_TEAM : T + E + A + M = 58 := by sorry

-- Theorem to prove
theorem value_of_T : T = 9 := by sorry

end value_of_T_l641_64147


namespace triangle_problem_l641_64180

theorem triangle_problem (a b c A B C : Real) (h1 : (2 * a - b) / c = Real.cos B / Real.cos C) :
  let f := fun x => 2 * Real.sin x * Real.cos x * Real.cos C + 2 * Real.sin x * Real.sin x * Real.sin C - Real.sqrt 3 / 2
  C = π / 3 ∧ Set.Icc (f 0) (f (π / 2)) = Set.Icc (-(Real.sqrt 3) / 2) 1 := by
  sorry

end triangle_problem_l641_64180


namespace annes_walking_time_l641_64177

/-- Anne's walking problem -/
theorem annes_walking_time (distance : ℝ) (speed : ℝ) (time : ℝ) 
  (h1 : distance = 6)
  (h2 : speed = 2)
  (h3 : distance = speed * time) : 
  time = 3 := by
  sorry

end annes_walking_time_l641_64177


namespace cat_weight_sum_l641_64155

/-- The combined weight of three cats -/
def combined_weight (w1 w2 w3 : ℕ) : ℕ := w1 + w2 + w3

/-- Theorem: The combined weight of cats weighing 2, 7, and 4 pounds is 13 pounds -/
theorem cat_weight_sum : combined_weight 2 7 4 = 13 := by
  sorry

end cat_weight_sum_l641_64155


namespace initial_money_calculation_l641_64181

theorem initial_money_calculation (initial_money : ℚ) : 
  (initial_money * (1 - 1/3) * (1 - 1/5) * (1 - 1/4) = 100) → 
  initial_money = 250 := by
sorry

end initial_money_calculation_l641_64181


namespace cone_height_ratio_l641_64139

theorem cone_height_ratio (original_circumference : ℝ) (original_height : ℝ) (new_volume : ℝ) :
  original_circumference = 20 * Real.pi →
  original_height = 40 →
  new_volume = 800 * Real.pi →
  ∃ (new_height : ℝ),
    (1 / 3) * Real.pi * (original_circumference / (2 * Real.pi))^2 * new_height = new_volume ∧
    new_height / original_height = 3 / 5 :=
by sorry

end cone_height_ratio_l641_64139


namespace tire_price_proof_l641_64101

theorem tire_price_proof : 
  let fourth_tire_discount : ℝ := 0.75
  let total_cost : ℝ := 270
  let regular_price : ℝ := 72
  (3 * regular_price + fourth_tire_discount * regular_price = total_cost) →
  regular_price = 72 :=
by
  sorry

end tire_price_proof_l641_64101


namespace problem_one_problem_two_l641_64107

-- Problem 1
theorem problem_one (x n : ℝ) (h : x^n = 2) : 
  (3*x^n)^2 - 4*(x^2)^n = 20 := by sorry

-- Problem 2
theorem problem_two (x y n : ℝ) (h1 : x = 2^n - 1) (h2 : y = 3 + 8^n) :
  y = 3 + (x + 1)^3 := by sorry

end problem_one_problem_two_l641_64107


namespace problem1_l641_64117

theorem problem1 (a b : ℝ) (h1 : a ≥ b) (h2 : b > 0) :
  2 * a^3 - b^3 ≥ 2 * a * b^2 - a^2 * b :=
by sorry

end problem1_l641_64117


namespace max_two_greater_than_half_l641_64111

theorem max_two_greater_than_half (α β γ : Real) 
  (h_acute_α : 0 < α ∧ α < π / 2)
  (h_acute_β : 0 < β ∧ β < π / 2)
  (h_acute_γ : 0 < γ ∧ γ < π / 2)
  (h_distinct : α ≠ β ∧ β ≠ γ ∧ α ≠ γ) :
  let values := [Real.sin α * Real.cos β, Real.sin β * Real.cos γ, Real.sin γ * Real.cos α]
  (values.filter (λ x => x > 1/2)).length ≤ 2 := by
sorry

end max_two_greater_than_half_l641_64111


namespace work_group_size_work_group_size_is_9_l641_64159

theorem work_group_size (days1 : ℕ) (days2 : ℕ) (men2 : ℕ) : ℕ :=
  let work_constant := men2 * days2
  let men1 := work_constant / days1
  men1

theorem work_group_size_is_9 :
  work_group_size 80 36 20 = 9 := by
  sorry

end work_group_size_work_group_size_is_9_l641_64159


namespace problem_solution_l641_64124

theorem problem_solution (x : ℝ) : (0.5 * x - (1/3) * x = 110) → x = 660 := by
  sorry

end problem_solution_l641_64124


namespace line_l_passes_through_fixed_point_line_l_not_in_fourth_quadrant_min_area_of_triangle_AOB_l641_64119

-- Define the line l: kx - y + 1 + 2k = 0
def line_l (k : ℝ) (x y : ℝ) : Prop := k * x - y + 1 + 2 * k = 0

-- Define the fixed point
def fixed_point : ℝ × ℝ := (-2, 1)

-- Define the fourth quadrant
def fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

-- Define the negative half of x-axis and positive half of y-axis
def neg_x_axis (x : ℝ) : Prop := x < 0
def pos_y_axis (y : ℝ) : Prop := y > 0

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Define the area of a triangle given three points
def triangle_area (a b c : ℝ × ℝ) : ℝ := sorry

-- Theorem statements
theorem line_l_passes_through_fixed_point :
  ∀ k : ℝ, line_l k (fixed_point.1) (fixed_point.2) := by sorry

theorem line_l_not_in_fourth_quadrant :
  ∀ k x y : ℝ, line_l k x y → ¬(fourth_quadrant x y) → k ≥ 0 := by sorry

theorem min_area_of_triangle_AOB :
  ∀ k x y : ℝ,
  line_l k x y →
  neg_x_axis x →
  pos_y_axis y →
  let a := (x, 0)
  let b := (0, y)
  triangle_area a origin b ≥ 4 ∧
  (triangle_area a origin b = 4 ↔ line_l (1/2) x y) := by sorry

end line_l_passes_through_fixed_point_line_l_not_in_fourth_quadrant_min_area_of_triangle_AOB_l641_64119


namespace lasagna_pieces_needed_l641_64125

/-- Represents the amount of lasagna each person eats relative to Manny's portion --/
structure LasagnaPortion where
  manny : ℚ
  lisa : ℚ
  raphael : ℚ
  aaron : ℚ
  kai : ℚ
  priya : ℚ

/-- Calculates the total number of lasagna pieces needed --/
def totalPieces (portions : LasagnaPortion) : ℚ :=
  portions.manny + portions.lisa + portions.kai + portions.priya

/-- The specific portions for each person based on the problem conditions --/
def givenPortions : LasagnaPortion :=
  { manny := 1
  , lisa := 2 + 1/2
  , raphael := 1/2
  , aaron := 0
  , kai := 2
  , priya := 1/3 }

theorem lasagna_pieces_needed : 
  ∃ n : ℕ, n > 0 ∧ n = ⌈totalPieces givenPortions⌉ ∧ n = 6 := by
  sorry

end lasagna_pieces_needed_l641_64125


namespace white_marbles_in_bag_a_l641_64100

/-- Represents the number of marbles of each color in Bag A -/
structure BagA where
  red : ℕ
  white : ℕ
  blue : ℕ

/-- Represents the ratios of marbles in Bag A -/
structure BagARatios where
  red_to_white : ℚ
  white_to_blue : ℚ

/-- Theorem stating that if Bag A contains 5 red marbles, it must contain 15 white marbles -/
theorem white_marbles_in_bag_a 
  (bag : BagA) 
  (ratios : BagARatios) 
  (h1 : ratios.red_to_white = 1 / 3) 
  (h2 : ratios.white_to_blue = 2 / 3) 
  (h3 : bag.red = 5) : 
  bag.white = 15 := by
  sorry

#check white_marbles_in_bag_a

end white_marbles_in_bag_a_l641_64100


namespace gas_cost_per_gallon_l641_64197

/-- Proves that the cost of gas per gallon is $4, given the specified conditions. -/
theorem gas_cost_per_gallon (miles_per_gallon : ℝ) (total_miles : ℝ) (total_cost : ℝ) :
  miles_per_gallon = 32 →
  total_miles = 304 →
  total_cost = 38 →
  total_cost / (total_miles / miles_per_gallon) = 4 := by
  sorry

#check gas_cost_per_gallon

end gas_cost_per_gallon_l641_64197


namespace least_three_digit_multiple_of_seven_l641_64108

theorem least_three_digit_multiple_of_seven : 
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 7 ∣ n → 105 ≤ n :=
by sorry

end least_three_digit_multiple_of_seven_l641_64108


namespace seating_arrangements_eq_360_l641_64113

/-- A rectangular table with 6 seats -/
structure RectangularTable :=
  (total_seats : ℕ)
  (longer_side_seats : ℕ)
  (shorter_side_seats : ℕ)
  (h_total : total_seats = 2 * longer_side_seats + 2 * shorter_side_seats)
  (h_longer : longer_side_seats = 2)
  (h_shorter : shorter_side_seats = 1)

/-- The number of ways to seat 5 persons at a rectangular table with 6 seats -/
def seating_arrangements (table : RectangularTable) (persons : ℕ) : ℕ := 
  3 * Nat.factorial (table.total_seats - 1)

/-- Theorem stating that the number of seating arrangements for 5 persons
    at the specified rectangular table is 360 -/
theorem seating_arrangements_eq_360 (table : RectangularTable) :
  seating_arrangements table 5 = 360 := by
  sorry

#eval seating_arrangements ⟨6, 2, 1, rfl, rfl, rfl⟩ 5

end seating_arrangements_eq_360_l641_64113


namespace jills_speed_l641_64129

/-- Proves that Jill's speed was 8 km/h given the conditions of the problem -/
theorem jills_speed (jack_distance1 jack_distance2 jack_speed1 jack_speed2 : ℝ)
  (h1 : jack_distance1 = 12)
  (h2 : jack_distance2 = 12)
  (h3 : jack_speed1 = 12)
  (h4 : jack_speed2 = 6)
  (jill_distance jill_time : ℝ)
  (h5 : jill_distance = jack_distance1 + jack_distance2)
  (h6 : jill_time = jack_distance1 / jack_speed1 + jack_distance2 / jack_speed2) :
  jill_distance / jill_time = 8 :=
sorry

end jills_speed_l641_64129


namespace alex_candles_left_l641_64162

theorem alex_candles_left (initial_candles used_candles : ℕ) 
  (h1 : initial_candles = 44)
  (h2 : used_candles = 32) :
  initial_candles - used_candles = 12 := by
  sorry

end alex_candles_left_l641_64162


namespace distance_to_bus_stand_l641_64166

/-- The distance to the bus stand in kilometers -/
def distance : ℝ := 13.5

/-- The time at which the bus arrives in hours -/
def bus_arrival_time : ℝ := 2.5

/-- Theorem stating that the distance to the bus stand is 13.5 km -/
theorem distance_to_bus_stand :
  (distance = 5 * (bus_arrival_time + 0.2)) ∧
  (distance = 6 * (bus_arrival_time - 0.25)) :=
by sorry

end distance_to_bus_stand_l641_64166


namespace vector_problem_l641_64152

def vector_a : Fin 2 → ℝ := ![3, -4]
def vector_b (x : ℝ) : Fin 2 → ℝ := ![2, x]
def vector_c (y : ℝ) : Fin 2 → ℝ := ![2, y]

def parallel (u v : Fin 2 → ℝ) : Prop :=
  ∃ k : ℝ, ∀ i, v i = k * u i

def perpendicular (u v : Fin 2 → ℝ) : Prop :=
  (u 0) * (v 0) + (u 1) * (v 1) = 0

theorem vector_problem (x y : ℝ) 
  (h1 : parallel vector_a (vector_b x))
  (h2 : perpendicular vector_a (vector_c y)) :
  (x = -8/3 ∧ y = 3/2) ∧ 
  perpendicular (vector_b (-8/3)) (vector_c (3/2)) := by
  sorry

end vector_problem_l641_64152


namespace prob_at_least_one_second_class_l641_64132

/-- The probability of selecting at least one second-class item when randomly choosing 3 items
    from a set of 10 items, where 6 are first-class and 4 are second-class. -/
theorem prob_at_least_one_second_class (total : Nat) (first_class : Nat) (second_class : Nat) (selected : Nat)
    (h1 : total = 10)
    (h2 : first_class = 6)
    (h3 : second_class = 4)
    (h4 : selected = 3)
    (h5 : total = first_class + second_class) :
    (1 : ℚ) - (Nat.choose first_class selected : ℚ) / (Nat.choose total selected : ℚ) = 5/6 := by
  sorry

end prob_at_least_one_second_class_l641_64132


namespace households_with_both_count_l641_64140

/-- Represents the distribution of car and bike ownership in a neighborhood -/
structure Neighborhood where
  total : ℕ
  neither : ℕ
  with_car : ℕ
  only_bike : ℕ

/-- Calculates the number of households with both a car and a bike -/
def households_with_both (n : Neighborhood) : ℕ :=
  n.with_car - n.only_bike

/-- Theorem stating the number of households with both a car and a bike -/
theorem households_with_both_count (n : Neighborhood) 
  (h1 : n.total = 90)
  (h2 : n.neither = 11)
  (h3 : n.with_car = 44)
  (h4 : n.only_bike = 35)
  (h5 : n.total = n.neither + n.with_car + n.only_bike) :
  households_with_both n = 9 := by
  sorry

#eval households_with_both { total := 90, neither := 11, with_car := 44, only_bike := 35 }

end households_with_both_count_l641_64140


namespace impossible_table_filling_l641_64167

theorem impossible_table_filling (n : ℕ) (h : n ≥ 3) :
  ¬ ∃ (table : Fin n → Fin (n + 3) → ℕ),
    (∀ i j, table i j ∈ Finset.range (n * (n + 3) + 1)) ∧
    (∀ i j₁ j₂, j₁ ≠ j₂ → table i j₁ ≠ table i j₂) ∧
    (∀ i, ∃ j₁ j₂ j₃, j₁ ≠ j₂ ∧ j₁ ≠ j₃ ∧ j₂ ≠ j₃ ∧
      table i j₁ * table i j₂ = table i j₃) :=
by sorry

end impossible_table_filling_l641_64167


namespace increasing_quadratic_coefficient_range_l641_64136

def f (m : ℝ) (x : ℝ) := 3 * x^2 + m * x + 2

theorem increasing_quadratic_coefficient_range (m : ℝ) :
  (∀ x ≥ 1, ∀ y > x, f m y > f m x) →
  m ≥ -6 :=
by sorry

end increasing_quadratic_coefficient_range_l641_64136


namespace prob_at_least_one_odd_is_nine_tenths_l641_64126

def numbers : Finset ℕ := {1, 2, 3, 4, 5}

def is_odd (n : ℕ) : Bool := n % 2 = 1

def prob_at_least_one_odd : ℚ :=
  1 - (Finset.filter (λ n => ¬(is_odd n)) numbers).card.choose 2 / numbers.card.choose 2

theorem prob_at_least_one_odd_is_nine_tenths :
  prob_at_least_one_odd = 9/10 := by sorry

end prob_at_least_one_odd_is_nine_tenths_l641_64126


namespace range_of_a_l641_64115

-- Define the condition that x^2 > 1 is necessary but not sufficient for x < a
def necessary_not_sufficient (a : ℝ) : Prop :=
  (∀ x : ℝ, x < a → x^2 > 1) ∧ 
  (∃ x : ℝ, x^2 > 1 ∧ x ≥ a)

-- Theorem stating the range of values for a
theorem range_of_a (a : ℝ) : 
  necessary_not_sufficient a ↔ a ≤ -1 :=
sorry

end range_of_a_l641_64115


namespace smallest_a_value_l641_64150

theorem smallest_a_value (a b : ℝ) : 
  a ≥ 0 → b ≥ 0 → 
  (∀ x : ℤ, Real.sin (a * (x : ℝ) + b) = Real.sin (37 * (x : ℝ))) → 
  ∀ a' ≥ 0, (∀ x : ℤ, Real.sin (a' * (x : ℝ) + b) = Real.sin (37 * (x : ℝ))) → 
  a' ≥ 37 := by
sorry

end smallest_a_value_l641_64150


namespace max_value_of_f_l641_64151

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x - 3

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), M = 5 ∧ ∀ x ∈ Set.Icc (-2 : ℝ) 4, f x ≤ M :=
sorry

end max_value_of_f_l641_64151


namespace fuchsia_purple_or_blue_count_l641_64158

/-- Represents the survey results about fuchsia color perception --/
structure FuchsiaSurvey where
  total : ℕ
  like_pink : ℕ
  like_pink_and_purple : ℕ
  like_none : ℕ
  like_all : ℕ

/-- Calculates the number of people who believe fuchsia is "like purple" or "like blue" --/
def purple_or_blue (survey : FuchsiaSurvey) : ℕ :=
  survey.total - survey.like_none - (survey.like_pink - survey.like_pink_and_purple)

/-- Theorem stating that for the given survey results, 64 people believe fuchsia is "like purple" or "like blue" --/
theorem fuchsia_purple_or_blue_count :
  let survey : FuchsiaSurvey := {
    total := 150,
    like_pink := 90,
    like_pink_and_purple := 47,
    like_none := 23,
    like_all := 20
  }
  purple_or_blue survey = 64 := by
  sorry

end fuchsia_purple_or_blue_count_l641_64158


namespace negation_of_all_cuboids_are_prisms_l641_64198

-- Define the universe of shapes
variable {Shape : Type}

-- Define properties
variable (isCuboid : Shape → Prop)
variable (isPrism : Shape → Prop)
variable (hasLateralFaces : Shape → ℕ → Prop)

-- The theorem
theorem negation_of_all_cuboids_are_prisms :
  (¬ ∀ x : Shape, isCuboid x → (isPrism x ∧ hasLateralFaces x 4)) ↔ 
  (∃ x : Shape, isCuboid x ∧ ¬(isPrism x ∧ hasLateralFaces x 4)) := by
sorry

end negation_of_all_cuboids_are_prisms_l641_64198


namespace solution_set_characterization_l641_64149

/-- An odd function satisfying certain conditions -/
def OddFunctionWithConditions (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x < 0, 2 * x * (deriv f (2 * x)) + f (2 * x) < 0) ∧
  f (-2) = 0

/-- The solution set of xf(2x) < 0 -/
def SolutionSet (f : ℝ → ℝ) : Set ℝ :=
  {x | x * f (2 * x) < 0}

/-- The main theorem -/
theorem solution_set_characterization (f : ℝ → ℝ) 
  (hf : OddFunctionWithConditions f) : 
  SolutionSet f = {x | -1 < x ∧ x < 1 ∧ x ≠ 0} := by
  sorry

end solution_set_characterization_l641_64149


namespace parabolas_coefficient_sum_zero_l641_64109

/-- Two distinct parabolas with leading coefficients p and q, where the vertex of each parabola lies on the other parabola -/
structure DistinctParabolas (p q : ℝ) : Prop where
  distinct : p ≠ q
  vertex_on_other : ∃ (a b : ℝ), a ≠ 0 ∧ b = p * a^2 ∧ 0 = q * a^2 + b

/-- The sum of leading coefficients of two distinct parabolas with vertices on each other is zero -/
theorem parabolas_coefficient_sum_zero {p q : ℝ} (h : DistinctParabolas p q) : p + q = 0 := by
  sorry

end parabolas_coefficient_sum_zero_l641_64109


namespace two_pow_2016_days_from_thursday_is_friday_l641_64112

-- Define the days of the week
inductive Day : Type
  | monday : Day
  | tuesday : Day
  | wednesday : Day
  | thursday : Day
  | friday : Day
  | saturday : Day
  | sunday : Day

def next_day (d : Day) : Day :=
  match d with
  | Day.monday => Day.tuesday
  | Day.tuesday => Day.wednesday
  | Day.wednesday => Day.thursday
  | Day.thursday => Day.friday
  | Day.friday => Day.saturday
  | Day.saturday => Day.sunday
  | Day.sunday => Day.monday

def days_from_now (start : Day) (n : ℕ) : Day :=
  match n with
  | 0 => start
  | n + 1 => next_day (days_from_now start n)

theorem two_pow_2016_days_from_thursday_is_friday :
  days_from_now Day.thursday (2^2016) = Day.friday :=
sorry

end two_pow_2016_days_from_thursday_is_friday_l641_64112


namespace number_problem_l641_64148

theorem number_problem : ∃ x : ℚ, x^2 + 95 = (x - 15)^2 ∧ x = 13/3 := by
  sorry

end number_problem_l641_64148


namespace right_triangle_set_l641_64144

theorem right_triangle_set : ∀ (a b c : ℝ),
  ((a = 1 ∧ b = Real.sqrt 2 ∧ c = 3) ∨
   (a = 3 ∧ b = 4 ∧ c = 5) ∨
   (a = 6 ∧ b = 8 ∧ c = 12) ∨
   (a = 5 ∧ b = 11 ∧ c = 13)) →
  (a^2 + b^2 = c^2 ↔ (a = 3 ∧ b = 4 ∧ c = 5)) :=
by sorry

end right_triangle_set_l641_64144


namespace blue_line_length_is_correct_l641_64127

/-- The length of the white line in inches -/
def white_line_length : ℝ := 7.67

/-- The difference in length between the white and blue lines in inches -/
def length_difference : ℝ := 4.33

/-- The length of the blue line in inches -/
def blue_line_length : ℝ := white_line_length - length_difference

theorem blue_line_length_is_correct : blue_line_length = 3.34 := by
  sorry

end blue_line_length_is_correct_l641_64127


namespace article_price_l641_64175

theorem article_price (selling_price : ℝ) (gain_percent : ℝ) 
  (h1 : selling_price = 110)
  (h2 : gain_percent = 10) : 
  ∃ (original_price : ℝ), 
    selling_price = original_price * (1 + gain_percent / 100) ∧ 
    original_price = 100 := by
  sorry

end article_price_l641_64175


namespace system_solutions_correct_l641_64176

theorem system_solutions_correct :
  -- System 1
  (∃ x y : ℚ, y = 2 * x ∧ 3 * y + 2 * x = 8 ∧ x = 1 ∧ y = 2) ∧
  -- System 2
  (∃ x y : ℚ, x - 3 * y = -2 ∧ 2 * x + 3 * y = 3 ∧ x = 1/3 ∧ y = 7/9) := by
  sorry

#check system_solutions_correct

end system_solutions_correct_l641_64176


namespace triangle_side_length_l641_64169

/-- Given a triangle ABC with sides a, b, c opposite angles A, B, C respectively,
    if c = 2a, b = 4, and cos B = 1/4, then c = 4 -/
theorem triangle_side_length 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h1 : c = 2 * a) 
  (h2 : b = 4) 
  (h3 : Real.cos B = 1 / 4) : 
  c = 4 := by
  sorry

end triangle_side_length_l641_64169


namespace fence_perimeter_is_177_l641_64193

/-- Calculates the outer perimeter of a rectangular fence with specified conditions -/
def fence_perimeter (num_posts : ℕ) (post_width : ℚ) (gap_width : ℚ) : ℚ :=
  let width_posts := num_posts / 4
  let length_posts := width_posts * 2
  let width := (width_posts - 1) * gap_width + width_posts * post_width
  let length := (length_posts - 1) * gap_width + length_posts * post_width
  2 * (width + length)

/-- The outer perimeter of the fence is 177 feet -/
theorem fence_perimeter_is_177 :
  fence_perimeter 36 (1/2) 3 = 177 := by sorry

end fence_perimeter_is_177_l641_64193


namespace energy_drink_consumption_l641_64171

/-- Represents the relationship between coding hours and energy drink consumption -/
def energy_drink_relation (hours : ℝ) (drinks : ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ hours * drinks = k

theorem energy_drink_consumption 
  (h1 : energy_drink_relation 8 3)
  (h2 : energy_drink_relation 10 x) :
  x = 2.4 := by
  sorry

end energy_drink_consumption_l641_64171


namespace trigonometric_simplification_l641_64184

theorem trigonometric_simplification :
  (Real.tan (20 * π / 180) + Real.tan (70 * π / 180) + Real.tan (80 * π / 180)) / Real.cos (30 * π / 180) =
  (1 + Real.cos (10 * π / 180) * Real.cos (20 * π / 180)) / (Real.cos (20 * π / 180) * Real.cos (70 * π / 180) * Real.cos (30 * π / 180)) :=
by sorry

end trigonometric_simplification_l641_64184


namespace bus_car_ratio_l641_64130

theorem bus_car_ratio (num_cars : ℕ) (num_buses : ℕ) : 
  num_cars = 65 →
  num_buses = num_cars - 60 →
  (num_buses : ℚ) / (num_cars : ℚ) = 1 / 13 := by
  sorry

end bus_car_ratio_l641_64130


namespace inequality_proof_l641_64120

theorem inequality_proof (a b : ℝ) (ha : |a| < 2) (hb : |b| < 2) : 2*|a + b| < |4 + a*b| := by
  sorry

end inequality_proof_l641_64120


namespace linear_function_max_value_l641_64194

theorem linear_function_max_value (a : ℝ) (h1 : a ≠ 0) :
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 4 → a * x - a + 2 ≤ 7) ∧
  (∃ x : ℝ, -1 ≤ x ∧ x ≤ 4 ∧ a * x - a + 2 = 7) →
  a = 5/3 ∨ a = -5/2 :=
by sorry

end linear_function_max_value_l641_64194


namespace total_socks_l641_64131

def sock_problem (red_socks black_socks white_socks : ℕ) : Prop :=
  red_socks = 40 ∧
  black_socks = red_socks / 2 ∧
  white_socks = 2 * (red_socks + black_socks)

theorem total_socks (red_socks black_socks white_socks : ℕ) 
  (h : sock_problem red_socks black_socks white_socks) : 
  red_socks + black_socks + white_socks = 180 :=
by sorry

end total_socks_l641_64131


namespace scientific_notation_equivalence_l641_64172

/-- Proves that 11,580,000 is equal to 1.158 × 10^7 in scientific notation -/
theorem scientific_notation_equivalence : 
  11580000 = 1.158 * (10 : ℝ)^7 := by
  sorry

end scientific_notation_equivalence_l641_64172


namespace triangle_problem_l641_64128

/-- Given a triangle ABC with sides a, b, c and corresponding angles A, B, C. -/
theorem triangle_problem (a b c A B C : Real) :
  -- Conditions
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (a * Real.cos B + Real.sqrt 3 * b * Real.sin A = c) →
  (a = 1) →
  (b * c * Real.cos A = 3) →
  -- Conclusions
  (A = π / 6) ∧ (b + c = Real.sqrt 3 + 2) := by
  sorry


end triangle_problem_l641_64128


namespace cubic_expression_zero_l641_64178

theorem cubic_expression_zero (x : ℝ) (h : x^2 + 3*x - 3 = 0) : 
  x^3 + 2*x^2 - 6*x + 3 = 0 := by
  sorry

end cubic_expression_zero_l641_64178


namespace tom_monthly_fluid_intake_l641_64189

/-- Represents Tom's daily fluid intake --/
structure DailyFluidIntake where
  soda : Nat
  water : Nat
  juice : Nat
  sports_drink : Nat

/-- Represents Tom's additional weekend fluid intake --/
structure WeekendExtraFluidIntake where
  smoothie : Nat

/-- Represents the structure of a month --/
structure Month where
  weeks : Nat
  days_per_week : Nat
  weekdays_per_week : Nat
  weekend_days_per_week : Nat

def weekday_intake (d : DailyFluidIntake) : Nat :=
  d.soda * 12 + d.water + d.juice * 8 + d.sports_drink * 16

def weekend_intake (d : DailyFluidIntake) (w : WeekendExtraFluidIntake) : Nat :=
  weekday_intake d + w.smoothie

def total_monthly_intake (d : DailyFluidIntake) (w : WeekendExtraFluidIntake) (m : Month) : Nat :=
  (weekday_intake d * m.weekdays_per_week * m.weeks) +
  (weekend_intake d w * m.weekend_days_per_week * m.weeks)

theorem tom_monthly_fluid_intake :
  let tom_daily := DailyFluidIntake.mk 5 64 3 2
  let tom_weekend := WeekendExtraFluidIntake.mk 32
  let month := Month.mk 4 7 5 2
  total_monthly_intake tom_daily tom_weekend month = 5296 := by
  sorry

end tom_monthly_fluid_intake_l641_64189


namespace unique_n_congruence_l641_64118

theorem unique_n_congruence : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ 123456 [MOD 8] := by
  sorry

end unique_n_congruence_l641_64118


namespace imaginary_part_of_z_l641_64137

theorem imaginary_part_of_z (z : ℂ) (h : z * (1 + 2*I) = 5) : z.im = -2 := by
  sorry

end imaginary_part_of_z_l641_64137


namespace smores_marshmallows_needed_l641_64106

def graham_crackers : ℕ := 48
def marshmallows : ℕ := 6
def crackers_per_smore : ℕ := 2
def marshmallows_per_smore : ℕ := 1

theorem smores_marshmallows_needed : 
  (graham_crackers / crackers_per_smore) - marshmallows = 18 :=
by sorry

end smores_marshmallows_needed_l641_64106


namespace no_partition_sum_product_l641_64186

theorem no_partition_sum_product (x y : ℕ) : 
  x ∈ Finset.range 15 → 
  y ∈ Finset.range 15 → 
  x ≠ y → 
  x * y ≠ 120 - x - y := by
sorry

end no_partition_sum_product_l641_64186


namespace equipment_purchase_problem_l641_64157

/-- Equipment purchase problem -/
theorem equipment_purchase_problem 
  (price_A : ℕ)
  (price_B : ℕ)
  (discount_B : ℕ)
  (total_units : ℕ)
  (min_B : ℕ)
  (h1 : price_A = 40)
  (h2 : 30 * price_B - 5 * discount_B = 1425)
  (h3 : 50 * price_B - 25 * discount_B = 2125)
  (h4 : total_units = 90)
  (h5 : min_B = 15) :
  ∃ (units_A units_B : ℕ),
    units_A + units_B = total_units ∧
    units_B ≥ min_B ∧
    units_B ≤ 2 * units_A ∧
    units_A * price_A + (min units_B 25) * price_B + 
      (max (units_B - 25) 0) * (price_B - discount_B) = 3675 ∧
    ∀ (a b : ℕ),
      a + b = total_units →
      b ≥ min_B →
      b ≤ 2 * a →
      a * price_A + (min b 25) * price_B + 
        (max (b - 25) 0) * (price_B - discount_B) ≥ 3675 := by
  sorry

end equipment_purchase_problem_l641_64157


namespace total_weight_calculation_l641_64134

/-- Given the weight of apples and the ratio of pears to apples, 
    calculate the total weight of apples and pears. -/
def total_weight (apple_weight : ℝ) (pear_to_apple_ratio : ℝ) : ℝ :=
  apple_weight + pear_to_apple_ratio * apple_weight

/-- Theorem stating that the total weight of apples and pears is equal to
    the weight of apples plus three times the weight of apples, 
    given that there are three times as many pears as apples. -/
theorem total_weight_calculation (apple_weight : ℝ) :
  total_weight apple_weight 3 = apple_weight + 3 * apple_weight :=
by
  sorry

#eval total_weight 240 3  -- Should output 960

end total_weight_calculation_l641_64134


namespace germination_probability_l641_64160

/-- The germination rate of seeds -/
def germination_rate : ℝ := 0.8

/-- The number of seeds sown -/
def num_seeds : ℕ := 5

/-- The probability of exactly k successes in n trials with probability p -/
def bernoulli_prob (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k : ℝ) * p^k * (1 - p)^(n - k)

/-- The probability of at least 4 out of 5 seeds germinating -/
def prob_at_least_4 : ℝ :=
  bernoulli_prob num_seeds 4 germination_rate + bernoulli_prob num_seeds 5 germination_rate

theorem germination_probability :
  prob_at_least_4 = 0.73728 := by sorry

end germination_probability_l641_64160


namespace chess_tournament_participants_l641_64121

theorem chess_tournament_participants : ∃ n : ℕ, 
  n > 0 ∧ 
  (n * (n - 1)) / 2 = 171 := by
  sorry

end chess_tournament_participants_l641_64121


namespace gcd_n_minus_three_n_plus_three_eq_one_l641_64191

theorem gcd_n_minus_three_n_plus_three_eq_one (n : ℕ+) 
  (h : (Nat.divisors (n.val^2 - 9)).card = 6) : 
  Nat.gcd (n.val - 3) (n.val + 3) = 1 := by
  sorry

end gcd_n_minus_three_n_plus_three_eq_one_l641_64191


namespace cone_height_ratio_l641_64196

theorem cone_height_ratio (base_circumference : Real) (original_height : Real) (shorter_volume : Real) :
  base_circumference = 20 * Real.pi →
  original_height = 40 →
  shorter_volume = 160 * Real.pi →
  ∃ (shorter_height : Real),
    (1 / 3) * Real.pi * ((base_circumference / (2 * Real.pi)) ^ 2) * shorter_height = shorter_volume ∧
    shorter_height / original_height = 3 / 25 := by
  sorry

end cone_height_ratio_l641_64196


namespace lowest_possible_score_l641_64188

/-- Represents a set of test scores -/
structure TestScores where
  scores : List Nat
  all_valid : ∀ s ∈ scores, s ≤ 100

/-- Calculates the average of a list of scores -/
def average (ts : TestScores) : Rat :=
  (ts.scores.sum : Rat) / ts.scores.length

/-- The problem statement -/
theorem lowest_possible_score 
  (first_two : TestScores)
  (h1 : first_two.scores = [82, 75])
  (h2 : first_two.scores.length = 2)
  : ∃ (last_two : TestScores),
    last_two.scores.length = 2 ∧ 
    (∃ (s : Nat), s ∈ last_two.scores ∧ s = 83) ∧
    average (TestScores.mk (first_two.scores ++ last_two.scores) sorry) = 85 ∧
    (∀ (other_last_two : TestScores),
      other_last_two.scores.length = 2 →
      average (TestScores.mk (first_two.scores ++ other_last_two.scores) sorry) = 85 →
      ∀ (s : Nat), s ∈ other_last_two.scores → s ≥ 83) := by
  sorry

end lowest_possible_score_l641_64188


namespace lesser_fraction_l641_64102

theorem lesser_fraction (x y : ℚ) 
  (h1 : x > 0) 
  (h2 : y > 0) 
  (h3 : x + y = 13/14) 
  (h4 : x * y = 1/8) : 
  min x y = 163/625 := by
sorry

end lesser_fraction_l641_64102


namespace ab_nonnegative_l641_64190

theorem ab_nonnegative (a b : ℚ) (ha : |a| = -a) (hb : |b| ≠ b) : a * b ≥ 0 := by
  sorry

end ab_nonnegative_l641_64190


namespace complex_square_simplification_l641_64122

theorem complex_square_simplification :
  let i : ℂ := Complex.I
  (4 - 3 * i)^2 = 7 - 24 * i :=
by sorry

end complex_square_simplification_l641_64122


namespace perpendicular_vectors_l641_64116

/-- Given vectors a and b in R², if a is perpendicular to (t*a + b), then t = -5 -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (t : ℝ) 
  (h1 : a = (1, -1))
  (h2 : b = (6, -4))
  (h3 : a.1 * (t * a.1 + b.1) + a.2 * (t * a.2 + b.2) = 0) :
  t = -5 := by
  sorry

end perpendicular_vectors_l641_64116


namespace system_solution_l641_64187

theorem system_solution (x y : ℝ) :
  (1 + x) * (1 + x^2) * (1 + x^4) = 1 + y^7 ∧
  (1 + y) * (1 + y^2) * (1 + y^4) = 1 + x^7 →
  (x = 0 ∧ y = 0) ∨ (x = -1 ∧ y = -1) := by
  sorry

end system_solution_l641_64187


namespace equation_solutions_l641_64173

theorem equation_solutions :
  (∀ x : ℝ, 25 * x^2 = 36 ↔ x = 6/5 ∨ x = -6/5) ∧
  (∀ x : ℝ, (1/3) * (x + 2)^3 - 9 = 0 ↔ x = 1) :=
by sorry

end equation_solutions_l641_64173


namespace sum_largest_smallest_prime_factors_1540_l641_64164

theorem sum_largest_smallest_prime_factors_1540 : 
  ∃ (p q : ℕ), 
    Nat.Prime p ∧ 
    Nat.Prime q ∧ 
    p ∣ 1540 ∧ 
    q ∣ 1540 ∧ 
    (∀ r : ℕ, Nat.Prime r → r ∣ 1540 → p ≤ r ∧ r ≤ q) ∧ 
    p + q = 13 :=
by sorry

end sum_largest_smallest_prime_factors_1540_l641_64164


namespace existence_and_digit_sum_l641_64163

/-- The sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Proves the existence of N and its properties -/
theorem existence_and_digit_sum :
  ∃ N : ℕ, N^2 = 36^50 * 50^36 ∧ sum_of_digits N = 54 := by sorry

end existence_and_digit_sum_l641_64163


namespace bee_count_l641_64153

theorem bee_count (legs_per_bee : ℕ) (total_legs : ℕ) (h1 : legs_per_bee = 6) (h2 : total_legs = 12) :
  total_legs / legs_per_bee = 2 := by
  sorry

end bee_count_l641_64153


namespace dog_bones_problem_l641_64105

theorem dog_bones_problem (buried_bones initial_bones final_bones : ℚ) : 
  buried_bones = 367.5 ∧ 
  final_bones = -860 ∧ 
  initial_bones - buried_bones = final_bones → 
  initial_bones = 367.5 := by
sorry


end dog_bones_problem_l641_64105


namespace fifth_number_13th_row_is_715_l641_64114

/-- The fifth number in the 13th row of Pascal's triangle -/
def fifth_number_13th_row : ℕ :=
  Nat.choose 13 4

/-- Theorem stating that the fifth number in the 13th row of Pascal's triangle is 715 -/
theorem fifth_number_13th_row_is_715 : fifth_number_13th_row = 715 := by
  sorry

end fifth_number_13th_row_is_715_l641_64114


namespace pure_imaginary_complex_number_l641_64183

theorem pure_imaginary_complex_number (m : ℝ) : 
  (m^2 - 3*m = 0) ∧ (m^2 - 5*m + 6 ≠ 0) → m = 0 := by
  sorry

end pure_imaginary_complex_number_l641_64183


namespace square_diagonal_l641_64174

theorem square_diagonal (A : ℝ) (h : A = 200) : 
  ∃ d : ℝ, d^2 = 2 * A ∧ d = 20 := by
  sorry

end square_diagonal_l641_64174


namespace unique_solution_for_exponential_equation_l641_64138

theorem unique_solution_for_exponential_equation :
  ∀ n m : ℕ+, 5^(n : ℕ) = 6*(m : ℕ)^2 + 1 ↔ n = 2 ∧ m = 2 := by
  sorry

end unique_solution_for_exponential_equation_l641_64138


namespace fraction_equality_l641_64182

def f (x : ℕ) : ℚ := (x^4 + 400 : ℚ)

def numerator : ℚ := f 15 * f 27 * f 39 * f 51 * f 63
def denominator : ℚ := f 5 * f 17 * f 29 * f 41 * f 53

theorem fraction_equality : numerator / denominator = 4115 / 45 := by
  sorry

end fraction_equality_l641_64182


namespace geometric_sequence_a3_l641_64142

def geometric_sequence (a : ℕ → ℝ) :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_a3 (a : ℕ → ℝ) :
  geometric_sequence a → a 1 = 1 → a 5 = 9 → a 3 = 3 :=
by
  sorry

end geometric_sequence_a3_l641_64142


namespace ellipse_line_slope_l641_64141

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 8 + y^2 / 4 = 1

-- Define the right focus
def right_focus : ℝ × ℝ := (2, 0)

-- Define a line passing through a point with a given slope
def line_through_point (m : ℝ) (p : ℝ × ℝ) (x y : ℝ) : Prop :=
  y - p.2 = m * (x - p.1)

-- Define a circle passing through three points
def circle_through_points (p1 p2 p3 : ℝ × ℝ) (center : ℝ × ℝ) : Prop :=
  (p1.1 - center.1)^2 + (p1.2 - center.2)^2 =
  (p2.1 - center.1)^2 + (p2.2 - center.2)^2 ∧
  (p1.1 - center.1)^2 + (p1.2 - center.2)^2 =
  (p3.1 - center.1)^2 + (p3.2 - center.2)^2

-- Theorem statement
theorem ellipse_line_slope :
  ∀ (A B : ℝ × ℝ) (m : ℝ),
    ellipse A.1 A.2 ∧
    ellipse B.1 B.2 ∧
    line_through_point m right_focus A.1 A.2 ∧
    line_through_point m right_focus B.1 B.2 ∧
    (∃ (t : ℝ), circle_through_points A B (-Real.sqrt 7, 0) (0, t)) →
    m = Real.sqrt 2 / 2 ∨ m = -Real.sqrt 2 / 2 :=
by sorry

end ellipse_line_slope_l641_64141


namespace geometric_series_common_ratio_l641_64145

/-- The first term of the geometric series -/
def a₁ : ℚ := 7/8

/-- The second term of the geometric series -/
def a₂ : ℚ := -14/27

/-- The third term of the geometric series -/
def a₃ : ℚ := 28/81

/-- The common ratio of the geometric series -/
def r : ℚ := -2/3

/-- Theorem stating that the given series is geometric with common ratio r -/
theorem geometric_series_common_ratio :
  a₂ = a₁ * r ∧ a₃ = a₂ * r := by sorry

end geometric_series_common_ratio_l641_64145


namespace point_C_coordinates_l641_64165

def A : ℝ × ℝ := (1, -2)
def B : ℝ × ℝ := (7, 2)

theorem point_C_coordinates :
  ∀ C : ℝ × ℝ,
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ C = (1 - t) • A + t • B) →
  (dist A C = 2 * dist C B) →
  C = (5, 2/3) :=
by sorry

end point_C_coordinates_l641_64165


namespace max_value_of_complex_expression_l641_64195

theorem max_value_of_complex_expression (z : ℂ) (h : Complex.abs z = 1) :
  ∃ (M : ℝ), M = 4 ∧ ∀ w : ℂ, Complex.abs w = 1 → Complex.abs (w + 2 * Real.sqrt 2 + Complex.I) ≤ M :=
by sorry

end max_value_of_complex_expression_l641_64195


namespace books_given_away_l641_64123

theorem books_given_away (original_books : Real) (books_left : Nat) : 
  original_books = 54.0 → books_left = 31 → original_books - books_left = 23 := by
  sorry

end books_given_away_l641_64123


namespace red_faces_up_possible_l641_64154

/-- Represents a cubic block with one red face and five white faces -/
structure Block where
  redFaceUp : Bool

/-- Represents an n x n chessboard with cubic blocks -/
structure Chessboard (n : ℕ) where
  blocks : Matrix (Fin n) (Fin n) Block

/-- Represents a rotation of blocks in a row or column -/
inductive Rotation
  | Row : Fin n → Rotation
  | Column : Fin n → Rotation

/-- Applies a rotation to the chessboard -/
def applyRotation (board : Chessboard n) (rot : Rotation) : Chessboard n :=
  sorry

/-- Checks if all blocks on the chessboard have their red faces up -/
def allRedFacesUp (board : Chessboard n) : Bool :=
  sorry

/-- Theorem stating that it's possible to turn all red faces up after a finite number of rotations -/
theorem red_faces_up_possible (n : ℕ) :
  ∃ (rotations : List Rotation), ∀ (initial : Chessboard n),
    allRedFacesUp (rotations.foldl applyRotation initial) := by
  sorry

end red_faces_up_possible_l641_64154


namespace prime_arithmetic_sequence_ones_digit_l641_64179

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def ones_digit (n : ℕ) : ℕ := n % 10

def arithmetic_sequence (a b c d : ℕ) : Prop :=
  ∃ (r : ℕ), r > 0 ∧ b = a + r ∧ c = b + r ∧ d = c + r

theorem prime_arithmetic_sequence_ones_digit
  (p q r s : ℕ)
  (h_prime : is_prime p ∧ is_prime q ∧ is_prime r ∧ is_prime s)
  (h_seq : arithmetic_sequence p q r s)
  (h_p_gt_5 : p > 5)
  (h_diff : ∃ (d : ℕ), d = 10 ∧ q = p + d ∧ r = q + d ∧ s = r + d) :
  ones_digit p = 1 ∨ ones_digit p = 3 ∨ ones_digit p = 7 ∨ ones_digit p = 9 :=
sorry

end prime_arithmetic_sequence_ones_digit_l641_64179


namespace difference_of_squares_503_497_l641_64192

theorem difference_of_squares_503_497 : 503^2 - 497^2 = 6000 := by
  sorry

end difference_of_squares_503_497_l641_64192
