import Mathlib

namespace shaded_shape_area_l2354_235468

/-- The area of a shape composed of a central square and four right triangles -/
theorem shaded_shape_area (grid_size : ℕ) (square_side : ℕ) (triangle_side : ℕ) : 
  grid_size = 10 → 
  square_side = 2 → 
  triangle_side = 5 → 
  (square_side * square_side + 4 * (triangle_side * triangle_side / 2 : ℚ)) = 54 := by
  sorry

#check shaded_shape_area

end shaded_shape_area_l2354_235468


namespace craig_final_apples_l2354_235419

def craig_initial_apples : ℕ := 20
def shared_apples : ℕ := 7

theorem craig_final_apples :
  craig_initial_apples - shared_apples = 13 := by
  sorry

end craig_final_apples_l2354_235419


namespace complement_intersection_theorem_l2354_235413

-- Define the universal set I
def I : Set ℕ := {0, 1, 2, 3, 4}

-- Define set M
def M : Set ℕ := {1, 2, 3}

-- Define set N
def N : Set ℕ := {0, 3, 4}

-- Theorem statement
theorem complement_intersection_theorem :
  (I \ M) ∩ N = {0, 4} := by
  sorry

end complement_intersection_theorem_l2354_235413


namespace shower_water_usage_l2354_235489

/-- The total water usage of Roman and Remy's showers -/
theorem shower_water_usage (R : ℝ) 
  (h1 : 3 * R + 1 = 25) : R + (3 * R + 1) = 33 := by
  sorry

end shower_water_usage_l2354_235489


namespace tenth_term_of_sequence_l2354_235471

def arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1) * d

theorem tenth_term_of_sequence (a₁ a₂ a₃ : ℚ) (h₁ : a₁ = 1/2) (h₂ : a₂ = 5/6) (h₃ : a₃ = 7/6) :
  arithmetic_sequence a₁ (a₂ - a₁) 10 = 7/2 := by
  sorry

end tenth_term_of_sequence_l2354_235471


namespace stratified_sample_problem_l2354_235432

/-- Given a stratified sample from a high school where:
  * The sample size is 55 students
  * 10 students are from the first year
  * 25 students are from the second year
  * There are 400 students in the third year
Prove that the total number of students in the first and second years combined is 700. -/
theorem stratified_sample_problem (sample_size : ℕ) (first_year_sample : ℕ) (second_year_sample : ℕ) (third_year_total : ℕ) :
  sample_size = 55 →
  first_year_sample = 10 →
  second_year_sample = 25 →
  third_year_total = 400 →
  ∃ (first_and_second_total : ℕ),
    first_and_second_total = 700 ∧
    (first_year_sample + second_year_sample : ℚ) / sample_size = first_and_second_total / (first_and_second_total + third_year_total) :=
by sorry

end stratified_sample_problem_l2354_235432


namespace elsa_angus_token_difference_l2354_235406

/-- Calculates the difference in total token value between two people -/
def tokenValueDifference (elsa_tokens : ℕ) (angus_tokens : ℕ) (token_value : ℕ) : ℕ :=
  (elsa_tokens * token_value) - (angus_tokens * token_value)

/-- Proves that the difference in token value between Elsa and Angus is $20 -/
theorem elsa_angus_token_difference :
  tokenValueDifference 60 55 4 = 20 := by
  sorry

end elsa_angus_token_difference_l2354_235406


namespace cow_chicken_problem_l2354_235498

theorem cow_chicken_problem (c h : ℕ) : 
  (4 * c + 2 * h = 2 * (c + h) + 18) → c = 9 :=
by
  sorry

end cow_chicken_problem_l2354_235498


namespace find_a_l2354_235443

-- Define the sets A and B
def A (a : ℤ) : Set ℤ := {1, 3, a}
def B (a : ℤ) : Set ℤ := {1, a^2}

-- State the theorem
theorem find_a : 
  ∀ a : ℤ, (A a ∪ B a = {1, 3, a}) → (a = 0 ∨ a = 1 ∨ a = -1) :=
by sorry

end find_a_l2354_235443


namespace p_or_q_is_true_l2354_235412

-- Define proposition p
def p (x y : ℝ) : Prop := x^2 + y^2 ≠ 0 → (x ≠ 0 ∨ y ≠ 0)

-- Define proposition q
def q (m : ℝ) : Prop := m > -2 → ∃ x : ℝ, x^2 + 2*x - m = 0

-- Theorem statement
theorem p_or_q_is_true :
  (∀ x y : ℝ, x^2 + y^2 ≠ 0 → (x ≠ 0 ∨ y ≠ 0)) →
  (∃ m : ℝ, m > -2 ∧ ¬(∃ x : ℝ, x^2 + 2*x - m = 0)) →
  ∀ x y m : ℝ, p x y ∨ q m :=
sorry

end p_or_q_is_true_l2354_235412


namespace sum_of_digits_product_35_42_base8_l2354_235423

def base8_to_base10 (n : Nat) : Nat :=
  (n / 10) * 8 + n % 10

def base10_to_base8 (n : Nat) : Nat :=
  if n < 8 then n
  else (base10_to_base8 (n / 8)) * 10 + n % 8

def sum_of_digits (n : Nat) : Nat :=
  if n < 10 then n
  else n % 10 + sum_of_digits (n / 10)

theorem sum_of_digits_product_35_42_base8 :
  sum_of_digits (base10_to_base8 (base8_to_base10 35 * base8_to_base10 42)) = 13 := by
  sorry

end sum_of_digits_product_35_42_base8_l2354_235423


namespace solve_exponential_equation_l2354_235434

theorem solve_exponential_equation :
  ∃ x : ℝ, (3 : ℝ)^4 * (3 : ℝ)^x = 81 ∧ x = 0 :=
by
  sorry

end solve_exponential_equation_l2354_235434


namespace farmer_truck_count_l2354_235416

/-- Proves the number of trucks a farmer has, given tank capacity, tanks per truck, and total water capacity. -/
theorem farmer_truck_count (tank_capacity : ℕ) (tanks_per_truck : ℕ) (total_capacity : ℕ) 
  (h1 : tank_capacity = 150)
  (h2 : tanks_per_truck = 3)
  (h3 : total_capacity = 1350) :
  total_capacity / (tank_capacity * tanks_per_truck) = 3 :=
by sorry

end farmer_truck_count_l2354_235416


namespace large_square_area_l2354_235431

/-- The area of a square formed by four congruent rectangles and a smaller square -/
theorem large_square_area (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 20) : 
  (x + y)^2 = 400 := by
  sorry

#check large_square_area

end large_square_area_l2354_235431


namespace unique_solution_2000_l2354_235492

-- Define the differential equation
def diff_eq (y : ℝ → ℝ) (x : ℝ) : Prop :=
  ∀ x > 0, deriv y x = Real.log (y x / x)

-- Define the solution y(x) with initial condition
noncomputable def y : ℝ → ℝ :=
  sorry

-- Main theorem
theorem unique_solution_2000 :
  ∃! x : ℝ, x > 0 ∧ y x = 2000 :=
by
  sorry


end unique_solution_2000_l2354_235492


namespace arithmetic_calculations_l2354_235405

theorem arithmetic_calculations :
  (3 * 232 + 456 = 1152) ∧
  (760 * 5 - 2880 = 920) ∧
  (805 / 7 = 115) ∧
  (45 + 255 / 5 = 96) := by
  sorry

end arithmetic_calculations_l2354_235405


namespace rectangle_perimeter_l2354_235445

/-- Given a triangle with sides 10, 24, and 26 units, and a rectangle with width 8 units
    and area equal to the triangle's area, the perimeter of the rectangle is 46 units. -/
theorem rectangle_perimeter (a b c w : ℝ) (h1 : a = 10) (h2 : b = 24) (h3 : c = 26) (h4 : w = 8)
  (h5 : w * (a * b / 2 / w) = a * b / 2) : 2 * (w + (a * b / 2 / w)) = 46 :=
by sorry

end rectangle_perimeter_l2354_235445


namespace toy_count_l2354_235458

/-- The position of the yellow toy from the left -/
def position_from_left : ℕ := 10

/-- The position of the yellow toy from the right -/
def position_from_right : ℕ := 7

/-- The total number of toys in the row -/
def total_toys : ℕ := position_from_left + position_from_right - 1

theorem toy_count : total_toys = 16 := by
  sorry

end toy_count_l2354_235458


namespace geometric_mean_minimum_l2354_235426

theorem geometric_mean_minimum (x y z : ℝ) (hx : x > 1) (hy : y > 1) (hz : z > 1) 
  (h_geom_mean : z^2 = x*y) : 
  (Real.log z) / (4 * Real.log x) + (Real.log z) / (Real.log y) ≥ 9/8 := by
sorry

end geometric_mean_minimum_l2354_235426


namespace arithmetic_progression_nth_term_l2354_235421

theorem arithmetic_progression_nth_term (a d n : ℕ) (Tn : ℕ) 
  (h1 : a = 2) 
  (h2 : d = 8) 
  (h3 : Tn = 90) 
  (h4 : Tn = a + (n - 1) * d) : n = 12 := by
  sorry

end arithmetic_progression_nth_term_l2354_235421


namespace player_a_not_losing_probability_l2354_235415

theorem player_a_not_losing_probability 
  (prob_draw : ℝ) 
  (prob_a_win : ℝ) 
  (h1 : prob_draw = 0.4) 
  (h2 : prob_a_win = 0.4) : 
  prob_draw + prob_a_win = 0.8 := by
  sorry

end player_a_not_losing_probability_l2354_235415


namespace coloring_theorem_l2354_235454

/-- Given finite sets A, B, C, D, this function calculates the number of ways
    to color three adjacent elements with the condition that adjacent elements
    must have different colors. -/
def colorThreeAdjacent (A B C : Finset α) : ℕ :=
  A.card * (B.card - 1) * (C.card - 1)

/-- Given finite sets A, B, C, D, this function calculates the number of ways
    to color four adjacent elements with the condition that adjacent elements
    must have different colors. -/
def colorFourAdjacent (A B C D : Finset α) : ℕ :=
  A.card * (B.card - 1) * (C.card - 1) * (D.card - 1)

theorem coloring_theorem (A B C D : Finset α) :
  (colorThreeAdjacent A B C = A.card * (B.card - 1) * (C.card - 1)) ∧
  (colorFourAdjacent A B C D = A.card * (B.card - 1) * (C.card - 1) * (D.card - 1)) := by
  sorry

end coloring_theorem_l2354_235454


namespace fraction_simplifiable_l2354_235473

theorem fraction_simplifiable (e : ℤ) : 
  (∃ (a b : ℤ), a ≠ 0 ∧ b ≠ 0 ∧ (16 * e - 10) * b = (10 * e - 3) * a) ↔ 
  (∃ (k : ℤ), e = 13 * k + 12) := by
sorry

end fraction_simplifiable_l2354_235473


namespace simple_interest_calculation_l2354_235427

/-- The simple interest calculation problem --/
theorem simple_interest_calculation (P : ℝ) : 
  (∀ (r : ℝ) (A : ℝ), 
    r = 0.04 → 
    A = 36.4 → 
    A = P + P * r) → 
  P = 35 := by
  sorry

end simple_interest_calculation_l2354_235427


namespace simplify_expression_l2354_235469

theorem simplify_expression : (256 : ℝ)^(1/4) * (125 : ℝ)^(1/3) = 20 := by
  sorry

end simplify_expression_l2354_235469


namespace charles_total_money_l2354_235461

/-- The value of a penny in cents -/
def penny_value : ℕ := 1

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The number of pennies Charles found -/
def pennies_found : ℕ := 6

/-- The number of nickels Charles had at home -/
def nickels_at_home : ℕ := 3

/-- Theorem stating that the total value of Charles' coins is 21 cents -/
theorem charles_total_money : 
  pennies_found * penny_value + nickels_at_home * nickel_value = 21 := by
  sorry

end charles_total_money_l2354_235461


namespace min_value_of_f_l2354_235480

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 6*x + 8

-- State the theorem
theorem min_value_of_f :
  ∃ (y : ℝ), (∀ (x : ℝ), x ≥ 0 → f x ≥ y) ∧ (∃ (x : ℝ), x ≥ 0 ∧ f x = y) ∧ y = -1 := by
  sorry

end min_value_of_f_l2354_235480


namespace car_dealership_problem_l2354_235410

/-- Represents the initial number of cars on the lot -/
def initial_cars : ℕ := 280

/-- Represents the number of cars in the new shipment -/
def new_shipment : ℕ := 80

/-- Represents the initial percentage of silver cars -/
def initial_silver_percent : ℚ := 20 / 100

/-- Represents the percentage of non-silver cars in the new shipment -/
def new_non_silver_percent : ℚ := 35 / 100

/-- Represents the final percentage of silver cars after the new shipment -/
def final_silver_percent : ℚ := 30 / 100

theorem car_dealership_problem :
  let initial_silver := initial_silver_percent * initial_cars
  let new_silver := (1 - new_non_silver_percent) * new_shipment
  let total_cars := initial_cars + new_shipment
  let total_silver := initial_silver + new_silver
  (total_silver : ℚ) / total_cars = final_silver_percent := by
  sorry

end car_dealership_problem_l2354_235410


namespace sequence_shortening_l2354_235465

/-- A sequence of digits where each digit is independently chosen from {0, 9} -/
def DigitSequence := Fin 2015 → Fin 10

/-- The probability of a digit being 0 or 9 -/
def p : ℝ := 0.1

/-- The number of digits in the original sequence -/
def n : ℕ := 2015

/-- The number of digits that can potentially be removed -/
def k : ℕ := 2014

theorem sequence_shortening (seq : DigitSequence) :
  /- The probability of the sequence shortening by exactly one digit -/
  (Nat.choose k 1 : ℝ) * p^1 * (1 - p)^(k - 1) = 
    (2014 : ℝ) * 0.1 * 0.9^2013 ∧
  /- The expected length of the new sequence -/
  (n : ℝ) - (k : ℝ) * p = 1813.6 := by
  sorry


end sequence_shortening_l2354_235465


namespace complex_square_in_fourth_quadrant_l2354_235488

theorem complex_square_in_fourth_quadrant :
  let z : ℂ := 2 - I
  (z^2).re > 0 ∧ (z^2).im < 0 :=
by sorry

end complex_square_in_fourth_quadrant_l2354_235488


namespace work_completion_indeterminate_l2354_235487

structure WorkScenario where
  men : ℕ
  days : ℕ
  hours_per_day : ℝ

def total_work (scenario : WorkScenario) : ℝ :=
  scenario.men * scenario.days * scenario.hours_per_day

theorem work_completion_indeterminate 
  (scenario1 scenario2 : WorkScenario)
  (h1 : scenario1.men = 8)
  (h2 : scenario1.days = 24)
  (h3 : scenario2.men = 12)
  (h4 : scenario2.days = 16)
  (h5 : scenario1.hours_per_day = scenario2.hours_per_day)
  (h6 : total_work scenario1 = total_work scenario2) :
  ∀ (h : ℝ), ∃ (scenario1' scenario2' : WorkScenario),
    scenario1'.men = scenario1.men ∧
    scenario1'.days = scenario1.days ∧
    scenario2'.men = scenario2.men ∧
    scenario2'.days = scenario2.days ∧
    scenario1'.hours_per_day = h ∧
    scenario2'.hours_per_day = h ∧
    total_work scenario1' = total_work scenario2' :=
sorry

end work_completion_indeterminate_l2354_235487


namespace necessary_not_sufficient_l2354_235493

theorem necessary_not_sufficient (a : ℝ) :
  (∀ a, 1 / a > 1 → a < 1) ∧ (∃ a, a < 1 ∧ 1 / a ≤ 1) := by
  sorry

end necessary_not_sufficient_l2354_235493


namespace jar_capacity_ratio_l2354_235464

theorem jar_capacity_ratio (capacity_x capacity_y : ℝ) : 
  capacity_x > 0 → 
  capacity_y > 0 → 
  (1/2 : ℝ) * capacity_x + (1/2 : ℝ) * capacity_y = (3/4 : ℝ) * capacity_x → 
  capacity_y / capacity_x = (1/2 : ℝ) := by
sorry

end jar_capacity_ratio_l2354_235464


namespace no_nonzero_solution_l2354_235499

theorem no_nonzero_solution (x y z : ℝ) : 
  x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 →
  (x^2 + x = y^2 - y ∧ 
   y^2 + y = z^2 - z ∧ 
   z^2 + z = x^2 - x) → 
  False :=
sorry

end no_nonzero_solution_l2354_235499


namespace paving_rate_calculation_l2354_235401

/-- Calculates the rate of paving per square meter given room dimensions and total cost -/
theorem paving_rate_calculation (length width total_cost : ℝ) :
  length = 5.5 ∧ width = 4 ∧ total_cost = 17600 →
  total_cost / (length * width) = 800 := by
  sorry

end paving_rate_calculation_l2354_235401


namespace power_multiplication_l2354_235417

theorem power_multiplication (a : ℝ) : a^3 * a^2 = a^5 := by
  sorry

end power_multiplication_l2354_235417


namespace debby_museum_pictures_l2354_235433

/-- The number of pictures Debby took at the zoo -/
def zoo_pictures : ℕ := 24

/-- The number of pictures Debby deleted -/
def deleted_pictures : ℕ := 14

/-- The number of pictures Debby had remaining after deletion -/
def remaining_pictures : ℕ := 22

/-- The number of pictures Debby took at the museum -/
def museum_pictures : ℕ := zoo_pictures + deleted_pictures - remaining_pictures

theorem debby_museum_pictures : museum_pictures = 12 := by
  sorry

end debby_museum_pictures_l2354_235433


namespace fraction_equality_l2354_235422

theorem fraction_equality (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a / b = 5 / 8) :
  (b - a) / a = 3 / 5 := by
  sorry

end fraction_equality_l2354_235422


namespace sum_of_squares_l2354_235446

theorem sum_of_squares (a b c : ℝ) : 
  a + b + c = 20 → 
  a * b + b * c + a * c = 131 → 
  a^2 + b^2 + c^2 = 138 := by
sorry

end sum_of_squares_l2354_235446


namespace smiths_class_a_students_l2354_235462

theorem smiths_class_a_students (johnson_total : ℕ) (johnson_a : ℕ) (smith_total : ℕ) :
  johnson_total = 20 →
  johnson_a = 12 →
  smith_total = 30 →
  (johnson_a : ℚ) / johnson_total = (smith_a : ℚ) / smith_total →
  smith_a = 18 :=
by
  sorry
where
  smith_a : ℕ := sorry

end smiths_class_a_students_l2354_235462


namespace rightward_translation_of_point_l2354_235456

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Translation to the right by a given distance -/
def translateRight (p : Point2D) (distance : ℝ) : Point2D :=
  { x := p.x + distance, y := p.y }

theorem rightward_translation_of_point :
  let initial_point : Point2D := { x := 4, y := -3 }
  let translated_point := translateRight initial_point 1
  translated_point = { x := 5, y := -3 } := by sorry

end rightward_translation_of_point_l2354_235456


namespace final_number_lower_bound_board_game_result_l2354_235440

/-- 
Given a positive integer n and a real number a ≥ n, 
we define a sequence of operations on a multiset of n real numbers,
initially all equal to a. In each step, we replace any two numbers
x and y in the multiset with (x+y)/4 until only one number remains.
-/
def final_number (n : ℕ+) (a : ℝ) (h : a ≥ n) : ℝ :=
  sorry

/-- 
The final number obtained after performing the operations
is always greater than or equal to a/n.
-/
theorem final_number_lower_bound (n : ℕ+) (a : ℝ) (h : a ≥ n) :
  final_number n a h ≥ a / n :=
  sorry

/--
For the specific case of 2023 numbers, each initially equal to 2023,
the final number is greater than 1.
-/
theorem board_game_result :
  final_number 2023 2023 (by norm_num) > 1 :=
  sorry

end final_number_lower_bound_board_game_result_l2354_235440


namespace price_of_added_toy_l2354_235482

/-- Given 5 toys with an average price of $10, adding one toy to make the new average $11 for 6 toys, prove the price of the added toy is $16. -/
theorem price_of_added_toy (num_toys : ℕ) (avg_price : ℚ) (new_num_toys : ℕ) (new_avg_price : ℚ) :
  num_toys = 5 →
  avg_price = 10 →
  new_num_toys = num_toys + 1 →
  new_avg_price = 11 →
  (new_num_toys : ℚ) * new_avg_price - (num_toys : ℚ) * avg_price = 16 := by
  sorry

end price_of_added_toy_l2354_235482


namespace four_digit_sum_l2354_235437

/-- The number of four-digit even numbers -/
def C : ℕ := 4500

/-- The number of four-digit numbers that are multiples of 7 -/
def D : ℕ := 1285

/-- Theorem stating that the sum of four-digit even numbers and four-digit multiples of 7 is 5785 -/
theorem four_digit_sum : C + D = 5785 := by
  sorry

end four_digit_sum_l2354_235437


namespace expression_factorization_l2354_235428

theorem expression_factorization (x : ℝ) :
  (3 * x^3 - 67 * x^2 - 14) - (-8 * x^3 + 3 * x^2 - 14) = x^2 * (11 * x - 70) := by
  sorry

end expression_factorization_l2354_235428


namespace jane_ice_cream_pudding_cost_difference_l2354_235474

theorem jane_ice_cream_pudding_cost_difference :
  let ice_cream_cones : ℕ := 15
  let pudding_cups : ℕ := 5
  let ice_cream_cost_per_cone : ℕ := 5
  let pudding_cost_per_cup : ℕ := 2
  let total_ice_cream_cost := ice_cream_cones * ice_cream_cost_per_cone
  let total_pudding_cost := pudding_cups * pudding_cost_per_cup
  total_ice_cream_cost - total_pudding_cost = 65 :=
by
  sorry

end jane_ice_cream_pudding_cost_difference_l2354_235474


namespace joe_age_l2354_235472

/-- Given that Joe has a daughter Jane, and their ages satisfy certain conditions,
    prove that Joe's age is 38. -/
theorem joe_age (joe_age jane_age : ℕ) 
  (sum_ages : joe_age + jane_age = 54)
  (diff_ages : joe_age - jane_age = 22) : 
  joe_age = 38 := by
sorry

end joe_age_l2354_235472


namespace johnnys_travel_time_l2354_235457

/-- Proves that given the specified conditions, Johnny's total travel time is 1.6 hours -/
theorem johnnys_travel_time 
  (distance_to_school : ℝ)
  (jogging_speed : ℝ)
  (bus_speed : ℝ)
  (h1 : distance_to_school = 6.461538461538462)
  (h2 : jogging_speed = 5)
  (h3 : bus_speed = 21) :
  distance_to_school / jogging_speed + distance_to_school / bus_speed = 1.6 :=
by sorry


end johnnys_travel_time_l2354_235457


namespace older_ate_twelve_l2354_235479

/-- Represents the pancake eating scenario --/
structure PancakeScenario where
  initial_pancakes : ℕ
  final_pancakes : ℕ
  younger_eats : ℕ
  older_eats : ℕ
  grandma_bakes : ℕ

/-- Calculates the number of pancakes eaten by the older grandchild --/
def older_grandchild_pancakes (scenario : PancakeScenario) : ℕ :=
  let net_reduction := scenario.younger_eats + scenario.older_eats - scenario.grandma_bakes
  let cycles := (scenario.initial_pancakes - scenario.final_pancakes) / net_reduction
  scenario.older_eats * cycles

/-- Theorem stating that the older grandchild ate 12 pancakes in the given scenario --/
theorem older_ate_twelve (scenario : PancakeScenario) 
  (h1 : scenario.initial_pancakes = 19)
  (h2 : scenario.final_pancakes = 11)
  (h3 : scenario.younger_eats = 1)
  (h4 : scenario.older_eats = 3)
  (h5 : scenario.grandma_bakes = 2) :
  older_grandchild_pancakes scenario = 12 := by
  sorry

end older_ate_twelve_l2354_235479


namespace jerry_lawsuit_years_l2354_235430

def salary_per_year : ℕ := 50000
def medical_bills : ℕ := 200000
def punitive_multiplier : ℕ := 3
def settlement_percentage : ℚ := 4/5
def total_received : ℕ := 5440000

def total_damages (Y : ℕ) : ℕ :=
  Y * salary_per_year + medical_bills + punitive_multiplier * (Y * salary_per_year + medical_bills)

theorem jerry_lawsuit_years :
  ∃ Y : ℕ, (↑total_received : ℚ) = settlement_percentage * (↑(total_damages Y) : ℚ) ∧ Y = 30 :=
by sorry

end jerry_lawsuit_years_l2354_235430


namespace distribute_items_eq_36_l2354_235484

/-- The number of ways to distribute 4 distinct items into 3 non-empty groups -/
def distribute_items : ℕ :=
  (Nat.choose 4 2 * Nat.choose 2 1 * Nat.choose 1 1) * Nat.factorial 3 / Nat.factorial 2

/-- Theorem stating that the number of ways to distribute 4 distinct items
    into 3 non-empty groups is 36 -/
theorem distribute_items_eq_36 : distribute_items = 36 := by
  sorry

end distribute_items_eq_36_l2354_235484


namespace starting_number_proof_l2354_235453

def is_divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem starting_number_proof (n : ℕ) : 
  (∃! m : ℕ, m > n ∧ m < 580 ∧ 
   (∃ l : List ℕ, l.length = 6 ∧ 
    (∀ x ∈ l, x > n ∧ x < 580 ∧ is_divisible_by x 45 ∧ is_divisible_by x 6) ∧
    (∀ y : ℕ, y > n ∧ y < 580 ∧ is_divisible_by y 45 ∧ is_divisible_by y 6 → y ∈ l))) →
  (∀ k : ℕ, k > n → 
    ¬(∃ l : List ℕ, l.length = 6 ∧ 
      (∀ x ∈ l, x > k ∧ x < 580 ∧ is_divisible_by x 45 ∧ is_divisible_by x 6) ∧
      (∀ y : ℕ, y > k ∧ y < 580 ∧ is_divisible_by y 45 ∧ is_divisible_by y 6 → y ∈ l))) →
  n = 450 := by
sorry

end starting_number_proof_l2354_235453


namespace geometric_sequence_common_ratio_l2354_235459

/-- Represents a geometric sequence with first term a and common ratio q -/
def GeometricSequence (a : ℝ) (q : ℝ) : ℕ → ℝ := fun n ↦ a * q ^ (n - 1)

/-- The common ratio of a geometric sequence satisfying given conditions is 2 -/
theorem geometric_sequence_common_ratio 
  (a : ℝ) (q : ℝ) (h_pos : q > 0) :
  let seq := GeometricSequence a q
  (seq 3 - 3 * seq 2 = 2) ∧ 
  (5 * seq 4 = (12 * seq 3 + 2 * seq 5) / 2) →
  q = 2 := by
sorry

end geometric_sequence_common_ratio_l2354_235459


namespace area_and_inequality_l2354_235460

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := 2 * abs (x - a) - x + a

-- State the theorem
theorem area_and_inequality (a : ℝ) (h : a > 0) :
  (∃ A : ℝ, A = (8:ℝ)/3 ∧ A = ∫ x in (2*a/3)..(2*a), (f a x - a)) → a = 2 ∧
  (∀ x : ℝ, f a x > x ↔ x < 3*a/4) :=
sorry

end area_and_inequality_l2354_235460


namespace time_to_clean_wall_l2354_235448

/-- Represents the dimensions of the wall in large squares -/
structure WallDimensions where
  height : ℕ
  width : ℕ

/-- Represents the cleaning progress and rate -/
structure CleaningProgress where
  totalArea : ℕ
  cleanedArea : ℕ
  timeSpent : ℕ

/-- Calculates the time needed to clean the remaining area -/
def timeToCleanRemaining (wall : WallDimensions) (progress : CleaningProgress) : ℕ :=
  let remainingArea := wall.height * wall.width - progress.cleanedArea
  (remainingArea * progress.timeSpent) / progress.cleanedArea

/-- Theorem: Given the wall dimensions and cleaning progress, 
    the time to clean the remaining area is 161 minutes -/
theorem time_to_clean_wall 
  (wall : WallDimensions) 
  (progress : CleaningProgress) 
  (h1 : wall.height = 6) 
  (h2 : wall.width = 12) 
  (h3 : progress.totalArea = wall.height * wall.width)
  (h4 : progress.cleanedArea = 9)
  (h5 : progress.timeSpent = 23) :
  timeToCleanRemaining wall progress = 161 := by
  sorry

end time_to_clean_wall_l2354_235448


namespace smallest_m_no_real_roots_l2354_235483

theorem smallest_m_no_real_roots : 
  ∀ m : ℤ, (∀ x : ℝ, 3*x*(m*x-5) - 2*x^2 + 7 ≠ 0) → m ≥ 4 :=
by sorry

end smallest_m_no_real_roots_l2354_235483


namespace modular_inverse_3_mod_197_l2354_235466

theorem modular_inverse_3_mod_197 :
  ∃ x : ℕ, x < 197 ∧ (3 * x) % 197 = 1 :=
by
  use 66
  sorry

end modular_inverse_3_mod_197_l2354_235466


namespace vector_operation_equals_two_l2354_235451

-- Define the vectors
def a : ℝ × ℝ := (4, 2)
def b : ℝ × ℝ := (1, -1)

-- Define the dot product operation
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

-- Define vector scalar multiplication
def scalar_mult (c : ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (c * v.1, c * v.2)

-- Define vector subtraction
def vector_sub (v w : ℝ × ℝ) : ℝ × ℝ :=
  (v.1 - w.1, v.2 - w.2)

-- Theorem statement
theorem vector_operation_equals_two :
  dot_product (vector_sub (scalar_mult 2 a) b) b = 2 := by
  sorry

end vector_operation_equals_two_l2354_235451


namespace employee_pay_l2354_235450

theorem employee_pay (total_pay x y : ℝ) : 
  total_pay = 880 ∧ x = 1.2 * y → y = 400 := by
  sorry

end employee_pay_l2354_235450


namespace smallest_perfect_square_divisible_l2354_235408

theorem smallest_perfect_square_divisible (n : ℕ) (h : n = 14400) :
  (∃ k : ℕ, k * k = n ∧ n / 5 = 2880) ∧
  (∀ m : ℕ, m < n → m / 5 = 2880 → ¬∃ j : ℕ, j * j = m) :=
by sorry

end smallest_perfect_square_divisible_l2354_235408


namespace solve_system_with_partial_info_l2354_235497

/-- Given a system of linear equations and information about its solutions,
    this theorem proves the values of the coefficients. -/
theorem solve_system_with_partial_info :
  ∀ (a b c : ℚ),
  (∀ x y : ℚ, a*x + b*y = 2 ∧ c*x - 3*y = -2 → x = 1 ∧ y = -1) →
  (a*2 + b*(-6) = 2) →
  (a = 5/2 ∧ b = 1/2 ∧ c = -5) :=
by sorry

end solve_system_with_partial_info_l2354_235497


namespace one_positive_real_solution_l2354_235442

/-- The polynomial function f(x) = x^11 + 5x^10 + 20x^9 + 1000x^8 - 800x^7 -/
def f (x : ℝ) : ℝ := x^11 + 5*x^10 + 20*x^9 + 1000*x^8 - 800*x^7

/-- Theorem stating that f(x) = 0 has exactly one positive real solution -/
theorem one_positive_real_solution : ∃! (x : ℝ), x > 0 ∧ f x = 0 := by
  sorry

end one_positive_real_solution_l2354_235442


namespace major_premise_wrong_l2354_235403

theorem major_premise_wrong : ¬ ∀ a b : ℝ, a > b → a^2 > b^2 := by
  sorry

end major_premise_wrong_l2354_235403


namespace bob_profit_is_1600_l2354_235407

/-- Calculates the total profit from selling puppies given the number of show dogs bought,
    cost per show dog, number of puppies, and selling price per puppy. -/
def calculate_profit (num_dogs : ℕ) (cost_per_dog : ℚ) (num_puppies : ℕ) (price_per_puppy : ℚ) : ℚ :=
  num_puppies * price_per_puppy - num_dogs * cost_per_dog

/-- Proves that Bob's total profit from selling puppies is $1,600.00 -/
theorem bob_profit_is_1600 :
  calculate_profit 2 250 6 350 = 1600 := by
  sorry

end bob_profit_is_1600_l2354_235407


namespace product_inequality_l2354_235485

theorem product_inequality (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (sum_eq_one : a + b + c = 1) :
  (1 + a) * (1 + b) * (1 + c) ≥ 8 * (1 - a) * (1 - b) * (1 - c) := by
  sorry

end product_inequality_l2354_235485


namespace fixed_point_of_exponential_function_l2354_235478

theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  ∃ x y : ℝ, x = 2 ∧ y = -2 ∧ ∀ a : ℝ, a > 0 → a ≠ 1 → a^(x - 2) - 3 = y :=
sorry

end fixed_point_of_exponential_function_l2354_235478


namespace fathers_age_l2354_235455

theorem fathers_age (son_age father_age : ℕ) : 
  father_age = 3 * son_age →
  father_age + 15 = 2 * (son_age + 15) →
  father_age = 45 := by
sorry

end fathers_age_l2354_235455


namespace equilateral_pyramid_volume_l2354_235495

/-- A pyramid with an equilateral triangle base -/
structure EquilateralPyramid where
  -- The side length of the base triangle
  base_side : ℝ
  -- The angle between two edges from the apex to the base
  apex_angle : ℝ

/-- The volume of an equilateral pyramid -/
noncomputable def volume (p : EquilateralPyramid) : ℝ :=
  (Real.sqrt 3 / 9) * (2 / 3 * Real.sqrt 3 + 1 / Real.tan (p.apex_angle / 2))

/-- Theorem: The volume of a specific equilateral pyramid -/
theorem equilateral_pyramid_volume :
    ∀ (p : EquilateralPyramid),
      p.base_side = 2 →
      volume p = (Real.sqrt 3 / 9) * (2 / 3 * Real.sqrt 3 + 1 / Real.tan (p.apex_angle / 2)) :=
by
  sorry

end equilateral_pyramid_volume_l2354_235495


namespace parabola_focus_and_intersection_l2354_235452

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x

-- Define the line
def line (n : ℝ) (x y : ℝ) : Prop := x = Real.sqrt 3 * y + n

-- Define the point E
def point_E : ℝ × ℝ := (4, 4)

-- Define the point D
def point_D (n : ℝ) : ℝ × ℝ := (n, 0)

-- Theorem statement
theorem parabola_focus_and_intersection
  (p : ℝ)
  (n : ℝ)
  (h1 : parabola p (point_E.1) (point_E.2))
  (h2 : ∃ (A B : ℝ × ℝ), A ≠ B ∧ A ≠ point_E ∧ B ≠ point_E ∧
        parabola p A.1 A.2 ∧ parabola p B.1 B.2 ∧
        line n A.1 A.2 ∧ line n B.1 B.2)
  (h3 : ∃ (A B : ℝ × ℝ), 
        parabola p A.1 A.2 ∧ parabola p B.1 B.2 ∧
        line n A.1 A.2 ∧ line n B.1 B.2 ∧
        (A.1 - (point_D n).1)^2 + A.2^2 * ((B.1 - (point_D n).1)^2 + B.2^2) = 64) :
  (p = 2 ∧ n = 4) :=
sorry

end parabola_focus_and_intersection_l2354_235452


namespace x_intercept_after_rotation_l2354_235496

/-- The equation of a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Rotate a line 90 degrees counterclockwise about a given point -/
def rotate90 (l : Line) (p : Point) : Line := sorry

/-- Find the x-intercept of a line -/
def xIntercept (l : Line) : ℝ := sorry

theorem x_intercept_after_rotation :
  let l : Line := { a := 2, b := -3, c := 30 }
  let p : Point := { x := 15, y := 10 }
  let k' := rotate90 l p
  xIntercept k' = 65 / 3 := by sorry

end x_intercept_after_rotation_l2354_235496


namespace total_cash_realized_eq_17364_82065_l2354_235491

/-- Calculates the total cash realized in INR from selling four stocks -/
def total_cash_realized (stock1_value stock1_brokerage stock2_value stock2_brokerage : ℚ)
                        (stock3_value stock3_brokerage stock4_value stock4_brokerage : ℚ)
                        (usd_to_inr : ℚ) : ℚ :=
  let stock1_realized := stock1_value * (1 - stock1_brokerage / 100)
  let stock2_realized := stock2_value * (1 - stock2_brokerage / 100)
  let stock3_realized := stock3_value * (1 - stock3_brokerage / 100) * usd_to_inr
  let stock4_realized := stock4_value * (1 - stock4_brokerage / 100) * usd_to_inr
  stock1_realized + stock2_realized + stock3_realized + stock4_realized

/-- Theorem stating that the total cash realized is equal to 17364.82065 INR -/
theorem total_cash_realized_eq_17364_82065 :
  total_cash_realized 120.50 (1/4) 210.75 0.5 80.90 0.3 150.55 0.65 74 = 17364.82065 := by
  sorry

end total_cash_realized_eq_17364_82065_l2354_235491


namespace probability_more_than_third_correct_l2354_235429

-- Define the number of questions
def n : ℕ := 12

-- Define the probability of guessing correctly
def p : ℚ := 1/2

-- Define the minimum number of correct answers needed
def k : ℕ := 5

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ := sorry

-- Define the probability of getting at least k correct answers
def prob_at_least_k (n k : ℕ) (p : ℚ) : ℚ := sorry

-- State the theorem
theorem probability_more_than_third_correct :
  prob_at_least_k n k p = 825/1024 := by sorry

end probability_more_than_third_correct_l2354_235429


namespace a_can_be_any_real_l2354_235486

theorem a_can_be_any_real (a b c d e : ℝ) (h1 : b * e ≠ 0) (h2 : a / b < c / b - d / e) :
  ∃ (x y z : ℝ), x > 0 ∧ y < 0 ∧ z = 0 ∧
  (a = x ∨ a = y ∨ a = z) :=
sorry

end a_can_be_any_real_l2354_235486


namespace pam_has_ten_bags_l2354_235402

/-- The number of apples in one of Gerald's bags -/
def geralds_bag_count : ℕ := 40

/-- The number of Gerald's bags equivalent to one of Pam's bags -/
def bags_ratio : ℕ := 3

/-- The total number of apples Pam has -/
def pams_total_apples : ℕ := 1200

/-- The number of bags Pam has -/
def pams_bag_count : ℕ := pams_total_apples / (bags_ratio * geralds_bag_count)

theorem pam_has_ten_bags : pams_bag_count = 10 := by
  sorry

end pam_has_ten_bags_l2354_235402


namespace t_divides_t_2n_plus_1_l2354_235435

def t : ℕ → ℕ
  | 0 => 1  -- Assuming t_0 = 1 for completeness
  | 1 => 2
  | 2 => 5
  | (n + 3) => 2 * t (n + 2) + t (n + 1)

theorem t_divides_t_2n_plus_1 (n : ℕ) : t n ∣ t (2 * n + 1) := by
  sorry

end t_divides_t_2n_plus_1_l2354_235435


namespace unique_solution_quadratic_l2354_235481

/-- 
If 3x^2 - 7x + m = 0 is a quadratic equation with exactly one solution for x, 
then m = 49/12.
-/
theorem unique_solution_quadratic (m : ℚ) : 
  (∃! x : ℚ, 3 * x^2 - 7 * x + m = 0) → m = 49 / 12 := by
  sorry

end unique_solution_quadratic_l2354_235481


namespace shopping_expenditure_l2354_235444

theorem shopping_expenditure (x : ℝ) 
  (h1 : x + 10 + 40 = 100) 
  (h2 : 0.04 * x + 0.08 * 40 = 5.2) : x = 50 := by
  sorry

end shopping_expenditure_l2354_235444


namespace g_150_zeros_l2354_235447

-- Define g₀
def g₀ (x : ℝ) : ℝ := x + |x - 200| - |x + 200|

-- Define gₙ recursively
def g (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0 => g₀ x
  | n + 1 => |g n x| - 1

-- Theorem statement
theorem g_150_zeros :
  ∃! (s : Finset ℝ), s.card = 4 ∧ ∀ x : ℝ, x ∈ s ↔ g 150 x = 0 :=
sorry

end g_150_zeros_l2354_235447


namespace positive_solutions_x_minus_y_nonnegative_l2354_235463

-- Define the system of linear equations
def system (x y m : ℝ) : Prop :=
  x + y = 3 * m ∧ 2 * x - 3 * y = m + 5

-- Part 1: Positive solutions
theorem positive_solutions (m : ℝ) :
  (∃ x y : ℝ, system x y m ∧ x > 0 ∧ y > 0) → m > 1 := by
  sorry

-- Part 2: x - y ≥ 0
theorem x_minus_y_nonnegative (m : ℝ) :
  (∃ x y : ℝ, system x y m ∧ x - y ≥ 0) → m ≥ -2 := by
  sorry

end positive_solutions_x_minus_y_nonnegative_l2354_235463


namespace miss_stevie_payment_l2354_235439

def jerry_painting_hours : ℕ := 8
def jerry_painting_rate : ℚ := 15
def jerry_mowing_hours : ℕ := 6
def jerry_mowing_rate : ℚ := 10
def jerry_plumbing_hours : ℕ := 4
def jerry_plumbing_rate : ℚ := 18
def jerry_discount : ℚ := 0.1

def randy_painting_hours : ℕ := 7
def randy_painting_rate : ℚ := 12
def randy_mowing_hours : ℕ := 4
def randy_mowing_rate : ℚ := 8
def randy_electrical_hours : ℕ := 3
def randy_electrical_rate : ℚ := 20
def randy_discount : ℚ := 0.05

def total_payment : ℚ := 394

theorem miss_stevie_payment :
  let jerry_total := (jerry_painting_hours * jerry_painting_rate +
                      jerry_mowing_hours * jerry_mowing_rate +
                      jerry_plumbing_hours * jerry_plumbing_rate) * (1 - jerry_discount)
  let randy_total := (randy_painting_hours * randy_painting_rate +
                      randy_mowing_hours * randy_mowing_rate +
                      randy_electrical_hours * randy_electrical_rate) * (1 - randy_discount)
  jerry_total + randy_total = total_payment := by
    sorry

end miss_stevie_payment_l2354_235439


namespace horner_method_v2_value_l2354_235414

def f (x : ℝ) : ℝ := 4*x^4 + 3*x^3 - 6*x^2 + x - 1

def horner_v2 (a₄ a₃ a₂ a₁ a₀ x : ℝ) : ℝ :=
  let v₀ := a₄
  let v₁ := v₀ * x + a₃
  v₁ * x + a₂

theorem horner_method_v2_value :
  horner_v2 4 3 (-6) 1 (-1) (-1) = -5 :=
by sorry

end horner_method_v2_value_l2354_235414


namespace existence_of_solution_l2354_235420

theorem existence_of_solution : ∃ (a b c d : ℕ+), 
  (a^3 + b^4 + c^5 = d^11) ∧ (a * b * c < 10^5) := by
  sorry

end existence_of_solution_l2354_235420


namespace fast_pulsar_period_scientific_notation_l2354_235400

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  significand : ℝ
  exponent : ℤ
  is_normalized : 1 ≤ |significand| ∧ |significand| < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem fast_pulsar_period_scientific_notation :
  toScientificNotation 0.00519 = ScientificNotation.mk 5.19 (-3) sorry := by
  sorry

end fast_pulsar_period_scientific_notation_l2354_235400


namespace g_negative_one_equals_three_l2354_235490

-- Define an odd function
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the theorem
theorem g_negative_one_equals_three
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_g_def : ∀ x, g x = f x + 2)
  (h_g_one : g 1 = 1) :
  g (-1) = 3 := by
  sorry

end g_negative_one_equals_three_l2354_235490


namespace circle_radius_is_zero_l2354_235418

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  2 * x^2 - 8 * x + 2 * y^2 + 4 * y + 10 = 0

/-- The radius of the circle -/
def circle_radius : ℝ := 0

/-- Theorem: The radius of the circle defined by the equation is 0 -/
theorem circle_radius_is_zero :
  ∀ x y : ℝ, circle_equation x y → ∃ h k : ℝ, (x - h)^2 + (y - k)^2 = circle_radius^2 :=
by sorry

end circle_radius_is_zero_l2354_235418


namespace f_derivative_at_zero_l2354_235449

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then x^2 * Real.exp (abs x) * Real.sin (1 / x^2) else 0

theorem f_derivative_at_zero :
  deriv f 0 = 0 := by sorry

end f_derivative_at_zero_l2354_235449


namespace y1_greater_y2_l2354_235475

/-- Given two points A(m-1, y₁) and B(m, y₂) on the line y = -2x + 1, prove that y₁ > y₂ -/
theorem y1_greater_y2 (m : ℝ) (y₁ y₂ : ℝ) 
  (hA : y₁ = -2 * (m - 1) + 1) 
  (hB : y₂ = -2 * m + 1) : 
  y₁ > y₂ := by
  sorry

end y1_greater_y2_l2354_235475


namespace consecutive_pairs_49_6_l2354_235438

/-- The number of ways to choose 6 elements among the first 49 positive integers
    with at least two consecutive elements -/
def consecutivePairs (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k - Nat.choose (n - k + 1) k

theorem consecutive_pairs_49_6 :
  consecutivePairs 49 6 = Nat.choose 49 6 - Nat.choose 44 6 := by
  sorry

end consecutive_pairs_49_6_l2354_235438


namespace thirtieth_triangular_number_l2354_235476

/-- Triangular number function -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The 30th triangular number represents the total number of coins in a stack with 30 layers -/
theorem thirtieth_triangular_number : triangular_number 30 = 465 := by
  sorry

end thirtieth_triangular_number_l2354_235476


namespace fruit_bowl_total_l2354_235409

/-- Represents the number of pieces of each type of fruit in the bowl -/
structure FruitBowl where
  apples : ℕ
  pears : ℕ
  bananas : ℕ

/-- Theorem stating the total number of fruits in the bowl under given conditions -/
theorem fruit_bowl_total (bowl : FruitBowl) 
  (h1 : bowl.pears = bowl.apples + 2)
  (h2 : bowl.bananas = bowl.pears + 3)
  (h3 : bowl.bananas = 9) : 
  bowl.apples + bowl.pears + bowl.bananas = 19 := by
  sorry

end fruit_bowl_total_l2354_235409


namespace intersection_points_theorem_l2354_235436

-- Define the three lines
def line1 (x y : ℝ) : Prop := x + y + 1 = 0
def line2 (x y : ℝ) : Prop := 2*x - y + 8 = 0
def line3 (a x y : ℝ) : Prop := a*x + 3*y - 5 = 0

-- Define the condition of at most 2 intersection points
def at_most_two_intersections (a : ℝ) : Prop :=
  ∀ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    (line1 x₁ y₁ ∧ line2 x₁ y₁ ∧ line3 a x₁ y₁) →
    (line1 x₂ y₂ ∧ line2 x₂ y₂ ∧ line3 a x₂ y₂) →
    (line1 x₃ y₃ ∧ line2 x₃ y₃ ∧ line3 a x₃ y₃) →
    ((x₁ = x₂ ∧ y₁ = y₂) ∨ (x₁ = x₃ ∧ y₁ = y₃) ∨ (x₂ = x₃ ∧ y₂ = y₃))

-- The theorem statement
theorem intersection_points_theorem :
  ∀ a : ℝ, at_most_two_intersections a ↔ (a = -3 ∨ a = -6) :=
by sorry

end intersection_points_theorem_l2354_235436


namespace tangent_circle_min_radius_l2354_235467

noncomputable section

-- Define the curve C
def C (x y : ℝ) : Prop := y^2 = 4*x

-- Define the point P on curve C
def P (x₀ y₀ : ℝ) : Prop := C x₀ y₀ ∧ y₀ > 0

-- Define the line l tangent to C at P
def l (x₀ y₀ k : ℝ) (x y : ℝ) : Prop := y - y₀ = k * (x - x₀)

-- Define the circle M centered at (a, 0)
def M (a r : ℝ) (x y : ℝ) : Prop := (x - a)^2 + y^2 = r^2

-- Main theorem
theorem tangent_circle_min_radius (a x₀ y₀ k r : ℝ) :
  a > 2 →
  P x₀ y₀ →
  (∀ x y, C x y → l x₀ y₀ k x y → x = x₀ ∧ y = y₀) →
  (∃ x y, l x₀ y₀ k x y ∧ M a r x y) →
  (∀ r' : ℝ, (∃ x y, l x₀ y₀ k x y ∧ M a r' x y) → r ≤ r') →
  a - x₀ = 2 :=
sorry

end tangent_circle_min_radius_l2354_235467


namespace amelia_win_probability_l2354_235441

/-- The probability of Amelia winning the coin toss game -/
def ameliaWinProbability (ameliaHeadProb blainHeadProb : ℚ) : ℚ :=
  ameliaHeadProb / (1 - (1 - ameliaHeadProb) * (1 - blainHeadProb))

/-- The coin toss game where Amelia goes first -/
theorem amelia_win_probability :
  let ameliaHeadProb : ℚ := 1/3
  let blaineHeadProb : ℚ := 2/5
  ameliaWinProbability ameliaHeadProb blaineHeadProb = 5/9 := by
  sorry

#eval ameliaWinProbability (1/3) (2/5)

end amelia_win_probability_l2354_235441


namespace cat_adoptions_correct_l2354_235424

/-- The number of families who adopted cats at an animal shelter event -/
def num_cat_adoptions : ℕ := 3

/-- Vet fees for dogs in dollars -/
def dog_fee : ℕ := 15

/-- Vet fees for cats in dollars -/
def cat_fee : ℕ := 13

/-- Number of families who adopted dogs -/
def num_dog_adoptions : ℕ := 8

/-- The fraction of fees donated back to the shelter -/
def donation_fraction : ℚ := 1/3

/-- The amount donated back to the shelter in dollars -/
def donation_amount : ℕ := 53

theorem cat_adoptions_correct : 
  (num_dog_adoptions * dog_fee + num_cat_adoptions * cat_fee) * donation_fraction = donation_amount :=
sorry

end cat_adoptions_correct_l2354_235424


namespace part_one_part_two_l2354_235470

-- Define polynomials A, B, and C
def A (x y : ℝ) : ℝ := x^2 + x*y + 3*y
def B (x y : ℝ) : ℝ := x^2 - x*y

-- Theorem for part 1
theorem part_one (x y : ℝ) : 3 * A x y - B x y = 2*x^2 + 4*x*y + 9*y := by sorry

-- Theorem for part 2
theorem part_two (x y : ℝ) :
  (∃ C : ℝ → ℝ → ℝ, A x y + (1/3) * C x y = 2*x*y + 5*y) →
  (∃ C : ℝ → ℝ → ℝ, C x y = -3*x^2 + 3*x*y + 6*y) := by sorry

end part_one_part_two_l2354_235470


namespace log_equation_l2354_235411

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_equation : (log10 5)^2 + log10 2 * log10 50 = 1 := by
  sorry

end log_equation_l2354_235411


namespace max_m_value_l2354_235404

theorem max_m_value (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ m : ℝ, m / (3 * a + b) - 3 / a - 1 / b ≤ 0) →
  (∃ m : ℝ, m = 16 ∧ ∀ n : ℝ, (n / (3 * a + b) - 3 / a - 1 / b ≤ 0) → n ≤ m) :=
by sorry

end max_m_value_l2354_235404


namespace initial_candies_l2354_235477

theorem initial_candies (eaten : ℕ) (left : ℕ) (h1 : eaten = 15) (h2 : left = 13) :
  eaten + left = 28 := by
  sorry

end initial_candies_l2354_235477


namespace product_of_roots_l2354_235494

theorem product_of_roots (x : ℝ) : (x + 3) * (x - 5) = 18 → ∃ y : ℝ, (x + 3) * (x - 5) = 18 ∧ (y + 3) * (y - 5) = 18 ∧ x * y = -33 := by
  sorry

end product_of_roots_l2354_235494


namespace min_value_trig_sum_l2354_235425

theorem min_value_trig_sum (θ : ℝ) : 
  1 / (2 - Real.cos θ ^ 2) + 1 / (2 - Real.sin θ ^ 2) ≥ 4 / 3 :=
sorry

end min_value_trig_sum_l2354_235425
