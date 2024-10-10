import Mathlib

namespace range_of_c_over_a_l477_47798

theorem range_of_c_over_a (a b c : ℝ) 
  (h1 : a > b) (h2 : b > c) (h3 : a + 2*b + c = 0) :
  ∃ (x : ℝ), -3 < x ∧ x < -1/3 ∧ x = c/a :=
sorry

end range_of_c_over_a_l477_47798


namespace simple_interest_problem_l477_47771

/-- Calculates the total amount after simple interest --/
def simpleInterestTotal (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Proves that given the initial conditions, the total amount after 7 years is $850 --/
theorem simple_interest_problem (initialSum : ℝ) (totalAfter2Years : ℝ) :
  initialSum = 500 →
  totalAfter2Years = 600 →
  ∃ (rate : ℝ),
    simpleInterestTotal initialSum rate 2 = totalAfter2Years ∧
    simpleInterestTotal initialSum rate 7 = 850 := by
  sorry

/-- The solution to the problem --/
def solution : ℝ := 850

end simple_interest_problem_l477_47771


namespace fruit_shop_results_l477_47710

/-- Represents the fruit inventory and pricing information for a shopkeeper --/
structure FruitShop where
  totalFruits : Nat
  oranges : Nat
  bananas : Nat
  apples : Nat
  rottenOrangesPercent : Rat
  rottenBananasPercent : Rat
  rottenApplesPercent : Rat
  orangePurchasePrice : Rat
  bananaPurchasePrice : Rat
  applePurchasePrice : Rat
  orangeSellingPrice : Rat
  bananaSellingPrice : Rat
  appleSellingPrice : Rat

/-- Calculates the percentage of fruits in good condition and the overall profit --/
def calculateResults (shop : FruitShop) : (Rat × Rat) :=
  sorry

/-- Theorem stating the correct percentage of good fruits and overall profit --/
theorem fruit_shop_results (shop : FruitShop) 
  (h1 : shop.totalFruits = 1000)
  (h2 : shop.oranges = 600)
  (h3 : shop.bananas = 300)
  (h4 : shop.apples = 100)
  (h5 : shop.rottenOrangesPercent = 15/100)
  (h6 : shop.rottenBananasPercent = 8/100)
  (h7 : shop.rottenApplesPercent = 20/100)
  (h8 : shop.orangePurchasePrice = 60/100)
  (h9 : shop.bananaPurchasePrice = 30/100)
  (h10 : shop.applePurchasePrice = 1)
  (h11 : shop.orangeSellingPrice = 120/100)
  (h12 : shop.bananaSellingPrice = 60/100)
  (h13 : shop.appleSellingPrice = 150/100) :
  calculateResults shop = (866/1000, 3476/10) := by
  sorry


end fruit_shop_results_l477_47710


namespace probability_product_not_odd_l477_47714

/-- Represents a standard six-sided die -/
def Die : Type := Fin 6

/-- The set of all possible outcomes when rolling two dice -/
def TwoDiceOutcomes : Type := Die × Die

/-- Predicate to check if a number is odd -/
def isOdd (n : ℕ) : Prop := n % 2 = 1

/-- Predicate to check if the product of two die rolls is not odd -/
def productNotOdd (roll : TwoDiceOutcomes) : Prop :=
  ¬isOdd ((roll.1.val + 1) * (roll.2.val + 1))

/-- The total number of possible outcomes when rolling two dice -/
def totalOutcomes : ℕ := 36

/-- The number of outcomes where the product is not odd -/
def favorableOutcomes : ℕ := 27

theorem probability_product_not_odd :
  (favorableOutcomes : ℚ) / totalOutcomes = 3 / 4 := by
  sorry

end probability_product_not_odd_l477_47714


namespace equation_solution_l477_47738

theorem equation_solution (x : ℚ) (h1 : x ≠ 3) (h2 : x ≠ -2) :
  (x + 4) / (x - 3) = (x - 2) / (x + 2) ↔ x = -2 / 11 := by
  sorry

end equation_solution_l477_47738


namespace even_digits_in_base7_of_315_l477_47705

/-- Converts a natural number from base 10 to base 7 --/
def toBase7 (n : ℕ) : List ℕ :=
  sorry

/-- Counts the number of even digits in a list of digits --/
def countEvenDigits (digits : List ℕ) : ℕ :=
  sorry

/-- Checks if a digit is even in base 7 --/
def isEvenInBase7 (digit : ℕ) : Bool :=
  sorry

theorem even_digits_in_base7_of_315 :
  let base7Repr := toBase7 315
  countEvenDigits (base7Repr.filter isEvenInBase7) = 2 := by
  sorry

end even_digits_in_base7_of_315_l477_47705


namespace dice_sum_product_l477_47792

theorem dice_sum_product (a b c d : ℕ) : 
  1 ≤ a ∧ a ≤ 6 →
  1 ≤ b ∧ b ≤ 6 →
  1 ≤ c ∧ c ≤ 6 →
  1 ≤ d ∧ d ≤ 6 →
  a * b * c * d = 360 →
  a + b + c + d ≠ 20 := by
sorry

end dice_sum_product_l477_47792


namespace instantaneous_velocity_at_3_l477_47729

-- Define the position function
def s (t : ℝ) : ℝ := 3 * t^2

-- Define the velocity function as the derivative of the position function
noncomputable def v (t : ℝ) : ℝ := deriv s t

-- Theorem statement
theorem instantaneous_velocity_at_3 : v 3 = 18 := by
  sorry

end instantaneous_velocity_at_3_l477_47729


namespace hospital_staff_count_l477_47740

theorem hospital_staff_count (total : ℕ) (ratio_doctors : ℕ) (ratio_nurses : ℕ) 
  (h1 : total = 250) 
  (h2 : ratio_doctors = 2) 
  (h3 : ratio_nurses = 3) : 
  (ratio_nurses * total) / (ratio_doctors + ratio_nurses) = 150 := by
  sorry

end hospital_staff_count_l477_47740


namespace coin_position_determinable_l477_47766

-- Define the coin values
def left_coin : ℕ := 10
def right_coin : ℕ := 15

-- Define the possible multipliers
def left_multipliers : List ℕ := [4, 10, 12, 26]
def right_multipliers : List ℕ := [7, 13, 21, 35]

-- Define a function to check if a number is even
def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

-- Define the possible configurations
structure Configuration :=
  (left_value : ℕ)
  (right_value : ℕ)
  (left_multiplier : ℕ)
  (right_multiplier : ℕ)

-- Define the theorem
theorem coin_position_determinable :
  ∀ (c : Configuration),
  c.left_value ∈ [left_coin, right_coin] ∧
  c.right_value ∈ [left_coin, right_coin] ∧
  c.left_value ≠ c.right_value ∧
  c.left_multiplier ∈ left_multipliers ∧
  c.right_multiplier ∈ right_multipliers →
  (is_even (c.left_value * c.left_multiplier + c.right_value * c.right_multiplier) ↔
   c.right_value = left_coin) :=
by sorry

end coin_position_determinable_l477_47766


namespace routes_equal_choose_l477_47744

/-- The number of routes in a 3x2 grid from top-left to bottom-right -/
def num_routes : ℕ := 10

/-- The number of ways to choose 2 items from a set of 5 items -/
def choose_2_from_5 : ℕ := Nat.choose 5 2

/-- Theorem stating that the number of routes is equal to choosing 2 from 5 -/
theorem routes_equal_choose :
  num_routes = choose_2_from_5 := by sorry

end routes_equal_choose_l477_47744


namespace square_equals_25_l477_47736

theorem square_equals_25 : {x : ℝ | x^2 = 25} = {-5, 5} := by sorry

end square_equals_25_l477_47736


namespace library_visitors_l477_47702

/-- Proves that the average number of visitors on non-Sunday days is 240 --/
theorem library_visitors (sunday_visitors : ℕ) (total_days : ℕ) (sundays : ℕ) (avg_visitors : ℕ) :
  sunday_visitors = 510 →
  total_days = 30 →
  sundays = 5 →
  avg_visitors = 285 →
  (sundays * sunday_visitors + (total_days - sundays) * 
    ((total_days * avg_visitors - sundays * sunday_visitors) / (total_days - sundays))) 
    / total_days = avg_visitors →
  (total_days * avg_visitors - sundays * sunday_visitors) / (total_days - sundays) = 240 := by
sorry

#eval (30 * 285 - 5 * 510) / (30 - 5)  -- Should output 240

end library_visitors_l477_47702


namespace probability_is_one_sixth_l477_47762

/-- Represents the duration of each color in the traffic light cycle -/
structure TrafficLightCycle where
  green : ℕ
  yellow : ℕ
  red : ℕ

/-- Calculates the probability of observing a color change in a traffic light
    during a randomly chosen five-second interval -/
def probabilityOfColorChange (cycle : TrafficLightCycle) : ℚ :=
  let totalCycleDuration := cycle.green + cycle.yellow + cycle.red
  let favorableDuration := 15 -- 5 seconds before each color change
  favorableDuration / totalCycleDuration

/-- The specific traffic light cycle from the problem -/
def problemCycle : TrafficLightCycle :=
  { green := 45
  , yellow := 5
  , red := 40 }

theorem probability_is_one_sixth :
  probabilityOfColorChange problemCycle = 1 / 6 := by
  sorry

#eval probabilityOfColorChange problemCycle

end probability_is_one_sixth_l477_47762


namespace car_travel_time_l477_47750

/-- Given a car's initial travel and additional distance, calculate the total travel time. -/
theorem car_travel_time (initial_distance : ℝ) (initial_time : ℝ) (additional_distance : ℝ) 
  (h1 : initial_distance = 180) 
  (h2 : initial_time = 4)
  (h3 : additional_distance = 135) :
  let speed := initial_distance / initial_time
  let additional_time := additional_distance / speed
  initial_time + additional_time = 7 := by sorry

end car_travel_time_l477_47750


namespace midpoint_of_specific_segment_l477_47720

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- Calculate the midpoint of two points in polar coordinates -/
def polarMidpoint (p1 p2 : PolarPoint) : PolarPoint :=
  sorry

theorem midpoint_of_specific_segment :
  let p1 : PolarPoint := ⟨10, π/4⟩
  let p2 : PolarPoint := ⟨10, 3*π/4⟩
  let midpoint := polarMidpoint p1 p2
  midpoint.r = 5 * Real.sqrt 2 ∧ midpoint.θ = π/2 := by
  sorry

end midpoint_of_specific_segment_l477_47720


namespace brenda_final_lead_l477_47775

theorem brenda_final_lead (initial_lead : ℕ) (brenda_play : ℕ) (david_play : ℕ) : 
  initial_lead = 22 → brenda_play = 15 → david_play = 32 → 
  initial_lead + brenda_play - david_play = 5 := by
  sorry

end brenda_final_lead_l477_47775


namespace fifty_percent_x_equals_690_l477_47786

theorem fifty_percent_x_equals_690 : ∃ x : ℝ, (0.5 * x = 0.25 * 1500 - 30) ∧ (x = 690) := by
  sorry

end fifty_percent_x_equals_690_l477_47786


namespace exists_n_with_uniform_200th_digit_distribution_l477_47754

def digit_at_position (x : ℝ) (pos : ℕ) : ℕ := sorry

def count_occurrences (digit : ℕ) (numbers : List ℝ) (pos : ℕ) : ℕ := sorry

theorem exists_n_with_uniform_200th_digit_distribution :
  ∃ (n : ℕ+),
    ∀ (digit : Fin 10),
      count_occurrences digit.val
        (List.map (λ k => Real.sqrt (n.val + k)) (List.range 1000))
        200 = 100 := by
  sorry

end exists_n_with_uniform_200th_digit_distribution_l477_47754


namespace youngest_child_age_l477_47711

/-- Given a family where:
    1. 10 years ago, the average age of 4 members was 24 years
    2. Two children were born with an age difference of 2 years
    3. The present average age of the family (now 6 members) is still 24 years
    Prove that the present age of the youngest child is 3 years -/
theorem youngest_child_age
  (past_average_age : ℕ)
  (past_family_size : ℕ)
  (years_passed : ℕ)
  (present_average_age : ℕ)
  (present_family_size : ℕ)
  (age_difference : ℕ)
  (h1 : past_average_age = 24)
  (h2 : past_family_size = 4)
  (h3 : years_passed = 10)
  (h4 : present_average_age = 24)
  (h5 : present_family_size = 6)
  (h6 : age_difference = 2) :
  ∃ (youngest_age : ℕ), youngest_age = 3 ∧
    present_average_age * present_family_size =
    (past_average_age * past_family_size + years_passed * past_family_size + youngest_age + (youngest_age + age_difference)) :=
by sorry

end youngest_child_age_l477_47711


namespace parcel_cost_formula_l477_47767

def parcel_cost (W : ℕ) : ℕ :=
  if W ≤ 10 then 5 * W + 10 else 7 * W - 10

theorem parcel_cost_formula (W : ℕ) :
  (W ≤ 10 → parcel_cost W = 5 * W + 10) ∧
  (W > 10 → parcel_cost W = 7 * W - 10) := by
  sorry

end parcel_cost_formula_l477_47767


namespace triangle_and_star_operations_l477_47723

-- Define the triangle operation
def triangle (a b : ℚ) : ℚ := a^2 - a * b

-- Define the star operation
def star (a b : ℚ) : ℚ := 3 * a * b - b^2

theorem triangle_and_star_operations : 
  (triangle (-3 : ℚ) 5 = 24) ∧ 
  (star (-4 : ℚ) (triangle 2 3) = 20) := by
  sorry

end triangle_and_star_operations_l477_47723


namespace units_digit_of_50_factorial_l477_47789

theorem units_digit_of_50_factorial (n : ℕ) : n = 50 → (n.factorial % 10 = 0) := by
  sorry

end units_digit_of_50_factorial_l477_47789


namespace meeting_attendees_l477_47707

theorem meeting_attendees (total_handshakes : ℕ) (h : total_handshakes = 66) :
  ∃ (n : ℕ), n * (n - 1) / 2 = total_handshakes ∧ n = 12 := by
  sorry

end meeting_attendees_l477_47707


namespace burgers_spending_l477_47769

def total_allowance : ℚ := 50

def movie_fraction : ℚ := 2 / 5
def video_game_fraction : ℚ := 1 / 10
def book_fraction : ℚ := 1 / 4

def spent_on_movies : ℚ := movie_fraction * total_allowance
def spent_on_video_games : ℚ := video_game_fraction * total_allowance
def spent_on_books : ℚ := book_fraction * total_allowance

def total_spent : ℚ := spent_on_movies + spent_on_video_games + spent_on_books

def remaining_for_burgers : ℚ := total_allowance - total_spent

theorem burgers_spending :
  remaining_for_burgers = 12.5 := by sorry

end burgers_spending_l477_47769


namespace remainder_of_sum_is_zero_l477_47703

-- Define the arithmetic sequence
def arithmeticSequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

-- Define the sum of the arithmetic sequence
def sumArithmeticSequence (a₁ : ℕ) (aₙ : ℕ) (n : ℕ) : ℕ := n * (a₁ + aₙ) / 2

-- Theorem statement
theorem remainder_of_sum_is_zero :
  let a₁ := 3
  let aₙ := 309
  let d := 6
  let n := (aₙ - a₁) / d + 1
  (sumArithmeticSequence a₁ aₙ n) % 6 = 0 := by
    sorry

end remainder_of_sum_is_zero_l477_47703


namespace arithmetic_mean_sqrt2_l477_47719

theorem arithmetic_mean_sqrt2 :
  (Real.sqrt 2 + 1 + (Real.sqrt 2 - 1)) / 2 = Real.sqrt 2 := by
  sorry

end arithmetic_mean_sqrt2_l477_47719


namespace f_positive_range_l477_47717

/-- A function f that is strictly increasing for x > 0 and symmetric about the y-axis -/
noncomputable def f (a b : ℝ) : ℝ → ℝ := λ x => (a * Real.exp x + b) * (x - 2)

/-- The theorem stating the range of m for which f(2-m) > 0 -/
theorem f_positive_range (a b : ℝ) :
  (∀ x > 0, Monotone (f a b)) →
  (∀ x, f a b x = f a b (-x)) →
  {m : ℝ | f a b (2 - m) > 0} = {m : ℝ | m < 0 ∨ m > 4} := by
  sorry

end f_positive_range_l477_47717


namespace cube_of_sum_and_reciprocal_l477_47708

theorem cube_of_sum_and_reciprocal (a : ℝ) (h : (a + 1/a)^2 = 3) :
  (a + 1/a)^3 = 3 * Real.sqrt 3 := by
  sorry

end cube_of_sum_and_reciprocal_l477_47708


namespace hiding_ways_correct_l477_47764

/-- The number of ways to hide 3 people in 6 cabinets with at most 2 people per cabinet -/
def hidingWays : ℕ := 210

/-- The number of people to be hidden -/
def numPeople : ℕ := 3

/-- The number of available cabinets -/
def numCabinets : ℕ := 6

/-- The maximum number of people that can be hidden in a single cabinet -/
def maxPerCabinet : ℕ := 2

theorem hiding_ways_correct :
  hidingWays = 
    (numCabinets * (numCabinets - 1) * (numCabinets - 2)) + 
    (Nat.choose numPeople 2 * Nat.choose numCabinets 1 * Nat.choose (numCabinets - 1) 1) := by
  sorry

#check hiding_ways_correct

end hiding_ways_correct_l477_47764


namespace work_completion_time_l477_47749

theorem work_completion_time (p_rate q_rate : ℚ) (work_left : ℚ) : 
  p_rate = 1/20 → q_rate = 1/10 → work_left = 7/10 → 
  (p_rate + q_rate) * 2 = 1 - work_left := by
sorry

end work_completion_time_l477_47749


namespace concert_ticket_sales_l477_47741

theorem concert_ticket_sales : ∀ (adult_tickets : ℕ) (senior_tickets : ℕ),
  -- Total tickets sold is 120
  adult_tickets + senior_tickets + adult_tickets = 120 →
  -- Total revenue is $1100
  12 * adult_tickets + 10 * senior_tickets + 6 * adult_tickets = 1100 →
  -- The number of senior tickets is 20
  senior_tickets = 20 := by
  sorry

end concert_ticket_sales_l477_47741


namespace egyptian_fraction_sum_l477_47721

theorem egyptian_fraction_sum : ∃ (b₂ b₃ b₄ b₅ b₆ : ℕ),
  (4 : ℚ) / 9 = b₂ / 2 + b₃ / 6 + b₄ / 24 + b₅ / 120 + b₆ / 720 ∧
  b₂ < 2 ∧ b₃ < 3 ∧ b₄ < 4 ∧ b₅ < 5 ∧ b₆ < 6 ∧
  b₂ + b₃ + b₄ + b₅ + b₆ = 9 ∧
  b₂ ≠ b₃ ∧ b₂ ≠ b₄ ∧ b₂ ≠ b₅ ∧ b₂ ≠ b₆ ∧
  b₃ ≠ b₄ ∧ b₃ ≠ b₅ ∧ b₃ ≠ b₆ ∧
  b₄ ≠ b₅ ∧ b₄ ≠ b₆ ∧
  b₅ ≠ b₆ := by
  sorry

end egyptian_fraction_sum_l477_47721


namespace fourteen_own_all_pets_l477_47756

/-- The number of people who own all three types of pets (cats, dogs, and rabbits) -/
def people_with_all_pets (total : ℕ) (cat_owners : ℕ) (dog_owners : ℕ) (rabbit_owners : ℕ) (two_pet_owners : ℕ) : ℕ :=
  cat_owners + dog_owners + rabbit_owners - two_pet_owners - total

/-- Theorem stating that given the conditions in the problem, 14 people own all three types of pets -/
theorem fourteen_own_all_pets :
  people_with_all_pets 60 30 40 16 12 = 14 := by
  sorry

end fourteen_own_all_pets_l477_47756


namespace angle_C_measure_triangle_perimeter_l477_47755

-- Define the right triangle ABC
structure RightTriangle where
  A : Real
  B : Real
  C : Real
  AB : Real
  BC : Real
  AC : Real
  right_angle : C = 90
  angle_sum : A + B + C = 180

-- Define the given condition
def tan_condition (t : RightTriangle) : Prop :=
  Real.tan t.A + Real.tan t.B + Real.tan t.A * Real.tan t.B = 1

-- Theorem for part 1
theorem angle_C_measure (t : RightTriangle) (h : tan_condition t) : t.C = 135 := by
  sorry

-- Theorem for part 2
theorem triangle_perimeter 
  (t : RightTriangle) 
  (h1 : tan_condition t) 
  (h2 : t.A = 15) 
  (h3 : t.AB = Real.sqrt 2) : 
  t.AB + t.BC + t.AC = (2 + Real.sqrt 6 + Real.sqrt 2) / 2 := by
  sorry

end angle_C_measure_triangle_perimeter_l477_47755


namespace brunchCombinationsCount_l477_47759

/-- The number of ways to choose one item from a set of 3, two different items from a set of 4, 
    and one item from another set of 3, where the order of selection doesn't matter. -/
def brunchCombinations : ℕ :=
  3 * (Nat.choose 4 2) * 3

/-- Theorem stating that the number of brunch combinations is 54. -/
theorem brunchCombinationsCount : brunchCombinations = 54 := by
  sorry

end brunchCombinationsCount_l477_47759


namespace cafeteria_extra_apples_l477_47722

/-- The number of extra apples in the cafeteria -/
def extra_apples (red_apples green_apples students_wanting_fruit : ℕ) : ℕ :=
  red_apples + green_apples - students_wanting_fruit

/-- Theorem: The cafeteria ends up with 73 extra apples -/
theorem cafeteria_extra_apples :
  ∀ (red_apples green_apples students_wanting_fruit : ℕ),
    red_apples = 43 →
    green_apples = 32 →
    students_wanting_fruit = 2 →
    extra_apples red_apples green_apples students_wanting_fruit = 73 :=
by
  sorry


end cafeteria_extra_apples_l477_47722


namespace infinitely_many_multiples_of_100_l477_47795

theorem infinitely_many_multiples_of_100 :
  ∀ k : ℕ, ∃ n : ℕ, n > k ∧ 100 ∣ (2^n + n^2) :=
sorry

end infinitely_many_multiples_of_100_l477_47795


namespace expected_cost_is_3500_l477_47790

/-- The number of machines -/
def total_machines : ℕ := 5

/-- The number of faulty machines -/
def faulty_machines : ℕ := 2

/-- The cost of testing one machine in yuan -/
def cost_per_test : ℕ := 1000

/-- The possible outcomes of the number of tests needed -/
def possible_tests : List ℕ := [2, 3, 4]

/-- The probabilities corresponding to each outcome -/
def probabilities : List ℚ := [1/10, 3/10, 3/5]

/-- The expected cost of testing in yuan -/
def expected_cost : ℚ := 3500

/-- Theorem stating that the expected cost of testing is 3500 yuan -/
theorem expected_cost_is_3500 :
  (List.sum (List.zipWith (· * ·) (List.map (λ n => n * cost_per_test) possible_tests) probabilities) : ℚ) = expected_cost :=
sorry

end expected_cost_is_3500_l477_47790


namespace trigonometric_equalities_l477_47727

theorem trigonometric_equalities : 
  (Real.sqrt 2 / 2) * (Real.cos (15 * π / 180) - Real.sin (15 * π / 180)) = 1/2 ∧
  Real.tan (22.5 * π / 180) / (1 - Real.tan (22.5 * π / 180)^2) = 1/2 := by
  sorry

end trigonometric_equalities_l477_47727


namespace smallest_among_given_rationals_l477_47748

theorem smallest_among_given_rationals :
  let S : Set ℚ := {5, -7, 0, -5/3}
  ∀ x ∈ S, -7 ≤ x :=
by sorry

end smallest_among_given_rationals_l477_47748


namespace expression_equality_l477_47779

theorem expression_equality : (3 + 2)^127 + 3 * (2^126 + 3^126) = 5^127 + 3 * 2^126 + 3 * 3^126 := by
  sorry

end expression_equality_l477_47779


namespace third_term_of_arithmetic_sequence_l477_47753

def arithmetic_sequence (a : ℤ) (d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

theorem third_term_of_arithmetic_sequence 
  (a : ℤ) (d : ℤ) 
  (h1 : arithmetic_sequence a d 20 = 17) 
  (h2 : arithmetic_sequence a d 21 = 20) : 
  arithmetic_sequence a d 3 = -34 := by
  sorry

end third_term_of_arithmetic_sequence_l477_47753


namespace marathon_remainder_yards_l477_47752

/-- Represents the distance of a marathon in miles and yards -/
structure Marathon :=
  (miles : ℕ)
  (yards : ℕ)

/-- Represents a total distance in miles and yards -/
structure TotalDistance :=
  (miles : ℕ)
  (yards : ℕ)

def marathon_distance : Marathon :=
  { miles := 26, yards := 395 }

def yards_per_mile : ℕ := 1760

def number_of_marathons : ℕ := 15

theorem marathon_remainder_yards :
  ∃ (m : ℕ) (y : ℕ), 
    y < yards_per_mile ∧
    TotalDistance.yards (
      { miles := m
      , yards := y } : TotalDistance
    ) = 645 ∧
    m * yards_per_mile + y = 
      number_of_marathons * (marathon_distance.miles * yards_per_mile + marathon_distance.yards) :=
by sorry

end marathon_remainder_yards_l477_47752


namespace total_games_calculation_l477_47730

/-- The number of football games in one month -/
def games_per_month : ℝ := 323.0

/-- The number of months in a season -/
def season_duration : ℝ := 17.0

/-- The total number of football games in a season -/
def total_games : ℝ := games_per_month * season_duration

theorem total_games_calculation :
  total_games = 5491.0 := by sorry

end total_games_calculation_l477_47730


namespace part_one_part_two_l477_47776

-- Define the propositions p and q
def p (x : ℝ) : Prop := (x + 2) * (x - 3) ≤ 0

def q (m x : ℝ) : Prop := 1 - m ≤ x ∧ x ≤ 1 + m

-- Part I
theorem part_one (m : ℝ) : 
  (m > 0 ∧ (∀ x, ¬(q m x) → ¬(p x))) ↔ (0 < m ∧ m ≤ 2) :=
sorry

-- Part II
theorem part_two (x : ℝ) :
  ((p x ∨ q 7 x) ∧ ¬(p x ∧ q 7 x)) ↔ ((-6 ≤ x ∧ x ≤ -2) ∨ (3 < x ∧ x < 8)) :=
sorry

end part_one_part_two_l477_47776


namespace triangle_equilateral_l477_47709

/-- A triangle with side lengths a, b, and c satisfying specific conditions is equilateral. -/
theorem triangle_equilateral (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a^4 = b^4 + c^4 - b^2*c^2) (h5 : b^4 = a^4 + c^4 - a^2*c^2) :
  a = b ∧ b = c := by
  sorry

end triangle_equilateral_l477_47709


namespace log_expression_equals_two_l477_47797

theorem log_expression_equals_two :
  Real.log 4 + Real.log 5 * Real.log 20 + (Real.log 5)^2 = 2 := by
  sorry

end log_expression_equals_two_l477_47797


namespace symmetry_of_point_l477_47733

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Symmetry with respect to y-axis -/
def symmetricToYAxis (p : Point) : Point :=
  ⟨-p.x, p.y⟩

theorem symmetry_of_point :
  let A : Point := ⟨6, 4⟩
  let B : Point := symmetricToYAxis A
  B = ⟨-6, 4⟩ := by sorry

end symmetry_of_point_l477_47733


namespace number_problem_l477_47757

theorem number_problem (x : ℝ) : 
  ((1/5 * 1/4 * x) - (5/100 * x)) + ((1/3 * x) - (1/7 * x)) = (1/10 * x - 12) → 
  x = -132 := by
sorry

end number_problem_l477_47757


namespace sqrt_equation_condition_l477_47796

theorem sqrt_equation_condition (x y : ℝ) : 
  Real.sqrt (3 * x^2 + y^2) = 2 * x + y ↔ x * (x + 4 * y) = 0 ∧ 2 * x + y ≥ 0 := by
  sorry

end sqrt_equation_condition_l477_47796


namespace inequality_solution_set_min_perimeter_rectangle_min_perimeter_achieved_l477_47745

-- Problem 1: Inequality solution set
theorem inequality_solution_set (x : ℝ) :
  x ∈ Set.Icc (-1 : ℝ) 3 ↔ x * (2 * x - 3) - 6 ≤ x := by sorry

-- Problem 2: Minimum perimeter of rectangle
theorem min_perimeter_rectangle (l w : ℝ) (h_area : l * w = 16) (h_positive : l > 0 ∧ w > 0) :
  2 * (l + w) ≥ 16 := by sorry

theorem min_perimeter_achieved (l w : ℝ) (h_area : l * w = 16) (h_positive : l > 0 ∧ w > 0) :
  2 * (l + w) = 16 ↔ l = 4 ∧ w = 4 := by sorry

end inequality_solution_set_min_perimeter_rectangle_min_perimeter_achieved_l477_47745


namespace custom_op_example_l477_47763

-- Define the custom operation ⊗
def custom_op (a b : ℤ) : ℤ := 2 * a - b

-- Theorem statement
theorem custom_op_example : custom_op 5 2 = 8 := by
  sorry

end custom_op_example_l477_47763


namespace binomial_expansion_coefficient_l477_47765

theorem binomial_expansion_coefficient (a : ℝ) (b : ℝ) :
  (∃ x, (1 + a * x)^5 = 1 + 10 * x + b * x^2 + x^3 * (1 + a * x)^2) →
  b = 40 := by
sorry

end binomial_expansion_coefficient_l477_47765


namespace sin_n_equals_cos_682_l477_47718

theorem sin_n_equals_cos_682 :
  ∃ n : ℤ, -120 ≤ n ∧ n ≤ 120 ∧ Real.sin (n * π / 180) = Real.cos (682 * π / 180) ∧ n = 128 := by
  sorry

end sin_n_equals_cos_682_l477_47718


namespace cubic_roots_bound_l477_47701

-- Define the polynomial
def cubic_polynomial (p q x : ℝ) : ℝ := x^3 + p*x + q

-- Define the condition for roots not exceeding 1 in modulus
def roots_within_unit_circle (p q : ℝ) : Prop :=
  ∀ x : ℂ, cubic_polynomial p q x.re = 0 → Complex.abs x ≤ 1

-- Theorem statement
theorem cubic_roots_bound (p q : ℝ) :
  roots_within_unit_circle p q ↔ p > abs q - 1 :=
sorry

end cubic_roots_bound_l477_47701


namespace kendra_remaining_words_l477_47760

/-- Theorem: Given Kendra's goal of learning 60 new words and having already learned 36 words,
    she needs to learn 24 more words to reach her goal. -/
theorem kendra_remaining_words (total_goal : ℕ) (learned : ℕ) (remaining : ℕ) :
  total_goal = 60 →
  learned = 36 →
  remaining = total_goal - learned →
  remaining = 24 :=
by sorry

end kendra_remaining_words_l477_47760


namespace march_first_is_monday_l477_47778

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

-- Define a function to get the day of the week for a given date in March
def marchDayOfWeek (date : Nat) : DayOfWeek := sorry

-- State the theorem
theorem march_first_is_monday : 
  marchDayOfWeek 8 = DayOfWeek.Monday → marchDayOfWeek 1 = DayOfWeek.Monday := by
  sorry

end march_first_is_monday_l477_47778


namespace q_age_is_40_l477_47715

/-- Represents a person with an age -/
structure Person where
  age : ℕ

/-- Given two people P and Q, proves that Q's age is 40 years
    under the specified conditions -/
theorem q_age_is_40 (P Q : Person) :
  (∃ (y : ℕ), P.age = 3 * (Q.age - y) ∧ P.age - y = Q.age) →
  P.age + Q.age = 100 →
  Q.age = 40 := by
sorry


end q_age_is_40_l477_47715


namespace largest_difference_l477_47737

def U : ℕ := 2 * 2010^2011
def V : ℕ := 2010^2011
def W : ℕ := 2009 * 2010^2010
def X : ℕ := 2 * 2010^2010
def Y : ℕ := 2010^2010
def Z : ℕ := 2010^2009

theorem largest_difference : 
  (U - V > V - W) ∧ 
  (U - V > W - X + 100) ∧ 
  (U - V > X - Y) ∧ 
  (U - V > Y - Z) := by
  sorry

end largest_difference_l477_47737


namespace base_number_proof_l477_47735

/-- 
Given a real number x, if (x^4 * 3.456789)^12 has 24 digits to the right of the decimal place 
when written as a single term, then x = 10^12.
-/
theorem base_number_proof (x : ℝ) : 
  (∃ n : ℕ, (x^4 * 3.456789)^12 * 10^24 = n) → x = 10^12 := by
  sorry

end base_number_proof_l477_47735


namespace swimmer_problem_l477_47743

theorem swimmer_problem (swimmer_speed : ℝ) (downstream_distance : ℝ) (upstream_distance : ℝ) :
  swimmer_speed = 5 →
  downstream_distance = 54 →
  upstream_distance = 6 →
  ∃ (time current_speed : ℝ),
    time > 0 ∧
    current_speed > 0 ∧
    current_speed < swimmer_speed ∧
    time = downstream_distance / (swimmer_speed + current_speed) ∧
    time = upstream_distance / (swimmer_speed - current_speed) ∧
    time = 6 := by
  sorry

end swimmer_problem_l477_47743


namespace opposite_reciprocal_expression_l477_47746

theorem opposite_reciprocal_expression (a b c d : ℝ) 
  (h1 : a = -b) 
  (h2 : c * d = 1) : 
  2023 * a + 2023 * b - 21 / (c * d) = -21 := by
  sorry

end opposite_reciprocal_expression_l477_47746


namespace rectangle_area_change_l477_47747

theorem rectangle_area_change (initial_area : ℝ) (length_increase : ℝ) (breadth_decrease : ℝ) :
  initial_area = 150 →
  length_increase = 37.5 →
  breadth_decrease = 18.2 →
  let new_length_factor := 1 + length_increase / 100
  let new_breadth_factor := 1 - breadth_decrease / 100
  let new_area := initial_area * new_length_factor * new_breadth_factor
  ∃ ε > 0, |new_area - 168.825| < ε :=
by
  sorry

end rectangle_area_change_l477_47747


namespace two_point_form_equation_l477_47724

/-- Two-point form equation of a line passing through two points -/
theorem two_point_form_equation (x1 y1 x2 y2 : ℝ) :
  let A : ℝ × ℝ := (x1, y1)
  let B : ℝ × ℝ := (x2, y2)
  x1 = 5 ∧ y1 = 6 ∧ x2 = -1 ∧ y2 = 2 →
  ∀ (x y : ℝ), (y - y1) / (y2 - y1) = (x - x1) / (x2 - x1) ↔
  (y - 6) / (2 - 6) = (x - 5) / (-1 - 5) :=
by sorry

end two_point_form_equation_l477_47724


namespace football_result_unique_solution_l477_47788

/-- Represents the result of a football team's performance -/
structure FootballResult where
  total_matches : ℕ
  lost_matches : ℕ
  total_points : ℕ
  wins : ℕ
  draws : ℕ

/-- Checks if a FootballResult is valid according to the given rules -/
def is_valid_result (r : FootballResult) : Prop :=
  r.total_matches = r.wins + r.draws + r.lost_matches ∧
  r.total_points = 3 * r.wins + r.draws

/-- Theorem stating the unique solution for the given problem -/
theorem football_result_unique_solution :
  ∃! (r : FootballResult),
    r.total_matches = 15 ∧
    r.lost_matches = 4 ∧
    r.total_points = 29 ∧
    is_valid_result r ∧
    r.wins = 9 ∧
    r.draws = 2 := by
  sorry

end football_result_unique_solution_l477_47788


namespace correct_ticket_count_l477_47777

/-- Represents the number of first-class tickets bought -/
def first_class_tickets : ℕ := 20

/-- Represents the number of second-class tickets bought -/
def second_class_tickets : ℕ := 45 - first_class_tickets

/-- The total cost of all tickets -/
def total_cost : ℕ := 400

theorem correct_ticket_count :
  first_class_tickets * 10 + second_class_tickets * 8 = total_cost ∧
  first_class_tickets + second_class_tickets = 45 :=
sorry

end correct_ticket_count_l477_47777


namespace miles_per_day_l477_47774

theorem miles_per_day (weekly_goal : ℕ) (days_run : ℕ) (miles_left : ℕ) : 
  weekly_goal = 24 → 
  days_run = 6 → 
  miles_left = 6 → 
  (weekly_goal - miles_left) / days_run = 3 := by
sorry

end miles_per_day_l477_47774


namespace solution_pair_l477_47712

theorem solution_pair : ∃ (x y : ℤ), 
  Real.sqrt (4 - 3 * Real.sin (30 * π / 180)) = x + y * (1 / Real.sin (30 * π / 180)) ∧ 
  x = 0 ∧ y = 1 := by
  sorry

#check solution_pair

end solution_pair_l477_47712


namespace subset_0_2_is_5th_subset_211_is_01467_l477_47791

/-- The set E with 10 elements -/
def E : Finset ℕ := Finset.range 10

/-- Function to calculate the k value for a given subset -/
def kValue (subset : Finset ℕ) : ℕ :=
  subset.sum (fun i => 2^i)

/-- The first theorem: {0, 2} (representing {a₁, a₃}) corresponds to k = 5 -/
theorem subset_0_2_is_5th : kValue {0, 2} = 5 := by sorry

/-- The second theorem: k = 211 corresponds to the subset {0, 1, 4, 6, 7} 
    (representing {a₁, a₂, a₅, a₇, a₈}) -/
theorem subset_211_is_01467 : 
  (Finset.filter (fun i => (211 / 2^i) % 2 = 1) E) = {0, 1, 4, 6, 7} := by sorry

end subset_0_2_is_5th_subset_211_is_01467_l477_47791


namespace server_performance_l477_47731

/-- Represents the number of multiplications a server can perform per second -/
def multiplications_per_second : ℕ := 5000

/-- Represents the number of seconds in half an hour -/
def seconds_in_half_hour : ℕ := 1800

/-- Represents the total number of multiplications in half an hour -/
def total_multiplications : ℕ := multiplications_per_second * seconds_in_half_hour

/-- Theorem stating that the server performs 9 million multiplications in half an hour -/
theorem server_performance : total_multiplications = 9000000 := by
  sorry

end server_performance_l477_47731


namespace f_min_max_on_interval_l477_47787

-- Define the function
def f (x : ℝ) : ℝ := 3 * x^4 - 6 * x^2 + 4

-- State the theorem
theorem f_min_max_on_interval :
  ∃ (min max : ℝ),
    (∀ x ∈ Set.Icc (-1) 3, f x ≥ min) ∧
    (∃ x ∈ Set.Icc (-1) 3, f x = min) ∧
    (∀ x ∈ Set.Icc (-1) 3, f x ≤ max) ∧
    (∃ x ∈ Set.Icc (-1) 3, f x = max) ∧
    min = 1 ∧ max = 193 :=
by sorry


end f_min_max_on_interval_l477_47787


namespace jellybean_probability_l477_47794

/-- The probability of selecting exactly one red and two blue jellybeans from a bowl -/
theorem jellybean_probability :
  let total_jellybeans : ℕ := 15
  let red_jellybeans : ℕ := 5
  let blue_jellybeans : ℕ := 3
  let white_jellybeans : ℕ := 7
  let picked_jellybeans : ℕ := 3

  -- Ensure the total number of jellybeans is correct
  total_jellybeans = red_jellybeans + blue_jellybeans + white_jellybeans →

  -- Calculate the probability
  (Nat.choose red_jellybeans 1 * Nat.choose blue_jellybeans 2 : ℚ) /
  Nat.choose total_jellybeans picked_jellybeans = 3 / 91 := by
  sorry

end jellybean_probability_l477_47794


namespace binary_1011011_eq_91_l477_47780

/-- Converts a list of binary digits to its decimal equivalent -/
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 1011011₂ -/
def binary_1011011 : List Bool := [true, true, false, true, true, false, true]

/-- Theorem: The decimal equivalent of 1011011₂ is 91 -/
theorem binary_1011011_eq_91 : binary_to_decimal binary_1011011 = 91 := by
  sorry

end binary_1011011_eq_91_l477_47780


namespace union_with_complement_l477_47706

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4}

-- Define set P
def P : Set Nat := {1, 2}

-- Define set Q
def Q : Set Nat := {1, 3}

-- Theorem statement
theorem union_with_complement :
  P ∪ (U \ Q) = {1, 2, 4} := by sorry

end union_with_complement_l477_47706


namespace abs_neg_three_l477_47713

theorem abs_neg_three : abs (-3 : ℤ) = 3 := by sorry

end abs_neg_three_l477_47713


namespace isosceles_triangle_perimeter_l477_47799

/-- The perimeter of an isosceles triangle given specific conditions -/
theorem isosceles_triangle_perimeter : ∀ (equilateral_perimeter isosceles_base : ℝ),
  equilateral_perimeter = 60 →
  isosceles_base = 30 →
  ∃ (isosceles_perimeter : ℝ),
    isosceles_perimeter = equilateral_perimeter / 3 + equilateral_perimeter / 3 + isosceles_base ∧
    isosceles_perimeter = 70 := by
  sorry

end isosceles_triangle_perimeter_l477_47799


namespace binomial_10_2_l477_47781

theorem binomial_10_2 : Nat.choose 10 2 = 45 := by sorry

end binomial_10_2_l477_47781


namespace hotel_room_charges_l477_47768

theorem hotel_room_charges (P R G : ℝ) 
  (h1 : P = R - 0.55 * R) 
  (h2 : P = G - 0.10 * G) : 
  R = 2 * G := by
sorry

end hotel_room_charges_l477_47768


namespace problem_statement_l477_47785

theorem problem_statement (x y : ℝ) (h1 : 2*x + 5*y = 10) (h2 : x*y = -10) :
  4*x^2 + 25*y^2 = 300 := by
sorry

end problem_statement_l477_47785


namespace square_overlap_ratio_l477_47704

theorem square_overlap_ratio (a b : ℝ) 
  (h1 : 0.52 * a^2 = a^2 - (a^2 - 0.73 * b^2)) 
  (h2 : a > 0) 
  (h3 : b > 0) : 
  a / b = 3 / 4 := by
sorry

end square_overlap_ratio_l477_47704


namespace incorrect_to_correct_ratio_l477_47772

theorem incorrect_to_correct_ratio (total : ℕ) (correct : ℕ) (incorrect : ℕ) :
  total = 75 →
  incorrect = 2 * correct →
  total = correct + incorrect →
  (incorrect : ℚ) / (correct : ℚ) = 2 / 1 := by
  sorry

end incorrect_to_correct_ratio_l477_47772


namespace tan_alpha_problem_l477_47784

theorem tan_alpha_problem (α : Real) (h : Real.tan α = 2) :
  (Real.tan (α + π/4) = -3) ∧
  ((Real.sin (2*α)) / (Real.sin α ^ 2 + Real.sin α * Real.cos α - Real.cos (2*α) - 1) = 1) := by
  sorry

end tan_alpha_problem_l477_47784


namespace number_multiplication_l477_47700

theorem number_multiplication (x : ℝ) : x - 7 = 9 → x * 5 = 80 := by
  sorry

end number_multiplication_l477_47700


namespace ferns_total_cost_l477_47770

/-- Calculates the total cost of Fern's purchase --/
def calculate_total_cost (high_heels_price : ℝ) (ballet_slippers_ratio : ℝ) 
  (ballet_slippers_count : ℕ) (purse_price : ℝ) (scarf_price : ℝ) 
  (high_heels_discount : ℝ) (sales_tax : ℝ) : ℝ :=
  let ballet_slippers_price := high_heels_price * ballet_slippers_ratio
  let total_ballet_slippers := ballet_slippers_price * ballet_slippers_count
  let discounted_high_heels := high_heels_price * (1 - high_heels_discount)
  let subtotal := discounted_high_heels + total_ballet_slippers + purse_price + scarf_price
  subtotal * (1 + sales_tax)

/-- Theorem stating that Fern's total cost is $348.30 --/
theorem ferns_total_cost : 
  calculate_total_cost 60 (2/3) 5 45 25 0.1 0.075 = 348.30 := by
  sorry

end ferns_total_cost_l477_47770


namespace angle_trig_values_l477_47726

/-- Given an angle α whose terminal side passes through the point (3,4),
    prove the values of sin α, cos α, and tan α. -/
theorem angle_trig_values (α : Real) :
  (∃ (r : Real), r > 0 ∧ r * Real.cos α = 3 ∧ r * Real.sin α = 4) →
  Real.sin α = 4/5 ∧ Real.cos α = 3/5 ∧ Real.tan α = 4/3 := by
  sorry

end angle_trig_values_l477_47726


namespace product_of_radicals_l477_47783

theorem product_of_radicals (q : ℝ) (hq : q > 0) :
  Real.sqrt (42 * q) * Real.sqrt (7 * q) * Real.sqrt (14 * q) = 98 * q * Real.sqrt (3 * q) := by
  sorry

end product_of_radicals_l477_47783


namespace simplify_expression_l477_47773

theorem simplify_expression : 3 * Real.sqrt 48 - 6 * Real.sqrt (1/3) + (Real.sqrt 3 - 1)^2 = 8 * Real.sqrt 3 + 4 := by
  sorry

end simplify_expression_l477_47773


namespace percentage_of_rejected_meters_l477_47732

def total_meters : ℕ := 100
def rejected_meters : ℕ := 10

theorem percentage_of_rejected_meters :
  (rejected_meters : ℚ) / (total_meters : ℚ) * 100 = 10 := by
  sorry

end percentage_of_rejected_meters_l477_47732


namespace rectangle_circle_area_ratio_l477_47782

/-- Given a rectangle intersecting a circle with specific chord properties,
    prove the ratio of their areas. -/
theorem rectangle_circle_area_ratio :
  ∀ (r : ℝ) (x y : ℝ),
    r > 0 →                 -- radius is positive
    x > 0 →                 -- shorter side is positive
    y > 0 →                 -- longer side is positive
    y = r →                 -- longer side equals radius (chord property)
    x = r / 2 →             -- shorter side equals half radius (chord property)
    y = 2 * x →             -- longer side is twice the shorter side
    (x * y) / (π * r^2) = 1 / (2 * π) :=
by
  sorry


end rectangle_circle_area_ratio_l477_47782


namespace num_small_triangles_odd_num_small_triangles_formula_l477_47739

/-- A triangle with interior points and connections -/
structure TriangleWithPoints where
  n : ℕ  -- number of interior points
  no_collinear : Bool  -- no three points (including vertices) are collinear
  max_connections : Bool  -- points are connected to maximize small triangles
  no_intersections : Bool  -- resulting segments do not intersect

/-- The number of small triangles formed in a TriangleWithPoints -/
def num_small_triangles (t : TriangleWithPoints) : ℕ := 2 * t.n + 1

/-- Theorem stating that the number of small triangles is odd -/
theorem num_small_triangles_odd (t : TriangleWithPoints) : 
  Odd (num_small_triangles t) := by
  sorry

/-- Theorem stating that the number of small triangles is 2n + 1 -/
theorem num_small_triangles_formula (t : TriangleWithPoints) : 
  num_small_triangles t = 2 * t.n + 1 := by
  sorry

end num_small_triangles_odd_num_small_triangles_formula_l477_47739


namespace triangle_lattice_distance_product_l477_47716

theorem triangle_lattice_distance_product (x y : ℝ) 
  (hx : ∃ (a b : ℤ), x^2 = a^2 + a*b + b^2)
  (hy : ∃ (c d : ℤ), y^2 = c^2 + c*d + d^2) :
  ∃ (e f : ℤ), (x*y)^2 = e^2 + e*f + f^2 := by
sorry

end triangle_lattice_distance_product_l477_47716


namespace last_digit_of_expression_l477_47751

theorem last_digit_of_expression : ∃ n : ℕ, (287 * 287 + 269 * 269 - 2 * 287 * 269) % 10 = 8 ∧ 10 * n + 8 = 287 * 287 + 269 * 269 - 2 * 287 * 269 := by
  sorry

end last_digit_of_expression_l477_47751


namespace tangent_line_at_point_2_minus_6_l477_47758

-- Define the function f(x) = x^3 - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 3

-- Theorem statement
theorem tangent_line_at_point_2_minus_6 :
  let P : ℝ × ℝ := (2, -6)
  let tangent_slope : ℝ := f' P.1
  let tangent_equation (x : ℝ) : ℝ := tangent_slope * (x - P.1) + P.2
  (∀ x, tangent_equation x = -3 * x) ∧ f P.1 = P.2 := by
  sorry


end tangent_line_at_point_2_minus_6_l477_47758


namespace total_cost_per_pineapple_l477_47761

/-- The cost per pineapple including shipping -/
def cost_per_pineapple (pineapple_cost : ℚ) (num_pineapples : ℕ) (shipping_cost : ℚ) : ℚ :=
  (pineapple_cost * num_pineapples + shipping_cost) / num_pineapples

/-- Theorem: The total cost per pineapple is $3.00 -/
theorem total_cost_per_pineapple :
  cost_per_pineapple (25/20) 12 21 = 3 := by
  sorry

end total_cost_per_pineapple_l477_47761


namespace expand_polynomial_l477_47725

theorem expand_polynomial (x : ℝ) : 
  (x - 2) * (x + 2) * (x^3 + 3*x + 1) = x^5 - x^3 + x^2 - 12*x - 4 := by
  sorry

end expand_polynomial_l477_47725


namespace inequality_preservation_l477_47728

theorem inequality_preservation (a b : ℝ) (h : a < b) : a - 5 < b - 5 := by
  sorry

end inequality_preservation_l477_47728


namespace arithmetic_sequence_2011_l477_47734

/-- Arithmetic sequence with first term a and common difference d -/
def arithmeticSequence (a d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

theorem arithmetic_sequence_2011 :
  ∃ n : ℕ, arithmeticSequence 1 3 n = 2011 ∧ n = 671 := by
  sorry

end arithmetic_sequence_2011_l477_47734


namespace complex_sum_real_l477_47742

theorem complex_sum_real (a : ℝ) : 
  (a / (1 + 2*I) + (1 + 2*I) / 5 : ℂ).im = 0 → a = 1 := by
  sorry

end complex_sum_real_l477_47742


namespace numeric_hex_count_l477_47793

/-- Represents a hexadecimal digit --/
inductive HexDigit
| Numeric (n : Fin 10)
| Alpha (a : Fin 6)

/-- Converts a natural number to its hexadecimal representation --/
def toHex (n : ℕ) : List HexDigit :=
  sorry

/-- Checks if a hexadecimal representation contains only numeric digits --/
def containsOnlyNumeric (hex : List HexDigit) : Bool :=
  sorry

/-- Counts the number of integers up to n with only numeric hexadecimal digits --/
def countNumericHex (n : ℕ) : ℕ :=
  sorry

/-- The main theorem --/
theorem numeric_hex_count :
  countNumericHex 500 = 199 :=
sorry

end numeric_hex_count_l477_47793
