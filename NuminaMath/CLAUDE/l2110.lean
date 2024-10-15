import Mathlib

namespace NUMINAMATH_CALUDE_alice_bob_race_difference_l2110_211044

/-- The time difference between two runners finishing a race -/
def race_time_difference (alice_speed bob_speed race_distance : ℝ) : ℝ :=
  bob_speed * race_distance - alice_speed * race_distance

/-- Theorem stating the time difference between Alice and Bob in a 12-mile race -/
theorem alice_bob_race_difference :
  race_time_difference 7 9 12 = 24 := by
  sorry

end NUMINAMATH_CALUDE_alice_bob_race_difference_l2110_211044


namespace NUMINAMATH_CALUDE_apartment_renovation_is_credence_good_decision_is_difficult_l2110_211047

-- Define the types
structure Service where
  name : String
  is_credence_good : Bool
  has_info_asymmetry : Bool
  quality_hard_to_assess : Bool

-- Define the apartment renovation service
def apartment_renovation : Service where
  name := "Complete Apartment Renovation"
  is_credence_good := true
  has_info_asymmetry := true
  quality_hard_to_assess := true

-- Theorem statement
theorem apartment_renovation_is_credence_good :
  apartment_renovation.is_credence_good ∧
  apartment_renovation.has_info_asymmetry ∧
  apartment_renovation.quality_hard_to_assess :=
by sorry

-- Define the provider types
inductive Provider
| ConstructionCompany
| PrivateRepairCrew

-- Define a function to represent the decision-making process
def choose_provider (service : Service) : Provider → Bool
| Provider.ConstructionCompany => true  -- Simplified for demonstration
| Provider.PrivateRepairCrew => false   -- Simplified for demonstration

-- Theorem about the difficulty of the decision
theorem decision_is_difficult (service : Service) :
  ∃ (p1 p2 : Provider), p1 ≠ p2 ∧ choose_provider service p1 = choose_provider service p2 :=
by sorry

end NUMINAMATH_CALUDE_apartment_renovation_is_credence_good_decision_is_difficult_l2110_211047


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l2110_211059

theorem quadratic_equal_roots (a : ℝ) : 
  (∃! x : ℝ, a * x^2 + 4 * x - 1 = 0) ↔ (a = -4 ∧ ∃ x : ℝ, x = 1/2 ∧ a * x^2 + 4 * x - 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l2110_211059


namespace NUMINAMATH_CALUDE_pizza_median_theorem_l2110_211063

/-- Represents the pizza sales data for a day -/
structure PizzaSalesData where
  total_slices : ℕ
  total_customers : ℕ
  min_slices_per_customer : ℕ

/-- Calculates the maximum possible median number of slices per customer -/
def max_possible_median (data : PizzaSalesData) : ℚ :=
  sorry

/-- Theorem stating the maximum possible median for the given scenario -/
theorem pizza_median_theorem (data : PizzaSalesData) 
  (h1 : data.total_slices = 310)
  (h2 : data.total_customers = 150)
  (h3 : data.min_slices_per_customer = 1) :
  max_possible_median data = 7 := by
  sorry

end NUMINAMATH_CALUDE_pizza_median_theorem_l2110_211063


namespace NUMINAMATH_CALUDE_ginos_white_bears_l2110_211097

theorem ginos_white_bears :
  ∀ (brown_bears white_bears black_bears total_bears : ℕ),
    brown_bears = 15 →
    black_bears = 27 →
    total_bears = 66 →
    total_bears = brown_bears + white_bears + black_bears →
    white_bears = 24 := by
  sorry

end NUMINAMATH_CALUDE_ginos_white_bears_l2110_211097


namespace NUMINAMATH_CALUDE_product_sequence_sum_l2110_211024

theorem product_sequence_sum (a b : ℕ) : 
  (a : ℚ) / 4 = 42 → b = a - 1 → a + b = 335 := by sorry

end NUMINAMATH_CALUDE_product_sequence_sum_l2110_211024


namespace NUMINAMATH_CALUDE_gcf_36_54_81_l2110_211089

theorem gcf_36_54_81 : Nat.gcd 36 (Nat.gcd 54 81) = 9 := by sorry

end NUMINAMATH_CALUDE_gcf_36_54_81_l2110_211089


namespace NUMINAMATH_CALUDE_zeros_in_square_of_near_power_of_ten_l2110_211029

theorem zeros_in_square_of_near_power_of_ten : 
  ∃ n : ℕ, (10^12 - 3)^2 = n * 10^11 ∧ n % 10 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_zeros_in_square_of_near_power_of_ten_l2110_211029


namespace NUMINAMATH_CALUDE_todd_remaining_money_l2110_211069

/-- Calculates the remaining money after Todd's purchases -/
def remaining_money (initial_amount : ℚ) (candy_price : ℚ) (candy_count : ℕ) 
  (gum_price : ℚ) (gum_count : ℕ) (soda_price : ℚ) (soda_count : ℕ) 
  (soda_discount : ℚ) : ℚ :=
  let candy_cost := candy_price * candy_count
  let gum_cost := gum_price * gum_count
  let soda_cost := soda_price * soda_count * (1 - soda_discount)
  let total_cost := candy_cost + gum_cost + soda_cost
  initial_amount - total_cost

/-- Theorem stating Todd's remaining money after purchases -/
theorem todd_remaining_money :
  remaining_money 50 2.5 7 1.5 5 3 3 0.2 = 17.8 := by
  sorry

end NUMINAMATH_CALUDE_todd_remaining_money_l2110_211069


namespace NUMINAMATH_CALUDE_geometric_sequence_single_digit_numbers_l2110_211009

theorem geometric_sequence_single_digit_numbers :
  ∃! (a b c : ℕ),
    a < 10 ∧ b < 10 ∧ c < 10 ∧
    ∃ (q : ℚ),
      b = a * q ∧
      (10 * a + c : ℚ) = a * q^2 ∧
      (10 * c + b : ℚ) = a * q^3 ∧
      a = 1 ∧ b = 4 ∧ c = 6 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_single_digit_numbers_l2110_211009


namespace NUMINAMATH_CALUDE_largest_non_sum_of_composites_l2110_211079

def isComposite (n : ℕ) : Prop :=
  ∃ k : ℕ, 1 < k ∧ k < n ∧ n % k = 0

def isSumOfTwoComposites (n : ℕ) : Prop :=
  ∃ a b : ℕ, isComposite a ∧ isComposite b ∧ n = a + b

theorem largest_non_sum_of_composites :
  (∀ n : ℕ, n > 11 → isSumOfTwoComposites n) ∧
  ¬isSumOfTwoComposites 11 :=
sorry

end NUMINAMATH_CALUDE_largest_non_sum_of_composites_l2110_211079


namespace NUMINAMATH_CALUDE_P_on_x_axis_P_parallel_y_axis_P_second_quadrant_distance_l2110_211057

-- Define point P
def P (a : ℝ) := (a - 1, 6 + 2*a)

-- Question 1
theorem P_on_x_axis (a : ℝ) : 
  P a = (-4, 0) ↔ (P a).2 = 0 := by sorry

-- Question 2
def Q : ℝ × ℝ := (5, 8)

theorem P_parallel_y_axis (a : ℝ) : 
  P a = (5, 18) ↔ (P a).1 = Q.1 := by sorry

-- Question 3
theorem P_second_quadrant_distance (a : ℝ) : 
  (P a).1 < 0 ∧ (P a).2 > 0 ∧ |(P a).2| = 2 * |(P a).1| → 
  a^2023 + 2024 = 2023 := by sorry

end NUMINAMATH_CALUDE_P_on_x_axis_P_parallel_y_axis_P_second_quadrant_distance_l2110_211057


namespace NUMINAMATH_CALUDE_positive_sum_and_product_iff_both_positive_l2110_211015

theorem positive_sum_and_product_iff_both_positive (a b : ℝ) :
  (a > 0 ∧ b > 0) ↔ (a + b > 0 ∧ a * b > 0) := by
  sorry

end NUMINAMATH_CALUDE_positive_sum_and_product_iff_both_positive_l2110_211015


namespace NUMINAMATH_CALUDE_total_video_game_cost_l2110_211085

/-- The cost of the basketball game -/
def basketball_cost : ℚ := 5.2

/-- The cost of the racing game -/
def racing_cost : ℚ := 4.23

/-- The total cost of the video games -/
def total_cost : ℚ := basketball_cost + racing_cost

/-- Theorem stating that the total cost of video games is $9.43 -/
theorem total_video_game_cost : total_cost = 9.43 := by sorry

end NUMINAMATH_CALUDE_total_video_game_cost_l2110_211085


namespace NUMINAMATH_CALUDE_cube_surface_area_l2110_211030

theorem cube_surface_area (volume : ℝ) (side : ℝ) (surface_area : ℝ) : 
  volume = 1000 →
  volume = side^3 →
  surface_area = 6 * side^2 →
  surface_area = 600 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_l2110_211030


namespace NUMINAMATH_CALUDE_ball_probabilities_l2110_211092

/-- Represents a bag of balls with a given number of black and white balls. -/
structure BagOfBalls where
  blackBalls : ℕ
  whiteBalls : ℕ

/-- Calculates the probability of drawing two black balls without replacement. -/
def probabilityTwoBlackBalls (bag : BagOfBalls) : ℚ :=
  let totalBalls := bag.blackBalls + bag.whiteBalls
  (bag.blackBalls.choose 2 : ℚ) / (totalBalls.choose 2)

/-- Calculates the probability of drawing a black ball on the second draw,
    given that a black ball was drawn on the first draw. -/
def probabilitySecondBlackGivenFirstBlack (bag : BagOfBalls) : ℚ :=
  (bag.blackBalls - 1 : ℚ) / (bag.blackBalls + bag.whiteBalls - 1)

theorem ball_probabilities (bag : BagOfBalls) 
  (h1 : bag.blackBalls = 6) (h2 : bag.whiteBalls = 4) : 
  probabilityTwoBlackBalls bag = 1/3 ∧ 
  probabilitySecondBlackGivenFirstBlack bag = 5/9 := by
  sorry


end NUMINAMATH_CALUDE_ball_probabilities_l2110_211092


namespace NUMINAMATH_CALUDE_bens_class_girls_l2110_211096

theorem bens_class_girls (total : ℕ) (girl_ratio boy_ratio : ℕ) (h1 : total = 35) (h2 : girl_ratio = 3) (h3 : boy_ratio = 4) :
  ∃ (girls boys : ℕ), girls + boys = total ∧ girls * boy_ratio = boys * girl_ratio ∧ girls = 15 := by
sorry

end NUMINAMATH_CALUDE_bens_class_girls_l2110_211096


namespace NUMINAMATH_CALUDE_smallest_angle_measure_l2110_211075

-- Define a right triangle with acute angles in the ratio 3:2
structure RightTriangle where
  angle1 : ℝ
  angle2 : ℝ
  right_angle : ℝ
  is_right_triangle : right_angle = 90
  acute_angle_sum : angle1 + angle2 = 90
  angle_ratio : angle1 / angle2 = 3 / 2

-- Theorem statement
theorem smallest_angle_measure (t : RightTriangle) : 
  min t.angle1 t.angle2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_smallest_angle_measure_l2110_211075


namespace NUMINAMATH_CALUDE_geometric_sequence_inequality_l2110_211082

/-- Given a geometric sequence {a_n}, prove that a_1^2 + a_3^2 ≥ 2a_2^2 -/
theorem geometric_sequence_inequality (a : ℕ → ℝ) (q : ℝ) 
  (h : ∀ n, a (n + 1) = a n * q) : 
  a 1^2 + a 3^2 ≥ 2 * a 2^2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_inequality_l2110_211082


namespace NUMINAMATH_CALUDE_bus_average_speed_l2110_211071

/-- Proves that the average speed of a bus line is 60 km/h given specific conditions -/
theorem bus_average_speed
  (stop_interval : ℕ) -- Time interval between stops in minutes
  (num_stops : ℕ) -- Number of stops to the destination
  (distance : ℝ) -- Distance to the destination in kilometers
  (h1 : stop_interval = 5)
  (h2 : num_stops = 8)
  (h3 : distance = 40) :
  distance / (↑(stop_interval * num_stops) / 60) = 60 :=
by sorry

end NUMINAMATH_CALUDE_bus_average_speed_l2110_211071


namespace NUMINAMATH_CALUDE_election_votes_l2110_211018

theorem election_votes (total_votes : ℕ) 
  (h1 : ∃ (winner loser : ℕ), winner + loser = total_votes) 
  (h2 : ∃ (winner : ℕ), winner = (70 * total_votes) / 100) 
  (h3 : ∃ (winner loser : ℕ), winner - loser = 188) : 
  total_votes = 470 := by
sorry

end NUMINAMATH_CALUDE_election_votes_l2110_211018


namespace NUMINAMATH_CALUDE_haley_washing_machine_capacity_l2110_211064

/-- The number of pieces of clothing Haley's washing machine can wash at a time -/
def washing_machine_capacity (total_clothes : ℕ) (num_loads : ℕ) : ℕ :=
  total_clothes / num_loads

theorem haley_washing_machine_capacity :
  let total_shirts : ℕ := 2
  let total_sweaters : ℕ := 33
  let total_clothes : ℕ := total_shirts + total_sweaters
  let num_loads : ℕ := 5
  washing_machine_capacity total_clothes num_loads = 7 := by
  sorry

end NUMINAMATH_CALUDE_haley_washing_machine_capacity_l2110_211064


namespace NUMINAMATH_CALUDE_patricks_pencil_loss_percentage_l2110_211000

/-- Calculates the overall loss percentage for Patrick's pencil sales -/
theorem patricks_pencil_loss_percentage : 
  let type_a_count : ℕ := 30
  let type_b_count : ℕ := 40
  let type_c_count : ℕ := 10
  let type_a_cost : ℚ := 1
  let type_b_cost : ℚ := 2
  let type_c_cost : ℚ := 3
  let type_a_discount : ℚ := 0.5
  let type_b_discount : ℚ := 1
  let type_c_discount : ℚ := 1.5
  let total_cost : ℚ := type_a_count * type_a_cost + type_b_count * type_b_cost + type_c_count * type_c_cost
  let total_revenue : ℚ := type_a_count * (type_a_cost - type_a_discount) + 
                           type_b_count * (type_b_cost - type_b_discount) + 
                           type_c_count * (type_c_cost - type_c_discount)
  let additional_loss : ℚ := type_a_count * (type_a_cost - type_a_discount)
  let total_loss : ℚ := total_cost - total_revenue + additional_loss
  let loss_percentage : ℚ := (total_loss / total_cost) * 100
  ∃ ε > 0, |loss_percentage - 60.71| < ε :=
by sorry

end NUMINAMATH_CALUDE_patricks_pencil_loss_percentage_l2110_211000


namespace NUMINAMATH_CALUDE_value_of_expression_l2110_211002

theorem value_of_expression (x : ℝ) (h : x = 4) : 3 * x + 5 = 17 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l2110_211002


namespace NUMINAMATH_CALUDE_inequality_solution_range_l2110_211045

theorem inequality_solution_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 2 * a * x + 2 ≥ 0) → 0 ≤ a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l2110_211045


namespace NUMINAMATH_CALUDE_unique_solution_for_abc_l2110_211052

theorem unique_solution_for_abc : ∃! (a b c : ℝ),
  a < b ∧ b < c ∧
  a + b + c = 21 / 4 ∧
  1 / a + 1 / b + 1 / c = 21 / 4 ∧
  a * b * c = 1 ∧
  a = 1 / 4 ∧ b = 1 ∧ c = 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_abc_l2110_211052


namespace NUMINAMATH_CALUDE_binomial_12_choose_10_l2110_211072

theorem binomial_12_choose_10 : Nat.choose 12 10 = 66 := by sorry

end NUMINAMATH_CALUDE_binomial_12_choose_10_l2110_211072


namespace NUMINAMATH_CALUDE_inequality_proof_l2110_211020

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a / Real.sqrt b + b / Real.sqrt a ≥ Real.sqrt a + Real.sqrt b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2110_211020


namespace NUMINAMATH_CALUDE_constant_term_binomial_expansion_l2110_211027

/-- The constant term in the expansion of (1/√x - x^2)^10 is 45 -/
theorem constant_term_binomial_expansion :
  let f (x : ℝ) := (1 / Real.sqrt x - x^2)^10
  ∃ (c : ℝ), ∀ (x : ℝ), x ≠ 0 → f x = c + x * (f x - c) ∧ c = 45 :=
sorry

end NUMINAMATH_CALUDE_constant_term_binomial_expansion_l2110_211027


namespace NUMINAMATH_CALUDE_eggs_left_proof_l2110_211090

def eggs_left (initial : ℕ) (harry_takes : ℕ) (jenny_takes : ℕ) : ℕ :=
  initial - (harry_takes + jenny_takes)

theorem eggs_left_proof :
  eggs_left 47 5 8 = 34 := by
  sorry

end NUMINAMATH_CALUDE_eggs_left_proof_l2110_211090


namespace NUMINAMATH_CALUDE_womens_average_age_l2110_211077

/-- The average age of two women given the following conditions:
    - There are initially 6 men
    - Two men aged 10 and 12 are replaced by two women
    - The average age increases by 2 years after the replacement
-/
theorem womens_average_age (initial_men : ℕ) (age_increase : ℝ) 
  (replaced_man1_age replaced_man2_age : ℕ) :
  initial_men = 6 →
  age_increase = 2 →
  replaced_man1_age = 10 →
  replaced_man2_age = 12 →
  ∃ (initial_avg : ℝ),
    ((initial_men : ℝ) * initial_avg - (replaced_man1_age + replaced_man2_age : ℝ) + 
     2 * ((initial_avg + age_increase) : ℝ)) / 2 = 17 :=
by sorry

end NUMINAMATH_CALUDE_womens_average_age_l2110_211077


namespace NUMINAMATH_CALUDE_movie_ticket_cost_l2110_211099

theorem movie_ticket_cost (ticket_count : ℕ) (borrowed_movie_cost change paid : ℚ) : 
  ticket_count = 2 → 
  borrowed_movie_cost = 679/100 → 
  change = 137/100 → 
  paid = 20 → 
  ∃ (ticket_cost : ℚ), 
    ticket_cost * ticket_count + borrowed_movie_cost = paid - change ∧ 
    ticket_cost = 592/100 := by
  sorry

end NUMINAMATH_CALUDE_movie_ticket_cost_l2110_211099


namespace NUMINAMATH_CALUDE_dans_car_efficiency_l2110_211055

/-- Represents the fuel efficiency of Dan's car in miles per gallon. -/
def miles_per_gallon : ℝ := 32

/-- Represents the cost of gas in dollars per gallon. -/
def gas_cost_per_gallon : ℝ := 4

/-- Represents the distance Dan's car can travel in miles. -/
def distance_traveled : ℝ := 368

/-- Represents the total cost of gas in dollars. -/
def total_gas_cost : ℝ := 46

/-- Proves that Dan's car gets 32 miles per gallon given the conditions. -/
theorem dans_car_efficiency :
  miles_per_gallon = distance_traveled / (total_gas_cost / gas_cost_per_gallon) := by
  sorry


end NUMINAMATH_CALUDE_dans_car_efficiency_l2110_211055


namespace NUMINAMATH_CALUDE_base_a_equations_l2110_211010

/-- Converts a base-10 number to base-a --/
def toBaseA (n : ℕ) (a : ℕ) : ℕ := sorry

/-- Converts a base-a number to base-10 --/
def fromBaseA (n : ℕ) (a : ℕ) : ℕ := sorry

theorem base_a_equations (a : ℕ) :
  (toBaseA 375 a + toBaseA 596 a = toBaseA (9 * a + fromBaseA 12 10) a) ∧
  (fromBaseA 12 10 = 12) ∧
  (toBaseA 697 a + toBaseA 226 a = toBaseA (9 * a + fromBaseA 13 10) a) ∧
  (fromBaseA 13 10 = 13) →
  a = 14 := by sorry

end NUMINAMATH_CALUDE_base_a_equations_l2110_211010


namespace NUMINAMATH_CALUDE_clerical_percentage_theorem_l2110_211065

/-- Represents the employee composition of a company -/
structure CompanyEmployees where
  total : ℕ
  clerical_ratio : ℚ
  management_ratio : ℚ
  clerical_reduction : ℚ

/-- Calculates the percentage of clerical employees after reduction -/
def clerical_percentage_after_reduction (c : CompanyEmployees) : ℚ :=
  let initial_clerical := c.clerical_ratio * c.total
  let reduced_clerical := initial_clerical - c.clerical_reduction * initial_clerical
  let total_after_reduction := c.total - (initial_clerical - reduced_clerical)
  (reduced_clerical / total_after_reduction) * 100

/-- Theorem stating the result of the employee reduction -/
theorem clerical_percentage_theorem (c : CompanyEmployees) 
  (h1 : c.total = 5000)
  (h2 : c.clerical_ratio = 3/7)
  (h3 : c.management_ratio = 1/3)
  (h4 : c.clerical_reduction = 3/8) :
  ∃ (ε : ℚ), abs (clerical_percentage_after_reduction c - 3194/100) < ε ∧ ε < 1/100 := by
  sorry

end NUMINAMATH_CALUDE_clerical_percentage_theorem_l2110_211065


namespace NUMINAMATH_CALUDE_system_solution_l2110_211061

theorem system_solution : ∃! (x y : ℝ), (x / 3 - (y + 1) / 2 = 1) ∧ (4 * x - (2 * y - 5) = 11) ∧ x = 0 ∧ y = -3 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2110_211061


namespace NUMINAMATH_CALUDE_wayne_shrimp_cost_l2110_211050

/-- Calculates the cost of shrimp for an appetizer given the number of shrimp per guest,
    number of guests, cost per pound, and number of shrimp per pound. -/
def shrimpAppetizer (shrimpPerGuest : ℕ) (numGuests : ℕ) (costPerPound : ℚ) (shrimpPerPound : ℕ) : ℚ :=
  (shrimpPerGuest * numGuests : ℚ) / shrimpPerPound * costPerPound

/-- Proves that Wayne will spend $170.00 on the shrimp appetizer given the specified conditions. -/
theorem wayne_shrimp_cost :
  shrimpAppetizer 5 40 17 20 = 170 := by
  sorry

end NUMINAMATH_CALUDE_wayne_shrimp_cost_l2110_211050


namespace NUMINAMATH_CALUDE_equation_solution_l2110_211060

theorem equation_solution : ∀ x : ℚ, (2/3 : ℚ) - (1/4 : ℚ) = 1/x → x = 12/5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2110_211060


namespace NUMINAMATH_CALUDE_election_vote_difference_l2110_211041

theorem election_vote_difference 
  (total_votes : ℕ) 
  (winner_votes second_votes third_votes fourth_votes : ℕ) 
  (h_total : total_votes = 979)
  (h_candidates : winner_votes + second_votes + third_votes + fourth_votes = total_votes)
  (h_second : winner_votes = second_votes + 53)
  (h_third : winner_votes = third_votes + 79)
  (h_fourth : fourth_votes = 199) :
  winner_votes - fourth_votes = 105 := by
sorry

end NUMINAMATH_CALUDE_election_vote_difference_l2110_211041


namespace NUMINAMATH_CALUDE_balance_theorem_l2110_211078

/-- Represents the weight of a ball in an arbitrary unit -/
@[ext] structure BallWeight where
  weight : ℚ

/-- Defines the weight relationships between different colored balls -/
structure BallWeights where
  red : BallWeight
  blue : BallWeight
  orange : BallWeight
  purple : BallWeight
  red_blue_balance : 4 * red.weight = 8 * blue.weight
  orange_blue_balance : 3 * orange.weight = 15/2 * blue.weight
  blue_purple_balance : 8 * blue.weight = 6 * purple.weight

/-- Theorem stating the balance of 68.5/3 blue balls with 5 red, 3 orange, and 4 purple balls -/
theorem balance_theorem (weights : BallWeights) :
  (68.5/3) * weights.blue.weight = 5 * weights.red.weight + 3 * weights.orange.weight + 4 * weights.purple.weight :=
by sorry

end NUMINAMATH_CALUDE_balance_theorem_l2110_211078


namespace NUMINAMATH_CALUDE_nested_average_calculation_l2110_211094

def average (a b c : ℚ) : ℚ := (a + b + c) / 3

theorem nested_average_calculation : 
  let x := average 2 3 1
  let y := average 4 1 0
  average x y 5 = 26 / 9 := by sorry

end NUMINAMATH_CALUDE_nested_average_calculation_l2110_211094


namespace NUMINAMATH_CALUDE_complement_of_35_degrees_l2110_211028

theorem complement_of_35_degrees :
  ∀ α : Real,
  α = 35 →
  90 - α = 55 :=
by
  sorry

end NUMINAMATH_CALUDE_complement_of_35_degrees_l2110_211028


namespace NUMINAMATH_CALUDE_point_inside_circle_range_l2110_211043

theorem point_inside_circle_range (a : ℝ) : 
  ((1 - a)^2 + (1 + a)^2 < 4) → (-1 < a ∧ a < 1) := by
  sorry

end NUMINAMATH_CALUDE_point_inside_circle_range_l2110_211043


namespace NUMINAMATH_CALUDE_first_reduction_percentage_l2110_211037

theorem first_reduction_percentage (x : ℝ) : 
  (1 - x / 100) * (1 - 0.1) = 0.81 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_first_reduction_percentage_l2110_211037


namespace NUMINAMATH_CALUDE_james_soda_consumption_l2110_211031

/-- Calculates the number of sodas James drinks per day given the following conditions:
  * James buys 5 packs of sodas
  * Each pack contains 12 sodas
  * James already had 10 sodas
  * He finishes all the sodas in 1 week (7 days)
-/
theorem james_soda_consumption 
  (packs : ℕ) 
  (sodas_per_pack : ℕ) 
  (initial_sodas : ℕ) 
  (days_to_finish : ℕ) 
  (h1 : packs = 5)
  (h2 : sodas_per_pack = 12)
  (h3 : initial_sodas = 10)
  (h4 : days_to_finish = 7) :
  (packs * sodas_per_pack + initial_sodas) / days_to_finish = 10 := by
  sorry

end NUMINAMATH_CALUDE_james_soda_consumption_l2110_211031


namespace NUMINAMATH_CALUDE_mission_duration_percentage_l2110_211076

/-- Proves that given the conditions of the problem, the first mission took 60% longer than planned. -/
theorem mission_duration_percentage (planned_duration : ℕ) (second_mission_duration : ℕ) (total_duration : ℕ) :
  planned_duration = 5 →
  second_mission_duration = 3 →
  total_duration = 11 →
  ∃ (percentage : ℚ),
    percentage = 60 ∧
    total_duration = planned_duration + (percentage / 100) * planned_duration + second_mission_duration :=
by
  sorry

#check mission_duration_percentage

end NUMINAMATH_CALUDE_mission_duration_percentage_l2110_211076


namespace NUMINAMATH_CALUDE_range_of_a_l2110_211012

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 + 2*x - 3 > 0
def q (x a : ℝ) : Prop := x > a

-- State the theorem
theorem range_of_a (h : ∀ x a : ℝ, (¬(p x) → ¬(q x a)) ∧ ∃ x, ¬(p x) ∧ (q x a)) :
  ∀ a : ℝ, a ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2110_211012


namespace NUMINAMATH_CALUDE_stating_two_students_math_course_l2110_211004

/-- The number of students -/
def num_students : ℕ := 4

/-- The number of courses -/
def num_courses : ℕ := 4

/-- The number of students who should choose mathematics -/
def math_students : ℕ := 2

/-- The number of remaining courses after mathematics -/
def remaining_courses : ℕ := 3

/-- Function to calculate the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := 
  Nat.choose n k

/-- 
Theorem stating that the number of ways in which exactly two out of four students 
can choose a mathematics tutoring course, while the other two choose from three 
remaining courses, is equal to 54.
-/
theorem two_students_math_course : 
  (choose num_students math_students) * (remaining_courses^(num_students - math_students)) = 54 := by
  sorry

end NUMINAMATH_CALUDE_stating_two_students_math_course_l2110_211004


namespace NUMINAMATH_CALUDE_max_ratio_squared_l2110_211014

theorem max_ratio_squared (a b x y : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_a_geq_b : a ≥ b)
  (h_eq : a^2 + y^2 = b^2 + x^2 ∧ b^2 + x^2 = Real.sqrt ((a - x)^2 + (b - y)^2))
  (h_bounds : 0 ≤ x ∧ x < a ∧ 0 ≤ y ∧ y < b)
  (h_right_triangle : (a - b + x)^2 + (b - a + y)^2 = a^2 + b^2) :
  (∀ ρ : ℝ, a ≤ ρ * b → ρ^2 ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_max_ratio_squared_l2110_211014


namespace NUMINAMATH_CALUDE_side_length_b_l2110_211080

open Real

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def isArithmeticSequence (t : Triangle) : Prop :=
  t.A + t.C = 2 * t.B

def hasCorrectArea (t : Triangle) : Prop :=
  1/2 * t.a * t.c * sin t.B = 5 * sqrt 3

-- Main theorem
theorem side_length_b (t : Triangle) 
  (h1 : isArithmeticSequence t)
  (h2 : t.a = 4)
  (h3 : hasCorrectArea t) :
  t.b = sqrt 21 := by
  sorry


end NUMINAMATH_CALUDE_side_length_b_l2110_211080


namespace NUMINAMATH_CALUDE_composition_equality_l2110_211048

def f (a b x : ℝ) : ℝ := a * x + b
def g (c d x : ℝ) : ℝ := c * x + d

theorem composition_equality (a b c d : ℝ) :
  (∀ x, f a b (g c d x) = g c d (f a b x)) ↔ b * (1 - c) - d * (1 - a) = 0 := by
  sorry

end NUMINAMATH_CALUDE_composition_equality_l2110_211048


namespace NUMINAMATH_CALUDE_right_triangle_area_l2110_211016

theorem right_triangle_area (a b : ℝ) (h1 : a = 40) (h2 : b = 42) :
  (1 / 2 : ℝ) * a * b = 840 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2110_211016


namespace NUMINAMATH_CALUDE_fruit_shop_problem_l2110_211062

theorem fruit_shop_problem (total_cost : ℕ) (total_profit : ℕ) 
  (lychee_cost : ℕ) (longan_cost : ℕ) (lychee_price : ℕ) (longan_price : ℕ) 
  (second_profit : ℕ) :
  total_cost = 3900 →
  total_profit = 1200 →
  lychee_cost = 120 →
  longan_cost = 100 →
  lychee_price = 150 →
  longan_price = 140 →
  second_profit = 960 →
  ∃ (lychee_boxes longan_boxes : ℕ) (discount_rate : ℚ),
    lychee_cost * lychee_boxes + longan_cost * longan_boxes = total_cost ∧
    (lychee_price - lychee_cost) * lychee_boxes + (longan_price - longan_cost) * longan_boxes = total_profit ∧
    lychee_boxes = 20 ∧
    longan_boxes = 15 ∧
    (lychee_price - lychee_cost) * lychee_boxes + 
      (longan_price * discount_rate - longan_cost) * (2 * longan_boxes) = second_profit ∧
    discount_rate = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_fruit_shop_problem_l2110_211062


namespace NUMINAMATH_CALUDE_polynomial_characterization_l2110_211095

-- Define a polynomial with real coefficients
def RealPolynomial := ℝ → ℝ

-- Define the condition for a, b, c
def TripleCondition (a b c : ℝ) : Prop := a * b + b * c + c * a = 0

-- Define the polynomial equality condition
def PolynomialCondition (P : RealPolynomial) : Prop :=
  ∀ a b c : ℝ, TripleCondition a b c →
    P (a - b) + P (b - c) + P (c - a) = 2 * P (a + b + c)

-- Define the quadratic-quartic polynomial form
def QuadraticQuarticForm (P : RealPolynomial) : Prop :=
  ∃ a₂ a₄ : ℝ, ∀ x : ℝ, P x = a₂ * x^2 + a₄ * x^4

-- The main theorem
theorem polynomial_characterization :
  ∀ P : RealPolynomial, PolynomialCondition P → QuadraticQuarticForm P :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_characterization_l2110_211095


namespace NUMINAMATH_CALUDE_b_share_yearly_profit_l2110_211013

/-- Investment proportions and profit distribution for partners A, B, C, and D --/
structure Partnership where
  b_invest : ℝ  -- B's investment (base unit)
  a_invest : ℝ := 2.5 * b_invest  -- A's investment
  c_invest : ℝ := 1.5 * b_invest  -- C's investment
  d_invest : ℝ := 1.25 * b_invest  -- D's investment
  total_invest : ℝ := a_invest + b_invest + c_invest  -- Total investment of A, B, and C
  profit_6months : ℝ := 6000  -- Profit for 6 months
  d_fixed_amount : ℝ := 500  -- D's fixed amount per 6 months
  profit_year : ℝ := 16900  -- Total profit for the year

/-- Theorem stating B's share of the yearly profit --/
theorem b_share_yearly_profit (p : Partnership) :
  (p.b_invest / p.total_invest) * (p.profit_year - 2 * p.d_fixed_amount) = 3180 := by
  sorry

end NUMINAMATH_CALUDE_b_share_yearly_profit_l2110_211013


namespace NUMINAMATH_CALUDE_battle_gathering_count_l2110_211019

theorem battle_gathering_count :
  -- Define the number of cannoneers
  ∀ (cannoneers : ℕ),
  -- Define the number of women as double the number of cannoneers
  ∀ (women : ℕ),
  women = 2 * cannoneers →
  -- Define the number of men as twice the number of women
  ∀ (men : ℕ),
  men = 2 * women →
  -- Given condition: there are 63 cannoneers
  cannoneers = 63 →
  -- Prove that the total number of people is 378
  men + women = 378 := by
sorry

end NUMINAMATH_CALUDE_battle_gathering_count_l2110_211019


namespace NUMINAMATH_CALUDE_triangle_area_theorem_l2110_211049

theorem triangle_area_theorem (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (∃ (x y : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ 4*a*x + 4*b*y = 48) →
  ((1/2) * (12/a) * (12/b) = 48) →
  a * b = 3/2 := by sorry

end NUMINAMATH_CALUDE_triangle_area_theorem_l2110_211049


namespace NUMINAMATH_CALUDE_second_car_speed_l2110_211098

/-- Given two cars traveling in the same direction for 3 hours, with one car
    traveling at 50 mph and ending up 60 miles ahead of the other car,
    prove that the speed of the second car is 30 mph. -/
theorem second_car_speed (speed1 : ℝ) (speed2 : ℝ) (time : ℝ) (distance_diff : ℝ)
    (h1 : speed1 = 50)
    (h2 : time = 3)
    (h3 : distance_diff = 60)
    (h4 : speed1 * time - speed2 * time = distance_diff) :
    speed2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_second_car_speed_l2110_211098


namespace NUMINAMATH_CALUDE_find_number_l2110_211073

theorem find_number : ∃! x : ℝ, 10 * ((2 * (x^2 + 2) + 3) / 5) = 50 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l2110_211073


namespace NUMINAMATH_CALUDE_equation_solution_l2110_211081

theorem equation_solution :
  ∃ x : ℝ, 45 - 5 = 3 * x + 10 ∧ x = 10 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2110_211081


namespace NUMINAMATH_CALUDE_determinant_theorem_l2110_211007

theorem determinant_theorem (a b c d : ℝ) : 
  a * d - b * c = -3 → 
  (a + 2*b) * d - (2*b - d) * (3*c) = -3 - 5*b*c + 2*b*d + 3*c*d := by
sorry

end NUMINAMATH_CALUDE_determinant_theorem_l2110_211007


namespace NUMINAMATH_CALUDE_apple_lovers_count_l2110_211040

/-- The number of people who like apple -/
def apple_lovers : ℕ := 40

/-- The number of people who like orange and mango but dislike apple -/
def orange_mango_lovers : ℕ := 7

/-- The number of people who like mango and apple and dislike orange -/
def mango_apple_lovers : ℕ := 10

/-- The number of people who like all three fruits -/
def all_fruit_lovers : ℕ := 4

/-- Theorem stating that the number of people who like apple is 40 -/
theorem apple_lovers_count : apple_lovers = 40 := by
  sorry

end NUMINAMATH_CALUDE_apple_lovers_count_l2110_211040


namespace NUMINAMATH_CALUDE_rectangle_diagonal_after_expansion_l2110_211026

/-- Given a rectangle with width 10 meters and area 150 square meters,
    if its length is increased such that the new area is 3.7 times the original area,
    then the length of the diagonal of the new rectangle is approximately 56.39 meters. -/
theorem rectangle_diagonal_after_expansion (ε : ℝ) (ε_pos : ε > 0) :
  ∃ (original_length new_length diagonal : ℝ),
    original_length > 0 ∧
    new_length > 0 ∧
    diagonal > 0 ∧
    10 * original_length = 150 ∧
    10 * new_length = 3.7 * 150 ∧
    diagonal ^ 2 = 10 ^ 2 + new_length ^ 2 ∧
    |diagonal - 56.39| < ε :=
by sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_after_expansion_l2110_211026


namespace NUMINAMATH_CALUDE_childrens_cookbook_cost_l2110_211025

theorem childrens_cookbook_cost 
  (dictionary_cost : ℕ)
  (dinosaur_book_cost : ℕ)
  (saved_amount : ℕ)
  (additional_amount_needed : ℕ)
  (h1 : dictionary_cost = 11)
  (h2 : dinosaur_book_cost = 19)
  (h3 : saved_amount = 8)
  (h4 : additional_amount_needed = 29) :
  saved_amount + additional_amount_needed - (dictionary_cost + dinosaur_book_cost) = 7 := by
  sorry

end NUMINAMATH_CALUDE_childrens_cookbook_cost_l2110_211025


namespace NUMINAMATH_CALUDE_bronze_to_silver_ratio_l2110_211056

def total_watches : ℕ := 88
def silver_watches : ℕ := 20
def gold_watches : ℕ := 9

def bronze_watches : ℕ := total_watches - silver_watches - gold_watches

theorem bronze_to_silver_ratio :
  bronze_watches * 20 = silver_watches * 59 := by sorry

end NUMINAMATH_CALUDE_bronze_to_silver_ratio_l2110_211056


namespace NUMINAMATH_CALUDE_second_to_first_angle_ratio_l2110_211086

def Triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 180

theorem second_to_first_angle_ratio 
  (a b c : ℝ) 
  (h_triangle : Triangle a b c)
  (h_second_multiple : ∃ k : ℝ, b = k * a)
  (h_third : c = 2 * a - 12)
  (h_measures : a = 32 ∧ b = 96 ∧ c = 52) :
  b / a = 3 := by
sorry

end NUMINAMATH_CALUDE_second_to_first_angle_ratio_l2110_211086


namespace NUMINAMATH_CALUDE_circle_equation_is_correct_l2110_211023

/-- A circle with center on the x-axis, radius 2, and passing through (1, 2) -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  passes_through : ℝ × ℝ
  center_on_x_axis : center.2 = 0
  radius_is_2 : radius = 2
  passes_through_point : passes_through = (1, 2)

/-- The equation of the circle -/
def circle_equation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + y^2 = c.radius^2

theorem circle_equation_is_correct (c : Circle) :
  circle_equation c = λ x y ↦ (x - 1)^2 + y^2 = 4 := by
sorry

end NUMINAMATH_CALUDE_circle_equation_is_correct_l2110_211023


namespace NUMINAMATH_CALUDE_special_function_value_l2110_211046

/-- A function satisfying f(xy) = f(x)/y for all positive real numbers x and y -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), x > 0 → y > 0 → f (x * y) = f x / y

theorem special_function_value (f : ℝ → ℝ) 
  (h : special_function f) (h1000 : f 1000 = 2) : f 750 = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_special_function_value_l2110_211046


namespace NUMINAMATH_CALUDE_fifteenth_even_multiple_of_four_l2110_211001

-- Define a function that represents the nth positive integer that is both even and a multiple of 4
def evenMultipleOfFour (n : ℕ) : ℕ := 4 * n

-- State the theorem
theorem fifteenth_even_multiple_of_four : evenMultipleOfFour 15 = 60 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_even_multiple_of_four_l2110_211001


namespace NUMINAMATH_CALUDE_apples_given_to_teachers_l2110_211033

/-- Given Sarah's apple distribution, prove the number given to teachers. -/
theorem apples_given_to_teachers 
  (initial_apples : Nat) 
  (final_apples : Nat) 
  (friends_given_apples : Nat) 
  (apples_eaten : Nat) 
  (h1 : initial_apples = 25)
  (h2 : final_apples = 3)
  (h3 : friends_given_apples = 5)
  (h4 : apples_eaten = 1) :
  initial_apples - final_apples - friends_given_apples - apples_eaten = 16 := by
  sorry

#check apples_given_to_teachers

end NUMINAMATH_CALUDE_apples_given_to_teachers_l2110_211033


namespace NUMINAMATH_CALUDE_mexica_numbers_less_than_2019_l2110_211032

/-- The number of positive divisors of n -/
def d (n : ℕ+) : ℕ := (Nat.divisors n.val).card

/-- A natural number is mexica if it's of the form n^(d(n)) -/
def is_mexica (m : ℕ) : Prop :=
  ∃ n : ℕ+, m = n.val ^ (d n)

/-- The set of mexica numbers less than 2019 -/
def mexica_set : Finset ℕ :=
  {1, 4, 9, 25, 49, 121, 169, 289, 361, 529, 841, 961, 1369, 1681, 1849, 64, 1296}

theorem mexica_numbers_less_than_2019 :
  {m : ℕ | is_mexica m ∧ m < 2019} = mexica_set := by sorry

end NUMINAMATH_CALUDE_mexica_numbers_less_than_2019_l2110_211032


namespace NUMINAMATH_CALUDE_ellipse_triangle_perimeter_l2110_211039

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 16 = 1

-- Define the foci
def F₁ : ℝ × ℝ := sorry
def F₂ : ℝ × ℝ := sorry

-- Define a point on the ellipse
def point_on_ellipse (p : ℝ × ℝ) : Prop :=
  ellipse p.1 p.2

-- Define collinearity
def collinear (p q r : ℝ × ℝ) : Prop := sorry

-- Define the distance between two points
def distance (p q : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem ellipse_triangle_perimeter 
  (A B : ℝ × ℝ) 
  (h₁ : point_on_ellipse A) 
  (h₂ : point_on_ellipse B) 
  (h₃ : collinear A B F₂) :
  distance F₁ A + distance F₁ B + distance A B = 20 := 
sorry

end NUMINAMATH_CALUDE_ellipse_triangle_perimeter_l2110_211039


namespace NUMINAMATH_CALUDE_reflection_over_x_axis_l2110_211005

def reflection_matrix : Matrix (Fin 2) (Fin 2) ℝ := !![1, 0; 0, -1]

def reflects_over_x_axis (M : Matrix (Fin 2) (Fin 2) ℝ) : Prop :=
  ∀ (x y : ℝ), M.mulVec ![x, y] = ![x, -y]

theorem reflection_over_x_axis :
  reflects_over_x_axis reflection_matrix := by sorry

end NUMINAMATH_CALUDE_reflection_over_x_axis_l2110_211005


namespace NUMINAMATH_CALUDE_multiply_square_roots_l2110_211070

theorem multiply_square_roots : -2 * Real.sqrt 10 * (3 * Real.sqrt 30) = -60 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_multiply_square_roots_l2110_211070


namespace NUMINAMATH_CALUDE_function_satisfying_divisibility_l2110_211074

theorem function_satisfying_divisibility (f : ℕ+ → ℕ+) :
  (∀ a b : ℕ+, (f a + b) ∣ (a^2 + f a * f b)) →
  ∀ n : ℕ+, f n = n :=
by sorry

end NUMINAMATH_CALUDE_function_satisfying_divisibility_l2110_211074


namespace NUMINAMATH_CALUDE_add_1457_minutes_to_3pm_l2110_211051

/-- Represents time in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  let newHours := totalMinutes / 60 % 24
  let newMinutes := totalMinutes % 60
  { hours := newHours, minutes := newMinutes }

/-- Theorem: Adding 1457 minutes to 3:00 p.m. results in 3:17 p.m. the next day -/
theorem add_1457_minutes_to_3pm (initial : Time) (final : Time) :
  initial = { hours := 15, minutes := 0 } →
  final = addMinutes initial 1457 →
  final = { hours := 15, minutes := 17 } :=
by sorry

end NUMINAMATH_CALUDE_add_1457_minutes_to_3pm_l2110_211051


namespace NUMINAMATH_CALUDE_convergence_of_difference_series_l2110_211017

open Topology
open Real

-- Define a monotonic sequence
def IsMonotonic (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n ≤ m → a n ≤ a m ∨ ∀ n m : ℕ, n ≤ m → a m ≤ a n

-- Define the theorem
theorem convergence_of_difference_series (a : ℕ → ℝ) 
  (h_monotonic : IsMonotonic a) 
  (h_converge : Summable a) :
  Summable (fun n => n • (a n - a (n + 1))) :=
sorry

end NUMINAMATH_CALUDE_convergence_of_difference_series_l2110_211017


namespace NUMINAMATH_CALUDE_linear_function_not_in_second_quadrant_l2110_211035

/-- Represents a point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a linear function y = kx + b -/
structure LinearFunction where
  k : ℝ
  b : ℝ

/-- Checks if a point is in the second quadrant -/
def isInSecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Theorem: A linear function with k > 0 and b < 0 does not pass through the second quadrant -/
theorem linear_function_not_in_second_quadrant (f : LinearFunction) 
    (h1 : f.k > 0) (h2 : f.b < 0) : 
    ∀ p : Point, p.y = f.k * p.x + f.b → ¬(isInSecondQuadrant p) := by
  sorry

end NUMINAMATH_CALUDE_linear_function_not_in_second_quadrant_l2110_211035


namespace NUMINAMATH_CALUDE_function_periodicity_l2110_211036

def periodic_function (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x, f (x + c) = f x

theorem function_periodicity 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, |f x| ≤ 1)
  (h2 : ∀ x, f (x + 13/42) + f x = f (x + 1/6) + f (x + 1/7)) :
  ∃ c > 0, periodic_function f c ∧ c = 1 :=
sorry

end NUMINAMATH_CALUDE_function_periodicity_l2110_211036


namespace NUMINAMATH_CALUDE_cube_prism_cuboid_rectangular_prism_subset_l2110_211091

-- Define the sets
variable (M : Set (Set ℝ)) -- Set of all right prisms
variable (N : Set (Set ℝ)) -- Set of all cuboids
variable (Q : Set (Set ℝ)) -- Set of all cubes
variable (P : Set (Set ℝ)) -- Set of all right rectangular prisms

-- State the theorem
theorem cube_prism_cuboid_rectangular_prism_subset : Q ⊆ M ∧ M ⊆ N ∧ N ⊆ P := by
  sorry

end NUMINAMATH_CALUDE_cube_prism_cuboid_rectangular_prism_subset_l2110_211091


namespace NUMINAMATH_CALUDE_cooking_cleaning_combinations_l2110_211022

-- Define the number of friends
def total_friends : ℕ := 5

-- Define the number of cooks
def num_cooks : ℕ := 3

-- Define the combination function
def combination (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- Theorem statement
theorem cooking_cleaning_combinations :
  combination total_friends num_cooks = 10 := by
  sorry

end NUMINAMATH_CALUDE_cooking_cleaning_combinations_l2110_211022


namespace NUMINAMATH_CALUDE_heart_shaped_chocolate_weight_l2110_211042

/-- Represents the weight of a chocolate bar -/
def chocolate_bar_weight (whole_squares : ℕ) (triangles : ℕ) (square_weight : ℕ) : ℕ :=
  whole_squares * square_weight + triangles * (square_weight / 2)

/-- Theorem stating the weight of the heart-shaped chocolate bar -/
theorem heart_shaped_chocolate_weight :
  chocolate_bar_weight 32 16 6 = 240 := by
  sorry

end NUMINAMATH_CALUDE_heart_shaped_chocolate_weight_l2110_211042


namespace NUMINAMATH_CALUDE_range_of_x_range_of_a_l2110_211053

-- Define propositions p and q
def p (a x : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0
def q (x : ℝ) : Prop := -x^2 + 5*x - 6 ≥ 0

-- Part 1
theorem range_of_x (x : ℝ) (h : p 1 x ∧ q x) : 2 ≤ x ∧ x < 3 :=
sorry

-- Part 2
theorem range_of_a (a : ℝ) (h1 : a > 0) 
  (h2 : ∀ x, ¬(p a x) → ¬(q x)) 
  (h3 : ∃ x, q x ∧ p a x) : 
  1 < a ∧ a < 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_x_range_of_a_l2110_211053


namespace NUMINAMATH_CALUDE_incorrect_transformation_l2110_211083

theorem incorrect_transformation (a b : ℝ) (h : a > b) : ¬(3 - a > 3 - b) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_transformation_l2110_211083


namespace NUMINAMATH_CALUDE_students_making_stars_l2110_211088

theorem students_making_stars (total_stars : ℕ) (stars_per_student : ℕ) (h1 : total_stars = 372) (h2 : stars_per_student = 3) :
  total_stars / stars_per_student = 124 := by
  sorry

end NUMINAMATH_CALUDE_students_making_stars_l2110_211088


namespace NUMINAMATH_CALUDE_inequality_one_inequality_two_l2110_211068

-- Part 1
theorem inequality_one (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (b + c - a) / a + (c + a - b) / b + (a + b - c) / c ≥ 3 := by sorry

-- Part 2
theorem inequality_two (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_one : a + b + c = 1) :
  a * b + b * c + a * c ≤ 1 / 3 := by sorry

end NUMINAMATH_CALUDE_inequality_one_inequality_two_l2110_211068


namespace NUMINAMATH_CALUDE_early_arrival_l2110_211038

/-- Given a boy who usually takes 14 minutes to reach school, if he walks at 7/6 of his usual rate, he will arrive 2 minutes early. -/
theorem early_arrival (usual_time : ℝ) (new_rate : ℝ) : 
  usual_time = 14 → new_rate = 7/6 → usual_time - (usual_time / new_rate) = 2 :=
by sorry

end NUMINAMATH_CALUDE_early_arrival_l2110_211038


namespace NUMINAMATH_CALUDE_geometric_sequence_general_term_l2110_211006

/-- A geometric sequence with positive terms satisfying a certain relation -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧
  (∃ q : ℝ, q > 1 ∧ ∀ n, a (n + 1) = q * a n) ∧
  (∀ n, 2 * (a n + a (n + 2)) = 5 * a (n + 1))

/-- The general term of the geometric sequence -/
def GeneralTerm (a : ℕ → ℝ) : Prop :=
  ∃ a₁ : ℝ, ∀ n, a n = a₁ * 2^(n - 1)

theorem geometric_sequence_general_term (a : ℕ → ℝ) :
  GeometricSequence a → GeneralTerm a := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_general_term_l2110_211006


namespace NUMINAMATH_CALUDE_peanuts_added_l2110_211066

theorem peanuts_added (initial_peanuts final_peanuts : ℕ) : 
  initial_peanuts = 4 →
  final_peanuts = 16 →
  final_peanuts - initial_peanuts = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_peanuts_added_l2110_211066


namespace NUMINAMATH_CALUDE_even_function_properties_l2110_211003

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define what it means for a function to be even
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g x = g (-x)

-- Define what it means for a function to be symmetric about a vertical line
def symmetric_about_vertical_line (g : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, g (a + x) = g (a - x)

-- Define what it means for a function to be symmetric about the y-axis
def symmetric_about_y_axis (g : ℝ → ℝ) : Prop :=
  ∀ x, g x = g (-x)

-- State the theorem
theorem even_function_properties (h : is_even (fun x => f (x + 1))) :
  (symmetric_about_vertical_line f 1) ∧
  (symmetric_about_y_axis (fun x => f (x + 1))) ∧
  (∀ x, f (1 + x) = f (1 - x)) :=
by sorry

end NUMINAMATH_CALUDE_even_function_properties_l2110_211003


namespace NUMINAMATH_CALUDE_function_range_l2110_211067

/-- Given a function f(x) = x³ - 3a²x + a where a > 0, 
    if its maximum value is positive and its minimum value is negative, 
    then a > √2/2 -/
theorem function_range (a : ℝ) (h1 : a > 0) 
  (f : ℝ → ℝ) (h2 : ∀ x, f x = x^3 - 3*a^2*x + a) 
  (h3 : ∃ M, ∀ x, f x ≤ M ∧ M > 0)  -- maximum value is positive
  (h4 : ∃ m, ∀ x, f x ≥ m ∧ m < 0)  -- minimum value is negative
  : a > Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_function_range_l2110_211067


namespace NUMINAMATH_CALUDE_parallel_linear_functions_theorem_l2110_211087

/-- Two linear functions with parallel graphs not parallel to coordinate axes -/
structure ParallelLinearFunctions where
  f : ℝ → ℝ
  g : ℝ → ℝ
  parallel : ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x + b ∧ g x = a * x + c
  not_axis_parallel : ∀ (a b c : ℝ), (∀ x, f x = a * x + b ∧ g x = a * x + c) → a ≠ 0

/-- The condition that (f(x))^2 touches -6g(x) -/
def touches_neg_6g (p : ParallelLinearFunctions) : Prop :=
  ∃! x, (p.f x)^2 = -6 * p.g x

/-- The condition that (g(x))^2 touches Af(x) -/
def touches_Af (p : ParallelLinearFunctions) (A : ℝ) : Prop :=
  ∃! x, (p.g x)^2 = A * p.f x

/-- The main theorem -/
theorem parallel_linear_functions_theorem (p : ParallelLinearFunctions) 
  (h : touches_neg_6g p) : 
  ∀ A, touches_Af p A ↔ (A = 6 ∨ A = 0) := by
  sorry

end NUMINAMATH_CALUDE_parallel_linear_functions_theorem_l2110_211087


namespace NUMINAMATH_CALUDE_max_value_constraint_l2110_211054

theorem max_value_constraint (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a^2 + b^2 + c^2 = 1) :
  2 * a * b * Real.sqrt 3 + 2 * a * c ≤ 8/5 := by
  sorry

end NUMINAMATH_CALUDE_max_value_constraint_l2110_211054


namespace NUMINAMATH_CALUDE_power_multiplication_l2110_211011

theorem power_multiplication (x : ℝ) : x^2 * x^3 = x^5 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l2110_211011


namespace NUMINAMATH_CALUDE_inscribed_circle_diameter_l2110_211021

theorem inscribed_circle_diameter (DE DF EF : ℝ) (h1 : DE = 13) (h2 : DF = 8) (h3 : EF = 9) :
  let s := (DE + DF + EF) / 2
  let area := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF))
  let diameter := 2 * area / s
  diameter = 4 * Real.sqrt 35 / 5 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_diameter_l2110_211021


namespace NUMINAMATH_CALUDE_sin_cos_fourth_power_range_l2110_211034

theorem sin_cos_fourth_power_range (x : ℝ) : 
  0.5 ≤ Real.sin x ^ 4 + Real.cos x ^ 4 ∧ Real.sin x ^ 4 + Real.cos x ^ 4 ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_fourth_power_range_l2110_211034


namespace NUMINAMATH_CALUDE_log_equality_implies_y_value_l2110_211058

-- Define the logarithm function
noncomputable def log : ℝ → ℝ := sorry

-- Define the variables
variable (a b c x : ℝ)
variable (p q r y : ℝ)

-- State the theorem
theorem log_equality_implies_y_value
  (h1 : log a / p = log b / q)
  (h2 : log b / q = log c / r)
  (h3 : log c / r = log x)
  (h4 : x ≠ 1)
  (h5 : b^3 / (a^2 * c) = x^y) :
  y = 3*q - 2*p - r := by
  sorry

end NUMINAMATH_CALUDE_log_equality_implies_y_value_l2110_211058


namespace NUMINAMATH_CALUDE_missing_angles_sum_l2110_211084

-- Define the properties of our polygon
def ConvexPolygon (n : ℕ) (knownSum missingSum : ℝ) : Prop :=
  -- The polygon has n sides
  n > 2 ∧
  -- The sum of known angles is 1620°
  knownSum = 1620 ∧
  -- There are two missing angles
  -- The total sum (known + missing) is divisible by 180°
  ∃ (k : ℕ), (knownSum + missingSum) = 180 * k

-- State the theorem
theorem missing_angles_sum (n : ℕ) (knownSum missingSum : ℝ) 
  (h : ConvexPolygon n knownSum missingSum) : missingSum = 180 := by
  sorry

end NUMINAMATH_CALUDE_missing_angles_sum_l2110_211084


namespace NUMINAMATH_CALUDE_coefficient_implies_a_value_l2110_211008

theorem coefficient_implies_a_value (a : ℝ) : 
  (5 / 2) * a^3 = 20 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_implies_a_value_l2110_211008


namespace NUMINAMATH_CALUDE_pentagonal_prism_with_pyramid_sum_l2110_211093

/-- A shape formed by adding a pyramid to one pentagonal face of a pentagonal prism -/
structure PentagonalPrismWithPyramid where
  faces : ℕ
  vertices : ℕ
  edges : ℕ

/-- The sum of faces, vertices, and edges for a PentagonalPrismWithPyramid -/
def PentagonalPrismWithPyramid.sum (shape : PentagonalPrismWithPyramid) : ℕ :=
  shape.faces + shape.vertices + shape.edges

theorem pentagonal_prism_with_pyramid_sum :
  ∃ (shape : PentagonalPrismWithPyramid), shape.sum = 42 :=
sorry

end NUMINAMATH_CALUDE_pentagonal_prism_with_pyramid_sum_l2110_211093
