import Mathlib

namespace zoo_animal_difference_l987_98725

theorem zoo_animal_difference : ∀ (parrots snakes monkeys elephants zebras : ℕ),
  parrots = 8 →
  snakes = 3 * parrots →
  monkeys = 2 * snakes →
  elephants = (parrots + snakes) / 2 →
  zebras + 3 = elephants →
  monkeys - zebras = 35 := by
  sorry

end zoo_animal_difference_l987_98725


namespace line_passes_through_point_l987_98731

theorem line_passes_through_point :
  ∀ (m : ℝ), m * (-1) - 2 + m + 2 = 0 := by sorry

end line_passes_through_point_l987_98731


namespace cube_root_of_x_sqrt_x_l987_98775

theorem cube_root_of_x_sqrt_x (x : ℝ) (hx : x > 0) : 
  (x * Real.sqrt x) ^ (1/3 : ℝ) = x ^ (1/2 : ℝ) := by sorry

end cube_root_of_x_sqrt_x_l987_98775


namespace line_ellipse_intersection_range_l987_98777

/-- The range of m for which a line y = kx + 1 and an ellipse (x^2)/5 + (y^2)/m = 1 always have common points -/
theorem line_ellipse_intersection_range (k : ℝ) :
  (∀ x y : ℝ, y = k * x + 1 → x^2 / 5 + y^2 / m = 1 → (∃ x' y' : ℝ, y' = k * x' + 1 ∧ x'^2 / 5 + y'^2 / m = 1)) →
  m ≥ 1 ∧ m ≠ 5 :=
sorry

end line_ellipse_intersection_range_l987_98777


namespace students_just_passed_l987_98717

theorem students_just_passed (total_students : ℕ) 
  (first_division_percent : ℚ) (second_division_percent : ℚ) :
  total_students = 300 →
  first_division_percent = 26 / 100 →
  second_division_percent = 54 / 100 →
  (total_students : ℚ) * (1 - first_division_percent - second_division_percent) = 60 := by
  sorry

end students_just_passed_l987_98717


namespace geordie_commute_cost_l987_98739

/-- Represents the cost calculation for Geordie's weekly commute -/
def weekly_commute_cost (car_toll : ℚ) (motorcycle_toll : ℚ) (mpg : ℚ) (distance : ℚ) (gas_price : ℚ) (car_trips : ℕ) (motorcycle_trips : ℕ) : ℚ :=
  let total_toll := car_toll * car_trips + motorcycle_toll * motorcycle_trips
  let total_miles := (distance * 2) * (car_trips + motorcycle_trips)
  let total_gas_cost := (total_miles / mpg) * gas_price
  total_toll + total_gas_cost

/-- Theorem stating that Geordie's weekly commute cost is $66.50 -/
theorem geordie_commute_cost :
  weekly_commute_cost 12.5 7 35 14 3.75 3 2 = 66.5 := by
  sorry

end geordie_commute_cost_l987_98739


namespace johnny_pencil_packs_l987_98769

theorem johnny_pencil_packs :
  ∀ (total_red_pencils : ℕ) (extra_red_packs : ℕ) (extra_red_per_pack : ℕ),
    total_red_pencils = 21 →
    extra_red_packs = 3 →
    extra_red_per_pack = 2 →
    ∃ (total_packs : ℕ),
      total_packs = (total_red_pencils - extra_red_packs * extra_red_per_pack) + extra_red_packs ∧
      total_packs = 18 :=
by sorry

end johnny_pencil_packs_l987_98769


namespace third_angle_relationship_l987_98759

theorem third_angle_relationship (a b c : ℝ) : 
  a = b → a = 36 → a + b + c = 180 → c = 3 * a := by sorry

end third_angle_relationship_l987_98759


namespace only_class_math_scores_comprehensive_l987_98771

/-- Represents a survey scenario -/
inductive SurveyScenario
  | NationwideVision
  | LightBulbLifespan
  | ClassMathScores
  | DistrictIncome

/-- Determines if a survey scenario is suitable for a comprehensive survey -/
def isSuitableForComprehensiveSurvey (scenario : SurveyScenario) : Prop :=
  match scenario with
  | .ClassMathScores => true
  | _ => false

/-- The main theorem stating that only ClassMathScores is suitable for a comprehensive survey -/
theorem only_class_math_scores_comprehensive :
  ∀ (scenario : SurveyScenario),
    isSuitableForComprehensiveSurvey scenario ↔ scenario = SurveyScenario.ClassMathScores :=
by
  sorry

/-- Helper lemma: NationwideVision is not suitable for a comprehensive survey -/
lemma nationwide_vision_not_comprehensive :
  ¬ isSuitableForComprehensiveSurvey SurveyScenario.NationwideVision :=
by
  sorry

/-- Helper lemma: LightBulbLifespan is not suitable for a comprehensive survey -/
lemma light_bulb_lifespan_not_comprehensive :
  ¬ isSuitableForComprehensiveSurvey SurveyScenario.LightBulbLifespan :=
by
  sorry

/-- Helper lemma: DistrictIncome is not suitable for a comprehensive survey -/
lemma district_income_not_comprehensive :
  ¬ isSuitableForComprehensiveSurvey SurveyScenario.DistrictIncome :=
by
  sorry

/-- Helper lemma: ClassMathScores is suitable for a comprehensive survey -/
lemma class_math_scores_comprehensive :
  isSuitableForComprehensiveSurvey SurveyScenario.ClassMathScores :=
by
  sorry

end only_class_math_scores_comprehensive_l987_98771


namespace inverse_variation_problem_l987_98726

/-- Given that y^3 varies inversely with z^2 and y = 3 when z = 2, 
    prove that z = √2/2 when y = 6 -/
theorem inverse_variation_problem (y z : ℝ) (k : ℝ) :
  (∀ y z, y^3 * z^2 = k) →  -- y^3 varies inversely with z^2
  (3^3 * 2^2 = k) →         -- y = 3 when z = 2
  (6^3 * z^2 = k) →         -- condition for y = 6
  z = Real.sqrt 2 / 2 :=    -- z = √2/2 when y = 6
by sorry

end inverse_variation_problem_l987_98726


namespace circle_center_radius_l987_98797

theorem circle_center_radius (x y : ℝ) :
  x^2 + y^2 - 4*x = 0 → (∃ (center : ℝ × ℝ) (radius : ℝ), 
    center = (2, 0) ∧ radius = 2 ∧ 
    (x - center.1)^2 + (y - center.2)^2 = radius^2) :=
by sorry

end circle_center_radius_l987_98797


namespace simple_interest_calculation_l987_98762

/-- Given a principal amount P, prove that if the compound interest at 4% for 2 years is $612,
    then the simple interest at 4% for 2 years is $600. -/
theorem simple_interest_calculation (P : ℝ) : 
  P * (1 + 0.04)^2 - P = 612 → P * 0.04 * 2 = 600 := by
  sorry

end simple_interest_calculation_l987_98762


namespace last_four_digits_pow_5_2017_l987_98746

/-- The last four digits of a natural number -/
def lastFourDigits (n : ℕ) : ℕ := n % 10000

/-- The cycle length of the last four digits of powers of 5 -/
def cycleLengthPowersOf5 : ℕ := 4

theorem last_four_digits_pow_5_2017 :
  lastFourDigits (5^2017) = lastFourDigits (5^5) :=
sorry

end last_four_digits_pow_5_2017_l987_98746


namespace ice_cream_parlor_distance_l987_98799

/-- The distance to the ice cream parlor -/
def D : ℝ := sorry

/-- Rita's upstream paddling speed -/
def upstream_speed : ℝ := 3

/-- Rita's downstream paddling speed -/
def downstream_speed : ℝ := 9

/-- Upstream wind speed -/
def upstream_wind : ℝ := 2

/-- Downstream wind speed -/
def downstream_wind : ℝ := 4

/-- Total trip time -/
def total_time : ℝ := 8

/-- Effective upstream speed -/
def effective_upstream_speed : ℝ := upstream_speed - upstream_wind

/-- Effective downstream speed -/
def effective_downstream_speed : ℝ := downstream_speed + downstream_wind

theorem ice_cream_parlor_distance : 
  D / effective_upstream_speed + D / effective_downstream_speed = total_time := by sorry

end ice_cream_parlor_distance_l987_98799


namespace valentines_day_cards_l987_98744

theorem valentines_day_cards (total_students : ℕ) (card_cost : ℚ) (total_money : ℚ) 
  (spend_percentage : ℚ) (h1 : total_students = 30) (h2 : card_cost = 2) 
  (h3 : total_money = 40) (h4 : spend_percentage = 0.9) : 
  (((total_money * spend_percentage) / card_cost) / total_students) * 100 = 60 := by
  sorry

end valentines_day_cards_l987_98744


namespace largest_multiple_of_seven_l987_98710

theorem largest_multiple_of_seven (n : ℤ) : n = 77 ↔ 
  (∃ k : ℤ, n = 7 * k) ∧ 
  (-n > -80) ∧
  (∀ m : ℤ, (∃ j : ℤ, m = 7 * j) → (-m > -80) → m ≤ n) := by
  sorry

end largest_multiple_of_seven_l987_98710


namespace shopping_solution_l987_98705

/-- The cost of Liz's shopping trip -/
def shopping_problem (recipe_book_cost : ℝ) : Prop :=
  let baking_dish_cost := 2 * recipe_book_cost
  let ingredients_cost := 5 * 3
  let apron_cost := recipe_book_cost + 1
  recipe_book_cost + baking_dish_cost + ingredients_cost + apron_cost = 40

/-- The solution to the shopping problem -/
theorem shopping_solution : ∃ (recipe_book_cost : ℝ), 
  shopping_problem recipe_book_cost ∧ recipe_book_cost = 6 := by
  sorry

end shopping_solution_l987_98705


namespace cookies_sold_first_village_l987_98733

/-- Given the total number of packs sold and the number sold in the second village,
    calculate the number of packs sold in the first village. -/
theorem cookies_sold_first_village 
  (total_packs : ℕ) 
  (second_village_packs : ℕ) 
  (h1 : total_packs = 51) 
  (h2 : second_village_packs = 28) : 
  total_packs - second_village_packs = 23 := by
  sorry

end cookies_sold_first_village_l987_98733


namespace max_pencil_length_in_square_hallway_l987_98770

/-- Represents the length of a pencil that can navigate a square turn in a hallway -/
def max_pencil_length (L : ℝ) : ℝ := 3 * L

/-- Theorem stating that the maximum length of a pencil that can navigate a square turn
    in a hallway of width and height L is 3L -/
theorem max_pencil_length_in_square_hallway (L : ℝ) (h : L > 0) :
  max_pencil_length L = 3 * L :=
by sorry

end max_pencil_length_in_square_hallway_l987_98770


namespace jerry_total_games_l987_98742

/-- Calculates the total number of games Jerry has after his birthday and trade --/
def total_games_after (initial_action : ℕ) (initial_strategy : ℕ) 
  (action_increase_percent : ℕ) (strategy_increase_percent : ℕ) 
  (action_traded : ℕ) (sports_received : ℕ) : ℕ :=
  let action_increase := (initial_action * action_increase_percent) / 100
  let strategy_increase := (initial_strategy * strategy_increase_percent) / 100
  let final_action := initial_action + action_increase - action_traded
  let final_strategy := initial_strategy + strategy_increase
  final_action + final_strategy + sports_received

/-- Theorem stating that Jerry's total games after birthday and trade is 16 --/
theorem jerry_total_games : 
  total_games_after 7 5 30 20 2 3 = 16 := by sorry

end jerry_total_games_l987_98742


namespace nested_square_roots_simplification_l987_98736

theorem nested_square_roots_simplification :
  Real.sqrt (36 * Real.sqrt (18 * Real.sqrt 9)) = 6 * Real.sqrt 6 := by
  sorry

end nested_square_roots_simplification_l987_98736


namespace delta_negative_two_three_l987_98728

-- Define the Delta operation
def Delta (a b : ℝ) : ℝ := a * b^2 + b + 1

-- Theorem statement
theorem delta_negative_two_three : Delta (-2) 3 = -14 := by
  sorry

end delta_negative_two_three_l987_98728


namespace particle_movement_l987_98713

def num_ways_to_point (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

theorem particle_movement :
  (num_ways_to_point 5 4 = 5) ∧ (num_ways_to_point 20 18 = 190) := by
  sorry

end particle_movement_l987_98713


namespace arithmetic_geometric_sequence_l987_98734

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) (d : ℝ) (h1 : d ≠ 0) 
  (h2 : ∀ n, a (n + 1) = a n + d) 
  (h3 : ∃ r, (a 3) / (a 2) = r ∧ (a 6) / (a 3) = r) : 
  (a 3) / (a 2) = 3 :=
sorry

end arithmetic_geometric_sequence_l987_98734


namespace f_properties_l987_98741

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 1 / 4^x - 1 / 2^x else 2^x - 4^x

theorem f_properties :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, f (-x) = -f x) ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) 0, f x = 1 / 4^x - 1 / 2^x) ∧
  (∀ x ∈ Set.Icc 0 1, f x = 2^x - 4^x) ∧
  (∀ x ∈ Set.Icc 0 1, f x ≤ 0) ∧
  (∃ x ∈ Set.Icc 0 1, f x = 0) := by
  sorry

#check f_properties

end f_properties_l987_98741


namespace harper_mineral_water_cost_l987_98763

/-- Harper's mineral water purchase problem -/
theorem harper_mineral_water_cost 
  (daily_consumption : ℚ) 
  (bottles_per_case : ℕ) 
  (cost_per_case : ℚ) 
  (days : ℕ) : 
  daily_consumption = 1/2 → 
  bottles_per_case = 24 → 
  cost_per_case = 12 → 
  days = 240 → 
  (days * daily_consumption / bottles_per_case).ceil * cost_per_case = 60 := by
  sorry

end harper_mineral_water_cost_l987_98763


namespace not_perfect_square_special_number_l987_98780

/-- A 100-digit number with all digits as fives except one is not a perfect square. -/
theorem not_perfect_square_special_number : 
  ∀ n : ℕ, 
  (n ≥ 10^99 ∧ n < 10^100) →  -- 100-digit number
  (∃! d : ℕ, d < 10 ∧ d ≠ 5 ∧ 
    ∀ i : ℕ, i < 100 → 
      (n / 10^i) % 10 = if (n / 10^i) % 10 = d then d else 5) →  -- All digits are fives except one
  ¬∃ m : ℕ, n = m^2 :=  -- Not a perfect square
by sorry

end not_perfect_square_special_number_l987_98780


namespace total_distance_walked_and_run_l987_98788

/-- Calculates the total distance traveled when walking and running at different rates for different durations. -/
theorem total_distance_walked_and_run 
  (walking_time : ℝ) (walking_rate : ℝ) (running_time : ℝ) (running_rate : ℝ) :
  walking_time = 45 →
  walking_rate = 4 →
  running_time = 30 →
  running_rate = 10 →
  (walking_time / 60) * walking_rate + (running_time / 60) * running_rate = 8 := by
  sorry

end total_distance_walked_and_run_l987_98788


namespace sum_of_squares_zero_l987_98723

theorem sum_of_squares_zero (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_sum : a / (b - c) + b / (c - a) + c / (a - b) = 0) :
  a^2 / (b - c)^2 + b^2 / (c - a)^2 + c^2 / (a - b)^2 = 0 := by
  sorry

end sum_of_squares_zero_l987_98723


namespace power_fraction_simplification_l987_98772

theorem power_fraction_simplification : (4 : ℝ)^800 / (8 : ℝ)^400 = (2 : ℝ)^400 := by sorry

end power_fraction_simplification_l987_98772


namespace fence_posts_for_grazing_area_l987_98701

/-- The minimum number of fence posts required to enclose a rectangular area -/
def min_fence_posts (length width post_spacing : ℕ) : ℕ :=
  let perimeter := 2 * length + width
  let num_intervals := perimeter / post_spacing
  num_intervals + 1

theorem fence_posts_for_grazing_area :
  min_fence_posts 60 36 12 = 12 :=
sorry

end fence_posts_for_grazing_area_l987_98701


namespace mango_crates_sold_l987_98768

-- Define the types of fruit
inductive Fruit
  | Grapes
  | Mangoes
  | PassionFruits

-- Define the total number of crates sold
def total_crates : ℕ := 50

-- Define the number of grape crates sold
def grape_crates : ℕ := 13

-- Define the number of passion fruit crates sold
def passion_fruit_crates : ℕ := 17

-- Define the function to calculate the number of mango crates
def mango_crates : ℕ := total_crates - (grape_crates + passion_fruit_crates)

-- Theorem statement
theorem mango_crates_sold : mango_crates = 20 := by
  sorry

end mango_crates_sold_l987_98768


namespace angle_from_point_l987_98740

theorem angle_from_point (a : Real) (h1 : 0 < a ∧ a < π/2) : 
  (∃ (x y : Real), x = 4 * Real.sin 3 ∧ y = -4 * Real.cos 3 ∧ 
   x = 4 * Real.sin a ∧ y = -4 * Real.cos a) → 
  a = 3 - π/2 := by
sorry

end angle_from_point_l987_98740


namespace farm_tax_land_percentage_l987_98766

theorem farm_tax_land_percentage 
  (total_tax : ℝ) 
  (individual_tax : ℝ) 
  (h1 : total_tax = 3840) 
  (h2 : individual_tax = 480) :
  individual_tax / total_tax = 0.125 := by
sorry

end farm_tax_land_percentage_l987_98766


namespace equation_solution_l987_98735

/-- The set of solutions for the equation x! + y! = 8z + 2017 -/
def SolutionSet : Set (ℕ × ℕ × ℤ) :=
  {(1, 4, -249), (4, 1, -249), (1, 5, -237), (5, 1, -237)}

/-- The equation x! + y! = 8z + 2017 -/
def Equation (x y : ℕ) (z : ℤ) : Prop :=
  Nat.factorial x + Nat.factorial y = 8 * z + 2017

/-- z is an odd integer -/
def IsOdd (z : ℤ) : Prop :=
  ∃ k : ℤ, z = 2 * k + 1

theorem equation_solution :
  ∀ x y : ℕ, ∀ z : ℤ,
    Equation x y z ∧ IsOdd z ↔ (x, y, z) ∈ SolutionSet :=
sorry

end equation_solution_l987_98735


namespace factorial_15_not_divisible_by_17_l987_98706

theorem factorial_15_not_divisible_by_17 : ¬(17 ∣ Nat.factorial 15) := by
  sorry

end factorial_15_not_divisible_by_17_l987_98706


namespace largest_angle_in_special_triangle_l987_98743

theorem largest_angle_in_special_triangle :
  ∀ (a b c : ℝ),
  a > 0 ∧ b > 0 ∧ c > 0 →
  a + b + c = 180 →
  a + b = 105 →
  b = a + 40 →
  max a (max b c) = 75 :=
sorry

end largest_angle_in_special_triangle_l987_98743


namespace mean_home_runs_l987_98730

theorem mean_home_runs : 
  let players_with_5 := 3
  let players_with_6 := 4
  let players_with_8 := 2
  let players_with_9 := 1
  let players_with_11 := 1
  let total_players := players_with_5 + players_with_6 + players_with_8 + players_with_9 + players_with_11
  let total_home_runs := 5 * players_with_5 + 6 * players_with_6 + 8 * players_with_8 + 9 * players_with_9 + 11 * players_with_11
  (total_home_runs : ℚ) / total_players = 75 / 11 := by
sorry

end mean_home_runs_l987_98730


namespace max_prob_second_game_l987_98720

variable (p₁ p₂ p₃ : ℝ)

def P_A := 2 * (p₁ * (p₂ + p₃) - 2 * p₁ * p₂ * p₃)
def P_B := 2 * (p₂ * (p₁ + p₃) - 2 * p₁ * p₂ * p₃)
def P_C := 2 * (p₁ * p₃ + p₂ * p₃ - 2 * p₁ * p₂ * p₃)

theorem max_prob_second_game (h1 : 0 < p₁) (h2 : p₁ < p₂) (h3 : p₂ < p₃) :
  P_C p₁ p₂ p₃ > P_A p₁ p₂ p₃ ∧ P_C p₁ p₂ p₃ > P_B p₁ p₂ p₃ :=
by sorry

end max_prob_second_game_l987_98720


namespace fourth_number_unit_digit_l987_98716

def unit_digit (n : ℕ) : ℕ := n % 10

def product_unit_digit (a b c d : ℕ) : ℕ :=
  unit_digit (unit_digit a * unit_digit b * unit_digit c * unit_digit d)

theorem fourth_number_unit_digit :
  ∃ (x : ℕ), product_unit_digit 624 708 463 x = 8 ∧ unit_digit x = 3 :=
by sorry

end fourth_number_unit_digit_l987_98716


namespace zoe_earnings_per_candy_bar_l987_98732

def trip_cost : ℚ := 485
def grandma_contribution : ℚ := 250
def candy_bars_to_sell : ℕ := 188

theorem zoe_earnings_per_candy_bar :
  (trip_cost - grandma_contribution) / candy_bars_to_sell = 1.25 := by
  sorry

end zoe_earnings_per_candy_bar_l987_98732


namespace dusty_change_l987_98708

/-- Represents the price of a single layer cake slice in dollars -/
def single_layer_price : ℕ := 4

/-- Represents the price of a double layer cake slice in dollars -/
def double_layer_price : ℕ := 7

/-- Represents the number of single layer cake slices Dusty buys -/
def single_layer_quantity : ℕ := 7

/-- Represents the number of double layer cake slices Dusty buys -/
def double_layer_quantity : ℕ := 5

/-- Represents the amount Dusty pays with in dollars -/
def payment : ℕ := 100

/-- Theorem stating that Dusty's change is $37 -/
theorem dusty_change : 
  payment - (single_layer_price * single_layer_quantity + double_layer_price * double_layer_quantity) = 37 := by
  sorry

end dusty_change_l987_98708


namespace molecular_weight_calculation_l987_98702

/-- The atomic weight of Hydrogen in atomic mass units (amu) -/
def atomic_weight_H : ℝ := 1.008

/-- The atomic weight of Chromium in atomic mass units (amu) -/
def atomic_weight_Cr : ℝ := 51.996

/-- The atomic weight of Oxygen in atomic mass units (amu) -/
def atomic_weight_O : ℝ := 15.999

/-- The number of Hydrogen atoms in the compound -/
def num_H : ℕ := 2

/-- The number of Chromium atoms in the compound -/
def num_Cr : ℕ := 1

/-- The number of Oxygen atoms in the compound -/
def num_O : ℕ := 4

/-- The molecular weight of the compound in atomic mass units (amu) -/
def molecular_weight : ℝ := 
  (num_H : ℝ) * atomic_weight_H + 
  (num_Cr : ℝ) * atomic_weight_Cr + 
  (num_O : ℝ) * atomic_weight_O

theorem molecular_weight_calculation : 
  molecular_weight = 118.008 := by sorry

end molecular_weight_calculation_l987_98702


namespace john_volunteer_hours_per_year_l987_98792

/-- 
Given that John volunteers twice a month for 3 hours each time, 
this theorem proves that he volunteers for 72 hours per year.
-/
theorem john_volunteer_hours_per_year 
  (times_per_month : ℕ) 
  (hours_per_time : ℕ) 
  (h1 : times_per_month = 2) 
  (h2 : hours_per_time = 3) : 
  times_per_month * 12 * hours_per_time = 72 := by
  sorry

end john_volunteer_hours_per_year_l987_98792


namespace two_leq_three_l987_98729

theorem two_leq_three : 2 ≤ 3 := by sorry

end two_leq_three_l987_98729


namespace gcd_of_225_and_135_l987_98779

theorem gcd_of_225_and_135 : Nat.gcd 225 135 = 45 := by
  sorry

end gcd_of_225_and_135_l987_98779


namespace complementary_angle_of_37_38_l987_98753

/-- Represents an angle in degrees and minutes -/
structure Angle :=
  (degrees : ℕ)
  (minutes : ℕ)

/-- The sum of two angles in degrees and minutes -/
def Angle.add (a b : Angle) : Angle :=
  let totalMinutes := a.minutes + b.minutes
  let carryDegrees := totalMinutes / 60
  { degrees := a.degrees + b.degrees + carryDegrees
  , minutes := totalMinutes % 60 }

/-- Checks if two angles are complementary -/
def are_complementary (a b : Angle) : Prop :=
  Angle.add a b = ⟨90, 0⟩

/-- The main theorem statement -/
theorem complementary_angle_of_37_38 :
  let angle : Angle := ⟨37, 38⟩
  let complement : Angle := ⟨52, 22⟩
  are_complementary angle complement :=
by sorry

end complementary_angle_of_37_38_l987_98753


namespace heart_op_calculation_l987_98783

def heart_op (a b : ℤ) : ℤ := Int.natAbs (a^2 - b^2)

theorem heart_op_calculation : heart_op 3 (heart_op 2 5) = 432 := by
  sorry

end heart_op_calculation_l987_98783


namespace six_digit_number_divisibility_l987_98774

theorem six_digit_number_divisibility (W : ℕ) :
  (100000 ≤ W) ∧ (W < 1000000) ∧
  (∃ a b c : ℕ, a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧
    W = 100000*a + 10000*b + 1000*c + 200*a + 20*b + 2*c) →
  2 ∣ W :=
by sorry

end six_digit_number_divisibility_l987_98774


namespace symmetric_point_x_axis_correct_l987_98712

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Symmetry with respect to x-axis -/
def symmetricPointXAxis (p : Point2D) : Point2D :=
  { x := p.x, y := -p.y }

theorem symmetric_point_x_axis_correct :
  let P : Point2D := { x := 5, y := -3 }
  let symmetricP : Point2D := { x := 5, y := 3 }
  symmetricPointXAxis P = symmetricP := by sorry

end symmetric_point_x_axis_correct_l987_98712


namespace root_sum_ratio_l987_98715

theorem root_sum_ratio (k₁ k₂ : ℝ) : 
  (∃ p q : ℝ, (k₁ * (p^2 - 2*p) + 3*p + 7 = 0 ∧ 
               k₂ * (q^2 - 2*q) + 3*q + 7 = 0) ∧
              (p / q + q / p = 6 / 7)) →
  k₁ / k₂ + k₂ / k₁ = 14 := by
sorry

end root_sum_ratio_l987_98715


namespace min_factors_to_remove_for_2_l987_98782

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def endsIn2 (n : ℕ) : Prop := n % 10 = 2

def factorsToRemove (n : ℕ) : ℕ := 
  let multiples_of_5 := n / 5
  multiples_of_5 + 1

theorem min_factors_to_remove_for_2 : 
  ∃ (removed : Finset ℕ), 
    removed.card = factorsToRemove 99 ∧ 
    endsIn2 ((factorial 99) / (removed.prod id)) ∧
    ∀ (other : Finset ℕ), other.card < factorsToRemove 99 → 
      ¬(endsIn2 ((factorial 99) / (other.prod id))) :=
sorry

end min_factors_to_remove_for_2_l987_98782


namespace exists_n_with_1000_steps_l987_98718

def largest_prime_le (n : ℕ) : ℕ :=
  (Finset.filter Nat.Prime (Finset.range (n + 1))).max' sorry

def reduction_process (n : ℕ) : ℕ → ℕ
| 0 => 0
| (k + 1) => 
  let n' := n - largest_prime_le n
  if n' ≤ 1 then n' else reduction_process n' k

theorem exists_n_with_1000_steps : 
  ∃ N : ℕ, reduction_process N 1000 = 0 ∧ ∀ k < 1000, reduction_process N k ≠ 0 :=
sorry

end exists_n_with_1000_steps_l987_98718


namespace distinct_roots_condition_l987_98786

-- Define the quadratic equation
def quadratic_equation (x k : ℝ) : Prop :=
  x^2 - 2*(k-1)*x + k^2 - 1 = 0

-- Define the discriminant of the quadratic equation
def discriminant (k : ℝ) : ℝ :=
  (-2*(k-1))^2 - 4*(k^2 - 1)

-- Theorem statement
theorem distinct_roots_condition (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ quadratic_equation x k ∧ quadratic_equation y k) ↔ k < 1 := by
  sorry

end distinct_roots_condition_l987_98786


namespace single_point_equation_l987_98789

/-- 
Theorem: If the equation 3x^2 + 4y^2 + 12x - 16y + d = 0 represents a single point, then d = 28.
-/
theorem single_point_equation (d : ℝ) : 
  (∃! p : ℝ × ℝ, 3 * p.1^2 + 4 * p.2^2 + 12 * p.1 - 16 * p.2 + d = 0) → 
  d = 28 := by
  sorry

end single_point_equation_l987_98789


namespace tyler_puppies_l987_98778

/-- The number of puppies Tyler has, given the number of dogs and puppies per dog. -/
def total_puppies (num_dogs : ℕ) (puppies_per_dog : ℕ) : ℕ :=
  num_dogs * puppies_per_dog

/-- Theorem stating that Tyler has 75 puppies given 15 dogs with 5 puppies each. -/
theorem tyler_puppies : total_puppies 15 5 = 75 := by
  sorry

end tyler_puppies_l987_98778


namespace shirley_eggs_left_shirley_eggs_problem_l987_98798

theorem shirley_eggs_left (initial_eggs : ℕ) (bought_eggs : ℕ) 
  (eggs_per_cupcake_batch1 : ℕ) (eggs_per_cupcake_batch2 : ℕ)
  (cupcakes_batch1 : ℕ) (cupcakes_batch2 : ℕ) : ℕ :=
  let total_eggs := initial_eggs + bought_eggs
  let eggs_used_batch1 := eggs_per_cupcake_batch1 * cupcakes_batch1
  let eggs_used_batch2 := eggs_per_cupcake_batch2 * cupcakes_batch2
  let total_eggs_used := eggs_used_batch1 + eggs_used_batch2
  total_eggs - total_eggs_used

theorem shirley_eggs_problem :
  shirley_eggs_left 98 8 5 7 6 4 = 48 := by
  sorry

end shirley_eggs_left_shirley_eggs_problem_l987_98798


namespace problem_paths_l987_98750

/-- Represents the graph of points and their connections -/
structure PointGraph where
  blue_points : Nat
  red_points : Nat
  red_connected_to_blue : Bool
  blue_connected_to_each_other : Bool

/-- Calculates the number of paths between red points -/
def count_paths (g : PointGraph) : Nat :=
  sorry

/-- The specific graph configuration from the problem -/
def problem_graph : PointGraph :=
  { blue_points := 8
  , red_points := 2
  , red_connected_to_blue := true
  , blue_connected_to_each_other := true }

/-- Theorem stating the number of paths in the problem -/
theorem problem_paths :
  count_paths problem_graph = 645120 :=
by sorry

end problem_paths_l987_98750


namespace circle_intersection_theorem_l987_98784

-- Define the circle
def Circle (r : ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = r^2}

-- Define the line
def Line := {p : ℝ × ℝ | p.2 = -p.1 + 2}

-- Define the origin
def O : ℝ × ℝ := (0, 0)

-- Theorem statement
theorem circle_intersection_theorem (r : ℝ) 
  (A B C : ℝ × ℝ) 
  (hA : A ∈ Circle r ∩ Line) 
  (hB : B ∈ Circle r ∩ Line) 
  (hC : C ∈ Circle r) 
  (hOC : C.1 * C.1 + C.2 * C.2 = (5/4 * A.1 + 3/4 * B.1)^2 + (5/4 * A.2 + 3/4 * B.2)^2) :
  r = Real.sqrt 10 := by
  sorry


end circle_intersection_theorem_l987_98784


namespace find_n_l987_98795

theorem find_n : ∃ n : ℕ, (1/5 : ℝ)^n * (1/4 : ℝ)^18 = 1 / (2 * (10 : ℝ)^35) → n = 35 := by
  sorry

end find_n_l987_98795


namespace mary_sheep_count_l987_98758

theorem mary_sheep_count : ∃ (m : ℕ), 
  (∀ (b : ℕ), b = 2 * m + 35 →
    m + 266 = b - 69) → m = 300 := by
  sorry

end mary_sheep_count_l987_98758


namespace exists_right_triangle_different_colors_l987_98765

-- Define the color type
inductive Color
| Blue
| Green
| Red

-- Define the plane as a type
def Plane := ℝ × ℝ

-- Define a coloring function
def coloring : Plane → Color := sorry

-- Define the existence of at least one point of each color
axiom exists_blue : ∃ p : Plane, coloring p = Color.Blue
axiom exists_green : ∃ p : Plane, coloring p = Color.Green
axiom exists_red : ∃ p : Plane, coloring p = Color.Red

-- Define a right triangle
def is_right_triangle (p q r : Plane) : Prop := sorry

-- Theorem statement
theorem exists_right_triangle_different_colors :
  ∃ p q r : Plane, 
    is_right_triangle p q r ∧ 
    coloring p ≠ coloring q ∧ 
    coloring q ≠ coloring r ∧ 
    coloring r ≠ coloring p :=
sorry

end exists_right_triangle_different_colors_l987_98765


namespace cat_head_start_l987_98764

/-- Proves that given a rabbit with speed 25 mph and a cat with speed 20 mph,
    if the rabbit catches up to the cat in 1 hour, then the cat's head start is 15 minutes. -/
theorem cat_head_start (rabbit_speed cat_speed : ℝ) (catch_up_time : ℝ) (head_start : ℝ) :
  rabbit_speed = 25 →
  cat_speed = 20 →
  catch_up_time = 1 →
  rabbit_speed * catch_up_time = cat_speed * (catch_up_time + head_start / 60) →
  head_start = 15 := by
  sorry

#check cat_head_start

end cat_head_start_l987_98764


namespace correct_inequalities_count_proof_correct_inequalities_count_l987_98760

theorem correct_inequalities_count : ℕ :=
  let inequality1 := ∀ a : ℝ, a^2 + 1 ≥ 2*a
  let inequality2 := ∀ x : ℝ, x ≥ 2
  let inequality3 := ∀ x : ℝ, x^2 + x ≥ 1
  2

theorem proof_correct_inequalities_count : 
  (inequality1 → True) ∧ (inequality2 → False) ∧ (inequality3 → True) →
  correct_inequalities_count = 2 :=
by
  sorry

end correct_inequalities_count_proof_correct_inequalities_count_l987_98760


namespace modified_array_sum_for_five_l987_98727

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

/-- The modified 1/p-array sum -/
def modifiedArraySum (p : ℕ) : ℚ :=
  (3 * p^2) / ((9 * p^2 - 12 * p + 4) * (p - 1))

theorem modified_array_sum_for_five :
  modifiedArraySum 5 = 75 / 676 := by sorry

end modified_array_sum_for_five_l987_98727


namespace exponent_multiplication_l987_98709

theorem exponent_multiplication (a : ℝ) : a * a^2 = a^3 := by
  sorry

end exponent_multiplication_l987_98709


namespace g_limit_pos_infinity_g_limit_neg_infinity_l987_98707

/-- The function g(x) = -3x^4 + 5x^3 - 6 -/
def g (x : ℝ) : ℝ := -3 * x^4 + 5 * x^3 - 6

/-- The limit of g(x) approaches negative infinity as x approaches positive infinity -/
theorem g_limit_pos_infinity :
  ∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x > N → g x < M :=
sorry

/-- The limit of g(x) approaches negative infinity as x approaches negative infinity -/
theorem g_limit_neg_infinity :
  ∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x < -N → g x < M :=
sorry

end g_limit_pos_infinity_g_limit_neg_infinity_l987_98707


namespace base6_addition_l987_98790

/-- Converts a base 6 number to base 10 --/
def base6_to_base10 (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

/-- Converts a base 10 number to base 6 --/
def base10_to_base6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc
    else aux (m / 6) ((m % 6) :: acc)
  aux n []

/-- Addition in base 6 --/
def add_base6 (a b : List Nat) : List Nat :=
  base10_to_base6 (base6_to_base10 a + base6_to_base10 b)

theorem base6_addition :
  add_base6 [2, 5, 4, 1] [4, 5, 3, 2] = [0, 5, 2, 4] := by sorry

end base6_addition_l987_98790


namespace ninety_eight_squared_l987_98721

theorem ninety_eight_squared : 98 * 98 = 9604 := by
  -- Proof goes here
  sorry

end ninety_eight_squared_l987_98721


namespace maintenance_check_interval_l987_98787

theorem maintenance_check_interval (original : ℝ) (new : ℝ) : 
  new = 1.5 * original → new = 45 → original = 30 := by
  sorry

end maintenance_check_interval_l987_98787


namespace hyperbola_equation_l987_98704

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 (a > 0, b > 0) and eccentricity √3,
    where the line x = -a²/c (c is the semi-latus rectum) coincides with the latus rectum
    of the parabola y² = 4x, prove that the equation of this hyperbola is x²/3 - y²/6 = 1. -/
theorem hyperbola_equation (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (c / a = Real.sqrt 3) → (a^2 / c = 1) → (x^2 / a^2 - y^2 / b^2 = 1) →
  (x^2 / 3 - y^2 / 6 = 1) :=
by sorry

end hyperbola_equation_l987_98704


namespace semi_circle_area_l987_98749

/-- The area of a semi-circle with diameter 10 meters is 12.5π square meters. -/
theorem semi_circle_area (π : ℝ) : 
  let diameter : ℝ := 10
  let radius : ℝ := diameter / 2
  let semi_circle_area : ℝ := π * radius^2 / 2
  semi_circle_area = 12.5 * π := by
  sorry

end semi_circle_area_l987_98749


namespace problem_solution_l987_98755

theorem problem_solution (a b c : ℝ) (h1 : a = 8 - b) (h2 : c^2 = a*b - 16) : 
  a + c = 4 := by
sorry

end problem_solution_l987_98755


namespace two_visits_count_l987_98748

/-- Represents the visiting schedule of friends -/
structure VisitSchedule where
  alice : Nat
  beatrix : Nat
  claire : Nat

/-- Calculates the number of days when exactly two friends visit -/
def exactlyTwoVisits (schedule : VisitSchedule) (totalDays : Nat) : Nat :=
  sorry

/-- The main theorem to prove -/
theorem two_visits_count (schedule : VisitSchedule) (totalDays : Nat) :
  schedule.alice = 5 →
  schedule.beatrix = 6 →
  schedule.claire = 8 →
  totalDays = 400 →
  exactlyTwoVisits schedule totalDays = 39 :=
sorry

end two_visits_count_l987_98748


namespace circle_M_equation_l987_98767

-- Define the circle M
def circle_M (a r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - a)^2 + p.2^2 = r^2}

-- Define the line l₁: x = -2
def line_l₁ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = -2}

-- Define the line l₂: 2x - √5y - 4 = 0
def line_l₂ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 2 * p.1 - Real.sqrt 5 * p.2 - 4 = 0}

theorem circle_M_equation 
  (a : ℝ) 
  (h1 : a > -2)
  (h2 : ∃ r : ℝ, 
    -- The chord formed by the intersection of M and l₁ has length 2√3
    (3 : ℝ) + (a + 2)^2 = r^2 ∧ 
    -- M is tangent to l₂
    r = |2 * a - 4| / 3) :
  circle_M a 2 = circle_M 1 2 := by sorry

end circle_M_equation_l987_98767


namespace smallest_b_for_factorization_l987_98752

theorem smallest_b_for_factorization : ∃ (b : ℕ), 
  (∀ (r s : ℤ), x^2 + b*x + 2016 = (x + r) * (x + s) → b ≥ 90) ∧
  (∃ (r s : ℤ), x^2 + 90*x + 2016 = (x + r) * (x + s)) :=
by sorry

end smallest_b_for_factorization_l987_98752


namespace janice_starting_sentences_l987_98751

/-- Represents the typing scenario for Janice --/
structure TypingScenario where
  typing_speed : ℕ
  first_session : ℕ
  second_session : ℕ
  third_session : ℕ
  erased_sentences : ℕ
  final_total : ℕ

/-- Calculates the number of sentences Janice started with today --/
def sentences_at_start (scenario : TypingScenario) : ℕ :=
  scenario.final_total - (scenario.typing_speed * (scenario.first_session + scenario.second_session + scenario.third_session) - scenario.erased_sentences)

/-- Theorem stating that Janice started with 258 sentences --/
theorem janice_starting_sentences (scenario : TypingScenario) 
  (h1 : scenario.typing_speed = 6)
  (h2 : scenario.first_session = 20)
  (h3 : scenario.second_session = 15)
  (h4 : scenario.third_session = 18)
  (h5 : scenario.erased_sentences = 40)
  (h6 : scenario.final_total = 536) :
  sentences_at_start scenario = 258 := by
  sorry

#eval sentences_at_start {
  typing_speed := 6,
  first_session := 20,
  second_session := 15,
  third_session := 18,
  erased_sentences := 40,
  final_total := 536
}

end janice_starting_sentences_l987_98751


namespace correct_ranking_l987_98747

-- Define the colleagues
inductive Colleague
| David
| Emily
| Frank

-- Define the years of service comparison
def has_more_years (a b : Colleague) : Prop := sorry

-- Define the statements
def statement_I : Prop := has_more_years Colleague.Emily Colleague.David ∧ has_more_years Colleague.Emily Colleague.Frank
def statement_II : Prop := ¬(has_more_years Colleague.David Colleague.Emily) ∨ ¬(has_more_years Colleague.David Colleague.Frank)
def statement_III : Prop := has_more_years Colleague.Frank Colleague.David ∨ has_more_years Colleague.Frank Colleague.Emily

-- Theorem to prove
theorem correct_ranking :
  (statement_I ∨ statement_II ∨ statement_III) ∧
  ¬(statement_I ∧ statement_II) ∧
  ¬(statement_I ∧ statement_III) ∧
  ¬(statement_II ∧ statement_III) →
  has_more_years Colleague.David Colleague.Frank ∧
  has_more_years Colleague.Frank Colleague.Emily :=
by sorry

end correct_ranking_l987_98747


namespace sum_of_eight_numbers_l987_98703

/-- Given a list of 8 real numbers with an average of 5.7, prove that their sum is 45.6 -/
theorem sum_of_eight_numbers (numbers : List ℝ) 
  (h1 : numbers.length = 8)
  (h2 : numbers.sum / numbers.length = 5.7) : 
  numbers.sum = 45.6 := by
  sorry

end sum_of_eight_numbers_l987_98703


namespace cosine_sine_identity_l987_98714

theorem cosine_sine_identity (α : Real) :
  (∃ (x y : Real), x = 2 ∧ y = 1 ∧ x = Real.cos α * Real.sqrt (x^2 + y^2) ∧ y = Real.sin α * Real.sqrt (x^2 + y^2)) →
  Real.cos α ^ 2 + Real.sin (2 * α) = 8 / 5 := by
  sorry

end cosine_sine_identity_l987_98714


namespace largest_prime_divisor_l987_98737

def base_seven_to_decimal (n : ℕ) : ℕ := 
  2 * 7^6 + 1 * 7^5 + 0 * 7^4 + 2 * 7^3 + 0 * 7^2 + 1 * 7^1 + 2 * 7^0

def number : ℕ := base_seven_to_decimal 2102012

theorem largest_prime_divisor :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ number ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ number → q ≤ p :=
by
  sorry

end largest_prime_divisor_l987_98737


namespace ratio_of_fourth_power_equality_l987_98754

theorem ratio_of_fourth_power_equality (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : (a + b * Complex.I) ^ 4 = (a - b * Complex.I) ^ 4) :
  b / a = 1 := by sorry

end ratio_of_fourth_power_equality_l987_98754


namespace barbara_candies_l987_98791

/-- The number of candies Barbara has left after using some -/
def candies_left (initial : ℝ) (used : ℝ) : ℝ :=
  initial - used

theorem barbara_candies : 
  let initial_candies : ℝ := 18.0
  let used_candies : ℝ := 9.0
  candies_left initial_candies used_candies = 9.0 := by
sorry

end barbara_candies_l987_98791


namespace expression_equals_one_l987_98776

theorem expression_equals_one (x : ℝ) (h1 : x^3 ≠ 1) (h2 : x^3 ≠ -1) :
  ((x^2 + 2*x + 2)^2 * (x^4 - x^2 + 1)^2) / (x^3 + 1)^3 *
  ((x^2 - 2*x + 2)^2 * (x^4 + x^2 + 1)^2) / (x^3 - 1)^3 = 1 := by
  sorry

end expression_equals_one_l987_98776


namespace rectangle_triangle_max_area_and_hypotenuse_l987_98761

theorem rectangle_triangle_max_area_and_hypotenuse (x y : ℝ) :
  x > 0 → y > 0 →  -- rectangle has positive dimensions
  x + y = 30 →     -- half the perimeter is 30
  (∃ h : ℝ, h^2 = x^2 + y^2) →  -- it's a right triangle
  x * y ≤ 225 ∧    -- max area is 225
  (x * y = 225 → ∃ h : ℝ, h^2 = x^2 + y^2 ∧ h = 15 * Real.sqrt 2) :=
by sorry

end rectangle_triangle_max_area_and_hypotenuse_l987_98761


namespace oak_grove_library_books_l987_98711

theorem oak_grove_library_books (total_books : ℕ) (public_library_books : ℕ) 
  (h1 : total_books = 7092) (h2 : public_library_books = 1986) :
  total_books - public_library_books = 5106 := by
  sorry

end oak_grove_library_books_l987_98711


namespace angle_greater_iff_sin_greater_l987_98700

-- Define a triangle with angles A, B, and C
structure Triangle where
  A : Real
  B : Real
  C : Real
  angle_sum : A + B + C = π
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C

-- State the theorem
theorem angle_greater_iff_sin_greater (t : Triangle) : t.A > t.B ↔ Real.sin t.A > Real.sin t.B := by
  sorry

end angle_greater_iff_sin_greater_l987_98700


namespace impossible_coin_probabilities_l987_98745

theorem impossible_coin_probabilities :
  ¬ ∃ (p₁ p₂ : ℝ), 0 ≤ p₁ ∧ p₁ ≤ 1 ∧ 0 ≤ p₂ ∧ p₂ ≤ 1 ∧
    (1 - p₁) * (1 - p₂) = p₁ * p₂ ∧
    p₁ * p₂ = p₁ * (1 - p₂) + p₂ * (1 - p₁) :=
by sorry

end impossible_coin_probabilities_l987_98745


namespace complex_fraction_sum_simplification_l987_98757

theorem complex_fraction_sum_simplification :
  let i : ℂ := Complex.I
  ((4 + 7 * i) / (4 - 7 * i) + (4 - 7 * i) / (4 + 7 * i)) = (-66 : ℚ) / 65 := by
  sorry

end complex_fraction_sum_simplification_l987_98757


namespace min_value_of_a_l987_98793

theorem min_value_of_a (p : ∃ x₀ : ℝ, |x₀ + 1| + |x₀ - 2| ≤ a) : 
  ∀ b : ℝ, b < 3 → ¬(∃ x₀ : ℝ, |x₀ + 1| + |x₀ - 2| ≤ b) :=
by sorry

end min_value_of_a_l987_98793


namespace find_m_find_k_l987_98785

-- Define the vectors
def a : ℝ × ℝ := (1, -3)
def b (m : ℝ) : ℝ × ℝ := (-2, m)

-- Define dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define vector subtraction
def vec_sub (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 - w.1, v.2 - w.2)

-- Define vector addition
def vec_add (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)

-- Define scalar multiplication
def scalar_mul (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)

-- Define parallel vectors
def parallel (v w : ℝ × ℝ) : Prop := ∃ (k : ℝ), v = scalar_mul k w

-- Theorem 1: Find the value of m
theorem find_m :
  ∃ (m : ℝ), dot_product a (vec_sub a (b m)) = 0 ∧ m = -4 :=
sorry

-- Theorem 2: Find the value of k
theorem find_k :
  ∃ (k : ℝ), parallel (vec_add (scalar_mul k a) (b (-4))) (vec_sub a (b (-4))) ∧ k = -1 :=
sorry

end find_m_find_k_l987_98785


namespace complement_of_A_in_U_l987_98781

-- Define the universal set U
def U : Set ℝ := {x | x > 0}

-- Define set A
def A : Set ℝ := {x | 1 < x ∧ x < 2}

-- Theorem statement
theorem complement_of_A_in_U :
  (U \ A) = {x | 0 < x ∧ x ≤ 1} ∪ {x | x ≥ 2} := by
  sorry

end complement_of_A_in_U_l987_98781


namespace good_carrots_count_l987_98796

theorem good_carrots_count (nancy_carrots : ℕ) (mom_carrots : ℕ) (bad_carrots : ℕ) :
  nancy_carrots = 38 →
  mom_carrots = 47 →
  bad_carrots = 14 →
  nancy_carrots + mom_carrots - bad_carrots = 71 := by
sorry

end good_carrots_count_l987_98796


namespace exists_square_with_1983_nines_l987_98738

theorem exists_square_with_1983_nines : ∃ n : ℕ, ∃ m : ℕ, n^2 = 10^3968 - 10^1985 + m ∧ m < 10^1985 := by
  sorry

end exists_square_with_1983_nines_l987_98738


namespace largest_three_digit_congruent_to_12_mod_15_l987_98756

theorem largest_three_digit_congruent_to_12_mod_15 : ∃ n : ℕ,
  n = 987 ∧
  n ≥ 100 ∧ n < 1000 ∧
  n % 15 = 12 ∧
  ∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ m % 15 = 12 → m ≤ n :=
by sorry

end largest_three_digit_congruent_to_12_mod_15_l987_98756


namespace parabola_vertex_x_coordinate_l987_98773

/-- Proves that for a quadratic function y = ax^2 + bx + c passing through points 
(-2,8), (4,8), and (7,15), the x-coordinate of the vertex is 1. -/
theorem parabola_vertex_x_coordinate (a b c : ℝ) : 
  (8 = a * (-2)^2 + b * (-2) + c) →
  (8 = a * 4^2 + b * 4 + c) →
  (15 = a * 7^2 + b * 7 + c) →
  (∃ (x : ℝ), x = 1 ∧ ∀ (t : ℝ), a * t^2 + b * t + c ≥ a * x^2 + b * x + c) :=
by sorry

end parabola_vertex_x_coordinate_l987_98773


namespace logans_average_speed_l987_98794

/-- Prove Logan's average speed given the driving conditions of Tamika and Logan -/
theorem logans_average_speed 
  (tamika_time : ℝ) 
  (tamika_speed : ℝ) 
  (logan_time : ℝ) 
  (distance_difference : ℝ) 
  (h1 : tamika_time = 8) 
  (h2 : tamika_speed = 45) 
  (h3 : logan_time = 5) 
  (h4 : tamika_time * tamika_speed = logan_time * logan_speed + distance_difference) 
  (h5 : distance_difference = 85) : 
  logan_speed = 55 := by
  sorry

end logans_average_speed_l987_98794


namespace external_tangent_distance_l987_98724

/-- Given two externally touching circles with radii R and r, 
    the distance AB between the points where their common external tangent 
    touches the circles is equal to 2√(Rr) -/
theorem external_tangent_distance (R r : ℝ) (hR : R > 0) (hr : r > 0) :
  ∃ (AB : ℝ), AB = 2 * Real.sqrt (R * r) := by
  sorry

end external_tangent_distance_l987_98724


namespace april_earnings_l987_98722

/-- The price of a rose in dollars -/
def rose_price : ℕ := 7

/-- The price of a lily in dollars -/
def lily_price : ℕ := 5

/-- The initial number of roses -/
def initial_roses : ℕ := 9

/-- The initial number of lilies -/
def initial_lilies : ℕ := 6

/-- The remaining number of roses -/
def remaining_roses : ℕ := 4

/-- The remaining number of lilies -/
def remaining_lilies : ℕ := 2

/-- The total earnings from the sale -/
def total_earnings : ℕ := 55

theorem april_earnings : 
  (initial_roses - remaining_roses) * rose_price + 
  (initial_lilies - remaining_lilies) * lily_price = total_earnings := by
  sorry

end april_earnings_l987_98722


namespace total_wood_planks_l987_98719

def initial_planks : ℕ := 15
def charlie_planks : ℕ := 10
def father_planks : ℕ := 10

theorem total_wood_planks : 
  initial_planks + charlie_planks + father_planks = 35 := by
  sorry

end total_wood_planks_l987_98719
