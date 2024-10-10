import Mathlib

namespace sqrt_1_minus_x_real_l1535_153578

theorem sqrt_1_minus_x_real (x : ℝ) : (∃ y : ℝ, y ^ 2 = 1 - x) ↔ x ≤ 1 := by sorry

end sqrt_1_minus_x_real_l1535_153578


namespace plane_seats_count_l1535_153539

theorem plane_seats_count : 
  ∀ (total_seats : ℕ),
  (30 : ℕ) + (total_seats / 5 : ℕ) + (total_seats - (30 + total_seats / 5) : ℕ) = total_seats →
  total_seats = 50 := by
sorry

end plane_seats_count_l1535_153539


namespace emily_toys_sale_l1535_153538

def sell_toys (initial : ℕ) (percent1 : ℕ) (percent2 : ℕ) : ℕ :=
  let remaining_after_day1 := initial - (initial * percent1 / 100)
  let remaining_after_day2 := remaining_after_day1 - (remaining_after_day1 * percent2 / 100)
  remaining_after_day2

theorem emily_toys_sale :
  sell_toys 35 50 60 = 8 := by
  sorry

end emily_toys_sale_l1535_153538


namespace perimeter_T_shape_specific_l1535_153523

/-- Calculates the perimeter of a T shape formed by two rectangles with given dimensions and overlap. -/
def perimeter_T_shape (rect1_length rect1_width rect2_length rect2_width overlap : ℝ) : ℝ :=
  2 * (rect1_length + rect1_width) + 2 * (rect2_length + rect2_width) - 2 * overlap

/-- The perimeter of a T shape formed by two rectangles (3 inch × 5 inch and 2 inch × 6 inch) with a 1-inch overlap is 30 inches. -/
theorem perimeter_T_shape_specific : perimeter_T_shape 3 5 2 6 1 = 30 := by
  sorry

#eval perimeter_T_shape 3 5 2 6 1

end perimeter_T_shape_specific_l1535_153523


namespace max_successful_teams_16_l1535_153565

/-- Represents a football championship --/
structure Championship :=
  (teams : ℕ)
  (points_for_win : ℕ)
  (points_for_draw : ℕ)
  (points_for_loss : ℕ)

/-- Definition of a successful team --/
def is_successful (c : Championship) (points : ℕ) : Prop :=
  points ≥ (c.teams - 1) * c.points_for_win / 2

/-- The maximum number of successful teams in the championship --/
def max_successful_teams (c : Championship) : ℕ := sorry

/-- The main theorem --/
theorem max_successful_teams_16 :
  ∀ c : Championship,
    c.teams = 16 ∧
    c.points_for_win = 3 ∧
    c.points_for_draw = 1 ∧
    c.points_for_loss = 0 →
    max_successful_teams c = 15 := by sorry

end max_successful_teams_16_l1535_153565


namespace modular_inverse_of_3_mod_17_l1535_153584

theorem modular_inverse_of_3_mod_17 :
  ∃ (x : ℤ), 0 ≤ x ∧ x ≤ 16 ∧ (3 * x) % 17 = 1 :=
by
  use 6
  sorry

end modular_inverse_of_3_mod_17_l1535_153584


namespace no_convex_function_exists_l1535_153508

theorem no_convex_function_exists : 
  ¬∃ f : ℝ → ℝ, ∀ x y : ℝ, (f x + f y) / 2 ≥ f ((x + y) / 2) + |x - y| :=
by sorry

end no_convex_function_exists_l1535_153508


namespace animal_shelter_problem_l1535_153505

theorem animal_shelter_problem (initial_cats initial_lizards : ℕ)
  (dog_adoption_rate cat_adoption_rate lizard_adoption_rate : ℚ)
  (new_pets total_pets_after_month : ℕ) :
  initial_cats = 28 →
  initial_lizards = 20 →
  dog_adoption_rate = 1/2 →
  cat_adoption_rate = 1/4 →
  lizard_adoption_rate = 1/5 →
  new_pets = 13 →
  total_pets_after_month = 65 →
  ∃ (initial_dogs : ℕ),
    initial_dogs = 30 ∧
    (1 - dog_adoption_rate) * initial_dogs +
    (1 - cat_adoption_rate) * initial_cats +
    (1 - lizard_adoption_rate) * initial_lizards +
    new_pets = total_pets_after_month :=
by sorry

end animal_shelter_problem_l1535_153505


namespace max_almonds_in_mixture_l1535_153571

/-- Represents the composition of the nut mixture -/
structure NutMixture where
  almonds : ℕ
  walnuts : ℕ
  cashews : ℕ
  pistachios : ℕ

/-- Represents the cost per pound of each nut type -/
structure NutCosts where
  almonds : ℚ
  walnuts : ℚ
  cashews : ℚ
  pistachios : ℚ

/-- Calculates the total cost of a given nut mixture -/
def totalCost (mixture : NutMixture) (costs : NutCosts) : ℚ :=
  mixture.almonds * costs.almonds +
  mixture.walnuts * costs.walnuts +
  mixture.cashews * costs.cashews +
  mixture.pistachios * costs.pistachios

/-- Theorem stating the maximum possible pounds of almonds in the mixture -/
theorem max_almonds_in_mixture
  (mixture : NutMixture)
  (costs : NutCosts)
  (budget : ℚ)
  (total_weight : ℕ)
  (h_composition : mixture.almonds = 5 ∧ mixture.walnuts = 2 ∧ mixture.cashews = 3 ∧ mixture.pistachios = 4)
  (h_costs : costs.almonds = 6 ∧ costs.walnuts = 5 ∧ costs.cashews = 8 ∧ costs.pistachios = 10)
  (h_budget : budget = 1500)
  (h_total_weight : total_weight = 800)
  (h_almond_percentage : (mixture.almonds : ℚ) / (mixture.almonds + mixture.walnuts + mixture.cashews + mixture.pistachios) ≥ 30 / 100) :
  (mixture.almonds : ℚ) / (mixture.almonds + mixture.walnuts + mixture.cashews + mixture.pistachios) * total_weight ≤ 240 ∧
  totalCost mixture costs ≤ budget :=
sorry

end max_almonds_in_mixture_l1535_153571


namespace original_worker_count_l1535_153555

/-- Given a work that can be completed by some workers in 65 days,
    and adding 10 workers reduces the time to 55 days,
    prove that the original number of workers is 55. -/
theorem original_worker_count (work : ℕ) : ∃ (workers : ℕ), 
  (workers * 65 = (workers + 10) * 55) ∧ 
  (workers = 55) := by
  sorry

end original_worker_count_l1535_153555


namespace autumn_pencils_l1535_153548

def pencil_count (initial : ℕ) (lost : ℕ) (broken : ℕ) (found : ℕ) (bought : ℕ) : ℕ :=
  initial - lost - broken + found + bought

theorem autumn_pencils :
  pencil_count 20 7 3 4 2 = 16 := by
  sorry

end autumn_pencils_l1535_153548


namespace two_numbers_sum_l1535_153501

theorem two_numbers_sum : ∃ (x y : ℝ), x * 15 = x + 196 ∧ y * 50 = y + 842 ∧ x + y = 31.2 := by
  sorry

end two_numbers_sum_l1535_153501


namespace snake_length_difference_l1535_153581

/-- Given two snakes with a combined length of 70 inches, where one snake is 41 inches long,
    prove that the difference in length between the longer and shorter snake is 12 inches. -/
theorem snake_length_difference (combined_length jake_length : ℕ)
  (h1 : combined_length = 70)
  (h2 : jake_length = 41)
  (h3 : jake_length > combined_length - jake_length) :
  jake_length - (combined_length - jake_length) = 12 := by
  sorry

end snake_length_difference_l1535_153581


namespace min_time_for_given_problem_l1535_153547

/-- Represents the chef's cooking problem -/
structure ChefProblem where
  total_potatoes : ℕ
  cooked_potatoes : ℕ
  cooking_time_per_potato : ℕ
  salad_prep_time : ℕ

/-- Calculates the minimum time needed to complete the cooking task -/
def min_time_needed (problem : ChefProblem) : ℕ :=
  max problem.salad_prep_time (problem.cooking_time_per_potato)

/-- Theorem stating the minimum time needed for the given problem -/
theorem min_time_for_given_problem :
  let problem : ChefProblem := {
    total_potatoes := 35,
    cooked_potatoes := 11,
    cooking_time_per_potato := 7,
    salad_prep_time := 15
  }
  min_time_needed problem = 15 := by sorry

end min_time_for_given_problem_l1535_153547


namespace opposite_of_negative_three_l1535_153568

theorem opposite_of_negative_three :
  -(- 3) = 3 := by sorry

end opposite_of_negative_three_l1535_153568


namespace sequence_a_500th_term_l1535_153595

def sequence_a (a : ℕ → ℕ) : Prop :=
  a 1 = 1007 ∧ 
  a 2 = 1008 ∧ 
  ∀ n : ℕ, n ≥ 1 → a n + a (n + 1) + a (n + 2) = n

theorem sequence_a_500th_term (a : ℕ → ℕ) (h : sequence_a a) : a 500 = 1173 := by
  sorry

end sequence_a_500th_term_l1535_153595


namespace tangent_segments_area_l1535_153513

/-- The area of the region formed by all line segments of length 2 units 
    tangent to a circle with radius 3 units is equal to 4π. -/
theorem tangent_segments_area (r : ℝ) (l : ℝ) (h1 : r = 3) (h2 : l = 2) : 
  let outer_radius := Real.sqrt (r^2 + l^2)
  π * outer_radius^2 - π * r^2 = 4 * π := by
sorry

end tangent_segments_area_l1535_153513


namespace quadratic_inequality_condition_l1535_153550

theorem quadratic_inequality_condition (a : ℝ) :
  (∀ x : ℝ, (a^2 - 1) * x^2 + (a + 1) * x + 1/2 > 0) ↔ 
  (a ≤ -1 ∨ a > 3) :=
sorry

end quadratic_inequality_condition_l1535_153550


namespace square_root_of_16_l1535_153504

theorem square_root_of_16 : {x : ℝ | x^2 = 16} = {4, -4} := by sorry

end square_root_of_16_l1535_153504


namespace kitty_cleaning_weeks_l1535_153549

/-- The time Kitty spends on each cleaning task and the total time spent -/
structure CleaningTime where
  pickup : ℕ       -- Time spent picking up toys and straightening
  vacuum : ℕ       -- Time spent vacuuming
  windows : ℕ      -- Time spent cleaning windows
  dusting : ℕ      -- Time spent dusting furniture
  total : ℕ        -- Total time spent cleaning

/-- Calculate the number of weeks Kitty has been cleaning -/
def weeks_cleaning (ct : CleaningTime) : ℕ :=
  ct.total / (ct.pickup + ct.vacuum + ct.windows + ct.dusting)

/-- Theorem stating that Kitty has been cleaning for 4 weeks -/
theorem kitty_cleaning_weeks :
  let ct : CleaningTime := {
    pickup := 5,
    vacuum := 20,
    windows := 15,
    dusting := 10,
    total := 200
  }
  weeks_cleaning ct = 4 := by sorry

end kitty_cleaning_weeks_l1535_153549


namespace june_design_white_tiles_l1535_153594

/-- The number of white tiles in June's design -/
def white_tiles (total : ℕ) (yellow : ℕ) (purple : ℕ) : ℕ :=
  total - (yellow + (yellow + 1) + purple)

/-- Theorem stating the number of white tiles in June's design -/
theorem june_design_white_tiles :
  white_tiles 20 3 6 = 7 := by
  sorry

end june_design_white_tiles_l1535_153594


namespace vacation_tents_l1535_153506

/-- Calculates the number of tents needed given the total number of people,
    the number of people the house can sleep, and the number of people per tent. -/
def tents_needed (total_people : ℕ) (house_capacity : ℕ) (people_per_tent : ℕ) : ℕ :=
  ((total_people - house_capacity) + (people_per_tent - 1)) / people_per_tent

theorem vacation_tents :
  tents_needed 14 4 2 = 5 := by
  sorry

end vacation_tents_l1535_153506


namespace ratio_expression_value_l1535_153585

theorem ratio_expression_value (P Q R : ℚ) (h : P / Q = 3 / 2 ∧ Q / R = 2 / 6) :
  (4 * P + 3 * Q) / (5 * R - 2 * P) = 5 / 8 := by
  sorry

end ratio_expression_value_l1535_153585


namespace league_games_count_l1535_153511

/-- The number of teams in the league -/
def num_teams : ℕ := 25

/-- The total number of games played in the league -/
def total_games : ℕ := num_teams * (num_teams - 1) / 2

/-- Theorem stating that the total number of games in the league is 300 -/
theorem league_games_count : total_games = 300 := by
  sorry

end league_games_count_l1535_153511


namespace rain_duration_l1535_153573

/-- Calculates the number of minutes it rained to fill a tank given initial conditions -/
theorem rain_duration (initial_water : ℕ) (evaporated : ℕ) (drained : ℕ) (rain_rate : ℕ) (final_water : ℕ) : 
  initial_water = 6000 →
  evaporated = 2000 →
  drained = 3500 →
  rain_rate = 350 →
  final_water = 1550 →
  (final_water - (initial_water - evaporated - drained)) / (rain_rate / 10) * 10 = 30 := by
  sorry

end rain_duration_l1535_153573


namespace river_depth_ratio_l1535_153587

/-- The ratio of river depths in July to June -/
theorem river_depth_ratio :
  let may_depth : ℝ := 5
  let june_depth : ℝ := may_depth + 10
  let july_depth : ℝ := 45
  july_depth / june_depth = 3 := by sorry

end river_depth_ratio_l1535_153587


namespace prob_same_color_specific_l1535_153559

/-- The probability of drawing 4 marbles of the same color from an urn -/
def prob_same_color (red white blue : ℕ) : ℚ :=
  let total := red + white + blue
  let prob_red := (red.descFactorial 4 : ℚ) / (total.descFactorial 4 : ℚ)
  let prob_white := (white.descFactorial 4 : ℚ) / (total.descFactorial 4 : ℚ)
  let prob_blue := (blue.descFactorial 4 : ℚ) / (total.descFactorial 4 : ℚ)
  prob_red + prob_white + prob_blue

/-- Theorem: The probability of drawing 4 marbles of the same color from an urn
    containing 5 red, 6 white, and 7 blue marbles is 55/3060 -/
theorem prob_same_color_specific : prob_same_color 5 6 7 = 55 / 3060 := by
  sorry


end prob_same_color_specific_l1535_153559


namespace three_valid_k_values_l1535_153545

/-- The sum of k consecutive natural numbers starting from n -/
def consecutiveSum (n k : ℕ) : ℕ := k * (2 * n + k - 1) / 2

/-- Predicate to check if k is a valid solution -/
def isValidK (k : ℕ) : Prop :=
  k > 1 ∧ ∃ n : ℕ, consecutiveSum n k = 2000

theorem three_valid_k_values :
  ∃! (s : Finset ℕ), s.card = 3 ∧ ∀ k, k ∈ s ↔ isValidK k :=
sorry

end three_valid_k_values_l1535_153545


namespace function_composition_theorem_l1535_153577

theorem function_composition_theorem (a b : ℝ) :
  (∀ x : ℝ, (3 * ((a * x + b) : ℝ) - 6 : ℝ) = 4 * x + 3) →
  a + b = 13 / 3 := by
sorry

end function_composition_theorem_l1535_153577


namespace total_value_calculation_l1535_153557

-- Define coin quantities
def us_quarters : ℕ := 25
def us_dimes : ℕ := 15
def us_nickels : ℕ := 12
def us_half_dollars : ℕ := 7
def us_dollar_coins : ℕ := 3
def us_pennies : ℕ := 375
def canadian_quarters : ℕ := 10
def canadian_dimes : ℕ := 5
def canadian_nickels : ℕ := 4

-- Define coin values in their respective currencies
def us_quarter_value : ℚ := 0.25
def us_dime_value : ℚ := 0.10
def us_nickel_value : ℚ := 0.05
def us_half_dollar_value : ℚ := 0.50
def us_dollar_coin_value : ℚ := 1.00
def us_penny_value : ℚ := 0.01
def canadian_quarter_value : ℚ := 0.25
def canadian_dime_value : ℚ := 0.10
def canadian_nickel_value : ℚ := 0.05

-- Define exchange rate
def cad_to_usd_rate : ℚ := 0.80

-- Theorem to prove
theorem total_value_calculation :
  (us_quarters * us_quarter_value +
   us_dimes * us_dime_value +
   us_nickels * us_nickel_value +
   us_half_dollars * us_half_dollar_value +
   us_dollar_coins * us_dollar_coin_value +
   us_pennies * us_penny_value) +
  ((canadian_quarters * canadian_quarter_value +
    canadian_dimes * canadian_dime_value +
    canadian_nickels * canadian_nickel_value) * cad_to_usd_rate) = 21.16 := by
  sorry

end total_value_calculation_l1535_153557


namespace min_value_a_plus_2b_l1535_153599

theorem min_value_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1 / (2 * a) + 1 / b = 1) :
  ∀ x y, x > 0 → y > 0 → 1 / (2 * x) + 1 / y = 1 → a + 2 * b ≤ x + 2 * y :=
sorry

end min_value_a_plus_2b_l1535_153599


namespace circle_trajectory_l1535_153596

-- Define the circles F₁ and F₂
def F₁ (x y : ℝ) : Prop := (x + 3)^2 + y^2 = 81
def F₂ (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 1

-- Define the center and radius of F₁
def F₁_center : ℝ × ℝ := (-3, 0)
def F₁_radius : ℝ := 9

-- Define the center and radius of F₂
def F₂_center : ℝ × ℝ := (3, 0)
def F₂_radius : ℝ := 1

-- Define the trajectory of the center of circle P
def trajectory (x y : ℝ) : Prop := x^2 / 16 + y^2 / 7 = 1

-- Theorem statement
theorem circle_trajectory :
  ∀ (x y r : ℝ),
  (∃ (x₁ y₁ : ℝ), F₁ x₁ y₁ ∧ (x - x₁)^2 + (y - y₁)^2 = r^2) →
  (∃ (x₂ y₂ : ℝ), F₂ x₂ y₂ ∧ (x - x₂)^2 + (y - y₂)^2 = r^2) →
  trajectory x y :=
by sorry

end circle_trajectory_l1535_153596


namespace discount_and_total_amount_l1535_153560

/-- Given two positive real numbers P and Q where P > Q, 
    this theorem proves the correct calculation of the percentage discount
    and the total amount paid for 10 items. -/
theorem discount_and_total_amount (P Q : ℝ) (h1 : P > Q) (h2 : Q > 0) :
  let d := 100 * (P - Q) / P
  let total := 10 * Q
  (d = 100 * (P - Q) / P) ∧ (total = 10 * Q) := by
  sorry

#check discount_and_total_amount

end discount_and_total_amount_l1535_153560


namespace min_value_expression_l1535_153582

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  4 * a^2 + 4 * b^2 + 1 / (a + b)^2 ≥ 2 ∧
  (4 * a^2 + 4 * b^2 + 1 / (a + b)^2 = 2 ↔ a = 1/2 ∧ b = 1/2) := by
  sorry

end min_value_expression_l1535_153582


namespace equal_sides_implies_rhombus_l1535_153546

/-- A quadrilateral is a polygon with four sides -/
structure Quadrilateral where
  sides : Fin 4 → ℝ
  positive : ∀ i, sides i > 0

/-- A rhombus is a quadrilateral with all sides of equal length -/
def is_rhombus (q : Quadrilateral) : Prop :=
  ∀ i j, q.sides i = q.sides j

/-- Theorem: A quadrilateral with all sides of equal length is a rhombus -/
theorem equal_sides_implies_rhombus (q : Quadrilateral) 
  (h : ∀ i j, q.sides i = q.sides j) : is_rhombus q := by
  sorry

end equal_sides_implies_rhombus_l1535_153546


namespace meal_price_before_coupon_l1535_153592

theorem meal_price_before_coupon
  (num_people : ℕ)
  (individual_contribution : ℝ)
  (coupon_value : ℝ)
  (h1 : num_people = 3)
  (h2 : individual_contribution = 21)
  (h3 : coupon_value = 4)
  : ↑num_people * individual_contribution + coupon_value = 67 :=
by sorry

end meal_price_before_coupon_l1535_153592


namespace smallest_positive_multiple_of_18_times_5_smallest_positive_multiple_is_90_l1535_153503

theorem smallest_positive_multiple_of_18_times_5 :
  ∀ n : ℕ+, n * (18 * 5) ≥ 18 * 5 :=
by
  sorry

theorem smallest_positive_multiple_is_90 :
  ∃ (n : ℕ+), n * (18 * 5) = 90 ∧ ∀ (m : ℕ+), m * (18 * 5) ≥ 90 :=
by
  sorry

end smallest_positive_multiple_of_18_times_5_smallest_positive_multiple_is_90_l1535_153503


namespace cubes_volume_percentage_l1535_153519

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents a cube with a given side length -/
structure Cube where
  sideLength : ℕ

/-- Calculates the volume of a rectangular box -/
def boxVolume (box : BoxDimensions) : ℕ :=
  box.length * box.width * box.height

/-- Calculates the volume of a cube -/
def cubeVolume (cube : Cube) : ℕ :=
  cube.sideLength * cube.sideLength * cube.sideLength

/-- Calculates the number of cubes that can fit along a given dimension -/
def cubesFitInDimension (dimension : ℕ) (cube : Cube) : ℕ :=
  dimension / cube.sideLength

/-- Calculates the total number of cubes that can fit in the box -/
def totalCubesFit (box : BoxDimensions) (cube : Cube) : ℕ :=
  (cubesFitInDimension box.length cube) *
  (cubesFitInDimension box.width cube) *
  (cubesFitInDimension box.height cube)

/-- Theorem: The percentage of volume occupied by 4-inch cubes in a 8x6x12 inch box is 66.67% -/
theorem cubes_volume_percentage :
  let box := BoxDimensions.mk 8 6 12
  let cube := Cube.mk 4
  let cubesVolume := (totalCubesFit box cube) * (cubeVolume cube)
  let totalVolume := boxVolume box
  let percentage := (cubesVolume : ℚ) / (totalVolume : ℚ) * 100
  percentage = 200/3 := by
  sorry

end cubes_volume_percentage_l1535_153519


namespace exponent_product_square_l1535_153522

theorem exponent_product_square (x y : ℝ) : (3 * x * y)^2 = 9 * x^2 * y^2 := by
  sorry

end exponent_product_square_l1535_153522


namespace stream_speed_l1535_153586

/-- Given a canoe that rows upstream at 4 km/hr and downstream at 12 km/hr, 
    the speed of the stream is 4 km/hr. -/
theorem stream_speed (upstream_speed downstream_speed : ℝ) 
  (h1 : upstream_speed = 4)
  (h2 : downstream_speed = 12) : 
  ∃ (canoe_speed stream_speed : ℝ),
    canoe_speed - stream_speed = upstream_speed ∧
    canoe_speed + stream_speed = downstream_speed ∧
    stream_speed = 4 := by
  sorry

end stream_speed_l1535_153586


namespace complex_modulus_l1535_153518

theorem complex_modulus (z : ℂ) (h : (3 + 2*I) * z = 5 - I) : Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_modulus_l1535_153518


namespace equation_solution_existence_l1535_153500

theorem equation_solution_existence (z : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, x^2 + y^2 + 4*z^2 + 2*x*y*z - 9 = 0) ↔ 1 ≤ |z| ∧ |z| ≤ 3/2 :=
by sorry

end equation_solution_existence_l1535_153500


namespace sum_of_numbers_l1535_153564

theorem sum_of_numbers (x y : ℝ) : 
  y = 2 * x - 3 →
  y = 37 →
  x + y = 57 := by
sorry

end sum_of_numbers_l1535_153564


namespace f_has_max_iff_solution_set_when_a_is_one_l1535_153541

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * abs (x - 2) + x

-- Theorem 1: f has a maximum value iff a ≤ -1
theorem f_has_max_iff (a : ℝ) : 
  (∃ (M : ℝ), ∀ (x : ℝ), f a x ≤ M) ↔ a ≤ -1 :=
sorry

-- Theorem 2: Solution set of f(x) < |2x - 3| when a = 1
theorem solution_set_when_a_is_one : 
  {x : ℝ | f 1 x < abs (2 * x - 3)} = {x : ℝ | x > 1/2} :=
sorry

end f_has_max_iff_solution_set_when_a_is_one_l1535_153541


namespace book_arrangement_proof_l1535_153567

def number_of_arrangements (n : ℕ) (k : ℕ) : ℕ := n.factorial / k.factorial

theorem book_arrangement_proof :
  let total_books : ℕ := 7
  let identical_books : ℕ := 3
  let distinct_books : ℕ := 4
  let books_to_arrange : ℕ := total_books - 1
  number_of_arrangements books_to_arrange identical_books = 120 := by
  sorry

end book_arrangement_proof_l1535_153567


namespace max_cards_proof_l1535_153533

/-- The maximum number of trading cards Jasmine can buy --/
def max_cards : ℕ := 8

/-- Jasmine's initial budget --/
def initial_budget : ℚ := 15

/-- Cost per card --/
def card_cost : ℚ := 1.25

/-- Fixed transaction fee --/
def transaction_fee : ℚ := 2

/-- Minimum amount Jasmine wants to keep --/
def min_remaining : ℚ := 3

theorem max_cards_proof :
  (card_cost * max_cards + transaction_fee ≤ initial_budget - min_remaining) ∧
  (∀ n : ℕ, n > max_cards → card_cost * n + transaction_fee > initial_budget - min_remaining) :=
by sorry

end max_cards_proof_l1535_153533


namespace line_intersection_l1535_153542

/-- Given a line with slope 3/4 passing through (400, 0), 
    prove that it intersects y = -39 at x = 348 -/
theorem line_intersection (m : ℚ) (x₀ y₀ x₁ y₁ : ℚ) : 
  m = 3/4 → x₀ = 400 → y₀ = 0 → y₁ = -39 →
  (y₁ - y₀) = m * (x₁ - x₀) →
  x₁ = 348 := by
sorry

end line_intersection_l1535_153542


namespace vertical_angles_equal_l1535_153532

-- Define a type for angles
def Angle : Type := ℝ

-- Define a function to create vertical angles
def verticalAngles (α β : Angle) : Prop := α = β

-- Theorem: Vertical angles are equal
theorem vertical_angles_equal (α β : Angle) (h : verticalAngles α β) : α = β := by
  sorry

-- Note: We don't define or assume anything about other angle relationships

end vertical_angles_equal_l1535_153532


namespace charging_station_profit_l1535_153597

/-- Represents the total profit function for electric car charging stations -/
def profit_function (a b c : ℝ) (x : ℕ+) : ℝ := a * (x : ℝ)^2 + b * (x : ℝ) + c

theorem charging_station_profit 
  (a b c : ℝ) 
  (h1 : profit_function a b c 3 = 2) 
  (h2 : profit_function a b c 6 = 11) 
  (h3 : ∀ x : ℕ+, profit_function a b c x ≤ 11) :
  (∀ x : ℕ+, profit_function a b c x = -10 * (x : ℝ)^2 + 120 * (x : ℝ) - 250) ∧ 
  (∀ x : ℕ+, (profit_function a b c x) / (x : ℝ) ≤ 2) :=
sorry

end charging_station_profit_l1535_153597


namespace fifth_number_in_specific_pascal_row_l1535_153521

/-- Represents a row in Pascal's triangle -/
def PascalRow (n : ℕ) := Fin (n + 1) → ℕ

/-- The nth row of Pascal's triangle -/
def nthPascalRow (n : ℕ) : PascalRow n := 
  fun k => Nat.choose n k.val

/-- The condition that a row starts with 1 and then 15 -/
def startsWithOneAndFifteen (row : PascalRow 15) : Prop :=
  row 0 = 1 ∧ row 1 = 15

theorem fifth_number_in_specific_pascal_row : 
  ∀ (row : PascalRow 15), 
    startsWithOneAndFifteen row → 
    row 4 = Nat.choose 15 4 ∧ 
    Nat.choose 15 4 = 1365 := by
  sorry

end fifth_number_in_specific_pascal_row_l1535_153521


namespace tabs_remaining_l1535_153552

theorem tabs_remaining (initial_tabs : ℕ) : 
  initial_tabs = 400 → 
  (initial_tabs - (initial_tabs / 4) - 
   ((initial_tabs - (initial_tabs / 4)) * 2 / 5) -
   ((initial_tabs - (initial_tabs / 4) - 
     ((initial_tabs - (initial_tabs / 4)) * 2 / 5)) / 2)) = 90 := by
  sorry

end tabs_remaining_l1535_153552


namespace classroom_ratio_problem_l1535_153553

theorem classroom_ratio_problem (num_girls : ℕ) (ratio : ℚ) (num_boys : ℕ) : 
  num_girls = 10 → ratio = 1/2 → num_girls = ratio * num_boys → num_boys = 20 := by
  sorry

end classroom_ratio_problem_l1535_153553


namespace no_integer_solution_for_rectangle_l1535_153544

theorem no_integer_solution_for_rectangle : 
  ¬ ∃ (w l : ℕ), w > 0 ∧ l > 0 ∧ w * l = 24 ∧ (w = l ∨ w = 2 * l) := by
  sorry

end no_integer_solution_for_rectangle_l1535_153544


namespace total_is_99_l1535_153590

/-- The total number of ducks and ducklings in Mary's observation --/
def total_ducks_and_ducklings : ℕ := by
  -- Define the number of ducks in each group
  let ducks_group1 : ℕ := 2
  let ducks_group2 : ℕ := 6
  let ducks_group3 : ℕ := 9
  
  -- Define the number of ducklings per duck in each group
  let ducklings_per_duck_group1 : ℕ := 5
  let ducklings_per_duck_group2 : ℕ := 3
  let ducklings_per_duck_group3 : ℕ := 6
  
  -- Calculate the total number of ducks and ducklings
  exact ducks_group1 * ducklings_per_duck_group1 +
        ducks_group2 * ducklings_per_duck_group2 +
        ducks_group3 * ducklings_per_duck_group3 +
        ducks_group1 + ducks_group2 + ducks_group3

/-- Theorem stating that the total number of ducks and ducklings is 99 --/
theorem total_is_99 : total_ducks_and_ducklings = 99 := by
  sorry

end total_is_99_l1535_153590


namespace max_gcd_14n_plus_5_9n_plus_4_l1535_153593

theorem max_gcd_14n_plus_5_9n_plus_4 :
  ∃ (k : ℕ), k > 0 ∧ ∀ (n : ℕ), n > 0 → Nat.gcd (14*n + 5) (9*n + 4) ≤ k ∧
  ∃ (m : ℕ), m > 0 ∧ Nat.gcd (14*m + 5) (9*m + 4) = k :=
by
  -- The proof goes here
  sorry

end max_gcd_14n_plus_5_9n_plus_4_l1535_153593


namespace paper_stack_height_l1535_153529

theorem paper_stack_height (sheets : ℕ) (height : ℝ) : 
  (400 : ℝ) / 4 = sheets / height → sheets = 600 :=
by
  sorry

end paper_stack_height_l1535_153529


namespace shaded_area_calculation_l1535_153536

/-- Given a square carpet with the described shaded squares, prove the total shaded area is 45 square feet. -/
theorem shaded_area_calculation (S T : ℝ) : 
  S > 0 → T > 0 → (12 : ℝ) / S = 4 → S / T = 2 → 
  S^2 + 16 * T^2 = 45 :=
by sorry

end shaded_area_calculation_l1535_153536


namespace angle_sum_equals_pi_l1535_153527

theorem angle_sum_equals_pi (x y : Real) : 
  x > 0 → x < π / 2 → y > 0 → y < π / 2 →
  4 * (Real.cos x)^2 + 3 * (Real.cos y)^2 = 2 →
  4 * Real.cos (2 * x) + 3 * Real.cos (2 * y) = 1 →
  2 * x + y = π := by
sorry

end angle_sum_equals_pi_l1535_153527


namespace line_symmetry_l1535_153537

-- Define the three lines
def line1 (x y : ℝ) : Prop := 2 * x - y + 3 = 0
def line2 (x y : ℝ) : Prop := x - y + 2 = 0
def line3 (x y : ℝ) : Prop := 2 * x - y + 1 = 0

-- Define symmetry with respect to a line
def symmetric_wrt (l1 l2 l3 : (ℝ → ℝ → Prop)) : Prop :=
  ∀ (x1 y1 x2 y2 : ℝ), 
    l1 x1 y1 → l2 x2 y2 → 
    ∃ (xm ym : ℝ), l3 xm ym ∧ 
      xm = (x1 + x2) / 2 ∧ 
      ym = (y1 + y2) / 2

-- Theorem statement
theorem line_symmetry : symmetric_wrt line1 line3 line2 := by
  sorry

end line_symmetry_l1535_153537


namespace alternating_sum_of_coefficients_l1535_153516

theorem alternating_sum_of_coefficients :
  ∀ (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℝ),
  (∀ x : ℝ, (x + 1)^2 * (x^2 - 7)^3 = a₀ + a₁*(x+2) + a₂*(x+2)^2 + a₃*(x+2)^3 + 
                                      a₄*(x+2)^4 + a₅*(x+2)^5 + a₆*(x+2)^6 + 
                                      a₇*(x+2)^7 + a₈*(x+2)^8) →
  a₁ - a₂ + a₃ - a₄ + a₅ - a₆ + a₇ = -58 := by
sorry

end alternating_sum_of_coefficients_l1535_153516


namespace mutually_inscribed_tetrahedra_exist_l1535_153507

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a tetrahedron as a set of four points
structure Tetrahedron where
  a : Point3D
  b : Point3D
  c : Point3D
  d : Point3D

-- Define the property of being coplanar
def coplanar (p q r s : Point3D) : Prop := sorry

-- Define the property of a tetrahedron being inscribed in another
def inscribed (t1 t2 : Tetrahedron) : Prop :=
  coplanar t2.a t1.b t1.c t1.d ∧
  coplanar t2.b t1.a t1.c t1.d ∧
  coplanar t2.c t1.a t1.b t1.d ∧
  coplanar t2.d t1.a t1.b t1.c

-- Define the property of two tetrahedra not sharing vertices
def no_shared_vertices (t1 t2 : Tetrahedron) : Prop :=
  t1.a ≠ t2.a ∧ t1.a ≠ t2.b ∧ t1.a ≠ t2.c ∧ t1.a ≠ t2.d ∧
  t1.b ≠ t2.a ∧ t1.b ≠ t2.b ∧ t1.b ≠ t2.c ∧ t1.b ≠ t2.d ∧
  t1.c ≠ t2.a ∧ t1.c ≠ t2.b ∧ t1.c ≠ t2.c ∧ t1.c ≠ t2.d ∧
  t1.d ≠ t2.a ∧ t1.d ≠ t2.b ∧ t1.d ≠ t2.c ∧ t1.d ≠ t2.d

-- The theorem to be proved
theorem mutually_inscribed_tetrahedra_exist : 
  ∃ (t1 t2 : Tetrahedron), inscribed t1 t2 ∧ inscribed t2 t1 ∧ no_shared_vertices t1 t2 := by
  sorry

end mutually_inscribed_tetrahedra_exist_l1535_153507


namespace staircase_problem_l1535_153563

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def jumps (step_size : ℕ) (total_steps : ℕ) : ℕ := 
  (total_steps + step_size - 1) / step_size

theorem staircase_problem (n : ℕ) : 
  is_prime n → 
  jumps 3 n - jumps 6 n = 25 → 
  ∃ m : ℕ, is_prime m ∧ 
           jumps 3 m - jumps 6 m = 25 ∧ 
           n + m = 300 :=
sorry

end staircase_problem_l1535_153563


namespace right_triangle_hypotenuse_l1535_153598

/-- The length of the hypotenuse of a right triangle with legs 140 and 336 units is 364 units. -/
theorem right_triangle_hypotenuse : ∀ (a b c : ℝ),
  a = 140 →
  b = 336 →
  c^2 = a^2 + b^2 →
  c = 364 :=
by
  sorry

end right_triangle_hypotenuse_l1535_153598


namespace ramanujan_number_l1535_153526

theorem ramanujan_number (h r : ℂ) : 
  h * r = 40 - 24 * I ∧ h = 4 + 4 * I → r = 2 - 8 * I :=
by sorry

end ramanujan_number_l1535_153526


namespace maple_trees_after_cutting_l1535_153531

/-- The number of maple trees remaining after cutting -/
def remaining_maple_trees (initial : ℝ) (cut : ℝ) : ℝ :=
  initial - cut

/-- Proof that the number of maple trees remaining is 7.0 -/
theorem maple_trees_after_cutting :
  remaining_maple_trees 9.0 2.0 = 7.0 := by
  sorry

end maple_trees_after_cutting_l1535_153531


namespace final_amount_is_16_l1535_153528

def purchase1 : ℚ := 215/100
def purchase2 : ℚ := 475/100
def purchase3 : ℚ := 1060/100
def discount_rate : ℚ := 1/10

def total_before_discount : ℚ := purchase1 + purchase2 + purchase3
def discounted_total : ℚ := total_before_discount * (1 - discount_rate)

def round_to_nearest_dollar (x : ℚ) : ℤ :=
  if x - ⌊x⌋ < 1/2 then ⌊x⌋ else ⌈x⌉

theorem final_amount_is_16 :
  round_to_nearest_dollar discounted_total = 16 := by
  sorry

end final_amount_is_16_l1535_153528


namespace perpendicular_line_through_vertex_l1535_153512

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = x^2 - 4*x + 4

/-- The given line equation -/
def given_line (x y : ℝ) : Prop := x/4 + y/3 = 1

/-- The perpendicular line equation to be proved -/
def perp_line (x y : ℝ) : Prop := y = (4/3)*x - 8/3

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (2, 0)

theorem perpendicular_line_through_vertex :
  ∃ (m b : ℝ), 
    (∀ x y, perp_line x y ↔ y = m*x + b) ∧ 
    perp_line vertex.1 vertex.2 ∧
    (∀ x₁ y₁ x₂ y₂, given_line x₁ y₁ → given_line x₂ y₂ → x₁ ≠ x₂ → 
      (y₂ - y₁)/(x₂ - x₁) * m = -1) :=
sorry

end perpendicular_line_through_vertex_l1535_153512


namespace mAssignment_is_valid_l1535_153514

/-- Represents a variable in a programming context -/
structure Variable where
  name : String

/-- Represents an expression in a programming context -/
inductive Expression where
  | Var : Variable → Expression
  | Neg : Expression → Expression

/-- Represents an assignment statement -/
structure Assignment where
  lhs : Variable
  rhs : Expression

/-- Checks if an assignment is valid according to programming rules -/
def isValidAssignment (a : Assignment) : Prop :=
  ∃ (v : Variable), a.lhs = v ∧ 
  (a.rhs = Expression.Var v ∨ a.rhs = Expression.Neg (Expression.Var v))

/-- The specific assignment M = -M -/
def mAssignment : Assignment where
  lhs := { name := "M" }
  rhs := Expression.Neg (Expression.Var { name := "M" })

theorem mAssignment_is_valid : isValidAssignment mAssignment := by
  sorry

end mAssignment_is_valid_l1535_153514


namespace dumbbell_weight_l1535_153530

/-- Given information about exercise bands and total weight, calculate the weight of the dumbbell. -/
theorem dumbbell_weight 
  (num_bands : ℕ) 
  (resistance_per_band : ℕ) 
  (total_weight : ℕ) 
  (h1 : num_bands = 2)
  (h2 : resistance_per_band = 5)
  (h3 : total_weight = 30) :
  total_weight - (num_bands * resistance_per_band) = 20 := by
  sorry

end dumbbell_weight_l1535_153530


namespace sequence_formula_l1535_153554

/-- Given a sequence {aₙ} with sum Sₙ = (3/2)aₙ - 3, prove aₙ = 3 * 2^n -/
theorem sequence_formula (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h : ∀ n : ℕ, n ≥ 1 → S n = (3/2) * a n - 3) :
  ∀ n : ℕ, n ≥ 1 → a n = 3 * 2^n := by
sorry

end sequence_formula_l1535_153554


namespace overtime_to_regular_pay_ratio_l1535_153502

/-- Proves that the ratio of overtime to regular pay rate is 2:1 given the problem conditions --/
theorem overtime_to_regular_pay_ratio :
  ∀ (regular_rate overtime_rate total_pay : ℚ) (regular_hours overtime_hours : ℕ),
    regular_rate = 3 →
    regular_hours = 40 →
    overtime_hours = 12 →
    total_pay = 192 →
    total_pay = regular_rate * regular_hours + overtime_rate * overtime_hours →
    overtime_rate / regular_rate = 2 := by
  sorry

end overtime_to_regular_pay_ratio_l1535_153502


namespace quadratic_equation_m_value_l1535_153580

/-- The equation ({m-2}){x^{m^2-2}}+4x-7=0 is quadratic -/
def is_quadratic (m : ℝ) : Prop :=
  (m^2 - 2 = 2) ∧ (m - 2 ≠ 0)

theorem quadratic_equation_m_value :
  ∀ m : ℝ, is_quadratic m → m = -2 :=
by
  sorry

end quadratic_equation_m_value_l1535_153580


namespace painted_cells_count_l1535_153574

/-- Represents a rectangular grid with alternating painted rows and columns -/
structure PaintedGrid where
  rows : Nat
  cols : Nat
  unpainted_cells : Nat

/-- Calculates the number of painted cells in the grid -/
def painted_cells (grid : PaintedGrid) : Nat :=
  grid.rows * grid.cols - grid.unpainted_cells

theorem painted_cells_count (grid : PaintedGrid) : 
  grid.rows = 5 ∧ grid.cols = 75 ∧ grid.unpainted_cells = 74 → painted_cells grid = 301 := by
  sorry

#check painted_cells_count

end painted_cells_count_l1535_153574


namespace division_of_powers_l1535_153575

theorem division_of_powers (a : ℝ) (h : a ≠ 0) : a^6 / (-a)^2 = a^4 := by
  sorry

end division_of_powers_l1535_153575


namespace R_24_divisible_by_R_4_Q_only_ones_and_zeros_zeros_in_Q_l1535_153540

/-- R_k represents an integer whose decimal representation consists of k consecutive 1s -/
def R (k : ℕ) : ℕ := (10^k - 1) / 9

/-- The quotient Q is defined as R_24 divided by R_4 -/
def Q : ℕ := R 24 / R 4

/-- count_zeros counts the number of zeros in the decimal representation of a natural number -/
def count_zeros (n : ℕ) : ℕ := sorry

theorem R_24_divisible_by_R_4 : R 24 % R 4 = 0 := sorry

theorem Q_only_ones_and_zeros : ∀ d : ℕ, d ∈ Q.digits 10 → d = 0 ∨ d = 1 := sorry

theorem zeros_in_Q : count_zeros Q = 15 := by sorry

end R_24_divisible_by_R_4_Q_only_ones_and_zeros_zeros_in_Q_l1535_153540


namespace quadratic_roots_product_l1535_153591

theorem quadratic_roots_product (x₁ x₂ : ℝ) : 
  x₁^2 - 3*x₁ + 2 = 0 → 
  x₂^2 - 3*x₂ + 2 = 0 → 
  (x₁ + 1) * (x₂ + 1) = 6 := by
  sorry

end quadratic_roots_product_l1535_153591


namespace chessboard_cut_parts_l1535_153510

/-- Represents a chessboard --/
structure Chessboard :=
  (size : ℕ)
  (white_squares : ℕ)
  (black_squares : ℕ)

/-- Represents the possible number of parts a chessboard can be cut into --/
def PossibleParts : Set ℕ := {2, 4, 8, 16, 32}

/-- Main theorem: The number of parts a chessboard can be cut into is a subset of PossibleParts --/
theorem chessboard_cut_parts (board : Chessboard) 
  (h1 : board.size = 8) 
  (h2 : board.white_squares = 32) 
  (h3 : board.black_squares = 32) : 
  ∃ (n : ℕ), n ∈ PossibleParts ∧ 
  (board.white_squares % n = 0) ∧ 
  (n > 1) ∧ 
  (n ≤ board.black_squares) :=
sorry

end chessboard_cut_parts_l1535_153510


namespace negative_sqrt_of_squared_negative_five_l1535_153509

theorem negative_sqrt_of_squared_negative_five :
  -Real.sqrt ((-5)^2) = -5 := by sorry

end negative_sqrt_of_squared_negative_five_l1535_153509


namespace pinwheel_shaded_area_l1535_153558

/-- Represents the pinwheel toy configuration -/
structure PinwheelToy where
  square_side : Real
  triangle_leg : Real
  π : Real

/-- Calculates the area of the shaded region in the pinwheel toy -/
def shaded_area (toy : PinwheelToy) : Real :=
  -- The actual calculation would go here
  286

/-- Theorem stating that the shaded area of the specific pinwheel toy is 286 square cm -/
theorem pinwheel_shaded_area :
  ∃ (toy : PinwheelToy),
    toy.square_side = 20 ∧
    toy.triangle_leg = 10 ∧
    toy.π = 3.14 ∧
    shaded_area toy = 286 := by
  sorry


end pinwheel_shaded_area_l1535_153558


namespace range_of_k_l1535_153589

/-- Given a function f(x) = (x^2 + x + 1) / (kx^2 + kx + 1) with domain R,
    the range of k is [0, 4) -/
theorem range_of_k (f : ℝ → ℝ) (k : ℝ) : 
  (∀ x, f x = (x^2 + x + 1) / (k*x^2 + k*x + 1)) → 
  (∀ x, k*x^2 + k*x + 1 ≠ 0) →
  (0 ≤ k ∧ k < 4) :=
sorry

end range_of_k_l1535_153589


namespace lemonade_sales_increase_l1535_153570

/-- Calculates the percentage increase in lemonade sales --/
def percentage_increase (last_week : ℕ) (total : ℕ) : ℚ :=
  let this_week := total - last_week
  ((this_week - last_week : ℚ) / last_week) * 100

/-- Theorem stating the percentage increase in lemonade sales --/
theorem lemonade_sales_increase :
  let last_week := 20
  let total := 46
  percentage_increase last_week total = 30 :=
by
  sorry

end lemonade_sales_increase_l1535_153570


namespace function_domain_range_implies_m_range_l1535_153588

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 4*x - 2

-- Define the theorem
theorem function_domain_range_implies_m_range 
  (m : ℝ) 
  (domain : Set ℝ)
  (range : Set ℝ)
  (h_domain : domain = Set.Icc 0 m)
  (h_range : range = Set.Icc (-6) (-2))
  (h_func_range : ∀ x ∈ domain, f x ∈ range) :
  m ∈ Set.Icc 2 4 :=
sorry

end function_domain_range_implies_m_range_l1535_153588


namespace max_distance_between_paths_l1535_153561

theorem max_distance_between_paths : 
  ∃ (C : ℝ), C = 3 * Real.sqrt 3 ∧ 
  ∀ (t : ℝ), 
    Real.sqrt ((t - (t - 5))^2 + (Real.sin t - Real.cos (t - 5))^2) ≤ C :=
sorry

end max_distance_between_paths_l1535_153561


namespace stating_max_pairs_remaining_l1535_153517

/-- Represents the number of shoe types -/
def num_types : ℕ := 5

/-- Represents the number of shoe colors -/
def num_colors : ℕ := 5

/-- Represents the initial number of shoe pairs -/
def initial_pairs : ℕ := 25

/-- Represents the number of individual shoes lost -/
def shoes_lost : ℕ := 9

/-- 
Theorem stating that given the initial conditions, the maximum number of 
complete pairs remaining after losing shoes is 22
-/
theorem max_pairs_remaining : 
  ∀ (remaining_pairs : ℕ),
  remaining_pairs ≤ initial_pairs ∧
  remaining_pairs ≥ initial_pairs - shoes_lost / 2 →
  remaining_pairs ≤ 22 :=
by sorry

end stating_max_pairs_remaining_l1535_153517


namespace binomial_coefficient_n_minus_two_l1535_153551

theorem binomial_coefficient_n_minus_two (n : ℕ) (h : n ≥ 2) :
  Nat.choose n (n - 2) = n * (n - 1) / 2 := by
  sorry

end binomial_coefficient_n_minus_two_l1535_153551


namespace complex_magnitude_theorem_l1535_153572

theorem complex_magnitude_theorem (z : ℂ) (h : z^2 - 2 * Complex.abs z + 3 = 0) : Complex.abs z = 1 := by
  sorry

end complex_magnitude_theorem_l1535_153572


namespace smallest_cubic_divisible_by_810_l1535_153525

theorem smallest_cubic_divisible_by_810 : ∃ (a : ℕ), 
  (∀ (n : ℕ), n < a → ¬(∃ (k : ℕ), n = k^3 ∧ 810 ∣ n)) ∧
  (∃ (k : ℕ), a = k^3) ∧ 
  (810 ∣ a) ∧
  a = 729000 := by
  sorry

end smallest_cubic_divisible_by_810_l1535_153525


namespace linear_relation_holds_l1535_153535

def points : List (ℤ × ℤ) := [(0, 200), (1, 160), (2, 120), (3, 80), (4, 40)]

theorem linear_relation_holds (p : ℤ × ℤ) (h : p ∈ points) : 
  (p.2 : ℤ) = 200 - 40 * p.1 := by
  sorry

end linear_relation_holds_l1535_153535


namespace triangle_angle_not_greater_than_60_l1535_153520

theorem triangle_angle_not_greater_than_60 (a b c : ℝ) (h_triangle : a + b + c = 180) 
  (h_positive : 0 < a ∧ 0 < b ∧ 0 < c) : 
  a ≤ 60 ∨ b ≤ 60 ∨ c ≤ 60 := by
  sorry

end triangle_angle_not_greater_than_60_l1535_153520


namespace undefined_rock_ratio_l1535_153562

-- Define the number of rocks Ted and Bill toss
def ted_rocks : ℕ := 10
def bill_rocks : ℕ := 0

-- Define a function to calculate the ratio
def rock_ratio (a b : ℕ) : Option ℚ :=
  if b = 0 then none else some (a / b)

-- Theorem statement
theorem undefined_rock_ratio :
  rock_ratio ted_rocks bill_rocks = none := by
sorry

end undefined_rock_ratio_l1535_153562


namespace parallelogram_area_l1535_153566

/-- The area of a parallelogram with base 21 and height 11 is 231 -/
theorem parallelogram_area : 
  ∀ (base height area : ℝ), 
    base = 21 → 
    height = 11 → 
    area = base * height → 
    area = 231 :=
by sorry

end parallelogram_area_l1535_153566


namespace polar_equation_pi_over_four_is_line_l1535_153556

/-- The curve defined by the polar equation θ = π/4 is a straight line -/
theorem polar_equation_pi_over_four_is_line : 
  ∀ (r : ℝ), ∃ (x y : ℝ), x = r * Real.cos (π/4) ∧ y = r * Real.sin (π/4) :=
sorry

end polar_equation_pi_over_four_is_line_l1535_153556


namespace symmetric_point_coordinates_l1535_153515

/-- A point in a 2D plane represented by its x and y coordinates -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Symmetry with respect to the x-axis -/
def symmetricXAxis (p : Point2D) : Point2D :=
  ⟨p.x, -p.y⟩

theorem symmetric_point_coordinates :
  let P : Point2D := ⟨-1, 2⟩
  let Q : Point2D := symmetricXAxis P
  Q.x = -1 ∧ Q.y = -2 := by
  sorry

end symmetric_point_coordinates_l1535_153515


namespace range_of_a_l1535_153569

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := 2 * a * x - a + 3

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∃ x₀ : ℝ, -1 < x₀ ∧ x₀ < 1 ∧ f a x₀ = 0) →
  (a < -3 ∨ a > 1) :=
by sorry

end range_of_a_l1535_153569


namespace real_part_of_complex_fraction_l1535_153543

theorem real_part_of_complex_fraction : 
  let i : ℂ := Complex.I
  Complex.re ((1 - i) / ((1 + i)^2)) = -1/2 := by sorry

end real_part_of_complex_fraction_l1535_153543


namespace parallel_transitive_l1535_153576

-- Define a type for vectors
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define a predicate for parallel vectors
def parallel (u v : V) : Prop := ∃ (k : ℝ), v = k • u

-- State the theorem
theorem parallel_transitive {a b c : V} (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hab : parallel a b) (hac : parallel a c) : parallel b c := by
  sorry

end parallel_transitive_l1535_153576


namespace complement_of_A_l1535_153524

def U : Set Int := {-2, -1, 0, 1, 2}

def A : Set Int := {x | x ∈ U ∧ x^2 + x - 2 < 0}

theorem complement_of_A : (U \ A) = {-2, 1, 2} := by sorry

end complement_of_A_l1535_153524


namespace missing_number_proof_l1535_153579

theorem missing_number_proof : ∃ x : ℚ, (306 / 34) * x + 270 = 405 ∧ x = 15 := by
  sorry

end missing_number_proof_l1535_153579


namespace reduced_oil_price_l1535_153583

/-- Represents the price and quantity of oil before and after a price reduction --/
structure OilPriceReduction where
  original_price : ℝ
  reduced_price : ℝ
  original_quantity : ℝ
  additional_quantity : ℝ
  total_cost : ℝ

/-- Theorem stating that given the conditions, the reduced price of oil is 60 --/
theorem reduced_oil_price
  (oil : OilPriceReduction)
  (price_reduction : oil.reduced_price = 0.75 * oil.original_price)
  (quantity_increase : oil.additional_quantity = 5)
  (cost_equality : oil.original_quantity * oil.original_price = 
                   (oil.original_quantity + oil.additional_quantity) * oil.reduced_price)
  (total_cost : oil.total_cost = 1200)
  : oil.reduced_price = 60 := by
  sorry

end reduced_oil_price_l1535_153583


namespace DE_DB_ratio_l1535_153534

-- Define the points
variable (A B C D E : ℝ × ℝ)

-- Define the conditions
axiom right_angle_ABC : (A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) = 0
axiom AC_length : Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2) = 4
axiom BC_length : Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) = 3
axiom right_angle_ABD : (B.1 - A.1) * (D.1 - A.1) + (B.2 - A.2) * (D.2 - A.2) = 0
axiom AD_length : Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2) = 15
axiom C_D_opposite : ((B.1 - A.1) * (C.2 - A.2) - (B.2 - A.2) * (C.1 - A.1)) *
                     ((B.1 - A.1) * (D.2 - A.2) - (B.2 - A.2) * (D.1 - A.1)) < 0
axiom D_parallel_AC : (D.2 - A.2) * (E.1 - C.1) = (D.1 - A.1) * (E.2 - C.2)
axiom E_on_CB_extended : ∃ t : ℝ, E = (B.1 + t * (B.1 - C.1), B.2 + t * (B.2 - C.2))

-- Define the theorem
theorem DE_DB_ratio :
  Real.sqrt ((D.1 - E.1)^2 + (D.2 - E.2)^2) / Real.sqrt ((D.1 - B.1)^2 + (D.2 - B.2)^2) = 57 / 80 :=
sorry

end DE_DB_ratio_l1535_153534
