import Mathlib

namespace line_translation_l2336_233679

/-- A line in the xy-plane. -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Vertical translation of a line. -/
def vertical_translate (l : Line) (d : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept - d }

theorem line_translation (l : Line) :
  l.slope = 3 ∧ l.intercept = 2 →
  vertical_translate l 3 = { slope := 3, intercept := -1 } := by
  sorry

end line_translation_l2336_233679


namespace freshman_class_size_l2336_233673

theorem freshman_class_size :
  ∃! n : ℕ, n < 400 ∧ n % 26 = 17 ∧ n % 24 = 6 :=
by
  use 379
  sorry

end freshman_class_size_l2336_233673


namespace triangle_max_perimeter_l2336_233682

theorem triangle_max_perimeter (a b y : ℕ) (ha : a = 7) (hb : b = 9) :
  (∃ (y : ℕ), a + b + y = (a + b + y).max (a + b + (a + b - 1))) :=
sorry

end triangle_max_perimeter_l2336_233682


namespace rectangular_prism_volume_l2336_233620

/-- Given a rectangular prism with base edge length 3 cm and lateral face diagonal 3√5 cm,
    prove that its volume is 54 cm³ -/
theorem rectangular_prism_volume (base_edge : ℝ) (lateral_diagonal : ℝ) (volume : ℝ) :
  base_edge = 3 →
  lateral_diagonal = 3 * Real.sqrt 5 →
  volume = base_edge * base_edge * Real.sqrt (lateral_diagonal^2 - base_edge^2) →
  volume = 54 := by
  sorry

end rectangular_prism_volume_l2336_233620


namespace harriet_round_trip_l2336_233674

/-- Harriet's round trip between A-ville and B-town -/
theorem harriet_round_trip 
  (speed_to_b : ℝ) 
  (speed_from_b : ℝ) 
  (time_to_b_minutes : ℝ) 
  (h1 : speed_to_b = 110) 
  (h2 : speed_from_b = 140) 
  (h3 : time_to_b_minutes = 168) : 
  let time_to_b := time_to_b_minutes / 60
  let distance := speed_to_b * time_to_b
  let time_from_b := distance / speed_from_b
  time_to_b + time_from_b = 5 := by
  sorry

end harriet_round_trip_l2336_233674


namespace complex_equation_solution_l2336_233647

theorem complex_equation_solution (z : ℂ) (i : ℂ) (h1 : i * i = -1) (h2 : i * z = 1) : z = -i := by
  sorry

end complex_equation_solution_l2336_233647


namespace expression_factorization_l2336_233671

theorem expression_factorization (x y z : ℤ) :
  x^2 - y^2 - z^2 + 2*y*z + x + y - z = (x + y - z) * (x - y + z + 1) := by
  sorry

end expression_factorization_l2336_233671


namespace irwin_family_hike_distance_l2336_233665

/-- The total distance hiked by Irwin's family during their camping trip. -/
def total_distance_hiked (car_to_stream stream_to_meadow meadow_to_campsite : ℝ) : ℝ :=
  car_to_stream + stream_to_meadow + meadow_to_campsite

/-- Theorem stating that the total distance hiked by Irwin's family is 0.7 miles. -/
theorem irwin_family_hike_distance :
  total_distance_hiked 0.2 0.4 0.1 = 0.7 := by
  sorry

end irwin_family_hike_distance_l2336_233665


namespace nine_chapters_problem_l2336_233638

/-- Represents the problem from "The Nine Chapters on the Mathematical Art" -/
theorem nine_chapters_problem (x y : ℤ) : 
  (∀ (z : ℤ), z * x = y → (8 * x - 3 = y ↔ z = 8) ∧ (7 * x + 4 = y ↔ z = 7)) →
  (8 * x - 3 = y ∧ 7 * x + 4 = y) :=
by sorry

end nine_chapters_problem_l2336_233638


namespace v_2007_equals_1_l2336_233678

-- Define the function g
def g : ℕ → ℕ
| 1 => 5
| 2 => 3
| 3 => 2
| 4 => 1
| 5 => 4
| _ => 0  -- For completeness, though not used in the problem

-- Define the sequence v
def v : ℕ → ℕ
| 0 => 5
| (n + 1) => g (v n)

-- Theorem statement
theorem v_2007_equals_1 : v 2007 = 1 := by
  sorry

end v_2007_equals_1_l2336_233678


namespace vector_collinearity_l2336_233631

/-- Given vectors a, b, and c in R², prove that if a = (-2, 0), b = (2, 1), c = (x, 1),
    and 3a + b is collinear with c, then x = -4. -/
theorem vector_collinearity (a b c : ℝ × ℝ) (x : ℝ) :
  a = (-2, 0) →
  b = (2, 1) →
  c = (x, 1) →
  ∃ (k : ℝ), k ≠ 0 ∧ (3 • a + b) = k • c →
  x = -4 := by
sorry

end vector_collinearity_l2336_233631


namespace initial_boys_count_l2336_233652

/-- Given a school with an initial number of girls and boys, and after some additions,
    prove that the initial number of boys was 214. -/
theorem initial_boys_count (initial_girls : ℕ) (initial_boys : ℕ) 
  (added_girls : ℕ) (added_boys : ℕ) (final_boys : ℕ) : 
  initial_girls = 135 → 
  added_girls = 496 → 
  added_boys = 910 → 
  final_boys = 1124 → 
  initial_boys + added_boys = final_boys → 
  initial_boys = 214 := by
sorry

end initial_boys_count_l2336_233652


namespace catch_up_time_tom_catches_jerry_l2336_233612

/-- Represents the figure-eight track --/
structure Track :=
  (small_loop : ℝ)
  (large_loop : ℝ)
  (large_loop_eq : large_loop = (4 / 3) * small_loop)

/-- Represents the runners Tom and Jerry --/
structure Runner :=
  (speed : ℝ)

/-- The problem setup --/
def problem (t : Track) (tom jerry : Runner) : Prop :=
  tom.speed > jerry.speed ∧
  tom.speed * 20 = t.large_loop ∧
  jerry.speed * 20 = t.small_loop ∧
  tom.speed * 15 = t.small_loop

/-- The theorem to be proved --/
theorem catch_up_time (t : Track) (tom jerry : Runner) 
  (h : problem t tom jerry) : ℝ := by
  sorry

/-- The main theorem stating that Tom catches up with Jerry in 80 minutes --/
theorem tom_catches_jerry (t : Track) (tom jerry : Runner) 
  (h : problem t tom jerry) : catch_up_time t tom jerry h = 80 := by
  sorry

end catch_up_time_tom_catches_jerry_l2336_233612


namespace logarithm_expression_equals_two_l2336_233662

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem logarithm_expression_equals_two :
  lg 2 * lg 2 + lg 2 * lg 5 + lg 50 = 2 := by
  sorry

end logarithm_expression_equals_two_l2336_233662


namespace sin_pi_plus_A_implies_cos_three_pi_half_minus_A_l2336_233687

theorem sin_pi_plus_A_implies_cos_three_pi_half_minus_A 
  (A : ℝ) (h : Real.sin (π + A) = 1/2) : 
  Real.cos ((3/2) * π - A) = 1/2 := by
  sorry

end sin_pi_plus_A_implies_cos_three_pi_half_minus_A_l2336_233687


namespace beth_double_age_in_8_years_l2336_233632

/-- The number of years until Beth is twice her sister's age -/
def years_until_double_age (beth_age : ℕ) (sister_age : ℕ) : ℕ :=
  beth_age + sister_age

theorem beth_double_age_in_8_years (beth_age : ℕ) (sister_age : ℕ) 
  (h1 : beth_age = 18) (h2 : sister_age = 5) :
  years_until_double_age beth_age sister_age = 8 :=
sorry

#check beth_double_age_in_8_years

end beth_double_age_in_8_years_l2336_233632


namespace tavern_keeper_pays_for_beer_l2336_233653

/-- Represents the currency of a country -/
structure Currency where
  name : String
  value : ℚ

/-- Represents a country with its currency and exchange rate -/
structure Country where
  name : String
  currency : Currency
  exchangeRate : ℚ

/-- Represents a transaction in a country -/
structure Transaction where
  country : Country
  amountPaid : ℚ
  itemCost : ℚ
  changeReceived : ℚ

/-- The beer lover's transactions -/
def beerLoverTransactions (anchuria gvaiasuela : Country) : List Transaction := sorry

/-- The tavern keeper's profit or loss -/
def tavernKeeperProfit (transactions : List Transaction) : ℚ := sorry

/-- Theorem stating that the tavern keeper pays for the beer -/
theorem tavern_keeper_pays_for_beer (anchuria gvaiasuela : Country) 
  (h1 : anchuria.currency.value = gvaiasuela.currency.value)
  (h2 : anchuria.exchangeRate = 90 / 100)
  (h3 : gvaiasuela.exchangeRate = 90 / 100)
  (h4 : ∀ t ∈ beerLoverTransactions anchuria gvaiasuela, t.itemCost = 10 / 100) :
  tavernKeeperProfit (beerLoverTransactions anchuria gvaiasuela) < 0 := by
  sorry

#check tavern_keeper_pays_for_beer

end tavern_keeper_pays_for_beer_l2336_233653


namespace simplify_fraction_l2336_233644

theorem simplify_fraction : 18 * (8 / 15) * (3 / 4) = 12 / 5 := by
  sorry

end simplify_fraction_l2336_233644


namespace arithmetic_progression_problem_l2336_233672

/-- 
Given an arithmetic progression with first term a₁ and common difference d,
if the product of the 3rd and 6th terms is 406,
and the 9th term divided by the 4th term gives a quotient of 2 with remainder 6,
then the first term is 4 and the common difference is 5.
-/
theorem arithmetic_progression_problem (a₁ d : ℚ) : 
  (a₁ + 2*d) * (a₁ + 5*d) = 406 →
  (a₁ + 8*d) = 2*(a₁ + 3*d) + 6 →
  a₁ = 4 ∧ d = 5 := by
  sorry


end arithmetic_progression_problem_l2336_233672


namespace hulk_jump_exceeds_500_l2336_233600

def hulk_jump (n : ℕ) : ℝ := 2 * (2 ^ (n - 1))

theorem hulk_jump_exceeds_500 :
  (∀ k < 9, hulk_jump k ≤ 500) ∧ hulk_jump 9 > 500 := by
  sorry

end hulk_jump_exceeds_500_l2336_233600


namespace julia_garden_area_l2336_233605

/-- Represents a rectangular garden with given walking constraints -/
structure RectangularGarden where
  length : ℝ
  width : ℝ
  length_walk : length * 30 = 1500
  perimeter_walk : (length + width) * 2 * 12 = 1500

/-- The area of Julia's garden is 625 square meters -/
theorem julia_garden_area (garden : RectangularGarden) : garden.length * garden.width = 625 := by
  sorry

#check julia_garden_area

end julia_garden_area_l2336_233605


namespace parallel_line_through_point_l2336_233640

/-- Given a line l: 3x + 4y - 12 = 0, prove that 3x + 4y - 9 = 0 is the equation of the line
    that passes through the point (-1, 3) and has the same slope as line l. -/
theorem parallel_line_through_point (x y : ℝ) : 
  let l : ℝ → ℝ → Prop := λ x y => 3 * x + 4 * y - 12 = 0
  let m : ℝ := -3 / 4  -- slope of line l
  let new_line : ℝ → ℝ → Prop := λ x y => 3 * x + 4 * y - 9 = 0
  (∀ x y, l x y → (y - 0) = m * (x - 0)) →  -- l has slope m
  new_line (-1) 3 →  -- new line passes through (-1, 3)
  (∀ x y, new_line x y → (y - 3) = m * (x - (-1))) →  -- new line has slope m
  ∀ x y, new_line x y ↔ 3 * x + 4 * y - 9 = 0 := by
sorry

end parallel_line_through_point_l2336_233640


namespace pen_count_problem_l2336_233602

theorem pen_count_problem (total_pens : ℕ) (difference : ℕ) 
  (h1 : total_pens = 140)
  (h2 : difference = 20) :
  ∃ (ballpoint_pens fountain_pens : ℕ),
    ballpoint_pens + fountain_pens = total_pens ∧
    ballpoint_pens + difference = fountain_pens ∧
    ballpoint_pens = 60 ∧
    fountain_pens = 80 := by
  sorry

end pen_count_problem_l2336_233602


namespace solution_set_implies_a_minus_b_l2336_233667

/-- The solution set of a quadratic inequality -/
def SolutionSet (a b : ℝ) : Set ℝ :=
  {x | a * x^2 + b * x + 2 > 0}

/-- The theorem stating the relationship between the solution set and the value of a - b -/
theorem solution_set_implies_a_minus_b (a b : ℝ) :
  SolutionSet a b = {x | -1/2 < x ∧ x < 1/3} → a - b = -10 := by
  sorry

end solution_set_implies_a_minus_b_l2336_233667


namespace lisa_walking_time_l2336_233601

/-- Given Lisa's walking speed and total distance over two days, prove she walks for 1 hour each day -/
theorem lisa_walking_time 
  (speed : ℝ)              -- Lisa's walking speed in meters per minute
  (total_distance : ℝ)     -- Total distance Lisa walks in two days
  (h1 : speed = 10)        -- Lisa walks 10 meters each minute
  (h2 : total_distance = 1200) -- Lisa walks 1200 meters in two days
  : (total_distance / 2) / speed / 60 = 1 := by
  sorry

end lisa_walking_time_l2336_233601


namespace cupcakes_per_package_l2336_233685

theorem cupcakes_per_package (packages : ℕ) (eaten : ℕ) (left : ℕ) :
  packages = 3 →
  eaten = 5 →
  left = 7 →
  ∃ cupcakes_per_package : ℕ,
    cupcakes_per_package * packages = eaten + left ∧
    cupcakes_per_package = 4 := by
  sorry

end cupcakes_per_package_l2336_233685


namespace largest_two_digit_number_with_condition_l2336_233646

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  units : Nat
  tens_valid : tens ≥ 1 ∧ tens ≤ 9
  units_valid : units ≥ 0 ∧ units ≤ 9

/-- Checks if a two-digit number satisfies the given condition -/
def satisfiesCondition (n : TwoDigitNumber) : Prop :=
  n.tens * n.units = n.tens + n.units + 17

/-- Theorem: 74 is the largest two-digit number satisfying the condition -/
theorem largest_two_digit_number_with_condition :
  ∀ n : TwoDigitNumber, satisfiesCondition n → n.tens * 10 + n.units ≤ 74 := by
  sorry

#check largest_two_digit_number_with_condition

end largest_two_digit_number_with_condition_l2336_233646


namespace cycling_trip_distances_l2336_233664

-- Define the total route distance
def total_distance : ℝ := 120

-- Define the distances traveled each day
def day1_distance : ℝ := 36
def day2_distance : ℝ := 40
def day3_distance : ℝ := 44

-- Theorem statement
theorem cycling_trip_distances :
  -- Day 1 condition
  day1_distance = total_distance / 3 - 4 ∧
  -- Day 2 condition
  day2_distance = (total_distance - day1_distance) / 2 - 2 ∧
  -- Day 3 condition
  day3_distance = (total_distance - day1_distance - day2_distance) * 10 / 11 + 4 ∧
  -- Total distance is the sum of all days
  total_distance = day1_distance + day2_distance + day3_distance :=
by sorry


end cycling_trip_distances_l2336_233664


namespace not_periodic_x_plus_cos_x_l2336_233684

theorem not_periodic_x_plus_cos_x : ¬∃ (T : ℝ), T ≠ 0 ∧ ∀ (x : ℝ), x + T + Real.cos (x + T) = x + Real.cos x := by
  sorry

end not_periodic_x_plus_cos_x_l2336_233684


namespace visit_either_not_both_l2336_233615

def probability_chile : ℝ := 0.5
def probability_madagascar : ℝ := 0.5

theorem visit_either_not_both :
  probability_chile + probability_madagascar - 2 * (probability_chile * probability_madagascar) = 0.5 := by
  sorry

end visit_either_not_both_l2336_233615


namespace complex_multiplication_l2336_233604

theorem complex_multiplication (i : ℂ) : i * i = -1 → i * (1 - 2*i) = 2 + i := by
  sorry

end complex_multiplication_l2336_233604


namespace min_value_theorem_l2336_233619

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (1 / a + 1 / b + 2 * Real.sqrt (a * b)) ≥ 4 ∧
  (1 / a + 1 / b + 2 * Real.sqrt (a * b) = 4 ↔ a = b) :=
sorry

end min_value_theorem_l2336_233619


namespace cylinder_radii_ratio_l2336_233642

/-- Given two cylinders of the same height, this theorem proves that if their volumes are 40 cc
    and 360 cc respectively, then the ratio of their radii is 1:3. -/
theorem cylinder_radii_ratio (h : ℝ) (r₁ r₂ : ℝ) 
  (h_pos : h > 0) (r₁_pos : r₁ > 0) (r₂_pos : r₂ > 0) :
  π * r₁^2 * h = 40 → π * r₂^2 * h = 360 → r₁ / r₂ = 1 / 3 := by
  sorry

#check cylinder_radii_ratio

end cylinder_radii_ratio_l2336_233642


namespace chicken_feed_requirement_l2336_233689

/-- Represents the problem of calculating chicken feed requirements --/
theorem chicken_feed_requirement 
  (chicken_price : ℝ) 
  (feed_price : ℝ) 
  (feed_weight : ℝ) 
  (num_chickens : ℕ) 
  (profit : ℝ) 
  (h1 : chicken_price = 1.5)
  (h2 : feed_price = 2)
  (h3 : feed_weight = 20)
  (h4 : num_chickens = 50)
  (h5 : profit = 65) :
  (num_chickens * chicken_price - profit) / feed_price * feed_weight / num_chickens = 2 := by
  sorry

#check chicken_feed_requirement

end chicken_feed_requirement_l2336_233689


namespace third_term_of_arithmetic_sequence_l2336_233660

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem third_term_of_arithmetic_sequence
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 1 + a 3 = 10) :
  a 2 = 5 :=
sorry

end third_term_of_arithmetic_sequence_l2336_233660


namespace arithmetic_mean_problem_l2336_233681

theorem arithmetic_mean_problem (original_list : List ℝ) 
  (a b c : ℝ) : 
  (original_list.length = 20) →
  (original_list.sum / original_list.length = 45) →
  (let new_list := original_list ++ [a, b, c]
   new_list.sum / new_list.length = 50) →
  (a + b + c) / 3 = 250 / 3 := by
sorry

end arithmetic_mean_problem_l2336_233681


namespace birds_in_tree_l2336_233649

theorem birds_in_tree (initial_birds final_birds : ℕ) 
  (h1 : initial_birds = 29)
  (h2 : final_birds = 42) :
  final_birds - initial_birds = 13 := by
  sorry

end birds_in_tree_l2336_233649


namespace not_all_even_numbers_multiple_of_eight_l2336_233621

theorem not_all_even_numbers_multiple_of_eight : ¬ (∀ n : ℕ, 2 ∣ n → 8 ∣ n) := by
  sorry

#check not_all_even_numbers_multiple_of_eight

end not_all_even_numbers_multiple_of_eight_l2336_233621


namespace base_conversion_equality_l2336_233618

/-- Given that 32₄ = 120ᵦ, prove that the unique positive integer b satisfying this equation is 2. -/
theorem base_conversion_equality (b : ℕ) : b > 0 ∧ (3 * 4 + 2) = (1 * b^2 + 2 * b + 0) → b = 2 := by
  sorry

end base_conversion_equality_l2336_233618


namespace co2_formation_l2336_233688

-- Define the chemical reaction
structure Reaction where
  hcl : ℕ
  nahco3 : ℕ
  co2 : ℕ

-- Define the stoichiometric ratio
def stoichiometric_ratio (r : Reaction) : Prop :=
  r.hcl = r.nahco3 ∧ r.co2 = r.hcl

-- Define the theorem
theorem co2_formation (r : Reaction) (h1 : stoichiometric_ratio r) (h2 : r.hcl = 3) (h3 : r.nahco3 = 3) :
  r.co2 = min r.hcl r.nahco3 := by
  sorry

#check co2_formation

end co2_formation_l2336_233688


namespace exchange_calculation_l2336_233641

/-- Exchange rate between Canadian and American dollars -/
def exchange_rate : ℚ := 120 / 80

/-- Amount of American dollars to be exchanged -/
def american_dollars : ℚ := 50

/-- Function to calculate Canadian dollars given American dollars -/
def exchange (usd : ℚ) : ℚ := usd * exchange_rate

theorem exchange_calculation :
  exchange american_dollars = 75 := by
  sorry

end exchange_calculation_l2336_233641


namespace initial_amount_is_100_l2336_233610

/-- The amount of money Jasmine spent on fruits -/
def spent_on_fruits : ℝ := 15

/-- The amount of money Jasmine had left to spend -/
def money_left : ℝ := 85

/-- The initial amount of money Jasmine's mom gave her -/
def initial_amount : ℝ := spent_on_fruits + money_left

/-- Theorem stating that the initial amount of money Jasmine's mom gave her is $100.00 -/
theorem initial_amount_is_100 : initial_amount = 100 := by sorry

end initial_amount_is_100_l2336_233610


namespace sequence_2023rd_term_l2336_233643

def sequence_term (n : ℕ) : ℚ := (-1)^n * (n : ℚ) / (n + 1)

theorem sequence_2023rd_term : sequence_term 2023 = -2023 / 2024 := by
  sorry

end sequence_2023rd_term_l2336_233643


namespace smallest_positive_integer_ending_in_3_divisible_by_5_l2336_233635

theorem smallest_positive_integer_ending_in_3_divisible_by_5 : 
  ∃ n : ℕ, n > 0 ∧ n % 10 = 3 ∧ n % 5 = 0 ∧ 
  ∀ m : ℕ, m > 0 → m % 10 = 3 → m % 5 = 0 → m ≥ n :=
by sorry

end smallest_positive_integer_ending_in_3_divisible_by_5_l2336_233635


namespace shopping_trip_tax_percentage_l2336_233624

/-- Represents the spending distribution and tax rates for a shopping trip -/
structure ShoppingTrip where
  clothing_percent : ℝ
  food_percent : ℝ
  other_percent : ℝ
  clothing_tax_rate : ℝ
  food_tax_rate : ℝ
  other_tax_rate : ℝ

/-- Calculates the total tax percentage for a given shopping trip -/
def totalTaxPercentage (trip : ShoppingTrip) : ℝ :=
  trip.clothing_percent * trip.clothing_tax_rate +
  trip.food_percent * trip.food_tax_rate +
  trip.other_percent * trip.other_tax_rate

/-- Theorem stating that for the given shopping trip, the total tax is 5% of the total amount spent excluding taxes -/
theorem shopping_trip_tax_percentage :
  let trip : ShoppingTrip := {
    clothing_percent := 0.5,
    food_percent := 0.2,
    other_percent := 0.3,
    clothing_tax_rate := 0.04,
    food_tax_rate := 0,
    other_tax_rate := 0.1
  }
  totalTaxPercentage trip = 0.05 := by
  sorry


end shopping_trip_tax_percentage_l2336_233624


namespace not_divisible_by_2n_plus_65_l2336_233613

theorem not_divisible_by_2n_plus_65 (n : ℕ+) : ¬(2^n.val + 65 ∣ 5^n.val - 3^n.val) := by
  sorry

end not_divisible_by_2n_plus_65_l2336_233613


namespace registration_count_l2336_233669

/-- The number of ways two students can register for universities --/
def registration_possibilities : ℕ :=
  let n_universities := 3
  let n_students := 2
  let choose_one := n_universities
  let choose_two := n_universities.choose 2
  (choose_one ^ n_students) + 
  (choose_two ^ n_students) + 
  (2 * choose_one * choose_two)

theorem registration_count : registration_possibilities = 36 := by
  sorry

end registration_count_l2336_233669


namespace hours_worked_per_day_l2336_233696

/-- Given a person who worked for 5 days and a total of 15 hours, 
    prove that the number of hours worked each day is equal to 3. -/
theorem hours_worked_per_day 
  (days_worked : ℕ) 
  (total_hours : ℕ) 
  (h1 : days_worked = 5) 
  (h2 : total_hours = 15) : 
  total_hours / days_worked = 3 := by
  sorry

end hours_worked_per_day_l2336_233696


namespace pyramid_base_side_length_l2336_233694

/-- Given a right pyramid with a square base, proves that the side length of the base is 10 meters 
    when the area of one lateral face is 120 square meters and the slant height is 24 meters. -/
theorem pyramid_base_side_length : 
  ∀ (base_side_length slant_height lateral_face_area : ℝ),
  slant_height = 24 →
  lateral_face_area = 120 →
  lateral_face_area = (1/2) * base_side_length * slant_height →
  base_side_length = 10 := by
sorry

end pyramid_base_side_length_l2336_233694


namespace sufficient_not_necessary_condition_l2336_233698

theorem sufficient_not_necessary_condition (m : ℝ) :
  (∀ m, m < -2 → ∃ x : ℝ, x^2 + m*x + 1 = 0) ∧
  ¬(∀ x : ℝ, x^2 + m*x + 1 = 0 → m < -2) :=
by sorry

end sufficient_not_necessary_condition_l2336_233698


namespace two_digit_powers_of_three_l2336_233639

theorem two_digit_powers_of_three :
  (∃! (s : Finset ℕ), ∀ n : ℕ, n ∈ s ↔ (10 ≤ 3^n ∧ 3^n ≤ 99)) ∧
  (∃! (s : Finset ℕ), ∀ n : ℕ, n ∈ s ↔ (10 ≤ 3^n ∧ 3^n ≤ 99) ∧ Finset.card s = 2) :=
by sorry

end two_digit_powers_of_three_l2336_233639


namespace final_weight_gain_l2336_233656

def weight_change (initial_weight : ℕ) (final_weight : ℕ) : ℕ :=
  let first_loss := 12
  let second_gain := 2 * first_loss
  let third_loss := 3 * first_loss
  let weight_after_third_loss := initial_weight - first_loss + second_gain - third_loss
  final_weight - weight_after_third_loss

theorem final_weight_gain (initial_weight final_weight : ℕ) 
  (h1 : initial_weight = 99)
  (h2 : final_weight = 81) :
  weight_change initial_weight final_weight = 6 := by
  sorry

#eval weight_change 99 81

end final_weight_gain_l2336_233656


namespace f_composition_theorem_l2336_233611

-- Define the function f
def f (p q : ℝ) (x : ℝ) : ℝ := x^2 + p*x + q

-- Define the condition that |f(x)| ≤ 1/2 for all x in [2, 4]
def f_condition (p q : ℝ) : Prop :=
  ∀ x : ℝ, 2 ≤ x ∧ x ≤ 4 → |f p q x| ≤ 1/2

-- Define the n-fold composition of f
def f_compose (p q : ℝ) : ℕ → ℝ → ℝ
  | 0, x => x
  | n+1, x => f p q (f_compose p q n x)

-- The theorem to prove
theorem f_composition_theorem (p q : ℝ) (h : f_condition p q) :
  f_compose p q 2017 ((5 - Real.sqrt 11) / 2) = (5 + Real.sqrt 11) / 2 :=
sorry

end f_composition_theorem_l2336_233611


namespace sum_of_products_l2336_233683

theorem sum_of_products (a b c : ℝ) (h1 : a^2 + b^2 + c^2 = 1) (h2 : a + b + c = 0) :
  a * b + a * c + b * c = -1/2 := by
sorry

end sum_of_products_l2336_233683


namespace chessboard_tiling_exists_l2336_233658

/-- Represents a chessboard of size 2^n x 2^n -/
structure Chessboard (n : ℕ) where
  size : Fin (2^n) × Fin (2^n)

/-- Represents an L-shaped triomino -/
inductive Triomino
  | L : Triomino

/-- Represents a tiling of a chessboard -/
def Tiling (n : ℕ) := Chessboard n → Option Triomino

/-- States that a tiling is valid for a chessboard with one square removed -/
def is_valid_tiling (n : ℕ) (t : Tiling n) (removed : Fin (2^n) × Fin (2^n)) : Prop :=
  ∀ (pos : Fin (2^n) × Fin (2^n)), pos ≠ removed → t ⟨pos⟩ = some Triomino.L

/-- Theorem: For any 2^n x 2^n chessboard with one square removed, 
    there exists a valid tiling using L-shaped triominoes -/
theorem chessboard_tiling_exists (n : ℕ) (removed : Fin (2^n) × Fin (2^n)) :
  ∃ (t : Tiling n), is_valid_tiling n t removed := by
  sorry

end chessboard_tiling_exists_l2336_233658


namespace apricot_trees_count_apricot_trees_proof_l2336_233697

theorem apricot_trees_count : ℕ → ℕ → Prop :=
  fun apricot_count peach_count =>
    peach_count = 3 * apricot_count →
    apricot_count + peach_count = 232 →
    apricot_count = 58

-- The proof is omitted as per instructions
theorem apricot_trees_proof : apricot_trees_count 58 174 := by
  sorry

end apricot_trees_count_apricot_trees_proof_l2336_233697


namespace sum_of_squares_unique_l2336_233699

theorem sum_of_squares_unique (p q r : ℕ+) : 
  p + q + r = 33 → 
  Nat.gcd p.val q.val + Nat.gcd q.val r.val + Nat.gcd r.val p.val = 11 → 
  p^2 + q^2 + r^2 = 419 :=
by sorry

end sum_of_squares_unique_l2336_233699


namespace fourteen_segments_l2336_233622

/-- A right triangle with integer leg lengths -/
structure RightTriangle where
  de : ℕ
  ef : ℕ

/-- The number of distinct integer lengths of line segments from E to DF -/
def numIntegerSegments (t : RightTriangle) : ℕ := sorry

/-- Our specific right triangle -/
def triangle : RightTriangle := { de := 24, ef := 25 }

/-- The theorem to prove -/
theorem fourteen_segments : numIntegerSegments triangle = 14 := by sorry

end fourteen_segments_l2336_233622


namespace unique_two_digit_integer_l2336_233690

theorem unique_two_digit_integer (s : ℕ) : 
  (∃! s, 10 ≤ s ∧ s < 100 ∧ 13 * s % 100 = 52) :=
by sorry

end unique_two_digit_integer_l2336_233690


namespace probability_of_specific_hand_l2336_233636

/-- Represents a standard 52-card deck -/
def StandardDeck : ℕ := 52

/-- Number of cards drawn -/
def NumDraws : ℕ := 5

/-- Number of Aces in a standard deck -/
def NumAces : ℕ := 4

/-- Number of suits in a standard deck -/
def NumSuits : ℕ := 4

/-- Probability of the specific outcome -/
def SpecificOutcomeProbability : ℚ := 3 / 832

theorem probability_of_specific_hand :
  let prob_ace : ℚ := NumAces / StandardDeck
  let prob_non_ace_suit : ℚ := (StandardDeck - NumAces) / StandardDeck
  let prob_specific_suit : ℚ := (StandardDeck / NumSuits) / StandardDeck
  NumSuits * prob_ace * prob_non_ace_suit * prob_specific_suit * prob_specific_suit = SpecificOutcomeProbability :=
sorry

end probability_of_specific_hand_l2336_233636


namespace sara_score_is_26_l2336_233614

/-- Represents a mathematics contest with a specific scoring system -/
structure MathContest where
  total_questions : Nat
  correct_points : Int
  incorrect_points : Int
  unanswered_points : Int

/-- Represents a contestant's performance in the math contest -/
structure ContestPerformance where
  correct_answers : Nat
  incorrect_answers : Nat
  unanswered : Nat

/-- Calculates the total score for a contestant given their performance and the contest rules -/
def calculate_score (contest : MathContest) (performance : ContestPerformance) : Int :=
  performance.correct_answers * contest.correct_points +
  performance.incorrect_answers * contest.incorrect_points +
  performance.unanswered * contest.unanswered_points

/-- The specific math contest Sara participated in -/
def sara_contest : MathContest :=
  { total_questions := 30
  , correct_points := 2
  , incorrect_points := -1
  , unanswered_points := 0 }

/-- Sara's performance in the contest -/
def sara_performance : ContestPerformance :=
  { correct_answers := 18
  , incorrect_answers := 10
  , unanswered := 2 }

/-- Theorem stating that Sara's score in the contest is 26 points -/
theorem sara_score_is_26 :
  calculate_score sara_contest sara_performance = 26 := by
  sorry

end sara_score_is_26_l2336_233614


namespace equation_solution_l2336_233609

theorem equation_solution :
  ∃ (square : ℚ),
    (((13/5 : ℚ) - ((17/2 : ℚ) - square) / (7/2 : ℚ)) / ((2 : ℚ) / 15)) = 2 ∧
    square = (1/3 : ℚ) := by
  sorry

end equation_solution_l2336_233609


namespace intersection_with_complement_l2336_233680

open Set

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {2, 3, 4}

theorem intersection_with_complement : A ∩ (U \ B) = {1} := by sorry

end intersection_with_complement_l2336_233680


namespace no_positive_integer_perfect_square_l2336_233693

theorem no_positive_integer_perfect_square : 
  ∀ n : ℕ+, ¬∃ y : ℤ, (n : ℤ)^2 - 21*(n : ℤ) + 110 = y^2 := by
  sorry

end no_positive_integer_perfect_square_l2336_233693


namespace triangle_count_is_83_l2336_233663

/-- Represents a rectangle divided into triangles -/
structure DividedRectangle where
  width : ℕ
  height : ℕ
  horizontal_divisions : ℕ
  vertical_divisions : ℕ

/-- Counts the number of triangles in the divided rectangle -/
def count_triangles (r : DividedRectangle) : ℕ :=
  sorry

/-- The specific rectangle from the problem -/
def problem_rectangle : DividedRectangle :=
  { width := 4
  , height := 5
  , horizontal_divisions := 4
  , vertical_divisions := 5 }

theorem triangle_count_is_83 : 
  count_triangles problem_rectangle = 83 :=
sorry

end triangle_count_is_83_l2336_233663


namespace regular_octagon_interior_angle_l2336_233651

theorem regular_octagon_interior_angle : ℝ :=
  let n : ℕ := 8  -- number of sides in an octagon
  let sum_of_interior_angles : ℝ := (n - 2) * 180
  let interior_angle : ℝ := sum_of_interior_angles / n
  135

/- Proof
sorry
-/

end regular_octagon_interior_angle_l2336_233651


namespace trigonometric_fraction_equals_two_l2336_233657

theorem trigonometric_fraction_equals_two :
  (3 - Real.sin (70 * π / 180)) / (2 - Real.cos (10 * π / 180) ^ 2) = 2 := by
  sorry

end trigonometric_fraction_equals_two_l2336_233657


namespace sphere_volume_ratio_l2336_233668

theorem sphere_volume_ratio (r R : ℝ) (h : r > 0) (H : R > 0) :
  (4 * Real.pi * r^2) / (4 * Real.pi * R^2) = 4 / 9 →
  ((4 / 3) * Real.pi * r^3) / ((4 / 3) * Real.pi * R^3) = 8 / 27 :=
by sorry

end sphere_volume_ratio_l2336_233668


namespace legendre_symbol_three_l2336_233627

-- Define the Legendre symbol
noncomputable def legendre_symbol (a p : ℕ) : ℤ := sorry

-- Define the theorem
theorem legendre_symbol_three (p : ℕ) (h_prime : Nat.Prime p) :
  (p % 12 = 1 ∨ p % 12 = 11 → legendre_symbol 3 p = 1) ∧
  (p % 12 = 5 ∨ p % 12 = 7 → legendre_symbol 3 p = -1) := by
  sorry

end legendre_symbol_three_l2336_233627


namespace interior_angle_sum_l2336_233606

theorem interior_angle_sum (n : ℕ) : 
  (180 * (n - 2) = 2160) → (180 * ((n + 3) - 2) = 2700) := by
  sorry

end interior_angle_sum_l2336_233606


namespace diameter_line_equation_l2336_233616

/-- Given a circle and a point inside it, prove the equation of the line containing the diameter through the point. -/
theorem diameter_line_equation (x y : ℝ) :
  (x - 1)^2 + y^2 = 4 →  -- Circle equation
  (2 : ℝ) - 1 < 2 →      -- Point (2,1) is inside the circle
  ∃ (m b : ℝ), x - y - 1 = 0 ∧ 
    (∀ (x' y' : ℝ), (x' - 1)^2 + y'^2 = 4 → (y' - 1) = m * (x' - 2) + b) :=
by sorry

end diameter_line_equation_l2336_233616


namespace octal_246_equals_166_l2336_233637

/-- Converts a base-8 digit to its base-10 equivalent -/
def octal_to_decimal (digit : ℕ) : ℕ := digit

/-- Represents a base-8 number as a list of digits -/
def octal_number : List ℕ := [2, 4, 6]

/-- Converts a base-8 number to its base-10 equivalent -/
def octal_to_decimal_conversion (num : List ℕ) : ℕ :=
  List.foldl (fun acc (digit : ℕ) => acc * 8 + octal_to_decimal digit) 0 num.reverse

theorem octal_246_equals_166 :
  octal_to_decimal_conversion octal_number = 166 := by
  sorry

end octal_246_equals_166_l2336_233637


namespace isosceles_triangle_perimeter_l2336_233630

/-- An isosceles triangle with side lengths 2 and 4 has a perimeter of 10 -/
theorem isosceles_triangle_perimeter (a b c : ℝ) : 
  a = 2 ∧ b = 4 ∧ c = 4 ∧  -- Two sides are equal (isosceles) and one side is 2
  a + b > c ∧ b + c > a ∧ a + c > b →  -- Triangle inequality
  a + b + c = 10 :=
by sorry

end isosceles_triangle_perimeter_l2336_233630


namespace right_triangle_hypotenuse_l2336_233692

theorem right_triangle_hypotenuse (a b c : ℝ) :
  a > 0 → b > 0 → c > 0 →
  a^2 = 5 → b^2 = 12 → c^2 = a^2 + b^2 →
  c^2 = 17 := by sorry

end right_triangle_hypotenuse_l2336_233692


namespace train_speed_l2336_233645

/-- Proves that a train of given length passing a point in a given time has a specific speed in kmph -/
theorem train_speed (train_length : Real) (time_to_pass : Real) (speed_kmph : Real) : 
  train_length = 20 →
  time_to_pass = 1.9998400127989762 →
  speed_kmph = (train_length / time_to_pass) * 3.6 →
  speed_kmph = 36.00287986320432 := by
  sorry

#check train_speed

end train_speed_l2336_233645


namespace quadratic_roots_max_value_l2336_233634

theorem quadratic_roots_max_value (t q : ℝ) (a₁ a₂ : ℝ) : 
  (∀ (n : ℕ), 1 ≤ n → n ≤ 2010 → a₁^n + a₂^n = a₁ + a₂) →
  a₁^2 - t*a₁ + q = 0 →
  a₂^2 - t*a₂ + q = 0 →
  (∀ (x : ℝ), x^2 - t*x + q ≠ 0 ∨ x = a₁ ∨ x = a₂) →
  (1 / a₁^2011 + 1 / a₂^2011) ≤ 2 :=
by sorry

end quadratic_roots_max_value_l2336_233634


namespace hundredth_card_is_ninth_l2336_233666

/-- Represents the cyclic order of cards in a standard deck --/
def cardCycle : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

/-- The number of cards in a cycle --/
def cycleLength : ℕ := 13

/-- The position we want to find --/
def targetPosition : ℕ := 100

/-- Function to get the equivalent position in the cycle --/
def cyclicPosition (n : ℕ) : ℕ :=
  (n - 1) % cycleLength + 1

theorem hundredth_card_is_ninth (h : targetPosition = 100) :
  cyclicPosition targetPosition = 9 := by
  sorry

end hundredth_card_is_ninth_l2336_233666


namespace max_x_minus_y_is_sqrt5_l2336_233629

theorem max_x_minus_y_is_sqrt5 (x y : ℝ) (h : x^2 + 2*x*y + y^2 + 4*x^2*y^2 = 4) :
  ∃ (max : ℝ), (∀ (a b : ℝ), a^2 + 2*a*b + b^2 + 4*a^2*b^2 = 4 → a - b ≤ max) ∧ max = Real.sqrt 5 :=
sorry

end max_x_minus_y_is_sqrt5_l2336_233629


namespace function_simplification_l2336_233648

theorem function_simplification (θ : Real) 
  (h1 : θ ∈ Set.Icc π (2 * π)) 
  (h2 : Real.tan θ = 2) : 
  ((1 + Real.sin θ + Real.cos θ) * (Real.sin (θ / 2) - Real.cos (θ / 2))) / 
   Real.sqrt (2 + 2 * Real.cos θ) = -Real.sqrt 5 / 5 := by
  sorry

end function_simplification_l2336_233648


namespace glee_club_female_members_l2336_233625

theorem glee_club_female_members 
  (total_members : ℕ) 
  (female_ratio : ℕ) 
  (male_ratio : ℕ) 
  (h1 : total_members = 18)
  (h2 : female_ratio = 2)
  (h3 : male_ratio = 1)
  (h4 : female_ratio * male_members + male_ratio * male_members = total_members)
  : female_ratio * male_members = 12 :=
by
  sorry

#check glee_club_female_members

end glee_club_female_members_l2336_233625


namespace rd_investment_exceeds_target_l2336_233655

def initial_investment : ℝ := 1.3
def annual_increase : ℝ := 0.12
def target_investment : ℝ := 2.0
def start_year : ℕ := 2015
def target_year : ℕ := 2019

theorem rd_investment_exceeds_target :
  (initial_investment * (1 + annual_increase) ^ (target_year - start_year) > target_investment) ∧
  (∀ y : ℕ, y < target_year → initial_investment * (1 + annual_increase) ^ (y - start_year) ≤ target_investment) :=
by sorry

end rd_investment_exceeds_target_l2336_233655


namespace sqrt_real_range_l2336_233603

theorem sqrt_real_range (x : ℝ) : (∃ y : ℝ, y ^ 2 = 4 + 2 * x) ↔ x ≥ -2 := by sorry

end sqrt_real_range_l2336_233603


namespace last_digit_of_max_value_l2336_233695

/-- Operation that combines two numbers a and b into a * b + 1 -/
def combine (a b : ℕ) : ℕ := a * b + 1

/-- Type representing the state of the blackboard -/
def Blackboard := List ℕ

/-- Function to perform one step of the operation -/
def performStep (board : Blackboard) : Blackboard :=
  match board with
  | a :: b :: rest => combine a b :: rest
  | _ => board

/-- Function to perform n steps of the operation -/
def performNSteps (n : ℕ) (board : Blackboard) : Blackboard :=
  match n with
  | 0 => board
  | n + 1 => performNSteps n (performStep board)

/-- The maximum possible value after 127 operations -/
def maxFinalValue : ℕ :=
  let initialBoard : Blackboard := List.replicate 128 1
  (performNSteps 127 initialBoard).head!

theorem last_digit_of_max_value :
  maxFinalValue % 10 = 2 := by
  sorry

end last_digit_of_max_value_l2336_233695


namespace decimal_to_binary_23_l2336_233633

theorem decimal_to_binary_23 : 
  (23 : ℕ) = (1 * 2^4 + 0 * 2^3 + 1 * 2^2 + 1 * 2^1 + 1 * 2^0) := by
  sorry

end decimal_to_binary_23_l2336_233633


namespace solve_p_q_system_l2336_233607

theorem solve_p_q_system (p q : ℝ) (hp : p > 1) (hq : q > 1)
  (h1 : 1 / p + 1 / q = 1) (h2 : p * q = 9) :
  q = (9 + 3 * Real.sqrt 5) / 2 := by
  sorry

end solve_p_q_system_l2336_233607


namespace problem_bottle_height_l2336_233628

/-- Represents a bottle constructed from two cylinders -/
structure Bottle where
  small_radius : ℝ
  large_radius : ℝ
  water_height_right_side_up : ℝ
  water_height_upside_down : ℝ

/-- Calculates the total height of the bottle -/
def total_height (b : Bottle) : ℝ :=
  sorry

/-- The specific bottle from the problem -/
def problem_bottle : Bottle :=
  { small_radius := 1
  , large_radius := 3
  , water_height_right_side_up := 20
  , water_height_upside_down := 28 }

/-- Theorem stating that the total height of the problem bottle is 29 cm -/
theorem problem_bottle_height :
  total_height problem_bottle = 29 := by sorry

end problem_bottle_height_l2336_233628


namespace probability_of_one_in_pascal_triangle_l2336_233623

def pascalTriangleElements (n : ℕ) : ℕ := n * (n + 1) / 2

def onesInPascalTriangle (n : ℕ) : ℕ := 1 + 2 * (n - 1)

theorem probability_of_one_in_pascal_triangle : 
  (onesInPascalTriangle 20 : ℚ) / (pascalTriangleElements 20 : ℚ) = 39 / 210 := by
  sorry

end probability_of_one_in_pascal_triangle_l2336_233623


namespace arithmetic_progression_reciprocals_squares_l2336_233617

theorem arithmetic_progression_reciprocals_squares (a b c : ℝ) :
  (2 / (c + a) = 1 / (b + c) + 1 / (b + a)) →
  (a^2 + c^2 = 2 * b^2) :=
by sorry

end arithmetic_progression_reciprocals_squares_l2336_233617


namespace no_equal_distribution_l2336_233670

/-- Represents the number of glasses --/
def num_glasses : ℕ := 2018

/-- Represents the total amount of champagne --/
def total_champagne : ℕ := 2019

/-- Represents a distribution of champagne among glasses --/
def Distribution := Fin num_glasses → ℚ

/-- Checks if a distribution is valid (sums to total_champagne) --/
def is_valid_distribution (d : Distribution) : Prop :=
  (Finset.sum Finset.univ (λ i => d i)) = total_champagne

/-- Represents the equalization operation on two glasses --/
def equalize (d : Distribution) (i j : Fin num_glasses) : Distribution :=
  λ k => if k = i ∨ k = j then (d i + d j) / 2 else d k

/-- Represents the property of all glasses having equal integer amount --/
def all_equal_integer (d : Distribution) : Prop :=
  ∃ n : ℕ, ∀ i : Fin num_glasses, d i = n

/-- The main theorem stating that no initial distribution can result in
    all glasses having equal integer amount after repeated equalization --/
theorem no_equal_distribution :
  ¬∃ (d : Distribution), is_valid_distribution d ∧
    ∃ (seq : ℕ → Fin num_glasses × Fin num_glasses),
      ∃ (n : ℕ), all_equal_integer (Nat.iterate (λ d' => equalize d' (seq n).1 (seq n).2) n d) := by
  sorry

end no_equal_distribution_l2336_233670


namespace max_sum_of_product_2401_l2336_233676

theorem max_sum_of_product_2401 :
  ∀ A B C : ℕ+,
  A ≠ B → B ≠ C → A ≠ C →
  A * B * C = 2401 →
  A + B + C ≤ 351 :=
by sorry

end max_sum_of_product_2401_l2336_233676


namespace product_plus_twenty_l2336_233654

theorem product_plus_twenty : ∃ n : ℕ, n = 5 * 7 ∧ n + 12 + 8 = 55 := by sorry

end product_plus_twenty_l2336_233654


namespace power_product_squared_l2336_233677

theorem power_product_squared (a b : ℝ) : (a^2 * b)^2 = a^4 * b^2 := by
  sorry

end power_product_squared_l2336_233677


namespace number_of_hens_l2336_233686

/-- Given the following conditions:
  - The total cost of pigs and hens is 1200 Rs
  - There are 3 pigs
  - The average price of a hen is 30 Rs
  - The average price of a pig is 300 Rs
  Prove that the number of hens bought is 10. -/
theorem number_of_hens (total_cost : ℕ) (num_pigs : ℕ) (hen_price : ℕ) (pig_price : ℕ) 
  (h1 : total_cost = 1200)
  (h2 : num_pigs = 3)
  (h3 : hen_price = 30)
  (h4 : pig_price = 300) :
  ∃ (num_hens : ℕ), num_hens * hen_price + num_pigs * pig_price = total_cost ∧ num_hens = 10 :=
by sorry

end number_of_hens_l2336_233686


namespace tile_difference_is_88_l2336_233626

/-- The number of tiles in the n-th square of the sequence -/
def tiles_in_square (n : ℕ) : ℕ := (2 * n - 1) ^ 2

/-- The difference in the number of tiles between the 7th and 5th squares -/
def tile_difference : ℕ := tiles_in_square 7 - tiles_in_square 5

theorem tile_difference_is_88 : tile_difference = 88 := by
  sorry

end tile_difference_is_88_l2336_233626


namespace wavelength_in_scientific_notation_l2336_233661

/-- Converts nanometers to meters -/
def nm_to_m (nm : ℝ) : ℝ := nm * 0.000000001

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  property : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def to_scientific_notation (x : ℝ) : ScientificNotation :=
  sorry

theorem wavelength_in_scientific_notation :
  let wavelength_nm : ℝ := 688
  let wavelength_m : ℝ := nm_to_m wavelength_nm
  let scientific : ScientificNotation := to_scientific_notation wavelength_m
  scientific.coefficient = 6.88 ∧ scientific.exponent = -7 :=
sorry

end wavelength_in_scientific_notation_l2336_233661


namespace soccer_team_activities_l2336_233675

/-- The number of activities required for a soccer team practice --/
def total_activities (total_players : ℕ) (goalies : ℕ) : ℕ :=
  let non_goalie_activities := goalies * (total_players - 1)
  2 * non_goalie_activities

theorem soccer_team_activities :
  total_activities 25 4 = 192 := by
  sorry

end soccer_team_activities_l2336_233675


namespace base7_addition_l2336_233608

/-- Converts a base 7 number represented as a list of digits to base 10 --/
def toBase10 (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (7 ^ i)) 0

/-- Converts a base 10 number to base 7 represented as a list of digits --/
def toBase7 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
  aux n []

theorem base7_addition :
  toBase7 (toBase10 [3, 5, 4, 1] + toBase10 [4, 1, 6, 3, 2]) = [5, 5, 4, 5, 2] := by
  sorry

end base7_addition_l2336_233608


namespace orthocenter_of_triangle_l2336_233659

/-- The orthocenter of a triangle ABC --/
def orthocenter (A B C : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

/-- Theorem: The orthocenter of triangle ABC with given coordinates --/
theorem orthocenter_of_triangle :
  let A : ℝ × ℝ × ℝ := (2, 3, 4)
  let B : ℝ × ℝ × ℝ := (4, 1, 1)
  let C : ℝ × ℝ × ℝ := (1, 5, 6)
  orthocenter A B C = (-79/3, 91/3, 41/3) := by sorry

end orthocenter_of_triangle_l2336_233659


namespace volleyball_team_selection_l2336_233691

def total_players : ℕ := 16
def quadruplets : ℕ := 4
def starters : ℕ := 7
def quadruplets_in_lineup : ℕ := 3

theorem volleyball_team_selection :
  (Nat.choose quadruplets quadruplets_in_lineup) *
  (Nat.choose (total_players - quadruplets) (starters - quadruplets_in_lineup)) = 1980 := by
  sorry

end volleyball_team_selection_l2336_233691


namespace coin_division_l2336_233650

theorem coin_division (n : ℕ) : 
  (n > 0) →
  (n % 6 = 4) → 
  (n % 5 = 3) → 
  (∀ m : ℕ, m > 0 ∧ m < n → (m % 6 ≠ 4 ∨ m % 5 ≠ 3)) →
  (n % 7 = 0) := by
sorry

end coin_division_l2336_233650
