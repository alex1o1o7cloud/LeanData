import Mathlib

namespace arithmetic_geometric_mean_squared_l1973_197317

theorem arithmetic_geometric_mean_squared (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a^2 + b^2) / 2 ≥ ((a + b) / 2)^2 := by
  sorry

end arithmetic_geometric_mean_squared_l1973_197317


namespace middle_box_statement_l1973_197330

/-- Represents the two possible statements on a box. -/
inductive BoxStatement
  | NoPrizeHere
  | PrizeInNeighbor

/-- Represents a configuration of boxes with their statements. -/
def BoxConfiguration := Fin 23 → BoxStatement

/-- Checks if the given configuration is valid according to the problem rules. -/
def isValidConfiguration (config : BoxConfiguration) (prizeBox : Fin 23) : Prop :=
  -- Exactly one statement is true
  (∃! i, (config i = BoxStatement.NoPrizeHere ∧ i = prizeBox) ∨
         (config i = BoxStatement.PrizeInNeighbor ∧ (i + 1 = prizeBox ∨ i - 1 = prizeBox))) ∧
  -- The prize box exists
  (∃ i, i = prizeBox)

/-- The middle box index (0-based). -/
def middleBoxIndex : Fin 23 := ⟨11, by norm_num⟩

/-- The main theorem stating that the middle box must be labeled "The prize is in the neighboring box." -/
theorem middle_box_statement (config : BoxConfiguration) (prizeBox : Fin 23) 
    (h : isValidConfiguration config prizeBox) :
    config middleBoxIndex = BoxStatement.PrizeInNeighbor := by
  sorry


end middle_box_statement_l1973_197330


namespace james_out_of_pocket_cost_l1973_197399

theorem james_out_of_pocket_cost (doctor_charge : ℝ) (insurance_coverage_percent : ℝ) 
  (h1 : doctor_charge = 300)
  (h2 : insurance_coverage_percent = 80) :
  doctor_charge * (1 - insurance_coverage_percent / 100) = 60 := by
  sorry

end james_out_of_pocket_cost_l1973_197399


namespace derivative_zero_necessary_not_sufficient_l1973_197309

-- Define a differentiable function
variable (f : ℝ → ℝ)
variable (hf : Differentiable ℝ f)

-- Define what it means for a function to have an extremum at a point
def HasExtremumAt (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ ε > 0, ∀ y ∈ Set.Ioo (x - ε) (x + ε), f y ≤ f x ∨ f y ≥ f x

-- State the theorem
theorem derivative_zero_necessary_not_sufficient :
  (∀ x : ℝ, HasExtremumAt f x → deriv f x = 0) ∧
  ¬(∀ x : ℝ, deriv f x = 0 → HasExtremumAt f x) :=
sorry

end derivative_zero_necessary_not_sufficient_l1973_197309


namespace unique_m_value_l1973_197337

theorem unique_m_value (m : ℝ) : 
  let A : Set ℝ := {0, m, m^2 - 3*m + 2}
  2 ∈ A → m = 3 :=
by sorry

end unique_m_value_l1973_197337


namespace johns_skateboarding_distance_l1973_197336

/-- The total distance John skateboarded, given his journey details -/
def total_skateboarding_distance (initial_skate : ℝ) (walk : ℝ) : ℝ :=
  2 * (initial_skate + walk) - walk

/-- Theorem stating that John's total skateboarding distance is 24 miles -/
theorem johns_skateboarding_distance :
  total_skateboarding_distance 10 4 = 24 := by
  sorry

end johns_skateboarding_distance_l1973_197336


namespace line_vector_to_slope_intercept_l1973_197319

/-- Given a line expressed in vector form, prove its slope-intercept form --/
theorem line_vector_to_slope_intercept :
  ∀ (x y : ℝ), 
  (2 : ℝ) * (x - 3) + (-1 : ℝ) * (y - (-4)) = 0 →
  y = 2 * x - 10 := by
sorry

end line_vector_to_slope_intercept_l1973_197319


namespace c_share_is_36_l1973_197361

/-- Represents the rental information for a person --/
structure RentalInfo where
  oxen : ℕ
  months : ℕ

/-- Calculates the share of rent for a given person --/
def calculateShare (totalRent : ℚ) (totalOxMonths : ℕ) (info : RentalInfo) : ℚ :=
  totalRent * (info.oxen * info.months : ℚ) / totalOxMonths

theorem c_share_is_36 
  (totalRent : ℚ)
  (a b c : RentalInfo)
  (h_total_rent : totalRent = 140)
  (h_a : a = ⟨10, 7⟩)
  (h_b : b = ⟨12, 5⟩)
  (h_c : c = ⟨15, 3⟩) :
  calculateShare totalRent (a.oxen * a.months + b.oxen * b.months + c.oxen * c.months) c = 36 := by
  sorry

#check c_share_is_36

end c_share_is_36_l1973_197361


namespace cos_x_plus_pi_sixth_l1973_197313

theorem cos_x_plus_pi_sixth (x : ℝ) (h : Real.sin (π / 3 - x) = 3 / 5) :
  Real.cos (x + π / 6) = 3 / 5 := by
  sorry

end cos_x_plus_pi_sixth_l1973_197313


namespace cube_volume_problem_l1973_197383

theorem cube_volume_problem (a : ℝ) : 
  (a - 2) * a * (a + 2) = a^3 - 8 → a^3 = 8 := by sorry

end cube_volume_problem_l1973_197383


namespace milk_water_ratio_l1973_197320

theorem milk_water_ratio (initial_volume : ℚ) (initial_milk_ratio : ℚ) (initial_water_ratio : ℚ) (added_water : ℚ) : 
  initial_volume = 45 →
  initial_milk_ratio = 4 →
  initial_water_ratio = 1 →
  added_water = 18 →
  let initial_milk := (initial_milk_ratio / (initial_milk_ratio + initial_water_ratio)) * initial_volume
  let initial_water := (initial_water_ratio / (initial_milk_ratio + initial_water_ratio)) * initial_volume
  let new_water := initial_water + added_water
  let new_milk_ratio := initial_milk / new_water
  let new_water_ratio := new_water / new_water
  (new_milk_ratio : ℚ) / (new_water_ratio : ℚ) = 4 / 3 := by
sorry

end milk_water_ratio_l1973_197320


namespace range_of_k_for_inequality_l1973_197302

theorem range_of_k_for_inequality (k : ℝ) : 
  (∀ a b : ℝ, (a - b)^2 ≥ k * a * b) ↔ k ∈ Set.Icc (-4) 0 := by
  sorry

end range_of_k_for_inequality_l1973_197302


namespace negation_of_implication_negation_of_positive_square_l1973_197380

theorem negation_of_implication (P Q : Prop) :
  ¬(P → Q) ↔ (P ∧ ¬Q) :=
by sorry

theorem negation_of_positive_square :
  ¬(∀ x : ℝ, x > 0 → x^2 > 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 ≤ 0) :=
by sorry

end negation_of_implication_negation_of_positive_square_l1973_197380


namespace special_line_equation_l1973_197386

/-- A line passing through point (3, -1) with equal absolute values of intercepts on both axes -/
structure SpecialLine where
  -- The equation of the line in the form ax + by + c = 0
  a : ℝ
  b : ℝ
  c : ℝ
  -- The line passes through (3, -1)
  passes_through : a * 3 + b * (-1) + c = 0
  -- The line has equal absolute values of intercepts on both axes
  equal_intercepts : |a / b| = |b / a| ∨ (a = 0 ∧ b ≠ 0) ∨ (b = 0 ∧ a ≠ 0)

/-- The possible equations for the special line -/
def possible_equations (l : SpecialLine) : Prop :=
  (l.a = 1 ∧ l.b = 1 ∧ l.c = -2) ∨
  (l.a = 1 ∧ l.b = -1 ∧ l.c = -4) ∨
  (l.a = 3 ∧ l.b = 1 ∧ l.c = 0)

/-- Theorem stating that any SpecialLine must have one of the possible equations -/
theorem special_line_equation (l : SpecialLine) : possible_equations l := by
  sorry

end special_line_equation_l1973_197386


namespace limit_at_neg_three_is_zero_l1973_197316

/-- The limit of (x^2 + 2x - 3)^2 / (x^3 + 4x^2 + 3x) as x approaches -3 is 0 -/
theorem limit_at_neg_three_is_zero :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 
    0 < |x + 3| ∧ |x + 3| < δ → 
    |(x^2 + 2*x - 3)^2 / (x^3 + 4*x^2 + 3*x) - 0| < ε :=
by
  sorry

#check limit_at_neg_three_is_zero

end limit_at_neg_three_is_zero_l1973_197316


namespace three_planes_intersection_count_l1973_197329

structure Plane

/-- Three planes that intersect pairwise -/
structure ThreePlanesIntersectingPairwise where
  plane1 : Plane
  plane2 : Plane
  plane3 : Plane
  intersect12 : plane1 ≠ plane2
  intersect23 : plane2 ≠ plane3
  intersect13 : plane1 ≠ plane3

/-- A line of intersection between two planes -/
def LineOfIntersection (p1 p2 : Plane) : Type := Unit

/-- Count the number of distinct lines of intersection -/
def CountLinesOfIntersection (t : ThreePlanesIntersectingPairwise) : Nat :=
  sorry

theorem three_planes_intersection_count
  (t : ThreePlanesIntersectingPairwise) :
  CountLinesOfIntersection t = 1 ∨ CountLinesOfIntersection t = 3 :=
sorry

end three_planes_intersection_count_l1973_197329


namespace max_triplets_coordinate_plane_l1973_197351

/-- Given 100 points on a coordinate plane, prove that the maximum number of triplets (A, B, C) 
    where A and B have the same y-coordinate and B and C have the same x-coordinate is 8100. -/
theorem max_triplets_coordinate_plane (points : Finset (ℝ × ℝ)) 
    (h : points.card = 100) : 
  (Finset.sum points (fun B => 
    (points.filter (fun A => A.2 = B.2)).card * 
    (points.filter (fun C => C.1 = B.1)).card
  )) ≤ 8100 := by
  sorry

end max_triplets_coordinate_plane_l1973_197351


namespace negation_of_proposition_l1973_197358

theorem negation_of_proposition (n : ℕ) :
  ¬(2^n > 1000) ↔ (2^n ≤ 1000) := by sorry

end negation_of_proposition_l1973_197358


namespace garden_fencing_length_l1973_197305

theorem garden_fencing_length (garden_area : ℝ) (π_approx : ℝ) (extra_length : ℝ) : 
  garden_area = 616 → 
  π_approx = 22 / 7 → 
  extra_length = 5 → 
  2 * π_approx * Real.sqrt (garden_area / π_approx) + extra_length = 93 := by
sorry

end garden_fencing_length_l1973_197305


namespace flight_duration_sum_l1973_197307

def flight_duration (departure_hour : Nat) (departure_minute : Nat)
                    (arrival_hour : Nat) (arrival_minute : Nat) : Nat :=
  (arrival_hour * 60 + arrival_minute) - (departure_hour * 60 + departure_minute)

theorem flight_duration_sum (h m : Nat) :
  flight_duration 15 42 18 57 = h * 60 + m →
  0 < m →
  m < 60 →
  h + m = 18 := by
  sorry

end flight_duration_sum_l1973_197307


namespace water_polo_team_selection_l1973_197308

def team_size : ℕ := 15
def starting_players : ℕ := 7
def coach_count : ℕ := 1

theorem water_polo_team_selection :
  (team_size * (team_size - 1) * (Nat.choose (team_size - 2) (starting_players - 2))) = 270270 := by
  sorry

end water_polo_team_selection_l1973_197308


namespace fermat_sum_divisibility_l1973_197360

theorem fermat_sum_divisibility (x y z : ℤ) 
  (hx : ¬ 7 ∣ x) (hy : ¬ 7 ∣ y) (hz : ¬ 7 ∣ z)
  (h_sum : (7:ℤ)^3 ∣ (x^7 + y^7 + z^7)) :
  (7:ℤ)^2 ∣ (x + y + z) := by
  sorry

end fermat_sum_divisibility_l1973_197360


namespace decimal_to_fraction_sum_l1973_197339

theorem decimal_to_fraction_sum (x : ℚ) (n d : ℕ) (v : ℕ) : 
  x = 2.52 →
  x = n / d →
  (∀ k : ℕ, k > 1 → ¬(k ∣ n ∧ k ∣ d)) →
  n + v = 349 →
  v = 286 := by sorry

end decimal_to_fraction_sum_l1973_197339


namespace geoffrey_initial_wallet_l1973_197355

/-- The amount of money Geoffrey had initially in his wallet --/
def initial_wallet_amount : ℕ := 50

/-- The amount Geoffrey received from his grandmother --/
def grandmother_gift : ℕ := 20

/-- The amount Geoffrey received from his aunt --/
def aunt_gift : ℕ := 25

/-- The amount Geoffrey received from his uncle --/
def uncle_gift : ℕ := 30

/-- The cost of each game --/
def game_cost : ℕ := 35

/-- The number of games Geoffrey bought --/
def num_games : ℕ := 3

/-- The amount left after the purchase --/
def amount_left : ℕ := 20

theorem geoffrey_initial_wallet :
  initial_wallet_amount = 
    (amount_left + num_games * game_cost) - (grandmother_gift + aunt_gift + uncle_gift) :=
by sorry

end geoffrey_initial_wallet_l1973_197355


namespace website_earnings_per_visit_l1973_197387

/-- Calculates the earnings per visit for a website -/
def earnings_per_visit (monthly_visits : ℕ) (daily_earnings : ℚ) : ℚ :=
  (30 * daily_earnings) / monthly_visits

/-- Theorem: Given 30,000 monthly visits and $10 daily earnings, the earnings per visit is $0.01 -/
theorem website_earnings_per_visit : 
  earnings_per_visit 30000 10 = 1/100 := by
  sorry

end website_earnings_per_visit_l1973_197387


namespace second_train_speed_l1973_197366

/-- Proves that the speed of the second train is 80 kmph given the conditions of the problem -/
theorem second_train_speed (first_train_speed : ℝ) (time_difference : ℝ) (meeting_distance : ℝ) :
  first_train_speed = 40 →
  time_difference = 1 →
  meeting_distance = 80 →
  (meeting_distance - first_train_speed * time_difference) / time_difference = 80 :=
by
  sorry

#check second_train_speed

end second_train_speed_l1973_197366


namespace pencil_count_l1973_197371

/-- The total number of colored pencils Cheryl, Cyrus, and Madeline have -/
def total_pencils (cheryl : ℕ) (cyrus : ℕ) (madeline : ℕ) : ℕ :=
  cheryl + cyrus + madeline

/-- Theorem stating the total number of colored pencils given the conditions -/
theorem pencil_count :
  ∀ (cheryl cyrus madeline : ℕ),
    cheryl = 3 * cyrus →
    madeline = 63 →
    cheryl = 2 * madeline →
    total_pencils cheryl cyrus madeline = 231 :=
by
  sorry

end pencil_count_l1973_197371


namespace power_of_product_with_exponent_l1973_197346

theorem power_of_product_with_exponent (x y : ℝ) : (-x * y^3)^2 = x^2 * y^6 := by
  sorry

end power_of_product_with_exponent_l1973_197346


namespace expected_weekly_rainfall_l1973_197349

/-- Represents the possible weather outcomes for a day -/
inductive Weather
  | Sun
  | Rain2Inches
  | Rain8Inches

/-- The probability of each weather outcome -/
def weather_probability : Weather → ℝ
  | Weather.Sun => 0.35
  | Weather.Rain2Inches => 0.40
  | Weather.Rain8Inches => 0.25

/-- The amount of rainfall for each weather outcome -/
def rainfall_amount : Weather → ℝ
  | Weather.Sun => 0
  | Weather.Rain2Inches => 2
  | Weather.Rain8Inches => 8

/-- The number of days in the week -/
def days_in_week : ℕ := 7

/-- The expected rainfall for a single day -/
def expected_daily_rainfall : ℝ :=
  (weather_probability Weather.Sun * rainfall_amount Weather.Sun) +
  (weather_probability Weather.Rain2Inches * rainfall_amount Weather.Rain2Inches) +
  (weather_probability Weather.Rain8Inches * rainfall_amount Weather.Rain8Inches)

/-- Theorem: The expected total rainfall for the week is 19.6 inches -/
theorem expected_weekly_rainfall :
  (days_in_week : ℝ) * expected_daily_rainfall = 19.6 := by
  sorry

end expected_weekly_rainfall_l1973_197349


namespace only_negative_number_l1973_197369

theorem only_negative_number (a b c d : ℝ) : 
  a = 2023 → b = -2023 → c = 1 / 2023 → d = 0 →
  (b < 0 ∧ a > 0 ∧ c > 0 ∧ d = 0) := by
  sorry

end only_negative_number_l1973_197369


namespace solution_satisfies_equation_l1973_197327

def F (a b c d : ℕ) : ℕ := a^b + c * d

theorem solution_satisfies_equation : F 3 6 5 15 = 490 := by sorry

end solution_satisfies_equation_l1973_197327


namespace gcd_lcm_product_48_75_l1973_197314

theorem gcd_lcm_product_48_75 : Nat.gcd 48 75 * Nat.lcm 48 75 = 3600 := by
  sorry

end gcd_lcm_product_48_75_l1973_197314


namespace ab_equality_l1973_197315

theorem ab_equality (a b : ℚ) (h1 : 10 * a = 20) (h2 : 6 * b = 20) : 120 * a * b = 800 := by
  sorry

end ab_equality_l1973_197315


namespace arithmetic_equality_l1973_197353

theorem arithmetic_equality : 4 * 8 + 5 * 11 - 2 * 3 + 7 * 9 = 144 := by
  sorry

end arithmetic_equality_l1973_197353


namespace cost_price_calculation_l1973_197323

theorem cost_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) 
  (h1 : selling_price = 240)
  (h2 : profit_percentage = 0.25) : 
  ∃ (cost_price : ℝ), cost_price = 192 ∧ selling_price = cost_price * (1 + profit_percentage) :=
by
  sorry

end cost_price_calculation_l1973_197323


namespace least_xy_value_l1973_197331

theorem least_xy_value (x y : ℕ+) (h : (1 : ℚ) / x + (1 : ℚ) / (3 * y) = (1 : ℚ) / 6) :
  (∀ a b : ℕ+, (1 : ℚ) / a + (1 : ℚ) / (3 * b) = (1 : ℚ) / 6 → x * y ≤ a * b) ∧ x * y = 64 := by
  sorry

end least_xy_value_l1973_197331


namespace range_of_a_l1973_197377

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 - 2*x + 2
  else x + a/x + 3*a

theorem range_of_a (a : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, f a x = y) ↔ a ∈ Set.Iio 0 ∪ Set.Ici 1 :=
sorry

end range_of_a_l1973_197377


namespace point_coordinates_l1973_197354

theorem point_coordinates (a b : ℝ) (ha : 0 < a) (ha' : a < 1) (hb : 1 < b) (hb' : b < 2) :
  (0 < a/2 ∧ a/2 < 1/2 ∧ 2 < b+1 ∧ b+1 < 3) ∧
  (-1 < a-1 ∧ a-1 < 0 ∧ 0 < b/2 ∧ b/2 < 1) ∧
  (-1 < -a ∧ -a < 0 ∧ -2 < -b ∧ -b < -1) ∧
  (0 < 1-a ∧ 1-a < 1 ∧ 0 < b-1 ∧ b-1 < 1) := by
  sorry

end point_coordinates_l1973_197354


namespace three_propositions_imply_l1973_197348

theorem three_propositions_imply (p q r : Prop) : 
  (((p ∨ (¬q ∧ r)) → ((p → q) → r)) ∧
   ((¬p ∨ (¬q ∧ r)) → ((p → q) → r)) ∧
   ((p ∨ (¬q ∧ ¬r)) → ¬((p → q) → r)) ∧
   ((¬p ∨ (q ∧ r)) → ((p → q) → r))) := by
  sorry

end three_propositions_imply_l1973_197348


namespace sunday_calorie_intake_l1973_197304

-- Define the calorie content for base meals
def breakfast_calories : ℝ := 500
def lunch_calories : ℝ := breakfast_calories * 1.25
def dinner_calories : ℝ := lunch_calories * 2
def snack_calories : ℝ := lunch_calories * 0.7
def morning_snack_calories : ℝ := breakfast_calories + 200
def afternoon_snack_calories : ℝ := lunch_calories * 0.8
def dessert_calories : ℝ := 350
def energy_drink_calories : ℝ := 220

-- Define the total calories for each day
def monday_calories : ℝ := breakfast_calories + lunch_calories + dinner_calories + snack_calories
def tuesday_calories : ℝ := breakfast_calories + morning_snack_calories + lunch_calories + afternoon_snack_calories + dinner_calories
def wednesday_calories : ℝ := breakfast_calories + lunch_calories + (dinner_calories * 0.85) + dessert_calories
def thursday_calories : ℝ := tuesday_calories
def friday_calories : ℝ := wednesday_calories + (2 * energy_drink_calories)
def weekend_calories : ℝ := tuesday_calories

-- Theorem to prove
theorem sunday_calorie_intake : weekend_calories = 3575 := by
  sorry

end sunday_calorie_intake_l1973_197304


namespace max_gcd_of_sum_1089_l1973_197324

theorem max_gcd_of_sum_1089 (c d : ℕ+) (h : c + d = 1089) :
  (∃ (x y : ℕ+), x + y = 1089 ∧ Nat.gcd x y = 363) ∧
  (∀ (a b : ℕ+), a + b = 1089 → Nat.gcd a b ≤ 363) := by
  sorry

end max_gcd_of_sum_1089_l1973_197324


namespace complex_fraction_equation_solution_l1973_197342

theorem complex_fraction_equation_solution :
  ∃ x : ℚ, 3 + 1 / (2 + 1 / (3 + 3 / (4 + x))) = 225/68 ∧ x = -50/19 := by
  sorry

end complex_fraction_equation_solution_l1973_197342


namespace steve_berry_picking_earnings_l1973_197385

/-- The amount of money earned per pound of lingonberries -/
def price_per_pound : ℕ := 2

/-- The amount of lingonberries picked on Monday -/
def monday_picking : ℕ := 8

/-- The amount of lingonberries picked on Tuesday -/
def tuesday_picking : ℕ := 3 * monday_picking

/-- The amount of lingonberries picked on Wednesday -/
def wednesday_picking : ℕ := 0

/-- The amount of lingonberries picked on Thursday -/
def thursday_picking : ℕ := 18

/-- The total money Steve wanted to make -/
def total_money : ℕ := 100

/-- Theorem stating that the total money Steve wanted to make is correct -/
theorem steve_berry_picking_earnings :
  (monday_picking + tuesday_picking + wednesday_picking + thursday_picking) * price_per_pound = total_money := by
  sorry

end steve_berry_picking_earnings_l1973_197385


namespace perpendicular_line_equation_l1973_197390

/-- Given two lines L1 and L2 in a 2D plane, where:
    - L1 has equation mx - m²y = 1
    - L2 is perpendicular to L1
    - L1 and L2 intersect at point P(2,1)
    Prove that the equation of L2 is x + y - 3 = 0 -/
theorem perpendicular_line_equation (m : ℝ) :
  (∀ x y, m * x - m^2 * y = 1 → x = 2 ∧ y = 1) →
  (∃ k : ℝ, k * m = -1) →
  ∀ x y, x + y - 3 = 0 :=
sorry

end perpendicular_line_equation_l1973_197390


namespace fifteenth_recalibration_in_march_l1973_197333

/-- Calculates the month of the nth recalibration given a start month and recalibration interval -/
def recalibrationMonth (startMonth : Nat) (interval : Nat) (n : Nat) : Nat :=
  ((startMonth - 1) + (n - 1) * interval) % 12 + 1

/-- The month of the 15th recalibration is March (month 3) -/
theorem fifteenth_recalibration_in_march :
  recalibrationMonth 1 7 15 = 3 := by
  sorry

#eval recalibrationMonth 1 7 15

end fifteenth_recalibration_in_march_l1973_197333


namespace intersection_length_circle_line_l1973_197365

/-- The intersection length of a circle and a line --/
theorem intersection_length_circle_line : 
  ∃ (A B : ℝ × ℝ),
    (A.1^2 + (A.2 - 1)^2 = 1) ∧ 
    (B.1^2 + (B.2 - 1)^2 = 1) ∧
    (A.1 - A.2 + 2 = 0) ∧ 
    (B.1 - B.2 + 2 = 0) ∧
    ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 2) :=
by sorry

end intersection_length_circle_line_l1973_197365


namespace ellipse_parallelogram_area_l1973_197364

-- Define the ellipse
def on_ellipse (x y : ℝ) : Prop := x^2 + 2*y^2 = 1

-- Define the slope product condition
def slope_product (x1 y1 x2 y2 : ℝ) : Prop := (y1/x1) * (y2/x2) = -1/2

-- Define the area of the parallelogram
def parallelogram_area (x1 y1 x2 y2 : ℝ) : ℝ := 2 * |x1*y2 - x2*y1|

-- Theorem statement
theorem ellipse_parallelogram_area 
  (x1 y1 x2 y2 : ℝ) 
  (h1 : on_ellipse x1 y1) 
  (h2 : on_ellipse x2 y2) 
  (h3 : slope_product x1 y1 x2 y2) : 
  parallelogram_area x1 y1 x2 y2 = Real.sqrt 2 := by
  sorry

end ellipse_parallelogram_area_l1973_197364


namespace distance_to_reflection_distance_z_to_z_reflected_l1973_197341

/-- The distance between a point and its reflection over the x-axis --/
theorem distance_to_reflection (x y : ℝ) : 
  let z : ℝ × ℝ := (x, y)
  let z_reflected : ℝ × ℝ := (x, -y)
  Real.sqrt ((z.1 - z_reflected.1)^2 + (z.2 - z_reflected.2)^2) = 2 * |y| :=
by sorry

/-- The specific case for the point Z(5, 2) --/
theorem distance_z_to_z_reflected : 
  let z : ℝ × ℝ := (5, 2)
  let z_reflected : ℝ × ℝ := (5, -2)
  Real.sqrt ((z.1 - z_reflected.1)^2 + (z.2 - z_reflected.2)^2) = 4 :=
by sorry

end distance_to_reflection_distance_z_to_z_reflected_l1973_197341


namespace divisibility_by_48_l1973_197392

theorem divisibility_by_48 (a b c : ℕ) 
  (ha : Nat.Prime a) (hb : Nat.Prime b) (hc : Nat.Prime c)
  (ga : a > 3) (gb : b > 3) (gc : c > 3) : 
  48 ∣ ((a - b) * (b - c) * (c - a)) := by
  sorry

end divisibility_by_48_l1973_197392


namespace negation_of_universal_proposition_negation_of_specific_proposition_l1973_197373

theorem negation_of_universal_proposition (P : ℝ → Prop) :
  (¬ ∀ x : ℝ, P x) ↔ (∃ x : ℝ, ¬ P x) :=
by sorry

theorem negation_of_specific_proposition :
  (¬ ∀ x : ℝ, x^2 + 2*x + 3 ≥ 0) ↔ (∃ x : ℝ, x^2 + 2*x + 3 < 0) :=
by sorry

end negation_of_universal_proposition_negation_of_specific_proposition_l1973_197373


namespace martin_answered_40_l1973_197301

/-- The number of questions Campbell answered correctly -/
def campbell_correct : ℕ := 35

/-- The number of questions Kelsey answered correctly -/
def kelsey_correct : ℕ := campbell_correct + 8

/-- The number of questions Martin answered correctly -/
def martin_correct : ℕ := kelsey_correct - 3

/-- Theorem stating that Martin answered 40 questions correctly -/
theorem martin_answered_40 : martin_correct = 40 := by
  sorry

end martin_answered_40_l1973_197301


namespace johns_total_earnings_l1973_197345

/-- Calculates the total earnings given the initial bonus, growth rate, and new salary -/
def total_earnings (initial_bonus : ℝ) (growth_rate : ℝ) (new_salary : ℝ) : ℝ :=
  let new_bonus := initial_bonus * (1 + growth_rate)
  new_salary + new_bonus

/-- Theorem: John's total earnings this year are $210,500 -/
theorem johns_total_earnings :
  let initial_bonus : ℝ := 10000
  let growth_rate : ℝ := 0.05
  let new_salary : ℝ := 200000
  total_earnings initial_bonus growth_rate new_salary = 210500 := by
  sorry

#eval total_earnings 10000 0.05 200000

end johns_total_earnings_l1973_197345


namespace vector_properties_l1973_197374

def e₁ : ℝ × ℝ := (1, 0)
def e₂ : ℝ × ℝ := (0, 1)
def a : ℝ × ℝ := (3 * e₁.1 - 2 * e₂.1, 3 * e₁.2 - 2 * e₂.2)
def b : ℝ × ℝ := (4 * e₁.1 + e₂.1, 4 * e₁.2 + e₂.2)

theorem vector_properties :
  (a.1 * b.1 + a.2 * b.2 = 10) ∧
  (Real.sqrt ((a.1 + b.1)^2 + (a.2 + b.2)^2) = 5 * Real.sqrt 2) ∧
  (((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))) = 10 * Real.sqrt 221 / 221) := by
  sorry

end vector_properties_l1973_197374


namespace smallest_value_between_one_and_two_l1973_197325

theorem smallest_value_between_one_and_two (y : ℝ) (h1 : 1 < y) (h2 : y < 2) :
  (1 / y < y) ∧ (1 / y < y^2) ∧ (1 / y < 2*y) ∧ (1 / y < Real.sqrt y) := by
  sorry

end smallest_value_between_one_and_two_l1973_197325


namespace triathlon_speed_l1973_197356

/-- Triathlon problem -/
theorem triathlon_speed (total_time : ℝ) (swim_dist swim_speed : ℝ) (run_dist run_speed : ℝ) (bike_dist : ℝ) :
  total_time = 2 →
  swim_dist = 0.5 →
  swim_speed = 3 →
  run_dist = 4 →
  run_speed = 8 →
  bike_dist = 20 →
  (swim_dist / swim_speed + run_dist / run_speed + bike_dist / (bike_dist / (total_time - (swim_dist / swim_speed + run_dist / run_speed))) = total_time) →
  bike_dist / (total_time - (swim_dist / swim_speed + run_dist / run_speed)) = 15 := by
sorry


end triathlon_speed_l1973_197356


namespace replacement_concentration_theorem_l1973_197303

/-- Given an initial solution concentration, a replacement solution concentration,
    and the fraction of solution replaced, calculate the new concentration. -/
def new_concentration (initial_conc : ℝ) (replacement_conc : ℝ) (fraction_replaced : ℝ) : ℝ :=
  (initial_conc * (1 - fraction_replaced) + replacement_conc * fraction_replaced)

/-- Theorem stating that replacing half of a 45% solution with a 25% solution
    results in a 35% solution. -/
theorem replacement_concentration_theorem :
  new_concentration 0.45 0.25 0.5 = 0.35 := by
  sorry

#eval new_concentration 0.45 0.25 0.5

end replacement_concentration_theorem_l1973_197303


namespace village_population_problem_l1973_197318

theorem village_population_problem (final_population : ℕ) 
  (h1 : final_population = 3168) : ∃ initial_population : ℕ,
  (initial_population : ℝ) * 0.9 * 0.8 = final_population ∧ 
  initial_population = 4400 := by
  sorry

end village_population_problem_l1973_197318


namespace fourth_grade_students_l1973_197372

/-- The number of students in fourth grade at the start of the year. -/
def initial_students : ℕ := 33

/-- The number of students who left during the year. -/
def students_left : ℕ := 18

/-- The number of new students who came during the year. -/
def new_students : ℕ := 14

/-- The number of students at the end of the year. -/
def final_students : ℕ := 29

theorem fourth_grade_students : 
  initial_students - students_left + new_students = final_students := by
  sorry

end fourth_grade_students_l1973_197372


namespace french_fries_cost_is_ten_l1973_197334

/-- Represents the cost of a meal at Wendy's -/
structure WendysMeal where
  taco_salad : ℕ
  hamburgers : ℕ
  lemonade : ℕ
  friends : ℕ
  individual_payment : ℕ

/-- Calculates the total cost of french fries in a Wendy's meal -/
def french_fries_cost (meal : WendysMeal) : ℕ :=
  meal.friends * meal.individual_payment -
  (meal.taco_salad + 5 * meal.hamburgers + 5 * meal.lemonade)

/-- Theorem stating that the total cost of french fries is $10 -/
theorem french_fries_cost_is_ten (meal : WendysMeal)
  (h1 : meal.taco_salad = 10)
  (h2 : meal.hamburgers = 5)
  (h3 : meal.lemonade = 2)
  (h4 : meal.friends = 5)
  (h5 : meal.individual_payment = 11) :
  french_fries_cost meal = 10 := by
  sorry

#eval french_fries_cost { taco_salad := 10, hamburgers := 5, lemonade := 2, friends := 5, individual_payment := 11 }

end french_fries_cost_is_ten_l1973_197334


namespace shaded_area_octagon_with_sectors_l1973_197388

/-- The area of the shaded region in a regular octagon with circular sectors --/
theorem shaded_area_octagon_with_sectors (side_length : Real) (sector_radius : Real) : 
  side_length = 5 → sector_radius = 3 → 
  ∃ (shaded_area : Real), shaded_area = 100 - 9 * Real.pi := by
  sorry

end shaded_area_octagon_with_sectors_l1973_197388


namespace regression_line_y_change_l1973_197347

/-- Represents a linear regression equation of the form ŷ = a + bx -/
structure RegressionLine where
  a : ℝ
  b : ℝ

/-- The change in y when x increases by 1 unit -/
def yChange (line : RegressionLine) : ℝ := line.b

theorem regression_line_y_change 
  (line : RegressionLine) 
  (h : line = { a := 2, b := -1.5 }) : 
  yChange line = -1.5 := by
  sorry

end regression_line_y_change_l1973_197347


namespace johns_notebooks_l1973_197310

theorem johns_notebooks (total_children : Nat) (wife_notebooks_per_child : Nat) (total_notebooks : Nat) :
  total_children = 3 →
  wife_notebooks_per_child = 5 →
  total_notebooks = 21 →
  ∃ (johns_notebooks_per_child : Nat),
    johns_notebooks_per_child * total_children + wife_notebooks_per_child * total_children = total_notebooks ∧
    johns_notebooks_per_child = 2 :=
by
  sorry

end johns_notebooks_l1973_197310


namespace ship_passengers_l1973_197332

theorem ship_passengers : ∃ (P : ℕ), 
  P > 0 ∧ 
  (P : ℚ) * (1/3 + 1/8 + 1/5 + 1/6) + 42 = P ∧ 
  P = 240 := by
  sorry

end ship_passengers_l1973_197332


namespace imaginary_part_of_z_plus_reciprocal_l1973_197396

def z : ℂ := 1 + Complex.I

theorem imaginary_part_of_z_plus_reciprocal (z : ℂ) (h : z = 1 + Complex.I) :
  Complex.im (z + z⁻¹) = -1/2 := by
  sorry

end imaginary_part_of_z_plus_reciprocal_l1973_197396


namespace rectangular_garden_width_l1973_197357

/-- A rectangular garden with length three times its width and area 507 square meters has a width of 13 meters. -/
theorem rectangular_garden_width (width length area : ℝ) : 
  length = 3 * width →
  area = length * width →
  area = 507 →
  width = 13 := by
sorry

end rectangular_garden_width_l1973_197357


namespace equation_solution_l1973_197391

theorem equation_solution : ∃ x : ℝ, 
  Real.sqrt (9 + Real.sqrt (27 + 9*x)) + Real.sqrt (3 + Real.sqrt (3 + x)) = 3 + 3 * Real.sqrt 3 :=
by
  use 9
  sorry

end equation_solution_l1973_197391


namespace exists_solution_l1973_197352

open Complex

/-- The equation that z must satisfy -/
def equation (z : ℂ) : Prop :=
  z * (z + I) * (z - 2 + I) * (z + 3*I) = 2018 * I

/-- The condition that b should be maximized -/
def b_maximized (z : ℂ) : Prop :=
  ∀ w : ℂ, equation w → z.im ≥ w.im

/-- The main theorem stating the existence of z satisfying the conditions -/
theorem exists_solution :
  ∃ z : ℂ, equation z ∧ b_maximized z :=
sorry

/-- Helper lemma to extract the real part of the solution -/
lemma solution_real_part (z : ℂ) (h : equation z ∧ b_maximized z) :
  ∃ a : ℝ, z.re = a :=
sorry

end exists_solution_l1973_197352


namespace complement_union_complement_equals_intersection_l1973_197389

theorem complement_union_complement_equals_intersection (P Q : Set α) :
  (Pᶜᶜ ∪ Qᶜ)ᶜ = P ∩ Q := by
  sorry

end complement_union_complement_equals_intersection_l1973_197389


namespace part_one_part_two_l1973_197375

-- Define propositions P and Q
def P (a : ℝ) : Prop := ∀ x ∈ (Set.Icc 1 2), x^2 - a ≥ 0
def Q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

-- Theorem for part (1)
theorem part_one (a : ℝ) : P a → a ≤ 1 := by sorry

-- Theorem for part (2)
theorem part_two (a : ℝ) : (P a ∨ Q a) ∧ ¬(P a ∧ Q a) → a > 1 ∨ (-2 < a ∧ a < 1) := by sorry

end part_one_part_two_l1973_197375


namespace fifteen_valid_pairs_l1973_197343

/-- A function that constructs the number 7ABABA from single digits A and B -/
def constructNumber (A B : Nat) : Nat :=
  700000 + 10000 * A + 1000 * B + 100 * A + 10 * B + A

/-- Predicate to check if a number is a single digit -/
def isSingleDigit (n : Nat) : Prop := n < 10

/-- The main theorem stating that there are exactly 15 valid pairs (A, B) -/
theorem fifteen_valid_pairs :
  ∃! (validPairs : Finset (Nat × Nat)),
    validPairs.card = 15 ∧
    ∀ (A B : Nat),
      (A, B) ∈ validPairs ↔
        isSingleDigit A ∧
        isSingleDigit B ∧
        (constructNumber A B % 6 = 0) :=
  sorry

end fifteen_valid_pairs_l1973_197343


namespace triangle_third_side_valid_third_side_l1973_197379

/-- Checks if three lengths can form a valid triangle -/
def is_valid_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_third_side (x : ℝ) : 
  (is_valid_triangle 7 10 x ∧ x > 0) ↔ (3 < x ∧ x < 17) :=
sorry

theorem valid_third_side : 
  is_valid_triangle 7 10 11 ∧ 
  ¬(is_valid_triangle 7 10 20) ∧ 
  ¬(is_valid_triangle 7 10 3) ∧ 
  ¬(is_valid_triangle 7 10 2) :=
sorry

end triangle_third_side_valid_third_side_l1973_197379


namespace distance_difference_l1973_197350

def time : ℝ := 6
def carlos_distance : ℝ := 108
def daniel_distance : ℝ := 90

theorem distance_difference : carlos_distance - daniel_distance = 18 := by
  sorry

end distance_difference_l1973_197350


namespace son_age_problem_l1973_197394

theorem son_age_problem (father_age son_age : ℕ) : 
  father_age = son_age + 26 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 24 := by
sorry

end son_age_problem_l1973_197394


namespace v_equation_l1973_197376

/-- Given that V = kZ - 6 and V = 14 when Z = 5, prove that V = 22 when Z = 7 -/
theorem v_equation (k : ℝ) : 
  (∀ Z, (k * Z - 6 = 14) → (Z = 5)) →
  (k * 7 - 6 = 22) :=
by sorry

end v_equation_l1973_197376


namespace inscribed_cube_properties_l1973_197397

/-- Given a cube with surface area 54 square meters, containing an inscribed sphere 
    which in turn contains an inscribed smaller cube, prove the surface area and volume 
    of the inner cube. -/
theorem inscribed_cube_properties (outer_cube : Real) (sphere : Real) (inner_cube : Real) :
  (6 * outer_cube ^ 2 = 54) →
  (sphere = outer_cube / 2) →
  (inner_cube * Real.sqrt 3 = outer_cube) →
  (6 * inner_cube ^ 2 = 18 ∧ inner_cube ^ 3 = 3 * Real.sqrt 3) := by
  sorry

#check inscribed_cube_properties

end inscribed_cube_properties_l1973_197397


namespace choir_arrangements_choir_arrangement_count_l1973_197340

theorem choir_arrangements (total_boys : Nat) (total_girls : Nat) 
  (selected_boys : Nat) (selected_girls : Nat) : Nat :=
  let boy_selections := Nat.choose total_boys selected_boys
  let girl_selections := Nat.choose total_girls selected_girls
  let boy_arrangements := Nat.factorial selected_boys
  let girl_positions := Nat.factorial (selected_boys + 1) / Nat.factorial (selected_boys + 1 - selected_girls)
  boy_selections * girl_selections * boy_arrangements * girl_positions

theorem choir_arrangement_count : 
  choir_arrangements 4 3 2 2 = 216 := by sorry

end choir_arrangements_choir_arrangement_count_l1973_197340


namespace interest_rate_calculation_l1973_197321

/-- Simple interest calculation -/
def simple_interest (principal rate time : ℚ) : ℚ :=
  principal * rate * time / 100

/-- Problem statement -/
theorem interest_rate_calculation (principal interest time : ℚ) 
  (h_principal : principal = 800)
  (h_interest : interest = 160)
  (h_time : time = 4)
  (h_simple_interest : simple_interest principal (5 : ℚ) time = interest) :
  simple_interest principal (5 : ℚ) time = interest :=
by sorry

end interest_rate_calculation_l1973_197321


namespace fourth_term_is_one_tenth_l1973_197395

def sequence_term (n : ℕ) : ℚ := 2 / (n^2 + n : ℚ)

theorem fourth_term_is_one_tenth : sequence_term 4 = 1/10 := by
  sorry

end fourth_term_is_one_tenth_l1973_197395


namespace line_equation_l1973_197328

/-- A line passing through a point and intersecting a circle with a given chord length -/
def intersecting_line (P : ℝ × ℝ) (center : ℝ × ℝ) (radius : ℝ) (chord_length : ℝ) :=
  {l : Set (ℝ × ℝ) | ∃ (A B : ℝ × ℝ),
    A ∈ l ∧ B ∈ l ∧
    P ∈ l ∧
    (A.1 - center.1)^2 + (A.2 - center.2)^2 = radius^2 ∧
    (B.1 - center.1)^2 + (B.2 - center.2)^2 = radius^2 ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = chord_length^2}

theorem line_equation (P : ℝ × ℝ) (center : ℝ × ℝ) (radius chord_length : ℝ)
  (h1 : P = (3, 6))
  (h2 : center = (0, 0))
  (h3 : radius = 5)
  (h4 : chord_length = 8) :
  ∀ l ∈ intersecting_line P center radius chord_length,
  (∀ (x y : ℝ), (x, y) ∈ l ↔ (x - 3 = 0 ∨ 3*x - 4*y + 15 = 0)) :=
by sorry

end line_equation_l1973_197328


namespace badminton_players_count_l1973_197326

/-- Represents a sports club with members playing badminton and tennis -/
structure SportsClub where
  total_members : ℕ
  badminton_players : ℕ
  tennis_players : ℕ
  both_players : ℕ
  neither_players : ℕ

/-- Theorem stating the number of badminton players in the given conditions -/
theorem badminton_players_count (club : SportsClub)
  (h1 : club.total_members = 30)
  (h2 : club.badminton_players = club.tennis_players)
  (h3 : club.neither_players = 2)
  (h4 : club.both_players = 6)
  (h5 : club.total_members = club.badminton_players + club.tennis_players - club.both_players + club.neither_players) :
  club.badminton_players = 17 := by
  sorry


end badminton_players_count_l1973_197326


namespace total_employee_costs_february_l1973_197338

/-- Represents an employee in the car dealership -/
structure Employee where
  name : String
  hoursPerWeek : Nat
  hourlyRate : Nat
  weeksWorked : Nat
  overtime : Nat
  overtimeRate : Nat
  bonus : Int
  deduction : Nat

/-- Calculates the monthly earnings for an employee -/
def monthlyEarnings (e : Employee) : Int :=
  e.hoursPerWeek * e.hourlyRate * e.weeksWorked +
  e.overtime * e.overtimeRate +
  e.bonus -
  e.deduction

/-- Theorem stating the total employee costs for February -/
theorem total_employee_costs_february :
  let fiona : Employee := ⟨"Fiona", 40, 20, 3, 0, 0, 0, 0⟩
  let john : Employee := ⟨"John", 30, 22, 4, 10, 33, 0, 0⟩
  let jeremy : Employee := ⟨"Jeremy", 25, 18, 4, 0, 0, 200, 0⟩
  let katie : Employee := ⟨"Katie", 35, 21, 4, 0, 0, 0, 150⟩
  let matt : Employee := ⟨"Matt", 28, 19, 4, 0, 0, 0, 0⟩
  monthlyEarnings fiona + monthlyEarnings john + monthlyEarnings jeremy +
  monthlyEarnings katie + monthlyEarnings matt = 13278 := by
  sorry


end total_employee_costs_february_l1973_197338


namespace log_product_identity_l1973_197382

theorem log_product_identity (b c : ℝ) (hb_pos : b > 0) (hc_pos : c > 0) (hb_ne_one : b ≠ 1) (hc_ne_one : c ≠ 1) :
  Real.log b / Real.log (2 * c) * Real.log (2 * c) / Real.log b = 1 := by
  sorry

end log_product_identity_l1973_197382


namespace power_difference_l1973_197300

theorem power_difference (a m n : ℝ) (h1 : a^m = 8) (h2 : a^n = 2) : a^(m-3*n) = 1 := by
  sorry

end power_difference_l1973_197300


namespace max_profit_computer_sales_profit_per_computer_type_l1973_197312

/-- Profit function for computer sales -/
def profit_function (m : ℕ) : ℝ := -50 * m + 15000

/-- Constraint on the number of type B computers -/
def type_b_constraint (m : ℕ) : Prop := 100 - m ≤ 2 * m

/-- Theorem stating the maximum profit and optimal purchase strategy -/
theorem max_profit_computer_sales :
  ∃ (m : ℕ),
    m = 34 ∧
    type_b_constraint m ∧
    profit_function m = 13300 ∧
    ∀ (n : ℕ), type_b_constraint n → profit_function n ≤ profit_function m :=
by
  sorry

/-- Theorem verifying the profit for each computer type -/
theorem profit_per_computer_type :
  ∃ (a b : ℝ),
    a = 100 ∧
    b = 150 ∧
    10 * a + 20 * b = 4000 ∧
    20 * a + 10 * b = 3500 :=
by
  sorry

end max_profit_computer_sales_profit_per_computer_type_l1973_197312


namespace lucas_payment_l1973_197306

/-- Calculates the payment for window cleaning based on given conditions --/
def calculate_payment (stories : ℕ) (windows_per_floor : ℕ) (payment_per_window : ℕ) 
  (deduction_per_2_days : ℕ) (days_taken : ℕ) : ℕ :=
  let total_windows := stories * windows_per_floor
  let base_payment := total_windows * payment_per_window
  let time_deductions := (days_taken / 2) * deduction_per_2_days
  base_payment - time_deductions

/-- Theorem stating that Lucas' father will pay him $33 --/
theorem lucas_payment :
  calculate_payment 4 5 2 1 14 = 33 := by
  sorry

end lucas_payment_l1973_197306


namespace min_value_of_f_l1973_197322

-- Define the function
def f (x : ℝ) : ℝ := x^3 + x^2 - x + 1

-- State the theorem
theorem min_value_of_f :
  ∃ (x : ℝ), x ∈ Set.Icc (-2) 1 ∧
  (∀ (y : ℝ), y ∈ Set.Icc (-2) 1 → f x ≤ f y) ∧
  f x = -1 :=
sorry

end min_value_of_f_l1973_197322


namespace inequality_part_1_inequality_part_2_l1973_197344

-- Part I
theorem inequality_part_1 : 
  ∀ x : ℝ, (|x - 3| + |x + 5| ≥ 2 * |x + 5|) ↔ (x ≤ -1) := by sorry

-- Part II
theorem inequality_part_2 : 
  ∀ a : ℝ, (∀ x : ℝ, |x - a| + |x + 5| ≥ 6) ↔ (a ≥ 1 ∨ a ≤ -11) := by sorry

end inequality_part_1_inequality_part_2_l1973_197344


namespace problem_solution_l1973_197367

theorem problem_solution (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = 3) :
  x + (x^4 / y^3) + (y^4 / x^3) + y = 1021 := by
  sorry

end problem_solution_l1973_197367


namespace arithmetic_progression_with_squares_is_integer_l1973_197368

/-- An arithmetic progression containing the squares of its first three terms consists of integers. -/
theorem arithmetic_progression_with_squares_is_integer (a : ℕ → ℚ) (d : ℚ) :
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic progression condition
  (∃ k l m : ℕ, a k = (a 1)^2 ∧ a l = (a 2)^2 ∧ a m = (a 3)^2) →  -- squares condition
  ∀ n, ∃ z : ℤ, a n = z :=
by sorry

end arithmetic_progression_with_squares_is_integer_l1973_197368


namespace min_rows_for_hockey_arena_l1973_197359

/-- Represents a hockey arena with rows of seats -/
structure Arena where
  seats_per_row : ℕ
  total_students : ℕ
  max_students_per_school : ℕ

/-- Calculates the minimum number of rows required in the arena -/
def min_rows_required (arena : Arena) : ℕ :=
  sorry

/-- The theorem stating the minimum number of rows required for the given conditions -/
theorem min_rows_for_hockey_arena :
  let arena : Arena := {
    seats_per_row := 168,
    total_students := 2016,
    max_students_per_school := 45
  }
  min_rows_required arena = 16 := by sorry

end min_rows_for_hockey_arena_l1973_197359


namespace valid_numbers_l1973_197362

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  (a + 1) * (b + 2) * (c + 3) * (d + 4) = 234

theorem valid_numbers :
  ∀ n : ℕ, is_valid_number n → n = 1109 ∨ n = 2009 :=
by sorry

end valid_numbers_l1973_197362


namespace max_set_size_l1973_197393

def is_valid_set (s : Finset Nat) : Prop :=
  s.card > 0 ∧ 10 ∉ s ∧ s.sum (λ x => x^2) = 2500

theorem max_set_size :
  (∃ (s : Finset Nat), is_valid_set s ∧ s.card = 17) ∧
  (∀ (s : Finset Nat), is_valid_set s → s.card ≤ 17) :=
sorry

end max_set_size_l1973_197393


namespace coefficient_of_y_l1973_197370

theorem coefficient_of_y (x y : ℝ) (a : ℝ) : 
  x / (2 * y) = 3 / 2 → 
  (7 * x + a * y) / (x - 2 * y) = 26 → 
  a = 5 := by
sorry

end coefficient_of_y_l1973_197370


namespace triangle_sum_theorem_l1973_197398

def triangle_numbers : Finset ℕ := {1, 3, 5, 7, 9, 11}

theorem triangle_sum_theorem (vertex_sum midpoint_sum : ℕ → ℕ → ℕ → ℕ) 
  (h1 : vertex_sum 19 19 19 = (vertex_sum 1 3 5 + vertex_sum 7 9 11))
  (h2 : ∀ a b c, a ∈ triangle_numbers → b ∈ triangle_numbers → c ∈ triangle_numbers → 
    a ≠ b ∧ b ≠ c ∧ a ≠ c → vertex_sum a b c + midpoint_sum a b c = 19)
  (h3 : ∀ a b c d e f, a ∈ triangle_numbers → b ∈ triangle_numbers → c ∈ triangle_numbers →
    d ∈ triangle_numbers → e ∈ triangle_numbers → f ∈ triangle_numbers →
    a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ e ∧ e ≠ f ∧ f ≠ a → 
    vertex_sum a c e + midpoint_sum b d f = 19) :
  ∃ a b c, a ∈ triangle_numbers ∧ b ∈ triangle_numbers ∧ c ∈ triangle_numbers ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ vertex_sum a b c = 21 :=
sorry

end triangle_sum_theorem_l1973_197398


namespace circle_symmetry_l1973_197381

/-- The original circle -/
def original_circle (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 4*y - 4 = 0

/-- The line of symmetry -/
def symmetry_line (x y : ℝ) : Prop :=
  x - y - 1 = 0

/-- The symmetric circle -/
def symmetric_circle (x y : ℝ) : Prop :=
  (x - 3)^2 + (y + 2)^2 = 9

/-- Theorem stating that the symmetric circle is indeed symmetric to the original circle
    with respect to the given line of symmetry -/
theorem circle_symmetry :
  ∀ (x y : ℝ), original_circle x y ↔ 
  ∃ (x' y' : ℝ), symmetric_circle x' y' ∧ 
  ((x + x')/2 = (y + y')/2 + 1) ∧
  (y' - y)/(x' - x) = -1 :=
sorry

end circle_symmetry_l1973_197381


namespace right_triangle_in_square_l1973_197384

theorem right_triangle_in_square (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∃ (s : ℝ), s > 0 ∧ s^2 = 16 ∧ a^2 + b^2 = s^2) →
  a * b = 16 :=
by sorry

end right_triangle_in_square_l1973_197384


namespace consecutive_integers_sum_50_l1973_197311

theorem consecutive_integers_sum_50 : 
  ∃ (x : ℕ), x > 0 ∧ x + (x + 1) + (x + 2) + (x + 3) = 50 :=
by sorry

end consecutive_integers_sum_50_l1973_197311


namespace hotel_to_ticket_ratio_l1973_197378

/-- Represents the trip expenses and calculates the ratio of hotel cost to ticket cost. -/
def tripExpenses (initialAmount ticketCost amountLeft : ℚ) : ℚ × ℚ := by
  -- Define total spent
  let totalSpent := initialAmount - amountLeft
  -- Define hotel cost
  let hotelCost := totalSpent - ticketCost
  -- Calculate the ratio
  let ratio := hotelCost / ticketCost
  -- Return the simplified ratio
  exact (1, 2)

/-- Theorem stating that the ratio of hotel cost to ticket cost is 1:2 for the given values. -/
theorem hotel_to_ticket_ratio :
  tripExpenses 760 300 310 = (1, 2) := by
  sorry

end hotel_to_ticket_ratio_l1973_197378


namespace oscillating_bounded_example_unbounded_oscillations_example_l1973_197335

-- Part a
def oscillating_bounded (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  a < b ∧
  ∀ ε > 0, ∀ X : ℝ, ∃ x₁ x₂ : ℝ, 
    x₁ > X ∧ x₂ > X ∧ 
    f x₁ < a + ε ∧ f x₂ > b - ε

theorem oscillating_bounded_example (a b : ℝ) (h : a < b) :
  oscillating_bounded (fun x ↦ a + (b - a) * Real.sin x ^ 2) a b :=
sorry

-- Part b
def unbounded_oscillations (f : ℝ → ℝ) : Prop :=
  ∀ M : ℝ, ∃ X : ℝ, ∀ x > X, ∃ y > x, 
    (f y > M ∧ f x < -M) ∨ (f y < -M ∧ f x > M)

theorem unbounded_oscillations_example :
  unbounded_oscillations (fun x ↦ x * Real.sin x) :=
sorry

end oscillating_bounded_example_unbounded_oscillations_example_l1973_197335


namespace complex_number_modulus_l1973_197363

theorem complex_number_modulus : 
  let z : ℂ := (1 + 3*I) / (1 - I)
  Complex.abs z = Real.sqrt 5 := by sorry

end complex_number_modulus_l1973_197363
