import Mathlib

namespace half_abs_diff_squares_l2557_255771

theorem half_abs_diff_squares : (1 / 2 : ℝ) * |23^2 - 19^2| = 84 := by
  sorry

end half_abs_diff_squares_l2557_255771


namespace solution_set_for_inequality_l2557_255720

theorem solution_set_for_inequality (a b : ℝ) : 
  (∀ x : ℝ, x^2 - a*x + b = 0 ↔ x = 1 ∨ x = 2) →
  {x : ℝ | x ≤ 1} = Set.Iic 1 := by
  sorry

end solution_set_for_inequality_l2557_255720


namespace basketball_team_chemistry_count_l2557_255759

theorem basketball_team_chemistry_count :
  ∀ (total_players physics_players both_players chemistry_players : ℕ),
    total_players = 15 →
    physics_players = 8 →
    both_players = 3 →
    physics_players + chemistry_players - both_players = total_players →
    chemistry_players = 10 := by
  sorry

end basketball_team_chemistry_count_l2557_255759


namespace cosine_value_proof_l2557_255729

theorem cosine_value_proof (α : Real) 
  (h1 : Real.sin (Real.pi - α) = 1/3)
  (h2 : Real.pi/2 ≤ α)
  (h3 : α ≤ Real.pi) :
  Real.cos α = -2 * Real.sqrt 2 / 3 := by
sorry

end cosine_value_proof_l2557_255729


namespace complex_product_equals_43_l2557_255767

theorem complex_product_equals_43 :
  let y : ℂ := Complex.exp (Complex.I * (2 * Real.pi / 9))
  (2 * y + y^2) * (2 * y^2 + y^4) * (2 * y^3 + y^6) * 
  (2 * y^4 + y^8) * (2 * y^5 + y^10) * (2 * y^6 + y^12) = 43 := by
  sorry

end complex_product_equals_43_l2557_255767


namespace tv_show_cost_l2557_255709

/-- Calculates the total cost of producing a TV show with the given parameters -/
def total_cost_tv_show (
  num_seasons : ℕ
  ) (first_season_cost_per_episode : ℕ
  ) (first_season_episodes : ℕ
  ) (last_season_episodes : ℕ
  ) : ℕ :=
  let other_season_cost_per_episode := 2 * first_season_cost_per_episode
  let first_season_cost := first_season_cost_per_episode * first_season_episodes
  let other_seasons_cost := 
    (other_season_cost_per_episode * first_season_episodes * 3 / 2) +
    (other_season_cost_per_episode * first_season_episodes * 9 / 4) +
    (other_season_cost_per_episode * first_season_episodes * 27 / 8) +
    (other_season_cost_per_episode * last_season_episodes)
  first_season_cost + other_seasons_cost

/-- The total cost of producing the TV show is $23,000,000 -/
theorem tv_show_cost :
  total_cost_tv_show 5 100000 12 24 = 23000000 := by
  sorry

end tv_show_cost_l2557_255709


namespace sunzi_wood_measurement_l2557_255726

theorem sunzi_wood_measurement (x y : ℝ) : 
  (x - y = 4.5 ∧ (1/2) * x + 1 = y) ↔ 
  (∃ (rope_length wood_length : ℝ),
    rope_length = x ∧
    wood_length = y ∧
    rope_length - wood_length = 4.5 ∧
    (1/2) * rope_length + 1 = wood_length) :=
by sorry

end sunzi_wood_measurement_l2557_255726


namespace decimal_to_base5_l2557_255788

theorem decimal_to_base5 :
  ∃ (a b c : ℕ), a < 5 ∧ b < 5 ∧ c < 5 ∧ 88 = c * 5^2 + b * 5^1 + a * 5^0 ∧ 
  (a = 3 ∧ b = 2 ∧ c = 3) := by
sorry

end decimal_to_base5_l2557_255788


namespace unique_solution_exponential_equation_l2557_255765

theorem unique_solution_exponential_equation :
  ∃! x : ℝ, (2 : ℝ)^(2*x) * 50^x = 250^3 := by
sorry

end unique_solution_exponential_equation_l2557_255765


namespace bathroom_length_l2557_255713

theorem bathroom_length (area : ℝ) (width : ℝ) (length : ℝ) : 
  area = 8 → width = 2 → area = length * width → length = 4 := by
  sorry

end bathroom_length_l2557_255713


namespace square_tile_area_l2557_255784

theorem square_tile_area (side_length : ℝ) (h : side_length = 7) :
  side_length * side_length = 49 := by
  sorry

end square_tile_area_l2557_255784


namespace diana_bike_time_l2557_255792

/-- Proves that Diana will take 6 hours to get home given the specified conditions -/
theorem diana_bike_time : 
  let total_distance : ℝ := 10
  let initial_speed : ℝ := 3
  let initial_time : ℝ := 2
  let tired_speed : ℝ := 1
  let initial_distance := initial_speed * initial_time
  let remaining_distance := total_distance - initial_distance
  let tired_time := remaining_distance / tired_speed
  initial_time + tired_time = 6 := by
  sorry

end diana_bike_time_l2557_255792


namespace teammates_score_l2557_255761

def volleyball_scores (total_team_score : ℕ) : Prop :=
  ∃ (lizzie nathalie aimee julia ellen other : ℕ),
    lizzie = 4 ∧
    nathalie = 2 * lizzie + 3 ∧
    aimee = 2 * (lizzie + nathalie) + 1 ∧
    julia = nathalie / 2 - 2 ∧
    ellen = Int.sqrt aimee * 3 ∧
    lizzie + nathalie + aimee + julia + ellen + other = total_team_score

theorem teammates_score :
  volleyball_scores 100 → ∃ other : ℕ, other = 36 :=
by sorry

end teammates_score_l2557_255761


namespace difference_23rd_21st_triangular_l2557_255790

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem difference_23rd_21st_triangular : 
  triangular_number 23 - triangular_number 21 = 45 := by
  sorry

end difference_23rd_21st_triangular_l2557_255790


namespace tangent_line_parallel_to_y_equals_4x_l2557_255772

noncomputable def f (x : ℝ) : ℝ := x^3 + x - 2

theorem tangent_line_parallel_to_y_equals_4x :
  ∃! P₀ : ℝ × ℝ, 
    P₀.1 = 1 ∧ 
    P₀.2 = 0 ∧ 
    (∀ x : ℝ, f x = x^3 + x - 2) ∧
    (deriv f P₀.1 = 4) :=
by sorry

end tangent_line_parallel_to_y_equals_4x_l2557_255772


namespace factor_expression_l2557_255727

theorem factor_expression (x : ℝ) : 5 * x * (x + 2) + 9 * (x + 2) = (x + 2) * (5 * x + 9) := by
  sorry

end factor_expression_l2557_255727


namespace total_pupils_after_addition_l2557_255778

/-- Given a school with an initial number of girls and boys, and additional girls joining,
    calculate the total number of pupils after the new girls joined. -/
theorem total_pupils_after_addition (initial_girls initial_boys additional_girls : ℕ) :
  initial_girls = 706 →
  initial_boys = 222 →
  additional_girls = 418 →
  initial_girls + initial_boys + additional_girls = 1346 := by
  sorry

end total_pupils_after_addition_l2557_255778


namespace complete_square_sum_l2557_255740

theorem complete_square_sum (a b c d : ℤ) : 
  (∀ x : ℝ, 64 * x^2 + 96 * x - 36 = 0 ↔ (a * x + b)^2 + d = c) →
  a > 0 →
  a + b + c + d = -94 :=
by sorry

end complete_square_sum_l2557_255740


namespace angle_A_value_l2557_255717

theorem angle_A_value (A B C : Real) (a b : Real) : 
  0 < A ∧ A < π/2 →  -- A is acute
  0 < B ∧ B < π/2 →  -- B is acute
  0 < C ∧ C < π/2 →  -- C is acute
  A + B + C = π →    -- sum of angles in a triangle
  a > 0 →            -- side length is positive
  b > 0 →            -- side length is positive
  2 * a * Real.sin B = b →  -- given condition
  a / Real.sin A = b / Real.sin B →  -- law of sines
  A = π/6 := by
sorry

end angle_A_value_l2557_255717


namespace intersection_distance_sum_l2557_255794

/-- Given a line and a circle in 2D space, prove that the sum of distances from a specific point to the intersection points of the line and circle is √6. -/
theorem intersection_distance_sum (l : Set (ℝ × ℝ)) (C : Set (ℝ × ℝ)) (P A B : ℝ × ℝ) :
  l = {(x, y) : ℝ × ℝ | x + y = 1} →
  C = {(x, y) : ℝ × ℝ | x^2 + y^2 - 2*x + 2*y = 0} →
  P = (1, 0) →
  A ∈ l ∩ C →
  B ∈ l ∩ C →
  A ≠ B →
  Real.sqrt ((A.1 - P.1)^2 + (A.2 - P.2)^2) + Real.sqrt ((B.1 - P.1)^2 + (B.2 - P.2)^2) = Real.sqrt 6 := by
  sorry

end intersection_distance_sum_l2557_255794


namespace regular_polygon_sides_l2557_255722

theorem regular_polygon_sides (d : ℕ) : d = 14 → ∃ n : ℕ, n > 2 ∧ d = n * (n - 3) / 2 ∧ n = 7 := by
  sorry

end regular_polygon_sides_l2557_255722


namespace complex_fraction_equals_two_l2557_255782

theorem complex_fraction_equals_two (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : a^2 + a*b + b^2 = 0) : 
  (a^12 + b^12) / (a + b)^12 = 2 := by
sorry

end complex_fraction_equals_two_l2557_255782


namespace symmetric_points_sum_l2557_255758

/-- Given two points P and Q in a Cartesian coordinate system that are symmetric
    with respect to the x-axis, prove that the sum of their x-coordinate and
    the y-coordinate (before the shift) is 3. -/
theorem symmetric_points_sum (a b : ℝ) : 
  (a - 3 = 2) →  -- x-coordinates are equal
  (1 = -(b + 1)) →  -- y-coordinates are opposites
  a + b = 3 := by
  sorry

end symmetric_points_sum_l2557_255758


namespace gcd_problem_l2557_255795

theorem gcd_problem (a : ℤ) (h : ∃ k : ℤ, a = (2*k + 1) * 7767) :
  Int.gcd (6*a^2 + 5*a + 108) (3*a + 9) = 9 := by
  sorry

end gcd_problem_l2557_255795


namespace median_pets_is_three_l2557_255762

/-- Represents the distribution of pet ownership --/
def PetDistribution : List (ℕ × ℕ) :=
  [(2, 5), (3, 6), (4, 1), (5, 4), (6, 3)]

/-- The total number of individuals in the survey --/
def TotalIndividuals : ℕ := 19

/-- Calculates the median position for an odd number of data points --/
def MedianPosition (n : ℕ) : ℕ :=
  (n + 1) / 2

/-- Finds the median number of pets owned given the distribution --/
def MedianPets (dist : List (ℕ × ℕ)) (total : ℕ) : ℕ :=
  sorry -- Proof to be implemented

theorem median_pets_is_three :
  MedianPets PetDistribution TotalIndividuals = 3 :=
by sorry

end median_pets_is_three_l2557_255762


namespace books_owned_by_three_l2557_255774

/-- The number of books owned by Harry, Flora, and Gary -/
def total_books (harry_books : ℕ) : ℕ :=
  let flora_books := 2 * harry_books
  let gary_books := harry_books / 2
  harry_books + flora_books + gary_books

/-- Theorem stating that the total number of books is 175 when Harry has 50 books -/
theorem books_owned_by_three (harry_books : ℕ) 
  (h : harry_books = 50) : total_books harry_books = 175 := by
  sorry

end books_owned_by_three_l2557_255774


namespace marcie_coffee_cups_l2557_255798

theorem marcie_coffee_cups (sandra_cups : ℕ) (total_cups : ℕ) (marcie_cups : ℕ) : 
  sandra_cups = 6 → total_cups = 8 → marcie_cups = total_cups - sandra_cups → marcie_cups = 2 := by
  sorry

end marcie_coffee_cups_l2557_255798


namespace total_pay_calculation_l2557_255741

theorem total_pay_calculation (pay_B : ℕ) (pay_A : ℕ) : 
  pay_B = 224 → 
  pay_A = (150 * pay_B) / 100 → 
  pay_A + pay_B = 560 :=
by
  sorry

end total_pay_calculation_l2557_255741


namespace star_four_three_l2557_255711

def star (a b : ℕ) : ℕ := 3 * a^2 + 5 * b

theorem star_four_three : star 4 3 = 63 := by
  sorry

end star_four_three_l2557_255711


namespace fraction_arrangement_equals_two_l2557_255735

theorem fraction_arrangement_equals_two : ∃ (f : ℚ → ℚ → ℚ → ℚ → ℚ), f (1/4) (1/4) (1/4) (1/4) = 2 := by
  sorry

end fraction_arrangement_equals_two_l2557_255735


namespace nicole_fish_tanks_l2557_255753

/-- Represents the number of fish tanks Nicole has -/
def num_tanks : ℕ := 4

/-- Represents the amount of water (in gallons) needed for each of the first two tanks -/
def water_first_two : ℕ := 8

/-- Represents the amount of water (in gallons) needed for each of the other two tanks -/
def water_other_two : ℕ := water_first_two - 2

/-- Represents the total amount of water (in gallons) needed for all tanks in one week -/
def total_water_per_week : ℕ := 2 * water_first_two + 2 * water_other_two

/-- Represents the number of weeks -/
def num_weeks : ℕ := 4

/-- Represents the total amount of water (in gallons) needed for all tanks in four weeks -/
def total_water_four_weeks : ℕ := 112

theorem nicole_fish_tanks :
  num_tanks = 4 ∧
  water_first_two = 8 ∧
  water_other_two = water_first_two - 2 ∧
  total_water_per_week = 2 * water_first_two + 2 * water_other_two ∧
  total_water_four_weeks = num_weeks * total_water_per_week :=
by sorry

end nicole_fish_tanks_l2557_255753


namespace largest_multiple_of_8_under_100_l2557_255786

theorem largest_multiple_of_8_under_100 : ∃ n : ℕ, n * 8 = 96 ∧ 
  (∀ m : ℕ, m * 8 < 100 → m * 8 ≤ 96) := by
  sorry

end largest_multiple_of_8_under_100_l2557_255786


namespace quadratic_minimum_minimizer_value_l2557_255705

theorem quadratic_minimum (c : ℝ) : 
  2 * c^2 - 7 * c + 4 ≥ 2 * (7/4)^2 - 7 * (7/4) + 4 := by
  sorry

theorem minimizer_value : 
  ∃ (c : ℝ), ∀ (x : ℝ), 2 * x^2 - 7 * x + 4 ≥ 2 * c^2 - 7 * c + 4 ∧ c = 7/4 := by
  sorry

end quadratic_minimum_minimizer_value_l2557_255705


namespace largest_element_complement_A_intersect_B_l2557_255710

def I : Set ℤ := {x | 1 ≤ x ∧ x ≤ 100}
def A : Set ℤ := {m ∈ I | ∃ k : ℤ, m = 2 * k + 1}
def B : Set ℤ := {n ∈ I | ∃ k : ℤ, n = 3 * k}

theorem largest_element_complement_A_intersect_B :
  ∃ x : ℤ, x ∈ (I \ A) ∩ B ∧ x = 96 ∧ ∀ y ∈ (I \ A) ∩ B, y ≤ x :=
sorry

end largest_element_complement_A_intersect_B_l2557_255710


namespace debbys_water_consumption_l2557_255775

/-- Given Debby's beverage consumption pattern, prove the number of water bottles she drank per day. -/
theorem debbys_water_consumption 
  (total_soda : ℕ) 
  (total_water : ℕ) 
  (soda_per_day : ℕ) 
  (soda_days : ℕ) 
  (water_days : ℕ) 
  (h1 : total_soda = 360)
  (h2 : total_water = 162)
  (h3 : soda_per_day = 9)
  (h4 : soda_days = 40)
  (h5 : water_days = 30)
  (h6 : total_soda = soda_per_day * soda_days) :
  (total_water : ℚ) / water_days = 5.4 := by
  sorry

end debbys_water_consumption_l2557_255775


namespace congruence_solution_l2557_255719

theorem congruence_solution : ∃! n : ℤ, 0 ≤ n ∧ n < 25 ∧ -150 ≡ n [ZMOD 25] := by
  sorry

end congruence_solution_l2557_255719


namespace total_accessories_count_l2557_255703

def dresses_per_day_first_period : ℕ := 4
def days_first_period : ℕ := 10
def dresses_per_day_second_period : ℕ := 5
def days_second_period : ℕ := 3
def ribbons_per_dress : ℕ := 3
def buttons_per_dress : ℕ := 2
def lace_trims_per_dress : ℕ := 1

theorem total_accessories_count :
  (dresses_per_day_first_period * days_first_period +
   dresses_per_day_second_period * days_second_period) *
  (ribbons_per_dress + buttons_per_dress + lace_trims_per_dress) = 330 := by
sorry

end total_accessories_count_l2557_255703


namespace aleesia_weight_loss_weeks_aleesia_weight_loss_weeks_proof_l2557_255769

theorem aleesia_weight_loss_weeks : ℝ → Prop :=
  fun w =>
    let aleesia_weekly_loss : ℝ := 1.5
    let alexei_weekly_loss : ℝ := 2.5
    let alexei_weeks : ℝ := 8
    let total_loss : ℝ := 35
    (aleesia_weekly_loss * w + alexei_weekly_loss * alexei_weeks = total_loss) →
    w = 10

-- The proof would go here
theorem aleesia_weight_loss_weeks_proof : aleesia_weight_loss_weeks 10 := by
  sorry

end aleesia_weight_loss_weeks_aleesia_weight_loss_weeks_proof_l2557_255769


namespace max_rooms_needed_l2557_255715

/-- Represents a group of football fans -/
structure FanGroup where
  team : Fin 3
  gender : Bool
  count : Nat

/-- The maximum number of fans that can be accommodated in one room -/
def maxFansPerRoom : Nat := 3

/-- The total number of football fans -/
def totalFans : Nat := 100

/-- Calculates the number of rooms needed for a given fan group -/
def roomsNeeded (group : FanGroup) : Nat :=
  (group.count + maxFansPerRoom - 1) / maxFansPerRoom

/-- Theorem stating the maximum number of rooms needed -/
theorem max_rooms_needed (fans : List FanGroup) 
  (h1 : fans.length = 6)
  (h2 : fans.foldl (λ acc g => acc + g.count) 0 = totalFans) : 
  (fans.foldl (λ acc g => acc + roomsNeeded g) 0) ≤ 37 := by
  sorry

end max_rooms_needed_l2557_255715


namespace pant_price_before_discount_l2557_255750

/-- The cost of a wardrobe given specific items and prices --/
def wardrobe_cost (skirt_price blouse_price pant_price : ℚ) : ℚ :=
  3 * skirt_price + 5 * blouse_price + (pant_price + pant_price / 2)

/-- Theorem stating the cost of pants before discount --/
theorem pant_price_before_discount :
  ∃ (pant_price : ℚ),
    wardrobe_cost 20 15 pant_price = 180 ∧
    pant_price = 30 :=
by
  sorry

#check pant_price_before_discount

end pant_price_before_discount_l2557_255750


namespace base8_532_equals_base7_1006_l2557_255751

/-- Converts a number from base 8 to base 10 --/
def base8ToDecimal (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 7 --/
def decimalToBase7 (n : ℕ) : ℕ := sorry

/-- Theorem stating that 532 in base 8 is equal to 1006 in base 7 --/
theorem base8_532_equals_base7_1006 : 
  decimalToBase7 (base8ToDecimal 532) = 1006 := by sorry

end base8_532_equals_base7_1006_l2557_255751


namespace negation_of_existence_negation_of_quadratic_equation_l2557_255791

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) := by sorry

theorem negation_of_quadratic_equation :
  (¬ ∃ x : ℝ, x^2 + 2*x + 5 = 0) ↔ (∀ x : ℝ, x^2 + 2*x + 5 ≠ 0) := by sorry

end negation_of_existence_negation_of_quadratic_equation_l2557_255791


namespace zach_babysitting_hours_l2557_255746

def bike_cost : ℕ := 100
def weekly_allowance : ℕ := 5
def lawn_mowing_pay : ℕ := 10
def babysitting_rate : ℕ := 7
def current_savings : ℕ := 65
def additional_needed : ℕ := 6

theorem zach_babysitting_hours :
  ∃ (hours : ℕ),
    bike_cost = current_savings + weekly_allowance + lawn_mowing_pay + babysitting_rate * hours + additional_needed ∧
    hours = 2 := by
  sorry

end zach_babysitting_hours_l2557_255746


namespace tangent_line_at_one_monotonicity_intervals_minimum_value_on_interval_l2557_255708

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 + 3 * a * x^2 + 1

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 6 * x^2 + 6 * a * x

-- Theorem for the tangent line equation when a = 0
theorem tangent_line_at_one (x y : ℝ) :
  f 0 1 = 3 ∧ f' 0 1 = 6 → (6 * x - y - 3 = 0 ↔ y - 3 = 6 * (x - 1)) :=
sorry

-- Theorem for monotonicity intervals
theorem monotonicity_intervals (a : ℝ) :
  (a = 0 → ∀ x, f' a x ≥ 0) ∧
  (a > 0 → ∀ x, (x < -a ∨ x > 0) ↔ f' a x > 0) ∧
  (a < 0 → ∀ x, (x < 0 ∨ x > -a) ↔ f' a x > 0) :=
sorry

-- Theorem for minimum value on [0, 2]
theorem minimum_value_on_interval (a : ℝ) :
  (a ≥ 0 → ∀ x ∈ Set.Icc 0 2, f a x ≥ f a 0) ∧
  (-2 < a ∧ a < 0 → ∀ x ∈ Set.Icc 0 2, f a x ≥ f a (-a)) ∧
  (a ≤ -2 → ∀ x ∈ Set.Icc 0 2, f a x ≥ f a 2) :=
sorry

end tangent_line_at_one_monotonicity_intervals_minimum_value_on_interval_l2557_255708


namespace f_odd_and_decreasing_l2557_255701

-- Define the function f(x) = -x^3
def f (x : ℝ) : ℝ := -x^3

-- State the theorem
theorem f_odd_and_decreasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f x > f y) :=
by sorry

end f_odd_and_decreasing_l2557_255701


namespace chess_tournament_players_l2557_255733

theorem chess_tournament_players (total_games : ℕ) (h : total_games = 380) :
  ∃ n : ℕ, n > 0 ∧ 2 * n * (n - 1) = total_games :=
by
  -- The proof would go here
  sorry

end chess_tournament_players_l2557_255733


namespace max_z_value_l2557_255712

theorem max_z_value (x y z : ℝ) (h1 : x + y + z = 5) (h2 : x*y + y*z + z*x = 3) :
  z ≤ 13/3 := by
sorry

end max_z_value_l2557_255712


namespace relay_race_theorem_l2557_255707

def relay_race_length (team_size : ℕ) (standard_distance : ℝ) (long_distance_multiplier : ℝ) : ℝ :=
  (team_size - 1) * standard_distance + long_distance_multiplier * standard_distance

theorem relay_race_theorem :
  relay_race_length 5 3 2 = 18 := by
  sorry

end relay_race_theorem_l2557_255707


namespace rectangle_painting_possibilities_l2557_255737

theorem rectangle_painting_possibilities : 
  ∃! n : ℕ, n = (Finset.filter 
    (fun p : ℕ × ℕ => 
      p.2 > p.1 ∧ 
      (p.1 - 4) * (p.2 - 4) = 2 * p.1 * p.2 / 3 ∧ 
      p.1 > 0 ∧ p.2 > 0)
    (Finset.product (Finset.range 1000) (Finset.range 1000))).card ∧ 
  n = 3 :=
sorry

end rectangle_painting_possibilities_l2557_255737


namespace tom_initial_investment_l2557_255718

/-- Represents the business scenario with Tom and Jose's investments --/
structure BusinessScenario where
  tom_investment : ℕ
  jose_investment : ℕ
  total_profit : ℕ
  jose_profit : ℕ

/-- Calculates Tom's initial investment given the business scenario --/
def calculate_tom_investment (scenario : BusinessScenario) : ℕ :=
  (scenario.total_profit - scenario.jose_profit) * 450000 / (scenario.jose_profit * 12)

/-- Theorem stating that Tom's initial investment was 30,000 --/
theorem tom_initial_investment :
  ∀ (scenario : BusinessScenario),
    scenario.jose_investment = 45000 ∧
    scenario.total_profit = 45000 ∧
    scenario.jose_profit = 25000 →
    calculate_tom_investment scenario = 30000 := by
  sorry


end tom_initial_investment_l2557_255718


namespace allen_book_pages_l2557_255781

/-- Calculates the total number of pages in a book based on daily reading rate and days to finish -/
def total_pages (pages_per_day : ℕ) (days_to_finish : ℕ) : ℕ :=
  pages_per_day * days_to_finish

/-- Proves that Allen's book has 120 pages given his reading rate and time to finish -/
theorem allen_book_pages :
  let pages_per_day : ℕ := 10
  let days_to_finish : ℕ := 12
  total_pages pages_per_day days_to_finish = 120 := by
  sorry

end allen_book_pages_l2557_255781


namespace christopher_age_l2557_255796

theorem christopher_age (c g : ℕ) : 
  c = 2 * g ∧ 
  c - 9 = 5 * (g - 9) → 
  c = 24 := by
sorry

end christopher_age_l2557_255796


namespace cyclic_equality_l2557_255702

theorem cyclic_equality (a b c x y z : ℝ) 
  (h1 : a = b * z + c * y)
  (h2 : b = c * x + a * z)
  (h3 : c = a * y + b * x)
  (hx : x ≠ 1 ∧ x ≠ -1)
  (hy : y ≠ 1 ∧ y ≠ -1)
  (hz : z ≠ 1 ∧ z ≠ -1) :
  a^2 / (1 - x^2) = b^2 / (1 - y^2) ∧ b^2 / (1 - y^2) = c^2 / (1 - z^2) := by
  sorry

end cyclic_equality_l2557_255702


namespace det_A_equals_two_l2557_255721

theorem det_A_equals_two (a d : ℝ) (A : Matrix (Fin 2) (Fin 2) ℝ) :
  A = !![a, 2; -3, d] →
  A + A⁻¹ = 0 →
  Matrix.det A = 2 := by
sorry

end det_A_equals_two_l2557_255721


namespace train_passes_jogger_l2557_255743

/-- Time for a train to pass a jogger given their speeds and initial positions -/
theorem train_passes_jogger (jogger_speed : ℝ) (train_speed : ℝ) (train_length : ℝ) (initial_distance : ℝ) :
  jogger_speed = 9 * (1000 / 3600) →
  train_speed = 45 * (1000 / 3600) →
  train_length = 120 →
  initial_distance = 260 →
  (initial_distance + train_length) / (train_speed - jogger_speed) = 38 := by
  sorry

end train_passes_jogger_l2557_255743


namespace cycle_cost_proof_l2557_255756

def cycle_problem (selling_price : ℕ) (gain_percentage : ℕ) : Prop :=
  let original_cost : ℕ := selling_price / 2
  selling_price = original_cost * (100 + gain_percentage) / 100 ∧
  original_cost = 1000

theorem cycle_cost_proof :
  cycle_problem 2000 100 :=
sorry

end cycle_cost_proof_l2557_255756


namespace triangle_OAB_and_point_C_l2557_255770

-- Define the points
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (2, -1)
def B : ℝ × ℝ := (1, 2)

-- Define the area of a triangle given three points
def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Define perpendicularity of two vectors
def perpendicular (v1 v2 : ℝ × ℝ) : Prop := sorry

-- Define parallelism of two vectors
def parallel (v1 v2 : ℝ × ℝ) : Prop := sorry

-- Define the vector between two points
def vector (p1 p2 : ℝ × ℝ) : ℝ × ℝ := sorry

-- Theorem statement
theorem triangle_OAB_and_point_C :
  -- Part 1: Area of triangle OAB
  triangle_area O A B = 5/2 ∧
  -- Part 2: Coordinates of point C
  ∃ C : ℝ × ℝ,
    perpendicular (vector B C) (vector A B) ∧
    parallel (vector A C) (vector O B) ∧
    C = (4, 3) := by
  sorry

end triangle_OAB_and_point_C_l2557_255770


namespace parabola_standard_form_l2557_255745

/-- A parabola with vertex at the origin and axis of symmetry x = -2 has the standard form equation y² = 8x -/
theorem parabola_standard_form (p : ℝ) (h : p / 2 = 2) :
  ∀ x y : ℝ, y^2 = 8 * x ↔ y^2 = 2 * p * x :=
by sorry

end parabola_standard_form_l2557_255745


namespace team_a_champion_probability_l2557_255704

/-- The probability of a team winning a single game -/
def game_win_prob : ℝ := 0.5

/-- The number of games Team A needs to win to become champion -/
def team_a_games_needed : ℕ := 1

/-- The number of games Team B needs to win to become champion -/
def team_b_games_needed : ℕ := 2

/-- The probability of Team A becoming the champion -/
def team_a_champion_prob : ℝ := 1 - game_win_prob ^ team_b_games_needed

theorem team_a_champion_probability :
  team_a_champion_prob = 0.75 := by sorry

end team_a_champion_probability_l2557_255704


namespace min_simultaneous_return_time_l2557_255783

def first_seven_primes : List Nat := [2, 3, 5, 7, 11, 13, 17]

def is_simultaneous_return (t : Nat) (horse_times : List Nat) : Bool :=
  (horse_times.filter (fun time => t % time = 0)).length ≥ 4

theorem min_simultaneous_return_time :
  let horse_times := first_seven_primes
  (∃ (t : Nat), t > 0 ∧ is_simultaneous_return t horse_times) ∧
  (∀ (t : Nat), 0 < t ∧ t < 210 → ¬is_simultaneous_return t horse_times) ∧
  is_simultaneous_return 210 horse_times :=
by sorry

end min_simultaneous_return_time_l2557_255783


namespace auction_bidding_l2557_255732

theorem auction_bidding (price_increase : ℕ) (start_price : ℕ) (end_price : ℕ) (num_bidders : ℕ) :
  price_increase = 5 →
  start_price = 15 →
  end_price = 65 →
  num_bidders = 2 →
  (end_price - start_price) / price_increase / num_bidders = 5 :=
by sorry

end auction_bidding_l2557_255732


namespace intersection_implies_a_value_l2557_255773

def P (a : ℝ) : Set ℝ := {a^2, a+1, -3}
def Q (a : ℝ) : Set ℝ := {a^2+1, 2*a-1, a-3}

theorem intersection_implies_a_value :
  ∀ a : ℝ, P a ∩ Q a = {-3} → a = -1 := by
sorry

end intersection_implies_a_value_l2557_255773


namespace infinitely_many_special_numbers_l2557_255757

/-- Sum of digits of a natural number's decimal representation -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Predicate for special numbers -/
def is_special (m : ℕ) : Prop :=
  ∀ n : ℕ, m ≠ n + sum_of_digits n

/-- Theorem stating that there are infinitely many special numbers -/
theorem infinitely_many_special_numbers :
  ∀ k : ℕ, ∃ S : Finset ℕ, (∀ m ∈ S, is_special m) ∧ S.card > k :=
sorry

end infinitely_many_special_numbers_l2557_255757


namespace sequence_sum_unique_value_l2557_255734

def is_strictly_increasing (s : ℕ → ℕ) : Prop :=
  ∀ n m : ℕ, n < m → s n < s m

theorem sequence_sum_unique_value
  (a b : ℕ → ℕ)
  (h_a_incr : is_strictly_increasing a)
  (h_b_incr : is_strictly_increasing b)
  (h_eq : a 10 = b 10)
  (h_lt : a 10 < 2017)
  (h_a_rec : ∀ n : ℕ, a (n + 2) = a (n + 1) + a n)
  (h_b_rec : ∀ n : ℕ, b (n + 1) = 2 * b n) :
  a 1 + b 1 = 5 := by
sorry

end sequence_sum_unique_value_l2557_255734


namespace tetrahedron_sphere_area_l2557_255780

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron -/
structure Tetrahedron where
  K : Point3D
  L : Point3D
  M : Point3D
  N : Point3D

/-- Represents a sphere -/
structure Sphere where
  center : Point3D
  radius : ℝ

/-- Calculates the Euclidean distance between two points -/
def distance (p q : Point3D) : ℝ := sorry

/-- Calculates the angle between three points -/
def angle (p q r : Point3D) : ℝ := sorry

/-- Calculates the spherical distance between two points on a sphere -/
def sphericalDistance (s : Sphere) (p q : Point3D) : ℝ := sorry

/-- Checks if a point is on a sphere -/
def isOnSphere (s : Sphere) (p : Point3D) : Prop := sorry

/-- The set of points on the sphere satisfying the distance condition -/
def distanceSet (s : Sphere) (t : Tetrahedron) : Set Point3D :=
  {p : Point3D | isOnSphere s p ∧ 
    sphericalDistance s p t.K + sphericalDistance s p t.L + 
    sphericalDistance s p t.M + sphericalDistance s p t.N ≤ 6 * Real.pi}

/-- Calculates the area of a set on a sphere -/
def sphericalArea (s : Sphere) (set : Set Point3D) : ℝ := sorry

theorem tetrahedron_sphere_area 
  (t : Tetrahedron) 
  (s : Sphere) 
  (h1 : distance t.K t.L = 5)
  (h2 : distance t.N t.M = 6)
  (h3 : angle t.L t.M t.N = 35 * Real.pi / 180)
  (h4 : angle t.K t.N t.M = 35 * Real.pi / 180)
  (h5 : angle t.L t.N t.M = 55 * Real.pi / 180)
  (h6 : angle t.K t.M t.N = 55 * Real.pi / 180)
  (h7 : isOnSphere s t.K ∧ isOnSphere s t.L ∧ isOnSphere s t.M ∧ isOnSphere s t.N) :
  sphericalArea s (distanceSet s t) = 18 * Real.pi := by
  sorry

end tetrahedron_sphere_area_l2557_255780


namespace curve_symmetry_l2557_255736

/-- The curve equation -/
def curve (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 6*y + 1 = 0

/-- The line equation -/
def line (m : ℝ) (x y : ℝ) : Prop := x + m*y + 4 = 0

/-- Two points are symmetric with respect to a line -/
def symmetric (P Q : ℝ × ℝ) (m : ℝ) : Prop :=
  ∃ (R : ℝ × ℝ), line m R.1 R.2 ∧ 
    R.1 = (P.1 + Q.1) / 2 ∧ 
    R.2 = (P.2 + Q.2) / 2

theorem curve_symmetry (m : ℝ) :
  (∃ (P Q : ℝ × ℝ), curve P.1 P.2 ∧ curve Q.1 Q.2 ∧ symmetric P Q m) →
  m = -1 := by
  sorry

end curve_symmetry_l2557_255736


namespace hyperbola_properties_l2557_255742

/-- The equation of a hyperbola passing through (1, 0) with asymptotes y = ±2x -/
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 - y^2/4 = 1

/-- The focus of the parabola y² = 4x -/
def parabola_focus : ℝ × ℝ := (1, 0)

/-- The asymptotes of the hyperbola -/
def asymptote_pos (x y : ℝ) : Prop := y = 2 * x
def asymptote_neg (x y : ℝ) : Prop := y = -2 * x

theorem hyperbola_properties :
  (∀ x y : ℝ, hyperbola_equation x y → (asymptote_pos x y ∨ asymptote_neg x y)) ∧
  hyperbola_equation parabola_focus.1 parabola_focus.2 :=
sorry

end hyperbola_properties_l2557_255742


namespace daisys_milk_problem_l2557_255766

theorem daisys_milk_problem (total_milk : ℝ) (cooking_percentage : ℝ) (leftover : ℝ) :
  total_milk = 16 ∧ cooking_percentage = 0.5 ∧ leftover = 2 →
  ∃ kids_consumption_percentage : ℝ,
    kids_consumption_percentage = 0.75 ∧
    leftover = (1 - cooking_percentage) * (total_milk - kids_consumption_percentage * total_milk) :=
by sorry

end daisys_milk_problem_l2557_255766


namespace intersection_of_M_and_complement_of_N_l2557_255725

-- Define the universal set U as ℝ
def U := ℝ

-- Define set M
def M : Set ℝ := {x : ℝ | 0 < x ∧ x < 2}

-- Define set N
def N : Set ℝ := {x : ℝ | x ≥ 1}

-- Theorem statement
theorem intersection_of_M_and_complement_of_N :
  M ∩ (Set.univ \ N) = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end intersection_of_M_and_complement_of_N_l2557_255725


namespace square_sheet_area_decrease_l2557_255764

theorem square_sheet_area_decrease (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (2 * b = 0.1 * 4 * a) → (1 - (a - b)^2 / a^2 = 0.04) := by
  sorry

end square_sheet_area_decrease_l2557_255764


namespace triangle_problem_l2557_255793

theorem triangle_problem (A B C : Real) (a b c : Real) :
  -- Triangle conditions
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  -- Side lengths are positive
  0 < a ∧ 0 < b ∧ 0 < c ∧
  -- Law of sines
  a / (Real.sin A) = b / (Real.sin B) ∧ b / (Real.sin B) = c / (Real.sin C) ∧
  -- Given conditions
  b = 2 ∧
  (1/2) * a * c * (Real.sin B) = Real.sqrt 3 →
  -- Conclusion
  B = π/3 ∧ a = 2 ∧ c = 2 := by
  sorry

end triangle_problem_l2557_255793


namespace sum_cube_inequality_l2557_255789

theorem sum_cube_inequality (x1 x2 x3 x4 : ℝ) 
  (h_pos1 : x1 > 0) (h_pos2 : x2 > 0) (h_pos3 : x3 > 0) (h_pos4 : x4 > 0)
  (h_cond1 : x1^3 + x3^3 + 3*x1*x3 = 1)
  (h_cond2 : x2 + x4 = 1) : 
  (x1 + 1/x1)^3 + (x2 + 1/x2)^3 + (x3 + 1/x3)^3 + (x4 + 1/x4)^3 ≥ 125/4 := by
sorry

end sum_cube_inequality_l2557_255789


namespace sum_ratio_equals_half_l2557_255724

theorem sum_ratio_equals_half : (1 + 2 + 3 + 4 + 5) / (2 + 4 + 6 + 8 + 10) = 1 / 2 := by
  sorry

end sum_ratio_equals_half_l2557_255724


namespace journey_speed_calculation_l2557_255730

/-- Proves that for a journey of given distance and original time,
    the average speed required to complete the same journey in a 
    multiple of the original time is as calculated. -/
theorem journey_speed_calculation 
  (distance : ℝ) 
  (original_time : ℝ) 
  (time_multiplier : ℝ) 
  (h1 : distance = 378) 
  (h2 : original_time = 6) 
  (h3 : time_multiplier = 3/2) :
  distance / (original_time * time_multiplier) = 42 :=
by
  sorry

#check journey_speed_calculation

end journey_speed_calculation_l2557_255730


namespace computer_table_price_l2557_255785

/-- Calculates the selling price given the cost price and markup percentage -/
def selling_price (cost_price : ℚ) (markup_percent : ℚ) : ℚ :=
  cost_price * (1 + markup_percent / 100)

/-- Proves that the selling price of a computer table with cost price 4480 and 25% markup is 5600 -/
theorem computer_table_price : selling_price 4480 25 = 5600 := by
  sorry

end computer_table_price_l2557_255785


namespace inequality_solution_l2557_255700

theorem inequality_solution (x : ℝ) : 
  (x ≠ 2 ∧ x ≠ 3 ∧ x ≠ 4 ∧ x ≠ 5) →
  (2 / (x - 2) - 3 / (x - 3) + 3 / (x - 4) - 2 / (x - 5) < 1 / 20 ↔
   x < -2 ∨ (-1 < x ∧ x < 2) ∨ (3 < x ∧ x < 4) ∨ (5 < x ∧ x < 7) ∨ x > 8) :=
by sorry

end inequality_solution_l2557_255700


namespace triangle_inequality_l2557_255787

theorem triangle_inequality (m : ℝ) : m > 0 → (3 + 4 > m ∧ 3 + m > 4 ∧ 4 + m > 3) → m = 5 := by
  sorry

#check triangle_inequality

end triangle_inequality_l2557_255787


namespace cube_volume_problem_l2557_255768

theorem cube_volume_problem (reference_cube_volume : ℝ) 
  (surface_area_ratio : ℝ) (target_cube_volume : ℝ) : 
  reference_cube_volume = 8 →
  surface_area_ratio = 3 →
  (6 * (reference_cube_volume ^ (1/3))^2) * surface_area_ratio = 6 * (target_cube_volume ^ (1/3))^2 →
  target_cube_volume = 24 * Real.sqrt 3 :=
by
  sorry

end cube_volume_problem_l2557_255768


namespace simplify_sqrt_sum_l2557_255731

theorem simplify_sqrt_sum : 
  Real.sqrt (12 + 8 * Real.sqrt 3) + Real.sqrt (12 - 8 * Real.sqrt 3) = 4 * Real.sqrt 3 := by
  sorry

end simplify_sqrt_sum_l2557_255731


namespace min_blocks_removed_for_cube_l2557_255706

/-- Given 59 cubic blocks, the minimum number of blocks that need to be taken away
    to construct a solid cube with none left over is 32. -/
theorem min_blocks_removed_for_cube (total_blocks : ℕ) (h : total_blocks = 59) :
  ∃ (n : ℕ), n^3 ≤ total_blocks ∧
             ∀ (m : ℕ), m^3 ≤ total_blocks → m ≤ n ∧
             total_blocks - n^3 = 32 :=
by sorry

end min_blocks_removed_for_cube_l2557_255706


namespace fruit_arrangements_proof_l2557_255777

def numFruitArrangements (apples oranges bananas totalDays : ℕ) : ℕ :=
  let bananasAsBlock := 1
  let nonBananaDays := totalDays - bananas + 1
  let arrangements := (Nat.factorial nonBananaDays) / 
    (Nat.factorial apples * Nat.factorial oranges * Nat.factorial bananasAsBlock)
  arrangements * nonBananaDays

theorem fruit_arrangements_proof :
  numFruitArrangements 4 3 3 10 = 2240 :=
by sorry

end fruit_arrangements_proof_l2557_255777


namespace eight_triangle_positions_l2557_255752

/-- A point on a 2D grid -/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- The area of a triangle given three grid points -/
def triangleArea (a b c : GridPoint) : ℚ :=
  (1/2) * |a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y)|

/-- Theorem: There are exactly 8 points C on the grid such that triangle ABC has area 4.5 -/
theorem eight_triangle_positions (a b : GridPoint) : 
  ∃! (s : Finset GridPoint), s.card = 8 ∧ 
    (∀ c ∈ s, triangleArea a b c = 4.5) ∧
    (∀ c : GridPoint, c ∉ s → triangleArea a b c ≠ 4.5) :=
sorry

end eight_triangle_positions_l2557_255752


namespace intersection_A_B_l2557_255738

def set_A : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.log (x - 1)}
def set_B : Set ℝ := {0, 1, 2, 3}

theorem intersection_A_B : set_A ∩ set_B = {2, 3} := by
  sorry

end intersection_A_B_l2557_255738


namespace train_crossing_time_l2557_255754

/-- Given a train traveling at a certain speed that crosses a platform of known length in a specific time,
    calculate the time it takes for the train to cross a man standing on the platform. -/
theorem train_crossing_time
  (train_speed_kmph : ℝ)
  (train_speed_mps : ℝ)
  (platform_length : ℝ)
  (platform_crossing_time : ℝ)
  (h1 : train_speed_kmph = 72)
  (h2 : train_speed_mps = 20)
  (h3 : platform_length = 220)
  (h4 : platform_crossing_time = 30)
  (h5 : train_speed_mps = train_speed_kmph * (1000 / 3600)) :
  (train_speed_mps * platform_crossing_time - platform_length) / train_speed_mps = 19 := by
  sorry

end train_crossing_time_l2557_255754


namespace max_rectangle_area_l2557_255714

/-- Given 40 feet of fencing for a rectangular pen, the maximum area enclosed is 100 square feet. -/
theorem max_rectangle_area (fencing : ℝ) (h : fencing = 40) :
  ∃ (length width : ℝ),
    length > 0 ∧ width > 0 ∧
    2 * (length + width) = fencing ∧
    ∀ (l w : ℝ), l > 0 → w > 0 → 2 * (l + w) = fencing → l * w ≤ length * width ∧
    length * width = 100 := by
  sorry

end max_rectangle_area_l2557_255714


namespace right_triangle_acute_angle_l2557_255755

theorem right_triangle_acute_angle (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) : 
  (a + b = 90) → (a / b = 7 / 2) → min a b = 20 := by
  sorry

end right_triangle_acute_angle_l2557_255755


namespace salt_solution_problem_l2557_255747

/-- Proves the initial water mass and percentage increase given final conditions --/
theorem salt_solution_problem (final_mass : ℝ) (final_concentration : ℝ) 
  (h_final_mass : final_mass = 850)
  (h_final_concentration : final_concentration = 0.36) : 
  ∃ (initial_mass : ℝ) (percentage_increase : ℝ),
    initial_mass = 544 ∧ 
    percentage_increase = 25 ∧
    final_mass = initial_mass * (1 + percentage_increase / 100)^2 ∧
    final_concentration = 1 - (initial_mass / final_mass) :=
by
  sorry


end salt_solution_problem_l2557_255747


namespace z_imaginary_and_fourth_quadrant_l2557_255799

def z (m : ℝ) : ℂ := m * (m + 2) + (m^2 + m - 2) * Complex.I

theorem z_imaginary_and_fourth_quadrant (m : ℝ) :
  (z m = Complex.I * Complex.im (z m) → m = 0) ∧
  (Complex.re (z m) > 0 ∧ Complex.im (z m) < 0 → 0 < m ∧ m < 1) :=
sorry

end z_imaginary_and_fourth_quadrant_l2557_255799


namespace gcd_12345_67890_l2557_255723

theorem gcd_12345_67890 : Nat.gcd 12345 67890 = 15 := by
  sorry

end gcd_12345_67890_l2557_255723


namespace opposite_of_negative_two_l2557_255779

theorem opposite_of_negative_two : 
  (∀ x : ℤ, x + (-x) = 0) → (-2 + 2 = 0) :=
by
  sorry

end opposite_of_negative_two_l2557_255779


namespace next_monday_birthday_l2557_255760

/-- Represents the day of the week -/
inductive DayOfWeek
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday
| Sunday

/-- Determines if a year is a leap year -/
def isLeapYear (year : Nat) : Bool :=
  year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)

/-- Calculates the day of the week for March 15 in a given year, 
    assuming March 15, 2012 was a Friday -/
def marchFifteenDayOfWeek (year : Nat) : DayOfWeek :=
  sorry

/-- Theorem: The next year after 2012 when March 15 falls on a Monday is 2025 -/
theorem next_monday_birthday (startYear : Nat) (startDay : DayOfWeek) :
  startYear = 2012 →
  startDay = DayOfWeek.Friday →
  (∀ y, startYear < y → y < 2025 → marchFifteenDayOfWeek y ≠ DayOfWeek.Monday) →
  marchFifteenDayOfWeek 2025 = DayOfWeek.Monday :=
by sorry

end next_monday_birthday_l2557_255760


namespace yellow_light_probability_l2557_255776

theorem yellow_light_probability (red_duration green_duration yellow_duration : ℕ) :
  red_duration = 30 →
  yellow_duration = 5 →
  green_duration = 45 →
  (yellow_duration : ℚ) / (red_duration + yellow_duration + green_duration) = 1 / 16 := by
  sorry

end yellow_light_probability_l2557_255776


namespace inequalities_hold_l2557_255763

theorem inequalities_hold (a b c x y z : ℝ) 
  (hx : x^2 < a^2) (hy : y^2 < b^2) (hz : z^2 < c^2) :
  (x^2*y^2 + y^2*z^2 + z^2*x^2 < a^2*b^2 + b^2*c^2 + c^2*a^2) ∧
  (x^4 + y^4 + z^4 < a^4 + b^4 + c^4) ∧
  (x^2*y^2*z^2 < a^2*b^2*c^2) := by
  sorry

end inequalities_hold_l2557_255763


namespace f_10_equals_10_l2557_255748

/-- An odd function satisfying certain conditions -/
def f (x : ℝ) : ℝ :=
  sorry

/-- Theorem stating the properties of f and the result to be proved -/
theorem f_10_equals_10 :
  (∀ x : ℝ, f (-x) = -f x) →  -- f is odd
  f 1 = 1 →  -- f(1) = 1
  (∀ x : ℝ, f (x + 2) = f x + f 2) →  -- f(x+2) = f(x) + f(2)
  f 10 = 10 :=
by sorry

end f_10_equals_10_l2557_255748


namespace bicycle_problem_l2557_255728

/-- Proves that given the conditions of the bicycle problem, student B's speed is 12 km/h -/
theorem bicycle_problem (distance : ℝ) (speed_ratio : ℝ) (time_difference : ℝ) :
  distance = 12 ∧
  speed_ratio = 1.2 ∧
  time_difference = 1/6 →
  ∃ (speed_B : ℝ),
    speed_B = 12 ∧
    distance / speed_B - distance / (speed_ratio * speed_B) = time_difference :=
by sorry

end bicycle_problem_l2557_255728


namespace first_night_rate_is_30_l2557_255739

/-- Represents the pricing structure of a guest house -/
structure GuestHousePricing where
  firstNightRate : ℕ  -- Flat rate for the first night
  additionalNightRate : ℕ  -- Fixed rate for each additional night

/-- Calculates the total cost for a stay -/
def totalCost (p : GuestHousePricing) (nights : ℕ) : ℕ :=
  p.firstNightRate + (nights - 1) * p.additionalNightRate

/-- Theorem stating that the flat rate for the first night is 30 -/
theorem first_night_rate_is_30 :
  ∃ (p : GuestHousePricing),
    totalCost p 4 = 210 ∧
    totalCost p 8 = 450 ∧
    p.firstNightRate = 30 := by
  sorry

end first_night_rate_is_30_l2557_255739


namespace smallest_angle_in_ratio_quadrilateral_l2557_255716

/-- 
Given a quadrilateral where the measures of interior angles are in the ratio 1:2:3:4,
prove that the measure of the smallest interior angle is 36°.
-/
theorem smallest_angle_in_ratio_quadrilateral : 
  ∀ (a b c d : ℝ),
  a > 0 → b > 0 → c > 0 → d > 0 →
  b = 2*a → c = 3*a → d = 4*a →
  a + b + c + d = 360 →
  a = 36 := by
sorry

end smallest_angle_in_ratio_quadrilateral_l2557_255716


namespace number_of_students_in_B_l2557_255749

/-- The number of students in school B -/
def students_B : ℕ := 30

/-- The number of students in school A -/
def students_A : ℕ := 4 * students_B

/-- The number of students in school C -/
def students_C : ℕ := 3 * students_B

/-- Theorem stating that the number of students in school B is 30 -/
theorem number_of_students_in_B : 
  students_A + students_C = 210 → students_B = 30 := by
  sorry


end number_of_students_in_B_l2557_255749


namespace initial_mat_weavers_l2557_255744

theorem initial_mat_weavers : ℕ :=
  let initial_weavers : ℕ := sorry
  let initial_mats : ℕ := 4
  let initial_days : ℕ := 4
  let second_weavers : ℕ := 14
  let second_mats : ℕ := 49
  let second_days : ℕ := 14

  have h1 : initial_weavers * initial_days * second_mats = second_weavers * second_days * initial_mats := by sorry

  have h2 : initial_weavers = 4 := by sorry

  4


end initial_mat_weavers_l2557_255744


namespace square_equation_solution_l2557_255797

theorem square_equation_solution : ∃ x : ℝ, (12 - x)^2 = (x + 3)^2 ∧ x = 9/2 := by
  sorry

end square_equation_solution_l2557_255797
