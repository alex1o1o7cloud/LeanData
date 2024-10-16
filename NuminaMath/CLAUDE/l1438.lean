import Mathlib

namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1438_143805

/-- A sequence is geometric if the ratio of successive terms is constant. -/
def IsGeometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Given a geometric sequence a_n where a_3 * a_5 * a_7 = (-√3)^3, prove a_2 * a_8 = 3 -/
theorem geometric_sequence_property (a : ℕ → ℝ) 
    (h_geometric : IsGeometric a) 
    (h_product : a 3 * a 5 * a 7 = (-Real.sqrt 3)^3) : 
  a 2 * a 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1438_143805


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_sum_l1438_143892

theorem quadratic_equation_solution_sum : ∃ (c d : ℝ), 
  (c^2 - 6*c + 15 = 25) ∧ 
  (d^2 - 6*d + 15 = 25) ∧ 
  (c ≥ d) ∧ 
  (3*c + 2*d = 15 + Real.sqrt 19) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_sum_l1438_143892


namespace NUMINAMATH_CALUDE_function_characterization_l1438_143848

theorem function_characterization
  (f : ℕ → ℕ)
  (α : ℕ)
  (h1 : ∀ (m n : ℕ), f (m * n^2) = f (m * n) + α * f n)
  (h2 : ∀ (n : ℕ) (p : ℕ), Nat.Prime p → p ∣ n → f p ≠ 0 ∧ f p ∣ f n) :
  ∃ (c : ℕ), 
    (α = 1) ∧
    (∀ (n : ℕ), 
      f n = c * (Nat.factorization n).sum (fun _ e => e)) :=
sorry

end NUMINAMATH_CALUDE_function_characterization_l1438_143848


namespace NUMINAMATH_CALUDE_remaining_money_after_ticket_l1438_143894

def octal_to_decimal (n : ℕ) : ℕ := sorry

theorem remaining_money_after_ticket : 
  let savings := octal_to_decimal 5376
  let ticket_cost := 1200
  savings - ticket_cost = 1614 := by
  sorry

end NUMINAMATH_CALUDE_remaining_money_after_ticket_l1438_143894


namespace NUMINAMATH_CALUDE_f_min_at_zero_l1438_143858

def f (x : ℝ) : ℝ := (x^2 - 4)^3 + 1

theorem f_min_at_zero :
  (∀ x : ℝ, f 0 ≤ f x) ∧ f 0 = -63 :=
sorry

end NUMINAMATH_CALUDE_f_min_at_zero_l1438_143858


namespace NUMINAMATH_CALUDE_cookie_store_spending_l1438_143888

theorem cookie_store_spending (ben david : ℝ) 
  (h1 : david = ben / 2)
  (h2 : ben = david + 20) : 
  ben + david = 60 := by
sorry

end NUMINAMATH_CALUDE_cookie_store_spending_l1438_143888


namespace NUMINAMATH_CALUDE_point_movement_theorem_point_M_movement_l1438_143861

/-- Represents a point on a number line -/
structure Point where
  value : ℝ

/-- Moves a point on the number line -/
def move (p : Point) (distance : ℝ) : Point :=
  ⟨p.value + distance⟩

theorem point_movement_theorem (M N : Point) :
  (M.value = 9) →
  (move (move N (-4)) 6 = M) →
  N.value = 7 :=
sorry

theorem point_M_movement (M : Point) :
  (M.value = 9) →
  (∃ (new_M : Point), (move M 4 = new_M ∨ move M (-4) = new_M) ∧ (new_M.value = 5 ∨ new_M.value = 13)) :=
sorry

end NUMINAMATH_CALUDE_point_movement_theorem_point_M_movement_l1438_143861


namespace NUMINAMATH_CALUDE_cos_arcsin_three_fifths_l1438_143829

theorem cos_arcsin_three_fifths : Real.cos (Real.arcsin (3/5)) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_arcsin_three_fifths_l1438_143829


namespace NUMINAMATH_CALUDE_seconds_in_week_l1438_143832

/-- The number of seconds in a minute -/
def seconds_per_minute : ℕ := 60

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- The number of hours in a day -/
def hours_per_day : ℕ := 24

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- Theorem: The product of seconds per minute, minutes per hour, hours per day, and days per week
    equals the number of seconds in a week -/
theorem seconds_in_week :
  seconds_per_minute * minutes_per_hour * hours_per_day * days_per_week =
  (seconds_per_minute * minutes_per_hour * hours_per_day) * days_per_week :=
by sorry

end NUMINAMATH_CALUDE_seconds_in_week_l1438_143832


namespace NUMINAMATH_CALUDE_lock_settings_count_l1438_143817

/-- The number of digits on each dial of the lock -/
def numDigits : ℕ := 10

/-- The number of dials on the lock -/
def numDials : ℕ := 4

/-- The set of all possible digits -/
def digitSet : Finset ℕ := Finset.range numDigits

/-- The set of valid first digits (excluding zero) -/
def validFirstDigits : Finset ℕ := digitSet.filter (λ x => x ≠ 0)

/-- The number of different settings possible for the lock -/
def numSettings : ℕ := validFirstDigits.card * (numDigits - 1) * (numDigits - 2) * (numDigits - 3)

theorem lock_settings_count :
  numSettings = 4536 :=
sorry

end NUMINAMATH_CALUDE_lock_settings_count_l1438_143817


namespace NUMINAMATH_CALUDE_area_to_paint_l1438_143812

/-- The area of the wall that Jane needs to paint, given the dimensions of the wall and the door. -/
theorem area_to_paint (wall_height wall_length door_width door_height : ℝ) 
  (h_wall_height : wall_height = 10)
  (h_wall_length : wall_length = 15)
  (h_door_width : door_width = 3)
  (h_door_height : door_height = 5) :
  wall_height * wall_length - door_width * door_height = 135 := by
  sorry

#check area_to_paint

end NUMINAMATH_CALUDE_area_to_paint_l1438_143812


namespace NUMINAMATH_CALUDE_talia_father_age_l1438_143856

/-- Represents the ages of Talia and her parents -/
structure FamilyAges where
  talia : ℕ
  mom : ℕ
  dad : ℕ

/-- Conditions given in the problem -/
def problem_conditions (ages : FamilyAges) : Prop :=
  ages.talia + 7 = 20 ∧
  ages.mom = 3 * ages.talia ∧
  ages.dad + 3 = ages.mom

/-- Theorem stating that Talia's father is 36 years old -/
theorem talia_father_age (ages : FamilyAges) :
  problem_conditions ages → ages.dad = 36 := by
  sorry


end NUMINAMATH_CALUDE_talia_father_age_l1438_143856


namespace NUMINAMATH_CALUDE_sons_age_l1438_143884

/-- Given a father and son, where the father is 46 years older than the son,
    and in two years the father's age will be twice the son's age,
    prove that the son's current age is 44 years. -/
theorem sons_age (son_age father_age : ℕ) : 
  father_age = son_age + 46 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 44 := by sorry

end NUMINAMATH_CALUDE_sons_age_l1438_143884


namespace NUMINAMATH_CALUDE_canister_capacity_ratio_l1438_143822

/-- Represents the ratio of capacities between two canisters -/
structure CanisterRatio where
  c : ℝ  -- Capacity of canister C
  d : ℝ  -- Capacity of canister D

/-- Theorem stating the ratio of canister capacities given the problem conditions -/
theorem canister_capacity_ratio (r : CanisterRatio) 
  (hc_half : r.c / 2 = r.c - (r.d / 3 - r.d / 12)) 
  (hd_third : r.d / 3 > 0) 
  (hc_positive : r.c > 0) 
  (hd_positive : r.d > 0) :
  r.d / r.c = 2 := by
  sorry

end NUMINAMATH_CALUDE_canister_capacity_ratio_l1438_143822


namespace NUMINAMATH_CALUDE_second_train_speed_is_16_l1438_143849

/-- The speed of the second train given the conditions of the problem -/
def second_train_speed (first_train_speed : ℝ) (distance_between_stations : ℝ) (distance_difference : ℝ) : ℝ :=
  let v : ℝ := 16  -- Speed of the second train
  v

/-- Theorem stating that under the given conditions, the speed of the second train is 16 km/hr -/
theorem second_train_speed_is_16 :
  second_train_speed 20 495 55 = 16 := by
  sorry

#check second_train_speed_is_16

end NUMINAMATH_CALUDE_second_train_speed_is_16_l1438_143849


namespace NUMINAMATH_CALUDE_odd_sum_power_divisibility_l1438_143880

theorem odd_sum_power_divisibility
  (a b l : ℕ) 
  (h_odd_a : Odd a) 
  (h_odd_b : Odd b)
  (h_a_gt_1 : a > 1)
  (h_b_gt_1 : b > 1)
  (h_sum : a + b = 2^l) :
  ∀ k : ℕ, k > 0 → (k^2 ∣ a^k + b^k) → k = 1 :=
sorry

end NUMINAMATH_CALUDE_odd_sum_power_divisibility_l1438_143880


namespace NUMINAMATH_CALUDE_friendly_angle_values_l1438_143828

-- Define a friendly triangle
def is_friendly_triangle (a b c : ℝ) : Prop :=
  a + b + c = 180 ∧ (a = 2*b ∨ b = 2*c ∨ c = 2*a)

-- Theorem statement
theorem friendly_angle_values :
  ∀ a b c : ℝ,
  is_friendly_triangle a b c →
  (a = 42 ∨ b = 42 ∨ c = 42) →
  (a = 42 ∨ a = 84 ∨ a = 92 ∨
   b = 42 ∨ b = 84 ∨ b = 92 ∨
   c = 42 ∨ c = 84 ∨ c = 92) :=
by sorry

end NUMINAMATH_CALUDE_friendly_angle_values_l1438_143828


namespace NUMINAMATH_CALUDE_smallest_positive_solution_l1438_143816

theorem smallest_positive_solution (x : ℝ) : 
  (x^4 - 40*x^2 + 400 = 0 ∧ x > 0 ∧ ∀ y > 0, y^4 - 40*y^2 + 400 = 0 → x ≤ y) → 
  x = 2 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_smallest_positive_solution_l1438_143816


namespace NUMINAMATH_CALUDE_somu_age_problem_l1438_143865

theorem somu_age_problem (somu_age father_age : ℕ) : 
  somu_age = father_age / 3 →
  somu_age - 5 = (father_age - 5) / 5 →
  somu_age = 10 := by
sorry

end NUMINAMATH_CALUDE_somu_age_problem_l1438_143865


namespace NUMINAMATH_CALUDE_f_2018_equals_neg_8_l1438_143886

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem f_2018_equals_neg_8 (f : ℝ → ℝ) 
  (h1 : is_even f)
  (h2 : ∀ x, f (x + 3) = -1 / f x)
  (h3 : ∀ x ∈ Set.Icc (-3) (-2), f x = 4 * x) :
  f 2018 = -8 := by
  sorry

end NUMINAMATH_CALUDE_f_2018_equals_neg_8_l1438_143886


namespace NUMINAMATH_CALUDE_fish_gone_bad_percentage_l1438_143854

theorem fish_gone_bad_percentage (fish_per_roll fish_bought rolls_made : ℕ) 
  (h1 : fish_per_roll = 40)
  (h2 : fish_bought = 400)
  (h3 : rolls_made = 8) :
  (fish_bought - rolls_made * fish_per_roll) / fish_bought * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_fish_gone_bad_percentage_l1438_143854


namespace NUMINAMATH_CALUDE_correct_prices_l1438_143808

/-- Prices of items in a shopping scenario -/
def shopping_prices (total belt pants shirt shoes : ℝ) : Prop :=
  -- Total cost condition
  total = belt + pants + shirt + shoes ∧
  -- Pants price condition
  pants = belt - 2.93 ∧
  -- Shirt price condition
  shirt = 1.5 * pants ∧
  -- Shoes price condition
  shoes = 3 * shirt

/-- Theorem stating the correct prices for the shopping scenario -/
theorem correct_prices : 
  ∃ (belt pants shirt shoes : ℝ),
    shopping_prices 205.93 belt pants shirt shoes ∧ 
    belt = 28.305 ∧ 
    pants = 25.375 ∧ 
    shirt = 38.0625 ∧ 
    shoes = 114.1875 :=
by
  sorry

end NUMINAMATH_CALUDE_correct_prices_l1438_143808


namespace NUMINAMATH_CALUDE_series_sum_equals_399002_l1438_143891

/-- The sum of the series 1-2-3+4+5-6-7+8+9-10-11+12+13-...-1994-1995+1996+1997 -/
def seriesSum : ℕ → ℤ
  | 0 => 0
  | n + 1 => seriesSum n + term (n + 1)
where
  term : ℕ → ℤ
  | n => if n % 5 ≤ 2 then -(n : ℤ) else (n : ℤ)

theorem series_sum_equals_399002 : seriesSum 1997 = 399002 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_equals_399002_l1438_143891


namespace NUMINAMATH_CALUDE_coltons_remaining_stickers_l1438_143826

/-- The number of stickers Colton has left after giving some to his friends. -/
def stickers_left (initial : ℕ) (per_friend : ℕ) (num_friends : ℕ) (mandy_extra : ℕ) (justin_less : ℕ) : ℕ :=
  let friends_total := per_friend * num_friends
  let mandy_stickers := friends_total + mandy_extra
  let justin_stickers := mandy_stickers - justin_less
  initial - (friends_total + mandy_stickers + justin_stickers)

/-- Theorem stating that Colton has 42 stickers left given the problem conditions. -/
theorem coltons_remaining_stickers :
  stickers_left 72 4 3 2 10 = 42 := by
  sorry

end NUMINAMATH_CALUDE_coltons_remaining_stickers_l1438_143826


namespace NUMINAMATH_CALUDE_jake_debt_l1438_143857

/-- The amount Jake originally owed given his payment and work details --/
def original_debt (prepaid_amount : ℕ) (hourly_rate : ℕ) (hours_worked : ℕ) : ℕ :=
  prepaid_amount + hourly_rate * hours_worked

/-- Theorem stating that Jake's original debt was $100 --/
theorem jake_debt : original_debt 40 15 4 = 100 := by
  sorry

end NUMINAMATH_CALUDE_jake_debt_l1438_143857


namespace NUMINAMATH_CALUDE_complex_power_magnitude_l1438_143872

theorem complex_power_magnitude : Complex.abs ((1 - Complex.I * 2) ^ 8) = 625 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_magnitude_l1438_143872


namespace NUMINAMATH_CALUDE_milk_price_proof_l1438_143864

/-- The original price of milk before discount -/
def milk_original_price : ℝ := 10

/-- Lily's initial budget -/
def initial_budget : ℝ := 60

/-- Cost of celery -/
def celery_cost : ℝ := 5

/-- Original price of cereal -/
def cereal_original_price : ℝ := 12

/-- Discount rate for cereal -/
def cereal_discount_rate : ℝ := 0.5

/-- Cost of bread -/
def bread_cost : ℝ := 8

/-- Discount rate for milk -/
def milk_discount_rate : ℝ := 0.1

/-- Cost of one potato -/
def potato_unit_cost : ℝ := 1

/-- Number of potatoes bought -/
def potato_quantity : ℕ := 6

/-- Amount left after buying all items -/
def amount_left : ℝ := 26

theorem milk_price_proof :
  let cereal_cost := cereal_original_price * (1 - cereal_discount_rate)
  let potato_cost := potato_unit_cost * potato_quantity
  let other_items_cost := celery_cost + cereal_cost + bread_cost + potato_cost
  let total_spent := initial_budget - amount_left
  let milk_discounted_price := total_spent - other_items_cost
  milk_original_price = milk_discounted_price / (1 - milk_discount_rate) :=
by sorry

end NUMINAMATH_CALUDE_milk_price_proof_l1438_143864


namespace NUMINAMATH_CALUDE_point_on_line_implies_m_value_l1438_143802

/-- Given a point P(1, -2) on the line 4x - my + 12 = 0, prove that m = -8 -/
theorem point_on_line_implies_m_value (m : ℝ) : 
  (4 * 1 - m * (-2) + 12 = 0) → m = -8 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_implies_m_value_l1438_143802


namespace NUMINAMATH_CALUDE_infinitely_many_even_floor_squares_l1438_143803

theorem infinitely_many_even_floor_squares (α : ℝ) (h : α > 0) :
  Set.Infinite {n : ℕ+ | Even ⌊(n : ℝ)^2 * α⌋} := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_even_floor_squares_l1438_143803


namespace NUMINAMATH_CALUDE_winnie_the_pooh_honey_l1438_143881

theorem winnie_the_pooh_honey (a b c d e : ℝ) 
  (non_neg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ e ≥ 0) 
  (total : a + b + c + d + e = 3) : 
  max (a + b) (max (b + c) (max (c + d) (d + e))) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_winnie_the_pooh_honey_l1438_143881


namespace NUMINAMATH_CALUDE_church_seating_problem_l1438_143870

/-- 
Proves that the number of chairs in each row is 6, given the conditions of the church seating problem.
-/
theorem church_seating_problem (rows : ℕ) (people_per_chair : ℕ) (total_capacity : ℕ) 
  (h1 : rows = 20)
  (h2 : people_per_chair = 5)
  (h3 : total_capacity = 600) :
  (total_capacity / (rows * people_per_chair) : ℚ) = 6 := by
  sorry

end NUMINAMATH_CALUDE_church_seating_problem_l1438_143870


namespace NUMINAMATH_CALUDE_restaurant_serving_totals_l1438_143814

/-- Represents the number of food items served at a meal -/
structure MealServing :=
  (hotDogs : ℕ)
  (hamburgers : ℕ)
  (sandwiches : ℕ)
  (salads : ℕ)

/-- Represents the meals served in a day -/
structure DayMeals :=
  (breakfast : MealServing)
  (lunch : MealServing)
  (dinner : MealServing)

def day1 : DayMeals := {
  breakfast := { hotDogs := 15, hamburgers := 8, sandwiches := 6, salads := 10 },
  lunch := { hotDogs := 20, hamburgers := 18, sandwiches := 12, salads := 15 },
  dinner := { hotDogs := 4, hamburgers := 10, sandwiches := 12, salads := 5 }
}

def day2 : DayMeals := {
  breakfast := { hotDogs := 6, hamburgers := 12, sandwiches := 9, salads := 7 },
  lunch := { hotDogs := 10, hamburgers := 20, sandwiches := 16, salads := 12 },
  dinner := { hotDogs := 3, hamburgers := 7, sandwiches := 5, salads := 8 }
}

def day3 : DayMeals := {
  breakfast := { hotDogs := 10, hamburgers := 14, sandwiches := 8, salads := 6 },
  lunch := { hotDogs := 12, hamburgers := 16, sandwiches := 10, salads := 9 },
  dinner := { hotDogs := 8, hamburgers := 9, sandwiches := 7, salads := 10 }
}

theorem restaurant_serving_totals :
  let breakfastLunchTotal := (day1.breakfast.hotDogs + day1.lunch.hotDogs + 
                              day2.breakfast.hotDogs + day2.lunch.hotDogs + 
                              day3.breakfast.hotDogs + day3.lunch.hotDogs) +
                             (day1.breakfast.hamburgers + day1.lunch.hamburgers + 
                              day2.breakfast.hamburgers + day2.lunch.hamburgers + 
                              day3.breakfast.hamburgers + day3.lunch.hamburgers) +
                             (day1.breakfast.sandwiches + day1.lunch.sandwiches + 
                              day2.breakfast.sandwiches + day2.lunch.sandwiches + 
                              day3.breakfast.sandwiches + day3.lunch.sandwiches)
  let saladTotal := day1.breakfast.salads + day1.lunch.salads + day1.dinner.salads +
                    day2.breakfast.salads + day2.lunch.salads + day2.dinner.salads +
                    day3.breakfast.salads + day3.lunch.salads + day3.dinner.salads
  breakfastLunchTotal = 222 ∧ saladTotal = 82 := by
  sorry


end NUMINAMATH_CALUDE_restaurant_serving_totals_l1438_143814


namespace NUMINAMATH_CALUDE_highway_extension_ratio_l1438_143830

/-- The ratio of miles built on the second day to the first day of highway extension -/
theorem highway_extension_ratio :
  let current_length : ℕ := 200
  let extended_length : ℕ := 650
  let first_day_miles : ℕ := 50
  let miles_remaining : ℕ := 250
  let second_day_miles : ℕ := extended_length - current_length - first_day_miles - miles_remaining
  (second_day_miles : ℚ) / first_day_miles = 3 / 1 :=
by sorry

end NUMINAMATH_CALUDE_highway_extension_ratio_l1438_143830


namespace NUMINAMATH_CALUDE_yellow_red_paper_area_comparison_l1438_143804

theorem yellow_red_paper_area_comparison (x : ℝ) (h : x > 0) :
  let yellow_area := 2 * x
  let larger_part := x / (1 - 0.25)
  let smaller_part := yellow_area - larger_part
  (x - smaller_part) / smaller_part = 0.5
  := by sorry

end NUMINAMATH_CALUDE_yellow_red_paper_area_comparison_l1438_143804


namespace NUMINAMATH_CALUDE_penny_splitting_game_result_l1438_143862

/-- Represents the result of the penny splitting game. -/
inductive GameResult
  | FirstPlayerWins
  | SecondPlayerWins

/-- The penny splitting game. -/
def pennySplittingGame (n : ℕ) : GameResult :=
  sorry

/-- Theorem stating the conditions for each player's victory. -/
theorem penny_splitting_game_result (n : ℕ) (h : n ≥ 3) :
  pennySplittingGame n = 
    if n = 3 ∨ n % 2 = 0 then
      GameResult.FirstPlayerWins
    else
      GameResult.SecondPlayerWins :=
  sorry

end NUMINAMATH_CALUDE_penny_splitting_game_result_l1438_143862


namespace NUMINAMATH_CALUDE_polynomial_expansion_l1438_143831

theorem polynomial_expansion (t : ℝ) :
  (3*t^3 + 2*t^2 - 4*t + 3) * (-4*t^3 + 3*t - 5) =
  -12*t^6 - 8*t^5 + 25*t^4 - 21*t^3 - 22*t^2 + 29*t - 15 := by
sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l1438_143831


namespace NUMINAMATH_CALUDE_candy_box_price_increase_l1438_143885

theorem candy_box_price_increase (new_price : ℝ) (increase_rate : ℝ) (original_price : ℝ) :
  new_price = 20 ∧ increase_rate = 0.25 ∧ new_price = original_price * (1 + increase_rate) →
  original_price = 16 := by
  sorry

end NUMINAMATH_CALUDE_candy_box_price_increase_l1438_143885


namespace NUMINAMATH_CALUDE_intersection_rectangular_prisms_cubes_l1438_143811

-- Define the set of all rectangular prisms
def rectangular_prisms : Set (ℝ × ℝ × ℝ) := {p | ∃ l w h, p = (l, w, h) ∧ l > 0 ∧ w > 0 ∧ h > 0}

-- Define the set of all cubes
def cubes : Set (ℝ × ℝ × ℝ) := {c | ∃ s, c = (s, s, s) ∧ s > 0}

-- Theorem statement
theorem intersection_rectangular_prisms_cubes :
  rectangular_prisms ∩ cubes = cubes :=
by sorry

end NUMINAMATH_CALUDE_intersection_rectangular_prisms_cubes_l1438_143811


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1438_143840

theorem geometric_sequence_problem (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- geometric sequence definition
  q > 0 →  -- positive common ratio
  a 1 = 2 →  -- given condition
  4 * a 2 * a 8 = (a 4) ^ 2 →  -- given condition
  a 3 = (1 : ℝ) / 2 := by  -- conclusion to prove
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1438_143840


namespace NUMINAMATH_CALUDE_twenty_times_nineteen_plus_twenty_plus_nineteen_l1438_143836

theorem twenty_times_nineteen_plus_twenty_plus_nineteen : 20 * 19 + 20 + 19 = 419 := by
  sorry

end NUMINAMATH_CALUDE_twenty_times_nineteen_plus_twenty_plus_nineteen_l1438_143836


namespace NUMINAMATH_CALUDE_slope_of_line_l1438_143819

theorem slope_of_line (x y : ℝ) : 4 * y = -6 * x + 12 → (y - 3) = (-3/2) * (x - 0) := by
  sorry

end NUMINAMATH_CALUDE_slope_of_line_l1438_143819


namespace NUMINAMATH_CALUDE_pizza_toppings_combinations_l1438_143866

theorem pizza_toppings_combinations : Nat.choose 8 3 = 56 := by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_combinations_l1438_143866


namespace NUMINAMATH_CALUDE_expected_vote_percentage_is_47_percent_l1438_143851

/-- The percentage of registered voters who are Democrats -/
def democrat_percentage : ℝ := 0.60

/-- The percentage of registered voters who are Republicans -/
def republican_percentage : ℝ := 1 - democrat_percentage

/-- The percentage of registered Democrat voters expected to vote for candidate A -/
def democrat_vote_percentage : ℝ := 0.65

/-- The percentage of registered Republican voters expected to vote for candidate A -/
def republican_vote_percentage : ℝ := 0.20

/-- The expected percentage of registered voters who will vote for candidate A -/
def expected_vote_percentage : ℝ :=
  democrat_percentage * democrat_vote_percentage +
  republican_percentage * republican_vote_percentage

theorem expected_vote_percentage_is_47_percent :
  expected_vote_percentage = 0.47 :=
sorry

end NUMINAMATH_CALUDE_expected_vote_percentage_is_47_percent_l1438_143851


namespace NUMINAMATH_CALUDE_plumber_max_shower_charge_l1438_143835

def plumber_problem (sink_charge toilet_charge shower_charge : ℕ) : Prop :=
  let job1 := 3 * toilet_charge + 3 * sink_charge
  let job2 := 2 * toilet_charge + 5 * sink_charge
  let job3 := toilet_charge + 2 * shower_charge + 3 * sink_charge
  sink_charge = 30 ∧
  toilet_charge = 50 ∧
  (job1 ≤ 250 ∧ job2 ≤ 250 ∧ job3 ≤ 250) ∧
  (job1 = 250 ∨ job2 = 250 ∨ job3 = 250) →
  shower_charge ≤ 55

theorem plumber_max_shower_charge :
  ∃ (shower_charge : ℕ), plumber_problem 30 50 shower_charge ∧
  ∀ (x : ℕ), x > shower_charge → ¬ plumber_problem 30 50 x :=
sorry

end NUMINAMATH_CALUDE_plumber_max_shower_charge_l1438_143835


namespace NUMINAMATH_CALUDE_sum_seven_terms_l1438_143834

/-- An arithmetic sequence with specific terms. -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n
  second_term : a 2 = 5 / 3
  sixth_term : a 6 = -7 / 3

/-- The sum of the first n terms of an arithmetic sequence. -/
def sum_n_terms (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

/-- The theorem stating that the sum of the first 7 terms is -7/3. -/
theorem sum_seven_terms (seq : ArithmeticSequence) : sum_n_terms seq 7 = -7 / 3 := by
  sorry


end NUMINAMATH_CALUDE_sum_seven_terms_l1438_143834


namespace NUMINAMATH_CALUDE_window_treatment_cost_for_three_windows_l1438_143800

/-- The cost of window treatments for a given number of windows -/
def window_treatment_cost (num_windows : ℕ) (sheer_cost drape_cost : ℚ) : ℚ :=
  num_windows * (sheer_cost + drape_cost)

/-- Theorem: The cost of window treatments for 3 windows with sheers at $40.00 and drapes at $60.00 is $300.00 -/
theorem window_treatment_cost_for_three_windows :
  window_treatment_cost 3 40 60 = 300 := by
  sorry

end NUMINAMATH_CALUDE_window_treatment_cost_for_three_windows_l1438_143800


namespace NUMINAMATH_CALUDE_carlos_class_size_l1438_143842

theorem carlos_class_size (n : ℕ) (carlos : ℕ) :
  (carlos = 75) →
  (n - carlos = 74) →
  (carlos - 1 = 74) →
  n = 149 := by
  sorry

end NUMINAMATH_CALUDE_carlos_class_size_l1438_143842


namespace NUMINAMATH_CALUDE_pencil_distribution_l1438_143847

theorem pencil_distribution (num_students : ℕ) (num_pencils : ℕ) 
  (h1 : num_students = 2) 
  (h2 : num_pencils = 18) :
  num_pencils / num_students = 9 := by
  sorry

end NUMINAMATH_CALUDE_pencil_distribution_l1438_143847


namespace NUMINAMATH_CALUDE_angle_is_three_pi_over_four_l1438_143820

def angle_between_vectors (a b : ℝ × ℝ) : ℝ := sorry

theorem angle_is_three_pi_over_four (a b : ℝ × ℝ) 
  (h1 : a.fst * (a.fst - 2 * b.fst) + a.snd * (a.snd - 2 * b.snd) = 3)
  (h2 : a.fst^2 + a.snd^2 = 1)
  (h3 : b = (1, 1)) :
  angle_between_vectors a b = 3 * π / 4 := by sorry

end NUMINAMATH_CALUDE_angle_is_three_pi_over_four_l1438_143820


namespace NUMINAMATH_CALUDE_resort_tips_l1438_143821

theorem resort_tips (total_months : ℕ) (other_months : ℕ) (avg_other_tips : ℝ) (aug_tips : ℝ) :
  total_months = other_months + 1 →
  aug_tips = 0.5 * (aug_tips + other_months * avg_other_tips) →
  aug_tips = 6 * avg_other_tips :=
by
  sorry

end NUMINAMATH_CALUDE_resort_tips_l1438_143821


namespace NUMINAMATH_CALUDE_equation_solution_l1438_143899

theorem equation_solution (n : ℚ) :
  (2 / (n + 2) + 4 / (n + 2) + n / (n + 2) = 4) → n = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1438_143899


namespace NUMINAMATH_CALUDE_second_grade_survey_count_l1438_143860

/-- Calculates the number of students to be surveyed from the second grade in a stratified sampling method. -/
theorem second_grade_survey_count
  (total_students : ℕ)
  (grade_ratio : Fin 3 → ℕ)
  (total_surveyed : ℕ)
  (h1 : total_students = 1500)
  (h2 : grade_ratio 0 = 4 ∧ grade_ratio 1 = 5 ∧ grade_ratio 2 = 6)
  (h3 : total_surveyed = 150) :
  (total_surveyed * grade_ratio 1) / (grade_ratio 0 + grade_ratio 1 + grade_ratio 2) = 50 :=
by sorry

end NUMINAMATH_CALUDE_second_grade_survey_count_l1438_143860


namespace NUMINAMATH_CALUDE_batsman_new_average_is_35_l1438_143895

/-- Represents a batsman's score history -/
structure Batsman where
  previousInnings : Nat
  previousTotalScore : Nat
  newInningScore : Nat
  averageIncrease : Nat

/-- Calculates the new average after the latest inning -/
def newAverage (b : Batsman) : Nat :=
  (b.previousTotalScore + b.newInningScore) / (b.previousInnings + 1)

/-- Theorem: Given the conditions, prove that the new average is 35 -/
theorem batsman_new_average_is_35 (b : Batsman)
  (h1 : b.previousInnings = 10)
  (h2 : b.newInningScore = 85)
  (h3 : b.averageIncrease = 5)
  (h4 : newAverage b = (b.previousTotalScore / b.previousInnings) + b.averageIncrease) :
  newAverage b = 35 := by
  sorry

#eval newAverage { previousInnings := 10, previousTotalScore := 300, newInningScore := 85, averageIncrease := 5 }

end NUMINAMATH_CALUDE_batsman_new_average_is_35_l1438_143895


namespace NUMINAMATH_CALUDE_max_value_of_expression_l1438_143844

theorem max_value_of_expression (n : ℤ) (h1 : 100 ≤ n) (h2 : n ≤ 999) :
  3 * (500 - n) ≤ 1200 ∧ ∃ (m : ℤ), 100 ≤ m ∧ m ≤ 999 ∧ 3 * (500 - m) = 1200 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l1438_143844


namespace NUMINAMATH_CALUDE_square_root_fourth_power_l1438_143879

theorem square_root_fourth_power (x : ℝ) : (Real.sqrt x)^4 = 256 → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_square_root_fourth_power_l1438_143879


namespace NUMINAMATH_CALUDE_C_power_50_l1438_143859

def C : Matrix (Fin 2) (Fin 2) ℤ := !![3, 1; -4, -1]

theorem C_power_50 : C^50 = !![101, 50; -200, -99] := by sorry

end NUMINAMATH_CALUDE_C_power_50_l1438_143859


namespace NUMINAMATH_CALUDE_ratio_difference_l1438_143813

theorem ratio_difference (a b c d : ℝ) : 
  a / b = 5.5 / 7.25 →
  b / c = 7.25 / 11.75 →
  c / d = 11.75 / 13.5 →
  c = 94 →
  b - d = -50 := by
sorry

end NUMINAMATH_CALUDE_ratio_difference_l1438_143813


namespace NUMINAMATH_CALUDE_f_properties_l1438_143874

noncomputable def f (x φ : ℝ) : ℝ :=
  (1/2) * Real.sin (2*x) * Real.sin φ + Real.cos x ^ 2 * Real.cos φ + (1/2) * Real.sin (3*Real.pi/2 - φ)

theorem f_properties (φ : ℝ) (h1 : 0 < φ) (h2 : φ < Real.pi) (h3 : f (Real.pi/6) φ = 1/2) :
  (∀ x ∈ Set.Icc (Real.pi/6) ((2*Real.pi)/3), StrictMonoOn f (Set.Icc (Real.pi/6) ((2*Real.pi)/3))) ∧
  (∀ x₀ : ℝ, x₀ ∈ Set.Ioo (Real.pi/2) Real.pi → Real.sin x₀ = 3/5 → f x₀ φ = (7 - 24*Real.sqrt 3) / 100) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1438_143874


namespace NUMINAMATH_CALUDE_square_sum_zero_implies_both_zero_l1438_143845

theorem square_sum_zero_implies_both_zero (a b : ℝ) : a^2 + b^2 = 0 → a = 0 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_zero_implies_both_zero_l1438_143845


namespace NUMINAMATH_CALUDE_imaginary_part_of_2_plus_i_l1438_143825

theorem imaginary_part_of_2_plus_i : Complex.im (2 + Complex.I) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_2_plus_i_l1438_143825


namespace NUMINAMATH_CALUDE_zoo_visitors_l1438_143873

theorem zoo_visitors (monday_children monday_adults tuesday_children : ℕ)
  (child_ticket_price adult_ticket_price : ℕ)
  (total_revenue : ℕ) :
  monday_children = 7 →
  monday_adults = 5 →
  tuesday_children = 4 →
  child_ticket_price = 3 →
  adult_ticket_price = 4 →
  total_revenue = 61 →
  ∃ tuesday_adults : ℕ,
    total_revenue =
      monday_children * child_ticket_price +
      monday_adults * adult_ticket_price +
      tuesday_children * child_ticket_price +
      tuesday_adults * adult_ticket_price ∧
    tuesday_adults = 2 :=
by sorry

end NUMINAMATH_CALUDE_zoo_visitors_l1438_143873


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_ratio_l1438_143843

theorem quadratic_equation_roots_ratio (k : ℝ) : 
  (∃ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ x = 3*y ∧ 
   x^2 + 10*x + k = 0 ∧ y^2 + 10*y + k = 0) → 
  k = 18.75 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_ratio_l1438_143843


namespace NUMINAMATH_CALUDE_difference_of_roots_quadratic_l1438_143883

theorem difference_of_roots_quadratic (x : ℝ) : 
  let roots := {r : ℝ | r^2 - 9*r + 14 = 0}
  ∃ (r₁ r₂ : ℝ), r₁ ∈ roots ∧ r₂ ∈ roots ∧ |r₁ - r₂| = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_difference_of_roots_quadratic_l1438_143883


namespace NUMINAMATH_CALUDE_special_triangle_l1438_143853

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Vector in 2D space -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Theorem about a special triangle -/
theorem special_triangle (t : Triangle) 
  (hm : Vector2D) 
  (hn : Vector2D) 
  (collinear : hm.x * hn.y = hm.y * hn.x) 
  (dot_product : t.a * t.c * Real.cos t.C = -27) 
  (hm_def : hm = ⟨t.a - t.b, Real.sin t.A + Real.sin t.C⟩) 
  (hn_def : hn = ⟨t.a - t.c, Real.sin (t.A + t.C)⟩) :
  t.C = π/3 ∧ 
  (∃ (min_AB : ℝ), min_AB = 3 * Real.sqrt 6 ∧ 
    ∀ (AB : ℝ), AB ≥ min_AB) :=
by sorry

end NUMINAMATH_CALUDE_special_triangle_l1438_143853


namespace NUMINAMATH_CALUDE_grocery_shopping_theorem_l1438_143893

def initial_amount : ℝ := 100
def roast_price : ℝ := 17
def vegetables_price : ℝ := 11
def wine_price : ℝ := 12
def dessert_price : ℝ := 8
def bread_price : ℝ := 4
def milk_price : ℝ := 2
def discount_rate : ℝ := 0.15
def tax_rate : ℝ := 0.05

def total_purchase : ℝ := roast_price + vegetables_price + wine_price + dessert_price + bread_price + milk_price

def discounted_total : ℝ := total_purchase * (1 - discount_rate)

def final_amount : ℝ := discounted_total * (1 + tax_rate)

def remaining_amount : ℝ := initial_amount - final_amount

theorem grocery_shopping_theorem : 
  ∃ (ε : ℝ), abs (remaining_amount - 51.80) < ε ∧ ε > 0 :=
by sorry

end NUMINAMATH_CALUDE_grocery_shopping_theorem_l1438_143893


namespace NUMINAMATH_CALUDE_power_sum_difference_l1438_143827

theorem power_sum_difference : 2^4 + 2^4 + 2^4 - 2^2 = 44 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_difference_l1438_143827


namespace NUMINAMATH_CALUDE_correct_answer_calculation_l1438_143823

theorem correct_answer_calculation (incorrect_answer : ℝ) (h : incorrect_answer = 115.15) :
  let original_value := incorrect_answer / 7
  let correct_answer := original_value / 7
  correct_answer = 2.35 := by
sorry

end NUMINAMATH_CALUDE_correct_answer_calculation_l1438_143823


namespace NUMINAMATH_CALUDE_no_power_of_three_l1438_143815

theorem no_power_of_three (a b : ℕ+) : ¬∃ k : ℕ, (15 * a + b) * (a + 15 * b) = 3^k := by
  sorry

end NUMINAMATH_CALUDE_no_power_of_three_l1438_143815


namespace NUMINAMATH_CALUDE_temperature_decrease_l1438_143889

/-- The temperature that is 6°C lower than -3°C is -9°C. -/
theorem temperature_decrease : ((-3 : ℤ) - 6) = -9 := by
  sorry

end NUMINAMATH_CALUDE_temperature_decrease_l1438_143889


namespace NUMINAMATH_CALUDE_shiela_colors_l1438_143876

theorem shiela_colors (total_blocks : ℕ) (blocks_per_color : ℕ) (h1 : total_blocks = 49) (h2 : blocks_per_color = 7) :
  total_blocks / blocks_per_color = 7 :=
by sorry

end NUMINAMATH_CALUDE_shiela_colors_l1438_143876


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l1438_143869

-- Define conditions p and q
def condition_p (x y : ℝ) : Prop := x + y > 4 ∧ x * y > 4
def condition_q (x y : ℝ) : Prop := x > 2 ∧ y > 2

-- Theorem statement
theorem p_necessary_not_sufficient_for_q :
  (∀ x y : ℝ, condition_q x y → condition_p x y) ∧
  ¬(∀ x y : ℝ, condition_p x y → condition_q x y) := by
  sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l1438_143869


namespace NUMINAMATH_CALUDE_max_snack_bars_l1438_143875

/-- Represents the number of snack bars in a pack -/
inductive PackSize
  | single : PackSize
  | twin : PackSize
  | four : PackSize

/-- Represents the price of a pack of snack bars -/
def price (p : PackSize) : ℚ :=
  match p with
  | PackSize.single => 1
  | PackSize.twin => 5/2
  | PackSize.four => 4

/-- Represents the number of snack bars in a pack -/
def bars_in_pack (p : PackSize) : ℕ :=
  match p with
  | PackSize.single => 1
  | PackSize.twin => 2
  | PackSize.four => 4

/-- The budget available for purchasing snack bars -/
def budget : ℚ := 10

/-- A purchase combination is represented as a function from PackSize to ℕ -/
def PurchaseCombination := PackSize → ℕ

/-- The total cost of a purchase combination -/
def total_cost (c : PurchaseCombination) : ℚ :=
  (c PackSize.single) * (price PackSize.single) +
  (c PackSize.twin) * (price PackSize.twin) +
  (c PackSize.four) * (price PackSize.four)

/-- The total number of snack bars in a purchase combination -/
def total_bars (c : PurchaseCombination) : ℕ :=
  (c PackSize.single) * (bars_in_pack PackSize.single) +
  (c PackSize.twin) * (bars_in_pack PackSize.twin) +
  (c PackSize.four) * (bars_in_pack PackSize.four)

/-- A purchase combination is valid if its total cost is within the budget -/
def is_valid_combination (c : PurchaseCombination) : Prop :=
  total_cost c ≤ budget

theorem max_snack_bars :
  ∃ (max : ℕ), 
    (∃ (c : PurchaseCombination), is_valid_combination c ∧ total_bars c = max) ∧
    (∀ (c : PurchaseCombination), is_valid_combination c → total_bars c ≤ max) ∧
    max = 10 :=
  sorry

end NUMINAMATH_CALUDE_max_snack_bars_l1438_143875


namespace NUMINAMATH_CALUDE_grandpa_lou_movie_time_l1438_143896

theorem grandpa_lou_movie_time :
  ∀ (tuesday_movies : ℕ),
    (tuesday_movies + 2 * tuesday_movies ≤ 9) →
    (tuesday_movies * 90 = 270) :=
by
  sorry

end NUMINAMATH_CALUDE_grandpa_lou_movie_time_l1438_143896


namespace NUMINAMATH_CALUDE_visible_sum_range_l1438_143833

/-- Represents a die with 6 faces -/
structure Die :=
  (faces : Fin 6 → Nat)
  (opposite_sum : ∀ i : Fin 6, faces i + faces (5 - i) = 7)
  (face_range : ∀ i : Fin 6, 1 ≤ faces i ∧ faces i ≤ 6)

/-- Represents the larger 3x3x3 cube made of 27 dice -/
def LargeCube := Fin 3 → Fin 3 → Fin 3 → Die

/-- Calculates the sum of visible face values on the larger cube -/
def visible_sum (cube : LargeCube) : Nat :=
  sorry

/-- Theorem stating the range of possible sums of visible face values -/
theorem visible_sum_range (cube : LargeCube) :
  90 ≤ visible_sum cube ∧ visible_sum cube ≤ 288 :=
sorry

end NUMINAMATH_CALUDE_visible_sum_range_l1438_143833


namespace NUMINAMATH_CALUDE_yellow_peaches_count_l1438_143850

theorem yellow_peaches_count (red green total : ℕ) 
  (h_red : red = 7)
  (h_green : green = 8)
  (h_total : total = 30)
  (h_sum : red + green + yellow = total) :
  yellow = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_yellow_peaches_count_l1438_143850


namespace NUMINAMATH_CALUDE_sphere_in_cone_surface_area_ratio_l1438_143863

theorem sphere_in_cone_surface_area_ratio (r : ℝ) (h : r > 0) :
  let cone_height : ℝ := 3 * r
  let triangle_side : ℝ := 2 * Real.sqrt 3 * r
  let sphere_surface_area : ℝ := 4 * Real.pi * r^2
  let cone_base_radius : ℝ := Real.sqrt 3 * r
  let cone_lateral_area : ℝ := Real.pi * cone_base_radius * triangle_side
  let cone_base_area : ℝ := Real.pi * cone_base_radius^2
  let cone_total_surface_area : ℝ := cone_lateral_area + cone_base_area
  cone_total_surface_area / sphere_surface_area = 9 / 4 :=
by sorry

end NUMINAMATH_CALUDE_sphere_in_cone_surface_area_ratio_l1438_143863


namespace NUMINAMATH_CALUDE_data_grouping_l1438_143890

theorem data_grouping (max_value min_value class_width : ℕ) 
  (h1 : max_value = 141)
  (h2 : min_value = 40)
  (h3 : class_width = 10) :
  Int.ceil ((max_value - min_value : ℝ) / class_width) = 11 := by
  sorry

#check data_grouping

end NUMINAMATH_CALUDE_data_grouping_l1438_143890


namespace NUMINAMATH_CALUDE_g_expression_l1438_143841

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * x + 3

-- Define the relationship between f and g
def g_relation (g : ℝ → ℝ) : Prop :=
  ∀ x, g (x + 2) = f x

-- Theorem statement
theorem g_expression (g : ℝ → ℝ) (h : g_relation g) :
  ∀ x, g x = 2 * x - 1 := by
  sorry

end NUMINAMATH_CALUDE_g_expression_l1438_143841


namespace NUMINAMATH_CALUDE_cylinder_max_volume_ratio_l1438_143867

/-- The ratio of height to base radius of a cylinder with surface area 6π when its volume is maximized -/
theorem cylinder_max_volume_ratio : 
  ∃ (h r : ℝ), 
    h > 0 ∧ r > 0 ∧  -- Ensure positive height and radius
    2 * π * r^2 + 2 * π * r * h = 6 * π ∧  -- Surface area condition
    (∀ (h' r' : ℝ), 
      h' > 0 ∧ r' > 0 ∧ 
      2 * π * r'^2 + 2 * π * r' * h' = 6 * π → 
      π * r^2 * h ≥ π * r'^2 * h') →  -- Volume maximization condition
    h / r = 2 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_max_volume_ratio_l1438_143867


namespace NUMINAMATH_CALUDE_glove_sequences_l1438_143824

/-- Represents the number of hands. -/
def num_hands : ℕ := 2

/-- Represents the number of layers of gloves. -/
def num_layers : ℕ := 2

/-- Represents whether the inner gloves are identical. -/
def inner_gloves_identical : Prop := True

/-- Represents whether the outer gloves are distinct for left and right hands. -/
def outer_gloves_distinct : Prop := True

/-- The number of different sequences for wearing the gloves. -/
def num_sequences : ℕ := 6

/-- Theorem stating that the number of different sequences for wearing the gloves is 6. -/
theorem glove_sequences :
  num_hands = 2 ∧ 
  num_layers = 2 ∧ 
  inner_gloves_identical ∧ 
  outer_gloves_distinct →
  num_sequences = 6 :=
by sorry


end NUMINAMATH_CALUDE_glove_sequences_l1438_143824


namespace NUMINAMATH_CALUDE_percentage_problem_l1438_143852

theorem percentage_problem (n : ℝ) (h : 1.2 * n = 2400) : 0.2 * n = 400 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1438_143852


namespace NUMINAMATH_CALUDE_overlap_area_is_three_quarters_l1438_143810

/-- Represents a point on a 3x3 grid --/
structure GridPoint where
  x : Fin 3
  y : Fin 3

/-- Represents a triangle on the grid --/
structure GridTriangle where
  p1 : GridPoint
  p2 : GridPoint
  p3 : GridPoint

/-- The first triangle connecting top left, middle right, and bottom center --/
def triangle1 : GridTriangle :=
  { p1 := ⟨0, 0⟩, p2 := ⟨2, 1⟩, p3 := ⟨1, 2⟩ }

/-- The second triangle connecting top right, middle left, and bottom center --/
def triangle2 : GridTriangle :=
  { p1 := ⟨2, 0⟩, p2 := ⟨0, 1⟩, p3 := ⟨1, 2⟩ }

/-- Calculates the area of overlap between two triangles on the grid --/
def areaOfOverlap (t1 t2 : GridTriangle) : ℝ :=
  sorry

/-- Theorem stating that the area of overlap between the two specific triangles is 0.75 --/
theorem overlap_area_is_three_quarters :
  areaOfOverlap triangle1 triangle2 = 0.75 := by sorry

end NUMINAMATH_CALUDE_overlap_area_is_three_quarters_l1438_143810


namespace NUMINAMATH_CALUDE_symmetry_implies_a_value_l1438_143897

/-- Two points are symmetric with respect to the y-axis if their x-coordinates are negatives of each other and their y-coordinates are the same. -/
def symmetric_y_axis (A B : ℝ × ℝ) : Prop :=
  A.1 = -B.1 ∧ A.2 = B.2

theorem symmetry_implies_a_value :
  ∀ a : ℝ, symmetric_y_axis (a, 1) (-3, 1) → a = 3 :=
by sorry

end NUMINAMATH_CALUDE_symmetry_implies_a_value_l1438_143897


namespace NUMINAMATH_CALUDE_number_minus_division_equals_l1438_143806

theorem number_minus_division_equals (x : ℝ) : x - (502 / 100.4) = 5015 → x = 5020 := by
  sorry

end NUMINAMATH_CALUDE_number_minus_division_equals_l1438_143806


namespace NUMINAMATH_CALUDE_lawsuit_probability_comparison_l1438_143807

def probability_lawsuit1_win : ℝ := 0.3
def probability_lawsuit2_win : ℝ := 0.5
def probability_lawsuit3_win : ℝ := 0.4

def probability_lawsuit1_lose : ℝ := 1 - probability_lawsuit1_win
def probability_lawsuit2_lose : ℝ := 1 - probability_lawsuit2_win
def probability_lawsuit3_lose : ℝ := 1 - probability_lawsuit3_win

def probability_win_all : ℝ := probability_lawsuit1_win * probability_lawsuit2_win * probability_lawsuit3_win
def probability_lose_all : ℝ := probability_lawsuit1_lose * probability_lawsuit2_lose * probability_lawsuit3_lose

theorem lawsuit_probability_comparison :
  (probability_lose_all - probability_win_all) / probability_win_all * 100 = 250 := by
sorry

end NUMINAMATH_CALUDE_lawsuit_probability_comparison_l1438_143807


namespace NUMINAMATH_CALUDE_diamond_negative_two_three_l1438_143868

-- Define the diamond operation
def diamond (a b : ℝ) : ℝ := a + a * b^2 - b + 1

-- Theorem statement
theorem diamond_negative_two_three : diamond (-2) 3 = -22 := by sorry

end NUMINAMATH_CALUDE_diamond_negative_two_three_l1438_143868


namespace NUMINAMATH_CALUDE_total_oranges_bought_l1438_143855

/-- The number of times Stephanie went to the store last month -/
def store_visits : ℕ := 8

/-- The number of oranges Stephanie buys each time she goes to the store -/
def oranges_per_visit : ℕ := 2

/-- Theorem: The total number of oranges Stephanie bought last month is 16 -/
theorem total_oranges_bought : store_visits * oranges_per_visit = 16 := by
  sorry

end NUMINAMATH_CALUDE_total_oranges_bought_l1438_143855


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1438_143898

-- Define the quadratic function
def f (x : ℝ) := x^2 + 3*x - 4

-- Define the solution set
def solution_set : Set ℝ := {x | -4 < x ∧ x < 1}

-- Theorem statement
theorem quadratic_inequality_solution :
  {x : ℝ | f x < 0} = solution_set := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1438_143898


namespace NUMINAMATH_CALUDE_jackson_inbox_problem_l1438_143846

theorem jackson_inbox_problem (initial_deleted : ℕ) (initial_received : ℕ)
  (subsequent_deleted : ℕ) (subsequent_received : ℕ) (final_count : ℕ)
  (h1 : initial_deleted = 50)
  (h2 : initial_received = 15)
  (h3 : subsequent_deleted = 20)
  (h4 : subsequent_received = 5)
  (h5 : final_count = 30) :
  final_count - (initial_received + subsequent_received) = 10 := by
  sorry

end NUMINAMATH_CALUDE_jackson_inbox_problem_l1438_143846


namespace NUMINAMATH_CALUDE_badminton_players_l1438_143809

theorem badminton_players (total : ℕ) (tennis : ℕ) (neither : ℕ) (both : ℕ) 
  (h1 : total = 28)
  (h2 : tennis = 19)
  (h3 : neither = 2)
  (h4 : both = 10) :
  ∃ badminton : ℕ, badminton = 17 ∧ 
    total = tennis + badminton - both + neither :=
by sorry

end NUMINAMATH_CALUDE_badminton_players_l1438_143809


namespace NUMINAMATH_CALUDE_conference_handshakes_l1438_143882

theorem conference_handshakes (total_attendees : ℕ) (leaders : ℕ) (participants : ℕ)
  (h1 : total_attendees = 30)
  (h2 : leaders = 5)
  (h3 : participants = total_attendees - leaders)
  (h4 : leaders + participants = total_attendees) :
  (leaders * (total_attendees - 1) - leaders * (leaders - 1) / 2) = 135 := by
  sorry

end NUMINAMATH_CALUDE_conference_handshakes_l1438_143882


namespace NUMINAMATH_CALUDE_complement_union_theorem_l1438_143818

def I : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 1, 2, 3}
def B : Set ℕ := {2, 3, 4}

theorem complement_union_theorem :
  (I \ A) ∪ (I \ B) = {0, 1, 4} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l1438_143818


namespace NUMINAMATH_CALUDE_wheel_revolutions_for_one_mile_l1438_143871

-- Define the wheel's diameter
def wheel_diameter : ℝ := 8

-- Define the length of a mile in feet
def mile_in_feet : ℝ := 5280

-- Theorem statement
theorem wheel_revolutions_for_one_mile :
  let wheel_circumference := π * wheel_diameter
  let revolutions := mile_in_feet / wheel_circumference
  revolutions = 660 / π :=
by
  sorry

end NUMINAMATH_CALUDE_wheel_revolutions_for_one_mile_l1438_143871


namespace NUMINAMATH_CALUDE_ivory_josh_riddle_difference_l1438_143877

theorem ivory_josh_riddle_difference :
  ∀ (ivory_riddles josh_riddles taso_riddles : ℕ),
    josh_riddles = 8 →
    taso_riddles = 24 →
    taso_riddles = 2 * ivory_riddles →
    ivory_riddles - josh_riddles = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_ivory_josh_riddle_difference_l1438_143877


namespace NUMINAMATH_CALUDE_quadratic_real_solutions_m_range_l1438_143801

theorem quadratic_real_solutions_m_range (m : ℝ) : 
  (∃ x : ℝ, m * x^2 + 2 * x + 1 = 0) → m ≤ 1 ∧ m ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_solutions_m_range_l1438_143801


namespace NUMINAMATH_CALUDE_units_digit_100_factorial_l1438_143837

theorem units_digit_100_factorial (n : ℕ) : n = 100 → n.factorial % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_100_factorial_l1438_143837


namespace NUMINAMATH_CALUDE_solution_set_part_I_range_of_a_part_II_l1438_143839

-- Define the function f
def f (a x : ℝ) : ℝ := |x - a| + 5*x

-- Theorem for part I
theorem solution_set_part_I :
  {x : ℝ | |x + 1| + 5*x ≤ 5*x + 3} = Set.Icc (-4) 2 := by sorry

-- Theorem for part II
theorem range_of_a_part_II :
  ∀ a : ℝ, (∀ x ≥ -1, f a x ≥ 0) ↔ (a ≥ 4 ∨ a ≤ -6) := by sorry

end NUMINAMATH_CALUDE_solution_set_part_I_range_of_a_part_II_l1438_143839


namespace NUMINAMATH_CALUDE_q_value_for_p_seven_l1438_143887

/-- Given the equation Q = 3rP - 6, where r is a constant, prove that if Q = 27 when P = 5, then Q = 40 when P = 7 -/
theorem q_value_for_p_seven (r : ℝ) : 
  (∃ Q : ℝ, Q = 3 * r * 5 - 6 ∧ Q = 27) →
  (∃ Q : ℝ, Q = 3 * r * 7 - 6 ∧ Q = 40) :=
by sorry

end NUMINAMATH_CALUDE_q_value_for_p_seven_l1438_143887


namespace NUMINAMATH_CALUDE_final_value_of_A_l1438_143838

theorem final_value_of_A : ∀ A : ℤ, A = 15 → (A = -15 + 5) → A = -10 := by
  sorry

end NUMINAMATH_CALUDE_final_value_of_A_l1438_143838


namespace NUMINAMATH_CALUDE_pet_store_combinations_l1438_143878

def num_puppies : ℕ := 15
def num_kittens : ℕ := 6
def num_hamsters : ℕ := 8
def num_friends : ℕ := 3

theorem pet_store_combinations : 
  num_puppies * num_kittens * num_hamsters * Nat.factorial num_friends = 4320 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_combinations_l1438_143878
