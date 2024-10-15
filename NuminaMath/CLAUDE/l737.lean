import Mathlib

namespace NUMINAMATH_CALUDE_average_median_relation_l737_73795

theorem average_median_relation (a b c : ℤ) : 
  (a + b + c) / 3 = 4 * b →
  a < b →
  b < c →
  a = 0 →
  c / b = 11 := by
  sorry

end NUMINAMATH_CALUDE_average_median_relation_l737_73795


namespace NUMINAMATH_CALUDE_twirly_tea_cups_l737_73733

theorem twirly_tea_cups (people_per_cup : ℕ) (total_people : ℕ) (num_cups : ℕ) :
  people_per_cup = 9 →
  total_people = 63 →
  num_cups * people_per_cup = total_people →
  num_cups = 7 := by
  sorry

end NUMINAMATH_CALUDE_twirly_tea_cups_l737_73733


namespace NUMINAMATH_CALUDE_paving_cost_calculation_l737_73738

/-- The cost of paving a rectangular floor -/
def paving_cost (length width rate : ℝ) : ℝ :=
  length * width * rate

/-- Theorem: The cost of paving a rectangular floor with given dimensions and rate -/
theorem paving_cost_calculation (length width rate : ℝ) 
  (h1 : length = 5.5)
  (h2 : width = 3.75)
  (h3 : rate = 1000) :
  paving_cost length width rate = 20625 := by
  sorry

end NUMINAMATH_CALUDE_paving_cost_calculation_l737_73738


namespace NUMINAMATH_CALUDE_sum_of_solutions_l737_73754

theorem sum_of_solutions (x y : ℝ) : 
  x * y = 1 ∧ x + y = 3 → ∃ (x₁ x₂ : ℝ), x₁ + x₂ = 3 ∧ 
  (x₁ * (3 - x₁) = 1 ∧ x₁ + (3 - x₁) = 3) ∧
  (x₂ * (3 - x₂) = 1 ∧ x₂ + (3 - x₂) = 3) :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_l737_73754


namespace NUMINAMATH_CALUDE_no_integer_solutions_l737_73767

theorem no_integer_solutions : ¬ ∃ (a b : ℤ), 3 * a^2 = b^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l737_73767


namespace NUMINAMATH_CALUDE_profit_sharing_problem_l737_73701

/-- Given a profit shared between two partners in the ratio 2:5, where the second partner
    receives $2500, prove that the first partner will have $800 left after spending $200. -/
theorem profit_sharing_problem (total_parts : ℕ) (first_partner_parts second_partner_parts : ℕ) 
    (second_partner_share : ℕ) (shirt_cost : ℕ) :
  total_parts = first_partner_parts + second_partner_parts →
  first_partner_parts = 2 →
  second_partner_parts = 5 →
  second_partner_share = 2500 →
  shirt_cost = 200 →
  (first_partner_parts * second_partner_share / second_partner_parts) - shirt_cost = 800 :=
by sorry


end NUMINAMATH_CALUDE_profit_sharing_problem_l737_73701


namespace NUMINAMATH_CALUDE_distance_covered_l737_73770

theorem distance_covered (time_minutes : ℝ) (speed_km_per_hour : ℝ) : 
  time_minutes = 42 → speed_km_per_hour = 10 → 
  (time_minutes / 60) * speed_km_per_hour = 7 := by
  sorry

end NUMINAMATH_CALUDE_distance_covered_l737_73770


namespace NUMINAMATH_CALUDE_fraction_denominator_l737_73784

theorem fraction_denominator (y a : ℝ) (h1 : y > 0) (h2 : (2 * y) / a + (3 * y) / a = 0.5 * y) : a = 10 := by
  sorry

end NUMINAMATH_CALUDE_fraction_denominator_l737_73784


namespace NUMINAMATH_CALUDE_second_year_interest_rate_l737_73764

/-- Calculates the interest rate for the second year given the initial principal,
    first year interest rate, and final amount after two years. -/
theorem second_year_interest_rate
  (initial_principal : ℝ)
  (first_year_rate : ℝ)
  (final_amount : ℝ)
  (h1 : initial_principal = 8000)
  (h2 : first_year_rate = 0.04)
  (h3 : final_amount = 8736) :
  let first_year_amount := initial_principal * (1 + first_year_rate)
  let second_year_rate := (final_amount / first_year_amount) - 1
  second_year_rate = 0.05 := by
sorry

end NUMINAMATH_CALUDE_second_year_interest_rate_l737_73764


namespace NUMINAMATH_CALUDE_solve_for_x_l737_73705

-- Define the € operation
def euro (x y : ℝ) : ℝ := 3 * x * y

-- State the theorem
theorem solve_for_x (y : ℝ) (h1 : y = 3) (h2 : euro y (euro 4 x) = 540) : x = 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_x_l737_73705


namespace NUMINAMATH_CALUDE_translation_lori_to_alex_l737_73796

/-- Represents a 2D point --/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a 2D translation --/
structure Translation where
  dx : ℝ
  dy : ℝ

/-- Lori's house location --/
def lori_house : Point := ⟨6, 3⟩

/-- Alex's house location --/
def alex_house : Point := ⟨-2, -4⟩

/-- Calculates the translation between two points --/
def calculate_translation (p1 p2 : Point) : Translation :=
  ⟨p2.x - p1.x, p2.y - p1.y⟩

/-- Theorem: The translation from Lori's house to Alex's house is 8 units left and 7 units down --/
theorem translation_lori_to_alex :
  let t := calculate_translation lori_house alex_house
  t.dx = -8 ∧ t.dy = -7 := by sorry

end NUMINAMATH_CALUDE_translation_lori_to_alex_l737_73796


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l737_73727

theorem complex_fraction_equality (a b : ℝ) :
  (1 + Complex.I) / (1 - Complex.I) = Complex.mk a b → b = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l737_73727


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l737_73713

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 52) 
  (h2 : a*b + b*c + c*a = 72) : 
  a + b + c = 14 := by sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l737_73713


namespace NUMINAMATH_CALUDE_hillary_stops_short_of_summit_l737_73766

/-- Represents the climbing scenario of Hillary and Eddy on Mt. Everest -/
structure ClimbingScenario where
  summit_distance : ℝ  -- Distance from base camp to summit in feet
  hillary_ascent_rate : ℝ  -- Hillary's ascent rate in ft/hr
  eddy_ascent_rate : ℝ  -- Eddy's ascent rate in ft/hr
  hillary_descent_rate : ℝ  -- Hillary's descent rate in ft/hr
  trip_duration : ℝ  -- Duration of the trip in hours

/-- Calculates the distance Hillary stops short of the summit -/
def distance_short_of_summit (scenario : ClimbingScenario) : ℝ :=
  scenario.hillary_ascent_rate * scenario.trip_duration - 
  (scenario.hillary_ascent_rate * scenario.trip_duration + 
   scenario.eddy_ascent_rate * scenario.trip_duration - scenario.summit_distance)

/-- Theorem stating that Hillary stops 3000 ft short of the summit -/
theorem hillary_stops_short_of_summit (scenario : ClimbingScenario) 
  (h1 : scenario.summit_distance = 5000)
  (h2 : scenario.hillary_ascent_rate = 800)
  (h3 : scenario.eddy_ascent_rate = 500)
  (h4 : scenario.hillary_descent_rate = 1000)
  (h5 : scenario.trip_duration = 6) :
  distance_short_of_summit scenario = 3000 := by
  sorry

#eval distance_short_of_summit {
  summit_distance := 5000,
  hillary_ascent_rate := 800,
  eddy_ascent_rate := 500,
  hillary_descent_rate := 1000,
  trip_duration := 6
}

end NUMINAMATH_CALUDE_hillary_stops_short_of_summit_l737_73766


namespace NUMINAMATH_CALUDE_no_zero_root_for_equations_l737_73781

theorem no_zero_root_for_equations :
  (∀ x : ℝ, 3 * x^2 - 5 = 50 → x ≠ 0) ∧
  (∀ x : ℝ, (3*x - 2)^2 = (x - 2)^2 → x ≠ 0) ∧
  (∀ x : ℝ, x^2 - 15 = x + 2 → x ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_no_zero_root_for_equations_l737_73781


namespace NUMINAMATH_CALUDE_extreme_values_when_a_neg_one_max_value_when_a_positive_l737_73761

noncomputable section

-- Define the function f(x) = (ax^2 + x + a)e^x
def f (a : ℝ) (x : ℝ) : ℝ := (a * x^2 + x + a) * Real.exp x

-- Theorem for part (1)
theorem extreme_values_when_a_neg_one :
  let f := f (-1)
  (∃ x, ∀ y, f y ≥ f x) ∧ (f 0 = -1) ∧
  (∃ x, ∀ y, f y ≤ f x) ∧ (f (-1) = -3 * Real.exp (-1)) := by sorry

-- Theorem for part (2)
theorem max_value_when_a_positive (a : ℝ) (h : a > 0) :
  let f := f a
  let max_value := if a > 1 then (2*a + 1) * Real.exp (-1 - 1/a)
                   else (5*a - 2) * Real.exp (-2)
  ∀ x ∈ Set.Icc (-2) (-1), f x ≤ max_value := by sorry

end

end NUMINAMATH_CALUDE_extreme_values_when_a_neg_one_max_value_when_a_positive_l737_73761


namespace NUMINAMATH_CALUDE_courier_packages_tomorrow_l737_73765

/-- The number of packages to be delivered tomorrow -/
def packages_to_deliver (yesterday : ℕ) (today : ℕ) : ℕ :=
  yesterday + today

/-- Theorem: The courier should deliver 240 packages tomorrow -/
theorem courier_packages_tomorrow :
  let yesterday := 80
  let today := 2 * yesterday
  packages_to_deliver yesterday today = 240 := by
  sorry

end NUMINAMATH_CALUDE_courier_packages_tomorrow_l737_73765


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l737_73778

theorem quadratic_roots_relation (m n p : ℝ) : 
  (∀ r : ℝ, (r^2 + p*r + m = 0) → ((3*r)^2 + m*(3*r) + n = 0)) →
  n / p = 27 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l737_73778


namespace NUMINAMATH_CALUDE_car_distance_theorem_l737_73769

/-- Calculates the total distance traveled by a car given its initial speed, acceleration, 
    acceleration time, constant speed, and constant speed time. -/
def total_distance (initial_speed : ℝ) (acceleration : ℝ) (accel_time : ℝ) 
                   (constant_speed : ℝ) (const_time : ℝ) : ℝ :=
  -- Distance covered during acceleration
  (initial_speed * accel_time + 0.5 * acceleration * accel_time^2) +
  -- Distance covered at constant speed
  (constant_speed * const_time)

/-- Theorem stating that a car with given parameters travels 250 miles in total -/
theorem car_distance_theorem : 
  total_distance 30 5 2 60 3 = 250 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_theorem_l737_73769


namespace NUMINAMATH_CALUDE_f_value_at_2_l737_73700

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x + 8

-- State the theorem
theorem f_value_at_2 (a b : ℝ) : f a b (-2) = 10 → f a b 2 = 6 := by sorry

end NUMINAMATH_CALUDE_f_value_at_2_l737_73700


namespace NUMINAMATH_CALUDE_problem_statement_l737_73792

theorem problem_statement (a b c : ℝ) : 
  (¬(∀ x y : ℝ, x > y → x^2 > y^2) ∧ ¬(∀ x y : ℝ, x^2 > y^2 → x > y)) ∧
  ((∀ x y z : ℝ, x*z^2 > y*z^2 → x > y) ∧ ¬(∀ x y z : ℝ, x > y → x*z^2 > y*z^2)) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l737_73792


namespace NUMINAMATH_CALUDE_optimal_plan_l737_73711

/-- Represents a sewage treatment equipment purchasing plan -/
structure Plan where
  typeA : ℕ
  typeB : ℕ

/-- Checks if a plan is valid according to the given constraints -/
def isValidPlan (p : Plan) : Prop :=
  p.typeA + p.typeB = 20 ∧
  120000 * p.typeA + 100000 * p.typeB ≤ 2300000 ∧
  240 * p.typeA + 200 * p.typeB ≥ 4500

/-- Calculates the total cost of a plan -/
def planCost (p : Plan) : ℕ :=
  120000 * p.typeA + 100000 * p.typeB

/-- Theorem stating that the optimal plan is 13 units of A and 7 units of B -/
theorem optimal_plan :
  ∃ (optimalPlan : Plan),
    isValidPlan optimalPlan ∧
    optimalPlan.typeA = 13 ∧
    optimalPlan.typeB = 7 ∧
    planCost optimalPlan = 2260000 ∧
    ∀ (p : Plan), isValidPlan p → planCost p ≥ planCost optimalPlan :=
  sorry


end NUMINAMATH_CALUDE_optimal_plan_l737_73711


namespace NUMINAMATH_CALUDE_trains_clearing_time_l737_73788

/-- Calculates the time for two trains to clear each other -/
theorem trains_clearing_time (length1 length2 speed1 speed2 : ℝ) : 
  length1 = 160 ∧ 
  length2 = 320 ∧ 
  speed1 = 42 ∧ 
  speed2 = 30 → 
  (length1 + length2) / ((speed1 + speed2) * (1000 / 3600)) = 24 := by
  sorry

#check trains_clearing_time

end NUMINAMATH_CALUDE_trains_clearing_time_l737_73788


namespace NUMINAMATH_CALUDE_min_value_product_min_value_achieved_l737_73750

theorem min_value_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x/y + y/z + z/x + y/x + z/y + x/z = 10) :
  (x/y + y/z + z/x) * (y/x + z/y + x/z) ≥ 25 := by
  sorry

theorem min_value_achieved (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x/y + y/z + z/x + y/x + z/y + x/z = 10) :
  ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧
    a/b + b/c + c/a + b/a + c/b + a/c = 10 ∧
    (a/b + b/c + c/a) * (b/a + c/b + a/c) = 25 := by
  sorry

end NUMINAMATH_CALUDE_min_value_product_min_value_achieved_l737_73750


namespace NUMINAMATH_CALUDE_perimeter_of_circular_sector_problem_perimeter_l737_73799

/-- The perimeter of a region formed by two radii and an arc of a circle -/
theorem perimeter_of_circular_sector (r : ℝ) (arc_fraction : ℝ) : 
  r > 0 → 
  0 < arc_fraction → 
  arc_fraction ≤ 1 → 
  2 * r + arc_fraction * (2 * π * r) = 2 * r + 2 * arc_fraction * π * r :=
by sorry

/-- The perimeter of the specific region in the problem -/
theorem problem_perimeter : 
  let r : ℝ := 8
  let arc_fraction : ℝ := 5/6
  2 * r + arc_fraction * (2 * π * r) = 16 + (40/3) * π :=
by sorry

end NUMINAMATH_CALUDE_perimeter_of_circular_sector_problem_perimeter_l737_73799


namespace NUMINAMATH_CALUDE_boys_insects_count_l737_73785

/-- The number of groups in the class -/
def num_groups : ℕ := 4

/-- The number of insects each group receives -/
def insects_per_group : ℕ := 125

/-- The number of insects collected by the girls -/
def girls_insects : ℕ := 300

/-- The number of insects collected by the boys -/
def boys_insects : ℕ := num_groups * insects_per_group - girls_insects

theorem boys_insects_count :
  boys_insects = 200 := by sorry

end NUMINAMATH_CALUDE_boys_insects_count_l737_73785


namespace NUMINAMATH_CALUDE_matrix_multiplication_result_l737_73739

def A : Matrix (Fin 2) (Fin 2) ℤ := !![1, 2; 2, 1]
def B : Matrix (Fin 2) (Fin 2) ℤ := !![-1, 2; 3, -4]
def C : Matrix (Fin 2) (Fin 2) ℤ := !![5, -6; 1, 0]

theorem matrix_multiplication_result : A * B = C := by sorry

end NUMINAMATH_CALUDE_matrix_multiplication_result_l737_73739


namespace NUMINAMATH_CALUDE_range_of_m_l737_73715

theorem range_of_m (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 4) 
  (h : ∀ m : ℝ, (4/x) + (16/y) > m^2 - 3*m + 5) :
  ∀ m : ℝ, -1 < m ∧ m < 4 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l737_73715


namespace NUMINAMATH_CALUDE_cross_number_puzzle_solution_l737_73720

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def is_power_of (base : ℕ) (n : ℕ) : Prop := ∃ m : ℕ, n = base ^ m

theorem cross_number_puzzle_solution :
  ∃! d : ℕ, d < 10 ∧
    (∃ n₃ n₇ : ℕ,
      is_three_digit n₃ ∧
      is_three_digit n₇ ∧
      is_power_of 3 n₃ ∧
      is_power_of 7 n₇ ∧
      (∃ k₃ k₇ : ℕ, n₃ % 10^k₃ / 10^(k₃-1) = d ∧ n₇ % 10^k₇ / 10^(k₇-1) = d)) :=
sorry

end NUMINAMATH_CALUDE_cross_number_puzzle_solution_l737_73720


namespace NUMINAMATH_CALUDE_negative_three_a_cubed_div_a_fourth_l737_73706

theorem negative_three_a_cubed_div_a_fourth (a : ℝ) (h : a ≠ 0) :
  -3 * a^3 / a^4 = -3 / a := by sorry

end NUMINAMATH_CALUDE_negative_three_a_cubed_div_a_fourth_l737_73706


namespace NUMINAMATH_CALUDE_smallest_y_in_arithmetic_series_l737_73783

theorem smallest_y_in_arithmetic_series (x y z : ℝ) : 
  x > 0 → y > 0 → z > 0 →  -- all terms are positive
  ∃ d : ℝ, x = y - d ∧ z = y + d →  -- arithmetic series
  x * y * z = 216 →  -- product is 216
  y ≥ 6 ∧ ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ 
    (∃ d₀ : ℝ, x₀ = y₀ - d₀ ∧ z₀ = y₀ + d₀) ∧ 
    x₀ * y₀ * z₀ = 216 ∧ y₀ = 6 :=
by sorry

end NUMINAMATH_CALUDE_smallest_y_in_arithmetic_series_l737_73783


namespace NUMINAMATH_CALUDE_petya_catches_up_l737_73749

/-- Represents the race scenario between Petya and Vasya -/
structure RaceScenario where
  total_distance : ℝ
  vasya_speed : ℝ
  petya_first_half_speed : ℝ

/-- Calculates Petya's required speed for the second half of the race -/
def petya_second_half_speed (race : RaceScenario) : ℝ :=
  2 * race.vasya_speed - race.petya_first_half_speed

/-- Theorem stating that Petya's speed for the second half must be 18 km/h -/
theorem petya_catches_up (race : RaceScenario) 
  (h1 : race.total_distance > 0)
  (h2 : race.vasya_speed = 12)
  (h3 : race.petya_first_half_speed = 9) :
  petya_second_half_speed race = 18 := by
  sorry

#eval petya_second_half_speed { total_distance := 100, vasya_speed := 12, petya_first_half_speed := 9 }

end NUMINAMATH_CALUDE_petya_catches_up_l737_73749


namespace NUMINAMATH_CALUDE_f_at_5_l737_73719

def f (x : ℝ) : ℝ := 2*x^5 - 5*x^4 - 4*x^3 + 3*x^2 - 524

theorem f_at_5 : f 5 = 2176 := by
  sorry

end NUMINAMATH_CALUDE_f_at_5_l737_73719


namespace NUMINAMATH_CALUDE_last_digit_of_2_to_2024_l737_73793

def last_digit (n : ℕ) : ℕ := n % 10

def power_of_two_last_digit (n : ℕ) : ℕ :=
  match n % 4 with
  | 1 => 2
  | 2 => 4
  | 3 => 8
  | 0 => 6
  | _ => 0  -- This case should never occur, but Lean requires it for completeness

theorem last_digit_of_2_to_2024 :
  last_digit (2^2024) = power_of_two_last_digit 2024 :=
by sorry

end NUMINAMATH_CALUDE_last_digit_of_2_to_2024_l737_73793


namespace NUMINAMATH_CALUDE_x_plus_y_value_l737_73748

theorem x_plus_y_value (x y : ℝ) 
  (eq1 : x + Real.cos y = 2023)
  (eq2 : x + 2023 * Real.sin y = 2022)
  (y_range : π / 2 ≤ y ∧ y ≤ π) :
  x + y = 2023 + π / 2 := by
sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l737_73748


namespace NUMINAMATH_CALUDE_park_planting_problem_l737_73743

/-- The number of short bushes to be planted in a park -/
def short_bushes_to_plant (current_short_bushes total_short_bushes_after : ℕ) : ℕ :=
  total_short_bushes_after - current_short_bushes

/-- Theorem stating that 20 short bushes will be planted -/
theorem park_planting_problem :
  short_bushes_to_plant 37 57 = 20 := by
  sorry

end NUMINAMATH_CALUDE_park_planting_problem_l737_73743


namespace NUMINAMATH_CALUDE_pencil_eraser_cost_theorem_l737_73729

theorem pencil_eraser_cost_theorem :
  ∃ (p e : ℕ), p > e ∧ p > 0 ∧ e > 0 ∧ 15 * p + 5 * e = 200 ∧ p + e = 18 := by
  sorry

end NUMINAMATH_CALUDE_pencil_eraser_cost_theorem_l737_73729


namespace NUMINAMATH_CALUDE_jasons_red_marbles_indeterminate_l737_73716

theorem jasons_red_marbles_indeterminate (jason_blue : ℕ) (tom_blue : ℕ) (total_blue : ℕ) 
  (h1 : jason_blue = 44)
  (h2 : tom_blue = 24)
  (h3 : total_blue = jason_blue + tom_blue)
  (h4 : total_blue = 68) :
  ∃ (x y : ℕ), x ≠ y ∧ (jason_blue + x = jason_blue + y) :=
sorry

end NUMINAMATH_CALUDE_jasons_red_marbles_indeterminate_l737_73716


namespace NUMINAMATH_CALUDE_quadratic_minimum_l737_73787

/-- The function f(x) = x^2 + 6x + 13 has a minimum value of 4 -/
theorem quadratic_minimum (x : ℝ) : ∀ y : ℝ, x^2 + 6*x + 13 ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l737_73787


namespace NUMINAMATH_CALUDE_simplify_complex_root_expression_l737_73772

theorem simplify_complex_root_expression (x : ℝ) (h : x ≥ 0) :
  (6 * x * (5 + 2 * Real.sqrt 6)) ^ (1/4) * Real.sqrt (3 * Real.sqrt (2 * x) - 2 * Real.sqrt (3 * x)) = Real.sqrt (6 * x) := by
  sorry

end NUMINAMATH_CALUDE_simplify_complex_root_expression_l737_73772


namespace NUMINAMATH_CALUDE_initial_time_is_six_hours_l737_73712

/-- Proves that the initial time to cover 288 km is 6 hours -/
theorem initial_time_is_six_hours 
  (distance : ℝ) 
  (new_speed : ℝ) 
  (time_ratio : ℝ) :
  distance = 288 →
  new_speed = 32 →
  time_ratio = 3 / 2 →
  ∃ (initial_time : ℝ), 
    initial_time = 6 ∧ 
    distance = new_speed * (time_ratio * initial_time) :=
by sorry

end NUMINAMATH_CALUDE_initial_time_is_six_hours_l737_73712


namespace NUMINAMATH_CALUDE_slope_of_line_l737_73740

theorem slope_of_line (x y : ℝ) :
  4 * x + 6 * y = 24 → (y - 4) / x = -2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_slope_of_line_l737_73740


namespace NUMINAMATH_CALUDE_row_swap_matrix_l737_73752

theorem row_swap_matrix : ∃ (N : Matrix (Fin 2) (Fin 2) ℝ),
  ∀ (A : Matrix (Fin 2) (Fin 2) ℝ), N * A = ![![A 1 0, A 1 1], ![A 0 0, A 0 1]] :=
by
  sorry

end NUMINAMATH_CALUDE_row_swap_matrix_l737_73752


namespace NUMINAMATH_CALUDE_downstream_speed_calculation_l737_73786

/-- The speed downstream of a boat, given its speed in still water and the speed of the current. -/
def speed_downstream (speed_still_water speed_current : ℝ) : ℝ :=
  speed_still_water + speed_current

/-- Theorem stating that the speed downstream is 77 kmph when the boat's speed in still water is 60 kmph and the current speed is 17 kmph. -/
theorem downstream_speed_calculation :
  speed_downstream 60 17 = 77 := by
  sorry

end NUMINAMATH_CALUDE_downstream_speed_calculation_l737_73786


namespace NUMINAMATH_CALUDE_triangle_segment_length_l737_73707

/-- Given a triangle ADE with point C on AD and point B on AC, prove that FC = 14.6 -/
theorem triangle_segment_length 
  (DC CB : ℝ) 
  (h1 : DC = 9) 
  (h2 : CB = 10) 
  (AD : ℝ) 
  (h3 : (1 : ℝ) / 3 * AD = AD - DC - CB) 
  (ED : ℝ) 
  (h4 : ED = 3 / 4 * AD) 
  (FC : ℝ) 
  (h5 : FC * AD = ED * (DC + CB + (1 / 3 * AD))) : 
  FC = 14.6 := by
sorry

end NUMINAMATH_CALUDE_triangle_segment_length_l737_73707


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_product_l737_73776

theorem partial_fraction_decomposition_product (M₁ M₂ : ℝ) :
  (∀ x : ℝ, x ≠ 1 ∧ x ≠ 3 → 
    (42 * x - 53) / (x^2 - 4*x + 3) = M₁ / (x - 1) + M₂ / (x - 3)) →
  M₁ * M₂ = 200.75 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_product_l737_73776


namespace NUMINAMATH_CALUDE_lollipops_for_class_l737_73745

/-- Calculates the number of lollipops given away based on the number of people and the lollipop distribution rate. -/
def lollipops_given (total_people : ℕ) (people_per_lollipop : ℕ) : ℕ :=
  total_people / people_per_lollipop

/-- Proves that given 60 people and 1 lollipop per 5 people, the teacher gives away 12 lollipops. -/
theorem lollipops_for_class (total_people : ℕ) (people_per_lollipop : ℕ) 
    (h1 : total_people = 60) 
    (h2 : people_per_lollipop = 5) : 
  lollipops_given total_people people_per_lollipop = 12 := by
  sorry

#eval lollipops_given 60 5  -- Expected output: 12

end NUMINAMATH_CALUDE_lollipops_for_class_l737_73745


namespace NUMINAMATH_CALUDE_original_houses_l737_73791

/-- The number of houses built during the housing boom in Lincoln County. -/
def houses_built : ℕ := 97741

/-- The current total number of houses in Lincoln County. -/
def current_total : ℕ := 118558

/-- Theorem stating that the original number of houses in Lincoln County is 20817. -/
theorem original_houses : current_total - houses_built = 20817 := by
  sorry

end NUMINAMATH_CALUDE_original_houses_l737_73791


namespace NUMINAMATH_CALUDE_max_value_fraction_l737_73774

theorem max_value_fraction (a b : ℝ) (h : a^2 + 2*b^2 = 6) :
  ∃ (M : ℝ), M = 1 ∧ ∀ x, x = b/(a-3) → x ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_fraction_l737_73774


namespace NUMINAMATH_CALUDE_min_value_of_fraction_l737_73782

theorem min_value_of_fraction (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + b = 1) :
  (2 / a + 1 / b) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_fraction_l737_73782


namespace NUMINAMATH_CALUDE_highest_x_value_l737_73763

theorem highest_x_value (x : ℝ) :
  (((15 * x^2 - 40 * x + 18) / (4 * x - 3) + 7 * x = 9 * x - 2) →
   x ≤ 4) ∧
  (∃ y : ℝ, ((15 * y^2 - 40 * y + 18) / (4 * y - 3) + 7 * y = 9 * y - 2) ∧ y = 4) :=
by sorry

end NUMINAMATH_CALUDE_highest_x_value_l737_73763


namespace NUMINAMATH_CALUDE_workshop_workers_l737_73760

theorem workshop_workers (total_average : ℝ) (tech_count : ℕ) (tech_average : ℝ) (nontech_average : ℝ) :
  total_average = 6750 →
  tech_count = 7 →
  tech_average = 12000 →
  nontech_average = 6000 →
  ∃ (total_workers : ℕ), total_workers = 56 ∧ 
    total_average * total_workers = tech_average * tech_count + nontech_average * (total_workers - tech_count) :=
by
  sorry

end NUMINAMATH_CALUDE_workshop_workers_l737_73760


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l737_73744

/-- The line equation is of the form (a-1)x - y + 2a + 1 = 0 where a is a real number -/
def line_equation (a x y : ℝ) : Prop :=
  (a - 1) * x - y + 2 * a + 1 = 0

/-- Theorem: The line always passes through the point (-2, 3) for all real values of a -/
theorem line_passes_through_fixed_point :
  ∀ a : ℝ, line_equation a (-2) 3 :=
by sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l737_73744


namespace NUMINAMATH_CALUDE_game_end_not_one_l737_73780

/-- Represents the state of the board with the number of ones and twos -/
structure BoardState where
  ones : Nat
  twos : Nat

/-- Represents a move in the game -/
inductive Move
  | SameDigits : Move
  | DifferentDigits : Move

/-- Applies a move to the board state -/
def applyMove (state : BoardState) (move : Move) : BoardState :=
  match move with
  | Move.SameDigits => 
    if state.ones ≥ 2 then BoardState.mk (state.ones - 2) (state.twos + 1)
    else BoardState.mk state.ones (state.twos - 1)
  | Move.DifferentDigits => 
    if state.ones > 0 && state.twos > 0 
    then BoardState.mk state.ones state.twos
    else state -- This case should not occur in a valid game

/-- The theorem stating that if we start with an even number of ones, 
    the game cannot end with a single one -/
theorem game_end_not_one (initialOnes : Nat) (initialTwos : Nat) :
  initialOnes % 2 = 0 → 
  ∀ (moves : List Move), 
    let finalState := moves.foldl applyMove (BoardState.mk initialOnes initialTwos)
    finalState.ones + finalState.twos = 1 → finalState.ones ≠ 1 :=
by sorry

end NUMINAMATH_CALUDE_game_end_not_one_l737_73780


namespace NUMINAMATH_CALUDE_polynomial_value_at_five_l737_73717

theorem polynomial_value_at_five (p : ℝ → ℝ) :
  (∃ a b c d : ℝ, ∀ x, p x = x^4 + a*x^3 + b*x^2 + c*x + d) →
  p 1 = 1 →
  p 2 = 2 →
  p 3 = 3 →
  p 4 = 4 →
  p 5 = 29 := by
sorry

end NUMINAMATH_CALUDE_polynomial_value_at_five_l737_73717


namespace NUMINAMATH_CALUDE_complement_P_intersect_Q_l737_73790

-- Define the sets P and Q
def P : Set ℝ := {x : ℝ | x^2 - 2*x ≥ 0}
def Q : Set ℝ := {x : ℝ | 1 < x ∧ x ≤ 2}

-- State the theorem
theorem complement_P_intersect_Q :
  (Set.univ \ P) ∩ Q = Set.Ioo 1 2 := by sorry

end NUMINAMATH_CALUDE_complement_P_intersect_Q_l737_73790


namespace NUMINAMATH_CALUDE_simplify_fraction_l737_73736

theorem simplify_fraction (k : ℝ) : 
  let expression := (6 * k + 12) / 6
  ∃ (c d : ℤ), expression = c * k + d ∧ (c : ℚ) / d = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_simplify_fraction_l737_73736


namespace NUMINAMATH_CALUDE_abs_eq_sqrt_square_l737_73762

theorem abs_eq_sqrt_square (x : ℝ) : |x - 1| = Real.sqrt ((x - 1)^2) := by sorry

end NUMINAMATH_CALUDE_abs_eq_sqrt_square_l737_73762


namespace NUMINAMATH_CALUDE_linear_function_increasing_l737_73724

/-- A linear function y = (2k-6)x + (2k+1) is increasing if and only if k > 3 -/
theorem linear_function_increasing (k : ℝ) :
  (∀ x y : ℝ, y = (2*k - 6)*x + (2*k + 1)) →
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → ((2*k - 6)*x₁ + (2*k + 1) < (2*k - 6)*x₂ + (2*k + 1))) ↔
  k > 3 :=
by sorry

end NUMINAMATH_CALUDE_linear_function_increasing_l737_73724


namespace NUMINAMATH_CALUDE_height_inequality_l737_73747

theorem height_inequality (a b : ℕ) (m : ℝ) (h_positive : a > 0 ∧ b > 0) 
  (h_right_triangle : m = (a * b : ℝ) / Real.sqrt ((a^2 + b^2 : ℕ) : ℝ)) :
  m ≤ Real.sqrt (((a^a * b^b : ℕ) : ℝ)^(1 / (a + b : ℝ))) / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_height_inequality_l737_73747


namespace NUMINAMATH_CALUDE_inequality_problem_l737_73710

/-- Given real numbers a, b, c satisfying c < b < a and ac < 0,
    prove that cb² < ca² is not necessarily true,
    while ab > ac, c(b-a) > 0, and ac(a-c) < 0 are always true. -/
theorem inequality_problem (a b c : ℝ) (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) :
  (∀ x y z : ℝ, x < y ∧ y < z ∧ x * z < 0 → x * y^2 < x * z^2 → False) ∧
  (a * b > a * c) ∧
  (c * (b - a) > 0) ∧
  (a * c * (a - c) < 0) :=
sorry

end NUMINAMATH_CALUDE_inequality_problem_l737_73710


namespace NUMINAMATH_CALUDE_orange_balls_count_l737_73718

def ball_problem (total red blue pink orange : ℕ) : Prop :=
  total = 50 ∧
  red = 20 ∧
  blue = 10 ∧
  total = red + blue + pink + orange ∧
  pink = 3 * orange

theorem orange_balls_count :
  ∀ total red blue pink orange : ℕ,
  ball_problem total red blue pink orange →
  orange = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_orange_balls_count_l737_73718


namespace NUMINAMATH_CALUDE_factorization_problems_l737_73779

theorem factorization_problems (m x n : ℝ) : 
  (m * x^2 - 2 * m^2 * x + m^3 = m * (x - m)^2) ∧ 
  (8 * m^2 * n + 2 * m * n = 2 * m * n * (4 * m + 1)) := by
  sorry

end NUMINAMATH_CALUDE_factorization_problems_l737_73779


namespace NUMINAMATH_CALUDE_puzzle_piece_increase_l737_73704

/-- Represents the number of puzzles John buys -/
def num_puzzles : ℕ := 3

/-- Represents the number of pieces in the first puzzle -/
def first_puzzle_pieces : ℕ := 1000

/-- Represents the total number of pieces in all puzzles -/
def total_pieces : ℕ := 4000

/-- Represents the percentage increase in pieces for the second and third puzzles -/
def percentage_increase : ℚ := 50

theorem puzzle_piece_increase :
  ∃ (second_puzzle_pieces third_puzzle_pieces : ℕ),
    second_puzzle_pieces = third_puzzle_pieces ∧
    second_puzzle_pieces = first_puzzle_pieces + (percentage_increase / 100) * first_puzzle_pieces ∧
    first_puzzle_pieces + second_puzzle_pieces + third_puzzle_pieces = total_pieces :=
by sorry

#check puzzle_piece_increase

end NUMINAMATH_CALUDE_puzzle_piece_increase_l737_73704


namespace NUMINAMATH_CALUDE_positive_integer_equation_l737_73735

theorem positive_integer_equation (m n p : ℕ+) : 
  3 * m.val + 3 / (n.val + 1 / p.val) = 17 → p = 2 := by
  sorry

end NUMINAMATH_CALUDE_positive_integer_equation_l737_73735


namespace NUMINAMATH_CALUDE_probability_three_defective_shipment_l737_73742

/-- The probability of selecting three defective smartphones from a shipment -/
def probability_three_defective (total : ℕ) (defective : ℕ) : ℚ :=
  (defective : ℚ) / total *
  ((defective - 1) : ℚ) / (total - 1) *
  ((defective - 2) : ℚ) / (total - 2)

/-- Theorem stating the probability of selecting three defective smartphones
    from a shipment of 400 smartphones, of which 150 are defective -/
theorem probability_three_defective_shipment :
  probability_three_defective 400 150 = 150 / 400 * 149 / 399 * 148 / 398 :=
by sorry

end NUMINAMATH_CALUDE_probability_three_defective_shipment_l737_73742


namespace NUMINAMATH_CALUDE_empire_state_building_height_l737_73798

/-- The height of the Empire State Building to the top floor, in feet -/
def height_to_top_floor : ℕ := 1250

/-- The height of the antenna spire, in feet -/
def antenna_height : ℕ := 204

/-- The total height of the Empire State Building, in feet -/
def total_height : ℕ := height_to_top_floor + antenna_height

/-- Theorem stating that the total height of the Empire State Building is 1454 feet -/
theorem empire_state_building_height : total_height = 1454 := by
  sorry

end NUMINAMATH_CALUDE_empire_state_building_height_l737_73798


namespace NUMINAMATH_CALUDE_integer_triple_divisibility_l737_73703

theorem integer_triple_divisibility (a b c : ℕ+) : 
  (∃ k₁ k₂ k₃ : ℕ+, (a + 1 : ℕ) = k₁ * b ∧ (b + 1 : ℕ) = k₂ * c ∧ (c + 1 : ℕ) = k₃ * a) →
  ((a, b, c) = (1, 1, 1) ∨ (a, b, c) = (1, 2, 1) ∨ (a, b, c) = (1, 1, 2) ∨ (a, b, c) = (2, 1, 1)) :=
by sorry


end NUMINAMATH_CALUDE_integer_triple_divisibility_l737_73703


namespace NUMINAMATH_CALUDE_max_table_height_l737_73755

/-- Given a triangle DEF with sides 25, 28, and 31, prove that the maximum possible height h'
    of a table constructed from this triangle is equal to 4√77 / 53. -/
theorem max_table_height (DE EF FD : ℝ) (h_DE : DE = 25) (h_EF : EF = 28) (h_FD : FD = 31) :
  let s := (DE + EF + FD) / 2
  let area := Real.sqrt (s * (s - DE) * (s - EF) * (s - FD))
  let h_DE := 2 * area / DE
  let h_EF := 2 * area / EF
  ∃ h' : ℝ, h' = (h_DE * h_EF) / (h_DE + h_EF) ∧ h' = 4 * Real.sqrt 77 / 53 :=
by sorry


end NUMINAMATH_CALUDE_max_table_height_l737_73755


namespace NUMINAMATH_CALUDE_expected_shots_value_l737_73751

/-- The probability of hitting the target -/
def p : ℝ := 0.8

/-- The maximum number of bullets -/
def max_shots : ℕ := 3

/-- The expected number of shots -/
def expected_shots : ℝ := p + 2 * (1 - p) * p + 3 * (1 - p) * (1 - p)

/-- Theorem stating that the expected number of shots is 1.24 -/
theorem expected_shots_value : expected_shots = 1.24 := by
  sorry

end NUMINAMATH_CALUDE_expected_shots_value_l737_73751


namespace NUMINAMATH_CALUDE_max_pieces_with_three_cuts_l737_73728

/-- The maximum number of identical pieces a cake can be divided into with a given number of cuts. -/
def max_cake_pieces (cuts : ℕ) : ℕ := 2^cuts

/-- Theorem: The maximum number of identical pieces a cake can be divided into with 3 cuts is 8. -/
theorem max_pieces_with_three_cuts :
  max_cake_pieces 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_max_pieces_with_three_cuts_l737_73728


namespace NUMINAMATH_CALUDE_balloon_permutations_l737_73725

def balloon_arrangements : ℕ := 1260

theorem balloon_permutations :
  let total_letters : ℕ := 7
  let repeated_l : ℕ := 2
  let repeated_o : ℕ := 2
  let unique_letters : ℕ := 3
  (total_letters = repeated_l + repeated_o + unique_letters) →
  (balloon_arrangements = (Nat.factorial total_letters) / ((Nat.factorial repeated_l) * (Nat.factorial repeated_o))) :=
by sorry

end NUMINAMATH_CALUDE_balloon_permutations_l737_73725


namespace NUMINAMATH_CALUDE_highest_powers_sum_12_factorial_l737_73768

theorem highest_powers_sum_12_factorial : 
  let n := 12
  let factorial_n := n.factorial
  let highest_power_of_10 := (factorial_n.factorization 2).min (factorial_n.factorization 5)
  let highest_power_of_6 := (factorial_n.factorization 2).min (factorial_n.factorization 3)
  highest_power_of_10 + highest_power_of_6 = 7 := by
  sorry

end NUMINAMATH_CALUDE_highest_powers_sum_12_factorial_l737_73768


namespace NUMINAMATH_CALUDE_trajectory_of_equidistant_complex_l737_73722

theorem trajectory_of_equidistant_complex (z : ℂ) :
  Complex.abs (z + 1 - Complex.I) = Complex.abs (z - 1 + Complex.I) →
  z.re = z.im :=
by sorry

end NUMINAMATH_CALUDE_trajectory_of_equidistant_complex_l737_73722


namespace NUMINAMATH_CALUDE_books_in_box_l737_73775

/-- The number of books in a box given the total weight and weight per book -/
def number_of_books (total_weight weight_per_book : ℚ) : ℚ :=
  total_weight / weight_per_book

/-- Theorem stating that a box weighing 42 pounds with books weighing 3 pounds each contains 14 books -/
theorem books_in_box : number_of_books 42 3 = 14 := by
  sorry

end NUMINAMATH_CALUDE_books_in_box_l737_73775


namespace NUMINAMATH_CALUDE_marathon_time_l737_73797

theorem marathon_time (dean_time : ℝ) 
  (h1 : dean_time > 0)
  (h2 : dean_time * (2/3) * (1 + 1/3) + dean_time * (3/2) + dean_time = 23) : 
  dean_time = 23/3 := by
sorry

end NUMINAMATH_CALUDE_marathon_time_l737_73797


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l737_73756

theorem polynomial_evaluation (x : ℕ) (h : x = 4) :
  x^4 + x^3 + x^2 + x + 1 = 341 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l737_73756


namespace NUMINAMATH_CALUDE_figure_2010_squares_l737_73734

/-- The number of squares in a figure of the sequence -/
def num_squares (n : ℕ) : ℕ := 1 + 4 * (n - 1)

/-- The theorem stating that Figure 2010 contains 8037 squares -/
theorem figure_2010_squares : num_squares 2010 = 8037 := by
  sorry

end NUMINAMATH_CALUDE_figure_2010_squares_l737_73734


namespace NUMINAMATH_CALUDE_abs_neg_three_halves_l737_73759

theorem abs_neg_three_halves : |(-3/2 : ℚ)| = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_three_halves_l737_73759


namespace NUMINAMATH_CALUDE_cosine_properties_l737_73753

theorem cosine_properties (x : ℝ) : 
  (fun (x : ℝ) => Real.cos x) (Real.pi + x) = -(fun (x : ℝ) => Real.cos x) x ∧ 
  (fun (x : ℝ) => Real.cos x) (-x) = (fun (x : ℝ) => Real.cos x) x :=
by sorry

end NUMINAMATH_CALUDE_cosine_properties_l737_73753


namespace NUMINAMATH_CALUDE_b_age_is_twelve_l737_73746

/-- Given three people a, b, and c, where a is two years older than b, b is twice as old as c, 
    and the sum of their ages is 32, prove that b is 12 years old. -/
theorem b_age_is_twelve (a b c : ℕ) 
    (h1 : a = b + 2) 
    (h2 : b = 2 * c) 
    (h3 : a + b + c = 32) : 
  b = 12 := by sorry

end NUMINAMATH_CALUDE_b_age_is_twelve_l737_73746


namespace NUMINAMATH_CALUDE_school_C_sample_size_l737_73773

/-- Represents the number of teachers in each school -/
structure SchoolPopulation where
  schoolA : ℕ
  schoolB : ℕ
  schoolC : ℕ

/-- Calculates the sample size for a given school in stratified sampling -/
def stratifiedSampleSize (totalSample : ℕ) (schoolPop : SchoolPopulation) (schoolSize : ℕ) : ℕ :=
  (schoolSize * totalSample) / (schoolPop.schoolA + schoolPop.schoolB + schoolPop.schoolC)

/-- Theorem: The stratified sample size for school C is 10 -/
theorem school_C_sample_size :
  let totalSample : ℕ := 60
  let schoolPop : SchoolPopulation := { schoolA := 180, schoolB := 270, schoolC := 90 }
  stratifiedSampleSize totalSample schoolPop schoolPop.schoolC = 10 := by
  sorry


end NUMINAMATH_CALUDE_school_C_sample_size_l737_73773


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l737_73714

theorem polynomial_divisibility (c : ℤ) : 
  (∃ q : Polynomial ℤ, (X^2 + X + c) * q = X^13 - X + 106) ↔ c = 2 := by
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l737_73714


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l737_73741

/-- A quadratic function with vertex (2, 0) passing through (0, -50) has a = -12.5 -/
theorem quadratic_coefficient (a b c : ℝ) : 
  (∀ x y : ℝ, y = a * x^2 + b * x + c) →
  (2, 0) = (2, a * 2^2 + b * 2 + c) →
  (0, -50) = (0, a * 0^2 + b * 0 + c) →
  a = -12.5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l737_73741


namespace NUMINAMATH_CALUDE_probability_heart_then_spade_or_club_l737_73737

-- Define the total number of cards in a standard deck
def total_cards : ℕ := 52

-- Define the number of hearts in a standard deck
def num_hearts : ℕ := 13

-- Define the number of spades and clubs combined in a standard deck
def num_spades_clubs : ℕ := 26

-- Theorem statement
theorem probability_heart_then_spade_or_club :
  (num_hearts / total_cards) * (num_spades_clubs / (total_cards - 1)) = 13 / 102 := by
  sorry

end NUMINAMATH_CALUDE_probability_heart_then_spade_or_club_l737_73737


namespace NUMINAMATH_CALUDE_sphere_surface_area_change_l737_73702

theorem sphere_surface_area_change (r₁ r₂ : ℝ) (h : r₁ > 0) (h' : r₂ > 0) : 
  (π * r₂^2 = 4 * π * r₁^2) → (4 * π * r₂^2 = 4 * (4 * π * r₁^2)) := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_change_l737_73702


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l737_73731

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 222) 
  (h2 : a*b + b*c + c*a = 131) : 
  a + b + c = 22 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l737_73731


namespace NUMINAMATH_CALUDE_evaluate_expression_l737_73723

theorem evaluate_expression : 
  Real.sqrt 8 * 2^(3/2) + 18 / 3 * 3 - 6^(5/2) = 26 - 36 * Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l737_73723


namespace NUMINAMATH_CALUDE_power_of_product_l737_73732

theorem power_of_product (a : ℝ) : (2 * a) ^ 3 = 8 * a ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l737_73732


namespace NUMINAMATH_CALUDE_sum_in_base_5_l737_73758

/-- Given a base b, returns the value of a number in base 10 -/
def toBase10 (x : ℕ) (b : ℕ) : ℕ := sorry

/-- Given a base b, returns the square of a number in base b -/
def squareInBase (x : ℕ) (b : ℕ) : ℕ := sorry

/-- Given a base b, returns the sum of three numbers in base b -/
def sumInBase (x y z : ℕ) (b : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to another base -/
def fromBase10 (x : ℕ) (b : ℕ) : ℕ := sorry

theorem sum_in_base_5 (b : ℕ) : 
  (squareInBase 14 b + squareInBase 18 b + squareInBase 20 b = toBase10 2850 b) →
  (fromBase10 (sumInBase 14 18 20 b) 5 = 62) := by
  sorry

end NUMINAMATH_CALUDE_sum_in_base_5_l737_73758


namespace NUMINAMATH_CALUDE_circle_tangent_origin_l737_73726

-- Define the circle equation
def circle_equation (x y D E F : ℝ) : Prop :=
  x^2 + y^2 + D*x + E*y + F = 0

-- Define the tangency condition
def tangent_at_origin (D E F : ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ 
  ∀ (x y : ℝ), circle_equation x y D E F → x^2 + y^2 ≥ r^2 ∧
  circle_equation 0 0 D E F

-- Theorem statement
theorem circle_tangent_origin (D E F : ℝ) :
  tangent_at_origin D E F → D = 0 ∧ F = 0 ∧ E ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_origin_l737_73726


namespace NUMINAMATH_CALUDE_red_candies_count_l737_73708

theorem red_candies_count (green blue : ℕ) (prob_blue : ℚ) (red : ℕ) : 
  green = 5 → blue = 3 → prob_blue = 1/4 → 
  (blue : ℚ) / ((green : ℚ) + (blue : ℚ) + (red : ℚ)) = prob_blue →
  red = 4 := by
  sorry

end NUMINAMATH_CALUDE_red_candies_count_l737_73708


namespace NUMINAMATH_CALUDE_intersection_of_A_and_complement_of_B_l737_73757

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | ∃ y, y = Real.sqrt x}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = -x^2}

-- Theorem statement
theorem intersection_of_A_and_complement_of_B :
  A ∩ (U \ B) = {x : ℝ | x > 0} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_complement_of_B_l737_73757


namespace NUMINAMATH_CALUDE_megan_popsicle_consumption_l737_73771

/-- The number of popsicles Megan eats in a given time period -/
def popsicles_eaten (minutes_per_popsicle : ℕ) (total_minutes : ℕ) : ℕ :=
  (total_minutes / minutes_per_popsicle : ℕ)

/-- Converts hours and minutes to total minutes -/
def to_minutes (hours : ℕ) (minutes : ℕ) : ℕ :=
  hours * 60 + minutes

theorem megan_popsicle_consumption :
  popsicles_eaten 12 (to_minutes 6 45) = 33 := by
  sorry

end NUMINAMATH_CALUDE_megan_popsicle_consumption_l737_73771


namespace NUMINAMATH_CALUDE_incorrect_statement_about_real_square_roots_l737_73777

theorem incorrect_statement_about_real_square_roots :
  ¬ (∀ a b : ℝ, a < b ∧ b < 0 → ¬∃ x y : ℝ, x^2 = a ∧ y^2 = b) :=
sorry

end NUMINAMATH_CALUDE_incorrect_statement_about_real_square_roots_l737_73777


namespace NUMINAMATH_CALUDE_diophantine_equation_solvable_l737_73721

theorem diophantine_equation_solvable (n : ℤ) :
  ∃ (x y z : ℤ), x^3 + 2*y^2 + 4*z = n := by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solvable_l737_73721


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l737_73794

theorem sufficient_not_necessary (x a : ℝ) (h : x > 0) :
  (a = 4 → ∀ x > 0, x + a / x ≥ 4) ∧
  ¬(∀ x > 0, x + a / x ≥ 4 → a = 4) :=
by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l737_73794


namespace NUMINAMATH_CALUDE_beeswax_number_l737_73730

/-- Represents a mapping from characters to digits -/
def CodeMapping : Char → Nat
| 'A' => 1
| 'T' => 2
| 'Q' => 3
| 'B' => 4
| 'K' => 5
| 'X' => 6
| 'S' => 7
| 'W' => 8
| 'E' => 9
| 'P' => 0
| _ => 0

/-- Converts a string to a number using the code mapping -/
def stringToNumber (s : String) : Nat :=
  s.foldl (fun acc c => acc * 10 + CodeMapping c) 0

/-- The subtraction equation given in the problem -/
axiom subtraction_equation :
  stringToNumber "EASEBSBSX" - stringToNumber "BPWWKSETQ" = stringToNumber "KPEPWEKKQ"

/-- The main theorem to prove -/
theorem beeswax_number : stringToNumber "BEESWAX" = 4997816 := by
  sorry


end NUMINAMATH_CALUDE_beeswax_number_l737_73730


namespace NUMINAMATH_CALUDE_h_value_l737_73709

theorem h_value (h : ℝ) : 
  (∃ x y : ℝ, x^2 - 4*h*x = 5 ∧ y^2 - 4*h*y = 5 ∧ x^2 + y^2 = 34) → 
  |h| = Real.sqrt (3/2) := by
sorry

end NUMINAMATH_CALUDE_h_value_l737_73709


namespace NUMINAMATH_CALUDE_complex_product_quadrant_l737_73789

theorem complex_product_quadrant : 
  let z : ℂ := (1 + 3*I) * (3 - I)
  (z.re > 0 ∧ z.im > 0) := by sorry

end NUMINAMATH_CALUDE_complex_product_quadrant_l737_73789
