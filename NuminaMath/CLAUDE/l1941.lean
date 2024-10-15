import Mathlib

namespace NUMINAMATH_CALUDE_range_of_2a_minus_b_l1941_194116

theorem range_of_2a_minus_b (a b : ℝ) (h1 : a > b) (h2 : 2*a^2 - a*b - b^2 - 4 = 0) :
  ∃ (k : ℝ), k ≥ 8/3 ∧ 2*a - b = k :=
sorry

end NUMINAMATH_CALUDE_range_of_2a_minus_b_l1941_194116


namespace NUMINAMATH_CALUDE_property_set_characterization_l1941_194162

/-- The property that a^(n+1) ≡ a (mod n) holds for all integers a -/
def has_property (n : ℕ) : Prop :=
  ∀ a : ℤ, (a^(n+1) : ℤ) ≡ a [ZMOD n]

/-- The set of integers satisfying the property -/
def property_set : Set ℕ := {n | has_property n}

/-- Theorem stating that the set of integers satisfying the property is exactly {1, 2, 6, 42, 1806} -/
theorem property_set_characterization :
  property_set = {1, 2, 6, 42, 1806} := by sorry

end NUMINAMATH_CALUDE_property_set_characterization_l1941_194162


namespace NUMINAMATH_CALUDE_gym_guests_first_hour_l1941_194149

/-- The number of guests who entered the gym in the first hour -/
def first_hour_guests : ℕ := 50

/-- The total number of towels available -/
def total_towels : ℕ := 300

/-- The number of hours the gym is open -/
def open_hours : ℕ := 4

/-- The increase rate for the second hour -/
def second_hour_rate : ℚ := 1.2

/-- The increase rate for the third hour -/
def third_hour_rate : ℚ := 1.25

/-- The increase rate for the fourth hour -/
def fourth_hour_rate : ℚ := 4/3

/-- The number of towels that need to be washed at the end of the day -/
def towels_to_wash : ℕ := 285

theorem gym_guests_first_hour :
  first_hour_guests * (1 + second_hour_rate + second_hour_rate * third_hour_rate +
    second_hour_rate * third_hour_rate * fourth_hour_rate) = towels_to_wash :=
sorry

end NUMINAMATH_CALUDE_gym_guests_first_hour_l1941_194149


namespace NUMINAMATH_CALUDE_bus_tour_sales_l1941_194142

/-- Given a bus tour with senior and regular tickets, calculate the total sales amount. -/
theorem bus_tour_sales (total_tickets : ℕ) (senior_price regular_price : ℕ) (senior_tickets : ℕ) 
  (h1 : total_tickets = 65)
  (h2 : senior_price = 10)
  (h3 : regular_price = 15)
  (h4 : senior_tickets = 24)
  (h5 : senior_tickets ≤ total_tickets) :
  senior_tickets * senior_price + (total_tickets - senior_tickets) * regular_price = 855 := by
  sorry


end NUMINAMATH_CALUDE_bus_tour_sales_l1941_194142


namespace NUMINAMATH_CALUDE_movie_of_the_year_requirement_l1941_194144

theorem movie_of_the_year_requirement (total_members : ℕ) (fraction : ℚ) : total_members = 775 → fraction = 1/4 → ↑(⌈total_members * fraction⌉) = 194 := by
  sorry

end NUMINAMATH_CALUDE_movie_of_the_year_requirement_l1941_194144


namespace NUMINAMATH_CALUDE_perpendicular_lines_slope_l1941_194173

-- Define the slopes of two lines
def slope1 (k : ℝ) := k
def slope2 : ℝ := 2

-- Define the condition for perpendicular lines
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

-- Theorem statement
theorem perpendicular_lines_slope (k : ℝ) :
  perpendicular (slope1 k) slope2 → k = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_slope_l1941_194173


namespace NUMINAMATH_CALUDE_addition_preserves_inequality_l1941_194124

theorem addition_preserves_inequality (a b : ℝ) (h : a > b) : a + 2 > b + 2 := by
  sorry

end NUMINAMATH_CALUDE_addition_preserves_inequality_l1941_194124


namespace NUMINAMATH_CALUDE_sector_area_l1941_194187

theorem sector_area (r : ℝ) (α : ℝ) (h1 : r = 3) (h2 : α = 2) :
  (1 / 2 : ℝ) * r^2 * α = 9 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l1941_194187


namespace NUMINAMATH_CALUDE_complex_power_sum_l1941_194125

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_power_sum : 2 * (i^13 + i^18 + i^23 + i^28 + i^33) = 2 * i := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l1941_194125


namespace NUMINAMATH_CALUDE_joan_gemstone_count_l1941_194109

/-- Proves that Joan has 21 gemstone samples given the conditions of the problem -/
theorem joan_gemstone_count :
  ∀ (minerals_yesterday minerals_today gemstones : ℕ),
    gemstones = minerals_yesterday / 2 →
    minerals_today = minerals_yesterday + 6 →
    minerals_today = 48 →
    gemstones = 21 := by
  sorry

end NUMINAMATH_CALUDE_joan_gemstone_count_l1941_194109


namespace NUMINAMATH_CALUDE_triangle_properties_l1941_194189

theorem triangle_properties (x : ℝ) (h : x > 0) :
  let a := 5*x
  let b := 12*x
  let c := 13*x
  (a^2 + b^2 = c^2) ∧ (∃ q : ℚ, (a / b : ℝ) = q) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l1941_194189


namespace NUMINAMATH_CALUDE_other_person_money_l1941_194140

/-- If Mia has $110 and this amount is $20 more than twice as much money as someone else, then that person has $45. -/
theorem other_person_money (mia_money : ℕ) (other_money : ℕ) : 
  mia_money = 110 → mia_money = 2 * other_money + 20 → other_money = 45 := by
  sorry

end NUMINAMATH_CALUDE_other_person_money_l1941_194140


namespace NUMINAMATH_CALUDE_truck_toll_theorem_l1941_194147

/-- Calculates the toll for a truck given the number of axles -/
def toll (x : ℕ) : ℚ := 2.5 + 0.5 * (x - 2)

/-- Calculates the number of axles for a truck given the total number of wheels,
    the number of wheels on the front axle, and the number of wheels on each other axle -/
def calculateAxles (totalWheels frontAxleWheels otherAxleWheels : ℕ) : ℕ :=
  1 + (totalWheels - frontAxleWheels) / otherAxleWheels

theorem truck_toll_theorem :
  let totalWheels : ℕ := 18
  let frontAxleWheels : ℕ := 2
  let otherAxleWheels : ℕ := 4
  let numAxles : ℕ := calculateAxles totalWheels frontAxleWheels otherAxleWheels
  toll numAxles = 4 := by
  sorry

end NUMINAMATH_CALUDE_truck_toll_theorem_l1941_194147


namespace NUMINAMATH_CALUDE_kenny_sunday_jumping_jacks_l1941_194175

/-- Represents the number of jumping jacks Kenny did on each day of the week -/
structure WeeklyJumpingJacks where
  sunday : ℕ
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ
  saturday : ℕ

/-- Calculates the total number of jumping jacks for a week -/
def totalJumpingJacks (week : WeeklyJumpingJacks) : ℕ :=
  week.sunday + week.monday + week.tuesday + week.wednesday + week.thursday + week.friday + week.saturday

theorem kenny_sunday_jumping_jacks 
  (lastWeek : ℕ) 
  (thisWeek : WeeklyJumpingJacks) 
  (h1 : lastWeek = 324)
  (h2 : thisWeek.tuesday = 0)
  (h3 : thisWeek.wednesday = 123)
  (h4 : thisWeek.thursday = 64)
  (h5 : thisWeek.friday = 23)
  (h6 : thisWeek.saturday = 61)
  (h7 : thisWeek.monday = 20 ∨ thisWeek.sunday = 20)
  (h8 : totalJumpingJacks thisWeek > lastWeek) :
  thisWeek.sunday = 33 := by
  sorry

end NUMINAMATH_CALUDE_kenny_sunday_jumping_jacks_l1941_194175


namespace NUMINAMATH_CALUDE_correct_probability_l1941_194118

/-- The number of options for the first three digits -/
def first_three_options : ℕ := 3

/-- The number of remaining digits to arrange -/
def remaining_digits : ℕ := 5

/-- The probability of correctly guessing the phone number -/
def probability_correct_guess : ℚ := 1 / (first_three_options * remaining_digits.factorial)

theorem correct_probability :
  probability_correct_guess = 1 / 360 := by
  sorry

end NUMINAMATH_CALUDE_correct_probability_l1941_194118


namespace NUMINAMATH_CALUDE_tabs_per_window_l1941_194129

theorem tabs_per_window (num_browsers : ℕ) (windows_per_browser : ℕ) (total_tabs : ℕ) :
  num_browsers = 2 →
  windows_per_browser = 3 →
  total_tabs = 60 →
  total_tabs / (num_browsers * windows_per_browser) = 10 :=
by sorry

end NUMINAMATH_CALUDE_tabs_per_window_l1941_194129


namespace NUMINAMATH_CALUDE_percentage_problem_l1941_194180

/-- Proves that the percentage is 50% given the problem conditions -/
theorem percentage_problem (x : ℝ) : 
  (x / 100) * 150 = 75 / 100 → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1941_194180


namespace NUMINAMATH_CALUDE_find_a_value_l1941_194188

theorem find_a_value (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = a * x^3 + 9 * x^2 + 6 * x - 7) →
  (((fun x ↦ 3 * a * x^2 + 18 * x + 6) (-1)) = 4) →
  a = 16/3 := by
  sorry

end NUMINAMATH_CALUDE_find_a_value_l1941_194188


namespace NUMINAMATH_CALUDE_product_zero_l1941_194126

/-- Given two real numbers x and y satisfying x - y = 6 and x³ - y³ = 162, their product xy equals 0. -/
theorem product_zero (x y : ℝ) (h1 : x - y = 6) (h2 : x^3 - y^3 = 162) : x * y = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_zero_l1941_194126


namespace NUMINAMATH_CALUDE_log_x3y2_equals_2_l1941_194183

theorem log_x3y2_equals_2 
  (x y : ℝ) 
  (h1 : Real.log (x * y^2) = 2) 
  (h2 : Real.log (x^2 * y^3) = 3) : 
  Real.log (x^3 * y^2) = 2 := by
sorry

end NUMINAMATH_CALUDE_log_x3y2_equals_2_l1941_194183


namespace NUMINAMATH_CALUDE_height_difference_l1941_194196

def elm_height : ℚ := 35 / 3
def oak_height : ℚ := 107 / 6

theorem height_difference : oak_height - elm_height = 37 / 6 := by
  sorry

end NUMINAMATH_CALUDE_height_difference_l1941_194196


namespace NUMINAMATH_CALUDE_bus_length_calculation_l1941_194160

/-- Calculates the length of a bus given its speed, the speed of a person moving in the opposite direction, and the time it takes for the bus to pass the person. -/
theorem bus_length_calculation (bus_speed : ℝ) (skater_speed : ℝ) (passing_time : ℝ) :
  bus_speed = 40 ∧ skater_speed = 8 ∧ passing_time = 1.125 →
  (bus_speed + skater_speed) * passing_time * (5 / 18) = 45 :=
by sorry

end NUMINAMATH_CALUDE_bus_length_calculation_l1941_194160


namespace NUMINAMATH_CALUDE_inequalities_hold_l1941_194115

theorem inequalities_hold (a b c : ℝ) (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) :
  (b / a > c / a) ∧ ((b - a) / c > 0) ∧ ((a - c) / (a * c) < 0) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_hold_l1941_194115


namespace NUMINAMATH_CALUDE_exists_polygon_different_centers_l1941_194193

/-- A polygon is represented by a list of its vertices --/
def Polygon := List (ℝ × ℝ)

/-- Calculate the center of gravity of a polygon's vertices --/
noncomputable def centerOfGravityVertices (p : Polygon) : ℝ × ℝ := sorry

/-- Calculate the center of gravity of a polygon plate --/
noncomputable def centerOfGravityPlate (p : Polygon) : ℝ × ℝ := sorry

/-- The theorem stating that there exists a polygon where the centers of gravity don't coincide --/
theorem exists_polygon_different_centers : 
  ∃ (p : Polygon), centerOfGravityVertices p ≠ centerOfGravityPlate p := by sorry

end NUMINAMATH_CALUDE_exists_polygon_different_centers_l1941_194193


namespace NUMINAMATH_CALUDE_total_cost_is_30_l1941_194137

def silverware_cost : ℝ := 20
def dinner_plates_cost_ratio : ℝ := 0.5

def total_cost : ℝ :=
  silverware_cost + (silverware_cost * dinner_plates_cost_ratio)

theorem total_cost_is_30 :
  total_cost = 30 := by sorry

end NUMINAMATH_CALUDE_total_cost_is_30_l1941_194137


namespace NUMINAMATH_CALUDE_sum_product_equality_l1941_194100

theorem sum_product_equality : 3 * 11 + 3 * 12 + 3 * 15 + 11 = 125 := by
  sorry

end NUMINAMATH_CALUDE_sum_product_equality_l1941_194100


namespace NUMINAMATH_CALUDE_counterexample_exists_l1941_194158

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

theorem counterexample_exists : ∃ n : ℕ, 
  (sum_of_digits n) % 9 = 0 ∧ 
  n % 3 = 0 ∧ 
  n % 9 ≠ 0 := by sorry

end NUMINAMATH_CALUDE_counterexample_exists_l1941_194158


namespace NUMINAMATH_CALUDE_bookstore_calculator_sales_l1941_194156

theorem bookstore_calculator_sales
  (price1 : ℕ) (price2 : ℕ) (total_sales : ℕ) (quantity2 : ℕ)
  (h1 : price1 = 15)
  (h2 : price2 = 67)
  (h3 : total_sales = 3875)
  (h4 : quantity2 = 35)
  (h5 : ∃ quantity1 : ℕ, price1 * quantity1 + price2 * quantity2 = total_sales) :
  ∃ total_quantity : ℕ, total_quantity = quantity2 + (total_sales - price2 * quantity2) / price1 ∧
                        total_quantity = 137 := by
  sorry

end NUMINAMATH_CALUDE_bookstore_calculator_sales_l1941_194156


namespace NUMINAMATH_CALUDE_triangle_side_range_l1941_194152

theorem triangle_side_range (a : ℝ) : 
  (3 : ℝ) > 0 ∧ (5 : ℝ) > 0 ∧ (1 - 2*a : ℝ) > 0 ∧
  3 + 5 > 1 - 2*a ∧
  3 + (1 - 2*a) > 5 ∧
  5 + (1 - 2*a) > 3 →
  -7/2 < a ∧ a < -1/2 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_range_l1941_194152


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l1941_194127

theorem stratified_sampling_theorem (total_population : ℕ) (category_size : ℕ) (sample_size : ℕ) 
  (h1 : total_population = 100)
  (h2 : category_size = 30)
  (h3 : sample_size = 20) :
  (category_size : ℚ) / total_population * sample_size = 6 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l1941_194127


namespace NUMINAMATH_CALUDE_emmas_drive_speed_l1941_194168

/-- Proves that given the conditions of Emma's drive, her average speed during the last 40 minutes was 75 mph -/
theorem emmas_drive_speed (total_distance : ℝ) (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ) 
  (h1 : total_distance = 120)
  (h2 : total_time = 2)
  (h3 : speed1 = 50)
  (h4 : speed2 = 55) :
  let segment_time := total_time / 3
  let speed3 := (total_distance - (speed1 + speed2) * segment_time) / segment_time
  speed3 = 75 := by sorry

end NUMINAMATH_CALUDE_emmas_drive_speed_l1941_194168


namespace NUMINAMATH_CALUDE_quadratic_function_bound_l1941_194198

def f (p q : ℝ) (x : ℝ) : ℝ := x^2 + p*x + q

theorem quadratic_function_bound (p q : ℝ) :
  (max (|f p q 1|) (max (|f p q 2|) (|f p q 3|))) ≥ (1/2 : ℝ) := by sorry

end NUMINAMATH_CALUDE_quadratic_function_bound_l1941_194198


namespace NUMINAMATH_CALUDE_area_max_cyclic_l1941_194154

/-- A quadrilateral with sides a, b, c, d and diagonals e, f -/
structure Quadrilateral (α : Type*) [LinearOrderedField α] :=
  (a b c d e f : α)

/-- The area of a quadrilateral -/
def area {α : Type*} [LinearOrderedField α] (q : Quadrilateral α) : α :=
  ((q.b + q.d - q.a + q.c) * (q.b + q.d + q.a - q.c) * 
   (q.a + q.c - q.b + q.d) * (q.a + q.b + q.c - q.d) - 
   4 * (q.a * q.c + q.b * q.d - q.e * q.f) * (q.a * q.c + q.b * q.d + q.e * q.f)) / 16

/-- The theorem stating that the area is maximized when ef = ac + bd -/
theorem area_max_cyclic {α : Type*} [LinearOrderedField α] (q : Quadrilateral α) :
  area q ≤ area { q with e := (q.a * q.c + q.b * q.d) / q.f, f := q.f } :=
sorry

end NUMINAMATH_CALUDE_area_max_cyclic_l1941_194154


namespace NUMINAMATH_CALUDE_largest_number_value_l1941_194107

theorem largest_number_value (a b c : ℝ) 
  (h_order : a < b ∧ b < c)
  (h_sum : a + b + c = 100)
  (h_diff_large : c - b = 10)
  (h_diff_small : b - a = 5) :
  c = 41.67 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_value_l1941_194107


namespace NUMINAMATH_CALUDE_work_completion_time_l1941_194184

/-- Given that two workers 'a' and 'b' can complete a job together in 4 days,
    and 'a' alone can complete the job in 12 days, prove that 'b' alone
    can complete the job in 6 days. -/
theorem work_completion_time (work_rate_a : ℚ) (work_rate_b : ℚ) :
  work_rate_a + work_rate_b = 1 / 4 →
  work_rate_a = 1 / 12 →
  work_rate_b = 1 / 6 := by
sorry

end NUMINAMATH_CALUDE_work_completion_time_l1941_194184


namespace NUMINAMATH_CALUDE_kopeck_ruble_exchange_l1941_194131

/-- Represents the denominations of coins available in kopecks -/
def Denominations : List ℕ := [1, 2, 5, 10, 20, 50, 100]

/-- Represents a valid coin exchange -/
def IsValidExchange (amount : ℕ) (coinCount : ℕ) : Prop :=
  ∃ (coins : List ℕ), 
    (coins.length = coinCount) ∧ 
    (coins.sum = amount) ∧
    (∀ c ∈ coins, c ∈ Denominations)

/-- The main theorem: if A kopecks can be exchanged with B coins,
    then B rubles can be exchanged with A coins -/
theorem kopeck_ruble_exchange 
  (A B : ℕ) 
  (h : IsValidExchange A B) : 
  IsValidExchange (100 * B) A := by
  sorry

#check kopeck_ruble_exchange

end NUMINAMATH_CALUDE_kopeck_ruble_exchange_l1941_194131


namespace NUMINAMATH_CALUDE_min_sticks_removal_part_a_result_part_b_result_l1941_194106

/-- Represents a rectangular fence made of sticks -/
structure Fence where
  m : Nat
  n : Nat
  sticks : Nat

/-- The number of ants in a fence is equal to the number of 1x1 squares -/
def num_ants (f : Fence) : Nat := f.m * f.n

/-- The minimum number of sticks to remove for all ants to escape -/
def min_sticks_to_remove (f : Fence) : Nat := num_ants f

/-- Theorem: The minimum number of sticks to remove for all ants to escape
    is equal to the number of ants in the fence -/
theorem min_sticks_removal (f : Fence) :
  min_sticks_to_remove f = num_ants f :=
by sorry

/-- Corollary: For a 1x4 fence with 13 sticks, 4 sticks need to be removed -/
theorem part_a_result :
  min_sticks_to_remove ⟨1, 4, 13⟩ = 4 :=
by sorry

/-- Corollary: For a 4x4 fence with 24 sticks, 9 sticks need to be removed -/
theorem part_b_result :
  min_sticks_to_remove ⟨4, 4, 24⟩ = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_sticks_removal_part_a_result_part_b_result_l1941_194106


namespace NUMINAMATH_CALUDE_point_not_on_line_l1941_194176

theorem point_not_on_line (p q : ℝ) (h : p * q < 0) :
  -101 ≠ 21 * p + q := by sorry

end NUMINAMATH_CALUDE_point_not_on_line_l1941_194176


namespace NUMINAMATH_CALUDE_endomorphism_characterization_l1941_194113

/-- An endomorphism of ℤ² --/
def Endomorphism : Type := ℤ × ℤ → ℤ × ℤ

/-- The group operation on ℤ² --/
def add : (ℤ × ℤ) → (ℤ × ℤ) → (ℤ × ℤ) :=
  λ (a b : ℤ × ℤ) => (a.1 + b.1, a.2 + b.2)

/-- A homomorphism respects the group operation --/
def is_homomorphism (φ : Endomorphism) : Prop :=
  ∀ a b : ℤ × ℤ, φ (add a b) = add (φ a) (φ b)

/-- Linear representation of an endomorphism --/
def linear_form (u v : ℤ × ℤ) : Endomorphism :=
  λ (x : ℤ × ℤ) => (x.1 * u.1 + x.2 * v.1, x.1 * u.2 + x.2 * v.2)

/-- Main theorem: Characterization of endomorphisms of ℤ² --/
theorem endomorphism_characterization :
  ∀ φ : Endomorphism, 
    is_homomorphism φ ↔ ∃ u v : ℤ × ℤ, φ = linear_form u v :=
by sorry

end NUMINAMATH_CALUDE_endomorphism_characterization_l1941_194113


namespace NUMINAMATH_CALUDE_tan_equation_l1941_194186

theorem tan_equation (α : Real) (h : Real.tan α = 2) :
  1 / (2 * Real.sin α * Real.cos α + Real.cos α ^ 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_equation_l1941_194186


namespace NUMINAMATH_CALUDE_joan_balloons_l1941_194105

theorem joan_balloons (initial : ℕ) (received : ℕ) (total : ℕ) : 
  initial = 8 → received = 2 → total = initial + received → total = 10 := by
  sorry

end NUMINAMATH_CALUDE_joan_balloons_l1941_194105


namespace NUMINAMATH_CALUDE_bee_flight_count_l1941_194133

/-- Represents the energy content of honey in terms of bee flight distance -/
def honey_energy : ℕ := 7000

/-- Represents the amount of honey available -/
def honey_amount : ℕ := 10

/-- Represents the distance each bee should fly -/
def flight_distance : ℕ := 1

/-- Theorem: Given the energy content of honey and the amount available,
    calculate the number of bees that can fly a specified distance -/
theorem bee_flight_count :
  (honey_energy * honey_amount) / flight_distance = 70000 := by
  sorry

end NUMINAMATH_CALUDE_bee_flight_count_l1941_194133


namespace NUMINAMATH_CALUDE_even_sum_difference_l1941_194191

def sum_even_range (a b : ℕ) : ℕ :=
  let n := (b - a) / 2 + 1
  n * (a + b) / 2

theorem even_sum_difference : sum_even_range 102 150 - sum_even_range 2 50 = 2500 := by
  sorry

end NUMINAMATH_CALUDE_even_sum_difference_l1941_194191


namespace NUMINAMATH_CALUDE_sally_pokemon_cards_l1941_194171

theorem sally_pokemon_cards (initial : ℕ) (dan_gift : ℕ) (sally_bought : ℕ) : 
  initial = 27 → dan_gift = 41 → sally_bought = 20 → 
  initial + dan_gift + sally_bought = 88 := by
sorry

end NUMINAMATH_CALUDE_sally_pokemon_cards_l1941_194171


namespace NUMINAMATH_CALUDE_probability_diamond_then_ace_l1941_194174

/-- Represents a standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Represents the number of diamonds in a standard deck -/
def DiamondCount : ℕ := 13

/-- Represents the number of aces in a standard deck -/
def AceCount : ℕ := 4

/-- Represents the remaining deck after one card (not a diamond ace) has been dealt -/
def RemainingDeck : ℕ := StandardDeck - 1

/-- Represents the number of diamonds (excluding ace) in the remaining deck -/
def RemainingDiamonds : ℕ := DiamondCount - 1

theorem probability_diamond_then_ace :
  (RemainingDiamonds : ℚ) / RemainingDeck * AceCount / (RemainingDeck - 1) = 24 / 1275 := by
  sorry

end NUMINAMATH_CALUDE_probability_diamond_then_ace_l1941_194174


namespace NUMINAMATH_CALUDE_total_hours_worked_l1941_194166

theorem total_hours_worked (saturday_hours sunday_hours : ℕ) 
  (h1 : saturday_hours = 6) 
  (h2 : sunday_hours = 4) : 
  saturday_hours + sunday_hours = 10 := by
  sorry

end NUMINAMATH_CALUDE_total_hours_worked_l1941_194166


namespace NUMINAMATH_CALUDE_solve_for_y_l1941_194165

theorem solve_for_y (x y z : ℝ) (h1 : x = 1) (h2 : z = 3) (h3 : x^2 * y * z - x * y * z^2 = 6) : y = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l1941_194165


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l1941_194139

theorem cubic_equation_solution (x : ℝ) (hx : x^3 + 6 * (x / (x - 3))^3 = 135) :
  let y := ((x - 3)^3 * (x + 4)) / (3 * x - 4)
  y = 0 ∨ y = 23382 / 122 := by
sorry


end NUMINAMATH_CALUDE_cubic_equation_solution_l1941_194139


namespace NUMINAMATH_CALUDE_vitya_older_probability_l1941_194128

/-- The number of days in June -/
def june_days : ℕ := 30

/-- The probability that Vitya is at least one day older than Masha -/
def probability_vitya_older (june_days : ℕ) : ℚ :=
  (june_days - 1).choose 2 / (june_days * june_days)

/-- Theorem stating the probability that Vitya is at least one day older than Masha -/
theorem vitya_older_probability :
  probability_vitya_older june_days = 29 / 60 := by
  sorry

end NUMINAMATH_CALUDE_vitya_older_probability_l1941_194128


namespace NUMINAMATH_CALUDE_fixed_point_exponential_function_l1941_194138

/-- For any positive real number a, the function f(x) = 2 + a^(x-1) always passes through the point (1, 3) -/
theorem fixed_point_exponential_function (a : ℝ) (ha : a > 0) :
  let f : ℝ → ℝ := λ x ↦ 2 + a^(x - 1)
  f 1 = 3 := by sorry

end NUMINAMATH_CALUDE_fixed_point_exponential_function_l1941_194138


namespace NUMINAMATH_CALUDE_inequality_relation_l1941_194195

theorem inequality_relation (x y : ℝ) :
  (x^3 + x > x^2*y + y) → (x - y > -1) ∧
  ¬(∀ x y : ℝ, x - y > -1 → x^3 + x > x^2*y + y) :=
by sorry

end NUMINAMATH_CALUDE_inequality_relation_l1941_194195


namespace NUMINAMATH_CALUDE_fraction_equality_l1941_194167

theorem fraction_equality : (1/3 - 1/4) / (1/2 - 1/3) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1941_194167


namespace NUMINAMATH_CALUDE_geometric_sum_example_l1941_194161

/-- Sum of a geometric sequence -/
def geometric_sum (a₀ : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a₀ * (1 - r^n) / (1 - r)

/-- Proof of the sum of the first eight terms of a specific geometric sequence -/
theorem geometric_sum_example : geometric_sum (1/3) (1/3) 8 = 3280/6561 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_example_l1941_194161


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1941_194197

/-- Given a quadratic equation and an isosceles triangle, prove the perimeter is 5 -/
theorem isosceles_triangle_perimeter (k : ℝ) : 
  let equation := fun x : ℝ => x^2 - (k+2)*x + 2*k
  ∃ (b c : ℝ), 
    equation b = 0 ∧ 
    equation c = 0 ∧ 
    b = c ∧ 
    b + c + 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1941_194197


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l1941_194178

def U : Set Nat := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set Nat := {3, 4, 5}
def B : Set Nat := {1, 3, 6}

theorem complement_intersection_theorem :
  (U \ A) ∩ (U \ B) = {2, 7, 8} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l1941_194178


namespace NUMINAMATH_CALUDE_brick_length_is_correct_l1941_194117

/-- The length of a brick in centimeters -/
def brick_length : ℝ := 25

/-- The width of a brick in centimeters -/
def brick_width : ℝ := 11.25

/-- The height of a brick in centimeters -/
def brick_height : ℝ := 6

/-- The length of the wall in centimeters -/
def wall_length : ℝ := 700

/-- The height of the wall in centimeters -/
def wall_height : ℝ := 600

/-- The width of the wall in centimeters -/
def wall_width : ℝ := 22.5

/-- The number of bricks needed to build the wall -/
def num_bricks : ℕ := 5600

/-- Theorem stating that the brick length is correct given the wall and brick dimensions -/
theorem brick_length_is_correct :
  brick_length * brick_width * brick_height * num_bricks = wall_length * wall_height * wall_width := by
  sorry

end NUMINAMATH_CALUDE_brick_length_is_correct_l1941_194117


namespace NUMINAMATH_CALUDE_isosceles_triangle_angle_measure_l1941_194141

/-- 
An isosceles triangle with one angle 20% smaller than a right angle 
has its two largest angles measuring 54 degrees each.
-/
theorem isosceles_triangle_angle_measure : 
  ∀ (a b c : ℝ),
  -- The triangle is isosceles
  a = b →
  -- The sum of angles in a triangle is 180°
  a + b + c = 180 →
  -- One angle (c) is 20% smaller than a right angle (90°)
  c = 0.8 * 90 →
  -- Each of the two largest angles (a and b) measures 54°
  a = 54 ∧ b = 54 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_angle_measure_l1941_194141


namespace NUMINAMATH_CALUDE_dividend_calculation_l1941_194192

/-- The dividend calculation problem -/
theorem dividend_calculation (divisor quotient remainder : ℝ) 
  (h_divisor : divisor = 176.22471910112358)
  (h_quotient : quotient = 89)
  (h_remainder : remainder = 14) :
  divisor * quotient + remainder = 15697.799999999998 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l1941_194192


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1941_194177

theorem polynomial_simplification (m : ℝ) : 
  (∀ x y : ℝ, (2 * m * x^2 + 4 * x^2 + 3 * x + 1) - (6 * x^2 - 4 * y^2 + 3 * x) = 4 * y^2 + 1) ↔ 
  m = 1 := by
sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1941_194177


namespace NUMINAMATH_CALUDE_interest_difference_l1941_194146

/-- Calculates the difference between compound interest and simple interest -/
theorem interest_difference (principal : ℝ) (rate : ℝ) (time : ℝ) :
  let compound_interest := principal * (1 + rate)^time - principal
  let simple_interest := principal * rate * time
  principal = 6500 ∧ rate = 0.04 ∧ time = 2 →
  compound_interest - simple_interest = 9.40 := by sorry

end NUMINAMATH_CALUDE_interest_difference_l1941_194146


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l1941_194101

/-- An arithmetic sequence with common difference 2 -/
def arithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + 2

/-- Condition that a_1, a_3, and a_4 form a geometric sequence -/
def geometricSubsequence (a : ℕ → ℤ) : Prop :=
  (a 3) ^ 2 = a 1 * a 4

theorem arithmetic_geometric_sequence (a : ℕ → ℤ) 
  (h_arith : arithmeticSequence a) 
  (h_geom : geometricSubsequence a) : 
  a 2 = -6 := by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l1941_194101


namespace NUMINAMATH_CALUDE_martinez_family_height_l1941_194122

def chiquita_height : ℝ := 5

def mr_martinez_height : ℝ := chiquita_height + 2

def mrs_martinez_height : ℝ := chiquita_height - 1

def son_height : ℝ := chiquita_height + 3

def combined_height : ℝ := chiquita_height + mr_martinez_height + mrs_martinez_height + son_height

theorem martinez_family_height : combined_height = 24 := by
  sorry

end NUMINAMATH_CALUDE_martinez_family_height_l1941_194122


namespace NUMINAMATH_CALUDE_problem_statement_l1941_194182

theorem problem_statement (a b c d : ℝ) :
  Real.sqrt (a + b + c + d) + Real.sqrt (a^2 - 2*a + 3 - b) - Real.sqrt (b - c^2 + 4*c - 8) = 3 →
  a - b + c - d = -7 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1941_194182


namespace NUMINAMATH_CALUDE_sequence_solution_l1941_194172

def sequence_problem (b : Fin 6 → ℝ) : Prop :=
  (∀ n : Fin 3, b (2 * n) = b (2 * n - 1) ^ 2) ∧
  (∀ n : Fin 2, b (2 * n + 1) = (b (2 * n) * b (2 * n - 1)) ^ 2) ∧
  b 6 = 65536 ∧ b 5 = 256 ∧ b 4 = 16 ∧
  (∀ i : Fin 6, 0 ≤ b i)

theorem sequence_solution (b : Fin 6 → ℝ) (h : sequence_problem b) : b 1 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sequence_solution_l1941_194172


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1941_194103

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- Given an arithmetic sequence a_n where a₁ + a₂ = 5 and a₃ + a₄ = 7,
    prove that a₅ + a₆ = 9 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a)
  (h_sum1 : a 1 + a 2 = 5)
  (h_sum2 : a 3 + a 4 = 7) :
  a 5 + a 6 = 9 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1941_194103


namespace NUMINAMATH_CALUDE_lake_distance_proof_l1941_194112

def lake_distance : Set ℝ := {d | d > 9 ∧ d < 10}

theorem lake_distance_proof (d : ℝ) :
  (¬ (d ≥ 10)) ∧ (¬ (d ≤ 9)) ∧ (d ≠ 7) ↔ d ∈ lake_distance := by
  sorry

end NUMINAMATH_CALUDE_lake_distance_proof_l1941_194112


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1941_194150

def complex_one_plus_i : ℂ := Complex.mk 1 1

theorem complex_equation_solution (a b : ℝ) : 
  let z : ℂ := complex_one_plus_i
  (z^2 + a*z + b) / (z^2 - z + 1) = 1 - Complex.I → a = -1 ∧ b = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1941_194150


namespace NUMINAMATH_CALUDE_charity_pastries_count_l1941_194148

theorem charity_pastries_count (total_volunteers : ℕ) 
  (group_a_percent group_b_percent group_c_percent : ℚ)
  (group_a_batches group_b_batches group_c_batches : ℕ)
  (group_a_trays group_b_trays group_c_trays : ℕ)
  (group_a_pastries group_b_pastries group_c_pastries : ℕ) :
  total_volunteers = 1500 →
  group_a_percent = 2/5 →
  group_b_percent = 7/20 →
  group_c_percent = 1/4 →
  group_a_batches = 10 →
  group_b_batches = 15 →
  group_c_batches = 8 →
  group_a_trays = 6 →
  group_b_trays = 4 →
  group_c_trays = 5 →
  group_a_pastries = 20 →
  group_b_pastries = 12 →
  group_c_pastries = 15 →
  (↑total_volunteers * group_a_percent).floor * group_a_batches * group_a_trays * group_a_pastries +
  (↑total_volunteers * group_b_percent).floor * group_b_batches * group_b_trays * group_b_pastries +
  (↑total_volunteers * group_c_percent).floor * group_c_batches * group_c_trays * group_c_pastries = 1323000 := by
  sorry


end NUMINAMATH_CALUDE_charity_pastries_count_l1941_194148


namespace NUMINAMATH_CALUDE_rectangle_least_area_l1941_194151

theorem rectangle_least_area (l w : ℕ) (h1 : l > 0) (h2 : w > 0) (h3 : 2 * (l + w) = 100) : 
  l * w ≥ 49 := by
sorry

end NUMINAMATH_CALUDE_rectangle_least_area_l1941_194151


namespace NUMINAMATH_CALUDE_polynomial_expansion_and_sum_l1941_194155

theorem polynomial_expansion_and_sum (A B C D E : ℤ) : 
  (∀ x : ℝ, (x + 3) * (4 * x^3 - 2 * x^2 + 7 * x - 6) = A * x^4 + B * x^3 + C * x^2 + D * x + E) →
  A = 4 ∧ B = 10 ∧ C = 1 ∧ D = 15 ∧ E = -18 ∧ A + B + C + D + E = 12 := by
sorry

end NUMINAMATH_CALUDE_polynomial_expansion_and_sum_l1941_194155


namespace NUMINAMATH_CALUDE_largest_n_satisfying_conditions_l1941_194179

theorem largest_n_satisfying_conditions : ∃ (m k : ℤ),
  181^2 = (m + 1)^3 - m^3 ∧
  2 * 181 + 79 = k^2 ∧
  ∀ (n : ℤ), n > 181 → ¬(∃ (m' k' : ℤ), n^2 = (m' + 1)^3 - m'^3 ∧ 2 * n + 79 = k'^2) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_satisfying_conditions_l1941_194179


namespace NUMINAMATH_CALUDE_lillians_candies_l1941_194185

theorem lillians_candies (initial_candies final_candies : ℕ) 
  (h1 : initial_candies = 88)
  (h2 : final_candies = 93) :
  final_candies - initial_candies = 5 := by
  sorry

end NUMINAMATH_CALUDE_lillians_candies_l1941_194185


namespace NUMINAMATH_CALUDE_marys_story_characters_l1941_194110

theorem marys_story_characters (total : ℕ) (init_a init_c init_d init_e : ℕ) : 
  total = 60 →
  init_a = total / 2 →
  init_c = init_a / 2 →
  init_d + init_e = total - init_a - init_c →
  init_d = 2 * init_e →
  init_d = 10 := by
  sorry

end NUMINAMATH_CALUDE_marys_story_characters_l1941_194110


namespace NUMINAMATH_CALUDE_three_inscribed_circles_exist_l1941_194121

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- Three circles are inscribed in a larger circle --/
structure InscribedCircles where
  outer : Circle
  inner1 : Circle
  inner2 : Circle
  inner3 : Circle

/-- The property of three circles being equal --/
def equal_circles (c1 c2 c3 : Circle) : Prop :=
  c1.radius = c2.radius ∧ c2.radius = c3.radius

/-- The property of two circles being tangent --/
def tangent_circles (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2

/-- The property of a circle being inscribed in another circle --/
def inscribed_circle (outer inner : Circle) : Prop :=
  let (x1, y1) := outer.center
  let (x2, y2) := inner.center
  (x2 - x1)^2 + (y2 - y1)^2 = (outer.radius - inner.radius)^2

/-- Theorem: Three equal circles can be inscribed in a larger circle,
    such that they are tangent to each other and to the larger circle --/
theorem three_inscribed_circles_exist (outer : Circle) :
  ∃ (ic : InscribedCircles),
    ic.outer = outer ∧
    equal_circles ic.inner1 ic.inner2 ic.inner3 ∧
    tangent_circles ic.inner1 ic.inner2 ∧
    tangent_circles ic.inner2 ic.inner3 ∧
    tangent_circles ic.inner3 ic.inner1 ∧
    inscribed_circle outer ic.inner1 ∧
    inscribed_circle outer ic.inner2 ∧
    inscribed_circle outer ic.inner3 :=
  sorry

end NUMINAMATH_CALUDE_three_inscribed_circles_exist_l1941_194121


namespace NUMINAMATH_CALUDE_oil_spend_is_500_l1941_194111

/-- Represents the price reduction, amount difference, and reduced price of oil --/
structure OilPriceData where
  reduction_percent : ℚ
  amount_difference : ℚ
  reduced_price : ℚ

/-- Calculates the amount spent on oil given the price reduction data --/
def calculate_oil_spend (data : OilPriceData) : ℚ :=
  let original_price := data.reduced_price / (1 - data.reduction_percent)
  let m := data.amount_difference * (data.reduced_price * original_price) / (original_price - data.reduced_price)
  m

/-- Theorem stating that given the specific conditions, the amount spent on oil is 500 --/
theorem oil_spend_is_500 (data : OilPriceData) 
  (h1 : data.reduction_percent = 1/4)
  (h2 : data.amount_difference = 5)
  (h3 : data.reduced_price = 25) : 
  calculate_oil_spend data = 500 := by
  sorry

end NUMINAMATH_CALUDE_oil_spend_is_500_l1941_194111


namespace NUMINAMATH_CALUDE_equation_solutions_l1941_194119

theorem equation_solutions : 
  (∃ x₁ x₂ : ℝ, (x₁ + 5)^2 = 16 ∧ (x₂ + 5)^2 = 16 ∧ x₁ = -9 ∧ x₂ = -1) ∧
  (∃ y₁ y₂ : ℝ, y₁^2 - 4*y₁ - 12 = 0 ∧ y₂^2 - 4*y₂ - 12 = 0 ∧ y₁ = 6 ∧ y₂ = -2) :=
by
  sorry

#check equation_solutions

end NUMINAMATH_CALUDE_equation_solutions_l1941_194119


namespace NUMINAMATH_CALUDE_perfect_square_bc_l1941_194181

theorem perfect_square_bc (a b c : ℕ) 
  (h : (a^2 / (a^2 + b^2) : ℚ) + (c^2 / (a^2 + c^2) : ℚ) = 2 * c / (b + c)) : 
  ∃ k : ℕ, b * c = k^2 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_bc_l1941_194181


namespace NUMINAMATH_CALUDE_matrix_is_own_inverse_l1941_194164

def A (a b : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  !![4, -2; a, b]

theorem matrix_is_own_inverse (a b : ℚ) :
  A a b * A a b = 1 ↔ a = 15/2 ∧ b = -4 := by
  sorry

end NUMINAMATH_CALUDE_matrix_is_own_inverse_l1941_194164


namespace NUMINAMATH_CALUDE_typing_speed_ratio_l1941_194130

-- Define Tim's typing speed
def tim_speed : ℝ := 2

-- Define Tom's normal typing speed
def tom_speed : ℝ := 10

-- Define Tom's increased typing speed (30% increase)
def tom_increased_speed : ℝ := tom_speed * 1.3

-- Theorem to prove
theorem typing_speed_ratio :
  -- Condition 1: Tim and Tom can type 12 pages in one hour together
  tim_speed + tom_speed = 12 →
  -- Condition 2: With Tom's increased speed, they can type 15 pages in one hour
  tim_speed + tom_increased_speed = 15 →
  -- Conclusion: The ratio of Tom's normal speed to Tim's is 5:1
  tom_speed / tim_speed = 5 := by
  sorry

end NUMINAMATH_CALUDE_typing_speed_ratio_l1941_194130


namespace NUMINAMATH_CALUDE_perpendicular_lines_from_perpendicular_planes_l1941_194134

structure Plane
structure Line

-- Define the perpendicular relationship between planes
def perp_planes (α β : Plane) : Prop := sorry

-- Define the intersection of two planes
def intersect_planes (α β : Plane) : Line := sorry

-- Define a line parallel to a plane
def parallel_line_plane (a : Line) (α : Plane) : Prop := sorry

-- Define a line perpendicular to a plane
def perp_line_plane (b : Line) (β : Plane) : Prop := sorry

-- Define a line perpendicular to another line
def perp_lines (b l : Line) : Prop := sorry

theorem perpendicular_lines_from_perpendicular_planes 
  (α β : Plane) (a b l : Line) 
  (h1 : perp_planes α β) 
  (h2 : intersect_planes α β = l) 
  (h3 : parallel_line_plane a α) 
  (h4 : perp_line_plane b β) : 
  perp_lines b l := sorry

end NUMINAMATH_CALUDE_perpendicular_lines_from_perpendicular_planes_l1941_194134


namespace NUMINAMATH_CALUDE_total_practice_time_is_135_l1941_194114

/-- The number of minutes Daniel practices on a school day -/
def school_day_practice : ℕ := 15

/-- The number of school days in a week -/
def school_days : ℕ := 5

/-- The number of weekend days -/
def weekend_days : ℕ := 2

/-- The number of minutes Daniel practices on a weekend day -/
def weekend_day_practice : ℕ := 2 * school_day_practice

/-- The total practice time for a whole week in minutes -/
def total_practice_time : ℕ := school_day_practice * school_days + weekend_day_practice * weekend_days

theorem total_practice_time_is_135 : total_practice_time = 135 := by
  sorry

end NUMINAMATH_CALUDE_total_practice_time_is_135_l1941_194114


namespace NUMINAMATH_CALUDE_log_equation_solution_l1941_194169

theorem log_equation_solution (x : ℝ) (h : Real.log 729 / Real.log (3 * x) = x) : x = 3 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l1941_194169


namespace NUMINAMATH_CALUDE_brayden_gavin_touchdowns_l1941_194132

theorem brayden_gavin_touchdowns :
  let touchdown_points : ℕ := 7
  let cole_freddy_touchdowns : ℕ := 9
  let point_difference : ℕ := 14
  let brayden_gavin_touchdowns : ℕ := 7

  touchdown_points * cole_freddy_touchdowns = 
  touchdown_points * brayden_gavin_touchdowns + point_difference :=
by
  sorry

end NUMINAMATH_CALUDE_brayden_gavin_touchdowns_l1941_194132


namespace NUMINAMATH_CALUDE_division_problem_l1941_194199

theorem division_problem (dividend quotient remainder divisor : ℕ) : 
  dividend = 122 → quotient = 6 → remainder = 2 → 
  dividend = divisor * quotient + remainder →
  divisor = 20 := by sorry

end NUMINAMATH_CALUDE_division_problem_l1941_194199


namespace NUMINAMATH_CALUDE_camping_trip_percentage_l1941_194194

theorem camping_trip_percentage
  (total_students : ℕ)
  (h1 : (14 : ℚ) / 100 * total_students = (25 : ℚ) / 100 * (56 : ℚ) / 100 * total_students)
  (h2 : (75 : ℚ) / 100 * (56 : ℚ) / 100 * total_students + (14 : ℚ) / 100 * total_students = (56 : ℚ) / 100 * total_students) :
  (56 : ℚ) / 100 * total_students = (56 : ℚ) / 100 * total_students :=
by sorry

end NUMINAMATH_CALUDE_camping_trip_percentage_l1941_194194


namespace NUMINAMATH_CALUDE_equation_solution_l1941_194190

theorem equation_solution :
  ∀ x : ℝ, (2*x - 3)^2 = (x - 2)^2 ↔ x = 1 ∨ x = 5/3 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l1941_194190


namespace NUMINAMATH_CALUDE_symmetric_circles_line_l1941_194104

-- Define the circles
def C₁ (x y a : ℝ) : Prop := x^2 + y^2 - a = 0
def C₂ (x y a : ℝ) : Prop := x^2 + y^2 + 2*x - 2*a*y + 3 = 0

-- Define the line
def line_l (x y : ℝ) : Prop := 2*x - 4*y + 5 = 0

-- State the theorem
theorem symmetric_circles_line (a : ℝ) :
  (∀ x y, C₁ x y a ↔ C₂ (2*x + 1) (2*y - a) a) →
  (∀ x y, line_l x y ↔ (∃ x₀ y₀, C₁ x₀ y₀ a ∧ C₂ (2*x - x₀) (2*y - y₀) a)) :=
sorry

end NUMINAMATH_CALUDE_symmetric_circles_line_l1941_194104


namespace NUMINAMATH_CALUDE_triangle_area_with_squares_l1941_194159

/-- Given a scalene triangle with adjoining squares, prove its area -/
theorem triangle_area_with_squares (a b c h : ℝ) : 
  a > 0 → b > 0 → c > 0 → h > 0 →
  a^2 = 100 → b^2 = 64 → c^2 = 49 → h^2 = 81 →
  (1/2 : ℝ) * a * h = 45 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_with_squares_l1941_194159


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_ratio_l1941_194143

/-- For a hyperbola with equation x^2/a^2 - y^2/b^2 = 1, where a > b and the angle between
    the asymptotes is 30°, the ratio a/b = 2 - √3. -/
theorem hyperbola_asymptote_ratio (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (Real.pi / 6 = Real.arctan ((2 * b / a) / (1 - (b / a)^2))) →
  a / b = 2 - Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_ratio_l1941_194143


namespace NUMINAMATH_CALUDE_manuscript_revision_cost_l1941_194108

/-- The cost per page for revision in a manuscript typing service --/
def revision_cost (total_pages : ℕ) (revised_once : ℕ) (revised_twice : ℕ) 
  (initial_cost : ℕ) (total_cost : ℕ) : ℚ :=
  let pages_not_revised := total_pages - revised_once - revised_twice
  let initial_typing_cost := total_pages * initial_cost
  let revision_pages := revised_once + 2 * revised_twice
  (total_cost - initial_typing_cost : ℚ) / revision_pages

/-- Theorem stating the revision cost for the given manuscript --/
theorem manuscript_revision_cost :
  revision_cost 200 80 20 5 1360 = 3 := by
  sorry

end NUMINAMATH_CALUDE_manuscript_revision_cost_l1941_194108


namespace NUMINAMATH_CALUDE_max_value_cyclic_expression_l1941_194170

theorem max_value_cyclic_expression (x y z : ℝ) 
  (h_nonneg_x : x ≥ 0) (h_nonneg_y : y ≥ 0) (h_nonneg_z : z ≥ 0)
  (h_sum : x + y + z = 3) :
  (x^2 - x*y + y^2) * (y^2 - y*z + z^2) * (z^2 - z*x + x^2) ≤ 27/8 ∧ 
  ∃ x y z : ℝ, x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x + y + z = 3 ∧
    (x^2 - x*y + y^2) * (y^2 - y*z + z^2) * (z^2 - z*x + x^2) = 27/8 :=
by sorry

end NUMINAMATH_CALUDE_max_value_cyclic_expression_l1941_194170


namespace NUMINAMATH_CALUDE_cannot_form_square_l1941_194120

/-- Represents the collection of sticks --/
structure StickCollection where
  twoLengthCount : Nat
  threeLengthCount : Nat
  sevenLengthCount : Nat

/-- Checks if it's possible to form a square with given sticks --/
def canFormSquare (sticks : StickCollection) : Prop :=
  ∃ (side : ℕ), 
    4 * side = 2 * sticks.twoLengthCount + 
               3 * sticks.threeLengthCount + 
               7 * sticks.sevenLengthCount ∧
    ∃ (a b c : ℕ), 
      a + b + c = 4 ∧
      a * 2 + b * 3 + c * 7 = 4 * side ∧
      a ≤ sticks.twoLengthCount ∧
      b ≤ sticks.threeLengthCount ∧
      c ≤ sticks.sevenLengthCount

/-- The given collection of sticks --/
def givenSticks : StickCollection :=
  { twoLengthCount := 5
    threeLengthCount := 5
    sevenLengthCount := 1 }

theorem cannot_form_square : ¬(canFormSquare givenSticks) := by
  sorry

end NUMINAMATH_CALUDE_cannot_form_square_l1941_194120


namespace NUMINAMATH_CALUDE_train_departure_time_l1941_194136

/-- Proves that the first train left Mumbai 2 hours before the meeting point -/
theorem train_departure_time 
  (first_train_speed : ℝ) 
  (second_train_speed : ℝ) 
  (time_difference : ℝ) 
  (meeting_distance : ℝ) 
  (h1 : first_train_speed = 45)
  (h2 : second_train_speed = 90)
  (h3 : time_difference = 1)
  (h4 : meeting_distance = 90) :
  ∃ (departure_time : ℝ), 
    departure_time = 2 ∧ 
    first_train_speed * (departure_time + time_difference) = 
    second_train_speed * time_difference ∧
    first_train_speed * departure_time = meeting_distance :=
by sorry


end NUMINAMATH_CALUDE_train_departure_time_l1941_194136


namespace NUMINAMATH_CALUDE_reverse_divisibility_implies_divides_99_l1941_194145

/-- Given a natural number, return the number formed by reversing its digits -/
def reverse_digits (n : ℕ) : ℕ := sorry

/-- A natural number k has the property that if it divides n, it also divides the reverse of n -/
def has_reverse_divisibility_property (k : ℕ) : Prop :=
  ∀ n : ℕ, k ∣ n → k ∣ reverse_digits n

theorem reverse_divisibility_implies_divides_99 (k : ℕ) :
  has_reverse_divisibility_property k → 99 ∣ k := by sorry

end NUMINAMATH_CALUDE_reverse_divisibility_implies_divides_99_l1941_194145


namespace NUMINAMATH_CALUDE_download_speed_scientific_notation_l1941_194157

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The theoretical download speed of the Huawei phone MateX on a 5G network in B/s -/
def download_speed : ℕ := 603000000

/-- Converts a natural number to its scientific notation representation -/
def to_scientific_notation (n : ℕ) : ScientificNotation :=
  sorry

theorem download_speed_scientific_notation :
  to_scientific_notation download_speed = ScientificNotation.mk 6.03 8 (by sorry) :=
sorry

end NUMINAMATH_CALUDE_download_speed_scientific_notation_l1941_194157


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1941_194102

def A : Set ℝ := {x : ℝ | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x : ℝ | 2 < x ∧ x < 10}

theorem union_of_A_and_B : A ∪ B = {x : ℝ | 2 < x ∧ x < 10} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1941_194102


namespace NUMINAMATH_CALUDE_x_minus_y_equals_negative_twelve_l1941_194153

theorem x_minus_y_equals_negative_twelve (x y : ℝ) 
  (hx : 2 = 0.25 * x) (hy : 2 = 0.10 * y) : x - y = -12 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_y_equals_negative_twelve_l1941_194153


namespace NUMINAMATH_CALUDE_lcm_and_sum_first_ten_l1941_194123

/-- The set of the first ten positive integers -/
def firstTenIntegers : Finset ℕ := Finset.range 10

/-- The least common multiple of the first ten positive integers -/
def lcmFirstTen : ℕ := Finset.lcm firstTenIntegers id

/-- The sum of the first ten positive integers -/
def sumFirstTen : ℕ := Finset.sum firstTenIntegers id

theorem lcm_and_sum_first_ten :
  lcmFirstTen = 2520 ∧ sumFirstTen = 55 := by sorry

end NUMINAMATH_CALUDE_lcm_and_sum_first_ten_l1941_194123


namespace NUMINAMATH_CALUDE_octagon_triangle_side_ratio_l1941_194135

theorem octagon_triangle_side_ratio : 
  ∀ (s_o s_t : ℝ), s_o > 0 → s_t > 0 →
  (2 * Real.sqrt 2) * s_o^2 = (Real.sqrt 3 / 4) * s_t^2 →
  s_t / s_o = 2 * (2 : ℝ)^(1/4) :=
by
  sorry

end NUMINAMATH_CALUDE_octagon_triangle_side_ratio_l1941_194135


namespace NUMINAMATH_CALUDE_min_distance_point_is_circumcenter_l1941_194163

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle in 2D space -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Checks if a triangle is acute -/
def isAcute (t : Triangle) : Prop :=
  sorry

/-- Calculates the squared distance between two points -/
def squaredDistance (p1 p2 : Point) : ℝ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

/-- Finds the foot of the perpendicular from a point to a line segment -/
def perpendicularFoot (p : Point) (a b : Point) : Point :=
  sorry

/-- Calculates the circumcenter of a triangle -/
def circumcenter (t : Triangle) : Point :=
  sorry

/-- Main theorem: The point that minimizes the sum of squared distances to the sides
    of an acute triangle is its circumcenter -/
theorem min_distance_point_is_circumcenter (t : Triangle) (h : isAcute t) :
  ∀ P : Point,
    let L := perpendicularFoot P t.B t.C
    let M := perpendicularFoot P t.C t.A
    let N := perpendicularFoot P t.A t.B
    squaredDistance P L + squaredDistance P M + squaredDistance P N ≥
    let C := circumcenter t
    let CL := perpendicularFoot C t.B t.C
    let CM := perpendicularFoot C t.C t.A
    let CN := perpendicularFoot C t.A t.B
    squaredDistance C CL + squaredDistance C CM + squaredDistance C CN :=
by
  sorry

end NUMINAMATH_CALUDE_min_distance_point_is_circumcenter_l1941_194163
