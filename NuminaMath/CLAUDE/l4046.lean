import Mathlib

namespace NUMINAMATH_CALUDE_fraction_sum_equality_l4046_404642

theorem fraction_sum_equality : (3 / 10 : ℚ) + (2 / 100 : ℚ) + (8 / 1000 : ℚ) + (8 / 10000 : ℚ) = 0.3288 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l4046_404642


namespace NUMINAMATH_CALUDE_infinite_solutions_iff_d_equals_five_l4046_404653

theorem infinite_solutions_iff_d_equals_five :
  ∀ d : ℝ, (∀ y : ℝ, 3 * (5 + d * y) = 15 * y + 15) ↔ d = 5 := by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_iff_d_equals_five_l4046_404653


namespace NUMINAMATH_CALUDE_laundry_time_difference_l4046_404654

theorem laundry_time_difference : ∀ (clothes_time towels_time sheets_time : ℕ),
  clothes_time = 30 →
  towels_time = 2 * clothes_time →
  clothes_time + towels_time + sheets_time = 135 →
  towels_time - sheets_time = 15 := by
sorry

end NUMINAMATH_CALUDE_laundry_time_difference_l4046_404654


namespace NUMINAMATH_CALUDE_acute_triangle_on_perpendicular_lines_l4046_404630

-- Define an acute-angled triangle
structure AcuteTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  acute_a : a^2 < b^2 + c^2
  acute_b : b^2 < a^2 + c^2
  acute_c : c^2 < a^2 + b^2

-- Theorem statement
theorem acute_triangle_on_perpendicular_lines (t : AcuteTriangle) :
  ∃ (x y z : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧
  x^2 + y^2 = t.c^2 ∧
  x^2 + z^2 = t.b^2 ∧
  y^2 + z^2 = t.a^2 :=
sorry

end NUMINAMATH_CALUDE_acute_triangle_on_perpendicular_lines_l4046_404630


namespace NUMINAMATH_CALUDE_midpoint_product_and_distance_l4046_404656

/-- Given that C is the midpoint of segment AB, prove that xy = -12 and d = 4√5 -/
theorem midpoint_product_and_distance (x y : ℝ) :
  (4 : ℝ) = (2 + x) / 2 →
  (2 : ℝ) = (6 + y) / 2 →
  x * y = -12 ∧ Real.sqrt ((x - 2)^2 + (y - 6)^2) = 4 * Real.sqrt 5 := by
  sorry

#check midpoint_product_and_distance

end NUMINAMATH_CALUDE_midpoint_product_and_distance_l4046_404656


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_l4046_404687

theorem quadratic_equation_solutions :
  ∀ x : ℝ, x * (x - 1) = 1 - x ↔ x = 1 ∨ x = -1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_l4046_404687


namespace NUMINAMATH_CALUDE_population_increase_rate_is_10_percent_l4046_404621

/-- The population increase rate given initial and final populations -/
def population_increase_rate (initial : ℕ) (final : ℕ) : ℚ :=
  (final - initial : ℚ) / initial * 100

/-- Theorem: The population increase rate is 10% given the conditions -/
theorem population_increase_rate_is_10_percent :
  let initial_population := 260
  let final_population := 286
  population_increase_rate initial_population final_population = 10 := by
  sorry

end NUMINAMATH_CALUDE_population_increase_rate_is_10_percent_l4046_404621


namespace NUMINAMATH_CALUDE_jake_sausage_cost_l4046_404610

-- Define the parameters
def package_weight : ℝ := 2
def num_packages : ℕ := 3
def price_per_pound : ℝ := 4

-- Define the theorem
theorem jake_sausage_cost :
  package_weight * num_packages * price_per_pound = 24 := by
  sorry

end NUMINAMATH_CALUDE_jake_sausage_cost_l4046_404610


namespace NUMINAMATH_CALUDE_racks_fit_on_shelf_l4046_404647

/-- Represents the number of CDs a single rack can hold -/
def cds_per_rack : ℕ := 8

/-- Represents the total number of CDs the shelf can hold -/
def total_cds : ℕ := 32

/-- Calculates the number of racks that can fit on the shelf -/
def racks_on_shelf : ℕ := total_cds / cds_per_rack

/-- Proves that the number of racks that can fit on the shelf is 4 -/
theorem racks_fit_on_shelf : racks_on_shelf = 4 := by
  sorry

end NUMINAMATH_CALUDE_racks_fit_on_shelf_l4046_404647


namespace NUMINAMATH_CALUDE_delegate_seating_probability_l4046_404674

-- Define the number of delegates and countries
def total_delegates : ℕ := 12
def num_countries : ℕ := 3
def delegates_per_country : ℕ := 4

-- Define the probability as a fraction
def probability : ℚ := 106 / 115

-- State the theorem
theorem delegate_seating_probability :
  let total_arrangements := (total_delegates.factorial) / ((delegates_per_country.factorial) ^ num_countries)
  let unwanted_arrangements := 
    (num_countries * total_delegates * ((total_delegates - delegates_per_country).factorial / 
    (delegates_per_country.factorial ^ (num_countries - 1)))) -
    (num_countries * total_delegates * (delegates_per_country + 2)) +
    (total_delegates * 2)
  (total_arrangements - unwanted_arrangements) / total_arrangements = probability := by
  sorry

end NUMINAMATH_CALUDE_delegate_seating_probability_l4046_404674


namespace NUMINAMATH_CALUDE_panda_babies_l4046_404607

theorem panda_babies (total_pandas : ℕ) (pregnancy_rate : ℚ) : 
  total_pandas = 16 → 
  pregnancy_rate = 1/4 → 
  (total_pandas / 2 : ℚ) * pregnancy_rate = 2 :=
sorry

end NUMINAMATH_CALUDE_panda_babies_l4046_404607


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l4046_404648

/-- Given a geometric sequence with first term a₁ and common ratio q,
    if the sum of the first three terms equals 3a₁, then q = 1 or q = -2 -/
theorem geometric_sequence_ratio (a₁ : ℝ) (q : ℝ) :
  a₁ ≠ 0 →
  a₁ + a₁ * q + a₁ * q^2 = 3 * a₁ →
  q = 1 ∨ q = -2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l4046_404648


namespace NUMINAMATH_CALUDE_cartesian_product_A_B_l4046_404640

def A : Set ℕ := {1, 2}
def B : Set ℕ := {1, 2, 3}

theorem cartesian_product_A_B :
  A ×ˢ B = {(1,1), (1,2), (1,3), (2,1), (2,2), (2,3)} := by
  sorry

end NUMINAMATH_CALUDE_cartesian_product_A_B_l4046_404640


namespace NUMINAMATH_CALUDE_john_can_lift_2800_pounds_l4046_404655

-- Define the given values
def original_squat : ℝ := 135
def training_increase : ℝ := 265
def bracer_multiplier : ℝ := 7  -- 600% increase means multiplying by 7 (1 + 6)

-- Define the calculation steps
def new_squat : ℝ := original_squat + training_increase
def final_lift : ℝ := new_squat * bracer_multiplier

-- Theorem statement
theorem john_can_lift_2800_pounds : 
  final_lift = 2800 := by sorry

end NUMINAMATH_CALUDE_john_can_lift_2800_pounds_l4046_404655


namespace NUMINAMATH_CALUDE_work_completion_l4046_404608

/-- Represents the number of days it takes to complete the work -/
structure WorkDays where
  together : ℝ
  a_alone : ℝ
  initial_together : ℝ
  a_remaining : ℝ

/-- Given work completion rates, proves that 'a' worked alone for 9 days after 'b' left -/
theorem work_completion (w : WorkDays) 
  (h1 : w.together = 40)
  (h2 : w.a_alone = 12)
  (h3 : w.initial_together = 10) : 
  w.a_remaining = 9 := by
sorry

end NUMINAMATH_CALUDE_work_completion_l4046_404608


namespace NUMINAMATH_CALUDE_quartic_two_real_roots_l4046_404695

theorem quartic_two_real_roots 
  (a b c d e : ℝ) 
  (ha : a ≠ 0) 
  (h_root : ∃ β : ℝ, β > 1 ∧ a * β^2 + (c - b) * β + (e - d) = 0) :
  ∃ x y : ℝ, x ≠ y ∧ a * x^4 + b * x^3 + c * x^2 + d * x + e = 0 ∧ 
                    a * y^4 + b * y^3 + c * y^2 + d * y + e = 0 :=
by sorry

end NUMINAMATH_CALUDE_quartic_two_real_roots_l4046_404695


namespace NUMINAMATH_CALUDE_rectangle_area_l4046_404650

theorem rectangle_area (L B : ℝ) (h1 : L - B = 23) (h2 : 2 * L + 2 * B = 186) : L * B = 2030 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l4046_404650


namespace NUMINAMATH_CALUDE_greatest_x_quadratic_inequality_l4046_404679

theorem greatest_x_quadratic_inequality :
  ∀ x : ℝ, x^2 - 16*x + 63 ≤ 0 → x ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_quadratic_inequality_l4046_404679


namespace NUMINAMATH_CALUDE_cubic_function_extrema_l4046_404611

def f (m : ℝ) (x : ℝ) : ℝ := x^3 + m*x^2 + (m + 6)*x + 1

theorem cubic_function_extrema (m : ℝ) :
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    IsLocalMax (f m) x₁ ∧ 
    IsLocalMin (f m) x₂) ↔ 
  (m < -3 ∨ m > 6) :=
sorry

end NUMINAMATH_CALUDE_cubic_function_extrema_l4046_404611


namespace NUMINAMATH_CALUDE_expression_value_l4046_404644

theorem expression_value (x y z : ℚ) 
  (eq1 : 3 * x - 2 * y - 4 * z = 0)
  (eq2 : 2 * x + y - 9 * z = 0)
  (z_neq_zero : z ≠ 0) :
  (x^2 - 2*x*y) / (y^2 + 2*z^2) = -24/25 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l4046_404644


namespace NUMINAMATH_CALUDE_cubic_polynomial_root_l4046_404645

theorem cubic_polynomial_root (x : ℝ) (h : x = Real.rpow 4 (1/3) + 1) : 
  x^3 - 3*x^2 + 3*x - 5 = 0 := by
sorry

end NUMINAMATH_CALUDE_cubic_polynomial_root_l4046_404645


namespace NUMINAMATH_CALUDE_first_place_points_l4046_404673

/-- Represents a tournament with the given conditions -/
structure Tournament :=
  (num_teams : Nat)
  (games_per_pair : Nat)
  (total_points : Nat)
  (last_place_points : Nat)

/-- Calculates the number of games played in the tournament -/
def num_games (t : Tournament) : Nat :=
  (t.num_teams.choose 2) * t.games_per_pair

/-- Theorem stating the first-place team's points in the given tournament conditions -/
theorem first_place_points (t : Tournament)
  (h1 : t.num_teams = 4)
  (h2 : t.games_per_pair = 2)
  (h3 : t.total_points = num_games t * 2)
  (h4 : t.last_place_points = 5) :
  ∃ (first_place_points : Nat),
    first_place_points = 7 ∧
    first_place_points + t.last_place_points ≤ t.total_points :=
by
  sorry


end NUMINAMATH_CALUDE_first_place_points_l4046_404673


namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_l4046_404606

theorem quadratic_real_roots_condition (k : ℝ) :
  (∃ x : ℝ, k * x^2 - 4 * x + 2 = 0) ↔ (k ≤ 2 ∧ k ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_l4046_404606


namespace NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l4046_404605

/-- An arithmetic sequence with specific properties -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_third_term
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_first : a 1 = 2)
  (h_fifth : a 5 = a 4 + 2) :
  a 3 = 6 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l4046_404605


namespace NUMINAMATH_CALUDE_franks_problems_per_type_l4046_404635

/-- The number of math problems composed by Bill. -/
def bills_problems : ℕ := 20

/-- The number of math problems composed by Ryan. -/
def ryans_problems : ℕ := 2 * bills_problems

/-- The number of math problems composed by Frank. -/
def franks_problems : ℕ := 3 * ryans_problems

/-- The number of different types of math problems each person composes. -/
def problem_types : ℕ := 4

/-- Theorem stating that Frank composes 30 problems of each type. -/
theorem franks_problems_per_type :
  franks_problems / problem_types = 30 := by sorry

end NUMINAMATH_CALUDE_franks_problems_per_type_l4046_404635


namespace NUMINAMATH_CALUDE_giant_spider_leg_pressure_l4046_404699

/-- Calculates the pressure on each leg of a giant spider -/
theorem giant_spider_leg_pressure (previous_weight : ℝ) (weight_multiplier : ℝ) (leg_area : ℝ) (num_legs : ℕ) : 
  previous_weight = 6.4 →
  weight_multiplier = 2.5 →
  leg_area = 0.5 →
  num_legs = 8 →
  (previous_weight * weight_multiplier) / (num_legs * leg_area) = 4 := by
sorry

end NUMINAMATH_CALUDE_giant_spider_leg_pressure_l4046_404699


namespace NUMINAMATH_CALUDE_imaginary_unit_power_2015_l4046_404615

theorem imaginary_unit_power_2015 (i : ℂ) (h : i^2 = -1) : i^2015 = -i := by
  sorry

end NUMINAMATH_CALUDE_imaginary_unit_power_2015_l4046_404615


namespace NUMINAMATH_CALUDE_total_colors_over_two_hours_l4046_404625

/-- Represents the number of colors the sky changes through in a 10-minute period -/
structure ColorChange where
  quick : ℕ
  slow : ℕ

/-- Calculates the total number of colors for one hour given a ColorChange pattern -/
def colorsPerHour (change : ColorChange) : ℕ :=
  (change.quick + change.slow) * 3

/-- The color change pattern for the first hour -/
def firstHourPattern : ColorChange :=
  { quick := 5, slow := 2 }

/-- The color change pattern for the second hour (doubled rate) -/
def secondHourPattern : ColorChange :=
  { quick := firstHourPattern.quick * 2, slow := firstHourPattern.slow * 2 }

/-- The main theorem stating the total number of colors over two hours -/
theorem total_colors_over_two_hours :
  colorsPerHour firstHourPattern + colorsPerHour secondHourPattern = 63 := by
  sorry


end NUMINAMATH_CALUDE_total_colors_over_two_hours_l4046_404625


namespace NUMINAMATH_CALUDE_simple_interest_problem_l4046_404609

/-- Given a sum P put at simple interest for 3 years, if increasing the interest rate
    by 1% results in an additional Rs. 75 interest, then P = 2500. -/
theorem simple_interest_problem (P : ℝ) (R : ℝ) : 
  P * (R + 1) * 3 / 100 - P * R * 3 / 100 = 75 → P = 2500 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l4046_404609


namespace NUMINAMATH_CALUDE_complex_product_sum_l4046_404670

theorem complex_product_sum (a b : ℝ) (i : ℂ) (h1 : i^2 = -1) 
  (h2 : (1 + i) * (2 + i) = a + b * i) : a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_sum_l4046_404670


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l4046_404649

theorem min_value_sum_reciprocals (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 2*x + y - 3 = 0) : 
  2/x + 1/y ≥ 3 ∧ (2/x + 1/y = 3 ↔ x = 1 ∧ y = 1) :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l4046_404649


namespace NUMINAMATH_CALUDE_leftover_coins_value_l4046_404652

/-- The number of nickels in a complete roll -/
def nickels_per_roll : ℕ := 40

/-- The number of pennies in a complete roll -/
def pennies_per_roll : ℕ := 50

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The value of a penny in cents -/
def penny_value : ℕ := 1

/-- Michael's nickels -/
def michael_nickels : ℕ := 183

/-- Michael's pennies -/
def michael_pennies : ℕ := 259

/-- Sarah's nickels -/
def sarah_nickels : ℕ := 167

/-- Sarah's pennies -/
def sarah_pennies : ℕ := 342

/-- The value of leftover coins in cents -/
def leftover_value : ℕ :=
  ((michael_nickels + sarah_nickels) % nickels_per_roll) * nickel_value +
  ((michael_pennies + sarah_pennies) % pennies_per_roll) * penny_value

theorem leftover_coins_value : leftover_value = 151 := by
  sorry

end NUMINAMATH_CALUDE_leftover_coins_value_l4046_404652


namespace NUMINAMATH_CALUDE_pascal_ratio_row_l4046_404616

/-- Pascal's Triangle entry -/
def pascal (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

/-- Checks if three consecutive entries in a row have the ratio 2:3:4 -/
def has_ratio_2_3_4 (n : ℕ) : Prop :=
  ∃ r : ℕ, 
    (pascal n r : ℚ) / (pascal n (r + 1)) = 2 / 3 ∧
    (pascal n (r + 1) : ℚ) / (pascal n (r + 2)) = 3 / 4

theorem pascal_ratio_row : 
  ∃ n : ℕ, has_ratio_2_3_4 n ∧ ∀ m : ℕ, m < n → ¬has_ratio_2_3_4 m :=
by sorry

end NUMINAMATH_CALUDE_pascal_ratio_row_l4046_404616


namespace NUMINAMATH_CALUDE_leonardo_nap_duration_l4046_404667

def minutes_per_hour : ℕ := 60

def fifth_of_hour (total_minutes : ℕ) : ℕ := total_minutes / 5

theorem leonardo_nap_duration : fifth_of_hour minutes_per_hour = 12 := by
  sorry

end NUMINAMATH_CALUDE_leonardo_nap_duration_l4046_404667


namespace NUMINAMATH_CALUDE_car_overtakes_buses_l4046_404671

/-- The time interval between bus departures in minutes -/
def bus_interval : ℕ := 3

/-- The time taken by a bus to reach the city centre in minutes -/
def bus_travel_time : ℕ := 60

/-- The time taken by the car to reach the city centre in minutes -/
def car_travel_time : ℕ := 35

/-- The number of buses overtaken by the car -/
def buses_overtaken : ℕ := (bus_travel_time - car_travel_time) / bus_interval

theorem car_overtakes_buses :
  buses_overtaken = 8 := by
  sorry

end NUMINAMATH_CALUDE_car_overtakes_buses_l4046_404671


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l4046_404664

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ) (a_val b_val : ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_a5 : a 5 = a_val)
  (h_a10 : a 10 = b_val) :
  a 15 = 2 * b_val - a_val :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l4046_404664


namespace NUMINAMATH_CALUDE_local_minimum_of_f_l4046_404637

-- Define the function f(x) = x³ - 12x
def f (x : ℝ) : ℝ := x^3 - 12*x

-- State the theorem
theorem local_minimum_of_f :
  ∃ (x₀ : ℝ), IsLocalMin f x₀ ∧ x₀ = 2 := by sorry

end NUMINAMATH_CALUDE_local_minimum_of_f_l4046_404637


namespace NUMINAMATH_CALUDE_continuity_reciprocal_quadratic_plus_four_l4046_404617

theorem continuity_reciprocal_quadratic_plus_four (x : ℝ) :
  Continuous (fun x => 1 / (x^2 + 4)) :=
sorry

end NUMINAMATH_CALUDE_continuity_reciprocal_quadratic_plus_four_l4046_404617


namespace NUMINAMATH_CALUDE_mrs_hilt_reading_l4046_404676

/-- The number of books Mrs. Hilt read -/
def num_books : ℕ := 4

/-- The number of chapters in each book -/
def chapters_per_book : ℕ := 17

/-- The total number of chapters Mrs. Hilt read -/
def total_chapters : ℕ := num_books * chapters_per_book

theorem mrs_hilt_reading :
  total_chapters = 68 := by sorry

end NUMINAMATH_CALUDE_mrs_hilt_reading_l4046_404676


namespace NUMINAMATH_CALUDE_midpoint_sum_and_distance_l4046_404692

/-- Given a line segment with endpoints (4, 6) and (10, 18), prove that the sum of the
    coordinates of its midpoint plus the distance from this midpoint to the point (2, 1)
    equals 19 + √146. -/
theorem midpoint_sum_and_distance :
  let x1 : ℝ := 4
  let y1 : ℝ := 6
  let x2 : ℝ := 10
  let y2 : ℝ := 18
  let midx : ℝ := (x1 + x2) / 2
  let midy : ℝ := (y1 + y2) / 2
  let sum_coords : ℝ := midx + midy
  let dist : ℝ := Real.sqrt ((midx - 2)^2 + (midy - 1)^2)
  sum_coords + dist = 19 + Real.sqrt 146 := by
sorry

end NUMINAMATH_CALUDE_midpoint_sum_and_distance_l4046_404692


namespace NUMINAMATH_CALUDE_forty_fifth_term_is_91_l4046_404697

/-- An arithmetic sequence is defined by its first term and common difference. -/
structure ArithmeticSequence where
  first_term : ℝ
  common_difference : ℝ

/-- The nth term of an arithmetic sequence. -/
def nth_term (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  seq.first_term + (n - 1) * seq.common_difference

/-- Theorem: In an arithmetic sequence where the first term is 3 and the fifteenth term is 31,
    the forty-fifth term is 91. -/
theorem forty_fifth_term_is_91 :
  ∀ seq : ArithmeticSequence,
  seq.first_term = 3 →
  nth_term seq 15 = 31 →
  nth_term seq 45 = 91 := by
sorry

end NUMINAMATH_CALUDE_forty_fifth_term_is_91_l4046_404697


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l4046_404604

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 9/y = 1) :
  x + y ≥ 16 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ 1/x + 9/y = 1 ∧ x + y = 16 := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l4046_404604


namespace NUMINAMATH_CALUDE_town_hall_eggs_l4046_404624

/-- The number of eggs Joe found in different locations --/
structure EggCount where
  clubHouse : ℕ
  park : ℕ
  townHall : ℕ
  total : ℕ

/-- Theorem stating that Joe found 3 eggs in the town hall garden --/
theorem town_hall_eggs (e : EggCount) 
  (h1 : e.clubHouse = 12)
  (h2 : e.park = 5)
  (h3 : e.total = 20)
  (h4 : e.total = e.clubHouse + e.park + e.townHall) :
  e.townHall = 3 := by
  sorry

#check town_hall_eggs

end NUMINAMATH_CALUDE_town_hall_eggs_l4046_404624


namespace NUMINAMATH_CALUDE_pages_copied_for_thirty_dollars_l4046_404677

/-- Given the rate of copying 5 pages for 8 cents, prove that $30 (3000 cents) will allow copying 1875 pages. -/
theorem pages_copied_for_thirty_dollars :
  let rate : ℚ := 5 / 8 -- pages per cent
  let total_cents : ℕ := 3000 -- $30 in cents
  (rate * total_cents : ℚ) = 1875 := by sorry

end NUMINAMATH_CALUDE_pages_copied_for_thirty_dollars_l4046_404677


namespace NUMINAMATH_CALUDE_sheilas_monthly_savings_l4046_404614

/-- Calculates the monthly savings amount given the initial savings, family contribution, 
    savings period in years, and final amount in the piggy bank. -/
def monthlySavings (initialSavings familyContribution : ℕ) (savingsPeriodYears : ℕ) 
    (finalAmount : ℕ) : ℚ :=
  let totalInitialAmount := initialSavings + familyContribution
  let amountToSave := finalAmount - totalInitialAmount
  let monthsInPeriod := savingsPeriodYears * 12
  (amountToSave : ℚ) / (monthsInPeriod : ℚ)

/-- Theorem stating that Sheila's monthly savings is $276 -/
theorem sheilas_monthly_savings : 
  monthlySavings 3000 7000 4 23248 = 276 := by
  sorry

end NUMINAMATH_CALUDE_sheilas_monthly_savings_l4046_404614


namespace NUMINAMATH_CALUDE_tangent_line_at_pi_third_l4046_404603

noncomputable def f (x : ℝ) : ℝ := (1/2) * x + Real.sin x

def tangent_line_equation (x y : ℝ) : Prop :=
  6 * x - 6 * y + 3 * Real.sqrt 3 - Real.pi = 0

theorem tangent_line_at_pi_third :
  tangent_line_equation (π/3) (f (π/3)) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_pi_third_l4046_404603


namespace NUMINAMATH_CALUDE_jake_and_sister_weight_l4046_404632

theorem jake_and_sister_weight (jake_weight : ℕ) (h : jake_weight = 188) :
  ∃ (sister_weight : ℕ),
    jake_weight - 8 = 2 * sister_weight ∧
    jake_weight + sister_weight = 278 := by
  sorry

end NUMINAMATH_CALUDE_jake_and_sister_weight_l4046_404632


namespace NUMINAMATH_CALUDE_geometric_series_first_term_l4046_404669

/-- Represents an infinite geometric series -/
structure GeometricSeries where
  a : ℝ  -- first term
  r : ℝ  -- common ratio
  sum : ℝ -- sum of the series

/-- Condition for the first, third, and fourth terms forming an arithmetic sequence -/
def arithmeticSequenceCondition (s : GeometricSeries) : Prop :=
  2 * s.a * s.r^2 = s.a + s.a * s.r^3

/-- The main theorem statement -/
theorem geometric_series_first_term 
  (s : GeometricSeries) 
  (h_sum : s.sum = 2020)
  (h_arith : arithmeticSequenceCondition s)
  (h_converge : abs s.r < 1) :
  s.a = 1010 * (1 + Real.sqrt 5) :=
sorry

end NUMINAMATH_CALUDE_geometric_series_first_term_l4046_404669


namespace NUMINAMATH_CALUDE_andrew_final_share_l4046_404629

def total_stickers : ℕ := 2800

def ratio_sum : ℚ := 3/5 + 1 + 3/4 + 1/2 + 7/4

def andrew_initial_share : ℚ := (1 : ℚ) * total_stickers / ratio_sum

def sam_initial_share : ℚ := (3/4 : ℚ) * total_stickers / ratio_sum

def sam_to_andrew : ℚ := 0.4 * sam_initial_share

theorem andrew_final_share :
  ⌊andrew_initial_share + sam_to_andrew⌋ = 791 :=
sorry

end NUMINAMATH_CALUDE_andrew_final_share_l4046_404629


namespace NUMINAMATH_CALUDE_min_value_trig_expression_limit_approaches_min_value_l4046_404675

open Real

theorem min_value_trig_expression (θ : ℝ) (h : 0 < θ ∧ θ < π/2) :
  3 * cos θ + 1 / (2 * sin θ) + 2 * sqrt 2 * tan θ ≥ 3 * (3 ^ (1/3)) * (sqrt 2 ^ (1/3)) :=
by sorry

theorem limit_approaches_min_value :
  ∀ ε > 0, ∃ δ > 0, ∀ θ, 0 < θ ∧ θ < δ →
    abs ((3 * cos θ + 1 / (2 * sin θ) + 2 * sqrt 2 * tan θ) - 3 * (3 ^ (1/3)) * (sqrt 2 ^ (1/3))) < ε :=
by sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_limit_approaches_min_value_l4046_404675


namespace NUMINAMATH_CALUDE_transformation_possible_l4046_404620

/-- Represents a move that can be applied to a sequence of numbers. -/
inductive Move
  | RotateThree (x y z : ℕ) : Move
  | SwapTwo (x y : ℕ) : Move

/-- Checks if a move is valid according to the rules. -/
def isValidMove (move : Move) : Prop :=
  match move with
  | Move.RotateThree x y z => (x + y + z) % 3 = 0
  | Move.SwapTwo x y => (x - y) % 3 = 0 ∨ (y - x) % 3 = 0

/-- Represents a sequence of numbers. -/
def Sequence := List ℕ

/-- Applies a move to a sequence. -/
def applyMove (seq : Sequence) (move : Move) : Sequence :=
  sorry

/-- Checks if a sequence can be transformed into another sequence using valid moves. -/
def canTransform (initial final : Sequence) : Prop :=
  ∃ (moves : List Move), (∀ move ∈ moves, isValidMove move) ∧
    (moves.foldl applyMove initial = final)

/-- The main theorem to be proved. -/
theorem transformation_possible (n : ℕ) :
  n > 1 →
  (canTransform (List.range n) ((n :: List.range (n-1)))) ↔ (n % 3 = 0 ∨ n % 3 = 1) :=
  sorry

end NUMINAMATH_CALUDE_transformation_possible_l4046_404620


namespace NUMINAMATH_CALUDE_fraction_simplification_l4046_404628

theorem fraction_simplification (a x b : ℝ) (hb : b > 0) :
  (Real.sqrt b * (Real.sqrt (a^2 + x^2) - (x^2 - a^2) / Real.sqrt (a^2 + x^2))) / (b * (a^2 + x^2)) =
  (2 * a^2 * Real.sqrt b) / (b * (a^2 + x^2)^(3/2)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l4046_404628


namespace NUMINAMATH_CALUDE_function_min_value_l4046_404631

-- Define the function f(x)
def f (x : ℝ) (m : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + m

-- State the theorem
theorem function_min_value 
  (m : ℝ) 
  (h_max : ∃ x ∈ Set.Icc (-2 : ℝ) 2, f x m = 3 ∧ ∀ y ∈ Set.Icc (-2 : ℝ) 2, f y m ≤ 3) :
  ∃ x ∈ Set.Icc (-2 : ℝ) 2, f x m = -37 ∧ ∀ y ∈ Set.Icc (-2 : ℝ) 2, f y m ≥ -37 :=
sorry

end NUMINAMATH_CALUDE_function_min_value_l4046_404631


namespace NUMINAMATH_CALUDE_two_points_determine_line_l4046_404681

-- Define a Point type
structure Point where
  x : ℝ
  y : ℝ

-- Define a Line type
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Function to check if a point lies on a line
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Theorem statement
theorem two_points_determine_line (p1 p2 : Point) (h : p1 ≠ p2) :
  ∃! l : Line, pointOnLine p1 l ∧ pointOnLine p2 l :=
sorry

end NUMINAMATH_CALUDE_two_points_determine_line_l4046_404681


namespace NUMINAMATH_CALUDE_no_guaranteed_primes_l4046_404613

theorem no_guaranteed_primes (n : ℕ) (h : n > 1) :
  ∀ p : ℕ, Prime p → (p ∉ Set.Ioo (n.factorial) (n.factorial + 2*n)) :=
sorry

end NUMINAMATH_CALUDE_no_guaranteed_primes_l4046_404613


namespace NUMINAMATH_CALUDE_ellipse_m_range_l4046_404646

/-- Represents a point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines the equation of the curve -/
def curve_equation (m : ℝ) (p : Point) : Prop :=
  m * (p.x^2 + p.y^2 + 2*p.y + 1) = (p.x - 2*p.y + 3)^2

/-- Defines what it means for the curve to be an ellipse -/
def is_ellipse (m : ℝ) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧
  ∀ (p : Point), curve_equation m p ↔ 
    (p.x - h)^2 / a^2 + (p.y - k)^2 / b^2 = 1
  where
    h := 0  -- center x-coordinate
    k := -1 -- center y-coordinate

/-- The main theorem stating the range of m for which the curve is an ellipse -/
theorem ellipse_m_range :
  ∀ m : ℝ, is_ellipse m ↔ m > 5 :=
sorry

end NUMINAMATH_CALUDE_ellipse_m_range_l4046_404646


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l4046_404612

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_product 
  (a : ℕ → ℝ) 
  (h_geo : geometric_sequence a) 
  (h_prod : a 5 * a 14 = 5) : 
  a 8 * a 9 * a 10 * a 11 = 25 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l4046_404612


namespace NUMINAMATH_CALUDE_num_lists_15_4_l4046_404641

/-- The number of elements in the set to draw from -/
def n : ℕ := 15

/-- The number of draws -/
def k : ℕ := 4

/-- The number of possible lists when drawing k times with replacement from a set of n elements -/
def num_lists (n k : ℕ) : ℕ := n^k

/-- Theorem: The number of possible lists when drawing 4 times with replacement from a set of 15 elements is 50625 -/
theorem num_lists_15_4 : num_lists n k = 50625 := by
  sorry

end NUMINAMATH_CALUDE_num_lists_15_4_l4046_404641


namespace NUMINAMATH_CALUDE_transportation_cost_l4046_404623

/-- Transportation problem theorem -/
theorem transportation_cost
  (city_A_supply : ℕ)
  (city_B_supply : ℕ)
  (market_C_demand : ℕ)
  (market_D_demand : ℕ)
  (cost_A_to_C : ℕ)
  (cost_A_to_D : ℕ)
  (cost_B_to_C : ℕ)
  (cost_B_to_D : ℕ)
  (x : ℕ)
  (h1 : city_A_supply = 240)
  (h2 : city_B_supply = 260)
  (h3 : market_C_demand = 200)
  (h4 : market_D_demand = 300)
  (h5 : cost_A_to_C = 20)
  (h6 : cost_A_to_D = 30)
  (h7 : cost_B_to_C = 24)
  (h8 : cost_B_to_D = 32)
  (h9 : x ≤ city_A_supply)
  (h10 : x ≤ market_C_demand) :
  (cost_A_to_C * x +
   cost_A_to_D * (city_A_supply - x) +
   cost_B_to_C * (market_C_demand - x) +
   cost_B_to_D * (market_D_demand - (city_A_supply - x))) =
  13920 - 2 * x :=
by sorry

end NUMINAMATH_CALUDE_transportation_cost_l4046_404623


namespace NUMINAMATH_CALUDE_product_equals_four_l4046_404691

theorem product_equals_four (m n : ℝ) (h : m + n = (1/2) * m * n) :
  (m - 2) * (n - 2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_four_l4046_404691


namespace NUMINAMATH_CALUDE_sum_of_digits_of_B_is_seven_l4046_404666

theorem sum_of_digits_of_B_is_seven :
  ∃ (A B : ℕ),
    (A ≡ (16^16 : ℕ) [MOD 9]) →
    (B ≡ A [MOD 9]) →
    (∃ (C : ℕ), C < 10 ∧ C ≡ B [MOD 9] ∧ C = 7) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_B_is_seven_l4046_404666


namespace NUMINAMATH_CALUDE_equation_four_solutions_l4046_404651

theorem equation_four_solutions 
  (a b c d e : ℝ) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) :
  ∃! (s : Finset ℝ), s.card = 4 ∧ ∀ x ∈ s, 
    (x - a) * (x - b) * (x - c) * (x - d) +
    (x - a) * (x - b) * (x - c) * (x - e) +
    (x - a) * (x - b) * (x - d) * (x - e) +
    (x - a) * (x - c) * (x - d) * (x - e) +
    (x - b) * (x - c) * (x - d) * (x - e) = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_four_solutions_l4046_404651


namespace NUMINAMATH_CALUDE_f_injective_l4046_404660

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def injective {α β : Type} (f : α → β) : Prop :=
  ∀ a b : α, f a = f b → a = b

theorem f_injective (f : ℕ → ℕ) 
  (h : ∀ (x y : ℕ), is_perfect_square (f x + y) ↔ is_perfect_square (x + f y)) :
  injective f := by
sorry

end NUMINAMATH_CALUDE_f_injective_l4046_404660


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l4046_404696

theorem greatest_divisor_with_remainders : 
  ∃ (d : ℕ), d > 0 ∧ 
    (∃ (q₁ q₂ q₃ : ℕ), 2674 = d * q₁ + 5 ∧ 3486 = d * q₂ + 7 ∧ 4328 = d * q₃ + 9) ∧
    ∀ (k : ℕ), k > 0 → 
      (∃ (r₁ r₂ r₃ : ℕ), 2674 = k * r₁ + 5 ∧ 3486 = k * r₂ + 7 ∧ 4328 = k * r₃ + 9) →
      k ≤ d :=
by sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l4046_404696


namespace NUMINAMATH_CALUDE_fraction_comparisons_and_absolute_value_l4046_404643

theorem fraction_comparisons_and_absolute_value :
  (-3 : ℚ) / 7 < (-8 : ℚ) / 21 ∧
  (-5 : ℚ) / 6 > (-6 : ℚ) / 7 ∧
  |3.1 - Real.pi| = Real.pi - 3.1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparisons_and_absolute_value_l4046_404643


namespace NUMINAMATH_CALUDE_organizing_related_to_excellent_scores_expectation_X_l4046_404601

-- Define the total number of students
def total_students : ℕ := 100

-- Define the number of students with excellent and poor math scores
def excellent_scores : ℕ := 40
def poor_scores : ℕ := 60

-- Define the number of students not organizing regularly
def not_organizing_excellent : ℕ := 8  -- 20% of 40
def not_organizing_poor : ℕ := 32

-- Define the chi-square statistic
def chi_square (a b c d : ℕ) : ℚ :=
  let n := a + b + c + d
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Define the critical value for 99% certainty
def critical_value : ℚ := 6635 / 1000

-- Theorem for the relationship between organizing regularly and excellent math scores
theorem organizing_related_to_excellent_scores :
  chi_square not_organizing_excellent (excellent_scores - not_organizing_excellent)
              not_organizing_poor (poor_scores - not_organizing_poor) > critical_value := by
  sorry

-- Define the probability distribution of X
def prob_X (x : ℕ) : ℚ :=
  match x with
  | 0 => 28 / 45
  | 1 => 16 / 45
  | 2 => 1 / 45
  | _ => 0

-- Theorem for the expectation of X
theorem expectation_X :
  (0 : ℚ) * prob_X 0 + 1 * prob_X 1 + 2 * prob_X 2 = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_organizing_related_to_excellent_scores_expectation_X_l4046_404601


namespace NUMINAMATH_CALUDE_triangle_semicircle_inequality_l4046_404668

-- Define a triangle by its semiperimeter and inradius
structure Triangle where
  s : ℝ  -- semiperimeter
  r : ℝ  -- inradius
  s_pos : 0 < s
  r_pos : 0 < r

-- Define the radius of the circle tangent to the three semicircles
noncomputable def t (tri : Triangle) : ℝ := sorry

-- State the theorem
theorem triangle_semicircle_inequality (tri : Triangle) :
  tri.s / 2 < t tri ∧ t tri ≤ tri.s / 2 + (1 - Real.sqrt 3 / 2) * tri.r := by
  sorry

end NUMINAMATH_CALUDE_triangle_semicircle_inequality_l4046_404668


namespace NUMINAMATH_CALUDE_number_of_hens_l4046_404600

theorem number_of_hens (total_animals : ℕ) (total_feet : ℕ) (hens : ℕ) (cows : ℕ) : 
  total_animals = 60 →
  total_feet = 200 →
  total_animals = hens + cows →
  total_feet = 2 * hens + 4 * cows →
  hens = 20 := by
sorry

end NUMINAMATH_CALUDE_number_of_hens_l4046_404600


namespace NUMINAMATH_CALUDE_intersection_M_N_l4046_404690

-- Define the sets M and N
def M : Set ℝ := {x | x - 2 > 0}
def N : Set ℝ := {y | ∃ x, y = Real.sqrt (x^2 + 1)}

-- State the theorem
theorem intersection_M_N : M ∩ N = {x | x > 2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l4046_404690


namespace NUMINAMATH_CALUDE_malcom_remaining_cards_l4046_404694

-- Define the number of cards Brandon has
def brandon_cards : ℕ := 20

-- Define the number of additional cards Malcom has compared to Brandon
def malcom_extra_cards : ℕ := 8

-- Define Malcom's initial number of cards
def malcom_initial_cards : ℕ := brandon_cards + malcom_extra_cards

-- Define the number of cards Malcom gives away
def malcom_cards_given : ℕ := malcom_initial_cards / 2

-- Theorem to prove
theorem malcom_remaining_cards :
  malcom_initial_cards - malcom_cards_given = 14 := by
  sorry

end NUMINAMATH_CALUDE_malcom_remaining_cards_l4046_404694


namespace NUMINAMATH_CALUDE_netGainDifference_l4046_404639

/-- Represents an applicant for a job position -/
structure Applicant where
  salary : ℕ
  revenue : ℕ
  trainingMonths : ℕ
  trainingCostPerMonth : ℕ
  hiringBonusPercent : ℕ

/-- Calculates the net gain for the company from an applicant -/
def netGain (a : Applicant) : ℕ :=
  a.revenue - a.salary - (a.trainingMonths * a.trainingCostPerMonth) - (a.salary * a.hiringBonusPercent / 100)

/-- The first applicant's details -/
def applicant1 : Applicant :=
  { salary := 42000
    revenue := 93000
    trainingMonths := 3
    trainingCostPerMonth := 1200
    hiringBonusPercent := 0 }

/-- The second applicant's details -/
def applicant2 : Applicant :=
  { salary := 45000
    revenue := 92000
    trainingMonths := 0
    trainingCostPerMonth := 0
    hiringBonusPercent := 1 }

/-- Theorem stating the difference in net gain between the two applicants -/
theorem netGainDifference : netGain applicant1 - netGain applicant2 = 850 := by
  sorry

end NUMINAMATH_CALUDE_netGainDifference_l4046_404639


namespace NUMINAMATH_CALUDE_andrews_game_preparation_time_l4046_404689

/-- The time it takes to prepare all games -/
def total_preparation_time (time_per_game : ℕ) (num_games : ℕ) : ℕ :=
  time_per_game * num_games

/-- Theorem: The total preparation time for 5 games, each taking 5 minutes, is 25 minutes -/
theorem andrews_game_preparation_time :
  total_preparation_time 5 5 = 25 := by
sorry

end NUMINAMATH_CALUDE_andrews_game_preparation_time_l4046_404689


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_m_part2_l4046_404622

-- Define the function f
def f (x m : ℝ) : ℝ := |x + 1| + |x - 2| - m

-- Part 1: Solution set for f(x) > 0 when m = 5
theorem solution_set_part1 : 
  {x : ℝ | f x 5 > 0} = {x : ℝ | x < -2 ∨ x > 3} := by sorry

-- Part 2: Range of m for which f(x) ≥ 2 has solution set ℝ
theorem range_of_m_part2 : 
  {m : ℝ | ∀ x, f x m ≥ 2} = {m : ℝ | m ≤ 1} := by sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_m_part2_l4046_404622


namespace NUMINAMATH_CALUDE_total_distance_via_intermediate_point_l4046_404686

/-- The total distance traveled from (2, 3) to (-3, 2) via (1, -1) is √17 + 5. -/
theorem total_distance_via_intermediate_point :
  let start : ℝ × ℝ := (2, 3)
  let intermediate : ℝ × ℝ := (1, -1)
  let end_point : ℝ × ℝ := (-3, 2)
  let distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  distance start intermediate + distance intermediate end_point = Real.sqrt 17 + 5 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_via_intermediate_point_l4046_404686


namespace NUMINAMATH_CALUDE_cash_realized_approx_103_74_l4046_404678

/-- The cash realized on selling a stock, given the brokerage rate and total amount including brokerage -/
def cash_realized (brokerage_rate : ℚ) (total_amount : ℚ) : ℚ :=
  total_amount / (1 + brokerage_rate)

/-- Theorem stating that the cash realized is approximately 103.74 given the problem conditions -/
theorem cash_realized_approx_103_74 :
  let brokerage_rate : ℚ := 1 / 400  -- 1/4% expressed as a fraction
  let total_amount : ℚ := 104
  |cash_realized brokerage_rate total_amount - 103.74| < 0.01 := by
  sorry

#eval cash_realized (1/400) 104

end NUMINAMATH_CALUDE_cash_realized_approx_103_74_l4046_404678


namespace NUMINAMATH_CALUDE_system_solution_l4046_404619

theorem system_solution : ∃! (x y : ℝ), x + y = 2 ∧ 3 * x + y = 4 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l4046_404619


namespace NUMINAMATH_CALUDE_ab_greater_than_b_squared_l4046_404665

theorem ab_greater_than_b_squared {a b : ℝ} (h1 : a < b) (h2 : b < 0) : a * b > b ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_ab_greater_than_b_squared_l4046_404665


namespace NUMINAMATH_CALUDE_pascals_triangle_51_numbers_l4046_404685

theorem pascals_triangle_51_numbers (n : ℕ) : n + 1 = 51 → Nat.choose n 2 = 1225 := by
  sorry

end NUMINAMATH_CALUDE_pascals_triangle_51_numbers_l4046_404685


namespace NUMINAMATH_CALUDE_op_35_77_l4046_404602

-- Define the operation @
def op (a b : ℕ+) : ℚ := (a * b) / (a + b)

-- Theorem statement
theorem op_35_77 : op 35 77 = 2695 / 112 := by
  sorry

end NUMINAMATH_CALUDE_op_35_77_l4046_404602


namespace NUMINAMATH_CALUDE_rope_length_difference_l4046_404662

/-- Given three ropes with lengths in ratio 4 : 5 : 6, where the shortest is 80 meters,
    prove that the sum of the longest and shortest is 100 meters more than the middle. -/
theorem rope_length_difference (shortest middle longest : ℝ) : 
  shortest = 80 ∧ 
  5 * shortest = 4 * middle ∧ 
  6 * shortest = 4 * longest →
  longest + shortest = middle + 100 := by
  sorry

end NUMINAMATH_CALUDE_rope_length_difference_l4046_404662


namespace NUMINAMATH_CALUDE_min_value_problem_l4046_404638

theorem min_value_problem (a b c d e f g h : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (pos_e : 0 < e) (pos_f : 0 < f) (pos_g : 0 < g) (pos_h : 0 < h)
  (h1 : a * b * c * d = 8)
  (h2 : e * f * g * h = 16)
  (h3 : a + b + c + d = e * f * g) :
  64 ≤ (a*e)^2 + (b*f)^2 + (c*g)^2 + (d*h)^2 ∧ 
  ∃ (a' b' c' d' e' f' g' h' : ℝ), 
    0 < a' ∧ 0 < b' ∧ 0 < c' ∧ 0 < d' ∧ 
    0 < e' ∧ 0 < f' ∧ 0 < g' ∧ 0 < h' ∧
    a' * b' * c' * d' = 8 ∧
    e' * f' * g' * h' = 16 ∧
    a' + b' + c' + d' = e' * f' * g' ∧
    (a'*e')^2 + (b'*f')^2 + (c'*g')^2 + (d'*h')^2 = 64 :=
by sorry

end NUMINAMATH_CALUDE_min_value_problem_l4046_404638


namespace NUMINAMATH_CALUDE_square_complex_real_iff_a_or_b_zero_l4046_404672

theorem square_complex_real_iff_a_or_b_zero (a b : ℝ) :
  let z : ℂ := Complex.mk a b
  (∃ (r : ℝ), z^2 = (r : ℂ)) ↔ a = 0 ∨ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_complex_real_iff_a_or_b_zero_l4046_404672


namespace NUMINAMATH_CALUDE_seating_arrangements_l4046_404698

def total_arrangements (n : ℕ) : ℕ := Nat.factorial n

def adjacent_arrangements (n : ℕ) : ℕ := (Nat.factorial (n - 1)) * 2

theorem seating_arrangements (n : ℕ) (h : n = 8) : 
  total_arrangements n - adjacent_arrangements n = 30240 :=
by sorry

end NUMINAMATH_CALUDE_seating_arrangements_l4046_404698


namespace NUMINAMATH_CALUDE_monic_cubic_polynomial_uniqueness_l4046_404659

/-- A monic cubic polynomial with real coefficients -/
def MonicCubicPolynomial (a b c : ℝ) : ℝ → ℂ :=
  fun x => (x : ℂ)^3 + a * (x : ℂ)^2 + b * (x : ℂ) + c

theorem monic_cubic_polynomial_uniqueness (a b c : ℝ) :
  let p := MonicCubicPolynomial a b c
  p (1 + 3*I) = 0 ∧ p 0 = -108 →
  a = -12.8 ∧ b = 31 ∧ c = -108 := by
  sorry

end NUMINAMATH_CALUDE_monic_cubic_polynomial_uniqueness_l4046_404659


namespace NUMINAMATH_CALUDE_polynomial_non_negative_l4046_404663

theorem polynomial_non_negative (p q : ℝ) (h : q > p^2) :
  ∀ x : ℝ, x^2 + 2*p*x + q ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_non_negative_l4046_404663


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l4046_404636

theorem fraction_to_decimal (n d : ℕ) (h : d ≠ 0) :
  (n : ℚ) / d = 0.33125 ↔ n = 53 ∧ d = 160 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l4046_404636


namespace NUMINAMATH_CALUDE_complement_of_M_l4046_404688

def U : Set Nat := {1, 2, 3, 4, 5}
def M : Set Nat := {4, 5}

theorem complement_of_M :
  (U \ M) = {1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_l4046_404688


namespace NUMINAMATH_CALUDE_larger_number_problem_l4046_404627

theorem larger_number_problem (x y : ℝ) : 
  x - y = 7 → x + y = 45 → x = 26 ∧ x > y := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l4046_404627


namespace NUMINAMATH_CALUDE_rank_inequality_l4046_404658

variable {n : ℕ}
variable (A B : Matrix (Fin n) (Fin n) ℝ)

theorem rank_inequality (h1 : n ≥ 2) (h2 : B * B = B) :
  Matrix.rank (A * B - B * A) ≤ Matrix.rank (A * B + B * A) := by
  sorry

end NUMINAMATH_CALUDE_rank_inequality_l4046_404658


namespace NUMINAMATH_CALUDE_triangle_is_equilateral_l4046_404684

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- angles
  (a b c : ℝ)  -- side lengths

-- Define the conditions
def is_valid_triangle (t : Triangle) : Prop :=
  t.B = 60 ∧ t.b^2 = t.a * t.c

-- Theorem statement
theorem triangle_is_equilateral (t : Triangle) (h : is_valid_triangle t) : 
  t.A = 60 ∧ t.B = 60 ∧ t.C = 60 :=
sorry

end NUMINAMATH_CALUDE_triangle_is_equilateral_l4046_404684


namespace NUMINAMATH_CALUDE_binomial_20_4_l4046_404634

theorem binomial_20_4 : Nat.choose 20 4 = 4845 := by
  sorry

end NUMINAMATH_CALUDE_binomial_20_4_l4046_404634


namespace NUMINAMATH_CALUDE_similar_triangles_leg_sum_l4046_404657

theorem similar_triangles_leg_sum (a b c d : ℝ) : 
  a > 0 → b > 0 → c > 0 → d > 0 →
  (1/2) * a * b = 10 →
  a^2 + b^2 = 36 →
  (1/2) * c * d = 360 →
  c = 6 * a →
  d = 6 * b →
  c + d = 16 * Real.sqrt 30 := by
sorry

end NUMINAMATH_CALUDE_similar_triangles_leg_sum_l4046_404657


namespace NUMINAMATH_CALUDE_factor_of_polynomial_l4046_404618

theorem factor_of_polynomial (x : ℝ) : 
  ∃ (k : ℝ), 5 * x^2 + 17 * x - 12 = (x + 4) * k := by
  sorry

end NUMINAMATH_CALUDE_factor_of_polynomial_l4046_404618


namespace NUMINAMATH_CALUDE_contractor_daily_wage_l4046_404693

/-- Proves that the daily wage is 25 given the contract conditions --/
theorem contractor_daily_wage
  (total_days : ℕ)
  (absent_days : ℕ)
  (daily_fine : ℚ)
  (total_payment : ℚ)
  (h1 : total_days = 30)
  (h2 : absent_days = 6)
  (h3 : daily_fine = 7.5)
  (h4 : total_payment = 555)
  : ∃ (daily_wage : ℚ),
    daily_wage * (total_days - absent_days : ℚ) - daily_fine * absent_days = total_payment ∧
    daily_wage = 25 := by
  sorry


end NUMINAMATH_CALUDE_contractor_daily_wage_l4046_404693


namespace NUMINAMATH_CALUDE_equation_solution_l4046_404683

theorem equation_solution : 
  ∀ x y : ℚ, 
  y = 3 * x → 
  (5 * y^2 + 2 * y + 3 = 3 * (9 * x^2 + y + 1)) → 
  (x = 0 ∨ x = 1/6) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l4046_404683


namespace NUMINAMATH_CALUDE_caspers_candies_l4046_404682

theorem caspers_candies (initial_candies : ℕ) : 
  let day1_after_eating := (3 * initial_candies) / 4
  let day1_remaining := day1_after_eating - 3
  let day2_after_eating := (4 * day1_remaining) / 5
  let day2_remaining := day2_after_eating - 5
  let day3_after_giving := day2_remaining - 7
  let final_candies := (5 * day3_after_giving) / 6
  final_candies = 10 → initial_candies = 44 := by
sorry

end NUMINAMATH_CALUDE_caspers_candies_l4046_404682


namespace NUMINAMATH_CALUDE_vector_equation_solution_l4046_404661

theorem vector_equation_solution (a b : ℝ × ℝ) (m n : ℝ) :
  a = (-3, 1) →
  b = (-1, 2) →
  m • a - n • b = (10, 0) →
  m = -4 ∧ n = -2 := by sorry

end NUMINAMATH_CALUDE_vector_equation_solution_l4046_404661


namespace NUMINAMATH_CALUDE_event_probability_l4046_404633

theorem event_probability (p : ℝ) : 
  (0 ≤ p) ∧ (p ≤ 1) →
  (1 - (1 - p)^3 = 63/64) →
  3 * p * (1 - p)^2 = 9/64 := by
sorry

end NUMINAMATH_CALUDE_event_probability_l4046_404633


namespace NUMINAMATH_CALUDE_exists_valid_arrangement_l4046_404680

/-- Represents a position on a regular 25-gon -/
inductive Position
| Vertex : Fin 25 → Position
| Midpoint : Fin 25 → Position

/-- Represents an arrangement of numbers on a regular 25-gon -/
def Arrangement := Position → Fin 50

/-- Checks if the sum of numbers at the ends and midpoint of a side is constant -/
def isConstantSum (arr : Arrangement) : Prop :=
  ∃ s : ℕ, ∀ i : Fin 25,
    (arr (Position.Vertex i)).val + 
    (arr (Position.Midpoint i)).val + 
    (arr (Position.Vertex ((i.val + 1) % 25 : Fin 25))).val = s

/-- Theorem stating the existence of a valid arrangement -/
theorem exists_valid_arrangement : ∃ arr : Arrangement, isConstantSum arr ∧ 
  (∀ p : Position, (arr p).val ≥ 1 ∧ (arr p).val ≤ 50) ∧
  (∀ p q : Position, p ≠ q → arr p ≠ arr q) :=
sorry

end NUMINAMATH_CALUDE_exists_valid_arrangement_l4046_404680


namespace NUMINAMATH_CALUDE_no_solution_condition_l4046_404626

theorem no_solution_condition (a : ℝ) : 
  (∀ x : ℝ, 3 * |x + 3*a| + |x + a^2| + 2*x ≠ a) ↔ (a < 0 ∨ a > 10) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_condition_l4046_404626
