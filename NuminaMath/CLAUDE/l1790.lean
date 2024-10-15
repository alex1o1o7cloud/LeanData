import Mathlib

namespace NUMINAMATH_CALUDE_fractional_equation_solution_l1790_179069

theorem fractional_equation_solution :
  ∃ x : ℝ, (x + 2) / (x - 1) = 0 ∧ x = -2 :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l1790_179069


namespace NUMINAMATH_CALUDE_nearest_integer_to_three_plus_sqrt_three_fourth_l1790_179011

theorem nearest_integer_to_three_plus_sqrt_three_fourth (x : ℝ) : 
  x = (3 + Real.sqrt 3)^4 → 
  ∃ n : ℤ, n = 504 ∧ ∀ m : ℤ, |x - n| ≤ |x - m| := by
  sorry

end NUMINAMATH_CALUDE_nearest_integer_to_three_plus_sqrt_three_fourth_l1790_179011


namespace NUMINAMATH_CALUDE_x_value_at_y_25_l1790_179096

/-- The constant ratio between (4x - 5) and (2y + 20) -/
def k : ℚ := (4 * 1 - 5) / (2 * 5 + 20)

/-- Theorem stating that given the constant ratio k and the initial condition,
    x equals 2/3 when y equals 25 -/
theorem x_value_at_y_25 (x y : ℚ) 
  (h1 : (4 * x - 5) / (2 * y + 20) = k) 
  (h2 : x = 1 → y = 5) :
  y = 25 → x = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_x_value_at_y_25_l1790_179096


namespace NUMINAMATH_CALUDE_complex_absolute_value_product_l1790_179051

theorem complex_absolute_value_product : 
  Complex.abs ((3 * Real.sqrt 5 - 6 * Complex.I) * (2 * Real.sqrt 2 + 4 * Complex.I)) = 18 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_product_l1790_179051


namespace NUMINAMATH_CALUDE_pool_filling_time_l1790_179038

theorem pool_filling_time (faster_pipe_rate : ℝ) (slower_pipe_rate : ℝ) :
  faster_pipe_rate = 1 / 9 →
  slower_pipe_rate = faster_pipe_rate / 1.25 →
  1 / (faster_pipe_rate + slower_pipe_rate) = 5 := by
  sorry

end NUMINAMATH_CALUDE_pool_filling_time_l1790_179038


namespace NUMINAMATH_CALUDE_large_bucket_relation_tank_capacity_is_21_l1790_179094

/-- The capacity of a small bucket in liters -/
def small_bucket_capacity : ℝ := 0.5

/-- The capacity of a large bucket in liters -/
def large_bucket_capacity : ℝ := 4

/-- The number of small buckets used to fill the tank -/
def num_small_buckets : ℕ := 2

/-- The number of large buckets used to fill the tank -/
def num_large_buckets : ℕ := 5

/-- The relationship between small and large bucket capacities -/
theorem large_bucket_relation : large_bucket_capacity = 2 * small_bucket_capacity + 3 := by sorry

/-- The capacity of the tank in liters -/
def tank_capacity : ℝ := num_small_buckets * small_bucket_capacity + num_large_buckets * large_bucket_capacity

theorem tank_capacity_is_21 : tank_capacity = 21 := by sorry

end NUMINAMATH_CALUDE_large_bucket_relation_tank_capacity_is_21_l1790_179094


namespace NUMINAMATH_CALUDE_min_value_of_f_l1790_179024

-- Define the quadratic function
def f (x : ℝ) : ℝ := x^2 + 6*x - 8

-- Theorem stating that f(x) achieves its minimum when x = -3
theorem min_value_of_f :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ x_min = -3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1790_179024


namespace NUMINAMATH_CALUDE_mans_speed_in_still_water_l1790_179081

/-- Proves that given a man who swims downstream 36 km in 6 hours and upstream 18 km in 6 hours, his speed in still water is 4.5 km/h. -/
theorem mans_speed_in_still_water 
  (downstream_distance : ℝ) 
  (upstream_distance : ℝ) 
  (time : ℝ) 
  (h1 : downstream_distance = 36) 
  (h2 : upstream_distance = 18) 
  (h3 : time = 6) : 
  ∃ (speed_still_water : ℝ) (stream_speed : ℝ),
    speed_still_water + stream_speed = downstream_distance / time ∧ 
    speed_still_water - stream_speed = upstream_distance / time ∧
    speed_still_water = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_mans_speed_in_still_water_l1790_179081


namespace NUMINAMATH_CALUDE_fraction_solution_l1790_179053

theorem fraction_solution : ∃ x : ℝ, (x - 4) / (x^2) = 0 ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_solution_l1790_179053


namespace NUMINAMATH_CALUDE_parabola_equation_l1790_179025

/-- A parabola with x-axis as axis of symmetry and vertex at origin -/
structure Parabola where
  p : ℝ
  eq : ℝ → ℝ → Prop := fun x y => y^2 = 2 * p * x

/-- The parabola passes through the point (-2, -4) -/
def passes_through (par : Parabola) : Prop :=
  par.eq (-2) (-4)

/-- The standard equation of the parabola is y^2 = -8x -/
def standard_equation (par : Parabola) : Prop :=
  par.p = -4

theorem parabola_equation :
  ∃ (par : Parabola), passes_through par ∧ standard_equation par :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l1790_179025


namespace NUMINAMATH_CALUDE_meaningful_fraction_l1790_179084

theorem meaningful_fraction (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x - 3)) ↔ x ≠ 3 :=
by sorry

end NUMINAMATH_CALUDE_meaningful_fraction_l1790_179084


namespace NUMINAMATH_CALUDE_fibonacci_inequality_l1790_179056

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_inequality (n : ℕ) :
  (fibonacci (n + 1) : ℝ) ^ (1 / n : ℝ) ≥ 1 + 1 / ((fibonacci n : ℝ) ^ (1 / n : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_inequality_l1790_179056


namespace NUMINAMATH_CALUDE_tom_barbados_cost_l1790_179089

/-- The total cost Tom has to pay for his trip to Barbados -/
def total_cost (num_vaccines : ℕ) (vaccine_cost : ℚ) (doctor_visit_cost : ℚ) 
  (insurance_coverage : ℚ) (trip_cost : ℚ) : ℚ :=
  let medical_cost := num_vaccines * vaccine_cost + doctor_visit_cost
  let insurance_payment := medical_cost * insurance_coverage
  let out_of_pocket_medical := medical_cost - insurance_payment
  out_of_pocket_medical + trip_cost

/-- Theorem stating the total cost Tom has to pay -/
theorem tom_barbados_cost : 
  total_cost 10 45 250 (4/5) 1200 = 1340 := by sorry

end NUMINAMATH_CALUDE_tom_barbados_cost_l1790_179089


namespace NUMINAMATH_CALUDE_data_grouping_l1790_179005

theorem data_grouping (max_value min_value class_width : ℕ) 
  (h1 : max_value = 141)
  (h2 : min_value = 40)
  (h3 : class_width = 10) :
  Int.ceil ((max_value - min_value : ℝ) / class_width) = 11 := by
  sorry

#check data_grouping

end NUMINAMATH_CALUDE_data_grouping_l1790_179005


namespace NUMINAMATH_CALUDE_quadratic_minimum_l1790_179050

/-- The quadratic function f(x) = x^2 + 4x - 5 has a minimum value of -9 at x = -2 -/
theorem quadratic_minimum : ∃ (f : ℝ → ℝ), 
  (∀ x, f x = x^2 + 4*x - 5) ∧ 
  (∀ x, f x ≥ f (-2)) ∧
  f (-2) = -9 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l1790_179050


namespace NUMINAMATH_CALUDE_total_slices_is_16_l1790_179013

/-- The number of pizzas Mrs. Hilt bought -/
def num_pizzas : ℕ := 2

/-- The number of slices per pizza -/
def slices_per_pizza : ℕ := 8

/-- The total number of pizza slices Mrs. Hilt had -/
def total_slices : ℕ := num_pizzas * slices_per_pizza

/-- Theorem stating that the total number of pizza slices is 16 -/
theorem total_slices_is_16 : total_slices = 16 := by
  sorry

end NUMINAMATH_CALUDE_total_slices_is_16_l1790_179013


namespace NUMINAMATH_CALUDE_number_satisfying_condition_l1790_179099

theorem number_satisfying_condition : ∃ x : ℝ, 0.65 * x = 0.8 * x - 21 ∧ x = 140 := by
  sorry

end NUMINAMATH_CALUDE_number_satisfying_condition_l1790_179099


namespace NUMINAMATH_CALUDE_musical_group_seats_l1790_179002

/-- Represents the number of seats needed for a musical group --/
def total_seats (F T Tr D C H S P V G : ℕ) : ℕ :=
  F + T + Tr + D + C + H + S + P + V + G

/-- Theorem stating the total number of seats needed for the musical group --/
theorem musical_group_seats :
  ∀ (F T Tr D C H S P V G : ℕ),
    F = 5 →
    T = 3 * F →
    Tr = T - 8 →
    D = Tr + 11 →
    C = 2 * F →
    H = Tr + 3 →
    S = (T + Tr) / 2 →
    P = D + 2 →
    V = H - C →
    G = 3 * F →
    total_seats F T Tr D C H S P V G = 111 :=
by
  sorry

end NUMINAMATH_CALUDE_musical_group_seats_l1790_179002


namespace NUMINAMATH_CALUDE_vacation_days_l1790_179043

theorem vacation_days (rainy_days clear_mornings clear_afternoons : ℕ) 
  (h1 : rainy_days = 13)
  (h2 : clear_mornings = 11)
  (h3 : clear_afternoons = 12)
  (h4 : ∀ d, (d ≤ rainy_days ↔ (d ≤ rainy_days - clear_mornings ∨ d ≤ rainy_days - clear_afternoons) ∧
                               ¬(d ≤ rainy_days - clear_mornings ∧ d ≤ rainy_days - clear_afternoons))) :
  rainy_days + clear_mornings = 18 :=
by sorry

end NUMINAMATH_CALUDE_vacation_days_l1790_179043


namespace NUMINAMATH_CALUDE_project_hours_difference_l1790_179003

theorem project_hours_difference (total_hours kate_hours pat_hours mark_hours : ℕ) : 
  total_hours = 117 →
  pat_hours = 2 * kate_hours →
  pat_hours * 3 = mark_hours →
  total_hours = kate_hours + pat_hours + mark_hours →
  mark_hours - kate_hours = 65 := by
  sorry

end NUMINAMATH_CALUDE_project_hours_difference_l1790_179003


namespace NUMINAMATH_CALUDE_power_equation_l1790_179066

theorem power_equation (a m n : ℝ) (h1 : a^m = 2) (h2 : a^n = 5) : a^(2*m + n) = 20 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_l1790_179066


namespace NUMINAMATH_CALUDE_area_between_circles_and_xaxis_l1790_179073

/-- The area of the region bound by two circles and the x-axis -/
theorem area_between_circles_and_xaxis :
  let c1_center : ℝ × ℝ := (5, 5)
  let c2_center : ℝ × ℝ := (14, 5)
  let radius : ℝ := 3
  let rectangle_area : ℝ := (14 - 5) * 5
  let circle_segment_area : ℝ := 2 * (π * radius^2 / 4)
  rectangle_area - circle_segment_area = 45 - 9 * π / 2 := by
  sorry

end NUMINAMATH_CALUDE_area_between_circles_and_xaxis_l1790_179073


namespace NUMINAMATH_CALUDE_equation_solution_l1790_179012

theorem equation_solution (x : ℝ) (h : x > 4) :
  (Real.sqrt (x - 4 * Real.sqrt (x - 4)) + 2 = Real.sqrt (x + 4 * Real.sqrt (x - 4)) - 2) ↔ x ≥ 8 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l1790_179012


namespace NUMINAMATH_CALUDE_balloon_expenses_l1790_179040

/-- The problem of calculating the total money Harry and Kevin brought to the store -/
theorem balloon_expenses (sheet_cost rope_cost propane_cost : ℕ)
  (helium_cost_per_oz : ℚ)
  (height_per_oz : ℕ)
  (max_height : ℕ) :
  sheet_cost = 42 →
  rope_cost = 18 →
  propane_cost = 14 →
  helium_cost_per_oz = 3/2 →
  height_per_oz = 113 →
  max_height = 9492 →
  ∃ (total_money : ℕ), total_money = 200 := by
  sorry

end NUMINAMATH_CALUDE_balloon_expenses_l1790_179040


namespace NUMINAMATH_CALUDE_pet_store_cats_l1790_179082

theorem pet_store_cats (siamese_cats : ℕ) (cats_sold : ℕ) (cats_left : ℕ) (house_cats : ℕ) : 
  siamese_cats = 38 → 
  cats_sold = 45 → 
  cats_left = 18 → 
  siamese_cats + house_cats - cats_sold = cats_left → 
  house_cats = 25 := by
sorry

end NUMINAMATH_CALUDE_pet_store_cats_l1790_179082


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l1790_179093

theorem fraction_sum_equality : 
  (1/4 - 1/5) / (2/5 - 1/4) + (1/6) / (1/3 - 1/4) = 7/3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l1790_179093


namespace NUMINAMATH_CALUDE_train_speed_kmph_l1790_179039

/-- Converts speed from meters per second to kilometers per hour -/
def mps_to_kmph (speed_mps : ℝ) : ℝ :=
  speed_mps * 3.6

/-- The speed of the train in meters per second -/
def train_speed_mps : ℝ := 37.503

/-- Theorem: The train's speed in kilometers per hour is 135.0108 -/
theorem train_speed_kmph : mps_to_kmph train_speed_mps = 135.0108 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_kmph_l1790_179039


namespace NUMINAMATH_CALUDE_opposite_seats_imply_38_seats_l1790_179076

/-- Represents a round table with equally spaced seats -/
structure RoundTable where
  total_seats : ℕ
  seats_numbered_clockwise : Bool

/-- Defines two people sitting opposite each other on a round table -/
structure OppositeSeats (table : RoundTable) where
  seat1 : ℕ
  seat2 : ℕ
  are_opposite : seat2 - seat1 = table.total_seats / 2

/-- Theorem stating that if two people sit in seats 10 and 29 opposite each other,
    then the total number of seats is 38 -/
theorem opposite_seats_imply_38_seats (table : RoundTable)
  (opposite_pair : OppositeSeats table)
  (h1 : opposite_pair.seat1 = 10)
  (h2 : opposite_pair.seat2 = 29)
  (h3 : table.seats_numbered_clockwise = true) :
  table.total_seats = 38 := by
  sorry

end NUMINAMATH_CALUDE_opposite_seats_imply_38_seats_l1790_179076


namespace NUMINAMATH_CALUDE_isosceles_triangle_condition_l1790_179098

/-- If a, b, c are the sides of a triangle and satisfy the given equation,
    then the triangle is isosceles. -/
theorem isosceles_triangle_condition (a b c : ℝ) :
  a > 0 → b > 0 → c > 0 →
  a + b > c → b + c > a → c + a > b →
  3 * a^3 + 6 * a^2 * b - 3 * a^2 * c - 6 * a * b * c = 0 →
  (a = b ∨ b = c ∨ c = a) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_condition_l1790_179098


namespace NUMINAMATH_CALUDE_area_regular_hexagon_inscribed_circle_l1790_179054

/-- The area of a regular hexagon inscribed in a circle with radius 3 units -/
theorem area_regular_hexagon_inscribed_circle (r : ℝ) (h : r = 3) : 
  (6 : ℝ) * ((r^2 * Real.sqrt 3) / 4) = (27 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_area_regular_hexagon_inscribed_circle_l1790_179054


namespace NUMINAMATH_CALUDE_john_chips_bought_l1790_179087

-- Define the cost of chips and corn chips
def chip_cost : ℚ := 2
def corn_chip_cost : ℚ := 3/2

-- Define John's budget
def budget : ℚ := 45

-- Define the number of corn chips John can buy with remaining money
def corn_chips_bought : ℚ := 10

-- Define the function to calculate the number of corn chips that can be bought with remaining money
def corn_chips_buyable (x : ℚ) : ℚ := (budget - chip_cost * x) / corn_chip_cost

-- Theorem statement
theorem john_chips_bought : 
  ∃ (x : ℚ), x = 15 ∧ corn_chips_buyable x = corn_chips_bought :=
sorry

end NUMINAMATH_CALUDE_john_chips_bought_l1790_179087


namespace NUMINAMATH_CALUDE_largest_n_binomial_equality_l1790_179088

theorem largest_n_binomial_equality : 
  (∃ n : ℕ, (Nat.choose 10 3 + Nat.choose 10 4 = Nat.choose 11 n)) ∧
  (∀ m : ℕ, m > 7 → Nat.choose 10 3 + Nat.choose 10 4 ≠ Nat.choose 11 m) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_binomial_equality_l1790_179088


namespace NUMINAMATH_CALUDE_max_subset_size_2021_l1790_179063

/-- Given a natural number N, returns the maximum size of a subset A of {1, ..., N}
    such that any two numbers in A are neither coprime nor have a divisibility relationship. -/
def maxSubsetSize (N : ℕ) : ℕ :=
  sorry

/-- The maximum subset size for N = 2021 is 505. -/
theorem max_subset_size_2021 : maxSubsetSize 2021 = 505 := by
  sorry

end NUMINAMATH_CALUDE_max_subset_size_2021_l1790_179063


namespace NUMINAMATH_CALUDE_price_restoration_l1790_179067

theorem price_restoration (original_price : ℝ) (markup_percentage : ℝ) (reduction_percentage : ℝ) : 
  markup_percentage = 25 →
  reduction_percentage = 20 →
  original_price * (1 + markup_percentage / 100) * (1 - reduction_percentage / 100) = original_price :=
by
  sorry

end NUMINAMATH_CALUDE_price_restoration_l1790_179067


namespace NUMINAMATH_CALUDE_sum_equality_l1790_179010

theorem sum_equality (a b : ℝ) (h : a/b + a/b^2 + a/b^3 + a/b^4 + a/b^5 = 3) :
  (∑' n, (2*a) / (a+b)^n) = (6*(1 - 1/b^5)) / (4 - 1/b^5) :=
by sorry

end NUMINAMATH_CALUDE_sum_equality_l1790_179010


namespace NUMINAMATH_CALUDE_pauls_hourly_rate_is_35_l1790_179031

/-- Paul's Plumbing's hourly labor charge -/
def pauls_hourly_rate : ℝ := 35

/-- Paul's Plumbing's site visit fee -/
def pauls_visit_fee : ℝ := 55

/-- Reliable Plumbing's site visit fee -/
def reliable_visit_fee : ℝ := 75

/-- Reliable Plumbing's hourly labor charge -/
def reliable_hourly_rate : ℝ := 30

/-- The number of hours worked -/
def hours_worked : ℝ := 4

theorem pauls_hourly_rate_is_35 :
  pauls_hourly_rate = 35 ∧
  pauls_visit_fee + hours_worked * pauls_hourly_rate =
  reliable_visit_fee + hours_worked * reliable_hourly_rate :=
by sorry

end NUMINAMATH_CALUDE_pauls_hourly_rate_is_35_l1790_179031


namespace NUMINAMATH_CALUDE_male_listeners_count_l1790_179028

/-- Represents the survey results of radio station XYZ -/
structure SurveyResults where
  total_listeners : ℕ
  female_listeners : ℕ
  male_non_listeners : ℕ
  total_non_listeners : ℕ

/-- Calculates the number of male listeners given the survey results -/
def male_listeners (survey : SurveyResults) : ℕ :=
  survey.total_listeners - survey.female_listeners

/-- Theorem stating that the number of male listeners is 85 -/
theorem male_listeners_count (survey : SurveyResults) 
  (h1 : survey.total_listeners = 160)
  (h2 : survey.female_listeners = 75) :
  male_listeners survey = 85 := by
  sorry

#eval male_listeners { total_listeners := 160, female_listeners := 75, male_non_listeners := 85, total_non_listeners := 180 }

end NUMINAMATH_CALUDE_male_listeners_count_l1790_179028


namespace NUMINAMATH_CALUDE_purely_imaginary_roots_l1790_179029

theorem purely_imaginary_roots (z : ℂ) (k : ℝ) : 
  (∀ r : ℂ, 20 * r^2 + 6 * Complex.I * r - k = 0 → ∃ b : ℝ, r = Complex.I * b) ↔ k = 9/5 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_roots_l1790_179029


namespace NUMINAMATH_CALUDE_largest_s_value_l1790_179037

theorem largest_s_value (r s : ℕ) (hr : r ≥ s) (hs : s ≥ 3) : 
  (59 * (s - 2) * r = 58 * s * (r - 2)) → s ≤ 117 ∧ ∃ r', r' ≥ s ∧ 59 * (117 - 2) * r' = 58 * 117 * (r' - 2) := by
  sorry

#check largest_s_value

end NUMINAMATH_CALUDE_largest_s_value_l1790_179037


namespace NUMINAMATH_CALUDE_files_deleted_l1790_179018

theorem files_deleted (initial_files : ℕ) (files_per_folder : ℕ) (num_folders : ℕ) : 
  initial_files = 27 → files_per_folder = 6 → num_folders = 3 →
  initial_files - (files_per_folder * num_folders) = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_files_deleted_l1790_179018


namespace NUMINAMATH_CALUDE_randy_initial_money_l1790_179046

theorem randy_initial_money :
  ∀ (initial_money : ℝ),
  let lunch_cost : ℝ := 10
  let remaining_after_lunch : ℝ := initial_money - lunch_cost
  let ice_cream_cost : ℝ := 5
  let ice_cream_fraction : ℝ := 1/4
  ice_cream_cost = ice_cream_fraction * remaining_after_lunch →
  initial_money = 30 :=
λ initial_money =>
  let lunch_cost : ℝ := 10
  let remaining_after_lunch : ℝ := initial_money - lunch_cost
  let ice_cream_cost : ℝ := 5
  let ice_cream_fraction : ℝ := 1/4
  λ h : ice_cream_cost = ice_cream_fraction * remaining_after_lunch =>
  sorry

#check randy_initial_money

end NUMINAMATH_CALUDE_randy_initial_money_l1790_179046


namespace NUMINAMATH_CALUDE_median_salary_is_25000_l1790_179048

/-- Represents a position in the company with its title, number of employees, and salary. -/
structure Position where
  title : String
  count : Nat
  salary : Nat

/-- Calculates the median salary given a list of positions. -/
def medianSalary (positions : List Position) : Nat :=
  sorry

/-- The list of positions in the company. -/
def companyPositions : List Position := [
  { title := "President", count := 1, salary := 135000 },
  { title := "Vice-President", count := 4, salary := 92000 },
  { title := "Director", count := 15, salary := 78000 },
  { title := "Associate Director", count := 8, salary := 55000 },
  { title := "Administrative Specialist", count := 30, salary := 25000 },
  { title := "Customer Service Representative", count := 12, salary := 20000 }
]

/-- The total number of employees in the company. -/
def totalEmployees : Nat :=
  (companyPositions.map (·.count)).sum

theorem median_salary_is_25000 :
  totalEmployees = 70 ∧ medianSalary companyPositions = 25000 := by
  sorry

end NUMINAMATH_CALUDE_median_salary_is_25000_l1790_179048


namespace NUMINAMATH_CALUDE_employee_count_l1790_179090

theorem employee_count (average_salary : ℕ) (new_average_salary : ℕ) (manager_salary : ℕ) :
  average_salary = 2400 →
  new_average_salary = 2500 →
  manager_salary = 4900 →
  ∃ n : ℕ, n * average_salary + manager_salary = (n + 1) * new_average_salary ∧ n = 24 :=
by sorry

end NUMINAMATH_CALUDE_employee_count_l1790_179090


namespace NUMINAMATH_CALUDE_matrix_paths_count_l1790_179004

/-- Represents a position on the 5x5 grid -/
structure Position :=
  (row : Fin 5)
  (col : Fin 5)

/-- Represents a letter in the word "MATRIX" -/
inductive Letter
| M | A | T | R | I | X

/-- Represents the 5x5 grid of letters -/
def grid : Position → Letter := sorry

/-- Checks if two positions are adjacent (horizontally, vertically, or diagonally) -/
def adjacent (p1 p2 : Position) : Prop := sorry

/-- Represents a valid path spelling "MATRIX" -/
def valid_path (path : List Position) : Prop := sorry

/-- Counts the number of valid paths starting from a given position -/
def count_paths_from (start : Position) : ℕ := sorry

/-- Counts the total number of valid paths in the grid -/
def total_paths : ℕ := sorry

/-- Theorem stating that the total number of paths spelling "MATRIX" is 48 -/
theorem matrix_paths_count :
  total_paths = 48 := by sorry

end NUMINAMATH_CALUDE_matrix_paths_count_l1790_179004


namespace NUMINAMATH_CALUDE_polynomial_identity_l1790_179016

theorem polynomial_identity (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (2*x + Real.sqrt 3)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_l1790_179016


namespace NUMINAMATH_CALUDE_sixty_degrees_to_radians_l1790_179055

theorem sixty_degrees_to_radians :
  let degree_to_radian (d : ℝ) := d * (π / 180)
  degree_to_radian 60 = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_sixty_degrees_to_radians_l1790_179055


namespace NUMINAMATH_CALUDE_space_shuttle_speed_conversion_l1790_179041

-- Define the speed in kilometers per second
def speed_km_per_second : ℝ := 6

-- Define the number of seconds in an hour
def seconds_per_hour : ℝ := 3600

-- Theorem to prove
theorem space_shuttle_speed_conversion :
  speed_km_per_second * seconds_per_hour = 21600 := by
  sorry

end NUMINAMATH_CALUDE_space_shuttle_speed_conversion_l1790_179041


namespace NUMINAMATH_CALUDE_elaines_earnings_increase_l1790_179023

-- Define Elaine's earnings last year
variable (E : ℝ)

-- Define the percentage increase in earnings
variable (P : ℝ)

-- Theorem statement
theorem elaines_earnings_increase :
  -- Last year's rent spending
  (0.20 * E) > 0 →
  -- This year's rent spending is 143.75% of last year's
  (0.25 * (E * (1 + P / 100))) = (1.4375 * (0.20 * E)) →
  -- Conclusion: Earnings increased by 15%
  P = 15 := by
sorry

end NUMINAMATH_CALUDE_elaines_earnings_increase_l1790_179023


namespace NUMINAMATH_CALUDE_faucet_filling_time_l1790_179022

/-- Given that five faucets can fill a 150-gallon tub in 10 minutes,
    prove that ten faucets will fill a 50-gallon tub in 100 seconds. -/
theorem faucet_filling_time 
  (fill_rate : ℝ)  -- Rate at which one faucet fills in gallons per minute
  (h1 : 5 * fill_rate * 10 = 150)  -- Five faucets fill 150 gallons in 10 minutes
  : 10 * fill_rate * (100 / 60) = 50  -- Ten faucets fill 50 gallons in 100 seconds
  := by sorry

end NUMINAMATH_CALUDE_faucet_filling_time_l1790_179022


namespace NUMINAMATH_CALUDE_four_greater_than_sqrt_fourteen_l1790_179001

theorem four_greater_than_sqrt_fourteen : 4 > Real.sqrt 14 := by
  sorry

end NUMINAMATH_CALUDE_four_greater_than_sqrt_fourteen_l1790_179001


namespace NUMINAMATH_CALUDE_polynomial_root_implication_l1790_179060

theorem polynomial_root_implication (a b : ℝ) : 
  (Complex.I : ℂ) ^ 2 = -1 →
  (2 - 3 * Complex.I : ℂ) ^ 3 + a * (2 - 3 * Complex.I : ℂ) ^ 2 + 3 * (2 - 3 * Complex.I : ℂ) + b = 0 →
  a = -3/2 ∧ b = 65/2 := by
sorry

end NUMINAMATH_CALUDE_polynomial_root_implication_l1790_179060


namespace NUMINAMATH_CALUDE_sum_of_first_and_third_l1790_179062

theorem sum_of_first_and_third (A B C : ℚ) : 
  A + B + C = 98 →
  A / B = 2 / 3 →
  B / C = 5 / 8 →
  B = 30 →
  A + C = 68 := by
sorry

end NUMINAMATH_CALUDE_sum_of_first_and_third_l1790_179062


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1790_179008

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ y => y^2 + 7*y + 10 + (y + 2)*(y + 8)
  (f (-2) = 0 ∧ f (-13/2) = 0) ∧
  ∀ y : ℝ, f y = 0 → (y = -2 ∨ y = -13/2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1790_179008


namespace NUMINAMATH_CALUDE_no_valid_n_l1790_179070

theorem no_valid_n : ¬ ∃ (n : ℕ), 
  n > 0 ∧ 
  (100 ≤ n / 4 ∧ n / 4 ≤ 999) ∧ 
  (100 ≤ 4 * n ∧ 4 * n ≤ 999) :=
sorry

end NUMINAMATH_CALUDE_no_valid_n_l1790_179070


namespace NUMINAMATH_CALUDE_tan_315_degrees_l1790_179032

theorem tan_315_degrees : Real.tan (315 * π / 180) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_315_degrees_l1790_179032


namespace NUMINAMATH_CALUDE_min_distance_to_circle_l1790_179049

theorem min_distance_to_circle (x y : ℝ) : 
  x^2 + y^2 = 1 → 
  (∃ (min : ℝ), min = Real.sqrt 5 - 1 ∧ 
    ∀ (u v : ℝ), u^2 + v^2 = 1 → 
      Real.sqrt ((u - 1)^2 + (v - 2)^2) ≥ min) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_to_circle_l1790_179049


namespace NUMINAMATH_CALUDE_divisible_by_two_and_three_implies_divisible_by_six_l1790_179092

theorem divisible_by_two_and_three_implies_divisible_by_six (n : ℕ) :
  (n % 2 = 0 ∧ n % 3 = 0) → n % 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_two_and_three_implies_divisible_by_six_l1790_179092


namespace NUMINAMATH_CALUDE_promotion_difference_l1790_179015

/-- Represents a shoe promotion strategy -/
inductive Promotion
  | A  -- Buy one pair, get second pair half price
  | B  -- Buy one pair, get $15 off second pair

/-- Calculate the total cost of two pairs of shoes under a given promotion -/
def calculateCost (p : Promotion) (price : ℕ) : ℕ :=
  match p with
  | Promotion.A => price + price / 2
  | Promotion.B => price + price - 15

/-- The difference in cost between Promotion B and Promotion A is $5 -/
theorem promotion_difference (shoePrice : ℕ) (h : shoePrice = 40) :
  calculateCost Promotion.B shoePrice - calculateCost Promotion.A shoePrice = 5 := by
  sorry

#eval calculateCost Promotion.B 40 - calculateCost Promotion.A 40

end NUMINAMATH_CALUDE_promotion_difference_l1790_179015


namespace NUMINAMATH_CALUDE_cos_240_degrees_l1790_179017

theorem cos_240_degrees : Real.cos (240 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_240_degrees_l1790_179017


namespace NUMINAMATH_CALUDE_symmetry_condition_l1790_179006

theorem symmetry_condition (p q r s : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) :
  (∀ x y : ℝ, y = (p * x + q) / (r * x + s) → -x = (p * (-y) + q) / (r * (-y) + s)) →
  p - s = 0 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_condition_l1790_179006


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l1790_179065

theorem arithmetic_calculations :
  ((-10) + (-7) - 3 + 2 = -18) ∧
  ((-2)^3 / 4 - (-1)^2023 + |(-6)| * (-1) = -7) ∧
  ((1/3 - 1/4 + 5/6) * (-24) = -22) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l1790_179065


namespace NUMINAMATH_CALUDE_triangle_theorem_l1790_179033

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem about the triangle -/
theorem triangle_theorem (t : Triangle) 
  (h1 : t.b^2 + t.c^2 - t.a^2 = (4 * Real.sqrt 2 / 3) * t.b * t.c)
  (h2 : 3 * t.c / t.a = Real.sqrt 2 * Real.sin t.B / Real.sin t.A)
  (h3 : 1/2 * t.b * t.c * Real.sin t.A = 2 * Real.sqrt 2) :
  Real.sin t.A = 1/3 ∧ 
  t.c = 2 * Real.sqrt 2 ∧ 
  Real.sin (2 * t.C - π/6) = (10 * Real.sqrt 6 - 23) / 54 := by
  sorry


end NUMINAMATH_CALUDE_triangle_theorem_l1790_179033


namespace NUMINAMATH_CALUDE_inequality_and_minimum_value_l1790_179045

theorem inequality_and_minimum_value (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a^2 / b + b^2 / a ≥ a + b) ∧
  (∀ x : ℝ, 0 < x → x < 1 → (1 - x)^2 / x + x^2 / (1 - x) ≥ 1) ∧
  (∃ x : ℝ, 0 < x ∧ x < 1 ∧ (1 - x)^2 / x + x^2 / (1 - x) = 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_minimum_value_l1790_179045


namespace NUMINAMATH_CALUDE_right_rectangular_prism_volume_l1790_179078

theorem right_rectangular_prism_volume
  (side_area front_area bottom_area : ℝ)
  (h_side : side_area = 18)
  (h_front : front_area = 12)
  (h_bottom : bottom_area = 8) :
  ∃ (a b c : ℝ),
    a * b = side_area ∧
    b * c = front_area ∧
    a * c = bottom_area ∧
    a * b * c = 24 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_right_rectangular_prism_volume_l1790_179078


namespace NUMINAMATH_CALUDE_root_product_theorem_l1790_179072

-- Define the polynomial f(x)
def f (x : ℝ) : ℝ := x^4 - x^3 + 2*x^2 - x + 1

-- Define the function g(x)
def g (x : ℝ) : ℝ := x^2 - 3

-- State the theorem
theorem root_product_theorem (x₁ x₂ x₃ x₄ : ℝ) 
  (h₁ : f x₁ = 0) (h₂ : f x₂ = 0) (h₃ : f x₃ = 0) (h₄ : f x₄ = 0) :
  g x₁ * g x₂ * g x₃ * g x₄ = 667 := by
  sorry

end NUMINAMATH_CALUDE_root_product_theorem_l1790_179072


namespace NUMINAMATH_CALUDE_simplified_expression_l1790_179007

theorem simplified_expression (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (2 * x⁻¹ + 3 * y⁻¹)⁻¹ = (x * y) / (2 * y + 3 * x) :=
by sorry

end NUMINAMATH_CALUDE_simplified_expression_l1790_179007


namespace NUMINAMATH_CALUDE_apple_cider_volume_l1790_179061

/-- The volume of apple cider in a cylindrical pot -/
theorem apple_cider_volume (h : Real) (d : Real) (fill_ratio : Real) (cider_ratio : Real) :
  h = 9 →
  d = 4 →
  fill_ratio = 2/3 →
  cider_ratio = 2/7 →
  (fill_ratio * h * π * (d/2)^2) * cider_ratio = 48*π/7 :=
by sorry

end NUMINAMATH_CALUDE_apple_cider_volume_l1790_179061


namespace NUMINAMATH_CALUDE_greatest_3digit_base8_divisible_by_7_l1790_179047

/-- Converts a base 8 number to base 10 --/
def base8To10 (n : Nat) : Nat :=
  (n / 100) * 64 + ((n / 10) % 10) * 8 + (n % 10)

/-- Checks if a number is a 3-digit base 8 number --/
def is3DigitBase8 (n : Nat) : Prop :=
  100 ≤ n ∧ n ≤ 777

theorem greatest_3digit_base8_divisible_by_7 :
  ∀ n : Nat, is3DigitBase8 n → base8To10 n % 7 = 0 → n ≤ 777 :=
sorry

end NUMINAMATH_CALUDE_greatest_3digit_base8_divisible_by_7_l1790_179047


namespace NUMINAMATH_CALUDE_intersection_with_complement_l1790_179026

open Set

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4}

theorem intersection_with_complement : A ∩ (U \ B) = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l1790_179026


namespace NUMINAMATH_CALUDE_matrix_equation_proof_l1790_179077

open Matrix

theorem matrix_equation_proof :
  let M : Matrix (Fin 2) (Fin 2) ℝ := !![2, 4; 1, 2]
  M^3 - 3 • M^2 + 4 • M = !![6, 12; 3, 6] := by sorry

end NUMINAMATH_CALUDE_matrix_equation_proof_l1790_179077


namespace NUMINAMATH_CALUDE_bird_puzzle_solution_l1790_179021

theorem bird_puzzle_solution :
  ∃! (x y z : ℕ),
    x + y + z = 30 ∧
    (x : ℚ) / 3 + (y : ℚ) / 2 + 2 * (z : ℚ) = 30 ∧
    x = 9 ∧ y = 10 ∧ z = 11 := by
  sorry

end NUMINAMATH_CALUDE_bird_puzzle_solution_l1790_179021


namespace NUMINAMATH_CALUDE_same_heads_probability_is_three_sixteenths_l1790_179086

/-- The number of pennies Keiko tosses -/
def keiko_pennies : ℕ := 2

/-- The number of pennies Ephraim tosses -/
def ephraim_pennies : ℕ := 3

/-- The total number of possible outcomes when tossing n pennies -/
def total_outcomes (n : ℕ) : ℕ := 2^n

/-- The number of favorable outcomes where Ephraim gets the same number of heads as Keiko -/
def favorable_outcomes : ℕ := 6

/-- The probability of Ephraim getting the same number of heads as Keiko -/
def same_heads_probability : ℚ :=
  favorable_outcomes / (total_outcomes keiko_pennies * total_outcomes ephraim_pennies)

theorem same_heads_probability_is_three_sixteenths :
  same_heads_probability = 3 / 16 := by sorry

end NUMINAMATH_CALUDE_same_heads_probability_is_three_sixteenths_l1790_179086


namespace NUMINAMATH_CALUDE_sum_of_fraction_and_constant_l1790_179091

theorem sum_of_fraction_and_constant (x : Real) (h : x = 8.0) : 0.75 * x + 2 = 8.0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fraction_and_constant_l1790_179091


namespace NUMINAMATH_CALUDE_cube_equation_solution_l1790_179097

theorem cube_equation_solution :
  ∃! x : ℝ, (x + 3)^3 = (1/27)⁻¹ :=
by
  -- The unique solution is x = 0
  use 0
  constructor
  · -- Prove that x = 0 satisfies the equation
    sorry
  · -- Prove that this is the only solution
    sorry

end NUMINAMATH_CALUDE_cube_equation_solution_l1790_179097


namespace NUMINAMATH_CALUDE_janous_inequality_l1790_179059

theorem janous_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x * y = 4) :
  1 / (x + 3) + 1 / (y + 3) ≤ 2 / 5 ∧
  (1 / (x + 3) + 1 / (y + 3) = 2 / 5 ↔ x = 2 ∧ y = 2) :=
by sorry

end NUMINAMATH_CALUDE_janous_inequality_l1790_179059


namespace NUMINAMATH_CALUDE_journey_distance_l1790_179035

theorem journey_distance (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ) 
  (h1 : total_time = 10)
  (h2 : speed1 = 21)
  (h3 : speed2 = 24) :
  ∃ (distance : ℝ), 
    distance = total_time * (speed1 + speed2) / 2 ∧ 
    distance = 224 := by
  sorry

end NUMINAMATH_CALUDE_journey_distance_l1790_179035


namespace NUMINAMATH_CALUDE_square_circle_union_area_l1790_179095

theorem square_circle_union_area : 
  let square_side : ℝ := 12
  let circle_radius : ℝ := 12
  let square_area : ℝ := square_side ^ 2
  let circle_area : ℝ := π * circle_radius ^ 2
  let overlap_area : ℝ := (1 / 4) * circle_area
  square_area + circle_area - overlap_area = 144 + 108 * π := by
  sorry

end NUMINAMATH_CALUDE_square_circle_union_area_l1790_179095


namespace NUMINAMATH_CALUDE_charm_bracelet_profit_l1790_179075

/-- Calculates the profit from selling charm bracelets -/
theorem charm_bracelet_profit
  (string_cost : ℕ)
  (bead_cost : ℕ)
  (selling_price : ℕ)
  (bracelets_sold : ℕ)
  (h1 : string_cost = 1)
  (h2 : bead_cost = 3)
  (h3 : selling_price = 6)
  (h4 : bracelets_sold = 25) :
  (selling_price * bracelets_sold) - ((string_cost + bead_cost) * bracelets_sold) = 50 :=
by sorry

end NUMINAMATH_CALUDE_charm_bracelet_profit_l1790_179075


namespace NUMINAMATH_CALUDE_significant_figures_of_number_l1790_179009

/-- Count the number of significant figures in a rational number represented as a string -/
def count_significant_figures (s : String) : ℕ := sorry

/-- The rational number in question -/
def number : String := "0.0050400"

/-- Theorem stating that the number of significant figures in 0.0050400 is 5 -/
theorem significant_figures_of_number : count_significant_figures number = 5 := by sorry

end NUMINAMATH_CALUDE_significant_figures_of_number_l1790_179009


namespace NUMINAMATH_CALUDE_find_b_l1790_179042

theorem find_b (a b c : ℚ) 
  (sum_eq : a + b + c = 150)
  (equal_after_changes : a + 10 = b - 10 ∧ b - 10 = 3 * c) : 
  b = 520 / 7 := by
sorry

end NUMINAMATH_CALUDE_find_b_l1790_179042


namespace NUMINAMATH_CALUDE_perfect_square_fraction_l1790_179071

theorem perfect_square_fraction (n : ℤ) : 
  n > 2020 → 
  (∃ m : ℤ, (n - 2020) / (2120 - n) = m^2) → 
  n = 2070 ∨ n = 2100 ∨ n = 2110 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_fraction_l1790_179071


namespace NUMINAMATH_CALUDE_parabola_equation_l1790_179014

/-- A parabola with vertex at the origin and focus on the y-axis. -/
structure Parabola where
  p : ℝ  -- The focal parameter of the parabola

/-- The line y = 2x + 1 -/
def line (x : ℝ) : ℝ := 2 * x + 1

/-- The chord length intercepted by the line y = 2x + 1 on the parabola -/
def chordLength (p : Parabola) : ℝ := sorry

theorem parabola_equation (p : Parabola) :
  chordLength p = Real.sqrt 15 →
  (∀ x y : ℝ, y = p.p * x^2 ∨ y = -3 * p.p * x^2) :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l1790_179014


namespace NUMINAMATH_CALUDE_unique_xxyy_square_l1790_179080

def is_xxyy_form (n : ℕ) : Prop :=
  ∃ (x y : ℕ), x < 10 ∧ y < 10 ∧ n = 1000 * x + 100 * x + 10 * y + y

theorem unique_xxyy_square : 
  ∀ n : ℕ, is_xxyy_form n ∧ ∃ m : ℕ, n = m^2 → n = 7744 :=
sorry

end NUMINAMATH_CALUDE_unique_xxyy_square_l1790_179080


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l1790_179074

/-- Given a geometric sequence {aₙ}, prove that if a₃ = 16 and a₄ = 8, then a₁ = 64. -/
theorem geometric_sequence_first_term (a : ℕ → ℝ) :
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r) →  -- Definition of geometric sequence
  a 3 = 16 →                                -- Condition: a₃ = 16
  a 4 = 8 →                                 -- Condition: a₄ = 8
  a 1 = 64 :=                               -- Conclusion: a₁ = 64
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l1790_179074


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1790_179027

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 2*x + 4 ≤ 0) ↔ (∃ x₀ : ℝ, x₀^2 - 2*x₀ + 4 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1790_179027


namespace NUMINAMATH_CALUDE_april_rainfall_calculation_l1790_179057

/-- The amount of rainfall in March, in inches -/
def march_rainfall : ℝ := 0.81

/-- The difference in rainfall between March and April, in inches -/
def rainfall_difference : ℝ := 0.35

/-- The amount of rainfall in April, in inches -/
def april_rainfall : ℝ := march_rainfall - rainfall_difference

theorem april_rainfall_calculation :
  april_rainfall = 0.46 := by sorry

end NUMINAMATH_CALUDE_april_rainfall_calculation_l1790_179057


namespace NUMINAMATH_CALUDE_max_rectangle_area_l1790_179058

/-- The maximum area of a rectangle with perimeter 156 feet and natural number sides --/
theorem max_rectangle_area (l w : ℕ) : 
  (2 * (l + w) = 156) → l * w ≤ 1521 := by
  sorry

end NUMINAMATH_CALUDE_max_rectangle_area_l1790_179058


namespace NUMINAMATH_CALUDE_ellipse_chords_and_bisector_l1790_179044

/-- Given an ellipse x²/2 + y² = 1, this theorem proves:
    1. The trajectory of midpoint of parallel chords with slope 2
    2. The trajectory of midpoint of chord defined by line passing through A(2,1)
    3. The line passing through P(1/2, 1/2) and bisected by P -/
theorem ellipse_chords_and_bisector 
  (x y : ℝ) (h : x^2/2 + y^2 = 1) :
  (∃ t : ℝ, x + 4*y = t) ∧ 
  (∃ s : ℝ, x^2 + 2*y^2 - 2*x - 2*y = s) ∧
  (2*x + 4*y - 3 = 0 → 
    ∃ (x₁ y₁ x₂ y₂ : ℝ), 
      x₁^2/2 + y₁^2 = 1 ∧ 
      x₂^2/2 + y₂^2 = 1 ∧ 
      (x₁ + x₂)/2 = 1/2 ∧ 
      (y₁ + y₂)/2 = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_chords_and_bisector_l1790_179044


namespace NUMINAMATH_CALUDE_count_happy_license_plates_l1790_179064

/-- The set of allowed letters on the license plate -/
def allowed_letters : Finset Char := {'А', 'В', 'Е', 'К', 'М', 'Н', 'О', 'Р', 'С', 'Т', 'У', 'Х'}

/-- The set of consonant letters from the allowed letters -/
def consonant_letters : Finset Char := {'В', 'К', 'М', 'Н', 'Р', 'С', 'Т', 'Х'}

/-- The set of odd digits -/
def odd_digits : Finset Nat := {1, 3, 5, 7, 9}

/-- A license plate is a tuple of 3 letters and 3 digits -/
structure LicensePlate :=
  (letter1 : Char)
  (letter2 : Char)
  (digit1 : Nat)
  (digit2 : Nat)
  (digit3 : Nat)
  (letter3 : Char)

/-- A license plate is happy if the first two letters are consonants and the third digit is odd -/
def is_happy (plate : LicensePlate) : Prop :=
  plate.letter1 ∈ consonant_letters ∧
  plate.letter2 ∈ consonant_letters ∧
  plate.digit3 ∈ odd_digits

/-- The set of all valid license plates -/
def all_license_plates : Finset LicensePlate :=
  sorry

/-- The set of all happy license plates -/
def happy_license_plates : Finset LicensePlate :=
  sorry

/-- The main theorem: there are 384000 happy license plates -/
theorem count_happy_license_plates :
  Finset.card happy_license_plates = 384000 :=
sorry

end NUMINAMATH_CALUDE_count_happy_license_plates_l1790_179064


namespace NUMINAMATH_CALUDE_f_is_quadratic_l1790_179085

/-- Definition of a quadratic equation in x -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing x² - 3x -/
def f (x : ℝ) : ℝ := x^2 - 3*x

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f :=
sorry

end NUMINAMATH_CALUDE_f_is_quadratic_l1790_179085


namespace NUMINAMATH_CALUDE_triangle_problem_l1790_179079

/-- Given a triangle ABC with sides a, b, c and angles A, B, C, 
    prove the measure of angle A and the area of the triangle 
    under specific conditions. -/
theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  (b^2 + c^2 = a^2 + Real.sqrt 3 * b * c) →  -- Given condition
  (0 < A ∧ A < π) →                          -- Angle A is in (0, π)
  (0 < B ∧ B < π) →                          -- Angle B is in (0, π)
  (0 < C ∧ C < π) →                          -- Angle C is in (0, π)
  (A + B + C = π) →                          -- Sum of angles in a triangle
  (a * Real.sin B = b * Real.sin A) →        -- Law of sines
  (a^2 = b^2 + c^2 - 2 * b * c * Real.cos A) → -- Law of cosines
  (A = π / 6) ∧                              -- First part of the theorem
  ((Real.cos B = 2 * Real.sqrt 2 / 3 ∧ a = Real.sqrt 2) →
   (1 / 2 * a * b * Real.sin C = (2 * Real.sqrt 2 + Real.sqrt 3) / 9)) -- Second part
  := by sorry

end NUMINAMATH_CALUDE_triangle_problem_l1790_179079


namespace NUMINAMATH_CALUDE_tangent_slope_implies_a_l1790_179000

-- Define the curve
def f (a : ℝ) (x : ℝ) : ℝ := x^4 + a*x^2 + 1

-- Define the derivative of the curve
def f' (a : ℝ) (x : ℝ) : ℝ := 4*x^3 + 2*a*x

theorem tangent_slope_implies_a (a : ℝ) :
  f a (-1) = a + 2 →  -- The curve passes through the point (-1, a+2)
  f' a (-1) = 8 →     -- The slope of the tangent line at x = -1 is 8
  a = -6 :=           -- Then a must equal -6
by
  sorry


end NUMINAMATH_CALUDE_tangent_slope_implies_a_l1790_179000


namespace NUMINAMATH_CALUDE_train_A_time_l1790_179068

/-- Represents the properties of a train journey --/
structure TrainJourney where
  speed : ℝ
  time : ℝ
  distance : ℝ

/-- The problem setup --/
def trainProblem (routeLength : ℝ) (meetingPoint : ℝ) (trainBTime : ℝ) : Prop :=
  ∃ (trainA trainB : TrainJourney),
    -- Total route length
    routeLength = 75 ∧
    -- Train B's journey
    trainB.distance = routeLength ∧
    trainB.time = trainBTime ∧
    trainB.speed = trainB.distance / trainB.time ∧
    -- Train A's journey
    trainA.distance = routeLength ∧
    -- Meeting point
    meetingPoint = 30 ∧
    -- Trains meet at the same time
    meetingPoint / trainA.speed = (routeLength - meetingPoint) / trainB.speed ∧
    -- Train A's time is the total distance divided by its speed
    trainA.time = trainA.distance / trainA.speed

/-- The theorem to prove --/
theorem train_A_time : 
  ∀ (routeLength meetingPoint trainBTime : ℝ),
    trainProblem routeLength meetingPoint trainBTime →
    ∃ (trainA : TrainJourney), trainA.time = 3 := by
  sorry

end NUMINAMATH_CALUDE_train_A_time_l1790_179068


namespace NUMINAMATH_CALUDE_sams_age_l1790_179034

theorem sams_age (sam drew alex jordan : ℕ) : 
  sam + drew + alex + jordan = 142 →
  sam = drew / 2 →
  alex = sam + 3 →
  jordan = 2 * alex →
  sam = 22 :=
by
  sorry

end NUMINAMATH_CALUDE_sams_age_l1790_179034


namespace NUMINAMATH_CALUDE_largest_power_divides_product_l1790_179030

/-- The largest power of the largest prime that divides n -/
def pow (n : ℕ) : ℕ :=
  sorry

/-- The product of pow(n) for n from 2 to 7200 -/
def product_pow : ℕ :=
  sorry

/-- 2020 raised to the power of the result -/
def power_2020 (m : ℕ) : ℕ :=
  2020^m

theorem largest_power_divides_product :
  ∃ m : ℕ, m = 72 ∧
  power_2020 m ∣ product_pow ∧
  ∀ k > m, ¬(power_2020 k ∣ product_pow) :=
sorry

end NUMINAMATH_CALUDE_largest_power_divides_product_l1790_179030


namespace NUMINAMATH_CALUDE_simplify_radicals_l1790_179036

theorem simplify_radicals (y z : ℝ) (h : y ≥ 0 ∧ z ≥ 0) : 
  Real.sqrt (32 * y) * Real.sqrt (75 * z) * Real.sqrt (14 * y) = 40 * y * Real.sqrt (21 * z) := by
  sorry

end NUMINAMATH_CALUDE_simplify_radicals_l1790_179036


namespace NUMINAMATH_CALUDE_line_through_points_l1790_179052

theorem line_through_points (a n : ℝ) :
  (∀ x y, x = 3 * y + 5 → 
    ((x = a ∧ y = n) ∨ (x = a + 2 ∧ y = n + 2/3))) →
  a = 3 * n + 5 :=
by sorry

end NUMINAMATH_CALUDE_line_through_points_l1790_179052


namespace NUMINAMATH_CALUDE_expression_equals_73_l1790_179020

def x : ℤ := 2
def y : ℤ := -3
def z : ℤ := 6

theorem expression_equals_73 : x^2 + y^2 + z^2 + 2*x*y - 2*y*z = 73 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_73_l1790_179020


namespace NUMINAMATH_CALUDE_sqrt_expression_value_l1790_179083

theorem sqrt_expression_value : 
  (Real.sqrt 1.21) / (Real.sqrt 0.81) + (Real.sqrt 0.81) / (Real.sqrt 0.49) = 158 / 63 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_value_l1790_179083


namespace NUMINAMATH_CALUDE_mistaken_divisor_l1790_179019

theorem mistaken_divisor (dividend : ℕ) (correct_divisor mistaken_divisor : ℕ) :
  correct_divisor = 21 →
  dividend = correct_divisor * 20 →
  dividend = mistaken_divisor * 35 →
  mistaken_divisor = 12 := by
sorry

end NUMINAMATH_CALUDE_mistaken_divisor_l1790_179019
