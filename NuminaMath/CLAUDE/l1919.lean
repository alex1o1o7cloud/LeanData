import Mathlib

namespace no_real_solution_quadratic_l1919_191943

theorem no_real_solution_quadratic : ¬ ∃ x : ℝ, x^2 + 3*x + 3 ≤ 0 := by sorry

end no_real_solution_quadratic_l1919_191943


namespace missing_number_is_1255_l1919_191912

def given_numbers : List ℕ := [744, 745, 747, 748, 749, 752, 752, 753, 755]
def total_count : ℕ := 10
def average : ℕ := 750

theorem missing_number_is_1255 :
  let sum_given := given_numbers.sum
  let total_sum := total_count * average
  total_sum - sum_given = 1255 := by sorry

end missing_number_is_1255_l1919_191912


namespace certain_number_value_l1919_191950

def is_smallest_multiplier (n : ℕ) (x : ℝ) : Prop :=
  n > 0 ∧ ∃ (y : ℕ), n * x = y^2 ∧
  ∀ (m : ℕ), m > 0 → m < n → ¬∃ (z : ℕ), m * x = z^2

theorem certain_number_value (n : ℕ) (x : ℝ) :
  is_smallest_multiplier n x → n = 3 → x = 1 :=
by sorry

end certain_number_value_l1919_191950


namespace median_in_60_64_interval_l1919_191979

/-- Represents the score intervals --/
inductive ScoreInterval
| interval_45_49
| interval_50_54
| interval_55_59
| interval_60_64
| interval_65_69

/-- Represents the frequency of each score interval --/
def frequency : ScoreInterval → Nat
| ScoreInterval.interval_45_49 => 10
| ScoreInterval.interval_50_54 => 15
| ScoreInterval.interval_55_59 => 20
| ScoreInterval.interval_60_64 => 25
| ScoreInterval.interval_65_69 => 30

/-- The total number of students --/
def totalStudents : Nat := 100

/-- Function to calculate the cumulative frequency up to a given interval --/
def cumulativeFrequency (interval : ScoreInterval) : Nat :=
  match interval with
  | ScoreInterval.interval_45_49 => frequency ScoreInterval.interval_45_49
  | ScoreInterval.interval_50_54 => frequency ScoreInterval.interval_45_49 + frequency ScoreInterval.interval_50_54
  | ScoreInterval.interval_55_59 => frequency ScoreInterval.interval_45_49 + frequency ScoreInterval.interval_50_54 + frequency ScoreInterval.interval_55_59
  | ScoreInterval.interval_60_64 => frequency ScoreInterval.interval_45_49 + frequency ScoreInterval.interval_50_54 + frequency ScoreInterval.interval_55_59 + frequency ScoreInterval.interval_60_64
  | ScoreInterval.interval_65_69 => totalStudents

/-- The median position --/
def medianPosition : Nat := (totalStudents + 1) / 2

/-- Theorem stating that the median score falls in the interval 60-64 --/
theorem median_in_60_64_interval :
  cumulativeFrequency ScoreInterval.interval_55_59 < medianPosition ∧
  medianPosition ≤ cumulativeFrequency ScoreInterval.interval_60_64 :=
sorry

end median_in_60_64_interval_l1919_191979


namespace length_breadth_difference_l1919_191999

/-- Represents a rectangular plot with given dimensions and fencing cost. -/
structure RectangularPlot where
  length : ℝ
  breadth : ℝ
  fencing_cost_per_meter : ℝ
  total_fencing_cost : ℝ

/-- Theorem stating the difference between length and breadth of the plot. -/
theorem length_breadth_difference (plot : RectangularPlot)
  (h1 : plot.length = 75)
  (h2 : plot.fencing_cost_per_meter = 26.5)
  (h3 : plot.total_fencing_cost = 5300)
  (h4 : plot.total_fencing_cost = (2 * plot.length + 2 * plot.breadth) * plot.fencing_cost_per_meter) :
  plot.length - plot.breadth = 50 := by
  sorry

#check length_breadth_difference

end length_breadth_difference_l1919_191999


namespace perfect_square_sum_l1919_191955

theorem perfect_square_sum (x y : ℕ) : 
  (∃ z : ℕ, 3^x + 7^y = z^2) → 
  Even y → 
  x = 1 ∧ y = 0 := by
sorry

end perfect_square_sum_l1919_191955


namespace inverse_as_linear_combination_l1919_191944

theorem inverse_as_linear_combination (N : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : N = ![![3, 0], ![2, -4]]) : 
  ∃ (c d : ℝ), N⁻¹ = c • N + d • (1 : Matrix (Fin 2) (Fin 2) ℝ) ∧ 
  c = (1 : ℝ) / 12 ∧ d = (1 : ℝ) / 12 := by
sorry

end inverse_as_linear_combination_l1919_191944


namespace power_windows_count_l1919_191949

theorem power_windows_count (total : ℕ) (power_steering : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : total = 65)
  (h2 : power_steering = 45)
  (h3 : both = 17)
  (h4 : neither = 12) :
  total - neither - (power_steering - both) = 25 := by
  sorry

end power_windows_count_l1919_191949


namespace one_is_monomial_l1919_191976

/-- Definition of a monomial as an algebraic expression with one term -/
def isMonomial (expr : ℕ → ℚ) : Prop :=
  ∃ (c : ℚ) (n : ℕ), ∀ (k : ℕ), expr k = if k = n then c else 0

/-- Theorem: 1 is a monomial -/
theorem one_is_monomial : isMonomial (fun _ ↦ 1) := by
  sorry

end one_is_monomial_l1919_191976


namespace point_in_first_quadrant_l1919_191926

theorem point_in_first_quadrant (x y : ℝ) (h1 : x + y = 2) (h2 : x - y = 1) : x > 0 ∧ y > 0 := by
  sorry

end point_in_first_quadrant_l1919_191926


namespace sum_of_divisors_420_l1919_191937

def sum_of_divisors (n : ℕ) : ℕ := (Finset.filter (·∣n) (Finset.range (n + 1))).sum id

theorem sum_of_divisors_420 : sum_of_divisors 420 = 1344 := by
  sorry

end sum_of_divisors_420_l1919_191937


namespace semicircle_problem_l1919_191925

theorem semicircle_problem (N : ℕ) (r : ℝ) (h_positive : r > 0) : 
  let A := N * (π * r^2 / 2)
  let B := (π * (N*r)^2 / 2) - A
  A / B = 1 / 9 → N = 10 := by
sorry

end semicircle_problem_l1919_191925


namespace shed_area_calculation_l1919_191929

/-- The total inside surface area of a rectangular shed -/
def shedSurfaceArea (width length height : ℝ) : ℝ :=
  2 * (width * height + length * height + width * length)

theorem shed_area_calculation :
  let width : ℝ := 12
  let length : ℝ := 15
  let height : ℝ := 7
  shedSurfaceArea width length height = 738 := by
  sorry

end shed_area_calculation_l1919_191929


namespace large_number_with_specific_divisors_l1919_191985

/-- A function that returns the list of divisors of a natural number -/
def divisors (n : ℕ) : List ℕ := sorry

/-- A predicate that checks if a list of natural numbers has alternating parity -/
def hasAlternatingParity (l : List ℕ) : Prop := sorry

theorem large_number_with_specific_divisors (n : ℕ) 
  (h1 : (divisors n).length = 1000)
  (h2 : hasAlternatingParity (divisors n)) :
  n > 10^150 := by sorry

end large_number_with_specific_divisors_l1919_191985


namespace least_common_period_l1919_191990

-- Define the property of the function
def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 5) + f (x - 5) = f x

-- Define what it means for a function to be periodic with period p
def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x : ℝ, f (x + p) = f x

-- The main theorem
theorem least_common_period :
  ∃ p : ℝ, p > 0 ∧
  (∀ f : ℝ → ℝ, satisfies_condition f → is_periodic f p) ∧
  (∀ q : ℝ, 0 < q ∧ q < p →
    ∃ g : ℝ → ℝ, satisfies_condition g ∧ ¬is_periodic g q) ∧
  p = 30 :=
sorry

end least_common_period_l1919_191990


namespace payment_difference_l1919_191939

def original_price : Float := 42.00000000000004
def discount_rate : Float := 0.10
def tip_rate : Float := 0.15

def discounted_price : Float := original_price * (1 - discount_rate)

def john_payment : Float := original_price + (original_price * tip_rate)
def jane_payment : Float := discounted_price + (discounted_price * tip_rate)

theorem payment_difference : john_payment - jane_payment = 4.830000000000005 := by
  sorry

end payment_difference_l1919_191939


namespace min_value_trig_expression_min_value_trig_expression_achievable_l1919_191933

theorem min_value_trig_expression (α β : ℝ) :
  (2 * Real.cos α + 5 * Real.sin β - 8)^2 + (2 * Real.sin α + 5 * Real.cos β - 15)^2 ≥ 100 :=
by sorry

theorem min_value_trig_expression_achievable :
  ∃ α β : ℝ, (2 * Real.cos α + 5 * Real.sin β - 8)^2 + (2 * Real.sin α + 5 * Real.cos β - 15)^2 = 100 :=
by sorry

end min_value_trig_expression_min_value_trig_expression_achievable_l1919_191933


namespace fuel_price_per_gallon_l1919_191958

/-- Given the following conditions:
    1. The total amount of fuel is 100 gallons
    2. The fuel consumption rate is $0.40 worth of fuel per hour
    3. It takes 175 hours to consume all the fuel
    Prove that the price per gallon of fuel is $0.70 -/
theorem fuel_price_per_gallon 
  (total_fuel : ℝ) 
  (consumption_rate : ℝ) 
  (total_hours : ℝ) 
  (h1 : total_fuel = 100) 
  (h2 : consumption_rate = 0.40) 
  (h3 : total_hours = 175) : 
  (consumption_rate * total_hours) / total_fuel = 0.70 := by
  sorry

#check fuel_price_per_gallon

end fuel_price_per_gallon_l1919_191958


namespace partial_fraction_decomposition_l1919_191934

theorem partial_fraction_decomposition (x : ℝ) (h1 : x ≠ 7) (h2 : x ≠ -2) :
  (3 * x + 12) / (x^2 - 5*x - 14) = (11/3) / (x - 7) + (-2/3) / (x + 2) := by
  sorry

end partial_fraction_decomposition_l1919_191934


namespace cubic_root_sum_l1919_191973

theorem cubic_root_sum (p q r : ℝ) : 
  p^3 - 2*p - 2 = 0 → 
  q^3 - 2*q - 2 = 0 → 
  r^3 - 2*r - 2 = 0 → 
  p*(q - r)^2 + q*(r - p)^2 + r*(p - q)^2 = -18 := by
sorry

end cubic_root_sum_l1919_191973


namespace zephyria_license_plates_l1919_191967

/-- The number of letters in the English alphabet -/
def num_letters : ℕ := 26

/-- The number of possible digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of letters in a Zephyrian license plate -/
def num_plate_letters : ℕ := 3

/-- The number of digits in a Zephyrian license plate -/
def num_plate_digits : ℕ := 4

/-- The total number of possible valid license plates in Zephyria -/
def total_license_plates : ℕ := num_letters ^ num_plate_letters * num_digits ^ num_plate_digits

theorem zephyria_license_plates :
  total_license_plates = 175760000 :=
by sorry

end zephyria_license_plates_l1919_191967


namespace product_trailing_zeros_l1919_191905

def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125) + (n / 625)

theorem product_trailing_zeros :
  trailingZeros 2014 = 501 := by
  sorry

end product_trailing_zeros_l1919_191905


namespace sum_of_squares_of_roots_l1919_191972

theorem sum_of_squares_of_roots (p q r : ℝ) : 
  (3 * p^3 - 2 * p^2 + 6 * p + 15 = 0) →
  (3 * q^3 - 2 * q^2 + 6 * q + 15 = 0) →
  (3 * r^3 - 2 * r^2 + 6 * r + 15 = 0) →
  p^2 + q^2 + r^2 = -32/9 := by
  sorry

end sum_of_squares_of_roots_l1919_191972


namespace pizza_problem_l1919_191981

theorem pizza_problem (total_money : ℕ) (pizza_cost : ℕ) (bill_initial : ℕ) (bill_final : ℕ) :
  total_money = 42 →
  pizza_cost = 11 →
  bill_initial = 30 →
  bill_final = 39 →
  (total_money - (bill_final - bill_initial)) / pizza_cost = 3 :=
by
  sorry

end pizza_problem_l1919_191981


namespace ab_value_l1919_191948

theorem ab_value (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 11) : a * b = 1 := by
  sorry

end ab_value_l1919_191948


namespace inequality_implies_upper_bound_l1919_191908

theorem inequality_implies_upper_bound (m : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Ioo 0 1 → x^2 - 4*x ≥ m) → m ≤ -3 := by
sorry

end inequality_implies_upper_bound_l1919_191908


namespace thirty_percent_less_than_ninety_l1919_191927

theorem thirty_percent_less_than_ninety (x : ℝ) : x = 42 → x + x/2 = 63 := by
  sorry

end thirty_percent_less_than_ninety_l1919_191927


namespace tan_double_angle_l1919_191904

theorem tan_double_angle (α : Real) 
  (h : (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 1/2) : 
  Real.tan (2 * α) = 3/4 := by sorry

end tan_double_angle_l1919_191904


namespace reflection_distance_C_l1919_191909

/-- The length of the segment from a point to its reflection over the x-axis --/
def reflection_distance (p : ℝ × ℝ) : ℝ :=
  2 * |p.2|

theorem reflection_distance_C : reflection_distance (-3, 2) = 4 := by
  sorry

end reflection_distance_C_l1919_191909


namespace percent_y_of_x_l1919_191959

theorem percent_y_of_x (x y : ℝ) (h : (1/2) * (x - y) = (1/5) * (x + y)) :
  y = (3/7) * x := by
  sorry

end percent_y_of_x_l1919_191959


namespace lawn_mowing_payment_l1919_191914

theorem lawn_mowing_payment (payment_rate : ℚ) (lawns_mowed : ℚ) : 
  payment_rate = 13/3 → lawns_mowed = 11/4 → payment_rate * lawns_mowed = 143/12 := by
  sorry

end lawn_mowing_payment_l1919_191914


namespace shopkeeper_payment_l1919_191974

/-- The total cost of a purchase given the quantity and price per unit -/
def totalCost (quantity : ℕ) (pricePerUnit : ℕ) : ℕ :=
  quantity * pricePerUnit

/-- The problem statement -/
theorem shopkeeper_payment : 
  let grapeQuantity : ℕ := 8
  let grapePrice : ℕ := 70
  let mangoQuantity : ℕ := 9
  let mangoPrice : ℕ := 50
  let grapeCost := totalCost grapeQuantity grapePrice
  let mangoCost := totalCost mangoQuantity mangoPrice
  grapeCost + mangoCost = 1010 := by
  sorry

end shopkeeper_payment_l1919_191974


namespace candy_box_prices_l1919_191922

theorem candy_box_prices (total_money : ℕ) (metal_boxes : ℕ) (paper_boxes : ℕ) :
  -- A buys 4 fewer boxes than B
  metal_boxes + 4 = paper_boxes →
  -- A has 6 yuan left
  ∃ (metal_price : ℕ), metal_price * metal_boxes + 6 = total_money →
  -- B uses all the money
  ∃ (paper_price : ℕ), paper_price * paper_boxes = total_money →
  -- If A used three times his original amount of money
  ∃ (new_metal_boxes : ℕ), 
    -- He would buy 31 more boxes than B
    new_metal_boxes = paper_boxes + 31 →
    -- And still have 6 yuan left
    metal_price * new_metal_boxes + 6 = 3 * total_money →
  -- Then the price of metal boxes is 12 yuan
  metal_price = 12 ∧
  -- And the price of paper boxes is 10 yuan
  paper_price = 10 := by
  sorry

end candy_box_prices_l1919_191922


namespace rain_on_tuesdays_l1919_191946

theorem rain_on_tuesdays 
  (monday_count : ℕ) 
  (tuesday_count : ℕ) 
  (rain_per_monday : ℝ) 
  (total_rain_difference : ℝ) 
  (h1 : monday_count = 7)
  (h2 : tuesday_count = 9)
  (h3 : rain_per_monday = 1.5)
  (h4 : total_rain_difference = 12) :
  (monday_count * rain_per_monday + total_rain_difference) / tuesday_count = 2.5 := by
sorry

end rain_on_tuesdays_l1919_191946


namespace equation_solution_l1919_191995

theorem equation_solution : 
  ∃ x : ℝ, (x + 6) / (x - 3) = 4 ∧ x = 6 := by sorry

end equation_solution_l1919_191995


namespace complex_magnitude_equation_l1919_191996

theorem complex_magnitude_equation (a : ℝ) : 
  Complex.abs ((1 + Complex.I) / (a * Complex.I)) = Real.sqrt 2 → a = 1 ∨ a = -1 := by
  sorry

end complex_magnitude_equation_l1919_191996


namespace parallel_to_same_plane_are_parallel_perpendicular_to_same_line_are_parallel_l1919_191900

-- Define the basic types
variable (P : Type) -- Type for planes
variable (L : Type) -- Type for lines

-- Define the relationships
variable (parallel : P → P → Prop) -- Parallel planes
variable (perpendicular : P → L → Prop) -- Plane perpendicular to a line

-- Axioms
axiom parallel_trans (p q r : P) : parallel p q → parallel q r → parallel p r
axiom parallel_symm (p q : P) : parallel p q → parallel q p

-- Theorem 1: Two different planes that are parallel to the same plane are parallel to each other
theorem parallel_to_same_plane_are_parallel (p q r : P) 
  (hp : parallel p r) (hq : parallel q r) (hne : p ≠ q) : 
  parallel p q :=
sorry

-- Theorem 2: Two different planes that are perpendicular to the same line are parallel to each other
theorem perpendicular_to_same_line_are_parallel (p q : P) (l : L)
  (hp : perpendicular p l) (hq : perpendicular q l) (hne : p ≠ q) :
  parallel p q :=
sorry

end parallel_to_same_plane_are_parallel_perpendicular_to_same_line_are_parallel_l1919_191900


namespace sin_2α_minus_π_over_6_l1919_191906

theorem sin_2α_minus_π_over_6 (α : Real) 
  (h : Real.cos (α + 2 * Real.pi / 3) = 3 / 5) : 
  Real.sin (2 * α - Real.pi / 6) = -7 / 25 := by
  sorry

end sin_2α_minus_π_over_6_l1919_191906


namespace factorization_problems_l1919_191965

theorem factorization_problems (m a x : ℝ) : 
  (9 * m^2 - 4 = (3 * m + 2) * (3 * m - 2)) ∧ 
  (2 * a * x^2 + 12 * a * x + 18 * a = 2 * a * (x + 3)^2) := by
  sorry

end factorization_problems_l1919_191965


namespace fifty_factorial_trailing_zeros_l1919_191993

/-- The number of trailing zeros in n! -/
def trailing_zeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25)

/-- 50! has exactly 12 trailing zeros -/
theorem fifty_factorial_trailing_zeros : trailing_zeros 50 = 12 := by
  sorry

end fifty_factorial_trailing_zeros_l1919_191993


namespace hunters_playing_time_l1919_191917

/-- Given Hunter's playing times for football and basketball, prove the total time played in hours. -/
theorem hunters_playing_time (football_minutes basketball_minutes : ℕ) 
  (h1 : football_minutes = 60) 
  (h2 : basketball_minutes = 30) : 
  (football_minutes + basketball_minutes : ℚ) / 60 = 1.5 := by
  sorry

#check hunters_playing_time

end hunters_playing_time_l1919_191917


namespace y_in_terms_of_x_l1919_191987

theorem y_in_terms_of_x (p : ℝ) (x y : ℝ) 
  (hx : x = 1 + 3^p) 
  (hy : y = 1 + 3^(-p)) : 
  y = x / (x - 1) := by
  sorry

end y_in_terms_of_x_l1919_191987


namespace remainder_415_420_mod_16_l1919_191901

theorem remainder_415_420_mod_16 : 415^420 ≡ 1 [MOD 16] := by
  sorry

end remainder_415_420_mod_16_l1919_191901


namespace bob_fruit_drink_cost_l1919_191932

/-- The cost of Bob's fruit drink -/
def fruit_drink_cost (andy_total bob_sandwich_cost : ℕ) : ℕ :=
  andy_total - bob_sandwich_cost

theorem bob_fruit_drink_cost :
  let andy_total := 5
  let bob_sandwich_cost := 3
  fruit_drink_cost andy_total bob_sandwich_cost = 2 := by
  sorry

end bob_fruit_drink_cost_l1919_191932


namespace julie_work_hours_l1919_191923

/-- Calculates the number of hours Julie needs to work per week during the school year -/
def school_year_hours (summer_weeks : ℕ) (summer_hours_per_week : ℕ) (summer_earnings : ℚ) 
                      (school_year_weeks : ℕ) (school_year_earnings : ℚ) : ℚ :=
  let hourly_wage := summer_earnings / (summer_weeks * summer_hours_per_week)
  let weekly_earnings := school_year_earnings / school_year_weeks
  weekly_earnings / hourly_wage

theorem julie_work_hours :
  school_year_hours 10 60 7500 50 7500 = 12 := by
  sorry

end julie_work_hours_l1919_191923


namespace sum_of_factors_180_l1919_191992

/-- The sum of positive factors of a natural number n -/
def sum_of_factors (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of positive factors of 180 is 546 -/
theorem sum_of_factors_180 : sum_of_factors 180 = 546 := by sorry

end sum_of_factors_180_l1919_191992


namespace percentage_problem_l1919_191941

theorem percentage_problem (N : ℝ) (h : 0.2 * N = 1000) : 1.2 * N = 6000 := by
  sorry

end percentage_problem_l1919_191941


namespace book_price_percentage_l1919_191968

theorem book_price_percentage (suggested_retail_price : ℝ) : 
  suggested_retail_price > 0 →
  let marked_price := 0.6 * suggested_retail_price
  let alice_paid := 0.6 * marked_price
  alice_paid / suggested_retail_price = 0.36 := by
  sorry

end book_price_percentage_l1919_191968


namespace problem_solution_l1919_191911

theorem problem_solution (x : ℚ) : (2 * x + 10 - 2) / 7 = 15 → x = 97 / 2 := by
  sorry

end problem_solution_l1919_191911


namespace total_fuel_consumption_l1919_191913

/-- Fuel consumption for city driving in liters per km -/
def city_fuel_rate : ℝ := 6

/-- Fuel consumption for highway driving in liters per km -/
def highway_fuel_rate : ℝ := 4

/-- Distance of the city-only trip in km -/
def city_trip : ℝ := 50

/-- Distance of the highway-only trip in km -/
def highway_trip : ℝ := 35

/-- Distance of the mixed trip's city portion in km -/
def mixed_trip_city : ℝ := 15

/-- Distance of the mixed trip's highway portion in km -/
def mixed_trip_highway : ℝ := 10

/-- Theorem stating that the total fuel consumption for all trips is 570 liters -/
theorem total_fuel_consumption :
  city_fuel_rate * city_trip +
  highway_fuel_rate * highway_trip +
  city_fuel_rate * mixed_trip_city +
  highway_fuel_rate * mixed_trip_highway = 570 := by
  sorry


end total_fuel_consumption_l1919_191913


namespace floor_abs_sum_l1919_191957

theorem floor_abs_sum : ⌊|(-7.9 : ℝ)|⌋ + |⌊(-7.9 : ℝ)⌋| = 15 := by sorry

end floor_abs_sum_l1919_191957


namespace total_chips_eaten_l1919_191963

/-- 
Given that John eats x bags of chips for dinner and 2x bags after dinner,
prove that the total number of bags eaten is 3x.
-/
theorem total_chips_eaten (x : ℕ) : x + 2*x = 3*x := by
  sorry

end total_chips_eaten_l1919_191963


namespace smallest_n_satisfying_conditions_l1919_191928

def is_multiple_of_75 (n : ℕ) : Prop := ∃ k : ℕ, n = 75 * k

def count_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

def satisfies_conditions (n : ℕ) : Prop :=
  is_multiple_of_75 n ∧ count_divisors n = 75

theorem smallest_n_satisfying_conditions :
  ∃! n : ℕ, satisfies_conditions n ∧ ∀ m : ℕ, satisfies_conditions m → n ≤ m :=
by sorry

end smallest_n_satisfying_conditions_l1919_191928


namespace least_positive_integer_with_remainders_l1919_191916

theorem least_positive_integer_with_remainders : ∃ (x : ℕ), 
  (x > 0) ∧ 
  (x % 2 = 1) ∧ 
  (x % 5 = 2) ∧ 
  (x % 6 = 3) ∧ 
  (∀ y : ℕ, y > 0 ∧ y % 2 = 1 ∧ y % 5 = 2 ∧ y % 6 = 3 → x ≤ y) ∧
  x = 27 := by
  sorry

end least_positive_integer_with_remainders_l1919_191916


namespace complex_powers_sum_l1919_191966

theorem complex_powers_sum : 
  (((Complex.I * Real.sqrt 3 - 1) / 2) ^ 6 + ((Complex.I * Real.sqrt 3 + 1) / (-2)) ^ 6 = 2) ∧
  (∀ n : ℕ, Odd n → ((Complex.I + 1) / Real.sqrt 2) ^ (4 * n) + ((1 - Complex.I) / Real.sqrt 2) ^ (4 * n) = -2) :=
by sorry

end complex_powers_sum_l1919_191966


namespace arithmetic_progression_first_term_l1919_191902

theorem arithmetic_progression_first_term
  (n : ℕ)
  (a_n : ℝ)
  (d : ℝ)
  (h1 : n = 15)
  (h2 : a_n = 44)
  (h3 : d = 3) :
  ∃ a₁ : ℝ, a₁ = 2 ∧ a_n = a₁ + (n - 1) * d :=
by sorry

end arithmetic_progression_first_term_l1919_191902


namespace concert_revenue_l1919_191964

theorem concert_revenue (total_attendees : ℕ) (reserved_price unreserved_price : ℚ)
  (reserved_sold unreserved_sold : ℕ) :
  total_attendees = reserved_sold + unreserved_sold →
  reserved_price = 25 →
  unreserved_price = 20 →
  reserved_sold = 246 →
  unreserved_sold = 246 →
  (reserved_sold : ℚ) * reserved_price + (unreserved_sold : ℚ) * unreserved_price = 11070 :=
by sorry

end concert_revenue_l1919_191964


namespace simplify_expression_l1919_191975

theorem simplify_expression (x : ℝ) :
  4 * x - 8 * x^2 + 10 - (5 - 4 * x + 8 * x^2) = -16 * x^2 + 8 * x + 5 := by
  sorry

end simplify_expression_l1919_191975


namespace probability_point_between_B_and_E_l1919_191945

/-- Given a line segment AB with points A, B, C, D, and E such that AB = 4AD = 8BE = 2BC,
    the probability that a randomly chosen point on AB lies between B and E is 1/8. -/
theorem probability_point_between_B_and_E (A B C D E : ℝ) : 
  A < D ∧ D < E ∧ E < B ∧ B < C →  -- Points are ordered on the line
  (B - A) = 4 * (D - A) →          -- AB = 4AD
  (B - A) = 8 * (B - E) →          -- AB = 8BE
  (B - A) = 2 * (C - B) →          -- AB = 2BC
  (B - E) / (B - A) = 1 / 8 :=     -- Probability is 1/8
by sorry

end probability_point_between_B_and_E_l1919_191945


namespace officer_average_salary_l1919_191918

/-- Proves that the average salary of officers is 450 Rs/month --/
theorem officer_average_salary
  (total_avg : ℝ)
  (non_officer_avg : ℝ)
  (officer_count : ℕ)
  (non_officer_count : ℕ)
  (h_total_avg : total_avg = 120)
  (h_non_officer_avg : non_officer_avg = 110)
  (h_officer_count : officer_count = 15)
  (h_non_officer_count : non_officer_count = 495) :
  (total_avg * (officer_count + non_officer_count) - non_officer_avg * non_officer_count) / officer_count = 450 :=
by sorry

end officer_average_salary_l1919_191918


namespace triangle_median_theorem_l1919_191915

/-- Represents a triangle with given side lengths and medians -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  m_a : ℝ
  m_b : ℝ
  m_c : ℝ
  area : ℝ

/-- The theorem stating the properties of the specific triangle -/
theorem triangle_median_theorem (t : Triangle) 
  (h1 : t.a = 5)
  (h2 : t.b = 8)
  (h3 : t.m_a = 4)
  (h4 : t.area = 12) :
  t.m_b = 4 := by
  sorry


end triangle_median_theorem_l1919_191915


namespace min_zeros_odd_period_two_l1919_191969

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function f: ℝ → ℝ has period 2 if f(x) = f(x+2) for all x ∈ ℝ -/
def HasPeriodTwo (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (x + 2)

/-- The number of zeros of a function f: ℝ → ℝ in an interval [a, b] -/
def NumberOfZeros (f : ℝ → ℝ) (a b : ℝ) : ℕ :=
  sorry  -- Definition omitted for brevity

/-- The theorem stating the minimum number of zeros for an odd function with period 2 -/
theorem min_zeros_odd_period_two (f : ℝ → ℝ) (h_odd : IsOdd f) (h_period : HasPeriodTwo f) :
  NumberOfZeros f 0 2009 ≥ 2010 :=
sorry

end min_zeros_odd_period_two_l1919_191969


namespace prime_squares_end_in_nine_l1919_191997

theorem prime_squares_end_in_nine :
  ∀ p q : ℕ,
  Prime p → Prime q →
  (p * p + q * q) % 10 = 9 →
  ((p = 2 ∧ q = 5) ∨ (p = 5 ∧ q = 2)) := by
sorry

end prime_squares_end_in_nine_l1919_191997


namespace triangle_median_and_symmetric_point_l1919_191907

/-- Triangle OAB with vertices O(0,0), A(2,0), and B(3,2) -/
structure Triangle :=
  (O : ℝ × ℝ)
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)

/-- Line l containing the median on side OA -/
structure MedianLine :=
  (slope : ℝ)
  (intercept : ℝ)

/-- Point symmetric to A with respect to line l -/
structure SymmetricPoint :=
  (x : ℝ)
  (y : ℝ)

theorem triangle_median_and_symmetric_point 
  (t : Triangle)
  (l : MedianLine)
  (h1 : t.O = (0, 0))
  (h2 : t.A = (2, 0))
  (h3 : t.B = (3, 2))
  (h4 : l.slope = 1)
  (h5 : l.intercept = -1)
  : ∃ (p : SymmetricPoint), 
    (∀ (x y : ℝ), y = l.slope * x + l.intercept ↔ y = x - 1) ∧
    p.x = 1 ∧ p.y = 1 := by
  sorry

end triangle_median_and_symmetric_point_l1919_191907


namespace honda_production_l1919_191953

theorem honda_production (day_shift : ℕ) (second_shift : ℕ) : 
  day_shift = 4 * second_shift → 
  day_shift + second_shift = 5500 → 
  day_shift = 4400 := by
sorry

end honda_production_l1919_191953


namespace impossibleConfig6_impossibleConfig4_impossibleConfig3_l1919_191942

/-- Represents the sign at a vertex -/
inductive Sign
| Positive
| Negative

/-- Represents a dodecagon configuration -/
def DodecagonConfig := Fin 12 → Sign

/-- Initial configuration with A₁ negative and others positive -/
def initialConfig : DodecagonConfig :=
  fun i => if i = 0 then Sign.Negative else Sign.Positive

/-- Applies the sign-flipping operation to n consecutive vertices starting at index i -/
def flipSigns (config : DodecagonConfig) (i : Fin 12) (n : Nat) : DodecagonConfig :=
  fun j => if (j - i) % 12 < n then
    match config j with
    | Sign.Positive => Sign.Negative
    | Sign.Negative => Sign.Positive
    else config j

/-- Checks if only A₂ is negative in the configuration -/
def onlyA₂Negative (config : DodecagonConfig) : Prop :=
  config 1 = Sign.Negative ∧ ∀ i, i ≠ 1 → config i = Sign.Positive

/-- Main theorem for 6 consecutive vertices -/
theorem impossibleConfig6 :
  ∀ (ops : List (Fin 12)), 
  let finalConfig := (ops.foldl (fun cfg i => flipSigns cfg i 6) initialConfig)
  ¬(onlyA₂Negative finalConfig) := by sorry

/-- Main theorem for 4 consecutive vertices -/
theorem impossibleConfig4 :
  ∀ (ops : List (Fin 12)), 
  let finalConfig := (ops.foldl (fun cfg i => flipSigns cfg i 4) initialConfig)
  ¬(onlyA₂Negative finalConfig) := by sorry

/-- Main theorem for 3 consecutive vertices -/
theorem impossibleConfig3 :
  ∀ (ops : List (Fin 12)), 
  let finalConfig := (ops.foldl (fun cfg i => flipSigns cfg i 3) initialConfig)
  ¬(onlyA₂Negative finalConfig) := by sorry

end impossibleConfig6_impossibleConfig4_impossibleConfig3_l1919_191942


namespace sector_central_angle_l1919_191930

/-- A sector with radius 1 cm and circumference 4 cm has a central angle of 2 radians. -/
theorem sector_central_angle (r : ℝ) (circ : ℝ) (h1 : r = 1) (h2 : circ = 4) :
  let arc_length := circ - 2 * r
  arc_length / r = 2 := by
sorry

end sector_central_angle_l1919_191930


namespace exists_term_divisible_by_2006_l1919_191971

theorem exists_term_divisible_by_2006 : ∃ n : ℤ, (2006 : ℤ) ∣ (n^3 - (2*n + 1)^2) := by
  sorry

end exists_term_divisible_by_2006_l1919_191971


namespace hash_property_l1919_191954

/-- Operation # for non-negative integers -/
def hash (a b : ℕ) : ℕ := 4 * a^2 + 4 * b^2 + 8 * a * b

/-- Theorem: If a # b = 100, then (a + b) + 5 = 10 for non-negative integers a and b -/
theorem hash_property (a b : ℕ) (h : hash a b = 100) : (a + b) + 5 = 10 := by
  sorry

end hash_property_l1919_191954


namespace root_implies_k_value_l1919_191952

theorem root_implies_k_value (k : ℝ) : 
  (3 : ℝ)^4 + k * (3 : ℝ)^2 + 27 = 0 → k = -12 := by
sorry

end root_implies_k_value_l1919_191952


namespace smallest_of_three_consecutive_sum_90_l1919_191920

theorem smallest_of_three_consecutive_sum_90 (x y z : ℤ) :
  y = x + 1 ∧ z = x + 2 ∧ x + y + z = 90 → x = 29 := by
  sorry

end smallest_of_three_consecutive_sum_90_l1919_191920


namespace april_rainfall_calculation_l1919_191903

def march_rainfall : ℝ := 0.81
def april_decrease : ℝ := 0.35

theorem april_rainfall_calculation :
  march_rainfall - april_decrease = 0.46 := by sorry

end april_rainfall_calculation_l1919_191903


namespace monotonic_function_a_range_l1919_191960

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a*x^2 - x - 2

-- State the theorem
theorem monotonic_function_a_range :
  (∀ a : ℝ, Monotone (f a) → a ∈ Set.Icc (- Real.sqrt 3) (Real.sqrt 3)) ∧
  (∀ a : ℝ, a ∈ Set.Icc (- Real.sqrt 3) (Real.sqrt 3) → Monotone (f a)) :=
sorry

end monotonic_function_a_range_l1919_191960


namespace digit_puzzle_solutions_l1919_191951

def is_valid_solution (a b : ℕ) : Prop :=
  a ≠ b ∧
  a < 10 ∧ b < 10 ∧
  10 ≤ 10 * b + a ∧ 10 * b + a < 100 ∧
  10 * b + a ≠ a * b ∧
  a ^ b = 10 * b + a

theorem digit_puzzle_solutions :
  {(a, b) : ℕ × ℕ | is_valid_solution a b} =
  {(2, 5), (6, 2), (4, 3)} := by sorry

end digit_puzzle_solutions_l1919_191951


namespace geometric_sequence_second_term_l1919_191947

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def GeometricSequence (a : ℕ → ℚ) : Prop :=
  ∃ r : ℚ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_second_term
  (a : ℕ → ℚ)
  (h_geometric : GeometricSequence a)
  (h_fifth : a 5 = 48)
  (h_sixth : a 6 = 72) :
  a 2 = 128 / 9 := by
sorry

end geometric_sequence_second_term_l1919_191947


namespace identical_numbers_iff_even_l1919_191989

/-- A function that represents the operation of selecting two numbers and replacing them with their sum. -/
def sumOperation (numbers : List ℕ) : List (List ℕ) :=
  sorry

/-- A predicate that checks if all numbers in a list are identical. -/
def allIdentical (numbers : List ℕ) : Prop :=
  sorry

/-- A proposition stating that it's possible to transform n numbers into n identical numbers
    using the sum operation if and only if n is even. -/
theorem identical_numbers_iff_even (n : ℕ) (h : n ≥ 2) :
  (∃ (initial : List ℕ) (final : List ℕ),
    initial.length = n ∧
    final.length = n ∧
    allIdentical final ∧
    final ∈ sumOperation initial) ↔ Even n :=
  sorry

end identical_numbers_iff_even_l1919_191989


namespace initial_wax_amount_l1919_191924

/-- Given the total required amount of wax and the additional amount needed,
    calculate the initial amount of wax available. -/
theorem initial_wax_amount (total_required additional_needed : ℕ) :
  total_required ≥ additional_needed →
  total_required - additional_needed = total_required - additional_needed :=
by sorry

end initial_wax_amount_l1919_191924


namespace transformed_system_solution_l1919_191910

theorem transformed_system_solution 
  (a₁ a₂ b₁ b₂ c₁ c₂ : ℝ) 
  (h₁ : a₁ * 3 - b₁ * 5 = c₁) 
  (h₂ : a₂ * 3 + b₂ * 5 = c₂) :
  a₁ * (11 - 2) - b₁ * 15 = 3 * c₁ ∧ 
  a₂ * (11 - 2) + b₂ * 15 = 3 * c₂ := by
  sorry

end transformed_system_solution_l1919_191910


namespace circle_intersection_theorem_l1919_191919

open Real

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 4

-- Define the line l
def line_l (x y m : ℝ) : Prop := x - y + m = 0

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) (m : ℝ) : Prop :=
  circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧ line_l A.1 A.2 m ∧ line_l B.1 B.2 m

-- Define the angle ACB
def angle_ACB (A B : ℝ × ℝ) : ℝ := sorry

-- Define the distance between two points
def distance (P Q : ℝ × ℝ) : ℝ := sorry

theorem circle_intersection_theorem (m : ℝ) (A B : ℝ × ℝ) :
  intersection_points A B m →
  ((angle_ACB A B = 2 * π / 3) ∨ (distance A B = 2 * sqrt 3)) →
  (m = sqrt 2 - 1 ∨ m = -sqrt 2 - 1) :=
sorry

end circle_intersection_theorem_l1919_191919


namespace opposite_face_is_E_l1919_191935

/-- Represents the faces of a cube --/
inductive Face
  | A | B | C | D | E | F

/-- Represents a cube --/
structure Cube where
  faces : List Face
  top : Face
  adjacent : Face → List Face
  opposite : Face → Face

/-- The cube configuration described in the problem --/
def problem_cube : Cube :=
  { faces := [Face.A, Face.B, Face.C, Face.D, Face.E, Face.F]
  , top := Face.F
  , adjacent := fun f => match f with
    | Face.D => [Face.A, Face.B, Face.C]
    | _ => sorry  -- We don't have information about other adjacencies
  , opposite := sorry  -- To be proven
  }

theorem opposite_face_is_E :
  problem_cube.opposite Face.A = Face.E :=
by sorry

end opposite_face_is_E_l1919_191935


namespace symmetric_point_wrt_x_axis_l1919_191986

/-- Given a point M with coordinates (3, -4), its symmetric point with respect to the x-axis has coordinates (3, 4) -/
theorem symmetric_point_wrt_x_axis :
  let M : ℝ × ℝ := (3, -4)
  let symmetric_point (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)
  symmetric_point M = (3, 4) := by sorry

end symmetric_point_wrt_x_axis_l1919_191986


namespace apron_sewing_ratio_l1919_191970

/-- Prove that the ratio of aprons sewn today to aprons sewn before today is 3:1 -/
theorem apron_sewing_ratio :
  let total_aprons : ℕ := 150
  let aprons_before : ℕ := 13
  let aprons_tomorrow : ℕ := 49
  let aprons_remaining : ℕ := 2 * aprons_tomorrow
  let aprons_sewn_before_tomorrow : ℕ := total_aprons - aprons_remaining
  let aprons_today : ℕ := aprons_sewn_before_tomorrow - aprons_before
  ∃ (n : ℕ), n > 0 ∧ aprons_today = 3 * n ∧ aprons_before = n :=
by
  sorry

end apron_sewing_ratio_l1919_191970


namespace palmer_photos_l1919_191978

theorem palmer_photos (initial_photos : ℕ) (final_photos : ℕ) (third_fourth_week_photos : ℕ) :
  initial_photos = 100 →
  final_photos = 380 →
  third_fourth_week_photos = 80 →
  ∃ (first_week_photos : ℕ),
    first_week_photos = 67 ∧
    final_photos = initial_photos + first_week_photos + 2 * first_week_photos + third_fourth_week_photos :=
by sorry

end palmer_photos_l1919_191978


namespace vector_equation_solution_parallel_vectors_solution_l1919_191984

/-- Given vectors in R^2 -/
def a : Fin 2 → ℚ := ![3, 2]
def b : Fin 2 → ℚ := ![-1, 2]
def c : Fin 2 → ℚ := ![4, 1]

/-- Part 1: Prove that m = 5/9 and n = 8/9 satisfy a = m*b + n*c -/
theorem vector_equation_solution :
  ∃ (m n : ℚ), (m = 5/9 ∧ n = 8/9) ∧ (∀ i : Fin 2, a i = m * b i + n * c i) :=
sorry

/-- Part 2: Prove that k = -16/13 makes (a + k*c) parallel to (2*b - a) -/
theorem parallel_vectors_solution :
  ∃ (k : ℚ), k = -16/13 ∧
  ∃ (t : ℚ), ∀ i : Fin 2, (a i + k * c i) = t * (2 * b i - a i) :=
sorry

end vector_equation_solution_parallel_vectors_solution_l1919_191984


namespace student_calculation_difference_l1919_191998

/-- Proves that dividing a number by 4/5 instead of multiplying it by 4/5 results in a specific difference -/
theorem student_calculation_difference (number : ℝ) (h : number = 40.000000000000014) :
  (number / (4/5)) - (number * (4/5)) = 18.00000000000001 := by
  sorry

end student_calculation_difference_l1919_191998


namespace m_range_l1919_191938

-- Define the condition function
def condition (x m : ℝ) : Prop := x^2 + m*x - 2*m^2 < 0

-- Define the sufficient condition
def sufficient_condition (x : ℝ) : Prop := -2 < x ∧ x < 3

-- State the theorem
theorem m_range (m : ℝ) :
  (m > 0) →
  (∀ x, sufficient_condition x → condition x m) →
  (∃ x, condition x m ∧ ¬sufficient_condition x) →
  m ≥ 3 :=
sorry

end m_range_l1919_191938


namespace only_zero_has_linear_factors_l1919_191940

/-- A polynomial in x and y with a parameter k -/
def poly (k : ℤ) (x y : ℤ) : ℤ := x^2 + 4*x*y + 2*x + k*y - k

/-- Predicate for linear factors with integer coefficients -/
def has_linear_integer_factors (k : ℤ) : Prop :=
  ∃ (a b c d e f : ℤ), ∀ (x y : ℤ),
    poly k x y = (a*x + b*y + c) * (d*x + e*y + f)

theorem only_zero_has_linear_factors :
  ∀ k : ℤ, has_linear_integer_factors k ↔ k = 0 :=
by sorry

end only_zero_has_linear_factors_l1919_191940


namespace combined_experience_is_68_l1919_191936

/-- Calculates the combined experience of James, John, and Mike -/
def combinedExperience (james_current : ℕ) (years_ago : ℕ) (john_multiplier : ℕ) (john_when_mike_started : ℕ) : ℕ :=
  let james_past := james_current - years_ago
  let john_past := john_multiplier * james_past
  let john_current := john_past + years_ago
  let mike_experience := john_current - john_when_mike_started
  james_current + john_current + mike_experience

/-- The combined experience of James, John, and Mike is 68 years -/
theorem combined_experience_is_68 :
  combinedExperience 20 8 2 16 = 68 := by
  sorry

end combined_experience_is_68_l1919_191936


namespace absolute_value_equation_product_l1919_191931

theorem absolute_value_equation_product (y₁ y₂ : ℝ) : 
  (|3 * y₁| + 7 = 40) ∧ (|3 * y₂| + 7 = 40) ∧ (y₁ ≠ y₂) → y₁ * y₂ = -121 := by
  sorry

end absolute_value_equation_product_l1919_191931


namespace hyperbola_asymptote_l1919_191921

theorem hyperbola_asymptote (m : ℝ) (h1 : m > 0) :
  (∃ x y, x^2 / m^2 - y^2 = 1 ∧ x + Real.sqrt 3 * y = 0) → m = Real.sqrt 3 :=
by sorry

end hyperbola_asymptote_l1919_191921


namespace ghost_entry_exit_ways_l1919_191956

def num_windows : ℕ := 8

theorem ghost_entry_exit_ways :
  (num_windows : ℕ) * (num_windows - 1) = 56 := by
  sorry

end ghost_entry_exit_ways_l1919_191956


namespace pi_power_zero_plus_two_power_neg_two_l1919_191988

theorem pi_power_zero_plus_two_power_neg_two :
  (-Real.pi)^(0 : ℤ) + 2^(-2 : ℤ) = 5/4 := by sorry

end pi_power_zero_plus_two_power_neg_two_l1919_191988


namespace max_power_of_15_l1919_191994

-- Define pow function
def pow (n : ℕ) : ℕ := sorry

-- Define the product of pow(n) from 2 to 2200
def product_pow : ℕ := sorry

-- Theorem statement
theorem max_power_of_15 :
  (∀ m : ℕ, m > 10 → ¬(pow (15^m) ∣ product_pow)) ∧
  (pow (15^10) ∣ product_pow) :=
sorry

end max_power_of_15_l1919_191994


namespace absolute_value_not_always_zero_l1919_191961

theorem absolute_value_not_always_zero : ¬ (∀ x : ℝ, |x| = 0) := by
  sorry

end absolute_value_not_always_zero_l1919_191961


namespace units_digit_sum_of_powers_l1919_191982

theorem units_digit_sum_of_powers : (24^4 + 42^4) % 10 = 2 := by
  sorry

end units_digit_sum_of_powers_l1919_191982


namespace cos_315_degrees_l1919_191983

theorem cos_315_degrees : Real.cos (315 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end cos_315_degrees_l1919_191983


namespace quadratic_equal_roots_l1919_191977

/-- For the quadratic equation x^2 + 4x√2 + k = 0, prove that k = 8 makes the discriminant zero and the roots real and equal. -/
theorem quadratic_equal_roots (k : ℝ) : 
  (∀ x, x^2 + 4*x*Real.sqrt 2 + k = 0) →
  (k = 8 ↔ (∃! r, r^2 + 4*r*Real.sqrt 2 + k = 0)) :=
by sorry

end quadratic_equal_roots_l1919_191977


namespace seven_classes_tournament_l1919_191980

/-- Calculate the number of matches in a round-robin tournament -/
def numberOfMatches (n : ℕ) : ℕ :=
  n * (n - 1) / 2

/-- Theorem: For 7 classes in a round-robin tournament, the total number of matches is 21 -/
theorem seven_classes_tournament : numberOfMatches 7 = 21 := by
  sorry

end seven_classes_tournament_l1919_191980


namespace area_midpoint_rectangle_l1919_191991

/-- Given a rectangle EFGH with width w and height h, and points P and Q that are
    midpoints of the longer sides EF and GH respectively, the area of EPGQ is
    half the area of EFGH. -/
theorem area_midpoint_rectangle (w h : ℝ) (hw : w > 0) (hh : h > 0) :
  let rect_area := w * h
  let midpoint_rect_area := (w / 2) * h
  midpoint_rect_area = rect_area / 2 := by
  sorry

#check area_midpoint_rectangle

end area_midpoint_rectangle_l1919_191991


namespace grid_removal_l1919_191962

theorem grid_removal (n : ℕ) (h : n ≥ 10) :
  ∀ (grid : Fin n → Fin n → Bool),
  (∃ (rows : Finset (Fin n)),
    rows.card = n - 10 ∧
    ∀ (j : Fin n), ∃ (i : Fin n), i ∉ rows ∧ grid i j = true) ∨
  (∃ (cols : Finset (Fin n)),
    cols.card = n - 10 ∧
    ∀ (i : Fin n), ∃ (j : Fin n), j ∉ cols ∧ grid i j = false) :=
sorry

end grid_removal_l1919_191962
