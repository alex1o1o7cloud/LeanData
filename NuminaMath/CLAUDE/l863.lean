import Mathlib

namespace NUMINAMATH_CALUDE_normal_carwash_cost_l863_86383

/-- The normal cost of a single carwash, given a discounted package deal -/
theorem normal_carwash_cost (package_size : ℕ) (discount_rate : ℚ) (package_price : ℚ) : 
  package_size = 20 →
  discount_rate = 3/5 →
  package_price = 180 →
  package_price = discount_rate * (package_size * (15 : ℚ)) := by
sorry

end NUMINAMATH_CALUDE_normal_carwash_cost_l863_86383


namespace NUMINAMATH_CALUDE_sum_digits_first_1500_even_l863_86342

/-- The number of digits in a positive integer -/
def num_digits (n : ℕ) : ℕ := sorry

/-- The sum of digits for all even numbers from 2 to n -/
def sum_digits_even (n : ℕ) : ℕ := sorry

/-- The 1500th positive even integer -/
def nth_even (n : ℕ) : ℕ := 2 * n

theorem sum_digits_first_1500_even :
  sum_digits_even (nth_even 1500) = 5448 := by sorry

end NUMINAMATH_CALUDE_sum_digits_first_1500_even_l863_86342


namespace NUMINAMATH_CALUDE_abs_x_squared_minus_4x_plus_3_lt_6_l863_86313

theorem abs_x_squared_minus_4x_plus_3_lt_6 (x : ℝ) :
  |x^2 - 4*x + 3| < 6 ↔ 1 < x ∧ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_x_squared_minus_4x_plus_3_lt_6_l863_86313


namespace NUMINAMATH_CALUDE_expected_waiting_time_correct_l863_86303

/-- Represents the arrival time of a bus in minutes past 8:00 AM -/
def BusArrivalTime := Fin 120

/-- Represents the probability distribution of bus arrivals -/
def BusDistribution := BusArrivalTime → ℝ

/-- The first bus arrives randomly between 8:00 and 9:00 -/
def firstBusDistribution : BusDistribution := sorry

/-- The second bus arrives randomly between 9:00 and 10:00 -/
def secondBusDistribution : BusDistribution := sorry

/-- The passenger arrival time in minutes past 8:00 AM -/
def passengerArrivalTime : ℕ := 20

/-- Expected waiting time function -/
def expectedWaitingTime (firstBus secondBus : BusDistribution) (passengerTime : ℕ) : ℝ := sorry

theorem expected_waiting_time_correct :
  expectedWaitingTime firstBusDistribution secondBusDistribution passengerArrivalTime = 160 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expected_waiting_time_correct_l863_86303


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l863_86330

def f (a x : ℝ) := a * x^2 + x + 1

theorem quadratic_function_properties (a : ℝ) (h_a : a > 0) :
  -- Part 1: Maximum value in the interval [-4, -2]
  (∀ x ∈ Set.Icc (-4) (-2), f a x ≤ (if a ≤ 1/6 then 4*a - 1 else 16*a - 3)) ∧
  (∃ x ∈ Set.Icc (-4) (-2), f a x = (if a ≤ 1/6 then 4*a - 1 else 16*a - 3)) ∧
  -- Part 2: Maximum value of a given root conditions
  (∀ x₁ x₂ : ℝ, f a x₁ = 0 → f a x₂ = 0 → x₁ ≠ x₂ → x₁ / x₂ ∈ Set.Icc (1/10) 10 → a ≤ 1/4) ∧
  (∃ x₁ x₂ : ℝ, f (1/4) x₁ = 0 ∧ f (1/4) x₂ = 0 ∧ x₁ ≠ x₂ ∧ x₁ / x₂ ∈ Set.Icc (1/10) 10) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l863_86330


namespace NUMINAMATH_CALUDE_diamond_expression_evaluation_l863_86315

-- Define the diamond operation
def diamond (a b : ℚ) : ℚ := a - 1 / b

-- State the theorem
theorem diamond_expression_evaluation :
  let x := diamond (diamond 2 3) 4
  let y := diamond 2 (diamond 3 4)
  x - y = -29/132 := by sorry

end NUMINAMATH_CALUDE_diamond_expression_evaluation_l863_86315


namespace NUMINAMATH_CALUDE_probability_theorem_l863_86379

/-- The probability of drawing one white ball and one black ball from an urn -/
def probability_one_white_one_black (a b : ℕ) : ℚ :=
  (2 * a * b : ℚ) / ((a + b) * (a + b - 1))

/-- Theorem stating the probability of drawing one white and one black ball -/
theorem probability_theorem (a b : ℕ) (h : a + b > 1) :
  probability_one_white_one_black a b =
    (2 * a * b : ℚ) / ((a + b) * (a + b - 1)) := by
  sorry

end NUMINAMATH_CALUDE_probability_theorem_l863_86379


namespace NUMINAMATH_CALUDE_probability_unqualified_example_l863_86396

/-- Represents the probability of selecting at least one unqualified can -/
def probability_unqualified (total_cans : ℕ) (qualified_cans : ℕ) (unqualified_cans : ℕ) (selected_cans : ℕ) : ℚ :=
  1 - (Nat.choose qualified_cans selected_cans : ℚ) / (Nat.choose total_cans selected_cans : ℚ)

/-- Theorem stating that the probability of selecting at least one unqualified can
    when randomly choosing 2 cans from a box containing 3 qualified cans and 2 unqualified cans
    is equal to 0.7 -/
theorem probability_unqualified_example : probability_unqualified 5 3 2 2 = 7/10 := by
  sorry

end NUMINAMATH_CALUDE_probability_unqualified_example_l863_86396


namespace NUMINAMATH_CALUDE_profit_share_difference_example_l863_86343

/-- Calculates the difference between profit shares of two partners given investments and one partner's profit share. -/
def profit_share_difference (inv_a inv_b inv_c b_profit : ℚ) : ℚ :=
  let total_inv := inv_a + inv_b + inv_c
  let total_profit := (total_inv / inv_b) * b_profit
  let a_share := (inv_a / total_inv) * total_profit
  let c_share := (inv_c / total_inv) * total_profit
  c_share - a_share

/-- Proves that the difference between profit shares of a and c is 600 given the specified investments and b's profit share. -/
theorem profit_share_difference_example : 
  profit_share_difference 8000 10000 12000 1500 = 600 := by
  sorry


end NUMINAMATH_CALUDE_profit_share_difference_example_l863_86343


namespace NUMINAMATH_CALUDE_second_bakery_sacks_per_week_l863_86397

/-- Proves that the second bakery needs 4 sacks per week given the conditions of Antoine's strawberry supply -/
theorem second_bakery_sacks_per_week 
  (total_sacks : ℕ) 
  (num_weeks : ℕ) 
  (first_bakery_sacks_per_week : ℕ) 
  (third_bakery_sacks_per_week : ℕ) 
  (h1 : total_sacks = 72) 
  (h2 : num_weeks = 4) 
  (h3 : first_bakery_sacks_per_week = 2) 
  (h4 : third_bakery_sacks_per_week = 12) : 
  (total_sacks - (first_bakery_sacks_per_week * num_weeks) - (third_bakery_sacks_per_week * num_weeks)) / num_weeks = 4 := by
sorry

end NUMINAMATH_CALUDE_second_bakery_sacks_per_week_l863_86397


namespace NUMINAMATH_CALUDE_pyramid_surface_area_l863_86304

-- Define the pyramid
structure RectangularPyramid where
  baseSideLength : ℝ
  sideEdgeLength : ℝ

-- Define the properties of the pyramid
def isPyramidOnSphere (p : RectangularPyramid) : Prop :=
  (p.baseSideLength ^ 2 + p.baseSideLength ^ 2 + p.sideEdgeLength ^ 2) / 4 = 1

def hasSquareBase (p : RectangularPyramid) : Prop :=
  p.baseSideLength ^ 2 + p.baseSideLength ^ 2 = 2

def sideEdgesPerpendicular (p : RectangularPyramid) : Prop :=
  p.sideEdgeLength ^ 2 + p.baseSideLength ^ 2 / 2 = 1

-- Define the surface area calculation
def surfaceArea (p : RectangularPyramid) : ℝ :=
  p.baseSideLength ^ 2 + 4 * p.baseSideLength * p.sideEdgeLength

-- Theorem statement
theorem pyramid_surface_area (p : RectangularPyramid) :
  isPyramidOnSphere p → hasSquareBase p → sideEdgesPerpendicular p →
  surfaceArea p = 2 + 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_surface_area_l863_86304


namespace NUMINAMATH_CALUDE_no_multiple_of_five_2c4_l863_86358

/-- A positive three-digit number in the form 2C4, where C is a single digit. -/
def number (c : ℕ) : ℕ := 200 + 10 * c + 4

/-- Predicate to check if a number is a multiple of 5. -/
def is_multiple_of_five (n : ℕ) : Prop := ∃ k : ℕ, n = 5 * k

/-- Theorem stating that there are no digits C such that 2C4 is a multiple of 5. -/
theorem no_multiple_of_five_2c4 :
  ¬ ∃ c : ℕ, c < 10 ∧ is_multiple_of_five (number c) := by
  sorry


end NUMINAMATH_CALUDE_no_multiple_of_five_2c4_l863_86358


namespace NUMINAMATH_CALUDE_johns_work_hours_l863_86329

/-- Calculates the number of hours John works every other day given his wage, raise percentage, total earnings, and days in a month. -/
theorem johns_work_hours 
  (former_wage : ℝ)
  (raise_percentage : ℝ)
  (total_earnings : ℝ)
  (days_in_month : ℕ)
  (h1 : former_wage = 20)
  (h2 : raise_percentage = 30)
  (h3 : total_earnings = 4680)
  (h4 : days_in_month = 30) :
  let new_wage := former_wage * (1 + raise_percentage / 100)
  let working_days := days_in_month / 2
  let total_hours := total_earnings / new_wage
  let hours_per_working_day := total_hours / working_days
  hours_per_working_day = 12 := by sorry

end NUMINAMATH_CALUDE_johns_work_hours_l863_86329


namespace NUMINAMATH_CALUDE_min_value_expression_l863_86302

theorem min_value_expression (x₁ x₂ : ℝ) (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₁ + x₂ = 1) :
  3 * x₁ / x₂ + 1 / (x₁ * x₂) ≥ 6 ∧ 
  ∃ x₁' x₂' : ℝ, x₁' > 0 ∧ x₂' > 0 ∧ x₁' + x₂' = 1 ∧ 3 * x₁' / x₂' + 1 / (x₁' * x₂') = 6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l863_86302


namespace NUMINAMATH_CALUDE_boys_without_notebooks_l863_86326

/-- Proves the number of boys without notebooks in Ms. Johnson's class -/
theorem boys_without_notebooks
  (total_boys : ℕ)
  (students_with_notebooks : ℕ)
  (girls_with_notebooks : ℕ)
  (h1 : total_boys = 24)
  (h2 : students_with_notebooks = 30)
  (h3 : girls_with_notebooks = 17) :
  total_boys - (students_with_notebooks - girls_with_notebooks) = 11 :=
by sorry

end NUMINAMATH_CALUDE_boys_without_notebooks_l863_86326


namespace NUMINAMATH_CALUDE_negation_of_forall_nonnegative_square_l863_86345

theorem negation_of_forall_nonnegative_square (p : Prop) : 
  (p ↔ ∀ x : ℝ, x^2 ≥ 0) → (¬p ↔ ∃ x : ℝ, x^2 < 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_forall_nonnegative_square_l863_86345


namespace NUMINAMATH_CALUDE_theater_ticket_income_theater_income_proof_l863_86348

/-- Calculate the total ticket income for a theater -/
theorem theater_ticket_income 
  (total_seats : ℕ) 
  (adult_price child_price : ℚ) 
  (children_count : ℕ) : ℚ :=
  let adult_count : ℕ := total_seats - children_count
  let adult_income : ℚ := adult_count * adult_price
  let child_income : ℚ := children_count * child_price
  adult_income + child_income

/-- Prove that the total ticket income for the given theater scenario is $510.00 -/
theorem theater_income_proof :
  theater_ticket_income 200 3 (3/2) 60 = 510 := by
  sorry

end NUMINAMATH_CALUDE_theater_ticket_income_theater_income_proof_l863_86348


namespace NUMINAMATH_CALUDE_smallest_quadratic_coefficient_l863_86362

theorem smallest_quadratic_coefficient (a b c : ℤ) :
  (∃ x y : ℝ, x ≠ y ∧ 0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 ∧ 
   a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0) →
  |a| ≥ 5 :=
sorry

end NUMINAMATH_CALUDE_smallest_quadratic_coefficient_l863_86362


namespace NUMINAMATH_CALUDE_missing_carton_dimension_l863_86399

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (dims : BoxDimensions) : ℝ :=
  dims.length * dims.width * dims.height

/-- Theorem: Given the dimensions of a carton and soap box, if 300 soap boxes fit exactly in the carton,
    then the missing dimension of the carton is 25 inches -/
theorem missing_carton_dimension
  (x : ℝ)
  (carton : BoxDimensions)
  (soap : BoxDimensions)
  (h1 : carton = { length := x, width := 48, height := 60 })
  (h2 : soap = { length := 8, width := 6, height := 5 })
  (h3 : (boxVolume carton) / (boxVolume soap) = 300)
  : x = 25 := by
  sorry

#check missing_carton_dimension

end NUMINAMATH_CALUDE_missing_carton_dimension_l863_86399


namespace NUMINAMATH_CALUDE_clouds_weather_relationship_l863_86318

/-- Represents the contingency table data --/
structure ContingencyTable where
  clouds_rain : Nat
  clouds_no_rain : Nat
  no_clouds_rain : Nat
  no_clouds_no_rain : Nat

/-- Represents the χ² test result --/
structure ChiSquareTest where
  calculated_value : Real
  critical_value : Real

/-- Theorem stating the relationship between clouds at sunset and nighttime weather --/
theorem clouds_weather_relationship (data : ContingencyTable) (test : ChiSquareTest) :
  data.clouds_rain + data.clouds_no_rain + data.no_clouds_rain + data.no_clouds_no_rain = 100 →
  data.clouds_rain + data.no_clouds_rain = 50 →
  data.clouds_no_rain + data.no_clouds_no_rain = 50 →
  test.calculated_value > test.critical_value →
  ∃ (relationship : Prop), relationship := by
  sorry

#check clouds_weather_relationship

end NUMINAMATH_CALUDE_clouds_weather_relationship_l863_86318


namespace NUMINAMATH_CALUDE_sugar_problem_l863_86350

/-- Calculates the remaining sugar after a bag is torn -/
def remaining_sugar (initial_sugar : ℕ) (num_bags : ℕ) (torn_bags : ℕ) : ℕ :=
  let sugar_per_bag := initial_sugar / num_bags
  let intact_sugar := sugar_per_bag * (num_bags - torn_bags)
  let torn_bag_sugar := sugar_per_bag / 2
  intact_sugar + (torn_bag_sugar * torn_bags)

/-- Theorem stating that 21 kilos of sugar remain after one bag is torn -/
theorem sugar_problem :
  remaining_sugar 24 4 1 = 21 := by
  sorry

end NUMINAMATH_CALUDE_sugar_problem_l863_86350


namespace NUMINAMATH_CALUDE_product_equals_32_l863_86337

theorem product_equals_32 : 
  (1 / 2) * 4 * (1 / 8) * 16 * (1 / 32) * 64 * (1 / 128) * 256 * (1 / 512) * 1024 = 32 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_32_l863_86337


namespace NUMINAMATH_CALUDE_maths_fraction_in_class_l863_86317

theorem maths_fraction_in_class (total_students : ℕ) 
  (maths_and_history_students : ℕ) :
  total_students = 25 →
  maths_and_history_students = 20 →
  ∃ (maths_fraction : ℚ),
    maths_fraction * total_students +
    (1 / 3 : ℚ) * (total_students - maths_fraction * total_students) +
    (total_students - maths_fraction * total_students - 
     (1 / 3 : ℚ) * (total_students - maths_fraction * total_students)) = total_students ∧
    maths_fraction * total_students +
    (total_students - maths_fraction * total_students - 
     (1 / 3 : ℚ) * (total_students - maths_fraction * total_students)) = maths_and_history_students ∧
    maths_fraction = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_maths_fraction_in_class_l863_86317


namespace NUMINAMATH_CALUDE_min_diagonal_of_rectangle_l863_86354

/-- Given a rectangle ABCD with perimeter 24 inches, 
    the minimum possible length of its diagonal AC is 6√2 inches. -/
theorem min_diagonal_of_rectangle (l w : ℝ) : 
  (l + w = 12) →  -- perimeter condition: 2l + 2w = 24, simplified
  ∃ (d : ℝ), d = Real.sqrt (l^2 + w^2) ∧ 
              d ≥ 6 * Real.sqrt 2 ∧
              (∀ l' w' : ℝ, l' + w' = 12 → 
                Real.sqrt (l'^2 + w'^2) ≥ 6 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_min_diagonal_of_rectangle_l863_86354


namespace NUMINAMATH_CALUDE_stockholm_uppsala_distance_l863_86338

/-- The scale of the map in km per cm -/
def map_scale : ℝ := 10

/-- The distance between Stockholm and Uppsala on the map in cm -/
def map_distance : ℝ := 45

/-- The actual distance between Stockholm and Uppsala in km -/
def actual_distance : ℝ := map_distance * map_scale

theorem stockholm_uppsala_distance :
  actual_distance = 450 := by sorry

end NUMINAMATH_CALUDE_stockholm_uppsala_distance_l863_86338


namespace NUMINAMATH_CALUDE_fraction_simplification_l863_86369

theorem fraction_simplification (x y : ℚ) (hx : x = 2/3) (hy : y = 5/8) :
  (6*x + 8*y) / (48*x*y) = 9/20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l863_86369


namespace NUMINAMATH_CALUDE_probability_of_specific_arrangement_l863_86376

def total_tiles : ℕ := 8
def x_tiles : ℕ := 5
def o_tiles : ℕ := 3

theorem probability_of_specific_arrangement :
  let total_arrangements := Nat.choose total_tiles x_tiles
  let specific_arrangement := 1
  (specific_arrangement : ℚ) / total_arrangements = 1 / 56 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_specific_arrangement_l863_86376


namespace NUMINAMATH_CALUDE_rationalize_denominator_l863_86324

theorem rationalize_denominator : 
  (Real.sqrt 18 + Real.sqrt 8) / (Real.sqrt 12 + Real.sqrt 8) = 2.5 * Real.sqrt 6 - 4 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l863_86324


namespace NUMINAMATH_CALUDE_max_binomial_probability_l863_86306

def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

theorem max_binomial_probability :
  ∃ (k : ℕ), k ≤ 5 ∧
  ∀ (j : ℕ), j ≤ 5 →
    binomial_probability 5 k (1/4) ≥ binomial_probability 5 j (1/4) ∧
  k = 1 :=
sorry

end NUMINAMATH_CALUDE_max_binomial_probability_l863_86306


namespace NUMINAMATH_CALUDE_sum_of_roots_equation_l863_86391

theorem sum_of_roots_equation (x : ℝ) : 
  (∃ a b : ℝ, (a - 3)^2 = 16 ∧ (b - 3)^2 = 16 ∧ a + b = 6) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_equation_l863_86391


namespace NUMINAMATH_CALUDE_shelby_today_stars_l863_86388

-- Define the variables
def yesterday_stars : ℕ := 4
def total_stars : ℕ := 7

-- State the theorem
theorem shelby_today_stars : 
  total_stars - yesterday_stars = 3 := by
  sorry

end NUMINAMATH_CALUDE_shelby_today_stars_l863_86388


namespace NUMINAMATH_CALUDE_square_kilometer_conversion_time_conversion_l863_86328

-- Define the conversion rates
def sq_km_to_hectares : ℝ := 100
def hour_to_minutes : ℝ := 60

-- Define the problem statements
def problem1 (sq_km : ℝ) (whole_sq_km : ℕ) (hectares : ℕ) : Prop :=
  sq_km = whole_sq_km + hectares / sq_km_to_hectares

def problem2 (hours : ℝ) (whole_hours : ℕ) (minutes : ℕ) : Prop :=
  hours = whole_hours + minutes / hour_to_minutes

-- Theorem statements
theorem square_kilometer_conversion :
  problem1 7.05 7 500 := by sorry

theorem time_conversion :
  problem2 6.7 6 42 := by sorry

end NUMINAMATH_CALUDE_square_kilometer_conversion_time_conversion_l863_86328


namespace NUMINAMATH_CALUDE_binary_to_decimal_conversion_l863_86305

/-- Converts a binary digit to its decimal value -/
def binaryToDecimal (digit : ℕ) (position : ℤ) : ℚ :=
  (digit : ℚ) * (2 : ℚ) ^ position

/-- Represents the binary number 111.11 -/
def binaryNumber : List (ℕ × ℤ) :=
  [(1, 2), (1, 1), (1, 0), (1, -1), (1, -2)]

/-- Theorem: The binary number 111.11 is equal to 7.75 in decimal -/
theorem binary_to_decimal_conversion :
  (binaryNumber.map (fun (digit, position) => binaryToDecimal digit position)).sum = 7.75 := by
  sorry

end NUMINAMATH_CALUDE_binary_to_decimal_conversion_l863_86305


namespace NUMINAMATH_CALUDE_nina_raisins_l863_86395

/-- The number of raisins Nina and Max received satisfies the given conditions -/
def raisin_distribution (nina max : ℕ) : Prop :=
  nina = max + 8 ∧ 
  max = nina / 3 ∧ 
  nina + max = 16

theorem nina_raisins :
  ∀ nina max : ℕ, raisin_distribution nina max → nina = 12 := by
  sorry

end NUMINAMATH_CALUDE_nina_raisins_l863_86395


namespace NUMINAMATH_CALUDE_coffee_table_price_correct_l863_86325

/-- Represents the price of the coffee table -/
def coffee_table_price : ℝ := 429.24

/-- Calculates the total cost before discount and tax -/
def total_before_discount_and_tax (coffee_table_price : ℝ) : ℝ :=
  1250 + 2 * 425 + 350 + 200 + coffee_table_price

/-- Calculates the discounted total -/
def discounted_total (coffee_table_price : ℝ) : ℝ :=
  0.9 * total_before_discount_and_tax coffee_table_price

/-- Calculates the final invoice amount after tax -/
def final_invoice_amount (coffee_table_price : ℝ) : ℝ :=
  1.06 * discounted_total coffee_table_price

/-- Theorem stating that the calculated coffee table price results in the given final invoice amount -/
theorem coffee_table_price_correct :
  final_invoice_amount coffee_table_price = 2937.60 := by
  sorry


end NUMINAMATH_CALUDE_coffee_table_price_correct_l863_86325


namespace NUMINAMATH_CALUDE_midpoint_vector_relation_l863_86394

variable {V : Type*} [AddCommGroup V] [Module ℝ V]
variable (A B C M E : V)

/-- Given a triangle ABC with M as the midpoint of BC and E on AC such that EC = 2AE,
    prove that EM = (1/6)AC - (1/2)AB -/
theorem midpoint_vector_relation 
  (h_midpoint : M = (1/2 : ℝ) • (B + C))
  (h_on_side : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ E = (1 - t) • A + t • C)
  (h_ec_ae : C - E = (2 : ℝ) • (E - A)) :
  E - M = (1/6 : ℝ) • (C - A) - (1/2 : ℝ) • (B - A) := by sorry

end NUMINAMATH_CALUDE_midpoint_vector_relation_l863_86394


namespace NUMINAMATH_CALUDE_gnuff_tutoring_cost_l863_86374

/-- Calculates the total amount paid for a tutoring session -/
def tutoring_cost (flat_rate : ℕ) (per_minute_rate : ℕ) (minutes : ℕ) : ℕ :=
  flat_rate + per_minute_rate * minutes

/-- Theorem: The total amount paid for Gnuff's tutoring session is $146 -/
theorem gnuff_tutoring_cost :
  tutoring_cost 20 7 18 = 146 := by
  sorry

end NUMINAMATH_CALUDE_gnuff_tutoring_cost_l863_86374


namespace NUMINAMATH_CALUDE_mikeys_leaves_theorem_l863_86346

/-- The number of leaves that blew away given initial and remaining leaf counts -/
def leaves_blown_away (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

/-- Theorem stating that 244 leaves blew away given the initial and remaining counts -/
theorem mikeys_leaves_theorem (initial : ℕ) (remaining : ℕ)
  (h1 : initial = 356)
  (h2 : remaining = 112) :
  leaves_blown_away initial remaining = 244 := by
sorry

end NUMINAMATH_CALUDE_mikeys_leaves_theorem_l863_86346


namespace NUMINAMATH_CALUDE_max_friends_theorem_l863_86360

/-- Represents the configuration of gnomes in towers --/
structure GnomeCity (n : ℕ) where
  (n_even : Even n)
  (n_pos : 0 < n)

/-- The maximal number of pairs of gnomes which are friends --/
def max_friends (city : GnomeCity n) : ℕ := n^3 / 4

/-- Theorem stating the maximal number of pairs of gnomes which are friends --/
theorem max_friends_theorem (n : ℕ) (city : GnomeCity n) :
  max_friends city = n^3 / 4 := by sorry

end NUMINAMATH_CALUDE_max_friends_theorem_l863_86360


namespace NUMINAMATH_CALUDE_book_page_sum_problem_l863_86323

theorem book_page_sum_problem :
  ∃ (n : ℕ) (p : ℕ), 
    0 < n ∧ 
    1 ≤ p ∧ 
    p ≤ n ∧ 
    n * (n + 1) / 2 + p = 2550 ∧ 
    p = 65 := by
  sorry

end NUMINAMATH_CALUDE_book_page_sum_problem_l863_86323


namespace NUMINAMATH_CALUDE_sign_determination_l863_86378

theorem sign_determination (a b : ℝ) (h1 : a * b > 0) (h2 : a + b < 0) : a < 0 ∧ b < 0 := by
  sorry

end NUMINAMATH_CALUDE_sign_determination_l863_86378


namespace NUMINAMATH_CALUDE_round_37_259_to_thousandth_l863_86353

/-- Represents a repeating decimal with a whole number part and a repeating fractional part. -/
structure RepeatingDecimal where
  wholePart : ℕ
  repeatingPart : ℕ

/-- Rounds a RepeatingDecimal to the nearest thousandth. -/
def roundToThousandth (x : RepeatingDecimal) : ℚ :=
  sorry

/-- The repeating decimal 37.259259... -/
def num : RepeatingDecimal :=
  { wholePart := 37, repeatingPart := 259 }

theorem round_37_259_to_thousandth :
  roundToThousandth num = 37259 / 1000 :=
sorry

end NUMINAMATH_CALUDE_round_37_259_to_thousandth_l863_86353


namespace NUMINAMATH_CALUDE_plot_perimeter_l863_86380

def rectangular_plot (length width : ℝ) : Prop :=
  length > 0 ∧ width > 0

theorem plot_perimeter (length width : ℝ) :
  rectangular_plot length width →
  length / width = 7 / 5 →
  length * width = 5040 →
  2 * (length + width) = 288 := by
  sorry

end NUMINAMATH_CALUDE_plot_perimeter_l863_86380


namespace NUMINAMATH_CALUDE_tangent_perpendicular_and_inequality_l863_86367

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := ((4*x + a) * log x) / (3*x + 1)

theorem tangent_perpendicular_and_inequality (a : ℝ) :
  (∃ k : ℝ, (deriv (f a) 1 = k) ∧ (k * (-1) = -1)) →
  (∃ m : ℝ, ∀ x ∈ Set.Icc 1 (exp 1), f a x ≤ m * x) →
  (a = 0 ∧ ∀ m : ℝ, (∀ x ∈ Set.Icc 1 (exp 1), f a x ≤ m * x) → m ≥ 4 / (3 * exp 1 + 1)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_perpendicular_and_inequality_l863_86367


namespace NUMINAMATH_CALUDE_f_neg_two_equals_thirteen_l863_86341

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^4 + b * x^2 - x + 1

-- State the theorem
theorem f_neg_two_equals_thirteen (a b : ℝ) (h : f a b 2 = 9) : f a b (-2) = 13 := by
  sorry

end NUMINAMATH_CALUDE_f_neg_two_equals_thirteen_l863_86341


namespace NUMINAMATH_CALUDE_cone_volume_l863_86310

/-- The volume of a cone with base diameter 12 cm and slant height 10 cm is 96π cubic centimeters -/
theorem cone_volume (π : ℝ) (diameter : ℝ) (slant_height : ℝ) : 
  diameter = 12 → slant_height = 10 → 
  (1 / 3) * π * ((diameter / 2) ^ 2) * (Real.sqrt (slant_height ^ 2 - (diameter / 2) ^ 2)) = 96 * π := by
  sorry


end NUMINAMATH_CALUDE_cone_volume_l863_86310


namespace NUMINAMATH_CALUDE_max_x_value_l863_86316

theorem max_x_value : 
  ∃ (x_max : ℝ), 
    (∀ x : ℝ, ((5*x - 20)/(4*x - 5))^2 + ((5*x - 20)/(4*x - 5)) = 20 → x ≤ x_max) ∧
    ((5*x_max - 20)/(4*x_max - 5))^2 + ((5*x_max - 20)/(4*x_max - 5)) = 20 ∧
    x_max = 9/5 :=
by sorry

end NUMINAMATH_CALUDE_max_x_value_l863_86316


namespace NUMINAMATH_CALUDE_train_length_l863_86339

/-- The length of a train given specific conditions -/
theorem train_length (t : ℝ) (v : ℝ) (b : ℝ) (h1 : t = 30) (h2 : v = 45) (h3 : b = 205) :
  v * (1000 / 3600) * t - b = 170 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l863_86339


namespace NUMINAMATH_CALUDE_truth_values_of_p_and_q_l863_86311

theorem truth_values_of_p_and_q (p q : Prop) 
  (h1 : ¬(p ∧ q)) 
  (h2 : ¬(¬p ∨ q)) : 
  p ∧ ¬q := by
  sorry

end NUMINAMATH_CALUDE_truth_values_of_p_and_q_l863_86311


namespace NUMINAMATH_CALUDE_average_marks_combined_classes_l863_86366

theorem average_marks_combined_classes (n1 n2 : ℕ) (avg1 avg2 : ℚ) :
  n1 = 30 →
  n2 = 50 →
  avg1 = 50 →
  avg2 = 60 →
  (n1 : ℚ) * avg1 + (n2 : ℚ) * avg2 = 4500 →
  (n1 + n2 : ℚ) = 80 →
  (n1 : ℚ) * avg1 + (n2 : ℚ) * avg2 / (n1 + n2 : ℚ) = 56.25 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_combined_classes_l863_86366


namespace NUMINAMATH_CALUDE_smallest_non_factor_product_l863_86327

theorem smallest_non_factor_product (a b : ℕ+) : 
  a ≠ b →
  a ∣ 48 →
  b ∣ 48 →
  ¬(a * b ∣ 48) →
  (∀ (c d : ℕ+), c ≠ d → c ∣ 48 → d ∣ 48 → ¬(c * d ∣ 48) → a * b ≤ c * d) →
  a * b = 18 :=
by sorry

end NUMINAMATH_CALUDE_smallest_non_factor_product_l863_86327


namespace NUMINAMATH_CALUDE_reciprocal_equals_self_l863_86319

theorem reciprocal_equals_self (x : ℚ) : x = x⁻¹ ↔ x = 1 ∨ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_equals_self_l863_86319


namespace NUMINAMATH_CALUDE_f_composed_three_roots_l863_86301

/-- A quadratic function f(x) = x^2 + 6x + c -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 + 6*x + c

/-- The composition of f with itself -/
def f_composed (c : ℝ) (x : ℝ) : ℝ := f c (f c x)

/-- Predicate to check if a function has exactly 3 distinct real roots -/
def has_three_distinct_real_roots (g : ℝ → ℝ) : Prop :=
  ∃ (r₁ r₂ r₃ : ℝ), (∀ x : ℝ, g x = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃) ∧ r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃

theorem f_composed_three_roots :
  ∀ c : ℝ, has_three_distinct_real_roots (f_composed c) ↔ c = (11 - Real.sqrt 13) / 2 :=
sorry

end NUMINAMATH_CALUDE_f_composed_three_roots_l863_86301


namespace NUMINAMATH_CALUDE_rhombus_diagonal_l863_86355

/-- Proves that in a rhombus with one diagonal of 40 m and an area of 600 m², 
    the length of the other diagonal is 30 m. -/
theorem rhombus_diagonal (d₁ d₂ : ℝ) (area : ℝ) : 
  d₁ = 40 → area = 600 → area = (d₁ * d₂) / 2 → d₂ = 30 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_l863_86355


namespace NUMINAMATH_CALUDE_special_angle_calculation_l863_86333

theorem special_angle_calculation :
  let tan30 := Real.sqrt 3 / 3
  let cos60 := 1 / 2
  let sin45 := Real.sqrt 2 / 2
  Real.sqrt 3 * tan30 + 2 * cos60 - Real.sqrt 2 * sin45 = 1 := by sorry

end NUMINAMATH_CALUDE_special_angle_calculation_l863_86333


namespace NUMINAMATH_CALUDE_sum_of_even_integers_302_to_400_l863_86392

theorem sum_of_even_integers_302_to_400 (sum_first_50 : ℕ) (sum_302_to_400 : ℕ) : 
  sum_first_50 = 2550 → sum_302_to_400 = 17550 → sum_302_to_400 - sum_first_50 = 15000 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_even_integers_302_to_400_l863_86392


namespace NUMINAMATH_CALUDE_right_handed_players_count_l863_86364

/-- Calculates the total number of right-handed players in a cricket team. -/
theorem right_handed_players_count 
  (total_players : ℕ) 
  (throwers : ℕ) 
  (h1 : total_players = 70)
  (h2 : throwers = 37)
  (h3 : (total_players - throwers) % 3 = 0) -- Ensures non-throwers can be divided into thirds
  (h4 : throwers ≤ total_players) : 
  throwers + ((total_players - throwers) * 2 / 3) = 59 := by
  sorry

#check right_handed_players_count

end NUMINAMATH_CALUDE_right_handed_players_count_l863_86364


namespace NUMINAMATH_CALUDE_square_sum_product_l863_86363

theorem square_sum_product (x : ℝ) : 
  (Real.sqrt (10 + x) + Real.sqrt (30 - x) = 8) → (10 + x) * (30 - x) = 144 := by
sorry

end NUMINAMATH_CALUDE_square_sum_product_l863_86363


namespace NUMINAMATH_CALUDE_sandys_age_l863_86344

theorem sandys_age (sandy_age molly_age : ℕ) 
  (h1 : molly_age = sandy_age + 18) 
  (h2 : sandy_age * 9 = molly_age * 7) : 
  sandy_age = 63 := by
  sorry

end NUMINAMATH_CALUDE_sandys_age_l863_86344


namespace NUMINAMATH_CALUDE_cone_generatrix_length_l863_86336

/-- Given a cone with base radius √2 and lateral surface that unfolds into a semicircle,
    the length of the generatrix is 2√2. -/
theorem cone_generatrix_length (r : ℝ) (l : ℝ) : 
  r = Real.sqrt 2 → 
  2 * Real.pi * r = Real.pi * l → 
  l = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_cone_generatrix_length_l863_86336


namespace NUMINAMATH_CALUDE_congruence_problem_l863_86332

theorem congruence_problem (x : ℤ) 
  (h1 : (4 + x) % 16 = 9)
  (h2 : (6 + x) % 36 = 16)
  (h3 : (8 + x) % 64 = 36) :
  x % 48 = 37 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l863_86332


namespace NUMINAMATH_CALUDE_tv_sales_decrease_l863_86314

theorem tv_sales_decrease (original_price original_quantity : ℝ) 
  (price_increase : ℝ) (revenue_increase : ℝ) (sales_decrease : ℝ) :
  price_increase = 0.4 →
  revenue_increase = 0.12 →
  (1 + price_increase) * (1 - sales_decrease) = 1 + revenue_increase →
  sales_decrease = 0.2 := by
sorry

end NUMINAMATH_CALUDE_tv_sales_decrease_l863_86314


namespace NUMINAMATH_CALUDE_fraction_sum_l863_86347

theorem fraction_sum : 3/8 + 9/12 = 9/8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l863_86347


namespace NUMINAMATH_CALUDE_total_coughs_after_20_minutes_l863_86398

/-- The number of coughs per minute for Georgia -/
def georgia_coughs_per_minute : ℕ := 5

/-- The number of coughs per minute for Robert -/
def robert_coughs_per_minute : ℕ := 2 * georgia_coughs_per_minute

/-- The duration in minutes -/
def duration : ℕ := 20

/-- The total number of coughs after the given duration -/
def total_coughs : ℕ := (georgia_coughs_per_minute + robert_coughs_per_minute) * duration

theorem total_coughs_after_20_minutes :
  total_coughs = 300 := by sorry

end NUMINAMATH_CALUDE_total_coughs_after_20_minutes_l863_86398


namespace NUMINAMATH_CALUDE_final_values_after_assignments_l863_86300

/-- This theorem proves that after a series of assignments, 
    the final values of a and b are both 4. -/
theorem final_values_after_assignments :
  let a₀ : ℕ := 3
  let b₀ : ℕ := 4
  let a₁ : ℕ := b₀
  let b₁ : ℕ := a₁
  (a₁ = 4 ∧ b₁ = 4) :=
by sorry

#check final_values_after_assignments

end NUMINAMATH_CALUDE_final_values_after_assignments_l863_86300


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l863_86307

theorem quadratic_roots_property : ∀ m n : ℝ, 
  (∀ x : ℝ, x^2 - 4*x - 1 = 0 ↔ x = m ∨ x = n) → 
  m + n - m*n = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l863_86307


namespace NUMINAMATH_CALUDE_no_prime_multiple_chain_l863_86340

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def primes_1_to_12 : Set ℕ := {n : ℕ | is_prime n ∧ n ≤ 12}

theorem no_prime_multiple_chain :
  ∀ a b c : ℕ, a ∈ primes_1_to_12 → b ∈ primes_1_to_12 → c ∈ primes_1_to_12 →
  a ≠ b → b ≠ c → a ≠ c →
  ¬(a ∣ b ∧ b ∣ c) :=
sorry

end NUMINAMATH_CALUDE_no_prime_multiple_chain_l863_86340


namespace NUMINAMATH_CALUDE_complex_magnitude_product_l863_86372

theorem complex_magnitude_product : 3 * Complex.abs (1 - 3*I) * Complex.abs (1 + 3*I) = 30 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_product_l863_86372


namespace NUMINAMATH_CALUDE_volume_of_specific_box_l863_86322

/-- The volume of a rectangular box -/
def box_volume (length width height : ℝ) : ℝ := length * width * height

/-- Theorem: The volume of a box with dimensions 20 cm, 15 cm, and 10 cm is 3000 cm³ -/
theorem volume_of_specific_box : box_volume 20 15 10 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_specific_box_l863_86322


namespace NUMINAMATH_CALUDE_spherical_to_rectangular_conversion_l863_86320

theorem spherical_to_rectangular_conversion :
  let ρ : ℝ := 3
  let θ : ℝ := 3 * π / 2
  let φ : ℝ := π / 3
  let x : ℝ := ρ * Real.sin φ * Real.cos θ
  let y : ℝ := ρ * Real.sin φ * Real.sin θ
  let z : ℝ := ρ * Real.cos φ
  (x, y, z) = (0, -3 * Real.sqrt 3 / 2, 3 / 2) :=
by
  sorry


end NUMINAMATH_CALUDE_spherical_to_rectangular_conversion_l863_86320


namespace NUMINAMATH_CALUDE_cos_difference_l863_86356

theorem cos_difference (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 1/2) 
  (h2 : Real.cos A + Real.cos B = 3/2) : 
  Real.cos (A - B) = 1/4 := by
sorry

end NUMINAMATH_CALUDE_cos_difference_l863_86356


namespace NUMINAMATH_CALUDE_chemical_mixture_volume_l863_86308

theorem chemical_mixture_volume : 
  ∀ (initial_volume : ℝ),
  initial_volume > 0 →
  0.30 * initial_volume + 20 = 0.44 * (initial_volume + 20) →
  initial_volume = 80 :=
λ initial_volume h_positive h_equation =>
  sorry

end NUMINAMATH_CALUDE_chemical_mixture_volume_l863_86308


namespace NUMINAMATH_CALUDE_pizza_three_toppings_l863_86381

/-- Represents a pizza with 24 slices and three toppings -/
structure Pizza :=
  (pepperoni : Finset Nat)
  (mushrooms : Finset Nat)
  (olives : Finset Nat)
  (h1 : pepperoni ∪ mushrooms ∪ olives = Finset.range 24)
  (h2 : pepperoni.card = 15)
  (h3 : mushrooms.card = 14)
  (h4 : olives.card = 12)
  (h5 : (pepperoni ∩ mushrooms).card = 6)
  (h6 : (mushrooms ∩ olives).card = 5)
  (h7 : (pepperoni ∩ olives).card = 4)

theorem pizza_three_toppings (p : Pizza) : (p.pepperoni ∩ p.mushrooms ∩ p.olives).card = 0 := by
  sorry

end NUMINAMATH_CALUDE_pizza_three_toppings_l863_86381


namespace NUMINAMATH_CALUDE_distance_and_speed_l863_86331

-- Define the variables
def distance : ℝ := sorry
def speed_second_car : ℝ := sorry
def speed_first_car : ℝ := sorry
def speed_third_car : ℝ := sorry

-- Define the relationships between the speeds
axiom speed_diff_first_second : speed_first_car = speed_second_car + 4
axiom speed_diff_second_third : speed_second_car = speed_third_car + 6

-- Define the time differences
axiom time_diff_first_second : distance / speed_first_car = distance / speed_second_car - 3 / 60
axiom time_diff_second_third : distance / speed_second_car = distance / speed_third_car - 5 / 60

-- Theorem to prove
theorem distance_and_speed : distance = 120 ∧ speed_second_car = 96 := by
  sorry

end NUMINAMATH_CALUDE_distance_and_speed_l863_86331


namespace NUMINAMATH_CALUDE_smallest_x_for_cube_l863_86365

theorem smallest_x_for_cube (x : ℕ+) (N : ℤ) : 
  (∀ y : ℕ+, y < x → ¬∃ M : ℤ, 1890 * y = M^3) ∧ 
  1890 * x = N^3 ↔ 
  x = 4900 := by
sorry

end NUMINAMATH_CALUDE_smallest_x_for_cube_l863_86365


namespace NUMINAMATH_CALUDE_cyclic_inequality_l863_86387

theorem cyclic_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq_3 : a + b + c = 3) : 
  18 * (1 / ((3 - a) * (4 - a)) + 1 / ((3 - b) * (4 - b)) + 1 / ((3 - c) * (4 - c))) + 
  2 * (a * b + b * c + c * a) ≥ 15 := by
sorry


end NUMINAMATH_CALUDE_cyclic_inequality_l863_86387


namespace NUMINAMATH_CALUDE_gunther_tractor_finance_l863_86335

/-- Calculates the total amount financed for a loan with no interest -/
def total_financed (monthly_payment : ℕ) (payment_duration_years : ℕ) : ℕ :=
  monthly_payment * (payment_duration_years * 12)

/-- Theorem stating that for Gunther's tractor loan, the total financed amount is $9000 -/
theorem gunther_tractor_finance :
  total_financed 150 5 = 9000 := by
  sorry

end NUMINAMATH_CALUDE_gunther_tractor_finance_l863_86335


namespace NUMINAMATH_CALUDE_vampire_blood_requirement_l863_86389

-- Define the constants
def pints_per_person : ℕ := 2
def people_per_day : ℕ := 4
def days_per_week : ℕ := 7
def pints_per_gallon : ℕ := 8

-- Define the theorem
theorem vampire_blood_requirement :
  (pints_per_person * people_per_day * days_per_week) / pints_per_gallon = 7 := by
  sorry

end NUMINAMATH_CALUDE_vampire_blood_requirement_l863_86389


namespace NUMINAMATH_CALUDE_stock_value_comparison_l863_86385

def initial_investment : ℝ := 200

def first_year_change_DD : ℝ := 1.10
def first_year_change_EE : ℝ := 0.85
def first_year_change_FF : ℝ := 1.05

def second_year_change_DD : ℝ := 1.05
def second_year_change_EE : ℝ := 1.15
def second_year_change_FF : ℝ := 0.90

def D : ℝ := initial_investment * first_year_change_DD * second_year_change_DD
def E : ℝ := initial_investment * first_year_change_EE * second_year_change_EE
def F : ℝ := initial_investment * first_year_change_FF * second_year_change_FF

theorem stock_value_comparison : F < E ∧ E < D := by
  sorry

end NUMINAMATH_CALUDE_stock_value_comparison_l863_86385


namespace NUMINAMATH_CALUDE_radio_price_proof_l863_86382

theorem radio_price_proof (selling_price : ℝ) (loss_percentage : ℝ) 
  (h1 : selling_price = 465.50)
  (h2 : loss_percentage = 5) : 
  ∃ (original_price : ℝ), 
    original_price = 490 ∧ 
    selling_price = original_price * (1 - loss_percentage / 100) := by
  sorry

end NUMINAMATH_CALUDE_radio_price_proof_l863_86382


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l863_86334

theorem simplify_and_evaluate : 
  let x : ℝ := -2
  (2 * x + 1) * (x - 2) - (2 - x)^2 = -4 := by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l863_86334


namespace NUMINAMATH_CALUDE_combined_cube_volume_l863_86321

theorem combined_cube_volume : 
  let lily_cubes := 4
  let lily_side_length := 3
  let mark_cubes := 3
  let mark_side_length := 4
  let zoe_cubes := 2
  let zoe_side_length := 5
  lily_cubes * lily_side_length^3 + 
  mark_cubes * mark_side_length^3 + 
  zoe_cubes * zoe_side_length^3 = 550 := by
sorry

end NUMINAMATH_CALUDE_combined_cube_volume_l863_86321


namespace NUMINAMATH_CALUDE_complex_fraction_real_l863_86312

theorem complex_fraction_real (a : ℝ) : 
  (((1 : ℂ) + a * I) / ((2 : ℂ) + I)).im = 0 → a = (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_real_l863_86312


namespace NUMINAMATH_CALUDE_frac_5_23_150th_digit_l863_86375

/-- The decimal expansion of 5/23 -/
def decimal_expansion : ℕ → ℕ := sorry

/-- The period of the decimal expansion of 5/23 -/
def period : ℕ := 23

theorem frac_5_23_150th_digit : 
  decimal_expansion ((150 - 1) % period + 1) = 1 := by sorry

end NUMINAMATH_CALUDE_frac_5_23_150th_digit_l863_86375


namespace NUMINAMATH_CALUDE_roots_eccentricity_l863_86370

theorem roots_eccentricity (x₁ x₂ : ℝ) : 
  x₁ * x₂ = 1 → x₁ + x₂ = 79 → (x₁ > 1 ∧ x₂ < 1) ∨ (x₁ < 1 ∧ x₂ > 1) := by
  sorry

end NUMINAMATH_CALUDE_roots_eccentricity_l863_86370


namespace NUMINAMATH_CALUDE_correct_price_reduction_equation_l863_86384

/-- Represents the price reduction of a vehicle over two months -/
def price_reduction (initial_price final_price monthly_rate : ℝ) : Prop :=
  initial_price * (1 - monthly_rate)^2 = final_price

/-- Theorem stating the correct equation for the given scenario -/
theorem correct_price_reduction_equation :
  ∃ x : ℝ, price_reduction 23 18.63 x := by
  sorry

end NUMINAMATH_CALUDE_correct_price_reduction_equation_l863_86384


namespace NUMINAMATH_CALUDE_only_one_valid_assignment_l863_86386

/-- Represents an assignment statement --/
inductive AssignmentStatement
  | Assign (lhs : String) (rhs : String)

/-- Checks if an assignment statement is valid --/
def isValidAssignment (stmt : AssignmentStatement) : Bool :=
  match stmt with
  | AssignmentStatement.Assign lhs rhs => 
    (lhs.all Char.isAlpha) && (rhs ≠ "")

/-- The list of given statements --/
def givenStatements : List AssignmentStatement := [
  AssignmentStatement.Assign "2" "A",
  AssignmentStatement.Assign "x+y" "2",
  AssignmentStatement.Assign "A-B" "-2",
  AssignmentStatement.Assign "A" "A*A"
]

/-- Theorem: Only one of the given statements is a valid assignment --/
theorem only_one_valid_assignment :
  (givenStatements.filter isValidAssignment).length = 1 :=
sorry

end NUMINAMATH_CALUDE_only_one_valid_assignment_l863_86386


namespace NUMINAMATH_CALUDE_parallel_planes_line_parallel_parallel_planes_perpendicular_line_not_always_perpendicular_planes_perpendicular_line_parallel_not_always_perpendicular_planes_line_perpendicular_l863_86351

-- Define the basic types
variable (P : Type) -- Type for planes
variable (L : Type) -- Type for lines

-- Define the relations
variable (parallel_planes : P → P → Prop)
variable (perpendicular_planes : P → P → Prop)
variable (line_in_plane : L → P → Prop)
variable (line_parallel_to_plane : L → P → Prop)
variable (line_perpendicular_to_plane : L → P → Prop)

-- State the theorems
theorem parallel_planes_line_parallel
  (p1 p2 : P) (l : L)
  (h1 : parallel_planes p1 p2)
  (h2 : line_in_plane l p1) :
  line_parallel_to_plane l p2 :=
sorry

theorem parallel_planes_perpendicular_line
  (p1 p2 : P) (l : L)
  (h1 : parallel_planes p1 p2)
  (h2 : line_perpendicular_to_plane l p1) :
  line_perpendicular_to_plane l p2 :=
sorry

theorem not_always_perpendicular_planes_perpendicular_line_parallel
  (p1 p2 : P) (l : L) :
  ¬ (∀ (h1 : perpendicular_planes p1 p2)
      (h2 : line_perpendicular_to_plane l p1),
      line_parallel_to_plane l p2) :=
sorry

theorem not_always_perpendicular_planes_line_perpendicular
  (p1 p2 : P) (l : L) :
  ¬ (∀ (h1 : perpendicular_planes p1 p2)
      (h2 : line_in_plane l p1),
      line_perpendicular_to_plane l p2) :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_line_parallel_parallel_planes_perpendicular_line_not_always_perpendicular_planes_perpendicular_line_parallel_not_always_perpendicular_planes_line_perpendicular_l863_86351


namespace NUMINAMATH_CALUDE_chess_tournament_solution_l863_86393

-- Define the tournament structure
structure ChessTournament where
  grade8_students : ℕ
  grade7_points : ℕ
  grade8_points : ℕ

-- Define the tournament conditions
def valid_tournament (t : ChessTournament) : Prop :=
  t.grade7_points = 8 ∧
  t.grade8_points * t.grade8_students = (t.grade8_students + 2) * (t.grade8_students + 1) / 2 - 8

-- Theorem statement
theorem chess_tournament_solution (t : ChessTournament) :
  valid_tournament t → (t.grade8_students = 7 ∨ t.grade8_students = 14) :=
by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_solution_l863_86393


namespace NUMINAMATH_CALUDE_set_union_implies_a_value_l863_86359

theorem set_union_implies_a_value (a : ℝ) :
  let A : Set ℝ := {2^a, 3}
  let B : Set ℝ := {2, 3}
  A ∪ B = {2, 3, 4} →
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_set_union_implies_a_value_l863_86359


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l863_86357

/-- An arithmetic sequence with given conditions -/
def ArithmeticSequence (a : ℕ → ℤ) : Prop :=
  (∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d) ∧ 
  a 5 = 3 ∧ 
  a 6 = -2

/-- The first term of the sequence -/
def FirstTerm (a : ℕ → ℤ) : ℤ := a 1

/-- The common difference of the sequence -/
def CommonDifference (a : ℕ → ℤ) : ℤ := a 2 - a 1

/-- The general term formula of the sequence -/
def GeneralTerm (a : ℕ → ℤ) (n : ℕ) : ℤ := 28 - 5 * n

theorem arithmetic_sequence_properties (a : ℕ → ℤ) 
  (h : ArithmeticSequence a) : 
  FirstTerm a = 23 ∧ 
  CommonDifference a = -5 ∧ 
  (∀ n : ℕ, a n = GeneralTerm a n) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l863_86357


namespace NUMINAMATH_CALUDE_tv_price_reduction_l863_86349

/-- Proves that a price reduction resulting in a 75% increase in sales and a 31.25% increase in total sale value implies a 25% price reduction -/
theorem tv_price_reduction (P : ℝ) (N : ℝ) (x : ℝ) 
  (h1 : P > 0) 
  (h2 : N > 0) 
  (h3 : x > 0) 
  (h4 : x < 100) 
  (h5 : P * (1 - x / 100) * (N * 1.75) = P * N * 1.3125) : 
  x = 25 := by
sorry

end NUMINAMATH_CALUDE_tv_price_reduction_l863_86349


namespace NUMINAMATH_CALUDE_motorcycle_speed_l863_86352

/-- Motorcycle trip problem -/
theorem motorcycle_speed (total_distance : ℝ) (ab_distance : ℝ) (bc_distance : ℝ)
  (inclination_angle : ℝ) (total_avg_speed : ℝ) (ab_time_ratio : ℝ) :
  total_distance = ab_distance + bc_distance →
  bc_distance = ab_distance / 2 →
  ab_distance = 120 →
  inclination_angle = 10 →
  total_avg_speed = 30 →
  ab_time_ratio = 3 →
  ∃ (bc_avg_speed : ℝ), bc_avg_speed = 40 := by
  sorry

#check motorcycle_speed

end NUMINAMATH_CALUDE_motorcycle_speed_l863_86352


namespace NUMINAMATH_CALUDE_vertices_form_parabola_l863_86361

/-- The set of vertices of a family of parabolas forms another parabola -/
theorem vertices_form_parabola (a c : ℝ) (h_a : a = 2) (h_c : c = 6) :
  ∃ (f : ℝ → ℝ × ℝ),
    (∀ t, f t = (-(t / (2 * a)), a * (-(t / (2 * a)))^2 + t * (-(t / (2 * a))) + c)) ∧
    (∃ (g : ℝ → ℝ), ∀ x, (x, g x) ∈ Set.range f ↔ g x = -a * x^2 + c) :=
by sorry

end NUMINAMATH_CALUDE_vertices_form_parabola_l863_86361


namespace NUMINAMATH_CALUDE_arrangement_count_is_72_l863_86368

/-- Represents the number of ways to arrange 5 people with specific conditions -/
def arrangement_count : ℕ := 72

/-- The number of people in the arrangement -/
def total_people : ℕ := 5

/-- The number of ways to arrange C and D together -/
def cd_arrangements : ℕ := 2

/-- The number of ways to arrange 3 entities (C-D unit, another person, and a space) -/
def entity_arrangements : ℕ := 6

/-- The number of ways to place A and B not adjacent in the remaining spaces -/
def ab_placements : ℕ := 6

/-- Theorem stating that the number of arrangements satisfying the conditions is 72 -/
theorem arrangement_count_is_72 :
  arrangement_count = cd_arrangements * entity_arrangements * ab_placements :=
sorry

end NUMINAMATH_CALUDE_arrangement_count_is_72_l863_86368


namespace NUMINAMATH_CALUDE_system_solution_l863_86377

theorem system_solution (x y z : ℝ) :
  (x^3 = z/y - 2*y/z ∧ y^3 = x/z - 2*z/x ∧ z^3 = y/x - 2*x/y) →
  ((x = 1 ∧ y = 1 ∧ z = -1) ∨
   (x = 1 ∧ y = -1 ∧ z = 1) ∨
   (x = -1 ∧ y = 1 ∧ z = 1) ∨
   (x = -1 ∧ y = -1 ∧ z = -1)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l863_86377


namespace NUMINAMATH_CALUDE_eighth_term_is_one_thirty_second_l863_86309

/-- The sequence defined by a_n = (-1)^n * n / 2^n -/
def a (n : ℕ) : ℚ := (-1)^n * n / 2^n

/-- The 8th term of the sequence is 1/32 -/
theorem eighth_term_is_one_thirty_second : a 8 = 1 / 32 := by
  sorry

end NUMINAMATH_CALUDE_eighth_term_is_one_thirty_second_l863_86309


namespace NUMINAMATH_CALUDE_arithmetic_sequence_minimum_l863_86373

theorem arithmetic_sequence_minimum (a : ℕ → ℝ) (m n : ℕ) :
  (∀ k, a k > 0) →  -- Positive terms
  (∀ k, a (k + 1) = 2 * a k) →  -- Common ratio q = 2
  (Real.sqrt (a m * a n) = 4 * a 1) →  -- Condition on a_m and a_n
  (∃ p q : ℕ, 1/p + 4/q ≤ 1/m + 4/n) →  -- Existence of minimum
  1/m + 4/n ≥ 3/2 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_minimum_l863_86373


namespace NUMINAMATH_CALUDE_abs_neg_2023_eq_2023_l863_86390

theorem abs_neg_2023_eq_2023 : |(-2023 : ℤ)| = 2023 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_2023_eq_2023_l863_86390


namespace NUMINAMATH_CALUDE_midpoint_coordinate_sum_l863_86371

/-- The sum of the coordinates of the midpoint of a line segment with endpoints (8, 16) and (-2, -10) is 6. -/
theorem midpoint_coordinate_sum : 
  let x₁ : ℝ := 8
  let y₁ : ℝ := 16
  let x₂ : ℝ := -2
  let y₂ : ℝ := -10
  let midpoint_x : ℝ := (x₁ + x₂) / 2
  let midpoint_y : ℝ := (y₁ + y₂) / 2
  midpoint_x + midpoint_y = 6 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_sum_l863_86371
