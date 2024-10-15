import Mathlib

namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l3343_334342

theorem inequality_and_equality_condition (x : ℝ) (hx : x ≠ 0) : 
  max 0 (Real.log (abs x)) ≥ 
  ((Real.sqrt 5 - 1) / (2 * Real.sqrt 5)) * Real.log (abs x) + 
  (1 / (2 * Real.sqrt 5)) * Real.log (abs (x^2 - 1)) + 
  (1 / 2) * Real.log ((Real.sqrt 5 + 1) / 2) ∧
  (max 0 (Real.log (abs x)) = 
  ((Real.sqrt 5 - 1) / (2 * Real.sqrt 5)) * Real.log (abs x) + 
  (1 / (2 * Real.sqrt 5)) * Real.log (abs (x^2 - 1)) + 
  (1 / 2) * Real.log ((Real.sqrt 5 + 1) / 2) ↔ 
  (x = (Real.sqrt 5 + 1) / 2 ∨ x = (Real.sqrt 5 - 1) / 2 ∨ 
   x = -(Real.sqrt 5 + 1) / 2 ∨ x = -(Real.sqrt 5 - 1) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l3343_334342


namespace NUMINAMATH_CALUDE_lcm_hcf_product_l3343_334338

theorem lcm_hcf_product (a b : ℕ+) (h1 : Nat.lcm a b = 72) (h2 : a * b = 432) :
  Nat.gcd a b = 6 := by
sorry

end NUMINAMATH_CALUDE_lcm_hcf_product_l3343_334338


namespace NUMINAMATH_CALUDE_dog_grouping_combinations_l3343_334340

def number_of_dogs : ℕ := 12
def group_sizes : List ℕ := [4, 5, 3]

def rocky_group : ℕ := 2
def nipper_group : ℕ := 1
def scruffy_group : ℕ := 0

def remaining_dogs : ℕ := number_of_dogs - 3

theorem dog_grouping_combinations : 
  (remaining_dogs.choose (group_sizes[rocky_group] - 1)) * 
  ((remaining_dogs - (group_sizes[rocky_group] - 1)).choose (group_sizes[scruffy_group] - 1)) = 1260 := by
  sorry

end NUMINAMATH_CALUDE_dog_grouping_combinations_l3343_334340


namespace NUMINAMATH_CALUDE_saturday_to_weekday_ratio_total_weekly_time_correct_total_weekly_time_is_four_hours_l3343_334384

/-- Represents the number of minutes Elle practices piano on different days of the week. -/
structure PracticeTimes where
  weekday : Nat  -- Practice time on each weekday (Monday to Friday)
  saturday : Nat -- Practice time on Saturday
  total_weekly : Nat -- Total practice time in the week

/-- Represents the practice schedule of Elle -/
def elles_practice : PracticeTimes where
  weekday := 30
  saturday := 90
  total_weekly := 240

/-- The ratio of Saturday practice time to weekday practice time is 3:1 -/
theorem saturday_to_weekday_ratio :
  elles_practice.saturday / elles_practice.weekday = 3 := by
  sorry

/-- The total weekly practice time is correct -/
theorem total_weekly_time_correct :
  elles_practice.total_weekly = elles_practice.weekday * 5 + elles_practice.saturday := by
  sorry

/-- The total weekly practice time is 4 hours -/
theorem total_weekly_time_is_four_hours :
  elles_practice.total_weekly = 4 * 60 := by
  sorry

end NUMINAMATH_CALUDE_saturday_to_weekday_ratio_total_weekly_time_correct_total_weekly_time_is_four_hours_l3343_334384


namespace NUMINAMATH_CALUDE_hall_length_l3343_334358

/-- The length of a hall given its width, number of stones, and stone dimensions -/
theorem hall_length (hall_width : ℝ) (num_stones : ℕ) (stone_length stone_width : ℝ) :
  hall_width = 15 ∧ 
  num_stones = 1350 ∧ 
  stone_length = 0.8 ∧ 
  stone_width = 0.5 →
  (stone_length * stone_width * num_stones) / hall_width = 36 :=
by sorry

end NUMINAMATH_CALUDE_hall_length_l3343_334358


namespace NUMINAMATH_CALUDE_second_year_sample_size_l3343_334366

/-- Represents the ratio of students in first, second, and third grades -/
structure GradeRatio where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Calculates the number of students in a specific grade for a stratified sample -/
def stratifiedSampleSize (ratio : GradeRatio) (totalSample : ℕ) (grade : ℕ) : ℕ :=
  (totalSample * grade) / (ratio.first + ratio.second + ratio.third)

/-- Theorem: In a stratified sample of 240 students with a grade ratio of 5:4:3,
    the number of second-year students in the sample is 80 -/
theorem second_year_sample_size :
  let ratio : GradeRatio := { first := 5, second := 4, third := 3 }
  stratifiedSampleSize ratio 240 ratio.second = 80 := by
  sorry

end NUMINAMATH_CALUDE_second_year_sample_size_l3343_334366


namespace NUMINAMATH_CALUDE_fraction_simplification_l3343_334351

theorem fraction_simplification (x : ℝ) (h : x = 3) :
  (x^10 + 15*x^5 + 125) / (x^5 + 5) = 248 + 25/62 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3343_334351


namespace NUMINAMATH_CALUDE_fraction_ordering_l3343_334328

theorem fraction_ordering : 8 / 31 < 11 / 33 ∧ 11 / 33 < 12 / 29 := by
  sorry

end NUMINAMATH_CALUDE_fraction_ordering_l3343_334328


namespace NUMINAMATH_CALUDE_seven_keys_three_adjacent_l3343_334313

/-- The number of distinct arrangements of keys on a keychain. -/
def keychain_arrangements (total_keys : ℕ) (adjacent_keys : ℕ) : ℕ :=
  (adjacent_keys.factorial * ((total_keys - adjacent_keys + 1 - 1).factorial / 2))

/-- Theorem stating the number of distinct arrangements for 7 keys with 3 adjacent -/
theorem seven_keys_three_adjacent :
  keychain_arrangements 7 3 = 72 := by
  sorry

end NUMINAMATH_CALUDE_seven_keys_three_adjacent_l3343_334313


namespace NUMINAMATH_CALUDE_correct_elderly_sample_size_l3343_334301

/-- Represents the number of people in each age group -/
structure Population where
  elderly : ℕ
  middleAged : ℕ
  young : ℕ

/-- Calculates the total population -/
def totalPopulation (p : Population) : ℕ :=
  p.elderly + p.middleAged + p.young

/-- Calculates the number of elderly people to be sampled -/
def elderlySampleSize (p : Population) (sampleSize : ℕ) : ℕ :=
  (p.elderly * sampleSize) / totalPopulation p

theorem correct_elderly_sample_size (p : Population) (sampleSize : ℕ) 
  (h1 : p.elderly = 30)
  (h2 : p.middleAged = 90)
  (h3 : p.young = 60)
  (h4 : sampleSize = 36) :
  elderlySampleSize p sampleSize = 6 := by
  sorry

end NUMINAMATH_CALUDE_correct_elderly_sample_size_l3343_334301


namespace NUMINAMATH_CALUDE_traci_flour_l3343_334334

/-- The amount of flour Harris has in his house -/
def harris_flour : ℕ := 400

/-- The amount of flour needed for each cake -/
def flour_per_cake : ℕ := 100

/-- The number of cakes Traci and Harris created each -/
def cakes_per_person : ℕ := 9

/-- The total number of cakes created -/
def total_cakes : ℕ := 2 * cakes_per_person

/-- The theorem stating the amount of flour Traci brought from her own house -/
theorem traci_flour : 
  harris_flour + (total_cakes * flour_per_cake - harris_flour) = 1400 := by
sorry

end NUMINAMATH_CALUDE_traci_flour_l3343_334334


namespace NUMINAMATH_CALUDE_car_rental_theorem_l3343_334320

/-- Represents a car rental company's pricing model -/
structure CarRental where
  totalVehicles : ℕ
  baseRentalFee : ℕ
  feeIncrement : ℕ
  rentedMaintCost : ℕ
  nonRentedMaintCost : ℕ

/-- Calculates the number of rented vehicles given a rental fee -/
def rentedVehicles (cr : CarRental) (rentalFee : ℕ) : ℕ :=
  cr.totalVehicles - (rentalFee - cr.baseRentalFee) / cr.feeIncrement

/-- Calculates the monthly revenue given a rental fee -/
def monthlyRevenue (cr : CarRental) (rentalFee : ℕ) : ℕ :=
  let rented := rentedVehicles cr rentalFee
  rentalFee * rented - cr.rentedMaintCost * rented - cr.nonRentedMaintCost * (cr.totalVehicles - rented)

/-- The main theorem about the car rental company -/
theorem car_rental_theorem (cr : CarRental) 
    (h1 : cr.totalVehicles = 100)
    (h2 : cr.baseRentalFee = 3000)
    (h3 : cr.feeIncrement = 60)
    (h4 : cr.rentedMaintCost = 160)
    (h5 : cr.nonRentedMaintCost = 60) :
  (rentedVehicles cr 3900 = 85) ∧
  (∃ maxRevenue : ℕ, maxRevenue = 324040 ∧ 
    ∀ fee, monthlyRevenue cr fee ≤ maxRevenue) ∧
  (∃ maxFee : ℕ, maxFee = 4560 ∧
    monthlyRevenue cr maxFee = 324040 ∧
    ∀ fee, monthlyRevenue cr fee ≤ monthlyRevenue cr maxFee) :=
  sorry


end NUMINAMATH_CALUDE_car_rental_theorem_l3343_334320


namespace NUMINAMATH_CALUDE_gain_percent_calculation_l3343_334385

theorem gain_percent_calculation (marked_price : ℝ) (marked_price_positive : marked_price > 0) :
  let cost_price := 0.25 * marked_price
  let discount := 0.5 * marked_price
  let selling_price := marked_price - discount
  let gain := selling_price - cost_price
  let gain_percent := (gain / cost_price) * 100
  gain_percent = 100 := by
sorry

end NUMINAMATH_CALUDE_gain_percent_calculation_l3343_334385


namespace NUMINAMATH_CALUDE_min_value_cubic_function_l3343_334330

/-- A cubic function f(x) = ax³ + bx² + cx + d -/
def cubic_function (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

/-- The function is monotonically increasing on ℝ -/
def monotonically_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

/-- The theorem statement -/
theorem min_value_cubic_function (a b c d : ℝ) :
  a > 0 →
  a < (2/3) * b →
  monotonically_increasing (cubic_function a b c d) →
  (c / (2 * b - 3 * a)) ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_min_value_cubic_function_l3343_334330


namespace NUMINAMATH_CALUDE_product_of_solutions_l3343_334382

-- Define the equation
def equation (x : ℝ) : Prop :=
  (2 * x + 3) / (3 * x + 3) = (5 * x + 4) / (8 * x + 5)

-- Theorem statement
theorem product_of_solutions : 
  ∃ (x₁ x₂ : ℝ), equation x₁ ∧ equation x₂ ∧ x₁ * x₂ = 3 :=
sorry

end NUMINAMATH_CALUDE_product_of_solutions_l3343_334382


namespace NUMINAMATH_CALUDE_train_length_l3343_334331

/-- The length of a train given crossing times -/
theorem train_length (tree_time platform_time platform_length : ℝ) 
  (h1 : tree_time = 120)
  (h2 : platform_time = 180)
  (h3 : platform_length = 600)
  (h4 : tree_time > 0)
  (h5 : platform_time > 0)
  (h6 : platform_length > 0) :
  (platform_time * platform_length) / (platform_time - tree_time) = 1200 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l3343_334331


namespace NUMINAMATH_CALUDE_bird_sanctuary_geese_percentage_l3343_334306

theorem bird_sanctuary_geese_percentage :
  let total_percentage : ℚ := 100
  let geese_percentage : ℚ := 40
  let swan_percentage : ℚ := 20
  let heron_percentage : ℚ := 15
  let duck_percentage : ℚ := 25
  let non_duck_percentage : ℚ := total_percentage - duck_percentage
  geese_percentage / non_duck_percentage * 100 = 53 + 1/3 :=
by sorry

end NUMINAMATH_CALUDE_bird_sanctuary_geese_percentage_l3343_334306


namespace NUMINAMATH_CALUDE_two_digit_sum_property_l3343_334359

def is_valid_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧
  ∃ (x y : ℕ),
    n = 10 * x + y ∧
    x < 10 ∧ y < 10 ∧
    (x + 1 + y + 2 - 10) / 2 = x + y ∧
    y + 2 ≥ 10

theorem two_digit_sum_property :
  ∀ n : ℕ, is_valid_number n ↔ (n = 68 ∨ n = 59) :=
sorry

end NUMINAMATH_CALUDE_two_digit_sum_property_l3343_334359


namespace NUMINAMATH_CALUDE_bank_account_increase_percentage_l3343_334354

def al_initial_balance : ℝ := 236.36
def eliot_initial_balance : ℝ := 200

theorem bank_account_increase_percentage :
  (al_initial_balance > eliot_initial_balance) →
  (al_initial_balance - eliot_initial_balance = (al_initial_balance + eliot_initial_balance) / 12) →
  (∃ p : ℝ, (al_initial_balance * 1.1 = eliot_initial_balance * (1 + p / 100) + 20) ∧ p = 20) :=
by sorry

end NUMINAMATH_CALUDE_bank_account_increase_percentage_l3343_334354


namespace NUMINAMATH_CALUDE_ferris_wheel_capacity_is_120_l3343_334314

/-- Calculates the maximum number of people who can ride a Ferris wheel simultaneously -/
def max_ferris_wheel_capacity (total_seats : ℕ) (people_per_seat : ℕ) (broken_seats : ℕ) : ℕ :=
  (total_seats - broken_seats) * people_per_seat

/-- Proves that the maximum capacity of the Ferris wheel under given conditions is 120 people -/
theorem ferris_wheel_capacity_is_120 :
  max_ferris_wheel_capacity 18 15 10 = 120 := by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_capacity_is_120_l3343_334314


namespace NUMINAMATH_CALUDE_symmetric_circle_equation_l3343_334370

/-- The equation of a circle symmetric to another circle with respect to the line y = x -/
theorem symmetric_circle_equation (x y : ℝ) : 
  (∃ (a b r : ℝ), (x + a)^2 + (y - b)^2 = r^2) →  -- Original circle
  (∃ (c d : ℝ), (x - c)^2 + (y + d)^2 = 5) :=     -- Symmetric circle
by sorry

end NUMINAMATH_CALUDE_symmetric_circle_equation_l3343_334370


namespace NUMINAMATH_CALUDE_quadratic_equation_translation_l3343_334364

/-- Quadratic form in two variables -/
def Q (a b c : ℝ) (x y : ℝ) : ℝ := a * x^2 + 2 * b * x * y + c * y^2

/-- Theorem: Transformation of quadratic equation using parallel translation -/
theorem quadratic_equation_translation
  (a b c d e f x₀ y₀ : ℝ)
  (h : a * c - b^2 ≠ 0) :
  ∃ f' : ℝ,
    (∀ x y, Q a b c x y + 2 * d * x + 2 * e * y = f) ↔
    (∀ x' y', Q a b c x' y' = f' ∧
      x' = x + x₀ ∧
      y' = y + y₀ ∧
      f' = f - Q a b c x₀ y₀ + 2 * (d * x₀ + e * y₀)) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_translation_l3343_334364


namespace NUMINAMATH_CALUDE_complex_number_location_l3343_334304

theorem complex_number_location :
  let z : ℂ := 1 / ((1 + Complex.I)^2 + 1)
  (z.re < 0) ∧ (z.im > 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_location_l3343_334304


namespace NUMINAMATH_CALUDE_museum_visitor_ratio_l3343_334369

/-- Represents the number of adults and children visiting a museum. -/
structure MuseumVisitors where
  adults : ℕ
  children : ℕ

/-- Calculates the total admission fee for a given number of adults and children. -/
def admissionFee (visitors : MuseumVisitors) : ℕ :=
  30 * visitors.adults + 15 * visitors.children

/-- Checks if the number of visitors satisfies the minimum requirement. -/
def satisfiesMinimum (visitors : MuseumVisitors) : Prop :=
  visitors.adults ≥ 2 ∧ visitors.children ≥ 2

/-- Calculates the ratio of adults to children. -/
def visitorRatio (visitors : MuseumVisitors) : ℚ :=
  visitors.adults / visitors.children

theorem museum_visitor_ratio :
  ∃ (visitors : MuseumVisitors),
    satisfiesMinimum visitors ∧
    admissionFee visitors = 2700 ∧
    visitorRatio visitors = 2 ∧
    (∀ (other : MuseumVisitors),
      satisfiesMinimum other →
      admissionFee other = 2700 →
      |visitorRatio other - 2| ≥ |visitorRatio visitors - 2|) := by
  sorry

end NUMINAMATH_CALUDE_museum_visitor_ratio_l3343_334369


namespace NUMINAMATH_CALUDE_at_least_seven_zeros_l3343_334335

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem at_least_seven_zeros (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_period : has_period f 3) 
  (h_zero : f 2 = 0) : 
  ∃ (S : Finset ℝ), S.card ≥ 7 ∧ (∀ x ∈ S, 0 < x ∧ x < 6 ∧ f x = 0) :=
sorry

end NUMINAMATH_CALUDE_at_least_seven_zeros_l3343_334335


namespace NUMINAMATH_CALUDE_michelle_racks_l3343_334356

/-- The number of drying racks Michelle owns -/
def current_racks : ℕ := 3

/-- The number of pounds of pasta per drying rack -/
def pasta_per_rack : ℕ := 3

/-- The number of cups of flour needed to make one pound of pasta -/
def flour_per_pound : ℕ := 2

/-- The number of cups in each bag of flour -/
def cups_per_bag : ℕ := 8

/-- The number of bags of flour Michelle has -/
def num_bags : ℕ := 3

/-- The total number of cups of flour Michelle has -/
def total_flour : ℕ := num_bags * cups_per_bag

/-- The total pounds of pasta Michelle can make -/
def total_pasta : ℕ := total_flour / flour_per_pound

/-- The number of racks needed for all the pasta Michelle can make -/
def racks_needed : ℕ := total_pasta / pasta_per_rack

theorem michelle_racks :
  current_racks = racks_needed - 1 :=
sorry

end NUMINAMATH_CALUDE_michelle_racks_l3343_334356


namespace NUMINAMATH_CALUDE_expression_is_equation_l3343_334388

-- Define what an equation is
def is_equation (e : Prop) : Prop :=
  ∃ (lhs rhs : ℝ → ℝ → ℝ), e = (∀ r y, lhs r y = rhs r y)

-- Define the expression we want to prove is an equation
def expression (r y : ℝ) : ℝ := 3 * r + y

-- Theorem statement
theorem expression_is_equation :
  is_equation (∀ r y : ℝ, expression r y = 5) :=
sorry

end NUMINAMATH_CALUDE_expression_is_equation_l3343_334388


namespace NUMINAMATH_CALUDE_max_value_AMC_l3343_334386

theorem max_value_AMC (A M C : ℕ) (sum_constraint : A + M + C = 10) :
  (∀ A' M' C' : ℕ, A' + M' + C' = 10 → 
    A' * M' * C' + A' * M' + M' * C' + C' * A' ≤ A * M * C + A * M + M * C + C * A) →
  A * M * C + A * M + M * C + C * A = 69 := by
  sorry

end NUMINAMATH_CALUDE_max_value_AMC_l3343_334386


namespace NUMINAMATH_CALUDE_disjoint_subsets_bound_l3343_334309

theorem disjoint_subsets_bound (m : ℕ) (A B : Finset ℕ) : 
  A ⊆ Finset.range m → 
  B ⊆ Finset.range m → 
  A ∩ B = ∅ → 
  A.sum id = B.sum id → 
  (A.card : ℝ) < m / Real.sqrt 2 ∧ (B.card : ℝ) < m / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_disjoint_subsets_bound_l3343_334309


namespace NUMINAMATH_CALUDE_not_power_of_integer_l3343_334343

theorem not_power_of_integer (m : ℕ) : ¬ ∃ (n k : ℕ), m * (m + 1) = n^k := by
  sorry

end NUMINAMATH_CALUDE_not_power_of_integer_l3343_334343


namespace NUMINAMATH_CALUDE_pencil_total_length_l3343_334362

-- Define the pencil sections
def purple_length : ℝ := 3
def black_length : ℝ := 2
def blue_length : ℝ := 1

-- Theorem statement
theorem pencil_total_length : 
  purple_length + black_length + blue_length = 6 := by sorry

end NUMINAMATH_CALUDE_pencil_total_length_l3343_334362


namespace NUMINAMATH_CALUDE_abc_inequality_l3343_334312

theorem abc_inequality (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum : a + b + c = 1) : 
  (a * b + b * c + c * a ≤ 1/3) ∧ 
  (a^2 / (b + c) + b^2 / (c + a) + c^2 / (a + b) ≥ 1/2) := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l3343_334312


namespace NUMINAMATH_CALUDE_fraction_repeating_block_length_l3343_334391

/-- The length of the repeating block in the decimal expansion of 7/13 -/
def repeatingBlockLength : ℕ := 6

/-- The fraction we're considering -/
def fraction : ℚ := 7/13

theorem fraction_repeating_block_length :
  ∃ (d : ℕ+) (n : ℕ), 
    fraction * d.val = n ∧ 
    d = 10^repeatingBlockLength - 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_repeating_block_length_l3343_334391


namespace NUMINAMATH_CALUDE_prize_pricing_and_quantity_l3343_334311

/-- Represents the price of a type A prize -/
def price_A : ℕ := sorry

/-- Represents the price of a type B prize -/
def price_B : ℕ := sorry

/-- The cost of one type A prize and two type B prizes -/
def cost_combination1 : ℕ := 220

/-- The cost of two type A prizes and three type B prizes -/
def cost_combination2 : ℕ := 360

/-- The total number of prizes to be purchased -/
def total_prizes : ℕ := 30

/-- The maximum total cost allowed -/
def max_total_cost : ℕ := 2300

/-- The minimum number of type A prizes that can be purchased -/
def min_type_A_prizes : ℕ := sorry

theorem prize_pricing_and_quantity :
  (price_A + 2 * price_B = cost_combination1) ∧
  (2 * price_A + 3 * price_B = cost_combination2) ∧
  (price_A = 60) ∧
  (price_B = 80) ∧
  (∀ m : ℕ, m ≥ min_type_A_prizes →
    price_A * m + price_B * (total_prizes - m) ≤ max_total_cost) ∧
  (min_type_A_prizes = 5) := by sorry

end NUMINAMATH_CALUDE_prize_pricing_and_quantity_l3343_334311


namespace NUMINAMATH_CALUDE_min_keys_required_l3343_334365

/-- Represents the hotel key distribution problem -/
structure HotelKeyProblem where
  rooms : ℕ
  guests : ℕ
  returningGuests : ℕ
  keys : ℕ

/-- Predicate to check if a key distribution is valid -/
def isValidDistribution (p : HotelKeyProblem) : Prop :=
  p.rooms > 0 ∧ 
  p.guests > p.rooms ∧ 
  p.returningGuests = p.rooms ∧
  ∀ (subset : Finset ℕ), subset.card = p.returningGuests → 
    ∃ (f : subset → Fin p.rooms), Function.Injective f

/-- Theorem stating the minimum number of keys required -/
theorem min_keys_required (p : HotelKeyProblem) 
  (h : isValidDistribution p) : p.keys ≥ p.rooms * (p.guests - p.rooms + 1) :=
sorry

end NUMINAMATH_CALUDE_min_keys_required_l3343_334365


namespace NUMINAMATH_CALUDE_unique_right_triangle_exists_l3343_334315

theorem unique_right_triangle_exists : ∃! (a : ℝ), 
  a > 0 ∧ 
  let b := 2 * a
  let c := Real.sqrt (a^2 + b^2)
  (a + b + c) - (1/2 * a * b) = c :=
by sorry

end NUMINAMATH_CALUDE_unique_right_triangle_exists_l3343_334315


namespace NUMINAMATH_CALUDE_shop_equations_correct_l3343_334333

/-- Represents a shop with rooms and guests -/
structure Shop where
  rooms : ℕ
  guests : ℕ

/-- The system of equations for the shop problem -/
def shop_equations (s : Shop) : Prop :=
  (7 * s.rooms + 7 = s.guests) ∧ (9 * (s.rooms - 1) = s.guests)

/-- Theorem stating that the shop equations correctly represent the given conditions -/
theorem shop_equations_correct (s : Shop) :
  (∀ (r : ℕ), r * s.rooms + 7 = s.guests → r = 7) ∧
  (∀ (r : ℕ), r * (s.rooms - 1) = s.guests → r = 9) →
  shop_equations s :=
sorry

end NUMINAMATH_CALUDE_shop_equations_correct_l3343_334333


namespace NUMINAMATH_CALUDE_simple_interest_calculation_l3343_334319

/-- Simple interest calculation -/
theorem simple_interest_calculation
  (principal : ℝ)
  (rate : ℝ)
  (time : ℝ)
  (h1 : principal = 780)
  (h2 : rate = 4.166666666666667 / 100)
  (h3 : time = 4) :
  principal * rate * time = 130 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_calculation_l3343_334319


namespace NUMINAMATH_CALUDE_hundredth_term_is_401_l3343_334374

/-- 
Represents a sequence of toothpicks where:
- The first term is 5
- Each subsequent term increases by 4
-/
def toothpick_sequence (n : ℕ) : ℕ := 5 + 4 * (n - 1)

/-- 
Theorem: The 100th term of the toothpick sequence is 401
-/
theorem hundredth_term_is_401 : toothpick_sequence 100 = 401 := by
  sorry

end NUMINAMATH_CALUDE_hundredth_term_is_401_l3343_334374


namespace NUMINAMATH_CALUDE_sticker_ratio_l3343_334376

/-- Proves that the ratio of Dan's stickers to Tom's stickers is 2:1 -/
theorem sticker_ratio :
  let bob_stickers : ℕ := 12
  let tom_stickers : ℕ := 3 * bob_stickers
  let dan_stickers : ℕ := 72
  (dan_stickers : ℚ) / tom_stickers = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_sticker_ratio_l3343_334376


namespace NUMINAMATH_CALUDE_sum_of_fifth_powers_l3343_334317

theorem sum_of_fifth_powers (α β γ : ℂ) 
  (h1 : α + β + γ = 2)
  (h2 : α^2 + β^2 + γ^2 = 5)
  (h3 : α^3 + β^3 + γ^3 = 8) :
  α^5 + β^5 + γ^5 = 46.5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fifth_powers_l3343_334317


namespace NUMINAMATH_CALUDE_adrian_holidays_in_year_l3343_334323

/-- The number of holidays Adrian took in a year -/
def adriansHolidays (daysOffPerMonth : ℕ) (monthsInYear : ℕ) : ℕ :=
  daysOffPerMonth * monthsInYear

/-- Theorem: Adrian took 48 holidays in the entire year -/
theorem adrian_holidays_in_year : 
  adriansHolidays 4 12 = 48 := by
  sorry

end NUMINAMATH_CALUDE_adrian_holidays_in_year_l3343_334323


namespace NUMINAMATH_CALUDE_quadratic_polynomials_inequalities_l3343_334381

/-- Given three quadratic polynomials with the specified properties, 
    exactly two out of three inequalities are satisfied. -/
theorem quadratic_polynomials_inequalities 
  (a b c d e f : ℝ) 
  (h1 : ∃ x : ℝ, (x^2 + a*x + b = 0 ∧ x^2 + c*x + d = 0) ∨ 
                 (x^2 + a*x + b = 0 ∧ x^2 + e*x + f = 0) ∨ 
                 (x^2 + c*x + d = 0 ∧ x^2 + e*x + f = 0))
  (h2 : ¬ ∃ x : ℝ, x^2 + a*x + b = 0 ∧ x^2 + c*x + d = 0 ∧ x^2 + e*x + f = 0) :
  (((a^2 + c^2 - e^2)/4 > b + d - f) ∧ 
   ((c^2 + e^2 - a^2)/4 > d + f - b) ∧ 
   ((e^2 + a^2 - c^2)/4 ≤ f + b - d)) ∨
  (((a^2 + c^2 - e^2)/4 > b + d - f) ∧ 
   ((c^2 + e^2 - a^2)/4 ≤ d + f - b) ∧ 
   ((e^2 + a^2 - c^2)/4 > f + b - d)) ∨
  (((a^2 + c^2 - e^2)/4 ≤ b + d - f) ∧ 
   ((c^2 + e^2 - a^2)/4 > d + f - b) ∧ 
   ((e^2 + a^2 - c^2)/4 > f + b - d)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomials_inequalities_l3343_334381


namespace NUMINAMATH_CALUDE_min_seating_arrangement_l3343_334347

/-- Given a circular table with 60 chairs, this theorem proves that the smallest number of people
    that can be seated such that any additional person must sit next to someone is 20. -/
theorem min_seating_arrangement (n : ℕ) : n = 20 ↔ (
  n ≤ 60 ∧ 
  (∀ m : ℕ, m < n → ∃ (arrangement : Fin 60 → Bool), 
    (∃ i : Fin 60, ¬arrangement i) ∧ 
    (∀ i : Fin 60, arrangement i → 
      (arrangement (i + 1) ∨ arrangement (i + 59)))) ∧
  (∀ m : ℕ, m > n → ¬∃ (arrangement : Fin 60 → Bool),
    (∀ i : Fin 60, arrangement i → 
      (arrangement (i + 1) ∨ arrangement (i + 59)))))
  := by sorry


end NUMINAMATH_CALUDE_min_seating_arrangement_l3343_334347


namespace NUMINAMATH_CALUDE_switches_in_position_A_after_process_l3343_334357

/-- Represents a switch position -/
inductive Position
| A | B | C | D | E

/-- Advances a position cyclically -/
def advance_position (p : Position) : Position :=
  match p with
  | Position.A => Position.B
  | Position.B => Position.C
  | Position.C => Position.D
  | Position.D => Position.E
  | Position.E => Position.A

/-- Represents a switch with its label and position -/
structure Switch :=
  (x y z w : Nat)
  (pos : Position)

/-- The total number of switches -/
def total_switches : Nat := 6860

/-- Creates the initial set of switches -/
def initial_switches : Finset Switch :=
  sorry

/-- Advances a switch and its divisors -/
def advance_switches (switches : Finset Switch) (i : Nat) : Finset Switch :=
  sorry

/-- Performs the entire 6860-step process -/
def process (switches : Finset Switch) : Finset Switch :=
  sorry

/-- Counts switches in position A -/
def count_position_A (switches : Finset Switch) : Nat :=
  sorry

/-- The main theorem to prove -/
theorem switches_in_position_A_after_process :
  count_position_A (process initial_switches) = 6455 :=
sorry

end NUMINAMATH_CALUDE_switches_in_position_A_after_process_l3343_334357


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l3343_334360

/-- Given a geometric sequence with first term b₁ and common ratio q,
    T_n represents the product of the first n terms. -/
def T (b₁ q : ℝ) (n : ℕ) : ℝ :=
  b₁^n * q^(n * (n - 1) / 2)

/-- Theorem: For a geometric sequence, T_4, T_8/T_4, T_12/T_8, T_16/T_12 form a geometric sequence -/
theorem geometric_sequence_property (b₁ q : ℝ) (b₁_pos : 0 < b₁) (q_pos : 0 < q) :
  ∃ r : ℝ, r ≠ 0 ∧
    (T b₁ q 8 / T b₁ q 4) = (T b₁ q 4) * r ∧
    (T b₁ q 12 / T b₁ q 8) = (T b₁ q 8 / T b₁ q 4) * r ∧
    (T b₁ q 16 / T b₁ q 12) = (T b₁ q 12 / T b₁ q 8) * r :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l3343_334360


namespace NUMINAMATH_CALUDE_AP_coordinates_l3343_334325

-- Define the vectors OA and OB
def OA : ℝ × ℝ × ℝ := (1, -1, 1)
def OB : ℝ × ℝ × ℝ := (2, 0, -1)

-- Define point P on line segment AB
def P : ℝ × ℝ × ℝ := sorry

-- Define the condition AP = 2PB
def AP_eq_2PB : ∃ (t : ℝ), 0 < t ∧ t < 1 ∧ P = (1 - t) • OA + t • OB ∧ t = 2/3 := sorry

-- Theorem to prove
theorem AP_coordinates : 
  let AP := P - OA
  AP = (2/3, 2/3, -4/3) :=
sorry

end NUMINAMATH_CALUDE_AP_coordinates_l3343_334325


namespace NUMINAMATH_CALUDE_ellipse_theorem_l3343_334324

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a line y = kx + m -/
structure Line where
  k : ℝ
  m : ℝ

/-- The main theorem about the ellipse and the range of |OP| -/
theorem ellipse_theorem (C : Ellipse) (l : Line) : 
  C.a^2 = 4 ∧ C.b^2 = 2 ∧ abs l.k ≤ Real.sqrt 2 / 2 →
  (∀ x y : ℝ, x^2 / C.a^2 + y^2 / C.b^2 = 1 ↔ x^2 / 4 + y^2 / 2 = 1) ∧
  (∀ P : ℝ × ℝ, P.1^2 / 4 + P.2^2 / 2 = 1 → 
    Real.sqrt 2 ≤ Real.sqrt (P.1^2 + P.2^2) ∧ 
    Real.sqrt (P.1^2 + P.2^2) ≤ Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_theorem_l3343_334324


namespace NUMINAMATH_CALUDE_arithmetic_sequence_difference_l3343_334371

def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_difference (a : ℕ → ℝ) (d : ℝ) :
  is_arithmetic_sequence a d →
  (3 * a 6 = a 3 + a 4 + a 5 + 6) →
  d = 1 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_difference_l3343_334371


namespace NUMINAMATH_CALUDE_abs_inequality_l3343_334327

theorem abs_inequality (a_n e : ℝ) (h : |a_n - e| < 1) : |a_n| < |e| + 1 := by
  sorry

end NUMINAMATH_CALUDE_abs_inequality_l3343_334327


namespace NUMINAMATH_CALUDE_container_volume_ratio_l3343_334392

theorem container_volume_ratio (V₁ V₂ V₃ : ℝ) 
  (h₁ : V₁ > 0) (h₂ : V₂ > 0) (h₃ : V₃ > 0)
  (h₄ : (3/4) * V₁ = (5/8) * V₂)
  (h₅ : (5/8) * V₂ = (1/2) * V₃) :
  V₁ / V₃ = 1/2 := by
sorry

end NUMINAMATH_CALUDE_container_volume_ratio_l3343_334392


namespace NUMINAMATH_CALUDE_house_cost_is_480000_l3343_334373

/-- Calculates the cost of a house given the following conditions:
  - A trailer costs $120,000
  - Each loan will be paid in monthly installments over 20 years
  - The monthly payment on the house is $1500 more than the trailer
-/
def house_cost (trailer_cost : ℕ) (loan_years : ℕ) (monthly_difference : ℕ) : ℕ :=
  let months : ℕ := loan_years * 12
  let trailer_monthly : ℕ := trailer_cost / months
  let house_monthly : ℕ := trailer_monthly + monthly_difference
  house_monthly * months

/-- Theorem stating that the cost of the house is $480,000 -/
theorem house_cost_is_480000 :
  house_cost 120000 20 1500 = 480000 := by
  sorry

end NUMINAMATH_CALUDE_house_cost_is_480000_l3343_334373


namespace NUMINAMATH_CALUDE_circle_properties_l3343_334302

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x - 1)^2 + (y + 1)^2 = 2

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x + y - 2 = 0

-- Theorem statement
theorem circle_properties :
  -- The circle passes through the origin
  circle_equation 0 0 ∧
  -- The circle contains the point (2,0)
  circle_equation 2 0 ∧
  -- The line contains the point (2,0)
  line_equation 2 0 ∧
  -- The circle is tangent to the line at (2,0)
  ∃ (t : ℝ), t ≠ 0 ∧
    ∀ (x y : ℝ),
      circle_equation x y ∧ line_equation x y →
      x = 2 ∧ y = 0 :=
sorry

end NUMINAMATH_CALUDE_circle_properties_l3343_334302


namespace NUMINAMATH_CALUDE_paul_books_left_l3343_334337

/-- The number of books Paul had left after the garage sale -/
def books_left (initial_books : ℕ) (books_sold : ℕ) : ℕ :=
  initial_books - books_sold

/-- Theorem stating that Paul had 66 books left after the garage sale -/
theorem paul_books_left : books_left 108 42 = 66 := by
  sorry

end NUMINAMATH_CALUDE_paul_books_left_l3343_334337


namespace NUMINAMATH_CALUDE_sqrt_9801_minus_99_proof_l3343_334326

theorem sqrt_9801_minus_99_proof (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) 
  (h3 : Real.sqrt 9801 - 99 = (Real.sqrt a - b)^3) : a + b = 11 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_9801_minus_99_proof_l3343_334326


namespace NUMINAMATH_CALUDE_barn_paint_area_l3343_334318

/-- Represents the dimensions of a rectangular barn -/
structure BarnDimensions where
  width : ℝ
  length : ℝ
  height : ℝ

/-- Calculates the total area to be painted in a barn with a dividing wall -/
def totalPaintArea (d : BarnDimensions) : ℝ :=
  let externalWallArea := 2 * (d.width * d.height + d.length * d.height)
  let dividingWallArea := 2 * (d.width * d.height)
  let ceilingArea := d.width * d.length
  2 * externalWallArea + dividingWallArea + ceilingArea

/-- The dimensions of the barn in the problem -/
def problemBarn : BarnDimensions :=
  { width := 12
  , length := 15
  , height := 5 }

theorem barn_paint_area :
  totalPaintArea problemBarn = 840 := by
  sorry


end NUMINAMATH_CALUDE_barn_paint_area_l3343_334318


namespace NUMINAMATH_CALUDE_sin_360_degrees_l3343_334389

theorem sin_360_degrees : Real.sin (2 * Real.pi) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_360_degrees_l3343_334389


namespace NUMINAMATH_CALUDE_blanket_warmth_l3343_334345

theorem blanket_warmth (total_blankets : ℕ) (used_fraction : ℚ) (total_warmth : ℕ) : 
  total_blankets = 14 →
  used_fraction = 1/2 →
  total_warmth = 21 →
  (total_warmth : ℚ) / (used_fraction * total_blankets) = 3 := by
  sorry

end NUMINAMATH_CALUDE_blanket_warmth_l3343_334345


namespace NUMINAMATH_CALUDE_A_eq_zero_two_l3343_334397

/-- The set of real numbers a for which the equation ax^2 - 4x + 2 = 0 has exactly one solution -/
def A : Set ℝ :=
  {a : ℝ | ∃! x : ℝ, a * x^2 - 4 * x + 2 = 0}

/-- Theorem stating that A is equal to the set {0, 2} -/
theorem A_eq_zero_two : A = {0, 2} := by
  sorry

end NUMINAMATH_CALUDE_A_eq_zero_two_l3343_334397


namespace NUMINAMATH_CALUDE_probability_neither_red_nor_purple_l3343_334305

theorem probability_neither_red_nor_purple :
  let total_balls : ℕ := 120
  let red_balls : ℕ := 15
  let purple_balls : ℕ := 3
  let neither_red_nor_purple : ℕ := total_balls - (red_balls + purple_balls)
  (neither_red_nor_purple : ℚ) / total_balls = 17 / 20 := by
  sorry

end NUMINAMATH_CALUDE_probability_neither_red_nor_purple_l3343_334305


namespace NUMINAMATH_CALUDE_carlas_order_cost_l3343_334348

/-- Calculates the final cost of Carla's order at McDonald's --/
theorem carlas_order_cost (base_cost : ℝ) (coupon_discount : ℝ) (senior_discount_rate : ℝ) (swap_charge : ℝ)
  (h1 : base_cost = 7.5)
  (h2 : coupon_discount = 2.5)
  (h3 : senior_discount_rate = 0.2)
  (h4 : swap_charge = 1.0) :
  base_cost - coupon_discount - (base_cost - coupon_discount) * senior_discount_rate + swap_charge = 5 :=
by sorry


end NUMINAMATH_CALUDE_carlas_order_cost_l3343_334348


namespace NUMINAMATH_CALUDE_pythagorean_inequality_l3343_334361

theorem pythagorean_inequality (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c)
  (h4 : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h5 : a^2 + b^2 = c^2 + a*b) :
  c^2 + a*b < a*c + b*c := by
  sorry

end NUMINAMATH_CALUDE_pythagorean_inequality_l3343_334361


namespace NUMINAMATH_CALUDE_price_after_discounts_l3343_334394

def initial_price : Float := 9649.12
def discount1 : Float := 0.20
def discount2 : Float := 0.10
def discount3 : Float := 0.05

def apply_discount (price : Float) (discount : Float) : Float :=
  price * (1 - discount)

def final_price : Float :=
  apply_discount (apply_discount (apply_discount initial_price discount1) discount2) discount3

theorem price_after_discounts :
  final_price = 6600.09808 := by sorry

end NUMINAMATH_CALUDE_price_after_discounts_l3343_334394


namespace NUMINAMATH_CALUDE_circle_diameter_l3343_334344

theorem circle_diameter (AE EB ED : ℝ) (h1 : AE = 2) (h2 : EB = 6) (h3 : ED = 3) :
  let AB := AE + EB
  let CE := (AE * EB) / ED
  let AM := (AB) / 2
  let OM := (AE + EB) / 2
  let OA := Real.sqrt (AM^2 + OM^2)
  let diameter := 2 * OA
  diameter = Real.sqrt 65 := by sorry

end NUMINAMATH_CALUDE_circle_diameter_l3343_334344


namespace NUMINAMATH_CALUDE_equality_proof_l3343_334395

theorem equality_proof (x y : ℤ) : 
  (x - 1) * (x + 4) * (x - 3) - (x + 1) * (x - 4) * (x + 3) = 
  (y - 1) * (y + 4) * (y - 3) - (y + 1) * (y - 4) * (y + 3) := by
  sorry

end NUMINAMATH_CALUDE_equality_proof_l3343_334395


namespace NUMINAMATH_CALUDE_final_value_of_A_l3343_334378

theorem final_value_of_A : ∀ (A : ℤ), A = 15 → -A + 5 = -10 := by
  sorry

end NUMINAMATH_CALUDE_final_value_of_A_l3343_334378


namespace NUMINAMATH_CALUDE_fraction_comparison_l3343_334398

theorem fraction_comparison : 
  (10^1966 + 1) / (10^1967 + 1) > (10^1967 + 1) / (10^1968 + 1) := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparison_l3343_334398


namespace NUMINAMATH_CALUDE_parallel_vectors_l3343_334322

def a (m : ℝ) : Fin 2 → ℝ := ![2, m]
def b : Fin 2 → ℝ := ![1, -2]

def parallel (u v : Fin 2 → ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ (∀ i, u i = k * v i)

theorem parallel_vectors (m : ℝ) :
  parallel (a m) (λ i => a m i + 2 * b i) → m = -4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_l3343_334322


namespace NUMINAMATH_CALUDE_triangle_problem_l3343_334339

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_problem (t : Triangle) 
  (h1 : t.a > t.c)
  (h2 : t.b * t.a * Real.cos t.B = 2)
  (h3 : Real.cos t.B = 1/3)
  (h4 : t.b = 3) :
  (t.a = 3 ∧ t.c = 2) ∧ 
  Real.cos (t.B - t.C) = 23/27 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l3343_334339


namespace NUMINAMATH_CALUDE_f_expression_l3343_334368

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem f_expression : 
  (∀ x : ℝ, f (x + 1) = 3 * x + 2) → 
  (∀ x : ℝ, f x = 3 * x - 1) :=
by
  sorry

end NUMINAMATH_CALUDE_f_expression_l3343_334368


namespace NUMINAMATH_CALUDE_elderly_arrangement_count_l3343_334390

/-- The number of ways to arrange n distinct objects. -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to choose k objects from n distinct objects, where order matters. -/
def arrangements (n k : ℕ) : ℕ := 
  if k ≤ n then
    Nat.factorial n / Nat.factorial (n - k)
  else
    0

/-- The number of ways to arrange 5 volunteers and 2 elderly people in a line,
    where the elderly people must be adjacent and not at the ends. -/
def elderly_arrangement : ℕ := 
  arrangements 5 2 * permutations 4 * permutations 2

theorem elderly_arrangement_count : elderly_arrangement = 960 := by
  sorry

end NUMINAMATH_CALUDE_elderly_arrangement_count_l3343_334390


namespace NUMINAMATH_CALUDE_cab_driver_first_day_income_l3343_334303

def cab_driver_income (day2 day3 day4 day5 : ℕ) (average : ℕ) : Prop :=
  ∃ day1 : ℕ,
    day2 = 250 ∧
    day3 = 450 ∧
    day4 = 400 ∧
    day5 = 800 ∧
    average = 500 ∧
    (day1 + day2 + day3 + day4 + day5) / 5 = average ∧
    day1 = 600

theorem cab_driver_first_day_income :
  ∀ day2 day3 day4 day5 average : ℕ,
    cab_driver_income day2 day3 day4 day5 average →
    ∃ day1 : ℕ, day1 = 600 :=
by
  sorry

end NUMINAMATH_CALUDE_cab_driver_first_day_income_l3343_334303


namespace NUMINAMATH_CALUDE_total_fish_is_996_l3343_334393

/-- The number of fish each friend has -/
structure FishCount where
  max : ℕ
  sam : ℕ
  joe : ℕ
  harry : ℕ

/-- The conditions of the fish distribution among friends -/
def fish_distribution (fc : FishCount) : Prop :=
  fc.max = 6 ∧
  fc.sam = 3 * fc.max ∧
  fc.joe = 9 * fc.sam ∧
  fc.harry = 5 * fc.joe

/-- The total number of fish for all friends -/
def total_fish (fc : FishCount) : ℕ :=
  fc.max + fc.sam + fc.joe + fc.harry

/-- Theorem stating that the total number of fish is 996 -/
theorem total_fish_is_996 (fc : FishCount) (h : fish_distribution fc) : total_fish fc = 996 := by
  sorry

end NUMINAMATH_CALUDE_total_fish_is_996_l3343_334393


namespace NUMINAMATH_CALUDE_algebraic_expression_simplification_l3343_334396

theorem algebraic_expression_simplification (x : ℝ) (h : x = -4) :
  (x^2 - 2*x) / (x - 3) / ((1 / (x + 3) + 1 / (x - 3))) = 3 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_simplification_l3343_334396


namespace NUMINAMATH_CALUDE_regular_polygon_162_degrees_l3343_334372

/-- A regular polygon with interior angles measuring 162 degrees has 20 sides -/
theorem regular_polygon_162_degrees : ∀ n : ℕ, 
  n > 2 → 
  (180 * (n - 2) : ℝ) / n = 162 → 
  n = 20 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_162_degrees_l3343_334372


namespace NUMINAMATH_CALUDE_rectangle_area_formula_l3343_334308

/-- Represents a rectangle with a specific ratio of length to width and diagonal length. -/
structure Rectangle where
  ratio_length : ℝ
  ratio_width : ℝ
  diagonal : ℝ
  ratio_condition : ratio_length / ratio_width = 5 / 2

/-- The theorem stating that the area of the rectangle can be expressed as (10/29)d^2 -/
theorem rectangle_area_formula (rect : Rectangle) : 
  ∃ (length width : ℝ), 
    length / width = rect.ratio_length / rect.ratio_width ∧
    length * width = (10/29) * rect.diagonal^2 := by
  sorry


end NUMINAMATH_CALUDE_rectangle_area_formula_l3343_334308


namespace NUMINAMATH_CALUDE_minimum_choir_size_l3343_334346

def is_valid_choir_size (n : ℕ) : Prop :=
  n % 8 = 0 ∧ n % 9 = 0 ∧ n % 10 = 0

theorem minimum_choir_size :
  ∃ (n : ℕ), is_valid_choir_size n ∧ ∀ (m : ℕ), m < n → ¬ is_valid_choir_size m :=
by
  use 360
  sorry

end NUMINAMATH_CALUDE_minimum_choir_size_l3343_334346


namespace NUMINAMATH_CALUDE_unique_solution_l3343_334367

theorem unique_solution : ∃! x : ℕ, 
  (∃ k : ℕ, x = 7 * k) ∧ 
  x^3 < 8000 ∧ 
  10 < x ∧ x < 30 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l3343_334367


namespace NUMINAMATH_CALUDE_max_m_value_inequality_proof_l3343_334387

-- Define the function f
def f (x : ℝ) : ℝ := |x| + |x - 1|

-- Theorem for part I
theorem max_m_value (m : ℝ) : 
  (∀ x, f x ≥ |m - 1|) → m ≤ 2 :=
sorry

-- Theorem for part II
theorem inequality_proof (a b : ℝ) :
  a > 0 → b > 0 → a^2 + b^2 = 2 → a + b ≥ 2 * a * b :=
sorry

end NUMINAMATH_CALUDE_max_m_value_inequality_proof_l3343_334387


namespace NUMINAMATH_CALUDE_sum_of_coefficients_expansion_l3343_334341

theorem sum_of_coefficients_expansion (x y : ℝ) : 
  (fun x y => (x + 2*y - 1)^6) 1 1 = 64 := by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_expansion_l3343_334341


namespace NUMINAMATH_CALUDE_election_vote_count_l3343_334363

theorem election_vote_count : ∃ (total_votes : ℕ), 
  (total_votes > 0) ∧ 
  (∃ (candidate_votes rival_votes : ℕ),
    (candidate_votes = (total_votes * 3) / 10) ∧
    (rival_votes = candidate_votes + 4000) ∧
    (candidate_votes + rival_votes = total_votes) ∧
    (total_votes = 10000)) := by
  sorry

end NUMINAMATH_CALUDE_election_vote_count_l3343_334363


namespace NUMINAMATH_CALUDE_power_division_equality_l3343_334379

theorem power_division_equality : (3 : ℕ)^12 / ((27 : ℕ)^2 * 3^3) = 27 := by
  sorry

end NUMINAMATH_CALUDE_power_division_equality_l3343_334379


namespace NUMINAMATH_CALUDE_second_divisor_problem_l3343_334355

theorem second_divisor_problem :
  ∃! D : ℤ, 19 < D ∧ D < 242 ∧
  (∃ N : ℤ, N % 242 = 100 ∧ N % D = 19) ∧
  D = 27 := by
sorry

end NUMINAMATH_CALUDE_second_divisor_problem_l3343_334355


namespace NUMINAMATH_CALUDE_jean_calories_eaten_l3343_334352

def pages_written : ℕ := 12
def pages_per_donut : ℕ := 2
def calories_per_donut : ℕ := 150

theorem jean_calories_eaten : 
  (pages_written / pages_per_donut) * calories_per_donut = 900 := by
  sorry

end NUMINAMATH_CALUDE_jean_calories_eaten_l3343_334352


namespace NUMINAMATH_CALUDE_function_property_k_value_l3343_334310

theorem function_property_k_value (f : ℝ → ℝ) (k : ℝ) 
  (h1 : f 1 = 4)
  (h2 : ∀ x y, f (x + y) = f x + f y + k * x * y + 4)
  (h3 : f 2 + f 5 = 125) :
  k = 7 := by sorry

end NUMINAMATH_CALUDE_function_property_k_value_l3343_334310


namespace NUMINAMATH_CALUDE_five_squared_sum_five_times_l3343_334329

theorem five_squared_sum_five_times : (5 * 5) + (5 * 5) + (5 * 5) + (5 * 5) + (5 * 5) = 125 := by
  sorry

end NUMINAMATH_CALUDE_five_squared_sum_five_times_l3343_334329


namespace NUMINAMATH_CALUDE_journey_problem_solution_exists_l3343_334321

/-- Proves the existence of a solution for the journey problem -/
theorem journey_problem_solution_exists :
  ∃ (x y T : ℝ),
    x > 0 ∧ y > 0 ∧ T > 0 ∧
    x < 150 ∧ y < x ∧
    (x / 30 + (150 - x) / 3 = T) ∧
    (x / 30 + y / 30 + (150 - (x - y)) / 30 = T) ∧
    ((x - y) / 10 + (150 - (x - y)) / 30 = T) :=
by sorry

#check journey_problem_solution_exists

end NUMINAMATH_CALUDE_journey_problem_solution_exists_l3343_334321


namespace NUMINAMATH_CALUDE_john_hourly_rate_is_20_l3343_334332

/-- Represents John's car repair scenario -/
structure CarRepairScenario where
  total_cars : ℕ
  standard_repair_time : ℕ  -- in minutes
  longer_repair_factor : ℚ  -- factor for longer repair time
  standard_repair_count : ℕ
  total_earnings : ℕ        -- in dollars

/-- Calculates John's hourly rate given the car repair scenario -/
def calculate_hourly_rate (scenario : CarRepairScenario) : ℚ :=
  -- Function body to be implemented
  sorry

/-- Theorem stating that John's hourly rate is $20 -/
theorem john_hourly_rate_is_20 (scenario : CarRepairScenario) 
  (h1 : scenario.total_cars = 5)
  (h2 : scenario.standard_repair_time = 40)
  (h3 : scenario.longer_repair_factor = 3/2)
  (h4 : scenario.standard_repair_count = 3)
  (h5 : scenario.total_earnings = 80) :
  calculate_hourly_rate scenario = 20 := by
  sorry

end NUMINAMATH_CALUDE_john_hourly_rate_is_20_l3343_334332


namespace NUMINAMATH_CALUDE_min_squares_partition_l3343_334380

/-- Represents a square with an integer side length -/
structure Square where
  side : ℕ

/-- Represents a partition of a square into smaller squares -/
structure Partition where
  squares : List Square

/-- Check if a partition is valid for an 11x11 square -/
def isValidPartition (p : Partition) : Prop :=
  (p.squares.map (λ s => s.side * s.side)).sum = 11 * 11 ∧
  p.squares.all (λ s => s.side > 0 ∧ s.side < 11)

/-- The theorem stating the minimum number of squares in a valid partition -/
theorem min_squares_partition :
  ∃ (p : Partition), isValidPartition p ∧ p.squares.length = 11 ∧
  ∀ (q : Partition), isValidPartition q → p.squares.length ≤ q.squares.length :=
sorry

end NUMINAMATH_CALUDE_min_squares_partition_l3343_334380


namespace NUMINAMATH_CALUDE_alex_class_size_l3343_334383

/-- Represents a student's ranking in a class -/
structure StudentRanking where
  best : Nat
  worst : Nat

/-- Calculates the total number of students in a class given a student's ranking -/
def totalStudents (ranking : StudentRanking) : Nat :=
  ranking.best + ranking.worst - 1

/-- Theorem: If a student is ranked 20th best and 20th worst, there are 39 students in the class -/
theorem alex_class_size (ranking : StudentRanking) 
  (h1 : ranking.best = 20) 
  (h2 : ranking.worst = 20) : 
  totalStudents ranking = 39 := by
  sorry

end NUMINAMATH_CALUDE_alex_class_size_l3343_334383


namespace NUMINAMATH_CALUDE_clerical_employee_fraction_l3343_334399

/-- Proves that the fraction of clerical employees is 4/15 given the conditions -/
theorem clerical_employee_fraction :
  let total_employees : ℕ := 3600
  let clerical_fraction : ℚ := 4/15
  let reduction_factor : ℚ := 3/4
  let remaining_fraction : ℚ := 1/5
  (clerical_fraction * total_employees : ℚ) * reduction_factor =
    remaining_fraction * total_employees :=
by sorry

end NUMINAMATH_CALUDE_clerical_employee_fraction_l3343_334399


namespace NUMINAMATH_CALUDE_xiao_ming_brother_age_l3343_334350

def has_no_repeated_digits (year : ℕ) : Prop := sorry

def is_multiple_of_19 (year : ℕ) : Prop := year % 19 = 0

theorem xiao_ming_brother_age (birth_year : ℕ) 
  (h1 : is_multiple_of_19 birth_year)
  (h2 : ∀ y : ℕ, birth_year ≤ y → y < 2013 → ¬(has_no_repeated_digits y))
  (h3 : has_no_repeated_digits 2013) :
  2013 - birth_year = 18 := by sorry

end NUMINAMATH_CALUDE_xiao_ming_brother_age_l3343_334350


namespace NUMINAMATH_CALUDE_floor_inequality_l3343_334307

theorem floor_inequality (α β : ℝ) : 
  ⌊2 * α⌋ + ⌊2 * β⌋ ≥ ⌊α⌋ + ⌊β⌋ + ⌊α + β⌋ := by
  sorry

end NUMINAMATH_CALUDE_floor_inequality_l3343_334307


namespace NUMINAMATH_CALUDE_smallest_non_prime_non_square_no_small_factors_l3343_334353

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

def has_no_prime_factor_less_than (n k : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → p < k → ¬(n % p = 0)

theorem smallest_non_prime_non_square_no_small_factors : 
  (∀ m : ℕ, m < 4091 → 
    is_prime m ∨ 
    is_perfect_square m ∨ 
    ¬(has_no_prime_factor_less_than m 60)) ∧ 
  ¬(is_prime 4091) ∧ 
  ¬(is_perfect_square 4091) ∧ 
  has_no_prime_factor_less_than 4091 60 :=
sorry

end NUMINAMATH_CALUDE_smallest_non_prime_non_square_no_small_factors_l3343_334353


namespace NUMINAMATH_CALUDE_f_of_7_eq_17_l3343_334300

/-- The polynomial function f(x) = 2x^4 - 17x^3 + 26x^2 - 24x - 60 -/
def f (x : ℝ) : ℝ := 2*x^4 - 17*x^3 + 26*x^2 - 24*x - 60

/-- Theorem: The value of f(7) is 17 -/
theorem f_of_7_eq_17 : f 7 = 17 := by
  sorry

end NUMINAMATH_CALUDE_f_of_7_eq_17_l3343_334300


namespace NUMINAMATH_CALUDE_total_pencils_l3343_334316

/-- Given that each child has 2 pencils and there are 15 children, 
    prove that the total number of pencils is 30. -/
theorem total_pencils (pencils_per_child : ℕ) (num_children : ℕ) 
  (h1 : pencils_per_child = 2) (h2 : num_children = 15) : 
  pencils_per_child * num_children = 30 := by
  sorry

end NUMINAMATH_CALUDE_total_pencils_l3343_334316


namespace NUMINAMATH_CALUDE_taxi_ride_distance_is_8_miles_l3343_334336

/-- Calculates the distance of a taxi ride given the fare structure and total charge -/
def taxi_ride_distance (initial_charge : ℚ) (additional_charge : ℚ) (total_charge : ℚ) : ℚ :=
  let remaining_charge := total_charge - initial_charge
  let additional_increments := remaining_charge / additional_charge
  (additional_increments + 1) * (1 / 5)

/-- Proves that the taxi ride distance is 8 miles given the specified fare structure and total charge -/
theorem taxi_ride_distance_is_8_miles :
  let initial_charge : ℚ := 21/10  -- $2.10
  let additional_charge : ℚ := 4/10  -- $0.40
  let total_charge : ℚ := 177/10  -- $17.70
  taxi_ride_distance initial_charge additional_charge total_charge = 8 := by
  sorry

#eval taxi_ride_distance (21/10) (4/10) (177/10)

end NUMINAMATH_CALUDE_taxi_ride_distance_is_8_miles_l3343_334336


namespace NUMINAMATH_CALUDE_present_age_of_b_l3343_334377

theorem present_age_of_b (a b : ℕ) 
  (h1 : a + 10 = 2 * (b - 10)) 
  (h2 : a = b + 9) : 
  b = 39 := by
  sorry

end NUMINAMATH_CALUDE_present_age_of_b_l3343_334377


namespace NUMINAMATH_CALUDE_car_distribution_l3343_334349

theorem car_distribution (total_cars : ℕ) (first_supplier : ℕ) : 
  total_cars = 5650000 →
  first_supplier = 1000000 →
  let second_supplier := first_supplier + 500000
  let third_supplier := first_supplier + second_supplier
  let remaining_cars := total_cars - (first_supplier + second_supplier + third_supplier)
  remaining_cars / 2 = 325000 :=
by sorry

end NUMINAMATH_CALUDE_car_distribution_l3343_334349


namespace NUMINAMATH_CALUDE_tile_arrangements_l3343_334375

def brown_tiles : ℕ := 1
def purple_tiles : ℕ := 2
def green_tiles : ℕ := 3
def yellow_tiles : ℕ := 2

def total_tiles : ℕ := brown_tiles + purple_tiles + green_tiles + yellow_tiles

theorem tile_arrangements :
  (Nat.factorial total_tiles) / 
  (Nat.factorial brown_tiles * Nat.factorial purple_tiles * 
   Nat.factorial green_tiles * Nat.factorial yellow_tiles) = 1680 := by
  sorry

end NUMINAMATH_CALUDE_tile_arrangements_l3343_334375
