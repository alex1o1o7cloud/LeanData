import Mathlib

namespace NUMINAMATH_CALUDE_center_value_of_arithmetic_array_l3497_349727

/-- Represents a 4x4 array where each row and column is an arithmetic sequence -/
def ArithmeticArray := Fin 4 → Fin 4 → ℚ

/-- The common difference of an arithmetic sequence given its first and last terms -/
def commonDifference (a₁ a₄ : ℚ) : ℚ := (a₄ - a₁) / 3

/-- Checks if a sequence is arithmetic -/
def isArithmeticSequence (seq : Fin 4 → ℚ) : Prop :=
  ∀ i j : Fin 4, i.val < j.val → seq j - seq i = commonDifference (seq 0) (seq 3) * (j - i)

/-- Properties of our specific arithmetic array -/
def isValidArray (arr : ArithmeticArray) : Prop :=
  (∀ i : Fin 4, isArithmeticSequence (λ j => arr i j)) ∧  -- Each row is arithmetic
  (∀ j : Fin 4, isArithmeticSequence (λ i => arr i j)) ∧  -- Each column is arithmetic
  arr 0 0 = 3 ∧ arr 0 3 = 21 ∧                            -- First row conditions
  arr 3 0 = 15 ∧ arr 3 3 = 45                             -- Fourth row conditions

theorem center_value_of_arithmetic_array (arr : ArithmeticArray) 
  (h : isValidArray arr) : arr 1 1 = 14 + 1/3 := by
  sorry

end NUMINAMATH_CALUDE_center_value_of_arithmetic_array_l3497_349727


namespace NUMINAMATH_CALUDE_sarah_picked_45_apples_l3497_349706

/-- The number of apples Sarah's brother picked -/
def brother_apples : ℕ := 9

/-- The factor by which Sarah picked more apples than her brother -/
def sarah_factor : ℕ := 5

/-- The number of apples Sarah picked -/
def sarah_apples : ℕ := sarah_factor * brother_apples

theorem sarah_picked_45_apples : sarah_apples = 45 := by
  sorry

end NUMINAMATH_CALUDE_sarah_picked_45_apples_l3497_349706


namespace NUMINAMATH_CALUDE_max_value_fraction_max_value_achievable_l3497_349731

theorem max_value_fraction (x y : ℝ) : 
  (2 * x + Real.sqrt 2 * y) / (2 * x^4 + 4 * y^4 + 9) ≤ 1/4 :=
by sorry

theorem max_value_achievable : 
  ∃ x y : ℝ, (2 * x + Real.sqrt 2 * y) / (2 * x^4 + 4 * y^4 + 9) = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_max_value_fraction_max_value_achievable_l3497_349731


namespace NUMINAMATH_CALUDE_james_distance_l3497_349783

/-- Calculates the distance traveled given speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: James rode 80 miles -/
theorem james_distance : distance 16 5 = 80 := by
  sorry

end NUMINAMATH_CALUDE_james_distance_l3497_349783


namespace NUMINAMATH_CALUDE_range_of_a_l3497_349709

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x > 0 → (x - a + Real.log (x / a)) * (-2 * x^2 + a * x + 10) ≤ 0) → 
  a = Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l3497_349709


namespace NUMINAMATH_CALUDE_equation_has_real_root_when_K_zero_l3497_349765

/-- The equation x = K³(x³ - 3x² + 2x + 1) has at least one real root when K = 0 -/
theorem equation_has_real_root_when_K_zero :
  ∃ x : ℝ, x = 0^3 * (x^3 - 3*x^2 + 2*x + 1) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_has_real_root_when_K_zero_l3497_349765


namespace NUMINAMATH_CALUDE_tv_watching_weeks_l3497_349712

/-- Represents Flynn's TV watching habits and total time --/
structure TVWatching where
  weekdayMinutes : ℕ  -- Minutes watched per weekday night
  weekendHours : ℕ    -- Additional hours watched on weekends
  totalHours : ℕ      -- Total hours watched

/-- Calculates the number of weeks based on TV watching habits --/
def calculateWeeks (tw : TVWatching) : ℚ :=
  let weekdayHours : ℚ := (tw.weekdayMinutes * 5 : ℚ) / 60
  let totalWeeklyHours : ℚ := weekdayHours + tw.weekendHours
  tw.totalHours / totalWeeklyHours

/-- Theorem stating that 234 hours of TV watching corresponds to 52 weeks --/
theorem tv_watching_weeks (tw : TVWatching) 
  (h1 : tw.weekdayMinutes = 30)
  (h2 : tw.weekendHours = 2)
  (h3 : tw.totalHours = 234) :
  calculateWeeks tw = 52 := by
  sorry

end NUMINAMATH_CALUDE_tv_watching_weeks_l3497_349712


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l3497_349725

/-- A rhombus with given diagonal lengths has the specified perimeter -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) :
  let side_length := Real.sqrt ((d1/2)^2 + (d2/2)^2)
  4 * side_length = 52 := by sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l3497_349725


namespace NUMINAMATH_CALUDE_calculation_proof_l3497_349793

theorem calculation_proof :
  ((-1/4 + 5/6 - 2/9) * (-36) = -13) ∧
  (-1^4 - 1/6 - (3 + (-3)^2) / (-1 - 1/2) = 6 + 5/6) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3497_349793


namespace NUMINAMATH_CALUDE_sqrt_2a_plus_b_is_6_l3497_349744

/-- Given that the square root of (a + 9) is -5 and the cube root of (2b - a) is -2,
    prove that the arithmetic square root of (2a + b) is 6 -/
theorem sqrt_2a_plus_b_is_6 (a b : ℝ) 
  (h1 : Real.sqrt (a + 9) = -5)
  (h2 : (2 * b - a) ^ (1/3 : ℝ) = -2) :
  Real.sqrt (2 * a + b) = 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_2a_plus_b_is_6_l3497_349744


namespace NUMINAMATH_CALUDE_alice_shopping_cost_l3497_349790

/-- Represents the shopping list and discounts --/
structure ShoppingTrip where
  apple_price : ℕ
  apple_quantity : ℕ
  bread_price : ℕ
  bread_quantity : ℕ
  cereal_price : ℕ
  cereal_quantity : ℕ
  cake_price : ℕ
  cheese_price : ℕ
  cereal_discount : ℕ
  bread_discount : Bool
  coupon_threshold : ℕ
  coupon_value : ℕ

/-- Calculates the total cost of the shopping trip --/
def calculate_total (trip : ShoppingTrip) : ℕ :=
  let apple_cost := trip.apple_price * trip.apple_quantity
  let bread_cost := if trip.bread_discount then trip.bread_price else trip.bread_price * trip.bread_quantity
  let cereal_cost := (trip.cereal_price - trip.cereal_discount) * trip.cereal_quantity
  let total := apple_cost + bread_cost + cereal_cost + trip.cake_price + trip.cheese_price
  if total ≥ trip.coupon_threshold then total - trip.coupon_value else total

/-- Theorem stating that Alice's shopping trip costs $38 --/
theorem alice_shopping_cost : 
  let trip : ShoppingTrip := {
    apple_price := 2,
    apple_quantity := 4,
    bread_price := 4,
    bread_quantity := 2,
    cereal_price := 5,
    cereal_quantity := 3,
    cake_price := 8,
    cheese_price := 6,
    cereal_discount := 1,
    bread_discount := true,
    coupon_threshold := 40,
    coupon_value := 10
  }
  calculate_total trip = 38 := by
sorry

end NUMINAMATH_CALUDE_alice_shopping_cost_l3497_349790


namespace NUMINAMATH_CALUDE_tom_lasagna_noodles_l3497_349739

/-- The number of packages of noodles Tom needs to buy for his lasagna -/
def noodle_packages_needed (beef_amount : ℕ) (noodle_ratio : ℕ) (existing_noodles : ℕ) (package_size : ℕ) : ℕ :=
  let total_noodles_needed := beef_amount * noodle_ratio
  let additional_noodles_needed := total_noodles_needed - existing_noodles
  (additional_noodles_needed + package_size - 1) / package_size

theorem tom_lasagna_noodles : noodle_packages_needed 10 2 4 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_tom_lasagna_noodles_l3497_349739


namespace NUMINAMATH_CALUDE_least_with_eight_factors_l3497_349781

/-- A function that returns the number of distinct positive factors of a natural number. -/
def num_factors (n : ℕ) : ℕ := sorry

/-- A predicate that checks if a natural number has exactly eight distinct positive factors. -/
def has_eight_factors (n : ℕ) : Prop := num_factors n = 8

/-- The theorem stating that 54 is the least positive integer with exactly eight distinct positive factors. -/
theorem least_with_eight_factors : 
  has_eight_factors 54 ∧ ∀ m : ℕ, m < 54 → ¬(has_eight_factors m) := by sorry

end NUMINAMATH_CALUDE_least_with_eight_factors_l3497_349781


namespace NUMINAMATH_CALUDE_oil_purchase_increase_l3497_349788

/-- Calculates the additional amount of oil that can be purchased after a price reduction -/
def additional_oil_purchase (price_reduction : ℚ) (budget : ℚ) (reduced_price : ℚ) : ℚ :=
  let original_price := reduced_price / (1 - price_reduction)
  let original_amount := budget / original_price
  let new_amount := budget / reduced_price
  new_amount - original_amount

/-- Proves that given a 30% price reduction, a budget of 700, and a reduced price of 70,
    the additional amount of oil that can be purchased is 3 -/
theorem oil_purchase_increase :
  additional_oil_purchase (30 / 100) 700 70 = 3 := by
  sorry

end NUMINAMATH_CALUDE_oil_purchase_increase_l3497_349788


namespace NUMINAMATH_CALUDE_molecular_weight_5_moles_AlBr3_l3497_349717

/-- The atomic weight of Aluminum in g/mol -/
def atomic_weight_Al : ℝ := 26.98

/-- The atomic weight of Bromine in g/mol -/
def atomic_weight_Br : ℝ := 79.90

/-- The number of Aluminum atoms in AlBr3 -/
def num_Al : ℕ := 1

/-- The number of Bromine atoms in AlBr3 -/
def num_Br : ℕ := 3

/-- The number of moles of AlBr3 -/
def num_moles : ℝ := 5

/-- The molecular weight of AlBr3 in g/mol -/
def molecular_weight_AlBr3 : ℝ :=
  num_Al * atomic_weight_Al + num_Br * atomic_weight_Br

/-- Theorem stating that the molecular weight of 5 moles of AlBr3 is 1333.40 grams -/
theorem molecular_weight_5_moles_AlBr3 :
  num_moles * molecular_weight_AlBr3 = 1333.40 := by
  sorry

end NUMINAMATH_CALUDE_molecular_weight_5_moles_AlBr3_l3497_349717


namespace NUMINAMATH_CALUDE_sequence_a_formula_l3497_349711

def sequence_a : ℕ → ℤ
  | 0 => 1
  | 1 => 5
  | (n + 2) => (2 * (sequence_a (n + 1))^2 - 3 * sequence_a (n + 1) - 9) / (2 * sequence_a n)

theorem sequence_a_formula : ∀ n : ℕ, sequence_a n = 2^(n + 2) - 3 := by
  sorry

end NUMINAMATH_CALUDE_sequence_a_formula_l3497_349711


namespace NUMINAMATH_CALUDE_new_car_distance_l3497_349719

theorem new_car_distance (old_car_speed : ℝ) (old_car_distance : ℝ) (new_car_speed : ℝ) : 
  old_car_distance = 150 →
  new_car_speed = old_car_speed * 1.3 →
  new_car_speed * (old_car_distance / old_car_speed) = 195 := by
sorry

end NUMINAMATH_CALUDE_new_car_distance_l3497_349719


namespace NUMINAMATH_CALUDE_talia_father_current_age_talia_future_age_talia_mom_current_age_talia_father_future_age_l3497_349757

-- Define the current year as a reference point
def current_year : ℕ := 0

-- Define Talia's age
def talia_age : ℕ → ℕ
  | year => 13 + year

-- Define Talia's mom's age
def talia_mom_age : ℕ → ℕ
  | year => 3 * talia_age current_year + year

-- Define Talia's father's age
def talia_father_age : ℕ → ℕ
  | year => talia_mom_age current_year + (year - 3)

-- State the theorem
theorem talia_father_current_age :
  talia_father_age current_year = 36 :=
by
  sorry

-- Conditions as separate theorems
theorem talia_future_age :
  talia_age 7 = 20 :=
by
  sorry

theorem talia_mom_current_age :
  talia_mom_age current_year = 3 * talia_age current_year :=
by
  sorry

theorem talia_father_future_age :
  talia_father_age 3 = talia_mom_age current_year :=
by
  sorry

end NUMINAMATH_CALUDE_talia_father_current_age_talia_future_age_talia_mom_current_age_talia_father_future_age_l3497_349757


namespace NUMINAMATH_CALUDE_conversion_factor_feet_to_miles_l3497_349741

/-- Conversion factor from feet to miles -/
def feet_per_mile : ℝ := 5280

/-- Speed of the object in miles per hour -/
def speed_mph : ℝ := 68.18181818181819

/-- Distance traveled by the object in feet -/
def distance_feet : ℝ := 400

/-- Time taken by the object in seconds -/
def time_seconds : ℝ := 4

/-- Theorem stating that the conversion factor from feet to miles is 5280 -/
theorem conversion_factor_feet_to_miles :
  feet_per_mile = (distance_feet / time_seconds) / (speed_mph / 3600) := by
  sorry

#check conversion_factor_feet_to_miles

end NUMINAMATH_CALUDE_conversion_factor_feet_to_miles_l3497_349741


namespace NUMINAMATH_CALUDE_exists_integer_term_l3497_349756

def sequence_rule (x : ℚ) : ℚ := x + 1 / (Int.floor x)

def is_valid_sequence (x : ℕ → ℚ) : Prop :=
  x 1 > 1 ∧ ∀ n : ℕ, x (n + 1) = sequence_rule (x n)

theorem exists_integer_term (x : ℕ → ℚ) (h : is_valid_sequence x) :
  ∃ k : ℕ, ∃ m : ℤ, x k = m :=
sorry

end NUMINAMATH_CALUDE_exists_integer_term_l3497_349756


namespace NUMINAMATH_CALUDE_function_property_l3497_349775

/-- Given a function f and a real number a, if f(a) + f(1) = 0, then a = -3 -/
theorem function_property (f : ℝ → ℝ) (a : ℝ) (h : f a + f 1 = 0) : a = -3 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l3497_349775


namespace NUMINAMATH_CALUDE_average_visitors_is_290_l3497_349730

/-- Calculates the average number of visitors per day in a 30-day month starting on a Sunday -/
def average_visitors_per_day (sunday_visitors : ℕ) (other_day_visitors : ℕ) : ℚ :=
  let total_sundays := 5
  let total_other_days := 25
  let total_visitors := sunday_visitors * total_sundays + other_day_visitors * total_other_days
  total_visitors / 30

/-- Theorem stating that the average number of visitors per day is 290 -/
theorem average_visitors_is_290 :
  average_visitors_per_day 540 240 = 290 := by
  sorry

end NUMINAMATH_CALUDE_average_visitors_is_290_l3497_349730


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l3497_349738

theorem polynomial_division_theorem (x : ℝ) :
  (x - 3) * (x^3 - 19*x^2 - 45*x - 148) + (-435) = x^4 - 22*x^3 + 12*x^2 - 13*x + 9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l3497_349738


namespace NUMINAMATH_CALUDE_planes_parallel_to_same_plane_are_parallel_planes_perpendicular_to_same_line_are_parallel_l3497_349745

-- Define the concept of a plane
variable (Plane : Type)

-- Define the concept of a line
variable (Line : Type)

-- Define the parallel relation between planes
variable (parallel : Plane → Plane → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the property of being non-coincident
variable (non_coincident : Plane → Plane → Prop)

-- Theorem 1: Two non-coincident planes parallel to the same plane are parallel
theorem planes_parallel_to_same_plane_are_parallel 
  (α β γ : Plane) 
  (h1 : parallel α γ) 
  (h2 : parallel β γ) 
  (h3 : non_coincident α β) : 
  parallel α β :=
sorry

-- Theorem 2: Two non-coincident planes perpendicular to the same line are parallel
theorem planes_perpendicular_to_same_line_are_parallel 
  (α β : Plane) 
  (a : Line) 
  (h1 : perpendicular a α) 
  (h2 : perpendicular a β) 
  (h3 : non_coincident α β) : 
  parallel α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_to_same_plane_are_parallel_planes_perpendicular_to_same_line_are_parallel_l3497_349745


namespace NUMINAMATH_CALUDE_sqrt_450_equals_15_l3497_349721

theorem sqrt_450_equals_15 : Real.sqrt 450 = 15 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_450_equals_15_l3497_349721


namespace NUMINAMATH_CALUDE_probability_no_shaded_square_l3497_349789

/-- Represents a rectangular grid with shaded squares -/
structure ShadedGrid :=
  (rows : Nat)
  (cols : Nat)
  (shaded_cols : Finset Nat)

/-- Calculates the total number of rectangles in the grid -/
def total_rectangles (grid : ShadedGrid) : Nat :=
  (grid.rows * Nat.choose grid.cols 2)

/-- Calculates the number of rectangles containing a shaded square -/
def shaded_rectangles (grid : ShadedGrid) : Nat :=
  grid.rows * (grid.shaded_cols.card * (grid.cols - grid.shaded_cols.card))

/-- Theorem stating the probability of selecting a rectangle without a shaded square -/
theorem probability_no_shaded_square (grid : ShadedGrid) 
  (h1 : grid.rows = 2)
  (h2 : grid.cols = 2005)
  (h3 : grid.shaded_cols = {1003}) :
  (total_rectangles grid - shaded_rectangles grid) / total_rectangles grid = 1002 / 2005 := by
  sorry

end NUMINAMATH_CALUDE_probability_no_shaded_square_l3497_349789


namespace NUMINAMATH_CALUDE_min_upper_bound_fraction_l3497_349732

theorem min_upper_bound_fraction (a₁ a₂ a₃ : ℝ) (h : a₁ ≠ 0 ∨ a₂ ≠ 0 ∨ a₃ ≠ 0) :
  ∃ M : ℝ, M = Real.sqrt 2 / 2 ∧
  (∀ x y : ℝ, x > 0 → y > 0 → x^2 + y^2 = 2 →
    (x * a₁ * a₂ + y * a₂ * a₃) / (a₁^2 + a₂^2 + a₃^2) ≤ M) ∧
  ∀ ε > 0, ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x^2 + y^2 = 2 ∧
    (x * a₁ * a₂ + y * a₂ * a₃) / (a₁^2 + a₂^2 + a₃^2) > M - ε :=
by sorry

end NUMINAMATH_CALUDE_min_upper_bound_fraction_l3497_349732


namespace NUMINAMATH_CALUDE_base_seven_65432_equals_16340_l3497_349771

def base_seven_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ (digits.length - 1 - i))) 0

theorem base_seven_65432_equals_16340 :
  base_seven_to_decimal [6, 5, 4, 3, 2] = 16340 := by
  sorry

end NUMINAMATH_CALUDE_base_seven_65432_equals_16340_l3497_349771


namespace NUMINAMATH_CALUDE_gcd_of_large_powers_l3497_349724

theorem gcd_of_large_powers (n m : ℕ) : 
  Nat.gcd (2^1050 - 1) (2^1062 - 1) = 2^12 - 1 :=
by sorry

end NUMINAMATH_CALUDE_gcd_of_large_powers_l3497_349724


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3497_349703

/-- Given a real number m, proves that if the solution to the system of linear inequalities
    (2x - 1 > 3(x - 2) and x < m) is x < 5, then m ≥ 5. -/
theorem inequality_system_solution (m : ℝ) : 
  (∀ x : ℝ, (2*x - 1 > 3*(x - 2) ∧ x < m) ↔ x < 5) → m ≥ 5 :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3497_349703


namespace NUMINAMATH_CALUDE_lcm_220_504_l3497_349760

theorem lcm_220_504 : Nat.lcm 220 504 = 27720 := by
  sorry

end NUMINAMATH_CALUDE_lcm_220_504_l3497_349760


namespace NUMINAMATH_CALUDE_shared_foci_implies_a_equals_one_l3497_349755

-- Define the ellipse equation
def ellipse (x y a : ℝ) : Prop := x^2 / 4 + y^2 / a^2 = 1

-- Define the hyperbola equation
def hyperbola (x y a : ℝ) : Prop := x^2 / a - y^2 / 2 = 1

-- Theorem statement
theorem shared_foci_implies_a_equals_one :
  ∀ a : ℝ, a > 0 →
  (∀ x y : ℝ, ellipse x y a ↔ hyperbola x y a) →
  a = 1 :=
by sorry

end NUMINAMATH_CALUDE_shared_foci_implies_a_equals_one_l3497_349755


namespace NUMINAMATH_CALUDE_least_N_for_probability_l3497_349720

def P (N : ℕ) : ℚ := 2 * (N / 3 + 1) / (N + 2)

def is_multiple_of_seven (N : ℕ) : Prop := ∃ k, N = 7 * k

theorem least_N_for_probability (N : ℕ) :
  is_multiple_of_seven N →
  (∀ M, is_multiple_of_seven M → M < N → P M ≥ 7/10) →
  P N < 7/10 →
  N = 700 :=
sorry

end NUMINAMATH_CALUDE_least_N_for_probability_l3497_349720


namespace NUMINAMATH_CALUDE_jake_final_balance_l3497_349715

/-- Represents Jake's bitcoin transactions and calculates his final balance --/
def jake_bitcoin_balance (initial_fortune : ℚ) (investment : ℚ) (first_donation : ℚ) 
  (brother_return : ℚ) (second_donation : ℚ) : ℚ :=
  let after_investment := initial_fortune - investment
  let after_first_donation := after_investment - (first_donation / 2)
  let after_giving_to_brother := after_first_donation / 2
  let after_brother_return := after_giving_to_brother + brother_return
  let after_quadrupling := after_brother_return * 4
  after_quadrupling - (second_donation * 4)

/-- Theorem stating that Jake ends up with 95 bitcoins --/
theorem jake_final_balance : 
  jake_bitcoin_balance 120 40 25 5 15 = 95 := by
  sorry

end NUMINAMATH_CALUDE_jake_final_balance_l3497_349715


namespace NUMINAMATH_CALUDE_jason_bookcase_weight_difference_l3497_349722

/-- Represents the bookcase and Jason's collection of items -/
structure Bookcase :=
  (shelves : Nat)
  (shelf_weight_limit : Nat)
  (hardcover_books : Nat)
  (textbooks : Nat)
  (knick_knacks : Nat)
  (max_hardcover_weight : Real)
  (max_textbook_weight : Real)
  (max_knick_knack_weight : Real)

/-- Calculates the maximum weight of the collection minus the bookcase's weight limit -/
def weight_difference (b : Bookcase) : Real :=
  b.hardcover_books * b.max_hardcover_weight +
  b.textbooks * b.max_textbook_weight +
  b.knick_knacks * b.max_knick_knack_weight -
  b.shelves * b.shelf_weight_limit

/-- Theorem stating that the weight difference for Jason's collection is 195 pounds -/
theorem jason_bookcase_weight_difference :
  ∃ (b : Bookcase),
    b.shelves = 4 ∧
    b.shelf_weight_limit = 20 ∧
    b.hardcover_books = 70 ∧
    b.textbooks = 30 ∧
    b.knick_knacks = 10 ∧
    b.max_hardcover_weight = 1.5 ∧
    b.max_textbook_weight = 3 ∧
    b.max_knick_knack_weight = 8 ∧
    weight_difference b = 195 :=
  sorry

end NUMINAMATH_CALUDE_jason_bookcase_weight_difference_l3497_349722


namespace NUMINAMATH_CALUDE_sqrt_simplification_l3497_349723

theorem sqrt_simplification (a : ℝ) (ha : a ≥ 0) :
  Real.sqrt (a^(1/2) * Real.sqrt (a^(1/2) * Real.sqrt a)) = a^(1/2) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_simplification_l3497_349723


namespace NUMINAMATH_CALUDE_remainder_2519_div_6_l3497_349749

theorem remainder_2519_div_6 : 2519 % 6 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_2519_div_6_l3497_349749


namespace NUMINAMATH_CALUDE_apple_banana_ratio_l3497_349713

/-- Proves that the ratio of apples to bananas is 2:1 given the total number of fruits,
    number of bananas, and number of oranges in a bowl of fruit. -/
theorem apple_banana_ratio (total : ℕ) (bananas : ℕ) (oranges : ℕ)
    (h_total : total = 12)
    (h_bananas : bananas = 2)
    (h_oranges : oranges = 6) :
    (total - bananas - oranges) / bananas = 2 := by
  sorry

end NUMINAMATH_CALUDE_apple_banana_ratio_l3497_349713


namespace NUMINAMATH_CALUDE_octal_number_check_l3497_349787

def is_octal_digit (d : Nat) : Prop := d < 8

def is_octal_number (n : Nat) : Prop :=
  ∀ d, d ∈ n.digits 8 → is_octal_digit d

theorem octal_number_check :
  ¬ is_octal_number 8102 ∧
  ¬ is_octal_number 793 ∧
  is_octal_number 214 ∧
  ¬ is_octal_number 998 := by sorry

end NUMINAMATH_CALUDE_octal_number_check_l3497_349787


namespace NUMINAMATH_CALUDE_perfect_square_power_of_two_plus_256_l3497_349750

theorem perfect_square_power_of_two_plus_256 (n : ℕ) :
  (∃ k : ℕ+, 2^n + 256 = k^2) → n = 11 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_power_of_two_plus_256_l3497_349750


namespace NUMINAMATH_CALUDE_quadruple_sequence_no_repetition_l3497_349746

/-- Transformation function for quadruples -/
def transform (q : ℝ × ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ × ℝ :=
  let (a, b, c, d) := q
  (a * b, b * c, c * d, d * a)

/-- Generates the sequence of quadruples starting from an initial quadruple -/
def quadruple_sequence (initial : ℝ × ℝ × ℝ × ℝ) : ℕ → ℝ × ℝ × ℝ × ℝ
  | 0 => initial
  | n + 1 => transform (quadruple_sequence initial n)

theorem quadruple_sequence_no_repetition (a₀ b₀ c₀ d₀ : ℝ) :
  (a₀, b₀, c₀, d₀) ≠ (1, 1, 1, 1) →
  ∀ i j : ℕ, i ≠ j →
    quadruple_sequence (a₀, b₀, c₀, d₀) i ≠ quadruple_sequence (a₀, b₀, c₀, d₀) j :=
by sorry

end NUMINAMATH_CALUDE_quadruple_sequence_no_repetition_l3497_349746


namespace NUMINAMATH_CALUDE_max_value_of_function_l3497_349758

theorem max_value_of_function (x : ℝ) (h : x < 5/4) :
  (∀ z : ℝ, z < 5/4 → 4*z - 2 + 1/(4*z - 5) ≤ 4*x - 2 + 1/(4*x - 5)) →
  4*x - 2 + 1/(4*x - 5) = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_function_l3497_349758


namespace NUMINAMATH_CALUDE_a_11_value_l3497_349714

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem a_11_value (a : ℕ → ℝ) 
    (h_arithmetic : arithmetic_sequence a)
    (h_a1 : a 1 = 1)
    (h_diff : ∀ n : ℕ, a (n + 2) - a n = 6) :
  a 11 = 31 := by
sorry

end NUMINAMATH_CALUDE_a_11_value_l3497_349714


namespace NUMINAMATH_CALUDE_f_inequality_solution_f_minimum_value_l3497_349701

def f (x : ℝ) : ℝ := |2*x + 1| - |x - 4|

theorem f_inequality_solution (x : ℝ) : 
  f x > 2 ↔ x < -7 ∨ (5/3 < x ∧ x < 4) ∨ x > 7 := by sorry

theorem f_minimum_value : 
  ∃ (x : ℝ), f x = -9/2 ∧ ∀ (y : ℝ), f y ≥ -9/2 := by sorry

end NUMINAMATH_CALUDE_f_inequality_solution_f_minimum_value_l3497_349701


namespace NUMINAMATH_CALUDE_min_value_fraction_l3497_349791

theorem min_value_fraction (x : ℝ) (h : x > 12) :
  x^2 / (x - 12) ≥ 48 ∧ (x^2 / (x - 12) = 48 ↔ x = 24) := by
  sorry

end NUMINAMATH_CALUDE_min_value_fraction_l3497_349791


namespace NUMINAMATH_CALUDE_equal_chords_length_squared_l3497_349763

/-- Two circles with radii 10 and 8, centers 15 units apart -/
structure CircleConfiguration where
  center_distance : ℝ
  radius1 : ℝ
  radius2 : ℝ
  center_distance_eq : center_distance = 15
  radius1_eq : radius1 = 10
  radius2_eq : radius2 = 8

/-- Point of intersection of the two circles -/
def IntersectionPoint (config : CircleConfiguration) : Type :=
  { p : ℝ × ℝ // 
    (p.1 - 0)^2 + p.2^2 = config.radius1^2 ∧ 
    (p.1 - config.center_distance)^2 + p.2^2 = config.radius2^2 }

/-- Line through intersection point creating equal chords -/
structure EqualChordsLine (config : CircleConfiguration) where
  p : IntersectionPoint config
  q : ℝ × ℝ
  r : ℝ × ℝ
  on_circle1 : (q.1 - 0)^2 + q.2^2 = config.radius1^2
  on_circle2 : (r.1 - config.center_distance)^2 + r.2^2 = config.radius2^2
  equal_chords : (q.1 - p.val.1)^2 + (q.2 - p.val.2)^2 = (r.1 - p.val.1)^2 + (r.2 - p.val.2)^2

/-- Theorem: The square of the length of QP is 164 -/
theorem equal_chords_length_squared 
  (config : CircleConfiguration) 
  (line : EqualChordsLine config) : 
  (line.q.1 - line.p.val.1)^2 + (line.q.2 - line.p.val.2)^2 = 164 := by
  sorry

end NUMINAMATH_CALUDE_equal_chords_length_squared_l3497_349763


namespace NUMINAMATH_CALUDE_perpendicular_vector_of_parallel_lines_l3497_349777

/-- Given two parallel lines l and m in 2D space, this theorem proves that
    the vector perpendicular to both lines, normalized such that its
    components sum to 7, is (2, 5). -/
theorem perpendicular_vector_of_parallel_lines :
  ∀ (l m : ℝ → ℝ × ℝ),
  (∃ (k : ℝ), k ≠ 0 ∧ (l 0).1 - (l 1).1 = k * ((m 0).1 - (m 1).1) ∧
                    (l 0).2 - (l 1).2 = k * ((m 0).2 - (m 1).2)) →
  ∃ (v : ℝ × ℝ),
    v.1 + v.2 = 7 ∧
    v.1 * ((l 0).1 - (l 1).1) + v.2 * ((l 0).2 - (l 1).2) = 0 ∧
    v = (2, 5) :=
by sorry


end NUMINAMATH_CALUDE_perpendicular_vector_of_parallel_lines_l3497_349777


namespace NUMINAMATH_CALUDE_no_universal_divisibility_l3497_349761

def concatenate_two_digits (a b : Nat) : Nat :=
  10 * a + b

def concatenate_three_digits (a n b : Nat) : Nat :=
  100 * a + 10 * n + b

theorem no_universal_divisibility :
  ∀ n : Nat, ∃ a b : Nat,
    1 ≤ a ∧ a ≤ 9 ∧
    1 ≤ b ∧ b ≤ 9 ∧
    ¬(concatenate_two_digits a b ∣ concatenate_three_digits a n b) := by
  sorry

end NUMINAMATH_CALUDE_no_universal_divisibility_l3497_349761


namespace NUMINAMATH_CALUDE_meaningful_reciprocal_range_l3497_349776

theorem meaningful_reciprocal_range (x : ℝ) :
  (∃ y : ℝ, y = 1 / (x + 2)) ↔ x ≠ -2 := by
  sorry

end NUMINAMATH_CALUDE_meaningful_reciprocal_range_l3497_349776


namespace NUMINAMATH_CALUDE_jellybeans_left_l3497_349753

/-- Calculates the number of jellybeans left in a jar after a class party --/
theorem jellybeans_left (total_jellybeans : ℕ) 
  (kindergarteners first_graders second_graders : ℕ)
  (absent_kindergarteners absent_second_graders : ℕ)
  (present_kindergartener_rate first_grader_rate : ℕ)
  (absent_kindergartener_rate absent_second_grader_rate : ℕ)
  (h1 : total_jellybeans = 500)
  (h2 : kindergarteners = 10)
  (h3 : first_graders = 10)
  (h4 : second_graders = 10)
  (h5 : absent_kindergarteners = 2)
  (h6 : absent_second_graders = 3)
  (h7 : present_kindergartener_rate = 3)
  (h8 : first_grader_rate = 5)
  (h9 : absent_kindergartener_rate = 5)
  (h10 : absent_second_grader_rate = 10) :
  total_jellybeans - 
  ((kindergarteners - absent_kindergarteners) * present_kindergartener_rate +
   first_graders * first_grader_rate +
   (second_graders - absent_second_graders) * (first_graders * first_grader_rate / 2)) = 176 := by
sorry


end NUMINAMATH_CALUDE_jellybeans_left_l3497_349753


namespace NUMINAMATH_CALUDE_cubic_root_sum_cubes_l3497_349752

/-- Given roots r, s, and t of the equation 10x³ + 500x + 1500 = 0,
    prove that (r+s)³ + (t+s)³ + (r+t)³ = -450 -/
theorem cubic_root_sum_cubes (r s t : ℝ) :
  (10 * r^3 + 500 * r + 1500 = 0) →
  (10 * s^3 + 500 * s + 1500 = 0) →
  (10 * t^3 + 500 * t + 1500 = 0) →
  (r + s)^3 + (t + s)^3 + (r + t)^3 = -450 := by
  sorry


end NUMINAMATH_CALUDE_cubic_root_sum_cubes_l3497_349752


namespace NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l3497_349705

theorem equilateral_triangle_perimeter (s : ℝ) (h : s > 0) : 
  (s^2 * Real.sqrt 3) / 4 = 2 * s → 3 * s = 8 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l3497_349705


namespace NUMINAMATH_CALUDE_divisibility_of_forms_l3497_349735

/-- Represents a six-digit number in the form ABCDEF --/
def SixDigitNumber (A B C D E F : ℕ) : ℕ := 
  100000 * A + 10000 * B + 1000 * C + 100 * D + 10 * E + F

/-- The form PQQPQQ --/
def FormA (P Q : ℕ) : ℕ := SixDigitNumber P Q Q P Q Q

/-- The form PQPQPQ --/
def FormB (P Q : ℕ) : ℕ := SixDigitNumber P Q P Q P Q

/-- The form QPQPQP --/
def FormC (P Q : ℕ) : ℕ := SixDigitNumber Q P Q P Q P

/-- The form PPPPPP --/
def FormD (P : ℕ) : ℕ := SixDigitNumber P P P P P P

/-- The form PPPQQQ --/
def FormE (P Q : ℕ) : ℕ := SixDigitNumber P P P Q Q Q

theorem divisibility_of_forms (P Q : ℕ) :
  (∃ (k : ℕ), FormA P Q = 7 * k) ∧
  (∃ (k : ℕ), FormB P Q = 7 * k) ∧
  (∃ (k : ℕ), FormC P Q = 7 * k) ∧
  (∃ (k : ℕ), FormD P = 7 * k) ∧
  ¬(∀ (P Q : ℕ), ∃ (k : ℕ), FormE P Q = 7 * k) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_forms_l3497_349735


namespace NUMINAMATH_CALUDE_coeff_x3_sum_l3497_349726

/-- The coefficient of x^3 in the expansion of (1-x)^n -/
def coeff_x3 (n : ℕ) : ℤ := (-1)^3 * Nat.choose n 3

/-- The sum of coefficients of x^3 in the expansion of (1-x)^5 + (1-x)^6 + (1-x)^7 + (1-x)^8 -/
def total_coeff : ℤ := coeff_x3 5 + coeff_x3 6 + coeff_x3 7 + coeff_x3 8

theorem coeff_x3_sum : total_coeff = -121 := by sorry

end NUMINAMATH_CALUDE_coeff_x3_sum_l3497_349726


namespace NUMINAMATH_CALUDE_factorial_8_divisors_l3497_349784

def factorial_8 : ℕ := 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1

theorem factorial_8_divisors :
  (factorial_8 = 2^7 * 3^2 * 5 * 7) →
  (∃ (even_divisors : Finset ℕ) (even_divisors_multiple_2_3 : Finset ℕ),
    (∀ d ∈ even_divisors, d ∣ factorial_8 ∧ 2 ∣ d) ∧
    (∀ d ∈ even_divisors_multiple_2_3, d ∣ factorial_8 ∧ 2 ∣ d ∧ 3 ∣ d) ∧
    even_divisors.card = 84 ∧
    even_divisors_multiple_2_3.card = 56) :=
by sorry

end NUMINAMATH_CALUDE_factorial_8_divisors_l3497_349784


namespace NUMINAMATH_CALUDE_five_fourths_of_three_and_one_third_l3497_349785

theorem five_fourths_of_three_and_one_third (x : ℚ) :
  x = 3 + 1 / 3 → (5 / 4 : ℚ) * x = 25 / 6 := by
  sorry

end NUMINAMATH_CALUDE_five_fourths_of_three_and_one_third_l3497_349785


namespace NUMINAMATH_CALUDE_sum_of_quadratic_solutions_l3497_349767

theorem sum_of_quadratic_solutions : 
  let f : ℝ → ℝ := λ x => x^2 - 6*x + 5 - (2*x - 8)
  ∃ x₁ x₂ : ℝ, (f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂) ∧ x₁ + x₂ = 8 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_quadratic_solutions_l3497_349767


namespace NUMINAMATH_CALUDE_simple_interest_problem_l3497_349759

theorem simple_interest_problem (P R : ℝ) : 
  P > 0 → R > 0 → 
  P * (R + 3) * 3 / 100 - P * R * 3 / 100 = 90 → 
  P = 1000 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l3497_349759


namespace NUMINAMATH_CALUDE_no_primes_in_range_l3497_349716

theorem no_primes_in_range (n : ℕ) (h : n > 1) :
  ∀ k, n! + 1 < k ∧ k < n! + 2*n → ¬ Nat.Prime k := by
  sorry

end NUMINAMATH_CALUDE_no_primes_in_range_l3497_349716


namespace NUMINAMATH_CALUDE_second_to_last_term_l3497_349794

-- Define the sequence type
def Sequence := Fin 201 → ℕ

-- Define the properties of the sequence
def ValidSequence (a : Sequence) : Prop :=
  (a 0 = 19999) ∧ 
  (a 200 = 19999) ∧
  (∃ t : ℕ+, ∀ n : Fin 199, 
    a (n + 1) + t = (a n + a (n + 2)) / 2)

-- Theorem statement
theorem second_to_last_term (a : Sequence) 
  (h : ValidSequence a) : a 199 = 19800 := by
  sorry

end NUMINAMATH_CALUDE_second_to_last_term_l3497_349794


namespace NUMINAMATH_CALUDE_cricket_match_average_l3497_349728

/-- Given five cricket match scores x, y, a, b, and c, prove that their average is 36 -/
theorem cricket_match_average (x y a b c : ℝ) : 
  (x + y) / 2 = 30 →
  (a + b + c) / 3 = 40 →
  x ≤ 60 ∧ y ≤ 60 ∧ a ≤ 60 ∧ b ≤ 60 ∧ c ≤ 60 →
  (x + y ≥ 100 ∨ a + b + c ≥ 100) →
  (x + y + a + b + c) / 5 = 36 := by
  sorry

end NUMINAMATH_CALUDE_cricket_match_average_l3497_349728


namespace NUMINAMATH_CALUDE_power_fraction_simplification_l3497_349704

theorem power_fraction_simplification :
  (5^2022)^2 - (5^2020)^2 / (5^2021)^2 - (5^2019)^2 = 5^2 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_simplification_l3497_349704


namespace NUMINAMATH_CALUDE_no_snow_probability_l3497_349797

theorem no_snow_probability (p : ℚ) (h : p = 2/3) :
  (1 - p)^5 = 1/243 := by
  sorry

end NUMINAMATH_CALUDE_no_snow_probability_l3497_349797


namespace NUMINAMATH_CALUDE_circle_area_ratio_after_tripling_radius_l3497_349733

theorem circle_area_ratio_after_tripling_radius (r : ℝ) (h : r > 0) :
  (π * r^2) / (π * (3*r)^2) = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_after_tripling_radius_l3497_349733


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3497_349774

theorem quadratic_equation_roots (m : ℝ) : 
  let f : ℝ → ℝ := λ x => x^2 - (2*m - 1)*x + m^2
  ∃ x₁ x₂ : ℝ, (∀ x : ℝ, f x = 0 ↔ x = x₁ ∨ x = x₂) ∧ 
  (x₁ ≠ x₂) ∧
  ((x₁ + 1) * (x₂ + 1) = 3) →
  m = -3 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3497_349774


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l3497_349772

/-- The distance between the foci of the ellipse 9x^2 + y^2 = 36 is 8√2 -/
theorem ellipse_foci_distance (x y : ℝ) :
  (9 * x^2 + y^2 = 36) → (∃ f₁ f₂ : ℝ × ℝ, 
    (f₁.1 - f₂.1)^2 + (f₁.2 - f₂.2)^2 = 128) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l3497_349772


namespace NUMINAMATH_CALUDE_sin_75_deg_l3497_349700

/-- Proves that the sine of 75 degrees is equal to (√6 + √2) / 4 -/
theorem sin_75_deg : Real.sin (75 * π / 180) = (Real.sqrt 6 + Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_75_deg_l3497_349700


namespace NUMINAMATH_CALUDE_birthday_month_l3497_349773

def is_valid_day (d : ℕ) : Prop := 1 ≤ d ∧ d ≤ 31

def is_valid_month (m : ℕ) : Prop := 1 ≤ m ∧ m ≤ 12

theorem birthday_month (d m : ℕ) (h1 : is_valid_day d) (h2 : is_valid_month m) 
  (h3 : d * m = 248) : m = 8 := by
  sorry

end NUMINAMATH_CALUDE_birthday_month_l3497_349773


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3497_349751

theorem quadratic_equation_solution : ∃ (x₁ x₂ : ℝ), 
  (x₁^2 - 2*x₁ - 3 = 0) ∧ 
  (x₂^2 - 2*x₂ - 3 = 0) ∧ 
  x₁ = 3 ∧ 
  x₂ = -1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3497_349751


namespace NUMINAMATH_CALUDE_divisibility_np_minus_n_l3497_349770

theorem divisibility_np_minus_n (p : Nat) (n : Int) (h : p = 3 ∨ p = 7 ∨ p = 13) :
  ∃ k : Int, n^p - n = k * p := by
  sorry

end NUMINAMATH_CALUDE_divisibility_np_minus_n_l3497_349770


namespace NUMINAMATH_CALUDE_total_pieces_l3497_349798

/-- Represents the number of small pieces in Figure n of Nair's puzzle -/
def small_pieces (n : ℕ) : ℕ := 4 * n

/-- Represents the number of large pieces in Figure n of Nair's puzzle -/
def large_pieces (n : ℕ) : ℕ := n^2 - n

/-- Theorem stating that the total number of pieces in Figure n is n^2 + 3n -/
theorem total_pieces (n : ℕ) : small_pieces n + large_pieces n = n^2 + 3*n := by
  sorry

#eval small_pieces 20 + large_pieces 20  -- Should output 460

end NUMINAMATH_CALUDE_total_pieces_l3497_349798


namespace NUMINAMATH_CALUDE_sum_of_digits_of_calculation_l3497_349786

def calculation : ℕ := 100 * 1 + 50 * 2 + 25 * 4 + 2010

def sum_of_digits (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n % 10 + sum_of_digits (n / 10))

theorem sum_of_digits_of_calculation :
  sum_of_digits calculation = 303 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_calculation_l3497_349786


namespace NUMINAMATH_CALUDE_determinant_scaling_l3497_349747

theorem determinant_scaling (x y z w : ℝ) :
  Matrix.det !![x, y; z, w] = 3 →
  Matrix.det !![3*x, 3*y; 6*z, 6*w] = 54 := by
  sorry

end NUMINAMATH_CALUDE_determinant_scaling_l3497_349747


namespace NUMINAMATH_CALUDE_keith_pears_l3497_349766

/-- Given that Jason picked 46 pears, Mike picked 12 pears, and the total number of pears picked was 105, prove that Keith picked 47 pears. -/
theorem keith_pears (jason_pears mike_pears total_pears : ℕ) 
  (h1 : jason_pears = 46)
  (h2 : mike_pears = 12)
  (h3 : total_pears = 105) :
  total_pears - (jason_pears + mike_pears) = 47 :=
by sorry

end NUMINAMATH_CALUDE_keith_pears_l3497_349766


namespace NUMINAMATH_CALUDE_second_machine_time_l3497_349718

theorem second_machine_time (t1 t_combined : ℝ) (h1 : t1 = 9) (h2 : t_combined = 4.235294117647059) : 
  let t2 := (t1 * t_combined) / (t1 - t_combined)
  t2 = 8 := by sorry

end NUMINAMATH_CALUDE_second_machine_time_l3497_349718


namespace NUMINAMATH_CALUDE_valid_domains_for_range_l3497_349710

def f (x : ℝ) := x^2 - 2*x + 2

theorem valid_domains_for_range (a b : ℝ) (h : a < b) :
  (∀ x ∈ Set.Icc a b, 1 ≤ f x ∧ f x ≤ 2) →
  (∀ y ∈ Set.Icc 1 2, ∃ x ∈ Set.Icc a b, f x = y) →
  (a = 0 ∧ b = 1) ∨ (a = 1/4 ∧ b = 2) :=
by sorry

end NUMINAMATH_CALUDE_valid_domains_for_range_l3497_349710


namespace NUMINAMATH_CALUDE_solution_set_is_singleton_l3497_349748

def solution_set : Set (ℝ × ℝ) := {(x, y) | 2*x + y = 0 ∧ x - y + 3 = 0}

theorem solution_set_is_singleton : solution_set = {(-1, 2)} := by sorry

end NUMINAMATH_CALUDE_solution_set_is_singleton_l3497_349748


namespace NUMINAMATH_CALUDE_mike_seeds_left_l3497_349743

/-- The number of seeds Mike has left after feeding the birds -/
def seeds_left (total : ℕ) (left : ℕ) (right_multiplier : ℕ) (late : ℕ) : ℕ :=
  total - (left + right_multiplier * left + late)

/-- Theorem stating that Mike has 30 seeds left -/
theorem mike_seeds_left :
  seeds_left 120 20 2 30 = 30 := by
  sorry

end NUMINAMATH_CALUDE_mike_seeds_left_l3497_349743


namespace NUMINAMATH_CALUDE_nth_root_equation_l3497_349754

theorem nth_root_equation (n : ℕ) : n = 3 →
  (((17 * Real.sqrt 5 + 38) ^ (1 / n : ℝ)) + ((17 * Real.sqrt 5 - 38) ^ (1 / n : ℝ))) = Real.sqrt 20 :=
by sorry

end NUMINAMATH_CALUDE_nth_root_equation_l3497_349754


namespace NUMINAMATH_CALUDE_point_on_line_expression_l3497_349795

/-- For any point (a,b) on the line y = 4x + 3, the expression 4a - b - 2 equals -5 -/
theorem point_on_line_expression (a b : ℝ) : b = 4 * a + 3 → 4 * a - b - 2 = -5 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_expression_l3497_349795


namespace NUMINAMATH_CALUDE_fifth_inequality_l3497_349780

theorem fifth_inequality (n : ℕ) (h : n = 6) : 
  1 + (1 / 2^2) + (1 / 3^2) + (1 / 4^2) + (1 / 5^2) + (1 / 6^2) < (2 * n - 1) / n :=
by sorry

end NUMINAMATH_CALUDE_fifth_inequality_l3497_349780


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l3497_349799

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A regular polygon satisfying the given condition has 3 sides -/
theorem regular_polygon_sides : ∃ (n : ℕ), n ≥ 3 ∧ n - num_diagonals n = 3 ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l3497_349799


namespace NUMINAMATH_CALUDE_cube_surface_area_l3497_349768

theorem cube_surface_area (volume : ℝ) (side : ℝ) (surface_area : ℝ) :
  volume = 8 →
  volume = side^3 →
  surface_area = 6 * side^2 →
  surface_area = 24 := by
sorry

end NUMINAMATH_CALUDE_cube_surface_area_l3497_349768


namespace NUMINAMATH_CALUDE_binary_1100_eq_12_l3497_349778

/-- Converts a binary number represented as a list of bits (0 or 1) to its decimal equivalent. -/
def binary_to_decimal (bits : List Nat) : Nat :=
  bits.reverse.enum.foldl (fun acc (i, b) => acc + b * 2^i) 0

/-- The binary representation of 1100 -/
def binary_1100 : List Nat := [1, 1, 0, 0]

/-- Theorem stating that the binary number 1100 is equal to the decimal number 12 -/
theorem binary_1100_eq_12 : binary_to_decimal binary_1100 = 12 := by
  sorry

end NUMINAMATH_CALUDE_binary_1100_eq_12_l3497_349778


namespace NUMINAMATH_CALUDE_permutation_of_two_equals_twelve_l3497_349740

theorem permutation_of_two_equals_twelve (n : ℕ) : n * (n - 1) = 12 → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_permutation_of_two_equals_twelve_l3497_349740


namespace NUMINAMATH_CALUDE_function_value_implies_a_value_l3497_349764

noncomputable def f (a x : ℝ) : ℝ := x - Real.log (x + 2) + Real.exp (x - a) + 4 * Real.exp (a - x)

theorem function_value_implies_a_value :
  ∃ (a x₀ : ℝ), f a x₀ = 3 → a = -Real.log 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_function_value_implies_a_value_l3497_349764


namespace NUMINAMATH_CALUDE_classroom_shirts_problem_l3497_349769

theorem classroom_shirts_problem (total_students : ℕ) 
  (striped_ratio : ℚ) (shorts_difference : ℕ) : 
  total_students = 81 →
  striped_ratio = 2 / 3 →
  shorts_difference = 19 →
  let striped := (total_students : ℚ) * striped_ratio
  let checkered := total_students - striped.floor
  let shorts := checkered + shorts_difference
  striped.floor - shorts = 8 := by
sorry

end NUMINAMATH_CALUDE_classroom_shirts_problem_l3497_349769


namespace NUMINAMATH_CALUDE_total_pumpkin_pies_l3497_349734

theorem total_pumpkin_pies (pinky helen emily : ℕ) 
  (h1 : pinky = 147) 
  (h2 : helen = 56) 
  (h3 : emily = 89) : 
  pinky + helen + emily = 292 := by
  sorry

end NUMINAMATH_CALUDE_total_pumpkin_pies_l3497_349734


namespace NUMINAMATH_CALUDE_triangle_perimeter_bound_l3497_349729

theorem triangle_perimeter_bound (a b c : ℝ) : 
  a = 7 → b = 23 → a + b > c → a + c > b → b + c > a → a + b + c < 60 := by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_bound_l3497_349729


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l3497_349742

/-- Given two lines in a plane, this theorem states that if one line passes through 
    a specific point and is parallel to the other line, then it has a specific equation. -/
theorem parallel_line_through_point (x y : ℝ) : 
  (x - 2*y + 3 = 0) →  -- Given line
  (2 = x ∧ 0 = y) →    -- Point (2, 0)
  (2*y - x + 2 = 0) →  -- Equation to prove
  ∃ (m b : ℝ), (y = m*x + b ∧ 2*y - x + 2 = 0) ∧ 
               (∃ (c : ℝ), x - 2*y + c = 0) :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l3497_349742


namespace NUMINAMATH_CALUDE_function_value_at_negative_one_l3497_349708

/-- Given a function f(x) = ax³ + b*sin(x) + 1 where f(1) = 5, prove that f(-1) = -3 -/
theorem function_value_at_negative_one 
  (a b : ℝ) 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * x^3 + b * Real.sin x + 1) 
  (h2 : f 1 = 5) : 
  f (-1) = -3 := by
sorry

end NUMINAMATH_CALUDE_function_value_at_negative_one_l3497_349708


namespace NUMINAMATH_CALUDE_unpainted_area_45_degree_cross_l3497_349796

/-- The area of the unpainted region when two boards cross at 45 degrees -/
theorem unpainted_area_45_degree_cross (board_width : ℝ) (cross_angle : ℝ) : 
  board_width = 5 → cross_angle = 45 → 
  (board_width * (board_width * Real.sqrt 2)) = 25 * Real.sqrt 2 := by
  sorry

#check unpainted_area_45_degree_cross

end NUMINAMATH_CALUDE_unpainted_area_45_degree_cross_l3497_349796


namespace NUMINAMATH_CALUDE_f_no_real_roots_l3497_349762

/-- Defines the polynomial f(x) for a given positive integer n -/
def f (n : ℕ+) (x : ℝ) : ℝ :=
  (2 * n.val + 1) * x^(2 * n.val) - 2 * n.val * x^(2 * n.val - 1) + 
  (2 * n.val - 1) * x^(2 * n.val - 2) - 3 * x^2 + 2 * x - 1

/-- Theorem stating that f(x) has no real roots for any positive integer n -/
theorem f_no_real_roots (n : ℕ+) : ∀ x : ℝ, f n x ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_f_no_real_roots_l3497_349762


namespace NUMINAMATH_CALUDE_quadratic_equation_m_value_l3497_349737

theorem quadratic_equation_m_value : 
  ∀ m : ℤ, 
  (∀ x : ℝ, ∃ a b c : ℝ, (m - 1) * x^(m^2 + 1) + 2*x - 3 = a*x^2 + b*x + c) →
  m = -1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_m_value_l3497_349737


namespace NUMINAMATH_CALUDE_square_side_length_sum_l3497_349779

theorem square_side_length_sum : ∃ (a b : ℕ), a^2 + b^2 = 100 ∧ a + b = 14 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_sum_l3497_349779


namespace NUMINAMATH_CALUDE_eggs_used_for_omelet_l3497_349736

theorem eggs_used_for_omelet (initial_eggs : ℕ) (chickens : ℕ) (eggs_per_chicken : ℕ) (final_eggs : ℕ) : 
  initial_eggs = 10 →
  chickens = 2 →
  eggs_per_chicken = 3 →
  final_eggs = 11 →
  initial_eggs + chickens * eggs_per_chicken - final_eggs = 7 :=
by
  sorry

#check eggs_used_for_omelet

end NUMINAMATH_CALUDE_eggs_used_for_omelet_l3497_349736


namespace NUMINAMATH_CALUDE_square_difference_sum_l3497_349782

theorem square_difference_sum : 
  21^2 - 19^2 + 17^2 - 15^2 + 13^2 - 11^2 + 9^2 - 7^2 + 5^2 - 3^2 = 240 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_sum_l3497_349782


namespace NUMINAMATH_CALUDE_inequality_proof_l3497_349792

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum : a + b + c = 1) :
  (a - b*c)/(a + b*c) + (b - c*a)/(b + c*a) + (c - a*b)/(c + a*b) ≤ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3497_349792


namespace NUMINAMATH_CALUDE_ladder_length_is_twice_h_l3497_349702

/-- The length of a ladder resting against two walls in an alley -/
def ladder_length (w h k : ℝ) : ℝ :=
  2 * h

/-- Theorem: The length of the ladder is twice the height at point Q -/
theorem ladder_length_is_twice_h (w h k : ℝ) (hw : w > 0) (hh : h > 0) (hk : k > 0) :
  ladder_length w h k = 2 * h :=
by
  sorry

#check ladder_length_is_twice_h

end NUMINAMATH_CALUDE_ladder_length_is_twice_h_l3497_349702


namespace NUMINAMATH_CALUDE_sequence_length_l3497_349707

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

theorem sequence_length : 
  ∃ n : ℕ, n = 757 ∧ arithmetic_sequence 2 4 n = 3026 := by sorry

end NUMINAMATH_CALUDE_sequence_length_l3497_349707
