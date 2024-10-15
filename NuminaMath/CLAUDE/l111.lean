import Mathlib

namespace NUMINAMATH_CALUDE_running_track_l111_11161

/-- Given two concentric circles with radii r₁ and r₂, where the difference in their circumferences is 24π feet, prove the width of the track and the enclosed area. -/
theorem running_track (r₁ r₂ : ℝ) (h : 2 * Real.pi * r₁ - 2 * Real.pi * r₂ = 24 * Real.pi) :
  r₁ - r₂ = 12 ∧ Real.pi * (r₁^2 - r₂^2) = Real.pi * (24 * r₂ + 144) :=
by sorry

end NUMINAMATH_CALUDE_running_track_l111_11161


namespace NUMINAMATH_CALUDE_evaluate_polynomial_l111_11102

theorem evaluate_polynomial (x : ℤ) (h : x = -2) : x^3 + x^2 + x + 1 = -5 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_polynomial_l111_11102


namespace NUMINAMATH_CALUDE_peach_cost_per_pound_l111_11159

def initial_amount : ℚ := 20
def final_amount : ℚ := 14
def pounds_of_peaches : ℚ := 3

theorem peach_cost_per_pound :
  (initial_amount - final_amount) / pounds_of_peaches = 2 := by
  sorry

end NUMINAMATH_CALUDE_peach_cost_per_pound_l111_11159


namespace NUMINAMATH_CALUDE_harriet_siblings_product_l111_11100

/-- Represents a family with a specific structure -/
structure Family where
  harry_sisters : Nat
  harry_brothers : Nat

/-- Calculates the number of Harriet's sisters (excluding herself) -/
def harriet_sisters (f : Family) : Nat :=
  f.harry_sisters - 1

/-- Calculates the number of Harriet's brothers -/
def harriet_brothers (f : Family) : Nat :=
  f.harry_brothers

/-- Theorem stating that the product of Harriet's siblings is 9 -/
theorem harriet_siblings_product (f : Family) 
  (h1 : f.harry_sisters = 4) 
  (h2 : f.harry_brothers = 3) : 
  (harriet_sisters f) * (harriet_brothers f) = 9 := by
  sorry


end NUMINAMATH_CALUDE_harriet_siblings_product_l111_11100


namespace NUMINAMATH_CALUDE_bread_distribution_problem_l111_11144

theorem bread_distribution_problem :
  ∃! (m w c : ℕ),
    m + w + c = 12 ∧
    2 * m + (1/2) * w + (1/4) * c = 12 ∧
    m ≥ 0 ∧ w ≥ 0 ∧ c ≥ 0 ∧
    m = 5 ∧ w = 1 ∧ c = 6 :=
by sorry

end NUMINAMATH_CALUDE_bread_distribution_problem_l111_11144


namespace NUMINAMATH_CALUDE_remainder_problem_l111_11118

theorem remainder_problem (N : ℕ) : 
  (∃ r : ℕ, N = 44 * 432 + r ∧ r < 44) → 
  (∃ q : ℕ, N = 30 * q + 18) → 
  N % 44 = 18 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l111_11118


namespace NUMINAMATH_CALUDE_consecutive_negative_integers_product_sum_l111_11125

theorem consecutive_negative_integers_product_sum (n : ℤ) :
  n < 0 ∧ n * (n + 1) = 2184 → n + (n + 1) = -95 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_negative_integers_product_sum_l111_11125


namespace NUMINAMATH_CALUDE_distance_before_meeting_is_100_l111_11182

/-- The distance between two trains one hour before they meet -/
def distance_before_meeting (total_distance : ℝ) (speed_A speed_B : ℝ) (delay : ℝ) : ℝ :=
  let relative_speed := speed_A + speed_B
  let time_to_meet := (total_distance - speed_A * delay) / relative_speed
  relative_speed

/-- Theorem stating the distance between trains one hour before meeting -/
theorem distance_before_meeting_is_100 :
  distance_before_meeting 435 45 55 (40/60) = 100 := by
  sorry

end NUMINAMATH_CALUDE_distance_before_meeting_is_100_l111_11182


namespace NUMINAMATH_CALUDE_waiter_dishes_served_l111_11164

theorem waiter_dishes_served : 
  let num_tables : ℕ := 7
  let women_per_table : ℕ := 7
  let men_per_table : ℕ := 2
  let courses_per_woman : ℕ := 3
  let courses_per_man : ℕ := 4
  let shared_courses_women : ℕ := 1
  let shared_courses_men : ℕ := 2

  let dishes_per_table : ℕ := 
    women_per_table * courses_per_woman + 
    men_per_table * courses_per_man - 
    shared_courses_women - 
    shared_courses_men

  num_tables * dishes_per_table = 182
  := by sorry

end NUMINAMATH_CALUDE_waiter_dishes_served_l111_11164


namespace NUMINAMATH_CALUDE_triangle_inequality_l111_11189

theorem triangle_inequality (A B C : Real) (h : A + B + C = π) :
  (Real.sqrt (Real.sin A * Real.sin B)) / (Real.sin (C / 2)) ≥ 3 * Real.sqrt 3 * Real.tan (A / 2) * Real.tan (B / 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l111_11189


namespace NUMINAMATH_CALUDE_irwin_two_point_baskets_l111_11130

/-- Represents the number of baskets scored for each point value -/
structure BasketCount where
  two_point : ℕ
  five_point : ℕ
  eleven_point : ℕ
  thirteen_point : ℕ

/-- Calculates the product of point values for a given BasketCount -/
def pointValueProduct (b : BasketCount) : ℕ :=
  2^b.two_point * 5^b.five_point * 11^b.eleven_point * 13^b.thirteen_point

theorem irwin_two_point_baskets :
  ∀ b : BasketCount,
    pointValueProduct b = 2420 →
    b.eleven_point = 2 →
    b.two_point = 2 := by
  sorry

end NUMINAMATH_CALUDE_irwin_two_point_baskets_l111_11130


namespace NUMINAMATH_CALUDE_roses_in_vase_l111_11137

/-- The number of roses initially in the vase -/
def initial_roses : ℕ := 7

/-- The number of orchids initially in the vase -/
def initial_orchids : ℕ := 12

/-- The number of orchids in the vase now -/
def current_orchids : ℕ := 20

/-- The difference between the number of orchids and roses in the vase now -/
def orchid_rose_difference : ℕ := 9

/-- The number of roses in the vase now -/
def current_roses : ℕ := 11

theorem roses_in_vase :
  current_orchids = current_roses + orchid_rose_difference :=
by sorry

end NUMINAMATH_CALUDE_roses_in_vase_l111_11137


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l111_11106

theorem sqrt_equation_solution (y : ℝ) : 
  Real.sqrt (5 * y + 15) = 15 → y = 42 := by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l111_11106


namespace NUMINAMATH_CALUDE_faster_train_speed_l111_11196

-- Define the parameters
def train_length : ℝ := 500  -- in meters
def slower_train_speed : ℝ := 30  -- in km/hr
def passing_time : ℝ := 47.99616030717543  -- in seconds

-- Define the theorem
theorem faster_train_speed :
  ∃ (faster_speed : ℝ),
    faster_speed > slower_train_speed ∧
    faster_speed = 45 ∧
    (faster_speed + slower_train_speed) * (passing_time / 3600) = 2 * train_length / 1000 :=
by sorry

end NUMINAMATH_CALUDE_faster_train_speed_l111_11196


namespace NUMINAMATH_CALUDE_sum_of_seven_place_values_l111_11103

theorem sum_of_seven_place_values (n : ℚ) (h : n = 87953.0727) :
  (7000 : ℚ) + (7 / 100 : ℚ) + (7 / 10000 : ℚ) = 7000.0707 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_seven_place_values_l111_11103


namespace NUMINAMATH_CALUDE_change_calculation_l111_11179

def laptop_price : ℝ := 600
def smartphone_price : ℝ := 400
def tablet_price : ℝ := 250
def headphone_price : ℝ := 100
def discount_rate : ℝ := 0.1
def tax_rate : ℝ := 0.05
def num_laptops : ℕ := 2
def num_smartphones : ℕ := 4
def num_tablets : ℕ := 3
def num_headphones : ℕ := 5
def initial_amount : ℝ := 5000

theorem change_calculation : 
  let total_before_discount := 
    num_laptops * laptop_price + 
    num_smartphones * smartphone_price + 
    num_tablets * tablet_price + 
    num_headphones * headphone_price
  let discount := 
    discount_rate * (num_laptops * laptop_price + num_tablets * tablet_price)
  let total_after_discount := total_before_discount - discount
  let tax := tax_rate * total_after_discount
  let final_price := total_after_discount + tax
  initial_amount - final_price = 952.25 := by sorry

end NUMINAMATH_CALUDE_change_calculation_l111_11179


namespace NUMINAMATH_CALUDE_crate_width_is_sixteen_l111_11170

/-- Represents the dimensions of a rectangular crate -/
structure CrateDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents a cylindrical gas tank -/
structure GasTank where
  radius : ℝ
  height : ℝ

/-- Checks if a gas tank can fit upright in a crate -/
def fitsInCrate (tank : GasTank) (crate : CrateDimensions) : Prop :=
  (tank.radius * 2 ≤ crate.length ∧ tank.radius * 2 ≤ crate.width) ∨
  (tank.radius * 2 ≤ crate.length ∧ tank.radius * 2 ≤ crate.height) ∨
  (tank.radius * 2 ≤ crate.width ∧ tank.radius * 2 ≤ crate.height)

/-- Theorem: The width of the crate must be 16 feet -/
theorem crate_width_is_sixteen
  (crate : CrateDimensions)
  (tank : GasTank)
  (h1 : crate.length = 12)
  (h2 : crate.height = 18)
  (h3 : tank.radius = 8)
  (h4 : fitsInCrate tank crate)
  (h5 : ∀ t : GasTank, fitsInCrate t crate → t.radius ≤ tank.radius) :
  crate.width = 16 := by
  sorry

end NUMINAMATH_CALUDE_crate_width_is_sixteen_l111_11170


namespace NUMINAMATH_CALUDE_complex_point_on_real_axis_l111_11160

theorem complex_point_on_real_axis (a : ℝ) : 
  (Complex.I + 1) * (Complex.I + a) ∈ Set.range Complex.ofReal → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_point_on_real_axis_l111_11160


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l111_11127

theorem regular_polygon_sides (n : ℕ) (h_regular : n ≥ 3) 
  (h_interior_angle : (n - 2) * 180 / n = 144) : n = 10 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l111_11127


namespace NUMINAMATH_CALUDE_solution_set_a_2_range_of_a_l111_11155

-- Define the function f(x, a)
def f (x a : ℝ) : ℝ := |x - a^2| + |x - 2*a + 1|

-- Part 1: Solution set when a = 2
theorem solution_set_a_2 :
  {x : ℝ | f x 2 ≥ 4} = {x : ℝ | x ≤ 3/2 ∨ x ≥ 11/2} := by sorry

-- Part 2: Range of values for a
theorem range_of_a :
  ∀ x : ℝ, f x a ≥ 4 → a ∈ Set.Iic (-1) ∪ Set.Ici 3 := by sorry

end NUMINAMATH_CALUDE_solution_set_a_2_range_of_a_l111_11155


namespace NUMINAMATH_CALUDE_simple_interest_problem_l111_11194

/-- Given a principal amount P, an unknown interest rate R, and a 10-year period,
    if increasing the interest rate by 5% results in Rs. 400 more interest,
    then P must equal Rs. 800. -/
theorem simple_interest_problem (P R : ℝ) (h : P > 0) (h_R : R > 0) :
  (P * (R + 5) * 10) / 100 = (P * R * 10) / 100 + 400 →
  P = 800 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l111_11194


namespace NUMINAMATH_CALUDE_yellow_jacket_incident_l111_11108

theorem yellow_jacket_incident (total_students : ℕ) 
  (initial_cafeteria_fraction : ℚ) (final_cafeteria_count : ℕ) 
  (cafeteria_to_outside : ℕ) : 
  total_students = 90 →
  initial_cafeteria_fraction = 2/3 →
  final_cafeteria_count = 67 →
  cafeteria_to_outside = 3 →
  (final_cafeteria_count - (initial_cafeteria_fraction * total_students).floor + cafeteria_to_outside) / 
  (total_students - (initial_cafeteria_fraction * total_students).floor) = 1/3 := by
sorry

end NUMINAMATH_CALUDE_yellow_jacket_incident_l111_11108


namespace NUMINAMATH_CALUDE_ice_cream_distribution_l111_11133

theorem ice_cream_distribution (total_sandwiches : ℕ) (num_nieces : ℕ) 
  (h1 : total_sandwiches = 143)
  (h2 : num_nieces = 11) :
  total_sandwiches / num_nieces = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_ice_cream_distribution_l111_11133


namespace NUMINAMATH_CALUDE_no_valid_arrangement_l111_11158

def is_valid_arrangement (perm : List Nat) : Prop :=
  perm.length = 8 ∧
  (∀ n, n ∈ perm → n ∈ [1, 2, 3, 4, 5, 6, 8, 9]) ∧
  (∀ i, i < 7 → (10 * perm[i]! + perm[i+1]!) % 7 = 0)

theorem no_valid_arrangement : ¬∃ perm : List Nat, is_valid_arrangement perm := by
  sorry

end NUMINAMATH_CALUDE_no_valid_arrangement_l111_11158


namespace NUMINAMATH_CALUDE_a_values_l111_11175

def A : Set ℝ := {x | x^2 - 8*x + 15 = 0}
def B (a : ℝ) : Set ℝ := {x | a*x - 1 = 0}

theorem a_values (h : ∀ a : ℝ, B a ⊆ A) : 
  {a : ℝ | B a ⊆ A} = {0, 1/3, 1/5} := by sorry

end NUMINAMATH_CALUDE_a_values_l111_11175


namespace NUMINAMATH_CALUDE_school_students_count_l111_11168

/-- Given the number of pencils and erasers ordered, and the number of each item given to each student,
    calculate the number of students in the school. -/
def calculate_students (total_pencils : ℕ) (total_erasers : ℕ) (pencils_per_student : ℕ) (erasers_per_student : ℕ) : ℕ :=
  min (total_pencils / pencils_per_student) (total_erasers / erasers_per_student)

/-- Theorem stating that the number of students in the school is 65. -/
theorem school_students_count : calculate_students 195 65 3 1 = 65 := by
  sorry

end NUMINAMATH_CALUDE_school_students_count_l111_11168


namespace NUMINAMATH_CALUDE_gcd_12345_6789_l111_11186

theorem gcd_12345_6789 : Nat.gcd 12345 6789 = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_12345_6789_l111_11186


namespace NUMINAMATH_CALUDE_palace_number_puzzle_l111_11185

theorem palace_number_puzzle :
  ∀ (x : ℕ),
    x < 15 →
    (15 - x) + (15 + x) = 30 →
    (15 + x) - (15 - x) = 2 * x →
    2 * x * 30 = 780 →
    x = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_palace_number_puzzle_l111_11185


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l111_11183

/-- Given two arithmetic sequences {a_n} and {b_n}, S_n and T_n are the sums of their first n terms respectively -/
def arithmetic_sequences (a b : ℕ → ℚ) (S T : ℕ → ℚ) : Prop :=
  ∀ n, S n = (n * (a 1 + a n)) / 2 ∧ T n = (n * (b 1 + b n)) / 2

/-- The ratio of S_n to T_n is (7n + 2) / (n + 3) for all n -/
def ratio_condition (S T : ℕ → ℚ) : Prop :=
  ∀ n, S n / T n = (7 * n + 2) / (n + 3)

theorem arithmetic_sequence_ratio
  (a b : ℕ → ℚ) (S T : ℕ → ℚ)
  (h1 : arithmetic_sequences a b S T)
  (h2 : ratio_condition S T) :
  a 5 / b 5 = 65 / 12 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l111_11183


namespace NUMINAMATH_CALUDE_roxy_daily_consumption_l111_11192

/-- Represents the daily water consumption of the siblings --/
structure WaterConsumption where
  theo : ℕ
  mason : ℕ
  roxy : ℕ

/-- Represents the total weekly water consumption of the siblings --/
def weekly_total (wc : WaterConsumption) : ℕ :=
  7 * (wc.theo + wc.mason + wc.roxy)

/-- Theorem stating that given the conditions, Roxy drinks 9 cups of water daily --/
theorem roxy_daily_consumption (wc : WaterConsumption) :
  wc.theo = 8 → wc.mason = 7 → weekly_total wc = 168 → wc.roxy = 9 := by
  sorry

#check roxy_daily_consumption

end NUMINAMATH_CALUDE_roxy_daily_consumption_l111_11192


namespace NUMINAMATH_CALUDE_polynomial_simplification_l111_11119

theorem polynomial_simplification (p : ℝ) :
  (4 * p^4 + 2 * p^3 - 7 * p + 3) + (5 * p^3 - 8 * p^2 + 3 * p + 2) =
  4 * p^4 + 7 * p^3 - 8 * p^2 - 4 * p + 5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l111_11119


namespace NUMINAMATH_CALUDE_identical_geometric_sequences_l111_11153

/-- Two geometric sequences with the same first term -/
def geometric_sequence (a₀ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₀ * q^n

theorem identical_geometric_sequences
  (a₀ : ℝ) (q r : ℝ) :
  (∀ n : ℕ, ∃ s : ℝ, geometric_sequence a₀ q n + geometric_sequence a₀ r n = geometric_sequence (2 * a₀) s n) →
  q = r :=
sorry

end NUMINAMATH_CALUDE_identical_geometric_sequences_l111_11153


namespace NUMINAMATH_CALUDE_no_integer_roots_for_primes_l111_11187

theorem no_integer_roots_for_primes (p q : ℕ) : 
  Prime p → Prime q → ¬∃ (x : ℤ), x^2 + 3*p*x + 5*q = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_roots_for_primes_l111_11187


namespace NUMINAMATH_CALUDE_sandra_theorem_l111_11147

def sandra_problem (savings : ℚ) (mother_gift : ℚ) (father_gift_multiplier : ℚ)
  (candy_cost : ℚ) (jelly_bean_cost : ℚ) (candy_count : ℕ) (jelly_bean_count : ℕ) : Prop :=
  let total_money := savings + mother_gift + (father_gift_multiplier * mother_gift)
  let total_cost := (candy_cost * candy_count) + (jelly_bean_cost * jelly_bean_count)
  let remaining_money := total_money - total_cost
  remaining_money = 11

theorem sandra_theorem :
  sandra_problem 10 4 2 (1/2) (1/5) 14 20 := by
  sorry

end NUMINAMATH_CALUDE_sandra_theorem_l111_11147


namespace NUMINAMATH_CALUDE_jills_shopping_trip_tax_percentage_l111_11141

/-- Represents the spending and tax information for a shopping trip -/
structure ShoppingTrip where
  clothing_percent : ℝ
  food_percent : ℝ
  other_percent : ℝ
  clothing_tax_rate : ℝ
  food_tax_rate : ℝ
  other_tax_rate : ℝ

/-- Calculates the total tax as a percentage of the total amount spent (excluding taxes) -/
def totalTaxPercentage (trip : ShoppingTrip) : ℝ :=
  (trip.clothing_percent * trip.clothing_tax_rate +
   trip.food_percent * trip.food_tax_rate +
   trip.other_percent * trip.other_tax_rate) * 100

/-- Theorem stating that the total tax percentage for Jill's shopping trip is 4.40% -/
theorem jills_shopping_trip_tax_percentage :
  let trip : ShoppingTrip := {
    clothing_percent := 0.50,
    food_percent := 0.20,
    other_percent := 0.30,
    clothing_tax_rate := 0.04,
    food_tax_rate := 0,
    other_tax_rate := 0.08
  }
  totalTaxPercentage trip = 4.40 := by
  sorry

end NUMINAMATH_CALUDE_jills_shopping_trip_tax_percentage_l111_11141


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l111_11120

theorem min_value_sum_reciprocals (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) 
  (sum_eq_3 : x + y + z = 3) : 
  (1/x) + (4/y) + (9/z) ≥ 12 ∧ 
  ∃ (a b c : ℝ), 0 < a ∧ 0 < b ∧ 0 < c ∧ 
    a + b + c = 3 ∧ (1/a) + (4/b) + (9/c) = 12 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l111_11120


namespace NUMINAMATH_CALUDE_mirror_wall_area_ratio_l111_11111

/-- Proves that the ratio of the area of a square mirror to the area of a rectangular wall is 1:2 -/
theorem mirror_wall_area_ratio (mirror_side : ℝ) (wall_width wall_length : ℝ)
  (h1 : mirror_side = 18)
  (h2 : wall_width = 32)
  (h3 : wall_length = 20.25) :
  (mirror_side^2) / (wall_width * wall_length) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_mirror_wall_area_ratio_l111_11111


namespace NUMINAMATH_CALUDE_roundness_of_900000_l111_11139

/-- Roundness of a positive integer is the sum of the exponents in its prime factorization -/
def roundness (n : ℕ+) : ℕ := sorry

/-- The roundness of 900,000 is 12 -/
theorem roundness_of_900000 : roundness 900000 = 12 := by sorry

end NUMINAMATH_CALUDE_roundness_of_900000_l111_11139


namespace NUMINAMATH_CALUDE_gcd_102_238_l111_11145

theorem gcd_102_238 : Nat.gcd 102 238 = 34 := by
  sorry

end NUMINAMATH_CALUDE_gcd_102_238_l111_11145


namespace NUMINAMATH_CALUDE_sum_of_digits_of_triangular_array_rows_l111_11151

def triangular_sum (n : ℕ) : ℕ := n * (n + 1) / 2

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem sum_of_digits_of_triangular_array_rows : ∃ N : ℕ, 
  triangular_sum N = 3003 ∧ sum_of_digits N = 14 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_triangular_array_rows_l111_11151


namespace NUMINAMATH_CALUDE_fourth_root_of_409600000_l111_11110

theorem fourth_root_of_409600000 : (409600000 : ℝ) ^ (1/4 : ℝ) = 80 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_of_409600000_l111_11110


namespace NUMINAMATH_CALUDE_door_width_calculation_l111_11107

/-- Calculates the width of a door given room dimensions and whitewashing costs -/
theorem door_width_calculation (room_length room_width room_height : ℝ)
  (door_height : ℝ) (window_width window_height : ℝ) (num_windows : ℕ)
  (cost_per_sqft total_cost : ℝ) : 
  room_length = 25 ∧ room_width = 15 ∧ room_height = 12 ∧
  door_height = 6 ∧ window_width = 4 ∧ window_height = 3 ∧
  num_windows = 3 ∧ cost_per_sqft = 9 ∧ total_cost = 8154 →
  ∃ (door_width : ℝ),
    (2 * (room_length * room_height + room_width * room_height) - 
     (door_height * door_width + num_windows * window_width * window_height)) * cost_per_sqft = total_cost ∧
    door_width = 3 := by
  sorry

end NUMINAMATH_CALUDE_door_width_calculation_l111_11107


namespace NUMINAMATH_CALUDE_whatsis_whosis_equals_so_plus_so_l111_11188

/-- A structure representing the variables in the problem -/
structure Variables where
  whatsis : ℝ
  whosis : ℝ
  is : ℝ
  so : ℝ
  pos_whatsis : 0 < whatsis
  pos_whosis : 0 < whosis
  pos_is : 0 < is
  pos_so : 0 < so

/-- The main theorem representing the problem -/
theorem whatsis_whosis_equals_so_plus_so (v : Variables) 
  (h1 : v.whatsis = v.so)
  (h2 : v.whosis = v.is)
  (h3 : v.so + v.so = v.is * v.so)
  (h4 : v.whosis = v.so)
  (h5 : v.so + v.so = v.so * v.so)
  (h6 : v.is = 2) :
  v.whosis * v.whatsis = v.so + v.so := by
  sorry


end NUMINAMATH_CALUDE_whatsis_whosis_equals_so_plus_so_l111_11188


namespace NUMINAMATH_CALUDE_min_comparisons_correct_l111_11134

/-- Represents a set of coins with different weights -/
structure CoinSet (n : ℕ) where
  coins : Fin n → ℝ
  different_weights : ∀ i j, i ≠ j → coins i ≠ coins j

/-- Represents a set of balances, including one faulty balance -/
structure BalanceSet (n : ℕ) where
  balances : Fin n → Bool
  one_faulty : ∃ i, balances i = false

/-- The minimum number of comparisons needed to find the heaviest coin -/
def min_comparisons (n : ℕ) : ℕ := 2 * n - 1

/-- The main theorem: proving the minimum number of comparisons -/
theorem min_comparisons_correct (n : ℕ) (h : n > 2) 
  (coins : CoinSet n) (balances : BalanceSet n) :
  min_comparisons n = 2 * n - 1 :=
sorry

end NUMINAMATH_CALUDE_min_comparisons_correct_l111_11134


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l111_11181

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ x : ℝ, (m^2 + 4*m - 5)*x^2 - 4*(m-1)*x + 3 > 0) ↔ (1 ≤ m ∧ m < 19) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l111_11181


namespace NUMINAMATH_CALUDE_gcd_n_cube_plus_25_and_n_plus_3_l111_11148

theorem gcd_n_cube_plus_25_and_n_plus_3 (n : ℕ) (h : n > 3^2) :
  Nat.gcd (n^3 + 5^2) (n + 3) = if (n + 3) % 2 = 0 then 2 else 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_n_cube_plus_25_and_n_plus_3_l111_11148


namespace NUMINAMATH_CALUDE_max_value_sum_fractions_l111_11138

theorem max_value_sum_fractions (a b c : ℝ) 
  (h_nonneg_a : 0 ≤ a) (h_nonneg_b : 0 ≤ b) (h_nonneg_c : 0 ≤ c)
  (h_sum : a + b + c = 2) :
  (a * b / (a + b) + a * c / (a + c) + b * c / (b + c)) ≤ 1 ∧
  ∃ (a' b' c' : ℝ), 0 ≤ a' ∧ 0 ≤ b' ∧ 0 ≤ c' ∧ a' + b' + c' = 2 ∧
    a' * b' / (a' + b') + a' * c' / (a' + c') + b' * c' / (b' + c') = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sum_fractions_l111_11138


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l111_11115

theorem arithmetic_mean_problem (a b c d : ℝ) :
  (a + b) / 2 = 115 →
  (b + c) / 2 = 160 →
  (b + d) / 2 = 175 →
  a - d = -120 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l111_11115


namespace NUMINAMATH_CALUDE_sean_shopping_cost_l111_11172

-- Define the prices and quantities
def soda_price : ℝ := 1
def soda_quantity : ℕ := 4
def soup_quantity : ℕ := 3
def sandwich_quantity : ℕ := 2
def salad_quantity : ℕ := 1

-- Define price relationships
def soup_price : ℝ := 2 * soda_price
def sandwich_price : ℝ := 4 * soup_price
def salad_price : ℝ := 2 * sandwich_price

-- Define discount and tax rates
def discount_rate : ℝ := 0.1
def tax_rate : ℝ := 0.05

-- Calculate total cost before discount and tax
def total_cost : ℝ :=
  soda_price * soda_quantity +
  soup_price * soup_quantity +
  sandwich_price * sandwich_quantity +
  salad_price * salad_quantity

-- Calculate final cost after discount and tax
def final_cost : ℝ :=
  total_cost * (1 - discount_rate) * (1 + tax_rate)

-- Theorem to prove
theorem sean_shopping_cost :
  final_cost = 39.69 := by sorry

end NUMINAMATH_CALUDE_sean_shopping_cost_l111_11172


namespace NUMINAMATH_CALUDE_power_of_product_l111_11109

theorem power_of_product (a b : ℝ) : (-a * b^2)^2 = a^2 * b^4 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l111_11109


namespace NUMINAMATH_CALUDE_union_of_intervals_l111_11142

open Set

theorem union_of_intervals (A B : Set ℝ) :
  A = {x : ℝ | -1 < x ∧ x < 4} →
  B = {x : ℝ | 2 < x ∧ x < 5} →
  A ∪ B = {x : ℝ | -1 < x ∧ x < 5} := by
  sorry

end NUMINAMATH_CALUDE_union_of_intervals_l111_11142


namespace NUMINAMATH_CALUDE_combined_alloy_force_problem_solution_l111_11121

/-- Represents an alloy of two metals -/
structure Alloy where
  mass : ℝ
  ratio : ℝ
  force : ℝ

/-- Theorem stating that the force exerted by a combination of two alloys
    is equal to the sum of their individual forces -/
theorem combined_alloy_force (A B : Alloy) :
  let C : Alloy := ⟨A.mass + B.mass, (A.mass * A.ratio + B.mass * B.ratio) / (A.mass + B.mass), A.force + B.force⟩
  C.force = A.force + B.force := by
  sorry

/-- Given alloys A and B with specified properties, prove that their combination
    exerts a force of 40 N -/
theorem problem_solution (A B : Alloy)
  (hA_mass : A.mass = 6)
  (hA_ratio : A.ratio = 2)
  (hA_force : A.force = 30)
  (hB_mass : B.mass = 3)
  (hB_ratio : B.ratio = 1/5)
  (hB_force : B.force = 10) :
  let C : Alloy := ⟨A.mass + B.mass, (A.mass * A.ratio + B.mass * B.ratio) / (A.mass + B.mass), A.force + B.force⟩
  C.force = 40 := by
  sorry

end NUMINAMATH_CALUDE_combined_alloy_force_problem_solution_l111_11121


namespace NUMINAMATH_CALUDE_other_number_is_two_l111_11123

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem other_number_is_two :
  ∃ n : ℕ, factorial 8 / factorial (8 - n) = 56 ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_other_number_is_two_l111_11123


namespace NUMINAMATH_CALUDE_song_performance_theorem_l111_11124

/-- Represents the number of songs performed by each kid -/
structure SongCounts where
  sarah : ℕ
  emily : ℕ
  daniel : ℕ
  oli : ℕ
  chris : ℕ

/-- The total number of songs performed -/
def totalSongs (counts : SongCounts) : ℕ :=
  (counts.sarah + counts.emily + counts.daniel + counts.oli + counts.chris) / 4

theorem song_performance_theorem (counts : SongCounts) :
  counts.chris = 9 →
  counts.sarah = 3 →
  counts.emily > counts.sarah →
  counts.daniel > counts.sarah →
  counts.oli > counts.sarah →
  counts.emily < counts.chris →
  counts.daniel < counts.chris →
  counts.oli < counts.chris →
  totalSongs counts = 6 :=
by sorry

end NUMINAMATH_CALUDE_song_performance_theorem_l111_11124


namespace NUMINAMATH_CALUDE_movie_expenses_split_l111_11122

theorem movie_expenses_split (num_friends : ℕ) (ticket_price popcorn_price parking_fee milk_tea_price candy_bar_price : ℚ)
  (num_tickets num_popcorn num_milk_tea num_candy_bars : ℕ) :
  num_friends = 4 ∧
  ticket_price = 7 ∧
  popcorn_price = 3/2 ∧
  parking_fee = 4 ∧
  milk_tea_price = 3 ∧
  candy_bar_price = 2 ∧
  num_tickets = 4 ∧
  num_popcorn = 2 ∧
  num_milk_tea = 3 ∧
  num_candy_bars = 4 →
  (num_tickets * ticket_price + num_popcorn * popcorn_price + parking_fee +
   num_milk_tea * milk_tea_price + num_candy_bars * candy_bar_price) / num_friends = 13 :=
by sorry

end NUMINAMATH_CALUDE_movie_expenses_split_l111_11122


namespace NUMINAMATH_CALUDE_max_average_annual_profit_l111_11116

/-- Represents the total profit (in million yuan) for operating 4 buses for x years -/
def total_profit (x : ℕ+) : ℚ :=
  16 * (-2 * x^2 + 23 * x - 50)

/-- Represents the average annual profit (in million yuan) for operating 4 buses for x years -/
def average_annual_profit (x : ℕ+) : ℚ :=
  total_profit x / x

/-- Theorem stating that the average annual profit is maximized when x = 5 -/
theorem max_average_annual_profit :
  ∀ x : ℕ+, average_annual_profit 5 ≥ average_annual_profit x :=
sorry

end NUMINAMATH_CALUDE_max_average_annual_profit_l111_11116


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l111_11135

def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {1, 2}

theorem intersection_of_M_and_N : M ∩ N = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l111_11135


namespace NUMINAMATH_CALUDE_sin_eq_sin_sin_solution_count_l111_11136

theorem sin_eq_sin_sin_solution_count :
  ∃! x : ℝ, 0 ≤ x ∧ x ≤ Real.arcsin 0.99 ∧ Real.sin x = Real.sin (Real.sin x) := by
  sorry

end NUMINAMATH_CALUDE_sin_eq_sin_sin_solution_count_l111_11136


namespace NUMINAMATH_CALUDE_simplify_expression_l111_11177

theorem simplify_expression :
  ∀ w x : ℝ, 3*w + 6*w + 9*w + 12*w + 15*w - 2*x - 4*x - 6*x - 8*x - 10*x + 24 = 45*w - 30*x + 24 :=
by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l111_11177


namespace NUMINAMATH_CALUDE_circular_fortress_volume_l111_11197

theorem circular_fortress_volume : 
  let base_circumference : ℝ := 48
  let height : ℝ := 11
  let π : ℝ := 3
  let radius := base_circumference / (2 * π)
  let volume := π * radius^2 * height
  volume = 2112 := by
  sorry

end NUMINAMATH_CALUDE_circular_fortress_volume_l111_11197


namespace NUMINAMATH_CALUDE_product_of_fractions_l111_11114

theorem product_of_fractions : (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l111_11114


namespace NUMINAMATH_CALUDE_a_10_value_l111_11126

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem a_10_value (a : ℕ → ℝ) :
  arithmetic_sequence a → a 2 = 2 → a 3 = 4 → a 10 = 18 := by
  sorry

end NUMINAMATH_CALUDE_a_10_value_l111_11126


namespace NUMINAMATH_CALUDE_expected_adjacent_pairs_l111_11176

/-- The expected number of adjacent boy-girl pairs in a random permutation of boys and girls -/
theorem expected_adjacent_pairs (b g : ℕ) (hb : b = 8) (hg : g = 12) :
  let total := b + g
  let prob_bg := (b : ℚ) / total * (g : ℚ) / (total - 1)
  let prob_pair := 2 * prob_bg
  let num_pairs := total - 1
  num_pairs * prob_pair = 912 / 95 := by sorry

end NUMINAMATH_CALUDE_expected_adjacent_pairs_l111_11176


namespace NUMINAMATH_CALUDE_quadratic_inequality_l111_11167

theorem quadratic_inequality (a b c : ℝ) :
  (∀ x : ℝ, |x| ≤ 1 → |a * x^2 + b * x + c| ≤ 1) →
  (∀ x : ℝ, |x| ≤ 1 → |c * x^2 + b * x + a| ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l111_11167


namespace NUMINAMATH_CALUDE_probability_not_adjacent_l111_11132

def total_chairs : ℕ := 10
def broken_chairs : Finset ℕ := {5, 8}
def available_chairs : ℕ := total_chairs - broken_chairs.card

def adjacent_pairs : ℕ := 6

theorem probability_not_adjacent :
  (1 - (adjacent_pairs : ℚ) / (available_chairs.choose 2)) = 11/14 := by sorry

end NUMINAMATH_CALUDE_probability_not_adjacent_l111_11132


namespace NUMINAMATH_CALUDE_tank_capacity_tank_capacity_proof_l111_11180

/-- Given a tank where adding 130 gallons when it's 1/6 full makes it 3/5 full,
    prove that the tank's total capacity is 300 gallons. -/
theorem tank_capacity : ℝ → Prop :=
  fun capacity =>
    (capacity / 6 + 130 = 3 * capacity / 5) → capacity = 300

-- The proof is omitted
theorem tank_capacity_proof : ∃ capacity, tank_capacity capacity :=
  sorry

end NUMINAMATH_CALUDE_tank_capacity_tank_capacity_proof_l111_11180


namespace NUMINAMATH_CALUDE_average_listening_time_approx_33_l111_11150

/-- Represents the distribution of audience members and their listening times --/
structure AudienceDistribution where
  total_audience : ℕ
  talk_duration : ℕ
  full_listeners_percent : ℚ
  sleepers_percent : ℚ
  half_listeners_percent : ℚ
  quarter_listeners_percent : ℚ

/-- Calculates the average listening time for the audience --/
def average_listening_time (dist : AudienceDistribution) : ℚ :=
  let full_listeners := (dist.full_listeners_percent * dist.total_audience) * dist.talk_duration
  let sleepers := 0
  let half_listeners := (dist.half_listeners_percent * dist.total_audience) * (dist.talk_duration / 2)
  let quarter_listeners := (dist.quarter_listeners_percent * dist.total_audience) * (dist.talk_duration / 4)
  (full_listeners + sleepers + half_listeners + quarter_listeners) / dist.total_audience

/-- The given audience distribution --/
def lecture_distribution : AudienceDistribution :=
  { total_audience := 200
  , talk_duration := 90
  , full_listeners_percent := 15 / 100
  , sleepers_percent := 15 / 100
  , half_listeners_percent := (1 / 4) * (70 / 100)
  , quarter_listeners_percent := (3 / 4) * (70 / 100)
  }

/-- Theorem stating that the average listening time is approximately 33 minutes --/
theorem average_listening_time_approx_33 :
  ∃ ε > 0, |average_listening_time lecture_distribution - 33| < ε :=
sorry

end NUMINAMATH_CALUDE_average_listening_time_approx_33_l111_11150


namespace NUMINAMATH_CALUDE_point_position_l111_11152

/-- A point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point is to the right of the y-axis -/
def isRightOfYAxis (p : Point) : Prop := p.x > 0

/-- Predicate to check if a point is below the x-axis -/
def isBelowXAxis (p : Point) : Prop := p.y < 0

/-- Theorem stating that if a point is to the right of the y-axis and below the x-axis,
    then its x-coordinate is positive and its y-coordinate is negative -/
theorem point_position (P : Point) 
  (h1 : isRightOfYAxis P) (h2 : isBelowXAxis P) : 
  P.x > 0 ∧ P.y < 0 := by
  sorry

#check point_position

end NUMINAMATH_CALUDE_point_position_l111_11152


namespace NUMINAMATH_CALUDE_salary_change_percentage_loss_l111_11169

theorem salary_change (original : ℝ) (h : original > 0) :
  let decreased := original * (1 - 0.5)
  let final := decreased * (1 + 0.5)
  final = original * 0.75 :=
by
  sorry

theorem percentage_loss : 
  1 - 0.75 = 0.25 :=
by
  sorry

end NUMINAMATH_CALUDE_salary_change_percentage_loss_l111_11169


namespace NUMINAMATH_CALUDE_solution_set_implies_m_value_l111_11131

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := m - |x - 3|

-- State the theorem
theorem solution_set_implies_m_value :
  ∃ m : ℝ, (∀ x : ℝ, f m x > 2 ↔ 2 < x ∧ x < 4) → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_implies_m_value_l111_11131


namespace NUMINAMATH_CALUDE_circle_equation_proof_l111_11104

theorem circle_equation_proof (x y : ℝ) :
  let circle_eq := (x - 5/3)^2 + y^2 = 25/9
  let line_eq := 3*x + y - 5 = 0
  let origin := (0, 0)
  let point := (3, -1)
  (∃ (center : ℝ × ℝ), 
    (center.1 - 5/3)^2 + center.2^2 = 25/9 ∧ 
    3*center.1 + center.2 - 5 = 0) ∧
  ((0 - 5/3)^2 + 0^2 = 25/9) ∧
  ((3 - 5/3)^2 + (-1)^2 = 25/9) →
  circle_eq
:= by sorry

end NUMINAMATH_CALUDE_circle_equation_proof_l111_11104


namespace NUMINAMATH_CALUDE_counterfeit_coin_identification_l111_11173

/-- Represents the result of a weighing operation -/
inductive WeighingResult
  | Left : WeighingResult  -- Left side is heavier
  | Right : WeighingResult -- Right side is heavier
  | Equal : WeighingResult -- Both sides are equal

/-- Represents a weighing operation -/
def Weighing := Nat → Nat → WeighingResult

/-- Represents a strategy to find the counterfeit coin -/
def Strategy := List (Nat × Nat) → Nat

/-- Checks if a strategy correctly identifies the counterfeit coin -/
def isValidStrategy (n : Nat) (strategy : Strategy) : Prop :=
  ∀ (counterfeit : Nat), counterfeit < n →
    ∃ (weighings : List (Nat × Nat)),
      (∀ w ∈ weighings, w.1 < n ∧ w.2 < n) ∧
      (weighings.length ≤ 3) ∧
      (strategy weighings = counterfeit)

theorem counterfeit_coin_identification (n : Nat) (h : n = 10 ∨ n = 27) :
  ∃ (strategy : Strategy), isValidStrategy n strategy :=
sorry

end NUMINAMATH_CALUDE_counterfeit_coin_identification_l111_11173


namespace NUMINAMATH_CALUDE_compound_interest_problem_l111_11146

theorem compound_interest_problem (P r : ℝ) 
  (h1 : P * (1 + r)^2 = 8820)
  (h2 : P * (1 + r)^3 = 9261) : 
  P = 8000 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_problem_l111_11146


namespace NUMINAMATH_CALUDE_least_reducible_fraction_l111_11128

def is_reducible (n : ℕ) : Prop :=
  n > 0 ∧ (n - 17).gcd (7 * n + 5) > 1

theorem least_reducible_fraction :
  (∀ m : ℕ, m < 48 → ¬(is_reducible m)) ∧ is_reducible 48 := by
  sorry

end NUMINAMATH_CALUDE_least_reducible_fraction_l111_11128


namespace NUMINAMATH_CALUDE_final_product_is_twelve_l111_11174

/-- Represents the count of each number on the blackboard -/
structure BoardState :=
  (ones : ℕ)
  (twos : ℕ)
  (threes : ℕ)
  (fours : ℕ)

/-- The operation performed on the board -/
def performOperation (state : BoardState) : BoardState :=
  { ones := state.ones - 1,
    twos := state.twos - 1,
    threes := state.threes - 1,
    fours := state.fours + 2 }

/-- Predicate to check if an operation can be performed -/
def canPerformOperation (state : BoardState) : Prop :=
  state.ones > 0 ∧ state.twos > 0 ∧ state.threes > 0

/-- Predicate to check if the board is in its final state -/
def isFinalState (state : BoardState) : Prop :=
  ¬(canPerformOperation state) ∧ 
  (state.ones + state.twos + state.threes + state.fours = 3)

/-- The initial state of the board -/
def initialState : BoardState :=
  { ones := 11, twos := 22, threes := 33, fours := 44 }

/-- The main theorem to prove -/
theorem final_product_is_twelve :
  ∃ (finalState : BoardState),
    (isFinalState finalState) ∧
    (finalState.ones * finalState.twos * finalState.threes * finalState.fours = 12) := by
  sorry

end NUMINAMATH_CALUDE_final_product_is_twelve_l111_11174


namespace NUMINAMATH_CALUDE_line_parallel_theorem_l111_11143

/-- Represents a plane in 3D space -/
structure Plane

/-- Represents a line in 3D space -/
structure Line

/-- Defines when a line is contained in a plane -/
def Line.containedIn (l : Line) (p : Plane) : Prop :=
  sorry

/-- Defines when a line is parallel to a plane -/
def Line.parallelToPlane (l : Line) (p : Plane) : Prop :=
  sorry

/-- Defines when two lines are coplanar -/
def Line.coplanar (l1 l2 : Line) : Prop :=
  sorry

/-- Defines when two lines are parallel -/
def Line.parallel (l1 l2 : Line) : Prop :=
  sorry

/-- Theorem: If m is contained in plane a, n is parallel to plane a,
    and m and n are coplanar, then m is parallel to n -/
theorem line_parallel_theorem (a : Plane) (m n : Line) :
  m.containedIn a → n.parallelToPlane a → m.coplanar n → m.parallel n :=
by sorry

end NUMINAMATH_CALUDE_line_parallel_theorem_l111_11143


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l111_11171

theorem absolute_value_inequality (x : ℝ) : 
  |x - 1| + |x - 2| > 5 ↔ x < -1 ∨ x > 4 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l111_11171


namespace NUMINAMATH_CALUDE_tims_income_percentage_l111_11162

theorem tims_income_percentage (tim mary juan : ℝ) 
  (h1 : mary = tim * 1.6)
  (h2 : mary = juan * 0.6400000000000001) : 
  tim = juan * 0.4 := by
  sorry

end NUMINAMATH_CALUDE_tims_income_percentage_l111_11162


namespace NUMINAMATH_CALUDE_nines_in_hundred_l111_11163

/-- Count of digit 9 in a single number -/
def count_nines (n : ℕ) : ℕ := sorry

/-- Sum of count_nines for all numbers from 1 to n -/
def total_nines (n : ℕ) : ℕ := sorry

/-- Theorem: The count of the digit 9 in all numbers from 1 to 100 (inclusive) is 20 -/
theorem nines_in_hundred : total_nines 100 = 20 := by sorry

end NUMINAMATH_CALUDE_nines_in_hundred_l111_11163


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l111_11101

/-- An arithmetic sequence with given conditions -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arithmetic : ArithmeticSequence a)
  (h_a2 : a 2 = 1)
  (h_sum : a 3 + a 4 = 8) :
  ∃ d : ℝ, d = 2 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l111_11101


namespace NUMINAMATH_CALUDE_kaylin_is_33_l111_11113

def freyja_age : ℕ := 10

def eli_age (freyja_age : ℕ) : ℕ := freyja_age + 9

def sarah_age (eli_age : ℕ) : ℕ := 2 * eli_age

def kaylin_age (sarah_age : ℕ) : ℕ := sarah_age - 5

theorem kaylin_is_33 : 
  kaylin_age (sarah_age (eli_age freyja_age)) = 33 := by
sorry

end NUMINAMATH_CALUDE_kaylin_is_33_l111_11113


namespace NUMINAMATH_CALUDE_merchant_gross_profit_l111_11117

/-- The merchant's gross profit on a jacket sale --/
theorem merchant_gross_profit (purchase_price : ℝ) (markup_percent : ℝ) (discount_percent : ℝ) : 
  purchase_price = 42 ∧ 
  markup_percent = 0.3 ∧ 
  discount_percent = 0.2 → 
  let selling_price := purchase_price / (1 - markup_percent)
  let discounted_price := selling_price * (1 - discount_percent)
  let gross_profit := discounted_price - purchase_price
  gross_profit = 6 := by
sorry


end NUMINAMATH_CALUDE_merchant_gross_profit_l111_11117


namespace NUMINAMATH_CALUDE_truck_toll_theorem_l111_11105

/-- Calculates the toll for a truck given the number of axles -/
def toll (axles : ℕ) : ℚ :=
  2.50 + 0.50 * (axles - 2)

/-- Calculates the number of axles for a truck given the total number of wheels,
    wheels on the front axle, and wheels on each other axle -/
def calculateAxles (totalWheels frontAxleWheels otherAxleWheels : ℕ) : ℕ :=
  1 + (totalWheels - frontAxleWheels) / otherAxleWheels

theorem truck_toll_theorem (totalWheels frontAxleWheels otherAxleWheels : ℕ)
    (h1 : totalWheels = 18)
    (h2 : frontAxleWheels = 2)
    (h3 : otherAxleWheels = 4) :
  toll (calculateAxles totalWheels frontAxleWheels otherAxleWheels) = 4 :=
by
  sorry

#eval toll (calculateAxles 18 2 4)

end NUMINAMATH_CALUDE_truck_toll_theorem_l111_11105


namespace NUMINAMATH_CALUDE_cos_2x_plus_sin_pi_half_minus_x_properties_l111_11129

/-- The function f(x) = cos(2x) + sin(π/2 - x) has both maximum and minimum values and is an even function. -/
theorem cos_2x_plus_sin_pi_half_minus_x_properties :
  ∃ (f : ℝ → ℝ), 
    (∀ x, f x = Real.cos (2 * x) + Real.sin (Real.pi / 2 - x)) ∧
    (∃ (max min : ℝ), ∀ x, min ≤ f x ∧ f x ≤ max) ∧
    (∀ x, f (-x) = f x) := by
  sorry


end NUMINAMATH_CALUDE_cos_2x_plus_sin_pi_half_minus_x_properties_l111_11129


namespace NUMINAMATH_CALUDE_nancy_bottle_caps_l111_11140

theorem nancy_bottle_caps (initial : ℕ) : initial + 88 = 179 → initial = 91 := by
  sorry

end NUMINAMATH_CALUDE_nancy_bottle_caps_l111_11140


namespace NUMINAMATH_CALUDE_first_player_wins_l111_11193

/-- Represents a game played on a regular polygon -/
structure PolygonGame where
  sides : ℕ
  is_regular : sides > 2

/-- Represents a move in the game -/
inductive Move
| connect (v1 v2 : ℕ)

/-- Represents the state of the game -/
structure GameState where
  game : PolygonGame
  moves : List Move

/-- Checks if a move is valid given the current game state -/
def is_valid_move (state : GameState) (move : Move) : Prop :=
  sorry

/-- Checks if the game is over (no more valid moves) -/
def is_game_over (state : GameState) : Prop :=
  sorry

/-- Determines the winner of the game -/
def winner (state : GameState) : Option Bool :=
  sorry

/-- Represents a strategy for playing the game -/
def Strategy := GameState → Move

/-- Checks if a strategy is winning for the first player -/
def is_winning_strategy (strat : Strategy) (game : PolygonGame) : Prop :=
  sorry

/-- The main theorem: there exists a winning strategy for the first player in a 1968-sided polygon game -/
theorem first_player_wins :
  ∃ (strat : Strategy), is_winning_strategy strat ⟨1968, by norm_num⟩ :=
sorry

end NUMINAMATH_CALUDE_first_player_wins_l111_11193


namespace NUMINAMATH_CALUDE_rectangle_area_l111_11199

theorem rectangle_area (c d w : ℝ) (h1 : w > 0) (h2 : w + 3 > w) 
  (h3 : (c + d)^2 = w^2 + (w + 3)^2) : w * (w + 3) = w^2 + 3*w := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l111_11199


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l111_11165

theorem imaginary_part_of_complex_fraction :
  Complex.im ((2 - Complex.I) / (1 + 2 * Complex.I)) = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l111_11165


namespace NUMINAMATH_CALUDE_log_sequence_l111_11157

theorem log_sequence (a b c : ℝ) (ha : a = Real.log 3 / Real.log 4)
    (hb : b = Real.log 6 / Real.log 4) (hc : c = Real.log 12 / Real.log 4) :
  (b - a = c - b) ∧ ¬(b / a = c / b) := by
  sorry

end NUMINAMATH_CALUDE_log_sequence_l111_11157


namespace NUMINAMATH_CALUDE_james_age_l111_11154

/-- Proves that James' current age is 11 years old, given the conditions of the problem. -/
theorem james_age (julio_age : ℕ) (years_later : ℕ) (james_age : ℕ) : 
  julio_age = 36 →
  years_later = 14 →
  julio_age + years_later = 2 * (james_age + years_later) →
  james_age = 11 := by
sorry

end NUMINAMATH_CALUDE_james_age_l111_11154


namespace NUMINAMATH_CALUDE_state_quarter_fraction_l111_11190

theorem state_quarter_fraction :
  ∀ (total_quarters state_quarters pennsylvania_quarters : ℕ),
    total_quarters = 35 →
    pennsylvania_quarters = 7 →
    2 * pennsylvania_quarters = state_quarters →
    (state_quarters : ℚ) / total_quarters = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_state_quarter_fraction_l111_11190


namespace NUMINAMATH_CALUDE_max_value_abc_l111_11149

theorem max_value_abc (a b : Real) (c : Fin 2) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) :
  ∃ (a₀ b₀ : Real) (c₀ : Fin 2) (ha₀ : 0 ≤ a₀ ∧ a₀ ≤ 1) (hb₀ : 0 ≤ b₀ ∧ b₀ ≤ 1),
    Real.sqrt (a * b * c.val) + Real.sqrt ((1 - a) * (1 - b) * (1 - c.val)) ≤ 1 ∧
    Real.sqrt (a₀ * b₀ * c₀.val) + Real.sqrt ((1 - a₀) * (1 - b₀) * (1 - c₀.val)) = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_abc_l111_11149


namespace NUMINAMATH_CALUDE_part_one_part_two_l111_11156

-- Define the set M
def M (D : Set ℝ) : Set (ℝ → ℝ) :=
  {f | ∀ x y, (x + y) / 2 ∈ D → f ((x + y) / 2) ≥ (f x + f y) / 2 ∧
       (f ((x + y) / 2) = (f x + f y) / 2 ↔ x = y)}

-- Part 1
theorem part_one (f : ℝ → ℝ) (h : f ∈ M (Set.Ioi 0)) :
  f 3 + f 5 ≤ 2 * f 4 := by sorry

-- Part 2
def g : ℝ → ℝ := λ x ↦ -x^2

theorem part_two : g ∈ M Set.univ := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l111_11156


namespace NUMINAMATH_CALUDE_custom_mul_neg_three_two_l111_11198

/-- Custom multiplication operation -/
def custom_mul (a b : ℝ) := a * b - a^2

/-- Theorem: The custom multiplication of -3 and 2 equals -15 -/
theorem custom_mul_neg_three_two :
  custom_mul (-3) 2 = -15 := by
  sorry

end NUMINAMATH_CALUDE_custom_mul_neg_three_two_l111_11198


namespace NUMINAMATH_CALUDE_kendra_pens_l111_11112

/-- Proves that Kendra has 4 packs of pens given the problem conditions -/
theorem kendra_pens (kendra_packs : ℕ) : 
  let tony_packs : ℕ := 2
  let pens_per_pack : ℕ := 3
  let pens_kept_each : ℕ := 2
  let friends_given_pens : ℕ := 14
  kendra_packs * pens_per_pack - pens_kept_each + 
    (tony_packs * pens_per_pack - pens_kept_each) = friends_given_pens →
  kendra_packs = 4 := by
sorry

end NUMINAMATH_CALUDE_kendra_pens_l111_11112


namespace NUMINAMATH_CALUDE_division_simplification_l111_11178

theorem division_simplification (x y : ℝ) (h : x ≠ 0) :
  6 * x^3 * y^2 / (3 * x) = 2 * x^2 * y^2 := by
  sorry

end NUMINAMATH_CALUDE_division_simplification_l111_11178


namespace NUMINAMATH_CALUDE_skew_diagonals_properties_l111_11195

/-- A cube with edge length 1 -/
structure UnitCube where
  edge_length : ℝ
  edge_length_eq_one : edge_length = 1

/-- Skew diagonals of two adjacent faces of a unit cube -/
structure SkewDiagonals (cube : UnitCube) where
  angle : ℝ
  distance : ℝ

/-- Theorem about the properties of skew diagonals in a unit cube -/
theorem skew_diagonals_properties (cube : UnitCube) :
  ∃ (sd : SkewDiagonals cube),
    sd.angle = Real.pi / 3 ∧ sd.distance = 1 / Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_skew_diagonals_properties_l111_11195


namespace NUMINAMATH_CALUDE_tangent_segment_region_area_l111_11184

/-- The area of the region formed by all line segments of length 6 that are tangent to a circle of radius 3 at their midpoints -/
theorem tangent_segment_region_area : Real := by
  -- Define the circle radius
  let circle_radius : Real := 3
  
  -- Define the line segment length
  let segment_length : Real := 6
  
  -- Define the region area
  let region_area : Real := 9 * Real.pi
  
  -- State that the line segments are tangent to the circle at their midpoints
  -- (This is implicitly used in the proof, but we don't need to explicitly define it in Lean)
  
  -- Prove that the area of the region is equal to 9π
  sorry

#check tangent_segment_region_area

end NUMINAMATH_CALUDE_tangent_segment_region_area_l111_11184


namespace NUMINAMATH_CALUDE_power_function_property_l111_11166

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, x > 0 → f x = x ^ α

-- State the theorem
theorem power_function_property (f : ℝ → ℝ) 
  (h_power : isPowerFunction f) 
  (h_condition : f 4 = 2 * f 2) : 
  f 3 = 3 := by
sorry

end NUMINAMATH_CALUDE_power_function_property_l111_11166


namespace NUMINAMATH_CALUDE_min_y_value_l111_11191

theorem min_y_value (x y : ℝ) (h : x^2 + y^2 = 20*x + 64*y) : 
  ∃ (y_min : ℝ), y_min = 32 - 2 * Real.sqrt 281 ∧ 
  ∀ (x' y' : ℝ), x'^2 + y'^2 = 20*x' + 64*y' → y' ≥ y_min :=
by sorry

end NUMINAMATH_CALUDE_min_y_value_l111_11191
