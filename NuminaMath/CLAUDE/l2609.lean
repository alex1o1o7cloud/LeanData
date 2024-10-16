import Mathlib

namespace NUMINAMATH_CALUDE_function_is_identity_l2609_260911

def is_valid_function (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x + y) - f x - f y ∈ ({0, 1} : Set ℝ)) ∧
  (∀ x : ℝ, ⌊f x⌋ = ⌊x⌋)

theorem function_is_identity (f : ℝ → ℝ) (h : is_valid_function f) :
  ∀ x : ℝ, f x = x := by
  sorry

end NUMINAMATH_CALUDE_function_is_identity_l2609_260911


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l2609_260986

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 4) → x ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l2609_260986


namespace NUMINAMATH_CALUDE_complex_multiplication_l2609_260971

theorem complex_multiplication (i : ℂ) : i * i = -1 → (1 - i) * (1 + 2*i) = 3 + i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l2609_260971


namespace NUMINAMATH_CALUDE_carpet_coverage_percentage_l2609_260931

theorem carpet_coverage_percentage (carpet_length : ℝ) (carpet_width : ℝ) (room_area : ℝ) :
  carpet_length = 4 →
  carpet_width = 9 →
  room_area = 60 →
  (carpet_length * carpet_width) / room_area * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_carpet_coverage_percentage_l2609_260931


namespace NUMINAMATH_CALUDE_bryan_tshirt_count_l2609_260991

def total_cost : ℕ := 1500
def tshirt_cost : ℕ := 100
def pants_cost : ℕ := 250
def pants_count : ℕ := 4

theorem bryan_tshirt_count :
  (total_cost - pants_count * pants_cost) / tshirt_cost = 5 := by
  sorry

end NUMINAMATH_CALUDE_bryan_tshirt_count_l2609_260991


namespace NUMINAMATH_CALUDE_correct_sample_size_l2609_260990

/-- Given a population with total students and girls, and a sample size,
    calculate the number of girls in the sample using stratified sampling. -/
def girlsInSample (totalStudents girls sampleSize : ℕ) : ℕ :=
  (girls * sampleSize) / totalStudents

/-- Theorem stating that for the given population and sample size,
    the number of girls in the sample should be 20. -/
theorem correct_sample_size :
  girlsInSample 30000 4000 150 = 20 := by
  sorry

end NUMINAMATH_CALUDE_correct_sample_size_l2609_260990


namespace NUMINAMATH_CALUDE_goods_train_speed_l2609_260983

/-- Calculates the speed of a goods train given the conditions of the problem. -/
theorem goods_train_speed
  (man_train_speed : ℝ)
  (passing_time : ℝ)
  (goods_train_length : ℝ)
  (h1 : man_train_speed = 20)
  (h2 : passing_time = 9)
  (h3 : goods_train_length = 280) :
  ∃ (goods_train_speed : ℝ),
    goods_train_speed = 92 ∧
    (man_train_speed + goods_train_speed) * (1 / 3.6) = goods_train_length / passing_time :=
by sorry

end NUMINAMATH_CALUDE_goods_train_speed_l2609_260983


namespace NUMINAMATH_CALUDE_joan_seashells_problem_l2609_260942

theorem joan_seashells_problem (initial : ℕ) (remaining : ℕ) (sam_to_lily_ratio : ℕ) :
  initial = 70 →
  remaining = 27 →
  sam_to_lily_ratio = 2 →
  ∃ (sam lily : ℕ),
    initial = remaining + sam + lily ∧
    sam = sam_to_lily_ratio * lily ∧
    sam = 28 :=
by sorry

end NUMINAMATH_CALUDE_joan_seashells_problem_l2609_260942


namespace NUMINAMATH_CALUDE_prime_divisibility_l2609_260958

theorem prime_divisibility (p q : Nat) : 
  Nat.Prime p → Nat.Prime q → q ∣ (3^p - 2^p) → p ∣ (q - 1) := by
  sorry

end NUMINAMATH_CALUDE_prime_divisibility_l2609_260958


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l2609_260993

/-- 
Prove that an arithmetic sequence with the given properties has 15 terms.
-/
theorem arithmetic_sequence_length :
  ∀ (a l d : ℤ) (n : ℕ),
  a = -5 →  -- First term
  l = 65 →  -- Last term
  d = 5 →   -- Common difference
  l = a + (n - 1) * d →  -- Arithmetic sequence formula
  n = 15 :=  -- Number of terms
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l2609_260993


namespace NUMINAMATH_CALUDE_log_equation_solution_l2609_260903

theorem log_equation_solution (x : ℝ) :
  Real.log (x + 8) / Real.log 8 = 3/2 → x = 8 * (2 * Real.sqrt 2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l2609_260903


namespace NUMINAMATH_CALUDE_exists_polygon_with_n_triangulations_l2609_260987

/-- A polygon is a closed planar figure with straight sides. -/
structure Polygon where
  -- We don't need to define the full structure of a polygon for this statement
  mk :: (dummy : Unit)

/-- The number of triangulations of a polygon. -/
def numTriangulations (p : Polygon) : ℕ := sorry

/-- For any positive integer n, there exists a polygon with exactly n triangulations. -/
theorem exists_polygon_with_n_triangulations :
  ∀ n : ℕ, n > 0 → ∃ p : Polygon, numTriangulations p = n := by sorry

end NUMINAMATH_CALUDE_exists_polygon_with_n_triangulations_l2609_260987


namespace NUMINAMATH_CALUDE_gcd_1729_1768_l2609_260906

theorem gcd_1729_1768 : Nat.gcd 1729 1768 = 13 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1729_1768_l2609_260906


namespace NUMINAMATH_CALUDE_shed_blocks_count_l2609_260930

/-- Represents the dimensions of a rectangular structure -/
structure Dimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular structure -/
def volume (d : Dimensions) : ℝ := d.length * d.width * d.height

/-- Represents the specifications of the shed -/
structure ShedSpecs where
  outer : Dimensions
  wallThickness : ℝ

/-- Calculates the inner dimensions of the shed -/
def innerDimensions (s : ShedSpecs) : Dimensions :=
  { length := s.outer.length - 2 * s.wallThickness,
    width := s.outer.width - 2 * s.wallThickness,
    height := s.outer.height - 2 * s.wallThickness }

/-- Calculates the number of blocks used in the shed construction -/
def blocksUsed (s : ShedSpecs) : ℝ :=
  volume s.outer - volume (innerDimensions s)

/-- The main theorem stating the number of blocks used in the shed construction -/
theorem shed_blocks_count :
  let shedSpecs : ShedSpecs := {
    outer := { length := 15, width := 12, height := 7 },
    wallThickness := 1.5
  }
  blocksUsed shedSpecs = 828 := by sorry

end NUMINAMATH_CALUDE_shed_blocks_count_l2609_260930


namespace NUMINAMATH_CALUDE_unique_determination_l2609_260901

/-- Two-digit number type -/
def TwoDigitNum := {n : ℕ // n ≥ 0 ∧ n ≤ 99}

/-- The sum function as defined in the problem -/
def sum (a b c : TwoDigitNum) (X Y Z : ℕ) : ℕ :=
  a.val * X + b.val * Y + c.val * Z

/-- Function to extract a from the sum -/
def extract_a (S : ℕ) : ℕ := S % 100

/-- Function to extract b from the sum -/
def extract_b (S : ℕ) : ℕ := (S / 100) % 100

/-- Function to extract c from the sum -/
def extract_c (S : ℕ) : ℕ := S / 10000

/-- Theorem stating that a, b, and c can be uniquely determined from the sum -/
theorem unique_determination (a b c : TwoDigitNum) :
  let X : ℕ := 1
  let Y : ℕ := 100
  let Z : ℕ := 10000
  let S := sum a b c X Y Z
  (extract_a S = a.val) ∧ (extract_b S = b.val) ∧ (extract_c S = c.val) := by
  sorry

end NUMINAMATH_CALUDE_unique_determination_l2609_260901


namespace NUMINAMATH_CALUDE_adams_fair_expense_is_171_l2609_260960

/-- The amount of money Adam spent on three rides at the fair -/
def adamsFairExpense (totalTickets ferrisWheelTickets rollerCoasterTickets bumperCarsTickets ticketCost : ℕ) : ℕ :=
  (ferrisWheelTickets + rollerCoasterTickets + bumperCarsTickets) * ticketCost

/-- Theorem stating that Adam's expense on the three rides is 171 dollars -/
theorem adams_fair_expense_is_171 :
  adamsFairExpense 25 6 8 5 9 = 171 := by
  sorry

#eval adamsFairExpense 25 6 8 5 9

end NUMINAMATH_CALUDE_adams_fair_expense_is_171_l2609_260960


namespace NUMINAMATH_CALUDE_expression_evaluation_l2609_260935

theorem expression_evaluation (a : ℤ) (h : a = -1) : 
  (2*a + 1) * (2*a - 1) - 4*a*(a - 1) = -5 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2609_260935


namespace NUMINAMATH_CALUDE_pure_imaginary_product_l2609_260922

theorem pure_imaginary_product (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) 
  (h3 : ∃ (y : ℝ), (3 - 4 * Complex.I) * (a + b * Complex.I) = y * Complex.I) : 
  a / b = -4 / 3 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_product_l2609_260922


namespace NUMINAMATH_CALUDE_median_salary_is_25000_l2609_260921

/-- Represents the salary distribution of a company -/
structure SalaryDistribution where
  ceo : ℕ × ℕ
  senior_manager : ℕ × ℕ
  manager : ℕ × ℕ
  assistant_manager : ℕ × ℕ
  clerk : ℕ × ℕ

/-- The total number of employees in the company -/
def total_employees (sd : SalaryDistribution) : ℕ :=
  sd.ceo.1 + sd.senior_manager.1 + sd.manager.1 + sd.assistant_manager.1 + sd.clerk.1

/-- The median index in a list of salaries -/
def median_index (n : ℕ) : ℕ :=
  (n + 1) / 2

/-- Finds the median salary given a salary distribution -/
def median_salary (sd : SalaryDistribution) : ℕ :=
  let total := total_employees sd
  let median_idx := median_index total
  if median_idx ≤ sd.ceo.1 then sd.ceo.2
  else if median_idx ≤ sd.ceo.1 + sd.senior_manager.1 then sd.senior_manager.2
  else if median_idx ≤ sd.ceo.1 + sd.senior_manager.1 + sd.manager.1 then sd.manager.2
  else if median_idx ≤ sd.ceo.1 + sd.senior_manager.1 + sd.manager.1 + sd.assistant_manager.1 then sd.assistant_manager.2
  else sd.clerk.2

/-- The company's salary distribution -/
def company_salaries : SalaryDistribution := {
  ceo := (1, 140000),
  senior_manager := (4, 95000),
  manager := (15, 80000),
  assistant_manager := (7, 55000),
  clerk := (40, 25000)
}

theorem median_salary_is_25000 :
  median_salary company_salaries = 25000 := by
  sorry

end NUMINAMATH_CALUDE_median_salary_is_25000_l2609_260921


namespace NUMINAMATH_CALUDE_largest_divisor_of_five_consecutive_integers_l2609_260966

theorem largest_divisor_of_five_consecutive_integers :
  ∀ n : ℤ, ∃ m : ℤ, (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) = 5 * m) ∧
  ∀ k : ℤ, k > 5 → ∃ a : ℤ, ¬(∃ b : ℤ, a + (a + 1) + (a + 2) + (a + 3) + (a + 4) = k * b) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_five_consecutive_integers_l2609_260966


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l2609_260961

theorem arithmetic_mean_problem (y : ℝ) : 
  (8 + 20 + 25 + 7 + 15 + y) / 6 = 15 → y = 15 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l2609_260961


namespace NUMINAMATH_CALUDE_cars_given_to_sister_l2609_260982

/- Define the problem parameters -/
def initial_cars : ℕ := 14
def bought_cars : ℕ := 28
def birthday_cars : ℕ := 12
def cars_to_vinnie : ℕ := 3
def cars_left : ℕ := 43

/- Define the theorem -/
theorem cars_given_to_sister :
  ∃ (cars_to_sister : ℕ),
    initial_cars + bought_cars + birthday_cars
    = cars_to_sister + cars_to_vinnie + cars_left ∧
    cars_to_sister = 8 := by
  sorry

end NUMINAMATH_CALUDE_cars_given_to_sister_l2609_260982


namespace NUMINAMATH_CALUDE_max_true_statements_l2609_260940

theorem max_true_statements (c d : ℝ) : 
  let statements := [
    (1 / c > 1 / d),
    (c^2 < d^2),
    (c > d),
    (c > 0),
    (d > 0)
  ]
  ∃ (true_statements : List Bool), 
    true_statements.length ≤ 4 ∧ 
    (∀ i, i < statements.length → 
      (true_statements.get! i = true ↔ statements.get! i)) :=
by sorry

end NUMINAMATH_CALUDE_max_true_statements_l2609_260940


namespace NUMINAMATH_CALUDE_lamp_configurations_l2609_260928

/-- Represents the number of reachable configurations for n lamps -/
def reachableConfigurations (n : ℕ) : ℕ :=
  if n % 3 = 0 then 2^(n-2) else 2^n

/-- Theorem stating the number of reachable configurations for n cyclically connected lamps -/
theorem lamp_configurations (n : ℕ) (h : n > 2) :
  reachableConfigurations n = if n % 3 = 0 then 2^(n-2) else 2^n :=
by sorry

end NUMINAMATH_CALUDE_lamp_configurations_l2609_260928


namespace NUMINAMATH_CALUDE_total_kernels_needed_l2609_260968

/-- Represents a popcorn preference with its kernel-to-popcorn ratio -/
structure PopcornPreference where
  cups_wanted : ℚ
  kernels : ℚ
  cups_produced : ℚ

/-- Calculates the amount of kernels needed for a given preference -/
def kernels_needed (pref : PopcornPreference) : ℚ :=
  pref.kernels * (pref.cups_wanted / pref.cups_produced)

/-- The list of popcorn preferences for the movie night -/
def movie_night_preferences : List PopcornPreference := [
  ⟨3, 3, 6⟩,  -- Joanie
  ⟨4, 2, 4⟩,  -- Mitchell
  ⟨6, 4, 8⟩,  -- Miles and Davis
  ⟨3, 1, 3⟩   -- Cliff
]

/-- Theorem stating that the total amount of kernels needed is 7.5 tablespoons -/
theorem total_kernels_needed :
  (movie_night_preferences.map kernels_needed).sum = 15/2 := by
  sorry


end NUMINAMATH_CALUDE_total_kernels_needed_l2609_260968


namespace NUMINAMATH_CALUDE_difference_in_half_dollars_l2609_260919

/-- The number of quarters Alice has -/
def alice_quarters (p : ℚ) : ℚ := 8 * p + 2

/-- The number of quarters Bob has -/
def bob_quarters (p : ℚ) : ℚ := 3 * p + 6

/-- Conversion factor from quarters to half-dollars -/
def quarter_to_half_dollar : ℚ := 1 / 2

theorem difference_in_half_dollars (p : ℚ) :
  (alice_quarters p - bob_quarters p) * quarter_to_half_dollar = 2.5 * p - 2 := by
  sorry

end NUMINAMATH_CALUDE_difference_in_half_dollars_l2609_260919


namespace NUMINAMATH_CALUDE_triangle_abc_is_right_angled_l2609_260963

theorem triangle_abc_is_right_angled (A B C : ℝ) (h1 : A = 60) (h2 : B = 3 * C) 
  (h3 : A + B + C = 180) : B = 90 :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_is_right_angled_l2609_260963


namespace NUMINAMATH_CALUDE_blood_cell_count_l2609_260943

theorem blood_cell_count (total : ℕ) (first_sample : ℕ) (second_sample : ℕ) 
  (h1 : total = 7341)
  (h2 : first_sample = 4221)
  (h3 : total = first_sample + second_sample) : 
  second_sample = 3120 := by
  sorry

end NUMINAMATH_CALUDE_blood_cell_count_l2609_260943


namespace NUMINAMATH_CALUDE_jamie_peeled_24_l2609_260952

/-- Represents the potato peeling scenario -/
structure PotatoPeeling where
  total_potatoes : ℕ
  sylvia_rate : ℕ
  jamie_rate : ℕ
  sylvia_solo_time : ℕ

/-- Calculates the number of potatoes Jamie peeled -/
def jamie_peeled (scenario : PotatoPeeling) : ℕ :=
  let sylvia_solo := scenario.sylvia_rate * scenario.sylvia_solo_time
  let remaining := scenario.total_potatoes - sylvia_solo
  let combined_rate := scenario.sylvia_rate + scenario.jamie_rate
  let combined_time := remaining / combined_rate
  scenario.jamie_rate * combined_time

/-- Theorem stating that Jamie peeled 24 potatoes -/
theorem jamie_peeled_24 (scenario : PotatoPeeling) 
    (h1 : scenario.total_potatoes = 60)
    (h2 : scenario.sylvia_rate = 4)
    (h3 : scenario.jamie_rate = 6)
    (h4 : scenario.sylvia_solo_time = 5) : 
  jamie_peeled scenario = 24 := by
  sorry

end NUMINAMATH_CALUDE_jamie_peeled_24_l2609_260952


namespace NUMINAMATH_CALUDE_ball_count_theorem_l2609_260977

/-- Represents the number of balls of each color in a box -/
structure BallCount where
  white : ℕ
  red : ℕ
  blue : ℕ

/-- Checks if the given ball counts satisfy the ratio 4:3:2 for white:red:blue -/
def satisfiesRatio (bc : BallCount) : Prop :=
  4 * bc.red = 3 * bc.white ∧ 4 * bc.blue = 2 * bc.white

theorem ball_count_theorem (bc : BallCount) 
    (h_ratio : satisfiesRatio bc) 
    (h_white : bc.white = 12) : 
    bc.red = 9 ∧ bc.blue = 6 := by
  sorry

end NUMINAMATH_CALUDE_ball_count_theorem_l2609_260977


namespace NUMINAMATH_CALUDE_multiply_negative_two_l2609_260936

theorem multiply_negative_two : 3 * (-2) = -6 := by
  sorry

end NUMINAMATH_CALUDE_multiply_negative_two_l2609_260936


namespace NUMINAMATH_CALUDE_profit_difference_l2609_260955

def business_problem (a b c : ℕ) (b_profit : ℕ) : Prop :=
  let total_capital := a + b + c
  let a_ratio := a * b_profit * 3 / b
  let c_ratio := c * b_profit * 3 / b
  c_ratio - a_ratio = 760

theorem profit_difference :
  business_problem 8000 10000 12000 1900 :=
sorry

end NUMINAMATH_CALUDE_profit_difference_l2609_260955


namespace NUMINAMATH_CALUDE_binomial_8_3_l2609_260962

theorem binomial_8_3 : Nat.choose 8 3 = 56 := by
  sorry

end NUMINAMATH_CALUDE_binomial_8_3_l2609_260962


namespace NUMINAMATH_CALUDE_trishas_walking_distance_l2609_260959

/-- Trisha's walking distances in New York City -/
theorem trishas_walking_distance
  (total_distance : ℝ)
  (hotel_to_tshirt : ℝ)
  (h1 : total_distance = 0.8888888888888888)
  (h2 : hotel_to_tshirt = 0.6666666666666666)
  (h3 : ∃ x : ℝ, total_distance = x + x + hotel_to_tshirt) :
  ∃ x : ℝ, x = 0.1111111111111111 ∧ total_distance = x + x + hotel_to_tshirt :=
by sorry

end NUMINAMATH_CALUDE_trishas_walking_distance_l2609_260959


namespace NUMINAMATH_CALUDE_magnitude_of_8_minus_15i_l2609_260916

theorem magnitude_of_8_minus_15i : Complex.abs (8 - 15*I) = 17 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_8_minus_15i_l2609_260916


namespace NUMINAMATH_CALUDE_right_triangle_altitude_ratio_l2609_260933

/-- 
Given a right triangle ABC with legs a and b (a ≤ b) and hypotenuse c,
where the triangle formed by its altitudes is also a right triangle,
prove that the ratio of the shorter leg to the longer leg is √((√5 - 1) / 2).
-/
theorem right_triangle_altitude_ratio (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a ≤ b) :
  a^2 + b^2 = c^2 →
  a^2 + (a^2 * b^2) / (a^2 + b^2) = b^2 →
  a / b = Real.sqrt ((Real.sqrt 5 - 1) / 2) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_altitude_ratio_l2609_260933


namespace NUMINAMATH_CALUDE_common_chord_length_l2609_260950

theorem common_chord_length (r : ℝ) (h : r = 15) : 
  let chord_length := 2 * r * Real.sqrt 3
  chord_length = 26 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_common_chord_length_l2609_260950


namespace NUMINAMATH_CALUDE_mona_unique_players_l2609_260923

/-- The number of unique players Mona grouped with in a video game --/
def unique_players (total_groups : ℕ) (players_per_group : ℕ) (repeated_players : ℕ) : ℕ :=
  total_groups * players_per_group - repeated_players

/-- Theorem: Mona grouped with 33 unique players --/
theorem mona_unique_players :
  unique_players 9 4 3 = 33 := by
  sorry

end NUMINAMATH_CALUDE_mona_unique_players_l2609_260923


namespace NUMINAMATH_CALUDE_cookies_milk_ratio_l2609_260995

-- Define the constants from the problem
def cookies_for_recipe : ℕ := 18
def quarts_for_recipe : ℕ := 3
def pints_per_quart : ℕ := 2
def cookies_to_bake : ℕ := 9

-- Define the function to calculate pints needed
def pints_needed (cookies : ℕ) : ℚ :=
  (cookies : ℚ) * (quarts_for_recipe * pints_per_quart : ℚ) / (cookies_for_recipe : ℚ)

-- Theorem statement
theorem cookies_milk_ratio :
  pints_needed cookies_to_bake = 3 := by
  sorry

end NUMINAMATH_CALUDE_cookies_milk_ratio_l2609_260995


namespace NUMINAMATH_CALUDE_lasagna_pieces_needed_l2609_260913

/-- Represents the amount of lasagna each person eats relative to Manny's portion --/
structure LasagnaPortion where
  manny : ℚ
  lisa : ℚ
  raphael : ℚ
  aaron : ℚ
  kai : ℚ
  priya : ℚ

/-- Calculates the total number of lasagna pieces needed --/
def totalPieces (portions : LasagnaPortion) : ℚ :=
  portions.manny + portions.lisa + portions.kai + portions.priya

/-- The specific portions for each person based on the problem conditions --/
def givenPortions : LasagnaPortion :=
  { manny := 1
  , lisa := 2 + 1/2
  , raphael := 1/2
  , aaron := 0
  , kai := 2
  , priya := 1/3 }

theorem lasagna_pieces_needed : 
  ∃ n : ℕ, n > 0 ∧ n = ⌈totalPieces givenPortions⌉ ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_lasagna_pieces_needed_l2609_260913


namespace NUMINAMATH_CALUDE_x_plus_y_equals_negative_27_l2609_260994

theorem x_plus_y_equals_negative_27 (x y : ℤ) 
  (h1 : x + 1 = y - 8) 
  (h2 : x = 2 * y) : 
  x + y = -27 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_equals_negative_27_l2609_260994


namespace NUMINAMATH_CALUDE_swimming_problem_solution_l2609_260980

/-- Represents the amount paid by each person -/
structure Payment where
  adam : ℕ
  bill : ℕ
  chris : ℕ

/-- The problem setup -/
def swimming_problem : Prop :=
  ∃ (cost_per_session : ℕ) (final_payment : Payment),
    -- Total number of sessions
    15 * cost_per_session = final_payment.adam + final_payment.bill + final_payment.chris
    -- Adam paid 8 times
    ∧ 8 * cost_per_session = final_payment.adam + 18
    -- Bill paid 7 times
    ∧ 7 * cost_per_session = final_payment.bill + 12
    -- Chris owes £30
    ∧ final_payment.chris = 30
    -- All have paid the same amount after Chris's payment
    ∧ final_payment.adam = final_payment.bill
    ∧ final_payment.bill = final_payment.chris

theorem swimming_problem_solution : swimming_problem := by
  sorry

end NUMINAMATH_CALUDE_swimming_problem_solution_l2609_260980


namespace NUMINAMATH_CALUDE_total_questions_on_test_l2609_260979

/-- Represents a student's test results -/
structure TestResult where
  score : Int
  correct : Nat
  total : Nat

/-- Calculates the score based on correct and incorrect responses -/
def calculateScore (correct : Nat) (incorrect : Nat) : Int :=
  correct - 2 * incorrect

/-- Theorem: Given the scoring system and Student A's results, prove the total number of questions -/
theorem total_questions_on_test (result : TestResult) 
  (h1 : result.score = calculateScore result.correct (result.total - result.correct))
  (h2 : result.score = 76)
  (h3 : result.correct = 92) :
  result.total = 100 := by
  sorry

#eval calculateScore 92 8

end NUMINAMATH_CALUDE_total_questions_on_test_l2609_260979


namespace NUMINAMATH_CALUDE_hyperbola_to_ellipse_l2609_260965

/-- Given a hyperbola with equation x²/4 - y²/12 = 1, prove that the equation of the ellipse
    that has the foci of the hyperbola as its vertices and the vertices of the hyperbola as its foci
    is x²/16 + y²/12 = 1. -/
theorem hyperbola_to_ellipse (x y : ℝ) :
  (x^2 / 4 - y^2 / 12 = 1) →
  ∃ (x' y' : ℝ), (x'^2 / 16 + y'^2 / 12 = 1 ∧
    (∀ (f_x f_y : ℝ), (f_x^2 / 4 - f_y^2 / 12 = 1 ∧ f_y = 0) → 
      (x' = f_x ∧ y' = f_y)) ∧
    (∀ (v_x v_y : ℝ), (v_x^2 / 4 - v_y^2 / 12 = 1 ∧ v_y = 0 ∧ v_x^2 < 16) → 
      ∃ (c : ℝ), x'^2 - c^2 * y'^2 = (1 - c^2) * v_x^2)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_to_ellipse_l2609_260965


namespace NUMINAMATH_CALUDE_isosceles_triangle_l2609_260974

theorem isosceles_triangle (A B C : ℝ) (hsum : A + B + C = π) 
  (h : (Real.sin A + Real.sin B) * (Real.cos A + Real.cos B) = 2 * Real.sin C) :
  A = B :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_l2609_260974


namespace NUMINAMATH_CALUDE_bowling_ball_weight_is_16_l2609_260910

/-- The weight of one bowling ball in pounds -/
def bowling_ball_weight : ℝ := 16

/-- The weight of one canoe in pounds -/
def canoe_weight : ℝ := 24

/-- Theorem stating the weight of one bowling ball is 16 pounds -/
theorem bowling_ball_weight_is_16 : 
  (9 * bowling_ball_weight = 6 * canoe_weight) → 
  (5 * canoe_weight = 120) → 
  bowling_ball_weight = 16 := by
  sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_is_16_l2609_260910


namespace NUMINAMATH_CALUDE_correct_mass_units_l2609_260932

-- Define the mass units
inductive MassUnit
| Kilogram
| Gram

-- Define a structure to represent a mass measurement
structure Mass where
  value : ℝ
  unit : MassUnit

-- Define Xiaogang's weight
def xiaogang_weight : Mass := { value := 25, unit := MassUnit.Kilogram }

-- Define chalk's weight
def chalk_weight : Mass := { value := 15, unit := MassUnit.Gram }

-- Theorem to prove the correct units for Xiaogang and chalk
theorem correct_mass_units :
  xiaogang_weight.unit = MassUnit.Kilogram ∧
  chalk_weight.unit = MassUnit.Gram :=
by sorry

end NUMINAMATH_CALUDE_correct_mass_units_l2609_260932


namespace NUMINAMATH_CALUDE_rhombus_area_l2609_260934

/-- The area of a rhombus given its perimeter and one diagonal -/
theorem rhombus_area (perimeter : ℝ) (diagonal : ℝ) : 
  perimeter > 0 → diagonal > 0 → diagonal < perimeter → 
  (perimeter * diagonal) / 8 = 96 → 
  (perimeter / 4) * (((perimeter / 4)^2 - (diagonal / 2)^2).sqrt) = 96 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l2609_260934


namespace NUMINAMATH_CALUDE_tan_sum_of_roots_l2609_260954

theorem tan_sum_of_roots (α β : ℝ) : 
  (∃ x y : ℝ, x^2 - 3 * Real.sqrt 3 * x + 4 = 0 ∧ 
              y^2 - 3 * Real.sqrt 3 * y + 4 = 0 ∧ 
              x = Real.tan α ∧ 
              y = Real.tan β) → 
  Real.tan (α + β) = -Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_tan_sum_of_roots_l2609_260954


namespace NUMINAMATH_CALUDE_min_value_of_f_l2609_260972

def f (x : ℝ) := x^2 - 4*x + 7

theorem min_value_of_f :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ f x_min = 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2609_260972


namespace NUMINAMATH_CALUDE_fred_remaining_cards_l2609_260912

def initial_cards : ℕ := 40
def purchase_percentage : ℚ := 375 / 1000

theorem fred_remaining_cards :
  initial_cards - (purchase_percentage * initial_cards).floor = 25 := by
  sorry

end NUMINAMATH_CALUDE_fred_remaining_cards_l2609_260912


namespace NUMINAMATH_CALUDE_g_neg_four_l2609_260989

def g (x : ℝ) : ℝ := 5 * x + 2

theorem g_neg_four : g (-4) = -18 := by
  sorry

end NUMINAMATH_CALUDE_g_neg_four_l2609_260989


namespace NUMINAMATH_CALUDE_special_number_exists_l2609_260908

/-- A function that removes the trailing zero from a binary representation -/
def removeTrailingZero (n : ℕ) : ℕ := sorry

/-- A function that converts a natural number to its ternary representation -/
def toTernary (n : ℕ) : ℕ := sorry

/-- The theorem stating the existence of the special number -/
theorem special_number_exists : ∃ N : ℕ, 
  N % 2 = 0 ∧ 
  removeTrailingZero N = toTernary (N / 3) := by
  sorry

end NUMINAMATH_CALUDE_special_number_exists_l2609_260908


namespace NUMINAMATH_CALUDE_work_completion_days_l2609_260984

/-- The number of days B takes to complete the work alone -/
def B : ℝ := 12

/-- The number of days A and B work together -/
def together_days : ℝ := 3

/-- The number of days B works alone after A leaves -/
def B_alone_days : ℝ := 3

/-- The number of days A takes to complete the work alone -/
def A : ℝ := 6

theorem work_completion_days : 
  together_days * (1 / A + 1 / B) + B_alone_days * (1 / B) = 1 :=
sorry

end NUMINAMATH_CALUDE_work_completion_days_l2609_260984


namespace NUMINAMATH_CALUDE_percentage_difference_l2609_260949

theorem percentage_difference (A B y : ℝ) : 
  A > B ∧ B > 0 → B = A * (1 - y / 100) → y = 100 * (A - B) / A :=
by sorry

end NUMINAMATH_CALUDE_percentage_difference_l2609_260949


namespace NUMINAMATH_CALUDE_polar_to_cartesian_l2609_260927

theorem polar_to_cartesian (x y : ℝ) : 
  (∃ (ρ θ : ℝ), ρ = 3 ∧ θ = π/6 ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) → 
  x = 3 * Real.sqrt 3 / 2 ∧ y = 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_l2609_260927


namespace NUMINAMATH_CALUDE_problem_solution_l2609_260920

theorem problem_solution (x : ℚ) : 
  (1 / 3 : ℚ) - (1 / 4 : ℚ) + (1 / 6 : ℚ) = 4 / x → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2609_260920


namespace NUMINAMATH_CALUDE_city_rentals_rate_proof_l2609_260973

/-- The daily rate for Safety Rent-a-Car in dollars -/
def safety_daily_rate : ℝ := 21.95

/-- The per-mile rate for Safety Rent-a-Car in dollars -/
def safety_mile_rate : ℝ := 0.19

/-- The daily rate for City Rentals in dollars -/
def city_daily_rate : ℝ := 18.95

/-- The number of miles driven -/
def miles_driven : ℝ := 150

/-- The per-mile rate for City Rentals in dollars -/
def city_mile_rate : ℝ := 0.21

theorem city_rentals_rate_proof :
  safety_daily_rate + safety_mile_rate * miles_driven =
  city_daily_rate + city_mile_rate * miles_driven :=
by sorry

end NUMINAMATH_CALUDE_city_rentals_rate_proof_l2609_260973


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l2609_260985

theorem tangent_line_to_circle (x y : ℝ) : 
  (∃ k : ℝ, (y = k * (x - Real.sqrt 2)) ∧ 
   ((k * x - y - k * Real.sqrt 2) ^ 2) / (k ^ 2 + 1) = 1) →
  (x - y - Real.sqrt 2 = 0 ∨ x + y - Real.sqrt 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l2609_260985


namespace NUMINAMATH_CALUDE_swap_positions_l2609_260969

/-- Represents the color of a checker -/
inductive Color
| Black
| White

/-- Represents a move in the game -/
structure Move where
  color : Color
  count : Nat

/-- Represents the state of the game -/
structure GameState where
  n : Nat
  blackPositions : List Nat
  whitePositions : List Nat

/-- Checks if a move is valid according to the game rules -/
def isValidMove (state : GameState) (move : Move) : Prop :=
  match move.color with
  | Color.Black => move.count ≤ state.n ∧ move.count > 0
  | Color.White => move.count ≤ state.n ∧ move.count > 0

/-- Applies a move to the current game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  sorry

/-- Generates the sequence of moves for the game -/
def generateMoves (n : Nat) : List Move :=
  sorry

/-- Checks if the final state has swapped positions -/
def isSwappedState (initialState : GameState) (finalState : GameState) : Prop :=
  sorry

/-- Theorem stating that the generated moves will swap the positions -/
theorem swap_positions (n : Nat) :
  let initialState : GameState := { n := n, blackPositions := List.range n, whitePositions := List.range n |>.map (λ x => 2*n - x) }
  let moves := generateMoves n
  let finalState := moves.foldl applyMove initialState
  (∀ move ∈ moves, isValidMove initialState move) ∧
  isSwappedState initialState finalState :=
sorry

end NUMINAMATH_CALUDE_swap_positions_l2609_260969


namespace NUMINAMATH_CALUDE_basketball_shooting_frequency_l2609_260937

/-- Given a basketball player who made 90 total shots with 63 successful shots,
    prove that the shooting frequency is equal to 0.7. -/
theorem basketball_shooting_frequency :
  let total_shots : ℕ := 90
  let successful_shots : ℕ := 63
  let shooting_frequency := (successful_shots : ℚ) / total_shots
  shooting_frequency = 0.7 := by sorry

end NUMINAMATH_CALUDE_basketball_shooting_frequency_l2609_260937


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2609_260924

theorem complex_equation_solution (a b : ℝ) (i : ℂ) 
  (h1 : i * i = -1) 
  (h2 : (a - 2*i) * i = b - i) : 
  a + b*i = -1 + 2*i := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2609_260924


namespace NUMINAMATH_CALUDE_cheryl_skittles_l2609_260978

theorem cheryl_skittles (initial : ℕ) (given : ℕ) (final : ℕ) : 
  given = 89 → final = 97 → initial + given = final → initial = 8 := by
sorry

end NUMINAMATH_CALUDE_cheryl_skittles_l2609_260978


namespace NUMINAMATH_CALUDE_power_mod_equivalence_l2609_260957

theorem power_mod_equivalence (x : ℤ) (h : x^77 % 7 = 6) : x^5 % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_power_mod_equivalence_l2609_260957


namespace NUMINAMATH_CALUDE_sequence_increasing_l2609_260988

theorem sequence_increasing (n : ℕ) (h : n ≥ 1) : 
  let a : ℕ → ℚ := fun k => (2 * k : ℚ) / (3 * k + 1)
  a (n + 1) > a n := by
sorry

end NUMINAMATH_CALUDE_sequence_increasing_l2609_260988


namespace NUMINAMATH_CALUDE_cosine_function_properties_l2609_260944

/-- Given a cosine function with specific properties, prove the value of ω and cos(α+β) -/
theorem cosine_function_properties (f : ℝ → ℝ) (ω α β : ℝ) :
  (∀ x, f x = 2 * Real.cos (ω * x + π / 6)) →
  ω > 0 →
  (∀ x, f (x + 10 * π) = f x) →
  (∀ y, y > 0 → y < 10 * π → ∀ x, f (x + y) ≠ f x) →
  α ∈ Set.Icc 0 (π / 2) →
  β ∈ Set.Icc 0 (π / 2) →
  f (5 * α + 5 * π / 3) = -6 / 5 →
  f (5 * β - 5 * π / 6) = 16 / 17 →
  ω = 1 / 5 ∧ Real.cos (α + β) = -13 / 85 := by
  sorry


end NUMINAMATH_CALUDE_cosine_function_properties_l2609_260944


namespace NUMINAMATH_CALUDE_non_square_sequence_250th_term_l2609_260953

/-- The sequence of positive integers omitting perfect squares -/
def non_square_sequence : ℕ → ℕ := sorry

/-- Predicate to check if a number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

/-- The 250th term of the non-square sequence -/
def term_250 : ℕ := non_square_sequence 250

theorem non_square_sequence_250th_term :
  term_250 = 265 := by sorry

end NUMINAMATH_CALUDE_non_square_sequence_250th_term_l2609_260953


namespace NUMINAMATH_CALUDE_ticket_sales_total_l2609_260997

/-- Calculates the total amount of money collected from ticket sales -/
def totalAmountCollected (adultPrice studentPrice : ℚ) (totalTickets studentTickets : ℕ) : ℚ :=
  let adultTickets := totalTickets - studentTickets
  adultPrice * adultTickets + studentPrice * studentTickets

/-- Theorem stating that the total amount collected is $222.50 given the problem conditions -/
theorem ticket_sales_total : 
  totalAmountCollected 4 (5/2) 59 9 = 445/2 := by
  sorry

#eval totalAmountCollected 4 (5/2) 59 9

end NUMINAMATH_CALUDE_ticket_sales_total_l2609_260997


namespace NUMINAMATH_CALUDE_regular_polygon_with_45_degree_exterior_angles_l2609_260975

theorem regular_polygon_with_45_degree_exterior_angles (n : ℕ) 
  (h1 : n > 2) 
  (h2 : (360 : ℝ) / n = 45) : n = 8 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_with_45_degree_exterior_angles_l2609_260975


namespace NUMINAMATH_CALUDE_solution_composition_l2609_260992

/-- Represents the initial percentage of liquid X in the solution -/
def initial_percentage : ℝ := 30

/-- The initial weight of the solution in kg -/
def initial_weight : ℝ := 10

/-- The weight of water that evaporates in kg -/
def evaporated_water : ℝ := 2

/-- The weight of the original solution added back in kg -/
def added_solution : ℝ := 2

/-- The final percentage of liquid X in the new solution -/
def final_percentage : ℝ := 36

theorem solution_composition :
  let remaining_weight := initial_weight - evaporated_water
  let new_total_weight := remaining_weight + added_solution
  let initial_liquid_x := initial_percentage / 100 * initial_weight
  let added_liquid_x := initial_percentage / 100 * added_solution
  let total_liquid_x := initial_liquid_x + added_liquid_x
  total_liquid_x / new_total_weight * 100 = final_percentage :=
by sorry

end NUMINAMATH_CALUDE_solution_composition_l2609_260992


namespace NUMINAMATH_CALUDE_inscribed_rectangle_area_l2609_260948

/-- The area of a rectangle inscribed in a trapezoid -/
theorem inscribed_rectangle_area (a b h x : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : b < a) (hh : 0 < h) (hx : 0 < x) (hxh : x < h) :
  let area := x * (a - b) * (h - x) / h
  area = x * (a - b) * (h - x) / h :=
by sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_area_l2609_260948


namespace NUMINAMATH_CALUDE_yellow_tint_percentage_l2609_260902

/-- Calculates the percentage of yellow tint in a new mixture after adding more yellow tint --/
theorem yellow_tint_percentage 
  (original_volume : ℝ) 
  (original_yellow_percentage : ℝ) 
  (added_yellow : ℝ) : 
  original_volume = 50 → 
  original_yellow_percentage = 0.5 → 
  added_yellow = 10 → 
  (((original_volume * original_yellow_percentage + added_yellow) / (original_volume + added_yellow)) * 100 : ℝ) = 58 := by
  sorry

end NUMINAMATH_CALUDE_yellow_tint_percentage_l2609_260902


namespace NUMINAMATH_CALUDE_pants_cost_theorem_l2609_260999

/-- Represents the cost and pricing strategy for a pair of pants -/
structure PantsPricing where
  cost : ℝ
  profit_percentage : ℝ
  discount_percentage : ℝ
  final_price : ℝ

/-- Calculates the selling price before discount -/
def selling_price (p : PantsPricing) : ℝ :=
  p.cost * (1 + p.profit_percentage)

/-- Calculates the final selling price after discount -/
def discounted_price (p : PantsPricing) : ℝ :=
  selling_price p * (1 - p.discount_percentage)

/-- Theorem stating the relationship between the cost and final price -/
theorem pants_cost_theorem (p : PantsPricing) 
  (h1 : p.profit_percentage = 0.30)
  (h2 : p.discount_percentage = 0.20)
  (h3 : p.final_price = 130)
  : p.cost = 125 := by
  sorry

end NUMINAMATH_CALUDE_pants_cost_theorem_l2609_260999


namespace NUMINAMATH_CALUDE_smallest_from_two_and_four_l2609_260939

/-- Given two single-digit numbers, returns the smallest two-digit number that can be formed using both digits. -/
def smallest_two_digit (a b : Nat) : Nat :=
  if a ≤ b then 10 * a + b else 10 * b + a

/-- Proves that the smallest two-digit number formed from 2 and 4 is 24. -/
theorem smallest_from_two_and_four :
  smallest_two_digit 2 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_smallest_from_two_and_four_l2609_260939


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2609_260905

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h1 : ∀ n : ℕ, a (n + 1) = a n * q) 
  (h2 : a 3 = 4) 
  (h3 : a 6 = 1/2) : 
  q = 1/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2609_260905


namespace NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l2609_260998

theorem least_positive_integer_with_remainders : ∃ b : ℕ+, 
  (b : ℤ) % 4 = 1 ∧ 
  (b : ℤ) % 5 = 2 ∧ 
  (b : ℤ) % 6 = 3 ∧ 
  (∀ c : ℕ+, c < b → 
    (c : ℤ) % 4 ≠ 1 ∨ 
    (c : ℤ) % 5 ≠ 2 ∨ 
    (c : ℤ) % 6 ≠ 3) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l2609_260998


namespace NUMINAMATH_CALUDE_find_numbers_l2609_260907

theorem find_numbers (A B C : ℕ) 
  (h1 : Nat.gcd A B = 2)
  (h2 : Nat.lcm A B = 60)
  (h3 : Nat.gcd A C = 3)
  (h4 : Nat.lcm A C = 42) :
  A = 6 ∧ B = 20 ∧ C = 21 := by
  sorry

end NUMINAMATH_CALUDE_find_numbers_l2609_260907


namespace NUMINAMATH_CALUDE_ram_selection_probability_l2609_260947

/-- Given two brothers Ram and Ravi, where the probability of Ravi's selection is 1/5
    and the probability of both being selected is 0.11428571428571428,
    prove that the probability of Ram's selection is 0.5714285714285714 -/
theorem ram_selection_probability
  (p_ravi : ℝ)
  (p_both : ℝ)
  (h1 : p_ravi = 1 / 5)
  (h2 : p_both = 0.11428571428571428) :
  p_both / p_ravi = 0.5714285714285714 := by
  sorry

end NUMINAMATH_CALUDE_ram_selection_probability_l2609_260947


namespace NUMINAMATH_CALUDE_odd_implies_symmetric_abs_not_vice_versa_l2609_260915

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The graph of |f(x)| is symmetric about the y-axis if |f(-x)| = |f(x)| for all x ∈ ℝ -/
def IsSymmetricAboutYAxis (f : ℝ → ℝ) : Prop :=
  ∀ x, |f (-x)| = |f x|

/-- If f is odd, then |f(x)| is symmetric about the y-axis, but not vice versa -/
theorem odd_implies_symmetric_abs_not_vice_versa :
  (∃ f : ℝ → ℝ, IsOdd f → IsSymmetricAboutYAxis f) ∧
  (∃ g : ℝ → ℝ, IsSymmetricAboutYAxis g ∧ ¬IsOdd g) := by
  sorry

end NUMINAMATH_CALUDE_odd_implies_symmetric_abs_not_vice_versa_l2609_260915


namespace NUMINAMATH_CALUDE_series_sum_equals_five_l2609_260967

theorem series_sum_equals_five (k : ℝ) (h1 : k > 1) 
  (h2 : ∑' n, (6 * n + 1) / k^n = 5) : k = 1.2 + 0.2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_equals_five_l2609_260967


namespace NUMINAMATH_CALUDE_p_half_q_age_years_ago_l2609_260956

/-- The number of years ago when p was half of q in age -/
def years_ago : ℕ := 12

/-- The present age of p -/
def p_age : ℕ := 18

/-- The present age of q -/
def q_age : ℕ := 24

/-- Theorem: Given the conditions, prove that p was half of q in age 12 years ago -/
theorem p_half_q_age_years_ago :
  (p_age : ℚ) / (q_age : ℚ) = 3 / 4 ∧
  p_age + q_age = 42 ∧
  (p_age - years_ago : ℚ) = (q_age - years_ago : ℚ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_p_half_q_age_years_ago_l2609_260956


namespace NUMINAMATH_CALUDE_hexagon_to_square_area_l2609_260938

/-- Given a regular hexagon with side length 4, prove that a square formed from its perimeter has an area of 36 -/
theorem hexagon_to_square_area :
  ∀ (hexagon_side : ℝ) (square_side : ℝ) (square_area : ℝ),
    hexagon_side = 4 →
    square_side = 6 * hexagon_side / 4 →
    square_area = square_side * square_side →
    square_area = 36 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_to_square_area_l2609_260938


namespace NUMINAMATH_CALUDE_fixed_point_on_graph_l2609_260900

theorem fixed_point_on_graph :
  ∀ (k : ℝ), 112 = 7 * (4 : ℝ)^2 + k * 4 - 4 * k := by sorry

end NUMINAMATH_CALUDE_fixed_point_on_graph_l2609_260900


namespace NUMINAMATH_CALUDE_beatrice_book_cost_l2609_260918

/-- Calculates the total cost of books given the pricing structure and number of books purchased. -/
def totalCost (regularPrice : ℕ) (discountAmount : ℕ) (discountThreshold : ℕ) (totalBooks : ℕ) : ℕ :=
  let regularCost := min totalBooks discountThreshold * regularPrice
  let discountedBooks := max (totalBooks - discountThreshold) 0
  let discountedCost := discountedBooks * (regularPrice - discountAmount)
  regularCost + discountedCost

/-- Theorem stating that Beatrice paid $370 for 20 books under the given pricing structure. -/
theorem beatrice_book_cost : totalCost 20 2 5 20 = 370 := by
  sorry

end NUMINAMATH_CALUDE_beatrice_book_cost_l2609_260918


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l2609_260964

-- Define the expression as a function of x
def f (x : ℝ) : ℝ := (x + 1) * (x - 1) + x * (2 - x) + (x - 1)^2

-- Theorem stating the simplification and evaluation
theorem simplify_and_evaluate :
  (∀ x : ℝ, f x = x^2) ∧ (f 100 = 10000) := by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l2609_260964


namespace NUMINAMATH_CALUDE_smallest_candy_number_l2609_260925

theorem smallest_candy_number : ∃ (n : ℕ), 
  100 ≤ n ∧ n < 1000 ∧ 
  (n + 7) % 9 = 0 ∧ 
  (n - 9) % 7 = 0 ∧
  ∀ m, 100 ≤ m ∧ m < n → ¬((m + 7) % 9 = 0 ∧ (m - 9) % 7 = 0) :=
by
  use 110
  sorry

end NUMINAMATH_CALUDE_smallest_candy_number_l2609_260925


namespace NUMINAMATH_CALUDE_game_probability_l2609_260926

/-- Represents a player in the game -/
inductive Player : Type
| Alice : Player
| Bob : Player
| Cindy : Player
| Dave : Player

/-- The game state is represented by a function from Player to ℕ (natural numbers) -/
def GameState : Type := Player → ℕ

/-- The initial state of the game where each player has 2 units -/
def initialState : GameState :=
  fun _ => 2

/-- A single round of the game -/
def gameRound (state : GameState) : GameState :=
  sorry -- Implementation details omitted

/-- The probability of a specific outcome after one round -/
def roundProbability (initialState finalState : GameState) : ℚ :=
  sorry -- Implementation details omitted

/-- The probability of all players having 2 units after 5 rounds -/
def finalProbability : ℚ :=
  sorry -- Implementation details omitted

/-- The main theorem stating the probability of all players having 2 units after 5 rounds -/
theorem game_probability : finalProbability = 4 / 81^5 := by
  sorry

end NUMINAMATH_CALUDE_game_probability_l2609_260926


namespace NUMINAMATH_CALUDE_range_of_a_l2609_260946

theorem range_of_a (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 + 2*x + a ≤ 0) ∧ 
  (∀ x > 0, x + 1/x > a) → 
  1 < a ∧ a < 2 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l2609_260946


namespace NUMINAMATH_CALUDE_no_zeros_in_interval_l2609_260941

open Real

theorem no_zeros_in_interval (ω : ℝ) (h_ω_pos : ω > 0) :
  (∀ x ∈ Set.Ioo (π / 2) (3 * π / 2), cos (ω * x - 5 * π / 6) ≠ 0) →
  ω ∈ Set.Ioc 0 (2 / 9) ∪ Set.Icc (2 / 3) (8 / 9) :=
by sorry

end NUMINAMATH_CALUDE_no_zeros_in_interval_l2609_260941


namespace NUMINAMATH_CALUDE_function_identity_l2609_260951

theorem function_identity (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x^2 + f y) = y + (f x)^2) : 
  ∀ x : ℝ, f x = x := by
sorry

end NUMINAMATH_CALUDE_function_identity_l2609_260951


namespace NUMINAMATH_CALUDE_three_digit_numbers_with_repeated_digits_l2609_260981

theorem three_digit_numbers_with_repeated_digits : 
  let total_three_digit_numbers := 999 - 100 + 1
  let distinct_digit_numbers := 9 * 9 * 8
  total_three_digit_numbers - distinct_digit_numbers = 252 := by
sorry

end NUMINAMATH_CALUDE_three_digit_numbers_with_repeated_digits_l2609_260981


namespace NUMINAMATH_CALUDE_prob_at_least_one_odd_is_nine_tenths_l2609_260914

def numbers : Finset ℕ := {1, 2, 3, 4, 5}

def is_odd (n : ℕ) : Bool := n % 2 = 1

def prob_at_least_one_odd : ℚ :=
  1 - (Finset.filter (λ n => ¬(is_odd n)) numbers).card.choose 2 / numbers.card.choose 2

theorem prob_at_least_one_odd_is_nine_tenths :
  prob_at_least_one_odd = 9/10 := by sorry

end NUMINAMATH_CALUDE_prob_at_least_one_odd_is_nine_tenths_l2609_260914


namespace NUMINAMATH_CALUDE_hyperbola_focus_m_value_l2609_260945

/-- A hyperbola with equation (y^2/m) - (x^2/9) = 1 -/
structure Hyperbola (m : ℝ) where
  equation : ∀ (x y : ℝ), y^2/m - x^2/9 = 1

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of a focus of a hyperbola -/
def is_focus (p : Point) (h : Hyperbola m) : Prop :=
  p.x = 0 ∧ p.y^2 = m + 9

theorem hyperbola_focus_m_value (m : ℝ) (h : Hyperbola m) :
  is_focus (Point.mk 0 5) h → m = 16 := by
  sorry


end NUMINAMATH_CALUDE_hyperbola_focus_m_value_l2609_260945


namespace NUMINAMATH_CALUDE_three_common_points_l2609_260917

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Function to check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Function to check if a point satisfies the first equation -/
def satisfiesEq1 (p : Point) : Prop :=
  (2 * p.x - 3 * p.y + 6) * (5 * p.x + 2 * p.y - 10) = 0

/-- Function to check if a point satisfies the second equation -/
def satisfiesEq2 (p : Point) : Prop :=
  (p.x - 2 * p.y + 1) * (3 * p.x - 4 * p.y + 8) = 0

/-- The main theorem stating that there are exactly 3 common points -/
theorem three_common_points :
  ∃ (p1 p2 p3 : Point),
    satisfiesEq1 p1 ∧ satisfiesEq2 p1 ∧
    satisfiesEq1 p2 ∧ satisfiesEq2 p2 ∧
    satisfiesEq1 p3 ∧ satisfiesEq2 p3 ∧
    p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 ∧
    ∀ (p : Point), satisfiesEq1 p ∧ satisfiesEq2 p → p = p1 ∨ p = p2 ∨ p = p3 :=
sorry


end NUMINAMATH_CALUDE_three_common_points_l2609_260917


namespace NUMINAMATH_CALUDE_range_of_a_l2609_260970

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 + (a - 1)*x + 1 > 0) → -1 < a ∧ a < 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2609_260970


namespace NUMINAMATH_CALUDE_largest_n_binomial_equality_l2609_260976

theorem largest_n_binomial_equality : 
  (∃ n : ℕ, (n : ℤ) = 7 ∧ 
    Nat.choose 10 3 + Nat.choose 10 4 = Nat.choose 11 n ∧
    ∀ m : ℕ, m > n → Nat.choose 10 3 + Nat.choose 10 4 ≠ Nat.choose 11 m) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_binomial_equality_l2609_260976


namespace NUMINAMATH_CALUDE_starters_combination_l2609_260996

-- Define the total number of players
def total_players : ℕ := 18

-- Define the number of quadruplets
def num_quadruplets : ℕ := 4

-- Define the number of starters to choose
def num_starters : ℕ := 7

-- Define the number of quadruplets that must be in the starting lineup
def required_quadruplets : ℕ := 3

-- Define the function to calculate the number of ways to choose the starters
def choose_starters (total : ℕ) (quad : ℕ) (starters : ℕ) (req_quad : ℕ) : ℕ :=
  (Nat.choose quad req_quad) * (Nat.choose (total - quad) (starters - req_quad))

-- Theorem statement
theorem starters_combination : 
  choose_starters total_players num_quadruplets num_starters required_quadruplets = 4004 := by
  sorry

end NUMINAMATH_CALUDE_starters_combination_l2609_260996


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l2609_260904

theorem pure_imaginary_complex_number (x : ℝ) : 
  (((x^2 - 1) : ℂ) + (x^2 + 3*x + 2)*I = (0 : ℂ) + y*I ∧ y ≠ 0) → x = 1 :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l2609_260904


namespace NUMINAMATH_CALUDE_greatest_integer_inequality_l2609_260929

theorem greatest_integer_inequality : ∀ x : ℤ, (5 : ℚ) / 8 > (x : ℚ) / 17 ↔ x ≤ 10 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_inequality_l2609_260929


namespace NUMINAMATH_CALUDE_lewis_earnings_l2609_260909

theorem lewis_earnings (weeks : ℕ) (weekly_rent : ℚ) (total_after_rent : ℚ) 
  (h1 : weeks = 233)
  (h2 : weekly_rent = 49)
  (h3 : total_after_rent = 93899) :
  (total_after_rent + weeks * weekly_rent) / weeks = 451.99 := by
sorry

end NUMINAMATH_CALUDE_lewis_earnings_l2609_260909
