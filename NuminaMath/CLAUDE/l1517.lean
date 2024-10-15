import Mathlib

namespace NUMINAMATH_CALUDE_counterexample_prime_plus_two_l1517_151786

theorem counterexample_prime_plus_two :
  ∃ n : ℕ, Nat.Prime n ∧ ¬(Nat.Prime (n + 2)) :=
sorry

end NUMINAMATH_CALUDE_counterexample_prime_plus_two_l1517_151786


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1517_151783

theorem complex_modulus_problem (x y : ℝ) (z : ℂ) :
  z = x + y * Complex.I →
  x / (1 - Complex.I) = 1 + y * Complex.I →
  Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1517_151783


namespace NUMINAMATH_CALUDE_sum_of_first_20_a_l1517_151778

def odd_number (n : ℕ) : ℕ := 2 * n - 1

def a (n : ℕ) : ℕ := 
  Finset.sum (Finset.range n) (λ k => odd_number (n * (n - 1) + 1 + k))

theorem sum_of_first_20_a : Finset.sum (Finset.range 20) (λ i => a (i + 1)) = 44100 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_first_20_a_l1517_151778


namespace NUMINAMATH_CALUDE_vessel_base_length_l1517_151729

/-- Given a cube immersed in a rectangular vessel, this theorem proves the length of the vessel's base. -/
theorem vessel_base_length 
  (cube_edge : ℝ) 
  (vessel_width : ℝ) 
  (water_rise : ℝ) 
  (h1 : cube_edge = 12)
  (h2 : vessel_width = 15)
  (h3 : water_rise = 5.76)
  : ∃ (vessel_length : ℝ), vessel_length = 20 :=
by
  sorry


end NUMINAMATH_CALUDE_vessel_base_length_l1517_151729


namespace NUMINAMATH_CALUDE_subway_length_l1517_151797

/-- Calculates the length of a subway given its speed, distance between stations, and time to pass a station. -/
theorem subway_length
  (speed : ℝ)                  -- Speed of the subway in km/min
  (station_distance : ℝ)       -- Distance between stations in km
  (passing_time : ℝ)           -- Time to pass the station in minutes
  (h1 : speed = 1.6)           -- Given speed
  (h2 : station_distance = 4.85) -- Given distance between stations
  (h3 : passing_time = 3.25)   -- Given time to pass the station
  : (speed * passing_time - station_distance) * 1000 = 350 :=
by sorry

end NUMINAMATH_CALUDE_subway_length_l1517_151797


namespace NUMINAMATH_CALUDE_sugar_cube_theorem_l1517_151753

/-- Represents a box of sugar cubes -/
structure SugarBox where
  height : Nat
  width : Nat
  depth : Nat

/-- Calculates the number of remaining cubes in a sugar box after eating layers -/
def remaining_cubes (box : SugarBox) : Set Nat :=
  if box.width * box.depth = 77 ∧ box.height * box.depth = 55 then
    if box.depth = 1 then {0}
    else if box.depth = 11 then {300}
    else ∅
  else ∅

/-- Theorem stating that the number of remaining cubes is either 300 or 0 -/
theorem sugar_cube_theorem (box : SugarBox) :
  remaining_cubes box ⊆ {0, 300} :=
by sorry

end NUMINAMATH_CALUDE_sugar_cube_theorem_l1517_151753


namespace NUMINAMATH_CALUDE_M_intersect_N_eq_M_l1517_151737

/-- Set M defined as {x | x^2 - x < 0} -/
def M : Set ℝ := {x | x^2 - x < 0}

/-- Set N defined as {x | |x| < 2} -/
def N : Set ℝ := {x | |x| < 2}

/-- Theorem stating that the intersection of M and N equals M -/
theorem M_intersect_N_eq_M : M ∩ N = M := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_eq_M_l1517_151737


namespace NUMINAMATH_CALUDE_kevins_initial_cards_l1517_151715

theorem kevins_initial_cards (found_cards end_cards : ℕ) 
  (h1 : found_cards = 47) 
  (h2 : end_cards = 54) : 
  end_cards - found_cards = 7 := by
  sorry

end NUMINAMATH_CALUDE_kevins_initial_cards_l1517_151715


namespace NUMINAMATH_CALUDE_birth_death_rate_decisive_l1517_151727

/-- Represents the various characteristics of a population -/
inductive PopulationCharacteristic
  | Density
  | AgeComposition
  | SexRatio
  | BirthRate
  | DeathRate
  | ImmigrationRate
  | EmigrationRate

/-- Represents the impact of a characteristic on population size change -/
inductive Impact
  | Decisive
  | Indirect
  | Basic

/-- Function that maps a population characteristic to its impact on population size change -/
def characteristicImpact : PopulationCharacteristic → Impact
  | PopulationCharacteristic.Density => Impact.Basic
  | PopulationCharacteristic.AgeComposition => Impact.Indirect
  | PopulationCharacteristic.SexRatio => Impact.Indirect
  | PopulationCharacteristic.BirthRate => Impact.Decisive
  | PopulationCharacteristic.DeathRate => Impact.Decisive
  | PopulationCharacteristic.ImmigrationRate => Impact.Decisive
  | PopulationCharacteristic.EmigrationRate => Impact.Decisive

theorem birth_death_rate_decisive :
  ∀ c : PopulationCharacteristic,
    characteristicImpact c = Impact.Decisive →
    c = PopulationCharacteristic.BirthRate ∨ c = PopulationCharacteristic.DeathRate :=
by sorry

end NUMINAMATH_CALUDE_birth_death_rate_decisive_l1517_151727


namespace NUMINAMATH_CALUDE_valid_numbers_l1517_151747

def is_valid_number (n : ℕ) : Prop :=
  ∃ x y : ℕ, 
    x ≤ 9 ∧ y ≤ 9 ∧
    n = 3000000 + x * 10000 + y * 100 + 3 ∧
    n % 13 = 0

theorem valid_numbers : 
  {n : ℕ | is_valid_number n} = 
  {3020303, 3050203, 3080103, 3090503, 3060603, 3030703, 3000803} := by
sorry

end NUMINAMATH_CALUDE_valid_numbers_l1517_151747


namespace NUMINAMATH_CALUDE_catch_up_distance_l1517_151789

/-- Proves that B catches up with A 200 km from the start given the specified conditions -/
theorem catch_up_distance (a_speed b_speed : ℝ) (time_diff : ℝ) (catch_up_dist : ℝ) : 
  a_speed = 10 →
  b_speed = 20 →
  time_diff = 10 →
  catch_up_dist = 200 →
  catch_up_dist = b_speed * (time_diff + catch_up_dist / b_speed) :=
by sorry

end NUMINAMATH_CALUDE_catch_up_distance_l1517_151789


namespace NUMINAMATH_CALUDE_shortest_altitude_right_triangle_l1517_151771

theorem shortest_altitude_right_triangle :
  ∀ (a b c h : ℝ),
  a = 9 ∧ b = 12 ∧ c = 15 →
  a^2 + b^2 = c^2 →
  h = (2 * (a * b) / 2) / c →
  h ≤ a ∧ h ≤ b →
  h = 7.2 :=
by sorry

end NUMINAMATH_CALUDE_shortest_altitude_right_triangle_l1517_151771


namespace NUMINAMATH_CALUDE_min_swaps_upper_bound_min_swaps_lower_bound_min_swaps_exact_l1517_151736

/-- A swap operation on a matrix -/
def swap (M : Matrix (Fin n) (Fin n) ℕ) (i j k l : Fin n) : Matrix (Fin n) (Fin n) ℕ :=
  sorry

/-- Predicate to check if a matrix contains all numbers from 1 to n² -/
def valid_matrix (M : Matrix (Fin n) (Fin n) ℕ) : Prop :=
  sorry

/-- The number of swaps needed to transform one matrix into another -/
def swaps_needed (A B : Matrix (Fin n) (Fin n) ℕ) : ℕ :=
  sorry

theorem min_swaps_upper_bound (n : ℕ) (h : n ≥ 2) :
  ∃ m : ℕ, ∀ (A B : Matrix (Fin n) (Fin n) ℕ),
    valid_matrix A → valid_matrix B →
    swaps_needed A B ≤ m ∧
    m = 2 * n * (n - 1) :=
  sorry

theorem min_swaps_lower_bound (n : ℕ) (h : n ≥ 2) :
  ∃ (A B : Matrix (Fin n) (Fin n) ℕ),
    valid_matrix A ∧ valid_matrix B ∧
    swaps_needed A B = 2 * n * (n - 1) :=
  sorry

theorem min_swaps_exact (n : ℕ) (h : n ≥ 2) :
  ∃! m : ℕ, (∀ (A B : Matrix (Fin n) (Fin n) ℕ),
    valid_matrix A → valid_matrix B →
    swaps_needed A B ≤ m) ∧
    (∃ (A B : Matrix (Fin n) (Fin n) ℕ),
      valid_matrix A ∧ valid_matrix B ∧
      swaps_needed A B = m) ∧
    m = 2 * n * (n - 1) :=
  sorry

end NUMINAMATH_CALUDE_min_swaps_upper_bound_min_swaps_lower_bound_min_swaps_exact_l1517_151736


namespace NUMINAMATH_CALUDE_dinnerCostTheorem_l1517_151741

/-- Represents the cost breakdown of a dinner -/
structure DinnerCost where
  preTax : ℝ
  taxRate : ℝ
  tipRate : ℝ
  total : ℝ

/-- The combined pre-tax cost of two dinners -/
def combinedPreTaxCost (d1 d2 : DinnerCost) : ℝ :=
  d1.preTax + d2.preTax

/-- Calculates the total cost of a dinner including tax and tip -/
def calculateTotal (d : DinnerCost) : ℝ :=
  d.preTax * (1 + d.taxRate + d.tipRate)

theorem dinnerCostTheorem (johnDinner sarahDinner : DinnerCost) :
  johnDinner.taxRate = 0.12 →
  johnDinner.tipRate = 0.16 →
  sarahDinner.taxRate = 0.09 →
  sarahDinner.tipRate = 0.10 →
  johnDinner.total = 35.20 →
  sarahDinner.total = 22.00 →
  calculateTotal johnDinner = johnDinner.total →
  calculateTotal sarahDinner = sarahDinner.total →
  combinedPreTaxCost johnDinner sarahDinner = 46 := by
  sorry

#eval 46  -- This line is added to ensure the statement can be built successfully

end NUMINAMATH_CALUDE_dinnerCostTheorem_l1517_151741


namespace NUMINAMATH_CALUDE_tan_alpha_value_l1517_151746

theorem tan_alpha_value (α : Real) (h : Real.tan (α + π/4) = 1/2) : Real.tan α = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l1517_151746


namespace NUMINAMATH_CALUDE_sum_345_75_base6_l1517_151719

/-- Converts a natural number from base 10 to base 6 -/
def toBase6 (n : ℕ) : List ℕ :=
  sorry

/-- Adds two numbers in base 6 -/
def addBase6 (a b : List ℕ) : List ℕ :=
  sorry

/-- Theorem stating that the sum of 345 and 75 in base 6 is 1540 -/
theorem sum_345_75_base6 :
  addBase6 (toBase6 345) (toBase6 75) = [1, 5, 4, 0] :=
sorry

end NUMINAMATH_CALUDE_sum_345_75_base6_l1517_151719


namespace NUMINAMATH_CALUDE_cube_volume_increase_l1517_151701

theorem cube_volume_increase (s : ℝ) (h : s > 0) : 
  let original_volume := s^3
  let new_edge_length := 1.4 * s
  let new_volume := new_edge_length^3
  (new_volume - original_volume) / original_volume * 100 = 174.4 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_increase_l1517_151701


namespace NUMINAMATH_CALUDE_read_distance_guangzhou_shenyang_l1517_151710

/-- Represents a number in words -/
inductive NumberWord
  | million : ℕ → NumberWord
  | thousand : ℕ → NumberWord
  | hundred : ℕ → NumberWord
  | ten : ℕ → NumberWord
  | one : ℕ → NumberWord

/-- Represents the distance from Guangzhou to Shenyang in meters -/
def distance_guangzhou_shenyang : ℕ := 3036000

/-- Converts a natural number to its word representation -/
def number_to_words (n : ℕ) : List NumberWord :=
  sorry

/-- Theorem stating that the correct way to read 3,036,000 is "three million thirty-six thousand" -/
theorem read_distance_guangzhou_shenyang :
  number_to_words distance_guangzhou_shenyang = 
    [NumberWord.million 3, NumberWord.thousand 36] :=
  sorry

end NUMINAMATH_CALUDE_read_distance_guangzhou_shenyang_l1517_151710


namespace NUMINAMATH_CALUDE_percentage_neither_is_twenty_percent_l1517_151711

/-- Represents the health survey data for teachers -/
structure HealthSurvey where
  total : ℕ
  high_bp : ℕ
  heart_trouble : ℕ
  both : ℕ

/-- Calculates the percentage of teachers with neither high blood pressure nor heart trouble -/
def percentage_neither (survey : HealthSurvey) : ℚ :=
  let neither := survey.total - (survey.high_bp + survey.heart_trouble - survey.both)
  (neither : ℚ) / survey.total * 100

/-- Theorem stating that the percentage of teachers with neither condition is 20% -/
theorem percentage_neither_is_twenty_percent (survey : HealthSurvey)
  (h_total : survey.total = 150)
  (h_high_bp : survey.high_bp = 90)
  (h_heart_trouble : survey.heart_trouble = 60)
  (h_both : survey.both = 30) :
  percentage_neither survey = 20 := by
  sorry

#eval percentage_neither { total := 150, high_bp := 90, heart_trouble := 60, both := 30 }

end NUMINAMATH_CALUDE_percentage_neither_is_twenty_percent_l1517_151711


namespace NUMINAMATH_CALUDE_cube_side_ratio_l1517_151743

/-- The ratio of side lengths of two cubes with given weights -/
theorem cube_side_ratio (w₁ w₂ : ℝ) (h₁ : w₁ > 0) (h₂ : w₂ > 0) :
  w₁ = 7 → w₂ = 56 → (w₂ / w₁)^(1/3 : ℝ) = 2 := by
  sorry

#check cube_side_ratio

end NUMINAMATH_CALUDE_cube_side_ratio_l1517_151743


namespace NUMINAMATH_CALUDE_book_stack_thickness_l1517_151772

/-- Calculates the thickness of a stack of books in inches -/
def stack_thickness (num_books : ℕ) (pages_per_book : ℕ) (pages_per_inch : ℕ) : ℚ :=
  (num_books * pages_per_book : ℚ) / pages_per_inch

/-- Proves that the thickness of a stack of 6 books, each with 160 pages,
    is 12 inches when 80 pages make one inch of thickness -/
theorem book_stack_thickness :
  stack_thickness 6 160 80 = 12 := by
  sorry

end NUMINAMATH_CALUDE_book_stack_thickness_l1517_151772


namespace NUMINAMATH_CALUDE_cookie_ratio_l1517_151721

theorem cookie_ratio (monday tuesday wednesday : ℕ) : 
  monday = 32 →
  tuesday = monday / 2 →
  monday + tuesday + (wednesday - 4) = 92 →
  wednesday / tuesday = 3 := by
  sorry

end NUMINAMATH_CALUDE_cookie_ratio_l1517_151721


namespace NUMINAMATH_CALUDE_discount_calculation_l1517_151713

def calculate_final_price (initial_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : ℝ :=
  initial_price * (1 - discount1) * (1 - discount2)

theorem discount_calculation (hat_price tie_price : ℝ) 
  (hat_discount1 hat_discount2 tie_discount1 tie_discount2 : ℝ) : 
  hat_price = 20 → tie_price = 15 → 
  hat_discount1 = 0.25 → hat_discount2 = 0.20 → 
  tie_discount1 = 0.10 → tie_discount2 = 0.30 → 
  calculate_final_price hat_price hat_discount1 hat_discount2 = 12 ∧ 
  calculate_final_price tie_price tie_discount1 tie_discount2 = 9.45 := by
  sorry

#check discount_calculation

end NUMINAMATH_CALUDE_discount_calculation_l1517_151713


namespace NUMINAMATH_CALUDE_orthogonality_condition_l1517_151706

/-- Two circles are orthogonal if their tangents at intersection points are perpendicular -/
def orthogonal (R₁ R₂ d : ℝ) : Prop :=
  d^2 = R₁^2 + R₂^2

theorem orthogonality_condition (R₁ R₂ d : ℝ) (h₁ : R₁ > 0) (h₂ : R₂ > 0) (h₃ : d > 0) :
  orthogonal R₁ R₂ d ↔ d^2 = R₁^2 + R₂^2 :=
sorry

end NUMINAMATH_CALUDE_orthogonality_condition_l1517_151706


namespace NUMINAMATH_CALUDE_total_football_games_l1517_151750

/-- The total number of football games in a year, given the number of games
    Keith attended and missed. -/
theorem total_football_games (attended : ℕ) (missed : ℕ) 
  (h1 : attended = 4) (h2 : missed = 4) : 
  attended + missed = 8 := by
  sorry

end NUMINAMATH_CALUDE_total_football_games_l1517_151750


namespace NUMINAMATH_CALUDE_employed_males_percentage_l1517_151718

theorem employed_males_percentage
  (total_population : ℝ)
  (employed_percentage : ℝ)
  (employed_females_percentage : ℝ)
  (h1 : employed_percentage = 60)
  (h2 : employed_females_percentage = 25)
  (h3 : total_population > 0) :
  let employed := (employed_percentage / 100) * total_population
  let employed_females := (employed_females_percentage / 100) * employed
  let employed_males := employed - employed_females
  (employed_males / total_population) * 100 = 45 := by
sorry

end NUMINAMATH_CALUDE_employed_males_percentage_l1517_151718


namespace NUMINAMATH_CALUDE_rotation_result_l1517_151775

-- Define the shapes
inductive Shape
| Triangle
| SmallCircle
| Pentagon

-- Define the positions
inductive Position
| Top
| LowerLeft
| LowerRight

-- Define the configuration as a function from Shape to Position
def Configuration := Shape → Position

-- Define the initial configuration
def initial_config : Configuration
| Shape.Triangle => Position.Top
| Shape.SmallCircle => Position.LowerLeft
| Shape.Pentagon => Position.LowerRight

-- Define the rotation function
def rotate_150_clockwise (config : Configuration) : Configuration :=
  fun shape =>
    match config shape with
    | Position.Top => Position.LowerRight
    | Position.LowerLeft => Position.Top
    | Position.LowerRight => Position.LowerLeft

-- Theorem statement
theorem rotation_result :
  let final_config := rotate_150_clockwise initial_config
  final_config Shape.Triangle = Position.LowerRight ∧
  final_config Shape.SmallCircle = Position.Top ∧
  final_config Shape.Pentagon = Position.LowerLeft :=
sorry

end NUMINAMATH_CALUDE_rotation_result_l1517_151775


namespace NUMINAMATH_CALUDE_acute_angle_range_l1517_151764

/-- The range of values for the acute angle of a line with slope k = 2m / (m^2 + 1) -/
theorem acute_angle_range (m : ℝ) (h1 : m ≥ 0) (h2 : m^2 + 1 ≥ 2*m) :
  let k := 2*m / (m^2 + 1)
  let θ := Real.arctan k
  0 ≤ θ ∧ θ ≤ π/4 :=
sorry

end NUMINAMATH_CALUDE_acute_angle_range_l1517_151764


namespace NUMINAMATH_CALUDE_part1_part2_l1517_151777

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 6

-- Part 1
theorem part1 : ∀ x : ℝ, f 5 x < 0 ↔ -3 < x ∧ x < -2 := by sorry

-- Part 2
theorem part2 : ∀ a : ℝ, (∀ x : ℝ, f a x > 0) ↔ -2 * Real.sqrt 6 < a ∧ a < 2 * Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_part1_part2_l1517_151777


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1517_151757

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum : a 3 + a 4 + a 5 = 12) :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 28 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1517_151757


namespace NUMINAMATH_CALUDE_spider_total_distance_l1517_151739

def spider_movement (start : ℤ) (first_move : ℤ) (second_move : ℤ) : ℕ :=
  (Int.natAbs (first_move - start)) + (Int.natAbs (second_move - first_move))

theorem spider_total_distance :
  spider_movement 3 (-4) 8 = 19 := by
  sorry

end NUMINAMATH_CALUDE_spider_total_distance_l1517_151739


namespace NUMINAMATH_CALUDE_y_completion_time_l1517_151712

/-- The time Y takes to complete the entire work alone, given:
  * X can do the entire work in 40 days
  * X works for 8 days
  * Y finishes the remaining work in 20 days
-/
theorem y_completion_time (x_total_days : ℕ) (x_worked_days : ℕ) (y_completion_days : ℕ) :
  x_total_days = 40 →
  x_worked_days = 8 →
  y_completion_days = 20 →
  (x_worked_days : ℚ) / x_total_days + (y_completion_days : ℚ) * (1 - (x_worked_days : ℚ) / x_total_days) = 1 →
  25 = (1 / (1 / y_completion_days * (1 - (x_worked_days : ℚ) / x_total_days))) := by
  sorry

end NUMINAMATH_CALUDE_y_completion_time_l1517_151712


namespace NUMINAMATH_CALUDE_integer_pair_divisibility_l1517_151765

theorem integer_pair_divisibility (m n : ℕ+) : 
  (∃ k : ℤ, (m : ℤ) + (n : ℤ)^2 = k * ((m : ℤ)^2 - (n : ℤ))) ∧ 
  (∃ l : ℤ, (n : ℤ) + (m : ℤ)^2 = l * ((n : ℤ)^2 - (m : ℤ))) ↔ 
  ((m = 2 ∧ n = 1) ∨ (m = 3 ∧ n = 2)) := by
sorry

end NUMINAMATH_CALUDE_integer_pair_divisibility_l1517_151765


namespace NUMINAMATH_CALUDE_stone_counting_135_l1517_151751

/-- Represents the stone counting pattern described in the problem -/
def stoneCounting (n : ℕ) : ℕ := 
  let cycle := n % 24
  if cycle ≤ 12 
  then (cycle + 1) / 2 
  else (25 - cycle) / 2

/-- The problem statement -/
theorem stone_counting_135 : stoneCounting 135 = 3 := by
  sorry

end NUMINAMATH_CALUDE_stone_counting_135_l1517_151751


namespace NUMINAMATH_CALUDE_quadratic_equation_complete_square_l1517_151705

theorem quadratic_equation_complete_square :
  ∃ (r s : ℝ), 
    (∀ x, 15 * x^2 - 60 * x - 135 = 0 ↔ (x + r)^2 = s) ∧
    r + s = 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_complete_square_l1517_151705


namespace NUMINAMATH_CALUDE_sam_pennies_l1517_151733

theorem sam_pennies (initial_pennies final_pennies : ℕ) 
  (h1 : initial_pennies = 98) 
  (h2 : final_pennies = 191) : 
  final_pennies - initial_pennies = 93 := by
  sorry

end NUMINAMATH_CALUDE_sam_pennies_l1517_151733


namespace NUMINAMATH_CALUDE_circle_bisection_l1517_151732

/-- Given two circles in the plane:
    Circle 1: (x-a)^2 + (y-b)^2 = b^2 + 1
    Circle 2: (x+1)^2 + (y+1)^2 = 4
    If Circle 1 always bisects the circumference of Circle 2,
    then the relationship between a and b satisfies: a^2 + 2a + 2b + 5 = 0 -/
theorem circle_bisection (a b : ℝ) : 
  (∀ x y : ℝ, (x - a)^2 + (y - b)^2 = b^2 + 1 → (x + 1)^2 + (y + 1)^2 = 4 → 
    ∃ t : ℝ, x = -1 + t * (2 + 2*a) ∧ y = -1 + t * (2 + 2*b)) → 
  a^2 + 2*a + 2*b + 5 = 0 :=
sorry

end NUMINAMATH_CALUDE_circle_bisection_l1517_151732


namespace NUMINAMATH_CALUDE_f_increasing_sufficient_not_necessary_l1517_151760

/-- The function f(x) defined as |x-a| + |x| --/
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x|

/-- f is increasing on [0, +∞) --/
def is_increasing_on_nonneg (a : ℝ) : Prop :=
  ∀ x y, 0 ≤ x → 0 ≤ y → x ≤ y → f a x ≤ f a y

theorem f_increasing_sufficient_not_necessary :
  (∀ a : ℝ, a < 0 → is_increasing_on_nonneg a) ∧
  (∃ a : ℝ, a ≥ 0 ∧ is_increasing_on_nonneg a) :=
sorry

end NUMINAMATH_CALUDE_f_increasing_sufficient_not_necessary_l1517_151760


namespace NUMINAMATH_CALUDE_divide_books_into_portions_l1517_151794

theorem divide_books_into_portions (n : ℕ) (k : ℕ) : n = 6 → k = 3 → 
  (Nat.choose n 2 * Nat.choose (n - 2) 2) / Nat.factorial k = 15 := by
  sorry

end NUMINAMATH_CALUDE_divide_books_into_portions_l1517_151794


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_3_a_value_when_p_intersect_q_equals_q_l1517_151798

-- Define the solution sets P and Q
def P (a : ℝ) : Set ℝ := {x : ℝ | (x - a) / (x + 1) < 0}
def Q : Set ℝ := {x : ℝ | |x - 1| ≤ 1}

-- Theorem 1: When a = 3, P = {x | -1 < x < 3}
theorem solution_set_when_a_is_3 : 
  P 3 = {x : ℝ | -1 < x ∧ x < 3} := by sorry

-- Theorem 2: When P ∩ Q = Q, a = 2
theorem a_value_when_p_intersect_q_equals_q : 
  (∃ a : ℝ, a > 0 ∧ P a ∩ Q = Q) → (∃ a : ℝ, a = 2 ∧ P a ∩ Q = Q) := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_3_a_value_when_p_intersect_q_equals_q_l1517_151798


namespace NUMINAMATH_CALUDE_product_congruence_l1517_151791

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : List ℕ :=
  List.range n |>.map (fun i => a₁ + i * d)

def product_of_list (l : List ℕ) : ℕ :=
  l.foldl (· * ·) 1

theorem product_congruence :
  let seq := arithmetic_sequence 3 5 21
  (product_of_list seq) % 6 = 3 := by
  sorry

end NUMINAMATH_CALUDE_product_congruence_l1517_151791


namespace NUMINAMATH_CALUDE_maria_change_l1517_151774

/-- The change Maria receives when buying apples -/
theorem maria_change (num_apples : ℕ) (price_per_apple : ℚ) (paid_amount : ℚ) : 
  num_apples = 5 → 
  price_per_apple = 3/4 → 
  paid_amount = 10 → 
  paid_amount - (num_apples : ℚ) * price_per_apple = 25/4 := by
  sorry

#check maria_change

end NUMINAMATH_CALUDE_maria_change_l1517_151774


namespace NUMINAMATH_CALUDE_one_row_with_ten_seats_l1517_151749

/-- Represents the seating arrangement in a theater --/
structure TheaterSeating where
  total_people : ℕ
  rows_with_ten : ℕ
  rows_with_nine : ℕ

/-- Checks if the seating arrangement is valid --/
def is_valid_seating (s : TheaterSeating) : Prop :=
  s.total_people = 55 ∧
  s.rows_with_ten * 10 + s.rows_with_nine * 9 = s.total_people

/-- Theorem stating that there is exactly one row seating 10 people --/
theorem one_row_with_ten_seats :
  ∃! s : TheaterSeating, is_valid_seating s ∧ s.rows_with_ten = 1 :=
sorry

end NUMINAMATH_CALUDE_one_row_with_ten_seats_l1517_151749


namespace NUMINAMATH_CALUDE_bicycle_route_length_l1517_151738

/-- The total length of a rectangular path given the length of one horizontal and one vertical side. -/
def rectangularPathLength (horizontal vertical : ℝ) : ℝ :=
  2 * (horizontal + vertical)

/-- Theorem: The total length of a rectangular path with horizontal sides of 13 km and vertical sides of 13 km is 52 km. -/
theorem bicycle_route_length : rectangularPathLength 13 13 = 52 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_route_length_l1517_151738


namespace NUMINAMATH_CALUDE_no_integer_solutions_for_fourth_power_equation_l1517_151781

theorem no_integer_solutions_for_fourth_power_equation :
  ¬ ∃ (a b c : ℤ), a^4 + b^4 = c^4 + 3 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_for_fourth_power_equation_l1517_151781


namespace NUMINAMATH_CALUDE_moon_arrangements_l1517_151785

def word : String := "MOON"

theorem moon_arrangements :
  (List.permutations (word.toList)).length = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_moon_arrangements_l1517_151785


namespace NUMINAMATH_CALUDE_bd_always_greater_than_10_l1517_151770

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  -- Right angle at C
  (C.1 - A.1) * (B.1 - A.1) + (C.2 - A.2) * (B.2 - A.2) = 0 ∧
  -- Angle at B is 45°
  (C.1 - B.1) * (A.1 - B.1) + (C.2 - B.2) * (A.2 - B.2) = 
    Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) * Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) / Real.sqrt 2 ∧
  -- Length of AB is 20
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 400

-- Define a point P inside the triangle
def InsideTriangle (P A B C : ℝ × ℝ) : Prop :=
  ∃ (α β γ : ℝ), α ≥ 0 ∧ β ≥ 0 ∧ γ ≥ 0 ∧ α + β + γ = 1 ∧
  P = (α * A.1 + β * B.1 + γ * C.1, α * A.2 + β * B.2 + γ * C.2)

-- Define point D as the intersection of BP and AC
def IntersectionPoint (D B P A C : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), D = (A.1 + t * (C.1 - A.1), A.2 + t * (C.2 - A.2)) ∧
              ∃ (s : ℝ), D = (B.1 + s * (P.1 - B.1), B.2 + s * (P.2 - B.2))

-- Theorem statement
theorem bd_always_greater_than_10 
  (A B C P D: ℝ × ℝ) 
  (h1 : Triangle A B C) 
  (h2 : InsideTriangle P A B C) 
  (h3 : IntersectionPoint D B P A C) : 
  (D.1 - B.1)^2 + (D.2 - B.2)^2 > 100 := by
  sorry

end NUMINAMATH_CALUDE_bd_always_greater_than_10_l1517_151770


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1517_151782

/-- Given a hyperbola with asymptote equations y = ± (1/3)x and one focus at (√10, 0),
    its standard equation is x²/9 - y² = 1 -/
theorem hyperbola_equation (x y : ℝ) :
  (∃ (k : ℝ), k > 0 ∧ y = k * x / 3 ∨ y = -k * x / 3) →  -- asymptote equations
  (∃ (c : ℝ), c^2 = 10 ∧ (c, 0) ∈ {p : ℝ × ℝ | p.1^2 / 9 - p.2^2 = 1}) →  -- focus condition
  (x^2 / 9 - y^2 = 1) :=  -- standard equation
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1517_151782


namespace NUMINAMATH_CALUDE_first_replaced_man_age_l1517_151759

/-- The age of the first replaced man in a group scenario --/
def age_of_first_replaced_man (initial_count : ℕ) (age_increase : ℕ) (second_replaced_age : ℕ) (new_men_average_age : ℕ) : ℕ :=
  initial_count * age_increase + new_men_average_age * 2 - second_replaced_age - initial_count * age_increase

/-- Theorem stating the age of the first replaced man is 21 --/
theorem first_replaced_man_age :
  age_of_first_replaced_man 15 2 23 37 = 21 := by
  sorry

#eval age_of_first_replaced_man 15 2 23 37

end NUMINAMATH_CALUDE_first_replaced_man_age_l1517_151759


namespace NUMINAMATH_CALUDE_filter_price_calculation_l1517_151730

/-- Proves that the price of each of the remaining 2 filters is $22.55 -/
theorem filter_price_calculation (kit_price : ℝ) (filter1_price : ℝ) (filter2_price : ℝ) 
  (discount_percentage : ℝ) :
  kit_price = 72.50 →
  filter1_price = 12.45 →
  filter2_price = 11.50 →
  discount_percentage = 0.1103448275862069 →
  ∃ (x : ℝ), 
    x = 22.55 ∧
    kit_price = (1 - discount_percentage) * (2 * filter1_price + 2 * x + filter2_price) := by
  sorry

end NUMINAMATH_CALUDE_filter_price_calculation_l1517_151730


namespace NUMINAMATH_CALUDE_parabola_vertex_coefficients_l1517_151762

/-- Prove that for a parabola y = ax² + bx with vertex at (3,3), the values of a and b are a = -1/3 and b = 2. -/
theorem parabola_vertex_coefficients (a b : ℝ) : 
  (∀ x, 3 = a * x^2 + b * x ↔ x = 3) ∧ (3 = a * 3^2 + b * 3) → 
  a = -1/3 ∧ b = 2 := by
sorry

end NUMINAMATH_CALUDE_parabola_vertex_coefficients_l1517_151762


namespace NUMINAMATH_CALUDE_car_profit_percentage_l1517_151768

theorem car_profit_percentage (P : ℝ) (h : P > 0) : 
  let buying_price := P * (1 - 0.2)
  let selling_price := buying_price * (1 + 0.45)
  let profit := selling_price - P
  profit / P * 100 = 16 := by
sorry

end NUMINAMATH_CALUDE_car_profit_percentage_l1517_151768


namespace NUMINAMATH_CALUDE_john_average_speed_l1517_151707

-- Define the start time, break time, end time, and total distance
def start_time : ℕ := 8 * 60 + 15  -- 8:15 AM in minutes
def break_start : ℕ := 12 * 60  -- 12:00 PM in minutes
def break_duration : ℕ := 30  -- 30 minutes
def end_time : ℕ := 14 * 60 + 45  -- 2:45 PM in minutes
def total_distance : ℕ := 240  -- miles

-- Calculate the total driving time in hours
def total_driving_time : ℚ :=
  (break_start - start_time + (end_time - (break_start + break_duration))) / 60

-- Define the average speed
def average_speed : ℚ := total_distance / total_driving_time

-- Theorem to prove
theorem john_average_speed :
  average_speed = 40 :=
sorry

end NUMINAMATH_CALUDE_john_average_speed_l1517_151707


namespace NUMINAMATH_CALUDE_subset_condition_disjoint_condition_l1517_151758

-- Define sets A and B
def A : Set ℝ := {x | 1 < x ∧ x < 2}
def B (a : ℝ) : Set ℝ := {x | 2*a - 1 < x ∧ x < 2*a + 1}

-- Theorem for A ⊆ B
theorem subset_condition (a : ℝ) : A ⊆ B a ↔ 1/2 ≤ a ∧ a ≤ 1 := by sorry

-- Theorem for A ∩ B = ∅
theorem disjoint_condition (a : ℝ) : A ∩ B a = ∅ ↔ a ≥ 3/2 ∨ a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_subset_condition_disjoint_condition_l1517_151758


namespace NUMINAMATH_CALUDE_world_expo_arrangements_l1517_151717

theorem world_expo_arrangements (n : ℕ) (k : ℕ) :
  n = 7 → k = 3 → (n.choose k) * ((n - k).choose k) = 140 := by
  sorry

end NUMINAMATH_CALUDE_world_expo_arrangements_l1517_151717


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l1517_151716

theorem pure_imaginary_condition (a : ℝ) : 
  let z : ℂ := (1 + Complex.I) / (1 + a * Complex.I)
  (∃ (b : ℝ), z = b * Complex.I) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l1517_151716


namespace NUMINAMATH_CALUDE_second_group_frequency_l1517_151780

theorem second_group_frequency (total : ℕ) (group1 group2 group3 group4 group5 : ℕ) 
  (h1 : total = 50)
  (h2 : group1 = 2)
  (h3 : group3 = 8)
  (h4 : group4 = 10)
  (h5 : group5 = 20)
  (h6 : total = group1 + group2 + group3 + group4 + group5) :
  group2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_second_group_frequency_l1517_151780


namespace NUMINAMATH_CALUDE_grocery_store_salary_l1517_151755

/-- Calculates the total daily salary of employees in a grocery store. -/
def total_daily_salary (manager_salary : ℕ) (clerk_salary : ℕ) (num_managers : ℕ) (num_clerks : ℕ) : ℕ :=
  manager_salary * num_managers + clerk_salary * num_clerks

/-- Proves that the total daily salary of all employees in the grocery store is $16. -/
theorem grocery_store_salary : total_daily_salary 5 2 2 3 = 16 := by
  sorry

end NUMINAMATH_CALUDE_grocery_store_salary_l1517_151755


namespace NUMINAMATH_CALUDE_parking_lot_cars_l1517_151784

/-- Given a parking lot with large and small cars, prove the number of each type. -/
theorem parking_lot_cars (total_vehicles : ℕ) (total_wheels : ℕ) 
  (large_car_wheels : ℕ) (small_car_wheels : ℕ) 
  (h_total_vehicles : total_vehicles = 6)
  (h_total_wheels : total_wheels = 32)
  (h_large_car_wheels : large_car_wheels = 6)
  (h_small_car_wheels : small_car_wheels = 4) :
  ∃ (large_cars small_cars : ℕ),
    large_cars + small_cars = total_vehicles ∧
    large_cars * large_car_wheels + small_cars * small_car_wheels = total_wheels ∧
    large_cars = 4 ∧
    small_cars = 2 := by
  sorry

end NUMINAMATH_CALUDE_parking_lot_cars_l1517_151784


namespace NUMINAMATH_CALUDE_pirate_treasure_distribution_l1517_151728

/-- The number of coins in the final round of distribution -/
def x : ℕ := sorry

/-- The sum of coins Pete gives himself in each round -/
def petes_coins (n : ℕ) : ℕ := n * (n + 1) / 2

theorem pirate_treasure_distribution :
  -- Paul ends up with x coins
  -- Pete ends up with 5x coins
  -- Pete's coins follow the pattern 1 + 2 + 3 + ... + x
  -- The total number of coins is 54
  x + 5 * x = 54 ∧ petes_coins x = 5 * x := by sorry

end NUMINAMATH_CALUDE_pirate_treasure_distribution_l1517_151728


namespace NUMINAMATH_CALUDE_fgh_supermarket_difference_l1517_151745

theorem fgh_supermarket_difference (total : ℕ) (us : ℕ) (h1 : total = 84) (h2 : us = 47) (h3 : us > total - us) : us - (total - us) = 10 := by
  sorry

end NUMINAMATH_CALUDE_fgh_supermarket_difference_l1517_151745


namespace NUMINAMATH_CALUDE_restaurant_budget_allocation_l1517_151725

/-- Given a restaurant's budget allocation, prove that the fraction of
    remaining budget spent on food and beverages is 1/4. -/
theorem restaurant_budget_allocation (B : ℝ) (B_pos : B > 0) :
  let rent : ℝ := (1 / 4) * B
  let remaining : ℝ := B - rent
  let food_and_beverages : ℝ := 0.1875 * B
  food_and_beverages / remaining = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_restaurant_budget_allocation_l1517_151725


namespace NUMINAMATH_CALUDE_kayla_apples_l1517_151722

theorem kayla_apples (total : ℕ) (kayla kylie : ℕ) : 
  total = 200 →
  kayla + kylie = total →
  kayla = kylie / 4 →
  kayla = 40 := by
sorry

end NUMINAMATH_CALUDE_kayla_apples_l1517_151722


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_for_real_solutions_l1517_151790

-- Define the function f
def f (x a : ℝ) : ℝ := |x + a^2| + |x + 2*a - 5|

-- Theorem for part (1)
theorem solution_set_when_a_is_one :
  {x : ℝ | |x + 1| + |x - 3| < 5} = Set.Ioo (-3/2) (7/2) := by sorry

-- Theorem for part (2)
theorem range_of_a_for_real_solutions :
  {a : ℝ | ∃ x, f x a < 5} = Set.Ioo 0 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_for_real_solutions_l1517_151790


namespace NUMINAMATH_CALUDE_number_calculation_l1517_151787

theorem number_calculation (x : ℝ) : (0.5 * x - 10 = 25) → x = 70 := by
  sorry

end NUMINAMATH_CALUDE_number_calculation_l1517_151787


namespace NUMINAMATH_CALUDE_g_of_3_equals_6_l1517_151724

-- Define the function g
def g (x : ℝ) : ℝ := x^3 - 3*x^2 + 2*x

-- Theorem statement
theorem g_of_3_equals_6 : g 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_g_of_3_equals_6_l1517_151724


namespace NUMINAMATH_CALUDE_mat_cost_per_square_meter_l1517_151726

/-- Calculates the cost per square meter of mat for a rectangular hall -/
theorem mat_cost_per_square_meter 
  (length width height : ℝ) 
  (total_expenditure : ℝ) 
  (h_length : length = 20) 
  (h_width : width = 15) 
  (h_height : height = 5) 
  (h_expenditure : total_expenditure = 38000) : 
  total_expenditure / (length * width + 2 * (length * height + width * height)) = 58.46 := by
  sorry

end NUMINAMATH_CALUDE_mat_cost_per_square_meter_l1517_151726


namespace NUMINAMATH_CALUDE_kaleb_toy_purchase_l1517_151709

def number_of_toys (initial_money game_cost saving_amount toy_cost : ℕ) : ℕ :=
  ((initial_money - game_cost - saving_amount) / toy_cost)

theorem kaleb_toy_purchase :
  number_of_toys 12 8 2 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_kaleb_toy_purchase_l1517_151709


namespace NUMINAMATH_CALUDE_amc10_participation_increase_l1517_151702

def participation : Fin 6 → ℕ
  | 0 => 50  -- 2010
  | 1 => 56  -- 2011
  | 2 => 62  -- 2012
  | 3 => 68  -- 2013
  | 4 => 77  -- 2014
  | 5 => 81  -- 2015

def percentageIncrease (a b : ℕ) : ℚ :=
  (b - a : ℚ) / a * 100

def largestIncreaseBetween2013And2014 : Prop :=
  ∀ i : Fin 5, percentageIncrease (participation i) (participation (i + 1)) ≤
    percentageIncrease (participation 3) (participation 4)

theorem amc10_participation_increase : largestIncreaseBetween2013And2014 := by
  sorry

end NUMINAMATH_CALUDE_amc10_participation_increase_l1517_151702


namespace NUMINAMATH_CALUDE_P_greater_than_Q_l1517_151714

theorem P_greater_than_Q (a : ℝ) (h : a > -38) :
  Real.sqrt (a + 40) - Real.sqrt (a + 41) > Real.sqrt (a + 38) - Real.sqrt (a + 39) := by
sorry

end NUMINAMATH_CALUDE_P_greater_than_Q_l1517_151714


namespace NUMINAMATH_CALUDE_boat_travel_distance_l1517_151703

/-- Prove that the distance between two destinations is 40 km given the specified conditions. -/
theorem boat_travel_distance 
  (boatsman_speed : ℝ) 
  (river_speed : ℝ) 
  (time_difference : ℝ) 
  (h1 : boatsman_speed = 7)
  (h2 : river_speed = 3)
  (h3 : time_difference = 6)
  (h4 : (boatsman_speed + river_speed) * (boatsman_speed - river_speed) * time_difference = 
        2 * river_speed * boatsman_speed * (boatsman_speed - river_speed)) :
  (boatsman_speed + river_speed) * (boatsman_speed - river_speed) * time_difference / 
  (2 * river_speed) = 40 := by
  sorry

end NUMINAMATH_CALUDE_boat_travel_distance_l1517_151703


namespace NUMINAMATH_CALUDE_exists_asymmetric_but_rotational_invariant_figure_l1517_151792

/-- A convex figure in a 2D plane. -/
structure ConvexFigure where
  -- We don't need to fully define the structure, just declare it exists
  dummy : Unit

/-- Represents a rotation in 2D space. -/
structure Rotation where
  angle : ℝ

/-- Checks if a figure has an axis of symmetry. -/
def hasAxisOfSymmetry (figure : ConvexFigure) : Prop :=
  sorry

/-- Applies a rotation to a figure. -/
def applyRotation (figure : ConvexFigure) (rotation : Rotation) : ConvexFigure :=
  sorry

/-- Checks if a figure is invariant under a given rotation. -/
def isInvariantUnderRotation (figure : ConvexFigure) (rotation : Rotation) : Prop :=
  applyRotation figure rotation = figure

/-- The main theorem: There exists a convex figure with no axis of symmetry
    but invariant under 120° rotation. -/
theorem exists_asymmetric_but_rotational_invariant_figure :
  ∃ (figure : ConvexFigure),
    ¬(hasAxisOfSymmetry figure) ∧
    isInvariantUnderRotation figure ⟨2 * Real.pi / 3⟩ := by
  sorry

end NUMINAMATH_CALUDE_exists_asymmetric_but_rotational_invariant_figure_l1517_151792


namespace NUMINAMATH_CALUDE_ellipse_equation_l1517_151776

/-- An ellipse with center at origin, eccentricity √3/2, and one focus coinciding with
    the focus of the parabola x² = -4√3y has the equation x² + y²/4 = 1 -/
theorem ellipse_equation (x y : ℝ) : 
  let e : ℝ := Real.sqrt 3 / 2
  let c : ℝ := Real.sqrt 3  -- Distance from center to focus
  let a : ℝ := c / e        -- Semi-major axis
  let b : ℝ := Real.sqrt (a^2 - c^2)  -- Semi-minor axis
  (e = Real.sqrt 3 / 2) → 
  (c = Real.sqrt 3) →      -- Focus coincides with parabola focus
  (x^2 + y^2 / 4 = 1) ↔ (x^2 / a^2 + y^2 / b^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l1517_151776


namespace NUMINAMATH_CALUDE_gcd_168_486_l1517_151731

def continuedProportionateReduction (a b : ℕ) : ℕ :=
  if a = 0 then b
  else if b = 0 then a
  else if a ≥ b then continuedProportionateReduction (a - b) b
  else continuedProportionateReduction a (b - a)

theorem gcd_168_486 :
  continuedProportionateReduction 168 486 = 6 ∧ 
  (∀ d : ℕ, d ∣ 168 ∧ d ∣ 486 → d ≤ 6) := by sorry

end NUMINAMATH_CALUDE_gcd_168_486_l1517_151731


namespace NUMINAMATH_CALUDE_nancy_marks_l1517_151769

theorem nancy_marks (history : ℕ) (home_economics : ℕ) (physical_education : ℕ) (art : ℕ) (average : ℕ) 
  (h1 : history = 75)
  (h2 : home_economics = 52)
  (h3 : physical_education = 68)
  (h4 : art = 89)
  (h5 : average = 70) :
  ∃ (american_literature : ℕ), 
    (history + home_economics + physical_education + art + american_literature) / 5 = average ∧ 
    american_literature = 66 := by
  sorry

end NUMINAMATH_CALUDE_nancy_marks_l1517_151769


namespace NUMINAMATH_CALUDE_dog_weight_gain_exists_l1517_151763

/-- Represents a dog with age and weight -/
structure Dog where
  age : ℕ
  weight : ℝ

/-- Represents the annual weight gain of a dog -/
def annualGain (d : Dog) (gain : ℝ) : Prop :=
  ∃ (initialWeight : ℝ), initialWeight + gain * (d.age - 1) = d.weight

/-- Theorem stating that for any dog, there exists some annual weight gain -/
theorem dog_weight_gain_exists (d : Dog) : ∃ (gain : ℝ), annualGain d gain :=
sorry

end NUMINAMATH_CALUDE_dog_weight_gain_exists_l1517_151763


namespace NUMINAMATH_CALUDE_unique_N_for_210_terms_l1517_151720

/-- The number of terms in the expansion of (a+b+c+d+1)^n that contain all four variables
    a, b, c, and d, each to some positive power -/
def numTermsWithAllVars (n : ℕ) : ℕ := Nat.choose n 4

theorem unique_N_for_210_terms :
  ∃! N : ℕ, N > 0 ∧ numTermsWithAllVars N = 210 := by sorry

end NUMINAMATH_CALUDE_unique_N_for_210_terms_l1517_151720


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_seventeen_sixths_l1517_151766

theorem sqrt_sum_equals_seventeen_sixths : 
  Real.sqrt (9 / 4) + Real.sqrt (16 / 9) = 17 / 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_seventeen_sixths_l1517_151766


namespace NUMINAMATH_CALUDE_quadratic_equation_root_and_sum_l1517_151793

theorem quadratic_equation_root_and_sum : 
  ∃ (a b c : ℚ), 
    (a = 1 ∧ b = 6 ∧ c = -4) ∧ 
    (a * (Real.sqrt 5 - 3)^2 + b * (Real.sqrt 5 - 3) + c = 0) ∧
    (∀ x y : ℝ, x^2 + 6*x - 4 = 0 ∧ y^2 + 6*y - 4 = 0 → x + y = -6) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_and_sum_l1517_151793


namespace NUMINAMATH_CALUDE_softball_team_ratio_l1517_151752

theorem softball_team_ratio : 
  ∀ (men women : ℕ), 
  women = men + 6 →
  men + women = 24 →
  (men : ℚ) / women = 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_softball_team_ratio_l1517_151752


namespace NUMINAMATH_CALUDE_complex_division_real_l1517_151708

def complex (a b : ℝ) : ℂ := a + b * Complex.I

theorem complex_division_real (b : ℝ) :
  let z₁ : ℂ := complex 3 (-b)
  let z₂ : ℂ := complex 1 (-2)
  (∃ (r : ℝ), z₁ / z₂ = r) → b = 6 := by sorry

end NUMINAMATH_CALUDE_complex_division_real_l1517_151708


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1517_151734

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (5 + 7 * i) / (2 + 3 * i) = 31 / 13 - (1 / 13) * i :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1517_151734


namespace NUMINAMATH_CALUDE_initial_speed_is_50_l1517_151742

/-- Represents the journey with increasing speed -/
structure Journey where
  distance : ℝ  -- Total distance in km
  time : ℝ      -- Total time in hours
  speedIncrease : ℝ  -- Speed increase in km/h
  intervalTime : ℝ   -- Time interval for speed increase in hours

/-- Calculates the initial speed given a journey -/
def calculateInitialSpeed (j : Journey) : ℝ :=
  sorry

/-- Theorem stating that for the given conditions, the initial speed is 50 km/h -/
theorem initial_speed_is_50 : 
  let j : Journey := {
    distance := 52,
    time := 48 / 60,  -- 48 minutes converted to hours
    speedIncrease := 10,
    intervalTime := 12 / 60  -- 12 minutes converted to hours
  }
  calculateInitialSpeed j = 50 := by
  sorry

end NUMINAMATH_CALUDE_initial_speed_is_50_l1517_151742


namespace NUMINAMATH_CALUDE_number_divisibility_l1517_151761

theorem number_divisibility (A B C D : ℤ) :
  let N := 1000*D + 100*C + 10*B + A
  (∃ k : ℤ, A + 2*B = 4*k → ∃ m : ℤ, N = 4*m) ∧
  (∃ k : ℤ, A + 2*B + 4*C = 8*k → ∃ m : ℤ, N = 8*m) ∧
  (∃ k : ℤ, A + 2*B + 4*C + 8*D = 16*k ∧ ∃ j : ℤ, B = 2*j → ∃ m : ℤ, N = 16*m) :=
by sorry

end NUMINAMATH_CALUDE_number_divisibility_l1517_151761


namespace NUMINAMATH_CALUDE_robert_reading_capacity_l1517_151754

/-- The number of full books Robert can read in a given time -/
def books_read (pages_per_hour : ℕ) (pages_per_book : ℕ) (hours : ℕ) : ℕ :=
  (pages_per_hour * hours) / pages_per_book

/-- Theorem: Robert can read 2 full 360-page books in 8 hours at 120 pages per hour -/
theorem robert_reading_capacity : books_read 120 360 8 = 2 := by
  sorry

end NUMINAMATH_CALUDE_robert_reading_capacity_l1517_151754


namespace NUMINAMATH_CALUDE_imaginary_part_of_one_plus_i_squared_l1517_151700

theorem imaginary_part_of_one_plus_i_squared (i : ℂ) : 
  i^2 = -1 → Complex.im ((1 + i)^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_one_plus_i_squared_l1517_151700


namespace NUMINAMATH_CALUDE_log_xyz_value_l1517_151779

-- Define the variables
variable (x y z : ℝ)
variable (log : ℝ → ℝ)

-- State the theorem
theorem log_xyz_value (h1 : log (x * y^3 * z) = 2) (h2 : log (x^2 * y * z^2) = 3) :
  log (x * y * z) = 8/5 := by
  sorry

end NUMINAMATH_CALUDE_log_xyz_value_l1517_151779


namespace NUMINAMATH_CALUDE_find_number_l1517_151767

theorem find_number : ∃ x : ℝ, 4.75 + 0.432 + x = 5.485 ∧ x = 0.303 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l1517_151767


namespace NUMINAMATH_CALUDE_distance_between_points_l1517_151723

/-- The distance between two points given specific travel conditions -/
theorem distance_between_points (speed_A speed_B : ℝ) (stop_time : ℝ) : 
  speed_A = 80 →
  speed_B = 70 →
  stop_time = 1/4 →
  ∃ (distance : ℝ), 
    distance / speed_A = distance / speed_B - stop_time ∧
    distance = 2240 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l1517_151723


namespace NUMINAMATH_CALUDE_unique_modular_congruence_l1517_151773

theorem unique_modular_congruence :
  ∃! n : ℤ, 0 ≤ n ∧ n ≤ 7 ∧ n ≡ -825 [ZMOD 8] := by
  sorry

end NUMINAMATH_CALUDE_unique_modular_congruence_l1517_151773


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_angle_implies_m_range_l1517_151744

/-- Given a hyperbola with equation x² + y²/m = 1, if the asymptote's inclination angle α 
    is in the interval (0, π/3), then m is in the interval (-3, 0). -/
theorem hyperbola_asymptote_angle_implies_m_range (m : ℝ) (α : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2/m = 1 → ∃ k : ℝ, y = k*x ∧ Real.arctan k = α) →
  0 < α ∧ α < π/3 →
  -3 < m ∧ m < 0 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_angle_implies_m_range_l1517_151744


namespace NUMINAMATH_CALUDE_least_sum_of_exponents_l1517_151799

theorem least_sum_of_exponents (h : ℕ+) (a b c d e : ℕ) 
  (h_div_225 : 225 ∣ h) (h_div_216 : 216 ∣ h) (h_div_847 : 847 ∣ h)
  (h_factorization : h = 2^a * 3^b * 5^c * 7^d * 11^e) :
  ∃ (a' b' c' d' e' : ℕ), 
    h = 2^a' * 3^b' * 5^c' * 7^d' * 11^e' ∧
    a' + b' + c' + d' + e' ≤ a + b + c + d + e ∧
    a' + b' + c' + d' + e' = 10 :=
sorry

end NUMINAMATH_CALUDE_least_sum_of_exponents_l1517_151799


namespace NUMINAMATH_CALUDE_ted_work_time_l1517_151704

theorem ted_work_time (julie_rate ted_rate : ℚ) (julie_finish_time : ℚ) : 
  julie_rate = 1/10 →
  ted_rate = 1/8 →
  julie_finish_time = 999999999999999799 / 1000000000000000000 →
  ∃ t : ℚ, t = 4 ∧ (julie_rate + ted_rate) * t + julie_rate * julie_finish_time = 1 :=
by sorry

end NUMINAMATH_CALUDE_ted_work_time_l1517_151704


namespace NUMINAMATH_CALUDE_car_sale_profit_l1517_151796

theorem car_sale_profit (P : ℝ) (h : P > 0) :
  let buying_price := 0.95 * P
  let selling_price := 1.52 * P
  (selling_price - buying_price) / buying_price * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_car_sale_profit_l1517_151796


namespace NUMINAMATH_CALUDE_water_bottle_cost_l1517_151748

/-- Given Barbara's shopping information, prove the cost of each water bottle -/
theorem water_bottle_cost
  (tuna_packs : ℕ)
  (tuna_cost_per_pack : ℚ)
  (water_bottles : ℕ)
  (total_spent : ℚ)
  (different_goods_cost : ℚ)
  (h1 : tuna_packs = 5)
  (h2 : tuna_cost_per_pack = 2)
  (h3 : water_bottles = 4)
  (h4 : total_spent = 56)
  (h5 : different_goods_cost = 40) :
  (total_spent - different_goods_cost - tuna_packs * tuna_cost_per_pack) / water_bottles = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_water_bottle_cost_l1517_151748


namespace NUMINAMATH_CALUDE_exists_point_on_line_with_sum_of_distances_l1517_151795

-- Define the line l
variable (l : Line)

-- Define points A and B
variable (A B : Point)

-- Define the given segment length
variable (a : ℝ)

-- Define the property that A and B are on the same side of l
def sameSideOfLine (A B : Point) (l : Line) : Prop := sorry

-- Define the distance between two points
def distance (P Q : Point) : ℝ := sorry

-- Define what it means for a point to be on a line
def onLine (P : Point) (l : Line) : Prop := sorry

-- State the theorem
theorem exists_point_on_line_with_sum_of_distances
  (h_same_side : sameSideOfLine A B l) :
  ∃ M : Point, onLine M l ∧ distance M A + distance M B = a := sorry

end NUMINAMATH_CALUDE_exists_point_on_line_with_sum_of_distances_l1517_151795


namespace NUMINAMATH_CALUDE_sum_mod_five_l1517_151740

theorem sum_mod_five : (9375 + 9376 + 9377 + 9378) % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_mod_five_l1517_151740


namespace NUMINAMATH_CALUDE_circle_area_ratio_l1517_151756

theorem circle_area_ratio (R_A R_B : ℝ) (h : R_A > 0 ∧ R_B > 0) :
  (60 : ℝ) / 360 * (2 * Real.pi * R_A) = (40 : ℝ) / 360 * (2 * Real.pi * R_B) →
  (R_A^2 * Real.pi) / (R_B^2 * Real.pi) = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l1517_151756


namespace NUMINAMATH_CALUDE_max_value_theorem_l1517_151735

theorem max_value_theorem (x y z : ℝ) 
  (h1 : 9 * x^2 + 4 * y^2 + 25 * z^2 = 1) 
  (h2 : z = 2 * y) : 
  ∀ a b c : ℝ, 9 * a^2 + 4 * b^2 + 25 * c^2 = 1 → c = 2 * b → 
  10 * x + 3 * y + 12 * z ≥ 10 * a + 3 * b + 12 * c ∧
  ∃ x₀ y₀ z₀ : ℝ, 9 * x₀^2 + 4 * y₀^2 + 25 * z₀^2 = 1 ∧ z₀ = 2 * y₀ ∧
  10 * x₀ + 3 * y₀ + 12 * z₀ = Real.sqrt 253 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l1517_151735


namespace NUMINAMATH_CALUDE_solution_form_l1517_151788

-- Define the system of equations
def equation1 (x y z : ℝ) : Prop :=
  x / y + y / z + z / x = x / z + z / y + y / x

def equation2 (x y z : ℝ) : Prop :=
  x^2 + y^2 + z^2 = x*y + y*z + z*x + 4

-- Theorem statement
theorem solution_form (x y z : ℝ) :
  equation1 x y z ∧ equation2 x y z →
  (∃ t : ℝ, (x = t ∧ y = t - 2 ∧ z = t - 2) ∨ (x = t ∧ y = t + 2 ∧ z = t + 2)) :=
by sorry

end NUMINAMATH_CALUDE_solution_form_l1517_151788
