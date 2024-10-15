import Mathlib

namespace NUMINAMATH_CALUDE_valid_integers_exist_l3350_335087

def is_valid_integer (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧
  (n / 1000 + (n / 100 % 10) + (n / 10 % 10) + (n % 10) = 18) ∧
  ((n / 100 % 10) + (n / 10 % 10) = 11) ∧
  (n / 1000 - n % 10 = 3) ∧
  n % 9 = 0

theorem valid_integers_exist : ∃ n : ℕ, is_valid_integer n :=
sorry

end NUMINAMATH_CALUDE_valid_integers_exist_l3350_335087


namespace NUMINAMATH_CALUDE_line_through_point_equal_intercepts_l3350_335052

theorem line_through_point_equal_intercepts :
  ∃ (m b : ℝ), (3 = m * 2 + b) ∧ (∃ (a : ℝ), a ≠ 0 ∧ (a = -b/m ∧ a = b)) :=
by sorry

end NUMINAMATH_CALUDE_line_through_point_equal_intercepts_l3350_335052


namespace NUMINAMATH_CALUDE_money_spent_on_blades_l3350_335001

def total_earned : ℕ := 42
def game_price : ℕ := 8
def num_games : ℕ := 4

theorem money_spent_on_blades : 
  total_earned - (game_price * num_games) = 10 := by
  sorry

end NUMINAMATH_CALUDE_money_spent_on_blades_l3350_335001


namespace NUMINAMATH_CALUDE_softball_team_size_l3350_335095

/-- Proves that a co-ed softball team with 5 more women than men and a men-to-women ratio of 0.5 has 15 total players -/
theorem softball_team_size (men women : ℕ) : 
  women = men + 5 →
  men / women = 1 / 2 →
  men + women = 15 := by
sorry

end NUMINAMATH_CALUDE_softball_team_size_l3350_335095


namespace NUMINAMATH_CALUDE_movements_correctly_classified_l3350_335054

-- Define an enumeration for movement types
inductive MovementType
  | Translation
  | Rotation

-- Define a structure for a movement
structure Movement where
  description : String
  classification : MovementType

-- Define the list of movements
def movements : List Movement := [
  { description := "Xiaoming walking forward 3 meters", classification := MovementType.Translation },
  { description := "Rocket launching into the sky", classification := MovementType.Translation },
  { description := "Car wheels constantly rotating", classification := MovementType.Rotation },
  { description := "Archer shooting an arrow onto the target", classification := MovementType.Translation }
]

-- Theorem statement
theorem movements_correctly_classified :
  movements.map (λ m => m.classification) = 
    [MovementType.Translation, MovementType.Translation, MovementType.Rotation, MovementType.Translation] := by
  sorry


end NUMINAMATH_CALUDE_movements_correctly_classified_l3350_335054


namespace NUMINAMATH_CALUDE_sally_balloons_l3350_335090

theorem sally_balloons (sally_balloons fred_balloons : ℕ) : 
  fred_balloons = 3 * sally_balloons →
  fred_balloons = 18 →
  sally_balloons = 6 := by
sorry

end NUMINAMATH_CALUDE_sally_balloons_l3350_335090


namespace NUMINAMATH_CALUDE_increasing_prime_sequence_ones_digit_l3350_335023

/-- A sequence of four increasing prime numbers with common difference 4 and first term greater than 3 -/
def IncreasingPrimeSequence (p₁ p₂ p₃ p₄ : ℕ) : Prop :=
  Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧
  p₁ > 3 ∧
  p₂ = p₁ + 4 ∧
  p₃ = p₂ + 4 ∧
  p₄ = p₃ + 4

/-- The ones digit of a natural number -/
def onesDigit (n : ℕ) : ℕ :=
  n % 10

theorem increasing_prime_sequence_ones_digit
  (p₁ p₂ p₃ p₄ : ℕ) (h : IncreasingPrimeSequence p₁ p₂ p₃ p₄) :
  onesDigit p₁ = 9 := by
  sorry

end NUMINAMATH_CALUDE_increasing_prime_sequence_ones_digit_l3350_335023


namespace NUMINAMATH_CALUDE_polygon_sides_from_diagonals_l3350_335071

theorem polygon_sides_from_diagonals (D : ℕ) (n : ℕ) : D = n * (n - 3) / 2 → D = 44 → n = 11 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_from_diagonals_l3350_335071


namespace NUMINAMATH_CALUDE_inequality_proof_l3350_335015

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + 3*c)/(a + 2*b + c) + 4*b/(a + b + 2*c) - 8*c/(a + b + 3*c) ≥ -17 + 12*Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3350_335015


namespace NUMINAMATH_CALUDE_divisibility_of_difference_quotient_l3350_335034

theorem divisibility_of_difference_quotient (a b n : ℝ) 
  (ha : a > 0) (hb : b > 0) (hn : n > 0) 
  (h_div : ∃ k : ℤ, a^n - b^n = n * k) :
  ∃ m : ℤ, (a^n - b^n) / (a - b) = n * m := by
sorry

end NUMINAMATH_CALUDE_divisibility_of_difference_quotient_l3350_335034


namespace NUMINAMATH_CALUDE_tangent_circles_m_value_l3350_335077

/-- Circle C₁ with equation x² + y² = 1 -/
def C₁ : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}

/-- Circle C₂ with equation x² + y² - 6x - 8y + m = 0 -/
def C₂ (m : ℝ) : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 - 6*p.1 - 8*p.2 + m = 0}

/-- Two circles are tangent if they intersect at exactly one point -/
def AreTangent (A B : Set (ℝ × ℝ)) : Prop :=
  ∃! p, p ∈ A ∧ p ∈ B

/-- The main theorem: If C₁ and C₂ are tangent, then m = 9 -/
theorem tangent_circles_m_value :
  AreTangent C₁ (C₂ 9) :=
sorry

end NUMINAMATH_CALUDE_tangent_circles_m_value_l3350_335077


namespace NUMINAMATH_CALUDE_m_range_l3350_335096

/-- Given conditions p and q, prove that the range of real numbers m is [-2, -1). -/
theorem m_range (p : ∀ x : ℝ, 2 * x > m * (x^2 + 1))
                (q : ∃ x₀ : ℝ, x₀^2 + 2 * x₀ - m - 1 = 0) :
  m ≥ -2 ∧ m < -1 :=
by sorry

end NUMINAMATH_CALUDE_m_range_l3350_335096


namespace NUMINAMATH_CALUDE_seed_selection_correct_l3350_335060

/-- Represents a random number table --/
def RandomNumberTable := List (List Nat)

/-- Checks if a number is a valid seed number --/
def isValidSeed (n : Nat) : Bool :=
  0 < n && n ≤ 500

/-- Extracts the next three-digit number from the random number table --/
def nextThreeDigitNumber (table : RandomNumberTable) (row : Nat) (col : Nat) : Option Nat :=
  sorry

/-- Selects the first n valid seeds from the random number table --/
def selectValidSeeds (table : RandomNumberTable) (startRow : Nat) (startCol : Nat) (n : Nat) : List Nat :=
  sorry

/-- The given random number table --/
def givenTable : RandomNumberTable := [
  [84, 42, 17, 53, 31, 57, 24, 55, 06, 88, 77, 04, 74, 47, 67],
  [21, 76, 33, 50, 25, 83, 92, 12, 06, 76],
  [63, 01, 63, 78, 59, 16, 95, 55, 67, 19, 98, 10, 50, 71, 75],
  [12, 86, 73, 58, 07, 44, 39, 52, 38, 79],
  [33, 21, 12, 34, 29, 78, 64, 56, 07, 82, 52, 42, 07, 44, 38],
  [15, 51, 00, 13, 42, 99, 66, 02, 79, 54]
]

theorem seed_selection_correct :
  selectValidSeeds givenTable 7 8 5 = [331, 455, 068, 047, 447] :=
sorry

end NUMINAMATH_CALUDE_seed_selection_correct_l3350_335060


namespace NUMINAMATH_CALUDE_volume_ratio_is_two_l3350_335017

/-- Represents the state of an ideal gas -/
structure GasState where
  volume : ℝ
  pressure : ℝ
  temperature : ℝ

/-- Represents a closed cycle of an ideal gas -/
structure GasCycle where
  initial : GasState
  state2 : GasState
  state3 : GasState

/-- The conditions of the gas cycle -/
class CycleConditions (cycle : GasCycle) where
  isobaric_1_2 : cycle.state2.pressure = cycle.initial.pressure
  volume_increase_1_2 : cycle.state2.volume = 4 * cycle.initial.volume
  isothermal_2_3 : cycle.state3.temperature = cycle.state2.temperature
  pressure_increase_2_3 : cycle.state3.pressure > cycle.state2.pressure
  compression_3_1 : ∃ (γ : ℝ), cycle.initial.temperature = γ * cycle.initial.volume^2

/-- The theorem to be proved -/
theorem volume_ratio_is_two 
  (cycle : GasCycle) 
  [conditions : CycleConditions cycle] : 
  cycle.state3.volume = 2 * cycle.initial.volume := by
  sorry

end NUMINAMATH_CALUDE_volume_ratio_is_two_l3350_335017


namespace NUMINAMATH_CALUDE_total_shared_amount_l3350_335075

def ken_share : ℕ := 1750
def tony_share : ℕ := 2 * ken_share

theorem total_shared_amount : ken_share + tony_share = 5250 := by
  sorry

end NUMINAMATH_CALUDE_total_shared_amount_l3350_335075


namespace NUMINAMATH_CALUDE_negative_sqrt_of_square_of_negative_three_l3350_335043

theorem negative_sqrt_of_square_of_negative_three :
  -Real.sqrt ((-3)^2) = -3 := by sorry

end NUMINAMATH_CALUDE_negative_sqrt_of_square_of_negative_three_l3350_335043


namespace NUMINAMATH_CALUDE_hotel_outlets_count_l3350_335059

/-- Represents the number of outlets required for different room types and the distribution of outlet types -/
structure HotelOutlets where
  standardRoomOutlets : ℕ
  suiteOutlets : ℕ
  standardRoomCount : ℕ
  suiteCount : ℕ
  typeAPercentage : ℚ
  typeBPercentage : ℚ
  typeCPercentage : ℚ

/-- Calculates the total number of outlets needed for a hotel -/
def totalOutlets (h : HotelOutlets) : ℕ :=
  h.standardRoomCount * h.standardRoomOutlets +
  h.suiteCount * h.suiteOutlets

/-- Theorem stating that the total number of outlets for the given hotel configuration is 650 -/
theorem hotel_outlets_count (h : HotelOutlets)
    (h_standard : h.standardRoomOutlets = 10)
    (h_suite : h.suiteOutlets = 15)
    (h_standard_count : h.standardRoomCount = 50)
    (h_suite_count : h.suiteCount = 10)
    (h_typeA : h.typeAPercentage = 2/5)
    (h_typeB : h.typeBPercentage = 3/5)
    (h_typeC : h.typeCPercentage = 1) :
  totalOutlets h = 650 := by
  sorry

end NUMINAMATH_CALUDE_hotel_outlets_count_l3350_335059


namespace NUMINAMATH_CALUDE_smallest_class_size_l3350_335050

theorem smallest_class_size (total_students : ℕ) 
  (h1 : total_students ≥ 50)
  (h2 : ∃ (x : ℕ), total_students = 4 * x + (x + 2))
  (h3 : ∀ (y : ℕ), y ≥ 50 → (∃ (z : ℕ), y = 4 * z + (z + 2)) → y ≥ total_students) :
  total_students = 52 := by
sorry

end NUMINAMATH_CALUDE_smallest_class_size_l3350_335050


namespace NUMINAMATH_CALUDE_quadratic_condition_l3350_335070

/-- A quadratic equation in x is of the form ax² + bx + c = 0, where a ≠ 0 -/
def is_quadratic_in_x (a b c : ℝ) : Prop := a ≠ 0

/-- The equation ax² - 2x + 3 = 0 -/
def equation (a : ℝ) (x : ℝ) : Prop := a * x^2 - 2*x + 3 = 0

theorem quadratic_condition (a : ℝ) :
  (∃ x, equation a x) ∧ is_quadratic_in_x a (-2) 3 ↔ a ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_condition_l3350_335070


namespace NUMINAMATH_CALUDE_cube_rotation_invariance_l3350_335074

-- Define a cube
structure Cube where
  position : ℕ × ℕ  -- Position on the plane
  topFace : Fin 6   -- Top face (numbered 1 to 6)
  rotation : Fin 4  -- Rotation of top face (0, 90, 180, or 270 degrees)

-- Define a roll operation
def roll (c : Cube) : Cube :=
  sorry

-- Define a sequence of rolls
def rollSequence (c : Cube) (n : ℕ) : Cube :=
  sorry

-- Theorem statement
theorem cube_rotation_invariance (c : Cube) (n : ℕ) :
  let c' := rollSequence c n
  c'.position = c.position ∧ c'.topFace = c.topFace →
  c'.rotation = c.rotation :=
sorry

end NUMINAMATH_CALUDE_cube_rotation_invariance_l3350_335074


namespace NUMINAMATH_CALUDE_smallest_divisor_power_l3350_335039

def Q (z : ℂ) : ℂ := z^10 + z^9 + z^6 + z^5 + z^4 + z + 1

theorem smallest_divisor_power : 
  ∃! k : ℕ, k > 0 ∧ 
  (∀ z : ℂ, Q z = 0 → z^k = 1) ∧
  (∀ m : ℕ, m > 0 → m < k → ∃ z : ℂ, Q z = 0 ∧ z^m ≠ 1) ∧
  k = 84 := by
sorry

end NUMINAMATH_CALUDE_smallest_divisor_power_l3350_335039


namespace NUMINAMATH_CALUDE_closest_to_one_l3350_335079

theorem closest_to_one : 
  let numbers : List ℝ := [3/4, 1.2, 0.81, 4/3, 7/10]
  ∀ x ∈ numbers, |0.81 - 1| ≤ |x - 1| := by
sorry

end NUMINAMATH_CALUDE_closest_to_one_l3350_335079


namespace NUMINAMATH_CALUDE_quadratic_point_order_l3350_335012

-- Define the quadratic function
def f (x : ℝ) : ℝ := x^2 + 2*x + 2

-- Define the points A, B, and C
def A : ℝ × ℝ := (-2, f (-2))
def B : ℝ × ℝ := (-1, f (-1))
def C : ℝ × ℝ := (2, f 2)

-- State the theorem
theorem quadratic_point_order :
  A.2 < B.2 ∧ B.2 < C.2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_point_order_l3350_335012


namespace NUMINAMATH_CALUDE_sum_of_pairwise_products_of_roots_l3350_335089

theorem sum_of_pairwise_products_of_roots (p q r : ℂ) : 
  2 * p^3 - 4 * p^2 + 8 * p - 5 = 0 →
  2 * q^3 - 4 * q^2 + 8 * q - 5 = 0 →
  2 * r^3 - 4 * r^2 + 8 * r - 5 = 0 →
  p * q + q * r + p * r = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_pairwise_products_of_roots_l3350_335089


namespace NUMINAMATH_CALUDE_max_probability_zero_units_digit_l3350_335091

def probability_zero_units_digit (N : ℕ+) : ℚ :=
  let q2 := (N / 2 : ℚ) / N
  let q5 := (N / 5 : ℚ) / N
  let q10 := (N / 10 : ℚ) / N
  q10 * (2 - q10) + 2 * (q2 - q10) * (q5 - q10)

theorem max_probability_zero_units_digit :
  ∀ N : ℕ+, probability_zero_units_digit N ≤ 27/100 := by
  sorry

end NUMINAMATH_CALUDE_max_probability_zero_units_digit_l3350_335091


namespace NUMINAMATH_CALUDE_damaged_tins_percentage_l3350_335026

theorem damaged_tins_percentage (cases : ℕ) (tins_per_case : ℕ) (remaining_tins : ℕ) : 
  cases = 15 → tins_per_case = 24 → remaining_tins = 342 →
  (cases * tins_per_case - remaining_tins) / (cases * tins_per_case) * 100 = 5 := by
  sorry

end NUMINAMATH_CALUDE_damaged_tins_percentage_l3350_335026


namespace NUMINAMATH_CALUDE_book_selection_combinations_l3350_335029

/-- The number of ways to choose one book from each of three genres -/
def book_combinations (mystery_count : ℕ) (fantasy_count : ℕ) (biography_count : ℕ) : ℕ :=
  mystery_count * fantasy_count * biography_count

/-- Theorem: Given 4 mystery novels, 3 fantasy novels, and 3 biographies,
    the number of ways to choose one book from each genre is 36 -/
theorem book_selection_combinations :
  book_combinations 4 3 3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_book_selection_combinations_l3350_335029


namespace NUMINAMATH_CALUDE_shell_difference_l3350_335061

theorem shell_difference (perfect_shells broken_shells non_spiral_perfect : ℕ) 
  (h1 : perfect_shells = 17)
  (h2 : broken_shells = 52)
  (h3 : non_spiral_perfect = 12) : 
  (broken_shells / 2) - (perfect_shells - non_spiral_perfect) = 21 := by
  sorry

end NUMINAMATH_CALUDE_shell_difference_l3350_335061


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l3350_335044

theorem algebraic_expression_value :
  ∀ x : ℝ, x = 2 * Real.sqrt 3 - 1 → x^2 + 2*x - 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l3350_335044


namespace NUMINAMATH_CALUDE_olivers_bags_weight_l3350_335081

theorem olivers_bags_weight (james_bag_weight : ℝ) (oliver_bag_ratio : ℝ) : 
  james_bag_weight = 18 →
  oliver_bag_ratio = 1 / 6 →
  2 * (oliver_bag_ratio * james_bag_weight) = 6 := by
  sorry

end NUMINAMATH_CALUDE_olivers_bags_weight_l3350_335081


namespace NUMINAMATH_CALUDE_arrangements_count_l3350_335038

/-- The number of students -/
def total_students : ℕ := 5

/-- The number of boys -/
def num_boys : ℕ := 2

/-- The number of girls -/
def num_girls : ℕ := 3

/-- A function that calculates the number of arrangements -/
def count_arrangements (n : ℕ) (b : ℕ) (g : ℕ) : ℕ :=
  sorry

/-- The main theorem -/
theorem arrangements_count :
  count_arrangements total_students num_boys num_girls = 48 :=
sorry

end NUMINAMATH_CALUDE_arrangements_count_l3350_335038


namespace NUMINAMATH_CALUDE_tangent_line_y_intercept_l3350_335062

open Real

noncomputable def f (x : ℝ) := exp x + exp (-x)

theorem tangent_line_y_intercept :
  let x₀ : ℝ := log (sqrt 2)
  let f' : ℝ → ℝ := λ x => exp x - exp (-x)
  let m : ℝ := f' x₀
  let b : ℝ := f x₀ - m * x₀
  (∀ x, f' (-x) = -f' x) →  -- f' is an odd function
  (m * (sqrt 2) / 2 = -1) →  -- tangent line is perpendicular to √2x + y + 1 = 0
  b = 3 * sqrt 2 / 2 - sqrt 2 / 4 * log 2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_y_intercept_l3350_335062


namespace NUMINAMATH_CALUDE_equal_discriminants_l3350_335025

theorem equal_discriminants (p1 p2 q1 q2 a1 a2 b1 b2 : ℝ) 
  (hP : ∀ x, x^2 + p1*x + q1 = (x - a1)*(x - a2))
  (hQ : ∀ x, x^2 + p2*x + q2 = (x - b1)*(x - b2))
  (ha : a1 ≠ a2)
  (hb : b1 ≠ b2)
  (h_eq : (b1^2 + p1*b1 + q1) + (b2^2 + p1*b2 + q1) = 
          (a1^2 + p2*a1 + q2) + (a2^2 + p2*a2 + q2)) :
  (a1 - a2)^2 = (b1 - b2)^2 := by
  sorry

end NUMINAMATH_CALUDE_equal_discriminants_l3350_335025


namespace NUMINAMATH_CALUDE_bus_dispatch_theorem_l3350_335065

/-- Represents the bus dispatch problem -/
structure BusDispatchProblem where
  initial_buses : ℕ := 15
  dispatch_interval : ℕ := 6
  entry_interval : ℕ := 8
  entry_delay : ℕ := 3
  total_time : ℕ := 840

/-- Calculates the time when the parking lot is first empty -/
def first_empty_time (problem : BusDispatchProblem) : ℕ :=
  sorry

/-- Calculates the time when buses can no longer be dispatched on time -/
def dispatch_failure_time (problem : BusDispatchProblem) : ℕ :=
  sorry

/-- Calculates the delay for the first bus that can't be dispatched on time -/
def first_delay_time (problem : BusDispatchProblem) : ℕ :=
  sorry

/-- Calculates the minimum interval for continuous dispatching -/
def min_continuous_interval (problem : BusDispatchProblem) : ℕ :=
  sorry

/-- Calculates the minimum number of additional buses needed for 6-minute interval dispatching -/
def min_additional_buses (problem : BusDispatchProblem) : ℕ :=
  sorry

theorem bus_dispatch_theorem (problem : BusDispatchProblem) :
  first_empty_time problem = 330 ∧
  dispatch_failure_time problem = 354 ∧
  first_delay_time problem = 1 ∧
  min_continuous_interval problem = 8 ∧
  min_additional_buses problem = 22 := by
  sorry

end NUMINAMATH_CALUDE_bus_dispatch_theorem_l3350_335065


namespace NUMINAMATH_CALUDE_installment_payment_installment_payment_proof_l3350_335073

theorem installment_payment (cash_price : ℕ) (down_payment : ℕ) (first_four : ℕ) 
  (last_four : ℕ) (installment_difference : ℕ) : ℕ :=
  let total_installment := cash_price + installment_difference
  let first_four_total := 4 * first_four
  let last_four_total := 4 * last_four
  let middle_four_total := total_installment - down_payment - first_four_total - last_four_total
  let middle_four_monthly := middle_four_total / 4
  middle_four_monthly

#check @installment_payment

theorem installment_payment_proof 
  (h1 : installment_payment 450 100 40 30 70 = 35) : True := by
  sorry

end NUMINAMATH_CALUDE_installment_payment_installment_payment_proof_l3350_335073


namespace NUMINAMATH_CALUDE_city_fuel_efficiency_l3350_335097

/-- Represents the fuel efficiency of a car in miles per gallon -/
structure FuelEfficiency where
  highway : ℝ
  city : ℝ

/-- Represents the distance a car can travel on a full tank in miles -/
structure TankDistance where
  highway : ℝ
  city : ℝ

theorem city_fuel_efficiency 
  (fe : FuelEfficiency) 
  (td : TankDistance) 
  (h1 : fe.city = fe.highway - 9)
  (h2 : td.highway = 462)
  (h3 : td.city = 336)
  (h4 : fe.highway * (td.city / fe.city) = td.highway) :
  fe.city = 24 := by
  sorry

end NUMINAMATH_CALUDE_city_fuel_efficiency_l3350_335097


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l3350_335085

/-- Given that α and β are the roots of x^2 + x - 1 = 0, prove that 2α^5 + β^3 = -13 ± 4√5 -/
theorem quadratic_roots_sum (α β : ℝ) : 
  α^2 + α - 1 = 0 → β^2 + β - 1 = 0 → 
  2 * α^5 + β^3 = -13 + 4 * Real.sqrt 5 ∨ 2 * α^5 + β^3 = -13 - 4 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l3350_335085


namespace NUMINAMATH_CALUDE_coffee_mix_price_l3350_335003

/-- The price of the first kind of coffee in dollars per pound -/
def price_first : ℚ := 215 / 100

/-- The price of the mixed coffee in dollars per pound -/
def price_mix : ℚ := 230 / 100

/-- The total weight of the mixed coffee in pounds -/
def total_weight : ℚ := 18

/-- The weight of each kind of coffee in the mix in pounds -/
def weight_each : ℚ := 9

/-- The price of the second kind of coffee in dollars per pound -/
def price_second : ℚ := 245 / 100

theorem coffee_mix_price :
  price_second = 
    (price_mix * total_weight - price_first * weight_each) / weight_each :=
by sorry

end NUMINAMATH_CALUDE_coffee_mix_price_l3350_335003


namespace NUMINAMATH_CALUDE_blake_receives_four_dollars_change_l3350_335006

/-- The amount of change Blake receives after purchasing lollipops and chocolate. -/
def blakes_change (lollipop_count : ℕ) (chocolate_pack_count : ℕ) (lollipop_price : ℕ) (bill_count : ℕ) (bill_value : ℕ) : ℕ :=
  let chocolate_pack_price := 4 * lollipop_price
  let total_cost := lollipop_count * lollipop_price + chocolate_pack_count * chocolate_pack_price
  let payment := bill_count * bill_value
  payment - total_cost

/-- Theorem stating that Blake's change is $4 given the problem conditions. -/
theorem blake_receives_four_dollars_change :
  blakes_change 4 6 2 6 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_blake_receives_four_dollars_change_l3350_335006


namespace NUMINAMATH_CALUDE_summer_reading_challenge_l3350_335045

def books_to_coupons (books : ℕ) : ℕ := books / 5

def quinn_books : ℕ := 5 * 5

def taylor_books : ℕ := 1 + 4 * 9

def jordan_books : ℕ := 3 * 10

theorem summer_reading_challenge : 
  books_to_coupons quinn_books + books_to_coupons taylor_books + books_to_coupons jordan_books = 18 := by
  sorry

end NUMINAMATH_CALUDE_summer_reading_challenge_l3350_335045


namespace NUMINAMATH_CALUDE_tan_sum_special_angle_l3350_335032

theorem tan_sum_special_angle (θ : Real) (h : Real.tan θ = 1/3) :
  Real.tan (θ + π/4) = 2 := by sorry

end NUMINAMATH_CALUDE_tan_sum_special_angle_l3350_335032


namespace NUMINAMATH_CALUDE_dagger_example_l3350_335098

-- Define the † operation
def dagger (m n p q : ℕ) (hm : m ≠ 0) : ℚ :=
  (m^2 * p * (q / n : ℚ)) + (p / m : ℚ)

-- Theorem statement
theorem dagger_example : dagger 5 9 6 2 (by norm_num) = 518 / 15 := by
  sorry

end NUMINAMATH_CALUDE_dagger_example_l3350_335098


namespace NUMINAMATH_CALUDE_mojave_population_increase_l3350_335072

/-- Calculates the percentage increase between two populations -/
def percentageIncrease (initial : ℕ) (final : ℕ) : ℚ :=
  (final - initial : ℚ) / initial * 100

theorem mojave_population_increase : 
  let initialPopulation : ℕ := 4000
  let currentPopulation : ℕ := initialPopulation * 3
  let futurePopulation : ℕ := 16800
  percentageIncrease currentPopulation futurePopulation = 40 := by
sorry

end NUMINAMATH_CALUDE_mojave_population_increase_l3350_335072


namespace NUMINAMATH_CALUDE_x_value_from_fraction_equality_l3350_335099

theorem x_value_from_fraction_equality (x y : ℝ) :
  x ≠ 2 →
  x / (x - 2) = (y^2 + 3*y - 4) / (y^2 + 3*y - 5) →
  x = 2*y^2 + 6*y - 8 := by
sorry

end NUMINAMATH_CALUDE_x_value_from_fraction_equality_l3350_335099


namespace NUMINAMATH_CALUDE_parking_spaces_available_l3350_335058

theorem parking_spaces_available (front_spaces back_spaces total_parked : ℕ) 
  (h1 : front_spaces = 52)
  (h2 : back_spaces = 38)
  (h3 : total_parked = 39)
  (h4 : total_parked = front_spaces + back_spaces / 2) : 
  front_spaces + back_spaces - total_parked = 51 := by
  sorry

end NUMINAMATH_CALUDE_parking_spaces_available_l3350_335058


namespace NUMINAMATH_CALUDE_comparison_inequality_range_of_linear_combination_l3350_335042

-- Part 1
theorem comparison_inequality (x y z : ℝ) : 
  5 * x^2 + y^2 + z^2 ≥ 2 * x * y + 4 * x + 2 * z - 2 := by sorry

-- Part 2
theorem range_of_linear_combination (a b : ℝ) 
  (h1 : 1 ≤ 2 * a + b) (h2 : 2 * a + b ≤ 4) 
  (h3 : -1 ≤ a - 2 * b) (h4 : a - 2 * b ≤ 2) : 
  -1 ≤ 10 * a - 5 * b ∧ 10 * a - 5 * b ≤ 20 := by sorry

end NUMINAMATH_CALUDE_comparison_inequality_range_of_linear_combination_l3350_335042


namespace NUMINAMATH_CALUDE_factor_implies_q_value_l3350_335036

theorem factor_implies_q_value (q : ℚ) :
  (∀ m : ℚ, (m - 8) ∣ (m^2 - q*m - 24)) → q = 5 := by
  sorry

end NUMINAMATH_CALUDE_factor_implies_q_value_l3350_335036


namespace NUMINAMATH_CALUDE_farm_feet_count_l3350_335020

/-- Given a farm with hens and cows, calculates the total number of feet -/
def total_feet (total_heads : ℕ) (num_hens : ℕ) : ℕ :=
  let num_cows := total_heads - num_hens
  let hen_feet := num_hens * 2
  let cow_feet := num_cows * 4
  hen_feet + cow_feet

/-- Theorem: In a farm with 46 total heads and 24 hens, there are 136 feet in total -/
theorem farm_feet_count : total_feet 46 24 = 136 := by
  sorry

end NUMINAMATH_CALUDE_farm_feet_count_l3350_335020


namespace NUMINAMATH_CALUDE_tank_leak_emptying_time_l3350_335069

/-- Given a tank that can be filled in 7 hours without a leak and 8 hours with a leak,
    prove that it takes 56 hours for the tank to become empty due to the leak. -/
theorem tank_leak_emptying_time :
  ∀ (fill_rate_no_leak fill_rate_with_leak leak_rate : ℚ),
    fill_rate_no_leak = 1 / 7 →
    fill_rate_with_leak = 1 / 8 →
    fill_rate_with_leak = fill_rate_no_leak - leak_rate →
    (1 : ℚ) / leak_rate = 56 := by
  sorry

end NUMINAMATH_CALUDE_tank_leak_emptying_time_l3350_335069


namespace NUMINAMATH_CALUDE_total_brass_l3350_335064

def brass_composition (copper zinc : ℝ) : Prop :=
  copper / zinc = 13 / 7

theorem total_brass (zinc : ℝ) (h : zinc = 35) :
  ∃ total : ℝ, brass_composition (total - zinc) zinc ∧ total = 100 :=
sorry

end NUMINAMATH_CALUDE_total_brass_l3350_335064


namespace NUMINAMATH_CALUDE_group_interval_equals_frequency_over_height_l3350_335031

/-- Given a group [a, b] in a sampling process with frequency m and histogram height h, 
    prove that the group interval |a-b| equals m/h -/
theorem group_interval_equals_frequency_over_height 
  (a b m h : ℝ) (hm : m > 0) (hh : h > 0) : |a - b| = m / h := by
  sorry

end NUMINAMATH_CALUDE_group_interval_equals_frequency_over_height_l3350_335031


namespace NUMINAMATH_CALUDE_airplane_seats_total_l3350_335066

/-- Represents the number of seats in an airplane -/
def AirplaneSeats (total : ℝ) : Prop :=
  let first_class : ℝ := 36
  let business_class : ℝ := 0.3 * total
  let economy : ℝ := 0.6 * total
  let premium_economy : ℝ := total - first_class - business_class - economy
  (first_class + business_class + economy + premium_economy = total) ∧
  (premium_economy ≥ 0)

/-- The total number of seats in the airplane is 360 -/
theorem airplane_seats_total : ∃ (total : ℝ), AirplaneSeats total ∧ total = 360 := by
  sorry

end NUMINAMATH_CALUDE_airplane_seats_total_l3350_335066


namespace NUMINAMATH_CALUDE_like_terms_characterization_l3350_335010

/-- Represents a term in an algebraic expression -/
structure Term where
  letters : List Char
  exponents : List Nat
  deriving Repr

/-- Defines when two terms are considered like terms -/
def like_terms (t1 t2 : Term) : Prop :=
  t1.letters = t2.letters ∧ t1.exponents = t2.exponents

theorem like_terms_characterization (t1 t2 : Term) :
  like_terms t1 t2 ↔ t1.letters = t2.letters ∧ t1.exponents = t2.exponents :=
by sorry

end NUMINAMATH_CALUDE_like_terms_characterization_l3350_335010


namespace NUMINAMATH_CALUDE_remainder_101_power_50_mod_100_l3350_335002

theorem remainder_101_power_50_mod_100 : 101^50 % 100 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_101_power_50_mod_100_l3350_335002


namespace NUMINAMATH_CALUDE_profit_maximized_at_150_l3350_335088

/-- The profit function for a company -/
def profit_function (a : ℝ) (x : ℝ) : ℝ := -a * x^2 + 7500 * x

/-- The derivative of the profit function -/
def profit_derivative (a : ℝ) (x : ℝ) : ℝ := -2 * a * x + 7500

theorem profit_maximized_at_150 (a : ℝ) :
  (profit_derivative a 150 = 0) → (a = 25) :=
by sorry

#check profit_maximized_at_150

end NUMINAMATH_CALUDE_profit_maximized_at_150_l3350_335088


namespace NUMINAMATH_CALUDE_three_prime_divisors_special_form_l3350_335093

theorem three_prime_divisors_special_form (n : ℕ) (x : ℕ) : 
  x = 2^n - 32 →
  (∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ p ≠ 2 ∧ q ≠ 2 ∧
    (∀ r : ℕ, Nat.Prime r → r ∣ x → r = 2 ∨ r = p ∨ r = q)) →
  x = 2016 ∨ x = 16352 := by
sorry

end NUMINAMATH_CALUDE_three_prime_divisors_special_form_l3350_335093


namespace NUMINAMATH_CALUDE_pirate_treasure_division_l3350_335014

theorem pirate_treasure_division (N : ℕ) : 
  220 ≤ N ∧ N ≤ 300 →
  let first_take := 2 + (N - 2) / 3
  let remain_after_first := N - first_take
  let second_take := 2 + (remain_after_first - 2) / 3
  let remain_after_second := remain_after_first - second_take
  let third_take := 2 + (remain_after_second - 2) / 3
  let final_remain := remain_after_second - third_take
  final_remain % 3 = 0 →
  first_take = 84 ∧ 
  second_take = 54 ∧ 
  third_take = 54 ∧
  final_remain / 3 = 54 := by
sorry


end NUMINAMATH_CALUDE_pirate_treasure_division_l3350_335014


namespace NUMINAMATH_CALUDE_twenty_seven_in_base_two_l3350_335055

theorem twenty_seven_in_base_two : 
  27 = 1 * 2^4 + 1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0 := by
  sorry

end NUMINAMATH_CALUDE_twenty_seven_in_base_two_l3350_335055


namespace NUMINAMATH_CALUDE_total_decorations_handed_out_l3350_335037

/-- Represents the contents of a decoration box -/
structure DecorationBox where
  tinsel : Nat
  tree : Nat
  snowGlobes : Nat

/-- Calculates the total number of decorations in a box -/
def totalDecorationsPerBox (box : DecorationBox) : Nat :=
  box.tinsel + box.tree + box.snowGlobes

/-- Theorem: The total number of decorations handed out is 120 -/
theorem total_decorations_handed_out :
  let standardBox : DecorationBox := { tinsel := 4, tree := 1, snowGlobes := 5 }
  let familyBoxes : Nat := 11
  let communityBoxes : Nat := 1
  totalDecorationsPerBox standardBox * (familyBoxes + communityBoxes) = 120 := by
  sorry

end NUMINAMATH_CALUDE_total_decorations_handed_out_l3350_335037


namespace NUMINAMATH_CALUDE_negation_of_proposition_negation_of_exponential_inequality_l3350_335084

theorem negation_of_proposition (p : ℝ → Prop) :
  (¬ ∀ x : ℝ, p x) ↔ (∃ x : ℝ, ¬ p x) := by sorry

theorem negation_of_exponential_inequality :
  (¬ ∀ x : ℝ, Real.exp x ≥ 1) ↔ (∃ x : ℝ, Real.exp x < 1) := by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_negation_of_exponential_inequality_l3350_335084


namespace NUMINAMATH_CALUDE_gcd_of_special_powers_l3350_335063

theorem gcd_of_special_powers :
  Nat.gcd (2^2020 - 1) (2^2000 - 1) = 2^20 - 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_special_powers_l3350_335063


namespace NUMINAMATH_CALUDE_star_properties_l3350_335094

/-- Custom multiplication operation -/
def star (a b : ℝ) : ℝ := (a + b)^2

/-- Theorem stating that exactly two of the given properties hold for the star operation -/
theorem star_properties :
  (∃! n : ℕ, n = 2 ∧ 
    (((∀ a b : ℝ, star a b = 0 → a = 0 ∧ b = 0) → n ≥ 1) ∧
     ((∀ a b : ℝ, star a b = star b a) → n ≥ 1) ∧
     ((∀ a b c : ℝ, star a (b + c) = star a b + star a c) → n ≥ 1) ∧
     ((∀ a b : ℝ, star a b = star (-a) (-b)) → n ≥ 1)) ∧
    n ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_star_properties_l3350_335094


namespace NUMINAMATH_CALUDE_circle_translation_l3350_335057

-- Define the original circle
def original_circle (x y : ℝ) : Prop := x^2 + y^2 = 16

-- Define the translation
def translation : ℝ × ℝ := (-5, -3)

-- Define the translated circle
def translated_circle (x y : ℝ) : Prop := (x+5)^2 + (y+3)^2 = 16

-- Theorem statement
theorem circle_translation :
  ∀ (x y : ℝ), original_circle (x + 5) (y + 3) ↔ translated_circle x y :=
by sorry

end NUMINAMATH_CALUDE_circle_translation_l3350_335057


namespace NUMINAMATH_CALUDE_freddys_age_l3350_335019

theorem freddys_age (job_age stephanie_age freddy_age : ℕ) : 
  job_age = 5 →
  stephanie_age = 4 * job_age →
  freddy_age = stephanie_age - 2 →
  freddy_age = 18 := by
  sorry

end NUMINAMATH_CALUDE_freddys_age_l3350_335019


namespace NUMINAMATH_CALUDE_mingis_test_pages_l3350_335028

/-- The number of pages in Mingi's math test -/
def pages_in_test (first_page last_page : ℕ) : ℕ :=
  last_page - first_page + 1

/-- Theorem stating the number of pages in Mingi's math test -/
theorem mingis_test_pages : pages_in_test 8 21 = 14 := by
  sorry

end NUMINAMATH_CALUDE_mingis_test_pages_l3350_335028


namespace NUMINAMATH_CALUDE_equality_condition_l3350_335033

theorem equality_condition (a b c : ℝ) : 
  a + 2*b*c = (a + 2*b)*(a + 2*c) ↔ a + 2*b + 2*c = 0 := by
  sorry

end NUMINAMATH_CALUDE_equality_condition_l3350_335033


namespace NUMINAMATH_CALUDE_sufficient_condition_range_l3350_335030

theorem sufficient_condition_range (a : ℝ) : 
  (∀ x : ℝ, x = 1 → x > a) → a < 1 := by sorry

end NUMINAMATH_CALUDE_sufficient_condition_range_l3350_335030


namespace NUMINAMATH_CALUDE_tan_inequality_equiv_l3350_335016

theorem tan_inequality_equiv (x : ℝ) : 
  Real.tan (2 * x - π / 4) ≤ 1 ↔ 
  ∃ k : ℤ, k * π / 2 - π / 8 < x ∧ x ≤ k * π / 2 + π / 4 := by
sorry

end NUMINAMATH_CALUDE_tan_inequality_equiv_l3350_335016


namespace NUMINAMATH_CALUDE_no_linear_term_implies_n_eq_neg_two_l3350_335041

theorem no_linear_term_implies_n_eq_neg_two (n : ℝ) : 
  (∀ x : ℝ, ∃ a b : ℝ, (x + n) * (x + 2) = a * x^2 + b) → n = -2 := by
  sorry

end NUMINAMATH_CALUDE_no_linear_term_implies_n_eq_neg_two_l3350_335041


namespace NUMINAMATH_CALUDE_bisection_method_for_f_l3350_335056

/-- The function f(x) = x^5 + 8x^3 - 1 -/
def f (x : ℝ) : ℝ := x^5 + 8*x^3 - 1

/-- Theorem stating the properties of the bisection method for f(x) -/
theorem bisection_method_for_f :
  f 0 < 0 →
  f 0.5 > 0 →
  ∃ (a b : ℝ), a = 0 ∧ b = 0.5 ∧
    (∃ (x : ℝ), a < x ∧ x < b ∧ f x = 0) ∧
    ((a + b) / 2 = 0.25) :=
sorry

end NUMINAMATH_CALUDE_bisection_method_for_f_l3350_335056


namespace NUMINAMATH_CALUDE_percentage_difference_l3350_335009

theorem percentage_difference : (70 : ℝ) / 100 * 100 - (60 : ℝ) / 100 * 80 = 22 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l3350_335009


namespace NUMINAMATH_CALUDE_field_area_diminished_l3350_335051

theorem field_area_diminished (L W : ℝ) (h : L > 0 ∧ W > 0) :
  let original_area := L * W
  let new_length := L * (1 - 0.4)
  let new_width := W * (1 - 0.4)
  let new_area := new_length * new_width
  (original_area - new_area) / original_area = 0.64 := by sorry

end NUMINAMATH_CALUDE_field_area_diminished_l3350_335051


namespace NUMINAMATH_CALUDE_inverse_proportionality_example_l3350_335027

/-- Definition of inverse proportionality -/
def is_inversely_proportional (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x : ℝ, x ≠ 0 → f x = k / x

/-- The function y = 6/x is inversely proportional -/
theorem inverse_proportionality_example :
  is_inversely_proportional (λ x : ℝ => 6 / x) :=
by
  sorry


end NUMINAMATH_CALUDE_inverse_proportionality_example_l3350_335027


namespace NUMINAMATH_CALUDE_lcm_hcf_problem_l3350_335046

theorem lcm_hcf_problem (a b : ℕ+) (h1 : Nat.lcm a b = 25974) (h2 : Nat.gcd a b = 107) (h3 : a = 4951) : b = 561 := by
  sorry

end NUMINAMATH_CALUDE_lcm_hcf_problem_l3350_335046


namespace NUMINAMATH_CALUDE_triangle_problem_l3350_335024

theorem triangle_problem (A B C : Real) (a b c : Real) (D : Real × Real) :
  -- Triangle ABC exists
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  -- Sides a, b, c are positive
  a > 0 ∧ b > 0 ∧ c > 0 →
  -- Given equation
  b * Real.sin (C + π/3) - c * Real.sin B = 0 →
  -- Area condition
  1/2 * a * b * Real.sin C = 10 * Real.sqrt 3 →
  -- D is midpoint of AC
  D.1 = (0 + a * Real.cos B) / 2 ∧ D.2 = (0 + a * Real.sin B) / 2 →
  -- Prove:
  C = π/3 ∧ 
  (∀ (BD : Real), BD^2 ≥ a^2 + b^2/4 - a*b*Real.cos C → BD ≥ 2 * Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l3350_335024


namespace NUMINAMATH_CALUDE_max_value_operation_l3350_335092

theorem max_value_operation (n : ℕ) (h : 100 ≤ n ∧ n ≤ 999) :
  (∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 → (300 - m)^2 - 10 ≤ (300 - n)^2 - 10) →
  (300 - n)^2 - 10 = 39990 :=
by sorry

end NUMINAMATH_CALUDE_max_value_operation_l3350_335092


namespace NUMINAMATH_CALUDE_inequality_proof_l3350_335008

theorem inequality_proof (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_xz : x * z = 1)
  (h_x_1z : x * (1 + z) > 1)
  (h_y_1x : y * (1 + x) > 1)
  (h_z_1y : z * (1 + y) > 1) :
  2 * (x + y + z) ≥ -1/x + 1/y + 1/z + 3 := by
sorry


end NUMINAMATH_CALUDE_inequality_proof_l3350_335008


namespace NUMINAMATH_CALUDE_book_sales_calculation_l3350_335067

/-- Calculates the total book sales over three days given specific sales patterns. -/
theorem book_sales_calculation (day1_sales : ℕ) : 
  day1_sales = 15 →
  (day1_sales + 3 * day1_sales + (3 * day1_sales) / 5 : ℕ) = 69 := by
  sorry

end NUMINAMATH_CALUDE_book_sales_calculation_l3350_335067


namespace NUMINAMATH_CALUDE_digit_difference_digit_difference_proof_l3350_335068

theorem digit_difference : ℕ → Prop :=
  fun n =>
    (∀ m : ℕ, m < 1000 → m < n) ∧
    (n < 10000) ∧
    (∀ k : ℕ, k < 1000) →
    n - 999 = 1

-- The proof
theorem digit_difference_proof : digit_difference 1000 := by
  sorry

end NUMINAMATH_CALUDE_digit_difference_digit_difference_proof_l3350_335068


namespace NUMINAMATH_CALUDE_set_equality_l3350_335080

def U : Set ℕ := Set.univ

def A : Set ℕ := {x | ∃ n : ℕ, x = 2 * n}

def B : Set ℕ := {x | ∃ n : ℕ, x = 4 * n}

theorem set_equality : U = A ∪ (U \ B) := by sorry

end NUMINAMATH_CALUDE_set_equality_l3350_335080


namespace NUMINAMATH_CALUDE_archipelago_islands_l3350_335004

theorem archipelago_islands (n : ℕ) : 
  (n * (n - 1)) / 2 + n = 28 →
  n + 1 = 8 :=
by
  sorry

#check archipelago_islands

end NUMINAMATH_CALUDE_archipelago_islands_l3350_335004


namespace NUMINAMATH_CALUDE_determinant_equals_negative_two_l3350_335078

-- Define the polynomial and its roots
def polynomial (p q : ℝ) (x : ℝ) : ℝ := x^3 - 3*p*x^2 + q*x - 2

-- Define the roots of the polynomial
def roots (p q : ℝ) : Set ℝ := {x | polynomial p q x = 0}

-- Assume the polynomial has exactly three roots
axiom three_roots (p q : ℝ) : ∃ (a b c : ℝ), roots p q = {a, b, c}

-- Define the determinant
def determinant (r a b c : ℝ) : ℝ :=
  (r + a) * ((r + b) * (r + c) - r^2) -
  r * (r * (r + c) - r^2) +
  r * (r * (r + b) - r^2)

-- State the theorem
theorem determinant_equals_negative_two (p q r : ℝ) :
  ∃ (a b c : ℝ), roots p q = {a, b, c} ∧ determinant r a b c = -2 := by
  sorry

end NUMINAMATH_CALUDE_determinant_equals_negative_two_l3350_335078


namespace NUMINAMATH_CALUDE_mean_steps_per_day_l3350_335049

theorem mean_steps_per_day (total_steps : ℕ) (num_days : ℕ) (h1 : total_steps = 243000) (h2 : num_days = 30) :
  total_steps / num_days = 8100 := by
  sorry

end NUMINAMATH_CALUDE_mean_steps_per_day_l3350_335049


namespace NUMINAMATH_CALUDE_shaltaev_boltaev_inequality_l3350_335047

theorem shaltaev_boltaev_inequality (S B : ℝ) 
  (h1 : S > 0) (h2 : B > 0) 
  (h3 : 175 * S > 125 * B) (h4 : 175 * S < 126 * B) : 
  3 * S + B > S := by
sorry

end NUMINAMATH_CALUDE_shaltaev_boltaev_inequality_l3350_335047


namespace NUMINAMATH_CALUDE_min_value_a_l3350_335021

theorem min_value_a (a x y : ℤ) (h1 : x - y^2 = a) (h2 : y - x^2 = a) (h3 : x ≠ y) (h4 : |x| ≤ 10) :
  ∃ (a_min : ℤ), a ≥ a_min ∧ a_min = -111 :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_l3350_335021


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3350_335076

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →  -- Ensure positive side lengths
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  a^2 + b^2 + c^2 = 980 →  -- Given condition
  c = 70 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3350_335076


namespace NUMINAMATH_CALUDE_smallest_factorizable_b_l3350_335035

theorem smallest_factorizable_b : ∃ (b : ℕ), 
  (∀ (x : ℤ), ∃ (p q : ℤ), x^2 + b*x + 2520 = (x + p) * (x + q)) ∧
  (∀ (b' : ℕ), b' < b → ¬∃ (p q : ℤ), ∀ (x : ℤ), x^2 + b'*x + 2520 = (x + p) * (x + q)) ∧
  b = 106 :=
sorry

end NUMINAMATH_CALUDE_smallest_factorizable_b_l3350_335035


namespace NUMINAMATH_CALUDE_dividend_calculation_l3350_335007

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 17)
  (h2 : quotient = 9)
  (h3 : remainder = 10) :
  divisor * quotient + remainder = 163 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l3350_335007


namespace NUMINAMATH_CALUDE_parallelogram_reflection_l3350_335040

-- Define the basic structures
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the parallelogram
structure Parallelogram where
  A : Point
  B : Point
  C : Point
  D : Point

-- Define the perpendicular line
def perpendicular_line (p : Parallelogram) (t : Line) : Prop :=
  -- Assuming some condition for perpendicularity
  sorry

-- Define the intersection points
def intersection_point (l1 l2 : Line) : Point :=
  -- Assuming some method to find intersection
  sorry

-- Define the reflection operation
def reflect_point (p : Point) (t : Line) : Point :=
  -- Assuming some method to reflect a point over a line
  sorry

-- The main theorem
theorem parallelogram_reflection 
  (p : Parallelogram) 
  (t : Line) 
  (h_perp : perpendicular_line p t) :
  ∃ (p' : Parallelogram),
    let K := intersection_point (Line.mk 0 1 0) t  -- Assuming AB is on y-axis for simplicity
    let L := intersection_point (Line.mk 0 1 0) t  -- Assuming CD is parallel to AB
    p'.A = reflect_point p.A t ∧
    p'.B = reflect_point p.B t ∧
    p'.C = reflect_point p.C t ∧
    p'.D = reflect_point p.D t ∧
    p'.A = Point.mk (2 * K.x - p.A.x) (2 * K.y - p.A.y) ∧
    p'.B = Point.mk (2 * K.x - p.B.x) (2 * K.y - p.B.y) ∧
    p'.C = Point.mk (2 * L.x - p.C.x) (2 * L.y - p.C.y) ∧
    p'.D = Point.mk (2 * L.x - p.D.x) (2 * L.y - p.D.y) :=
  by sorry

end NUMINAMATH_CALUDE_parallelogram_reflection_l3350_335040


namespace NUMINAMATH_CALUDE_division_problem_l3350_335005

theorem division_problem (L S Q : ℕ) : 
  L - S = 1365 → 
  L = 1636 → 
  L = Q * S + 10 → 
  Q = 6 := by sorry

end NUMINAMATH_CALUDE_division_problem_l3350_335005


namespace NUMINAMATH_CALUDE_erica_ride_percentage_longer_l3350_335022

-- Define the ride times for Dave, Chuck, and Erica
def dave_ride_time : ℕ := 10
def chuck_ride_time : ℕ := 5 * dave_ride_time
def erica_ride_time : ℕ := 65

-- Define the percentage difference
def percentage_difference : ℚ := (erica_ride_time - chuck_ride_time : ℚ) / chuck_ride_time * 100

-- Theorem statement
theorem erica_ride_percentage_longer :
  percentage_difference = 30 := by sorry

end NUMINAMATH_CALUDE_erica_ride_percentage_longer_l3350_335022


namespace NUMINAMATH_CALUDE_special_number_exists_l3350_335082

def digit_product (n : ℕ) : ℕ := sorry

def digit_sum (n : ℕ) : ℕ := sorry

def is_cube (n : ℕ) : Prop := ∃ k : ℕ, n = k^3

theorem special_number_exists : ∃ x : ℕ, 
  (digit_product x = 44 * x - 86868) ∧ 
  (is_cube (digit_sum x)) := by sorry

end NUMINAMATH_CALUDE_special_number_exists_l3350_335082


namespace NUMINAMATH_CALUDE_kenny_lawn_mowing_l3350_335013

theorem kenny_lawn_mowing (cost_per_lawn : ℕ) (cost_per_game : ℕ) (cost_per_book : ℕ)
  (num_games : ℕ) (num_books : ℕ) 
  (h1 : cost_per_lawn = 15)
  (h2 : cost_per_game = 45)
  (h3 : cost_per_book = 5)
  (h4 : num_games = 5)
  (h5 : num_books = 60) :
  cost_per_lawn * 35 = cost_per_game * num_games + cost_per_book * num_books :=
by sorry

end NUMINAMATH_CALUDE_kenny_lawn_mowing_l3350_335013


namespace NUMINAMATH_CALUDE_fifteenth_student_age_l3350_335018

theorem fifteenth_student_age 
  (total_students : ℕ) 
  (avg_age_all : ℝ) 
  (num_group1 : ℕ) 
  (avg_age_group1 : ℝ) 
  (num_group2 : ℕ) 
  (avg_age_group2 : ℝ) 
  (h1 : total_students = 15)
  (h2 : avg_age_all = 15)
  (h3 : num_group1 = 5)
  (h4 : avg_age_group1 = 13)
  (h5 : num_group2 = 9)
  (h6 : avg_age_group2 = 16)
  (h7 : num_group1 + num_group2 + 1 = total_students) :
  (total_students : ℝ) * avg_age_all - 
  ((num_group1 : ℝ) * avg_age_group1 + (num_group2 : ℝ) * avg_age_group2) = 16 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_student_age_l3350_335018


namespace NUMINAMATH_CALUDE_difference_divisible_by_18_l3350_335053

theorem difference_divisible_by_18 (a b : ℤ) : 
  18 ∣ ((3*a + 2)^2 - (3*b + 2)^2) := by sorry

end NUMINAMATH_CALUDE_difference_divisible_by_18_l3350_335053


namespace NUMINAMATH_CALUDE_line_l_passes_through_M_line_l1_properties_l3350_335011

-- Define the line l
def line_l (m : ℝ) (x y : ℝ) : Prop :=
  (2 + m) * x + (1 - 2 * m) * y + 4 - 3 * m = 0

-- Define the point M
def point_M : ℝ × ℝ := (-1, -2)

-- Define the line l1
def line_l1 (x y : ℝ) : Prop :=
  2 * x + y + 4 = 0

-- Theorem 1: Line l passes through point M for all real m
theorem line_l_passes_through_M :
  ∀ m : ℝ, line_l m (point_M.1) (point_M.2) := by sorry

-- Theorem 2: Line l1 passes through point M and is bisected by M
theorem line_l1_properties :
  line_l1 (point_M.1) (point_M.2) ∧
  ∃ (A B : ℝ × ℝ),
    (A.1 = 0 ∨ A.2 = 0) ∧
    (B.1 = 0 ∨ B.2 = 0) ∧
    line_l1 A.1 A.2 ∧
    line_l1 B.1 B.2 ∧
    ((A.1 + B.1) / 2 = point_M.1 ∧ (A.2 + B.2) / 2 = point_M.2) := by sorry

end NUMINAMATH_CALUDE_line_l_passes_through_M_line_l1_properties_l3350_335011


namespace NUMINAMATH_CALUDE_percentage_of_l3350_335048

theorem percentage_of (a b : ℝ) (h : b ≠ 0) :
  (a / b) * 100 = 250 → a = 150 ∧ b = 60 :=
by sorry

end NUMINAMATH_CALUDE_percentage_of_l3350_335048


namespace NUMINAMATH_CALUDE_max_value_theorem_range_of_a_l3350_335000

-- Define the constraint function
def constraint (x y z : ℝ) : Prop := x^2 + y^2 + z^2 = 1

-- Define the objective function
def objective (x y z : ℝ) : ℝ := x + 2*y + 2*z

-- Theorem 1: Maximum value of the objective function
theorem max_value_theorem (x y z : ℝ) (h_pos : x > 0 ∧ y > 0 ∧ z > 0) (h_constraint : constraint x y z) :
  objective x y z ≤ 3 :=
sorry

-- Theorem 2: Range of a
theorem range_of_a (a : ℝ) :
  (∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → constraint x y z → |a - 3| ≥ objective x y z) ↔
  a ≤ 0 ∨ a ≥ 6 :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_range_of_a_l3350_335000


namespace NUMINAMATH_CALUDE_pond_algae_free_day_24_l3350_335083

/-- Represents the coverage of algae in the pond on a given day -/
def algae_coverage (day : ℕ) : ℝ := sorry

/-- The algae coverage triples every two days -/
axiom triple_every_two_days (d : ℕ) : algae_coverage (d + 2) = 3 * algae_coverage d

/-- The pond is completely covered on day 28 -/
axiom full_coverage_day_28 : algae_coverage 28 = 1

/-- Theorem: The pond is 88.89% algae-free on day 24 -/
theorem pond_algae_free_day_24 : algae_coverage 24 = 1 - 0.8889 := by sorry

end NUMINAMATH_CALUDE_pond_algae_free_day_24_l3350_335083


namespace NUMINAMATH_CALUDE_stage_20_toothpicks_l3350_335086

/-- Calculates the number of toothpicks in a given stage of the pattern -/
def toothpicks (stage : ℕ) : ℕ :=
  3 + 3 * (stage - 1)

/-- Theorem: The 20th stage of the pattern has 60 toothpicks -/
theorem stage_20_toothpicks : toothpicks 20 = 60 := by
  sorry

end NUMINAMATH_CALUDE_stage_20_toothpicks_l3350_335086
