import Mathlib

namespace NUMINAMATH_CALUDE_minimum_reciprocal_sum_l2492_249215

theorem minimum_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 20) :
  (1 / x + 1 / y) ≥ 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_minimum_reciprocal_sum_l2492_249215


namespace NUMINAMATH_CALUDE_factorization_identity_l2492_249219

theorem factorization_identity (x y a : ℝ) : x * (a - y) - y * (y - a) = (x + y) * (a - y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_identity_l2492_249219


namespace NUMINAMATH_CALUDE_quadratic_properties_l2492_249270

/-- A quadratic function y = ax² + bx + c with given points -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  h_a_neq_0 : a ≠ 0
  h_point_neg1 : a * (-1)^2 + b * (-1) + c = -1
  h_point_0 : c = 3
  h_point_1 : a + b + c = 5
  h_point_3 : 9 * a + 3 * b + c = 3

/-- Theorem stating the properties of the quadratic function -/
theorem quadratic_properties (f : QuadraticFunction) :
  f.a * f.c < 0 ∧ f.a * 3^2 + (f.b - 1) * 3 + f.c = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_properties_l2492_249270


namespace NUMINAMATH_CALUDE_sum_specific_arithmetic_progression_l2492_249253

/-- Sum of an arithmetic progression -/
def sum_arithmetic_progression (a : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a + (n - 1) * d) / 2

/-- Number of terms in an arithmetic progression -/
def num_terms_arithmetic_progression (a : ℤ) (d : ℤ) (l : ℤ) : ℕ :=
  ((l - a) / d).toNat + 1

theorem sum_specific_arithmetic_progression :
  let a : ℤ := -45  -- First term
  let d : ℤ := 2    -- Common difference
  let l : ℤ := 23   -- Last term
  let n : ℕ := num_terms_arithmetic_progression a d l
  sum_arithmetic_progression a d n = -385 := by
sorry

end NUMINAMATH_CALUDE_sum_specific_arithmetic_progression_l2492_249253


namespace NUMINAMATH_CALUDE_no_integer_solutions_l2492_249259

theorem no_integer_solutions (p₁ p₂ α n : ℕ) : 
  Prime p₁ → Prime p₂ → Odd p₁ → Odd p₂ → α > 1 → n > 1 →
  ¬ ∃ (α n : ℕ), ((p₂ - 1) / 2)^p₁ + ((p₂ + 1) / 2)^p₁ = α^n :=
by sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l2492_249259


namespace NUMINAMATH_CALUDE_min_value_sum_of_reciprocals_l2492_249258

theorem min_value_sum_of_reciprocals (n : ℕ) (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  1 / (1 + a^n) + 1 / (1 + b^n) ≥ 1 ∧
  (1 / (1 + a^n) + 1 / (1 + b^n) = 1 ↔ a = 1 ∧ b = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_of_reciprocals_l2492_249258


namespace NUMINAMATH_CALUDE_opera_house_rows_l2492_249264

/-- Represents an opera house with a certain number of rows -/
structure OperaHouse where
  rows : ℕ

/-- Represents a show at the opera house -/
structure Show where
  earnings : ℕ
  occupancyRate : ℚ

/-- Calculates the total number of seats in the opera house -/
def totalSeats (oh : OperaHouse) : ℕ := oh.rows * 10

/-- Calculates the number of tickets sold for a show -/
def ticketsSold (s : Show) : ℕ := s.earnings / 10

/-- Theorem: Given the conditions, the opera house has 150 rows -/
theorem opera_house_rows (oh : OperaHouse) (s : Show) :
  totalSeats oh = ticketsSold s / s.occupancyRate →
  s.earnings = 12000 →
  s.occupancyRate = 4/5 →
  oh.rows = 150 := by
  sorry


end NUMINAMATH_CALUDE_opera_house_rows_l2492_249264


namespace NUMINAMATH_CALUDE_shaded_area_circle_in_square_l2492_249268

/-- The area of the shaded region between a circle inscribed in a square,
    where the circle touches the midpoints of the square's sides. -/
theorem shaded_area_circle_in_square (side_length : ℝ) (h : side_length = 12) :
  side_length ^ 2 - π * (side_length / 2) ^ 2 = side_length ^ 2 - π * 36 :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_circle_in_square_l2492_249268


namespace NUMINAMATH_CALUDE_intersection_and_union_of_A_and_B_l2492_249221

def A : Set ℝ := {x | x^2 - 4 = 0}
def B : Set ℝ := {y | ∃ x, y = x^2 - 4}

theorem intersection_and_union_of_A_and_B :
  (A ∩ B = A) ∧ (A ∪ B = B) := by sorry

end NUMINAMATH_CALUDE_intersection_and_union_of_A_and_B_l2492_249221


namespace NUMINAMATH_CALUDE_factor_in_range_l2492_249276

theorem factor_in_range : ∃ (n : ℕ), 
  1210000 < n ∧ 
  n < 1220000 ∧ 
  1464101210001 % n = 0 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_factor_in_range_l2492_249276


namespace NUMINAMATH_CALUDE_fraction_equals_eight_over_twentyseven_l2492_249239

def numerator : ℕ := 1*2*4 + 2*4*8 + 3*6*12 + 4*8*16
def denominator : ℕ := 1*3*9 + 2*6*18 + 3*9*27 + 4*12*36

theorem fraction_equals_eight_over_twentyseven :
  (numerator : ℚ) / denominator = 8 / 27 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_eight_over_twentyseven_l2492_249239


namespace NUMINAMATH_CALUDE_petya_wins_l2492_249283

/-- Represents the game state -/
structure GameState where
  total_players : Nat
  vasya_turn : Bool

/-- Represents the result of the game -/
inductive GameResult
  | VasyaWins
  | PetyaWins

/-- Optimal play function -/
def optimal_play (state : GameState) : GameResult :=
  sorry

/-- The main theorem -/
theorem petya_wins :
  ∀ (initial_state : GameState),
    initial_state.total_players = 2022 →
    initial_state.vasya_turn = true →
    optimal_play initial_state = GameResult.PetyaWins :=
  sorry

end NUMINAMATH_CALUDE_petya_wins_l2492_249283


namespace NUMINAMATH_CALUDE_square_sum_identity_l2492_249213

theorem square_sum_identity (x : ℝ) : (x + 1)^2 + 2*(x + 1)*(3 - x) + (3 - x)^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_identity_l2492_249213


namespace NUMINAMATH_CALUDE_max_value_of_expression_l2492_249250

theorem max_value_of_expression (x y : ℝ) : 
  |x + 1| - |x - 1| - |y - 4| - |y| ≤ -2 := by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l2492_249250


namespace NUMINAMATH_CALUDE_cos_difference_value_l2492_249297

theorem cos_difference_value (A B : ℝ) 
  (h1 : Real.cos A + Real.cos B = 1/2) 
  (h2 : Real.sin A + Real.sin B = 3/2) : 
  Real.cos (A - B) = 1/4 := by
sorry

end NUMINAMATH_CALUDE_cos_difference_value_l2492_249297


namespace NUMINAMATH_CALUDE_bisection_uses_all_structures_l2492_249255

/-- Represents the different algorithm structures -/
inductive AlgorithmStructure
  | Sequential
  | Conditional
  | Loop

/-- Represents the bisection method for a specific equation -/
structure BisectionMethod where
  equation : ℝ → ℝ
  approximateRoot : ℝ → ℝ → ℝ → ℝ

/-- The bisection method for x^2 - 10 = 0 -/
def bisectionForXSquaredMinus10 : BisectionMethod :=
  { equation := λ x => x^2 - 10,
    approximateRoot := sorry }

/-- Checks if a given algorithm structure is used in the bisection method -/
def usesStructure (b : BisectionMethod) (s : AlgorithmStructure) : Prop :=
  match s with
  | AlgorithmStructure.Sequential => sorry
  | AlgorithmStructure.Conditional => sorry
  | AlgorithmStructure.Loop => sorry

theorem bisection_uses_all_structures :
  ∀ s : AlgorithmStructure, usesStructure bisectionForXSquaredMinus10 s := by
  sorry

end NUMINAMATH_CALUDE_bisection_uses_all_structures_l2492_249255


namespace NUMINAMATH_CALUDE_number_with_specific_totient_l2492_249257

theorem number_with_specific_totient (N : ℕ) (α β γ : ℕ) :
  N = 3^α * 5^β * 7^γ →
  Nat.totient N = 3600 →
  N = 7875 := by
sorry

end NUMINAMATH_CALUDE_number_with_specific_totient_l2492_249257


namespace NUMINAMATH_CALUDE_fraction_meaningful_iff_not_neg_two_l2492_249295

theorem fraction_meaningful_iff_not_neg_two (x : ℝ) :
  (∃ y : ℝ, y = 1 / (x + 2)) ↔ x ≠ -2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_meaningful_iff_not_neg_two_l2492_249295


namespace NUMINAMATH_CALUDE_relationship_proof_l2492_249209

theorem relationship_proof (a b : ℝ) (h1 : a + b > 0) (h2 : b < 0) : a > -b ∧ -b > b ∧ b > -a := by
  sorry

end NUMINAMATH_CALUDE_relationship_proof_l2492_249209


namespace NUMINAMATH_CALUDE_smallest_number_with_conditions_l2492_249260

def has_exactly_six_divisors (n : ℕ) : Prop :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card = 6

def all_divisors_accommodate (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∣ n → n % d = 0

theorem smallest_number_with_conditions : 
  ∃ n : ℕ, 
    n % 18 = 0 ∧ 
    has_exactly_six_divisors n ∧
    all_divisors_accommodate n ∧
    (∀ m : ℕ, m < n → 
      ¬(m % 18 = 0 ∧ 
        has_exactly_six_divisors m ∧ 
        all_divisors_accommodate m)) ∧
    n = 72 :=
  sorry

end NUMINAMATH_CALUDE_smallest_number_with_conditions_l2492_249260


namespace NUMINAMATH_CALUDE_time_after_1457_minutes_l2492_249267

/-- Represents a time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  hValid : hours < 24 ∧ minutes < 60

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : ℕ) : Time :=
  sorry

/-- Converts a number to a 24-hour time -/
def minutesToTime (m : ℕ) : Time :=
  sorry

theorem time_after_1457_minutes :
  let start_time : Time := ⟨3, 0, sorry⟩
  let added_minutes : ℕ := 1457
  let end_time : Time := addMinutes start_time added_minutes
  end_time = ⟨3, 17, sorry⟩ :=
sorry

end NUMINAMATH_CALUDE_time_after_1457_minutes_l2492_249267


namespace NUMINAMATH_CALUDE_largest_integer_with_gcd_six_largest_integer_is_138_l2492_249280

theorem largest_integer_with_gcd_six (n : ℕ) : n < 150 ∧ Nat.gcd n 18 = 6 → n ≤ 138 :=
sorry

theorem largest_integer_is_138 : ∃ n : ℕ, n = 138 ∧ n < 150 ∧ Nat.gcd n 18 = 6 :=
sorry

end NUMINAMATH_CALUDE_largest_integer_with_gcd_six_largest_integer_is_138_l2492_249280


namespace NUMINAMATH_CALUDE_total_miles_traveled_l2492_249247

theorem total_miles_traveled (initial_reading additional_distance : Real) 
  (h1 : initial_reading = 212.3)
  (h2 : additional_distance = 372.0) : 
  initial_reading + additional_distance = 584.3 := by
sorry

end NUMINAMATH_CALUDE_total_miles_traveled_l2492_249247


namespace NUMINAMATH_CALUDE_tan_three_pi_fourth_l2492_249216

theorem tan_three_pi_fourth : Real.tan (3 * π / 4) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_three_pi_fourth_l2492_249216


namespace NUMINAMATH_CALUDE_fuel_calculation_correct_l2492_249202

/-- Calculates the total fuel needed for a plane trip given the specified conditions -/
def total_fuel_needed (base_fuel_per_mile : ℕ) (fuel_increase_per_person : ℕ) 
  (fuel_increase_per_bag : ℕ) (passengers : ℕ) (crew : ℕ) (bags_per_person : ℕ) 
  (trip_distance : ℕ) : ℕ :=
  let total_people := passengers + crew
  let total_bags := total_people * bags_per_person
  let fuel_per_mile := base_fuel_per_mile + 
    total_people * fuel_increase_per_person + 
    total_bags * fuel_increase_per_bag
  fuel_per_mile * trip_distance

/-- Theorem stating that the total fuel needed for the given conditions is 106,000 gallons -/
theorem fuel_calculation_correct : 
  total_fuel_needed 20 3 2 30 5 2 400 = 106000 := by
  sorry

end NUMINAMATH_CALUDE_fuel_calculation_correct_l2492_249202


namespace NUMINAMATH_CALUDE_geometric_sequence_a4_l2492_249262

/-- A geometric sequence with real terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), ∀ n, a (n + 1) = r * a n

theorem geometric_sequence_a4 (a : ℕ → ℝ) :
  GeometricSequence a → a 2 = 9 → a 6 = 1 → a 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a4_l2492_249262


namespace NUMINAMATH_CALUDE_circle_and_line_properties_l2492_249289

-- Define the circle M
def circle_M (x y : ℝ) : Prop :=
  ∃ (a : ℝ), (x - a)^2 + (y - a)^2 = 36 ∧ a = 4

-- Define the line m
def line_m (x y : ℝ) : Prop :=
  3*x - 4*y - 16 = 0 ∨ x = 0

-- Theorem statement
theorem circle_and_line_properties :
  -- Circle M passes through A(√2, -√2) and B(10, 4)
  circle_M (Real.sqrt 2) (-Real.sqrt 2) ∧ circle_M 10 4 ∧
  -- The center of circle M lies on the line y = x
  ∃ (a : ℝ), circle_M a a ∧
  -- A line m passing through (0, -4) intersects circle M to form a chord of length 4√5
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    line_m x₁ y₁ ∧ line_m x₂ y₂ ∧
    circle_M x₁ y₁ ∧ circle_M x₂ y₂ ∧
    line_m 0 (-4) ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 80 →
  -- The standard equation of circle M is (x-4)² + (y-4)² = 36
  ∀ (x y : ℝ), circle_M x y ↔ (x - 4)^2 + (y - 4)^2 = 36 ∧
  -- The equation of line m is either 3x - 4y - 16 = 0 or x = 0
  ∀ (x y : ℝ), line_m x y ↔ (3*x - 4*y - 16 = 0 ∨ x = 0) :=
by sorry

end NUMINAMATH_CALUDE_circle_and_line_properties_l2492_249289


namespace NUMINAMATH_CALUDE_linear_equation_solution_l2492_249298

theorem linear_equation_solution (m : ℝ) : 
  (1 : ℝ) * m - 3 = 3 → m = 6 := by sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l2492_249298


namespace NUMINAMATH_CALUDE_shaded_area_theorem_l2492_249293

/-- The fraction of shaded area in each subdivision -/
def shaded_fraction : ℚ := 7 / 16

/-- The ratio of area of each subdivision to the whole square -/
def subdivision_ratio : ℚ := 1 / 16

/-- The total shaded fraction of the square -/
def total_shaded_fraction : ℚ := 7 / 15

theorem shaded_area_theorem :
  (shaded_fraction * (1 - subdivision_ratio)⁻¹ : ℚ) = total_shaded_fraction := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_theorem_l2492_249293


namespace NUMINAMATH_CALUDE_smallest_integer_solution_l2492_249220

theorem smallest_integer_solution :
  ∀ y : ℤ, (8 - 3 * y ≤ 23) → y ≥ -5 ∧ ∀ z : ℤ, z < -5 → (8 - 3 * z > 23) :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_solution_l2492_249220


namespace NUMINAMATH_CALUDE_lawn_care_time_l2492_249248

/-- The time it takes Max to mow the lawn, in minutes -/
def mow_time : ℕ := 40

/-- The time it takes Max to fertilize the lawn, in minutes -/
def fertilize_time : ℕ := 2 * mow_time

/-- The total time it takes Max to both mow and fertilize the lawn, in minutes -/
def total_time : ℕ := mow_time + fertilize_time

theorem lawn_care_time : total_time = 120 := by
  sorry

end NUMINAMATH_CALUDE_lawn_care_time_l2492_249248


namespace NUMINAMATH_CALUDE_product_seven_consecutive_divisible_by_ten_l2492_249228

/-- The product of any seven consecutive positive integers is divisible by 10 -/
theorem product_seven_consecutive_divisible_by_ten (n : ℕ) : 
  10 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4) * (n + 5) * (n + 6)) := by
sorry

end NUMINAMATH_CALUDE_product_seven_consecutive_divisible_by_ten_l2492_249228


namespace NUMINAMATH_CALUDE_speeding_fine_calculation_l2492_249285

/-- Calculates the base fine for speeding given the total amount owed and other fees --/
theorem speeding_fine_calculation 
  (speed_limit : ℕ) 
  (actual_speed : ℕ) 
  (fine_increase_per_mph : ℕ) 
  (court_costs : ℕ) 
  (lawyer_fee_per_hour : ℕ) 
  (lawyer_hours : ℕ) 
  (total_owed : ℕ) : 
  speed_limit = 30 →
  actual_speed = 75 →
  fine_increase_per_mph = 2 →
  court_costs = 300 →
  lawyer_fee_per_hour = 80 →
  lawyer_hours = 3 →
  total_owed = 820 →
  ∃ (base_fine : ℕ),
    base_fine = 190 ∧
    total_owed = base_fine + 
      2 * (actual_speed - speed_limit) * 2 + 
      court_costs + 
      lawyer_fee_per_hour * lawyer_hours :=
by sorry

end NUMINAMATH_CALUDE_speeding_fine_calculation_l2492_249285


namespace NUMINAMATH_CALUDE_max_dot_product_unit_vector_l2492_249203

theorem max_dot_product_unit_vector (a b : ℝ × ℝ) :
  (∀ (x y : ℝ), a = (x, y) → x^2 + y^2 = 1) →
  b = (Real.sqrt 3, -1) →
  (∃ (m : ℝ), m = (a.1 * b.1 + a.2 * b.2) ∧ 
    ∀ (x y : ℝ), x^2 + y^2 = 1 → (x * b.1 + y * b.2) ≤ m) →
  (∃ (max : ℝ), max = 2 ∧ 
    ∀ (x y : ℝ), x^2 + y^2 = 1 → (x * b.1 + y * b.2) ≤ max) :=
by
  sorry

end NUMINAMATH_CALUDE_max_dot_product_unit_vector_l2492_249203


namespace NUMINAMATH_CALUDE_f_positive_solution_set_m_upper_bound_l2492_249245

def f (x : ℝ) := |x - 2| - |2*x + 1|

theorem f_positive_solution_set :
  {x : ℝ | f x > 0} = Set.Ioo (-3) (1/3) :=
sorry

theorem m_upper_bound (m : ℝ) :
  (∃ x₀ : ℝ, f x₀ > 2*m + 1) → m < 3/4 :=
sorry

end NUMINAMATH_CALUDE_f_positive_solution_set_m_upper_bound_l2492_249245


namespace NUMINAMATH_CALUDE_graph_horizontal_shift_l2492_249299

-- Define a generic function g
variable (g : ℝ → ℝ)

-- Define a point (x, y) on the graph of y = g(x)
variable (x y : ℝ)

-- Define the horizontal shift
def h : ℝ := 3

-- Theorem statement
theorem graph_horizontal_shift :
  y = g x ↔ y = g (x - h) :=
sorry

end NUMINAMATH_CALUDE_graph_horizontal_shift_l2492_249299


namespace NUMINAMATH_CALUDE_min_abs_a_for_solvable_equation_l2492_249294

theorem min_abs_a_for_solvable_equation :
  ∀ (a b : ℤ),
  (a + 2 * b = 32) →
  (∀ a' : ℤ, a' > 0 ∧ (∃ b' : ℤ, a' + 2 * b' = 32) → a' ≥ 4) →
  (∃ b'' : ℤ, (-2) + 2 * b'' = 32) →
  (∃ a₀ : ℤ, |a₀| = 2 ∧ (∃ b₀ : ℤ, a₀ + 2 * b₀ = 32) ∧
    ∀ a' : ℤ, (∃ b' : ℤ, a' + 2 * b' = 32) → |a'| ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_min_abs_a_for_solvable_equation_l2492_249294


namespace NUMINAMATH_CALUDE_pythagorean_triple_identification_l2492_249235

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a * a + b * b = c * c

theorem pythagorean_triple_identification :
  ¬ is_pythagorean_triple 3 4 5 ∧
  ¬ is_pythagorean_triple 3 4 6 ∧
  is_pythagorean_triple 5 12 13 ∧
  ¬ is_pythagorean_triple 9 12 15 :=
by sorry

end NUMINAMATH_CALUDE_pythagorean_triple_identification_l2492_249235


namespace NUMINAMATH_CALUDE_russian_pairing_probability_l2492_249282

def total_players : ℕ := 10
def russian_players : ℕ := 4

theorem russian_pairing_probability :
  let remaining_players := total_players - 1
  let remaining_russian_players := russian_players - 1
  let first_pair_prob := remaining_russian_players / remaining_players
  let second_pair_prob := 1 / (remaining_players - 1)
  first_pair_prob * second_pair_prob = 1 / 21 := by
  sorry

end NUMINAMATH_CALUDE_russian_pairing_probability_l2492_249282


namespace NUMINAMATH_CALUDE_hamburger_price_is_5_l2492_249206

-- Define the variables
def num_hamburgers : ℕ := 2
def num_cola : ℕ := 3
def cola_price : ℚ := 2
def discount : ℚ := 4
def total_paid : ℚ := 12

-- Define the theorem
theorem hamburger_price_is_5 :
  ∃ (hamburger_price : ℚ),
    hamburger_price * num_hamburgers + cola_price * num_cola - discount = total_paid ∧
    hamburger_price = 5 :=
by sorry

end NUMINAMATH_CALUDE_hamburger_price_is_5_l2492_249206


namespace NUMINAMATH_CALUDE_non_shaded_perimeter_l2492_249207

/-- Given a large rectangle containing a smaller shaded rectangle, 
    where the total area is 180 square inches and the shaded area is 120 square inches,
    prove that the perimeter of the non-shaded region is 32 inches. -/
theorem non_shaded_perimeter (total_area shaded_area : ℝ) 
  (h1 : total_area = 180)
  (h2 : shaded_area = 120)
  (h3 : ∃ (a b : ℝ), a * b = total_area - shaded_area ∧ a + b = 16) :
  2 * 16 = 32 := by
  sorry

end NUMINAMATH_CALUDE_non_shaded_perimeter_l2492_249207


namespace NUMINAMATH_CALUDE_correct_calculation_l2492_249281

theorem correct_calculation : ∃ x : ℝ, 5 * x = 40 ∧ 2 * x = 16 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2492_249281


namespace NUMINAMATH_CALUDE_gcf_5_factorial_6_factorial_l2492_249249

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem gcf_5_factorial_6_factorial : 
  Nat.gcd (factorial 5) (factorial 6) = factorial 5 := by
  sorry

end NUMINAMATH_CALUDE_gcf_5_factorial_6_factorial_l2492_249249


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l2492_249208

theorem complex_modulus_problem (i : ℂ) (h : i^2 = -1) : 
  Complex.abs (4 * i / (1 - i)) = 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l2492_249208


namespace NUMINAMATH_CALUDE_fold_paper_sum_l2492_249254

/-- The fold line equation --/
def fold_line (x y : ℝ) : Prop := y = 2 * x - 4

/-- The relation between (8,4) and (m,n) --/
def point_relation (m n : ℝ) : Prop := 2 * n - 8 = -m + 8

/-- The theorem stating that m + n = 32/3 --/
theorem fold_paper_sum (m n : ℝ) 
  (h1 : fold_line ((1 + 5) / 2) ((3 + 1) / 2))
  (h2 : fold_line ((8 + m) / 2) ((4 + n) / 2))
  (h3 : point_relation m n) :
  m + n = 32 / 3 := by sorry

end NUMINAMATH_CALUDE_fold_paper_sum_l2492_249254


namespace NUMINAMATH_CALUDE_min_value_expression_l2492_249201

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h : x * y^2 * z = 72) : 
  x^2 + 4*x*y + 4*y^2 + 2*z^2 ≥ 120 ∧ 
  (x^2 + 4*x*y + 4*y^2 + 2*z^2 = 120 ↔ x = 6 ∧ y = 3 ∧ z = 4) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2492_249201


namespace NUMINAMATH_CALUDE_bobs_weight_l2492_249227

/-- Given two people, Jim and Bob, prove Bob's weight under specific conditions. -/
theorem bobs_weight (jim_weight bob_weight : ℝ) : 
  (jim_weight + bob_weight = 200) →
  (bob_weight + jim_weight = bob_weight / 3) →
  bob_weight = 120 := by
  sorry

end NUMINAMATH_CALUDE_bobs_weight_l2492_249227


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2492_249229

-- Define a proposition P to represent the given condition
variable (P : Prop)

-- Define a proposition Q to represent the conclusion
variable (Q : Prop)

-- Theorem stating that P is sufficient but not necessary for Q
theorem sufficient_but_not_necessary : (P → Q) ∧ ¬(Q → P) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2492_249229


namespace NUMINAMATH_CALUDE_four_digit_divisible_by_eleven_l2492_249232

theorem four_digit_divisible_by_eleven (B : ℕ) : 
  (4000 + 100 * B + 10 * B + 6) % 11 = 0 → B = 5 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_divisible_by_eleven_l2492_249232


namespace NUMINAMATH_CALUDE_S_five_three_l2492_249274

def S (a b : ℝ) : ℝ := 4 * a + 6 * b - 2 * a * b

theorem S_five_three : S 5 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_S_five_three_l2492_249274


namespace NUMINAMATH_CALUDE_all_b_k_divisible_by_six_l2492_249214

/-- The number obtained by writing the integers from 1 to n from left to right -/
def b (n : ℕ) : ℕ := sorry

/-- The sum of the squares of the digits of b_n -/
def g (n : ℕ) : ℕ := (n * (n + 1) * (2 * n + 1)) / 6

theorem all_b_k_divisible_by_six (k : ℕ) (h : 1 ≤ k ∧ k ≤ 50) : 
  6 ∣ g k := by sorry

end NUMINAMATH_CALUDE_all_b_k_divisible_by_six_l2492_249214


namespace NUMINAMATH_CALUDE_ice_cream_distribution_l2492_249288

theorem ice_cream_distribution (total_sandwiches : ℕ) (num_nieces : ℕ) 
  (h1 : total_sandwiches = 143) (h2 : num_nieces = 11) :
  total_sandwiches / num_nieces = 13 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_distribution_l2492_249288


namespace NUMINAMATH_CALUDE_PQ_length_is_correct_l2492_249237

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the lengths of the sides
def side_lengths (t : Triangle) : ℝ × ℝ × ℝ :=
  (7, 8, 9)

-- Define the altitude AH
def altitude (t : Triangle) : ℝ × ℝ → ℝ × ℝ := sorry

-- Define the angle bisectors BD and CE
def angle_bisector_BD (t : Triangle) : ℝ × ℝ → ℝ × ℝ := sorry
def angle_bisector_CE (t : Triangle) : ℝ × ℝ → ℝ × ℝ := sorry

-- Define the intersection points P and Q
def P (t : Triangle) : ℝ × ℝ := sorry
def Q (t : Triangle) : ℝ × ℝ := sorry

-- Define the length of PQ
def PQ_length (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem PQ_length_is_correct (t : Triangle) :
  PQ_length t = (8 / 15) * Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_PQ_length_is_correct_l2492_249237


namespace NUMINAMATH_CALUDE_missing_number_proof_l2492_249290

theorem missing_number_proof : ∃ x : ℝ, x + Real.sqrt (-4 + 6 * 4 / 3) = 13 ∧ x = 11 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_proof_l2492_249290


namespace NUMINAMATH_CALUDE_square_containing_circle_l2492_249269

/-- The area and perimeter of the smallest square containing a circle --/
theorem square_containing_circle (r : ℝ) (h : r = 6) :
  ∃ (area perimeter : ℝ),
    area = (2 * r) ^ 2 ∧
    perimeter = 4 * (2 * r) ∧
    area = 144 ∧
    perimeter = 48 := by
  sorry

end NUMINAMATH_CALUDE_square_containing_circle_l2492_249269


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2492_249224

def geometric_sequence (a : ℕ → ℚ) := ∀ n, a (n + 1) = a n * (a 2 / a 1)

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℚ) 
  (h_geom : geometric_sequence a) 
  (h_a1 : a 1 = 1/8) 
  (h_a4 : a 4 = -1) : 
  a 2 / a 1 = -2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2492_249224


namespace NUMINAMATH_CALUDE_triangle_negative_five_sixths_one_half_l2492_249200

/-- The triangle operation on rational numbers -/
def triangle (a b : ℚ) : ℚ := b - a

theorem triangle_negative_five_sixths_one_half :
  triangle (-5/6) (1/2) = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_negative_five_sixths_one_half_l2492_249200


namespace NUMINAMATH_CALUDE_square_plot_with_path_l2492_249279

theorem square_plot_with_path (path_area : ℝ) (edge_diff : ℝ) (total_area : ℝ) :
  path_area = 464 →
  edge_diff = 32 →
  (∃ x y : ℝ,
    x > 0 ∧
    y > 0 ∧
    x^2 - y^2 = path_area ∧
    4 * (x - y) = edge_diff ∧
    total_area = x^2) →
  total_area = 1089 := by
  sorry

end NUMINAMATH_CALUDE_square_plot_with_path_l2492_249279


namespace NUMINAMATH_CALUDE_parallelogram_intersection_theorem_l2492_249238

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a parallelogram ABCD with point F -/
structure Parallelogram :=
  (A B C D F : Point)
  (isParallelogram : sorry) -- Condition that ABCD is a parallelogram
  (F_on_AD_extension : sorry) -- Condition that F is on the extension of AD

/-- Represents the intersection points E and G -/
structure Intersections (p : Parallelogram) :=
  (E : Point)
  (G : Point)
  (E_on_AC_BF : sorry) -- Condition that E is on both AC and BF
  (G_on_DC_BF : sorry) -- Condition that G is on both DC and BF

/-- Calculate the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- The main theorem -/
theorem parallelogram_intersection_theorem (p : Parallelogram) (i : Intersections p) :
  distance i.E p.F = 40 → distance i.G p.F = 18 → distance p.B i.E = 20 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_parallelogram_intersection_theorem_l2492_249238


namespace NUMINAMATH_CALUDE_min_b_value_l2492_249217

theorem min_b_value (a c b : ℕ+) (h1 : a < c) (h2 : c < b)
  (h3 : ∃! p : ℝ × ℝ, 3 * p.1 + p.2 = 3000 ∧ 
    p.2 = |p.1 - a.val| + |p.1 - c.val| + |p.1 - b.val|) :
  ∀ b' : ℕ+, (∃ a' c' : ℕ+, a' < c' ∧ c' < b' ∧
    ∃! p : ℝ × ℝ, 3 * p.1 + p.2 = 3000 ∧ 
    p.2 = |p.1 - a'.val| + |p.1 - c'.val| + |p.1 - b'.val|) → 
  9 ≤ b'.val := by
  sorry

end NUMINAMATH_CALUDE_min_b_value_l2492_249217


namespace NUMINAMATH_CALUDE_equal_sum_squared_distances_exist_l2492_249223

-- Define a triangle as a tuple of three points in a plane
def Triangle := (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)

-- Define a function to calculate the sum of squared distances from a point to triangle vertices
def sumSquaredDistances (p : ℝ × ℝ) (t : Triangle) : ℝ :=
  let (a, b, c) := t
  (p.1 - a.1)^2 + (p.2 - a.2)^2 +
  (p.1 - b.1)^2 + (p.2 - b.2)^2 +
  (p.1 - c.1)^2 + (p.2 - c.2)^2

-- State the theorem
theorem equal_sum_squared_distances_exist (t1 t2 t3 : Triangle) :
  ∃ (p : ℝ × ℝ), sumSquaredDistances p t1 = sumSquaredDistances p t2 ∧
                 sumSquaredDistances p t2 = sumSquaredDistances p t3 :=
sorry

end NUMINAMATH_CALUDE_equal_sum_squared_distances_exist_l2492_249223


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l2492_249261

theorem simplify_and_rationalize (x : ℝ) : 
  1 / (2 + 1 / (Real.sqrt 5 + 2)) = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l2492_249261


namespace NUMINAMATH_CALUDE_constant_term_expansion_l2492_249204

theorem constant_term_expansion (a : ℝ) : 
  a > 0 → (∃ k : ℕ, k = (a^2 * 2^2 * 6 : ℝ) ∧ k = 96) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l2492_249204


namespace NUMINAMATH_CALUDE_paint_used_after_four_weeks_l2492_249210

/-- Calculates the amount of paint used over 4 weeks given an initial amount and usage fractions --/
def paint_used (initial : ℝ) (w1_frac w2_frac w3_frac w4_frac : ℝ) : ℝ :=
  let w1_used := w1_frac * initial
  let w1_remaining := initial - w1_used
  let w2_used := w2_frac * w1_remaining
  let w2_remaining := w1_remaining - w2_used
  let w3_used := w3_frac * w2_remaining
  let w3_remaining := w2_remaining - w3_used
  let w4_used := w4_frac * w3_remaining
  w1_used + w2_used + w3_used + w4_used

/-- The theorem stating the amount of paint used after 4 weeks --/
theorem paint_used_after_four_weeks :
  let initial_paint := 360
  let week1_fraction := 1/4
  let week2_fraction := 1/3
  let week3_fraction := 2/5
  let week4_fraction := 3/7
  abs (paint_used initial_paint week1_fraction week2_fraction week3_fraction week4_fraction - 298.2857) < 0.0001 := by
  sorry


end NUMINAMATH_CALUDE_paint_used_after_four_weeks_l2492_249210


namespace NUMINAMATH_CALUDE_problem_statement_l2492_249263

theorem problem_statement (x y z : ℝ) 
  (h1 : x * z / (x + y) + y * x / (y + z) + z * y / (z + x) = 2)
  (h2 : z * y / (x + y) + x * z / (y + z) + y * x / (z + x) = 3) :
  y / (x + y) + z / (y + z) + x / (z + x) = 1 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2492_249263


namespace NUMINAMATH_CALUDE_equation_root_implies_m_value_l2492_249271

theorem equation_root_implies_m_value (x m : ℝ) :
  x > 0 →
  (x - 1) / (x - 5) = m * x / (10 - 2 * x) →
  m = -8/5 :=
by sorry

end NUMINAMATH_CALUDE_equation_root_implies_m_value_l2492_249271


namespace NUMINAMATH_CALUDE_lunchroom_total_people_l2492_249246

def num_tables : ℕ := 34
def first_table_students : ℕ := 6
def teacher_count : ℕ := 5

def arithmetic_sum (n : ℕ) (a1 : ℕ) (d : ℕ) : ℕ :=
  n * (2 * a1 + (n - 1) * d) / 2

theorem lunchroom_total_people :
  arithmetic_sum num_tables first_table_students 1 + teacher_count = 770 := by
  sorry

end NUMINAMATH_CALUDE_lunchroom_total_people_l2492_249246


namespace NUMINAMATH_CALUDE_sams_ribbon_length_l2492_249287

/-- The total length of a ribbon cut into equal pieces -/
def total_ribbon_length (piece_length : ℕ) (num_pieces : ℕ) : ℕ :=
  piece_length * num_pieces

/-- Theorem: The total length of Sam's ribbon is 3723 cm -/
theorem sams_ribbon_length : 
  total_ribbon_length 73 51 = 3723 := by
  sorry

end NUMINAMATH_CALUDE_sams_ribbon_length_l2492_249287


namespace NUMINAMATH_CALUDE_mark_total_eggs_l2492_249244

/-- The number of people sharing the eggs -/
def num_people : ℕ := 4

/-- The number of eggs each person gets when distributed equally -/
def eggs_per_person : ℕ := 6

/-- The total number of eggs Mark has -/
def total_eggs : ℕ := num_people * eggs_per_person

theorem mark_total_eggs : total_eggs = 24 := by
  sorry

end NUMINAMATH_CALUDE_mark_total_eggs_l2492_249244


namespace NUMINAMATH_CALUDE_ab_plus_one_gt_a_plus_b_l2492_249233

-- Define the set M
def M : Set ℝ := {x | 0 < x ∧ x < 1}

-- State the theorem
theorem ab_plus_one_gt_a_plus_b (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) :
  a * b + 1 > a + b := by
  sorry

end NUMINAMATH_CALUDE_ab_plus_one_gt_a_plus_b_l2492_249233


namespace NUMINAMATH_CALUDE_student_not_asked_probability_l2492_249275

/-- The probability of a student not being asked in either of two consecutive lessons -/
theorem student_not_asked_probability
  (total_students : ℕ)
  (selected_students : ℕ)
  (previous_lesson_pool : ℕ)
  (h1 : total_students = 30)
  (h2 : selected_students = 3)
  (h3 : previous_lesson_pool = 10)
  : ℚ :=
  11 / 30

/-- The proof of the theorem -/
lemma student_not_asked_probability_proof :
  student_not_asked_probability 30 3 10 rfl rfl rfl = 11 / 30 := by
  sorry

end NUMINAMATH_CALUDE_student_not_asked_probability_l2492_249275


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l2492_249251

/-- Given a geometric sequence {aₙ} where a₁ and a₁₃ are the roots of x² - 8x + 1 = 0,
    the product a₅ · a₇ · a₉ equals 1. -/
theorem geometric_sequence_product (a : ℕ → ℝ) : 
  (∀ n m : ℕ, a (n + m) = a n * a m) →  -- geometric sequence property
  (a 1)^2 - 8*(a 1) + 1 = 0 →           -- a₁ is a root
  (a 13)^2 - 8*(a 13) + 1 = 0 →         -- a₁₃ is a root
  a 5 * a 7 * a 9 = 1 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l2492_249251


namespace NUMINAMATH_CALUDE_curve_point_coordinates_l2492_249240

theorem curve_point_coordinates (θ : Real) (x y : Real) :
  0 ≤ θ ∧ θ ≤ π →
  x = 3 * Real.cos θ →
  y = 4 * Real.sin θ →
  y = x →
  x = 12/5 ∧ y = 12/5 := by
sorry

end NUMINAMATH_CALUDE_curve_point_coordinates_l2492_249240


namespace NUMINAMATH_CALUDE_symmetry_implies_values_l2492_249226

/-- Two points are symmetric with respect to the y-axis if their x-coordinates are opposite and y-coordinates are equal -/
def symmetric_wrt_y_axis (p1 p2 : ℝ × ℝ) : Prop :=
  p1.1 = -p2.1 ∧ p1.2 = p2.2

theorem symmetry_implies_values :
  ∀ (a b : ℝ), symmetric_wrt_y_axis (a, 1) (5, b) → a = -5 ∧ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_implies_values_l2492_249226


namespace NUMINAMATH_CALUDE_tuesday_rain_amount_l2492_249284

/-- The amount of rain on Monday in inches -/
def monday_rain : ℝ := 0.9

/-- The difference in rain between Monday and Tuesday in inches -/
def rain_difference : ℝ := 0.7

/-- The amount of rain on Tuesday in inches -/
def tuesday_rain : ℝ := monday_rain - rain_difference

theorem tuesday_rain_amount : tuesday_rain = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_tuesday_rain_amount_l2492_249284


namespace NUMINAMATH_CALUDE_lyn_donation_l2492_249265

theorem lyn_donation (X : ℝ) : 
  (1/3 * X + 1/2 * X + 1/4 * (X - 1/3 * X - 1/2 * X) + 30 = X) → X = 240 :=
by sorry

end NUMINAMATH_CALUDE_lyn_donation_l2492_249265


namespace NUMINAMATH_CALUDE_jade_tower_levels_l2492_249241

/-- Calculates the number of complete levels in a Lego tower. -/
def towerLevels (totalPieces piecesPerLevel unusedPieces : ℕ) : ℕ :=
  (totalPieces - unusedPieces) / piecesPerLevel

/-- Proves that given the specific conditions, the tower has 11 levels. -/
theorem jade_tower_levels :
  towerLevels 100 7 23 = 11 := by
  sorry

end NUMINAMATH_CALUDE_jade_tower_levels_l2492_249241


namespace NUMINAMATH_CALUDE_roots_relation_l2492_249277

-- Define the original quadratic equation
def original_equation (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0

-- Define the roots of the original equation
def root1 (a b c : ℝ) : ℝ := sorry
def root2 (a b c : ℝ) : ℝ := sorry

-- Define the new quadratic equation
def new_equation (a b c y : ℝ) : Prop := a^2 * y^2 + a * (b - c) * y - b * c = 0

-- State the theorem
theorem roots_relation (a b c : ℝ) (ha : a ≠ 0) :
  (∃ y1 y2 : ℝ, new_equation a b c y1 ∧ new_equation a b c y2 ∧
    y1 = root1 a b c + root2 a b c ∧
    y2 = root1 a b c * root2 a b c) :=
sorry

end NUMINAMATH_CALUDE_roots_relation_l2492_249277


namespace NUMINAMATH_CALUDE_factorial_equation_l2492_249236

theorem factorial_equation : 6 * 10 * 4 * 168 = Nat.factorial 8 := by
  sorry

end NUMINAMATH_CALUDE_factorial_equation_l2492_249236


namespace NUMINAMATH_CALUDE_cyclist_speed_l2492_249222

/-- The cyclist's problem -/
theorem cyclist_speed :
  ∀ (expected_speed actual_speed : ℝ),
  expected_speed > 0 →
  actual_speed > 0 →
  actual_speed = expected_speed + 1 →
  96 / actual_speed = 96 / expected_speed - 2 →
  96 / expected_speed = 1.25 →
  actual_speed = 16 := by
  sorry

end NUMINAMATH_CALUDE_cyclist_speed_l2492_249222


namespace NUMINAMATH_CALUDE_root_magnitude_theorem_l2492_249218

theorem root_magnitude_theorem (A B C D : ℝ) 
  (h1 : ∀ x : ℂ, x^2 + A*x + B = 0 → Complex.abs x < 1)
  (h2 : ∀ x : ℂ, x^2 + C*x + D = 0 → Complex.abs x < 1) :
  ∀ x : ℂ, x^2 + (A+C)/2*x + (B+D)/2 = 0 → Complex.abs x < 1 :=
sorry

end NUMINAMATH_CALUDE_root_magnitude_theorem_l2492_249218


namespace NUMINAMATH_CALUDE_childrens_tickets_sold_l2492_249211

theorem childrens_tickets_sold 
  (adult_price : ℚ) 
  (child_price : ℚ) 
  (total_tickets : ℕ) 
  (total_revenue : ℚ) 
  (h1 : adult_price = 6)
  (h2 : child_price = 9/2)
  (h3 : total_tickets = 400)
  (h4 : total_revenue = 2100) :
  ∃ (adult_tickets child_tickets : ℕ),
    adult_tickets + child_tickets = total_tickets ∧
    adult_price * adult_tickets + child_price * child_tickets = total_revenue ∧
    child_tickets = 200 := by
  sorry

end NUMINAMATH_CALUDE_childrens_tickets_sold_l2492_249211


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2492_249273

/-- A quadratic function f(x) = ax^2 + bx + c where the solution set of f(x) ≥ 0 is [-1, 4] -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_inequality (a b c : ℝ) (h1 : a ≠ 0) 
  (h2 : Set.Icc (-1 : ℝ) 4 = {x | QuadraticFunction a b c x ≥ 0}) :
  QuadraticFunction a b c 2 > QuadraticFunction a b c 3 ∧ 
  QuadraticFunction a b c 3 > QuadraticFunction a b c (-1/2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2492_249273


namespace NUMINAMATH_CALUDE_funfair_unsold_tickets_l2492_249256

/-- Calculates the number of unsold tickets at a school funfair --/
theorem funfair_unsold_tickets (total_rolls : ℕ) (tickets_per_roll : ℕ)
  (fourth_grade_percent : ℚ) (fifth_grade_percent : ℚ) (sixth_grade_percent : ℚ)
  (seventh_grade_percent : ℚ) (eighth_grade_percent : ℚ) (ninth_grade_tickets : ℕ) :
  total_rolls = 50 →
  tickets_per_roll = 250 →
  fourth_grade_percent = 30 / 100 →
  fifth_grade_percent = 40 / 100 →
  sixth_grade_percent = 25 / 100 →
  seventh_grade_percent = 35 / 100 →
  eighth_grade_percent = 20 / 100 →
  ninth_grade_tickets = 150 →
  ∃ (unsold : ℕ), unsold = 1898 := by
  sorry

#check funfair_unsold_tickets

end NUMINAMATH_CALUDE_funfair_unsold_tickets_l2492_249256


namespace NUMINAMATH_CALUDE_geometric_progression_special_ratio_l2492_249205

/-- A geometric progression with positive terms where any term is equal to the sum of the next two following terms has a common ratio of (√5 - 1)/2. -/
theorem geometric_progression_special_ratio (a : ℝ) (r : ℝ) :
  a > 0 →  -- First term is positive
  r > 0 →  -- Common ratio is positive
  (∀ n : ℕ, a * r^n = a * r^(n+1) + a * r^(n+2)) →  -- Any term is sum of next two
  r = (Real.sqrt 5 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_special_ratio_l2492_249205


namespace NUMINAMATH_CALUDE_x_axis_symmetry_y_axis_symmetry_l2492_249243

-- Define the region
def region (x y : ℝ) : Prop := abs (x + 2*y) + abs (2*x - y) ≤ 8

-- Theorem: The region is symmetric about the x-axis
theorem x_axis_symmetry :
  ∀ x y : ℝ, region x y ↔ region x (-y) :=
sorry

-- Theorem: The region is symmetric about the y-axis
theorem y_axis_symmetry :
  ∀ x y : ℝ, region x y ↔ region (-x) y :=
sorry

end NUMINAMATH_CALUDE_x_axis_symmetry_y_axis_symmetry_l2492_249243


namespace NUMINAMATH_CALUDE_total_students_l2492_249252

/-- The total number of students in five classes given specific conditions -/
theorem total_students (finley johnson garcia smith patel : ℕ) : 
  finley = 24 →
  johnson = finley / 2 + 10 →
  garcia = 2 * johnson →
  smith = finley / 3 →
  patel = (3 * (finley + johnson + garcia)) / 4 →
  finley + johnson + garcia + smith + patel = 166 := by
sorry

end NUMINAMATH_CALUDE_total_students_l2492_249252


namespace NUMINAMATH_CALUDE_min_value_cubic_function_l2492_249212

/-- The function f(x) = x^3 - 3x has a minimum value of -2. -/
theorem min_value_cubic_function :
  ∃ m : ℝ, m = -2 ∧ ∀ x : ℝ, x^3 - 3*x ≥ m :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_cubic_function_l2492_249212


namespace NUMINAMATH_CALUDE_power_multiplication_l2492_249286

theorem power_multiplication (x : ℝ) : x^2 * x^3 = x^5 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l2492_249286


namespace NUMINAMATH_CALUDE_transformed_quadratic_roots_l2492_249234

theorem transformed_quadratic_roots 
  (a b c r s : ℝ) 
  (h1 : a ≠ 0) 
  (h2 : a * r^2 + b * r + c = 0) 
  (h3 : a * s^2 + b * s + c = 0) : 
  (a * r + b)^2 - b * (a * r + b) + a * c = 0 ∧ 
  (a * s + b)^2 - b * (a * s + b) + a * c = 0 := by
  sorry

end NUMINAMATH_CALUDE_transformed_quadratic_roots_l2492_249234


namespace NUMINAMATH_CALUDE_second_train_length_second_train_length_solution_l2492_249225

/-- Calculates the length of the second train given the speeds of two trains, 
    the time they take to clear each other, and the length of the first train. -/
theorem second_train_length 
  (speed1 : ℝ) 
  (speed2 : ℝ) 
  (clear_time : ℝ) 
  (length1 : ℝ) : ℝ :=
  let relative_speed := speed1 + speed2
  let relative_speed_ms := relative_speed * 1000 / 3600
  let total_distance := relative_speed_ms * clear_time
  total_distance - length1

/-- The length of the second train is approximately 165.12 meters. -/
theorem second_train_length_solution :
  ∃ ε > 0, |second_train_length 80 65 7.596633648618456 141 - 165.12| < ε :=
sorry

end NUMINAMATH_CALUDE_second_train_length_second_train_length_solution_l2492_249225


namespace NUMINAMATH_CALUDE_development_inheritance_relationship_false_l2492_249230

/-- Development is a prerequisite for inheritance -/
def development_prerequisite_for_inheritance : Prop := sorry

/-- Inheritance is a requirement for development -/
def inheritance_requirement_for_development : Prop := sorry

/-- The statement that development is a prerequisite for inheritance
    and inheritance is a requirement for development is false -/
theorem development_inheritance_relationship_false :
  ¬(development_prerequisite_for_inheritance ∧ inheritance_requirement_for_development) :=
sorry

end NUMINAMATH_CALUDE_development_inheritance_relationship_false_l2492_249230


namespace NUMINAMATH_CALUDE_number_value_l2492_249231

theorem number_value (N : ℝ) (h : (1/2) * N = 1) : N = 2 := by
  sorry

end NUMINAMATH_CALUDE_number_value_l2492_249231


namespace NUMINAMATH_CALUDE_same_color_shoe_probability_l2492_249291

/-- The number of pairs of shoes -/
def num_pairs : ℕ := 9

/-- The total number of shoes -/
def total_shoes : ℕ := 2 * num_pairs

/-- The probability of selecting two shoes of the same color -/
def prob_same_color : ℚ := 9 / 2601

theorem same_color_shoe_probability :
  (num_pairs : ℚ) / (total_shoes - 1) / (total_shoes.choose 2) = prob_same_color := by
  sorry

end NUMINAMATH_CALUDE_same_color_shoe_probability_l2492_249291


namespace NUMINAMATH_CALUDE_no_integer_solutions_l2492_249292

theorem no_integer_solutions : ¬∃ (x y z : ℤ),
  (x^2 - 4*x*y + 3*y^2 - z^2 = 25) ∧
  (-x^2 + 4*y*z + 3*z^2 = 36) ∧
  (x^2 + 2*x*y + 9*z^2 = 121) := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l2492_249292


namespace NUMINAMATH_CALUDE_jerrys_breakfast_calories_l2492_249278

/-- Calculates the total calories in Jerry's breakfast. -/
theorem jerrys_breakfast_calories :
  let pancake_calories : ℕ := 7 * 120
  let bacon_calories : ℕ := 3 * 100
  let orange_juice_calories : ℕ := 2 * 300
  let cereal_calories : ℕ := 200
  let muffin_calories : ℕ := 350
  pancake_calories + bacon_calories + orange_juice_calories + cereal_calories + muffin_calories = 2290 :=
by sorry

end NUMINAMATH_CALUDE_jerrys_breakfast_calories_l2492_249278


namespace NUMINAMATH_CALUDE_triangle_properties_l2492_249272

-- Define the triangle ABC
def A : ℝ × ℝ := (-4, 0)
def B : ℝ × ℝ := (0, 2)
def C : ℝ × ℝ := (2, -2)

-- Define the altitude line equation
def altitude_equation (x y : ℝ) : Prop :=
  2 * x + y - 2 = 0

-- Define the circumcircle equation
def circumcircle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 4 * x + 4 * y - 8 = 0

-- Theorem statement
theorem triangle_properties :
  (∀ x y : ℝ, altitude_equation x y ↔ 
    (x - A.1) * (B.2 - C.2) = (y - A.2) * (B.1 - C.1)) ∧
  (∀ x y : ℝ, circumcircle_equation x y ↔ 
    (x - A.1)^2 + (y - A.2)^2 = (x - B.1)^2 + (y - B.2)^2 ∧
    (x - B.1)^2 + (y - B.2)^2 = (x - C.1)^2 + (y - C.2)^2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l2492_249272


namespace NUMINAMATH_CALUDE_sphere_cylinder_equal_area_l2492_249242

theorem sphere_cylinder_equal_area (r : ℝ) : 
  (4 * Real.pi * r^2 = 2 * Real.pi * 6 * 12) → r = 6 := by
  sorry

end NUMINAMATH_CALUDE_sphere_cylinder_equal_area_l2492_249242


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2492_249266

theorem sum_of_roots_quadratic : ∃ (x₁ x₂ : ℝ),
  x₁^2 - 7*x₁ + 12 = 0 ∧
  x₂^2 - 7*x₂ + 12 = 0 ∧
  x₁ + x₂ = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2492_249266


namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_l2492_249296

/-- Defines the quadratic equation kx^2 - x - 1 = 0 -/
def quadratic_equation (k x : ℝ) : Prop :=
  k * x^2 - x - 1 = 0

/-- Defines when a quadratic equation has real roots -/
def has_real_roots (k : ℝ) : Prop :=
  ∃ x : ℝ, quadratic_equation k x

/-- Theorem stating the condition for the quadratic equation to have real roots -/
theorem quadratic_real_roots_condition (k : ℝ) :
  has_real_roots k ↔ k ≥ -1/4 ∧ k ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_l2492_249296
