import Mathlib

namespace NUMINAMATH_CALUDE_probability_of_red_ball_l2343_234386

/-- The probability of drawing a red ball from a bag with red and black balls -/
theorem probability_of_red_ball (red_balls black_balls : ℕ) : 
  red_balls = 3 → black_balls = 9 → 
  (red_balls : ℚ) / (red_balls + black_balls) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_red_ball_l2343_234386


namespace NUMINAMATH_CALUDE_factor_calculation_l2343_234384

theorem factor_calculation : ∃ f : ℝ, (2 * 9 + 6) * f = 72 ∧ f = 3 := by
  sorry

end NUMINAMATH_CALUDE_factor_calculation_l2343_234384


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l2343_234355

-- Define the universe U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define the set M
def M : Set ℝ := {x | x < -2 ∨ x > 8}

-- State the theorem
theorem complement_of_M_in_U : 
  Set.compl M = {x : ℝ | -2 ≤ x ∧ x ≤ 8} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l2343_234355


namespace NUMINAMATH_CALUDE_fraction_inequality_l2343_234359

theorem fraction_inequality (a b : ℕ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : (a : ℚ) / b < (a + 1 : ℚ) / (b + 1)) : 
  (2012 * a : ℚ) / b > 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l2343_234359


namespace NUMINAMATH_CALUDE_intersection_point_P_equation_l2343_234344

-- Define the curves C₁ and C₂
def C₁ (ρ θ : ℝ) : Prop := ρ * Real.cos θ = 3
def C₂ (ρ θ : ℝ) : Prop := ρ = 4 * Real.cos θ ∧ 0 ≤ θ ∧ θ < Real.pi / 2

-- Theorem for the intersection point
theorem intersection_point :
  ∃ ρ θ, C₁ ρ θ ∧ C₂ ρ θ ∧ ρ = 2 * Real.sqrt 3 ∧ θ = Real.pi / 6 :=
sorry

-- Define the relationship between Q and P
def Q_P_relation (ρ_Q θ_Q ρ_P θ_P : ℝ) : Prop :=
  C₂ ρ_Q θ_Q ∧ ρ_Q = (2/3) * ρ_P ∧ θ_Q = θ_P

-- Theorem for the polar coordinate equation of P
theorem P_equation :
  ∀ ρ_P θ_P, (∃ ρ_Q θ_Q, Q_P_relation ρ_Q θ_Q ρ_P θ_P) →
  ρ_P = 10 * Real.cos θ_P ∧ 0 ≤ θ_P ∧ θ_P < Real.pi / 2 :=
sorry

end NUMINAMATH_CALUDE_intersection_point_P_equation_l2343_234344


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_squares_minimum_l2343_234323

theorem quadratic_roots_sum_squares_minimum (a : ℝ) 
  (x₁ x₂ : ℝ) (h₁ : x₁^2 + 2*a*x₁ + a^2 + 4*a - 2 = 0) 
  (h₂ : x₂^2 + 2*a*x₂ + a^2 + 4*a - 2 = 0) 
  (h₃ : x₁ ≠ x₂) :
  x₁^2 + x₂^2 ≥ 1/2 ∧ 
  (x₁^2 + x₂^2 = 1/2 ↔ a = 1/2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_squares_minimum_l2343_234323


namespace NUMINAMATH_CALUDE_blue_pill_cost_l2343_234364

/-- Represents the cost of pills for Alice's medication --/
structure PillCosts where
  red : ℝ
  blue : ℝ
  yellow : ℝ

/-- The conditions of Alice's medication costs --/
def medication_conditions (costs : PillCosts) : Prop :=
  costs.blue = costs.red + 3 ∧
  costs.yellow = 2 * costs.red - 2 ∧
  21 * (costs.red + costs.blue + costs.yellow) = 924

/-- Theorem stating the cost of the blue pill --/
theorem blue_pill_cost (costs : PillCosts) :
  medication_conditions costs → costs.blue = 13.75 := by
  sorry


end NUMINAMATH_CALUDE_blue_pill_cost_l2343_234364


namespace NUMINAMATH_CALUDE_purely_imaginary_modulus_l2343_234365

theorem purely_imaginary_modulus (a : ℝ) :
  let z : ℂ := a + Complex.I
  (z.re = 0) → Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_modulus_l2343_234365


namespace NUMINAMATH_CALUDE_shipbuilding_profit_optimization_l2343_234354

def R (x : ℕ) : ℤ := 3700 * x + 45 * x^2 - 10 * x^3
def C (x : ℕ) : ℤ := 460 * x + 500

def p (x : ℕ) : ℤ := R x - C x

def Mp (x : ℕ) : ℤ := p (x + 1) - p x

theorem shipbuilding_profit_optimization (x : ℕ) (h : 1 ≤ x ∧ x ≤ 20) :
  p x = -10 * x^3 + 45 * x^2 + 3240 * x - 500 ∧
  (∀ y : ℕ, 1 ≤ y ∧ y ≤ 20 → p y ≤ p 12) ∧
  Mp x = -30 * x^2 + 60 * x + 3275 ∧
  (∀ y z : ℕ, 1 ≤ y ∧ y < z ∧ z ≤ 19 → Mp z ≤ Mp y) :=
sorry

end NUMINAMATH_CALUDE_shipbuilding_profit_optimization_l2343_234354


namespace NUMINAMATH_CALUDE_fixed_point_on_circle_l2343_234371

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 3 = 1

-- Define the endpoints of the transverse axis
def A₁ : ℝ × ℝ := (2, 0)
def A₂ : ℝ × ℝ := (-2, 0)

-- Define a point P on the hyperbola
def P : ℝ × ℝ → Prop := λ p => 
  hyperbola p.1 p.2 ∧ p ≠ A₁ ∧ p ≠ A₂

-- Define the line x = 1
def line_x_1 (x y : ℝ) : Prop := x = 1

-- Define the intersection points M₁ and M₂
def M₁ (p : ℝ × ℝ) : ℝ × ℝ := sorry
def M₂ (p : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define a circle with diameter M₁M₂
def circle_M₁M₂ (p c : ℝ × ℝ) : Prop := sorry

-- The theorem to prove
theorem fixed_point_on_circle : 
  ∃ c : ℝ × ℝ, ∀ p : ℝ × ℝ, P p → circle_M₁M₂ p c := by sorry

end NUMINAMATH_CALUDE_fixed_point_on_circle_l2343_234371


namespace NUMINAMATH_CALUDE_largest_n_for_equation_l2343_234331

theorem largest_n_for_equation : 
  (∀ n : ℕ, n > 4 → ¬∃ x y z : ℕ+, n^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 4*x + 4*y + 4*z - 12) ∧
  (∃ x y z : ℕ+, 4^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 4*x + 4*y + 4*z - 12) := by
  sorry

end NUMINAMATH_CALUDE_largest_n_for_equation_l2343_234331


namespace NUMINAMATH_CALUDE_complex_power_magnitude_l2343_234332

theorem complex_power_magnitude : Complex.abs ((4/5 : ℂ) + (3/5 : ℂ) * Complex.I) ^ 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_magnitude_l2343_234332


namespace NUMINAMATH_CALUDE_solution_range_l2343_234328

theorem solution_range (m : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ (1 - m) * x = 2 - 3 * x) → m < 4 := by
sorry

end NUMINAMATH_CALUDE_solution_range_l2343_234328


namespace NUMINAMATH_CALUDE_polyhedron_volume_l2343_234356

theorem polyhedron_volume (prism_volume : ℝ) (pyramid_base_side : ℝ) (pyramid_height : ℝ) :
  prism_volume = Real.sqrt 2 - 1 →
  pyramid_base_side = 1 →
  pyramid_height = 1 / 2 →
  prism_volume + 2 * (1 / 3 * pyramid_base_side^2 * pyramid_height) = Real.sqrt 2 - 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_polyhedron_volume_l2343_234356


namespace NUMINAMATH_CALUDE_trigonometric_equation_l2343_234316

theorem trigonometric_equation (α β : Real) 
  (h : (Real.cos α)^3 / Real.cos β + (Real.sin α)^3 / Real.sin β = 2) :
  (Real.sin β)^3 / Real.sin α + (Real.cos β)^3 / Real.cos α = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equation_l2343_234316


namespace NUMINAMATH_CALUDE_min_operations_to_250_l2343_234385

/-- Represents the possible operations: adding 1 or multiplying by 2 -/
inductive Operation
  | addOne
  | multiplyTwo

/-- Applies an operation to a natural number -/
def applyOperation (n : ℕ) (op : Operation) : ℕ :=
  match op with
  | Operation.addOne => n + 1
  | Operation.multiplyTwo => n * 2

/-- Checks if a sequence of operations transforms 1 into the target -/
def isValidSequence (target : ℕ) (ops : List Operation) : Prop :=
  ops.foldl applyOperation 1 = target

/-- The minimum number of operations needed to transform 1 into 250 -/
def minOperations : ℕ := 12

/-- Theorem stating that the minimum number of operations to reach 250 from 1 is 12 -/
theorem min_operations_to_250 :
  (∃ (ops : List Operation), isValidSequence 250 ops ∧ ops.length = minOperations) ∧
  (∀ (ops : List Operation), isValidSequence 250 ops → ops.length ≥ minOperations) :=
sorry

end NUMINAMATH_CALUDE_min_operations_to_250_l2343_234385


namespace NUMINAMATH_CALUDE_least_possible_b_l2343_234322

-- Define Fibonacci sequence
def isFibonacci (n : ℕ) : Prop :=
  ∃ k : ℕ, n = (((1 + Real.sqrt 5) / 2) ^ k - ((1 - Real.sqrt 5) / 2) ^ k) / Real.sqrt 5

-- Define the problem
theorem least_possible_b (a b : ℕ) : 
  (a + b = 90) →  -- Sum of acute angles in a right triangle
  (a > b) →       -- a is greater than b
  isFibonacci a → -- a is a Fibonacci number
  isFibonacci b → -- b is a Fibonacci number
  (∀ c : ℕ, c < b → (c + a ≠ 90 ∨ ¬isFibonacci c ∨ ¬isFibonacci a)) →
  b = 1 :=
by sorry

end NUMINAMATH_CALUDE_least_possible_b_l2343_234322


namespace NUMINAMATH_CALUDE_geometry_relationships_l2343_234320

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (line_parallel : Line → Line → Prop)
variable (line_perpendicular : Line → Line → Prop)

-- State the theorem
theorem geometry_relationships 
  (l m : Line) (a : Plane) (h_diff : l ≠ m) :
  (perpendicular l a ∧ contains a m → line_perpendicular l m) ∧
  (perpendicular l a ∧ line_parallel l m → perpendicular m a) ∧
  ¬(parallel l a ∧ contains a m → line_parallel l m) ∧
  ¬(parallel l a ∧ parallel m a → line_parallel l m) :=
sorry

end NUMINAMATH_CALUDE_geometry_relationships_l2343_234320


namespace NUMINAMATH_CALUDE_power_sum_division_equals_seventeen_l2343_234382

theorem power_sum_division_equals_seventeen :
  1^234 + 4^6 / 4^4 = 17 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_division_equals_seventeen_l2343_234382


namespace NUMINAMATH_CALUDE_at_most_one_root_l2343_234309

theorem at_most_one_root (f : ℝ → ℝ) (h : ∀ a b, a < b → f a < f b) :
  ∃! x, f x = 0 :=
sorry

end NUMINAMATH_CALUDE_at_most_one_root_l2343_234309


namespace NUMINAMATH_CALUDE_original_savings_calculation_l2343_234339

theorem original_savings_calculation (savings : ℚ) : 
  (3 / 4 : ℚ) * savings + (1 / 4 : ℚ) * savings = savings ∧ 
  (1 / 4 : ℚ) * savings = 240 → 
  savings = 960 := by
  sorry

end NUMINAMATH_CALUDE_original_savings_calculation_l2343_234339


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l2343_234370

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The origin point (0, 0, 0) -/
def origin : Point3D := ⟨0, 0, 0⟩

/-- Symmetric point with respect to the origin -/
def symmetricPoint (p : Point3D) : Point3D :=
  ⟨-p.x, -p.y, -p.z⟩

theorem symmetric_point_coordinates :
  let p : Point3D := ⟨1, -2, 1⟩
  let q : Point3D := symmetricPoint p
  q = ⟨-1, 2, -1⟩ := by sorry

end NUMINAMATH_CALUDE_symmetric_point_coordinates_l2343_234370


namespace NUMINAMATH_CALUDE_sunzi_wood_measurement_l2343_234342

/-- Proves the equation for the wood measurement problem from "The Mathematical Classic of Sunzi" -/
theorem sunzi_wood_measurement (x : ℝ) 
  (h1 : ∃ (rope_length : ℝ), rope_length = x + 4.5) 
  (h2 : ∃ (half_rope : ℝ), half_rope = x - 1 ∧ half_rope = (x + 4.5) / 2) : 
  (x + 4.5) / 2 = x - 1 := by
  sorry

end NUMINAMATH_CALUDE_sunzi_wood_measurement_l2343_234342


namespace NUMINAMATH_CALUDE_perfect_square_completion_l2343_234317

theorem perfect_square_completion (ε : ℝ) (hε : ε > 0) : 
  ∃ x : ℝ, ∃ y : ℝ, 
    (12.86 * 12.86 + 12.86 * x + 0.14 * 0.14 = y * y) ∧ 
    (|x - 0.28| < ε) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_completion_l2343_234317


namespace NUMINAMATH_CALUDE_mets_fan_count_l2343_234387

/-- Represents the number of fans for each team -/
structure FanCount where
  yankees : ℕ
  mets : ℕ
  red_sox : ℕ

/-- The total number of fans in the town -/
def total_fans : ℕ := 330

/-- The fan count satisfies the given ratios and total -/
def is_valid_fan_count (fc : FanCount) : Prop :=
  3 * fc.mets = 2 * fc.yankees ∧
  4 * fc.red_sox = 5 * fc.mets ∧
  fc.yankees + fc.mets + fc.red_sox = total_fans

theorem mets_fan_count (fc : FanCount) (h : is_valid_fan_count fc) : fc.mets = 88 := by
  sorry


end NUMINAMATH_CALUDE_mets_fan_count_l2343_234387


namespace NUMINAMATH_CALUDE_num_divisors_3960_l2343_234394

/-- The number of positive divisors of a natural number n -/
def num_positive_divisors (n : ℕ) : ℕ := sorry

/-- Theorem: The number of positive divisors of 3960 is 48 -/
theorem num_divisors_3960 : num_positive_divisors 3960 = 48 := by sorry

end NUMINAMATH_CALUDE_num_divisors_3960_l2343_234394


namespace NUMINAMATH_CALUDE_imperial_examination_middle_volume_l2343_234337

/-- The number of candidates admitted in the Middle volume given a total number of candidates and a proportion -/
def middle_volume_candidates (total : ℕ) (south north middle : ℕ) : ℕ :=
  total * middle /(south + north + middle)

/-- Theorem stating that given 100 total candidates and a proportion of 11:7:2,
    the number of candidates in the Middle volume is 10 -/
theorem imperial_examination_middle_volume :
  middle_volume_candidates 100 11 7 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_imperial_examination_middle_volume_l2343_234337


namespace NUMINAMATH_CALUDE_smallest_valid_number_l2343_234333

def is_valid (n : ℕ+) : Prop :=
  (Finset.card (Nat.divisors n) = 144) ∧
  (∃ k : ℕ, ∀ i : Fin 10, (k + i) ∈ Nat.divisors n)

theorem smallest_valid_number : 
  (is_valid 110880) ∧ (∀ m : ℕ+, m < 110880 → ¬(is_valid m)) :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_number_l2343_234333


namespace NUMINAMATH_CALUDE_conference_room_seating_l2343_234392

/-- Represents the seating arrangement in a conference room. -/
structure ConferenceRoom where
  totalPeople : ℕ
  rowCapacities : List ℕ
  allSeatsFilled : totalPeople = rowCapacities.sum

/-- Checks if a conference room arrangement is valid. -/
def isValidArrangement (room : ConferenceRoom) : Prop :=
  ∀ capacity ∈ room.rowCapacities, capacity = 9 ∨ capacity = 10

/-- The main theorem about the conference room seating arrangement. -/
theorem conference_room_seating
  (room : ConferenceRoom)
  (validArrangement : isValidArrangement room)
  (h : room.totalPeople = 54) :
  (room.rowCapacities.filter (· = 10)).length = 0 := by
  sorry


end NUMINAMATH_CALUDE_conference_room_seating_l2343_234392


namespace NUMINAMATH_CALUDE_cuboid_diagonal_l2343_234358

theorem cuboid_diagonal (x y z : ℝ) 
  (h1 : y * z = Real.sqrt 2)
  (h2 : z * x = Real.sqrt 3)
  (h3 : x * y = Real.sqrt 6) :
  Real.sqrt (x^2 + y^2 + z^2) = Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_cuboid_diagonal_l2343_234358


namespace NUMINAMATH_CALUDE_percentage_of_125_equal_to_70_l2343_234321

theorem percentage_of_125_equal_to_70 : 
  ∃ p : ℝ, p * 125 = 70 ∧ p = 56 / 100 := by sorry

end NUMINAMATH_CALUDE_percentage_of_125_equal_to_70_l2343_234321


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_l2343_234390

theorem closest_integer_to_cube_root (n : ℕ) : 
  ∃ (m : ℤ), ∀ (k : ℤ), |k - (5^3 + 9^3 : ℝ)^(1/3)| ≥ |m - (5^3 + 9^3 : ℝ)^(1/3)| ∧ m = 9 := by
  sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_l2343_234390


namespace NUMINAMATH_CALUDE_infinitely_many_divisible_numbers_l2343_234348

theorem infinitely_many_divisible_numbers :
  ∃ (a : ℕ → ℕ), (∀ n : ℕ, a n ∣ 2^(a n) + 3^(a n)) ∧
                 (∀ n : ℕ, a n < a (n + 1)) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_divisible_numbers_l2343_234348


namespace NUMINAMATH_CALUDE_profit_sharing_ratio_l2343_234325

/-- Represents the investment and time period for a partner --/
structure Partner where
  investment : ℕ
  months : ℕ

/-- Calculates the effective capital of a partner --/
def effectiveCapital (p : Partner) : ℕ := p.investment * p.months

/-- Calculates the ratio of two numbers --/
def ratio (a b : ℕ) : ℕ × ℕ :=
  let gcd := a.gcd b
  (a / gcd, b / gcd)

/-- Theorem stating the profit sharing ratio between P and Q --/
theorem profit_sharing_ratio (p q : Partner)
  (h1 : p.investment = 4000)
  (h2 : p.months = 12)
  (h3 : q.investment = 9000)
  (h4 : q.months = 8) :
  ratio (effectiveCapital p) (effectiveCapital q) = (2, 3) := by
  sorry

#check profit_sharing_ratio

end NUMINAMATH_CALUDE_profit_sharing_ratio_l2343_234325


namespace NUMINAMATH_CALUDE_intersection_condition_l2343_234399

def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 5}
def B (p : ℝ) : Set ℝ := {x : ℝ | p + 1 ≤ x ∧ x ≤ 2*p - 1}

theorem intersection_condition (p : ℝ) : A ∩ B p = B p ↔ p ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_condition_l2343_234399


namespace NUMINAMATH_CALUDE_auction_bids_l2343_234383

theorem auction_bids (initial_price final_price : ℕ) (price_increase : ℕ) (num_bidders : ℕ) :
  initial_price = 15 →
  final_price = 65 →
  price_increase = 5 →
  num_bidders = 2 →
  (final_price - initial_price) / price_increase / num_bidders = 5 :=
by sorry

end NUMINAMATH_CALUDE_auction_bids_l2343_234383


namespace NUMINAMATH_CALUDE_relationship_between_a_b_c_l2343_234307

theorem relationship_between_a_b_c : ∀ (a b c : ℝ),
  a = -(1^2) →
  b = (3 - Real.pi)^0 →
  c = (-0.25)^2023 * 4^2024 →
  b > a ∧ a > c := by
  sorry

end NUMINAMATH_CALUDE_relationship_between_a_b_c_l2343_234307


namespace NUMINAMATH_CALUDE_circle_chord_intersection_l2343_234381

theorem circle_chord_intersection (r : ℝ) (chord_length : ℝ) :
  r = 8 →
  chord_length = 12 →
  ∃ (ak kb : ℝ),
    ak = 8 - 2 * Real.sqrt 7 ∧
    kb = 8 + 2 * Real.sqrt 7 ∧
    ak + kb = 2 * r :=
by sorry

end NUMINAMATH_CALUDE_circle_chord_intersection_l2343_234381


namespace NUMINAMATH_CALUDE_fourth_root_simplification_l2343_234341

theorem fourth_root_simplification : Real.sqrt (Real.sqrt (2^8 * 3^4 * 11^0)) = 12 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_simplification_l2343_234341


namespace NUMINAMATH_CALUDE_weekly_sales_equals_63_l2343_234305

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The average number of hamburgers sold per day -/
def avg_daily_sales : ℕ := 9

/-- The total number of hamburgers sold in a week -/
def total_weekly_sales : ℕ := days_in_week * avg_daily_sales

theorem weekly_sales_equals_63 : total_weekly_sales = 63 := by
  sorry

end NUMINAMATH_CALUDE_weekly_sales_equals_63_l2343_234305


namespace NUMINAMATH_CALUDE_root_square_minus_three_x_minus_one_l2343_234315

theorem root_square_minus_three_x_minus_one (m : ℝ) : 
  m^2 - 3*m - 1 = 0 → 2*m^2 - 6*m = 2 := by
  sorry

end NUMINAMATH_CALUDE_root_square_minus_three_x_minus_one_l2343_234315


namespace NUMINAMATH_CALUDE_smallest_factorial_with_43_zeroes_l2343_234368

/-- Count the number of trailing zeroes in n! -/
def trailingZeroes (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- Theorem: 175 is the smallest positive integer k such that k! ends in at least 43 zeroes -/
theorem smallest_factorial_with_43_zeroes :
  (∀ k : ℕ, k > 0 → k < 175 → trailingZeroes k < 43) ∧ trailingZeroes 175 = 43 := by
  sorry

#eval trailingZeroes 175  -- Should output 43

end NUMINAMATH_CALUDE_smallest_factorial_with_43_zeroes_l2343_234368


namespace NUMINAMATH_CALUDE_total_jelly_beans_l2343_234350

/-- The number of vanilla jelly beans -/
def vanilla_jb : ℕ := 120

/-- The number of grape jelly beans -/
def grape_jb : ℕ := 5 * vanilla_jb + 50

/-- The total number of jelly beans -/
def total_jb : ℕ := vanilla_jb + grape_jb

theorem total_jelly_beans : total_jb = 770 := by
  sorry

end NUMINAMATH_CALUDE_total_jelly_beans_l2343_234350


namespace NUMINAMATH_CALUDE_max_profit_l2343_234380

noncomputable def T (x : ℝ) : ℝ :=
  if 1 ≤ x ∧ x < 6 then (9*x - 2*x^2) / (6 - x)
  else 0

theorem max_profit :
  ∃ (x : ℝ), 1 ≤ x ∧ x < 6 ∧ T x = 3 ∧ ∀ y, T y ≤ T x :=
by
  sorry

end NUMINAMATH_CALUDE_max_profit_l2343_234380


namespace NUMINAMATH_CALUDE_shower_usage_solution_l2343_234375

/-- The water usage for Roman and Remy's showers -/
def shower_usage (R : ℝ) : Prop :=
  let remy_usage := 3 * R + 1
  R + remy_usage = 33 ∧ remy_usage = 25

/-- Theorem stating that there exists a value for Roman's usage satisfying the conditions -/
theorem shower_usage_solution : ∃ R : ℝ, shower_usage R := by
  sorry

end NUMINAMATH_CALUDE_shower_usage_solution_l2343_234375


namespace NUMINAMATH_CALUDE_clinic_cats_count_l2343_234391

theorem clinic_cats_count (dog_cost cat_cost dog_count total_cost : ℕ) 
  (h1 : dog_cost = 60)
  (h2 : cat_cost = 40)
  (h3 : dog_count = 20)
  (h4 : total_cost = 3600)
  : ∃ cat_count : ℕ, dog_cost * dog_count + cat_cost * cat_count = total_cost ∧ cat_count = 60 := by
  sorry

end NUMINAMATH_CALUDE_clinic_cats_count_l2343_234391


namespace NUMINAMATH_CALUDE_range_of_a_l2343_234357

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x - a| ≤ 1 → x^2 - 5*x + 4 ≤ 0) →
  a ∈ Set.Icc 2 3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2343_234357


namespace NUMINAMATH_CALUDE_triangle_property_l2343_234327

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem about a specific triangle with given conditions -/
theorem triangle_property (t : Triangle) 
  (h1 : Real.sin t.A + Real.sin t.B = 5/4 * Real.sin t.C)
  (h2 : t.a + t.b + t.c = 9)
  (h3 : 1/2 * t.a * t.b * Real.sin t.C = 3 * Real.sin t.C) :
  t.c = 4 ∧ Real.cos t.C = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_property_l2343_234327


namespace NUMINAMATH_CALUDE_probability_two_green_balls_l2343_234330

def total_balls : ℕ := 12
def red_balls : ℕ := 3
def yellow_balls : ℕ := 5
def green_balls : ℕ := 4
def drawn_balls : ℕ := 3

theorem probability_two_green_balls :
  (Nat.choose green_balls 2 * Nat.choose (total_balls - green_balls) 1) /
  Nat.choose total_balls drawn_balls = 12 / 55 :=
by sorry

end NUMINAMATH_CALUDE_probability_two_green_balls_l2343_234330


namespace NUMINAMATH_CALUDE_factorial_bounds_l2343_234347

theorem factorial_bounds (n : ℕ) (h : n ≥ 1) : 2^(n-1) ≤ n! ∧ n! ≤ n^n := by
  sorry

end NUMINAMATH_CALUDE_factorial_bounds_l2343_234347


namespace NUMINAMATH_CALUDE_increasing_quadratic_function_parameter_range_l2343_234388

theorem increasing_quadratic_function_parameter_range 
  (f : ℝ → ℝ) (a : ℝ) 
  (h1 : ∀ x, f x = x^2 - 2*a*x + 2) 
  (h2 : ∀ x y, x ≥ 3 → y ≥ 3 → x < y → f x < f y) : 
  a ∈ Set.Iic 3 := by
sorry

end NUMINAMATH_CALUDE_increasing_quadratic_function_parameter_range_l2343_234388


namespace NUMINAMATH_CALUDE_ball_returns_to_start_l2343_234366

/-- The number of girls in the circle -/
def n : ℕ := 13

/-- The number of positions to advance in each throw -/
def k : ℕ := 5

/-- The function that determines the next girl to receive the ball -/
def next (x : ℕ) : ℕ := (x + k) % n

/-- The sequence of girls who receive the ball, starting from position 1 -/
def ball_sequence : ℕ → ℕ
  | 0 => 1
  | i + 1 => next (ball_sequence i)

theorem ball_returns_to_start :
  ∃ m : ℕ, m > 0 ∧ ball_sequence m = 1 ∧ ∀ i < m, ball_sequence i ≠ 1 :=
sorry

end NUMINAMATH_CALUDE_ball_returns_to_start_l2343_234366


namespace NUMINAMATH_CALUDE_smallest_n_fourth_fifth_power_l2343_234373

theorem smallest_n_fourth_fifth_power : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (x : ℕ), 3 * n = x^4) ∧ 
  (∃ (y : ℕ), 2 * n = y^5) ∧ 
  (∀ (m : ℕ), m > 0 → 
    (∃ (a : ℕ), 3 * m = a^4) → 
    (∃ (b : ℕ), 2 * m = b^5) → 
    m ≥ 6912) ∧
  n = 6912 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_fourth_fifth_power_l2343_234373


namespace NUMINAMATH_CALUDE_triangle_side_length_l2343_234367

theorem triangle_side_length (a b c : ℝ) (A : ℝ) : 
  A = Real.pi / 3 →  -- 60 degrees in radians
  a * b = 11 →
  a + b = 7 →
  a > c ∧ c > b →
  c = 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2343_234367


namespace NUMINAMATH_CALUDE_find_other_number_l2343_234361

theorem find_other_number (A B : ℕ) (hA : A = 24) (hHCF : Nat.gcd A B = 14) (hLCM : Nat.lcm A B = 312) : B = 182 := by
  sorry

end NUMINAMATH_CALUDE_find_other_number_l2343_234361


namespace NUMINAMATH_CALUDE_quadrilateral_circumscription_l2343_234311

def can_be_circumscribed (a b c d : ℝ) : Prop :=
  a + b + c + d = 360 ∧ a + c = 180 ∧ b + d = 180

theorem quadrilateral_circumscription :
  (∃ (x : ℝ), can_be_circumscribed (2*x) (4*x) (5*x) (3*x)) ∧
  (∀ (x : ℝ), ¬can_be_circumscribed (5*x) (7*x) (8*x) (9*x)) := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_circumscription_l2343_234311


namespace NUMINAMATH_CALUDE_sarah_bottle_caps_l2343_234395

/-- The total number of bottle caps Sarah has after buying more -/
def total_bottle_caps (initial : ℕ) (bought : ℕ) : ℕ :=
  initial + bought

/-- Theorem stating that Sarah has 29 bottle caps in total -/
theorem sarah_bottle_caps : total_bottle_caps 26 3 = 29 := by
  sorry

end NUMINAMATH_CALUDE_sarah_bottle_caps_l2343_234395


namespace NUMINAMATH_CALUDE_count_divides_sum_product_l2343_234303

def divides_sum_product (n : ℕ+) : Prop :=
  (n.val * (n.val + 1) / 2) ∣ (10 * n.val)

theorem count_divides_sum_product :
  ∃ (S : Finset ℕ+), (∀ n, n ∈ S ↔ divides_sum_product n) ∧ S.card = 5 :=
sorry

end NUMINAMATH_CALUDE_count_divides_sum_product_l2343_234303


namespace NUMINAMATH_CALUDE_correct_operation_l2343_234301

theorem correct_operation (x y : ℝ) : 4 * x^3 * y^2 * (x^2 * y^3) = 4 * x^5 * y^5 := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l2343_234301


namespace NUMINAMATH_CALUDE_car_price_increase_l2343_234314

/-- Proves that given a discount and profit on the original price, 
    we can calculate the percentage increase on the discounted price. -/
theorem car_price_increase 
  (original_price : ℝ) 
  (discount_rate : ℝ) 
  (profit_rate : ℝ) 
  (h1 : discount_rate = 0.40) 
  (h2 : profit_rate = 0.08000000000000007) : 
  let discounted_price := original_price * (1 - discount_rate)
  let selling_price := original_price * (1 + profit_rate)
  let increase_rate := (selling_price - discounted_price) / discounted_price
  increase_rate = 0.8000000000000001 := by
  sorry

end NUMINAMATH_CALUDE_car_price_increase_l2343_234314


namespace NUMINAMATH_CALUDE_marbles_cost_l2343_234397

def total_spent : ℚ := 20.52
def football_cost : ℚ := 4.95
def baseball_cost : ℚ := 6.52

theorem marbles_cost : total_spent - (football_cost + baseball_cost) = 9.05 := by
  sorry

end NUMINAMATH_CALUDE_marbles_cost_l2343_234397


namespace NUMINAMATH_CALUDE_perpendicular_line_proof_l2343_234334

-- Define the given line
def given_line (x y : ℝ) : Prop := 2 * x + y - 5 = 0

-- Define the point that the perpendicular line passes through
def point : ℝ × ℝ := (0, 3)

-- Define the perpendicular line
def perpendicular_line (x y : ℝ) : Prop := x - 2 * y + 6 = 0

-- Theorem statement
theorem perpendicular_line_proof :
  (∀ x y : ℝ, perpendicular_line x y → given_line x y → x * x + y * y = 0) ∧
  perpendicular_line point.1 point.2 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_proof_l2343_234334


namespace NUMINAMATH_CALUDE_train_speed_proof_l2343_234302

/-- Proves that the speed of a train is 23.4 km/hr given specific conditions -/
theorem train_speed_proof (train_length : Real) (crossing_time : Real) (total_length : Real) :
  train_length = 180 →
  crossing_time = 30 →
  total_length = 195 →
  (total_length / crossing_time) * 3.6 = 23.4 :=
by
  sorry

#check train_speed_proof

end NUMINAMATH_CALUDE_train_speed_proof_l2343_234302


namespace NUMINAMATH_CALUDE_peanut_mixture_proof_l2343_234318

/-- Given the following:
    - 10 pounds of Virginia peanuts cost $3.50 per pound
    - Spanish peanuts cost $3.00 per pound
    - The desired mixture should cost $3.40 per pound
    Prove that 2.5 pounds of Spanish peanuts should be used to create the mixture. -/
theorem peanut_mixture_proof (virginia_weight : ℝ) (virginia_price : ℝ) (spanish_price : ℝ) 
  (mixture_price : ℝ) (spanish_weight : ℝ) :
  virginia_weight = 10 →
  virginia_price = 3.5 →
  spanish_price = 3 →
  mixture_price = 3.4 →
  spanish_weight = 2.5 →
  (virginia_weight * virginia_price + spanish_weight * spanish_price) / (virginia_weight + spanish_weight) = mixture_price :=
by sorry

end NUMINAMATH_CALUDE_peanut_mixture_proof_l2343_234318


namespace NUMINAMATH_CALUDE_prime_square_mod_twelve_l2343_234338

theorem prime_square_mod_twelve (p : ℕ) (hp : Prime p) (hp_gt_3 : p > 3) :
  p ^ 2 % 12 = 1 := by
sorry

end NUMINAMATH_CALUDE_prime_square_mod_twelve_l2343_234338


namespace NUMINAMATH_CALUDE_fraction_equality_l2343_234346

theorem fraction_equality (a b : ℝ) (h : a ≠ b) : (-a + b) / (a - b) = -1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2343_234346


namespace NUMINAMATH_CALUDE_expression_value_l2343_234306

theorem expression_value (x y : ℤ) (hx : x = -5) (hy : y = 8) :
  2 * (x - y)^2 - x * y = 378 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2343_234306


namespace NUMINAMATH_CALUDE_y_minus_x_value_l2343_234340

theorem y_minus_x_value (x y : ℝ) (hx : |x| = 5) (hy : |y| = 9) (hxy : x < y) :
  y - x = 4 ∨ y - x = 14 := by
  sorry

end NUMINAMATH_CALUDE_y_minus_x_value_l2343_234340


namespace NUMINAMATH_CALUDE_correct_systematic_sampling_l2343_234345

/-- Represents the systematic sampling of students -/
def systematic_sampling (total_students : ℕ) (students_to_select : ℕ) : List ℕ :=
  sorry

/-- The theorem stating the correct systematic sampling for the given problem -/
theorem correct_systematic_sampling :
  systematic_sampling 50 5 = [5, 15, 25, 35, 45] := by
  sorry

end NUMINAMATH_CALUDE_correct_systematic_sampling_l2343_234345


namespace NUMINAMATH_CALUDE_percentage_to_decimal_two_percent_to_decimal_l2343_234329

theorem percentage_to_decimal (p : ℚ) : p / 100 = p * (1 / 100) := by sorry

theorem two_percent_to_decimal : (2 : ℚ) / 100 = 0.02 := by sorry

end NUMINAMATH_CALUDE_percentage_to_decimal_two_percent_to_decimal_l2343_234329


namespace NUMINAMATH_CALUDE_complex_power_six_l2343_234353

theorem complex_power_six : (1 + 2 * Complex.I) ^ 6 = 117 + 44 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_power_six_l2343_234353


namespace NUMINAMATH_CALUDE_min_distance_MN_min_distance_is_two_l2343_234376

noncomputable section

def f (x : ℝ) : ℝ := Real.exp x - (1/2) * x^2
def g (x : ℝ) : ℝ := x - 1

def M (x₁ : ℝ) : ℝ × ℝ := (x₁, f x₁)
def N (x₂ : ℝ) : ℝ × ℝ := (x₂, g x₂)

theorem min_distance_MN (x₁ x₂ : ℝ) (h₁ : x₁ ≥ 0) (h₂ : x₂ > 0) 
  (h₃ : f x₁ = g x₂) : 
  ∀ y₁ y₂ : ℝ, y₁ ≥ 0 → y₂ > 0 → f y₁ = g y₂ → 
  |x₂ - x₁| ≤ |y₂ - y₁| := by sorry

theorem min_distance_is_two (x₁ x₂ : ℝ) (h₁ : x₁ ≥ 0) (h₂ : x₂ > 0) 
  (h₃ : f x₁ = g x₂) : 
  |x₂ - x₁| = 2 := by sorry

end NUMINAMATH_CALUDE_min_distance_MN_min_distance_is_two_l2343_234376


namespace NUMINAMATH_CALUDE_drinks_left_calculation_l2343_234351

-- Define the initial amounts of drinks
def initial_coke : ℝ := 35.5
def initial_cider : ℝ := 27.2

-- Define the amount of coke drunk
def coke_drunk : ℝ := 1.75

-- Theorem statement
theorem drinks_left_calculation :
  initial_coke + initial_cider - coke_drunk = 60.95 := by
  sorry

end NUMINAMATH_CALUDE_drinks_left_calculation_l2343_234351


namespace NUMINAMATH_CALUDE_max_both_writers_and_editors_l2343_234319

/-- Represents the number of people at the newspaper conference --/
def total_people : ℕ := 150

/-- Represents the number of writers at the conference --/
def writers : ℕ := 50

/-- Represents the number of editors at the conference --/
def editors : ℕ := 66

/-- Represents the number of people who are both writers and editors --/
def both (x : ℕ) : ℕ := x

/-- Represents the number of people who are neither writers nor editors --/
def neither (x : ℕ) : ℕ := 3 * x

/-- States that the number of editors is more than 65 --/
axiom editors_more_than_65 : editors > 65

/-- Theorem stating that the maximum number of people who are both writers and editors is 17 --/
theorem max_both_writers_and_editors :
  ∃ (x : ℕ), x ≤ 17 ∧
  total_people = writers + editors - both x + neither x ∧
  ∀ (y : ℕ), y > x →
    total_people ≠ writers + editors - both y + neither y :=
sorry

end NUMINAMATH_CALUDE_max_both_writers_and_editors_l2343_234319


namespace NUMINAMATH_CALUDE_ways_to_sum_2022_is_338_l2343_234362

/-- The number of ways to write 2022 as a sum of 2s and 3s -/
def ways_to_sum_2022 : ℕ :=
  Nat.succ (337 - 0)

/-- Theorem stating that the number of ways to write 2022 as a sum of 2s and 3s is 338 -/
theorem ways_to_sum_2022_is_338 : ways_to_sum_2022 = 338 := by
  sorry

#eval ways_to_sum_2022

end NUMINAMATH_CALUDE_ways_to_sum_2022_is_338_l2343_234362


namespace NUMINAMATH_CALUDE_equation_equivalence_l2343_234313

theorem equation_equivalence (x y : ℕ) : 
  (∃ (a b c d : ℕ), x = a + 2*b + 3*c + 7*d ∧ y = b + 2*c + 5*d) ↔ 
  5*x ≥ 7*y := by
sorry

end NUMINAMATH_CALUDE_equation_equivalence_l2343_234313


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_13231_l2343_234363

theorem largest_prime_factor_of_13231 : ∃ p : ℕ, 
  Nat.Prime p ∧ p ∣ 13231 ∧ ∀ q : ℕ, Nat.Prime q → q ∣ 13231 → q ≤ p :=
by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_13231_l2343_234363


namespace NUMINAMATH_CALUDE_shopkeeper_cloth_cost_price_l2343_234398

/-- Given a shopkeeper sells cloth at a loss, calculate the cost price per meter. -/
theorem shopkeeper_cloth_cost_price
  (total_meters : ℕ)
  (total_selling_price : ℕ)
  (loss_per_meter : ℕ)
  (h1 : total_meters = 400)
  (h2 : total_selling_price = 18000)
  (h3 : loss_per_meter = 5) :
  total_selling_price / total_meters + loss_per_meter = 50 := by
  sorry

#check shopkeeper_cloth_cost_price

end NUMINAMATH_CALUDE_shopkeeper_cloth_cost_price_l2343_234398


namespace NUMINAMATH_CALUDE_partition_naturals_l2343_234389

theorem partition_naturals (c : ℚ) (hc : c > 0) (hc_ne_one : c ≠ 1) :
  ∃ (A B : Set ℕ), (A ∪ B = Set.univ) ∧ (A ∩ B = ∅) ∧
  (∀ (a₁ a₂ : ℕ), a₁ ∈ A → a₂ ∈ A → a₁ ≠ 0 → a₂ ≠ 0 → (a₁ : ℚ) / a₂ ≠ c) ∧
  (∀ (b₁ b₂ : ℕ), b₁ ∈ B → b₂ ∈ B → b₁ ≠ 0 → b₂ ≠ 0 → (b₁ : ℚ) / b₂ ≠ c) :=
by sorry

end NUMINAMATH_CALUDE_partition_naturals_l2343_234389


namespace NUMINAMATH_CALUDE_number_subtraction_problem_l2343_234312

theorem number_subtraction_problem : ∃! x : ℝ, 0.4 * x - 11 = 23 := by
  sorry

end NUMINAMATH_CALUDE_number_subtraction_problem_l2343_234312


namespace NUMINAMATH_CALUDE_parallel_vectors_k_l2343_234372

def a : Fin 2 → ℝ := ![2, -1]
def b : Fin 2 → ℝ := ![1, 1]
def c : Fin 2 → ℝ := ![-5, 1]

theorem parallel_vectors_k (k : ℝ) :
  (∀ i : Fin 2, (a i + k * b i) * c (1 - i) = (a (1 - i) + k * b (1 - i)) * c i) →
  k = 1/2 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_l2343_234372


namespace NUMINAMATH_CALUDE_incorrect_locus_definition_l2343_234336

-- Define the type for points in our space
variable {X : Type*}

-- Define the locus as a set of points
variable (locus : Set X)

-- Define the condition as a predicate on points
variable (condition : X → Prop)

-- Statement to be proven incorrect
theorem incorrect_locus_definition :
  ¬(∀ x : X, condition x → x ∈ locus) ∧
  (∃ x : X, x ∈ locus ∧ condition x) →
  ¬(∀ x : X, x ∈ locus ↔ condition x) :=
by sorry

end NUMINAMATH_CALUDE_incorrect_locus_definition_l2343_234336


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l2343_234360

theorem least_addition_for_divisibility (n : ℕ) : 
  let x := 278
  (∀ y : ℕ, y < x → ¬((1056 + y) % 23 = 0 ∧ (1056 + y) % 29 = 0)) ∧
  ((1056 + x) % 23 = 0 ∧ (1056 + x) % 29 = 0) := by
  sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l2343_234360


namespace NUMINAMATH_CALUDE_smallest_n_for_geometric_sum_l2343_234379

/-- The sum of the first n terms of a geometric sequence with first term a and common ratio r -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The statement to prove -/
theorem smallest_n_for_geometric_sum : 
  ∀ n : ℕ, n > 0 → 
    (geometric_sum (1/3) (1/3) n = 80/243 ↔ n ≥ 5) ∧ 
    (geometric_sum (1/3) (1/3) 5 = 80/243) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_geometric_sum_l2343_234379


namespace NUMINAMATH_CALUDE_polynomial_remainder_l2343_234374

theorem polynomial_remainder (x : ℝ) : 
  (x^5 + 2*x^2 + 1) % (x - 2) = 41 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l2343_234374


namespace NUMINAMATH_CALUDE_point_transformation_to_third_quadrant_l2343_234324

/-- Given a point (a, b) in the fourth quadrant, prove that (a/b, 2b-a) is in the third quadrant -/
theorem point_transformation_to_third_quadrant (a b : ℝ) 
  (h1 : a > 0) (h2 : b < 0) : (a / b < 0) ∧ (2 * b - a < 0) := by
  sorry

end NUMINAMATH_CALUDE_point_transformation_to_third_quadrant_l2343_234324


namespace NUMINAMATH_CALUDE_x_cube_x_square_order_l2343_234310

theorem x_cube_x_square_order (x : ℝ) (h : -1 < x ∧ x < 0) : x < x^3 ∧ x^3 < x^2 := by
  sorry

end NUMINAMATH_CALUDE_x_cube_x_square_order_l2343_234310


namespace NUMINAMATH_CALUDE_value_of_a_fourth_plus_reciprocal_l2343_234393

theorem value_of_a_fourth_plus_reciprocal (a : ℝ) (h : (a + 1/a)^2 = 5) :
  a^4 + 1/a^4 = 7 := by
sorry

end NUMINAMATH_CALUDE_value_of_a_fourth_plus_reciprocal_l2343_234393


namespace NUMINAMATH_CALUDE_oliver_stickers_l2343_234304

theorem oliver_stickers (initial_stickers : ℕ) (used_fraction : ℚ) (kept_stickers : ℕ) 
  (h1 : initial_stickers = 135)
  (h2 : used_fraction = 1/3)
  (h3 : kept_stickers = 54) :
  let remaining_stickers := initial_stickers - (used_fraction * initial_stickers).num
  let given_stickers := remaining_stickers - kept_stickers
  (given_stickers : ℚ) / remaining_stickers = 2/5 := by
sorry

end NUMINAMATH_CALUDE_oliver_stickers_l2343_234304


namespace NUMINAMATH_CALUDE_paityn_red_hats_l2343_234396

/-- Proves that Paityn has 20 red hats given the problem conditions -/
theorem paityn_red_hats :
  ∀ (paityn_red : ℕ) (paityn_blue : ℕ) (zola_red : ℕ) (zola_blue : ℕ),
  paityn_blue = 24 →
  zola_red = (4 * paityn_red) / 5 →
  zola_blue = 2 * paityn_blue →
  paityn_red + paityn_blue + zola_red + zola_blue = 108 →
  paityn_red = 20 := by
sorry


end NUMINAMATH_CALUDE_paityn_red_hats_l2343_234396


namespace NUMINAMATH_CALUDE_smallest_c_value_l2343_234308

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

theorem smallest_c_value (a b c d e : ℕ) :
  a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e ∧
  is_perfect_square (b + c + d) ∧
  is_perfect_cube (a + b + c + d + e) →
  c ≥ 675 ∧ ∃ (a' b' c' d' e' : ℕ),
    a' + 1 = b' ∧ b' + 1 = c' ∧ c' + 1 = d' ∧ d' + 1 = e' ∧
    is_perfect_square (b' + c' + d') ∧
    is_perfect_cube (a' + b' + c' + d' + e') ∧
    c' = 675 :=
by sorry

end NUMINAMATH_CALUDE_smallest_c_value_l2343_234308


namespace NUMINAMATH_CALUDE_g_50_eq_zero_l2343_234326

/-- A function satisfying the given functional equation -/
def FunctionalEquation (g : ℝ → ℝ) : Prop :=
  ∀ x y, x > 0 → y > 0 → x * g y + y * g x = g (x * y)

/-- The main theorem stating that g(50) = 0 for any function satisfying the functional equation -/
theorem g_50_eq_zero (g : ℝ → ℝ) (h : FunctionalEquation g) : g 50 = 0 := by
  sorry

end NUMINAMATH_CALUDE_g_50_eq_zero_l2343_234326


namespace NUMINAMATH_CALUDE_total_sightings_l2343_234352

def animal_sightings (january february march : ℕ) : Prop :=
  february = 3 * january ∧ march = february / 2

theorem total_sightings (january : ℕ) (h : animal_sightings january (3 * january) ((3 * january) / 2)) :
  january + (3 * january) + ((3 * january) / 2) = 143 :=
by
  sorry

#check total_sightings 26

end NUMINAMATH_CALUDE_total_sightings_l2343_234352


namespace NUMINAMATH_CALUDE_trapezoid_area_l2343_234300

/-- The area of a trapezoid with height x, one base 4x, and the other base (4x - 2x) is 3x² -/
theorem trapezoid_area (x : ℝ) : 
  let height := x
  let base1 := 4 * x
  let base2 := 4 * x - 2 * x
  (base1 + base2) / 2 * height = 3 * x^2 :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_area_l2343_234300


namespace NUMINAMATH_CALUDE_binomial_15_4_l2343_234343

theorem binomial_15_4 : Nat.choose 15 4 = 1365 := by sorry

end NUMINAMATH_CALUDE_binomial_15_4_l2343_234343


namespace NUMINAMATH_CALUDE_ab2_minus_41_equals_591_l2343_234369

/-- Given two single-digit numbers A and B, where AB2 is a three-digit number,
    prove that when A = 6 and B = 2, the equation AB2 - 41 = 591 is valid. -/
theorem ab2_minus_41_equals_591 (A B : Nat) : 
  A < 10 → B < 10 → 100 ≤ A * 100 + B * 10 + 2 → A * 100 + B * 10 + 2 < 1000 →
  A = 6 → B = 2 → A * 100 + B * 10 + 2 - 41 = 591 := by
sorry

end NUMINAMATH_CALUDE_ab2_minus_41_equals_591_l2343_234369


namespace NUMINAMATH_CALUDE_gcd_840_1764_gcd_561_255_l2343_234378

-- Part 1: GCD of 840 and 1764
theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by sorry

-- Part 2: GCD of 561 and 255
theorem gcd_561_255 : Nat.gcd 561 255 = 51 := by sorry

end NUMINAMATH_CALUDE_gcd_840_1764_gcd_561_255_l2343_234378


namespace NUMINAMATH_CALUDE_sarah_sells_more_than_tamara_l2343_234335

/-- Represents the bake sale competition between Tamara and Sarah -/
structure BakeSale where
  -- Tamara's baked goods
  tamara_brownie_pans : ℕ
  tamara_cookie_trays : ℕ
  tamara_brownie_pieces_per_pan : ℕ
  tamara_cookie_pieces_per_tray : ℕ
  tamara_small_brownie_price : ℚ
  tamara_large_brownie_price : ℚ
  tamara_cookie_price : ℚ
  tamara_small_brownies_sold : ℕ

  -- Sarah's baked goods
  sarah_cupcake_batches : ℕ
  sarah_muffin_dozens : ℕ
  sarah_cupcakes_per_batch : ℕ
  sarah_chocolate_cupcake_price : ℚ
  sarah_vanilla_cupcake_price : ℚ
  sarah_strawberry_cupcake_price : ℚ
  sarah_muffin_price : ℚ
  sarah_chocolate_cupcakes_sold : ℕ
  sarah_vanilla_cupcakes_sold : ℕ

/-- Calculates the total sales for Tamara -/
def tamara_total_sales (bs : BakeSale) : ℚ :=
  let total_brownies := bs.tamara_brownie_pans * bs.tamara_brownie_pieces_per_pan
  let large_brownies_sold := total_brownies - bs.tamara_small_brownies_sold
  let total_cookies := bs.tamara_cookie_trays * bs.tamara_cookie_pieces_per_tray
  bs.tamara_small_brownies_sold * bs.tamara_small_brownie_price +
  large_brownies_sold * bs.tamara_large_brownie_price +
  total_cookies * bs.tamara_cookie_price

/-- Calculates the total sales for Sarah -/
def sarah_total_sales (bs : BakeSale) : ℚ :=
  let total_cupcakes := bs.sarah_cupcake_batches * bs.sarah_cupcakes_per_batch
  let strawberry_cupcakes_sold := total_cupcakes - bs.sarah_chocolate_cupcakes_sold - bs.sarah_vanilla_cupcakes_sold
  let total_muffins := bs.sarah_muffin_dozens * 12
  total_muffins * bs.sarah_muffin_price +
  bs.sarah_chocolate_cupcakes_sold * bs.sarah_chocolate_cupcake_price +
  bs.sarah_vanilla_cupcakes_sold * bs.sarah_vanilla_cupcake_price +
  strawberry_cupcakes_sold * bs.sarah_strawberry_cupcake_price

/-- Theorem stating the difference in sales between Sarah and Tamara -/
theorem sarah_sells_more_than_tamara (bs : BakeSale) :
  bs.tamara_brownie_pans = 2 ∧
  bs.tamara_cookie_trays = 3 ∧
  bs.tamara_brownie_pieces_per_pan = 8 ∧
  bs.tamara_cookie_pieces_per_tray = 12 ∧
  bs.tamara_small_brownie_price = 2 ∧
  bs.tamara_large_brownie_price = 3 ∧
  bs.tamara_cookie_price = 3/2 ∧
  bs.tamara_small_brownies_sold = 4 ∧
  bs.sarah_cupcake_batches = 3 ∧
  bs.sarah_muffin_dozens = 2 ∧
  bs.sarah_cupcakes_per_batch = 10 ∧
  bs.sarah_chocolate_cupcake_price = 5/2 ∧
  bs.sarah_vanilla_cupcake_price = 2 ∧
  bs.sarah_strawberry_cupcake_price = 11/4 ∧
  bs.sarah_muffin_price = 7/4 ∧
  bs.sarah_chocolate_cupcakes_sold = 7 ∧
  bs.sarah_vanilla_cupcakes_sold = 8 →
  sarah_total_sales bs - tamara_total_sales bs = 75/4 := by
  sorry

end NUMINAMATH_CALUDE_sarah_sells_more_than_tamara_l2343_234335


namespace NUMINAMATH_CALUDE_gift_card_value_l2343_234349

theorem gift_card_value (coffee_price : ℝ) (pounds_bought : ℝ) (remaining_balance : ℝ) :
  coffee_price = 8.58 →
  pounds_bought = 4 →
  remaining_balance = 35.68 →
  coffee_price * pounds_bought + remaining_balance = 70 :=
by
  sorry

end NUMINAMATH_CALUDE_gift_card_value_l2343_234349


namespace NUMINAMATH_CALUDE_two_digit_number_equals_three_times_square_of_units_digit_l2343_234377

theorem two_digit_number_equals_three_times_square_of_units_digit :
  ∀ n : ℕ, 10 ≤ n ∧ n < 100 →
    (n = 3 * (n % 10)^2) ↔ (n = 12 ∨ n = 75) := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_equals_three_times_square_of_units_digit_l2343_234377
