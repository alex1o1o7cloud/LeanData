import Mathlib

namespace NUMINAMATH_CALUDE_parallel_vectors_component_l3940_394016

/-- Given two vectors a and b in ℝ², prove that if a is parallel to b,
    then the first component of a must be -1. -/
theorem parallel_vectors_component (a b : ℝ × ℝ) :
  a.1 = m ∧ a.2 = Real.sqrt 3 ∧ b.1 = Real.sqrt 3 ∧ b.2 = -3 ∧
  ∃ (k : ℝ), a = k • b →
  m = -1 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_component_l3940_394016


namespace NUMINAMATH_CALUDE_min_value_a_plus_2b_l3940_394011

theorem min_value_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 3/b = 1) :
  ∃ (min : ℝ), (∀ x y, x > 0 → y > 0 → 1/x + 3/y = 1 → x + 2*y ≥ min) ∧ (a + 2*b = min) :=
by
  -- The minimum value is 7 + 2√6
  let min := 7 + 2 * Real.sqrt 6
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_min_value_a_plus_2b_l3940_394011


namespace NUMINAMATH_CALUDE_range_of_a_range_of_m_l3940_394012

-- Define sets A, B, and C
def A (a : ℝ) := {x : ℝ | a^2 - a*x + x - 1 = 0}
def B (m : ℝ) := {x : ℝ | x^2 + x + m = 0}
def C := {x : ℝ | Real.sqrt (x^2) = x}

-- Theorem for part (1)
theorem range_of_a (a : ℝ) : A a ∪ C = C → a ∈ Set.Icc (-1) 1 ∪ Set.Ioi 1 := by
  sorry

-- Theorem for part (2)
theorem range_of_m (m : ℝ) : C ∩ B m = ∅ → m ∈ Set.Ioi 0 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_range_of_m_l3940_394012


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l3940_394038

-- Define the quadratic function
def f (a x : ℝ) : ℝ := a * x^2 - 2*x + 1

-- Define the property of having a non-empty solution set
def has_solution (a : ℝ) : Prop := ∃ x, f a x < 0

-- State the theorem
theorem necessary_but_not_sufficient_condition :
  (∀ a, has_solution a → a ≤ 1) ∧
  ¬(∀ a, a ≤ 1 → has_solution a) :=
sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l3940_394038


namespace NUMINAMATH_CALUDE_vector_sum_proof_l3940_394036

/-- Given vectors a = (2,3) and b = (-1,2), prove their sum is (1,5) -/
theorem vector_sum_proof :
  let a : Fin 2 → ℝ := ![2, 3]
  let b : Fin 2 → ℝ := ![-1, 2]
  (a + b) = ![1, 5] := by sorry

end NUMINAMATH_CALUDE_vector_sum_proof_l3940_394036


namespace NUMINAMATH_CALUDE_initial_amount_of_liquid_a_l3940_394070

/-- Given a mixture of liquids A and B, prove the initial amount of A. -/
theorem initial_amount_of_liquid_a (a b : ℝ) : 
  a > 0 → b > 0 →  -- Ensure positive quantities
  a / b = 4 / 1 →  -- Initial ratio
  (a - 24) / (b - 6 + 30) = 2 / 3 →  -- New ratio after replacement
  a = 48 := by
sorry

end NUMINAMATH_CALUDE_initial_amount_of_liquid_a_l3940_394070


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l3940_394026

theorem inscribed_circle_radius (A₁ A₂ : ℝ) (h1 : A₁ > 0) (h2 : A₂ > 0) : 
  (A₁ + A₂ = π * 8^2) →
  (A₂ = (A₁ + (A₁ + A₂)) / 2) →
  A₁ = π * ((8 * Real.sqrt 3) / 3)^2 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l3940_394026


namespace NUMINAMATH_CALUDE_quadrilateral_inequality_l3940_394044

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define an interior point
def interior_point (q : Quadrilateral) (O : ℝ × ℝ) : Prop := sorry

-- Define distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define area of a triangle
def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Define s₃ and s₄
def s₃ (q : Quadrilateral) (O : ℝ × ℝ) : ℝ :=
  distance O q.A + distance O q.B + distance O q.C + distance O q.D

def s₄ (q : Quadrilateral) : ℝ :=
  distance q.A q.B + distance q.B q.C + distance q.C q.D + distance q.D q.A

-- State the theorem
theorem quadrilateral_inequality (q : Quadrilateral) (O : ℝ × ℝ) :
  interior_point q O →
  triangle_area O q.A q.B = triangle_area O q.C q.D →
  s₃ q O ≥ (1/2) * s₄ q ∧ s₃ q O ≤ s₄ q :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_inequality_l3940_394044


namespace NUMINAMATH_CALUDE_arithmetic_sequence_second_term_l3940_394071

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_second_term
  (a : ℕ → ℤ)
  (h_arithmetic : ArithmeticSequence a)
  (h_10th : a 10 = 15)
  (h_11th : a 11 = 18) :
  a 2 = -9 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_second_term_l3940_394071


namespace NUMINAMATH_CALUDE_sum_of_squares_and_square_of_sum_l3940_394073

theorem sum_of_squares_and_square_of_sum : (3 + 7)^2 + (3^2 + 7^2 + 5^2) = 183 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_and_square_of_sum_l3940_394073


namespace NUMINAMATH_CALUDE_extra_bananas_l3940_394028

theorem extra_bananas (total_children absent_children original_bananas : ℕ) 
  (h1 : total_children = 840)
  (h2 : absent_children = 420)
  (h3 : original_bananas = 2) : 
  let present_children := total_children - absent_children
  let total_bananas := total_children * original_bananas
  let actual_bananas := total_bananas / present_children
  actual_bananas - original_bananas = 2 := by sorry

end NUMINAMATH_CALUDE_extra_bananas_l3940_394028


namespace NUMINAMATH_CALUDE_optimal_rectangle_dimensions_l3940_394078

-- Define the rectangle dimensions
def width : ℝ := 14.625
def length : ℝ := 34.25

-- Define the conditions
def area_constraint (w l : ℝ) : Prop := w * l ≥ 500
def length_constraint (w l : ℝ) : Prop := l = 2 * w + 5

-- Define the perimeter function
def perimeter (w l : ℝ) : ℝ := 2 * (w + l)

theorem optimal_rectangle_dimensions :
  area_constraint width length ∧
  length_constraint width length ∧
  ∀ w l : ℝ, w > 0 → l > 0 →
    area_constraint w l →
    length_constraint w l →
    perimeter width length ≤ perimeter w l :=
sorry

end NUMINAMATH_CALUDE_optimal_rectangle_dimensions_l3940_394078


namespace NUMINAMATH_CALUDE_pie_eating_contest_l3940_394077

theorem pie_eating_contest (erik_pie frank_pie : Float) 
  (h1 : erik_pie = 0.6666666666666666)
  (h2 : frank_pie = 0.3333333333333333) :
  erik_pie - frank_pie = 0.3333333333333333 := by
  sorry

end NUMINAMATH_CALUDE_pie_eating_contest_l3940_394077


namespace NUMINAMATH_CALUDE_robot_returns_to_start_l3940_394098

/-- Represents a robot's movement pattern -/
structure RobotMovement where
  turn_interval : ℕ  -- Time in seconds between turns
  turn_angle : ℕ     -- Angle of turn in degrees

/-- Represents the state of the robot -/
structure RobotState where
  position : ℤ × ℤ   -- (x, y) coordinates
  direction : ℕ      -- 0: North, 1: East, 2: South, 3: West

/-- Calculates the new position after one movement -/
def move (state : RobotState) : RobotState :=
  match state.direction with
  | 0 => { state with position := (state.position.1, state.position.2 + 1) }
  | 1 => { state with position := (state.position.1 + 1, state.position.2) }
  | 2 => { state with position := (state.position.1, state.position.2 - 1) }
  | 3 => { state with position := (state.position.1 - 1, state.position.2) }
  | _ => state

/-- Calculates the new direction after turning -/
def turn (state : RobotState) : RobotState :=
  { state with direction := (state.direction + 1) % 4 }

/-- Simulates the robot's movement for a given number of seconds -/
def simulate (movement : RobotMovement) (initial_state : RobotState) (time : ℕ) : RobotState :=
  if time = 0 then initial_state
  else
    let new_state := if time % movement.turn_interval = 0 
                     then turn (move initial_state)
                     else move initial_state
    simulate movement new_state (time - 1)

/-- Theorem: The robot returns to its starting point after 6 minutes -/
theorem robot_returns_to_start (movement : RobotMovement) 
  (h1 : movement.turn_interval = 15)
  (h2 : movement.turn_angle = 90) :
  let initial_state : RobotState := ⟨(0, 0), 0⟩
  let final_state := simulate movement initial_state (6 * 60)
  final_state.position = initial_state.position :=
by sorry


end NUMINAMATH_CALUDE_robot_returns_to_start_l3940_394098


namespace NUMINAMATH_CALUDE_sqrt_sum_fractions_l3940_394010

theorem sqrt_sum_fractions : Real.sqrt (1/4 + 1/9) = Real.sqrt 13 / 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_fractions_l3940_394010


namespace NUMINAMATH_CALUDE_gas_station_lighter_price_l3940_394093

/-- The cost of a single lighter at the gas station -/
def gas_station_price : ℝ := 1.75

/-- The cost of a pack of 12 lighters on Amazon -/
def amazon_pack_price : ℝ := 5

/-- The number of lighters in a pack on Amazon -/
def lighters_per_pack : ℕ := 12

/-- The number of lighters Amanda is considering buying -/
def total_lighters : ℕ := 24

/-- The amount saved by buying online instead of at the gas station -/
def savings : ℝ := 32

theorem gas_station_lighter_price :
  gas_station_price = 1.75 ∧
  amazon_pack_price * (total_lighters / lighters_per_pack) + savings =
    gas_station_price * total_lighters :=
by sorry

end NUMINAMATH_CALUDE_gas_station_lighter_price_l3940_394093


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3940_394056

-- Define the sets M and N
def M : Set ℝ := {x | ∃ t : ℝ, x = 2^t}
def N : Set ℝ := {x | ∃ t : ℝ, x = Real.sin t}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = Set.Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3940_394056


namespace NUMINAMATH_CALUDE_modulus_z_is_sqrt_5_l3940_394055

theorem modulus_z_is_sqrt_5 (z : ℂ) (h : (1 + Complex.I) * z = 3 + Complex.I) :
  Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_z_is_sqrt_5_l3940_394055


namespace NUMINAMATH_CALUDE_quadruplet_solution_l3940_394062

theorem quadruplet_solution (x y z w : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0)
  (h_eq : (x*y*z + 1)/(x + 1) = (y*z*w + 1)/(y + 1) ∧
          (y*z*w + 1)/(y + 1) = (z*w*x + 1)/(z + 1) ∧
          (z*w*x + 1)/(z + 1) = (w*x*y + 1)/(w + 1))
  (h_sum : x + y + z + w = 48) :
  x = 12 ∧ y = 12 ∧ z = 12 ∧ w = 12 := by
sorry

end NUMINAMATH_CALUDE_quadruplet_solution_l3940_394062


namespace NUMINAMATH_CALUDE_franklins_gathering_theorem_l3940_394046

/-- Represents the number of handshakes in Franklin's gathering --/
def franklins_gathering_handshakes (num_couples : ℕ) : ℕ :=
  let num_men := num_couples
  let num_women := num_couples
  let handshakes_among_men := num_men * (num_men - 1 + num_women - 1) / 2
  let franklins_handshakes := num_women
  handshakes_among_men + franklins_handshakes

/-- Theorem stating that the number of handshakes in Franklin's gathering with 15 couples is 225 --/
theorem franklins_gathering_theorem :
  franklins_gathering_handshakes 15 = 225 := by
  sorry

#eval franklins_gathering_handshakes 15

end NUMINAMATH_CALUDE_franklins_gathering_theorem_l3940_394046


namespace NUMINAMATH_CALUDE_sum_of_decimals_l3940_394061

theorem sum_of_decimals :
  5.256 + 2.89 + 3.75 = 11.96 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_decimals_l3940_394061


namespace NUMINAMATH_CALUDE_power_subtraction_l3940_394014

theorem power_subtraction (x a b : ℝ) (ha : x^a = 3) (hb : x^b = 5) : x^(a - b) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_power_subtraction_l3940_394014


namespace NUMINAMATH_CALUDE_heart_properties_l3940_394053

def heart (x y : ℝ) : ℝ := x^2 - 2*x*y + y^2

theorem heart_properties :
  (∀ x y : ℝ, heart x y = heart y x) ∧
  (∃ x y : ℝ, 2 * (heart x y) ≠ heart (2*x) (2*y)) ∧
  (∀ x : ℝ, heart x 0 = x^2) ∧
  (∀ x : ℝ, heart x x = 0) ∧
  (∀ x y : ℝ, x ≠ y → heart x y > 0) :=
by sorry

end NUMINAMATH_CALUDE_heart_properties_l3940_394053


namespace NUMINAMATH_CALUDE_squirrels_and_nuts_l3940_394080

theorem squirrels_and_nuts (squirrels : ℕ) (nuts : ℕ) : 
  squirrels = 4 → squirrels = nuts + 2 → nuts = 2 := by
  sorry

end NUMINAMATH_CALUDE_squirrels_and_nuts_l3940_394080


namespace NUMINAMATH_CALUDE_remainder_67_power_67_plus_67_mod_68_l3940_394003

theorem remainder_67_power_67_plus_67_mod_68 : (67^67 + 67) % 68 = 66 := by
  sorry

end NUMINAMATH_CALUDE_remainder_67_power_67_plus_67_mod_68_l3940_394003


namespace NUMINAMATH_CALUDE_units_digit_of_p_plus_two_l3940_394020

def is_positive_even (n : ℕ) : Prop := n > 0 ∧ n % 2 = 0

def has_positive_units_digit (n : ℕ) : Prop := n % 10 > 0

def units_digit (n : ℕ) : ℕ := n % 10

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem units_digit_of_p_plus_two (p q : ℕ) (x : ℕ+) :
  is_positive_even p →
  is_positive_even q →
  has_positive_units_digit p →
  has_positive_units_digit q →
  units_digit (p^3) - units_digit (p^2) = 0 →
  sum_of_digits p % q = 0 →
  p^(x : ℕ) = q →
  units_digit (p + 2) = 8 :=
by sorry

end NUMINAMATH_CALUDE_units_digit_of_p_plus_two_l3940_394020


namespace NUMINAMATH_CALUDE_intersection_when_a_is_3_empty_intersection_iff_a_in_range_l3940_394023

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | 2 - a ≤ x ∧ x ≤ 2 + a}
def B : Set ℝ := {x | x ≤ 1 ∨ 4 ≤ x}

-- Theorem 1
theorem intersection_when_a_is_3 :
  A 3 ∩ B = {x | -1 ≤ x ∧ x ≤ 1 ∨ 4 ≤ x ∧ x ≤ 5} := by sorry

-- Theorem 2
theorem empty_intersection_iff_a_in_range (a : ℝ) :
  (a > 0) → (A a ∩ B = ∅ ↔ 0 < a ∧ a < 1) := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_3_empty_intersection_iff_a_in_range_l3940_394023


namespace NUMINAMATH_CALUDE_ring_arrangement_count_l3940_394041

def number_of_rings : ℕ := 10
def rings_to_arrange : ℕ := 6
def number_of_fingers : ℕ := 4

def ring_arrangements (n k : ℕ) : ℕ :=
  Nat.choose n k * Nat.factorial k * Nat.choose (k + number_of_fingers) number_of_fingers

theorem ring_arrangement_count :
  ring_arrangements number_of_rings rings_to_arrange = 31752000 :=
by sorry

end NUMINAMATH_CALUDE_ring_arrangement_count_l3940_394041


namespace NUMINAMATH_CALUDE_expression_equals_one_l3940_394019

theorem expression_equals_one : 
  (50^2 - 9^2) / (40^2 - 8^2) * ((40 - 8) * (40 + 8)) / ((50 - 9) * (50 + 9)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_l3940_394019


namespace NUMINAMATH_CALUDE_factor_polynomial_l3940_394000

theorem factor_polynomial (x : ℝ) : 
  x^2 - 6*x + 9 - 64*x^4 = (8*x^2 + x - 3)*(-8*x^2 + x - 3) := by
sorry

end NUMINAMATH_CALUDE_factor_polynomial_l3940_394000


namespace NUMINAMATH_CALUDE_bicycle_wheels_count_l3940_394096

theorem bicycle_wheels_count (num_bicycles num_tricycles tricycle_wheels total_wheels : ℕ) 
  (h1 : num_bicycles = 24)
  (h2 : num_tricycles = 14)
  (h3 : tricycle_wheels = 3)
  (h4 : total_wheels = 90)
  (h5 : total_wheels = num_bicycles * bicycle_wheels + num_tricycles * tricycle_wheels) :
  bicycle_wheels = 2 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_wheels_count_l3940_394096


namespace NUMINAMATH_CALUDE_abs_diff_eq_diff_implies_le_l3940_394090

theorem abs_diff_eq_diff_implies_le (x y : ℝ) :
  |x - y| = y - x → x ≤ y := by
  sorry

end NUMINAMATH_CALUDE_abs_diff_eq_diff_implies_le_l3940_394090


namespace NUMINAMATH_CALUDE_brown_eyed_brunettes_l3940_394005

theorem brown_eyed_brunettes (total : ℕ) (blonde_blue : ℕ) (brunette : ℕ) (brown : ℕ)
  (h1 : total = 50)
  (h2 : blonde_blue = 14)
  (h3 : brunette = 31)
  (h4 : brown = 18) :
  brunette + blonde_blue - (total - brown) = 13 := by
  sorry

end NUMINAMATH_CALUDE_brown_eyed_brunettes_l3940_394005


namespace NUMINAMATH_CALUDE_arithmetic_problem_l3940_394083

theorem arithmetic_problem : 4 * (8 - 3) / 2 - 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_problem_l3940_394083


namespace NUMINAMATH_CALUDE_sign_of_a_equals_sign_of_r_l3940_394030

-- Define the variables and their properties
variable (x y : ℝ → ℝ) -- x and y are real-valued functions
variable (r : ℝ) -- r is the correlation coefficient
variable (a b : ℝ) -- a and b are coefficients in the regression line equation

-- Define the linear relationship and regression line
def linear_relationship (x y : ℝ → ℝ) : Prop :=
  ∃ (m c : ℝ), ∀ t, y t = m * (x t) + c

-- Define the regression line equation
def regression_line (x y : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ t, y t = a * (x t) + b

-- Define the correlation coefficient
def correlation_coefficient (x y : ℝ → ℝ) (r : ℝ) : Prop :=
  ∃ (cov_xy std_x std_y : ℝ), r = cov_xy / (std_x * std_y) ∧ std_x > 0 ∧ std_y > 0

-- State the theorem
theorem sign_of_a_equals_sign_of_r
  (h_linear : linear_relationship x y)
  (h_regression : regression_line x y a b)
  (h_correlation : correlation_coefficient x y r) :
  (a > 0 ↔ r > 0) ∧ (a < 0 ↔ r < 0) :=
sorry

end NUMINAMATH_CALUDE_sign_of_a_equals_sign_of_r_l3940_394030


namespace NUMINAMATH_CALUDE_max_m_value_l3940_394047

theorem max_m_value (m : ℝ) : 
  (∀ x : ℝ, x < m → x^2 - 2*x - 8 > 0) ∧ 
  (∃ x : ℝ, x^2 - 2*x - 8 > 0 ∧ x ≥ m) →
  m ≤ -2 :=
by sorry

end NUMINAMATH_CALUDE_max_m_value_l3940_394047


namespace NUMINAMATH_CALUDE_sum_of_three_pentagons_l3940_394060

/-- The value of a square -/
def square_value : ℚ := sorry

/-- The value of a pentagon -/
def pentagon_value : ℚ := sorry

/-- First equation: 3 squares + 2 pentagons = 27 -/
axiom eq1 : 3 * square_value + 2 * pentagon_value = 27

/-- Second equation: 2 squares + 3 pentagons = 25 -/
axiom eq2 : 2 * square_value + 3 * pentagon_value = 25

/-- Theorem: The sum of three pentagons equals 63/5 -/
theorem sum_of_three_pentagons : 3 * pentagon_value = 63 / 5 := by sorry

end NUMINAMATH_CALUDE_sum_of_three_pentagons_l3940_394060


namespace NUMINAMATH_CALUDE_dog_ratio_proof_l3940_394079

/-- Proves that for 12 dogs with 36 paws on the ground, split equally between those on back legs and all fours, the ratio of dogs on back legs to all fours is 1:1 -/
theorem dog_ratio_proof (total_dogs : ℕ) (total_paws : ℕ) 
  (h1 : total_dogs = 12) 
  (h2 : total_paws = 36) 
  (h3 : ∃ x y : ℕ, x + y = total_dogs ∧ x = y ∧ 2*x + 4*y = total_paws) : 
  ∃ x y : ℕ, x + y = total_dogs ∧ x = y ∧ x / y = 1 := by
  sorry

#check dog_ratio_proof

end NUMINAMATH_CALUDE_dog_ratio_proof_l3940_394079


namespace NUMINAMATH_CALUDE_land_development_break_even_l3940_394058

/-- Calculates the break-even price per lot given the total acreage, price per acre, and number of lots. -/
def breakEvenPricePerLot (totalAcres : ℕ) (pricePerAcre : ℕ) (numberOfLots : ℕ) : ℕ :=
  (totalAcres * pricePerAcre) / numberOfLots

/-- Proves that for 4 acres at $1,863 per acre split into 9 lots, the break-even price is $828 per lot. -/
theorem land_development_break_even :
  breakEvenPricePerLot 4 1863 9 = 828 := by
  sorry

#eval breakEvenPricePerLot 4 1863 9

end NUMINAMATH_CALUDE_land_development_break_even_l3940_394058


namespace NUMINAMATH_CALUDE_championship_games_l3940_394013

theorem championship_games (n : ℕ) (n_ge_2 : n ≥ 2) : 
  (n * (n - 1)) / 2 = (Finset.sum (Finset.range (n - 1)) (λ i => n - 1 - i)) :=
by sorry

end NUMINAMATH_CALUDE_championship_games_l3940_394013


namespace NUMINAMATH_CALUDE_vowel_count_l3940_394002

theorem vowel_count (total_alphabets : ℕ) (num_vowels : ℕ) (h1 : total_alphabets = 20) (h2 : num_vowels = 5) :
  total_alphabets / num_vowels = 4 := by
  sorry

end NUMINAMATH_CALUDE_vowel_count_l3940_394002


namespace NUMINAMATH_CALUDE_square_difference_division_problem_solution_l3940_394099

theorem square_difference_division (a b : ℕ) (h : a > b) :
  (a^2 - b^2) / (a - b) = a + b :=
by
  sorry

theorem problem_solution : (315^2 - 285^2) / 30 = 600 :=
by
  sorry

end NUMINAMATH_CALUDE_square_difference_division_problem_solution_l3940_394099


namespace NUMINAMATH_CALUDE_equation_solution_l3940_394025

theorem equation_solution (x : ℝ) : 
  x = 46 →
  (8 / (Real.sqrt (x - 10) - 10) + 
   2 / (Real.sqrt (x - 10) - 5) + 
   9 / (Real.sqrt (x - 10) + 5) + 
   15 / (Real.sqrt (x - 10) + 10) = 0) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3940_394025


namespace NUMINAMATH_CALUDE_members_playing_both_sports_l3940_394008

/-- Represents a sports club with members playing badminton and tennis -/
structure SportsClub where
  total : ℕ
  badminton : ℕ
  tennis : ℕ
  neither : ℕ

/-- Calculates the number of members playing both badminton and tennis -/
def both_sports (club : SportsClub) : ℕ :=
  club.badminton + club.tennis - (club.total - club.neither)

/-- Theorem stating the number of members playing both sports in the given club -/
theorem members_playing_both_sports (club : SportsClub)
  (h1 : club.total = 30)
  (h2 : club.badminton = 16)
  (h3 : club.tennis = 19)
  (h4 : club.neither = 2) :
  both_sports club = 7 := by
  sorry

end NUMINAMATH_CALUDE_members_playing_both_sports_l3940_394008


namespace NUMINAMATH_CALUDE_induction_principle_l3940_394049

theorem induction_principle (P : ℕ → Prop) :
  (∀ k, P k → P (k + 1)) →
  ¬ P 4 →
  ∀ n, n ≤ 4 → ¬ P n :=
sorry

end NUMINAMATH_CALUDE_induction_principle_l3940_394049


namespace NUMINAMATH_CALUDE_sixth_root_of_1061520150601_l3940_394072

theorem sixth_root_of_1061520150601 :
  let n : ℕ := 1061520150601
  ∃ (m : ℕ), m = 101 ∧ m^6 = n :=
by
  sorry

end NUMINAMATH_CALUDE_sixth_root_of_1061520150601_l3940_394072


namespace NUMINAMATH_CALUDE_student_distribution_ways_l3940_394015

def num_universities : ℕ := 8
def num_students : ℕ := 3
def num_selected_universities : ℕ := 2

theorem student_distribution_ways :
  (num_students.choose 1) * (num_selected_universities.choose 2) * (num_universities.choose 2) = 168 := by
  sorry

end NUMINAMATH_CALUDE_student_distribution_ways_l3940_394015


namespace NUMINAMATH_CALUDE_upper_bound_for_expression_l3940_394017

theorem upper_bound_for_expression (n : ℤ) : 
  (∃ ub : ℤ, 
    (ub = 40) ∧ 
    (∀ m : ℤ, 1 < 4*m + 7 → 4*m + 7 < ub) ∧
    (∃! (l : List ℤ), l.length = 10 ∧ 
      (∀ k : ℤ, k ∈ l ↔ (1 < 4*k + 7 ∧ 4*k + 7 < ub)))) :=
by sorry

end NUMINAMATH_CALUDE_upper_bound_for_expression_l3940_394017


namespace NUMINAMATH_CALUDE_contractor_absent_days_l3940_394097

/-- Represents the contractor's work scenario -/
structure ContractorScenario where
  totalDays : ℕ
  payPerWorkDay : ℚ
  finePerAbsentDay : ℚ
  totalPay : ℚ

/-- Calculates the number of absent days for a given contractor scenario -/
def absentDays (scenario : ContractorScenario) : ℚ :=
  (scenario.totalDays * scenario.payPerWorkDay - scenario.totalPay) / (scenario.payPerWorkDay + scenario.finePerAbsentDay)

/-- Theorem stating that for the given scenario, the number of absent days is 2 -/
theorem contractor_absent_days :
  let scenario : ContractorScenario := {
    totalDays := 30,
    payPerWorkDay := 25,
    finePerAbsentDay := 7.5,
    totalPay := 685
  }
  absentDays scenario = 2 := by sorry

end NUMINAMATH_CALUDE_contractor_absent_days_l3940_394097


namespace NUMINAMATH_CALUDE_equilateral_triangles_in_54gon_l3940_394059

/-- Represents a regular polygon with its center -/
structure RegularPolygonWithCenter (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  center : ℝ × ℝ

/-- Represents a selection of three points -/
structure TriangleSelection (n : ℕ) where
  p1 : Fin (n + 1)
  p2 : Fin (n + 1)
  p3 : Fin (n + 1)

/-- Checks if three points form an equilateral triangle -/
def isEquilateralTriangle (n : ℕ) (poly : RegularPolygonWithCenter n) (sel : TriangleSelection n) : Prop :=
  sorry

/-- Counts the number of equilateral triangles in a regular polygon with center -/
def countEquilateralTriangles (n : ℕ) (poly : RegularPolygonWithCenter n) : ℕ :=
  sorry

/-- The main theorem: there are 72 ways to select three points forming an equilateral triangle in a regular 54-gon with center -/
theorem equilateral_triangles_in_54gon :
  ∀ (poly : RegularPolygonWithCenter 54),
  countEquilateralTriangles 54 poly = 72 :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangles_in_54gon_l3940_394059


namespace NUMINAMATH_CALUDE_ellipse_equation_l3940_394034

/-- Given an ellipse with specific properties, prove its equation --/
theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  let e := (Real.sqrt 5) / 5
  let c := e * a
  (c^2 = a^2 - b^2) →
  (b = 2) →
  (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ↔ x^2 / 5 + y^2 / 4 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l3940_394034


namespace NUMINAMATH_CALUDE_squirrel_nuts_problem_l3940_394006

theorem squirrel_nuts_problem 
  (a b c d : ℕ) 
  (h1 : a + b + c + d = 2020)
  (h2 : a ≥ 103 ∧ b ≥ 103 ∧ c ≥ 103 ∧ d ≥ 103)
  (h3 : a > b ∧ a > c ∧ a > d)
  (h4 : b + c = 1277) :
  a = 640 := by
  sorry

end NUMINAMATH_CALUDE_squirrel_nuts_problem_l3940_394006


namespace NUMINAMATH_CALUDE_min_steps_to_one_l3940_394043

/-- Represents the allowed operations in one step -/
inductive Operation
  | AddOne
  | DivideByTwo
  | DivideByThree

/-- Applies an operation to a number -/
def applyOperation (n : ℕ) (op : Operation) : ℕ :=
  match op with
  | Operation.AddOne => n + 1
  | Operation.DivideByTwo => n / 2
  | Operation.DivideByThree => n / 3

/-- Checks if a sequence of operations is valid -/
def isValidSequence (start : ℕ) (ops : List Operation) : Bool :=
  ops.foldl (fun acc op => applyOperation acc op) start = 1

/-- The minimum number of steps to reach 1 from the starting number -/
def minSteps (start : ℕ) : ℕ :=
  sorry

theorem min_steps_to_one :
  minSteps 19 = 6 :=
sorry

end NUMINAMATH_CALUDE_min_steps_to_one_l3940_394043


namespace NUMINAMATH_CALUDE_trapezium_other_side_length_l3940_394022

-- Define the trapezium properties
def trapezium_side1 : ℝ := 20
def trapezium_height : ℝ := 15
def trapezium_area : ℝ := 285

-- Define the theorem
theorem trapezium_other_side_length :
  ∃ (side2 : ℝ), 
    (1/2 : ℝ) * (trapezium_side1 + side2) * trapezium_height = trapezium_area ∧
    side2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_trapezium_other_side_length_l3940_394022


namespace NUMINAMATH_CALUDE_right_triangle_area_l3940_394009

theorem right_triangle_area (a b : ℝ) (h1 : a^2 - 7*a + 12 = 0) (h2 : b^2 - 7*b + 12 = 0) (h3 : a ≠ b) : 
  let c := Real.sqrt (a^2 + b^2)
  let area1 := (1/2) * a * b
  let area2 := (1/2) * a * Real.sqrt (c^2 - a^2)
  area1 = 6 ∨ area2 = (3 * Real.sqrt 7) / 2 := by
sorry


end NUMINAMATH_CALUDE_right_triangle_area_l3940_394009


namespace NUMINAMATH_CALUDE_parabola_transformation_l3940_394068

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally -/
def shift_horizontal (p : Parabola) (h : ℝ) : Parabola :=
  { a := p.a
  , b := -2 * p.a * h + p.b
  , c := p.a * h^2 - p.b * h + p.c }

/-- Shifts a parabola vertically -/
def shift_vertical (p : Parabola) (v : ℝ) : Parabola :=
  { a := p.a
  , b := p.b
  , c := p.c + v }

/-- The original parabola y = x^2 -/
def original_parabola : Parabola :=
  { a := 1
  , b := 0
  , c := 0 }

theorem parabola_transformation :
  let p1 := shift_horizontal original_parabola 3
  let p2 := shift_vertical p1 4
  p2 = { a := 1, b := -6, c := 13 } := by sorry

end NUMINAMATH_CALUDE_parabola_transformation_l3940_394068


namespace NUMINAMATH_CALUDE_max_stores_visited_l3940_394069

theorem max_stores_visited (total_stores : ℕ) (total_visits : ℕ) (total_shoppers : ℕ) 
  (double_visitors : ℕ) (h1 : total_stores = 7) (h2 : total_visits = 21) 
  (h3 : total_shoppers = 11) (h4 : double_visitors = 7) 
  (h5 : double_visitors ≤ total_shoppers) 
  (h6 : 2 * double_visitors ≤ total_visits) : 
  ∃ (max_visits : ℕ), max_visits ≤ 4 ∧ 
  ∀ (individual_visits : ℕ), individual_visits ≤ max_visits := by
  sorry

end NUMINAMATH_CALUDE_max_stores_visited_l3940_394069


namespace NUMINAMATH_CALUDE_scale_model_height_emilys_model_height_l3940_394052

/-- Given an obelisk with height h and base area A, and a scale model with base area a,
    the height of the scale model is h * √(a/A) -/
theorem scale_model_height
  (h : ℝ) -- height of the original obelisk
  (A : ℝ) -- base area of the original obelisk
  (a : ℝ) -- base area of the scale model
  (h_pos : h > 0)
  (A_pos : A > 0)
  (a_pos : a > 0) :
  h * Real.sqrt (a / A) = (h * Real.sqrt a) / Real.sqrt A :=
by sorry

/-- The height of Emily's scale model obelisk is 5√10 meters -/
theorem emilys_model_height
  (h : ℝ) -- height of the original obelisk
  (A : ℝ) -- base area of the original obelisk
  (a : ℝ) -- base area of the scale model
  (h_eq : h = 50)
  (A_eq : A = 25)
  (a_eq : a = 0.025) :
  h * Real.sqrt (a / A) = 5 * Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_scale_model_height_emilys_model_height_l3940_394052


namespace NUMINAMATH_CALUDE_game_probability_game_probability_value_l3940_394051

theorem game_probability : ℝ :=
  let total_outcomes : ℕ := 16 * 16
  let matching_outcomes : ℕ := 16
  let non_matching_outcomes : ℕ := total_outcomes - matching_outcomes
  (non_matching_outcomes : ℝ) / total_outcomes

theorem game_probability_value : game_probability = 15 / 16 := by sorry

end NUMINAMATH_CALUDE_game_probability_game_probability_value_l3940_394051


namespace NUMINAMATH_CALUDE_janes_bagels_l3940_394086

theorem janes_bagels (muffin_cost bagel_cost : ℕ) (total_days : ℕ) : 
  muffin_cost = 60 →
  bagel_cost = 80 →
  total_days = 7 →
  ∃! (num_bagels : ℕ), 
    num_bagels ≤ total_days ∧
    ∃ (total_cost : ℕ), 
      total_cost * 100 = num_bagels * bagel_cost + (total_days - num_bagels) * muffin_cost ∧
      num_bagels = 4 :=
by sorry

end NUMINAMATH_CALUDE_janes_bagels_l3940_394086


namespace NUMINAMATH_CALUDE_intersection_M_N_l3940_394037

-- Define set M
def M : Set ℝ := {x | ∃ y, y = Real.sqrt (1 - x)}

-- Define set N
def N : Set ℝ := {x | 0 < x ∧ x < 2}

-- Theorem statement
theorem intersection_M_N : M ∩ N = {x | 0 < x ∧ x ≤ 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3940_394037


namespace NUMINAMATH_CALUDE_machine_production_time_l3940_394091

theorem machine_production_time (T : ℝ) : 
  T > 0 ∧ 
  (1 / T + 1 / 30 = 1 / 12) → 
  T = 45 := by
sorry

end NUMINAMATH_CALUDE_machine_production_time_l3940_394091


namespace NUMINAMATH_CALUDE_rug_strip_width_l3940_394067

/-- Given a rectangular floor and a rug, proves that the width of the uncovered strip is 2 meters -/
theorem rug_strip_width (floor_length floor_width rug_area : ℝ) 
  (h1 : floor_length = 10) 
  (h2 : floor_width = 8) 
  (h3 : rug_area = 24) : 
  ∃ w : ℝ, w > 0 ∧ w < floor_width / 2 ∧ 
  (floor_length - 2 * w) * (floor_width - 2 * w) = rug_area ∧ 
  w = 2 :=
sorry

end NUMINAMATH_CALUDE_rug_strip_width_l3940_394067


namespace NUMINAMATH_CALUDE_rural_school_absence_percentage_l3940_394007

theorem rural_school_absence_percentage :
  let total_students : ℕ := 120
  let boys : ℕ := 70
  let girls : ℕ := 50
  let absent_boys : ℕ := boys / 5
  let absent_girls : ℕ := girls / 4
  let total_absent : ℕ := absent_boys + absent_girls
  (total_absent : ℚ) / total_students * 100 = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_rural_school_absence_percentage_l3940_394007


namespace NUMINAMATH_CALUDE_problem_solution_l3940_394029

theorem problem_solution (x : ℝ) (h1 : x > 0) (h2 : x * ⌊x⌋ = 72) : x = 9 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3940_394029


namespace NUMINAMATH_CALUDE_average_temperature_problem_l3940_394004

theorem average_temperature_problem (T₁ T₂ T₃ T₄ T₅ : ℚ) : 
  (T₁ + T₂ + T₃ + T₄) / 4 = 58 →
  T₁ / T₅ = 7 / 8 →
  T₅ = 32 →
  (T₂ + T₃ + T₄ + T₅) / 4 = 59 :=
by sorry

end NUMINAMATH_CALUDE_average_temperature_problem_l3940_394004


namespace NUMINAMATH_CALUDE_weekend_price_is_105_l3940_394054

def original_price : ℝ := 250
def sale_discount : ℝ := 0.4
def weekend_discount : ℝ := 0.3

def sale_price : ℝ := original_price * (1 - sale_discount)
def weekend_price : ℝ := sale_price * (1 - weekend_discount)

theorem weekend_price_is_105 : weekend_price = 105 := by sorry

end NUMINAMATH_CALUDE_weekend_price_is_105_l3940_394054


namespace NUMINAMATH_CALUDE_max_value_quadratic_l3940_394065

/-- The function f(x) = -9x^2 + 27x + 15 has a maximum value of 141/4. -/
theorem max_value_quadratic : ∃ (M : ℝ), M = (141 : ℝ) / 4 ∧ 
  ∀ (x : ℝ), -9 * x^2 + 27 * x + 15 ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l3940_394065


namespace NUMINAMATH_CALUDE_logistics_problem_l3940_394089

/-- Represents the freight rates and charges for a logistics company. -/
structure FreightData where
  rateA : ℝ  -- Freight rate for goods A
  rateB : ℝ  -- Freight rate for goods B
  totalCharge : ℝ  -- Total freight charge

/-- Calculates the quantities of goods A and B transported given freight data for two months. -/
def calculateQuantities (march : FreightData) (april : FreightData) : ℝ × ℝ :=
  sorry

/-- Theorem stating that given the specific freight data for March and April,
    the quantities of goods A and B transported are 100 tons and 140 tons respectively. -/
theorem logistics_problem (march : FreightData) (april : FreightData) 
  (h1 : march.rateA = 50)
  (h2 : march.rateB = 30)
  (h3 : march.totalCharge = 9500)
  (h4 : april.rateA = 70)  -- 50 * 1.4 = 70
  (h5 : april.rateB = 40)
  (h6 : april.totalCharge = 13000) :
  calculateQuantities march april = (100, 140) :=
sorry

end NUMINAMATH_CALUDE_logistics_problem_l3940_394089


namespace NUMINAMATH_CALUDE_congruence_solution_l3940_394074

theorem congruence_solution (p q : Nat) (n : Nat) : 
  Nat.Prime p → Nat.Prime q → Odd p → Odd q → n > 1 →
  (q ^ (n + 2) % (p ^ n) = 3 ^ (n + 2) % (p ^ n)) →
  (p ^ (n + 2) % (q ^ n) = 3 ^ (n + 2) % (q ^ n)) →
  (p = 3 ∧ q = 3) := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l3940_394074


namespace NUMINAMATH_CALUDE_G_fraction_is_lowest_terms_denominator_minus_numerator_l3940_394085

/-- G is defined as the infinite repeating decimal 0.837837837... -/
def G : ℚ := 837 / 999

/-- The fraction representation of G in lowest terms -/
def G_fraction : ℚ := 31 / 37

theorem G_fraction_is_lowest_terms : G = G_fraction := by sorry

theorem denominator_minus_numerator : Nat.gcd 31 37 = 1 ∧ 37 - 31 = 6 := by sorry

end NUMINAMATH_CALUDE_G_fraction_is_lowest_terms_denominator_minus_numerator_l3940_394085


namespace NUMINAMATH_CALUDE_ellipses_same_foci_l3940_394027

/-- Given two ellipses with equations x²/9 + y²/4 = 1 and x²/(9-k) + y²/(4-k) = 1,
    where k < 4, prove that they have the same foci. -/
theorem ellipses_same_foci (k : ℝ) (h : k < 4) :
  let e1 := {(x, y) : ℝ × ℝ | x^2 / 9 + y^2 / 4 = 1}
  let e2 := {(x, y) : ℝ × ℝ | x^2 / (9 - k) + y^2 / (4 - k) = 1}
  let foci1 := {(x, y) : ℝ × ℝ | x^2 + y^2 = 5 ∧ y = 0}
  let foci2 := {(x, y) : ℝ × ℝ | x^2 + y^2 = 5 ∧ y = 0}
  foci1 = foci2 := by
sorry


end NUMINAMATH_CALUDE_ellipses_same_foci_l3940_394027


namespace NUMINAMATH_CALUDE_work_duration_l3940_394057

/-- Given two workers p and q, where p can complete a job in 15 days and q in 20 days,
    this theorem proves that if 0.5333333333333333 of the job remains after they work
    together for d days, then d must equal 4. -/
theorem work_duration (p q d : ℝ) : 
  p = 1 / 15 →
  q = 1 / 20 →
  1 - (p + q) * d = 0.5333333333333333 →
  d = 4 := by
  sorry


end NUMINAMATH_CALUDE_work_duration_l3940_394057


namespace NUMINAMATH_CALUDE_smallest_number_with_conditions_l3940_394081

/-- Given a natural number, returns true if it ends with 56 -/
def ends_with_56 (n : ℕ) : Prop :=
  n % 100 = 56

/-- Given a natural number, returns the sum of its digits -/
def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

theorem smallest_number_with_conditions :
  ∃ (n : ℕ), 
    ends_with_56 n ∧ 
    n % 56 = 0 ∧ 
    digit_sum n = 56 ∧
    (∀ m : ℕ, m < n → ¬(ends_with_56 m ∧ m % 56 = 0 ∧ digit_sum m = 56)) ∧
    n = 29899856 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_conditions_l3940_394081


namespace NUMINAMATH_CALUDE_petr_ivanovich_insurance_contract_l3940_394094

/-- Represents an insurance tool --/
inductive InsuranceTool
| AggregateInsuranceAmount
| Deductible

/-- Represents an insurance document --/
inductive InsuranceDocument
| InsuranceRules

/-- Represents a person --/
structure Person where
  name : String

/-- Represents an insurance contract --/
structure InsuranceContract where
  owner : Person
  tools : List InsuranceTool
  appendix : InsuranceDocument

/-- Theorem stating the correct insurance tools and document for Petr Ivanovich's contract --/
theorem petr_ivanovich_insurance_contract :
  ∃ (contract : InsuranceContract),
    contract.owner = Person.mk "Petr Ivanovich" ∧
    contract.tools = [InsuranceTool.AggregateInsuranceAmount, InsuranceTool.Deductible] ∧
    contract.appendix = InsuranceDocument.InsuranceRules :=
by sorry

end NUMINAMATH_CALUDE_petr_ivanovich_insurance_contract_l3940_394094


namespace NUMINAMATH_CALUDE_least_candies_l3940_394063

theorem least_candies (c : ℕ) : 
  c < 150 ∧ 
  c % 5 = 4 ∧ 
  c % 6 = 3 ∧ 
  c % 8 = 5 ∧
  (∀ k : ℕ, k < c → ¬(k < 150 ∧ k % 5 = 4 ∧ k % 6 = 3 ∧ k % 8 = 5)) →
  c = 69 := by
sorry

end NUMINAMATH_CALUDE_least_candies_l3940_394063


namespace NUMINAMATH_CALUDE_train_length_l3940_394064

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 120 → time = 15 → ∃ (length : ℝ), abs (length - 500) < 1 :=
by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l3940_394064


namespace NUMINAMATH_CALUDE_molecular_weight_AlCl3_l3940_394045

/-- The molecular weight of 4 moles of AlCl3 -/
theorem molecular_weight_AlCl3 (atomic_weight_Al atomic_weight_Cl : ℝ) 
  (h1 : atomic_weight_Al = 26.98)
  (h2 : atomic_weight_Cl = 35.45) : ℝ := by
  sorry

#check molecular_weight_AlCl3

end NUMINAMATH_CALUDE_molecular_weight_AlCl3_l3940_394045


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l3940_394018

theorem product_of_three_numbers (x y z n : ℝ) 
  (sum_eq : x + y + z = 180)
  (x_smallest : x ≤ y ∧ x ≤ z)
  (y_largest : y ≥ x ∧ y ≥ z)
  (n_def : n = 8 * x)
  (y_def : y = n + 10)
  (z_def : z = n - 10) :
  x * y * z = (180 / 17) * ((1440 / 17)^2 - 100) := by
  sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l3940_394018


namespace NUMINAMATH_CALUDE_rearrange_3622_l3940_394066

def digits : List ℕ := [3, 6, 2, 2]

theorem rearrange_3622 : (List.permutations digits).length = 12 := by
  sorry

end NUMINAMATH_CALUDE_rearrange_3622_l3940_394066


namespace NUMINAMATH_CALUDE_poetic_line_contrast_l3940_394050

/-- Represents a poetic line with two parts -/
structure PoeticLine :=
  (part1 : String)
  (part2 : String)

/-- Determines if a given part of a poetic line represents stillness -/
def isStillness (part : String) : Prop :=
  sorry

/-- Determines if a given part of a poetic line represents motion -/
def isMotion (part : String) : Prop :=
  sorry

/-- Determines if a poetic line contrasts stillness and motion -/
def contrastsStillnessAndMotion (line : PoeticLine) : Prop :=
  (isStillness line.part1 ∧ isMotion line.part2) ∨ (isMotion line.part1 ∧ isStillness line.part2)

/-- The four poetic lines given in the problem -/
def lineA : PoeticLine :=
  { part1 := "The bridge echoes with the distant barking of dogs"
  , part2 := "and the courtyard is empty with people asleep" }

def lineB : PoeticLine :=
  { part1 := "The stove fire illuminates the heaven and earth"
  , part2 := "and the red stars are mixed with the purple smoke" }

def lineC : PoeticLine :=
  { part1 := "The cold trees begin to have bird activities"
  , part2 := "and the frosty bridge has no human passage yet" }

def lineD : PoeticLine :=
  { part1 := "The crane cries over the quiet Chu mountain"
  , part2 := "and the frost is white on the autumn river in the morning" }

theorem poetic_line_contrast :
  contrastsStillnessAndMotion lineA ∧
  contrastsStillnessAndMotion lineB ∧
  contrastsStillnessAndMotion lineC ∧
  ¬contrastsStillnessAndMotion lineD :=
sorry

end NUMINAMATH_CALUDE_poetic_line_contrast_l3940_394050


namespace NUMINAMATH_CALUDE_prime_count_inequality_l3940_394024

/-- p_n denotes the nth prime number -/
def p (n : ℕ) : ℕ := sorry

/-- π(x) denotes the number of primes less than or equal to x -/
def π (x : ℝ) : ℕ := sorry

/-- The product of the first n primes -/
def primeProduct (n : ℕ) : ℕ := sorry

theorem prime_count_inequality (n : ℕ) (h : n ≥ 6) :
  π (Real.sqrt (primeProduct n : ℝ)) > 2 * n := by
  sorry

end NUMINAMATH_CALUDE_prime_count_inequality_l3940_394024


namespace NUMINAMATH_CALUDE_orange_juice_bottles_l3940_394039

/-- The number of fluid ounces Christine must buy -/
def min_fl_oz : ℝ := 60

/-- The size of each bottle in milliliters -/
def bottle_size_ml : ℝ := 250

/-- The number of fluid ounces in 1 liter -/
def fl_oz_per_liter : ℝ := 33.8

/-- The smallest number of bottles Christine could buy -/
def min_bottles : ℕ := 8

theorem orange_juice_bottles :
  ∃ (n : ℕ), n = min_bottles ∧
  n * bottle_size_ml / 1000 * fl_oz_per_liter ≥ min_fl_oz ∧
  ∀ (m : ℕ), m * bottle_size_ml / 1000 * fl_oz_per_liter ≥ min_fl_oz → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_orange_juice_bottles_l3940_394039


namespace NUMINAMATH_CALUDE_investment_period_l3940_394088

/-- Proves that given a sum of 7000 invested at 15% p.a. and 12% p.a., 
    if the difference in interest received is 420, then the investment period is 2 years. -/
theorem investment_period (principal : ℝ) (rate_high : ℝ) (rate_low : ℝ) (interest_diff : ℝ) :
  principal = 7000 →
  rate_high = 0.15 →
  rate_low = 0.12 →
  interest_diff = 420 →
  ∃ (years : ℝ), principal * rate_high * years - principal * rate_low * years = interest_diff ∧ years = 2 :=
by sorry

end NUMINAMATH_CALUDE_investment_period_l3940_394088


namespace NUMINAMATH_CALUDE_fraction_of_single_men_l3940_394035

theorem fraction_of_single_men (total : ℕ) (h1 : total > 0) :
  let women := (60 : ℚ) / 100 * total
  let men := total - women
  let married := (60 : ℚ) / 100 * total
  let married_men := (1 : ℚ) / 4 * men
  (men - married_men) / men = (3 : ℚ) / 4 :=
by sorry

end NUMINAMATH_CALUDE_fraction_of_single_men_l3940_394035


namespace NUMINAMATH_CALUDE_toms_favorite_numbers_l3940_394033

def is_toms_favorite (n : ℕ) : Prop :=
  100 < n ∧ n < 150 ∧
  n % 13 = 0 ∧
  n % 3 ≠ 0 ∧
  (n / 100 + (n / 10) % 10 + n % 10) % 4 = 0

theorem toms_favorite_numbers :
  ∀ n : ℕ, is_toms_favorite n ↔ n = 130 ∨ n = 143 :=
sorry

end NUMINAMATH_CALUDE_toms_favorite_numbers_l3940_394033


namespace NUMINAMATH_CALUDE_solve_for_y_l3940_394031

theorem solve_for_y (t : ℝ) (x y : ℝ) : 
  x = 3 - 2*t → y = 5*t + 3 → x = -7 → y = 28 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l3940_394031


namespace NUMINAMATH_CALUDE_quadratic_two_roots_l3940_394048

theorem quadratic_two_roots : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  (2 * x₁^2 + x₁ - 1 = 0) ∧ (2 * x₂^2 + x₂ - 1 = 0) ∧
  (∀ x : ℝ, 2 * x^2 + x - 1 = 0 → (x = x₁ ∨ x = x₂)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_roots_l3940_394048


namespace NUMINAMATH_CALUDE_soccer_team_wins_l3940_394076

theorem soccer_team_wins (total_games : ℕ) (win_percentage : ℚ) (games_won : ℕ) : 
  total_games = 158 →
  win_percentage = 40 / 100 →
  games_won = (total_games : ℚ) * win_percentage →
  games_won = 63 := by
sorry

end NUMINAMATH_CALUDE_soccer_team_wins_l3940_394076


namespace NUMINAMATH_CALUDE_tissue_packs_per_box_l3940_394032

/-- Proves that the number of packs in each box is 20 given the specified conditions -/
theorem tissue_packs_per_box :
  ∀ (total_boxes : ℕ) 
    (tissues_per_pack : ℕ) 
    (cost_per_tissue : ℚ) 
    (total_cost : ℚ),
  total_boxes = 10 →
  tissues_per_pack = 100 →
  cost_per_tissue = 5 / 100 →
  total_cost = 1000 →
  (total_cost / total_boxes) / (tissues_per_pack * cost_per_tissue) = 20 := by
sorry

end NUMINAMATH_CALUDE_tissue_packs_per_box_l3940_394032


namespace NUMINAMATH_CALUDE_x_intercept_of_perpendicular_line_l3940_394021

/-- A line in two-dimensional space. -/
structure Line where
  slope : ℚ
  y_intercept : ℚ

/-- The x-intercept of a line. -/
def x_intercept (l : Line) : ℚ := -l.y_intercept / l.slope

/-- The slope of a line perpendicular to a given line. -/
def perpendicular_slope (m : ℚ) : ℚ := -1 / m

/-- The slope of the line 4x - 3y = 12. -/
def given_line_slope : ℚ := 4 / 3

theorem x_intercept_of_perpendicular_line :
  let perpendicular_line := Line.mk (perpendicular_slope given_line_slope) 4
  x_intercept perpendicular_line = 16 / 3 := by sorry

end NUMINAMATH_CALUDE_x_intercept_of_perpendicular_line_l3940_394021


namespace NUMINAMATH_CALUDE_increasing_order_x_z_y_l3940_394042

theorem increasing_order_x_z_y (x : ℝ) (hx : 0.8 < x ∧ x < 0.9) :
  x < x^(x^x) ∧ x^(x^x) < x^x := by sorry

end NUMINAMATH_CALUDE_increasing_order_x_z_y_l3940_394042


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l3940_394087

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x + y = (x * y)^2) : 
  1 / x + 1 / y = 1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l3940_394087


namespace NUMINAMATH_CALUDE_point_on_number_line_l3940_394001

/-- Given a number line where point A represents 7 and point B is 3 units to the right of A, 
    prove that B represents 10. -/
theorem point_on_number_line (A B : ℝ) : A = 7 → B = A + 3 → B = 10 := by
  sorry

end NUMINAMATH_CALUDE_point_on_number_line_l3940_394001


namespace NUMINAMATH_CALUDE_inequality_proof_l3940_394040

theorem inequality_proof (a b c : ℝ) 
  (ha : a = (1/3)^(1/3)) 
  (hb : b = Real.log (1/2)) 
  (hc : c = Real.log (1/4) / Real.log (1/3)) : 
  b < a ∧ a < c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3940_394040


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3940_394092

-- Define the quadratic function
def f (x : ℝ) := -x^2 + 3*x + 28

-- Define the solution set
def solution_set := {x : ℝ | x ≤ -4 ∨ x ≥ 7}

-- Theorem statement
theorem quadratic_inequality_solution :
  {x : ℝ | f x ≤ 0} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3940_394092


namespace NUMINAMATH_CALUDE_percentage_calculation_l3940_394095

theorem percentage_calculation (x : ℝ) (h : 0.25 * x = 1200) : 0.35 * x = 1680 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l3940_394095


namespace NUMINAMATH_CALUDE_intersection_integer_point_l3940_394082

/-- A point with integer coordinates -/
structure IntegerPoint where
  x : ℤ
  y : ℤ

/-- The intersection point of two lines -/
def intersection (m : ℤ) : ℚ × ℚ :=
  let x := (4 + 2*m) / (1 - m)
  let y := x - 4
  (x, y)

/-- Predicate to check if a point has integer coordinates -/
def isIntegerPoint (p : ℚ × ℚ) : Prop :=
  ∃ (ip : IntegerPoint), (ip.x : ℚ) = p.1 ∧ (ip.y : ℚ) = p.2

theorem intersection_integer_point :
  ∃ (m : ℤ), isIntegerPoint (intersection m) ∧ m = 8 :=
sorry

end NUMINAMATH_CALUDE_intersection_integer_point_l3940_394082


namespace NUMINAMATH_CALUDE_f_is_even_and_decreasing_l3940_394084

-- Define the function
def f (x : ℝ) : ℝ := -x^2 + 1

-- State the theorem
theorem f_is_even_and_decreasing :
  (∀ x : ℝ, f x = f (-x)) ∧ 
  (∀ x y : ℝ, 0 < x → x < y → f y < f x) := by
sorry

end NUMINAMATH_CALUDE_f_is_even_and_decreasing_l3940_394084


namespace NUMINAMATH_CALUDE_bluetooth_module_stock_l3940_394075

theorem bluetooth_module_stock (total_modules : ℕ) (total_cost : ℚ)
  (expensive_cost cheap_cost : ℚ) :
  total_modules = 11 →
  total_cost = 45 →
  expensive_cost = 10 →
  cheap_cost = 7/2 →
  ∃ (expensive_count cheap_count : ℕ),
    expensive_count + cheap_count = total_modules ∧
    expensive_count * expensive_cost + cheap_count * cheap_cost = total_cost ∧
    cheap_count = 10 := by
  sorry

end NUMINAMATH_CALUDE_bluetooth_module_stock_l3940_394075
