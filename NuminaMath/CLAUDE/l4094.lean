import Mathlib

namespace NUMINAMATH_CALUDE_black_cube_difference_l4094_409416

/-- Represents a 3x3x3 cube built with unit cubes -/
structure Cube :=
  (size : Nat)
  (total_cubes : Nat)
  (surface_area : Nat)

/-- Represents the distribution of colors on the cube's surface -/
structure SurfaceColor :=
  (black : Nat)
  (grey : Nat)
  (white : Nat)

/-- Defines a valid 3x3x3 cube with equal surface color distribution -/
def valid_cube (c : Cube) (sc : SurfaceColor) : Prop :=
  c.size = 3 ∧
  c.total_cubes = 27 ∧
  c.surface_area = 54 ∧
  sc.black = sc.grey ∧
  sc.grey = sc.white ∧
  sc.black + sc.grey + sc.white = c.surface_area

/-- The minimum number of black cubes that can be used -/
def min_black_cubes (c : Cube) (sc : SurfaceColor) : Nat :=
  sorry

/-- The maximum number of black cubes that can be used -/
def max_black_cubes (c : Cube) (sc : SurfaceColor) : Nat :=
  sorry

/-- Theorem stating the difference between max and min black cubes -/
theorem black_cube_difference (c : Cube) (sc : SurfaceColor) :
  valid_cube c sc → max_black_cubes c sc - min_black_cubes c sc = 7 :=
  sorry

end NUMINAMATH_CALUDE_black_cube_difference_l4094_409416


namespace NUMINAMATH_CALUDE_john_learning_time_l4094_409403

/-- The number of vowels in the English alphabet -/
def num_vowels : ℕ := 5

/-- The total number of days John needs to learn all vowels -/
def total_days : ℕ := 15

/-- The number of days John needs to learn one alphabet (vowel) -/
def days_per_alphabet : ℚ := total_days / num_vowels

theorem john_learning_time : days_per_alphabet = 3 := by
  sorry

end NUMINAMATH_CALUDE_john_learning_time_l4094_409403


namespace NUMINAMATH_CALUDE_largest_constant_inequality_l4094_409448

theorem largest_constant_inequality (x y z : ℝ) :
  ∃ (C : ℝ), C = (2 + 2 * Real.sqrt 7) / 3 ∧
  (∀ (x y z : ℝ), x^2 + y^2 + z^2 + 2 ≥ C * (x + y + z - 1)) ∧
  (∀ (C' : ℝ), C' > C → ∃ (x y z : ℝ), x^2 + y^2 + z^2 + 2 < C' * (x + y + z - 1)) :=
by sorry

end NUMINAMATH_CALUDE_largest_constant_inequality_l4094_409448


namespace NUMINAMATH_CALUDE_arithmetic_sequence_x_value_l4094_409446

/-- Given an arithmetic sequence with first three terms 2x-3, 3x, and 5x+1, prove that x = 2 -/
theorem arithmetic_sequence_x_value :
  ∀ x : ℝ,
  let a₁ := 2*x - 3
  let a₂ := 3*x
  let a₃ := 5*x + 1
  (a₂ - a₁ = a₃ - a₂) → x = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_x_value_l4094_409446


namespace NUMINAMATH_CALUDE_volleyball_lineup_count_l4094_409481

def total_players : ℕ := 16
def lineup_size : ℕ := 7
def num_twins : ℕ := 2

theorem volleyball_lineup_count : 
  (Nat.choose total_players lineup_size) - 
  (Nat.choose (total_players - num_twins) lineup_size) = 8008 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_lineup_count_l4094_409481


namespace NUMINAMATH_CALUDE_stating_simultaneous_ring_theorem_l4094_409495

/-- The time interval (in minutes) between bell rings for the post office -/
def post_office_interval : ℕ := 18

/-- The time interval (in minutes) between bell rings for the train station -/
def train_station_interval : ℕ := 24

/-- The time interval (in minutes) between bell rings for the town hall -/
def town_hall_interval : ℕ := 30

/-- The time (in minutes) after which all bells ring simultaneously again -/
def simultaneous_ring_time : ℕ := 360

/-- 
Theorem stating that the time after which all bells ring simultaneously
is the least common multiple of their individual intervals
-/
theorem simultaneous_ring_theorem :
  simultaneous_ring_time = Nat.lcm post_office_interval (Nat.lcm train_station_interval town_hall_interval) :=
by sorry

end NUMINAMATH_CALUDE_stating_simultaneous_ring_theorem_l4094_409495


namespace NUMINAMATH_CALUDE_side_length_S2_is_correct_l4094_409408

/-- The side length of square S2 in a specific arrangement of rectangles and squares. -/
def side_length_S2 : ℕ :=
  let total_width : ℕ := 4422
  let total_height : ℕ := 2420
  -- S1 and S3 have the same side length, which is also the smaller dimension of R1 and R2
  -- Let r be this common side length
  -- Let s be the side length of S2
  -- From the height: 2r + s = total_height
  -- From the width: 2r + 3s = total_width
  -- Solving this system of equations gives s = 1001
  1001

/-- Theorem stating that the side length of S2 is correct given the conditions. -/
theorem side_length_S2_is_correct :
  let total_width : ℕ := 4422
  let total_height : ℕ := 2420
  ∃ (r : ℕ),
    (2 * r + side_length_S2 = total_height) ∧
    (2 * r + 3 * side_length_S2 = total_width) :=
by sorry

#eval side_length_S2  -- Should output 1001

end NUMINAMATH_CALUDE_side_length_S2_is_correct_l4094_409408


namespace NUMINAMATH_CALUDE_expression_value_l4094_409462

theorem expression_value (a b : ℤ) (ha : a = 4) (hb : b = -3) :
  -a - b^2 + a*b = -25 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l4094_409462


namespace NUMINAMATH_CALUDE_quadratic_sequence_formula_l4094_409471

theorem quadratic_sequence_formula (a : ℕ → ℚ) (α β : ℚ) :
  (∀ n : ℕ, a n * α^2 - a (n + 1) * α + 1 = 0) →
  (∀ n : ℕ, a n * β^2 - a (n + 1) * β + 1 = 0) →
  (6 * α - 2 * α * β + 6 * β = 3) →
  (a 1 = 7 / 6) →
  (∀ n : ℕ, a n = (1 / 2)^n + 2 / 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_sequence_formula_l4094_409471


namespace NUMINAMATH_CALUDE_max_four_digit_binary_is_15_l4094_409410

/-- The maximum value of a four-digit binary number in decimal -/
def max_four_digit_binary : ℕ := 15

/-- A function to convert a four-digit binary number to decimal -/
def binary_to_decimal (b₃ b₂ b₁ b₀ : Bool) : ℕ :=
  (if b₃ then 8 else 0) + (if b₂ then 4 else 0) + (if b₁ then 2 else 0) + (if b₀ then 1 else 0)

/-- Theorem stating that the maximum value of a four-digit binary number is 15 -/
theorem max_four_digit_binary_is_15 :
  ∀ b₃ b₂ b₁ b₀ : Bool, binary_to_decimal b₃ b₂ b₁ b₀ ≤ max_four_digit_binary :=
by sorry

end NUMINAMATH_CALUDE_max_four_digit_binary_is_15_l4094_409410


namespace NUMINAMATH_CALUDE_tom_siblings_count_l4094_409479

/-- The number of siblings Tom invited -/
def num_siblings : ℕ :=
  let total_plates : ℕ := 144
  let days : ℕ := 4
  let meals_per_day : ℕ := 3
  let plates_per_meal : ℕ := 2
  let tom_and_parents : ℕ := 3
  let plates_per_person : ℕ := days * meals_per_day * plates_per_meal
  let total_people : ℕ := total_plates / plates_per_person
  total_people - tom_and_parents

theorem tom_siblings_count : num_siblings = 3 := by
  sorry

end NUMINAMATH_CALUDE_tom_siblings_count_l4094_409479


namespace NUMINAMATH_CALUDE_math_majors_consecutive_probability_l4094_409487

/-- The number of people sitting at the round table -/
def total_people : ℕ := 9

/-- The number of math majors -/
def math_majors : ℕ := 4

/-- The number of ways to choose seats for math majors -/
def total_arrangements : ℕ := Nat.choose total_people math_majors

/-- The number of ways for math majors to sit in consecutive seats -/
def consecutive_arrangements : ℕ := total_people

/-- The probability that all math majors sit in consecutive seats -/
def probability : ℚ := consecutive_arrangements / total_arrangements

theorem math_majors_consecutive_probability :
  probability = 1 / 14 := by sorry

end NUMINAMATH_CALUDE_math_majors_consecutive_probability_l4094_409487


namespace NUMINAMATH_CALUDE_opposite_of_2023_l4094_409424

-- Define the opposite of a real number
def opposite (x : ℝ) : ℝ := -x

-- State the theorem
theorem opposite_of_2023 : opposite 2023 = -2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l4094_409424


namespace NUMINAMATH_CALUDE_square_sum_of_integers_l4094_409493

theorem square_sum_of_integers (x y : ℕ+) 
  (h1 : x * y + x + y = 117)
  (h2 : x^2 * y + x * y^2 = 1512) : 
  x^2 + y^2 = 549 := by
sorry

end NUMINAMATH_CALUDE_square_sum_of_integers_l4094_409493


namespace NUMINAMATH_CALUDE_unique_digit_sum_l4094_409418

theorem unique_digit_sum (A B C D X Y Z : ℕ) : 
  (A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧ X < 10 ∧ Y < 10 ∧ Z < 10) →
  (A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ X ∧ A ≠ Y ∧ A ≠ Z ∧
   B ≠ C ∧ B ≠ D ∧ B ≠ X ∧ B ≠ Y ∧ B ≠ Z ∧
   C ≠ D ∧ C ≠ X ∧ C ≠ Y ∧ C ≠ Z ∧
   D ≠ X ∧ D ≠ Y ∧ D ≠ Z ∧
   X ≠ Y ∧ X ≠ Z ∧
   Y ≠ Z) →
  (10 * A + B) + (10 * C + D) = 100 * X + 10 * Y + Z →
  Y = X + 1 →
  Z = X + 2 →
  A + B + C + D + X + Y + Z = 24 :=
by sorry

end NUMINAMATH_CALUDE_unique_digit_sum_l4094_409418


namespace NUMINAMATH_CALUDE_like_terms_imply_abs_diff_l4094_409469

/-- 
If -5x^3y^(n-2) and 3x^(2m+5)y are like terms, then |n-5m| = 8.
-/
theorem like_terms_imply_abs_diff (n m : ℤ) : 
  (2 * m + 5 = 3 ∧ n - 2 = 1) → |n - 5 * m| = 8 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_imply_abs_diff_l4094_409469


namespace NUMINAMATH_CALUDE_tangent_line_parallel_points_l4094_409401

def f (x : ℝ) : ℝ := x^3 + x - 2

theorem tangent_line_parallel_points :
  ∀ x y : ℝ, f x = y → (3 * x^2 + 1 = 4) ↔ ((x = 1 ∧ y = 0) ∨ (x = -1 ∧ y = -4)) := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_parallel_points_l4094_409401


namespace NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l4094_409434

theorem least_positive_integer_with_remainders : ∃ (M : ℕ), 
  (M > 0) ∧ 
  (M % 6 = 5) ∧ 
  (M % 7 = 6) ∧ 
  (M % 9 = 8) ∧ 
  (M % 10 = 9) ∧ 
  (M % 11 = 10) ∧ 
  (∀ (N : ℕ), 
    (N > 0) ∧ 
    (N % 6 = 5) ∧ 
    (N % 7 = 6) ∧ 
    (N % 9 = 8) ∧ 
    (N % 10 = 9) ∧ 
    (N % 11 = 10) → 
    M ≤ N) ∧
  M = 6929 :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l4094_409434


namespace NUMINAMATH_CALUDE_orange_profit_calculation_l4094_409472

/-- Calculates the profit from an orange selling operation -/
def orange_profit (buy_quantity : ℕ) (buy_price : ℚ) (sell_quantity : ℕ) (sell_price : ℚ) 
                  (transport_cost : ℚ) (storage_fee : ℚ) : ℚ :=
  let total_cost := buy_price + 2 * transport_cost + storage_fee
  let revenue := sell_price
  revenue - total_cost

/-- The profit from the orange selling operation is -4r -/
theorem orange_profit_calculation : 
  orange_profit 11 10 10 11 2 1 = -4 := by
  sorry

end NUMINAMATH_CALUDE_orange_profit_calculation_l4094_409472


namespace NUMINAMATH_CALUDE_probability_qualified_bulb_factory_A_l4094_409478

/-- The probability of buying a qualified light bulb produced by Factory A from the market -/
theorem probability_qualified_bulb_factory_A 
  (factory_A_production_rate : ℝ) 
  (factory_A_pass_rate : ℝ) 
  (h1 : factory_A_production_rate = 0.7)
  (h2 : factory_A_pass_rate = 0.95) : 
  factory_A_production_rate * factory_A_pass_rate = 0.665 := by
sorry

end NUMINAMATH_CALUDE_probability_qualified_bulb_factory_A_l4094_409478


namespace NUMINAMATH_CALUDE_average_battery_lifespan_l4094_409411

def battery_lifespans : List ℝ := [30, 35, 25, 25, 30, 34, 26, 25, 29, 21]

theorem average_battery_lifespan :
  (List.sum battery_lifespans) / (List.length battery_lifespans) = 28 := by
  sorry

end NUMINAMATH_CALUDE_average_battery_lifespan_l4094_409411


namespace NUMINAMATH_CALUDE_ceiling_floor_product_range_l4094_409456

theorem ceiling_floor_product_range (y : ℝ) :
  y < 0 → ⌈y⌉ * ⌊y⌋ = 132 → y ∈ Set.Ioo (-12) (-11) := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_product_range_l4094_409456


namespace NUMINAMATH_CALUDE_sin_cos_sixth_power_sum_l4094_409405

theorem sin_cos_sixth_power_sum (α : ℝ) (h : Real.sin (2 * α) = 1/2) : 
  Real.sin α ^ 6 + Real.cos α ^ 6 = 5/8 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sixth_power_sum_l4094_409405


namespace NUMINAMATH_CALUDE_coin_arrangement_count_l4094_409417

/-- Represents a coin with its type and orientation -/
inductive Coin
| Gold : Bool → Coin
| Silver : Bool → Coin

/-- Checks if two adjacent coins are not face to face -/
def notFaceToFace (c1 c2 : Coin) : Prop := sorry

/-- Checks if three consecutive coins do not have the same orientation -/
def notSameOrientation (c1 c2 c3 : Coin) : Prop := sorry

/-- Represents a valid arrangement of coins -/
def ValidArrangement (arrangement : List Coin) : Prop :=
  arrangement.length = 10 ∧
  (arrangement.filter (λ c => match c with | Coin.Gold _ => true | _ => false)).length = 5 ∧
  (arrangement.filter (λ c => match c with | Coin.Silver _ => true | _ => false)).length = 5 ∧
  (∀ i, i < 9 → notFaceToFace (arrangement.get ⟨i, sorry⟩) (arrangement.get ⟨i+1, sorry⟩)) ∧
  (∀ i, i < 8 → notSameOrientation (arrangement.get ⟨i, sorry⟩) (arrangement.get ⟨i+1, sorry⟩) (arrangement.get ⟨i+2, sorry⟩))

/-- The number of valid arrangements -/
def numValidArrangements : ℕ := sorry

theorem coin_arrangement_count :
  numValidArrangements = 8568 := by sorry

end NUMINAMATH_CALUDE_coin_arrangement_count_l4094_409417


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l4094_409447

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 2 * x + a ≥ 0) ↔ a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l4094_409447


namespace NUMINAMATH_CALUDE_pencil_distribution_l4094_409492

def colored_pencils : ℕ := 14
def black_pencils : ℕ := 35
def siblings : ℕ := 3
def kept_pencils : ℕ := 10

theorem pencil_distribution :
  (colored_pencils + black_pencils - kept_pencils) / siblings = 13 :=
by sorry

end NUMINAMATH_CALUDE_pencil_distribution_l4094_409492


namespace NUMINAMATH_CALUDE_z_has_max_min_iff_a_in_range_l4094_409425

/-- The set A defined by the given inequalities -/
def A (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | a * p.1 - 2 * p.2 + 8 ≥ 0 ∧ p.1 - p.2 - 1 ≤ 0 ∧ 2 * p.1 + a * p.2 - 2 ≤ 0}

/-- The function z defined as y - x -/
def z (p : ℝ × ℝ) : ℝ := p.2 - p.1

/-- Theorem stating the equivalence between the existence of max and min values for z
    and the range of a -/
theorem z_has_max_min_iff_a_in_range (a : ℝ) :
  (∃ (max min : ℝ), ∀ p ∈ A a, min ≤ z p ∧ z p ≤ max) ↔ a ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_z_has_max_min_iff_a_in_range_l4094_409425


namespace NUMINAMATH_CALUDE_preimage_of_4_3_l4094_409432

/-- The mapping f from ℝ² to ℝ² defined by f(x, y) = (x + 2y, 2x - y) -/
def f (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + 2 * p.2, 2 * p.1 - p.2)

/-- Theorem stating that (2, 1) is the pre-image of (4, 3) under the mapping f -/
theorem preimage_of_4_3 :
  f (2, 1) = (4, 3) ∧ ∀ p : ℝ × ℝ, f p = (4, 3) → p = (2, 1) :=
by sorry

end NUMINAMATH_CALUDE_preimage_of_4_3_l4094_409432


namespace NUMINAMATH_CALUDE_remainder_of_5n_mod_11_l4094_409439

theorem remainder_of_5n_mod_11 (n : ℤ) (h : n % 11 = 1) : (5 * n) % 11 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_5n_mod_11_l4094_409439


namespace NUMINAMATH_CALUDE_total_instruments_is_21_instrument_group_equality_l4094_409468

-- Define the number of body parts
def num_fingers : Nat := 10
def num_hands : Nat := 2
def num_heads : Nat := 1

-- Define the number of each instrument based on the conditions
def num_trumpets : Nat := num_fingers - 3
def num_guitars : Nat := num_hands + 2
def num_trombones : Nat := num_heads + 2
def num_french_horns : Nat := num_guitars - 1
def num_violins : Nat := num_trumpets / 2
def num_saxophones : Nat := num_trombones / 3

-- State the theorem
theorem total_instruments_is_21 :
  num_trumpets + num_guitars + num_trombones + num_french_horns + num_violins + num_saxophones = 21 :=
by sorry

-- Additional condition: equality of instrument groups
theorem instrument_group_equality :
  num_trumpets + num_guitars = num_trombones + num_violins + num_saxophones :=
by sorry

end NUMINAMATH_CALUDE_total_instruments_is_21_instrument_group_equality_l4094_409468


namespace NUMINAMATH_CALUDE_binomial_expansion_sum_l4094_409444

/-- Given that (1 - 2/x)³ = a₀ + a₁·(1/x) + a₂·(1/x)² + a₃·(1/x)³, prove that a₁ + a₂ = 6 -/
theorem binomial_expansion_sum (x : ℝ) (a₀ a₁ a₂ a₃ : ℝ) 
  (h : (1 - 2/x)^3 = a₀ + a₁ * (1/x) + a₂ * (1/x)^2 + a₃ * (1/x)^3) :
  a₁ + a₂ = 6 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_sum_l4094_409444


namespace NUMINAMATH_CALUDE_cos_shift_l4094_409459

theorem cos_shift (x : ℝ) : 
  Real.cos (1/2 * x + π/3) = Real.cos (1/2 * (x + 2*π/3)) := by
  sorry

end NUMINAMATH_CALUDE_cos_shift_l4094_409459


namespace NUMINAMATH_CALUDE_unique_solution_for_system_l4094_409486

theorem unique_solution_for_system :
  ∀ (x y z : ℝ),
  (x^2 + y^2 + z^2 = 2) →
  (x - z = 2) →
  (x = 1 ∧ y = 0 ∧ z = -1) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_for_system_l4094_409486


namespace NUMINAMATH_CALUDE_train_speed_l4094_409426

/-- Proves that a train with given length and time to cross a pole has a specific speed in km/h -/
theorem train_speed (train_length : ℝ) (crossing_time : ℝ) (h1 : train_length = 300) (h2 : crossing_time = 15) :
  (train_length / crossing_time) * 3.6 = 72 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l4094_409426


namespace NUMINAMATH_CALUDE_line_constant_value_l4094_409400

/-- Given a line passing through points (m, n) and (m + 2, n + 0.5) with equation x = k * y + 5, prove that k = 4 -/
theorem line_constant_value (m n k : ℝ) : 
  (m = k * n + 5) ∧ (m + 2 = k * (n + 0.5) + 5) → k = 4 := by
  sorry

end NUMINAMATH_CALUDE_line_constant_value_l4094_409400


namespace NUMINAMATH_CALUDE_parabola_focus_coordinates_l4094_409433

/-- The focus of the parabola (y-1)^2 = 4(x-1) has coordinates (0, 1) -/
theorem parabola_focus_coordinates (x y : ℝ) : 
  ((y - 1)^2 = 4*(x - 1)) → (x = 0 ∧ y = 1) := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_coordinates_l4094_409433


namespace NUMINAMATH_CALUDE_largest_number_l4094_409413

-- Define a function to convert a number from base b to decimal
def to_decimal (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * b ^ i) 0

-- Define the numbers in their respective bases
def num_A : Nat := to_decimal [2, 1, 1] 6
def num_B : Nat := 41
def num_C : Nat := to_decimal [6, 4] 9
def num_D : Nat := to_decimal [11, 2] 16

-- State the theorem
theorem largest_number :
  num_A > num_B ∧ num_A > num_C ∧ num_A > num_D :=
sorry

end NUMINAMATH_CALUDE_largest_number_l4094_409413


namespace NUMINAMATH_CALUDE_cos_five_pi_sixth_minus_x_l4094_409489

theorem cos_five_pi_sixth_minus_x (x : ℝ) 
  (h : Real.sin (π / 3 - x) = 3 / 5) : 
  Real.cos (5 * π / 6 - x) = -(3 / 5) := by
sorry

end NUMINAMATH_CALUDE_cos_five_pi_sixth_minus_x_l4094_409489


namespace NUMINAMATH_CALUDE_puppy_food_consumption_l4094_409460

def feeding_schedule (days : ℕ) (portions_per_day : ℕ) (portion_size : ℚ) : ℚ :=
  (days : ℚ) * (portions_per_day : ℚ) * portion_size

theorem puppy_food_consumption : 
  let first_two_weeks := feeding_schedule 14 3 (1/4)
  let second_two_weeks := feeding_schedule 14 2 (1/2)
  let today := (1/2 : ℚ)
  first_two_weeks + second_two_weeks + today = 25
  := by sorry

end NUMINAMATH_CALUDE_puppy_food_consumption_l4094_409460


namespace NUMINAMATH_CALUDE_tea_canister_production_balance_l4094_409428

/-- Represents the production balance in a factory producing cylindrical tea canisters -/
theorem tea_canister_production_balance 
  (total_workers : ℕ) 
  (bodies_per_hour : ℕ) 
  (bottoms_per_hour : ℕ) 
  (bottoms_per_body : ℕ) 
  (body_workers : ℕ) :
  total_workers = 44 →
  bodies_per_hour = 50 →
  bottoms_per_hour = 120 →
  bottoms_per_body = 2 →
  body_workers ≤ total_workers →
  (2 * bottoms_per_hour * (total_workers - body_workers) = bodies_per_hour * body_workers) ↔
  (bottoms_per_body * bottoms_per_hour * (total_workers - body_workers) = bodies_per_hour * body_workers) :=
by sorry

end NUMINAMATH_CALUDE_tea_canister_production_balance_l4094_409428


namespace NUMINAMATH_CALUDE_divisor_difference_two_l4094_409453

theorem divisor_difference_two (k : ℕ+) :
  ∃ (m : ℕ) (d : Fin (m + 1) → ℕ),
    (∀ i, d i ∣ (4 * k)) ∧
    (d 0 = 1) ∧
    (d (Fin.last m) = 4 * k) ∧
    (∀ i j, i < j → d i < d j) ∧
    (∃ i : Fin m, d i.succ - d i = 2) :=
by sorry

end NUMINAMATH_CALUDE_divisor_difference_two_l4094_409453


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l4094_409484

theorem inverse_proportion_problem (a b : ℝ) (k : ℝ) (h1 : a * b = k) (h2 : a + b = 60) (h3 : a = 3 * b) :
  let b' := k / (-12)
  b' = -225/4 :=
by sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l4094_409484


namespace NUMINAMATH_CALUDE_lyn_donation_l4094_409461

theorem lyn_donation (X : ℝ) : 
  (1 / 3 : ℝ) * X + (1 / 2 : ℝ) * X + (1 / 4 : ℝ) * ((1 : ℝ) - (1 / 3 : ℝ) - (1 / 2 : ℝ)) * X + 30 = X 
  → X = 240 := by
sorry

end NUMINAMATH_CALUDE_lyn_donation_l4094_409461


namespace NUMINAMATH_CALUDE_cone_volume_increase_l4094_409436

/-- Theorem: Volume increase of a cone with height increase of 160% and radius increase of k% -/
theorem cone_volume_increase (h r k : ℝ) (h_pos : h > 0) (r_pos : r > 0) (k_nonneg : k ≥ 0) :
  let new_height := 2.60 * h
  let new_radius := r * (1 + k / 100)
  let volume_ratio := (new_radius^2 * new_height) / (r^2 * h)
  let percentage_increase := (volume_ratio - 1) * 100
  percentage_increase = ((1 + k / 100)^2 * 2.60 - 1) * 100 :=
by sorry

end NUMINAMATH_CALUDE_cone_volume_increase_l4094_409436


namespace NUMINAMATH_CALUDE_remainder_of_quotient_l4094_409435

theorem remainder_of_quotient (q₁ q₂ : ℝ → ℝ) (r₁ r₂ : ℝ) :
  (∃ k₁ : ℝ → ℝ, ∀ x, x^9 = (x - 1/3) * q₁ x + r₁) →
  (∃ k₂ : ℝ → ℝ, ∀ x, q₁ x = (x - 1/3) * q₂ x + r₂) →
  r₂ = 1/6561 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_quotient_l4094_409435


namespace NUMINAMATH_CALUDE_expression_evaluation_l4094_409450

theorem expression_evaluation : 
  60 + (105 / 15) + (25 * 16) - 250 + (324 / 9)^2 = 1513 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4094_409450


namespace NUMINAMATH_CALUDE_smallest_square_with_40_and_49_existence_of_2000_square_smallest_2000_square_l4094_409443

theorem smallest_square_with_40_and_49 :
  ∀ n : ℕ, 
    (∃ a b : ℕ, a > 0 ∧ b > 0 ∧ n * n = 40 * 40 * a + 49 * 49 * b) →
    n ≥ 2000 :=
by sorry

theorem existence_of_2000_square :
  ∃ a b : ℕ, a > 0 ∧ b > 0 ∧ 2000 * 2000 = 40 * 40 * a + 49 * 49 * b :=
by sorry

theorem smallest_2000_square :
  (∀ n : ℕ, 
    (∃ a b : ℕ, a > 0 ∧ b > 0 ∧ n * n = 40 * 40 * a + 49 * 49 * b) →
    n ≥ 2000) ∧
  (∃ a b : ℕ, a > 0 ∧ b > 0 ∧ 2000 * 2000 = 40 * 40 * a + 49 * 49 * b) :=
by sorry

end NUMINAMATH_CALUDE_smallest_square_with_40_and_49_existence_of_2000_square_smallest_2000_square_l4094_409443


namespace NUMINAMATH_CALUDE_shirt_pricing_solution_l4094_409422

/-- Represents the shirt pricing problem with given conditions --/
structure ShirtPricingProblem where
  cost_price : ℝ
  initial_sales : ℝ
  initial_profit_per_shirt : ℝ
  price_reduction_effect : ℝ
  target_daily_profit : ℝ

/-- Calculates the daily sales based on the price reduction --/
def daily_sales (p : ShirtPricingProblem) (selling_price : ℝ) : ℝ :=
  p.initial_sales + p.price_reduction_effect * (p.cost_price + p.initial_profit_per_shirt - selling_price)

/-- Calculates the daily profit based on the selling price --/
def daily_profit (p : ShirtPricingProblem) (selling_price : ℝ) : ℝ :=
  (selling_price - p.cost_price) * (daily_sales p selling_price)

/-- Theorem stating that the selling price should be either $105 or $120 --/
theorem shirt_pricing_solution (p : ShirtPricingProblem)
  (h1 : p.cost_price = 80)
  (h2 : p.initial_sales = 30)
  (h3 : p.initial_profit_per_shirt = 50)
  (h4 : p.price_reduction_effect = 2)
  (h5 : p.target_daily_profit = 2000) :
  ∃ (x : ℝ), (x = 105 ∨ x = 120) ∧ daily_profit p x = p.target_daily_profit :=
sorry

end NUMINAMATH_CALUDE_shirt_pricing_solution_l4094_409422


namespace NUMINAMATH_CALUDE_professor_seating_arrangements_l4094_409464

/-- Represents the number of chairs in a row -/
def total_chairs : ℕ := 14

/-- Represents the number of professors -/
def num_professors : ℕ := 4

/-- Represents the number of students -/
def num_students : ℕ := 10

/-- Represents the number of possible positions for professors (excluding first and last chair) -/
def professor_positions : ℕ := total_chairs - 2

/-- Theorem stating the number of ways professors can choose their chairs -/
theorem professor_seating_arrangements :
  (∃ (two_adjacent : ℕ) (three_adjacent : ℕ) (four_adjacent : ℕ),
    two_adjacent = (professor_positions - 1) * (Nat.choose (professor_positions - 2) 2) * (Nat.factorial num_professors / 2) ∧
    three_adjacent = (professor_positions - 2) * (professor_positions - 3) * (Nat.factorial num_professors) ∧
    four_adjacent = (professor_positions - 3) * (Nat.factorial num_professors) ∧
    two_adjacent + three_adjacent + four_adjacent = 5346) :=
by sorry

end NUMINAMATH_CALUDE_professor_seating_arrangements_l4094_409464


namespace NUMINAMATH_CALUDE_cookies_per_bag_l4094_409483

/-- Given 26 bags with an equal number of cookies and 52 cookies in total,
    prove that each bag contains 2 cookies. -/
theorem cookies_per_bag :
  ∀ (bags : ℕ) (total_cookies : ℕ) (cookies_per_bag : ℕ),
    bags = 26 →
    total_cookies = 52 →
    total_cookies = bags * cookies_per_bag →
    cookies_per_bag = 2 := by
  sorry

end NUMINAMATH_CALUDE_cookies_per_bag_l4094_409483


namespace NUMINAMATH_CALUDE_sum_of_solutions_is_nine_l4094_409406

theorem sum_of_solutions_is_nine : 
  let f (x : ℝ) := (12 * x) / (x^2 - 4) - (3 * x) / (x + 2) + 9 / (x - 2)
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ + x₂ = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_is_nine_l4094_409406


namespace NUMINAMATH_CALUDE_remaining_work_time_is_three_l4094_409445

/-- The time taken by A to finish the remaining work after B has worked for 10 days -/
def remaining_work_time (a_time b_time b_worked_days : ℚ) : ℚ :=
  let b_work_rate := 1 / b_time
  let b_work_done := b_work_rate * b_worked_days
  let remaining_work := 1 - b_work_done
  let a_work_rate := 1 / a_time
  remaining_work / a_work_rate

/-- Theorem stating that A will take 3 days to finish the remaining work -/
theorem remaining_work_time_is_three :
  remaining_work_time 9 15 10 = 3 := by
  sorry

#eval remaining_work_time 9 15 10

end NUMINAMATH_CALUDE_remaining_work_time_is_three_l4094_409445


namespace NUMINAMATH_CALUDE_kylie_coins_left_l4094_409463

/-- The number of coins Kylie has after all transactions -/
def coins_left (piggy_bank : ℕ) (from_brother : ℕ) (from_father : ℕ) (given_to_friend : ℕ) : ℕ :=
  piggy_bank + from_brother + from_father - given_to_friend

/-- Theorem stating that Kylie is left with 15 coins -/
theorem kylie_coins_left : 
  coins_left 15 13 8 21 = 15 := by sorry

end NUMINAMATH_CALUDE_kylie_coins_left_l4094_409463


namespace NUMINAMATH_CALUDE_sculpture_cost_brl_l4094_409438

/-- Exchange rate from USD to AUD -/
def usd_to_aud : ℝ := 5

/-- Exchange rate from USD to BRL -/
def usd_to_brl : ℝ := 10

/-- Cost of the sculpture in AUD -/
def sculpture_cost_aud : ℝ := 200

/-- Theorem stating the equivalent cost of the sculpture in BRL -/
theorem sculpture_cost_brl : 
  (sculpture_cost_aud / usd_to_aud) * usd_to_brl = 400 := by
  sorry

end NUMINAMATH_CALUDE_sculpture_cost_brl_l4094_409438


namespace NUMINAMATH_CALUDE_least_time_six_horses_at_start_l4094_409496

def horse_lap_time (k : ℕ) : ℕ := 2 * k - 1

def is_at_start (t : ℕ) (k : ℕ) : Prop :=
  t % (horse_lap_time k) = 0

def at_least_six_at_start (t : ℕ) : Prop :=
  ∃ (s : Finset ℕ), s.card ≥ 6 ∧ s ⊆ Finset.range 12 ∧ ∀ k ∈ s, is_at_start t (k + 1)

theorem least_time_six_horses_at_start :
  ∃! t : ℕ, t > 0 ∧ at_least_six_at_start t ∧ ∀ s, s > 0 ∧ s < t → ¬(at_least_six_at_start s) :=
by sorry

end NUMINAMATH_CALUDE_least_time_six_horses_at_start_l4094_409496


namespace NUMINAMATH_CALUDE_smallest_multiple_thirty_six_satisfies_thirty_six_is_smallest_l4094_409455

theorem smallest_multiple (x : ℕ) : x > 0 ∧ 450 * x % 648 = 0 → x ≥ 36 := by
  sorry

theorem thirty_six_satisfies : 450 * 36 % 648 = 0 := by
  sorry

theorem thirty_six_is_smallest : ∃ (x : ℕ), x > 0 ∧ 450 * x % 648 = 0 ∧ x = 36 := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_thirty_six_satisfies_thirty_six_is_smallest_l4094_409455


namespace NUMINAMATH_CALUDE_barycentric_vector_relation_l4094_409423

/-- For a triangle ABC and a point X with barycentric coordinates (α:β:γ) where α + β + γ = 1,
    the vector →XA is equal to β→BA + γ→CA. -/
theorem barycentric_vector_relation (A B C X : EuclideanSpace ℝ (Fin 3))
  (α β γ : ℝ) (h_barycentric : α + β + γ = 1)
  (h_X : X = α • A + β • B + γ • C) :
  X - A = β • (B - A) + γ • (C - A) := by
  sorry

end NUMINAMATH_CALUDE_barycentric_vector_relation_l4094_409423


namespace NUMINAMATH_CALUDE_divisibility_property_l4094_409430

theorem divisibility_property (a m n : ℕ) (ha : a > 1) (hdiv : (a^m + 1) ∣ (a^n + 1)) : m ∣ n := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l4094_409430


namespace NUMINAMATH_CALUDE_min_sum_of_roots_l4094_409449

theorem min_sum_of_roots (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h1 : ∃ x : ℝ, x^2 + a*x + 3*b = 0) 
  (h2 : ∃ x : ℝ, x^2 + 3*b*x + a = 0) : 
  a + b ≥ 48/27 + 9/4 * (9216/6561)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_roots_l4094_409449


namespace NUMINAMATH_CALUDE_x_plus_y_equals_four_l4094_409480

-- Define the conditions
def conditions (x y : ℝ) : Prop :=
  x ≥ -2 ∧ 
  y ≥ -3 ∧ 
  x - 2 * Real.sqrt (x + 2) = 2 * Real.sqrt (y + 3) - y

-- Theorem statement
theorem x_plus_y_equals_four (x y : ℝ) (h : conditions x y) : x + y = 4 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_equals_four_l4094_409480


namespace NUMINAMATH_CALUDE_complex_product_theorem_l4094_409476

theorem complex_product_theorem (z₁ z₂ : ℂ) 
  (h1 : Complex.abs z₁ = 2) 
  (h2 : Complex.abs z₂ = 3) 
  (h3 : 3 * z₁ - 2 * z₂ = (3/2 : ℂ) - Complex.I) : 
  z₁ * z₂ = -30/13 + 72/13 * Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_product_theorem_l4094_409476


namespace NUMINAMATH_CALUDE_equation_always_has_solution_l4094_409420

theorem equation_always_has_solution (a b : ℝ) (ha : a ≠ 0) 
  (h_at_most_one : ∃! x, a * x^2 - b * x - a + 3 = 0) :
  ∃ x, (b - 3) * x^2 + (a - 2 * b) * x + 3 * a + 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_always_has_solution_l4094_409420


namespace NUMINAMATH_CALUDE_no_factors_l4094_409475

/-- The main polynomial -/
def f (x : ℝ) : ℝ := x^4 + 3*x^2 + 8

/-- Potential factors -/
def g₁ (x : ℝ) : ℝ := x^2 + 4
def g₂ (x : ℝ) : ℝ := x + 2
def g₃ (x : ℝ) : ℝ := x^2 - 4
def g₄ (x : ℝ) : ℝ := x^2 - x - 2

theorem no_factors : 
  (¬ ∃ (h : ℝ → ℝ), f = g₁ * h) ∧
  (¬ ∃ (h : ℝ → ℝ), f = g₂ * h) ∧
  (¬ ∃ (h : ℝ → ℝ), f = g₃ * h) ∧
  (¬ ∃ (h : ℝ → ℝ), f = g₄ * h) :=
by sorry

end NUMINAMATH_CALUDE_no_factors_l4094_409475


namespace NUMINAMATH_CALUDE_triple_equation_solution_l4094_409414

theorem triple_equation_solution :
  ∀ a b c : ℝ,
    a + b + c = 14 ∧
    a^2 + b^2 + c^2 = 84 ∧
    a^3 + b^3 + c^3 = 584 →
    ((a = 4 ∧ b = 2 ∧ c = 8) ∨
     (a = 2 ∧ b = 4 ∧ c = 8) ∨
     (a = 8 ∧ b = 2 ∧ c = 4)) :=
by
  sorry

end NUMINAMATH_CALUDE_triple_equation_solution_l4094_409414


namespace NUMINAMATH_CALUDE_friend_reading_time_l4094_409494

theorem friend_reading_time (my_time : ℝ) (friend_speed_multiplier : ℝ) (distraction_time : ℝ) :
  my_time = 1.5 →
  friend_speed_multiplier = 5 →
  distraction_time = 0.25 →
  (my_time * 60) / friend_speed_multiplier + distraction_time = 33 :=
by sorry

end NUMINAMATH_CALUDE_friend_reading_time_l4094_409494


namespace NUMINAMATH_CALUDE_work_completion_time_l4094_409404

theorem work_completion_time (a b : ℝ) (h1 : a = 30) 
  (h2 : 1 / a + 1 / b = 0.5 / 10) : b = 60 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l4094_409404


namespace NUMINAMATH_CALUDE_positive_integer_solutions_for_mn_equation_l4094_409452

theorem positive_integer_solutions_for_mn_equation :
  ∀ m n : ℕ+,
  m^(n : ℕ) = n^((m : ℕ) - (n : ℕ)) →
  ((m = 9 ∧ n = 3) ∨ (m = 8 ∧ n = 2)) :=
by sorry

end NUMINAMATH_CALUDE_positive_integer_solutions_for_mn_equation_l4094_409452


namespace NUMINAMATH_CALUDE_problem_solution_l4094_409421

-- Define the function f
def f (t : ℝ) (x : ℝ) : ℝ := |x - 4| - t

-- State the theorem
theorem problem_solution :
  ∀ t : ℝ,
  (∀ x : ℝ, f t x ≤ 2 ↔ -1 ≤ x ∧ x ≤ 5) →
  (t = 1 ∧
   ∀ a b c : ℝ,
   a > 0 → b > 0 → c > 0 →
   a + b + c = t →
   a^2 / b + b^2 / c + c^2 / a ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l4094_409421


namespace NUMINAMATH_CALUDE_garage_sale_earnings_l4094_409440

/-- The total earnings from selling necklaces at a garage sale -/
def total_earnings (bead_count gemstone_count crystal_count wooden_count : ℕ)
                   (bead_price gemstone_price crystal_price wooden_price : ℕ) : ℕ :=
  bead_count * bead_price + 
  gemstone_count * gemstone_price + 
  crystal_count * crystal_price + 
  wooden_count * wooden_price

/-- Theorem stating that the total earnings from selling the specified necklaces is $53 -/
theorem garage_sale_earnings : 
  total_earnings 4 3 2 5 3 7 5 2 = 53 := by
  sorry

end NUMINAMATH_CALUDE_garage_sale_earnings_l4094_409440


namespace NUMINAMATH_CALUDE_smallest_multiples_sum_l4094_409454

/-- The smallest positive two-digit multiple of 5 -/
def c : ℕ := 10

/-- The smallest positive three-digit multiple of 6 -/
def d : ℕ := 102

theorem smallest_multiples_sum : c + d = 112 := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiples_sum_l4094_409454


namespace NUMINAMATH_CALUDE_less_crowded_detector_time_is_ten_l4094_409466

/-- Represents the time Mark spends on courthouse activities in a week -/
structure CourthouseTime where
  workDays : ℕ
  parkingTime : ℕ
  walkingTime : ℕ
  crowdedDetectorDays : ℕ
  crowdedDetectorTime : ℕ
  totalWeeklyTime : ℕ

/-- Calculates the time it takes to get through the metal detector on less crowded days -/
def lessCrowdedDetectorTime (ct : CourthouseTime) : ℕ :=
  let weeklyParkingTime := ct.workDays * ct.parkingTime
  let weeklyWalkingTime := ct.workDays * ct.walkingTime
  let weeklyCrowdedDetectorTime := ct.crowdedDetectorDays * ct.crowdedDetectorTime
  let remainingTime := ct.totalWeeklyTime - weeklyParkingTime - weeklyWalkingTime - weeklyCrowdedDetectorTime
  remainingTime / (ct.workDays - ct.crowdedDetectorDays)

theorem less_crowded_detector_time_is_ten (ct : CourthouseTime)
  (h1 : ct.workDays = 5)
  (h2 : ct.parkingTime = 5)
  (h3 : ct.walkingTime = 3)
  (h4 : ct.crowdedDetectorDays = 2)
  (h5 : ct.crowdedDetectorTime = 30)
  (h6 : ct.totalWeeklyTime = 130) :
  lessCrowdedDetectorTime ct = 10 := by
  sorry

#eval lessCrowdedDetectorTime ⟨5, 5, 3, 2, 30, 130⟩

end NUMINAMATH_CALUDE_less_crowded_detector_time_is_ten_l4094_409466


namespace NUMINAMATH_CALUDE_fibonacci_identities_l4094_409412

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

theorem fibonacci_identities (n : ℕ) : 
  (fib (2*n + 1) * fib (2*n - 1) = fib (2*n)^2 + 1) ∧ 
  (fib (2*n + 1)^2 + fib (2*n - 1)^2 + 1 = 3 * fib (2*n + 1) * fib (2*n - 1)) := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_identities_l4094_409412


namespace NUMINAMATH_CALUDE_complex_to_exponential_l4094_409497

theorem complex_to_exponential : 
  let z : ℂ := 1 + Complex.I * Real.sqrt 3
  ∃ (r : ℝ) (θ : ℝ), z = r * Complex.exp (Complex.I * θ) ∧ r = 2 ∧ θ = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_to_exponential_l4094_409497


namespace NUMINAMATH_CALUDE_train_speed_problem_l4094_409474

/-- Given two trains A and B with lengths 225 m and 150 m respectively,
    if it takes 15 seconds for train A to completely cross train B,
    then the speed of train A is 90 km/hr. -/
theorem train_speed_problem (length_A length_B time_to_cross : ℝ) :
  length_A = 225 →
  length_B = 150 →
  time_to_cross = 15 →
  (length_A + length_B) / time_to_cross * 3.6 = 90 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_problem_l4094_409474


namespace NUMINAMATH_CALUDE_range_of_even_quadratic_function_l4094_409441

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 2

-- Define the property of being an even function
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

-- State the theorem
theorem range_of_even_quadratic_function (a b : ℝ) :
  (∃ x, f a b x = 1 + a) →  -- Lower bound of the domain
  (∃ x, f a b x = 2) →      -- Upper bound of the domain
  is_even (f a b) →         -- f is an even function
  (∀ x, f a b x ∈ Set.Icc (-10) 2) ∧ 
  (∃ x, f a b x = -10) ∧ 
  (∃ x, f a b x = 2) :=
by sorry


end NUMINAMATH_CALUDE_range_of_even_quadratic_function_l4094_409441


namespace NUMINAMATH_CALUDE_plant_branches_l4094_409491

theorem plant_branches (x : ℕ) 
  (h1 : x > 0)
  (h2 : 1 + x + x * x = 31) : x = 5 := by
  sorry

end NUMINAMATH_CALUDE_plant_branches_l4094_409491


namespace NUMINAMATH_CALUDE_smallest_integer_for_negative_quadratic_l4094_409488

theorem smallest_integer_for_negative_quadratic : 
  ∃ (x : ℤ), (∀ (y : ℤ), y^2 - 11*y + 24 < 0 → x ≤ y) ∧ (x^2 - 11*x + 24 < 0) ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_for_negative_quadratic_l4094_409488


namespace NUMINAMATH_CALUDE_potatoes_already_cooked_l4094_409470

theorem potatoes_already_cooked 
  (total_potatoes : ℕ) 
  (cooking_time_per_potato : ℕ) 
  (remaining_cooking_time : ℕ) 
  (h1 : total_potatoes = 16)
  (h2 : cooking_time_per_potato = 5)
  (h3 : remaining_cooking_time = 45) :
  total_potatoes - (remaining_cooking_time / cooking_time_per_potato) = 7 :=
by sorry

end NUMINAMATH_CALUDE_potatoes_already_cooked_l4094_409470


namespace NUMINAMATH_CALUDE_parallel_transitive_perpendicular_from_line_l4094_409429

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Plane → Plane → Prop)
variable (line_parallel : Line → Plane → Prop)
variable (line_perpendicular : Line → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)

-- Theorem for proposition ①
theorem parallel_transitive (α β γ : Plane) :
  parallel α β → parallel α γ → parallel γ β := by sorry

-- Theorem for proposition ③
theorem perpendicular_from_line (m : Line) (α β : Plane) :
  line_perpendicular m α → line_parallel m β → perpendicular α β := by sorry

end NUMINAMATH_CALUDE_parallel_transitive_perpendicular_from_line_l4094_409429


namespace NUMINAMATH_CALUDE_y_multiples_l4094_409437

theorem y_multiples : ∃ (a b c d : ℤ),
  let y := 112 + 160 + 272 + 432 + 1040 + 1264 + 4256
  y = 16 * a ∧ y = 8 * b ∧ y = 4 * c ∧ y = 2 * d :=
by sorry

end NUMINAMATH_CALUDE_y_multiples_l4094_409437


namespace NUMINAMATH_CALUDE_share_ratio_l4094_409409

def problem (total a b c : ℚ) : Prop :=
  total = 527 ∧
  a = 372 ∧
  b = 93 ∧
  c = 62 ∧
  a = (2/3) * b ∧
  total = a + b + c

theorem share_ratio (total a b c : ℚ) (h : problem total a b c) :
  b / c = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_share_ratio_l4094_409409


namespace NUMINAMATH_CALUDE_expected_heads_alice_given_more_than_bob_l4094_409458

/-- The number of coins each person flips -/
def n : ℕ := 20

/-- The expected number of heads Alice flipped given she flipped at least as many heads as Bob -/
noncomputable def expected_heads : ℝ :=
  n * (2^(2*n - 2) + Nat.choose (2*n - 1) (n - 1)) / (2^(2*n - 1) + Nat.choose (2*n - 1) (n - 1))

/-- Theorem stating the expected number of heads Alice flipped -/
theorem expected_heads_alice_given_more_than_bob :
  expected_heads = n * (2^(2*n - 2) + Nat.choose (2*n - 1) (n - 1)) / (2^(2*n - 1) + Nat.choose (2*n - 1) (n - 1)) :=
by sorry

end NUMINAMATH_CALUDE_expected_heads_alice_given_more_than_bob_l4094_409458


namespace NUMINAMATH_CALUDE_smallest_money_for_pizza_l4094_409485

theorem smallest_money_for_pizza (x : ℕ) : x ≥ 6 ↔ ∃ (a b : ℕ), x - 1 = 5 * a + 7 * b := by
  sorry

end NUMINAMATH_CALUDE_smallest_money_for_pizza_l4094_409485


namespace NUMINAMATH_CALUDE_equation_A_is_quadratic_l4094_409482

/-- Definition of a quadratic equation in terms of x -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x² = -1 -/
def equation_A (x : ℝ) : ℝ := x^2 + 1

/-- Theorem: The equation x² = -1 is a quadratic equation -/
theorem equation_A_is_quadratic : is_quadratic_equation equation_A := by
  sorry


end NUMINAMATH_CALUDE_equation_A_is_quadratic_l4094_409482


namespace NUMINAMATH_CALUDE_power_function_through_point_is_sqrt_l4094_409431

/-- A power function that passes through the point (4, 2) is equal to the square root function. -/
theorem power_function_through_point_is_sqrt (f : ℝ → ℝ) :
  (∃ a : ℝ, ∀ x : ℝ, f x = x ^ a) →  -- f is a power function
  f 4 = 2 →                         -- f passes through (4, 2)
  ∀ x : ℝ, f x = Real.sqrt x :=     -- f is the square root function
by sorry

end NUMINAMATH_CALUDE_power_function_through_point_is_sqrt_l4094_409431


namespace NUMINAMATH_CALUDE_area_ratio_concentric_circles_l4094_409457

/-- Given two concentric circles where a 60-degree arc on the smaller circle
    has the same length as a 30-degree arc on the larger circle,
    the ratio of the area of the smaller circle to the area of the larger circle is 1/4. -/
theorem area_ratio_concentric_circles (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) :
  (60 / 360 * (2 * Real.pi * r₁) = 30 / 360 * (2 * Real.pi * r₂)) →
  (Real.pi * r₁^2) / (Real.pi * r₂^2) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_concentric_circles_l4094_409457


namespace NUMINAMATH_CALUDE_coin_value_equality_l4094_409498

theorem coin_value_equality (n : ℕ) : 
  (15 * 25 + 20 * 10 = 5 * 25 + n * 10) → n = 45 := by
  sorry

end NUMINAMATH_CALUDE_coin_value_equality_l4094_409498


namespace NUMINAMATH_CALUDE_polynomial_root_product_l4094_409415

theorem polynomial_root_product (d e : ℝ) : 
  (∀ x : ℝ, x^2 + d*x + e = 0 ↔ x = Real.cos (π/9) ∨ x = Real.cos (2*π/9)) →
  d * e = -5/64 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_root_product_l4094_409415


namespace NUMINAMATH_CALUDE_min_circle_area_l4094_409442

theorem min_circle_area (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : (3 / (2 + x)) + (3 / (2 + y)) = 1) :
  xy ≥ 16 ∧ (xy = 16 ↔ x = 4 ∧ y = 4) :=
by sorry

end NUMINAMATH_CALUDE_min_circle_area_l4094_409442


namespace NUMINAMATH_CALUDE_line_param_solution_l4094_409427

/-- The line equation -/
def line_equation (x y : ℝ) : Prop := y = -x + 3

/-- The parameterization of the line -/
def parameterization (u v m : ℝ) (x y : ℝ) : Prop :=
  x = 2 + u * m ∧ y = v + u * 8

/-- Theorem stating that v = 1 and m = -8 satisfy the line equation and parameterization -/
theorem line_param_solution :
  ∃ (v m : ℝ), v = 1 ∧ m = -8 ∧
  (∀ (x y u : ℝ), parameterization u v m x y → line_equation x y) :=
sorry

end NUMINAMATH_CALUDE_line_param_solution_l4094_409427


namespace NUMINAMATH_CALUDE_box_volume_problem_l4094_409473

theorem box_volume_problem :
  ∃! (x : ℕ+), (2 * x.val - 5 > 0) ∧
  ((x.val^2 + 5) * (2 * x.val - 5) * (x.val + 25) < 1200) := by
  sorry

end NUMINAMATH_CALUDE_box_volume_problem_l4094_409473


namespace NUMINAMATH_CALUDE_inequality_proof_l4094_409402

theorem inequality_proof (x y : ℝ) (n : ℕ+) (hx : x > 0) (hy : y > 0) :
  (x^n.val / (1 + x^2)) + (y^n.val / (1 + y^2)) ≤ (x^n.val + y^n.val) / (1 + x*y) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4094_409402


namespace NUMINAMATH_CALUDE_horse_distribution_l4094_409465

theorem horse_distribution (total_horses : ℕ) (son1_horses son2_horses son3_horses : ℕ) :
  total_horses = 17 ∧
  son1_horses = 9 ∧
  son2_horses = 6 ∧
  son3_horses = 2 →
  son1_horses / total_horses = 1/2 ∧
  son2_horses / total_horses = 1/3 ∧
  son3_horses / total_horses = 1/9 ∧
  son1_horses + son2_horses + son3_horses = total_horses :=
by
  sorry

#check horse_distribution

end NUMINAMATH_CALUDE_horse_distribution_l4094_409465


namespace NUMINAMATH_CALUDE_amelias_dinner_leftover_l4094_409499

/-- Calculates the amount of money Amelia has left after her dinner --/
def ameliasDinner (initialAmount : ℝ) (firstCourseCost : ℝ) (secondCourseExtra : ℝ) 
  (dessertPercent : ℝ) (drinkPercent : ℝ) (tipPercent : ℝ) : ℝ :=
  let secondCourseCost := firstCourseCost + secondCourseExtra
  let dessertCost := dessertPercent * secondCourseCost
  let firstThreeCoursesTotal := firstCourseCost + secondCourseCost + dessertCost
  let drinkCost := drinkPercent * firstThreeCoursesTotal
  let billBeforeTip := firstThreeCoursesTotal + drinkCost
  let tipAmount := tipPercent * billBeforeTip
  let totalBill := billBeforeTip + tipAmount
  initialAmount - totalBill

/-- Theorem stating that Amelia will have $4.80 left after her dinner --/
theorem amelias_dinner_leftover :
  ameliasDinner 60 15 5 0.25 0.20 0.15 = 4.80 := by
  sorry

#eval ameliasDinner 60 15 5 0.25 0.20 0.15

end NUMINAMATH_CALUDE_amelias_dinner_leftover_l4094_409499


namespace NUMINAMATH_CALUDE_eight_bead_bracelet_arrangements_l4094_409419

/-- The number of unique ways to place n distinct beads on a rotatable, non-flippable bracelet -/
def braceletArrangements (n : ℕ) : ℕ := Nat.factorial n / n

/-- Theorem: The number of unique ways to place 8 distinct beads on a bracelet
    that can be rotated but not flipped is 5040 -/
theorem eight_bead_bracelet_arrangements :
  braceletArrangements 8 = 5040 := by
  sorry

end NUMINAMATH_CALUDE_eight_bead_bracelet_arrangements_l4094_409419


namespace NUMINAMATH_CALUDE_midpoint_property_l4094_409477

/-- Given two points D and E, if F is their midpoint, then 3x - 5y = 9 --/
theorem midpoint_property (D E F : ℝ × ℝ) : 
  D = (30, 10) → 
  E = (6, 8) → 
  F.1 = (D.1 + E.1) / 2 → 
  F.2 = (D.2 + E.2) / 2 → 
  3 * F.1 - 5 * F.2 = 9 := by
sorry

end NUMINAMATH_CALUDE_midpoint_property_l4094_409477


namespace NUMINAMATH_CALUDE_vector_projection_l4094_409407

/-- The projection of vector a in the direction of vector b is equal to √65/5 -/
theorem vector_projection (a b : ℝ × ℝ) : 
  a = (2, 3) → b = (-4, 7) → 
  (a.1 * b.1 + a.2 * b.2) / Real.sqrt (b.1^2 + b.2^2) = Real.sqrt 65 / 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_projection_l4094_409407


namespace NUMINAMATH_CALUDE_pasta_preference_ratio_l4094_409451

theorem pasta_preference_ratio (total_students : ℕ) (ravioli_preference : ℕ) (tortellini_preference : ℕ)
  (h1 : total_students = 800)
  (h2 : ravioli_preference = 300)
  (h3 : tortellini_preference = 150) :
  (ravioli_preference : ℚ) / tortellini_preference = 2 := by
  sorry

end NUMINAMATH_CALUDE_pasta_preference_ratio_l4094_409451


namespace NUMINAMATH_CALUDE_function_characterization_l4094_409467

-- Define the property that the function f must satisfy
def SatisfiesInequality (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, f (a^2) - f (b^2) ≤ (f a + b) * (a - f b)

-- State the theorem
theorem function_characterization (f : ℝ → ℝ) (h : SatisfiesInequality f) :
  (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = -x) :=
by sorry

end NUMINAMATH_CALUDE_function_characterization_l4094_409467


namespace NUMINAMATH_CALUDE_square_root_sum_l4094_409490

theorem square_root_sum (x : ℝ) : 
  (Real.sqrt (64 - x^2) - Real.sqrt (36 - x^2) = 4) → 
  (Real.sqrt (64 - x^2) + Real.sqrt (36 - x^2) + Real.sqrt (16 - x^2) = 9) := by
  sorry

end NUMINAMATH_CALUDE_square_root_sum_l4094_409490
