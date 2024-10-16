import Mathlib

namespace NUMINAMATH_CALUDE_sum_series_eq_factorial_minus_one_l4159_415941

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def sum_series (n : ℕ) : ℕ := 
  Finset.sum (Finset.range (n + 1)) (λ k => k * factorial k)

theorem sum_series_eq_factorial_minus_one (n : ℕ) : 
  sum_series n = factorial (n + 1) - 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_series_eq_factorial_minus_one_l4159_415941


namespace NUMINAMATH_CALUDE_gas_bill_payment_l4159_415986

def electricity_bill : ℚ := 60
def gas_bill : ℚ := 40
def water_bill : ℚ := 40
def internet_bill : ℚ := 25

def gas_bill_paid_initially : ℚ := (3 / 4) * gas_bill
def water_bill_paid : ℚ := (1 / 2) * water_bill
def internet_bill_paid : ℚ := 4 * 5

def remaining_to_pay : ℚ := 30

theorem gas_bill_payment (payment : ℚ) : 
  gas_bill + water_bill + internet_bill - 
  (gas_bill_paid_initially + water_bill_paid + internet_bill_paid + payment) = 
  remaining_to_pay → 
  payment = 5 := by sorry

end NUMINAMATH_CALUDE_gas_bill_payment_l4159_415986


namespace NUMINAMATH_CALUDE_sum_of_squares_values_l4159_415959

theorem sum_of_squares_values (x y z : ℝ) 
  (distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z)
  (eq1 : x^2 = 2 + y)
  (eq2 : y^2 = 2 + z)
  (eq3 : z^2 = 2 + x) :
  x^2 + y^2 + z^2 = 5 ∨ x^2 + y^2 + z^2 = 6 ∨ x^2 + y^2 + z^2 = 9 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_values_l4159_415959


namespace NUMINAMATH_CALUDE_random_walk_prob_4_in_3_to_9_l4159_415985

/-- A one-dimensional random walk on integers -/
def RandomWalk := ℕ → ℤ

/-- The probability of a random walk reaching a specific distance -/
def prob_reach_distance (w : RandomWalk) (d : ℕ) (steps : ℕ) : ℚ :=
  sorry

/-- The probability of a random walk reaching a specific distance at least once within a range of steps -/
def prob_reach_distance_in_range (w : RandomWalk) (d : ℕ) (min_steps max_steps : ℕ) : ℚ :=
  sorry

/-- The main theorem: probability of reaching distance 4 at least once in 3 to 9 steps is 47/224 -/
theorem random_walk_prob_4_in_3_to_9 (w : RandomWalk) :
  prob_reach_distance_in_range w 4 3 9 = 47 / 224 := by
  sorry

end NUMINAMATH_CALUDE_random_walk_prob_4_in_3_to_9_l4159_415985


namespace NUMINAMATH_CALUDE_max_value_theorem_max_value_achievable_l4159_415932

theorem max_value_theorem (x y : ℝ) :
  (3 * x + 4 * y + 5) / Real.sqrt (3 * x^2 + 4 * y^2 + 6) ≤ Real.sqrt 50 :=
by sorry

theorem max_value_achievable :
  ∃ x y : ℝ, (3 * x + 4 * y + 5) / Real.sqrt (3 * x^2 + 4 * y^2 + 6) = Real.sqrt 50 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_max_value_achievable_l4159_415932


namespace NUMINAMATH_CALUDE_smallest_n_with_abc_property_l4159_415995

def has_abc_property (n : ℕ) : Prop :=
  ∀ (A B : Set ℕ), A ∪ B = Finset.range n →
    (∃ (a b c : ℕ), a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ a * b = c) ∨
    (∃ (a b c : ℕ), a ∈ B ∧ b ∈ B ∧ c ∈ B ∧ a * b = c)

theorem smallest_n_with_abc_property :
  (∀ k < 243, ¬ has_abc_property k) ∧ has_abc_property 243 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_with_abc_property_l4159_415995


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l4159_415952

theorem min_value_sum_reciprocals (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 2 → 1/x + 4/y ≥ 1/a + 4/b) →
  1/a + 4/b = 9/2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l4159_415952


namespace NUMINAMATH_CALUDE_f_min_at_neg_one_l4159_415902

/-- The quadratic function f(x) = 3x^2 + 6x + 4 -/
def f (x : ℝ) : ℝ := 3 * x^2 + 6 * x + 4

/-- The theorem stating that the minimum of f occurs at x = -1 -/
theorem f_min_at_neg_one :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ x_min = -1 :=
sorry

end NUMINAMATH_CALUDE_f_min_at_neg_one_l4159_415902


namespace NUMINAMATH_CALUDE_integral_equality_l4159_415969

theorem integral_equality : ∫ (x : ℝ) in (0 : ℝ)..(1 : ℝ), (Real.sqrt (1 - (x - 1)^2) - x) = (Real.pi - 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_integral_equality_l4159_415969


namespace NUMINAMATH_CALUDE_trigonometric_product_equals_three_fourths_l4159_415930

theorem trigonometric_product_equals_three_fourths :
  let cos30 : ℝ := Real.sqrt 3 / 2
  let sin60 : ℝ := Real.sqrt 3 / 2
  let sin30 : ℝ := 1 / 2
  let cos60 : ℝ := 1 / 2
  (1 - 1 / cos30) * (1 + 1 / sin60) * (1 - 1 / sin30) * (1 + 1 / cos60) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_product_equals_three_fourths_l4159_415930


namespace NUMINAMATH_CALUDE_johns_out_of_pocket_expense_l4159_415904

/-- The amount of money John spent out of pocket when buying a computer and accessories
    after selling his PlayStation --/
theorem johns_out_of_pocket_expense (computer_cost accessories_cost playstation_value : ℚ)
  (h1 : computer_cost = 700)
  (h2 : accessories_cost = 200)
  (h3 : playstation_value = 400)
  (discount_rate : ℚ)
  (h4 : discount_rate = 1/5) : -- 20% expressed as a fraction
  computer_cost + accessories_cost - playstation_value * (1 - discount_rate) = 580 := by
  sorry

end NUMINAMATH_CALUDE_johns_out_of_pocket_expense_l4159_415904


namespace NUMINAMATH_CALUDE_cubes_arrangement_theorem_l4159_415979

/-- Represents the colors used to paint the cubes -/
inductive Color
  | White
  | Black
  | Red

/-- Represents a cube with 6 faces, each painted with a color -/
structure Cube :=
  (faces : Fin 6 → Color)

/-- Represents the set of 16 cubes -/
def CubeSet := Fin 16 → Cube

/-- Represents an arrangement of the 16 cubes -/
structure Arrangement :=
  (placement : Fin 16 → Fin 3 × Fin 3 × Fin 3)
  (orientation : Fin 16 → Fin 6)

/-- Predicate to check if an arrangement shows only one color -/
def ShowsOnlyOneColor (cs : CubeSet) (arr : Arrangement) (c : Color) : Prop :=
  ∀ i : Fin 16, (cs i).faces (arr.orientation i) = c

/-- Theorem stating that it's possible to arrange the cubes to show only one color -/
theorem cubes_arrangement_theorem (cs : CubeSet) :
  ∃ (arr : Arrangement) (c : Color), ShowsOnlyOneColor cs arr c :=
sorry

end NUMINAMATH_CALUDE_cubes_arrangement_theorem_l4159_415979


namespace NUMINAMATH_CALUDE_A_empty_iff_A_singleton_iff_A_singleton_elements_l4159_415998

def A (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 - 3 * x + 2 = 0}

theorem A_empty_iff (a : ℝ) : A a = ∅ ↔ a > 9/8 := by sorry

theorem A_singleton_iff (a : ℝ) : (∃! x, x ∈ A a) ↔ a = 0 ∨ a = 9/8 := by sorry

theorem A_singleton_elements (a : ℝ) :
  (a = 0 → A a = {2/3}) ∧ (a = 9/8 → A a = {4/3}) := by sorry

end NUMINAMATH_CALUDE_A_empty_iff_A_singleton_iff_A_singleton_elements_l4159_415998


namespace NUMINAMATH_CALUDE_exit_times_theorem_l4159_415911

/-- Represents the time in minutes it takes to exit through a door -/
structure ExitTime where
  time : ℝ
  time_positive : time > 0

/-- Represents the cinema with two doors -/
structure Cinema where
  wide_door : ExitTime
  narrow_door : ExitTime
  combined_exit_time : ℝ
  combined_exit_time_value : combined_exit_time = 3.75
  door_time_difference : narrow_door.time = wide_door.time + 4

theorem exit_times_theorem (c : Cinema) :
  c.wide_door.time = 6 ∧ c.narrow_door.time = 10 := by
  sorry

#check exit_times_theorem

end NUMINAMATH_CALUDE_exit_times_theorem_l4159_415911


namespace NUMINAMATH_CALUDE_quadratic_root_condition_l4159_415977

theorem quadratic_root_condition (a : ℝ) : 
  (∃! x : ℝ, x ∈ (Set.Ioo 0 1) ∧ x^2 - a*x + 1 = 0) → a > 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_condition_l4159_415977


namespace NUMINAMATH_CALUDE_cindy_calculation_l4159_415924

theorem cindy_calculation (x : ℝ) : (2 * (x - 9)) / 6 = 36 → (x - 12) / 8 = 13.125 := by
  sorry

end NUMINAMATH_CALUDE_cindy_calculation_l4159_415924


namespace NUMINAMATH_CALUDE_gcd_612_468_is_36_l4159_415963

theorem gcd_612_468_is_36 : Nat.gcd 612 468 = 36 := by
  sorry

end NUMINAMATH_CALUDE_gcd_612_468_is_36_l4159_415963


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l4159_415934

theorem arithmetic_calculations : 
  (78 - 14 * 2 = 50) ∧ 
  (500 - 296 - 104 = 100) ∧ 
  (360 - 300 / 5 = 300) ∧ 
  (84 / (16 / 4) = 21) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l4159_415934


namespace NUMINAMATH_CALUDE_line_vector_proof_l4159_415966

def line_vector (t : ℝ) : ℝ × ℝ × ℝ := sorry

theorem line_vector_proof :
  (line_vector 0 = (1, 5, 9)) →
  (line_vector 1 = (6, 0, 4)) →
  (line_vector 4 = (21, -15, -11)) := by
  sorry

end NUMINAMATH_CALUDE_line_vector_proof_l4159_415966


namespace NUMINAMATH_CALUDE_meeting_time_and_distance_l4159_415983

/-- Represents the time in hours since 7:45 AM -/
def time_since_start : ℝ → ℝ := λ t => t

/-- Samantha's speed in miles per hour -/
def samantha_speed : ℝ := 15

/-- Adam's speed in miles per hour -/
def adam_speed : ℝ := 20

/-- Time difference between Samantha's and Adam's start times in hours -/
def start_time_diff : ℝ := 0.5

/-- Total distance between Town A and Town B in miles -/
def total_distance : ℝ := 75

/-- Calculates Samantha's traveled distance at time t -/
def samantha_distance (t : ℝ) : ℝ := samantha_speed * t

/-- Calculates Adam's traveled distance at time t -/
def adam_distance (t : ℝ) : ℝ := adam_speed * (t - start_time_diff)

/-- Theorem stating the meeting time and Samantha's traveled distance -/
theorem meeting_time_and_distance :
  ∃ t : ℝ, 
    samantha_distance t + adam_distance t = total_distance ∧
    time_since_start t = 2.4333333333333 ∧ 
    samantha_distance t = 36 := by
  sorry

end NUMINAMATH_CALUDE_meeting_time_and_distance_l4159_415983


namespace NUMINAMATH_CALUDE_petrol_consumption_reduction_l4159_415912

/-- Theorem: Calculation of required reduction in petrol consumption to maintain constant expenditure --/
theorem petrol_consumption_reduction
  (price_increase_A : ℝ) (price_increase_B : ℝ)
  (maintenance_cost_ratio : ℝ) (maintenance_cost_increase : ℝ)
  (h1 : price_increase_A = 0.20)
  (h2 : price_increase_B = 0.15)
  (h3 : maintenance_cost_ratio = 0.30)
  (h4 : maintenance_cost_increase = 0.10) :
  let avg_price_increase := (1 + price_increase_A + 1 + price_increase_B) / 2 - 1
  let total_maintenance_increase := maintenance_cost_ratio * maintenance_cost_increase
  let total_increase := avg_price_increase + total_maintenance_increase
  total_increase = 0.205 := by sorry

end NUMINAMATH_CALUDE_petrol_consumption_reduction_l4159_415912


namespace NUMINAMATH_CALUDE_sally_balloons_l4159_415974

/-- The number of orange balloons Sally has after losing some -/
def remaining_balloons (initial : ℕ) (lost : ℕ) : ℕ := initial - lost

/-- Proof that Sally has 7 orange balloons after losing 2 from her initial 9 -/
theorem sally_balloons : remaining_balloons 9 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_sally_balloons_l4159_415974


namespace NUMINAMATH_CALUDE_jack_plates_left_l4159_415923

/-- Represents the number of plates Jack has with different patterns -/
structure PlateCollection where
  flower : ℕ
  checked : ℕ
  polkadot : ℕ

/-- Calculates the total number of plates after Jack's actions -/
def total_plates_after_actions (initial : PlateCollection) : ℕ :=
  (initial.flower - 1) + initial.checked + (2 * initial.checked)

/-- Theorem stating that Jack has 27 plates left after his actions -/
theorem jack_plates_left (initial : PlateCollection) 
  (h1 : initial.flower = 4)
  (h2 : initial.checked = 8)
  (h3 : initial.polkadot = 0) : 
  total_plates_after_actions initial = 27 := by
  sorry

#check jack_plates_left

end NUMINAMATH_CALUDE_jack_plates_left_l4159_415923


namespace NUMINAMATH_CALUDE_city_population_ratio_l4159_415994

def population_ratio (pop_Z : ℝ) : Prop :=
  let pop_Y := 2.5 * pop_Z
  let pop_X := 6 * pop_Y
  let pop_A := 3 * pop_X
  let pop_B := 4 * pop_Y
  (pop_X / (pop_Z + pop_B)) = 15 / 11

theorem city_population_ratio :
  ∀ pop_Z : ℝ, pop_Z > 0 → population_ratio pop_Z :=
by
  sorry

end NUMINAMATH_CALUDE_city_population_ratio_l4159_415994


namespace NUMINAMATH_CALUDE_incorrect_average_calculation_l4159_415992

theorem incorrect_average_calculation (n : ℕ) (correct_num incorrect_num : ℚ) (correct_avg : ℚ) :
  n = 10 ∧ 
  correct_num = 86 ∧ 
  incorrect_num = 26 ∧ 
  correct_avg = 26 →
  (n * correct_avg - correct_num + incorrect_num) / n = 20 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_average_calculation_l4159_415992


namespace NUMINAMATH_CALUDE_smallest_a_value_l4159_415962

theorem smallest_a_value (a b : ℕ) (h : b^3 = 1176*a) : 
  (∀ x : ℕ, x < a → ¬∃ y : ℕ, y^3 = 1176*x) → a = 63 := by
  sorry

end NUMINAMATH_CALUDE_smallest_a_value_l4159_415962


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l4159_415960

/-- An arithmetic sequence with given first and third terms -/
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℤ) :
  arithmetic_sequence a → a 1 = 1 → a 3 = -3 →
  a 1 - a 2 - a 3 - a 4 - a 5 = 17 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l4159_415960


namespace NUMINAMATH_CALUDE_outfit_combinations_l4159_415999

/-- The number of red shirts -/
def red_shirts : ℕ := 8

/-- The number of green shirts -/
def green_shirts : ℕ := 7

/-- The number of blue pants -/
def blue_pants : ℕ := 8

/-- The number of red hats -/
def red_hats : ℕ := 10

/-- The number of green hats -/
def green_hats : ℕ := 9

/-- The number of black belts -/
def black_belts : ℕ := 5

/-- The number of brown belts -/
def brown_belts : ℕ := 4

/-- The total number of possible outfits -/
def total_outfits : ℕ := red_shirts * blue_pants * green_hats * brown_belts + 
                         green_shirts * blue_pants * red_hats * black_belts

theorem outfit_combinations : total_outfits = 5104 := by
  sorry

end NUMINAMATH_CALUDE_outfit_combinations_l4159_415999


namespace NUMINAMATH_CALUDE_stating_volume_division_ratio_l4159_415993

/-- Represents a truncated triangular pyramid -/
structure TruncatedTriangularPyramid where
  -- The ratio of corresponding sides of the upper and lower bases
  base_ratio : ℝ
  -- Assume base_ratio > 0
  base_ratio_pos : base_ratio > 0

/-- 
  Theorem stating that for a truncated triangular pyramid with base ratio 1:2,
  a plane drawn through a side of the upper base parallel to the opposite lateral edge
  divides the volume in the ratio 3:4
-/
theorem volume_division_ratio 
  (pyramid : TruncatedTriangularPyramid) 
  (h_ratio : pyramid.base_ratio = 1/2) :
  ∃ (v1 v2 : ℝ), v1 > 0 ∧ v2 > 0 ∧ v1 / v2 = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_stating_volume_division_ratio_l4159_415993


namespace NUMINAMATH_CALUDE_x_value_l4159_415926

theorem x_value (w y z x : ℤ) 
  (hw : w = 90)
  (hz : z = w + 25)
  (hy : y = z + 12)
  (hx : x = y + 7) : 
  x = 134 := by sorry

end NUMINAMATH_CALUDE_x_value_l4159_415926


namespace NUMINAMATH_CALUDE_billion_yuan_eq_scientific_notation_l4159_415957

/-- Represents the value in billions of yuan -/
def billion_yuan : ℝ := 98.36

/-- Represents the same value in scientific notation -/
def scientific_notation : ℝ := 9.836 * (10 ^ 9)

/-- Theorem stating that the billion yuan value is equal to its scientific notation -/
theorem billion_yuan_eq_scientific_notation : billion_yuan * (10 ^ 9) = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_billion_yuan_eq_scientific_notation_l4159_415957


namespace NUMINAMATH_CALUDE_fraction_equality_l4159_415910

theorem fraction_equality (x : ℝ) (h : x / (x^2 + x - 1) = 1 / 7) :
  x^2 / (x^4 - x^2 + 1) = 1 / 37 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l4159_415910


namespace NUMINAMATH_CALUDE_circle_center_fourth_quadrant_l4159_415996

/-- Given a real number a, if the equation x^2 + y^2 - 2ax + 4ay + 6a^2 - a = 0 
    represents a circle with its center in the fourth quadrant, then 0 < a < 1. -/
theorem circle_center_fourth_quadrant (a : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 - 2*a*x + 4*a*y + 6*a^2 - a = 0 → 
    ∃ r : ℝ, r > 0 ∧ ∀ x' y' : ℝ, (x' - a)^2 + (y' + 2*a)^2 = r^2) →
  (a > 0 ∧ -2*a < 0) →
  0 < a ∧ a < 1 := by
sorry

end NUMINAMATH_CALUDE_circle_center_fourth_quadrant_l4159_415996


namespace NUMINAMATH_CALUDE_cube_sum_eq_product_l4159_415950

theorem cube_sum_eq_product (m : ℕ) :
  (m = 1 ∨ m = 2 → ¬∃ (x y z : ℕ+), x^3 + y^3 + z^3 = m * x * y * z) ∧
  (m = 3 → ∀ (x y z : ℕ+), x^3 + y^3 + z^3 = 3 * x * y * z ↔ x = y ∧ y = z) :=
by sorry

end NUMINAMATH_CALUDE_cube_sum_eq_product_l4159_415950


namespace NUMINAMATH_CALUDE_min_rooms_correct_min_rooms_optimal_l4159_415978

/-- The minimum number of rooms required for 100 tourists given k rooms under renovation -/
def min_rooms (k : ℕ) : ℕ :=
  if k % 2 = 0
  then 100 * (k / 2 + 1)
  else 100 * ((k - 1) / 2 + 1) + 1

/-- Theorem stating the minimum number of rooms required for 100 tourists -/
theorem min_rooms_correct (k : ℕ) :
  ∀ n : ℕ, n ≥ min_rooms k →
    ∃ (strategy : Fin 100 → Fin n → Option (Fin n)),
      ∀ (permutation : Fin 100 → Fin 100) (renovated : Finset (Fin n)),
        renovated.card = k →
        ∀ i : Fin 100, ∃ j : Fin n,
          strategy (permutation i) j = some j ∧
          j ∉ renovated ∧
          ∀ i' : Fin 100, i' < i →
            ∀ j' : Fin n, strategy (permutation i') j' = some j' → j ≠ j' :=
by sorry

/-- Theorem stating the optimality of the minimum number of rooms -/
theorem min_rooms_optimal (k : ℕ) :
  ∀ n : ℕ, n < min_rooms k →
    ¬∃ (strategy : Fin 100 → Fin n → Option (Fin n)),
      ∀ (permutation : Fin 100 → Fin 100) (renovated : Finset (Fin n)),
        renovated.card = k →
        ∀ i : Fin 100, ∃ j : Fin n,
          strategy (permutation i) j = some j ∧
          j ∉ renovated ∧
          ∀ i' : Fin 100, i' < i →
            ∀ j' : Fin n, strategy (permutation i') j' = some j' → j ≠ j' :=
by sorry

end NUMINAMATH_CALUDE_min_rooms_correct_min_rooms_optimal_l4159_415978


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l4159_415936

theorem quadratic_equation_roots (k : ℝ) : 
  (2 : ℝ) ^ 2 + k * 2 - 10 = 0 → k = 3 ∧ ∃ x : ℝ, x ≠ 2 ∧ x ^ 2 + k * x - 10 = 0 ∧ x = -5 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l4159_415936


namespace NUMINAMATH_CALUDE_onions_on_scale_l4159_415984

/-- The number of onions initially on the scale -/
def N : ℕ := sorry

/-- The total weight of onions in grams -/
def W : ℕ := 7680

/-- The average weight of remaining onions in grams -/
def avg_remaining : ℕ := 190

/-- The average weight of removed onions in grams -/
def avg_removed : ℕ := 206

/-- The number of removed onions -/
def removed : ℕ := 5

theorem onions_on_scale :
  W = (N - removed) * avg_remaining + removed * avg_removed ∧ N = 40 := by sorry

end NUMINAMATH_CALUDE_onions_on_scale_l4159_415984


namespace NUMINAMATH_CALUDE_base_conversion_arithmetic_l4159_415916

/-- Converts a number from base b to base 10 --/
def toBase10 (digits : List Nat) (b : Nat) : Nat :=
  digits.foldr (fun d acc => d + b * acc) 0

/-- Rounds a rational number to the nearest integer --/
def roundToNearest (x : ℚ) : ℤ :=
  (x + 1/2).floor

theorem base_conversion_arithmetic : 
  let base8_2468 := toBase10 [8, 6, 4, 2] 8
  let base4_110 := toBase10 [0, 1, 1] 4
  let base9_3571 := toBase10 [1, 7, 5, 3] 9
  let base10_1357 := 1357
  roundToNearest (base8_2468 / base4_110) - base9_3571 + base10_1357 = -1232 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_arithmetic_l4159_415916


namespace NUMINAMATH_CALUDE_triangle_side_length_l4159_415940

theorem triangle_side_length (X Y Z : ℝ) : 
  -- Triangle XYZ with right angle at X
  X^2 + Y^2 = Z^2 →
  -- YZ = 20
  Z = 20 →
  -- tan Z = 3 cos Y
  (Real.tan Z) = 3 * (Real.cos Y) →
  -- XY = (40√2)/3
  Y = (40 * Real.sqrt 2) / 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l4159_415940


namespace NUMINAMATH_CALUDE_hypotenuse_length_squared_l4159_415943

/-- Given complex numbers p, q, and r that are zeros of a polynomial Q(z) = z^3 + sz + t,
    if |p|^2 + |q|^2 + |r|^2 = 300, p + q + r = 0, and p, q, and r form a right triangle
    in the complex plane, then the square of the length of the hypotenuse of this triangle is 450. -/
theorem hypotenuse_length_squared (p q r s t : ℂ) : 
  (Q : ℂ → ℂ) = (fun z ↦ z^3 + s*z + t) →
  p^3 + s*p + t = 0 →
  q^3 + s*q + t = 0 →
  r^3 + s*r + t = 0 →
  Complex.abs p^2 + Complex.abs q^2 + Complex.abs r^2 = 300 →
  p + q + r = 0 →
  ∃ (a b : ℝ), Complex.abs (p - q)^2 = a^2 ∧ Complex.abs (q - r)^2 = b^2 ∧ Complex.abs (p - r)^2 = a^2 + b^2 →
  Complex.abs (p - r)^2 = 450 :=
by sorry

end NUMINAMATH_CALUDE_hypotenuse_length_squared_l4159_415943


namespace NUMINAMATH_CALUDE_inverse_of_B_squared_l4159_415958

open Matrix

theorem inverse_of_B_squared (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B⁻¹ = !![1, 4; -2, -7]) : 
  (B^2)⁻¹ = !![(-7), (-24); 12, 41] := by
sorry

end NUMINAMATH_CALUDE_inverse_of_B_squared_l4159_415958


namespace NUMINAMATH_CALUDE_prime_power_sum_l4159_415982

theorem prime_power_sum (w x y z : ℕ) :
  3^w * 5^x * 7^y * 11^z = 2310 →
  3*w + 5*x + 7*y + 11*z = 26 := by
sorry

end NUMINAMATH_CALUDE_prime_power_sum_l4159_415982


namespace NUMINAMATH_CALUDE_fraction_simplification_l4159_415922

theorem fraction_simplification (a b : ℝ) (h : b ≠ 0) : (3 * a) / (3 * b) = a / b := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l4159_415922


namespace NUMINAMATH_CALUDE_eighth_group_sample_digit_l4159_415976

/-- Represents the systematic sampling method described in the problem -/
def systematicSample (t : ℕ) (k : ℕ) : ℕ :=
  (t + k) % 10

/-- The theorem to prove -/
theorem eighth_group_sample_digit (t : ℕ) (h : t = 7) : systematicSample t 8 = 5 := by
  sorry

end NUMINAMATH_CALUDE_eighth_group_sample_digit_l4159_415976


namespace NUMINAMATH_CALUDE_z_range_is_closed_interval_l4159_415956

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop := x^2/16 + y^2/9 = 1

-- Define z as a function of x and y
def z (x y : ℝ) : ℝ := x + y

-- Theorem statement
theorem z_range_is_closed_interval :
  ∃ (a b : ℝ), a = -5 ∧ b = 5 ∧
  (∀ (x y : ℝ), ellipse_equation x y → a ≤ z x y ∧ z x y ≤ b) ∧
  (∀ t : ℝ, a ≤ t ∧ t ≤ b → ∃ (x y : ℝ), ellipse_equation x y ∧ z x y = t) :=
sorry

end NUMINAMATH_CALUDE_z_range_is_closed_interval_l4159_415956


namespace NUMINAMATH_CALUDE_hockey_arena_rows_l4159_415942

/-- The minimum number of rows required in a hockey arena -/
def min_rows (seats_per_row : ℕ) (total_students : ℕ) (max_students_per_school : ℕ) : ℕ :=
  let schools_per_row := seats_per_row / max_students_per_school
  let total_schools := (total_students + max_students_per_school - 1) / max_students_per_school
  (total_schools + schools_per_row - 1) / schools_per_row

/-- Theorem stating the minimum number of rows required for the given conditions -/
theorem hockey_arena_rows :
  min_rows 168 2016 45 = 16 := by
  sorry

#eval min_rows 168 2016 45

end NUMINAMATH_CALUDE_hockey_arena_rows_l4159_415942


namespace NUMINAMATH_CALUDE_fraction_comparison_l4159_415919

theorem fraction_comparison : 
  (14/10 : ℚ) = 7/5 ∧ 
  (1 + 2/5 : ℚ) = 7/5 ∧ 
  (1 + 14/35 : ℚ) = 7/5 ∧ 
  (1 + 4/20 : ℚ) ≠ 7/5 ∧ 
  (1 + 3/15 : ℚ) ≠ 7/5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparison_l4159_415919


namespace NUMINAMATH_CALUDE_a_months_is_32_l4159_415987

/-- Represents the pasture rental problem -/
structure PastureRental where
  total_cost : ℕ
  a_horses : ℕ
  b_horses : ℕ
  c_horses : ℕ
  b_months : ℕ
  c_months : ℕ
  b_payment : ℕ

/-- Calculates the number of months a put in the horses -/
def calculate_a_months (p : PastureRental) : ℕ :=
  ((p.total_cost - p.b_payment - p.c_horses * p.c_months) / p.a_horses)

/-- Theorem stating that a put in the horses for 32 months -/
theorem a_months_is_32 (p : PastureRental) 
  (h1 : p.total_cost = 841)
  (h2 : p.a_horses = 12)
  (h3 : p.b_horses = 16)
  (h4 : p.c_horses = 18)
  (h5 : p.b_months = 9)
  (h6 : p.c_months = 6)
  (h7 : p.b_payment = 348) :
  calculate_a_months p = 32 := by
  sorry

#eval calculate_a_months { 
  total_cost := 841, 
  a_horses := 12, 
  b_horses := 16, 
  c_horses := 18, 
  b_months := 9, 
  c_months := 6, 
  b_payment := 348 
}

end NUMINAMATH_CALUDE_a_months_is_32_l4159_415987


namespace NUMINAMATH_CALUDE_f_max_value_l4159_415967

/-- The quadratic function f(x) = -9x^2 + 27x + 15 -/
def f (x : ℝ) : ℝ := -9 * x^2 + 27 * x + 15

/-- The maximum value of f(x) is 35.25 -/
theorem f_max_value : ∃ (M : ℝ), M = 35.25 ∧ ∀ (x : ℝ), f x ≤ M := by
  sorry

end NUMINAMATH_CALUDE_f_max_value_l4159_415967


namespace NUMINAMATH_CALUDE_inequality_proof_l4159_415971

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (2 * Real.sqrt (a * b)) / (Real.sqrt a + Real.sqrt b) ≤ (a * b) ^ (1/4) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4159_415971


namespace NUMINAMATH_CALUDE_power_equality_l4159_415949

theorem power_equality (x : ℝ) : (1/8 : ℝ) * 2^36 = 4^x → x = 16.5 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l4159_415949


namespace NUMINAMATH_CALUDE_shaded_region_area_l4159_415947

/-- The number of congruent squares in the shaded region -/
def total_squares : ℕ := 20

/-- The number of shaded squares in the larger square -/
def squares_in_larger : ℕ := 4

/-- The length of the diagonal of the larger square in cm -/
def diagonal_length : ℝ := 10

/-- The area of the entire shaded region in square cm -/
def shaded_area : ℝ := 250

theorem shaded_region_area :
  ∀ (total_squares squares_in_larger : ℕ) (diagonal_length shaded_area : ℝ),
  total_squares = 20 →
  squares_in_larger = 4 →
  diagonal_length = 10 →
  shaded_area = total_squares * (diagonal_length / (2 * Real.sqrt 2))^2 →
  shaded_area = 250 := by
sorry

end NUMINAMATH_CALUDE_shaded_region_area_l4159_415947


namespace NUMINAMATH_CALUDE_perp_necessary_not_sufficient_l4159_415948

-- Define the plane α
variable (α : Plane)

-- Define lines l, m, and n
variable (l m n : Line)

-- Define the property that a line is in a plane
def line_in_plane (l : Line) (p : Plane) : Prop := sorry

-- Define perpendicularity between a line and a plane
def line_perp_plane (l : Line) (p : Plane) : Prop := sorry

-- Define perpendicularity between two lines
def line_perp_line (l1 l2 : Line) : Prop := sorry

-- State the theorem
theorem perp_necessary_not_sufficient :
  (line_in_plane m α ∧ line_in_plane n α) →
  (∀ l, line_perp_plane l α → (line_perp_line l m ∧ line_perp_line l n)) ∧
  (∃ l, line_perp_line l m ∧ line_perp_line l n ∧ ¬line_perp_plane l α) := by
  sorry

end NUMINAMATH_CALUDE_perp_necessary_not_sufficient_l4159_415948


namespace NUMINAMATH_CALUDE_find_x_l4159_415921

theorem find_x : ∃ x : ℝ, 
  (∃ y : ℝ, y = 1.5 * x ∧ 0.5 * x - 10 = 0.25 * y) → x = 80 := by
  sorry

end NUMINAMATH_CALUDE_find_x_l4159_415921


namespace NUMINAMATH_CALUDE_binomial_coefficient_x_plus_two_to_seven_coefficient_of_x_fifth_power_l4159_415901

theorem binomial_coefficient_x_plus_two_to_seven (x : ℝ) : 
  (Finset.range 8).sum (λ k => (Nat.choose 7 k : ℝ) * x^k * 2^(7-k)) = 
    x^7 + 14*x^6 + 84*x^5 + 280*x^4 + 560*x^3 + 672*x^2 + 448*x + 128 :=
by sorry

theorem coefficient_of_x_fifth_power : 
  (Finset.range 8).sum (λ k => (Nat.choose 7 k : ℝ) * 1^k * 2^(7-k) * 
    (if k = 5 then 1 else 0)) = 84 :=
by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_x_plus_two_to_seven_coefficient_of_x_fifth_power_l4159_415901


namespace NUMINAMATH_CALUDE_area_maximized_at_m_pm1_l4159_415909

/-- Ellipse E with equation x²/6 + y²/2 = 1 -/
def ellipse_E (x y : ℝ) : Prop := x^2 / 6 + y^2 / 2 = 1

/-- Focus F₁ at (-2, 0) -/
def F₁ : ℝ × ℝ := (-2, 0)

/-- Line l with equation x - my - 2 = 0 -/
def line_l (m x y : ℝ) : Prop := x - m * y - 2 = 0

/-- Intersection points of ellipse E and line l -/
def intersection_points (m : ℝ) : Set (ℝ × ℝ) :=
  {p | ellipse_E p.1 p.2 ∧ line_l m p.1 p.2}

/-- Area of quadrilateral AF₁BC -/
noncomputable def area_AF₁BC (m : ℝ) : ℝ := sorry

/-- Theorem: Area of AF₁BC is maximized when m = ±1 -/
theorem area_maximized_at_m_pm1 :
  ∀ m : ℝ, area_AF₁BC m ≤ area_AF₁BC 1 ∧ area_AF₁BC m ≤ area_AF₁BC (-1) :=
sorry

end NUMINAMATH_CALUDE_area_maximized_at_m_pm1_l4159_415909


namespace NUMINAMATH_CALUDE_fraction_meaningful_iff_not_equal_two_l4159_415944

theorem fraction_meaningful_iff_not_equal_two (x : ℝ) : 
  (∃ y : ℝ, y = 7 / (x - 2)) ↔ x ≠ 2 := by
sorry

end NUMINAMATH_CALUDE_fraction_meaningful_iff_not_equal_two_l4159_415944


namespace NUMINAMATH_CALUDE_bacteria_growth_3hours_l4159_415913

/-- The number of bacteria after a given time, given an initial population and doubling time. -/
def bacteriaPopulation (initialPopulation : ℕ) (doublingTimeMinutes : ℕ) (totalTimeMinutes : ℕ) : ℕ :=
  initialPopulation * 2 ^ (totalTimeMinutes / doublingTimeMinutes)

/-- Theorem stating that after 3 hours, starting with 1 bacterium that doubles every 20 minutes, 
    the population will be 512. -/
theorem bacteria_growth_3hours :
  bacteriaPopulation 1 20 180 = 512 := by
  sorry

end NUMINAMATH_CALUDE_bacteria_growth_3hours_l4159_415913


namespace NUMINAMATH_CALUDE_right_triangle_to_square_l4159_415954

theorem right_triangle_to_square (a b : ℝ) : 
  b = 10 → -- longer leg is 10
  a * b / 2 = a^2 → -- area of triangle equals area of square
  b = 2 * a → -- longer leg is twice the shorter leg
  a = 5 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_to_square_l4159_415954


namespace NUMINAMATH_CALUDE_sum_of_solutions_squared_equation_l4159_415988

theorem sum_of_solutions_squared_equation : 
  ∃ (x₁ x₂ : ℝ), (x₁ - 4)^2 = 49 ∧ (x₂ - 4)^2 = 49 ∧ x₁ + x₂ = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_squared_equation_l4159_415988


namespace NUMINAMATH_CALUDE_negative_eight_to_negative_four_thirds_l4159_415927

theorem negative_eight_to_negative_four_thirds :
  Real.rpow (-8 : ℝ) (-4/3 : ℝ) = (1/16 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_negative_eight_to_negative_four_thirds_l4159_415927


namespace NUMINAMATH_CALUDE_tank_capacity_l4159_415939

/-- Represents a tank with a leak and an inlet pipe. -/
structure Tank where
  capacity : ℝ
  leak_empty_time : ℝ
  inlet_rate : ℝ
  combined_empty_time : ℝ

/-- Theorem stating the relationship between tank properties and its capacity. -/
theorem tank_capacity (t : Tank) 
  (h1 : t.leak_empty_time = 6)
  (h2 : t.inlet_rate = 3 * 60)  -- 3 liters per minute converted to per hour
  (h3 : t.combined_empty_time = 8) :
  t.capacity = 4320 / 7 := by
  sorry

#check tank_capacity

end NUMINAMATH_CALUDE_tank_capacity_l4159_415939


namespace NUMINAMATH_CALUDE_elon_has_13_teslas_l4159_415908

/-- The number of teslas Chris has -/
def chris_teslas : ℕ := 6

/-- The number of teslas Sam has -/
def sam_teslas : ℕ := chris_teslas / 2

/-- The number of teslas Elon has -/
def elon_teslas : ℕ := sam_teslas + 10

/-- Theorem stating that Elon has 13 teslas -/
theorem elon_has_13_teslas : elon_teslas = 13 := by
  sorry

end NUMINAMATH_CALUDE_elon_has_13_teslas_l4159_415908


namespace NUMINAMATH_CALUDE_acme_savings_at_min_shirts_l4159_415973

/-- Acme T-Shirt Company's pricing function -/
def acme_cost (n : ℕ) : ℚ :=
  if n ≤ 20 then 60 + 10 * n
  else (60 + 10 * n) * (9/10)

/-- Beta T-Shirt Company's pricing function -/
def beta_cost (n : ℕ) : ℚ := 15 * n

/-- The minimum number of shirts for which Acme is cheaper than Beta -/
def min_shirts_for_acme_savings : ℕ := 13

theorem acme_savings_at_min_shirts :
  acme_cost min_shirts_for_acme_savings < beta_cost min_shirts_for_acme_savings ∧
  ∀ k : ℕ, k < min_shirts_for_acme_savings →
    acme_cost k ≥ beta_cost k :=
by sorry

end NUMINAMATH_CALUDE_acme_savings_at_min_shirts_l4159_415973


namespace NUMINAMATH_CALUDE_max_product_sum_l4159_415997

theorem max_product_sum (X Y Z : ℕ) (sum_constraint : X + Y + Z = 15) :
  (∀ a b c : ℕ, a + b + c = 15 → X * Y * Z + X * Y + Y * Z + Z * X ≥ a * b * c + a * b + b * c + c * a) ∧
  X * Y * Z + X * Y + Y * Z + Z * X = 200 :=
sorry

end NUMINAMATH_CALUDE_max_product_sum_l4159_415997


namespace NUMINAMATH_CALUDE_inequality_upper_bound_upper_bound_tight_smallest_upper_bound_l4159_415955

theorem inequality_upper_bound (x y z w : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w > 0) :
  Real.sqrt (x / (y + z + w)) + Real.sqrt (y / (x + z + w)) +
  Real.sqrt (z / (x + y + w)) + Real.sqrt (w / (x + y + z)) ≤ 4 / Real.sqrt 3 :=
by sorry

theorem upper_bound_tight : 
  ∃ (x y z w : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0 ∧
  Real.sqrt (x / (y + z + w)) + Real.sqrt (y / (x + z + w)) +
  Real.sqrt (z / (x + y + w)) + Real.sqrt (w / (x + y + z)) = 4 / Real.sqrt 3 :=
by sorry

theorem smallest_upper_bound :
  ∀ M : ℝ, M < 4 / Real.sqrt 3 →
  ∃ (x y z w : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0 ∧
  Real.sqrt (x / (y + z + w)) + Real.sqrt (y / (x + z + w)) +
  Real.sqrt (z / (x + y + w)) + Real.sqrt (w / (x + y + z)) > M :=
by sorry

end NUMINAMATH_CALUDE_inequality_upper_bound_upper_bound_tight_smallest_upper_bound_l4159_415955


namespace NUMINAMATH_CALUDE_subset_condition_A_eq_zero_four_six_A_proper_subsets_l4159_415965

def M : Set ℝ := {x | x^2 - 5*x + 6 = 0}
def N (a : ℝ) : Set ℝ := {x | a*x = 12}

def A : Set ℝ := {a | N a ⊆ M}

theorem subset_condition (a : ℝ) : a ∈ A ↔ a = 0 ∨ a = 4 ∨ a = 6 := by sorry

theorem A_eq_zero_four_six : A = {0, 4, 6} := by sorry

def proper_subsets (S : Set ℝ) : Set (Set ℝ) :=
  {T | T ⊆ S ∧ T ≠ ∅ ∧ T ≠ S}

theorem A_proper_subsets :
  proper_subsets A = {{0}, {4}, {6}, {0, 4}, {0, 6}, {4, 6}} := by sorry

end NUMINAMATH_CALUDE_subset_condition_A_eq_zero_four_six_A_proper_subsets_l4159_415965


namespace NUMINAMATH_CALUDE_rectangular_solid_volume_l4159_415990

theorem rectangular_solid_volume (x y z : ℝ) 
  (h1 : x * y = 3) 
  (h2 : x * z = 5) 
  (h3 : y * z = 15) : 
  x * y * z = 15 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_volume_l4159_415990


namespace NUMINAMATH_CALUDE_tangent_line_at_one_l4159_415931

open Real

noncomputable def f (x : ℝ) := log x - 3 * x

theorem tangent_line_at_one :
  ∃ (m b : ℝ), (∀ x y, y = m * x + b ↔ 2 * x + y + 1 = 0) ∧
               m = deriv f 1 ∧
               f 1 = m * 1 + b := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_l4159_415931


namespace NUMINAMATH_CALUDE_function_composition_property_l4159_415953

/-- Given a function f(x) = (ax + b) / (cx + d), prove that if f(f(f(1))) = 1 and f(f(f(2))) = 3, then f(1) = 1. -/
theorem function_composition_property (a b c d : ℝ) :
  let f (x : ℝ) := (a * x + b) / (c * x + d)
  (f (f (f 1)) = 1) → (f (f (f 2)) = 3) → (f 1 = 1) := by
  sorry

end NUMINAMATH_CALUDE_function_composition_property_l4159_415953


namespace NUMINAMATH_CALUDE_min_book_cover_area_l4159_415945

/-- Given a book cover with reported dimensions of 5 inches by 7 inches,
    where each dimension can vary by ±0.5 inches, the minimum possible area
    of the book cover is 29.25 square inches. -/
theorem min_book_cover_area (reported_length : ℝ) (reported_width : ℝ)
    (actual_length : ℝ) (actual_width : ℝ) :
  reported_length = 5 →
  reported_width = 7 →
  abs (actual_length - reported_length) ≤ 0.5 →
  abs (actual_width - reported_width) ≤ 0.5 →
  ∀ area : ℝ, area = actual_length * actual_width →
    area ≥ 29.25 :=
by sorry

end NUMINAMATH_CALUDE_min_book_cover_area_l4159_415945


namespace NUMINAMATH_CALUDE_contrapositive_statement_l4159_415918

theorem contrapositive_statement (a b : ℝ) :
  (a > 0 ∧ a + b < 0) → b < 0 := by
  sorry

end NUMINAMATH_CALUDE_contrapositive_statement_l4159_415918


namespace NUMINAMATH_CALUDE_system_solution_l4159_415989

theorem system_solution : 
  ∀ x y : ℝ, (y^2 + x*y = 15 ∧ x^2 + x*y = 10) ↔ ((x = 2 ∧ y = 3) ∨ (x = -2 ∧ y = -3)) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l4159_415989


namespace NUMINAMATH_CALUDE_angle_C_in_triangle_l4159_415929

theorem angle_C_in_triangle (A B C : ℝ) (h1 : 4 * Real.sin A + 2 * Real.cos B = 4)
    (h2 : (1/2) * Real.sin B + Real.cos A = Real.sqrt 3 / 2) : C = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_in_triangle_l4159_415929


namespace NUMINAMATH_CALUDE_a0_value_l4159_415905

theorem a0_value (x : ℝ) (a0 a1 a2 a3 a4 a5 : ℝ) 
  (h : ∀ x, (x + 1)^5 = a0 + a1*(x - 1) + a2*(x - 1)^2 + a3*(x - 1)^3 + a4*(x - 1)^4 + a5*(x - 1)^5) : 
  a0 = 32 := by
sorry

end NUMINAMATH_CALUDE_a0_value_l4159_415905


namespace NUMINAMATH_CALUDE_domain_of_g_l4159_415906

-- Define the original function f
def f : ℝ → ℝ := sorry

-- Define the new function g
def g (x : ℝ) : ℝ := f (2 * x + 1)

-- State the theorem
theorem domain_of_g :
  (∀ x, f x ≠ 0 → x ∈ Set.Icc (-2) 3) →
  (∀ x, g x ≠ 0 → x ∈ Set.Icc (-3/2) 1) :=
sorry

end NUMINAMATH_CALUDE_domain_of_g_l4159_415906


namespace NUMINAMATH_CALUDE_f_properties_l4159_415917

noncomputable def f (x : ℝ) := (1 + Real.sqrt 3 * Real.tan x) * (Real.cos x)^2

theorem f_properties :
  (∀ x : ℝ, f x ≠ 0 → ∃ k : ℤ, x ≠ Real.pi / 2 + k * Real.pi) ∧
  (∀ x : ℝ, f (x + Real.pi) = f x) ∧
  (∀ x : ℝ, x ∈ Set.Ioo 0 (Real.pi / 2) → f x ∈ Set.Ioc 0 (3 / 2)) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l4159_415917


namespace NUMINAMATH_CALUDE_smallest_n_with_four_trailing_zeros_l4159_415970

def is_divisible_by_10000 (n : ℕ) : Prop :=
  (n.choose 4) % 10000 = 0

theorem smallest_n_with_four_trailing_zeros : 
  ∀ k : ℕ, k ≥ 4 ∧ k < 8128 → ¬(is_divisible_by_10000 k) ∧ is_divisible_by_10000 8128 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_with_four_trailing_zeros_l4159_415970


namespace NUMINAMATH_CALUDE_orthocenter_quadrilateral_congruence_l4159_415928

/-- A cyclic quadrilateral is a quadrilateral whose vertices all lie on a single circle. -/
def CyclicQuadrilateral (A B C D : Point) : Prop := sorry

/-- The orthocenter of a triangle is the point where the three altitudes of the triangle intersect. -/
def Orthocenter (H A B C : Point) : Prop := sorry

/-- Two quadrilaterals are congruent if they have the same shape and size. -/
def CongruentQuadrilaterals (A B C D A' B' C' D' : Point) : Prop := sorry

theorem orthocenter_quadrilateral_congruence 
  (A B C D A' B' C' D' : Point) :
  CyclicQuadrilateral A B C D →
  Orthocenter A' B C D →
  Orthocenter B' A C D →
  Orthocenter C' A B D →
  Orthocenter D' A B C →
  CongruentQuadrilaterals A B C D A' B' C' D' :=
sorry

end NUMINAMATH_CALUDE_orthocenter_quadrilateral_congruence_l4159_415928


namespace NUMINAMATH_CALUDE_defective_pencils_count_l4159_415925

/-- The probability of selecting 3 non-defective pencils out of N non-defective pencils from a total of 6 pencils. -/
def probability (N : ℕ) : ℚ :=
  (Nat.choose N 3 : ℚ) / (Nat.choose 6 3 : ℚ)

/-- The number of defective pencils in a box of 6 pencils. -/
def num_defective (N : ℕ) : ℕ := 6 - N

theorem defective_pencils_count :
  ∃ N : ℕ, N ≤ 6 ∧ probability N = 1/5 ∧ num_defective N = 2 := by
  sorry

#check defective_pencils_count

end NUMINAMATH_CALUDE_defective_pencils_count_l4159_415925


namespace NUMINAMATH_CALUDE_cost_per_set_is_correct_l4159_415900

/-- The cost of each set of drill bits -/
def cost_per_set : ℝ := 6

/-- The number of sets bought -/
def num_sets : ℕ := 5

/-- The tax rate -/
def tax_rate : ℝ := 0.1

/-- The total amount paid -/
def total_paid : ℝ := 33

/-- Theorem stating that the cost per set is correct given the conditions -/
theorem cost_per_set_is_correct : 
  num_sets * cost_per_set * (1 + tax_rate) = total_paid :=
sorry

end NUMINAMATH_CALUDE_cost_per_set_is_correct_l4159_415900


namespace NUMINAMATH_CALUDE_angle_equality_l4159_415951

theorem angle_equality (θ : Real) (h1 : Real.cos (60 * π / 180) = Real.cos (45 * π / 180) * Real.cos θ) 
  (h2 : 0 ≤ θ) (h3 : θ ≤ π / 2) : θ = 45 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_angle_equality_l4159_415951


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l4159_415933

theorem smallest_prime_divisor_of_sum (n : ℕ) : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ (3^19 + 11^13) ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ (3^19 + 11^13) → p ≤ q :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l4159_415933


namespace NUMINAMATH_CALUDE_orthogonal_vectors_x_value_l4159_415937

def vector_a (x : ℝ) : Fin 2 → ℝ := ![x, 2]
def vector_b : Fin 2 → ℝ := ![2, -1]

def orthogonal (u v : Fin 2 → ℝ) : Prop :=
  (u 0) * (v 0) + (u 1) * (v 1) = 0

theorem orthogonal_vectors_x_value :
  ∃ x : ℝ, orthogonal (vector_a x) vector_b ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_orthogonal_vectors_x_value_l4159_415937


namespace NUMINAMATH_CALUDE_quadratic_function_equality_l4159_415972

theorem quadratic_function_equality (a b c d : ℝ) : 
  (∀ x, (x^2 + a*x + b) = ((2*x + 1)^2 + a*(2*x + 1) + b)) → 
  (∀ x, 4*(x^2 + c*x + d) = ((2*x + 1)^2 + a*(2*x + 1) + b)) → 
  (∀ x, 2*x + a = 2*x + c) → 
  (5^2 + 5*a + b = 30) → 
  (a = 2 ∧ b = -5 ∧ c = 2 ∧ d = -1/2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_equality_l4159_415972


namespace NUMINAMATH_CALUDE_assignment_schemes_proof_l4159_415914

/-- The number of ways to assign 3 out of 5 volunteers to 3 distinct tasks -/
def assignment_schemes : ℕ := 60

/-- The total number of volunteers -/
def total_volunteers : ℕ := 5

/-- The number of volunteers to be selected -/
def selected_volunteers : ℕ := 3

/-- The number of tasks -/
def num_tasks : ℕ := 3

theorem assignment_schemes_proof :
  assignment_schemes = (total_volunteers.factorial) / ((total_volunteers - selected_volunteers).factorial) :=
by sorry

end NUMINAMATH_CALUDE_assignment_schemes_proof_l4159_415914


namespace NUMINAMATH_CALUDE_desired_line_equation_l4159_415975

-- Define the lines
def line1 (x y : ℝ) : Prop := 2 * x - y + 4 = 0
def line2 (x y : ℝ) : Prop := x - y + 5 = 0
def line3 (x y : ℝ) : Prop := x - 2 * y = 0

-- Define the intersection point
def intersection (x y : ℝ) : Prop := line1 x y ∧ line2 x y

-- Define perpendicularity
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

-- The theorem to prove
theorem desired_line_equation : 
  ∃ (x y : ℝ), intersection x y ∧ 
  (∃ (m : ℝ), perpendicular m (1/2) ∧ 
  (∀ (x' y' : ℝ), 2 * x' + y' - 8 = 0 ↔ y' - y = m * (x' - x))) :=
sorry

end NUMINAMATH_CALUDE_desired_line_equation_l4159_415975


namespace NUMINAMATH_CALUDE_correct_categorization_l4159_415935

def numbers : List ℚ := [2020, 1, -1, -2021, 1/2, 1/10, -1/3, -3/4, 0, 1/5]

def is_integer (q : ℚ) : Prop := ∃ (n : ℤ), q = n
def is_positive_integer (q : ℚ) : Prop := ∃ (n : ℕ), q = n ∧ n > 0
def is_negative_integer (q : ℚ) : Prop := ∃ (n : ℤ), q = n ∧ n < 0
def is_positive_fraction (q : ℚ) : Prop := q > 0 ∧ q < 1
def is_negative_fraction (q : ℚ) : Prop := q < 0 ∧ q > -1

def integers : List ℚ := [2020, 1, -1, -2021, 0]
def positive_integers : List ℚ := [2020, 1]
def negative_integers : List ℚ := [-1, -2021]
def positive_fractions : List ℚ := [1/2, 1/10, 1/5]
def negative_fractions : List ℚ := [-1/3, -3/4]

theorem correct_categorization :
  (∀ q ∈ integers, is_integer q) ∧
  (∀ q ∈ positive_integers, is_positive_integer q) ∧
  (∀ q ∈ negative_integers, is_negative_integer q) ∧
  (∀ q ∈ positive_fractions, is_positive_fraction q) ∧
  (∀ q ∈ negative_fractions, is_negative_fraction q) ∧
  (∀ q ∈ numbers, 
    (is_integer q → q ∈ integers) ∧
    (is_positive_integer q → q ∈ positive_integers) ∧
    (is_negative_integer q → q ∈ negative_integers) ∧
    (is_positive_fraction q → q ∈ positive_fractions) ∧
    (is_negative_fraction q → q ∈ negative_fractions)) :=
by sorry

end NUMINAMATH_CALUDE_correct_categorization_l4159_415935


namespace NUMINAMATH_CALUDE_julie_school_year_hours_l4159_415964

/-- Calculates the number of hours Julie needs to work per week during the school year -/
def school_year_hours_per_week (summer_hours_per_week : ℕ) (summer_weeks : ℕ) (summer_earnings : ℕ) 
  (school_year_weeks : ℕ) (school_year_earnings : ℕ) : ℕ :=
  let summer_total_hours := summer_hours_per_week * summer_weeks
  let hourly_wage := summer_earnings / summer_total_hours
  let school_year_total_hours := school_year_earnings / hourly_wage
  school_year_total_hours / school_year_weeks

/-- Theorem stating that Julie needs to work 20 hours per week during the school year -/
theorem julie_school_year_hours : 
  school_year_hours_per_week 60 8 6000 40 10000 = 20 := by
  sorry

end NUMINAMATH_CALUDE_julie_school_year_hours_l4159_415964


namespace NUMINAMATH_CALUDE_valid_three_digit_numbers_count_l4159_415961

/-- The count of three-digit numbers -/
def total_three_digit_numbers : ℕ := 900

/-- The count of three-digit numbers with two adjacent identical digits -/
def numbers_with_two_adjacent_identical_digits : ℕ := 153

/-- The count of valid three-digit numbers according to the problem conditions -/
def valid_three_digit_numbers : ℕ := total_three_digit_numbers - numbers_with_two_adjacent_identical_digits

theorem valid_three_digit_numbers_count :
  valid_three_digit_numbers = 747 := by
  sorry

end NUMINAMATH_CALUDE_valid_three_digit_numbers_count_l4159_415961


namespace NUMINAMATH_CALUDE_meal_combinations_count_l4159_415907

/-- The number of main dishes available on the menu -/
def num_main_dishes : ℕ := 12

/-- The number of appetizers available to choose from -/
def num_appetizers : ℕ := 5

/-- The number of people ordering -/
def num_people : ℕ := 2

/-- Calculates the number of different meal combinations -/
def meal_combinations : ℕ := num_main_dishes ^ num_people * num_appetizers

/-- Theorem stating that the number of meal combinations is 720 -/
theorem meal_combinations_count : meal_combinations = 720 := by sorry

end NUMINAMATH_CALUDE_meal_combinations_count_l4159_415907


namespace NUMINAMATH_CALUDE_committee_formation_count_l4159_415915

def total_members : ℕ := 25
def male_members : ℕ := 15
def female_members : ℕ := 10
def committee_size : ℕ := 5
def min_females : ℕ := 2

theorem committee_formation_count : 
  (Finset.sum (Finset.range (committee_size - min_females + 1))
    (fun k => Nat.choose female_members (k + min_females) * 
              Nat.choose male_members (committee_size - k - min_females))) = 36477 := by
  sorry

end NUMINAMATH_CALUDE_committee_formation_count_l4159_415915


namespace NUMINAMATH_CALUDE_lateral_surface_area_theorem_l4159_415946

/-- Represents a regular quadrilateral pyramid -/
structure RegularQuadrilateralPyramid where
  /-- The dihedral angle at the lateral edge -/
  dihedral_angle : ℝ
  /-- The area of the diagonal section -/
  diagonal_section_area : ℝ

/-- The lateral surface area of a regular quadrilateral pyramid -/
noncomputable def lateral_surface_area (p : RegularQuadrilateralPyramid) : ℝ :=
  4 * p.diagonal_section_area

/-- Theorem: The lateral surface area of a regular quadrilateral pyramid is 4S,
    where S is the area of its diagonal section, given that the dihedral angle
    at the lateral edge is 120° -/
theorem lateral_surface_area_theorem (p : RegularQuadrilateralPyramid) 
  (h : p.dihedral_angle = 120) : 
  lateral_surface_area p = 4 * p.diagonal_section_area := by
  sorry

end NUMINAMATH_CALUDE_lateral_surface_area_theorem_l4159_415946


namespace NUMINAMATH_CALUDE_max_abs_z5_l4159_415980

theorem max_abs_z5 (z₁ z₂ z₃ z₄ z₅ : ℂ) 
  (h1 : Complex.abs z₁ ≤ 1)
  (h2 : Complex.abs z₂ ≤ 1)
  (h3 : Complex.abs (2 * z₃ - (z₁ + z₂)) ≤ Complex.abs (z₁ - z₂))
  (h4 : Complex.abs (2 * z₄ - (z₁ + z₂)) ≤ Complex.abs (z₁ - z₂))
  (h5 : Complex.abs (2 * z₅ - (z₃ + z₄)) ≤ Complex.abs (z₃ - z₄)) :
  Complex.abs z₅ ≤ Real.sqrt 3 ∧ ∃ z₁ z₂ z₃ z₄ z₅, 
    Complex.abs z₁ ≤ 1 ∧ 
    Complex.abs z₂ ≤ 1 ∧
    Complex.abs (2 * z₃ - (z₁ + z₂)) ≤ Complex.abs (z₁ - z₂) ∧
    Complex.abs (2 * z₄ - (z₁ + z₂)) ≤ Complex.abs (z₁ - z₂) ∧
    Complex.abs (2 * z₅ - (z₃ + z₄)) ≤ Complex.abs (z₃ - z₄) ∧
    Complex.abs z₅ = Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_max_abs_z5_l4159_415980


namespace NUMINAMATH_CALUDE_identity_element_is_negative_four_l4159_415938

-- Define the operation ⊕
def circplus (a b : ℝ) : ℝ := a + b + 4

-- Define the property of being an identity element for ⊕
def is_identity (e : ℝ) : Prop :=
  ∀ a : ℝ, circplus e a = a

-- Theorem statement
theorem identity_element_is_negative_four :
  ∃ e : ℝ, is_identity e ∧ e = -4 := by
  sorry

end NUMINAMATH_CALUDE_identity_element_is_negative_four_l4159_415938


namespace NUMINAMATH_CALUDE_product_and_sum_with_reciprocal_bounds_l4159_415903

/-- Given positive real numbers a and b that sum to 1, this theorem proves
    the range of their product and the minimum value of their product plus its reciprocal. -/
theorem product_and_sum_with_reciprocal_bounds (a b : ℝ) 
    (ha : a > 0) (hb : b > 0) (hab : a + b = 1) : 
    (0 < a * b ∧ a * b ≤ 1/4) ∧
    (∀ x : ℝ, x > 0 → x ≤ 1/4 → a * b + 1 / (a * b) ≤ x + 1 / x) ∧
    a * b + 1 / (a * b) = 17/4 := by
  sorry

end NUMINAMATH_CALUDE_product_and_sum_with_reciprocal_bounds_l4159_415903


namespace NUMINAMATH_CALUDE_new_average_weight_l4159_415968

/-- Given 19 students with an average weight of 15 kg and a new student weighing 11 kg,
    the new average weight of all 20 students is 14.8 kg. -/
theorem new_average_weight (initial_students : ℕ) (initial_avg_weight : ℝ) 
  (new_student_weight : ℝ) : 
  initial_students = 19 → 
  initial_avg_weight = 15 → 
  new_student_weight = 11 → 
  (initial_students * initial_avg_weight + new_student_weight) / (initial_students + 1) = 14.8 := by
  sorry

end NUMINAMATH_CALUDE_new_average_weight_l4159_415968


namespace NUMINAMATH_CALUDE_a_equals_permutation_l4159_415920

-- Define a as the product n(n-1)(n-2)...(n-50)
def a (n : ℕ) : ℕ := (List.range 51).foldl (λ acc i => acc * (n - i)) n

-- Define the permutation function A_n^k
def permutation (n k : ℕ) : ℕ := (List.range k).foldl (λ acc i => acc * (n - i)) 1

-- Theorem statement
theorem a_equals_permutation (n : ℕ) : a n = permutation n 51 := by sorry

end NUMINAMATH_CALUDE_a_equals_permutation_l4159_415920


namespace NUMINAMATH_CALUDE_seventh_term_is_28_l4159_415981

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  -- First term of the sequence
  a : ℝ
  -- Common difference of the sequence
  d : ℝ
  -- Sum of first three terms is 9
  sum_first_three : a + (a + d) + (a + 2 * d) = 9
  -- Third term is 8
  third_term : a + 2 * d = 8

/-- The seventh term of the arithmetic sequence is 28 -/
theorem seventh_term_is_28 (seq : ArithmeticSequence) : 
  seq.a + 6 * seq.d = 28 := by
sorry

end NUMINAMATH_CALUDE_seventh_term_is_28_l4159_415981


namespace NUMINAMATH_CALUDE_brookes_social_studies_problems_l4159_415991

/-- Calculates the number of social studies problems in Brooke's homework -/
theorem brookes_social_studies_problems :
  ∀ (math_problems science_problems : ℕ)
    (math_time social_studies_time science_time total_time : ℚ),
  math_problems = 15 →
  science_problems = 10 →
  math_time = 2 →
  social_studies_time = 1/2 →
  science_time = 3/2 →
  total_time = 48 →
  ∃ (social_studies_problems : ℕ),
    social_studies_problems = 6 ∧
    (math_problems : ℚ) * math_time +
    (social_studies_problems : ℚ) * social_studies_time +
    (science_problems : ℚ) * science_time = total_time :=
by sorry

end NUMINAMATH_CALUDE_brookes_social_studies_problems_l4159_415991
