import Mathlib

namespace NUMINAMATH_CALUDE_girls_in_choir_l1873_187385

theorem girls_in_choir (orchestra_students band_students choir_students total_students boys_in_choir : ℕ)
  (h1 : orchestra_students = 20)
  (h2 : band_students = 2 * orchestra_students)
  (h3 : boys_in_choir = 12)
  (h4 : total_students = 88)
  (h5 : total_students = orchestra_students + band_students + choir_students) :
  choir_students - boys_in_choir = 16 := by
  sorry

end NUMINAMATH_CALUDE_girls_in_choir_l1873_187385


namespace NUMINAMATH_CALUDE_people_per_column_l1873_187342

theorem people_per_column (total_people : ℕ) (x : ℕ) : 
  total_people = 16 * x ∧ total_people = 12 * 40 → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_people_per_column_l1873_187342


namespace NUMINAMATH_CALUDE_a_periodic_a_minimal_period_l1873_187356

/-- The last digit of a natural number -/
def lastDigit (n : ℕ) : ℕ := n % 10

/-- The sequence a_n defined as the last digit of n^(n^n) -/
def a (n : ℕ+) : ℕ := lastDigit (n.val ^ (n.val ^ n.val))

/-- The theorem stating that the sequence a_n is periodic with period 20 -/
theorem a_periodic : ∃ (p : ℕ+), p = 20 ∧ ∀ (n : ℕ+), a n = a (n + p) :=
sorry

/-- The theorem stating that 20 is the minimal period of the sequence a_n -/
theorem a_minimal_period : 
  ∀ (q : ℕ+), (∀ (n : ℕ+), a n = a (n + q)) → 20 ≤ q :=
sorry

end NUMINAMATH_CALUDE_a_periodic_a_minimal_period_l1873_187356


namespace NUMINAMATH_CALUDE_initial_mean_calculation_l1873_187333

theorem initial_mean_calculation (n : ℕ) (correct_mean wrong_value correct_value : ℝ) :
  n = 30 ∧
  correct_mean = 140.33333333333334 ∧
  wrong_value = 135 ∧
  correct_value = 145 →
  (n * correct_mean - (correct_value - wrong_value)) / n = 140 := by
  sorry

end NUMINAMATH_CALUDE_initial_mean_calculation_l1873_187333


namespace NUMINAMATH_CALUDE_fishbowl_water_volume_l1873_187398

/-- Calculates the volume of water in a cuboid-shaped container. -/
def water_volume (length width water_height : ℝ) : ℝ :=
  length * width * water_height

/-- Proves that the volume of water in the given cuboid-shaped container is 600 cm³. -/
theorem fishbowl_water_volume :
  water_volume 12 10 5 = 600 := by
  sorry

end NUMINAMATH_CALUDE_fishbowl_water_volume_l1873_187398


namespace NUMINAMATH_CALUDE_chinese_remainder_theorem_l1873_187373

theorem chinese_remainder_theorem (y : ℤ) : 
  (y + 4 ≡ 3^2 [ZMOD 3^3]) → 
  (y + 4 ≡ 4^2 [ZMOD 5^3]) → 
  (y + 4 ≡ 6^2 [ZMOD 7^3]) → 
  (y ≡ 32 [ZMOD 105]) := by
  sorry

end NUMINAMATH_CALUDE_chinese_remainder_theorem_l1873_187373


namespace NUMINAMATH_CALUDE_angle_A_is_pi_third_triangle_area_l1873_187365

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the vectors
def m (t : Triangle) : ℝ × ℝ := (t.a + t.b + t.c, 3 * t.c)
def n (t : Triangle) : ℝ × ℝ := (t.b, t.c + t.b - t.a)

-- Define parallel vectors
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

-- Theorem 1
theorem angle_A_is_pi_third (t : Triangle) 
  (h : parallel (m t) (n t)) : t.A = π / 3 := by
  sorry

-- Theorem 2
theorem triangle_area (t : Triangle)
  (h1 : t.a = Real.sqrt 3)
  (h2 : t.b = 1)
  (h3 : t.A = π / 3) :
  (1/2) * t.b * t.c * Real.sin t.A = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_angle_A_is_pi_third_triangle_area_l1873_187365


namespace NUMINAMATH_CALUDE_valid_sequence_only_for_3_and_4_l1873_187327

/-- A sequence of positive integers satisfying the given recurrence relation -/
def ValidSequence (a : ℕ → ℕ) (n : ℕ) : Prop :=
  ∀ k, 2 ≤ k → k < n → a (k+1) = (a k ^ 2 + 1) / (a (k-1) + 1) - 1

/-- The theorem stating that only n = 3 and n = 4 satisfy the condition -/
theorem valid_sequence_only_for_3_and_4 :
  ∀ n : ℕ, n > 0 → (∃ a : ℕ → ℕ, ValidSequence a n) ↔ (n = 3 ∨ n = 4) :=
sorry

end NUMINAMATH_CALUDE_valid_sequence_only_for_3_and_4_l1873_187327


namespace NUMINAMATH_CALUDE_converse_of_quadratic_equation_l1873_187364

theorem converse_of_quadratic_equation (x : ℝ) : x = 1 ∨ x = 2 → x^2 - 3*x + 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_converse_of_quadratic_equation_l1873_187364


namespace NUMINAMATH_CALUDE_difference_of_squares_division_l1873_187361

theorem difference_of_squares_division : (315^2 - 291^2) / 24 = 606 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_division_l1873_187361


namespace NUMINAMATH_CALUDE_cube_split_2017_l1873_187367

/-- The function that gives the first odd number in the split for m^3 -/
def first_split (m : ℕ) : ℕ := 2 * m * (m - 1) + 1

/-- The predicate that checks if a number is in the split for m^3 -/
def in_split (n m : ℕ) : Prop :=
  ∃ k, 0 < k ∧ k ≤ m^3 ∧ n = first_split m + 2 * (k - 1)

theorem cube_split_2017 :
  ∀ m : ℕ, m > 1 → (in_split 2017 m ↔ m = 47) :=
sorry

end NUMINAMATH_CALUDE_cube_split_2017_l1873_187367


namespace NUMINAMATH_CALUDE_divisibility_implication_l1873_187354

theorem divisibility_implication (a : ℤ) : 
  (8 ∣ (5*a + 3) * (3*a + 1)) → (16 ∣ (5*a + 3) * (3*a + 1)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implication_l1873_187354


namespace NUMINAMATH_CALUDE_g_increasing_on_negative_l1873_187389

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions on f
variable (h1 : ∀ x y, x < y → f x < f y)  -- f is increasing
variable (h2 : ∀ x, f x < 0)  -- f is always negative

-- Define the function g
def g (x : ℝ) : ℝ := x^2 * f x

-- State the theorem
theorem g_increasing_on_negative (x y : ℝ) (hx : x < 0) (hy : y < 0) (hxy : x < y) :
  g f x < g f y := by sorry

end NUMINAMATH_CALUDE_g_increasing_on_negative_l1873_187389


namespace NUMINAMATH_CALUDE_larger_number_of_sum_and_product_l1873_187355

theorem larger_number_of_sum_and_product (x y : ℝ) 
  (sum_eq : x + y = 60) 
  (product_eq : x * y = 882) : 
  max x y = 30 + 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_of_sum_and_product_l1873_187355


namespace NUMINAMATH_CALUDE_max_stones_upper_bound_max_stones_achievable_max_stones_theorem_l1873_187357

/-- Represents the state of the piles -/
def PileState := List Nat

/-- The initial state of the piles -/
def initial_state : PileState := List.replicate 2009 2

/-- The operation of transferring stones -/
def transfer (state : PileState) : PileState :=
  sorry

/-- Predicate to check if a state is valid -/
def is_valid_state (state : PileState) : Prop :=
  state.sum = 2009 * 2 ∧ state.all (· ≥ 1)

/-- The maximum number of stones in any pile -/
def max_stones (state : PileState) : Nat :=
  state.foldl Nat.max 0

theorem max_stones_upper_bound :
  ∀ (state : PileState), is_valid_state state → max_stones state ≤ 2010 :=
  sorry

theorem max_stones_achievable :
  ∃ (state : PileState), is_valid_state state ∧ max_stones state = 2010 :=
  sorry

theorem max_stones_theorem :
  (∀ (state : PileState), is_valid_state state → max_stones state ≤ 2010) ∧
  (∃ (state : PileState), is_valid_state state ∧ max_stones state = 2010) :=
  sorry

end NUMINAMATH_CALUDE_max_stones_upper_bound_max_stones_achievable_max_stones_theorem_l1873_187357


namespace NUMINAMATH_CALUDE_max_sum_of_factors_l1873_187388

theorem max_sum_of_factors (A B C : ℕ+) : 
  A ≠ B → B ≠ C → A ≠ C → A * B * C = 2310 → 
  ∀ (X Y Z : ℕ+), X ≠ Y → Y ≠ Z → X ≠ Z → X * Y * Z = 2310 → 
  A + B + C ≤ X + Y + Z → A + B + C ≤ 390 := by
sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_l1873_187388


namespace NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l1873_187339

theorem quadratic_inequality_equivalence (x : ℝ) :
  3 * x^2 + 5 * x < 8 ↔ -4 < x ∧ x < 2/3 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l1873_187339


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1873_187319

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) - a n = d

/-- Given an arithmetic sequence a where a₄ = 1 and a₇ + a₉ = 16, prove that a₁₂ = 15 -/
theorem arithmetic_sequence_problem (a : ℕ → ℚ) 
    (h_arith : ArithmeticSequence a) 
    (h_a4 : a 4 = 1) 
    (h_sum : a 7 + a 9 = 16) : 
  a 12 = 15 := by
sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1873_187319


namespace NUMINAMATH_CALUDE_triangle_third_side_valid_third_side_l1873_187324

/-- Checks if three lengths can form a valid triangle -/
def is_valid_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_third_side (x : ℝ) : 
  (is_valid_triangle 7 10 x ∧ x > 0) ↔ (3 < x ∧ x < 17) :=
sorry

theorem valid_third_side : 
  is_valid_triangle 7 10 11 ∧ 
  ¬(is_valid_triangle 7 10 20) ∧ 
  ¬(is_valid_triangle 7 10 3) ∧ 
  ¬(is_valid_triangle 7 10 2) :=
sorry

end NUMINAMATH_CALUDE_triangle_third_side_valid_third_side_l1873_187324


namespace NUMINAMATH_CALUDE_oscillating_bounded_example_unbounded_oscillations_example_l1873_187320

-- Part a
def oscillating_bounded (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  a < b ∧
  ∀ ε > 0, ∀ X : ℝ, ∃ x₁ x₂ : ℝ, 
    x₁ > X ∧ x₂ > X ∧ 
    f x₁ < a + ε ∧ f x₂ > b - ε

theorem oscillating_bounded_example (a b : ℝ) (h : a < b) :
  oscillating_bounded (fun x ↦ a + (b - a) * Real.sin x ^ 2) a b :=
sorry

-- Part b
def unbounded_oscillations (f : ℝ → ℝ) : Prop :=
  ∀ M : ℝ, ∃ X : ℝ, ∀ x > X, ∃ y > x, 
    (f y > M ∧ f x < -M) ∨ (f y < -M ∧ f x > M)

theorem unbounded_oscillations_example :
  unbounded_oscillations (fun x ↦ x * Real.sin x) :=
sorry

end NUMINAMATH_CALUDE_oscillating_bounded_example_unbounded_oscillations_example_l1873_187320


namespace NUMINAMATH_CALUDE_vector_operations_and_parallel_condition_l1873_187308

def a : Fin 2 → ℝ := ![2, 0]
def b : Fin 2 → ℝ := ![1, 4]

theorem vector_operations_and_parallel_condition :
  (2 • a + 3 • b = ![7, 12]) ∧
  (a - 2 • b = ![0, -8]) ∧
  (∃ (k : ℝ), ∃ (t : ℝ), k • a + b = t • (a + 2 • b) → k = 1/2) := by sorry

end NUMINAMATH_CALUDE_vector_operations_and_parallel_condition_l1873_187308


namespace NUMINAMATH_CALUDE_girls_count_l1873_187353

def total_students : ℕ := 8

def probability_2boys_1girl : ℚ := 15/28

theorem girls_count (x : ℕ) 
  (h1 : x ≤ total_students)
  (h2 : Nat.choose (total_students - x) 2 * Nat.choose x 1 / Nat.choose total_students 3 = probability_2boys_1girl) :
  x = 2 ∨ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_girls_count_l1873_187353


namespace NUMINAMATH_CALUDE_cube_sum_reciprocal_l1873_187395

theorem cube_sum_reciprocal (a : ℝ) (h : (a + 1/a)^2 = 5) : 
  a^3 + 1/a^3 = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_reciprocal_l1873_187395


namespace NUMINAMATH_CALUDE_cheeseburger_cost_is_three_l1873_187360

def restaurant_problem (cheeseburger_cost : ℝ) : Prop :=
  let jim_money : ℝ := 20
  let cousin_money : ℝ := 10
  let total_money : ℝ := jim_money + cousin_money
  let spent_percentage : ℝ := 0.8
  let milkshake_cost : ℝ := 5
  let cheese_fries_cost : ℝ := 8
  let total_spent : ℝ := total_money * spent_percentage
  let num_cheeseburgers : ℕ := 2
  let num_milkshakes : ℕ := 2
  total_spent = num_cheeseburgers * cheeseburger_cost + num_milkshakes * milkshake_cost + cheese_fries_cost

theorem cheeseburger_cost_is_three :
  restaurant_problem 3 := by sorry

end NUMINAMATH_CALUDE_cheeseburger_cost_is_three_l1873_187360


namespace NUMINAMATH_CALUDE_ship_departure_theorem_l1873_187318

/-- Represents the travel times and expected delivery for a cargo shipment --/
structure CargoShipment where
  travelDays : ℕ        -- Days for ship travel
  customsDays : ℕ       -- Days for customs processing
  deliveryDays : ℕ      -- Days from port to warehouse
  expectedArrival : ℕ   -- Days until expected arrival at warehouse

/-- Calculates the number of days ago the ship should have departed --/
def departureDays (shipment : CargoShipment) : ℕ :=
  shipment.travelDays + shipment.customsDays + shipment.deliveryDays - shipment.expectedArrival

/-- Theorem stating that for the given conditions, the ship should have departed 30 days ago --/
theorem ship_departure_theorem (shipment : CargoShipment) 
  (h1 : shipment.travelDays = 21)
  (h2 : shipment.customsDays = 4)
  (h3 : shipment.deliveryDays = 7)
  (h4 : shipment.expectedArrival = 2) :
  departureDays shipment = 30 := by
  sorry

#eval departureDays { travelDays := 21, customsDays := 4, deliveryDays := 7, expectedArrival := 2 }

end NUMINAMATH_CALUDE_ship_departure_theorem_l1873_187318


namespace NUMINAMATH_CALUDE_alternate_color_probability_l1873_187358

/-- The probability of drawing BWBW from a box with 5 white and 6 black balls -/
theorem alternate_color_probability :
  let initial_white : ℕ := 5
  let initial_black : ℕ := 6
  let total_balls : ℕ := initial_white + initial_black
  let prob_first_black : ℚ := initial_black / total_balls
  let prob_second_white : ℚ := initial_white / (total_balls - 1)
  let prob_third_black : ℚ := (initial_black - 1) / (total_balls - 2)
  let prob_fourth_white : ℚ := (initial_white - 1) / (total_balls - 3)
  prob_first_black * prob_second_white * prob_third_black * prob_fourth_white = 2 / 33 :=
by sorry

end NUMINAMATH_CALUDE_alternate_color_probability_l1873_187358


namespace NUMINAMATH_CALUDE_fraction_simplification_l1873_187375

theorem fraction_simplification :
  1 / (1 / (1/3)^1 + 1 / (1/3)^2 + 1 / (1/3)^3) = 1 / 39 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1873_187375


namespace NUMINAMATH_CALUDE_unique_function_property_l1873_187344

theorem unique_function_property (k : ℕ) (f : ℕ → ℕ) 
  (h1 : ∀ n, f n < f (n + 1)) 
  (h2 : ∀ n, f (f n) = n + 2 * k) : 
  ∀ n, f n = n + k := by
  sorry

end NUMINAMATH_CALUDE_unique_function_property_l1873_187344


namespace NUMINAMATH_CALUDE_axis_of_symmetry_l1873_187392

-- Define the quadratic function
def f (x : ℝ) : ℝ := (x - 3) * (x + 1)

-- Theorem statement
theorem axis_of_symmetry :
  ∃ (a b c : ℝ), (∀ x, f x = a * x^2 + b * x + c) ∧ 
  (a ≠ 0) ∧
  (- b / (2 * a) = -1) := by
  sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_l1873_187392


namespace NUMINAMATH_CALUDE_length_of_AB_l1873_187391

/-- Given a line segment AB with points P and Q on it, prove that AB has length 43.2 -/
theorem length_of_AB (A B P Q : ℝ × ℝ) : 
  (P.1 - A.1) / (B.1 - A.1) = 3 / 8 →  -- P divides AB in ratio 3:5
  (Q.1 - A.1) / (B.1 - A.1) = 4 / 9 →  -- Q divides AB in ratio 4:5
  P.1 < Q.1 →  -- P and Q are on the same side of midpoint
  (Q.1 - P.1)^2 + (Q.2 - P.2)^2 = 9 →  -- Distance between P and Q is 3
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = 43.2^2 := by
sorry

end NUMINAMATH_CALUDE_length_of_AB_l1873_187391


namespace NUMINAMATH_CALUDE_fraction_value_l1873_187370

theorem fraction_value (a b c d : ℚ) 
  (h1 : a = 3 * b) 
  (h2 : b = 2 * c) 
  (h3 : c = 4 * d) 
  (h4 : d ≠ 0) : a * c / (b * d) = 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l1873_187370


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1873_187326

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ ∀ n, a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n, a n > 0) →
  a 1 + a 2 = 4/9 →
  a 3 + a 4 + a 5 + a 6 = 40 →
  (a 7 + a 8 + a 9) / 9 = 117 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1873_187326


namespace NUMINAMATH_CALUDE_max_blue_chips_l1873_187383

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem max_blue_chips 
  (total : ℕ) 
  (red : ℕ) 
  (blue : ℕ) 
  (h_total : total = 72)
  (h_sum : red + blue = total)
  (h_prime : ∃ p : ℕ, is_prime p ∧ red = blue + p) :
  blue ≤ 35 ∧ ∃ blue_max : ℕ, blue_max = 35 ∧ 
    ∃ red_max : ℕ, ∃ p_min : ℕ, 
      is_prime p_min ∧ 
      red_max + blue_max = total ∧ 
      red_max = blue_max + p_min :=
sorry

end NUMINAMATH_CALUDE_max_blue_chips_l1873_187383


namespace NUMINAMATH_CALUDE_square_area_with_side_30_l1873_187328

theorem square_area_with_side_30 :
  let side : ℝ := 30
  let square_area := side * side
  square_area = 900 := by sorry

end NUMINAMATH_CALUDE_square_area_with_side_30_l1873_187328


namespace NUMINAMATH_CALUDE_range_of_m_and_n_l1873_187345

-- Define the sets A and B
def A (m : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | 2 * p.1 - p.2 + m > 0}
def B (n : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 + p.2 - n ≤ 0}

-- Define the point P
def P : ℝ × ℝ := (2, 3)

-- State the theorem
theorem range_of_m_and_n (m n : ℝ) : 
  P ∈ A m ∧ P ∉ B n → m > -1 ∧ n < 5 := by
  sorry


end NUMINAMATH_CALUDE_range_of_m_and_n_l1873_187345


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l1873_187397

theorem complex_fraction_equality : (4 - 2*I) / (1 + I)^2 = -1 - 2*I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l1873_187397


namespace NUMINAMATH_CALUDE_total_fish_count_l1873_187310

/-- Given 261 fishbowls, each containing 23 fish, prove that the total number of fish is 6003. -/
theorem total_fish_count (num_fishbowls : ℕ) (fish_per_bowl : ℕ) 
  (h1 : num_fishbowls = 261) 
  (h2 : fish_per_bowl = 23) : 
  num_fishbowls * fish_per_bowl = 6003 := by
  sorry

end NUMINAMATH_CALUDE_total_fish_count_l1873_187310


namespace NUMINAMATH_CALUDE_unique_fixed_point_l1873_187359

-- Define the plane
variable (Plane : Type)

-- Define the set of all lines in the plane
variable (S : Set (Set Plane))

-- Define the function f
variable (f : Set Plane → Plane)

-- Define the notion of a point being on a line
variable (on_line : Plane → Set Plane → Prop)

-- Define the notion of a line passing through a point
variable (passes_through : Set Plane → Plane → Prop)

-- Define the notion of points being on the same circle
variable (on_same_circle : Plane → Plane → Plane → Plane → Prop)

-- Main theorem
theorem unique_fixed_point
  (h1 : ∀ l ∈ S, on_line (f l) l)
  (h2 : ∀ (X : Plane) (l₁ l₂ l₃ : Set Plane),
        l₁ ∈ S → l₂ ∈ S → l₃ ∈ S →
        passes_through l₁ X → passes_through l₂ X → passes_through l₃ X →
        on_same_circle (f l₁) (f l₂) (f l₃) X) :
  ∃! P : Plane, ∀ l ∈ S, passes_through l P → f l = P :=
sorry

end NUMINAMATH_CALUDE_unique_fixed_point_l1873_187359


namespace NUMINAMATH_CALUDE_min_value_quadratic_l1873_187305

theorem min_value_quadratic (k : ℝ) : 
  (∀ x y : ℝ, 3 * x^2 - 4 * k * x * y + (2 * k^2 + 3) * y^2 - 6 * x - 3 * y + 9 ≥ 0) ∧ 
  (∃ x y : ℝ, 3 * x^2 - 4 * k * x * y + (2 * k^2 + 3) * y^2 - 6 * x - 3 * y + 9 = 0) ↔ 
  k = 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l1873_187305


namespace NUMINAMATH_CALUDE_marbles_cost_calculation_l1873_187316

/-- The amount spent on marbles, given the total spent on toys and the cost of a football -/
def marbles_cost (total_spent : ℚ) (football_cost : ℚ) : ℚ :=
  total_spent - football_cost

/-- Theorem stating that the amount spent on marbles is $6.59 -/
theorem marbles_cost_calculation :
  marbles_cost 12.30 5.71 = 6.59 := by
  sorry

end NUMINAMATH_CALUDE_marbles_cost_calculation_l1873_187316


namespace NUMINAMATH_CALUDE_wood_square_weight_relation_second_wood_square_weight_l1873_187329

/-- Represents the properties of a square piece of wood -/
structure WoodSquare where
  side_length : ℝ
  weight : ℝ

/-- Theorem stating the relationship between two square pieces of wood with uniform density and thickness -/
theorem wood_square_weight_relation (w1 w2 : WoodSquare) 
  (h1 : w1.side_length = 3)
  (h2 : w1.weight = 12)
  (h3 : w2.side_length = 6) :
  w2.weight = 48 := by
  sorry

/-- Main theorem proving the weight of the second piece of wood -/
theorem second_wood_square_weight :
  ∃ (w1 w2 : WoodSquare), 
    w1.side_length = 3 ∧ 
    w1.weight = 12 ∧ 
    w2.side_length = 6 ∧ 
    w2.weight = 48 := by
  sorry

end NUMINAMATH_CALUDE_wood_square_weight_relation_second_wood_square_weight_l1873_187329


namespace NUMINAMATH_CALUDE_increasing_power_function_l1873_187330

/-- A function f(x) = (m^2 - 2m - 2)x^(m^2 + m - 1) is increasing on (0, +∞) if and only if
    m^2 - 2m - 2 > 0 and m^2 + m - 1 > 0 -/
theorem increasing_power_function (m : ℝ) :
  let f := fun (x : ℝ) => (m^2 - 2*m - 2) * x^(m^2 + m - 1)
  (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ → f x₁ < f x₂) ↔ 
  (m^2 - 2*m - 2 > 0 ∧ m^2 + m - 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_increasing_power_function_l1873_187330


namespace NUMINAMATH_CALUDE_race_result_l1873_187394

/-- Represents a runner in a race -/
structure Runner where
  speed : ℝ
  time : ℝ

/-- Calculates the distance traveled by a runner given time -/
def distance (r : Runner) (t : ℝ) : ℝ := r.speed * t

theorem race_result (a b : Runner) 
  (h1 : a.time = 240)
  (h2 : b.time = a.time + 10)
  (h3 : distance a a.time = 1000) :
  distance a a.time - distance b a.time = 40 := by
  sorry

end NUMINAMATH_CALUDE_race_result_l1873_187394


namespace NUMINAMATH_CALUDE_min_value_a_squared_plus_b_squared_l1873_187384

theorem min_value_a_squared_plus_b_squared (a b : ℝ) : 
  (∃ x : ℝ, x^4 + a*x^3 + b*x^2 + a*x + 1 = 0) → 
  (∀ a' b' : ℝ, (∃ x : ℝ, x^4 + a'*x^3 + b'*x^2 + a'*x + 1 = 0) → a'^2 + b'^2 ≥ 4) ∧
  (∃ a' b' : ℝ, (∃ x : ℝ, x^4 + a'*x^3 + b'*x^2 + a'*x + 1 = 0) ∧ a'^2 + b'^2 = 4) :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_squared_plus_b_squared_l1873_187384


namespace NUMINAMATH_CALUDE_problem_1_l1873_187336

theorem problem_1 : 96 * 15 / (45 * 16) = 2 := by sorry

end NUMINAMATH_CALUDE_problem_1_l1873_187336


namespace NUMINAMATH_CALUDE_pie_eating_contest_l1873_187331

theorem pie_eating_contest (bill_pies adam_pies sierra_pies : ℕ) : 
  adam_pies = bill_pies + 3 →
  sierra_pies = 2 * bill_pies →
  bill_pies + adam_pies + sierra_pies = 27 →
  sierra_pies = 12 := by
sorry

end NUMINAMATH_CALUDE_pie_eating_contest_l1873_187331


namespace NUMINAMATH_CALUDE_sum_of_digits_next_l1873_187346

def S (n : ℕ) : ℕ := sorry  -- Sum of digits function

theorem sum_of_digits_next (n : ℕ) : S n = 1384 → S (n + 1) = 15 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_next_l1873_187346


namespace NUMINAMATH_CALUDE_race_remaining_distance_l1873_187335

/-- The remaining distance in a race with specific lead changes -/
def remaining_distance (total_length initial_even alex_lead1 max_lead alex_lead2 : ℕ) : ℕ :=
  total_length - (initial_even + alex_lead1 + max_lead + alex_lead2)

/-- Theorem stating the remaining distance in the specific race scenario -/
theorem race_remaining_distance :
  remaining_distance 5000 200 300 170 440 = 3890 := by
  sorry

end NUMINAMATH_CALUDE_race_remaining_distance_l1873_187335


namespace NUMINAMATH_CALUDE_gem_stone_necklaces_count_l1873_187315

/-- Proves that the number of gem stone necklaces sold is 3, given the conditions of the problem -/
theorem gem_stone_necklaces_count :
  let bead_necklaces : ℕ := 4
  let price_per_necklace : ℕ := 3
  let total_earnings : ℕ := 21
  let gem_stone_necklaces : ℕ := (total_earnings - bead_necklaces * price_per_necklace) / price_per_necklace
  gem_stone_necklaces = 3 := by sorry

end NUMINAMATH_CALUDE_gem_stone_necklaces_count_l1873_187315


namespace NUMINAMATH_CALUDE_tracy_book_collection_l1873_187343

theorem tracy_book_collection (first_week : ℕ) (total_books : ℕ) : 
  total_books = 99 → 
  first_week + 5 * (10 * first_week) = total_books →
  first_week = 9 := by
sorry

end NUMINAMATH_CALUDE_tracy_book_collection_l1873_187343


namespace NUMINAMATH_CALUDE_derivative_of_f_l1873_187387

noncomputable def f (x : ℝ) : ℝ := x + 1 / x

theorem derivative_of_f :
  ∀ x : ℝ, x ≠ 0 → deriv f x = 1 - 1 / x^2 := by sorry

end NUMINAMATH_CALUDE_derivative_of_f_l1873_187387


namespace NUMINAMATH_CALUDE_fib_equals_tiling_pred_l1873_187372

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Number of ways to tile a 1 × n rectangle with 1 × 1 squares and 1 × 2 dominos -/
def tiling : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => tiling (n + 1) + tiling n

/-- Theorem: The n-th Fibonacci number equals the number of ways to tile a 1 × (n-1) rectangle -/
theorem fib_equals_tiling_pred (n : ℕ) : fib n = tiling (n - 1) := by
  sorry

end NUMINAMATH_CALUDE_fib_equals_tiling_pred_l1873_187372


namespace NUMINAMATH_CALUDE_female_students_count_l1873_187371

theorem female_students_count (total_average : ℚ) (male_count : ℕ) (male_average : ℚ) (female_average : ℚ) :
  total_average = 90 →
  male_count = 8 →
  male_average = 82 →
  female_average = 92 →
  ∃ (female_count : ℕ),
    (male_count * male_average + female_count * female_average) / (male_count + female_count) = total_average ∧
    female_count = 32 :=
by sorry

end NUMINAMATH_CALUDE_female_students_count_l1873_187371


namespace NUMINAMATH_CALUDE_integer_between_sqrt2_and_sqrt8_l1873_187322

theorem integer_between_sqrt2_and_sqrt8 (a : ℤ) : Real.sqrt 2 < a ∧ a < Real.sqrt 8 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_integer_between_sqrt2_and_sqrt8_l1873_187322


namespace NUMINAMATH_CALUDE_bill_with_tip_divisibility_l1873_187399

theorem bill_with_tip_divisibility (x : ℕ) : ∃ k : ℕ, (11 * x) = (10 * k) := by
  sorry

end NUMINAMATH_CALUDE_bill_with_tip_divisibility_l1873_187399


namespace NUMINAMATH_CALUDE_ratio_sum_equality_l1873_187312

theorem ratio_sum_equality (a b c d : ℚ) 
  (h1 : a / b = 3 / 4) 
  (h2 : c / d = 3 / 4) 
  (h3 : b ≠ 0) 
  (h4 : d ≠ 0) 
  (h5 : b + d ≠ 0) : 
  (a + c) / (b + d) = 3 / 4 := by
sorry

end NUMINAMATH_CALUDE_ratio_sum_equality_l1873_187312


namespace NUMINAMATH_CALUDE_candy_bar_cost_l1873_187303

/-- The cost of the candy bar given the total spent and the cost of cookies -/
theorem candy_bar_cost (total_spent : ℕ) (cookie_cost : ℕ) (h1 : total_spent = 53) (h2 : cookie_cost = 39) :
  total_spent - cookie_cost = 14 := by
  sorry

end NUMINAMATH_CALUDE_candy_bar_cost_l1873_187303


namespace NUMINAMATH_CALUDE_product_of_six_consecutive_divisible_by_ten_l1873_187366

theorem product_of_six_consecutive_divisible_by_ten (n : ℕ+) :
  10 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4) * (n + 5)) := by
  sorry

end NUMINAMATH_CALUDE_product_of_six_consecutive_divisible_by_ten_l1873_187366


namespace NUMINAMATH_CALUDE_bug_position_after_2010_jumps_l1873_187301

/-- Represents the points on the circle -/
inductive Point : Type
| one : Point
| two : Point
| three : Point
| four : Point
| five : Point
| six : Point

/-- Determines if a point is odd or even -/
def isOdd (p : Point) : Bool :=
  match p with
  | Point.one => true
  | Point.three => true
  | Point.five => true
  | _ => false

/-- Performs one jump based on the current position -/
def jump (p : Point) : Point :=
  if isOdd p then
    match p with
    | Point.one => Point.two
    | Point.three => Point.four
    | Point.five => Point.six
    | _ => p  -- This case should never occur
  else
    match p with
    | Point.two => Point.five
    | Point.four => Point.one
    | Point.six => Point.three
    | _ => p  -- This case should never occur

/-- Performs multiple jumps -/
def multipleJumps (start : Point) (n : Nat) : Point :=
  match n with
  | 0 => start
  | n + 1 => jump (multipleJumps start n)

/-- The main theorem to prove -/
theorem bug_position_after_2010_jumps :
  multipleJumps Point.six 2010 = Point.two := by
  sorry


end NUMINAMATH_CALUDE_bug_position_after_2010_jumps_l1873_187301


namespace NUMINAMATH_CALUDE_cos_150_degrees_l1873_187313

theorem cos_150_degrees : Real.cos (150 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_150_degrees_l1873_187313


namespace NUMINAMATH_CALUDE_geometric_series_sum_l1873_187304

/-- Sum of a geometric series with n terms -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The problem statement -/
theorem geometric_series_sum :
  let a : ℚ := 1/6
  let r : ℚ := -1/2
  let n : ℕ := 7
  geometric_sum a r n = 129/1152 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l1873_187304


namespace NUMINAMATH_CALUDE_school_girls_count_l1873_187379

theorem school_girls_count (boys : ℕ) (girls_boys_diff : ℕ) : 
  boys = 469 → girls_boys_diff = 228 → boys + girls_boys_diff = 697 := by
  sorry

end NUMINAMATH_CALUDE_school_girls_count_l1873_187379


namespace NUMINAMATH_CALUDE_smallest_n_for_Q_less_than_1_over_3020_l1873_187350

def Q (n : ℕ+) : ℚ := (Nat.factorial (3*n-1)) / (Nat.factorial (3*n+1))

theorem smallest_n_for_Q_less_than_1_over_3020 :
  ∀ k : ℕ+, k < 19 → Q k ≥ 1/3020 ∧ Q 19 < 1/3020 := by sorry

end NUMINAMATH_CALUDE_smallest_n_for_Q_less_than_1_over_3020_l1873_187350


namespace NUMINAMATH_CALUDE_problem_solution_l1873_187352

theorem problem_solution : ∃ n : ℕ+, 
  (24 ∣ n) ∧ 
  (8.2 < (n : ℝ) ^ (1/3 : ℝ)) ∧ 
  ((n : ℝ) ^ (1/3 : ℝ) < 8.3) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1873_187352


namespace NUMINAMATH_CALUDE_product_of_squares_minus_seven_squares_l1873_187369

theorem product_of_squares_minus_seven_squares 
  (a b c d : ℤ) : (a^2 - 7*b^2) * (c^2 - 7*d^2) = (a*c + 7*b*d)^2 - 7*(a*d + b*c)^2 :=
by sorry

end NUMINAMATH_CALUDE_product_of_squares_minus_seven_squares_l1873_187369


namespace NUMINAMATH_CALUDE_original_number_proof_l1873_187338

theorem original_number_proof (x y : ℝ) : 
  10 * x + 22 * y = 780 → 
  y = 37.66666666666667 → 
  x + y = 32.7 := by
sorry

end NUMINAMATH_CALUDE_original_number_proof_l1873_187338


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1873_187334

theorem sufficient_not_necessary (a b : ℝ) :
  (∀ a b : ℝ, (a - b) * a^2 < 0 → a < b) ∧
  (∃ a b : ℝ, a < b ∧ (a - b) * a^2 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1873_187334


namespace NUMINAMATH_CALUDE_movie_theater_tickets_l1873_187362

theorem movie_theater_tickets (matinee_price evening_price threeD_price : ℕ)
  (evening_sold threeD_sold total_revenue : ℕ) :
  matinee_price = 5 →
  evening_price = 12 →
  threeD_price = 20 →
  evening_sold = 300 →
  threeD_sold = 100 →
  total_revenue = 6600 →
  ∃ (matinee_sold : ℕ), 
    matinee_sold * matinee_price + 
    evening_sold * evening_price + 
    threeD_sold * threeD_price = total_revenue ∧
    matinee_sold = 200 :=
by sorry

end NUMINAMATH_CALUDE_movie_theater_tickets_l1873_187362


namespace NUMINAMATH_CALUDE_descending_order_abc_l1873_187382

theorem descending_order_abc : 3^34 > 2^51 ∧ 2^51 > 4^25 := by
  sorry

end NUMINAMATH_CALUDE_descending_order_abc_l1873_187382


namespace NUMINAMATH_CALUDE_range_of_m_for_positive_functions_l1873_187332

theorem range_of_m_for_positive_functions (m : ℝ) : 
  (∀ x : ℝ, (2 * m * x^2 - 2 * m * x - 8 * x + 9 > 0) ∨ (m * x - m > 0)) →
  (0 < m ∧ m < 8) := by
sorry

end NUMINAMATH_CALUDE_range_of_m_for_positive_functions_l1873_187332


namespace NUMINAMATH_CALUDE_rectangular_prism_equal_surface_volume_l1873_187347

theorem rectangular_prism_equal_surface_volume (a b c : ℕ) :
  (2 * (a * b + b * c + c * a) = a * b * c) ∧ (c = a * b / 2) →
  ((a = 3 ∧ b = 10 ∧ c = 15) ∨ (a = 4 ∧ b = 6 ∧ c = 12)) := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_equal_surface_volume_l1873_187347


namespace NUMINAMATH_CALUDE_base_7_addition_problem_l1873_187337

/-- Convert a base 7 number to base 10 -/
def to_base_10 (a b c : ℕ) : ℕ := a * 7^2 + b * 7 + c

/-- Convert a base 10 number to base 7 -/
def to_base_7 (n : ℕ) : ℕ × ℕ × ℕ :=
  let hundreds := n / 49
  let remainder := n % 49
  let tens := remainder / 7
  let ones := remainder % 7
  (hundreds, tens, ones)

theorem base_7_addition_problem (X Y : ℕ) :
  (to_base_7 (to_base_10 5 X Y + to_base_10 0 5 2) = (6, 4, X)) →
  X + Y = 10 := by
  sorry

end NUMINAMATH_CALUDE_base_7_addition_problem_l1873_187337


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l1873_187314

theorem triangle_abc_properties (a b c A B C : Real) :
  -- Given conditions
  (2 * b * Real.sin B = (2 * a + c) * Real.sin A + (2 * c + a) * Real.sin C) →
  (b = 2 * Real.sqrt 3) →
  (A = π / 4) →
  -- Conclusions
  (B = 2 * π / 3) ∧
  (1/2 * b * c * Real.sin A = (3 - Real.sqrt 3) / 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l1873_187314


namespace NUMINAMATH_CALUDE_discount_calculation_l1873_187381

-- Define the number of pens bought and the equivalent marked price
def pens_bought : ℕ := 50
def marked_price_equivalent : ℕ := 46

-- Define the profit percentage
def profit_percent : ℚ := 7608695652173914 / 100000000000000000

-- Define the discount percentage (to be proven)
def discount_percent : ℚ := 1 / 100

theorem discount_calculation :
  let cost_price := marked_price_equivalent
  let selling_price := cost_price * (1 + profit_percent)
  let discount := pens_bought - selling_price
  discount / pens_bought = discount_percent := by sorry

end NUMINAMATH_CALUDE_discount_calculation_l1873_187381


namespace NUMINAMATH_CALUDE_monotonic_increasing_cubic_linear_l1873_187393

/-- The function f(x) = x^3 - ax is monotonically increasing over ℝ if and only if a ≤ 0 -/
theorem monotonic_increasing_cubic_linear (a : ℝ) :
  (∀ x : ℝ, Monotone (fun x => x^3 - a*x)) ↔ a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_monotonic_increasing_cubic_linear_l1873_187393


namespace NUMINAMATH_CALUDE_unique_divisible_number_l1873_187302

theorem unique_divisible_number : ∃! n : ℕ, 
  45400 ≤ n ∧ n < 45500 ∧ 
  n % 2 = 0 ∧ 
  n % 7 = 0 ∧ 
  n % 9 = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_divisible_number_l1873_187302


namespace NUMINAMATH_CALUDE_problem_solution_l1873_187307

noncomputable section

-- Define the line l
def line_l (t : ℝ) : ℝ × ℝ := (t, Real.sqrt 2 / 2 + Real.sqrt 3 * t)

-- Define the curve C in polar form
def curve_C (θ : ℝ) : ℝ := 2 * Real.cos (θ - Real.pi / 4)

-- Define point P
def point_P : ℝ × ℝ := (0, Real.sqrt 2 / 2)

-- Theorem statement
theorem problem_solution :
  -- 1. The slope angle of line l is π/3
  (let slope := (Real.sqrt 3);
   Real.arctan slope = Real.pi / 3) ∧
  -- 2. The rectangular equation of curve C
  (∀ x y : ℝ, (x - Real.sqrt 2 / 2)^2 + (y - Real.sqrt 2 / 2)^2 = 1 ↔
    ∃ θ : ℝ, x = curve_C θ * Real.cos θ ∧ y = curve_C θ * Real.sin θ) ∧
  -- 3. If line l intersects curve C at points A and B, then |PA| + |PB| = √10/2
  (∃ A B : ℝ × ℝ,
    (∃ t : ℝ, line_l t = A) ∧
    (∃ t : ℝ, line_l t = B) ∧
    (∃ θ : ℝ, A.1 = curve_C θ * Real.cos θ ∧ A.2 = curve_C θ * Real.sin θ) ∧
    (∃ θ : ℝ, B.1 = curve_C θ * Real.cos θ ∧ B.2 = curve_C θ * Real.sin θ) ∧
    Real.sqrt ((A.1 - point_P.1)^2 + (A.2 - point_P.2)^2) +
    Real.sqrt ((B.1 - point_P.1)^2 + (B.2 - point_P.2)^2) =
    Real.sqrt 10 / 2) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l1873_187307


namespace NUMINAMATH_CALUDE_infinite_square_double_numbers_l1873_187306

/-- Definition of a double number -/
def is_double_number (x : ℕ) : Prop :=
  ∃ (d : ℕ), x = d * (10^(Nat.log 10 d + 1) + 1) ∧ d ≠ 0

/-- The main theorem -/
theorem infinite_square_double_numbers :
  ∀ k : ℕ, ∃ N : ℕ,
    let n := 21 * (1 + 14 * k)
    is_double_number (N * (10^n + 1)) ∧
    ∃ m : ℕ, N * (10^n + 1) = m^2 :=
by sorry

end NUMINAMATH_CALUDE_infinite_square_double_numbers_l1873_187306


namespace NUMINAMATH_CALUDE_negation_of_implication_negation_of_positive_square_l1873_187325

theorem negation_of_implication (P Q : Prop) :
  ¬(P → Q) ↔ (P ∧ ¬Q) :=
by sorry

theorem negation_of_positive_square :
  ¬(∀ x : ℝ, x > 0 → x^2 > 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_implication_negation_of_positive_square_l1873_187325


namespace NUMINAMATH_CALUDE_cos_2alpha_plus_cos_2beta_l1873_187300

theorem cos_2alpha_plus_cos_2beta (α β : ℝ) 
  (h1 : Real.sin α + Real.sin β = 1) 
  (h2 : Real.cos α + Real.cos β = 0) : 
  Real.cos (2 * α) + Real.cos (2 * β) = 1 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_plus_cos_2beta_l1873_187300


namespace NUMINAMATH_CALUDE_factorization_1_l1873_187386

theorem factorization_1 (m n : ℤ) : 3 * m * n - 6 * m^2 * n^2 = 3 * m * n * (1 - 2 * m * n) :=
by sorry

end NUMINAMATH_CALUDE_factorization_1_l1873_187386


namespace NUMINAMATH_CALUDE_same_even_on_all_dice_l1873_187351

/-- A standard six-sided die -/
def StandardDie : Type := Fin 6

/-- The probability of rolling an even number on a standard die -/
def probEven : ℚ := 1/2

/-- The probability of rolling a specific number on a standard die -/
def probSpecific : ℚ := 1/6

/-- The number of dice being rolled -/
def numDice : ℕ := 4

/-- Theorem: The probability of all dice showing the same even number -/
theorem same_even_on_all_dice : 
  probEven * probSpecific^(numDice - 1) = 1/432 := by sorry

end NUMINAMATH_CALUDE_same_even_on_all_dice_l1873_187351


namespace NUMINAMATH_CALUDE_function_properties_l1873_187341

open Real

theorem function_properties (f g : ℝ → ℝ) 
  (hf : ∀ x, f x = Real.sin x * Real.cos x)
  (hg : ∀ x, g x = Real.sin x + Real.cos x) :
  (∀ x y, 0 < x ∧ x < y ∧ y < π/4 → f x < f y ∧ g x < g y) ∧
  (∃ x, f x + g x = 1/2 + Real.sqrt 2 ∧
    ∀ y, f y + g y ≤ 1/2 + Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l1873_187341


namespace NUMINAMATH_CALUDE_shopping_cost_l1873_187309

def toilet_paper_quantity : ℕ := 10
def paper_towel_quantity : ℕ := 7
def tissue_quantity : ℕ := 3

def toilet_paper_price : ℚ := 3/2
def paper_towel_price : ℚ := 2
def tissue_price : ℚ := 2

def total_cost : ℚ := 
  toilet_paper_quantity * toilet_paper_price + 
  paper_towel_quantity * paper_towel_price + 
  tissue_quantity * tissue_price

theorem shopping_cost : total_cost = 35 := by
  sorry

end NUMINAMATH_CALUDE_shopping_cost_l1873_187309


namespace NUMINAMATH_CALUDE_largest_n_value_l1873_187349

/-- Represents a digit in base 8 or 9 -/
def Digit := Fin 9

/-- Converts a number from base 8 to base 10 -/
def base8ToBase10 (a b c : Digit) : ℕ :=
  64 * a.val + 8 * b.val + c.val

/-- Converts a number from base 9 to base 10 -/
def base9ToBase10 (c b a : Digit) : ℕ :=
  81 * c.val + 9 * b.val + a.val

/-- Checks if a number is even -/
def isEven (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2 * k

theorem largest_n_value (a b c : Digit) 
    (h1 : base8ToBase10 a b c = base9ToBase10 c b a)
    (h2 : isEven c.val)
    (h3 : ∀ x y z : Digit, 
      base8ToBase10 x y z = base9ToBase10 z y x → 
      isEven z.val → 
      base8ToBase10 x y z ≤ base8ToBase10 a b c) :
  base8ToBase10 a b c = 120 := by
  sorry

end NUMINAMATH_CALUDE_largest_n_value_l1873_187349


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l1873_187363

theorem binomial_coefficient_equality (n : ℕ) : 
  (∃ k : ℕ, k ∈ Finset.range (n - 1) ∧ 
    2 * Nat.choose n k = Nat.choose n (k - 1) + Nat.choose n (k + 1)) ↔ 
  (∃ m : ℕ, m ≥ 3 ∧ n = m^2 - 2) :=
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l1873_187363


namespace NUMINAMATH_CALUDE_sum_digits_greatest_prime_divisor_of_16385_l1873_187321

def n : ℕ := 16385

-- Define a function to get the greatest prime divisor
def greatest_prime_divisor (m : ℕ) : ℕ := 
  sorry

-- Define a function to sum the digits of a number
def sum_of_digits (m : ℕ) : ℕ := 
  sorry

-- Theorem statement
theorem sum_digits_greatest_prime_divisor_of_16385 : 
  sum_of_digits (greatest_prime_divisor n) = 13 := by
  sorry

end NUMINAMATH_CALUDE_sum_digits_greatest_prime_divisor_of_16385_l1873_187321


namespace NUMINAMATH_CALUDE_smallest_whole_number_with_odd_factors_l1873_187368

theorem smallest_whole_number_with_odd_factors : ∃ n : ℕ, 
  n > 100 ∧ 
  (∀ m : ℕ, m > 100 → (∃ k : ℕ, k * k = m) → m ≥ n) ∧
  (∃ k : ℕ, k * k = n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_whole_number_with_odd_factors_l1873_187368


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l1873_187340

theorem inequality_and_equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (Real.sqrt (a^2 - a*b + b^2) + Real.sqrt (b^2 - b*c + c^2) ≥ Real.sqrt (a^2 + a*c + c^2)) ∧
  ((Real.sqrt (a^2 - a*b + b^2) + Real.sqrt (b^2 - b*c + c^2) = Real.sqrt (a^2 + a*c + c^2)) ↔ 
   (1/b = 1/a + 1/c)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l1873_187340


namespace NUMINAMATH_CALUDE_inequality_proofs_l1873_187311

theorem inequality_proofs :
  (∀ x : ℝ, 4 * x - 2 < 1 - 2 * x ↔ x < 1 / 2) ∧
  (∀ x : ℝ, 3 - 2 * x ≥ x - 6 ∧ (3 * x + 1) / 2 < 2 * x ↔ 1 < x ∧ x ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proofs_l1873_187311


namespace NUMINAMATH_CALUDE_decimal_multiplication_addition_l1873_187378

theorem decimal_multiplication_addition : 0.45 * 0.65 + 0.1 * 0.2 = 0.3125 := by
  sorry

end NUMINAMATH_CALUDE_decimal_multiplication_addition_l1873_187378


namespace NUMINAMATH_CALUDE_orange_ratio_l1873_187377

def total_oranges : ℕ := 180
def alice_oranges : ℕ := 120

theorem orange_ratio : 
  let emily_oranges := total_oranges - alice_oranges
  (alice_oranges : ℚ) / emily_oranges = 2 := by
sorry

end NUMINAMATH_CALUDE_orange_ratio_l1873_187377


namespace NUMINAMATH_CALUDE_prob_two_of_three_suits_l1873_187376

/-- The probability of drawing a specific suit from a standard 52-card deck -/
def prob_suit : ℚ := 1/4

/-- The number of cards drawn -/
def num_draws : ℕ := 6

/-- The number of desired cards for each suit (hearts, diamonds, clubs) -/
def num_each_suit : ℕ := 2

/-- The probability of drawing exactly two hearts, two diamonds, and two clubs
    when drawing six cards with replacement from a standard 52-card deck -/
theorem prob_two_of_three_suits : 
  (num_draws.choose num_each_suit * num_draws.choose num_each_suit * num_draws.choose num_each_suit) *
  (prob_suit ^ num_each_suit * prob_suit ^ num_each_suit * prob_suit ^ num_each_suit) = 90/4096 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_of_three_suits_l1873_187376


namespace NUMINAMATH_CALUDE_unique_solution_l1873_187390

def A (p : ℝ) : Set ℝ := {x | x^2 - p*x - 2 = 0}
def B (q r : ℝ) : Set ℝ := {x | x^2 + q*x + r = 0}

theorem unique_solution :
  ∃! (p q r : ℝ),
    (A p ∪ B q r = {-2, 1, 7}) ∧
    (A p ∩ B q r = {-2}) ∧
    p = -1 ∧ q = -5 ∧ r = -14 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l1873_187390


namespace NUMINAMATH_CALUDE_fourth_term_value_l1873_187317

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) = a n * a 2
  first_term : a 1 = 9
  fifth_term : a 5 = a 3 * (a 4)^2

/-- The fourth term of the geometric sequence is ± 1/3 -/
theorem fourth_term_value (seq : GeometricSequence) : 
  seq.a 4 = 1/3 ∨ seq.a 4 = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_fourth_term_value_l1873_187317


namespace NUMINAMATH_CALUDE_binomial_60_3_l1873_187380

theorem binomial_60_3 : Nat.choose 60 3 = 34220 := by
  sorry

end NUMINAMATH_CALUDE_binomial_60_3_l1873_187380


namespace NUMINAMATH_CALUDE_coefficient_x4_in_product_l1873_187348

/-- The coefficient of x^4 in the product of two specific polynomials -/
theorem coefficient_x4_in_product : 
  let p1 : Polynomial ℚ := X^5 - 4*X^4 + 6*X^3 - 7*X^2 + 2*X - 1
  let p2 : Polynomial ℚ := 3*X^4 - 2*X^3 + 5*X - 8
  (p1 * p2).coeff 4 = 27 := by sorry

end NUMINAMATH_CALUDE_coefficient_x4_in_product_l1873_187348


namespace NUMINAMATH_CALUDE_min_moves_to_capture_pawns_l1873_187323

/-- Represents a position on a chessboard -/
structure Position :=
  (row : Fin 8)
  (col : Fin 8)

/-- The knight's move function -/
def knightMove (p : Position) : List Position :=
  let moves := [(1,2), (2,1), (2,-1), (1,-2), (-1,-2), (-2,-1), (-2,1), (-1,2)]
  moves.filterMap (fun (dr, dc) =>
    let newRow := p.row + dr
    let newCol := p.col + dc
    if newRow < 8 && newCol < 8 && newRow ≥ 0 && newCol ≥ 0
    then some ⟨newRow, newCol⟩
    else none)

/-- The minimum number of moves for a knight to capture both pawns -/
def minMovesToCapturePawns : ℕ :=
  let start : Position := ⟨0, 1⟩  -- B1
  let pawn1 : Position := ⟨7, 1⟩  -- B8
  let pawn2 : Position := ⟨7, 6⟩  -- G8
  7  -- The actual minimum number of moves

/-- Theorem stating the minimum number of moves to capture both pawns -/
theorem min_moves_to_capture_pawns :
  minMovesToCapturePawns = 7 :=
sorry

end NUMINAMATH_CALUDE_min_moves_to_capture_pawns_l1873_187323


namespace NUMINAMATH_CALUDE_max_average_profit_l1873_187396

def profit (t : ℕ+) : ℚ := -2 * (t : ℚ)^2 + 30 * (t : ℚ) - 98

def average_profit (t : ℕ+) : ℚ := (profit t) / (t : ℚ)

theorem max_average_profit :
  ∃ (t : ℕ+), ∀ (k : ℕ+), average_profit t ≥ average_profit k ∧ t = 7 :=
sorry

end NUMINAMATH_CALUDE_max_average_profit_l1873_187396


namespace NUMINAMATH_CALUDE_marys_oranges_l1873_187374

theorem marys_oranges (jason_oranges total_oranges : ℕ) 
  (h1 : jason_oranges = 105)
  (h2 : total_oranges = 227)
  : total_oranges - jason_oranges = 122 := by
  sorry

end NUMINAMATH_CALUDE_marys_oranges_l1873_187374
