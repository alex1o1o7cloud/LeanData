import Mathlib

namespace NUMINAMATH_CALUDE_sum_xyz_equals_five_l79_7977

theorem sum_xyz_equals_five (x y z : ℝ) 
  (eq1 : x + 2*y + 3*z = 10) 
  (eq2 : 4*x + 3*y + 2*z = 15) : 
  x + y + z = 5 := by
sorry

end NUMINAMATH_CALUDE_sum_xyz_equals_five_l79_7977


namespace NUMINAMATH_CALUDE_lcm_problem_l79_7907

theorem lcm_problem (m n : ℕ) : 
  m > 0 ∧ 
  100 ≤ n ∧ n < 1000 ∧ 
  Nat.lcm m n = 690 ∧ 
  ¬(3 ∣ n) ∧ 
  ¬(2 ∣ m) → 
  n = 230 := by
sorry

end NUMINAMATH_CALUDE_lcm_problem_l79_7907


namespace NUMINAMATH_CALUDE_initial_average_calculation_l79_7976

/-- Calculates the initial average daily production given the number of days,
    today's production, and the new average. -/
def initial_average (n : ℕ) (today_production : ℕ) (new_average : ℕ) : ℚ :=
  ((n + 1 : ℕ) * new_average - today_production) / n

theorem initial_average_calculation :
  initial_average 12 115 55 = 50 := by sorry

end NUMINAMATH_CALUDE_initial_average_calculation_l79_7976


namespace NUMINAMATH_CALUDE_max_k_value_l79_7968

theorem max_k_value (k : ℝ) : 
  (∃ x y : ℝ, x^2 + k*x + 8 = 0 ∧ y^2 + k*y + 8 = 0 ∧ |x - y| = Real.sqrt 84) →
  k ≤ 2 * Real.sqrt 29 :=
sorry

end NUMINAMATH_CALUDE_max_k_value_l79_7968


namespace NUMINAMATH_CALUDE_function_inequality_l79_7914

open Set

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x < 0, 2 * f x + x * deriv f x > x^2) :
  {x : ℝ | (x + 2017)^2 * f (x + 2017) - 4 * f (-2) > 0} = Iio (-2019) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l79_7914


namespace NUMINAMATH_CALUDE_color_distance_existence_l79_7941

-- Define the color type
inductive Color
| Red
| Blue

-- Define a point in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define the coloring function
def coloring : Point → Color := sorry

-- Define the distance between two points
def distance (p1 p2 : Point) : ℝ := sorry

-- The main theorem
theorem color_distance_existence (x : ℝ) (h : x > 0) :
  ∃ (c : Color), ∃ (p1 p2 : Point), coloring p1 = c ∧ coloring p2 = c ∧ distance p1 p2 = x := by
  sorry

end NUMINAMATH_CALUDE_color_distance_existence_l79_7941


namespace NUMINAMATH_CALUDE_sum_of_distances_is_ten_l79_7982

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 12 * x

-- Define the focus F
def focus : ℝ × ℝ := (3, 0)

-- Define point P
def P : ℝ × ℝ := (2, 1)

-- Define that P is on the line AB
def P_on_line_AB (A B : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, P = (1 - t) • A + t • B

-- Define that P is the midpoint of AB
def P_is_midpoint (A B : ℝ × ℝ) : Prop :=
  P = (A + B) / 2

-- Define that A and B are on the parabola
def points_on_parabola (A B : ℝ × ℝ) : Prop :=
  parabola A.1 A.2 ∧ parabola B.1 B.2

-- State the theorem
theorem sum_of_distances_is_ten (A B : ℝ × ℝ) 
  (h1 : P_on_line_AB A B)
  (h2 : P_is_midpoint A B)
  (h3 : points_on_parabola A B) :
  dist A focus + dist B focus = 10 := by sorry

end NUMINAMATH_CALUDE_sum_of_distances_is_ten_l79_7982


namespace NUMINAMATH_CALUDE_roger_coin_count_l79_7981

/-- The total number of coins in Roger's collection -/
def total_coins (quarters : List Nat) (dimes : List Nat) (nickels : List Nat) (pennies : List Nat) : Nat :=
  quarters.sum + dimes.sum + nickels.sum + pennies.sum

/-- Theorem stating that Roger has 93 coins in total -/
theorem roger_coin_count :
  let quarters := [8, 6, 7, 5]
  let dimes := [7, 5, 9]
  let nickels := [4, 6]
  let pennies := [10, 3, 8, 2, 13]
  total_coins quarters dimes nickels pennies = 93 := by
  sorry

#eval total_coins [8, 6, 7, 5] [7, 5, 9] [4, 6] [10, 3, 8, 2, 13]

end NUMINAMATH_CALUDE_roger_coin_count_l79_7981


namespace NUMINAMATH_CALUDE_fraction_evaluation_l79_7989

theorem fraction_evaluation : (3^6 - 9 * 3^3 + 27) / (3^3 - 3) = 24.75 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l79_7989


namespace NUMINAMATH_CALUDE_simplified_expression_l79_7993

theorem simplified_expression (a : ℤ) 
  (h1 : (a - 1) / 2 < 2)
  (h2 : (a + 1) / 2 ≥ (4 - a) / 3)
  (h3 : a ≠ 2)
  (h4 : a ≠ 4) :
  (16 - a^2) / (a^2 + 8*a + 16) / ((1 / 2) - (4 / (a + 4))) * (1 / (2*a - 4)) = -1 / (a - 2) :=
by sorry

end NUMINAMATH_CALUDE_simplified_expression_l79_7993


namespace NUMINAMATH_CALUDE_books_given_to_sandy_l79_7964

/-- Given that Benny initially had 24 books, Tim has 33 books, and the total number of books
    among Benny, Tim, and Sandy is 47, prove that Benny gave Sandy 10 books. -/
theorem books_given_to_sandy (benny_initial : ℕ) (tim : ℕ) (total : ℕ)
    (h1 : benny_initial = 24)
    (h2 : tim = 33)
    (h3 : total = 47)
    : benny_initial - (total - tim) = 10 := by
  sorry

end NUMINAMATH_CALUDE_books_given_to_sandy_l79_7964


namespace NUMINAMATH_CALUDE_exactly_two_correct_propositions_l79_7996

-- Define the concept of related curves
def related_curves (C1 C2 : ℝ → ℝ → Prop) : Prop :=
  ∃ (l : ℝ → ℝ → Prop), ∃ (x1 y1 x2 y2 : ℝ),
    C1 x1 y1 ∧ C2 x2 y2 ∧
    (∀ x y, l x y ↔ (y - y1) = (x - x1) * ((y2 - y1) / (x2 - x1))) ∧
    (∀ x y, l x y → (C1 x y ∨ C2 x y))

-- Define the curves
def C1_1 (x y : ℝ) : Prop := x^2 + y^2 = 4
def C2_1 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 2*y + 4 = 0
def C1_2 (x y : ℝ) : Prop := 4*y^2 - x^2 = 1
def C2_2 (x y : ℝ) : Prop := x^2 - 4*y^2 = 1
def C1_3 (x y : ℝ) : Prop := y = Real.log x
def C2_3 (x y : ℝ) : Prop := y = x^2 - x

-- Define the propositions
def prop1 : Prop := ∃! (l1 l2 : ℝ → ℝ → Prop), 
  related_curves C1_1 C2_1 ∧ (∀ x y, l1 x y → (C1_1 x y ∨ C2_1 x y)) ∧
  (∀ x y, l2 x y → (C1_1 x y ∨ C2_1 x y)) ∧ l1 ≠ l2

def prop2 : Prop := related_curves C1_2 C2_2

def prop3 : Prop := related_curves C1_3 C2_3

-- The theorem to prove
theorem exactly_two_correct_propositions : 
  (prop1 ∧ ¬prop2 ∧ prop3) ∨ (prop1 ∧ prop2 ∧ ¬prop3) ∨ (¬prop1 ∧ prop2 ∧ prop3) :=
sorry

end NUMINAMATH_CALUDE_exactly_two_correct_propositions_l79_7996


namespace NUMINAMATH_CALUDE_octal_addition_l79_7951

/-- Converts a base-10 integer to its octal representation -/
def to_octal (n : ℕ) : ℕ := sorry

/-- Converts an octal representation to base-10 integer -/
def from_octal (n : ℕ) : ℕ := sorry

theorem octal_addition : to_octal (from_octal 321 + from_octal 127) = 450 := by sorry

end NUMINAMATH_CALUDE_octal_addition_l79_7951


namespace NUMINAMATH_CALUDE_sqrt_three_simplification_l79_7967

theorem sqrt_three_simplification : 3 * Real.sqrt 3 - 2 * Real.sqrt 3 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_simplification_l79_7967


namespace NUMINAMATH_CALUDE_stating_acceleration_implies_speed_increase_l79_7991

/-- Represents a train's acceleration scenario -/
structure TrainAcceleration where
  s : ℝ  -- distance traveled before acceleration (km)
  v : ℝ  -- acceleration rate (km/h)
  x : ℝ  -- initial speed (km/h)

/-- The equation holds for the given train acceleration scenario -/
def equation_holds (t : TrainAcceleration) : Prop :=
  t.s / t.x + t.v = (t.s + 50) / t.x

/-- The train's speed increases by v km/h after acceleration -/
def speed_increase (t : TrainAcceleration) : Prop :=
  ∃ (final_speed : ℝ), final_speed = t.x + t.v

/-- 
Theorem stating that if the equation holds, 
then the train's speed increases by v km/h after acceleration 
-/
theorem acceleration_implies_speed_increase 
  (t : TrainAcceleration) (h : equation_holds t) : speed_increase t :=
sorry

end NUMINAMATH_CALUDE_stating_acceleration_implies_speed_increase_l79_7991


namespace NUMINAMATH_CALUDE_sqrt_product_plus_one_equals_341_l79_7974

theorem sqrt_product_plus_one_equals_341 : 
  Real.sqrt ((20 : ℝ) * 19 * 18 * 17 + 1) = 341 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_plus_one_equals_341_l79_7974


namespace NUMINAMATH_CALUDE_monotonic_decreasing_implies_a_bound_l79_7994

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (a - 1)*x + 2

-- State the theorem
theorem monotonic_decreasing_implies_a_bound (a : ℝ) :
  (∀ x y : ℝ, x < y → y ≤ 4 → f a y < f a x) → a ≤ -7 := by
  sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_implies_a_bound_l79_7994


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l79_7930

/-- 
Given an arithmetic sequence {a_n} with common difference d ≥ 0,
if a_2^2 is the arithmetic mean of a_1^2 and a_3^2 - 2, then d = 1.
-/
theorem arithmetic_sequence_common_difference (a : ℕ → ℝ) (d : ℝ) :
  d ≥ 0 →
  (∀ n, a (n + 1) = a n + d) →
  a 2^2 = (a 1^2 + (a 3^2 - 2)) / 2 →
  d = 1 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l79_7930


namespace NUMINAMATH_CALUDE_speed_ratio_proof_l79_7916

/-- The speed of A in yards per minute -/
def speed_A : ℝ := 333.33

/-- The speed of B in yards per minute -/
def speed_B : ℝ := 433.33

/-- The initial distance of B from point O in yards -/
def initial_distance_B : ℝ := 1000

/-- The time when A and B are first equidistant from O in minutes -/
def time_first_equidistant : ℝ := 3

/-- The time when A and B are second equidistant from O in minutes -/
def time_second_equidistant : ℝ := 10

theorem speed_ratio_proof :
  (∀ t : ℝ, t = time_first_equidistant → 
    (speed_A * t)^2 = (initial_distance_B - speed_B * t)^2) ∧
  (∀ t : ℝ, t = time_second_equidistant → 
    (speed_A * t)^2 = (initial_distance_B - speed_B * t)^2) →
  speed_A / speed_B = 333 / 433 := by
  sorry

end NUMINAMATH_CALUDE_speed_ratio_proof_l79_7916


namespace NUMINAMATH_CALUDE_warehouse_loading_theorem_l79_7918

/-- Represents the warehouse loading problem -/
def warehouseLoading (crateCapacity : ℕ) (numCrates : ℕ) 
                     (nailBags : ℕ) (nailWeight : ℕ) 
                     (hammerBags : ℕ) (hammerWeight : ℕ) 
                     (plankBags : ℕ) (plankWeight : ℕ) : Prop :=
  let totalWeight := nailBags * nailWeight + hammerBags * hammerWeight + plankBags * plankWeight
  let totalCapacity := crateCapacity * numCrates
  totalWeight - totalCapacity = 80

/-- Theorem stating the weight to be left out in the warehouse loading problem -/
theorem warehouse_loading_theorem : 
  warehouseLoading 20 15 4 5 12 5 10 30 := by
  sorry

end NUMINAMATH_CALUDE_warehouse_loading_theorem_l79_7918


namespace NUMINAMATH_CALUDE_factor_probability_l79_7979

/-- The number of factors of m -/
def d (m : ℕ) : ℕ := (Nat.divisors m).card

/-- The probability of selecting a factor of m from 1 to m -/
def prob_factor (m : ℕ) : ℚ := (d m : ℚ) / m

theorem factor_probability (m : ℕ) (p : ℕ) (h : prob_factor m = p / 39) : p = 4 := by
  sorry

end NUMINAMATH_CALUDE_factor_probability_l79_7979


namespace NUMINAMATH_CALUDE_gear_rotations_l79_7924

/-- Represents a gear with a given number of teeth -/
structure Gear where
  teeth : ℕ

/-- Represents a system of two engaged gears -/
structure GearSystem where
  gearA : Gear
  gearB : Gear

/-- Checks if the rotations of two gears are valid (i.e., they mesh properly) -/
def validRotations (gs : GearSystem) (rotA : ℕ) (rotB : ℕ) : Prop :=
  rotA * gs.gearA.teeth = rotB * gs.gearB.teeth

/-- Checks if the given rotations are the smallest possible -/
def smallestRotations (gs : GearSystem) (rotA : ℕ) (rotB : ℕ) : Prop :=
  ∀ (a b : ℕ), validRotations gs a b → (rotA ≤ a ∧ rotB ≤ b)

/-- The main theorem to prove -/
theorem gear_rotations (gs : GearSystem) (h1 : gs.gearA.teeth = 12) (h2 : gs.gearB.teeth = 54) :
  smallestRotations gs 9 2 :=
sorry

end NUMINAMATH_CALUDE_gear_rotations_l79_7924


namespace NUMINAMATH_CALUDE_test_questions_count_l79_7905

theorem test_questions_count (S I C : ℕ) : 
  S = C - 2 * I →
  S = 73 →
  C = 91 →
  C + I = 100 := by
sorry

end NUMINAMATH_CALUDE_test_questions_count_l79_7905


namespace NUMINAMATH_CALUDE_digit_A_is_zero_l79_7902

def is_divisible_by (n m : ℕ) : Prop := n % m = 0

theorem digit_A_is_zero (A : ℕ) (h1 : A < 10) 
  (h2 : is_divisible_by (353808 * 10 + A) 2)
  (h3 : is_divisible_by (353808 * 10 + A) 3)
  (h4 : is_divisible_by (353808 * 10 + A) 5)
  (h5 : is_divisible_by (353808 * 10 + A) 6)
  (h6 : is_divisible_by (353808 * 10 + A) 9) : 
  A = 0 := by
  sorry

end NUMINAMATH_CALUDE_digit_A_is_zero_l79_7902


namespace NUMINAMATH_CALUDE_chloe_carrots_initial_count_l79_7945

/-- Proves that the initial number of carrots Chloe picked is 48, given the conditions of the problem. -/
theorem chloe_carrots_initial_count : ∃ x : ℕ, 
  (x - 45 + 42 = 45) ∧ 
  (x = 48) := by
  sorry

end NUMINAMATH_CALUDE_chloe_carrots_initial_count_l79_7945


namespace NUMINAMATH_CALUDE_complex_equation_solution_l79_7922

theorem complex_equation_solution (i : ℂ) (h_i : i^2 = -1) :
  ∀ z : ℂ, i * z = 1 → z = -i := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l79_7922


namespace NUMINAMATH_CALUDE_line_and_symmetric_point_l79_7906

/-- A line with inclination angle 135° passing through (1, 1) -/
structure Line :=
  (equation : ℝ → ℝ → Prop)
  (passes_through : equation 1 1)
  (inclination : Real.tan (135 * π / 180) = -1)

/-- The symmetric point of A with respect to a line -/
def symmetric_point (l : Line) (A : ℝ × ℝ) : ℝ × ℝ :=
  sorry

theorem line_and_symmetric_point (l : Line) :
  l.equation = fun x y ↦ x + y - 2 = 0 ∧
  symmetric_point l (3, 4) = (-2, -1) :=
sorry

end NUMINAMATH_CALUDE_line_and_symmetric_point_l79_7906


namespace NUMINAMATH_CALUDE_zeros_product_lower_bound_l79_7956

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x - a * x^2 + x

theorem zeros_product_lower_bound {a : ℝ} {x₁ x₂ : ℝ} 
  (h₁ : f a x₁ = 0)
  (h₂ : f a x₂ = 0)
  (h₃ : x₂ > 2 * x₁)
  (h₄ : x₁ > 0)
  (h₅ : x₂ > 0) :
  x₁ * x₂ > 8 / Real.exp 2 := by
  sorry

end NUMINAMATH_CALUDE_zeros_product_lower_bound_l79_7956


namespace NUMINAMATH_CALUDE_corresponding_angles_equality_incomplete_l79_7917

-- Define the concept of corresponding angles
def corresponding_angles (α β : ℝ) : Prop := sorry

-- Define the concept of parallel lines
def parallel_lines (l₁ l₂ : Set (ℝ × ℝ)) : Prop := sorry

-- Theorem stating that the proposition "corresponding angles are equal" is false
-- when not explicitly specifying that the lines are parallel
theorem corresponding_angles_equality_incomplete :
  ¬ ∀ (α β : ℝ) (l₁ l₂ : Set (ℝ × ℝ)), corresponding_angles α β → α = β :=
sorry

end NUMINAMATH_CALUDE_corresponding_angles_equality_incomplete_l79_7917


namespace NUMINAMATH_CALUDE_second_number_proof_l79_7927

theorem second_number_proof (x y z : ℚ) 
  (sum_eq : x + y + z = 150)
  (ratio_xy : x / y = 3 / 4)
  (ratio_yz : y / z = 5 / 7) :
  y = 1000 / 21 := by
  sorry

end NUMINAMATH_CALUDE_second_number_proof_l79_7927


namespace NUMINAMATH_CALUDE_g_of_5_equals_27_l79_7975

-- Define the function g
def g (x : ℝ) : ℝ := 5 * x + 2

-- Theorem statement
theorem g_of_5_equals_27 : g 5 = 27 := by
  sorry

end NUMINAMATH_CALUDE_g_of_5_equals_27_l79_7975


namespace NUMINAMATH_CALUDE_determine_sequence_from_final_state_l79_7954

/-- Represents the state of the cards at each step -/
structure CardState where
  red : ℤ
  blue : ℤ

/-- Applies the transformation to the cards given k -/
def transform (state : CardState) (k : ℕ+) : CardState :=
  { red := k * state.red + state.blue
  , blue := state.red }

/-- Applies n transformations to the initial state using the sequence ks -/
def apply_transformations (initial : CardState) (ks : List ℕ+) : CardState :=
  ks.foldl transform initial

/-- States that it's possible to determine the sequence from the final state -/
theorem determine_sequence_from_final_state 
  (n : ℕ) 
  (ks : List ℕ+) 
  (h_length : ks.length = n) 
  (initial : CardState) 
  (h_initial : initial.red > initial.blue) :
  ∃ (f : CardState → List ℕ+), 
    f (apply_transformations initial ks) = ks :=
sorry

end NUMINAMATH_CALUDE_determine_sequence_from_final_state_l79_7954


namespace NUMINAMATH_CALUDE_jessica_age_l79_7944

theorem jessica_age :
  (∀ (jessica_age claire_age : ℕ),
    jessica_age = claire_age + 6 →
    claire_age + 2 = 20 →
    jessica_age = 24) :=
by sorry

end NUMINAMATH_CALUDE_jessica_age_l79_7944


namespace NUMINAMATH_CALUDE_hockey_league_face_count_l79_7963

/-- The number of times each team faces other teams in a hockey league -/
def faceCount (n : ℕ) (total_games : ℕ) : ℕ :=
  total_games / (n * (n - 1) / 2)

/-- Theorem: In a hockey league with 19 teams and 1710 total games, each team faces others 5 times -/
theorem hockey_league_face_count :
  faceCount 19 1710 = 5 := by
  sorry

end NUMINAMATH_CALUDE_hockey_league_face_count_l79_7963


namespace NUMINAMATH_CALUDE_square_field_area_l79_7908

/-- The area of a square field with side length 6 meters is 36 square meters. -/
theorem square_field_area :
  let side_length : ℝ := 6
  let field_area : ℝ := side_length ^ 2
  field_area = 36 := by sorry

end NUMINAMATH_CALUDE_square_field_area_l79_7908


namespace NUMINAMATH_CALUDE_tree_growth_rate_l79_7933

def initial_height : ℝ := 600
def final_height : ℝ := 720
def growth_period : ℝ := 240

theorem tree_growth_rate :
  (final_height - initial_height) / growth_period = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_tree_growth_rate_l79_7933


namespace NUMINAMATH_CALUDE_christina_age_fraction_l79_7953

/-- Christina's current age -/
def christina_age : ℕ := sorry

/-- Oscar's current age -/
def oscar_age : ℕ := 6

/-- The fraction of Christina's age in 5 years to 80 years -/
def christina_fraction : ℚ := (christina_age + 5) / 80

theorem christina_age_fraction :
  (oscar_age + 15 = 3 * christina_age / 5) →
  christina_fraction = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_christina_age_fraction_l79_7953


namespace NUMINAMATH_CALUDE_coin_distribution_problem_l79_7986

theorem coin_distribution_problem :
  ∃! (x y : ℕ), x > 0 ∧ y > 0 ∧ 9 * x + 17 * y = 70 ∧ x = 4 ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_coin_distribution_problem_l79_7986


namespace NUMINAMATH_CALUDE_add_5_16_base8_l79_7961

/-- Converts a base-8 number to base-10 --/
def base8ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a base-10 number to base-8 --/
def base10ToBase8 (n : ℕ) : ℕ := sorry

/-- Adds two base-8 numbers and returns the result in base-8 --/
def addBase8 (a b : ℕ) : ℕ :=
  base10ToBase8 (base8ToBase10 a + base8ToBase10 b)

theorem add_5_16_base8 :
  addBase8 5 16 = 23 := by sorry

end NUMINAMATH_CALUDE_add_5_16_base8_l79_7961


namespace NUMINAMATH_CALUDE_f_derivative_at_zero_l79_7932

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then Real.arctan (x^2 * Real.sin (1 / (5 * x)))
  else 0

theorem f_derivative_at_zero :
  deriv f 0 = 0 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_zero_l79_7932


namespace NUMINAMATH_CALUDE_average_of_numbers_l79_7920

def numbers : List ℕ := [12, 13, 14, 510, 520, 530, 1115, 1120, 1, 1252140, 2345]

theorem average_of_numbers : (numbers.sum : ℚ) / numbers.length = 114391 := by
  sorry

end NUMINAMATH_CALUDE_average_of_numbers_l79_7920


namespace NUMINAMATH_CALUDE_line_x_axis_intersection_l79_7909

theorem line_x_axis_intersection (x y : ℝ) :
  (5 * y - 7 * x = 14) ∧ (y = 0) → (x = -2 ∧ y = 0) :=
by sorry

end NUMINAMATH_CALUDE_line_x_axis_intersection_l79_7909


namespace NUMINAMATH_CALUDE_part1_part2_l79_7934

-- Part 1
def f (m x : ℝ) : ℝ := x^2 - (m + 2) * x + 3

def has_max_min_in_range (m : ℝ) : Prop :=
  ∃ (M N : ℝ), (∀ x ∈ Set.Icc 1 2, f m x ≤ M ∧ f m x ≥ N) ∧ M - N ≤ 2

theorem part1 (m : ℝ) : has_max_min_in_range m → m ∈ Set.Icc (-1) 3 := by sorry

-- Part 2
def has_solution_in_range (m : ℝ) : Prop :=
  ∃ x ∈ Set.Icc 0 2, x^2 - (m + 2) * x + 3 = -(2 * m + 1) * x + 2

theorem part2 (m : ℝ) : has_solution_in_range m → m ∈ Set.Iic (-1) := by sorry

end NUMINAMATH_CALUDE_part1_part2_l79_7934


namespace NUMINAMATH_CALUDE_carpet_design_problem_l79_7940

/-- Represents the dimensions of a rectangular region in the carpet design. -/
structure RegionDimensions where
  length : ℝ
  width : ℝ

/-- Represents the area of a region in the carpet design. -/
def area (d : RegionDimensions) : ℝ := d.length * d.width

/-- Checks if three real numbers form an arithmetic sequence. -/
def isArithmeticSequence (a b c : ℝ) : Prop := b - a = c - b

/-- The carpet design problem. -/
theorem carpet_design_problem (inner middle outer : RegionDimensions) 
    (h1 : inner.width = 2)
    (h2 : middle.width = inner.width + 4)
    (h3 : middle.length = inner.length + 4)
    (h4 : outer.width = middle.width + 4)
    (h5 : outer.length = middle.length + 4)
    (h6 : isArithmeticSequence (area inner) (area middle) (area outer)) :
    inner.length = 4 := by
  sorry

end NUMINAMATH_CALUDE_carpet_design_problem_l79_7940


namespace NUMINAMATH_CALUDE_find_a_value_l79_7958

-- Define the function f
def f (x : ℝ) : ℝ := sorry

-- State the theorem
theorem find_a_value :
  (∀ x, f (x + 3) = 3 * f x) →
  (∀ x ∈ Set.Ioo 0 3, f x = Real.log x - a * x) →
  (a > 1/3) →
  (Set.Ioo (-6) (-3)).image f ⊆ Set.Iic (-1/9) →
  (∃ x ∈ Set.Ioo (-6) (-3), f x = -1/9) →
  a = 1 := by sorry

end NUMINAMATH_CALUDE_find_a_value_l79_7958


namespace NUMINAMATH_CALUDE_min_fold_length_l79_7965

theorem min_fold_length (width height : ℝ) (hw : width = 8) (hh : height = 11) :
  let min_length := fun y : ℝ => Real.sqrt (width^2 + (y - height)^2)
  ∃ (y : ℝ), y ∈ Set.Icc 0 height ∧
    ∀ (z : ℝ), z ∈ Set.Icc 0 height → min_length y ≤ min_length z ∧
    min_length y = width :=
by sorry

end NUMINAMATH_CALUDE_min_fold_length_l79_7965


namespace NUMINAMATH_CALUDE_tuesday_temperature_l79_7947

def sunday_temp : ℝ := 40
def monday_temp : ℝ := 50
def wednesday_temp : ℝ := 36
def thursday_temp : ℝ := 82
def friday_temp : ℝ := 72
def saturday_temp : ℝ := 26
def average_temp : ℝ := 53
def days_in_week : ℕ := 7

theorem tuesday_temperature :
  ∃ tuesday_temp : ℝ,
    (sunday_temp + monday_temp + tuesday_temp + wednesday_temp +
     thursday_temp + friday_temp + saturday_temp) / days_in_week = average_temp ∧
    tuesday_temp = 65 := by
  sorry

end NUMINAMATH_CALUDE_tuesday_temperature_l79_7947


namespace NUMINAMATH_CALUDE_age_ratio_theorem_l79_7952

/-- Given a person's present age is 14 years, this theorem proves that the ratio of their age 
    16 years hence to their age 4 years ago is 3:1. -/
theorem age_ratio_theorem (present_age : ℕ) (h : present_age = 14) : 
  (present_age + 16) / (present_age - 4) = 3 := by
  sorry

#check age_ratio_theorem

end NUMINAMATH_CALUDE_age_ratio_theorem_l79_7952


namespace NUMINAMATH_CALUDE_multiply_mixed_number_l79_7926

theorem multiply_mixed_number : (7 : ℚ) * (9 + 2/5) = 65 + 4/5 := by
  sorry

end NUMINAMATH_CALUDE_multiply_mixed_number_l79_7926


namespace NUMINAMATH_CALUDE_negative_one_greater_than_negative_two_l79_7915

theorem negative_one_greater_than_negative_two : -1 > -2 := by
  sorry

end NUMINAMATH_CALUDE_negative_one_greater_than_negative_two_l79_7915


namespace NUMINAMATH_CALUDE_triangle_side_range_l79_7955

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the sine law
def sineLaw (t : Triangle) : Prop :=
  t.a / Real.sin t.A = t.b / Real.sin t.B ∧
  t.b / Real.sin t.B = t.c / Real.sin t.C ∧
  t.c / Real.sin t.C = t.a / Real.sin t.A

-- Define the condition for two solutions
def hasTwoSolutions (t : Triangle) : Prop :=
  ∃ (t1 t2 : Triangle), t1 ≠ t2 ∧ sineLaw t1 ∧ sineLaw t2 ∧
    t1.b = t.b ∧ t1.B = t.B ∧ t2.b = t.b ∧ t2.B = t.B

-- State the theorem
theorem triangle_side_range (t : Triangle) 
  (h1 : t.b = 2)
  (h2 : t.B = π/4)  -- 45° in radians
  (h3 : hasTwoSolutions t) :
  2 < t.a ∧ t.a < 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_triangle_side_range_l79_7955


namespace NUMINAMATH_CALUDE_probability_JQKA_standard_deck_l79_7919

/-- Represents a standard deck of 52 playing cards -/
def StandardDeck : ℕ := 52

/-- Represents the number of cards of each rank (Jack, Queen, King, Ace) in a standard deck -/
def CardsPerRank : ℕ := 4

/-- Calculates the probability of drawing a specific sequence of four cards (Jack, Queen, King, Ace) from a standard deck without replacement -/
def probability_JQKA (deck : ℕ) (cards_per_rank : ℕ) : ℚ :=
  (cards_per_rank : ℚ) / deck *
  (cards_per_rank : ℚ) / (deck - 1) *
  (cards_per_rank : ℚ) / (deck - 2) *
  (cards_per_rank : ℚ) / (deck - 3)

/-- Theorem stating that the probability of drawing Jack, Queen, King, Ace in order from a standard deck without replacement is 64/1624350 -/
theorem probability_JQKA_standard_deck :
  probability_JQKA StandardDeck CardsPerRank = 64 / 1624350 := by
  sorry

end NUMINAMATH_CALUDE_probability_JQKA_standard_deck_l79_7919


namespace NUMINAMATH_CALUDE_second_group_size_l79_7966

/-- The number of men in the first group -/
def first_group : ℕ := 20

/-- The number of days taken by the first group -/
def first_days : ℕ := 30

/-- The number of days taken by the second group -/
def second_days : ℕ := 24

/-- The total amount of work in man-days -/
def total_work : ℕ := first_group * first_days

/-- The number of men in the second group -/
def second_group : ℕ := total_work / second_days

theorem second_group_size : second_group = 25 := by
  sorry

end NUMINAMATH_CALUDE_second_group_size_l79_7966


namespace NUMINAMATH_CALUDE_milk_water_ratio_problem_l79_7923

/-- Proves that the initial ratio of milk to water was 4:1 given the conditions of the mixture problem. -/
theorem milk_water_ratio_problem (initial_volume : ℝ) (added_water : ℝ) (final_ratio : ℝ) :
  initial_volume = 45 →
  added_water = 21 →
  final_ratio = 1.2 →
  ∃ (initial_milk initial_water : ℝ),
    initial_milk + initial_water = initial_volume ∧
    initial_milk / (initial_water + added_water) = final_ratio ∧
    initial_milk / initial_water = 4 / 1 :=
by
  sorry

#check milk_water_ratio_problem

end NUMINAMATH_CALUDE_milk_water_ratio_problem_l79_7923


namespace NUMINAMATH_CALUDE_x_plus_y_equals_three_l79_7910

theorem x_plus_y_equals_three (x y : ℝ) 
  (h1 : |x| + x + 5*y = 2) 
  (h2 : |y| - y + x = 7) : 
  x + y = 3 := by
sorry

end NUMINAMATH_CALUDE_x_plus_y_equals_three_l79_7910


namespace NUMINAMATH_CALUDE_village_children_average_l79_7925

/-- Given a village with families and children, calculates the average number of children in families with children -/
def average_children_in_families_with_children (total_families : ℕ) (total_average : ℚ) (childless_families : ℕ) : ℚ :=
  let total_children := total_families * total_average
  let families_with_children := total_families - childless_families
  total_children / families_with_children

/-- Proves that in a village with 12 families, an average of 3 children per family, and 3 childless families, 
    the average number of children in families with children is 4.0 -/
theorem village_children_average : average_children_in_families_with_children 12 3 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_village_children_average_l79_7925


namespace NUMINAMATH_CALUDE_circumcircle_equation_l79_7942

/-- Given the vertices of triangle ABC: A(4,4), B(5,3), and C(1,1),
    prove that the equation of the circumcircle is x^2 + y^2 - 6x - 4y + 8 = 0 -/
theorem circumcircle_equation (A B C : ℝ × ℝ) :
  A = (4, 4) → B = (5, 3) → C = (1, 1) →
  ∃ D E F : ℝ, ∀ x y : ℝ,
    (x^2 + y^2 + D*x + E*y + F = 0 ↔
     ((x - 4)^2 + (y - 4)^2 = 0 ∨
      (x - 5)^2 + (y - 3)^2 = 0 ∨
      (x - 1)^2 + (y - 1)^2 = 0)) →
    D = -6 ∧ E = -4 ∧ F = 8 := by
  sorry


end NUMINAMATH_CALUDE_circumcircle_equation_l79_7942


namespace NUMINAMATH_CALUDE_complex_square_simplification_l79_7901

theorem complex_square_simplification :
  let i : ℂ := Complex.I
  (4 + 3 * i)^2 = 7 + 24 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_square_simplification_l79_7901


namespace NUMINAMATH_CALUDE_triangle_angle_proof_l79_7900

theorem triangle_angle_proof (A B : ℝ) (a b : ℝ) : 
  0 < A ∧ 0 < B ∧ A + B < π →  -- Ensuring A and B are valid triangle angles
  B = 2 * A →                  -- Given condition
  a / b = 1 / Real.sqrt 3 →    -- Given ratio of sides
  A = π / 6                    -- Conclusion (30° in radians)
  := by sorry

end NUMINAMATH_CALUDE_triangle_angle_proof_l79_7900


namespace NUMINAMATH_CALUDE_john_feeds_twice_daily_l79_7962

/-- Represents the scenario of John feeding his horses -/
structure HorseFeeding where
  num_horses : ℕ
  food_per_feeding : ℕ
  days : ℕ
  bags_bought : ℕ
  bag_weight : ℕ

/-- Calculates the number of feedings per horse per day -/
def feedings_per_horse_per_day (hf : HorseFeeding) : ℚ :=
  let total_food := hf.bags_bought * hf.bag_weight
  let food_per_day := total_food / hf.days
  let feedings_per_day := food_per_day / hf.food_per_feeding
  feedings_per_day / hf.num_horses

/-- Theorem stating that John feeds each horse twice a day -/
theorem john_feeds_twice_daily : 
  ∀ (hf : HorseFeeding), 
    hf.num_horses = 25 → 
    hf.food_per_feeding = 20 → 
    hf.days = 60 → 
    hf.bags_bought = 60 → 
    hf.bag_weight = 1000 → 
    feedings_per_horse_per_day hf = 2 := by
  sorry


end NUMINAMATH_CALUDE_john_feeds_twice_daily_l79_7962


namespace NUMINAMATH_CALUDE_larger_root_of_equation_l79_7950

theorem larger_root_of_equation (x : ℚ) : 
  (x - 2/3) * (x - 2/3) + 2 * (x - 2/3) * (x - 4/5) = 0 →
  (x = 2/3 ∨ x = 14/15) ∧ 
  (∀ y, (y - 2/3) * (y - 2/3) + 2 * (y - 2/3) * (y - 4/5) = 0 → y ≤ 14/15) :=
by sorry

#check larger_root_of_equation

end NUMINAMATH_CALUDE_larger_root_of_equation_l79_7950


namespace NUMINAMATH_CALUDE_cottonwood_fiber_diameter_scientific_notation_l79_7984

theorem cottonwood_fiber_diameter_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 0.0000108 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.08 ∧ n = -5 :=
by sorry

end NUMINAMATH_CALUDE_cottonwood_fiber_diameter_scientific_notation_l79_7984


namespace NUMINAMATH_CALUDE_sum_of_S_and_T_is_five_l79_7998

theorem sum_of_S_and_T_is_five : 
  ∀ (S T : ℝ),
  let line_length : ℝ := 5
  let num_parts : ℕ := 20
  let part_length : ℝ := line_length / num_parts
  S = 5 * part_length →
  T = line_length - 5 * part_length →
  S + T = 5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_S_and_T_is_five_l79_7998


namespace NUMINAMATH_CALUDE_double_sum_equals_seven_fourths_l79_7959

/-- The double sum of 1/(i^2j + 2ij + ij^2) over positive integers i and j from 1 to infinity equals 7/4 -/
theorem double_sum_equals_seven_fourths :
  (∑' i : ℕ+, ∑' j : ℕ+, (1 : ℝ) / ((i.val^2 * j.val) + (2 * i.val * j.val) + (i.val * j.val^2))) = 7/4 :=
by sorry

end NUMINAMATH_CALUDE_double_sum_equals_seven_fourths_l79_7959


namespace NUMINAMATH_CALUDE_gcd_7384_12873_l79_7912

theorem gcd_7384_12873 : Nat.gcd 7384 12873 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_7384_12873_l79_7912


namespace NUMINAMATH_CALUDE_unique_positive_root_interval_l79_7992

theorem unique_positive_root_interval :
  ∃! r : ℝ, r > 0 ∧ r^3 - r - 1 = 0 →
  ∃ r : ℝ, r ∈ Set.Ioo 1 2 ∧ r^3 - r - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_root_interval_l79_7992


namespace NUMINAMATH_CALUDE_cricketer_average_score_l79_7978

theorem cricketer_average_score (total_matches : ℕ) (overall_average : ℚ) 
  (last_matches : ℕ) (last_average : ℚ) (some_average : ℚ) :
  total_matches = 10 →
  overall_average = 389/10 →
  last_matches = 4 →
  last_average = 137/4 →
  some_average = 42 →
  ∃ (x : ℕ), x + last_matches = total_matches ∧ 
    (x : ℚ) * some_average + (last_matches : ℚ) * last_average = (total_matches : ℚ) * overall_average ∧
    x = 6 :=
by sorry

end NUMINAMATH_CALUDE_cricketer_average_score_l79_7978


namespace NUMINAMATH_CALUDE_feline_sanctuary_tigers_l79_7971

theorem feline_sanctuary_tigers (lions cougars tigers : ℕ) : 
  lions = 12 →
  cougars = (lions + tigers) / 2 →
  lions + tigers + cougars = 39 →
  tigers = 14 := by
sorry

end NUMINAMATH_CALUDE_feline_sanctuary_tigers_l79_7971


namespace NUMINAMATH_CALUDE_jakes_weight_l79_7990

/-- Proves Jake's present weight given the conditions of the problem -/
theorem jakes_weight (jake kendra : ℕ) 
  (h1 : jake - 8 = 2 * kendra)
  (h2 : jake + kendra = 290) : 
  jake = 196 := by
  sorry

end NUMINAMATH_CALUDE_jakes_weight_l79_7990


namespace NUMINAMATH_CALUDE_professors_arrangement_count_l79_7904

/-- The number of ways to arrange professors among students. -/
def arrange_professors (num_students : ℕ) (num_professors : ℕ) : ℕ :=
  Nat.descFactorial (num_students - 1) num_professors

/-- Theorem stating that arranging 3 professors among 6 students results in 60 possibilities. -/
theorem professors_arrangement_count :
  arrange_professors 6 3 = 60 := by
  sorry

end NUMINAMATH_CALUDE_professors_arrangement_count_l79_7904


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_right_triangle_l79_7957

theorem inscribed_circle_radius_right_triangle 
  (a b c : ℝ) 
  (h_right : a^2 + b^2 = c^2) 
  (h_leg : a = 15) 
  (h_proj : c - b = 16) : 
  (a + b - c) / 2 = 5 := by
sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_right_triangle_l79_7957


namespace NUMINAMATH_CALUDE_hiring_probability_l79_7929

/-- The number of candidates -/
def numCandidates : ℕ := 4

/-- The number of people to be hired -/
def numHired : ℕ := 2

/-- The probability of hiring at least one of two specific candidates -/
def probAtLeastOne : ℚ := 5/6

theorem hiring_probability :
  (numCandidates : ℚ) > 0 ∧ numHired ≤ numCandidates →
  (1 : ℚ) - (Nat.choose (numCandidates - 2) numHired : ℚ) / (Nat.choose numCandidates numHired : ℚ) = probAtLeastOne :=
sorry

end NUMINAMATH_CALUDE_hiring_probability_l79_7929


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_area_l79_7937

/-- The area of a right-angled isosceles triangle with hypotenuse length 1 is 1/4 -/
theorem isosceles_right_triangle_area (A B C : ℝ × ℝ) : 
  (A.1 = 0 ∧ A.2 = 0) →  -- A is at origin
  (B.1 = 1 ∧ B.2 = 0) →  -- B is at (1, 0)
  (C.1 = 0 ∧ C.2 = 1) →  -- C is at (0, 1)
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = (C.1 - A.1)^2 + (C.2 - A.2)^2 →  -- AB = AC (isosceles)
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0 →  -- right angle at A
  (1/2) * (B.1 - A.1) * (C.2 - A.2) = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_area_l79_7937


namespace NUMINAMATH_CALUDE_principal_calculation_l79_7983

/-- Compound interest calculation -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Problem statement -/
theorem principal_calculation (sum : ℝ) (rate : ℝ) (time : ℕ) 
  (h_sum : sum = 3969)
  (h_rate : rate = 0.05)
  (h_time : time = 2) :
  ∃ (principal : ℝ), 
    compound_interest principal rate time = sum ∧ 
    principal = 3600 := by
  sorry

end NUMINAMATH_CALUDE_principal_calculation_l79_7983


namespace NUMINAMATH_CALUDE_tan_identities_l79_7911

theorem tan_identities (α : Real) (h : Real.tan α = 2) :
  (Real.tan (α + π/4) = -3) ∧
  ((Real.sin α + Real.cos α) / (2 * Real.sin α - Real.cos α) = 1) := by
  sorry

end NUMINAMATH_CALUDE_tan_identities_l79_7911


namespace NUMINAMATH_CALUDE_no_solution_in_A_l79_7943

-- Define the set A
def A : Set ℕ :=
  {n : ℕ | ∃ k : ℤ, |n * Real.sqrt 2022 - 1/3 - ↑k| ≤ 1/2022}

-- State the theorem
theorem no_solution_in_A :
  ∀ x y z : ℕ, x ∈ A → y ∈ A → z ∈ A → 20 * x + 21 * y ≠ 22 * z :=
by sorry

end NUMINAMATH_CALUDE_no_solution_in_A_l79_7943


namespace NUMINAMATH_CALUDE_common_number_in_overlapping_sets_l79_7995

theorem common_number_in_overlapping_sets (numbers : List ℝ) : 
  numbers.length = 9 →
  (numbers.take 5).sum / 5 = 7 →
  (numbers.drop 4).sum / 5 = 10 →
  numbers.sum / 9 = 74 / 9 →
  ∃ x ∈ numbers.take 5 ∩ numbers.drop 4, x = 11 := by
  sorry

end NUMINAMATH_CALUDE_common_number_in_overlapping_sets_l79_7995


namespace NUMINAMATH_CALUDE_smallest_with_properties_l79_7946

def is_smallest_with_properties (n : ℕ) : Prop :=
  (∃ (divisors : Finset ℕ), divisors.card = 144 ∧ (∀ d ∈ divisors, n % d = 0)) ∧
  (∃ (start : ℕ), ∀ i ∈ Finset.range 10, n % (start + i) = 0) ∧
  (∀ m < n, ¬(∃ (divisors : Finset ℕ), divisors.card = 144 ∧ (∀ d ∈ divisors, m % d = 0)) ∨
           ¬(∃ (start : ℕ), ∀ i ∈ Finset.range 10, m % (start + i) = 0))

theorem smallest_with_properties : is_smallest_with_properties 110880 := by
  sorry

end NUMINAMATH_CALUDE_smallest_with_properties_l79_7946


namespace NUMINAMATH_CALUDE_complex_arithmetic_evaluation_l79_7970

theorem complex_arithmetic_evaluation :
  1234562 - ((12 * 3 * (2 + 7))^2 / 6) + 18 = 1217084 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_evaluation_l79_7970


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l79_7972

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_fraction_simplification :
  ((2 + i) * (3 - 4*i)) / (2 - i) = 5 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l79_7972


namespace NUMINAMATH_CALUDE_carla_candy_bags_l79_7935

/-- Calculates the number of bags bought given the original price, discount percentage, and total amount spent -/
def bags_bought (original_price : ℚ) (discount_percentage : ℚ) (total_spent : ℚ) : ℚ :=
  total_spent / (original_price * (1 - discount_percentage))

/-- Proves that Carla bought 2 bags of candy -/
theorem carla_candy_bags : 
  let original_price : ℚ := 6
  let discount_percentage : ℚ := 3/4
  let total_spent : ℚ := 3
  bags_bought original_price discount_percentage total_spent = 2 := by
sorry

#eval bags_bought 6 (3/4) 3

end NUMINAMATH_CALUDE_carla_candy_bags_l79_7935


namespace NUMINAMATH_CALUDE_sin_15_mul_sin_75_eq_quarter_l79_7969

theorem sin_15_mul_sin_75_eq_quarter : Real.sin (15 * π / 180) * Real.sin (75 * π / 180) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_15_mul_sin_75_eq_quarter_l79_7969


namespace NUMINAMATH_CALUDE_restaurant_bill_proof_l79_7949

theorem restaurant_bill_proof (total_friends : Nat) (paying_friends : Nat) (extra_payment : ℝ) :
  total_friends = 10 →
  paying_friends = 9 →
  extra_payment = 3 →
  ∃ (bill : ℝ), (paying_friends : ℝ) * ((bill / total_friends) + extra_payment) = bill ∧ bill = 270 :=
by
  sorry

end NUMINAMATH_CALUDE_restaurant_bill_proof_l79_7949


namespace NUMINAMATH_CALUDE_final_s_value_l79_7987

/-- Represents the state of the variables in the loop -/
structure LoopState where
  i : ℕ
  s : ℤ

/-- Defines a single iteration of the loop -/
def loopIteration (state : LoopState) : LoopState :=
  { i := state.i + 1,
    s := 2 * state.s - 1 }

/-- Applies the loop iteration n times -/
def applyNTimes (n : ℕ) (state : LoopState) : LoopState :=
  match n with
  | 0 => state
  | n + 1 => loopIteration (applyNTimes n state)

/-- The main theorem to prove -/
theorem final_s_value :
  let initialState : LoopState := { i := 1, s := 0 }
  let finalState := applyNTimes 5 initialState
  finalState.s = -31 := by sorry

end NUMINAMATH_CALUDE_final_s_value_l79_7987


namespace NUMINAMATH_CALUDE_kids_difference_l79_7938

theorem kids_difference (monday : ℕ) (tuesday : ℕ) 
  (h1 : monday = 6) (h2 : tuesday = 5) : monday - tuesday = 1 := by
  sorry

end NUMINAMATH_CALUDE_kids_difference_l79_7938


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l79_7931

theorem sum_of_fractions_equals_one 
  {x y z : ℝ} (h : x * y * z = 1) : 
  1 / (1 + x + x * y) + 1 / (1 + y + y * z) + 1 / (1 + z + z * x) = 1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l79_7931


namespace NUMINAMATH_CALUDE_shelter_cat_count_l79_7999

/-- Represents the state of the animal shelter --/
structure AnimalShelter where
  initialCats : ℕ
  newCats : ℕ
  adoptedCats : ℕ
  bornKittens : ℕ
  claimedPets : ℕ

/-- Calculates the final number of cats in the shelter --/
def finalCatCount (shelter : AnimalShelter) : ℕ :=
  shelter.initialCats + shelter.newCats - shelter.adoptedCats + shelter.bornKittens - shelter.claimedPets

/-- Theorem stating the final number of cats in the shelter --/
theorem shelter_cat_count : ∃ (shelter : AnimalShelter),
  shelter.initialCats = 60 ∧
  shelter.newCats = 30 ∧
  shelter.adoptedCats = 20 ∧
  shelter.bornKittens = 15 ∧
  shelter.claimedPets = 2 ∧
  finalCatCount shelter = 83 := by
  sorry

#check shelter_cat_count

end NUMINAMATH_CALUDE_shelter_cat_count_l79_7999


namespace NUMINAMATH_CALUDE_rectangle_area_theorem_l79_7903

/-- Rectangle with known side length and area -/
structure Rectangle1 where
  side : ℝ
  area : ℝ

/-- Rectangle similar to Rectangle1 with known diagonal -/
structure Rectangle2 where
  diagonal : ℝ

/-- The area of Rectangle2 given the properties of Rectangle1 -/
def area_rectangle2 (r1 : Rectangle1) (r2 : Rectangle2) : ℝ :=
  sorry

theorem rectangle_area_theorem (r1 : Rectangle1) (r2 : Rectangle2) :
  r1.side = 3 ∧ r1.area = 18 ∧ r2.diagonal = 20 → area_rectangle2 r1 r2 = 160 :=
sorry

end NUMINAMATH_CALUDE_rectangle_area_theorem_l79_7903


namespace NUMINAMATH_CALUDE_part_one_part_two_l79_7921

-- Define the function f
def f (x a : ℝ) : ℝ := |x + a| + |x - a^2|

-- Define the solution set for part (1)
def solution_set : Set ℝ := {x | x ≤ -2 ∨ x ≥ 2}

-- Theorem for part (1)
theorem part_one : 
  ∀ x, f x 1 ≥ 4 ↔ x ∈ solution_set :=
sorry

-- Theorem for part (2)
theorem part_two :
  (∀ x, ∃ a ∈ Set.Ioo (-1) 3, m < f x a) → m < 12 :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l79_7921


namespace NUMINAMATH_CALUDE_percentage_of_b_l79_7973

theorem percentage_of_b (a b c : ℝ) (h1 : 8 = 0.02 * a) (h2 : c = b / a) : 
  ∃ p : ℝ, p * b = 2 ∧ p = 0.005 := by sorry

end NUMINAMATH_CALUDE_percentage_of_b_l79_7973


namespace NUMINAMATH_CALUDE_positive_integer_problem_l79_7985

theorem positive_integer_problem (n : ℕ+) (h : (12 : ℝ) * n.val = n.val ^ 2 + 36) : n = 6 := by
  sorry

end NUMINAMATH_CALUDE_positive_integer_problem_l79_7985


namespace NUMINAMATH_CALUDE_equation_represents_two_lines_l79_7980

/-- The equation x^2 - 72y^2 - 16x + 64 = 0 represents two lines in the xy-plane. -/
theorem equation_represents_two_lines :
  ∃ (a b c d : ℝ), ∀ (x y : ℝ),
    (x^2 - 72*y^2 - 16*x + 64 = 0) ↔ ((x = a*y + b) ∨ (x = c*y + d)) :=
by sorry

end NUMINAMATH_CALUDE_equation_represents_two_lines_l79_7980


namespace NUMINAMATH_CALUDE_grid_property_l79_7997

-- Define a 4x4 grid of rational numbers
def Grid := Matrix (Fin 4) (Fin 4) ℚ

-- Define what it means for a row to be an arithmetic sequence
def is_arithmetic_row (g : Grid) (i : Fin 4) : Prop :=
  ∃ a d : ℚ, ∀ j : Fin 4, g i j = a + d * j

-- Define what it means for a column to be an arithmetic sequence
def is_arithmetic_col (g : Grid) (j : Fin 4) : Prop :=
  ∃ a d : ℚ, ∀ i : Fin 4, g i j = a + d * i

-- Main theorem
theorem grid_property (g : Grid) : 
  (∀ i : Fin 4, is_arithmetic_row g i) →
  (∀ j : Fin 4, is_arithmetic_col g j) →
  g 0 0 = 3 →
  g 0 3 = 18 →
  g 3 0 = 11 →
  g 3 3 = 50 →
  g 1 2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_grid_property_l79_7997


namespace NUMINAMATH_CALUDE_num_parallel_planes_zero_or_one_l79_7913

/-- Two lines in 3D space are skew if they are not parallel and do not intersect. -/
def are_skew (a b : Line3D) : Prop := sorry

/-- A point is outside a line if it does not lie on the line. -/
def is_outside (P : Point3D) (l : Line3D) : Prop := sorry

/-- A plane is parallel to a line if they do not intersect. -/
def plane_parallel_to_line (π : Plane3D) (l : Line3D) : Prop := sorry

/-- The number of planes passing through a point and parallel to two lines. -/
def num_parallel_planes (P : Point3D) (a b : Line3D) : ℕ := sorry

theorem num_parallel_planes_zero_or_one 
  (P : Point3D) (a b : Line3D) 
  (h_skew : are_skew a b) 
  (h_outside_a : is_outside P a) 
  (h_outside_b : is_outside P b) : 
  num_parallel_planes P a b = 0 ∨ num_parallel_planes P a b = 1 := by
  sorry

end NUMINAMATH_CALUDE_num_parallel_planes_zero_or_one_l79_7913


namespace NUMINAMATH_CALUDE_combined_yellow_ratio_approx_32_percent_l79_7948

/-- Represents a bag of jelly beans -/
structure JellyBeanBag where
  total : ℕ
  yellow_ratio : ℚ

/-- Calculates the ratio of yellow jelly beans to all beans when multiple bags are combined -/
def combined_yellow_ratio (bags : List JellyBeanBag) : ℚ :=
  let total_beans := bags.map (λ bag => bag.total) |>.sum
  let total_yellow := bags.map (λ bag => (bag.total : ℚ) * bag.yellow_ratio) |>.sum
  total_yellow / total_beans

/-- The theorem to be proved -/
theorem combined_yellow_ratio_approx_32_percent : 
  let bags := [
    JellyBeanBag.mk 24 (2/5),
    JellyBeanBag.mk 32 (3/10),
    JellyBeanBag.mk 34 (1/4)
  ]
  abs (combined_yellow_ratio bags - 32222/100000) < 1/10000 := by
  sorry

end NUMINAMATH_CALUDE_combined_yellow_ratio_approx_32_percent_l79_7948


namespace NUMINAMATH_CALUDE_constant_function_from_functional_equation_l79_7928

/-- A continuous function f satisfying f(x) + f(x^2) = 2 for all real x is constant and equal to 1. -/
theorem constant_function_from_functional_equation (f : ℝ → ℝ) 
  (hf : Continuous f) 
  (h : ∀ x : ℝ, f x + f (x^2) = 2) : 
  ∀ x : ℝ, f x = 1 := by
  sorry

end NUMINAMATH_CALUDE_constant_function_from_functional_equation_l79_7928


namespace NUMINAMATH_CALUDE_function_is_identity_l79_7988

-- Define the property that the function f must satisfy
def satisfies_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (2 * x + f y) = x + y + f x

-- Theorem statement
theorem function_is_identity 
  (f : ℝ → ℝ) 
  (h : satisfies_equation f) : 
  ∀ x : ℝ, f x = x := by
  sorry

end NUMINAMATH_CALUDE_function_is_identity_l79_7988


namespace NUMINAMATH_CALUDE_line_through_point_l79_7939

/-- Given a line equation 2 - 3kx = -4y that passes through the point (3, -2),
    prove that k = -2/3 is the unique value that satisfies the equation. -/
theorem line_through_point (k : ℚ) : 
  (2 - 3 * k * 3 = -4 * (-2)) ↔ k = -2/3 := by sorry

end NUMINAMATH_CALUDE_line_through_point_l79_7939


namespace NUMINAMATH_CALUDE_initial_goldfish_count_l79_7960

/-- The number of goldfish Paige initially raised in the pond -/
def initial_goldfish : ℕ := 15

/-- The number of goldfish remaining in the pond -/
def remaining_goldfish : ℕ := 4

/-- The number of goldfish that disappeared -/
def disappeared_goldfish : ℕ := 11

/-- Theorem: The initial number of goldfish is equal to the sum of the remaining and disappeared goldfish -/
theorem initial_goldfish_count : initial_goldfish = remaining_goldfish + disappeared_goldfish := by
  sorry

end NUMINAMATH_CALUDE_initial_goldfish_count_l79_7960


namespace NUMINAMATH_CALUDE_largest_n_polynomials_l79_7936

/-- A type representing real polynomials -/
def RealPolynomial := ℝ → ℝ

/-- Predicate to check if a real polynomial has no real roots -/
def HasNoRealRoots (p : RealPolynomial) : Prop :=
  ∀ x : ℝ, p x ≠ 0

/-- Predicate to check if a real polynomial has at least one real root -/
def HasRealRoot (p : RealPolynomial) : Prop :=
  ∃ x : ℝ, p x = 0

/-- The main theorem statement -/
theorem largest_n_polynomials :
  (∃ (n : ℕ) (P : Fin n → RealPolynomial),
    (∀ (i j : Fin n) (hij : i ≠ j), HasNoRealRoots (fun x ↦ P i x + P j x)) ∧
    (∀ (i j k : Fin n) (hijk : i ≠ j ∧ j ≠ k ∧ i ≠ k), HasRealRoot (fun x ↦ P i x + P j x + P k x))) →
  (∃ (P : Fin 3 → RealPolynomial),
    (∀ (i j : Fin 3) (hij : i ≠ j), HasNoRealRoots (fun x ↦ P i x + P j x)) ∧
    (∀ (i j k : Fin 3) (hijk : i ≠ j ∧ j ≠ k ∧ i ≠ k), HasRealRoot (fun x ↦ P i x + P j x + P k x))) ∧
  (∀ (n : ℕ) (hn : n > 3),
    ¬∃ (P : Fin n → RealPolynomial),
      (∀ (i j : Fin n) (hij : i ≠ j), HasNoRealRoots (fun x ↦ P i x + P j x)) ∧
      (∀ (i j k : Fin n) (hijk : i ≠ j ∧ j ≠ k ∧ i ≠ k), HasRealRoot (fun x ↦ P i x + P j x + P k x))) :=
by
  sorry

end NUMINAMATH_CALUDE_largest_n_polynomials_l79_7936
