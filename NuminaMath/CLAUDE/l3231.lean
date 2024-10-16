import Mathlib

namespace NUMINAMATH_CALUDE_calculate_product_l3231_323145

theorem calculate_product : 500 * 1986 * 0.3972 * 100 = 20 * 1986^2 := by
  sorry

end NUMINAMATH_CALUDE_calculate_product_l3231_323145


namespace NUMINAMATH_CALUDE_impossible_to_flip_all_l3231_323116

/-- Represents the state of a coin: true if facing up, false if facing down -/
def Coin := Bool

/-- Represents the state of 5 coins -/
def CoinState := Fin 5 → Coin

/-- An operation that flips exactly 4 coins -/
def FlipFour (state : CoinState) : CoinState :=
  sorry

/-- The initial state where all coins are facing up -/
def initialState : CoinState := fun _ => true

/-- The target state where all coins are facing down -/
def targetState : CoinState := fun _ => false

/-- Predicate to check if a state can be reached from the initial state -/
def Reachable (state : CoinState) : Prop :=
  sorry

theorem impossible_to_flip_all :
  ¬ Reachable targetState :=
sorry

end NUMINAMATH_CALUDE_impossible_to_flip_all_l3231_323116


namespace NUMINAMATH_CALUDE_smallest_z_for_inequality_l3231_323129

theorem smallest_z_for_inequality : ∃ (z : ℕ), (∀ (y : ℕ), 27 ^ y > 3 ^ 24 → z ≤ y) ∧ 27 ^ z > 3 ^ 24 := by
  sorry

end NUMINAMATH_CALUDE_smallest_z_for_inequality_l3231_323129


namespace NUMINAMATH_CALUDE_max_different_sums_l3231_323105

def penny : ℚ := 1 / 100
def nickel : ℚ := 5 / 100
def dime : ℚ := 10 / 100
def half_dollar : ℚ := 50 / 100

def coin_set : Finset ℚ := {penny, nickel, nickel, dime, dime, half_dollar}

def sum_pairs (s : Finset ℚ) : Finset ℚ :=
  (s.product s).image (λ (x, y) => x + y)

theorem max_different_sums :
  (sum_pairs coin_set).card = 8 := by sorry

end NUMINAMATH_CALUDE_max_different_sums_l3231_323105


namespace NUMINAMATH_CALUDE_disneyland_attractions_permutations_l3231_323128

theorem disneyland_attractions_permutations :
  Nat.factorial 6 = 720 := by
  sorry

end NUMINAMATH_CALUDE_disneyland_attractions_permutations_l3231_323128


namespace NUMINAMATH_CALUDE_no_perfect_square_E_l3231_323121

-- Define E(x) as the integer closest to x on the number line
noncomputable def E (x : ℝ) : ℤ :=
  round x

-- Theorem statement
theorem no_perfect_square_E (n : ℕ+) : ¬∃ (k : ℕ), E (n + Real.sqrt n) = k^2 := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_square_E_l3231_323121


namespace NUMINAMATH_CALUDE_fraction_equality_l3231_323185

theorem fraction_equality (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  (1 / a + 1 / b = 4 / (a + b)) → (a / b + b / a = 2) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3231_323185


namespace NUMINAMATH_CALUDE_catherine_wins_l3231_323143

/-- Represents a point on a circle -/
structure CirclePoint where
  -- Define necessary properties for a point on a circle

/-- Represents a triangle formed by three points on the circle -/
structure Triangle where
  vertex1 : CirclePoint
  vertex2 : CirclePoint
  vertex3 : CirclePoint

/-- Represents the state of the game -/
structure GameState where
  chosenTriangles : List Triangle
  currentPlayer : Bool  -- True for Peter, False for Catherine

/-- Checks if a set of triangles has a common interior point -/
def hasCommonInteriorPoint (triangles : List Triangle) : Bool :=
  sorry

/-- Checks if a triangle is valid to be chosen -/
def isValidTriangle (triangle : Triangle) (state : GameState) : Bool :=
  sorry

/-- Represents a move in the game -/
def makeMove (state : GameState) (triangle : Triangle) : Option GameState :=
  sorry

/-- Theorem stating Catherine has a winning strategy -/
theorem catherine_wins (points : List CirclePoint) 
  (h1 : points.length = 100)
  (h2 : points.Nodup) : 
  ∃ (strategy : GameState → Triangle), 
    ∀ (finalState : GameState), 
      (finalState.currentPlayer = false → 
        ∃ (move : Triangle), isValidTriangle move finalState) ∧
      (finalState.currentPlayer = true → 
        ¬∃ (move : Triangle), isValidTriangle move finalState) :=
  sorry

end NUMINAMATH_CALUDE_catherine_wins_l3231_323143


namespace NUMINAMATH_CALUDE_lcm_of_45_60_120_150_l3231_323161

theorem lcm_of_45_60_120_150 : Nat.lcm 45 (Nat.lcm 60 (Nat.lcm 120 150)) = 1800 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_45_60_120_150_l3231_323161


namespace NUMINAMATH_CALUDE_f_value_at_2_l3231_323152

/-- Given a function f(x) = x^5 + ax^3 + bx - 8 where f(-2) = 10, prove that f(2) = -26 -/
theorem f_value_at_2 (a b : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = x^5 + a*x^3 + b*x - 8)
  (h2 : f (-2) = 10) : 
  f 2 = -26 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_2_l3231_323152


namespace NUMINAMATH_CALUDE_oil_quantity_function_correct_l3231_323189

/-- Represents the remaining oil quantity in liters after t minutes -/
def Q (t : ℝ) : ℝ := 20 - 0.2 * t

/-- The initial oil quantity in liters -/
def initial_quantity : ℝ := 20

/-- The outflow rate in liters per minute -/
def outflow_rate : ℝ := 0.2

theorem oil_quantity_function_correct : 
  ∀ t : ℝ, t ≥ 0 → Q t = initial_quantity - outflow_rate * t :=
sorry

end NUMINAMATH_CALUDE_oil_quantity_function_correct_l3231_323189


namespace NUMINAMATH_CALUDE_intersection_P_Q_l3231_323188

/-- The set P -/
def P : Set ℝ := {x : ℝ | -5 < x ∧ x < 5}

/-- The set Q -/
def Q : Set ℝ := {x : ℝ | |x - 5| < 3}

/-- The open interval (2, 5) -/
def open_interval_2_5 : Set ℝ := {x : ℝ | 2 < x ∧ x < 5}

theorem intersection_P_Q : P ∩ Q = open_interval_2_5 := by sorry

end NUMINAMATH_CALUDE_intersection_P_Q_l3231_323188


namespace NUMINAMATH_CALUDE_equal_intercept_line_equation_l3231_323176

/-- A line passing through (1,2) with equal intercepts on both axes -/
structure EqualInterceptLine where
  /-- The slope of the line -/
  k : ℝ
  /-- The y-intercept of the line -/
  b : ℝ
  /-- The line passes through (1,2) -/
  point_condition : k + b = 2
  /-- The line has equal intercepts on both axes -/
  equal_intercepts : k ≠ -1 → b = k * b

/-- The equation of the line is either 2x - y = 0 or x + y - 3 = 0 -/
theorem equal_intercept_line_equation (l : EqualInterceptLine) :
  (l.k = 2 ∧ l.b = 0) ∨ (l.k = 1 ∧ l.b = 1) :=
sorry

end NUMINAMATH_CALUDE_equal_intercept_line_equation_l3231_323176


namespace NUMINAMATH_CALUDE_quadratic_minimum_l3231_323179

theorem quadratic_minimum : ∃ (min : ℝ), 
  (∀ x : ℝ, x^2 + 12*x + 18 ≥ min) ∧ 
  (∃ x : ℝ, x^2 + 12*x + 18 = min) ∧
  (min = -18) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l3231_323179


namespace NUMINAMATH_CALUDE_ball_bounce_distance_l3231_323194

/-- The total distance traveled by a bouncing ball -/
def totalDistance (initialHeight : ℝ) (reboundFactor : ℝ) (bounces : ℕ) : ℝ :=
  let descendDistances := Finset.sum (Finset.range bounces) (fun i => initialHeight * reboundFactor^i)
  let ascendDistances := Finset.sum (Finset.range (bounces - 1)) (fun i => initialHeight * reboundFactor^(i+1))
  descendDistances + ascendDistances

/-- Theorem stating the total distance traveled by the ball -/
theorem ball_bounce_distance :
  totalDistance 120 (1/3) 5 = 278.52 := by
  sorry

end NUMINAMATH_CALUDE_ball_bounce_distance_l3231_323194


namespace NUMINAMATH_CALUDE_longest_tape_l3231_323177

theorem longest_tape (minji seungyeon hyesu : ℝ) 
  (h_minji : minji = 0.74)
  (h_seungyeon : seungyeon = 13/20)
  (h_hyesu : hyesu = 4/5) :
  hyesu > minji ∧ hyesu > seungyeon :=
by sorry

end NUMINAMATH_CALUDE_longest_tape_l3231_323177


namespace NUMINAMATH_CALUDE_pizza_order_l3231_323151

theorem pizza_order (boys : ℕ) (girls : ℕ) (total_boys_pizza : ℕ) (total_pizza : ℕ) : 
  boys > girls → 
  girls = 13 → 
  total_boys_pizza = 22 → 
  (total_pizza - total_boys_pizza) * boys = total_boys_pizza * girls → 
  total_pizza = 33 := by
sorry

end NUMINAMATH_CALUDE_pizza_order_l3231_323151


namespace NUMINAMATH_CALUDE_bowl_glass_pairings_l3231_323182

/-- The number of bowls -/
def num_bowls : ℕ := 5

/-- The number of glasses -/
def num_glasses : ℕ := 5

/-- The total number of possible pairings -/
def total_pairings : ℕ := num_bowls * num_glasses

/-- Theorem: The number of possible pairings of bowls and glasses is 25 -/
theorem bowl_glass_pairings :
  total_pairings = 25 := by sorry

end NUMINAMATH_CALUDE_bowl_glass_pairings_l3231_323182


namespace NUMINAMATH_CALUDE_complement_union_M_N_l3231_323106

def U : Finset ℕ := {1,2,3,4,5,6,7,8}
def M : Finset ℕ := {1,3,5,7}
def N : Finset ℕ := {5,6,7}

theorem complement_union_M_N :
  (U \ (M ∪ N)) = {2,4,8} := by sorry

end NUMINAMATH_CALUDE_complement_union_M_N_l3231_323106


namespace NUMINAMATH_CALUDE_simplify_fraction_l3231_323120

theorem simplify_fraction : 48 / 72 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3231_323120


namespace NUMINAMATH_CALUDE_range_of_y_given_inequality_l3231_323168

/-- Custom multiplication operation on real numbers -/
def custom_mult (x y : ℝ) : ℝ := x * (1 - y)

/-- The theorem stating the range of y given the conditions -/
theorem range_of_y_given_inequality :
  (∀ x : ℝ, custom_mult (x - y) (x + y) < 1) →
  ∃ a b : ℝ, a = -1/2 ∧ b = 3/2 ∧ ∀ y : ℝ, a < y ∧ y < b :=
by sorry

end NUMINAMATH_CALUDE_range_of_y_given_inequality_l3231_323168


namespace NUMINAMATH_CALUDE_louis_age_l3231_323114

/-- Given that Carla will be 30 years old in 6 years and the sum of Carla and Louis's current ages is 55, prove that Louis is currently 31 years old. -/
theorem louis_age (carla_future_age : ℕ) (years_until_future : ℕ) (sum_of_ages : ℕ) :
  carla_future_age = 30 →
  years_until_future = 6 →
  sum_of_ages = 55 →
  sum_of_ages - (carla_future_age - years_until_future) = 31 := by
  sorry

end NUMINAMATH_CALUDE_louis_age_l3231_323114


namespace NUMINAMATH_CALUDE_triangle_properties_l3231_323199

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  a / (Real.sin A) = b / (Real.sin B) →
  a / (Real.sin A) = c / (Real.sin C) →
  a = b * (Real.cos C) + b →
  (Real.sin C = Real.tan B) ∧
  (a = 1 ∧ C < π/2 → 1/2 < c ∧ c < Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l3231_323199


namespace NUMINAMATH_CALUDE_spinner_probability_l3231_323139

theorem spinner_probability (p_A p_B p_C : ℚ) : 
  p_A = 1/3 → p_B = 1/2 → p_A + p_B + p_C = 1 → p_C = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_spinner_probability_l3231_323139


namespace NUMINAMATH_CALUDE_arccos_one_over_sqrt_two_l3231_323157

theorem arccos_one_over_sqrt_two (π : ℝ) : Real.arccos (1 / Real.sqrt 2) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_arccos_one_over_sqrt_two_l3231_323157


namespace NUMINAMATH_CALUDE_area_bisecting_line_property_l3231_323132

/-- Triangle PQR with vertices P(0, 10), Q(3, 0), and R(9, 0) -/
structure Triangle where
  P : ℝ × ℝ := (0, 10)
  Q : ℝ × ℝ := (3, 0)
  R : ℝ × ℝ := (9, 0)

/-- A line represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- Function to check if a line bisects the area of the triangle -/
def bisects_area (t : Triangle) (l : Line) : Prop :=
  sorry -- Definition of area bisection

/-- Function to check if a line passes through a point -/
def passes_through (l : Line) (p : ℝ × ℝ) : Prop :=
  sorry -- Definition of line passing through a point

/-- Theorem stating the property of the area-bisecting line -/
theorem area_bisecting_line_property (t : Triangle) :
  ∃ l : Line, bisects_area t l ∧ passes_through l t.Q ∧ l.slope + l.y_intercept = -20/3 :=
sorry

end NUMINAMATH_CALUDE_area_bisecting_line_property_l3231_323132


namespace NUMINAMATH_CALUDE_unique_integer_divisible_by_20_with_cube_root_between_9_and_91_tenths_l3231_323158

theorem unique_integer_divisible_by_20_with_cube_root_between_9_and_91_tenths : 
  ∃! n : ℕ+, 
    (∃ k : ℕ, n = 20 * k) ∧ 
    (9 : ℝ) < n.val ^ (1/3 : ℝ) ∧ 
    n.val ^ (1/3 : ℝ) < (91/10 : ℝ) ∧
    n = 740 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_divisible_by_20_with_cube_root_between_9_and_91_tenths_l3231_323158


namespace NUMINAMATH_CALUDE_unique_operation_assignment_l3231_323104

-- Define the type for arithmetic operations
inductive ArithOp
| Add
| Sub
| Mul
| Div
| Eq

-- Define a function to apply an arithmetic operation
def apply_op (op : ArithOp) (x y : ℤ) : Prop :=
  match op with
  | ArithOp.Add => x + y = 0
  | ArithOp.Sub => x - y = 0
  | ArithOp.Mul => x * y = 0
  | ArithOp.Div => y ≠ 0 ∧ x / y = 0
  | ArithOp.Eq => x = y

-- Define the theorem
theorem unique_operation_assignment :
  ∃! (A B C D E : ArithOp),
    (A ≠ B) ∧ (A ≠ C) ∧ (A ≠ D) ∧ (A ≠ E) ∧
    (B ≠ C) ∧ (B ≠ D) ∧ (B ≠ E) ∧
    (C ≠ D) ∧ (C ≠ E) ∧
    (D ≠ E) ∧
    apply_op A 4 2 ∧ apply_op B 2 2 ∧
    apply_op B 8 0 ∧ apply_op C 4 2 ∧
    apply_op D 2 3 ∧ apply_op B 5 5 ∧
    apply_op B 4 0 ∧ apply_op E 5 1 :=
sorry

end NUMINAMATH_CALUDE_unique_operation_assignment_l3231_323104


namespace NUMINAMATH_CALUDE_treasure_hunt_probability_l3231_323163

def num_islands : ℕ := 8
def num_treasure_islands : ℕ := 4
def prob_treasure : ℚ := 1/5
def prob_traps : ℚ := 1/10
def prob_neither : ℚ := 7/10

theorem treasure_hunt_probability :
  (Nat.choose num_islands num_treasure_islands) *
  (prob_treasure ^ num_treasure_islands) *
  (prob_neither ^ (num_islands - num_treasure_islands)) =
  67/2500 := by sorry

end NUMINAMATH_CALUDE_treasure_hunt_probability_l3231_323163


namespace NUMINAMATH_CALUDE_expand_product_l3231_323175

theorem expand_product (y : ℝ) : 5 * (y - 6) * (y + 9) = 5 * y^2 + 15 * y - 270 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l3231_323175


namespace NUMINAMATH_CALUDE_optimal_price_reduction_maximizes_profit_l3231_323178

/-- Profit function for shirt sales based on price reduction -/
def profit (x : ℝ) : ℝ := (2 * x + 20) * (40 - x)

/-- The price reduction that maximizes profit -/
def optimal_reduction : ℝ := 15

theorem optimal_price_reduction_maximizes_profit :
  ∀ x : ℝ, 0 ≤ x → x ≤ 40 → profit x ≤ profit optimal_reduction := by
  sorry

#check optimal_price_reduction_maximizes_profit

end NUMINAMATH_CALUDE_optimal_price_reduction_maximizes_profit_l3231_323178


namespace NUMINAMATH_CALUDE_range_of_a_l3231_323113

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) ∧ 
  (∃ x : ℝ, x^2 + 2*a*x + (2-a) = 0) → 
  a ≤ -2 ∨ a = 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3231_323113


namespace NUMINAMATH_CALUDE_max_value_theorem_l3231_323191

/-- Given a line ax + 2by - 1 = 0 intercepting a chord of length 2√3 on the circle x^2 + y^2 = 4,
    the maximum value of 3a + 2b is √10. -/
theorem max_value_theorem (a b : ℝ) : 
  (∃ x y : ℝ, a * x + 2 * b * y - 1 = 0 ∧ x^2 + y^2 = 4) →  -- Line intersects circle
  (∃ x₁ y₁ x₂ y₂ : ℝ, a * x₁ + 2 * b * y₁ - 1 = 0 ∧ 
                     a * x₂ + 2 * b * y₂ - 1 = 0 ∧
                     x₁^2 + y₁^2 = 4 ∧
                     x₂^2 + y₂^2 = 4 ∧
                     (x₁ - x₂)^2 + (y₁ - y₂)^2 = 12) →  -- Chord length is 2√3
  (a^2 + 4 * b^2 = 1) →  -- Distance from center to line is 1
  (∀ c : ℝ, 3 * a + 2 * b ≤ c → c ≥ Real.sqrt 10) ∧ 
  (∃ a₀ b₀ : ℝ, 3 * a₀ + 2 * b₀ = Real.sqrt 10) :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3231_323191


namespace NUMINAMATH_CALUDE_gcf_lcm_sum_l3231_323195

def numbers : List Nat := [18, 24, 36]

theorem gcf_lcm_sum (A B : Nat) 
  (h1 : A = Nat.gcd 18 (Nat.gcd 24 36))
  (h2 : B = Nat.lcm 18 (Nat.lcm 24 36)) : 
  A + B = 78 := by
  sorry

end NUMINAMATH_CALUDE_gcf_lcm_sum_l3231_323195


namespace NUMINAMATH_CALUDE_marty_painting_combinations_l3231_323159

/-- The number of available colors -/
def num_colors : ℕ := 5

/-- The number of available painting methods -/
def num_methods : ℕ := 4

/-- The number of restricted combinations (white paint with spray) -/
def num_restricted : ℕ := 1

theorem marty_painting_combinations :
  (num_colors - 1) * num_methods + (num_methods - 1) = 19 := by
  sorry

end NUMINAMATH_CALUDE_marty_painting_combinations_l3231_323159


namespace NUMINAMATH_CALUDE_ticket_price_possibilities_l3231_323140

theorem ticket_price_possibilities : ∃ (divisors : Finset ℕ), 
  (∀ x ∈ divisors, x ∣ 60 ∧ x ∣ 90) ∧ 
  (∀ x : ℕ, x ∣ 60 ∧ x ∣ 90 → x ∈ divisors) ∧
  Finset.card divisors = 8 :=
sorry

end NUMINAMATH_CALUDE_ticket_price_possibilities_l3231_323140


namespace NUMINAMATH_CALUDE_class_mean_score_l3231_323198

theorem class_mean_score (total_students : ℕ) (first_day_students : ℕ) (second_day_students : ℕ)
  (first_day_mean : ℚ) (second_day_mean : ℚ) :
  total_students = 50 →
  first_day_students = 40 →
  second_day_students = 10 →
  first_day_mean = 80 / 100 →
  second_day_mean = 90 / 100 →
  let overall_mean := (first_day_students * first_day_mean + second_day_students * second_day_mean) / total_students
  overall_mean = 82 / 100 := by
sorry

end NUMINAMATH_CALUDE_class_mean_score_l3231_323198


namespace NUMINAMATH_CALUDE_value_of_M_l3231_323181

theorem value_of_M : ∃ M : ℝ, (0.25 * M = 0.35 * 1504) ∧ (M = 2105.6) := by sorry

end NUMINAMATH_CALUDE_value_of_M_l3231_323181


namespace NUMINAMATH_CALUDE_equation_solution_l3231_323101

theorem equation_solution : 
  {x : ℝ | (16:ℝ)^x - (5/2) * (2:ℝ)^(2*x+1) + 4 = 0} = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3231_323101


namespace NUMINAMATH_CALUDE_lucy_fish_purchase_l3231_323118

/-- The number of fish Lucy bought -/
def fish_bought (initial final : ℝ) : ℝ := final - initial

/-- Proof that Lucy bought 280 fish -/
theorem lucy_fish_purchase : fish_bought 212.0 492 = 280 := by
  sorry

end NUMINAMATH_CALUDE_lucy_fish_purchase_l3231_323118


namespace NUMINAMATH_CALUDE_average_odd_one_digit_l3231_323172

def is_odd_one_digit (n : ℕ) : Prop := n % 2 = 1 ∧ n ≥ 1 ∧ n ≤ 9

def odd_one_digit_numbers : List ℕ := [1, 3, 5, 7, 9]

theorem average_odd_one_digit : 
  (List.sum odd_one_digit_numbers) / (List.length odd_one_digit_numbers) = 5 := by
  sorry

end NUMINAMATH_CALUDE_average_odd_one_digit_l3231_323172


namespace NUMINAMATH_CALUDE_largest_modulus_root_real_part_l3231_323103

theorem largest_modulus_root_real_part 
  (z : ℂ) 
  (hz : 5 * z^4 + 10 * z^3 + 10 * z^2 + 5 * z + 1 = 0) 
  (hmax : ∀ w : ℂ, 5 * w^4 + 10 * w^3 + 10 * w^2 + 5 * w + 1 = 0 → Complex.abs w ≤ Complex.abs z) :
  z.re = -1/2 :=
sorry

end NUMINAMATH_CALUDE_largest_modulus_root_real_part_l3231_323103


namespace NUMINAMATH_CALUDE_log2_derivative_l3231_323147

theorem log2_derivative (x : ℝ) (h : x > 0) : 
  deriv (λ x => Real.log x / Real.log 2) x = 1 / (x * Real.log 2) := by
  sorry

end NUMINAMATH_CALUDE_log2_derivative_l3231_323147


namespace NUMINAMATH_CALUDE_evaluate_expression_l3231_323170

theorem evaluate_expression : 5 - 7 * (8 - 12 / (3^2)) * 6 = -275 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3231_323170


namespace NUMINAMATH_CALUDE_digit_150_is_2_l3231_323135

/-- The sequence of digits formed by concatenating all integers from 100 down to 50 -/
def digit_sequence : List Nat := sorry

/-- The 150th digit in the sequence -/
def digit_150 : Nat := sorry

/-- Theorem stating that the 150th digit in the sequence is 2 -/
theorem digit_150_is_2 : digit_150 = 2 := by sorry

end NUMINAMATH_CALUDE_digit_150_is_2_l3231_323135


namespace NUMINAMATH_CALUDE_equation_solution_l3231_323186

theorem equation_solution :
  ∃ x : ℝ, x ≠ 0 ∧ (2 / x + (3 / x) / (6 / x) + 2 = 4) ∧ x = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3231_323186


namespace NUMINAMATH_CALUDE_initial_tangerines_count_l3231_323165

/-- The number of tangerines initially in the basket -/
def initial_tangerines : ℕ := sorry

/-- The number of tangerines Eunji ate -/
def eaten_tangerines : ℕ := 9

/-- The number of tangerines mother added -/
def added_tangerines : ℕ := 5

/-- The final number of tangerines in the basket -/
def final_tangerines : ℕ := 20

/-- Theorem stating that the initial number of tangerines was 24 -/
theorem initial_tangerines_count : initial_tangerines = 24 :=
by
  have h : initial_tangerines - eaten_tangerines + added_tangerines = final_tangerines := sorry
  sorry


end NUMINAMATH_CALUDE_initial_tangerines_count_l3231_323165


namespace NUMINAMATH_CALUDE_bowtie_equation_l3231_323100

-- Define the operation ⊛
noncomputable def bowtie (a b : ℝ) : ℝ :=
  a + Real.sqrt (b + Real.sqrt (b + 1 + Real.sqrt (b + 1 + Real.sqrt (b + 1 + Real.sqrt (b + 1)))))

-- State the theorem
theorem bowtie_equation (h : ℝ) :
  bowtie 5 h = 8 → h = 9 - Real.sqrt 10 := by sorry

end NUMINAMATH_CALUDE_bowtie_equation_l3231_323100


namespace NUMINAMATH_CALUDE_min_sum_squares_min_sum_squares_achievable_l3231_323107

theorem min_sum_squares (a b c : ℕ) (h : a + 2*b + 3*c = 73) : 
  a^2 + b^2 + c^2 ≥ 381 := by
sorry

theorem min_sum_squares_achievable : 
  ∃ (a b c : ℕ), a + 2*b + 3*c = 73 ∧ a^2 + b^2 + c^2 = 381 := by
sorry

end NUMINAMATH_CALUDE_min_sum_squares_min_sum_squares_achievable_l3231_323107


namespace NUMINAMATH_CALUDE_target_line_is_perpendicular_l3231_323138

/-- A line passing through a point and perpendicular to another line -/
def perpendicular_line (x y : ℝ) : Prop :=
  ∃ (A B C : ℝ), 
    (A * 3 + B * 4 + C = 0) ∧ 
    (A * B = -1) ∧
    (A * 2 + B * (-1) = 0)

/-- The specific line we're looking for -/
def target_line (x y : ℝ) : Prop :=
  x + 2 * y - 11 = 0

/-- Theorem stating that the target line satisfies the conditions -/
theorem target_line_is_perpendicular : 
  perpendicular_line 3 4 ↔ target_line 3 4 :=
sorry

end NUMINAMATH_CALUDE_target_line_is_perpendicular_l3231_323138


namespace NUMINAMATH_CALUDE_dvd_total_count_l3231_323130

theorem dvd_total_count (store_dvds : ℕ) (online_dvds : ℕ) : 
  store_dvds = 8 → online_dvds = 2 → store_dvds + online_dvds = 10 := by
  sorry

end NUMINAMATH_CALUDE_dvd_total_count_l3231_323130


namespace NUMINAMATH_CALUDE_range_of_inequality_l3231_323148

/-- An even function that is monotonically decreasing on (-∞,0] -/
class EvenDecreasingFunction (f : ℝ → ℝ) : Prop where
  even : ∀ x, f x = f (-x)
  decreasing : ∀ {x y}, x ≤ y → y ≤ 0 → f y ≤ f x

/-- The theorem statement -/
theorem range_of_inequality (f : ℝ → ℝ) [EvenDecreasingFunction f] :
  {x : ℝ | f (2*x + 1) < f 3} = Set.Ioo (-2) 1 := by sorry

end NUMINAMATH_CALUDE_range_of_inequality_l3231_323148


namespace NUMINAMATH_CALUDE_jean_side_spots_l3231_323174

/-- Represents the number of spots on different parts of Jean the jaguar's body. -/
structure JeanSpots where
  total : ℕ
  upperTorso : ℕ
  backHindquarters : ℕ
  sides : ℕ

/-- Theorem stating the number of spots on Jean's sides given the conditions. -/
theorem jean_side_spots (j : JeanSpots) 
    (h1 : j.upperTorso = j.total / 2)
    (h2 : j.backHindquarters = j.total / 3)
    (h3 : j.upperTorso = 30)
    (h4 : j.total = j.upperTorso + j.backHindquarters + j.sides) :
    j.sides = 10 := by
  sorry

end NUMINAMATH_CALUDE_jean_side_spots_l3231_323174


namespace NUMINAMATH_CALUDE_hundredthOddPositiveInteger_l3231_323142

/-- The nth odd positive integer -/
def nthOddPositiveInteger (n : ℕ) : ℕ := 2 * n - 1

/-- The 100th odd positive integer is 199 -/
theorem hundredthOddPositiveInteger : nthOddPositiveInteger 100 = 199 := by
  sorry

end NUMINAMATH_CALUDE_hundredthOddPositiveInteger_l3231_323142


namespace NUMINAMATH_CALUDE_linear_function_range_l3231_323153

theorem linear_function_range (m : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → (m + 2) * x₁ + (1 - m) > (m + 2) * x₂ + (1 - m)) →
  (∃ x : ℝ, x > 0 ∧ (m + 2) * x + (1 - m) = 0) →
  m < -2 := by
sorry

end NUMINAMATH_CALUDE_linear_function_range_l3231_323153


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l3231_323110

theorem pure_imaginary_complex_number (m : ℝ) :
  let z : ℂ := (m^2 - 1) + (m + 1) * Complex.I
  (z.re = 0 ∧ z ≠ 0) → m = 1 := by sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l3231_323110


namespace NUMINAMATH_CALUDE_pizza_toppings_combinations_l3231_323126

theorem pizza_toppings_combinations : Nat.choose 9 3 = 84 := by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_combinations_l3231_323126


namespace NUMINAMATH_CALUDE_quadratic_factorization_sum_l3231_323196

theorem quadratic_factorization_sum (a b c : ℤ) : 
  (∀ x : ℝ, x^2 + 15*x + 36 = (x + a)*(x + b)) →
  (∀ x : ℝ, x^2 + 7*x - 60 = (x + b)*(x - c)) →
  a + b + c = 20 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_sum_l3231_323196


namespace NUMINAMATH_CALUDE_percentage_decrease_l3231_323141

theorem percentage_decrease (initial : ℝ) (increase : ℝ) (final : ℝ) :
  initial = 1500 →
  increase = 20 →
  final = 1080 →
  ∃ y : ℝ, y = 40 ∧ final = (initial * (1 + increase / 100)) * (1 - y / 100) :=
by sorry

end NUMINAMATH_CALUDE_percentage_decrease_l3231_323141


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l3231_323155

theorem fraction_to_decimal : (17 : ℚ) / 50 = 0.34 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l3231_323155


namespace NUMINAMATH_CALUDE_price_reduction_proof_l3231_323122

/-- The original selling price in yuan -/
def original_price : ℝ := 40

/-- The cost price in yuan -/
def cost_price : ℝ := 30

/-- The initial daily sales volume -/
def initial_sales : ℕ := 48

/-- The price after two consecutive reductions in yuan -/
def reduced_price : ℝ := 32.4

/-- The increase in daily sales for every 0.5 yuan reduction in price -/
def sales_increase_rate : ℝ := 8

/-- The desired daily profit in yuan -/
def desired_profit : ℝ := 504

/-- The percentage reduction that results in the reduced price after two consecutive reductions -/
def percentage_reduction : ℝ := 0.1

/-- The price reduction that achieves the desired daily profit -/
def price_reduction : ℝ := 3

theorem price_reduction_proof :
  (∃ x : ℝ, original_price * (1 - x)^2 = reduced_price ∧ 0 < x ∧ x < 1 ∧ x = percentage_reduction) ∧
  (∃ y : ℝ, (original_price - cost_price - y) * (initial_sales + sales_increase_rate * y) = desired_profit ∧ y = price_reduction) :=
sorry

end NUMINAMATH_CALUDE_price_reduction_proof_l3231_323122


namespace NUMINAMATH_CALUDE_rect_to_spherical_conversion_l3231_323162

/-- Conversion from rectangular to spherical coordinates -/
theorem rect_to_spherical_conversion
  (x y z : ℝ)
  (ρ θ φ : ℝ)
  (h_ρ : ρ > 0)
  (h_θ : 0 ≤ θ ∧ θ < 2 * Real.pi)
  (h_φ : 0 ≤ φ ∧ φ ≤ Real.pi)
  (h_x : x = 0)
  (h_y : y = -3 * Real.sqrt 3)
  (h_z : z = 3)
  (h_ρ_val : ρ = 6)
  (h_θ_val : θ = 3 * Real.pi / 2)
  (h_φ_val : φ = Real.pi / 3) :
  x = ρ * Real.sin φ * Real.cos θ ∧
  y = ρ * Real.sin φ * Real.sin θ ∧
  z = ρ * Real.cos φ :=
by
  sorry

#check rect_to_spherical_conversion

end NUMINAMATH_CALUDE_rect_to_spherical_conversion_l3231_323162


namespace NUMINAMATH_CALUDE_beacon_population_l3231_323171

/-- Given the populations of three cities with specific relationships, prove the population of Beacon. -/
theorem beacon_population
  (richmond victoria beacon : ℕ)
  (h1 : richmond = victoria + 1000)
  (h2 : victoria = 4 * beacon)
  (h3 : richmond = 3000) :
  beacon = 500 := by
  sorry

end NUMINAMATH_CALUDE_beacon_population_l3231_323171


namespace NUMINAMATH_CALUDE_expressions_are_integers_l3231_323160

-- Define the expressions as functions
def expr1 (m n : ℕ) : ℚ := (m + n).factorial / (m.factorial * n.factorial)

def expr2 (m n : ℕ) : ℚ := ((2*m).factorial * (2*n).factorial) / 
  (m.factorial * n.factorial * (m + n).factorial)

def expr3 (m n : ℕ) : ℚ := ((5*m).factorial * (5*n).factorial) / 
  (m.factorial * n.factorial * (3*m + n).factorial * (3*n + m).factorial)

def expr4 (m n : ℕ) : ℚ := ((3*m + 3*n).factorial * (3*n).factorial * (2*m).factorial * (2*n).factorial) / 
  ((2*m + 3*n).factorial * (m + 2*n).factorial * m.factorial * (n.factorial^2) * (m + n).factorial)

-- Theorem statement
theorem expressions_are_integers (m n : ℕ) : 
  (∃ k : ℤ, expr1 m n = k) ∧ 
  (∃ k : ℤ, expr2 m n = k) ∧ 
  (∃ k : ℤ, expr3 m n = k) ∧ 
  (∃ k : ℤ, expr4 m n = k) := by
  sorry

end NUMINAMATH_CALUDE_expressions_are_integers_l3231_323160


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3231_323131

theorem trigonometric_identity (t : ℝ) (h : 3 * Real.cos (2 * t) - Real.sin (2 * t) ≠ 0) :
  (6 * Real.cos (2 * t)^3 + 2 * Real.sin (2 * t)^3) / (3 * Real.cos (2 * t) - Real.sin (2 * t))
  = Real.cos (4 * t) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3231_323131


namespace NUMINAMATH_CALUDE_product_of_points_on_line_l3231_323187

/-- A line passing through the origin with slope 1/4 -/
def line_k (x y : ℝ) : Prop := y = (1/4) * x

theorem product_of_points_on_line (x y : ℝ) :
  line_k x 8 → line_k 20 y → x * y = 160 := by
  sorry

end NUMINAMATH_CALUDE_product_of_points_on_line_l3231_323187


namespace NUMINAMATH_CALUDE_intersection_P_Q_l3231_323164

def P : Set ℝ := {-3, 0, 2, 4}
def Q : Set ℝ := {x : ℝ | -1 < x ∧ x < 3}

theorem intersection_P_Q : P ∩ Q = {0, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_P_Q_l3231_323164


namespace NUMINAMATH_CALUDE_prime_divisor_form_l3231_323133

theorem prime_divisor_form (n : ℕ) (q : ℕ) (h_prime : Nat.Prime q) (h_divides : q ∣ 2^(2^n) + 1) :
  ∃ x : ℤ, q = 2^(n + 1) * x + 1 :=
sorry

end NUMINAMATH_CALUDE_prime_divisor_form_l3231_323133


namespace NUMINAMATH_CALUDE_smallest_p_is_three_l3231_323123

theorem smallest_p_is_three (p q s r : ℕ) : 
  Nat.Prime p → Nat.Prime q → Nat.Prime s → Nat.Prime r →
  p + q + s = r →
  2 < p → p < q → q < s →
  ∀ p' : ℕ, (Nat.Prime p' ∧ 
             (∃ q' s' r' : ℕ, Nat.Prime q' ∧ Nat.Prime s' ∧ Nat.Prime r' ∧
                              p' + q' + s' = r' ∧
                              2 < p' ∧ p' < q' ∧ q' < s')) →
            p' ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_p_is_three_l3231_323123


namespace NUMINAMATH_CALUDE_min_omega_value_l3231_323150

open Real

/-- Given a function f(x) = 2sin(ωx + φ) where ω > 0, 
    if the graph is symmetrical about the line x = π/3 and f(π/12) = 0, 
    then the minimum value of ω is 2. -/
theorem min_omega_value (ω φ : ℝ) (hω : ω > 0) :
  (∀ x, 2 * sin (ω * x + φ) = 2 * sin (ω * (2 * π/3 - x) + φ)) →
  2 * sin (ω * π/12 + φ) = 0 →
  ω ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_min_omega_value_l3231_323150


namespace NUMINAMATH_CALUDE_point_P_conditions_l3231_323180

def point_P (m : ℝ) : ℝ × ℝ := (3*m - 6, m + 1)

def point_A : ℝ × ℝ := (-1, 2)

theorem point_P_conditions (m : ℝ) :
  (∃ m, point_P m = (-9, 0) ∧ (point_P m).2 = 0) ∧
  (∃ m, point_P m = (-1, 8/3) ∧ (point_P m).1 = (point_A).1) :=
by sorry

end NUMINAMATH_CALUDE_point_P_conditions_l3231_323180


namespace NUMINAMATH_CALUDE_initial_concentration_proof_l3231_323169

/-- Proves that the initial concentration of a solution is 45% given the specified conditions -/
theorem initial_concentration_proof (initial_concentration : ℝ) : 
  (0.5 * initial_concentration + 0.5 * 0.25 = 0.35) → initial_concentration = 0.45 := by
  sorry

end NUMINAMATH_CALUDE_initial_concentration_proof_l3231_323169


namespace NUMINAMATH_CALUDE_initials_count_l3231_323149

/-- The number of letters available (A through H) -/
def num_letters : ℕ := 8

/-- The length of each set of initials -/
def set_length : ℕ := 4

/-- The number of different four-letter sets of initials possible using letters A through H -/
theorem initials_count : (num_letters ^ set_length : ℕ) = 4096 := by
  sorry

end NUMINAMATH_CALUDE_initials_count_l3231_323149


namespace NUMINAMATH_CALUDE_cans_per_row_is_twelve_l3231_323173

/-- The number of rows on one shelf -/
def rows_per_shelf : ℕ := 4

/-- The number of shelves in one closet -/
def shelves_per_closet : ℕ := 10

/-- The total number of cans Jack can store in one closet -/
def cans_per_closet : ℕ := 480

/-- The number of cans Jack can fit in one row -/
def cans_per_row : ℕ := cans_per_closet / (shelves_per_closet * rows_per_shelf)

theorem cans_per_row_is_twelve : cans_per_row = 12 := by
  sorry

end NUMINAMATH_CALUDE_cans_per_row_is_twelve_l3231_323173


namespace NUMINAMATH_CALUDE_triangle_angle_ratio_l3231_323190

theorem triangle_angle_ratio (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- All angles are positive
  a + b + c = 180 →        -- Sum of angles is 180 degrees
  a = 20 →                 -- Smallest angle is 20 degrees
  c = 5 * a →              -- Largest angle is 5 times the smallest
  a ≤ b ∧ b ≤ c →          -- a is smallest, c is largest
  b / a = 3 :=             -- Ratio of middle to smallest is 3:1
by sorry

end NUMINAMATH_CALUDE_triangle_angle_ratio_l3231_323190


namespace NUMINAMATH_CALUDE_coefficient_a2_l3231_323184

theorem coefficient_a2 (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, a₀ + a₁ * (2 * x - 1) + a₂ * (2 * x - 1)^2 + a₃ * (2 * x - 1)^3 + a₄ * (2 * x - 1)^4 = x^4) →
  a₂ = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_a2_l3231_323184


namespace NUMINAMATH_CALUDE_opposite_of_three_l3231_323112

-- Define the opposite function for real numbers
def opposite (x : ℝ) : ℝ := -x

-- State the theorem
theorem opposite_of_three : opposite 3 = -3 := by sorry

end NUMINAMATH_CALUDE_opposite_of_three_l3231_323112


namespace NUMINAMATH_CALUDE_tan_a_value_l3231_323137

theorem tan_a_value (a : Real) (h : Real.tan (a + π/4) = 1/7) : 
  Real.tan a = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_tan_a_value_l3231_323137


namespace NUMINAMATH_CALUDE_givenCurve_is_parabola_l3231_323111

/-- A curve in 2D space represented by parametric equations -/
structure ParametricCurve where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- Definition of a parabola in standard form -/
def IsParabola (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The given parametric curve -/
def givenCurve : ParametricCurve where
  x := λ t => t
  y := λ t => t^2 + 1

/-- Theorem stating that the given curve is a parabola -/
theorem givenCurve_is_parabola :
  IsParabola (λ x => givenCurve.y (givenCurve.x⁻¹ x)) :=
sorry

end NUMINAMATH_CALUDE_givenCurve_is_parabola_l3231_323111


namespace NUMINAMATH_CALUDE_may_greatest_drop_l3231_323193

/-- Represents the months in the first half of 2022 -/
inductive Month
  | january
  | february
  | march
  | april
  | may
  | june

/-- Represents the price change for each month -/
def priceChange (m : Month) : ℝ :=
  match m with
  | .january => -1.0
  | .february => 1.5
  | .march => -3.0
  | .april => 2.0
  | .may => -4.0
  | .june => -1.5

/-- The economic event occurred in May -/
def economicEventMonth : Month := .may

/-- Defines the greatest monthly drop in price -/
def hasGreatestDrop (m : Month) : Prop :=
  ∀ m', priceChange m ≤ priceChange m'

/-- Theorem stating that May has the greatest monthly drop in price -/
theorem may_greatest_drop :
  hasGreatestDrop .may :=
sorry

end NUMINAMATH_CALUDE_may_greatest_drop_l3231_323193


namespace NUMINAMATH_CALUDE_quadratic_roots_implies_composite_l3231_323154

/-- A number is composite if it's the product of two integers each greater than 1 -/
def IsComposite (n : ℕ) : Prop :=
  ∃ (p q : ℕ), p > 1 ∧ q > 1 ∧ n = p * q

/-- The roots of the quadratic x^2 + ax + b + 1 are positive integers -/
def HasPositiveIntegerRoots (a b : ℤ) : Prop :=
  ∃ (c d : ℕ), c > 0 ∧ d > 0 ∧ c^2 + a*c + b + 1 = 0 ∧ d^2 + a*d + b + 1 = 0

/-- If x^2 + ax + b + 1 has positive integer roots, then a^2 + b^2 is composite -/
theorem quadratic_roots_implies_composite (a b : ℤ) :
  HasPositiveIntegerRoots a b → IsComposite (Int.natAbs (a^2 + b^2)) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_implies_composite_l3231_323154


namespace NUMINAMATH_CALUDE_g_composition_equals_49_l3231_323197

def g (n : ℕ) : ℕ :=
  if n < 5 then n^2 + 1 else 2*n + 3

theorem g_composition_equals_49 : g (g (g 3)) = 49 := by
  sorry

end NUMINAMATH_CALUDE_g_composition_equals_49_l3231_323197


namespace NUMINAMATH_CALUDE_willys_age_proof_l3231_323125

theorem willys_age_proof :
  ∃ (P : ℤ → ℤ) (A : ℤ),
    (∀ x, ∃ (a₀ a₁ a₂ a₃ : ℤ), P x = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3) ∧
    P 7 = 77 ∧
    P 8 = 85 ∧
    A > 8 ∧
    P A = 0 ∧
    A = 14 := by
  sorry

end NUMINAMATH_CALUDE_willys_age_proof_l3231_323125


namespace NUMINAMATH_CALUDE_ivy_cupcakes_l3231_323156

theorem ivy_cupcakes (morning_cupcakes : ℕ) (afternoon_difference : ℕ) : 
  morning_cupcakes = 20 →
  afternoon_difference = 15 →
  morning_cupcakes + (morning_cupcakes + afternoon_difference) = 55 :=
by
  sorry

end NUMINAMATH_CALUDE_ivy_cupcakes_l3231_323156


namespace NUMINAMATH_CALUDE_sqrt_relationship_l3231_323134

theorem sqrt_relationship (h : Real.sqrt 22500 = 150) : Real.sqrt 0.0225 = 0.15 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_relationship_l3231_323134


namespace NUMINAMATH_CALUDE_max_t_value_min_y_value_equality_condition_l3231_323192

-- Define the inequality function
def f (x t : ℝ) : ℝ := |3*x + 2| + |3*x - 1| - t

-- Part 1: Maximum value of t
theorem max_t_value :
  (∀ x : ℝ, f x 3 ≥ 0) ∧ 
  (∀ t : ℝ, t > 3 → ∃ x : ℝ, f x t < 0) :=
sorry

-- Part 2: Minimum value of y
theorem min_y_value :
  ∀ m n : ℝ, m > 0 → n > 0 → 4*m + 5*n = 3 →
  1 / (m + 2*n) + 4 / (3*m + 3*n) ≥ 3 :=
sorry

-- Equality condition
theorem equality_condition :
  ∀ m n : ℝ, m > 0 → n > 0 → 4*m + 5*n = 3 →
  (1 / (m + 2*n) + 4 / (3*m + 3*n) = 3 ↔ m = 1/3 ∧ n = 1/3) :=
sorry

end NUMINAMATH_CALUDE_max_t_value_min_y_value_equality_condition_l3231_323192


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3231_323183

/-- An isosceles triangle with side lengths 2 and 5 has a perimeter of 12. -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ),
  a > 0 ∧ b > 0 ∧ c > 0 →  -- positive side lengths
  (a = b ∧ (a = 2 ∧ c = 5 ∨ a = 5 ∧ c = 2) ∨
   a = c ∧ (a = 2 ∧ b = 5 ∨ a = 5 ∧ b = 2) ∨
   b = c ∧ (b = 2 ∧ a = 5 ∨ b = 5 ∧ a = 2)) →  -- isosceles with sides 2 and 5
  a + b + c = 12  -- perimeter is 12
:= by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3231_323183


namespace NUMINAMATH_CALUDE_inequality_proof_l3231_323124

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a*b + b*c + c*a) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3231_323124


namespace NUMINAMATH_CALUDE_min_value_theorem_l3231_323119

theorem min_value_theorem (x : ℝ) (h : x > 0) : 
  6 * x + 1 / x^6 ≥ 7 ∧ ∃ y > 0, 6 * y + 1 / y^6 = 7 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3231_323119


namespace NUMINAMATH_CALUDE_absolute_value_simplification_l3231_323115

theorem absolute_value_simplification : |-4^2 + 6| = 10 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_simplification_l3231_323115


namespace NUMINAMATH_CALUDE_price_increase_decrease_l3231_323166

theorem price_increase_decrease (x : ℝ) : 
  (100 + x) * (100 - 23.076923076923077) / 100 = 100 → x = 30 := by
sorry

end NUMINAMATH_CALUDE_price_increase_decrease_l3231_323166


namespace NUMINAMATH_CALUDE_consecutive_sum_100_l3231_323117

theorem consecutive_sum_100 (n : ℕ) :
  (∃ (m : ℕ), m = n ∧ 
    n + (n + 1) + (n + 2) + (n + 3) + (n + 4) = 100) →
  n = 18 := by
sorry

end NUMINAMATH_CALUDE_consecutive_sum_100_l3231_323117


namespace NUMINAMATH_CALUDE_binary_11100_to_quaternary_l3231_323136

/-- Converts a binary number (represented as a list of bits) to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Converts a natural number to its quaternary representation (as a list of digits) -/
def decimal_to_quaternary (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
    aux n []

/-- The binary number 11100₂ -/
def binary_11100 : List Bool := [true, true, true, false, false]

theorem binary_11100_to_quaternary :
  decimal_to_quaternary (binary_to_decimal binary_11100) = [1, 3, 0] :=
sorry

end NUMINAMATH_CALUDE_binary_11100_to_quaternary_l3231_323136


namespace NUMINAMATH_CALUDE_factorial_sum_unique_solution_l3231_323127

theorem factorial_sum_unique_solution :
  ∀ n a b c : ℕ, n.factorial = a.factorial + b.factorial + c.factorial →
  n = 3 ∧ a = 2 ∧ b = 2 ∧ c = 2 :=
by sorry

end NUMINAMATH_CALUDE_factorial_sum_unique_solution_l3231_323127


namespace NUMINAMATH_CALUDE_car_trip_distance_l3231_323108

theorem car_trip_distance (D : ℝ) 
  (h1 : D > 0)
  (h2 : (1/2) * D + (1/4) * ((1/2) * D) + (1/3) * ((1/2) * D - (1/4) * ((1/2) * D)) + 270 = D) :
  (1/4) * D = 270 := by
  sorry

end NUMINAMATH_CALUDE_car_trip_distance_l3231_323108


namespace NUMINAMATH_CALUDE_inequality_proof_l3231_323102

theorem inequality_proof (a b c d : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (h_eq : (a + 1)⁻¹ + (b + 1)⁻¹ + (c + 1)⁻¹ + (d + 1)⁻¹ = 3) : 
  (a * b * c)^(1/3) + (b * c * d)^(1/3) + (c * d * a)^(1/3) + (d * a * b)^(1/3) ≤ 4/3 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3231_323102


namespace NUMINAMATH_CALUDE_average_tickets_sold_l3231_323144

/-- Proves that the average number of tickets sold per member is 66 given the conditions -/
theorem average_tickets_sold (male_count : ℕ) (female_count : ℕ) 
  (male_female_ratio : female_count = 2 * male_count)
  (female_avg : ℝ) (male_avg : ℝ)
  (h_female_avg : female_avg = 70)
  (h_male_avg : male_avg = 58) :
  let total_tickets := female_count * female_avg + male_count * male_avg
  let total_members := male_count + female_count
  total_tickets / total_members = 66 := by
sorry

end NUMINAMATH_CALUDE_average_tickets_sold_l3231_323144


namespace NUMINAMATH_CALUDE_number_division_problem_l3231_323167

theorem number_division_problem : ∃ x : ℚ, (x / 5) - (x / 6) = 30 ∧ x = 900 := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l3231_323167


namespace NUMINAMATH_CALUDE_logarithm_equation_solution_l3231_323109

theorem logarithm_equation_solution (b x : ℝ) (hb_pos : b > 0) (hb_neq_one : b ≠ 1) (hx_neq_one : x ≠ 1) 
  (h_eq : (Real.log x) / (3 * Real.log b) + (Real.log b) / (3 * Real.log x) = 1) :
  x = b ^ ((3 + Real.sqrt 5) / 2) ∨ x = b ^ ((3 - Real.sqrt 5) / 2) := by
  sorry

end NUMINAMATH_CALUDE_logarithm_equation_solution_l3231_323109


namespace NUMINAMATH_CALUDE_candidate_vote_difference_l3231_323146

theorem candidate_vote_difference (total_votes : ℝ) (candidate_percentage : ℝ) : 
  total_votes = 10000.000000000002 →
  candidate_percentage = 0.4 →
  (total_votes * (1 - candidate_percentage) - total_votes * candidate_percentage) = 2000 := by
sorry

end NUMINAMATH_CALUDE_candidate_vote_difference_l3231_323146
