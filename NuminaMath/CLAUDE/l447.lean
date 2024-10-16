import Mathlib

namespace NUMINAMATH_CALUDE_parallel_transitivity_perpendicular_to_parallel_not_always_intersects_l447_44730

-- Define a 3D space
structure Space3D where
  -- Add necessary fields for 3D space

-- Define a line in 3D space
structure Line3D where
  -- Add necessary fields for a line in 3D space

-- Define parallel lines in 3D space
def parallel (l1 l2 : Line3D) : Prop :=
  sorry

-- Define perpendicular lines in 3D space
def perpendicular (l1 l2 : Line3D) : Prop :=
  sorry

-- Define line intersection in 3D space
def intersects (l1 l2 : Line3D) : Prop :=
  sorry

theorem parallel_transitivity (l1 l2 l3 : Line3D) :
  parallel l1 l2 → parallel l2 l3 → parallel l1 l3 :=
  sorry

theorem perpendicular_to_parallel (l1 l2 l3 : Line3D) :
  parallel l1 l2 → perpendicular l3 l1 → perpendicular l3 l2 :=
  sorry

theorem not_always_intersects (l1 l2 l3 : Line3D) :
  ¬(parallel l1 l2 → intersects l3 l1 → intersects l3 l2) :=
  sorry

end NUMINAMATH_CALUDE_parallel_transitivity_perpendicular_to_parallel_not_always_intersects_l447_44730


namespace NUMINAMATH_CALUDE_max_b_value_l447_44707

/-- Given a box with volume 360 cubic units and integer dimensions a, b, and c
    satisfying 1 < c < b < a, the maximum possible value of b is 12. -/
theorem max_b_value (a b c : ℕ) : 
  a * b * c = 360 → 
  1 < c → c < b → b < a → 
  b ≤ 12 ∧ ∃ (a' b' c' : ℕ), a' * b' * c' = 360 ∧ 1 < c' ∧ c' < b' ∧ b' < a' ∧ b' = 12 :=
by sorry

end NUMINAMATH_CALUDE_max_b_value_l447_44707


namespace NUMINAMATH_CALUDE_square_rectangle_perimeter_equality_l447_44704

theorem square_rectangle_perimeter_equality :
  ∀ (square_side : ℝ) (rect_length rect_area : ℝ),
    square_side = 15 →
    rect_length = 18 →
    rect_area = 216 →
    4 * square_side = 2 * (rect_length + (rect_area / rect_length)) := by
  sorry

end NUMINAMATH_CALUDE_square_rectangle_perimeter_equality_l447_44704


namespace NUMINAMATH_CALUDE_student_assistant_sequences_l447_44766

/-- The number of students in the class -/
def num_students : ℕ := 15

/-- The number of class meetings per week -/
def meetings_per_week : ℕ := 5

/-- The number of different sequences of student assistants possible in one week -/
def num_sequences : ℕ := num_students ^ meetings_per_week

theorem student_assistant_sequences :
  num_sequences = 759375 :=
sorry

end NUMINAMATH_CALUDE_student_assistant_sequences_l447_44766


namespace NUMINAMATH_CALUDE_inequality_solution_l447_44739

theorem inequality_solution : 
  {x : ℕ | x > 0 ∧ 3 * x - 4 < 2 * x} = {1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l447_44739


namespace NUMINAMATH_CALUDE_absolute_value_inequality_range_l447_44703

theorem absolute_value_inequality_range :
  ∀ a : ℝ, (∀ x : ℝ, |x + 3| + |x - 1| ≥ a) ↔ a ≤ 4 :=
by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_range_l447_44703


namespace NUMINAMATH_CALUDE_sock_knitting_problem_l447_44765

/-- The number of days it takes A to knit a pair of socks -/
def days_A : ℝ := sorry

/-- The number of days it takes B to knit a pair of socks -/
def days_B : ℝ := 6

/-- The number of days it takes A and B together to knit two pairs of socks -/
def days_together : ℝ := 4

/-- The number of pairs of socks A and B knit together -/
def pairs_together : ℝ := 2

theorem sock_knitting_problem :
  (1 / days_A + 1 / days_B) * days_together = pairs_together ∧ days_A = 3 := by sorry

end NUMINAMATH_CALUDE_sock_knitting_problem_l447_44765


namespace NUMINAMATH_CALUDE_cyclic_inequality_l447_44732

theorem cyclic_inequality (x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ x₉ : ℝ) 
  (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₃ > 0) (h₄ : x₄ > 0) (h₅ : x₅ > 0)
  (h₆ : x₆ > 0) (h₇ : x₇ > 0) (h₈ : x₈ > 0) (h₉ : x₉ > 0) :
  (x₁ - x₃) / (x₁ * x₃ + 2 * x₂ * x₃ + x₂^2) +
  (x₂ - x₄) / (x₂ * x₄ + 2 * x₃ * x₄ + x₃^2) +
  (x₃ - x₅) / (x₃ * x₅ + 2 * x₄ * x₅ + x₄^2) +
  (x₄ - x₆) / (x₄ * x₆ + 2 * x₅ * x₆ + x₅^2) +
  (x₅ - x₇) / (x₅ * x₇ + 2 * x₆ * x₇ + x₆^2) +
  (x₆ - x₈) / (x₆ * x₈ + 2 * x₇ * x₈ + x₇^2) +
  (x₇ - x₉) / (x₇ * x₉ + 2 * x₈ * x₉ + x₈^2) +
  (x₈ - x₁) / (x₈ * x₁ + 2 * x₉ * x₁ + x₉^2) +
  (x₉ - x₂) / (x₉ * x₂ + 2 * x₁ * x₂ + x₁^2) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_inequality_l447_44732


namespace NUMINAMATH_CALUDE_negative_sqrt_two_less_than_negative_one_l447_44738

theorem negative_sqrt_two_less_than_negative_one : -Real.sqrt 2 < -1 := by
  sorry

end NUMINAMATH_CALUDE_negative_sqrt_two_less_than_negative_one_l447_44738


namespace NUMINAMATH_CALUDE_largest_five_digit_with_product_120_l447_44737

def is_five_digit (n : ℕ) : Prop := n ≥ 10000 ∧ n < 100000

def digit_product (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.prod

theorem largest_five_digit_with_product_120 :
  ∀ n : ℕ, is_five_digit n → digit_product n = 120 → n ≤ 85311 :=
by sorry

end NUMINAMATH_CALUDE_largest_five_digit_with_product_120_l447_44737


namespace NUMINAMATH_CALUDE_water_height_in_conical_tank_l447_44799

/-- The height of water in an inverted conical tank -/
theorem water_height_in_conical_tank 
  (tank_radius : ℝ) 
  (tank_height : ℝ) 
  (water_volume_percentage : ℝ) 
  (h : water_volume_percentage = 0.4) 
  (r : tank_radius = 10) 
  (h : tank_height = 60) : 
  ∃ (water_height : ℝ), water_height = 12 * (3 ^ (1/3 : ℝ)) :=
sorry

end NUMINAMATH_CALUDE_water_height_in_conical_tank_l447_44799


namespace NUMINAMATH_CALUDE_cube_root_sixteen_over_thirtytwo_l447_44748

theorem cube_root_sixteen_over_thirtytwo : 
  (16 / 32 : ℝ)^(1/3) = 1 / 2^(1/3) := by sorry

end NUMINAMATH_CALUDE_cube_root_sixteen_over_thirtytwo_l447_44748


namespace NUMINAMATH_CALUDE_derivative_of_2_sqrt_x_cubed_l447_44797

theorem derivative_of_2_sqrt_x_cubed (x : ℝ) (h : x > 0) :
  deriv (λ x => 2 * Real.sqrt (x^3)) x = 3 * Real.sqrt x :=
by sorry

end NUMINAMATH_CALUDE_derivative_of_2_sqrt_x_cubed_l447_44797


namespace NUMINAMATH_CALUDE_towel_area_decrease_l447_44734

theorem towel_area_decrease (L B : ℝ) (hL : L > 0) (hB : B > 0) :
  let original_area := L * B
  let new_length := L * (1 - 0.2)
  let new_breadth := B * (1 - 0.1)
  let new_area := new_length * new_breadth
  (original_area - new_area) / original_area * 100 = 28 := by
sorry

end NUMINAMATH_CALUDE_towel_area_decrease_l447_44734


namespace NUMINAMATH_CALUDE_mitch_hourly_rate_l447_44770

/-- Mitch's hourly rate calculation --/
theorem mitch_hourly_rate :
  ∀ (weekday_hours_per_day : ℕ) 
    (weekend_hours_per_day : ℕ) 
    (weekday_count : ℕ) 
    (weekend_count : ℕ) 
    (weekend_multiplier : ℕ) 
    (weekly_earnings : ℕ),
  weekday_hours_per_day = 5 →
  weekend_hours_per_day = 3 →
  weekday_count = 5 →
  weekend_count = 2 →
  weekend_multiplier = 2 →
  weekly_earnings = 111 →
  (weekly_earnings : ℚ) / 
    (weekday_hours_per_day * weekday_count + 
     weekend_hours_per_day * weekend_count * weekend_multiplier) = 3 := by
  sorry

#check mitch_hourly_rate

end NUMINAMATH_CALUDE_mitch_hourly_rate_l447_44770


namespace NUMINAMATH_CALUDE_train_speed_before_accelerating_l447_44775

/-- Calculates the average speed of a train before accelerating -/
theorem train_speed_before_accelerating
  (v : ℝ) (s : ℝ) 
  (h1 : v > 0) (h2 : s > 0) :
  ∃ (x : ℝ), x > 0 ∧ s / x = (s + 50) / (x + v) ∧ x = s * v / 50 :=
by sorry

end NUMINAMATH_CALUDE_train_speed_before_accelerating_l447_44775


namespace NUMINAMATH_CALUDE_reflection_sum_l447_44712

/-- Given that the point (-4, 2) is reflected across the line y = mx + b to the point (6, -2),
    prove that m + b = 0 -/
theorem reflection_sum (m b : ℝ) : 
  (∃ (x y : ℝ), y = m * x + b ∧ 
   (x - (-4))^2 + (y - 2)^2 = (x - 6)^2 + (y - (-2))^2 ∧
   (x - (-4)) * (6 - x) + (y - 2) * (-2 - y) = 0) →
  m + b = 0 := by
sorry

end NUMINAMATH_CALUDE_reflection_sum_l447_44712


namespace NUMINAMATH_CALUDE_carlos_pesos_l447_44758

/-- The exchange rate from Mexican pesos to U.S. dollars -/
def exchange_rate : ℚ := 8 / 14

/-- The amount spent in U.S. dollars -/
def amount_spent : ℕ := 50

/-- The remaining amount is three times the spent amount -/
def remaining_ratio : ℕ := 3

/-- The number of Mexican pesos Carlos had -/
def p : ℕ := 350

theorem carlos_pesos :
  p * exchange_rate - amount_spent = remaining_ratio * amount_spent := by
  sorry

end NUMINAMATH_CALUDE_carlos_pesos_l447_44758


namespace NUMINAMATH_CALUDE_corner_start_winning_strategy_adjacent_start_winning_strategy_l447_44714

/-- Represents the players in the game -/
inductive Player
| A
| B

/-- Represents the game state -/
structure GameState where
  n : Nat
  currentPosition : Nat × Nat
  visitedPositions : Set (Nat × Nat)
  currentPlayer : Player

/-- Defines a winning strategy for a player -/
def HasWinningStrategy (player : Player) (initialState : GameState) : Prop :=
  ∃ (strategy : GameState → Nat × Nat),
    ∀ (gameState : GameState),
      gameState.currentPlayer = player →
      (strategy gameState ∉ gameState.visitedPositions) →
      (∃ (nextState : GameState), 
        nextState.currentPosition = strategy gameState ∧
        nextState.visitedPositions = insert gameState.currentPosition gameState.visitedPositions ∧
        nextState.currentPlayer ≠ player)

theorem corner_start_winning_strategy :
  ∀ (n : Nat),
    (n % 2 = 0 → HasWinningStrategy Player.A (GameState.mk n (0, 0) {(0, 0)} Player.A)) ∧
    (n % 2 = 1 → HasWinningStrategy Player.B (GameState.mk n (0, 0) {(0, 0)} Player.A)) :=
sorry

theorem adjacent_start_winning_strategy :
  ∀ (n : Nat) (startPos : Nat × Nat),
    (startPos = (0, 1) ∨ startPos = (1, 0)) →
    HasWinningStrategy Player.A (GameState.mk n startPos {startPos} Player.A) :=
sorry

end NUMINAMATH_CALUDE_corner_start_winning_strategy_adjacent_start_winning_strategy_l447_44714


namespace NUMINAMATH_CALUDE_quadratic_equation_unique_solution_l447_44721

/-- The quadratic equation ax^2 - x - 1 = 0 has exactly one solution in the interval (0, 1) if and only if a > 2 -/
theorem quadratic_equation_unique_solution (a : ℝ) : 
  (∃! x : ℝ, 0 < x ∧ x < 1 ∧ a * x^2 - x - 1 = 0) ↔ a > 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_unique_solution_l447_44721


namespace NUMINAMATH_CALUDE_fair_haired_women_percentage_l447_44793

/-- Given a company where:
  * 20% of employees are women with fair hair
  * 50% of employees have fair hair
  Prove that 40% of fair-haired employees are women -/
theorem fair_haired_women_percentage
  (total_employees : ℕ)
  (women_fair_hair_percent : ℚ)
  (fair_hair_percent : ℚ)
  (h1 : women_fair_hair_percent = 20 / 100)
  (h2 : fair_hair_percent = 50 / 100) :
  (women_fair_hair_percent * total_employees) / (fair_hair_percent * total_employees) = 40 / 100 :=
sorry

end NUMINAMATH_CALUDE_fair_haired_women_percentage_l447_44793


namespace NUMINAMATH_CALUDE_moon_speed_conversion_l447_44792

/-- Converts a speed from kilometers per second to kilometers per hour -/
def km_per_second_to_km_per_hour (speed_km_per_second : ℝ) : ℝ :=
  speed_km_per_second * 3600

/-- The moon's speed in kilometers per second -/
def moon_speed_km_per_second : ℝ := 1.02

theorem moon_speed_conversion :
  km_per_second_to_km_per_hour moon_speed_km_per_second = 3672 := by
  sorry

end NUMINAMATH_CALUDE_moon_speed_conversion_l447_44792


namespace NUMINAMATH_CALUDE_science_fair_participants_l447_44708

theorem science_fair_participants (total : ℕ) (j s : ℕ) : 
  total = 240 →
  j + s = total →
  (3 * j) / 4 = s / 2 →
  (3 * j) / 4 + s / 2 = 144 :=
by sorry

end NUMINAMATH_CALUDE_science_fair_participants_l447_44708


namespace NUMINAMATH_CALUDE_quadratic_inequalities_l447_44726

def quadratic_inequality_A (x : ℝ) := x^2 - 12*x + 20 > 0

def quadratic_inequality_B (x : ℝ) := x^2 - 5*x + 6 < 0

def quadratic_inequality_C (x : ℝ) := 9*x^2 - 6*x + 1 > 0

def quadratic_inequality_D (x : ℝ) := -2*x^2 + 2*x - 3 > 0

theorem quadratic_inequalities :
  (∀ x, quadratic_inequality_A x ↔ (x < 2 ∨ x > 10)) ∧
  (∀ x, quadratic_inequality_B x ↔ (2 < x ∧ x < 3)) ∧
  (∃ x, ¬quadratic_inequality_C x) ∧
  (∀ x, ¬quadratic_inequality_D x) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequalities_l447_44726


namespace NUMINAMATH_CALUDE_empty_solution_implies_a_leq_5_l447_44782

theorem empty_solution_implies_a_leq_5 (a : ℝ) :
  (∀ x : ℝ, ¬(|x - 2| + |x + 3| < a)) → a ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_empty_solution_implies_a_leq_5_l447_44782


namespace NUMINAMATH_CALUDE_correct_product_is_341_l447_44767

/-- Reverses the digits of a two-digit number -/
def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

/-- Theorem: Under the given conditions, the correct product is 341 -/
theorem correct_product_is_341 
  (c : ℕ) 
  (d : ℕ) 
  (h1 : c ≥ 10 ∧ c < 100)  -- c is a two-digit number
  (h2 : (reverse_digits c) * d = 143) :
  c * d = 341 := by
  sorry

end NUMINAMATH_CALUDE_correct_product_is_341_l447_44767


namespace NUMINAMATH_CALUDE_cubic_expression_equality_l447_44786

theorem cubic_expression_equality (x y : ℝ) (hx : x = 3) (hy : y = 4) :
  (x^3 + 3*y^3) / 9 = 73/3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expression_equality_l447_44786


namespace NUMINAMATH_CALUDE_swallowing_not_complete_disappearance_l447_44719

/-- Represents a snake in the swallowing process --/
structure Snake where
  length : ℝ
  swallowed : ℝ

/-- Represents the state of two snakes swallowing each other --/
structure SwallowingState where
  snake1 : Snake
  snake2 : Snake
  ring_size : ℝ

/-- The swallowing process between two snakes --/
def swallowing_process (initial_state : SwallowingState) : Prop :=
  ∀ t : ℝ, t ≥ 0 →
    ∃ state : SwallowingState,
      state.ring_size < initial_state.ring_size ∧
      state.snake1.swallowed > initial_state.snake1.swallowed ∧
      state.snake2.swallowed > initial_state.snake2.swallowed ∧
      state.snake1.length + state.snake2.length > 0

/-- Theorem stating that the swallowing process does not result in complete disappearance --/
theorem swallowing_not_complete_disappearance (initial_state : SwallowingState) :
  swallowing_process initial_state →
  ∃ final_state : SwallowingState, final_state.snake1.length + final_state.snake2.length > 0 :=
by sorry

end NUMINAMATH_CALUDE_swallowing_not_complete_disappearance_l447_44719


namespace NUMINAMATH_CALUDE_trig_problem_l447_44736

theorem trig_problem (θ : ℝ) 
  (h : (2 * Real.cos ((3/2) * Real.pi + θ) + Real.cos (Real.pi + θ)) / 
       (3 * Real.sin (Real.pi - θ) + 2 * Real.sin ((5/2) * Real.pi + θ)) = 1/5) : 
  Real.tan θ = 1 ∧ Real.sin θ^2 + 3 * Real.sin θ * Real.cos θ = 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_problem_l447_44736


namespace NUMINAMATH_CALUDE_complex_division_example_l447_44700

/-- The imaginary unit -/
def i : ℂ := Complex.I

/-- Theorem stating that the complex number 2i/(1+i) equals 1+i -/
theorem complex_division_example : (2 * i) / (1 + i) = 1 + i := by
  sorry

end NUMINAMATH_CALUDE_complex_division_example_l447_44700


namespace NUMINAMATH_CALUDE_divisors_sum_product_l447_44731

theorem divisors_sum_product (n a b : ℕ) : 
  n ≥ 1 → 
  a > 0 → 
  b > 0 → 
  n % a = 0 → 
  n % b = 0 → 
  a + b + a * b = n → 
  a = b := by
sorry

end NUMINAMATH_CALUDE_divisors_sum_product_l447_44731


namespace NUMINAMATH_CALUDE_otimes_properties_l447_44762

-- Define the custom operation
def otimes (a b : ℝ) : ℝ := a * (1 - b)

-- Theorem statement
theorem otimes_properties : 
  (otimes 2 (-2) = 6) ∧
  (¬ ∀ a b : ℝ, otimes a b = otimes b a) ∧
  (∀ a : ℝ, otimes 5 a + otimes 6 a = otimes 11 a) ∧
  (¬ ∀ b : ℝ, otimes 3 b = 3 → b = 1) :=
sorry

end NUMINAMATH_CALUDE_otimes_properties_l447_44762


namespace NUMINAMATH_CALUDE_not_A_necessary_not_sufficient_for_not_B_l447_44768

-- Define propositions A and B
variable (A B : Prop)

-- Define what it means for A to be sufficient but not necessary for B
def sufficient_not_necessary (A B : Prop) : Prop :=
  (A → B) ∧ ¬(B → A)

-- Theorem statement
theorem not_A_necessary_not_sufficient_for_not_B
  (h : sufficient_not_necessary A B) :
  (¬B → ¬A) ∧ ¬(¬A → ¬B) := by
  sorry

end NUMINAMATH_CALUDE_not_A_necessary_not_sufficient_for_not_B_l447_44768


namespace NUMINAMATH_CALUDE_greatest_whole_number_inequality_l447_44794

theorem greatest_whole_number_inequality :
  ∀ x : ℤ, (5 * x - 4 < 3 - 2 * x) → x ≤ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_whole_number_inequality_l447_44794


namespace NUMINAMATH_CALUDE_age_difference_l447_44702

theorem age_difference (sachin_age rahul_age : ℝ) : 
  sachin_age = 24.5 → 
  sachin_age / rahul_age = 7 / 9 → 
  rahul_age - sachin_age = 7 := by
sorry

end NUMINAMATH_CALUDE_age_difference_l447_44702


namespace NUMINAMATH_CALUDE_eighth_of_two_power_44_l447_44796

theorem eighth_of_two_power_44 (x : ℤ) :
  (2^44 : ℚ) / 8 = 2^x → x = 41 := by
  sorry

end NUMINAMATH_CALUDE_eighth_of_two_power_44_l447_44796


namespace NUMINAMATH_CALUDE_largest_four_digit_with_product_72_l447_44750

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digit_product (n : ℕ) : ℕ :=
  (n / 1000) * ((n / 100) % 10) * ((n / 10) % 10) * (n % 10)

def digit_sum (n : ℕ) : ℕ :=
  (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

theorem largest_four_digit_with_product_72 :
  ∃ M : ℕ, is_four_digit M ∧ 
    digit_product M = 72 ∧
    (∀ n : ℕ, is_four_digit n → digit_product n = 72 → n ≤ M) ∧
    digit_sum M = 17 := by
  sorry

end NUMINAMATH_CALUDE_largest_four_digit_with_product_72_l447_44750


namespace NUMINAMATH_CALUDE_inverse_functions_same_monotonicity_function_symmetry_origin_exists_odd_function_without_inverse_l447_44787

-- Define a type for real-valued functions
def RealFunction := ℝ → ℝ

-- Define inverse functions
def IsInverse (f g : RealFunction) : Prop :=
  ∀ x, g (f x) = x ∧ f (g x) = x

-- Define monotonicity
def IsMonotoneIncreasing (f : RealFunction) : Prop :=
  ∀ x y, x < y → f x < f y

def IsMonotoneDecreasing (f : RealFunction) : Prop :=
  ∀ x y, x < y → f x > f y

-- Define odd function
def IsOdd (f : RealFunction) : Prop :=
  ∀ x, f (-x) = -f x

-- Define symmetry with respect to the origin
def IsSymmetricToOrigin (f g : RealFunction) : Prop :=
  ∀ x, g x = -f (-x)

-- Theorem 1: Inverse functions have the same monotonicity
theorem inverse_functions_same_monotonicity (f g : RealFunction) 
  (h : IsInverse f g) : 
  (IsMonotoneIncreasing f ↔ IsMonotoneIncreasing g) ∧ 
  (IsMonotoneDecreasing f ↔ IsMonotoneDecreasing g) := by
  sorry

-- Theorem 2: Function symmetry with respect to the origin
theorem function_symmetry_origin (f : RealFunction) :
  IsSymmetricToOrigin f (λ x => -f (-x)) := by
  sorry

-- Theorem 3: Existence of an odd function without an inverse
theorem exists_odd_function_without_inverse :
  ∃ f : RealFunction, IsOdd f ∧ ¬(∃ g : RealFunction, IsInverse f g) := by
  sorry

end NUMINAMATH_CALUDE_inverse_functions_same_monotonicity_function_symmetry_origin_exists_odd_function_without_inverse_l447_44787


namespace NUMINAMATH_CALUDE_last_three_digits_of_5_to_1999_l447_44723

theorem last_three_digits_of_5_to_1999 : 5^1999 ≡ 125 [ZMOD 1000] := by
  sorry

end NUMINAMATH_CALUDE_last_three_digits_of_5_to_1999_l447_44723


namespace NUMINAMATH_CALUDE_fermat_prime_l447_44729

theorem fermat_prime (n : ℕ) (p : ℕ) (h1 : p = 2^n + 1) 
  (h2 : (3^((p-1)/2) + 1) % p = 0) : Nat.Prime p := by
  sorry

end NUMINAMATH_CALUDE_fermat_prime_l447_44729


namespace NUMINAMATH_CALUDE_sin_cos_value_l447_44763

theorem sin_cos_value (α : Real) (h : 1 / Real.sin α + 1 / Real.cos α = Real.sqrt 3) :
  Real.sin α * Real.cos α = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_value_l447_44763


namespace NUMINAMATH_CALUDE_cameron_tour_theorem_l447_44705

def cameron_tour_problem (questions_per_tourist : ℕ) (total_tours : ℕ) 
  (group1_size : ℕ) (group2_size : ℕ) (group3_size : ℕ) 
  (inquisitive_factor : ℕ) (total_questions : ℕ) : Prop :=
  let group1_questions := group1_size * questions_per_tourist
  let group2_questions := group2_size * questions_per_tourist
  let group3_questions := group3_size * questions_per_tourist + 
                          (inquisitive_factor - 1) * questions_per_tourist
  let remaining_questions := total_questions - (group1_questions + group2_questions + group3_questions)
  let last_group_size := remaining_questions / questions_per_tourist
  last_group_size = 7

theorem cameron_tour_theorem : 
  cameron_tour_problem 2 4 6 11 8 3 68 := by
  sorry

end NUMINAMATH_CALUDE_cameron_tour_theorem_l447_44705


namespace NUMINAMATH_CALUDE_length_ad_is_12_95_l447_44706

/-- A quadrilateral ABCD with specific properties -/
structure Quadrilateral where
  /-- Length of side AB -/
  ab : ℝ
  /-- Length of side BC -/
  bc : ℝ
  /-- Length of side CD -/
  cd : ℝ
  /-- Angle B in radians -/
  angle_b : ℝ
  /-- Angle C in radians -/
  angle_c : ℝ
  /-- Condition: AB = 6 -/
  hab : ab = 6
  /-- Condition: BC = 8 -/
  hbc : bc = 8
  /-- Condition: CD = 15 -/
  hcd : cd = 15
  /-- Condition: Angle B is obtuse -/
  hb_obtuse : π / 2 < angle_b ∧ angle_b < π
  /-- Condition: Angle C is obtuse -/
  hc_obtuse : π / 2 < angle_c ∧ angle_c < π
  /-- Condition: sin C = 4/5 -/
  hsin_c : Real.sin angle_c = 4/5
  /-- Condition: cos B = -4/5 -/
  hcos_b : Real.cos angle_b = -4/5

/-- The length of side AD in the quadrilateral ABCD -/
def lengthAD (q : Quadrilateral) : ℝ := sorry

/-- Theorem stating that the length of side AD is 12.95 -/
theorem length_ad_is_12_95 (q : Quadrilateral) : lengthAD q = 12.95 := by
  sorry

end NUMINAMATH_CALUDE_length_ad_is_12_95_l447_44706


namespace NUMINAMATH_CALUDE_chord_intersects_diameter_l447_44755

/-- In a circle with radius 6, a chord of length 10 intersects a diameter,
    dividing it into segments of lengths 6 - √11 and 6 + √11 -/
theorem chord_intersects_diameter (r : ℝ) (chord_length : ℝ) 
  (h1 : r = 6) (h2 : chord_length = 10) : 
  ∃ (s1 s2 : ℝ), s1 = 6 - Real.sqrt 11 ∧ s2 = 6 + Real.sqrt 11 ∧ s1 + s2 = 2 * r :=
sorry

end NUMINAMATH_CALUDE_chord_intersects_diameter_l447_44755


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l447_44771

/-- 
Given an arithmetic sequence {a_n} with first term a₁ = 19 and integer common difference d,
if the 6th term is negative and the 5th term is non-negative, then the common difference is -4.
-/
theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℤ) 
  (d : ℤ) 
  (h1 : a 1 = 19)
  (h2 : ∀ n : ℕ, a (n + 1) = a n + d)
  (h3 : a 6 < 0)
  (h4 : a 5 ≥ 0) :
  d = -4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l447_44771


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l447_44724

/-- A geometric sequence is a sequence where each term after the first is found by 
    multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_ratio 
  (a : ℕ → ℝ) (q : ℝ) (h : IsGeometricSequence a q) 
  (h_eq : 16 * a 6 = a 2) :
  q = 1/2 ∨ q = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l447_44724


namespace NUMINAMATH_CALUDE_probability_is_zero_l447_44709

/-- Represents a runner on a circular track -/
structure Runner where
  lapTime : ℝ
  direction : Bool  -- True for counterclockwise, False for clockwise

/-- Represents the picture taken of the track -/
structure Picture where
  coverageFraction : ℝ
  centerPosition : ℝ  -- Position on track (0 ≤ position < 1)

/-- Calculates the probability of both runners being in the picture -/
def probabilityBothInPicture (rachel : Runner) (robert : Runner) (pic : Picture) (timeElapsed : ℝ) : ℝ :=
  sorry

/-- Theorem stating the probability is zero for the given conditions -/
theorem probability_is_zero :
  ∀ (rachel : Runner) (robert : Runner) (pic : Picture) (t : ℝ),
    rachel.lapTime = 120 →
    robert.lapTime = 75 →
    rachel.direction = true →
    robert.direction = false →
    pic.coverageFraction = 1/3 →
    pic.centerPosition = 0 →
    15 * 60 ≤ t ∧ t < 16 * 60 →
    probabilityBothInPicture rachel robert pic t = 0 :=
  sorry

end NUMINAMATH_CALUDE_probability_is_zero_l447_44709


namespace NUMINAMATH_CALUDE_paperback_ratio_l447_44753

/-- Represents the number and types of books Thabo owns -/
structure BookCollection where
  total : ℕ
  paperback_fiction : ℕ
  paperback_nonfiction : ℕ
  hardcover_nonfiction : ℕ

/-- The properties of Thabo's book collection -/
def thabos_books : BookCollection where
  total := 220
  paperback_fiction := 120
  paperback_nonfiction := 60
  hardcover_nonfiction := 40

/-- Theorem stating the ratio of paperback fiction to paperback nonfiction books -/
theorem paperback_ratio (b : BookCollection) 
  (h1 : b.total = 220)
  (h2 : b.paperback_fiction + b.paperback_nonfiction + b.hardcover_nonfiction = b.total)
  (h3 : b.paperback_nonfiction = b.hardcover_nonfiction + 20)
  (h4 : b.hardcover_nonfiction = 40) :
  b.paperback_fiction / b.paperback_nonfiction = 2 := by
  sorry

#check paperback_ratio thabos_books

end NUMINAMATH_CALUDE_paperback_ratio_l447_44753


namespace NUMINAMATH_CALUDE_triangle_area_l447_44781

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x) ^ 2 + Real.sqrt 3 * Real.sin (2 * x)

theorem triangle_area (A B C : ℝ) (a b c : ℝ) :
  (0 < A) ∧ (A < π) ∧
  (0 < B) ∧ (B < π) ∧
  (0 < C) ∧ (C < π) ∧
  (A + B + C = π) ∧
  (f A = 2) ∧
  (a = Real.sqrt 7) ∧
  (Real.sin B = 2 * Real.sin C) ∧
  (a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A) →
  (1/2 * a * b * Real.sin C) = 7 * Real.sqrt 3 / 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l447_44781


namespace NUMINAMATH_CALUDE_hoseok_number_problem_l447_44798

theorem hoseok_number_problem (x : ℝ) : 15 * x = 45 → x - 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_hoseok_number_problem_l447_44798


namespace NUMINAMATH_CALUDE_product_equivalence_l447_44711

theorem product_equivalence : 
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * (4^16 + 5^16) * 
  (4^32 + 5^32) * (4^64 + 5^64) * (4^128 + 5^128) = 5^256 - 4^256 := by
  sorry

end NUMINAMATH_CALUDE_product_equivalence_l447_44711


namespace NUMINAMATH_CALUDE_hyperbola_ellipse_shared_foci_l447_44779

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 5 = 1

-- Define the ellipse
def ellipse (a x y : ℝ) : Prop := x^2 / a^2 + y^2 / 16 = 1

-- Define the condition that a > 0
def a_positive (a : ℝ) : Prop := a > 0

-- Define the condition that the hyperbola and ellipse share the same foci
def same_foci (a : ℝ) : Prop := ∃ c : ℝ, c^2 = 9 ∧ 
  (∀ x y : ℝ, hyperbola x y ↔ x^2 / 4 - y^2 / 5 = 1) ∧
  (∀ x y : ℝ, ellipse a x y ↔ x^2 / a^2 + y^2 / 16 = 1)

-- Theorem statement
theorem hyperbola_ellipse_shared_foci (a : ℝ) :
  a_positive a → same_foci a → a = 5 := by sorry

end NUMINAMATH_CALUDE_hyperbola_ellipse_shared_foci_l447_44779


namespace NUMINAMATH_CALUDE_nested_square_root_simplification_l447_44772

theorem nested_square_root_simplification :
  Real.sqrt (25 * Real.sqrt (25 * Real.sqrt 25)) = 5 * Real.sqrt (5 * Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_nested_square_root_simplification_l447_44772


namespace NUMINAMATH_CALUDE_conference_tables_needed_l447_44749

-- Define the base 7 number
def base7_number : ℕ := 312

-- Define the base conversion function
def base7_to_decimal (n : ℕ) : ℕ :=
  3 * 7^2 + 1 * 7^1 + 2 * 7^0

-- Define the number of attendees per table
def attendees_per_table : ℕ := 3

-- Theorem statement
theorem conference_tables_needed :
  (base7_to_decimal base7_number) / attendees_per_table = 52 := by
  sorry

end NUMINAMATH_CALUDE_conference_tables_needed_l447_44749


namespace NUMINAMATH_CALUDE_seashell_collection_l447_44774

theorem seashell_collection (stefan vail aiguo fatima : ℕ) : 
  stefan = vail + 16 →
  vail + 5 = aiguo →
  aiguo = 20 →
  fatima = 2 * aiguo →
  stefan + vail + aiguo + fatima = 106 := by
sorry

end NUMINAMATH_CALUDE_seashell_collection_l447_44774


namespace NUMINAMATH_CALUDE_simplify_expression_l447_44742

theorem simplify_expression (x : ℝ) :
  (2*x - 1)^2 - (3*x + 1)*(3*x - 1) + 5*x*(x - 1) = -9*x + 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l447_44742


namespace NUMINAMATH_CALUDE_max_common_ratio_arithmetic_geometric_l447_44789

theorem max_common_ratio_arithmetic_geometric (a : ℕ → ℝ) (d q : ℝ) (k : ℕ) :
  (∀ n, a (n + 1) - a n = d) →  -- arithmetic sequence condition
  d ≠ 0 →  -- non-zero common difference
  k ≥ 2 →  -- k condition
  a k / a 1 = q →  -- geometric sequence condition for a_1 and a_k
  a (2 * k) / a k = q →  -- geometric sequence condition for a_k and a_2k
  q ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_max_common_ratio_arithmetic_geometric_l447_44789


namespace NUMINAMATH_CALUDE_no_positive_integer_sequence_exists_positive_irrational_sequence_l447_44710

/-- Part 1: Non-existence of positive integer sequence --/
theorem no_positive_integer_sequence :
  ¬ ∃ f : ℕ+ → ℕ+, ∀ n : ℕ+, (f (n + 1))^2 ≥ 2 * (f n) * (f (n + 2)) :=
sorry

/-- Part 2: Existence of positive irrational sequence --/
theorem exists_positive_irrational_sequence :
  ∃ f : ℕ+ → ℝ, (∀ n : ℕ+, Irrational (f n)) ∧
    (∀ n : ℕ+, f n > 0) ∧
    (∀ n : ℕ+, (f (n + 1))^2 ≥ 2 * (f n) * (f (n + 2))) :=
sorry

end NUMINAMATH_CALUDE_no_positive_integer_sequence_exists_positive_irrational_sequence_l447_44710


namespace NUMINAMATH_CALUDE_complex_number_problem_l447_44743

def complex_i : ℂ := Complex.I

theorem complex_number_problem (z₁ z₂ : ℂ) 
  (h1 : (z₁ - 2) * (1 + complex_i) = 1 - complex_i)
  (h2 : z₂.im = 2)
  (h3 : (z₁ * z₂).im = 0) :
  z₁ = 2 - complex_i ∧ z₂ = 4 + 2 * complex_i := by
  sorry

end NUMINAMATH_CALUDE_complex_number_problem_l447_44743


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_least_subtraction_62575_99_l447_44733

theorem least_subtraction_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (k : ℕ), k < d ∧ (n - k) % d = 0 ∧ ∀ (m : ℕ), m < k → (n - m) % d ≠ 0 :=
by
  sorry

theorem least_subtraction_62575_99 :
  ∃ (k : ℕ), k < 99 ∧ (62575 - k) % 99 = 0 ∧ ∀ (m : ℕ), m < k → (62575 - m) % 99 ≠ 0 ∧ k = 43 :=
by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_least_subtraction_62575_99_l447_44733


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_l447_44751

theorem systematic_sampling_interval
  (population_size : ℕ)
  (sample_size : ℕ)
  (h1 : population_size = 800)
  (h2 : sample_size = 40)
  : population_size / sample_size = 20 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_interval_l447_44751


namespace NUMINAMATH_CALUDE_inequality_proof_l447_44740

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  (a^2014 / (1 + 2*b*c)) + (b^2014 / (1 + 2*a*c)) + (c^2014 / (1 + 2*a*b)) ≥ 3 / (a*b + b*c + c*a) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l447_44740


namespace NUMINAMATH_CALUDE_jack_marbles_l447_44754

/-- Calculates the final number of marbles Jack has after sharing and finding more -/
def final_marbles (initial : ℕ) (shared : ℕ) (multiplier : ℕ) : ℕ :=
  let remaining := initial - shared
  let found := remaining * multiplier
  remaining + found

/-- Theorem stating that Jack ends up with 232 marbles -/
theorem jack_marbles :
  final_marbles 62 33 7 = 232 := by
  sorry

end NUMINAMATH_CALUDE_jack_marbles_l447_44754


namespace NUMINAMATH_CALUDE_equal_area_division_l447_44717

/-- A point on a grid --/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- A figure on a grid --/
structure GridFigure where
  area : ℚ
  points : Set GridPoint

/-- A ray on a grid --/
structure GridRay where
  start : GridPoint
  direction : GridPoint

/-- Theorem: There exists a ray that divides a figure of area 9 into two equal parts --/
theorem equal_area_division (fig : GridFigure) (A : GridPoint) :
  fig.area = 9 →
  ∃ (B : GridPoint) (ray : GridRay),
    B ≠ A ∧
    ray.start = A ∧
    (∃ (t : ℚ), ray.start.x + t * ray.direction.x = B.x ∧ ray.start.y + t * ray.direction.y = B.y) ∧
    ∃ (left_area right_area : ℚ),
      left_area = right_area ∧
      left_area + right_area = fig.area := by
  sorry

end NUMINAMATH_CALUDE_equal_area_division_l447_44717


namespace NUMINAMATH_CALUDE_one_percent_of_x_l447_44746

theorem one_percent_of_x (x : ℝ) (h : (89 / 100) * 19 = (19 / 100) * x) : 
  (1 / 100) * x = 89 / 100 := by
sorry

end NUMINAMATH_CALUDE_one_percent_of_x_l447_44746


namespace NUMINAMATH_CALUDE_housing_boom_proof_l447_44747

/-- The number of houses in Lawrence County before the housing boom -/
def houses_before : ℕ := 1426

/-- The number of houses in Lawrence County after the housing boom -/
def houses_after : ℕ := 2000

/-- The number of houses built during the housing boom -/
def houses_built : ℕ := houses_after - houses_before

theorem housing_boom_proof : houses_built = 574 := by
  sorry

end NUMINAMATH_CALUDE_housing_boom_proof_l447_44747


namespace NUMINAMATH_CALUDE_composition_of_linear_functions_l447_44769

theorem composition_of_linear_functions (a b : ℝ) : 
  (∀ x : ℝ, (3 * (a * x + b) - 4) = 4 * x + 2) → 
  a + b = 10/3 := by
  sorry

end NUMINAMATH_CALUDE_composition_of_linear_functions_l447_44769


namespace NUMINAMATH_CALUDE_double_shot_espresso_price_l447_44780

/-- Represents the cost of a coffee order -/
structure CoffeeOrder where
  drip_coffee : ℕ
  drip_coffee_price : ℚ
  latte : ℕ
  latte_price : ℚ
  vanilla_syrup : ℕ
  vanilla_syrup_price : ℚ
  cold_brew : ℕ
  cold_brew_price : ℚ
  cappuccino : ℕ
  cappuccino_price : ℚ
  double_shot_espresso : ℕ
  total_price : ℚ

/-- Calculates the cost of the double shot espresso -/
def double_shot_espresso_cost (order : CoffeeOrder) : ℚ :=
  order.total_price -
  (order.drip_coffee * order.drip_coffee_price +
   order.latte * order.latte_price +
   order.vanilla_syrup * order.vanilla_syrup_price +
   order.cold_brew * order.cold_brew_price +
   order.cappuccino * order.cappuccino_price)

/-- Theorem stating that the double shot espresso costs $3.50 -/
theorem double_shot_espresso_price (order : CoffeeOrder) 
  (h1 : order.drip_coffee = 2)
  (h2 : order.drip_coffee_price = 2.25)
  (h3 : order.latte = 2)
  (h4 : order.latte_price = 4)
  (h5 : order.vanilla_syrup = 1)
  (h6 : order.vanilla_syrup_price = 0.5)
  (h7 : order.cold_brew = 2)
  (h8 : order.cold_brew_price = 2.5)
  (h9 : order.cappuccino = 1)
  (h10 : order.cappuccino_price = 3.5)
  (h11 : order.double_shot_espresso = 1)
  (h12 : order.total_price = 25) :
  double_shot_espresso_cost order = 3.5 := by
  sorry


end NUMINAMATH_CALUDE_double_shot_espresso_price_l447_44780


namespace NUMINAMATH_CALUDE_negative_cube_squared_l447_44777

theorem negative_cube_squared (a b : ℝ) : (-a^3 * b)^2 = a^6 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_negative_cube_squared_l447_44777


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l447_44735

/-- An isosceles triangle with two sides of length 3 and 7 has a perimeter of 17 -/
theorem isosceles_triangle_perimeter : ∀ a b c : ℝ,
  a > 0 ∧ b > 0 ∧ c > 0 →
  (a = b ∧ (a = 3 ∧ c = 7 ∨ a = 7 ∧ c = 3)) →
  a + b + c = 17 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l447_44735


namespace NUMINAMATH_CALUDE_number_problem_l447_44728

theorem number_problem : ∃ x : ℚ, x - (3/5) * x = 56 ∧ x = 140 := by sorry

end NUMINAMATH_CALUDE_number_problem_l447_44728


namespace NUMINAMATH_CALUDE_total_pencils_l447_44713

/-- Given the number of pencils in different locations, prove the total number of pencils. -/
theorem total_pencils (drawer : ℕ) (desk_initial : ℕ) (desk_added : ℕ) :
  drawer = 43 →
  desk_initial = 19 →
  desk_added = 16 →
  drawer + desk_initial + desk_added = 78 :=
by sorry

end NUMINAMATH_CALUDE_total_pencils_l447_44713


namespace NUMINAMATH_CALUDE_choir_members_count_l447_44788

theorem choir_members_count :
  ∃ n : ℕ,
    (n + 4) % 10 = 0 ∧
    (n + 5) % 11 = 0 ∧
    200 < n ∧
    n < 300 ∧
    n = 226 := by
  sorry

end NUMINAMATH_CALUDE_choir_members_count_l447_44788


namespace NUMINAMATH_CALUDE_find_constant_b_l447_44720

theorem find_constant_b (d e : ℚ) :
  (∀ x : ℚ, (7 * x^2 - 2 * x + 4/3) * (d * x^2 + b * x + e) = 28 * x^4 - 10 * x^3 + 18 * x^2 - 8 * x + 5/3) →
  b = -2/7 := by
sorry

end NUMINAMATH_CALUDE_find_constant_b_l447_44720


namespace NUMINAMATH_CALUDE_a_equals_one_sufficient_not_necessary_l447_44716

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 4*a*x + 3

-- Define what it means for a function to be increasing on an interval
def is_increasing_on (f : ℝ → ℝ) (l r : ℝ) : Prop :=
  ∀ x y, l ≤ x → x < y → y ≤ r → f x < f y

-- State the theorem
theorem a_equals_one_sufficient_not_necessary :
  (∀ x y, 2 ≤ x → x < y → is_increasing_on (f 1) 2 y) ∧
  ¬(∀ a : ℝ, (∀ x y, 2 ≤ x → x < y → is_increasing_on (f a) 2 y) → a = 1) :=
sorry

end NUMINAMATH_CALUDE_a_equals_one_sufficient_not_necessary_l447_44716


namespace NUMINAMATH_CALUDE_inverse_multiplication_l447_44776

theorem inverse_multiplication (a : ℝ) (h : a ≠ 0) : a * a⁻¹ = 1 := by
  sorry

end NUMINAMATH_CALUDE_inverse_multiplication_l447_44776


namespace NUMINAMATH_CALUDE_lars_daily_bread_production_l447_44715

-- Define the baking rates and working hours
def loaves_per_hour : ℕ := 10
def baguettes_per_two_hours : ℕ := 30
def hours_per_day : ℕ := 6

-- Define the function to calculate total breads
def total_breads : ℕ :=
  (loaves_per_hour * hours_per_day) + 
  (baguettes_per_two_hours * (hours_per_day / 2))

-- Theorem statement
theorem lars_daily_bread_production :
  total_breads = 150 := by
  sorry

end NUMINAMATH_CALUDE_lars_daily_bread_production_l447_44715


namespace NUMINAMATH_CALUDE_marble_jar_ratio_l447_44795

/-- Given three jars of marbles with specific conditions, prove the ratio of marbles in Jar C to Jar B -/
theorem marble_jar_ratio :
  let jar_a : ℕ := 28
  let jar_b : ℕ := jar_a + 12
  let total : ℕ := 148
  let jar_c : ℕ := total - (jar_a + jar_b)
  (jar_c : ℚ) / jar_b = 2 / 1 :=
by
  sorry

end NUMINAMATH_CALUDE_marble_jar_ratio_l447_44795


namespace NUMINAMATH_CALUDE_standard_deviation_is_eight_l447_44756

/-- Represents the age distribution of job applicants -/
structure AgeDistribution where
  average_age : ℕ
  num_different_ages : ℕ
  standard_deviation : ℕ

/-- Checks if the age distribution satisfies the given conditions -/
def is_valid_distribution (d : AgeDistribution) : Prop :=
  d.average_age = 30 ∧
  d.num_different_ages = 17 ∧
  d.num_different_ages = 2 * d.standard_deviation + 1

/-- Theorem stating that the standard deviation must be 8 given the conditions -/
theorem standard_deviation_is_eight (d : AgeDistribution) 
  (h : is_valid_distribution d) : d.standard_deviation = 8 := by
  sorry

#check standard_deviation_is_eight

end NUMINAMATH_CALUDE_standard_deviation_is_eight_l447_44756


namespace NUMINAMATH_CALUDE_circles_externally_tangent_l447_44790

theorem circles_externally_tangent (x y : ℝ) : 
  let circle1 := {(x, y) : ℝ × ℝ | x^2 + y^2 - 6*x = 0}
  let circle2 := {(x, y) : ℝ × ℝ | x^2 + y^2 + 8*y + 12 = 0}
  let center1 := (3, 0)
  let center2 := (0, -4)
  let radius1 := 3
  let radius2 := 2
  (∀ (p : ℝ × ℝ), p ∈ circle1 ↔ (p.1 - center1.1)^2 + (p.2 - center1.2)^2 = radius1^2) ∧
  (∀ (p : ℝ × ℝ), p ∈ circle2 ↔ (p.1 - center2.1)^2 + (p.2 - center2.2)^2 = radius2^2) ∧
  (center1.1 - center2.1)^2 + (center1.2 - center2.2)^2 = (radius1 + radius2)^2 :=
by
  sorry

end NUMINAMATH_CALUDE_circles_externally_tangent_l447_44790


namespace NUMINAMATH_CALUDE_shirt_price_ratio_l447_44744

theorem shirt_price_ratio (marked_price : ℝ) (h1 : marked_price > 0) : 
  let discount_rate : ℝ := 2 / 5
  let selling_price : ℝ := marked_price * (1 - discount_rate)
  let cost_price : ℝ := selling_price * (4 / 5)
  cost_price / marked_price = 12 / 25 := by
sorry

end NUMINAMATH_CALUDE_shirt_price_ratio_l447_44744


namespace NUMINAMATH_CALUDE_complex_equation_solution_l447_44759

theorem complex_equation_solution :
  ∃ x : ℂ, (3 : ℂ) + 2 * Complex.I * x = 4 - 5 * Complex.I * x ∧ x = -Complex.I / 7 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l447_44759


namespace NUMINAMATH_CALUDE_expand_product_l447_44725

theorem expand_product (x : ℝ) : (x + 3) * (x + 4) = x^2 + 7*x + 12 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l447_44725


namespace NUMINAMATH_CALUDE_legs_on_ground_l447_44757

theorem legs_on_ground (num_horses : ℕ) (num_men : ℕ) (num_riding : ℕ) : 
  num_horses = 8 →
  num_men = num_horses →
  num_riding = num_men / 2 →
  (4 * num_horses + 2 * (num_men - num_riding)) = 40 :=
by sorry

end NUMINAMATH_CALUDE_legs_on_ground_l447_44757


namespace NUMINAMATH_CALUDE_jeff_sunday_morning_laps_l447_44764

/-- The number of laps Jeff swam on Sunday morning before the break -/
def sunday_morning_laps (total_laps required_laps saturday_laps remaining_laps : ℕ) : ℕ :=
  total_laps - saturday_laps - remaining_laps

theorem jeff_sunday_morning_laps :
  sunday_morning_laps 98 27 56 = 15 := by
  sorry

end NUMINAMATH_CALUDE_jeff_sunday_morning_laps_l447_44764


namespace NUMINAMATH_CALUDE_sandy_shopping_money_l447_44760

theorem sandy_shopping_money (initial_amount : ℝ) (spent_percentage : ℝ) (remaining_amount : ℝ) : 
  spent_percentage = 30 →
  remaining_amount = 140 →
  (1 - spent_percentage / 100) * initial_amount = remaining_amount →
  initial_amount = 200 := by
  sorry

end NUMINAMATH_CALUDE_sandy_shopping_money_l447_44760


namespace NUMINAMATH_CALUDE_product_expansion_l447_44722

theorem product_expansion (y : ℝ) (h : y ≠ 0) :
  (3 / 7) * ((7 / y) + 14 * y^3) = 3 / y + 6 * y^3 := by
  sorry

end NUMINAMATH_CALUDE_product_expansion_l447_44722


namespace NUMINAMATH_CALUDE_least_valid_number_l447_44791

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 100 ∧ (n % 100 = n / 10)

theorem least_valid_number : 
  (∃ (n : ℕ), is_valid_number n) ∧ 
  (∀ (m : ℕ), is_valid_number m → m ≥ 900) :=
sorry

end NUMINAMATH_CALUDE_least_valid_number_l447_44791


namespace NUMINAMATH_CALUDE_expand_expression_l447_44741

theorem expand_expression (x y : ℝ) : 25 * (3 * x + 6 - 4 * y) = 75 * x + 150 - 100 * y := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l447_44741


namespace NUMINAMATH_CALUDE_amc_12_scoring_problem_l447_44783

/-- The minimum number of correctly solved problems to achieve the target score -/
def min_correct_problems (total_problems : ℕ) (attempted_problems : ℕ) (points_correct : ℕ) 
  (points_unanswered : ℕ) (target_score : ℕ) : ℕ :=
  let unanswered := total_problems - attempted_problems
  let points_from_unanswered := unanswered * points_unanswered
  let required_points := target_score - points_from_unanswered
  (required_points + points_correct - 1) / points_correct

theorem amc_12_scoring_problem :
  min_correct_problems 30 25 7 2 120 = 16 := by
  sorry

end NUMINAMATH_CALUDE_amc_12_scoring_problem_l447_44783


namespace NUMINAMATH_CALUDE_four_color_theorem_l447_44718

/-- Represents a map as a planar graph -/
structure Map where
  vertices : Set Nat
  edges : Set (Nat × Nat)
  is_planar : Bool

/-- A coloring of a map -/
def Coloring (m : Map) := Nat → Fin 4

/-- Checks if a coloring is valid for a given map -/
def is_valid_coloring (m : Map) (c : Coloring m) : Prop :=
  ∀ (v₁ v₂ : Nat), (v₁, v₂) ∈ m.edges → c v₁ ≠ c v₂

/-- The Four Color Theorem -/
theorem four_color_theorem (m : Map) (h : m.is_planar = true) :
  ∃ (c : Coloring m), is_valid_coloring m c :=
sorry

end NUMINAMATH_CALUDE_four_color_theorem_l447_44718


namespace NUMINAMATH_CALUDE_converse_not_always_true_l447_44784

-- Define the types for points, lines, and planes in space
variable (Point Line Plane : Type)

-- Define the relations
variable (contains : Plane → Line → Prop)  -- plane contains line
variable (perp : Line → Plane → Prop)      -- line perpendicular to plane
variable (perp_planes : Plane → Plane → Prop)  -- plane perpendicular to plane

-- State the theorem
theorem converse_not_always_true 
  (b : Line) (α β : Plane) : 
  ¬(∀ b α β, (contains α b ∧ perp b β → perp_planes α β) → 
             (perp_planes α β → contains α b ∧ perp b β)) :=
sorry

end NUMINAMATH_CALUDE_converse_not_always_true_l447_44784


namespace NUMINAMATH_CALUDE_brandon_lost_skittles_l447_44727

theorem brandon_lost_skittles (initial : ℕ) (final : ℕ) (lost : ℕ) : 
  initial = 96 → final = 87 → initial = final + lost → lost = 9 := by sorry

end NUMINAMATH_CALUDE_brandon_lost_skittles_l447_44727


namespace NUMINAMATH_CALUDE_tan_double_angle_gt_double_tan_l447_44752

theorem tan_double_angle_gt_double_tan (α : Real) (h1 : 0 < α) (h2 : α < π/4) :
  Real.tan (2 * α) > 2 * Real.tan α := by
  sorry

end NUMINAMATH_CALUDE_tan_double_angle_gt_double_tan_l447_44752


namespace NUMINAMATH_CALUDE_line_segment_parameter_sum_of_squares_l447_44701

/-- Given a line segment connecting (1, -3) and (-4, 9), parameterized by x = at + b and y = ct + d
    where 0 ≤ t ≤ 1 and t = 0 corresponds to (1, -3), prove that a^2 + b^2 + c^2 + d^2 = 179 -/
theorem line_segment_parameter_sum_of_squares :
  ∀ (a b c d : ℝ),
  (∀ t : ℝ, 0 ≤ t → t ≤ 1 → ∃ x y : ℝ, x = a * t + b ∧ y = c * t + d) →
  (b = 1 ∧ d = -3) →
  (a + b = -4 ∧ c + d = 9) →
  a^2 + b^2 + c^2 + d^2 = 179 := by
sorry

end NUMINAMATH_CALUDE_line_segment_parameter_sum_of_squares_l447_44701


namespace NUMINAMATH_CALUDE_total_red_stripes_on_ten_flags_l447_44773

/-- Represents an American flag -/
structure AmericanFlag where
  stripes : ℕ
  firstStripeRed : Bool
  halfRemainingRed : Bool

/-- Calculates the number of red stripes on a single American flag -/
def redStripesPerFlag (flag : AmericanFlag) : ℕ :=
  if flag.firstStripeRed ∧ flag.halfRemainingRed then
    1 + (flag.stripes - 1) / 2
  else
    0

/-- Theorem stating the total number of red stripes on 10 American flags -/
theorem total_red_stripes_on_ten_flags :
  ∀ (flag : AmericanFlag),
    flag.stripes = 13 →
    flag.firstStripeRed = true →
    flag.halfRemainingRed = true →
    (redStripesPerFlag flag * 10 = 70) :=
by
  sorry

end NUMINAMATH_CALUDE_total_red_stripes_on_ten_flags_l447_44773


namespace NUMINAMATH_CALUDE_pedometer_miles_calculation_l447_44761

/-- Calculates the approximate number of miles walked given pedometer data --/
theorem pedometer_miles_calculation 
  (max_steps : ℕ) 
  (flips : ℕ) 
  (final_reading : ℕ) 
  (steps_per_mile : ℕ) : 
  (((flips * (max_steps + 1) + final_reading) : ℚ) / steps_per_mile).floor = 2619 :=
by
  sorry

#check pedometer_miles_calculation 99999 52 38200 2000

end NUMINAMATH_CALUDE_pedometer_miles_calculation_l447_44761


namespace NUMINAMATH_CALUDE_divisor_of_a_l447_44745

theorem divisor_of_a (a b c d : ℕ+) 
  (h1 : Nat.gcd a b = 18)
  (h2 : Nat.gcd b c = 45)
  (h3 : Nat.gcd c d = 60)
  (h4 : 90 < Nat.gcd d a ∧ Nat.gcd d a < 120) :
  5 ∣ a := by
  sorry

end NUMINAMATH_CALUDE_divisor_of_a_l447_44745


namespace NUMINAMATH_CALUDE_blackboard_number_increase_l447_44785

theorem blackboard_number_increase (n k : ℕ+) :
  let new_k := k + Nat.gcd k n
  (new_k - k = 1) ∨ (Nat.Prime (new_k - k)) :=
sorry

end NUMINAMATH_CALUDE_blackboard_number_increase_l447_44785


namespace NUMINAMATH_CALUDE_quadratic_root_implies_d_l447_44778

theorem quadratic_root_implies_d (d : ℚ) : 
  (∀ x : ℝ, 2 * x^2 + 14 * x + d = 0 ↔ x = -7 + Real.sqrt 15 ∨ x = -7 - Real.sqrt 15) →
  d = 181 / 8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_d_l447_44778
