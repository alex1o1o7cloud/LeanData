import Mathlib

namespace NUMINAMATH_CALUDE_total_snow_volume_is_101_25_l2105_210504

/-- The volume of snow on a rectangular section of sidewalk -/
def snow_volume (length width depth : ℝ) : ℝ := length * width * depth

/-- The total volume of snow on two rectangular sections of sidewalk -/
def total_snow_volume (l1 w1 d1 l2 w2 d2 : ℝ) : ℝ :=
  snow_volume l1 w1 d1 + snow_volume l2 w2 d2

/-- Theorem: The total volume of snow on the given sidewalk sections is 101.25 cubic feet -/
theorem total_snow_volume_is_101_25 :
  total_snow_volume 25 3 0.75 15 3 1 = 101.25 := by
  sorry

#eval total_snow_volume 25 3 0.75 15 3 1

end NUMINAMATH_CALUDE_total_snow_volume_is_101_25_l2105_210504


namespace NUMINAMATH_CALUDE_triangle_area_l2105_210560

-- Define the vertices of the triangle
def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (7, 1)
def C : ℝ × ℝ := (5, 6)

-- State the theorem
theorem triangle_area : 
  let area := abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2)
  area = 13 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l2105_210560


namespace NUMINAMATH_CALUDE_cookie_calories_is_250_l2105_210511

/-- The number of calories in a cookie, given the total lunch calories,
    burger calories, carrot stick calories, and number of carrot sticks. -/
def cookie_calories (total_lunch_calories burger_calories carrot_stick_calories num_carrot_sticks : ℕ) : ℕ :=
  total_lunch_calories - (burger_calories + carrot_stick_calories * num_carrot_sticks)

/-- Theorem stating that each cookie has 250 calories under the given conditions. -/
theorem cookie_calories_is_250 :
  cookie_calories 750 400 20 5 = 250 := by
  sorry

end NUMINAMATH_CALUDE_cookie_calories_is_250_l2105_210511


namespace NUMINAMATH_CALUDE_recurring_decimal_to_fraction_l2105_210579

theorem recurring_decimal_to_fraction : 
  ∃ (x : ℚ), x = 4 + 56 / 99 ∧ x = 452 / 99 := by sorry

end NUMINAMATH_CALUDE_recurring_decimal_to_fraction_l2105_210579


namespace NUMINAMATH_CALUDE_zoo_animals_count_l2105_210576

/-- The number of penguins in the zoo -/
def num_penguins : ℕ := 21

/-- The number of polar bears in the zoo -/
def num_polar_bears : ℕ := 2 * num_penguins

/-- The total number of animals in the zoo -/
def total_animals : ℕ := num_penguins + num_polar_bears

theorem zoo_animals_count : total_animals = 63 := by
  sorry

end NUMINAMATH_CALUDE_zoo_animals_count_l2105_210576


namespace NUMINAMATH_CALUDE_skew_parameter_calculation_l2105_210577

/-- Dilation matrix -/
def D (k : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![k, 0; 0, k]

/-- Skew transformation matrix -/
def S (a : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![1, a; 0, 1]

/-- The problem statement -/
theorem skew_parameter_calculation (k : ℝ) (a : ℝ) (h1 : k > 0) :
  S a * D k = !![10, 5; 0, 10] →
  a = 1/2 := by
sorry

end NUMINAMATH_CALUDE_skew_parameter_calculation_l2105_210577


namespace NUMINAMATH_CALUDE_complex_number_equality_l2105_210541

theorem complex_number_equality (z : ℂ) : (z - 2) * Complex.I = 1 + Complex.I → z = 3 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_equality_l2105_210541


namespace NUMINAMATH_CALUDE_stratified_sampling_sophomores_l2105_210508

theorem stratified_sampling_sophomores (total_students : ℕ) (sophomores : ℕ) (selected : ℕ) 
  (h1 : total_students = 2800) 
  (h2 : sophomores = 930) 
  (h3 : selected = 280) :
  (sophomores * selected) / total_students = 93 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_sophomores_l2105_210508


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2105_210548

theorem hyperbola_equation (a b : ℝ) (h1 : a = 6) (h2 : b = Real.sqrt 35) :
  ∀ x y : ℝ, (y^2 / 36 - x^2 / 35 = 1) ↔ 
  (∃ c : ℝ, c^2 = a^2 + b^2 ∧ 
   ∀ F₁ F₂ : ℝ × ℝ, F₁ = (0, c) ∧ F₂ = (0, -c) → 
   (y - F₁.2)^2 + x^2 - (y - F₂.2)^2 - x^2 = 4 * a^2) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2105_210548


namespace NUMINAMATH_CALUDE_lidia_remaining_money_l2105_210521

/-- Proves the remaining money after Lidia buys her needed apps -/
theorem lidia_remaining_money 
  (app_cost : ℕ) 
  (apps_needed : ℕ) 
  (available_money : ℕ) 
  (h1 : app_cost = 4)
  (h2 : apps_needed = 15)
  (h3 : available_money = 66) :
  available_money - (app_cost * apps_needed) = 6 :=
by sorry

end NUMINAMATH_CALUDE_lidia_remaining_money_l2105_210521


namespace NUMINAMATH_CALUDE_maggie_subscriptions_to_parents_l2105_210574

-- Define the price per subscription
def price_per_subscription : ℕ := 5

-- Define the number of subscriptions sold to different people
def subscriptions_to_grandfather : ℕ := 1
def subscriptions_to_next_door : ℕ := 2
def subscriptions_to_another_neighbor : ℕ := 2 * subscriptions_to_next_door

-- Define the total earnings
def total_earnings : ℕ := 55

-- Define the number of subscriptions sold to parents
def subscriptions_to_parents : ℕ := 4

-- Theorem to prove
theorem maggie_subscriptions_to_parents :
  subscriptions_to_parents * price_per_subscription +
  (subscriptions_to_grandfather + subscriptions_to_next_door + subscriptions_to_another_neighbor) * price_per_subscription =
  total_earnings :=
by sorry

end NUMINAMATH_CALUDE_maggie_subscriptions_to_parents_l2105_210574


namespace NUMINAMATH_CALUDE_power_sum_equality_l2105_210546

theorem power_sum_equality : (3^2)^2 + (2^3)^3 = 593 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equality_l2105_210546


namespace NUMINAMATH_CALUDE_dividend_calculation_l2105_210509

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 18) 
  (h2 : quotient = 9) 
  (h3 : remainder = 3) : 
  divisor * quotient + remainder = 165 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l2105_210509


namespace NUMINAMATH_CALUDE_round_trip_ticket_percentage_l2105_210561

/-- Given that 25% of all passengers held round-trip tickets and took their cars aboard,
    and 60% of passengers with round-trip tickets did not take their cars aboard,
    prove that 62.5% of all passengers held round-trip tickets. -/
theorem round_trip_ticket_percentage
  (total_passengers : ℝ)
  (h1 : total_passengers > 0)
  (h2 : (25 : ℝ) / 100 * total_passengers = (40 : ℝ) / 100 * ((100 : ℝ) / 100 * total_passengers)) :
  (62.5 : ℝ) / 100 * total_passengers = (100 : ℝ) / 100 * total_passengers :=
sorry

end NUMINAMATH_CALUDE_round_trip_ticket_percentage_l2105_210561


namespace NUMINAMATH_CALUDE_prop_1_prop_4_l2105_210518

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (intersect : Plane → Plane → Line → Prop)

variable (m n : Line)
variable (α β : Plane)

-- Proposition 1
theorem prop_1 (h1 : parallel_planes α β) (h2 : subset m α) :
  parallel_line_plane m β := by sorry

-- Proposition 4
theorem prop_4 (h1 : parallel_line_plane m β) (h2 : subset m α) (h3 : intersect α β n) :
  parallel_lines m n := by sorry

end NUMINAMATH_CALUDE_prop_1_prop_4_l2105_210518


namespace NUMINAMATH_CALUDE_quadratic_roots_not_integers_l2105_210525

/-- 
Given a quadratic polynomial p(x) = ax² + bx + c where a, b, and c are odd integers,
if the roots x₁ and x₂ exist, they cannot both be integers.
-/
theorem quadratic_roots_not_integers 
  (a b c : ℤ) 
  (ha : Odd a) (hb : Odd b) (hc : Odd c)
  (hroots : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0) :
  ¬∃ (y₁ y₂ : ℤ), (y₁ : ℝ) = x₁ ∧ (y₂ : ℝ) = x₂ :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_not_integers_l2105_210525


namespace NUMINAMATH_CALUDE_initial_commission_rate_l2105_210551

theorem initial_commission_rate 
  (income_unchanged : ℝ → ℝ → ℝ → ℝ → Bool)
  (new_rate : ℝ)
  (business_slump : ℝ)
  (initial_rate : ℝ) :
  income_unchanged initial_rate new_rate business_slump initial_rate →
  new_rate = 5 →
  business_slump = 20.000000000000007 →
  initial_rate = 4 := by
sorry

end NUMINAMATH_CALUDE_initial_commission_rate_l2105_210551


namespace NUMINAMATH_CALUDE_smallest_k_square_root_diff_l2105_210580

/-- Represents a card with a number from 1 to 2016 -/
def Card := {n : ℕ // 1 ≤ n ∧ n ≤ 2016}

/-- The property that two cards have numbers whose square roots differ by less than 1 -/
def SquareRootDiffLessThanOne (a b : Card) : Prop :=
  |Real.sqrt a.val - Real.sqrt b.val| < 1

/-- The theorem stating that 45 is the smallest number of cards guaranteeing
    two cards with square root difference less than 1 -/
theorem smallest_k_square_root_diff : 
  (∀ (S : Finset Card), S.card = 45 → 
    ∃ (a b : Card), a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ SquareRootDiffLessThanOne a b) ∧
  (∀ (k : ℕ), k < 45 → 
    ∃ (S : Finset Card), S.card = k ∧
      ∀ (a b : Card), a ∈ S → b ∈ S → a ≠ b → ¬SquareRootDiffLessThanOne a b) :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_square_root_diff_l2105_210580


namespace NUMINAMATH_CALUDE_player_positions_satisfy_distances_l2105_210528

/-- Represents the positions of four soccer players on a number line -/
def PlayerPositions : Fin 4 → ℝ
| 0 => 0
| 1 => 1
| 2 => 4
| 3 => 6

/-- Calculates the distance between two players -/
def distance (i j : Fin 4) : ℝ :=
  |PlayerPositions i - PlayerPositions j|

/-- The set of required pairwise distances -/
def RequiredDistances : Set ℝ := {1, 2, 3, 4, 5, 6}

/-- Theorem stating that the player positions satisfy the required distances -/
theorem player_positions_satisfy_distances :
  ∀ i j : Fin 4, i ≠ j → distance i j ∈ RequiredDistances :=
sorry

end NUMINAMATH_CALUDE_player_positions_satisfy_distances_l2105_210528


namespace NUMINAMATH_CALUDE_adjacent_permutations_of_six_l2105_210539

/-- The number of permutations of n elements where two specific elements are always adjacent -/
def adjacentPermutations (n : ℕ) : ℕ :=
  2 * Nat.factorial (n - 1)

/-- Given 6 people with two specific individuals, the number of permutations
    where these two individuals are always adjacent is 240 -/
theorem adjacent_permutations_of_six :
  adjacentPermutations 6 = 240 := by
  sorry

end NUMINAMATH_CALUDE_adjacent_permutations_of_six_l2105_210539


namespace NUMINAMATH_CALUDE_trig_identity_l2105_210558

theorem trig_identity : 
  (Real.cos (20 * π / 180) * Real.sin (20 * π / 180)) / 
  (Real.cos (25 * π / 180)^2 - Real.sin (25 * π / 180)^2) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2105_210558


namespace NUMINAMATH_CALUDE_plane_perpendicular_parallel_transitive_l2105_210572

/-- A structure representing a 3D space with planes and perpendicularity/parallelism relations -/
structure Space3D where
  Plane : Type
  perpendicular : Plane → Plane → Prop
  parallel : Plane → Plane → Prop

/-- The main theorem to be proved -/
theorem plane_perpendicular_parallel_transitive 
  (S : Space3D) (α β γ : S.Plane) : 
  S.perpendicular α β → S.parallel α γ → S.perpendicular β γ := by
  sorry

/-- Helper lemma: If two planes are parallel, they are not perpendicular -/
lemma parallel_not_perpendicular 
  (S : Space3D) (p q : S.Plane) :
  S.parallel p q → ¬S.perpendicular p q := by
  sorry

/-- Helper lemma: Perpendicularity is symmetric -/
lemma perpendicular_symmetric 
  (S : Space3D) (p q : S.Plane) :
  S.perpendicular p q → S.perpendicular q p := by
  sorry

/-- Helper lemma: Parallelism is symmetric -/
lemma parallel_symmetric 
  (S : Space3D) (p q : S.Plane) :
  S.parallel p q → S.parallel q p := by
  sorry

end NUMINAMATH_CALUDE_plane_perpendicular_parallel_transitive_l2105_210572


namespace NUMINAMATH_CALUDE_correct_reasoning_directions_l2105_210593

-- Define the types of reasoning
inductive ReasoningType
| Inductive
| Deductive
| Analogical

-- Define the direction of reasoning
inductive ReasoningDirection
| SpecificToGeneral
| GeneralToSpecific
| SpecificToSpecific

-- Define a function that describes the direction of each reasoning type
def reasoningDirection (rt : ReasoningType) : ReasoningDirection :=
  match rt with
  | ReasoningType.Inductive => ReasoningDirection.SpecificToGeneral
  | ReasoningType.Deductive => ReasoningDirection.GeneralToSpecific
  | ReasoningType.Analogical => ReasoningDirection.SpecificToSpecific

-- Theorem stating that the reasoning directions are correct
theorem correct_reasoning_directions :
  (reasoningDirection ReasoningType.Inductive = ReasoningDirection.SpecificToGeneral) ∧
  (reasoningDirection ReasoningType.Deductive = ReasoningDirection.GeneralToSpecific) ∧
  (reasoningDirection ReasoningType.Analogical = ReasoningDirection.SpecificToSpecific) :=
by sorry

end NUMINAMATH_CALUDE_correct_reasoning_directions_l2105_210593


namespace NUMINAMATH_CALUDE_unique_solution_system_l2105_210531

theorem unique_solution_system :
  ∃! (x y : ℚ), (3 * x - 2 * y = (6 - 2 * x) + (6 - 2 * y)) ∧
                 (x + 3 * y = (2 * x + 1) - (2 * y + 1)) ∧
                 x = 12 / 5 ∧ y = 12 / 25 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_system_l2105_210531


namespace NUMINAMATH_CALUDE_jack_minimum_cars_per_hour_l2105_210563

/-- The minimum number of cars Jack can change oil in per hour -/
def jack_cars_per_hour : ℝ := 3

/-- The number of hours worked per day -/
def hours_per_day : ℝ := 8

/-- The number of cars Paul can change oil in per hour -/
def paul_cars_per_hour : ℝ := 2

/-- The minimum number of cars both mechanics can finish per day -/
def min_cars_per_day : ℝ := 40

theorem jack_minimum_cars_per_hour :
  jack_cars_per_hour * hours_per_day + paul_cars_per_hour * hours_per_day ≥ min_cars_per_day ∧
  ∀ x : ℝ, x * hours_per_day + paul_cars_per_hour * hours_per_day ≥ min_cars_per_day → x ≥ jack_cars_per_hour :=
by sorry

end NUMINAMATH_CALUDE_jack_minimum_cars_per_hour_l2105_210563


namespace NUMINAMATH_CALUDE_remainder_divisibility_l2105_210595

theorem remainder_divisibility (x : ℤ) : x % 52 = 19 → x % 7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_divisibility_l2105_210595


namespace NUMINAMATH_CALUDE_container_capacity_l2105_210554

theorem container_capacity (C : ℝ) : 
  C > 0 → 
  0.30 * C + 36 = 0.75 * C → 
  C = 80 := by
sorry

end NUMINAMATH_CALUDE_container_capacity_l2105_210554


namespace NUMINAMATH_CALUDE_g_of_2_equals_11_l2105_210535

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x^2 + 2 * x - 5

-- Theorem statement
theorem g_of_2_equals_11 : g 2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_g_of_2_equals_11_l2105_210535


namespace NUMINAMATH_CALUDE_expression_value_l2105_210542

theorem expression_value (x y : ℝ) (h1 : x ≠ y) 
  (h2 : 1 / (1 + x^2) + 1 / (1 + y^2) = 2 / (1 + x*y)) : 
  1 / (1 + x^2) + 1 / (1 + y^2) + 2 / (1 + x*y) = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2105_210542


namespace NUMINAMATH_CALUDE_calculation_proof_l2105_210586

theorem calculation_proof : (((2207 - 2024) ^ 2 * 4) : ℚ) / 144 = 930.25 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2105_210586


namespace NUMINAMATH_CALUDE_hot_sauce_servings_per_day_l2105_210549

/-- Proves the number of hot sauce servings used per day -/
theorem hot_sauce_servings_per_day 
  (serving_size : Real) 
  (jar_size : Real) 
  (duration : Nat) 
  (h1 : serving_size = 0.5)
  (h2 : jar_size = 32 - 2)
  (h3 : duration = 20) :
  (jar_size / duration) / serving_size = 3 := by
  sorry

end NUMINAMATH_CALUDE_hot_sauce_servings_per_day_l2105_210549


namespace NUMINAMATH_CALUDE_sum_of_two_numbers_l2105_210543

theorem sum_of_two_numbers (x y : ℝ) (h1 : x - y = 9) (h2 : y = 18.5) : x + y = 46 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_numbers_l2105_210543


namespace NUMINAMATH_CALUDE_money_distribution_problem_l2105_210547

/-- Represents the shares of P, Q, and R in the money distribution problem. -/
structure Shares where
  p : ℕ
  q : ℕ
  r : ℕ

/-- Represents the problem constraints and solution. -/
theorem money_distribution_problem (s : Shares) : 
  -- The ratio condition
  s.p + s.q + s.r > 0 ∧ 
  3 * s.q = 7 * s.p ∧ 
  3 * s.r = 4 * s.q ∧ 
  -- The difference between P and Q's shares
  s.q - s.p = 2800 ∧ 
  -- Total amount condition
  50000 ≤ s.p + s.q + s.r ∧ 
  s.p + s.q + s.r ≤ 75000 ∧ 
  -- Minimum and maximum share conditions
  s.p ≥ 5000 ∧ 
  s.r ≤ 45000 
  -- The difference between Q and R's shares
  → s.r - s.q = 14000 := by sorry

end NUMINAMATH_CALUDE_money_distribution_problem_l2105_210547


namespace NUMINAMATH_CALUDE_books_loaned_out_l2105_210599

/-- The number of books in the special collection at the beginning of the month -/
def initial_books : ℕ := 150

/-- The percentage of loaned books that are returned -/
def return_rate : ℚ := 85 / 100

/-- The number of books in the special collection at the end of the month -/
def final_books : ℕ := 135

/-- The number of books damaged or lost and replaced -/
def replaced_books : ℕ := 5

/-- The number of books loaned out during the month -/
def loaned_books : ℕ := 133

theorem books_loaned_out : 
  initial_books - loaned_books + (return_rate * loaned_books).floor + replaced_books = final_books :=
sorry

end NUMINAMATH_CALUDE_books_loaned_out_l2105_210599


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l2105_210557

theorem simplify_trig_expression (x : ℝ) :
  (Real.sin x + Real.sin (3 * x)) / (1 + Real.cos x + Real.cos (3 * x)) =
  (4 * Real.sin x * Real.cos x ^ 2) / (1 - 4 * Real.cos x ^ 2) := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l2105_210557


namespace NUMINAMATH_CALUDE_remainder_theorem_l2105_210565

theorem remainder_theorem (n : ℤ) (h : ∃ k : ℤ, n = 25 * k - 1) :
  (n^2 + 3*n + 5) % 25 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2105_210565


namespace NUMINAMATH_CALUDE_not_right_triangle_4_6_8_l2105_210582

/-- Checks if three line segments can form a right triangle -/
def is_right_triangle (a b c : ℝ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

/-- Theorem stating that the line segments 4, 6, and 8 cannot form a right triangle -/
theorem not_right_triangle_4_6_8 : ¬ is_right_triangle 4 6 8 := by
  sorry

end NUMINAMATH_CALUDE_not_right_triangle_4_6_8_l2105_210582


namespace NUMINAMATH_CALUDE_coefficient_x5y2_is_90_l2105_210524

/-- The coefficient of x^5y^2 in the expansion of (x^2 + 3x - y)^5 -/
def coefficient_x5y2 : ℕ :=
  let n : ℕ := 5
  let k : ℕ := 3
  let binomial_coeff : ℕ := n.choose k
  let x_coeff : ℕ := 9  -- Coefficient of x^5 in (x^2 + 3x)^3
  binomial_coeff * x_coeff

/-- The coefficient of x^5y^2 in the expansion of (x^2 + 3x - y)^5 is 90 -/
theorem coefficient_x5y2_is_90 : coefficient_x5y2 = 90 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x5y2_is_90_l2105_210524


namespace NUMINAMATH_CALUDE_gas_used_l2105_210530

theorem gas_used (initial_gas final_gas : ℝ) (h1 : initial_gas = 0.5) (h2 : final_gas = 0.17) :
  initial_gas - final_gas = 0.33 := by
sorry

end NUMINAMATH_CALUDE_gas_used_l2105_210530


namespace NUMINAMATH_CALUDE_fraction_simplification_l2105_210512

theorem fraction_simplification (a : ℚ) (h : a ≠ 2) :
  (a^2 / (a - 2)) - (4 / (a - 2)) = a + 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2105_210512


namespace NUMINAMATH_CALUDE_solve_equation_l2105_210567

theorem solve_equation : 
  let x := 70 / (8 - 3/4)
  x = 280/29 := by sorry

end NUMINAMATH_CALUDE_solve_equation_l2105_210567


namespace NUMINAMATH_CALUDE_different_graphs_l2105_210501

-- Define the three equations
def eq1 (x y : ℝ) : Prop := y = x - 3
def eq2 (x y : ℝ) : Prop := y = (x^2 - 9) / (x + 3)
def eq3 (x y : ℝ) : Prop := (x + 3) * y = x^2 - 9

-- Define what it means for two equations to have the same graph
def same_graph (f g : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, f x y ↔ g x y

-- Theorem stating that all three equations have different graphs
theorem different_graphs : 
  ¬(same_graph eq1 eq2) ∧ ¬(same_graph eq1 eq3) ∧ ¬(same_graph eq2 eq3) :=
sorry

end NUMINAMATH_CALUDE_different_graphs_l2105_210501


namespace NUMINAMATH_CALUDE_odd_increasing_nonneg_implies_increasing_neg_l2105_210545

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- A function f is increasing on a set S if f(x) ≤ f(y) for all x, y ∈ S with x ≤ y -/
def IsIncreasingOn (f : ℝ → ℝ) (S : Set ℝ) : Prop :=
  ∀ x y, x ∈ S → y ∈ S → x ≤ y → f x ≤ f y

theorem odd_increasing_nonneg_implies_increasing_neg
  (f : ℝ → ℝ) (h_odd : IsOdd f) (h_incr_nonneg : IsIncreasingOn f (Set.Ici 0)) :
  IsIncreasingOn f (Set.Iic 0) :=
sorry

end NUMINAMATH_CALUDE_odd_increasing_nonneg_implies_increasing_neg_l2105_210545


namespace NUMINAMATH_CALUDE_math_competition_results_l2105_210596

/-- Represents the number of joint math competitions --/
def num_competitions : ℕ := 5

/-- Represents the probability of ranking in the top 20 in each competition --/
def prob_top20 : ℚ := 1/4

/-- Represents the number of top 20 rankings needed to qualify for provincial training --/
def qualify_threshold : ℕ := 2

/-- Models the outcome of a student's participation in the math competitions --/
structure StudentOutcome where
  num_participated : ℕ
  num_top20 : ℕ
  qualified : Bool

/-- Calculates the probability of a specific outcome --/
noncomputable def prob_outcome (outcome : StudentOutcome) : ℚ :=
  sorry

/-- Calculates the probability of qualifying for provincial training --/
noncomputable def prob_qualify : ℚ :=
  sorry

/-- Calculates the expected number of competitions participated in, given qualification or completion --/
noncomputable def expected_num_competitions : ℚ :=
  sorry

/-- Main theorem stating the probabilities and expected value --/
theorem math_competition_results :
  prob_qualify = 67/256 ∧ expected_num_competitions = 65/16 :=
by sorry

end NUMINAMATH_CALUDE_math_competition_results_l2105_210596


namespace NUMINAMATH_CALUDE_age_difference_l2105_210556

theorem age_difference (anand_age_10_years_ago bala_age_10_years_ago : ℕ) : 
  anand_age_10_years_ago = bala_age_10_years_ago / 3 →
  anand_age_10_years_ago + 10 = 15 →
  (bala_age_10_years_ago + 10) - (anand_age_10_years_ago + 10) = 10 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l2105_210556


namespace NUMINAMATH_CALUDE_cos_m_eq_sin_318_l2105_210564

theorem cos_m_eq_sin_318 (m : ℤ) (h1 : -180 ≤ m) (h2 : m ≤ 180) (h3 : Real.cos (m * π / 180) = Real.sin (318 * π / 180)) :
  m = 132 ∨ m = -132 := by
sorry

end NUMINAMATH_CALUDE_cos_m_eq_sin_318_l2105_210564


namespace NUMINAMATH_CALUDE_unique_cinema_ticket_price_l2105_210578

/-- The price of adult and child cinema tickets satisfying given conditions -/
def cinema_tickets (adult_price : ℚ) : Prop :=
  let child_price := adult_price / 2
  3 * adult_price + 5 * child_price = 27 ∧
  10 * adult_price + 15 * child_price = 945 / 11

/-- Theorem stating that there exists a unique adult ticket price satisfying the conditions -/
theorem unique_cinema_ticket_price :
  ∃! adult_price : ℚ, cinema_tickets adult_price :=
sorry

end NUMINAMATH_CALUDE_unique_cinema_ticket_price_l2105_210578


namespace NUMINAMATH_CALUDE_goldfish_equality_l2105_210597

theorem goldfish_equality (n : ℕ) : (∀ k : ℕ, k < n → 4^(k+1) ≠ 128 * 2^k) ∧ 4^(n+1) = 128 * 2^n ↔ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_goldfish_equality_l2105_210597


namespace NUMINAMATH_CALUDE_certain_number_value_l2105_210585

/-- Represents the number system in the certain country -/
structure CountryNumber where
  value : ℕ

/-- Multiplication operation in the country's number system -/
def country_mul (a b : CountryNumber) : CountryNumber :=
  ⟨a.value * b.value⟩

/-- Division operation in the country's number system -/
def country_div (a b : CountryNumber) : CountryNumber :=
  ⟨a.value / b.value⟩

/-- Equality in the country's number system -/
def country_eq (a b : CountryNumber) : Prop :=
  a.value = b.value

theorem certain_number_value :
  ∀ (eight seven five : CountryNumber),
    country_eq (country_div eight seven) five →
    ∀ (x : CountryNumber),
      country_eq (country_div x ⟨5⟩) ⟨35⟩ →
      country_eq x ⟨175⟩ :=
by sorry

end NUMINAMATH_CALUDE_certain_number_value_l2105_210585


namespace NUMINAMATH_CALUDE_circle_line_intersection_l2105_210519

/-- Given a circle and a line with a specific chord length, prove the possible values of 'a' -/
theorem circle_line_intersection (a : ℝ) : 
  (∃ (x y : ℝ), x^2 + (y + a)^2 = 4 ∧ x - y - 2 = 0) →  -- Circle and line equations
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁^2 + (y₁ + a)^2 = 4 ∧ 
    x₁ - y₁ - 2 = 0 ∧
    x₂^2 + (y₂ + a)^2 = 4 ∧ 
    x₂ - y₂ - 2 = 0 ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 8) →  -- Chord length condition
  a = 0 ∨ a = 4 := by
sorry

end NUMINAMATH_CALUDE_circle_line_intersection_l2105_210519


namespace NUMINAMATH_CALUDE_three_number_problem_l2105_210591

theorem three_number_problem (x y z : ℝ) 
  (sum_eq : x + y + z = 500)
  (x_eq : x = 200)
  (y_eq : y = 2 * z)
  (diff_eq : x - z = 0.5 * y) :
  z = 100 := by
  sorry

end NUMINAMATH_CALUDE_three_number_problem_l2105_210591


namespace NUMINAMATH_CALUDE_largest_three_digit_multiple_of_7_with_digit_sum_21_l2105_210587

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

theorem largest_three_digit_multiple_of_7_with_digit_sum_21 :
  ∃ (n : ℕ), is_three_digit n ∧ n % 7 = 0 ∧ digit_sum n = 21 ∧
  ∀ (m : ℕ), is_three_digit m ∧ m % 7 = 0 ∧ digit_sum m = 21 → m ≤ n :=
by
  use 966
  sorry

end NUMINAMATH_CALUDE_largest_three_digit_multiple_of_7_with_digit_sum_21_l2105_210587


namespace NUMINAMATH_CALUDE_root_implies_m_value_always_real_roots_l2105_210503

-- Define the quadratic equation
def quadratic (m x : ℝ) : ℝ := x^2 - m*x + m - 1

-- Theorem 1: If x = 3 is a root, then m = 4
theorem root_implies_m_value (m : ℝ) : quadratic m 3 = 0 → m = 4 := by sorry

-- Theorem 2: The quadratic equation always has real roots
theorem always_real_roots (m : ℝ) : ∃ x : ℝ, quadratic m x = 0 := by sorry

end NUMINAMATH_CALUDE_root_implies_m_value_always_real_roots_l2105_210503


namespace NUMINAMATH_CALUDE_square_sum_equation_solutions_l2105_210573

/-- The Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- The property that a number is the sum of two squares -/
def isSumOfTwoSquares (n : ℕ) : Prop :=
  ∃ x y : ℕ, x^2 + y^2 = n

/-- The main theorem -/
theorem square_sum_equation_solutions :
  (∃ k : ℕ+, 
    (∃ a b c : ℕ+, a^2 + b^2 + c^2 = k * a * b * c) ∧ 
    (∀ n : ℕ, ∃ a_n b_n c_n : ℕ+,
      a_n^2 + b_n^2 + c_n^2 = k * a_n * b_n * c_n ∧
      isSumOfTwoSquares (a_n * b_n) ∧
      isSumOfTwoSquares (b_n * c_n) ∧
      isSumOfTwoSquares (c_n * a_n))) ↔
  (k = 1 ∨ k = 3) := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equation_solutions_l2105_210573


namespace NUMINAMATH_CALUDE_pizza_problem_l2105_210540

/-- Represents a pizza with a given number of slices and topping distribution. -/
structure Pizza where
  total_slices : ℕ
  pepperoni_slices : ℕ
  mushroom_slices : ℕ
  both_toppings_slices : ℕ

/-- The pizza satisfies the given conditions. -/
def valid_pizza (p : Pizza) : Prop :=
  p.total_slices = 15 ∧
  p.pepperoni_slices = 8 ∧
  p.mushroom_slices = 12 ∧
  p.both_toppings_slices ≤ p.pepperoni_slices ∧
  p.both_toppings_slices ≤ p.mushroom_slices ∧
  p.pepperoni_slices + p.mushroom_slices - p.both_toppings_slices = p.total_slices

theorem pizza_problem (p : Pizza) (h : valid_pizza p) : p.both_toppings_slices = 5 := by
  sorry

end NUMINAMATH_CALUDE_pizza_problem_l2105_210540


namespace NUMINAMATH_CALUDE_f_negative_a_eq_zero_l2105_210522

noncomputable def f (x : ℝ) : ℝ := x^3 + Real.sin x + 1

theorem f_negative_a_eq_zero (a : ℝ) (h : f a = 2) : f (-a) = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_negative_a_eq_zero_l2105_210522


namespace NUMINAMATH_CALUDE_min_trig_expression_l2105_210581

theorem min_trig_expression (x : ℝ) : 
  (Real.sin x)^8 + (Real.cos x)^8 + 1 ≥ 7/18 * ((Real.sin x)^6 + (Real.cos x)^6 + 1) := by
  sorry

end NUMINAMATH_CALUDE_min_trig_expression_l2105_210581


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2105_210589

theorem arithmetic_sequence_problem (a : ℕ → ℝ) (d : ℝ) :
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence condition
  a 3 + a 8 = 10 →              -- given condition
  3 * a 5 + a 7 = 20 :=         -- conclusion to prove
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2105_210589


namespace NUMINAMATH_CALUDE_attractions_permutations_l2105_210502

theorem attractions_permutations : Nat.factorial 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_attractions_permutations_l2105_210502


namespace NUMINAMATH_CALUDE_cylindrical_surface_is_cylinder_l2105_210517

/-- A point in cylindrical coordinates -/
structure CylindricalPoint where
  r : ℝ
  θ : ℝ
  z : ℝ

/-- The set of points satisfying r = c in cylindrical coordinates -/
def CylindricalSurface (c : ℝ) : Set CylindricalPoint :=
  {p : CylindricalPoint | p.r = c}

/-- Definition of a cylinder in cylindrical coordinates -/
def IsCylinder (S : Set CylindricalPoint) : Prop :=
  ∃ c : ℝ, c > 0 ∧ S = CylindricalSurface c

theorem cylindrical_surface_is_cylinder (c : ℝ) (h : c > 0) :
  IsCylinder (CylindricalSurface c) := by
  sorry

end NUMINAMATH_CALUDE_cylindrical_surface_is_cylinder_l2105_210517


namespace NUMINAMATH_CALUDE_min_sum_inverse_squares_min_sum_inverse_squares_value_min_sum_inverse_squares_equality_l2105_210584

theorem min_sum_inverse_squares (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (sum_squares : x^2 + y^2 + z^2 = 1) :
  ∀ a b c : ℝ, 0 < a ∧ 0 < b ∧ 0 < c → a^2 + b^2 + c^2 = 1 →
  1/x^2 + 1/y^2 + 1/z^2 ≤ 1/a^2 + 1/b^2 + 1/c^2 :=
by sorry

theorem min_sum_inverse_squares_value (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (sum_squares : x^2 + y^2 + z^2 = 1) :
  1/x^2 + 1/y^2 + 1/z^2 ≥ 9 :=
by sorry

theorem min_sum_inverse_squares_equality :
  ∃ x y z : ℝ, 0 < x ∧ 0 < y ∧ 0 < z ∧ 
  x^2 + y^2 + z^2 = 1 ∧ 1/x^2 + 1/y^2 + 1/z^2 = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_inverse_squares_min_sum_inverse_squares_value_min_sum_inverse_squares_equality_l2105_210584


namespace NUMINAMATH_CALUDE_equation_solution_l2105_210594

theorem equation_solution :
  ∃! x : ℚ, (x + 5) / (x - 3) = (x - 2) / (x + 4) :=
by
  use -1
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2105_210594


namespace NUMINAMATH_CALUDE_range_of_f_when_a_is_2_properties_of_M_l2105_210537

-- Define the function f
def f (a x : ℝ) : ℝ := x^2 - a*x + 4 - a^2

-- Theorem for the range of f when a = 2
theorem range_of_f_when_a_is_2 :
  ∃ (y : ℝ), y ∈ Set.Icc (-1) 8 ↔ ∃ (x : ℝ), x ∈ Set.Icc (-2) 3 ∧ f 2 x = y :=
sorry

-- Define the set M
def M : Set ℝ := {4}

-- Theorem for the properties of M
theorem properties_of_M :
  (4 ∈ M) ∧
  (∀ a ∈ M, ∀ x ∈ Set.Icc (-2) 2, f a x ≤ 0) ∧
  (∃ b ∉ M, ∀ x ∈ Set.Icc (-2) 2, f b x ≤ 0) :=
sorry

end NUMINAMATH_CALUDE_range_of_f_when_a_is_2_properties_of_M_l2105_210537


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2105_210544

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_fraction_simplification :
  (2 - 3 * i) / (4 + 5 * i + 3 * i^2) = (-1/2 : ℂ) - (1/2 : ℂ) * i :=
by
  -- The proof would go here, but we're skipping it as per instructions
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2105_210544


namespace NUMINAMATH_CALUDE_books_per_bookshelf_l2105_210598

theorem books_per_bookshelf (num_bookshelves : ℕ) (magazines_per_bookshelf : ℕ) (total_items : ℕ) : 
  num_bookshelves = 29 →
  magazines_per_bookshelf = 61 →
  total_items = 2436 →
  (total_items - num_bookshelves * magazines_per_bookshelf) / num_bookshelves = 23 := by
sorry

end NUMINAMATH_CALUDE_books_per_bookshelf_l2105_210598


namespace NUMINAMATH_CALUDE_function_coefficient_sum_l2105_210562

/-- Given a function f : ℝ → ℝ satisfying certain conditions, prove that a + b + c = 3 -/
theorem function_coefficient_sum (f : ℝ → ℝ) (a b c : ℝ) :
  (∀ x, f (x + 3) = 4 * x^2 + 9 * x + 5) →
  (∀ x, f x = a * x^2 + b * x + c) →
  a + b + c = 3 := by
sorry

end NUMINAMATH_CALUDE_function_coefficient_sum_l2105_210562


namespace NUMINAMATH_CALUDE_pure_imaginary_fraction_l2105_210550

theorem pure_imaginary_fraction (a : ℝ) : 
  (((a : ℂ) - Complex.I) / (1 + Complex.I)).re = 0 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_fraction_l2105_210550


namespace NUMINAMATH_CALUDE_shopkeeper_profit_l2105_210516

theorem shopkeeper_profit (cost_price : ℝ) (discount_rate : ℝ) (profit_rate_with_discount : ℝ) 
  (h1 : discount_rate = 0.05)
  (h2 : profit_rate_with_discount = 0.273) :
  let selling_price_with_discount := cost_price * (1 + profit_rate_with_discount)
  let marked_price := selling_price_with_discount / (1 - discount_rate)
  let profit_rate_without_discount := (marked_price - cost_price) / cost_price
  profit_rate_without_discount = 0.34 := by
sorry

end NUMINAMATH_CALUDE_shopkeeper_profit_l2105_210516


namespace NUMINAMATH_CALUDE_base3_product_theorem_l2105_210555

/-- Converts a base 3 number to decimal --/
def base3ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * 3^i) 0

/-- Converts a decimal number to base 3 --/
def decimalToBase3 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc
      else aux (m / 3) ((m % 3) :: acc)
    aux n []

/-- Multiplies two base 3 numbers --/
def multiplyBase3 (a b : List Nat) : List Nat :=
  decimalToBase3 (base3ToDecimal a * base3ToDecimal b)

theorem base3_product_theorem :
  multiplyBase3 [2, 0, 1] [2, 1] = [2, 0, 2, 1] := by sorry

end NUMINAMATH_CALUDE_base3_product_theorem_l2105_210555


namespace NUMINAMATH_CALUDE_point_not_in_reflected_rectangle_l2105_210529

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point across the y-axis -/
def reflect_y (p : Point) : Point :=
  ⟨-p.x, p.y⟩

/-- The set of vertices of the original rectangle -/
def original_vertices : Set Point :=
  {⟨1, 3⟩, ⟨1, 1⟩, ⟨4, 1⟩, ⟨4, 3⟩}

/-- The set of vertices of the reflected rectangle -/
def reflected_vertices : Set Point :=
  original_vertices.image reflect_y

/-- The point in question -/
def point_to_check : Point :=
  ⟨-3, 4⟩

theorem point_not_in_reflected_rectangle :
  point_to_check ∉ reflected_vertices :=
sorry

end NUMINAMATH_CALUDE_point_not_in_reflected_rectangle_l2105_210529


namespace NUMINAMATH_CALUDE_cylinder_volume_l2105_210510

/-- The volume of a cylinder with equal base diameter and height, and lateral area π. -/
theorem cylinder_volume (r h : ℝ) (h1 : h = 2 * r) (h2 : 2 * π * r * h = π) : π * r^2 * h = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_l2105_210510


namespace NUMINAMATH_CALUDE_total_dangerous_animals_l2105_210568

def crocodiles : ℕ := 22
def alligators : ℕ := 23
def vipers : ℕ := 5

theorem total_dangerous_animals : crocodiles + alligators + vipers = 50 := by
  sorry

end NUMINAMATH_CALUDE_total_dangerous_animals_l2105_210568


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l2105_210559

def repeating_decimal_3 : ℚ := 1 / 3
def repeating_decimal_56 : ℚ := 56 / 99

theorem sum_of_repeating_decimals : 
  repeating_decimal_3 + repeating_decimal_56 = 89 / 99 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l2105_210559


namespace NUMINAMATH_CALUDE_max_quarters_count_l2105_210500

/-- Represents the value of a quarter in dollars -/
def quarter_value : ℚ := 0.25

/-- Represents the value of a dime in dollars -/
def dime_value : ℚ := 0.10

/-- Represents the total value of coins Sasha has in dollars -/
def total_value : ℚ := 2.10

theorem max_quarters_count : 
  ∀ q : ℕ, 
    (q : ℚ) * (quarter_value + dime_value) ≤ total_value → 
    q ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_max_quarters_count_l2105_210500


namespace NUMINAMATH_CALUDE_quadratic_value_theorem_l2105_210533

theorem quadratic_value_theorem (x : ℝ) : 
  x^2 - 2*x - 3 = 0 → 2*x^2 - 4*x + 12 = 18 := by
sorry

end NUMINAMATH_CALUDE_quadratic_value_theorem_l2105_210533


namespace NUMINAMATH_CALUDE_triangle_line_equations_l2105_210514

/-- Triangle ABC with vertices A(-1, 5), B(-2, -1), and C(4, 3) -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

def triangle_ABC : Triangle :=
  { A := (-1, 5)
    B := (-2, -1)
    C := (4, 3) }

/-- The equation of the line on which side AB lies -/
def line_AB : LineEquation :=
  { a := 6
    b := -1
    c := 11 }

/-- The equation of the line on which the altitude from C to AB lies -/
def altitude_C : LineEquation :=
  { a := 1
    b := 6
    c := -22 }

theorem triangle_line_equations (t : Triangle) (lab : LineEquation) (lc : LineEquation) :
  t = triangle_ABC →
  lab = line_AB →
  lc = altitude_C →
  (∀ x y : ℝ, lab.a * x + lab.b * y + lab.c = 0 ↔ (x, y) ∈ Set.Icc t.A t.B) ∧
  (∀ x y : ℝ, lc.a * x + lc.b * y + lc.c = 0 ↔ 
    (x - t.C.1) * lab.a + (y - t.C.2) * lab.b = 0) :=
sorry

end NUMINAMATH_CALUDE_triangle_line_equations_l2105_210514


namespace NUMINAMATH_CALUDE_ashutosh_completion_time_l2105_210590

theorem ashutosh_completion_time 
  (suresh_completion_time : ℝ) 
  (suresh_work_time : ℝ) 
  (ashutosh_remaining_time : ℝ) 
  (h1 : suresh_completion_time = 15)
  (h2 : suresh_work_time = 9)
  (h3 : ashutosh_remaining_time = 8)
  : ∃ (ashutosh_alone_time : ℝ), ashutosh_alone_time = 20 := by
  sorry

end NUMINAMATH_CALUDE_ashutosh_completion_time_l2105_210590


namespace NUMINAMATH_CALUDE_percentage_increase_l2105_210538

theorem percentage_increase (initial : ℝ) (final : ℝ) : initial = 1200 → final = 1680 → (final - initial) / initial * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_l2105_210538


namespace NUMINAMATH_CALUDE_hyperbola_parameters_l2105_210566

-- Define the hyperbola parameters
variable (a b : ℝ)

-- Define the conditions
def hyperbola_condition := a > 0 ∧ b > 0
def focus_condition := ∃ (y : ℝ), (6^2 : ℝ) = a^2 + b^2
def asymptote_condition := b / a = Real.sqrt 3

-- State the theorem
theorem hyperbola_parameters
  (h1 : hyperbola_condition a b)
  (h2 : focus_condition a b)
  (h3 : asymptote_condition a b) :
  a^2 = 9 ∧ b^2 = 27 := by sorry

end NUMINAMATH_CALUDE_hyperbola_parameters_l2105_210566


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2105_210515

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The given condition for the sequence -/
def sequence_condition (a : ℕ → ℝ) : Prop :=
  a 2 + 2 * a 8 + a 14 = 120

theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) (h2 : sequence_condition a) : 
  2 * a 9 - a 10 = 30 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2105_210515


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2105_210552

open Real

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > -1 → Real.log (x + 1) < x) ↔ (∃ x : ℝ, x > -1 ∧ Real.log (x + 1) ≥ x) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2105_210552


namespace NUMINAMATH_CALUDE_randy_piano_expertise_l2105_210553

/-- Represents the number of days in a year --/
def daysPerYear : ℕ := 365

/-- Represents the number of weeks in a year --/
def weeksPerYear : ℕ := 52

/-- Represents Randy's current age --/
def currentAge : ℕ := 12

/-- Represents Randy's target age to become an expert --/
def targetAge : ℕ := 20

/-- Represents the number of practice days per week --/
def practiceDaysPerWeek : ℕ := 5

/-- Represents the number of practice hours per day --/
def practiceHoursPerDay : ℕ := 5

/-- Represents the total hours needed to become an expert --/
def expertiseHours : ℕ := 10000

/-- Theorem stating that Randy can take 10 days of vacation per year and still achieve expertise --/
theorem randy_piano_expertise :
  ∃ (vacationDaysPerYear : ℕ),
    vacationDaysPerYear = 10 ∧
    (targetAge - currentAge) * weeksPerYear * practiceDaysPerWeek * practiceHoursPerDay -
    (targetAge - currentAge) * vacationDaysPerYear * practiceHoursPerDay ≥ expertiseHours :=
by sorry

end NUMINAMATH_CALUDE_randy_piano_expertise_l2105_210553


namespace NUMINAMATH_CALUDE_square_root_729_l2105_210523

theorem square_root_729 (x : ℝ) (h1 : x > 0) (h2 : x^2 = 729) : x = 27 := by
  sorry

end NUMINAMATH_CALUDE_square_root_729_l2105_210523


namespace NUMINAMATH_CALUDE_euclidean_division_123456789_by_37_l2105_210513

theorem euclidean_division_123456789_by_37 :
  ∃ (q r : ℤ), 123456789 = 37 * q + r ∧ 0 ≤ r ∧ r < 37 ∧ q = 3336669 ∧ r = 36 := by
  sorry

end NUMINAMATH_CALUDE_euclidean_division_123456789_by_37_l2105_210513


namespace NUMINAMATH_CALUDE_survey_result_l2105_210506

theorem survey_result (total : ℕ) (tv_dislike_percent : ℚ) (both_dislike_percent : ℚ)
  (h_total : total = 1800)
  (h_tv_dislike : tv_dislike_percent = 40 / 100)
  (h_both_dislike : both_dislike_percent = 25 / 100) :
  ↑⌊tv_dislike_percent * both_dislike_percent * total⌋ = 180 :=
by sorry

end NUMINAMATH_CALUDE_survey_result_l2105_210506


namespace NUMINAMATH_CALUDE_ratio_equality_l2105_210532

theorem ratio_equality (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_eq : y / (x + z) = (x - y) / z ∧ y / (x + z) = x / (y + 2*z)) : 
  x / y = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_ratio_equality_l2105_210532


namespace NUMINAMATH_CALUDE_largest_prime_factor_l2105_210571

theorem largest_prime_factor : ∃ (p : ℕ), Nat.Prime p ∧ 
  p ∣ (15^3 + 10^4 - 5^5) ∧ 
  ∀ (q : ℕ), Nat.Prime q → q ∣ (15^3 + 10^4 - 5^5) → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_l2105_210571


namespace NUMINAMATH_CALUDE_A_intersect_B_l2105_210527

-- Define set A
def A : Set ℝ := {x : ℝ | |x - 2| ≤ 1}

-- Define set B
def B : Set ℝ := {x : ℝ | x^2 - 2*x - 3 < 0}

-- Theorem stating the intersection of A and B
theorem A_intersect_B : A ∩ B = {x : ℝ | 1 ≤ x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_l2105_210527


namespace NUMINAMATH_CALUDE_negation_of_at_most_one_l2105_210583

theorem negation_of_at_most_one (P : Type → Prop) :
  (¬ (∃! x, P x)) ↔ (∃ x y, P x ∧ P y ∧ x ≠ y) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_at_most_one_l2105_210583


namespace NUMINAMATH_CALUDE_smallest_prime_ten_less_than_square_l2105_210592

theorem smallest_prime_ten_less_than_square : ∃ (n : ℕ), 
  (∀ (m : ℕ), m > 0 ∧ Nat.Prime m ∧ (∃ (k : ℕ), m = k^2 - 10) → n ≤ m) ∧
  n > 0 ∧ Nat.Prime n ∧ (∃ (k : ℕ), n = k^2 - 10) ∧ n = 71 := by
sorry

end NUMINAMATH_CALUDE_smallest_prime_ten_less_than_square_l2105_210592


namespace NUMINAMATH_CALUDE_shaded_region_perimeter_l2105_210536

/-- The perimeter of the shaded region formed by four touching circles -/
theorem shaded_region_perimeter (c : ℝ) (h : c = 48) : 
  let r := c / (2 * Real.pi)
  let arc_length := c / 4
  4 * arc_length = 48 := by
  sorry

end NUMINAMATH_CALUDE_shaded_region_perimeter_l2105_210536


namespace NUMINAMATH_CALUDE_sin_equality_solution_l2105_210569

theorem sin_equality_solution (x : Real) (h1 : x ∈ Set.Icc 0 (2 * Real.pi)) 
  (h2 : Real.sin x = Real.sin (Real.arcsin (2/3) - Real.arcsin (-1/3))) : 
  x = Real.arcsin ((4 * Real.sqrt 2 + Real.sqrt 5) / 9) ∨ 
  x = Real.pi - Real.arcsin ((4 * Real.sqrt 2 + Real.sqrt 5) / 9) := by
  sorry

end NUMINAMATH_CALUDE_sin_equality_solution_l2105_210569


namespace NUMINAMATH_CALUDE_coffee_consumption_l2105_210505

def vacation_duration : ℕ := 40
def pods_per_box : ℕ := 30
def cost_per_box : ℚ := 8
def total_spent : ℚ := 32

def cups_per_day : ℚ := total_spent / cost_per_box * pods_per_box / vacation_duration

theorem coffee_consumption : cups_per_day = 3 := by sorry

end NUMINAMATH_CALUDE_coffee_consumption_l2105_210505


namespace NUMINAMATH_CALUDE_smallest_n_for_exact_tax_l2105_210588

theorem smallest_n_for_exact_tax : ∃ (x : ℕ), (105 * x) % 10000 = 0 ∧ 
  (∀ (y : ℕ), y < 21 → (105 * y) % 10000 ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_exact_tax_l2105_210588


namespace NUMINAMATH_CALUDE_lidia_app_purchase_l2105_210575

/-- Proves that Lidia will be left with $15 after purchasing apps with a discount --/
theorem lidia_app_purchase (app_cost : ℝ) (num_apps : ℕ) (budget : ℝ) (discount_rate : ℝ) :
  app_cost = 4 →
  num_apps = 15 →
  budget = 66 →
  discount_rate = 0.15 →
  budget - (num_apps * app_cost * (1 - discount_rate)) = 15 :=
by
  sorry

#check lidia_app_purchase

end NUMINAMATH_CALUDE_lidia_app_purchase_l2105_210575


namespace NUMINAMATH_CALUDE_orthogonal_vectors_l2105_210507

/-- Two vectors are orthogonal if and only if their dot product is zero -/
def orthogonal (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

/-- The problem statement -/
theorem orthogonal_vectors (x : ℝ) :
  let a : ℝ × ℝ := (1, -2)
  let b : ℝ × ℝ := (-3, x)
  orthogonal a b ↔ x = -3/2 := by
sorry

end NUMINAMATH_CALUDE_orthogonal_vectors_l2105_210507


namespace NUMINAMATH_CALUDE_union_of_M_and_P_l2105_210526

-- Define the sets M and P
def M : Set ℝ := {x : ℝ | -2 < x ∧ x < 1}
def P : Set ℝ := {x : ℝ | -2 ≤ x ∧ x < 2}

-- State the theorem
theorem union_of_M_and_P : M ∪ P = {x : ℝ | -2 ≤ x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_P_l2105_210526


namespace NUMINAMATH_CALUDE_linear_function_problem_l2105_210520

-- Define a linear function
def LinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ m b : ℝ, ∀ x, f x = m * x + b

-- State the theorem
theorem linear_function_problem (f : ℝ → ℝ) 
  (h_linear : LinearFunction f)
  (h_diff : f 10 - f 5 = 20)
  (h_f0 : f 0 = 3) :
  f 15 - f 5 = 40 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_problem_l2105_210520


namespace NUMINAMATH_CALUDE_sector_area_l2105_210534

/-- Given a sector with perimeter 10 and central angle 2 radians, its area is 25/4 -/
theorem sector_area (r : ℝ) (l : ℝ) (h1 : l + 2*r = 10) (h2 : l = 2*r) : 
  (1/2) * r * l = 25/4 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l2105_210534


namespace NUMINAMATH_CALUDE_min_quotient_base12_number_l2105_210570

/-- Represents a digit in base 12, ranging from 1 to 10 (in base 10) -/
def Digit12 := {d : ℕ // 1 ≤ d ∧ d ≤ 10}

/-- Converts a base 12 number to base 10 -/
def toBase10 (a b c : Digit12) : ℕ :=
  144 * a.val + 12 * b.val + c.val

/-- Calculates the sum of digits in base 10 -/
def digitSum (a b c : Digit12) : ℕ :=
  a.val + b.val + c.val

/-- The main theorem stating the minimum quotient -/
theorem min_quotient_base12_number :
  ∀ a b c : Digit12,
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  (toBase10 a b c : ℚ) / (digitSum a b c) ≥ 24.5 :=
sorry

end NUMINAMATH_CALUDE_min_quotient_base12_number_l2105_210570
