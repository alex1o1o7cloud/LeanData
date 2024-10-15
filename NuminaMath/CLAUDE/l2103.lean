import Mathlib

namespace NUMINAMATH_CALUDE_election_majority_l2103_210308

theorem election_majority (total_votes : ℕ) (winning_percentage : ℚ) : 
  total_votes = 6900 →
  winning_percentage = 60 / 100 →
  (winning_percentage * total_votes : ℚ).floor - ((1 - winning_percentage) * total_votes : ℚ).floor = 1380 :=
by sorry

end NUMINAMATH_CALUDE_election_majority_l2103_210308


namespace NUMINAMATH_CALUDE_expected_sixes_is_one_third_l2103_210350

/-- The probability of rolling a 6 on a standard die -/
def prob_six : ℚ := 1 / 6

/-- The probability of not rolling a 6 on a standard die -/
def prob_not_six : ℚ := 1 - prob_six

/-- The expected number of 6's when rolling two standard dice -/
def expected_sixes : ℚ := 
  0 * (prob_not_six ^ 2) + 
  1 * (2 * prob_six * prob_not_six) + 
  2 * (prob_six ^ 2)

theorem expected_sixes_is_one_third : expected_sixes = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expected_sixes_is_one_third_l2103_210350


namespace NUMINAMATH_CALUDE_special_function_form_l2103_210353

/-- A function satisfying the given conditions -/
structure SpecialFunction where
  f : ℝ → ℝ
  differentiable : Differentiable ℝ f
  f_zero_eq_one : f 0 = 1
  f_inequality : ∀ x₁ x₂, f (x₁ + x₂) ≥ f x₁ * f x₂

/-- The main theorem: any function satisfying the given conditions is of the form e^(kx) -/
theorem special_function_form (φ : SpecialFunction) :
  ∃ k : ℝ, ∀ x, φ.f x = Real.exp (k * x) := by
  sorry

end NUMINAMATH_CALUDE_special_function_form_l2103_210353


namespace NUMINAMATH_CALUDE_main_theorem_l2103_210332

-- Define the type for multiplicative functions
def MultFun := ℕ → Fin 2

-- Define the property of being multiplicative
def is_multiplicative (f : MultFun) : Prop :=
  ∀ a b : ℕ, f (a * b) = f a * f b

theorem main_theorem (a b c d : ℕ) (f g : MultFun)
  (h1 : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0)
  (h2 : a * b * c * d ≠ 1)
  (h3 : Nat.gcd a b = 1 ∧ Nat.gcd a c = 1 ∧ Nat.gcd a d = 1 ∧
        Nat.gcd b c = 1 ∧ Nat.gcd b d = 1 ∧ Nat.gcd c d = 1)
  (h4 : is_multiplicative f ∧ is_multiplicative g)
  (h5 : ∀ n : ℕ, f (a * n + b) = g (c * n + d)) :
  (∀ n : ℕ, f (a * n + b) = 0 ∧ g (c * n + d) = 0) ∨
  (∃ k : ℕ, k > 0 ∧ ∀ n : ℕ, Nat.gcd n k = 1 → f n = 1 ∧ g n = 1) :=
by sorry

end NUMINAMATH_CALUDE_main_theorem_l2103_210332


namespace NUMINAMATH_CALUDE_g_g_two_roots_l2103_210382

/-- The function g(x) defined as x^2 + 2x + c^2 -/
def g (c : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + c^2

/-- The theorem stating that g(g(x)) has exactly two distinct real roots iff c = ±1 -/
theorem g_g_two_roots (c : ℝ) :
  (∃! (r₁ r₂ : ℝ), r₁ ≠ r₂ ∧ ∀ x, g c (g c x) = 0 ↔ x = r₁ ∨ x = r₂) ↔ c = 1 ∨ c = -1 :=
sorry

end NUMINAMATH_CALUDE_g_g_two_roots_l2103_210382


namespace NUMINAMATH_CALUDE_integral_x_squared_l2103_210347

theorem integral_x_squared : ∫ x in (0:ℝ)..(1:ℝ), x^2 = (1:ℝ)/3 := by sorry

end NUMINAMATH_CALUDE_integral_x_squared_l2103_210347


namespace NUMINAMATH_CALUDE_tournament_claim_inconsistency_l2103_210328

/-- Represents a single-elimination tournament -/
structure Tournament :=
  (participants : ℕ)
  (games_played : ℕ)

/-- Calculates the number of games in a single-elimination tournament -/
def games_in_tournament (t : Tournament) : ℕ := t.participants - 1

/-- Represents the claim made by some players -/
structure Claim :=
  (num_players : ℕ)
  (games_per_player : ℕ)

/-- Calculates the minimum number of games implied by a claim -/
def min_games_from_claim (c : Claim) : ℕ :=
  c.num_players * (c.games_per_player - 1)

/-- The main theorem -/
theorem tournament_claim_inconsistency (t : Tournament) (c : Claim) 
  (h1 : t.participants = 18)
  (h2 : c.num_players = 6)
  (h3 : c.games_per_player = 4) :
  min_games_from_claim c > games_in_tournament t :=
by sorry

end NUMINAMATH_CALUDE_tournament_claim_inconsistency_l2103_210328


namespace NUMINAMATH_CALUDE_minotaur_returns_l2103_210376

/-- A room in the Minotaur's palace -/
structure Room where
  id : Nat

/-- A direction the Minotaur can turn -/
inductive Direction
  | Left
  | Right

/-- The state of the Minotaur's journey -/
structure State where
  room : Room
  enteredThrough : Nat
  nextTurn : Direction

/-- The palace with its room connections -/
structure Palace where
  rooms : Finset Room
  connections : Room → Finset (Nat × Room)
  room_count : rooms.card = 1000000
  three_corridors : ∀ r : Room, (connections r).card = 3

/-- The function that determines the next state based on the current state -/
def nextState (p : Palace) (s : State) : State :=
  sorry

/-- The theorem stating that the Minotaur will eventually return to the starting room -/
theorem minotaur_returns (p : Palace) (start : State) :
  ∃ n : Nat, (Nat.iterate (nextState p) n start).room = start.room :=
sorry

end NUMINAMATH_CALUDE_minotaur_returns_l2103_210376


namespace NUMINAMATH_CALUDE_largest_non_expressible_not_expressible_83_largest_non_expressible_is_83_l2103_210337

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def is_expressible (n : ℕ) : Prop :=
  ∃ k c, k > 0 ∧ is_composite c ∧ n = 36 * k + c

theorem largest_non_expressible : ∀ n : ℕ, n > 83 → is_expressible n :=
  sorry

theorem not_expressible_83 : ¬ is_expressible 83 :=
  sorry

theorem largest_non_expressible_is_83 :
  (∀ n : ℕ, n > 83 → is_expressible n) ∧ ¬ is_expressible 83 :=
  sorry

end NUMINAMATH_CALUDE_largest_non_expressible_not_expressible_83_largest_non_expressible_is_83_l2103_210337


namespace NUMINAMATH_CALUDE_apartment_price_ratio_l2103_210342

theorem apartment_price_ratio :
  ∀ (a b : ℝ),
  a > 0 → b > 0 →
  1.21 * a + 1.11 * b = 1.15 * (a + b) →
  b / a = 1.5 := by
sorry

end NUMINAMATH_CALUDE_apartment_price_ratio_l2103_210342


namespace NUMINAMATH_CALUDE_imaginary_unit_sum_l2103_210367

theorem imaginary_unit_sum : ∃ i : ℂ, i * i = -1 ∧ i + i^2 + i^3 = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_unit_sum_l2103_210367


namespace NUMINAMATH_CALUDE_linear_equation_system_l2103_210304

theorem linear_equation_system (a b c : ℝ) 
  (eq1 : a + 2*b - 3*c = 4)
  (eq2 : 5*a - 6*b + 7*c = 8) :
  9*a + 2*b - 5*c = 24 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_system_l2103_210304


namespace NUMINAMATH_CALUDE_alternating_arrangements_2_3_l2103_210355

/-- The number of ways to arrange m men and w women in a row, such that no two men or two women are adjacent -/
def alternating_arrangements (m : ℕ) (w : ℕ) : ℕ := sorry

theorem alternating_arrangements_2_3 :
  alternating_arrangements 2 3 = 24 := by sorry

end NUMINAMATH_CALUDE_alternating_arrangements_2_3_l2103_210355


namespace NUMINAMATH_CALUDE_sum_squares_consecutive_even_numbers_l2103_210339

/-- Given 6 consecutive even numbers with a sum of 72, prove that the sum of their squares is 1420 -/
theorem sum_squares_consecutive_even_numbers :
  ∀ (a : ℕ), 
  (∃ (n : ℕ), a = 2*n) →  -- a is even
  (a + (a + 2) + (a + 4) + (a + 6) + (a + 8) + (a + 10) = 72) →  -- sum is 72
  (a^2 + (a + 2)^2 + (a + 4)^2 + (a + 6)^2 + (a + 8)^2 + (a + 10)^2 = 1420) :=
by sorry


end NUMINAMATH_CALUDE_sum_squares_consecutive_even_numbers_l2103_210339


namespace NUMINAMATH_CALUDE_simplified_ratio_l2103_210371

def sarah_apples : ℕ := 45
def brother_apples : ℕ := 9
def cousin_apples : ℕ := 27

def gcd_three (a b c : ℕ) : ℕ := Nat.gcd a (Nat.gcd b c)

theorem simplified_ratio :
  let common_divisor := gcd_three sarah_apples brother_apples cousin_apples
  (sarah_apples / common_divisor : ℕ) = 5 ∧
  (brother_apples / common_divisor : ℕ) = 1 ∧
  (cousin_apples / common_divisor : ℕ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_simplified_ratio_l2103_210371


namespace NUMINAMATH_CALUDE_max_median_redistribution_l2103_210300

theorem max_median_redistribution (x : ℕ) :
  let initial_amounts : List ℕ := [28, 72, 98, x]
  let total : ℕ := initial_amounts.sum
  let redistributed : ℚ := (total : ℚ) / 4
  (∀ (a : ℕ), a ∈ initial_amounts → (a : ℚ) ≤ redistributed) →
  redistributed ≤ 98 →
  x ≤ 194 →
  (x = 194 → redistributed = 98) :=
by sorry

end NUMINAMATH_CALUDE_max_median_redistribution_l2103_210300


namespace NUMINAMATH_CALUDE_class_size_l2103_210388

theorem class_size (total_average : ℝ) (group1_size : ℕ) (group1_average : ℝ)
                   (group2_size : ℕ) (group2_average : ℝ) (last_student_age : ℕ) :
  total_average = 15 →
  group1_size = 5 →
  group1_average = 12 →
  group2_size = 9 →
  group2_average = 16 →
  last_student_age = 21 →
  ∃ (n : ℕ), n = 15 ∧ n * total_average = group1_size * group1_average + group2_size * group2_average + last_student_age :=
by
  sorry

#check class_size

end NUMINAMATH_CALUDE_class_size_l2103_210388


namespace NUMINAMATH_CALUDE_tan_seventeen_pi_over_four_l2103_210335

theorem tan_seventeen_pi_over_four : Real.tan (17 * π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_seventeen_pi_over_four_l2103_210335


namespace NUMINAMATH_CALUDE_parallel_condition_l2103_210399

/-- Two lines in R² are parallel if and only if their slopes are equal -/
def are_parallel (a b c d e f : ℝ) : Prop :=
  (a * f = b * d) ∧ (a * e ≠ b * c ∨ c * f ≠ d * e)

/-- The condition for two lines to be parallel -/
theorem parallel_condition (a : ℝ) :
  (∀ x y : ℝ, are_parallel a 1 (-1) 1 a 1) ↔ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_condition_l2103_210399


namespace NUMINAMATH_CALUDE_data_set_average_l2103_210386

theorem data_set_average (x : ℝ) : 
  (2 + 1 + 4 + x + 6) / 5 = 4 → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_data_set_average_l2103_210386


namespace NUMINAMATH_CALUDE_line_perpendicular_to_plane_l2103_210319

-- Define the types for our geometric objects
variable (Point Line Plane : Type)

-- Define the relations we need
variable (lies_in : Line → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)
variable (intersection_line : Plane → Plane → Line)

-- State the theorem
theorem line_perpendicular_to_plane 
  (α β : Plane) (a c : Line) :
  perpendicular_planes α β →
  lies_in a α →
  c = intersection_line α β →
  perpendicular_lines a c →
  perpendicular_line_plane a β :=
sorry

end NUMINAMATH_CALUDE_line_perpendicular_to_plane_l2103_210319


namespace NUMINAMATH_CALUDE_carnation_tulip_difference_l2103_210333

theorem carnation_tulip_difference :
  let carnations : ℕ := 13
  let tulips : ℕ := 7
  carnations - tulips = 6 :=
by sorry

end NUMINAMATH_CALUDE_carnation_tulip_difference_l2103_210333


namespace NUMINAMATH_CALUDE_absolute_value_equation_l2103_210313

theorem absolute_value_equation (y : ℝ) :
  (|y - 25| + |y - 23| = |2*y - 46|) → y = 24 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_l2103_210313


namespace NUMINAMATH_CALUDE_sqrt_factorial_squared_l2103_210397

-- Define factorial function
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- State the theorem
theorem sqrt_factorial_squared :
  (((factorial 5 * factorial 4 : ℕ) : ℝ).sqrt ^ 2 : ℝ) = 2880 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_factorial_squared_l2103_210397


namespace NUMINAMATH_CALUDE_racing_game_cost_l2103_210318

/-- The cost of the racing game given the total spent and the cost of the basketball game -/
theorem racing_game_cost (total_spent basketball_cost : ℚ) 
  (h1 : total_spent = 9.43)
  (h2 : basketball_cost = 5.2) : 
  total_spent - basketball_cost = 4.23 := by
  sorry

end NUMINAMATH_CALUDE_racing_game_cost_l2103_210318


namespace NUMINAMATH_CALUDE_a_lt_2_necessary_not_sufficient_for_a_sq_lt_4_l2103_210301

theorem a_lt_2_necessary_not_sufficient_for_a_sq_lt_4 :
  (∀ a : ℝ, a^2 < 4 → a < 2) ∧
  (∃ a : ℝ, a < 2 ∧ a^2 ≥ 4) := by
  sorry

end NUMINAMATH_CALUDE_a_lt_2_necessary_not_sufficient_for_a_sq_lt_4_l2103_210301


namespace NUMINAMATH_CALUDE_smallest_block_size_l2103_210392

/-- 
Given a rectangular block with dimensions a × b × c formed by N congruent 1-cm cubes,
where (a-1)(b-1)(c-1) = 252, the smallest possible value of N is 224.
-/
theorem smallest_block_size (a b c N : ℕ) : 
  (a - 1) * (b - 1) * (c - 1) = 252 → 
  N = a * b * c → 
  (∀ a' b' c' N', (a' - 1) * (b' - 1) * (c' - 1) = 252 → N' = a' * b' * c' → N ≤ N') →
  N = 224 :=
by sorry

end NUMINAMATH_CALUDE_smallest_block_size_l2103_210392


namespace NUMINAMATH_CALUDE_log_equation_solution_l2103_210316

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  Real.log x / Real.log 3 + Real.log x / Real.log 9 = 5 → x = 3^(10/3) := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l2103_210316


namespace NUMINAMATH_CALUDE_set_operations_and_subset_l2103_210374

open Set

def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | x > 2}
def C (a : ℝ) : Set ℝ := {x | 1 < x ∧ x < a}

theorem set_operations_and_subset :
  (A ∪ B = {x | 2 < x ∧ x ≤ 3}) ∧
  ((Bᶜ) ∩ A = {x | 1 ≤ x ∧ x ≤ 2}) ∧
  (∀ a : ℝ, C a ⊆ A → a ≤ 3) := by sorry

end NUMINAMATH_CALUDE_set_operations_and_subset_l2103_210374


namespace NUMINAMATH_CALUDE_problem_statement_l2103_210354

theorem problem_statement : (-1)^49 + 2^(3^3 + 5^2 - 48^2) = -1 + 1 / 2^2252 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2103_210354


namespace NUMINAMATH_CALUDE_depth_of_iron_cone_in_mercury_l2103_210375

/-- The depth of submersion of an iron cone in mercury -/
noncomputable def depth_of_submersion (cone_volume : ℝ) (iron_density : ℝ) (mercury_density : ℝ) : ℝ :=
  let submerged_volume := (iron_density * cone_volume) / mercury_density
  (3 * submerged_volume / Real.pi) ^ (1/3)

/-- The theorem stating the depth of submersion for the given problem -/
theorem depth_of_iron_cone_in_mercury :
  let cone_volume : ℝ := 350
  let iron_density : ℝ := 7.2
  let mercury_density : ℝ := 13.6
  abs (depth_of_submersion cone_volume iron_density mercury_density - 5.6141) < 0.0001 := by
  sorry


end NUMINAMATH_CALUDE_depth_of_iron_cone_in_mercury_l2103_210375


namespace NUMINAMATH_CALUDE_simplify_fraction_l2103_210363

theorem simplify_fraction : 
  (1 / ((1 / (Real.sqrt 3 + 1)) + (3 / (Real.sqrt 5 - 2)))) = 
  (2 / (Real.sqrt 3 + 6 * Real.sqrt 5 + 11)) := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2103_210363


namespace NUMINAMATH_CALUDE_jelly_bean_count_l2103_210383

/-- The number of jelly beans in the jar. -/
def total_jelly_beans : ℕ := 200

/-- Thomas's share of jelly beans as a fraction. -/
def thomas_share : ℚ := 1/10

/-- The ratio of Barry's share to Emmanuel's share. -/
def barry_emmanuel_ratio : ℚ := 4/5

/-- Emmanuel's share of jelly beans. -/
def emmanuel_share : ℕ := 100

/-- Theorem stating the total number of jelly beans in the jar. -/
theorem jelly_bean_count :
  total_jelly_beans = 200 ∧
  thomas_share = 1/10 ∧
  barry_emmanuel_ratio = 4/5 ∧
  emmanuel_share = 100 ∧
  emmanuel_share = (5/9 : ℚ) * ((1 - thomas_share) * total_jelly_beans) :=
by sorry

end NUMINAMATH_CALUDE_jelly_bean_count_l2103_210383


namespace NUMINAMATH_CALUDE_farmer_picked_thirty_today_l2103_210315

/-- Represents the number of tomatoes picked today by a farmer -/
def tomatoes_picked_today (initial : ℕ) (picked_yesterday : ℕ) (left_after_today : ℕ) : ℕ :=
  initial - picked_yesterday - left_after_today

/-- Theorem stating that the farmer picked 30 tomatoes today -/
theorem farmer_picked_thirty_today :
  tomatoes_picked_today 171 134 7 = 30 := by
  sorry

end NUMINAMATH_CALUDE_farmer_picked_thirty_today_l2103_210315


namespace NUMINAMATH_CALUDE_symmetric_line_equation_l2103_210356

/-- A line in the xy-plane represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The equation of a line in slope-intercept form -/
def Line.equation (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.intercept

/-- Two lines are symmetric with respect to the y-axis -/
def symmetric_about_y_axis (l₁ l₂ : Line) : Prop :=
  l₁.slope = -l₂.slope ∧ l₁.intercept = l₂.intercept

theorem symmetric_line_equation (l₁ l₂ : Line) :
  l₁.equation x y = (y = 2 * x + 3) →
  symmetric_about_y_axis l₁ l₂ →
  l₂.equation x y = (y = -2 * x + 3) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_line_equation_l2103_210356


namespace NUMINAMATH_CALUDE_s_1000_eq_720_l2103_210348

def s : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => 
    if n % 2 = 0 then s (n / 2)
    else if (n - 1) % 4 = 0 then s ((n - 1) / 2 + 1)
    else s ((n + 1) / 2 - 1) + (s ((n + 1) / 2 - 1))^2 / s ((n + 1) / 4 - 1)

theorem s_1000_eq_720 : s 1000 = 720 := by
  sorry

end NUMINAMATH_CALUDE_s_1000_eq_720_l2103_210348


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l2103_210378

theorem perfect_square_trinomial (m : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 + m*x + 25 = (x + a)^2) → m = 10 ∨ m = -10 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l2103_210378


namespace NUMINAMATH_CALUDE_value_of_a_minus_b_l2103_210379

theorem value_of_a_minus_b (a b : ℝ) 
  (ha : |a| = 5) 
  (hb : |b| = 4) 
  (hab : a + b < 0) : 
  a - b = -9 ∨ a - b = -1 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_minus_b_l2103_210379


namespace NUMINAMATH_CALUDE_ordering_of_trig_functions_l2103_210373

theorem ordering_of_trig_functions (a b c d : ℝ) : 
  a = Real.sin (Real.cos (2015 * π / 180)) →
  b = Real.sin (Real.sin (2015 * π / 180)) →
  c = Real.cos (Real.sin (2015 * π / 180)) →
  d = Real.cos (Real.cos (2015 * π / 180)) →
  c > d ∧ d > b ∧ b > a := by sorry

end NUMINAMATH_CALUDE_ordering_of_trig_functions_l2103_210373


namespace NUMINAMATH_CALUDE_simplify_expression_l2103_210306

theorem simplify_expression (a : ℝ) : 6*a - 5*a + 4*a - 3*a + 2*a - a = 3*a := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2103_210306


namespace NUMINAMATH_CALUDE_vector_b_value_l2103_210391

/-- Given two vectors a and b in ℝ³, prove that b equals (-2, 4, -2) -/
theorem vector_b_value (a b : ℝ × ℝ × ℝ) : 
  a = (1, -2, 1) → a + b = (-1, 2, -1) → b = (-2, 4, -2) := by
  sorry

end NUMINAMATH_CALUDE_vector_b_value_l2103_210391


namespace NUMINAMATH_CALUDE_indira_cricket_time_l2103_210377

/-- Sean's daily cricket playing time in minutes -/
def sean_daily_time : ℕ := 50

/-- Number of days Sean played cricket -/
def sean_days : ℕ := 14

/-- Total time Sean and Indira played cricket together in minutes -/
def total_time : ℕ := 1512

/-- Calculate Indira's cricket playing time -/
def indira_time : ℕ := total_time - (sean_daily_time * sean_days)

/-- Theorem stating Indira's cricket playing time -/
theorem indira_cricket_time : indira_time = 812 := by sorry

end NUMINAMATH_CALUDE_indira_cricket_time_l2103_210377


namespace NUMINAMATH_CALUDE_sum_of_digits_l2103_210362

/-- Given a three-digit number of the form 3a7 and another three-digit number 7c1,
    where a and c are single digits, prove that if 3a7 + 414 = 7c1 and 7c1 is
    divisible by 11, then a + c = 14. -/
theorem sum_of_digits (a c : ℕ) : 
  (a < 10) →
  (c < 10) →
  (300 + 10 * a + 7 + 414 = 700 + 10 * c + 1) →
  (700 + 10 * c + 1) % 11 = 0 →
  a + c = 14 := by
sorry

end NUMINAMATH_CALUDE_sum_of_digits_l2103_210362


namespace NUMINAMATH_CALUDE_base_subtraction_l2103_210305

/-- Convert a number from base b to base 10 -/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.foldr (fun d acc => d + b * acc) 0

/-- Express 343₈ - 265₇ as a base 10 integer -/
theorem base_subtraction : 
  let base_8_num := to_base_10 [3, 4, 3] 8
  let base_7_num := to_base_10 [2, 6, 5] 7
  base_8_num - base_7_num = 82 := by
sorry

end NUMINAMATH_CALUDE_base_subtraction_l2103_210305


namespace NUMINAMATH_CALUDE_translated_parabola_vertex_l2103_210334

/-- The original parabola function -/
def f (x : ℝ) : ℝ := x^2 - 4*x + 3

/-- The translation amount to the right -/
def translation : ℝ := 2

/-- The new parabola function after translation -/
def f_translated (x : ℝ) : ℝ := f (x - translation)

/-- Theorem stating the coordinates of the vertex of the translated parabola -/
theorem translated_parabola_vertex :
  ∃ (x y : ℝ), x = 4 ∧ y = -1 ∧
  ∀ (t : ℝ), f_translated t ≥ f_translated x :=
sorry

end NUMINAMATH_CALUDE_translated_parabola_vertex_l2103_210334


namespace NUMINAMATH_CALUDE_yoongi_has_second_largest_number_l2103_210368

/-- Represents a student with their assigned number -/
structure Student where
  name : String
  number : Nat

/-- Checks if a student has the second largest number among a list of students -/
def hasSecondLargestNumber (s : Student) (students : List Student) : Prop :=
  ∃ (larger smaller : Student),
    larger ∈ students ∧
    smaller ∈ students ∧
    s ∈ students ∧
    larger.number > s.number ∧
    s.number > smaller.number ∧
    ∀ (other : Student), other ∈ students → other.number ≤ larger.number

theorem yoongi_has_second_largest_number :
  let yoongi := Student.mk "Yoongi" 7
  let jungkook := Student.mk "Jungkook" 6
  let yuna := Student.mk "Yuna" 9
  let students := [yoongi, jungkook, yuna]
  hasSecondLargestNumber yoongi students := by
  sorry

end NUMINAMATH_CALUDE_yoongi_has_second_largest_number_l2103_210368


namespace NUMINAMATH_CALUDE_price_per_deck_l2103_210361

def initial_decks : ℕ := 5
def remaining_decks : ℕ := 3
def total_earnings : ℕ := 4

theorem price_per_deck :
  (total_earnings : ℚ) / (initial_decks - remaining_decks) = 2 := by
  sorry

end NUMINAMATH_CALUDE_price_per_deck_l2103_210361


namespace NUMINAMATH_CALUDE_election_result_theorem_l2103_210325

/-- Represents the result of a mayoral election. -/
structure ElectionResult where
  total_votes : ℕ
  candidates : ℕ
  winner_votes : ℕ
  second_place_votes : ℕ
  third_place_votes : ℕ
  fourth_place_votes : ℕ
  winner_third_diff : ℕ
  winner_fourth_diff : ℕ

/-- Theorem stating the conditions and the result to be proved. -/
theorem election_result_theorem (e : ElectionResult) 
  (h1 : e.total_votes = 979)
  (h2 : e.candidates = 4)
  (h3 : e.winner_votes = e.fourth_place_votes + e.winner_fourth_diff)
  (h4 : e.winner_votes = e.third_place_votes + e.winner_third_diff)
  (h5 : e.fourth_place_votes = 199)
  (h6 : e.winner_fourth_diff = 105)
  (h7 : e.winner_third_diff = 79)
  (h8 : e.total_votes = e.winner_votes + e.second_place_votes + e.third_place_votes + e.fourth_place_votes) :
  e.winner_votes - e.second_place_votes = 53 := by
  sorry


end NUMINAMATH_CALUDE_election_result_theorem_l2103_210325


namespace NUMINAMATH_CALUDE_simplify_expression_l2103_210314

theorem simplify_expression (y : ℝ) : 3 * y + 4.5 * y + 7 * y = 14.5 * y := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2103_210314


namespace NUMINAMATH_CALUDE_range_of_x_l2103_210307

theorem range_of_x (a : ℝ) (h1 : a > 1) :
  {x : ℝ | a^(2*x + 1) > (1/a)^(2*x)} = {x : ℝ | x > -1/4} := by sorry

end NUMINAMATH_CALUDE_range_of_x_l2103_210307


namespace NUMINAMATH_CALUDE_expression_equality_l2103_210338

theorem expression_equality : (1 + 0.25) / (2 * (3/4) - 0.75) + (3 * 0.5) / (1.5 + 3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2103_210338


namespace NUMINAMATH_CALUDE_xyz_product_l2103_210320

theorem xyz_product (x y z : ℂ) 
  (eq1 : x * y + 5 * y = -25)
  (eq2 : y * z + 5 * z = -25)
  (eq3 : z * x + 5 * x = -25) :
  x * y * z = 125 := by
sorry

end NUMINAMATH_CALUDE_xyz_product_l2103_210320


namespace NUMINAMATH_CALUDE_prob_odd_then_even_eq_17_45_l2103_210329

/-- A box containing 6 cards numbered 1 to 6 -/
def Box : Finset ℕ := {1, 2, 3, 4, 5, 6}

/-- The probability of drawing a specific card from the box -/
def prob_draw (n : ℕ) : ℚ := if n ∈ Box then 1 / 6 else 0

/-- The probability of drawing an even number from the remaining cards after drawing 'a' -/
def prob_even_after (a : ℕ) : ℚ :=
  let remaining := Box.filter (λ x => x > a)
  let even_remaining := remaining.filter (λ x => x % 2 = 0)
  (even_remaining.card : ℚ) / remaining.card

/-- The probability of the event: first draw is odd and second draw is even -/
def prob_odd_then_even : ℚ :=
  (prob_draw 1 * prob_even_after 1) +
  (prob_draw 3 * prob_even_after 3) +
  (prob_draw 5 * prob_even_after 5)

theorem prob_odd_then_even_eq_17_45 : prob_odd_then_even = 17 / 45 := by
  sorry

end NUMINAMATH_CALUDE_prob_odd_then_even_eq_17_45_l2103_210329


namespace NUMINAMATH_CALUDE_incorrect_statement_l2103_210349

theorem incorrect_statement : ¬(
  (∀ x : ℝ, x ∈ [0, 1] → Real.exp x ≥ 1) ∧
  (∃ x : ℝ, x^2 + x + 1 < 0)
) := by sorry

end NUMINAMATH_CALUDE_incorrect_statement_l2103_210349


namespace NUMINAMATH_CALUDE_all_equations_are_equalities_negative_two_solves_equation_one_and_negative_two_solve_equation_l2103_210309

-- Define what it means for a real number to be a solution to an equation
def IsSolution (x : ℝ) (f : ℝ → ℝ) : Prop := f x = 0

-- All equations are equalities
theorem all_equations_are_equalities : ∀ (f : ℝ → ℝ), ∃ (x : ℝ), f x = 0 → (∃ (y : ℝ), f x = f y) :=
sorry

-- -2 is a solution to 3 - 2x = 7
theorem negative_two_solves_equation : IsSolution (-2 : ℝ) (λ x => 3 - 2*x - 7) :=
sorry

-- 1 and -2 are solutions to (x - 1)(x + 2) = 0
theorem one_and_negative_two_solve_equation : 
  IsSolution (1 : ℝ) (λ x => (x - 1)*(x + 2)) ∧ 
  IsSolution (-2 : ℝ) (λ x => (x - 1)*(x + 2)) :=
sorry

end NUMINAMATH_CALUDE_all_equations_are_equalities_negative_two_solves_equation_one_and_negative_two_solve_equation_l2103_210309


namespace NUMINAMATH_CALUDE_arithmetic_evaluation_l2103_210365

theorem arithmetic_evaluation : 1537 + 180 / 60 * 15 - 237 = 1345 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_evaluation_l2103_210365


namespace NUMINAMATH_CALUDE_elevator_theorem_l2103_210326

/-- Represents the elevator system described in the problem -/
structure ElevatorSystem where
  /-- The probability of moving up on the nth press is current_floor / (n-1) -/
  move_up_prob : (current_floor : ℕ) → (n : ℕ) → ℚ
  move_up_prob_def : ∀ (current_floor n : ℕ), move_up_prob current_floor n = current_floor / (n - 1)

/-- The expected number of pairs of consecutive presses that both move up -/
def expected_consecutive_up_pairs (system : ElevatorSystem) (start_press end_press : ℕ) : ℚ :=
  sorry

/-- The main theorem to prove -/
theorem elevator_theorem (system : ElevatorSystem) :
  expected_consecutive_up_pairs system 3 100 = 97 / 3 := by sorry

end NUMINAMATH_CALUDE_elevator_theorem_l2103_210326


namespace NUMINAMATH_CALUDE_didi_fundraiser_amount_l2103_210344

/-- Calculates the total amount raised from cake sales and donations --/
def total_amount_raised (num_cakes : ℕ) (slices_per_cake : ℕ) (price_per_slice : ℚ) 
  (donation1_per_slice : ℚ) (donation2_per_slice : ℚ) : ℚ :=
  let total_slices := num_cakes * slices_per_cake
  let sales_amount := total_slices * price_per_slice
  let donation1_amount := total_slices * donation1_per_slice
  let donation2_amount := total_slices * donation2_per_slice
  sales_amount + donation1_amount + donation2_amount

/-- Theorem stating that under the given conditions, the total amount raised is $140 --/
theorem didi_fundraiser_amount :
  total_amount_raised 10 8 1 (1/2) (1/4) = 140 := by
  sorry

end NUMINAMATH_CALUDE_didi_fundraiser_amount_l2103_210344


namespace NUMINAMATH_CALUDE_blue_packs_bought_l2103_210311

/- Define the problem parameters -/
def white_pack_size : ℕ := 6
def blue_pack_size : ℕ := 9
def white_packs_bought : ℕ := 5
def total_tshirts : ℕ := 57

/- Define the theorem -/
theorem blue_packs_bought :
  ∃ (blue_packs : ℕ),
    blue_packs * blue_pack_size + white_packs_bought * white_pack_size = total_tshirts ∧
    blue_packs = 3 := by
  sorry

end NUMINAMATH_CALUDE_blue_packs_bought_l2103_210311


namespace NUMINAMATH_CALUDE_aaron_age_proof_l2103_210317

def has_all_digits (n : ℕ) : Prop :=
  ∀ d : ℕ, d < 10 → ∃ k : ℕ, n / 10^k % 10 = d

theorem aaron_age_proof :
  ∃! m : ℕ,
    1000 ≤ m^3 ∧ m^3 < 10000 ∧
    100000 ≤ m^4 ∧ m^4 < 1000000 ∧
    has_all_digits (m^3 + m^4) ∧
    m = 18 := by
  sorry

end NUMINAMATH_CALUDE_aaron_age_proof_l2103_210317


namespace NUMINAMATH_CALUDE_quadratic_root_product_l2103_210359

theorem quadratic_root_product (x₁ x₂ : ℝ) : 
  (x₁^2 - 4*x₁ - 2 = 0) → 
  (x₂^2 - 4*x₂ - 2 = 0) → 
  x₁ * x₂ = -2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_product_l2103_210359


namespace NUMINAMATH_CALUDE_parking_lot_car_decrease_l2103_210360

theorem parking_lot_car_decrease (initial_cars : ℕ) (cars_out : ℕ) (cars_in : ℕ) : 
  initial_cars = 25 → cars_out = 18 → cars_in = 12 → 
  initial_cars - ((initial_cars - cars_out) + cars_in) = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_parking_lot_car_decrease_l2103_210360


namespace NUMINAMATH_CALUDE_soaps_in_package_l2103_210310

/-- Given a number of boxes, packages per box, and total soaps, calculates soaps per package -/
def soaps_per_package (num_boxes : ℕ) (packages_per_box : ℕ) (total_soaps : ℕ) : ℕ :=
  total_soaps / (num_boxes * packages_per_box)

/-- Theorem: There are 192 soaps in one package -/
theorem soaps_in_package :
  soaps_per_package 2 6 2304 = 192 := by sorry

end NUMINAMATH_CALUDE_soaps_in_package_l2103_210310


namespace NUMINAMATH_CALUDE_biased_coin_probability_l2103_210380

theorem biased_coin_probability (p : ℝ) : 
  p < (1 : ℝ) / 2 →
  6 * p^2 * (1 - p)^2 = (1 : ℝ) / 6 →
  p = (3 - Real.sqrt 3) / 6 := by
  sorry

end NUMINAMATH_CALUDE_biased_coin_probability_l2103_210380


namespace NUMINAMATH_CALUDE_squat_lift_loss_percentage_l2103_210381

/-- Calculates the percentage of squat lift lost given the original lifts and new total lift -/
theorem squat_lift_loss_percentage
  (orig_squat : ℝ)
  (orig_bench : ℝ)
  (orig_deadlift : ℝ)
  (deadlift_loss : ℝ)
  (new_total : ℝ)
  (h1 : orig_squat = 700)
  (h2 : orig_bench = 400)
  (h3 : orig_deadlift = 800)
  (h4 : deadlift_loss = 200)
  (h5 : new_total = 1490) :
  (orig_squat - (new_total - (orig_bench + (orig_deadlift - deadlift_loss)))) / orig_squat * 100 = 30 :=
by sorry

end NUMINAMATH_CALUDE_squat_lift_loss_percentage_l2103_210381


namespace NUMINAMATH_CALUDE_sequence_a_correct_l2103_210346

def sequence_a (n : ℕ) : ℚ :=
  if n = 1 then 1
  else 1 / (2 * n - 1 : ℚ) - 1 / (2 * n - 3 : ℚ)

def sum_S (n : ℕ) : ℚ :=
  if n = 1 then 1
  else 1 / (2 * n - 1 : ℚ)

theorem sequence_a_correct :
  ∀ n : ℕ, n ≥ 1 →
    (n = 1 ∧ sequence_a n = 1) ∨
    (n ≥ 2 ∧ (sum_S n)^2 = sequence_a n * (sum_S n - 1/2)) :=
by sorry

end NUMINAMATH_CALUDE_sequence_a_correct_l2103_210346


namespace NUMINAMATH_CALUDE_hall_width_proof_l2103_210393

/-- Given a rectangular hall with specified dimensions and cost constraints, 
    prove that the width of the hall is 15 meters. -/
theorem hall_width_proof (length height : ℝ) (cost_per_sqm total_cost : ℝ) :
  length = 20 →
  height = 5 →
  cost_per_sqm = 30 →
  total_cost = 28500 →
  ∃ w : ℝ, w > 0 ∧ 
    (2 * length * w + 2 * length * height + 2 * w * height) * cost_per_sqm = total_cost ∧
    w = 15 := by
  sorry

#check hall_width_proof

end NUMINAMATH_CALUDE_hall_width_proof_l2103_210393


namespace NUMINAMATH_CALUDE_function_value_determination_l2103_210364

theorem function_value_determination (A : ℝ) (α : ℝ) 
  (h1 : A ≠ 0)
  (h2 : α ∈ Set.Icc 0 π)
  (h3 : A * Real.sin (α + π/4) = Real.cos (2*α))
  (h4 : Real.sin (2*α) = -7/9) :
  A = -4*Real.sqrt 2/3 := by
sorry

end NUMINAMATH_CALUDE_function_value_determination_l2103_210364


namespace NUMINAMATH_CALUDE_book_arrangement_count_l2103_210395

/-- Represents the number of math books -/
def num_math_books : ℕ := 3

/-- Represents the number of physics books -/
def num_physics_books : ℕ := 2

/-- Represents the number of chemistry books -/
def num_chem_books : ℕ := 1

/-- Represents the total number of books -/
def total_books : ℕ := num_math_books + num_physics_books + num_chem_books

/-- Calculates the number of arrangements of books on a shelf -/
def num_arrangements : ℕ := 72

/-- Theorem stating that the number of arrangements of books on a shelf,
    where math books are adjacent and physics books are not adjacent,
    is equal to 72 -/
theorem book_arrangement_count :
  num_arrangements = 72 ∧
  num_math_books = 3 ∧
  num_physics_books = 2 ∧
  num_chem_books = 1 ∧
  total_books = num_math_books + num_physics_books + num_chem_books :=
by sorry

end NUMINAMATH_CALUDE_book_arrangement_count_l2103_210395


namespace NUMINAMATH_CALUDE_cans_in_cat_package_l2103_210387

/-- Represents the number of cans in each package of cat food -/
def cans_per_cat_package : ℕ := sorry

/-- The number of packages of cat food Adam bought -/
def cat_packages : ℕ := 9

/-- The number of packages of dog food Adam bought -/
def dog_packages : ℕ := 7

/-- The number of cans in each package of dog food -/
def cans_per_dog_package : ℕ := 5

/-- The difference between the total number of cat food cans and dog food cans -/
def can_difference : ℕ := 55

theorem cans_in_cat_package : 
  cans_per_cat_package * cat_packages = 
  cans_per_dog_package * dog_packages + can_difference ∧ 
  cans_per_cat_package = 10 := by sorry

end NUMINAMATH_CALUDE_cans_in_cat_package_l2103_210387


namespace NUMINAMATH_CALUDE_boat_distance_along_stream_l2103_210384

/-- The distance traveled by a boat along a stream in one hour -/
def distance_along_stream (boat_speed : ℝ) (against_stream_distance : ℝ) : ℝ :=
  boat_speed + (boat_speed - against_stream_distance)

/-- Theorem: The boat travels 11 km along the stream in one hour -/
theorem boat_distance_along_stream :
  distance_along_stream 9 7 = 11 := by
  sorry

end NUMINAMATH_CALUDE_boat_distance_along_stream_l2103_210384


namespace NUMINAMATH_CALUDE_tan_30_degrees_l2103_210303

theorem tan_30_degrees :
  let sin_30 := (1 : ℝ) / 2
  let cos_30 := Real.sqrt 3 / 2
  let tan_30 := sin_30 / cos_30
  tan_30 = Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_tan_30_degrees_l2103_210303


namespace NUMINAMATH_CALUDE_quadratic_function_property_l2103_210351

/-- Given a quadratic function y = ax^2 + bx + 2 passing through (-1, 0), 
    prove that 2a - 2b = -4 -/
theorem quadratic_function_property (a b : ℝ) : 
  (∀ x y : ℝ, y = a * x^2 + b * x + 2) → 
  (0 = a * (-1)^2 + b * (-1) + 2) → 
  2 * a - 2 * b = -4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l2103_210351


namespace NUMINAMATH_CALUDE_opposite_of_negative_three_l2103_210390

-- Define the concept of opposite for integers
def opposite (n : Int) : Int := -n

-- Theorem stating that the opposite of -3 is 3
theorem opposite_of_negative_three : opposite (-3) = 3 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_three_l2103_210390


namespace NUMINAMATH_CALUDE_fifteen_switch_network_connections_l2103_210312

/-- Represents a network of switches -/
structure SwitchNetwork where
  num_switches : ℕ
  connections_per_switch : ℕ

/-- Calculates the total number of connections in the network -/
def total_connections (network : SwitchNetwork) : ℕ :=
  (network.num_switches * network.connections_per_switch) / 2

/-- Theorem: In a network of 15 switches, where each switch is connected to 4 others,
    the total number of connections is 30 -/
theorem fifteen_switch_network_connections :
  let network : SwitchNetwork := ⟨15, 4⟩
  total_connections network = 30 := by
  sorry


end NUMINAMATH_CALUDE_fifteen_switch_network_connections_l2103_210312


namespace NUMINAMATH_CALUDE_total_carrots_l2103_210330

theorem total_carrots (sally fred mary : ℕ) 
  (h1 : sally = 6) 
  (h2 : fred = 4) 
  (h3 : mary = 10) : 
  sally + fred + mary = 20 := by
  sorry

end NUMINAMATH_CALUDE_total_carrots_l2103_210330


namespace NUMINAMATH_CALUDE_evaluate_expression_l2103_210322

theorem evaluate_expression (x y : ℚ) (hx : x = 4/8) (hy : y = 5/6) :
  (8*x + 6*y) / (72*x*y) = 3/10 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2103_210322


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2103_210331

theorem complex_fraction_simplification :
  let a : ℂ := 4 + 6*I
  let b : ℂ := 4 - 6*I
  (a/b) * (b/a) + (b/a) * (a/b) = 2 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2103_210331


namespace NUMINAMATH_CALUDE_cost_price_satisfies_profit_condition_l2103_210340

/-- The cost price of an article satisfies the given profit condition -/
theorem cost_price_satisfies_profit_condition (C : ℝ) : C > 0 → (0.27 * C) - (0.12 * C) = 108 ↔ C = 720 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_satisfies_profit_condition_l2103_210340


namespace NUMINAMATH_CALUDE_ellipse_theorem_l2103_210345

/-- Given an ellipse with semi-major axis a, semi-minor axis b, and eccentricity e -/
structure Ellipse where
  a : ℝ
  b : ℝ
  e : ℝ
  h_a_pos : 0 < a
  h_b_pos : 0 < b
  h_a_gt_b : b < a
  h_e_eq : e = 1/2
  h_e_def : e^2 = 1 - (b/a)^2

/-- The equation of the ellipse -/
def ellipse_equation (E : Ellipse) (x y : ℝ) : Prop :=
  x^2 / E.a^2 + y^2 / E.b^2 = 1

/-- The range of t for a line passing through (t,0) intersecting the ellipse -/
def t_range (t : ℝ) : Prop :=
  (t ≤ (4 - 6 * Real.sqrt 2) / 7 ∨ (4 + 6 * Real.sqrt 2) / 7 ≤ t) ∧ t ≠ 1

theorem ellipse_theorem (E : Ellipse) :
  (∀ x y, ellipse_equation E x y ↔ x^2 / 4 + y^2 / 3 = 1) ∧
  (∀ t, t_range t ↔
    ∃ A B : ℝ × ℝ,
      ellipse_equation E A.1 A.2 ∧
      ellipse_equation E B.1 B.2 ∧
      (A.1 - 1) * (B.1 - 1) + A.2 * B.2 = 0 ∧
      A.1 = t ∧ B.1 = t) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_theorem_l2103_210345


namespace NUMINAMATH_CALUDE_gcd_power_two_minus_one_l2103_210366

theorem gcd_power_two_minus_one (a b : ℕ+) :
  Nat.gcd ((2 : ℕ) ^ a.val - 1) ((2 : ℕ) ^ b.val - 1) = (2 : ℕ) ^ (Nat.gcd a.val b.val) - 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_power_two_minus_one_l2103_210366


namespace NUMINAMATH_CALUDE_no_sum_of_150_consecutive_integers_l2103_210343

theorem no_sum_of_150_consecutive_integers : ¬ ∃ (k : ℤ),
  (150 * k + 11325 = 678900) ∨
  (150 * k + 11325 = 1136850) ∨
  (150 * k + 11325 = 1000000) ∨
  (150 * k + 11325 = 2251200) ∨
  (150 * k + 11325 = 1876800) :=
by sorry

end NUMINAMATH_CALUDE_no_sum_of_150_consecutive_integers_l2103_210343


namespace NUMINAMATH_CALUDE_segments_complete_circle_num_segments_minimal_l2103_210352

/-- The number of equal segments that can be drawn around a circle,
    where each segment subtends an arc of 120°. -/
def num_segments : ℕ := 3

/-- The measure of the arc subtended by each segment in degrees. -/
def arc_measure : ℕ := 120

/-- Theorem stating that the number of segments multiplied by the arc measure
    equals a full circle (360°). -/
theorem segments_complete_circle :
  num_segments * arc_measure = 360 := by sorry

/-- Theorem stating that num_segments is the smallest positive integer
    that satisfies the segments_complete_circle property. -/
theorem num_segments_minimal :
  ∀ n : ℕ, 0 < n → n * arc_measure = 360 → num_segments ≤ n := by sorry

end NUMINAMATH_CALUDE_segments_complete_circle_num_segments_minimal_l2103_210352


namespace NUMINAMATH_CALUDE_inscribed_equilateral_triangle_in_five_moves_l2103_210385

/-- Represents a point in the plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a circle in the plane -/
structure Circle :=
  (center : Point)
  (radius : ℝ)

/-- Represents a line in the plane -/
structure Line :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

/-- Represents the game state -/
structure GameState :=
  (knownPoints : Set Point)
  (lines : Set Line)
  (circles : Set Circle)

/-- Represents a move in the game -/
inductive Move
  | DrawLine (p1 p2 : Point)
  | DrawCircle (center : Point) (throughPoint : Point)

/-- Checks if a point is on a circle -/
def isOnCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- Checks if three points form an equilateral triangle -/
def isEquilateralTriangle (p1 p2 p3 : Point) : Prop :=
  let d12 := ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)
  let d23 := ((p2.x - p3.x)^2 + (p2.y - p3.y)^2)
  let d31 := ((p3.x - p1.x)^2 + (p3.y - p1.y)^2)
  d12 = d23 ∧ d23 = d31

/-- The main theorem -/
theorem inscribed_equilateral_triangle_in_five_moves 
  (initialCircle : Circle) (initialPoint : Point) 
  (h : isOnCircle initialPoint initialCircle) :
  ∃ (moves : List Move) (p1 p2 p3 : Point),
    moves.length = 5 ∧
    isEquilateralTriangle p1 p2 p3 ∧
    isOnCircle p1 initialCircle ∧
    isOnCircle p2 initialCircle ∧
    isOnCircle p3 initialCircle :=
  sorry

end NUMINAMATH_CALUDE_inscribed_equilateral_triangle_in_five_moves_l2103_210385


namespace NUMINAMATH_CALUDE_problem_statement_l2103_210336

variables (a b c : ℝ)

def f (x : ℝ) := a * x^2 + b * x + c
def g (x : ℝ) := a * x + b

theorem problem_statement :
  (∀ x : ℝ, abs x ≤ 1 → abs (f a b c x) ≤ 1) →
  (abs c ≤ 1 ∧ ∀ x : ℝ, abs x ≤ 1 → abs (g a b x) ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l2103_210336


namespace NUMINAMATH_CALUDE_smallest_dual_base_palindrome_l2103_210302

/-- Checks if a number is a palindrome in a given base -/
def isPalindrome (n : ℕ) (base : ℕ) : Prop :=
  sorry

/-- Converts a number from base 10 to another base -/
def toBase (n : ℕ) (base : ℕ) : List ℕ :=
  sorry

theorem smallest_dual_base_palindrome :
  ∀ n : ℕ,
    n > 5 →
    isPalindrome n 2 →
    isPalindrome n 4 →
    (∀ m : ℕ, m > 5 ∧ m < n → ¬(isPalindrome m 2 ∧ isPalindrome m 4)) →
    n = 15 :=
  sorry

end NUMINAMATH_CALUDE_smallest_dual_base_palindrome_l2103_210302


namespace NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_6_with_digit_sum_12_l2103_210394

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem smallest_three_digit_multiple_of_6_with_digit_sum_12 :
  ∀ n : ℕ, is_three_digit n → n % 6 = 0 → digit_sum n = 12 → n ≥ 204 :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_6_with_digit_sum_12_l2103_210394


namespace NUMINAMATH_CALUDE_factorization_a_squared_minus_2a_l2103_210369

theorem factorization_a_squared_minus_2a (a : ℝ) : a^2 - 2*a = a*(a - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_a_squared_minus_2a_l2103_210369


namespace NUMINAMATH_CALUDE_f_increasing_on_positive_reals_l2103_210321

def f (x : ℝ) : ℝ := x^2 + x

theorem f_increasing_on_positive_reals :
  ∀ x y, 0 < x → 0 < y → x < y → f x < f y := by
  sorry

end NUMINAMATH_CALUDE_f_increasing_on_positive_reals_l2103_210321


namespace NUMINAMATH_CALUDE_fixed_point_power_function_l2103_210396

theorem fixed_point_power_function (a : ℝ) :
  let f : ℝ → ℝ := fun x ↦ (x - 2)^a + 1
  f 3 = 2 := by
sorry

end NUMINAMATH_CALUDE_fixed_point_power_function_l2103_210396


namespace NUMINAMATH_CALUDE_quadratic_sum_of_coefficients_l2103_210327

theorem quadratic_sum_of_coefficients (a b : ℝ) : 
  (∀ x, a * x^2 + b * x - 2 = 0 ↔ x = -2 ∨ x = 1/3) → 
  a + b = 8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_of_coefficients_l2103_210327


namespace NUMINAMATH_CALUDE_sum_divisible_by_ten_l2103_210389

theorem sum_divisible_by_ten : ∃ k : ℤ, 111^111 + 112^112 + 113^113 = 10 * k := by
  sorry

end NUMINAMATH_CALUDE_sum_divisible_by_ten_l2103_210389


namespace NUMINAMATH_CALUDE_multiplication_of_negative_half_and_two_l2103_210370

theorem multiplication_of_negative_half_and_two :
  (-1/2 : ℚ) * 2 = -1 := by sorry

end NUMINAMATH_CALUDE_multiplication_of_negative_half_and_two_l2103_210370


namespace NUMINAMATH_CALUDE_smallest_b_in_arithmetic_series_l2103_210398

theorem smallest_b_in_arithmetic_series (a b c : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- all terms are positive
  (∃ d : ℝ, a = b - d ∧ c = b + d) →  -- arithmetic series condition
  a * b * c = 125 →  -- product condition
  ∀ x : ℝ, (∃ y z : ℝ, 
    0 < y ∧ 0 < x ∧ 0 < z ∧  -- positivity for new terms
    (∃ e : ℝ, y = x - e ∧ z = x + e) ∧  -- arithmetic series for new terms
    y * x * z = 125) →  -- product condition for new terms
  x ≥ b →
  b ≥ 5 :=
sorry

end NUMINAMATH_CALUDE_smallest_b_in_arithmetic_series_l2103_210398


namespace NUMINAMATH_CALUDE_polynomial_sum_of_coefficients_l2103_210358

theorem polynomial_sum_of_coefficients :
  ∀ (a a₁ a₂ a₃ a₄ a₅ : ℝ),
  (∀ x : ℝ, x^5 + 1 = a + a₁*(x-1) + a₂*(x-1)^2 + a₃*(x-1)^3 + a₄*(x-1)^4 + a₅*(x-1)^5) →
  a + a₁ + a₂ + a₃ + a₄ + a₅ = 33 := by
sorry

end NUMINAMATH_CALUDE_polynomial_sum_of_coefficients_l2103_210358


namespace NUMINAMATH_CALUDE_A_minus_2B_value_of_2B_minus_A_l2103_210372

/-- Given two expressions A and B in terms of a and b -/
def A (a b : ℝ) : ℝ := 2*a^2 + a*b + 3*b

def B (a b : ℝ) : ℝ := a^2 - a*b + a

/-- Theorem stating the equality of A - 2B and its simplified form -/
theorem A_minus_2B (a b : ℝ) : A a b - 2 * B a b = 3*a*b + 3*b - 2*a := by sorry

/-- Theorem stating the value of 2B - A under the given condition -/
theorem value_of_2B_minus_A (a b : ℝ) (h : (a + 1)^2 + |b - 3| = 0) : 
  2 * B a b - A a b = -2 := by sorry

end NUMINAMATH_CALUDE_A_minus_2B_value_of_2B_minus_A_l2103_210372


namespace NUMINAMATH_CALUDE_sqrt_difference_equals_two_sqrt_three_l2103_210357

theorem sqrt_difference_equals_two_sqrt_three :
  Real.sqrt (7 + 4 * Real.sqrt 3) - Real.sqrt (7 - 4 * Real.sqrt 3) = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equals_two_sqrt_three_l2103_210357


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2103_210324

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- State the problem
theorem complex_fraction_simplification :
  (2 + 4 * i) / (1 - 5 * i) = -9/13 + (7/13) * i :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2103_210324


namespace NUMINAMATH_CALUDE_rachel_plant_arrangement_count_l2103_210323

/-- Represents the types of plants Rachel has -/
inductive Plant
| Basil
| Aloe
| Cactus

/-- Represents the colors of lamps Rachel has -/
inductive LampColor
| White
| Red
| Blue

/-- A configuration of plants under lamps -/
structure Configuration where
  plantUnderLamp : Plant → LampColor

/-- Checks if a configuration is valid according to the given conditions -/
def isValidConfiguration (config : Configuration) : Prop :=
  -- Each plant is under exactly one lamp
  (∀ p : Plant, ∃! l : LampColor, config.plantUnderLamp p = l) ∧
  -- No lamp is used for just one plant unless it's the red one
  (∀ l : LampColor, l ≠ LampColor.Red → (∃ p₁ p₂ : Plant, p₁ ≠ p₂ ∧ config.plantUnderLamp p₁ = l ∧ config.plantUnderLamp p₂ = l))

/-- The number of valid configurations -/
def validConfigurationsCount : ℕ := sorry

theorem rachel_plant_arrangement_count :
  validConfigurationsCount = 4 := by sorry

end NUMINAMATH_CALUDE_rachel_plant_arrangement_count_l2103_210323


namespace NUMINAMATH_CALUDE_square_area_on_parabola_l2103_210341

/-- The area of a square with one side on y = 7 and endpoints on y = x^2 + 4x + 3 is 32 -/
theorem square_area_on_parabola : ∃ (x₁ x₂ : ℝ),
  (x₁^2 + 4*x₁ + 3 = 7) ∧
  (x₂^2 + 4*x₂ + 3 = 7) ∧
  (x₁ ≠ x₂) ∧
  ((x₂ - x₁)^2 = 32) := by
  sorry

end NUMINAMATH_CALUDE_square_area_on_parabola_l2103_210341
