import Mathlib

namespace NUMINAMATH_CALUDE_isabel_initial_candy_l974_97465

/- Given conditions -/
def initial_candy : ℕ → Prop := λ x => True  -- Initial amount of candy (unknown)
def friend_gave : ℕ := 25                    -- Amount of candy given by friend
def total_candy : ℕ := 93                    -- Total amount of candy after receiving from friend

/- Theorem to prove -/
theorem isabel_initial_candy :
  ∃ x : ℕ, initial_candy x ∧ x + friend_gave = total_candy ∧ x = 68 :=
by sorry

end NUMINAMATH_CALUDE_isabel_initial_candy_l974_97465


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l974_97454

theorem algebraic_expression_value (x : ℝ) (h : x * (x + 2) = 2023) :
  2 * (x + 3) * (x - 1) - 2018 = 2022 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l974_97454


namespace NUMINAMATH_CALUDE_worksheets_graded_l974_97450

theorem worksheets_graded (total_worksheets : ℕ) (problems_per_worksheet : ℕ) (problems_left : ℕ) : 
  total_worksheets = 9 →
  problems_per_worksheet = 4 →
  problems_left = 16 →
  total_worksheets - (problems_left / problems_per_worksheet) = 5 :=
by sorry

end NUMINAMATH_CALUDE_worksheets_graded_l974_97450


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l974_97446

theorem fraction_to_decimal : (7 : ℚ) / 16 = (4375 : ℚ) / 10000 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l974_97446


namespace NUMINAMATH_CALUDE_f_has_unique_zero_in_interval_l974_97461

/-- The function f(x) = -x³ + x² + x - 2 -/
def f (x : ℝ) := -x^3 + x^2 + x - 2

/-- The theorem stating that f has exactly one zero in (-∞, -1/3) -/
theorem f_has_unique_zero_in_interval :
  ∃! x, x < -1/3 ∧ f x = 0 :=
sorry

end NUMINAMATH_CALUDE_f_has_unique_zero_in_interval_l974_97461


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_q_l974_97466

theorem p_sufficient_not_necessary_q :
  (∀ x : ℝ, 0 < x ∧ x < 2 → -1 < x ∧ x < 3) ∧
  (∃ x : ℝ, -1 < x ∧ x < 3 ∧ ¬(0 < x ∧ x < 2)) := by
  sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_q_l974_97466


namespace NUMINAMATH_CALUDE_only_D_correct_l974_97464

/-- Represents the contestants in the singing competition -/
inductive Contestant : Type
  | one | two | three | four | five | six

/-- Represents the students who made guesses -/
inductive Student : Type
  | A | B | C | D

/-- The guess made by each student -/
def studentGuess (s : Student) : Contestant → Prop :=
  match s with
  | Student.A => λ c => c = Contestant.four ∨ c = Contestant.five
  | Student.B => λ c => c ≠ Contestant.three
  | Student.C => λ c => c = Contestant.one ∨ c = Contestant.two ∨ c = Contestant.six
  | Student.D => λ c => c ≠ Contestant.four ∧ c ≠ Contestant.five ∧ c ≠ Contestant.six

/-- The theorem to be proved -/
theorem only_D_correct :
  ∃ (winner : Contestant),
    (∀ (s : Student), s ≠ Student.D → ¬(studentGuess s winner)) ∧
    (studentGuess Student.D winner) :=
  sorry

end NUMINAMATH_CALUDE_only_D_correct_l974_97464


namespace NUMINAMATH_CALUDE_dunk_a_clown_tickets_l974_97488

def total_tickets : ℕ := 40
def num_rides : ℕ := 3
def tickets_per_ride : ℕ := 4

theorem dunk_a_clown_tickets : 
  total_tickets - (num_rides * tickets_per_ride) = 28 := by
  sorry

end NUMINAMATH_CALUDE_dunk_a_clown_tickets_l974_97488


namespace NUMINAMATH_CALUDE_max_value_of_sqrt_sum_l974_97417

theorem max_value_of_sqrt_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 5) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 5 → Real.sqrt (x + 1) + Real.sqrt (y + 3) ≤ 3 * Real.sqrt 2) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 5 ∧ Real.sqrt (x + 1) + Real.sqrt (y + 3) = 3 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_sqrt_sum_l974_97417


namespace NUMINAMATH_CALUDE_smallest_value_w4_plus_z4_l974_97442

theorem smallest_value_w4_plus_z4 (w z : ℂ) 
  (h1 : Complex.abs (w + z) = 2)
  (h2 : Complex.abs (w^2 + z^2) = 10) :
  Complex.abs (w^4 + z^4) = 82 := by
  sorry

end NUMINAMATH_CALUDE_smallest_value_w4_plus_z4_l974_97442


namespace NUMINAMATH_CALUDE_y_axis_intersection_x_axis_intersections_l974_97474

-- Define the parabola function
def f (x : ℝ) : ℝ := x^2 - 3*x + 2

-- Theorem for y-axis intersection
theorem y_axis_intersection : f 0 = 2 := by sorry

-- Theorem for x-axis intersections
theorem x_axis_intersections :
  (f 2 = 0 ∧ f 1 = 0) ∧ ∀ x : ℝ, f x = 0 → (x = 2 ∨ x = 1) := by sorry

end NUMINAMATH_CALUDE_y_axis_intersection_x_axis_intersections_l974_97474


namespace NUMINAMATH_CALUDE_election_winner_votes_l974_97443

theorem election_winner_votes 
  (total_votes : ℕ) 
  (winner_percentage : ℚ) 
  (vote_difference : ℕ) 
  (h1 : winner_percentage = 62 / 100) 
  (h2 : winner_percentage * total_votes - (1 - winner_percentage) * total_votes = vote_difference) 
  (h3 : vote_difference = 348) : 
  ⌊winner_percentage * total_votes⌋ = 899 :=
sorry

end NUMINAMATH_CALUDE_election_winner_votes_l974_97443


namespace NUMINAMATH_CALUDE_binary_rep_of_23_l974_97483

/-- Converts a natural number to its binary representation as a list of bits -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
    let rec go (m : ℕ) : List Bool :=
      if m = 0 then [] else (m % 2 = 1) :: go (m / 2)
    go n

/-- Theorem: The binary representation of 23 is [true, true, true, false, true] -/
theorem binary_rep_of_23 : toBinary 23 = [true, true, true, false, true] := by
  sorry

end NUMINAMATH_CALUDE_binary_rep_of_23_l974_97483


namespace NUMINAMATH_CALUDE_base8_cube_c_is_zero_l974_97447

/-- Represents a number in base 8 of the form 4c3 --/
def base8Number (c : ℕ) : ℕ := 4 * 8^2 + c * 8 + 3

/-- Checks if a number is a perfect cube --/
def isPerfectCube (n : ℕ) : Prop := ∃ m : ℕ, m^3 = n

theorem base8_cube_c_is_zero :
  ∃ c : ℕ, isPerfectCube (base8Number c) → c = 0 :=
sorry

end NUMINAMATH_CALUDE_base8_cube_c_is_zero_l974_97447


namespace NUMINAMATH_CALUDE_min_value_expression_l974_97423

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2*b = 1) :
  a^2 + 4*b^2 + 1/(a*b) ≥ 17/2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l974_97423


namespace NUMINAMATH_CALUDE_flour_per_pizza_l974_97440

def carnival_time : ℕ := 7 * 60 -- 7 hours in minutes
def flour_amount : ℚ := 22 -- 22 kg of flour
def pizza_time : ℕ := 10 -- 10 minutes per pizza
def extra_pizzas : ℕ := 2 -- 2 additional pizzas from leftover flour

theorem flour_per_pizza :
  let total_pizzas := carnival_time / pizza_time + extra_pizzas
  flour_amount / total_pizzas = 1/2 := by sorry

end NUMINAMATH_CALUDE_flour_per_pizza_l974_97440


namespace NUMINAMATH_CALUDE_arcsin_equation_solution_l974_97486

theorem arcsin_equation_solution :
  ∃ x : ℝ, x = 1 ∧ Real.arcsin x + Real.arcsin (x - 1) = Real.arccos (1 - x) :=
by sorry

end NUMINAMATH_CALUDE_arcsin_equation_solution_l974_97486


namespace NUMINAMATH_CALUDE_shoe_price_calculation_l974_97459

theorem shoe_price_calculation (initial_price : ℝ) (increase_rate : ℝ) (discount_rate : ℝ) : 
  initial_price = 50 →
  increase_rate = 0.2 →
  discount_rate = 0.15 →
  initial_price * (1 + increase_rate) * (1 - discount_rate) = 51 :=
by sorry

end NUMINAMATH_CALUDE_shoe_price_calculation_l974_97459


namespace NUMINAMATH_CALUDE_merchant_pricing_strategy_l974_97452

theorem merchant_pricing_strategy (list_price : ℝ) (list_price_pos : list_price > 0) :
  let purchase_price := list_price * 0.7
  let marked_price := list_price * 1.25
  let selling_price := marked_price * 0.8
  let profit := selling_price - purchase_price
  profit = selling_price * 0.3 := by sorry

end NUMINAMATH_CALUDE_merchant_pricing_strategy_l974_97452


namespace NUMINAMATH_CALUDE_log_equation_solution_l974_97441

theorem log_equation_solution (a b c x : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  Real.log x = Real.log a + 3 * Real.log b - 5 * Real.log c →
  x = a * b^3 / c^5 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l974_97441


namespace NUMINAMATH_CALUDE_melissa_points_per_game_l974_97437

-- Define the total points scored
def total_points : ℕ := 1200

-- Define the number of games played
def num_games : ℕ := 10

-- Define the points per game
def points_per_game : ℕ := total_points / num_games

-- Theorem statement
theorem melissa_points_per_game : points_per_game = 120 := by
  sorry

end NUMINAMATH_CALUDE_melissa_points_per_game_l974_97437


namespace NUMINAMATH_CALUDE_square_difference_plus_double_l974_97495

theorem square_difference_plus_double (x y : ℝ) (h : x + y = 1) : x^2 - y^2 + 2*y = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_plus_double_l974_97495


namespace NUMINAMATH_CALUDE_circle_line_intersection_l974_97430

/-- Given a circle (x-a)^2 + y^2 = 4 and a line x - y = 2, 
    if the chord length intercepted by the circle on the line is 2√2, 
    then a = 0 or a = 4 -/
theorem circle_line_intersection (a : ℝ) : 
  (∃ x y : ℝ, (x - a)^2 + y^2 = 4 ∧ x - y = 2) →  -- circle and line intersect
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    (x₁ - a)^2 + y₁^2 = 4 ∧ x₁ - y₁ = 2 ∧  -- first intersection point
    (x₂ - a)^2 + y₂^2 = 4 ∧ x₂ - y₂ = 2 ∧  -- second intersection point
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 8) →      -- chord length is 2√2
  a = 0 ∨ a = 4 := by
sorry

end NUMINAMATH_CALUDE_circle_line_intersection_l974_97430


namespace NUMINAMATH_CALUDE_janet_pill_count_l974_97492

def pills_per_day_first_two_weeks : ℕ := 2 + 3
def pills_per_day_last_two_weeks : ℕ := 2 + 1
def days_per_week : ℕ := 7
def weeks_in_month : ℕ := 4

theorem janet_pill_count :
  (pills_per_day_first_two_weeks * days_per_week * (weeks_in_month / 2)) +
  (pills_per_day_last_two_weeks * days_per_week * (weeks_in_month / 2)) = 112 :=
by sorry

end NUMINAMATH_CALUDE_janet_pill_count_l974_97492


namespace NUMINAMATH_CALUDE_proposition_and_variants_true_l974_97436

theorem proposition_and_variants_true :
  (∀ x y : ℝ, x^2 + y^2 = 0 → x = 0 ∧ y = 0) ∧
  (∀ x y : ℝ, x = 0 ∧ y = 0 → x^2 + y^2 = 0) ∧
  (∀ x y : ℝ, ¬(x = 0 ∧ y = 0) → x^2 + y^2 ≠ 0) ∧
  (∀ x y : ℝ, x^2 + y^2 ≠ 0 → ¬(x = 0 ∧ y = 0)) :=
by sorry

end NUMINAMATH_CALUDE_proposition_and_variants_true_l974_97436


namespace NUMINAMATH_CALUDE_f_of_3_eq_neg_1_l974_97419

-- Define the function f
def f (x : ℝ) : ℝ := 
  let t := 2 * x + 1
  x^2 - 2*x

-- Theorem statement
theorem f_of_3_eq_neg_1 : f 1 = -1 := by
  sorry

end NUMINAMATH_CALUDE_f_of_3_eq_neg_1_l974_97419


namespace NUMINAMATH_CALUDE_train_crossing_time_l974_97462

/-- Time taken for two trains to cross each other -/
theorem train_crossing_time (train_length : ℝ) (fast_speed : ℝ) : 
  train_length = 100 →
  fast_speed = 24 →
  (50 : ℝ) / 9 = (2 * train_length) / (fast_speed + fast_speed / 2) := by
  sorry

#eval (50 : ℚ) / 9

end NUMINAMATH_CALUDE_train_crossing_time_l974_97462


namespace NUMINAMATH_CALUDE_spinner_final_direction_l974_97476

-- Define the direction type
inductive Direction
  | North
  | East
  | South
  | West

-- Define the rotation type
inductive Rotation
  | Clockwise
  | Counterclockwise

-- Define a function to calculate the final direction after a rotation
def rotateSpinner (initialDir : Direction) (rotation : Rotation) (revolutions : ℚ) : Direction :=
  sorry

-- Theorem statement
theorem spinner_final_direction :
  let initialDir := Direction.North
  let clockwiseRot := 7/2
  let counterclockwiseRot := 21/4
  let finalDir := rotateSpinner (rotateSpinner initialDir Rotation.Clockwise clockwiseRot) Rotation.Counterclockwise counterclockwiseRot
  finalDir = Direction.East := by sorry

end NUMINAMATH_CALUDE_spinner_final_direction_l974_97476


namespace NUMINAMATH_CALUDE_exists_distinct_power_sum_l974_97434

/-- Represents a sum of distinct powers of 3, 4, and 7 -/
structure DistinctPowerSum where
  powers_of_3 : List Nat
  powers_of_4 : List Nat
  powers_of_7 : List Nat
  distinct : powers_of_3.Nodup ∧ powers_of_4.Nodup ∧ powers_of_7.Nodup

/-- Calculates the sum of the powers in a DistinctPowerSum -/
def sumPowers (dps : DistinctPowerSum) : Nat :=
  (dps.powers_of_3.map (fun x => 3^x)).sum +
  (dps.powers_of_4.map (fun x => 4^x)).sum +
  (dps.powers_of_7.map (fun x => 7^x)).sum

/-- Theorem: Every positive integer can be represented as a sum of distinct powers of 3, 4, and 7 -/
theorem exists_distinct_power_sum (n : Nat) (h : n > 0) :
  ∃ (dps : DistinctPowerSum), sumPowers dps = n := by
  sorry

end NUMINAMATH_CALUDE_exists_distinct_power_sum_l974_97434


namespace NUMINAMATH_CALUDE_ali_fish_weight_l974_97451

/-- Proves that Ali caught 12 kg of fish given the conditions of the fishing problem -/
theorem ali_fish_weight (peter_weight : ℝ) 
  (h1 : peter_weight + 2 * peter_weight + (peter_weight + 1) = 25) : 
  2 * peter_weight = 12 := by
  sorry

end NUMINAMATH_CALUDE_ali_fish_weight_l974_97451


namespace NUMINAMATH_CALUDE_range_of_a_l974_97477

/-- The set A defined by the equation x^2 + 4x = 0 -/
def A : Set ℝ := {x | x^2 + 4*x = 0}

/-- The set B defined by the equation x^2 + ax + a = 0, where a is a parameter -/
def B (a : ℝ) : Set ℝ := {x | x^2 + a*x + a = 0}

/-- The theorem stating the range of a given the conditions -/
theorem range_of_a (a : ℝ) : A ∪ B a = A ↔ 0 ≤ a ∧ a < 4 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l974_97477


namespace NUMINAMATH_CALUDE_inequality_theorem_l974_97491

theorem inequality_theorem (a b c d : ℝ) (h1 : a > b) (h2 : c = d) : a + c > b + d := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_l974_97491


namespace NUMINAMATH_CALUDE_initial_amount_proof_l974_97484

theorem initial_amount_proof (P : ℝ) : 
  (P * (1 + 1/8) * (1 + 1/8) = 2025) → P = 1600 := by
  sorry

end NUMINAMATH_CALUDE_initial_amount_proof_l974_97484


namespace NUMINAMATH_CALUDE_line_through_p_equally_divided_l974_97497

/-- The line passing through P(3,0) and equally divided by P in the segment AB
    between lines 2x - y - 2 = 0 and x + y + 3 = 0 has the equation 4x - 5y = 12 -/
theorem line_through_p_equally_divided : 
  let P : ℝ × ℝ := (3, 0)
  let line1 : ℝ → ℝ → Prop := λ x y => 2 * x - y - 2 = 0
  let line2 : ℝ → ℝ → Prop := λ x y => x + y + 3 = 0
  let sought_line : ℝ → ℝ → Prop := λ x y => 4 * x - 5 * y = 12
  ∃ A B : ℝ × ℝ,
    (line1 A.1 A.2 ∧ sought_line A.1 A.2) ∧ 
    (line2 B.1 B.2 ∧ sought_line B.1 B.2) ∧
    (P.1 = (A.1 + B.1) / 2 ∧ P.2 = (A.2 + B.2) / 2) ∧
    sought_line P.1 P.2 :=
by sorry

end NUMINAMATH_CALUDE_line_through_p_equally_divided_l974_97497


namespace NUMINAMATH_CALUDE_pigs_in_barn_l974_97409

/-- The total number of pigs after more pigs join the barn -/
def total_pigs (initial : Float) (joined : Float) : Float :=
  initial + joined

/-- Theorem stating that given 64.0 initial pigs and 86.0 pigs joining, the total is 150.0 -/
theorem pigs_in_barn : total_pigs 64.0 86.0 = 150.0 := by
  sorry

end NUMINAMATH_CALUDE_pigs_in_barn_l974_97409


namespace NUMINAMATH_CALUDE_gecko_infertile_eggs_percentage_l974_97418

theorem gecko_infertile_eggs_percentage 
  (total_eggs : ℕ) 
  (hatched_eggs : ℕ) 
  (calcification_rate : ℚ) :
  total_eggs = 30 →
  hatched_eggs = 16 →
  calcification_rate = 1/3 →
  ∃ (infertile_percentage : ℚ),
    infertile_percentage = 20/100 ∧
    hatched_eggs = (total_eggs : ℚ) * (1 - infertile_percentage) * (1 - calcification_rate) :=
by sorry

end NUMINAMATH_CALUDE_gecko_infertile_eggs_percentage_l974_97418


namespace NUMINAMATH_CALUDE_marbles_distribution_l974_97414

theorem marbles_distribution (total_marbles : ℕ) (num_boys : ℕ) (marbles_per_boy : ℕ) :
  total_marbles = 99 →
  num_boys = 11 →
  total_marbles = num_boys * marbles_per_boy →
  marbles_per_boy = 9 := by
  sorry

end NUMINAMATH_CALUDE_marbles_distribution_l974_97414


namespace NUMINAMATH_CALUDE_walter_chores_l974_97401

theorem walter_chores (total_days : ℕ) (total_earnings : ℕ) 
  (regular_pay : ℕ) (exceptional_pay : ℕ) :
  total_days = 15 →
  total_earnings = 47 →
  regular_pay = 3 →
  exceptional_pay = 4 →
  ∃ (regular_days exceptional_days : ℕ),
    regular_days + exceptional_days = total_days ∧
    regular_days * regular_pay + exceptional_days * exceptional_pay = total_earnings ∧
    exceptional_days = 2 :=
by sorry

end NUMINAMATH_CALUDE_walter_chores_l974_97401


namespace NUMINAMATH_CALUDE_equation_undefined_at_five_l974_97407

theorem equation_undefined_at_five :
  ¬∃ (y : ℝ), (1 / (5 + 5) + 1 / (5 - 5) : ℝ) = y :=
sorry

end NUMINAMATH_CALUDE_equation_undefined_at_five_l974_97407


namespace NUMINAMATH_CALUDE_composite_function_inverse_l974_97410

theorem composite_function_inverse (a b : ℝ) : 
  let f (x : ℝ) := a * x + b
  let g (x : ℝ) := -2 * x^2 + 4 * x - 1
  let h := f ∘ g
  (∀ x, h.invFun x = 2 * x - 3) →
  2 * a - 3 * b = -91 / 32 := by
sorry

end NUMINAMATH_CALUDE_composite_function_inverse_l974_97410


namespace NUMINAMATH_CALUDE_no_solution_of_double_composition_l974_97453

theorem no_solution_of_double_composition
  (P : ℝ → ℝ)
  (h_continuous : Continuous P)
  (h_no_solution : ∀ x : ℝ, P x ≠ x) :
  ∀ x : ℝ, P (P x) ≠ x :=
by sorry

end NUMINAMATH_CALUDE_no_solution_of_double_composition_l974_97453


namespace NUMINAMATH_CALUDE_solve_eggs_problem_l974_97467

def eggs_problem (total_cost : ℝ) (price_per_egg : ℝ) (remaining_eggs : ℕ) : Prop :=
  let eggs_sold := total_cost / price_per_egg
  let initial_eggs := eggs_sold + remaining_eggs
  initial_eggs = 30

theorem solve_eggs_problem :
  eggs_problem 5 0.20 5 :=
sorry

end NUMINAMATH_CALUDE_solve_eggs_problem_l974_97467


namespace NUMINAMATH_CALUDE_divisor_coloring_game_strategy_l974_97469

/-- A player in the divisor coloring game -/
inductive Player
| A
| B

/-- The result of the divisor coloring game -/
inductive GameResult
| AWins
| BWins

/-- The divisor coloring game for a positive integer n -/
def divisorColoringGame (n : ℕ+) : GameResult := sorry

/-- Check if a number is a perfect square -/
def isPerfectSquare (n : ℕ+) : Prop := ∃ m : ℕ+, n = m * m

/-- Theorem: Player A wins if and only if n is a perfect square or prime -/
theorem divisor_coloring_game_strategy (n : ℕ+) :
  divisorColoringGame n = GameResult.AWins ↔ isPerfectSquare n ∨ Nat.Prime n.val := by sorry

end NUMINAMATH_CALUDE_divisor_coloring_game_strategy_l974_97469


namespace NUMINAMATH_CALUDE_pizza_combinations_l974_97472

theorem pizza_combinations (n_toppings : ℕ) (n_crusts : ℕ) (k_toppings : ℕ) : 
  n_toppings = 8 → n_crusts = 2 → k_toppings = 5 → 
  (Nat.choose n_toppings k_toppings) * n_crusts = 112 := by
  sorry

end NUMINAMATH_CALUDE_pizza_combinations_l974_97472


namespace NUMINAMATH_CALUDE_success_arrangements_l974_97480

-- Define the total number of letters
def total_letters : ℕ := 7

-- Define the repetitions of each letter
def s_count : ℕ := 3
def c_count : ℕ := 2
def u_count : ℕ := 1
def e_count : ℕ := 1

-- Define the function to calculate the number of arrangements
def arrangements : ℕ := total_letters.factorial / (s_count.factorial * c_count.factorial * u_count.factorial * e_count.factorial)

-- State the theorem
theorem success_arrangements : arrangements = 420 := by
  sorry

end NUMINAMATH_CALUDE_success_arrangements_l974_97480


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l974_97475

theorem hyperbola_eccentricity_range (a : ℝ) (h : a > 1) :
  let e := Real.sqrt (1 + 1 / a^2)
  1 < e ∧ e < Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l974_97475


namespace NUMINAMATH_CALUDE_board_covering_l974_97402

def can_cover (m n : ℕ) : Prop :=
  ∃ (a b : ℕ), m * n = 3 * a + 10 * b

def excluded_pairs : Set (ℕ × ℕ) :=
  {(4,4), (2,2), (2,4), (2,7)}

def excluded_1xn (n : ℕ) : Prop :=
  ∃ (k : ℕ), n = 3*k + 1 ∨ n = 3*k + 2

theorem board_covering (m n : ℕ) :
  can_cover m n ↔ (m, n) ∉ excluded_pairs ∧ (m ≠ 1 ∨ ¬excluded_1xn n) :=
sorry

end NUMINAMATH_CALUDE_board_covering_l974_97402


namespace NUMINAMATH_CALUDE_greatest_n_squared_l974_97438

theorem greatest_n_squared (n : ℤ) (V : ℝ) : 
  (∀ m : ℤ, 102 * m^2 ≤ V → m ≤ 8) →
  (102 * 8^2 ≤ V) →
  V = 6528 := by
  sorry

end NUMINAMATH_CALUDE_greatest_n_squared_l974_97438


namespace NUMINAMATH_CALUDE_union_of_A_and_B_range_of_a_l974_97435

-- Define sets A and B
def A (a : ℝ) := {x : ℝ | 0 < a * x - 1 ∧ a * x - 1 ≤ 5}
def B := {x : ℝ | -1/2 < x ∧ x ≤ 2}

-- Part I
theorem union_of_A_and_B (a : ℝ) (h : a = 1) :
  A a ∪ B = {x : ℝ | -1/2 < x ∧ x ≤ 6} := by sorry

-- Part II
theorem range_of_a (a : ℝ) (h1 : A a ∩ B = ∅) (h2 : a > 0) :
  0 < a ∧ a ≤ 1/2 := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_range_of_a_l974_97435


namespace NUMINAMATH_CALUDE_danielas_age_l974_97412

/-- Given the ages and relationships of several people, prove Daniela's age --/
theorem danielas_age (clara_age : ℕ) (daniela_age evelina_age fidel_age caitlin_age : ℕ) :
  clara_age = 60 →
  daniela_age = evelina_age - 8 →
  evelina_age = clara_age / 3 →
  fidel_age = 2 * caitlin_age →
  fidel_age = evelina_age - 6 →
  daniela_age = 12 := by
sorry


end NUMINAMATH_CALUDE_danielas_age_l974_97412


namespace NUMINAMATH_CALUDE_star_two_one_l974_97490

-- Define the ∗ operation for real numbers
def star (x y : ℝ) : ℝ := x - y + x * y

-- State the theorem
theorem star_two_one : star 2 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_star_two_one_l974_97490


namespace NUMINAMATH_CALUDE_min_perimeter_sum_l974_97432

/-- Represents a chessboard configuration -/
structure ChessboardConfig (m : ℕ) where
  size : Fin (2^m) → Fin (2^m) → Bool
  diagonal_unit : ∀ i : Fin (2^m), size i i = true

/-- Calculates the sum of perimeters for a given chessboard configuration -/
def sumPerimeters (m : ℕ) (config : ChessboardConfig m) : ℕ :=
  sorry

/-- Theorem: The minimum sum of perimeters for a 2^m × 2^m chessboard configuration -/
theorem min_perimeter_sum (m : ℕ) : 
  (∃ (config : ChessboardConfig m), 
    ∀ (other_config : ChessboardConfig m), 
      sumPerimeters m config ≤ sumPerimeters m other_config) ∧
  (∃ (config : ChessboardConfig m), sumPerimeters m config = 2^(m+2) * (m+1)) := by
  sorry

end NUMINAMATH_CALUDE_min_perimeter_sum_l974_97432


namespace NUMINAMATH_CALUDE_manuscript_cost_theorem_l974_97482

/-- Represents the cost of typing and revising a manuscript. -/
def manuscript_cost (
  total_pages : ℕ
  ) (
  first_type_cost : ℕ
  ) (
  first_revision_cost : ℕ
  ) (
  second_revision_cost : ℕ
  ) (
  third_plus_revision_cost : ℕ
  ) (
  pages_revised_once : ℕ
  ) (
  pages_revised_twice : ℕ
  ) (
  pages_revised_thrice : ℕ
  ) (
  pages_revised_four_times : ℕ
  ) : ℕ :=
  total_pages * first_type_cost +
  pages_revised_once * first_revision_cost +
  pages_revised_twice * (first_revision_cost + second_revision_cost) +
  pages_revised_thrice * (first_revision_cost + second_revision_cost + third_plus_revision_cost) +
  pages_revised_four_times * (first_revision_cost + second_revision_cost + 2 * third_plus_revision_cost)

/-- Theorem: The total cost of typing and revising the manuscript is $2240. -/
theorem manuscript_cost_theorem :
  manuscript_cost 270 5 3 2 1 90 60 30 20 = 2240 := by
  sorry


end NUMINAMATH_CALUDE_manuscript_cost_theorem_l974_97482


namespace NUMINAMATH_CALUDE_prob_exactly_two_correct_prob_at_least_two_correct_prob_all_incorrect_l974_97425

/-- The number of students and backpacks -/
def n : ℕ := 4

/-- The total number of ways to pick up backpacks -/
def total_outcomes : ℕ := 24

/-- The number of outcomes where exactly two students pick up their correct backpacks -/
def exactly_two_correct : ℕ := 6

/-- The number of outcomes where at least two students pick up their correct backpacks -/
def at_least_two_correct : ℕ := 7

/-- The number of outcomes where all backpacks are picked up incorrectly -/
def all_incorrect : ℕ := 9

/-- The probability of exactly two students picking up the correct backpacks -/
theorem prob_exactly_two_correct : 
  exactly_two_correct / total_outcomes = 1 / 4 := by sorry

/-- The probability of at least two students picking up the correct backpacks -/
theorem prob_at_least_two_correct : 
  at_least_two_correct / total_outcomes = 7 / 24 := by sorry

/-- The probability of all backpacks being picked up incorrectly -/
theorem prob_all_incorrect : 
  all_incorrect / total_outcomes = 3 / 8 := by sorry

end NUMINAMATH_CALUDE_prob_exactly_two_correct_prob_at_least_two_correct_prob_all_incorrect_l974_97425


namespace NUMINAMATH_CALUDE_a_squared_coefficient_zero_l974_97470

theorem a_squared_coefficient_zero (p : ℚ) : 
  (∀ a : ℚ, (a^2 - p*a + 6) * (2*a - 1) = (-p*a + 6) * (2*a - 1)) → p = -1/2 :=
by sorry

end NUMINAMATH_CALUDE_a_squared_coefficient_zero_l974_97470


namespace NUMINAMATH_CALUDE_progression_to_floor_pushups_l974_97493

/-- The number of weeks it takes to progress to floor push-ups -/
def weeks_to_floor_pushups (days_per_week : ℕ) (levels_before_floor : ℕ) (days_per_level : ℕ) : ℕ :=
  (levels_before_floor * days_per_level) / days_per_week

/-- Theorem stating that it takes 9 weeks to progress to floor push-ups under given conditions -/
theorem progression_to_floor_pushups :
  weeks_to_floor_pushups 5 3 15 = 9 := by
  sorry

end NUMINAMATH_CALUDE_progression_to_floor_pushups_l974_97493


namespace NUMINAMATH_CALUDE_crayons_in_box_l974_97499

def blue_crayons : ℕ := 3

def red_crayons : ℕ := 4 * blue_crayons

def total_crayons : ℕ := red_crayons + blue_crayons

theorem crayons_in_box : total_crayons = 15 := by
  sorry

end NUMINAMATH_CALUDE_crayons_in_box_l974_97499


namespace NUMINAMATH_CALUDE_max_divisible_arrangement_l974_97415

def is_divisible (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k

def valid_arrangement (arr : List ℕ) : Prop :=
  ∀ i : ℕ, i < arr.length - 1 → 
    is_divisible (arr.get ⟨i, by sorry⟩) (arr.get ⟨i+1, by sorry⟩) ∨ 
    is_divisible (arr.get ⟨i+1, by sorry⟩) (arr.get ⟨i, by sorry⟩)

def cards : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9]

theorem max_divisible_arrangement :
  (∃ (arr : List ℕ), arr.length = 8 ∧ 
    (∀ x ∈ arr, x ∈ cards) ∧ 
    valid_arrangement arr) ∧
  (∀ (arr : List ℕ), arr.length > 8 → 
    (∀ x ∈ arr, x ∈ cards) → 
    ¬valid_arrangement arr) := by sorry

end NUMINAMATH_CALUDE_max_divisible_arrangement_l974_97415


namespace NUMINAMATH_CALUDE_cycling_time_difference_l974_97408

-- Define the distances and speeds for each day
def monday_distance : ℝ := 3
def monday_speed : ℝ := 6
def tuesday_distance : ℝ := 4
def tuesday_speed : ℝ := 4
def thursday_distance : ℝ := 3
def thursday_speed : ℝ := 3
def saturday_distance : ℝ := 2
def saturday_speed : ℝ := 8

-- Define the constant speed
def constant_speed : ℝ := 5

-- Define the total distance
def total_distance : ℝ := monday_distance + tuesday_distance + thursday_distance + saturday_distance

-- Theorem statement
theorem cycling_time_difference : 
  let actual_time := (monday_distance / monday_speed) + 
                     (tuesday_distance / tuesday_speed) + 
                     (thursday_distance / thursday_speed) + 
                     (saturday_distance / saturday_speed)
  let constant_time := total_distance / constant_speed
  ((actual_time - constant_time) * 60) = 21 := by
  sorry

end NUMINAMATH_CALUDE_cycling_time_difference_l974_97408


namespace NUMINAMATH_CALUDE_accommodation_arrangements_count_l974_97439

/-- Represents the types of rooms in the hotel -/
inductive RoomType
  | Triple
  | Double
  | Single

/-- Represents a person staying in the hotel -/
inductive Person
  | Adult
  | Child

/-- Calculates the number of ways to arrange accommodation for 3 adults and 2 children
    in a hotel with one triple room, one double room, and one single room,
    where children must be accompanied by an adult -/
def accommodationArrangements (rooms : List RoomType) (people : List Person) : Nat :=
  sorry

/-- The main theorem stating that there are 27 different ways to arrange the accommodation -/
theorem accommodation_arrangements_count :
  accommodationArrangements
    [RoomType.Triple, RoomType.Double, RoomType.Single]
    [Person.Adult, Person.Adult, Person.Adult, Person.Child, Person.Child] = 27 :=
by sorry

end NUMINAMATH_CALUDE_accommodation_arrangements_count_l974_97439


namespace NUMINAMATH_CALUDE_min_odd_in_A_P_l974_97416

/-- A polynomial of degree 8 -/
def Polynomial8 : Type := ℝ → ℝ

/-- The set A_P for a polynomial P -/
def A_P (P : Polynomial8) : Set ℝ := {x : ℝ | ∃ c : ℝ, P x = c}

/-- Statement: If 8 is in A_P, then A_P contains at least one odd number -/
theorem min_odd_in_A_P (P : Polynomial8) (h : 8 ∈ A_P P) : 
  ∃ x : ℤ, x % 2 = 1 ∧ (x : ℝ) ∈ A_P P :=
sorry

end NUMINAMATH_CALUDE_min_odd_in_A_P_l974_97416


namespace NUMINAMATH_CALUDE_no_solution_for_pair_C_solutions_for_other_pairs_roots_of_original_equation_l974_97449

theorem no_solution_for_pair_C (x y : ℝ) : ¬(y = x ∧ y = x + 1) := by sorry

theorem solutions_for_other_pairs :
  (∃ x y : ℝ, y = x^2 ∧ y = 5*x - 6 ∧ (x = 2 ∨ x = 3)) ∧
  (∃ x : ℝ, x^2 - 5*x + 6 = 0 ∧ (x = 2 ∨ x = 3)) ∧
  (∃ x y : ℝ, y = x^2 - 5*x + 7 ∧ y = 1 ∧ (x = 2 ∨ x = 3)) ∧
  (∃ x y : ℝ, y = x^2 - 1 ∧ y = 5*x - 7 ∧ (x = 2 ∨ x = 3)) := by sorry

theorem roots_of_original_equation (x : ℝ) : x^2 - 5*x + 6 = 0 ↔ (x = 2 ∨ x = 3) := by sorry

end NUMINAMATH_CALUDE_no_solution_for_pair_C_solutions_for_other_pairs_roots_of_original_equation_l974_97449


namespace NUMINAMATH_CALUDE_pizza_toppings_l974_97404

theorem pizza_toppings (total_slices : ℕ) (pepperoni_slices : ℕ) (mushroom_slices : ℕ) 
  (h1 : total_slices = 14)
  (h2 : pepperoni_slices = 8)
  (h3 : mushroom_slices = 12)
  (h4 : ∀ slice, slice ∈ Finset.range total_slices → 
    (slice ∈ Finset.range pepperoni_slices ∨ slice ∈ Finset.range mushroom_slices)) :
  (pepperoni_slices + mushroom_slices - total_slices : ℕ) = 6 := by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_l974_97404


namespace NUMINAMATH_CALUDE_distance_after_three_minutes_l974_97444

/-- The distance between two vehicles after a given time -/
def distance_between_vehicles (v1 v2 : ℝ) (t : ℝ) : ℝ :=
  (v2 - v1) * t

/-- Theorem: The distance between two vehicles moving at 65 km/h and 85 km/h after 3 minutes is 1 km -/
theorem distance_after_three_minutes :
  let v1 : ℝ := 65  -- Speed of the truck in km/h
  let v2 : ℝ := 85  -- Speed of the car in km/h
  let t : ℝ := 3 / 60  -- 3 minutes converted to hours
  distance_between_vehicles v1 v2 t = 1 := by
  sorry


end NUMINAMATH_CALUDE_distance_after_three_minutes_l974_97444


namespace NUMINAMATH_CALUDE_odd_root_symmetry_l974_97445

theorem odd_root_symmetry (x : ℝ) (n : ℕ) : 
  (x ^ (1 / (2 * n + 1 : ℝ))) = -((-x) ^ (1 / (2 * n + 1 : ℝ))) := by
  sorry

end NUMINAMATH_CALUDE_odd_root_symmetry_l974_97445


namespace NUMINAMATH_CALUDE_total_sum_is_2743_l974_97489

/-- The total sum lent, given the conditions of the problem -/
def total_sum_lent : ℕ := 2743

/-- The second part of the sum -/
def second_part : ℕ := 1688

/-- Calculates the interest for a given principal, rate, and time -/
def interest (principal : ℕ) (rate : ℚ) (time : ℕ) : ℚ :=
  principal * rate * time

/-- Theorem stating that the total sum lent is 2743 -/
theorem total_sum_is_2743 :
  ∃ (first_part : ℕ),
    interest first_part (3/100) 8 = interest second_part (5/100) 3 ∧
    first_part + second_part = total_sum_lent :=
by sorry

end NUMINAMATH_CALUDE_total_sum_is_2743_l974_97489


namespace NUMINAMATH_CALUDE_odd_function_value_at_one_l974_97405

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + (a-5)*x^2 + a*x

-- State the theorem
theorem odd_function_value_at_one :
  ∀ a : ℝ, (∀ x : ℝ, f a (-x) = -(f a x)) → f a 1 = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_odd_function_value_at_one_l974_97405


namespace NUMINAMATH_CALUDE_system_solvability_l974_97457

-- Define the system of equations
def system (x y a : ℝ) : Prop :=
  (x - a)^2 = 4*(y - x + a - 1) ∧
  x ≠ 1 ∧ x > 0 ∧
  (Real.sqrt y - 1) / (Real.sqrt x - 1) = 1

-- Define the solution set for a
def solution_set (a : ℝ) : Prop :=
  a > 1 ∧ a ≠ 5

-- Theorem statement
theorem system_solvability (a : ℝ) :
  (∃ x y, system x y a) ↔ solution_set a :=
sorry

end NUMINAMATH_CALUDE_system_solvability_l974_97457


namespace NUMINAMATH_CALUDE_ordering_of_a_b_c_l974_97448

theorem ordering_of_a_b_c :
  let a : ℝ := (2 : ℝ) ^ (4/3)
  let b : ℝ := (4 : ℝ) ^ (2/5)
  let c : ℝ := (5 : ℝ) ^ (2/3)
  c > a ∧ a > b := by sorry

end NUMINAMATH_CALUDE_ordering_of_a_b_c_l974_97448


namespace NUMINAMATH_CALUDE_area_PTW_approx_34_l974_97400

-- Define the areas of triangles as functions of x
def area_PUW (x : ℝ) : ℝ := 4*x + 4
def area_SUW (x : ℝ) : ℝ := 2*x + 20
def area_SVW (x : ℝ) : ℝ := 5*x + 20
def area_SVR (x : ℝ) : ℝ := 5*x + 11
def area_QVR (x : ℝ) : ℝ := 8*x + 32
def area_QVW (x : ℝ) : ℝ := 8*x + 50

-- Define the equation for solving x
def solve_for_x (x : ℝ) : Prop :=
  (area_QVW x) / (area_SVW x) = (area_QVR x) / (area_SVR x)

-- Define the area of triangle PTW
noncomputable def area_PTW (x : ℝ) : ℝ := 
  sorry  -- The exact formula is not provided in the problem

-- State the theorem
theorem area_PTW_approx_34 :
  ∃ (x : ℝ), solve_for_x x ∧ 
  (∀ (y : ℝ), abs (area_PTW x - 34) ≤ abs (area_PTW x - y) ∨ y = 34) :=
sorry

end NUMINAMATH_CALUDE_area_PTW_approx_34_l974_97400


namespace NUMINAMATH_CALUDE_select_two_with_boy_l974_97458

/-- The number of ways to select 2 people from 4 boys and 2 girls, with at least one boy -/
def select_with_boy (total : ℕ) (boys : ℕ) (girls : ℕ) (to_select : ℕ) : ℕ :=
  Nat.choose total to_select - Nat.choose girls to_select

theorem select_two_with_boy :
  select_with_boy 6 4 2 2 = 14 :=
by sorry

end NUMINAMATH_CALUDE_select_two_with_boy_l974_97458


namespace NUMINAMATH_CALUDE_quadratic_inequality_l974_97496

theorem quadratic_inequality (x : ℝ) : -9 * x^2 + 6 * x + 8 > 0 ↔ -2/3 < x ∧ x < 4/3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l974_97496


namespace NUMINAMATH_CALUDE_cos_pi_minus_2theta_l974_97455

theorem cos_pi_minus_2theta (θ : Real) (h : ∃ (x y : Real), x = 3 ∧ y = -4 ∧ x = Real.cos θ * Real.sqrt (x^2 + y^2) ∧ y = Real.sin θ * Real.sqrt (x^2 + y^2)) :
  Real.cos (π - 2*θ) = 7/25 := by
sorry

end NUMINAMATH_CALUDE_cos_pi_minus_2theta_l974_97455


namespace NUMINAMATH_CALUDE_no_solutions_for_divisor_sum_equation_l974_97413

/-- Sum of positive divisors function -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- Theorem: No integers between 1 and 10000 satisfy f(i) = 1 + 2√i + i -/
theorem no_solutions_for_divisor_sum_equation :
  ∀ i : ℕ, 1 ≤ i ∧ i ≤ 10000 →
    sum_of_divisors i ≠ 1 + 2 * (Int.sqrt i).toNat + i := by
  sorry

end NUMINAMATH_CALUDE_no_solutions_for_divisor_sum_equation_l974_97413


namespace NUMINAMATH_CALUDE_abigail_cans_collected_l974_97421

/-- Given:
  - The total number of cans needed is 100
  - Alyssa has collected 30 cans
  - They still need to collect 27 more cans
  Prove that Abigail has collected 43 cans -/
theorem abigail_cans_collected 
  (total_cans : ℕ) 
  (alyssa_cans : ℕ) 
  (more_cans_needed : ℕ) 
  (h1 : total_cans = 100)
  (h2 : alyssa_cans = 30)
  (h3 : more_cans_needed = 27) :
  total_cans - (alyssa_cans + more_cans_needed) = 43 := by
  sorry

end NUMINAMATH_CALUDE_abigail_cans_collected_l974_97421


namespace NUMINAMATH_CALUDE_sin_alpha_value_l974_97427

theorem sin_alpha_value (α : Real) 
  (h1 : α > -π/2 ∧ α < π/2)
  (h2 : Real.tan α = Real.sin (76 * π / 180) * Real.cos (46 * π / 180) - 
                     Real.cos (76 * π / 180) * Real.sin (46 * π / 180)) : 
  Real.sin α = Real.sqrt 5 / 5 := by
sorry

end NUMINAMATH_CALUDE_sin_alpha_value_l974_97427


namespace NUMINAMATH_CALUDE_nested_subtraction_simplification_l974_97433

theorem nested_subtraction_simplification (x : ℝ) : 1 - (2 - (3 - (4 - (5 - x)))) = 3 - x := by
  sorry

end NUMINAMATH_CALUDE_nested_subtraction_simplification_l974_97433


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l974_97498

theorem simplify_sqrt_expression : Real.sqrt (68 - 28 * Real.sqrt 2) = 6 - 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l974_97498


namespace NUMINAMATH_CALUDE_coffee_shop_sales_l974_97468

/-- Represents the number of lattes sold by the coffee shop. -/
def lattes : ℕ := sorry

/-- Represents the number of teas sold by the coffee shop. -/
def teas : ℕ := 6

/-- The relationship between lattes and teas sold. -/
axiom latte_tea_relation : lattes = 4 * teas + 8

theorem coffee_shop_sales : lattes = 32 := by sorry

end NUMINAMATH_CALUDE_coffee_shop_sales_l974_97468


namespace NUMINAMATH_CALUDE_total_nailcutter_sounds_l974_97431

/-- The number of nails per person (fingers and toes combined) -/
def nails_per_person : ℕ := 20

/-- The number of customers -/
def num_customers : ℕ := 3

/-- The number of sounds produced when trimming one nail -/
def sounds_per_nail : ℕ := 1

/-- Theorem: The total number of nailcutter sounds for 3 customers is 60 -/
theorem total_nailcutter_sounds :
  nails_per_person * num_customers * sounds_per_nail = 60 := by
  sorry

end NUMINAMATH_CALUDE_total_nailcutter_sounds_l974_97431


namespace NUMINAMATH_CALUDE_championship_assignments_l974_97456

theorem championship_assignments (n_students : ℕ) (n_titles : ℕ) :
  n_students = 4 → n_titles = 3 →
  (n_students ^ n_titles : ℕ) = 64 := by
  sorry

end NUMINAMATH_CALUDE_championship_assignments_l974_97456


namespace NUMINAMATH_CALUDE_man_swimming_speed_l974_97481

/-- The speed of a man in still water given his downstream and upstream swimming times and distances -/
theorem man_swimming_speed
  (downstream_distance : ℝ)
  (downstream_time : ℝ)
  (upstream_distance : ℝ)
  (upstream_time : ℝ)
  (h1 : downstream_distance = 50)
  (h2 : downstream_time = 4)
  (h3 : upstream_distance = 30)
  (h4 : upstream_time = 6)
  : ∃ (v_m : ℝ), v_m = 8.75 ∧ 
    downstream_distance / downstream_time = v_m + (downstream_distance / downstream_time - v_m) ∧
    upstream_distance / upstream_time = v_m - (downstream_distance / downstream_time - v_m) :=
by
  sorry

#check man_swimming_speed

end NUMINAMATH_CALUDE_man_swimming_speed_l974_97481


namespace NUMINAMATH_CALUDE_area_enclosed_by_graph_l974_97428

/-- The area enclosed by the graph of |x| + |3y| = 12 -/
def areaEnclosedByGraph : ℝ := 96

/-- The equation of the graph -/
def graphEquation (x y : ℝ) : Prop := (abs x) + (abs (3 * y)) = 12

theorem area_enclosed_by_graph :
  areaEnclosedByGraph = 96 := by sorry

end NUMINAMATH_CALUDE_area_enclosed_by_graph_l974_97428


namespace NUMINAMATH_CALUDE_library_book_distribution_l974_97471

/-- The number of ways to distribute books between the library and checked-out status -/
def distribution_count (total : ℕ) (min_in_library : ℕ) (min_checked_out : ℕ) : ℕ :=
  (total - min_in_library - min_checked_out + 1)

/-- Theorem: For 8 identical books with at least 2 in the library and 2 checked out,
    there are 5 different ways to distribute the books -/
theorem library_book_distribution :
  distribution_count 8 2 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_library_book_distribution_l974_97471


namespace NUMINAMATH_CALUDE_correct_registration_sequence_l974_97487

-- Define the registration steps
inductive RegistrationStep
  | collectTicket
  | register
  | takeTests
  | takePhoto

-- Define a type for sequences of registration steps
def RegistrationSequence := List RegistrationStep

-- Define the given sequence of steps
def givenSequence : RegistrationSequence := 
  [RegistrationStep.register, RegistrationStep.takePhoto, 
   RegistrationStep.collectTicket, RegistrationStep.takeTests]

-- Define a function to check if a sequence is correct
def isCorrectSequence (seq : RegistrationSequence) : Prop :=
  seq = givenSequence

-- Theorem stating that the given sequence is correct
theorem correct_registration_sequence :
  isCorrectSequence givenSequence := by
  sorry

end NUMINAMATH_CALUDE_correct_registration_sequence_l974_97487


namespace NUMINAMATH_CALUDE_max_basketballs_part1_max_basketballs_part2_l974_97485

/-- Represents the prices and quantities of basketballs and soccer balls -/
structure BallPurchase where
  basketball_price : ℕ
  soccer_ball_price : ℕ
  basketball_quantity : ℕ
  soccer_ball_quantity : ℕ

/-- Calculates the total cost of the purchase -/
def total_cost (purchase : BallPurchase) : ℕ :=
  purchase.basketball_price * purchase.basketball_quantity +
  purchase.soccer_ball_price * purchase.soccer_ball_quantity

/-- Calculates the total quantity of balls purchased -/
def total_quantity (purchase : BallPurchase) : ℕ :=
  purchase.basketball_quantity + purchase.soccer_ball_quantity

/-- Theorem for part 1 of the problem -/
theorem max_basketballs_part1 (purchase : BallPurchase) 
  (h1 : purchase.basketball_price = 100)
  (h2 : purchase.soccer_ball_price = 80)
  (h3 : total_cost purchase = 5600)
  (h4 : total_quantity purchase = 60) :
  purchase.basketball_quantity = 40 ∧ purchase.soccer_ball_quantity = 20 := by
  sorry

/-- Theorem for part 2 of the problem -/
theorem max_basketballs_part2 (purchase : BallPurchase) 
  (h1 : purchase.basketball_price = 100)
  (h2 : purchase.soccer_ball_price = 80)
  (h3 : total_cost purchase ≤ 6890)
  (h4 : total_quantity purchase = 80) :
  purchase.basketball_quantity ≤ 24 := by
  sorry

end NUMINAMATH_CALUDE_max_basketballs_part1_max_basketballs_part2_l974_97485


namespace NUMINAMATH_CALUDE_remainder_divisibility_l974_97403

theorem remainder_divisibility (N : ℤ) : 
  N % 342 = 47 → N % 19 = 9 := by
  sorry

end NUMINAMATH_CALUDE_remainder_divisibility_l974_97403


namespace NUMINAMATH_CALUDE_fixed_internet_charge_is_4_l974_97473

/-- Represents Elvin's monthly telephone bill structure -/
structure MonthlyBill where
  callCharge : ℝ
  internetCharge : ℝ

/-- Calculates the total bill amount -/
def totalBill (bill : MonthlyBill) : ℝ :=
  bill.callCharge + bill.internetCharge

theorem fixed_internet_charge_is_4
  (january : MonthlyBill)
  (february : MonthlyBill)
  (h1 : totalBill january = 40)
  (h2 : totalBill february = 76)
  (h3 : february.callCharge = 2 * january.callCharge)
  (h4 : january.internetCharge = february.internetCharge) :
  january.internetCharge = 4 := by
  sorry

#check fixed_internet_charge_is_4

end NUMINAMATH_CALUDE_fixed_internet_charge_is_4_l974_97473


namespace NUMINAMATH_CALUDE_sum_of_integers_l974_97429

theorem sum_of_integers (x y : ℕ+) (h1 : x.val - y.val = 18) (h2 : x.val * y.val = 98) :
  x.val + y.val = 2 * Real.sqrt 179 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l974_97429


namespace NUMINAMATH_CALUDE_record_storage_space_theorem_l974_97479

/-- Represents the dimensions of a box in inches -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℝ :=
  d.length * d.width * d.height

theorem record_storage_space_theorem 
  (box_dims : BoxDimensions)
  (storage_cost_per_box : ℝ)
  (total_monthly_payment : ℝ)
  (h1 : box_dims.length = 15)
  (h2 : box_dims.width = 12)
  (h3 : box_dims.height = 10)
  (h4 : storage_cost_per_box = 0.2)
  (h5 : total_monthly_payment = 120) :
  (total_monthly_payment / storage_cost_per_box) * boxVolume box_dims = 1080000 := by
  sorry

#check record_storage_space_theorem

end NUMINAMATH_CALUDE_record_storage_space_theorem_l974_97479


namespace NUMINAMATH_CALUDE_number_square_equation_l974_97411

theorem number_square_equation : ∃ x : ℝ, x^2 + 100 = (x - 20)^2 ∧ x = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_number_square_equation_l974_97411


namespace NUMINAMATH_CALUDE_sin_graph_transformation_l974_97426

theorem sin_graph_transformation :
  ∀ (x y : ℝ),
  (y = Real.sin x) →
  (∃ (x' y' : ℝ),
    x' = 2 * (x + π / 10) ∧
    y' = y ∧
    y' = Real.sin (x' / 2 - π / 10)) :=
by sorry

end NUMINAMATH_CALUDE_sin_graph_transformation_l974_97426


namespace NUMINAMATH_CALUDE_smallest_prime_with_digit_sum_23_l974_97460

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, 1 < d → d < n → ¬(d ∣ n)

theorem smallest_prime_with_digit_sum_23 :
  ∀ p : ℕ, is_prime p → digit_sum p = 23 → p ≥ 757 :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_with_digit_sum_23_l974_97460


namespace NUMINAMATH_CALUDE_fuel_tank_capacity_l974_97478

/-- Given a fuel tank partially filled with fuel A and then filled to capacity with fuel B,
    prove that the capacity of the tank is 162.5 gallons. -/
theorem fuel_tank_capacity
  (capacity : ℝ)
  (fuel_a_volume : ℝ)
  (fuel_a_ethanol_percent : ℝ)
  (fuel_b_ethanol_percent : ℝ)
  (total_ethanol : ℝ)
  (h1 : fuel_a_volume = 49.99999999999999)
  (h2 : fuel_a_ethanol_percent = 0.12)
  (h3 : fuel_b_ethanol_percent = 0.16)
  (h4 : total_ethanol = 30)
  (h5 : fuel_a_ethanol_percent * fuel_a_volume +
        fuel_b_ethanol_percent * (capacity - fuel_a_volume) = total_ethanol) :
  capacity = 162.5 := by
  sorry

end NUMINAMATH_CALUDE_fuel_tank_capacity_l974_97478


namespace NUMINAMATH_CALUDE_power_function_value_l974_97494

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, f x = x ^ a

-- Define the theorem
theorem power_function_value (f : ℝ → ℝ) :
  isPowerFunction f → f 2 = 4 → f (-3) = 9 := by
  sorry

end NUMINAMATH_CALUDE_power_function_value_l974_97494


namespace NUMINAMATH_CALUDE_total_birds_is_148_l974_97463

/-- The number of birds seen on Monday -/
def monday_birds : ℕ := 70

/-- The number of birds seen on Tuesday -/
def tuesday_birds : ℕ := monday_birds / 2

/-- The number of birds seen on Wednesday -/
def wednesday_birds : ℕ := tuesday_birds + 8

/-- The total number of birds seen from Monday to Wednesday -/
def total_birds : ℕ := monday_birds + tuesday_birds + wednesday_birds

/-- Theorem stating that the total number of birds seen is 148 -/
theorem total_birds_is_148 : total_birds = 148 := by
  sorry

end NUMINAMATH_CALUDE_total_birds_is_148_l974_97463


namespace NUMINAMATH_CALUDE_series_solution_l974_97406

/-- The sum of the infinite geometric series with first term a and common ratio r -/
noncomputable def geometric_sum (a : ℝ) (r : ℝ) : ℝ := a / (1 - r)

/-- The series in question -/
noncomputable def series (k : ℝ) : ℝ :=
  5 + geometric_sum ((5 + k) / 3) (1 / 3)

theorem series_solution :
  ∃ k : ℝ, series k = 15 ∧ k = 7.5 := by sorry

end NUMINAMATH_CALUDE_series_solution_l974_97406


namespace NUMINAMATH_CALUDE_xy_value_l974_97424

theorem xy_value (x y : ℤ) (h : (30 : ℚ) / 2 * (x * y) = 21 * x + 20 * y - 13) : x * y = 6 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l974_97424


namespace NUMINAMATH_CALUDE_vasims_share_l974_97422

/-- Represents the distribution of money among three people -/
structure Distribution where
  faruk : ℕ
  vasim : ℕ
  ranjith : ℕ

/-- Checks if the distribution follows the given ratio -/
def is_valid_ratio (d : Distribution) : Prop :=
  11 * d.faruk = 3 * d.ranjith ∧ 5 * d.faruk = 3 * d.vasim

/-- The main theorem to prove -/
theorem vasims_share (d : Distribution) :
  is_valid_ratio d → d.ranjith - d.faruk = 2400 → d.vasim = 1500 := by
  sorry


end NUMINAMATH_CALUDE_vasims_share_l974_97422


namespace NUMINAMATH_CALUDE_unique_three_digit_number_divisible_by_11_l974_97420

theorem unique_three_digit_number_divisible_by_11 :
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 
  n % 10 = 7 ∧ 
  (n / 100) % 10 = 8 ∧ 
  n % 11 = 0 ∧
  n = 847 :=
by sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_divisible_by_11_l974_97420
