import Mathlib

namespace NUMINAMATH_GPT_total_balls_l314_31400

theorem total_balls (black_balls : ℕ) (prob_pick_black : ℚ) (total_balls : ℕ) :
  black_balls = 4 → prob_pick_black = 1 / 3 → total_balls = 12 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_total_balls_l314_31400


namespace NUMINAMATH_GPT_enough_cat_food_for_six_days_l314_31490

variable (B S : ℝ)

theorem enough_cat_food_for_six_days :
  (B > S) →
  (B < 2 * S) →
  (B + 2 * S = 2 * ((B + 2 * S) / 2)) →
  ¬ (4 * B + 4 * S ≥ 6 * ((B + 2 * S) / 2)) :=
by
  -- Proof logic goes here.
  sorry

end NUMINAMATH_GPT_enough_cat_food_for_six_days_l314_31490


namespace NUMINAMATH_GPT_largest_n_l314_31497

variable {a : ℕ → ℤ}
variable {S : ℕ → ℤ}

-- Conditions
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 1) - a n = a 2 - a 1

axiom a1_gt_zero : a 1 > 0
axiom a2011_a2012_sum_gt_zero : a 2011 + a 2012 > 0
axiom a2011_a2012_prod_lt_zero : a 2011 * a 2012 < 0

-- Sum of first n terms of an arithmetic sequence
def sequence_sum (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  n * (a 1 + a n) / 2

-- Problem statement to prove
theorem largest_n (H : is_arithmetic_sequence a) :
  ∀ n, (sequence_sum a 4022 > 0) ∧ (sequence_sum a 4023 < 0) → n = 4022 := by
  sorry

end NUMINAMATH_GPT_largest_n_l314_31497


namespace NUMINAMATH_GPT_min_value_y_l314_31445

theorem min_value_y (x : ℝ) (h : x > 0) : ∃ y, y = x + 4 / x^2 ∧ (∀ z, z = x + 4 / x^2 → y ≤ z) := 
sorry

end NUMINAMATH_GPT_min_value_y_l314_31445


namespace NUMINAMATH_GPT_minimum_value_l314_31413

noncomputable def expr (x y : ℝ) := x^2 + x * y + y^2 - 3 * y

theorem minimum_value :
  ∃ x y : ℝ, expr x y = -3 ∧
  ∀ x' y' : ℝ, expr x' y' ≥ -3 :=
sorry

end NUMINAMATH_GPT_minimum_value_l314_31413


namespace NUMINAMATH_GPT_product_469157_9999_l314_31415

theorem product_469157_9999 : 469157 * 9999 = 4690872843 := by
  -- computation and its proof would go here
  sorry

end NUMINAMATH_GPT_product_469157_9999_l314_31415


namespace NUMINAMATH_GPT_arithmetic_geometric_sequence_l314_31433

-- Let {a_n} be an arithmetic sequence
-- And let a_1, a_2, a_3 form a geometric sequence
-- Given that a_5 = 1, we aim to prove that a_10 = 1
theorem arithmetic_geometric_sequence (a : ℕ → ℝ)
  (h_arith : ∀ n, a (n + 1) - a n = a 2 - a 1)
  (h_geom : a 1 * a 3 = (a 2) ^ 2)
  (h_a5 : a 5 = 1) :
  a 10 = 1 :=
sorry

end NUMINAMATH_GPT_arithmetic_geometric_sequence_l314_31433


namespace NUMINAMATH_GPT_cos_2alpha_l314_31460

theorem cos_2alpha (α : ℝ) (h : Real.cos (Real.pi / 2 + α) = (1 : ℝ) / 3) : 
  Real.cos (2 * α) = (7 : ℝ) / 9 := 
by
  sorry

end NUMINAMATH_GPT_cos_2alpha_l314_31460


namespace NUMINAMATH_GPT_equation_holds_iff_b_eq_c_l314_31416

theorem equation_holds_iff_b_eq_c (a b c : ℕ) (h1 : a < 10) (h2 : b < 10) (h3 : c < 10) (h4 : a ≠ b) (h5 : a ≠ c) (h6 : b ≠ c) :
  (10 * a + b + 1) * (10 * a + c) = 100 * a * a + 100 * a + b + c ↔ b = c :=
by sorry

end NUMINAMATH_GPT_equation_holds_iff_b_eq_c_l314_31416


namespace NUMINAMATH_GPT_light_travel_50_years_l314_31422

theorem light_travel_50_years :
  let one_year_distance := 9460800000000 -- distance light travels in one year
  let fifty_years_distance := 50 * one_year_distance
  let scientific_notation_distance := 473.04 * 10^12
  fifty_years_distance = scientific_notation_distance :=
by
  sorry

end NUMINAMATH_GPT_light_travel_50_years_l314_31422


namespace NUMINAMATH_GPT_largest_square_side_l314_31444

theorem largest_square_side {m n : ℕ} (h1 : m = 72) (h2 : n = 90) : Nat.gcd m n = 18 :=
by
  sorry

end NUMINAMATH_GPT_largest_square_side_l314_31444


namespace NUMINAMATH_GPT_percentage_of_non_technicians_l314_31489

theorem percentage_of_non_technicians (total_workers technicians non_technicians permanent_technicians permanent_non_technicians temporary_workers : ℝ)
  (h1 : technicians = 0.5 * total_workers)
  (h2 : non_technicians = total_workers - technicians)
  (h3 : permanent_technicians = 0.5 * technicians)
  (h4 : permanent_non_technicians = 0.5 * non_technicians)
  (h5 : temporary_workers = 0.5 * total_workers) :
  (non_technicians / total_workers) * 100 = 50 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_percentage_of_non_technicians_l314_31489


namespace NUMINAMATH_GPT_interest_difference_l314_31447

noncomputable def principal : ℝ := 6200
noncomputable def rate : ℝ := 5 / 100
noncomputable def time : ℝ := 10

noncomputable def interest (P R T : ℝ) : ℝ := P * R * T / 100

theorem interest_difference :
  (principal - interest principal rate time) = 3100 := 
by
  sorry

end NUMINAMATH_GPT_interest_difference_l314_31447


namespace NUMINAMATH_GPT_find_altitude_to_hypotenuse_l314_31410

-- define the conditions
def area : ℝ := 540
def hypotenuse : ℝ := 36
def altitude : ℝ := 30

-- define the problem statement
theorem find_altitude_to_hypotenuse (A : ℝ) (c : ℝ) (h : ℝ) 
  (h_area : A = 540) (h_hypotenuse : c = 36) : h = 30 :=
by
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_find_altitude_to_hypotenuse_l314_31410


namespace NUMINAMATH_GPT_compare_negatives_l314_31418

theorem compare_negatives : -3 < -2 := 
by { sorry }

end NUMINAMATH_GPT_compare_negatives_l314_31418


namespace NUMINAMATH_GPT_evaluate_expression_l314_31403

theorem evaluate_expression : 150 * (150 - 5) - (150 * 150 - 7) = -743 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l314_31403


namespace NUMINAMATH_GPT_perimeter_equal_l314_31402

theorem perimeter_equal (x : ℕ) (hx : x = 4)
    (side_square : ℕ := x + 2) 
    (side_triangle : ℕ := 2 * x) 
    (perimeter_square : ℕ := 4 * side_square)
    (perimeter_triangle : ℕ := 3 * side_triangle) :
    perimeter_square = perimeter_triangle :=
by
    -- Given x = 4
    -- Calculate side lengths
    -- side_square = x + 2 = 4 + 2 = 6
    -- side_triangle = 2 * x = 2 * 4 = 8
    -- Calculate perimeters
    -- perimeter_square = 4 * side_square = 4 * 6 = 24
    -- perimeter_triangle = 3 * side_triangle = 3 * 8 = 24
    -- Therefore, perimeter_square = perimeter_triangle = 24
    sorry

end NUMINAMATH_GPT_perimeter_equal_l314_31402


namespace NUMINAMATH_GPT_blue_balloons_l314_31467

theorem blue_balloons (total_balloons red_balloons green_balloons purple_balloons : ℕ)
  (h1 : total_balloons = 135)
  (h2 : red_balloons = 45)
  (h3 : green_balloons = 27)
  (h4 : purple_balloons = 32) :
  total_balloons - (red_balloons + green_balloons + purple_balloons) = 31 :=
by
  sorry

end NUMINAMATH_GPT_blue_balloons_l314_31467


namespace NUMINAMATH_GPT_lines_parallel_m_values_l314_31471

theorem lines_parallel_m_values (m : ℝ) :
    (∀ x y : ℝ, (m - 2) * x - y - 1 = 0 ↔ 3 * x - m * y = 0) ↔ (m = -1 ∨ m = 3) :=
by
  sorry

end NUMINAMATH_GPT_lines_parallel_m_values_l314_31471


namespace NUMINAMATH_GPT_tan_15_simplification_l314_31488

theorem tan_15_simplification :
  (1 + Real.tan (Real.pi / 12)) / (1 - Real.tan (Real.pi / 12)) = Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_tan_15_simplification_l314_31488


namespace NUMINAMATH_GPT_only_odd_digit_squared_n_l314_31475

def is_odd_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

def has_only_odd_digits (n : ℕ) : Prop :=
  ∀ (d : ℕ), d ∈ n.digits 10 → is_odd_digit d

theorem only_odd_digit_squared_n (n : ℕ) :
  0 < n ∧ has_only_odd_digits (n * n) ↔ n = 1 ∨ n = 3 :=
sorry

end NUMINAMATH_GPT_only_odd_digit_squared_n_l314_31475


namespace NUMINAMATH_GPT_cafeteria_pies_l314_31451

theorem cafeteria_pies (total_apples handed_out_apples apples_per_pie : ℕ) (h1 : total_apples = 47) (h2 : handed_out_apples = 27) (h3 : apples_per_pie = 4) :
  (total_apples - handed_out_apples) / apples_per_pie = 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_cafeteria_pies_l314_31451


namespace NUMINAMATH_GPT_maximize_profit_l314_31483

-- Define the variables
variables (x y a b : ℝ)
variables (P : ℝ)

-- Define the conditions and the proof goal
theorem maximize_profit
  (h1 : x + 3 * y = 240)
  (h2 : 2 * x + y = 130)
  (h3 : a + b = 100)
  (h4 : a ≥ 4 * b)
  (ha : a = 80)
  (hb : b = 20) :
  x = 30 ∧ y = 70 ∧ P = (40 * a + 90 * b) - (30 * a + 70 * b) := 
by
  -- We assume the solution steps are solved correctly as provided
  sorry

end NUMINAMATH_GPT_maximize_profit_l314_31483


namespace NUMINAMATH_GPT_subcommittee_count_l314_31414

def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem subcommittee_count : 
  let R := 10
  let D := 4
  let subR := 4
  let subD := 2
  binomial R subR * binomial D subD = 1260 := 
by
  sorry

end NUMINAMATH_GPT_subcommittee_count_l314_31414


namespace NUMINAMATH_GPT_min_objective_value_l314_31485

theorem min_objective_value (x y : ℝ) 
  (h1 : x + y ≥ 2) 
  (h2 : x - y ≤ 2) 
  (h3 : y ≥ 1) : ∃ (z : ℝ), z = x + 3 * y ∧ z = 4 :=
by
  -- Provided proof omitted
  sorry

end NUMINAMATH_GPT_min_objective_value_l314_31485


namespace NUMINAMATH_GPT_digit_equation_l314_31462

-- Definitions for digits and the equation components
def is_digit (x : ℤ) : Prop := 0 ≤ x ∧ x ≤ 9

def three_digit_number (A B C : ℤ) : ℤ := 100 * A + 10 * B + C
def two_digit_number (A D : ℤ) : ℤ := 10 * A + D
def four_digit_number (A D C : ℤ) : ℤ := 1000 * A + 100 * D + 10 * D + C

-- Statement of the theorem
theorem digit_equation (A B C D : ℤ) (hA : is_digit A) (hB : is_digit B) (hC : is_digit C) (hD : is_digit D) :
  three_digit_number A B C * two_digit_number A D = four_digit_number A D C :=
sorry

end NUMINAMATH_GPT_digit_equation_l314_31462


namespace NUMINAMATH_GPT_fg_of_3_eq_29_l314_31455

def g (x : ℕ) : ℕ := x * x
def f (x : ℕ) : ℕ := 3 * x + 2

theorem fg_of_3_eq_29 : f (g 3) = 29 :=
by
  sorry

end NUMINAMATH_GPT_fg_of_3_eq_29_l314_31455


namespace NUMINAMATH_GPT_domain_of_sqrt_function_l314_31466

theorem domain_of_sqrt_function :
  {x : ℝ | (1 / (Real.log x / Real.log 2) - 2 ≥ 0) ∧ (x > 0) ∧ (x ≠ 1)} 
  = {x : ℝ | 1 < x ∧ x ≤ Real.sqrt 10} :=
sorry

end NUMINAMATH_GPT_domain_of_sqrt_function_l314_31466


namespace NUMINAMATH_GPT_find_abc_l314_31438

noncomputable def f (a b c x : ℝ) := x^3 + a*x^2 + b*x + c
noncomputable def f' (a b x : ℝ) := 3*x^2 + 2*a*x + b

theorem find_abc (a b c : ℝ) :
  (f' a b -2 = 0) ∧
  (f' a b 1 = -3) ∧
  (f a b c 1 = 0) →
  a = 1 ∧ b = -8 ∧ c = 6 :=
sorry

end NUMINAMATH_GPT_find_abc_l314_31438


namespace NUMINAMATH_GPT_net_percentage_change_is_correct_l314_31480

def initial_price : Float := 100.0

def price_after_first_year (initial: Float) := initial * (1 - 0.05)

def price_after_second_year (price1: Float) := price1 * (1 + 0.10)

def price_after_third_year (price2: Float) := price2 * (1 + 0.04)

def price_after_fourth_year (price3: Float) := price3 * (1 - 0.03)

def price_after_fifth_year (price4: Float) := price4 * (1 + 0.08)

def final_price := price_after_fifth_year (price_after_fourth_year (price_after_third_year (price_after_second_year (price_after_first_year initial_price))))

def net_percentage_change (initial final: Float) := ((final - initial) / initial) * 100

theorem net_percentage_change_is_correct :
  net_percentage_change initial_price final_price = 13.85 := by
  sorry

end NUMINAMATH_GPT_net_percentage_change_is_correct_l314_31480


namespace NUMINAMATH_GPT_find_abc_integers_l314_31417

theorem find_abc_integers (a b c : ℕ) (h1 : 1 < a) (h2 : a < b) (h3 : b < c) 
(h4 : (a - 1) * (b - 1) * (c - 1) ∣ a * b * c - 1) : (a = 3 ∧ b = 5 ∧ c = 15) ∨ 
(a = 2 ∧ b = 4 ∧ c = 8) :=
sorry

end NUMINAMATH_GPT_find_abc_integers_l314_31417


namespace NUMINAMATH_GPT_two_numbers_max_product_l314_31406

theorem two_numbers_max_product :
  ∃ x y : ℝ, x - y = 4 ∧ x + y = 35 ∧ ∀ z w : ℝ, z - w = 4 → z + w = 35 → z * w ≤ x * y :=
by
  sorry

end NUMINAMATH_GPT_two_numbers_max_product_l314_31406


namespace NUMINAMATH_GPT_no_integer_triplets_for_equation_l314_31469

theorem no_integer_triplets_for_equation (a b c : ℤ) : ¬ (a^2 + b^2 + 1 = 4 * c) :=
by
  sorry

end NUMINAMATH_GPT_no_integer_triplets_for_equation_l314_31469


namespace NUMINAMATH_GPT_simplify_expression_l314_31429

theorem simplify_expression (x : ℝ) (h₁ : x ≠ -1) (h₂ : x ≠ 1) :
  ( ((x+1)^2 * (x^2 - x + 1)^2 / (x^3 + 1)^2)^2 *
    ((x-1)^2 * (x^2 + x + 1)^2 / (x^3 - 1)^2)^2
  ) = 1 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l314_31429


namespace NUMINAMATH_GPT_find_real_root_a_l314_31423

theorem find_real_root_a (a b c : ℂ) (ha : a.im = 0) (h1 : a + b + c = 5) (h2 : a * b + b * c + c * a = 7) (h3 : a * b * c = 3) : a = 1 :=
sorry

end NUMINAMATH_GPT_find_real_root_a_l314_31423


namespace NUMINAMATH_GPT_min_value_frac_sum_l314_31426

theorem min_value_frac_sum (m n : ℝ) (h1 : 2 * m + n = 2) (h2 : m * n > 0) : 
  ∃ c : ℝ, c = 4 ∧ (∀ m n, 2 * m + n = 2 → m * n > 0 → (1 / m + 2 / n) ≥ c) :=
sorry

end NUMINAMATH_GPT_min_value_frac_sum_l314_31426


namespace NUMINAMATH_GPT_gambler_final_amount_l314_31484

-- Define initial amount of money
def initial_amount := 100

-- Define the multipliers
def win_multiplier := 4 / 3
def loss_multiplier := 2 / 3
def double_win_multiplier := 5 / 3

-- Define the gambler scenario (WWLWLWLW)
def scenario := [double_win_multiplier, win_multiplier, loss_multiplier, win_multiplier, loss_multiplier, win_multiplier, loss_multiplier, win_multiplier]

-- Function to compute final amount given initial amount, number of wins and losses, and the scenario
def final_amount (initial: ℚ) (multipliers: List ℚ) : ℚ :=
  multipliers.foldl (· * ·) initial

-- Prove that the final amount after all multipliers are applied is approximately equal to 312.12
theorem gambler_final_amount : abs (final_amount initial_amount scenario - 312.12) < 0.01 :=
by
  sorry

end NUMINAMATH_GPT_gambler_final_amount_l314_31484


namespace NUMINAMATH_GPT_distance_3_units_l314_31482

theorem distance_3_units (x : ℤ) (h : |x + 2| = 3) : x = -5 ∨ x = 1 := by
  sorry

end NUMINAMATH_GPT_distance_3_units_l314_31482


namespace NUMINAMATH_GPT_range_of_a_l314_31463

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 2 * a * x + 1 > 0) ↔ (0 ≤ a ∧ a < 1) := 
sorry

end NUMINAMATH_GPT_range_of_a_l314_31463


namespace NUMINAMATH_GPT_distance_between_parallel_lines_l314_31456

theorem distance_between_parallel_lines
  (O A B C D P Q : ℝ) -- Points on the circle with P and Q as defined midpoints
  (r d : ℝ) -- Radius of the circle and distance between the parallel lines
  (h_AB : dist A B = 36) -- Length of chord AB
  (h_CD : dist C D = 36) -- Length of chord CD
  (h_BC : dist B C = 40) -- Length of chord BC
  (h_OA : dist O A = r) 
  (h_OB : dist O B = r)
  (h_OC : dist O C = r)
  (h_PQ_parallel : dist P Q = d) -- Midpoints
  : d = 4 * Real.sqrt 19 / 3 :=
sorry

end NUMINAMATH_GPT_distance_between_parallel_lines_l314_31456


namespace NUMINAMATH_GPT_right_triangles_with_leg_2012_l314_31464

theorem right_triangles_with_leg_2012 :
  ∀ (a b c : ℕ), a = 2012 ∧ a ^ 2 + b ^ 2 = c ^ 2 → 
  (b = 253005 ∧ c = 253013) ∨ 
  (b = 506016 ∧ c = 506020) ∨ 
  (b = 1012035 ∧ c = 1012037) ∨ 
  (b = 1509 ∧ c = 2515) :=
by
  intros
  sorry

end NUMINAMATH_GPT_right_triangles_with_leg_2012_l314_31464


namespace NUMINAMATH_GPT_loss_percentage_is_11_percent_l314_31457

-- Definitions based on conditions
def costPrice : ℝ := 1500
def sellingPrice : ℝ := 1335

-- The statement to prove
theorem loss_percentage_is_11_percent :
  ((costPrice - sellingPrice) / costPrice) * 100 = 11 := by
  sorry

end NUMINAMATH_GPT_loss_percentage_is_11_percent_l314_31457


namespace NUMINAMATH_GPT_find_function_l314_31487

theorem find_function (f : ℝ → ℝ) :
  (∀ x y z : ℝ, x + y + z = 0 → f (x^3) + f (y)^3 + f (z)^3 = 3 * x * y * z) → 
  f = id :=
by sorry

end NUMINAMATH_GPT_find_function_l314_31487


namespace NUMINAMATH_GPT_Anne_carrying_four_cats_weight_l314_31440

theorem Anne_carrying_four_cats_weight : 
  let w1 := 2
  let w2 := 1.5 * w1
  let m1 := 2 * w1
  let m2 := w1 + w2
  w1 + w2 + m1 + m2 = 14 :=
by
  sorry

end NUMINAMATH_GPT_Anne_carrying_four_cats_weight_l314_31440


namespace NUMINAMATH_GPT_quadratic_inequality_real_roots_l314_31450

theorem quadratic_inequality_real_roots (c : ℝ) (h_pos : 0 < c) (h_ineq : c < 25) :
  ∃ x : ℝ, x^2 - 10 * x + c < 0 :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_real_roots_l314_31450


namespace NUMINAMATH_GPT_stubborn_robot_returns_to_start_l314_31476

inductive Direction
| East | North | West | South

inductive Command
| STEP | LEFT

structure Robot :=
  (position : ℤ × ℤ)
  (direction : Direction)

def turnLeft : Direction → Direction
| Direction.East  => Direction.North
| Direction.North => Direction.West
| Direction.West  => Direction.South
| Direction.South => Direction.East

def moveStep : Robot → Robot
| ⟨(x, y), Direction.East⟩  => ⟨(x + 1, y), Direction.East⟩
| ⟨(x, y), Direction.North⟩ => ⟨(x, y + 1), Direction.North⟩
| ⟨(x, y), Direction.West⟩  => ⟨(x - 1, y), Direction.West⟩
| ⟨(x, y), Direction.South⟩ => ⟨(x, y - 1), Direction.South⟩

def executeCommand : Command → Robot → Robot
| Command.STEP, robot => moveStep robot
| Command.LEFT, robot => ⟨robot.position, turnLeft robot.direction⟩

def invertCommand : Command → Command
| Command.STEP => Command.LEFT
| Command.LEFT => Command.STEP

def executeSequence (seq : List Command) (robot : Robot) : Robot :=
  seq.foldl (λ r cmd => executeCommand cmd r) robot

def executeInvertedSequence (seq : List Command) (robot : Robot) : Robot :=
  seq.foldl (λ r cmd => executeCommand (invertCommand cmd) r) robot

def initialRobot : Robot := ⟨(0, 0), Direction.East⟩

def exampleProgram : List Command :=
  [Command.LEFT, Command.LEFT, Command.LEFT, Command.LEFT, Command.STEP, Command.STEP,
   Command.LEFT, Command.LEFT]

theorem stubborn_robot_returns_to_start :
  let robot := executeSequence exampleProgram initialRobot
  executeInvertedSequence exampleProgram robot = initialRobot :=
by
  sorry

end NUMINAMATH_GPT_stubborn_robot_returns_to_start_l314_31476


namespace NUMINAMATH_GPT_denominator_expression_l314_31498

theorem denominator_expression (x y a b E : ℝ)
  (h1 : x / y = 3)
  (h2 : (2 * a - x) / E = 3)
  (h3 : a / b = 4.5) : E = 3 * b - y :=
sorry

end NUMINAMATH_GPT_denominator_expression_l314_31498


namespace NUMINAMATH_GPT_yoongi_division_l314_31446

theorem yoongi_division (x : ℕ) (h : 5 * x = 100) : x / 10 = 2 := by
  sorry

end NUMINAMATH_GPT_yoongi_division_l314_31446


namespace NUMINAMATH_GPT_leon_required_score_l314_31411

noncomputable def leon_scores : List ℕ := [72, 68, 75, 81, 79]

theorem leon_required_score (n : ℕ) :
  (List.sum leon_scores + n) / (List.length leon_scores + 1) ≥ 80 ↔ n ≥ 105 :=
by sorry

end NUMINAMATH_GPT_leon_required_score_l314_31411


namespace NUMINAMATH_GPT_cannot_all_be_zero_l314_31443

theorem cannot_all_be_zero :
  ¬ ∃ (f : ℕ → ℕ), (∀ i, f i ∈ { x : ℕ | 1 ≤ x ∧ x ≤ 1989 }) ∧
                   (∀ i j, f (i + j) = f i - f j) ∧
                   (∃ n, ∀ i, f (i + n) = 0) :=
by
  sorry

end NUMINAMATH_GPT_cannot_all_be_zero_l314_31443


namespace NUMINAMATH_GPT_free_throw_percentage_l314_31432

theorem free_throw_percentage (p : ℚ) :
  (1 - p)^2 + 2 * p * (1 - p) = 16 / 25 → p = 3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_free_throw_percentage_l314_31432


namespace NUMINAMATH_GPT_total_sum_lent_l314_31473

theorem total_sum_lent (x : ℝ) (second_part : ℝ) (total_sum : ℝ)
  (h1 : second_part = 1648)
  (h2 : (x * 3 / 100 * 8) = (second_part * 5 / 100 * 3))
  (h3 : total_sum = x + second_part) :
  total_sum = 2678 := 
  sorry

end NUMINAMATH_GPT_total_sum_lent_l314_31473


namespace NUMINAMATH_GPT_identify_a_b_l314_31493

theorem identify_a_b (a b : ℝ) (h : ∀ x y : ℝ, (⌊a * x + b * y⌋ + ⌊b * x + a * y⌋ = (a + b) * ⌊x + y⌋)) : 
  (a = 0 ∧ b = 0) ∨ (a = 1 ∧ b = 1) :=
sorry

end NUMINAMATH_GPT_identify_a_b_l314_31493


namespace NUMINAMATH_GPT_smaller_of_two_numbers_in_ratio_l314_31442

theorem smaller_of_two_numbers_in_ratio (x y a b c : ℝ) (h0 : 0 < a) (h1 : a < b) (h2 : x / y = a / b) (h3 : x + y = c) : 
  min x y = (a * c) / (a + b) :=
by
  sorry

end NUMINAMATH_GPT_smaller_of_two_numbers_in_ratio_l314_31442


namespace NUMINAMATH_GPT_nec_but_not_suff_condition_l314_31428

variables {p q : Prop}

theorem nec_but_not_suff_condition (hp : ¬p) : 
  (p ∨ q → False) ↔ (¬p) ∧ ¬(¬p → p ∨ q) :=
by {
  sorry
}

end NUMINAMATH_GPT_nec_but_not_suff_condition_l314_31428


namespace NUMINAMATH_GPT_probability_B_winning_l314_31412

def P_A : ℝ := 0.2
def P_D : ℝ := 0.5
def P_B : ℝ := 1 - (P_A + P_D)

theorem probability_B_winning : P_B = 0.3 :=
by
  -- Proof steps go here
  sorry

end NUMINAMATH_GPT_probability_B_winning_l314_31412


namespace NUMINAMATH_GPT_triangle_shape_and_maximum_tan_B_minus_C_l314_31434

open Real

variable (A B C : ℝ)
variable (sin cos tan : ℝ → ℝ)

-- Given conditions
axiom sin2A_plus_3sin2C_equals_3sin2B : sin A ^ 2 + 3 * sin C ^ 2 = 3 * sin B ^ 2
axiom sinB_cosC_equals_2div3 : sin B * cos C = 2 / 3

-- Prove
theorem triangle_shape_and_maximum_tan_B_minus_C :
  (A = π / 2) ∧ (∀ x y : ℝ, (x = B - C) → tan x ≤ sqrt 2 / 4) :=
by sorry

end NUMINAMATH_GPT_triangle_shape_and_maximum_tan_B_minus_C_l314_31434


namespace NUMINAMATH_GPT_seat_notation_l314_31441

theorem seat_notation (row1 col1 row2 col2 : ℕ) (h : (row1, col1) = (5, 2)) : (row2, col2) = (7, 3) :=
 by
  sorry

end NUMINAMATH_GPT_seat_notation_l314_31441


namespace NUMINAMATH_GPT_poodle_barked_24_times_l314_31468

-- Defining the conditions and question in Lean
def poodle_barks (terrier_barks_per_hush times_hushed: ℕ) : ℕ :=
  2 * terrier_barks_per_hush * times_hushed

theorem poodle_barked_24_times (terrier_barks_per_hush times_hushed: ℕ) :
  terrier_barks_per_hush = 2 → times_hushed = 6 → poodle_barks terrier_barks_per_hush times_hushed = 24 :=
by
  intros
  sorry

end NUMINAMATH_GPT_poodle_barked_24_times_l314_31468


namespace NUMINAMATH_GPT_fish_left_in_tank_l314_31461

-- Define the initial number of fish and the number of fish moved
def initialFish : Real := 212.0
def movedFish : Real := 68.0

-- Define the number of fish left in the tank
def fishLeft (initialFish : Real) (movedFish : Real) : Real := initialFish - movedFish

-- Theorem stating the problem
theorem fish_left_in_tank : fishLeft initialFish movedFish = 144.0 := by
  sorry

end NUMINAMATH_GPT_fish_left_in_tank_l314_31461


namespace NUMINAMATH_GPT_slope_of_line_l314_31419

theorem slope_of_line : 
  ∀ (x1 y1 x2 y2 : ℝ), 
  x1 = 1 → y1 = 3 → x2 = 6 → y2 = -7 → 
  (x1 ≠ x2) → ((y2 - y1) / (x2 - x1) = -2) :=
by
  intros x1 y1 x2 y2 hx1 hy1 hx2 hy2 hx1_ne_x2
  rw [hx1, hy1, hx2, hy2]
  sorry

end NUMINAMATH_GPT_slope_of_line_l314_31419


namespace NUMINAMATH_GPT_sequence_general_term_l314_31404

noncomputable def a₁ : ℕ → ℚ := sorry

variable (S : ℕ → ℚ)

axiom h₀ : a₁ 1 = -1
axiom h₁ : ∀ n : ℕ, a₁ (n + 1) = S n * S (n + 1)

theorem sequence_general_term (n : ℕ) : S n = -1 / n := by
  sorry

end NUMINAMATH_GPT_sequence_general_term_l314_31404


namespace NUMINAMATH_GPT_geometric_sequence_sum_S6_l314_31408

theorem geometric_sequence_sum_S6 (S : ℕ → ℝ) (S_2_eq_4 : S 2 = 4) (S_4_eq_16 : S 4 = 16) :
  S 6 = 52 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_sum_S6_l314_31408


namespace NUMINAMATH_GPT_prime_k_values_l314_31448

theorem prime_k_values (n : ℕ) (h1 : n ≥ 2) 
  (h2 : ∀ k : ℕ, 0 ≤ k ∧ k ≤ Nat.sqrt (n / 3) → Nat.Prime (k^2 + k + n)) :
  ∀ k : ℕ, 0 ≤ k ∧ k ≤ n - 2 → Nat.Prime (k^2 + k + n) :=
by
  sorry

end NUMINAMATH_GPT_prime_k_values_l314_31448


namespace NUMINAMATH_GPT_identical_machine_production_l314_31494

-- Definitions based on given conditions
def machine_production_rate (machines : ℕ) (rate : ℕ) :=
  rate / machines

def bottles_in_minute (machines : ℕ) (rate_per_machine : ℕ) :=
  machines * rate_per_machine

def total_bottles (bottle_rate_per_minute : ℕ) (minutes : ℕ) :=
  bottle_rate_per_minute * minutes

-- Theorem to prove based on the question == answer given conditions
theorem identical_machine_production :
  ∀ (machines_initial machines_final : ℕ) (bottles_per_minute : ℕ) (minutes : ℕ),
    machines_initial = 6 →
    machines_final = 12 →
    bottles_per_minute = 270 →
    minutes = 4 →
    total_bottles (bottles_in_minute machines_final (machine_production_rate machines_initial bottles_per_minute)) minutes = 2160 := by
  intros
  sorry

end NUMINAMATH_GPT_identical_machine_production_l314_31494


namespace NUMINAMATH_GPT_length_down_correct_l314_31470

variable (rate_up rate_down time_up time_down length_down : ℕ)
variable (h1 : rate_up = 8)
variable (h2 : time_up = 2)
variable (h3 : time_down = time_up)
variable (h4 : rate_down = (3 / 2) * rate_up)
variable (h5 : length_down = rate_down * time_down)

theorem length_down_correct : length_down = 24 := by
  sorry

end NUMINAMATH_GPT_length_down_correct_l314_31470


namespace NUMINAMATH_GPT_possible_values_of_cubic_sum_l314_31492

theorem possible_values_of_cubic_sum (x y z : ℂ) (h1 : (Matrix.of ![
    ![x, y, z],
    ![y, z, x],
    ![z, x, y]
  ] ^ 2 = 3 • (1 : Matrix (Fin 3) (Fin 3) ℂ))) (h2 : x * y * z = -1) :
  x^3 + y^3 + z^3 = -3 + 3 * Real.sqrt 3 ∨ x^3 + y^3 + z^3 = -3 - 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_possible_values_of_cubic_sum_l314_31492


namespace NUMINAMATH_GPT_unique_element_set_l314_31424

theorem unique_element_set (a : ℝ) : 
  (∃! x, (a - 1) * x^2 + 3 * x - 2 = 0) ↔ (a = 1 ∨ a = -1 / 8) :=
by sorry

end NUMINAMATH_GPT_unique_element_set_l314_31424


namespace NUMINAMATH_GPT_max_value_of_x_l314_31449

theorem max_value_of_x : ∃ x : ℝ, 
  ( (4*x - 16) / (3*x - 4) )^2 + ( (4*x - 16) / (3*x - 4) ) = 18 
  ∧ x = (3 * Real.sqrt 73 + 28) / (11 - Real.sqrt 73) :=
sorry

end NUMINAMATH_GPT_max_value_of_x_l314_31449


namespace NUMINAMATH_GPT_colored_line_midpoint_l314_31437

theorem colored_line_midpoint (L : ℝ → Prop) (p1 p2 : ℝ) :
  (L p1 → L p2) →
  (∃ A B C : ℝ, L A = L B ∧ L B = L C ∧ 2 * B = A + C ∧ L A = L C) :=
sorry

end NUMINAMATH_GPT_colored_line_midpoint_l314_31437


namespace NUMINAMATH_GPT_sum_of_areas_l314_31454

theorem sum_of_areas (radii : ℕ → ℝ) (areas : ℕ → ℝ) (h₁ : radii 0 = 2) 
  (h₂ : ∀ n, radii (n + 1) = radii n / 3) 
  (h₃ : ∀ n, areas n = π * (radii n) ^ 2) : 
  ∑' n, areas n = (9 * π) / 2 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_areas_l314_31454


namespace NUMINAMATH_GPT_Dhoni_spending_difference_l314_31427

-- Definitions
def RentPercent := 20
def LeftOverPercent := 61
def TotalSpendPercent := 100 - LeftOverPercent
def DishwasherPercent := TotalSpendPercent - RentPercent

-- Theorem statement
theorem Dhoni_spending_difference :
  DishwasherPercent = RentPercent - 1 := 
by
  sorry

end NUMINAMATH_GPT_Dhoni_spending_difference_l314_31427


namespace NUMINAMATH_GPT_f_2013_eq_2_l314_31452

noncomputable def f : ℝ → ℝ := sorry

axiom h1 : ∀ x : ℝ, f (-x) = -f x
axiom h2 : ∀ x : ℝ, f (x + 4) = f x + f 2
axiom h3 : f (-1) = -2

theorem f_2013_eq_2 : f 2013 = 2 := 
by 
  sorry

end NUMINAMATH_GPT_f_2013_eq_2_l314_31452


namespace NUMINAMATH_GPT_value_of_f_g3_l314_31479

def g (x : ℝ) : ℝ := 4 * x - 5
def f (x : ℝ) : ℝ := 6 * x + 11

theorem value_of_f_g3 : f (g 3) = 53 := by
  sorry

end NUMINAMATH_GPT_value_of_f_g3_l314_31479


namespace NUMINAMATH_GPT_problem_statement_l314_31409

variable (p q : ℝ)

def condition := p ^ 2 / q ^ 3 = 4 / 5

theorem problem_statement (hpq : condition p q) : 11 / 7 + (2 * q ^ 3 - p ^ 2) / (2 * q ^ 3 + p ^ 2) = 2 :=
sorry

end NUMINAMATH_GPT_problem_statement_l314_31409


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l314_31495

theorem quadratic_inequality_solution (x : ℝ) : (x^2 - x - 12 < 0) ↔ (-3 < x ∧ x < 4) :=
by sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l314_31495


namespace NUMINAMATH_GPT_calculate_a_minus_b_l314_31431

theorem calculate_a_minus_b : 
  ∀ (a b : ℚ), (y = a * x + b) 
  ∧ (y = 4 ↔ x = 3) 
  ∧ (y = 22 ↔ x = 10) 
  → (a - b = 6 + 2 / 7)
:= sorry

end NUMINAMATH_GPT_calculate_a_minus_b_l314_31431


namespace NUMINAMATH_GPT_calories_per_cookie_l314_31439

theorem calories_per_cookie :
  ∀ (cookies_per_bag bags_per_box total_calories total_number_cookies : ℕ),
  cookies_per_bag = 20 →
  bags_per_box = 4 →
  total_calories = 1600 →
  total_number_cookies = cookies_per_bag * bags_per_box →
  (total_calories / total_number_cookies) = 20 :=
by sorry

end NUMINAMATH_GPT_calories_per_cookie_l314_31439


namespace NUMINAMATH_GPT_no_solution_for_inequalities_l314_31478

theorem no_solution_for_inequalities :
  ¬ ∃ x : ℝ, 3 * x - 2 < (x + 2) ^ 2 ∧ (x + 2) ^ 2 < 8 * x - 5 := by 
  sorry

end NUMINAMATH_GPT_no_solution_for_inequalities_l314_31478


namespace NUMINAMATH_GPT_ocean_depth_l314_31481

theorem ocean_depth (t : ℕ) (v : ℕ) (h : ℕ)
  (h_t : t = 8)
  (h_v : v = 1500) :
  h = 6000 :=
by
  sorry

end NUMINAMATH_GPT_ocean_depth_l314_31481


namespace NUMINAMATH_GPT_intersection_line_constant_l314_31435

-- Definitions based on conditions provided:
def circle1_eq (x y : ℝ) : Prop := (x + 6)^2 + (y - 2)^2 = 144
def circle2_eq (x y : ℝ) : Prop := (x - 3)^2 + (y - 9)^2 = 65

-- The theorem statement
theorem intersection_line_constant (c : ℝ) : 
  (∃ x y : ℝ, circle1_eq x y ∧ circle2_eq x y ∧ x + y = c) ↔ c = 6 :=
by
  sorry

end NUMINAMATH_GPT_intersection_line_constant_l314_31435


namespace NUMINAMATH_GPT_quadratic_roots_range_l314_31420

theorem quadratic_roots_range (k : ℝ) : 
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  (k * x₁^2 - 4 * x₁ + 1 = 0) ∧ (k * x₂^2 - 4 * x₂ + 1 = 0)) 
  ↔ (k < 4 ∧ k ≠ 0) := 
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_range_l314_31420


namespace NUMINAMATH_GPT_coloring_time_saved_percentage_l314_31405

variable (n : ℕ := 10) -- number of pictures
variable (draw_time : ℝ := 2) -- time to draw each picture in hours
variable (total_time : ℝ := 34) -- total time spent on drawing and coloring in hours

/-- 
  Prove the percentage of time saved on coloring each picture compared to drawing 
  given the specified conditions.
-/
theorem coloring_time_saved_percentage (n : ℕ) (draw_time total_time : ℝ) 
  (h1 : draw_time > 0)
  (draw_total_time : draw_time * n = 20)
  (total_picture_time : draw_time * n + coloring_total_time = total_time) :
  (draw_time - (coloring_total_time / n)) / draw_time * 100 = 30 := 
by
  sorry

end NUMINAMATH_GPT_coloring_time_saved_percentage_l314_31405


namespace NUMINAMATH_GPT_negation_of_p_l314_31436

variable (x : ℝ)

def p : Prop := ∀ x : ℝ, x^2 - x + 1 > 0

theorem negation_of_p : (¬p) ↔ ∃ x : ℝ, x^2 - x + 1 ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_p_l314_31436


namespace NUMINAMATH_GPT_evaluate_heartsuit_l314_31459

-- Define the given operation
def heartsuit (x y : ℝ) : ℝ := abs (x - y)

-- State the proof problem in Lean
theorem evaluate_heartsuit (a b : ℝ) (h_a : a = 3) (h_b : b = -1) :
  heartsuit (heartsuit a b) (heartsuit (2 * a) (2 * b)) = 4 :=
by
  -- acknowledging that it's correct without providing the solution steps
  sorry

end NUMINAMATH_GPT_evaluate_heartsuit_l314_31459


namespace NUMINAMATH_GPT_parallel_line_slope_l314_31474

theorem parallel_line_slope (a b c : ℝ) (m : ℝ) :
  (5 * a + 10 * b = -35) →
  (∃ m : ℝ, b = m * a + c) →
  m = -1/2 :=
by sorry

end NUMINAMATH_GPT_parallel_line_slope_l314_31474


namespace NUMINAMATH_GPT_mean_equality_l314_31401

theorem mean_equality (x : ℤ) (h : (8 + 10 + 24) / 3 = (16 + x + 18) / 3) : x = 8 := by 
sorry

end NUMINAMATH_GPT_mean_equality_l314_31401


namespace NUMINAMATH_GPT_exponentiation_problem_l314_31472

theorem exponentiation_problem :
  (-0.125 ^ 2003) * (-8 ^ 2004) = -8 := 
sorry

end NUMINAMATH_GPT_exponentiation_problem_l314_31472


namespace NUMINAMATH_GPT_three_divides_two_pow_n_plus_one_l314_31421

theorem three_divides_two_pow_n_plus_one (n : ℕ) (hn : n > 0) : 
  (3 ∣ 2^n + 1) ↔ Odd n := 
sorry

end NUMINAMATH_GPT_three_divides_two_pow_n_plus_one_l314_31421


namespace NUMINAMATH_GPT_minimum_value_ineq_l314_31491

noncomputable def problem_statement (a b c : ℝ) (h : a + b + c = 3) : Prop :=
  (a > 0) ∧ (b > 0) ∧ (c > 0) → (1 / (a + b) + 1 / (b + c) + 1 / (c + a) ≥ 3 / 2)

theorem minimum_value_ineq (a b c : ℝ) (h : a + b + c = 3) : problem_statement a b c h :=
  sorry

end NUMINAMATH_GPT_minimum_value_ineq_l314_31491


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l314_31430

theorem quadratic_inequality_solution (a b c : ℝ) (h1 : a < 0)
  (h2 : ∀ x : ℝ, -2 < x ∧ x < 3 ↔ ax^2 + bx + c > 0) :
  ∀ x : ℝ, -3 < x ∧ x < 2 ↔ ax^2 - bx + c > 0 := 
sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l314_31430


namespace NUMINAMATH_GPT_martin_bell_ringing_l314_31486

theorem martin_bell_ringing (B S : ℕ) (hB : B = 36) (hS : S = B / 3 + 4) : S + B = 52 :=
sorry

end NUMINAMATH_GPT_martin_bell_ringing_l314_31486


namespace NUMINAMATH_GPT_triangle_side_length_l314_31477

theorem triangle_side_length (AB AC BC BX CX : ℕ)
  (h1 : AB = 86)
  (h2 : AC = 97)
  (h3 : BX + CX = BC)
  (h4 : AX = AB)
  (h5 : AX = 86)
  (h6 : AB * AB * CX + AC * AC * BX = BC * (BX * CX + AX * AX))
  : BC = 61 := 
sorry

end NUMINAMATH_GPT_triangle_side_length_l314_31477


namespace NUMINAMATH_GPT_exists_int_solutions_for_equations_l314_31458

theorem exists_int_solutions_for_equations : 
  ∃ (x y : ℤ), x * y = 4747 ∧ x - y = -54 :=
by
  sorry

end NUMINAMATH_GPT_exists_int_solutions_for_equations_l314_31458


namespace NUMINAMATH_GPT_catch_up_time_l314_31453

-- Define the speeds of Person A and Person B.
def speed_A : ℝ := 10 -- kilometers per hour
def speed_B : ℝ := 7  -- kilometers per hour

-- Define the initial distance between Person A and Person B.
def initial_distance : ℝ := 15 -- kilometers

-- Prove the time it takes for person A to catch up with person B is 5 hours.
theorem catch_up_time :
  initial_distance / (speed_A - speed_B) = 5 :=
by
  -- Proof can be added here
  sorry

end NUMINAMATH_GPT_catch_up_time_l314_31453


namespace NUMINAMATH_GPT_taxable_income_l314_31425

theorem taxable_income (tax_paid : ℚ) (state_tax_rate : ℚ) (months_resident : ℚ) (total_months : ℚ) (T : ℚ) :
  tax_paid = 1275 ∧ state_tax_rate = 0.04 ∧ months_resident = 9 ∧ total_months = 12 → 
  T = 42500 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_taxable_income_l314_31425


namespace NUMINAMATH_GPT_sun_city_population_l314_31465

theorem sun_city_population (W R S : ℕ) (h1 : W = 2000)
    (h2 : R = 3 * W - 500) (h3 : S = 2 * R + 1000) : S = 12000 :=
by
    -- Use the provided conditions (h1, h2, h3) to state the theorem
    sorry

end NUMINAMATH_GPT_sun_city_population_l314_31465


namespace NUMINAMATH_GPT_inverse_variation_l314_31407

theorem inverse_variation (a b : ℝ) (k : ℝ) (h1 : a * b = k) (h2 : 800 * 0.5 = k) (h3 : a = 1600) : b = 0.25 :=
by 
  sorry

end NUMINAMATH_GPT_inverse_variation_l314_31407


namespace NUMINAMATH_GPT_find_dividend_l314_31499

def dividend (divisor quotient remainder : ℕ) : ℕ :=
  (divisor * quotient) + remainder

theorem find_dividend (divisor quotient remainder : ℕ) (h_divisor : divisor = 16) (h_quotient : quotient = 8) (h_remainder : remainder = 4) :
  dividend divisor quotient remainder = 132 :=
by
  sorry

end NUMINAMATH_GPT_find_dividend_l314_31499


namespace NUMINAMATH_GPT_sum_of_infinite_perimeters_l314_31496

noncomputable def geometric_series_sum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem sum_of_infinite_perimeters (a : ℝ) :
  let first_perimeter := 3 * a
  let common_ratio := (1/3 : ℝ)
  let S := geometric_series_sum first_perimeter common_ratio 0
  S = (9 * a / 2) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_infinite_perimeters_l314_31496
