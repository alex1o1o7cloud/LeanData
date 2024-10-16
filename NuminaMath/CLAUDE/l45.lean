import Mathlib

namespace NUMINAMATH_CALUDE_same_parity_smallest_largest_l45_4577

/-- A set with certain properties related to positioning in a function or polynomial -/
def A_P : Set ℤ := sorry

/-- The smallest element of A_P -/
def smallest (A : Set ℤ) : ℤ := sorry

/-- The largest element of A_P -/
def largest (A : Set ℤ) : ℤ := sorry

/-- A function to determine if a number is even -/
def isEven (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

theorem same_parity_smallest_largest : 
  isEven (smallest A_P) ↔ isEven (largest A_P) := by sorry

end NUMINAMATH_CALUDE_same_parity_smallest_largest_l45_4577


namespace NUMINAMATH_CALUDE_sister_age_when_john_is_50_l45_4531

/-- Calculates the sister's age when John reaches a given age -/
def sisterAge (johnsCurrentAge : ℕ) (johnsFutureAge : ℕ) : ℕ :=
  let sisterCurrentAge := 2 * johnsCurrentAge
  sisterCurrentAge + (johnsFutureAge - johnsCurrentAge)

/-- Theorem stating that when John is 50, his sister will be 60, given their current ages -/
theorem sister_age_when_john_is_50 :
  sisterAge 10 50 = 60 := by sorry

end NUMINAMATH_CALUDE_sister_age_when_john_is_50_l45_4531


namespace NUMINAMATH_CALUDE_closest_perfect_square_to_325_l45_4549

theorem closest_perfect_square_to_325 : 
  ∀ n : ℕ, n ≠ 18 → (n^2 : ℤ) - 325 ≥ (18^2 : ℤ) - 325 ∨ (n^2 : ℤ) - 325 ≤ (18^2 : ℤ) - 325 :=
sorry

end NUMINAMATH_CALUDE_closest_perfect_square_to_325_l45_4549


namespace NUMINAMATH_CALUDE_total_carriages_l45_4578

theorem total_carriages (euston norfolk norwich flying_scotsman : ℕ) : 
  euston = norfolk + 20 →
  norwich = 100 →
  flying_scotsman = norwich + 20 →
  euston = 130 →
  euston + norfolk + norwich + flying_scotsman = 460 := by
  sorry

end NUMINAMATH_CALUDE_total_carriages_l45_4578


namespace NUMINAMATH_CALUDE_boots_sold_l45_4586

theorem boots_sold (sneakers sandals total : ℕ) 
  (h1 : sneakers = 2)
  (h2 : sandals = 4)
  (h3 : total = 17) :
  total - (sneakers + sandals) = 11 := by
  sorry

end NUMINAMATH_CALUDE_boots_sold_l45_4586


namespace NUMINAMATH_CALUDE_repeating_decimal_subtraction_l45_4516

def repeating_decimal_246 : ℚ := 246 / 999
def repeating_decimal_135 : ℚ := 135 / 999
def repeating_decimal_579 : ℚ := 579 / 999

theorem repeating_decimal_subtraction :
  repeating_decimal_246 - repeating_decimal_135 - repeating_decimal_579 = -24 / 51 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_subtraction_l45_4516


namespace NUMINAMATH_CALUDE_min_white_fraction_4x4x4_cube_l45_4599

/-- Represents a cube composed of smaller unit cubes -/
structure CompositeCube where
  edge_length : ℕ
  red_cubes : ℕ
  white_cubes : ℕ

/-- The minimum fraction of white surface area for a given composite cube -/
def min_white_fraction (c : CompositeCube) : ℚ :=
  sorry

theorem min_white_fraction_4x4x4_cube :
  let c : CompositeCube := ⟨4, 50, 14⟩
  min_white_fraction c = 1 / 16 := by sorry

end NUMINAMATH_CALUDE_min_white_fraction_4x4x4_cube_l45_4599


namespace NUMINAMATH_CALUDE_max_intersections_circle_line_parabola_l45_4576

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- A parabola in a plane -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The maximum number of intersection points between a circle and a line -/
def max_intersections_circle_line : ℕ := 2

/-- The maximum number of intersection points between a parabola and a line -/
def max_intersections_parabola_line : ℕ := 2

/-- The maximum number of intersection points between a circle and a parabola -/
def max_intersections_circle_parabola : ℕ := 4

/-- Theorem: The maximum number of intersection points among a circle, a line, and a parabola on a plane is 8 -/
theorem max_intersections_circle_line_parabola :
  ∀ (c : Circle) (l : Line) (p : Parabola),
  max_intersections_circle_line +
  max_intersections_parabola_line +
  max_intersections_circle_parabola = 8 :=
by sorry

end NUMINAMATH_CALUDE_max_intersections_circle_line_parabola_l45_4576


namespace NUMINAMATH_CALUDE_baking_powder_difference_l45_4519

theorem baking_powder_difference (yesterday_amount today_amount : ℝ) 
  (h1 : yesterday_amount = 0.4)
  (h2 : today_amount = 0.3) : 
  yesterday_amount - today_amount = 0.1 := by
sorry

end NUMINAMATH_CALUDE_baking_powder_difference_l45_4519


namespace NUMINAMATH_CALUDE_tennis_players_count_l45_4543

/-- Represents a sports club with members playing badminton and tennis -/
structure SportsClub where
  total_members : ℕ
  badminton_players : ℕ
  neither_players : ℕ
  both_players : ℕ

/-- Calculates the number of tennis players in the sports club -/
def tennis_players (club : SportsClub) : ℕ :=
  club.total_members - club.neither_players - (club.badminton_players - club.both_players)

/-- Theorem stating the number of tennis players in the given sports club configuration -/
theorem tennis_players_count (club : SportsClub)
  (h1 : club.total_members = 30)
  (h2 : club.badminton_players = 17)
  (h3 : club.neither_players = 3)
  (h4 : club.both_players = 9) :
  tennis_players club = 19 := by
  sorry

#eval tennis_players { total_members := 30, badminton_players := 17, neither_players := 3, both_players := 9 }

end NUMINAMATH_CALUDE_tennis_players_count_l45_4543


namespace NUMINAMATH_CALUDE_min_gigabytes_plan_y_more_expensive_l45_4521

/-- Represents the cost of Plan Y in cents for a given number of gigabytes -/
def planYCost (gigabytes : ℕ) : ℕ := 3000 + 200 * gigabytes

/-- Represents the cost of Plan X in cents -/
def planXCost : ℕ := 5000

/-- Theorem stating that 11 gigabytes is the minimum at which Plan Y becomes more expensive than Plan X -/
theorem min_gigabytes_plan_y_more_expensive :
  ∀ g : ℕ, g ≥ 11 ↔ planYCost g > planXCost :=
by sorry

end NUMINAMATH_CALUDE_min_gigabytes_plan_y_more_expensive_l45_4521


namespace NUMINAMATH_CALUDE_hugo_win_given_six_l45_4575

/-- The number of players in the game -/
def num_players : ℕ := 5

/-- The number of sides on the die -/
def die_sides : ℕ := 6

/-- The probability of winning the game -/
def prob_win : ℚ := 1 / num_players

/-- The probability of rolling a 6 -/
def prob_roll_six : ℚ := 1 / die_sides

/-- The probability that no other player rolls a 6 -/
def prob_no_other_six : ℚ := (1 - 1 / die_sides) ^ (num_players - 1)

theorem hugo_win_given_six (
  hugo_first_roll : ℕ
) : 
  hugo_first_roll = die_sides →
  (prob_roll_six * prob_no_other_six) / prob_win = 3125 / 7776 := by
  sorry

end NUMINAMATH_CALUDE_hugo_win_given_six_l45_4575


namespace NUMINAMATH_CALUDE_pen_cost_l45_4522

/-- The cost of a pen in cents, given the following conditions:
  * Pencils cost 25 cents each
  * Susan spent 20 dollars in total
  * Susan bought a total of 36 pens and pencils
  * Susan bought 16 pencils
-/
theorem pen_cost (pencil_cost : ℕ) (total_spent : ℕ) (total_items : ℕ) (pencils_bought : ℕ) :
  pencil_cost = 25 →
  total_spent = 2000 →
  total_items = 36 →
  pencils_bought = 16 →
  ∃ (pen_cost : ℕ), pen_cost = 80 :=
by sorry

end NUMINAMATH_CALUDE_pen_cost_l45_4522


namespace NUMINAMATH_CALUDE_exists_complete_list_l45_4581

-- Define the tournament structure
structure Tournament where
  players : Type
  played : players → players → Prop
  winner : players → players → Prop
  no_draw : ∀ (a b : players), played a b → (winner a b ∨ winner b a)
  all_play : ∀ (a b : players), a ≠ b → played a b
  no_self_play : ∀ (a : players), ¬played a a

-- Define the list of beaten players for each player
def beaten_list (t : Tournament) (a : t.players) : Set t.players :=
  {b | t.winner a b ∨ ∃ c, t.winner a c ∧ t.winner c b}

-- Theorem statement
theorem exists_complete_list (t : Tournament) :
  ∃ a : t.players, ∀ b : t.players, b ≠ a → b ∈ beaten_list t a :=
sorry

end NUMINAMATH_CALUDE_exists_complete_list_l45_4581


namespace NUMINAMATH_CALUDE_not_right_triangle_l45_4507

theorem not_right_triangle (a b c : ℝ) (ha : a = Real.sqrt 3) (hb : b = Real.sqrt 4) (hc : c = Real.sqrt 5) :
  ¬(a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) :=
by sorry

end NUMINAMATH_CALUDE_not_right_triangle_l45_4507


namespace NUMINAMATH_CALUDE_alpha_plus_beta_equals_111_l45_4597

theorem alpha_plus_beta_equals_111 :
  ∀ α β : ℝ, (∀ x : ℝ, (x - α) / (x + β) = (x^2 - 72*x + 1343) / (x^2 + 63*x - 3360)) →
  α + β = 111 :=
by
  sorry

end NUMINAMATH_CALUDE_alpha_plus_beta_equals_111_l45_4597


namespace NUMINAMATH_CALUDE_quadratic_coefficients_l45_4538

/-- A quadratic function f(x) = ax^2 + bx + c satisfying specific conditions -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_positive : a > 0
  condition_1 : |a + b + c| = 3
  condition_2 : |4*a + 2*b + c| = 3
  condition_3 : |9*a + 3*b + c| = 3

/-- The theorem stating the possible coefficients of the quadratic function -/
theorem quadratic_coefficients (f : QuadraticFunction) :
  (f.a = 6 ∧ f.b = -24 ∧ f.c = 21) ∨
  (f.a = 3 ∧ f.b = -15 ∧ f.c = 15) ∨
  (f.a = 3 ∧ f.b = -9 ∧ f.c = 3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficients_l45_4538


namespace NUMINAMATH_CALUDE_line_intersection_y_axis_l45_4585

/-- The line passing through points (2, 9) and (4, 15) intersects the y-axis at (0, 3) -/
theorem line_intersection_y_axis :
  let p₁ : ℝ × ℝ := (2, 9)
  let p₂ : ℝ × ℝ := (4, 15)
  let m : ℝ := (p₂.2 - p₁.2) / (p₂.1 - p₁.1)
  let b : ℝ := p₁.2 - m * p₁.1
  let line (x : ℝ) : ℝ := m * x + b
  (0, line 0) = (0, 3) := by
  sorry

end NUMINAMATH_CALUDE_line_intersection_y_axis_l45_4585


namespace NUMINAMATH_CALUDE_place_two_after_two_digit_number_l45_4589

theorem place_two_after_two_digit_number (a b : ℕ) (h1 : a ≥ 1) (h2 : a ≤ 9) (h3 : b ≤ 9) : 
  (10 * a + b) * 10 + 2 = 100 * a + 10 * b + 2 := by
  sorry

end NUMINAMATH_CALUDE_place_two_after_two_digit_number_l45_4589


namespace NUMINAMATH_CALUDE_min_cost_at_one_l45_4583

/-- Represents the transportation problem for mangoes between supermarkets and destinations -/
structure MangoTransportation where
  supermarket_A_stock : ℝ
  supermarket_B_stock : ℝ
  destination_X_demand : ℝ
  destination_Y_demand : ℝ
  cost_A_to_X : ℝ
  cost_A_to_Y : ℝ
  cost_B_to_X : ℝ
  cost_B_to_Y : ℝ

/-- Calculates the total transportation cost given the amount transported from A to X -/
def total_cost (mt : MangoTransportation) (x : ℝ) : ℝ :=
  mt.cost_A_to_X * x + 
  mt.cost_A_to_Y * (mt.supermarket_A_stock - x) + 
  mt.cost_B_to_X * (mt.destination_X_demand - x) + 
  mt.cost_B_to_Y * (x - 1)

/-- Theorem stating that the minimum transportation cost occurs when x = 1 -/
theorem min_cost_at_one (mt : MangoTransportation) 
  (h1 : mt.supermarket_A_stock = 15)
  (h2 : mt.supermarket_B_stock = 15)
  (h3 : mt.destination_X_demand = 16)
  (h4 : mt.destination_Y_demand = 14)
  (h5 : mt.cost_A_to_X = 50)
  (h6 : mt.cost_A_to_Y = 30)
  (h7 : mt.cost_B_to_X = 60)
  (h8 : mt.cost_B_to_Y = 45)
  (h9 : ∀ x, 1 ≤ x ∧ x ≤ 15 → total_cost mt 1 ≤ total_cost mt x) :
  ∃ (min_x : ℝ), min_x = 1 ∧ 
    ∀ x, 1 ≤ x ∧ x ≤ 15 → total_cost mt min_x ≤ total_cost mt x :=
  sorry

end NUMINAMATH_CALUDE_min_cost_at_one_l45_4583


namespace NUMINAMATH_CALUDE_max_digits_product_3digit_2digit_l45_4508

theorem max_digits_product_3digit_2digit :
  ∀ (a b : ℕ), 100 ≤ a ∧ a ≤ 999 ∧ 10 ≤ b ∧ b ≤ 99 →
  a * b < 100000 :=
sorry

end NUMINAMATH_CALUDE_max_digits_product_3digit_2digit_l45_4508


namespace NUMINAMATH_CALUDE_randys_initial_amount_l45_4501

/-- Proves that Randy's initial amount was $3000 given the described transactions --/
theorem randys_initial_amount (initial final smith_gave sally_received : ℕ) :
  final = initial + smith_gave - sally_received →
  smith_gave = 200 →
  sally_received = 1200 →
  final = 2000 →
  initial = 3000 := by
  sorry

end NUMINAMATH_CALUDE_randys_initial_amount_l45_4501


namespace NUMINAMATH_CALUDE_quadratic_form_sum_l45_4509

theorem quadratic_form_sum (x : ℝ) : ∃ (d e : ℝ), x^2 - 18*x + 81 = (x + d)^2 + e ∧ d + e = -9 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_sum_l45_4509


namespace NUMINAMATH_CALUDE_octal_subtraction_example_l45_4584

/-- Represents a number in base 8 as a list of digits (least significant first) --/
def OctalNumber := List Nat

/-- Subtraction operation for octal numbers --/
def octal_subtract (a b : OctalNumber) : OctalNumber :=
  sorry

/-- Conversion from a natural number to its octal representation --/
def to_octal (n : Nat) : OctalNumber :=
  sorry

theorem octal_subtraction_example :
  octal_subtract [4, 3, 5, 7] [7, 6, 2, 3] = [3, 4, 2, 4] :=
sorry

end NUMINAMATH_CALUDE_octal_subtraction_example_l45_4584


namespace NUMINAMATH_CALUDE_blue_shirt_percentage_l45_4542

theorem blue_shirt_percentage (total_students : ℕ) 
  (red_percentage green_percentage : ℚ) (other_count : ℕ) :
  total_students = 900 →
  red_percentage = 28 / 100 →
  green_percentage = 10 / 100 →
  other_count = 162 →
  (1 - (red_percentage + green_percentage + (other_count : ℚ) / total_students)) = 44 / 100 := by
sorry

end NUMINAMATH_CALUDE_blue_shirt_percentage_l45_4542


namespace NUMINAMATH_CALUDE_forester_tree_planting_l45_4588

theorem forester_tree_planting (initial_trees : ℕ) (monday_multiplier : ℕ) (tuesday_fraction : ℚ) : 
  initial_trees = 30 →
  monday_multiplier = 3 →
  tuesday_fraction = 1/3 →
  (monday_multiplier * initial_trees - initial_trees) + 
  (tuesday_fraction * (monday_multiplier * initial_trees - initial_trees)) = 80 := by
sorry

end NUMINAMATH_CALUDE_forester_tree_planting_l45_4588


namespace NUMINAMATH_CALUDE_circle_equation_from_conditions_l45_4557

/-- The equation of a circle given specific conditions -/
theorem circle_equation_from_conditions :
  ∀ (M : ℝ × ℝ),
  (2 * M.1 + M.2 - 1 = 0) →  -- M lies on the line 2x + y - 1 = 0
  (∃ (r : ℝ), r > 0 ∧
    ((M.1 - 3)^2 + M.2^2 = r^2) ∧  -- (3,0) is on the circle
    ((M.1 - 0)^2 + (M.2 - 1)^2 = r^2)) →  -- (0,1) is on the circle
  (∀ (x y : ℝ), (x - 1)^2 + (y + 1)^2 = 5 ↔
    ((x - M.1)^2 + (y - M.2)^2 = ((M.1 - 3)^2 + M.2^2))) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_from_conditions_l45_4557


namespace NUMINAMATH_CALUDE_probability_open_path_correct_l45_4511

/-- The probability of being able to go from the first floor to the last floor through only open doors in a building with n floors and randomly locked doors. -/
def probability_open_path (n : ℕ) : ℚ :=
  if n ≤ 1 then 1
  else (2 ^ (n - 1) : ℚ) / (Nat.choose (2 * (n - 1)) (n - 1) : ℚ)

/-- Theorem stating the probability of an open path in the building. -/
theorem probability_open_path_correct (n : ℕ) (h : n > 1) :
  probability_open_path n =
    (2 ^ (n - 1) : ℚ) / (Nat.choose (2 * (n - 1)) (n - 1) : ℚ) := by
  sorry

#eval probability_open_path 5

end NUMINAMATH_CALUDE_probability_open_path_correct_l45_4511


namespace NUMINAMATH_CALUDE_certain_number_equation_l45_4560

theorem certain_number_equation : ∃ x : ℤ, 9548 + 7314 = x + 13500 ∧ x = 3362 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_equation_l45_4560


namespace NUMINAMATH_CALUDE_units_digit_of_17_pow_2007_l45_4532

theorem units_digit_of_17_pow_2007 : ∃ n : ℕ, 17^2007 ≡ 3 [ZMOD 10] :=
sorry

end NUMINAMATH_CALUDE_units_digit_of_17_pow_2007_l45_4532


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l45_4541

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ) (d : ℝ)
  (h_arithmetic : arithmetic_sequence a d)
  (h_mean_1_2 : (a 1 + a 2) / 2 = 1)
  (h_mean_2_3 : (a 2 + a 3) / 2 = 2) :
  d = 1 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l45_4541


namespace NUMINAMATH_CALUDE_trig_fraction_simplification_l45_4534

theorem trig_fraction_simplification (α : ℝ) : 
  (Real.cos (π + α) * Real.sin (α + 2*π)) / (Real.sin (-α - π) * Real.cos (-π - α)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_fraction_simplification_l45_4534


namespace NUMINAMATH_CALUDE_root_minus_one_l45_4510

theorem root_minus_one (p : ℝ) (hp : p ≠ 1 ∧ p ≠ -1) : 
  (2 * (1 - p + p^2) / (1 - p^2)) * (-1)^2 + 
  ((2 - p) / (1 + p)) * (-1) - 
  (p / (1 - p)) = 0 := by
sorry

end NUMINAMATH_CALUDE_root_minus_one_l45_4510


namespace NUMINAMATH_CALUDE_symmetry_about_x_equals_one_symmetry_about_x_equals_three_halves_odd_function_shift_l45_4512

open Function

-- Define a function f from reals to reals
variable (f : ℝ → ℝ)

-- Statement ①
theorem symmetry_about_x_equals_one :
  (∀ x, f (x - 1) = f (1 - x)) ↔ 
  (∀ x, f (2 - x) = f x) :=
sorry

-- Statement ②
theorem symmetry_about_x_equals_three_halves 
  (h1 : ∀ x, f (-3/2 - x) = f x) 
  (h2 : ∀ x, f (x + 3/2) = -f x) :
  ∀ x, f (3 - x) = f x :=
sorry

-- Statement ③
theorem odd_function_shift
  (h : ∀ x, f (x + 2) = -f (-x + 4)) :
  ∀ x, f ((x + 3) + 3) = -f (-(x + 3) + 3) :=
sorry

end NUMINAMATH_CALUDE_symmetry_about_x_equals_one_symmetry_about_x_equals_three_halves_odd_function_shift_l45_4512


namespace NUMINAMATH_CALUDE_whispered_numbers_l45_4537

/-- Represents a digit sum calculation step -/
def DigitSumStep (n : ℕ) : ℕ := sorry

/-- The maximum possible digit sum for a 2022-digit number -/
def MaxInitialSum : ℕ := 2022 * 9

theorem whispered_numbers (initial_number : ℕ) 
  (h1 : initial_number ≤ MaxInitialSum) 
  (whisper1 : ℕ) 
  (h2 : whisper1 = DigitSumStep initial_number)
  (whisper2 : ℕ) 
  (h3 : whisper2 = DigitSumStep whisper1)
  (h4 : 10 ≤ whisper2 ∧ whisper2 ≤ 99)
  (h5 : DigitSumStep whisper2 = 1) :
  whisper1 = 19 ∨ whisper1 = 28 := by sorry

end NUMINAMATH_CALUDE_whispered_numbers_l45_4537


namespace NUMINAMATH_CALUDE_guessing_game_difference_l45_4528

theorem guessing_game_difference : (2 * 51) - (3 * 33) = 3 := by
  sorry

end NUMINAMATH_CALUDE_guessing_game_difference_l45_4528


namespace NUMINAMATH_CALUDE_quadratic_polynomial_condition_l45_4556

/-- 
Given a polynomial p(x) = 2a x^4 + 5a x^3 - 13 x^2 - x^4 + 2021 + 2x + bx^3 - bx^4 - 13x^3,
if p(x) is a quadratic polynomial, then a^2 + b^2 = 13.
-/
theorem quadratic_polynomial_condition (a b : ℝ) : 
  (∀ x : ℝ, (2*a - b - 1) * x^4 + (5*a + b - 13) * x^3 - 13 * x^2 + 2 * x + 2021 = 
             -13 * x^2 + 2 * x + 2021) → 
  a^2 + b^2 = 13 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_condition_l45_4556


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l45_4514

theorem hyperbola_asymptote_slope (m : ℝ) (α : ℝ) :
  (∀ x y : ℝ, x^2 + y^2/m = 1) →
  (0 < α ∧ α < π/3) →
  (-3 < m ∧ m < 0) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l45_4514


namespace NUMINAMATH_CALUDE_midpoint_distance_to_y_axis_l45_4502

noncomputable section

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus
def focus : ℝ × ℝ := (1, 0)

-- Define a point on the parabola
def point_on_parabola (p : ℝ × ℝ) : Prop :=
  parabola p.1 p.2

-- Define the distance between two points
def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem midpoint_distance_to_y_axis 
  (A B : ℝ × ℝ) 
  (h_A : point_on_parabola A) 
  (h_B : point_on_parabola B) 
  (h_dist : distance A focus + distance B focus = 12) :
  (A.1 + B.1) / 2 = 5 := by sorry

end

end NUMINAMATH_CALUDE_midpoint_distance_to_y_axis_l45_4502


namespace NUMINAMATH_CALUDE_functional_equation_identity_l45_4565

theorem functional_equation_identity (f : ℝ → ℝ) : 
  (∀ x y : ℝ, f (2 * f x + f y) = 2 * x + f y) → 
  (∀ x : ℝ, f x = x) := by
sorry

end NUMINAMATH_CALUDE_functional_equation_identity_l45_4565


namespace NUMINAMATH_CALUDE_age_difference_l45_4503

def hiram_age : ℕ := 40
def allyson_age : ℕ := 28

theorem age_difference : 
  (2 * allyson_age) - (hiram_age + 12) = 4 :=
by sorry

end NUMINAMATH_CALUDE_age_difference_l45_4503


namespace NUMINAMATH_CALUDE_probability_no_growth_pies_l45_4526

def total_pies : ℕ := 6
def growth_pies : ℕ := 2
def given_pies : ℕ := 3

theorem probability_no_growth_pies :
  let shrink_pies := total_pies - growth_pies
  let prob_mary_no_growth := (shrink_pies.choose given_pies : ℚ) / (total_pies.choose given_pies : ℚ)
  let prob_alice_no_growth := 1 - (1 - prob_mary_no_growth)
  prob_mary_no_growth + prob_alice_no_growth = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_probability_no_growth_pies_l45_4526


namespace NUMINAMATH_CALUDE_tinas_tangerines_l45_4523

/-- Represents the contents of Tina's bag -/
structure BagContents where
  apples : Nat
  oranges : Nat
  tangerines : Nat

/-- The condition after removing some fruits -/
def condition (b : BagContents) : Prop :=
  b.tangerines - 10 = (b.oranges - 2) + 4

/-- Theorem stating the number of tangerines in Tina's bag -/
theorem tinas_tangerines :
  ∃ (b : BagContents), b.apples = 9 ∧ b.oranges = 5 ∧ condition b ∧ b.tangerines = 17 :=
by sorry

end NUMINAMATH_CALUDE_tinas_tangerines_l45_4523


namespace NUMINAMATH_CALUDE_minimum_value_implies_a_l45_4554

noncomputable section

-- Define the functions
def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x + a) / Real.log (1/2)
def g (x : ℝ) : ℝ := x^2 + 4*x - 2
def h (a : ℝ) (x : ℝ) : ℝ := if f a x ≥ g x then f a x else g x

-- State the theorem
theorem minimum_value_implies_a (a : ℝ) : 
  (∀ x, h a x ≥ -2) ∧ (∃ x, h a x = -2) → a = 4 := by
  sorry

end

end NUMINAMATH_CALUDE_minimum_value_implies_a_l45_4554


namespace NUMINAMATH_CALUDE_bernie_chocolate_savings_l45_4551

/-- Calculates the savings over a given number of weeks when buying chocolates at a discounted price --/
def chocolate_savings (chocolates_per_week : ℕ) (regular_price discount_price : ℚ) (weeks : ℕ) : ℚ :=
  (chocolates_per_week * (regular_price - discount_price)) * weeks

/-- The savings over three weeks when buying two chocolates per week at a store with a $2 price instead of a store with a $3 price is equal to $6 --/
theorem bernie_chocolate_savings :
  chocolate_savings 2 3 2 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_bernie_chocolate_savings_l45_4551


namespace NUMINAMATH_CALUDE_triangle_area_is_correct_l45_4545

/-- The slope of the first line -/
def m₁ : ℚ := 1/3

/-- The slope of the second line -/
def m₂ : ℚ := 3

/-- The point of intersection of the first two lines -/
def A : ℚ × ℚ := (3, 3)

/-- The equation of the third line: x + y = 12 -/
def line3 (x y : ℚ) : Prop := x + y = 12

/-- The area of the triangle formed by the three lines -/
noncomputable def triangle_area : ℚ := sorry

theorem triangle_area_is_correct : triangle_area = 8625/1000 := by sorry

end NUMINAMATH_CALUDE_triangle_area_is_correct_l45_4545


namespace NUMINAMATH_CALUDE_intersection_A_B_empty_complement_A_union_B_a_geq_one_l45_4566

-- Define the sets A, B, U, and C
def A : Set ℝ := {x | x^2 + 6*x + 5 < 0}
def B : Set ℝ := {x | -1 ≤ x ∧ x < 1}
def U : Set ℝ := {x | |x| < 5}
def C (a : ℝ) : Set ℝ := {x | x < a}

-- Theorem 1: A ∩ B = ∅
theorem intersection_A_B_empty : A ∩ B = ∅ := by sorry

-- Theorem 2: ∁_U(A ∪ B) = {x | 1 ≤ x < 5}
theorem complement_A_union_B : 
  (A ∪ B)ᶜ ∩ U = {x : ℝ | 1 ≤ x ∧ x < 5} := by sorry

-- Theorem 3: B ∩ C = B implies a ≥ 1
theorem a_geq_one (a : ℝ) (h : B ∩ C a = B) : a ≥ 1 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_empty_complement_A_union_B_a_geq_one_l45_4566


namespace NUMINAMATH_CALUDE_so3_required_moles_l45_4558

/-- Represents a chemical species in a reaction -/
inductive Species
| SO3
| H2O
| H2SO4

/-- Represents the stoichiometric coefficient of a species in a reaction -/
def stoich_coeff (s : Species) : ℚ :=
  match s with
  | Species.SO3 => 1
  | Species.H2O => 1
  | Species.H2SO4 => 1

/-- The amount of H2O available in moles -/
def h2o_available : ℚ := 2

/-- The amount of H2SO4 to be formed in moles -/
def h2so4_formed : ℚ := 2

/-- Theorem: The number of moles of SO3 required is 2 -/
theorem so3_required_moles : 
  let so3_moles := h2so4_formed / stoich_coeff Species.H2SO4 * stoich_coeff Species.SO3
  so3_moles = 2 := by sorry

end NUMINAMATH_CALUDE_so3_required_moles_l45_4558


namespace NUMINAMATH_CALUDE_excellent_students_increase_l45_4546

theorem excellent_students_increase (total_students : ℕ) 
  (first_semester_percent : ℚ) (second_semester_percent : ℚ) :
  total_students = 650 →
  first_semester_percent = 70 / 100 →
  second_semester_percent = 80 / 100 →
  ⌈(second_semester_percent - first_semester_percent) * total_students⌉ = 65 := by
  sorry

end NUMINAMATH_CALUDE_excellent_students_increase_l45_4546


namespace NUMINAMATH_CALUDE_pyramid_width_height_difference_l45_4563

/-- The Great Pyramid of Giza's dimensions --/
structure PyramidDimensions where
  height : ℝ
  width : ℝ
  height_is_520 : height = 520
  width_greater_than_height : width > height
  sum_of_dimensions : height + width = 1274

/-- The difference between the width and height of the pyramid is 234 feet --/
theorem pyramid_width_height_difference (p : PyramidDimensions) : 
  p.width - p.height = 234 := by
sorry

end NUMINAMATH_CALUDE_pyramid_width_height_difference_l45_4563


namespace NUMINAMATH_CALUDE_pentagon_angle_sum_l45_4524

/-- Given a pentagon with two additional interior lines forming angles as described,
    prove that the sum of two specific angles is 138°. -/
theorem pentagon_angle_sum (P Q R x z : ℝ) : 
  P = 34 → Q = 76 → R = 28 → 
  (360 - x) + Q + P + 90 + (118 - z) = 540 →
  x + z = 138 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_angle_sum_l45_4524


namespace NUMINAMATH_CALUDE_geometric_sum_first_eight_terms_l45_4564

/-- The sum of the first n terms of a geometric sequence with first term a and common ratio r -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The first term of the geometric sequence -/
def a : ℚ := 1/3

/-- The common ratio of the geometric sequence -/
def r : ℚ := 1/3

/-- The number of terms to sum -/
def n : ℕ := 8

theorem geometric_sum_first_eight_terms :
  geometric_sum a r n = 3280/6561 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_first_eight_terms_l45_4564


namespace NUMINAMATH_CALUDE_upstream_downstream_time_ratio_l45_4596

/-- Proves that the ratio of upstream to downstream rowing time is 2:1 given specific speeds -/
theorem upstream_downstream_time_ratio 
  (man_speed : ℝ) 
  (current_speed : ℝ) 
  (h1 : man_speed = 4.5)
  (h2 : current_speed = 1.5) : 
  (man_speed - current_speed) / (man_speed + current_speed) = 1 / 2 := by
  sorry

#check upstream_downstream_time_ratio

end NUMINAMATH_CALUDE_upstream_downstream_time_ratio_l45_4596


namespace NUMINAMATH_CALUDE_amanda_borrowed_amount_l45_4536

/-- Calculates the earnings for a given number of hours based on the specified payment cycle -/
def calculateEarnings (hours : Nat) : Nat :=
  let cycleEarnings := [2, 4, 6, 8, 10, 12]
  let fullCycles := hours / 6
  let remainingHours := hours % 6
  fullCycles * (cycleEarnings.sum) + (cycleEarnings.take remainingHours).sum

/-- The amount Amanda borrowed is equal to her earnings from 45 hours of mowing -/
theorem amanda_borrowed_amount : calculateEarnings 45 = 306 := by
  sorry

#eval calculateEarnings 45

end NUMINAMATH_CALUDE_amanda_borrowed_amount_l45_4536


namespace NUMINAMATH_CALUDE_cyclic_quadrilateral_property_l45_4594

-- Define a structure for a cyclic quadrilateral
structure CyclicQuadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  h_a : ℝ
  h_b : ℝ
  h_c : ℝ
  h_d : ℝ
  a_positive : a > 0
  b_positive : b > 0
  c_positive : c > 0
  d_positive : d > 0
  h_a_positive : h_a > 0
  h_b_positive : h_b > 0
  h_c_positive : h_c > 0
  h_d_positive : h_d > 0
  is_cyclic : True  -- Placeholder for the cyclic property
  center_inside : True  -- Placeholder for the center being inside the quadrilateral

-- State the theorem
theorem cyclic_quadrilateral_property (q : CyclicQuadrilateral) :
  q.a * q.h_c + q.c * q.h_a = q.b * q.h_d + q.d * q.h_b := by
  sorry

end NUMINAMATH_CALUDE_cyclic_quadrilateral_property_l45_4594


namespace NUMINAMATH_CALUDE_island_width_is_five_l45_4529

/-- Represents a rectangular island -/
structure Island where
  length : ℝ
  width : ℝ
  area : ℝ

/-- The area of a rectangular island is equal to its length multiplied by its width -/
axiom island_area (i : Island) : i.area = i.length * i.width

/-- Given an island with area 50 square miles and length 10 miles, prove its width is 5 miles -/
theorem island_width_is_five (i : Island) 
  (h_area : i.area = 50) 
  (h_length : i.length = 10) : 
  i.width = 5 := by
sorry

end NUMINAMATH_CALUDE_island_width_is_five_l45_4529


namespace NUMINAMATH_CALUDE_percentage_of_200_to_50_percentage_proof_l45_4570

theorem percentage_of_200_to_50 : ℝ → Prop :=
  fun x => (200 / 50) * 100 = x ∧ x = 400

-- The proof would go here
theorem percentage_proof : percentage_of_200_to_50 400 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_200_to_50_percentage_proof_l45_4570


namespace NUMINAMATH_CALUDE_stating_max_principals_in_period_l45_4562

/-- Represents the duration of the entire period in years -/
def total_period : ℕ := 10

/-- Represents the duration of each principal's term in years -/
def term_length : ℕ := 4

/-- Represents the maximum number of principals that can serve during the total period -/
def max_principals : ℕ := 3

/-- 
Theorem stating that given a total period of 10 years and principals serving 
exactly one 4-year term each, the maximum number of principals that can serve 
during this period is 3.
-/
theorem max_principals_in_period : 
  ∀ (num_principals : ℕ), 
  (num_principals * term_length ≥ total_period) → 
  (num_principals ≤ max_principals) :=
by sorry

end NUMINAMATH_CALUDE_stating_max_principals_in_period_l45_4562


namespace NUMINAMATH_CALUDE_binary_1101101_equals_decimal_109_l45_4535

-- Define a function to convert binary to decimal
def binary_to_decimal (binary : List Bool) : Nat :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

-- Define the binary number 1101101
def binary_number : List Bool := [true, false, true, true, false, true, true]

-- Theorem statement
theorem binary_1101101_equals_decimal_109 :
  binary_to_decimal binary_number = 109 := by
  sorry

end NUMINAMATH_CALUDE_binary_1101101_equals_decimal_109_l45_4535


namespace NUMINAMATH_CALUDE_red_packs_count_l45_4525

/-- The number of packs of red bouncy balls Maggie bought -/
def red_packs : ℕ := sorry

/-- The number of packs of yellow bouncy balls Maggie bought -/
def yellow_packs : ℕ := 8

/-- The number of packs of green bouncy balls Maggie bought -/
def green_packs : ℕ := 4

/-- The number of bouncy balls in each package -/
def balls_per_pack : ℕ := 10

/-- The total number of bouncy balls Maggie bought -/
def total_balls : ℕ := 160

theorem red_packs_count : red_packs = 4 := by
  sorry

end NUMINAMATH_CALUDE_red_packs_count_l45_4525


namespace NUMINAMATH_CALUDE_train_speed_problem_l45_4500

theorem train_speed_problem (distance : ℝ) (time : ℝ) (speed_A : ℝ) : 
  distance = 480 →
  time = 2.5 →
  speed_A = 102 →
  (distance / time) - speed_A = 90 := by
sorry

end NUMINAMATH_CALUDE_train_speed_problem_l45_4500


namespace NUMINAMATH_CALUDE_equation_solutions_l45_4595

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = (-3 + Real.sqrt 13) / 2 ∧ x₂ = (-3 - Real.sqrt 13) / 2 ∧
    x₁^2 + 3*x₁ - 1 = 0 ∧ x₂^2 + 3*x₂ - 1 = 0) ∧
  (∃ y₁ y₂ : ℝ, y₁ = 2 ∧ y₂ = 4 ∧
    (y₁ - 2)^2 = 2*(y₁ - 2) ∧ (y₂ - 2)^2 = 2*(y₂ - 2)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l45_4595


namespace NUMINAMATH_CALUDE_spinsters_cats_ratio_l45_4518

theorem spinsters_cats_ratio : 
  ∀ (spinsters cats : ℕ),
    spinsters = 12 →
    cats = spinsters + 42 →
    (spinsters : ℚ) / cats = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_spinsters_cats_ratio_l45_4518


namespace NUMINAMATH_CALUDE_basic_computer_price_theorem_l45_4593

/-- Represents the prices of computer components and setups -/
structure ComputerPrices where
  basic_total : ℝ  -- Total price of basic setup
  enhanced_total : ℝ  -- Total price of enhanced setup
  printer_ratio : ℝ  -- Ratio of printer price to enhanced total
  monitor_ratio : ℝ  -- Ratio of monitor price to enhanced total
  keyboard_ratio : ℝ  -- Ratio of keyboard price to enhanced total

/-- Calculates the price of the basic computer given the prices and ratios -/
def basic_computer_price (prices : ComputerPrices) : ℝ :=
  let enhanced_computer := prices.enhanced_total * (1 - prices.printer_ratio - prices.monitor_ratio - prices.keyboard_ratio)
  enhanced_computer - (prices.enhanced_total - prices.basic_total)

/-- Theorem stating that the basic computer price is approximately $975.83 -/
theorem basic_computer_price_theorem (prices : ComputerPrices) 
  (h1 : prices.basic_total = 2500)
  (h2 : prices.enhanced_total = prices.basic_total + 600)
  (h3 : prices.printer_ratio = 1/6)
  (h4 : prices.monitor_ratio = 1/5)
  (h5 : prices.keyboard_ratio = 1/8) :
  ∃ ε > 0, |basic_computer_price prices - 975.83| < ε :=
sorry

end NUMINAMATH_CALUDE_basic_computer_price_theorem_l45_4593


namespace NUMINAMATH_CALUDE_circle_equation_l45_4553

-- Define the circle C
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the tangent line
def tangentLine (x y : ℝ) : Prop := 3 * x + 4 * y - 14 = 0

-- Define the line on which the center lies
def centerLine (x y : ℝ) : Prop := x + y - 11 = 0

-- Define the point of tangency
def tangentPoint : ℝ × ℝ := (2, 2)

-- Theorem statement
theorem circle_equation (C : Circle) :
  (tangentLine tangentPoint.1 tangentPoint.2) ∧
  (centerLine C.center.1 C.center.2) →
  ∀ (x y : ℝ), (x - C.center.1)^2 + (y - C.center.2)^2 = C.radius^2 ↔
  (x - 5)^2 + (y - 6)^2 = 25 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l45_4553


namespace NUMINAMATH_CALUDE_negation_of_implication_l45_4567

theorem negation_of_implication (A B : Set α) :
  ¬(∀ a, a ∈ A → b ∈ B) ↔ ∃ a, a ∈ A ∧ b ∉ B :=
by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l45_4567


namespace NUMINAMATH_CALUDE_six_different_squares_cannot_form_rectangle_l45_4520

/-- A square with a given side length -/
structure Square where
  sideLength : ℝ
  positive : sideLength > 0

/-- A collection of squares -/
def SquareCollection := List Square

/-- Predicate to check if all squares in a collection have different sizes -/
def allDifferentSizes (squares : SquareCollection) : Prop :=
  ∀ i j, i ≠ j → (squares.get i).sideLength ≠ (squares.get j).sideLength

/-- Predicate to check if squares can form a rectangle -/
def canFormRectangle (squares : SquareCollection) : Prop :=
  ∃ (width height : ℝ), width > 0 ∧ height > 0 ∧
    (squares.map (λ s => s.sideLength ^ 2)).sum = width * height

theorem six_different_squares_cannot_form_rectangle :
  ∀ (squares : SquareCollection),
    squares.length = 6 →
    allDifferentSizes squares →
    ¬ canFormRectangle squares :=
by
  sorry

end NUMINAMATH_CALUDE_six_different_squares_cannot_form_rectangle_l45_4520


namespace NUMINAMATH_CALUDE_internal_angle_pentadecagon_is_156_l45_4592

/-- The measure of one internal angle of a regular pentadecagon -/
def internal_angle_pentadecagon : ℝ :=
  156

/-- The number of sides in a pentadecagon -/
def pentadecagon_sides : ℕ := 15

theorem internal_angle_pentadecagon_is_156 :
  internal_angle_pentadecagon = 156 :=
by
  sorry

#check internal_angle_pentadecagon_is_156

end NUMINAMATH_CALUDE_internal_angle_pentadecagon_is_156_l45_4592


namespace NUMINAMATH_CALUDE_street_length_calculation_l45_4587

/-- Proves that given a speed of 5.31 km/h and a time of 8 minutes, the distance traveled is 708 meters. -/
theorem street_length_calculation (speed : ℝ) (time : ℝ) : 
  speed = 5.31 → time = 8 → speed * time * (1000 / 60) = 708 :=
by sorry

end NUMINAMATH_CALUDE_street_length_calculation_l45_4587


namespace NUMINAMATH_CALUDE_polynomial_difference_divisibility_l45_4530

theorem polynomial_difference_divisibility
  (a b c d x y : ℤ)
  (h : x ≠ y) :
  ∃ k : ℤ, a * x^3 + b * x^2 + c * x + d - (a * y^3 + b * y^2 + c * y + d) = (x - y) * k :=
sorry

end NUMINAMATH_CALUDE_polynomial_difference_divisibility_l45_4530


namespace NUMINAMATH_CALUDE_ab_power_2022_l45_4559

theorem ab_power_2022 (a b : ℝ) (h : (a - 1/2)^2 + |b + 2| = 0) : (a * b)^2022 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ab_power_2022_l45_4559


namespace NUMINAMATH_CALUDE_rod_length_proof_l45_4574

/-- Given that a 12-meter long rod weighs 14 kg, prove that a rod weighing 7 kg is 6 meters long -/
theorem rod_length_proof (weight_per_meter : ℝ) (h1 : weight_per_meter = 14 / 12) : 
  7 / weight_per_meter = 6 := by
  sorry

end NUMINAMATH_CALUDE_rod_length_proof_l45_4574


namespace NUMINAMATH_CALUDE_reciprocal_problem_l45_4590

theorem reciprocal_problem (x : ℝ) (h : 5 * x = 2) : 100 * (1 / x) = 250 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_problem_l45_4590


namespace NUMINAMATH_CALUDE_work_completion_time_l45_4540

theorem work_completion_time (x_time y_worked x_remaining : ℕ) (h1 : x_time = 20) (h2 : y_worked = 9) (h3 : x_remaining = 8) :
  ∃ (y_time : ℕ), y_time = 15 ∧ 
  (y_worked : ℚ) / y_time + x_remaining / x_time = 1 :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l45_4540


namespace NUMINAMATH_CALUDE_competition_matches_l45_4517

theorem competition_matches (n : ℕ) (h : n = 6) : n * (n - 1) / 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_competition_matches_l45_4517


namespace NUMINAMATH_CALUDE_lily_pad_growth_rate_l45_4569

/-- Represents the coverage of the lake by lily pads -/
def LakeCoverage := ℝ

/-- The time it takes for the lily pads to cover the entire lake -/
def fullCoverageTime : ℕ := 50

/-- The time it takes for the lily pads to cover half the lake -/
def halfCoverageTime : ℕ := 49

/-- The growth rate of the lily pad patch -/
def growthRate : ℝ → Prop := λ r => 
  ∀ t : ℝ, (1 : ℝ) = (1/2 : ℝ) * (1 + r) ^ (t + 1) → t = (fullCoverageTime - halfCoverageTime : ℝ)

theorem lily_pad_growth_rate : 
  growthRate 1 := by sorry

end NUMINAMATH_CALUDE_lily_pad_growth_rate_l45_4569


namespace NUMINAMATH_CALUDE_cash_percentage_is_ten_percent_l45_4561

def total_amount : ℝ := 1000
def raw_materials_cost : ℝ := 500
def machinery_cost : ℝ := 400

theorem cash_percentage_is_ten_percent :
  (total_amount - (raw_materials_cost + machinery_cost)) / total_amount * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_cash_percentage_is_ten_percent_l45_4561


namespace NUMINAMATH_CALUDE_simplify_fraction_l45_4555

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l45_4555


namespace NUMINAMATH_CALUDE_and_or_relationship_l45_4527

theorem and_or_relationship (p q : Prop) : 
  ((p ∧ q) → (p ∨ q)) ∧ ¬((p ∨ q) → (p ∧ q)) := by
  sorry

end NUMINAMATH_CALUDE_and_or_relationship_l45_4527


namespace NUMINAMATH_CALUDE_function_transformation_l45_4591

/-- Given a function f such that f(1/x) = x/(1-x) for all x ≠ 0 and x ≠ 1,
    prove that f(x) = 1/(x-1) for all x ≠ 0 and x ≠ 1. -/
theorem function_transformation (f : ℝ → ℝ) :
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ 1 → f (1/x) = x / (1 - x)) →
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ 1 → f x = 1 / (x - 1)) :=
by sorry

end NUMINAMATH_CALUDE_function_transformation_l45_4591


namespace NUMINAMATH_CALUDE_quadratic_common_root_l45_4505

theorem quadratic_common_root (p1 p2 q1 q2 : ℂ) :
  (∃ x : ℂ, x^2 + p1*x + q1 = 0 ∧ x^2 + p2*x + q2 = 0) ↔
  (q2 - q1)^2 + (p1 - p2)*(p1*q2 - q1*p2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_common_root_l45_4505


namespace NUMINAMATH_CALUDE_gcd_lcm_product_90_150_l45_4579

theorem gcd_lcm_product_90_150 : 
  (Nat.gcd 90 150) * (Nat.lcm 90 150) = 13500 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_90_150_l45_4579


namespace NUMINAMATH_CALUDE_garden_length_theorem_l45_4580

/-- Represents a rectangular garden with given dimensions and area allocations. -/
structure Garden where
  length : ℝ
  width : ℝ
  tilled_ratio : ℝ
  trellised_ratio : ℝ
  raised_bed_area : ℝ

/-- Theorem stating the conditions and conclusion about the garden's length. -/
theorem garden_length_theorem (g : Garden) : 
  g.width = 120 ∧ 
  g.tilled_ratio = 1/2 ∧ 
  g.trellised_ratio = 1/3 ∧ 
  g.raised_bed_area = 8800 →
  g.length = 220 := by
  sorry

#check garden_length_theorem

end NUMINAMATH_CALUDE_garden_length_theorem_l45_4580


namespace NUMINAMATH_CALUDE_min_value_theorem_l45_4504

/-- Two linear functions with parallel graphs not parallel to coordinate axes -/
structure ParallelLinearFunctions where
  f : ℝ → ℝ
  g : ℝ → ℝ
  parallel : ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x + b ∧ g x = a * x + c

/-- The theorem statement -/
theorem min_value_theorem (funcs : ParallelLinearFunctions) 
  (h : ∃ (x : ℝ), ∀ (y : ℝ), (funcs.f y)^2 + 8 * funcs.g y ≥ (funcs.f x)^2 + 8 * funcs.g x)
  (min_value : (funcs.f x)^2 + 8 * funcs.g x = -29) :
  ∃ (z : ℝ), ∀ (w : ℝ), (funcs.g w)^2 + 8 * funcs.f w ≥ (funcs.g z)^2 + 8 * funcs.f z ∧ 
  (funcs.g z)^2 + 8 * funcs.f z = -3 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l45_4504


namespace NUMINAMATH_CALUDE_max_real_sum_l45_4572

/-- The zeroes of z^10 - 2^30 -/
def zeroes : Finset ℂ :=
  sorry

/-- A function that chooses either z or iz to maximize the real part -/
def w (z : ℂ) : ℂ :=
  sorry

/-- The sum of w(z) for all zeroes -/
def sum_w : ℂ :=
  sorry

/-- The maximum possible value of the real part of the sum -/
theorem max_real_sum :
  (sum_w.re : ℝ) = 16 * (1 + Real.cos (π / 5) + Real.cos (2 * π / 5) - Real.sin (3 * π / 5) - Real.sin (4 * π / 5)) :=
sorry

end NUMINAMATH_CALUDE_max_real_sum_l45_4572


namespace NUMINAMATH_CALUDE_equation_solution_l45_4571

/-- The solutions to the equation (8y^2 + 135y + 5) / (3y + 35) = 4y + 2 -/
theorem equation_solution : 
  let y₁ : ℂ := (-11 + Complex.I * Real.sqrt 919) / 8
  let y₂ : ℂ := (-11 - Complex.I * Real.sqrt 919) / 8
  ∀ y : ℂ, (8 * y^2 + 135 * y + 5) / (3 * y + 35) = 4 * y + 2 ↔ y = y₁ ∨ y = y₂ :=
by sorry

#check equation_solution

end NUMINAMATH_CALUDE_equation_solution_l45_4571


namespace NUMINAMATH_CALUDE_value_of_N_l45_4513

theorem value_of_N : 
  let N := (Real.sqrt (Real.sqrt 5 + 2) + Real.sqrt (Real.sqrt 5 - 2)) / 
           Real.sqrt (Real.sqrt 5 + 1) - 
           Real.sqrt (3 - 2 * Real.sqrt 2)
  N = 1 := by sorry

end NUMINAMATH_CALUDE_value_of_N_l45_4513


namespace NUMINAMATH_CALUDE_M_remainder_1000_l45_4582

/-- M is the greatest integer multiple of 9 with no two digits being the same -/
def M : ℕ :=
  sorry

/-- The remainder when M is divided by 1000 is 621 -/
theorem M_remainder_1000 : M % 1000 = 621 :=
  sorry

end NUMINAMATH_CALUDE_M_remainder_1000_l45_4582


namespace NUMINAMATH_CALUDE_x_value_when_y_is_4_l45_4515

-- Define the inverse square relationship between x and y
def inverse_square_relation (x y : ℝ) : Prop :=
  ∃ k : ℝ, x = k / (y ^ 2)

-- State the theorem
theorem x_value_when_y_is_4 :
  ∀ x₀ y₀ x₁ y₁ : ℝ,
  inverse_square_relation x₀ y₀ →
  inverse_square_relation x₁ y₁ →
  x₀ = 1 →
  y₀ = 3 →
  y₁ = 4 →
  x₁ = 0.5625 := by
sorry

end NUMINAMATH_CALUDE_x_value_when_y_is_4_l45_4515


namespace NUMINAMATH_CALUDE_complex_expression_evaluation_l45_4552

theorem complex_expression_evaluation : 
  (((3.2 - 1.7) / 0.003) / ((29 / 35 - 3 / 7) * 4 / 0.2) - 
   ((1 + 13 / 20 - 1.5) * 1.5) / ((2.44 + (1 + 14 / 25)) * (1 / 8))) / (62 + 1 / 20) + 
  (1.364 / 0.124) = 12 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_evaluation_l45_4552


namespace NUMINAMATH_CALUDE_saras_basketball_games_l45_4550

theorem saras_basketball_games (won_games lost_games : ℕ) 
  (h1 : won_games = 12) 
  (h2 : lost_games = 4) : 
  won_games + lost_games = 16 := by
  sorry

end NUMINAMATH_CALUDE_saras_basketball_games_l45_4550


namespace NUMINAMATH_CALUDE_hyperbola_vertex_distance_l45_4544

/-- The distance between vertices of a hyperbola with equation x^2/16 - y^2/9 = 1 is 8 -/
theorem hyperbola_vertex_distance :
  let h : ℝ × ℝ → ℝ := fun (x, y) ↦ x^2/16 - y^2/9 - 1
  ∃ v₁ v₂ : ℝ × ℝ, h v₁ = 0 ∧ h v₂ = 0 ∧ ‖v₁ - v₂‖ = 8 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_vertex_distance_l45_4544


namespace NUMINAMATH_CALUDE_sin_arccos_twelve_thirteenths_l45_4539

theorem sin_arccos_twelve_thirteenths : Real.sin (Real.arccos (12/13)) = 5/13 := by
  sorry

end NUMINAMATH_CALUDE_sin_arccos_twelve_thirteenths_l45_4539


namespace NUMINAMATH_CALUDE_circle_regions_l45_4598

/-- Number of regions created by n circles -/
def P (n : ℕ) : ℕ := 2 + n * (n - 1)

/-- The problem statement -/
theorem circle_regions : P 2011 ≡ 2112 [ZMOD 10000] := by
  sorry

end NUMINAMATH_CALUDE_circle_regions_l45_4598


namespace NUMINAMATH_CALUDE_tan_alpha_value_l45_4573

theorem tan_alpha_value (α : ℝ) (h : (Real.sin α - 2 * Real.cos α) / (3 * Real.sin α + 5 * Real.cos α) = 5) :
  Real.tan α = -27/14 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l45_4573


namespace NUMINAMATH_CALUDE_stamps_in_first_book_l45_4533

theorem stamps_in_first_book (x : ℕ) : 
  (4 * x + 6 * 15 = 130) → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_stamps_in_first_book_l45_4533


namespace NUMINAMATH_CALUDE_midpoint_segments_equal_l45_4506

/-- A structure representing a rectangle with a circle intersection --/
structure RectangleWithCircle where
  /-- The rectangle --/
  rectangle : Set (ℝ × ℝ)
  /-- The circle --/
  circle : Set (ℝ × ℝ)
  /-- The four right triangles formed by the intersection --/
  triangles : Fin 4 → Set (ℝ × ℝ)
  /-- The midpoints of the hypotenuses of the triangles --/
  midpoints : Fin 4 → ℝ × ℝ

/-- The theorem stating that A₀C₀ = B₀D₀ --/
theorem midpoint_segments_equal (rc : RectangleWithCircle) :
  dist (rc.midpoints 0) (rc.midpoints 2) = dist (rc.midpoints 1) (rc.midpoints 3) :=
sorry

end NUMINAMATH_CALUDE_midpoint_segments_equal_l45_4506


namespace NUMINAMATH_CALUDE_rectangle_perimeter_change_l45_4548

theorem rectangle_perimeter_change (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  2 * (1.3 * a + 0.8 * b) = 2 * (a + b) →
  2 * (0.8 * a + 1.3 * b) = 1.1 * (2 * (a + b)) := by
sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_change_l45_4548


namespace NUMINAMATH_CALUDE_astronomy_collections_l45_4568

/-- Represents the distinct letters in "ASTRONOMY" --/
inductive AstronomyLetter
| A
| O
| S
| T
| R
| N
| M
| Y

/-- The number of each letter in "ASTRONOMY" --/
def letter_count : AstronomyLetter → Nat
| AstronomyLetter.A => 1
| AstronomyLetter.O => 2
| AstronomyLetter.S => 1
| AstronomyLetter.T => 1
| AstronomyLetter.R => 1
| AstronomyLetter.N => 2
| AstronomyLetter.M => 1
| AstronomyLetter.Y => 1

/-- The set of vowels in "ASTRONOMY" --/
def vowels : Set AstronomyLetter := {AstronomyLetter.A, AstronomyLetter.O}

/-- The set of consonants in "ASTRONOMY" --/
def consonants : Set AstronomyLetter := {AstronomyLetter.S, AstronomyLetter.T, AstronomyLetter.R, AstronomyLetter.N, AstronomyLetter.M, AstronomyLetter.Y}

/-- The number of distinct ways to choose 3 vowels and 3 consonants from "ASTRONOMY" --/
def distinct_collections : Nat := 100

theorem astronomy_collections :
  distinct_collections = 100 := by sorry


end NUMINAMATH_CALUDE_astronomy_collections_l45_4568


namespace NUMINAMATH_CALUDE_area_bisecting_line_sum_l45_4547

/-- Triangle ABC with vertices A(0, 10), B(3, 0), C(9, 0) -/
structure Triangle :=
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)
  (C : ℝ × ℝ)

/-- A line represented by its slope and y-intercept -/
structure Line :=
  (slope : ℝ)
  (y_intercept : ℝ)

/-- Predicate to check if a line bisects the area of a triangle through a specific vertex -/
def bisects_area (t : Triangle) (l : Line) (vertex : ℝ × ℝ) : Prop :=
  sorry

/-- The triangle ABC with given vertices -/
def triangle_ABC : Triangle :=
  { A := (0, 10),
    B := (3, 0),
    C := (9, 0) }

theorem area_bisecting_line_sum :
  ∃ l : Line, bisects_area triangle_ABC l triangle_ABC.B ∧ l.slope + l.y_intercept = -20/3 := by
  sorry

end NUMINAMATH_CALUDE_area_bisecting_line_sum_l45_4547
