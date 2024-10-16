import Mathlib

namespace NUMINAMATH_CALUDE_marks_age_relation_l1643_164316

/-- Proves that Mark's age will be 2 years more than twice Aaron's age in 4 years -/
theorem marks_age_relation (mark_current_age aaron_current_age : ℕ) : 
  mark_current_age = 28 →
  mark_current_age - 3 = 3 * (aaron_current_age - 3) + 1 →
  (mark_current_age + 4) = 2 * (aaron_current_age + 4) + 2 := by
  sorry

end NUMINAMATH_CALUDE_marks_age_relation_l1643_164316


namespace NUMINAMATH_CALUDE_age_sum_is_37_l1643_164362

/-- Given the ages of A, B, and C, prove their sum is 37 -/
theorem age_sum_is_37 (a b c : ℕ) (h1 : a = b + 2) (h2 : b = 2 * c) (h3 : b = 14) :
  a + b + c = 37 := by
  sorry

end NUMINAMATH_CALUDE_age_sum_is_37_l1643_164362


namespace NUMINAMATH_CALUDE_imaginary_unit_sum_l1643_164376

theorem imaginary_unit_sum (i : ℂ) (hi : i * i = -1) : i + i^2 + i^3 + i^4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_unit_sum_l1643_164376


namespace NUMINAMATH_CALUDE_sin_90_degrees_l1643_164347

theorem sin_90_degrees : Real.sin (π / 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_90_degrees_l1643_164347


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l1643_164368

theorem fraction_to_decimal : (5 : ℚ) / 16 = 0.3125 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l1643_164368


namespace NUMINAMATH_CALUDE_altitude_segment_length_l1643_164340

/-- An acute triangle with two altitudes dividing the sides -/
structure AcuteTriangleWithAltitudes where
  /-- The triangle is acute -/
  is_acute : Bool
  /-- Lengths of segments created by altitudes -/
  segment1 : ℝ
  segment2 : ℝ
  segment3 : ℝ
  segment4 : ℝ
  /-- Conditions on segment lengths -/
  h1 : segment1 = 6
  h2 : segment2 = 4
  h3 : segment3 = 3

/-- The theorem stating that the fourth segment length is 9/7 -/
theorem altitude_segment_length (t : AcuteTriangleWithAltitudes) : t.segment4 = 9/7 := by
  sorry

end NUMINAMATH_CALUDE_altitude_segment_length_l1643_164340


namespace NUMINAMATH_CALUDE_classroom_weight_distribution_exists_l1643_164379

theorem classroom_weight_distribution_exists :
  ∃ (n : ℕ) (b g : ℕ) (boy_weights girl_weights : List ℝ),
    n < 35 ∧
    n = b + g ∧
    b > 0 ∧
    g > 0 ∧
    boy_weights.length = b ∧
    girl_weights.length = g ∧
    (List.sum boy_weights + List.sum girl_weights) / n = 53.5 ∧
    List.sum boy_weights / b = 60 ∧
    List.sum girl_weights / g = 47 ∧
    List.minimum boy_weights < List.minimum girl_weights ∧
    List.maximum boy_weights < List.maximum girl_weights :=
by sorry

end NUMINAMATH_CALUDE_classroom_weight_distribution_exists_l1643_164379


namespace NUMINAMATH_CALUDE_complex_on_imaginary_axis_l1643_164338

theorem complex_on_imaginary_axis (a : ℝ) : 
  (Complex.I * ((2 * a + Complex.I) * (1 + Complex.I))).re = 0 → a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_complex_on_imaginary_axis_l1643_164338


namespace NUMINAMATH_CALUDE_election_winner_margin_l1643_164309

theorem election_winner_margin (total_votes : ℕ) (winner_votes : ℕ) (winner_percentage : ℚ) : 
  winner_percentage = 62 / 100 ∧ 
  winner_votes = 775 ∧ 
  winner_votes = (winner_percentage * total_votes).floor →
  winner_votes - (total_votes - winner_votes) = 300 := by
sorry

end NUMINAMATH_CALUDE_election_winner_margin_l1643_164309


namespace NUMINAMATH_CALUDE_board_cut_lengths_l1643_164377

/-- Given a board of 180 cm cut into three pieces, prove the lengths of the pieces. -/
theorem board_cut_lengths :
  ∀ (L M S : ℝ),
  L + M + S = 180 ∧
  L = M + S + 30 ∧
  M = L / 2 - 10 →
  L = 105 ∧ M = 42.5 ∧ S = 32.5 :=
by
  sorry

end NUMINAMATH_CALUDE_board_cut_lengths_l1643_164377


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_35_l1643_164358

theorem smallest_four_digit_divisible_by_35 :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 → n % 35 = 0 → n ≥ 1015 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_35_l1643_164358


namespace NUMINAMATH_CALUDE_friends_with_boxes_eq_two_l1643_164343

/-- The number of pencils in one color box -/
def pencils_per_box : ℕ := 7

/-- The total number of pencils Serenity and her friends have -/
def total_pencils : ℕ := 21

/-- The number of color boxes Serenity bought -/
def serenity_boxes : ℕ := 1

/-- The number of Serenity's friends who bought the color box -/
def friends_with_boxes : ℕ := (total_pencils / pencils_per_box) - serenity_boxes

theorem friends_with_boxes_eq_two : friends_with_boxes = 2 := by
  sorry

end NUMINAMATH_CALUDE_friends_with_boxes_eq_two_l1643_164343


namespace NUMINAMATH_CALUDE_binary_110101_to_base_7_l1643_164380

/-- Converts a binary number (represented as a list of bits) to its decimal representation -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Converts a natural number to its base-7 representation (as a list of digits) -/
def to_base_7 (n : ℕ) : List ℕ :=
  if n < 7 then [n]
  else (n % 7) :: to_base_7 (n / 7)

/-- The given binary number 110101₂ -/
def binary_110101 : List Bool := [true, false, true, false, true, true]

theorem binary_110101_to_base_7 :
  to_base_7 (binary_to_decimal binary_110101) = [4, 0, 1] :=
sorry

end NUMINAMATH_CALUDE_binary_110101_to_base_7_l1643_164380


namespace NUMINAMATH_CALUDE_ant_movement_l1643_164310

-- Define the type for a 2D position
def Position := ℝ × ℝ

-- Define the initial position
def initial_position : Position := (-2, 4)

-- Define the horizontal movement
def horizontal_movement : ℝ := 3

-- Define the vertical movement
def vertical_movement : ℝ := -2

-- Define the function to calculate the final position
def final_position (initial : Position) (horizontal : ℝ) (vertical : ℝ) : Position :=
  (initial.1 + horizontal, initial.2 + vertical)

-- Theorem statement
theorem ant_movement :
  final_position initial_position horizontal_movement vertical_movement = (1, 2) := by
  sorry

end NUMINAMATH_CALUDE_ant_movement_l1643_164310


namespace NUMINAMATH_CALUDE_volleyball_lineup_combinations_l1643_164391

theorem volleyball_lineup_combinations (total_players : ℕ) 
  (starting_lineup_size : ℕ) (required_players : ℕ) : 
  total_players = 15 → 
  starting_lineup_size = 7 → 
  required_players = 3 → 
  Nat.choose (total_players - required_players) (starting_lineup_size - required_players) = 495 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_lineup_combinations_l1643_164391


namespace NUMINAMATH_CALUDE_negation_statement_is_false_l1643_164308

theorem negation_statement_is_false : ¬(
  (∃ x : ℝ, x^2 + 1 > 3*x) ↔ (∀ x : ℝ, x^2 + 1 ≤ 3*x)
) := by sorry

end NUMINAMATH_CALUDE_negation_statement_is_false_l1643_164308


namespace NUMINAMATH_CALUDE_quadratic_square_completion_l1643_164371

theorem quadratic_square_completion (d e : ℤ) : 
  (∀ x, x^2 - 10*x + 13 = 0 ↔ (x + d)^2 = e) → d + e = 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_square_completion_l1643_164371


namespace NUMINAMATH_CALUDE_two_digit_product_sum_l1643_164395

theorem two_digit_product_sum : ∃ (a b : ℕ), 
  10 ≤ a ∧ a < 100 ∧ 
  10 ≤ b ∧ b < 100 ∧ 
  a * b = 3024 ∧ 
  a + b = 120 := by
sorry

end NUMINAMATH_CALUDE_two_digit_product_sum_l1643_164395


namespace NUMINAMATH_CALUDE_ratio_a_to_d_l1643_164314

theorem ratio_a_to_d (a b c d : ℚ) 
  (hab : a / b = 3 / 4)
  (hbc : b / c = 7 / 9)
  (hcd : c / d = 5 / 7) :
  a / d = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_ratio_a_to_d_l1643_164314


namespace NUMINAMATH_CALUDE_product_expansion_terms_count_l1643_164364

theorem product_expansion_terms_count :
  let a_terms := 3  -- number of terms in (a₁ + a₂ + a₃)
  let b_terms := 4  -- number of terms in (b₁ + b₂ + b₃ + b₄)
  let c_terms := 5  -- number of terms in (c₁ + c₂ + c₃ + c₄ + c₅)
  a_terms * b_terms * c_terms = 60 := by
sorry

end NUMINAMATH_CALUDE_product_expansion_terms_count_l1643_164364


namespace NUMINAMATH_CALUDE_total_volume_of_four_cubes_l1643_164349

theorem total_volume_of_four_cubes (edge_length : ℝ) (num_cubes : ℕ) :
  edge_length = 5 → num_cubes = 4 → (edge_length ^ 3) * num_cubes = 500 := by
  sorry

end NUMINAMATH_CALUDE_total_volume_of_four_cubes_l1643_164349


namespace NUMINAMATH_CALUDE_problem_solution_l1643_164332

theorem problem_solution (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h1 : x₁ + 4*x₂ + 9*x₃ + 16*x₄ + 25*x₅ = 2)
  (h2 : 4*x₁ + 9*x₂ + 16*x₃ + 25*x₄ + 36*x₅ = 15)
  (h3 : 9*x₁ + 16*x₂ + 25*x₃ + 36*x₄ + 49*x₅ = 130) :
  16*x₁ + 25*x₂ + 36*x₃ + 49*x₄ + 64*x₅ = 347 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1643_164332


namespace NUMINAMATH_CALUDE_inequality_implication_l1643_164394

theorem inequality_implication (x y : ℝ) : 5 * x > -5 * y → x + y > 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l1643_164394


namespace NUMINAMATH_CALUDE_purple_cars_count_l1643_164324

theorem purple_cars_count (total : ℕ) (blue red orange yellow purple green : ℕ) : 
  total = 1423 →
  blue = 2 * red →
  red = 3 * orange →
  yellow = orange / 2 →
  yellow = 3 * purple →
  green = 5 * purple →
  blue ≥ 200 →
  red ≥ 50 →
  total = blue + red + orange + yellow + purple + green →
  purple = 20 := by
  sorry

#check purple_cars_count

end NUMINAMATH_CALUDE_purple_cars_count_l1643_164324


namespace NUMINAMATH_CALUDE_pump_emptying_time_l1643_164361

/-- Given a pool and two pumps A and B:
    * Pump A can empty the pool in 4 hours alone
    * Pumps A and B together can empty the pool in 80 minutes
    Prove that pump B can empty the pool in 2 hours alone -/
theorem pump_emptying_time (pool : ℝ) (pump_a pump_b : ℝ → ℝ) :
  (pump_a pool = pool / 4) →  -- Pump A empties the pool in 4 hours
  (pump_a pool + pump_b pool = pool / (80 / 60)) →  -- A and B together empty the pool in 80 minutes
  (pump_b pool = pool / 2) :=  -- Pump B empties the pool in 2 hours
by sorry

end NUMINAMATH_CALUDE_pump_emptying_time_l1643_164361


namespace NUMINAMATH_CALUDE_picture_book_shelves_l1643_164334

theorem picture_book_shelves (books_per_shelf : ℕ) (mystery_shelves : ℕ) (total_books : ℕ) : 
  books_per_shelf = 7 → 
  mystery_shelves = 8 → 
  total_books = 70 → 
  (total_books - mystery_shelves * books_per_shelf) / books_per_shelf = 2 := by
sorry

end NUMINAMATH_CALUDE_picture_book_shelves_l1643_164334


namespace NUMINAMATH_CALUDE_speed_ratio_inverse_of_time_ratio_l1643_164363

/-- Proves that the ratio of speeds for two runners completing the same race
    is the inverse of the ratio of their completion times. -/
theorem speed_ratio_inverse_of_time_ratio
  (total_time : ℝ)
  (rickey_time : ℝ)
  (prejean_time : ℝ)
  (h1 : total_time = rickey_time + prejean_time)
  (h2 : rickey_time = 40)
  (h3 : total_time = 70)
  : (prejean_time / rickey_time) = (3 : ℝ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_speed_ratio_inverse_of_time_ratio_l1643_164363


namespace NUMINAMATH_CALUDE_sqrt_x_equals_3_x_squared_equals_y_squared_l1643_164330

-- Define x and y as functions of a
def x (a : ℝ) : ℝ := 1 - 2*a
def y (a : ℝ) : ℝ := 3*a - 4

-- Theorem 1: When √x = 3, a = -4
theorem sqrt_x_equals_3 : ∃ a : ℝ, x a = 9 ∧ a = -4 := by sorry

-- Theorem 2: There exist values of a such that x² = y² = 1 or x² = y² = 25
theorem x_squared_equals_y_squared :
  (∃ a : ℝ, (x a)^2 = (y a)^2 ∧ (x a)^2 = 1) ∨
  (∃ a : ℝ, (x a)^2 = (y a)^2 ∧ (x a)^2 = 25) := by sorry

end NUMINAMATH_CALUDE_sqrt_x_equals_3_x_squared_equals_y_squared_l1643_164330


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1643_164369

def A : Set ℝ := {0, 1, 2}
def B : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}

theorem intersection_of_A_and_B : A ∩ B = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1643_164369


namespace NUMINAMATH_CALUDE_total_tiles_l1643_164336

theorem total_tiles (yellow blue purple white : ℕ) : 
  yellow = 3 → 
  blue = yellow + 1 → 
  purple = 6 → 
  white = 7 → 
  yellow + blue + purple + white = 20 := by
sorry

end NUMINAMATH_CALUDE_total_tiles_l1643_164336


namespace NUMINAMATH_CALUDE_halloween_candy_theorem_l1643_164321

-- Define the number of candy pieces collected by each sibling
def maggies_candy : ℕ := 50
def harpers_candy : ℕ := maggies_candy + (maggies_candy * 3 / 10)
def neils_candy : ℕ := harpers_candy + (harpers_candy * 2 / 5)
def liams_candy : ℕ := neils_candy + (neils_candy * 1 / 5)

-- Define the total candy collected
def total_candy : ℕ := maggies_candy + harpers_candy + neils_candy + liams_candy

-- Theorem statement
theorem halloween_candy_theorem : total_candy = 315 := by
  sorry

end NUMINAMATH_CALUDE_halloween_candy_theorem_l1643_164321


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1643_164335

theorem quadratic_equation_solution :
  let x₁ : ℝ := (3 + Real.sqrt 41) / 4
  let x₂ : ℝ := (3 - Real.sqrt 41) / 4
  2 * x₁^2 - 3 * x₁ - 4 = 0 ∧ 2 * x₂^2 - 3 * x₂ - 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1643_164335


namespace NUMINAMATH_CALUDE_cuboid_circumscribed_sphere_area_l1643_164313

theorem cuboid_circumscribed_sphere_area (x y z : ℝ) : 
  x > 0 ∧ y > 0 ∧ z > 0 ∧ 
  x * y = Real.sqrt 6 ∧ 
  y * z = Real.sqrt 2 ∧ 
  z * x = Real.sqrt 3 → 
  4 * Real.pi * ((x^2 + y^2 + z^2) / 4) = 6 * Real.pi := by
sorry

end NUMINAMATH_CALUDE_cuboid_circumscribed_sphere_area_l1643_164313


namespace NUMINAMATH_CALUDE_solution_range_l1643_164342

-- Define the equation
def equation (x m : ℝ) : Prop :=
  (x + m) / (x - 2) - 3 = (x - 1) / (2 - x)

-- Define the theorem
theorem solution_range (m : ℝ) : 
  (∃ x : ℝ, x ≥ 0 ∧ x ≠ 2 ∧ equation x m) ↔ (m ≥ -5 ∧ m ≠ -3) :=
sorry

end NUMINAMATH_CALUDE_solution_range_l1643_164342


namespace NUMINAMATH_CALUDE_line_slope_intercept_sum_l1643_164307

/-- Given a line passing through points (1, -2) and (3, 4), 
    prove that the sum of its slope and y-intercept is -2 -/
theorem line_slope_intercept_sum : 
  ∀ (m b : ℝ), 
  (∀ (x y : ℝ), y = m * x + b → 
    ((x = 1 ∧ y = -2) ∨ (x = 3 ∧ y = 4))) → 
  m + b = -2 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_intercept_sum_l1643_164307


namespace NUMINAMATH_CALUDE_plan_y_cheaper_at_min_usage_l1643_164304

/-- Cost of Plan X in cents for z MB of data usage -/
def cost_plan_x (z : ℕ) : ℕ := 15 * z

/-- Cost of Plan Y in cents for z MB of data usage, without discount -/
def cost_plan_y_no_discount (z : ℕ) : ℕ := 3000 + 7 * z

/-- Cost of Plan Y in cents for z MB of data usage, with discount -/
def cost_plan_y_with_discount (z : ℕ) : ℕ := 
  if z > 500 then cost_plan_y_no_discount z - 1000 else cost_plan_y_no_discount z

/-- The minimum usage in MB where Plan Y becomes cheaper than Plan X -/
def min_usage : ℕ := 501

theorem plan_y_cheaper_at_min_usage : 
  cost_plan_y_with_discount min_usage < cost_plan_x min_usage ∧
  ∀ z : ℕ, z < min_usage → cost_plan_x z ≤ cost_plan_y_with_discount z :=
by sorry


end NUMINAMATH_CALUDE_plan_y_cheaper_at_min_usage_l1643_164304


namespace NUMINAMATH_CALUDE_fourth_number_value_l1643_164396

theorem fourth_number_value (a b c d e f g : ℝ) 
  (h1 : (a + b + c + d) / 4 = 4)
  (h2 : (d + e + f + g) / 4 = 4)
  (h3 : (a + b + c + d + e + f + g) / 7 = 3) :
  d = 11 := by sorry

end NUMINAMATH_CALUDE_fourth_number_value_l1643_164396


namespace NUMINAMATH_CALUDE_no_infinite_power_arithmetic_progression_l1643_164348

/-- Represents a term in the sequence of the form a^b -/
def PowerTerm := Nat → Nat

/-- Represents an arithmetic progression -/
def ArithmeticProgression (f : ℕ → ℕ) :=
  ∃ d : ℕ, ∀ n : ℕ, f (n + 1) = f n + d

/-- A function that checks if a number is of the form a^b with a, b positive integers and b ≥ 2 -/
def IsPowerForm (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 0 ∧ b ≥ 2 ∧ n = a^b

/-- The main theorem stating that no infinite non-constant arithmetic progression
    exists where each term is of the form a^b with a, b positive integers and b ≥ 2 -/
theorem no_infinite_power_arithmetic_progression :
  ¬∃ f : PowerTerm, ArithmeticProgression f ∧
    (∀ n, IsPowerForm (f n)) ∧
    (∃ d : ℕ, d > 0 ∧ ∀ n : ℕ, f (n + 1) = f n + d) :=
sorry

end NUMINAMATH_CALUDE_no_infinite_power_arithmetic_progression_l1643_164348


namespace NUMINAMATH_CALUDE_vkontakte_users_l1643_164312

-- Define the people as propositions (being on VKontakte)
variable (M : Prop) -- Marya Ivanovna
variable (I : Prop) -- Ivan Ilyich
variable (A : Prop) -- Alexandra Varfolomeevna
variable (P : Prop) -- Petr Petrovich

-- Define the conditions
def condition1 : Prop := M → (I ∧ A)
def condition2 : Prop := (A ∧ ¬P) ∨ (¬A ∧ P)
def condition3 : Prop := I ∨ M
def condition4 : Prop := I ↔ P

-- Theorem statement
theorem vkontakte_users 
  (h1 : condition1 M I A)
  (h2 : condition2 A P)
  (h3 : condition3 I M)
  (h4 : condition4 I P) :
  I ∧ P ∧ ¬M ∧ ¬A :=
sorry

end NUMINAMATH_CALUDE_vkontakte_users_l1643_164312


namespace NUMINAMATH_CALUDE_inequality_solution_l1643_164322

theorem inequality_solution (x : ℝ) : (3 * x - 9) / ((x - 3)^2) < 0 ↔ x < 3 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l1643_164322


namespace NUMINAMATH_CALUDE_bus_stop_time_l1643_164317

/-- Calculates the time a bus stops per hour given its speeds with and without stoppages -/
theorem bus_stop_time (speed_without_stops : ℝ) (speed_with_stops : ℝ) : 
  speed_without_stops = 48 →
  speed_with_stops = 12 →
  (1 - speed_with_stops / speed_without_stops) * 60 = 45 := by
  sorry

#check bus_stop_time

end NUMINAMATH_CALUDE_bus_stop_time_l1643_164317


namespace NUMINAMATH_CALUDE_simplify_product_of_square_roots_l1643_164306

theorem simplify_product_of_square_roots (x : ℝ) (hx : x ≥ 0) :
  Real.sqrt (50 * x) * Real.sqrt (18 * x) * Real.sqrt (32 * x) = 120 * x * Real.sqrt (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_simplify_product_of_square_roots_l1643_164306


namespace NUMINAMATH_CALUDE_employee_discount_percentage_l1643_164301

/-- Proves that the employee discount percentage is 10% given the problem conditions --/
theorem employee_discount_percentage 
  (wholesale_cost : ℝ)
  (markup_percentage : ℝ)
  (employee_paid_price : ℝ)
  (h1 : wholesale_cost = 200)
  (h2 : markup_percentage = 20)
  (h3 : employee_paid_price = 216) :
  let retail_price := wholesale_cost * (1 + markup_percentage / 100)
  let discount_amount := retail_price - employee_paid_price
  let discount_percentage := (discount_amount / retail_price) * 100
  discount_percentage = 10 := by sorry

end NUMINAMATH_CALUDE_employee_discount_percentage_l1643_164301


namespace NUMINAMATH_CALUDE_cube_volume_in_box_l1643_164392

theorem cube_volume_in_box (box_length box_width box_height : ℝ)
  (num_cubes : ℕ) (cube_volume : ℝ) :
  box_length = 8 →
  box_width = 9 →
  box_height = 12 →
  num_cubes = 24 →
  cube_volume * num_cubes = box_length * box_width * box_height →
  cube_volume = 36 :=
by sorry

end NUMINAMATH_CALUDE_cube_volume_in_box_l1643_164392


namespace NUMINAMATH_CALUDE_rectangle_tromino_subdivision_l1643_164302

theorem rectangle_tromino_subdivision (a b c d : ℕ) : 
  a = 1961 ∧ b = 1963 ∧ c = 1963 ∧ d = 1965 → 
  (¬(a * b % 3 = 0) ∧ c * d % 3 = 0) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_tromino_subdivision_l1643_164302


namespace NUMINAMATH_CALUDE_apple_pyramid_count_l1643_164393

/-- Represents the number of apples in a layer of the pyramid -/
def layer_count (length width : ℕ) : ℕ := length * width

/-- Represents the pyramid-like stack of apples -/
def apple_pyramid : ℕ :=
  let base := layer_count 4 6
  let second := layer_count 3 5
  let third := layer_count 2 4
  let top := layer_count 2 3  -- double row on top
  base + second + third + top

/-- Theorem stating that the apple pyramid contains exactly 53 apples -/
theorem apple_pyramid_count : apple_pyramid = 53 := by
  sorry

end NUMINAMATH_CALUDE_apple_pyramid_count_l1643_164393


namespace NUMINAMATH_CALUDE_carter_goals_l1643_164352

theorem carter_goals (carter shelby judah : ℝ) 
  (shelby_half : shelby = carter / 2)
  (judah_calc : judah = 2 * shelby - 3)
  (total_goals : carter + shelby + judah = 7) :
  carter = 4 := by
sorry

end NUMINAMATH_CALUDE_carter_goals_l1643_164352


namespace NUMINAMATH_CALUDE_principal_sum_from_interest_difference_l1643_164383

/-- Proves that for a given interest rate and time period, if the difference between
    compound interest and simple interest is 41, then the principal sum is 4100. -/
theorem principal_sum_from_interest_difference
  (rate : ℝ) (time : ℝ) (diff : ℝ) (p : ℝ) :
  rate = 10 →
  time = 2 →
  diff = 41 →
  diff = p * ((1 + rate / 100) ^ time - 1) - p * (rate * time / 100) →
  p = 4100 := by
  sorry

#check principal_sum_from_interest_difference

end NUMINAMATH_CALUDE_principal_sum_from_interest_difference_l1643_164383


namespace NUMINAMATH_CALUDE_samuel_has_twelve_apples_left_l1643_164315

/-- The number of apples Samuel has left after buying, eating, and making pie -/
def samuels_remaining_apples (bonnies_apples : ℕ) (samuels_extra_apples : ℕ) : ℕ :=
  let samuels_apples := bonnies_apples + samuels_extra_apples
  let after_eating := samuels_apples / 2
  let used_for_pie := after_eating / 7
  after_eating - used_for_pie

/-- Theorem stating that Samuel has 12 apples left -/
theorem samuel_has_twelve_apples_left :
  samuels_remaining_apples 8 20 = 12 := by
  sorry

#eval samuels_remaining_apples 8 20

end NUMINAMATH_CALUDE_samuel_has_twelve_apples_left_l1643_164315


namespace NUMINAMATH_CALUDE_virgo_boat_trip_duration_l1643_164386

/-- Represents the duration of a trip to Virgo island -/
structure VirgoTrip where
  boat_time : ℝ
  plane_time : ℝ
  total_time : ℝ

/-- Conditions for a valid Virgo trip -/
def is_valid_virgo_trip (trip : VirgoTrip) : Prop :=
  trip.plane_time = 4 * trip.boat_time ∧
  trip.total_time = trip.boat_time + trip.plane_time ∧
  trip.total_time = 10

theorem virgo_boat_trip_duration :
  ∀ (trip : VirgoTrip), is_valid_virgo_trip trip → trip.boat_time = 2 := by
  sorry

end NUMINAMATH_CALUDE_virgo_boat_trip_duration_l1643_164386


namespace NUMINAMATH_CALUDE_diamond_value_l1643_164350

theorem diamond_value (diamond : ℕ) : 
  diamond < 10 →  -- Ensuring diamond is a digit
  (9 * diamond + 3 = 10 * diamond + 2) →  -- Equivalent to ◇3_9 = ◇2_10
  diamond = 1 := by
sorry

end NUMINAMATH_CALUDE_diamond_value_l1643_164350


namespace NUMINAMATH_CALUDE_area_between_curves_l1643_164327

-- Define the two functions
def f (x : ℝ) := 3 * x
def g (x : ℝ) := x^2

-- Define the intersection points
def x₁ : ℝ := 0
def x₂ : ℝ := 3

-- State the theorem
theorem area_between_curves :
  ∫ x in x₁..x₂, (f x - g x) = 9/2 := by sorry

end NUMINAMATH_CALUDE_area_between_curves_l1643_164327


namespace NUMINAMATH_CALUDE_melissa_games_played_l1643_164382

theorem melissa_games_played (points_per_game : ℕ) (total_points : ℕ) (h1 : points_per_game = 12) (h2 : total_points = 36) :
  total_points / points_per_game = 3 := by
  sorry

end NUMINAMATH_CALUDE_melissa_games_played_l1643_164382


namespace NUMINAMATH_CALUDE_boxes_sold_proof_l1643_164397

/-- The number of boxes sold on Friday -/
def friday_boxes : ℕ := 40

/-- The number of boxes sold on Saturday -/
def saturday_boxes : ℕ := 2 * friday_boxes - 10

/-- The number of boxes sold on Sunday -/
def sunday_boxes : ℕ := (saturday_boxes) / 2

theorem boxes_sold_proof :
  friday_boxes + saturday_boxes + sunday_boxes = 145 :=
by sorry

end NUMINAMATH_CALUDE_boxes_sold_proof_l1643_164397


namespace NUMINAMATH_CALUDE_diamond_spade_ratio_l1643_164331

structure Deck :=
  (clubs : ℕ)
  (diamonds : ℕ)
  (hearts : ℕ)
  (spades : ℕ)

def is_valid_deck (d : Deck) : Prop :=
  d.clubs + d.diamonds + d.hearts + d.spades = 13 ∧
  d.clubs + d.spades = 7 ∧
  d.diamonds + d.hearts = 6 ∧
  d.hearts = 2 * d.diamonds ∧
  d.clubs = 6

theorem diamond_spade_ratio (d : Deck) (h : is_valid_deck d) :
  d.diamonds = 2 ∧ d.spades = 1 :=
sorry

end NUMINAMATH_CALUDE_diamond_spade_ratio_l1643_164331


namespace NUMINAMATH_CALUDE_older_females_count_l1643_164319

/-- Represents the population of a town divided into equal groups -/
structure TownPopulation where
  total : ℕ
  num_groups : ℕ
  h_positive : 0 < num_groups

/-- Calculates the size of each group in the town -/
def group_size (town : TownPopulation) : ℕ :=
  town.total / town.num_groups

/-- Theorem: In a town with 1000 people divided into 5 equal groups,
    the number of people in each group is 200 -/
theorem older_females_count (town : TownPopulation)
    (h_total : town.total = 1000)
    (h_groups : town.num_groups = 5) :
    group_size town = 200 := by
  sorry

#eval group_size ⟨1000, 5, by norm_num⟩

end NUMINAMATH_CALUDE_older_females_count_l1643_164319


namespace NUMINAMATH_CALUDE_halfway_fraction_l1643_164353

theorem halfway_fraction (a b : ℚ) (ha : a = 1/7) (hb : b = 1/4) :
  (a + b) / 2 = 11/56 := by sorry

end NUMINAMATH_CALUDE_halfway_fraction_l1643_164353


namespace NUMINAMATH_CALUDE_book_selling_price_l1643_164341

theorem book_selling_price 
  (num_books : ℕ) 
  (buying_price : ℚ) 
  (price_difference : ℚ) 
  (h1 : num_books = 15)
  (h2 : buying_price = 11)
  (h3 : price_difference = 210) :
  ∃ (selling_price : ℚ), 
    selling_price * num_books - buying_price * num_books = price_difference ∧ 
    selling_price = 25 :=
by sorry

end NUMINAMATH_CALUDE_book_selling_price_l1643_164341


namespace NUMINAMATH_CALUDE_problem_solution_l1643_164367

theorem problem_solution (p q : ℝ) 
  (h1 : 1 < p) (h2 : p < q) 
  (h3 : 1 / p + 1 / q = 1) 
  (h4 : p * q = 8) : 
  q = 4 + 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1643_164367


namespace NUMINAMATH_CALUDE_eight_book_distribution_l1643_164365

/-- The number of ways to distribute n identical books between two locations,
    with at least one book in each location. -/
def distribution_ways (n : ℕ) : ℕ :=
  if n ≥ 2 then n - 1 else 0

/-- Theorem stating that there are 7 ways to distribute 8 identical books
    between storage and students, with at least one book in each location. -/
theorem eight_book_distribution :
  distribution_ways 8 = 7 := by
  sorry

end NUMINAMATH_CALUDE_eight_book_distribution_l1643_164365


namespace NUMINAMATH_CALUDE_right_triangle_side_length_l1643_164303

theorem right_triangle_side_length 
  (a b c : ℝ) 
  (hyp : a^2 + b^2 = c^2) -- Pythagorean theorem
  (hyp_length : c = 5) -- Hypotenuse length
  (side_length : a = 3) -- Known side length
  : b = 4 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_side_length_l1643_164303


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l1643_164385

/-- A geometric sequence with first term a and common ratio q -/
def geometric_sequence (a q : ℝ) : ℕ → ℝ := fun n => a * q ^ (n - 1)

/-- Theorem: For a geometric sequence with common ratio q, if a_1 + a_3 = 10 and a_4 + a_6 = 5/4, then q = 1/2 -/
theorem geometric_sequence_ratio 
  (a q : ℝ) 
  (h1 : geometric_sequence a q 1 + geometric_sequence a q 3 = 10)
  (h2 : geometric_sequence a q 4 + geometric_sequence a q 6 = 5/4) :
  q = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l1643_164385


namespace NUMINAMATH_CALUDE_extreme_value_conditions_l1643_164375

def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

theorem extreme_value_conditions (a b : ℝ) :
  f a b 1 = 10 ∧ 
  (deriv (f a b)) 1 = 0 →
  a = 4 ∧ b = -11 := by sorry

end NUMINAMATH_CALUDE_extreme_value_conditions_l1643_164375


namespace NUMINAMATH_CALUDE_system_solution_l1643_164351

theorem system_solution : 
  ∃ (x y : ℚ), 5 * x + 3 * y = 17 ∧ 3 * x + 5 * y = 16 → x = 37 / 16 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1643_164351


namespace NUMINAMATH_CALUDE_road_renovation_rates_l1643_164323

-- Define the daily renovation rates for Team A and Team B
def daily_rate_A (x : ℝ) : ℝ := x + 20
def daily_rate_B (x : ℝ) : ℝ := x

-- Define the condition that the time to renovate 200m for Team A equals the time to renovate 150m for Team B
def time_equality (x : ℝ) : Prop := 200 / (daily_rate_A x) = 150 / (daily_rate_B x)

-- Theorem stating the solution
theorem road_renovation_rates :
  ∃ x : ℝ, time_equality x ∧ daily_rate_A x = 80 ∧ daily_rate_B x = 60 := by
  sorry

end NUMINAMATH_CALUDE_road_renovation_rates_l1643_164323


namespace NUMINAMATH_CALUDE_inverse_proportion_y_relationship_l1643_164356

theorem inverse_proportion_y_relationship :
  ∀ (y₁ y₂ y₃ : ℝ),
  (y₁ = -3 / (-3)) →
  (y₂ = -3 / (-1)) →
  (y₃ = -3 / (1/3)) →
  (y₃ < y₁) ∧ (y₁ < y₂) := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_y_relationship_l1643_164356


namespace NUMINAMATH_CALUDE_fraction_value_at_x_equals_one_l1643_164370

theorem fraction_value_at_x_equals_one :
  let f (x : ℝ) := (x^2 - 3*x - 10) / (x^2 - 4)
  f 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_at_x_equals_one_l1643_164370


namespace NUMINAMATH_CALUDE_solve_salary_problem_l1643_164366

def salary_problem (salaries : List ℝ) (mean : ℝ) : Prop :=
  let n : ℕ := salaries.length + 1
  let total : ℝ := mean * n
  let sum_known : ℝ := salaries.sum
  let sixth_salary : ℝ := total - sum_known
  salaries.length = 5 ∧ 
  mean = 2291.67 ∧
  sixth_salary = 2000.02

theorem solve_salary_problem (salaries : List ℝ) (mean : ℝ) 
  (h1 : salaries = [1000, 2500, 3100, 3650, 1500]) 
  (h2 : mean = 2291.67) : 
  salary_problem salaries mean := by
  sorry

end NUMINAMATH_CALUDE_solve_salary_problem_l1643_164366


namespace NUMINAMATH_CALUDE_parallelepiped_volume_exists_parallelepiped_with_volume_144_l1643_164398

/-- Represents the dimensions of a rectangular parallelepiped with a right triangle base -/
structure Parallelepiped where
  a : ℕ
  height : ℕ
  base_is_right_triangle : a^2 + (a+1)^2 = (a+2)^2

/-- The volume of the parallelepiped is 144 -/
theorem parallelepiped_volume (p : Parallelepiped) (h : p.height = 12) : a * (a + 1) * p.height = 144 := by
  sorry

/-- There exists a parallelepiped satisfying the conditions with volume 144 -/
theorem exists_parallelepiped_with_volume_144 : ∃ (p : Parallelepiped), p.height = 12 ∧ a * (a + 1) * p.height = 144 := by
  sorry

end NUMINAMATH_CALUDE_parallelepiped_volume_exists_parallelepiped_with_volume_144_l1643_164398


namespace NUMINAMATH_CALUDE_additional_three_pointers_l1643_164328

def points_to_tie : ℕ := 17
def points_over_record : ℕ := 5
def old_record : ℕ := 257
def free_throws : ℕ := 5
def regular_baskets : ℕ := 4
def normal_three_pointers : ℕ := 2

def points_per_free_throw : ℕ := 1
def points_per_regular_basket : ℕ := 2
def points_per_three_pointer : ℕ := 3

def total_points_final_game : ℕ := points_to_tie + points_over_record
def points_from_free_throws : ℕ := free_throws * points_per_free_throw
def points_from_regular_baskets : ℕ := regular_baskets * points_per_regular_basket
def points_from_three_pointers : ℕ := total_points_final_game - points_from_free_throws - points_from_regular_baskets

theorem additional_three_pointers (
  h1 : points_from_three_pointers % points_per_three_pointer = 0
) : (points_from_three_pointers / points_per_three_pointer) - normal_three_pointers = 1 := by
  sorry

end NUMINAMATH_CALUDE_additional_three_pointers_l1643_164328


namespace NUMINAMATH_CALUDE_essay_competition_probability_l1643_164360

theorem essay_competition_probability (n : ℕ) (h : n = 6) :
  let total_outcomes := n * n
  let favorable_outcomes := n * (n - 1)
  (favorable_outcomes : ℚ) / total_outcomes = 5 / 6 :=
by sorry

end NUMINAMATH_CALUDE_essay_competition_probability_l1643_164360


namespace NUMINAMATH_CALUDE_sphere_volume_with_diameter_10_l1643_164374

/-- The volume of a sphere with diameter 10 meters is 500/3 * π cubic meters. -/
theorem sphere_volume_with_diameter_10 :
  let diameter : ℝ := 10
  let radius : ℝ := diameter / 2
  let volume : ℝ := (4 / 3) * π * radius^3
  volume = (500 / 3) * π :=
by sorry

end NUMINAMATH_CALUDE_sphere_volume_with_diameter_10_l1643_164374


namespace NUMINAMATH_CALUDE_expression_one_l1643_164373

theorem expression_one : 5 * (-2)^2 - (-2)^3 / 4 = 22 := by sorry

end NUMINAMATH_CALUDE_expression_one_l1643_164373


namespace NUMINAMATH_CALUDE_derivative_value_at_five_l1643_164337

theorem derivative_value_at_five (f : ℝ → ℝ) (hf : ∀ x, f x = 3 * x^2 + 2 * x * (deriv f 2)) :
  deriv f 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_derivative_value_at_five_l1643_164337


namespace NUMINAMATH_CALUDE_cubic_function_properties_l1643_164345

-- Define the function f(x) = ax^3
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3

-- Theorem statement
theorem cubic_function_properties (a : ℝ) (h : a ≠ 0) :
  (∀ x y : ℝ, x < y → a < 0 → f a x > f a y) ∧
  (∃ x : ℝ, x ≠ 1 ∧ f a x = 3 * a * x - 2 * a) :=
by sorry

end NUMINAMATH_CALUDE_cubic_function_properties_l1643_164345


namespace NUMINAMATH_CALUDE_first_group_size_correct_l1643_164387

/-- The number of persons in the first group that can repair a road -/
def first_group_size : ℕ := 63

/-- The number of days the first group takes to repair the road -/
def first_group_days : ℕ := 12

/-- The number of hours per day the first group works -/
def first_group_hours : ℕ := 5

/-- The number of persons in the second group -/
def second_group_size : ℕ := 30

/-- The number of days the second group takes to repair the road -/
def second_group_days : ℕ := 21

/-- The number of hours per day the second group works -/
def second_group_hours : ℕ := 6

/-- The theorem stating that the first group size is correct -/
theorem first_group_size_correct :
  first_group_size * first_group_days * first_group_hours =
  second_group_size * second_group_days * second_group_hours :=
by sorry

end NUMINAMATH_CALUDE_first_group_size_correct_l1643_164387


namespace NUMINAMATH_CALUDE_g_composition_three_l1643_164357

def g (n : ℕ) : ℕ :=
  if n < 5 then n^2 + 1 else 2*n + 3

theorem g_composition_three : g (g (g 3)) = 49 := by
  sorry

end NUMINAMATH_CALUDE_g_composition_three_l1643_164357


namespace NUMINAMATH_CALUDE_nonagon_diagonals_l1643_164381

/-- A convex nonagon is a 9-sided polygon -/
def ConvexNonagon : Type := Unit

/-- The number of sides in a convex nonagon -/
def num_sides (n : ConvexNonagon) : ℕ := 9

/-- The number of distinct diagonals in a convex nonagon -/
def num_diagonals (n : ConvexNonagon) : ℕ := 27

theorem nonagon_diagonals (n : ConvexNonagon) : 
  num_diagonals n = (num_sides n * (num_sides n - 3)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_diagonals_l1643_164381


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1643_164346

theorem trigonometric_identity :
  (Real.sin (160 * π / 180) + Real.sin (40 * π / 180)) *
  (Real.sin (140 * π / 180) + Real.sin (20 * π / 180)) +
  (Real.sin (50 * π / 180) - Real.sin (70 * π / 180)) *
  (Real.sin (130 * π / 180) - Real.sin (110 * π / 180)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1643_164346


namespace NUMINAMATH_CALUDE_factory_sampling_is_systematic_l1643_164390

/-- Represents a sampling method -/
inductive SamplingMethod
| SimpleRandom
| Systematic
| Stratified

/-- Represents the characteristics of a sampling process -/
structure SamplingProcess where
  orderedArrangement : Bool
  fixedInterval : Bool

/-- Determines the sampling method based on the sampling process characteristics -/
def determineSamplingMethod (process : SamplingProcess) : SamplingMethod :=
  if process.orderedArrangement && process.fixedInterval then
    SamplingMethod.Systematic
  else
    SamplingMethod.SimpleRandom -- Default case, not actually used in this problem

/-- Theorem stating that the given sampling process is systematic sampling -/
theorem factory_sampling_is_systematic 
  (process : SamplingProcess)
  (h1 : process.orderedArrangement = true)
  (h2 : process.fixedInterval = true) :
  determineSamplingMethod process = SamplingMethod.Systematic := by
  sorry


end NUMINAMATH_CALUDE_factory_sampling_is_systematic_l1643_164390


namespace NUMINAMATH_CALUDE_alok_payment_l1643_164389

/-- Represents the order and prices of items in Alok's purchase --/
structure AlokOrder where
  chapati_quantity : ℕ
  rice_quantity : ℕ
  vegetable_quantity : ℕ
  icecream_quantity : ℕ
  chapati_price : ℕ
  rice_price : ℕ
  vegetable_price : ℕ

/-- Calculates the total cost of Alok's order --/
def total_cost (order : AlokOrder) : ℕ :=
  order.chapati_quantity * order.chapati_price +
  order.rice_quantity * order.rice_price +
  order.vegetable_quantity * order.vegetable_price

/-- Theorem stating that Alok's total payment is 811 --/
theorem alok_payment (order : AlokOrder)
  (h1 : order.chapati_quantity = 16)
  (h2 : order.rice_quantity = 5)
  (h3 : order.vegetable_quantity = 7)
  (h4 : order.icecream_quantity = 6)
  (h5 : order.chapati_price = 6)
  (h6 : order.rice_price = 45)
  (h7 : order.vegetable_price = 70) :
  total_cost order = 811 := by
  sorry

end NUMINAMATH_CALUDE_alok_payment_l1643_164389


namespace NUMINAMATH_CALUDE_median_divided_triangle_area_l1643_164305

/-- Given a triangle with sides 13, 14, and 15 cm, the area of each smaller triangle
    formed by its medians is 14 cm². -/
theorem median_divided_triangle_area (a b c : ℝ) (h1 : a = 13) (h2 : b = 14) (h3 : c = 15) :
  let s := (a + b + c) / 2
  let total_area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  total_area / 6 = 14 := by
  sorry

end NUMINAMATH_CALUDE_median_divided_triangle_area_l1643_164305


namespace NUMINAMATH_CALUDE_share_ratio_l1643_164388

/-- Proves that the ratio of B's share to C's share is 3:2 given the problem conditions -/
theorem share_ratio (total amount : ℕ) (a_share b_share c_share : ℕ) :
  amount = 544 →
  a_share = 384 →
  b_share = 96 →
  c_share = 64 →
  amount = a_share + b_share + c_share →
  a_share = (2 : ℚ) / 3 * b_share →
  b_share / c_share = (3 : ℚ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_share_ratio_l1643_164388


namespace NUMINAMATH_CALUDE_BaBr2_molecular_weight_l1643_164339

/-- The atomic weight of barium in g/mol -/
def atomic_weight_Ba : ℝ := 137.33

/-- The atomic weight of bromine in g/mol -/
def atomic_weight_Br : ℝ := 79.90

/-- The molecular weight of BaBr2 in g/mol -/
def molecular_weight_BaBr2 : ℝ := atomic_weight_Ba + 2 * atomic_weight_Br

/-- Theorem stating that the molecular weight of BaBr2 is 297.13 g/mol -/
theorem BaBr2_molecular_weight : 
  molecular_weight_BaBr2 = 297.13 := by sorry

end NUMINAMATH_CALUDE_BaBr2_molecular_weight_l1643_164339


namespace NUMINAMATH_CALUDE_min_rooms_for_departments_l1643_164325

def minRooms (d1 d2 d3 : Nat) : Nat :=
  let gcd := Nat.gcd (Nat.gcd d1 d2) d3
  (d1 / gcd) + (d2 / gcd) + (d3 / gcd)

theorem min_rooms_for_departments :
  minRooms 72 58 24 = 77 := by
  sorry

end NUMINAMATH_CALUDE_min_rooms_for_departments_l1643_164325


namespace NUMINAMATH_CALUDE_quadratic_equation_root_l1643_164320

theorem quadratic_equation_root (k : ℝ) : 
  (2 : ℝ) ∈ {x : ℝ | 2 * x^2 - 8 * x + k = 0} → k = 8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_l1643_164320


namespace NUMINAMATH_CALUDE_trig_identity_simplification_l1643_164300

theorem trig_identity_simplification (x y : ℝ) : 
  Real.cos (x + y) * Real.sin y - Real.sin (x + y) * Real.cos y = -Real.sin (x + y) := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_simplification_l1643_164300


namespace NUMINAMATH_CALUDE_zach_ben_score_difference_l1643_164399

theorem zach_ben_score_difference :
  ∀ (zach_score ben_score : ℕ),
    zach_score = 42 →
    ben_score = 21 →
    zach_score - ben_score = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_zach_ben_score_difference_l1643_164399


namespace NUMINAMATH_CALUDE_hexagon_interior_angles_sum_l1643_164354

/-- The sum of interior angles of a polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℕ := (n - 2) * 180

/-- A hexagon has 6 sides -/
def hexagon_sides : ℕ := 6

theorem hexagon_interior_angles_sum :
  sum_interior_angles hexagon_sides = 720 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_interior_angles_sum_l1643_164354


namespace NUMINAMATH_CALUDE_line_equation_sum_l1643_164378

/-- Given a line passing through points (1,2) and (4,11) with equation y = mx + b, prove that m + b = 2 -/
theorem line_equation_sum (m b : ℝ) : 
  (∀ x y : ℝ, y = m * x + b) →   -- Line equation
  (2 : ℝ) = m * 1 + b →          -- Point (1,2) satisfies the equation
  (11 : ℝ) = m * 4 + b →         -- Point (4,11) satisfies the equation
  m + b = 2 := by
sorry

end NUMINAMATH_CALUDE_line_equation_sum_l1643_164378


namespace NUMINAMATH_CALUDE_total_books_l1643_164329

theorem total_books (tim_books sam_books : ℕ) 
  (h1 : tim_books = 44) 
  (h2 : sam_books = 52) : 
  tim_books + sam_books = 96 := by
sorry

end NUMINAMATH_CALUDE_total_books_l1643_164329


namespace NUMINAMATH_CALUDE_vector_norm_equation_solution_l1643_164333

theorem vector_norm_equation_solution :
  let v : ℝ × ℝ := (3, -2)
  let w : ℝ × ℝ := (6, -1)
  let norm_squared (x : ℝ × ℝ) := x.1^2 + x.2^2
  { k : ℝ | norm_squared (k * v.1 - w.1, k * v.2 - w.2) = 34 } = {3, 1/13} := by
  sorry

end NUMINAMATH_CALUDE_vector_norm_equation_solution_l1643_164333


namespace NUMINAMATH_CALUDE_principal_amount_proof_l1643_164359

-- Define the interest rates for each year
def r1 : ℝ := 0.08
def r2 : ℝ := 0.10
def r3 : ℝ := 0.12
def r4 : ℝ := 0.09
def r5 : ℝ := 0.11

-- Define the total compound interest
def total_interest : ℝ := 4016.25

-- Define the compound interest factor
def compound_factor : ℝ := (1 + r1) * (1 + r2) * (1 + r3) * (1 + r4) * (1 + r5)

-- State the theorem
theorem principal_amount_proof :
  ∃ P : ℝ, P * (compound_factor - 1) = total_interest ∧ 
  abs (P - 7065.84) < 0.01 := by
sorry

end NUMINAMATH_CALUDE_principal_amount_proof_l1643_164359


namespace NUMINAMATH_CALUDE_total_gas_spent_l1643_164318

/-- Calculates the total amount spent on gas by Jim in North Carolina and Virginia -/
theorem total_gas_spent (nc_gallons : ℝ) (nc_price : ℝ) (va_gallons : ℝ) (price_difference : ℝ) :
  nc_gallons = 10 ∧ 
  nc_price = 2 ∧ 
  va_gallons = 10 ∧ 
  price_difference = 1 →
  nc_gallons * nc_price + va_gallons * (nc_price + price_difference) = 50 := by
  sorry

#check total_gas_spent

end NUMINAMATH_CALUDE_total_gas_spent_l1643_164318


namespace NUMINAMATH_CALUDE_max_value_implies_ratio_l1643_164355

/-- Given a function f(x) = 3sin(x) + 4cos(x) that reaches its maximum value at x = θ,
    prove that (sin(2θ) + cos²(θ) + 1) / cos(2θ) = 15/7 -/
theorem max_value_implies_ratio (θ : ℝ) 
  (h : ∀ x, 3 * Real.sin x + 4 * Real.cos x ≤ 3 * Real.sin θ + 4 * Real.cos θ) :
  (Real.sin (2 * θ) + Real.cos θ ^ 2 + 1) / Real.cos (2 * θ) = 15 / 7 := by
  sorry

end NUMINAMATH_CALUDE_max_value_implies_ratio_l1643_164355


namespace NUMINAMATH_CALUDE_car_speed_l1643_164384

/-- Proves that if a car travels 1 km in 5 seconds more than it would take at 90 km/hour, then its speed is 80 km/hour. -/
theorem car_speed (v : ℝ) (h : v > 0) : 
  (3600 / v) = (3600 / 90) + 5 → v = 80 := by
sorry

end NUMINAMATH_CALUDE_car_speed_l1643_164384


namespace NUMINAMATH_CALUDE_computer_table_markup_l1643_164311

/-- Calculate the percentage markup given the selling price and cost price -/
def percentage_markup (selling_price cost_price : ℚ) : ℚ :=
  (selling_price - cost_price) / cost_price * 100

/-- Theorem: The percentage markup for a computer table with selling price 3000 and cost price 2500 is 20% -/
theorem computer_table_markup :
  percentage_markup 3000 2500 = 20 := by
  sorry

end NUMINAMATH_CALUDE_computer_table_markup_l1643_164311


namespace NUMINAMATH_CALUDE_salt_water_evaporation_l1643_164344

/-- Given a salt water solution with initial weight of 200 grams and 5% salt concentration,
    if the salt concentration becomes 8% after evaporation,
    then 75 grams of water has evaporated. -/
theorem salt_water_evaporation (initial_weight : ℝ) (initial_concentration : ℝ) 
    (final_concentration : ℝ) (evaporated_water : ℝ) : 
  initial_weight = 200 →
  initial_concentration = 0.05 →
  final_concentration = 0.08 →
  initial_weight * initial_concentration = 
    (initial_weight - evaporated_water) * final_concentration →
  evaporated_water = 75 := by
  sorry

#check salt_water_evaporation

end NUMINAMATH_CALUDE_salt_water_evaporation_l1643_164344


namespace NUMINAMATH_CALUDE_kalebs_savings_l1643_164326

/-- Kaleb's initial savings problem -/
theorem kalebs_savings : ∀ (x : ℕ), 
  (x + 25 = 8 * 8) → x = 39 := by sorry

end NUMINAMATH_CALUDE_kalebs_savings_l1643_164326


namespace NUMINAMATH_CALUDE_largest_integer_with_remainder_l1643_164372

theorem largest_integer_with_remainder : ∃ n : ℕ, n < 100 ∧ n % 6 = 4 ∧ ∀ m : ℕ, m < 100 → m % 6 = 4 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_with_remainder_l1643_164372
