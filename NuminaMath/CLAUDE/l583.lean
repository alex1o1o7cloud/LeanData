import Mathlib

namespace NUMINAMATH_CALUDE_melanie_plums_l583_58385

def initial_plums : ℕ := 7
def plums_given : ℕ := 3

theorem melanie_plums : initial_plums - plums_given = 4 := by
  sorry

end NUMINAMATH_CALUDE_melanie_plums_l583_58385


namespace NUMINAMATH_CALUDE_card_drawing_theorem_l583_58317

/-- The number of cards of each color -/
def cards_per_color : ℕ := 4

/-- The total number of cards -/
def total_cards : ℕ := 4 * cards_per_color

/-- The number of cards to be drawn -/
def cards_drawn : ℕ := 3

/-- The number of ways to draw cards satisfying the conditions -/
def valid_draws : ℕ := 472

theorem card_drawing_theorem : 
  (Nat.choose total_cards cards_drawn) - 
  (4 * Nat.choose cards_per_color cards_drawn) - 
  (Nat.choose cards_per_color 2 * Nat.choose (total_cards - cards_per_color) 1) = 
  valid_draws := by sorry

end NUMINAMATH_CALUDE_card_drawing_theorem_l583_58317


namespace NUMINAMATH_CALUDE_philip_farm_animals_l583_58383

/-- The number of animals on Philip's farm -/
def total_animals (cows ducks pigs : ℕ) : ℕ := cows + ducks + pigs

/-- The number of cows on Philip's farm -/
def number_of_cows : ℕ := 20

/-- The number of ducks on Philip's farm -/
def number_of_ducks : ℕ := number_of_cows + (number_of_cows / 2)

/-- The number of pigs on Philip's farm -/
def number_of_pigs : ℕ := (number_of_cows + number_of_ducks) / 5

theorem philip_farm_animals :
  total_animals number_of_cows number_of_ducks number_of_pigs = 60 := by
  sorry

end NUMINAMATH_CALUDE_philip_farm_animals_l583_58383


namespace NUMINAMATH_CALUDE_not_divisible_by_8_main_result_l583_58355

def last_three_digits (n : ℕ) : ℕ :=
  n % 1000

theorem not_divisible_by_8 (n : ℕ) (h : n = 456294604884) :
  ¬(8 ∣ n) ↔ ¬(8 ∣ last_three_digits n) :=
by
  sorry

theorem main_result : ¬(8 ∣ 456294604884) :=
by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_8_main_result_l583_58355


namespace NUMINAMATH_CALUDE_painting_price_l583_58359

theorem painting_price (purchase_price : ℝ) (original_price : ℝ) : 
  purchase_price = 200 → 
  purchase_price = (1/4) * original_price → 
  original_price = 800 := by
sorry

end NUMINAMATH_CALUDE_painting_price_l583_58359


namespace NUMINAMATH_CALUDE_marker_sale_savings_l583_58311

/-- Calculates the savings when buying markers during a sale --/
def calculate_savings (original_price : ℚ) (num_markers : ℕ) (discount_rate : ℚ) : ℚ :=
  let original_total := original_price * num_markers
  let discounted_price := original_price * (1 - discount_rate)
  let free_markers := num_markers / 4
  let effective_markers := num_markers + free_markers
  let sale_total := discounted_price * num_markers
  original_total - sale_total

theorem marker_sale_savings :
  let original_price : ℚ := 3
  let num_markers : ℕ := 8
  let discount_rate : ℚ := 0.3
  calculate_savings original_price num_markers discount_rate = 36/5 := by sorry

end NUMINAMATH_CALUDE_marker_sale_savings_l583_58311


namespace NUMINAMATH_CALUDE_inequality_proof_l583_58374

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  Real.sqrt (1 + 2*a) + Real.sqrt (1 + 2*b) ≤ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l583_58374


namespace NUMINAMATH_CALUDE_symmetry_axis_implies_phi_l583_58348

theorem symmetry_axis_implies_phi (φ : ℝ) : 
  (∀ x, 2 * Real.sin (3 * x + φ) = 2 * Real.sin (3 * (π / 6 - x) + φ)) →
  |φ| < π / 2 →
  φ = π / 4 := by
sorry

end NUMINAMATH_CALUDE_symmetry_axis_implies_phi_l583_58348


namespace NUMINAMATH_CALUDE_horse_speed_around_square_field_l583_58388

theorem horse_speed_around_square_field 
  (field_area : ℝ) 
  (time_to_run_around : ℝ) 
  (horse_speed : ℝ) : 
  field_area = 400 ∧ 
  time_to_run_around = 4 → 
  horse_speed = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_horse_speed_around_square_field_l583_58388


namespace NUMINAMATH_CALUDE_unique_hour_conversion_l583_58373

theorem unique_hour_conversion : 
  ∃! n : ℕ, 
    (∃ x y : ℕ, x < 10 ∧ y < 10 ∧ n = 234000 + x * 1000 + y * 100) ∧ 
    (n % 3600 = 0) ∧
    (∃ h : ℕ, n = h * 3600) :=
by
  sorry

end NUMINAMATH_CALUDE_unique_hour_conversion_l583_58373


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_72_8712_l583_58316

theorem gcd_lcm_sum_72_8712 : Nat.gcd 72 8712 + Nat.lcm 72 8712 = 26160 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_72_8712_l583_58316


namespace NUMINAMATH_CALUDE_tammy_climbing_l583_58399

/-- Tammy's mountain climbing problem -/
theorem tammy_climbing (total_time total_distance : ℝ) 
  (speed_diff time_diff : ℝ) : 
  total_time = 14 →
  speed_diff = 0.5 →
  time_diff = 2 →
  total_distance = 52 →
  ∃ (speed1 time1 : ℝ),
    speed1 * time1 + (speed1 + speed_diff) * (time1 - time_diff) = total_distance ∧
    time1 + (time1 - time_diff) = total_time ∧
    speed1 + speed_diff = 4 :=
by sorry

end NUMINAMATH_CALUDE_tammy_climbing_l583_58399


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_proposition_l583_58310

theorem arithmetic_geometric_sequence_proposition :
  let p : Prop := ∀ (a : ℕ → ℝ), (∀ n, a (n + 1) - a n = a 1 - a 0) → a 1 - a 0 ≠ 0
  let q : Prop := ∀ (g : ℕ → ℝ), (∀ n, g (n + 1) / g n = g 1 / g 0) → g 1 / g 0 ≠ 1
  ¬p ∧ ¬q → (¬p ∨ ¬q) :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_proposition_l583_58310


namespace NUMINAMATH_CALUDE_second_reduction_percentage_store_price_reduction_l583_58340

theorem second_reduction_percentage 
  (initial_reduction : Real) 
  (final_price_percentage : Real) : Real :=
  let price_after_first_reduction := 1 - initial_reduction
  let second_reduction := (price_after_first_reduction - final_price_percentage) / price_after_first_reduction
  14 / 100

-- The main theorem
theorem store_price_reduction 
  (initial_reduction : Real)
  (final_price_percentage : Real)
  (h1 : initial_reduction = 10 / 100)
  (h2 : final_price_percentage = 77.4 / 100) :
  second_reduction_percentage initial_reduction final_price_percentage = 14 / 100 := by
  sorry

end NUMINAMATH_CALUDE_second_reduction_percentage_store_price_reduction_l583_58340


namespace NUMINAMATH_CALUDE_problem_solution_l583_58392

theorem problem_solution (a b c : ℕ+) 
  (eq1 : a^3 + 32*b + 2*c = 2018)
  (eq2 : b^3 + 32*a + 2*c = 1115) :
  a^2 + b^2 + c^2 = 226 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l583_58392


namespace NUMINAMATH_CALUDE_stairs_climbed_l583_58367

theorem stairs_climbed (jonny_stairs : ℕ) (julia_stairs : ℕ) : 
  jonny_stairs = 4872 → 
  julia_stairs = Int.floor (2 * Real.sqrt (jonny_stairs / 2) + 15) → 
  jonny_stairs + julia_stairs = 4986 := by
sorry

end NUMINAMATH_CALUDE_stairs_climbed_l583_58367


namespace NUMINAMATH_CALUDE_prism_pyramid_height_relation_l583_58350

/-- Given an equilateral triangle with side length a, prove that if a prism and a pyramid
    are constructed on this triangle with height m, and the lateral surface area of the prism
    equals the lateral surface area of the pyramid, then m = a/6 -/
theorem prism_pyramid_height_relation (a : ℝ) (m : ℝ) (h_pos : a > 0) : 
  (3 * a * m = (3 * a / 2) * Real.sqrt (m^2 + a^2 / 12)) → m = a / 6 := by
  sorry

end NUMINAMATH_CALUDE_prism_pyramid_height_relation_l583_58350


namespace NUMINAMATH_CALUDE_square_reciprocal_sum_l583_58360

theorem square_reciprocal_sum (m : ℝ) (h : m + 1/m = 10) : m^2 + 1/m^2 + 4 = 102 := by
  sorry

end NUMINAMATH_CALUDE_square_reciprocal_sum_l583_58360


namespace NUMINAMATH_CALUDE_constant_age_difference_l583_58338

/-- The age difference between two brothers remains constant over time -/
theorem constant_age_difference (a b x : ℕ) : (a + x) - (b + x) = a - b := by
  sorry

end NUMINAMATH_CALUDE_constant_age_difference_l583_58338


namespace NUMINAMATH_CALUDE_linear_expression_bounds_l583_58331

/-- Given a system of equations and constraints, prove the bounds of a linear expression. -/
theorem linear_expression_bounds (x y z : ℝ) : 
  x ≥ 0 → y ≥ 0 → z ≥ 0 → 
  x - 2*y - 3*z = -10 → 
  x + 2*y + z = 6 → 
  ∃ (A_min A_max : ℝ), 
    (∀ A, A = 1.5*x + y - z → A ≥ A_min ∧ A ≤ A_max) ∧
    A_min = -1 ∧ A_max = 0 := by
  sorry

end NUMINAMATH_CALUDE_linear_expression_bounds_l583_58331


namespace NUMINAMATH_CALUDE_collinear_complex_points_l583_58393

theorem collinear_complex_points (z : ℂ) : 
  (∃ (t : ℝ), z = 1 + t * (Complex.I - 1)) → Complex.abs z = 5 → 
  (z = 4 - 3 * Complex.I ∨ z = -3 + 4 * Complex.I) :=
by sorry

end NUMINAMATH_CALUDE_collinear_complex_points_l583_58393


namespace NUMINAMATH_CALUDE_candidate_votes_proof_l583_58308

theorem candidate_votes_proof (total_votes : ℕ) (invalid_percentage : ℚ) (candidate_percentage : ℚ) :
  total_votes = 560000 →
  invalid_percentage = 15 / 100 →
  candidate_percentage = 80 / 100 →
  ⌊(1 - invalid_percentage) * candidate_percentage * total_votes⌋ = 380800 := by
  sorry

end NUMINAMATH_CALUDE_candidate_votes_proof_l583_58308


namespace NUMINAMATH_CALUDE_homework_students_l583_58378

theorem homework_students (total : ℕ) (silent_reading : ℚ) (board_games : ℚ) (group_discussions : ℚ)
  (h_total : total = 120)
  (h_silent : silent_reading = 2 / 5)
  (h_board : board_games = 3 / 10)
  (h_group : group_discussions = 1 / 8) :
  total - (silent_reading * total + board_games * total + group_discussions * total).floor = 21 :=
by sorry

end NUMINAMATH_CALUDE_homework_students_l583_58378


namespace NUMINAMATH_CALUDE_parallel_line_length_l583_58318

/-- Given a triangle with base 36, prove that a line parallel to the base
    that divides the area into two equal parts has a length of 18√2. -/
theorem parallel_line_length (base : ℝ) (h_base : base = 36) :
  ∃ (line_length : ℝ), line_length = 18 * Real.sqrt 2 ∧
  ∀ (triangle_area smaller_area : ℝ),
    smaller_area = triangle_area / 2 →
    line_length ^ 2 / base ^ 2 = smaller_area / triangle_area :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_length_l583_58318


namespace NUMINAMATH_CALUDE_age_difference_l583_58335

/-- The difference in ages between (x + y) and (y + z) is 12 years, given that z is 12 years younger than x -/
theorem age_difference (x y z : ℕ) (h : z = x - 12) :
  (x + y) - (y + z) = 12 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l583_58335


namespace NUMINAMATH_CALUDE_kfc_chicken_legs_l583_58301

/-- Given the number of thighs, wings, and platters, calculate the number of legs baked. -/
def chicken_legs_baked (thighs wings platters : ℕ) : ℕ :=
  let thighs_per_platter := thighs / platters
  thighs_per_platter * platters

/-- Theorem stating that 144 chicken legs were baked given the problem conditions. -/
theorem kfc_chicken_legs :
  let thighs := 144
  let wings := 224
  let platters := 16
  chicken_legs_baked thighs wings platters = 144 := by
  sorry

#eval chicken_legs_baked 144 224 16

end NUMINAMATH_CALUDE_kfc_chicken_legs_l583_58301


namespace NUMINAMATH_CALUDE_division_of_decimals_l583_58387

theorem division_of_decimals : (0.45 : ℚ) / (0.005 : ℚ) = 90 := by sorry

end NUMINAMATH_CALUDE_division_of_decimals_l583_58387


namespace NUMINAMATH_CALUDE_third_house_price_l583_58361

/-- Brian's commission rate as a decimal -/
def commission_rate : ℚ := 0.02

/-- Selling price of the first house -/
def house1_price : ℚ := 157000

/-- Selling price of the second house -/
def house2_price : ℚ := 499000

/-- Total commission Brian earned from all three sales -/
def total_commission : ℚ := 15620

/-- The selling price of the third house -/
def house3_price : ℚ := (total_commission - (house1_price * commission_rate + house2_price * commission_rate)) / commission_rate

theorem third_house_price :
  house3_price = 125000 :=
by sorry

end NUMINAMATH_CALUDE_third_house_price_l583_58361


namespace NUMINAMATH_CALUDE_simplify_expression_l583_58330

theorem simplify_expression (y : ℝ) : 
  5 * y - 6 * y^2 + 9 - (4 - 5 * y + 2 * y^2) = -8 * y^2 + 10 * y + 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l583_58330


namespace NUMINAMATH_CALUDE_gcd_lcm_properties_l583_58370

theorem gcd_lcm_properties (a b : ℕ+) (h1 : Nat.gcd a b = 20) (h2 : Nat.lcm a b = 100) : 
  (a * b = 2000) ∧ 
  (Nat.lcm (10 * a) (10 * b) = 10 * Nat.lcm a b) ∧ 
  ((10 * a) * (10 * b) = 100 * (a * b)) := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_properties_l583_58370


namespace NUMINAMATH_CALUDE_cost_price_of_ball_l583_58375

theorem cost_price_of_ball (selling_price : ℕ) (num_balls_sold : ℕ) (num_balls_loss : ℕ) :
  selling_price = 720 →
  num_balls_sold = 15 →
  num_balls_loss = 5 →
  ∃ (cost_price : ℕ), 
    cost_price * num_balls_sold - cost_price * num_balls_loss = selling_price ∧
    cost_price = 72 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_of_ball_l583_58375


namespace NUMINAMATH_CALUDE_cubic_root_sum_cubes_l583_58398

theorem cubic_root_sum_cubes (a b c : ℝ) : 
  (4 * a^3 + 502 * a + 1004 = 0) →
  (4 * b^3 + 502 * b + 1004 = 0) →
  (4 * c^3 + 502 * c + 1004 = 0) →
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 753 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_cubes_l583_58398


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l583_58364

def B : Matrix (Fin 3) (Fin 3) ℚ := !![1, 2, 3; 0, 1, 2; 1, 0, 1]

theorem matrix_equation_solution :
  B^3 + (-5 : ℚ) • B^2 + 3 • B + (-6 : ℚ) • (1 : Matrix (Fin 3) (Fin 3) ℚ) = 0 := by
  sorry

end NUMINAMATH_CALUDE_matrix_equation_solution_l583_58364


namespace NUMINAMATH_CALUDE_dice_roll_probability_prob_first_less_than_second_l583_58309

/-- The probability that when rolling two fair six-sided dice, the first roll is less than the second roll -/
theorem dice_roll_probability : ℚ :=
  5/12

/-- A fair six-sided die -/
def fair_die : Finset ℕ := Finset.range 6

/-- The sample space of rolling two dice -/
def two_dice_rolls : Finset (ℕ × ℕ) :=
  fair_die.product fair_die

/-- The event where the first roll is less than the second roll -/
def first_less_than_second : Set (ℕ × ℕ) :=
  {p | p.1 < p.2}

/-- The probability of the event where the first roll is less than the second roll -/
theorem prob_first_less_than_second :
  (two_dice_rolls.filter (λ p => p.1 < p.2)).card / two_dice_rolls.card = dice_roll_probability := by
  sorry

end NUMINAMATH_CALUDE_dice_roll_probability_prob_first_less_than_second_l583_58309


namespace NUMINAMATH_CALUDE_h_range_proof_range_sum_l583_58344

noncomputable def h (x : ℝ) : ℝ := 3 / (3 + 9 * x^2)

theorem h_range_proof :
  ∀ x : ℝ, h x > 0 ∧
  (∀ ε > 0, ∃ N : ℝ, ∀ x : ℝ, |x| > N → h x < ε) ∧
  (∀ x : ℝ, h x ≤ h 0) →
  Set.range h = Set.Ioo 0 1 := by
sorry

theorem range_sum :
  Set.range h = Set.Ioo 0 1 →
  0 + 1 = 1 := by
sorry

end NUMINAMATH_CALUDE_h_range_proof_range_sum_l583_58344


namespace NUMINAMATH_CALUDE_range_of_m_l583_58371

-- Define the sets A and B
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2 * m - 1}

-- State the theorem
theorem range_of_m (m : ℝ) : (B m ∩ A = B m) ↔ m ≤ 3 :=
  sorry

end NUMINAMATH_CALUDE_range_of_m_l583_58371


namespace NUMINAMATH_CALUDE_y_value_l583_58332

theorem y_value (x y : ℤ) (h1 : x + y = 270) (h2 : x - y = 200) : y = 35 := by
  sorry

end NUMINAMATH_CALUDE_y_value_l583_58332


namespace NUMINAMATH_CALUDE_cube_surface_area_l583_58358

/-- The surface area of a cube with edge length 11 centimeters is 726 square centimeters. -/
theorem cube_surface_area : 
  let edge_length : ℝ := 11
  let surface_area : ℝ := 6 * edge_length^2
  surface_area = 726 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_l583_58358


namespace NUMINAMATH_CALUDE_brothers_ages_l583_58356

theorem brothers_ages (a b c : ℕ+) :
  a * b * c = 36 ∧ 
  a + b + c = 13 ∧ 
  (a ≤ b ∧ b ≤ c) ∧
  (b < c ∨ a < b) →
  a = 2 ∧ b = 2 ∧ c = 9 := by
  sorry

end NUMINAMATH_CALUDE_brothers_ages_l583_58356


namespace NUMINAMATH_CALUDE_f_n_ratio_theorem_l583_58346

noncomputable section

def f (x : ℝ) : ℝ := (x^2 + 1) / (2*x)

def f_n : ℕ → ℝ → ℝ
| 0, x => x
| n+1, x => f (f_n n x)

def N (n : ℕ) : ℕ := 2^n

theorem f_n_ratio_theorem (x : ℝ) (n : ℕ) (hx : x ≠ -1 ∧ x ≠ 0 ∧ x ≠ 1) :
  (f_n n x) / (f_n (n+1) x) = 1 + 1 / f ((((x+1)/(x-1)) ^ (N n))) :=
sorry

end NUMINAMATH_CALUDE_f_n_ratio_theorem_l583_58346


namespace NUMINAMATH_CALUDE_smoothie_cost_l583_58337

def burger_cost : ℝ := 5
def sandwich_cost : ℝ := 4
def total_order_cost : ℝ := 17
def num_smoothies : ℕ := 2

theorem smoothie_cost :
  let non_smoothie_cost := burger_cost + sandwich_cost
  let smoothie_total_cost := total_order_cost - non_smoothie_cost
  let smoothie_cost := smoothie_total_cost / num_smoothies
  smoothie_cost = 4 := by sorry

end NUMINAMATH_CALUDE_smoothie_cost_l583_58337


namespace NUMINAMATH_CALUDE_book_ratio_l583_58339

def book_tournament (candice amanda kara patricia taylor : ℕ) : Prop :=
  candice = 3 * amanda ∧
  kara = amanda / 2 ∧
  patricia = 7 * kara ∧
  taylor = (candice + amanda + kara + patricia) / 4 ∧
  candice = 18

theorem book_ratio (candice amanda kara patricia taylor : ℕ) :
  book_tournament candice amanda kara patricia taylor →
  taylor * 5 = candice + amanda + kara + patricia + taylor :=
by sorry

end NUMINAMATH_CALUDE_book_ratio_l583_58339


namespace NUMINAMATH_CALUDE_glutinous_rice_ball_probability_l583_58363

theorem glutinous_rice_ball_probability :
  let total_balls : ℕ := 6
  let sesame_balls : ℕ := 1
  let peanut_balls : ℕ := 2
  let red_bean_balls : ℕ := 3
  let selection_size : ℕ := 2

  (sesame_balls + peanut_balls + red_bean_balls = total_balls) →
  (Nat.choose total_balls selection_size ≠ 0) →

  (Nat.choose peanut_balls selection_size +
   peanut_balls * (sesame_balls + red_bean_balls)) /
  Nat.choose total_balls selection_size =
  3 / 5 :=
by sorry

end NUMINAMATH_CALUDE_glutinous_rice_ball_probability_l583_58363


namespace NUMINAMATH_CALUDE_probability_N_16_mod_7_eq_1_l583_58390

theorem probability_N_16_mod_7_eq_1 (N : ℕ) : 
  (∃ (k : ℕ), N = k ∧ 1 ≤ k ∧ k ≤ 2027) →
  (Nat.card {k : ℕ | 1 ≤ k ∧ k ≤ 2027 ∧ (k^16 % 7 = 1)}) / 2027 = 2 / 7 :=
by sorry

end NUMINAMATH_CALUDE_probability_N_16_mod_7_eq_1_l583_58390


namespace NUMINAMATH_CALUDE_problem_solution_l583_58322

/-- Given that (k-1)x^|k| + 3 ≥ 0 is a one-variable linear inequality about x and (k-1) ≠ 0, prove that k = -1 -/
theorem problem_solution (k : ℝ) : 
  (∀ x, ∃ a b, (k - 1) * x^(|k|) + 3 = a * x + b) → -- Linear inequality condition
  (k - 1 ≠ 0) →                                     -- Non-zero coefficient condition
  k = -1 :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l583_58322


namespace NUMINAMATH_CALUDE_brother_scores_double_l583_58394

/-- Represents the hockey goal scoring scenario of Louie and his brother -/
structure HockeyScenario where
  louie_last_match : ℕ
  louie_previous : ℕ
  brother_seasons : ℕ
  games_per_season : ℕ
  total_goals : ℕ

/-- The ratio of Louie's brother's goals per game to Louie's goals in the last match -/
def brother_to_louie_ratio (h : HockeyScenario) : ℚ :=
  let brother_total_games := h.brother_seasons * h.games_per_season
  let brother_total_goals := h.total_goals - (h.louie_last_match + h.louie_previous)
  (brother_total_goals / brother_total_games : ℚ) / h.louie_last_match

/-- The main theorem stating the ratio is 2:1 -/
theorem brother_scores_double (h : HockeyScenario) 
    (h_louie_last : h.louie_last_match = 4)
    (h_louie_prev : h.louie_previous = 40)
    (h_seasons : h.brother_seasons = 3)
    (h_games : h.games_per_season = 50)
    (h_total : h.total_goals = 1244) : 
  brother_to_louie_ratio h = 2 := by
  sorry

end NUMINAMATH_CALUDE_brother_scores_double_l583_58394


namespace NUMINAMATH_CALUDE_train_speed_l583_58329

/-- Proves that a train with given parameters has a speed of 45 km/hr -/
theorem train_speed (train_length : Real) (crossing_time : Real) (total_length : Real) :
  train_length = 100 →
  crossing_time = 30 →
  total_length = 275 →
  (total_length - train_length) / crossing_time * 3.6 = 45 :=
by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_train_speed_l583_58329


namespace NUMINAMATH_CALUDE_natural_number_representation_l583_58305

theorem natural_number_representation (k : ℕ) : 
  ∃ n : ℕ, k = 3*n ∨ k = 3*n + 1 ∨ k = 3*n + 2 :=
sorry

end NUMINAMATH_CALUDE_natural_number_representation_l583_58305


namespace NUMINAMATH_CALUDE_a_equals_zero_l583_58334

theorem a_equals_zero (a : ℝ) : 
  let A : Set ℝ := {a + 2, (a + 1)^2, a^2 + 3*a + 3}
  1 ∈ A → a = 0 := by
sorry

end NUMINAMATH_CALUDE_a_equals_zero_l583_58334


namespace NUMINAMATH_CALUDE_greatest_prime_factor_f_36_l583_58304

def f (m : ℕ) : ℕ := Finset.prod (Finset.filter (λ x => Even x) (Finset.range (m + 1))) id

theorem greatest_prime_factor_f_36 :
  ∃ (p : ℕ), Prime p ∧ p ∣ f 36 ∧ ∀ (q : ℕ), Prime q → q ∣ f 36 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_f_36_l583_58304


namespace NUMINAMATH_CALUDE_log_243_between_consecutive_integers_l583_58380

theorem log_243_between_consecutive_integers (a b : ℤ) :
  (a : ℝ) < Real.log 243 / Real.log 5 ∧
  Real.log 243 / Real.log 5 < (b : ℝ) ∧
  b = a + 1 →
  a + b = 7 := by sorry

end NUMINAMATH_CALUDE_log_243_between_consecutive_integers_l583_58380


namespace NUMINAMATH_CALUDE_lcm_factor_is_twelve_l583_58384

def problem (A B X : ℕ) : Prop :=
  A > 0 ∧ B > 0 ∧
  Nat.gcd A B = 42 ∧
  A = 504 ∧
  Nat.lcm A B = 42 * X

theorem lcm_factor_is_twelve :
  ∀ A B X, problem A B X → X = 12 := by sorry

end NUMINAMATH_CALUDE_lcm_factor_is_twelve_l583_58384


namespace NUMINAMATH_CALUDE_deal_or_no_deal_probability_l583_58381

/-- The set of values in the Deal or No Deal game -/
def deal_values : Finset ℕ := {1, 5, 10, 25, 50, 75, 100, 200, 300, 400, 500, 750, 1000, 5000, 10000, 25000, 50000, 75000, 100000, 200000, 300000, 400000, 500000, 750000, 1000000}

/-- The number of boxes in the game -/
def total_boxes : ℕ := 26

/-- The threshold value for high-value boxes -/
def threshold : ℕ := 200000

/-- The set of high-value boxes -/
def high_value_boxes : Finset ℕ := deal_values.filter (λ x => x ≥ threshold)

/-- The number of boxes to eliminate -/
def boxes_to_eliminate : ℕ := 14

theorem deal_or_no_deal_probability :
  (total_boxes - boxes_to_eliminate) / 2 = high_value_boxes.card ∧
  (total_boxes - boxes_to_eliminate) % 2 = 0 :=
sorry

#eval deal_values.card
#eval total_boxes
#eval high_value_boxes
#eval boxes_to_eliminate

end NUMINAMATH_CALUDE_deal_or_no_deal_probability_l583_58381


namespace NUMINAMATH_CALUDE_hash_2_3_1_5_equals_6_l583_58376

/-- The # operation for real numbers -/
def hash (a b c d : ℝ) : ℝ := b^2 - 4*a*c + d

/-- Theorem stating that #(2, 3, 1, 5) = 6 -/
theorem hash_2_3_1_5_equals_6 : hash 2 3 1 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_hash_2_3_1_5_equals_6_l583_58376


namespace NUMINAMATH_CALUDE_sixty_percent_high_profit_puppies_l583_58320

/-- Represents a litter of puppies with their spot counts -/
structure PuppyLitter where
  total : Nat
  fiveSpots : Nat
  fourSpots : Nat
  twoSpots : Nat

/-- Calculates the percentage of puppies with more than 4 spots -/
def percentageHighProfitPuppies (litter : PuppyLitter) : Rat :=
  (litter.fiveSpots : Rat) / (litter.total : Rat) * 100

/-- The theorem stating that for the given litter, 60% of puppies can be sold for greater profit -/
theorem sixty_percent_high_profit_puppies (litter : PuppyLitter)
    (h1 : litter.total = 10)
    (h2 : litter.fiveSpots = 6)
    (h3 : litter.fourSpots = 3)
    (h4 : litter.twoSpots = 1)
    (h5 : litter.fiveSpots + litter.fourSpots + litter.twoSpots = litter.total) :
    percentageHighProfitPuppies litter = 60 := by
  sorry

end NUMINAMATH_CALUDE_sixty_percent_high_profit_puppies_l583_58320


namespace NUMINAMATH_CALUDE_shirt_cost_l583_58302

/-- Given the cost equations for jeans, shirts, and hats, prove the cost of a shirt. -/
theorem shirt_cost (j s h : ℚ) 
  (eq1 : 3 * j + 2 * s + h = 89)
  (eq2 : 2 * j + 3 * s + 2 * h = 102)
  (eq3 : 4 * j + s + 3 * h = 125) :
  s = 12.53 := by
  sorry

end NUMINAMATH_CALUDE_shirt_cost_l583_58302


namespace NUMINAMATH_CALUDE_negation_of_existence_is_universal_l583_58365

variable (a : ℝ)

theorem negation_of_existence_is_universal :
  (¬ ∃ x : ℝ, x^2 + a*x + 1 < 0) ↔ (∀ x : ℝ, x^2 + a*x + 1 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_is_universal_l583_58365


namespace NUMINAMATH_CALUDE_converse_and_inverse_false_l583_58327

-- Define the properties of triangles
def IsEquilateral (t : Type) : Prop := sorry
def IsIsosceles (t : Type) : Prop := sorry

-- Given statement
axiom given_statement : ∀ t, IsEquilateral t → IsIsosceles t

-- Theorem to prove
theorem converse_and_inverse_false :
  (¬ ∀ t, IsIsosceles t → IsEquilateral t) ∧
  (¬ ∀ t, ¬IsEquilateral t → ¬IsIsosceles t) :=
by sorry

end NUMINAMATH_CALUDE_converse_and_inverse_false_l583_58327


namespace NUMINAMATH_CALUDE_largest_number_with_sum_14_l583_58323

def is_valid_digit (d : ℕ) : Prop := d = 1 ∨ d = 2 ∨ d = 3

def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def all_valid_digits (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, is_valid_digit d

theorem largest_number_with_sum_14 :
  ∀ n : ℕ,
    all_valid_digits n →
    digit_sum n = 14 →
    n ≤ 3332 :=
sorry

end NUMINAMATH_CALUDE_largest_number_with_sum_14_l583_58323


namespace NUMINAMATH_CALUDE_modular_exponentiation_difference_l583_58333

theorem modular_exponentiation_difference (n : ℕ) :
  (45^2011 - 23^2011) % 7 = 5 := by sorry

end NUMINAMATH_CALUDE_modular_exponentiation_difference_l583_58333


namespace NUMINAMATH_CALUDE_baylor_freelance_earnings_l583_58349

theorem baylor_freelance_earnings (initial_amount : ℝ) : initial_amount = 4000 → 
  let first_payment := initial_amount / 2
  let second_payment := first_payment + (2/5 * first_payment)
  let third_payment := 2 * (first_payment + second_payment)
  initial_amount + first_payment + second_payment + third_payment = 18400 := by
sorry

end NUMINAMATH_CALUDE_baylor_freelance_earnings_l583_58349


namespace NUMINAMATH_CALUDE_dollars_to_dozen_quarters_l583_58379

theorem dollars_to_dozen_quarters (dollars : ℕ) (quarters_per_dollar : ℕ) (items_per_dozen : ℕ) :
  dollars = 9 →
  quarters_per_dollar = 4 →
  items_per_dozen = 12 →
  (dollars * quarters_per_dollar) / items_per_dozen = 3 :=
by sorry

end NUMINAMATH_CALUDE_dollars_to_dozen_quarters_l583_58379


namespace NUMINAMATH_CALUDE_congruence_solution_unique_solution_in_range_l583_58345

theorem congruence_solution (m : ℤ) : 
  (13 * m ≡ 9 [ZMOD 47]) ↔ (m ≡ 26 [ZMOD 47]) :=
by sorry

theorem unique_solution_in_range : 
  ∃! x : ℕ, x < 47 ∧ (13 * x ≡ 9 [ZMOD 47]) :=
by sorry

end NUMINAMATH_CALUDE_congruence_solution_unique_solution_in_range_l583_58345


namespace NUMINAMATH_CALUDE_remainder_theorem_l583_58352

theorem remainder_theorem (x y u v : ℕ) (h1 : y > 0) (h2 : x = u * y + v) (h3 : v < y) (h4 : u + v < y) :
  (x + 3 * u * y + u) % y = u + v :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l583_58352


namespace NUMINAMATH_CALUDE_sufficient_not_imply_necessary_l583_58328

-- Define propositions A and B
variable (A B : Prop)

-- Define sufficient condition
def sufficient (B A : Prop) : Prop := B → A

-- Define necessary condition
def necessary (B A : Prop) : Prop := A → B

-- Theorem: B being sufficient for A does not imply B is necessary for A
theorem sufficient_not_imply_necessary :
  sufficient B A → ¬(necessary B A → sufficient B A) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_imply_necessary_l583_58328


namespace NUMINAMATH_CALUDE_liters_to_gallons_conversion_l583_58307

/-- Conversion factor from liters to gallons -/
def liters_to_gallons : ℝ := 0.26

/-- The volume in liters -/
def volume_in_liters : ℝ := 2.5

/-- Theorem stating that 2.5 liters is equal to 0.65 gallons -/
theorem liters_to_gallons_conversion :
  volume_in_liters * liters_to_gallons = 0.65 := by
  sorry

end NUMINAMATH_CALUDE_liters_to_gallons_conversion_l583_58307


namespace NUMINAMATH_CALUDE_inscribed_square_area_l583_58336

/-- The parabola function -/
def parabola (x : ℝ) : ℝ := x^2 - 8*x + 16

/-- A point on the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A square inscribed in the region bound by the parabola and x-axis -/
structure InscribedSquare where
  E : Point
  F : Point
  G : Point
  H : Point
  -- E and F are on x-axis
  h1 : E.y = 0
  h2 : F.y = 0
  -- G is on the parabola
  h3 : G.y = parabola G.x
  -- EFGH forms a square
  h4 : (F.x - E.x)^2 + (G.y - F.y)^2 = (G.x - F.x)^2 + (G.y - F.y)^2

/-- The theorem stating that the area of the inscribed square is 16 -/
theorem inscribed_square_area (s : InscribedSquare) : (s.F.x - s.E.x)^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l583_58336


namespace NUMINAMATH_CALUDE_triangle_problem_l583_58343

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

/-- The main theorem -/
theorem triangle_problem (t : Triangle) 
  (h1 : Real.sqrt 3 * t.b * Real.sin t.A = t.a * Real.cos t.B)
  (h2 : t.b = 3)
  (h3 : Real.sin t.C = Real.sqrt 3 * Real.sin t.A) : 
  t.B = π / 6 ∧ t.a = 3 ∧ t.c = 3 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l583_58343


namespace NUMINAMATH_CALUDE_square_of_1003_l583_58369

theorem square_of_1003 : (1003 : ℕ)^2 = 1006009 := by
  sorry

end NUMINAMATH_CALUDE_square_of_1003_l583_58369


namespace NUMINAMATH_CALUDE_coin_in_corner_l583_58382

/-- Represents a 2×n rectangle with coins --/
structure Rectangle (n : ℕ) where
  coins : Fin 2 → Fin n → ℕ

/-- Represents an operation of moving coins --/
inductive Operation
  | MoveRight : Fin 2 → Fin n → Operation
  | MoveUp : Fin 2 → Fin n → Operation

/-- Applies an operation to a rectangle --/
def applyOperation (rect : Rectangle n) (op : Operation) : Rectangle n :=
  sorry

/-- Checks if a sequence of operations results in a coin in (1,n) --/
def validSequence (rect : Rectangle n) (ops : List Operation) : Prop :=
  sorry

/-- Main theorem: There exists a sequence of operations to put a coin in (1,n) --/
theorem coin_in_corner (n : ℕ) (rect : Rectangle n) : 
  ∃ (ops : List Operation), validSequence rect ops :=
sorry

end NUMINAMATH_CALUDE_coin_in_corner_l583_58382


namespace NUMINAMATH_CALUDE_kates_savings_l583_58312

/-- Kate's savings and purchases problem -/
theorem kates_savings (march april may june : ℕ) 
  (keyboard mouse headset video_game : ℕ) : 
  march = 27 → 
  april = 13 → 
  may = 28 → 
  june = 35 → 
  keyboard = 49 → 
  mouse = 5 → 
  headset = 15 → 
  video_game = 25 → 
  (march + april + may + june + 2 * april) - 
  (keyboard + mouse + headset + video_game) = 35 := by
  sorry

end NUMINAMATH_CALUDE_kates_savings_l583_58312


namespace NUMINAMATH_CALUDE_cottage_rental_cost_l583_58326

theorem cottage_rental_cost (cost_per_hour : ℝ) (rental_hours : ℝ) (num_friends : ℕ) :
  cost_per_hour = 5 →
  rental_hours = 8 →
  num_friends = 2 →
  (cost_per_hour * rental_hours) / num_friends = 20 := by
  sorry

end NUMINAMATH_CALUDE_cottage_rental_cost_l583_58326


namespace NUMINAMATH_CALUDE_cube_equation_solutions_l583_58396

theorem cube_equation_solutions :
  ∀ a b c : ℕ+,
    (a^3 - b^3 - c^3 = 3*a*b*c) ∧
    (a^2 = 2*(a + b + c)) →
    ((a = 4 ∧ b = 1 ∧ c = 3) ∨
     (a = 4 ∧ b = 2 ∧ c = 2) ∨
     (a = 4 ∧ b = 3 ∧ c = 1)) :=
by sorry

end NUMINAMATH_CALUDE_cube_equation_solutions_l583_58396


namespace NUMINAMATH_CALUDE_watch_angle_difference_l583_58319

/-- Represents the angle between the hour and minute hands of a watch -/
def watchAngle (hours minutes : ℝ) : ℝ :=
  |30 * hours - 5.5 * minutes|

/-- Theorem stating that the time difference between two 120° angles of watch hands between 7:00 PM and 8:00 PM is 30 minutes -/
theorem watch_angle_difference : ∃ (t₁ t₂ : ℝ),
  0 < t₁ ∧ t₁ < t₂ ∧ t₂ < 60 ∧
  watchAngle (7 + t₁ / 60) t₁ = 120 ∧
  watchAngle (7 + t₂ / 60) t₂ = 120 ∧
  t₂ - t₁ = 30 := by
  sorry

end NUMINAMATH_CALUDE_watch_angle_difference_l583_58319


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l583_58362

-- Define the polynomial
def poly (x : ℝ) : ℝ := x^3 - 15*x^2 + 50*x - 60

-- Define the theorem
theorem root_sum_reciprocal (p q r A B C : ℝ) :
  (p ≠ q ∧ q ≠ r ∧ p ≠ r) →  -- p, q, r are distinct
  (poly p = 0 ∧ poly q = 0 ∧ poly r = 0) →  -- p, q, r are roots of poly
  (∀ s, s ≠ p ∧ s ≠ q ∧ s ≠ r →
    1 / (s^3 - 15*s^2 + 50*s - 60) = A / (s - p) + B / (s - q) + C / (s - r)) →
  1 / A + 1 / B + 1 / C = 135 :=
by sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l583_58362


namespace NUMINAMATH_CALUDE_inequality_solution_set_l583_58354

theorem inequality_solution_set :
  ∀ x : ℝ, (7 / 30 + |x - 7 / 60| < 11 / 20) ↔ (-1 / 5 < x ∧ x < 13 / 30) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l583_58354


namespace NUMINAMATH_CALUDE_sqrt_representation_l583_58389

theorem sqrt_representation (n : ℕ+) :
  (∃ (x : ℝ), x > 0 ∧ x^2 = n ∧ x = Real.sqrt (Real.sqrt n)) ↔ n = 1 ∧
  (∀ (x : ℝ), x > 0 ∧ x^2 = n → ∃ (m k : ℕ+), x = (k : ℝ) ^ (1 / m : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_representation_l583_58389


namespace NUMINAMATH_CALUDE_root_product_equation_l583_58306

theorem root_product_equation (m p q : ℝ) (a b : ℝ) : 
  (a^2 - m*a + 6 = 0) → 
  (b^2 - m*b + 6 = 0) → 
  ((a + 1/b)^2 - p*(a + 1/b) + q = 0) → 
  ((b + 1/a)^2 - p*(b + 1/a) + q = 0) → 
  q = 49/6 := by
sorry

end NUMINAMATH_CALUDE_root_product_equation_l583_58306


namespace NUMINAMATH_CALUDE_parabola_translation_l583_58314

/-- Represents a parabola in the form y = ax^2 + bx + c --/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically --/
def translate (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
    b := -2 * p.a * h + p.b
    c := p.a * h^2 - p.b * h + p.c + v }

theorem parabola_translation :
  let original := Parabola.mk 1 0 0  -- y = x^2
  let translated := translate original 2 1
  translated = Parabola.mk 1 (-4) 5  -- y = (x-2)^2 + 1
  := by sorry

end NUMINAMATH_CALUDE_parabola_translation_l583_58314


namespace NUMINAMATH_CALUDE_actual_average_height_l583_58300

-- Define the problem parameters
def totalStudents : ℕ := 50
def initialAverage : ℚ := 175
def incorrectHeights : List ℚ := [162, 150, 155]
def actualHeights : List ℚ := [142, 135, 145]

-- Define the theorem
theorem actual_average_height :
  let totalInitialHeight : ℚ := initialAverage * totalStudents
  let heightDifference : ℚ := (List.sum incorrectHeights) - (List.sum actualHeights)
  let correctedTotalHeight : ℚ := totalInitialHeight - heightDifference
  let actualAverage : ℚ := correctedTotalHeight / totalStudents
  actualAverage = 174.1 := by
  sorry

end NUMINAMATH_CALUDE_actual_average_height_l583_58300


namespace NUMINAMATH_CALUDE_max_profit_at_optimal_price_l583_58351

/-- Represents the product pricing and sales model -/
structure ProductModel where
  initial_price : ℝ
  initial_sales : ℝ
  price_demand_slope : ℝ
  cost_price : ℝ

/-- Calculates the profit function for a given price decrease -/
def profit_function (model : ProductModel) (x : ℝ) : ℝ :=
  let new_price := model.initial_price - x
  let new_sales := model.initial_sales + model.price_demand_slope * x
  (new_price - model.cost_price) * new_sales

/-- Theorem stating the maximum profit and optimal price decrease -/
theorem max_profit_at_optimal_price (model : ProductModel) 
  (h_initial_price : model.initial_price = 60)
  (h_initial_sales : model.initial_sales = 300)
  (h_price_demand_slope : model.price_demand_slope = 30)
  (h_cost_price : model.cost_price = 40) :
  ∃ (x : ℝ), 
    x = 5 ∧ 
    profit_function model x = 6750 ∧ 
    ∀ (y : ℝ), profit_function model y ≤ profit_function model x :=
  sorry

#eval profit_function ⟨60, 300, 30, 40⟩ 5

end NUMINAMATH_CALUDE_max_profit_at_optimal_price_l583_58351


namespace NUMINAMATH_CALUDE_rabbit_cleaner_amount_l583_58353

/-- The amount of cleaner used for a dog stain in ounces -/
def dog_cleaner : ℝ := 6

/-- The amount of cleaner used for a cat stain in ounces -/
def cat_cleaner : ℝ := 4

/-- The total amount of cleaner used in ounces -/
def total_cleaner : ℝ := 49

/-- The number of dogs -/
def num_dogs : ℕ := 6

/-- The number of cats -/
def num_cats : ℕ := 3

/-- The number of rabbits -/
def num_rabbits : ℕ := 1

/-- The amount of cleaner used for a rabbit stain in ounces -/
def rabbit_cleaner : ℝ := 1

theorem rabbit_cleaner_amount :
  dog_cleaner * num_dogs + cat_cleaner * num_cats + rabbit_cleaner * num_rabbits = total_cleaner := by
  sorry

end NUMINAMATH_CALUDE_rabbit_cleaner_amount_l583_58353


namespace NUMINAMATH_CALUDE_complex_number_simplification_l583_58315

theorem complex_number_simplification :
  (-5 - 3 * Complex.I) * 2 - (2 + 5 * Complex.I) = -12 - 11 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_simplification_l583_58315


namespace NUMINAMATH_CALUDE_regular_decagon_diagonal_intersections_eq_choose_l583_58368

/-- The number of distinct interior intersection points of diagonals in a regular decagon -/
def regular_decagon_diagonal_intersections : ℕ := 210

/-- A regular decagon has 10 sides -/
def regular_decagon_sides : ℕ := 10

/-- Theorem: The number of distinct interior intersection points of diagonals 
    in a regular decagon is equal to the number of ways to choose 4 vertices from 10 -/
theorem regular_decagon_diagonal_intersections_eq_choose :
  regular_decagon_diagonal_intersections = Nat.choose regular_decagon_sides 4 := by
  sorry

#eval regular_decagon_diagonal_intersections
#eval Nat.choose regular_decagon_sides 4

end NUMINAMATH_CALUDE_regular_decagon_diagonal_intersections_eq_choose_l583_58368


namespace NUMINAMATH_CALUDE_complex_root_quadratic_equation_l583_58377

theorem complex_root_quadratic_equation (q : ℝ) : 
  (2 * (Complex.mk (-3) 2)^2 + 12 * Complex.mk (-3) 2 + q = 0) → q = 26 := by
  sorry

end NUMINAMATH_CALUDE_complex_root_quadratic_equation_l583_58377


namespace NUMINAMATH_CALUDE_pinky_bought_36_apples_l583_58347

/-- The number of apples Danny the Duck bought -/
def danny_apples : ℕ := 73

/-- The total number of apples Pinky the Pig and Danny the Duck have -/
def total_apples : ℕ := 109

/-- The number of apples Pinky the Pig bought -/
def pinky_apples : ℕ := total_apples - danny_apples

theorem pinky_bought_36_apples : pinky_apples = 36 := by
  sorry

end NUMINAMATH_CALUDE_pinky_bought_36_apples_l583_58347


namespace NUMINAMATH_CALUDE_prism_height_l583_58386

/-- Represents a triangular prism with a regular triangular base -/
structure TriangularPrism where
  baseSideLength : ℝ
  totalEdgeLength : ℝ
  height : ℝ

/-- Theorem: The height of a specific triangular prism -/
theorem prism_height (p : TriangularPrism) 
  (h1 : p.baseSideLength = 10)
  (h2 : p.totalEdgeLength = 84) :
  p.height = 8 := by
  sorry

end NUMINAMATH_CALUDE_prism_height_l583_58386


namespace NUMINAMATH_CALUDE_sum_of_binary_numbers_l583_58341

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The first binary number 101₂ -/
def binary1 : List Bool := [true, false, true]

/-- The second binary number 110₂ -/
def binary2 : List Bool := [false, true, true]

theorem sum_of_binary_numbers :
  binary_to_decimal binary1 + binary_to_decimal binary2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_binary_numbers_l583_58341


namespace NUMINAMATH_CALUDE_olympic_medal_awards_l583_58357

/-- The number of ways to award medals in the Olympic 100-meter finals -/
def medal_award_ways (total_sprinters : ℕ) (canadian_sprinters : ℕ) (medals : ℕ) : ℕ :=
  let non_canadian_sprinters := total_sprinters - canadian_sprinters
  let no_canadian_medal := non_canadian_sprinters * (non_canadian_sprinters - 1) * (non_canadian_sprinters - 2)
  let one_canadian_medal := canadian_sprinters * medals * (non_canadian_sprinters) * (non_canadian_sprinters - 1)
  no_canadian_medal + one_canadian_medal

/-- Theorem: The number of ways to award medals in the given scenario is 480 -/
theorem olympic_medal_awards : medal_award_ways 10 4 3 = 480 := by
  sorry

end NUMINAMATH_CALUDE_olympic_medal_awards_l583_58357


namespace NUMINAMATH_CALUDE_regular_ngon_inscribed_circle_l583_58366

theorem regular_ngon_inscribed_circle (n : ℕ) (R : ℝ) (h : R > 0) :
  (n : ℝ) / 2 * R^2 * Real.sin (2 * Real.pi / n) = 3 * R^2 → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_regular_ngon_inscribed_circle_l583_58366


namespace NUMINAMATH_CALUDE_triangle_probability_l583_58325

def segment_lengths : List ℝ := [1, 3, 5, 7, 9]

def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def valid_triangle_count : ℕ := 3

def total_combinations : ℕ := 10

theorem triangle_probability : 
  (valid_triangle_count : ℚ) / (total_combinations : ℚ) = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_triangle_probability_l583_58325


namespace NUMINAMATH_CALUDE_absolute_value_and_sqrt_simplification_l583_58391

theorem absolute_value_and_sqrt_simplification :
  |-Real.sqrt 3| + Real.sqrt 12 + Real.sqrt 3 * (Real.sqrt 3 - 3) = 3 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_and_sqrt_simplification_l583_58391


namespace NUMINAMATH_CALUDE_divisibility_property_l583_58397

theorem divisibility_property (a b : ℤ) (h : 0 ≤ b ∧ b ≤ 9) :
  (∃ k : ℤ, 10 * a + b = k * 323) →
  ∃ m : ℤ, (2 * (a + b))^2 - a^2 = m * 323 := by
sorry

end NUMINAMATH_CALUDE_divisibility_property_l583_58397


namespace NUMINAMATH_CALUDE_parallel_lines_m_values_l583_58324

/-- Two lines in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem parallel_lines_m_values :
  ∀ m : ℝ,
  let l1 : Line := { a := 3, b := m, c := -1 }
  let l2 : Line := { a := m + 2, b := -(m - 2), c := 2 }
  parallel l1 l2 → m = 1 ∨ m = -6 := by
    sorry

end NUMINAMATH_CALUDE_parallel_lines_m_values_l583_58324


namespace NUMINAMATH_CALUDE_certain_number_proof_l583_58303

theorem certain_number_proof (p q x : ℝ) 
  (h1 : 3 / p = x)
  (h2 : 3 / q = 15)
  (h3 : p - q = 0.3) :
  x = 6 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l583_58303


namespace NUMINAMATH_CALUDE_certain_number_is_sixty_l583_58321

theorem certain_number_is_sixty : 
  ∃ x : ℝ, (10 + 20 + x) / 3 = (10 + 40 + 25) / 3 + 5 → x = 60 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_is_sixty_l583_58321


namespace NUMINAMATH_CALUDE_fibonacci_last_four_zeros_exist_l583_58395

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fibonacci n + fibonacci (n + 1)

def last_four_digits (n : ℕ) : ℕ := n % 10000

theorem fibonacci_last_four_zeros_exist :
  ∃ n, n < 100000001 ∧ last_four_digits (fibonacci n) = 0 := by
sorry

end NUMINAMATH_CALUDE_fibonacci_last_four_zeros_exist_l583_58395


namespace NUMINAMATH_CALUDE_simplify_fraction_1_simplify_fraction_2_l583_58342

-- Problem 1
theorem simplify_fraction_1 (a : ℝ) (h : a ≠ 1) : 
  1 / (a - 1) - a + 1 = (2*a - a^2) / (a - 1) := by sorry

-- Problem 2
theorem simplify_fraction_2 (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 2) :
  ((x + 2) / (x^2 - 2*x) - 1 / (x - 2)) / (2 / x) = 1 / (x - 2) := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_1_simplify_fraction_2_l583_58342


namespace NUMINAMATH_CALUDE_problem_solution_l583_58372

theorem problem_solution :
  (2017^2 - 2016 * 2018 = 1) ∧
  (∀ a b : ℤ, a + b = 7 → a * b = -1 → 
    ((a + b)^2 = 49) ∧ (a^2 - 3*a*b + b^2 = 54)) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l583_58372


namespace NUMINAMATH_CALUDE_min_value_expression_l583_58313

theorem min_value_expression (x : ℝ) (h : x > 4) :
  (x + 18) / Real.sqrt (x - 4) ≥ 2 * Real.sqrt 22 ∧
  ∃ x₀ > 4, (x₀ + 18) / Real.sqrt (x₀ - 4) = 2 * Real.sqrt 22 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l583_58313
