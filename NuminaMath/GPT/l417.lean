import Mathlib

namespace NUMINAMATH_GPT_cells_after_3_hours_l417_41783

noncomputable def cell_division_problem (t : ℕ) : ℕ :=
  2 ^ (t * 2)

theorem cells_after_3_hours : cell_division_problem 3 = 64 := by
  sorry

end NUMINAMATH_GPT_cells_after_3_hours_l417_41783


namespace NUMINAMATH_GPT_find_p_l417_41735

/-- Given conditions about the coordinates of points on a line, we want to prove p = 3. -/
theorem find_p (m n p : ℝ) 
  (h1 : m = n / 3 - 2 / 5)
  (h2 : m + p = (n + 9) / 3 - 2 / 5) 
  : p = 3 := by 
  sorry

end NUMINAMATH_GPT_find_p_l417_41735


namespace NUMINAMATH_GPT_op_4_6_l417_41749

-- Define the operation @ in Lean
def op (a b : ℕ) : ℤ := 2 * (a : ℤ)^2 - 2 * (b : ℤ)^2

-- State the theorem to prove
theorem op_4_6 : op 4 6 = -40 :=
by sorry

end NUMINAMATH_GPT_op_4_6_l417_41749


namespace NUMINAMATH_GPT_monic_polynomial_roots_l417_41777

theorem monic_polynomial_roots (r1 r2 r3 : ℝ) (h : ∀ x : ℝ, x^3 - 4*x^2 + 5 = 0 ↔ x = r1 ∨ x = r2 ∨ x = r3) :
  ∀ x : ℝ, x^3 - 12*x^2 + 135 = 0 ↔ x = 3*r1 ∨ x = 3*r2 ∨ x = 3*r3 :=
by
  sorry

end NUMINAMATH_GPT_monic_polynomial_roots_l417_41777


namespace NUMINAMATH_GPT_regular_train_passes_by_in_4_seconds_l417_41744

theorem regular_train_passes_by_in_4_seconds
    (l_high_speed : ℕ)
    (l_regular : ℕ)
    (t_observed : ℕ)
    (v_relative : ℕ)
    (h_length_high_speed : l_high_speed = 80)
    (h_length_regular : l_regular = 100)
    (h_time_observed : t_observed = 5)
    (h_velocity : v_relative = l_regular / t_observed) :
    v_relative * 4 = l_high_speed :=
by
  sorry

end NUMINAMATH_GPT_regular_train_passes_by_in_4_seconds_l417_41744


namespace NUMINAMATH_GPT_constant_function_of_zero_derivative_l417_41791

theorem constant_function_of_zero_derivative
  {f : ℝ → ℝ}
  (h : ∀ x : ℝ, deriv f x = 0) :
  ∃ c : ℝ, ∀ x : ℝ, f x = c :=
sorry

end NUMINAMATH_GPT_constant_function_of_zero_derivative_l417_41791


namespace NUMINAMATH_GPT_trapezoid_area_l417_41733

-- Definitions based on the given conditions
variable (BD AC h : ℝ)
variable (BD_perpendicular_AC : BD * AC = 0)
variable (BD_val : BD = 13)
variable (h_val : h = 12)

-- Statement of the theorem to prove the area of the trapezoid
theorem trapezoid_area (BD AC h : ℝ)
  (BD_perpendicular_AC : BD * AC = 0)
  (BD_val : BD = 13)
  (h_val : h = 12) :
  0.5 * 13 * 12 = 1014 / 5 := sorry

end NUMINAMATH_GPT_trapezoid_area_l417_41733


namespace NUMINAMATH_GPT_no_parallelogram_on_convex_graph_l417_41780

-- Definition of strictly convex function
def is_strictly_convex (f : ℝ → ℝ) : Prop :=
  ∀ ⦃x t y : ℝ⦄, (x < t ∧ t < y) → f t < ((f y - f x) / (y - x)) * (t - x) + f x

-- The main statement of the problem
theorem no_parallelogram_on_convex_graph (f : ℝ → ℝ) :
  is_strictly_convex f →
  ¬ ∃ (a b c d : ℝ), a < b ∧ b < c ∧ c < d ∧
    (f b < (f c - f a) / (c - a) * (b - a) + f a) ∧
    (f c < (f d - f b) / (d - b) * (c - b) + f b) :=
sorry

end NUMINAMATH_GPT_no_parallelogram_on_convex_graph_l417_41780


namespace NUMINAMATH_GPT_find_particular_number_l417_41767

theorem find_particular_number (x : ℤ) (h : x - 29 + 64 = 76) : x = 41 :=
by
  sorry

end NUMINAMATH_GPT_find_particular_number_l417_41767


namespace NUMINAMATH_GPT_xyz_value_l417_41702

theorem xyz_value (x y z : ℕ) (h1 : x + 2 * y = z) (h2 : x^2 - 4 * y^2 + z^2 = 310) :
  xyz = 4030 ∨ xyz = 23870 :=
by
  -- placeholder for proof steps
  sorry

end NUMINAMATH_GPT_xyz_value_l417_41702


namespace NUMINAMATH_GPT_hunting_dogs_theorem_l417_41710

noncomputable def hunting_dogs_problem : Prop :=
  ∃ (courtiers : Finset (Finset (Fin 100))) (h1 : courtiers.card = 100),
  ∀ (c1 c2 : Finset (Fin 100)), c1 ∈ courtiers → c2 ∈ courtiers → c1 ≠ c2 → (c1 ∩ c2).card ≥ 2 → 
  ∃ (c₁ c₂ : Finset (Fin 100)), c₁ ∈ courtiers ∧ c₂ ∈ courtiers ∧ c₁ = c₂

theorem hunting_dogs_theorem : hunting_dogs_problem :=
sorry

end NUMINAMATH_GPT_hunting_dogs_theorem_l417_41710


namespace NUMINAMATH_GPT_billy_horses_l417_41775

theorem billy_horses (each_horse_oats_per_meal : ℕ) (meals_per_day : ℕ) (total_oats_needed : ℕ) (days : ℕ) 
    (h_each_horse_oats_per_meal : each_horse_oats_per_meal = 4)
    (h_meals_per_day : meals_per_day = 2)
    (h_total_oats_needed : total_oats_needed = 96)
    (h_days : days = 3) :
    (total_oats_needed / (days * (each_horse_oats_per_meal * meals_per_day)) = 4) :=
by
  sorry

end NUMINAMATH_GPT_billy_horses_l417_41775


namespace NUMINAMATH_GPT_simplify_expression_l417_41724

theorem simplify_expression (x y : ℝ) : 
  (5 * x ^ 2 - 3 * x + 2) * (107 - 107) + (7 * y ^ 2 + 4 * y - 1) * (93 - 93) = 0 := 
by 
  sorry

end NUMINAMATH_GPT_simplify_expression_l417_41724


namespace NUMINAMATH_GPT_hyperbola_asymptote_solution_l417_41799

theorem hyperbola_asymptote_solution (b : ℝ) (hb : b > 0)
  (h_asym : ∀ x y, (∀ y : ℝ, y = (1 / 2) * x ∨ y = - (1 / 2) * x) → (x^2 / 4 - y^2 / b^2 = 1)) :
  b = 1 :=
sorry

end NUMINAMATH_GPT_hyperbola_asymptote_solution_l417_41799


namespace NUMINAMATH_GPT_height_of_given_cylinder_l417_41766

noncomputable def height_of_cylinder (P d : ℝ) : ℝ :=
  let r := P / (2 * Real.pi)
  let l := P
  let h := Real.sqrt (d^2 - l^2)
  h

theorem height_of_given_cylinder : height_of_cylinder 6 10 = 8 :=
by
  show height_of_cylinder 6 10 = 8
  sorry

end NUMINAMATH_GPT_height_of_given_cylinder_l417_41766


namespace NUMINAMATH_GPT_solve_for_x_l417_41793

theorem solve_for_x (x y : ℝ) (h₁ : x - y = 8) (h₂ : x + y = 16) (h₃ : x * y = 48) : x = 12 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l417_41793


namespace NUMINAMATH_GPT_shobha_current_age_l417_41751

variable (S B : ℕ)
variable (h_ratio : 4 * B = 3 * S)
variable (h_future_age : S + 6 = 26)

theorem shobha_current_age : B = 15 :=
by
  sorry

end NUMINAMATH_GPT_shobha_current_age_l417_41751


namespace NUMINAMATH_GPT_fixed_chord_property_l417_41739

theorem fixed_chord_property (d : ℝ) (h₁ : d = 3 / 2) :
  ∀ (x1 x2 m : ℝ) (h₀ : x1 + x2 = m) (h₂ : x1 * x2 = 1 - d),
    ((1 / ((x1 ^ 2) + (m * x1) ^ 2)) + (1 / ((x2 ^ 2) + (m * x2) ^ 2))) = 4 / 9 :=
by
  sorry

end NUMINAMATH_GPT_fixed_chord_property_l417_41739


namespace NUMINAMATH_GPT_x_squared_plus_y_squared_l417_41758

theorem x_squared_plus_y_squared (x y : ℝ) 
  (h1 : (1/x) + (1/y) = 5) 
  (h2 : x * y + x + y = 11) : 
  x^2 + y^2 = 2893 / 36 := 
by 
  sorry

end NUMINAMATH_GPT_x_squared_plus_y_squared_l417_41758


namespace NUMINAMATH_GPT_amy_money_left_l417_41718

def amount_left (initial_amount doll_price board_game_price comic_book_price doll_qty board_game_qty comic_book_qty board_game_discount sales_tax_rate : ℝ) :
    ℝ :=
  let cost_dolls := doll_qty * doll_price
  let cost_board_games := board_game_qty * board_game_price
  let cost_comic_books := comic_book_qty * comic_book_price
  let discounted_cost_board_games := cost_board_games * (1 - board_game_discount)
  let total_cost_before_tax := cost_dolls + discounted_cost_board_games + cost_comic_books
  let sales_tax := total_cost_before_tax * sales_tax_rate
  let total_cost_after_tax := total_cost_before_tax + sales_tax
  initial_amount - total_cost_after_tax

theorem amy_money_left :
  amount_left 100 1.25 12.75 3.50 3 2 4 0.10 0.08 = 56.04 :=
by
  sorry

end NUMINAMATH_GPT_amy_money_left_l417_41718


namespace NUMINAMATH_GPT_tile_difference_l417_41737

theorem tile_difference :
  let initial_blue_tiles := 20
  let initial_green_tiles := 15
  let first_border_tiles := 18
  let second_border_tiles := 18
  let total_green_tiles := initial_green_tiles + first_border_tiles + second_border_tiles
  let total_blue_tiles := initial_blue_tiles
  total_green_tiles - total_blue_tiles = 31 := 
by
  sorry

end NUMINAMATH_GPT_tile_difference_l417_41737


namespace NUMINAMATH_GPT_consecutive_lucky_years_l417_41779

def is_lucky (Y : ℕ) : Prop := 
  let first_two_digits := Y / 100
  let last_two_digits := Y % 100
  Y % (first_two_digits + last_two_digits) = 0

theorem consecutive_lucky_years : ∃ Y : ℕ, is_lucky Y ∧ is_lucky (Y + 1) :=
by
  sorry

end NUMINAMATH_GPT_consecutive_lucky_years_l417_41779


namespace NUMINAMATH_GPT_triangle_evaluation_l417_41734

def triangle (a b : ℤ) : ℤ := a^2 - 2 * b

theorem triangle_evaluation : triangle (-2) (triangle 3 2) = -6 := by
  sorry

end NUMINAMATH_GPT_triangle_evaluation_l417_41734


namespace NUMINAMATH_GPT_johns_final_weight_is_200_l417_41797

-- Define the initial weight, percentage of weight loss, and weight gain
def initial_weight : ℝ := 220
def weight_loss_percentage : ℝ := 0.10
def weight_gain : ℝ := 2

-- Define a function to calculate the final weight
def final_weight (initial_weight : ℝ) (weight_loss_percentage : ℝ) (weight_gain : ℝ) : ℝ := 
  let weight_lost := initial_weight * weight_loss_percentage
  let weight_after_loss := initial_weight - weight_lost
  weight_after_loss + weight_gain

-- The proof problem is to show that the final weight is 200 pounds
theorem johns_final_weight_is_200 :
  final_weight initial_weight weight_loss_percentage weight_gain = 200 := 
by
  sorry

end NUMINAMATH_GPT_johns_final_weight_is_200_l417_41797


namespace NUMINAMATH_GPT_alpha_plus_beta_l417_41703

theorem alpha_plus_beta :
  (∃ α β : ℝ, 
    (∀ x : ℝ, x ≠ -β ∧ x ≠ 45 → (x - α) / (x + β) = (x^2 - 90 * x + 1980) / (x^2 + 70 * x - 3570))
  ) → (∃ α β : ℝ, α + β = 123) :=
by {
  sorry
}

end NUMINAMATH_GPT_alpha_plus_beta_l417_41703


namespace NUMINAMATH_GPT_unique_solution_7x_eq_3y_plus_4_l417_41788

theorem unique_solution_7x_eq_3y_plus_4 (x y : ℕ) (hx : 1 ≤ x) (hy : 1 ≤ y) :
    7^x = 3^y + 4 ↔ (x = 1 ∧ y = 1) :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_7x_eq_3y_plus_4_l417_41788


namespace NUMINAMATH_GPT_minimum_value_l417_41729

theorem minimum_value (x : ℝ) (hx : 0 < x) : ∃ y, (y = x + 4 / (x + 1)) ∧ (∀ z, (x > 0 → z = x + 4 / (x + 1)) → 3 ≤ z) := sorry

end NUMINAMATH_GPT_minimum_value_l417_41729


namespace NUMINAMATH_GPT_least_pos_int_for_multiple_of_5_l417_41787

theorem least_pos_int_for_multiple_of_5 (n : ℕ) (h1 : n = 725) : ∃ x : ℕ, x > 0 ∧ (725 + x) % 5 = 0 ∧ x = 5 :=
by
  sorry

end NUMINAMATH_GPT_least_pos_int_for_multiple_of_5_l417_41787


namespace NUMINAMATH_GPT_noah_large_paintings_last_month_l417_41728

-- problem definitions
def large_painting_price : ℕ := 60
def small_painting_price : ℕ := 30
def small_paintings_sold_last_month : ℕ := 4
def sales_this_month : ℕ := 1200

-- to be proven
theorem noah_large_paintings_last_month (L : ℕ) (last_month_sales_eq : large_painting_price * L + small_painting_price * small_paintings_sold_last_month = S) 
   (this_month_sales_eq : 2 * S = sales_this_month) :
  L = 8 :=
sorry

end NUMINAMATH_GPT_noah_large_paintings_last_month_l417_41728


namespace NUMINAMATH_GPT_foundation_cost_l417_41782

theorem foundation_cost (volume_per_house : ℝ)
    (density : ℝ)
    (cost_per_pound : ℝ)
    (num_houses : ℕ) 
    (dimension_len : ℝ)
    (dimension_wid : ℝ)
    (dimension_height : ℝ)
    : cost_per_pound = 0.02 → density = 150 → dimension_len = 100 → dimension_wid = 100 → dimension_height = 0.5 → num_houses = 3
    → volume_per_house = dimension_len * dimension_wid * dimension_height 
    → (num_houses : ℝ) * (volume_per_house * density * cost_per_pound) = 45000 := 
by 
  sorry

end NUMINAMATH_GPT_foundation_cost_l417_41782


namespace NUMINAMATH_GPT_divides_expression_l417_41752

theorem divides_expression (y : ℕ) (hy : y ≠ 0) : (y - 1) ∣ (y^(y^2) - 2 * y^(y + 1) + 1) := 
by
  sorry

end NUMINAMATH_GPT_divides_expression_l417_41752


namespace NUMINAMATH_GPT_calculate_f_g2_l417_41711

def f (x : ℝ) : ℝ := x^2 + 1
def g (x : ℝ) : ℝ := 2 * x^3 - 1

theorem calculate_f_g2 : f (g 2) = 226 := by
  sorry

end NUMINAMATH_GPT_calculate_f_g2_l417_41711


namespace NUMINAMATH_GPT_find_n_l417_41796

theorem find_n
  (n : ℤ)
  (h : n + (n + 1) + (n + 2) + (n + 3) = 30) :
  n = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l417_41796


namespace NUMINAMATH_GPT_max_gold_coins_l417_41726

theorem max_gold_coins (k : ℕ) (n : ℕ) (h : n = 13 * k + 3 ∧ n < 150) : n = 146 :=
by 
  sorry

end NUMINAMATH_GPT_max_gold_coins_l417_41726


namespace NUMINAMATH_GPT_quadratic_real_roots_and_value_l417_41742

theorem quadratic_real_roots_and_value (m x1 x2: ℝ) 
  (h1: ∀ (a: ℝ), ∃ (b c: ℝ), a = x^2 - 4 * x - 2 * m + 5) 
  (h2: x1 * x2 + x1 + x2 = m^2 + 6):
  m ≥ 1/2 ∧ m = 1 := 
by
  sorry

end NUMINAMATH_GPT_quadratic_real_roots_and_value_l417_41742


namespace NUMINAMATH_GPT_cookies_per_child_l417_41781

theorem cookies_per_child (total_cookies : ℕ) (adults : ℕ) (children : ℕ) (fraction_eaten_by_adults : ℚ) 
  (h1 : total_cookies = 120) (h2 : adults = 2) (h3 : children = 4) (h4 : fraction_eaten_by_adults = 1/3) :
  total_cookies * (1 - fraction_eaten_by_adults) / children = 20 := 
by
  sorry

end NUMINAMATH_GPT_cookies_per_child_l417_41781


namespace NUMINAMATH_GPT_smallest_lcm_4_digit_integers_l417_41795

theorem smallest_lcm_4_digit_integers (k l : ℕ) (h1 : 1000 ≤ k ∧ k ≤ 9999) (h2 : 1000 ≤ l ∧ l ≤ 9999) (h3 : Nat.gcd k l = 11) : Nat.lcm k l = 92092 :=
by
  sorry

end NUMINAMATH_GPT_smallest_lcm_4_digit_integers_l417_41795


namespace NUMINAMATH_GPT_ted_speed_l417_41709

variables (T F : ℝ)

-- Ted runs two-thirds as fast as Frank
def condition1 : Prop := T = (2 / 3) * F

-- In two hours, Frank runs eight miles farther than Ted
def condition2 : Prop := 2 * F = 2 * T + 8

-- Prove that Ted runs at a speed of 8 mph
theorem ted_speed (h1 : condition1 T F) (h2 : condition2 T F) : T = 8 :=
by
  sorry

end NUMINAMATH_GPT_ted_speed_l417_41709


namespace NUMINAMATH_GPT_sequence_contains_composite_l417_41722

noncomputable def is_composite (n : ℕ) : Prop :=
  ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

theorem sequence_contains_composite (a : ℕ → ℕ) (h : ∀ n, a (n+1) = 2 * a n + 1 ∨ a (n+1) = 2 * a n - 1) :
  ∃ n, is_composite (a n) :=
sorry

end NUMINAMATH_GPT_sequence_contains_composite_l417_41722


namespace NUMINAMATH_GPT_simplify_and_evaluate_l417_41708

theorem simplify_and_evaluate (x y : ℝ) (hx : x = -1) (hy : y = -1/3) :
  ((3 * x^2 + x * y + 2 * y) - 2 * (5 * x * y - 4 * x^2 + y)) = 8 := by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l417_41708


namespace NUMINAMATH_GPT_relationship_above_l417_41754

noncomputable def a : ℝ := Real.log 5 / Real.log 2
noncomputable def b : ℝ := Real.log 15 / (2 * Real.log 2)
noncomputable def c : ℝ := Real.sqrt 2

theorem relationship_above (ha : a = Real.log 5 / Real.log 2) 
                           (hb : b = Real.log 15 / (2 * Real.log 2))
                           (hc : c = Real.sqrt 2) : a > b ∧ b > c :=
by
  sorry

end NUMINAMATH_GPT_relationship_above_l417_41754


namespace NUMINAMATH_GPT_find_number_l417_41792

theorem find_number (x : ℤ) (h : x + x^2 + 15 = 96) : x = -9 :=
sorry

end NUMINAMATH_GPT_find_number_l417_41792


namespace NUMINAMATH_GPT_greatest_integer_solution_l417_41732

theorem greatest_integer_solution (n : ℤ) (h : n^2 - 12 * n + 28 ≤ 0) : 6 ≤ n :=
sorry

end NUMINAMATH_GPT_greatest_integer_solution_l417_41732


namespace NUMINAMATH_GPT_xiaoming_correct_answers_l417_41731

theorem xiaoming_correct_answers (x : ℕ) (h1 : x ≤ 10) (h2 : 5 * x - (10 - x) > 30) : x ≥ 7 := 
by
  sorry

end NUMINAMATH_GPT_xiaoming_correct_answers_l417_41731


namespace NUMINAMATH_GPT_divisible_bc_ad_l417_41770

theorem divisible_bc_ad
  (a b c d u : ℤ)
  (h1 : u ∣ a * c)
  (h2 : u ∣ b * c + a * d)
  (h3 : u ∣ b * d) :
  u ∣ b * c ∧ u ∣ a * d :=
by
  sorry

end NUMINAMATH_GPT_divisible_bc_ad_l417_41770


namespace NUMINAMATH_GPT_max_x_squared_plus_y_squared_l417_41720

theorem max_x_squared_plus_y_squared (x y : ℝ) 
  (h : 3 * x^2 + 2 * y^2 = 2 * x) : x^2 + y^2 ≤ 4 / 9 :=
sorry

end NUMINAMATH_GPT_max_x_squared_plus_y_squared_l417_41720


namespace NUMINAMATH_GPT_combined_income_is_16800_l417_41761

-- Given conditions
def ErnieOldIncome : ℕ := 6000
def ErnieCurrentIncome : ℕ := (4 * ErnieOldIncome) / 5
def JackCurrentIncome : ℕ := 2 * ErnieOldIncome

-- Proof that their combined income is $16800
theorem combined_income_is_16800 : ErnieCurrentIncome + JackCurrentIncome = 16800 := by
  sorry

end NUMINAMATH_GPT_combined_income_is_16800_l417_41761


namespace NUMINAMATH_GPT_radius_of_sphere_l417_41784

theorem radius_of_sphere {r x : ℝ} (h1 : 15^2 + x^2 = r^2) (h2 : r = x + 12) :
    r = 123 / 8 :=
  by
  sorry

end NUMINAMATH_GPT_radius_of_sphere_l417_41784


namespace NUMINAMATH_GPT_translate_line_upwards_l417_41757

-- Define the original line equation
def original_line_eq (x : ℝ) : ℝ := 3 * x - 3

-- Define the translation operation
def translate_upwards (y_translation : ℝ) (line_eq : ℝ → ℝ) (x : ℝ) : ℝ :=
  line_eq x + y_translation

-- Define the proof problem
theorem translate_line_upwards :
  ∀ (x : ℝ), translate_upwards 5 original_line_eq x = 3 * x + 2 :=
by
  intros x
  simp [translate_upwards, original_line_eq]
  sorry

end NUMINAMATH_GPT_translate_line_upwards_l417_41757


namespace NUMINAMATH_GPT_sum_geometric_series_l417_41715

-- Given the conditions
def q : ℕ := 2
def a3 : ℕ := 16
def n : ℕ := 2017
def a1 : ℕ := 4

-- Define the sum of the first n terms of a geometric series
noncomputable def geometricSeriesSum (a1 q n : ℕ) : ℕ :=
  a1 * (1 - q^n) / (1 - q)

-- State the problem
theorem sum_geometric_series :
  geometricSeriesSum a1 q n = 2^2019 - 4 :=
sorry

end NUMINAMATH_GPT_sum_geometric_series_l417_41715


namespace NUMINAMATH_GPT_sin_expression_value_l417_41719

theorem sin_expression_value (α : ℝ) (h : Real.cos (α + π / 5) = 4 / 5) :
  Real.sin (2 * α + 9 * π / 10) = 7 / 25 :=
sorry

end NUMINAMATH_GPT_sin_expression_value_l417_41719


namespace NUMINAMATH_GPT_box_calories_l417_41785

theorem box_calories :
  let cookies_per_bag := 20
  let bags_per_box := 4
  let calories_per_cookie := 20
  (cookies_per_bag * bags_per_box) * calories_per_cookie = 1600 :=
by
  sorry

end NUMINAMATH_GPT_box_calories_l417_41785


namespace NUMINAMATH_GPT_vertical_angles_equal_l417_41769

-- Given: Definition for pairs of adjacent angles summing up to 180 degrees
def adjacent_add_to_straight_angle (α β : ℝ) : Prop := 
  α + β = 180

-- Given: Two intersecting lines forming angles
variables (α β γ δ : ℝ)

-- Given: Relationship of adjacent angles being supplementary
axiom adj1 : adjacent_add_to_straight_angle α β
axiom adj2 : adjacent_add_to_straight_angle β γ
axiom adj3 : adjacent_add_to_straight_angle γ δ
axiom adj4 : adjacent_add_to_straight_angle δ α

-- Question: Prove that vertical angles are equal
theorem vertical_angles_equal : α = γ :=
by sorry

end NUMINAMATH_GPT_vertical_angles_equal_l417_41769


namespace NUMINAMATH_GPT_simplify_fraction_l417_41700

theorem simplify_fraction (a : ℝ) (h : a ≠ 2) : 
  (a^2 / (a - 2) - (4 * a - 4) / (a - 2)) = a - 2 :=
  sorry

end NUMINAMATH_GPT_simplify_fraction_l417_41700


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l417_41771

-- Definition of the sequence
def a (n : ℕ) (k : ℚ) : ℚ := (k * n - 3) / (n - 3 / 2)

-- The first condition proof problem
theorem problem1 (k : ℚ) : (∀ n : ℕ, a n k = (a (n + 1) k + a (n - 1) k) / 2) → k = 2 :=
sorry

-- The second condition proof problem
theorem problem2 (k : ℚ) : 
  k ≠ 2 → 
  (if k > 2 then (a 1 k < k ∧ a 2 k = max (a 1 k) (a 2 k))
   else if k < 2 then (a 2 k < k ∧ a 1 k = max (a 1 k) (a 2 k))
   else False) :=
sorry

-- The third condition proof problem
theorem problem3 (k : ℚ) : 
  (∀ n : ℕ, n > 0 → a n k > (k * 2^n + (-1)^n) / 2^n) → 
  101 / 48 < k ∧ k < 13 / 6 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_l417_41771


namespace NUMINAMATH_GPT_depth_of_well_l417_41764

noncomputable def volume_of_cylinder (radius : ℝ) (depth : ℝ) : ℝ :=
  Real.pi * radius^2 * depth

theorem depth_of_well (volume depth : ℝ) (r : ℝ) : 
  r = 1 ∧ volume = 25.132741228718345 ∧ 2 * r = 2 → depth = 8 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_depth_of_well_l417_41764


namespace NUMINAMATH_GPT_general_term_formula_l417_41712

noncomputable def xSeq : ℕ → ℝ
| 0       => 3
| (n + 1) => (xSeq n)^2 + 2 / (2 * (xSeq n) - 1)

theorem general_term_formula (n : ℕ) : 
  xSeq n = (2 * 2^2^n + 1) / (2^2^n - 1) := 
sorry

end NUMINAMATH_GPT_general_term_formula_l417_41712


namespace NUMINAMATH_GPT_cost_of_first_variety_l417_41774

theorem cost_of_first_variety (x : ℝ) (cost2 : ℝ) (cost_mix : ℝ) (ratio : ℝ) :
    cost2 = 8.75 →
    cost_mix = 7.50 →
    ratio = 0.625 →
    (x - cost_mix) / (cost2 - cost_mix) = ratio →
    x = 8.28125 := 
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_cost_of_first_variety_l417_41774


namespace NUMINAMATH_GPT_additional_votes_in_revote_l417_41704

theorem additional_votes_in_revote (a b a' b' n : ℕ) :
  a + b = 300 →
  b - a = n →
  a' - b' = 3 * n →
  a' + b' = 300 →
  a' = (7 * b) / 6 →
  a' - a = 55 :=
by 
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_additional_votes_in_revote_l417_41704


namespace NUMINAMATH_GPT_emma_ate_more_than_liam_l417_41705

-- Definitions based on conditions
def emma_oranges : ℕ := 8
def liam_oranges : ℕ := 1

-- Lean statement to prove the question
theorem emma_ate_more_than_liam : emma_oranges - liam_oranges = 7 := by
  sorry

end NUMINAMATH_GPT_emma_ate_more_than_liam_l417_41705


namespace NUMINAMATH_GPT_sum_of_three_sqrt_139_l417_41765

theorem sum_of_three_sqrt_139 {x y z : ℝ} (h1 : x >= 0) (h2 : y >= 0) (h3 : z >= 0)
  (hx : x^2 + y^2 + z^2 = 75) (hy : x * y + y * z + z * x = 32) : x + y + z = Real.sqrt 139 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_three_sqrt_139_l417_41765


namespace NUMINAMATH_GPT_secret_reaches_2186_students_on_seventh_day_l417_41756

/-- 
Alice tells a secret to three friends on Sunday. The next day, each of those friends tells the secret to three new friends.
Each time a person hears the secret, they tell three other new friends the following day.
On what day will 2186 students know the secret?
-/
theorem secret_reaches_2186_students_on_seventh_day :
  ∃ (n : ℕ), 1 + 3 * ((3^n - 1)/2) = 2186 ∧ n = 7 :=
by
  sorry

end NUMINAMATH_GPT_secret_reaches_2186_students_on_seventh_day_l417_41756


namespace NUMINAMATH_GPT_train_journey_time_l417_41701

theorem train_journey_time :
  ∃ T : ℝ, (30 : ℝ) / 60 = (7 / 6 * T) - T ∧ T = 3 :=
by
  sorry

end NUMINAMATH_GPT_train_journey_time_l417_41701


namespace NUMINAMATH_GPT_smallest_integer_divisible_20_perfect_cube_square_l417_41741

theorem smallest_integer_divisible_20_perfect_cube_square :
  ∃ (n : ℕ), n > 0 ∧ n % 20 = 0 ∧ (∃ (m : ℕ), n^2 = m^3) ∧ (∃ (k : ℕ), n^3 = k^2) ∧ n = 1000000 :=
by {
  sorry -- Replace this placeholder with an appropriate proof.
}

end NUMINAMATH_GPT_smallest_integer_divisible_20_perfect_cube_square_l417_41741


namespace NUMINAMATH_GPT_find_sum_of_numbers_l417_41794

theorem find_sum_of_numbers 
  (a b : ℕ)
  (h₁ : a.gcd b = 5)
  (h₂ : a * b / a.gcd b = 120)
  (h₃ : (1 : ℚ) / a + 1 / b = 0.09166666666666666) :
  a + b = 55 := 
sorry

end NUMINAMATH_GPT_find_sum_of_numbers_l417_41794


namespace NUMINAMATH_GPT_unique_point_value_l417_41736

noncomputable def unique_point_condition : Prop :=
  ∀ (x y : ℝ), 3 * x^2 + y^2 + 6 * x - 6 * y + 12 = 0

theorem unique_point_value (d : ℝ) : unique_point_condition ↔ d = 12 := 
sorry

end NUMINAMATH_GPT_unique_point_value_l417_41736


namespace NUMINAMATH_GPT_determine_phi_l417_41743

theorem determine_phi (f : ℝ → ℝ) (φ : ℝ): 
  (∀ x : ℝ, f x = 2 * Real.sin (2 * x + 3 * φ)) ∧ 
  (∀ x : ℝ, f (-x) = -f x) → 
  (∃ k : ℤ, φ = k * Real.pi / 3) :=
by 
  sorry

end NUMINAMATH_GPT_determine_phi_l417_41743


namespace NUMINAMATH_GPT_problem_statement_l417_41776

def U : Set ℤ := {x | True}
def A : Set ℤ := {-1, 1, 3, 5, 7, 9}
def B : Set ℤ := {-1, 5, 7}
def complement (B : Set ℤ) : Set ℤ := {x | x ∉ B}

theorem problem_statement : (A ∩ (complement B)) = {1, 3, 9} :=
by {
  sorry
}

end NUMINAMATH_GPT_problem_statement_l417_41776


namespace NUMINAMATH_GPT_total_books_l417_41713

def keith_books : ℕ := 20
def jason_books : ℕ := 21

theorem total_books : keith_books + jason_books = 41 :=
by
  sorry

end NUMINAMATH_GPT_total_books_l417_41713


namespace NUMINAMATH_GPT_roshini_sweets_cost_correct_l417_41786

noncomputable def roshini_sweet_cost_before_discounts_and_tax : ℝ := 10.54

theorem roshini_sweets_cost_correct (R F1 F2 F3 : ℝ) (h1 : R + F1 + F2 + F3 = 10.54)
    (h2 : R * 0.9 = (10.50 - 9.20) / 1.08)
    (h3 : F1 + F2 + F3 = 3.40 + 4.30 + 1.50) :
    R + F1 + F2 + F3 = roshini_sweet_cost_before_discounts_and_tax :=
by
  sorry

end NUMINAMATH_GPT_roshini_sweets_cost_correct_l417_41786


namespace NUMINAMATH_GPT_simplify_and_sum_of_exponents_l417_41753

-- Define the given expression
def radicand (x y z : ℝ) : ℝ := 40 * x ^ 5 * y ^ 7 * z ^ 9

-- Define what cube root stands for
noncomputable def cbrt (a : ℝ) := a ^ (1 / 3 : ℝ)

-- Define the simplified expression outside the cube root
noncomputable def simplified_outside_exponents (x y z : ℝ) : ℝ := x * y * z ^ 3

-- Define the sum of the exponents outside the radical
def sum_of_exponents_outside (x y z : ℝ) : ℝ := (1 + 1 + 3 : ℝ)

-- Statement of the problem in Lean
theorem simplify_and_sum_of_exponents (x y z : ℝ) :
  sum_of_exponents_outside x y z = 5 :=
by 
  sorry

end NUMINAMATH_GPT_simplify_and_sum_of_exponents_l417_41753


namespace NUMINAMATH_GPT_find_omega_find_period_and_intervals_find_solution_set_l417_41747

noncomputable def omega_condition (ω : ℝ) :=
  0 < ω ∧ ω < 2

noncomputable def function_fx (ω : ℝ) (x : ℝ) := 
  3 * Real.sin (2 * ω * x + Real.pi / 3)

noncomputable def center_of_symmetry_condition (ω : ℝ) := 
  function_fx ω (-Real.pi / 6) = 0

noncomputable def period_condition (ω : ℝ) :=
  Real.pi / abs ω

noncomputable def intervals_of_increase (ω : ℝ) (x : ℝ) : Prop :=
  ∃ k : ℤ, ((Real.pi / 12 + k * Real.pi) ≤ x) ∧ (x < (5 * Real.pi / 12 + k * Real.pi))

noncomputable def solution_set_fx_ge_half (x : ℝ) : Prop :=
  ∃ k : ℤ, (Real.pi / 12 + k * Real.pi) ≤ x ∧ (x ≤ 5 * Real.pi / 12 + k * Real.pi)

theorem find_omega : ∀ ω : ℝ, omega_condition ω ∧ center_of_symmetry_condition ω → omega = 1 := sorry

theorem find_period_and_intervals : 
  ∀ ω : ℝ, omega_condition ω ∧ (ω = 1) → period_condition ω = Real.pi :=
sorry

theorem find_solution_set :
  ∀ ω : ℝ, omega_condition ω ∧ (ω = 1) → (∀ x, solution_set_fx_ge_half x) :=
sorry

end NUMINAMATH_GPT_find_omega_find_period_and_intervals_find_solution_set_l417_41747


namespace NUMINAMATH_GPT_intersection_sets_l417_41798

-- Define the sets A and B as given in the problem conditions
def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {0, 2, 4}

-- Lean theorem statement for proving the intersection of sets A and B is {0, 2}
theorem intersection_sets : A ∩ B = {0, 2} := 
by
  sorry

end NUMINAMATH_GPT_intersection_sets_l417_41798


namespace NUMINAMATH_GPT_arithmetic_sequence_middle_term_l417_41746

theorem arithmetic_sequence_middle_term :
  ∀ (a b : ℕ) (z : ℕ), a = 9 → b = 81 → z = (a + b) / 2 → z = 45 :=
by
  intros a b z h_a h_b h_z
  rw [h_a, h_b] at h_z
  exact h_z

end NUMINAMATH_GPT_arithmetic_sequence_middle_term_l417_41746


namespace NUMINAMATH_GPT_original_fraction_l417_41750

def fraction (a b c : ℕ) := 10 * a + b / 10 * c + a

theorem original_fraction (a b c : ℕ) (ha: a < 10) (hb : b < 10) (hc : c < 10) (h : b ≠ c):
  (fraction a b c = b / c) →
  (fraction 6 4 1 = 64 / 16) ∨ (fraction 9 8 4 = 98 / 49) ∨
  (fraction 9 5 1 = 95 / 19) ∨ (fraction 6 5 2 = 65 / 26) :=
sorry

end NUMINAMATH_GPT_original_fraction_l417_41750


namespace NUMINAMATH_GPT_min_value_expr_l417_41730

theorem min_value_expr : ∃ x : ℝ, (15 - x) * (9 - x) * (15 + x) * (9 + x) = -5184 :=
by
  sorry

end NUMINAMATH_GPT_min_value_expr_l417_41730


namespace NUMINAMATH_GPT_line_tangent_to_ellipse_l417_41714

theorem line_tangent_to_ellipse (m : ℝ) :
  (∀ x y : ℝ, y = m * x + 2 ∧ 3 * x^2 + 6 * y^2 = 6 → ∃! y : ℝ, 3 * x^2 + 6 * y^2 = 6) →
  m^2 = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_line_tangent_to_ellipse_l417_41714


namespace NUMINAMATH_GPT_arccos_zero_eq_pi_div_two_l417_41759

theorem arccos_zero_eq_pi_div_two : Real.arccos 0 = Real.pi / 2 :=
by
  sorry

end NUMINAMATH_GPT_arccos_zero_eq_pi_div_two_l417_41759


namespace NUMINAMATH_GPT_cube_sphere_surface_area_l417_41740

open Real

noncomputable def cube_edge_length := 1
noncomputable def cube_space_diagonal := sqrt 3
noncomputable def sphere_radius := cube_space_diagonal / 2
noncomputable def sphere_surface_area := 4 * π * (sphere_radius ^ 2)

theorem cube_sphere_surface_area :
  sphere_surface_area = 3 * π :=
by
  sorry

end NUMINAMATH_GPT_cube_sphere_surface_area_l417_41740


namespace NUMINAMATH_GPT_paperboy_delivery_count_l417_41707

def no_miss_four_consecutive (n : ℕ) (E : ℕ → ℕ) : Prop :=
  ∀ k > 3, E k = E (k - 1) + E (k - 2) + E (k - 3)

def base_conditions (E : ℕ → ℕ) : Prop :=
  E 1 = 2 ∧ E 2 = 4 ∧ E 3 = 8

theorem paperboy_delivery_count : ∃ (E : ℕ → ℕ), 
  base_conditions E ∧ no_miss_four_consecutive 12 E ∧ E 12 = 1854 :=
by
  sorry

end NUMINAMATH_GPT_paperboy_delivery_count_l417_41707


namespace NUMINAMATH_GPT_infinite_castle_hall_unique_l417_41716

theorem infinite_castle_hall_unique :
  (∀ (n : ℕ), ∃ hall : ℕ, ∀ m : ℕ, ((m = 2 * n + 1) ∨ (m = 3 * n + 1)) → hall = m) →
  ∀ (hall1 hall2 : ℕ), hall1 = hall2 :=
by
  sorry

end NUMINAMATH_GPT_infinite_castle_hall_unique_l417_41716


namespace NUMINAMATH_GPT_find_x_l417_41762

theorem find_x (x : ℝ) : 0.003 + 0.158 + x = 2.911 → x = 2.750 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l417_41762


namespace NUMINAMATH_GPT_cylinder_lateral_area_cylinder_volume_cylinder_surface_area_cone_volume_l417_41706

-- Problem 1
theorem cylinder_lateral_area (C H : ℝ) (hC : C = 1.8) (hH : H = 1.5) :
  C * H = 2.7 := by sorry 

-- Problem 2
theorem cylinder_volume (D H : ℝ) (hD : D = 3) (hH : H = 8) :
  (3.14 * ((D * 10 / 2) ^ 2) * H) = 5652 :=
by sorry

-- Problem 3
theorem cylinder_surface_area (r h : ℝ) (hr : r = 6) (hh : h = 5) :
    (3.14 * r * 2 * h + 3.14 * r ^ 2 * 2) = 414.48 :=
by sorry

-- Problem 4
theorem cone_volume (B H : ℝ) (hB : B = 18.84) (hH : H = 6) :
  (1 / 3 * B * H) = 37.68 :=
by sorry

end NUMINAMATH_GPT_cylinder_lateral_area_cylinder_volume_cylinder_surface_area_cone_volume_l417_41706


namespace NUMINAMATH_GPT_find_y_l417_41763

theorem find_y (x y : ℕ) (h1 : x % y = 9) (h2 : (x : ℝ) / y = 86.12) : y = 75 :=
sorry

end NUMINAMATH_GPT_find_y_l417_41763


namespace NUMINAMATH_GPT_apple_equals_pear_l417_41772

-- Define the masses of the apple and pear.
variable (A G : ℝ)

-- The equilibrium condition on the balance scale.
axiom equilibrium_condition : A + 2 * G = 2 * A + G

-- Prove the mass of an apple equals the mass of a pear.
theorem apple_equals_pear (A G : ℝ) (h : A + 2 * G = 2 * A + G) : A = G :=
by
  -- Proof goes here, but we use sorry to indicate the proof's need.
  sorry

end NUMINAMATH_GPT_apple_equals_pear_l417_41772


namespace NUMINAMATH_GPT_true_proposition_is_b_l417_41789

open Real

theorem true_proposition_is_b :
  (∃ n : ℝ, ∀ m : ℝ, m * n = m) ∧
  (¬ ∀ n : ℝ, n^2 ≥ n) ∧
  (¬ ∀ n : ℝ, ∃ m : ℝ, m^2 < n) ∧
  (¬ ∀ n : ℝ, n^2 < n) :=
  by
    sorry

end NUMINAMATH_GPT_true_proposition_is_b_l417_41789


namespace NUMINAMATH_GPT_sum_of_smallest_and_largest_eq_2y_l417_41721

variable (a n y : ℤ) (hn_even : Even n) (hy : y = a + n - 1)

theorem sum_of_smallest_and_largest_eq_2y : a + (a + 2 * (n - 1)) = 2 * y := 
by
  sorry

end NUMINAMATH_GPT_sum_of_smallest_and_largest_eq_2y_l417_41721


namespace NUMINAMATH_GPT_min_max_transformation_a_min_max_transformation_b_l417_41727

theorem min_max_transformation_a {a b : ℝ} (hmin : ∀ x : ℝ, ∀ z : ℝ, (z = (x - 1) / (x^2 + 1)) → (z ≥ a))
  (hmax : ∀ x : ℝ, ∀ z : ℝ, (z = (x - 1) / (x^2 + 1)) → (z ≤ b)) :
  (∀ x : ℝ, ∀ z : ℝ, z = (x^3 - 1) / (x^6 + 1) → z ≥ a) ∧
  (∀ x : ℝ, ∀ z : ℝ, z = (x^3 - 1) / (x^6 + 1) → z ≤ b) :=
sorry

theorem min_max_transformation_b {a b : ℝ} (hmin : ∀ x : ℝ, ∀ z : ℝ, (z = (x - 1) / (x^2 + 1)) → (z ≥ a))
  (hmax : ∀ x : ℝ, ∀ z : ℝ, (z = (x - 1) / (x^2 + 1)) → (z ≤ b)) :
  (∀ x : ℝ, ∀ z : ℝ, z = (x + 1) / (x^2 + 1) → z ≥ -b) ∧
  (∀ x : ℝ, ∀ z : ℝ, z = (x + 1) / (x^2 + 1) → z ≤ -a) :=
sorry

end NUMINAMATH_GPT_min_max_transformation_a_min_max_transformation_b_l417_41727


namespace NUMINAMATH_GPT_distance_between_stations_l417_41748

theorem distance_between_stations :
  ∀ (x t : ℕ), 
    (20 * t = x) ∧ 
    (25 * t = x + 70) →
    (2 * x + 70 = 630) :=
by
  sorry

end NUMINAMATH_GPT_distance_between_stations_l417_41748


namespace NUMINAMATH_GPT_total_bottle_caps_l417_41717

-- Define the conditions
def bottle_caps_per_child : ℕ := 5
def number_of_children : ℕ := 9

-- Define the main statement to be proven
theorem total_bottle_caps : bottle_caps_per_child * number_of_children = 45 :=
by sorry

end NUMINAMATH_GPT_total_bottle_caps_l417_41717


namespace NUMINAMATH_GPT_range_of_d_l417_41723

theorem range_of_d (d : ℝ) : (∃ x : ℝ, |2017 - x| + |2018 - x| ≤ d) ↔ d ≥ 1 :=
sorry

end NUMINAMATH_GPT_range_of_d_l417_41723


namespace NUMINAMATH_GPT_invalid_votes_percentage_is_correct_l417_41725

-- Definitions based on conditions
def total_votes : ℕ := 5500
def other_candidate_votes : ℕ := 1980
def valid_votes_percentage_other : ℚ := 0.45

-- Derived values
def valid_votes : ℚ := other_candidate_votes / valid_votes_percentage_other
def invalid_votes : ℚ := total_votes - valid_votes
def invalid_votes_percentage : ℚ := (invalid_votes / total_votes) * 100

-- Proof statement
theorem invalid_votes_percentage_is_correct :
  invalid_votes_percentage = 20 := sorry

end NUMINAMATH_GPT_invalid_votes_percentage_is_correct_l417_41725


namespace NUMINAMATH_GPT_find_k_values_l417_41768

theorem find_k_values (k : ℚ) 
  (h1 : ∀ k, ∃ m, m = (3 * k + 9) / (7 - k))
  (h2 : ∀ k, m = 2 * k) : 
  (k = 9 / 2 ∨ k = 1) :=
by
  sorry

end NUMINAMATH_GPT_find_k_values_l417_41768


namespace NUMINAMATH_GPT_gcd_two_5_digit_integers_l417_41738

theorem gcd_two_5_digit_integers (a b : ℕ) 
  (h1 : 10^4 ≤ a ∧ a < 10^5)
  (h2 : 10^4 ≤ b ∧ b < 10^5)
  (h3 : 10^8 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^9) :
  Nat.gcd a b < 10^2 :=
by
  sorry  -- Skip the proof

end NUMINAMATH_GPT_gcd_two_5_digit_integers_l417_41738


namespace NUMINAMATH_GPT_hyperbola_equation_l417_41745

theorem hyperbola_equation
  (a b m n e e' c' : ℝ)
  (h1 : 2 * a^2 + b^2 = 2)
  (h2 : e * e' = 1)
  (h_c : c' = e * m)
  (h_b : b^2 = m^2 - n^2)
  (h_e : e = n / m) : 
  y^2 - x^2 = 2 := 
sorry

end NUMINAMATH_GPT_hyperbola_equation_l417_41745


namespace NUMINAMATH_GPT_grain_spilled_correct_l417_41773

variable (original_grain : ℕ) (remaining_grain : ℕ) (grain_spilled : ℕ)

theorem grain_spilled_correct : 
  original_grain = 50870 → remaining_grain = 918 → grain_spilled = original_grain - remaining_grain → grain_spilled = 49952 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_grain_spilled_correct_l417_41773


namespace NUMINAMATH_GPT_shift_right_symmetric_l417_41755

open Real

/-- Given the function y = sin(2x + π/3), after shifting the graph of the function right
    by φ (0 < φ < π/2) units, the resulting graph is symmetric about the y-axis.
    Prove that the value of φ is 5π/12.
-/
theorem shift_right_symmetric (φ : ℝ) (hφ₁ : 0 < φ) (hφ₂ : φ < π / 2)
  (h_sym : ∃ k : ℤ, -2 * φ + π / 3 = k * π + π / 2) : φ = 5 * π / 12 :=
sorry

end NUMINAMATH_GPT_shift_right_symmetric_l417_41755


namespace NUMINAMATH_GPT_store_breaks_even_l417_41760

-- Defining the conditions based on the problem statement.
def cost_price_piece1 (profitable : ℝ → Prop) : Prop :=
  ∃ x, profitable x ∧ 1.5 * x = 150

def cost_price_piece2 (loss : ℝ → Prop) : Prop :=
  ∃ y, loss y ∧ 0.75 * y = 150

def profitable (x : ℝ) : Prop := x + 0.5 * x = 150
def loss (y : ℝ) : Prop := y - 0.25 * y = 150

-- Store breaks even if the total cost price equals the total selling price
theorem store_breaks_even (x y : ℝ)
  (P1 : cost_price_piece1 profitable)
  (P2 : cost_price_piece2 loss) :
  (x + y = 100 + 200) → (150 + 150) = 300 :=
by
  sorry

end NUMINAMATH_GPT_store_breaks_even_l417_41760


namespace NUMINAMATH_GPT_arman_is_6_times_older_than_sister_l417_41778

def sisterWasTwoYearsOldFourYearsAgo := 2
def yearsAgo := 4
def armansAgeInFourYears := 40

def currentAgeOfSister := sisterWasTwoYearsOldFourYearsAgo + yearsAgo
def currentAgeOfArman := armansAgeInFourYears - yearsAgo

theorem arman_is_6_times_older_than_sister :
  currentAgeOfArman = 6 * currentAgeOfSister :=
by
  sorry

end NUMINAMATH_GPT_arman_is_6_times_older_than_sister_l417_41778


namespace NUMINAMATH_GPT_flowers_count_l417_41790

theorem flowers_count (lilies : ℕ) (sunflowers : ℕ) (daisies : ℕ) (total_flowers : ℕ) (roses : ℕ)
  (h1 : lilies = 40) (h2 : sunflowers = 40) (h3 : daisies = 40) (h4 : total_flowers = 160) :
  lilies + sunflowers + daisies + roses = 160 → roses = 40 := 
by
  sorry

end NUMINAMATH_GPT_flowers_count_l417_41790
