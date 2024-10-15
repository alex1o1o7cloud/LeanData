import Mathlib

namespace NUMINAMATH_GPT_opposite_of_neg_2023_l523_52366

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end NUMINAMATH_GPT_opposite_of_neg_2023_l523_52366


namespace NUMINAMATH_GPT_min_possible_value_of_x_l523_52367

theorem min_possible_value_of_x :
  ∀ (x y : ℝ),
  (69 + 53 + 69 + 71 + 78 + x + y) / 7 = 66 →
  (∀ y ≤ 100, x ≥ 0) →
  x ≥ 22 :=
by
  intros x y h_avg h_y 
  -- proof steps go here
  sorry

end NUMINAMATH_GPT_min_possible_value_of_x_l523_52367


namespace NUMINAMATH_GPT_pencil_price_units_l523_52394

def pencil_price_in_units (pencil_price : ℕ) : ℚ := pencil_price / 10000

theorem pencil_price_units 
  (price_of_pencil : ℕ) 
  (h1 : price_of_pencil = 5000 - 20) : 
  pencil_price_in_units price_of_pencil = 0.5 := 
by
  sorry

end NUMINAMATH_GPT_pencil_price_units_l523_52394


namespace NUMINAMATH_GPT_sum_X_Y_Z_l523_52368

theorem sum_X_Y_Z (X Y Z : ℕ) (hX : X ∈ Finset.range 10) (hY : Y ∈ Finset.range 10) (hZ : Z = 0)
     (div9 : (1 + 3 + 0 + 7 + 6 + 7 + 4 + X + 2 + 0 + Y + 0 + 0 + 8 + 0) % 9 = 0) 
     (div7 : (307674 * 10 + X * 20 + Y * 10 + 800) % 7 = 0) :
  X + Y + Z = 7 := 
sorry

end NUMINAMATH_GPT_sum_X_Y_Z_l523_52368


namespace NUMINAMATH_GPT_cube_relation_l523_52373

theorem cube_relation (x : ℝ) (h : x - 1/x = 5) : x^3 - 1/x^3 = 140 :=
by
  sorry

end NUMINAMATH_GPT_cube_relation_l523_52373


namespace NUMINAMATH_GPT_derivative_of_odd_is_even_l523_52382

variable (f : ℝ → ℝ) (g : ℝ → ℝ)

-- Assume f is an odd function
axiom f_odd : ∀ x, f (-x) = -f x

-- Assume g is the derivative of f
axiom g_derivative : ∀ x, g x = deriv f x

-- Goal: Prove that g is an even function, i.e., g(-x) = g(x)
theorem derivative_of_odd_is_even : ∀ x, g (-x) = g x :=
by
  sorry

end NUMINAMATH_GPT_derivative_of_odd_is_even_l523_52382


namespace NUMINAMATH_GPT_line_eq_l523_52301

theorem line_eq (x_1 y_1 x_2 y_2 : ℝ) (h1 : x_1 + x_2 = 8) (h2 : y_1 + y_2 = 2)
  (h3 : x_1^2 - 4 * y_1^2 = 4) (h4 : x_2^2 - 4 * y_2^2 = 4) :
  ∃ l : ℝ, ∀ x y : ℝ, x - y - 3 = l :=
by sorry

end NUMINAMATH_GPT_line_eq_l523_52301


namespace NUMINAMATH_GPT_compute_xy_l523_52341

theorem compute_xy (x y : ℝ) (h1 : x + y = 9) (h2 : x^3 + y^3 = 351) : x * y = 14 :=
by
  sorry

end NUMINAMATH_GPT_compute_xy_l523_52341


namespace NUMINAMATH_GPT_arith_sqrt_abs_neg_nine_l523_52353

theorem arith_sqrt_abs_neg_nine : Real.sqrt (abs (-9)) = 3 := by
  sorry

end NUMINAMATH_GPT_arith_sqrt_abs_neg_nine_l523_52353


namespace NUMINAMATH_GPT_pencils_sold_is_correct_l523_52397

-- Define the conditions
def first_two_students_pencils : Nat := 2 * 2
def next_six_students_pencils : Nat := 6 * 3
def last_two_students_pencils : Nat := 2 * 1
def total_pencils_sold : Nat := first_two_students_pencils + next_six_students_pencils + last_two_students_pencils

-- Prove that all pencils sold equals 24
theorem pencils_sold_is_correct : total_pencils_sold = 24 :=
by 
  -- Add the statement to be proved here
  sorry

end NUMINAMATH_GPT_pencils_sold_is_correct_l523_52397


namespace NUMINAMATH_GPT_S_30_value_l523_52335

noncomputable def geometric_sequence_sum (n : ℕ) : ℝ := sorry

axiom S_10 : geometric_sequence_sum 10 = 10
axiom S_20 : geometric_sequence_sum 20 = 30

theorem S_30_value : geometric_sequence_sum 30 = 70 :=
by
  sorry

end NUMINAMATH_GPT_S_30_value_l523_52335


namespace NUMINAMATH_GPT_evaluate_expression_l523_52354

theorem evaluate_expression :
  (827 * 827) - ((827 - 1) * (827 + 1)) = 1 :=
sorry

end NUMINAMATH_GPT_evaluate_expression_l523_52354


namespace NUMINAMATH_GPT_ten_faucets_fill_time_l523_52381

theorem ten_faucets_fill_time (rate : ℕ → ℕ → ℝ) (gallons : ℕ) (minutes : ℝ) :
  rate 5 9 = 150 / 5 ∧
  rate 10 135 = 75 / 30 * rate 10 9 / 0.9 * 60 →
  9 * 60 / 30 * 75 / 10 * 60 = 135 :=
sorry

end NUMINAMATH_GPT_ten_faucets_fill_time_l523_52381


namespace NUMINAMATH_GPT_no_nat_solutions_for_m2_eq_n2_plus_2014_l523_52332

theorem no_nat_solutions_for_m2_eq_n2_plus_2014 :
  ∀ m n : ℕ, ¬(m^2 = n^2 + 2014) := by
sorry

end NUMINAMATH_GPT_no_nat_solutions_for_m2_eq_n2_plus_2014_l523_52332


namespace NUMINAMATH_GPT_proof_problem_l523_52314

noncomputable def p : Prop := ∃ x : ℝ, Real.sin x > 1
noncomputable def q : Prop := ∀ x : ℝ, Real.exp (-x) < 0

theorem proof_problem : ¬ (p ∨ q) :=
by sorry

end NUMINAMATH_GPT_proof_problem_l523_52314


namespace NUMINAMATH_GPT_correct_assignment_l523_52386

-- Definition of conditions
def is_variable_free (e : String) : Prop := -- a simplistic placeholder
  e ∈ ["A", "B", "C", "D", "x"]

def valid_assignment (lhs : String) (rhs : String) : Prop :=
  is_variable_free lhs ∧ ¬(is_variable_free rhs)

-- The statement of the proof problem
theorem correct_assignment : valid_assignment "A" "A * A + A - 2" :=
by
  sorry

end NUMINAMATH_GPT_correct_assignment_l523_52386


namespace NUMINAMATH_GPT_speed_of_current_l523_52338

-- Definitions of the given conditions
def downstream_time := 6 / 60 -- time in hours to travel 1 km downstream
def upstream_time := 10 / 60 -- time in hours to travel 1 km upstream

-- Definition of speeds
def downstream_speed := 1 / downstream_time -- speed in km/h downstream
def upstream_speed := 1 / upstream_time -- speed in km/h upstream

-- Theorem statement
theorem speed_of_current : 
  (downstream_speed - upstream_speed) / 2 = 2 := 
by 
  -- We skip the proof for now
  sorry

end NUMINAMATH_GPT_speed_of_current_l523_52338


namespace NUMINAMATH_GPT_smallest_n_term_dec_l523_52352

theorem smallest_n_term_dec (n : ℕ) (h_pos : 0 < n) (h : ∀ d, 0 < d → d = n + 150 → ∀ p, p ∣ d → (p = 2 ∨ p = 5)) :
  n = 10 :=
by {
  sorry
}

end NUMINAMATH_GPT_smallest_n_term_dec_l523_52352


namespace NUMINAMATH_GPT_possible_teams_count_l523_52329

-- Defining the problem
def team_group_division : Prop :=
  ∃ (g1 g2 g3 g4 : ℕ), (g1 ≥ 2) ∧ (g2 ≥ 2) ∧ (g3 ≥ 2) ∧ (g4 ≥ 2) ∧
  (66 = (g1 * (g1 - 1) / 2) + (g2 * (g2 - 1) / 2) + (g3 * (g3 - 1) / 2) + 
       (g4 * (g4 - 1) / 2)) ∧ 
  ((g1 + g2 + g3 + g4 = 21) ∨ (g1 + g2 + g3 + g4 = 22) ∨ 
   (g1 + g2 + g3 + g4 = 23) ∨ (g1 + g2 + g3 + g4 = 24) ∨ 
   (g1 + g2 + g3 + g4 = 25))

-- Theorem statement to prove
theorem possible_teams_count : team_group_division :=
sorry

end NUMINAMATH_GPT_possible_teams_count_l523_52329


namespace NUMINAMATH_GPT_sum_of_possible_values_l523_52379

theorem sum_of_possible_values :
  ∀ x, (|x - 5| - 4 = 3) → x = 12 ∨ x = -2 → (12 + (-2) = 10) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_possible_values_l523_52379


namespace NUMINAMATH_GPT_tank_capacity_is_correct_l523_52355

-- Definition of the problem conditions
def initial_fraction := 1 / 3
def added_water := 180
def final_fraction := 2 / 3

-- Capacity of the tank
noncomputable def tank_capacity : ℕ := 540

-- Proof statement
theorem tank_capacity_is_correct (x : ℕ) :
  (initial_fraction * x + added_water = final_fraction * x) → x = tank_capacity := 
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_tank_capacity_is_correct_l523_52355


namespace NUMINAMATH_GPT_jam_fraction_left_after_dinner_l523_52393

noncomputable def jam_left_after_dinner (initial: ℚ) (lunch_fraction: ℚ) (dinner_fraction: ℚ) : ℚ :=
  initial - (initial * lunch_fraction) - ((initial - (initial * lunch_fraction)) * dinner_fraction)

theorem jam_fraction_left_after_dinner :
  jam_left_after_dinner 1 (1/3) (1/7) = (4/7) :=
by
  sorry

end NUMINAMATH_GPT_jam_fraction_left_after_dinner_l523_52393


namespace NUMINAMATH_GPT_sum_product_of_integers_l523_52323

theorem sum_product_of_integers (a b c : ℕ) (h₁ : c = a + b) (h₂ : N = a * b * c) (h₃ : N = 8 * (a + b + c)) : 
  a * b * (a + b) = 16 * (a + b) :=
by {
  sorry
}

end NUMINAMATH_GPT_sum_product_of_integers_l523_52323


namespace NUMINAMATH_GPT_only_B_forms_triangle_l523_52395

/-- Check if a set of line segments can form a triangle --/
def can_form_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem only_B_forms_triangle :
  ¬ can_form_triangle 2 6 3 ∧
  can_form_triangle 6 7 8 ∧
  ¬ can_form_triangle 1 7 9 ∧
  ¬ can_form_triangle (3 / 2) 4 (5 / 2) :=
by
  sorry

end NUMINAMATH_GPT_only_B_forms_triangle_l523_52395


namespace NUMINAMATH_GPT_part_a_part_b_part_c_l523_52374

-- Part (a)
theorem part_a (m : ℤ) : (m^2 + 10) % (m - 2) = 0 ∧ (m^2 + 10) % (m + 4) = 0 ↔ m = -5 ∨ m = 9 := 
sorry

-- Part (b)
theorem part_b (n : ℤ) : ∃ m : ℤ, (m^2 + n^2 + 1) % (m - n + 1) = 0 ∧ (m^2 + n^2 + 1) % (m + n + 1) = 0 :=
sorry

-- Part (c)
theorem part_c (n : ℤ) : ∃ N : ℕ, ∀ m : ℤ, (m^2 + n^2 + 1) % (m - n + 1) = 0 ∧ (m^2 + n^2 + 1) % (m + n + 1) = 0 → m < N :=
sorry

end NUMINAMATH_GPT_part_a_part_b_part_c_l523_52374


namespace NUMINAMATH_GPT_parabola_through_point_l523_52388

theorem parabola_through_point (a b : ℝ) (ha : 0 < a) :
  ∃ f : ℝ → ℝ, (∀ x, f x = -a*x^2 + b*x + 1) ∧ f 0 = 1 :=
by
  -- We are given a > 0
  -- We need to show there exists a parabola of the form y = -a*x^2 + b*x + 1 passing through (0,1)
  sorry

end NUMINAMATH_GPT_parabola_through_point_l523_52388


namespace NUMINAMATH_GPT_quad_root_sum_product_l523_52359

theorem quad_root_sum_product (α β : ℝ) (h₁ : α ≠ β) (h₂ : α * α - 5 * α - 2 = 0) (h₃ : β * β - 5 * β - 2 = 0) : 
  α + β + α * β = 3 := 
by
  sorry

end NUMINAMATH_GPT_quad_root_sum_product_l523_52359


namespace NUMINAMATH_GPT_sarah_initial_money_l523_52340

-- Definitions based on conditions
def cost_toy_car := 11
def cost_scarf := 10
def cost_beanie := 14
def remaining_money := 7
def total_cost := 2 * cost_toy_car + cost_scarf + cost_beanie
def initial_money := total_cost + remaining_money

-- Statement of the theorem
theorem sarah_initial_money : initial_money = 53 :=
by
  sorry

end NUMINAMATH_GPT_sarah_initial_money_l523_52340


namespace NUMINAMATH_GPT_fibonacci_p_arithmetic_periodic_l523_52342

-- Define p-arithmetic system and its properties
def p_arithmetic (p : ℕ) : Prop :=
  ∀ (a : ℤ), a ≠ 0 → a^(p-1) = 1

-- Define extraction of sqrt(5)
def sqrt5_extractable (p : ℕ) : Prop :=
  ∃ (r : ℝ), r^2 = 5

-- Define Fibonacci sequence in p-arithmetic
def fibonacci_p_arithmetic (p : ℕ) (v : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, v (n+2) = v (n+1) + v n

-- Main Theorem
theorem fibonacci_p_arithmetic_periodic (p : ℕ) (v : ℕ → ℤ) :
  p_arithmetic p →
  sqrt5_extractable p →
  fibonacci_p_arithmetic p v →
  (∀ k : ℕ, v (k + p) = v k) :=
by
  intros _ _ _
  sorry

end NUMINAMATH_GPT_fibonacci_p_arithmetic_periodic_l523_52342


namespace NUMINAMATH_GPT_handshake_count_l523_52328

theorem handshake_count :
  let total_people := 5 * 4
  let handshakes_per_person := total_people - 1 - 3
  let total_handshakes_with_double_count := total_people * handshakes_per_person
  let total_handshakes := total_handshakes_with_double_count / 2
  total_handshakes = 160 :=
by
-- We include "sorry" to indicate that the proof is not provided.
sorry

end NUMINAMATH_GPT_handshake_count_l523_52328


namespace NUMINAMATH_GPT_annual_average_growth_rate_l523_52316

theorem annual_average_growth_rate (x : ℝ) :
  7200 * (1 + x)^2 = 8450 :=
sorry

end NUMINAMATH_GPT_annual_average_growth_rate_l523_52316


namespace NUMINAMATH_GPT_jerry_age_l523_52326

theorem jerry_age (M J : ℕ) (h1 : M = 2 * J - 2) (h2 : M = 18) : J = 10 := by
  sorry

end NUMINAMATH_GPT_jerry_age_l523_52326


namespace NUMINAMATH_GPT_f_2011_is_zero_l523_52380

noncomputable def f : ℝ → ℝ := sorry

-- Conditions
axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom periodic_f : ∀ x : ℝ, f (x + 2) = f (x) + f (1)

-- Theorem stating the mathematically equivalent proof problem
theorem f_2011_is_zero : f (2011) = 0 :=
sorry

end NUMINAMATH_GPT_f_2011_is_zero_l523_52380


namespace NUMINAMATH_GPT_liam_total_money_l523_52305

-- Define the conditions as noncomputable since they involve monetary calculations
noncomputable def liam_money (initial_bottles : ℕ) (price_per_bottle : ℕ) (bottles_sold : ℕ) (extra_money : ℕ) : ℚ :=
  let cost := initial_bottles * price_per_bottle
  let money_after_selling_part := cost + extra_money
  let selling_price_per_bottle := money_after_selling_part / bottles_sold
  let total_revenue := initial_bottles * selling_price_per_bottle
  total_revenue

-- State the theorem with the given problem
theorem liam_total_money :
  let initial_bottles := 50
  let price_per_bottle := 1
  let bottles_sold := 40
  let extra_money := 10
  liam_money initial_bottles price_per_bottle bottles_sold extra_money = 75 := 
sorry

end NUMINAMATH_GPT_liam_total_money_l523_52305


namespace NUMINAMATH_GPT_intersection_point_l523_52337

-- Definitions of the lines
def line1 (x y : ℚ) : Prop := 8 * x - 5 * y = 10
def line2 (x y : ℚ) : Prop := 6 * x + 2 * y = 20

-- Theorem stating the intersection point
theorem intersection_point : line1 (60 / 23) (50 / 23) ∧ line2 (60 / 23) (50 / 23) :=
by {
  sorry
}

end NUMINAMATH_GPT_intersection_point_l523_52337


namespace NUMINAMATH_GPT_rectangles_cannot_cover_large_rectangle_l523_52377

theorem rectangles_cannot_cover_large_rectangle (n m : ℕ) (a b c d: ℕ) : 
  n = 14 → m = 9 → a = 2 → b = 3 → c = 3 → d = 2 → 
  (∀ (v_rects : ℕ) (h_rects : ℕ), v_rects = 10 → h_rects = 11 →
    (∀ (rect_area : ℕ), rect_area = n * m →
      (∀ (small_rect_area : ℕ), 
        small_rect_area = (v_rects * (a * b)) + (h_rects * (c * d)) →
        small_rect_area = rect_area → 
        false))) :=
by
  intros n_eq m_eq a_eq b_eq c_eq d_eq
       v_rects h_rects v_rects_eq h_rects_eq
       rect_area rect_area_eq small_rect_area small_rect_area_eq area_sum_eq
  sorry

end NUMINAMATH_GPT_rectangles_cannot_cover_large_rectangle_l523_52377


namespace NUMINAMATH_GPT_final_books_is_correct_l523_52398

def initial_books : ℝ := 35.5
def books_bought : ℝ := 12.3
def books_given_to_friends : ℝ := 7.2
def books_donated : ℝ := 20.8

theorem final_books_is_correct :
  (initial_books + books_bought - books_given_to_friends - books_donated) = 19.8 := by
  sorry

end NUMINAMATH_GPT_final_books_is_correct_l523_52398


namespace NUMINAMATH_GPT_intersection_points_l523_52343

-- Definitions and conditions
def is_ellipse (e : ℝ → ℝ → Prop) : Prop :=
  ∀ x y : ℝ, e x y ↔ x^2 + 2*y^2 = 2

def is_tangent_or_intersects (l : ℝ → ℝ) (e : ℝ → ℝ → Prop) : Prop :=
  ∃ z1 z2 : ℝ, (e z1 (l z1) ∨ e z2 (l z2))

def lines_intersect (l1 l2 : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, l1 x = l2 x

theorem intersection_points :
  ∀ (e : ℝ → ℝ → Prop) (l1 l2 : ℝ → ℝ),
  is_ellipse e →
  is_tangent_or_intersects l1 e →
  is_tangent_or_intersects l2 e →
  lines_intersect l1 l2 →
  ∃ n : ℕ, n = 2 ∨ n = 3 ∨ n = 4 :=
by
  intros e l1 l2 he hto1 hto2 hl
  sorry

end NUMINAMATH_GPT_intersection_points_l523_52343


namespace NUMINAMATH_GPT_stratified_sampling_l523_52371

theorem stratified_sampling
  (total_products : ℕ)
  (sample_size : ℕ)
  (workshop_products : ℕ)
  (h1 : total_products = 2048)
  (h2 : sample_size = 128)
  (h3 : workshop_products = 256) :
  (workshop_products / total_products) * sample_size = 16 := 
by
  rw [h1, h2, h3]
  norm_num
  
  sorry

end NUMINAMATH_GPT_stratified_sampling_l523_52371


namespace NUMINAMATH_GPT_ara_current_height_l523_52396

theorem ara_current_height (original_height : ℚ) (shea_growth_ratio : ℚ) (ara_growth_ratio : ℚ) (shea_current_height : ℚ) (h1 : shea_growth_ratio = 0.25) (h2 : ara_growth_ratio = 0.75) (h3 : shea_current_height = 75) (h4 : shea_current_height = original_height * (1 + shea_growth_ratio)) : 
  original_height * (1 + ara_growth_ratio * shea_growth_ratio) = 71.25 := 
by
  sorry

end NUMINAMATH_GPT_ara_current_height_l523_52396


namespace NUMINAMATH_GPT_find_a_l523_52322

theorem find_a (a : ℝ) : 
  (∃ (r : ℕ), r = 3 ∧ 
  ((-1)^r * (Nat.choose 5 r : ℝ) * a^(5 - r) = -40)) ↔ a = 2 ∨ a = -2 :=
by
    sorry

end NUMINAMATH_GPT_find_a_l523_52322


namespace NUMINAMATH_GPT_value_of_fraction_l523_52315

theorem value_of_fraction (x y z w : ℕ) (h₁ : x = 4 * y) (h₂ : y = 3 * z) (h₃ : z = 5 * w) :
  x * z / (y * w) = 20 := by
  sorry

end NUMINAMATH_GPT_value_of_fraction_l523_52315


namespace NUMINAMATH_GPT_find_speed_of_B_l523_52362

namespace BicycleSpeed

variables (d : ℝ) (t_diff : ℝ) (v_A v_B : ℝ)

-- Given conditions
def given_conditions := 
d = 12 ∧ 
t_diff = 1/6 ∧ 
v_A = 1.2 * v_B ∧ 
(12 / v_B - 12 / v_A = t_diff)

theorem find_speed_of_B
  (h : given_conditions d t_diff v_A v_B) : 
  v_B = 12 :=
sorry

end BicycleSpeed

end NUMINAMATH_GPT_find_speed_of_B_l523_52362


namespace NUMINAMATH_GPT_measure_one_kg_grain_l523_52383

/-- Proving the possibility of measuring exactly 1 kg of grain
    using a balance scale, one 3 kg weight, and three weighings. -/
theorem measure_one_kg_grain :
  ∃ (weighings : ℕ) (balance_scale : ℕ → ℤ) (weight_3kg : ℤ → Prop),
  weighings = 3 ∧
  (∀ w, weight_3kg w ↔ w = 3) ∧
  ∀ n m, balance_scale n = 0 ∧ balance_scale m = 1 → true :=
sorry

end NUMINAMATH_GPT_measure_one_kg_grain_l523_52383


namespace NUMINAMATH_GPT_friend_spent_more_l523_52317

variable (total_spent : ℕ)
variable (friend_spent : ℕ)
variable (you_spent : ℕ)

-- Conditions
axiom total_is_11 : total_spent = 11
axiom friend_is_7 : friend_spent = 7
axiom spending_relation : total_spent = friend_spent + you_spent

-- Question
theorem friend_spent_more : friend_spent - you_spent = 3 :=
by
  sorry -- Here should be the formal proof

end NUMINAMATH_GPT_friend_spent_more_l523_52317


namespace NUMINAMATH_GPT_probability_of_hitting_target_at_least_once_l523_52385

noncomputable def prob_hit_target_once : ℚ := 2/3

noncomputable def prob_miss_target_once : ℚ := 1 - prob_hit_target_once

noncomputable def prob_miss_target_three_times : ℚ := prob_miss_target_once ^ 3

noncomputable def prob_hit_target_at_least_once : ℚ := 1 - prob_miss_target_three_times

theorem probability_of_hitting_target_at_least_once :
  prob_hit_target_at_least_once = 26 / 27 := 
sorry

end NUMINAMATH_GPT_probability_of_hitting_target_at_least_once_l523_52385


namespace NUMINAMATH_GPT_pages_left_to_read_l523_52365

-- Defining the given conditions
def total_pages : ℕ := 500
def read_first_night : ℕ := (20 * total_pages) / 100
def read_second_night : ℕ := (20 * total_pages) / 100
def read_third_night : ℕ := (30 * total_pages) / 100

-- The total pages read over the three nights
def total_read : ℕ := read_first_night + read_second_night + read_third_night

-- The remaining pages to be read
def remaining_pages : ℕ := total_pages - total_read

theorem pages_left_to_read : remaining_pages = 150 :=
by
  -- Leaving the proof as a placeholder
  sorry

end NUMINAMATH_GPT_pages_left_to_read_l523_52365


namespace NUMINAMATH_GPT_households_with_both_car_and_bike_l523_52345

theorem households_with_both_car_and_bike 
  (total_households : ℕ) 
  (households_without_either : ℕ) 
  (households_with_car : ℕ) 
  (households_with_bike_only : ℕ)
  (H1 : total_households = 90)
  (H2 : households_without_either = 11)
  (H3 : households_with_car = 44)
  (H4 : households_with_bike_only = 35)
  : ∃ B : ℕ, households_with_car - households_with_bike_only = B ∧ B = 9 := 
by
  sorry

end NUMINAMATH_GPT_households_with_both_car_and_bike_l523_52345


namespace NUMINAMATH_GPT_find_m_from_permutation_l523_52358

theorem find_m_from_permutation (A : Nat → Nat → Nat) (m : Nat) (hA : A 11 m = 11 * 10 * 9 * 8 * 7 * 6 * 5) : m = 7 :=
sorry

end NUMINAMATH_GPT_find_m_from_permutation_l523_52358


namespace NUMINAMATH_GPT_problem_l523_52390

theorem problem (a₅ b₅ a₆ b₆ a₇ b₇ : ℤ) (S₇ S₅ T₆ T₄ : ℤ)
  (h1 : a₅ = b₅)
  (h2 : a₆ = b₆)
  (h3 : S₇ - S₅ = 4 * (T₆ - T₄)) :
  (a₇ + a₅) / (b₇ + b₅) = -1 :=
sorry

end NUMINAMATH_GPT_problem_l523_52390


namespace NUMINAMATH_GPT_nancy_hourly_wage_l523_52309

theorem nancy_hourly_wage 
  (tuition_per_semester : ℕ := 22000) 
  (parents_cover : ℕ := tuition_per_semester / 2) 
  (scholarship : ℕ := 3000) 
  (student_loan : ℕ := 2 * scholarship) 
  (work_hours : ℕ := 200) 
  (remaining_tuition : ℕ := parents_cover - scholarship - student_loan) :
  (remaining_tuition / work_hours = 10) :=
  by
  sorry

end NUMINAMATH_GPT_nancy_hourly_wage_l523_52309


namespace NUMINAMATH_GPT_probability_of_sequence_l523_52312

theorem probability_of_sequence :
  let total_cards := 52
  let face_cards := 12
  let hearts := 13
  let first_card_face_prob := (face_cards : ℝ) / total_cards
  let second_card_heart_prob := (10 : ℝ) / (total_cards - 1)
  let third_card_face_prob := (11 : ℝ) / (total_cards - 2)
  let total_prob := first_card_face_prob * second_card_heart_prob * third_card_face_prob
  total_prob = 1 / 100.455 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_sequence_l523_52312


namespace NUMINAMATH_GPT_cyclist_speed_l523_52334

/-- 
  Two cyclists A and B start at the same time from Newton to Kingston, a distance of 50 miles. 
  Cyclist A travels 5 mph slower than cyclist B. After reaching Kingston, B immediately turns 
  back and meets A 10 miles from Kingston. --/
theorem cyclist_speed (a b : ℕ) (h1 : b = a + 5) (h2 : 40 / a = 60 / b) : a = 10 :=
by
  sorry

end NUMINAMATH_GPT_cyclist_speed_l523_52334


namespace NUMINAMATH_GPT_find_x_value_l523_52327

noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then 2^(-x)
  else Real.log x * Real.log 81

theorem find_x_value (x : ℝ) (h : f x = 1 / 4) : x = 3 :=
sorry

end NUMINAMATH_GPT_find_x_value_l523_52327


namespace NUMINAMATH_GPT_red_ball_value_l523_52349

theorem red_ball_value (r b g : ℕ) (blue_points green_points : ℕ)
  (h1 : blue_points = 4)
  (h2 : green_points = 5)
  (h3 : b = g)
  (h4 : r^4 * blue_points^b * green_points^g = 16000)
  (h5 : b = 6) :
  r = 1 :=
by
  sorry

end NUMINAMATH_GPT_red_ball_value_l523_52349


namespace NUMINAMATH_GPT_find_c_find_A_l523_52357

open Real

noncomputable def acute_triangle_sides (A B C a b c : ℝ) : Prop :=
  a = b * cos C + (sqrt 3 / 3) * c * sin B

theorem find_c (A B C a b c : ℝ) (ha : a = 2) (hb : b = sqrt 7) 
  (hab : acute_triangle_sides A B C a b c) : c = 3 := 
sorry

theorem find_A (A B C : ℝ) (h : sqrt 3 * sin (2 * A - π / 6) - 2 * (sin (C - π / 12))^2 = 0)
  (h_range : π / 6 < A ∧ A < π / 2) : A = π / 4 :=
sorry

end NUMINAMATH_GPT_find_c_find_A_l523_52357


namespace NUMINAMATH_GPT_point_in_second_quadrant_l523_52399

variable (m : ℝ)

/-- 
If point P(m-1, 3) is in the second quadrant, 
then a possible value of m is -1
--/
theorem point_in_second_quadrant (h1 : (m - 1 < 0)) : m = -1 :=
by sorry

end NUMINAMATH_GPT_point_in_second_quadrant_l523_52399


namespace NUMINAMATH_GPT_decrement_from_each_observation_l523_52347

theorem decrement_from_each_observation (n : Nat) (mean_original mean_updated decrement : ℝ)
  (h1 : n = 50)
  (h2 : mean_original = 200)
  (h3 : mean_updated = 191)
  (h4 : decrement = 9) :
  (mean_original - mean_updated) * (n : ℝ) / n = decrement :=
by
  sorry

end NUMINAMATH_GPT_decrement_from_each_observation_l523_52347


namespace NUMINAMATH_GPT_days_required_by_x_l523_52336

theorem days_required_by_x (x y : ℝ) 
  (h1 : (1 / x + 1 / y = 1 / 12)) 
  (h2 : (1 / y = 1 / 24)) : 
  x = 24 := 
by
  sorry

end NUMINAMATH_GPT_days_required_by_x_l523_52336


namespace NUMINAMATH_GPT_larger_integer_exists_l523_52304

theorem larger_integer_exists (a b : ℤ) (h1 : a - b = 8) (h2 : a * b = 272) : a = 17 :=
sorry

end NUMINAMATH_GPT_larger_integer_exists_l523_52304


namespace NUMINAMATH_GPT_school_selection_theorem_l523_52303

-- Define the basic setup and conditions
def school_selection_problem : Prop :=
  let schools := ["A", "B", "C", "D"]
  let total_schools := 4
  let selected_schools := 2
  let combinations := Nat.choose total_schools selected_schools
  let favorable_outcomes := Nat.choose (total_schools - 1) (selected_schools - 1)
  let probability := (favorable_outcomes : ℚ) / (combinations : ℚ)
  probability = 1 / 2

-- Proof is yet to be provided
theorem school_selection_theorem : school_selection_problem := sorry

end NUMINAMATH_GPT_school_selection_theorem_l523_52303


namespace NUMINAMATH_GPT_div_of_abs_values_l523_52339

theorem div_of_abs_values (x y : ℝ) (hx : |x| = 4) (hy : |y| = 2) (hxy : x < y) : x / y = -2 := 
by
  sorry

end NUMINAMATH_GPT_div_of_abs_values_l523_52339


namespace NUMINAMATH_GPT_min_value_of_a_l523_52360

theorem min_value_of_a (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (x^2 + 2*x*y ≤ a*(x^2 + y^2)) → (a ≥ (Real.sqrt 5 + 1) / 2) := 
sorry

end NUMINAMATH_GPT_min_value_of_a_l523_52360


namespace NUMINAMATH_GPT_coordinate_inequality_l523_52307

theorem coordinate_inequality (x y : ℝ) :
  (xy > 0 → (x - 2)^2 + (y + 1)^2 < 5) ∧ (xy < 0 → (x - 2)^2 + (y + 1)^2 > 5) :=
by
  sorry

end NUMINAMATH_GPT_coordinate_inequality_l523_52307


namespace NUMINAMATH_GPT_visitors_on_previous_day_is_246_l523_52310

def visitors_on_previous_day : Nat := 246
def total_visitors_in_25_days : Nat := 949

theorem visitors_on_previous_day_is_246 :
  visitors_on_previous_day = 246 := 
by
  rfl

end NUMINAMATH_GPT_visitors_on_previous_day_is_246_l523_52310


namespace NUMINAMATH_GPT_only_solution_xyz_l523_52351

theorem only_solution_xyz : 
  ∀ (x y z : ℕ), x^3 + 4 * y^3 = 16 * z^3 + 4 * x * y * z → x = 0 ∧ y = 0 ∧ z = 0 :=
by
  intro x y z
  intro h
  sorry

end NUMINAMATH_GPT_only_solution_xyz_l523_52351


namespace NUMINAMATH_GPT_solve_system_equations_l523_52350

theorem solve_system_equations (a b c : ℝ) (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) :
    ∃ x y z : ℝ,  
      (x * y = (z - a) ^ 2) ∧
      (y * z = (x - b) ^ 2) ∧
      (z * x = (y - c) ^ 2) ∧
      x = ((b ^ 2 - a * c) ^ 2) / (a ^ 3 + b ^ 3 + c ^ 3 - 3 * a * b * c) ∧
      y = ((c ^ 2 - a * b) ^ 2) / (a ^ 3 + b ^ 3 + c ^ 3 - 3 * a * b * c) ∧
      z = ((a ^ 2 - b * c) ^ 2) / (a ^ 3 + b ^ 3 + c ^ 3 - 3 * a * b * c) :=
sorry

end NUMINAMATH_GPT_solve_system_equations_l523_52350


namespace NUMINAMATH_GPT_values_of_n_l523_52356

theorem values_of_n (a b d : ℕ) :
  7 * a + 77 * b + 7777 * d = 6700 →
  ∃ n : ℕ, ∃ (count : ℕ), count = 107 ∧ n = a + 2 * b + 4 * d := 
by
  sorry

end NUMINAMATH_GPT_values_of_n_l523_52356


namespace NUMINAMATH_GPT_geo_seq_decreasing_l523_52306

variables (a_1 q : ℝ) (a : ℕ → ℝ)
-- Define the geometric sequence
def geo_seq (a_1 q : ℝ) (n : ℕ) : ℝ := a_1 * q ^ n

-- The problem statement as a Lean theorem
theorem geo_seq_decreasing (h1 : a_1 * (q - 1) < 0) (h2 : q > 0) :
  ∀ n : ℕ, geo_seq a_1 q (n + 1) < geo_seq a_1 q n :=
by
  sorry

end NUMINAMATH_GPT_geo_seq_decreasing_l523_52306


namespace NUMINAMATH_GPT_skateboarder_speed_l523_52331

theorem skateboarder_speed (d t : ℕ) (ft_per_mile hr_to_sec : ℕ)
  (h1 : d = 660) (h2 : t = 30) (h3 : ft_per_mile = 5280) (h4 : hr_to_sec = 3600) :
  ((d / t) / ft_per_mile) * hr_to_sec = 15 :=
by sorry

end NUMINAMATH_GPT_skateboarder_speed_l523_52331


namespace NUMINAMATH_GPT_remainder_when_dividing_n_by_d_l523_52313

def n : ℕ := 25197638
def d : ℕ := 4
def r : ℕ := 2

theorem remainder_when_dividing_n_by_d :
  n % d = r :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_dividing_n_by_d_l523_52313


namespace NUMINAMATH_GPT_number_less_than_value_l523_52320

-- Definition for the conditions
def exceeds_condition (x y : ℕ) : Prop := x - 18 = 3 * (y - x)
def specific_value (x : ℕ) : Prop := x = 69

-- Statement of the theorem
theorem number_less_than_value : ∃ y : ℕ, (exceeds_condition 69 y) ∧ (specific_value 69) → y = 86 :=
by
  -- To be proved
  sorry

end NUMINAMATH_GPT_number_less_than_value_l523_52320


namespace NUMINAMATH_GPT_jane_earnings_in_two_weeks_l523_52372

-- Define the conditions in the lean environment
def number_of_chickens : ℕ := 10
def eggs_per_chicken_per_week : ℕ := 6
def selling_price_per_dozen : ℕ := 2

-- Statement of the proof problem
theorem jane_earnings_in_two_weeks :
  (number_of_chickens * eggs_per_chicken_per_week * 2) / 12 * selling_price_per_dozen = 20 :=
by
  sorry

end NUMINAMATH_GPT_jane_earnings_in_two_weeks_l523_52372


namespace NUMINAMATH_GPT_divisibility_problem_l523_52319

theorem divisibility_problem (n : ℕ) : n-1 ∣ n^n - 7*n + 5*n^2024 + 3*n^2 - 2 := 
by
  sorry

end NUMINAMATH_GPT_divisibility_problem_l523_52319


namespace NUMINAMATH_GPT_discriminant_nonnegative_l523_52384

theorem discriminant_nonnegative {x : ℤ} (a : ℝ) (h₁ : x^2 * (49 - 40 * x^2) ≥ 0) :
  a = 0 ∨ a = 1 ∨ a = -1 ∨ a = 5/2 ∨ a = -5/2 := sorry

end NUMINAMATH_GPT_discriminant_nonnegative_l523_52384


namespace NUMINAMATH_GPT_correct_calculation_l523_52375

variable (a : ℝ)

theorem correct_calculation : (-2 * a) ^ 3 = -8 * a ^ 3 := by
  sorry

end NUMINAMATH_GPT_correct_calculation_l523_52375


namespace NUMINAMATH_GPT_continuous_piecewise_function_l523_52348

theorem continuous_piecewise_function (a c : ℝ) (h1 : 2 * a * 2 + 6 = 3 * 2 - 2) (h2 : 4 * (-2) + 2 * c = 3 * (-2) - 2) : 
  a + c = -1/2 := 
sorry

end NUMINAMATH_GPT_continuous_piecewise_function_l523_52348


namespace NUMINAMATH_GPT_min_side_length_l523_52389

noncomputable def side_length_min : ℝ := 30

theorem min_side_length (s r : ℝ) (hs₁ : s^2 ≥ 900) (hr₁ : π * r^2 ≥ 100) (hr₂ : 2 * r ≤ s) :
  s ≥ side_length_min :=
by
  sorry

end NUMINAMATH_GPT_min_side_length_l523_52389


namespace NUMINAMATH_GPT_purely_imaginary_roots_iff_l523_52321

theorem purely_imaginary_roots_iff (z : ℂ) (k : ℝ) (i : ℂ) (h_i2 : i^2 = -1) :
  (∀ r : ℂ, (20 * r^2 + 6 * i * r - ↑k = 0) → (∃ b : ℝ, r = b * i)) ↔ (k = 9 / 5) :=
sorry

end NUMINAMATH_GPT_purely_imaginary_roots_iff_l523_52321


namespace NUMINAMATH_GPT_cosine_sum_sine_half_sum_leq_l523_52333

variable {A B C : ℝ}

theorem cosine_sum_sine_half_sum_leq (h : A + B + C = Real.pi) :
  (Real.cos A + Real.cos B + Real.cos C) ≤ (Real.sin (A / 2) + Real.sin (B / 2) + Real.sin (C / 2)) :=
sorry

end NUMINAMATH_GPT_cosine_sum_sine_half_sum_leq_l523_52333


namespace NUMINAMATH_GPT_solution_set_for_log_inequality_l523_52324

noncomputable def f : ℝ → ℝ := sorry

def isEven (f : ℝ → ℝ) := ∀ x, f (-x) = f x

def isIncreasingOnNonNeg (f : ℝ → ℝ) := ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y

def f_positive_at_third : Prop := f (1 / 3) > 0

theorem solution_set_for_log_inequality
  (hf_even : isEven f)
  (hf_increasing : isIncreasingOnNonNeg f)
  (hf_positive : f_positive_at_third) :
  {x : ℝ | f (Real.log x / Real.log (1/8)) > 0} = {x : ℝ | 0 < x ∧ x < 1/2} ∪ {x : ℝ | 2 < x} := sorry

end NUMINAMATH_GPT_solution_set_for_log_inequality_l523_52324


namespace NUMINAMATH_GPT_ants_harvest_time_l523_52387

theorem ants_harvest_time :
  ∃ h : ℕ, (∀ h : ℕ, 24 - 4 * h = 12) ∧ h = 3 := sorry

end NUMINAMATH_GPT_ants_harvest_time_l523_52387


namespace NUMINAMATH_GPT_tank_capacity_l523_52311

theorem tank_capacity (x : ℝ) (h₁ : 0.25 * x = 60) (h₂ : 0.05 * x = 12) : x = 240 :=
sorry

end NUMINAMATH_GPT_tank_capacity_l523_52311


namespace NUMINAMATH_GPT_exists_root_abs_leq_2_abs_c_div_b_l523_52363

theorem exists_root_abs_leq_2_abs_c_div_b (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h_real_roots : ∃ x1 x2 : ℝ, a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0) :
  ∃ x : ℝ, a * x^2 + b * x + c = 0 ∧ |x| ≤ 2 * |c / b| :=
by
  sorry

end NUMINAMATH_GPT_exists_root_abs_leq_2_abs_c_div_b_l523_52363


namespace NUMINAMATH_GPT_mean_first_second_fifth_sixth_diff_l523_52308

def six_numbers_arithmetic_mean_condition (a1 a2 a3 a4 a5 a6 A : ℝ) :=
  (a1 + a2 + a3 + a4 + a5 + a6) / 6 = A

def mean_first_four_numbers (a1 a2 a3 a4 A : ℝ) :=
  (a1 + a2 + a3 + a4) / 4 = A + 10

def mean_last_four_numbers (a3 a4 a5 a6 A : ℝ) :=
  (a3 + a4 + a5 + a6) / 4 = A - 7

theorem mean_first_second_fifth_sixth_diff (a1 a2 a3 a4 a5 a6 A : ℝ) :
  six_numbers_arithmetic_mean_condition a1 a2 a3 a4 a5 a6 A →
  mean_first_four_numbers a1 a2 a3 a4 A →
  mean_last_four_numbers a3 a4 a5 a6 A →
  ((a1 + a2 + a5 + a6) / 4) = A - 3 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_mean_first_second_fifth_sixth_diff_l523_52308


namespace NUMINAMATH_GPT_convert_base7_to_base2_l523_52300

-- Definitions and conditions
def base7_to_decimal (n : ℕ) : ℕ :=
  2 * 7^1 + 5 * 7^0

def decimal_to_binary (n : ℕ) : ℕ :=
  -- Reversing the binary conversion steps
  -- 1 * 2^4 + 0 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0 = 19
  1 * 2^4 + 0 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0

-- Proof problem
theorem convert_base7_to_base2 : decimal_to_binary (base7_to_decimal 25) = 10011 :=
by {
  sorry
}

end NUMINAMATH_GPT_convert_base7_to_base2_l523_52300


namespace NUMINAMATH_GPT_total_people_in_class_l523_52318

def likes_both (n : ℕ) := n = 5
def likes_only_baseball (n : ℕ) := n = 2
def likes_only_football (n : ℕ) := n = 3
def likes_neither (n : ℕ) := n = 6

theorem total_people_in_class
  (h1 : likes_both n1)
  (h2 : likes_only_baseball n2)
  (h3 : likes_only_football n3)
  (h4 : likes_neither n4) :
  n1 + n2 + n3 + n4 = 16 :=
by 
  sorry

end NUMINAMATH_GPT_total_people_in_class_l523_52318


namespace NUMINAMATH_GPT_Tim_marbles_l523_52361

theorem Tim_marbles (Fred_marbles : ℕ) (Tim_marbles : ℕ) (h1 : Fred_marbles = 110) (h2 : Fred_marbles = 22 * Tim_marbles) : 
  Tim_marbles = 5 :=
by
  sorry

end NUMINAMATH_GPT_Tim_marbles_l523_52361


namespace NUMINAMATH_GPT_original_flow_rate_l523_52392

theorem original_flow_rate (x : ℝ) (h : 2 = 0.6 * x - 1) : x = 5 :=
by
  sorry

end NUMINAMATH_GPT_original_flow_rate_l523_52392


namespace NUMINAMATH_GPT_race_course_length_l523_52325

theorem race_course_length (v : ℝ) (d : ℝ) (h1 : 4 * (d - 69) = d) : d = 92 :=
by
  sorry

end NUMINAMATH_GPT_race_course_length_l523_52325


namespace NUMINAMATH_GPT_sister_ages_l523_52391

theorem sister_ages (x y : ℕ) (h1 : x = y + 4) (h2 : x^3 - y^3 = 988) : y = 7 ∧ x = 11 :=
by
  sorry

end NUMINAMATH_GPT_sister_ages_l523_52391


namespace NUMINAMATH_GPT_ratio_of_area_of_smaller_circle_to_larger_rectangle_l523_52369

noncomputable def ratio_areas (w : ℝ) : ℝ :=
  (3.25 * Real.pi * w^2 / 4) / (1.5 * w^2)

theorem ratio_of_area_of_smaller_circle_to_larger_rectangle (w : ℝ) : 
  ratio_areas w = 13 * Real.pi / 24 := 
by 
  sorry

end NUMINAMATH_GPT_ratio_of_area_of_smaller_circle_to_larger_rectangle_l523_52369


namespace NUMINAMATH_GPT_fraction_meaningful_condition_l523_52364

theorem fraction_meaningful_condition (x : ℝ) : 3 - x ≠ 0 ↔ x ≠ 3 :=
by sorry

end NUMINAMATH_GPT_fraction_meaningful_condition_l523_52364


namespace NUMINAMATH_GPT_true_discount_correct_l523_52330

noncomputable def true_discount (banker_gain : ℝ) (average_rate : ℝ) (time_years : ℝ) : ℝ :=
  let r := average_rate
  let t := time_years
  let exp_factor := Real.exp (-r * t)
  let face_value := banker_gain / (1 - exp_factor)
  face_value - (face_value * exp_factor)

theorem true_discount_correct : 
  true_discount 15.8 0.145 5 = 15.8 := 
by
  sorry

end NUMINAMATH_GPT_true_discount_correct_l523_52330


namespace NUMINAMATH_GPT_checker_on_diagonal_l523_52346

theorem checker_on_diagonal
  (board : ℕ)
  (n_checkers : ℕ)
  (symmetric : (ℕ → ℕ → Prop))
  (diag_check : ∀ i j, symmetric i j -> symmetric j i)
  (num_checkers_odd : Odd n_checkers)
  (board_size : board = 25)
  (checkers : n_checkers = 25) :
  ∃ i, i < 25 ∧ symmetric i i := by
  sorry

end NUMINAMATH_GPT_checker_on_diagonal_l523_52346


namespace NUMINAMATH_GPT_arithmetic_sequence_common_difference_l523_52302

theorem arithmetic_sequence_common_difference (d : ℚ) (a₁ : ℚ) (h : a₁ = -10)
  (h₁ : ∀ n ≥ 10, a₁ + (n - 1) * d > 0) :
  10 / 9 < d ∧ d ≤ 5 / 4 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_common_difference_l523_52302


namespace NUMINAMATH_GPT_isosceles_right_triangle_ratio_l523_52370

theorem isosceles_right_triangle_ratio {a : ℝ} (h_pos : 0 < a) :
  (a + 2 * a) / Real.sqrt (a^2 + a^2) = 3 * Real.sqrt 2 / 2 :=
sorry

end NUMINAMATH_GPT_isosceles_right_triangle_ratio_l523_52370


namespace NUMINAMATH_GPT_nabla_eq_37_l523_52376

def nabla (a b : ℤ) : ℤ := a * b + a - b

theorem nabla_eq_37 : nabla (-5) (-7) = 37 := by
  sorry

end NUMINAMATH_GPT_nabla_eq_37_l523_52376


namespace NUMINAMATH_GPT_matchstick_equality_l523_52378

theorem matchstick_equality :
  abs ((22 : ℝ) / 7 - Real.pi) < 0.1 := 
sorry

end NUMINAMATH_GPT_matchstick_equality_l523_52378


namespace NUMINAMATH_GPT_Angelina_speed_grocery_to_gym_l523_52344

-- Define parameters for distances and times
def distance_home_to_grocery : ℕ := 720
def distance_grocery_to_gym : ℕ := 480
def time_difference : ℕ := 40

-- Define speeds
variable (v : ℕ) -- speed in meters per second from home to grocery
def speed_home_to_grocery := v
def speed_grocery_to_gym := 2 * v

-- Define times using given speeds and distances
def time_home_to_grocery := distance_home_to_grocery / speed_home_to_grocery
def time_grocery_to_gym := distance_grocery_to_gym / speed_grocery_to_gym

-- Proof statement for the problem
theorem Angelina_speed_grocery_to_gym
  (v_pos : 0 < v)
  (condition : time_home_to_grocery - time_difference = time_grocery_to_gym) :
  speed_grocery_to_gym = 24 := by
  sorry

end NUMINAMATH_GPT_Angelina_speed_grocery_to_gym_l523_52344
