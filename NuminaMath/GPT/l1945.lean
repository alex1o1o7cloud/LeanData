import Mathlib

namespace NUMINAMATH_GPT_exists_xy_such_that_x2_add_y2_eq_n_mod_p_p_mod_4_eq_1_implies_n_can_be_0_p_mod_4_eq_3_implies_n_cannot_be_0_l1945_194594

theorem exists_xy_such_that_x2_add_y2_eq_n_mod_p
  (p : ℕ) [Fact (Nat.Prime p)] (n : ℤ)
  (hp1 : p > 5) :
  (∃ x y : ℤ, x ≠ 0 ∧ y ≠ 0 ∧ (x^2 + y^2) % p = n % p) :=
sorry

theorem p_mod_4_eq_1_implies_n_can_be_0
  (p : ℕ) [Fact (Nat.Prime p)] (hp1 : p % 4 = 1) : 
  (∃ x y : ℤ, x ≠ 0 ∧ y ≠ 0 ∧ (x^2 + y^2) % p = 0) :=
sorry

theorem p_mod_4_eq_3_implies_n_cannot_be_0
  (p : ℕ) [Fact (Nat.Prime p)] (hp : p % 4 = 3) :
  ¬(∃ x y : ℤ, x ≠ 0 ∧ y ≠ 0 ∧ (x^2 + y^2) % p = 0) :=
sorry

end NUMINAMATH_GPT_exists_xy_such_that_x2_add_y2_eq_n_mod_p_p_mod_4_eq_1_implies_n_can_be_0_p_mod_4_eq_3_implies_n_cannot_be_0_l1945_194594


namespace NUMINAMATH_GPT_compare_magnitudes_l1945_194537

theorem compare_magnitudes : -0.5 > -0.75 :=
by
  have h1 : |(-0.5: ℝ)| = 0.5 := by norm_num
  have h2 : |(-0.75: ℝ)| = 0.75 := by norm_num
  have h3 : (0.5: ℝ) < 0.75 := by norm_num
  sorry

end NUMINAMATH_GPT_compare_magnitudes_l1945_194537


namespace NUMINAMATH_GPT_delta_zeta_finish_time_l1945_194568

noncomputable def delta_epsilon_zeta_proof_problem (D E Z : ℝ) (k : ℝ) : Prop :=
  (1 / D + 1 / E + 1 / Z = 1 / (D - 4)) ∧
  (1 / D + 1 / E + 1 / Z = 1 / (E - 3.5)) ∧
  (1 / E + 1 / Z = 2 / E) → 
  k = 2

-- Now we prepare the theorem statement
theorem delta_zeta_finish_time (D E Z k : ℝ) (h1 : 1 / D + 1 / E + 1 / Z = 1 / (D - 4))
                                (h2 : 1 / D + 1 / E + 1 / Z = 1 / (E - 3.5))
                                (h3 : 1 / E + 1 / Z = 2 / E) 
                                (h4 : E = 6) :
  k = 2 := 
sorry

end NUMINAMATH_GPT_delta_zeta_finish_time_l1945_194568


namespace NUMINAMATH_GPT_machine_minutes_worked_l1945_194591

-- Definitions based on conditions
def shirts_made_yesterday : ℕ := 9
def shirts_per_minute : ℕ := 3

-- The proof problem statement
theorem machine_minutes_worked (shirts_made_yesterday shirts_per_minute : ℕ) : 
  shirts_made_yesterday / shirts_per_minute = 3 := 
by
  sorry

end NUMINAMATH_GPT_machine_minutes_worked_l1945_194591


namespace NUMINAMATH_GPT_range_of_p_l1945_194502

noncomputable def a_n (p : ℝ) (n : ℕ) : ℝ := -2 * n + p
noncomputable def b_n (n : ℕ) : ℝ := 2 ^ (n - 7)

noncomputable def c_n (p : ℝ) (n : ℕ) : ℝ :=
if a_n p n <= b_n n then a_n p n else b_n n

theorem range_of_p (p : ℝ) :
  (∀ n : ℕ, n ≠ 10 → c_n p 10 > c_n p n) ↔ 24 < p ∧ p < 30 :=
sorry

end NUMINAMATH_GPT_range_of_p_l1945_194502


namespace NUMINAMATH_GPT_total_number_of_boys_in_all_class_sections_is_380_l1945_194522

theorem total_number_of_boys_in_all_class_sections_is_380 :
  let students_section1 := 160
  let students_section2 := 200
  let students_section3 := 240
  let girls_section1 := students_section1 / 4
  let boys_section1 := students_section1 - girls_section1
  let boys_section2 := (3 / 5) * students_section2
  let total_parts := 7 + 5
  let boys_section3 := (7 / total_parts) * students_section3
  boys_section1 + boys_section2 + boys_section3 = 380 :=
sorry

end NUMINAMATH_GPT_total_number_of_boys_in_all_class_sections_is_380_l1945_194522


namespace NUMINAMATH_GPT_each_child_plays_for_90_minutes_l1945_194585

-- Definitions based on the conditions
def total_playing_time : ℕ := 180
def children_playing_at_a_time : ℕ := 3
def total_children : ℕ := 6

-- The proof problem statement
theorem each_child_plays_for_90_minutes :
  (children_playing_at_a_time * total_playing_time) / total_children = 90 := by
  sorry

end NUMINAMATH_GPT_each_child_plays_for_90_minutes_l1945_194585


namespace NUMINAMATH_GPT_tangent_parallel_line_l1945_194593

open Function

def f (x : ℝ) : ℝ := x^4 - x

def f' (x : ℝ) : ℝ := 4 * x^3 - 1

theorem tangent_parallel_line {P : ℝ × ℝ} (hP : ∃ x y, P = (x, y) ∧ f' x = 3) :
  P = (1, 0) := by
  sorry

end NUMINAMATH_GPT_tangent_parallel_line_l1945_194593


namespace NUMINAMATH_GPT_sale_in_third_month_l1945_194501

theorem sale_in_third_month 
  (sale1 sale2 sale4 sale5 sale6 : ℕ) 
  (avg_sale_months : ℕ) 
  (total_sales : ℕ)
  (h1 : sale1 = 6435) 
  (h2 : sale2 = 6927) 
  (h4 : sale4 = 7230) 
  (h5 : sale5 = 6562) 
  (h6 : sale6 = 7991) 
  (h_avg : avg_sale_months = 7000) 
  (h_total : total_sales = 6 * avg_sale_months) 
  : (total_sales - (sale1 + sale2 + sale4 + sale5 + sale6)) = 6855 :=
by
  have sales_sum := sale1 + sale2 + sale4 + sale5 + sale6
  have required_sales := total_sales - sales_sum
  sorry

end NUMINAMATH_GPT_sale_in_third_month_l1945_194501


namespace NUMINAMATH_GPT_calculate_expression_l1945_194508

theorem calculate_expression :
  -1 ^ 2023 + (Real.pi - 3.14) ^ 0 + |(-2 : ℝ)| = 2 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l1945_194508


namespace NUMINAMATH_GPT_crop_yield_solution_l1945_194587

variable (x y : ℝ)

axiom h1 : 3 * x + 6 * y = 4.7
axiom h2 : 5 * x + 3 * y = 5.5

theorem crop_yield_solution :
  x = 0.9 ∧ y = 1/3 :=
by
  sorry

end NUMINAMATH_GPT_crop_yield_solution_l1945_194587


namespace NUMINAMATH_GPT_nat_forms_6n_plus_1_or_5_prod_6n_plus_1_prod_6n_plus_5_prod_6n_plus_1_and_5_l1945_194577

theorem nat_forms_6n_plus_1_or_5 (x : ℕ) (h1 : ¬ (x % 2 = 0) ∧ ¬ (x % 3 = 0)) :
  ∃ n : ℕ, x = 6 * n + 1 ∨ x = 6 * n + 5 := 
sorry

theorem prod_6n_plus_1 (m n : ℕ) :
  (6 * m + 1) * (6 * n + 1) = 6 * (6 * m * n + m + n) + 1 :=
sorry

theorem prod_6n_plus_5 (m n : ℕ) :
  (6 * m + 5) * (6 * n + 5) = 6 * (6 * m * n + 5 * m + 5 * n + 4) + 1 :=
sorry

theorem prod_6n_plus_1_and_5 (m n : ℕ) :
  (6 * m + 1) * (6 * n + 5) = 6 * (6 * m * n + 5 * m + n) + 5 :=
sorry

end NUMINAMATH_GPT_nat_forms_6n_plus_1_or_5_prod_6n_plus_1_prod_6n_plus_5_prod_6n_plus_1_and_5_l1945_194577


namespace NUMINAMATH_GPT_similar_triangles_height_ratio_l1945_194532

-- Given condition: two similar triangles have a similarity ratio of 3:5
def similar_triangles (ratio : ℕ) : Prop := ratio = 3 ∧ ratio = 5

-- Goal: What is the ratio of their corresponding heights?
theorem similar_triangles_height_ratio (r : ℕ) (h : similar_triangles r) :
  r = 3 / 5 :=
sorry

end NUMINAMATH_GPT_similar_triangles_height_ratio_l1945_194532


namespace NUMINAMATH_GPT_gcf_of_36_and_54_l1945_194586

theorem gcf_of_36_and_54 : Nat.gcd 36 54 = 18 := 
by
  sorry

end NUMINAMATH_GPT_gcf_of_36_and_54_l1945_194586


namespace NUMINAMATH_GPT_Granger_payment_correct_l1945_194567

noncomputable def Granger_total_payment : ℝ :=
  let spam_per_can := 3.0
  let peanut_butter_per_jar := 5.0
  let bread_per_loaf := 2.0
  let spam_quantity := 12
  let peanut_butter_quantity := 3
  let bread_quantity := 4
  let spam_dis := 0.1
  let peanut_butter_tax := 0.05
  let spam_cost := spam_quantity * spam_per_can
  let peanut_butter_cost := peanut_butter_quantity * peanut_butter_per_jar
  let bread_cost := bread_quantity * bread_per_loaf
  let spam_discount := spam_dis * spam_cost
  let peanut_butter_tax_amount := peanut_butter_tax * peanut_butter_cost
  let spam_final_cost := spam_cost - spam_discount
  let peanut_butter_final_cost := peanut_butter_cost + peanut_butter_tax_amount
  let total := spam_final_cost + peanut_butter_final_cost + bread_cost
  total

theorem Granger_payment_correct :
  Granger_total_payment = 56.15 :=
by
  sorry

end NUMINAMATH_GPT_Granger_payment_correct_l1945_194567


namespace NUMINAMATH_GPT_solution_exists_l1945_194504

theorem solution_exists :
  ∃ x : ℝ, x = 2 ∧ (-2 * x + 4 = 0) :=
sorry

end NUMINAMATH_GPT_solution_exists_l1945_194504


namespace NUMINAMATH_GPT_solution_set_abs_inequality_l1945_194592

theorem solution_set_abs_inequality (x : ℝ) : |3 - x| + |x - 7| ≤ 8 ↔ 1 ≤ x ∧ x ≤ 9 :=
sorry

end NUMINAMATH_GPT_solution_set_abs_inequality_l1945_194592


namespace NUMINAMATH_GPT_liter_kerosene_cost_friday_l1945_194582

-- Define initial conditions.
def cost_pound_rice_monday : ℚ := 0.36
def cost_dozen_eggs_monday : ℚ := cost_pound_rice_monday
def cost_half_liter_kerosene_monday : ℚ := (8 / 12) * cost_dozen_eggs_monday

-- Define the Wednesday price increase.
def percent_increase_rice : ℚ := 0.20
def cost_pound_rice_wednesday : ℚ := cost_pound_rice_monday * (1 + percent_increase_rice)
def cost_half_liter_kerosene_wednesday : ℚ := cost_half_liter_kerosene_monday * (1 + percent_increase_rice)

-- Define the Friday discount on eggs.
def percent_discount_eggs : ℚ := 0.10
def cost_dozen_eggs_friday : ℚ := cost_dozen_eggs_monday * (1 - percent_discount_eggs)
def cost_per_egg_friday : ℚ := cost_dozen_eggs_friday / 12

-- Define the price calculation for a liter of kerosene on Wednesday.
def cost_liter_kerosene_wednesday : ℚ := 2 * cost_half_liter_kerosene_wednesday

-- Define the final goal.
def cost_liter_kerosene_friday := cost_liter_kerosene_wednesday

theorem liter_kerosene_cost_friday : cost_liter_kerosene_friday = 0.576 := by
  sorry

end NUMINAMATH_GPT_liter_kerosene_cost_friday_l1945_194582


namespace NUMINAMATH_GPT_distinct_real_roots_absolute_sum_l1945_194535

theorem distinct_real_roots_absolute_sum {r1 r2 p : ℝ} (h_root1 : r1 ^ 2 + p * r1 + 7 = 0) 
(h_root2 : r2 ^ 2 + p * r2 + 7 = 0) (h_distinct : r1 ≠ r2) : 
|r1 + r2| > 2 * Real.sqrt 7 := 
sorry

end NUMINAMATH_GPT_distinct_real_roots_absolute_sum_l1945_194535


namespace NUMINAMATH_GPT_polygon_sides_exterior_interior_sum_l1945_194596

theorem polygon_sides_exterior_interior_sum (n : ℕ) (h : ((n - 2) * 180 = 360)) : n = 4 :=
by sorry

end NUMINAMATH_GPT_polygon_sides_exterior_interior_sum_l1945_194596


namespace NUMINAMATH_GPT_store_A_profit_margin_l1945_194533

theorem store_A_profit_margin
  (x y : ℝ)
  (hx : x > 0)
  (hy : y > x)
  (h : (y - x) / x + 0.12 = (y - 0.9 * x) / (0.9 * x)) :
  (y - x) / x = 0.08 :=
by {
  sorry
}

end NUMINAMATH_GPT_store_A_profit_margin_l1945_194533


namespace NUMINAMATH_GPT_center_of_circle_is_2_1_l1945_194559

-- Definition of the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4 * x - 2 * y - 5 = 0

-- Theorem stating the center of the circle
theorem center_of_circle_is_2_1 (x y : ℝ) (h : circle_eq x y) : (x, y) = (2, 1) := sorry

end NUMINAMATH_GPT_center_of_circle_is_2_1_l1945_194559


namespace NUMINAMATH_GPT_pool_depths_l1945_194512

theorem pool_depths (J S Su : ℝ) 
  (h1 : J = 15) 
  (h2 : J = 2 * S + 5) 
  (h3 : Su = J + S - 3) : 
  S = 5 ∧ Su = 17 := 
by 
  -- proof steps go here
  sorry

end NUMINAMATH_GPT_pool_depths_l1945_194512


namespace NUMINAMATH_GPT_max_y_diff_eq_0_l1945_194545

-- Definitions for the given conditions
def eq1 (x : ℝ) : ℝ := 4 - 2 * x + x^2
def eq2 (x : ℝ) : ℝ := 2 + 2 * x + x^2

-- Statement of the proof problem
theorem max_y_diff_eq_0 : 
  (∀ x y, eq1 x = y ∧ eq2 x = y → y = (13 / 4)) →
  ∀ (x1 x2 : ℝ), (∃ y1 y2, eq1 x1 = y1 ∧ eq2 x1 = y1 ∧ eq1 x2 = y2 ∧ eq2 x2 = y2) → 
  (x1 = x2) → (y1 = y2) →
  0 = 0 := 
by
  sorry

end NUMINAMATH_GPT_max_y_diff_eq_0_l1945_194545


namespace NUMINAMATH_GPT_square_fits_in_unit_cube_l1945_194539

theorem square_fits_in_unit_cube (x : ℝ) (h₀ : 0 < x) (h₁ : x < 1) :
  let PQ := Real.sqrt (2 * (1 - x) ^ 2)
  let PS := Real.sqrt (1 + 2 * x ^ 2)
  (PQ > 1.05 ∧ PS > 1.05) :=
by
  sorry

end NUMINAMATH_GPT_square_fits_in_unit_cube_l1945_194539


namespace NUMINAMATH_GPT_find_m_geq_9_l1945_194525

-- Define the real numbers
variables {x m : ℝ}

-- Define the conditions
def p (x : ℝ) := x ≤ 2
def q (x m : ℝ) := x^2 - 2*x + 1 - m^2 ≤ 0

-- Main theorem statement based on the given problem
theorem find_m_geq_9 (m : ℝ) (hm : m > 0) :
  (¬ p x → ¬ q x m) → (p x → q x m) → m ≥ 9 :=
  sorry

end NUMINAMATH_GPT_find_m_geq_9_l1945_194525


namespace NUMINAMATH_GPT_percentage_of_loss_l1945_194583

theorem percentage_of_loss (CP SP : ℕ) (h1 : CP = 1750) (h2 : SP = 1610) : 
  (CP - SP) * 100 / CP = 8 := by
  sorry

end NUMINAMATH_GPT_percentage_of_loss_l1945_194583


namespace NUMINAMATH_GPT_find_max_a_l1945_194505

def f (a x : ℝ) := a * x^3 - x

theorem find_max_a (a : ℝ) (h : ∃ t : ℝ, |f a (t + 2) - f a t| ≤ 2 / 3) :
  a ≤ 4 / 3 :=
sorry

end NUMINAMATH_GPT_find_max_a_l1945_194505


namespace NUMINAMATH_GPT_sandwiches_per_person_l1945_194519

open Nat

theorem sandwiches_per_person (total_sandwiches : ℕ) (total_people : ℕ) (h1 : total_sandwiches = 657) (h2 : total_people = 219) : 
(total_sandwiches / total_people) = 3 :=
by
  -- a proof would go here
  sorry

end NUMINAMATH_GPT_sandwiches_per_person_l1945_194519


namespace NUMINAMATH_GPT_part_one_part_two_l1945_194584

def M (n : ℤ) : ℤ := n - 3
def M_frac (n : ℚ) : ℚ := - (1 / n^2)

theorem part_one 
    : M 28 * M_frac (1/5) = -1 :=
by {
  sorry
}

theorem part_two 
    : -1 / M 39 / (- M_frac (1/6)) = -1 :=
by {
  sorry
}

end NUMINAMATH_GPT_part_one_part_two_l1945_194584


namespace NUMINAMATH_GPT_quad_root_l1945_194556

theorem quad_root (m : ℝ) (β : ℝ) (root_condition : ∃ α : ℝ, α = -5 ∧ (α + β) * (α * β) = x^2 + m * x - 10) : β = 2 :=
by
  sorry

end NUMINAMATH_GPT_quad_root_l1945_194556


namespace NUMINAMATH_GPT_second_candidate_marks_l1945_194553

variable (T : ℝ) (pass_mark : ℝ := 160)

-- Conditions
def condition1 : Prop := 0.20 * T + 40 = pass_mark
def condition2 : Prop := 0.30 * T - pass_mark > 0 

-- The statement we want to prove
theorem second_candidate_marks (h1 : condition1 T) (h2 : condition2 T) : 
  (0.30 * T - pass_mark = 20) :=
by 
  -- Skipping proof steps as per the guidelines
  sorry

end NUMINAMATH_GPT_second_candidate_marks_l1945_194553


namespace NUMINAMATH_GPT_V_product_is_V_form_l1945_194509

noncomputable def V (a b c : ℝ) : ℝ := a^3 + b^3 + c^3 - 3 * a * b * c

theorem V_product_is_V_form (a b c x y z : ℝ) :
  V a b c * V x y z = V (a * x + b * y + c * z) (b * x + c * y + a * z) (c * x + a * y + b * z) := by
  sorry

end NUMINAMATH_GPT_V_product_is_V_form_l1945_194509


namespace NUMINAMATH_GPT_unique_solution_integer_equation_l1945_194528

theorem unique_solution_integer_equation : 
  ∃! (x y z : ℤ), x^2 + y^2 + z^2 = x^2 * y^2 :=
by sorry

end NUMINAMATH_GPT_unique_solution_integer_equation_l1945_194528


namespace NUMINAMATH_GPT_monthly_rent_of_shop_l1945_194549

theorem monthly_rent_of_shop
  (length width : ℕ)
  (annual_rent_per_sq_ft : ℕ)
  (length_def : length = 18)
  (width_def : width = 22)
  (annual_rent_per_sq_ft_def : annual_rent_per_sq_ft = 68) :
  (18 * 22 * 68) / 12 = 2244 := 
by
  sorry

end NUMINAMATH_GPT_monthly_rent_of_shop_l1945_194549


namespace NUMINAMATH_GPT_roots_opposite_k_eq_2_l1945_194524

theorem roots_opposite_k_eq_2 (k : ℝ) :
  (∃ x1 x2 : ℝ, x1 + x2 = 0 ∧ x1 * x2 = -1 ∧ x1 ≠ x2 ∧ x1*x1 + (k-2)*x1 - 1 = 0 ∧ x2*x2 + (k-2)*x2 - 1 = 0) → k = 2 :=
by
  sorry

end NUMINAMATH_GPT_roots_opposite_k_eq_2_l1945_194524


namespace NUMINAMATH_GPT_train_speed_proof_l1945_194547

theorem train_speed_proof :
  (∀ (speed : ℝ), 
    let train_length := 120
    let cross_time := 16
    let total_distance := 240
    let relative_speed := total_distance / cross_time
    let individual_speed := relative_speed / 2
    let speed_kmh := individual_speed * 3.6
    (speed_kmh = 27) → speed = 27
  ) :=
by
  sorry

end NUMINAMATH_GPT_train_speed_proof_l1945_194547


namespace NUMINAMATH_GPT_tank_emptying_time_correct_l1945_194588

noncomputable def tank_emptying_time : ℝ :=
  let initial_volume := 1 / 5
  let fill_rate := 1 / 15
  let empty_rate := 1 / 6
  let combined_rate := fill_rate - empty_rate
  initial_volume / combined_rate

theorem tank_emptying_time_correct :
  tank_emptying_time = 2 :=
by
  -- Proof will be provided here
  sorry

end NUMINAMATH_GPT_tank_emptying_time_correct_l1945_194588


namespace NUMINAMATH_GPT_same_number_assigned_to_each_point_l1945_194565

namespace EqualNumberAssignment

def is_arithmetic_mean (f : ℤ × ℤ → ℕ) (p : ℤ × ℤ) : Prop :=
  let (x, y) := p
  f (x, y) = (f (x + 1, y) + f (x - 1, y) + f (x, y + 1) + f (x, y - 1)) / 4

theorem same_number_assigned_to_each_point (f : ℤ × ℤ → ℕ) :
  (∀ p : ℤ × ℤ, is_arithmetic_mean f p) → ∃ m : ℕ, ∀ p : ℤ × ℤ, f p = m :=
by
  intros h
  sorry

end EqualNumberAssignment

end NUMINAMATH_GPT_same_number_assigned_to_each_point_l1945_194565


namespace NUMINAMATH_GPT_remainder_of_2_pow_87_plus_3_mod_7_l1945_194542

theorem remainder_of_2_pow_87_plus_3_mod_7 : (2^87 + 3) % 7 = 4 := by
  sorry

end NUMINAMATH_GPT_remainder_of_2_pow_87_plus_3_mod_7_l1945_194542


namespace NUMINAMATH_GPT_cost_of_one_dozen_pens_l1945_194546

variable (x : ℝ)

-- Conditions 1 and 2 as assumptions
def pen_cost := 5 * x
def pencil_cost := x

axiom cost_equation  : 3 * pen_cost + 5 * pencil_cost = 200
axiom cost_ratio     : pen_cost / pencil_cost = 5 / 1 -- ratio is given

-- Question and target statement
theorem cost_of_one_dozen_pens : 12 * pen_cost = 600 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_one_dozen_pens_l1945_194546


namespace NUMINAMATH_GPT_houses_with_animals_l1945_194562

theorem houses_with_animals (n A B C x y : ℕ) (h1 : n = 2017) (h2 : A = 1820) (h3 : B = 1651) (h4 : C = 1182) 
    (hx : x = 1182) (hy : y = 619) : x - y = 563 := 
by {
  sorry
}

end NUMINAMATH_GPT_houses_with_animals_l1945_194562


namespace NUMINAMATH_GPT_white_ball_probability_l1945_194555

theorem white_ball_probability (m : ℕ) 
  (initial_black : ℕ := 6) 
  (initial_white : ℕ := 10) 
  (added_white := 14) 
  (probability := 0.8) :
  (10 + added_white) / (16 + added_white) = probability :=
by
  -- no proof required
  sorry

end NUMINAMATH_GPT_white_ball_probability_l1945_194555


namespace NUMINAMATH_GPT_trapezoid_area_l1945_194581

variable (x y : ℝ)

def condition1 : Prop := abs (y - 3 * x) ≥ abs (2 * y + x) ∧ -1 ≤ y - 3 ∧ y - 3 ≤ 1

def condition2 : Prop := (2 * y + y - y + 3 * x) * (2 * y + x + y - 3 * x) ≤ 0 ∧ 2 ≤ y ∧ y ≤ 4

theorem trapezoid_area (h1 : condition1 x y) (h2 : condition2 x y) :
  let A := (3, 2)
  let B := (-1/2, 2)
  let C := (-1, 4)
  let D := (6, 4)
  let S := (1/2) * (2 * (7 + 3.5))
  S = 10.5 :=
sorry

end NUMINAMATH_GPT_trapezoid_area_l1945_194581


namespace NUMINAMATH_GPT_product_of_divisors_sum_l1945_194529

theorem product_of_divisors_sum :
  ∃ (a b c : ℕ), (a ∣ 11^3) ∧ (b ∣ 11^3) ∧ (c ∣ 11^3) ∧ (a ≠ b) ∧ (b ≠ c) ∧ (a ≠ c) ∧ (a * b * c = 11^3) ∧ (a + b + c = 133) :=
sorry

end NUMINAMATH_GPT_product_of_divisors_sum_l1945_194529


namespace NUMINAMATH_GPT_total_movies_purchased_l1945_194541

theorem total_movies_purchased (x : ℕ) (h1 : 17 * x > 0) (h2 : 4 * x > 0) (h3 : 4 * x - 4 > 0) :
  (17 * x) / (4 * x - 4) = 9 / 2 → 17 * x + 4 * x = 378 :=
by 
  intro hab
  sorry

end NUMINAMATH_GPT_total_movies_purchased_l1945_194541


namespace NUMINAMATH_GPT_notification_probability_l1945_194560

theorem notification_probability
  (num_students : ℕ)
  (num_notified_Li : ℕ)
  (num_notified_Zhang : ℕ)
  (prob_Li : ℚ)
  (prob_Zhang : ℚ)
  (h1 : num_students = 10)
  (h2 : num_notified_Li = 4)
  (h3 : num_notified_Zhang = 4)
  (h4 : prob_Li = (4 : ℚ) / 10)
  (h5 : prob_Zhang = (4 : ℚ) / 10) :
  prob_Li + prob_Zhang - prob_Li * prob_Zhang = (16 : ℚ) / 25 := 
by 
  sorry

end NUMINAMATH_GPT_notification_probability_l1945_194560


namespace NUMINAMATH_GPT_sqrt_diff_eq_neg_four_sqrt_five_l1945_194514

theorem sqrt_diff_eq_neg_four_sqrt_five : 
  (Real.sqrt (16 - 8 * Real.sqrt 5) - Real.sqrt (16 + 8 * Real.sqrt 5)) = -4 * Real.sqrt 5 := 
sorry

end NUMINAMATH_GPT_sqrt_diff_eq_neg_four_sqrt_five_l1945_194514


namespace NUMINAMATH_GPT_any_power_ends_in_12890625_l1945_194590

theorem any_power_ends_in_12890625 (a : ℕ) (m k : ℕ) (h : a = 10^m * k + 12890625) : ∀ (n : ℕ), 0 < n → ((a ^ n) % 10^8 = 12890625 % 10^8) :=
by
  intros
  sorry

end NUMINAMATH_GPT_any_power_ends_in_12890625_l1945_194590


namespace NUMINAMATH_GPT_general_term_of_sequence_l1945_194513

variable (a : ℕ → ℕ)
variable (h1 : ∀ m : ℕ, a (m^2) = a m ^ 2)
variable (h2 : ∀ m k : ℕ, a (m^2 + k^2) = a m * a k)

theorem general_term_of_sequence : ∀ n : ℕ, n > 0 → a n = 1 :=
by
  intros n hn
  sorry

end NUMINAMATH_GPT_general_term_of_sequence_l1945_194513


namespace NUMINAMATH_GPT_incorrect_equation_l1945_194580

noncomputable def x : ℂ := (-1 + Real.sqrt 3 * Complex.I) / 2
noncomputable def y : ℂ := (-1 - Real.sqrt 3 * Complex.I) / 2

theorem incorrect_equation : x^9 + y^9 ≠ -1 := sorry

end NUMINAMATH_GPT_incorrect_equation_l1945_194580


namespace NUMINAMATH_GPT_hemisphere_surface_area_l1945_194517

theorem hemisphere_surface_area (base_area : ℝ) (r : ℝ) (total_surface_area : ℝ) 
(h1: base_area = 64 * Real.pi) 
(h2: r^2 = 64)
(h3: total_surface_area = base_area + 2 * Real.pi * r^2) : 
total_surface_area = 192 * Real.pi := 
sorry

end NUMINAMATH_GPT_hemisphere_surface_area_l1945_194517


namespace NUMINAMATH_GPT_radius_any_positive_real_l1945_194563

theorem radius_any_positive_real (r : ℝ) (h₁ : r > 0) 
    (h₂ : r * (2 * Real.pi * r) = 2 * Real.pi * r^2) : True :=
by
  sorry

end NUMINAMATH_GPT_radius_any_positive_real_l1945_194563


namespace NUMINAMATH_GPT_problem_statement_l1945_194540

noncomputable def square : ℝ := sorry -- We define a placeholder
noncomputable def pentagon : ℝ := sorry -- We define a placeholder

axiom eq1 : 2 * square + 4 * pentagon = 25
axiom eq2 : 3 * square + 3 * pentagon = 22

theorem problem_statement : 4 * pentagon = 20.67 := 
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1945_194540


namespace NUMINAMATH_GPT_value_of_3x_plus_5y_l1945_194564

variable (x y : ℚ)

theorem value_of_3x_plus_5y
  (h1 : x + 4 * y = 5) 
  (h2 : 5 * x + 6 * y = 7) : 3 * x + 5 * y = 6 := 
sorry

end NUMINAMATH_GPT_value_of_3x_plus_5y_l1945_194564


namespace NUMINAMATH_GPT_weight_of_new_person_l1945_194523

theorem weight_of_new_person 
  (average_weight_first_20 : ℕ → ℕ → ℕ)
  (new_average_weight : ℕ → ℕ → ℕ) 
  (total_weight_21 : ℕ): 
  (average_weight_first_20 1200 20 = 60) → 
  (new_average_weight (1200 + total_weight_21) 21 = 55) → 
  total_weight_21 = 55 := 
by 
  intros 
  sorry

end NUMINAMATH_GPT_weight_of_new_person_l1945_194523


namespace NUMINAMATH_GPT_sum_of_other_two_angles_is_108_l1945_194510

theorem sum_of_other_two_angles_is_108 (A B C : Type) (angleA angleB angleC : ℝ) 
  (h_angle_sum : angleA + angleB + angleC = 180) (h_angleB : angleB = 72) :
  angleA + angleC = 108 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_other_two_angles_is_108_l1945_194510


namespace NUMINAMATH_GPT_arithmetic_sequence_n_value_l1945_194598

theorem arithmetic_sequence_n_value
  (a : ℕ → ℚ)
  (h1 : a 1 = 1 / 3)
  (h2 : a 2 + a 5 = 4)
  (h3 : a n = 33)
  : n = 50 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_n_value_l1945_194598


namespace NUMINAMATH_GPT_intersection_M_N_l1945_194518

noncomputable def M := {x : ℝ | x > 1}
noncomputable def N := {x : ℝ | x < 2}

theorem intersection_M_N :
  M ∩ N = {x : ℝ | 1 < x ∧ x < 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l1945_194518


namespace NUMINAMATH_GPT_bottles_left_l1945_194538

variable (initial_bottles : ℕ) (jason_bottles : ℕ) (harry_bottles : ℕ)

theorem bottles_left (h1 : initial_bottles = 35) (h2 : jason_bottles = 5) (h3 : harry_bottles = 6) :
    initial_bottles - (jason_bottles + harry_bottles) = 24 := by
  sorry

end NUMINAMATH_GPT_bottles_left_l1945_194538


namespace NUMINAMATH_GPT_area_triangle_ABC_area_figure_DEFGH_area_triangle_JKL_l1945_194521

-- (a) Proving the area of triangle ABC
theorem area_triangle_ABC (AB BC : ℝ) (hAB : AB = 2) (hBC : BC = 3) (h_right : true) : 
  (1 / 2) * AB * BC = 3 := sorry

-- (b) Proving the area of figure DEFGH
theorem area_figure_DEFGH (DH HG : ℝ) (hDH : DH = 5) (hHG : HG = 5) (triangle_area : ℝ) (hEPF : triangle_area = 3) : 
  DH * HG - triangle_area = 22 := sorry

-- (c) Proving the area of triangle JKL 
theorem area_triangle_JKL (side_area : ℝ) (h_side : side_area = 25) 
  (area_JSK : ℝ) (h_JSK : area_JSK = 3) 
  (area_LQJ : ℝ) (h_LQJ : area_LQJ = 15/2) 
  (area_LRK : ℝ) (h_LRK : area_LRK = 5) : 
  side_area - area_JSK - area_LQJ - area_LRK = 19/2 := sorry

end NUMINAMATH_GPT_area_triangle_ABC_area_figure_DEFGH_area_triangle_JKL_l1945_194521


namespace NUMINAMATH_GPT_sum_of_squares_not_divisible_by_4_or_8_l1945_194558

theorem sum_of_squares_not_divisible_by_4_or_8 (n : ℤ) (h : n % 2 = 1) :
  let a := n - 2
  let b := n
  let c := n + 2
  let sum_squares := a^2 + b^2 + c^2
  ¬(4 ∣ sum_squares ∨ 8 ∣ sum_squares) :=
by
  let a := n - 2
  let b := n
  let c := n + 2
  let sum_squares := a^2 + b^2 + c^2
  sorry

end NUMINAMATH_GPT_sum_of_squares_not_divisible_by_4_or_8_l1945_194558


namespace NUMINAMATH_GPT_triangle_inequality_l1945_194551

noncomputable def f (K : ℝ) (x : ℝ) : ℝ :=
  (x^4 + K * x^2 + 1) / (x^4 + x^2 + 1)

theorem triangle_inequality (K : ℝ) (a b c : ℝ) :
  (-1 / 2) < K ∧ K < 4 → ∃ (A B C : ℝ), A = f K a ∧ B = f K b ∧ C = f K c ∧ A + B > C ∧ A + C > B ∧ B + C > A :=
by
  sorry

end NUMINAMATH_GPT_triangle_inequality_l1945_194551


namespace NUMINAMATH_GPT_num_girls_went_to_spa_l1945_194536

-- Define the condition that each girl has 20 nails
def nails_per_girl : ℕ := 20

-- Define the total number of nails polished
def total_nails_polished : ℕ := 40

-- Define the number of girls
def number_of_girls : ℕ := total_nails_polished / nails_per_girl

-- The theorem we want to prove
theorem num_girls_went_to_spa : number_of_girls = 2 :=
by
  unfold number_of_girls
  unfold total_nails_polished
  unfold nails_per_girl
  sorry

end NUMINAMATH_GPT_num_girls_went_to_spa_l1945_194536


namespace NUMINAMATH_GPT_fraction_calls_processed_by_team_B_l1945_194506

variable (A B C_A C_B : ℕ)

theorem fraction_calls_processed_by_team_B 
  (h1 : A = (5 / 8) * B)
  (h2 : C_A = (2 / 5) * C_B) :
  (B * C_B) / ((A * C_A) + (B * C_B)) = 8 / 9 := by
  sorry

end NUMINAMATH_GPT_fraction_calls_processed_by_team_B_l1945_194506


namespace NUMINAMATH_GPT_regular_decagon_interior_angle_degree_measure_l1945_194534

theorem regular_decagon_interior_angle_degree_measure :
  ∀ (n : ℕ), n = 10 → (2 * 180 / n : ℝ) = 144 :=
by
  sorry

end NUMINAMATH_GPT_regular_decagon_interior_angle_degree_measure_l1945_194534


namespace NUMINAMATH_GPT_tim_total_payment_correct_l1945_194597

-- Define the conditions stated in the problem
def doc_visit_cost : ℝ := 300
def insurance_coverage_percent : ℝ := 0.75
def cat_visit_cost : ℝ := 120
def pet_insurance_coverage : ℝ := 60

-- Define the amounts covered by insurance 
def insurance_coverage_amount : ℝ := doc_visit_cost * insurance_coverage_percent
def tim_payment_for_doc_visit : ℝ := doc_visit_cost - insurance_coverage_amount
def tim_payment_for_cat_visit : ℝ := cat_visit_cost - pet_insurance_coverage

-- Define the total payment Tim needs to make
def tim_total_payment : ℝ := tim_payment_for_doc_visit + tim_payment_for_cat_visit

-- State the main theorem
theorem tim_total_payment_correct : tim_total_payment = 135 := by
  sorry

end NUMINAMATH_GPT_tim_total_payment_correct_l1945_194597


namespace NUMINAMATH_GPT_find_divisor_nearest_to_3105_l1945_194578

def nearest_divisible_number (n : ℕ) (d : ℕ) : ℕ :=
  if n % d = 0 then n else n + d - (n % d)

theorem find_divisor_nearest_to_3105 (d : ℕ) (h : nearest_divisible_number 3105 d = 3108) : d = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_divisor_nearest_to_3105_l1945_194578


namespace NUMINAMATH_GPT_katy_read_books_l1945_194511

theorem katy_read_books (juneBooks : ℕ) (julyBooks : ℕ) (augustBooks : ℕ)
  (H1 : juneBooks = 8)
  (H2 : julyBooks = 2 * juneBooks)
  (H3 : augustBooks = julyBooks - 3) :
  juneBooks + julyBooks + augustBooks = 37 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_katy_read_books_l1945_194511


namespace NUMINAMATH_GPT_dot_product_a_b_l1945_194569

open Real

noncomputable def cos_deg (x : ℝ) := cos (x * π / 180)
noncomputable def sin_deg (x : ℝ) := sin (x * π / 180)

theorem dot_product_a_b :
  let a_magnitude := 2 * cos_deg 15
  let b_magnitude := 4 * sin_deg 15
  let angle_ab := 30
  a_magnitude * b_magnitude * cos_deg angle_ab = sqrt 3 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_dot_product_a_b_l1945_194569


namespace NUMINAMATH_GPT_geometric_sequence_sum_l1945_194544

theorem geometric_sequence_sum (a : ℕ → ℝ) (r : ℝ) 
    (h : ∀ n : ℕ, a (n + 1) = a n * r) 
    (h1 : a 1 + a 2 = 40) 
    (h2 : a 3 + a 4 = 60) : 
    a 7 + a 8 = 135 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l1945_194544


namespace NUMINAMATH_GPT_sarah_apples_calc_l1945_194574

variable (brother_apples : ℕ)
variable (sarah_apples : ℕ)
variable (multiplier : ℕ)

theorem sarah_apples_calc
  (h1 : brother_apples = 9)
  (h2 : multiplier = 5)
  (h3 : sarah_apples = multiplier * brother_apples) : sarah_apples = 45 := by
  sorry

end NUMINAMATH_GPT_sarah_apples_calc_l1945_194574


namespace NUMINAMATH_GPT_total_lunch_bill_l1945_194552

theorem total_lunch_bill (cost_hotdog cost_salad : ℝ) (h1 : cost_hotdog = 5.36) (h2 : cost_salad = 5.10) : 
  cost_hotdog + cost_salad = 10.46 := 
by 
  sorry

end NUMINAMATH_GPT_total_lunch_bill_l1945_194552


namespace NUMINAMATH_GPT_natural_numbers_satisfying_conditions_l1945_194516

variable (a b : ℕ)

theorem natural_numbers_satisfying_conditions :
  (90 < a + b ∧ a + b < 100) ∧ (0.9 < (a : ℝ) / b ∧ (a : ℝ) / b < 0.91) ↔ (a = 46 ∧ b = 51) ∨ (a = 47 ∧ b = 52) := by
  sorry

end NUMINAMATH_GPT_natural_numbers_satisfying_conditions_l1945_194516


namespace NUMINAMATH_GPT_max_value_fraction_l1945_194520

theorem max_value_fraction {a b c : ℝ} (h1 : c = Real.sqrt (a^2 + b^2)) 
  (h2 : a > 0) (h3 : b > 0) (A : ℝ) (hA : A = 1 / 2 * a * b) :
  ∃ x : ℝ, x = (a + b + A) / c ∧ x ≤ (5 / 4) * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_max_value_fraction_l1945_194520


namespace NUMINAMATH_GPT_total_cakes_needed_l1945_194576

theorem total_cakes_needed (C : ℕ) (h : C / 4 - C / 12 = 10) : C = 60 := by
  sorry

end NUMINAMATH_GPT_total_cakes_needed_l1945_194576


namespace NUMINAMATH_GPT_proof_complex_ratio_l1945_194589

noncomputable def condition1 (x y : ℂ) (k : ℝ) : Prop :=
  (x + k * y) / (x - k * y) + (x - k * y) / (x + k * y) = 1

theorem proof_complex_ratio (x y : ℂ) (k : ℝ) (h : condition1 x y k) :
  (x^6 + y^6) / (x^6 - y^6) + (x^6 - y^6) / (x^6 + y^6) = (41 / 20 : ℂ) :=
by 
  sorry

end NUMINAMATH_GPT_proof_complex_ratio_l1945_194589


namespace NUMINAMATH_GPT_circle_properties_l1945_194561

theorem circle_properties (m : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 - 2 * m * x - 4 * y + 5 * m = 0) →
  (m < 1 ∨ m > 4) ∧
  (m = -2 → ∃ d : ℝ, d = 2 * Real.sqrt (18 - 5)) :=
by
  sorry

end NUMINAMATH_GPT_circle_properties_l1945_194561


namespace NUMINAMATH_GPT_radar_placement_coverage_l1945_194527

noncomputable def max_distance_radars (r : ℝ) (n : ℕ) : ℝ :=
  r / Real.sin (Real.pi / n)

noncomputable def coverage_ring_area (r : ℝ) (width : ℝ) (n : ℕ) : ℝ :=
  (1440 * Real.pi) / Real.tan (Real.pi / n)

theorem radar_placement_coverage :
  let r := 41
  let width := 18
  let n := 7
  max_distance_radars r n = 40 / Real.sin (Real.pi / 7) ∧
  coverage_ring_area r width n = (1440 * Real.pi) / Real.tan (Real.pi / 7) :=
by
  sorry

end NUMINAMATH_GPT_radar_placement_coverage_l1945_194527


namespace NUMINAMATH_GPT_range_of_a_l1945_194595

noncomputable def f : ℝ → ℝ := sorry

variables (a : ℝ)
variable (is_even : ∀ x : ℝ, f (x) = f (-x)) -- f is even
variable (monotonic_incr : ∀ x y : ℝ, 0 ≤ x → x ≤ y → f x ≤ f y) -- f is monotonically increasing in [0, +∞)

theorem range_of_a
  (h : f (Real.log a / Real.log 2) + f (Real.log (1/a) / Real.log 2) ≤ 2 * f 1) : 
  1 / 2 ≤ a ∧ a ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1945_194595


namespace NUMINAMATH_GPT_ice_cream_to_afford_games_l1945_194507

theorem ice_cream_to_afford_games :
  let game_cost := 60
  let ice_cream_price := 5
  (game_cost * 2) / ice_cream_price = 24 :=
by
  let game_cost := 60
  let ice_cream_price := 5
  show (game_cost * 2) / ice_cream_price = 24
  sorry

end NUMINAMATH_GPT_ice_cream_to_afford_games_l1945_194507


namespace NUMINAMATH_GPT_range_of_m_l1945_194526

noncomputable def unique_zero_point (m : ℝ) : Prop :=
  ∀ x : ℝ, m * (1/4)^x - (1/2)^x + 1 = 0 → ∀ x' : ℝ, m * (1/4)^x' - (1/2)^x' + 1 = 0 → x = x'

theorem range_of_m (m : ℝ) : unique_zero_point m → (m ≤ 0 ∨ m = 1/4) :=
sorry

end NUMINAMATH_GPT_range_of_m_l1945_194526


namespace NUMINAMATH_GPT_vampire_pints_per_person_l1945_194550

-- Definitions based on conditions
def gallons_per_week : ℕ := 7
def pints_per_gallon : ℕ := 8
def days_per_week : ℕ := 7
def people_per_day : ℕ := 4

-- The statement to be proven
theorem vampire_pints_per_person :
  (gallons_per_week * pints_per_gallon) / (days_per_week * people_per_day) = 2 :=
by
  sorry

end NUMINAMATH_GPT_vampire_pints_per_person_l1945_194550


namespace NUMINAMATH_GPT_abc_le_one_eighth_l1945_194571

theorem abc_le_one_eighth (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a / (1 + a) + b / (1 + b) + c / (1 + c) = 1) : 
  a * b * c ≤ 1 / 8 := 
by
  sorry

end NUMINAMATH_GPT_abc_le_one_eighth_l1945_194571


namespace NUMINAMATH_GPT_negation_equiv_l1945_194531

-- Define the proposition that the square of all real numbers is positive
def pos_of_all_squares : Prop := ∀ x : ℝ, x^2 > 0

-- Define the negation of the proposition
def neg_pos_of_all_squares : Prop := ∃ x : ℝ, x^2 ≤ 0

theorem negation_equiv (h : ¬ pos_of_all_squares) : neg_pos_of_all_squares :=
  sorry

end NUMINAMATH_GPT_negation_equiv_l1945_194531


namespace NUMINAMATH_GPT_problem_HMMT_before_HMT_l1945_194557
noncomputable def probability_of_sequence (seq: List Char) : ℚ := sorry
def probability_H : ℚ := 1 / 3
def probability_M : ℚ := 1 / 3
def probability_T : ℚ := 1 / 3

theorem problem_HMMT_before_HMT : probability_of_sequence ['H', 'M', 'M', 'T'] = 1 / 4 :=
sorry

end NUMINAMATH_GPT_problem_HMMT_before_HMT_l1945_194557


namespace NUMINAMATH_GPT_root_interval_l1945_194515

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1) + 2 * x - 1

theorem root_interval : ∃ m : ℝ, f m = 0 ∧ 0 < m ∧ m < 1 :=
by
  have h_decreasing : ∀ x y : ℝ, x < y → f x < f y :=
    sorry -- Proof that f is increasing on (-1, +∞)
  have h_f0 : f 0 = -1 := by
    sorry -- Calculation that f(0) = -1
  have h_f1 : f 1 = Real.log 2 + 1 := by
    sorry -- Calculation that f(1) = ln(2) + 1
  have h_exist_root : ∃ m : ℝ, f m = 0 ∧ 0 < m ∧ m < 1 :=
    by
      sorry -- Existence of a root in (0,1)
  exact h_exist_root

end NUMINAMATH_GPT_root_interval_l1945_194515


namespace NUMINAMATH_GPT_find_costs_l1945_194543

theorem find_costs (a b : ℝ) (h1 : a - b = 3) (h2 : 3 * b - 2 * a = 3) : a = 12 ∧ b = 9 :=
sorry

end NUMINAMATH_GPT_find_costs_l1945_194543


namespace NUMINAMATH_GPT_joint_purchases_popular_l1945_194500

-- Define the conditions stating what makes joint purchases feasible
structure Conditions where
  cost_saving : Prop  -- Joint purchases allow significant cost savings.
  shared_overhead : Prop  -- Overhead costs are distributed among all members.
  collective_quality_assessment : Prop  -- Enhanced quality assessment via collective feedback.
  community_trust : Prop  -- Trust within the community encourages honest feedback.

-- Define the proposition stating the popularity of joint purchases
theorem joint_purchases_popular (cond : Conditions) : 
  cond.cost_saving ∧ cond.shared_overhead ∧ cond.collective_quality_assessment ∧ cond.community_trust → 
  Prop := 
by 
  intro h
  sorry

end NUMINAMATH_GPT_joint_purchases_popular_l1945_194500


namespace NUMINAMATH_GPT_abs_neg_2035_l1945_194548

theorem abs_neg_2035 : abs (-2035) = 2035 := 
by {
  sorry
}

end NUMINAMATH_GPT_abs_neg_2035_l1945_194548


namespace NUMINAMATH_GPT_find_a_l1945_194579

theorem find_a 
  (x y a : ℝ)
  (h₁ : x - 3 ≤ 0)
  (h₂ : y - a ≤ 0)
  (h₃ : x + y ≥ 0)
  (h₄ : ∃ (x y : ℝ), 2*x + y = 10): a = 4 :=
sorry

end NUMINAMATH_GPT_find_a_l1945_194579


namespace NUMINAMATH_GPT_avg_cost_apple_tv_200_l1945_194572

noncomputable def average_cost_apple_tv (iphones_sold ipads_sold apple_tvs_sold iphone_cost ipad_cost overall_avg_cost: ℝ) : ℝ :=
  (overall_avg_cost * (iphones_sold + ipads_sold + apple_tvs_sold) - (iphones_sold * iphone_cost + ipads_sold * ipad_cost)) / apple_tvs_sold

theorem avg_cost_apple_tv_200 :
  let iphones_sold := 100
  let ipads_sold := 20
  let apple_tvs_sold := 80
  let iphone_cost := 1000
  let ipad_cost := 900
  let overall_avg_cost := 670
  average_cost_apple_tv iphones_sold ipads_sold apple_tvs_sold iphone_cost ipad_cost overall_avg_cost = 200 :=
by
  sorry

end NUMINAMATH_GPT_avg_cost_apple_tv_200_l1945_194572


namespace NUMINAMATH_GPT_probability_calculations_l1945_194570

-- Define the number of students
def total_students : ℕ := 2006

-- Number of students eliminated in the first step
def eliminated_students : ℕ := 6

-- Number of students remaining after elimination
def remaining_students : ℕ := total_students - eliminated_students

-- Number of students to be selected in the second step
def selected_students : ℕ := 50

-- Calculate the probability of a specific student being eliminated
def elimination_probability := (6 : ℚ) / total_students

-- Calculate the probability of a specific student being selected from the remaining students
def selection_probability := (50 : ℚ) / remaining_students

-- The theorem to prove our equivalent proof problem
theorem probability_calculations :
  elimination_probability = (3 : ℚ) / 1003 ∧
  selection_probability = (25 : ℚ) / 1003 :=
by
  sorry

end NUMINAMATH_GPT_probability_calculations_l1945_194570


namespace NUMINAMATH_GPT_john_task_completion_time_l1945_194503

/-- John can complete a task alone in 18 days given the conditions. -/
theorem john_task_completion_time :
  ∀ (John Jane taskDays : ℝ), 
    Jane = 12 → 
    taskDays = 10.8 → 
    (10.8 - 6) * (1 / 12) + 10.8 * (1 / John) = 1 → 
    John = 18 :=
by
  intros John Jane taskDays hJane hTaskDays hWorkDone
  sorry

end NUMINAMATH_GPT_john_task_completion_time_l1945_194503


namespace NUMINAMATH_GPT_jerry_age_is_10_l1945_194554

-- Define the ages of Mickey and Jerry
def MickeyAge : ℝ := 20
def mickey_eq_jerry (JerryAge : ℝ) : Prop := MickeyAge = 2.5 * JerryAge - 5

theorem jerry_age_is_10 : ∃ JerryAge : ℝ, mickey_eq_jerry JerryAge ∧ JerryAge = 10 :=
by
  -- By solving the equation MickeyAge = 2.5 * JerryAge - 5,
  -- we can find that Jerry's age must be 10.
  use 10
  sorry

end NUMINAMATH_GPT_jerry_age_is_10_l1945_194554


namespace NUMINAMATH_GPT_difference_between_c_and_a_l1945_194573

variable (a b c : ℝ)

theorem difference_between_c_and_a (h1 : (a + b) / 2 = 30) (h2 : c - a = 60) : c - a = 60 :=
by
  exact h2

end NUMINAMATH_GPT_difference_between_c_and_a_l1945_194573


namespace NUMINAMATH_GPT_stable_points_of_g_fixed_points_subset_stable_points_range_of_a_l1945_194530

-- Definitions of fixed points and stable points
def is_fixed_point(f : ℝ → ℝ) (x : ℝ) : Prop := f x = x
def is_stable_point(f : ℝ → ℝ) (x : ℝ) : Prop := f (f x) = x 

-- Problem 1: Stable points of g(x) = 2x - 1
def g (x : ℝ) : ℝ := 2 * x - 1

theorem stable_points_of_g : {x : ℝ | is_stable_point g x} = {1} :=
sorry

-- Problem 2: Prove A ⊂ B for any function f
theorem fixed_points_subset_stable_points (f : ℝ → ℝ) : 
  {x : ℝ | is_fixed_point f x} ⊆ {x : ℝ | is_stable_point f x} :=
sorry

-- Problem 3: Range of a for f(x) = ax^2 - 1 when A = B ≠ ∅
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 1

theorem range_of_a (a : ℝ) (h : ∃ x, is_fixed_point (f a) x ∧ is_stable_point (f a) x):
  - (1/4 : ℝ) ≤ a ∧ a ≤ (3/4 : ℝ) :=
sorry

end NUMINAMATH_GPT_stable_points_of_g_fixed_points_subset_stable_points_range_of_a_l1945_194530


namespace NUMINAMATH_GPT_area_of_figure_l1945_194575
-- Import necessary libraries

-- Define the conditions as functions/constants
def length_left : ℕ := 7
def width_top : ℕ := 6
def height_middle : ℕ := 3
def width_middle : ℕ := 4
def height_right : ℕ := 5
def width_right : ℕ := 5

-- State the problem as a theorem
theorem area_of_figure : 
  (length_left * width_top) + 
  (width_middle * height_middle) + 
  (width_right * height_right) = 79 := 
  by
  sorry

end NUMINAMATH_GPT_area_of_figure_l1945_194575


namespace NUMINAMATH_GPT_natural_number_triplets_l1945_194566

theorem natural_number_triplets (x y z : ℕ) : 
  3^x + 4^y = 5^z → 
  (x = 2 ∧ y = 2 ∧ z = 2) ∨ (x = 0 ∧ y = 1 ∧ z = 1) :=
by 
  sorry

end NUMINAMATH_GPT_natural_number_triplets_l1945_194566


namespace NUMINAMATH_GPT_range_of_a_l1945_194599

open Set

def A (a : ℝ) : Set ℝ := { x | a - 1 ≤ x ∧ x ≤ 2 * a + 1 }
def B : Set ℝ := { x | -2 ≤ x ∧ x ≤ 4 }

theorem range_of_a (a : ℝ) (h : A a ∪ B = B) : a ∈ Iio (-2) ∪ Icc (-1) (3 / 2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1945_194599
