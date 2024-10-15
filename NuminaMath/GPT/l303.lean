import Mathlib

namespace NUMINAMATH_GPT_number_of_zeros_of_f_l303_30375

noncomputable def f (x : ℝ) := (1 / 3) * x ^ 3 - x ^ 2 - 3 * x + 9

theorem number_of_zeros_of_f : ∃ (z : ℕ), z = 2 ∧ ∀ x : ℝ, (f x = 0 → x = -3 ∨ x = -2 / 3 ∨ x = 1 ∨ x = 3) := 
sorry

end NUMINAMATH_GPT_number_of_zeros_of_f_l303_30375


namespace NUMINAMATH_GPT_minimum_expr_value_l303_30332

noncomputable def expr_min_value (a : ℝ) (h : a > 1) : ℝ :=
  a + 2 / (a - 1)

theorem minimum_expr_value (a : ℝ) (h : a > 1) :
  expr_min_value a h = 1 + 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_minimum_expr_value_l303_30332


namespace NUMINAMATH_GPT_slope_tangent_at_pi_div_six_l303_30343

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * x - 2 * Real.cos x

theorem slope_tangent_at_pi_div_six : (deriv f π / 6) = 3 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_slope_tangent_at_pi_div_six_l303_30343


namespace NUMINAMATH_GPT_negation_of_p_l303_30355

def p := ∀ x, x ≤ 0 → Real.exp x ≤ 1

theorem negation_of_p : ¬ p ↔ ∃ x, x ≤ 0 ∧ Real.exp x > 1 := 
by
  sorry

end NUMINAMATH_GPT_negation_of_p_l303_30355


namespace NUMINAMATH_GPT_volume_of_triangular_prism_l303_30394

theorem volume_of_triangular_prism (S_side_face : ℝ) (distance : ℝ) :
  ∃ (Volume_prism : ℝ), Volume_prism = 1/2 * (S_side_face * distance) :=
by sorry

end NUMINAMATH_GPT_volume_of_triangular_prism_l303_30394


namespace NUMINAMATH_GPT_tangent_parallel_to_line_l303_30348

noncomputable def f (x : ℝ) : ℝ := x^3 + x - 2

theorem tangent_parallel_to_line :
  ∃ a b : ℝ, (f a = b) ∧ (3 * a^2 + 1 = 4) ∧ (P = (1, 0) ∨ P = (-1, -4)) :=
by
  sorry

end NUMINAMATH_GPT_tangent_parallel_to_line_l303_30348


namespace NUMINAMATH_GPT_ned_weekly_sales_l303_30395

-- Define the conditions given in the problem
def normal_mouse_price : ℝ := 120
def normal_keyboard_price : ℝ := 80
def normal_scissor_price : ℝ := 30

def lt_hand_mouse_price := normal_mouse_price * 1.3
def lt_hand_keyboard_price := normal_keyboard_price * 1.2
def lt_hand_scissor_price := normal_scissor_price * 1.5

def lt_hand_mouse_daily_sales : ℝ := 25 * lt_hand_mouse_price
def lt_hand_keyboard_daily_sales : ℝ := 10 * lt_hand_keyboard_price
def lt_hand_scissor_daily_sales : ℝ := 15 * lt_hand_scissor_price

def total_daily_sales := lt_hand_mouse_daily_sales + lt_hand_keyboard_daily_sales + lt_hand_scissor_daily_sales
def days_open_per_week : ℝ := 4

def weekly_sales := total_daily_sales * days_open_per_week

-- The theorem to prove
theorem ned_weekly_sales : weekly_sales = 22140 := by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_ned_weekly_sales_l303_30395


namespace NUMINAMATH_GPT_find_physics_marks_l303_30340

variable (P C M : ℕ)

theorem find_physics_marks
  (h1 : P + C + M = 225)
  (h2 : P + M = 180)
  (h3 : P + C = 140) : 
  P = 95 :=
by
  sorry

end NUMINAMATH_GPT_find_physics_marks_l303_30340


namespace NUMINAMATH_GPT_fraction_simplification_l303_30311

theorem fraction_simplification :
  (20 + 16 * 20) / (20 * 16) = 17 / 16 :=
by
  sorry

end NUMINAMATH_GPT_fraction_simplification_l303_30311


namespace NUMINAMATH_GPT_tire_radius_increase_l303_30308

noncomputable def radius_increase (initial_radius : ℝ) (odometer_initial : ℝ) (odometer_winter : ℝ) : ℝ :=
  let rotations := odometer_initial / ((2 * Real.pi * initial_radius) / 63360)
  let winter_circumference := (odometer_winter / rotations) * 63360
  let new_radius := winter_circumference / (2 * Real.pi)
  new_radius - initial_radius

theorem tire_radius_increase : radius_increase 16 520 505 = 0.32 := by
  sorry

end NUMINAMATH_GPT_tire_radius_increase_l303_30308


namespace NUMINAMATH_GPT_min_questions_30_cards_min_questions_31_cards_min_questions_32_cards_min_questions_50_cards_circle_l303_30304

-- Statements for minimum questions required for different number of cards 

theorem min_questions_30_cards (cards : Fin 30 → Int) (h : ∀ i, cards i = 1 ∨ cards i = -1) :
  ∃ n, n = 10 :=
by
  sorry

theorem min_questions_31_cards (cards : Fin 31 → Int) (h : ∀ i, cards i = 1 ∨ cards i = -1) :
  ∃ n, n = 11 :=
by
  sorry

theorem min_questions_32_cards (cards : Fin 32 → Int) (h : ∀ i, cards i = 1 ∨ cards i = -1) :
  ∃ n, n = 12 :=
by
  sorry

theorem min_questions_50_cards_circle (cards : Fin 50 → Int) (h : ∀ i, cards i = 1 ∨ cards i = -1) :
  ∃ n, n = 50 :=
by
  sorry

end NUMINAMATH_GPT_min_questions_30_cards_min_questions_31_cards_min_questions_32_cards_min_questions_50_cards_circle_l303_30304


namespace NUMINAMATH_GPT_focus_of_parabola_l303_30333

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 4 * y

-- Define the coordinates of the focus
def is_focus (x y : ℝ) : Prop := (x = 0) ∧ (y = 1)

-- The theorem statement
theorem focus_of_parabola : 
  (∃ x y : ℝ, parabola x y ∧ is_focus x y) :=
sorry

end NUMINAMATH_GPT_focus_of_parabola_l303_30333


namespace NUMINAMATH_GPT_people_came_later_l303_30356

theorem people_came_later (lollipop_ratio initial_people lollipops : ℕ) 
  (h1 : lollipop_ratio = 5) 
  (h2 : initial_people = 45) 
  (h3 : lollipops = 12) : 
  (lollipops * lollipop_ratio - initial_people) = 15 := by 
  sorry

end NUMINAMATH_GPT_people_came_later_l303_30356


namespace NUMINAMATH_GPT_total_weight_l303_30352

-- Define the weights of almonds and pecans.
def weight_almonds : ℝ := 0.14
def weight_pecans : ℝ := 0.38

-- Prove that the total weight of nuts is 0.52 kilograms.
theorem total_weight (almonds pecans : ℝ) (h_almonds : almonds = 0.14) (h_pecans : pecans = 0.38) :
  almonds + pecans = 0.52 :=
by
  sorry

end NUMINAMATH_GPT_total_weight_l303_30352


namespace NUMINAMATH_GPT_tan_alpha_plus_pi_div_4_l303_30383

theorem tan_alpha_plus_pi_div_4 (α : ℝ) (hcos : Real.cos α = 3 / 5) (h0 : 0 < α) (hpi : α < Real.pi) :
  Real.tan (α + Real.pi / 4) = -7 :=
by
  sorry

end NUMINAMATH_GPT_tan_alpha_plus_pi_div_4_l303_30383


namespace NUMINAMATH_GPT_expansion_contains_no_x2_l303_30319

theorem expansion_contains_no_x2 (n : ℕ) (h1 : 5 ≤ n ∧ n ≤ 8) :
  ¬ (∃ k, (x + 1)^2 * (x + 1 / x^3)^n = k * x^2) → n = 7 :=
sorry

end NUMINAMATH_GPT_expansion_contains_no_x2_l303_30319


namespace NUMINAMATH_GPT_part_a_part_b_l303_30374

noncomputable def triangle_exists (h1 h2 h3 : ℕ) : Prop :=
  ∃ a b c, 2 * a = h1 * (b + c) ∧ 2 * b = h2 * (a + c) ∧ 2 * c = h3 * (a + b)

theorem part_a : ¬ triangle_exists 2 3 6 :=
sorry

theorem part_b : triangle_exists 2 3 5 :=
sorry

end NUMINAMATH_GPT_part_a_part_b_l303_30374


namespace NUMINAMATH_GPT_four_points_nonexistent_l303_30305

theorem four_points_nonexistent :
  ¬ (∃ (A B C D : ℝ × ℝ × ℝ), 
    dist A B = 8 ∧ 
    dist C D = 8 ∧ 
    dist A C = 10 ∧ 
    dist B D = 10 ∧ 
    dist A D = 13 ∧ 
    dist B C = 13) :=
by
  sorry

end NUMINAMATH_GPT_four_points_nonexistent_l303_30305


namespace NUMINAMATH_GPT_problem_statement_l303_30300

theorem problem_statement (x y : ℝ) (h1 : 1/x + 1/y = 5) (h2 : x * y + x + y = 7) : 
  x^2 * y + x * y^2 = 245 / 36 := 
by
  sorry

end NUMINAMATH_GPT_problem_statement_l303_30300


namespace NUMINAMATH_GPT_solve_system_l303_30371

theorem solve_system :
  ∃ (x y z : ℝ), x + y + z = 9 ∧ (1/x + 1/y + 1/z = 1) ∧ (x * y + x * z + y * z = 27) ∧ x = 3 ∧ y = 3 ∧ z = 3 := by
sorry

end NUMINAMATH_GPT_solve_system_l303_30371


namespace NUMINAMATH_GPT_straws_to_adult_pigs_l303_30313

theorem straws_to_adult_pigs (total_straws : ℕ) (num_piglets : ℕ) (straws_per_piglet : ℕ)
  (straws_adult_pigs : ℕ) (straws_piglets : ℕ) :
  total_straws = 300 →
  num_piglets = 20 →
  straws_per_piglet = 6 →
  (straws_piglets = num_piglets * straws_per_piglet) →
  (straws_adult_pigs = straws_piglets) →
  straws_adult_pigs = 120 :=
by
  intros h_total h_piglets h_straws_per_piglet h_straws_piglets h_equal
  subst h_total
  subst h_piglets
  subst h_straws_per_piglet
  subst h_straws_piglets
  subst h_equal
  sorry

end NUMINAMATH_GPT_straws_to_adult_pigs_l303_30313


namespace NUMINAMATH_GPT_value_of_f_at_2_and_neg_log2_3_l303_30351

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then Real.log x / Real.log 2 else 2^(-x)

theorem value_of_f_at_2_and_neg_log2_3 :
  f 2 * f (-Real.log 3 / Real.log 2) = 3 := by
  sorry

end NUMINAMATH_GPT_value_of_f_at_2_and_neg_log2_3_l303_30351


namespace NUMINAMATH_GPT_four_m0_as_sum_of_primes_l303_30373

theorem four_m0_as_sum_of_primes (m0 : ℕ) (h1 : m0 > 1) 
  (h2 : ∀ n : ℕ, ∃ p : ℕ, Prime p ∧ n ≤ p ∧ p ≤ 2 * n) 
  (h3 : ∀ p1 p2 : ℕ, Prime p1 → Prime p2 → (2 * m0 ≠ p1 + p2)) : 
  ∃ p1 p2 p3 p4 : ℕ, Prime p1 ∧ Prime p2 ∧ Prime p3 ∧ Prime p4 ∧ (4 * m0 = p1 + p2 + p3 + p4) ∨ (∃ p1 p2 p3 : ℕ, Prime p1 ∧ Prime p2 ∧ Prime p3 ∧ 4 * m0 = p1 + p2 + p3) :=
by sorry

end NUMINAMATH_GPT_four_m0_as_sum_of_primes_l303_30373


namespace NUMINAMATH_GPT_function_satisfies_condition_l303_30339

noncomputable def f (x : ℝ) : ℝ := (x - 3) / (x^2 - x + 4)

theorem function_satisfies_condition :
  ∀ (x : ℝ), 2 * f (1 - x) + 1 = x * f x :=
by
  intro x
  unfold f
  sorry

end NUMINAMATH_GPT_function_satisfies_condition_l303_30339


namespace NUMINAMATH_GPT_set_representation_l303_30377

open Nat

def isInPositiveNaturals (x : ℕ) : Prop :=
  x ≠ 0

def isPositiveDivisor (a b : ℕ) : Prop :=
  b ≠ 0 ∧ a % b = 0

theorem set_representation :
  {x | isInPositiveNaturals x ∧ isPositiveDivisor 6 (6 - x)} = {3, 4, 5} :=
by
  sorry

end NUMINAMATH_GPT_set_representation_l303_30377


namespace NUMINAMATH_GPT_push_mower_cuts_one_acre_per_hour_l303_30314

noncomputable def acres_per_hour_push_mower : ℕ :=
  let total_acres := 8
  let fraction_riding := 3 / 4
  let riding_mower_rate := 2
  let mowing_hours := 5
  let acres_riding := fraction_riding * total_acres
  let time_riding_mower := acres_riding / riding_mower_rate
  let remaining_hours := mowing_hours - time_riding_mower
  let remaining_acres := total_acres - acres_riding
  remaining_acres / remaining_hours

theorem push_mower_cuts_one_acre_per_hour :
  acres_per_hour_push_mower = 1 := 
by 
  -- Detailed proof steps would go here.
  sorry

end NUMINAMATH_GPT_push_mower_cuts_one_acre_per_hour_l303_30314


namespace NUMINAMATH_GPT_no_n_in_range_l303_30381

theorem no_n_in_range :
  ¬ ∃ n : ℤ, 10 ≤ n ∧ n ≤ 15 ∧ n % 7 = 10467 % 7 := by
  sorry

end NUMINAMATH_GPT_no_n_in_range_l303_30381


namespace NUMINAMATH_GPT_evaluation_of_expression_l303_30398

theorem evaluation_of_expression: 
  (3^10 + 3^7) / (3^10 - 3^7) = 14 / 13 := 
  sorry

end NUMINAMATH_GPT_evaluation_of_expression_l303_30398


namespace NUMINAMATH_GPT_inequality_always_true_l303_30364

theorem inequality_always_true 
  (a b : ℝ) 
  (h1 : ab > 0) : 
  (b / a) + (a / b) ≥ 2 := 
by sorry

end NUMINAMATH_GPT_inequality_always_true_l303_30364


namespace NUMINAMATH_GPT_total_distance_hiked_l303_30385

def distance_car_to_stream : ℝ := 0.2
def distance_stream_to_meadow : ℝ := 0.4
def distance_meadow_to_campsite : ℝ := 0.1

theorem total_distance_hiked : 
  distance_car_to_stream + distance_stream_to_meadow + distance_meadow_to_campsite = 0.7 := by
  sorry

end NUMINAMATH_GPT_total_distance_hiked_l303_30385


namespace NUMINAMATH_GPT_maximum_rubles_received_l303_30321

def four_digit_number_of_form_20xx (n : ℕ) : Prop :=
  2000 ≤ n ∧ n < 2100

def divisible_by (d : ℕ) (n : ℕ) : Prop :=
  n % d = 0

theorem maximum_rubles_received :
  ∃ (n : ℕ), four_digit_number_of_form_20xx n ∧
             divisible_by 1 n ∧
             divisible_by 3 n ∧
             divisible_by 7 n ∧
             divisible_by 9 n ∧
             divisible_by 11 n ∧
             ¬ divisible_by 5 n ∧
             1 + 3 + 7 + 9 + 11 = 31 :=
sorry

end NUMINAMATH_GPT_maximum_rubles_received_l303_30321


namespace NUMINAMATH_GPT_roses_given_to_mother_is_6_l303_30342

-- Define the initial conditions
def initial_roses : ℕ := 20
def roses_to_grandmother : ℕ := 9
def roses_to_sister : ℕ := 4
def roses_kept : ℕ := 1

-- Define the expected number of roses given to mother
def roses_given_to_mother : ℕ := initial_roses - (roses_to_grandmother + roses_to_sister + roses_kept)

-- The theorem stating the number of roses given to the mother
theorem roses_given_to_mother_is_6 : roses_given_to_mother = 6 := by
  sorry

end NUMINAMATH_GPT_roses_given_to_mother_is_6_l303_30342


namespace NUMINAMATH_GPT_mark_theater_expense_l303_30347

noncomputable def price_per_performance (hours_per_performance : ℕ) (price_per_hour : ℕ) : ℕ :=
  hours_per_performance * price_per_hour

noncomputable def total_cost (num_weeks : ℕ) (num_visits_per_week : ℕ) (price_per_performance : ℕ) : ℕ :=
  num_weeks * num_visits_per_week * price_per_performance

theorem mark_theater_expense :
  ∀(num_weeks num_visits_per_week hours_per_performance price_per_hour : ℕ),
  num_weeks = 6 →
  num_visits_per_week = 1 →
  hours_per_performance = 3 →
  price_per_hour = 5 →
  total_cost num_weeks num_visits_per_week (price_per_performance hours_per_performance price_per_hour) = 90 :=
by
  intros num_weeks num_visits_per_week hours_per_performance price_per_hour
  intro h_num_weeks h_num_visits_per_week h_hours_per_performance h_price_per_hour
  rw [h_num_weeks, h_num_visits_per_week, h_hours_per_performance, h_price_per_hour]
  sorry

end NUMINAMATH_GPT_mark_theater_expense_l303_30347


namespace NUMINAMATH_GPT_monkey_height_37_minutes_l303_30309

noncomputable def monkey_climb (minutes : ℕ) : ℕ :=
if minutes = 37 then 60 else 0

theorem monkey_height_37_minutes : (monkey_climb 37) = 60 := 
by
  sorry

end NUMINAMATH_GPT_monkey_height_37_minutes_l303_30309


namespace NUMINAMATH_GPT_largest_term_l303_30315

-- Given conditions
def U : ℕ := 2 * (2010 ^ 2011)
def V : ℕ := 2010 ^ 2011
def W : ℕ := 2009 * (2010 ^ 2010)
def X : ℕ := 2 * (2010 ^ 2010)
def Y : ℕ := 2010 ^ 2010
def Z : ℕ := 2010 ^ 2009

-- Proposition to prove
theorem largest_term : 
  (U - V) > (V - W) ∧ 
  (U - V) > (W - X + 100) ∧ 
  (U - V) > (X - Y) ∧ 
  (U - V) > (Y - Z) := 
by 
  sorry

end NUMINAMATH_GPT_largest_term_l303_30315


namespace NUMINAMATH_GPT_restore_triangle_Nagel_point_l303_30361

-- Define the variables and types involved
variables {Point : Type}

-- Assume a structure to capture the properties of a triangle
structure Triangle (Point : Type) :=
(A B C : Point)

-- Define the given conditions
variables (N B E : Point)

-- Statement of the main Lean theorem to reconstruct the triangle ABC
theorem restore_triangle_Nagel_point 
    (N B E : Point) :
    ∃ (ABC : Triangle Point), 
      (ABC).B = B ∧
      -- Additional properties of the triangle to be stated here
      sorry
    :=
sorry

end NUMINAMATH_GPT_restore_triangle_Nagel_point_l303_30361


namespace NUMINAMATH_GPT_partition_exists_l303_30382
open Set Real

theorem partition_exists (r : ℚ) (hr : r > 1) :
  ∃ (A B : ℕ → Prop), (∀ n, A n ∨ B n) ∧ (∀ n, ¬(A n ∧ B n)) ∧ 
  (∀ k l, A k → A l → (k : ℚ) / (l : ℚ) ≠ r) ∧ 
  (∀ k l, B k → B l → (k : ℚ) / (l : ℚ) ≠ r) :=
sorry

end NUMINAMATH_GPT_partition_exists_l303_30382


namespace NUMINAMATH_GPT_outfits_count_l303_30399

-- Definitions of the counts of each type of clothing item
def num_blue_shirts : Nat := 6
def num_green_shirts : Nat := 4
def num_pants : Nat := 7
def num_blue_hats : Nat := 9
def num_green_hats : Nat := 7

-- Statement of the problem to prove
theorem outfits_count :
  (num_blue_shirts * num_pants * num_green_hats) + (num_green_shirts * num_pants * num_blue_hats) = 546 :=
by
  sorry

end NUMINAMATH_GPT_outfits_count_l303_30399


namespace NUMINAMATH_GPT_online_sale_discount_l303_30362

theorem online_sale_discount (purchase_amount : ℕ) (discount_per_100 : ℕ) (total_paid : ℕ) : 
  purchase_amount = 250 → 
  discount_per_100 = 10 → 
  total_paid = purchase_amount - (purchase_amount / 100) * discount_per_100 → 
  total_paid = 230 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_online_sale_discount_l303_30362


namespace NUMINAMATH_GPT_transform_identity_l303_30397

theorem transform_identity (a b c d : ℝ) : 
  (a^2 + b^2) * (c^2 + d^2) = (a * c + b * d)^2 + (a * d - b * c)^2 := 
sorry

end NUMINAMATH_GPT_transform_identity_l303_30397


namespace NUMINAMATH_GPT_equal_functions_A_l303_30329

-- Define the functions
def f₁ (x : ℝ) : ℝ := x^2 - 2*x - 1
def f₂ (t : ℝ) : ℝ := t^2 - 2*t - 1

-- Theorem stating that f₁ is equal to f₂
theorem equal_functions_A : ∀ x : ℝ, f₁ x = f₂ x :=
by
  intros x
  sorry

end NUMINAMATH_GPT_equal_functions_A_l303_30329


namespace NUMINAMATH_GPT_vector_equality_l303_30336

variable (V : Type*) [AddCommGroup V] [Module ℝ V]

theorem vector_equality {a x : V} (h : 2 • x - 3 • (x - 2 • a) = 0) : x = 6 • a :=
by
  sorry

end NUMINAMATH_GPT_vector_equality_l303_30336


namespace NUMINAMATH_GPT_bags_of_sugar_bought_l303_30306

-- Define the conditions as constants
def cups_at_home : ℕ := 3
def cups_per_bag : ℕ := 6
def cups_per_batter_dozen : ℕ := 1
def cups_per_frosting_dozen : ℕ := 2
def dozens_of_cupcakes : ℕ := 5

-- Prove that the number of bags of sugar Lillian bought is 2
theorem bags_of_sugar_bought : ∃ bags : ℕ, bags = 2 :=
by
  let total_cups_batter := dozens_of_cupcakes * cups_per_batter_dozen
  let total_cups_frosting := dozens_of_cupcakes * cups_per_frosting_dozen
  let total_cups_needed := total_cups_batter + total_cups_frosting
  let cups_to_buy := total_cups_needed - cups_at_home
  let bags := cups_to_buy / cups_per_bag
  have h : bags = 2 := sorry
  exact ⟨bags, h⟩

end NUMINAMATH_GPT_bags_of_sugar_bought_l303_30306


namespace NUMINAMATH_GPT_jack_kids_solution_l303_30325

def jack_kids (k : ℕ) : Prop :=
  7 * 3 * k = 63

theorem jack_kids_solution : jack_kids 3 :=
by
  sorry

end NUMINAMATH_GPT_jack_kids_solution_l303_30325


namespace NUMINAMATH_GPT_edward_money_proof_l303_30334

def edward_total_money (earned_per_lawn : ℕ) (number_of_lawns : ℕ) (saved_up : ℕ) : ℕ :=
  earned_per_lawn * number_of_lawns + saved_up

theorem edward_money_proof :
  edward_total_money 8 5 7 = 47 :=
by
  sorry

end NUMINAMATH_GPT_edward_money_proof_l303_30334


namespace NUMINAMATH_GPT_sofiya_wins_l303_30324

/-- Define the initial configuration and game rules -/
def initial_configuration : Type := { n : Nat // n = 2025 }

/--
  Define the game such that Sofiya starts and follows the strategy of always
  removing a neighbor from the arc with an even number of people.
-/
def winning_strategy (n : initial_configuration) : Prop :=
  n.1 % 2 = 1 ∧ 
  (∀ turn : Nat, turn % 2 = 0 → 
    (∃ arc : initial_configuration, arc.1 % 2 = 0 ∧ arc.1 < n.1) ∧
    (∀ marquis_turn : Nat, marquis_turn % 2 = 1 → 
      (∃ arc : initial_configuration, arc.1 % 2 = 1)))

/-- Sofiya has the winning strategy given the conditions of the game -/
theorem sofiya_wins : winning_strategy ⟨2025, rfl⟩ :=
sorry

end NUMINAMATH_GPT_sofiya_wins_l303_30324


namespace NUMINAMATH_GPT_walnut_price_l303_30303

theorem walnut_price {total_weight total_value walnut_price hazelnut_price : ℕ} 
  (h1 : total_weight = 55)
  (h2 : total_value = 1978)
  (h3 : walnut_price > hazelnut_price)
  (h4 : ∃ a b : ℕ, walnut_price = 10 * a + b ∧ hazelnut_price = 10 * b + a ∧ 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9)
  (h5 : ∃ a b : ℕ, walnut_price = 10 * a + b ∧ b = a - 1) : 
  walnut_price = 43 := 
sorry

end NUMINAMATH_GPT_walnut_price_l303_30303


namespace NUMINAMATH_GPT_percentage_of_alcohol_in_new_mixture_l303_30363

def original_solution_volume : ℕ := 11
def added_water_volume : ℕ := 3
def alcohol_percentage_original : ℝ := 0.42

def total_volume : ℕ := original_solution_volume + added_water_volume
def amount_of_alcohol : ℝ := alcohol_percentage_original * original_solution_volume

theorem percentage_of_alcohol_in_new_mixture :
  (amount_of_alcohol / total_volume) * 100 = 33 := by
  sorry

end NUMINAMATH_GPT_percentage_of_alcohol_in_new_mixture_l303_30363


namespace NUMINAMATH_GPT_prob_all_meet_standard_prob_at_least_one_meets_standard_l303_30396

def P_meeting_standard_A := 0.8
def P_meeting_standard_B := 0.6
def P_meeting_standard_C := 0.5

theorem prob_all_meet_standard :
  (P_meeting_standard_A * P_meeting_standard_B * P_meeting_standard_C) = 0.24 :=
by
  sorry

theorem prob_at_least_one_meets_standard :
  (1 - ((1 - P_meeting_standard_A) * (1 - P_meeting_standard_B) * (1 - P_meeting_standard_C))) = 0.96 :=
by
  sorry

end NUMINAMATH_GPT_prob_all_meet_standard_prob_at_least_one_meets_standard_l303_30396


namespace NUMINAMATH_GPT_mr_a_loss_l303_30350

noncomputable def house_initial_value := 12000
noncomputable def first_transaction_loss := 15 / 100
noncomputable def second_transaction_gain := 20 / 100

def house_value_after_first_transaction (initial_value loss : ℝ) : ℝ :=
  initial_value * (1 - loss)

def house_value_after_second_transaction (value_after_first gain : ℝ) : ℝ :=
  value_after_first * (1 + gain)

theorem mr_a_loss :
  let initial_value := house_initial_value
  let loss := first_transaction_loss
  let gain := second_transaction_gain
  let value_after_first := house_value_after_first_transaction initial_value loss
  let value_after_second := house_value_after_second_transaction value_after_first gain
  value_after_second - initial_value = 240 :=
by
  sorry

end NUMINAMATH_GPT_mr_a_loss_l303_30350


namespace NUMINAMATH_GPT_intersection_A_B_l303_30337

def A : Set (ℝ × ℝ) := { p | ∃ x y, p = (x, y) ∧ (x^2 / 4 + 3 * y^2 / 4 = 1) }
def B : Set (ℝ × ℝ) := { p | ∃ x y, p = (x, y) ∧ (y = x^2) }

theorem intersection_A_B :
  {x : ℝ | 0 ≤ x ∧ x ≤ 2} = 
  {x : ℝ | ∃ y : ℝ, ((x, y) ∈ A ∧ (x, y) ∈ B)} :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l303_30337


namespace NUMINAMATH_GPT_cost_of_cheese_without_coupon_l303_30341

theorem cost_of_cheese_without_coupon
    (cost_bread : ℝ := 4.00)
    (cost_meat : ℝ := 5.00)
    (coupon_cheese : ℝ := 1.00)
    (coupon_meat : ℝ := 1.00)
    (cost_sandwich : ℝ := 2.00)
    (num_sandwiches : ℝ := 10)
    (C : ℝ) : 
    (num_sandwiches * cost_sandwich = (cost_bread + (cost_meat - coupon_meat) + cost_meat + (C - coupon_cheese) + C)) → (C = 4.50) :=
by {
    sorry
}

end NUMINAMATH_GPT_cost_of_cheese_without_coupon_l303_30341


namespace NUMINAMATH_GPT_quadrilateral_EFGH_area_l303_30376

-- Definitions based on conditions
def quadrilateral_EFGH_right_angles (F H : ℝ) : Prop :=
  ∃ E G, E - F = 0 ∧ H - G = 0

def quadrilateral_length_hypotenuse (E G : ℝ) : Prop :=
  E - G = 5

def distinct_integer_lengths (EF FG EH HG : ℝ) : Prop :=
  EF ≠ FG ∧ EH ≠ HG ∧ ∃ a b : ℕ, EF = a ∧ FG = b ∧ EH = b ∧ HG = a ∧ a * a + b * b = 25

-- Proof statement
theorem quadrilateral_EFGH_area (F H : ℝ) 
  (EF FG EH HG E G : ℝ) 
  (h1 : quadrilateral_EFGH_right_angles F H) 
  (h2 : quadrilateral_length_hypotenuse E G)
  (h3 : distinct_integer_lengths EF FG EH HG) 
: 
  EF * FG / 2 + EH * HG / 2 = 12 := 
sorry

end NUMINAMATH_GPT_quadrilateral_EFGH_area_l303_30376


namespace NUMINAMATH_GPT_math_problem_l303_30379

theorem math_problem (L S : ℕ) (h1 : L - S = 1365) (h2 : L = 6 * S + 35) : L = 1631 := 
by
  sorry

end NUMINAMATH_GPT_math_problem_l303_30379


namespace NUMINAMATH_GPT_difference_of_fractions_l303_30387

theorem difference_of_fractions (h₁ : 1/10 * 8000 = 800) (h₂ : (1/20) / 100 * 8000 = 4) : 800 - 4 = 796 :=
by
  sorry

end NUMINAMATH_GPT_difference_of_fractions_l303_30387


namespace NUMINAMATH_GPT_range_of_b_div_a_l303_30392

theorem range_of_b_div_a (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
(h1 : a ≤ b + c) (h2 : b + c ≤ 2 * a) (h3 : b ≤ a + c) (h4 : a + c ≤ 2 * b) :
  (2 / 3 : ℝ) ≤ b / a ∧ b / a ≤ (3 / 2 : ℝ) :=
sorry

end NUMINAMATH_GPT_range_of_b_div_a_l303_30392


namespace NUMINAMATH_GPT_dan_bought_one_candy_bar_l303_30360

-- Define the conditions
def initial_money : ℕ := 4
def cost_per_candy_bar : ℕ := 3
def money_left : ℕ := 1

-- Define the number of candy bars Dan bought
def number_of_candy_bars_bought : ℕ := (initial_money - money_left) / cost_per_candy_bar

-- Prove the number of candy bars bought is equal to 1
theorem dan_bought_one_candy_bar : number_of_candy_bars_bought = 1 := by
  sorry

end NUMINAMATH_GPT_dan_bought_one_candy_bar_l303_30360


namespace NUMINAMATH_GPT_total_games_correct_l303_30368

noncomputable def number_of_games_per_month : ℕ := 13
noncomputable def number_of_months_in_season : ℕ := 14
noncomputable def total_games_in_season : ℕ := number_of_games_per_month * number_of_months_in_season

theorem total_games_correct : total_games_in_season = 182 := by
  sorry

end NUMINAMATH_GPT_total_games_correct_l303_30368


namespace NUMINAMATH_GPT_shaded_area_correct_l303_30357

def unit_triangle_area : ℕ := 10

def small_shaded_area : ℕ := unit_triangle_area

def medium_shaded_area : ℕ := 6 * unit_triangle_area

def large_shaded_area : ℕ := 7 * unit_triangle_area

def total_shaded_area : ℕ :=
  small_shaded_area + medium_shaded_area + large_shaded_area

theorem shaded_area_correct : total_shaded_area = 110 := 
  by
    sorry

end NUMINAMATH_GPT_shaded_area_correct_l303_30357


namespace NUMINAMATH_GPT_neg_p_implies_neg_q_sufficient_but_not_necessary_l303_30358

variables (x : ℝ) (p : Prop) (q : Prop)

def p_condition := (1 < x ∨ x < -3)
def q_condition := (5 * x - 6 > x ^ 2)

theorem neg_p_implies_neg_q_sufficient_but_not_necessary :
  p_condition x → q_condition x → ((¬ p_condition x) → (¬ q_condition x)) :=
by 
  intro h1 h2
  sorry

end NUMINAMATH_GPT_neg_p_implies_neg_q_sufficient_but_not_necessary_l303_30358


namespace NUMINAMATH_GPT_jordan_rectangle_width_l303_30327

theorem jordan_rectangle_width
  (length_carol : ℕ) (width_carol : ℕ) (length_jordan : ℕ) (width_jordan : ℕ)
  (h1 : length_carol = 5) (h2 : width_carol = 24) (h3 : length_jordan = 2)
  (h4 : length_carol * width_carol = length_jordan * width_jordan) :
  width_jordan = 60 := by
  sorry

end NUMINAMATH_GPT_jordan_rectangle_width_l303_30327


namespace NUMINAMATH_GPT_largest_value_of_a_l303_30386

noncomputable def largest_possible_value_of_a (a b c d : ℕ) 
  (h1 : a < 3 * b) (h2 : b < 4 * c) (h3 : c < 5 * d) (h4 : c % 2 = 0) (h5 : d < 150) : Prop :=
  a = 8924

theorem largest_value_of_a (a b c d : ℕ)
  (h1 : a < 3 * b) (h2 : b < 4 * c) (h3 : c < 5 * d) (h4 : c % 2 = 0) (h5 : d < 150)
  (h6 : largest_possible_value_of_a a b c d h1 h2 h3 h4 h5) : a = 8924 := h6

end NUMINAMATH_GPT_largest_value_of_a_l303_30386


namespace NUMINAMATH_GPT_BethsHighSchoolStudents_l303_30369

-- Define the variables
variables (B P : ℕ)

-- Define the conditions given in the problem
def condition1 : Prop := B = 4 * P
def condition2 : Prop := B + P = 5000

-- The theorem to be proved
theorem BethsHighSchoolStudents (h1 : condition1 B P) (h2 : condition2 B P) : B = 4000 :=
by
  -- Proof will be here
  sorry

end NUMINAMATH_GPT_BethsHighSchoolStudents_l303_30369


namespace NUMINAMATH_GPT_prove_x_plus_y_leq_zero_l303_30354

-- Definitions of the conditions
def valid_powers (a b : ℝ) (x y : ℝ) : Prop :=
  1 < a ∧ a < b ∧ a^x + b^y ≤ a^(-x) + b^(-y)

-- The theorem statement
theorem prove_x_plus_y_leq_zero (a b x y : ℝ) (h : valid_powers a b x y) : 
  x + y ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_prove_x_plus_y_leq_zero_l303_30354


namespace NUMINAMATH_GPT_range_of_x_l303_30310

variable (a b x : ℝ)

def conditions : Prop := (a > 0) ∧ (b > 0)

theorem range_of_x (h : conditions a b) : (x^2 + 2*x < 8) -> (-4 < x) ∧ (x < 2) := 
by
  sorry

end NUMINAMATH_GPT_range_of_x_l303_30310


namespace NUMINAMATH_GPT_tan_family_total_cost_l303_30307

-- Define the number of people in each age group and respective discounts
def num_children : ℕ := 2
def num_adults : ℕ := 2
def num_seniors : ℕ := 2

def price_adult_ticket : ℝ := 10
def discount_senior : ℝ := 0.30
def discount_child : ℝ := 0.20
def group_discount : ℝ := 0.10

-- Calculate the cost for each group with discounts applied
def price_senior_ticket := price_adult_ticket * (1 - discount_senior)
def price_child_ticket := price_adult_ticket * (1 - discount_child)

-- Calculate the total cost of tickets before group discount
def total_cost_before_group_discount :=
  (price_senior_ticket * num_seniors) +
  (price_child_ticket * num_children) +
  (price_adult_ticket * num_adults)

-- Check if the family qualifies for group discount and apply if necessary
def total_cost_after_group_discount :=
  if (num_children + num_adults + num_seniors > 5)
  then total_cost_before_group_discount * (1 - group_discount)
  else total_cost_before_group_discount

-- Main theorem statement
theorem tan_family_total_cost : total_cost_after_group_discount = 45 := by
  sorry

end NUMINAMATH_GPT_tan_family_total_cost_l303_30307


namespace NUMINAMATH_GPT_number_of_insects_l303_30346

theorem number_of_insects (total_legs : ℕ) (legs_per_insect : ℕ) (h : total_legs = 54) (k : legs_per_insect = 6) :
  total_legs / legs_per_insect = 9 := by
  sorry

end NUMINAMATH_GPT_number_of_insects_l303_30346


namespace NUMINAMATH_GPT_robie_initial_cards_l303_30330

-- Definitions of the problem conditions
def each_box_cards : ℕ := 25
def extra_cards : ℕ := 11
def given_away_boxes : ℕ := 6
def remaining_boxes : ℕ := 12

-- The final theorem we need to prove
theorem robie_initial_cards : 
  (given_away_boxes + remaining_boxes) * each_box_cards + extra_cards = 461 :=
by
  sorry

end NUMINAMATH_GPT_robie_initial_cards_l303_30330


namespace NUMINAMATH_GPT_perpendicular_line_through_intersection_l303_30372

theorem perpendicular_line_through_intersection :
  ∃ (x y : ℝ), (x + y - 2 = 0) ∧ (3 * x + 2 * y - 5 = 0) ∧ (4 * x - 3 * y - 1 = 0) :=
sorry

end NUMINAMATH_GPT_perpendicular_line_through_intersection_l303_30372


namespace NUMINAMATH_GPT_time_difference_between_shoes_l303_30317

-- Define the conditions
def time_per_mile_regular := 10
def time_per_mile_new := 13
def distance_miles := 5

-- Define the theorem to be proven
theorem time_difference_between_shoes :
  (distance_miles * time_per_mile_new) - (distance_miles * time_per_mile_regular) = 15 :=
by
  sorry

end NUMINAMATH_GPT_time_difference_between_shoes_l303_30317


namespace NUMINAMATH_GPT_car_distance_calculation_l303_30318

noncomputable def total_distance (u a v t1 t2: ℝ) : ℝ :=
  let d1 := (u * t1) + (1 / 2) * a * t1^2
  let d2 := v * t2
  d1 + d2

theorem car_distance_calculation :
  total_distance 30 5 60 2 3 = 250 :=
by
  unfold total_distance
  -- next steps include simplifying the math, but we'll defer details to proof
  sorry

end NUMINAMATH_GPT_car_distance_calculation_l303_30318


namespace NUMINAMATH_GPT_no_such_function_exists_l303_30301

theorem no_such_function_exists :
  ¬ ∃ (f : ℕ → ℕ), ∀ (n : ℕ), f (f n) = n + 1 :=
by
  sorry

end NUMINAMATH_GPT_no_such_function_exists_l303_30301


namespace NUMINAMATH_GPT_classical_prob_exp_is_exp1_l303_30322

-- Define the conditions under which an experiment is a classical probability model
def classical_probability_model (experiment : String) : Prop :=
  match experiment with
  | "exp1" => true  -- experiment ①: finite outcomes and equal likelihood
  | "exp2" => false -- experiment ②: infinite outcomes
  | "exp3" => false -- experiment ③: unequal likelihood
  | "exp4" => false -- experiment ④: infinite outcomes
  | _ => false

theorem classical_prob_exp_is_exp1 : classical_probability_model "exp1" = true ∧
                                      classical_probability_model "exp2" = false ∧
                                      classical_probability_model "exp3" = false ∧
                                      classical_probability_model "exp4" = false :=
by
  sorry

end NUMINAMATH_GPT_classical_prob_exp_is_exp1_l303_30322


namespace NUMINAMATH_GPT_average_price_l303_30384

theorem average_price (books1 books2 : ℕ) (price1 price2 : ℝ)
  (h1 : books1 = 65) (h2 : price1 = 1380)
  (h3 : books2 = 55) (h4 : price2 = 900) :
  (price1 + price2) / (books1 + books2) = 19 :=
by
  sorry

end NUMINAMATH_GPT_average_price_l303_30384


namespace NUMINAMATH_GPT_contrapositive_proposition_l303_30326

-- Define the necessary elements in the context of real numbers
variables {a b c d : ℝ}

-- The statement of the contrapositive
theorem contrapositive_proposition : (a + c ≠ b + d) → (a ≠ b ∨ c ≠ d) :=
sorry

end NUMINAMATH_GPT_contrapositive_proposition_l303_30326


namespace NUMINAMATH_GPT_avg_one_fourth_class_l303_30353

variable (N : ℕ) (A : ℕ)
variable (h1 : ((N : ℝ) * 80) = (N / 4) * A + (3 * N / 4) * 76)

theorem avg_one_fourth_class : A = 92 :=
by
  sorry

end NUMINAMATH_GPT_avg_one_fourth_class_l303_30353


namespace NUMINAMATH_GPT_man_older_than_son_l303_30302

variables (M S : ℕ)

theorem man_older_than_son
  (h_son_age : S = 26)
  (h_future_age : M + 2 = 2 * (S + 2)) :
  M - S = 28 :=
by sorry

end NUMINAMATH_GPT_man_older_than_son_l303_30302


namespace NUMINAMATH_GPT_calc_x_squared_y_squared_l303_30367

theorem calc_x_squared_y_squared (x y : ℝ) (h1 : (x + y)^2 = 4) (h2 : x * y = -9) : x^2 + y^2 = 22 := by
  sorry

end NUMINAMATH_GPT_calc_x_squared_y_squared_l303_30367


namespace NUMINAMATH_GPT_probability_odd_product_sum_divisible_by_5_l303_30344

theorem probability_odd_product_sum_divisible_by_5 :
  (∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 20 ∧ 1 ≤ b ∧ b ≤ 20 ∧ a ≠ b ∧ (a * b % 2 = 1 ∧ (a + b) % 5 = 0)) →
  ∃ (p : ℚ), p = 3 / 95 :=
by
  sorry

end NUMINAMATH_GPT_probability_odd_product_sum_divisible_by_5_l303_30344


namespace NUMINAMATH_GPT_gcd_90_405_l303_30380

theorem gcd_90_405 : Nat.gcd 90 405 = 45 := by
  sorry

end NUMINAMATH_GPT_gcd_90_405_l303_30380


namespace NUMINAMATH_GPT_ordered_triples_count_l303_30345

theorem ordered_triples_count :
  ∃ (count : ℕ), count = 4 ∧
  (∃ a b c : ℕ,
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    Nat.lcm a b = 90 ∧
    Nat.lcm a c = 980 ∧
    Nat.lcm b c = 630) :=
by
  sorry

end NUMINAMATH_GPT_ordered_triples_count_l303_30345


namespace NUMINAMATH_GPT_trigonometric_problem_l303_30338

open Real

noncomputable def problem1 (α : ℝ) : Prop :=
  2 * sin α = 2 * (sin (α / 2))^2 - 1

noncomputable def problem2 (β : ℝ) : Prop :=
  3 * (tan β)^2 - 2 * tan β = 1

theorem trigonometric_problem (α β : ℝ) (hα : 0 < α ∧ α < π) (hβ : π / 2 < β ∧ β < π)
  (h1 : problem1 α) (h2 : problem2 β) :
  sin (2 * α) + cos (2 * α) = -1 / 5 ∧ α + β = 7 * π / 4 :=
  sorry

end NUMINAMATH_GPT_trigonometric_problem_l303_30338


namespace NUMINAMATH_GPT_ratio_bones_child_to_adult_woman_l303_30391

noncomputable def num_skeletons : ℕ := 20
noncomputable def num_adult_women : ℕ := num_skeletons / 2
noncomputable def num_adult_men_and_children : ℕ := num_skeletons - num_adult_women
noncomputable def num_adult_men : ℕ := num_adult_men_and_children / 2
noncomputable def num_children : ℕ := num_adult_men_and_children / 2
noncomputable def bones_per_adult_woman : ℕ := 20
noncomputable def bones_per_adult_man : ℕ := bones_per_adult_woman + 5
noncomputable def total_bones : ℕ := 375
noncomputable def bones_per_child : ℕ := (total_bones - (num_adult_women * bones_per_adult_woman + num_adult_men * bones_per_adult_man)) / num_children

theorem ratio_bones_child_to_adult_woman : 
  (bones_per_child : ℚ) / (bones_per_adult_woman : ℚ) = 1 / 2 := by
sorry

end NUMINAMATH_GPT_ratio_bones_child_to_adult_woman_l303_30391


namespace NUMINAMATH_GPT_ratio_of_enclosed_area_l303_30328

theorem ratio_of_enclosed_area
  (R : ℝ)
  (h_chords_eq : ∀ (A B C : ℝ), A = B → A = C)
  (h_inscribed_angle : ∀ (A B C O : ℝ), AOC = 30 * π / 180)
  : ((π * R^2 / 6) + (R^2 / 2)) / (π * R^2) = (π + 3) / (6 * π) :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_enclosed_area_l303_30328


namespace NUMINAMATH_GPT_find_m_l303_30323

noncomputable def f (m : ℝ) (x : ℝ) := (x^2 + m * x) * Real.exp x

def is_monotonically_decreasing_on_interval (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f y ≤ f x

theorem find_m 
  (a b : ℝ) 
  (h_interval : a = -3/2 ∧ b = 1)
  (h_decreasing : is_monotonically_decreasing_on_interval (f m) a b) :
  m = -3/2 := 
sorry

end NUMINAMATH_GPT_find_m_l303_30323


namespace NUMINAMATH_GPT_seven_power_product_prime_count_l303_30316

theorem seven_power_product_prime_count (n : ℕ) :
  ∃ primes: List ℕ, (∀ p ∈ primes, Prime p) ∧ primes.prod = 7^(7^n) + 1 ∧ primes.length ≥ 2*n + 3 :=
by
  sorry

end NUMINAMATH_GPT_seven_power_product_prime_count_l303_30316


namespace NUMINAMATH_GPT_harriet_speed_l303_30331

-- Define the conditions
def return_speed := 140 -- speed from B-town to A-ville in km/h
def total_trip_time := 5 -- total trip time in hours
def trip_time_to_B := 2.8 -- trip time from A-ville to B-town in hours

-- Define the theorem to prove
theorem harriet_speed {r_speed : ℝ} {t_time : ℝ} {t_time_B : ℝ} 
  (h1 : r_speed = 140) 
  (h2 : t_time = 5) 
  (h3 : t_time_B = 2.8) : 
  ((r_speed * (t_time - t_time_B)) / t_time_B) = 110 :=
by 
  -- Assume we have completed proof steps here.
  sorry

end NUMINAMATH_GPT_harriet_speed_l303_30331


namespace NUMINAMATH_GPT_calculate_a_mul_a_sub_3_l303_30389

variable (a : ℝ)

theorem calculate_a_mul_a_sub_3 : a * (a - 3) = a^2 - 3 * a := 
by
  sorry

end NUMINAMATH_GPT_calculate_a_mul_a_sub_3_l303_30389


namespace NUMINAMATH_GPT_smallest_scalene_triangle_perimeter_is_prime_l303_30390

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def consecutive_primes (p1 p2 p3 : ℕ) : Prop :=
  p1 < p2 ∧ p2 < p3 ∧ is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧
  (p2 = p1 + 2) ∧ (p3 = p1 + 6)

noncomputable def smallest_prime_perimeter : ℕ :=
  5 + 7 + 11

theorem smallest_scalene_triangle_perimeter_is_prime :
  ∃ (p1 p2 p3 : ℕ), p1 < p2 ∧ p2 < p3 ∧ consecutive_primes p1 p2 p3 ∧ is_prime (p1 + p2 + p3) ∧ (p1 + p2 + p3 = smallest_prime_perimeter) :=
by 
  sorry

end NUMINAMATH_GPT_smallest_scalene_triangle_perimeter_is_prime_l303_30390


namespace NUMINAMATH_GPT_total_cans_from_recycling_l303_30335

noncomputable def recycleCans (n : ℕ) : ℕ :=
  if n < 6 then 0 else n / 6 + recycleCans (n / 6 + n % 6)

theorem total_cans_from_recycling:
  recycleCans 486 = 96 :=
by
  sorry

end NUMINAMATH_GPT_total_cans_from_recycling_l303_30335


namespace NUMINAMATH_GPT_value_of_a_when_b_is_24_l303_30320

variable (a b k : ℝ)

theorem value_of_a_when_b_is_24 (h1 : a = k / b^2) (h2 : 40 = k / 12^2) (h3 : b = 24) : a = 10 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_when_b_is_24_l303_30320


namespace NUMINAMATH_GPT_find_m_for_local_minimum_l303_30366

noncomputable def f (x m : ℝ) := x * (x - m) ^ 2

theorem find_m_for_local_minimum :
  ∃ m : ℝ, (∀ x : ℝ, (x = 1 → deriv (λ x => f x m) x = 0) ∧ 
                  (x = 1 → deriv (deriv (λ x => f x m)) x > 0)) ∧ 
            m = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_m_for_local_minimum_l303_30366


namespace NUMINAMATH_GPT_smaller_root_of_quadratic_l303_30359

theorem smaller_root_of_quadratic :
  ∃ (x₁ x₂ : ℝ), (x₁ ≠ x₂) ∧ (x₁^2 - 14 * x₁ + 45 = 0) ∧ (x₂^2 - 14 * x₂ + 45 = 0) ∧ (min x₁ x₂ = 5) :=
sorry

end NUMINAMATH_GPT_smaller_root_of_quadratic_l303_30359


namespace NUMINAMATH_GPT_triangle_area_ratios_l303_30365

theorem triangle_area_ratios (K : ℝ) 
  (hCD : ∃ AC, ∃ CD, CD = AC / 4) 
  (hAE : ∃ AB, ∃ AE, AE = AB / 5) 
  (hBF : ∃ BC, ∃ BF, BF = BC / 3) :
  ∃ area_N1N2N3, area_N1N2N3 = (8 / 15) * K :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_ratios_l303_30365


namespace NUMINAMATH_GPT_rolling_dice_probability_l303_30388

-- Defining variables and conditions
def total_outcomes : Nat := 6^7

def favorable_outcomes : Nat :=
  Nat.choose 7 2 * 6 * (Nat.factorial 5) -- Calculation for exactly one pair of identical numbers

def probability : Rat :=
  favorable_outcomes / total_outcomes

-- The main theorem to prove the probability is 5/18
theorem rolling_dice_probability :
  probability = 5 / 18 := by
  sorry

end NUMINAMATH_GPT_rolling_dice_probability_l303_30388


namespace NUMINAMATH_GPT_smallest_number_is_28_l303_30393

theorem smallest_number_is_28 (a b c : ℕ) (h1 : (a + b + c) / 3 = 30) (h2 : b = 28) (h3 : b = c - 6) : a = 28 :=
by sorry

end NUMINAMATH_GPT_smallest_number_is_28_l303_30393


namespace NUMINAMATH_GPT_hindi_speaking_students_l303_30370

theorem hindi_speaking_students 
    (G M T A : ℕ)
    (Total : ℕ)
    (hG : G = 6)
    (hM : M = 6)
    (hT : T = 2)
    (hA : A = 1)
    (hTotal : Total = 22)
    : ∃ H, Total = G + H + M - (T - A) + A ∧ H = 10 := by
  sorry

end NUMINAMATH_GPT_hindi_speaking_students_l303_30370


namespace NUMINAMATH_GPT_arithmetic_sequence_common_difference_l303_30349

noncomputable def common_difference (a b : ℝ) : ℝ := a - 1

theorem arithmetic_sequence_common_difference :
  ∀ (a b : ℝ), 
    (a - 1 = b - a) → 
    ((a + 2) ^ 2 = 3 * (b + 5)) → 
    common_difference a b = 3 := by
  intros a b h1 h2
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_common_difference_l303_30349


namespace NUMINAMATH_GPT_students_play_neither_l303_30312

def total_students : ℕ := 35
def play_football : ℕ := 26
def play_tennis : ℕ := 20
def play_both : ℕ := 17

theorem students_play_neither : (total_students - (play_football + play_tennis - play_both)) = 6 := by
  sorry

end NUMINAMATH_GPT_students_play_neither_l303_30312


namespace NUMINAMATH_GPT_polynomial_real_roots_l303_30378

theorem polynomial_real_roots :
  (∃ x : ℝ, x^4 - 3*x^3 - 2*x^2 + 6*x + 9 = 0) ↔ (x = 1 ∨ x = 3) := 
by
  sorry

end NUMINAMATH_GPT_polynomial_real_roots_l303_30378
