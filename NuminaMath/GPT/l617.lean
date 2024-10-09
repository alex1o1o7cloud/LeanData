import Mathlib

namespace only_B_forms_triangle_l617_61700

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

end only_B_forms_triangle_l617_61700


namespace k_value_for_root_multiplicity_l617_61791

theorem k_value_for_root_multiplicity (k : ℝ) :
  (∃ x : ℝ, (x - 1) / (x - 3) = k / (x - 3) ∧ (x-3 = 0)) → k = 2 :=
by
  sorry

end k_value_for_root_multiplicity_l617_61791


namespace final_books_is_correct_l617_61742

def initial_books : ℝ := 35.5
def books_bought : ℝ := 12.3
def books_given_to_friends : ℝ := 7.2
def books_donated : ℝ := 20.8

theorem final_books_is_correct :
  (initial_books + books_bought - books_given_to_friends - books_donated) = 19.8 := by
  sorry

end final_books_is_correct_l617_61742


namespace arithmetic_geometric_sequence_l617_61799

theorem arithmetic_geometric_sequence
    (a : ℕ → ℕ)
    (b : ℕ → ℕ)
    (h_arith_seq : ∀ n, a (n + 1) - a n = a 1 - a 0) -- Definition of arithmetic sequence
    (h_geom_seq : ∀ n, b (n + 1) / b n = b 1 / b 0) -- Definition of geometric sequence
    (h_a3_a11 : a 3 + a 11 = 8) -- Condition a_3 + a_11 = 8
    (h_b7_a7 : b 7 = a 7) -- Condition b_7 = a_7
    : b 6 * b 8 = 16 := -- Prove that b_6 * b_8 = 16
sorry

end arithmetic_geometric_sequence_l617_61799


namespace smaller_angle_at_3_20_correct_l617_61712

noncomputable def smaller_angle_at_3_20 : Float :=
  let degrees_per_minute_for_minute_hand := 360 / 60
  let degrees_per_minute_for_hour_hand := 360 / (60 * 12)
  let initial_hour_hand_position := 90.0  -- 3 o'clock position
  let minute_past_three := 20
  let minute_hand_movement := minute_past_three * degrees_per_minute_for_minute_hand
  let hour_hand_movement := minute_past_three * degrees_per_minute_for_hour_hand
  let current_hour_hand_position := initial_hour_hand_position + hour_hand_movement
  let angle_between_hands := minute_hand_movement - current_hour_hand_position
  if angle_between_hands < 0 then
    -angle_between_hands
  else
    angle_between_hands

theorem smaller_angle_at_3_20_correct : smaller_angle_at_3_20 = 20.0 := by
  sorry

end smaller_angle_at_3_20_correct_l617_61712


namespace kellys_apples_l617_61765

def apples_kelly_needs_to_pick := 49
def total_apples := 105

theorem kellys_apples :
  ∃ x : ℕ, x + apples_kelly_needs_to_pick = total_apples ∧ x = 56 :=
sorry

end kellys_apples_l617_61765


namespace kaleb_final_score_l617_61789

variable (score_first_half : ℝ) (bonus_special_q : ℝ) (bonus_streak : ℝ) (score_second_half : ℝ) (penalty_speed_round : ℝ) (penalty_lightning_round : ℝ)

-- Given conditions from the problem statement
def kaleb_initial_scores (score_first_half score_second_half : ℝ) := 
  score_first_half = 43 ∧ score_second_half = 23

def kaleb_bonuses (score_first_half bonus_special_q bonus_streak : ℝ) :=
  bonus_special_q = 0.20 * score_first_half ∧ bonus_streak = 0.05 * score_first_half

def kaleb_penalties (score_second_half penalty_speed_round penalty_lightning_round : ℝ) := 
  penalty_speed_round = 0.10 * score_second_half ∧ penalty_lightning_round = 0.08 * score_second_half

-- The final score adjusted with all bonuses and penalties
def kaleb_adjusted_score (score_first_half score_second_half bonus_special_q bonus_streak penalty_speed_round penalty_lightning_round : ℝ) : ℝ := 
  score_first_half + bonus_special_q + bonus_streak + score_second_half - penalty_speed_round - penalty_lightning_round

theorem kaleb_final_score :
  kaleb_initial_scores score_first_half score_second_half ∧
  kaleb_bonuses score_first_half bonus_special_q bonus_streak ∧
  kaleb_penalties score_second_half penalty_speed_round penalty_lightning_round →
  kaleb_adjusted_score score_first_half score_second_half bonus_special_q bonus_streak penalty_speed_round penalty_lightning_round = 72.61 :=
by
  intros
  sorry

end kaleb_final_score_l617_61789


namespace min_jumps_to_visit_all_points_and_return_l617_61795

theorem min_jumps_to_visit_all_points_and_return (n : ℕ) (h : n = 2016) : 
  ∀ jumps : ℕ, (∀ p : Fin n, ∃ k : ℕ, p = (2 * k) % n ∨ p = (3 * k) % n) → 
  jumps = 2017 :=
by 
  intros jumps h
  sorry

end min_jumps_to_visit_all_points_and_return_l617_61795


namespace pencils_sold_is_correct_l617_61731

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

end pencils_sold_is_correct_l617_61731


namespace new_cube_edge_length_l617_61757

theorem new_cube_edge_length
  (a1 a2 a3 : ℝ)
  (h1 : a1 = 3) 
  (h2 : a2 = 4) 
  (h3 : a3 = 5) :
  (a1^3 + a2^3 + a3^3)^(1/3) = 6 := by
sorry

end new_cube_edge_length_l617_61757


namespace exponent_tower_divisibility_l617_61740

theorem exponent_tower_divisibility (h1 h2 : ℕ) (Hh1 : h1 ≥ 3) (Hh2 : h2 ≥ 3) : 
  (2 ^ (5 ^ (2 ^ (5 ^ h1))) + 4 ^ (5 ^ (4 ^ (5 ^ h2)))) % 2008 = 0 := by
  sorry

end exponent_tower_divisibility_l617_61740


namespace ten_faucets_fill_time_l617_61729

theorem ten_faucets_fill_time (rate : ℕ → ℕ → ℝ) (gallons : ℕ) (minutes : ℝ) :
  rate 5 9 = 150 / 5 ∧
  rate 10 135 = 75 / 30 * rate 10 9 / 0.9 * 60 →
  9 * 60 / 30 * 75 / 10 * 60 = 135 :=
sorry

end ten_faucets_fill_time_l617_61729


namespace ara_current_height_l617_61730

theorem ara_current_height (original_height : ℚ) (shea_growth_ratio : ℚ) (ara_growth_ratio : ℚ) (shea_current_height : ℚ) (h1 : shea_growth_ratio = 0.25) (h2 : ara_growth_ratio = 0.75) (h3 : shea_current_height = 75) (h4 : shea_current_height = original_height * (1 + shea_growth_ratio)) : 
  original_height * (1 + ara_growth_ratio * shea_growth_ratio) = 71.25 := 
by
  sorry

end ara_current_height_l617_61730


namespace power_simplification_l617_61724

noncomputable def sqrt2_six : ℝ := 6 ^ (1 / 2)
noncomputable def sqrt3_six : ℝ := 6 ^ (1 / 3)

theorem power_simplification :
  (sqrt2_six / sqrt3_six) = 6 ^ (1 / 6) :=
  sorry

end power_simplification_l617_61724


namespace correct_assignment_l617_61744

-- Definition of conditions
def is_variable_free (e : String) : Prop := -- a simplistic placeholder
  e ∈ ["A", "B", "C", "D", "x"]

def valid_assignment (lhs : String) (rhs : String) : Prop :=
  is_variable_free lhs ∧ ¬(is_variable_free rhs)

-- The statement of the proof problem
theorem correct_assignment : valid_assignment "A" "A * A + A - 2" :=
by
  sorry

end correct_assignment_l617_61744


namespace large_circle_diameter_l617_61768

theorem large_circle_diameter (r : ℝ) (R : ℝ) (R' : ℝ) :
  r = 2 ∧ R = 2 * r ∧ R' = R + r → 2 * R' = 12 :=
by
  intros h
  sorry

end large_circle_diameter_l617_61768


namespace original_flow_rate_l617_61727

theorem original_flow_rate (x : ℝ) (h : 2 = 0.6 * x - 1) : x = 5 :=
by
  sorry

end original_flow_rate_l617_61727


namespace sequence_a_n_2013_l617_61706

theorem sequence_a_n_2013 (a : ℕ → ℤ) (h1 : a 1 = 3) (h2 : a 2 = 6)
  (h : ∀ n, a (n + 2) = a (n + 1) - a n) :
  a 2013 = 3 :=
sorry

end sequence_a_n_2013_l617_61706


namespace equilateral_triangle_area_l617_61735

theorem equilateral_triangle_area (h : ℝ) 
  (height_eq : h = 2 * Real.sqrt 3) :
  ∃ (A : ℝ), A = 4 * Real.sqrt 3 :=
by
  sorry

end equilateral_triangle_area_l617_61735


namespace even_function_f_l617_61741

noncomputable def f (a b c x : ℝ) := a * Real.cos x + b * x^2 + c

theorem even_function_f (a b c : ℝ) (h1 : f a b c 1 = 1) : f a b c (-1) = f a b c 1 := by
  sorry

end even_function_f_l617_61741


namespace expression_evaluation_l617_61711

theorem expression_evaluation (a b : ℤ) (h1 : a = 4) (h2 : b = -2) : -a - b^4 + a * b = -28 := 
by 
  sorry

end expression_evaluation_l617_61711


namespace find_expression_for_f_l617_61762

noncomputable def f (x a b : ℝ) : ℝ := (x + a) * (b * x + 2 * a)

-- Assuming a, b ∈ ℝ, f(x) is even, and range of f(x) is (-∞, 2]
theorem find_expression_for_f (a b : ℝ) (h1 : ∀ x : ℝ, f x a b = f (-x) a b) (h2 : ∀ y : ℝ, ∃ x : ℝ, f x a b = y → y ≤ 2):
  f x a b = -x^2 + 2 :=
by 
  sorry

end find_expression_for_f_l617_61762


namespace discriminant_nonnegative_l617_61705

theorem discriminant_nonnegative {x : ℤ} (a : ℝ) (h₁ : x^2 * (49 - 40 * x^2) ≥ 0) :
  a = 0 ∨ a = 1 ∨ a = -1 ∨ a = 5/2 ∨ a = -5/2 := sorry

end discriminant_nonnegative_l617_61705


namespace fraction_computation_l617_61775

theorem fraction_computation (p q s u : ℚ)
  (hpq : p / q = 5 / 2)
  (hsu : s / u = 7 / 11) :
  (5 * p * s - 3 * q * u) / (7 * q * u - 4 * p * s) = 109 / 14 := 
by
  sorry

end fraction_computation_l617_61775


namespace packed_tents_and_food_truck_arrangements_minimum_transportation_cost_l617_61778

-- Define the conditions
def total_items : ℕ := 320
def tents_more_than_food : ℕ := 80
def total_trucks : ℕ := 8
def type_A_tent_capacity : ℕ := 40
def type_A_food_capacity : ℕ := 10
def type_B_tent_capacity : ℕ := 20
def type_B_food_capacity : ℕ := 20
def type_A_cost : ℕ := 4000
def type_B_cost : ℕ := 3600

-- Questions to prove:
theorem packed_tents_and_food:
  ∃ t f : ℕ, t + f = total_items ∧ t = f + tents_more_than_food ∧ t = 200 ∧ f = 120 :=
sorry

theorem truck_arrangements:
  ∃ A B : ℕ, A + B = total_trucks ∧
    (A * type_A_tent_capacity + B * type_B_tent_capacity = 200) ∧
    (A * type_A_food_capacity + B * type_B_food_capacity = 120) ∧
    ((A = 2 ∧ B = 6) ∨ (A = 3 ∧ B = 5) ∨ (A = 4 ∧ B = 4)) :=
sorry

theorem minimum_transportation_cost:
  ∃ A B : ℕ, A = 2 ∧ B = 6 ∧ A * type_A_cost + B * type_B_cost = 29600 :=
sorry

end packed_tents_and_food_truck_arrangements_minimum_transportation_cost_l617_61778


namespace find_n_l617_61788

theorem find_n (q d : ℕ) (hq : q = 25) (hd : d = 10) (h : 30 * q + 20 * d = 5 * q + n * d) : n = 83 := by
  sorry

end find_n_l617_61788


namespace negation_of_universal_statement_l617_61766

theorem negation_of_universal_statement :
  (¬ (∀ x : ℝ, |x| + x^2 ≥ 0)) ↔ (∃ x : ℝ, |x| + x^2 < 0) :=
by
  sorry

end negation_of_universal_statement_l617_61766


namespace cousin_points_correct_l617_61760

-- Conditions translated to definitions
def paul_points : ℕ := 3103
def total_points : ℕ := 5816

-- Dependent condition to get cousin's points
def cousin_points : ℕ := total_points - paul_points

-- The goal of our proof problem
theorem cousin_points_correct : cousin_points = 2713 :=
by
    sorry

end cousin_points_correct_l617_61760


namespace intercepts_of_line_l617_61701

theorem intercepts_of_line :
  (∀ x y : ℝ, (x = 4 ∨ y = -3) → (x / 4 - y / 3 = 1)) ∧ (∀ x y : ℝ, (x / 4 = 1 ∧ y = 0) ∧ (x = 0 ∧ y / 3 = -1)) :=
by
  sorry

end intercepts_of_line_l617_61701


namespace num_integers_satisfying_conditions_l617_61780

theorem num_integers_satisfying_conditions : 
  ∃ n : ℕ, 
    (120 < n) ∧ (n < 250) ∧ (n % 5 = n % 7) :=
sorry

axiom num_integers_with_conditions : ℕ
@[simp] lemma val_num_integers_with_conditions : num_integers_with_conditions = 25 :=
sorry

end num_integers_satisfying_conditions_l617_61780


namespace longest_side_l617_61792

theorem longest_side (l w : ℝ) 
  (h1 : 2 * l + 2 * w = 240)
  (h2 : l * w = 2880) :
  l = 86.835 ∨ w = 86.835 :=
sorry

end longest_side_l617_61792


namespace probability_of_exactly_two_dice_showing_3_l617_61754

-- Definition of the problem conditions
def n_dice : ℕ := 5
def sides : ℕ := 5
def prob_showing_3 : ℚ := 1/5
def prob_not_showing_3 : ℚ := 4/5
def way_to_choose_2_of_5 : ℕ := Nat.choose 5 2

-- Lean proof problem statement
theorem probability_of_exactly_two_dice_showing_3 : 
  (10 : ℚ) * (prob_showing_3 ^ 2) * (prob_not_showing_3 ^ 3) = 640 / 3125 := 
by sorry

end probability_of_exactly_two_dice_showing_3_l617_61754


namespace inequality_solution_l617_61748

theorem inequality_solution (x : ℝ) : x^2 + x - 20 < 0 ↔ -5 < x ∧ x < 4 := 
by
  sorry

end inequality_solution_l617_61748


namespace point_in_second_quadrant_l617_61743

variable (m : ℝ)

/-- 
If point P(m-1, 3) is in the second quadrant, 
then a possible value of m is -1
--/
theorem point_in_second_quadrant (h1 : (m - 1 < 0)) : m = -1 :=
by sorry

end point_in_second_quadrant_l617_61743


namespace total_area_of_removed_triangles_l617_61764

theorem total_area_of_removed_triangles (x r s : ℝ) (h1 : (x - r)^2 + (x - s)^2 = 15^2) :
  4 * (1/2 * r * s) = 112.5 :=
by
  sorry

end total_area_of_removed_triangles_l617_61764


namespace probability_correct_l617_61736

noncomputable def probability_parallel_not_coincident : ℚ :=
  let total_points := 6
  let lines := total_points.choose 2
  let total_ways := lines * lines
  let parallel_not_coincident_pairs := 12
  parallel_not_coincident_pairs / total_ways

theorem probability_correct :
  probability_parallel_not_coincident = 4 / 75 := by
  sorry

end probability_correct_l617_61736


namespace jam_fraction_left_after_dinner_l617_61728

noncomputable def jam_left_after_dinner (initial: ℚ) (lunch_fraction: ℚ) (dinner_fraction: ℚ) : ℚ :=
  initial - (initial * lunch_fraction) - ((initial - (initial * lunch_fraction)) * dinner_fraction)

theorem jam_fraction_left_after_dinner :
  jam_left_after_dinner 1 (1/3) (1/7) = (4/7) :=
by
  sorry

end jam_fraction_left_after_dinner_l617_61728


namespace age_of_jerry_l617_61755

variable (M J : ℕ)

theorem age_of_jerry (h1 : M = 2 * J - 5) (h2 : M = 19) : J = 12 := by
  sorry

end age_of_jerry_l617_61755


namespace benches_count_l617_61785

theorem benches_count (num_people_base6 : ℕ) (people_per_bench : ℕ) (num_people_base10 : ℕ) (num_benches : ℕ) :
  num_people_base6 = 204 ∧ people_per_bench = 2 ∧ num_people_base10 = 76 ∧ num_benches = 38 →
  (num_people_base10 = 2 * 6^2 + 0 * 6^1 + 4 * 6^0) ∧
  (num_benches = num_people_base10 / people_per_bench) :=
by
  sorry

end benches_count_l617_61785


namespace smallest_positive_integer_a_l617_61726

theorem smallest_positive_integer_a (a : ℕ) (h1 : 0 < a) (h2 : ∃ b : ℕ, 3150 * a = b^2) : a = 14 :=
by
  sorry

end smallest_positive_integer_a_l617_61726


namespace find_solution_l617_61771

def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

def sum_first_p_squares (p : ℕ) : ℕ := p * (p + 1) * (2 * p + 1) / 6

theorem find_solution : ∃ (n p : ℕ), p.Prime ∧ sum_first_n n = 3 * sum_first_p_squares p ∧ (n, p) = (5, 2) := 
by
  sorry

end find_solution_l617_61771


namespace shiela_bottles_l617_61773

theorem shiela_bottles (num_stars : ℕ) (stars_per_bottle : ℕ) (num_bottles : ℕ) 
  (h1 : num_stars = 45) (h2 : stars_per_bottle = 5) : num_bottles = 9 :=
sorry

end shiela_bottles_l617_61773


namespace proved_problem_l617_61779

theorem proved_problem (x y p n k : ℕ) (h_eq : x^n + y^n = p^k)
  (h1 : n > 1)
  (h2 : n % 2 = 1)
  (h3 : Nat.Prime p)
  (h4 : p % 2 = 1) :
  ∃ l : ℕ, n = p^l :=
by sorry

end proved_problem_l617_61779


namespace opposite_of_neg_one_fourth_l617_61747

def opposite_of (x : ℝ) : ℝ := -x

theorem opposite_of_neg_one_fourth :
  opposite_of (-1/4) = 1/4 :=
by
  sorry

end opposite_of_neg_one_fourth_l617_61747


namespace edward_initial_amount_l617_61769

theorem edward_initial_amount (spent received final_amount : ℤ) 
  (h_spent : spent = 17) 
  (h_received : received = 10) 
  (h_final : final_amount = 7) : 
  ∃ initial_amount : ℤ, (initial_amount - spent + received = final_amount) ∧ (initial_amount = 14) :=
by
  sorry

end edward_initial_amount_l617_61769


namespace hired_is_B_l617_61794

-- Define the individuals
inductive Person : Type
| A : Person
| B : Person
| C : Person
| D : Person

open Person

-- Define the statements made by each person
def statement (p : Person) (hired : Person) : Prop :=
  match p with
  | A => hired = C
  | B => hired ≠ B
  | C => hired = D
  | D => hired ≠ D

-- The main theorem is to prove B is hired given the conditions
theorem hired_is_B :
  (∃! p : Person, ∃ t : Person → Prop,
    (∀ h : Person, t h ↔ h = p) ∧
    (∃ q : Person, statement q q ∧ ∀ r : Person, r ≠ q → ¬statement r q) ∧
    t B) :=
by
  sorry

end hired_is_B_l617_61794


namespace orange_juice_serving_size_l617_61734

theorem orange_juice_serving_size (n_servings : ℕ) (c_concentrate : ℕ) (v_concentrate : ℕ) (c_water_per_concentrate : ℕ)
    (v_cans : ℕ) (expected_serving_size : ℕ) 
    (h1 : n_servings = 200)
    (h2 : c_concentrate = 60)
    (h3 : v_concentrate = 5)
    (h4 : c_water_per_concentrate = 3)
    (h5 : v_cans = 5)
    (h6 : expected_serving_size = 6) : 
   (c_concentrate * v_concentrate + c_concentrate * c_water_per_concentrate * v_cans) / n_servings = expected_serving_size := 
by 
  sorry

end orange_juice_serving_size_l617_61734


namespace initial_percentage_of_alcohol_l617_61761

theorem initial_percentage_of_alcohol 
  (P: ℝ)
  (h_condition1 : 18 * P / 100 = 21 * 17.14285714285715 / 100) : 
  P = 20 :=
by 
  sorry

end initial_percentage_of_alcohol_l617_61761


namespace correct_option_D_l617_61713

variables (a b c : ℤ)

theorem correct_option_D : -2 * a + 3 * (b - 1) = -2 * a + 3 * b - 3 := 
by
  sorry

end correct_option_D_l617_61713


namespace neg_p_l617_61793

variable (x : ℝ)

def p : Prop := ∃ x : ℝ, Real.exp x - x - 1 ≤ 0

theorem neg_p : ¬p ↔ (∀ x : ℝ, Real.exp x - x - 1 > 0) :=
by
  sorry

end neg_p_l617_61793


namespace sum_gcf_lcm_l617_61745

theorem sum_gcf_lcm (a b : ℕ) (h_a : a = 8) (h_b : b = 12) :
  Nat.gcd a b + Nat.lcm a b = 28 :=
by
  rw [h_a, h_b]
  -- here you'd typically use known theorems about gcd and lcm for specific numbers
  sorry

end sum_gcf_lcm_l617_61745


namespace f_2011_is_zero_l617_61738

noncomputable def f : ℝ → ℝ := sorry

-- Conditions
axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom periodic_f : ∀ x : ℝ, f (x + 2) = f (x) + f (1)

-- Theorem stating the mathematically equivalent proof problem
theorem f_2011_is_zero : f (2011) = 0 :=
sorry

end f_2011_is_zero_l617_61738


namespace arithmetic_mean_of_fractions_l617_61717

theorem arithmetic_mean_of_fractions :
  let a := (9 : ℝ) / 12
  let b := (5 : ℝ) / 6
  let c := (11 : ℝ) / 12
  (a + c) / 2 = b := 
by
  sorry

end arithmetic_mean_of_fractions_l617_61717


namespace frac_add_eq_l617_61702

theorem frac_add_eq : (2 / 5) + (3 / 10) = 7 / 10 := 
by
  sorry

end frac_add_eq_l617_61702


namespace parabola_through_point_l617_61715

theorem parabola_through_point (a b : ℝ) (ha : 0 < a) :
  ∃ f : ℝ → ℝ, (∀ x, f x = -a*x^2 + b*x + 1) ∧ f 0 = 1 :=
by
  -- We are given a > 0
  -- We need to show there exists a parabola of the form y = -a*x^2 + b*x + 1 passing through (0,1)
  sorry

end parabola_through_point_l617_61715


namespace option_A_option_C_l617_61739

/-- Definition of the set M such that M = {a | a = x^2 - y^2, x, y ∈ ℤ} -/
def M := {a : ℤ | ∃ x y : ℤ, a = x^2 - y^2}

/-- Definition of the set B such that B = {b | b = 2n + 1, n ∈ ℕ} -/
def B := {b : ℤ | ∃ n : ℕ, b = 2 * n + 1}

theorem option_A (a1 a2 : ℤ) (ha1 : a1 ∈ M) (ha2 : a2 ∈ M) : a1 * a2 ∈ M := sorry

theorem option_C : B ⊆ M := sorry

end option_A_option_C_l617_61739


namespace number_of_permutations_l617_61786

noncomputable def num_satisfying_permutations : ℕ :=
  Nat.choose 15 7

theorem number_of_permutations : num_satisfying_permutations = 6435 := by
  sorry

end number_of_permutations_l617_61786


namespace kerosene_consumption_reduction_l617_61723

variable (P C : ℝ)

/-- In the new budget, with the price of kerosene oil rising by 25%, 
    we need to prove that consumption must be reduced by 20% to maintain the same expenditure. -/
theorem kerosene_consumption_reduction (h : 1.25 * P * C_new = P * C) : C_new = 0.8 * C := by
  sorry

end kerosene_consumption_reduction_l617_61723


namespace pencils_left_l617_61797

def initial_pencils := 4527
def given_to_dorothy := 1896
def given_to_samuel := 754
def given_to_alina := 307
def total_given := given_to_dorothy + given_to_samuel + given_to_alina
def remaining_pencils := initial_pencils - total_given

theorem pencils_left : remaining_pencils = 1570 := by
  sorry

end pencils_left_l617_61797


namespace range_of_a_l617_61777

namespace InequalityProblem

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (1 < x ∧ x < 2) → (x - 1)^2 < Real.log x / Real.log a) ↔ (1 < a ∧ a ≤ 2) :=
by
  sorry

end InequalityProblem

end range_of_a_l617_61777


namespace find_side_c_of_triangle_ABC_l617_61709

theorem find_side_c_of_triangle_ABC
  (a b : ℝ)
  (cosA : ℝ)
  (c : ℝ) :
  a = 4 * Real.sqrt 5 →
  b = 5 →
  cosA = 3 / 5 →
  c^2 - 3 * c - 55 = 0 →
  c = 11 := by
  intros ha hb hcosA hquadratic
  sorry

end find_side_c_of_triangle_ABC_l617_61709


namespace geometric_sequence_arithmetic_condition_l617_61746

variable {a_n : ℕ → ℝ}
variable {q : ℝ}

-- Conditions of the problem
def is_geometric_sequence (a_n : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a_n (n + 1) = a_n n * q

def positive_terms (a_n : ℕ → ℝ) : Prop :=
  ∀ n, a_n n > 0

def arithmetic_sequence_cond (a_n : ℕ → ℝ) : Prop :=
  a_n 2 - (1 / 2) * a_n 3 = (1 / 2) * a_n 3 - a_n 1

-- Problem: Prove the required ratio equals the given value
theorem geometric_sequence_arithmetic_condition
  (h_geo: is_geometric_sequence a_n q)
  (h_pos: positive_terms a_n)
  (h_arith: arithmetic_sequence_cond a_n)
  (h_q_ne_one: q ≠ 1) :
  (a_n 4 + a_n 5) / (a_n 3 + a_n 4) = (1 + Real.sqrt 5) / 2 :=
sorry

end geometric_sequence_arithmetic_condition_l617_61746


namespace triangle_area_ab_l617_61774

theorem triangle_area_ab (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
    (hline : ∀ x y : ℝ, 2 * a * x + 3 * b * y = 12) (harea : (1/2) * (6 / a) * (4 / b) = 9) : 
    a * b = 4 / 3 :=
by 
  sorry

end triangle_area_ab_l617_61774


namespace base_length_of_parallelogram_l617_61756

theorem base_length_of_parallelogram (A : ℕ) (H : ℕ) (Base : ℕ) (hA : A = 576) (hH : H = 48) (hArea : A = Base * H) : 
  Base = 12 := 
by 
  -- We skip the proof steps since we only need to provide the Lean theorem statement.
  sorry

end base_length_of_parallelogram_l617_61756


namespace renovation_project_truck_load_l617_61782

theorem renovation_project_truck_load (sand : ℝ) (dirt : ℝ) (cement : ℝ)
  (h1 : sand = 0.17) (h2 : dirt = 0.33) (h3 : cement = 0.17) :
  sand + dirt + cement = 0.67 :=
by
  sorry

end renovation_project_truck_load_l617_61782


namespace school_problem_proof_l617_61787

noncomputable def solve_school_problem (B G x y z : ℕ) :=
  B + G = 300 ∧
  B * y = x * G ∧
  G = (x * 300) / 100 →
  z = 300 - 3 * x - (300 * x) / (x + y)

theorem school_problem_proof (B G x y z : ℕ) :
  solve_school_problem B G x y z :=
by
  sorry

end school_problem_proof_l617_61787


namespace complement_U_A_l617_61721

-- Define the universal set U and the subset A
def U : Set ℕ := {1, 2, 3, 4}
def A : Set ℕ := {1, 3, 4}

-- Define the complement of A relative to the universal set U
def complement (U A : Set ℕ) : Set ℕ := { x | x ∈ U ∧ x ∉ A }

-- The theorem we want to prove
theorem complement_U_A : complement U A = {2} := by
  sorry

end complement_U_A_l617_61721


namespace not_divisible_by_n_plus_4_l617_61737

theorem not_divisible_by_n_plus_4 (n : ℕ) (h : 0 < n) : ¬ (n + 4 ∣ n^2 + 8 * n + 15) := 
sorry

end not_divisible_by_n_plus_4_l617_61737


namespace star_polygon_n_value_l617_61733

theorem star_polygon_n_value (n : ℕ) (A B : ℕ → ℝ) (h1 : ∀ i, A i = B i - 20)
    (h2 : 360 = n * 20) : n = 18 :=
by {
  sorry
}

end star_polygon_n_value_l617_61733


namespace probability_of_green_ball_l617_61772

theorem probability_of_green_ball :
  let P_X := 0.2
  let P_Y := 0.5
  let P_Z := 0.3
  let P_green_given_X := 5 / 10
  let P_green_given_Y := 3 / 10
  let P_green_given_Z := 8 / 10
  P_green_given_X * P_X + P_green_given_Y * P_Y + P_green_given_Z * P_Z = 0.49 :=
by {
  sorry
}

end probability_of_green_ball_l617_61772


namespace g_at_3_l617_61790

def g (x : ℝ) : ℝ := x^3 - 2 * x^2 + x

theorem g_at_3 : g 3 = 12 := by
  sorry

end g_at_3_l617_61790


namespace all_terms_are_positive_integers_terms_product_square_l617_61749

def seq (x : ℕ → ℕ) : Prop :=
  x 1 = 1 ∧
  x 2 = 4 ∧
  ∀ n > 1, x n = Nat.sqrt (x (n - 1) * x (n + 1) + 1)

theorem all_terms_are_positive_integers (x : ℕ → ℕ) (h : seq x) : ∀ n, x n > 0 :=
sorry

theorem terms_product_square (x : ℕ → ℕ) (h : seq x) : ∀ n ≥ 1, ∃ k, 2 * x n * x (n + 1) + 1 = k ^ 2 :=
sorry

end all_terms_are_positive_integers_terms_product_square_l617_61749


namespace total_apples_and_pears_l617_61763

theorem total_apples_and_pears (x y : ℤ) 
  (h1 : x = 3 * (y / 2 + 1)) 
  (h2 : x = 5 * (y / 4 - 3)) : 
  x + y = 39 :=
sorry

end total_apples_and_pears_l617_61763


namespace visit_orders_l617_61710

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

noncomputable def num_permutations_cities (pohang busan geoncheon gimhae gyeongju : Type) : ℕ :=
  factorial 4

theorem visit_orders (pohang busan geoncheon gimhae gyeongju : Type) :
  num_permutations_cities pohang busan geoncheon gimhae gyeongju = 24 :=
by
  unfold num_permutations_cities
  norm_num
  sorry

end visit_orders_l617_61710


namespace digit_sum_of_4_digit_number_l617_61751

theorem digit_sum_of_4_digit_number (abcd : ℕ) (H1 : 1000 ≤ abcd ∧ abcd < 10000) (erased_digit: ℕ) (H2: erased_digit < 10) (H3 : 100*(abcd / 1000) + 10*(abcd % 1000 / 100) + (abcd % 100 / 10) + erased_digit = 6031): 
    (abcd / 1000 + abcd % 1000 / 100 + abcd % 100 / 10 + abcd % 10 = 20) :=
sorry

end digit_sum_of_4_digit_number_l617_61751


namespace students_behind_minyoung_l617_61752

-- Definition of the initial conditions
def total_students : ℕ := 35
def students_in_front_of_minyoung : ℕ := 27

-- The question we want to prove
theorem students_behind_minyoung : (total_students - (students_in_front_of_minyoung + 1) = 7) := 
by 
  sorry

end students_behind_minyoung_l617_61752


namespace more_students_than_guinea_pigs_l617_61798

theorem more_students_than_guinea_pigs (students_per_classroom guinea_pigs_per_classroom classrooms : ℕ)
  (h1 : students_per_classroom = 24) 
  (h2 : guinea_pigs_per_classroom = 3) 
  (h3 : classrooms = 6) : 
  (students_per_classroom * classrooms) - (guinea_pigs_per_classroom * classrooms) = 126 := 
by
  sorry

end more_students_than_guinea_pigs_l617_61798


namespace minimum_value_a_plus_4b_l617_61719

theorem minimum_value_a_plus_4b (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : (1 / a) + (1 / b) = 1) : a + 4 * b ≥ 9 :=
sorry

end minimum_value_a_plus_4b_l617_61719


namespace math_equivalence_l617_61796

theorem math_equivalence (m n : ℤ) (h : |m - 2023| + (n + 2024)^2 = 0) : (m + n) ^ 2023 = -1 := 
by
  sorry

end math_equivalence_l617_61796


namespace smallest_prime_after_seven_consecutive_nonprimes_l617_61750

theorem smallest_prime_after_seven_consecutive_nonprimes :
  ∃ p, p > 96 ∧ Nat.Prime p ∧ ∀ n, 90 ≤ n ∧ n ≤ 96 → ¬Nat.Prime n :=
by
  sorry

end smallest_prime_after_seven_consecutive_nonprimes_l617_61750


namespace initial_average_production_l617_61781

theorem initial_average_production (n : ℕ) (today_production : ℕ) 
  (new_average : ℕ) (initial_average : ℕ) :
  n = 1 → today_production = 60 → new_average = 55 → initial_average = (new_average * (n + 1) - today_production) → initial_average = 50 :=
by
  intros h1 h2 h3 h4
  -- Insert further proof here
  sorry

end initial_average_production_l617_61781


namespace peaches_picked_up_l617_61776

variable (initial_peaches : ℕ) (final_peaches : ℕ)

theorem peaches_picked_up :
  initial_peaches = 13 →
  final_peaches = 55 →
  final_peaches - initial_peaches = 42 :=
by
  intros
  sorry

end peaches_picked_up_l617_61776


namespace average_value_function_example_l617_61758

def average_value_function (f : ℝ → ℝ) (a b : ℝ) :=
  ∃ x0 : ℝ, a < x0 ∧ x0 < b ∧ f x0 = (f b - f a) / (b - a)

theorem average_value_function_example :
  average_value_function (λ x => x^2 - m * x - 1) (-1) (1) → 
  ∃ m : ℝ, 0 < m ∧ m < 2 :=
by
  intros h
  sorry

end average_value_function_example_l617_61758


namespace blossom_room_area_l617_61716

theorem blossom_room_area
  (ft_to_in : ℕ)
  (length_ft : ℕ)
  (width_ft : ℕ)
  (ft_to_in_def : ft_to_in = 12)
  (length_width_def : length_ft = 10)
  (room_square : length_ft = width_ft) :
  (length_ft * ft_to_in) * (width_ft * ft_to_in) = 14400 := 
by
  -- ft_to_in is the conversion factor from feet to inches
  -- length_ft and width_ft are both 10 according to length_width_def and room_square
  -- So, we have (10 * 12) * (10 * 12) = 14400
  sorry

end blossom_room_area_l617_61716


namespace cottonCandyToPopcornRatio_l617_61720

variable (popcornEarningsPerDay : ℕ) (netEarnings : ℕ) (rentCost : ℕ) (ingredientCost : ℕ)

theorem cottonCandyToPopcornRatio
  (h_popcorn : popcornEarningsPerDay = 50)
  (h_net : netEarnings = 895)
  (h_rent : rentCost = 30)
  (h_ingredient : ingredientCost = 75)
  (h : ∃ C : ℕ, 5 * C + 5 * popcornEarningsPerDay - rentCost - ingredientCost = netEarnings) :
  ∃ r : ℕ, r = 3 :=
by
  sorry

end cottonCandyToPopcornRatio_l617_61720


namespace factorize_diff_squares_1_factorize_diff_squares_2_factorize_common_term_l617_61725

-- Proof Problem 1
theorem factorize_diff_squares_1 (x y : ℝ) :
  4 * x^2 - 9 * y^2 = (2 * x + 3 * y) * (2 * x - 3 * y) :=
sorry

-- Proof Problem 2
theorem factorize_diff_squares_2 (a b : ℝ) :
  -16 * a^2 + 25 * b^2 = (5 * b + 4 * a) * (5 * b - 4 * a) :=
sorry

-- Proof Problem 3
theorem factorize_common_term (x y : ℝ) :
  x^3 * y - x * y^3 = x * y * (x + y) * (x - y) :=
sorry

end factorize_diff_squares_1_factorize_diff_squares_2_factorize_common_term_l617_61725


namespace yellow_ball_range_l617_61770

-- Definitions
def probability_condition (x : ℕ) : Prop :=
  (20 / 100 : ℝ) ≤ (4 * x / ((x + 2) * (x + 1))) ∧ (4 * x / ((x + 2) * (x + 1))) ≤ (33 / 100)

theorem yellow_ball_range (x : ℕ) : probability_condition x ↔ 9 ≤ x ∧ x ≤ 16 := 
by
  sorry

end yellow_ball_range_l617_61770


namespace card_2_in_box_Q_l617_61784

theorem card_2_in_box_Q (P Q : Finset ℕ) (hP : P.card = 3) (hQ : Q.card = 5) 
  (hdisjoint : Disjoint P Q) (huniv : P ∪ Q = (Finset.range 9).erase 0)
  (hsum_eq : P.sum id = Q.sum id) :
  2 ∈ Q := 
sorry

end card_2_in_box_Q_l617_61784


namespace original_ghee_quantity_l617_61707

theorem original_ghee_quantity (x : ℝ) (H1 : 0.60 * x + 10 = ((1 + 0.40 * x) * 0.80)) :
  x = 10 :=
sorry

end original_ghee_quantity_l617_61707


namespace min_value_of_n_l617_61783

theorem min_value_of_n (n : ℕ) (k : ℚ) (h1 : k > 0.9999) 
    (h2 : 4 * n * (n - 1) * (1 - k) = 1) : 
    n = 51 :=
sorry

end min_value_of_n_l617_61783


namespace linear_system_solution_l617_61753

theorem linear_system_solution (x y m : ℝ) (h1 : x + 2 * y = m) (h2 : 2 * x - 3 * y = 4) (h3 : x + y = 7) : 
  m = 9 :=
sorry

end linear_system_solution_l617_61753


namespace permutations_divisibility_l617_61722

theorem permutations_divisibility (n : ℕ) (a b : Fin n → ℕ) 
  (h_n : 2 < n)
  (h_a_perm : ∀ i, ∃ j, a j = i)
  (h_b_perm : ∀ i, ∃ j, b j = i) :
  ∃ (i j : Fin n), i ≠ j ∧ n ∣ (a i * b i - a j * b j) :=
by sorry

end permutations_divisibility_l617_61722


namespace total_cost_of_books_l617_61718

theorem total_cost_of_books
  (C1 : ℝ) (C2 : ℝ)
  (h1 : C1 = 315)
  (h2 : 0.85 * C1 = 1.19 * C2) :
  C1 + C2 = 2565 :=
by 
  sorry

end total_cost_of_books_l617_61718


namespace part1_tangent_line_max_min_values_l617_61759

def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a * x^2
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * x^2 + 2 * a * x
def tangent_line_at (a : ℝ) (x y : ℝ) : ℝ := 9 * x + y - 4

theorem part1 (a : ℝ) : f' a 1 = -9 → a = -6 :=
by
  sorry

theorem tangent_line (a : ℝ) (x y : ℝ) : a = -6 → f a 1 = -5 → tangent_line_at a 1 (-5) = 0 :=
by
  sorry

def interval := Set.Icc (-5 : ℝ) 5

theorem max_min_values (a : ℝ) : a = -6 →
  (∀ x ∈ interval, f a (-5) = -275 ∨ f a 0 = 0 ∨ f a 4 = -32 ∨ f a 5 = -25) →
  (∀ x ∈ interval, f a x ≤ 0 ∧ f a x ≥ -275) :=
by
  sorry

end part1_tangent_line_max_min_values_l617_61759


namespace ants_harvest_time_l617_61714

theorem ants_harvest_time :
  ∃ h : ℕ, (∀ h : ℕ, 24 - 4 * h = 12) ∧ h = 3 := sorry

end ants_harvest_time_l617_61714


namespace multiple_of_5_add_multiple_of_10_l617_61767

theorem multiple_of_5_add_multiple_of_10 (p q : ℤ) (hp : ∃ m : ℤ, p = 5 * m) (hq : ∃ n : ℤ, q = 10 * n) : ∃ k : ℤ, p + q = 5 * k :=
by
  sorry

end multiple_of_5_add_multiple_of_10_l617_61767


namespace Amy_total_crumbs_eq_3z_l617_61732

variable (T C z : ℕ)

-- Given conditions
def total_crumbs_Arthur := T * C = z
def trips_Amy := 2 * T
def crumbs_per_trip_Amy := 3 * C / 2

-- Problem statement
theorem Amy_total_crumbs_eq_3z (h : total_crumbs_Arthur T C z) :
  (trips_Amy T) * (crumbs_per_trip_Amy C) = 3 * z :=
sorry

end Amy_total_crumbs_eq_3z_l617_61732


namespace p_is_necessary_but_not_sufficient_for_q_l617_61703

variable (x : ℝ)

def p : Prop := -1 ≤ x ∧ x ≤ 5
def q : Prop := (x - 5) * (x + 1) < 0

theorem p_is_necessary_but_not_sufficient_for_q : (∀ x, p x → q x) ∧ ¬ (∀ x, q x → p x) := 
sorry

end p_is_necessary_but_not_sufficient_for_q_l617_61703


namespace ratio_of_silver_to_gold_l617_61708

-- Definitions for balloon counts
def gold_balloons : Nat := 141
def black_balloons : Nat := 150
def total_balloons : Nat := 573

-- Define the number of silver balloons S
noncomputable def silver_balloons : Nat :=
  total_balloons - gold_balloons - black_balloons

-- The goal is to prove the ratio of silver to gold balloons is 2
theorem ratio_of_silver_to_gold :
  (silver_balloons / gold_balloons) = 2 := by
  sorry

end ratio_of_silver_to_gold_l617_61708


namespace pencil_price_units_l617_61704

def pencil_price_in_units (pencil_price : ℕ) : ℚ := pencil_price / 10000

theorem pencil_price_units 
  (price_of_pencil : ℕ) 
  (h1 : price_of_pencil = 5000 - 20) : 
  pencil_price_in_units price_of_pencil = 0.5 := 
by
  sorry

end pencil_price_units_l617_61704
