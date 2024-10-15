import Mathlib

namespace NUMINAMATH_GPT_part_I_part_II_l1074_107419

-- Conditions
def p (x m : ℝ) : Prop := x > m → 2 * x - 5 > 0
def q (m : ℝ) : Prop := ∃ x y : ℝ, (x^2 / (m - 1)) + (y^2 / (2 - m)) = 1

-- Statements for proof
theorem part_I (m x : ℝ) (hq: q m) (hp: p x m) : 
  m < 1 ∨ (2 < m ∧ m ≤ 5 / 2) :=
sorry

theorem part_II (m x : ℝ) (hq: ¬ q m ∧ ¬(p x m ∧ q m) ∧ (p x m ∨ q m)) : 
  (1 ≤ m ∧ m ≤ 2) ∨ (m > 5 / 2) :=
sorry

end NUMINAMATH_GPT_part_I_part_II_l1074_107419


namespace NUMINAMATH_GPT_find_numbers_l1074_107423

variables {x y : ℤ}

theorem find_numbers (x y : ℤ) (h1 : x - y = 11) (h2 : x^2 + y^2 = 185) (h3 : (x - y)^2 = 121) :
  (x = 13 ∧ y = 2) ∨ (x = -5 ∧ y = -16) :=
sorry

end NUMINAMATH_GPT_find_numbers_l1074_107423


namespace NUMINAMATH_GPT_businessmen_drink_one_type_l1074_107417

def total_businessmen : ℕ := 35
def coffee_drinkers : ℕ := 18
def tea_drinkers : ℕ := 15
def juice_drinkers : ℕ := 8
def coffee_and_tea_drinkers : ℕ := 6
def tea_and_juice_drinkers : ℕ := 4
def coffee_and_juice_drinkers : ℕ := 3
def all_three_drinkers : ℕ := 2

theorem businessmen_drink_one_type : 
  coffee_drinkers - coffee_and_tea_drinkers - coffee_and_juice_drinkers + all_three_drinkers +
  tea_drinkers - coffee_and_tea_drinkers - tea_and_juice_drinkers + all_three_drinkers +
  juice_drinkers - tea_and_juice_drinkers - coffee_and_juice_drinkers + all_three_drinkers = 21 := 
sorry

end NUMINAMATH_GPT_businessmen_drink_one_type_l1074_107417


namespace NUMINAMATH_GPT_coconut_grove_l1074_107495

theorem coconut_grove (x N : ℕ) (h1 : (x + 4) * 60 + x * N + (x - 4) * 180 = 3 * x * 100) (hx : x = 8) : N = 120 := 
by
  subst hx
  sorry

end NUMINAMATH_GPT_coconut_grove_l1074_107495


namespace NUMINAMATH_GPT_initial_ducks_l1074_107459

theorem initial_ducks (D : ℕ) (h1 : D + 20 = 33) : D = 13 :=
by sorry

end NUMINAMATH_GPT_initial_ducks_l1074_107459


namespace NUMINAMATH_GPT_evaluate_expression_l1074_107497

theorem evaluate_expression : 6 + 4 / 2 = 8 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1074_107497


namespace NUMINAMATH_GPT_maria_baggies_count_l1074_107444

def total_cookies (chocolate_chip : ℕ) (oatmeal : ℕ) : ℕ :=
  chocolate_chip + oatmeal

def baggies_count (total_cookies : ℕ) (cookies_per_bag : ℕ) : ℕ :=
  total_cookies / cookies_per_bag

theorem maria_baggies_count :
  let choco_chip := 2
  let oatmeal := 16
  let cookies_per_bag := 3
  baggies_count (total_cookies choco_chip oatmeal) cookies_per_bag = 6 :=
by
  sorry

end NUMINAMATH_GPT_maria_baggies_count_l1074_107444


namespace NUMINAMATH_GPT_no_integer_solution_system_l1074_107413

theorem no_integer_solution_system (
  x y z : ℤ
) : x^6 + x^3 + x^3 * y + y ≠ 147 ^ 137 ∨ x^3 + x^3 * y + y^2 + y + z^9 ≠ 157 ^ 117 :=
by
  sorry

end NUMINAMATH_GPT_no_integer_solution_system_l1074_107413


namespace NUMINAMATH_GPT_Wendy_did_not_recycle_2_bags_l1074_107436

theorem Wendy_did_not_recycle_2_bags (points_per_bag : ℕ) (total_bags : ℕ) (points_earned : ℕ) (did_not_recycle : ℕ) : 
  points_per_bag = 5 → 
  total_bags = 11 → 
  points_earned = 45 → 
  5 * (11 - did_not_recycle) = 45 → 
  did_not_recycle = 2 :=
by
  intros h_points_per_bag h_total_bags h_points_earned h_equation
  sorry

end NUMINAMATH_GPT_Wendy_did_not_recycle_2_bags_l1074_107436


namespace NUMINAMATH_GPT_range_of_a_l1074_107430

theorem range_of_a 
    (a : ℝ) 
    (f : ℝ → ℝ) 
    (h : ∀ x : ℝ, f x = Real.exp (|x - a|)) 
    (increasing_on_interval : ∀ x y : ℝ, 1 ≤ x → x ≤ y → f x ≤ f y) :
    a ≤ 1 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1074_107430


namespace NUMINAMATH_GPT_g_ge_one_l1074_107484

noncomputable def f (x : ℝ) : ℝ := 3 * x^2 + Real.log x + 4

noncomputable def g (x : ℝ) : ℝ := Real.exp (x - 1) - Real.log x

theorem g_ge_one (x : ℝ) (h : 0 < x) : g x ≥ 1 :=
sorry

end NUMINAMATH_GPT_g_ge_one_l1074_107484


namespace NUMINAMATH_GPT_minimum_value_y_l1074_107403

theorem minimum_value_y (x : ℝ) (hx : x > 2) : 
  ∃ y, y = x + 4 / (x - 2) ∧ ∀ z, (z = x + 4 / (x - 2) → z ≥ 6) :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_y_l1074_107403


namespace NUMINAMATH_GPT_floor_width_is_120_l1074_107469

def tile_length := 25 -- cm
def tile_width := 16 -- cm
def floor_length := 180 -- cm
def max_tiles := 54

theorem floor_width_is_120 :
  ∃ (W : ℝ), W = 120 ∧ (floor_length / tile_width) * W = max_tiles * (tile_length * tile_width) := 
sorry

end NUMINAMATH_GPT_floor_width_is_120_l1074_107469


namespace NUMINAMATH_GPT_thirtieth_term_value_l1074_107416

-- Define the initial term and common difference
def initial_term : ℤ := 3
def common_difference : ℤ := 4

-- Define the nth term of the arithmetic sequence
def nth_term (n : ℕ) : ℤ :=
  initial_term + (n - 1) * common_difference

-- Theorem stating the value of the 30th term
theorem thirtieth_term_value : nth_term 30 = 119 :=
by sorry

end NUMINAMATH_GPT_thirtieth_term_value_l1074_107416


namespace NUMINAMATH_GPT_cars_sold_l1074_107447

theorem cars_sold (sales_Mon sales_Tue sales_Wed cars_Thu_Fri_Sat : ℕ) 
  (mean : ℝ) (h1 : sales_Mon = 8) 
  (h2 : sales_Tue = 3) 
  (h3 : sales_Wed = 10) 
  (h4 : mean = 5.5) 
  (h5 : mean * 6 = sales_Mon + sales_Tue + sales_Wed + cars_Thu_Fri_Sat):
  cars_Thu_Fri_Sat = 12 :=
sorry

end NUMINAMATH_GPT_cars_sold_l1074_107447


namespace NUMINAMATH_GPT_mark_boxes_sold_l1074_107434

theorem mark_boxes_sold (n : ℕ) (M A : ℕ) (h1 : A = n - 2) (h2 : M + A < n) (h3 :  1 ≤ M) (h4 : 1 ≤ A) (hn : n = 12) : M = 1 :=
by
  sorry

end NUMINAMATH_GPT_mark_boxes_sold_l1074_107434


namespace NUMINAMATH_GPT_emily_mean_seventh_score_l1074_107425

theorem emily_mean_seventh_score :
  let a1 := 85
  let a2 := 88
  let a3 := 90
  let a4 := 94
  let a5 := 96
  let a6 := 92
  (a1 + a2 + a3 + a4 + a5 + a6 + a7) / 7 = 91 → a7 = 92 :=
by
  intros
  sorry

end NUMINAMATH_GPT_emily_mean_seventh_score_l1074_107425


namespace NUMINAMATH_GPT_cricket_initial_average_l1074_107424

theorem cricket_initial_average (A : ℕ) (h1 : ∀ A, A * 20 + 137 = 21 * (A + 5)) : A = 32 := by
  -- assumption and proof placeholder
  sorry

end NUMINAMATH_GPT_cricket_initial_average_l1074_107424


namespace NUMINAMATH_GPT_sum_of_fractions_l1074_107426

theorem sum_of_fractions :
  (1 / 10) + (2 / 10) + (3 / 10) + (4 / 10) + (5 / 10) + (6 / 10) + (7 / 10) + (8 / 10) + (10 / 10) + (60 / 10) = 10.6 := by
  sorry

end NUMINAMATH_GPT_sum_of_fractions_l1074_107426


namespace NUMINAMATH_GPT_tangent_expression_l1074_107451

open Real

theorem tangent_expression
  (a : ℕ → ℝ) (b : ℕ → ℝ)
  (geom_seq : ∀ n m, a (n + m) = a n * a m) 
  (arith_seq : ∀ n, b (n + 1) = b n + (b 2 - b 1))
  (cond1 : a 1 * a 6 * a 11 = -3 * sqrt 3)
  (cond2 : b 1 + b 6 + b 11 = 7 * pi) :
  tan ( (b 3 + b 9) / (1 - a 4 * a 8) ) = -sqrt 3 :=
sorry

end NUMINAMATH_GPT_tangent_expression_l1074_107451


namespace NUMINAMATH_GPT_cookie_problem_l1074_107448

theorem cookie_problem : 
  ∃ (B : ℕ), B = 130 ∧ B - 80 = 50 ∧ B/2 + 20 = 85 :=
by
  sorry

end NUMINAMATH_GPT_cookie_problem_l1074_107448


namespace NUMINAMATH_GPT_pipe_fill_time_l1074_107437

variable (t : ℝ)

theorem pipe_fill_time (h1 : 0 < t) (h2 : 0 < t / 5) (h3 : (1 / t) + (5 / t) = 1 / 5) : t = 30 :=
by
  sorry

end NUMINAMATH_GPT_pipe_fill_time_l1074_107437


namespace NUMINAMATH_GPT_divisor_greater_than_8_l1074_107487

-- Define the condition that remainder is 8
def remainder_is_8 (n m : ℕ) : Prop :=
  n % m = 8

-- Theorem: If n divided by m has remainder 8, then m must be greater than 8
theorem divisor_greater_than_8 (m : ℕ) (hm : m ≤ 8) : ¬ exists n, remainder_is_8 n m :=
by
  sorry

end NUMINAMATH_GPT_divisor_greater_than_8_l1074_107487


namespace NUMINAMATH_GPT_nickel_ate_2_chocolates_l1074_107406

def nickels_chocolates (r n : Nat) : Prop :=
r = n + 7

theorem nickel_ate_2_chocolates (r : Nat) (h : r = 9) (h1 : nickels_chocolates r 2) : 2 = 2 :=
by
  sorry

end NUMINAMATH_GPT_nickel_ate_2_chocolates_l1074_107406


namespace NUMINAMATH_GPT_price_of_pants_l1074_107467

theorem price_of_pants (S P H : ℝ) (h1 : 0.8 * S + P + H = 340) (h2 : S = (3 / 4) * P) (h3 : H = P + 10) : P = 91.67 :=
by sorry

end NUMINAMATH_GPT_price_of_pants_l1074_107467


namespace NUMINAMATH_GPT_food_waste_in_scientific_notation_l1074_107476

-- Given condition that 1 billion equals 10^9
def billion : ℕ := 10 ^ 9

-- Problem statement: expressing 530 billion kilograms in scientific notation
theorem food_waste_in_scientific_notation :
  (530 * billion : ℝ) = 5.3 * 10^10 := 
  sorry

end NUMINAMATH_GPT_food_waste_in_scientific_notation_l1074_107476


namespace NUMINAMATH_GPT_find_u_plus_v_l1074_107498

theorem find_u_plus_v (u v : ℚ) (h1: 5 * u - 3 * v = 26) (h2: 3 * u + 5 * v = -19) :
  u + v = -101 / 34 :=
sorry

end NUMINAMATH_GPT_find_u_plus_v_l1074_107498


namespace NUMINAMATH_GPT_minimum_uninteresting_vertices_correct_maximum_unusual_vertices_correct_l1074_107441

-- Definition for the minimum number of uninteresting vertices
def minimum_uninteresting_vertices (n : ℕ) (h : n > 3) : ℕ := 2

-- Theorem for the minimum number of uninteresting vertices
theorem minimum_uninteresting_vertices_correct (n : ℕ) (h : n > 3) :
  minimum_uninteresting_vertices n h = 2 := 
sorry

-- Definition for the maximum number of unusual vertices
def maximum_unusual_vertices (n : ℕ) (h : n > 3) : ℕ := 3

-- Theorem for the maximum number of unusual vertices
theorem maximum_unusual_vertices_correct (n : ℕ) (h : n > 3) :
  maximum_unusual_vertices n h = 3 :=
sorry

end NUMINAMATH_GPT_minimum_uninteresting_vertices_correct_maximum_unusual_vertices_correct_l1074_107441


namespace NUMINAMATH_GPT_prove_fn_value_l1074_107412

noncomputable def f (x : ℝ) : ℝ := 2^x / (2^x + 3 * x)

theorem prove_fn_value
  (m n : ℝ)
  (h1 : 2^(m + n) = 3 * m * n)
  (h2 : f m = -1 / 3) :
  f n = 4 :=
by
  sorry

end NUMINAMATH_GPT_prove_fn_value_l1074_107412


namespace NUMINAMATH_GPT_abs_gt_one_iff_square_inequality_l1074_107401

theorem abs_gt_one_iff_square_inequality (x : ℝ) : |x| > 1 ↔ x^2 - 1 > 0 := 
sorry

end NUMINAMATH_GPT_abs_gt_one_iff_square_inequality_l1074_107401


namespace NUMINAMATH_GPT_minimum_value_y_l1074_107457

theorem minimum_value_y (x y : ℕ) (h1 : x + y = 64) (h2 : 3 * x + 4 * y = 200) : y = 8 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_y_l1074_107457


namespace NUMINAMATH_GPT_volumes_comparison_l1074_107446

variable (a : ℝ) (h_a : a ≠ 3)

def volume_A := 3 * 3 * 3
def volume_B := 3 * 3 * a
def volume_C := a * a * 3
def volume_D := a * a * a

theorem volumes_comparison (h_a : a ≠ 3) :
  (volume_A + volume_D) > (volume_B + volume_C) :=
by
  have volume_A : ℝ := 27
  have volume_B := 9 * a
  have volume_C := 3 * a * a
  have volume_D := a * a * a
  sorry

end NUMINAMATH_GPT_volumes_comparison_l1074_107446


namespace NUMINAMATH_GPT_initial_cabinets_l1074_107492

theorem initial_cabinets (C : ℤ) (h1 : 26 = C + 6 * C + 5) : C = 3 := 
by 
  sorry

end NUMINAMATH_GPT_initial_cabinets_l1074_107492


namespace NUMINAMATH_GPT_redistribute_marbles_l1074_107474

theorem redistribute_marbles :
  let d := 14
  let m := 20
  let p := 19
  let v := 7
  let n := 4
  (d + m + p + v) / n = 15 :=
by
  let d := 14
  let m := 20
  let p := 19
  let v := 7
  let n := 4
  sorry

end NUMINAMATH_GPT_redistribute_marbles_l1074_107474


namespace NUMINAMATH_GPT_initial_mixture_two_l1074_107415

theorem initial_mixture_two (x : ℝ) (h : 0.25 * (x + 0.4) = 0.10 * x + 0.4) : x = 2 :=
by
  sorry

end NUMINAMATH_GPT_initial_mixture_two_l1074_107415


namespace NUMINAMATH_GPT_horner_value_at_neg4_l1074_107433

noncomputable def f (x : ℝ) : ℝ := 10 + 25 * x - 8 * x^2 + x^4 + 6 * x^5 + 2 * x^6

def horner_rewrite (x : ℝ) : ℝ := (((((2 * x + 6) * x + 1) * x + 0) * x - 8) * x + 25) * x + 10

theorem horner_value_at_neg4 : horner_rewrite (-4) = -36 :=
by sorry

end NUMINAMATH_GPT_horner_value_at_neg4_l1074_107433


namespace NUMINAMATH_GPT_abs_le_and_interval_iff_l1074_107493

variable (x : ℝ)

theorem abs_le_and_interval_iff :
  (|x - 2| ≤ 5) ↔ (-3 ≤ x ∧ x ≤ 7) :=
by
  sorry

end NUMINAMATH_GPT_abs_le_and_interval_iff_l1074_107493


namespace NUMINAMATH_GPT_f_of_f_inv_e_eq_inv_e_l1074_107414

noncomputable def f : ℝ → ℝ := λ x =>
  if x ≤ 0 then Real.exp x else Real.log x

theorem f_of_f_inv_e_eq_inv_e : f (f (1 / Real.exp 1)) = 1 / Real.exp 1 := by
  sorry

end NUMINAMATH_GPT_f_of_f_inv_e_eq_inv_e_l1074_107414


namespace NUMINAMATH_GPT_treasure_value_l1074_107438

theorem treasure_value
    (fonzie_paid : ℕ) (auntbee_paid : ℕ) (lapis_paid : ℕ)
    (lapis_share : ℚ) (lapis_received : ℕ) (total_value : ℚ)
    (h1 : fonzie_paid = 7000) 
    (h2 : auntbee_paid = 8000) 
    (h3 : lapis_paid = 9000) 
    (h4 : fonzie_paid + auntbee_paid + lapis_paid = 24000) 
    (h5 : lapis_share = lapis_paid / (fonzie_paid + auntbee_paid + lapis_paid)) 
    (h6 : lapis_received = 337500) 
    (h7 : lapis_share * total_value = lapis_received) :
  total_value = 1125000 := by
  sorry

end NUMINAMATH_GPT_treasure_value_l1074_107438


namespace NUMINAMATH_GPT_melanie_total_dimes_l1074_107480

-- Definitions based on the problem conditions
def initial_dimes : ℕ := 7
def dad_dimes : ℕ := 8
def mom_dimes : ℕ := 4

def total_dimes : ℕ := initial_dimes + dad_dimes + mom_dimes

-- Proof statement based on the correct answer
theorem melanie_total_dimes : total_dimes = 19 := by 
  -- Proof here is omitted as per instructions
  sorry

end NUMINAMATH_GPT_melanie_total_dimes_l1074_107480


namespace NUMINAMATH_GPT_beetles_eaten_per_day_l1074_107472
-- Import the Mathlib library

-- Declare the conditions as constants
def bird_eats_beetles_per_day : Nat := 12
def snake_eats_birds_per_day : Nat := 3
def jaguar_eats_snakes_per_day : Nat := 5
def number_of_jaguars : Nat := 6

-- Define the theorem and provide the expected proof
theorem beetles_eaten_per_day :
  12 * (3 * (5 * 6)) = 1080 := by
  sorry

end NUMINAMATH_GPT_beetles_eaten_per_day_l1074_107472


namespace NUMINAMATH_GPT_original_ratio_l1074_107458

theorem original_ratio (x y : ℤ) (h1 : y = 24) (h2 : (x + 6) / y = 1 / 2) : x / y = 1 / 4 := by
  sorry

end NUMINAMATH_GPT_original_ratio_l1074_107458


namespace NUMINAMATH_GPT_trapezoid_EFBA_area_l1074_107461

theorem trapezoid_EFBA_area {a : ℚ} (AE BF : ℚ) (area_ABCD : ℚ) (column_areas : List ℚ)
  (h_grid : column_areas = [a, 2 * a, 4 * a, 8 * a])
  (h_total_area : 3 * (a + 2 * a + 4 * a + 8 * a) = 48)
  (h_AE : AE = 2)
  (h_BF : BF = 4) :
  let AFGB_area := 15 * a
  let triangle_EF_area := 7 * a
  let total_trapezoid_area := AFGB_area + (triangle_EF_area / 2)
  total_trapezoid_area = 352 / 15 :=
by
  sorry

end NUMINAMATH_GPT_trapezoid_EFBA_area_l1074_107461


namespace NUMINAMATH_GPT_find_divisor_l1074_107418

theorem find_divisor (d : ℕ) (N : ℕ) (a b : ℕ)
  (h1 : a = 9) (h2 : b = 79) (h3 : N = 7) :
  (∃ d, (∀ k : ℕ, a ≤ k*d ∧ k*d ≤ b → (k*d) % d = 0) ∧
   ∀ count : ℕ, count = (b / d) - ((a - 1) / d) → count = N) →
  d = 11 :=
by
  sorry

end NUMINAMATH_GPT_find_divisor_l1074_107418


namespace NUMINAMATH_GPT_sharks_at_other_beach_is_12_l1074_107455

-- Define the conditions
def cape_may_sharks := 32
def sharks_other_beach (S : ℕ) := 2 * S + 8

-- Statement to prove
theorem sharks_at_other_beach_is_12 (S : ℕ) (h : cape_may_sharks = sharks_other_beach S) : S = 12 :=
by
  -- Sorry statement to skip the proof part
  sorry

end NUMINAMATH_GPT_sharks_at_other_beach_is_12_l1074_107455


namespace NUMINAMATH_GPT_total_legs_among_tables_l1074_107445

noncomputable def total_legs (total_tables four_legged_tables: ℕ) : ℕ :=
  let three_legged_tables := total_tables - four_legged_tables
  4 * four_legged_tables + 3 * three_legged_tables

theorem total_legs_among_tables : total_legs 36 16 = 124 := by
  sorry

end NUMINAMATH_GPT_total_legs_among_tables_l1074_107445


namespace NUMINAMATH_GPT_find_y_l1074_107454

variables (y : ℝ)

def rectangle_vertices (A B C D : (ℝ × ℝ)) : Prop :=
  (A = (-2, y)) ∧ (B = (10, y)) ∧ (C = (-2, 1)) ∧ (D = (10, 1))

def rectangle_area (length height : ℝ) : Prop :=
  length * height = 108

def positive_value (x : ℝ) : Prop :=
  0 < x

theorem find_y (A B C D : (ℝ × ℝ)) (hV : rectangle_vertices y A B C D) (hA : rectangle_area 12 (y - 1)) (hP : positive_value y) :
  y = 10 :=
sorry

end NUMINAMATH_GPT_find_y_l1074_107454


namespace NUMINAMATH_GPT_problem_1_simplified_problem_2_simplified_l1074_107470

noncomputable def problem_1 : ℝ :=
  2 * Real.sqrt 18 - Real.sqrt 50 + (1/2) * Real.sqrt 32

theorem problem_1_simplified : problem_1 = 3 * Real.sqrt 2 :=
  sorry

noncomputable def problem_2 : ℝ :=
  (Real.sqrt 5 + Real.sqrt 6) * (Real.sqrt 5 - Real.sqrt 6) - (Real.sqrt 5 - 1)^2

theorem problem_2_simplified : problem_2 = -7 + 2 * Real.sqrt 5 :=
  sorry

end NUMINAMATH_GPT_problem_1_simplified_problem_2_simplified_l1074_107470


namespace NUMINAMATH_GPT_percentage_increase_in_weight_l1074_107468

theorem percentage_increase_in_weight :
  ∀ (num_plates : ℕ) (weight_per_plate lowered_weight : ℝ),
    num_plates = 10 →
    weight_per_plate = 30 →
    lowered_weight = 360 →
    ((lowered_weight - num_plates * weight_per_plate) / (num_plates * weight_per_plate)) * 100 = 20 :=
by
  intros num_plates weight_per_plate lowered_weight h_num_plates h_weight_per_plate h_lowered_weight
  sorry

end NUMINAMATH_GPT_percentage_increase_in_weight_l1074_107468


namespace NUMINAMATH_GPT_vector_subtraction_parallel_l1074_107409

theorem vector_subtraction_parallel (t : ℝ) 
  (h_parallel : -1 / 2 = -3 / t) : 
  ( (-1 : ℝ), -3 ) - ( 2, t ) = (-3, -9) :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_vector_subtraction_parallel_l1074_107409


namespace NUMINAMATH_GPT_simplification_of_expression_l1074_107477

variable {a b : ℚ}

theorem simplification_of_expression (h1a : a ≠ 0) (h1b : b ≠ 0) (h2 : 3 * a - b / 3 ≠ 0) :
  (3 * a - b / 3)⁻¹ * ( (3 * a)⁻¹ - (b / 3)⁻¹ ) = -(a * b)⁻¹ := 
sorry

end NUMINAMATH_GPT_simplification_of_expression_l1074_107477


namespace NUMINAMATH_GPT_factor_difference_of_squares_l1074_107450

theorem factor_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_GPT_factor_difference_of_squares_l1074_107450


namespace NUMINAMATH_GPT_initial_oranges_l1074_107465

theorem initial_oranges (X : ℕ) : 
  (X - 9 + 38 = 60) → X = 31 :=
sorry

end NUMINAMATH_GPT_initial_oranges_l1074_107465


namespace NUMINAMATH_GPT_average_of_a_and_b_l1074_107475

theorem average_of_a_and_b (a b c M : ℝ)
  (h1 : (a + b) / 2 = M)
  (h2 : (b + c) / 2 = 180)
  (h3 : a - c = 200) : 
  M = 280 :=
sorry

end NUMINAMATH_GPT_average_of_a_and_b_l1074_107475


namespace NUMINAMATH_GPT_sum_of_arithmetic_sequence_l1074_107410

variable {α : Type*} [LinearOrderedField α]

def sum_arithmetic_sequence (a₁ d : α) (n : ℕ) : α :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem sum_of_arithmetic_sequence {a₁ d : α}
  (h₁ : sum_arithmetic_sequence a₁ d 10 = 12) :
  (a₁ + 4 * d) + (a₁ + 5 * d) = 12 / 5 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_arithmetic_sequence_l1074_107410


namespace NUMINAMATH_GPT_union_of_A_and_B_l1074_107411

def A : Set ℕ := {1, 2, 4}
def B : Set ℕ := {2, 4}

theorem union_of_A_and_B : A ∪ B = {1, 2, 4} := by
  sorry

end NUMINAMATH_GPT_union_of_A_and_B_l1074_107411


namespace NUMINAMATH_GPT_actual_length_correct_l1074_107405

-- Definitions based on the conditions
def blueprint_scale : ℝ := 20
def measured_length_cm : ℝ := 16

-- Statement of the proof problem
theorem actual_length_correct :
  measured_length_cm * blueprint_scale = 320 := 
sorry

end NUMINAMATH_GPT_actual_length_correct_l1074_107405


namespace NUMINAMATH_GPT_problems_solved_by_trainees_l1074_107420

theorem problems_solved_by_trainees (n m : ℕ) (h : ∀ t, t < m → (∃ p, p < n → p ≥ n / 2)) :
  ∃ p < n, (∃ t, t < m → t ≥ m / 2) :=
by
  sorry

end NUMINAMATH_GPT_problems_solved_by_trainees_l1074_107420


namespace NUMINAMATH_GPT_multiply_millions_l1074_107408

theorem multiply_millions :
  (5 * 10^6) * (8 * 10^6) = 40 * 10^12 :=
by 
  sorry

end NUMINAMATH_GPT_multiply_millions_l1074_107408


namespace NUMINAMATH_GPT_factorize_x_cubic_l1074_107499

-- Define the function and the condition
def factorize (x : ℝ) : Prop := x^3 - 9 * x = x * (x + 3) * (x - 3)

-- Prove the factorization property
theorem factorize_x_cubic (x : ℝ) : factorize x :=
by
  sorry

end NUMINAMATH_GPT_factorize_x_cubic_l1074_107499


namespace NUMINAMATH_GPT_compare_f_values_l1074_107482

noncomputable def f (x : ℝ) : ℝ := Real.sin x - x

theorem compare_f_values : 
  f (-π / 4) > f 1 ∧ f 1 > f (π / 3) := 
sorry

end NUMINAMATH_GPT_compare_f_values_l1074_107482


namespace NUMINAMATH_GPT_total_sum_of_money_l1074_107422

theorem total_sum_of_money (x : ℝ) (A B C D E : ℝ) (hA : A = x) (hB : B = 0.75 * x) 
  (hC : C = 0.60 * x) (hD : D = 0.50 * x) (hE1 : E = 0.40 * x) (hE2 : E = 84) : 
  A + B + C + D + E = 682.50 := 
by sorry

end NUMINAMATH_GPT_total_sum_of_money_l1074_107422


namespace NUMINAMATH_GPT_card_average_2023_l1074_107428

theorem card_average_2023 (n : ℕ) (h_pos : 0 < n) (h_avg : (2 * n + 1) / 3 = 2023) : n = 3034 := by
  sorry

end NUMINAMATH_GPT_card_average_2023_l1074_107428


namespace NUMINAMATH_GPT_solve_inequality_f_ge_x_no_positive_a_b_satisfy_conditions_l1074_107440

noncomputable def f (x : ℝ) : ℝ :=
  |2 * x - 1| - |2 * x - 2|

theorem solve_inequality_f_ge_x :
  {x : ℝ | f x >= x} = {x : ℝ | x <= -1 ∨ x = 1} :=
by sorry

theorem no_positive_a_b_satisfy_conditions :
  ∀ (a b : ℝ), a > 0 → b > 0 → (a + 2 * b = 1) → (2 / a + 1 / b = 4 - 1 / (a * b)) → false :=
by sorry

end NUMINAMATH_GPT_solve_inequality_f_ge_x_no_positive_a_b_satisfy_conditions_l1074_107440


namespace NUMINAMATH_GPT_fraction_blue_after_doubling_l1074_107483

theorem fraction_blue_after_doubling (x : ℕ) (h1 : ∃ x, (2 : ℚ) / 3 * x + (1 : ℚ) / 3 * x = x) :
  ((2 * (2 / 3 * x)) / ((2 / 3 * x) + (1 / 3 * x))) = (4 / 5) := by
  sorry

end NUMINAMATH_GPT_fraction_blue_after_doubling_l1074_107483


namespace NUMINAMATH_GPT_angles_MAB_NAC_l1074_107494

/-- Given equal chords AB and AC, and a tangent MAN, with arc BC's measure (excluding point A) being 200 degrees,
prove that the angles MAB and NAC are either 40 degrees or 140 degrees. -/
theorem angles_MAB_NAC (AB AC : ℝ) (tangent_MAN : Prop)
    (arc_BC_measure : ∀ A : ℝ , A = 200) : 
    ∃ θ : ℝ, (θ = 40 ∨ θ = 140) :=
by
  sorry

end NUMINAMATH_GPT_angles_MAB_NAC_l1074_107494


namespace NUMINAMATH_GPT_johns_original_earnings_l1074_107435

def JohnsEarningsBeforeRaise (currentEarnings: ℝ) (percentageIncrease: ℝ) := 
  ∀ x, currentEarnings = x + x * percentageIncrease → x = 50

theorem johns_original_earnings : 
  JohnsEarningsBeforeRaise 80 0.60 :=
by
  intro x
  intro h
  sorry

end NUMINAMATH_GPT_johns_original_earnings_l1074_107435


namespace NUMINAMATH_GPT_find_c_l1074_107421

-- Definitions
def is_root (x c : ℝ) : Prop := x^2 - 3*x + c = 0

-- Main statement
theorem find_c (c : ℝ) (h : is_root 1 c) : c = 2 :=
sorry

end NUMINAMATH_GPT_find_c_l1074_107421


namespace NUMINAMATH_GPT_determine_M_l1074_107452

noncomputable def M : Set ℤ :=
  {a | ∃ k : ℕ, k > 0 ∧ 6 = k * (5 - a)}

theorem determine_M : M = {-1, 2, 3, 4} :=
  sorry

end NUMINAMATH_GPT_determine_M_l1074_107452


namespace NUMINAMATH_GPT_rain_total_duration_l1074_107456

theorem rain_total_duration : 
  let first_day_hours := 17 - 7
  let second_day_hours := first_day_hours + 2
  let third_day_hours := 2 * second_day_hours
  first_day_hours + second_day_hours + third_day_hours = 46 :=
by
  sorry

end NUMINAMATH_GPT_rain_total_duration_l1074_107456


namespace NUMINAMATH_GPT_cosine_inequality_l1074_107463

theorem cosine_inequality
  (x y z : ℝ)
  (hx : 0 < x ∧ x < π / 2)
  (hy : 0 < y ∧ y < π / 2)
  (hz : 0 < z ∧ z < π / 2) :
  (x * Real.cos x + y * Real.cos y + z * Real.cos z) / (x + y + z) ≤
  (Real.cos x + Real.cos y + Real.cos z) / 3 := sorry

end NUMINAMATH_GPT_cosine_inequality_l1074_107463


namespace NUMINAMATH_GPT_total_numbers_is_eight_l1074_107400

theorem total_numbers_is_eight
  (avg_all : ∀ n : ℕ, (total_sum : ℝ) / n = 25)
  (avg_first_two : ∀ a₁ a₂ : ℝ, (a₁ + a₂) / 2 = 20)
  (avg_next_three : ∀ a₃ a₄ a₅ : ℝ, (a₃ + a₄ + a₅) / 3 = 26)
  (h_sixth : ∀ a₆ a₇ a₈ : ℝ, a₆ + 4 = a₇ ∧ a₆ + 6 = a₈)
  (last_num : ∀ a₈ : ℝ, a₈ = 30) :
  ∃ n : ℕ, n = 8 :=
by
  sorry

end NUMINAMATH_GPT_total_numbers_is_eight_l1074_107400


namespace NUMINAMATH_GPT_find_m_l1074_107443

-- Given definitions and conditions
def is_ellipse (x y m : ℝ) := (x^2 / m) + (y^2 / 4) = 1
def eccentricity (m : ℝ) := Real.sqrt ((m - 4) / m) = 1 / 2

-- Prove that m = 16 / 3 given the conditions
theorem find_m (m : ℝ) (cond1 : is_ellipse 1 1 m) (cond2 : eccentricity m) (cond3 : m > 4) : m = 16 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l1074_107443


namespace NUMINAMATH_GPT_flower_pots_count_l1074_107407

noncomputable def total_flower_pots (x : ℕ) : ℕ :=
  if h : ((x / 2) + (x / 4) + (x / 7) ≤ x - 1) then x else 0

theorem flower_pots_count : total_flower_pots 28 = 28 :=
by
  sorry

end NUMINAMATH_GPT_flower_pots_count_l1074_107407


namespace NUMINAMATH_GPT_rahul_savings_l1074_107431

variable (NSC PPF total_savings : ℕ)

theorem rahul_savings (h1 : NSC / 3 = PPF / 2) (h2 : PPF = 72000) : total_savings = 180000 :=
by
  sorry

end NUMINAMATH_GPT_rahul_savings_l1074_107431


namespace NUMINAMATH_GPT_new_person_weight_l1074_107432

theorem new_person_weight
  (initial_weight : ℝ)
  (average_increase : ℝ)
  (num_people : ℕ)
  (weight_replace : ℝ)
  (total_increase : ℝ)
  (W : ℝ)
  (h1 : num_people = 10)
  (h2 : average_increase = 3.5)
  (h3 : weight_replace = 65)
  (h4 : total_increase = num_people * average_increase)
  (h5 : total_increase = 35)
  (h6 : W = weight_replace + total_increase) :
  W = 100 := sorry

end NUMINAMATH_GPT_new_person_weight_l1074_107432


namespace NUMINAMATH_GPT_minimum_problems_45_l1074_107478

-- Define the types for problems and their corresponding points
structure Problem :=
(points : ℕ)

def isValidScore (s : ℕ) : Prop :=
  ∃ x y z : ℕ, 3 * x + 8 * y + 10 * z = s

def minimumProblems (s : ℕ) (min_problems : ℕ) : Prop :=
  ∃ x y z : ℕ, 3 * x + 8 * y + 10 * z = s ∧ x + y + z = min_problems

-- Main statement
theorem minimum_problems_45 : minimumProblems 45 6 :=
by 
  sorry

end NUMINAMATH_GPT_minimum_problems_45_l1074_107478


namespace NUMINAMATH_GPT_div_identity_l1074_107402

theorem div_identity :
  let a := 6 / 2
  let b := a * 3
  120 / b = 120 / 9 :=
by
  sorry

end NUMINAMATH_GPT_div_identity_l1074_107402


namespace NUMINAMATH_GPT_a_greater_than_b_for_n_ge_2_l1074_107486

theorem a_greater_than_b_for_n_ge_2 
  (n : ℕ) 
  (hn : n ≥ 2) 
  (a b : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (h1 : a^n = a + 1) 
  (h2 : b^(2 * n) = b + 3 * a) : 
  a > b := 
  sorry

end NUMINAMATH_GPT_a_greater_than_b_for_n_ge_2_l1074_107486


namespace NUMINAMATH_GPT_energy_conservation_l1074_107491

-- Define the conditions
variables (m : ℝ) (v_train v_ball : ℝ)
-- The speed of the train and the ball, converted to m/s
variables (v := 60 * 1000 / 3600) -- 60 km/h in m/s
variables (E_initial : ℝ := 0.5 * m * (v ^ 2))

-- Kinetic energy of the ball when thrown in the same direction
variables (E_same_direction : ℝ := 0.5 * m * (2 * v)^2)

-- Kinetic energy of the ball when thrown in the opposite direction
variables (E_opposite_direction : ℝ := 0.5 * m * (0)^2)

-- Prove energy conservation
theorem energy_conservation : 
  (E_same_direction - E_initial) + (E_opposite_direction - E_initial) = 0 :=
sorry

end NUMINAMATH_GPT_energy_conservation_l1074_107491


namespace NUMINAMATH_GPT_coordinates_of_B_l1074_107462

theorem coordinates_of_B (a : ℝ) (h : a - 2 = 0) : (a + 2, a - 1) = (4, 1) :=
by
  sorry

end NUMINAMATH_GPT_coordinates_of_B_l1074_107462


namespace NUMINAMATH_GPT_problem_part1_problem_part2_problem_part3_l1074_107481

noncomputable def f (x : ℝ) : ℝ := 2^x + 2^(-x)

theorem problem_part1 : f 1 = 5 / 2 ∧ f 2 = 17 / 4 := 
by
  sorry

theorem problem_part2 : ∀ x : ℝ, f (-x) = f x :=
by
  sorry

theorem problem_part3 : ∀ x1 x2 : ℝ, x1 < x2 → x1 < 0 → x2 < 0 → f x1 > f x2 :=
by
  sorry

end NUMINAMATH_GPT_problem_part1_problem_part2_problem_part3_l1074_107481


namespace NUMINAMATH_GPT_all_possible_values_of_k_l1074_107429

def is_partition_possible (k : ℕ) : Prop :=
  ∃ (A B : Finset ℕ), (A ∪ B = Finset.range (k + 1)) ∧ (A ∩ B = ∅) ∧ (A.sum id = 2 * B.sum id)

theorem all_possible_values_of_k (k : ℕ) : 
  is_partition_possible k → ∃ m : ℕ, k = 3 * m ∨ k = 3 * m - 1 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_all_possible_values_of_k_l1074_107429


namespace NUMINAMATH_GPT_baseball_glove_price_l1074_107427

noncomputable def original_price_glove : ℝ := 42.50

theorem baseball_glove_price (cards bat glove_discounted cleats total : ℝ) 
  (h1 : cards = 25) 
  (h2 : bat = 10) 
  (h3 : cleats = 2 * 10)
  (h4 : total = 79) 
  (h5 : glove_discounted = total - (cards + bat + cleats)) 
  (h6 : glove_discounted = 0.80 * original_price_glove) : 
  original_price_glove = 42.50 := by 
  sorry

end NUMINAMATH_GPT_baseball_glove_price_l1074_107427


namespace NUMINAMATH_GPT_power_of_product_l1074_107490

variable (x y : ℝ)

theorem power_of_product (x y : ℝ) : (x * y^2)^2 = x^2 * y^4 :=
  sorry

end NUMINAMATH_GPT_power_of_product_l1074_107490


namespace NUMINAMATH_GPT_ways_to_change_12_dollars_into_nickels_and_quarters_l1074_107453

theorem ways_to_change_12_dollars_into_nickels_and_quarters :
  ∃ n q : ℕ, 5 * n + 25 * q = 1200 ∧ n > 0 ∧ q > 0 ∧ ∀ q', (q' ≥ 1 ∧ q' ≤ 47) ↔ (n = 240 - 5 * q') :=
by
  sorry

end NUMINAMATH_GPT_ways_to_change_12_dollars_into_nickels_and_quarters_l1074_107453


namespace NUMINAMATH_GPT_prod_of_three_consec_ints_l1074_107496

theorem prod_of_three_consec_ints (a : ℤ) (h : a + (a + 1) + (a + 2) = 27) :
  a * (a + 1) * (a + 2) = 720 :=
by
  sorry

end NUMINAMATH_GPT_prod_of_three_consec_ints_l1074_107496


namespace NUMINAMATH_GPT_value_range_for_positive_roots_l1074_107489

theorem value_range_for_positive_roots (a : ℝ) :
  (∀ x : ℝ, x > 0 → a * |x| + |x + a| = 0) ↔ (-1 < a ∧ a < 0) :=
by
  sorry

end NUMINAMATH_GPT_value_range_for_positive_roots_l1074_107489


namespace NUMINAMATH_GPT_solve_for_x_l1074_107473

theorem solve_for_x : 
  (∀ x : ℝ, x ≠ -2 → (x^2 - x - 2) / (x + 2) = x - 1 ↔ x = 0) := 
by 
  sorry

end NUMINAMATH_GPT_solve_for_x_l1074_107473


namespace NUMINAMATH_GPT_fred_weekend_earnings_l1074_107442

noncomputable def fred_initial_dollars : ℕ := 19
noncomputable def fred_final_dollars : ℕ := 40

theorem fred_weekend_earnings :
  fred_final_dollars - fred_initial_dollars = 21 :=
by
  sorry

end NUMINAMATH_GPT_fred_weekend_earnings_l1074_107442


namespace NUMINAMATH_GPT_min_additional_games_l1074_107466

-- Definitions of parameters
def initial_total_games : ℕ := 5
def initial_falcon_wins : ℕ := 2
def win_percentage_threshold : ℚ := 91 / 100

-- Theorem stating the minimum value for N
theorem min_additional_games (N : ℕ) : (initial_falcon_wins + N : ℚ) / (initial_total_games + N : ℚ) ≥ win_percentage_threshold → N ≥ 29 :=
by
  sorry

end NUMINAMATH_GPT_min_additional_games_l1074_107466


namespace NUMINAMATH_GPT_gcd_102_238_l1074_107471

theorem gcd_102_238 : Nat.gcd 102 238 = 34 := by
  sorry

end NUMINAMATH_GPT_gcd_102_238_l1074_107471


namespace NUMINAMATH_GPT_silver_excess_in_third_chest_l1074_107460

theorem silver_excess_in_third_chest :
  ∀ (x1 y1 x2 y2 x3 y3 : ℕ),
    x1 + x2 + x3 = 40 →
    y1 + y2 + y3 = 40 →
    x1 = y1 + 7 →
    y2 = x2 - 15 →
    y3 = x3 + 22 :=
by
  intros x1 y1 x2 y2 x3 y3 h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_silver_excess_in_third_chest_l1074_107460


namespace NUMINAMATH_GPT_equal_areas_of_shapes_l1074_107464

noncomputable def hexagon_area (s : ℝ) : ℝ :=
  (3 * Real.sqrt 3 / 2) * s^2

noncomputable def semicircle_area (r : ℝ) : ℝ :=
  (Real.pi * r^2) / 2

noncomputable def sector_area (theta : ℝ) (r : ℝ) : ℝ :=
  (theta / (2 * Real.pi)) * Real.pi * r^2

noncomputable def shape1_area (s : ℝ) : ℝ :=
  hexagon_area s + 3 * semicircle_area (s / 4) - 6 * sector_area (Real.pi / 3) (s / 4)

noncomputable def shape2_area (s : ℝ) : ℝ :=
  hexagon_area s + 6 * sector_area (2 * Real.pi / 3) (s / 4) - 3 * semicircle_area (s / 4)

theorem equal_areas_of_shapes (s : ℝ) : shape1_area s = shape2_area s :=
by {
  sorry
}

end NUMINAMATH_GPT_equal_areas_of_shapes_l1074_107464


namespace NUMINAMATH_GPT_mrs_lee_earnings_percentage_l1074_107404

theorem mrs_lee_earnings_percentage 
  (M F : ℝ)
  (H1 : 1.20 * M = 0.5454545454545454 * (1.20 * M + F)) :
  M = 0.5 * (M + F) :=
by sorry

end NUMINAMATH_GPT_mrs_lee_earnings_percentage_l1074_107404


namespace NUMINAMATH_GPT_distance_p_runs_l1074_107439

-- Given conditions
def runs_faster (speed_q : ℝ) : ℝ := 1.20 * speed_q
def head_start : ℝ := 50

-- Proof statement
theorem distance_p_runs (speed_q distance_q : ℝ) (h1 : runs_faster speed_q = 1.20 * speed_q)
                         (h2 : head_start = 50)
                         (h3 : (distance_q / speed_q) = ((distance_q + head_start) / (runs_faster speed_q))) :
                         (distance_q + head_start = 300) :=
by
  sorry

end NUMINAMATH_GPT_distance_p_runs_l1074_107439


namespace NUMINAMATH_GPT_investment_worth_l1074_107485

noncomputable def initial_investment (total_earning : ℤ) : ℤ := total_earning / 2

noncomputable def current_worth (initial_investment total_earning : ℤ) : ℤ :=
  initial_investment + total_earning

theorem investment_worth (monthly_earning : ℤ) (months : ℤ) (earnings : ℤ)
  (h1 : monthly_earning * months = earnings)
  (h2 : earnings = 2 * initial_investment earnings) :
  current_worth (initial_investment earnings) earnings = 90 := 
by
  -- We proceed to show the current worth is $90
  -- Proof will be constructed here
  sorry
  
end NUMINAMATH_GPT_investment_worth_l1074_107485


namespace NUMINAMATH_GPT_valid_P_values_l1074_107479

/-- 
Construct a 3x3 grid of distinct natural numbers where the product of the numbers 
in each row and each column is equal. Verify the valid values of P among the given set.
-/
theorem valid_P_values (P : ℕ) :
  (∃ (a b c d e f g h i : ℕ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧ 
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧ 
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧ 
    d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧ 
    e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧ 
    f ≠ g ∧ f ≠ h ∧ f ≠ i ∧ 
    g ≠ h ∧ g ≠ i ∧ 
    h ≠ i ∧ 
    a * b * c = P ∧ 
    d * e * f = P ∧ 
    g * h * i = P ∧ 
    a * d * g = P ∧ 
    b * e * h = P ∧ 
    c * f * i = P ∧ 
    P = (Nat.sqrt ((1 * 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9)) )) ↔ P = 1998 ∨ P = 2000 :=
sorry

end NUMINAMATH_GPT_valid_P_values_l1074_107479


namespace NUMINAMATH_GPT_sin_90_eq_one_l1074_107488

-- Definition of the rotation by 90 degrees counterclockwise
def rotate90 (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, p.1)

-- Definition of the sine function for a 90 degree angle
def sin90 : ℝ :=
  let initial_point := (1, 0)
  let rotated_point := rotate90 initial_point
  rotated_point.2

-- Theorem to be proven: sin90 should be equal to 1
theorem sin_90_eq_one : sin90 = 1 :=
by
  sorry

end NUMINAMATH_GPT_sin_90_eq_one_l1074_107488


namespace NUMINAMATH_GPT_contribution_per_person_l1074_107449

-- Define constants for the given conditions
def total_price : ℕ := 67
def coupon : ℕ := 4
def number_of_people : ℕ := 3

-- The theorem to prove
theorem contribution_per_person : (total_price - coupon) / number_of_people = 21 :=
by 
  -- The proof is omitted for this exercise
  sorry

end NUMINAMATH_GPT_contribution_per_person_l1074_107449
