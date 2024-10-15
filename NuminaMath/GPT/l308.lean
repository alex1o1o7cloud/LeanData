import Mathlib

namespace NUMINAMATH_GPT_f_neg_two_l308_30835

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^5 + b * x - c / x + 2

theorem f_neg_two (a b c : ℝ) (h : f a b c 2 = 4) : f a b c (-2) = 0 :=
sorry

end NUMINAMATH_GPT_f_neg_two_l308_30835


namespace NUMINAMATH_GPT_comic_cost_is_4_l308_30894

-- Define initial amount of money Raul had.
def initial_money : ℕ := 87

-- Define number of comics bought by Raul.
def num_comics : ℕ := 8

-- Define the amount of money left after buying comics.
def money_left : ℕ := 55

-- Define the hypothesis condition about the money spent.
def total_spent : ℕ := initial_money - money_left

-- Define the main assertion that each comic cost $4.
def cost_per_comic (total_spent : ℕ) (num_comics : ℕ) : Prop :=
  total_spent / num_comics = 4

-- Main theorem statement
theorem comic_cost_is_4 : cost_per_comic total_spent num_comics :=
by
  -- Here we're skipping the proof for this exercise.
  sorry

end NUMINAMATH_GPT_comic_cost_is_4_l308_30894


namespace NUMINAMATH_GPT_max_area_isosceles_triangle_l308_30876

theorem max_area_isosceles_triangle (b : ℝ) (h : ℝ) (area : ℝ) 
  (h_cond : h^2 = 1 - b^2 / 4)
  (area_def : area = 1 / 2 * b * h) : 
  area ≤ 2 * Real.sqrt 2 / 3 := 
sorry

end NUMINAMATH_GPT_max_area_isosceles_triangle_l308_30876


namespace NUMINAMATH_GPT_carlos_finishes_first_l308_30810

theorem carlos_finishes_first
  (a : ℝ) -- Andy's lawn area
  (r : ℝ) -- Andy's mowing rate
  (hBeth_lawn : ∀ (b : ℝ), b = a / 3) -- Beth's lawn area
  (hCarlos_lawn : ∀ (c : ℝ), c = a / 4) -- Carlos' lawn area
  (hCarlos_Beth_rate : ∀ (rc rb : ℝ), rc = r / 2 ∧ rb = r / 2) -- Carlos' and Beth's mowing rate
  : (∃ (ta tb tc : ℝ), ta = a / r ∧ tb = (2 * a) / (3 * r) ∧ tc = a / (2 * r) ∧ tc < tb ∧ tc < ta) :=
-- Prove that the mowing times are such that Carlos finishes first
sorry

end NUMINAMATH_GPT_carlos_finishes_first_l308_30810


namespace NUMINAMATH_GPT_prime_k_for_equiangular_polygons_l308_30875

-- Definitions for conditions in Lean 4
def is_equiangular_polygon (n : ℕ) (angle : ℕ) : Prop :=
  angle = 180 - 360 / n

def is_prime (k : ℕ) : Prop :=
  Nat.Prime k

def valid_angle (x : ℕ) (k : ℕ) : Prop :=
  x < 180 / k

-- The main statement
theorem prime_k_for_equiangular_polygons (n1 n2 x k : ℕ) :
  is_equiangular_polygon n1 x →
  is_equiangular_polygon n2 (k * x) →
  1 < k →
  is_prime k →
  k = 3 :=
by sorry -- proof is not required

end NUMINAMATH_GPT_prime_k_for_equiangular_polygons_l308_30875


namespace NUMINAMATH_GPT_equation1_solution_equation2_solution_l308_30865

theorem equation1_solution (x : ℝ) (h : 2 * (x - 1) = 2 - 5 * (x + 2)) : x = -6 / 7 :=
sorry

theorem equation2_solution (x : ℝ) (h : (5 * x + 1) / 2 - (6 * x + 2) / 4 = 1) : x = 1 :=
sorry

end NUMINAMATH_GPT_equation1_solution_equation2_solution_l308_30865


namespace NUMINAMATH_GPT_multiply_and_simplify_l308_30813

variable (a b : ℝ)

theorem multiply_and_simplify :
  (3 * a + 2 * b) * (a - 2 * b) = 3 * a^2 - 4 * a * b - 4 * b^2 :=
by
  sorry

end NUMINAMATH_GPT_multiply_and_simplify_l308_30813


namespace NUMINAMATH_GPT_consumption_increase_percentage_l308_30804

theorem consumption_increase_percentage (T C : ℝ) (T_pos : 0 < T) (C_pos : 0 < C) :
  (0.7 * (1 + x / 100) * T * C = 0.84 * T * C) → x = 20 :=
by sorry

end NUMINAMATH_GPT_consumption_increase_percentage_l308_30804


namespace NUMINAMATH_GPT_new_water_intake_recommendation_l308_30883

noncomputable def current_consumption : ℝ := 25
noncomputable def increase_percentage : ℝ := 0.75
noncomputable def increased_amount : ℝ := increase_percentage * current_consumption
noncomputable def new_recommended_consumption : ℝ := current_consumption + increased_amount

theorem new_water_intake_recommendation :
  new_recommended_consumption = 43.75 := 
by 
  sorry

end NUMINAMATH_GPT_new_water_intake_recommendation_l308_30883


namespace NUMINAMATH_GPT_atomic_weight_O_l308_30820

-- We define the atomic weights of sodium and chlorine
def atomic_weight_Na : ℝ := 22.99
def atomic_weight_Cl : ℝ := 35.45

-- We define the molecular weight of the compound
def molecular_weight_compound : ℝ := 74.0

-- We want to prove that the atomic weight of oxygen (O) is 15.56 given the above conditions
theorem atomic_weight_O : 
  (molecular_weight_compound = atomic_weight_Na + atomic_weight_Cl + w -> w = 15.56) :=
by
  sorry

end NUMINAMATH_GPT_atomic_weight_O_l308_30820


namespace NUMINAMATH_GPT_perpendicular_lines_a_value_l308_30846

theorem perpendicular_lines_a_value (a : ℝ) :
  (a * (a + 2) = -1) → a = -1 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_perpendicular_lines_a_value_l308_30846


namespace NUMINAMATH_GPT_king_arthur_actual_weight_l308_30859

theorem king_arthur_actual_weight (K H E : ℤ) 
  (h1 : K + E = 19) 
  (h2 : H + E = 101) 
  (h3 : K + H + E = 114) : K = 13 := 
by 
  -- Introduction for proof to be skipped
  sorry

end NUMINAMATH_GPT_king_arthur_actual_weight_l308_30859


namespace NUMINAMATH_GPT_correct_statements_l308_30880

-- Define the propositions p and q
variables (p q : Prop)

-- Define the given statements as logical conditions
def statement1 := (p ∧ q) → (p ∨ q)
def statement2 := ¬(p ∧ q) → (p ∨ q)
def statement3 := (p ∨ q) ↔ ¬¬p
def statement4 := (¬p) → ¬(p ∧ q)

-- Define the proof problem
theorem correct_statements :
  ((statement1 p q) ∧ (¬statement2 p q) ∧ (statement3 p q) ∧ (¬statement4 p q)) :=
by {
  -- Here you would prove that
  -- statement1 is correct,
  -- statement2 is incorrect,
  -- statement3 is correct,
  -- statement4 is incorrect
  sorry
}

end NUMINAMATH_GPT_correct_statements_l308_30880


namespace NUMINAMATH_GPT_right_triangle_condition_l308_30852

def fib (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | 1 => 4
  | 2 => 4
  | n + 3 => fib (n + 2) + fib (n + 1)

theorem right_triangle_condition (n : ℕ) : 
  ∃ a b c, a = fib n * fib (n + 4) ∧ 
           b = fib (n + 1) * fib (n + 3) ∧ 
           c = 2 * fib (n + 2) ∧
           a * a + b * b = c * c :=
by sorry

end NUMINAMATH_GPT_right_triangle_condition_l308_30852


namespace NUMINAMATH_GPT_value_of_f_inv_sum_l308_30808

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def f_inv (y : ℝ) : ℝ := sorry

axiom f_inv_is_inverse : ∀ x : ℝ, f (f_inv x) = x ∧ f_inv (f x) = x
axiom f_condition : ∀ x : ℝ, f x + f (-x) = 2

theorem value_of_f_inv_sum (x : ℝ) : f_inv (2008 - x) + f_inv (x - 2006) = 0 :=
sorry

end NUMINAMATH_GPT_value_of_f_inv_sum_l308_30808


namespace NUMINAMATH_GPT_card_statements_true_l308_30897

def statement1 (statements : Fin 5 → Prop) : Prop :=
  ∃! i, i < 5 ∧ statements i

def statement2 (statements : Fin 5 → Prop) : Prop :=
  (∃ i j, i < 5 ∧ j < 5 ∧ i ≠ j ∧ statements i ∧ statements j) ∧ ¬(∃ h k l, h < 5 ∧ k < 5 ∧ l < 5 ∧ h ≠ k ∧ h ≠ l ∧ k ≠ l ∧ statements h ∧ statements k ∧ statements l)

def statement3 (statements : Fin 5 → Prop) : Prop :=
  (∃ i j k, i < 5 ∧ j < 5 ∧ k < 5 ∧ i ≠ j ∧ i ≠ k ∧ j ≠ k ∧ statements i ∧ statements j ∧ statements k) ∧ ¬(∃ a b c d, a < 5 ∧ b < 5 ∧ c < 5 ∧ d < 5 ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ statements a ∧ statements b ∧ statements c ∧ statements d)

def statement4 (statements : Fin 5 → Prop) : Prop :=
  (∃ i j k l, i < 5 ∧ j < 5 ∧ k < 5 ∧ l < 5 ∧ i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧ statements i ∧ statements j ∧ statements k ∧ statements l) ∧ ¬(∃ a b c d e, a < 5 ∧ b < 5 ∧ c < 5 ∧ d < 5 ∧ e < 5 ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧ statements a ∧ statements b ∧ statements c ∧ statements d ∧ statements e)

def statement5 (statements : Fin 5 → Prop) : Prop :=
  ∀ i, i < 5 → statements i

theorem card_statements_true : ∃ (statements : Fin 5 → Prop), 
  statement1 statements ∨ statement2 statements ∨ statement3 statements ∨ statement4 statements ∨ statement5 statements 
  ∧ statement3 statements := 
sorry

end NUMINAMATH_GPT_card_statements_true_l308_30897


namespace NUMINAMATH_GPT_proof_l308_30854

open Set

-- Universal set U
def U : Set ℕ := {x | x ∈ Finset.range 7}

-- Set A
def A : Set ℕ := {1, 3, 5}

-- Set B
def B : Set ℕ := {4, 5, 6}

-- Complement of A in U
def CU (s : Set ℕ) : Set ℕ := U \ s

-- Proof statement
theorem proof : (CU A) ∩ B = {4, 6} :=
by
  sorry

end NUMINAMATH_GPT_proof_l308_30854


namespace NUMINAMATH_GPT_final_range_a_l308_30857

open Real

noncomputable def f (a x : ℝ) : ℝ := log x + x^2 - a * x

lemma increasing_function_range_a (a : ℝ) (h : ∀ x : ℝ, x > 0 → deriv (f a) x ≥ 0) :
  a ≤ 2 * sqrt 2 :=
sorry

lemma condition_range_a (a : ℝ) (h1 : ∀ x : ℝ, x > 0 → deriv (f a) x ≥ 0)
  (h2 : ∀ x : ℝ, 0 < x ∧ x ≤ 1 → f a x ≤ 1/2 * (3 * x^2 + 1 / x^2 - 6 * x)) :
  2 ≤ a :=
sorry

theorem final_range_a (a : ℝ)
  (h1 : ∀ x : ℝ, x > 0 → deriv (f a) x ≥ 0)
  (h2 : ∀ x : ℝ, 0 < x ∧ x ≤ 1 → f a x ≤ 1/2 * (3 * x^2 + 1 / x^2 - 6 * x)) :
  2 ≤ a ∧ a ≤ 2 * sqrt 2 :=
sorry

end NUMINAMATH_GPT_final_range_a_l308_30857


namespace NUMINAMATH_GPT_max_value_of_f_l308_30815

noncomputable def f (x : ℝ) : ℝ := min (2^x) (min (x + 2) (10 - x))

theorem max_value_of_f : ∃ M, (∀ x ≥ 0, f x ≤ M) ∧ (∃ x ≥ 0, f x = M) ∧ M = 6 :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_f_l308_30815


namespace NUMINAMATH_GPT_kay_exercise_time_l308_30882

variable (A W : ℕ)
variable (exercise_total : A + W = 250) 
variable (ratio_condition : A * 2 = 3 * W)

theorem kay_exercise_time :
  A = 150 ∧ W = 100 :=
by
  sorry

end NUMINAMATH_GPT_kay_exercise_time_l308_30882


namespace NUMINAMATH_GPT_first_plane_passengers_l308_30807

-- Definitions and conditions
def speed_plane_empty : ℕ := 600
def slowdown_per_passenger : ℕ := 2
def second_plane_passengers : ℕ := 60
def third_plane_passengers : ℕ := 40
def average_speed : ℕ := 500

-- Definition of the speed of a plane given number of passengers
def speed (passengers : ℕ) : ℕ := speed_plane_empty - slowdown_per_passenger * passengers

-- The problem statement rewritten in Lean 4
theorem first_plane_passengers (P : ℕ) (h_avg : (speed P + speed second_plane_passengers + speed third_plane_passengers) / 3 = average_speed) : P = 50 :=
sorry

end NUMINAMATH_GPT_first_plane_passengers_l308_30807


namespace NUMINAMATH_GPT_difference_of_sides_l308_30818

-- Definitions based on conditions
def smaller_square_side (s : ℝ) := s
def larger_square_side (S s : ℝ) (h : (S^2 : ℝ) = 4 * s^2) := S

-- Theorem statement based on the proof problem
theorem difference_of_sides (s S : ℝ) (h : (S^2 : ℝ) = 4 * s^2) : S - s = s := 
by
  sorry

end NUMINAMATH_GPT_difference_of_sides_l308_30818


namespace NUMINAMATH_GPT_sandy_tokens_ratio_l308_30805

theorem sandy_tokens_ratio :
  ∀ (total_tokens : ℕ) (num_siblings : ℕ) (difference : ℕ),
  total_tokens = 1000000 →
  num_siblings = 4 →
  difference = 375000 →
  ∃ (sandy_tokens : ℕ),
  sandy_tokens = (total_tokens - (num_siblings * ((total_tokens - difference) / (num_siblings + 1)))) ∧
  sandy_tokens / total_tokens = 1 / 2 :=
by 
  intros total_tokens num_siblings difference h1 h2 h3
  sorry

end NUMINAMATH_GPT_sandy_tokens_ratio_l308_30805


namespace NUMINAMATH_GPT_moms_took_chocolates_l308_30868

theorem moms_took_chocolates (N : ℕ) (A : ℕ) (M : ℕ) : 
  N = 10 → 
  A = 3 * N →
  A - M = N + 15 →
  M = 5 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_moms_took_chocolates_l308_30868


namespace NUMINAMATH_GPT_cos_90_eq_zero_l308_30895

theorem cos_90_eq_zero : Real.cos (90 * Real.pi / 180) = 0 := by 
  sorry

end NUMINAMATH_GPT_cos_90_eq_zero_l308_30895


namespace NUMINAMATH_GPT_find_D_double_prime_l308_30811

def reflectY (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

def translateUp1 (p : ℝ × ℝ) : ℝ × ℝ := (p.1, p.2 + 1)

def reflectYeqX (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

def translateDown1 (p : ℝ × ℝ) : ℝ × ℝ := (p.1, p.2 - 1)

def D'' (D : ℝ × ℝ) : ℝ × ℝ :=
  translateDown1 (reflectYeqX (translateUp1 (reflectY D)))

theorem find_D_double_prime :
  let D := (5, 0)
  D'' D = (-1, 4) :=
by
  sorry

end NUMINAMATH_GPT_find_D_double_prime_l308_30811


namespace NUMINAMATH_GPT_smallest_even_in_sequence_sum_400_l308_30830

theorem smallest_even_in_sequence_sum_400 :
  ∃ (n : ℤ), (n - 6) + (n - 4) + (n - 2) + n + (n + 2) + (n + 4) + (n + 6) = 400 ∧ (n - 6) % 2 = 0 ∧ n - 6 = 52 :=
sorry

end NUMINAMATH_GPT_smallest_even_in_sequence_sum_400_l308_30830


namespace NUMINAMATH_GPT_maria_initial_cookies_l308_30836

theorem maria_initial_cookies (X : ℕ) 
  (h1: X - 5 = 2 * (5 + 2)) 
  (h2: X ≥ 5)
  : X = 19 := 
by
  sorry

end NUMINAMATH_GPT_maria_initial_cookies_l308_30836


namespace NUMINAMATH_GPT_part1_solution_set_part2_min_value_l308_30816

-- Part 1
noncomputable def f (x : ℝ) : ℝ := 2 * |x + 1| + |3 * x|

theorem part1_solution_set :
  {x : ℝ | f x ≥ 3 * |x| + 1} = {x : ℝ | x ≥ -1/2} ∪ {x : ℝ | x ≤ -3/2} :=
by
  sorry

-- Part 2
noncomputable def f_min (x a b : ℝ) : ℝ := 2 * |x + a| + |3 * x - b|

theorem part2_min_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : ∃ x, f_min x a b = 2) :
  3 * a + b = 3 :=
by
  sorry

end NUMINAMATH_GPT_part1_solution_set_part2_min_value_l308_30816


namespace NUMINAMATH_GPT_shaded_area_square_semicircles_l308_30869

theorem shaded_area_square_semicircles :
  let side_length := 2
  let radius_circle := side_length * Real.sqrt 2 / 2
  let area_circle := Real.pi * radius_circle^2
  let area_square := side_length^2
  let area_semicircle := Real.pi * (side_length / 2)^2 / 2
  let total_area_semicircles := 4 * area_semicircle
  let shaded_area := total_area_semicircles - area_circle
  shaded_area = 4 :=
by
  sorry

end NUMINAMATH_GPT_shaded_area_square_semicircles_l308_30869


namespace NUMINAMATH_GPT_total_rowing_and_hiking_l308_30848

def total_campers : ℕ := 80
def morning_rowing : ℕ := 41
def morning_hiking : ℕ := 4
def morning_swimming : ℕ := 15
def afternoon_rowing : ℕ := 26
def afternoon_hiking : ℕ := 8
def afternoon_swimming : ℕ := total_campers - afternoon_rowing - afternoon_hiking - (total_campers - morning_rowing - morning_hiking - morning_swimming)

theorem total_rowing_and_hiking : 
  (morning_rowing + afternoon_rowing) + (morning_hiking + afternoon_hiking) = 79 :=
by
  sorry

end NUMINAMATH_GPT_total_rowing_and_hiking_l308_30848


namespace NUMINAMATH_GPT_zane_total_payment_l308_30806

open Real

noncomputable def shirt1_price := 50.0
noncomputable def shirt2_price := 50.0
noncomputable def discount1 := 0.4 * shirt1_price
noncomputable def discount2 := 0.3 * shirt2_price
noncomputable def price1_after_discount := shirt1_price - discount1
noncomputable def price2_after_discount := shirt2_price - discount2
noncomputable def total_before_tax := price1_after_discount + price2_after_discount
noncomputable def sales_tax := 0.08 * total_before_tax
noncomputable def total_cost := total_before_tax + sales_tax

-- We want to prove:
theorem zane_total_payment : total_cost = 70.20 := by sorry

end NUMINAMATH_GPT_zane_total_payment_l308_30806


namespace NUMINAMATH_GPT_find_multiple_l308_30824

theorem find_multiple (x k : ℕ) (hx : x > 0) (h_eq : x + 17 = k * (1/x)) (h_x : x = 3) : k = 60 :=
by
  sorry

end NUMINAMATH_GPT_find_multiple_l308_30824


namespace NUMINAMATH_GPT_problem_293_l308_30800

theorem problem_293 (s : ℝ) (R' : ℝ) (rectangle1 : ℝ) (circle1 : ℝ) 
  (condition1 : s = 4) 
  (condition2 : rectangle1 = 2 * 4) 
  (condition3 : circle1 = Real.pi * 1^2) 
  (condition4 : R' = s^2 - (rectangle1 + circle1)) 
  (fraction_form : ∃ m n : ℕ, gcd m n = 1 ∧ R' = m / n) : 
  (∃ m n : ℕ, gcd m n = 1 ∧ R' = m / n ∧ m + n = 293) := 
sorry

end NUMINAMATH_GPT_problem_293_l308_30800


namespace NUMINAMATH_GPT_find_circle_center_l308_30823

def circle_center_condition (x y : ℝ) : Prop :=
  (3 * x - 4 * y = 24 ∨ 3 * x - 4 * y = -12) ∧ 3 * x + 2 * y = 0

theorem find_circle_center :
  ∃ (x y : ℝ), circle_center_condition x y ∧ (x, y) = (2/3, -1) :=
by
  sorry

end NUMINAMATH_GPT_find_circle_center_l308_30823


namespace NUMINAMATH_GPT_value_of_c_l308_30879

theorem value_of_c :
  ∃ (a b c : ℕ), 
  30 = 2 * (10 + a) ∧ 
  b = 2 * (a + 30) ∧ 
  c = 2 * (b + 30) ∧ 
  c = 200 := 
sorry

end NUMINAMATH_GPT_value_of_c_l308_30879


namespace NUMINAMATH_GPT_bike_license_combinations_l308_30833

theorem bike_license_combinations : 
  let letters := 3
  let digits := 10
  let total_combinations := letters * digits^4
  total_combinations = 30000 := by
  let letters := 3
  let digits := 10
  let total_combinations := letters * digits^4
  sorry

end NUMINAMATH_GPT_bike_license_combinations_l308_30833


namespace NUMINAMATH_GPT_Diane_net_loss_l308_30872

variable (x y a b: ℝ)

axiom h1 : x * a = 65
axiom h2 : y * b = 150

theorem Diane_net_loss : (y * b) - (x * a) = 50 := by
  sorry

end NUMINAMATH_GPT_Diane_net_loss_l308_30872


namespace NUMINAMATH_GPT_shaded_area_is_correct_l308_30821

noncomputable def square_shaded_area (side : ℝ) (beta : ℝ) (cos_beta : ℝ) : ℝ :=
  if (0 < beta) ∧ (beta < 90) ∧ (cos_beta = 3 / 5) ∧ (side = 2) then 3 / 10 
  else 0

theorem shaded_area_is_correct :
  square_shaded_area 2 beta (3 / 5) = 3 / 10 :=
by
  sorry

end NUMINAMATH_GPT_shaded_area_is_correct_l308_30821


namespace NUMINAMATH_GPT_wendy_packages_chocolates_l308_30858

variable (packages_per_5min : Nat := 2)
variable (dozen_size : Nat := 12)
variable (minutes_in_hour : Nat := 60)
variable (hours : Nat := 4)

theorem wendy_packages_chocolates (h1 : packages_per_5min = 2) 
                                 (h2 : dozen_size = 12) 
                                 (h3 : minutes_in_hour = 60) 
                                 (h4 : hours = 4) : 
    let chocolates_per_5min := packages_per_5min * dozen_size
    let intervals_per_hour := minutes_in_hour / 5
    let chocolates_per_hour := chocolates_per_5min * intervals_per_hour
    let chocolates_in_4hours := chocolates_per_hour * hours
    chocolates_in_4hours = 1152 := 
by
  let chocolates_per_5min := packages_per_5min * dozen_size
  let intervals_per_hour := minutes_in_hour / 5
  let chocolates_per_hour := chocolates_per_5min * intervals_per_hour
  let chocolates_in_4hours := chocolates_per_hour * hours
  sorry

end NUMINAMATH_GPT_wendy_packages_chocolates_l308_30858


namespace NUMINAMATH_GPT_min_radius_circle_line_intersection_l308_30838

theorem min_radius_circle_line_intersection (θ : ℝ) (hθ : 0 ≤ θ ∧ θ < 2 * Real.pi) (r : ℝ) (hr : r > 0)
    (intersect : ∃ (x y : ℝ), (x - Real.cos θ)^2 + (y - Real.sin θ)^2 = r^2 ∧ 2 * x - y - 10 = 0) :
    r ≥ 2 * Real.sqrt 5 - 1 :=
  sorry

end NUMINAMATH_GPT_min_radius_circle_line_intersection_l308_30838


namespace NUMINAMATH_GPT_surface_area_of_given_cylinder_l308_30870

noncomputable def surface_area_of_cylinder (length width : ℝ) : ℝ :=
  let r := (length / (2 * Real.pi))
  let h := width
  2 * Real.pi * r^2 + 2 * Real.pi * r * h

theorem surface_area_of_given_cylinder : 
  surface_area_of_cylinder (4 * Real.pi) 2 = 16 * Real.pi :=
by
  -- Proof will be filled here
  sorry

end NUMINAMATH_GPT_surface_area_of_given_cylinder_l308_30870


namespace NUMINAMATH_GPT_packed_lunch_needs_l308_30825

-- Definitions based on conditions
def students_A : ℕ := 10
def students_B : ℕ := 15
def students_C : ℕ := 20

def total_students : ℕ := students_A + students_B + students_C

def slices_per_sandwich : ℕ := 4
def sandwiches_per_student : ℕ := 2
def bread_slices_per_student : ℕ := sandwiches_per_student * slices_per_sandwich
def total_bread_slices : ℕ := total_students * bread_slices_per_student

def bags_of_chips_per_student : ℕ := 1
def total_bags_of_chips : ℕ := total_students * bags_of_chips_per_student

def apples_per_student : ℕ := 3
def total_apples : ℕ := total_students * apples_per_student

def granola_bars_per_student : ℕ := 1
def total_granola_bars : ℕ := total_students * granola_bars_per_student

-- Proof goals
theorem packed_lunch_needs :
  total_bread_slices = 360 ∧
  total_bags_of_chips = 45 ∧
  total_apples = 135 ∧
  total_granola_bars = 45 :=
by
  sorry

end NUMINAMATH_GPT_packed_lunch_needs_l308_30825


namespace NUMINAMATH_GPT_points_per_member_l308_30826

def numMembersTotal := 12
def numMembersAbsent := 4
def totalPoints := 64

theorem points_per_member (h : numMembersTotal - numMembersAbsent = 12 - 4) :
  (totalPoints / (numMembersTotal - numMembersAbsent)) = 8 := 
  sorry

end NUMINAMATH_GPT_points_per_member_l308_30826


namespace NUMINAMATH_GPT_increasing_interval_of_f_maximum_value_of_f_l308_30828

open Real

def f (x : ℝ) : ℝ := x^2 - 2 * x

-- Consider x in the interval [-2, 4]
def domain_x (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 4

theorem increasing_interval_of_f :
  ∃a b : ℝ, (a, b) = (1, 4) ∧ ∀ x y : ℝ, domain_x x → domain_x y → a ≤ x → x < y → y ≤ b → f x < f y := sorry

theorem maximum_value_of_f :
  ∃ M : ℝ, M = 8 ∧ ∀ x : ℝ, domain_x x → f x ≤ M := sorry

end NUMINAMATH_GPT_increasing_interval_of_f_maximum_value_of_f_l308_30828


namespace NUMINAMATH_GPT_perspective_square_area_l308_30834

theorem perspective_square_area (a b : ℝ) (ha : a = 4 ∨ b = 4) : 
  a * a = 16 ∨ (2 * b) * (2 * b) = 64 :=
by 
sorry

end NUMINAMATH_GPT_perspective_square_area_l308_30834


namespace NUMINAMATH_GPT_triangle_area_ellipse_l308_30896

open Real

noncomputable def ellipse_foci (a b : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  let c := sqrt (a^2 + b^2)
  ((-c, 0), (c, 0))

theorem triangle_area_ellipse 
  (a : ℝ) (b : ℝ) 
  (h1 : a = sqrt 2) (h2 : b = 1) 
  (F1 F2 : ℝ × ℝ) 
  (hfoci : ellipse_foci a b = (F1, F2))
  (hF2 : F2 = (sqrt 3, 0))
  (A B : ℝ × ℝ)
  (hA : A = (0, -1))
  (hB : B = (0, -1))
  (h_inclination : ∃ θ, θ = pi / 4 ∧ (B.1 - A.1) / (B.2 - A.2) = tan θ) :
  F1 = (-sqrt 3, 0) → 
  1/2 * (B.1 - A.1) * (B.2 - A.2) = 4/3 :=
sorry

end NUMINAMATH_GPT_triangle_area_ellipse_l308_30896


namespace NUMINAMATH_GPT_vacation_costs_l308_30864

theorem vacation_costs :
  let a := 15
  let b := 22.5
  let c := 22.5
  a + b + c = 45 → b - a = 7.5 := by
sorry

end NUMINAMATH_GPT_vacation_costs_l308_30864


namespace NUMINAMATH_GPT_largest_integer_b_l308_30839

theorem largest_integer_b (b : ℤ) : (b^2 < 60) → b ≤ 7 :=
by sorry

end NUMINAMATH_GPT_largest_integer_b_l308_30839


namespace NUMINAMATH_GPT_chinese_money_plant_sales_l308_30819

/-- 
Consider a scenario where a plant supplier sells 20 pieces of orchids for $50 each 
and some pieces of potted Chinese money plant for $25 each. He paid his two workers $40 each 
and bought new pots worth $150. The plant supplier had $1145 left from his earnings. 
Prove that the number of pieces of potted Chinese money plants sold by the supplier is 15.
-/
theorem chinese_money_plant_sales (earnings_orchids earnings_per_orchid: ℤ)
  (num_orchids: ℤ)
  (earnings_plants earnings_per_plant: ℤ)
  (worker_wage num_workers: ℤ)
  (new_pots_cost remaining_money: ℤ)
  (earnings: ℤ)
  (P : earnings_orchids = num_orchids * earnings_per_orchid)
  (Q : earnings = earnings_orchids + earnings_plants)
  (R : earnings - (worker_wage * num_workers + new_pots_cost) = remaining_money)
  (conditions: earnings_per_orchid = 50 ∧ num_orchids = 20 ∧ earnings_per_plant = 25 ∧ worker_wage = 40 ∧ num_workers = 2 ∧ new_pots_cost = 150 ∧ remaining_money = 1145):
  earnings_plants / earnings_per_plant = 15 := 
by
  sorry

end NUMINAMATH_GPT_chinese_money_plant_sales_l308_30819


namespace NUMINAMATH_GPT_harry_items_left_l308_30851

def sea_stars : ℕ := 34
def seashells : ℕ := 21
def snails : ℕ := 29
def lost_items : ℕ := 25

def total_items : ℕ := sea_stars + seashells + snails
def remaining_items : ℕ := total_items - lost_items

theorem harry_items_left : remaining_items = 59 := by
  -- proof skipped
  sorry

end NUMINAMATH_GPT_harry_items_left_l308_30851


namespace NUMINAMATH_GPT_arithmetic_sequence_a4_eq_1_l308_30874

theorem arithmetic_sequence_a4_eq_1 
  (a : ℕ → ℝ)
  (h_arith_seq : ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d)
  (h_pos : ∀ n : ℕ, a n > 0)
  (h_eq : a 2 ^ 2 + 2 * a 2 * a 6 + a 6 ^ 2 - 4 = 0) : 
  a 4 = 1 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_a4_eq_1_l308_30874


namespace NUMINAMATH_GPT_sufficient_not_necessary_condition_l308_30878

theorem sufficient_not_necessary_condition (x k : ℝ) (p : x ≥ k) (q : (2 - x) / (x + 1) < 0) :
  (∀ x, x ≥ k → ((2 - x) / (x + 1) < 0)) ∧ (∃ x, (2 - x) / (x + 1) < 0 ∧ x < k) → k > 2 := by
  sorry

end NUMINAMATH_GPT_sufficient_not_necessary_condition_l308_30878


namespace NUMINAMATH_GPT_ship_speed_in_still_water_l308_30862

theorem ship_speed_in_still_water
  (x y : ℝ)
  (h1: x + y = 32)
  (h2: x - y = 28)
  (h3: x > y) : 
  x = 30 := 
sorry

end NUMINAMATH_GPT_ship_speed_in_still_water_l308_30862


namespace NUMINAMATH_GPT_angle_A_is_pi_over_4_l308_30889

theorem angle_A_is_pi_over_4
  (A B C : ℝ)
  (a b c : ℝ)
  (h : a^2 = b^2 + c^2 - 2 * b * c * Real.sin A) :
  A = Real.pi / 4 :=
  sorry

end NUMINAMATH_GPT_angle_A_is_pi_over_4_l308_30889


namespace NUMINAMATH_GPT_juanitas_dessert_cost_l308_30871

theorem juanitas_dessert_cost :
  let brownie_cost := 2.50
  let ice_cream_cost := 1.00
  let syrup_cost := 0.50
  let nuts_cost := 1.50
  let num_scoops_ice_cream := 2
  let num_syrups := 2
  let total_cost := brownie_cost + num_scoops_ice_cream * ice_cream_cost + num_syrups * syrup_cost + nuts_cost
  total_cost = 7.00 :=
by
  sorry

end NUMINAMATH_GPT_juanitas_dessert_cost_l308_30871


namespace NUMINAMATH_GPT_max_amount_paul_received_l308_30801

theorem max_amount_paul_received :
  ∃ (numBplus numA numAplus : ℕ),
  (numBplus + numA + numAplus = 10) ∧ 
  (numAplus ≥ 2 → 
    let BplusReward := 5;
    let AReward := 2 * BplusReward;
    let AplusReward := 15;
    let Total := numAplus * AplusReward + numA * (2 * AReward) + numBplus * (2 * BplusReward);
    Total = 190
  ) :=
sorry

end NUMINAMATH_GPT_max_amount_paul_received_l308_30801


namespace NUMINAMATH_GPT_percentage_deducted_from_list_price_l308_30886

-- Definitions based on conditions
def cost_price : ℝ := 85.5
def marked_price : ℝ := 112.5
def profit_rate : ℝ := 0.25 -- 25% profit

noncomputable def selling_price : ℝ := cost_price * (1 + profit_rate)

theorem percentage_deducted_from_list_price:
  ∃ d : ℝ, d = 5 ∧ selling_price = marked_price * (1 - d / 100) :=
by
  sorry

end NUMINAMATH_GPT_percentage_deducted_from_list_price_l308_30886


namespace NUMINAMATH_GPT_simplify_expression_l308_30849

theorem simplify_expression (x y z : ℝ) : ((x + y) - (z - y)) - ((x + z) - (y + z)) = 3 * y - z := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l308_30849


namespace NUMINAMATH_GPT_most_appropriate_sampling_l308_30881

def total_students := 126 + 280 + 95
def adjusted_total_students := 126 - 1 + 280 + 95
def required_sample_size := 100

def elementary_proportion (total : Nat) (sample : Nat) : Nat := (sample * 126) / total
def middle_proportion (total : Nat) (sample : Nat) : Nat := (sample * 280) / total
def high_proportion (total : Nat) (sample : Nat) : Nat := (sample * 95) / total

theorem most_appropriate_sampling :
  required_sample_size = elementary_proportion adjusted_total_students required_sample_size + 
                         middle_proportion adjusted_total_students required_sample_size + 
                         high_proportion adjusted_total_students required_sample_size :=
by
  sorry

end NUMINAMATH_GPT_most_appropriate_sampling_l308_30881


namespace NUMINAMATH_GPT_intersection_M_N_l308_30803

def M (x : ℝ) : Prop := 2 - x > 0
def N (x : ℝ) : Prop := 1 ≤ x ∧ x ≤ 3

theorem intersection_M_N:
  {x : ℝ | M x} ∩ {x : ℝ | N x} = {x : ℝ | 1 ≤ x ∧ x < 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l308_30803


namespace NUMINAMATH_GPT_max_area_rectangle_l308_30827

theorem max_area_rectangle (P : ℕ) (hP : P = 40) : ∃ A : ℕ, A = 100 ∧ ∀ x y : ℕ, 2 * (x + y) = P → x * y ≤ A := by
  sorry

end NUMINAMATH_GPT_max_area_rectangle_l308_30827


namespace NUMINAMATH_GPT_magazine_cost_l308_30899

theorem magazine_cost (C M : ℝ) 
  (h1 : 4 * C = 8 * M) 
  (h2 : 12 * C = 24) : 
  M = 1 :=
by
  sorry

end NUMINAMATH_GPT_magazine_cost_l308_30899


namespace NUMINAMATH_GPT_bobby_consumption_l308_30893

theorem bobby_consumption :
  let initial_candy := 28
  let additional_candy_portion := 3/4 * 42
  let chocolate_portion := 1/2 * 63
  initial_candy + additional_candy_portion + chocolate_portion = 91 := 
by {
  let initial_candy : ℝ := 28
  let additional_candy_portion : ℝ := 3/4 * 42
  let chocolate_portion : ℝ := 1/2 * 63
  sorry
}

end NUMINAMATH_GPT_bobby_consumption_l308_30893


namespace NUMINAMATH_GPT_pascal_triangle_eighth_row_l308_30844

def sum_interior_numbers (n : ℕ) : ℕ :=
  2^(n-1) - 2

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose (n-1) (k-1) 

theorem pascal_triangle_eighth_row:
  sum_interior_numbers 8 = 126 ∧ binomial_coefficient 8 3 = 21 :=
by
  sorry

end NUMINAMATH_GPT_pascal_triangle_eighth_row_l308_30844


namespace NUMINAMATH_GPT_find_oxygen_weight_l308_30863

-- Definitions of given conditions
def molecular_weight : ℝ := 68
def weight_hydrogen : ℝ := 1
def weight_chlorine : ℝ := 35.5

-- Definition of unknown atomic weight of oxygen
def weight_oxygen : ℝ := 15.75

-- Mathematical statement to prove
theorem find_oxygen_weight :
  weight_hydrogen + weight_chlorine + 2 * weight_oxygen = molecular_weight := by
sorry

end NUMINAMATH_GPT_find_oxygen_weight_l308_30863


namespace NUMINAMATH_GPT_power_inequality_l308_30840

theorem power_inequality (a b c : ℕ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hcb : c ≥ b) : 
  a^b * (a + b)^c > c^b * a^c := 
sorry

end NUMINAMATH_GPT_power_inequality_l308_30840


namespace NUMINAMATH_GPT_soda_cost_l308_30898

theorem soda_cost (b s f : ℕ) (h1 : 3 * b + 2 * s + 2 * f = 590) (h2 : 2 * b + 3 * s + f = 610) : s = 140 :=
sorry

end NUMINAMATH_GPT_soda_cost_l308_30898


namespace NUMINAMATH_GPT_network_structure_l308_30892

theorem network_structure 
  (n : ℕ)
  (is_acquainted : Fin n → Fin n → Prop)
  (H_symmetric : ∀ x y, is_acquainted x y = is_acquainted y x) 
  (H_common_acquaintance : ∀ x y, ¬ is_acquainted x y → ∃! z : Fin n, is_acquainted x z ∧ is_acquainted y z) :
  ∃ (G : SimpleGraph (Fin n)), (∀ x y, G.Adj x y = is_acquainted x y) ∧
    (∀ x y, ¬ G.Adj x y → (∃ (z1 z2 : Fin n), G.Adj x z1 ∧ G.Adj y z1 ∧ G.Adj x z2 ∧ G.Adj y z2)) :=
by
  sorry

end NUMINAMATH_GPT_network_structure_l308_30892


namespace NUMINAMATH_GPT_inequality_a4_b4_c4_geq_l308_30850

theorem inequality_a4_b4_c4_geq (a b c : ℝ) : 
  a^4 + b^4 + c^4 ≥ a^2 * b^2 + b^2 * c^2 + c^2 * a^2 := 
by
  sorry

end NUMINAMATH_GPT_inequality_a4_b4_c4_geq_l308_30850


namespace NUMINAMATH_GPT_no_values_satisfy_equation_l308_30842

-- Define the sum of the digits function S
noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

-- Define the sum of digits of the sum of the digits function S(S(n))
noncomputable def sum_of_sum_of_digits (n : ℕ) : ℕ :=
  sum_of_digits (sum_of_digits n)

-- Theorem statement about the number of n satisfying n + S(n) + S(S(n)) = 2099
theorem no_values_satisfy_equation :
  (∃ n : ℕ, n > 0 ∧ n + sum_of_digits n + sum_of_sum_of_digits n = 2099) ↔ False := sorry

end NUMINAMATH_GPT_no_values_satisfy_equation_l308_30842


namespace NUMINAMATH_GPT_find_nth_term_of_arithmetic_seq_l308_30809

def is_arithmetic_seq (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

def is_geometric_progression (a1 a2 a5 : ℝ) :=
  a1 * a5 = a2^2

theorem find_nth_term_of_arithmetic_seq (a : ℕ → ℝ) (d : ℝ) (h_arith : is_arithmetic_seq a d)
    (h_a1 : a 1 = 1) (h_nonzero : d ≠ 0) (h_geom : is_geometric_progression (a 1) (a 2) (a 5)) : 
    ∀ n, a n = 2 * n - 1 :=
by
  sorry

end NUMINAMATH_GPT_find_nth_term_of_arithmetic_seq_l308_30809


namespace NUMINAMATH_GPT_geometric_sequence_common_ratio_l308_30831

theorem geometric_sequence_common_ratio {a : ℕ → ℝ} {S : ℕ → ℝ} (q : ℝ) 
  (h_pos : ∀ n, 0 < a n)
  (h_geo : ∀ n, S n = a 0 * (1 - q ^ n) / (1 - q))  
  (h_condition : ∀ n : ℕ+, S (2 * n) / S n < 5) :
  0 < q ∧ q ≤ 1 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_common_ratio_l308_30831


namespace NUMINAMATH_GPT_annual_increase_in_living_space_l308_30812

-- Definitions based on conditions
def population_2000 : ℕ := 200000
def living_space_2000_per_person : ℝ := 8
def target_living_space_2004_per_person : ℝ := 10
def annual_growth_rate : ℝ := 0.01
def years : ℕ := 4

-- Goal stated as a theorem
theorem annual_increase_in_living_space :
  let final_population := population_2000 * (1 + annual_growth_rate)^years
  let total_living_space_2004 := target_living_space_2004_per_person * final_population
  let initial_living_space := living_space_2000_per_person * population_2000
  let total_additional_space := total_living_space_2004 - initial_living_space
  let average_annual_increase := total_additional_space / years
  average_annual_increase = 120500.0 :=
sorry

end NUMINAMATH_GPT_annual_increase_in_living_space_l308_30812


namespace NUMINAMATH_GPT_range_of_a_l308_30887

theorem range_of_a
  (x y : ℝ) (hx : 0 < x) (hy : 0 < y) 
  (h1 : x + 2 * y + 4 = 4 * x * y)
  (h2 : ∀ a : ℝ, (x + 2 * y) * a ^ 2 + 2 * a + 2 * x * y - 34 ≥ 0) : 
  ∀ a : ℝ, a ≤ -3 ∨ a ≥ 5 / 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l308_30887


namespace NUMINAMATH_GPT_power_mod_7_l308_30817

theorem power_mod_7 {a : ℤ} (h : a = 3) : (a ^ 123) % 7 = 6 := by
  sorry

end NUMINAMATH_GPT_power_mod_7_l308_30817


namespace NUMINAMATH_GPT_remainder_x3_minus_4x2_plus_3x_plus_2_div_x_minus_1_l308_30884

def p (x : ℝ) : ℝ := x^3 - 4 * x^2 + 3 * x + 2

theorem remainder_x3_minus_4x2_plus_3x_plus_2_div_x_minus_1 :
  p 1 = 2 := by
  -- solution needed, for now we put a placeholder
  sorry

end NUMINAMATH_GPT_remainder_x3_minus_4x2_plus_3x_plus_2_div_x_minus_1_l308_30884


namespace NUMINAMATH_GPT_sum_pos_integers_9_l308_30891

theorem sum_pos_integers_9 (x y z : ℕ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z)
  (h_eq : 30 / 7 = x + 1 / (y + 1 / z)) : x + y + z = 9 :=
sorry

end NUMINAMATH_GPT_sum_pos_integers_9_l308_30891


namespace NUMINAMATH_GPT_number_of_common_tangents_between_circleC_and_circleD_l308_30845

noncomputable def circleC := { p : ℝ × ℝ | p.1^2 + p.2^2 = 1 }

noncomputable def circleD := { p : ℝ × ℝ | p.1^2 + p.2^2 - 4 * p.1 + 2 * p.2 - 4 = 0 }

theorem number_of_common_tangents_between_circleC_and_circleD : 
    ∃ (num_tangents : ℕ), num_tangents = 2 :=
by
    -- Proving the number of common tangents is 2
    sorry

end NUMINAMATH_GPT_number_of_common_tangents_between_circleC_and_circleD_l308_30845


namespace NUMINAMATH_GPT_f_10_l308_30877

noncomputable def f : ℕ → ℕ
| 0       => 1
| (n + 1) => 2 * f n

theorem f_10 : f 10 = 2^10 :=
by
  -- This would be filled in with the necessary proof steps to show f(10) = 2^10
  sorry

end NUMINAMATH_GPT_f_10_l308_30877


namespace NUMINAMATH_GPT_proof_l308_30832

-- Definition of the logical statements
def all_essays_correct (maria : Type) : Prop := sorry
def passed_course (maria : Type) : Prop := sorry

-- Condition provided in the problem
axiom condition : ∀ (maria : Type), all_essays_correct maria → passed_course maria

-- We need to prove this
theorem proof (maria : Type) : ¬ (passed_course maria) → ¬ (all_essays_correct maria) :=
by sorry

end NUMINAMATH_GPT_proof_l308_30832


namespace NUMINAMATH_GPT_kingfisher_catch_difference_l308_30873

def pelicanFish : Nat := 13
def fishermanFish (K : Nat) : Nat := 3 * (pelicanFish + K)
def fishermanConditionFish : Nat := pelicanFish + 86

theorem kingfisher_catch_difference (K : Nat) (h1 : K > pelicanFish)
  (h2 : fishermanFish K = fishermanConditionFish) :
  K - pelicanFish = 7 := by
  sorry

end NUMINAMATH_GPT_kingfisher_catch_difference_l308_30873


namespace NUMINAMATH_GPT_cyclist_go_south_speed_l308_30866

noncomputable def speed_of_cyclist_go_south (v : ℝ) : Prop :=
  let north_speed := 10 -- speed of cyclist going north in kmph
  let time := 2 -- time in hours
  let distance := 50 -- distance apart in km
  (north_speed + v) * time = distance

theorem cyclist_go_south_speed (v : ℝ) : speed_of_cyclist_go_south v → v = 15 :=
by
  intro h
  -- Proof part is skipped
  sorry

end NUMINAMATH_GPT_cyclist_go_south_speed_l308_30866


namespace NUMINAMATH_GPT_primes_unique_l308_30841

-- Let's define that p, q, r are prime numbers, and define the main conditions.
def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem primes_unique (p q r : ℕ) 
  (hp : is_prime p) 
  (hq : is_prime q)
  (hr : is_prime r)
  (div1 : (p^4 - 1) % (q * r) = 0)
  (div2 : (q^4 - 1) % (p * r) = 0)
  (div3 : (r^4 - 1) % (p * q) = 0) :
  (p = 2 ∧ q = 3 ∧ r = 5) ∨ (p = 2 ∧ q = 5 ∧ r = 3) ∨ (p = 3 ∧ q = 2 ∧ r = 5) ∨ 
  (p = 3 ∧ q = 5 ∧ r = 2) ∨ (p = 5 ∧ q = 2 ∧ r = 3) ∨ (p = 5 ∧ q = 3 ∧ r = 2) :=
by sorry

end NUMINAMATH_GPT_primes_unique_l308_30841


namespace NUMINAMATH_GPT_angle_quadrant_l308_30856

theorem angle_quadrant (α : ℝ) (h1 : Real.sin α > 0) (h2 : Real.cos α < 0) : 
  0 < (π - α) ∧ (π - α) < π  :=
by
  sorry

end NUMINAMATH_GPT_angle_quadrant_l308_30856


namespace NUMINAMATH_GPT_union_of_sets_l308_30822

-- Define the sets A and B
def A : Set ℝ := { x | x^2 - x - 2 < 0 }
def B : Set ℝ := { x | 1 < x ∧ x < 4 }

-- Define the set representing the union's result
def C : Set ℝ := { x | -1 < x ∧ x < 4 }

-- The theorem statement
theorem union_of_sets : ∀ x : ℝ, (x ∈ (A ∪ B) ↔ x ∈ C) :=
by
  sorry

end NUMINAMATH_GPT_union_of_sets_l308_30822


namespace NUMINAMATH_GPT_probability_all_same_flips_l308_30860

noncomputable def four_same_flips_probability : ℚ := 
  (∑' n : ℕ, if n > 0 then (1/2)^(4*n) else 0)

theorem probability_all_same_flips : 
  four_same_flips_probability = 1 / 15 := 
sorry

end NUMINAMATH_GPT_probability_all_same_flips_l308_30860


namespace NUMINAMATH_GPT_smallest_power_of_7_not_palindrome_l308_30829

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

theorem smallest_power_of_7_not_palindrome : ∃ n : ℕ, n > 0 ∧ 7^n = 2401 ∧ ¬is_palindrome (7^n) ∧ (∀ m : ℕ, m > 0 ∧ ¬is_palindrome (7^m) → 7^n ≤ 7^m) :=
by
  sorry

end NUMINAMATH_GPT_smallest_power_of_7_not_palindrome_l308_30829


namespace NUMINAMATH_GPT_proof_expectation_red_balls_drawn_l308_30861

noncomputable def expectation_red_balls_drawn : Prop :=
  let total_ways := Nat.choose 5 2
  let ways_2_red := Nat.choose 3 2
  let ways_1_red_1_yellow := Nat.choose 3 1 * Nat.choose 2 1
  let p_X_eq_2 := (ways_2_red : ℝ) / total_ways
  let p_X_eq_1 := (ways_1_red_1_yellow : ℝ) / total_ways
  let expectation := 2 * p_X_eq_2 + 1 * p_X_eq_1
  expectation = 1.2

theorem proof_expectation_red_balls_drawn :
  expectation_red_balls_drawn :=
by
  sorry

end NUMINAMATH_GPT_proof_expectation_red_balls_drawn_l308_30861


namespace NUMINAMATH_GPT_highest_visitors_at_4pm_yellow_warning_time_at_12_30pm_l308_30885

-- Definitions for cumulative visitors entering and leaving
def y (x : ℕ) : ℕ := 850 * x + 100
def z (x : ℕ) : ℕ := 200 * x - 200

-- Definition for total number of visitors at time x
def w (x : ℕ) : ℕ := y x - z x

-- Proof problem statements
theorem highest_visitors_at_4pm :
  ∀x, x ≤ 9 → w 9 ≥ w x :=
sorry

theorem yellow_warning_time_at_12_30pm :
  ∃x, w x = 2600 :=
sorry

end NUMINAMATH_GPT_highest_visitors_at_4pm_yellow_warning_time_at_12_30pm_l308_30885


namespace NUMINAMATH_GPT_intersecting_lines_l308_30855

theorem intersecting_lines (p q r s t : ℝ) : (∃ u v : ℝ, p * u^2 + q * v^2 + r * u + s * v + t = 0) →
  ( ∃ p q : ℝ, p * q < 0 ∧ 4 * t = r^2 / p + s^2 / q ) :=
sorry

end NUMINAMATH_GPT_intersecting_lines_l308_30855


namespace NUMINAMATH_GPT_field_trip_buses_l308_30888

-- Definitions of conditions
def fifth_graders : ℕ := 109
def sixth_graders : ℕ := 115
def seventh_graders : ℕ := 118
def teachers_per_grade : ℕ := 4
def parents_per_grade : ℕ := 2
def grades : ℕ := 3
def seats_per_bus : ℕ := 72

-- Total calculations
def total_students : ℕ := fifth_graders + sixth_graders + seventh_graders
def chaperones_per_grade : ℕ := teachers_per_grade + parents_per_grade
def total_chaperones : ℕ := chaperones_per_grade * grades
def total_people : ℕ := total_students + total_chaperones
def buses_needed : ℕ := (total_people + seats_per_bus - 1) / seats_per_bus

theorem field_trip_buses : buses_needed = 6 := by
  unfold buses_needed
  unfold total_people total_students total_chaperones chaperones_per_grade
  norm_num
  sorry

end NUMINAMATH_GPT_field_trip_buses_l308_30888


namespace NUMINAMATH_GPT_area_of_rectangular_field_l308_30867

-- Define the conditions
variables (l w : ℝ)

def perimeter_condition : Prop := 2 * l + 2 * w = 100
def length_width_relation : Prop := l = 3 * w

-- Define the area
def area : ℝ := l * w

-- Prove the area given the conditions
theorem area_of_rectangular_field (h1 : perimeter_condition l w) (h2 : length_width_relation l w) : area l w = 468.75 :=
by sorry

end NUMINAMATH_GPT_area_of_rectangular_field_l308_30867


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l308_30802

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

theorem necessary_but_not_sufficient (f : ℝ → ℝ) :
  (f 0 = 0) →
  (∀ x : ℝ, f (-x) = -f x) →
  ¬∀ f' : ℝ → ℝ, (f' 0 = 0 → ∀ y : ℝ, f' (-y) = -f' y)
:= by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l308_30802


namespace NUMINAMATH_GPT_proof_problem_l308_30847

noncomputable def f (x : ℝ) := Real.tan (x + (Real.pi / 4))

theorem proof_problem :
  (- (3 * Real.pi) / 4 < 1 - Real.pi ∧ 1 - Real.pi < -1 ∧ -1 < 0 ∧ 0 < Real.pi / 4) →
  f 0 > f (-1) ∧ f (-1) > f 1 := by
  sorry

end NUMINAMATH_GPT_proof_problem_l308_30847


namespace NUMINAMATH_GPT_log_identity_l308_30890

theorem log_identity :
  (Real.log 25 / Real.log 10) - 2 * (Real.log (1 / 2) / Real.log 10) = 2 :=
by
  sorry

end NUMINAMATH_GPT_log_identity_l308_30890


namespace NUMINAMATH_GPT_minimize_dot_product_l308_30843

def vector := ℝ × ℝ

def OA : vector := (2, 2)
def OB : vector := (4, 1)

def dot_product (u v : vector) : ℝ :=
  u.1 * v.1 + u.2 * v.2

def AP (P : vector) : vector :=
  (P.1 - OA.1, P.2 - OA.2)

def BP (P : vector) : vector :=
  (P.1 - OB.1, P.2 - OB.2)

def is_on_x_axis (P : vector) : Prop :=
  P.2 = 0

theorem minimize_dot_product :
  ∃ (P : vector), is_on_x_axis P ∧ dot_product (AP P) (BP P) = ( (P.1 - 3) ^ 2 + 1) ∧ P = (3, 0) :=
by
  sorry

end NUMINAMATH_GPT_minimize_dot_product_l308_30843


namespace NUMINAMATH_GPT_rectangle_other_side_l308_30853

theorem rectangle_other_side (A x y : ℝ) (hA : A = 1 / 8) (hx : x = 1 / 2) (hArea : A = x * y) :
    y = 1 / 4 := 
  sorry

end NUMINAMATH_GPT_rectangle_other_side_l308_30853


namespace NUMINAMATH_GPT_paco_cookies_l308_30814

theorem paco_cookies :
  let initial_cookies := 25
  let ate_cookies := 5
  let remaining_cookies_after_eating := initial_cookies - ate_cookies
  let gave_away_cookies := 4
  let remaining_cookies_after_giving := remaining_cookies_after_eating - gave_away_cookies
  let bought_cookies := 3
  let final_cookies := remaining_cookies_after_giving + bought_cookies
  let combined_bought_and_gave_away := gave_away_cookies + bought_cookies
  (ate_cookies - combined_bought_and_gave_away) = -2 :=
by sorry

end NUMINAMATH_GPT_paco_cookies_l308_30814


namespace NUMINAMATH_GPT_div_identity_l308_30837

theorem div_identity (a b c : ℚ) (h1 : a / b = 3) (h2 : b / c = 2 / 5) : c / a = 5 / 6 := by
  sorry

end NUMINAMATH_GPT_div_identity_l308_30837
