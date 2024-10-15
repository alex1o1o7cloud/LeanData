import Mathlib

namespace NUMINAMATH_GPT_intersection_M_N_l2214_221435

def M : Set ℝ := { x | -2 ≤ x ∧ x < 2 }
def N : Set ℝ := { x | x ≥ -2 }

theorem intersection_M_N : M ∩ N = { x | -2 ≤ x ∧ x < 2 } := by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l2214_221435


namespace NUMINAMATH_GPT_max_median_cans_per_customer_l2214_221486

theorem max_median_cans_per_customer : 
    ∀ (total_cans : ℕ) (total_customers : ℕ), 
    total_cans = 252 → total_customers = 100 →
    (∀ (cans_per_customer : ℕ),
    1 ≤ cans_per_customer) →
    (∃ (max_median : ℝ),
    max_median = 3.5) :=
by
  sorry

end NUMINAMATH_GPT_max_median_cans_per_customer_l2214_221486


namespace NUMINAMATH_GPT_inequality_solution_l2214_221489

theorem inequality_solution (x : ℝ) : 3 * x ^ 2 + x - 2 < 0 ↔ -1 < x ∧ x < 2 / 3 :=
by
  -- The proof should factor the quadratic expression and apply the rule for solving strict inequalities
  sorry

end NUMINAMATH_GPT_inequality_solution_l2214_221489


namespace NUMINAMATH_GPT_fourth_house_number_l2214_221465

theorem fourth_house_number (sum: ℕ) (k x: ℕ) (h1: sum = 78) (h2: k ≥ 4)
  (h3: (k+1) * (x + k) = 78) : x + 6 = 14 :=
by
  sorry

end NUMINAMATH_GPT_fourth_house_number_l2214_221465


namespace NUMINAMATH_GPT_chenny_candies_l2214_221473

def friends_count : ℕ := 7
def candies_per_friend : ℕ := 2
def candies_have : ℕ := 10

theorem chenny_candies : 
    (friends_count * candies_per_friend - candies_have) = 4 := by
    sorry

end NUMINAMATH_GPT_chenny_candies_l2214_221473


namespace NUMINAMATH_GPT_multiplication_result_l2214_221469

theorem multiplication_result : 
  (500 * 2468 * 0.2468 * 100) = 30485120 :=
by
  sorry

end NUMINAMATH_GPT_multiplication_result_l2214_221469


namespace NUMINAMATH_GPT_complement_union_eq_l2214_221442

variable (U : Set ℝ) (M N : Set ℝ)

noncomputable def complement_union (U M N : Set ℝ) : Set ℝ :=
  U \ (M ∪ N)

theorem complement_union_eq :
  U = Set.univ → 
  M = {x | |x| < 1} → 
  N = {y | ∃ x, y = 2^x} → 
  complement_union U M N = {x | x ≤ -1} :=
by
  intros hU hM hN
  unfold complement_union
  sorry

end NUMINAMATH_GPT_complement_union_eq_l2214_221442


namespace NUMINAMATH_GPT_slope_of_line_determined_by_solutions_l2214_221419

theorem slope_of_line_determined_by_solutions :
  (∀ (x₁ y₁ x₂ y₂ : ℝ), (4 / x₁ + 6 / y₁ = 0) ∧ (4 / x₂ + 6 / y₂ = 0) →
    (y₂ - y₁) / (x₂ - x₁) = -3 / 2) :=
sorry

end NUMINAMATH_GPT_slope_of_line_determined_by_solutions_l2214_221419


namespace NUMINAMATH_GPT_minimum_value_of_sum_l2214_221492

theorem minimum_value_of_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = 1) : 
    1 / (2 * a + b) + 1 / (2 * b + c) + 1 / (2 * c + a) >= 3 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_sum_l2214_221492


namespace NUMINAMATH_GPT_simplify_expression_l2214_221407

theorem simplify_expression (a b c : ℝ) : a - (a - b + c) = b - c :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l2214_221407


namespace NUMINAMATH_GPT_train_crossing_time_l2214_221495

theorem train_crossing_time
    (length_of_train : ℕ)
    (speed_of_train_kmph : ℕ)
    (length_of_bridge : ℕ)
    (h_train_length : length_of_train = 160)
    (h_speed_kmph : speed_of_train_kmph = 45)
    (h_bridge_length : length_of_bridge = 215)
  : length_of_train + length_of_bridge / ((speed_of_train_kmph * 1000) / 3600) = 30 :=
by
  rw [h_train_length, h_speed_kmph, h_bridge_length]
  norm_num
  sorry

end NUMINAMATH_GPT_train_crossing_time_l2214_221495


namespace NUMINAMATH_GPT_space_diagonals_Q_l2214_221414

-- Definitions based on the conditions
def vertices (Q : Type) : ℕ := 30
def edges (Q : Type) : ℕ := 70
def faces (Q : Type) : ℕ := 40
def triangular_faces (Q : Type) : ℕ := 20
def quadrilateral_faces (Q : Type) : ℕ := 15
def pentagon_faces (Q : Type) : ℕ := 5

-- Problem Statement
theorem space_diagonals_Q :
  ∀ (Q : Type),
  vertices Q = 30 →
  edges Q = 70 →
  faces Q = 40 →
  triangular_faces Q = 20 →
  quadrilateral_faces Q = 15 →
  pentagon_faces Q = 5 →
  ∃ d : ℕ, d = 310 := 
by
  -- At this point only the structure of the proof is set up.
  sorry

end NUMINAMATH_GPT_space_diagonals_Q_l2214_221414


namespace NUMINAMATH_GPT_percentage_cats_less_dogs_l2214_221471

theorem percentage_cats_less_dogs (C D F : ℕ) (h1 : C < D) (h2 : F = 2 * D) (h3 : C + D + F = 304) (h4 : F = 160) :
  ((D - C : ℕ) * 100 / D : ℕ) = 20 := 
sorry

end NUMINAMATH_GPT_percentage_cats_less_dogs_l2214_221471


namespace NUMINAMATH_GPT_ball_hits_ground_l2214_221406

theorem ball_hits_ground (t : ℝ) (y : ℝ) : 
  (y = -8 * t^2 - 12 * t + 72) → 
  (y = 0) → 
  t = 3 := 
by
  sorry

end NUMINAMATH_GPT_ball_hits_ground_l2214_221406


namespace NUMINAMATH_GPT_part_I_extreme_values_part_II_three_distinct_real_roots_part_III_compare_sizes_l2214_221457

noncomputable def f (x : ℝ) (p q : ℝ) : ℝ := (1 / 3) * x^3 + (1 / 2) * (p - 1) * x^2 + q * x

theorem part_I_extreme_values : 
  (∀ x, f x (-3) 3 = (1 / 3) * x^3 - 2 * x^2 + 3 * x) → 
  (f 1 (-3) 3 = f 3 (-3) 3) := 
sorry

theorem part_II_three_distinct_real_roots : 
  (∀ x, f x (-3) 3 = (1 / 3) * x^3 - 2 * x^2 + 3 * x) → 
  (∀ g : ℝ → ℝ, g x = f x (-3) 3 - 1 → 
  (∀ x, g x ≠ 0) → 
  ∃ a b c, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ g a = 0 ∧ g b = 0 ∧ g c = 0) :=
sorry

theorem part_III_compare_sizes (x1 x2 p a l q: ℝ) :
  f (x : ℝ) (-3) 3 = (1 / 3) * x^3 - 2 * x^2 + 3 * x → 
  x1 < x2 → 
  x2 - x1 > l → 
  x1 > a → 
  (a^2 + p * a + q) > x1 := 
sorry

end NUMINAMATH_GPT_part_I_extreme_values_part_II_three_distinct_real_roots_part_III_compare_sizes_l2214_221457


namespace NUMINAMATH_GPT_marly_needs_3_bags_l2214_221452

-- Definitions based on the problem conditions
def milk : ℕ := 2
def chicken_stock : ℕ := 3 * milk
def vegetables : ℕ := 1
def total_soup : ℕ := milk + chicken_stock + vegetables
def bag_capacity : ℕ := 3

-- The theorem to prove the number of bags required
theorem marly_needs_3_bags : total_soup / bag_capacity = 3 := 
sorry

end NUMINAMATH_GPT_marly_needs_3_bags_l2214_221452


namespace NUMINAMATH_GPT_number_of_members_in_league_l2214_221475

-- Define the costs of the items considering the conditions
def sock_cost : ℕ := 6
def tshirt_cost : ℕ := sock_cost + 3
def shorts_cost : ℕ := sock_cost + 2

-- Define the total cost for one member
def total_cost_one_member : ℕ := 
  2 * (sock_cost + tshirt_cost + shorts_cost)

-- Given total expenditure
def total_expenditure : ℕ := 4860

-- Define the theorem to be proved
theorem number_of_members_in_league :
  total_expenditure / total_cost_one_member = 106 :=
by 
  sorry

end NUMINAMATH_GPT_number_of_members_in_league_l2214_221475


namespace NUMINAMATH_GPT_tic_tac_toe_alex_wins_second_X_l2214_221421

theorem tic_tac_toe_alex_wins_second_X :
  ∃ b : ℕ, b = 12 := 
sorry

end NUMINAMATH_GPT_tic_tac_toe_alex_wins_second_X_l2214_221421


namespace NUMINAMATH_GPT_two_trains_cross_time_l2214_221445

/-- Definition for the two trains' parameters -/
structure Train :=
  (length : ℝ)  -- length in meters
  (speed : ℝ)  -- speed in km/hr

/-- The parameters of Train 1 and Train 2 -/
def train1 : Train := { length := 140, speed := 60 }
def train2 : Train := { length := 160, speed := 40 }

noncomputable def relative_speed_mps (t1 t2 : Train) : ℝ :=
  (t1.speed + t2.speed) * (5 / 18)

noncomputable def total_length (t1 t2 : Train) : ℝ :=
  t1.length + t2.length

noncomputable def time_to_cross (t1 t2 : Train) : ℝ :=
  total_length t1 t2 / relative_speed_mps t1 t2

theorem two_trains_cross_time :
  time_to_cross train1 train2 = 10.8 := by
  sorry

end NUMINAMATH_GPT_two_trains_cross_time_l2214_221445


namespace NUMINAMATH_GPT_lcm_of_two_numbers_l2214_221400

-- Definitions based on the conditions
variable (a b l : ℕ)

-- The conditions from the problem
def hcf_ab : Nat := 9
def prod_ab : Nat := 1800

-- The main statement to prove
theorem lcm_of_two_numbers : Nat.lcm a b = 200 :=
by
  -- Skipping the proof implementation
  sorry

end NUMINAMATH_GPT_lcm_of_two_numbers_l2214_221400


namespace NUMINAMATH_GPT_quotient_remainder_div_by_18_l2214_221470

theorem quotient_remainder_div_by_18 (M q : ℕ) (h : M = 54 * q + 37) : 
  ∃ k r, M = 18 * k + r ∧ r < 18 ∧ k = 3 * q + 2 ∧ r = 1 :=
by sorry

end NUMINAMATH_GPT_quotient_remainder_div_by_18_l2214_221470


namespace NUMINAMATH_GPT_value_of_g_at_3_l2214_221466

theorem value_of_g_at_3 (g : ℕ → ℕ) (h : ∀ x, g (x + 2) = 2 * x + 3) : g 3 = 5 := by
  sorry

end NUMINAMATH_GPT_value_of_g_at_3_l2214_221466


namespace NUMINAMATH_GPT_difference_of_numbers_l2214_221468

variables (x y : ℝ)

-- Definitions corresponding to the conditions
def sum_of_numbers (x y : ℝ) : Prop := x + y = 30
def product_of_numbers (x y : ℝ) : Prop := x * y = 200

-- The proof statement in Lean
theorem difference_of_numbers (x y : ℝ) 
  (h1: sum_of_numbers x y) 
  (h2: product_of_numbers x y) : x - y = 10 ∨ y - x = 10 :=
by
  sorry

end NUMINAMATH_GPT_difference_of_numbers_l2214_221468


namespace NUMINAMATH_GPT_locker_number_problem_l2214_221464

theorem locker_number_problem 
  (cost_per_digit : ℝ)
  (total_cost : ℝ)
  (one_digit_cost : ℝ)
  (two_digit_cost : ℝ)
  (three_digit_cost : ℝ) :
  cost_per_digit = 0.03 →
  one_digit_cost = 0.27 →
  two_digit_cost = 5.40 →
  three_digit_cost = 81.00 →
  total_cost = 206.91 →
  10 * cost_per_digit = six_cents →
  9 * cost_per_digit = three_cents →
  1 * 9 * cost_per_digit = one_digit_cost →
  2 * 45 * cost_per_digit = two_digit_cost →
  3 * 300 * cost_per_digit = three_digit_cost →
  (999 * 3 + x * 4 = 6880) →
  ∀ total_locker : ℕ, total_locker = 2001 := sorry

end NUMINAMATH_GPT_locker_number_problem_l2214_221464


namespace NUMINAMATH_GPT_circle_diameter_and_circumference_l2214_221427

theorem circle_diameter_and_circumference (A : ℝ) (hA : A = 225 * π) : 
  ∃ r d C, r = 15 ∧ d = 2 * r ∧ C = 2 * π * r ∧ d = 30 ∧ C = 30 * π :=
by
  sorry

end NUMINAMATH_GPT_circle_diameter_and_circumference_l2214_221427


namespace NUMINAMATH_GPT_polyhedron_space_diagonals_l2214_221447

theorem polyhedron_space_diagonals (V E F T P : ℕ) (total_pairs_of_vertices total_edges total_face_diagonals : ℕ)
  (hV : V = 30)
  (hE : E = 70)
  (hF : F = 40)
  (hT : T = 30)
  (hP : P = 10)
  (h_total_pairs_of_vertices : total_pairs_of_vertices = 30 * 29 / 2)
  (h_total_face_diagonals : total_face_diagonals = 5 * 10)
  :
  total_pairs_of_vertices - E - total_face_diagonals = 315 := 
by
  sorry

end NUMINAMATH_GPT_polyhedron_space_diagonals_l2214_221447


namespace NUMINAMATH_GPT_mindy_messages_total_l2214_221497

theorem mindy_messages_total (P : ℕ) (h1 : 83 = 9 * P - 7) : 83 + P = 93 :=
  by
    sorry

end NUMINAMATH_GPT_mindy_messages_total_l2214_221497


namespace NUMINAMATH_GPT_technicans_permanent_50pct_l2214_221438

noncomputable def percentage_technicians_permanent (p : ℝ) : Prop :=
  let technicians := 0.5
  let non_technicians := 0.5
  let temporary := 0.5
  (0.5 * (1 - 0.5)) + (technicians * p) = 0.5 ->
  p = 0.5

theorem technicans_permanent_50pct (p : ℝ) :
  percentage_technicians_permanent p :=
sorry

end NUMINAMATH_GPT_technicans_permanent_50pct_l2214_221438


namespace NUMINAMATH_GPT_toby_steps_l2214_221412

theorem toby_steps (sunday tuesday wednesday thursday friday_saturday monday : ℕ) :
    sunday = 9400 →
    tuesday = 8300 →
    wednesday = 9200 →
    thursday = 8900 →
    friday_saturday = 9050 →
    7 * 9000 = 63000 →
    monday = 63000 - (sunday + tuesday + wednesday + thursday + 2 * friday_saturday) → monday = 9100 :=
by
  intros hs ht hw hth hfs htc hnm
  sorry

end NUMINAMATH_GPT_toby_steps_l2214_221412


namespace NUMINAMATH_GPT_number_of_items_l2214_221449

variable (s d : ℕ)
variable (total_money cost_sandwich cost_drink discount : ℝ)
variable (s_purchase_criterion : s > 5)
variable (total_money_value : total_money = 50.00)
variable (cost_sandwich_value : cost_sandwich = 6.00)
variable (cost_drink_value : cost_drink = 1.50)
variable (discount_value : discount = 5.00)

theorem number_of_items (h1 : total_money = 50.00)
(h2 : cost_sandwich = 6.00)
(h3 : cost_drink = 1.50)
(h4 : discount = 5.00)
(h5 : s > 5) :
  s + d = 9 :=
by
  sorry

end NUMINAMATH_GPT_number_of_items_l2214_221449


namespace NUMINAMATH_GPT_final_spent_l2214_221402

-- Define all the costs.
def albertoExpenses : ℤ := 2457 + 374 + 520 + 129 + 799
def albertoDiscountExhaust : ℤ := (799 * 5) / 100
def albertoTotalBeforeLoyaltyDiscount : ℤ := albertoExpenses - albertoDiscountExhaust
def albertoLoyaltyDiscount : ℤ := (albertoTotalBeforeLoyaltyDiscount * 7) / 100
def albertoFinal : ℤ := albertoTotalBeforeLoyaltyDiscount - albertoLoyaltyDiscount

def samaraExpenses : ℤ := 25 + 467 + 79 + 175 + 599 + 225
def samaraSalesTax : ℤ := (samaraExpenses * 6) / 100
def samaraFinal : ℤ := samaraExpenses + samaraSalesTax

def difference : ℤ := albertoFinal - samaraFinal

theorem final_spent (h : difference = 2278) : true :=
  sorry

end NUMINAMATH_GPT_final_spent_l2214_221402


namespace NUMINAMATH_GPT_quadratic_j_value_l2214_221467

theorem quadratic_j_value (a b c : ℝ) (h : a * (0 : ℝ)^2 + b * (0 : ℝ) + c = 5 * ((0 : ℝ) - 3)^2 + 15) :
  ∃ m j n, 4 * a * (0 : ℝ)^2 + 4 * b * (0 : ℝ) + 4 * c = m * ((0 : ℝ) - j)^2 + n ∧ j = 3 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_j_value_l2214_221467


namespace NUMINAMATH_GPT_max_value_expression_l2214_221440

noncomputable def factorize_15000 := 2^3 * 3 * 5^4

theorem max_value_expression (x y : ℕ) (h1 : 6 * x^2 - 5 * x * y + y^2 = 0) (h2 : x ∣ factorize_15000) : 
  2 * x + 3 * y ≤ 60000 := sorry

end NUMINAMATH_GPT_max_value_expression_l2214_221440


namespace NUMINAMATH_GPT_find_cost_per_sq_foot_l2214_221477

noncomputable def monthly_rent := 2800 / 2
noncomputable def old_annual_rent (C : ℝ) := 750 * C * 12
noncomputable def new_annual_rent := monthly_rent * 12
noncomputable def annual_savings := old_annual_rent - new_annual_rent

theorem find_cost_per_sq_foot (C : ℝ):
    (750 * C * 12 - 2800 / 2 * 12 = 1200) ↔ (C = 2) :=
sorry

end NUMINAMATH_GPT_find_cost_per_sq_foot_l2214_221477


namespace NUMINAMATH_GPT_min_sum_of_grid_numbers_l2214_221422

-- Definition of the 2x2 grid and the problem conditions
variables (a b c d : ℕ)
variables (a_pos : a > 0) (b_pos : b > 0) (c_pos : c > 0) (d_pos : d > 0)

-- Lean statement for the minimum sum proof problem
theorem min_sum_of_grid_numbers :
  a + b + c + d + a * b + c * d + a * c + b * d = 2015 → a + b + c + d = 88 :=
by
  sorry

end NUMINAMATH_GPT_min_sum_of_grid_numbers_l2214_221422


namespace NUMINAMATH_GPT_planar_figure_area_l2214_221410

noncomputable def side_length : ℝ := 10
noncomputable def area_of_square : ℝ := side_length * side_length
noncomputable def number_of_squares : ℕ := 6
noncomputable def total_area_of_planar_figure : ℝ := number_of_squares * area_of_square

theorem planar_figure_area : total_area_of_planar_figure = 600 :=
by
  sorry

end NUMINAMATH_GPT_planar_figure_area_l2214_221410


namespace NUMINAMATH_GPT_geometric_sequence_ratio_l2214_221439

-- Definitions and conditions from part a)
def q : ℚ := 1 / 2

def sum_of_first_n (a1 : ℚ) (n : ℕ) : ℚ :=
  a1 * (1 - q ^ n) / (1 - q)

def a_n (a1 : ℚ) (n : ℕ) : ℚ :=
  a1 * q ^ (n - 1)

-- Theorem representing the proof problem from part c)
theorem geometric_sequence_ratio (a1 : ℚ) : 
  (sum_of_first_n a1 4) / (a_n a1 3) = 15 / 2 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_ratio_l2214_221439


namespace NUMINAMATH_GPT_verify_other_root_l2214_221425

variable {a b c x : ℝ}

-- Given conditions
axiom distinct_non_zero_constants : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a

axiom root_two : a * 2^2 - (a + b + c) * 2 + (b + c) = 0

-- Function under test
noncomputable def other_root (a b c : ℝ) : ℝ :=
  (b + c - a) / a

-- The goal statement
theorem verify_other_root :
  ∀ (a b c : ℝ), (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a) → (a * 2^2 - (a + b + c) * 2 + (b + c) = 0) → 
  (∀ x, (a * x^2 - (a + b + c) * x + (b + c) = 0) → (x = 2 ∨ x = (b + c - a) / a)) :=
by
  intros a b c h1 h2 x h3
  sorry

end NUMINAMATH_GPT_verify_other_root_l2214_221425


namespace NUMINAMATH_GPT_divides_of_exponentiation_l2214_221499

theorem divides_of_exponentiation (n : ℕ) : 7 ∣ 3^(12 * n + 1) + 2^(6 * n + 2) := 
  sorry

end NUMINAMATH_GPT_divides_of_exponentiation_l2214_221499


namespace NUMINAMATH_GPT_sum_of_three_numbers_l2214_221490

def a : ℚ := 859 / 10
def b : ℚ := 531 / 100
def c : ℚ := 43 / 2

theorem sum_of_three_numbers : a + b + c = 11271 / 100 := by
  sorry

end NUMINAMATH_GPT_sum_of_three_numbers_l2214_221490


namespace NUMINAMATH_GPT_range_of_a_l2214_221430

theorem range_of_a (a : ℝ) : (∀ x > 0, a - x - |Real.log x| ≤ 0) → a ≤ 1 := by
  sorry

end NUMINAMATH_GPT_range_of_a_l2214_221430


namespace NUMINAMATH_GPT_functional_eq_uniq_l2214_221444

noncomputable def f (x : ℝ) : ℝ := sorry

theorem functional_eq_uniq (f : ℝ → ℝ) (h : ∀ x y : ℝ, (f x * f y - f (x * y)) / 4 = x^2 + y^2 + 2) : 
  ∀ x : ℝ, f x = x^2 + 3 :=
by 
  sorry

end NUMINAMATH_GPT_functional_eq_uniq_l2214_221444


namespace NUMINAMATH_GPT_profit_per_meter_l2214_221405

theorem profit_per_meter 
  (total_meters : ℕ)
  (cost_price_per_meter : ℝ)
  (total_selling_price : ℝ)
  (h1 : total_meters = 92)
  (h2 : cost_price_per_meter = 83.5)
  (h3 : total_selling_price = 9890) : 
  (total_selling_price - total_meters * cost_price_per_meter) / total_meters = 24.1 :=
by
  sorry

end NUMINAMATH_GPT_profit_per_meter_l2214_221405


namespace NUMINAMATH_GPT_nina_running_distance_l2214_221454

theorem nina_running_distance (total_distance : ℝ) (initial_run : ℝ) (num_initial_runs : ℕ) :
  total_distance = 0.8333333333333334 →
  initial_run = 0.08333333333333333 →
  num_initial_runs = 2 →
  (total_distance - initial_run * num_initial_runs = 0.6666666666666667) :=
by
  intros h_total h_initial h_num
  sorry

end NUMINAMATH_GPT_nina_running_distance_l2214_221454


namespace NUMINAMATH_GPT_unique_solution_for_a_l2214_221413

theorem unique_solution_for_a (a : ℝ) :
  (∃! (x y : ℝ), 
    (x * Real.cos a + y * Real.sin a = 5 * Real.cos a + 2 * Real.sin a) ∧
    (-3 ≤ x + 2 * y ∧ x + 2 * y ≤ 7) ∧
    (-9 ≤ 3 * x - 4 * y ∧ 3 * x - 4 * y ≤ 1)) ↔ 
  (∃ k : ℤ, a = Real.arctan 4 + k * Real.pi ∨ a = -Real.arctan 2 + k * Real.pi) :=
sorry

end NUMINAMATH_GPT_unique_solution_for_a_l2214_221413


namespace NUMINAMATH_GPT_probability_of_chosen_figure_is_circle_l2214_221426

-- Define the total number of figures and number of circles.
def total_figures : ℕ := 12
def number_of_circles : ℕ := 5

-- Define the probability calculation.
def probability_of_circle (total : ℕ) (circles : ℕ) : ℚ := circles / total

-- State the theorem using the defined conditions.
theorem probability_of_chosen_figure_is_circle : 
  probability_of_circle total_figures number_of_circles = 5 / 12 :=
by
  sorry  -- Placeholder for the actual proof.

end NUMINAMATH_GPT_probability_of_chosen_figure_is_circle_l2214_221426


namespace NUMINAMATH_GPT_gcd_sum_abcde_edcba_l2214_221451

-- Definition to check if digits are consecutive
def consecutive_digits (a b c d e : ℤ) : Prop :=
  b = a + 1 ∧ c = a + 2 ∧ d = a + 3 ∧ e = a + 4

-- Definition of the five-digit number in the form abcde
def abcde (a b c d e : ℤ) : ℤ :=
  10000 * a + 1000 * b + 100 * c + 10 * d + e

-- Definition of the five-digit number in the form edcba
def edcba (a b c d e : ℤ) : ℤ :=
  10000 * e + 1000 * d + 100 * c + 10 * b + a

-- Definition which sums both abcde and edcba
def sum_abcde_edcba (a b c d e : ℤ) : ℤ :=
  abcde a b c d e + edcba a b c d e

-- Lean theorem statement for the problem
theorem gcd_sum_abcde_edcba (a b c d e : ℤ) (h : consecutive_digits a b c d e) :
  Int.gcd (sum_abcde_edcba a b c d e) 11211 = 11211 :=
by
  sorry

end NUMINAMATH_GPT_gcd_sum_abcde_edcba_l2214_221451


namespace NUMINAMATH_GPT_remainder_is_one_l2214_221494

theorem remainder_is_one (dividend divisor quotient remainder : ℕ) 
  (h1 : dividend = 222) 
  (h2 : divisor = 13)
  (h3 : quotient = 17)
  (h4 : dividend = divisor * quotient + remainder) : remainder = 1 :=
sorry

end NUMINAMATH_GPT_remainder_is_one_l2214_221494


namespace NUMINAMATH_GPT_concert_songs_l2214_221441

def total_songs (g : ℕ) : ℕ := (9 + 3 + 9 + g) / 3

theorem concert_songs 
  (g : ℕ) 
  (h1 : 9 + 3 + 9 + g = 3 * total_songs g) 
  (h2 : 3 + g % 4 = 0) 
  (h3 : 4 ≤ g ∧ g ≤ 9) 
  : total_songs g = 9 ∨ total_songs g = 10 := 
sorry

end NUMINAMATH_GPT_concert_songs_l2214_221441


namespace NUMINAMATH_GPT_money_brought_to_store_l2214_221482

theorem money_brought_to_store : 
  let sheet_cost := 42
  let rope_cost := 18
  let propane_and_burner_cost := 14
  let helium_cost_per_ounce := 1.5
  let height_per_ounce := 113
  let max_height := 9492
  let total_item_cost := sheet_cost + rope_cost + propane_and_burner_cost
  let helium_needed := max_height / height_per_ounce
  let helium_total_cost := helium_needed * helium_cost_per_ounce
  total_item_cost + helium_total_cost = 200 :=
by
  sorry

end NUMINAMATH_GPT_money_brought_to_store_l2214_221482


namespace NUMINAMATH_GPT_point_symmetric_about_y_axis_l2214_221429

theorem point_symmetric_about_y_axis (A B : ℝ × ℝ) 
  (hA : A = (1, -2)) 
  (hSym : B = (-A.1, A.2)) :
  B = (-1, -2) := 
by 
  sorry

end NUMINAMATH_GPT_point_symmetric_about_y_axis_l2214_221429


namespace NUMINAMATH_GPT_non_student_ticket_price_l2214_221453

theorem non_student_ticket_price (x : ℕ) : 
  (∃ (n_student_ticket_price ticket_count total_revenue student_tickets : ℕ),
    n_student_ticket_price = 9 ∧
    ticket_count = 2000 ∧
    total_revenue = 20960 ∧
    student_tickets = 520 ∧
    (student_tickets * n_student_ticket_price + (ticket_count - student_tickets) * x = total_revenue)) -> 
  x = 11 := 
by
  -- placeholder for proof
  sorry

end NUMINAMATH_GPT_non_student_ticket_price_l2214_221453


namespace NUMINAMATH_GPT_rational_roots_of_quadratic_l2214_221485

theorem rational_roots_of_quadratic (k : ℤ) (h : k > 0) :
  (∃ x : ℚ, k * x^2 + 12 * x + k = 0) ↔ (k = 3 ∨ k = 6) :=
by
  sorry

end NUMINAMATH_GPT_rational_roots_of_quadratic_l2214_221485


namespace NUMINAMATH_GPT_max_erasers_l2214_221423

theorem max_erasers (p n e : ℕ) (h₁ : p ≥ 1) (h₂ : n ≥ 1) (h₃ : e ≥ 1) (h₄ : 3 * p + 4 * n + 8 * e = 60) :
  e ≤ 5 :=
sorry

end NUMINAMATH_GPT_max_erasers_l2214_221423


namespace NUMINAMATH_GPT_hyperbola_ratio_l2214_221431

theorem hyperbola_ratio (a b c : ℝ)
  (h1 : a > 0) (h2 : b > 0)
  (h_eq : a^2 - b^2 = 1)
  (h_ecc : 2 = c / a)
  (h_focus : c = 1) :
  a / b = Real.sqrt 3 / 3 := by
  have ha : a = 1 / 2 := sorry
  have hc : c = 1 := h_focus
  have hb : b = Real.sqrt 3 / 2 := sorry
  exact sorry

end NUMINAMATH_GPT_hyperbola_ratio_l2214_221431


namespace NUMINAMATH_GPT_like_terms_exponents_l2214_221491

theorem like_terms_exponents (m n : ℤ) 
  (h1 : 3 = m - 2) 
  (h2 : n + 1 = 2) : m - n = 4 := 
by
  sorry

end NUMINAMATH_GPT_like_terms_exponents_l2214_221491


namespace NUMINAMATH_GPT_perpendicular_lines_slope_l2214_221462

theorem perpendicular_lines_slope {a : ℝ} :
  (∃ (a : ℝ), (∀ x y : ℝ, x + 2 * y - 1 = 0 → a * x - y - 1 = 0) ∧ (a * (-1 / 2)) = -1) → a = 2 :=
by sorry

end NUMINAMATH_GPT_perpendicular_lines_slope_l2214_221462


namespace NUMINAMATH_GPT_find_g_5_l2214_221483

theorem find_g_5 (g : ℝ → ℝ) (h : ∀ x : ℝ, g x + 3 * g (1 - x) = 2 * x ^ 2 + 1) : g 5 = 8 :=
sorry

end NUMINAMATH_GPT_find_g_5_l2214_221483


namespace NUMINAMATH_GPT_det_of_matrix_l2214_221476

variable {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ)

theorem det_of_matrix (h1 : 1 ≤ n)
  (h2 : A ^ 7 + A ^ 5 + A ^ 3 + A - 1 = 0) :
  0 < Matrix.det A :=
sorry

end NUMINAMATH_GPT_det_of_matrix_l2214_221476


namespace NUMINAMATH_GPT_remaining_apps_eq_files_plus_more_initial_apps_eq_16_l2214_221484

-- Defining the initial number of files
def initial_files: ℕ := 9

-- Defining the remaining number of files and apps
def remaining_files: ℕ := 5
def remaining_apps: ℕ := 12

-- Given: Dave has 7 more apps than files left
def apps_more_than_files: ℕ := 7

-- Equating the given condition 12 = 5 + 7
theorem remaining_apps_eq_files_plus_more :
  remaining_apps = remaining_files + apps_more_than_files := by
  sorry -- This would trivially prove as 12 = 5+7

-- Proving the number of initial apps
theorem initial_apps_eq_16 (A: ℕ) (h1: initial_files = 9) (h2: remaining_files = 5) (h3: remaining_apps = 12) (h4: apps_more_than_files = 7):
  A - remaining_apps = initial_files - remaining_files → A = 16 := by
  sorry

end NUMINAMATH_GPT_remaining_apps_eq_files_plus_more_initial_apps_eq_16_l2214_221484


namespace NUMINAMATH_GPT_tina_total_income_is_correct_l2214_221417

-- Definitions based on the conditions
def hourly_wage : ℝ := 18.0
def regular_hours_per_day : ℝ := 8
def overtime_hours_per_day_weekday : ℝ := 2
def double_overtime_hours_per_day_weekend : ℝ := 2

def overtime_rate : ℝ := hourly_wage + 0.5 * hourly_wage
def double_overtime_rate : ℝ := 2 * hourly_wage

def weekday_hours_per_day : ℝ := 10
def weekend_hours_per_day : ℝ := 12

def regular_pay_per_day : ℝ := hourly_wage * regular_hours_per_day
def overtime_pay_per_day_weekday : ℝ := overtime_rate * overtime_hours_per_day_weekday
def double_overtime_pay_per_day_weekend : ℝ := double_overtime_rate * double_overtime_hours_per_day_weekend

def total_weekday_pay_per_day : ℝ := regular_pay_per_day + overtime_pay_per_day_weekday
def total_weekend_pay_per_day : ℝ := regular_pay_per_day + overtime_pay_per_day_weekday + double_overtime_pay_per_day_weekend

def number_of_weekdays : ℝ := 5
def number_of_weekends : ℝ := 2

def total_weekday_income : ℝ := total_weekday_pay_per_day * number_of_weekdays
def total_weekend_income : ℝ := total_weekend_pay_per_day * number_of_weekends

def total_weekly_income : ℝ := total_weekday_income + total_weekend_income

-- The theorem we need to prove
theorem tina_total_income_is_correct : total_weekly_income = 1530 := by
  sorry

end NUMINAMATH_GPT_tina_total_income_is_correct_l2214_221417


namespace NUMINAMATH_GPT_syllogistic_reasoning_l2214_221416

theorem syllogistic_reasoning (a b c : Prop) (h1 : b → c) (h2 : a → b) : a → c :=
by sorry

end NUMINAMATH_GPT_syllogistic_reasoning_l2214_221416


namespace NUMINAMATH_GPT_min_value_l2214_221458

theorem min_value (a b : ℝ) (h_pos : 0 < a ∧ 0 < b) (h_ab : a * b = 1) (h_a_2b : a = 2 * b) :
  a + 2 * b = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_min_value_l2214_221458


namespace NUMINAMATH_GPT_find_x0_l2214_221479

noncomputable def f (x : ℝ) : ℝ := 13 - 8 * x + x^2

theorem find_x0 :
  (∃ x0 : ℝ, deriv f x0 = 4) → ∃ x0 : ℝ, x0 = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_x0_l2214_221479


namespace NUMINAMATH_GPT_f_1_eq_2_f_6_plus_f_7_eq_15_f_2012_eq_3849_l2214_221472

noncomputable def f : ℕ+ → ℕ+ := sorry

axiom f_properties (n : ℕ+) : f (f n) = 3 * n

axiom f_increasing (n : ℕ+) : f (n + 1) > f n

-- Proof for f(1)
theorem f_1_eq_2 : f 1 = 2 := 
by
sorry

-- Proof for f(6) + f(7)
theorem f_6_plus_f_7_eq_15 : f 6 + f 7 = 15 := 
by
sorry

-- Proof for f(2012)
theorem f_2012_eq_3849 : f 2012 = 3849 := 
by
sorry

end NUMINAMATH_GPT_f_1_eq_2_f_6_plus_f_7_eq_15_f_2012_eq_3849_l2214_221472


namespace NUMINAMATH_GPT_ms_warren_running_time_l2214_221408

theorem ms_warren_running_time 
  (t : ℝ) 
  (ht_total_distance : 6 * t + 2 * 0.5 = 3) : 
  60 * t = 20 := by 
  sorry

end NUMINAMATH_GPT_ms_warren_running_time_l2214_221408


namespace NUMINAMATH_GPT_value_of_A_l2214_221436

theorem value_of_A (A C : ℤ) (h₁ : 2 * A - C + 4 = 26) (h₂ : C = 6) : A = 14 :=
by sorry

end NUMINAMATH_GPT_value_of_A_l2214_221436


namespace NUMINAMATH_GPT_tickets_sold_second_half_l2214_221404

-- Definitions from conditions
def total_tickets := 9570
def first_half_tickets := 3867

-- Theorem to prove the number of tickets sold in the second half of the season
theorem tickets_sold_second_half : total_tickets - first_half_tickets = 5703 :=
by sorry

end NUMINAMATH_GPT_tickets_sold_second_half_l2214_221404


namespace NUMINAMATH_GPT_cat_finishes_food_on_sunday_l2214_221474

-- Define the constants and parameters
def daily_morning_consumption : ℚ := 2 / 5
def daily_evening_consumption : ℚ := 1 / 5
def total_food : ℕ := 8
def days_in_week : ℕ := 7

-- Define the total daily consumption
def total_daily_consumption : ℚ := daily_morning_consumption + daily_evening_consumption

-- Define the sum of consumptions over each day until the day when all food is consumed
def food_remaining_after_days (days : ℕ) : ℚ := total_food - days * total_daily_consumption

-- Proposition that the food is finished on Sunday
theorem cat_finishes_food_on_sunday :
  ∃ days : ℕ, (food_remaining_after_days days ≤ 0) ∧ days ≡ 7 [MOD days_in_week] :=
sorry

end NUMINAMATH_GPT_cat_finishes_food_on_sunday_l2214_221474


namespace NUMINAMATH_GPT_calculation_result_l2214_221459

theorem calculation_result :
  let a := 0.0088
  let b := 4.5
  let c := 0.05
  let d := 0.1
  let e := 0.008
  (a * b) / (c * d * e) = 990 :=
by
  sorry

end NUMINAMATH_GPT_calculation_result_l2214_221459


namespace NUMINAMATH_GPT_maximum_a_for_monotonically_increasing_interval_l2214_221496

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x)
noncomputable def g (x : ℝ) : ℝ := f (x - (Real.pi / 4))

theorem maximum_a_for_monotonically_increasing_interval :
  ∀ a : ℝ, (∀ x y : ℝ, 0 ≤ x ∧ x ≤ a ∧ 0 ≤ y ∧ y ≤ a ∧ x < y → g x < g y) → a ≤ Real.pi / 4 := 
by
  sorry

end NUMINAMATH_GPT_maximum_a_for_monotonically_increasing_interval_l2214_221496


namespace NUMINAMATH_GPT_find_n_l2214_221433

theorem find_n (n k : ℕ) (h_pos : k > 0) (h_calls : ∀ (s : Finset (Fin n)), s.card = n-2 → (∃ (f : Finset (Fin n × Fin n)), f.card = 3^k ∧ ∀ (x y : Fin n), (x, y) ∈ f → x ≠ y)) : n = 5 := 
sorry

end NUMINAMATH_GPT_find_n_l2214_221433


namespace NUMINAMATH_GPT_visited_neither_l2214_221415

def people_total : ℕ := 90
def visited_iceland : ℕ := 55
def visited_norway : ℕ := 33
def visited_both : ℕ := 51

theorem visited_neither :
  people_total - (visited_iceland + visited_norway - visited_both) = 53 := by
  sorry

end NUMINAMATH_GPT_visited_neither_l2214_221415


namespace NUMINAMATH_GPT_find_coeff_and_root_range_l2214_221434

def f (x : ℝ) (a b : ℝ) : ℝ := a * x^3 - b * x + 4

theorem find_coeff_and_root_range (a b : ℝ)
  (h1 : f 2 a b = - (4/3))
  (h2 : deriv (λ x => f x a b) 2 = 0) :
  a = 1 / 3 ∧ b = 4 ∧ 
  (∀ k : ℝ, (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ f x1 (1/3) 4 = k ∧ f x2 (1/3) 4 = k ∧ f x3 (1/3) 4 = k) ↔ - (4/3) < k ∧ k < 28/3) :=
sorry

end NUMINAMATH_GPT_find_coeff_and_root_range_l2214_221434


namespace NUMINAMATH_GPT_simplify_complex_expr_l2214_221480

theorem simplify_complex_expr : ∀ i : ℂ, i^2 = -1 → 3 * (4 - 2 * i) + 2 * i * (3 - i) = 14 :=
by 
  intro i 
  intro h
  sorry

end NUMINAMATH_GPT_simplify_complex_expr_l2214_221480


namespace NUMINAMATH_GPT_oxen_grazing_months_l2214_221456

theorem oxen_grazing_months (a_oxen : ℕ) (a_months : ℕ) (b_oxen : ℕ) (c_oxen : ℕ) (c_months : ℕ) (total_rent : ℝ) (c_share_rent : ℝ) (x : ℕ) :
  a_oxen = 10 →
  a_months = 7 →
  b_oxen = 12 →
  c_oxen = 15 →
  c_months = 3 →
  total_rent = 245 →
  c_share_rent = 63 →
  (c_oxen * c_months) / ((a_oxen * a_months) + (b_oxen * x) + (c_oxen * c_months)) = c_share_rent / total_rent →
  x = 5 :=
sorry

end NUMINAMATH_GPT_oxen_grazing_months_l2214_221456


namespace NUMINAMATH_GPT_find_original_percentage_of_acid_l2214_221409

noncomputable def percentage_of_acid (a w : ℕ) : ℚ :=
  (a : ℚ) / (a + w : ℚ) * 100

theorem find_original_percentage_of_acid (a w : ℕ) 
  (h1 : (a : ℚ) / (a + w + 2 : ℚ) = 1 / 4)
  (h2 : (a + 2 : ℚ) / (a + w + 4 : ℚ) = 2 / 5) : 
  percentage_of_acid a w = 33.33 :=
by 
  sorry

end NUMINAMATH_GPT_find_original_percentage_of_acid_l2214_221409


namespace NUMINAMATH_GPT_eval_f_at_neg_twenty_three_sixth_pi_l2214_221493

noncomputable def f (α : ℝ) : ℝ := 
    (2 * (Real.sin (2 * Real.pi - α)) * (Real.cos (2 * Real.pi + α)) - Real.cos (-α)) / 
    (1 + Real.sin α ^ 2 + Real.sin (2 * Real.pi + α) - Real.cos (4 * Real.pi - α) ^ 2)

theorem eval_f_at_neg_twenty_three_sixth_pi : 
  f (-23 / 6 * Real.pi) = -Real.sqrt 3 :=
  sorry

end NUMINAMATH_GPT_eval_f_at_neg_twenty_three_sixth_pi_l2214_221493


namespace NUMINAMATH_GPT_avg_score_first_4_l2214_221455

-- Definitions based on conditions
def average_score_all_7 : ℝ := 56
def total_matches : ℕ := 7
def average_score_last_3 : ℝ := 69.33333333333333
def matches_first : ℕ := 4
def matches_last : ℕ := 3

-- Calculation of total runs from average scores.
def total_runs_all_7 : ℝ := average_score_all_7 * total_matches
def total_runs_last_3 : ℝ := average_score_last_3 * matches_last

-- Total runs for the first 4 matches
def total_runs_first_4 : ℝ := total_runs_all_7 - total_runs_last_3

-- Prove the average score for the first 4 matches.
theorem avg_score_first_4 :
  (total_runs_first_4 / matches_first) = 46 := 
sorry

end NUMINAMATH_GPT_avg_score_first_4_l2214_221455


namespace NUMINAMATH_GPT_taller_tree_height_l2214_221448

-- Definitions and Variables
variables (h : ℝ)

-- Conditions as Definitions
def top_difference_condition := (h - 20) / h = 5 / 7

-- Proof Statement
theorem taller_tree_height (h : ℝ) (H : top_difference_condition h) : h = 70 := 
by {
  sorry
}

end NUMINAMATH_GPT_taller_tree_height_l2214_221448


namespace NUMINAMATH_GPT_time_for_B_work_alone_l2214_221401

def work_rate_A : ℚ := 1 / 6
def work_rate_combined : ℚ := 1 / 3
def work_share_C : ℚ := 1 / 8

theorem time_for_B_work_alone : 
  ∃ x : ℚ, (work_rate_A + 1 / x = work_rate_combined - work_share_C) → x = 24 := 
sorry

end NUMINAMATH_GPT_time_for_B_work_alone_l2214_221401


namespace NUMINAMATH_GPT_gabrielle_peaches_l2214_221498

theorem gabrielle_peaches (B G : ℕ) 
  (h1 : 16 = 2 * B + 6)
  (h2 : B = G / 3) :
  G = 15 :=
by
  sorry

end NUMINAMATH_GPT_gabrielle_peaches_l2214_221498


namespace NUMINAMATH_GPT_packaging_combinations_l2214_221411

theorem packaging_combinations :
  let wraps := 10
  let ribbons := 4
  let cards := 5
  let stickers := 6
  wraps * ribbons * cards * stickers = 1200 :=
by
  rfl

end NUMINAMATH_GPT_packaging_combinations_l2214_221411


namespace NUMINAMATH_GPT_solve_for_y_l2214_221418

-- Given condition
def equation (y : ℚ) := (8 * y^2 + 90 * y + 5) / (3 * y^2 + 4 * y + 49) = 4 * y + 1

-- Prove the resulting polynomial equation
theorem solve_for_y (y : ℚ) (h : equation y) : 12 * y^3 + 11 * y^2 + 110 * y + 44 = 0 :=
sorry

end NUMINAMATH_GPT_solve_for_y_l2214_221418


namespace NUMINAMATH_GPT_problem_correct_statements_l2214_221487

def T (a b x y : ℚ) : ℚ := a * x * y + b * x - 4

theorem problem_correct_statements (a b : ℚ) (h₁ : T a b 2 1 = 2) (h₂ : T a b (-1) 2 = -8) :
  (a = 1 ∧ b = 2) ∧
  (∀ m n : ℚ, T 1 2 m n = 0 ∧ n ≠ -2 → m = 4 / (n + 2)) ∧
  ¬ (∃ m n : ℤ, T 1 2 m n = 0 ∧ n ≠ -2 ∧ m + n = 3) ∧
  (∀ k x y : ℚ, T 1 2 (k * x) y = T 1 2 (k * x) y → y = -2) ∧
  (∀ k x y : ℚ, x ≠ y → T 1 2 (k * x) y = T 1 2 (k * y) x → k = 0) :=
by
  sorry

end NUMINAMATH_GPT_problem_correct_statements_l2214_221487


namespace NUMINAMATH_GPT_maximize_revenue_l2214_221450

-- Define the conditions
def price (p : ℝ) := p ≤ 30
def toys_sold (p : ℝ) : ℝ := 150 - 4 * p
def revenue (p : ℝ) := p * (toys_sold p)

-- State the theorem to solve the problem
theorem maximize_revenue : ∃ p : ℝ, price p ∧ 
  (∀ q : ℝ, price q → revenue q ≤ revenue p) ∧ p = 18.75 :=
by {
  sorry
}

end NUMINAMATH_GPT_maximize_revenue_l2214_221450


namespace NUMINAMATH_GPT_part1_distance_part2_equation_l2214_221461

noncomputable section

-- Define the conditions for Part 1
def hyperbola_C1 (x y : ℝ) : Prop := (x^2 / 4) - (y^2 / 12) = 1

-- Define the point M(3, t) existing on hyperbola C₁
def point_on_hyperbola_C1 (t : ℝ) : Prop := hyperbola_C1 3 t

-- Define the right focus of hyperbola C1
def right_focus_C1 : ℝ × ℝ := (4, 0)

-- Part 1: Distance from point M to the right focus
theorem part1_distance (t : ℝ) (h : point_on_hyperbola_C1 t) :  
  let distance := Real.sqrt ((3 - 4)^2 + (t - 0)^2)
  distance = 4 := sorry

-- Define the conditions for Part 2
def hyperbola_C2 (x y : ℝ) (m : ℝ) : Prop := (x^2 / 4) - (y^2 / 12) = m

-- Define the point (-3, 2√6) existing on hyperbola C₂
def point_on_hyperbola_C2 (m : ℝ) : Prop := hyperbola_C2 (-3) (2 * Real.sqrt 6) m

-- Part 2: The standard equation of hyperbola C₂
theorem part2_equation (h : point_on_hyperbola_C2 (1/4)) : 
  ∀ (x y : ℝ), hyperbola_C2 x y (1/4) ↔ (x^2 - (y^2 / 3) = 1) := sorry

end NUMINAMATH_GPT_part1_distance_part2_equation_l2214_221461


namespace NUMINAMATH_GPT_find_m_l2214_221463

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x + m
noncomputable def g (x : ℝ) : ℝ := 2 * x - 2

theorem find_m : 
  ∃ m : ℝ, ∀ x : ℝ, f m x = g x → m = -2 := by
  sorry

end NUMINAMATH_GPT_find_m_l2214_221463


namespace NUMINAMATH_GPT_cone_sphere_ratio_l2214_221432

noncomputable def volume_of_sphere (r : ℝ) : ℝ :=
  (4 / 3) * Real.pi * r^3

noncomputable def volume_of_cone (r : ℝ) (h : ℝ) : ℝ :=
  (1 / 3) * Real.pi * (2 * r)^2 * h

theorem cone_sphere_ratio (r h : ℝ) (V_cone V_sphere : ℝ) (h_sphere : V_sphere = volume_of_sphere r)
  (h_cone : V_cone = volume_of_cone r h) (h_relation : V_cone = (1/3) * V_sphere) :
  (h / (2 * r) = 1 / 6) :=
by
  sorry

end NUMINAMATH_GPT_cone_sphere_ratio_l2214_221432


namespace NUMINAMATH_GPT_curve_is_hyperbola_l2214_221437

theorem curve_is_hyperbola (u : ℝ) (x y : ℝ) 
  (h1 : x = Real.cos u ^ 2)
  (h2 : y = Real.sin u ^ 4) : 
  ∃ (a b : ℝ), a ≠ 0 ∧  b ≠ 0 ∧ x / a ^ 2 - y / b ^ 2 = 1 := 
sorry

end NUMINAMATH_GPT_curve_is_hyperbola_l2214_221437


namespace NUMINAMATH_GPT_company_production_l2214_221460

theorem company_production (bottles_per_case number_of_cases total_bottles : ℕ)
  (h1 : bottles_per_case = 12)
  (h2 : number_of_cases = 10000)
  (h3 : total_bottles = number_of_cases * bottles_per_case) : 
  total_bottles = 120000 :=
by {
  -- Proof is omitted, add actual proof here
  sorry
}

end NUMINAMATH_GPT_company_production_l2214_221460


namespace NUMINAMATH_GPT_composite_for_all_n_greater_than_one_l2214_221478

theorem composite_for_all_n_greater_than_one (n : ℕ) (h : n > 1) : ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n^4 + 4^n = a * b :=
by
  sorry

end NUMINAMATH_GPT_composite_for_all_n_greater_than_one_l2214_221478


namespace NUMINAMATH_GPT_remainder_of_number_divisor_l2214_221403

-- Define the interesting number and the divisor
def number := 2519
def divisor := 9
def expected_remainder := 8

-- State the theorem to prove the remainder condition
theorem remainder_of_number_divisor :
  number % divisor = expected_remainder := by
  sorry

end NUMINAMATH_GPT_remainder_of_number_divisor_l2214_221403


namespace NUMINAMATH_GPT_point_equal_distances_l2214_221428

theorem point_equal_distances (x y : ℝ) (hx : y = x) (hxy : y - 4 = -x) (hline : x + y = 4) : x = 2 :=
by sorry

end NUMINAMATH_GPT_point_equal_distances_l2214_221428


namespace NUMINAMATH_GPT_SarahsScoreIs135_l2214_221481

variable (SarahsScore GregsScore : ℕ)

-- Conditions
def ScoreDifference (SarahsScore GregsScore : ℕ) : Prop := SarahsScore = GregsScore + 50
def AverageScore (SarahsScore GregsScore : ℕ) : Prop := (SarahsScore + GregsScore) / 2 = 110

-- Theorem statement
theorem SarahsScoreIs135 (h1 : ScoreDifference SarahsScore GregsScore) (h2 : AverageScore SarahsScore GregsScore) : SarahsScore = 135 :=
sorry

end NUMINAMATH_GPT_SarahsScoreIs135_l2214_221481


namespace NUMINAMATH_GPT_age_of_oldest_child_l2214_221446

theorem age_of_oldest_child (a1 a2 a3 x : ℕ) (h1 : a1 = 5) (h2 : a2 = 7) (h3 : a3 = 10) (h_avg : (a1 + a2 + a3 + x) / 4 = 8) : x = 10 :=
by
  sorry

end NUMINAMATH_GPT_age_of_oldest_child_l2214_221446


namespace NUMINAMATH_GPT_mitchell_more_than_antonio_l2214_221443

-- Definitions based on conditions
def mitchell_pencils : ℕ := 30
def total_pencils : ℕ := 54

-- Definition of the main question
def antonio_pencils : ℕ := total_pencils - mitchell_pencils

-- The theorem to be proved
theorem mitchell_more_than_antonio : mitchell_pencils - antonio_pencils = 6 :=
by
-- Proof is omitted
sorry

end NUMINAMATH_GPT_mitchell_more_than_antonio_l2214_221443


namespace NUMINAMATH_GPT_extremum_range_l2214_221420

noncomputable def f (a x : ℝ) : ℝ := x^3 + 2 * x^2 - a * x + 1

noncomputable def f_prime (a x : ℝ) : ℝ := 3 * x^2 + 4 * x - a

theorem extremum_range 
  (h : ∀ a : ℝ, (∃ (x : ℝ) (hx : -1 < x ∧ x < 1), f_prime a x = 0) → 
                (∀ x : ℝ, -1 < x ∧ x < 1 → f_prime a x ≠ 0)):
  ∀ a : ℝ, -1 < a ∧ a < 7 :=
sorry

end NUMINAMATH_GPT_extremum_range_l2214_221420


namespace NUMINAMATH_GPT_profit_percentage_A_is_20_l2214_221424

-- Definitions of conditions
def cost_price_A := 156 -- Cost price of the cricket bat for A
def selling_price_C := 234 -- Selling price of the cricket bat to C
def profit_percent_B := 25 / 100 -- Profit percentage for B

-- Calculations
def cost_price_B := selling_price_C / (1 + profit_percent_B) -- Cost price of the cricket bat for B
def selling_price_A := cost_price_B -- Selling price of the cricket bat for A

-- Profit and profit percentage calculations
def profit_A := selling_price_A - cost_price_A -- Profit for A
def profit_percent_A := profit_A / cost_price_A * 100 -- Profit percentage for A

-- Statement to prove
theorem profit_percentage_A_is_20 : profit_percent_A = 20 :=
by
  sorry

end NUMINAMATH_GPT_profit_percentage_A_is_20_l2214_221424


namespace NUMINAMATH_GPT_physics_students_l2214_221488

variable (B : Nat) (G : Nat) (Biology : Nat) (Physics : Nat)

axiom h1 : B = 25
axiom h2 : G = 3 * B
axiom h3 : Biology = B + G
axiom h4 : Physics = 2 * Biology

theorem physics_students : Physics = 200 :=
by
  sorry

end NUMINAMATH_GPT_physics_students_l2214_221488
