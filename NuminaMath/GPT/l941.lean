import Mathlib

namespace total_cost_correct_l941_94191

def sandwich_cost : ℝ := 2.45
def soda_cost : ℝ := 0.87
def sandwich_quantity : ℕ := 2
def soda_quantity : ℕ := 4

theorem total_cost_correct :
  sandwich_quantity * sandwich_cost + soda_quantity * soda_cost = 8.38 := 
  by
    sorry

end total_cost_correct_l941_94191


namespace relationship_abc_l941_94111

noncomputable def a (x : ℝ) (hx : 0 < x ∧ x < 1) : ℝ := (Real.sin x) / x
noncomputable def b (x : ℝ) (hx : 0 < x ∧ x < 1) : ℝ := (Real.sin (x^3)) / (x^3)
noncomputable def c (x : ℝ) (hx : 0 < x ∧ x < 1) : ℝ := ((Real.sin x)^3) / (x^3)

theorem relationship_abc (x : ℝ) (hx : 0 < x ∧ x < 1) : b x hx > a x hx ∧ a x hx > c x hx :=
by
  sorry

end relationship_abc_l941_94111


namespace number_of_participants_eq_14_l941_94103

theorem number_of_participants_eq_14 (n : ℕ) (h : n * (n - 1) / 2 = 91) : n = 14 :=
by
  sorry

end number_of_participants_eq_14_l941_94103


namespace Sarah_score_l941_94115

theorem Sarah_score (G S : ℕ) (h1 : S = G + 60) (h2 : (S + G) / 2 = 108) : S = 138 :=
by
  sorry

end Sarah_score_l941_94115


namespace part1_part2_l941_94188

-- Definitions based on the conditions
def original_sales : ℕ := 30
def profit_per_shirt_initial : ℕ := 40

-- Additional shirts sold for each 1 yuan price reduction
def additional_shirts_per_yuan : ℕ := 2

-- Price reduction example of 3 yuan
def price_reduction_example : ℕ := 3

-- New sales quantity after 3 yuan reduction
def new_sales_quantity_example := 
  original_sales + (price_reduction_example * additional_shirts_per_yuan)

-- Prove that the sales quantity is 36 shirts for a reduction of 3 yuan
theorem part1 : new_sales_quantity_example = 36 := by
  sorry

-- General price reduction variable
def price_reduction_per_item (x : ℕ) : ℕ := x
def new_profit_per_shirt (x : ℕ) : ℕ := profit_per_shirt_initial - x
def new_sales_quantity (x : ℕ) : ℕ := original_sales + (additional_shirts_per_yuan * x)
def daily_sales_profit (x : ℕ) : ℕ := (new_profit_per_shirt x) * (new_sales_quantity x)

-- Goal for daily sales profit of 1200 yuan
def goal_profit : ℕ := 1200

-- Prove that a price reduction of 25 yuan per shirt achieves a daily sales profit of 1200 yuan
theorem part2 : daily_sales_profit 25 = goal_profit := by
  sorry

end part1_part2_l941_94188


namespace cooks_number_l941_94198

variable (C W : ℕ)

theorem cooks_number (h1 : 10 * C = 3 * W) (h2 : 14 * C = 3 * (W + 12)) : C = 9 :=
by
  sorry

end cooks_number_l941_94198


namespace total_tagged_numbers_l941_94114

theorem total_tagged_numbers {W X Y Z : ℕ} 
  (hW : W = 200)
  (hX : X = W / 2)
  (hY : Y = W + X)
  (hZ : Z = 400) :
  W + X + Y + Z = 1000 := by
  sorry

end total_tagged_numbers_l941_94114


namespace classroom_activity_solution_l941_94138

theorem classroom_activity_solution 
  (x y : ℕ) 
  (h1 : x - y = 6) 
  (h2 : x * y = 45) : 
  x = 11 ∧ y = 5 :=
by
  sorry

end classroom_activity_solution_l941_94138


namespace gcd_of_36_and_54_l941_94182

theorem gcd_of_36_and_54 : Nat.gcd 36 54 = 18 := by
  -- Proof details are omitted; replaced with sorry.
  sorry

end gcd_of_36_and_54_l941_94182


namespace evaluate_integral_l941_94107

noncomputable def integral_problem : Real :=
  ∫ x in (-2 : Real)..(2 : Real), (Real.sqrt (4 - x^2) - x^2017)

theorem evaluate_integral :
  integral_problem = 2 * Real.pi :=
sorry

end evaluate_integral_l941_94107


namespace max_neg_integers_l941_94116

-- Definitions for the conditions
def areIntegers (a b c d e f : Int) : Prop := True
def sumOfProductsNeg (a b c d e f : Int) : Prop := (a * b + c * d * e * f) < 0

-- The theorem to prove
theorem max_neg_integers (a b c d e f : Int) (h1 : areIntegers a b c d e f) (h2 : sumOfProductsNeg a b c d e f) : 
  ∃ s : Nat, s = 4 := 
sorry

end max_neg_integers_l941_94116


namespace range_of_m_l941_94170

open Set

variable (f : ℝ → ℝ) (m : ℝ)

theorem range_of_m (h1 : ∀ x1 x2 : ℝ, x1 < x2 → f x1 > f x2)
  (h2 : f (2 * m) > f (1 + m)) : m < 1 :=
by {
  -- The proof would go here.
  sorry
}

end range_of_m_l941_94170


namespace maciek_total_purchase_cost_l941_94129

-- Define the cost of pretzels
def pretzel_cost : ℕ := 4

-- Define the cost of chips
def chip_cost : ℕ := pretzel_cost + (75 * pretzel_cost) / 100

-- Calculate the total cost
def total_cost : ℕ := 2 * pretzel_cost + 2 * chip_cost

-- Rewrite the math proof problem statement
theorem maciek_total_purchase_cost : total_cost = 22 :=
by
  -- Skip the proof
  sorry

end maciek_total_purchase_cost_l941_94129


namespace find_x_in_inches_l941_94110

noncomputable def x_value (x : ℝ) : Prop :=
  let area_larger_square := (4 * x) ^ 2
  let area_smaller_square := (3 * x) ^ 2
  let area_triangle := (1 / 2) * (3 * x) * (4 * x)
  let total_area := area_larger_square + area_smaller_square + area_triangle
  total_area = 1100 ∧ x = Real.sqrt (1100 / 31)

theorem find_x_in_inches (x : ℝ) : x_value x :=
by sorry

end find_x_in_inches_l941_94110


namespace factor_polynomial_sum_l941_94197

theorem factor_polynomial_sum (P Q : ℤ) :
  (∀ x : ℂ, (x^2 + 4*x + 5) ∣ (x^4 + P*x^2 + Q)) → P + Q = 19 :=
by
  intro h
  sorry

end factor_polynomial_sum_l941_94197


namespace max_roses_l941_94194

theorem max_roses (individual_cost dozen_cost two_dozen_cost budget : ℝ) 
  (h1 : individual_cost = 7.30) 
  (h2 : dozen_cost = 36) 
  (h3 : two_dozen_cost = 50) 
  (h4 : budget = 680) : 
  ∃ n, n = 316 :=
by
  sorry

end max_roses_l941_94194


namespace smaller_cube_edge_length_l941_94165

-- Given conditions
variables (s : ℝ) (volume_large_cube : ℝ) (n : ℝ)
-- n = 8 (number of smaller cubes), volume_large_cube = 1000 cm³

theorem smaller_cube_edge_length (h1 : n = 8) (h2 : volume_large_cube = 1000) :
  s^3 = volume_large_cube / n → s = 5 :=
by
  sorry

end smaller_cube_edge_length_l941_94165


namespace tangent_range_of_a_l941_94112

theorem tangent_range_of_a 
  (a : ℝ)
  (circle_eq : ∀ x y : ℝ, x^2 + y^2 + a * x + 2 * y + a^2 = 0)
  (A : ℝ × ℝ) 
  (A_eq : A = (1, 2)) :
  -2 * Real.sqrt 3 / 3 < a ∧ a < 2 * Real.sqrt 3 / 3 :=
by
  sorry

end tangent_range_of_a_l941_94112


namespace union_of_complements_eq_l941_94172

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)

theorem union_of_complements_eq :
  U = {1, 2, 3, 4, 5, 6, 7} →
  A = {2, 4, 5, 7} →
  B = {3, 4, 5} →
  ((U \ A) ∪ (U \ B) = {1, 2, 3, 6, 7}) :=
by
  intros hU hA hB
  sorry

end union_of_complements_eq_l941_94172


namespace two_pow_1000_mod_17_l941_94132

theorem two_pow_1000_mod_17 : 2^1000 % 17 = 0 :=
by {
  sorry
}

end two_pow_1000_mod_17_l941_94132


namespace arith_seq_ratio_l941_94130

theorem arith_seq_ratio {a b : ℕ → ℕ} {S T : ℕ → ℕ}
  (h₁ : ∀ n, S n = (n * (2 * a n - a 1)) / 2)
  (h₂ : ∀ n, T n = (n * (2 * b n - b 1)) / 2)
  (h₃ : ∀ n, S n / T n = (5 * n + 3) / (2 * n + 7)) :
  (a 9 / b 9 = 88 / 41) :=
sorry

end arith_seq_ratio_l941_94130


namespace people_with_diploma_percentage_l941_94181

-- Definitions of the given conditions
def P_j_and_not_d := 0.12
def P_not_j_and_d := 0.15
def P_j := 0.40

-- Definitions for intermediate values
def P_not_j := 1 - P_j
def P_not_j_d := P_not_j * P_not_j_and_d

-- Definition of the result to prove
def P_d := (P_j - P_j_and_not_d) + P_not_j_d

theorem people_with_diploma_percentage : P_d = 0.43 := by
  -- Placeholder for the proof
  sorry

end people_with_diploma_percentage_l941_94181


namespace increase_in_surface_area_l941_94180

-- Define the edge length of the original cube and other conditions
variable (a : ℝ)

-- Define the increase in surface area problem
theorem increase_in_surface_area (h : 1 ≤ 27) : 
  let original_surface_area := 6 * a^2
  let smaller_cube_edge := a / 3
  let smaller_surface_area := 6 * (smaller_cube_edge)^2
  let total_smaller_surface_area := 27 * smaller_surface_area
  total_smaller_surface_area - original_surface_area = 12 * a^2 :=
by
  -- Provided the proof to satisfy Lean 4 syntax requirements to check for correctness
  sorry

end increase_in_surface_area_l941_94180


namespace triangle_area_ratio_l941_94126

noncomputable def area_ratio (AD DC : ℝ) (h : ℝ) : ℝ :=
  (1 / 2) * AD * h / ((1 / 2) * DC * h)

theorem triangle_area_ratio (AD DC : ℝ) (h : ℝ) (condition1 : AD = 5) (condition2 : DC = 7) :
  area_ratio AD DC h = 5 / 7 :=
by
  sorry

end triangle_area_ratio_l941_94126


namespace find_c_l941_94190

def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

theorem find_c (a b c : ℝ) 
  (h1 : perpendicular (a / 2) (-2 / b))
  (h2 : a = b)
  (h3 : a * 1 - 2 * (-5) = c) 
  (h4 : 2 * 1 + b * (-5) = -c) : 
  c = 13 := by
  sorry

end find_c_l941_94190


namespace Dorottya_should_go_first_l941_94195

def probability_roll_1_or_2 : ℚ := 2 / 10

def probability_no_roll_1_or_2 : ℚ := 1 - probability_roll_1_or_2

variables {P_1 P_2 P_3 P_4 P_5 P_6 : ℚ}
  (hP1 : P_1 = probability_roll_1_or_2 * ∑' n, (probability_no_roll_1_or_2 ^ (6 * n)))
  (hP2 : P_2 = probability_roll_1_or_2 * (probability_no_roll_1_or_2 ^ 1) * ∑' n, (probability_no_roll_1_or_2 ^ (6 * n)))
  (hP3 : P_3 = probability_roll_1_or_2 * (probability_no_roll_1_or_2 ^ 2) * ∑' n, (probability_no_roll_1_or_2 ^ (6 * n)))
  (hP4 : P_4 = probability_roll_1_or_2 * (probability_no_roll_1_or_2 ^ 3) * ∑' n, (probability_no_roll_1_or_2 ^ (6 * n)))
  (hP5 : P_5 = probability_roll_1_or_2 * (probability_no_roll_1_or_2 ^ 4) * ∑' n, (probability_no_roll_1_or_2 ^ (6 * n)))
  (hP6 : P_6 = probability_roll_1_or_2 * (probability_no_roll_1_or_2 ^ 5) * ∑' n, (probability_no_roll_1_or_2 ^ (6 * n)))

theorem Dorottya_should_go_first : P_1 > P_2 ∧ P_2 > P_3 ∧ P_3 > P_4 ∧ P_4 > P_5 ∧ P_5 > P_6 :=
by {
  -- Skipping actual proof steps
  sorry
}

end Dorottya_should_go_first_l941_94195


namespace remainder_when_M_divided_by_52_l941_94186

def M : Nat := 123456789101112131415161718192021222324252627282930313233343536373839404142434445464748495051

theorem remainder_when_M_divided_by_52 : M % 52 = 0 :=
by
  sorry

end remainder_when_M_divided_by_52_l941_94186


namespace trajectory_eq_find_m_l941_94154

-- First problem: Trajectory equation
theorem trajectory_eq (P : ℝ × ℝ) (A B : ℝ × ℝ) :
  A = (1, 0) → B = (-1, 0) → 
  (dist P A) * (dist A B) = (dist P B) * (dist A B) → 
  P.snd ^ 2 = 4 * P.fst :=
by sorry

-- Second problem: Value of m
theorem find_m (P : ℝ × ℝ) (M N : ℝ × ℝ) (m : ℝ) :
  P.snd ^ 2 = 4 * P.fst → 
  M.snd = M.fst + m → 
  N.snd = N.fst + m →
  (M.fst - N.fst) * (M.snd - N.snd) + (N.snd - M.snd) * (N.fst - M.fst) = 0 →
  m ≠ 0 →
  m < 1 →
  m = -4 :=
by sorry

end trajectory_eq_find_m_l941_94154


namespace proof_problem_l941_94185

def label_sum_of_domains_specified (labels: List Nat) (domains: List Nat) : Nat :=
  let relevant_labels := labels.filter (fun l => domains.contains l)
  relevant_labels.foldl (· + ·) 0

def label_product_of_continuous_and_invertible (labels: List Nat) (properties: List Bool) : Nat :=
  let relevant_labels := labels.zip properties |>.filter (fun (_, p) => p) |>.map (·.fst)
  relevant_labels.foldl (· * ·) 1

theorem proof_problem :
  label_sum_of_domains_specified [1, 2, 3, 4] [4] = 4 ∧ label_product_of_continuous_and_invertible [1, 2, 3, 4] [true, false, true, false] = 3 :=
by
  sorry

end proof_problem_l941_94185


namespace exactly_two_talents_l941_94124

open Nat

def total_students : Nat := 50
def cannot_sing_students : Nat := 20
def cannot_dance_students : Nat := 35
def cannot_act_students : Nat := 15

theorem exactly_two_talents : 
  (total_students - cannot_sing_students) + 
  (total_students - cannot_dance_students) + 
  (total_students - cannot_act_students) - total_students = 30 := by
  sorry

end exactly_two_talents_l941_94124


namespace max_knights_between_knights_l941_94151

theorem max_knights_between_knights (knights samurais total : Nat) (condition_knights_right samurai : Nat) :
  knights = 40 → samurais = 10 → condition_knights_right = 7 → total = knights + samurais →
  ∃ max_knights, max_knights = 32 ∧ 
  (∃ (k : Nat), k ≤ total ∧ (∀ n, (0 < n) → (n < 7) → max_knights = knights - n)) :=
by
  sorry

end max_knights_between_knights_l941_94151


namespace train_length_is_500_l941_94176

def speed_kmph : ℕ := 360
def time_sec : ℕ := 5

def speed_mps (v_kmph : ℕ) : ℕ :=
  v_kmph * 1000 / 3600

def length_of_train (v_mps : ℕ) (t_sec : ℕ) : ℕ :=
  v_mps * t_sec

theorem train_length_is_500 :
  length_of_train (speed_mps speed_kmph) time_sec = 500 := 
sorry

end train_length_is_500_l941_94176


namespace solve_for_A_plus_B_l941_94146

theorem solve_for_A_plus_B (A B : ℤ) (h : ∀ ω, ω^2 + ω + 1 = 0 → ω^103 + A * ω + B = 0) : A + B = -1 :=
sorry

end solve_for_A_plus_B_l941_94146


namespace find_k_l941_94125

theorem find_k (k : ℝ) : (∀ x y : ℝ, y = k * x + 3 → (x, y) = (2, 5)) → k = 1 :=
by
  sorry

end find_k_l941_94125


namespace triangle_semicircle_l941_94153

noncomputable def triangle_semicircle_ratio : ℝ :=
  let AB := 8
  let BC := 6
  let CA := 2 * Real.sqrt 7
  let radius_AB := AB / 2
  let radius_BC := BC / 2
  let radius_CA := CA / 2
  let area_semicircle_AB := (1 / 2) * Real.pi * radius_AB ^ 2
  let area_semicircle_BC := (1 / 2) * Real.pi * radius_BC ^ 2
  let area_semicircle_CA := (1 / 2) * Real.pi * radius_CA ^ 2
  let area_triangle := AB * BC / 2
  let total_shaded_area := (area_semicircle_AB + area_semicircle_BC + area_semicircle_CA) - area_triangle
  let area_circle_CA := Real.pi * (radius_CA ^ 2)
  total_shaded_area / area_circle_CA

theorem triangle_semicircle : triangle_semicircle_ratio = 2 - (12 * Real.sqrt 3) / (7 * Real.pi) := by
  sorry

end triangle_semicircle_l941_94153


namespace pipe_a_filling_time_l941_94189

theorem pipe_a_filling_time
  (pipeA_fill_time : ℝ)
  (pipeB_fill_time : ℝ)
  (both_pipes_open : Bool)
  (pipeB_shutoff_time : ℝ)
  (overflow_time : ℝ)
  (pipeB_rate : ℝ)
  (combined_rate : ℝ)
  (a_filling_time : ℝ) :
  pipeA_fill_time = 1 / 2 :=
by
  -- Definitions directly from conditions in a)
  let pipeA_fill_time := a_filling_time
  let pipeB_fill_time := 1  -- Pipe B fills in 1 hour
  let both_pipes_open := True
  let pipeB_shutoff_time := 0.5 -- Pipe B shuts 30 minutes before overflow
  let overflow_time := 0.5  -- Tank overflows in 30 minutes
  let pipeB_rate := 1 / pipeB_fill_time
  
  -- Goal to prove
  sorry

end pipe_a_filling_time_l941_94189


namespace english_class_students_l941_94178

variables (e f s u v w : ℕ)

theorem english_class_students
  (h1 : e + u + v + w + f + s + 2 = 40)
  (h2 : e + u + v = 3 * (f + w))
  (h3 : e + u + w = 2 * (s + v)) : 
  e = 30 := 
sorry

end english_class_students_l941_94178


namespace represent_259BC_as_neg259_l941_94105

def year_AD (n: ℤ) : ℤ := n

def year_BC (n: ℕ) : ℤ := -(n : ℤ)

theorem represent_259BC_as_neg259 : year_BC 259 = -259 := 
by 
  rw [year_BC]
  norm_num

end represent_259BC_as_neg259_l941_94105


namespace length_of_major_axis_l941_94156

theorem length_of_major_axis (x y : ℝ) (h : (x^2 / 25) + (y^2 / 16) = 1) : 10 = 10 :=
by
  sorry

end length_of_major_axis_l941_94156


namespace loss_equates_to_balls_l941_94120

theorem loss_equates_to_balls
    (SP_20 : ℕ) (CP_1: ℕ) (Loss: ℕ) (x: ℕ)
    (h1 : SP_20 = 720)
    (h2 : CP_1 = 48)
    (h3 : Loss = (20 * CP_1 - SP_20))
    (h4 : Loss = x * CP_1) :
    x = 5 :=
by
  sorry

end loss_equates_to_balls_l941_94120


namespace total_bill_l941_94183

def number_of_adults := 2
def number_of_children := 5
def meal_cost := 3

theorem total_bill : number_of_adults * meal_cost + number_of_children * meal_cost = 21 :=
by
  sorry

end total_bill_l941_94183


namespace Jonah_paid_commensurate_l941_94134

def price_per_pineapple (P : ℝ) :=
  let number_of_pineapples := 6
  let rings_per_pineapple := 12
  let total_rings := number_of_pineapples * rings_per_pineapple
  let price_per_4_rings := 5
  let price_per_ring := price_per_4_rings / 4
  let total_revenue := total_rings * price_per_ring
  let profit := 72
  total_revenue - number_of_pineapples * P = profit

theorem Jonah_paid_commensurate {P : ℝ} (h : price_per_pineapple P) :
  P = 3 :=
  sorry

end Jonah_paid_commensurate_l941_94134


namespace distance_from_dormitory_to_city_l941_94118

theorem distance_from_dormitory_to_city (D : ℝ) 
  (h : (1/5) * D + (2/3) * D + 4 = D) : 
  D = 30 :=
sorry

end distance_from_dormitory_to_city_l941_94118


namespace unbroken_seashells_left_l941_94139

-- Definitions based on given conditions
def total_seashells : ℕ := 6
def cone_shells : ℕ := 3
def conch_shells : ℕ := 3
def broken_cone_shells : ℕ := 2
def broken_conch_shells : ℕ := 2
def given_away_conch_shells : ℕ := 1

-- Mathematical statement to prove the final count of unbroken seashells
theorem unbroken_seashells_left : 
  (cone_shells - broken_cone_shells) + (conch_shells - broken_conch_shells - given_away_conch_shells) = 1 :=
by 
  -- Calculation (steps omitted per instructions)
  sorry

end unbroken_seashells_left_l941_94139


namespace bc_fraction_of_ad_l941_94145

theorem bc_fraction_of_ad
  {A B D C : Type}
  (length_AB length_BD length_AC length_CD length_AD length_BC : ℝ)
  (h1 : length_AB = 3 * length_BD)
  (h2 : length_AC = 4 * length_CD)
  (h3 : length_AD = length_AB + length_BD + length_CD)
  (h4 : length_BC = length_AC - length_AB) :
  length_BC / length_AD = 5 / 6 :=
by sorry

end bc_fraction_of_ad_l941_94145


namespace samir_climbed_318_stairs_l941_94173

theorem samir_climbed_318_stairs 
  (S : ℕ)
  (h1 : ∀ {V : ℕ}, V = (S / 2) + 18 → S + V = 495) 
  (half_S : ∃ k : ℕ, S = k * 2) -- assumes S is even 
  : S = 318 := 
by
  sorry

end samir_climbed_318_stairs_l941_94173


namespace sum_of_numbers_l941_94101

theorem sum_of_numbers : 
  12345 + 23451 + 34512 + 45123 + 51234 = 166665 := 
sorry

end sum_of_numbers_l941_94101


namespace paintable_wall_area_l941_94192

theorem paintable_wall_area :
  let bedrooms := 4
  let length := 14
  let width := 11
  let height := 9
  let doorway_window_area := 70
  let area_one_bedroom := 
    2 * (length * height) + 2 * (width * height) - doorway_window_area
  let total_paintable_area := bedrooms * area_one_bedroom
  total_paintable_area = 1520 := by
  sorry

end paintable_wall_area_l941_94192


namespace intersection_of_M_and_N_l941_94113

def M : Set ℝ := { x | (x - 3) / (x - 1) ≤ 0 }
def N : Set ℝ := { x | -6 * x^2 + 11 * x - 4 > 0 }

theorem intersection_of_M_and_N : M ∩ N = { x | 1 < x ∧ x < 4 / 3 } :=
by 
  sorry

end intersection_of_M_and_N_l941_94113


namespace painting_problem_l941_94187

theorem painting_problem (initial_painters : ℕ) (initial_days : ℚ) (initial_rate : ℚ) (new_days : ℚ) (new_rate : ℚ) : 
  initial_painters = 6 ∧ initial_days = 5/2 ∧ initial_rate = 2 ∧ new_days = 2 ∧ new_rate = 2.5 →
  ∃ additional_painters : ℕ, additional_painters = 0 :=
by
  intros h
  sorry

end painting_problem_l941_94187


namespace equal_water_and_alcohol_l941_94121

variable (a m : ℝ)

-- Conditions:
-- Cup B initially contains m liters of water.
-- Transfers as specified in the problem.

theorem equal_water_and_alcohol (h : m > 0) :
  (a * (m / (m + a)) = a * (m / (m + a))) :=
by
  sorry

end equal_water_and_alcohol_l941_94121


namespace total_jumps_l941_94157

def taehyung_jumps_per_day : ℕ := 56
def taehyung_days : ℕ := 3
def namjoon_jumps_per_day : ℕ := 35
def namjoon_days : ℕ := 4

theorem total_jumps : taehyung_jumps_per_day * taehyung_days + namjoon_jumps_per_day * namjoon_days = 308 :=
by
  sorry

end total_jumps_l941_94157


namespace car_travel_time_l941_94161

theorem car_travel_time (speed distance : ℝ) (h₁ : speed = 65) (h₂ : distance = 455) :
  distance / speed = 7 :=
by
  -- We will invoke the conditions h₁ and h₂ to conclude the theorem
  sorry

end car_travel_time_l941_94161


namespace frequency_of_a_is_3_l941_94137

def sentence : String := "Happy Teachers'Day!"

def frequency_of_a_in_sentence (s : String) : Nat :=
  s.foldl (λ acc c => if c = 'a' then acc + 1 else acc) 0

theorem frequency_of_a_is_3 : frequency_of_a_in_sentence sentence = 3 :=
  by
    sorry

end frequency_of_a_is_3_l941_94137


namespace tips_earned_l941_94143

theorem tips_earned
  (total_customers : ℕ)
  (no_tip_customers : ℕ)
  (tip_amount : ℕ)
  (tip_customers := total_customers - no_tip_customers)
  (total_tips := tip_customers * tip_amount)
  (h1 : total_customers = 9)
  (h2 : no_tip_customers = 5)
  (h3 : tip_amount = 8) :
  total_tips = 32 := by
  -- Proof goes here
  sorry

end tips_earned_l941_94143


namespace base10_to_base7_conversion_l941_94167

theorem base10_to_base7_conversion :
  ∃ b1 b2 b3 b4 b5 : ℕ, 3 * 7^3 + 1 * 7^2 + 6 * 7^1 + 6 * 7^0 = 3527 ∧ 
  b1 = 1 ∧ b2 = 3 ∧ b3 = 1 ∧ b4 = 6 ∧ b5 = 6 ∧ (3527:ℕ) = (1*7^4 + b1*7^3 + b2*7^2 + b3*7^1 + b4*7^0) := by
sorry

end base10_to_base7_conversion_l941_94167


namespace line_equation_passing_through_point_and_opposite_intercepts_l941_94108

theorem line_equation_passing_through_point_and_opposite_intercepts 
  : ∃ (a b : ℝ), (y = a * x) ∨ (x - y = b) :=
by
  use (3/2), (-1)
  sorry

end line_equation_passing_through_point_and_opposite_intercepts_l941_94108


namespace shopkeeper_percentage_gain_l941_94159

theorem shopkeeper_percentage_gain 
    (original_price : ℝ) 
    (price_increase : ℝ) 
    (first_discount : ℝ) 
    (second_discount : ℝ)
    (new_price : ℝ) 
    (discounted_price1 : ℝ) 
    (final_price : ℝ) 
    (percentage_gain : ℝ) 
    (h1 : original_price = 100)
    (h2 : price_increase = original_price * 0.34)
    (h3 : new_price = original_price + price_increase)
    (h4 : first_discount = new_price * 0.10)
    (h5 : discounted_price1 = new_price - first_discount)
    (h6 : second_discount = discounted_price1 * 0.15)
    (h7 : final_price = discounted_price1 - second_discount)
    (h8 : percentage_gain = ((final_price - original_price) / original_price) * 100) :
    percentage_gain = 2.51 :=
by sorry

end shopkeeper_percentage_gain_l941_94159


namespace library_charge_l941_94147

-- Definitions according to given conditions
def daily_charge : ℝ := 0.5
def days_in_may : ℕ := 31
def days_borrowed1 : ℕ := 20
def days_borrowed2 : ℕ := 31

-- Calculation of total charge
theorem library_charge :
  let total_charge := (daily_charge * days_borrowed1) + (2 * daily_charge * days_borrowed2)
  total_charge = 41 :=
by
  sorry

end library_charge_l941_94147


namespace initial_masses_l941_94175

def area_of_base : ℝ := 15
def density_water : ℝ := 1
def density_ice : ℝ := 0.92
def change_in_water_level : ℝ := 5
def final_height_of_water : ℝ := 115

theorem initial_masses (m_ice m_water : ℝ) :
  m_ice = 675 ∧ m_water = 1050 :=
by
  -- Calculate the change in volume of water
  let delta_v := area_of_base * change_in_water_level

  -- Relate this volume change to the volume difference between ice and water
  let lhs := m_ice / density_ice - m_ice / density_water
  let eq1 := delta_v

  -- Solve for the mass of ice
  have h_ice : m_ice = 675 := 
  sorry

  -- Determine the final volume of water
  let final_volume_of_water := final_height_of_water * area_of_base

  -- Determine the initial mass of water
  let mass_of_water_total := density_water * final_volume_of_water
  let initial_mass_of_water :=
    mass_of_water_total - m_ice

  have h_water : m_water = 1050 := 
  sorry

  exact ⟨h_ice, h_water⟩

end initial_masses_l941_94175


namespace algebra_problem_l941_94109

-- Definition of variable y
variable (y : ℝ)

-- Given the condition
axiom h : 2 * y^2 + 3 * y + 7 = 8

-- We need to prove that 4 * y^2 + 6 * y - 9 = -7 given the condition
theorem algebra_problem : 4 * y^2 + 6 * y - 9 = -7 :=
by sorry

end algebra_problem_l941_94109


namespace four_genuine_coin_probability_l941_94150

noncomputable def probability_all_genuine_given_equal_weight : ℚ :=
  let total_coins := 20
  let genuine_coins := 12
  let counterfeit_coins := 8

  -- Calculate the probability of selecting two genuine coins from total coins
  let prob_first_pair_genuine := (genuine_coins / total_coins) * 
                                    ((genuine_coins - 1) / (total_coins - 1))

  -- Updating remaining counts after selecting the first pair
  let remaining_genuine_coins := genuine_coins - 2
  let remaining_total_coins := total_coins - 2

  -- Calculate the probability of selecting another two genuine coins
  let prob_second_pair_genuine := (remaining_genuine_coins / remaining_total_coins) * 
                                    ((remaining_genuine_coins - 1) / (remaining_total_coins - 1))

  -- Probability of A ∩ B
  let prob_A_inter_B := prob_first_pair_genuine * prob_second_pair_genuine

  -- Assuming prob_B represents the weighted probabilities including complexities
  let prob_B := (110 / 1077) -- This is an estimated combined probability for the purpose of this definition

  -- Conditional probability P(A | B)
  prob_A_inter_B / prob_B

theorem four_genuine_coin_probability :
  probability_all_genuine_given_equal_weight = 110 / 1077 := sorry

end four_genuine_coin_probability_l941_94150


namespace repeating_decimal_as_fraction_l941_94144

-- Define the repeating decimal
def repeating_decimal := 3 + (127 / 999)

-- State the goal
theorem repeating_decimal_as_fraction : repeating_decimal = (3124 / 999) := 
by 
  sorry

end repeating_decimal_as_fraction_l941_94144


namespace heads_at_least_once_in_three_tosses_l941_94106

theorem heads_at_least_once_in_three_tosses :
  let total_outcomes := 8
  let all_tails_outcome := 1
  (1 - (all_tails_outcome / total_outcomes) = (7 / 8)) :=
by
  let total_outcomes := 8
  let all_tails_outcome := 1
  sorry

end heads_at_least_once_in_three_tosses_l941_94106


namespace cloud_ratio_l941_94104

theorem cloud_ratio (D Carson Total : ℕ) (h1 : Carson = 6) (h2 : Total = 24) (h3 : Carson + D = Total) :
  (D / Carson) = 3 := by
  sorry

end cloud_ratio_l941_94104


namespace symmetric_point_of_A_is_correct_l941_94142

def symmetric_point_with_respect_to_x_axis (A : ℝ × ℝ) : ℝ × ℝ :=
  (A.1, -A.2)

theorem symmetric_point_of_A_is_correct :
  symmetric_point_with_respect_to_x_axis (3, 4) = (3, -4) :=
by
  sorry

end symmetric_point_of_A_is_correct_l941_94142


namespace prob_KH_then_Ace_l941_94155

noncomputable def probability_KH_then_Ace_drawn_in_sequence : ℚ :=
  let prob_first_card_is_KH := 1 / 52
  let prob_second_card_is_Ace := 4 / 51
  prob_first_card_is_KH * prob_second_card_is_Ace

theorem prob_KH_then_Ace : probability_KH_then_Ace_drawn_in_sequence = 1 / 663 := by
  sorry

end prob_KH_then_Ace_l941_94155


namespace mary_sheep_purchase_l941_94117

theorem mary_sheep_purchase: 
  ∀ (mary_sheep bob_sheep add_sheep : ℕ), 
    mary_sheep = 300 → 
    bob_sheep = 2 * mary_sheep + 35 → 
    add_sheep = (bob_sheep - 69) - mary_sheep → 
    add_sheep = 266 :=
by
  intros mary_sheep bob_sheep add_sheep _ _
  sorry

end mary_sheep_purchase_l941_94117


namespace side_lengths_sum_eq_225_l941_94141

noncomputable def GX (x y z : ℝ) : ℝ :=
  (1 / 3) * (x + y + z) - x

noncomputable def GY (x y z : ℝ) : ℝ :=
  (1 / 3) * (x + y + z) - y

noncomputable def GZ (x y z : ℝ) : ℝ :=
  (1 / 3) * (x + y + z) - z

theorem side_lengths_sum_eq_225
  (x y z : ℝ)
  (h : GX x y z ^ 2 + GY x y z ^ 2 + GZ x y z ^ 2 = 75) :
  (x - y) ^ 2 + (x - z) ^ 2 + (y - z) ^ 2 = 225 := by {
  sorry
}

end side_lengths_sum_eq_225_l941_94141


namespace max_snacks_l941_94169

-- Define the conditions and the main statement we want to prove

def single_snack_cost : ℕ := 2
def four_snack_pack_cost : ℕ := 6
def six_snack_pack_cost : ℕ := 8
def budget : ℕ := 20

def max_snacks_purchased : ℕ := 14

theorem max_snacks (h1 : single_snack_cost = 2) 
                   (h2 : four_snack_pack_cost = 6) 
                   (h3 : six_snack_pack_cost = 8) 
                   (h4 : budget = 20) : 
                   max_snacks_purchased = 14 := 
by {
  sorry
}

end max_snacks_l941_94169


namespace pages_needed_l941_94140

def new_cards : ℕ := 2
def old_cards : ℕ := 10
def cards_per_page : ℕ := 3
def total_cards : ℕ := new_cards + old_cards

theorem pages_needed : total_cards / cards_per_page = 4 := by
  sorry

end pages_needed_l941_94140


namespace solve_linear_system_l941_94196

/-- Let x₁, x₂, x₃, x₄, x₅, x₆, x₇, x₈ be real numbers that satisfy the following system of equations:
1. x₁ + x₂ + x₃ = 6
2. x₂ + x₃ + x₄ = 9
3. x₃ + x₄ + x₅ = 3
4. x₄ + x₅ + x₆ = -3
5. x₅ + x₆ + x₇ = -9
6. x₆ + x₇ + x₈ = -6
7. x₇ + x₈ + x₁ = -2
8. x₈ + x₁ + x₂ = 2
Prove that the solution is
  x₁ = 1, x₂ = 2, x₃ = 3, x₄ = 4, x₅ = -4, x₆ = -3, x₇ = -2, x₈ = -1
-/
theorem solve_linear_system :
  ∀ (x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ : ℝ),
  x₁ + x₂ + x₃ = 6 →
  x₂ + x₃ + x₄ = 9 →
  x₃ + x₄ + x₅ = 3 →
  x₄ + x₅ + x₆ = -3 →
  x₅ + x₆ + x₇ = -9 →
  x₆ + x₇ + x₈ = -6 →
  x₇ + x₈ + x₁ = -2 →
  x₈ + x₁ + x₂ = 2 →
  x₁ = 1 ∧ x₂ = 2 ∧ x₃ = 3 ∧ x₄ = 4 ∧ x₅ = -4 ∧ x₆ = -3 ∧ x₇ = -2 ∧ x₈ = -1 :=
by
  intros x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ h1 h2 h3 h4 h5 h6 h7 h8
  -- Here, the proof steps would go
  sorry

end solve_linear_system_l941_94196


namespace polygon_sides_l941_94152

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 1980) : n = 13 := 
by sorry

end polygon_sides_l941_94152


namespace ratio_transformation_l941_94193

theorem ratio_transformation (x1 y1 x2 y2 : ℚ) (h₁ : x1 / y1 = 7 / 5) (h₂ : x2 = x1 * y1) (h₃ : y2 = y1 * x1) : x2 / y2 = 1 := by
  sorry

end ratio_transformation_l941_94193


namespace remainder_mod_5_l941_94158

theorem remainder_mod_5 :
  let a := 1492
  let b := 1776
  let c := 1812
  let d := 1996
  (a * b * c * d) % 5 = 4 :=
by
  sorry

end remainder_mod_5_l941_94158


namespace unique_ab_for_interval_condition_l941_94168

theorem unique_ab_for_interval_condition : 
  ∃! (a b : ℝ), (∀ x, (0 ≤ x ∧ x ≤ 1) → |x^2 - a * x - b| ≤ 1 / 8) ∧ a = 1 ∧ b = -1 / 8 := by
  sorry

end unique_ab_for_interval_condition_l941_94168


namespace abc_perfect_ratio_l941_94136

theorem abc_perfect_ratio {a b c : ℚ} (h1 : ∃ t : ℤ, a + b + c = t ∧ a^2 + b^2 + c^2 = t) :
  ∃ (p q : ℤ), (abc = p^3 / q^2) ∧ (IsCoprime p q) := 
sorry

end abc_perfect_ratio_l941_94136


namespace max_contribution_l941_94166

theorem max_contribution (n : ℕ) (total : ℝ) (min_contribution : ℝ)
  (h1 : n = 12) (h2 : total = 20) (h3 : min_contribution = 1)
  (h4 : ∀ i : ℕ, i < n → min_contribution ≤ min_contribution) :
  ∃ max_contrib : ℝ, max_contrib = 9 :=
by
  sorry

end max_contribution_l941_94166


namespace calculate_value_l941_94179

theorem calculate_value (x y : ℝ) (h : 2 * x + y = 6) : 
    ((x - y)^2 - (x + y)^2 + y * (2 * x - y)) / (-2 * y) = 3 :=
by 
  sorry

end calculate_value_l941_94179


namespace factorial_units_digit_l941_94122

theorem factorial_units_digit (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (hba : a < b) : 
  ¬ (∃ k : ℕ, (b! - a!) % 10 = 7) := 
sorry

end factorial_units_digit_l941_94122


namespace square_area_l941_94164

theorem square_area (x : ℝ) (A B C D E F : ℝ)
  (h1 : E = x / 3)
  (h2 : F = (2 * x) / 3)
  (h3 : abs (B - E) = 40)
  (h4 : abs (E - F) = 40)
  (h5 : abs (F - D) = 40) :
  x^2 = 2880 :=
by
  -- Main proof here
  sorry

end square_area_l941_94164


namespace find_m_l941_94171

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {1, 2, m}
def B : Set ℝ := {3, 4}

-- The intersection condition
def intersect_condition (m : ℝ) : Prop := A m ∩ B = {3}

-- The statement to prove
theorem find_m : ∃ m : ℝ, intersect_condition m → m = 3 :=
by {
  use 3,
  sorry
}

end find_m_l941_94171


namespace problem1_problem2_problem3_problem4_l941_94148

-- Problem 1
theorem problem1 :
  -11 - (-8) + (-13) + 12 = -4 :=
  sorry

-- Problem 2
theorem problem2 :
  3 + 1 / 4 + (- (2 + 3 / 5)) + (5 + 3 / 4) - (8 + 2 / 5) = -2 :=
  sorry

-- Problem 3
theorem problem3 :
  -36 * (5 / 6 - 4 / 9 + 11 / 12) = -47 :=
  sorry

-- Problem 4
theorem problem4 :
  12 * (-1 / 6) + 27 / abs (3 ^ 2) + (-2) ^ 3 = -7 :=
  sorry

end problem1_problem2_problem3_problem4_l941_94148


namespace smallest_constant_l941_94174

theorem smallest_constant (a b c : ℝ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
  (a^2 + b^2 + a * b) / c^2 ≥ 3 / 4 :=
sorry

end smallest_constant_l941_94174


namespace speed_difference_valid_l941_94149

-- Definitions of the conditions
def speed (s : ℕ) : ℕ := s^2 + 2 * s

-- Theorem statement that needs to be proven
theorem speed_difference_valid : 
  (speed 5 - speed 3) = 20 :=
  sorry

end speed_difference_valid_l941_94149


namespace joel_age_when_dad_is_twice_l941_94133

-- Given Conditions
def joel_age_now : ℕ := 5
def dad_age_now : ℕ := 32
def age_difference : ℕ := dad_age_now - joel_age_now

-- Proof Problem Statement
theorem joel_age_when_dad_is_twice (x : ℕ) (hx : dad_age_now - joel_age_now = 27) : x = 27 :=
by
  sorry

end joel_age_when_dad_is_twice_l941_94133


namespace exists_n_ge_1_le_2020_l941_94184

theorem exists_n_ge_1_le_2020
  (a : ℕ → ℕ)
  (h_distinct : ∀ i j : ℕ, 1 ≤ i → i ≤ 2020 → 1 ≤ j → j ≤ 2020 → i ≠ j → a i ≠ a j)
  (h_periodic1 : a 2021 = a 1)
  (h_periodic2 : a 2022 = a 2) :
  ∃ n : ℕ, 1 ≤ n ∧ n ≤ 2020 ∧ a n ^ 2 + a (n + 1) ^ 2 ≥ a (n + 2) ^ 2 + n ^ 2 + 3 := 
sorry

end exists_n_ge_1_le_2020_l941_94184


namespace perpendicular_line_through_center_l941_94123

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop := x^2 + 2 * x + y^2 - 3 = 0

-- Define the equation of the line
def line_eq (x y : ℝ) : Prop := x + y - 1 = 0

-- Define the line we want to prove passes through the center of the circle and is perpendicular to the given line
def wanted_line_eq (x y : ℝ) : Prop := x - y + 1 = 0

-- Main statement: Prove that the line that passes through the center of the circle and is perpendicular to the given line has the equation x - y + 1 = 0
theorem perpendicular_line_through_center (x y : ℝ) :
  (circle_eq (-1) 0) ∧ (line_eq x y) → wanted_line_eq x y :=
by
  sorry

end perpendicular_line_through_center_l941_94123


namespace total_meat_supply_l941_94102

-- Definitions of the given conditions
def lion_consumption_per_day : ℕ := 25
def tiger_consumption_per_day : ℕ := 20
def duration_days : ℕ := 2

-- Statement of the proof problem
theorem total_meat_supply :
  (lion_consumption_per_day + tiger_consumption_per_day) * duration_days = 90 :=
by
  sorry

end total_meat_supply_l941_94102


namespace translated_quadratic_range_of_translated_quadratic_min_value_on_interval_l941_94160

def f_translation : ℝ → ℝ :=
  fun x => (x - 1)^2 - 2

def f_quad (a x : ℝ) : ℝ :=
  x^2 - 2*a*x - 1

theorem translated_quadratic :
  ∀ x, f_translation x = (x - 1)^2 - 2 :=
by
  intro x
  simp [f_translation]

theorem range_of_translated_quadratic :
  ∀ x, 0 ≤ x ∧ x ≤ 4 → -2 ≤ f_translation x ∧ f_translation x ≤ 7 :=
by
  sorry

theorem min_value_on_interval :
  ∀ a, 
    (a ≤ 0 → ∀ x, 0 ≤ x ∧ x ≤ 2 → (f_quad a x ≥ -1)) ∧
    (0 < a ∧ a < 2 → f_quad a a = -a^2 - 1) ∧
    (a ≥ 2 → ∀ x, 0 ≤ x ∧ x ≤ 2 → (f_quad a 2 = -4*a + 3)) :=
by
  sorry

end translated_quadratic_range_of_translated_quadratic_min_value_on_interval_l941_94160


namespace solve_for_A_l941_94177

def f (A B x : ℝ) : ℝ := A * x ^ 2 - 3 * B ^ 3
def g (B x : ℝ) : ℝ := 2 * B * x + B ^ 2

theorem solve_for_A (B : ℝ) (hB : B ≠ 0) (h : f A B (g B 2) = 0) :
  A = 3 / (16 / B + 8 + B ^ 3) :=
by
  sorry

end solve_for_A_l941_94177


namespace geom_seq_arith_seq_l941_94163

variable (a : ℕ → ℝ) (q : ℝ)

noncomputable def isGeomSeq (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = q * a n

theorem geom_seq_arith_seq (h1 : ∀ n, 0 < a n) 
  (h2 : isGeomSeq a q)
  (h3 : 2 * (1 / 2 * a 5) = a 3 + a 4)
  : (a 3 + a 5) / (a 4 + a 6) = (Real.sqrt 5 - 1) / 2 := 
sorry

end geom_seq_arith_seq_l941_94163


namespace abs_x_minus_y_l941_94162

theorem abs_x_minus_y (x y : ℝ) (h₁ : x^3 + y^3 = 26) (h₂ : xy * (x + y) = -6) : |x - y| = 4 :=
by
  sorry

end abs_x_minus_y_l941_94162


namespace mass_percentage_Al_in_Al2O3_l941_94199

-- Define the atomic masses and formula unit
def atomic_mass_Al : ℝ := 26.98
def atomic_mass_O : ℝ := 16.00
def molar_mass_Al2O3 : ℝ := (2 * atomic_mass_Al) + (3 * atomic_mass_O)
def mass_Al_in_Al2O3 : ℝ := 2 * atomic_mass_Al

-- Define the statement for the mass percentage of Al in Al2O3
theorem mass_percentage_Al_in_Al2O3 : (mass_Al_in_Al2O3 / molar_mass_Al2O3) * 100 = 52.91 :=
by
  sorry -- Proof to be filled in

end mass_percentage_Al_in_Al2O3_l941_94199


namespace sum_of_digits_correct_l941_94100

theorem sum_of_digits_correct :
  ∃ a b c : ℕ,
    (1 + 7 + 3 + a) % 9 = 0 ∧
    (1 + 3 - (7 + b)) % 11 = 0 ∧
    (c % 2 = 0) ∧
    ((1 + 7 + 3 + c) % 3 = 0) ∧
    (a + b + c = 19) :=
sorry

end sum_of_digits_correct_l941_94100


namespace average_of_three_quantities_l941_94131

theorem average_of_three_quantities 
  (five_avg : ℚ) (three_avg : ℚ) (two_avg : ℚ) 
  (h_five_avg : five_avg = 10) 
  (h_two_avg : two_avg = 19) : 
  three_avg = 4 := 
by 
  let sum_5 := 5 * 10
  let sum_2 := 2 * 19
  let sum_3 := sum_5 - sum_2
  let three_avg := sum_3 / 3
  sorry

end average_of_three_quantities_l941_94131


namespace CarlYardAreaIsCorrect_l941_94128

noncomputable def CarlRectangularYardArea (post_count : ℕ) (distance_between_posts : ℕ) (long_side_factor : ℕ) :=
  let x := post_count / (2 * (1 + long_side_factor))
  let short_side := (x - 1) * distance_between_posts
  let long_side := (long_side_factor * x - 1) * distance_between_posts
  short_side * long_side

theorem CarlYardAreaIsCorrect :
  CarlRectangularYardArea 24 5 3 = 825 := 
by
  -- calculation steps if needed or
  sorry

end CarlYardAreaIsCorrect_l941_94128


namespace stratified_sampling_seniors_l941_94119

theorem stratified_sampling_seniors
  (total_students : ℕ)
  (seniors : ℕ)
  (sample_size : ℕ)
  (senior_sample_size : ℕ)
  (h1 : total_students = 4500)
  (h2 : seniors = 1500)
  (h3 : sample_size = 300)
  (h4 : senior_sample_size = seniors * sample_size / total_students) :
  senior_sample_size = 100 :=
  sorry

end stratified_sampling_seniors_l941_94119


namespace find_f_neg_6_l941_94127

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  if x ≥ 0 then Real.log (x + 2) / Real.log 2 + (a - 1) * x + b else -(Real.log (-x + 2) / Real.log 2 + (a - 1) * -x + b)

theorem find_f_neg_6 (a b : ℝ) (h1 : ∀ x : ℝ, f x a b = -f (-x) a b) 
                     (h2 : ∀ x : ℝ, x ≥ 0 → f x a b = Real.log (x + 2) / Real.log 2 + (a - 1) * x + b)
                     (h3 : f 2 a b = -1) : f (-6) 0 (-1) = 4 :=
by
  sorry

end find_f_neg_6_l941_94127


namespace xiao_wang_fourth_place_l941_94135

section Competition
  -- Define the participants and positions
  inductive Participant
  | XiaoWang : Participant
  | XiaoZhang : Participant
  | XiaoZhao : Participant
  | XiaoLi : Participant

  inductive Position
  | First : Position
  | Second : Position
  | Third : Position
  | Fourth : Position

  open Participant Position

  -- Conditions given in the problem
  variables
    (place : Participant → Position)
    (hA1 : place XiaoWang = First → place XiaoZhang = Third)
    (hA2 : place XiaoWang = First → place XiaoZhang ≠ Third)
    (hB1 : place XiaoLi = First → place XiaoZhao = Fourth)
    (hB2 : place XiaoLi = First → place XiaoZhao ≠ Fourth)
    (hC1 : place XiaoZhao = Second → place XiaoWang = Third)
    (hC2 : place XiaoZhao = Second → place XiaoWang ≠ Third)
    (no_ties : ∀ x y, place x = place y → x = y)
    (half_correct : ∀ p, (p = A → ((place XiaoWang = First ∨ place XiaoZhang = Third) ∧ (place XiaoWang ≠ First ∨ place XiaoZhang ≠ Third)))
                          ∧ (p = B → ((place XiaoLi = First ∨ place XiaoZhao = Fourth) ∧ (place XiaoLi ≠ First ∨ place XiaoZhao ≠ Fourth)))
                          ∧ (p = C → ((place XiaoZhao = Second ∨ place XiaoWang = Third) ∧ (place XiaoZhao ≠ Second ∨ place XiaoWang ≠ Third)))) 

  -- The goal to prove
  theorem xiao_wang_fourth_place : place XiaoWang = Fourth :=
  sorry
end Competition

end xiao_wang_fourth_place_l941_94135
