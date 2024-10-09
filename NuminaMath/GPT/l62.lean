import Mathlib

namespace xy_condition_l62_6216

theorem xy_condition (x y z : ℝ) (hxz : x ≠ z) (hxy : x ≠ y) (hyz : y ≠ z) (posx : 0 < x) (posy : 0 < y) (posz : 0 < z) 
  (h : y / (x - z) = (x + y) / z ∧ (x + y) / z = x / y) : x / y = 2 :=
by
  sorry

end xy_condition_l62_6216


namespace picture_distance_l62_6239

theorem picture_distance (w t s p d : ℕ) (h1 : w = 25) (h2 : t = 2) (h3 : s = 1) (h4 : 2 * p + s = t + s + t) 
  (h5 : w = 2 * d + p) : d = 10 :=
by
  sorry

end picture_distance_l62_6239


namespace inequality_transformation_l62_6279

theorem inequality_transformation (m n : ℝ) (h : -m / 2 < -n / 6) : 3 * m > n := by
  sorry

end inequality_transformation_l62_6279


namespace balance_of_diamondsuits_and_bullets_l62_6292

variable (a b c : ℕ)

theorem balance_of_diamondsuits_and_bullets 
  (h1 : 4 * a + 2 * b = 12 * c)
  (h2 : a = b + 3 * c) :
  3 * b = 6 * c := 
sorry

end balance_of_diamondsuits_and_bullets_l62_6292


namespace initial_distance_l62_6288

-- Definitions based on conditions
def speed_thief : ℝ := 8 -- in km/hr
def speed_policeman : ℝ := 10 -- in km/hr
def distance_thief_runs : ℝ := 0.7 -- in km

-- Theorem statement
theorem initial_distance
  (relative_speed := speed_policeman - speed_thief) -- Relative speed (in km/hr)
  (time_to_overtake := distance_thief_runs / relative_speed) -- Time for the policeman to overtake the thief (in hours)
  (initial_distance := speed_policeman * time_to_overtake) -- Initial distance (in km)
  : initial_distance = 3.5 :=
by
  sorry

end initial_distance_l62_6288


namespace area_of_triangle_example_l62_6282

noncomputable def area_of_triangle (A B C : (ℝ × ℝ)) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem area_of_triangle_example : 
  area_of_triangle (3, 3) (3, 10) (12, 19) = 31.5 :=
by
  sorry

end area_of_triangle_example_l62_6282


namespace train_crosses_man_in_6_seconds_l62_6218

/-- A train of length 240 meters, traveling at a speed of 144 km/h, will take 6 seconds to cross a man standing on the platform. -/
theorem train_crosses_man_in_6_seconds
  (length_of_train : ℕ)
  (speed_of_train : ℕ)
  (conversion_factor : ℕ)
  (speed_in_m_per_s : ℕ)
  (time_to_cross : ℕ)
  (h1 : length_of_train = 240)
  (h2 : speed_of_train = 144)
  (h3 : conversion_factor = 1000 / 3600)
  (h4 : speed_in_m_per_s = speed_of_train * conversion_factor)
  (h5 : speed_in_m_per_s = 40)
  (h6 : time_to_cross = length_of_train / speed_in_m_per_s) :
  time_to_cross = 6 := by
  sorry

end train_crosses_man_in_6_seconds_l62_6218


namespace problem_solution_l62_6245

theorem problem_solution :
  let m := 9
  let n := 20
  let lhs := (2 + 4 + 6 + 8) / (1 + 3 + 5 + 7) - (1 + 3 + 5 + 7) / (2 + 4 + 6 + 8)
  let rhs := 9 / 20
  lhs = rhs → 10 * m + n = 110 :=
by sorry

end problem_solution_l62_6245


namespace fish_price_relation_l62_6217

variables (b_c m_c b_v m_v : ℝ)

axiom cond1 : 3 * b_c + m_c = 5 * b_v
axiom cond2 : 2 * b_c + m_c = 3 * b_v + m_v

theorem fish_price_relation : 5 * m_v = b_c + 2 * m_c :=
by
  sorry

end fish_price_relation_l62_6217


namespace canoes_rented_more_than_kayaks_l62_6272

-- Defining the constants
def canoe_cost : ℕ := 11
def kayak_cost : ℕ := 16
def total_revenue : ℕ := 460
def canoe_ratio : ℕ := 4
def kayak_ratio : ℕ := 3

-- Main statement to prove
theorem canoes_rented_more_than_kayaks :
  ∃ (C K : ℕ), canoe_cost * C + kayak_cost * K = total_revenue ∧ (canoe_ratio * K = kayak_ratio * C) ∧ (C - K = 5) :=
by
  have h1 : canoe_cost = 11 := rfl
  have h2 : kayak_cost = 16 := rfl
  have h3 : total_revenue = 460 := rfl
  have h4 : canoe_ratio = 4 := rfl
  have h5 : kayak_ratio = 3 := rfl
  sorry

end canoes_rented_more_than_kayaks_l62_6272


namespace find_quotient_l62_6236

def dividend : ℝ := 13787
def remainder : ℝ := 14
def divisor : ℝ := 154.75280898876406
def quotient : ℝ := 89

theorem find_quotient :
  (dividend - remainder) / divisor = quotient :=
sorry

end find_quotient_l62_6236


namespace wall_with_5_peaks_has_14_cubes_wall_with_2014_peaks_has_6041_cubes_painted_area_wall_with_2014_peaks_l62_6283

noncomputable def number_of_cubes (n : ℕ) : ℕ :=
  n + (n - 1) + n

noncomputable def painted_area (n : ℕ) : ℕ :=
  (5 * n) + (3 * (n + 1)) + (2 * (n - 2))

theorem wall_with_5_peaks_has_14_cubes : number_of_cubes 5 = 14 :=
  by sorry

theorem wall_with_2014_peaks_has_6041_cubes : number_of_cubes 2014 = 6041 :=
  by sorry

theorem painted_area_wall_with_2014_peaks : painted_area 2014 = 20139 :=
  by sorry

end wall_with_5_peaks_has_14_cubes_wall_with_2014_peaks_has_6041_cubes_painted_area_wall_with_2014_peaks_l62_6283


namespace sum_eq_prod_nat_numbers_l62_6258

theorem sum_eq_prod_nat_numbers (A B C D E F : ℕ) :
  A + B + C + D + E + F = A * B * C * D * E * F →
  (A = 0 ∧ B = 0 ∧ C = 0 ∧ D = 0 ∧ E = 0 ∧ F = 0) ∨
  (A = 1 ∧ B = 1 ∧ C = 1 ∧ D = 1 ∧ E = 2 ∧ F = 6) :=
by
  sorry

end sum_eq_prod_nat_numbers_l62_6258


namespace winning_strategy_for_B_l62_6243

theorem winning_strategy_for_B (N : ℕ) (h : N < 15) : N = 7 ↔ (∃ strategy : (Fin 6 → ℕ) → ℕ, ∀ f : Fin 6 → ℕ, (strategy f) % 1001 = 0) :=
by
  sorry

end winning_strategy_for_B_l62_6243


namespace cost_per_unit_range_of_type_A_purchases_maximum_profit_l62_6249

-- Definitions of the problem conditions
def cost_type_A : ℕ := 15
def cost_type_B : ℕ := 20

def profit_type_A : ℕ := 3
def profit_type_B : ℕ := 4

def budget_min : ℕ := 2750
def budget_max : ℕ := 2850

def total_units : ℕ := 150
def profit_min : ℕ := 565

-- Main proof statements as Lean theorems
theorem cost_per_unit : 
  ∃ (x y : ℕ), 
    2 * x + 3 * y = 90 ∧ 
    3 * x + y = 65 ∧ 
    x = cost_type_A ∧ 
    y = cost_type_B := 
sorry

theorem range_of_type_A_purchases : 
  ∃ (a : ℕ), 
    30 ≤ a ∧ 
    a ≤ 50 ∧ 
    budget_min ≤ cost_type_A * a + cost_type_B * (total_units - a) ∧ 
    cost_type_A * a + cost_type_B * (total_units - a) ≤ budget_max := 
sorry

theorem maximum_profit : 
  ∃ (a : ℕ), 
    30 ≤ a ∧ 
    a ≤ 35 ∧ 
    profit_min ≤ profit_type_A * a + profit_type_B * (total_units - a) ∧ 
    ¬∃ (b : ℕ), 
      30 ≤ b ∧ 
      b ≤ 35 ∧ 
      b ≠ a ∧ 
      profit_type_A * b + profit_type_B * (total_units - b) > profit_type_A * a + profit_type_B * (total_units - a) :=
sorry

end cost_per_unit_range_of_type_A_purchases_maximum_profit_l62_6249


namespace rodney_lift_l62_6274

theorem rodney_lift :
  ∃ (Ry : ℕ), 
  (∃ (Re R Ro : ℕ), 
  Re + Ry + R + Ro = 450 ∧
  Ry = 2 * R ∧
  R = Ro + 5 ∧
  Re = 3 * Ro - 20 ∧
  20 ≤ Ry ∧ Ry ≤ 200 ∧
  20 ≤ R ∧ R ≤ 200 ∧
  20 ≤ Ro ∧ Ro ≤ 200 ∧
  20 ≤ Re ∧ Re ≤ 200) ∧
  Ry = 140 :=
by
  sorry

end rodney_lift_l62_6274


namespace find_a_9_l62_6276

variable {a : ℕ → ℤ}
variable {S : ℕ → ℤ}
variable (d : ℤ)

-- Assumptions and definitions from the problem
def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop := ∀ n : ℕ, a (n + 1) = a n + d
def sum_of_arithmetic_sequence (S : ℕ → ℤ) (a : ℕ → ℤ) : Prop := ∀ n : ℕ, S n = n * (a 1 + a n) / 2
def condition_one (a : ℕ → ℤ) : Prop := (a 1) + (a 2)^2 = -3
def condition_two (S : ℕ → ℤ) : Prop := S 5 = 10

-- Main theorem statement
theorem find_a_9 (h_arithmetic : arithmetic_sequence a d)
                 (h_sum : sum_of_arithmetic_sequence S a)
                 (h_cond1 : condition_one a)
                 (h_cond2 : condition_two S) : a 9 = 20 := 
sorry

end find_a_9_l62_6276


namespace total_marks_more_than_physics_l62_6201

variable (P C M : ℕ)

theorem total_marks_more_than_physics :
  (P + C + M > P) ∧ ((C + M) / 2 = 75) → (P + C + M) - P = 150 := by
  intros h
  sorry

end total_marks_more_than_physics_l62_6201


namespace boat_speed_in_still_water_l62_6290

theorem boat_speed_in_still_water (b : ℝ) (h : (36 / (b - 2)) - (36 / (b + 2)) = 1.5) : b = 10 :=
by
  sorry

end boat_speed_in_still_water_l62_6290


namespace sum_of_real_values_l62_6269

theorem sum_of_real_values (x : ℝ) (h : |3 * x - 15| + |x - 5| = 92) : (x = 28 ∨ x = -18) → x + 10 = 0 := by
  sorry

end sum_of_real_values_l62_6269


namespace fraction_spent_on_food_l62_6212

variable (salary : ℝ) (food_fraction rent_fraction clothes_fraction remaining_amount : ℝ)
variable (salary_condition : salary = 180000)
variable (rent_fraction_condition : rent_fraction = 1/10)
variable (clothes_fraction_condition : clothes_fraction = 3/5)
variable (remaining_amount_condition : remaining_amount = 18000)

theorem fraction_spent_on_food :
  rent_fraction * salary + clothes_fraction * salary + food_fraction * salary + remaining_amount = salary →
  food_fraction = 1/5 :=
by
  intros
  sorry

end fraction_spent_on_food_l62_6212


namespace smallest_x_multiple_of_53_l62_6246

theorem smallest_x_multiple_of_53 :
  ∃ x : ℕ, (3 * x + 41) % 53 = 0 ∧ x > 0 ∧ x = 4 :=
by 
  sorry

end smallest_x_multiple_of_53_l62_6246


namespace son_completion_time_l62_6286

theorem son_completion_time (M S F : ℝ) 
  (h1 : M = 1 / 10) 
  (h2 : M + S = 1 / 5) 
  (h3 : S + F = 1 / 4) : 
  1 / S = 10 := 
  sorry

end son_completion_time_l62_6286


namespace M_inter_N_eq_singleton_l62_6210

def M (x y : ℝ) : Prop := x + y = 2
def N (x y : ℝ) : Prop := x - y = 4

theorem M_inter_N_eq_singleton :
  {p : ℝ × ℝ | M p.1 p.2} ∩ {p : ℝ × ℝ | N p.1 p.2} = { (3, -1) } :=
by
  sorry

end M_inter_N_eq_singleton_l62_6210


namespace arrow_in_48th_position_l62_6231

def arrow_sequence : List (String) := ["→", "↑", "↓", "←", "↘"]

theorem arrow_in_48th_position :
  arrow_sequence.get? ((48 % 5) - 1) = some "↓" :=
by
  norm_num
  sorry

end arrow_in_48th_position_l62_6231


namespace inequality_f_lt_g_range_of_a_l62_6219

def f (x : ℝ) : ℝ := |x - 4|
def g (x : ℝ) : ℝ := |2 * x + 1|

theorem inequality_f_lt_g :
  ∀ x : ℝ, f x = |x - 4| ∧ g x = |2 * x + 1| →
  (f x < g x ↔ (x < -5 ∨ x > 1)) :=
by
   sorry

theorem range_of_a :
  ∀ x a : ℝ, f x = |x - 4| ∧ g x = |2 * x + 1| →
  (2 * f x + g x > a * x) →
  (-4 ≤ a ∧ a < 9/4) :=
by
   sorry

end inequality_f_lt_g_range_of_a_l62_6219


namespace line_equation_l62_6226

theorem line_equation (x y : ℝ) (hx : ∃ t : ℝ, t ≠ 0 ∧ x = t * -3) (hy : ∃ t : ℝ, t ≠ 0 ∧ y = t * 4) :
  4 * x - 3 * y + 12 = 0 := 
sorry

end line_equation_l62_6226


namespace y_pow_x_eq_nine_l62_6225

theorem y_pow_x_eq_nine (x y : ℝ) (h : x^2 + y^2 - 4 * x + 6 * y + 13 = 0) : y^x = 9 := by
  sorry

end y_pow_x_eq_nine_l62_6225


namespace least_number_to_make_divisible_by_3_l62_6265

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem least_number_to_make_divisible_by_3 : ∃ k : ℕ, (∃ n : ℕ, 
  sum_of_digits 625573 ≡ 28 [MOD 3] ∧ 
  (625573 + k) % 3 = 0 ∧ 
  k = 2) :=
by
  sorry

end least_number_to_make_divisible_by_3_l62_6265


namespace max_hawthorns_satisfying_conditions_l62_6277

theorem max_hawthorns_satisfying_conditions :
  ∃ x : ℕ, 
    x > 100 ∧ 
    x % 3 = 1 ∧ 
    x % 4 = 2 ∧ 
    x % 5 = 3 ∧ 
    x % 6 = 4 ∧ 
    (∀ y : ℕ, 
      y > 100 ∧ 
      y % 3 = 1 ∧ 
      y % 4 = 2 ∧ 
      y % 5 = 3 ∧ 
      y % 6 = 4 → y ≤ 178) :=
sorry

end max_hawthorns_satisfying_conditions_l62_6277


namespace find_coefficients_l62_6289

theorem find_coefficients (A B : ℚ) :
  (∀ x : ℚ, 2 * x + 7 = A * (x + 7) + B * (x - 9)) →
  A = 25 / 16 ∧ B = 7 / 16 :=
by
  intro h
  sorry

end find_coefficients_l62_6289


namespace lcm_gcf_ratio_240_630_l62_6200

theorem lcm_gcf_ratio_240_630 :
  let a := 240
  let b := 630
  Nat.lcm a b / Nat.gcd a b = 168 := by
  sorry

end lcm_gcf_ratio_240_630_l62_6200


namespace larger_number_is_l62_6207

-- Given definitions and conditions
def HCF (a b: ℕ) : ℕ := 23
def other_factor_1 : ℕ := 11
def other_factor_2 : ℕ := 12
def LCM (a b: ℕ) : ℕ := HCF a b * other_factor_1 * other_factor_2

-- Statement to be proven
theorem larger_number_is (a b: ℕ) (h: HCF a b = 23) (hA: a = 23 * 12) (hB: b ∣ a) : a = 276 :=
by { sorry }

end larger_number_is_l62_6207


namespace abs_m_minus_1_greater_eq_abs_m_minus_1_l62_6251

theorem abs_m_minus_1_greater_eq_abs_m_minus_1 (m : ℝ) : |m - 1| ≥ |m| - 1 := 
sorry

end abs_m_minus_1_greater_eq_abs_m_minus_1_l62_6251


namespace oranges_count_l62_6230

def oranges_per_box : ℝ := 10
def boxes_per_day : ℝ := 2650
def total_oranges (x y : ℝ) : ℝ := x * y

theorem oranges_count :
  total_oranges oranges_per_box boxes_per_day = 26500 := 
  by sorry

end oranges_count_l62_6230


namespace calculate_expression_l62_6263

theorem calculate_expression :
  (Int.floor ((15:ℚ)/8 * ((-34:ℚ)/4)) - Int.ceil ((15:ℚ)/8 * Int.floor ((-34:ℚ)/4))) = 0 := 
  by sorry

end calculate_expression_l62_6263


namespace value_of_a_l62_6291

theorem value_of_a (a b c : ℤ) (h1 : a < b) (h2 : b < c) (h3 : (a + b + c) / 3 = 4 * b) (h4 : c / b = 11) : a = 0 :=
by
  sorry

end value_of_a_l62_6291


namespace number_of_selection_plans_l62_6295

-- Definitions based on conditions
def male_students : Nat := 5
def female_students : Nat := 4
def total_volunteers : Nat := 3

def choose (n k : Nat) : Nat :=
  Nat.choose n k

def arrangement_count : Nat :=
  Nat.factorial total_volunteers

-- Theorem that states the total number of selection plans
theorem number_of_selection_plans :
  (choose male_students 2 * choose female_students 1 + choose male_students 1 * choose female_students 2) * arrangement_count = 420 :=
by
  sorry

end number_of_selection_plans_l62_6295


namespace problem_statement_l62_6233

theorem problem_statement (a x m : ℝ) (h₀ : |a| ≤ 1) (h₁ : |x| ≤ 1) :
  (∀ x a, |x^2 - a * x - a^2| ≤ m) ↔ m ≥ 5/4 :=
sorry

end problem_statement_l62_6233


namespace number_of_ordered_pairs_l62_6266

theorem number_of_ordered_pairs : 
  ∃ (S : Finset (ℕ × ℕ)), (∀ x ∈ S, (x.1 * x.2 = 64) ∧ (x.1 > 0) ∧ (x.2 > 0)) ∧ S.card = 7 := 
sorry

end number_of_ordered_pairs_l62_6266


namespace second_term_geometric_sequence_l62_6213

-- Given conditions
def a3 : ℕ := 12
def a4 : ℕ := 18
def q := a4 / a3 -- common ratio

-- Geometric progression definition
noncomputable def a2 := a3 / q

-- Theorem to prove
theorem second_term_geometric_sequence : a2 = 8 :=
by
  -- proof not required
  sorry

end second_term_geometric_sequence_l62_6213


namespace continuity_at_x_0_l62_6275

def f (x : ℝ) := -2 * x^2 + 9
def x_0 : ℝ := 4

theorem continuity_at_x_0 :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - x_0| < δ → |f x - f x_0| < ε :=
by
  sorry

end continuity_at_x_0_l62_6275


namespace find_f2_l62_6247

-- Define the function f(x) = ax + b
def f (a b x : ℝ) : ℝ := a * x + b

-- Condition: f'(x) = a
def f_derivative (a b x : ℝ) : ℝ := a

-- Given conditions
variables (a b : ℝ)
axiom h1 : f a b 1 = 2
axiom h2 : f_derivative a b 1 = 2

theorem find_f2 : f a b 2 = 4 :=
by
  sorry

end find_f2_l62_6247


namespace oranges_for_juice_l62_6208

theorem oranges_for_juice 
  (bags : ℕ) (oranges_per_bag : ℕ) (rotten_oranges : ℕ) (oranges_sold : ℕ)
  (h_bags : bags = 10)
  (h_oranges_per_bag : oranges_per_bag = 30)
  (h_rotten_oranges : rotten_oranges = 50)
  (h_oranges_sold : oranges_sold = 220):
  (bags * oranges_per_bag - rotten_oranges - oranges_sold = 30) :=
by 
  sorry

end oranges_for_juice_l62_6208


namespace correct_expression_l62_6237

theorem correct_expression :
  (2 + Real.sqrt 3 ≠ 2 * Real.sqrt 3) ∧ 
  (Real.sqrt 8 - Real.sqrt 3 ≠ Real.sqrt 5) ∧ 
  (Real.sqrt 3 * Real.sqrt 2 = Real.sqrt 6) ∧ 
  (Real.sqrt 27 / Real.sqrt 3 ≠ 9) := 
by
  sorry

end correct_expression_l62_6237


namespace train_length_eq_l62_6255

-- Definitions
def train_speed_kmh : Float := 45
def crossing_time_s : Float := 30
def total_length_m : Float := 245

-- Theorem statement
theorem train_length_eq :
  ∃ (train_length bridge_length: Float),
  bridge_length = total_length_m - train_length ∧
  train_speed_kmh * 1000 / 3600 * crossing_time_s = train_length + bridge_length ∧
  train_length = 130 :=
by
  sorry

end train_length_eq_l62_6255


namespace sqrt_meaningful_implies_x_ge_2_l62_6232

theorem sqrt_meaningful_implies_x_ge_2 (x : ℝ) (h : 0 ≤ x - 2) : x ≥ 2 := 
sorry

end sqrt_meaningful_implies_x_ge_2_l62_6232


namespace sin_P_equals_one_l62_6281

theorem sin_P_equals_one
  (x y : ℝ) (h1 : (1 / 2) * x * y * Real.sin 1 = 50) (h2 : x * y = 100) :
  Real.sin 1 = 1 :=
by sorry

end sin_P_equals_one_l62_6281


namespace cos_330_eq_sqrt3_over_2_l62_6234

theorem cos_330_eq_sqrt3_over_2 : Real.cos (330 * Real.pi / 180) = Real.sqrt 3 / 2 :=
by
  sorry

end cos_330_eq_sqrt3_over_2_l62_6234


namespace simple_interest_correct_l62_6227

theorem simple_interest_correct (P R T : ℝ) (hP : P = 400) (hR : R = 12.5) (hT : T = 2) : 
  (P * R * T) / 100 = 50 :=
by
  sorry -- Proof to be provided

end simple_interest_correct_l62_6227


namespace cos_sin_sum_l62_6297

open Real

theorem cos_sin_sum (α : ℝ) (h : (cos (2 * α)) / (sin (α - π / 4)) = -sqrt 2 / 2) : cos α + sin α = 1 / 2 := by
  sorry

end cos_sin_sum_l62_6297


namespace sum_of_ages_equal_to_grandpa_l62_6250

-- Conditions
def grandpa_age : Nat := 75
def grandchild_age_1 : Nat := 13
def grandchild_age_2 : Nat := 15
def grandchild_age_3 : Nat := 17

-- Main Statement
theorem sum_of_ages_equal_to_grandpa (t : Nat) :
  (grandchild_age_1 + t) + (grandchild_age_2 + t) + (grandchild_age_3 + t) = grandpa_age + t 
  ↔ t = 15 := 
by {
  sorry
}

end sum_of_ages_equal_to_grandpa_l62_6250


namespace find_x1_l62_6209

theorem find_x1 (x1 x2 x3 x4 : ℝ) 
  (h1 : 0 ≤ x4) 
  (h2 : x4 ≤ x3) 
  (h3 : x3 ≤ x2) 
  (h4 : x2 ≤ x1) 
  (h5 : x1 ≤ 1) 
  (condition : (1 - x1)^2 + (x1 - x2)^2 + (x2 - x3)^2 + (x3 - x4)^2 + x4^2 = 1 / 5) : 
  x1 = 4 / 5 := 
sorry

end find_x1_l62_6209


namespace sue_cost_l62_6253

def cost_of_car : ℝ := 2100
def total_days_in_week : ℝ := 7
def sue_days : ℝ := 3

theorem sue_cost : (cost_of_car * (sue_days / total_days_in_week)) = 899.99 :=
by
  sorry

end sue_cost_l62_6253


namespace operation_correct_l62_6273

def operation (a b : ℝ) : ℝ := 2 * a + 5 * b

theorem operation_correct :
  operation 4 2 = 18 :=
by
  show 2 * 4 + 5 * 2 = 18
  sorry

end operation_correct_l62_6273


namespace digits_divisibility_property_l62_6203

-- Definition: Example function to sum the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.foldl (· + ·) 0

-- Theorem: Prove the correctness of the given mathematical problem
theorem digits_divisibility_property:
  ∀ n : ℕ, (n = 18 ∨ n = 27 ∨ n = 45 ∨ n = 63) →
  (sum_of_digits n % 9 = 0) → (n % 9 = 0) := by
  sorry

end digits_divisibility_property_l62_6203


namespace remainder_div_x_plus_1_l62_6280

noncomputable def polynomial1 : Polynomial ℝ := Polynomial.X ^ 11 - 1

theorem remainder_div_x_plus_1 :
  Polynomial.eval (-1) polynomial1 = -2 := by
  sorry

end remainder_div_x_plus_1_l62_6280


namespace first_group_checked_correctly_l62_6298

-- Define the given conditions
def total_factories : ℕ := 169
def checked_by_second_group : ℕ := 52
def remaining_unchecked : ℕ := 48

-- Define the number of factories checked by the first group
def checked_by_first_group : ℕ := total_factories - checked_by_second_group - remaining_unchecked

-- State the theorem to be proved
theorem first_group_checked_correctly : checked_by_first_group = 69 :=
by
  -- The proof is not provided, use sorry to skip the proof steps
  sorry

end first_group_checked_correctly_l62_6298


namespace infinite_series_sum_eq_l62_6285

noncomputable def infinite_series_sum : Rat :=
  ∑' n : ℕ, (2 * n + 1) * (2000⁻¹) ^ n

theorem infinite_series_sum_eq : infinite_series_sum = (2003000 / 3996001) := by
  sorry

end infinite_series_sum_eq_l62_6285


namespace probability_odd_and_multiple_of_5_l62_6202

/-- Given three distinct integers selected at random between 1 and 2000, inclusive, the probability that the product of the three integers is odd and a multiple of 5 is between 0.01 and 0.05. -/
theorem probability_odd_and_multiple_of_5 :
  ∃ p : ℚ, (0.01 < p ∧ p < 0.05) :=
sorry

end probability_odd_and_multiple_of_5_l62_6202


namespace find_number_l62_6235

-- Define the conditions as stated in the problem
def fifteen_percent_of_x_is_ninety (x : ℝ) : Prop :=
  (15 / 100) * x = 90

-- Define the theorem to prove that given the condition, x must be 600
theorem find_number (x : ℝ) (h : fifteen_percent_of_x_is_ninety x) : x = 600 :=
sorry

end find_number_l62_6235


namespace geometric_series_sum_test_l62_6294

-- Let's define all necessary variables
variable (a : ℤ) (r : ℤ) (n : ℕ)

-- Define the geometric series sum formula
noncomputable def geometric_series_sum (a r : ℤ) (n : ℕ) : ℤ :=
  a * ((r ^ n - 1) / (r - 1))

-- Define the specific test case as per our conditions
theorem geometric_series_sum_test :
  geometric_series_sum (-2) 3 7 = -2186 :=
by
  sorry

end geometric_series_sum_test_l62_6294


namespace isosceles_triangle_perimeter_l62_6240

def is_isosceles_triangle (a b c : ℝ) : Prop :=
(a = b ∨ b = c ∨ c = a) ∧ a + b > c ∧ a + c > b ∧ b + c > a

def perimeter (a b c : ℝ) : ℝ := a + b + c

theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a = 5) (h2 : b = 10) :
∃ c : ℝ, is_isosceles_triangle a b c ∧ perimeter a b c = 25 :=
by {
  sorry
}

end isosceles_triangle_perimeter_l62_6240


namespace total_ice_cream_volume_l62_6256

def cone_height : ℝ := 10
def cone_radius : ℝ := 1.5
def cylinder_height : ℝ := 2
def cylinder_radius : ℝ := 1.5
def hemisphere_radius : ℝ := 1.5

theorem total_ice_cream_volume : 
  (1 / 3 * π * cone_radius ^ 2 * cone_height) +
  (π * cylinder_radius ^ 2 * cylinder_height) +
  (2 / 3 * π * hemisphere_radius ^ 3) = 14.25 * π :=
by sorry

end total_ice_cream_volume_l62_6256


namespace sphere_surface_area_l62_6214

theorem sphere_surface_area (R r : ℝ) (h1 : 2 * OM = R) (h2 : ∀ r, π * r^2 = 3 * π) : 4 * π * R^2 = 16 * π :=
by
  sorry

end sphere_surface_area_l62_6214


namespace broken_line_AEC_correct_l62_6223

noncomputable def length_of_broken_line_AEC 
  (side_length : ℝ)
  (height_of_pyramid : ℝ)
  (radius_of_equiv_circle : ℝ) 
  (length_AE : ℝ)
  (length_AEC : ℝ) : Prop :=
  side_length = 230.0 ∧
  height_of_pyramid = 146.423 ∧
  radius_of_equiv_circle = height_of_pyramid ∧
  length_AE = ((230.0 * 186.184) / 218.837) ∧
  length_AEC = 2 * length_AE ∧
  round (length_AEC * 100) = 39136

theorem broken_line_AEC_correct :
  length_of_broken_line_AEC 230 146.423 (146.423) 195.681 391.362 :=
by
  sorry

end broken_line_AEC_correct_l62_6223


namespace carol_initial_cupcakes_l62_6293

/--
For the school bake sale, Carol made some cupcakes. She sold 9 of them and then made 28 more.
Carol had 49 cupcakes. We need to show that Carol made 30 cupcakes initially.
-/
theorem carol_initial_cupcakes (x : ℕ) 
  (h1 : x - 9 + 28 = 49) : 
  x = 30 :=
by 
  -- The proof is not required as per instruction.
  sorry

end carol_initial_cupcakes_l62_6293


namespace total_production_first_four_days_max_min_production_difference_total_wage_for_week_l62_6261

open Int

/-- Problem Statement -/
def planned_production : Int := 220

def production_change : List Int :=
  [5, -2, -4, 13, -10, 16, -9]

/-- Proof problem for total production in the first four days -/
theorem total_production_first_four_days :
  let first_four_days := production_change.take 4
  let total_change := first_four_days.sum
  let planned_first_four_days := planned_production * 4
  planned_first_four_days + total_change = 892 := 
by
  sorry

/-- Proof problem for difference in production between highest and lowest days -/
theorem max_min_production_difference :
  let max_change := production_change.maximum.getD 0
  let min_change := production_change.minimum.getD 0
  max_change - min_change = 26 := 
by
  sorry

/-- Proof problem for total wage calculation for the week -/
theorem total_wage_for_week :
  let total_change := production_change.sum
  let planned_week_total := planned_production * 7
  let actual_total := planned_week_total + total_change
  let base_wage := actual_total * 100
  let additional_wage := total_change * 20
  base_wage + additional_wage = 155080 := 
by
  sorry

end total_production_first_four_days_max_min_production_difference_total_wage_for_week_l62_6261


namespace initial_stock_before_shipment_l62_6259

-- Define the conditions for the problem
def initial_stock (total_shelves new_shipment_bears bears_per_shelf: ℕ) : ℕ :=
  let total_bears_on_shelves := total_shelves * bears_per_shelf
  total_bears_on_shelves - new_shipment_bears

-- State the theorem with the conditions
theorem initial_stock_before_shipment : initial_stock 2 10 7 = 4 := by
  -- Mathematically, the calculation details will be handled here
  sorry

end initial_stock_before_shipment_l62_6259


namespace shopkeeper_total_cards_l62_6248

-- Conditions
def num_standard_decks := 3
def cards_per_standard_deck := 52
def num_tarot_decks := 2
def cards_per_tarot_deck := 72
def num_trading_sets := 5
def cards_per_trading_set := 100
def additional_random_cards := 27

-- Calculate total cards
def total_standard_cards := num_standard_decks * cards_per_standard_deck
def total_tarot_cards := num_tarot_decks * cards_per_tarot_deck
def total_trading_cards := num_trading_sets * cards_per_trading_set
def total_cards := total_standard_cards + total_tarot_cards + total_trading_cards + additional_random_cards

-- Proof statement
theorem shopkeeper_total_cards : total_cards = 827 := by
    sorry

end shopkeeper_total_cards_l62_6248


namespace minimum_value_fraction_l62_6296

theorem minimum_value_fraction (a : ℝ) (h : a > 1) : (a^2 - a + 1) / (a - 1) ≥ 3 :=
by
  sorry

end minimum_value_fraction_l62_6296


namespace zero_point_six_six_six_is_fraction_l62_6224

def is_fraction (x : ℝ) : Prop := ∃ (n d : ℤ), d ≠ 0 ∧ x = (n : ℝ) / (d : ℝ)

theorem zero_point_six_six_six_is_fraction:
  let sqrt_2_div_3 := (Real.sqrt 2) / 3
  let neg_sqrt_4 := - Real.sqrt 4
  let zero_point_six_six_six := 0.666
  let one_seventh := 1 / 7
  is_fraction zero_point_six_six_six :=
by sorry

end zero_point_six_six_six_is_fraction_l62_6224


namespace greatest_perfect_square_power_of_3_under_200_l62_6228

theorem greatest_perfect_square_power_of_3_under_200 :
  ∃ n : ℕ, n < 200 ∧ (∃ k : ℕ, k % 2 = 0 ∧ n = 3 ^ k) ∧ ∀ m : ℕ, (m < 200 ∧ (∃ k : ℕ, k % 2 = 0 ∧ m = 3 ^ k)) → m ≤ n :=
  sorry

end greatest_perfect_square_power_of_3_under_200_l62_6228


namespace average_speed_eq_l62_6244

variables (v₁ v₂ : ℝ) (t₁ t₂ : ℝ)

theorem average_speed_eq (h₁ : t₁ > 0) (h₂ : t₂ > 0) : 
  ((v₁ * t₁) + (v₂ * t₂)) / (t₁ + t₂) = (v₁ + v₂) / 2 := 
sorry

end average_speed_eq_l62_6244


namespace base9_39457_to_base10_is_26620_l62_6270

-- Define the components of the base 9 number 39457_9
def base9_39457 : ℕ := 39457
def base9_digits : List ℕ := [3, 9, 4, 5, 7]

-- Define the base
def base : ℕ := 9

-- Convert each position to its base 10 equivalent
def base9_to_base10 : ℕ :=
  3 * base ^ 4 + 9 * base ^ 3 + 4 * base ^ 2 + 5 * base ^ 1 + 7 * base ^ 0

-- State the theorem
theorem base9_39457_to_base10_is_26620 : base9_to_base10 = 26620 := by
  sorry

end base9_39457_to_base10_is_26620_l62_6270


namespace find_missing_number_l62_6268

theorem find_missing_number
  (x y : ℕ)
  (h1 : 30 = 6 * 5)
  (h2 : 600 = 30 * x)
  (h3 : x = 5 * y) :
  y = 4 :=
by
  sorry

end find_missing_number_l62_6268


namespace ratio_night_to_day_l62_6267

-- Definitions based on conditions
def birds_day : ℕ := 8
def birds_total : ℕ := 24
def birds_night : ℕ := birds_total - birds_day

-- Theorem statement
theorem ratio_night_to_day : birds_night / birds_day = 2 := by
  sorry

end ratio_night_to_day_l62_6267


namespace solution_set_inequality_x0_1_solution_set_inequality_x0_half_l62_6271

noncomputable def f (x : ℝ) : ℝ := abs (Real.log x)

theorem solution_set_inequality_x0_1 : 
  ∀ (c : ℝ), (∀ x, 0 < x → f x - f 1 ≥ c * (x - 1)) ↔ c ∈ Set.Icc (-1) 1 := 
by
  sorry

theorem solution_set_inequality_x0_half : 
  ∀ (c : ℝ), (∀ x, 0 < x → f x - f (1 / 2) ≥ c * (x - 1 / 2)) ↔ c = -2 :=
by
  sorry

end solution_set_inequality_x0_1_solution_set_inequality_x0_half_l62_6271


namespace reciprocal_lcm_of_24_and_208_l62_6299

theorem reciprocal_lcm_of_24_and_208 :
  (1 / (Nat.lcm 24 208)) = (1 / 312) :=
by
  sorry

end reciprocal_lcm_of_24_and_208_l62_6299


namespace two_wheeler_wheels_l62_6257

-- Define the total number of wheels and the number of four-wheelers
def total_wheels : Nat := 46
def num_four_wheelers : Nat := 11

-- Define the number of wheels per vehicle type
def wheels_per_four_wheeler : Nat := 4
def wheels_per_two_wheeler : Nat := 2

-- Define the number of two-wheelers
def num_two_wheelers : Nat := (total_wheels - num_four_wheelers * wheels_per_four_wheeler) / wheels_per_two_wheeler

-- Proposition stating the number of wheels of the two-wheeler
theorem two_wheeler_wheels : wheels_per_two_wheeler * num_two_wheelers = 2 := by
  sorry

end two_wheeler_wheels_l62_6257


namespace yard_length_l62_6204

-- Define the given conditions
def num_trees : ℕ := 26
def dist_between_trees : ℕ := 13

-- Calculate the length of the yard
def num_gaps : ℕ := num_trees - 1
def length_of_yard : ℕ := num_gaps * dist_between_trees

-- Theorem statement: the length of the yard is 325 meters
theorem yard_length : length_of_yard = 325 := by
  sorry

end yard_length_l62_6204


namespace percent_enclosed_by_hexagons_l62_6284

variable (b : ℝ) -- side length of smaller squares

def area_of_small_square : ℝ := b^2
def area_of_large_square : ℝ := 16 * area_of_small_square b
def area_of_hexagon : ℝ := 3 * area_of_small_square b
def total_area_of_hexagons : ℝ := 2 * area_of_hexagon b

theorem percent_enclosed_by_hexagons :
  (total_area_of_hexagons b / area_of_large_square b) * 100 = 37.5 :=
by
  -- Proof omitted
  sorry

end percent_enclosed_by_hexagons_l62_6284


namespace sum_of_interior_angles_10th_polygon_l62_6260

theorem sum_of_interior_angles_10th_polygon (n : ℕ) (h1 : n = 10) : 
  180 * (n - 2) = 1440 :=
by
  sorry

end sum_of_interior_angles_10th_polygon_l62_6260


namespace parabola_standard_equation_l62_6238

theorem parabola_standard_equation :
  ∃ p1 p2 : ℝ, p1 > 0 ∧ p2 > 0 ∧ (y^2 = 2 * p1 * x ∨ x^2 = 2 * p2 * y) ∧ ((6, 4) ∈ {(x, y) | y^2 = 2 * p1 * x} ∨ (6, 4) ∈ {(x, y) | x^2 = 2 * p2 * y}) := 
  sorry

end parabola_standard_equation_l62_6238


namespace jack_jill_total_difference_l62_6254

theorem jack_jill_total_difference :
  let original_price := 90.00
  let discount_rate := 0.20
  let tax_rate := 0.06

  -- Jack's calculation
  let jack_total :=
    let price_with_tax := original_price * (1 + tax_rate)
    price_with_tax * (1 - discount_rate)
  
  -- Jill's calculation
  let jill_total :=
    let discounted_price := original_price * (1 - discount_rate)
    discounted_price * (1 + tax_rate)

  -- Equality check
  jack_total = jill_total := 
by
  -- Place the proof here
  sorry

end jack_jill_total_difference_l62_6254


namespace bryden_collection_value_l62_6241

-- Define the conditions
def face_value_half_dollar : ℝ := 0.5
def face_value_quarter : ℝ := 0.25
def num_half_dollars : ℕ := 5
def num_quarters : ℕ := 3
def multiplier : ℝ := 30

-- Define the problem statement as a theorem
theorem bryden_collection_value : 
  (multiplier * (num_half_dollars * face_value_half_dollar + num_quarters * face_value_quarter)) = 97.5 :=
by
  -- Proof is skipped since it's not required
  sorry

end bryden_collection_value_l62_6241


namespace ellipse_eccentricity_l62_6220

theorem ellipse_eccentricity (x y : ℝ) (h : x^2 / 25 + y^2 / 9 = 1) : 
  let a := 5
  let b := 3
  let c := 4
  let e := c / a
  e = 4 / 5 :=
by
  sorry

end ellipse_eccentricity_l62_6220


namespace bottles_needed_l62_6262

-- Define specific values provided in conditions
def servings_per_guest : ℕ := 2
def number_of_guests : ℕ := 120
def servings_per_bottle : ℕ := 6

-- Define total servings needed
def total_servings : ℕ := servings_per_guest * number_of_guests

-- Define the number of bottles needed (as a proof statement)
theorem bottles_needed : total_servings / servings_per_bottle = 40 := by
  /-
    The proof will go here. For now we place a sorry to mark the place where
    a proof would be required. The statement should check the equivalence of 
    number of bottles needed being 40 given the total servings divided by 
    servings per bottle.
  -/
  sorry

end bottles_needed_l62_6262


namespace sum_of_A_and_B_l62_6206

theorem sum_of_A_and_B (A B : ℕ) (h1 : 7 - B = 3) (h2 : A - 5 = 4) (h_diff : A ≠ B) : A + B = 13 :=
sorry

end sum_of_A_and_B_l62_6206


namespace inequality_comparison_l62_6211

theorem inequality_comparison 
  (a : ℝ) (b : ℝ) (c : ℝ) 
  (h₁ : a = (1 / Real.log 3 / Real.log 2))
  (h₂ : b = Real.exp 0.5)
  (h₃ : c = Real.log 2) :
  b > c ∧ c > a := 
by
  sorry

end inequality_comparison_l62_6211


namespace initial_amount_l62_6229

-- Define the conditions
def cost_small_glass : ℕ := 3
def cost_large_glass : ℕ := 5
def num_small_glasses : ℕ := 8
def num_large_glasses : ℕ := 5
def change_left : ℕ := 1

-- Define the pieces based on conditions
def total_cost_small_glasses : ℕ := num_small_glasses * cost_small_glass
def total_cost_large_glasses : ℕ := num_large_glasses * cost_large_glass
def total_cost_glasses : ℕ := total_cost_small_glasses + total_cost_large_glasses

-- The theorem we need to prove
theorem initial_amount (h1 : total_cost_small_glasses = 24)
                       (h2 : total_cost_large_glasses = 25)
                       (h3 : total_cost_glasses = 49) : total_cost_glasses + change_left = 50 :=
by sorry

end initial_amount_l62_6229


namespace calc_result_l62_6205

theorem calc_result :
  12 / 4 - 3 - 16 + 4 * 6 = 8 := by
  sorry

end calc_result_l62_6205


namespace smallest_positive_multiple_l62_6287

theorem smallest_positive_multiple (a : ℕ) (h₁ : a % 6 = 0) (h₂ : a % 15 = 0) : a = 30 :=
sorry

end smallest_positive_multiple_l62_6287


namespace baker_new_cakes_l62_6215

theorem baker_new_cakes :
  ∀ (initial_bought new_bought sold final : ℕ),
  initial_bought = 173 →
  sold = 86 →
  final = 190 →
  final = initial_bought + new_bought - sold →
  new_bought = 103 :=
by
  intros initial_bought new_bought sold final H_initial H_sold H_final H_eq
  sorry

end baker_new_cakes_l62_6215


namespace largest_number_is_A_l62_6264

def numA : ℝ := 0.989
def numB : ℝ := 0.9879
def numC : ℝ := 0.98809
def numD : ℝ := 0.9807
def numE : ℝ := 0.9819

theorem largest_number_is_A :
  (numA > numB) ∧ (numA > numC) ∧ (numA > numD) ∧ (numA > numE) :=
by sorry

end largest_number_is_A_l62_6264


namespace sqrt_6_approx_l62_6222

noncomputable def newton_iteration (x : ℝ) : ℝ :=
  (1 / 2) * x + (3 / x)

theorem sqrt_6_approx :
  let x0 : ℝ := 2
  let x1 : ℝ := newton_iteration x0
  let x2 : ℝ := newton_iteration x1
  let x3 : ℝ := newton_iteration x2
  abs (x3 - 2.4495) < 0.0001 :=
by
  sorry

end sqrt_6_approx_l62_6222


namespace none_of_these_l62_6252

noncomputable def x (t : ℝ) : ℝ := t ^ (3 / (t - 1))
noncomputable def y (t : ℝ) : ℝ := t ^ ((t + 1) / (t - 1))

theorem none_of_these (t : ℝ) (ht_pos : t > 0) (ht_ne_one : t ≠ 1) :
  ¬ (y t ^ x t = x t ^ y t) ∧ ¬ (x t ^ x t = y t ^ y t) ∧
  ¬ (x t ^ (y t ^ x t) = y t ^ (x t ^ y t)) ∧ ¬ (x t ^ y t = y t ^ x t) :=
sorry

end none_of_these_l62_6252


namespace apples_hand_out_l62_6242

theorem apples_hand_out (t p a h : ℕ) (h_t : t = 62) (h_p : p = 6) (h_a : a = 9) : h = t - (p * a) → h = 8 :=
by
  intros
  sorry

end apples_hand_out_l62_6242


namespace minimum_bounces_to_reach_height_l62_6278

noncomputable def height_after_bounces (initial_height : ℝ) (bounce_factor : ℝ) (k : ℕ) : ℝ :=
  initial_height * (bounce_factor ^ k)

theorem minimum_bounces_to_reach_height
  (initial_height : ℝ) (bounce_factor : ℝ) (min_height : ℝ) :
  initial_height = 800 → bounce_factor = 0.5 → min_height = 2 →
  (∀ k : ℕ, height_after_bounces initial_height bounce_factor k < min_height ↔ k ≥ 9) := 
by
  intros h₀ b₀ m₀
  rw [h₀, b₀, m₀]
  sorry

end minimum_bounces_to_reach_height_l62_6278


namespace value_at_neg_9_over_2_l62_6221

def f : ℝ → ℝ := sorry 

axiom odd_function (x : ℝ) : f (-x) + f x = 0

axiom symmetric_y_axis (x : ℝ) : f (1 + x) = f (1 - x)

axiom functional_eq (x k : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hk : 0 ≤ k ∧ k ≤ 1) : f (k * x) + 1 = (f x + 1) ^ k

axiom f_at_1 : f 1 = - (1 / 2)

theorem value_at_neg_9_over_2 : f (- (9 / 2)) = 1 - (Real.sqrt 2) / 2 := 
sorry

end value_at_neg_9_over_2_l62_6221
