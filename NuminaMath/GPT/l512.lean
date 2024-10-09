import Mathlib

namespace problem1_problem2_l512_51205

-- Problem 1
theorem problem1 : 23 + (-13) + (-17) + 8 = 1 :=
by
  sorry

-- Problem 2
theorem problem2 : - (2^3) - (1 + 0.5) / (1/3) * (-3) = 11/2 :=
by
  sorry

end problem1_problem2_l512_51205


namespace no_digit_B_divisible_by_4_l512_51251

theorem no_digit_B_divisible_by_4 : 
  ∀ B : ℕ, B < 10 → ¬ (8 * 1000000 + B * 100000 + 4 * 10000 + 6 * 1000 + 3 * 100 + 5 * 10 + 1) % 4 = 0 :=
by
  intros B hB_lt_10
  sorry

end no_digit_B_divisible_by_4_l512_51251


namespace kayla_waiting_years_l512_51218

def minimum_driving_age : ℕ := 18
def kimiko_age : ℕ := 26
def kayla_age : ℕ := kimiko_age / 2
def years_until_kayla_can_drive : ℕ := minimum_driving_age - kayla_age

theorem kayla_waiting_years : years_until_kayla_can_drive = 5 :=
by
  sorry

end kayla_waiting_years_l512_51218


namespace tetrahedron_vertex_equality_l512_51247

theorem tetrahedron_vertex_equality
  (r1 r2 r3 r4 j1 j2 j3 j4 : ℝ) (hr1 : r1 > 0) (hr2 : r2 > 0) (hr3 : r3 > 0) (hr4 : r4 > 0)
  (hj1 : j1 > 0) (hj2 : j2 > 0) (hj3 : j3 > 0) (hj4 : j4 > 0) 
  (h1 : r2 * r3 + r3 * r4 + r4 * r2 = j2 * j3 + j3 * j4 + j4 * j2)
  (h2 : r1 * r3 + r3 * r4 + r4 * r1 = j1 * j3 + j3 * j4 + j4 * j1)
  (h3 : r1 * r2 + r2 * r4 + r4 * r1 = j1 * j2 + j2 * j4 + j4 * j1)
  (h4 : r1 * r2 + r2 * r3 + r3 * r1 = j1 * j2 + j2 * j3 + j3 * j1) :
  r1 = j1 ∧ r2 = j2 ∧ r3 = j3 ∧ r4 = j4 := by
  sorry

end tetrahedron_vertex_equality_l512_51247


namespace wood_length_equation_l512_51224

theorem wood_length_equation (x : ℝ) : 
  (∃ r : ℝ, r - x = 4.5 ∧ r/2 + 1 = x) → 1/2 * (x + 4.5) = x - 1 :=
sorry

end wood_length_equation_l512_51224


namespace factorize_cubic_l512_51215

theorem factorize_cubic (x : ℝ) : x^3 - x = x * (x + 1) * (x - 1) :=
by sorry

end factorize_cubic_l512_51215


namespace miles_driven_each_day_l512_51245

theorem miles_driven_each_day
  (total_distance : ℕ)
  (days_in_semester : ℕ)
  (h_total : total_distance = 1600)
  (h_days : days_in_semester = 80):
  total_distance / days_in_semester = 20 := by
  sorry

end miles_driven_each_day_l512_51245


namespace roots_sum_of_quadratic_l512_51208

theorem roots_sum_of_quadratic:
  (∃ a b : ℝ, (a ≠ b) ∧ (a * b = 5) ∧ (a + b = 8)) →
  (a + b = 8) :=
by
  sorry

end roots_sum_of_quadratic_l512_51208


namespace cricket_target_runs_l512_51234

def run_rate_first_20_overs : ℝ := 4.2
def overs_first_20 : ℝ := 20
def run_rate_remaining_30_overs : ℝ := 5.533333333333333
def overs_remaining_30 : ℝ := 30
def total_runs_first_20 : ℝ := run_rate_first_20_overs * overs_first_20
def total_runs_remaining_30 : ℝ := run_rate_remaining_30_overs * overs_remaining_30

theorem cricket_target_runs :
  (total_runs_first_20 + total_runs_remaining_30) = 250 :=
by
  sorry

end cricket_target_runs_l512_51234


namespace john_wages_decrease_percentage_l512_51274

theorem john_wages_decrease_percentage (W : ℝ) (P : ℝ) :
  (0.20 * (W - P/100 * W)) = 0.50 * (0.30 * W) → P = 25 :=
by 
  intro h
  -- Simplification and other steps omitted; focus on structure
  sorry

end john_wages_decrease_percentage_l512_51274


namespace find_c_l512_51277

theorem find_c (x c : ℝ) (h₁ : 3 * x + 6 = 0) (h₂ : c * x - 15 = -3) : c = -6 := 
by
  -- sorry is used here as we are not required to provide the proof steps
  sorry

end find_c_l512_51277


namespace triangle_side_lengths_l512_51210

theorem triangle_side_lengths
  (x y z : ℕ)
  (h1 : x > y)
  (h2 : y > z)
  (h3 : x + y + z = 240)
  (h4 : 3 * x - 2 * (y + z) = 5 * z + 10)
  (h5 : x < y + z) :
  (x = 113 ∧ y = 112 ∧ z = 15) ∨
  (x = 114 ∧ y = 110 ∧ z = 16) ∨
  (x = 115 ∧ y = 108 ∧ z = 17) ∨
  (x = 116 ∧ y = 106 ∧ z = 18) ∨
  (x = 117 ∧ y = 104 ∧ z = 19) ∨
  (x = 118 ∧ y = 102 ∧ z = 20) ∨
  (x = 119 ∧ y = 100 ∧ z = 21) := by
  sorry

end triangle_side_lengths_l512_51210


namespace keith_attended_games_l512_51214

-- Definitions from the conditions
def total_games : ℕ := 20
def missed_games : ℕ := 9

-- The statement to prove
theorem keith_attended_games : (total_games - missed_games) = 11 :=
by
  sorry

end keith_attended_games_l512_51214


namespace bottle_caps_proof_l512_51263

def bottle_caps_difference (found thrown : ℕ) := found - thrown

theorem bottle_caps_proof : bottle_caps_difference 50 6 = 44 := by
  sorry

end bottle_caps_proof_l512_51263


namespace edwards_initial_money_l512_51223

variable (spent1 spent2 current remaining : ℕ)

def initial_money (spent1 spent2 current remaining : ℕ) : ℕ :=
  spent1 + spent2 + current

theorem edwards_initial_money :
  spent1 = 9 → spent2 = 8 → remaining = 17 →
  initial_money spent1 spent2 remaining remaining = 34 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end edwards_initial_money_l512_51223


namespace percent_full_time_more_than_three_years_l512_51202

variable (total_associates : ℕ)
variable (second_year_percentage : ℕ)
variable (third_year_percentage : ℕ)
variable (non_first_year_percentage : ℕ)
variable (part_time_percentage : ℕ)
variable (part_time_more_than_two_years_percentage : ℕ)
variable (full_time_more_than_three_years_percentage : ℕ)

axiom condition_1 : second_year_percentage = 30
axiom condition_2 : third_year_percentage = 20
axiom condition_3 : non_first_year_percentage = 60
axiom condition_4 : part_time_percentage = 10
axiom condition_5 : part_time_more_than_two_years_percentage = 5

theorem percent_full_time_more_than_three_years : 
  full_time_more_than_three_years_percentage = 10 := 
sorry

end percent_full_time_more_than_three_years_l512_51202


namespace blocks_differ_in_two_ways_l512_51236

/-- 
A child has a set of 120 distinct blocks. Each block is one of 3 materials (plastic, wood, metal), 
3 sizes (small, medium, large), 4 colors (blue, green, red, yellow), and 5 shapes (circle, hexagon, 
square, triangle, pentagon). How many blocks in the set differ from the 'metal medium blue hexagon' 
in exactly 2 ways?
-/
def num_blocks_differ_in_two_ways : Nat := 44

theorem blocks_differ_in_two_ways (blocks : Fin 120)
    (materials : Fin 3)
    (sizes : Fin 3)
    (colors : Fin 4)
    (shapes : Fin 5)
    (fixed_block : {m // m = 2} × {s // s = 1} × {c // c = 0} × {sh // sh = 1}) :
    num_blocks_differ_in_two_ways = 44 :=
by
  -- proof steps are omitted
  sorry

end blocks_differ_in_two_ways_l512_51236


namespace value_of_m_l512_51287

theorem value_of_m
  (x y m : ℝ)
  (h1 : 2 * x + 3 * y = 4)
  (h2 : 3 * x + 2 * y = 2 * m - 3)
  (h3 : x + y = -3/5) :
  m = -2 :=
sorry

end value_of_m_l512_51287


namespace right_triangle_third_side_l512_51221

theorem right_triangle_third_side (a b c : ℝ) (h₁ : a = 3) (h₂ : b = 4) (h₃ : c = 5) (h₄ : c = Real.sqrt (7) ∨ c = 5) :
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2 := by
  sorry

end right_triangle_third_side_l512_51221


namespace problem1_problem2_l512_51284

theorem problem1 : -7 + 13 - 6 + 20 = 20 := 
by
  sorry

theorem problem2 : -2^3 + (2 - 3) - 2 * (-1)^2023 = -7 := 
by
  sorry

end problem1_problem2_l512_51284


namespace find_square_number_divisible_by_five_l512_51296

noncomputable def is_square (n : ℕ) : Prop :=
∃ k : ℕ, k * k = n

theorem find_square_number_divisible_by_five :
  ∃ x : ℕ, x ≥ 50 ∧ x ≤ 120 ∧ is_square x ∧ x % 5 = 0 ↔ x = 100 := by
sorry

end find_square_number_divisible_by_five_l512_51296


namespace sam_received_87_l512_51254

def sam_total_money : Nat :=
  sorry

theorem sam_received_87 (spent left_over : Nat) (h1 : spent = 64) (h2 : left_over = 23) :
  sam_total_money = spent + left_over :=
by
  rw [h1, h2]
  sorry

example : sam_total_money = 64 + 23 :=
  sam_received_87 64 23 rfl rfl

end sam_received_87_l512_51254


namespace function_properties_l512_51226

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6)

theorem function_properties :
  (∀ x, f (x + Real.pi) = f x) ∧
  (f (Real.pi / 3) = 1) ∧
  (∀ x y, -Real.pi / 6 ≤ x → x ≤ y → y ≤ Real.pi / 3 → f x ≤ f y) := by
  sorry

end function_properties_l512_51226


namespace net_difference_in_expenditure_l512_51211

variable (P Q : ℝ)
-- Condition 1: Price increased by 25%
def new_price (P : ℝ) : ℝ := P * 1.25

-- Condition 2: Purchased 72% of the originally required amount
def new_quantity (Q : ℝ) : ℝ := Q * 0.72

-- Definition of original expenditure
def original_expenditure (P Q : ℝ) : ℝ := P * Q

-- Definition of new expenditure
def new_expenditure (P Q : ℝ) : ℝ := new_price P * new_quantity Q

-- Statement of the proof problem.
theorem net_difference_in_expenditure
  (P Q : ℝ) : new_expenditure P Q - original_expenditure P Q = -0.1 * original_expenditure P Q := 
by
  sorry

end net_difference_in_expenditure_l512_51211


namespace simplify_fraction_l512_51242

theorem simplify_fraction : 5 * (21 / 8) * (32 / -63) = -20 / 3 := by
  sorry

end simplify_fraction_l512_51242


namespace mean_proportional_234_104_l512_51237

theorem mean_proportional_234_104 : Real.sqrt (234 * 104) = 156 :=
by 
  sorry

end mean_proportional_234_104_l512_51237


namespace misread_weight_l512_51207

theorem misread_weight (avg_initial : ℝ) (avg_correct : ℝ) (n : ℕ) (actual_weight : ℝ) (x : ℝ) : 
  avg_initial = 58.4 → avg_correct = 58.7 → n = 20 → actual_weight = 62 → 
  (n * avg_correct - n * avg_initial = actual_weight - x) → x = 56 :=
by
  intros
  sorry

end misread_weight_l512_51207


namespace circle_area_with_diameter_CD_l512_51280

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem circle_area_with_diameter_CD (C D E : ℝ × ℝ)
  (hC : C = (-1, 2)) (hD : D = (5, -6)) (hE : E = (2, -2))
  (hE_midpoint : E = ((C.1 + D.1) / 2, (C.2 + D.2) / 2)) :
  ∃ (A : ℝ), A = 25 * Real.pi :=
by
  -- Define the coordinates of points C and D
  let Cx := -1
  let Cy := 2
  let Dx := 5
  let Dy := -6

  -- Calculate the distance (diameter) between C and D
  let diameter := distance Cx Cy Dx Dy

  -- Calculate the radius of the circle
  let radius := diameter / 2

  -- Calculate the area of the circle
  let area := Real.pi * radius^2

  -- Prove the area is 25π
  use area
  sorry

end circle_area_with_diameter_CD_l512_51280


namespace max_single_player_salary_l512_51289

theorem max_single_player_salary
    (num_players : ℕ) (min_salary : ℕ) (total_salary_cap : ℕ)
    (num_player_min_salary : ℕ) (max_salary : ℕ)
    (h1 : num_players = 18)
    (h2 : min_salary = 20000)
    (h3 : total_salary_cap = 600000)
    (h4 : num_player_min_salary = 17)
    (h5 : num_players = num_player_min_salary + 1)
    (h6 : total_salary_cap = num_player_min_salary * min_salary + max_salary) :
    max_salary = 260000 :=
by
  sorry

end max_single_player_salary_l512_51289


namespace quadratic_residue_iff_l512_51264

open Nat

theorem quadratic_residue_iff (p : ℕ) [Fact (Nat.Prime p)] (hp : p % 2 = 1) (n : ℤ) (hn : n % p ≠ 0) :
  (∃ a : ℤ, (a^2) % p = n % p) ↔ (n ^ ((p - 1) / 2)) % p = 1 :=
sorry

end quadratic_residue_iff_l512_51264


namespace quadratic_function_a_equals_one_l512_51282

noncomputable def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + b * x + c

theorem quadratic_function_a_equals_one
  (a b c : ℝ)
  (h1 : 1 < x)
  (h2 : x < c)
  (h_neg : ∀ x, 1 < x → x < c → quadratic_function a b c x < 0):
  a = 1 := by
  sorry

end quadratic_function_a_equals_one_l512_51282


namespace find_k_b_find_x_when_y_neg_8_l512_51276

theorem find_k_b (k b : ℤ) (h1 : -20 = 4 * k + b) (h2 : 16 = -2 * k + b) : k = -6 ∧ b = 4 := 
sorry

theorem find_x_when_y_neg_8 (x : ℤ) (k b : ℤ) (h_k : k = -6) (h_b : b = 4) (h_target : -8 = k * x + b) : x = 2 := 
sorry

end find_k_b_find_x_when_y_neg_8_l512_51276


namespace person_speed_l512_51243

theorem person_speed (d_meters : ℕ) (t_minutes : ℕ) (d_km t_hours : ℝ) :
  (d_meters = 1800) →
  (t_minutes = 12) →
  (d_km = d_meters / 1000) →
  (t_hours = t_minutes / 60) →
  d_km / t_hours = 9 :=
by
  intros
  sorry

end person_speed_l512_51243


namespace product_plus_one_eq_216_l512_51246

variable (a b c : ℝ)

theorem product_plus_one_eq_216 
  (h1 : a * b + a + b = 35)
  (h2 : b * c + b + c = 35)
  (h3 : c * a + c + a = 35)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c) :
  (a + 1) * (b + 1) * (c + 1) = 216 := 
sorry

end product_plus_one_eq_216_l512_51246


namespace only_p_eq_3_l512_51298

theorem only_p_eq_3 (p : ℕ) (h1 : Prime p) (h2 : Prime (8 * p ^ 2 + 1)) : p = 3 := 
by
  sorry

end only_p_eq_3_l512_51298


namespace find_k_l512_51268

theorem find_k (k x y : ℝ) (h_ne_zero : k ≠ 0) (h_x : x = 4) (h_y : y = -1/2) (h_eq : y = k / x) : k = -2 :=
by
  -- This is where the proof would go
  sorry

end find_k_l512_51268


namespace alex_downhill_time_l512_51228

theorem alex_downhill_time
  (speed_flat : ℝ)
  (time_flat : ℝ)
  (speed_uphill : ℝ)
  (time_uphill : ℝ)
  (speed_downhill : ℝ)
  (distance_walked : ℝ)
  (total_distance : ℝ)
  (h_flat : speed_flat = 20)
  (h_time_flat : time_flat = 4.5)
  (h_uphill : speed_uphill = 12)
  (h_time_uphill : time_uphill = 2.5)
  (h_downhill : speed_downhill = 24)
  (h_walked : distance_walked = 8)
  (h_total : total_distance = 164)
  : (156 - (speed_flat * time_flat + speed_uphill * time_uphill)) / speed_downhill = 1.5 :=
by 
  sorry

end alex_downhill_time_l512_51228


namespace sweater_markup_percentage_l512_51231

variables (W R : ℝ)
variables (h1 : 0.30 * R = 1.40 * W)

theorem sweater_markup_percentage :
  (R = (1.40 / 0.30) * W) →
  (R - W) / W * 100 = 367 := 
by
  intro hR
  sorry

end sweater_markup_percentage_l512_51231


namespace john_total_amount_to_pay_l512_51269

-- Define constants for the problem
def total_cost : ℝ := 6650
def rebate_percentage : ℝ := 0.06
def sales_tax_percentage : ℝ := 0.10

-- The main theorem to prove the final amount John needs to pay
theorem john_total_amount_to_pay : total_cost * (1 - rebate_percentage) * (1 + sales_tax_percentage) = 6876.10 := by
  sorry    -- Proof skipped

end john_total_amount_to_pay_l512_51269


namespace find_f_x_l512_51285

theorem find_f_x (f : ℝ → ℝ) (h : ∀ x : ℝ, f x + 1 = 3 * x + 2) : ∀ x : ℝ, f x = 3 * x - 1 :=
by
  sorry

end find_f_x_l512_51285


namespace toys_produced_each_day_l512_51229

def toys_produced_per_week : ℕ := 6000
def work_days_per_week : ℕ := 4

theorem toys_produced_each_day :
  (toys_produced_per_week / work_days_per_week) = 1500 := 
by
  -- The details of the proof are omitted
  -- The correct answer given the conditions is 1500 toys
  sorry

end toys_produced_each_day_l512_51229


namespace negative_value_option_D_l512_51230

theorem negative_value_option_D :
  (-7) * (-6) > 0 ∧
  (-7) - (-15) > 0 ∧
  0 * (-2) * (-3) = 0 ∧
  (-6) + (-4) < 0 :=
by
  sorry

end negative_value_option_D_l512_51230


namespace final_composite_score_is_correct_l512_51299

-- Defining scores
def written_exam_score : ℝ := 94
def interview_score : ℝ := 80
def practical_operation_score : ℝ := 90

-- Defining weights
def written_exam_weight : ℝ := 5
def interview_weight : ℝ := 2
def practical_operation_weight : ℝ := 3
def total_weight : ℝ := written_exam_weight + interview_weight + practical_operation_weight

-- Final composite score
noncomputable def composite_score : ℝ :=
  (written_exam_score * written_exam_weight + interview_score * interview_weight + practical_operation_score * practical_operation_weight)
  / total_weight

-- The theorem to be proved
theorem final_composite_score_is_correct : composite_score = 90 := by
  sorry

end final_composite_score_is_correct_l512_51299


namespace enterprise_b_pays_more_in_2015_l512_51275

variable (a b x y : ℝ)
variable (ha2x : a + 2 * x = b)
variable (ha1y : a * (1+y)^2 = b)

theorem enterprise_b_pays_more_in_2015 : b * (1 + y) > b + x := by
  sorry

end enterprise_b_pays_more_in_2015_l512_51275


namespace solve_equation_l512_51212

theorem solve_equation :
  ∃ x : ℝ, (x - 2)^2 - (x + 3) * (x - 3) = 4 * x - 1 ∧ x = 7 / 4 := 
by
  sorry

end solve_equation_l512_51212


namespace root_value_l512_51293

theorem root_value (a : ℝ) (h: 3 * a^2 - 4 * a + 1 = 0) : 6 * a^2 - 8 * a + 5 = 3 := 
by 
  sorry

end root_value_l512_51293


namespace statement_1_correct_statement_3_correct_correct_statements_l512_51203

-- Definition for Acute Angles
def is_acute_angle (α : Real) : Prop :=
  0 < α ∧ α < 90

-- Definition for First Quadrant Angles
def is_first_quadrant_angle (β : Real) : Prop :=
  ∃ k : Int, k * 360 < β ∧ β < 90 + k * 360

-- Conditions
theorem statement_1_correct (α : Real) : is_acute_angle α → is_first_quadrant_angle α :=
sorry

theorem statement_3_correct (β : Real) : is_first_quadrant_angle β :=
sorry

-- Final Proof Statement
theorem correct_statements (α β : Real) :
  (is_acute_angle α → is_first_quadrant_angle α) ∧ (is_first_quadrant_angle β) :=
⟨statement_1_correct α, statement_3_correct β⟩

end statement_1_correct_statement_3_correct_correct_statements_l512_51203


namespace necessary_but_not_sufficient_l512_51222

theorem necessary_but_not_sufficient (x : ℝ) (h : x ≠ 1) : x^2 - 3 * x + 2 ≠ 0 :=
by
  intro h1
  -- Insert the proof here
  sorry

end necessary_but_not_sufficient_l512_51222


namespace polygon_side_count_l512_51200

theorem polygon_side_count (s : ℝ) (hs : s ≠ 0) : 
  ∀ (side_length_ratio : ℝ) (sides_first sides_second : ℕ),
  sides_first = 50 ∧ side_length_ratio = 3 ∧ 
  sides_first * side_length_ratio * s = sides_second * s → sides_second = 150 :=
by
  sorry

end polygon_side_count_l512_51200


namespace max_digit_sum_in_24_hour_format_l512_51239

def digit_sum (n : ℕ) : ℕ := 
  (n / 10) + (n % 10)

theorem max_digit_sum_in_24_hour_format :
  (∃ (h m : ℕ), 0 ≤ h ∧ h < 24 ∧ 0 ≤ m ∧ m < 60 ∧ digit_sum h + digit_sum m = 19) ∧
  ∀ (h m : ℕ), 0 ≤ h ∧ h < 24 ∧ 0 ≤ m ∧ m < 60 → digit_sum h + digit_sum m ≤ 19 :=
by
  sorry

end max_digit_sum_in_24_hour_format_l512_51239


namespace find_a_plus_d_l512_51273

noncomputable def f (a b c d x : ℚ) : ℚ := (a * x + b) / (c * x + d)

theorem find_a_plus_d (a b c d : ℚ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) (h₃ : d ≠ 0)
  (h₄ : ∀ x : ℚ, f a b c d (f a b c d x) = x) :
  a + d = 0 := by
  sorry

end find_a_plus_d_l512_51273


namespace simplify_expression_l512_51279

variable (y : ℝ)

theorem simplify_expression : 
  3 * y - 5 * y^2 + 2 + (8 - 5 * y + 2 * y^2) = -3 * y^2 - 2 * y + 10 := 
by
  sorry

end simplify_expression_l512_51279


namespace min_troublemakers_l512_51248

theorem min_troublemakers (n : ℕ) (students : ℕ → Prop) 
  (h : n = 29)
  (condition1 : ∀ i, students i → (students ((i - 1) % n) ↔ ¬ students ((i + 1) % n)))
  (condition2 : ∀ i, ¬ students i → (students ((i - 1) % n) ∧ students ((i + 1) % n)))
  : ∃ L : ℕ, (L ≤ 29 ∧ L ≥ 10) :=
by sorry

end min_troublemakers_l512_51248


namespace cost_of_corn_per_acre_l512_51255

def TotalLand : ℕ := 4500
def CostWheat : ℕ := 35
def Capital : ℕ := 165200
def LandWheat : ℕ := 3400
def LandCorn := TotalLand - LandWheat

theorem cost_of_corn_per_acre :
  ∃ C : ℕ, (Capital = (C * LandCorn) + (CostWheat * LandWheat)) ∧ C = 42 :=
by
  sorry

end cost_of_corn_per_acre_l512_51255


namespace sinB_law_of_sines_l512_51216

variable (A B C : Type)
variable [MetricSpace A] [MetricSpace B] [MetricSpace C]

-- Assuming a triangle with sides and angles as described
variable (a b : ℝ) (sinA sinB : ℝ)
variable (h₁ : a = 3) (h₂ : b = 5) (h₃ : sinA = 1 / 3)

theorem sinB_law_of_sines : sinB = 5 / 9 :=
by
  -- Placeholder for the proof
  sorry

end sinB_law_of_sines_l512_51216


namespace wendy_first_album_pictures_l512_51232

theorem wendy_first_album_pictures 
  (total_pictures : ℕ)
  (num_albums : ℕ)
  (pics_per_album : ℕ)
  (pics_in_first_album : ℕ)
  (h1 : total_pictures = 79)
  (h2 : num_albums = 5)
  (h3 : pics_per_album = 7)
  (h4 : total_pictures = pics_in_first_album + num_albums * pics_per_album) : 
  pics_in_first_album = 44 :=
by
  sorry

end wendy_first_album_pictures_l512_51232


namespace find_vasya_floor_l512_51257

theorem find_vasya_floor (steps_petya: ℕ) (steps_vasya: ℕ) (petya_floors: ℕ) (steps_per_floor: ℝ):
  steps_petya = 36 → petya_floors = 2 → steps_vasya = 72 → 
  steps_per_floor = steps_petya / petya_floors → 
  (1 + (steps_vasya / steps_per_floor)) = 5 := by 
  intros h1 h2 h3 h4 
  sorry

end find_vasya_floor_l512_51257


namespace x_pow_10_eq_correct_answer_l512_51253

noncomputable def x : ℝ := sorry

theorem x_pow_10_eq_correct_answer (h : x + (1 / x) = Real.sqrt 5) : 
  x^10 = (50 + 25 * Real.sqrt 5) / 2 := 
sorry

end x_pow_10_eq_correct_answer_l512_51253


namespace angle_between_hour_and_minute_hand_at_3_40_l512_51225

def angle_between_hands (hour minute : ℕ) : ℝ :=
  let minute_angle := (360 / 60) * minute
  let hour_angle := (360 / 12) + (30 / 60) * minute
  abs (minute_angle - hour_angle)

theorem angle_between_hour_and_minute_hand_at_3_40 : angle_between_hands 3 40 = 130 :=
by
  sorry

end angle_between_hour_and_minute_hand_at_3_40_l512_51225


namespace no_common_solution_l512_51201

theorem no_common_solution :
  ¬(∃ y : ℚ, (6 * y^2 + 11 * y - 1 = 0) ∧ (18 * y^2 + y - 1 = 0)) :=
by
  sorry

end no_common_solution_l512_51201


namespace eliminate_denominators_l512_51235

theorem eliminate_denominators (x : ℝ) :
  (4 * (2 * x - 1) - 3 * (3 * x - 4) = 12) ↔ ((2 * x - 1) / 3 - (3 * x - 4) / 4 = 1) := 
by
  sorry

end eliminate_denominators_l512_51235


namespace largest_N_with_square_in_base_nine_l512_51262

theorem largest_N_with_square_in_base_nine:
  ∃ N: ℕ, (9^2 ≤ N^2 ∧ N^2 < 9^3) ∧ ∀ M: ℕ, (9^2 ≤ M^2 ∧ M^2 < 9^3) → M ≤ N ∧ N = 26 := 
sorry

end largest_N_with_square_in_base_nine_l512_51262


namespace nathalie_total_coins_l512_51286

theorem nathalie_total_coins
  (quarters dimes nickels : ℕ)
  (ratio_condition : quarters = 9 * nickels ∧ dimes = 3 * nickels)
  (value_condition : 25 * quarters + 10 * dimes + 5 * nickels = 1820) :
  quarters + dimes + nickels = 91 :=
by
  sorry

end nathalie_total_coins_l512_51286


namespace AM_minus_GM_lower_bound_l512_51270

theorem AM_minus_GM_lower_bound (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x > y) : 
  (x + y) / 2 - Real.sqrt (x * y) ≥ (x - y)^2 / (8 * x) := 
by {
  sorry -- Proof to be filled in
}

end AM_minus_GM_lower_bound_l512_51270


namespace M_eq_N_l512_51278

def M (u : ℤ) : Prop := ∃ m n l : ℤ, u = 12 * m + 8 * n + 4 * l
def N (u : ℤ) : Prop := ∃ p q r : ℤ, u = 20 * p + 16 * q + 12 * r

theorem M_eq_N : ∀ u : ℤ, M u ↔ N u := by
  sorry

end M_eq_N_l512_51278


namespace smallest_with_20_divisors_is_144_l512_51252

def has_exactly_20_divisors (n : ℕ) : Prop :=
  let factors := n.factors;
  let divisors_count := factors.foldr (λ a b => (a + 1) * b) 1;
  divisors_count = 20

theorem smallest_with_20_divisors_is_144 : ∀ (n : ℕ), has_exactly_20_divisors n → (n < 144) → False :=
by
  sorry

end smallest_with_20_divisors_is_144_l512_51252


namespace number_of_common_tangents_l512_51244

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem number_of_common_tangents
  (C₁ C₂ : ℝ × ℝ) (r₁ r₂ : ℝ)
  (h₁ : ∀ (x y : ℝ), x^2 + y^2 - 2 * x = 0 → (C₁ = (1, 0)) ∧ (r₁ = 1))
  (h₂ : ∀ (x y : ℝ), x^2 + y^2 - 4 * y + 3 = 0 → (C₂ = (0, 2)) ∧ (r₂ = 1))
  (d : distance C₁ C₂ = Real.sqrt 5) :
  4 = 4 := 
by sorry

end number_of_common_tangents_l512_51244


namespace non_negative_sequence_l512_51283

theorem non_negative_sequence
  (a : Fin 100 → ℝ)
  (h₁ : a 0 = a 99)
  (h₂ : ∀ i : Fin 97, a i - 2 * a (i+1) + a (i+2) ≤ 0)
  (h₃ : a 0 ≥ 0) :
  ∀ i : Fin 100, a i ≥ 0 :=
by
  sorry

end non_negative_sequence_l512_51283


namespace choose_three_positive_or_two_negative_l512_51272

theorem choose_three_positive_or_two_negative (n : ℕ) (hn : n ≥ 3) (a : Fin n → ℝ) :
  ∃ (i j k : Fin n), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ (0 < a i + a j + a k) ∨ ∃ (i j : Fin n), i ≠ j ∧ (a i + a j < 0) := sorry

end choose_three_positive_or_two_negative_l512_51272


namespace carrots_total_l512_51259

-- Define the initial number of carrots Maria picked
def initial_carrots : ℕ := 685

-- Define the number of carrots Maria threw out
def thrown_out : ℕ := 156

-- Define the number of carrots Maria picked the next day
def picked_next_day : ℕ := 278

-- Define the total number of carrots Maria has after these actions
def total_carrots : ℕ :=
  initial_carrots - thrown_out + picked_next_day

-- The proof statement
theorem carrots_total : total_carrots = 807 := by
  sorry

end carrots_total_l512_51259


namespace length_of_body_diagonal_l512_51271

theorem length_of_body_diagonal (a b c : ℝ) 
  (h1 : 2 * (a * b + b * c + a * c) = 11)
  (h2 : 4 * (a + b + c) = 24) :
  (a^2 + b^2 + c^2).sqrt = 5 :=
by {
  -- proof to be filled
  sorry
}

end length_of_body_diagonal_l512_51271


namespace find_value_of_a_l512_51258

theorem find_value_of_a (a b : ℝ) (h1 : ∀ x, (2 < x ∧ x < 4) ↔ (a - b < x ∧ x < a + b)) : a = 3 := by
  sorry

end find_value_of_a_l512_51258


namespace exp_ineq_of_r_gt_one_l512_51250

theorem exp_ineq_of_r_gt_one {x r : ℝ} (hx : x > 0) (hr : r > 1) : (1 + x)^r > 1 + r * x :=
by
  sorry

end exp_ineq_of_r_gt_one_l512_51250


namespace roots_polynomial_l512_51206

noncomputable def roots_are (a b c : ℝ) : Prop :=
  a^3 - 18 * a^2 + 20 * a - 8 = 0 ∧ b^3 - 18 * b^2 + 20 * b - 8 = 0 ∧ c^3 - 18 * c^2 + 20 * c - 8 = 0

theorem roots_polynomial (a b c : ℝ) (h : roots_are a b c) : 
  (2 + a) * (2 + b) * (2 + c) = 128 :=
by
  sorry

end roots_polynomial_l512_51206


namespace cherries_left_l512_51288

def initial_cherries : ℕ := 77
def cherries_used : ℕ := 60

theorem cherries_left : initial_cherries - cherries_used = 17 := by
  sorry

end cherries_left_l512_51288


namespace total_votes_cast_l512_51213

-- Define the variables and constants
def total_votes (V : ℝ) : Prop :=
  let A := 0.32 * V
  let B := 0.28 * V
  let C := 0.22 * V
  let D := 0.18 * V
  -- Candidate A defeated Candidate B by 1200 votes
  0.32 * V - 0.28 * V = 1200 ∧
  -- Candidate A defeated Candidate C by 2200 votes
  0.32 * V - 0.22 * V = 2200 ∧
  -- Candidate B defeated Candidate D by 900 votes
  0.28 * V - 0.18 * V = 900

noncomputable def V := 30000

-- State the theorem
theorem total_votes_cast : total_votes V := by
  sorry

end total_votes_cast_l512_51213


namespace zoo_peacocks_l512_51260

theorem zoo_peacocks (R P : ℕ) (h1 : R + P = 60) (h2 : 4 * R + 2 * P = 192) : P = 24 :=
by
  sorry

end zoo_peacocks_l512_51260


namespace unit_prices_minimum_B_seedlings_l512_51217

-- Definition of the problem conditions and the results of Part 1
theorem unit_prices (x : ℝ) : 
  (1200 / (1.5 * x) + 10 = 900 / x) ↔ x = 10 :=
by
  sorry

-- Definition of the problem conditions and the result of Part 2
theorem minimum_B_seedlings (m : ℕ) : 
  (10 * m + 15 * (100 - m) ≤ 1314) ↔ m ≥ 38 :=
by
  sorry

end unit_prices_minimum_B_seedlings_l512_51217


namespace two_squares_inequality_l512_51209

theorem two_squares_inequality (a b : ℝ) : 2 * (a^2 + b^2) ≥ (a + b)^2 := 
sorry

end two_squares_inequality_l512_51209


namespace remaining_distance_l512_51220

-- Definitions based on the conditions
def total_distance : ℕ := 78
def first_leg : ℕ := 35
def second_leg : ℕ := 18

-- The theorem we want to prove
theorem remaining_distance : total_distance - (first_leg + second_leg) = 25 := by
  sorry

end remaining_distance_l512_51220


namespace zeroSeq_arithmetic_not_geometric_l512_51204

-- Define what it means for a sequence to be arithmetic
def isArithmeticSequence (seq : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, seq (n + 1) = seq n + d

-- Define what it means for a sequence to be geometric
def isGeometricSequence (seq : ℕ → ℝ) : Prop :=
  ∃ q, ∀ n, seq n ≠ 0 → seq (n + 1) = seq n * q

-- Define the sequence of zeros
def zeroSeq (n : ℕ) : ℝ := 0

theorem zeroSeq_arithmetic_not_geometric :
  isArithmeticSequence zeroSeq ∧ ¬ isGeometricSequence zeroSeq :=
by
  sorry

end zeroSeq_arithmetic_not_geometric_l512_51204


namespace each_group_has_two_bananas_l512_51241

theorem each_group_has_two_bananas (G T : ℕ) (hG : G = 196) (hT : T = 392) : T / G = 2 :=
by
  sorry

end each_group_has_two_bananas_l512_51241


namespace find_integer_solutions_xy_l512_51281

theorem find_integer_solutions_xy :
  ∀ (x y : ℕ), (x * y = x + y + 3) → (x, y) = (2, 5) ∨ (x, y) = (5, 2) ∨ (x, y) = (3, 3) := by
  intros x y h
  sorry

end find_integer_solutions_xy_l512_51281


namespace three_digit_number_uniq_l512_51292

theorem three_digit_number_uniq (n : ℕ) (h : 100 ≤ n ∧ n < 1000)
  (hundreds_digit : n / 100 = 5) (units_digit : n % 10 = 3)
  (div_by_9 : n % 9 = 0) : n = 513 :=
sorry

end three_digit_number_uniq_l512_51292


namespace students_count_l512_51256

theorem students_count (x : ℕ) (h1 : x / 2 + x / 4 + x / 7 + 3 = x) : x = 28 :=
  sorry

end students_count_l512_51256


namespace intersecting_absolute_value_functions_l512_51291

theorem intersecting_absolute_value_functions (a b c d : ℝ) (h1 : -|2 - a| + b = 5) (h2 : -|8 - a| + b = 3) (h3 : |2 - c| + d = 5) (h4 : |8 - c| + d = 3) (ha : 2 < a) (h8a : a < 8) (hc : 2 < c) (h8c : c < 8) : a + c = 10 :=
sorry

end intersecting_absolute_value_functions_l512_51291


namespace geometric_series_solution_l512_51261

-- Let a, r : ℝ be real numbers representing the parameters from the problem's conditions.
variables (a r : ℝ)

-- Define the conditions as hypotheses.
def condition1 : Prop := a / (1 - r) = 20
def condition2 : Prop := a / (1 - r^2) = 8

-- The theorem states that under these conditions, r equals 3/2.
theorem geometric_series_solution (hc1 : condition1 a r) (hc2 : condition2 a r) : r = 3 / 2 :=
sorry

end geometric_series_solution_l512_51261


namespace a4_binomial_coefficient_l512_51267

theorem a4_binomial_coefficient :
  ∀ (a_n a_1 a_2 a_3 a_4 a_5 : ℝ) (x : ℝ),
  (x^5 = a_n + a_1 * (x - 1) + a_2 * (x - 1)^2 + a_3 * (x - 1)^3 + a_4 * (x - 1)^4 + a_5 * (x - 1)^5) →
  (x^5 = (1 + (x - 1))^5) →
  a_4 = 5 :=
by
  intros a_n a_1 a_2 a_3 a_4 a_5 x hx1 hx2
  sorry

end a4_binomial_coefficient_l512_51267


namespace find_dallas_age_l512_51266

variable (Dallas_last_year Darcy_last_year Dexter_age Darcy_this_year Derek this_year_age : ℕ)

-- Conditions
axiom cond1 : Dallas_last_year = 3 * Darcy_last_year
axiom cond2 : Darcy_this_year = 2 * Dexter_age
axiom cond3 : Dexter_age = 8
axiom cond4 : Derek = this_year_age + 4

-- Theorem: Proving Dallas's current age
theorem find_dallas_age (Dallas_last_year : ℕ)
  (H1 : Dallas_last_year = 3 * (Darcy_this_year - 1))
  (H2 : Darcy_this_year = 2 * Dexter_age)
  (H3 : Dexter_age = 8)
  (H4 : Derek = (Dallas_last_year + 1) + 4) :
  Dallas_last_year + 1 = 46 :=
by
  sorry

end find_dallas_age_l512_51266


namespace range_of_a_l512_51238

variable {x a : ℝ}

def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0
def q (x : ℝ) : Prop := x^2 - 6*x + 8 > 0

theorem range_of_a (h : (∀ x a, p x a → q x) ∧ (∃ x a, q x ∧ ¬ p x a)) :
  a ≥ 4 ∨ (0 < a ∧ a ≤ 2/3) :=
sorry

end range_of_a_l512_51238


namespace snow_at_least_once_three_days_l512_51294

-- Define the probability of snow on a given day
def prob_snow : ℚ := 2 / 3

-- Define the event that it snows at least once in three days
def prob_snow_at_least_once_in_three_days : ℚ :=
  1 - (1 - prob_snow)^3

-- State the theorem
theorem snow_at_least_once_three_days : prob_snow_at_least_once_in_three_days = 26 / 27 :=
by
  sorry

end snow_at_least_once_three_days_l512_51294


namespace smallest_positive_integer_for_terminating_decimal_l512_51297

theorem smallest_positive_integer_for_terminating_decimal: ∃ n: ℕ, (n > 0) ∧ (∀ p : ℕ, (p ∣ (n + 150)) → (p=1 ∨ p=2 ∨ p=4 ∨ p=5 ∨ p=8 ∨ p=10 ∨ p=16 ∨ p=20 ∨ p=25 ∨ p=32 ∨ p=40 ∨ p=50 ∨ p=64 ∨ p=80 ∨ p=100 ∨ p=125 ∨ p=128 ∨ p=160)) ∧ n = 10 :=
by
  sorry

end smallest_positive_integer_for_terminating_decimal_l512_51297


namespace find_f_six_l512_51295

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_six (f : ℝ → ℝ)
  (h1 : ∀ x y : ℝ, x * f y = y * f x)
  (h2 : f 18 = 24) :
  f 6 = 8 :=
sorry

end find_f_six_l512_51295


namespace gloria_coins_l512_51265

theorem gloria_coins (qd qda qdc : ℕ) (h1 : qdc = 350) (h2 : qda = qdc / 5) (h3 : qd = qda - (2 * qda / 5)) :
  qd + qdc = 392 :=
by sorry

end gloria_coins_l512_51265


namespace solve_inequality_system_l512_51240

theorem solve_inequality_system :
  (∀ x : ℝ, (1 - 3 * (x - 1) < 8 - x) ∧ ((x - 3) / 2 + 2 ≥ x)) →
  ∃ (integers : Set ℤ), integers = {x : ℤ | -2 < (x : ℝ) ∧ (x : ℝ) ≤ 1} ∧ integers = {-1, 0, 1} :=
by
  sorry

end solve_inequality_system_l512_51240


namespace originally_planned_days_l512_51219

def man_days (men : ℕ) (days : ℕ) : ℕ := men * days

theorem originally_planned_days (D : ℕ) (h : man_days 5 10 = man_days 10 D) : D = 5 :=
by 
  sorry

end originally_planned_days_l512_51219


namespace triangle_inequality_l512_51290

theorem triangle_inequality (a b c R r : ℝ) 
  (habc : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_area1 : a * b * c = 4 * R * S)
  (h_area2 : S = r * (a + b + c) / 2) :
  (b^2 + c^2) / (2 * b * c) ≤ R / (2 * r) := 
sorry

end triangle_inequality_l512_51290


namespace max_M_is_7524_l512_51233

-- Define the conditions
def is_valid_t (t : ℕ) : Prop :=
  let a := t / 1000
  let b := (t % 1000) / 100
  let c := (t % 100) / 10
  let d := t % 10
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
  a + c = 9 ∧
  b - d = 1 ∧
  (2 * (2 * a + d)) % (2 * b + c) = 0

-- Define function M
def M (a b c d : ℕ) : ℕ := 2000 * a + 100 * b + 10 * c + d

-- Define the maximum value of M
def max_valid_M : ℕ :=
  let m_values := [5544, 7221, 7322, 7524]
  m_values.foldl max 0

theorem max_M_is_7524 : max_valid_M = 7524 := by
  -- The proof would be written here. For now, we indicate the theorem as
  -- not yet proven.
  sorry

end max_M_is_7524_l512_51233


namespace basketball_team_total_players_l512_51227

theorem basketball_team_total_players (total_points : ℕ) (min_points : ℕ) (max_points : ℕ) (team_size : ℕ)
  (h1 : total_points = 100)
  (h2 : min_points = 7)
  (h3 : max_points = 23)
  (h4 : ∀ (n : ℕ), n ≥ min_points)
  (h5 : max_points = 23)
  : team_size = 12 :=
sorry

end basketball_team_total_players_l512_51227


namespace average_salary_rest_l512_51249

theorem average_salary_rest (total_workers : ℕ) (avg_salary_all : ℝ)
  (num_technicians : ℕ) (avg_salary_technicians : ℝ) :
  total_workers = 21 →
  avg_salary_all = 8000 →
  num_technicians = 7 →
  avg_salary_technicians = 12000 →
  (avg_salary_all * total_workers - avg_salary_technicians * num_technicians) / (total_workers - num_technicians) = 6000 :=
by intros h1 h2 h3 h4; sorry

end average_salary_rest_l512_51249
