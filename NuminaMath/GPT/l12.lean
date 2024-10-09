import Mathlib

namespace right_triangle_legs_solutions_l12_1265

theorem right_triangle_legs_solutions (R r : ℝ) (h_cond : R / r ≥ 1 + Real.sqrt 2) :
  ∃ (a b : ℝ), 
    a = r + R + Real.sqrt (R^2 - 2 * r * R - r^2) ∧ 
    b = r + R - Real.sqrt (R^2 - 2 * r * R - r^2) ∧ 
    (2 * R)^2 = a^2 + b^2 := by
  sorry

end right_triangle_legs_solutions_l12_1265


namespace hunter_rats_l12_1254

-- Defining the conditions
variable (H : ℕ) (E : ℕ := H + 30) (K : ℕ := 3 * (H + E)) 
  
-- Defining the total number of rats condition
def total_rats : Prop := H + E + K = 200

-- Defining the goal: Prove Hunter has 10 rats
theorem hunter_rats (h : total_rats H) : H = 10 := by
  sorry

end hunter_rats_l12_1254


namespace peach_cost_l12_1278

theorem peach_cost 
  (total_fruits : ℕ := 32)
  (total_cost : ℕ := 52)
  (plum_cost : ℕ := 2)
  (num_plums : ℕ := 20)
  (cost_peach : ℕ) :
  (total_cost - (num_plums * plum_cost)) = cost_peach * (total_fruits - num_plums) →
  cost_peach = 1 :=
by
  intro h
  sorry

end peach_cost_l12_1278


namespace expand_expression_l12_1255

theorem expand_expression (x y z : ℝ) :
  (2 * x + 15) * (3 * y + 20 * z + 25) = 
  6 * x * y + 40 * x * z + 50 * x + 45 * y + 300 * z + 375 :=
by
  sorry

end expand_expression_l12_1255


namespace find_numbers_l12_1294

theorem find_numbers 
  (x y z : ℕ) 
  (h1 : y = 2 * x - 3) 
  (h2 : x + y = 51) 
  (h3 : z = 4 * x - y) : 
  x = 18 ∧ y = 33 ∧ z = 39 :=
by sorry

end find_numbers_l12_1294


namespace max_value_M_l12_1246

def J_k (k : ℕ) : ℕ := 10^(k + 3) + 1600

def M (k : ℕ) : ℕ := (J_k k).factors.count 2

theorem max_value_M : ∃ k > 0, (M k) = 7 ∧ ∀ m > 0, M m ≤ 7 :=
by 
  sorry

end max_value_M_l12_1246


namespace problem_value_eq_13_l12_1206

theorem problem_value_eq_13 : 8 / 4 - 3^2 + 4 * 5 = 13 :=
by
  sorry

end problem_value_eq_13_l12_1206


namespace sum_of_medians_bounds_l12_1203

theorem sum_of_medians_bounds (a b c m_a m_b m_c : ℝ) 
    (h1 : m_a < (b + c) / 2)
    (h2 : m_b < (a + c) / 2)
    (h3 : m_c < (a + b) / 2)
    (h4 : ∀a b c : ℝ, a + b > c) :
    (3 / 4) * (a + b + c) < m_a + m_b + m_c ∧ m_a + m_b + m_c < a + b + c := 
by
  sorry

end sum_of_medians_bounds_l12_1203


namespace average_age_combined_rooms_l12_1290

theorem average_age_combined_rooms :
  (8 * 30 + 5 * 22) / (8 + 5) = 26.9 := by
  sorry

end average_age_combined_rooms_l12_1290


namespace ratio_largest_smallest_root_geometric_progression_l12_1273

theorem ratio_largest_smallest_root_geometric_progression (a b c d : ℤ)
  (h_poly : a * x^3 + b * x^2 + c * x + d = 0) 
  (h_in_geo_prog : ∃ r1 r2 r3 q : ℝ, r1 < r2 ∧ r2 < r3 ∧ r1 * q = r2 ∧ r2 * q = r3 ∧ q ≠ 0) : 
  ∃ R : ℝ, R = 1 := 
by
  sorry

end ratio_largest_smallest_root_geometric_progression_l12_1273


namespace prove_sum_eq_9_l12_1213

theorem prove_sum_eq_9 (a b : ℝ) (h : i * (a - i) = b - (2 * i) ^ 3) : a + b = 9 :=
by
  sorry

end prove_sum_eq_9_l12_1213


namespace new_paint_intensity_l12_1296

theorem new_paint_intensity : 
  let I_original : ℝ := 0.5
  let I_added : ℝ := 0.2
  let replacement_fraction : ℝ := 1 / 3
  let remaining_fraction : ℝ := 2 / 3
  let I_new := remaining_fraction * I_original + replacement_fraction * I_added
  I_new = 0.4 :=
by
  -- sorry is used to skip the actual proof
  sorry

end new_paint_intensity_l12_1296


namespace company_employees_count_l12_1285

theorem company_employees_count :
  (females : ℕ) ->
  (advanced_degrees : ℕ) ->
  (college_degree_only_males : ℕ) ->
  (advanced_degrees_females : ℕ) ->
  (110 = females) ->
  (90 = advanced_degrees) ->
  (35 = college_degree_only_males) ->
  (55 = advanced_degrees_females) ->
  (females - advanced_degrees_females + college_degree_only_males + advanced_degrees = 180) :=
by
  intros females advanced_degrees college_degree_only_males advanced_degrees_females
  intro h_females h_advanced_degrees h_college_degree_only_males h_advanced_degrees_females
  sorry

end company_employees_count_l12_1285


namespace find_x0_l12_1220

noncomputable def f (x : ℝ) : ℝ :=
if 0 ≤ x ∧ x ≤ 2 then x^2 - 4 else 2 * x

theorem find_x0 (x0 : ℝ) (h : f x0 = 8) : x0 = 4 := by
  sorry

end find_x0_l12_1220


namespace factor_81_minus_27_x_cubed_l12_1227

theorem factor_81_minus_27_x_cubed (x : ℝ) : 
  81 - 27 * x ^ 3 = 27 * (3 - x) * (9 + 3 * x + x ^ 2) :=
by sorry

end factor_81_minus_27_x_cubed_l12_1227


namespace fraction_meaningful_condition_l12_1271

theorem fraction_meaningful_condition (x : ℝ) : (∃ y, y = 1 / (x - 3)) ↔ x ≠ 3 :=
by
  sorry

end fraction_meaningful_condition_l12_1271


namespace total_cars_l12_1258

theorem total_cars (yesterday today : ℕ) (h_yesterday : yesterday = 60) (h_today : today = 2 * yesterday) : yesterday + today = 180 := 
sorry

end total_cars_l12_1258


namespace solve_system1_l12_1263

theorem solve_system1 (x y : ℝ) :
  x + y + 3 = 10 ∧ 4 * (x + y) - y = 25 →
  x = 4 ∧ y = 3 :=
by
  sorry

end solve_system1_l12_1263


namespace points_comparison_l12_1284

def quadratic_function (m x : ℝ) : ℝ :=
  (x + m - 3) * (x - m) + 3

def point_on_graph (m x y : ℝ) : Prop :=
  y = quadratic_function m x

theorem points_comparison (m x1 x2 y1 y2 : ℝ)
  (h1 : point_on_graph m x1 y1)
  (h2 : point_on_graph m x2 y2)
  (hx : x1 < x2)
  (h_sum : x1 + x2 < 3) :
  y1 > y2 := 
  sorry

end points_comparison_l12_1284


namespace find_function_expression_l12_1274

variable (f : ℝ → ℝ)
variable (P : ℝ → ℝ → ℝ)

-- conditions
axiom a1 : f 1 = 1
axiom a2 : ∀ (x y : ℝ), f (x + y) = f x + f y + 2 * y * (x + y) + 1

-- proof statement
theorem find_function_expression (x : ℕ) (h : x ≠ 0) : f x = x^2 + 3*x - 3 := sorry

end find_function_expression_l12_1274


namespace trapezoid_area_eq_c_l12_1231

theorem trapezoid_area_eq_c (b c : ℝ) (hb : b = Real.sqrt c) (hc : 0 < c) :
    let shorter_base := b - 3
    let altitude := b
    let longer_base := b + 3
    let K := (1/2) * (shorter_base + longer_base) * altitude
    K = c :=
by
    sorry

end trapezoid_area_eq_c_l12_1231


namespace smallest_yellow_marbles_l12_1269

theorem smallest_yellow_marbles :
  ∃ n : ℕ, (n ≡ 0 [MOD 20]) ∧
           (∃ b : ℕ, b = n / 4) ∧
           (∃ r : ℕ, r = n / 5) ∧
           (∃ g : ℕ, g = 10) ∧
           (∃ y : ℕ, y = n - (b + r + g) ∧ y = 1) :=
sorry

end smallest_yellow_marbles_l12_1269


namespace intersection_correct_l12_1261

open Set

def M : Set ℤ := {0, 1, 2, -1}
def N : Set ℤ := {0, 1, 2, 3}

theorem intersection_correct : M ∩ N = {0, 1, 2} :=
by 
  -- Proof omitted
  sorry

end intersection_correct_l12_1261


namespace isosceles_triangle_base_length_l12_1283

theorem isosceles_triangle_base_length (a b : ℝ) (h1 : a = 3 ∨ b = 3) (h2 : a + a + b = 15 ∨ a + b + b = 15) :
  b = 3 := 
sorry

end isosceles_triangle_base_length_l12_1283


namespace part1_l12_1291

def f (x : ℝ) := x^2 - 2*x

theorem part1 (x : ℝ) :
  (|f x| + |x^2 + 2*x| ≥ 6*|x|) ↔ (x ≤ -3 ∨ 3 ≤ x ∨ x = 0) :=
sorry

end part1_l12_1291


namespace largest_of_five_consecutive_odd_integers_with_product_93555_l12_1210

theorem largest_of_five_consecutive_odd_integers_with_product_93555 : 
  ∃ n, (n * (n + 2) * (n + 4) * (n + 6) * (n + 8) = 93555) ∧ (n + 8 = 19) :=
sorry

end largest_of_five_consecutive_odd_integers_with_product_93555_l12_1210


namespace volume_conversion_l12_1212

-- Define the given conditions
def V_feet : ℕ := 216
def C_factor : ℕ := 27

-- State the theorem to prove
theorem volume_conversion : V_feet / C_factor = 8 :=
  sorry

end volume_conversion_l12_1212


namespace number_of_sections_l12_1275

def total_seats : ℕ := 270
def seats_per_section : ℕ := 30

theorem number_of_sections : total_seats / seats_per_section = 9 := 
by sorry

end number_of_sections_l12_1275


namespace description_of_T_l12_1201

-- Define the set T
def T : Set (ℝ × ℝ) := 
  {p | (p.1 = 1 ∧ p.2 ≤ 9) ∨ (p.2 = 9 ∧ p.1 ≤ 1) ∨ (p.2 = p.1 + 8 ∧ p.1 ≥ 1)}

-- State the formal proof problem: T is three rays with a common point
theorem description_of_T :
  (∃ p : ℝ × ℝ, p = (1, 9) ∧ 
    ∀ q ∈ T, 
      (q.1 = 1 ∧ q.2 ≤ 9) ∨ 
      (q.2 = 9 ∧ q.1 ≤ 1) ∨ 
      (q.2 = q.1 + 8 ∧ q.1 ≥ 1)) :=
by
  sorry

end description_of_T_l12_1201


namespace max_value_of_4x_plus_3y_l12_1286

theorem max_value_of_4x_plus_3y (x y : ℝ) (h : x^2 + y^2 = 18 * x + 8 * y + 10) :
  4 * x + 3 * y ≤ 45 :=
sorry

end max_value_of_4x_plus_3y_l12_1286


namespace lizz_team_loses_by_8_points_l12_1222

-- Definitions of the given conditions
def initial_deficit : ℕ := 20
def free_throw_points : ℕ := 5 * 1
def three_pointer_points : ℕ := 3 * 3
def jump_shot_points : ℕ := 4 * 2
def liz_points : ℕ := free_throw_points + three_pointer_points + jump_shot_points
def other_team_points : ℕ := 10
def points_caught_up : ℕ := liz_points - other_team_points
def final_deficit : ℕ := initial_deficit - points_caught_up

-- Theorem proving Liz's team loses by 8 points
theorem lizz_team_loses_by_8_points : final_deficit = 8 :=
  by
    -- Proof will be here
    sorry

end lizz_team_loses_by_8_points_l12_1222


namespace nat_know_albums_l12_1205

/-- Define the number of novels, comics, documentaries and crates properties --/
def novels := 145
def comics := 271
def documentaries := 419
def crates := 116
def items_per_crate := 9

/-- Define the total capacity of crates --/
def total_capacity := crates * items_per_crate

/-- Define the total number of other items --/
def other_items := novels + comics + documentaries

/-- Define the number of albums --/
def albums := total_capacity - other_items

/-- Theorem: Prove that the number of albums is equal to 209 --/
theorem nat_know_albums : albums = 209 := by
  sorry

end nat_know_albums_l12_1205


namespace sum_of_cube_faces_l12_1236

theorem sum_of_cube_faces :
  ∃ (a b c d e f : ℕ), 
    (a = 12) ∧ 
    (b = a + 3) ∧ 
    (c = b + 3) ∧ 
    (d = c + 3) ∧ 
    (e = d + 3) ∧ 
    (f = e + 3) ∧ 
    (a + f = 39) ∧ 
    (b + e = 39) ∧ 
    (c + d = 39) ∧ 
    (a + b + c + d + e + f = 117) :=
by
  let a := 12
  let b := a + 3
  let c := b + 3
  let d := c + 3
  let e := d + 3
  let f := e + 3
  have h1 : a + f = 39 := sorry
  have h2 : b + e = 39 := sorry
  have h3 : c + d = 39 := sorry
  have sum : a + b + c + d + e + f = 117 := sorry
  exact ⟨a, b, c, d, e, f, rfl, rfl, rfl, rfl, rfl, rfl, h1, h2, h3, sum⟩

end sum_of_cube_faces_l12_1236


namespace linear_function_intersects_x_axis_at_two_units_l12_1200

theorem linear_function_intersects_x_axis_at_two_units (k : ℝ) :
  (∃ x : ℝ, y = k * x + 2 ∧ y = 0 ∧ |x| = 2) ↔ k = 1 ∨ k = -1 :=
by
  sorry

end linear_function_intersects_x_axis_at_two_units_l12_1200


namespace determinant_identity_l12_1277

variable (a b : ℝ)

theorem determinant_identity :
  Matrix.det ![
      ![1, Real.sin (a - b), Real.sin a],
      ![Real.sin (a - b), 1, Real.sin b],
      ![Real.sin a, Real.sin b, 1]
  ] = 0 :=
by sorry

end determinant_identity_l12_1277


namespace number_of_interviewees_l12_1216

theorem number_of_interviewees (n : ℕ) (h : (6 : ℚ) / (n * (n - 1)) = 1 / 70) : n = 21 :=
sorry

end number_of_interviewees_l12_1216


namespace montoya_budget_l12_1225

def percentage_food (groceries: ℝ) (eating_out: ℝ) : ℝ :=
  groceries + eating_out

def percentage_transportation_rent_utilities (transportation: ℝ) (rent: ℝ) (utilities: ℝ) : ℝ :=
  transportation + rent + utilities

def total_percentage (food: ℝ) (transportation_rent_utilities: ℝ) : ℝ :=
  food + transportation_rent_utilities

theorem montoya_budget :
  ∀ (groceries : ℝ) (eating_out : ℝ) (transportation : ℝ) (rent : ℝ) (utilities : ℝ),
    groceries = 0.6 → eating_out = 0.2 → transportation = 0.1 → rent = 0.05 → utilities = 0.05 →
    total_percentage (percentage_food groceries eating_out) (percentage_transportation_rent_utilities transportation rent utilities) = 1 :=
by
sorry

end montoya_budget_l12_1225


namespace principal_amount_l12_1252

theorem principal_amount (P R : ℝ) (h1 : P + (P * R * 2) / 100 = 780) (h2 : P + (P * R * 7) / 100 = 1020) : P = 684 := 
sorry

end principal_amount_l12_1252


namespace max_value_l12_1281

theorem max_value (x y : ℝ) : 
  (x + 3 * y + 4) / (Real.sqrt (x ^ 2 + y ^ 2 + 4)) ≤ Real.sqrt 26 :=
by
  -- Proof should be here
  sorry

end max_value_l12_1281


namespace total_washer_dryer_cost_l12_1287

def washer_cost : ℕ := 710
def dryer_cost : ℕ := washer_cost - 220

theorem total_washer_dryer_cost :
  washer_cost + dryer_cost = 1200 :=
  by sorry

end total_washer_dryer_cost_l12_1287


namespace three_digit_number_formed_by_1198th_1200th_digits_l12_1251

def albertSequenceDigit (n : ℕ) : ℕ :=
  -- Define the nth digit in Albert's sequence
  sorry

theorem three_digit_number_formed_by_1198th_1200th_digits :
  let d1198 := albertSequenceDigit 1198
  let d1199 := albertSequenceDigit 1199
  let d1200 := albertSequenceDigit 1200
  (d1198 * 100 + d1199 * 10 + d1200) = 220 :=
by
  sorry

end three_digit_number_formed_by_1198th_1200th_digits_l12_1251


namespace arithmetic_sequence_max_n_pos_sum_l12_1235

noncomputable def max_n (a : ℕ → ℤ) (d : ℤ) : ℕ :=
  8

theorem arithmetic_sequence_max_n_pos_sum
  (a : ℕ → ℤ)
  (d : ℤ)
  (h_arith_seq : ∀ n, a (n+1) = a 1 + n * d)
  (h_a1 : a 1 > 0)
  (h_a4_a5_sum_pos : a 4 + a 5 > 0)
  (h_a4_a5_prod_neg : a 4 * a 5 < 0) :
  max_n a d = 8 := by
  sorry

end arithmetic_sequence_max_n_pos_sum_l12_1235


namespace sale_price_relationship_l12_1214

/-- Elaine's Gift Shop increased the original prices of all items by 10% 
  and then offered a 30% discount on these new prices in a clearance sale 
  - proving the relationship between the final sale price and the original price of an item -/

theorem sale_price_relationship (p : ℝ) : 
  (0.7 * (1.1 * p) = 0.77 * p) :=
by 
  sorry

end sale_price_relationship_l12_1214


namespace polynomial_roots_property_l12_1288

theorem polynomial_roots_property (a b : ℝ) (h : ∀ x, x^2 + x - 2024 = 0 → x = a ∨ x = b) : 
  a^2 + 2 * a + b = 2023 :=
by
  sorry

end polynomial_roots_property_l12_1288


namespace real_root_of_system_l12_1230

theorem real_root_of_system :
  (∃ x : ℝ, x^3 + 9 = 0 ∧ x + 3 = 0) ↔ x = -3 := 
by 
  sorry

end real_root_of_system_l12_1230


namespace coeff_x3_in_product_l12_1226

open Polynomial

noncomputable def p : Polynomial ℤ := 3 * X^3 + 2 * X^2 + 5 * X + 3
noncomputable def q : Polynomial ℤ := 4 * X^3 + 5 * X^2 + 6 * X + 8

theorem coeff_x3_in_product :
  (p * q).coeff 3 = 61 :=
by sorry

end coeff_x3_in_product_l12_1226


namespace product_of_base_9_digits_of_9876_l12_1292

def base9_digits (n : ℕ) : List ℕ := 
  let rec digits_aux (n : ℕ) (acc : List ℕ) : List ℕ :=
    if n = 0 then acc else digits_aux (n / 9) ((n % 9) :: acc)
  digits_aux n []

def product (lst : List ℕ) : ℕ := lst.foldl (· * ·) 1

theorem product_of_base_9_digits_of_9876 :
  product (base9_digits 9876) = 192 :=
by 
  sorry

end product_of_base_9_digits_of_9876_l12_1292


namespace abs_neg_one_fourth_l12_1267

theorem abs_neg_one_fourth : |(- (1 / 4))| = (1 / 4) :=
by
  sorry

end abs_neg_one_fourth_l12_1267


namespace rectangle_lengths_correct_l12_1241

-- Definitions of the parameters and their relationships
noncomputable def AB := 1200
noncomputable def BC := 150
noncomputable def AB_ext := AB
noncomputable def BC_ext := BC + 350
noncomputable def CD := AB
noncomputable def DA := BC

-- Definitions of the calculated distances using the conditions
noncomputable def AP := Real.sqrt (AB^2 + BC_ext^2)
noncomputable def PD := Real.sqrt (BC_ext^2 + AB^2)

-- Using similarity of triangles for PQ and CQ
noncomputable def PQ := (350 / 500) * AP
noncomputable def CQ := (350 / 500) * AB

-- The theorem to prove the final results
theorem rectangle_lengths_correct :
    AP = 1300 ∧
    PD = 1250 ∧
    PQ = 910 ∧
    CQ = 840 :=
    by
    sorry

end rectangle_lengths_correct_l12_1241


namespace sum_first_ten_multiples_of_nine_l12_1250

theorem sum_first_ten_multiples_of_nine :
  let a := 9
  let d := 9
  let n := 10
  let S_n := n * (2 * a + (n - 1) * d) / 2
  S_n = 495 := 
by
  sorry

end sum_first_ten_multiples_of_nine_l12_1250


namespace ruel_usable_stamps_l12_1221

def totalStamps (books10 books15 books25 books30 : ℕ) (stamps10 stamps15 stamps25 stamps30 : ℕ) : ℕ :=
  books10 * stamps10 + books15 * stamps15 + books25 * stamps25 + books30 * stamps30

def damagedStamps (damaged25 damaged30 : ℕ) : ℕ :=
  damaged25 + damaged30

def usableStamps (books10 books15 books25 books30 stamps10 stamps15 stamps25 stamps30 damaged25 damaged30 : ℕ) : ℕ :=
  totalStamps books10 books15 books25 books30 stamps10 stamps15 stamps25 stamps30 - damagedStamps damaged25 damaged30

theorem ruel_usable_stamps :
  usableStamps 4 6 3 2 10 15 25 30 5 3 = 257 := by
  sorry

end ruel_usable_stamps_l12_1221


namespace breadth_of_rectangle_l12_1276

theorem breadth_of_rectangle (b l : ℝ) (h1 : l * b = 24 * b) (h2 : l - b = 10) : b = 14 :=
by
  sorry

end breadth_of_rectangle_l12_1276


namespace race_time_diff_l12_1245

-- Define the speeds and race distance
def Malcolm_speed : ℕ := 5  -- in minutes per mile
def Joshua_speed : ℕ := 7   -- in minutes per mile
def Alice_speed : ℕ := 6    -- in minutes per mile
def race_distance : ℕ := 12 -- in miles

-- Calculate times
def Malcolm_time : ℕ := Malcolm_speed * race_distance
def Joshua_time : ℕ := Joshua_speed * race_distance
def Alice_time : ℕ := Alice_speed * race_distance

-- Lean 4 statement to prove the time differences
theorem race_time_diff :
  Joshua_time - Malcolm_time = 24 ∧ Alice_time - Malcolm_time = 12 := by
  sorry

end race_time_diff_l12_1245


namespace solution_set_inequality_l12_1249

theorem solution_set_inequality (x : ℝ) : (1 < x ∧ x < 3) ↔ (x^2 - 4*x + 3 < 0) :=
by sorry

end solution_set_inequality_l12_1249


namespace phone_calls_to_reach_Davina_l12_1234

theorem phone_calls_to_reach_Davina : 
  (∀ (a b : ℕ), (0 ≤ a ∧ a < 10) ∧ (0 ≤ b ∧ b < 10)) → (least_num_calls : ℕ) = 100 :=
by
  sorry

end phone_calls_to_reach_Davina_l12_1234


namespace steven_name_day_44_l12_1202

def W (n : ℕ) : ℕ :=
  2 * (n / 2) + 4 * ((n - 1) / 2)

theorem steven_name_day_44 : ∃ n : ℕ, W n = 44 :=
  by 
  existsi 16
  sorry

end steven_name_day_44_l12_1202


namespace reduced_price_per_dozen_is_3_l12_1282

variable (P : ℝ) -- original price of an apple
variable (R : ℝ) -- reduced price of an apple
variable (A : ℝ) -- number of apples originally bought for Rs. 40
variable (cost_per_dozen_reduced : ℝ) -- reduced price per dozen apples

-- Define the conditions
axiom reduction_condition : R = 0.60 * P
axiom apples_bought_condition : 40 = A * P
axiom more_apples_condition : 40 = (A + 64) * R

-- Define the proof problem
theorem reduced_price_per_dozen_is_3 : cost_per_dozen_reduced = 3 :=
by
  sorry

end reduced_price_per_dozen_is_3_l12_1282


namespace average_salary_of_employees_l12_1279

theorem average_salary_of_employees (A : ℝ)
  (h1 : 24 * A + 11500 = 25 * (A + 400)) :
  A = 1500 := 
by
  sorry

end average_salary_of_employees_l12_1279


namespace unique_zero_location_l12_1260

theorem unique_zero_location (f : ℝ → ℝ) (h : ∃! x, f x = 0 ∧ 1 < x ∧ x < 3) :
  ¬ (∃ x, 2 < x ∧ x < 5 ∧ f x = 0) :=
sorry

end unique_zero_location_l12_1260


namespace biggest_number_l12_1204

noncomputable def Yoongi_collected : ℕ := 4
noncomputable def Jungkook_collected : ℕ := 6 * 3
noncomputable def Yuna_collected : ℕ := 5

theorem biggest_number :
  Jungkook_collected = 18 ∧ Jungkook_collected > Yoongi_collected ∧ Jungkook_collected > Yuna_collected :=
by
  sorry

end biggest_number_l12_1204


namespace seeds_total_l12_1218

theorem seeds_total (wednesday_seeds thursday_seeds : ℕ) (h_wed : wednesday_seeds = 20) (h_thu : thursday_seeds = 2) : (wednesday_seeds + thursday_seeds) = 22 := by
  sorry

end seeds_total_l12_1218


namespace book_cost_l12_1272

-- Define the problem parameters
variable (p : ℝ) -- cost of one book in dollars

-- Conditions given in the problem
def seven_copies_cost_less_than_15 (p : ℝ) : Prop := 7 * p < 15
def eleven_copies_cost_more_than_22 (p : ℝ) : Prop := 11 * p > 22

-- The theorem stating the cost is between the given bounds
theorem book_cost (p : ℝ) (h1 : seven_copies_cost_less_than_15 p) (h2 : eleven_copies_cost_more_than_22 p) : 
    2 < p ∧ p < (15 / 7 : ℝ) :=
sorry

end book_cost_l12_1272


namespace total_oranges_l12_1228

def monday_oranges : ℕ := 100
def tuesday_oranges : ℕ := 3 * monday_oranges
def wednesday_oranges : ℕ := 70

theorem total_oranges : monday_oranges + tuesday_oranges + wednesday_oranges = 470 := by
  sorry

end total_oranges_l12_1228


namespace mass_percentage_C_in_CO_l12_1243

noncomputable def atomic_mass_C : ℚ := 12.01
noncomputable def atomic_mass_O : ℚ := 16.00
noncomputable def molecular_mass_CO : ℚ := atomic_mass_C + atomic_mass_O

theorem mass_percentage_C_in_CO : (atomic_mass_C / molecular_mass_CO) * 100 = 42.88 :=
by
  have atomic_mass_C_div_total : atomic_mass_C / molecular_mass_CO = 12.01 / 28.01 := sorry
  have mass_percentage : (atomic_mass_C / molecular_mass_CO) * 100 = 42.88 := sorry
  exact mass_percentage

end mass_percentage_C_in_CO_l12_1243


namespace monomials_like_terms_l12_1239

theorem monomials_like_terms (a b : ℕ) (h1 : 3 = a) (h2 : 4 = 2 * b) : a = 3 ∧ b = 2 :=
by
  sorry

end monomials_like_terms_l12_1239


namespace mowing_field_time_l12_1215

theorem mowing_field_time (h1 : (1 / 28 : ℝ) = (3 / 84 : ℝ))
                         (h2 : (1 / 84 : ℝ) = (1 / 84 : ℝ))
                         (h3 : (1 / 28 + 1 / 84 : ℝ) = (1 / 21 : ℝ)) :
                         21 = 1 / ((1 / 28) + (1 / 84)) := 
by {
  sorry
}

end mowing_field_time_l12_1215


namespace sum_of_coefficients_l12_1233

-- Given polynomial definition
def P (x : ℝ) : ℝ := (1 + x - 3 * x^2) ^ 1965

-- Lean 4 statement for the proof problem
theorem sum_of_coefficients :
  P 1 = -1 :=
by
  -- Proof placeholder
  sorry

end sum_of_coefficients_l12_1233


namespace number_of_technicians_l12_1297

/-- 
In a workshop, the average salary of all the workers is Rs. 8000. 
The average salary of some technicians is Rs. 12000 and the average salary of the rest is Rs. 6000. 
The total number of workers in the workshop is 24.
Prove that there are 8 technicians in the workshop.
-/
theorem number_of_technicians 
  (total_workers : ℕ) 
  (avg_salary_all : ℕ) 
  (avg_salary_technicians : ℕ) 
  (avg_salary_rest : ℕ) 
  (num_technicians rest_workers : ℕ) 
  (h_total : total_workers = num_technicians + rest_workers)
  (h_avg_salary : (num_technicians * avg_salary_technicians + rest_workers * avg_salary_rest) = total_workers * avg_salary_all)
  (h1 : total_workers = 24)
  (h2 : avg_salary_all = 8000)
  (h3 : avg_salary_technicians = 12000)
  (h4 : avg_salary_rest = 6000) :
  num_technicians = 8 :=
by
  sorry

end number_of_technicians_l12_1297


namespace multiplication_solution_l12_1262

theorem multiplication_solution 
  (x : ℤ) 
  (h : 72517 * x = 724807415) : 
  x = 9999 := 
sorry

end multiplication_solution_l12_1262


namespace compute_a_plus_b_l12_1247

theorem compute_a_plus_b (a b : ℝ) (h : ∃ (u v w : ℕ), u ≠ v ∧ v ≠ w ∧ u ≠ w ∧ u + v + w = 8 ∧ u * v * w = b ∧ u * v + v * w + w * u = a) : 
  a + b = 27 :=
by
  -- The proof is omitted.
  sorry

end compute_a_plus_b_l12_1247


namespace magnitude_of_z_l12_1208

open Complex

theorem magnitude_of_z (z : ℂ) (h : z^2 + Complex.normSq z = 4 - 7 * Complex.I) : 
  Complex.normSq z = 65 / 8 := 
by
  sorry

end magnitude_of_z_l12_1208


namespace rebus_decrypt_correct_l12_1219

-- Definitions
def is_digit (d : ℕ) : Prop := 0 ≤ d ∧ d ≤ 9
def is_odd (d : ℕ) : Prop := is_digit d ∧ d % 2 = 1
def is_even (d : ℕ) : Prop := is_digit d ∧ d % 2 = 0

-- Variables representing ċharacters H, Ч (C), A, D, Y, E, F, B, K
variables (H C A D Y E F B K : ℕ)

-- Conditions
axiom H_odd : is_odd H
axiom C_even : is_even C
axiom A_even : is_even A
axiom D_odd : is_odd D
axiom Y_even : is_even Y
axiom E_even : is_even E
axiom F_odd : is_odd F
axiom B_digit : is_digit B
axiom K_odd : is_odd K

-- Correct answers
def H_val : ℕ := 5
def C_val : ℕ := 3
def A_val : ℕ := 2
def D_val : ℕ := 9
def Y_val : ℕ := 8
def E_val : ℕ := 8
def F_val : ℕ := 5
def B_any : ℕ := B
def K_val : ℕ := 5

-- Proof statement
theorem rebus_decrypt_correct : 
  H = H_val ∧
  C = C_val ∧
  A = A_val ∧
  D = D_val ∧
  Y = Y_val ∧
  E = E_val ∧
  F = F_val ∧
  K = K_val :=
sorry

end rebus_decrypt_correct_l12_1219


namespace smallest_n_contains_constant_term_l12_1211

theorem smallest_n_contains_constant_term :
  ∃ n : ℕ, (∀ x : ℝ, x ≠ 0 → (2 * x^3 + 1 / x^(1/2))^n = c ↔ n = 7) :=
by
  sorry

end smallest_n_contains_constant_term_l12_1211


namespace whisky_replacement_l12_1270

variable (V x : ℝ)

/-- The initial whisky in the jar contains 40% alcohol -/
def initial_volume_of_alcohol (V : ℝ) : ℝ := 0.4 * V

/-- A part (x liters) of this whisky is replaced by another containing 19% alcohol -/
def volume_replaced_whisky (x : ℝ) : ℝ := x
def remaining_whisky (V x : ℝ) : ℝ := V - x

/-- The percentage of alcohol in the jar after replacement is 24% -/
def final_volume_of_alcohol (V x : ℝ) : ℝ := 0.4 * (remaining_whisky V x) + 0.19 * (volume_replaced_whisky x)

/- Prove that the quantity of whisky replaced is 0.16/0.21 times the total volume -/
theorem whisky_replacement :
  final_volume_of_alcohol V x = 0.24 * V → x = (0.16 / 0.21) * V :=
by sorry

end whisky_replacement_l12_1270


namespace lottery_most_frequent_number_l12_1209

noncomputable def m (i : ℕ) : ℚ :=
  ((i - 1) * (90 - i) * (89 - i) * (88 - i)) / 6

theorem lottery_most_frequent_number :
  ∀ (i : ℕ), 2 ≤ i ∧ i ≤ 87 → m 23 ≥ m i :=
by 
  sorry -- Proof goes here. This placeholder allows the file to compile.

end lottery_most_frequent_number_l12_1209


namespace range_of_m_l12_1298

def f (x : ℝ) : ℝ := -x^3 - 2*x^2 + 4*x

theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Icc (-3 : ℝ) 3, f x ≥ m^2 - 14 * m) ↔ 3 ≤ m ∧ m ≤ 11 :=
by
  sorry

end range_of_m_l12_1298


namespace quadratic_roots_l12_1257

theorem quadratic_roots (a b c : ℝ) (h1 : a ≠ 0) (h2 : a - b + c = 0) (h3 : (b^2 - 4 * a * c) = 0) : 2 * a - b = 0 :=
by {
  sorry
}

end quadratic_roots_l12_1257


namespace connected_distinct_points_with_slope_change_l12_1280

-- Defining the cost function based on the given conditions
def cost_function (n : ℕ) : ℕ := 
  if n <= 10 then 20 * n else 18 * n

-- The main theorem to prove the nature of the graph as described in the problem
theorem connected_distinct_points_with_slope_change : 
  (∀ n, (1 ≤ n ∧ n ≤ 20) → 
    (∃ k, cost_function n = k ∧ 
    (n <= 10 → cost_function n = 20 * n) ∧ 
    (n > 10 → cost_function n = 18 * n))) ∧
  (∃ n, n = 10 ∧ cost_function n = 200 ∧ cost_function (n + 1) = 198) :=
sorry

end connected_distinct_points_with_slope_change_l12_1280


namespace solve_system_l12_1264

-- Define the conditions from the problem
def system_of_equations (x y : ℝ) : Prop :=
  (x = 4 * y) ∧ (x + 2 * y = -12)

-- Define the solution we want to prove
def solution (x y : ℝ) : Prop :=
  (x = -8) ∧ (y = -2)

-- State the theorem
theorem solve_system :
  ∃ x y : ℝ, system_of_equations x y ∧ solution x y :=
by 
  sorry

end solve_system_l12_1264


namespace color_guard_team_row_length_l12_1293

theorem color_guard_team_row_length (n : ℕ) (p d : ℝ)
  (h_n : n = 40)
  (h_p : p = 0.4)
  (h_d : d = 0.5) :
  (n - 1) * d + n * p = 35.5 :=
by
  sorry

end color_guard_team_row_length_l12_1293


namespace perpendicular_bisectors_intersect_at_one_point_l12_1289

-- Define the key geometric concepts
variables {Point : Type*} [MetricSpace Point]

-- Define the given conditions 
variables (A B C M : Point)
variables (h1 : dist M A = dist M B)
variables (h2 : dist M B = dist M C)

-- Define the theorem to be proven
theorem perpendicular_bisectors_intersect_at_one_point :
  dist M A = dist M C :=
by 
  -- Proof to be filled in later
  sorry

end perpendicular_bisectors_intersect_at_one_point_l12_1289


namespace perimeter_smallest_square_l12_1256

theorem perimeter_smallest_square 
  (d : ℝ) (side_largest : ℝ)
  (h1 : d = 3) 
  (h2 : side_largest = 22) : 
  4 * (side_largest - 2 * d - 2 * d) = 40 := by
  sorry

end perimeter_smallest_square_l12_1256


namespace cannot_form_complex_pattern_l12_1299

structure GeometricPieces where
  triangles : Nat
  squares : Nat

def possibleToForm (pieces : GeometricPieces) : Bool :=
  sorry -- Since the formation logic is unknown, it is incomplete.

theorem cannot_form_complex_pattern : 
  let pieces := GeometricPieces.mk 8 7
  ¬ possibleToForm pieces = true := 
sorry

end cannot_form_complex_pattern_l12_1299


namespace maximize_area_of_quadrilateral_l12_1217

theorem maximize_area_of_quadrilateral (k : ℝ) (h0 : 0 < k) (h1 : k < 1) 
    (hE : ∀ E : ℝ, E = 2 * k) (hF : ∀ F : ℝ, F = 2 * k) :
    k = 1/2 ∧ (2 * (1 - k) ^ 2) = 1/2 := 
by 
  sorry

end maximize_area_of_quadrilateral_l12_1217


namespace history_book_pages_l12_1224

-- Conditions
def science_pages : ℕ := 600
def novel_pages (science: ℕ) : ℕ := science / 4
def history_pages (novel: ℕ) : ℕ := novel * 2

-- Theorem to prove
theorem history_book_pages : history_pages (novel_pages science_pages) = 300 :=
by
  sorry

end history_book_pages_l12_1224


namespace angle_bisector_correct_length_l12_1240

-- Define the isosceles triangle with the given conditions
structure IsoscelesTriangle :=
  (base : ℝ)
  (lateral : ℝ)
  (is_isosceles : lateral = 20 ∧ base = 5)

-- Define the problem of finding the angle bisector
noncomputable def angle_bisector_length (tri : IsoscelesTriangle) : ℝ :=
  6

-- The main theorem to state the problem
theorem angle_bisector_correct_length (tri : IsoscelesTriangle) : 
  angle_bisector_length tri = 6 :=
by
  -- We state the theorem, skipping the proof (sorry)
  sorry

end angle_bisector_correct_length_l12_1240


namespace geom_series_eq_l12_1237

noncomputable def C (n : ℕ) := 256 * (1 - 1 / (4^n)) / (3 / 4)
noncomputable def D (n : ℕ) := 1024 * (1 - 1 / ((-2)^n)) / (3 / 2)

theorem geom_series_eq (n : ℕ) (h : n ≥ 1) : C n = D n ↔ n = 1 :=
by
  sorry

end geom_series_eq_l12_1237


namespace largest_of_7_consecutive_numbers_with_average_20_l12_1229

variable (n : ℤ) 

theorem largest_of_7_consecutive_numbers_with_average_20
  (h_avg : (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6))/7 = 20) : 
  (n + 6) = 23 :=
by
  -- Placeholder for the actual proof
  sorry

end largest_of_7_consecutive_numbers_with_average_20_l12_1229


namespace sum_of_cubes_eq_zero_l12_1207

theorem sum_of_cubes_eq_zero (a b : ℝ) (h1 : a + b = 0) (h2 : a * b = -4) : a^3 + b^3 = 0 :=
sorry

end sum_of_cubes_eq_zero_l12_1207


namespace truck_cargo_solution_l12_1295

def truck_cargo_problem (x : ℝ) (n : ℕ) : Prop :=
  (∀ (x : ℝ) (n : ℕ), x = (x / n - 0.5) * (n + 4)) ∧ (55 ≤ x ∧ x ≤ 64)

theorem truck_cargo_solution :
  ∃ y : ℝ, y = 2.5 :=
sorry

end truck_cargo_solution_l12_1295


namespace wire_cut_problem_l12_1266

variable (x : ℝ)

theorem wire_cut_problem 
  (h₁ : x + (5 / 2) * x = 49) : x = 14 :=
by
  sorry

end wire_cut_problem_l12_1266


namespace max_true_statements_l12_1242

theorem max_true_statements :
  ∃ x : ℝ, 
  (0 < x ∧ x < 1) ∧ -- Statement 4
  (0 < x^3 ∧ x^3 < 1) ∧ -- Statement 1
  (0 < x - x^3 ∧ x - x^3 < 1) ∧ -- Statement 5
  ¬(x^3 > 1) ∧ -- Not Statement 2
  ¬(-1 < x ∧ x < 0) := -- Not Statement 3
sorry

end max_true_statements_l12_1242


namespace exists_positive_m_f99_divisible_1997_l12_1232

def f (x : ℕ) : ℕ := 3 * x + 2

noncomputable
def higher_order_f (n : ℕ) (x : ℕ) : ℕ :=
  match n with
  | 0 => sorry  -- placeholder since f^0 isn't defined in this context
  | 1 => f x
  | k + 1 => f (higher_order_f k x)

theorem exists_positive_m_f99_divisible_1997 :
  ∃ m : ℕ, m > 0 ∧ higher_order_f 99 m % 1997 = 0 :=
sorry

end exists_positive_m_f99_divisible_1997_l12_1232


namespace probability_diff_colors_l12_1223

/-!
There are 5 identical balls, including 3 white balls and 2 black balls. 
If 2 balls are drawn at once, the probability of the event "the 2 balls have different colors" 
occurring is \( \frac{3}{5} \).
-/

theorem probability_diff_colors 
    (white_balls : ℕ) (black_balls : ℕ) (total_balls : ℕ) (drawn_balls : ℕ) 
    (h_white : white_balls = 3) (h_black : black_balls = 2) (h_total : total_balls = 5) (h_drawn : drawn_balls = 2) :
    let total_ways := Nat.choose total_balls drawn_balls
    let diff_color_ways := (Nat.choose white_balls 1) * (Nat.choose black_balls 1)
    (diff_color_ways : ℚ) / (total_ways : ℚ) = 3 / 5 := 
by
    -- Step 1: Calculate total ways to draw 2 balls out of 5
    -- total_ways = 10 (by binomial coefficient)
    -- Step 2: Calculate favorable outcomes (1 white, 1 black)
    -- diff_color_ways = 6
    -- Step 3: Calculate probability
    -- Probability = 6 / 10 = 3 / 5
    sorry

end probability_diff_colors_l12_1223


namespace seating_arrangements_family_van_correct_l12_1268

noncomputable def num_seating_arrangements (parents : Fin 2) (children : Fin 3) : Nat :=
  let perm3_2 := Nat.factorial 3 / Nat.factorial (3 - 2)
  2 * 1 * perm3_2

theorem seating_arrangements_family_van_correct :
  num_seating_arrangements 2 3 = 12 :=
by
  sorry

end seating_arrangements_family_van_correct_l12_1268


namespace probability_no_defective_pencils_l12_1253

theorem probability_no_defective_pencils :
  let total_pencils := 6
  let defective_pencils := 2
  let pencils_chosen := 3
  let non_defective_pencils := total_pencils - defective_pencils
  let total_ways := Nat.choose total_pencils pencils_chosen
  let non_defective_ways := Nat.choose non_defective_pencils pencils_chosen
  (non_defective_ways / total_ways : ℚ) = 1 / 5 :=
by
  sorry

end probability_no_defective_pencils_l12_1253


namespace find_third_number_x_l12_1238

variable {a b : ℝ}

theorem find_third_number_x (h : a < b) :
  (∃ x : ℝ, x = a * b / (2 * b - a) ∧ x < a) ∨ 
  (∃ x : ℝ, x = 2 * a * b / (a + b) ∧ a < x ∧ x < b) ∨ 
  (∃ x : ℝ, x = a * b / (2 * a - b) ∧ a < b ∧ b < x) :=
sorry

end find_third_number_x_l12_1238


namespace blending_marker_drawings_correct_l12_1259

-- Define the conditions
def total_drawings : ℕ := 25
def colored_pencil_drawings : ℕ := 14
def charcoal_drawings : ℕ := 4

-- Define the target proof statement
def blending_marker_drawings : ℕ := total_drawings - (colored_pencil_drawings + charcoal_drawings)

-- Proof goal
theorem blending_marker_drawings_correct : blending_marker_drawings = 7 := by
  sorry

end blending_marker_drawings_correct_l12_1259


namespace chess_tournament_l12_1244

-- Define the number of chess amateurs
def num_amateurs : ℕ := 5

-- Define the number of games each amateur plays
def games_per_amateur : ℕ := 4

-- Define the total number of chess games possible
def total_games : ℕ := num_amateurs * (num_amateurs - 1) / 2

-- The main statement to prove
theorem chess_tournament : total_games = 10 := 
by
  -- here should be the proof, but according to the task, we use sorry to skip
  sorry

end chess_tournament_l12_1244


namespace intersection_product_distance_eq_eight_l12_1248

noncomputable def parametricCircle : ℝ → ℝ × ℝ :=
  λ θ => (4 * Real.cos θ, 4 * Real.sin θ)

noncomputable def parametricLine : ℝ → ℝ × ℝ :=
  λ t => (2 + (1 / 2) * t, 2 + (Real.sqrt 3 / 2) * t)

theorem intersection_product_distance_eq_eight :
  ∀ θ t,
    let (x1, y1) := parametricCircle θ
    let (x2, y2) := parametricLine t
    (x1^2 + y1^2 = 16) ∧ (x2 = x1 ∧ y2 = y1) →
    ∃ t1 t2,
      x1 = 2 + (1 / 2) * t1 ∧ y1 = 2 + (Real.sqrt 3 / 2) * t1 ∧
      x1 = 2 + (1 / 2) * t2 ∧ y1 = 2 + (Real.sqrt 3 / 2) * t2 ∧
      (t1 * t2 = -8) ∧ (|t1 * t2| = 8) := 
by
  intros θ t
  dsimp only
  intro h
  sorry

end intersection_product_distance_eq_eight_l12_1248
