import Mathlib

namespace correct_sum_is_826_l1202_120215

theorem correct_sum_is_826 (ABC : ℕ)
  (h1 : 100 ≤ ABC ∧ ABC < 1000)  -- Ensuring ABC is a three-digit number
  (h2 : ∃ A B C : ℕ, ABC = 100 * A + 10 * B + C ∧ C = 6) -- Misread ones digit is 6
  (incorrect_sum : ℕ)
  (h3 : incorrect_sum = ABC + 57)  -- Sum obtained by Yoongi was 823
  (h4 : incorrect_sum = 823) : ABC + 57 + 3 = 826 :=  -- Correcting the sum considering the 6 to 9 error
by
  sorry

end correct_sum_is_826_l1202_120215


namespace minimum_f_l1202_120204

def f (x : ℝ) : ℝ := |3 - x| + |x - 7|

theorem minimum_f : ∀ x : ℝ, min (f x) = 4 := sorry

end minimum_f_l1202_120204


namespace female_athletes_drawn_l1202_120241

theorem female_athletes_drawn (total_athletes male_athletes female_athletes sample_size : ℕ)
  (h_total : total_athletes = male_athletes + female_athletes)
  (h_team : male_athletes = 48 ∧ female_athletes = 36)
  (h_sample_size : sample_size = 35) :
  (female_athletes * sample_size) / total_athletes = 15 :=
by
  sorry

end female_athletes_drawn_l1202_120241


namespace distance_C_distance_BC_l1202_120200

variable (A B C D : ℕ)

theorem distance_C
  (hA : A = 350)
  (hAB : A + B = 600)
  (hABCD : A + B + C + D = 1500)
  (hD : D = 275)
  : C = 625 :=
by
  sorry

theorem distance_BC
  (A B C D : ℕ)
  (hA : A = 350)
  (hAB : A + B = 600)
  (hABCD : A + B + C + D = 1500)
  (hD : D = 275)
  : B + C = 875 :=
by
  sorry

end distance_C_distance_BC_l1202_120200


namespace find_F_l1202_120205

theorem find_F (F C : ℝ) (h1 : C = 35) (h2 : C = (7/12) * (F - 40)) : F = 100 :=
by
  sorry

end find_F_l1202_120205


namespace Joan_video_game_expense_l1202_120274

theorem Joan_video_game_expense : 
  let basketball_price := 5.20
  let racing_price := 4.23
  let action_price := 7.12
  let discount_rate := 0.10
  let sales_tax_rate := 0.06
  let discounted_basketball_price := basketball_price * (1 - discount_rate)
  let discounted_racing_price := racing_price * (1 - discount_rate)
  let discounted_action_price := action_price * (1 - discount_rate)
  let total_cost_before_tax := discounted_basketball_price + discounted_racing_price + discounted_action_price
  let sales_tax := total_cost_before_tax * sales_tax_rate
  let total_cost := total_cost_before_tax + sales_tax
  total_cost = 15.79 :=
by
  sorry

end Joan_video_game_expense_l1202_120274


namespace scientific_notation_of_1206_million_l1202_120276

theorem scientific_notation_of_1206_million :
  (1206 * 10^6 : ℝ) = 1.206 * 10^7 :=
by
  sorry

end scientific_notation_of_1206_million_l1202_120276


namespace distance_between_parallel_lines_l1202_120278

theorem distance_between_parallel_lines :
  let a := 4
  let b := -3
  let c1 := 2
  let c2 := -1
  let d := (abs (c1 - c2)) / (Real.sqrt (a^2 + b^2))
  d = 3 / 5 :=
by
  sorry

end distance_between_parallel_lines_l1202_120278


namespace total_apples_purchased_l1202_120273

theorem total_apples_purchased (M : ℝ) (T : ℝ) (W : ℝ) 
    (hM : M = 15.5)
    (hT : T = 3.2 * M)
    (hW : W = 1.05 * T) :
    M + T + W = 117.18 := by
  sorry

end total_apples_purchased_l1202_120273


namespace pow_div_pow_l1202_120234

variable (a : ℝ)
variable (A B : ℕ)

theorem pow_div_pow (a : ℝ) (A B : ℕ) : a^A / a^B = a^(A - B) :=
  sorry

example : a^6 / a^2 = a^4 :=
  pow_div_pow a 6 2

end pow_div_pow_l1202_120234


namespace lines_intersection_l1202_120253

theorem lines_intersection (n c : ℝ) : 
    (∀ x y : ℝ, y = n * x + 5 → y = 4 * x + c → (x, y) = (8, 9)) → 
    n + c = -22.5 := 
by
    intro h
    sorry

end lines_intersection_l1202_120253


namespace angie_bought_18_pretzels_l1202_120221

theorem angie_bought_18_pretzels
  (B : ℕ := 12) -- Barry bought 12 pretzels
  (S : ℕ := B / 2) -- Shelly bought half as many pretzels as Barry
  (A : ℕ := 3 * S) -- Angie bought three times as many pretzels as Shelly
  : A = 18 := sorry

end angie_bought_18_pretzels_l1202_120221


namespace point_in_fourth_quadrant_l1202_120242

theorem point_in_fourth_quadrant (m : ℝ) : (m-1 > 0 ∧ 2-m < 0) ↔ m > 2 :=
by
  sorry

end point_in_fourth_quadrant_l1202_120242


namespace initial_students_per_class_l1202_120240

theorem initial_students_per_class
  (S : ℕ) 
  (parents chaperones left_students left_chaperones : ℕ)
  (teachers remaining_individuals : ℕ)
  (h1 : parents = 5)
  (h2 : chaperones = 2)
  (h3 : left_students = 10)
  (h4 : left_chaperones = 2)
  (h5 : teachers = 2)
  (h6 : remaining_individuals = 15)
  (h7 : 2 * S + parents + teachers - left_students - left_chaperones = remaining_individuals) :
  S = 10 :=
by
  sorry

end initial_students_per_class_l1202_120240


namespace first_divisor_l1202_120293

theorem first_divisor (y : ℝ) (x : ℝ) (h1 : 320 / (y * 3) = x) (h2 : x = 53.33) : y = 2 :=
sorry

end first_divisor_l1202_120293


namespace corrected_mean_l1202_120285

theorem corrected_mean (incorrect_mean : ℕ) (num_observations : ℕ) (wrong_value actual_value : ℕ) : 
  (50 * 36 + (43 - 23)) / 50 = 36.4 :=
by
  sorry

end corrected_mean_l1202_120285


namespace diagonal_length_of_rhombus_l1202_120219

-- Definitions for the conditions
def side_length_of_square : ℝ := 8
def area_of_square : ℝ := side_length_of_square ^ 2
def area_of_rhombus : ℝ := 64
def d2 : ℝ := 8
-- Question
theorem diagonal_length_of_rhombus (d1 : ℝ) : (d1 * d2) / 2 = area_of_rhombus ↔ d1 = 16 := by
  sorry

end diagonal_length_of_rhombus_l1202_120219


namespace investment_ratio_correct_l1202_120290

variable (P Q : ℝ)
variable (investment_ratio: ℝ := 7 / 5)
variable (profit_ratio: ℝ := 7 / 10)
variable (time_p: ℝ := 7)
variable (time_q: ℝ := 14)

theorem investment_ratio_correct :
  (P * time_p) / (Q * time_q) = profit_ratio → (P / Q) = investment_ratio := 
by
  sorry

end investment_ratio_correct_l1202_120290


namespace markers_blue_l1202_120243

theorem markers_blue {total_markers red_markers blue_markers : ℝ} 
  (h_total : total_markers = 64.0) 
  (h_red : red_markers = 41.0) 
  (h_blue : blue_markers = total_markers - red_markers) : 
  blue_markers = 23.0 := 
by 
  sorry

end markers_blue_l1202_120243


namespace max_min_difference_l1202_120245

variable (x y z : ℝ)

theorem max_min_difference :
  x + y + z = 3 →
  x^2 + y^2 + z^2 = 18 →
  (max z (-z)) - ((min z (-z))) = 6 :=
  by
    intros h1 h2
    sorry

end max_min_difference_l1202_120245


namespace range_of_m_l1202_120214

theorem range_of_m (x m : ℝ) :
  (∀ x, (x - 1) / 2 ≥ (x - 2) / 3 → 2 * x - m ≥ x → x ≥ m) ↔ m ≥ -1 := by
  sorry

end range_of_m_l1202_120214


namespace jose_investment_l1202_120229

theorem jose_investment (P T : ℝ) (X : ℝ) (months_tom months_jose : ℝ) (profit_total profit_jose profit_tom : ℝ) :
  T = 30000 →
  months_tom = 12 →
  months_jose = 10 →
  profit_total = 54000 →
  profit_jose = 30000 →
  profit_tom = profit_total - profit_jose →
  profit_tom / profit_jose = (T * months_tom) / (X * months_jose) →
  X = 45000 :=
by sorry

end jose_investment_l1202_120229


namespace mass_of_substance_l1202_120299

-- The conditions
def substance_density (mass_cubic_meter_kg : ℝ) (volume_cubic_meter_cm3 : ℝ) : Prop :=
  mass_cubic_meter_kg = 100 ∧ volume_cubic_meter_cm3 = 1*1000000

def specific_amount_volume_cm3 (volume_cm3 : ℝ) : Prop :=
  volume_cm3 = 10

-- The Proof Statement
theorem mass_of_substance (mass_cubic_meter_kg : ℝ) (volume_cubic_meter_cm3 : ℝ) (volume_cm3 : ℝ) (mass_grams : ℝ) :
  substance_density mass_cubic_meter_kg volume_cubic_meter_cm3 →
  specific_amount_volume_cm3 volume_cm3 →
  mass_grams = 10 :=
by
  intros hDensity hVolume
  sorry

end mass_of_substance_l1202_120299


namespace min_max_expression_l1202_120225

theorem min_max_expression (x : ℝ) (h : 2 ≤ x ∧ x ≤ 7) :
  ∃ (a : ℝ) (b : ℝ), a = 11 / 3 ∧ b = 87 / 16 ∧ 
  (∀ y, 2 ≤ y ∧ y ≤ 7 → 11 / 3 ≤ (y^2 + 4*y + 10) / (2*y + 2)) ∧
  (∀ y, 2 ≤ y ∧ y ≤ 7 → (y^2 + 4*y + 10) / (2*y + 2) ≤ 87 / 16) :=
sorry

end min_max_expression_l1202_120225


namespace value_of_expression_l1202_120228

-- Defining the given conditions as Lean definitions
def x : ℚ := 2 / 3
def y : ℚ := 5 / 2

-- The theorem statement to prove that the given expression equals the correct answer
theorem value_of_expression : (1 / 3) * x^7 * y^6 = 125 / 261 :=
by
  sorry

end value_of_expression_l1202_120228


namespace total_pieces_of_junk_mail_l1202_120267

def houses : ℕ := 6
def pieces_per_house : ℕ := 4

theorem total_pieces_of_junk_mail : houses * pieces_per_house = 24 :=
by 
  sorry

end total_pieces_of_junk_mail_l1202_120267


namespace completion_days_for_B_l1202_120270

-- Conditions
def A_completion_days := 20
def B_completion_days (x : ℕ) := x
def project_completion_days := 20
def A_work_days := project_completion_days - 10
def B_work_days := project_completion_days
def A_work_rate := 1 / A_completion_days
def B_work_rate (x : ℕ) := 1 / B_completion_days x
def combined_work_rate (x : ℕ) := A_work_rate + B_work_rate x
def A_project_completed := A_work_days * A_work_rate
def B_project_remaining (x : ℕ) := 1 - A_project_completed
def B_project_completion (x : ℕ) := B_work_days * B_work_rate x

-- Proof statement
theorem completion_days_for_B (x : ℕ) 
  (h : B_project_completion x = B_project_remaining x ∧ combined_work_rate x > 0) :
  x = 40 :=
sorry

end completion_days_for_B_l1202_120270


namespace smallest_abs_value_l1202_120218

theorem smallest_abs_value : 
    ∀ (a b c d : ℝ), 
    a = -1/2 → b = -2/3 → c = 4 → d = -5 → 
    abs a < abs b ∧ abs a < abs c ∧ abs a < abs d := 
by
  intros a b c d ha hb hc hd
  rw [ha, hb, hc, hd]
  simp
  -- Proof omitted for brevity
  sorry

end smallest_abs_value_l1202_120218


namespace total_balls_in_box_l1202_120213

theorem total_balls_in_box :
  ∀ (W B R : ℕ), 
    W = 16 →
    B = W + 12 →
    R = 2 * B →
    W + B + R = 100 :=
by
  intros W B R hW hB hR
  sorry

end total_balls_in_box_l1202_120213


namespace greatest_possible_length_l1202_120288

theorem greatest_possible_length (a b c : ℕ) (h1 : a = 28) (h2 : b = 45) (h3 : c = 63) : 
  Nat.gcd (Nat.gcd a b) c = 7 :=
by
  sorry

end greatest_possible_length_l1202_120288


namespace find_baking_soda_boxes_l1202_120233

-- Define the quantities and costs
def num_flour_boxes := 3
def cost_per_flour_box := 3
def num_egg_trays := 3
def cost_per_egg_tray := 10
def num_milk_liters := 7
def cost_per_milk_liter := 5
def baking_soda_cost_per_box := 3
def total_cost := 80

-- Define the total cost of flour, eggs, and milk
def total_flour_cost := num_flour_boxes * cost_per_flour_box
def total_egg_cost := num_egg_trays * cost_per_egg_tray
def total_milk_cost := num_milk_liters * cost_per_milk_liter

-- Define the total cost of non-baking soda items
def total_non_baking_soda_cost := total_flour_cost + total_egg_cost + total_milk_cost

-- Define the remaining cost for baking soda
def baking_soda_total_cost := total_cost - total_non_baking_soda_cost

-- Define the number of baking soda boxes
def num_baking_soda_boxes := baking_soda_total_cost / baking_soda_cost_per_box

theorem find_baking_soda_boxes : num_baking_soda_boxes = 2 :=
by
  sorry

end find_baking_soda_boxes_l1202_120233


namespace pq_logic_l1202_120224

theorem pq_logic (p q : Prop) (h1 : p ∨ q) (h2 : ¬p) : ¬p ∧ q :=
by
  sorry

end pq_logic_l1202_120224


namespace fraction_tabs_closed_l1202_120287

theorem fraction_tabs_closed (x : ℝ) (h₁ : 400 * (1 - x) * (3/5) * (1/2) = 90) : 
  x = 1 / 4 :=
by
  have := h₁
  sorry

end fraction_tabs_closed_l1202_120287


namespace polynomial_abs_sum_roots_l1202_120248

theorem polynomial_abs_sum_roots (p q r m : ℤ) (h1 : p + q + r = 0) (h2 : p * q + q * r + r * p = -2500) (h3 : p * q * r = -m) :
  |p| + |q| + |r| = 100 :=
sorry

end polynomial_abs_sum_roots_l1202_120248


namespace min_voters_for_tall_24_l1202_120212

/-
There are 105 voters divided into 5 districts, each district divided into 7 sections, with each section having 3 voters.
A section is won by a majority vote. A district is won by a majority of sections. The contest is won by a majority of districts.
Tall won the contest. Prove that the minimum number of voters who could have voted for Tall is 24.
-/
noncomputable def min_voters_for_tall (total_voters districts sections voters_per_section : ℕ) (sections_needed_to_win_district districts_needed_to_win_contest : ℕ) : ℕ :=
  let voters_needed_per_section := voters_per_section / 2 + 1
  sections_needed_to_win_district * districts_needed_to_win_contest * voters_needed_per_section

theorem min_voters_for_tall_24 :
  min_voters_for_tall 105 5 7 3 4 3 = 24 :=
sorry

end min_voters_for_tall_24_l1202_120212


namespace road_width_l1202_120282

theorem road_width
  (road_length : ℝ) 
  (truckload_area : ℝ) 
  (truckload_cost : ℝ) 
  (sales_tax : ℝ) 
  (total_cost : ℝ) :
  road_length = 2000 ∧
  truckload_area = 800 ∧
  truckload_cost = 75 ∧
  sales_tax = 0.20 ∧
  total_cost = 4500 →
  ∃ width : ℝ, width = 20 :=
by
  sorry

end road_width_l1202_120282


namespace find_n_l1202_120255

theorem find_n (n : ℕ) :
  Int.lcm n 16 = 52 ∧ Nat.gcd n 16 = 8 → n = 26 :=
by
  sorry

end find_n_l1202_120255


namespace find_c_value_l1202_120284

theorem find_c_value :
  ∃ c : ℝ, (∀ x y : ℝ, (x + 10) ^ 2 + (y + 4) ^ 2 = 169 ∧ (x - 3) ^ 2 + (y - 9) ^ 2 = 65 → x + y = c) ∧ c = 3 :=
sorry

end find_c_value_l1202_120284


namespace minimum_monkeys_required_l1202_120275

theorem minimum_monkeys_required (total_weight : ℕ) (weapon_max_weight : ℕ) (monkey_max_capacity : ℕ) 
  (num_monkeys : ℕ) (total_weapons : ℕ) 
  (H1 : total_weight = 600) 
  (H2 : weapon_max_weight = 30) 
  (H3 : monkey_max_capacity = 50) 
  (H4 : total_weapons = 600 / 30) 
  (H5 : num_monkeys = 23) : 
  num_monkeys ≤ (total_weapons * weapon_max_weight) / monkey_max_capacity :=
sorry

end minimum_monkeys_required_l1202_120275


namespace boxes_with_neither_l1202_120281

-- Definitions translating the conditions from the problem
def total_boxes : Nat := 15
def boxes_with_markers : Nat := 8
def boxes_with_crayons : Nat := 4
def boxes_with_both : Nat := 3

-- The theorem statement to prove
theorem boxes_with_neither : total_boxes - (boxes_with_markers + boxes_with_crayons - boxes_with_both) = 6 := by
  -- Proof will go here
  sorry

end boxes_with_neither_l1202_120281


namespace find_xy_l1202_120231

noncomputable def xy_value (x y : ℝ) := x * y

theorem find_xy :
  ∃ x y : ℝ, (x + y = 2) ∧ (x^2 * y^3 + y^2 * x^3 = 32) ∧ xy_value x y = -8 :=
by
  sorry

end find_xy_l1202_120231


namespace length_of_book_l1202_120292

theorem length_of_book (A W L : ℕ) (hA : A = 50) (hW : W = 10) (hArea : A = L * W) : L = 5 := 
sorry

end length_of_book_l1202_120292


namespace parallelogram_sides_l1202_120227

theorem parallelogram_sides (x y : ℝ) 
  (h1 : 5 * x - 7 = 14) 
  (h2 : 3 * y + 4 = 8 * y - 3) : 
  x + y = 5.6 :=
sorry

end parallelogram_sides_l1202_120227


namespace find_x_l1202_120217

-- Definitions for the vectors a and b
def a : ℝ × ℝ := (3, 4)
def b : ℝ × ℝ := (2, 1)

-- Definition for the condition of parallel vectors
def are_parallel (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.2 = v1.2 * v2.1

-- Mathematical statement to prove
theorem find_x (x : ℝ) 
  (h_parallel : are_parallel (a.1 + x * b.1, a.2 + x * b.2) (a.1 - b.1, a.2 - b.2)) : 
  x = -1 :=
sorry

end find_x_l1202_120217


namespace sum_of_number_and_preceding_l1202_120277

theorem sum_of_number_and_preceding (n : ℤ) (h : 6 * n - 2 = 100) : n + (n - 1) = 33 :=
by {
  sorry
}

end sum_of_number_and_preceding_l1202_120277


namespace elizabeth_time_l1202_120244

-- Defining the conditions
def tom_time_minutes : ℕ := 120
def time_ratio : ℕ := 4

-- Proving Elizabeth's time
theorem elizabeth_time : tom_time_minutes / time_ratio = 30 := 
by
  sorry

end elizabeth_time_l1202_120244


namespace certain_number_equation_l1202_120271

theorem certain_number_equation (x : ℤ) (h : 16 * x + 17 * x + 20 * x + 11 = 170) : x = 3 :=
by {
  sorry
}

end certain_number_equation_l1202_120271


namespace quadratic_sum_l1202_120289

theorem quadratic_sum (b c : ℝ) : 
  (∀ x : ℝ, x^2 - 24 * x + 50 = (x + b)^2 + c) → b + c = -106 :=
by
  intro h
  sorry

end quadratic_sum_l1202_120289


namespace tax_percentage_l1202_120216

-- Definitions
def salary_before_taxes := 5000
def rent_expense_per_month := 1350
def total_late_rent_payments := 2 * rent_expense_per_month
def fraction_of_next_salary_after_taxes := (3 / 5 : ℚ)

-- Main statement to prove
theorem tax_percentage (T : ℚ) : 
  fraction_of_next_salary_after_taxes * (salary_before_taxes - (T / 100) * salary_before_taxes) = total_late_rent_payments → 
  T = 10 :=
by
  sorry

end tax_percentage_l1202_120216


namespace dvds_rented_l1202_120249

def total_cost : ℝ := 4.80
def cost_per_dvd : ℝ := 1.20

theorem dvds_rented : total_cost / cost_per_dvd = 4 := 
by
  sorry

end dvds_rented_l1202_120249


namespace valid_k_range_l1202_120260

noncomputable def fx (k : ℝ) (x : ℝ) : ℝ :=
  k * x^2 + k * x + k + 3

theorem valid_k_range:
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 3 → fx k x ≥ 0) ↔ (k ≥ -3 / 13) :=
by
  sorry

end valid_k_range_l1202_120260


namespace cot_sum_simplified_l1202_120280

noncomputable def cot (x : ℝ) : ℝ := (Real.cos x) / (Real.sin x)

theorem cot_sum_simplified : cot (π / 24) + cot (π / 8) = 96 / (π^2) := 
by 
  sorry

end cot_sum_simplified_l1202_120280


namespace exponential_monotonicity_example_l1202_120265

theorem exponential_monotonicity_example (m n : ℕ) (a b : ℝ) (h1 : a = 0.2 ^ m) (h2 : b = 0.2 ^ n) (h3 : m > n) : a < b :=
by
  sorry

end exponential_monotonicity_example_l1202_120265


namespace quadratic_rational_solutions_product_l1202_120266

theorem quadratic_rational_solutions_product :
  ∃ (c₁ c₂ : ℕ), (7 * x^2 + 15 * x + c₁ = 0 ∧ 225 - 28 * c₁ = k^2 ∧ ∃ k : ℤ, k^2 = 225 - 28 * c₁) ∧
                 (7 * x^2 + 15 * x + c₂ = 0 ∧ 225 - 28 * c₂ = k^2 ∧ ∃ k : ℤ, k^2 = 225 - 28 * c₂) ∧
                 (c₁ = 1) ∧ (c₂ = 8) ∧ (c₁ * c₂ = 8) :=
by
  sorry

end quadratic_rational_solutions_product_l1202_120266


namespace merchant_marked_price_l1202_120269

-- Definitions
def list_price : ℝ := 100
def purchase_price (L : ℝ) : ℝ := 0.8 * L
def selling_price_with_discount (x : ℝ) : ℝ := 0.75 * x
def profit (purchase_price : ℝ) (selling_price : ℝ) : ℝ := selling_price - purchase_price
def desired_profit (selling_price : ℝ) : ℝ := 0.3 * selling_price

-- Statement to prove
theorem merchant_marked_price :
  ∃ (x : ℝ), 
    profit (purchase_price list_price) (selling_price_with_discount x) = desired_profit (selling_price_with_discount x) ∧
    x / list_price = 152.38 / 100 :=
sorry

end merchant_marked_price_l1202_120269


namespace problem_I_problem_II_problem_III_l1202_120296

-- The function f(x)
noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) := (1/2) * x^2 - a * Real.log x + b

-- Tangent line at x = 1
def tangent_condition (a : ℝ) (b : ℝ) :=
  1 - a = 3 ∧ f 1 a b = 0

-- Extreme point at x = 1
def extreme_condition (a : ℝ) :=
  1 - a = 0 

-- Monotonicity and minimum m
def inequality_condition (a m : ℝ) :=
  -2 ≤ a ∧ a < 0 ∧ ∀ (x1 x2 : ℝ), 0 < x1 ∧ x1 ≤ 2 ∧ 0 < x2 ∧ x2 ≤ 2 → 
  |f x1 a (0 : ℝ) - f x2 a 0| ≤ m * |1 / x1 - 1 / x2|

-- Proof problem 1
theorem problem_I : ∃ (a b : ℝ), tangent_condition a b → a = -2 ∧ b = -0.5 := sorry

-- Proof problem 2
theorem problem_II : ∃ (a : ℝ), extreme_condition a → a = 1 := sorry

-- Proof problem 3
theorem problem_III : ∃ (m : ℝ), inequality_condition (-2 : ℝ) m → m = 12 := sorry

end problem_I_problem_II_problem_III_l1202_120296


namespace certain_number_minus_15_l1202_120252

theorem certain_number_minus_15 (n : ℕ) (h : n / 10 = 6) : n - 15 = 45 :=
sorry

end certain_number_minus_15_l1202_120252


namespace pyramid_top_value_l1202_120202

theorem pyramid_top_value 
  (p : ℕ) (q : ℕ) (z : ℕ) (m : ℕ) (n : ℕ) (left_mid : ℕ) (right_mid : ℕ) 
  (left_upper : ℕ) (right_upper : ℕ) (x_pre : ℕ) (x : ℕ) : 
  p = 20 → 
  q = 6 → 
  z = 44 → 
  m = p + 34 → 
  n = q + z → 
  left_mid = 17 + 29 → 
  right_mid = m + n → 
  left_upper = 36 + left_mid → 
  right_upper = right_mid + 42 → 
  x_pre = left_upper + 78 → 
  x = 2 * x_pre → 
  x = 320 :=
by
  intros
  sorry

end pyramid_top_value_l1202_120202


namespace min_time_to_pass_l1202_120208

noncomputable def tunnel_length : ℝ := 2150
noncomputable def num_vehicles : ℝ := 55
noncomputable def vehicle_length : ℝ := 10
noncomputable def speed_limit : ℝ := 20
noncomputable def max_speed : ℝ := 40

noncomputable def distance_between_vehicles (x : ℝ) : ℝ :=
if 0 < x ∧ x ≤ 10 then 20 else
if 10 < x ∧ x ≤ 20 then (1/6) * x ^ 2 + (1/3) * x else
0

noncomputable def time_to_pass_through_tunnel (x : ℝ) : ℝ :=
if 0 < x ∧ x ≤ 10 then (2150 + 10 * 55 + 20 * (55 - 1)) / x else
if 10 < x ∧ x ≤ 20 then (2150 + 10 * 55 + ((1/6) * x^2 + (1/3) * x) * (55 - 1)) / x + 9 * x + 18 else
0

theorem min_time_to_pass : ∃ x : ℝ, (10 < x ∧ x ≤ 20) ∧ x = 17.3 ∧ time_to_pass_through_tunnel x = 329.4 :=
sorry

end min_time_to_pass_l1202_120208


namespace jeanne_should_buy_more_tickets_l1202_120211

theorem jeanne_should_buy_more_tickets :
  let cost_ferris_wheel := 5
  let cost_roller_coaster := 4
  let cost_bumper_cars := 4
  let jeanne_current_tickets := 5
  let total_tickets_needed := cost_ferris_wheel + cost_roller_coaster + cost_bumper_cars
  let tickets_needed_to_buy := total_tickets_needed - jeanne_current_tickets
  tickets_needed_to_buy = 8 :=
by
  sorry

end jeanne_should_buy_more_tickets_l1202_120211


namespace largest_common_value_l1202_120236

/-- The largest value less than 300 that appears in both sequences 
    {7, 14, 21, 28, ...} and {5, 15, 25, 35, ...} -/
theorem largest_common_value (a : ℕ) (n m k : ℕ) :
  (a = 7 * (1 + n)) ∧ (a = 5 + 10 * m) ∧ (a < 300) ∧ (∀ k, (55 + 70 * k < 300) → (55 + 70 * k) ≤ a) 
  → a = 265 :=
by
  sorry

end largest_common_value_l1202_120236


namespace geometric_sequence_sum_l1202_120261

variable {a : ℕ → ℕ}

def is_geometric_sequence_with_common_product (k : ℕ) (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a n * a (n + 1) * a (n + 2) = k

theorem geometric_sequence_sum :
  is_geometric_sequence_with_common_product 27 a →
  a 1 = 1 →
  a 2 = 3 →
  (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 +
   a 11 + a 12 + a 13 + a 14 + a 15 + a 16 + a 17 + a 18) = 78 :=
by
  intros h_geom h_a1 h_a2
  sorry

end geometric_sequence_sum_l1202_120261


namespace find_article_cost_l1202_120237

noncomputable def original_cost_price (C S : ℝ) :=
  (S = 1.25 * C) ∧
  (S - 6.30 = 1.04 * C)

theorem find_article_cost (C S : ℝ) (h : original_cost_price C S) : C = 30 :=
by sorry

end find_article_cost_l1202_120237


namespace subset_N_M_l1202_120238

def M : Set ℝ := { x | ∃ (k : ℤ), x = k / 2 + 1 / 3 }
def N : Set ℝ := { x | ∃ (k : ℤ), x = k + 1 / 3 }

theorem subset_N_M : N ⊆ M := 
  sorry

end subset_N_M_l1202_120238


namespace pilot_speed_outbound_l1202_120201

theorem pilot_speed_outbound (v : ℝ) (d : ℝ) (s_return : ℝ) (t_total : ℝ) 
    (return_time : ℝ := d / s_return) 
    (outbound_time : ℝ := t_total - return_time) 
    (speed_outbound : ℝ := d / outbound_time) :
  d = 1500 → s_return = 500 → t_total = 8 → speed_outbound = 300 :=
by
  intros hd hs ht
  sorry

end pilot_speed_outbound_l1202_120201


namespace max_abcsum_l1202_120259

theorem max_abcsum (a b c : ℕ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_eq : a * b^2 * c^3 = 1350) : 
  a + b + c ≤ 154 :=
sorry

end max_abcsum_l1202_120259


namespace sum_of_dimensions_l1202_120297

theorem sum_of_dimensions (A B C : ℝ) (h1 : A * B = 50) (h2 : A * C = 90) (h3 : B * C = 100) : A + B + C = 24 :=
  sorry

end sum_of_dimensions_l1202_120297


namespace quadratic_roots_ratio_l1202_120232

theorem quadratic_roots_ratio (r1 r2 p q n : ℝ) (h1 : p = r1 * r2) (h2 : q = -(r1 + r2)) (h3 : p ≠ 0) (h4 : q ≠ 0) (h5 : n ≠ 0) (h6 : r1 ≠ 0) (h7 : r2 ≠ 0) (h8 : x^2 + q * x + p = 0) (h9 : x^2 + p * x + n = 0) :
  n / q = -3 :=
by
  sorry

end quadratic_roots_ratio_l1202_120232


namespace min_value_of_a_sq_plus_b_sq_over_a_minus_b_l1202_120256

theorem min_value_of_a_sq_plus_b_sq_over_a_minus_b {a b : ℝ} (h1 : a > b) (h2 : a * b = 1) : 
  ∃ x, x = 2 * Real.sqrt 2 ∧ ∀ y, y = (a^2 + b^2) / (a - b) → y ≥ x :=
by {
  sorry
}

end min_value_of_a_sq_plus_b_sq_over_a_minus_b_l1202_120256


namespace find_p_l1202_120298

variable (m n p : ℚ)

theorem find_p (h1 : m = 8 * n + 5) (h2 : m + 2 = 8 * (n + p) + 5) : p = 1 / 4 :=
by
  sorry

end find_p_l1202_120298


namespace max_a_condition_range_a_condition_l1202_120203

-- Definitions of the functions f and g
def f (x a : ℝ) : ℝ := |2 * x - a| + a
def g (x : ℝ) : ℝ := |2 * x - 1|

-- Problem (I)
theorem max_a_condition (a : ℝ) :
  (∀ x, g x ≤ 5 → f x a ≤ 6) → a ≤ 1 :=
sorry

-- Problem (II)
theorem range_a_condition (a : ℝ) :
  (∀ x, f x a + g x ≥ 3) → a ≥ 2 :=
sorry

end max_a_condition_range_a_condition_l1202_120203


namespace carlos_wins_one_game_l1202_120264

def games_Won_Laura : ℕ := 5
def games_Lost_Laura : ℕ := 4
def games_Won_Mike : ℕ := 7
def games_Lost_Mike : ℕ := 2
def games_Lost_Carlos : ℕ := 5
variable (C : ℕ) -- Carlos's wins

theorem carlos_wins_one_game :
  games_Won_Laura + games_Won_Mike + C = (games_Won_Laura + games_Lost_Laura + games_Won_Mike + games_Lost_Mike + C + games_Lost_Carlos) / 2 →
  C = 1 :=
by
  sorry

end carlos_wins_one_game_l1202_120264


namespace slower_pipe_fills_tank_in_200_minutes_l1202_120262

noncomputable def slower_pipe_filling_time (F S : ℝ) (h1 : F = 4 * S) (h2 : F + S = 1 / 40) : ℝ :=
  1 / S

theorem slower_pipe_fills_tank_in_200_minutes (F S : ℝ) (h1 : F = 4 * S) (h2 : F + S = 1 / 40) :
  slower_pipe_filling_time F S h1 h2 = 200 :=
sorry

end slower_pipe_fills_tank_in_200_minutes_l1202_120262


namespace total_people_in_line_l1202_120209

theorem total_people_in_line (n : ℕ) (h : n = 5): n + 2 = 7 :=
by
  -- This is where the proof would normally go, but we omit it with "sorry"
  sorry

end total_people_in_line_l1202_120209


namespace complement_intersection_eq_interval_l1202_120294

open Set

noncomputable def M : Set ℝ := {x | 3 * x - 1 >= 0}
noncomputable def N : Set ℝ := {x | 0 < x ∧ x < 1 / 2}

theorem complement_intersection_eq_interval :
  (M ∩ N)ᶜ = (Iio (1 / 3) ∪ Ici (1 / 2)) :=
by
  -- proof will go here in the actual development
  sorry

end complement_intersection_eq_interval_l1202_120294


namespace trains_cross_time_l1202_120258

noncomputable def timeToCross (length1 length2 speed1 speed2 : ℝ) : ℝ :=
  let speed1_mps := speed1 * (5 / 18)
  let speed2_mps := speed2 * (5 / 18)
  let relative_speed := speed1_mps + speed2_mps
  let total_length := length1 + length2
  total_length / relative_speed

theorem trains_cross_time
  (length1 length2 : ℝ)
  (speed1 speed2 : ℝ)
  (h_length1 : length1 = 250)
  (h_length2 : length2 = 250)
  (h_speed1 : speed1 = 90)
  (h_speed2 : speed2 = 110) :
  timeToCross length1 length2 speed1 speed2 = 9 := 
by sorry

end trains_cross_time_l1202_120258


namespace max_bicycle_distance_l1202_120210

-- Define the properties of the tires
def front_tire_duration : ℕ := 5000
def rear_tire_duration : ℕ := 3000

-- Define the maximum distance the bicycle can travel
def max_distance : ℕ := 3750

-- The main statement to be proven (proof is not required)
theorem max_bicycle_distance 
  (swap_usage : ∀ (d1 d2 : ℕ), d1 + d2 <= front_tire_duration + rear_tire_duration) : 
  ∃ (x : ℕ), x = max_distance := 
sorry

end max_bicycle_distance_l1202_120210


namespace donna_paid_165_l1202_120263

def original_price : ℝ := 200
def discount_rate : ℝ := 0.25
def tax_rate : ℝ := 0.1

def sale_price := original_price * (1 - discount_rate)
def tax := sale_price * tax_rate
def total_amount_paid := sale_price + tax

theorem donna_paid_165 : total_amount_paid = 165 := by
  sorry

end donna_paid_165_l1202_120263


namespace sum_of_x_y_possible_values_l1202_120246

theorem sum_of_x_y_possible_values (x y : ℝ) (h : x^3 + 21 * x * y + y^3 = 343) : x + y = 7 ∨ x + y = -14 := 
sorry

end sum_of_x_y_possible_values_l1202_120246


namespace fewer_twos_result_100_l1202_120223

theorem fewer_twos_result_100 :
  (222 / 2) - (22 / 2) = 100 := by
  sorry

end fewer_twos_result_100_l1202_120223


namespace part_I_part_II_l1202_120250

def sequence_sn (n : ℕ) : ℚ := (3 / 2 : ℚ) * n^2 + (1 / 2 : ℚ) * n

def sequence_a (n : ℕ) : ℕ := 3 * n - 1

def sequence_b (n : ℕ) : ℚ := (1 / 2 : ℚ)^n

def sequence_C (n : ℕ) : ℚ := sequence_a (sequence_a n) + sequence_b (sequence_a n)

def sum_of_first_n_terms (f : ℕ → ℚ) (n : ℕ) : ℚ :=
  (Finset.range n).sum f

theorem part_I (n : ℕ) : sequence_a n = 3 * n - 1 ∧ sequence_b n = (1 / 2)^n :=
by {
  sorry
}

theorem part_II (n : ℕ) : sum_of_first_n_terms sequence_C n =
  (n * (9 * n + 1) / 2) - (2 / 7) * (1 / 8)^n + (2 / 7) :=
by {
  sorry
}

end part_I_part_II_l1202_120250


namespace rhombus_second_diagonal_l1202_120268

theorem rhombus_second_diagonal (perimeter : ℝ) (d1 : ℝ) (side : ℝ) (half_d2 : ℝ) (d2 : ℝ) :
  perimeter = 52 → d1 = 24 → side = 13 → (half_d2 = 5) → d2 = 2 * half_d2 → d2 = 10 :=
by
  sorry

end rhombus_second_diagonal_l1202_120268


namespace arithmetic_sequence_ratio_l1202_120257

theorem arithmetic_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : ∀ n, S n = (n * (a 1 + a n)) / 2)
  (h2 : ∀ n, S n / a n = (n + 1) / 2) :
  (a 2 / a 3 = 2 / 3) :=
sorry

end arithmetic_sequence_ratio_l1202_120257


namespace problem_statement_l1202_120239

-- Definitions based on problem conditions
def p (a b c : ℝ) : Prop := a > b → (a * c^2 > b * c^2)

def q : Prop := ∃ x_0 : ℝ, (x_0 > 0) ∧ (x_0 - 1 + Real.log x_0 = 0)

-- Main theorem
theorem problem_statement : (¬ (∀ a b c : ℝ, p a b c)) ∧ q :=
by sorry

end problem_statement_l1202_120239


namespace sequence_is_geometric_not_arithmetic_l1202_120206

noncomputable def a_n (n : ℕ) : ℕ :=
  if n = 1 then 1
  else 2^(n-1)

def S_n (n : ℕ) : ℕ :=
  2^n - 1

theorem sequence_is_geometric_not_arithmetic (n : ℕ) : 
  (∀ n ≥ 2, a_n n = S_n n - S_n (n - 1)) ∧
  (a_n 1 = 1) ∧
  (∃ r : ℕ, r > 1 ∧ ∀ n ≥ 1, a_n (n + 1) = r * a_n n) ∧
  ¬(∃ d : ℤ, ∀ n, (a_n (n + 1) : ℤ) = a_n n + d) :=
by
  sorry

end sequence_is_geometric_not_arithmetic_l1202_120206


namespace value_of_M_l1202_120291

theorem value_of_M (M : ℝ) (H : 0.25 * M = 0.55 * 1500) : M = 3300 := 
by
  sorry

end value_of_M_l1202_120291


namespace b_eq_6_l1202_120222

theorem b_eq_6 (a b : ℤ) (h₁ : |a| = 1) (h₂ : ∀ x : ℝ, a * x^2 - 2 * x - b + 5 = 0 → x < 0) : b = 6 := 
by
  sorry

end b_eq_6_l1202_120222


namespace percentage_problem_l1202_120207

theorem percentage_problem 
    (y : ℝ)
    (h₁ : 0.47 * 1442 = 677.74)
    (h₂ : (677.74 - (y / 100) * 1412) + 63 = 3) :
    y = 52.25 :=
by sorry

end percentage_problem_l1202_120207


namespace present_age_of_B_l1202_120279

theorem present_age_of_B 
  (a b : ℕ)
  (h1 : a + 10 = 2 * (b - 10))
  (h2 : a = b + 9) :
  b = 39 :=
by
  sorry

end present_age_of_B_l1202_120279


namespace sphere_radius_same_volume_l1202_120272

noncomputable def tent_radius : ℝ := 3
noncomputable def tent_height : ℝ := 9

theorem sphere_radius_same_volume : 
  (4 / 3) * Real.pi * ( (20.25)^(1/3) )^3 = (1 / 3) * Real.pi * tent_radius^2 * tent_height :=
by
  sorry

end sphere_radius_same_volume_l1202_120272


namespace train_length_is_199_95_l1202_120295

noncomputable def convert_speed_to_m_s (speed_kmh : ℝ) : ℝ :=
  (speed_kmh * 1000) / 3600

noncomputable def length_of_train (bridge_length : ℝ) (time_seconds : ℝ) (speed_kmh : ℝ) : ℝ :=
  let speed_ms := convert_speed_to_m_s speed_kmh
  speed_ms * time_seconds - bridge_length

theorem train_length_is_199_95 :
  length_of_train 300 45 40 = 199.95 := by
  sorry

end train_length_is_199_95_l1202_120295


namespace two_point_five_one_million_in_scientific_notation_l1202_120226

theorem two_point_five_one_million_in_scientific_notation :
  (2.51 * 10^6 : ℝ) = 2.51e6 := 
sorry

end two_point_five_one_million_in_scientific_notation_l1202_120226


namespace solve_system_addition_l1202_120286

theorem solve_system_addition (a b : ℝ) (h1 : 3 * a + 7 * b = 1977) (h2 : 5 * a + b = 2007) : a + b = 498 :=
by
  sorry

end solve_system_addition_l1202_120286


namespace Q_at_1_eq_neg_1_l1202_120235

def P (x : ℝ) : ℝ := 3 * x^3 - 5 * x^2 + 2 * x - 1

noncomputable def mean_coefficient : ℝ := (3 - 5 + 2 - 1) / 4

noncomputable def Q (x : ℝ) : ℝ := mean_coefficient * x^3 + mean_coefficient * x^2 + mean_coefficient * x + mean_coefficient

theorem Q_at_1_eq_neg_1 : Q 1 = -1 := by
  sorry

end Q_at_1_eq_neg_1_l1202_120235


namespace probability_one_no_GP_l1202_120283

def num_pies : ℕ := 6
def growth_pies : ℕ := 2
def shrink_pies : ℕ := 4
def picked_pies : ℕ := 3
def total_outcomes : ℕ := Nat.choose num_pies picked_pies

def fav_outcomes : ℕ := Nat.choose shrink_pies 2 -- Choosing 2 out of the 4 SP

def probability_complementary : ℚ := fav_outcomes / total_outcomes
def probability : ℚ := 1 - probability_complementary

theorem probability_one_no_GP :
  probability = 0.4 := by
  sorry

end probability_one_no_GP_l1202_120283


namespace soldiers_line_l1202_120247

theorem soldiers_line (n x y z : ℕ) (h₁ : y = 6 * x) (h₂ : y = 7 * z)
                      (h₃ : n = x + y) (h₄ : n = 7 * x) (h₅ : n = 8 * z) : n = 98 :=
by 
  sorry

end soldiers_line_l1202_120247


namespace apps_more_than_files_l1202_120220

theorem apps_more_than_files
  (initial_apps : ℕ)
  (initial_files : ℕ)
  (deleted_apps : ℕ)
  (deleted_files : ℕ)
  (remaining_apps : ℕ)
  (remaining_files : ℕ)
  (h1 : initial_apps - deleted_apps = remaining_apps)
  (h2 : initial_files - deleted_files = remaining_files)
  (h3 : initial_apps = 24)
  (h4 : initial_files = 9)
  (h5 : remaining_apps = 12)
  (h6 : remaining_files = 5) :
  remaining_apps - remaining_files = 7 :=
by {
  sorry
}

end apps_more_than_files_l1202_120220


namespace tan_beta_value_l1202_120251

theorem tan_beta_value (α β : ℝ) (h1 : Real.tan α = -3 / 4) (h2 : Real.tan (α + β) = 1) : Real.tan β = 7 :=
sorry

end tan_beta_value_l1202_120251


namespace distance_between_parallel_sides_l1202_120230

-- Define the givens
def length_side_a : ℝ := 24  -- length of one parallel side
def length_side_b : ℝ := 14  -- length of the other parallel side
def area_trapezium : ℝ := 342  -- area of the trapezium

-- We need to prove that the distance between parallel sides (h) is 18 cm
theorem distance_between_parallel_sides (h : ℝ)
  (H1 :  area_trapezium = (1/2) * (length_side_a + length_side_b) * h) :
  h = 18 :=
by sorry

end distance_between_parallel_sides_l1202_120230


namespace calculate_otimes_l1202_120254

def otimes (a b : ℚ) : ℚ := (a + b) / (a - b)

theorem calculate_otimes :
  otimes (otimes 8 6) 12 = -19 / 5 := by
  sorry

end calculate_otimes_l1202_120254
