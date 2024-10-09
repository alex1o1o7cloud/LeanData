import Mathlib

namespace distance_between_points_l1492_149205

theorem distance_between_points : 
  let p1 := (3, -2) 
  let p2 := (-7, 4) 
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = Real.sqrt 136 :=
by
  sorry

end distance_between_points_l1492_149205


namespace find_number_l1492_149219

theorem find_number (x : ℕ) (h : x * 625 = 584638125) : x = 935420 :=
sorry

end find_number_l1492_149219


namespace rectangle_side_lengths_l1492_149282

theorem rectangle_side_lengths:
  ∃ x : ℝ, ∃ y : ℝ, (2 * (x + y) * 2 = x * y) ∧ (y = x + 3) ∧ (x > 0) ∧ (y > 0) ∧ x = 8 ∧ y = 11 :=
by
  sorry

end rectangle_side_lengths_l1492_149282


namespace total_beads_sue_necklace_l1492_149260

theorem total_beads_sue_necklace (purple blue green : ℕ) (h1 : purple = 7)
  (h2 : blue = 2 * purple) (h3 : green = blue + 11) : 
  purple + blue + green = 46 := 
by 
  sorry

end total_beads_sue_necklace_l1492_149260


namespace students_taking_German_l1492_149218

theorem students_taking_German 
  (total_students : ℕ)
  (students_taking_French : ℕ)
  (students_taking_both : ℕ)
  (students_not_taking_either : ℕ) 
  (students_taking_German : ℕ) 
  (h1 : total_students = 69)
  (h2 : students_taking_French = 41)
  (h3 : students_taking_both = 9)
  (h4 : students_not_taking_either = 15)
  (h5 : students_taking_German = 22) :
  total_students - students_not_taking_either = students_taking_French + students_taking_German - students_taking_both :=
sorry

end students_taking_German_l1492_149218


namespace find_r_divisibility_l1492_149235

theorem find_r_divisibility (r : ℝ) :
  (∃ s : ℝ, 10 * (x - r)^2 * (x - s) = 10 * x^3 - 5 * x^2 - 52 * x + 56) → r = 4 / 3 :=
by
  sorry

end find_r_divisibility_l1492_149235


namespace apples_to_eat_raw_l1492_149244

/-- Proof of the number of apples left to eat raw given the conditions -/
theorem apples_to_eat_raw 
  (total_apples : ℕ)
  (pct_wormy : ℕ)
  (pct_moldy : ℕ)
  (wormy_apples_offset : ℕ)
  (wormy_apples bruised_apples moldy_apples apples_left : ℕ) 
  (h1 : total_apples = 120)
  (h2 : pct_wormy = 20)
  (h3 : pct_moldy = 30)
  (h4 : wormy_apples = pct_wormy * total_apples / 100)
  (h5 : moldy_apples = pct_moldy * total_apples / 100)
  (h6 : bruised_apples = wormy_apples + wormy_apples_offset)
  (h7 : wormy_apples_offset = 9)
  (h8 : apples_left = total_apples - (wormy_apples + moldy_apples + bruised_apples))
  : apples_left = 27 :=
sorry

end apples_to_eat_raw_l1492_149244


namespace parabola_intersections_l1492_149276

-- Define the first parabola
def parabola1 (x : ℝ) : ℝ :=
  2 * x^2 - 10 * x - 10

-- Define the second parabola
def parabola2 (x : ℝ) : ℝ :=
  x^2 - 4 * x + 6

-- Define the theorem stating the points of intersection
theorem parabola_intersections :
  ∀ (p : ℝ × ℝ), (parabola1 p.1 = p.2) ∧ (parabola2 p.1 = p.2) ↔ (p = (-2, 18) ∨ p = (8, 38)) :=
by
  sorry

end parabola_intersections_l1492_149276


namespace no_solution_to_equation_l1492_149234

theorem no_solution_to_equation :
  ¬ ∃ x : ℝ, x ≠ 5 ∧ (1 / (x + 5) + 1 / (x - 5) = 1 / (x - 5)) :=
by 
  sorry

end no_solution_to_equation_l1492_149234


namespace least_pos_int_solution_l1492_149297

theorem least_pos_int_solution (x : ℤ) : x + 4609 ≡ 2104 [ZMOD 12] → x = 3 := by
  sorry

end least_pos_int_solution_l1492_149297


namespace max_m_eq_4_inequality_a_b_c_l1492_149223

noncomputable def f (x : ℝ) : ℝ :=
  |x - 3| + |x + 2|

theorem max_m_eq_4 (m : ℝ) (h : ∀ x : ℝ, f x ≥ |m + 1|) : m ≤ 4 ∧ m ≥ -6 :=
  sorry

theorem inequality_a_b_c (a b c : ℝ) (h : a + 2 * b + c = 4) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 / (a + b) + 1 / (b + c) ≥ 1 :=
  sorry

end max_m_eq_4_inequality_a_b_c_l1492_149223


namespace find_x_l1492_149277

-- Define the digits used
def digits : List ℕ := [1, 4, 5]

-- Define the sum of all four-digit numbers formed
def sum_of_digits (x : ℕ) : ℕ :=
  24 * (1 + 4 + 5 + x)

-- State the theorem
theorem find_x (x : ℕ) (h : sum_of_digits x = 288) : x = 2 :=
  by
    sorry

end find_x_l1492_149277


namespace smallest_n_square_smallest_n_cube_l1492_149243

theorem smallest_n_square (n : ℕ) : 
  (∃ x y : ℕ, x * (x + n) = y ^ 2) ↔ n = 3 := 
by sorry

theorem smallest_n_cube (n : ℕ) : 
  (∃ x y : ℕ, x * (x + n) = y ^ 3) ↔ n = 2 := 
by sorry

end smallest_n_square_smallest_n_cube_l1492_149243


namespace tangent_line_eqn_l1492_149252

noncomputable def f (x : ℝ) : ℝ := x * Real.log x + 1
noncomputable def f' (x : ℝ) : ℝ := Real.log x + 1

theorem tangent_line_eqn (h : f' x = 2) : 2 * x - y - Real.exp 1 + 1 = 0 :=
by
  sorry

end tangent_line_eqn_l1492_149252


namespace positive_number_property_l1492_149245

theorem positive_number_property (x : ℝ) (h_pos : x > 0) (h_property : 0.01 * x * x = 4) : x = 20 :=
sorry

end positive_number_property_l1492_149245


namespace alex_final_bill_l1492_149280

def original_bill : ℝ := 500
def first_late_charge (bill : ℝ) : ℝ := bill * 1.02
def final_bill (bill : ℝ) : ℝ := first_late_charge bill * 1.03

theorem alex_final_bill : final_bill original_bill = 525.30 :=
by sorry

end alex_final_bill_l1492_149280


namespace min_knights_in_village_l1492_149215

theorem min_knights_in_village :
  ∃ (K L : ℕ), K + L = 7 ∧ 2 * K * L = 24 ∧ K ≥ 3 :=
by
  sorry

end min_knights_in_village_l1492_149215


namespace quadratic_roots_l1492_149257

theorem quadratic_roots (x : ℝ) : (x^2 - 8 * x - 2 = 0) ↔ (x = 4 + 3 * Real.sqrt 2) ∨ (x = 4 - 3 * Real.sqrt 2) := by
  sorry

end quadratic_roots_l1492_149257


namespace sector_angle_l1492_149246

theorem sector_angle (r : ℝ) (θ : ℝ) 
  (area_eq : (1 / 2) * θ * r^2 = 1)
  (perimeter_eq : 2 * r + θ * r = 4) : θ = 2 := 
by
  sorry

end sector_angle_l1492_149246


namespace cosine_of_angle_l1492_149200

theorem cosine_of_angle (α : ℝ) (h : Real.sin (Real.pi / 6 + α) = Real.sqrt 3 / 2) : 
  Real.cos (Real.pi / 3 - α) = Real.sqrt 3 / 2 := 
by
  sorry

end cosine_of_angle_l1492_149200


namespace sum_of_a_b_c_l1492_149285

theorem sum_of_a_b_c (a b c : ℝ) (h1 : a * b = 24) (h2 : a * c = 36) (h3 : b * c = 54) : a + b + c = 19 :=
by
  -- The proof would go here
  sorry

end sum_of_a_b_c_l1492_149285


namespace precision_mult_10_decreases_precision_mult_35_decreases_precision_div_10_increases_precision_div_35_increases_l1492_149206

-- Given definitions for precision adjustment
def initial_precision := 3

def new_precision_mult (x : ℕ): ℕ :=
  initial_precision - 1   -- Example: Multiplying by 10 moves decimal point right decreasing precision by 1

def new_precision_mult_large (x : ℕ): ℕ := 
  initial_precision - 2   -- Example: Multiplying by 35 generally decreases precision by 2

def new_precision_div (x : ℕ): ℕ := 
  initial_precision + 1   -- Example: Dividing by 10 moves decimal point left increasing precision by 1

def new_precision_div_large (x : ℕ): ℕ := 
  initial_precision + 1   -- Example: Dividing by 35 generally increases precision by 1

-- Statements to prove
theorem precision_mult_10_decreases: 
  new_precision_mult 10 = 2 := 
by 
  sorry

theorem precision_mult_35_decreases: 
  new_precision_mult_large 35 = 1 := 
by 
  sorry

theorem precision_div_10_increases: 
  new_precision_div 10 = 4 := 
by 
  sorry

theorem precision_div_35_increases: 
  new_precision_div_large 35 = 4 := 
by 
  sorry

end precision_mult_10_decreases_precision_mult_35_decreases_precision_div_10_increases_precision_div_35_increases_l1492_149206


namespace find_f_1_l1492_149270

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_1 : (∀ x : ℝ, f x + 3 * f (-x) = Real.logb 2 (x + 3)) → f 1 = 1 / 8 := 
by 
  sorry

end find_f_1_l1492_149270


namespace log_eqn_proof_l1492_149287

theorem log_eqn_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h1 : Real.log a / Real.log 2 + Real.log b / Real.log 4 = 8)
  (h2 : Real.log a / Real.log 4 + Real.log b / Real.log 8 = 2) :
  Real.log a / Real.log 8 + Real.log b / Real.log 2 = -52 / 3 := 
by
  sorry

end log_eqn_proof_l1492_149287


namespace necessary_but_not_sufficient_l1492_149259

theorem necessary_but_not_sufficient (a : ℝ) (ha : a > 1) : a^2 > a :=
sorry

end necessary_but_not_sufficient_l1492_149259


namespace sqrt_div_add_l1492_149294

theorem sqrt_div_add :
  let sqrt_0_81 := 0.9
  let sqrt_1_44 := 1.2
  let sqrt_0_49 := 0.7
  (Real.sqrt 1.1 / sqrt_0_81) + (sqrt_1_44 / sqrt_0_49) = 2.8793 :=
by
  -- Prove equality using the given conditions
  sorry

end sqrt_div_add_l1492_149294


namespace most_reasonable_sampling_method_l1492_149279

-- Define the conditions
axiom significant_differences_in_educational_stages : Prop
axiom insignificant_differences_between_genders : Prop

-- Define the options
inductive SamplingMethod
| SimpleRandomSampling
| StratifiedSamplingByGender
| StratifiedSamplingByEducationalStage
| SystematicSampling

-- State the problem as a theorem
theorem most_reasonable_sampling_method
  (H1 : significant_differences_in_educational_stages)
  (H2 : insignificant_differences_between_genders) :
  SamplingMethod.StratifiedSamplingByEducationalStage = SamplingMethod.StratifiedSamplingByEducationalStage :=
by
  -- Proof is skipped
  sorry

end most_reasonable_sampling_method_l1492_149279


namespace distance_to_building_materials_l1492_149227

theorem distance_to_building_materials (D : ℝ) 
  (h1 : 2 * 10 * 4 * D = 8000) : 
  D = 100 := 
by
  sorry

end distance_to_building_materials_l1492_149227


namespace total_points_first_half_l1492_149225

noncomputable def raiders_wildcats_scores := 
  ∃ (a b d r : ℕ),
    (a = b + 1) ∧
    (a * (1 + r + r^2 + r^3) = 4 * b + 6 * d + 2) ∧
    (a + a * r ≤ 100) ∧
    (b + b + d ≤ 100)

theorem total_points_first_half : 
  raiders_wildcats_scores → 
  ∃ (total : ℕ), total = 25 :=
by
  sorry

end total_points_first_half_l1492_149225


namespace negation_of_implication_l1492_149266

variable (a b c : ℝ)

theorem negation_of_implication :
  (¬(a + b + c = 3) → a^2 + b^2 + c^2 < 3) ↔
  ¬((a + b + c = 3) → a^2 + b^2 + c^2 ≥ 3) := by
sorry

end negation_of_implication_l1492_149266


namespace exponential_sum_sequence_l1492_149281

noncomputable def Sn (n : ℕ) : ℝ :=
  Real.log (1 + 1 / n)

theorem exponential_sum_sequence : 
  e^(Sn 9 - Sn 6) = (20 : ℝ) / 21 := by
  sorry

end exponential_sum_sequence_l1492_149281


namespace find_side_length_a_l1492_149290

variable {a b c : ℝ}
variable {B : ℝ}

theorem find_side_length_a (h_b : b = 7) (h_c : c = 5) (h_B : B = 2 * Real.pi / 3) :
  a = 3 :=
sorry

end find_side_length_a_l1492_149290


namespace negation_exists_zero_product_l1492_149231

variable {R : Type} [LinearOrderedField R]

variable (f g : R → R)

theorem negation_exists_zero_product :
  (¬ ∃ x : R, f x * g x = 0) ↔ ∀ x : R, f x ≠ 0 ∧ g x ≠ 0 :=
by
  sorry

end negation_exists_zero_product_l1492_149231


namespace corresponding_angles_equal_l1492_149253

theorem corresponding_angles_equal 
  (α β γ : ℝ) 
  (h1 : α + β + γ = 180) 
  (h2 : (180 - α) + β + γ = 180) : 
  α = 90 ∧ β + γ = 90 ∧ (180 - α = 90) :=
by
  sorry

end corresponding_angles_equal_l1492_149253


namespace surface_area_of_T_is_630_l1492_149211

noncomputable def s : ℕ := 582
noncomputable def t : ℕ := 42
noncomputable def u : ℕ := 6

theorem surface_area_of_T_is_630 : s + t + u = 630 :=
by
  sorry

end surface_area_of_T_is_630_l1492_149211


namespace student_solved_correctly_l1492_149256

theorem student_solved_correctly (c e : ℕ) (h1 : c + e = 80) (h2 : 5 * c - 3 * e = 8) : c = 31 :=
sorry

end student_solved_correctly_l1492_149256


namespace find_number_l1492_149214

-- Given conditions
variables (x y : ℕ)

-- The conditions from the problem statement
def digit_sum : Prop := x + y = 12
def reverse_condition : Prop := (10 * x + y) + 36 = 10 * y + x

-- The final statement
theorem find_number (h1 : digit_sum x y) (h2 : reverse_condition x y) : 10 * x + y = 48 :=
sorry

end find_number_l1492_149214


namespace paint_ratio_l1492_149275

theorem paint_ratio
  (blue yellow white : ℕ)
  (ratio_b : ℕ := 4)
  (ratio_y : ℕ := 3)
  (ratio_w : ℕ := 5)
  (total_white : ℕ := 15)
  : yellow = 9 := by
  have ratio := ratio_b + ratio_y + ratio_w
  have white_parts := total_white * ratio_w / ratio_w
  have yellow_parts := white_parts * ratio_y / ratio_w
  exact sorry

end paint_ratio_l1492_149275


namespace regular_polygon_enclosure_l1492_149247

theorem regular_polygon_enclosure (m n : ℕ) (h : m = 12)
    (h_enc : ∀ p : ℝ, p = 360 / ↑n → (2 * (180 / ↑n)) = (360 / ↑m)) :
    n = 12 :=
by
  sorry

end regular_polygon_enclosure_l1492_149247


namespace cost_of_each_big_apple_l1492_149233

theorem cost_of_each_big_apple :
  ∀ (small_cost medium_cost : ℝ) (big_cost : ℝ) (num_small num_medium num_big : ℕ) (total_cost : ℝ),
  small_cost = 1.5 →
  medium_cost = 2 →
  num_small = 6 →
  num_medium = 6 →
  num_big = 8 →
  total_cost = 45 →
  total_cost = num_small * small_cost + num_medium * medium_cost + num_big * big_cost →
  big_cost = 3 :=
by
  intros small_cost medium_cost big_cost num_small num_medium num_big total_cost
  sorry

end cost_of_each_big_apple_l1492_149233


namespace math_club_team_selection_l1492_149237

theorem math_club_team_selection :
  let boys := 10
  let girls := 12
  let total := boys + girls
  let team_size := 8
  (Nat.choose total team_size - Nat.choose girls team_size - Nat.choose boys team_size = 319230) :=
by
  sorry

end math_club_team_selection_l1492_149237


namespace complement_intersection_l1492_149292

open Set

-- Definitions of the sets U, M, and N
def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {3, 4, 5}

-- The theorem we want to prove
theorem complement_intersection :
  (compl M ∩ N) = {4, 5} :=
by
  sorry

end complement_intersection_l1492_149292


namespace determine_quarters_given_l1492_149298

def total_initial_coins (dimes quarters nickels : ℕ) : ℕ :=
  dimes + quarters + nickels

def updated_dimes (original_dimes added_dimes : ℕ) : ℕ :=
  original_dimes + added_dimes

def updated_nickels (original_nickels factor : ℕ) : ℕ :=
  original_nickels + original_nickels * factor

def total_coins_after_addition (dimes quarters nickels : ℕ) (added_dimes added_quarters added_nickels_factor : ℕ) : ℕ :=
  updated_dimes dimes added_dimes +
  (quarters + added_quarters) +
  updated_nickels nickels added_nickels_factor

def quarters_given_by_mother (total_coins initial_dimes initial_quarters initial_nickels added_dimes added_nickels_factor : ℕ) : ℕ :=
  total_coins - total_initial_coins initial_dimes initial_quarters initial_nickels - added_dimes - initial_nickels * added_nickels_factor

theorem determine_quarters_given :
  quarters_given_by_mother 35 2 6 5 2 2 = 10 :=
by
  sorry

end determine_quarters_given_l1492_149298


namespace total_fruits_l1492_149204

theorem total_fruits (cucumbers : ℕ) (watermelons : ℕ) 
  (h1 : cucumbers = 18) 
  (h2 : watermelons = cucumbers + 8) : 
  cucumbers + watermelons = 44 := 
by {
  sorry
}

end total_fruits_l1492_149204


namespace normal_line_at_x0_is_correct_l1492_149217

noncomputable def curve (x : ℝ) : ℝ := x^(2/3) - 20

def x0 : ℝ := -8

def normal_line_equation (x : ℝ) : ℝ := 3 * x + 8

theorem normal_line_at_x0_is_correct : 
  ∃ y0 : ℝ, curve x0 = y0 ∧ y0 = curve x0 ∧ normal_line_equation x0 = y0 :=
sorry

end normal_line_at_x0_is_correct_l1492_149217


namespace original_faculty_size_l1492_149255

theorem original_faculty_size (F : ℝ) (h1 : F * 0.85 * 0.80 = 195) : F = 287 :=
by
  sorry

end original_faculty_size_l1492_149255


namespace price_increase_for_desired_profit_l1492_149242

/--
In Xianyou Yonghui Supermarket, the profit from selling Pomelos is 10 yuan per kilogram.
They can sell 500 kilograms per day. Market research has found that, with a constant cost price, if the price per kilogram increases by 1 yuan, the daily sales volume will decrease by 20 kilograms.
Now, the supermarket wants to ensure a daily profit of 6000 yuan while also offering the best deal to the customers.
-/
theorem price_increase_for_desired_profit :
  ∃ x : ℝ, (10 + x) * (500 - 20 * x) = 6000 ∧ x = 5 :=
sorry

end price_increase_for_desired_profit_l1492_149242


namespace find_f_3_l1492_149203

def f (x : ℝ) : ℝ := sorry

theorem find_f_3 : (∀ y : ℝ, y > 0 → f ((4 * y + 1) / (y + 1)) = 1 / y) → f 3 = 1 / 2 :=
by
  intro h
  sorry

end find_f_3_l1492_149203


namespace price_decrease_percentage_l1492_149232

-- Definitions based on given conditions
def price_in_2007 (x : ℝ) : ℝ := x
def price_in_2008 (x : ℝ) : ℝ := 1.25 * x
def desired_price_in_2009 (x : ℝ) : ℝ := 1.1 * x

-- Theorem statement to prove the price decrease from 2008 to 2009
theorem price_decrease_percentage (x : ℝ) (h : x > 0) : 
  (1.25 * x - 1.1 * x) / (1.25 * x) = 0.12 := 
sorry

end price_decrease_percentage_l1492_149232


namespace cone_radius_l1492_149240

theorem cone_radius (r l : ℝ) 
  (surface_area_eq : π * r^2 + π * r * l = 12 * π)
  (net_is_semicircle : π * l = 2 * π * r) : 
  r = 2 :=
by
  sorry

end cone_radius_l1492_149240


namespace tangent_line_at_P_l1492_149262

noncomputable def y (x : ℝ) : ℝ := 2 * x^2 + 1

def P : ℝ × ℝ := (-1, 3)

theorem tangent_line_at_P :
    ∀ (x y : ℝ), (y = 2*x^2 + 1) →
    (x, y) = P →
    ∃ m b : ℝ, b = -1 ∧ m = -4 ∧ (y = m*x + b) :=
by
  sorry

end tangent_line_at_P_l1492_149262


namespace smallest_n_conditions_l1492_149265

theorem smallest_n_conditions :
  ∃ n : ℕ, 0 < n ∧ (∃ k1 : ℕ, 2 * n = k1^2) ∧ (∃ k2 : ℕ, 3 * n = k2^4) ∧ n = 54 :=
by
  sorry

end smallest_n_conditions_l1492_149265


namespace product_of_digits_l1492_149264

theorem product_of_digits (A B : ℕ) (h1 : A + B = 14) (h2 : (10 * A + B) % 4 = 0) : A * B = 48 :=
by
  sorry

end product_of_digits_l1492_149264


namespace necessary_but_not_sufficient_l1492_149222

variable (p q : Prop)
-- Condition p: The base of a right prism is a rhombus.
def base_of_right_prism_is_rhombus := p
-- Condition q: A prism is a right rectangular prism.
def prism_is_right_rectangular := q

-- Proof: p is a necessary but not sufficient condition for q.
theorem necessary_but_not_sufficient (p q : Prop) 
  (h1 : base_of_right_prism_is_rhombus p)
  (h2 : prism_is_right_rectangular q) : 
  (q → p) ∧ ¬ (p → q) :=
sorry

end necessary_but_not_sufficient_l1492_149222


namespace vector_sum_eq_l1492_149230

variables (x y : ℝ)
def a : ℝ × ℝ := (2, 3)
def b : ℝ × ℝ := (3, 3)
def c : ℝ × ℝ := (7, 8)

theorem vector_sum_eq :
  ∃ (x y : ℝ), c = (x • a.1 + y • b.1, x • a.2 + y • b.2) ∧ x + y = 8 / 3 :=
by
  have h1 : 7 = 2 * x + 3 * y := sorry
  have h2 : 8 = 3 * x + 3 * y := sorry
  sorry

end vector_sum_eq_l1492_149230


namespace find_divisor_l1492_149236

theorem find_divisor (d : ℕ) (n : ℕ) (least : ℕ)
  (h1 : least = 2)
  (h2 : n = 433124)
  (h3 : ∀ d : ℕ, (d ∣ (n + least)) → d = 2) :
  d = 2 := 
sorry

end find_divisor_l1492_149236


namespace when_to_sell_goods_l1492_149291

variable (a : ℝ) (currentMonthProfit nextMonthProfitWithStorage : ℝ) 
          (interestRate storageFee thisMonthProfit nextMonthProfit : ℝ)
          (hm1 : interestRate = 0.005)
          (hm2 : storageFee = 5)
          (hm3 : thisMonthProfit = 100)
          (hm4 : nextMonthProfit = 120)
          (hm5 : currentMonthProfit = thisMonthProfit + (a + thisMonthProfit) * interestRate)
          (hm6 : nextMonthProfitWithStorage = nextMonthProfit - storageFee)

theorem when_to_sell_goods :
  (a > 2900 → currentMonthProfit > nextMonthProfitWithStorage) ∧
  (a = 2900 → currentMonthProfit = nextMonthProfitWithStorage) ∧
  (a < 2900 → currentMonthProfit < nextMonthProfitWithStorage) := by
  sorry

end when_to_sell_goods_l1492_149291


namespace trigonometric_inequality_l1492_149202

theorem trigonometric_inequality (x : Real) (h1 : 0 < x) (h2 : x < (3 * Real.pi) / 8) :
  (1 / Real.sin (x / 3) + 1 / Real.sin (8 * x / 3) > (Real.sin (3 * x / 2)) / (Real.sin (x / 2) * Real.sin (2 * x))) :=
  by
  sorry

end trigonometric_inequality_l1492_149202


namespace length_error_probability_l1492_149249

theorem length_error_probability
  (μ σ : ℝ)
  (X : ℝ → ℝ)
  (h_norm_dist : ∀ x : ℝ, X x = (Real.exp (-(x - μ) ^ 2 / (2 * σ ^ 2)) / (σ * Real.sqrt (2 * Real.pi))))
  (h_max_density : X 0 = 1 / (3 * Real.sqrt (2 * Real.pi)))
  (P : Set ℝ → ℝ)
  (h_prop1 : P {x | μ - σ < x ∧ x < μ + σ} = 0.6826)
  (h_prop2 : P {x | μ - 2 * σ < x ∧ x < μ + 2 * σ} = 0.9544) :
  P {x | 3 < x ∧ x < 6} = 0.1359 :=
sorry

end length_error_probability_l1492_149249


namespace value_division_l1492_149288

theorem value_division (x y : ℝ) (h1 : y ≠ 0) (h2 : 2 * x - y = 1.75 * x) 
                       (h3 : x / y = n) : n = 4 := 
by 
sorry

end value_division_l1492_149288


namespace find_f_at_4_l1492_149254

noncomputable def f : ℝ → ℝ := sorry -- We assume such a function exists

theorem find_f_at_4:
  (∀ x : ℝ, f (4^x) + x * f (4^(-x)) = 3) → f (4) = 0 := by
  intro h
  -- Proof would go here, but is omitted as per instructions
  sorry

end find_f_at_4_l1492_149254


namespace B_more_cost_effective_l1492_149274

variable (x y : ℝ)
variable (hx : x ≠ y)

theorem B_more_cost_effective (x y : ℝ) (hx : x ≠ y) :
  (1/2 * x + 1/2 * y) > (2 * x * y / (x + y)) :=
by
  sorry

end B_more_cost_effective_l1492_149274


namespace evaluate_expression_l1492_149210

theorem evaluate_expression :
  1002^3 - 1001 * 1002^2 - 1001^2 * 1002 + 1001^3 - 1000^3 = 2009007 :=
by
  sorry

end evaluate_expression_l1492_149210


namespace maximum_S_n_l1492_149299

noncomputable def a_n (a_1 d : ℝ) (n : ℕ) : ℝ :=
  a_1 + (n - 1) * d

noncomputable def S_n (a_1 d : ℝ) (n : ℕ) : ℝ :=
  n / 2 * (2 * a_1 + (n - 1) * d)

theorem maximum_S_n (a_1 : ℝ) (h : a_1 > 0)
  (h_sequence : 3 * a_n a_1 (2 * a_1 / 39) 8 = 5 * a_n a_1 (2 * a_1 / 39) 13)
  : ∀ n : ℕ, S_n a_1 (2 * a_1 / 39) n ≤ S_n a_1 (2 * a_1 / 39) 20 :=
sorry

end maximum_S_n_l1492_149299


namespace last_digit_of_prime_l1492_149208

theorem last_digit_of_prime (n : ℕ) (h1 : 859433 = 214858 * 4 + 1) : (2 ^ 859433 - 1) % 10 = 1 := by
  sorry

end last_digit_of_prime_l1492_149208


namespace find_X_l1492_149226

theorem find_X :
  (15.2 * 0.25 - 48.51 / 14.7) / X = ((13 / 44 - 2 / 11 - 5 / 66) / (5 / 2) * (6 / 5)) / (3.2 + 0.8 * (5.5 - 3.25)) ->
  X = 137.5 :=
by
  intro h
  sorry

end find_X_l1492_149226


namespace percentage_of_males_l1492_149209

theorem percentage_of_males (P : ℝ) (total_employees : ℝ) (below_50_male_count : ℝ) :
  total_employees = 2800 →
  0.70 * (P / 100 * total_employees) = below_50_male_count →
  below_50_male_count = 490 →
  P = 25 :=
by
  intros h_total h_eq h_below_50
  sorry

end percentage_of_males_l1492_149209


namespace Sheila_attend_probability_l1492_149269

noncomputable def prob_rain := 0.3
noncomputable def prob_sunny := 0.4
noncomputable def prob_cloudy := 0.3

noncomputable def prob_attend_if_rain := 0.25
noncomputable def prob_attend_if_sunny := 0.9
noncomputable def prob_attend_if_cloudy := 0.5

noncomputable def prob_attend :=
  prob_rain * prob_attend_if_rain +
  prob_sunny * prob_attend_if_sunny +
  prob_cloudy * prob_attend_if_cloudy

theorem Sheila_attend_probability : prob_attend = 0.585 := by
  sorry

end Sheila_attend_probability_l1492_149269


namespace looms_employed_l1492_149258

def sales_value := 500000
def manufacturing_expenses := 150000
def establishment_charges := 75000
def profit_decrease := 5000

def profit_per_loom (L : ℕ) : ℕ := (sales_value / L) - (manufacturing_expenses / L)

theorem looms_employed (L : ℕ) (h : profit_per_loom L = profit_decrease) : L = 70 :=
by
  have h_eq : profit_per_loom L = (sales_value - manufacturing_expenses) / L := by
    sorry
  have profit_expression : profit_per_loom L = profit_decrease := by
    sorry
  have L_value : L = (sales_value - manufacturing_expenses) / profit_decrease := by
    sorry
  have L_is_70 : L = 70 := by
    sorry
  exact L_is_70

end looms_employed_l1492_149258


namespace max_value_of_f_l1492_149272

noncomputable def f (x : ℝ) : ℝ :=
  (1 / 5) * Real.sin (x + Real.pi / 3) + Real.cos (x - Real.pi / 6)

theorem max_value_of_f : ∃ x, f x ≤ 6 / 5 :=
sorry

end max_value_of_f_l1492_149272


namespace smallest_four_digit_divisible_by_53_ending_in_3_l1492_149228

theorem smallest_four_digit_divisible_by_53_ending_in_3 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 53 ∣ n ∧ n % 10 = 3 ∧ n = 1113 := 
by
  sorry

end smallest_four_digit_divisible_by_53_ending_in_3_l1492_149228


namespace total_legs_l1492_149221

def human_legs : Nat := 2
def num_humans : Nat := 2
def dog_legs : Nat := 4
def num_dogs : Nat := 2

theorem total_legs :
  num_humans * human_legs + num_dogs * dog_legs = 12 := by
  sorry

end total_legs_l1492_149221


namespace hyperbola_eccentricity_l1492_149296

theorem hyperbola_eccentricity : 
  let a := Real.sqrt 2
  let b := 1
  let c := Real.sqrt (a^2 + b^2)
  (c / a) = Real.sqrt 6 / 2 := 
by
  sorry

end hyperbola_eccentricity_l1492_149296


namespace biology_vs_reading_diff_l1492_149284

def math_hw_pages : ℕ := 2
def reading_hw_pages : ℕ := 3
def total_hw_pages : ℕ := 15

def biology_hw_pages : ℕ := total_hw_pages - (math_hw_pages + reading_hw_pages)

theorem biology_vs_reading_diff : (biology_hw_pages - reading_hw_pages) = 7 := by
  sorry

end biology_vs_reading_diff_l1492_149284


namespace gambler_initial_games_l1492_149271

theorem gambler_initial_games (x : ℕ)
  (h1 : ∀ x, ∃ (wins : ℝ), wins = 0.40 * x) 
  (h2 : ∀ x, ∃ (total_games : ℕ), total_games = x + 30)
  (h3 : ∀ x, ∃ (total_wins : ℝ), total_wins = 0.40 * x + 24)
  (h4 : ∀ x, ∃ (final_win_rate : ℝ), final_win_rate = (0.40 * x + 24) / (x + 30))
  (h5 : ∃ (final_win_rate_target : ℝ), final_win_rate_target = 0.60) :
  x = 30 :=
by
  sorry

end gambler_initial_games_l1492_149271


namespace cross_prod_correct_l1492_149273

open Matrix

def vec1 : ℝ × ℝ × ℝ := (3, -1, 4)
def vec2 : ℝ × ℝ × ℝ := (-4, 6, 2)
def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (a.2.1 * b.2.2 - a.2.2 * b.2.1,
  a.2.2 * b.1 - a.1 * b.2.2,
  a.1 * b.2.1 - a.2.1 * b.1)

theorem cross_prod_correct :
  cross_product vec1 vec2 = (-26, -22, 14) := by
  -- sorry is used to simplify the proof.
  sorry

end cross_prod_correct_l1492_149273


namespace fraction_given_to_sofia_is_correct_l1492_149212

-- Pablo, Sofia, Mia, and Ana's initial egg counts
variables {m : ℕ}
def mia_initial (m : ℕ) := m
def sofia_initial (m : ℕ) := 3 * m
def pablo_initial (m : ℕ) := 12 * m
def ana_initial (m : ℕ) := m / 2

-- Total eggs and desired equal distribution
def total_eggs (m : ℕ) := 12 * m + 3 * m + m + m / 2
def equal_distribution (m : ℕ) := 33 * m / 4

-- Eggs each need to be equal
def sofia_needed (m : ℕ) := equal_distribution m - sofia_initial m
def mia_needed (m : ℕ) := equal_distribution m - mia_initial m
def ana_needed (m : ℕ) := equal_distribution m - ana_initial m

-- Fraction of eggs given to Sofia
def pablo_fraction_to_sofia (m : ℕ) := sofia_needed m / pablo_initial m

theorem fraction_given_to_sofia_is_correct (m : ℕ) :
  pablo_fraction_to_sofia m = 7 / 16 :=
sorry

end fraction_given_to_sofia_is_correct_l1492_149212


namespace volume_of_sand_pile_l1492_149213

theorem volume_of_sand_pile (d h : ℝ) (π : ℝ) (r : ℝ) (vol : ℝ) :
  d = 8 →
  h = (3 / 4) * d →
  r = d / 2 →
  vol = (1 / 3) * π * r^2 * h →
  vol = 32 * π :=
by
  intros hd hh hr hv
  subst hd
  subst hh
  subst hr
  subst hv
  sorry

end volume_of_sand_pile_l1492_149213


namespace tan_value_l1492_149267

open Real

theorem tan_value (α : ℝ) 
  (h1 : sin (α + π / 6) = -3 / 5)
  (h2 : -2 * π / 3 < α ∧ α < -π / 6) : 
  tan (4 * π / 3 - α) = -4 / 3 :=
sorry

end tan_value_l1492_149267


namespace average_score_girls_cedar_drake_l1492_149293

theorem average_score_girls_cedar_drake
  (C c D d : ℕ)
  (cedar_boys_score cedar_girls_score cedar_combined_score
   drake_boys_score drake_girls_score drake_combined_score combined_boys_score : ℝ)
  (h1 : cedar_boys_score = 68)
  (h2 : cedar_girls_score = 80)
  (h3 : cedar_combined_score = 73)
  (h4 : drake_boys_score = 75)
  (h5 : drake_girls_score = 88)
  (h6 : drake_combined_score = 83)
  (h7 : combined_boys_score = 74)
  (h8 : (68 * C + 80 * c) / (C + c) = 73)
  (h9 : (75 * D + 88 * d) / (D + d) = 83)
  (h10 : (68 * C + 75 * D) / (C + D) = 74) :
  (80 * c + 88 * d) / (c + d) = 87 :=
by
  -- proof is omitted
  sorry

end average_score_girls_cedar_drake_l1492_149293


namespace john_total_distance_l1492_149207

-- Define the parameters according to the conditions
def daily_distance : ℕ := 1700
def number_of_days : ℕ := 6
def total_distance : ℕ := daily_distance * number_of_days

-- Lean theorem statement to prove the total distance run by John
theorem john_total_distance : total_distance = 10200 := by
  -- Here, the proof would go, but it is omitted as per instructions
  sorry

end john_total_distance_l1492_149207


namespace hyperbola_h_k_a_b_sum_eq_l1492_149220

theorem hyperbola_h_k_a_b_sum_eq :
  ∃ (h k a b : ℝ), 
  h = 0 ∧ 
  k = 0 ∧ 
  a = 4 ∧ 
  (c : ℝ) = 8 ∧ 
  c^2 = a^2 + b^2 ∧ 
  h + k + a + b = 4 + 4 * Real.sqrt 3 := by
{ sorry }

end hyperbola_h_k_a_b_sum_eq_l1492_149220


namespace circle_equation_range_of_k_l1492_149251

theorem circle_equation_range_of_k (k : ℝ) : 
  (∃ x y : ℝ, x^2 + y^2 + 4*k*x - 2*y + 5*k = 0) ↔ (k > 1 ∨ k < 1/4) :=
by
  sorry

end circle_equation_range_of_k_l1492_149251


namespace angle_B_area_of_triangle_l1492_149263

/-
Given a triangle ABC with angle A, B, C and sides a, b, c opposite to these angles respectively.
Consider the conditions:
- A = π/6
- b = (4 + 2 * sqrt 3) * a * cos B
- b = 1

Prove:
1. B = 5 * π / 12
2. The area of triangle ABC = 1 / 4
-/

namespace TriangleProof

open Real

def triangle_conditions (A B C a b c : ℝ) : Prop :=
  A = π / 6 ∧
  b = (4 + 2 * sqrt 3) * a * cos B ∧
  b = 1

theorem angle_B (A B C a b c : ℝ) 
  (h : triangle_conditions A B C a b c) : 
  B = 5 * π / 12 :=
sorry

theorem area_of_triangle (A B C a b c : ℝ) 
  (h : triangle_conditions A B C a b c) : 
  1 / 2 * b * c * sin A = 1 / 4 :=
sorry

end TriangleProof

end angle_B_area_of_triangle_l1492_149263


namespace tyler_remaining_money_l1492_149283

def initial_amount : ℕ := 100
def scissors_qty : ℕ := 8
def scissor_cost : ℕ := 5
def erasers_qty : ℕ := 10
def eraser_cost : ℕ := 4

def remaining_amount (initial_amount scissors_qty scissor_cost erasers_qty eraser_cost : ℕ) : ℕ :=
  initial_amount - (scissors_qty * scissor_cost + erasers_qty * eraser_cost)

theorem tyler_remaining_money :
  remaining_amount initial_amount scissors_qty scissor_cost erasers_qty eraser_cost = 20 := by
  sorry

end tyler_remaining_money_l1492_149283


namespace inequality_solution_set_l1492_149241

def solution_set (a b x : ℝ) : Set ℝ := {x | |a - b * x| - 5 ≤ 0}

theorem inequality_solution_set (x : ℝ) :
  solution_set 4 3 x = {x | - (1 : ℝ) / 3 ≤ x ∧ x ≤ 3} :=
by {
  sorry
}

end inequality_solution_set_l1492_149241


namespace area_of_triangle_ADE_l1492_149229

theorem area_of_triangle_ADE (A B C D E : Type) (AB BC AC : ℝ) (AD AE : ℝ)
  (h1 : AB = 8) (h2 : BC = 13) (h3 : AC = 15) (h4 : AD = 3) (h5 : AE = 11) :
  let s := (AB + BC + AC) / 2
  let area_ABC := Real.sqrt (s * (s - AB) * (s - BC) * (s - AC))
  let sinA := 2 * area_ABC / (AB * AC)
  let area_ADE := (1 / 2) * AD * AE * sinA
  area_ADE = (33 * Real.sqrt 3) / 4 :=
by 
  have s := (8 + 13 + 15) / 2
  have area_ABC := Real.sqrt (s * (s - 8) * (s - 13) * (s - 15))
  have sinA := 2 * area_ABC / (8 * 15)
  have area_ADE := (1 / 2) * 3 * 11 * sinA
  sorry

end area_of_triangle_ADE_l1492_149229


namespace anniversary_sale_total_cost_l1492_149238

-- Definitions of conditions
def original_price_ice_cream : ℕ := 12
def discount_ice_cream : ℕ := 2
def sale_price_ice_cream : ℕ := original_price_ice_cream - discount_ice_cream

def price_per_five_cans_juice : ℕ := 2
def cans_per_five_pack : ℕ := 5

-- Definition of total cost
def total_cost : ℕ := 2 * sale_price_ice_cream + (10 / cans_per_five_pack) * price_per_five_cans_juice

-- The goal is to prove that total_cost is 24
theorem anniversary_sale_total_cost : total_cost = 24 :=
by
  sorry

end anniversary_sale_total_cost_l1492_149238


namespace all_sets_form_right_angled_triangle_l1492_149261

theorem all_sets_form_right_angled_triangle :
    (6 * 6 + 8 * 8 = 10 * 10) ∧
    (7 * 7 + 24 * 24 = 25 * 25) ∧
    (3 * 3 + 4 * 4 = 5 * 5) ∧
    (Real.sqrt 2 * Real.sqrt 2 + Real.sqrt 3 * Real.sqrt 3 = Real.sqrt 5 * Real.sqrt 5) :=
by {
  sorry
}

end all_sets_form_right_angled_triangle_l1492_149261


namespace solve_quadratic_equation_l1492_149295

theorem solve_quadratic_equation (x : ℝ) : 2 * (x + 1) ^ 2 - 49 = 1 ↔ (x = 4 ∨ x = -6) := 
sorry

end solve_quadratic_equation_l1492_149295


namespace smith_a_students_l1492_149248

-- Definitions representing the conditions

def johnson_a_students : ℕ := 12
def johnson_total_students : ℕ := 20
def smith_total_students : ℕ := 30

def johnson_ratio := johnson_a_students / johnson_total_students

-- Statement to prove
theorem smith_a_students :
  (johnson_a_students / johnson_total_students) = (18 / smith_total_students) :=
sorry

end smith_a_students_l1492_149248


namespace otimes_2_1_equals_3_l1492_149289

namespace MathProof

-- Define the operation
def otimes (a b : ℝ) : ℝ := a^2 - b

-- The main theorem to prove
theorem otimes_2_1_equals_3 : otimes 2 1 = 3 :=
by
  -- Proof content not needed
  sorry

end MathProof

end otimes_2_1_equals_3_l1492_149289


namespace additional_men_required_l1492_149278

variables (W_r : ℚ) (W : ℚ) (D : ℚ) (M : ℚ) (E : ℚ)

-- Given variables
def initial_work_rate := (2.5 : ℚ) / (50 * 100)
def remaining_work_length := (12.5 : ℚ)
def remaining_days := (200 : ℚ)
def initial_men := (50 : ℚ)
def additional_men_needed := (75 : ℚ)

-- Calculating the additional men required
theorem additional_men_required
  (calc_wr : W_r = initial_work_rate)
  (calc_wr_remain : W = remaining_work_length)
  (calc_days_remain : D = remaining_days)
  (calc_initial_men : M = initial_men)
  (calc_additional_men : M + E = (125 : ℚ)) :
  E = additional_men_needed :=
sorry

end additional_men_required_l1492_149278


namespace find_a_value_l1492_149250

theorem find_a_value (a a_1 a_2 a_3 a_4 a_5 : ℝ) :
  (∀ x : ℝ, (x + 1)^5 = a + a_1 * (x - 1) + a_2 * (x - 1)^2 + a_3 * (x - 1)^3 + a_4 * (x - 1)^4 + a_5 * (x - 1)^5) → 
  a = 32 :=
by
  sorry

end find_a_value_l1492_149250


namespace divides_f_of_nat_l1492_149201

variable {n : ℕ}

theorem divides_f_of_nat (n : ℕ) : 5 ∣ (76 * n^5 + 115 * n^4 + 19 * n) := 
sorry

end divides_f_of_nat_l1492_149201


namespace molecular_weight_is_correct_l1492_149216

structure Compound :=
  (H C N Br O : ℕ)

structure AtomicWeights :=
  (H C N Br O : ℝ)

noncomputable def molecularWeight (compound : Compound) (weights : AtomicWeights) : ℝ :=
  compound.H * weights.H +
  compound.C * weights.C +
  compound.N * weights.N +
  compound.Br * weights.Br +
  compound.O * weights.O

def givenCompound : Compound :=
  { H := 2, C := 2, N := 1, Br := 1, O := 4 }

def givenWeights : AtomicWeights :=
  { H := 1.008, C := 12.011, N := 14.007, Br := 79.904, O := 15.999 }

theorem molecular_weight_is_correct : molecularWeight givenCompound givenWeights = 183.945 := by
  sorry

end molecular_weight_is_correct_l1492_149216


namespace find_complex_number_z_l1492_149224

-- Given the complex number z and the equation \(\frac{z}{1+i} = i^{2015} + i^{2016}\)
-- prove that z = -2i
theorem find_complex_number_z (z : ℂ) (h : z / (1 + (1 : ℂ) * I) = I ^ 2015 + I ^ 2016) : z = -2 * I := 
by
  sorry

end find_complex_number_z_l1492_149224


namespace sequence_product_is_128_l1492_149239

-- Define the sequence of fractions
def fractional_sequence (n : ℕ) : Rat :=
  if n % 2 = 0 then 1 / (2 : ℕ) ^ ((n + 2) / 2)
  else (2 : ℕ) ^ ((n + 1) / 2)

-- The target theorem: prove the product of the sequence results in 128
theorem sequence_product_is_128 : 
  (List.prod (List.map fractional_sequence [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])) = 128 := 
by
  sorry

end sequence_product_is_128_l1492_149239


namespace complex_power_difference_l1492_149286

theorem complex_power_difference (i : ℂ) (h : i^2 = -1) : (1 + i) ^ 16 - (1 - i) ^ 16 = 0 := by
  sorry

end complex_power_difference_l1492_149286


namespace find_k_l1492_149268

variable {x y k : ℝ}

theorem find_k (h1 : 3 * x + 4 * y = k + 2) 
             (h2 : 2 * x + y = 4) 
             (h3 : x + y = 2) :
  k = 4 := 
by
  sorry

end find_k_l1492_149268
