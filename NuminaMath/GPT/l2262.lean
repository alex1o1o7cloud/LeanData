import Mathlib

namespace find_d_l2262_226266

variable (x y d : ℤ)

-- Condition from the problem
axiom condition1 : (7 * x + 4 * y) / (x - 2 * y) = 13

-- The main proof goal
theorem find_d : x = 5 * y → x / (2 * y) = d / 2 → d = 5 :=
by
  intro h1 h2
  -- proof goes here
  sorry

end find_d_l2262_226266


namespace negated_proposition_false_l2262_226299

theorem negated_proposition_false : ¬ ∀ x : ℝ, 2^x + x^2 > 1 :=
by 
sorry

end negated_proposition_false_l2262_226299


namespace divisible_by_27000_l2262_226287

theorem divisible_by_27000 (k : ℕ) (h₁ : k = 30) : ∃ n : ℕ, k^3 = 27000 * n :=
by {
  sorry
}

end divisible_by_27000_l2262_226287


namespace distances_from_median_l2262_226295

theorem distances_from_median (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) :
  ∃ (x y : ℝ), x = (b * c) / (a + b) ∧ y = (a * c) / (a + b) ∧ x + y = c :=
by
  sorry

end distances_from_median_l2262_226295


namespace find_a_maximize_profit_sets_sold_after_increase_l2262_226265

variable (a x m : ℕ)

-- Condition for finding 'a'
def condition_for_a (a : ℕ) : Prop :=
  600 * (a - 110) = 160 * a

-- The equation after solving
def solution_for_a (a : ℕ) : Prop :=
  a = 150

theorem find_a : condition_for_a a → solution_for_a a :=
sorry

-- Profit maximization constraints
def condition_for_max_profit (x : ℕ) : Prop :=
  x + 5 * x + 20 ≤ 200

-- Total number of items purchased
def total_items_purchased (x : ℕ) : ℕ :=
  x + 5 * x + 20

-- Profit expression
def profit (x : ℕ) : ℕ :=
  215 * x + 600

-- Maximized profit
def maximum_profit (W : ℕ) : Prop :=
  W = 7050

theorem maximize_profit (x : ℕ) (W : ℕ) :
  condition_for_max_profit x → x ≤ 30 → total_items_purchased x ≤ 200 → maximum_profit W → x = 30 :=
sorry

-- Condition for sets sold after increase
def condition_for_sets_sold (a m : ℕ) : Prop :=
  let new_table_price := 160
  let new_chair_price := 50
  let profit_m_after_increase := (500 - new_table_price - 4 * new_chair_price) * m +
                                (30 - m) * (270 - new_table_price) +
                                (170 - 4 * m) * (70 - new_chair_price)
  profit_m_after_increase + 2250 = 7050 - 2250

-- Solved for 'm'
def quantity_of_sets_sold (m : ℕ) : Prop :=
  m = 20

theorem sets_sold_after_increase (a m : ℕ) :
  condition_for_sets_sold a m → quantity_of_sets_sold m :=
sorry

end find_a_maximize_profit_sets_sold_after_increase_l2262_226265


namespace problem_statement_l2262_226297

noncomputable def lhs: ℝ := 8^6 * 27^6 * 8^27 * 27^8
noncomputable def rhs: ℝ := 216^14 * 8^19

theorem problem_statement : lhs = rhs :=
by
  sorry

end problem_statement_l2262_226297


namespace find_distance_AC_l2262_226206

noncomputable def distance_AC : ℝ :=
  let speed := 25  -- km per hour
  let angleA := 30  -- degrees
  let angleB := 135 -- degrees
  let distanceBC := 25 -- km
  (distanceBC * Real.sin (angleB * Real.pi / 180)) / (Real.sin (angleA * Real.pi / 180))

theorem find_distance_AC :
  distance_AC = 25 * Real.sqrt 2 :=
by
  sorry

end find_distance_AC_l2262_226206


namespace arithmetic_sequence_term_l2262_226272

theorem arithmetic_sequence_term :
  (∀ (a_n : ℕ → ℚ) (S : ℕ → ℚ),
    (∀ n, a_n n = a_n 1 + (n - 1) * 1) → -- Arithmetic sequence with common difference of 1
    (∀ n, S n = n * a_n 1 + (n * (n - 1)) / 2) →  -- Sum of first n terms of sequence
    S 8 = 4 * S 4 →
    a_n 10 = 19 / 2) :=
by
  intros a_n S ha_n hSn hS8_eq
  sorry

end arithmetic_sequence_term_l2262_226272


namespace machine_copies_l2262_226205

theorem machine_copies (x : ℕ) (h1 : ∀ t : ℕ, t = 30 → 30 * t = 900)
  (h2 : 900 + 30 * 30 = 2550) : x = 55 :=
by
  sorry

end machine_copies_l2262_226205


namespace find_repeating_digits_l2262_226235

-- Specify given conditions
def incorrect_result (a : ℚ) (b : ℚ) : ℚ := 54 * b - 1.8
noncomputable def correct_multiplication_value (d: ℚ) := 2 + d
noncomputable def repeating_decimal_value : ℚ := 2 + 35 / 99

-- Define what needs to be proved
theorem find_repeating_digits : ∃ (x : ℕ), x * 100 = 35 := by
  sorry

end find_repeating_digits_l2262_226235


namespace triangle_inequality_third_side_l2262_226228

theorem triangle_inequality_third_side (x : ℝ) (h1 : 5 > 0) (h2 : 8 > 0) :
  3 < x ∧ x < 13 ↔ (5 + 8 > x) ∧ (5 + x > 8) ∧ (8 + x > 5) :=
by 
  -- Placeholder for proof
  sorry

end triangle_inequality_third_side_l2262_226228


namespace hall_paving_l2262_226274

theorem hall_paving :
  ∀ (hall_length hall_breadth stone_length stone_breadth : ℕ),
    hall_length = 72 →
    hall_breadth = 30 →
    stone_length = 8 →
    stone_breadth = 10 →
    let Area_hall := hall_length * hall_breadth
    let Length_stone := stone_length / 10
    let Breadth_stone := stone_breadth / 10
    let Area_stone := Length_stone * Breadth_stone 
    (Area_hall / Area_stone) = 2700 :=
by
  intros hall_length hall_breadth stone_length stone_breadth
  intro h1 h2 h3 h4
  let Area_hall := hall_length * hall_breadth
  let Length_stone := stone_length / 10
  let Breadth_stone := stone_breadth / 10
  let Area_stone := Length_stone * Breadth_stone 
  have h5 : Area_hall / Area_stone = 2700 := sorry
  exact h5

end hall_paving_l2262_226274


namespace find_constants_l2262_226268

theorem find_constants (a b : ℚ) (h1 : 3 * a + b = 7) (h2 : a + 4 * b = 5) :
  a = 61 / 33 ∧ b = 8 / 11 :=
by
  sorry

end find_constants_l2262_226268


namespace min_value_inequality_l2262_226294

open Real

theorem min_value_inequality (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x + 3 * y = 4) :
  ∃ z, z = (2 / x + 3 / y) ∧ z = 25 / 4 :=
by
  sorry

end min_value_inequality_l2262_226294


namespace value_of_a_l2262_226284

theorem value_of_a (a : ℝ) (h : 1 ∈ ({a, a ^ 2} : Set ℝ)) : a = -1 :=
sorry

end value_of_a_l2262_226284


namespace f_def_pos_l2262_226230

-- Define f to be an odd function
variable (f : ℝ → ℝ)
-- Define f as an odd function
axiom odd_f (x : ℝ) : f (-x) = -f x

-- Define f when x < 0
axiom f_def_neg (x : ℝ) (h : x < 0) : f x = (Real.cos (3 * x)) + (Real.sin (2 * x))

-- State the theorem to be proven:
theorem f_def_pos (x : ℝ) (h : 0 < x) : f x = - (Real.cos (3 * x)) + (Real.sin (2 * x)) :=
sorry

end f_def_pos_l2262_226230


namespace xy_over_y_plus_x_l2262_226275

theorem xy_over_y_plus_x {x y z : ℝ} (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (h : 1/x + 1/y = 1/z) : z = xy/(y+x) :=
sorry

end xy_over_y_plus_x_l2262_226275


namespace relationship_abc_l2262_226221

noncomputable def a : ℝ := 5 ^ (Real.log 3.4 / Real.log 3)
noncomputable def b : ℝ := 5 ^ (Real.log 3.6 / Real.log 3)
noncomputable def c : ℝ := (1 / 5) ^ (Real.log 0.5 / Real.log 3)

theorem relationship_abc : b > a ∧ a > c := by 
  sorry

end relationship_abc_l2262_226221


namespace max_additional_spheres_in_cone_l2262_226222

-- Definition of spheres O_{1} and O_{2} properties
def O₁_radius : ℝ := 2
def O₂_radius : ℝ := 3
def height_cone : ℝ := 8

-- Conditions:
def O₁_on_axis (h : ℝ) := height_cone > 0 ∧ h = O₁_radius
def O₁_tangent_top_base := height_cone = O₁_radius + O₁_radius
def O₂_tangent_O₁ := O₁_radius + O₂_radius = 5
def O₂_on_base := O₂_radius = 3

-- Lean theorem stating mathematically equivalent proof problem
theorem max_additional_spheres_in_cone (h : ℝ) :
  O₁_on_axis h → O₁_tangent_top_base →
  O₂_tangent_O₁ → O₂_on_base →
  ∃ n : ℕ, n = 2 :=
by
  sorry

end max_additional_spheres_in_cone_l2262_226222


namespace min_value_of_expression_l2262_226290

theorem min_value_of_expression (x y : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : 4 * x + 3 * y = 1) :
  1 / (2 * x - y) + 2 / (x + 2 * y) = 9 :=
sorry

end min_value_of_expression_l2262_226290


namespace sqrt7_sub_m_div_n_gt_inv_mn_l2262_226282

variables (m n : ℤ)
variables (h_m_nonneg : m ≥ 1) (h_n_nonneg : n ≥ 1)
variables (h_ineq : Real.sqrt 7 - (m : ℝ) / (n : ℝ) > 0)

theorem sqrt7_sub_m_div_n_gt_inv_mn : 
  Real.sqrt 7 - (m : ℝ) / (n : ℝ) > 1 / ((m : ℝ) * (n : ℝ)) :=
by
  sorry

end sqrt7_sub_m_div_n_gt_inv_mn_l2262_226282


namespace complex_inverse_l2262_226281

noncomputable def complex_expression (i : ℂ) (h_i : i ^ 2 = -1) : ℂ :=
  (3 * i - 3 * (1 / i))⁻¹

theorem complex_inverse (i : ℂ) (h_i : i^2 = -1) :
  complex_expression i h_i = -i / 6 :=
by
  -- the proof part is omitted
  sorry

end complex_inverse_l2262_226281


namespace jamie_collects_oysters_l2262_226264

theorem jamie_collects_oysters (d : ℕ) (p : ℕ) (r : ℕ) (x : ℕ)
  (h1 : d = 14)
  (h2 : p = 56)
  (h3 : r = 25)
  (h4 : x = p / d * 100 / r) :
  x = 16 :=
by
  sorry

end jamie_collects_oysters_l2262_226264


namespace projection_equal_p_l2262_226258

open Real EuclideanSpace

noncomputable def vector1 : ℝ × ℝ := (-3, 4)
noncomputable def vector2 : ℝ × ℝ := (1, 6)
noncomputable def v : ℝ × ℝ := (4, 2)
noncomputable def p : ℝ × ℝ := (-2.2, 4.4)

theorem projection_equal_p (p_ortho : (p.1 * v.1 + p.2 * v.2) = 0) : p = (4 * (1 / 5) - 3, 2 * (1 / 5) + 4) :=
by
  sorry

end projection_equal_p_l2262_226258


namespace square_division_l2262_226236

theorem square_division (n : Nat) : (n > 5 → ∃ smaller_squares : List (Nat × Nat), smaller_squares.length = n ∧ (∀ s ∈ smaller_squares, IsSquare s)) ∧ (n = 2 ∨ n = 3 → ¬ ∃ smaller_squares : List (Nat × Nat), smaller_squares.length = n ∧ (∀ s ∈ smaller_squares, IsSquare s)) := 
by
  sorry

end square_division_l2262_226236


namespace remainder_4015_div_32_l2262_226286

theorem remainder_4015_div_32 : 4015 % 32 = 15 := by
  sorry

end remainder_4015_div_32_l2262_226286


namespace fraction_work_left_l2262_226207

theorem fraction_work_left (A_days B_days : ℕ) (together_days : ℕ) 
  (H_A : A_days = 20) (H_B : B_days = 30) (H_t : together_days = 4) : 
  (1 : ℚ) - (together_days * ((1 : ℚ) / A_days + (1 : ℚ) / B_days)) = 2 / 3 :=
by
  sorry

end fraction_work_left_l2262_226207


namespace united_telephone_additional_charge_l2262_226244

theorem united_telephone_additional_charge :
  ∃ x : ℝ, 
    (11 + 20 * x = 16) ↔ (x = 0.25) := by
  sorry

end united_telephone_additional_charge_l2262_226244


namespace find_n_l2262_226218

theorem find_n : ∃ n : ℕ, 5^3 - 7 = 6^2 + n ∧ n = 82 := by
  use 82
  sorry

end find_n_l2262_226218


namespace value_of_B_l2262_226248

theorem value_of_B (B : ℚ) (h : 3 * B - 5 = 23) : B = 28 / 3 :=
by
  sorry

-- Explanation:
-- B is declared as a rational number (ℚ) because the answer involves a fraction.
-- h is the condition 3 * B - 5 = 23.
-- The theorem states that given h, B equals 28 / 3.

end value_of_B_l2262_226248


namespace root_power_division_l2262_226227

noncomputable def root4 (a : ℝ) : ℝ := a^(1/4)
noncomputable def root6 (a : ℝ) : ℝ := a^(1/6)

theorem root_power_division : 
  (root4 7) / (root6 7) = 7^(1/12) :=
by sorry

end root_power_division_l2262_226227


namespace probability_both_red_is_one_fourth_l2262_226217

noncomputable def probability_of_both_red (total_cards : ℕ) (red_cards : ℕ) (draws : ℕ) : ℚ :=
  (red_cards / total_cards) ^ draws

theorem probability_both_red_is_one_fourth :
  probability_of_both_red 52 26 2 = 1/4 :=
by
  sorry

end probability_both_red_is_one_fourth_l2262_226217


namespace walk_time_to_LakePark_restaurant_l2262_226242

/-
  It takes 15 minutes for Dante to go to Hidden Lake.
  From Hidden Lake, it takes him 7 minutes to walk back to the Park Office.
  Dante will have been gone from the Park Office for a total of 32 minutes.
  Prove that the walk from the Park Office to the Lake Park restaurant is 10 minutes.
-/

def T_HiddenLake_to : ℕ := 15
def T_HiddenLake_from : ℕ := 7
def T_total : ℕ := 32
def T_LakePark_restaurant : ℕ := T_total - (T_HiddenLake_to + T_HiddenLake_from)

theorem walk_time_to_LakePark_restaurant : 
  T_LakePark_restaurant = 10 :=
by
  unfold T_LakePark_restaurant T_HiddenLake_to T_HiddenLake_from T_total
  sorry

end walk_time_to_LakePark_restaurant_l2262_226242


namespace initial_people_count_25_l2262_226270

-- Definition of the initial number of people (X) and the condition
def initial_people (X : ℕ) : Prop := X - 8 + 13 = 30

-- The theorem stating that the initial number of people is 25
theorem initial_people_count_25 : ∃ (X : ℕ), initial_people X ∧ X = 25 :=
by
  -- We add sorry here to skip the actual proof
  sorry

end initial_people_count_25_l2262_226270


namespace volume_of_dug_earth_l2262_226252

theorem volume_of_dug_earth :
  let r := 2
  let h := 14
  ∃ V : ℝ, V = Real.pi * r^2 * h ∧ V = 56 * Real.pi :=
by
  sorry

end volume_of_dug_earth_l2262_226252


namespace derivative_of_f_l2262_226203

noncomputable def f (x : ℝ) : ℝ := (Real.sin (1 / x)) ^ 3

theorem derivative_of_f (x : ℝ) (hx : x ≠ 0) : 
  deriv f x = - (3 / x ^ 2) * (Real.sin (1 / x)) ^ 2 * Real.cos (1 / x) :=
by
  sorry 

end derivative_of_f_l2262_226203


namespace find_monthly_growth_rate_l2262_226251

-- Define all conditions.
variables (March_sales May_sales : ℝ) (monthly_growth_rate : ℝ)

-- The conditions from the given problem
def initial_sales (March_sales : ℝ) : Prop := March_sales = 4 * 10^6
def final_sales (May_sales : ℝ) : Prop := May_sales = 9 * 10^6
def growth_occurred (March_sales May_sales : ℝ) (monthly_growth_rate : ℝ) : Prop :=
  May_sales = March_sales * (1 + monthly_growth_rate)^2

-- The Lean 4 theorem to be proven.
theorem find_monthly_growth_rate 
  (h1 : initial_sales March_sales) 
  (h2 : final_sales May_sales) 
  (h3 : growth_occurred March_sales May_sales monthly_growth_rate) : 
  400 * (1 + monthly_growth_rate)^2 = 900 := 
sorry

end find_monthly_growth_rate_l2262_226251


namespace dice_tower_even_n_l2262_226259

/-- Given that n standard dice are stacked in a vertical tower,
and the total visible dots on each of the four vertical walls are all odd,
prove that n must be even.
-/
theorem dice_tower_even_n (n : ℕ)
  (h : ∀ (S T : ℕ), (S + T = 7 * n → (S % 2 = 1 ∧ T % 2 = 1))) : n % 2 = 0 :=
by sorry

end dice_tower_even_n_l2262_226259


namespace number_of_diagonals_intersections_l2262_226219

theorem number_of_diagonals_intersections (n : ℕ) (h : n ≥ 4) : 
  (∃ (I : ℕ), I = (n * (n - 1) * (n - 2) * (n - 3)) / 24) :=
by {
  sorry
}

end number_of_diagonals_intersections_l2262_226219


namespace lower_bound_of_range_of_expression_l2262_226260

theorem lower_bound_of_range_of_expression :
  ∃ L, (∀ n : ℤ, L < 4*n + 7 → 4*n + 7 < 100) ∧
  (∃! n_min n_max : ℤ, 4*n_min + 7 = L ∧ 4*n_max + 7 = 99 ∧ (n_max - n_min + 1 = 25)) :=
sorry

end lower_bound_of_range_of_expression_l2262_226260


namespace jack_salt_amount_l2262_226202

noncomputable def amount_of_salt (volume_salt_1 : ℝ) (volume_salt_2 : ℝ) : ℝ :=
  volume_salt_1 + volume_salt_2

noncomputable def total_salt_ml (total_salt_l : ℝ) : ℝ :=
  total_salt_l * 1000

theorem jack_salt_amount :
  let day1_water_l := 4.0
  let day2_water_l := 4.0
  let day1_salt_percentage := 0.18
  let day2_salt_percentage := 0.22
  let total_salt_before_evaporation := amount_of_salt (day1_water_l * day1_salt_percentage) (day2_water_l * day2_salt_percentage)
  let final_salt_ml := total_salt_ml total_salt_before_evaporation
  final_salt_ml = 1600 :=
by
  sorry

end jack_salt_amount_l2262_226202


namespace isosceles_trapezoid_larger_base_l2262_226200

theorem isosceles_trapezoid_larger_base (AD BC AC : ℝ) (h1 : AD = 10) (h2 : BC = 6) (h3 : AC = 14) :
  ∃ (AB : ℝ), AB = 16 :=
by
  sorry

end isosceles_trapezoid_larger_base_l2262_226200


namespace determine_operation_l2262_226250

theorem determine_operation (a b c d : Int) : ((a - b) + c - (3 * 1) = d) → ((a - b) + 2 = 6) → (a - b = 4) :=
by
  sorry

end determine_operation_l2262_226250


namespace sum_of_three_numbers_l2262_226280

theorem sum_of_three_numbers (a b c : ℕ) (h1 : a ≤ b) (h2 : b ≤ c) (h3 : b = 12)
    (h4 : (a + b + c) / 3 = a + 8) (h5 : (a + b + c) / 3 = c - 18) : 
    a + b + c = 66 := 
sorry

end sum_of_three_numbers_l2262_226280


namespace find_C_l2262_226269

theorem find_C (A B C : ℕ) (h1 : A + B + C = 900) (h2 : A + C = 400) (h3 : B + C = 750) : C = 250 :=
by
  sorry

end find_C_l2262_226269


namespace ratio_of_fifteenth_terms_l2262_226256

def S (n : ℕ) : ℝ := sorry
def T (n : ℕ) : ℝ := sorry
def a (n : ℕ) : ℝ := sorry
def b (n : ℕ) : ℝ := sorry

theorem ratio_of_fifteenth_terms 
  (h1: ∀ n, S n / T n = (5 * n + 3) / (3 * n + 35))
  (h2: ∀ n, a n = S n) -- Example condition
  (h3: ∀ n, b n = T n) -- Example condition
  : (a 15 / b 15) = 59 / 57 := 
  by 
  -- Placeholder proof
  sorry

end ratio_of_fifteenth_terms_l2262_226256


namespace no_such_triples_l2262_226249

theorem no_such_triples : ¬ ∃ (x y z : ℤ), (xy + yz + zx ≠ 0) ∧ (x^2 + y^2 + z^2) / (xy + yz + zx) = 2016 :=
by
  sorry

end no_such_triples_l2262_226249


namespace rate_second_year_l2262_226273

/-- Define the principal amount at the start. -/
def P : ℝ := 4000

/-- Define the rate of interest for the first year. -/
def rate_first_year : ℝ := 0.04

/-- Define the final amount after 2 years. -/
def A : ℝ := 4368

/-- Define the amount after the first year. -/
def P1 : ℝ := P + P * rate_first_year

/-- Define the interest for the second year. -/
def Interest2 : ℝ := A - P1

/-- Define the principal amount for the second year, which is the amount after the first year. -/
def P2 : ℝ := P1

/-- Prove that the rate of interest for the second year is 5%. -/
theorem rate_second_year : (Interest2 / P2) * 100 = 5 :=
by
  sorry

end rate_second_year_l2262_226273


namespace number_notebooks_in_smaller_package_l2262_226214

theorem number_notebooks_in_smaller_package 
  (total_notebooks : ℕ)
  (large_packs : ℕ)
  (notebooks_per_large_pack : ℕ)
  (condition_1 : total_notebooks = 69)
  (condition_2 : large_packs = 7)
  (condition_3 : notebooks_per_large_pack = 7)
  (condition_4 : ∃ x : ℕ, x < 7 ∧ (total_notebooks - (large_packs * notebooks_per_large_pack)) % x = 0) :
  ∃ x : ℕ, x < 7 ∧ x = 5 := 
by 
  sorry

end number_notebooks_in_smaller_package_l2262_226214


namespace final_price_l2262_226254

variable (OriginalPrice : ℝ)

def salePrice (OriginalPrice : ℝ) : ℝ :=
  0.6 * OriginalPrice

def priceAfterCoupon (SalePrice : ℝ) : ℝ :=
  0.75 * SalePrice

theorem final_price (OriginalPrice : ℝ) :
  priceAfterCoupon (salePrice OriginalPrice) = 0.45 * OriginalPrice := by
  sorry

end final_price_l2262_226254


namespace compare_negatives_l2262_226220

theorem compare_negatives : -0.5 > -0.7 := 
by 
  exact sorry 

end compare_negatives_l2262_226220


namespace max_area_of_pen_l2262_226201

theorem max_area_of_pen (perimeter : ℝ) (h : perimeter = 60) : 
  ∃ (x : ℝ), (3 * x + x = 60) ∧ (2 * x * x = 450) :=
by
  -- This theorem states that there exists an x such that
  -- the total perimeter with internal divider equals 60,
  -- and the total area of the two squares equals 450.
  use 15
  sorry

end max_area_of_pen_l2262_226201


namespace prove_a_eq_b_l2262_226215

theorem prove_a_eq_b 
  (p q a b : ℝ) 
  (h1 : p + q = 1) 
  (h2 : p * q ≠ 0) 
  (h3 : p / a + q / b = 1 / (p * a + q * b)) : 
  a = b := 
sorry

end prove_a_eq_b_l2262_226215


namespace line_length_after_erasure_l2262_226212

-- Defining the initial and erased lengths
def initial_length_cm : ℕ := 100
def erased_length_cm : ℕ := 33

-- The statement we need to prove
theorem line_length_after_erasure : initial_length_cm - erased_length_cm = 67 := by
  sorry

end line_length_after_erasure_l2262_226212


namespace find_max_marks_l2262_226224

variable (M : ℝ)
variable (pass_mark : ℝ := 60 / 100)
variable (obtained_marks : ℝ := 200)
variable (additional_marks_needed : ℝ := 80)

theorem find_max_marks (h1 : pass_mark * M = obtained_marks + additional_marks_needed) : M = 467 := 
by
  sorry

end find_max_marks_l2262_226224


namespace trapezoid_perimeter_l2262_226285

theorem trapezoid_perimeter (AB CD AD BC h : ℝ)
  (AB_eq : AB = 40)
  (CD_eq : CD = 70)
  (AD_eq_BC : AD = BC)
  (h_eq : h = 24)
  : AB + BC + CD + AD = 110 + 2 * Real.sqrt 801 :=
by
  -- Proof goes here, you can replace this comment with actual proof.
  sorry

end trapezoid_perimeter_l2262_226285


namespace sum_of_consecutive_integers_largest_set_of_consecutive_positive_integers_l2262_226267

theorem sum_of_consecutive_integers (n : ℕ) (a : ℕ) (h : n ≥ 1) (h_sum : n * (2 * a + n - 1) = 56) : n ≤ 7 := 
by
  sorry

theorem largest_set_of_consecutive_positive_integers : ∃ n a, n ≥ 1 ∧ n * (2 * a + n - 1) = 56 ∧ n = 7 := 
by
  use 7, 1
  repeat {split}
  sorry

end sum_of_consecutive_integers_largest_set_of_consecutive_positive_integers_l2262_226267


namespace solve_equation_l2262_226240

theorem solve_equation : ∀ x : ℝ, 4 * x + 4 - x - 2 * x + 2 - 2 - x + 2 + 6 = 0 → x = 0 :=
by 
  intro x h
  sorry

end solve_equation_l2262_226240


namespace sum_of_squares_iff_double_l2262_226296

theorem sum_of_squares_iff_double (x : ℤ) :
  (∃ a b : ℤ, x = a^2 + b^2) ↔ (∃ u v : ℤ, 2 * x = u^2 + v^2) :=
by
  sorry

end sum_of_squares_iff_double_l2262_226296


namespace part1_part2_l2262_226263

noncomputable def f (x : ℝ) : ℝ := 2 * |x + 1| + |x - 2|

theorem part1 : {x : ℝ | f x ≥ 4} = {x : ℝ | x ≤ -4/3 ∨ x ≥ 0} := sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, 0 < x → f x + a * x - 1 > 0) → a > -5/2 := sorry

end part1_part2_l2262_226263


namespace additional_savings_correct_l2262_226232

def initial_order_amount : ℝ := 10000

def option1_discount1 : ℝ := 0.20
def option1_discount2 : ℝ := 0.20
def option1_discount3 : ℝ := 0.10
def option2_discount1 : ℝ := 0.40
def option2_discount2 : ℝ := 0.05
def option2_discount3 : ℝ := 0.05

def final_price_option1 : ℝ :=
  initial_order_amount * (1 - option1_discount1) *
  (1 - option1_discount2) *
  (1 - option1_discount3)

def final_price_option2 : ℝ :=
  initial_order_amount * (1 - option2_discount1) *
  (1 - option2_discount2) *
  (1 - option2_discount3)

def additional_savings : ℝ :=
  final_price_option1 - final_price_option2

theorem additional_savings_correct : additional_savings = 345 :=
by
  sorry

end additional_savings_correct_l2262_226232


namespace math_proof_problem_l2262_226243

noncomputable def problem_statement : Prop :=
  ∀ (x a b : ℕ), 
  (x + 2 = 5 ∧ x=3) ∧
  (60 / (x + 2) = 36 / x) ∧ 
  (a + b = 90) ∧ 
  (b ≥ 3 * a) ∧ 
  ( ∃ a_max : ℕ, (a_max ≤ a) ∧ (110*a_max + (30*b) = 10520))
  
theorem math_proof_problem : problem_statement := 
  by sorry

end math_proof_problem_l2262_226243


namespace sum_of_decimals_l2262_226293

theorem sum_of_decimals : 5.27 + 4.19 = 9.46 :=
by
  sorry

end sum_of_decimals_l2262_226293


namespace problem_1_problem_2_problem_3_l2262_226262

open Set Real

def U : Set ℝ := univ
def A : Set ℝ := { x | 1 ≤ x ∧ x < 5 }
def B : Set ℝ := { x | 2 < x ∧ x < 8 }
def C (a : ℝ) : Set ℝ := { x | -a < x ∧ x ≤ a + 3 }

theorem problem_1 :
  (A ∪ B) = { x | 1 ≤ x ∧ x < 8 } :=
sorry

theorem problem_2 :
  (U \ A) ∩ B = { x | 5 ≤ x ∧ x < 8 } :=
sorry

theorem problem_3 (a : ℝ) (h : C a ∩ A = C a) :
  a ≤ -1 :=
sorry

end problem_1_problem_2_problem_3_l2262_226262


namespace probability_heads_at_least_10_out_of_12_l2262_226204

theorem probability_heads_at_least_10_out_of_12 (n m : Nat) (hn : n = 12) (hm : m = 10):
  let total_outcomes := 2^n
  let ways_10 := Nat.choose n m
  let ways_11 := Nat.choose n (m + 1)
  let ways_12 := Nat.choose n (m + 2)
  let successful_outcomes := ways_10 + ways_11 + ways_12
  total_outcomes = 4096 →
  successful_outcomes = 79 →
  (successful_outcomes : ℚ) / total_outcomes = 79 / 4096 :=
by
  sorry

end probability_heads_at_least_10_out_of_12_l2262_226204


namespace length_of_AD_l2262_226276

-- Define the segment AD and points B, C, and M as given conditions
variable (x : ℝ) -- Assuming x is the length of segments AB, BC, CD
variable (AD : ℝ)
variable (MC : ℝ)

-- Conditions given in the problem statement
def trisect (AD : ℝ) : Prop :=
  ∃ (x : ℝ), AD = 3 * x ∧ 0 < x

def one_third_way (M AD : ℝ) : Prop :=
  M = AD / 3

def distance_MC (M C : ℝ) : ℝ :=
  C - M

noncomputable def D : Prop := sorry

-- The main theorem statement
theorem length_of_AD (AD : ℝ) (M : ℝ) (MC : ℝ) : trisect AD → one_third_way M AD → MC = M / 3 → AD = 15 :=
by
  intro H1 H2 H3
  -- sorry is added to skip the actual proof
  sorry

end length_of_AD_l2262_226276


namespace work_completion_times_l2262_226237

variable {M P S : ℝ} -- Let M, P, and S be work rates for Matt, Peter, and Sarah.

theorem work_completion_times (h1 : M + P + S = 1 / 15)
                             (h2 : 10 * (P + S) = 7 / 15) :
                             (1 / M = 50) ∧ (1 / (P + S) = 150 / 7) :=
by
  -- Proof comes here
  -- Calculation skipped
  sorry

end work_completion_times_l2262_226237


namespace eval_g_l2262_226209

def g (x : ℝ) : ℝ := 3 * x^2 - 5 * x + 7

theorem eval_g : 3 * g 2 + 4 * g (-4) = 327 := 
by
  sorry

end eval_g_l2262_226209


namespace angle_A_measure_find_a_l2262_226234

theorem angle_A_measure (a b c : ℝ) (A B C : ℝ) (h1 : (a + b) * (Real.sin A - Real.sin B) = (c - b) * Real.sin C) :
  A = π / 3 :=
by
  -- proof steps are omitted
  sorry

theorem find_a (a b c : ℝ) (A : ℝ) (h2 : 2 * c = 3 * b) (area : ℝ) (h3 : area = 6 * Real.sqrt 3)
  (h4 : A = π / 3) :
  a = 2 * Real.sqrt 21 / 3 :=
by
  -- proof steps are omitted
  sorry

end angle_A_measure_find_a_l2262_226234


namespace solution_l2262_226216

noncomputable def given_conditions (θ : ℝ) : Prop := 
  let a := (3, 1)
  let b := (Real.sin θ, Real.cos θ)
  (a.1 : ℝ) / b.1 = a.2 / b.2 

theorem solution (θ : ℝ) (h: given_conditions θ) :
  2 + Real.sin θ * Real.cos θ - Real.cos θ ^ 2 = 5 / 2 :=
by
  sorry

end solution_l2262_226216


namespace sum_geometric_sequence_l2262_226213

theorem sum_geometric_sequence (a : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ) (h1 : a 1 = 2) (h2 : ∀ n, 2 * a n - 2 = S n) : 
  S n = 2^(n+1) - 2 :=
sorry

end sum_geometric_sequence_l2262_226213


namespace tim_weekly_payment_l2262_226233

-- Define the given conditions
def hourly_rate_bodyguard : ℕ := 20
def number_bodyguards : ℕ := 2
def hours_per_day : ℕ := 8
def days_per_week : ℕ := 7

-- Define the total weekly payment calculation
def weekly_payment : ℕ := (hourly_rate_bodyguard * number_bodyguards) * hours_per_day * days_per_week

-- The proof statement
theorem tim_weekly_payment : weekly_payment = 2240 := by
  sorry

end tim_weekly_payment_l2262_226233


namespace negation_of_proposition_l2262_226277

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, x > 0 → (x + 1) * Real.exp x > 1)) ↔
  (∃ x₀ : ℝ, x₀ ≤ 0 ∧ (x₀ + 1) * Real.exp x₀ ≤ 1) := 
sorry

end negation_of_proposition_l2262_226277


namespace repeating_decimal_sum_l2262_226289

noncomputable def x : ℚ := 2 / 9
noncomputable def y : ℚ := 1 / 33

theorem repeating_decimal_sum :
  x + y = 25 / 99 :=
by
  -- Note that Lean can automatically simplify rational expressions.
  sorry

end repeating_decimal_sum_l2262_226289


namespace find_values_of_x_l2262_226210

noncomputable def solution_x (x y : ℝ) : Prop :=
  x ≠ 0 ∧ y ≠ 0 ∧ 
  x^2 + 1/y = 13 ∧ 
  y^2 + 1/x = 8 ∧ 
  (x = Real.sqrt 13 ∨ x = -Real.sqrt 13)

theorem find_values_of_x (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h1 : x^2 + 1/y = 13) (h2 : y^2 + 1/x = 8) : x = Real.sqrt 13 ∨ x = -Real.sqrt 13 :=
by { sorry }

end find_values_of_x_l2262_226210


namespace no_real_x_satisfying_quadratic_inequality_l2262_226288

theorem no_real_x_satisfying_quadratic_inequality (a : ℝ) :
  ¬(∃ x : ℝ, x^2 + (a - 1) * x + 1 ≤ 0) ↔ -1 < a ∧ a < 3 :=
by sorry

end no_real_x_satisfying_quadratic_inequality_l2262_226288


namespace inequality_proving_l2262_226247

theorem inequality_proving (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x^2 + y^2 + z^2 = 1) :
  (1 / x + 1 / y + 1 / z) - (x + y + z) ≥ 2 * Real.sqrt 3 :=
by
  sorry

end inequality_proving_l2262_226247


namespace pairs_satisfying_x2_minus_y2_eq_45_l2262_226223

theorem pairs_satisfying_x2_minus_y2_eq_45 :
  (∃ p : Finset (ℕ × ℕ), (∀ (x y : ℕ), ((x, y) ∈ p → x^2 - y^2 = 45) ∧ (∀ (x y : ℕ), (x, y) ∈ p → 0 < x ∧ 0 < y)) ∧ p.card = 3) :=
by
  sorry

end pairs_satisfying_x2_minus_y2_eq_45_l2262_226223


namespace quadratic_binomial_plus_int_l2262_226225

theorem quadratic_binomial_plus_int (y : ℝ) : y^2 + 14*y + 60 = (y + 7)^2 + 11 :=
by sorry

end quadratic_binomial_plus_int_l2262_226225


namespace john_trip_time_30_min_l2262_226226

-- Definitions of the given conditions
variables {D : ℝ} -- Distance John traveled
variables {T : ℝ} -- Time John took
variable (T_john : ℝ) -- Time it took John (in hours)
variable (T_beth : ℝ) -- Time it took Beth (in hours)
variable (D_john : ℝ) -- Distance John traveled (in miles)
variable (D_beth : ℝ) -- Distance Beth traveled (in miles)

-- Given conditions
def john_speed := 40 -- John's speed in mph
def beth_speed := 30 -- Beth's speed in mph
def additional_distance := 5 -- Additional distance Beth traveled in miles
def additional_time := 1 / 3 -- Additional time Beth took in hours

-- Proving the time it took John to complete the trip is 30 minutes (0.5 hours)
theorem john_trip_time_30_min : 
  ∀ (T_john T_beth : ℝ), 
    T_john = (D) / john_speed →
    T_beth = (D + additional_distance) / beth_speed →
    (T_beth = T_john + additional_time) →
    T_john = 1 / 2 :=
by
  intro T_john T_beth
  sorry

end john_trip_time_30_min_l2262_226226


namespace value_of_expression_l2262_226279

theorem value_of_expression (a b : ℝ) (h1 : 3 * a^2 + 9 * a - 21 = 0) (h2 : 3 * b^2 + 9 * b - 21 = 0) :
  (3 * a - 4) * (5 * b - 6) = -27 :=
by
  -- The proof is omitted, place 'sorry' to indicate it.
  sorry

end value_of_expression_l2262_226279


namespace julia_garden_area_l2262_226246

theorem julia_garden_area
  (length perimeter walk_distance : ℝ)
  (h_length : length * 30 = walk_distance)
  (h_perimeter : perimeter * 12 = walk_distance)
  (h_perimeter_def : perimeter = 2 * (length + width))
  (h_walk_distance : walk_distance = 1500) :
  (length * width = 625) :=
by
  sorry

end julia_garden_area_l2262_226246


namespace eliminate_y_by_subtraction_l2262_226241

theorem eliminate_y_by_subtraction (m n : ℝ) :
  (6 * x + m * y = 3) ∧ (2 * x - n * y = -6) →
  (∀ x y : ℝ, 4 * x + (m + n) * y = 9) → (m + n = 0) :=
by
  intros h eq_subtracted
  sorry

end eliminate_y_by_subtraction_l2262_226241


namespace variance_of_dataset_l2262_226298

theorem variance_of_dataset (a : ℝ) 
  (h1 : (4 + a + 5 + 3 + 8) / 5 = a) :
  (1 / 5) * ((4 - a) ^ 2 + (a - a) ^ 2 + (5 - a) ^ 2 + (3 - a) ^ 2 + (8 - a) ^ 2) = 14 / 5 :=
by
  sorry

end variance_of_dataset_l2262_226298


namespace ratio_of_sum_l2262_226278

theorem ratio_of_sum (x y : ℚ) (h1 : 2 * x + y = 6) (h2 : x + 2 * y = 5) : 
  (x + y) / 3 = 11 / 9 := 
by 
  sorry

end ratio_of_sum_l2262_226278


namespace correct_factorization_l2262_226238

theorem correct_factorization :
  ∀ x : ℝ, x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end correct_factorization_l2262_226238


namespace find_y_coordinate_of_C_l2262_226239

def point (x : ℝ) (y : ℝ) : Prop := y^2 = x + 4

def perp_slope (x1 y1 x2 y2 x3 y3 : ℝ) : Prop :=
  (y2 - y1) / (x2 - x1) * (y3 - y2) / (x3 - x2) = -1

def valid_y_coordinate_C (x0 : ℝ) : Prop :=
  x0 ≤ 0 ∨ 4 ≤ x0

theorem find_y_coordinate_of_C (x0 : ℝ) :
  (∀ (x y : ℝ), point x y) →
  (∃ (x2 y2 x3 y3 : ℝ), point x2 y2 ∧ point x3 y3 ∧ perp_slope 0 2 x2 y2 x3 y3) →
  valid_y_coordinate_C x0 :=
sorry

end find_y_coordinate_of_C_l2262_226239


namespace correct_option_l2262_226257

-- Define the conditions
def c1 (a : ℝ) : Prop := (2 * a^2)^3 ≠ 6 * a^6
def c2 (a : ℝ) : Prop := (a^8) / (a^2) ≠ a^4
def c3 (x y : ℝ) : Prop := (4 * x^2 * y) / (-2 * x * y) ≠ -2
def c4 : Prop := Real.sqrt ((-2)^2) = 2

-- The main statement to be proved
theorem correct_option (a x y : ℝ) (h1 : c1 a) (h2 : c2 a) (h3 : c3 x y) (h4 : c4) : c4 :=
by
  apply h4

end correct_option_l2262_226257


namespace largest_common_divisor_476_330_l2262_226291

theorem largest_common_divisor_476_330 :
  ∀ (S₁ S₂ : Finset ℕ), 
    S₁ = {1, 2, 4, 7, 14, 28, 17, 34, 68, 119, 238, 476} → 
    S₂ = {1, 2, 3, 5, 6, 10, 11, 15, 22, 30, 33, 55, 66, 110, 165, 330} → 
    ∃ D, D ∈ S₁ ∧ D ∈ S₂ ∧ ∀ x, x ∈ S₁ ∧ x ∈ S₂ → x ≤ D ∧ D = 2 :=
by
  intros S₁ S₂ hS₁ hS₂
  use 2
  sorry

end largest_common_divisor_476_330_l2262_226291


namespace value_of_a_8_l2262_226271

-- Definitions of the sequence and sum of first n terms
def sum_first_terms (S : ℕ → ℕ) := ∀ n : ℕ, n > 0 → S n = n^2

-- Definition of the term a_n
def a_n (S : ℕ → ℕ) (n : ℕ) := S n - S (n - 1)

-- The theorem we want to prove: a_8 = 15
theorem value_of_a_8 (S : ℕ → ℕ) (h_sum : sum_first_terms S) : a_n S 8 = 15 :=
by
  sorry

end value_of_a_8_l2262_226271


namespace katya_total_notebooks_l2262_226253

-- Definitions based on the conditions provided
def cost_per_notebook : ℕ := 4
def total_rubles : ℕ := 150
def stickers_for_free_notebook : ℕ := 5
def initial_stickers : ℕ := total_rubles / cost_per_notebook

-- Hypothesis stating the total notebooks Katya can obtain
theorem katya_total_notebooks : initial_stickers + (initial_stickers / stickers_for_free_notebook) + 
    ((initial_stickers % stickers_for_free_notebook + (initial_stickers / stickers_for_free_notebook)) / stickers_for_free_notebook) +
    (((initial_stickers % stickers_for_free_notebook + (initial_stickers / stickers_for_free_notebook)) % stickers_for_free_notebook + 1) / stickers_for_free_notebook) = 46 :=
by
  sorry

end katya_total_notebooks_l2262_226253


namespace LukaNeeds24CupsOfWater_l2262_226211

theorem LukaNeeds24CupsOfWater
  (L S W : ℕ)
  (h1 : S = 2 * L)
  (h2 : W = 4 * S)
  (h3 : L = 3) :
  W = 24 := by
  sorry

end LukaNeeds24CupsOfWater_l2262_226211


namespace min_ineq_l2262_226229

theorem min_ineq (x : ℝ) (hx : x > 0) : 3*x + 1/x^2 ≥ 4 :=
sorry

end min_ineq_l2262_226229


namespace sufficient_condition_of_necessary_condition_l2262_226261

-- Define the necessary condition
def necessary_condition (A B : Prop) : Prop := A → B

-- The proof problem statement
theorem sufficient_condition_of_necessary_condition
  {A B : Prop} (h : necessary_condition A B) : necessary_condition A B :=
by
  exact h

end sufficient_condition_of_necessary_condition_l2262_226261


namespace libby_quarters_left_l2262_226283

theorem libby_quarters_left (initial_quarters : ℕ) (dress_cost_dollars : ℕ) (quarters_per_dollar : ℕ) 
  (h1 : initial_quarters = 160) (h2 : dress_cost_dollars = 35) (h3 : quarters_per_dollar = 4) : 
  initial_quarters - (dress_cost_dollars * quarters_per_dollar) = 20 := by
  sorry

end libby_quarters_left_l2262_226283


namespace isoscelesTriangleDistanceFromAB_l2262_226245

-- Given definitions
def isoscelesTriangleAreaInsideEquilateral (t m c x : ℝ) : Prop :=
  let halfEquilateralAltitude := m / 2
  let equilateralTriangleArea := (c^2 * (Real.sqrt 3)) / 4
  let equalsAltitudeCondition := x = m / 2
  let distanceFormula := x = (m + Real.sqrt (m^2 - 4 * t * Real.sqrt 3)) / 2 ∨ x = (m - Real.sqrt (m^2 - 4 * t * Real.sqrt 3)) / 2
  (2 * t = halfEquilateralAltitude * c / 2) ∧ 
  equalsAltitudeCondition ∧ distanceFormula

-- The theorem to prove given the above definition
theorem isoscelesTriangleDistanceFromAB (t m c x : ℝ) :
  isoscelesTriangleAreaInsideEquilateral t m c x →
  x = (m + Real.sqrt (m^2 - 4 * t * Real.sqrt 3)) / 2 ∨ x = (m - Real.sqrt (m^2 - 4 * t * Real.sqrt 3)) / 2 :=
sorry

end isoscelesTriangleDistanceFromAB_l2262_226245


namespace find_fx_sum_roots_l2262_226208

noncomputable def f : ℝ → ℝ
| x => if x = 2 then 1 else Real.log (abs (x - 2))

theorem find_fx_sum_roots
  (b c : ℝ)
  (x1 x2 x3 x4 x5 : ℝ)
  (h : ∀ x, (f x) ^ 2 + b * (f x) + c = 0)
  (h_distinct : x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x1 ≠ x5 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x2 ≠ x5 ∧ x3 ≠ x4 ∧ x3 ≠ x5 ∧ x4 ≠ x5 ) :
  f (x1 + x2 + x3 + x4 + x5) = Real.log 8 :=
sorry

end find_fx_sum_roots_l2262_226208


namespace inequality_abc_l2262_226255

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 1) : 
  (1 / (1 + 2 * a)) + (1 / (1 + 2 * b)) + (1 / (1 + 2 * c)) ≥ 1 :=
by
  sorry

end inequality_abc_l2262_226255


namespace logan_snowfall_total_l2262_226231

theorem logan_snowfall_total (wednesday thursday friday : ℝ) :
  wednesday = 0.33 → thursday = 0.33 → friday = 0.22 → wednesday + thursday + friday = 0.88 :=
by
  intros hw ht hf
  rw [hw, ht, hf]
  exact (by norm_num : (0.33 : ℝ) + 0.33 + 0.22 = 0.88)

end logan_snowfall_total_l2262_226231


namespace area_of_BCD_l2262_226292

theorem area_of_BCD (S_ABC : ℝ) (a_CD : ℝ) (h_ratio : ℝ) (h_ABC : ℝ) :
  S_ABC = 36 ∧ a_CD = 30 ∧ h_ratio = 0.5 ∧ h_ABC = 12 → 
  (1 / 2) * a_CD * (h_ratio * h_ABC) = 90 :=
by
  intros h
  sorry

end area_of_BCD_l2262_226292
