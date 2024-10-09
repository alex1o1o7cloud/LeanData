import Mathlib

namespace train_length_proof_l1418_141855

noncomputable def train_speed_kmh : ℝ := 50
noncomputable def crossing_time_s : ℝ := 9
noncomputable def length_of_train_m : ℝ := 125

theorem train_length_proof:
  ∀ (speed_kmh: ℝ) (time_s: ℝ), 
  speed_kmh = train_speed_kmh →
  time_s = crossing_time_s →
  (speed_kmh * (1000 / 3600) * time_s) = length_of_train_m :=
by intros speed_kmh time_s h_speed_kmh h_time_s
   -- Proof omitted
   sorry

end train_length_proof_l1418_141855


namespace arithmetic_operations_result_eq_one_over_2016_l1418_141851

theorem arithmetic_operations_result_eq_one_over_2016 :
  (∃ op1 op2 : ℚ → ℚ → ℚ, op1 (1/8) (op2 (1/9) (1/28)) = 1/2016) :=
sorry

end arithmetic_operations_result_eq_one_over_2016_l1418_141851


namespace bake_sale_total_money_l1418_141820

def dozens_to_pieces (dozens : Nat) : Nat :=
  dozens * 12

def total_money_raised
  (betty_chocolate_chip_dozen : Nat)
  (betty_oatmeal_raisin_dozen : Nat)
  (betty_brownies_dozen : Nat)
  (paige_sugar_cookies_dozen : Nat)
  (paige_blondies_dozen : Nat)
  (paige_cream_cheese_brownies_dozen : Nat)
  (price_per_cookie : Rat)
  (price_per_brownie_blondie : Rat) : Rat :=
let betty_cookies := dozens_to_pieces betty_chocolate_chip_dozen + dozens_to_pieces betty_oatmeal_raisin_dozen
let paige_cookies := dozens_to_pieces paige_sugar_cookies_dozen
let total_cookies := betty_cookies + paige_cookies
let betty_brownies := dozens_to_pieces betty_brownies_dozen
let paige_brownies_blondies := dozens_to_pieces paige_blondies_dozen + dozens_to_pieces paige_cream_cheese_brownies_dozen
let total_brownies_blondies := betty_brownies + paige_brownies_blondies
(total_cookies * price_per_cookie) + (total_brownies_blondies * price_per_brownie_blondie)

theorem bake_sale_total_money :
  total_money_raised 4 6 2 6 3 5 1 2 = 432 :=
by
  sorry

end bake_sale_total_money_l1418_141820


namespace evaluate_f_at_1_l1418_141896

def f (x : ℝ) : ℝ := x^2 + 2*x + 3

theorem evaluate_f_at_1 : f 1 = 6 := 
  sorry

end evaluate_f_at_1_l1418_141896


namespace find_theta_l1418_141885

noncomputable def f (x θ : ℝ) : ℝ := Real.sin (2 * x + θ)
noncomputable def g (x θ : ℝ) : ℝ := Real.sin (2 * (x + Real.pi / 8) + θ)

def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

theorem find_theta (θ : ℝ) : 
  (∀ x, g x θ = g (-x) θ) → θ = Real.pi / 4 :=
by
  intros h
  sorry

end find_theta_l1418_141885


namespace range_of_m_l1418_141802

noncomputable def quadratic_inequality_solution_set_is_R (m : ℝ) : Prop :=
  ∀ x : ℝ, (m - 1) * x^2 + (m - 1) * x + 2 > 0

theorem range_of_m :
  { m : ℝ | quadratic_inequality_solution_set_is_R m } = { m : ℝ | 1 ≤ m ∧ m < 9 } :=
by
  sorry

end range_of_m_l1418_141802


namespace max_value_2ab_2bc_2cd_2da_l1418_141841

theorem max_value_2ab_2bc_2cd_2da {a b c d : ℕ} :
  (a = 2 ∨ a = 3 ∨ a = 5 ∨ a = 7) ∧
  (b = 2 ∨ b = 3 ∨ b = 5 ∨ b = 7) ∧
  (c = 2 ∨ c = 3 ∨ c = 5 ∨ c = 7) ∧
  (d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7) ∧
  (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧
  (b ≠ c) ∧ (b ≠ d) ∧
  (c ≠ d)
  → 2 * (a * b + b * c + c * d + d * a) ≤ 144 :=
by
  sorry

end max_value_2ab_2bc_2cd_2da_l1418_141841


namespace auditorium_total_chairs_l1418_141854

theorem auditorium_total_chairs 
  (n : ℕ)
  (h1 : 2 + 5 - 1 = n)   -- n is the number of rows which is equal to 6
  (h2 : 3 + 4 - 1 = n)   -- n is the number of chairs per row which is also equal to 6
  : n * n = 36 :=        -- the total number of chairs is 36
by
  sorry

end auditorium_total_chairs_l1418_141854


namespace no_very_convex_function_exists_l1418_141876

-- Definition of very convex function
def very_convex (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, (f x + f y) / 2 ≥ f ((x + y) / 2) + |x - y|

-- Theorem stating the non-existence of very convex functions
theorem no_very_convex_function_exists : ¬∃ f : ℝ → ℝ, very_convex f :=
by {
  sorry
}

end no_very_convex_function_exists_l1418_141876


namespace solve_fraction_equation_l1418_141856

theorem solve_fraction_equation (x : ℝ) (h₁ : x ≠ 0) (h₂ : x ≠ 1) :
  (3 / (x - 1) - 2 / x = 0) ↔ x = -2 := by
sorry

end solve_fraction_equation_l1418_141856


namespace age_sum_l1418_141836

-- Defining the ages of Henry and Jill
def Henry_age : ℕ := 20
def Jill_age : ℕ := 13

-- The statement we need to prove
theorem age_sum : Henry_age + Jill_age = 33 := by
  -- Proof goes here
  sorry

end age_sum_l1418_141836


namespace ferry_tourist_total_l1418_141809

theorem ferry_tourist_total :
  let number_of_trips := 8
  let a := 120 -- initial number of tourists
  let d := -2  -- common difference
  let total_tourists := (number_of_trips * (2 * a + (number_of_trips - 1) * d)) / 2
  total_tourists = 904 := 
by {
  sorry
}

end ferry_tourist_total_l1418_141809


namespace sin_300_eq_neg_sqrt3_div_2_l1418_141897

/-- Prove that the sine of 300 degrees is equal to -√3/2. -/
theorem sin_300_eq_neg_sqrt3_div_2 : Real.sin (300 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
sorry

end sin_300_eq_neg_sqrt3_div_2_l1418_141897


namespace towel_bleach_decrease_l1418_141828

theorem towel_bleach_decrease (L B L' B' A A' : ℝ)
    (hB : B' = 0.6 * B)
    (hA : A' = 0.42 * A)
    (hA_def : A = L * B)
    (hA'_def : A' = L' * B') :
    L' = 0.7 * L :=
by
  sorry

end towel_bleach_decrease_l1418_141828


namespace minimum_hexagon_perimeter_l1418_141832

-- Define the conditions given in the problem
def small_equilateral_triangle (side_length : ℝ) (triangle_count : ℕ) :=
  triangle_count = 57 ∧ side_length = 1

def hexagon_with_conditions (angle_condition : ℝ → Prop) :=
  ∀ θ, angle_condition θ → θ ≤ 180 ∧ θ > 0

-- State the main problem as a theorem
theorem minimum_hexagon_perimeter : ∀ n : ℕ, ∃ p : ℕ,
  (small_equilateral_triangle 1 57) → 
  (∃ angle_condition, hexagon_with_conditions angle_condition) →
  (n = 57) →
  p = 19 :=
by
  sorry

end minimum_hexagon_perimeter_l1418_141832


namespace find_number_of_terms_l1418_141812

theorem find_number_of_terms (n : ℕ) (a : ℕ → ℚ) (S : ℕ → ℚ) :
  (∀ n, a n = (2^n - 1) / (2^n)) → S n = 321 / 64 → n = 6 :=
by
  sorry

end find_number_of_terms_l1418_141812


namespace quadrilateral_area_l1418_141871

theorem quadrilateral_area (d h1 h2 : ℝ) (hd : d = 40) (hh1 : h1 = 9) (hh2 : h2 = 6) :
  1 / 2 * d * h1 + 1 / 2 * d * h2 = 300 := 
by
  sorry

end quadrilateral_area_l1418_141871


namespace driver_travel_distance_per_week_l1418_141842

noncomputable def daily_distance := 30 * 3 + 25 * 4 + 40 * 2

noncomputable def total_weekly_distance := daily_distance * 6 + 35 * 5

theorem driver_travel_distance_per_week : total_weekly_distance = 1795 := by
  simp [daily_distance, total_weekly_distance]
  done

end driver_travel_distance_per_week_l1418_141842


namespace jim_anne_mary_paul_report_time_l1418_141819

def typing_rate_jim := 1 / 12
def typing_rate_anne := 1 / 20
def combined_typing_rate := typing_rate_jim + typing_rate_anne
def typing_time := 1 / combined_typing_rate

def editing_rate_mary := 1 / 30
def editing_rate_paul := 1 / 10
def combined_editing_rate := editing_rate_mary + editing_rate_paul
def editing_time := 1 / combined_editing_rate

theorem jim_anne_mary_paul_report_time : 
  typing_time + editing_time = 15 := by
  sorry

end jim_anne_mary_paul_report_time_l1418_141819


namespace factorize_m_factorize_x_factorize_xy_l1418_141825

theorem factorize_m (m : ℝ) : m^2 + 7 * m - 18 = (m - 2) * (m + 9) := 
sorry

theorem factorize_x (x : ℝ) : x^2 - 2 * x - 8 = (x + 2) * (x - 4) :=
sorry

theorem factorize_xy (x y : ℝ) : (x * y)^2 - 7 * (x * y) + 10 = (x * y - 2) * (x * y - 5) := 
sorry

end factorize_m_factorize_x_factorize_xy_l1418_141825


namespace fg_of_2_eq_15_l1418_141891

def g (x : ℝ) : ℝ := x^3
def f (x : ℝ) : ℝ := 2 * x - 1

theorem fg_of_2_eq_15 : f (g 2) = 15 :=
by
  -- The detailed proof would go here
  sorry

end fg_of_2_eq_15_l1418_141891


namespace hash_difference_l1418_141800

def hash (x y : ℕ) : ℤ := x * y - 3 * x + y

theorem hash_difference : (hash 8 5) - (hash 5 8) = -12 := by
  sorry

end hash_difference_l1418_141800


namespace units_digit_of_17_pow_3_mul_24_l1418_141821

def unit_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_17_pow_3_mul_24 :
  unit_digit (17^3 * 24) = 2 :=
by
  sorry

end units_digit_of_17_pow_3_mul_24_l1418_141821


namespace find_the_number_l1418_141873

theorem find_the_number (n : ℤ) 
    (h : 45 - (28 - (n - (15 - 18))) = 57) :
    n = 37 := 
sorry

end find_the_number_l1418_141873


namespace first_two_cards_black_prob_l1418_141857

noncomputable def probability_first_two_black : ℚ :=
  let total_cards := 52
  let black_cards := 26
  let first_draw_prob := black_cards / total_cards
  let second_draw_prob := (black_cards - 1) / (total_cards - 1)
  first_draw_prob * second_draw_prob

theorem first_two_cards_black_prob :
  probability_first_two_black = 25 / 102 :=
by
  sorry

end first_two_cards_black_prob_l1418_141857


namespace popsicle_count_l1418_141843

-- Define the number of each type of popsicles
def num_grape_popsicles : Nat := 2
def num_cherry_popsicles : Nat := 13
def num_banana_popsicles : Nat := 2

-- Prove the total number of popsicles
theorem popsicle_count : num_grape_popsicles + num_cherry_popsicles + num_banana_popsicles = 17 := by
  sorry

end popsicle_count_l1418_141843


namespace unsatisfactory_tests_l1418_141835

theorem unsatisfactory_tests {n k : ℕ} (h1 : n < 50) 
  (h2 : n % 7 = 0) 
  (h3 : n % 3 = 0) 
  (h4 : n % 2 = 0)
  (h5 : n = 7 * (n / 7) + 3 * (n / 3) + 2 * (n / 2) + k) : 
  k = 1 := 
by 
  sorry

end unsatisfactory_tests_l1418_141835


namespace intervals_of_monotonicity_of_f_l1418_141848

noncomputable def f (a b c d : ℝ) (x : ℝ) := a * x^3 + b * x^2 + c * x + d

theorem intervals_of_monotonicity_of_f (a b c d : ℝ)
  (h1 : ∃ P : ℝ × ℝ, P.1 = 0 ∧ d = P.2 ∧ (12 * P.1 - P.2 - 4 = 0))
  (h2 : ∃ x : ℝ, x = 2 ∧ (f a b c d x = 0) ∧ (∃ x : ℝ, x = 0 ∧ (3 * a * x^2 + 2 * b * x + c = 12))) 
  : ( ∃ a b c d : ℝ , (f a b c d) = (2 * x^3 - 9 * x^2 + 12 * x -4)) := 
  sorry

end intervals_of_monotonicity_of_f_l1418_141848


namespace exponentiation_property_l1418_141829

variable (a : ℝ)

theorem exponentiation_property : a^2 * a^3 = a^5 := by
  sorry

end exponentiation_property_l1418_141829


namespace equilateral_triangle_roots_l1418_141845

theorem equilateral_triangle_roots (p q : ℂ) (z1 z2 : ℂ) (h1 : z2 = Complex.exp (2 * Real.pi * Complex.I / 3) * z1)
  (h2 : 0 + p * z1 + q = 0) (h3 : p = -z1 - z2) (h4 : q = z1 * z2) : (p^2 / q) = 1 :=
by
  sorry

end equilateral_triangle_roots_l1418_141845


namespace min_third_side_of_right_triangle_l1418_141868

theorem min_third_side_of_right_triangle (a b : ℕ) (h1 : a = 4) (h2 : b = 5) :
  ∃ c : ℕ, (min c (4 + 5 - 3) - (4 - 3)) = 3 :=
sorry

end min_third_side_of_right_triangle_l1418_141868


namespace sin_cos_15_deg_l1418_141811

noncomputable def sin_deg (deg : ℝ) : ℝ := Real.sin (deg * Real.pi / 180)
noncomputable def cos_deg (deg : ℝ) : ℝ := Real.cos (deg * Real.pi / 180)

theorem sin_cos_15_deg :
  (sin_deg 15 + cos_deg 15) * (sin_deg 15 - cos_deg 15) = -Real.sqrt 3 / 2 :=
by
  sorry

end sin_cos_15_deg_l1418_141811


namespace father_age_l1418_141899

variable (F S x : ℕ)

-- Conditions
axiom h1 : F + S = 75
axiom h2 : F = 8 * (S - x)
axiom h3 : F - x = S

-- Theorem to prove
theorem father_age : F = 48 :=
sorry

end father_age_l1418_141899


namespace rationalize_value_of_a2_minus_2a_value_of_2a3_minus_4a2_minus_1_l1418_141858

variable (a : ℂ)

theorem rationalize (h : a = 1 / (Real.sqrt 2 - 1)) : a = Real.sqrt 2 + 1 := by
  sorry

theorem value_of_a2_minus_2a (h : a = Real.sqrt 2 + 1) : a ^ 2 - 2 * a = 1 := by
  sorry

theorem value_of_2a3_minus_4a2_minus_1 (h : a = Real.sqrt 2 + 1) : 2 * a ^ 3 - 4 * a ^ 2 - 1 = 2 * Real.sqrt 2 + 1 := by
  sorry

end rationalize_value_of_a2_minus_2a_value_of_2a3_minus_4a2_minus_1_l1418_141858


namespace license_plate_count_l1418_141867

theorem license_plate_count :
  let letters := 26
  let digits := 10
  let second_char_options := letters - 1 + digits
  let third_char_options := digits - 1
  letters * second_char_options * third_char_options = 8190 :=
by
  sorry

end license_plate_count_l1418_141867


namespace smallest_n_for_multiples_of_7_l1418_141827

theorem smallest_n_for_multiples_of_7 (x y : ℤ) (h1 : x ≡ 4 [ZMOD 7]) (h2 : y ≡ 5 [ZMOD 7]) :
  ∃ n : ℕ, 0 < n ∧ (x^2 + x * y + y^2 + n ≡ 0 [ZMOD 7]) ∧ ∀ m : ℕ, 0 < m ∧ (x^2 + x * y + y^2 + m ≡ 0 [ZMOD 7]) → n ≤ m :=
by
  sorry

end smallest_n_for_multiples_of_7_l1418_141827


namespace divide_coal_l1418_141892

noncomputable def part_of_pile (whole: ℚ) (parts: ℕ) := whole / parts
noncomputable def part_tons (total_tons: ℚ) (fraction: ℚ) := total_tons * fraction

theorem divide_coal (total_tons: ℚ) (parts: ℕ) (h: total_tons = 3 ∧ parts = 5):
  (part_of_pile 1 parts = 1/parts) ∧ (part_tons total_tons (1/parts) = total_tons / parts) :=
by
  sorry

end divide_coal_l1418_141892


namespace paving_cost_l1418_141888

def length : ℝ := 5.5
def width : ℝ := 3.75
def rate : ℝ := 1000
def area : ℝ := length * width
def cost : ℝ := area * rate

theorem paving_cost :
  cost = 20625 := by sorry

end paving_cost_l1418_141888


namespace height_difference_l1418_141801

variables (H1 H2 H3 : ℕ)
variable (x : ℕ)
variable (h_ratio : H1 = 4 * x ∧ H2 = 5 * x ∧ H3 = 6 * x)
variable (h_lightest : H1 = 120)

theorem height_difference :
  (H1 + H3) - H2 = 150 :=
by
  -- Proof will go here
  sorry

end height_difference_l1418_141801


namespace distance_on_dirt_road_l1418_141893

theorem distance_on_dirt_road :
  ∀ (initial_gap distance_gap_on_city dirt_road_distance : ℝ),
  initial_gap = 2 → 
  distance_gap_on_city = initial_gap - ((initial_gap - (40 * (1 / 30)))) → 
  dirt_road_distance = distance_gap_on_city * (40 / 60) * (70 / 40) * (30 / 70) →
  dirt_road_distance = 1 :=
by
  intros initial_gap distance_gap_on_city dirt_road_distance h1 h2 h3
  -- The proof would go here
  sorry

end distance_on_dirt_road_l1418_141893


namespace enemy_defeat_points_l1418_141889

theorem enemy_defeat_points 
    (points_per_enemy : ℕ) (total_enemies : ℕ) (undefeated_enemies : ℕ) (defeated : ℕ) (points_earned : ℕ) :
    points_per_enemy = 8 →
    total_enemies = 7 →
    undefeated_enemies = 2 →
    defeated = total_enemies - undefeated_enemies →
    points_earned = defeated * points_per_enemy →
    points_earned = 40 :=
by
  intros
  sorry

end enemy_defeat_points_l1418_141889


namespace avg_children_in_families_with_children_l1418_141837

noncomputable def avg_children_with_children (total_families : ℕ) (avg_children : ℝ) (childless_families : ℕ) : ℝ :=
  let total_children := total_families * avg_children
  let families_with_children := total_families - childless_families
  total_children / families_with_children

theorem avg_children_in_families_with_children :
  avg_children_with_children 15 3 3 = 3.8 := by
  sorry

end avg_children_in_families_with_children_l1418_141837


namespace triangle_angle_sum_property_l1418_141895

theorem triangle_angle_sum_property (A B C : ℝ) (h1: C = 3 * B) (h2: B = 15) : A = 120 :=
by
  -- Proof goes here
  sorry

end triangle_angle_sum_property_l1418_141895


namespace average_price_blankets_l1418_141869

theorem average_price_blankets :
  let cost_blankets1 := 3 * 100
  let cost_blankets2 := 5 * 150
  let cost_blankets3 := 550
  let total_cost := cost_blankets1 + cost_blankets2 + cost_blankets3
  let total_blankets := 3 + 5 + 2
  total_cost / total_blankets = 160 :=
by
  sorry

end average_price_blankets_l1418_141869


namespace correct_operation_l1418_141878

theorem correct_operation :
  (2 * a - a ≠ 2) ∧ ((a - 1) * (a - 1) ≠ a ^ 2 - 1) ∧ (a ^ 6 / a ^ 3 ≠ a ^ 2) ∧ ((-2 * a ^ 3) ^ 2 = 4 * a ^ 6) :=
by
  sorry

end correct_operation_l1418_141878


namespace perimeter_of_polygon_is_15_l1418_141898

-- Definitions for the problem conditions
def side_length_of_square : ℕ := 5
def fraction_of_square_occupied (n : ℕ) : ℚ := 3 / 4

-- Problem statement: Prove that the perimeter of the polygon is 15 units
theorem perimeter_of_polygon_is_15 :
  4 * side_length_of_square * (fraction_of_square_occupied side_length_of_square) = 15 := 
by
  sorry

end perimeter_of_polygon_is_15_l1418_141898


namespace find_ivans_number_l1418_141847

theorem find_ivans_number :
  ∃ (a b c d e f g h i j k l : ℕ),
    10 ≤ a ∧ a < 100 ∧
    10 ≤ b ∧ b < 100 ∧
    10 ≤ c ∧ c < 100 ∧
    10 ≤ d ∧ d < 100 ∧
    1000 ≤ e ∧ e < 10000 ∧
    (a * 10^10 + b * 10^8 + c * 10^6 + d * 10^4 + e) = 132040530321 := sorry

end find_ivans_number_l1418_141847


namespace inequality_proof_l1418_141805

theorem inequality_proof (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (h : x^2 + y^2 + z^2 = 2 * (x * y + y * z + z * x)) :
  (x + y + z) / 3 ≥ (2 * x * y * z)^(1/3 : ℝ) :=
by
  sorry

end inequality_proof_l1418_141805


namespace tiffany_initial_lives_l1418_141817

variable (x : ℝ) -- Define the variable x representing the initial number of lives

-- Define the conditions
def condition1 : Prop := x + 14.0 + 27.0 = 84.0

-- Prove the initial number of lives
theorem tiffany_initial_lives (h : condition1 x) : x = 43.0 := by
  sorry

end tiffany_initial_lives_l1418_141817


namespace Mildred_final_oranges_l1418_141803

def initial_oranges : ℕ := 215
def father_oranges : ℕ := 3 * initial_oranges
def total_after_father : ℕ := initial_oranges + father_oranges
def sister_takes_away : ℕ := 174
def after_sister : ℕ := total_after_father - sister_takes_away
def final_oranges : ℕ := 2 * after_sister

theorem Mildred_final_oranges : final_oranges = 1372 := by
  sorry

end Mildred_final_oranges_l1418_141803


namespace train_length_l1418_141834

theorem train_length (v : ℝ) (t : ℝ) (conversion_factor : ℝ) : v = 45 → t = 16 → conversion_factor = 1000 / 3600 → (v * (conversion_factor) * t) = 200 :=
  by
  intros hv ht hcf
  rw [hv, ht, hcf]
  -- Proof steps skipped
  sorry

end train_length_l1418_141834


namespace expected_value_twelve_sided_die_l1418_141826

theorem expected_value_twelve_sided_die : 
  (1 / 12 * (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12)) = 6.5 := by
  sorry

end expected_value_twelve_sided_die_l1418_141826


namespace vacationers_city_correctness_l1418_141818

noncomputable def vacationer_cities : Prop :=
  ∃ (city : String → String),
    (city "Amelie" = "Acapulco" ∨ city "Amelie" = "Brest" ∨ city "Amelie" = "Madrid") ∧
    (city "Benoit" = "Acapulco" ∨ city "Benoit" = "Brest" ∨ city "Benoit" = "Madrid") ∧
    (city "Pierre" = "Paris" ∨ city "Pierre" = "Brest" ∨ city "Pierre" = "Madrid") ∧
    (city "Melanie" = "Acapulco" ∨ city "Melanie" = "Brest" ∨ city "Melanie" = "Madrid") ∧
    (city "Charles" = "Acapulco" ∨ city "Charles" = "Brest" ∨ city "Charles" = "Madrid") ∧
    -- Conditions stated by participants
    ((city "Amelie" = "Acapulco") ∨ (city "Amelie" ≠ "Acapulco" ∧ city "Benoit" = "Acapulco" ∧ city "Pierre" = "Paris")) ∧
    ((city "Benoit" = "Brest") ∨ (city "Benoit" ≠ "Brest" ∧ city "Charles" = "Brest" ∧ city "Pierre" = "Paris")) ∧
    ((city "Pierre" ≠ "France") ∨ (city "Pierre" = "Paris" ∧ city "Amelie" ≠ "France" ∧ city "Melanie" = "Madrid")) ∧
    ((city "Melanie" = "Clermont-Ferrand") ∨ (city "Melanie" ≠ "Clermont-Ferrand" ∧ city "Amelie" = "Acapulco" ∧ city "Pierre" = "Paris")) ∧
    ((city "Charles" = "Clermont-Ferrand") ∨ (city "Charles" ≠ "Clermont-Ferrand" ∧ city "Amelie" = "Acapulco" ∧ city "Benoit" = "Acapulco"))

theorem vacationers_city_correctness : vacationer_cities :=
  sorry

end vacationers_city_correctness_l1418_141818


namespace shift_down_two_units_l1418_141872

def original_function (x : ℝ) : ℝ := 2 * x + 1

def shifted_function (x : ℝ) : ℝ := original_function x - 2

theorem shift_down_two_units :
  ∀ x : ℝ, shifted_function x = 2 * x - 1 :=
by 
  intros x
  simp [shifted_function, original_function]
  sorry

end shift_down_two_units_l1418_141872


namespace range_of_m_l1418_141882

theorem range_of_m (m : ℝ) : (∀ x : ℝ, x^2 - m * x + 1 > 0) ↔ (-2 < m ∧ m < 2) :=
  sorry

end range_of_m_l1418_141882


namespace total_amount_paid_l1418_141824

-- Define the quantities and rates as constants
def quantity_grapes : ℕ := 8
def rate_grapes : ℕ := 70
def quantity_mangoes : ℕ := 9
def rate_mangoes : ℕ := 55

-- Define the cost functions
def cost_grapes (q : ℕ) (r : ℕ) : ℕ := q * r
def cost_mangoes (q : ℕ) (r : ℕ) : ℕ := q * r

-- Define the total cost function
def total_cost (c1 : ℕ) (c2 : ℕ) : ℕ := c1 + c2

-- State the proof problem
theorem total_amount_paid :
  total_cost (cost_grapes quantity_grapes rate_grapes) (cost_mangoes quantity_mangoes rate_mangoes) = 1055 :=
by
  sorry

end total_amount_paid_l1418_141824


namespace oven_clock_actual_time_l1418_141861

theorem oven_clock_actual_time :
  ∀ (h : ℕ), (oven_time : h = 10) →
  (oven_gains : ℕ) = 8 →
  (initial_time : ℕ) = 18 →          
  (initial_wall_time : ℕ) = 18 →
  (wall_time_after_one_hour : ℕ) = 19 →
  (oven_time_after_one_hour : ℕ) = 19 + 8/60 →
  ℕ := sorry

end oven_clock_actual_time_l1418_141861


namespace number_of_blue_fish_l1418_141833

def total_fish : ℕ := 22
def goldfish : ℕ := 15
def blue_fish : ℕ := total_fish - goldfish

theorem number_of_blue_fish : blue_fish = 7 :=
by
  -- proof goes here
  sorry

end number_of_blue_fish_l1418_141833


namespace tommys_profit_l1418_141890

-- Definitions of the conditions
def crateA_cost : ℕ := 220
def crateB_cost : ℕ := 375
def crateC_cost : ℕ := 180

def crateA_count : ℕ := 2
def crateB_count : ℕ := 3
def crateC_count : ℕ := 1

def crateA_capacity : ℕ := 20
def crateB_capacity : ℕ := 25
def crateC_capacity : ℕ := 30

def crateA_rotten : ℕ := 4
def crateB_rotten : ℕ := 5
def crateC_rotten : ℕ := 3

def crateA_price_per_kg : ℕ := 5
def crateB_price_per_kg : ℕ := 6
def crateC_price_per_kg : ℕ := 7

-- Calculations based on the conditions
def total_cost : ℕ := crateA_cost + crateB_cost + crateC_cost

def sellable_weightA : ℕ := crateA_count * crateA_capacity - crateA_rotten
def sellable_weightB : ℕ := crateB_count * crateB_capacity - crateB_rotten
def sellable_weightC : ℕ := crateC_count * crateC_capacity - crateC_rotten

def revenueA : ℕ := sellable_weightA * crateA_price_per_kg
def revenueB : ℕ := sellable_weightB * crateB_price_per_kg
def revenueC : ℕ := sellable_weightC * crateC_price_per_kg

def total_revenue : ℕ := revenueA + revenueB + revenueC

def profit : ℕ := total_revenue - total_cost

-- The theorem we want to verify
theorem tommys_profit : profit = 14 := by
  sorry

end tommys_profit_l1418_141890


namespace find_k_l1418_141830

theorem find_k 
  (A B X Y : ℝ × ℝ)
  (hA : A = (-3, 0))
  (hB : B = (0, -3))
  (hX : X = (0, 9))
  (Yx : Y.1 = 15)
  (hXY_parallel : (Y.2 - X.2) / (Y.1 - X.1) = (B.2 - A.2) / (B.1 - A.1)) :
  Y.2 = -6 := by
  -- proofs are omitted as per the requirements
  sorry

end find_k_l1418_141830


namespace x_y_difference_is_perfect_square_l1418_141831

theorem x_y_difference_is_perfect_square (x y : ℕ) (h : 3 * x^2 + x = 4 * y^2 + y) : ∃ k : ℕ, k^2 = x - y :=
by {sorry}

end x_y_difference_is_perfect_square_l1418_141831


namespace equations_have_different_graphs_l1418_141849

theorem equations_have_different_graphs :
  ¬(∀ x : ℝ, (2 * (x - 3)) / (x + 3) = 2 * (x - 3) ∧ 
              (x + 3) * ((2 * x^2 - 18) / (x + 3)) = 2 * x^2 - 18 ∧
              (2 * x - 3) = (2 * (x - 3)) ∧ 
              (2 * x - 3) = (2 * x - 3)) :=
by
  sorry

end equations_have_different_graphs_l1418_141849


namespace maximum_term_of_sequence_l1418_141894

noncomputable def a (n : ℕ) : ℝ := n * (3 / 4)^n

theorem maximum_term_of_sequence : ∃ n : ℕ, a n = a 3 ∧ ∀ m : ℕ, a m ≤ a 3 :=
by sorry

end maximum_term_of_sequence_l1418_141894


namespace selling_price_correct_l1418_141814

noncomputable def cost_price : ℝ := 100
noncomputable def gain_percent : ℝ := 0.15
noncomputable def profit : ℝ := gain_percent * cost_price
noncomputable def selling_price : ℝ := cost_price + profit

theorem selling_price_correct : selling_price = 115 := by
  sorry

end selling_price_correct_l1418_141814


namespace unique_solution_l1418_141879

noncomputable def is_solution (f : ℝ → ℝ) : Prop :=
    (∀ x, x ≥ 1 → f x ≤ 2 * (x + 1)) ∧
    (∀ x, x ≥ 1 → f (x + 1) = (1 / x) * ((f x)^2 - 1))

theorem unique_solution (f : ℝ → ℝ) :
    is_solution f → (∀ x, x ≥ 1 → f x = x + 1) := 
sorry

end unique_solution_l1418_141879


namespace decompose_two_over_eleven_decompose_two_over_n_l1418_141862

-- Problem 1: Decompose 2/11
theorem decompose_two_over_eleven : (2 : ℚ) / 11 = (1 / 6) + (1 / 66) :=
  sorry

-- Problem 2: General form for 2/n for odd n >= 5
theorem decompose_two_over_n (n : ℕ) (hn : n ≥ 5) (odd_n : n % 2 = 1) :
  (2 : ℚ) / n = (1 / ((n + 1) / 2)) + (1 / (n * (n + 1) / 2)) :=
  sorry

end decompose_two_over_eleven_decompose_two_over_n_l1418_141862


namespace log_monotonic_increasing_l1418_141860

noncomputable def f (a x : ℝ) := Real.log x / Real.log a

theorem log_monotonic_increasing (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) (h3 : 1 < a) :
  f a (a + 1) > f a 2 := 
by
  -- Here the actual proof will be added.
  sorry

end log_monotonic_increasing_l1418_141860


namespace not_less_than_x3_y5_for_x2y_l1418_141864

theorem not_less_than_x3_y5_for_x2y (x y : ℝ) (hx : 0 < x ∧ x < 1) (hy : 0 < y ∧ y < 1) : x^2 * y ≥ x^3 + y^5 :=
sorry

end not_less_than_x3_y5_for_x2y_l1418_141864


namespace determine_c_l1418_141865

theorem determine_c (c : ℝ) (r : ℝ) (h1 : 2 * r^2 - 8 * r - c = 0) (h2 : r ≠ 0) (h3 : 2 * (r + 5.5)^2 + 5 * (r + 5.5) = c) :
  c = 12 :=
sorry

end determine_c_l1418_141865


namespace find_original_price_l1418_141823

variable (P : ℝ)

def final_price (discounted_price : ℝ) (discount_rate : ℝ) (original_price : ℝ) : Prop :=
  discounted_price = (1 - discount_rate) * original_price

theorem find_original_price (h1 : final_price 120 0.4 P) : P = 200 := 
by
  sorry

end find_original_price_l1418_141823


namespace compare_abc_l1418_141804

theorem compare_abc 
  (a : ℝ := 1 / 11) 
  (b : ℝ := Real.sqrt (1 / 10)) 
  (c : ℝ := Real.log (11 / 10)) : 
  b > c ∧ c > a := 
by
  sorry

end compare_abc_l1418_141804


namespace curve_cartesian_equation_max_value_3x_plus_4y_l1418_141844

noncomputable def polar_to_cartesian (rho theta : ℝ) : ℝ × ℝ := (rho * Real.cos theta, rho * Real.sin theta)

theorem curve_cartesian_equation :
  (∀ (rho theta : ℝ), rho^2 = 36 / (4 * (Real.cos theta)^2 + 9 * (Real.sin theta)^2)) →
  ∀ x y : ℝ, (∃ theta : ℝ, x = 3 * Real.cos theta ∧ y = 2 * Real.sin theta) → (x^2) / 9 + (y^2) / 4 = 1 :=
sorry

theorem max_value_3x_plus_4y :
  (∀ (rho theta : ℝ), rho^2 = 36 / (4 * (Real.cos theta)^2 + 9 * (Real.sin theta)^2)) →
  ∃ x y : ℝ, (∃ theta : ℝ, x = 3 * Real.cos theta ∧ y = 2 * Real.sin theta) ∧ (∀ ϴ : ℝ, 3 * (3 * Real.cos ϴ) + 4 * (2 * Real.sin ϴ) ≤ Real.sqrt 145) :=
sorry

end curve_cartesian_equation_max_value_3x_plus_4y_l1418_141844


namespace train_crossing_time_l1418_141846

def train_length : ℕ := 100  -- length of the train in meters
def bridge_length : ℕ := 180  -- length of the bridge in meters
def train_speed_kmph : ℕ := 36  -- speed of the train in kmph

theorem train_crossing_time 
  (TL : ℕ := train_length) 
  (BL : ℕ := bridge_length) 
  (TSK : ℕ := train_speed_kmph) : 
  (TL + BL) / ((TSK * 1000) / 3600) = 28 := by
  sorry

end train_crossing_time_l1418_141846


namespace pedro_plums_l1418_141870

theorem pedro_plums :
  ∃ P Q : ℕ, P + Q = 32 ∧ 2 * P + Q = 52 ∧ P = 20 :=
by
  sorry

end pedro_plums_l1418_141870


namespace find_PA_PB_sum_2sqrt6_l1418_141863

noncomputable def polar_equation (ρ θ : ℝ) : Prop :=
  ρ - 2 * Real.cos θ - 6 * Real.sin θ + 1 / ρ = 0

noncomputable def parametric_line (t x y : ℝ) : Prop :=
  x = 3 + 1 / 2 * t ∧ y = 3 + Real.sqrt 3 / 2 * t

def point_P (x y : ℝ) : Prop :=
  x = 3 ∧ y = 3

theorem find_PA_PB_sum_2sqrt6 :
  (∃ ρ θ t₁ t₂, polar_equation ρ θ ∧ parametric_line t₁ 3 3 ∧ parametric_line t₂ 3 3 ∧
  point_P 3 3 ∧ |t₁| + |t₂| = 2 * Real.sqrt 6) := sorry

end find_PA_PB_sum_2sqrt6_l1418_141863


namespace clean_per_hour_l1418_141887

-- Definitions of the conditions
def total_pieces : ℕ := 80
def start_time : ℕ := 8
def end_time : ℕ := 12
def total_hours : ℕ := end_time - start_time

-- Proof statement
theorem clean_per_hour : total_pieces / total_hours = 20 := by
  -- Proof is omitted
  sorry

end clean_per_hour_l1418_141887


namespace complement_intersect_A_B_range_of_a_l1418_141877

-- Definitions for sets A and B
def setA : Set ℝ := {x | -2 < x ∧ x < 0}
def setB : Set ℝ := {x | ∃ y, y = Real.sqrt (x + 1)}

-- First statement to prove
theorem complement_intersect_A_B : (setAᶜ ∩ setB) = {x | x ≥ 0} :=
  sorry

-- Definition for set C
def setC (a : ℝ) : Set ℝ := {x | a < x ∧ x < 2 * a + 1}

-- Second statement to prove
theorem range_of_a (a : ℝ) : (setC a ⊆ setA) ↔ (a ≤ -1) ∨ (-1 ≤ a ∧ a ≤ -1 / 2) :=
  sorry

end complement_intersect_A_B_range_of_a_l1418_141877


namespace unique_a_for_system_solution_l1418_141838

-- Define the variables
variables (a b x y : ℝ)

-- Define the system of equations
def system_has_solution (a b : ℝ) : Prop :=
  ∃ x y : ℝ, 2^(b * x) + (a + 1) * b * y^2 = a^2 ∧ (a-1) * x^3 + y^3 = 1

-- Main theorem statement
theorem unique_a_for_system_solution :
  a = -1 ↔ ∀ b : ℝ, system_has_solution a b :=
sorry

end unique_a_for_system_solution_l1418_141838


namespace probability_first_ge_second_l1418_141850

-- Define the number of faces
def faces : ℕ := 10

-- Define the total number of outcomes excluding the duplicates
def total_outcomes : ℕ := faces * faces - faces

-- Calculate the number of favorable outcomes
def favorable_outcomes : ℕ := 10 + 9 + 8 + 7 + 6 + 5 + 4 + 3 + 2 + 1

-- Define the probability as a fraction
def probability : ℚ := favorable_outcomes / total_outcomes

-- The statement we want to prove
theorem probability_first_ge_second :
  probability = 11 / 18 :=
sorry

end probability_first_ge_second_l1418_141850


namespace chocolate_chips_per_cookie_l1418_141881

theorem chocolate_chips_per_cookie
  (num_batches : ℕ)
  (cookies_per_batch : ℕ)
  (num_people : ℕ)
  (chocolate_chips_per_person : ℕ) :
  (num_batches = 3) →
  (cookies_per_batch = 12) →
  (num_people = 4) →
  (chocolate_chips_per_person = 18) →
  (chocolate_chips_per_person / (num_batches * cookies_per_batch / num_people) = 2) :=
by
  sorry

end chocolate_chips_per_cookie_l1418_141881


namespace percentage_problem_l1418_141874

variable (x : ℝ)

theorem percentage_problem (h : 0.4 * x = 160) : 240 / x = 0.6 :=
by sorry

end percentage_problem_l1418_141874


namespace birgit_time_to_travel_8km_l1418_141875

theorem birgit_time_to_travel_8km
  (total_hours : ℝ)
  (total_distance : ℝ)
  (speed_difference : ℝ)
  (distance_to_travel : ℝ)
  (total_minutes := total_hours * 60)
  (average_speed := total_minutes / total_distance)
  (birgit_speed := average_speed - speed_difference) :
  total_hours = 3.5 →
  total_distance = 21 →
  speed_difference = 4 →
  distance_to_travel = 8 →
  (birgit_speed * distance_to_travel) = 48 :=
by
  sorry

end birgit_time_to_travel_8km_l1418_141875


namespace expression_always_positive_l1418_141810

theorem expression_always_positive (x : ℝ) : x^2 + |x| + 1 > 0 :=
by 
  sorry

end expression_always_positive_l1418_141810


namespace cone_to_cylinder_water_height_l1418_141866

theorem cone_to_cylinder_water_height :
  let r_cone := 15 -- radius of the cone
  let h_cone := 24 -- height of the cone
  let r_cylinder := 18 -- radius of the cylinder
  let V_cone := (1 / 3: ℝ) * Real.pi * r_cone^2 * h_cone -- volume of the cone
  let h_cylinder := V_cone / (Real.pi * r_cylinder^2) -- height of the water in the cylinder
  h_cylinder = 8.33 := by
  sorry

end cone_to_cylinder_water_height_l1418_141866


namespace basketball_not_table_tennis_l1418_141840

-- Definitions and conditions
def total_students := 30
def like_basketball := 15
def like_table_tennis := 10
def do_not_like_either := 8
def like_both (x : ℕ) := x

-- Theorem statement
theorem basketball_not_table_tennis (x : ℕ) (H : (like_basketball - x) + (like_table_tennis - x) + x + do_not_like_either = total_students) : (like_basketball - x) = 12 :=
by
  sorry

end basketball_not_table_tennis_l1418_141840


namespace ordered_pairs_bound_l1418_141859

variable (m n : ℕ) (a b : ℕ → ℝ)

theorem ordered_pairs_bound
  (h_m : m ≥ n)
  (h_n : n ≥ 2022)
  : (∃ (pairs : Finset (ℕ × ℕ)), 
      (∀ i j, (i, j) ∈ pairs → 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n ∧ |a i + b j - (i * j)| ≤ m) ∧
      pairs.card ≤ 3 * n * Real.sqrt (m * Real.log (n))) := 
  sorry

end ordered_pairs_bound_l1418_141859


namespace area_of_triangle_l1418_141808

-- Define the function to calculate the area of a right isosceles triangle given the side lengths of squares
theorem area_of_triangle (a b c : ℝ) (h1 : a = 10) (h2 : b = 8) (h3 : c = 10) (right_isosceles : true) :
  (1 / 2) * a * c = 50 :=
by
  -- We state the theorem but leave the proof as sorry.
  sorry

end area_of_triangle_l1418_141808


namespace quadratic_other_x_intercept_l1418_141806

theorem quadratic_other_x_intercept (a b c : ℝ) (h_vertex : ∀ x, x = 5 → a * x^2 + b * x + c = -3)
  (h_intercept : a * 1^2 + b * 1 + c = 0) : 
  ∃ x0 : ℝ, x0 = 9 ∧ a * x0^2 + b * x0 + c = 0 :=
by
  sorry

end quadratic_other_x_intercept_l1418_141806


namespace julie_can_print_100_newspapers_l1418_141884

def num_boxes : ℕ := 2
def packages_per_box : ℕ := 5
def sheets_per_package : ℕ := 250
def sheets_per_newspaper : ℕ := 25

theorem julie_can_print_100_newspapers :
  (num_boxes * packages_per_box * sheets_per_package) / sheets_per_newspaper = 100 := by
  sorry

end julie_can_print_100_newspapers_l1418_141884


namespace exponent_property_l1418_141822

theorem exponent_property (a b : ℕ) : (a * b^2)^3 = a^3 * b^6 :=
by sorry

end exponent_property_l1418_141822


namespace solve_equation_l1418_141807

theorem solve_equation (x : ℝ) (h : (x^2 - x + 2) / (x - 1) = x + 3) (h1 : x ≠ 1) : 
  x = 5 / 3 :=
sorry

end solve_equation_l1418_141807


namespace simplify_and_compute_l1418_141853

theorem simplify_and_compute (x : ℝ) (h : x = 4) : (x^8 - 32*x^4 + 256) / (x^4 - 16) = 240 := by
  sorry

end simplify_and_compute_l1418_141853


namespace rectangle_area_l1418_141852

variable (w l A P : ℝ)
variable (h1 : l = w + 6)
variable (h2 : A = w * l)
variable (h3 : P = 2 * (w + l))
variable (h4 : A = 2 * P)
variable (h5 : w = 3)

theorem rectangle_area
  (w l A P : ℝ)
  (h1 : l = w + 6)
  (h2 : A = w * l)
  (h3 : P = 2 * (w + l))
  (h4 : A = 2 * P)
  (h5 : w = 3) :
  A = 27 := 
sorry

end rectangle_area_l1418_141852


namespace cory_prime_sum_l1418_141839

def primes_between_30_and_60 : List ℕ := [31, 37, 41, 43, 47, 53, 59]

theorem cory_prime_sum :
  let smallest := 31
  let largest := 59
  let median := 43
  smallest ∈ primes_between_30_and_60 ∧
  largest ∈ primes_between_30_and_60 ∧
  median ∈ primes_between_30_and_60 ∧
  primes_between_30_and_60 = [31, 37, 41, 43, 47, 53, 59] → 
  smallest + largest + median = 133 := 
by
  intros; sorry

end cory_prime_sum_l1418_141839


namespace proof_problem_l1418_141815

-- Definitions for the given conditions in the problem
def equations (a x y : ℝ) : Prop :=
(x + 5 * y = 4 - a) ∧ (x - y = 3 * a)

-- The conclusions from the problem
def conclusion1 (a x y : ℝ) : Prop :=
a = 1 → x + y = 4 - a

def conclusion2 (a x y : ℝ) : Prop :=
a = -2 → x = -y

def conclusion3 (a x y : ℝ) : Prop :=
2 * x + 7 * y = 6

def conclusion4 (a x y : ℝ) : Prop :=
x ≤ 1 → y > 4 / 7

-- The main theorem to be proven
theorem proof_problem (a x y : ℝ) :
  equations a x y →
  (¬ conclusion1 a x y ∨ ¬ conclusion2 a x y ∨ ¬ conclusion3 a x y ∨ ¬ conclusion4 a x y) →
  (∃ n : ℕ, n = 2 ∧ ((conclusion1 a x y ∨ conclusion2 a x y ∨ conclusion3 a x y ∨ conclusion4 a x y) → false)) :=
by {
  sorry
}

end proof_problem_l1418_141815


namespace find_n_after_folding_l1418_141813

theorem find_n_after_folding (n : ℕ) (h : 2 ^ n = 128) : n = 7 := by
  sorry

end find_n_after_folding_l1418_141813


namespace g_is_odd_function_l1418_141816

noncomputable def g (x : ℝ) := 5 / (3 * x^5 - 7 * x)

theorem g_is_odd_function : ∀ x : ℝ, g (-x) = -g x :=
by
  intro x
  unfold g
  sorry

end g_is_odd_function_l1418_141816


namespace inequality_positives_l1418_141886

theorem inequality_positives (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (a * (a + 1)) / (b + 1) + (b * (b + 1)) / (a + 1) ≥ a + b :=
sorry

end inequality_positives_l1418_141886


namespace germination_rate_sunflower_l1418_141883

variable (s_d s_s f_d f_s p : ℕ) (g_d g_f : ℚ)

-- Define the conditions
def conditions :=
  s_d = 25 ∧ s_s = 25 ∧ g_d = 0.60 ∧ g_f = 0.80 ∧ p = 28 ∧ f_d = 12 ∧ f_s = 16

-- Define the statement to be proved
theorem germination_rate_sunflower (h : conditions s_d s_s f_d f_s p g_d g_f) : 
  (f_s / (g_f * (s_s : ℚ))) > 0.0 ∧ (f_s / (g_f * (s_s : ℚ)) * 100) = 80 := 
by
  sorry

end germination_rate_sunflower_l1418_141883


namespace vanya_speed_increased_by_4_l1418_141880

variable (v : ℝ)
variable (h1 : (v + 2) / v = 2.5)

theorem vanya_speed_increased_by_4 (h1 : (v + 2) / v = 2.5) : (v + 4) / v = 4 := 
sorry

end vanya_speed_increased_by_4_l1418_141880
