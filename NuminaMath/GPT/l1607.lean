import Mathlib

namespace train_speed_in_km_per_hr_l1607_160786

def train_length : ℝ := 116.67 -- length of the train in meters
def crossing_time : ℝ := 7 -- time to cross the pole in seconds

theorem train_speed_in_km_per_hr : 
  (train_length / 1000) / (crossing_time / 3600) = 60 := by
  sorry

end train_speed_in_km_per_hr_l1607_160786


namespace closed_broken_line_impossible_l1607_160792

theorem closed_broken_line_impossible (n : ℕ) (h : n = 1989) : ¬ (∃ a b : ℕ, 2 * (a + b) = n) :=
by {
  sorry
}

end closed_broken_line_impossible_l1607_160792


namespace hancho_height_l1607_160722

theorem hancho_height (Hansol_height : ℝ) (h1 : Hansol_height = 134.5) (ratio : ℝ) (h2 : ratio = 1.06) :
  Hansol_height * ratio = 142.57 := by
  sorry

end hancho_height_l1607_160722


namespace shea_buys_corn_l1607_160776

noncomputable def num_pounds_corn (c b : ℚ) : ℚ :=
  if b + c = 24 ∧ 45 * b + 99 * c = 1809 then c else -1

theorem shea_buys_corn (c b : ℚ) : b + c = 24 ∧ 45 * b + 99 * c = 1809 → c = 13.5 :=
by
  intros h
  sorry

end shea_buys_corn_l1607_160776


namespace expression_value_l1607_160751

theorem expression_value {a b : ℝ} (h : a * b = -3) : a * Real.sqrt (-b / a) + b * Real.sqrt (-a / b) = 0 :=
by
  sorry

end expression_value_l1607_160751


namespace like_terms_sum_l1607_160717

theorem like_terms_sum (m n : ℕ) (h1 : 2 * m = 4) (h2 : 3 - n = 1) : m + n = 4 :=
by
  sorry

end like_terms_sum_l1607_160717


namespace ezekiel_new_shoes_l1607_160727

-- condition Ezekiel bought 3 pairs of shoes
def pairs_of_shoes : ℕ := 3

-- condition Each pair consists of 2 shoes
def shoes_per_pair : ℕ := 2

-- proving the number of new shoes Ezekiel has
theorem ezekiel_new_shoes (pairs_of_shoes shoes_per_pair : ℕ) : pairs_of_shoes * shoes_per_pair = 6 :=
by
  sorry

end ezekiel_new_shoes_l1607_160727


namespace area_of_30_60_90_triangle_hypotenuse_6sqrt2_l1607_160777

theorem area_of_30_60_90_triangle_hypotenuse_6sqrt2 :
  ∀ (a b c : ℝ),
  a = 3 * Real.sqrt 2 →
  b = 3 * Real.sqrt 6 →
  c = 6 * Real.sqrt 2 →
  c = 2 * a →
  (1 / 2) * a * b = 18 * Real.sqrt 3 :=
by
  intro a b c ha hb hc h2a
  sorry

end area_of_30_60_90_triangle_hypotenuse_6sqrt2_l1607_160777


namespace min_squared_sum_l1607_160749

theorem min_squared_sum {x y z : ℝ} (h : 2 * x + y + 2 * z = 6) : x^2 + y^2 + z^2 ≥ 4 :=
sorry

end min_squared_sum_l1607_160749


namespace num_lists_correct_l1607_160788

def num_balls : ℕ := 18
def num_draws : ℕ := 4

theorem num_lists_correct : (num_balls ^ num_draws) = 104976 :=
by
  sorry

end num_lists_correct_l1607_160788


namespace number_sum_20_eq_30_l1607_160737

theorem number_sum_20_eq_30 : ∃ x : ℤ, 20 + x = 30 → x = 10 :=
by {
  sorry
}

end number_sum_20_eq_30_l1607_160737


namespace f_neg_two_l1607_160761

def f (a b : ℝ) (x : ℝ) :=
  -a * x^5 - x^3 + b * x - 7

theorem f_neg_two (a b : ℝ) (h : f a b 2 = -9) : f a b (-2) = -5 :=
by sorry

end f_neg_two_l1607_160761


namespace table_capacity_l1607_160799

def invited_people : Nat := 18
def no_show_people : Nat := 12
def number_of_tables : Nat := 2
def attendees := invited_people - no_show_people
def people_per_table : Nat := attendees / number_of_tables

theorem table_capacity : people_per_table = 3 :=
by
  sorry

end table_capacity_l1607_160799


namespace simplify_expression_l1607_160755

variable (a : ℝ)

theorem simplify_expression : 
  (a^2 / (a^(1/2) * a^(2/3))) = a^(5/6) :=
by
  sorry

end simplify_expression_l1607_160755


namespace sum_d_e_f_equals_23_l1607_160723

theorem sum_d_e_f_equals_23
  (d e f : ℤ)
  (h1 : ∀ x : ℝ, x^2 + 9 * x + 20 = (x + d) * (x + e))
  (h2 : ∀ x : ℝ, x^2 + 11 * x - 60 = (x + e) * (x - f)) :
  d + e + f = 23 :=
by
  sorry

end sum_d_e_f_equals_23_l1607_160723


namespace price_diff_is_correct_l1607_160732

-- Define initial conditions
def initial_price : ℝ := 30
def flat_discount : ℝ := 5
def percent_discount : ℝ := 0.25
def sales_tax : ℝ := 0.10

def price_after_flat_discount (price : ℝ) : ℝ :=
  price - flat_discount

def price_after_percent_discount (price : ℝ) : ℝ :=
  price * (1 - percent_discount)

def price_after_tax (price : ℝ) : ℝ :=
  price * (1 + sales_tax)

def final_price_method1 : ℝ :=
  price_after_tax (price_after_percent_discount (price_after_flat_discount initial_price))

def final_price_method2 : ℝ :=
  price_after_tax (price_after_flat_discount (price_after_percent_discount initial_price))

def difference_in_cents : ℝ :=
  (final_price_method1 - final_price_method2) * 100

-- Lean statement to prove the final difference in cents
theorem price_diff_is_correct : difference_in_cents = 137.5 :=
  by sorry

end price_diff_is_correct_l1607_160732


namespace arithmetic_sequence_sum_l1607_160712

theorem arithmetic_sequence_sum (S : ℕ → ℝ) (h_arith_seq: ∀ n: ℕ, S n = S 0 + n * (S 1 - S 0)) 
  (h5 : S 5 = 10) (h10 : S 10 = 30) : S 15 = 60 :=
by
  sorry

end arithmetic_sequence_sum_l1607_160712


namespace triangle_area_l1607_160762

-- Define the line equation as a condition.
def line_equation (x : ℝ) : ℝ :=
  4 * x + 8

-- Define the y-intercept (condition 1).
def y_intercept := line_equation 0

-- Define the x-intercept (condition 2).
def x_intercept := (-8) / 4

-- Define the area of the triangle given the intercepts and prove it equals 8 (question and correct answer).
theorem triangle_area :
  (1 / 2) * abs x_intercept * y_intercept = 8 :=
by
  sorry

end triangle_area_l1607_160762


namespace perimeter_after_adding_tiles_l1607_160791

-- Initial perimeter given
def initial_perimeter : ℕ := 20

-- Number of initial tiles
def initial_tiles : ℕ := 10

-- Number of additional tiles to be added
def additional_tiles : ℕ := 2

-- New tile side must be adjacent to an existing tile
def adjacent_tile_side : Prop := true

-- Condition about the tiles being 1x1 squares
def sq_tile (n : ℕ) : Prop := n = 1

-- The perimeter should be calculated after adding the tiles
def new_perimeter_after_addition : ℕ := 19

theorem perimeter_after_adding_tiles :
  ∃ (new_perimeter : ℕ), 
    new_perimeter = 19 ∧ 
    initial_perimeter = 20 ∧ 
    initial_tiles = 10 ∧ 
    additional_tiles = 2 ∧ 
    adjacent_tile_side ∧ 
    sq_tile 1 :=
sorry

end perimeter_after_adding_tiles_l1607_160791


namespace measure_of_acute_angle_l1607_160778

theorem measure_of_acute_angle (x : ℝ) (h_complement : 90 - x = (1/2) * (180 - x) + 20) (h_acute : 0 < x ∧ x < 90) : x = 40 :=
  sorry

end measure_of_acute_angle_l1607_160778


namespace original_number_is_repeating_decimal_l1607_160721

theorem original_number_is_repeating_decimal :
  ∃ N : ℚ, (N * 10 ^ 28) % 10^30 = 15 ∧ N * 5 = 0.7894736842105263 ∧ 
  (N = 3 / 19) :=
sorry

end original_number_is_repeating_decimal_l1607_160721


namespace smallest_a_mod_remainders_l1607_160736

theorem smallest_a_mod_remainders:
  (∀ d ∈ [2, 3, 4, 5, 6, 7, 8, 9], 2521 % d = 1) ∧
  (∀ n : ℕ, ∃ a : ℕ, a = 2520 * n + 1 ∧ (∀ d ∈ [2, 3, 4, 5, 6, 7, 8, 9], a % d = 1)) :=
by
  sorry

end smallest_a_mod_remainders_l1607_160736


namespace initial_wage_of_illiterate_l1607_160766

-- Definitions from the conditions
def illiterate_employees : ℕ := 20
def literate_employees : ℕ := 10
def total_employees := illiterate_employees + literate_employees

-- Given that the daily average wages of illiterate employees decreased to Rs. 10
def daily_wages_after_decrease : ℝ := 10
-- The total decrease in the average salary of all employees by Rs. 10 per day
def decrease_in_avg_wage : ℝ := 10

-- To be proved: the initial daily average wage of the illiterate employees was Rs. 25.
theorem initial_wage_of_illiterate (I : ℝ) :
  (illiterate_employees * I - illiterate_employees * daily_wages_after_decrease = total_employees * decrease_in_avg_wage) → 
  I = 25 := 
by
  sorry

end initial_wage_of_illiterate_l1607_160766


namespace major_axis_length_l1607_160716

-- Define the problem setup
structure Cylinder :=
  (base_radius : ℝ)
  (height : ℝ)

structure Sphere :=
  (radius : ℝ)

-- Define the conditions
def cylinder : Cylinder :=
  { base_radius := 6, height := 0 }  -- height isn't significant for this problem

def sphere1 : Sphere :=
  { radius := 6 }

def sphere2 : Sphere :=
  { radius := 6 }

def distance_between_centers : ℝ :=
  13

-- Statement of the problem in Lean 4
theorem major_axis_length : 
  cylinder.base_radius = 6 →
  sphere1.radius = 6 →
  sphere2.radius = 6 →
  distance_between_centers = 13 →
  ∃ major_axis_length : ℝ, major_axis_length = 13 :=
by
  intros h1 h2 h3 h4
  existsi 13
  sorry

end major_axis_length_l1607_160716


namespace total_dogs_on_farm_l1607_160714

-- Definitions based on conditions from part a)
def num_dog_houses : ℕ := 5
def num_dogs_per_house : ℕ := 4

-- Statement to prove
theorem total_dogs_on_farm : num_dog_houses * num_dogs_per_house = 20 :=
by
  sorry

end total_dogs_on_farm_l1607_160714


namespace simplify_sqrt_product_l1607_160711

theorem simplify_sqrt_product : (Real.sqrt (3 * 5) * Real.sqrt (3 ^ 5 * 5 ^ 5) = 3375) :=
  sorry

end simplify_sqrt_product_l1607_160711


namespace quadratic_roots_l1607_160726

theorem quadratic_roots (x : ℝ) : (x^2 + 4*x + 3 = 0) ↔ (x = -3 ∨ x = -1) := 
sorry

end quadratic_roots_l1607_160726


namespace ac_length_l1607_160758

theorem ac_length (AB : ℝ) (H1 : AB = 100)
    (BC AC : ℝ)
    (H2 : AC = (1 + Real.sqrt 5)/2 * BC)
    (H3 : AC + BC = AB) : AC = 75 - 25 * Real.sqrt 5 :=
by
  sorry

end ac_length_l1607_160758


namespace triangles_with_positive_area_l1607_160701

theorem triangles_with_positive_area (x y : ℕ) (h₁ : 1 ≤ x ∧ x ≤ 5) (h₂ : 1 ≤ y ∧ y ≤ 3) : 
    ∃ (n : ℕ), n = 420 := 
sorry

end triangles_with_positive_area_l1607_160701


namespace rectangle_diagonal_length_l1607_160730

theorem rectangle_diagonal_length
    (PQ QR : ℝ) (RT RU ST : ℝ) (Area_RST : ℝ)
    (hPQ : PQ = 8) (hQR : QR = 10)
    (hRT_RU : RT = RU)
    (hArea_RST: Area_RST = (1/5) * (PQ * QR)) :
    ST = 8 :=
by
  sorry

end rectangle_diagonal_length_l1607_160730


namespace employee_payment_sum_l1607_160794

theorem employee_payment_sum :
  ∀ (A B : ℕ), 
  (A = 3 * B / 2) → 
  (B = 180) → 
  (A + B = 450) :=
by
  intros A B hA hB
  sorry

end employee_payment_sum_l1607_160794


namespace not_difference_of_squares_2021_l1607_160774

theorem not_difference_of_squares_2021:
  ¬ ∃ (a b : ℕ), (a > b) ∧ (a^2 - b^2 = 2021) :=
sorry

end not_difference_of_squares_2021_l1607_160774


namespace cases_in_1995_l1607_160783

theorem cases_in_1995 (initial_cases cases_2010 : ℕ) (years_total : ℕ) (years_passed : ℕ) (cases_1995 : ℕ)
  (h1 : initial_cases = 700000) 
  (h2 : cases_2010 = 1000) 
  (h3 : years_total = 40) 
  (h4 : years_passed = 25)
  (h5 : cases_1995 = initial_cases - (years_passed * (initial_cases - cases_2010) / years_total)) : 
  cases_1995 = 263125 := 
sorry

end cases_in_1995_l1607_160783


namespace expressions_equal_when_a_plus_b_plus_c_eq_1_l1607_160738

theorem expressions_equal_when_a_plus_b_plus_c_eq_1
  (a b c : ℝ) (h : a + b + c = 1) :
  a + b * c = (a + b) * (a + c) :=
sorry

end expressions_equal_when_a_plus_b_plus_c_eq_1_l1607_160738


namespace min_width_of_garden_l1607_160750

theorem min_width_of_garden (w : ℝ) (h : w*(w + 10) ≥ 150) : w ≥ 10 :=
by
  sorry

end min_width_of_garden_l1607_160750


namespace inequality_proof_l1607_160748

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (2 * x^2 / (y + z) + 2 * y^2 / (z + x) + 2 * z^2 / (x + y) ≥ x + y + z) :=
by
  sorry

end inequality_proof_l1607_160748


namespace general_formula_a_general_formula_c_l1607_160779

-- Definition of the sequence {a_n}
def S (n : ℕ) : ℕ := n^2 + 2 * n
def a (n : ℕ) : ℕ := S n - S (n - 1)

theorem general_formula_a (n : ℕ) (hn : n > 0) : a n = 2 * n + 1 := sorry

-- Definitions for the second problem
def f (x : ℝ) : ℝ := x^2 + 2 * x
def f' (x : ℝ) : ℝ := 2 * x + 2
def k (n : ℕ) : ℝ := 2 * n + 2

def Q (k : ℝ) : Prop := ∃ (n : ℕ), k = 2 * n + 2
def R (k : ℝ) : Prop := ∃ (n : ℕ), k = 4 * n + 2

def c (n : ℕ) : ℕ := 12 * n - 6

theorem general_formula_c (n : ℕ) (hn1 : 0 < c 10)
    (hn2 : c 10 < 115) : c n = 12 * n - 6 := sorry

end general_formula_a_general_formula_c_l1607_160779


namespace minimum_a_value_l1607_160707

theorem minimum_a_value (a : ℝ) : 
  (∀ (x y : ℝ), 0 < x → 0 < y → x^2 + 2 * x * y ≤ a * (x^2 + y^2)) ↔ a ≥ (Real.sqrt 5 + 1) / 2 := 
sorry

end minimum_a_value_l1607_160707


namespace remove_two_fractions_sum_is_one_l1607_160798

theorem remove_two_fractions_sum_is_one :
  let fractions := [1/2, 1/4, 1/6, 1/8, 1/10, 1/12]
  let total_sum := (fractions.sum : ℚ)
  let remaining_sum := total_sum - (1/8 + 1/10)
  remaining_sum = 1 := by
    sorry

end remove_two_fractions_sum_is_one_l1607_160798


namespace speed_of_man_l1607_160745

noncomputable def train_length : ℝ := 150
noncomputable def time_to_pass : ℝ := 6
noncomputable def train_speed_kmh : ℝ := 83.99280057595394

/-- The speed of the man in km/h -/
theorem speed_of_man (train_length time_to_pass train_speed_kmh : ℝ) (h_train_length : train_length = 150) (h_time_to_pass : time_to_pass = 6) (h_train_speed_kmh : train_speed_kmh = 83.99280057595394) : 
  (train_length / time_to_pass * 3600 / 1000 - train_speed_kmh) * 3600 / 1000 = 6.0072 :=
by
  sorry

end speed_of_man_l1607_160745


namespace daisies_sold_on_fourth_day_l1607_160704

-- Number of daisies sold on the first day
def first_day_daisies : ℕ := 45

-- Number of daisies sold on the second day
def second_day_daisies : ℕ := first_day_daisies + 20

-- Number of daisies sold on the third day
def third_day_daisies : ℕ := 2 * second_day_daisies - 10

-- Total number of daisies sold in the first three days
def total_first_three_days_daisies : ℕ := first_day_daisies + second_day_daisies + third_day_daisies

-- Total number of daisies sold in four days
def total_four_days_daisies : ℕ := 350

-- Number of daisies sold on the fourth day
def fourth_day_daisies : ℕ := total_four_days_daisies - total_first_three_days_daisies

-- Theorem that states the number of daisies sold on the fourth day is 120
theorem daisies_sold_on_fourth_day : fourth_day_daisies = 120 :=
by sorry

end daisies_sold_on_fourth_day_l1607_160704


namespace percentage_of_cobalt_is_15_l1607_160797

-- Define the given percentages of lead and copper
def percent_lead : ℝ := 25
def percent_copper : ℝ := 60

-- Define the weights of lead and copper used in the mixture
def weight_lead : ℝ := 5
def weight_copper : ℝ := 12

-- Define the total weight of the mixture
def total_weight : ℝ := weight_lead + weight_copper

-- Prove that the percentage of cobalt is 15%
theorem percentage_of_cobalt_is_15 :
  (100 - (percent_lead + percent_copper) = 15) :=
by
  sorry

end percentage_of_cobalt_is_15_l1607_160797


namespace main_theorem_l1607_160713

-- Define the interval (3π/4, π)
def theta_range (θ : ℝ) : Prop :=
  (3 * Real.pi / 4) < θ ∧ θ < Real.pi

-- Define the condition
def inequality_condition (θ x : ℝ) : Prop :=
  x^2 * Real.sin θ - x * (1 - x) + (1 - x)^2 * Real.cos θ + 2 * x * (1 - x) * Real.sqrt (Real.cos θ * Real.sin θ) > 0

-- The main theorem
theorem main_theorem (θ x : ℝ) (hθ : theta_range θ) (hx : 0 ≤ x ∧ x ≤ 1) : inequality_condition θ x :=
by
  sorry

end main_theorem_l1607_160713


namespace kanul_cash_percentage_l1607_160720

theorem kanul_cash_percentage (raw_materials : ℕ) (machinery : ℕ) (total_amount : ℕ) (cash_percentage : ℕ)
  (H1 : raw_materials = 80000)
  (H2 : machinery = 30000)
  (H3 : total_amount = 137500)
  (H4 : cash_percentage = 20) :
  ((total_amount - (raw_materials + machinery)) * 100 / total_amount) = cash_percentage := by
    sorry

end kanul_cash_percentage_l1607_160720


namespace period_of_f_cos_theta_l1607_160709

open Real

noncomputable def alpha (x : ℝ) : ℝ × ℝ :=
  (sqrt 3 * sin (2 * x), cos x + sin x)

noncomputable def beta (x : ℝ) : ℝ × ℝ :=
  (1, cos x - sin x)

noncomputable def f (x : ℝ) : ℝ :=
  let (α1, α2) := alpha x
  let (β1, β2) := beta x
  α1 * β1 + α2 * β2

theorem period_of_f :
  (∀ x : ℝ, f (x + π) = f x) ∧ (∀ T : ℝ, (T > 0 ∧ ∀ x : ℝ, f (x + T) = f x) → T = π) :=
sorry

theorem cos_theta :
  ∀ θ : ℝ, 0 < θ ∧ θ < π / 2 ∧ f θ = 1 → cos (θ - π / 6) = sqrt 3 / 2 :=
sorry

end period_of_f_cos_theta_l1607_160709


namespace fraction_simplification_l1607_160734

theorem fraction_simplification (a b : ℚ) (h : b / a = 3 / 5) : (a - b) / a = 2 / 5 :=
by
  sorry

end fraction_simplification_l1607_160734


namespace solve_for_a_l1607_160715

def quadratic_has_roots (a x1 x2 : ℝ) : Prop :=
  x1 + x2 = a ∧ x1 * x2 = -6 * a^2

theorem solve_for_a (a x1 x2 : ℝ) (h1 : a > 0) (h2 : quadratic_has_roots a x1 x2) (h3 : x2 - x1 = 10) : a = 2 :=
by
  sorry

end solve_for_a_l1607_160715


namespace winner_is_Junsu_l1607_160725

def Younghee_water_intake : ℝ := 1.4
def Jimin_water_intake : ℝ := 1.8
def Junsu_water_intake : ℝ := 2.1

theorem winner_is_Junsu : 
  Junsu_water_intake > Younghee_water_intake ∧ Junsu_water_intake > Jimin_water_intake :=
by sorry

end winner_is_Junsu_l1607_160725


namespace cos_pi_over_6_minus_a_eq_5_over_12_l1607_160763

theorem cos_pi_over_6_minus_a_eq_5_over_12 (a : ℝ) (h : Real.sin (Real.pi / 3 + a) = 5 / 12) :
  Real.cos (Real.pi / 6 - a) = 5 / 12 :=
by
  sorry

end cos_pi_over_6_minus_a_eq_5_over_12_l1607_160763


namespace abs_quadratic_eq_linear_iff_l1607_160752

theorem abs_quadratic_eq_linear_iff (x : ℝ) : 
  (|x^2 - 5*x + 6| = x + 2) ↔ (x = 3 + Real.sqrt 5 ∨ x = 3 - Real.sqrt 5) :=
by
  sorry

end abs_quadratic_eq_linear_iff_l1607_160752


namespace percentage_within_one_standard_deviation_l1607_160772

-- Define the constants
def m : ℝ := sorry     -- mean
def g : ℝ := sorry     -- standard deviation
def P : ℝ → ℝ := sorry -- cumulative distribution function

-- The condition that 84% of the distribution is less than m + g
def condition1 : Prop := P (m + g) = 0.84

-- The condition that the distribution is symmetric about the mean
def symmetric_distribution (P : ℝ → ℝ) (m : ℝ) : Prop :=
  ∀ x, P (m + (m - x)) = 1 - P x

-- The problem asks to prove that 68% of the distribution lies within one standard deviation of the mean
theorem percentage_within_one_standard_deviation 
  (h₁ : condition1)
  (h₂ : symmetric_distribution P m) : 
  P (m + g) - P (m - g) = 0.68 :=
sorry

end percentage_within_one_standard_deviation_l1607_160772


namespace xyz_value_l1607_160740

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 36) 
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 10) : 
  x * y * z = 26 / 3 := 
by
  sorry

end xyz_value_l1607_160740


namespace largest_sum_of_two_3_digit_numbers_l1607_160780

theorem largest_sum_of_two_3_digit_numbers : 
  ∃ (a b c d e f : ℕ), 
    (1 ≤ a ∧ a ≤ 6) ∧ (1 ≤ b ∧ b ≤ 6) ∧ (1 ≤ c ∧ c ≤ 6) ∧
    (1 ≤ d ∧ d ≤ 6) ∧ (1 ≤ e ∧ e ≤ 6) ∧ (1 ≤ f ∧ f ≤ 6) ∧
    (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ 
     b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ 
     c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ 
     d ≠ e ∧ d ≠ f ∧ 
     e ≠ f) ∧ 
    (100 * (a + d) + 10 * (b + e) + (c + f) = 1173) :=
by
  sorry

end largest_sum_of_two_3_digit_numbers_l1607_160780


namespace animal_population_l1607_160757

theorem animal_population
  (number_of_lions : ℕ)
  (number_of_leopards : ℕ)
  (number_of_elephants : ℕ)
  (h1 : number_of_lions = 200)
  (h2 : number_of_lions = 2 * number_of_leopards)
  (h3 : number_of_elephants = (number_of_lions + number_of_leopards) / 2) :
  number_of_lions + number_of_leopards + number_of_elephants = 450 :=
sorry

end animal_population_l1607_160757


namespace expand_polynomial_l1607_160784

theorem expand_polynomial (t : ℝ) :
  (3 * t^3 - 2 * t^2 + t - 4) * (2 * t^2 - t + 3) = 6 * t^5 - 7 * t^4 + 5 * t^3 - 15 * t^2 + 7 * t - 12 :=
by sorry

end expand_polynomial_l1607_160784


namespace best_discount_sequence_l1607_160742

/-- 
The initial price of the book is 30.
Stay focused on two sequences of discounts.
Sequence 1: $5 off, then 10% off, then $2 off if applicable.
Sequence 2: 10% off, then $5 off, then $2 off if applicable.
Compare the final prices obtained from applying these sequences.
-/
noncomputable def initial_price : ℝ := 30
noncomputable def five_off (price : ℝ) : ℝ := price - 5
noncomputable def ten_percent_off (price : ℝ) : ℝ := 0.9 * price
noncomputable def additional_two_off_if_applicable (price : ℝ) : ℝ := 
  if price > 20 then price - 2 else price

noncomputable def sequence1_final_price : ℝ := 
  additional_two_off_if_applicable (ten_percent_off (five_off initial_price))

noncomputable def sequence2_final_price : ℝ := 
  additional_two_off_if_applicable (five_off (ten_percent_off initial_price))

theorem best_discount_sequence : 
  sequence2_final_price = 20 ∧ 
  sequence2_final_price < sequence1_final_price ∧ 
  sequence1_final_price - sequence2_final_price = 0.5 :=
by
  sorry

end best_discount_sequence_l1607_160742


namespace correct_system_of_equations_l1607_160764

variable (x y : ℕ) -- We assume non-negative numbers for counts of chickens and rabbits

theorem correct_system_of_equations :
  (x + y = 35) ∧ (2 * x + 4 * y = 94) ↔
  (∃ (a b : ℕ), a = x ∧ b = y) :=
by
  sorry

end correct_system_of_equations_l1607_160764


namespace avg_goals_per_game_l1607_160769

def carter_goals_per_game := 4
def shelby_goals_per_game := carter_goals_per_game / 2
def judah_goals_per_game := (2 * shelby_goals_per_game) - 3
def average_total_goals_team := carter_goals_per_game + shelby_goals_per_game + judah_goals_per_game

theorem avg_goals_per_game : average_total_goals_team = 7 :=
by
  -- Proof would go here
  sorry

end avg_goals_per_game_l1607_160769


namespace remainder_with_conditions_l1607_160733

theorem remainder_with_conditions (a b c d : ℕ) (h1 : a % 53 = 33) (h2 : b % 53 = 15) (h3 : c % 53 = 27) (h4 : d % 53 = 8) :
  ((a + b + c + d + 10) % 53) = 40 :=
by
  sorry

end remainder_with_conditions_l1607_160733


namespace arithmetic_progression_sum_l1607_160793

theorem arithmetic_progression_sum (a d : ℝ)
  (h1 : 10 * (2 * a + 19 * d) = 200)
  (h2 : 25 * (2 * a + 49 * d) = 0) :
  35 * (2 * a + 69 * d) = -466.67 :=
by
  sorry

end arithmetic_progression_sum_l1607_160793


namespace lcm_12_15_18_l1607_160787

theorem lcm_12_15_18 : Nat.lcm (Nat.lcm 12 15) 18 = 180 := by 
  sorry

end lcm_12_15_18_l1607_160787


namespace solve_inequality_l1607_160753

theorem solve_inequality :
  ∀ x : ℝ, (x - 2) * (x + 2) < 5 ↔ -3 < x ∧ x < 3 :=
by
  sorry

end solve_inequality_l1607_160753


namespace charles_richard_difference_in_dimes_l1607_160743

variable (q : ℕ)

-- Charles' quarters
def charles_quarters : ℕ := 5 * q + 1

-- Richard's quarters
def richard_quarters : ℕ := q + 5

-- Difference in quarters
def diff_quarters : ℕ := charles_quarters q - richard_quarters q

-- Difference in dimes
def diff_dimes : ℕ := (diff_quarters q) * 5 / 2

theorem charles_richard_difference_in_dimes : diff_dimes q = 10 * (q - 1) := by
  sorry

end charles_richard_difference_in_dimes_l1607_160743


namespace find_multiplier_l1607_160759

theorem find_multiplier (n x : ℤ) (h1: n = 12) (h2: 4 * n - 3 = (n - 7) * x) : x = 9 :=
by {
  sorry
}

end find_multiplier_l1607_160759


namespace symmetric_function_cannot_be_even_l1607_160760

noncomputable def f : ℝ → ℝ := sorry

theorem symmetric_function_cannot_be_even :
  (∀ x, f (f x) = x^2) ∧ (∀ x ≥ 0, f (x^2) = x) → ¬ (∀ x, f x = f (-x)) :=
by 
  intros
  sorry -- Proof is not required

end symmetric_function_cannot_be_even_l1607_160760


namespace roots_reciprocal_l1607_160741

theorem roots_reciprocal (a b c x1 x2 x3 x4 : ℝ) 
  (h1 : a ≠ 0)
  (h2 : c ≠ 0)
  (hx1 : a * x1^2 + b * x1 + c = 0)
  (hx2 : a * x2^2 + b * x2 + c = 0)
  (hx3 : c * x3^2 + b * x3 + a = 0)
  (hx4 : c * x4^2 + b * x4 + a = 0) :
  (x3 = 1/x1 ∧ x4 = 1/x2) :=
  sorry

end roots_reciprocal_l1607_160741


namespace principal_amount_is_1200_l1607_160735

-- Define the given conditions
def simple_interest (P : ℝ) : ℝ := 0.10 * P
def compound_interest (P : ℝ) : ℝ := 0.1025 * P

-- Define given difference
def interest_difference (P : ℝ) : ℝ := compound_interest P - simple_interest P

-- The main goal is to prove that the principal amount P that satisfies the difference condition is 1200
theorem principal_amount_is_1200 : ∃ P : ℝ, interest_difference P = 3 ∧ P = 1200 :=
by
  sorry -- Proof to be completed

end principal_amount_is_1200_l1607_160735


namespace total_students_after_new_classes_l1607_160744

def initial_classes : ℕ := 15
def students_per_class : ℕ := 20
def new_classes : ℕ := 5

theorem total_students_after_new_classes :
  initial_classes * students_per_class + new_classes * students_per_class = 400 :=
by
  sorry

end total_students_after_new_classes_l1607_160744


namespace equal_values_of_means_l1607_160702

theorem equal_values_of_means (f : ℤ × ℤ → ℤ) 
  (h_pos : ∀ p, 0 < f p)
  (h_mean : ∀ p, f p = (f (p.1 + 1, p.2) + f (p.1 - 1, p.2) + f (p.1, p.2 + 1) + f (p.1, p.2 - 1)) / 4):
  ∃ m : ℤ, ∀ p, f p = m := sorry

end equal_values_of_means_l1607_160702


namespace range_of_m_l1607_160728

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, mx^2 + 2 * m * x + 1 > 0) ↔ (0 ≤ m ∧ m < 1) := by
  sorry

end range_of_m_l1607_160728


namespace distance_AB_l1607_160729

theorem distance_AB : 
  let A := -1
  let B := 2020
  |A - B| = 2021 := by
  sorry

end distance_AB_l1607_160729


namespace negation_of_proposition_l1607_160706

variables (x : ℝ)

def proposition (x : ℝ) : Prop := x > 0 → (x ≠ 2 → (x^3 / (x - 2) > 0))

theorem negation_of_proposition : ∃ x : ℝ, x > 0 ∧ 0 ≤ x ∧ x ≤ 2 :=
by
  sorry

end negation_of_proposition_l1607_160706


namespace sara_quarters_l1607_160790

-- Conditions
def usd_to_eur (usd : ℝ) : ℝ := usd * 0.85
def eur_to_usd (eur : ℝ) : ℝ := eur * 1.15
def value_of_quarter_usd : ℝ := 0.25
def dozen : ℕ := 12

-- Theorem
theorem sara_quarters (sara_savings_usd : ℝ) (usd_to_eur_ratio : ℝ) (eur_to_usd_ratio : ℝ) (quarter_value_usd : ℝ) (doz : ℕ) : sara_savings_usd = 9 → usd_to_eur_ratio = 0.85 → eur_to_usd_ratio = 1.15 → quarter_value_usd = 0.25 → doz = 12 → 
  ∃ dozens : ℕ, dozens = 2 :=
by
  sorry

end sara_quarters_l1607_160790


namespace number_of_acceptable_ages_l1607_160771

theorem number_of_acceptable_ages (avg_age : ℤ) (std_dev : ℤ) (a b : ℤ) (h_avg : avg_age = 10) (h_std : std_dev = 8)
    (h1 : a = avg_age - std_dev) (h2 : b = avg_age + std_dev) :
    b - a + 1 = 17 :=
by {
    sorry
}

end number_of_acceptable_ages_l1607_160771


namespace power_of_two_divisor_l1607_160795

theorem power_of_two_divisor {n : ℕ} (h_pos : n > 0) : 
  (∃ m : ℤ, (2^n - 1) ∣ (m^2 + 9)) → ∃ r : ℕ, n = 2^r :=
by
  sorry

end power_of_two_divisor_l1607_160795


namespace f_odd_f_monotonic_increasing_intervals_f_no_max_value_f_extreme_points_l1607_160746

noncomputable def f (x : ℝ) : ℝ := x^3 - x

theorem f_odd (x : ℝ) : f (-x) = -f (x) :=
by sorry

theorem f_monotonic_increasing_intervals :
  ∀ x : ℝ, (x < -Real.sqrt 3 / 3 ∨ x > Real.sqrt 3 / 3) → f x' > f x :=
by sorry

theorem f_no_max_value :
  ∀ x : ℝ, ¬(∃ M, f x ≤ M) :=
by sorry

theorem f_extreme_points :
  f (-Real.sqrt 3 / 3) = -2 * Real.sqrt 3 / 9 ∧ f (Real.sqrt 3 / 3) = -2 * Real.sqrt 3 / 9 :=
by sorry

end f_odd_f_monotonic_increasing_intervals_f_no_max_value_f_extreme_points_l1607_160746


namespace correct_calculation_l1607_160718

variable {a : ℝ}

theorem correct_calculation : a^2 * a^3 = a^5 :=
by sorry

end correct_calculation_l1607_160718


namespace arithmetic_sequence_diff_l1607_160775

theorem arithmetic_sequence_diff (a : ℕ → ℝ)
  (h1 : a 5 * a 7 = 6)
  (h2 : a 2 + a 10 = 5) :
  a 10 - a 6 = 2 ∨ a 10 - a 6 = -2 := by
  sorry

end arithmetic_sequence_diff_l1607_160775


namespace minimize_perimeter_isosceles_l1607_160708

noncomputable def inradius (A B C : ℝ) (r : ℝ) : Prop := sorry -- Define inradius

theorem minimize_perimeter_isosceles (A B C : ℝ) (r : ℝ) 
  (h1 : A + B + C = 180) -- Angles sum to 180 degrees
  (h2 : inradius A B C r) -- Given inradius
  (h3 : A = fixed_angle) -- Given fixed angle A
  : B = C :=
by sorry

end minimize_perimeter_isosceles_l1607_160708


namespace quadratic_unique_root_l1607_160731

theorem quadratic_unique_root (b c : ℝ)
  (h₁ : b = c^2 + 1)
  (h₂ : (x^2 + b * x + c = 0) → ∃! x : ℝ, x^2 + b * x + c = 0) :
  c = 1 ∨ c = -1 := 
sorry

end quadratic_unique_root_l1607_160731


namespace trigonometric_simplification_l1607_160768

noncomputable def tan : ℝ → ℝ := λ x => Real.sin x / Real.cos x
noncomputable def simp_expr : ℝ :=
  (tan (96 * Real.pi / 180) - tan (12 * Real.pi / 180) * (1 + 1 / Real.sin (6 * Real.pi / 180)))
  /
  (1 + tan (96 * Real.pi / 180) * tan (12 * Real.pi / 180) * (1 + 1 / Real.sin (6 * Real.pi / 180)))

theorem trigonometric_simplification : simp_expr = Real.sqrt 3 / 3 :=
by
  sorry

end trigonometric_simplification_l1607_160768


namespace candy_bar_multiple_l1607_160765

theorem candy_bar_multiple (s m x : ℕ) (h1 : s = m * x + 6) (h2 : x = 24) (h3 : s = 78) : m = 3 :=
by
  sorry

end candy_bar_multiple_l1607_160765


namespace g_extreme_value_f_ge_g_l1607_160789

noncomputable def f (x : ℝ) : ℝ := Real.exp (x + 1) - 2 / x + 1
noncomputable def g (x : ℝ) : ℝ := Real.log x / x + 2

theorem g_extreme_value :
  ∃ (x : ℝ), x = Real.exp 1 ∧ g x = 1 / Real.exp 1 + 2 :=
by sorry

theorem f_ge_g (x : ℝ) (hx : 0 < x) : f x >= g x :=
by sorry

end g_extreme_value_f_ge_g_l1607_160789


namespace find_diameter_of_hemisphere_l1607_160710

theorem find_diameter_of_hemisphere (r a : ℝ) (hr : r = a / 2) (volume : ℝ) (hV : volume = 18 * Real.pi) : 
  2/3 * Real.pi * r ^ 3 = 18 * Real.pi → a = 6 := by
  intro h
  sorry

end find_diameter_of_hemisphere_l1607_160710


namespace ninggao_intercity_project_cost_in_scientific_notation_l1607_160754

theorem ninggao_intercity_project_cost_in_scientific_notation :
  let length_kilometers := 55
  let cost_per_kilometer_million := 140
  let total_cost_million := length_kilometers * cost_per_kilometer_million
  let total_cost_scientific := 7.7 * 10^6
  total_cost_million = total_cost_scientific := 
  sorry

end ninggao_intercity_project_cost_in_scientific_notation_l1607_160754


namespace union_of_P_and_neg_RQ_l1607_160719

noncomputable def R : Set ℝ := Set.univ

noncomputable def P : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

noncomputable def Q : Set ℝ := {x | -2 < x ∧ x < 2}

noncomputable def neg_RQ : Set ℝ := {x | x ≤ -2 ∨ x ≥ 2}

theorem union_of_P_and_neg_RQ : 
  P ∪ neg_RQ = {x | x ≤ -2 ∨ 1 ≤ x ∧ x ≤ 3} :=
by
  sorry

end union_of_P_and_neg_RQ_l1607_160719


namespace mod_equiv_l1607_160796

theorem mod_equiv (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  (25 * m + 3 * n) % 83 = 0 ↔ (3 * m + 7 * n) % 83 = 0 :=
by
  sorry

end mod_equiv_l1607_160796


namespace arithmetic_sequence_b3b7_l1607_160770

theorem arithmetic_sequence_b3b7 (b : ℕ → ℤ) (d : ℤ)
  (h_arith_seq : ∀ n, b (n + 1) = b n + d)
  (h_increasing : ∀ n, b n < b (n + 1))
  (h_cond : b 4 * b 6 = 17) : 
  b 3 * b 7 = -175 :=
sorry

end arithmetic_sequence_b3b7_l1607_160770


namespace largest_n_satisfying_equation_l1607_160781

theorem largest_n_satisfying_equation :
  ∃ (x y z : ℕ), 0 < x ∧ 0 < y ∧ 0 < z ∧ ∀ n : ℕ,
  (n * n = x * x + y * y + z * z + 2 * x * y + 2 * y * z + 2 * z * x + 4 * x + 4 * y + 4 * z - 12) →
  n ≤ 2 :=
by
  sorry

end largest_n_satisfying_equation_l1607_160781


namespace cost_per_square_inch_l1607_160700

def length : ℕ := 9
def width : ℕ := 12
def total_cost : ℕ := 432

theorem cost_per_square_inch :
  total_cost / ((length * width) / 2) = 8 := 
by 
  sorry

end cost_per_square_inch_l1607_160700


namespace range_of_a_l1607_160782

noncomputable def range_a : Set ℝ :=
  {a : ℝ | 0 < a ∧ a ≤ 1/2}

theorem range_of_a (O P : ℝ × ℝ) (Q : ℝ × ℝ)
  (hO : O = (0, 0))
  (hP : P = (a, 0))
  (ha : 0 < a)
  (hQ : ∃ m : ℝ, Q = (m^2, m))
  (hPQ_PO : ∀ Q, Q = (m^2, m) → dist P Q ≥ dist O P) :
  a ∈ range_a :=
sorry

end range_of_a_l1607_160782


namespace parallel_statements_l1607_160703

-- Definitions for lines and planes
variables {Line Plane : Type}

-- Parallelism between a line and another line or a plane
variables (a b : Line) (α : Plane)

-- Parallel relationship assertions
axiom parallel_lines (l1 l2 : Line) : Prop -- l1 is parallel to l2
axiom line_in_plane (l : Line) (p : Plane) : Prop -- line l is in plane p
axiom parallel_line_plane (l : Line) (p : Plane) : Prop -- line l is parallel to plane p

-- Problem statement
theorem parallel_statements :
  (parallel_lines a b ∧ line_in_plane b α → parallel_line_plane a α) ∧
  (parallel_lines a b ∧ parallel_line_plane a α → parallel_line_plane b α) :=
sorry

end parallel_statements_l1607_160703


namespace original_number_value_l1607_160705

theorem original_number_value (x : ℝ) (h : 0 < x) (h_eq : 10^4 * x = 4 / x) : x = 0.02 :=
sorry

end original_number_value_l1607_160705


namespace sequence_behavior_l1607_160739

theorem sequence_behavior (b : ℕ → ℕ) :
  (∀ n, b n = n) ∨ ∃ N, ∀ n, n ≥ N → b n = b N :=
sorry

end sequence_behavior_l1607_160739


namespace color_changes_probability_l1607_160773

-- Define the durations of the traffic lights
def green_duration := 40
def yellow_duration := 5
def red_duration := 45

-- Define the total cycle duration
def total_cycle_duration := green_duration + yellow_duration + red_duration

-- Define the duration of the interval Mary watches
def watch_duration := 4

-- Define the change windows where the color changes can be witnessed
def change_windows :=
  [green_duration - watch_duration,
   green_duration + yellow_duration - watch_duration,
   green_duration + yellow_duration + red_duration - watch_duration]

-- Define the total change window duration
def total_change_window_duration := watch_duration * (change_windows.length)

-- Calculate the probability of witnessing a change
def probability_witnessing_change := (total_change_window_duration : ℚ) / total_cycle_duration

-- The theorem to prove
theorem color_changes_probability :
  probability_witnessing_change = 2 / 15 := by sorry

end color_changes_probability_l1607_160773


namespace x_can_be_positive_negative_or_zero_l1607_160724

noncomputable
def characteristics_of_x (x y z w : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0) (h4 : w ≠ 0) 
  (h5 : (x / y) > (z / w)) (h6 : (y * w) > 0) : Prop :=
  ∃ r : ℝ, r = x

theorem x_can_be_positive_negative_or_zero (x y z w : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0) 
  (h4 : w ≠ 0) (h5 : (x / y) > (z / w)) (h6 : (y * w) > 0) : 
  (characteristics_of_x x y z w h1 h2 h3 h4 h5 h6) :=
sorry

end x_can_be_positive_negative_or_zero_l1607_160724


namespace find_a_l1607_160747

theorem find_a (a : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + 2*x - 4*y = 0 → 3*x + y + a = 0) →
  a = 1 :=
by
  sorry

end find_a_l1607_160747


namespace count_bases_for_last_digit_l1607_160756

theorem count_bases_for_last_digit (n : ℕ) : n = 729 → ∃ S : Finset ℕ, S.card = 2 ∧ ∀ b ∈ S, 2 ≤ b ∧ b ≤ 10 ∧ (n - 5) % b = 0 :=
by
  sorry

end count_bases_for_last_digit_l1607_160756


namespace perpendicular_lines_m_value_l1607_160785

theorem perpendicular_lines_m_value
  (l1 : ∀ (x y : ℝ), x - 2 * y + 1 = 0)
  (l2 : ∀ (x y : ℝ), m * x + y - 3 = 0)
  (perpendicular : ∀ (m : ℝ) (l1_slope l2_slope : ℝ), l1_slope * l2_slope = -1) : 
  m = 2 :=
by
  sorry

end perpendicular_lines_m_value_l1607_160785


namespace jar_weight_percentage_l1607_160767

theorem jar_weight_percentage (J B : ℝ) (h : 0.60 * (J + B) = J + 1 / 3 * B) :
  (J / (J + B)) = 0.403 :=
by
  sorry

end jar_weight_percentage_l1607_160767
