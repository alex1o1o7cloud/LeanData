import Mathlib

namespace NUMINAMATH_GPT_value_of_a_l1281_128133

theorem value_of_a (a b : ℝ) (h1 : b = 4 * a) (h2 : b = 24 - 4 * a) : a = 3 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_l1281_128133


namespace NUMINAMATH_GPT_ball_hits_ground_time_l1281_128134

def ball_height (t : ℝ) : ℝ := -20 * t^2 + 30 * t + 60

theorem ball_hits_ground_time :
  ∃ t : ℝ, ball_height t = 0 ∧ t = (3 + Real.sqrt 57) / 4 :=
sorry

end NUMINAMATH_GPT_ball_hits_ground_time_l1281_128134


namespace NUMINAMATH_GPT_annual_growth_rate_l1281_128112

-- definitions based on the conditions in the problem
def FirstYear : ℝ := 400
def ThirdYear : ℝ := 625
def n : ℕ := 2

-- the main statement to prove the corresponding equation
theorem annual_growth_rate (x : ℝ) : 400 * (1 + x)^2 = 625 :=
sorry

end NUMINAMATH_GPT_annual_growth_rate_l1281_128112


namespace NUMINAMATH_GPT_pizzas_served_today_l1281_128161

theorem pizzas_served_today (lunch_pizzas : ℕ) (dinner_pizzas : ℕ) (h1 : lunch_pizzas = 9) (h2 : dinner_pizzas = 6) : lunch_pizzas + dinner_pizzas = 15 :=
by sorry

end NUMINAMATH_GPT_pizzas_served_today_l1281_128161


namespace NUMINAMATH_GPT_rope_segments_divided_l1281_128102

theorem rope_segments_divided (folds1 folds2 : ℕ) (cut : ℕ) (h_folds1 : folds1 = 3) (h_folds2 : folds2 = 2) (h_cut : cut = 1) :
  (folds1 * folds2 + cut = 7) :=
by {
  -- Proof steps would go here
  sorry
}

end NUMINAMATH_GPT_rope_segments_divided_l1281_128102


namespace NUMINAMATH_GPT_original_deck_size_l1281_128186

-- Let's define the number of red and black cards initially
def numRedCards (r : ℕ) : ℕ := r
def numBlackCards (b : ℕ) : ℕ := b

-- Define the initial condition as given in the problem
def initial_prob_red (r b : ℕ) : Prop :=
  r / (r + b) = 2 / 5

-- Define the condition after adding 7 black cards
def prob_red_after_adding_black (r b : ℕ) : Prop :=
  r / (r + (b + 7)) = 1 / 3

-- The proof statement to verify original number of cards in the deck
theorem original_deck_size (r b : ℕ) (h1 : initial_prob_red r b) (h2 : prob_red_after_adding_black r b) : r + b = 35 := by
  sorry

end NUMINAMATH_GPT_original_deck_size_l1281_128186


namespace NUMINAMATH_GPT_unique_n_divisors_satisfies_condition_l1281_128188

theorem unique_n_divisors_satisfies_condition:
  ∃ (n : ℕ), (∃ d1 d2 d3 : ℕ, d1 = 1 ∧ d2 > d1 ∧ d3 > d2 ∧ n = d3 ∧
  n = d2^2 + d3^3) ∧ n = 68 := by
  sorry

end NUMINAMATH_GPT_unique_n_divisors_satisfies_condition_l1281_128188


namespace NUMINAMATH_GPT_negation_universal_to_existential_l1281_128196

theorem negation_universal_to_existential :
  ¬ (∀ x : ℝ, x^2 - x ≥ 0) ↔ ∃ x : ℝ, x^2 - x < 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_universal_to_existential_l1281_128196


namespace NUMINAMATH_GPT_ticket_distribution_count_l1281_128189

theorem ticket_distribution_count :
  let A := 2
  let B := 2
  let C := 1
  let D := 1
  let total_tickets := A + B + C + D
  ∃ (num_dist : ℕ), num_dist = 180 :=
by {
  sorry
}

end NUMINAMATH_GPT_ticket_distribution_count_l1281_128189


namespace NUMINAMATH_GPT_cost_of_traveling_all_roads_l1281_128173

noncomputable def total_cost_of_roads (length width road_width : ℝ) (cost_per_sq_m : ℝ) : ℝ :=
  let area_road_parallel_length := length * road_width
  let area_road_parallel_breadth := width * road_width
  let diagonal_length := Real.sqrt (length^2 + width^2)
  let area_road_diagonal := diagonal_length * road_width
  let total_area := area_road_parallel_length + area_road_parallel_breadth + area_road_diagonal
  total_area * cost_per_sq_m

theorem cost_of_traveling_all_roads :
  total_cost_of_roads 80 50 10 3 = 6730.2 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_traveling_all_roads_l1281_128173


namespace NUMINAMATH_GPT_abs_non_positive_eq_zero_l1281_128160

theorem abs_non_positive_eq_zero (y : ℚ) (h : |4 * y - 7| ≤ 0) : y = 7 / 4 :=
by
  sorry

end NUMINAMATH_GPT_abs_non_positive_eq_zero_l1281_128160


namespace NUMINAMATH_GPT_min_units_l1281_128181

theorem min_units (x : ℕ) (h1 : 5500 * 60 + 5000 * (x - 60) > 550000) : x ≥ 105 := 
by {
  sorry
}

end NUMINAMATH_GPT_min_units_l1281_128181


namespace NUMINAMATH_GPT_jeremy_uncle_money_l1281_128190

def total_cost (num_jerseys : Nat) (cost_per_jersey : Nat) (basketball_cost : Nat) (shorts_cost : Nat) : Nat :=
  (num_jerseys * cost_per_jersey) + basketball_cost + shorts_cost

def total_money_given (total_cost : Nat) (money_left : Nat) : Nat :=
  total_cost + money_left

theorem jeremy_uncle_money :
  total_money_given (total_cost 5 2 18 8) 14 = 50 :=
by
  sorry

end NUMINAMATH_GPT_jeremy_uncle_money_l1281_128190


namespace NUMINAMATH_GPT_price_per_sq_ft_l1281_128113

def house_sq_ft : ℕ := 2400
def barn_sq_ft : ℕ := 1000
def total_property_value : ℝ := 333200

theorem price_per_sq_ft : 
  (total_property_value / (house_sq_ft + barn_sq_ft)) = 98 := 
by 
  sorry

end NUMINAMATH_GPT_price_per_sq_ft_l1281_128113


namespace NUMINAMATH_GPT_Jaco_total_gift_budget_l1281_128115

theorem Jaco_total_gift_budget :
  let friends_gifts := 8 * 9
  let parents_gifts := 2 * 14
  friends_gifts + parents_gifts = 100 :=
by
  let friends_gifts := 8 * 9
  let parents_gifts := 2 * 14
  show friends_gifts + parents_gifts = 100
  sorry

end NUMINAMATH_GPT_Jaco_total_gift_budget_l1281_128115


namespace NUMINAMATH_GPT_ab_cd_zero_l1281_128192

theorem ab_cd_zero (a b c d : ℝ) 
  (h1 : a^2 + b^2 = 1) 
  (h2 : c^2 + d^2 = 1) 
  (h3 : ac + bd = 0) : 
  ab + cd = 0 := 
sorry

end NUMINAMATH_GPT_ab_cd_zero_l1281_128192


namespace NUMINAMATH_GPT_sum_integers_30_to_50_subtract_15_l1281_128103

-- Definitions and proof problem based on conditions
def sumIntSeries (a b : ℕ) : ℕ :=
  (b - a + 1) * (a + b) / 2

theorem sum_integers_30_to_50_subtract_15 : sumIntSeries 30 50 - 15 = 825 := by
  -- We are stating that the sum of the integers from 30 to 50 minus 15 is equal to 825
  sorry


end NUMINAMATH_GPT_sum_integers_30_to_50_subtract_15_l1281_128103


namespace NUMINAMATH_GPT_xy_value_l1281_128141

variable (x y : ℕ)

def condition1 : Prop := 8^x / 4^(x + y) = 16
def condition2 : Prop := 16^(x + y) / 4^(7 * y) = 256

theorem xy_value (h1 : condition1 x y) (h2 : condition2 x y) : x * y = 48 := by
  sorry

end NUMINAMATH_GPT_xy_value_l1281_128141


namespace NUMINAMATH_GPT_novels_per_month_l1281_128125

theorem novels_per_month (pages_per_novel : ℕ) (total_pages_per_year : ℕ) (months_in_year : ℕ) 
  (h1 : pages_per_novel = 200) (h2 : total_pages_per_year = 9600) (h3 : months_in_year = 12) : 
  (total_pages_per_year / pages_per_novel) / months_in_year = 4 :=
by
  have novels_per_year := total_pages_per_year / pages_per_novel
  have novels_per_month := novels_per_year / months_in_year
  sorry

end NUMINAMATH_GPT_novels_per_month_l1281_128125


namespace NUMINAMATH_GPT_percentage_increase_consumption_l1281_128162

theorem percentage_increase_consumption
  (T C : ℝ) 
  (h_tax : ∀ t, t = 0.60 * T)
  (h_revenue : ∀ r, r = 0.75 * T * C) :
  1.25 * C = (0.75 * T * C) / (0.60 * T) := by
sorry

end NUMINAMATH_GPT_percentage_increase_consumption_l1281_128162


namespace NUMINAMATH_GPT_ratio_of_first_term_to_common_difference_l1281_128153

theorem ratio_of_first_term_to_common_difference
  (a d : ℝ)
  (h : (8 / 2 * (2 * a + 7 * d)) = 3 * (5 / 2 * (2 * a + 4 * d))) :
  a / d = 2 / 7 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_first_term_to_common_difference_l1281_128153


namespace NUMINAMATH_GPT_F_equiv_A_l1281_128121

-- Define the function F
def F : ℝ → ℝ := sorry

-- Given condition
axiom F_property (x : ℝ) : F ((1 - x) / (1 + x)) = x

-- The theorem that needs to be proved
theorem F_equiv_A (x : ℝ) : F (-2 - x) = -2 - F x := sorry

end NUMINAMATH_GPT_F_equiv_A_l1281_128121


namespace NUMINAMATH_GPT_composite_sum_l1281_128140

open Nat

theorem composite_sum (a b c d : ℕ) (h1 : c > b) (h2 : a + b + c + d = a * b - c * d) : ∃ x y : ℕ, x > 1 ∧ y > 1 ∧ a + c = x * y :=
by
  sorry

end NUMINAMATH_GPT_composite_sum_l1281_128140


namespace NUMINAMATH_GPT_find_g_l1281_128126

noncomputable def g : ℝ → ℝ
| x => 2 * (4^x - 3^x)

theorem find_g :
  (g 1 = 2) ∧
  (∀ x y : ℝ, g (x + y) = 4^y * g x + 3^x * g y) →
  ∀ x : ℝ, g x = 2 * (4^x - 3^x) := by
  sorry

end NUMINAMATH_GPT_find_g_l1281_128126


namespace NUMINAMATH_GPT_max_surface_area_of_rectangular_solid_on_sphere_l1281_128127

noncomputable def max_surface_area_rectangular_solid (a b c : ℝ) :=
  2 * a * b + 2 * a * c + 2 * b * c

theorem max_surface_area_of_rectangular_solid_on_sphere :
  (∀ (a b c : ℝ), a^2 + b^2 + c^2 = 36 → max_surface_area_rectangular_solid a b c ≤ 72) :=
by
  intros a b c h
  sorry

end NUMINAMATH_GPT_max_surface_area_of_rectangular_solid_on_sphere_l1281_128127


namespace NUMINAMATH_GPT_inequality_proof_l1281_128158

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^3 / (x^3 + 2 * y^2 * z) + y^3 / (y^3 + 2 * z^2 * x) + z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 := 
by 
  sorry

end NUMINAMATH_GPT_inequality_proof_l1281_128158


namespace NUMINAMATH_GPT_point_M_first_quadrant_distances_length_of_segment_MN_l1281_128149

-- Proof problem 1
theorem point_M_first_quadrant_distances (m : ℝ) (h1 : 2 * m + 1 > 0) (h2 : m + 3 > 0) (h3 : m + 3 = 2 * (2 * m + 1)) :
  m = 1 / 3 :=
by
  sorry

-- Proof problem 2
theorem length_of_segment_MN (m : ℝ) (h4 : m + 3 = 1) :
  let Mx := 2 * m + 1
  let My := m + 3
  let Nx := 2
  let Ny := 1
  let distMN := abs (Nx - Mx)
  distMN = 5 :=
by
  sorry

end NUMINAMATH_GPT_point_M_first_quadrant_distances_length_of_segment_MN_l1281_128149


namespace NUMINAMATH_GPT_conner_ties_sydney_l1281_128122

def sydney_initial_collect := 837
def conner_initial_collect := 723

def sydney_collect_day_one := 4
def conner_collect_day_one := 8 * sydney_collect_day_one / 2

def sydney_collect_day_two := (sydney_initial_collect + sydney_collect_day_one) - ((sydney_initial_collect + sydney_collect_day_one) / 10)
def conner_collect_day_two := conner_initial_collect + conner_collect_day_one + 123

def sydney_collect_day_three := sydney_collect_day_two + 2 * conner_collect_day_one
def conner_collect_day_three := (conner_collect_day_two - (123 / 4))

theorem conner_ties_sydney :
  sydney_collect_day_three <= conner_collect_day_three :=
by
  sorry

end NUMINAMATH_GPT_conner_ties_sydney_l1281_128122


namespace NUMINAMATH_GPT_work_rate_c_l1281_128172

theorem work_rate_c (A B C : ℝ) 
  (h1 : A + B = 1 / 15) 
  (h2 : A + B + C = 1 / 5) :
  (1 / C) = 7.5 :=
by 
  sorry

end NUMINAMATH_GPT_work_rate_c_l1281_128172


namespace NUMINAMATH_GPT_quadratic_equation_completing_square_l1281_128164

theorem quadratic_equation_completing_square :
  ∃ a b c : ℤ, a > 0 ∧ (25 * x^2 + 30 * x - 75 = 0 → (a * x + b)^2 = c) ∧ a + b + c = -58 :=
  sorry

end NUMINAMATH_GPT_quadratic_equation_completing_square_l1281_128164


namespace NUMINAMATH_GPT_problem_a2_b_c_in_M_l1281_128136

def P : Set ℤ := {x | ∃ k : ℤ, x = 3 * k + 1}
def Q : Set ℤ := {x | ∃ k : ℤ, x = 3 * k - 1}
def M : Set ℤ := {x | ∃ k : ℤ, x = 3 * k}

theorem problem_a2_b_c_in_M (a b c : ℤ) (ha : a ∈ P) (hb : b ∈ Q) (hc : c ∈ M) : 
  a^2 + b - c ∈ M :=
sorry

end NUMINAMATH_GPT_problem_a2_b_c_in_M_l1281_128136


namespace NUMINAMATH_GPT_prime_a_b_l1281_128135

theorem prime_a_b (a b : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) (h : a^11 + b = 2089) : 49 * b - a = 2007 :=
sorry

end NUMINAMATH_GPT_prime_a_b_l1281_128135


namespace NUMINAMATH_GPT_investment_ratio_l1281_128178

noncomputable def ratio_A_B (profit : ℝ) (profit_C : ℝ) (ratio_A_C : ℝ) (ratio_C_A : ℝ) := 
  3 / 1

theorem investment_ratio (total_profit : ℝ) (C_profit : ℝ) (A_C_ratio : ℝ) (C_A_ratio : ℝ) :
  total_profit = 60000 → C_profit = 20000 → A_C_ratio = 3 / 2 → ratio_A_B total_profit C_profit A_C_ratio C_A_ratio = 3 / 1 :=
by 
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_investment_ratio_l1281_128178


namespace NUMINAMATH_GPT_min_liars_in_presidium_l1281_128109

-- Define the conditions of the problem
def liars_and_truthlovers (grid : ℕ → ℕ → Prop) : Prop :=
  ∃ n : ℕ, n = 32 ∧ 
  (∀ i j, i < 4 ∧ j < 8 → 
    (∃ ni nj, (ni = i + 1 ∨ ni = i - 1 ∨ ni = i ∨ nj = j + 1 ∨ nj = j - 1 ∨ nj = j) ∧
      (ni < 4 ∧ nj < 8) → (grid i j ↔ ¬ grid ni nj)))

-- Define proof problem
theorem min_liars_in_presidium (grid : ℕ → ℕ → Prop) :
  liars_and_truthlovers grid → (∃ l, l = 8) := by
  sorry

end NUMINAMATH_GPT_min_liars_in_presidium_l1281_128109


namespace NUMINAMATH_GPT_load_transportable_l1281_128169

theorem load_transportable :
  ∃ (n : ℕ), n ≤ 11 ∧ (∀ (box_weight : ℕ) (total_weight : ℕ),
    total_weight = 13500 ∧ 
    box_weight ≤ 350 ∧ 
    (n * 1500) ≥ total_weight) :=
by
  sorry

end NUMINAMATH_GPT_load_transportable_l1281_128169


namespace NUMINAMATH_GPT_ratio_Umar_Yusaf_l1281_128185

variable (AliAge YusafAge UmarAge : ℕ)

-- Given conditions:
def Ali_is_8_years_old : Prop := AliAge = 8
def Ali_is_3_years_older_than_Yusaf : Prop := AliAge = YusafAge + 3
def Umar_is_10_years_old : Prop := UmarAge = 10

-- Proof statement:
theorem ratio_Umar_Yusaf (h1 : Ali_is_8_years_old AliAge)
                         (h2 : Ali_is_3_years_older_than_Yusaf AliAge YusafAge)
                         (h3 : Umar_is_10_years_old UmarAge) :
  UmarAge / YusafAge = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_Umar_Yusaf_l1281_128185


namespace NUMINAMATH_GPT_find_f_a5_a6_l1281_128106

-- Define the function properties and initial conditions
variables {f : ℝ → ℝ} {a : ℕ → ℝ} {S : ℕ → ℝ}

-- Conditions for the function f
axiom f_odd : ∀ x, f (-x) = -f x
axiom f_period : ∀ x, f (3/2 - x) = f x
axiom f_minus_2 : f (-2) = -3

-- Initial sequence condition and recursive relation
axiom a_1 : a 1 = -1
axiom S_def : ∀ n, S n = 2 * a n + n
axiom seq_recursive : ∀ n ≥ 2, S (n - 1) = 2 * a (n - 1) + (n - 1)

-- Theorem to prove
theorem find_f_a5_a6 : f (a 5) + f (a 6) = 3 := by
  sorry

end NUMINAMATH_GPT_find_f_a5_a6_l1281_128106


namespace NUMINAMATH_GPT_equation_of_line_BC_l1281_128191

/-
Given:
1. Point A(3, -1)
2. The line containing the median from A to side BC: 6x + 10y - 59 = 0
3. The line containing the angle bisector of ∠B: x - 4y + 10 = 0

Prove:
The equation of the line containing side BC is 2x + 9y - 65 = 0.
-/

noncomputable def point_A : (ℝ × ℝ) := (3, -1)

noncomputable def median_line (x y : ℝ) : Prop := 6 * x + 10 * y - 59 = 0

noncomputable def angle_bisector_line_B (x y : ℝ) : Prop := x - 4 * y + 10 = 0

theorem equation_of_line_BC :
  ∃ (x y : ℝ), 2 * x + 9 * y - 65 = 0 :=
sorry

end NUMINAMATH_GPT_equation_of_line_BC_l1281_128191


namespace NUMINAMATH_GPT_corn_height_after_three_weeks_l1281_128119

theorem corn_height_after_three_weeks 
  (week1_growth : ℕ) (week2_growth : ℕ) (week3_growth : ℕ) 
  (h1 : week1_growth = 2)
  (h2 : week2_growth = 2 * week1_growth)
  (h3 : week3_growth = 4 * week2_growth) :
  week1_growth + week2_growth + week3_growth = 22 :=
by {
  sorry
}

end NUMINAMATH_GPT_corn_height_after_three_weeks_l1281_128119


namespace NUMINAMATH_GPT_common_roots_product_l1281_128165

theorem common_roots_product
  (p q r s : ℝ)
  (hpqrs1 : p + q + r = 0)
  (hpqrs2 : pqr = -20)
  (hpqrs3 : p + q + s = -4)
  (hpqrs4 : pqs = -80)
  : p * q = 20 :=
sorry

end NUMINAMATH_GPT_common_roots_product_l1281_128165


namespace NUMINAMATH_GPT_shaded_area_fraction_l1281_128148

-- Define the problem conditions
def total_squares : ℕ := 18
def half_squares : ℕ := 10
def whole_squares : ℕ := 3

-- Define the total shaded area given the conditions
def shaded_area := (half_squares * (1/2) + whole_squares)

-- Define the total area of the rectangle
def total_area := total_squares

-- Lean 4 theorem statement
theorem shaded_area_fraction :
  shaded_area / total_area = (4 : ℚ) / 9 :=
by sorry

end NUMINAMATH_GPT_shaded_area_fraction_l1281_128148


namespace NUMINAMATH_GPT_multiple_of_x_l1281_128111

theorem multiple_of_x (k x y : ℤ) (hk : k * x + y = 34) (hx : 2 * x - y = 20) (hy : y^2 = 4) : k = 4 :=
sorry

end NUMINAMATH_GPT_multiple_of_x_l1281_128111


namespace NUMINAMATH_GPT_jogs_per_day_l1281_128175

-- Definitions of conditions
def weekdays_per_week : ℕ := 5
def total_weeks : ℕ := 3
def total_miles : ℕ := 75

-- Define the number of weekdays in total weeks
def total_weekdays : ℕ := total_weeks * weekdays_per_week

-- Theorem to prove Damien jogs 5 miles per day on weekdays
theorem jogs_per_day : total_miles / total_weekdays = 5 := by
  sorry

end NUMINAMATH_GPT_jogs_per_day_l1281_128175


namespace NUMINAMATH_GPT_expression_evaluation_l1281_128151

theorem expression_evaluation (x y : ℕ) (h1 : x = 3) (h2 : y = 2) : 3 * x^2 - 4 * y + 5 = 24 :=
by
  sorry

end NUMINAMATH_GPT_expression_evaluation_l1281_128151


namespace NUMINAMATH_GPT_line_form_l1281_128198

-- Given vector equation for a line
def line_eq (x y : ℝ) : Prop :=
  (3 * (x - 4) + 7 * (y - 14)) = 0

-- Prove that the line can be written in the form y = mx + b
theorem line_form (x y : ℝ) (h : line_eq x y) :
  y = (-3/7) * x + (110/7) :=
sorry

end NUMINAMATH_GPT_line_form_l1281_128198


namespace NUMINAMATH_GPT_find_C_l1281_128144

theorem find_C
  (A B C D : ℕ)
  (h1 : 0 ≤ A ∧ A ≤ 9)
  (h2 : 0 ≤ B ∧ B ≤ 9)
  (h3 : 0 ≤ C ∧ C ≤ 9)
  (h4 : 0 ≤ D ∧ D ≤ 9)
  (h5 : 4 * 1000 + A * 100 + 5 * 10 + B + (C * 1000 + 2 * 100 + D * 10 + 7) = 8070) :
  C = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_C_l1281_128144


namespace NUMINAMATH_GPT_ratio_of_rooms_l1281_128193

theorem ratio_of_rooms (rooms_danielle : ℕ) (rooms_grant : ℕ) (ratio_grant_heidi : ℚ)
  (h1 : rooms_danielle = 6)
  (h2 : rooms_grant = 2)
  (h3 : ratio_grant_heidi = 1/9) :
  (18 : ℚ) / rooms_danielle = 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_rooms_l1281_128193


namespace NUMINAMATH_GPT_b_catches_A_distance_l1281_128194

noncomputable def speed_A := 10 -- kmph
noncomputable def speed_B := 20 -- kmph
noncomputable def time_diff := 7 -- hours
noncomputable def distance_A := speed_A * time_diff -- km
noncomputable def relative_speed := speed_B - speed_A -- kmph
noncomputable def catch_up_time := distance_A / relative_speed -- hours
noncomputable def distance_B := speed_B * catch_up_time -- km

theorem b_catches_A_distance :
  distance_B = 140 := by
  sorry

end NUMINAMATH_GPT_b_catches_A_distance_l1281_128194


namespace NUMINAMATH_GPT_single_point_graph_value_of_d_l1281_128146

theorem single_point_graph_value_of_d (d : ℝ) :
  (∀ x y : ℝ, 3 * x^2 + y^2 + 12 * x - 6 * y + d = 0 → x = -2 ∧ y = 3) ↔ d = 21 := 
by 
  sorry

end NUMINAMATH_GPT_single_point_graph_value_of_d_l1281_128146


namespace NUMINAMATH_GPT_unique_prime_solution_l1281_128179

-- Define the problem in terms of prime numbers and checking the conditions
open Nat

def is_prime (n : ℕ) : Prop := 2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem unique_prime_solution (p : ℕ) (hp : is_prime p) (h1 : is_prime (p^2 - 6)) (h2 : is_prime (p^2 + 6)) : p = 5 := 
sorry

end NUMINAMATH_GPT_unique_prime_solution_l1281_128179


namespace NUMINAMATH_GPT_probability_all_vertical_faces_green_l1281_128118

theorem probability_all_vertical_faces_green :
  let color_prob := (1 / 2 : ℚ)
  let total_arrangements := 2^6
  let valid_arrangements := 2 + 12 + 6
  ((valid_arrangements : ℚ) / total_arrangements) = 5 / 16 := by
  sorry

end NUMINAMATH_GPT_probability_all_vertical_faces_green_l1281_128118


namespace NUMINAMATH_GPT_minimalYellowFraction_l1281_128110

-- Definitions
def totalSurfaceArea (sideLength : ℕ) : ℕ := 6 * (sideLength * sideLength)

def minimalYellowExposedArea : ℕ := 15

theorem minimalYellowFraction (sideLength : ℕ) (totalYellow : ℕ) (totalBlue : ℕ) 
    (totalCubes : ℕ) (yellowExposed : ℕ) :
    sideLength = 4 → totalYellow = 16 → totalBlue = 48 →
    totalCubes = 64 → yellowExposed = minimalYellowExposedArea →
    (yellowExposed / (totalSurfaceArea sideLength) : ℚ) = 5 / 32 :=
by
  sorry

end NUMINAMATH_GPT_minimalYellowFraction_l1281_128110


namespace NUMINAMATH_GPT_yogurt_banana_slices_l1281_128183

/--
Given:
1. Each banana yields 10 slices.
2. Vivian needs to make 5 yogurts.
3. She needs to buy 4 bananas.

Prove:
The number of banana slices needed for each yogurt is 8.
-/
theorem yogurt_banana_slices 
    (slices_per_banana : ℕ)
    (bananas_bought : ℕ)
    (yogurts_needed : ℕ)
    (h1 : slices_per_banana = 10)
    (h2 : yogurts_needed = 5)
    (h3 : bananas_bought = 4) : 
    (bananas_bought * slices_per_banana) / yogurts_needed = 8 :=
by
  sorry

end NUMINAMATH_GPT_yogurt_banana_slices_l1281_128183


namespace NUMINAMATH_GPT_x_investment_amount_l1281_128104

variable (X : ℝ)
variable (investment_y : ℝ := 15000)
variable (total_profit : ℝ := 1600)
variable (x_share : ℝ := 400)

theorem x_investment_amount :
  (total_profit - x_share) / investment_y = x_share / X → X = 5000 :=
by
  intro ratio
  have h1: 1200 / 15000 = 400 / 5000 :=
    by sorry
  have h2: X = 5000 :=
    by sorry
  exact h2

end NUMINAMATH_GPT_x_investment_amount_l1281_128104


namespace NUMINAMATH_GPT_sum_of_first_100_positive_odd_integers_is_correct_l1281_128124

def sum_first_100_positive_odd_integers : ℕ :=
  10000

theorem sum_of_first_100_positive_odd_integers_is_correct :
  sum_first_100_positive_odd_integers = 10000 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_first_100_positive_odd_integers_is_correct_l1281_128124


namespace NUMINAMATH_GPT_total_value_of_coins_is_correct_l1281_128168

def rolls_dollars : ℕ := 6
def rolls_half_dollars : ℕ := 5
def rolls_quarters : ℕ := 7
def rolls_dimes : ℕ := 4
def rolls_nickels : ℕ := 3
def rolls_pennies : ℕ := 2

def coins_per_dollar_roll : ℕ := 20
def coins_per_half_dollar_roll : ℕ := 25
def coins_per_quarter_roll : ℕ := 40
def coins_per_dime_roll : ℕ := 50
def coins_per_nickel_roll : ℕ := 40
def coins_per_penny_roll : ℕ := 50

def value_per_dollar : ℚ := 1
def value_per_half_dollar : ℚ := 0.5
def value_per_quarter : ℚ := 0.25
def value_per_dime : ℚ := 0.10
def value_per_nickel : ℚ := 0.05
def value_per_penny : ℚ := 0.01

theorem total_value_of_coins_is_correct : 
  rolls_dollars * coins_per_dollar_roll * value_per_dollar +
  rolls_half_dollars * coins_per_half_dollar_roll * value_per_half_dollar +
  rolls_quarters * coins_per_quarter_roll * value_per_quarter +
  rolls_dimes * coins_per_dime_roll * value_per_dime +
  rolls_nickels * coins_per_nickel_roll * value_per_nickel +
  rolls_pennies * coins_per_penny_roll * value_per_penny = 279.50 := 
sorry

end NUMINAMATH_GPT_total_value_of_coins_is_correct_l1281_128168


namespace NUMINAMATH_GPT_proof_problem_l1281_128180

noncomputable def arithmetic_mean (a b : ℝ) : ℝ :=
  (a + b) / 2

noncomputable def geometric_mean (x y : ℝ) : ℝ :=
  Real.sqrt (x * y)

theorem proof_problem (a b c x y z m : ℝ) (x_pos : 0 < x) (y_pos : 0 < y) (z_pos : 0 < z) (m_pos : 0 < m) (m_ne_one : m ≠ 1) 
  (h_b : b = arithmetic_mean a c) (h_y : y = geometric_mean x z) :
  (b - c) * Real.logb m x + (c - a) * Real.logb m y + (a - b) * Real.logb m z = 0 := by
  sorry

end NUMINAMATH_GPT_proof_problem_l1281_128180


namespace NUMINAMATH_GPT_geometric_sum_sequence_l1281_128155

theorem geometric_sum_sequence (n : ℕ) (a : ℕ → ℕ) (a1 : a 1 = 2) (a4 : a 4 = 16) :
    (∃ q : ℕ, a 2 = a 1 * q) → (∃ S_n : ℕ, S_n = 2 * (2 ^ n - 1)) :=
by
  sorry

end NUMINAMATH_GPT_geometric_sum_sequence_l1281_128155


namespace NUMINAMATH_GPT_milton_apple_pie_slices_l1281_128159

theorem milton_apple_pie_slices :
  ∀ (A : ℕ),
  (∀ (peach_pie_slices_per : ℕ), peach_pie_slices_per = 6) →
  (∀ (apple_pie_slices_sold : ℕ), apple_pie_slices_sold = 56) →
  (∀ (peach_pie_slices_sold : ℕ), peach_pie_slices_sold = 48) →
  (∀ (total_pies_sold : ℕ), total_pies_sold = 15) →
  (∃ (apple_pie_slices : ℕ), apple_pie_slices = 56 / (total_pies_sold - (peach_pie_slices_sold / peach_pie_slices_per))) → 
  A = 8 :=
by sorry

end NUMINAMATH_GPT_milton_apple_pie_slices_l1281_128159


namespace NUMINAMATH_GPT_dave_coins_l1281_128184

theorem dave_coins :
  ∃ n : ℕ, n ≡ 2 [MOD 7] ∧ n ≡ 3 [MOD 5] ∧ n ≡ 1 [MOD 3] ∧ n = 58 :=
sorry

end NUMINAMATH_GPT_dave_coins_l1281_128184


namespace NUMINAMATH_GPT_number_of_indeterminate_conditions_l1281_128100

noncomputable def angle_sum (A B C : ℝ) : Prop := A + B + C = 180
noncomputable def condition1 (A B C : ℝ) : Prop := A + B = C
noncomputable def condition2 (A B C : ℝ) : Prop := A = C / 6 ∧ B = 2 * (C / 6)
noncomputable def condition3 (A B : ℝ) : Prop := A = 90 - B
noncomputable def condition4 (A B C : ℝ) : Prop := A = B ∧ B = C
noncomputable def condition5 (A B C : ℝ) : Prop := 2 * A = C ∧ 2 * B = C
noncomputable def is_right_triangle (C : ℝ) : Prop := C = 90

theorem number_of_indeterminate_conditions (A B C : ℝ) :
  (angle_sum A B C) →
  (condition1 A B C → is_right_triangle C) →
  (condition2 A B C → is_right_triangle C) →
  (condition3 A B → is_right_triangle C) →
  (condition4 A B C → ¬ is_right_triangle C) →
  (condition5 A B C → is_right_triangle C) →
  ∃ n, n = 1 :=
sorry

end NUMINAMATH_GPT_number_of_indeterminate_conditions_l1281_128100


namespace NUMINAMATH_GPT_smallest_four_digit_divisible_by_35_l1281_128145

theorem smallest_four_digit_divisible_by_35 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 35 = 0 → n ≤ m ∧ n = 1006 :=
by
  sorry

end NUMINAMATH_GPT_smallest_four_digit_divisible_by_35_l1281_128145


namespace NUMINAMATH_GPT_evaluate_expression_l1281_128195

noncomputable def cuberoot (x : ℝ) : ℝ := x ^ (1 / 3)

theorem evaluate_expression : 
  cuberoot (1 + 27) * cuberoot (1 + cuberoot 27) = cuberoot 112 := 
by 
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1281_128195


namespace NUMINAMATH_GPT_roof_collapse_days_l1281_128117

def leaves_per_pound : ℕ := 1000
def pounds_limit_of_roof : ℕ := 500
def leaves_per_day : ℕ := 100

theorem roof_collapse_days : (pounds_limit_of_roof * leaves_per_pound) / leaves_per_day = 5000 := by
  sorry

end NUMINAMATH_GPT_roof_collapse_days_l1281_128117


namespace NUMINAMATH_GPT_count_8_digit_even_ending_l1281_128147

theorem count_8_digit_even_ending : 
  let choices_first_digit := 9
  let choices_middle_digits := 10 ^ 6
  let choices_last_digit := 5
  (choices_first_digit * choices_middle_digits * choices_last_digit) = 45000000 :=
by
  let choices_first_digit := 9
  let choices_middle_digits := 10 ^ 6
  let choices_last_digit := 5
  sorry

end NUMINAMATH_GPT_count_8_digit_even_ending_l1281_128147


namespace NUMINAMATH_GPT_distribute_6_balls_in_3_boxes_l1281_128101

def number_of_ways_to_distribute_balls (balls boxes : Nat) : Nat :=
  boxes ^ balls

theorem distribute_6_balls_in_3_boxes : number_of_ways_to_distribute_balls 6 3 = 729 := by
  sorry

end NUMINAMATH_GPT_distribute_6_balls_in_3_boxes_l1281_128101


namespace NUMINAMATH_GPT_min_value_x_add_y_div_2_l1281_128142

theorem min_value_x_add_y_div_2 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x * y - 2 * x - y = 0) :
  ∃ x y, 0 < x ∧ 0 < y ∧ (x * y - 2 * x - y = 0 ∧ x + y / 2 = 4) :=
sorry

end NUMINAMATH_GPT_min_value_x_add_y_div_2_l1281_128142


namespace NUMINAMATH_GPT_fraction_sequence_calc_l1281_128167

theorem fraction_sequence_calc : 
  (2 / 3) * (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) * (8 / 9) - 1 = -(7 / 9) := 
by 
  sorry

end NUMINAMATH_GPT_fraction_sequence_calc_l1281_128167


namespace NUMINAMATH_GPT_parabola_opens_downwards_iff_l1281_128176

theorem parabola_opens_downwards_iff (a : ℝ) : (∀ x : ℝ, (a - 1) * x^2 + 2 * x ≤ 0) ↔ a < 1 := 
sorry

end NUMINAMATH_GPT_parabola_opens_downwards_iff_l1281_128176


namespace NUMINAMATH_GPT_andrei_stamps_l1281_128170

theorem andrei_stamps (x : ℕ) : 
  (x % 3 = 1) ∧ (x % 5 = 3) ∧ (x % 7 = 5) ∧ (150 < x) ∧ (x ≤ 300) → 
  x = 208 :=
sorry

end NUMINAMATH_GPT_andrei_stamps_l1281_128170


namespace NUMINAMATH_GPT_second_storm_duration_l1281_128105

theorem second_storm_duration (x y : ℕ) 
  (h1 : x + y = 45) 
  (h2 : 30 * x + 15 * y = 975) : 
  y = 25 :=
by
  sorry

end NUMINAMATH_GPT_second_storm_duration_l1281_128105


namespace NUMINAMATH_GPT_sophomores_stratified_sampling_l1281_128131

theorem sophomores_stratified_sampling 
  (total_students freshmen sophomores seniors selected_total : ℕ) 
  (H1 : total_students = 2800) 
  (H2 : freshmen = 970) 
  (H3 : sophomores = 930) 
  (H4 : seniors = 900) 
  (H_selected_total : selected_total = 280) : 
  (sophomores / total_students) * selected_total = 93 :=
by sorry

end NUMINAMATH_GPT_sophomores_stratified_sampling_l1281_128131


namespace NUMINAMATH_GPT_lincoln_county_houses_l1281_128107

theorem lincoln_county_houses (original_houses : ℕ) (built_houses : ℕ) (total_houses : ℕ) 
(h1 : original_houses = 20817) 
(h2 : built_houses = 97741) 
(h3 : total_houses = original_houses + built_houses) : 
total_houses = 118558 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_lincoln_county_houses_l1281_128107


namespace NUMINAMATH_GPT_tan_half_sum_l1281_128123

theorem tan_half_sum (p q : ℝ)
  (h1 : Real.cos p + Real.cos q = (1:ℝ)/3)
  (h2 : Real.sin p + Real.sin q = (8:ℝ)/17) :
  Real.tan ((p + q) / 2) = (24:ℝ)/17 := 
sorry

end NUMINAMATH_GPT_tan_half_sum_l1281_128123


namespace NUMINAMATH_GPT_unique_triple_solution_l1281_128157

theorem unique_triple_solution (x y z : ℝ) :
  (1 + x^4 ≤ 2 * (y - z)^2) →
  (1 + y^4 ≤ 2 * (z - x)^2) →
  (1 + z^4 ≤ 2 * (x - y)^2) →
  (x = 1 ∧ y = 0 ∧ z = -1) :=
sorry

end NUMINAMATH_GPT_unique_triple_solution_l1281_128157


namespace NUMINAMATH_GPT_sum_of_coordinates_point_D_l1281_128129

theorem sum_of_coordinates_point_D 
(M : ℝ × ℝ) (C D : ℝ × ℝ) 
(hM : M = (3, 5)) 
(hC : C = (1, 10)) 
(hmid : M = ((C.1 + D.1) / 2, (C.2 + D.2) / 2))
: D.1 + D.2 = 5 :=
sorry

end NUMINAMATH_GPT_sum_of_coordinates_point_D_l1281_128129


namespace NUMINAMATH_GPT_adults_eat_one_third_l1281_128163

theorem adults_eat_one_third (n c k : ℕ) (hn : n = 120) (hc : c = 4) (hk : k = 20) :
  ((n - c * k) / n : ℚ) = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_adults_eat_one_third_l1281_128163


namespace NUMINAMATH_GPT_jesse_total_carpet_l1281_128137

theorem jesse_total_carpet : 
  let length_rect := 12
  let width_rect := 8
  let base_tri := 10
  let height_tri := 6
  let area_rect := length_rect * width_rect
  let area_tri := (base_tri * height_tri) / 2
  area_rect + area_tri = 126 :=
by
  sorry

end NUMINAMATH_GPT_jesse_total_carpet_l1281_128137


namespace NUMINAMATH_GPT_roots_difference_one_l1281_128199

theorem roots_difference_one (p : ℝ) :
  (∃ (x y : ℝ), (x^3 - 7 * x + p = 0) ∧ (y^3 - 7 * y + p = 0) ∧ (x - y = 1)) ↔ (p = 6 ∨ p = -6) :=
sorry

end NUMINAMATH_GPT_roots_difference_one_l1281_128199


namespace NUMINAMATH_GPT_ratio_proof_l1281_128171

theorem ratio_proof (a b c d : ℝ) (h1 : a / b = 20) (h2 : c / b = 5) (h3 : c / d = 1 / 8) : 
  a / d = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_proof_l1281_128171


namespace NUMINAMATH_GPT_carrie_spent_money_l1281_128187

variable (cost_per_tshirt : ℝ) (num_tshirts : ℕ)

theorem carrie_spent_money (h1 : cost_per_tshirt = 9.95) (h2 : num_tshirts = 20) :
  cost_per_tshirt * num_tshirts = 199 := by
  sorry

end NUMINAMATH_GPT_carrie_spent_money_l1281_128187


namespace NUMINAMATH_GPT_katya_solves_enough_l1281_128174

theorem katya_solves_enough (x : ℕ) :
  (0 ≤ x ∧ x ≤ 20) → -- x should be within the valid range of problems
  (4 / 5) * x + (1 / 2) * (20 - x) ≥ 13 → 
  x ≥ 10 :=
by 
  intros h₁ h₂
  -- Formalize the expected value equation and the inequality transformations
  sorry

end NUMINAMATH_GPT_katya_solves_enough_l1281_128174


namespace NUMINAMATH_GPT_linear_function_quadrant_l1281_128143

theorem linear_function_quadrant (x y : ℝ) (h : y = 2 * x - 3) : ¬(∃ x y : ℝ, x < 0 ∧ y > 0 ∧ y = 2 * x - 3) :=
sorry

end NUMINAMATH_GPT_linear_function_quadrant_l1281_128143


namespace NUMINAMATH_GPT_gcd_in_base3_l1281_128132

def gcd_2134_1455_is_97 : ℕ :=
  gcd 2134 1455

def base3 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (n : ℕ) : List ℕ :=
      if n = 0 then [] else aux (n / 3) ++ [n % 3]
    aux n

theorem gcd_in_base3 :
  gcd_2134_1455_is_97 = 97 ∧ base3 97 = [1, 0, 1, 2, 1] :=
by
  sorry

end NUMINAMATH_GPT_gcd_in_base3_l1281_128132


namespace NUMINAMATH_GPT_integer_solutions_exist_l1281_128152

theorem integer_solutions_exist (a : ℕ) (ha : 0 < a) :
  ∃ x y : ℤ, x^2 - y^2 = a^3 := 
sorry

end NUMINAMATH_GPT_integer_solutions_exist_l1281_128152


namespace NUMINAMATH_GPT_servings_needed_l1281_128182

theorem servings_needed
  (pieces_per_serving : ℕ)
  (jared_consumption : ℕ)
  (three_friends_consumption : ℕ)
  (another_three_friends_consumption : ℕ)
  (last_four_friends_consumption : ℕ) : 
  pieces_per_serving = 60 →
  jared_consumption = 150 →
  three_friends_consumption = 3 * 80 →
  another_three_friends_consumption = 3 * 200 →
  last_four_friends_consumption = 4 * 100 →
  ∃ (s : ℕ), s = 24 :=
by
  intros
  sorry

end NUMINAMATH_GPT_servings_needed_l1281_128182


namespace NUMINAMATH_GPT_not_B_l1281_128114

def op (x y : ℝ) := (x - y) ^ 2

theorem not_B (x y : ℝ) : 2 * (op x y) ≠ op (2 * x) (2 * y) :=
by
  sorry

end NUMINAMATH_GPT_not_B_l1281_128114


namespace NUMINAMATH_GPT_rectangle_area_l1281_128166

theorem rectangle_area (P L W : ℝ) (hP : P = 2 * (L + W)) (hRatio : L / W = 5 / 2) (hP_val : P = 280) : 
  L * W = 4000 :=
by 
  sorry

end NUMINAMATH_GPT_rectangle_area_l1281_128166


namespace NUMINAMATH_GPT_max_and_min_of_expression_l1281_128197

variable {x y : ℝ}

theorem max_and_min_of_expression (h : |5 * x + y| + |5 * x - y| = 20) : 
  (∃ (maxQ minQ : ℝ), maxQ = 124 ∧ minQ = 3 ∧ 
  (∀ z, z = x^2 - x * y + y^2 → z <= 124 ∧ z >= 3)) :=
sorry

end NUMINAMATH_GPT_max_and_min_of_expression_l1281_128197


namespace NUMINAMATH_GPT_library_pupils_count_l1281_128150

-- Definitions for the conditions provided in the problem
def num_rectangular_tables : Nat := 7
def num_pupils_per_rectangular_table : Nat := 10
def num_square_tables : Nat := 5
def num_pupils_per_square_table : Nat := 4

-- Theorem stating the problem's question and the required proof
theorem library_pupils_count :
  num_rectangular_tables * num_pupils_per_rectangular_table + 
  num_square_tables * num_pupils_per_square_table = 90 :=
sorry

end NUMINAMATH_GPT_library_pupils_count_l1281_128150


namespace NUMINAMATH_GPT_find_RS_length_l1281_128108

-- Define the conditions and the problem in Lean

theorem find_RS_length
  (radius : ℝ)
  (P Q R S T : ℝ)
  (center_to_T : ℝ)
  (PT : ℝ)
  (PQ : ℝ)
  (RT TS : ℝ)
  (h_radius : radius = 7)
  (h_center_to_T : center_to_T = 3)
  (h_PT : PT = 8)
  (h_bisect_PQ : PQ = 2 * PT)
  (h_intersecting_chords : PT * (PQ / 2) = RT * TS)
  (h_perfect_square : ∃ k : ℝ, k^2 = RT * TS) :
  RS = 16 :=
by
  sorry

end NUMINAMATH_GPT_find_RS_length_l1281_128108


namespace NUMINAMATH_GPT_solve_cubic_eq_l1281_128156

theorem solve_cubic_eq (x : ℝ) : (8 - x)^3 = x^3 → x = 8 :=
by
  sorry

end NUMINAMATH_GPT_solve_cubic_eq_l1281_128156


namespace NUMINAMATH_GPT_find_x_when_y_equals_two_l1281_128128

theorem find_x_when_y_equals_two (x : ℝ) (y : ℝ) (h1 : y = 1 / (4 * x + 2)) (h2 : y = 2) : x = -3 / 8 :=
by
  sorry

end NUMINAMATH_GPT_find_x_when_y_equals_two_l1281_128128


namespace NUMINAMATH_GPT_math_problem_statements_are_correct_l1281_128116

theorem math_problem_statements_are_correct (a b : ℝ) (h : a > b ∧ b > 0) :
  (¬ (b / a > (b + 3) / (a + 3))) ∧ ((3 * a + 2 * b) / (2 * a + 3 * b) < a / b) ∧
  (¬ (2 * Real.sqrt a < Real.sqrt (a - b) + Real.sqrt b)) ∧ 
  (Real.log ((a + b) / 2) > (Real.log a + Real.log b) / 2) :=
by
  sorry

end NUMINAMATH_GPT_math_problem_statements_are_correct_l1281_128116


namespace NUMINAMATH_GPT_inequality_for_positive_integer_l1281_128154

theorem inequality_for_positive_integer (n : ℕ) (h : n > 0) :
  n^n ≤ (n!)^2 ∧ (n!)^2 ≤ ((n + 1) * (n + 2) / 6)^n := by
  sorry

end NUMINAMATH_GPT_inequality_for_positive_integer_l1281_128154


namespace NUMINAMATH_GPT_problem_statement_l1281_128177

theorem problem_statement :
  ∃ (w x y z : ℕ), (2^w * 3^x * 5^y * 7^z = 588) ∧ (2 * w + 3 * x + 5 * y + 7 * z = 21) :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1281_128177


namespace NUMINAMATH_GPT_range_of_c_l1281_128138

theorem range_of_c (x y c : ℝ) (h1 : x^2 + (y - 2)^2 = 1) (h2 : x^2 + y^2 + c ≤ 0) : c ≤ -9 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_range_of_c_l1281_128138


namespace NUMINAMATH_GPT_geometric_sequence_a3_eq_2_l1281_128139

theorem geometric_sequence_a3_eq_2 
  (a_1 a_3 a_5 : ℝ) 
  (h1 : a_1 * a_3 * a_5 = 8) 
  (h2 : a_3^2 = a_1 * a_5) : 
  a_3 = 2 :=
by 
  sorry

end NUMINAMATH_GPT_geometric_sequence_a3_eq_2_l1281_128139


namespace NUMINAMATH_GPT_set_A_main_inequality_l1281_128130

def f (x : ℝ) : ℝ := |2 * x + 1| + |x - 2|
def A : Set ℝ := {x | f x < 3}

theorem set_A :
  A = {x | -2 / 3 < x ∧ x < 0} :=
sorry

theorem main_inequality (s t : ℝ) (hs : -2 / 3 < s ∧ s < 0) (ht : -2 / 3 < t ∧ t < 0) :
  |1 - t / s| < |t - 1 / s| :=
sorry

end NUMINAMATH_GPT_set_A_main_inequality_l1281_128130


namespace NUMINAMATH_GPT_x_to_the_12_eq_14449_l1281_128120

/-
Given the condition x + 1/x = 2*sqrt(2), prove that x^12 = 14449.
-/

theorem x_to_the_12_eq_14449 (x : ℂ) (hx : x + 1/x = 2 * Real.sqrt 2) : x^12 = 14449 := 
sorry

end NUMINAMATH_GPT_x_to_the_12_eq_14449_l1281_128120
