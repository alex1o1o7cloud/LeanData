import Mathlib

namespace NUMINAMATH_GPT_box_width_l861_86121

theorem box_width (rate : ℝ) (time : ℝ) (length : ℝ) (depth : ℝ) (volume : ℝ) (width : ℝ) : 
  rate = 4 ∧ time = 21 ∧ length = 7 ∧ depth = 2 ∧ volume = rate * time ∧ volume = length * width * depth → width = 6 :=
by
  sorry

end NUMINAMATH_GPT_box_width_l861_86121


namespace NUMINAMATH_GPT_oil_truck_radius_l861_86128

/-- 
A full stationary oil tank that is a right circular cylinder has a radius of 100 feet 
and a height of 25 feet. Oil is pumped from the stationary tank to an oil truck that 
has a tank that is a right circular cylinder. The oil level dropped 0.025 feet in the stationary tank. 
The oil truck's tank has a height of 10 feet. The radius of the oil truck's tank is 5 feet. 
--/
theorem oil_truck_radius (r_stationary : ℝ) (h_stationary : ℝ) (h_truck : ℝ) 
  (Δh : ℝ) (r_truck : ℝ) 
  (h_stationary_pos : 0 < h_stationary) (h_truck_pos : 0 < h_truck) (r_stationary_pos : 0 < r_stationary) :
  r_stationary = 100 → h_stationary = 25 → Δh = 0.025 → h_truck = 10 → r_truck = 5 → 
  π * (r_stationary ^ 2) * Δh = π * (r_truck ^ 2) * h_truck :=
by 
  -- Use the conditions and perform algebra to show the equality.
  sorry

end NUMINAMATH_GPT_oil_truck_radius_l861_86128


namespace NUMINAMATH_GPT_insurance_plan_percentage_l861_86135

theorem insurance_plan_percentage
(MSRP : ℝ) (I : ℝ) (total_cost : ℝ) (state_tax_rate : ℝ)
(hMSRP : MSRP = 30)
(htotal_cost : total_cost = 54)
(hstate_tax_rate : state_tax_rate = 0.5)
(h_total_cost_eq : MSRP + I + state_tax_rate * (MSRP + I) = total_cost) :
(I / MSRP) * 100 = 20 :=
by
  -- You can leave the proof as sorry, as it's not needed for the problem
  sorry

end NUMINAMATH_GPT_insurance_plan_percentage_l861_86135


namespace NUMINAMATH_GPT_classrooms_student_hamster_difference_l861_86150

-- Define the problem conditions
def students_per_classroom := 22
def hamsters_per_classroom := 3
def number_of_classrooms := 5

-- Define the problem statement
theorem classrooms_student_hamster_difference :
  (students_per_classroom * number_of_classrooms) - 
  (hamsters_per_classroom * number_of_classrooms) = 95 :=
by
  sorry

end NUMINAMATH_GPT_classrooms_student_hamster_difference_l861_86150


namespace NUMINAMATH_GPT_z_is_greater_by_50_percent_of_w_l861_86177

variable (w q y z : ℝ)

def w_is_60_percent_q : Prop := w = 0.60 * q
def q_is_60_percent_y : Prop := q = 0.60 * y
def z_is_54_percent_y : Prop := z = 0.54 * y

theorem z_is_greater_by_50_percent_of_w (h1 : w_is_60_percent_q w q) 
                                        (h2 : q_is_60_percent_y q y) 
                                        (h3 : z_is_54_percent_y z y) : 
  ((z - w) / w) * 100 = 50 :=
sorry

end NUMINAMATH_GPT_z_is_greater_by_50_percent_of_w_l861_86177


namespace NUMINAMATH_GPT_c_over_e_l861_86142

theorem c_over_e (a b c d e : ℝ) (h1 : 1 * 2 * 3 * a + 1 * 2 * 4 * a + 1 * 3 * 4 * a + 2 * 3 * 4 * a = -d)
  (h2 : 1 * 2 * 3 * 4 = e / a)
  (h3 : 1 * 2 * a + 1 * 3 * a + 1 * 4 * a + 2 * 3 * a + 2 * 4 * a + 3 * 4 * a = c) :
  c / e = 35 / 24 :=
by
  sorry

end NUMINAMATH_GPT_c_over_e_l861_86142


namespace NUMINAMATH_GPT_arith_general_formula_geom_general_formula_geom_sum_formula_l861_86102

-- Arithmetic Sequence Conditions
def arith_seq (a₈ a₁₀ : ℕ → ℝ) := a₈ = 6 ∧ a₁₀ = 0

-- General formula for arithmetic sequence
theorem arith_general_formula (a₁ : ℝ) (d : ℝ) (h₈ : 6 = a₁ + 7 * d) (h₁₀ : 0 = a₁ + 9 * d) :
  ∀ n : ℕ, aₙ = 30 - 3 * (n - 1) :=
sorry

-- General formula for geometric sequence
def geom_seq (a₁ a₄ : ℕ → ℝ) := a₁ = 1/2 ∧ a₄ = 4

theorem geom_general_formula (a₁ : ℝ) (q : ℝ) (h₁ : a₁ = 1 / 2) (h₄ : 4 = a₁ * q ^ 3) :
  ∀ n : ℕ, aₙ = 2^(n-2) :=
sorry

-- Sum of the first n terms of geometric sequence
theorem geom_sum_formula (a₁ : ℝ) (q : ℝ) (h₁ : a₁ = 1 / 2) (h₄ : 4 = a₁ * q ^ 3) :
  ∀ n : ℕ, Sₙ = 2^(n-1) - 1 / 2 :=
sorry

end NUMINAMATH_GPT_arith_general_formula_geom_general_formula_geom_sum_formula_l861_86102


namespace NUMINAMATH_GPT_opposite_of_one_sixth_l861_86198

theorem opposite_of_one_sixth : (-(1 / 6) : ℚ) = -1 / 6 := 
by
  sorry

end NUMINAMATH_GPT_opposite_of_one_sixth_l861_86198


namespace NUMINAMATH_GPT_min_sides_regular_polygon_l861_86167

/-- A regular polygon can accurately be placed back in its original position 
    when rotated by 50°.  Prove that the minimum number of sides the polygon 
    should have is 36. -/

theorem min_sides_regular_polygon (n : ℕ) (h : ∃ k : ℕ, 50 * k = 360 / n) : n = 36 :=
  sorry

end NUMINAMATH_GPT_min_sides_regular_polygon_l861_86167


namespace NUMINAMATH_GPT_project_completion_days_l861_86172

-- A's work rate per day
def A_work_rate : ℚ := 1 / 20

-- B's work rate per day
def B_work_rate : ℚ := 1 / 30

-- Combined work rate per day
def combined_work_rate : ℚ := A_work_rate + B_work_rate

-- Work done by B alone in the last 5 days
def B_alone_work : ℚ := 5 * B_work_rate

-- Let variable x represent the number of days A and B work together
def x (x_days : ℚ) := x_days / combined_work_rate + B_alone_work = 1

theorem project_completion_days (x_days : ℚ) (total_days : ℚ) :
  A_work_rate = 1 / 20 → B_work_rate = 1 / 30 → combined_work_rate = 1 / 12 → x_days / 12 + 1 / 6 = 1 → x_days = 10 → total_days = x_days + 5 → total_days = 15 :=
by
  intros _ _ _ _ _ _
  sorry

end NUMINAMATH_GPT_project_completion_days_l861_86172


namespace NUMINAMATH_GPT_simplify_and_evaluate_l861_86191

variable (x y : ℝ)

theorem simplify_and_evaluate (h : x / y = 3) : 
  (1 + y^2 / (x^2 - y^2)) * (x - y) / x = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l861_86191


namespace NUMINAMATH_GPT_room_width_l861_86169

theorem room_width (length : ℝ) (cost : ℝ) (rate : ℝ) (h_length : length = 5.5)
                    (h_cost : cost = 16500) (h_rate : rate = 800) : 
                    (cost / rate / length = 3.75) :=
by 
  sorry

end NUMINAMATH_GPT_room_width_l861_86169


namespace NUMINAMATH_GPT_opposite_of_a_is_2_l861_86136

theorem opposite_of_a_is_2 (a : ℤ) (h : -a = 2) : a = -2 := 
by
  -- proof to be provided
  sorry

end NUMINAMATH_GPT_opposite_of_a_is_2_l861_86136


namespace NUMINAMATH_GPT_rectangle_area_l861_86118

theorem rectangle_area (x : ℕ) (L W : ℕ) (h₁ : L * W = 864) (h₂ : L + W = 60) (h₃ : L = W + x) : 
  ((60 - x) / 2) * ((60 + x) / 2) = 864 :=
sorry

end NUMINAMATH_GPT_rectangle_area_l861_86118


namespace NUMINAMATH_GPT_division_of_cubics_l861_86192

theorem division_of_cubics (a b c : ℕ) (h_a : a = 7) (h_b : b = 6) (h_c : c = 1) :
  (a^3 + b^3) / (a^2 - a * b + b^2 + c) = 559 / 44 :=
by
  rw [h_a, h_b, h_c]
  -- After these substitutions, the problem is reduced to proving
  -- (7^3 + 6^3) / (7^2 - 7 * 6 + 6^2 + 1) = 559 / 44
  sorry

end NUMINAMATH_GPT_division_of_cubics_l861_86192


namespace NUMINAMATH_GPT_spider_eyes_solution_l861_86180

def spider_eyes_problem: Prop :=
  ∃ (x : ℕ), (3 * x) + (50 * 2) = 124 ∧ x = 8

theorem spider_eyes_solution : spider_eyes_problem :=
  sorry

end NUMINAMATH_GPT_spider_eyes_solution_l861_86180


namespace NUMINAMATH_GPT_child_l861_86156

noncomputable def child's_ticket_cost : ℕ :=
  let adult_ticket_price := 7
  let total_tickets := 900
  let total_revenue := 5100
  let childs_tickets_sold := 400
  let adult_tickets_sold := total_tickets - childs_tickets_sold
  let total_adult_revenue := adult_tickets_sold * adult_ticket_price
  let total_child_revenue := total_revenue - total_adult_revenue
  let child's_ticket_price := total_child_revenue / childs_tickets_sold
  child's_ticket_price

theorem child's_ticket_cost_is_4 : child's_ticket_cost = 4 :=
by
  have adult_ticket_price := 7
  have total_tickets := 900
  have total_revenue := 5100
  have childs_tickets_sold := 400
  have adult_tickets_sold := total_tickets - childs_tickets_sold
  have total_adult_revenue := adult_tickets_sold * adult_ticket_price
  have total_child_revenue := total_revenue - total_adult_revenue
  have child's_ticket_price := total_child_revenue / childs_tickets_sold
  show child's_ticket_cost = 4
  sorry

end NUMINAMATH_GPT_child_l861_86156


namespace NUMINAMATH_GPT_prove_x_eq_one_l861_86157

variables (x y : ℕ)

theorem prove_x_eq_one 
  (hx : x > 0) 
  (hy : y > 0) 
  (hdiv : ∀ n : ℕ, n > 0 → (2^n * y + 1) ∣ (x^2^n - 1)) : 
  x = 1 :=
sorry

end NUMINAMATH_GPT_prove_x_eq_one_l861_86157


namespace NUMINAMATH_GPT_evaluate_gg2_l861_86151

noncomputable def g (x : ℚ) : ℚ := 1 / (x^2) + (x^2) / (1 + x^2)

theorem evaluate_gg2 : g (g 2) = 530881 / 370881 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_gg2_l861_86151


namespace NUMINAMATH_GPT_stocks_higher_price_l861_86195

theorem stocks_higher_price
  (total_stocks : ℕ)
  (percent_increase : ℝ)
  (H L : ℝ)
  (H_eq : H = 1.35 * L)
  (sum_eq : H + L = 4200)
  (percent_increase_eq : percent_increase = 0.35)
  (total_stocks_eq : ↑total_stocks = 4200) :
  total_stocks = 2412 :=
by 
  sorry

end NUMINAMATH_GPT_stocks_higher_price_l861_86195


namespace NUMINAMATH_GPT_division_of_decimals_l861_86137

theorem division_of_decimals :
  (0.1 / 0.001 = 100) ∧ (1 / 0.01 = 100) := by
  sorry

end NUMINAMATH_GPT_division_of_decimals_l861_86137


namespace NUMINAMATH_GPT_inequality_solution_l861_86138

theorem inequality_solution (p : ℝ) (h1 : 18 * p < 10) (h2 : p > 0.5) : 0.5 < p ∧ p < (5 / 9) :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l861_86138


namespace NUMINAMATH_GPT_hyperbola_range_m_l861_86162

theorem hyperbola_range_m (m : ℝ) :
  (∃ x y : ℝ, (x^2 / (|m| - 1)) - (y^2 / (m - 2)) = 1) ↔ (m < -1) ∨ (m > 2) := 
by
  sorry

end NUMINAMATH_GPT_hyperbola_range_m_l861_86162


namespace NUMINAMATH_GPT_sum_of_three_numbers_l861_86190

theorem sum_of_three_numbers : ∃ (a b c : ℝ), a ≤ b ∧ b ≤ c ∧ b = 8 ∧ 
  (a + b + c) / 3 = a + 8 ∧ (a + b + c) / 3 = c - 20 ∧ a + b + c = 60 :=
sorry

end NUMINAMATH_GPT_sum_of_three_numbers_l861_86190


namespace NUMINAMATH_GPT_find_b_from_quadratic_l861_86123

theorem find_b_from_quadratic (b n : ℤ)
  (h1 : b > 0)
  (h2 : (x : ℤ) → (x + n)^2 - 6 = x^2 + b * x + 19) :
  b = 10 :=
sorry

end NUMINAMATH_GPT_find_b_from_quadratic_l861_86123


namespace NUMINAMATH_GPT_yoongi_flowers_left_l861_86146

theorem yoongi_flowers_left (initial_flowers given_to_eunji given_to_yuna : ℕ) 
  (h_initial : initial_flowers = 28) 
  (h_eunji : given_to_eunji = 7) 
  (h_yuna : given_to_yuna = 9) : 
  initial_flowers - (given_to_eunji + given_to_yuna) = 12 := 
by 
  sorry

end NUMINAMATH_GPT_yoongi_flowers_left_l861_86146


namespace NUMINAMATH_GPT_sector_area_l861_86116

theorem sector_area (radius area : ℝ) (θ : ℝ) (h1 : 2 * radius + θ * radius = 16) (h2 : θ = 2) : area = 16 :=
  sorry

end NUMINAMATH_GPT_sector_area_l861_86116


namespace NUMINAMATH_GPT_border_area_is_correct_l861_86115

def framed_area (height width border: ℝ) : ℝ :=
  (height + 2 * border) * (width + 2 * border)

def photograph_area (height width: ℝ) : ℝ :=
  height * width

theorem border_area_is_correct (h w b : ℝ) (h6 : h = 6) (w8 : w = 8) (b3 : b = 3) :
  (framed_area h w b - photograph_area h w) = 120 := by
  sorry

end NUMINAMATH_GPT_border_area_is_correct_l861_86115


namespace NUMINAMATH_GPT_points_on_curve_l861_86189

theorem points_on_curve (x y : ℝ) :
  (∃ p : ℝ, y = p^2 + (2 * p - 1) * x + 2 * x^2) ↔ y ≥ x^2 - x :=
by
  sorry

end NUMINAMATH_GPT_points_on_curve_l861_86189


namespace NUMINAMATH_GPT_bus_costs_unique_min_buses_cost_A_l861_86181

-- Defining the main conditions
def condition1 (x y : ℕ) : Prop := x + 2 * y = 300
def condition2 (x y : ℕ) : Prop := 2 * x + y = 270

-- Part 1: Proving individual bus costs
theorem bus_costs_unique (x y : ℕ) (h1 : condition1 x y) (h2 : condition2 x y) :
  x = 80 ∧ y = 110 := 
by 
  sorry

-- Part 2: Minimum buses of type A and total cost constraint
def total_buses := 10
def total_cost (x y a : ℕ) : Prop := 
  x * a + y * (total_buses - a) ≤ 1000

theorem min_buses_cost_A (x y : ℕ) (hx : x = 80) (hy : y = 110) :
  ∃ a cost, total_cost x y a ∧ a >= 4 ∧ cost = x * 4 + y * (total_buses - 4) ∧ cost = 980 :=
by
  sorry

end NUMINAMATH_GPT_bus_costs_unique_min_buses_cost_A_l861_86181


namespace NUMINAMATH_GPT_fg_of_one_eq_onehundredandfive_l861_86112

def f (x : ℝ) : ℝ := 4 * x - 3
def g (x : ℝ) : ℝ := (x + 2)^3

theorem fg_of_one_eq_onehundredandfive : f (g 1) = 105 :=
by
  -- proof would go here
  sorry

end NUMINAMATH_GPT_fg_of_one_eq_onehundredandfive_l861_86112


namespace NUMINAMATH_GPT_base_seven_representation_l861_86184

theorem base_seven_representation 
  (k : ℕ) 
  (h1 : 4 ≤ k) 
  (h2 : k < 8) 
  (h3 : 500 / k^3 < k) 
  (h4 : 500 ≥ k^3) 
  : ∃ n m o p : ℕ, (500 = n * k^3 + m * k^2 + o * k + p) ∧ (p % 2 = 1) ∧ (n ≠ 0 ) :=
sorry

end NUMINAMATH_GPT_base_seven_representation_l861_86184


namespace NUMINAMATH_GPT_schlaf_flachs_divisible_by_271_l861_86170

theorem schlaf_flachs_divisible_by_271 
(S C F H L A : ℕ) 
(hS : S ≠ 0) 
(hF : F ≠ 0) 
(hS_digit : S < 10)
(hC_digit : C < 10)
(hF_digit : F < 10)
(hH_digit : H < 10)
(hL_digit : L < 10)
(hA_digit : A < 10) :
  (100000 * S + 10000 * C + 1000 * H + 100 * L + 10 * A + F - 
   (100000 * F + 10000 * L + 1000 * A + 100 * C + 10 * H + S)) % 271 = 0 ↔ 
  C = L ∧ H = A := 
sorry

end NUMINAMATH_GPT_schlaf_flachs_divisible_by_271_l861_86170


namespace NUMINAMATH_GPT_max_ab_min_3x_4y_max_f_l861_86117

-- Proof Problem 1
theorem max_ab (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 4 * a + b = 1) : ab <= 1/16 :=
  sorry

-- Proof Problem 2
theorem min_3x_4y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 3 * y = 5 * x * y) : 3 * x + 4 * y >= 5 :=
  sorry

-- Proof Problem 3
theorem max_f (x : ℝ) (h1 : x < 5/4) : 4 * x - 2 + 1 / (4 * x - 5) <= 1 :=
  sorry

end NUMINAMATH_GPT_max_ab_min_3x_4y_max_f_l861_86117


namespace NUMINAMATH_GPT_find_amount_of_alcohol_l861_86134

theorem find_amount_of_alcohol (A W : ℝ) (h₁ : A / W = 4 / 3) (h₂ : A / (W + 7) = 4 / 5) : A = 14 := 
sorry

end NUMINAMATH_GPT_find_amount_of_alcohol_l861_86134


namespace NUMINAMATH_GPT_carly_cooks_in_72_minutes_l861_86143

def total_time_to_cook_burgers (total_guests : ℕ) (cook_time_per_side : ℕ) (burgers_per_grill : ℕ) : ℕ :=
  let guests_who_want_two_burgers := total_guests / 2
  let guests_who_want_one_burger := total_guests - guests_who_want_two_burgers
  let total_burgers := (guests_who_want_two_burgers * 2) + guests_who_want_one_burger
  let total_batches := (total_burgers + burgers_per_grill - 1) / burgers_per_grill  -- ceil division for total batches
  total_batches * (2 * cook_time_per_side)  -- total time

theorem carly_cooks_in_72_minutes : 
  total_time_to_cook_burgers 30 4 5 = 72 :=
by 
  sorry

end NUMINAMATH_GPT_carly_cooks_in_72_minutes_l861_86143


namespace NUMINAMATH_GPT_eventually_periodic_of_rational_cubic_l861_86194

noncomputable def is_rational_sequence (P : ℚ → ℚ) (q : ℕ → ℚ) :=
  ∀ n : ℕ, q (n + 1) = P (q n)

theorem eventually_periodic_of_rational_cubic (P : ℚ → ℚ) (q : ℕ → ℚ) (hP : ∃ a b c d : ℚ, ∀ x : ℚ, P x = a * x^3 + b * x^2 + c * x + d) (hq : is_rational_sequence P q) : 
  ∃ k ≥ 1, ∀ n ≥ 1, q (n + k) = q n := 
sorry

end NUMINAMATH_GPT_eventually_periodic_of_rational_cubic_l861_86194


namespace NUMINAMATH_GPT_complex_product_l861_86140

theorem complex_product : (3 + 4 * I) * (-2 - 3 * I) = -18 - 17 * I :=
by
  sorry

end NUMINAMATH_GPT_complex_product_l861_86140


namespace NUMINAMATH_GPT_games_per_season_l861_86148

-- Define the problem parameters
def total_goals : ℕ := 1244
def louie_last_match_goals : ℕ := 4
def louie_previous_goals : ℕ := 40
def louie_season_total_goals := louie_last_match_goals + louie_previous_goals
def brother_goals_per_game := 2 * louie_last_match_goals
def seasons : ℕ := 3

-- Prove the number of games in each season
theorem games_per_season : ∃ G : ℕ, louie_season_total_goals + (seasons * brother_goals_per_game * G) = total_goals ∧ G = 50 := 
by {
  sorry
}

end NUMINAMATH_GPT_games_per_season_l861_86148


namespace NUMINAMATH_GPT_arithmetic_sequence_solution_l861_86105

theorem arithmetic_sequence_solution (a : ℕ → ℝ) (d : ℝ) 
(h1 : d ≠ 0) 
(h2 : a 1 = 2) 
(h3 : a 1 * a 4 = (a 2) ^ 2) :
∀ n, a n = 2 * n :=
by 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_solution_l861_86105


namespace NUMINAMATH_GPT_initial_population_correct_l861_86144

-- Definitions based on conditions
def initial_population (P : ℝ) := P
def population_after_bombardment (P : ℝ) := 0.9 * P
def population_after_fear (P : ℝ) := 0.8 * (population_after_bombardment P)
def final_population := 3240

-- Theorem statement
theorem initial_population_correct (P : ℝ) (h : population_after_fear P = final_population) :
  initial_population P = 4500 :=
sorry

end NUMINAMATH_GPT_initial_population_correct_l861_86144


namespace NUMINAMATH_GPT_odd_square_mod_eight_l861_86133

theorem odd_square_mod_eight (k : ℤ) : ((2 * k + 1) ^ 2) % 8 = 1 := 
sorry

end NUMINAMATH_GPT_odd_square_mod_eight_l861_86133


namespace NUMINAMATH_GPT_cost_price_of_article_l861_86125

theorem cost_price_of_article (x : ℝ) :
  (86 - x = x - 42) → x = 64 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_cost_price_of_article_l861_86125


namespace NUMINAMATH_GPT_book_cost_l861_86141

theorem book_cost (C_1 C_2 : ℝ)
  (h1 : C_1 + C_2 = 420)
  (h2 : C_1 * 0.85 = C_2 * 1.19) :
  C_1 = 245 :=
by
  -- We skip the proof here using sorry.
  sorry

end NUMINAMATH_GPT_book_cost_l861_86141


namespace NUMINAMATH_GPT_spring_problem_l861_86158

theorem spring_problem (x y : ℝ) : 
  (∀ x, y = 0.5 * x + 12) →
  (0.5 * 3 + 12 = 13.5) ∧
  (y = 0.5 * x + 12) ∧
  (0.5 * 5.5 + 12 = 14.75) ∧
  (20 = 0.5 * 16 + 12) :=
by 
  sorry

end NUMINAMATH_GPT_spring_problem_l861_86158


namespace NUMINAMATH_GPT_sine_five_l861_86101

noncomputable def sine_value (x : ℝ) : ℝ :=
  Real.sin (5 * x)

theorem sine_five : sine_value 1 = -0.959 := 
  by
  sorry

end NUMINAMATH_GPT_sine_five_l861_86101


namespace NUMINAMATH_GPT_positive_diff_between_two_numbers_l861_86111

theorem positive_diff_between_two_numbers (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 20) :  |x - y| = 2 := 
by
  sorry

end NUMINAMATH_GPT_positive_diff_between_two_numbers_l861_86111


namespace NUMINAMATH_GPT_labor_productivity_increase_l861_86153

noncomputable def regression_equation (x : ℝ) : ℝ := 50 + 60 * x

theorem labor_productivity_increase (Δx : ℝ) (hx : Δx = 1) :
  regression_equation (x + Δx) - regression_equation x = 60 :=
by
  sorry

end NUMINAMATH_GPT_labor_productivity_increase_l861_86153


namespace NUMINAMATH_GPT_single_cakes_needed_l861_86131

theorem single_cakes_needed :
  ∀ (layer_cake_frosting single_cake_frosting cupcakes_frosting brownies_frosting : ℝ)
  (layer_cakes cupcakes brownies total_frosting : ℕ)
  (single_cakes_needed : ℝ),
  layer_cake_frosting = 1 →
  single_cake_frosting = 0.5 →
  cupcakes_frosting = 0.5 →
  brownies_frosting = 0.5 →
  layer_cakes = 3 →
  cupcakes = 6 →
  brownies = 18 →
  total_frosting = 21 →
  single_cakes_needed = (total_frosting - (layer_cakes * layer_cake_frosting + cupcakes * cupcakes_frosting + brownies * brownies_frosting)) / single_cake_frosting →
  single_cakes_needed = 12 :=
by
  intros
  sorry

end NUMINAMATH_GPT_single_cakes_needed_l861_86131


namespace NUMINAMATH_GPT_evaluate_expression_at_4_l861_86173

theorem evaluate_expression_at_4 :
  ∀ x : ℝ, x = 4 → (x^2 - 3 * x - 10) / (x - 5) = 6 :=
by
  intro x
  intro hx
  sorry

end NUMINAMATH_GPT_evaluate_expression_at_4_l861_86173


namespace NUMINAMATH_GPT_songs_in_each_album_l861_86171

variable (X : ℕ)

theorem songs_in_each_album (h : 6 * X + 2 * X = 72) : X = 9 :=
by sorry

end NUMINAMATH_GPT_songs_in_each_album_l861_86171


namespace NUMINAMATH_GPT_job_completion_in_time_l861_86110

theorem job_completion_in_time (t_total t_1 w_1 : ℕ) (work_done : ℚ) (h : (t_total = 30) ∧ (t_1 = 6) ∧ (w_1 = 8) ∧ (work_done = 1/3)) :
  ∃ w : ℕ, w = 4 ∧ (t_total - t_1) * w_1 / t_1 * (1 / work_done) / w = 3 :=
by
  sorry

end NUMINAMATH_GPT_job_completion_in_time_l861_86110


namespace NUMINAMATH_GPT_ratio_height_radius_l861_86182

variable (V r h : ℝ)

theorem ratio_height_radius (h_eq_2r : h = 2 * r) (volume_eq : π * r^2 * h = V) : h / r = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_height_radius_l861_86182


namespace NUMINAMATH_GPT_log_problem_l861_86104

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_problem :
  let x := (log_base 8 2) ^ (log_base 2 8)
  log_base 3 x = -3 :=
by
  sorry

end NUMINAMATH_GPT_log_problem_l861_86104


namespace NUMINAMATH_GPT_daily_profit_35_selling_price_for_600_profit_no_900_profit_possible_l861_86165

-- Definitions based on conditions
def purchase_price : ℝ := 30
def max_selling_price : ℝ := 55
def linear_relationship (x : ℝ) : ℝ := -2 * x + 140
def profit (x : ℝ) : ℝ := (x - purchase_price) * linear_relationship x

-- Part 1: Daily profit when selling price is 35 yuan
theorem daily_profit_35 : profit 35 = 350 :=
  sorry

-- Part 2: Selling price for a daily profit of 600 yuan
theorem selling_price_for_600_profit (x : ℝ) (h1 : 30 ≤ x) (h2 : x ≤ 55) : profit x = 600 → x = 40 :=
  sorry

-- Part 3: Possibility of daily profit of 900 yuan
theorem no_900_profit_possible (h1 : ∀ x, 30 ≤ x ∧ x ≤ 55 → profit x ≠ 900) : ¬ ∃ x, 30 ≤ x ∧ x ≤ 55 ∧ profit x = 900 :=
  sorry

end NUMINAMATH_GPT_daily_profit_35_selling_price_for_600_profit_no_900_profit_possible_l861_86165


namespace NUMINAMATH_GPT_secretaries_ratio_l861_86124

theorem secretaries_ratio (A B C : ℝ) (hA: A = 75) (h_total: A + B + C = 120) : B + C = 45 :=
by {
  -- sorry: We define this part to be explored by the theorem prover
  sorry
}

end NUMINAMATH_GPT_secretaries_ratio_l861_86124


namespace NUMINAMATH_GPT_calculate_expression_l861_86199

theorem calculate_expression (b : ℝ) (hb : b ≠ 0) : 
  (1 / 25) * b^0 + (1 / (25 * b))^0 - 81^(-1 / 4 : ℝ) - (-27)^(-1 / 3 : ℝ) = 26 / 25 :=
by sorry

end NUMINAMATH_GPT_calculate_expression_l861_86199


namespace NUMINAMATH_GPT_quadratic_roots_ratio_l861_86196

noncomputable def value_of_m (m : ℚ) : Prop :=
  ∃ r s : ℚ, r ≠ 0 ∧ s ≠ 0 ∧ (r / s = 3) ∧ (r + s = -9) ∧ (r * s = m)

theorem quadratic_roots_ratio (m : ℚ) (h : value_of_m m) : m = 243 / 16 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_ratio_l861_86196


namespace NUMINAMATH_GPT_find_k_shelf_life_at_11_22_l861_86178

noncomputable def food_shelf_life (k b x : ℝ) : ℝ := Real.exp (k * x + b)

-- Given conditions
def condition1 : food_shelf_life k b 0 = 192 := by sorry
def condition2 : food_shelf_life k b 33 = 24 := by sorry

-- Prove that k = - (Real.log 2) / 11
theorem find_k (k b : ℝ) (h1 : food_shelf_life k b 0 = 192) (h2 : food_shelf_life k b 33 = 24) : 
  k = - (Real.log 2) / 11 :=
by sorry

-- Use the found value of k to determine the shelf life at 11°C and 22°C
theorem shelf_life_at_11_22 (k b : ℝ) (h1 : food_shelf_life k b 0 = 192) (h2 : food_shelf_life k b 33 = 24) 
  (hk : k = - (Real.log 2) / 11) : 
  food_shelf_life k b 11 = 96 ∧ food_shelf_life k b 22 = 48 :=
by sorry

end NUMINAMATH_GPT_find_k_shelf_life_at_11_22_l861_86178


namespace NUMINAMATH_GPT_trapezoid_bd_length_l861_86154

theorem trapezoid_bd_length
  (AB CD AC BD : ℝ)
  (tanC tanB : ℝ)
  (h1 : AB = 24)
  (h2 : CD = 15)
  (h3 : AC = 30)
  (h4 : tanC = 2)
  (h5 : tanB = 1.25)
  (h6 : AC ^ 2 = AB ^ 2 + (CD - AB) ^ 2) :
  BD = 9 * Real.sqrt 11 := by
  sorry

end NUMINAMATH_GPT_trapezoid_bd_length_l861_86154


namespace NUMINAMATH_GPT_find_f_sum_l861_86100

noncomputable def f : ℝ → ℝ := sorry

axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom functional_eq : ∀ x : ℝ, f (2 + x) + f (2 - x) = 0
axiom f_at_one : f 1 = 9

theorem find_f_sum :
  f 2010 + f 2011 + f 2012 = -9 :=
sorry

end NUMINAMATH_GPT_find_f_sum_l861_86100


namespace NUMINAMATH_GPT_same_grades_percentage_l861_86114

theorem same_grades_percentage (total_students same_grades_A same_grades_B same_grades_C same_grades_D : ℕ) 
  (total_eq : total_students = 50) 
  (same_A : same_grades_A = 3) 
  (same_B : same_grades_B = 6) 
  (same_C : same_grades_C = 7) 
  (same_D : same_grades_D = 2) : 
  (same_grades_A + same_grades_B + same_grades_C + same_grades_D) * 100 / total_students = 36 := 
by
  sorry

end NUMINAMATH_GPT_same_grades_percentage_l861_86114


namespace NUMINAMATH_GPT_time_addition_sum_l861_86103

theorem time_addition_sum (A B C : ℕ) (h1 : A = 7) (h2 : B = 59) (h3 : C = 59) : A + B + C = 125 :=
sorry

end NUMINAMATH_GPT_time_addition_sum_l861_86103


namespace NUMINAMATH_GPT_intersection_a_eq_1_parallel_lines_value_of_a_l861_86185

-- Define lines
def line1 (a : ℝ) (x y : ℝ) : Prop := x + a * y - a + 2 = 0
def line2 (a : ℝ) (x y : ℝ) : Prop := 2 * a * x + (a + 3) * y + a - 5 = 0

-- Part 1: Prove intersection point for a = 1
theorem intersection_a_eq_1 :
  line1 1 (-4) 3 ∧ line2 1 (-4) 3 :=
by sorry

-- Part 2: Prove value of a for which lines are parallel
theorem parallel_lines_value_of_a :
  ∃ a : ℝ, ∀ x y : ℝ, line1 a x y ∧ line2 a x y →
  (2 * a^2 - a - 3 = 0 ∧ a ≠ -1 ∧ a = 3/2) :=
by sorry

end NUMINAMATH_GPT_intersection_a_eq_1_parallel_lines_value_of_a_l861_86185


namespace NUMINAMATH_GPT_faucets_fill_time_l861_86132

theorem faucets_fill_time (fill_time_4faucets_200gallons_12min : 4 * 12 * faucet_rate = 200) 
    (fill_time_m_50gallons_seconds : ∃ (rate: ℚ), 8 * t_to_seconds * rate = 50) : 
    8 * t_to_seconds / 33.33 = 90 :=
by sorry


end NUMINAMATH_GPT_faucets_fill_time_l861_86132


namespace NUMINAMATH_GPT_athlete_a_catches_up_and_race_duration_l861_86108

-- Track is 1000 meters
def track_length : ℕ := 1000

-- Athlete A's speed: first minute, increasing until 5th minute and decreasing until 600 meters/min
def athlete_A_speed (minute : ℕ) : ℕ :=
  match minute with
  | 0 => 1000
  | 1 => 1000
  | 2 => 1200
  | 3 => 1400
  | 4 => 1600
  | 5 => 1400
  | 6 => 1200
  | 7 => 1000
  | 8 => 800
  | 9 => 600
  | _ => 600

-- Athlete B's constant speed
def athlete_B_speed : ℕ := 1200

-- Function to compute distance covered in given minutes, assuming starts at 0
def total_distance (speed : ℕ → ℕ) (minutes : ℕ) : ℕ :=
  (List.range minutes).map speed |>.sum

-- Defining the maximum speed moment for A
def athlete_A_max_speed_distance : ℕ := total_distance athlete_A_speed 4
def athlete_B_max_speed_distance : ℕ := athlete_B_speed * 4

-- Proof calculation for target time 10 2/3 minutes
def time_catch : ℚ := 10 + 2 / 3

-- Defining the theorem to be proven
theorem athlete_a_catches_up_and_race_duration :
  athlete_A_max_speed_distance > athlete_B_max_speed_distance ∧ time_catch = 32 / 3 :=
by
  -- Place holder for the proof's details
  sorry

end NUMINAMATH_GPT_athlete_a_catches_up_and_race_duration_l861_86108


namespace NUMINAMATH_GPT_geometric_sequence_product_l861_86149

theorem geometric_sequence_product
  (a : ℕ → ℝ)
  (h1 : a 1 * a 3 * a 11 = 8) :
  a 2 * a 8 = 4 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_product_l861_86149


namespace NUMINAMATH_GPT_am_hm_inequality_l861_86127

noncomputable def smallest_possible_value (a b c d : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) : ℝ :=
  (a + b + c) * ((1 / (a + b + d)) + (1 / (a + c + d)) + (1 / (b + c + d)))

theorem am_hm_inequality (a b c d : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) :
  smallest_possible_value a b c d h1 h2 h3 h4 ≥ 9 / 2 :=
by
  sorry

end NUMINAMATH_GPT_am_hm_inequality_l861_86127


namespace NUMINAMATH_GPT_probability_sum_8_9_10_l861_86188

/-- The faces of the first die -/
def first_die := [2, 2, 3, 3, 5, 5]

/-- The faces of the second die -/
def second_die := [1, 3, 4, 5, 6, 7]

/-- Predicate that checks if the sum of two numbers is either 8, 9, or 10 -/
def valid_sum (a b : ℕ) : Prop := a + b = 8 ∨ a + b = 9 ∨ a + b = 10

/-- Calculate the probability of a sum being 8, 9, or 10 according to the given dice setup -/
def calc_probability : ℚ := 
  let valid_pairs := [(2, 6), (2, 7), (3, 5), (3, 6), (3, 7), (5, 3), (5, 4), (5, 5)] 
  (valid_pairs.length : ℚ) / (first_die.length * second_die.length : ℚ)

theorem probability_sum_8_9_10 : calc_probability = 4 / 9 :=
by
  sorry

end NUMINAMATH_GPT_probability_sum_8_9_10_l861_86188


namespace NUMINAMATH_GPT_fraction_value_l861_86106

theorem fraction_value : (5 * 7) / 10.0 = 3.5 := by
  sorry

end NUMINAMATH_GPT_fraction_value_l861_86106


namespace NUMINAMATH_GPT_regular_dinosaur_weight_l861_86193

namespace DinosaurWeight

-- Given Conditions
def Barney_weight (x : ℝ) : ℝ := 5 * x + 1500
def combined_weight (x : ℝ) : ℝ := Barney_weight x + 5 * x

-- Target Proof
theorem regular_dinosaur_weight :
  (∃ x : ℝ, combined_weight x = 9500) -> 
  ∃ x : ℝ, x = 800 :=
by {
  sorry
}

end DinosaurWeight

end NUMINAMATH_GPT_regular_dinosaur_weight_l861_86193


namespace NUMINAMATH_GPT_find_number_of_adults_l861_86152

variable (A : ℕ) -- Variable representing the number of adults.
def C : ℕ := 5  -- Number of children.

def meal_cost : ℕ := 3  -- Cost per meal in dollars.
def total_cost (A : ℕ) : ℕ := (A + C) * meal_cost  -- Total cost formula.

theorem find_number_of_adults 
  (h1 : meal_cost = 3)
  (h2 : total_cost A = 21)
  (h3 : C = 5) :
  A = 2 :=
sorry

end NUMINAMATH_GPT_find_number_of_adults_l861_86152


namespace NUMINAMATH_GPT_quadratic_radicals_x_le_10_l861_86130

theorem quadratic_radicals_x_le_10 (a x : ℝ) (h1 : 3 * a - 8 = 17 - 2 * a) (h2 : 4 * a - 2 * x ≥ 0) : x ≤ 10 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_radicals_x_le_10_l861_86130


namespace NUMINAMATH_GPT_solution_of_system_of_equations_l861_86176

-- Define the conditions of the problem.
def system_of_equations (x y : ℝ) : Prop :=
  (x + y = 6) ∧ (x = 2 * y)

-- Define the correct answer as a set.
def solution_set : Set (ℝ × ℝ) :=
  { (4, 2) }

-- State the proof problem.
theorem solution_of_system_of_equations : 
  {p : ℝ × ℝ | ∃ x y, p = (x, y) ∧ system_of_equations x y} = solution_set :=
  sorry

end NUMINAMATH_GPT_solution_of_system_of_equations_l861_86176


namespace NUMINAMATH_GPT_volume_ratio_sum_is_26_l861_86166

noncomputable def volume_of_dodecahedron (s : ℝ) : ℝ :=
  (15 + 7 * Real.sqrt 5) * s ^ 3 / 4

noncomputable def volume_of_cube (s : ℝ) : ℝ :=
  s ^ 3

noncomputable def volume_ratio_sum (s : ℝ) : ℝ :=
  let ratio := (volume_of_dodecahedron s) / (volume_of_cube s)
  let numerator := 15 + 7 * Real.sqrt 5
  let denominator := 4
  numerator + denominator

theorem volume_ratio_sum_is_26 (s : ℝ) : volume_ratio_sum s = 26 := by
  sorry

end NUMINAMATH_GPT_volume_ratio_sum_is_26_l861_86166


namespace NUMINAMATH_GPT_distance_traveled_l861_86119

theorem distance_traveled :
  ∫ t in (3:ℝ)..(5:ℝ), (2 * t + 3 : ℝ) = 22 :=
by
  sorry

end NUMINAMATH_GPT_distance_traveled_l861_86119


namespace NUMINAMATH_GPT_symmetric_point_P_l861_86109

-- Define the point P
def P : ℝ × ℝ := (1, -2)

-- Define the function to get the symmetric point with respect to the origin
def symmetric_point (point : ℝ × ℝ) : ℝ × ℝ :=
  (-point.1, -point.2)

-- State the theorem that proves the symmetric point of P is (-1, 2)
theorem symmetric_point_P :
  symmetric_point P = (-1, 2) :=
  sorry

end NUMINAMATH_GPT_symmetric_point_P_l861_86109


namespace NUMINAMATH_GPT_apples_in_each_basket_l861_86122

theorem apples_in_each_basket (total_apples : ℕ) (baskets : ℕ) (apples_per_basket : ℕ) 
  (h1 : total_apples = 495) 
  (h2 : baskets = 19) 
  (h3 : apples_per_basket = total_apples / baskets) : 
  apples_per_basket = 26 :=
by 
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_apples_in_each_basket_l861_86122


namespace NUMINAMATH_GPT_distance_from_diagonal_intersection_to_base_l861_86163

theorem distance_from_diagonal_intersection_to_base (AD BC AB R : ℝ) (O : ℝ → Prop) (M N Q : ℝ) :
  (AD + BC + 2 * AB = 8) ∧
  (AD + BC) = 4 ∧
  (R = 1 / 2) ∧
  (2 = R * (AD + BC) / 2) ∧
  (BC = AD + 2 * AB) ∧
  (∀ x, x * (2 - x) = (1 / 2) ^ 2)  →
  (Q = (2 - Real.sqrt 3) / 4) :=
by
  intros
  sorry

end NUMINAMATH_GPT_distance_from_diagonal_intersection_to_base_l861_86163


namespace NUMINAMATH_GPT_percentage_decrease_l861_86174

-- Define conditions and variables
def original_selling_price : ℝ := 659.9999999999994
def profit_rate1 : ℝ := 0.10
def increase_in_selling_price : ℝ := 42
def profit_rate2 : ℝ := 0.30

-- Define the actual proof problem
theorem percentage_decrease (C C_prime : ℝ) 
    (h1 : 1.10 * C = original_selling_price) 
    (h2 : 1.30 * C_prime = original_selling_price + increase_in_selling_price) : 
    ((C - C_prime) / C) * 100 = 10 := 
sorry

end NUMINAMATH_GPT_percentage_decrease_l861_86174


namespace NUMINAMATH_GPT_sum_of_decimals_l861_86159

theorem sum_of_decimals :
  0.3 + 0.04 + 0.005 + 0.0006 + 0.00007 = (34567 / 100000 : ℚ) :=
by
  -- The proof details would go here
  sorry

end NUMINAMATH_GPT_sum_of_decimals_l861_86159


namespace NUMINAMATH_GPT_simplify_f_value_f_second_quadrant_l861_86126

noncomputable def f (α : ℝ) : ℝ := 
  (Real.sin (3 * Real.pi - α) * Real.cos (2 * Real.pi - α) * Real.sin (3 * Real.pi / 2 - α)) / 
  (Real.cos (Real.pi - α) * Real.sin (-Real.pi - α))

theorem simplify_f (α : ℝ) : 
  f α = Real.cos α := 
sorry

theorem value_f_second_quadrant (α : ℝ) (hα : π / 2 < α ∧ α < π) (hcosα : Real.cos (π / 2 + α) = -1 / 3) :
  f α = - (2 * Real.sqrt 2) / 3 := 
sorry

end NUMINAMATH_GPT_simplify_f_value_f_second_quadrant_l861_86126


namespace NUMINAMATH_GPT_chocolate_cost_proof_l861_86139

/-- The initial amount of money Dan has. -/
def initial_amount : ℕ := 7

/-- The cost of the candy bar. -/
def candy_bar_cost : ℕ := 2

/-- The remaining amount of money Dan has after the purchases. -/
def remaining_amount : ℕ := 2

/-- The cost of the chocolate. -/
def chocolate_cost : ℕ := initial_amount - candy_bar_cost - remaining_amount

/-- Expected cost of the chocolate. -/
def expected_chocolate_cost : ℕ := 3

/-- Prove that the cost of the chocolate equals the expected cost. -/
theorem chocolate_cost_proof : chocolate_cost = expected_chocolate_cost :=
by
  sorry

end NUMINAMATH_GPT_chocolate_cost_proof_l861_86139


namespace NUMINAMATH_GPT_rachel_math_homework_l861_86129

/-- Rachel had to complete some pages of math homework. 
Given:
- 4 more pages of math homework than reading homework
- 3 pages of reading homework
Prove that Rachel had to complete 7 pages of math homework.
--/
theorem rachel_math_homework
  (r : ℕ) (h_r : r = 3)
  (m : ℕ) (h_m : m = r + 4) :
  m = 7 := by
  sorry

end NUMINAMATH_GPT_rachel_math_homework_l861_86129


namespace NUMINAMATH_GPT_correct_option_l861_86107

-- Definitions
def option_A (a : ℕ) : Prop := a^2 * a^3 = a^5
def option_B (a : ℕ) : Prop := a^6 / a^2 = a^3
def option_C (a b : ℕ) : Prop := (a * b^3) ^ 2 = a^2 * b^9
def option_D (a : ℕ) : Prop := 5 * a - 2 * a = 3

-- Theorem statement
theorem correct_option :
  (∃ (a : ℕ), option_A a) ∧
  (∀ (a : ℕ), ¬option_B a) ∧
  (∀ (a b : ℕ), ¬option_C a b) ∧
  (∀ (a : ℕ), ¬option_D a) :=
by
  sorry

end NUMINAMATH_GPT_correct_option_l861_86107


namespace NUMINAMATH_GPT_relationship_even_increasing_l861_86160

-- Even function definition
def even_function (f : ℝ → ℝ) := ∀ x, f (-x) = f x

-- Monotonically increasing function definition on interval
def increasing_on (f : ℝ → ℝ) (a b : ℝ) := ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

variable {f : ℝ → ℝ}

-- The proof problem statement
theorem relationship_even_increasing (h_even : even_function f) (h_increasing : increasing_on f 0 1) :
  f 0 < f (-0.5) ∧ f (-0.5) < f (-1) :=
by
  sorry

end NUMINAMATH_GPT_relationship_even_increasing_l861_86160


namespace NUMINAMATH_GPT_difference_of_squares_l861_86187

theorem difference_of_squares (x y : ℝ) (h1 : (x + y)^2 = 64) (h2 : x * y = 15) : (x - y)^2 = 4 :=
by
  sorry

end NUMINAMATH_GPT_difference_of_squares_l861_86187


namespace NUMINAMATH_GPT_couple_tickets_sold_l861_86147

theorem couple_tickets_sold (S C : ℕ) :
  20 * S + 35 * C = 2280 ∧ S + 2 * C = 128 -> C = 56 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_couple_tickets_sold_l861_86147


namespace NUMINAMATH_GPT_smallest_n_property_l861_86197

theorem smallest_n_property (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
 (hxy : x ∣ y^3) (hyz : y ∣ z^3) (hzx : z ∣ x^3) : 
  x * y * z ∣ (x + y + z) ^ 13 := 
by sorry

end NUMINAMATH_GPT_smallest_n_property_l861_86197


namespace NUMINAMATH_GPT_minimize_S_n_l861_86120

noncomputable def S_n (n : ℕ) : ℝ := 2 * (n : ℝ) ^ 2 - 30 * (n : ℝ)

theorem minimize_S_n :
  ∃ n : ℕ, S_n n = 2 * (7 : ℝ) ^ 2 - 30 * (7 : ℝ) ∨ S_n n = 2 * (8 : ℝ) ^ 2 - 30 * (8 : ℝ) := by
  sorry

end NUMINAMATH_GPT_minimize_S_n_l861_86120


namespace NUMINAMATH_GPT_stock_price_end_of_third_year_l861_86183

def stock_price_after_years (initial_price : ℝ) (year1_increase : ℝ) (year2_decrease : ℝ) (year3_increase : ℝ) : ℝ :=
  let price_after_year1 := initial_price * (1 + year1_increase)
  let price_after_year2 := price_after_year1 * (1 - year2_decrease)
  let price_after_year3 := price_after_year2 * (1 + year3_increase)
  price_after_year3

theorem stock_price_end_of_third_year :
  stock_price_after_years 120 0.80 0.30 0.50 = 226.8 := 
by
  sorry

end NUMINAMATH_GPT_stock_price_end_of_third_year_l861_86183


namespace NUMINAMATH_GPT_range_of_a_l861_86168

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x^2 - a * x - a ≤ -3) ↔ (a ≤ -6 ∨ a ≥ 2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l861_86168


namespace NUMINAMATH_GPT_arithmetic_sequence_ratio_l861_86145

noncomputable def sum_first_n_terms (a₁ d : ℚ) (n : ℕ) : ℚ :=
  n * a₁ + (n * (n - 1) / 2) * d

theorem arithmetic_sequence_ratio (d : ℚ) (h : d ≠ 0) :
  let a₁ := 8 * d
  let S₅ := sum_first_n_terms a₁ d 5
  let S₇ := sum_first_n_terms a₁ d 7
  (7 * S₅) / (5 * S₇) = 10 / 11 :=
by 
  let a₁ := 8 * d
  let S₅ := sum_first_n_terms a₁ d 5
  let S₇ := sum_first_n_terms a₁ d 7
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_ratio_l861_86145


namespace NUMINAMATH_GPT_is_quadratic_equation_l861_86186

open Real

-- Define the candidate equations as statements in Lean 4
def equation_A (x : ℝ) : Prop := 3 * x^2 = 1 - 1 / (3 * x)
def equation_B (x m : ℝ) : Prop := (m - 2) * x^2 - m * x + 3 = 0
def equation_C (x : ℝ) : Prop := (x^2 - 3) * (x - 1) = 0
def equation_D (x : ℝ) : Prop := x^2 = 2

-- Prove that among the given equations, equation_D is the only quadratic equation
theorem is_quadratic_equation (x : ℝ) :
  (∃ a b c : ℝ, a ≠ 0 ∧ equation_A x = (a * x^2 + b * x + c = 0)) ∨
  (∃ m a b c : ℝ, a ≠ 0 ∧ equation_B x m = (a * x^2 + b * x + c = 0)) ∨
  (∃ a b c : ℝ, a ≠ 0 ∧ equation_C x = (a * x^2 + b * x + c = 0)) ∨
  (∃ a b c : ℝ, a ≠ 0 ∧ equation_D x = (a * x^2 + b * x + c = 0)) := by
  sorry

end NUMINAMATH_GPT_is_quadratic_equation_l861_86186


namespace NUMINAMATH_GPT_number_of_sets_given_to_sister_l861_86179

-- Defining the total number of cards, sets given to his brother and friend, total cards given away,
-- number of cards per set, and expected answer for sets given to his sister.
def total_cards := 365
def sets_given_to_brother := 8
def sets_given_to_friend := 2
def total_cards_given_away := 195
def cards_per_set := 13
def sets_given_to_sister := 5

theorem number_of_sets_given_to_sister :
  sets_given_to_brother * cards_per_set + 
  sets_given_to_friend * cards_per_set + 
  sets_given_to_sister * cards_per_set = total_cards_given_away :=
by
  -- It skips the proof but ensures the statement is set up correctly.
  sorry

end NUMINAMATH_GPT_number_of_sets_given_to_sister_l861_86179


namespace NUMINAMATH_GPT_classroom_has_total_books_l861_86113

-- Definitions for the conditions
def num_children : Nat := 10
def books_per_child : Nat := 7
def additional_books : Nat := 8

-- Total number of books the children have
def total_books_from_children : Nat := num_children * books_per_child

-- The expected total number of books in the classroom
def total_books : Nat := total_books_from_children + additional_books

-- The main theorem to be proven
theorem classroom_has_total_books : total_books = 78 :=
by
  sorry

end NUMINAMATH_GPT_classroom_has_total_books_l861_86113


namespace NUMINAMATH_GPT_natural_number_base_conversion_l861_86175

theorem natural_number_base_conversion (n : ℕ) (h7 : n = 4 * 7 + 1) (h9 : n = 3 * 9 + 2) : 
  n = 3 * 8 + 5 := 
by 
  sorry

end NUMINAMATH_GPT_natural_number_base_conversion_l861_86175


namespace NUMINAMATH_GPT_absolute_inequality_solution_l861_86155

theorem absolute_inequality_solution (x : ℝ) (hx : x > 0) :
  |5 - 2 * x| ≤ 8 ↔ 0 ≤ x ∧ x ≤ 6.5 :=
by sorry

end NUMINAMATH_GPT_absolute_inequality_solution_l861_86155


namespace NUMINAMATH_GPT_average_correct_l861_86164

theorem average_correct :
  (12 + 13 + 14 + 510 + 520 + 530 + 1115 + 1120 + 1252140 + 2345) / 10 = 125831.9 := 
sorry

end NUMINAMATH_GPT_average_correct_l861_86164


namespace NUMINAMATH_GPT_num_lineups_l861_86161

-- Define the given conditions
def num_players : ℕ := 12
def num_lineman : ℕ := 4
def num_qb_among_lineman : ℕ := 2
def num_running_backs : ℕ := 3

-- State the problem and the result as a theorem
theorem num_lineups : 
  (num_lineman * (num_qb_among_lineman) * (num_running_backs) * (num_players - num_lineman - num_qb_among_lineman - num_running_backs + 3) = 216) := 
by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_num_lineups_l861_86161
