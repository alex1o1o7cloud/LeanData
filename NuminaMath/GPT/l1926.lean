import Mathlib

namespace find_x_coordinate_l1926_192669

open Real

noncomputable def point_on_parabola (x y : ℝ) : Prop :=
  y^2 = 6 * x ∧ x > 0 

noncomputable def is_twice_distance (x : ℝ) : Prop :=
  let focus_x : ℝ := 3 / 2
  let d1 := x + focus_x
  let d2 := x
  d1 = 2 * d2

theorem find_x_coordinate (x y : ℝ) :
  point_on_parabola x y →
  is_twice_distance x →
  x = 3 / 2 :=
by
  intros
  sorry

end find_x_coordinate_l1926_192669


namespace min_passengers_on_vehicle_with_no_adjacent_seats_l1926_192689

-- Define the seating arrangement and adjacency rules

structure Seat :=
(row : Fin 2) (col : Fin 5)

def adjacent (a b : Seat) : Prop :=
(a.row = b.row ∧ (a.col = b.col + 1 ∨ a.col + 1 = b.col)) ∨
(a.col = b.col ∧ (a.row = b.row + 1 ∨ a.row + 1 = b.row))

def valid_seating (seated : List Seat) : Prop :=
∀ (i j : Seat), i ∈ seated → j ∈ seated → adjacent i j → false

def min_passengers : ℕ :=
5

theorem min_passengers_on_vehicle_with_no_adjacent_seats :
∃ seated : List Seat, valid_seating seated ∧ List.length seated = min_passengers :=
sorry

end min_passengers_on_vehicle_with_no_adjacent_seats_l1926_192689


namespace temperature_lower_than_freezing_point_is_minus_three_l1926_192682

-- Define the freezing point of water
def freezing_point := 0 -- in degrees Celsius

-- Define the temperature lower by a certain value
def lower_temperature (t: Int) (delta: Int) := t - delta

-- State the theorem to be proved
theorem temperature_lower_than_freezing_point_is_minus_three:
  lower_temperature freezing_point 3 = -3 := by
  sorry

end temperature_lower_than_freezing_point_is_minus_three_l1926_192682


namespace percentage_difference_l1926_192685

variable (x y : ℝ)
variable (p : ℝ)  -- percentage by which x is less than y

theorem percentage_difference (h1 : y = x * 1.3333333333333333) : p = 25 :=
by
  sorry

end percentage_difference_l1926_192685


namespace find_y_l1926_192678

-- Definitions based on conditions
variables (x y : ℝ)
def inversely_proportional (x y : ℝ) : Prop := ∃ k : ℝ, x * y = k

-- Lean statement capturing the problem
theorem find_y
  (h1 : inversely_proportional x y)
  (h2 : x + y = 60)
  (h3 : x = 3 * y)
  (h4 : x = -12) :
  y = -56.25 :=
sorry  -- Proof omitted

end find_y_l1926_192678


namespace ratio_black_white_l1926_192692

-- Definitions of the parameters
variables (B W : ℕ)
variables (h1 : B + W = 200)
variables (h2 : 30 * B + 25 * W = 5500)

theorem ratio_black_white (B W : ℕ) (h1 : B + W = 200) (h2 : 30 * B + 25 * W = 5500) :
  B = W :=
by
  -- Proof omitted
  sorry

end ratio_black_white_l1926_192692


namespace rationalize_sqrt_35_l1926_192646

theorem rationalize_sqrt_35 : (35 / Real.sqrt 35) = Real.sqrt 35 :=
  sorry

end rationalize_sqrt_35_l1926_192646


namespace total_area_of_folded_blankets_l1926_192671

-- Define the initial conditions
def initial_area : ℕ := 8 * 8
def folds : ℕ := 4
def num_blankets : ℕ := 3

-- Define the hypothesis about folding
def folded_area (initial_area : ℕ) (folds : ℕ) : ℕ :=
  initial_area / (2 ^ folds)

-- The total area of all folded blankets
def total_folded_area (initial_area : ℕ) (folds : ℕ) (num_blankets : ℕ) : ℕ :=
  num_blankets * folded_area initial_area folds

-- The theorem we want to prove
theorem total_area_of_folded_blankets : total_folded_area initial_area folds num_blankets = 12 := by
  sorry

end total_area_of_folded_blankets_l1926_192671


namespace statement_B_is_false_l1926_192666

def heartsuit (x y : ℝ) : ℝ := x^2 - y^2

theorem statement_B_is_false (x y : ℝ) : 3 * (heartsuit x y) ≠ heartsuit (3 * x) y := by
  sorry

end statement_B_is_false_l1926_192666


namespace Joseph_has_122_socks_l1926_192640

def JosephSocks : Nat := 
  let red_pairs := 9 / 2
  let white_pairs := red_pairs + 2
  let green_pairs := 2 * red_pairs
  let blue_pairs := 3 * green_pairs
  let black_pairs := blue_pairs - 5
  (red_pairs + white_pairs + green_pairs + blue_pairs + black_pairs) * 2

theorem Joseph_has_122_socks : JosephSocks = 122 := 
  by
  sorry

end Joseph_has_122_socks_l1926_192640


namespace smallest_possible_b_l1926_192698

theorem smallest_possible_b
  (a c b : ℤ)
  (h1 : a < c)
  (h2 : c < b)
  (h3 : c = (a + b) / 2)
  (h4 : b^2 / c = a) :
  b = 2 :=
sorry

end smallest_possible_b_l1926_192698


namespace factorization_correct_l1926_192631

theorem factorization_correct (x : ℝ) : 2 * x^2 - 4 * x + 2 = 2 * (x - 1)^2 :=
by sorry

end factorization_correct_l1926_192631


namespace amount_distribution_l1926_192603

theorem amount_distribution :
  ∃ (P Q R S T : ℝ), 
    (P + Q + R + S + T = 24000) ∧ 
    (R = (3 / 5) * (P + Q)) ∧ 
    (S = 0.45 * 24000) ∧ 
    (T = (1 / 2) * R) ∧ 
    (P + Q = 7000) ∧ 
    (R = 4200) ∧ 
    (S = 10800) ∧ 
    (T = 2100) :=
by
  sorry

end amount_distribution_l1926_192603


namespace find_ordered_pair_l1926_192600

theorem find_ordered_pair : ∃ k a : ℤ, 
  (∀ x : ℝ, (x^3 - 4*x^2 + 9*x - 6) % (x^2 - x + k) = 2*x + a) ∧ k = 4 ∧ a = 6 :=
sorry

end find_ordered_pair_l1926_192600


namespace angle_value_l1926_192612

theorem angle_value (y : ℝ) (h1 : 2 * y + 140 = 360) : y = 110 :=
by {
  -- Proof will be written here
  sorry
}

end angle_value_l1926_192612


namespace range_of_m_l1926_192632

open Real

theorem range_of_m (a b m : ℝ) (x : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 1 / a + 9 / b = 1) :
  a + b ≥ -x^2 + 4 * x + 18 - m ↔ m ≥ 6 :=
by sorry

end range_of_m_l1926_192632


namespace mark_reading_pages_before_injury_l1926_192683

theorem mark_reading_pages_before_injury:
  ∀ (h_increased: Nat) (pages_week: Nat), 
  (h_increased = 2 + (2 * 3/2)) ∧ (pages_week = 1750) → 100 = pages_week / 7 / h_increased * 2 := 
by
  sorry

end mark_reading_pages_before_injury_l1926_192683


namespace range_of_m_l1926_192618

-- Define the ellipse and conditions
def ellipse (x y : ℝ) (m : ℝ) : Prop := (x^2 / m) + (y^2 / 2) = 1
def point_exists (M : ℝ × ℝ) (C : ℝ → ℝ → ℝ → Prop) : Prop := ∃ p : ℝ × ℝ, C p.1 p.2 (M.1 + M.2)

-- State the theorem
theorem range_of_m (m : ℝ) (h₁ : ellipse x y m) (h₂ : point_exists M ellipse) :
  (0 < m ∧ m <= 1/2) ∨ (8 <= m) := 
sorry

end range_of_m_l1926_192618


namespace mr_wang_returns_to_start_elevator_electricity_consumption_l1926_192659

-- Definition for the first part of the problem
def floor_movements : List Int := [6, -3, 10, -8, 12, -7, -10]

theorem mr_wang_returns_to_start : List.sum floor_movements = 0 := by
  -- Calculation here, we'll replace with sorry for now.
  sorry

-- Definitions for the second part of the problem
def height_per_floor : Int := 3
def electricity_per_meter : Float := 0.2

-- Calculation of electricity consumption (distance * electricity_per_meter per floor)
def total_distance_traveled : Int := 
  (floor_movements.map Int.natAbs).sum * height_per_floor

theorem elevator_electricity_consumption : 
  (Float.ofInt total_distance_traveled) * electricity_per_meter = 33.6 := by
  -- Calculation here, we'll replace with sorry for now.
  sorry

end mr_wang_returns_to_start_elevator_electricity_consumption_l1926_192659


namespace cube_sum_inequality_l1926_192634

theorem cube_sum_inequality (a b : ℝ) (ha : a < 0) (hb : b < 0) : 
  a^3 + b^3 ≤ a * b^2 + a^2 * b :=
sorry

end cube_sum_inequality_l1926_192634


namespace base_conversion_l1926_192630

theorem base_conversion (A B : ℕ) (hA : A < 8) (hB : B < 6) (h : 7 * A = 5 * B) : 8 * A + B = 47 :=
by
  sorry

end base_conversion_l1926_192630


namespace students_in_grades_v_vi_l1926_192655

theorem students_in_grades_v_vi (n a b c p q : ℕ) (h1 : n = 100*a + 10*b + c)
  (h2 : a * b * c = p) (h3 : (p / 10) * (p % 10) = q) : n = 144 :=
sorry

end students_in_grades_v_vi_l1926_192655


namespace integer_to_the_fourth_l1926_192665

theorem integer_to_the_fourth (a : ℤ) (h : a = 243) : 3^12 * 3^8 = a^4 :=
by {
  sorry
}

end integer_to_the_fourth_l1926_192665


namespace isosceles_triangle_base_length_l1926_192614

theorem isosceles_triangle_base_length
  (a b : ℕ)
  (ha : a = 8)
  (hp : 2 * a + b = 25)
  : b = 9 :=
by
  sorry

end isosceles_triangle_base_length_l1926_192614


namespace annual_rent_per_square_foot_l1926_192681

theorem annual_rent_per_square_foot (length width : ℕ) (monthly_rent : ℕ)
  (h_length : length = 20) (h_width : width = 15) (h_monthly_rent : monthly_rent = 3600) :
  let area := length * width
  let annual_rent := monthly_rent * 12
  let annual_rent_per_sq_ft := annual_rent / area
  annual_rent_per_sq_ft = 144 := by
  sorry

end annual_rent_per_square_foot_l1926_192681


namespace jenny_profit_l1926_192670

-- Define the constants given in the problem
def cost_per_pan : ℝ := 10.00
def price_per_pan : ℝ := 25.00
def num_pans : ℝ := 20.0

-- Define the total revenue function
def total_revenue (num_pans : ℝ) (price_per_pan : ℝ) : ℝ := num_pans * price_per_pan

-- Define the total cost function
def total_cost (num_pans : ℝ) (cost_per_pan : ℝ) : ℝ := num_pans * cost_per_pan

-- Define the profit function as the total revenue minus the total cost
def total_profit (num_pans : ℝ) (price_per_pan : ℝ) (cost_per_pan : ℝ) : ℝ := 
  total_revenue num_pans price_per_pan - total_cost num_pans cost_per_pan

-- The statement to prove in Lean
theorem jenny_profit : total_profit num_pans price_per_pan cost_per_pan = 300.00 := 
by 
  sorry

end jenny_profit_l1926_192670


namespace area_of_scalene_right_triangle_l1926_192650

noncomputable def area_of_triangle_DEF (DE EF : ℝ) (h1 : DE > 0) (h2 : EF > 0) (h3 : DE / EF = 3) (h4 : DE^2 + EF^2 = 16) : ℝ :=
1 / 2 * DE * EF

theorem area_of_scalene_right_triangle (DE EF : ℝ) 
  (h1 : DE > 0)
  (h2 : EF > 0)
  (h3 : DE / EF = 3)
  (h4 : DE^2 + EF^2 = 16) :
  area_of_triangle_DEF DE EF h1 h2 h3 h4 = 2.4 :=
sorry

end area_of_scalene_right_triangle_l1926_192650


namespace jamshid_takes_less_time_l1926_192645

open Real

theorem jamshid_takes_less_time (J : ℝ) (hJ : J < 15) (h_work_rate : (1 / J) + (1 / 15) = 1 / 5) :
  (15 - J) / 15 * 100 = 50 :=
by
  sorry

end jamshid_takes_less_time_l1926_192645


namespace rita_book_pages_l1926_192627

theorem rita_book_pages (x : ℕ) (h1 : ∃ n₁, n₁ = (1/6 : ℚ) * x + 10) 
                                  (h2 : ∃ n₂, n₂ = (1/5 : ℚ) * ((5/6 : ℚ) * x - 10) + 20)
                                  (h3 : ∃ n₃, n₃ = (1/4 : ℚ) * ((4/5 : ℚ) * ((5/6 : ℚ) * x - 10) - 20) + 25)
                                  (h4 : ((3/4 : ℚ) * ((2/3 : ℚ) * x - 28) - 25) = 50) :
    x = 192 := 
sorry

end rita_book_pages_l1926_192627


namespace number_of_boys_girls_l1926_192615

-- Define the initial conditions.
def group_size : ℕ := 8
def total_ways : ℕ := 90

-- Define the actual proof problem.
theorem number_of_boys_girls 
  (n m : ℕ) 
  (h1 : n + m = group_size) 
  (h2 : Nat.choose n 2 * Nat.choose m 1 * Nat.factorial 3 = total_ways) 
  : n = 3 ∧ m = 5 :=
sorry

end number_of_boys_girls_l1926_192615


namespace lana_total_winter_clothing_l1926_192616

-- Define the number of boxes, scarves per box, and mittens per box as given in the conditions
def num_boxes : ℕ := 5
def scarves_per_box : ℕ := 7
def mittens_per_box : ℕ := 8

-- The total number of pieces of winter clothing is calculated as total scarves plus total mittens
def total_winter_clothing : ℕ := num_boxes * scarves_per_box + num_boxes * mittens_per_box

-- State the theorem that needs to be proven
theorem lana_total_winter_clothing : total_winter_clothing = 75 := by
  sorry

end lana_total_winter_clothing_l1926_192616


namespace sufficient_but_not_necessary_condition_l1926_192668

open Real

theorem sufficient_but_not_necessary_condition {x y : ℝ} :
  (x = y → |x| = |y|) ∧ (|x| = |y| → x = y) = false :=
by
  sorry

end sufficient_but_not_necessary_condition_l1926_192668


namespace sqrt_a_add_4b_eq_pm3_l1926_192699

theorem sqrt_a_add_4b_eq_pm3
  (a b : ℝ)
  (A_sol : a * (-1) + 5 * (-1) = 15)
  (B_sol : 4 * 5 - b * 2 = -2) :
  (a + 4 * b)^(1/2) = 3 ∨ (a + 4 * b)^(1/2) = -3 := by
  sorry

end sqrt_a_add_4b_eq_pm3_l1926_192699


namespace correct_calculation_l1926_192647

theorem correct_calculation (a b x y : ℝ) :
  (7 * a^2 * b - 7 * b * a^2 = 0) ∧ 
  (¬ (6 * a + 4 * b = 10 * a * b)) ∧ 
  (¬ (7 * x^2 * y - 3 * x^2 * y = 4 * x^4 * y^2)) ∧ 
  (¬ (8 * x^2 + 8 * x^2 = 16 * x^4)) :=
sorry

end correct_calculation_l1926_192647


namespace hamburger_combinations_l1926_192656

theorem hamburger_combinations : 
  let condiments := 10  -- Number of available condiments
  let patty_choices := 4 -- Number of meat patty options
  2^condiments * patty_choices = 4096 :=
by sorry

end hamburger_combinations_l1926_192656


namespace parallel_vectors_l1926_192677

def vec_a (x : ℝ) : ℝ × ℝ := (x, x + 2)
def vec_b : ℝ × ℝ := (1, 2)

theorem parallel_vectors (x : ℝ) : vec_a x = (2, 4) → x = 2 := by
  sorry

end parallel_vectors_l1926_192677


namespace dave_total_earnings_l1926_192611

def hourly_wage (day : ℕ) : ℝ :=
  if day = 0 then 6 else
  if day = 1 then 7 else
  if day = 2 then 9 else
  if day = 3 then 8 else 
  0

def hours_worked (day : ℕ) : ℝ :=
  if day = 0 then 6 else
  if day = 1 then 2 else
  if day = 2 then 3 else
  if day = 3 then 5 else 
  0

def unpaid_break (day : ℕ) : ℝ :=
  if day = 0 then 0.5 else
  if day = 1 then 0.25 else
  if day = 2 then 0 else
  if day = 3 then 0.5 else 
  0

def daily_earnings (day : ℕ) : ℝ :=
  (hours_worked day - unpaid_break day) * hourly_wage day

def net_earnings (day : ℕ) : ℝ :=
  daily_earnings day - (daily_earnings day * 0.1)

def total_net_earnings : ℝ :=
  net_earnings 0 + net_earnings 1 + net_earnings 2 + net_earnings 3

theorem dave_total_earnings : total_net_earnings = 97.43 := by
  sorry

end dave_total_earnings_l1926_192611


namespace Jolene_raised_total_money_l1926_192622

-- Definitions for the conditions
def babysits_earning_per_family : ℤ := 30
def number_of_families : ℤ := 4
def cars_earning_per_car : ℤ := 12
def number_of_cars : ℤ := 5

-- Calculation of total earnings
def babysitting_earnings : ℤ := babysits_earning_per_family * number_of_families
def car_washing_earnings : ℤ := cars_earning_per_car * number_of_cars
def total_earnings : ℤ := babysitting_earnings + car_washing_earnings

-- The proof statement
theorem Jolene_raised_total_money : total_earnings = 180 := by
  sorry

end Jolene_raised_total_money_l1926_192622


namespace find_d_l1926_192638

theorem find_d (a b c d : ℕ) (ha : 1 < a) (hb : 1 < b) (hc : 1 < c) (hd : 1 < d) 
  (h_eq : ∀ M : ℝ, M ≠ 1 → (M^(1/a)) * (M^(1/(a * b))) * (M^(1/(a * b * c))) * (M^(1/(a * b * c * d))) = M^(17/24)) : d = 8 :=
sorry

end find_d_l1926_192638


namespace sum_of_shaded_cells_l1926_192654

theorem sum_of_shaded_cells (a b c d e f : ℕ) 
  (h1: (a = 1 ∨ a = 2 ∨ a = 3) ∧ (b = 1 ∨ b = 2 ∨ b = 3) ∧ (c = 1 ∨ c = 2 ∨ c = 3) ∧ 
       (d = 1 ∨ d = 2 ∨ d = 3) ∧ (e = 1 ∨ e = 2 ∨ e = 3) ∧ (f = 1 ∨ f = 2 ∨ f = 3))
  (h2: (a ≠ b ∧ a ≠ c ∧ b ≠ c) ∧ 
       (d ≠ e ∧ d ≠ f ∧ e ≠ f) ∧ 
       (a ≠ d ∧ a ≠ f ∧ d ≠ f ∧ 
        b ≠ e ∧ b ≠ f ∧ c ≠ e ∧ c ≠ f))
  (h3: c = 3 ∧ d = 3 ∧ b = 2 ∧ e = 2)
  : b + e = 4 := 
sorry

end sum_of_shaded_cells_l1926_192654


namespace cyclic_sum_inequality_l1926_192680

theorem cyclic_sum_inequality (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : 
  ( ( (a - b) * (a - c) / (a + b + c) ) + 
    ( (b - c) * (b - d) / (b + c + d) ) + 
    ( (c - d) * (c - a) / (c + d + a) ) + 
    ( (d - a) * (d - b) / (d + a + b) ) ) ≥ 0 := 
by
  sorry

end cyclic_sum_inequality_l1926_192680


namespace sum_of_ages_l1926_192658

variable (S M : ℝ)  -- Variables for Sarah's and Matt's ages

-- Conditions
def sarah_older := S = M + 8
def future_age_relationship := S + 10 = 3 * (M - 5)

-- Theorem: The sum of their current ages is 41
theorem sum_of_ages (h1 : sarah_older S M) (h2 : future_age_relationship S M) : S + M = 41 := by
  sorry

end sum_of_ages_l1926_192658


namespace eq_or_neg_eq_of_eq_frac_l1926_192610

theorem eq_or_neg_eq_of_eq_frac (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h : a^2 + b^3 / a = b^2 + a^3 / b) :
  a = b ∨ a = -b :=
by
  sorry

end eq_or_neg_eq_of_eq_frac_l1926_192610


namespace percentage_increase_in_area_l1926_192606

-- Defining the lengths and widths in terms of real numbers
variables (L W : ℝ)

-- Defining the new lengths and widths
def new_length := 1.2 * L
def new_width := 1.2 * W

-- Original area of the rectangle
def original_area := L * W

-- New area of the rectangle
def new_area := new_length L * new_width W

-- Proof statement for the percentage increase
theorem percentage_increase_in_area : 
  ((new_area L W - original_area L W) / original_area L W) * 100 = 44 :=
by
  sorry

end percentage_increase_in_area_l1926_192606


namespace find_a8_l1926_192690

variable (a : ℕ+ → ℕ)

theorem find_a8 (h : ∀ m n : ℕ+, a (m * n) = a m * a n) (h2 : a 2 = 3) : a 8 = 27 := 
by
  sorry

end find_a8_l1926_192690


namespace basketball_game_l1926_192619

variable (H E : ℕ)

theorem basketball_game (h_eq_sum : H + E = 50) (h_margin : H = E + 6) : E = 22 := by
  sorry

end basketball_game_l1926_192619


namespace max_ab_ac_bc_l1926_192608

theorem max_ab_ac_bc (a b c : ℝ) (h : a + 3 * b + c = 6) : 
    ab + ac + bc <= 8 :=
sorry

end max_ab_ac_bc_l1926_192608


namespace hyperbola_condition_l1926_192625

theorem hyperbola_condition (k : ℝ) : 
  (0 < k ∧ k < 1) → ¬((k > 1 ∨ k < -2) ↔ (0 < k ∧ k < 1)) :=
by
  intro hk
  sorry

end hyperbola_condition_l1926_192625


namespace average_visitors_per_day_l1926_192629

theorem average_visitors_per_day (avg_visitors_Sunday : ℕ) (avg_visitors_other_days : ℕ) (total_days : ℕ) (starts_on_Sunday : Bool) :
  avg_visitors_Sunday = 500 → 
  avg_visitors_other_days = 140 → 
  total_days = 30 → 
  starts_on_Sunday = true → 
  (4 * avg_visitors_Sunday + 26 * avg_visitors_other_days) / total_days = 188 :=
by
  intros h1 h2 h3 h4
  sorry

end average_visitors_per_day_l1926_192629


namespace floor_sum_value_l1926_192641

theorem floor_sum_value (a b c d : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_pos_d : d > 0)
(h1 : a^2 + b^2 = 2016) (h2 : c^2 + d^2 = 2016) (h3 : a * c = 1024) (h4 : b * d = 1024) :
  ⌊a + b + c + d⌋ = 127 := sorry

end floor_sum_value_l1926_192641


namespace change_in_profit_rate_l1926_192637

theorem change_in_profit_rate (A B C : Type) (P : ℝ) (r1 r2 : ℝ) (income_increase : ℝ) (capital : ℝ) :
  (A_receives : ℝ) = (2 / 3) → 
  (B_C_divide : ℝ) = (1 - (2 / 3)) / 2 → 
  income_increase = 300 → 
  capital = 15000 →
  ((2 / 3) * capital * (r2 / 100) - (2 / 3) * capital * (r1 / 100)) = income_increase →
  (r2 - r1) = 3 :=
by
  intros
  sorry

end change_in_profit_rate_l1926_192637


namespace binom_inequality_l1926_192620

-- Defining the conditions as non-computable functions
def is_nonneg_integer := ℕ

-- Defining the binomial coefficient function
noncomputable def binom (n k : ℕ) : ℕ := Nat.choose n k

-- The statement of the theorem
theorem binom_inequality (n k h : ℕ) (hn : n ≥ k + h) : binom n (k + h) ≥ binom (n - k) h :=
  sorry

end binom_inequality_l1926_192620


namespace aluminum_atomic_weight_l1926_192657

theorem aluminum_atomic_weight (Al_w : ℤ) 
  (compound_molecular_weight : ℤ) 
  (num_fluorine_atoms : ℕ) 
  (fluorine_atomic_weight : ℤ) 
  (h1 : compound_molecular_weight = 84) 
  (h2 : num_fluorine_atoms = 3) 
  (h3 : fluorine_atomic_weight = 19) :
  Al_w = 27 := 
by
  -- Proof goes here, but it is skipped.
  sorry

end aluminum_atomic_weight_l1926_192657


namespace number_of_apples_and_erasers_l1926_192633

def totalApplesAndErasers (a e : ℕ) : Prop :=
  a + e = 84

def applesPerFriend (a : ℕ) : ℕ :=
  a / 3

def erasersPerTeacher (e : ℕ) : ℕ :=
  e / 2

theorem number_of_apples_and_erasers (a e : ℕ) (h : totalApplesAndErasers a e) :
  applesPerFriend a = a / 3 ∧ erasersPerTeacher e = e / 2 :=
by
  sorry

end number_of_apples_and_erasers_l1926_192633


namespace Agnes_age_now_l1926_192679

variable (A : ℕ) (J : ℕ := 6)

theorem Agnes_age_now :
  (2 * (J + 13) = A + 13) → A = 25 :=
by
  intro h
  sorry

end Agnes_age_now_l1926_192679


namespace number_of_sides_of_polygon_l1926_192687

-- Given definition about angles and polygons
def exterior_angle (sides: ℕ) : ℝ := 30

-- The sum of exterior angles of any polygon
def sum_exterior_angles : ℝ := 360

-- The proof statement
theorem number_of_sides_of_polygon (k : ℕ) 
  (h1 : exterior_angle k = 30) 
  (h2 : sum_exterior_angles = 360):
  k = 12 :=
sorry

end number_of_sides_of_polygon_l1926_192687


namespace necessary_but_not_sufficient_condition_l1926_192675

variable (a : ℝ)

theorem necessary_but_not_sufficient_condition (h : 0 ≤ a ∧ a ≤ 4) :
  (∀ x : ℝ, x^2 + a * x + a > 0) → (0 ≤ a ∧ a ≤ 4 ∧ ¬ (∀ x : ℝ, x^2 + a * x + a > 0)) :=
sorry

end necessary_but_not_sufficient_condition_l1926_192675


namespace company_percentage_increase_l1926_192648

/-- Company P had 426.09 employees in January and 490 employees in December.
    Prove that the percentage increase in employees from January to December is 15%. --/
theorem company_percentage_increase :
  ∀ (employees_jan employees_dec : ℝ),
  employees_jan = 426.09 → 
  employees_dec = 490 → 
  ((employees_dec - employees_jan) / employees_jan) * 100 = 15 :=
by
  intros employees_jan employees_dec h_jan h_dec
  sorry

end company_percentage_increase_l1926_192648


namespace evaluate_expression_l1926_192649

theorem evaluate_expression : 500 * (500 ^ 500) * 500 = 500 ^ 502 := by
  sorry

end evaluate_expression_l1926_192649


namespace cans_collected_is_232_l1926_192623

-- Definitions of the conditions
def total_students : ℕ := 30
def half_students : ℕ := total_students / 2
def cans_per_half_student : ℕ := 12
def remaining_students : ℕ := 13
def cans_per_remaining_student : ℕ := 4

-- Calculate total cans collected
def total_cans_collected : ℕ := (half_students * cans_per_half_student) + (remaining_students * cans_per_remaining_student)

-- The theorem to be proved
theorem cans_collected_is_232 : total_cans_collected = 232 := by
  -- Proof would go here
  sorry

end cans_collected_is_232_l1926_192623


namespace rectangle_area_l1926_192643

theorem rectangle_area (w l : ℝ) (h_width : w = 4) (h_perimeter : 2 * l + 2 * w = 30) :
    l * w = 44 :=
by 
  sorry

end rectangle_area_l1926_192643


namespace percentage_less_than_y_l1926_192672

variable (w x y z : ℝ)

-- Given conditions
variable (h1 : w = 0.60 * x)
variable (h2 : x = 0.60 * y)
variable (h3 : z = 1.50 * w)

theorem percentage_less_than_y : ( (y - z) / y) * 100 = 46 := by
  sorry

end percentage_less_than_y_l1926_192672


namespace trigonometric_expression_eq_neg3_l1926_192653

theorem trigonometric_expression_eq_neg3
  {α : ℝ} (h : Real.tan α = 1 / 2) :
  (1 + 2 * Real.sin (π - α) * Real.cos (-2 * π - α)) /
  ((Real.sin (-α))^2 - (Real.sin (5 * π / 2 - α))^2) = -3 :=
sorry

end trigonometric_expression_eq_neg3_l1926_192653


namespace least_positive_t_geometric_progression_l1926_192642

open Real

theorem least_positive_t_geometric_progression (α : ℝ) (hα1 : 0 < α) (hα2 : α < π / 2) : 
  ∃ t : ℕ, ∀ t' : ℕ, (t' > 0) → 
  (|arcsin (sin (t' * α)) - 8 * α| = 0) → t = 8 :=
by
  sorry

end least_positive_t_geometric_progression_l1926_192642


namespace quadratic_function_distinct_zeros_l1926_192662

theorem quadratic_function_distinct_zeros (a : ℝ) (h : a ≠ 0) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a * x1^2 + 4 * x1 - 2 = 0 ∧ a * x2^2 + 4 * x2 - 2 = 0) ↔ (a ∈ Set.Ioo (-2) 0 ∪ Set.Ioi 0) := 
by
  sorry

end quadratic_function_distinct_zeros_l1926_192662


namespace fish_offspring_base10_l1926_192635

def convert_base_7_to_10 (n : ℕ) : ℕ :=
  let d2 := n / 49
  let r2 := n % 49
  let d1 := r2 / 7
  let d0 := r2 % 7
  d2 * 49 + d1 * 7 + d0

theorem fish_offspring_base10 :
  convert_base_7_to_10 265 = 145 :=
by
  sorry

end fish_offspring_base10_l1926_192635


namespace number_of_possible_values_r_l1926_192694

noncomputable def is_closest_approx (r : ℝ) : Prop :=
  (r >= 0.2857) ∧ (r < 0.2858)

theorem number_of_possible_values_r : 
  ∃ n : ℕ, (∀ r : ℝ, is_closest_approx r ↔ r = 0.2857 ∨ r = 0.2858 ∨ r = 0.2859) ∧ n = 3 :=
by
  sorry

end number_of_possible_values_r_l1926_192694


namespace cube_edge_length_l1926_192617

-- Definitions based on given conditions
def paper_cost_per_kg : ℝ := 60
def paper_area_coverage_per_kg : ℝ := 20
def total_expenditure : ℝ := 1800
def surface_area_of_cube (a : ℝ) : ℝ := 6 * a^2

-- The main proof problem
theorem cube_edge_length :
  ∃ a : ℝ, surface_area_of_cube a = paper_area_coverage_per_kg * (total_expenditure / paper_cost_per_kg) ∧ a = 10 :=
by
  sorry

end cube_edge_length_l1926_192617


namespace relationship_between_abc_l1926_192652

open Real

-- Define the constants for the problem
noncomputable def a : ℝ := sqrt 2023 - sqrt 2022
noncomputable def b : ℝ := sqrt 2022 - sqrt 2021
noncomputable def c : ℝ := sqrt 2021 - sqrt 2020

-- State the theorem we want to prove
theorem relationship_between_abc : c > b ∧ b > a := 
sorry

end relationship_between_abc_l1926_192652


namespace find_smaller_number_l1926_192661

theorem find_smaller_number (x : ℕ) (h : 3 * x + 4 * x = 420) : 3 * x = 180 :=
by
  sorry

end find_smaller_number_l1926_192661


namespace solve_ordered_pair_l1926_192621

theorem solve_ordered_pair (x y : ℝ) 
  (h1 : x + y = (7 - x) + (7 - y))
  (h2 : x^2 - y = (x - 2) + (y - 2)) :
  (x = -5 ∧ y = 12) ∨ (x = 2 ∧ y = 5) :=
  sorry

end solve_ordered_pair_l1926_192621


namespace trajectory_equation_l1926_192686

theorem trajectory_equation :
  ∀ (N : ℝ × ℝ), (∃ (F : ℝ × ℝ) (P : ℝ × ℝ) (M : ℝ × ℝ), 
    F = (1, 0) ∧ 
    (∃ b : ℝ, P = (0, b)) ∧ 
    (∃ a : ℝ, a ≠ 0 ∧ M = (a, 0)) ∧ 
    (N.fst = -(M.fst) ∧ N.snd = 2 * P.snd) ∧ 
    ((-M.fst) * F.fst + (-(M.snd)) * (-(P.snd)) = 0) ∧ 
    ((-M.fst, -M.snd) + (N.fst, N.snd) = (0,0))) → 
  (N.snd)^2 = 4 * (N.fst) :=
by
  intros N h
  sorry

end trajectory_equation_l1926_192686


namespace socorro_training_hours_l1926_192651

theorem socorro_training_hours :
  let daily_multiplication_time := 10  -- in minutes
  let daily_division_time := 20        -- in minutes
  let training_days := 10              -- in days
  let minutes_per_hour := 60           -- minutes in an hour
  let daily_total_time := daily_multiplication_time + daily_division_time
  let total_training_time := daily_total_time * training_days
  total_training_time / minutes_per_hour = 5 :=
by sorry

end socorro_training_hours_l1926_192651


namespace more_stickers_correct_l1926_192604

def total_stickers : ℕ := 58
def first_box_stickers : ℕ := 23
def second_box_stickers : ℕ := total_stickers - first_box_stickers
def more_stickers_in_second_box : ℕ := second_box_stickers - first_box_stickers

theorem more_stickers_correct : more_stickers_in_second_box = 12 := by
  sorry

end more_stickers_correct_l1926_192604


namespace distance_apart_after_skating_l1926_192697

theorem distance_apart_after_skating :
  let Ann_speed := 6 -- Ann's speed in miles per hour
  let Glenda_speed := 8 -- Glenda's speed in miles per hour
  let skating_time := 3 -- Time spent skating in hours
  let Distance_Ann := Ann_speed * skating_time
  let Distance_Glenda := Glenda_speed * skating_time
  let Total_Distance := Distance_Ann + Distance_Glenda
  Total_Distance = 42 :=
by
  let Ann_speed := 6
  let Glenda_speed := 8
  let skating_time := 3
  let Distance_Ann := Ann_speed * skating_time
  let Distance_Glenda := Glenda_speed * skating_time
  let Total_Distance := Distance_Ann + Distance_Glenda
  sorry

end distance_apart_after_skating_l1926_192697


namespace mangoes_total_l1926_192626

theorem mangoes_total (Dilan Ashley Alexis : ℕ) (h1 : Alexis = 4 * (Dilan + Ashley)) (h2 : Ashley = 2 * Dilan) (h3 : Alexis = 60) : Dilan + Ashley + Alexis = 75 :=
by
  sorry

end mangoes_total_l1926_192626


namespace problem_1_problem_2_problem_3_l1926_192696

-- Problem 1
theorem problem_1 (f : ℝ → ℝ) : (∀ x : ℝ, f (x + 1) = x^2 + 4*x + 1) → (∀ x : ℝ, f x = x^2 + 2*x - 2) :=
by
  intro h
  sorry

-- Problem 2
theorem problem_2 (f : ℝ → ℝ) : (∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b) → (∀ x : ℝ, 3 * f (x + 1) - f x = 2 * x + 9) → (∀ x : ℝ, f x = x + 3) :=
by
  intros h1 h2
  sorry

-- Problem 3
theorem problem_3 (f : ℝ → ℝ) : (∀ x : ℝ, 2 * f x + f (1 / x) = 3 * x) → (∀ x : ℝ, f x = 2 * x - 1 / x) :=
by
  intro h
  sorry

end problem_1_problem_2_problem_3_l1926_192696


namespace p_q_false_of_not_or_l1926_192663

variables (p q : Prop)

theorem p_q_false_of_not_or (h : ¬(p ∨ q)) : ¬p ∧ ¬q :=
by {
  sorry
}

end p_q_false_of_not_or_l1926_192663


namespace pure_imaginary_condition_l1926_192693

def z1 : ℂ := 3 - 2 * Complex.I
def z2 (m : ℝ) : ℂ := 1 + m * Complex.I

theorem pure_imaginary_condition (m : ℝ) : z1 * z2 m ∈ {z : ℂ | z.re = 0} ↔ m = -3 / 2 := by
  sorry

end pure_imaginary_condition_l1926_192693


namespace compound_interest_double_l1926_192602

theorem compound_interest_double (t : ℕ) (r : ℝ) (n : ℕ) (P : ℝ) :
  r = 0.15 → n = 1 → (2 : ℝ) < (1 + r)^t → t ≥ 5 :=
by
  intros hr hn h
  sorry

end compound_interest_double_l1926_192602


namespace dice_five_prob_l1926_192644

-- Define a standard six-sided die probability
def prob_five : ℚ := 1 / 6

-- Define the probability of all four dice showing five
def prob_all_five : ℚ := prob_five * prob_five * prob_five * prob_five

-- State the theorem
theorem dice_five_prob : prob_all_five = 1 / 1296 := by
  sorry

end dice_five_prob_l1926_192644


namespace total_interest_rate_l1926_192684

theorem total_interest_rate (I_total I_11: ℝ) (r_9 r_11: ℝ) (h1: I_total = 100000) (h2: I_11 = 12499.999999999998) (h3: I_11 < I_total):
  r_9 = 0.09 →
  r_11 = 0.11 →
  ( ((I_total - I_11) * r_9 + I_11 * r_11) / I_total * 100 = 9.25 ) :=
by
  sorry

end total_interest_rate_l1926_192684


namespace no_three_consecutive_geometric_l1926_192660

open Nat

def a (n : ℕ) : ℤ := 3^n - 2^n

theorem no_three_consecutive_geometric :
  ∀ (k : ℕ), ¬ (∃ n m : ℕ, m = n + 1 ∧ k = m + 1 ∧ (a n) * (a k) = (a m)^2) :=
by
  sorry

end no_three_consecutive_geometric_l1926_192660


namespace rectangle_circle_diameter_l1926_192673

theorem rectangle_circle_diameter:
  ∀ (m n : ℕ), (∃ (x : ℚ), m + n = 47 ∧ (∀ (r : ℚ), r = (20 / 7)) →
  (2 * r = (40 / 7))) :=
by
  sorry

end rectangle_circle_diameter_l1926_192673


namespace scale_model_height_l1926_192609

theorem scale_model_height 
  (scale_ratio : ℚ) (actual_height : ℚ)
  (h_ratio : scale_ratio = 1/30)
  (h_actual_height : actual_height = 305) 
  : Int.ceil (actual_height * scale_ratio) = 10 := by
  -- Define variables and the necessary conditions
  let height_of_model: ℚ := actual_height * scale_ratio
  -- Skip the proof steps
  sorry

end scale_model_height_l1926_192609


namespace abc_eq_one_and_sum_eq_reciprocal_implies_one_is_one_l1926_192601

theorem abc_eq_one_and_sum_eq_reciprocal_implies_one_is_one 
  (a b c : ℝ) 
  (h1 : a * b * c = 1) 
  (h2 : a + b + c = 1 / a + 1 / b + 1 / c) : 
  (a = 1) ∨ (b = 1) ∨ (c = 1) :=
by
  sorry

end abc_eq_one_and_sum_eq_reciprocal_implies_one_is_one_l1926_192601


namespace dad_gave_nickels_l1926_192664

-- Definitions
def original_nickels : ℕ := 9
def total_nickels_after : ℕ := 12

-- Theorem to be proven
theorem dad_gave_nickels {original_nickels total_nickels_after : ℕ} : 
    total_nickels_after - original_nickels = 3 := 
by
  /- Sorry proof omitted -/
  sorry

end dad_gave_nickels_l1926_192664


namespace price_of_baseball_cards_l1926_192607

theorem price_of_baseball_cards 
    (packs_Digimon : ℕ)
    (price_per_pack : ℝ)
    (total_spent : ℝ)
    (total_cost_Digimon : ℝ) 
    (price_baseball_deck : ℝ) 
    (h1 : packs_Digimon = 4) 
    (h2 : price_per_pack = 4.45) 
    (h3 : total_spent = 23.86) 
    (h4 : total_cost_Digimon = packs_Digimon * price_per_pack) 
    (h5 : price_baseball_deck = total_spent - total_cost_Digimon) : 
    price_baseball_deck = 6.06 :=
sorry

end price_of_baseball_cards_l1926_192607


namespace interval_of_n_l1926_192639

theorem interval_of_n (n : ℕ) (h_pos : 0 < n) (h_lt_2000 : n < 2000) 
                      (h_div_99999999 : 99999999 % n = 0) (h_div_999999 : 999999 % (n + 6) = 0) : 
                      801 ≤ n ∧ n ≤ 1200 :=
by {
  sorry
}

end interval_of_n_l1926_192639


namespace maximize_profit_l1926_192636

noncomputable def profit (x : ℝ) : ℝ :=
  16 - 4/(x+1) - x

theorem maximize_profit (a : ℝ) (h : 0 ≤ a) :
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ a ∧ profit x = max 13 (16 - 4/(a+1) - a) := by
  sorry

end maximize_profit_l1926_192636


namespace a_oxen_count_l1926_192695

-- Define the conditions from the problem
def total_rent : ℝ := 210
def c_share_rent : ℝ := 54
def oxen_b : ℝ := 12
def oxen_c : ℝ := 15
def months_b : ℝ := 5
def months_c : ℝ := 3
def months_a : ℝ := 7
def oxen_c_months : ℝ := oxen_c * months_c
def total_ox_months (oxen_a : ℝ) : ℝ := (oxen_a * months_a) + (oxen_b * months_b) + oxen_c_months

-- The theorem we want to prove
theorem a_oxen_count (oxen_a : ℝ) (h : c_share_rent / total_rent = oxen_c_months / total_ox_months oxen_a) :
  oxen_a = 10 := by sorry

end a_oxen_count_l1926_192695


namespace cos_30_deg_plus_2a_l1926_192628

theorem cos_30_deg_plus_2a (a : ℝ) (h : Real.cos (Real.pi * (75 / 180) - a) = 1 / 3) : 
  Real.cos (Real.pi * (30 / 180) + 2 * a) = 7 / 9 := 
by 
  sorry

end cos_30_deg_plus_2a_l1926_192628


namespace maximum_value_of_expression_l1926_192691

theorem maximum_value_of_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (xyz * (x + y + z)) / ((x + y)^2 * (y + z)^2) ≤ (1 / 4) :=
sorry

end maximum_value_of_expression_l1926_192691


namespace f_zero_is_118_l1926_192688

theorem f_zero_is_118
  (f : ℕ → ℕ)
  (eq1 : ∀ m n : ℕ, f (m^2 + n^2) = (f m - f n)^2 + f (2 * m * n))
  (eq2 : 8 * f 0 + 9 * f 1 = 2006) :
  f 0 = 118 :=
sorry

end f_zero_is_118_l1926_192688


namespace min_value_expression_l1926_192674

theorem min_value_expression (a b : ℝ) (h1 : 2 * a + b = 1) (h2 : a > 0) (h3 : b > 0) :
  (1 / a) + ((1 - b) / b) = 2 + 2 * Real.sqrt 2 :=
by
  sorry

end min_value_expression_l1926_192674


namespace house_spirits_elevator_l1926_192624

-- Define the given conditions
def first_floor_domovoi := 1
def middle_floor_domovoi := 2
def last_floor_domovoi := 1
def total_floors := 7
def spirits_per_cycle := first_floor_domovoi + 5 * middle_floor_domovoi + last_floor_domovoi

-- Prove the statement
theorem house_spirits_elevator (n : ℕ) (floor : ℕ) (h1 : total_floors = 7) (h2 : spirits_per_cycle = 12) (h3 : n = 1000) :
  floor = 4 :=
by
  sorry

end house_spirits_elevator_l1926_192624


namespace find_positive_number_l1926_192605
-- Prove the positive number x that satisfies the condition is 8
theorem find_positive_number (x : ℝ) (hx : 0 < x) :
    x + 8 = 128 * (1 / x) → x = 8 :=
by
  intro h
  sorry

end find_positive_number_l1926_192605


namespace roja_speed_l1926_192613

theorem roja_speed (R : ℕ) (h1 : 3 + R = 7) : R = 7 - 3 :=
by sorry

end roja_speed_l1926_192613


namespace problem_solution_l1926_192676

theorem problem_solution (x1 x2 : ℝ) (h1 : x1^2 + x1 - 4 = 0) (h2 : x2^2 + x2 - 4 = 0) (h3 : x1 + x2 = -1) : 
  x1^3 - 5 * x2^2 + 10 = -19 := 
by 
  sorry

end problem_solution_l1926_192676


namespace last_digit_of_expression_l1926_192667

def last_digit (n : ℕ) : ℕ := n % 10

theorem last_digit_of_expression (n : ℕ) : last_digit (n ^ 9999 - n ^ 5555) = 0 :=
by
  sorry

end last_digit_of_expression_l1926_192667
