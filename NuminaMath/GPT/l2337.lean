import Mathlib

namespace blue_balls_count_l2337_233759

theorem blue_balls_count:
  ∀ (T : ℕ),
  (1/4 * T) + (1/8 * T) + (1/12 * T) + 26 = T → 
  (1 / 8) * T = 6 := by
  intros T h
  sorry

end blue_balls_count_l2337_233759


namespace geometric_sequence_problem_l2337_233789

theorem geometric_sequence_problem
  (a : ℕ → ℝ) (r : ℝ)
  (h₀ : ∀ n, a n > 0)
  (h₁ : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25)
  (h₂ : ∀ n, a (n + 1) = a n * r) :
  a 3 + a 5 = 5 :=
sorry

end geometric_sequence_problem_l2337_233789


namespace f_decreasing_on_neg_infty_2_l2337_233712

def f (x : ℝ) := x^2 - 4 * x + 3

theorem f_decreasing_on_neg_infty_2 :
  ∀ x y : ℝ, x < y → y ≤ 2 → f y < f x :=
by
  sorry

end f_decreasing_on_neg_infty_2_l2337_233712


namespace solve_fractional_equation_l2337_233754

theorem solve_fractional_equation :
  {x : ℝ | 1 / (x^2 + 8 * x - 6) + 1 / (x^2 + 5 * x - 6) + 1 / (x^2 - 14 * x - 6) = 0}
  = {3, -2, -6, 1} :=
by
  sorry

end solve_fractional_equation_l2337_233754


namespace remainder_250_div_k_l2337_233771

theorem remainder_250_div_k {k : ℕ} (h1 : 0 < k) (h2 : 180 % (k * k) = 12) : 250 % k = 10 := by
  sorry

end remainder_250_div_k_l2337_233771


namespace marcus_scored_50_percent_l2337_233775

variable (three_point_goals : ℕ) (two_point_goals : ℕ) (team_total_points : ℕ)

def marcus_percentage_points (three_point_goals two_point_goals team_total_points : ℕ) : ℚ :=
  let marcus_points := three_point_goals * 3 + two_point_goals * 2
  (marcus_points : ℚ) / team_total_points * 100

theorem marcus_scored_50_percent (h1 : three_point_goals = 5) (h2 : two_point_goals = 10) (h3 : team_total_points = 70) :
  marcus_percentage_points three_point_goals two_point_goals team_total_points = 50 :=
by
  sorry

end marcus_scored_50_percent_l2337_233775


namespace manuscript_fee_tax_l2337_233702

theorem manuscript_fee_tax (fee : ℕ) (tax_paid : ℕ) :
  (tax_paid = 0 ∧ fee ≤ 800) ∨ 
  (tax_paid = (14 * (fee - 800) / 100) ∧ 800 < fee ∧ fee ≤ 4000) ∨ 
  (tax_paid = 11 * fee / 100 ∧ fee > 4000) →
  tax_paid = 420 →
  fee = 3800 :=
by 
  intro h_eq h_tax;
  sorry

end manuscript_fee_tax_l2337_233702


namespace P_Q_sum_l2337_233718

noncomputable def find_P_Q_sum (P Q : ℚ) : Prop :=
  ∀ x : ℚ, (x^2 + 3 * x + 7) * (x^2 + (51/7) * x - 2) = x^4 + P * x^3 + Q * x^2 + 45 * x - 14

theorem P_Q_sum :
  ∃ P Q : ℚ, find_P_Q_sum P Q ∧ (P + Q = 260 / 7) :=
by
  sorry

end P_Q_sum_l2337_233718


namespace solve_for_x_l2337_233796

theorem solve_for_x (x y : ℤ) (h1 : x + y = 14) (h2 : x - y = 60) :
  x = 37 :=
sorry

end solve_for_x_l2337_233796


namespace max_value_ab_bc_cd_da_l2337_233772

theorem max_value_ab_bc_cd_da (a b c d : ℝ) (a_nonneg : 0 ≤ a) (b_nonneg : 0 ≤ b) (c_nonneg : 0 ≤ c)
  (d_nonneg : 0 ≤ d) (sum_eq_200 : a + b + c + d = 200) : 
  ab + bc + cd + 0.5 * d * a ≤ 11250 := 
sorry


end max_value_ab_bc_cd_da_l2337_233772


namespace sufficient_but_not_necessary_condition_l2337_233791

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

theorem sufficient_but_not_necessary_condition {x y : ℝ} :
  (floor x = floor y) → (abs (x - y) < 1) ∧ (¬ (abs (x - y) < 1) → (floor x ≠ floor y)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l2337_233791


namespace school_year_length_l2337_233780

theorem school_year_length
  (children : ℕ)
  (juice_boxes_per_child_per_day : ℕ)
  (days_per_week : ℕ)
  (total_juice_boxes : ℕ)
  (w : ℕ)
  (h1 : children = 3)
  (h2 : juice_boxes_per_child_per_day = 1)
  (h3 : days_per_week = 5)
  (h4 : total_juice_boxes = 375)
  (h5 : total_juice_boxes = children * juice_boxes_per_child_per_day * days_per_week * w)
  : w = 25 :=
by
  sorry

end school_year_length_l2337_233780


namespace parallel_vectors_implies_m_eq_neg1_l2337_233703

theorem parallel_vectors_implies_m_eq_neg1 (m : ℝ) :
  let a := (m, -1)
  let b := (1, m + 2)
  a.1 * b.2 = a.2 * b.1 → m = -1 :=
by
  intro h
  sorry

end parallel_vectors_implies_m_eq_neg1_l2337_233703


namespace pages_copied_l2337_233732

theorem pages_copied (cost_per_page total_cents : ℤ) (h1 : cost_per_page = 3) (h2 : total_cents = 1500) :
  total_cents / cost_per_page = 500 :=
by
  sorry

end pages_copied_l2337_233732


namespace count_perfect_squares_diff_l2337_233725

theorem count_perfect_squares_diff (a b : ℕ) : 
  ∃ (count : ℕ), 
  count = 25 ∧ 
  (∀ (a : ℕ), (∃ (b : ℕ), a^2 = 2 * b + 1 ∧ a^2 < 2500) ↔ (∃ (k : ℕ), 1 ≤ k ∧ k ≤ 25 ∧ 2 * k - 1 = a)) :=
by
  sorry

end count_perfect_squares_diff_l2337_233725


namespace find_BA_prime_l2337_233774

theorem find_BA_prime (BA BC A_prime C_1 : ℝ) 
  (h1 : BA = 3)
  (h2 : BC = 2)
  (h3 : A_prime < BA)
  (h4 : A_prime * C_1 = 3) : A_prime = 3 / 2 := 
by 
  sorry

end find_BA_prime_l2337_233774


namespace games_needed_to_declare_winner_l2337_233746

def single_elimination_games (T : ℕ) : ℕ :=
  T - 1

theorem games_needed_to_declare_winner (T : ℕ) :
  (single_elimination_games 23 = 22) :=
by
  sorry

end games_needed_to_declare_winner_l2337_233746


namespace larger_segment_of_triangle_l2337_233798

theorem larger_segment_of_triangle (x y : ℝ) (h1 : 40^2 = x^2 + y^2) 
  (h2 : 90^2 = (100 - x)^2 + y^2) :
  100 - x = 82.5 :=
by {
  sorry
}

end larger_segment_of_triangle_l2337_233798


namespace Linda_outfits_l2337_233704

theorem Linda_outfits (skirts blouses shoes : ℕ) 
  (hskirts : skirts = 5) 
  (hblouses : blouses = 8) 
  (hshoes : shoes = 2) :
  skirts * blouses * shoes = 80 := by
  -- We provide the proof here
  sorry

end Linda_outfits_l2337_233704


namespace tank_fill_time_l2337_233735

-- Define the conditions
def start_time : ℕ := 1 -- 1 pm
def first_hour_rainfall : ℕ := 2 -- 2 inches rainfall in the first hour from 1 pm to 2 pm
def next_four_hours_rate : ℕ := 1 -- 1 inch/hour rainfall rate from 2 pm to 6 pm
def following_rate : ℕ := 3 -- 3 inches/hour rainfall rate from 6 pm onwards
def tank_height : ℕ := 18 -- 18 inches tall fish tank

-- Define what needs to be proved
theorem tank_fill_time : 
  ∃ t : ℕ, t = 22 ∧ (tank_height ≤ (first_hour_rainfall + 4 * next_four_hours_rate + (t - 6)) + (t - 6 - 4) * following_rate) := 
by 
  sorry

end tank_fill_time_l2337_233735


namespace tan_alpha_tan_beta_l2337_233757

/-- Given the cosine values of the sum and difference of two angles, 
    find the value of the product of their tangents. -/
theorem tan_alpha_tan_beta (α β : ℝ) 
  (h1 : Real.cos (α + β) = 1/3) 
  (h2 : Real.cos (α - β) = 1/5) : 
  Real.tan α * Real.tan β = -1/4 := sorry

end tan_alpha_tan_beta_l2337_233757


namespace trajectory_eq_of_midpoint_l2337_233786

theorem trajectory_eq_of_midpoint (x y m n : ℝ) (hM_on_circle : m^2 + n^2 = 1)
  (hP_midpoint : (2*x = 3 + m) ∧ (2*y = n)) :
  (2*x - 3)^2 + 4*y^2 = 1 := 
sorry

end trajectory_eq_of_midpoint_l2337_233786


namespace min_value_2x_y_l2337_233743

noncomputable def min_value_of_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (heq : Real.log (x + 2 * y) = Real.log x + Real.log y) : ℝ :=
  2 * x + y

theorem min_value_2x_y : ∀ (x y : ℝ), 0 < x → 0 < y → Real.log (x + 2 * y) = Real.log x + Real.log y → 2 * x + y ≥ 9 :=
by
  intros x y hx hy heq
  sorry

end min_value_2x_y_l2337_233743


namespace matrix_det_zero_l2337_233783

variables {α β γ : ℝ}

theorem matrix_det_zero (h : α + β + γ = π) :
  Matrix.det ![
    ![Real.cos β, Real.cos α, -1],
    ![Real.cos γ, -1, Real.cos α],
    ![-1, Real.cos γ, Real.cos β]
  ] = 0 :=
sorry

end matrix_det_zero_l2337_233783


namespace expected_coins_basilio_20_l2337_233790

noncomputable def binomialExpectation (n : ℕ) (p : ℚ) : ℚ :=
  n * p

noncomputable def expectedCoinsDifference : ℚ :=
  0.5

noncomputable def expectedCoinsBasilio (n : ℕ) (p : ℚ) : ℚ :=
  (binomialExpectation n p + expectedCoinsDifference) / 2

theorem expected_coins_basilio_20 :
  expectedCoinsBasilio 20 (1/2) = 5.25 :=
by
  -- here you would fill in the proof steps
  sorry

end expected_coins_basilio_20_l2337_233790


namespace toluene_production_l2337_233706

def molar_mass_benzene : ℝ := 78.11 -- The molar mass of benzene in g/mol
def benzene_mass : ℝ := 156 -- The mass of benzene in grams
def methane_moles : ℝ := 2 -- The moles of methane

-- Define the balanced chemical reaction
def balanced_reaction (benzene methanol toluene hydrogen : ℝ) : Prop :=
  benzene + methanol = toluene + hydrogen

-- The main theorem statement
theorem toluene_production (h1 : balanced_reaction benzene_mass methane_moles 1 1)
  (h2 : benzene_mass / molar_mass_benzene = 2) :
  ∃ toluene_moles : ℝ, toluene_moles = 2 :=
by
  sorry

end toluene_production_l2337_233706


namespace smallest_positive_integer_x_l2337_233764

-- Definitions based on the conditions given
def is_multiple (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

-- Statement of the problem
theorem smallest_positive_integer_x (x : ℕ) :
  (is_multiple (900 * x) 640) → x = 32 :=
sorry

end smallest_positive_integer_x_l2337_233764


namespace find_value_of_expression_l2337_233736

theorem find_value_of_expression (x : ℝ) (h : x^2 + (1 / x^2) = 5) : x^4 + (1 / x^4) = 23 :=
by
  sorry

end find_value_of_expression_l2337_233736


namespace daily_sales_change_l2337_233716

theorem daily_sales_change
    (mon_sales : ℕ)
    (week_total_sales : ℕ)
    (days_in_week : ℕ)
    (avg_sales_per_day : ℕ)
    (other_days_total_sales : ℕ)
    (x : ℕ)
    (h1 : days_in_week = 7)
    (h2 : avg_sales_per_day = 5)
    (h3 : week_total_sales = avg_sales_per_day * days_in_week)
    (h4 : mon_sales = 2)
    (h5 : week_total_sales = mon_sales + other_days_total_sales)
    (h6 : other_days_total_sales = 33)
    (h7 : 2 + x + 2 + 2*x + 2 + 3*x + 2 + 4*x + 2 + 5*x + 2 + 6*x = other_days_total_sales) : 
  x = 1 :=
by
sorry

end daily_sales_change_l2337_233716


namespace dongzhi_daylight_hours_l2337_233744

theorem dongzhi_daylight_hours:
  let total_hours_in_day := 24
  let daytime_ratio := 5
  let nighttime_ratio := 7
  let total_parts := daytime_ratio + nighttime_ratio
  let daylight_hours := total_hours_in_day * daytime_ratio / total_parts
  daylight_hours = 10 :=
by
  sorry

end dongzhi_daylight_hours_l2337_233744


namespace average_seven_numbers_l2337_233711

theorem average_seven_numbers (A B C D E F G : ℝ) 
  (h1 : (A + B + C + D) / 4 = 4)
  (h2 : (D + E + F + G) / 4 = 4)
  (hD : D = 11) : 
  (A + B + C + D + E + F + G) / 7 = 3 :=
by
  sorry

end average_seven_numbers_l2337_233711


namespace fraction_neither_cell_phones_nor_pagers_l2337_233784

theorem fraction_neither_cell_phones_nor_pagers
  (E : ℝ) -- total number of employees (E must be positive)
  (h1 : 0 < E)
  (frac_cell_phones : ℝ)
  (H1 : frac_cell_phones = (2 / 3))
  (frac_pagers : ℝ)
  (H2 : frac_pagers = (2 / 5))
  (frac_both : ℝ)
  (H3 : frac_both = 0.4) :
  (1 / 3) = (1 - frac_cell_phones - frac_pagers + frac_both) :=
by
  -- setup definitions, conditions and final proof
  sorry

end fraction_neither_cell_phones_nor_pagers_l2337_233784


namespace estimate_mass_of_ice_floe_l2337_233751

noncomputable def mass_of_ice_floe (d : ℝ) (D : ℝ) (m : ℝ) : ℝ :=
  (m * d) / (D - d)

theorem estimate_mass_of_ice_floe :
  mass_of_ice_floe 9.5 10 600 = 11400 := 
by
  sorry

end estimate_mass_of_ice_floe_l2337_233751


namespace bus_avg_speed_l2337_233767

noncomputable def average_speed_of_bus 
  (bicycle_speed : ℕ) 
  (initial_distance_behind : ℕ) 
  (catch_up_time : ℕ) :
  ℕ :=
  (initial_distance_behind + bicycle_speed * catch_up_time) / catch_up_time

theorem bus_avg_speed 
  (bicycle_speed : ℕ) 
  (initial_distance_behind : ℕ) 
  (catch_up_time : ℕ) 
  (h_bicycle_speed : bicycle_speed = 15) 
  (h_initial_distance_behind : initial_distance_behind = 195)
  (h_catch_up_time : catch_up_time = 3) :
  average_speed_of_bus bicycle_speed initial_distance_behind catch_up_time = 80 :=
by
  sorry

end bus_avg_speed_l2337_233767


namespace Beth_and_Jan_total_money_l2337_233797

theorem Beth_and_Jan_total_money (B J : ℝ) 
  (h1 : B + 35 = 105)
  (h2 : J - 10 = B) : 
  B + J = 150 :=
by
  -- Proof omitted
  sorry

end Beth_and_Jan_total_money_l2337_233797


namespace journey_total_time_l2337_233705

def journey_time (d1 d2 : ℕ) (total_distance : ℕ) (car_speed walk_speed : ℕ) : ℕ :=
  d1 / car_speed + (total_distance - d1) / walk_speed

theorem journey_total_time :
  let total_distance := 150
  let car_speed := 30
  let walk_speed := 3
  let d1 := 50
  let d2 := 15
  
  journey_time d1 d2 total_distance car_speed walk_speed =
  max (journey_time d1 0 total_distance car_speed walk_speed / car_speed + 
       (total_distance - d1) / walk_speed)
      ((d1 / car_speed + (d1 - d2) / car_speed + (total_distance - d1 + d2) / car_speed)) :=
by
  sorry

end journey_total_time_l2337_233705


namespace elaine_earnings_increase_l2337_233770

variable (E : ℝ) (P : ℝ)

theorem elaine_earnings_increase
  (h1 : E > 0) 
  (h2 : 0.30 * E * (1 + P / 100) = 1.80 * 0.20 * E) : 
  P = 20 :=
by
  sorry

end elaine_earnings_increase_l2337_233770


namespace savings_amount_l2337_233745

-- Define the conditions for Celia's spending
def food_spending_per_week : ℝ := 100
def weeks : ℕ := 4
def rent_spending : ℝ := 1500
def video_streaming_services_spending : ℝ := 30
def cell_phone_usage_spending : ℝ := 50
def savings_rate : ℝ := 0.10

-- Define the total spending calculation
def total_spending : ℝ :=
  food_spending_per_week * weeks + rent_spending + video_streaming_services_spending + cell_phone_usage_spending

-- Define the savings calculation
def savings : ℝ :=
  savings_rate * total_spending

-- Prove the amount of savings
theorem savings_amount : savings = 198 :=
by
  -- This is the statement that needs to be proven, hence adding a placeholder proof.
  sorry

end savings_amount_l2337_233745


namespace d_share_l2337_233782

theorem d_share (x : ℝ) (d c : ℝ)
  (h1 : c = 3 * x + 500)
  (h2 : d = 3 * x)
  (h3 : c = 4 * x) :
  d = 1500 := 
by 
  sorry

end d_share_l2337_233782


namespace cups_remaining_l2337_233734

-- Definitions based on problem conditions
def initial_cups : ℕ := 12
def mary_morning_cups : ℕ := 1
def mary_evening_cups : ℕ := 1
def frank_afternoon_cups : ℕ := 1
def frank_late_evening_cups : ℕ := 2 * frank_afternoon_cups

-- Hypothesis combining all conditions:
def total_given_cups : ℕ :=
  mary_morning_cups + mary_evening_cups + frank_afternoon_cups + frank_late_evening_cups

-- Theorem to prove
theorem cups_remaining : initial_cups - total_given_cups = 7 :=
  sorry

end cups_remaining_l2337_233734


namespace recruits_line_l2337_233750

theorem recruits_line
  (x y z : ℕ) 
  (hx : x + y + z + 3 = 211) 
  (hx_peter : x = 50) 
  (hy_nikolai : y = 100) 
  (hz_denis : z = 170) 
  (hxy_ratio : x = 4 * z) : 
  x + y + z + 3 = 211 :=
by
  sorry

end recruits_line_l2337_233750


namespace numberOfWaysToPlaceCoinsSix_l2337_233748

def numberOfWaysToPlaceCoins (n : ℕ) : ℕ :=
  if n = 0 then 1 else 2 * numberOfWaysToPlaceCoins (n - 1)

theorem numberOfWaysToPlaceCoinsSix : numberOfWaysToPlaceCoins 6 = 32 :=
by
  sorry

end numberOfWaysToPlaceCoinsSix_l2337_233748


namespace chess_games_won_l2337_233709

theorem chess_games_won (W L : ℕ) (h1 : W + L = 44) (h2 : 4 * L = 7 * W) : W = 16 :=
by
  sorry

end chess_games_won_l2337_233709


namespace percent_of_a_is_4b_l2337_233717

theorem percent_of_a_is_4b (b : ℝ) (a : ℝ) (h : a = 1.8 * b) : (4 * b / a) * 100 = 222.22 := 
by {
  sorry
}

end percent_of_a_is_4b_l2337_233717


namespace largest_common_number_in_arithmetic_sequences_l2337_233788

theorem largest_common_number_in_arithmetic_sequences (x : ℕ)
  (h1 : x ≡ 2 [MOD 8])
  (h2 : x ≡ 5 [MOD 9])
  (h3 : x < 200) : x = 194 :=
by sorry

end largest_common_number_in_arithmetic_sequences_l2337_233788


namespace range_of_m_l2337_233733

-- Define the variables and main theorem
theorem range_of_m (m : ℝ) (a b c : ℝ) 
  (h₀ : a = 3) (h₁ : b = (1 - 2 * m)) (h₂ : c = 8)
  : -5 < m ∧ m < -2 :=
by
  -- Given that a, b, and c are sides of a triangle, we use the triangle inequality theorem
  -- This code will remain as a placeholder of that proof
  sorry

end range_of_m_l2337_233733


namespace problem1_problem2_problem2_zero_problem2_neg_l2337_233730

-- Definitions
def f (a x : ℝ) : ℝ := x^2 + a*x + a
def g (a x : ℝ) : ℝ := a*(f a x) - a^2*(x + 1) - 2*x

-- Problem 1
theorem problem1 (a : ℝ) :
  (∃ x1 x2 : ℝ, 0 < x1 ∧ x1 < x2 ∧ x2 < 1 ∧ f a x1 - x1 = 0 ∧ f a x2 - x2 = 0) →
  (0 < a ∧ a < 3 - 2*Real.sqrt 2) :=
sorry

-- Problem 2
theorem problem2 (a : ℝ) (h1 : a > 0) : 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → g a x ≥ 
    if a < 1 then a-2 
    else -1/a) :=
sorry

theorem problem2_zero (h2 : a = 0) : 
  g a 1 = -2 :=
sorry

theorem problem2_neg (a : ℝ) (h3 : a < 0) : 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → g a x ≥ a - 2) :=
sorry

end problem1_problem2_problem2_zero_problem2_neg_l2337_233730


namespace maximum_x_value_l2337_233741

theorem maximum_x_value (x y z : ℝ) (h1 : x + y + z = 10) (h2 : x * y + x * z + y * z = 20) : 
  x ≤ 10 / 3 := sorry

end maximum_x_value_l2337_233741


namespace arabella_first_step_time_l2337_233779

def time_first_step (x : ℝ) : Prop :=
  let time_second_step := x / 2
  let time_third_step := x + x / 2
  (x + time_second_step + time_third_step = 90)

theorem arabella_first_step_time (x : ℝ) (h : time_first_step x) : x = 30 :=
by
  sorry

end arabella_first_step_time_l2337_233779


namespace rosie_circles_track_24_l2337_233793

-- Definition of the problem conditions
def lou_distance := 3 -- Lou's total distance in miles
def track_length := 1 / 4 -- Length of the circular track in miles
def rosie_speed_factor := 2 -- Rosie runs at twice the speed of Lou

-- Define the number of times Rosie circles the track as a result
def rosie_circles_the_track : Nat :=
  let lou_circles := lou_distance / track_length
  let rosie_distance := lou_distance * rosie_speed_factor
  let rosie_circles := rosie_distance / track_length
  rosie_circles

-- The theorem stating that Rosie circles the track 24 times
theorem rosie_circles_track_24 : rosie_circles_the_track = 24 := by
  sorry

end rosie_circles_track_24_l2337_233793


namespace initial_pages_l2337_233737

/-
Given:
1. Sammy uses 25% of the pages for his science project.
2. Sammy uses another 10 pages for his math homework.
3. There are 80 pages remaining in the pad.

Prove that the initial number of pages in the pad (P) is 120.
-/

theorem initial_pages (P : ℝ) (h1 : P * 0.25 + 10 + 80 = P) : 
  P = 120 :=
by 
  sorry

end initial_pages_l2337_233737


namespace cubic_sum_identity_l2337_233707

variables (x y z : ℝ)

theorem cubic_sum_identity (h1 : x + y + z = 10) (h2 : xy + xz + yz = 30) :
  x^3 + y^3 + z^3 - 3 * x * y * z = 100 :=
sorry

end cubic_sum_identity_l2337_233707


namespace numbers_not_crossed_out_l2337_233765

/-- Total numbers between 1 and 90 after crossing out multiples of 3 and 5 is 48. -/
theorem numbers_not_crossed_out : 
  let n := 90 
  let multiples_of_3 := n / 3 
  let multiples_of_5 := n / 5 
  let multiples_of_15 := n / 15 
  let crossed_out := multiples_of_3 + multiples_of_5 - multiples_of_15
  n - crossed_out = 48 :=
by {
  sorry
}

end numbers_not_crossed_out_l2337_233765


namespace rational_root_uniqueness_l2337_233721

theorem rational_root_uniqueness (c : ℚ) :
  ∀ x1 x2 : ℚ, (x1 ≠ x2) →
  (x1^3 - 3 * c * x1^2 - 3 * x1 + c = 0) →
  (x2^3 - 3 * c * x2^2 - 3 * x2 + c = 0) →
  false := 
by
  intros x1 x2 h1 h2 h3
  sorry

end rational_root_uniqueness_l2337_233721


namespace circle_radius_is_six_l2337_233708

open Real

theorem circle_radius_is_six
  (r : ℝ)
  (h : 2 * 3 * 2 * π * r = 2 * π * r^2) :
  r = 6 := sorry

end circle_radius_is_six_l2337_233708


namespace intersection_A_B_l2337_233753

-- Define set A based on the given condition.
def setA : Set ℝ := {x | x^2 - 4 < 0}

-- Define set B based on the given condition.
def setB : Set ℝ := {x | x < 0}

-- Prove that the intersection of sets A and B is the given set.
theorem intersection_A_B : setA ∩ setB = {x | -2 < x ∧ x < 0} := by
  sorry

end intersection_A_B_l2337_233753


namespace women_at_each_table_l2337_233739

theorem women_at_each_table (W : ℕ) (h1 : ∃ W, ∀ i : ℕ, (i < 7) → W + 2 = 7 * W + 14) (h2 : 7 * W + 14 = 63) : W = 7 :=
by
  sorry

end women_at_each_table_l2337_233739


namespace variance_of_sample_l2337_233760

theorem variance_of_sample
  (x : ℝ)
  (h : (2 + 3 + x + 6 + 8) / 5 = 5) : 
  (1 / 5) * ((2 - 5) ^ 2 + (3 - 5) ^ 2 + (x - 5) ^ 2 + (6 - 5) ^ 2 + (8 - 5) ^ 2) = 24 / 5 :=
by
  sorry

end variance_of_sample_l2337_233760


namespace expression_equals_required_value_l2337_233755

-- Define the expression as needed
def expression : ℚ := (((((4 + 2)⁻¹ + 2)⁻¹) + 2)⁻¹) + 2

-- Define the theorem stating that the expression equals the required value
theorem expression_equals_required_value : 
  expression = 77 / 32 := 
sorry

end expression_equals_required_value_l2337_233755


namespace eval_diamond_expr_l2337_233726

def diamond (a b : ℕ) : ℕ :=
  match (a, b) with
  | (1, 1) => 4
  | (1, 2) => 3
  | (1, 3) => 2
  | (1, 4) => 1
  | (2, 1) => 1
  | (2, 2) => 4
  | (2, 3) => 3
  | (2, 4) => 2
  | (3, 1) => 2
  | (3, 2) => 1
  | (3, 3) => 4
  | (3, 4) => 3
  | (4, 1) => 3
  | (4, 2) => 2
  | (4, 3) => 1
  | (4, 4) => 4
  | (_, _) => 0  -- This handles any case outside of 1,2,3,4 which should ideally not happen

theorem eval_diamond_expr : diamond (diamond 3 4) (diamond 2 1) = 2 := by
  sorry

end eval_diamond_expr_l2337_233726


namespace seventy_five_inverse_mod_seventy_six_l2337_233714

-- Lean 4 statement for the problem.
theorem seventy_five_inverse_mod_seventy_six : (75 : ℤ) * 75 % 76 = 1 :=
by
  sorry

end seventy_five_inverse_mod_seventy_six_l2337_233714


namespace ball_hits_ground_in_3_seconds_l2337_233729

noncomputable def ball_height (t : ℝ) : ℝ := -16 * t^2 - 32 * t + 240

theorem ball_hits_ground_in_3_seconds :
  ∃ t : ℝ, ball_height t = 0 ∧ t = 3 :=
sorry

end ball_hits_ground_in_3_seconds_l2337_233729


namespace solve_inequality_l2337_233752

theorem solve_inequality {x : ℝ} :
  (3 / (5 - 3 * x) > 1) ↔ (2/3 < x ∧ x < 5/3) :=
by
  sorry

end solve_inequality_l2337_233752


namespace time_interval_between_recordings_is_5_seconds_l2337_233715

theorem time_interval_between_recordings_is_5_seconds
  (instances_per_hour : ℕ)
  (seconds_per_hour : ℕ)
  (h1 : instances_per_hour = 720)
  (h2 : seconds_per_hour = 3600) :
  seconds_per_hour / instances_per_hour = 5 :=
by
  -- proof omitted
  sorry

end time_interval_between_recordings_is_5_seconds_l2337_233715


namespace cost_price_for_a_l2337_233794

-- Definitions from the conditions
def selling_price_c : ℝ := 225
def profit_b : ℝ := 0.25
def profit_a : ℝ := 0.60

-- To prove: The cost price of the bicycle for A (cp_a) is 112.5
theorem cost_price_for_a : 
  ∃ (cp_a : ℝ), 
  (∃ (cp_b : ℝ), cp_b = (selling_price_c / (1 + profit_b)) ∧ 
   cp_a = (cp_b / (1 + profit_a))) ∧ 
   cp_a = 112.5 :=
by
  sorry

end cost_price_for_a_l2337_233794


namespace gcd_pow_sub_l2337_233713

theorem gcd_pow_sub (h1001 h1012 : ℕ) (h : 1001 ≤ 1012) : 
  (Nat.gcd (2 ^ 1001 - 1) (2 ^ 1012 - 1)) = 2047 := sorry

end gcd_pow_sub_l2337_233713


namespace calculate_expression_l2337_233795

theorem calculate_expression : (5 + 7 + 3) / 3 - 2 / 3 = 13 / 3 := by
  sorry

end calculate_expression_l2337_233795


namespace domain_of_function_l2337_233799

def function_domain : Set ℝ := { x : ℝ | x + 1 ≥ 0 ∧ 2 - x ≠ 0 }

theorem domain_of_function :
  function_domain = { x : ℝ | x ≥ -1 ∧ x ≠ 2 } :=
sorry

end domain_of_function_l2337_233799


namespace letters_identity_l2337_233720

theorem letters_identity (l1 l2 l3 : Prop) 
  (h1 : l1 → l2 → false)
  (h2 : ¬(l1 ∧ l3))
  (h3 : ¬(l2 ∧ l3))
  (h4 : l3 → ¬l1 ∧ l2 ∧ ¬(l1 ∧ ¬l2)) :
  (¬l1 ∧ l2 ∧ ¬l3) :=
by 
  sorry

end letters_identity_l2337_233720


namespace max_value_f_l2337_233738

noncomputable def f (x : ℝ) : ℝ := (4 * x - 4 * x^3) / (1 + 2 * x^2 + x^4)

theorem max_value_f : ∃ x : ℝ, (f x = 1) ∧ (∀ y : ℝ, f y ≤ 1) :=
sorry

end max_value_f_l2337_233738


namespace find_total_salary_l2337_233781

noncomputable def total_salary (salary_left : ℕ) : ℚ :=
  salary_left * (120 / 19)

theorem find_total_salary
  (food : ℚ) (house_rent : ℚ) (clothes : ℚ) (transport : ℚ) (remaining : ℕ) :
  food = 1 / 4 →
  house_rent = 1 / 8 →
  clothes = 3 / 10 →
  transport = 1 / 6 →
  remaining = 35000 →
  total_salary remaining = 210552.63 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end find_total_salary_l2337_233781


namespace medals_award_count_l2337_233722

theorem medals_award_count :
  let total_ways (n k : ℕ) := n.factorial / (n - k).factorial
  ∃ (award_ways : ℕ), 
    let no_americans := total_ways 6 3
    let one_american := 4 * 3 * total_ways 6 2
    award_ways = no_americans + one_american ∧
    award_ways = 480 :=
by
  sorry

end medals_award_count_l2337_233722


namespace total_amount_paid_is_correct_l2337_233724

-- Define the initial conditions
def tireA_price : ℕ := 75
def tireA_discount : ℕ := 20
def tireB_price : ℕ := 90
def tireB_discount : ℕ := 30
def tireC_price : ℕ := 120
def tireC_discount : ℕ := 45
def tireD_price : ℕ := 150
def tireD_discount : ℕ := 60
def installation_fee : ℕ := 15
def disposal_fee : ℕ := 5

-- Calculate the total amount paid
def total_paid : ℕ :=
  let tireA_total := (tireA_price - tireA_discount) + installation_fee + disposal_fee
  let tireB_total := (tireB_price - tireB_discount) + installation_fee + disposal_fee
  let tireC_total := (tireC_price - tireC_discount) + installation_fee + disposal_fee
  let tireD_total := (tireD_price - tireD_discount) + installation_fee + disposal_fee
  tireA_total + tireB_total + tireC_total + tireD_total

-- Statement of the theorem
theorem total_amount_paid_is_correct :
  total_paid = 360 :=
by
  -- proof goes here
  sorry

end total_amount_paid_is_correct_l2337_233724


namespace difference_of_sums_l2337_233776

noncomputable def sum_of_first_n_even (n : ℕ) : ℕ :=
  n * (n + 1)

noncomputable def sum_of_first_n_odd (n : ℕ) : ℕ :=
  n * n

theorem difference_of_sums : 
  sum_of_first_n_even 2004 - sum_of_first_n_odd 2003 = 6017 := 
by sorry

end difference_of_sums_l2337_233776


namespace lance_more_pebbles_l2337_233710

-- Given conditions
def candy_pebbles : ℕ := 4
def lance_pebbles : ℕ := 3 * candy_pebbles

-- Proof statement
theorem lance_more_pebbles : lance_pebbles - candy_pebbles = 8 :=
by
  sorry

end lance_more_pebbles_l2337_233710


namespace remainder_3_302_plus_302_div_by_3_151_plus_3_101_plus_1_l2337_233758

-- Definitions from the conditions
def a : ℕ := 3^302
def b : ℕ := 3^151 + 3^101 + 1

-- Theorem: Prove that the remainder when a + 302 is divided by b is 302.
theorem remainder_3_302_plus_302_div_by_3_151_plus_3_101_plus_1 :
  (a + 302) % b = 302 :=
by {
  sorry
}

end remainder_3_302_plus_302_div_by_3_151_plus_3_101_plus_1_l2337_233758


namespace find_y_l2337_233787

theorem find_y (x y : ℤ) 
  (h1 : x^2 + 4 = y - 2) 
  (h2 : x = 6) : 
  y = 42 := 
by 
  sorry

end find_y_l2337_233787


namespace common_ratio_is_half_l2337_233762

variable {a₁ q : ℝ}

-- Given the conditions of the geometric sequence

-- First condition
axiom h1 : a₁ + a₁ * q ^ 2 = 10

-- Second condition
axiom h2 : a₁ * q ^ 3 + a₁ * q ^ 5 = 5 / 4

-- Proving that the common ratio q is 1/2
theorem common_ratio_is_half : q = 1 / 2 :=
by
  -- The proof details will be filled in here.
  sorry

end common_ratio_is_half_l2337_233762


namespace opposite_of_neg3_l2337_233769

def opposite (a : Int) : Int := -a

theorem opposite_of_neg3 : opposite (-3) = 3 := by
  unfold opposite
  show (-(-3)) = 3
  sorry

end opposite_of_neg3_l2337_233769


namespace cos_two_pi_over_three_l2337_233761

theorem cos_two_pi_over_three : Real.cos (2 * Real.pi / 3) = -1 / 2 := 
by
  sorry

end cos_two_pi_over_three_l2337_233761


namespace first_term_and_common_difference_l2337_233792

theorem first_term_and_common_difference (a : ℕ → ℤ) (h : ∀ n, a n = 4 * n - 3) :
  a 1 = 1 ∧ (a 2 - a 1) = 4 :=
by
  sorry

end first_term_and_common_difference_l2337_233792


namespace find_angle_x_l2337_233768

theorem find_angle_x (x : ℝ) (h1 : x + x + 140 = 360) : x = 110 :=
by
  sorry

end find_angle_x_l2337_233768


namespace number_of_dolls_combined_l2337_233773

-- Defining the given conditions as variables
variables (aida sophie vera : ℕ)

-- Given conditions
def condition1 : Prop := aida = 2 * sophie
def condition2 : Prop := sophie = 2 * vera
def condition3 : Prop := vera = 20

-- The final proof statement we need to prove
theorem number_of_dolls_combined (h1 : condition1 aida sophie) (h2 : condition2 sophie vera) (h3 : condition3 vera) : 
  aida + sophie + vera = 140 :=
  by sorry

end number_of_dolls_combined_l2337_233773


namespace solution_set_of_inequality_l2337_233756

theorem solution_set_of_inequality (x : ℝ) : -x^2 + 3 * x - 2 > 0 ↔ 1 < x ∧ x < 2 :=
by
  sorry

end solution_set_of_inequality_l2337_233756


namespace Jenny_has_6_cards_l2337_233777

variable (J : ℕ)

noncomputable def Jenny_number := J
noncomputable def Orlando_number := J + 2
noncomputable def Richard_number := 3 * (J + 2)
noncomputable def Total_number := J + (J + 2) + 3 * (J + 2)

theorem Jenny_has_6_cards
  (h1 : Orlando_number J = J + 2)
  (h2 : Richard_number J = 3 * (J + 2))
  (h3 : Total_number J = 38) : J = 6 :=
by
  sorry

end Jenny_has_6_cards_l2337_233777


namespace pizza_slices_needed_l2337_233723

theorem pizza_slices_needed (couple_slices : ℕ) (children : ℕ) (children_slices : ℕ) (pizza_slices : ℕ)
    (hc : couple_slices = 3)
    (hcouple : children = 6)
    (hch : children_slices = 1)
    (hpizza : pizza_slices = 4) : 
    (2 * couple_slices + children * children_slices) / pizza_slices = 3 := 
by
    sorry

end pizza_slices_needed_l2337_233723


namespace number_of_arrangements_word_l2337_233778

noncomputable def factorial (n : Nat) : Nat := 
  if n = 0 then 1 else n * factorial (n - 1)

theorem number_of_arrangements_word (letters : List Char) (n : Nat) (r1 r2 r3 : Nat) 
  (h1 : letters = ['M', 'A', 'T', 'H', 'E', 'M', 'A', 'T', 'I', 'C', 'S'])
  (h2 : 2 = r1) (h3 : 2 = r2) (h4 : 2 = r3) :
  n = 11 → 
  factorial n / (factorial r1 * factorial r2 * factorial r3) = 4989600 := 
by
  sorry

end number_of_arrangements_word_l2337_233778


namespace fraction_addition_simplest_form_l2337_233766

theorem fraction_addition_simplest_form :
  (7 / 8) + (3 / 5) = 59 / 40 :=
by sorry

end fraction_addition_simplest_form_l2337_233766


namespace father_age_when_rachel_is_25_l2337_233763

-- Definitions for Rachel's age, Grandfather's age, Mother's age, and Father's age
def rachel_age : ℕ := 12
def grandfather_age : ℕ := 7 * rachel_age
def mother_age : ℕ := grandfather_age / 2
def father_age : ℕ := mother_age + 5
def years_until_rachel_is_25 : ℕ := 25 - rachel_age
def fathers_age_when_rachel_is_25 : ℕ := father_age + years_until_rachel_is_25

-- Theorem to prove that Rachel's father will be 60 years old when Rachel is 25 years old
theorem father_age_when_rachel_is_25 : fathers_age_when_rachel_is_25 = 60 := by
  sorry

end father_age_when_rachel_is_25_l2337_233763


namespace at_least_one_not_less_than_2_l2337_233701

-- Definitions for the problem
variables {a b c : ℝ}
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

-- The Lean 4 statement for the problem
theorem at_least_one_not_less_than_2 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (2 ≤ a + 1/b) ∨ (2 ≤ b + 1/c) ∨ (2 ≤ c + 1/a) :=
sorry

end at_least_one_not_less_than_2_l2337_233701


namespace pencils_combined_length_l2337_233719

theorem pencils_combined_length (length_pencil1 length_pencil2 : Nat) (h1 : length_pencil1 = 12) (h2 : length_pencil2 = 12) :
  length_pencil1 + length_pencil2 = 24 := by
  sorry

end pencils_combined_length_l2337_233719


namespace eight_n_plus_nine_is_perfect_square_l2337_233728

theorem eight_n_plus_nine_is_perfect_square 
  (n : ℕ) (N : ℤ) 
  (hN : N = 2 ^ (4 * n + 1) - 4 ^ n - 1)
  (hdiv : 9 ∣ N) :
  ∃ k : ℤ, 8 * N + 9 = k ^ 2 :=
by
  sorry

end eight_n_plus_nine_is_perfect_square_l2337_233728


namespace minimum_width_l2337_233742

theorem minimum_width (A l w : ℝ) (hA : A >= 150) (hl : l = 2 * w) (hA_def : A = w * l) : 
  w >= 5 * Real.sqrt 3 := 
  by
    -- Using the given conditions, we can prove that w >= 5 * sqrt(3)
    sorry

end minimum_width_l2337_233742


namespace elf_distribution_finite_l2337_233731

theorem elf_distribution_finite (infinite_rubies : ℕ → ℕ) (infinite_sapphires : ℕ → ℕ) :
  (∃ n : ℕ, ∀ i j : ℕ, i < n → j < n → (infinite_rubies i > infinite_rubies j → infinite_sapphires i < infinite_sapphires j) ∧
  (infinite_rubies i ≥ infinite_rubies j → infinite_sapphires i < infinite_sapphires j)) ↔
  ∃ k : ℕ, ∀ j : ℕ, j < k :=
sorry

end elf_distribution_finite_l2337_233731


namespace find_cost_price_l2337_233749

variables (SP CP : ℝ)
variables (discount profit : ℝ)
variable (h1 : SP = 24000)
variable (h2 : discount = 0.10)
variable (h3 : profit = 0.08)

theorem find_cost_price 
  (h1 : SP = 24000)
  (h2 : discount = 0.10)
  (h3 : profit = 0.08)
  (h4 : SP * (1 - discount) = CP * (1 + profit)) :
  CP = 20000 := 
by
  sorry

end find_cost_price_l2337_233749


namespace second_pipe_fill_time_l2337_233747

theorem second_pipe_fill_time :
  ∃ x : ℝ, x ≠ 0 ∧ (1 / 10 + 1 / x - 1 / 20 = 1 / 7.5) ∧ x = 60 :=
by
  sorry

end second_pipe_fill_time_l2337_233747


namespace powers_of_i_l2337_233700

theorem powers_of_i (i : ℂ) (h1 : i^1 = i) (h2 : i^2 = -1) (h3 : i^3 = -i) (h4 : i^4 = 1) : 
  i^22 + i^222 = -2 :=
by {
  -- Proof will go here
  sorry
}

end powers_of_i_l2337_233700


namespace necessary_condition_for_A_l2337_233727

variable {x a : ℝ}

def A : Set ℝ := { x | (x - 2) / (x + 1) ≤ 0 }

theorem necessary_condition_for_A (x : ℝ) (h : x ∈ A) (ha : x ≥ a) : a ≤ -1 :=
sorry

end necessary_condition_for_A_l2337_233727


namespace probability_at_least_one_five_or_six_l2337_233740

theorem probability_at_least_one_five_or_six
  (P_neither_five_nor_six: ℚ)
  (h: P_neither_five_nor_six = 4 / 9) :
  (1 - P_neither_five_nor_six) = 5 / 9 :=
by
  sorry

end probability_at_least_one_five_or_six_l2337_233740


namespace probability_passing_exam_l2337_233785

-- Define probabilities for sets A, B, and C, and passing conditions
def P_A := 0.3
def P_B := 0.3
def P_C := 1 - P_A - P_B
def P_D_given_A := 0.8
def P_D_given_B := 0.6
def P_D_given_C := 0.8

-- Total probability of passing
def P_D := P_A * P_D_given_A + P_B * P_D_given_B + P_C * P_D_given_C

-- Proof that the total probability of passing is 0.74
theorem probability_passing_exam : P_D = 0.74 :=
by
  -- (skip the proof steps)
  sorry

end probability_passing_exam_l2337_233785
