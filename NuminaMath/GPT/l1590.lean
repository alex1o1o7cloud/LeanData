import Mathlib

namespace find_integers_l1590_159026

theorem find_integers (a b : ℤ) (h1 : a * b = a + b) (h2 : a * b = a - b) : a = 0 ∧ b = 0 :=
by 
  sorry

end find_integers_l1590_159026


namespace fraction_powers_sum_l1590_159012

theorem fraction_powers_sum : 
  ( (5:ℚ) / (3:ℚ) )^6 + ( (2:ℚ) / (3:ℚ) )^6 = (15689:ℚ) / (729:ℚ) :=
by
  sorry

end fraction_powers_sum_l1590_159012


namespace average_time_per_other_class_l1590_159011

theorem average_time_per_other_class (school_hours : ℚ) (num_classes : ℕ) (hist_chem_hours : ℚ)
  (total_school_time_minutes : ℕ) (hist_chem_time_minutes : ℕ) (num_other_classes : ℕ)
  (other_classes_time_minutes : ℕ) (average_time_other_classes : ℕ) :
  school_hours = 7.5 →
  num_classes = 7 →
  hist_chem_hours = 1.5 →
  total_school_time_minutes = school_hours * 60 →
  hist_chem_time_minutes = hist_chem_hours * 60 →
  other_classes_time_minutes = total_school_time_minutes - hist_chem_time_minutes →
  num_other_classes = num_classes - 2 →
  average_time_other_classes = other_classes_time_minutes / num_other_classes →
  average_time_other_classes = 72 :=
by
  intros
  sorry

end average_time_per_other_class_l1590_159011


namespace rightmost_three_digits_of_7_pow_1987_l1590_159097

theorem rightmost_three_digits_of_7_pow_1987 :
  7^1987 % 1000 = 543 :=
by
  sorry

end rightmost_three_digits_of_7_pow_1987_l1590_159097


namespace particle_max_height_l1590_159051

noncomputable def max_height (r ω g : ℝ) : ℝ :=
  (r * ω + g / ω) ^ 2 / (2 * g)

theorem particle_max_height (r ω g : ℝ) (h : ω > Real.sqrt (g / r)) :
    max_height r ω g = (r * ω + g / ω) ^ 2 / (2 * g) :=
sorry

end particle_max_height_l1590_159051


namespace apples_in_pile_l1590_159061

-- Define the initial number of apples in the pile
def initial_apples : ℕ := 8

-- Define the number of added apples
def added_apples : ℕ := 5

-- Define the total number of apples
def total_apples : ℕ := initial_apples + added_apples

-- Prove that the total number of apples is 13
theorem apples_in_pile : total_apples = 13 :=
by
  sorry

end apples_in_pile_l1590_159061


namespace final_volume_of_water_in_tank_l1590_159062

theorem final_volume_of_water_in_tank (capacity : ℕ) (initial_fraction full_volume : ℕ)
  (percent_empty percent_fill final_volume : ℕ) :
  capacity = 8000 →
  initial_fraction = 3 / 4 →
  percent_empty = 40 →
  percent_fill = 30 →
  full_volume = capacity * initial_fraction →
  final_volume = full_volume - (full_volume * percent_empty / 100) + ((full_volume - (full_volume * percent_empty / 100)) * percent_fill / 100) →
  final_volume = 4680 :=
by
  sorry

end final_volume_of_water_in_tank_l1590_159062


namespace consecutive_days_without_meeting_l1590_159081

/-- In March 1987, there are 31 days, starting on a Sunday.
There are 11 club meetings to be held, and no meetings are on Saturdays or Sundays.
This theorem proves that there will be at least three consecutive days without a meeting. -/
theorem consecutive_days_without_meeting (meetings : Finset ℕ) :
  (∀ x ∈ meetings, 1 ≤ x ∧ x ≤ 31 ∧ ¬ ∃ k, x = 7 * k + 1 ∨ x = 7 * k + 2) →
  meetings.card = 11 →
  ∃ i, 1 ≤ i ∧ i + 2 ≤ 31 ∧ ¬ (i ∈ meetings ∨ (i + 1) ∈ meetings ∨ (i + 2) ∈ meetings) :=
by
  sorry

end consecutive_days_without_meeting_l1590_159081


namespace sequence_first_five_terms_l1590_159045

noncomputable def a_n (n : ℕ) : ℤ := (-1) ^ n + (n : ℤ)

theorem sequence_first_five_terms :
  a_n 1 = 0 ∧
  a_n 2 = 3 ∧
  a_n 3 = 2 ∧
  a_n 4 = 5 ∧
  a_n 5 = 4 :=
by
  sorry

end sequence_first_five_terms_l1590_159045


namespace trains_meet_in_32_seconds_l1590_159049

noncomputable def length_first_train : ℕ := 400
noncomputable def length_second_train : ℕ := 200
noncomputable def initial_distance : ℕ := 200

noncomputable def speed_first_train : ℕ := 15
noncomputable def speed_second_train : ℕ := 10

noncomputable def relative_speed : ℕ := speed_first_train + speed_second_train
noncomputable def total_distance : ℕ := length_first_train + length_second_train + initial_distance
noncomputable def time_to_meet := total_distance / relative_speed

theorem trains_meet_in_32_seconds : time_to_meet = 32 := by
  sorry

end trains_meet_in_32_seconds_l1590_159049


namespace power_of_three_l1590_159042

theorem power_of_three (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_mult : (3^a) * (3^b) = 81) : (3^a)^b = 81 :=
sorry

end power_of_three_l1590_159042


namespace find_alpha_l1590_159048

theorem find_alpha (α : ℝ) (h : Real.sin α * (1 + Real.sqrt 3 * Real.tan (10 * Real.pi / 180)) = 1) :
  α = 13 * Real.pi / 18 :=
sorry

end find_alpha_l1590_159048


namespace inequality_proof_l1590_159039

theorem inequality_proof (m n : ℝ) (h1 : m < n) (h2 : n < 0) : (n / m) + (m / n) > 2 :=
sorry

end inequality_proof_l1590_159039


namespace solve_for_m_l1590_159068

-- Define the operation ◎ for real numbers a and b
def op (a b : ℝ) : ℝ := (a + b)^2 - (a - b)^2

-- Lean statement for the proof problem
theorem solve_for_m (m : ℝ) (h : op (m + 1) (m - 2) = 16) : m = 3 ∨ m = -2 :=
sorry

end solve_for_m_l1590_159068


namespace graph_is_point_l1590_159074

theorem graph_is_point : ∀ x y : ℝ, x^2 + 3 * y^2 - 4 * x - 6 * y + 7 = 0 ↔ (x = 2 ∧ y = 1) :=
by
  sorry

end graph_is_point_l1590_159074


namespace triangle_cosine_identity_l1590_159087

theorem triangle_cosine_identity
  (a b c : ℝ)
  (α β γ : ℝ)
  (hα : α = Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c)))
  (hβ : β = Real.arccos ((a^2 + c^2 - b^2) / (2 * a * c)))
  (hγ : γ = Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b)))
  (habc_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  (b / c + c / b) * Real.cos α + 
  (c / a + a / c) * Real.cos β + 
  (a / b + b / a) * Real.cos γ = 3 := 
sorry

end triangle_cosine_identity_l1590_159087


namespace am_gm_inequality_l1590_159069

theorem am_gm_inequality (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) : 
  (a^2 * b + b^2 * c + c^2 * a) * (a * b^2 + b * c^2 + c * a^2) ≥ 9 * (a * b * c)^2 :=
by
  sorry

end am_gm_inequality_l1590_159069


namespace r_daily_earnings_l1590_159099

-- Given conditions as definitions
def daily_earnings (P Q R : ℕ) : Prop :=
(P + Q + R) * 9 = 1800 ∧ (P + R) * 5 = 600 ∧ (Q + R) * 7 = 910

-- Theorem statement corresponding to the problem
theorem r_daily_earnings : ∃ R : ℕ, ∀ P Q : ℕ, daily_earnings P Q R → R = 50 :=
by sorry

end r_daily_earnings_l1590_159099


namespace value_of_m_l1590_159044

theorem value_of_m 
  (m : ℤ) 
  (h : ∀ x : ℤ, x^2 - 2 * (m + 1) * x + 16 = (x - 4)^2) : 
  m = 3 := 
sorry

end value_of_m_l1590_159044


namespace distribute_books_l1590_159017

theorem distribute_books (m n : ℕ) (h1 : m = 3*n + 8) (h2 : ∃k, m = 5*k + r ∧ r < 5 ∧ r > 0) : 
  n = 5 ∨ n = 6 :=
by sorry

end distribute_books_l1590_159017


namespace polar_to_line_distance_l1590_159034

theorem polar_to_line_distance : 
  let point_polar := (2, Real.pi / 3)
  let line_polar := (2, 0)  -- Corresponding (rho, theta) for the given line
  let point_rect := (2 * Real.cos (Real.pi / 3), 2 * Real.sin (Real.pi / 3))
  let line_rect := 2  -- x = 2 in rectangular coordinates
  let distance := abs (line_rect - point_rect.1)
  distance = 1 := by
{
  sorry
}

end polar_to_line_distance_l1590_159034


namespace find_k_value_l1590_159002

theorem find_k_value :
  ∃ k : ℝ, (∀ x : ℝ, 1 ≤ x^2 - 3 * x + k ∧ x^2 - 3 * x + k ≤ 5) ∧ 
          (∃ a b : ℝ, b - a = 8 ∧ (∀ x : ℝ, a ≤ x ∧ x ≤ b → 1 ≤ x^2 - 3 * x + k ∧ x^2 - 3 * x + k ≤ 5)) ∧ 
          k = 9 / 4 :=
sorry

end find_k_value_l1590_159002


namespace term_5th_in_sequence_l1590_159010

theorem term_5th_in_sequence : 
  ∃ n : ℕ, n = 5 ∧ ( ∃ t : ℕ, t = 28 ∧ 3^t ∈ { 3^(7 * (k - 1)) | k : ℕ } ) :=
by {
  sorry
}

end term_5th_in_sequence_l1590_159010


namespace Tod_drove_time_l1590_159098

section
variable (distance_north: ℕ) (distance_west: ℕ) (speed: ℕ)

theorem Tod_drove_time :
  distance_north = 55 → distance_west = 95 → speed = 25 → 
  (distance_north + distance_west) / speed = 6 :=
by
  intros
  sorry
end

end Tod_drove_time_l1590_159098


namespace brittany_age_after_vacation_l1590_159003

-- Definitions of the conditions
def rebecca_age : ℕ := 25
def age_difference : ℕ := 3
def vacation_years : ℕ := 4

-- Prove the main statement
theorem brittany_age_after_vacation : rebecca_age + age_difference + vacation_years = 32 := by
  sorry

end brittany_age_after_vacation_l1590_159003


namespace four_consecutive_none_multiple_of_5_l1590_159057

theorem four_consecutive_none_multiple_of_5 (n : ℤ) :
  (∃ k : ℤ, n + (n + 1) + (n + 2) + (n + 3) = 5 * k) →
  ¬ (∃ m : ℤ, (n = 5 * m) ∨ (n + 1 = 5 * m) ∨ (n + 2 = 5 * m) ∨ (n + 3 = 5 * m)) :=
by sorry

end four_consecutive_none_multiple_of_5_l1590_159057


namespace original_price_is_135_l1590_159076

-- Problem Statement:
variable (P : ℝ)  -- Let P be the original price of the potion

-- Conditions
axiom potion_cost : (1 / 15) * P = 9

-- Proof Goal
theorem original_price_is_135 : P = 135 :=
by
  sorry

end original_price_is_135_l1590_159076


namespace steven_erasers_l1590_159030

theorem steven_erasers (skittles erasers groups items_per_group total_items : ℕ)
  (h1 : skittles = 4502)
  (h2 : groups = 154)
  (h3 : items_per_group = 57)
  (h4 : total_items = groups * items_per_group)
  (h5 : total_items - skittles = erasers) :
  erasers = 4276 :=
by
  sorry

end steven_erasers_l1590_159030


namespace fuel_efficiency_problem_l1590_159007

theorem fuel_efficiency_problem :
  let F_highway := 30
  let F_urban := 25
  let F_hill := 20
  let D_highway := 100
  let D_urban := 60
  let D_hill := 40
  let gallons_highway := D_highway / F_highway
  let gallons_urban := D_urban / F_urban
  let gallons_hill := D_hill / F_hill
  let total_gallons := gallons_highway + gallons_urban + gallons_hill
  total_gallons = 7.73 := 
by 
  sorry

end fuel_efficiency_problem_l1590_159007


namespace additional_people_needed_l1590_159093

-- Define the conditions
def num_people_initial := 9
def work_done_initial := 3 / 5
def days_initial := 14
def days_remaining := 4

-- Calculated values based on conditions
def work_rate_per_person : ℚ :=
  work_done_initial / (num_people_initial * days_initial)

def work_remaining : ℚ := 1 - work_done_initial

def total_people_needed : ℚ :=
  work_remaining / (work_rate_per_person * days_remaining)

-- Formulate the statement to prove
theorem additional_people_needed :
  total_people_needed - num_people_initial = 12 :=
by
  sorry

end additional_people_needed_l1590_159093


namespace simplify_expression_l1590_159065

theorem simplify_expression (x : ℝ) : (2 * x)^5 + (3 * x) * x^4 + 2 * x^3 = 35 * x^5 + 2 * x^3 :=
by
  sorry

end simplify_expression_l1590_159065


namespace is_linear_equation_D_l1590_159079

theorem is_linear_equation_D :
  (∀ (x y : ℝ), 2 * x + 3 * y = 7 → false) ∧
  (∀ (x : ℝ), 3 * x ^ 2 = 3 → false) ∧
  (∀ (x : ℝ), 6 = 2 / x - 1 → false) ∧
  (∀ (x : ℝ), 2 * x - 1 = 20 → true) 
:= by {
  sorry
}

end is_linear_equation_D_l1590_159079


namespace finite_steps_iff_power_of_2_l1590_159084

-- Define the conditions of the problem
def S (k n : ℕ) : ℕ := (k * (k + 1) / 2) % n

-- Define the predicate to check if the game finishes in finite number of steps
def game_completes (n : ℕ) : Prop :=
  ∃ k : ℕ, ∀ i : ℕ, i < n → S (k + i) n ≠ S k n

-- The main statement to prove
theorem finite_steps_iff_power_of_2 (n : ℕ) : game_completes n ↔ ∃ t : ℕ, n = 2^t :=
sorry  -- Placeholder for the proof

end finite_steps_iff_power_of_2_l1590_159084


namespace standard_equation_of_parabola_l1590_159095

theorem standard_equation_of_parabola (focus : ℝ × ℝ): 
  (focus.1 - 2 * focus.2 - 4 = 0) → 
  ((focus = (4, 0) → (∃ a : ℝ, ∀ x y : ℝ, y^2 = 4 * a * x)) ∨
   (focus = (0, -2) → (∃ b : ℝ, ∀ x y : ℝ, x^2 = 4 * b * y))) :=
by
  sorry

end standard_equation_of_parabola_l1590_159095


namespace math_olympiad_problem_l1590_159055

theorem math_olympiad_problem (students : Fin 11 → Finset (Fin n)) (h_solved : ∀ i, (students i).card = 3)
  (h_distinct : ∀ i j, i ≠ j → ∃ p, p ∈ students i ∧ p ∉ students j) : 
  6 ≤ n := 
sorry

end math_olympiad_problem_l1590_159055


namespace trader_profit_percentage_l1590_159028

theorem trader_profit_percentage
  (P : ℝ)
  (h1 : P > 0)
  (buy_price : ℝ := 0.80 * P)
  (sell_price : ℝ := 1.60 * P) :
  (sell_price - P) / P * 100 = 60 := 
by sorry

end trader_profit_percentage_l1590_159028


namespace jon_awake_hours_per_day_l1590_159072

def regular_bottle_size : ℕ := 16
def larger_bottle_size : ℕ := 20
def weekly_fluid_intake : ℕ := 728
def larger_bottle_daily_intake : ℕ := 40
def larger_bottle_weekly_intake : ℕ := 280
def regular_bottle_weekly_intake : ℕ := 448
def regular_bottles_per_week : ℕ := 28
def regular_bottles_per_day : ℕ := 4
def hours_per_bottle : ℕ := 4

theorem jon_awake_hours_per_day
  (h1 : jon_drinks_regular_bottle_every_4_hours)
  (h2 : jon_drinks_two_larger_bottles_daily)
  (h3 : jon_drinks_728_ounces_per_week) :
  jon_is_awake_hours_per_day = 16 :=
by
  sorry

def jon_drinks_regular_bottle_every_4_hours : Prop :=
  ∀ hours : ℕ, hours * regular_bottle_size / hours_per_bottle = 1

def jon_drinks_two_larger_bottles_daily : Prop :=
  larger_bottle_size = (regular_bottle_size * 5) / 4 ∧ 
  larger_bottle_daily_intake = 2 * larger_bottle_size

def jon_drinks_728_ounces_per_week : Prop :=
  weekly_fluid_intake = 728

def jon_is_awake_hours_per_day : ℕ :=
  regular_bottles_per_day * hours_per_bottle

end jon_awake_hours_per_day_l1590_159072


namespace cost_price_equals_selling_price_l1590_159014

theorem cost_price_equals_selling_price (C : ℝ) (x : ℝ) (h1 : 20 * C = 1.25 * C * x) : x = 16 :=
by
  -- This proof is omitted at the moment
  sorry

end cost_price_equals_selling_price_l1590_159014


namespace minimum_value_l1590_159020

-- Define the expression E(a, b, c)
def E (a b c : ℝ) : ℝ := a^2 + 8 * a * b + 24 * b^2 + 16 * b * c + 6 * c^2

-- State the minimum value theorem
theorem minimum_value (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  E a b c = 18 :=
sorry

end minimum_value_l1590_159020


namespace pool_water_after_eight_hours_l1590_159040

-- Define the conditions
def hour1_fill_rate := 8
def hour2_and_hour3_fill_rate := 10
def hour4_and_hour5_fill_rate := 14
def hour6_fill_rate := 12
def hour7_fill_rate := 12
def hour8_fill_rate := 12
def hour7_leak := -8
def hour8_leak := -5

-- Calculate the water added in each time period
def water_added := hour1_fill_rate +
                   (hour2_and_hour3_fill_rate * 2) +
                   (hour4_and_hour5_fill_rate * 2) +
                   (hour6_fill_rate + hour7_fill_rate + hour8_fill_rate)

-- Calculate the water lost due to leaks
def water_lost := hour7_leak + hour8_leak  -- Note: Leaks are already negative

-- The final calculation: total water added minus total water lost
def final_water := water_added + water_lost

theorem pool_water_after_eight_hours : final_water = 79 :=
by {
  -- proof steps to check equality are omitted here
  sorry
}

end pool_water_after_eight_hours_l1590_159040


namespace increasing_C_l1590_159070

theorem increasing_C (e R r : ℝ) (n : ℕ) (h₁ : 0 < e) (h₂ : 0 < R) (h₃ : 0 < r) (h₄ : 0 < n) :
    ∀ n1 n2 : ℕ, n1 < n2 → (e^2 * n1) / (R + n1 * r) < (e^2 * n2) / (R + n2 * r) :=
by
  sorry

end increasing_C_l1590_159070


namespace wilson_buys_3_bottles_of_cola_l1590_159075

theorem wilson_buys_3_bottles_of_cola
    (num_hamburgers : ℕ := 2) 
    (cost_per_hamburger : ℕ := 5) 
    (cost_per_cola : ℕ := 2) 
    (discount : ℕ := 4) 
    (total_paid : ℕ := 12) :
    num_hamburgers * cost_per_hamburger - discount + x * cost_per_cola = total_paid → x = 3 :=
by
  sorry

end wilson_buys_3_bottles_of_cola_l1590_159075


namespace expected_steps_unit_interval_l1590_159038

noncomputable def expected_steps_to_color_interval : ℝ := 
  -- Placeholder for the function calculating expected steps
  sorry 

theorem expected_steps_unit_interval : expected_steps_to_color_interval = 5 :=
  sorry

end expected_steps_unit_interval_l1590_159038


namespace inequality_no_real_solutions_l1590_159060

theorem inequality_no_real_solutions (a b : ℝ) 
  (h : ∀ x : ℝ, a * Real.cos x + b * Real.cos (3 * x) ≤ 1) : 
  |b| ≤ 1 :=
sorry

end inequality_no_real_solutions_l1590_159060


namespace closest_point_on_parabola_l1590_159077

/-- The coordinates of the point on the parabola y^2 = x that is closest to the line x - 2y + 4 = 0 are (1,1). -/
theorem closest_point_on_parabola (y : ℝ) (x : ℝ) (h_parabola : y^2 = x) (h_line : x - 2*y + 4 = 0) :
  (x = 1 ∧ y = 1) :=
sorry

end closest_point_on_parabola_l1590_159077


namespace car_maintenance_fraction_l1590_159054

variable (p : ℝ) (f : ℝ)

theorem car_maintenance_fraction (hp : p = 5200)
  (he : p - f * p - (p - 320) = 200) : f = 3 / 130 :=
by
  have hp_pos : p ≠ 0 := by linarith [hp]
  sorry

end car_maintenance_fraction_l1590_159054


namespace circle_equation_l1590_159032

theorem circle_equation :
  ∃ M : ℝ × ℝ, (2 * M.1 + M.2 - 1 = 0) ∧
    (∃ r : ℝ, r ≥ 0 ∧ 
      ((3 - M.1)^2 + (0 - M.2)^2 = r^2) ∧
      ((0 - M.1)^2 + (1 - M.2)^2 = r^2)) ∧
    (∃ x y : ℝ, ((x - 1)^2 + (y + 1)^2 = 5)) := 
sorry

end circle_equation_l1590_159032


namespace f_eq_f_at_neg_one_f_at_neg_500_l1590_159091

noncomputable def f : ℝ → ℝ := sorry

theorem f_eq : ∀ x y : ℝ, f (x * y) + x = x * f y + f x := sorry
theorem f_at_neg_one : f (-1) = 1 := sorry

theorem f_at_neg_500 : f (-500) = 999 := sorry

end f_eq_f_at_neg_one_f_at_neg_500_l1590_159091


namespace find_PS_eq_13point625_l1590_159016

theorem find_PS_eq_13point625 (PQ PR QR : ℝ) (h : ℝ) (QS SR : ℝ)
  (h_QS : QS^2 = 225 - h^2)
  (h_SR : SR^2 = 400 - h^2)
  (h_ratio : QS / SR = 3 / 7) :
  PS = 13.625 :=
by
  sorry

end find_PS_eq_13point625_l1590_159016


namespace find_f_of_neg_1_l1590_159090

-- Define the conditions
variables (a b c : ℝ)
variables (g f : ℝ → ℝ)
axiom g_definition : ∀ x, g x = x^3 + a*x^2 + 2*x + 15
axiom f_definition : ∀ x, f x = x^4 + x^3 + b*x^2 + 150*x + c

axiom g_has_distinct_roots : ∃ r1 r2 r3 : ℝ, r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3 ∧ ∀ x, g x = 0 ↔ (x = r1 ∨ x = r2 ∨ x = r3)
axiom roots_of_g_are_roots_of_f : ∀ x, g x = 0 → f x = 0

-- Prove the value of f(-1) given the conditions
theorem find_f_of_neg_1 (a : ℝ) (b : ℝ) (c : ℝ) (g f : ℝ → ℝ)
  (h_g_def : ∀ x, g x = x^3 + a*x^2 + 2*x + 15)
  (h_f_def : ∀ x, f x = x^4 + x^3 + b*x^2 + 150*x + c)
  (h_g_has_distinct_roots : ∃ r1 r2 r3 : ℝ, r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3 ∧ ∀ x, g x = 0 ↔ (x = r1 ∨ x = r2 ∨ x = r3))
  (h_roots : ∀ x, g x = 0 → f x = 0) :
  f (-1) = 3733.25 := 
by {
  sorry
}

end find_f_of_neg_1_l1590_159090


namespace total_books_after_donations_l1590_159036

variable (Boris_books : Nat := 24)
variable (Cameron_books : Nat := 30)

theorem total_books_after_donations :
  (Boris_books - Boris_books / 4) + (Cameron_books - Cameron_books / 3) = 38 := by
  sorry

end total_books_after_donations_l1590_159036


namespace repeating_decimal_to_fraction_l1590_159006

theorem repeating_decimal_to_fraction :
  let x := 0.431431431 + 0.000431431431 + 0.000000431431431
  let y := 0.4 + x
  y = 427 / 990 :=
by
  sorry

end repeating_decimal_to_fraction_l1590_159006


namespace shopkeeper_total_profit_percentage_l1590_159009

noncomputable def profit_percentage (actual_weight faulty_weight ratio : ℕ) : ℝ :=
  (actual_weight - faulty_weight) / actual_weight * 100 * ratio

noncomputable def total_profit_percentage (ratios profits : List ℝ) : ℝ :=
  (List.sum (List.zipWith (· * ·) ratios profits)) / (List.sum ratios)

theorem shopkeeper_total_profit_percentage :
  let actual_weight := 1000
  let faulty_weights := [900, 850, 950]
  let profit_percentages := [10, 15, 5]
  let ratios := [3, 2, 1]
  total_profit_percentage ratios profit_percentages = 10.83 :=
by
  sorry

end shopkeeper_total_profit_percentage_l1590_159009


namespace marcie_cups_coffee_l1590_159088

theorem marcie_cups_coffee (S M T : ℕ) (h1 : S = 6) (h2 : S + M = 8) : M = 2 :=
by
  sorry

end marcie_cups_coffee_l1590_159088


namespace matchsticks_in_20th_stage_l1590_159019

-- Define the first term and common difference
def first_term : ℕ := 4
def common_difference : ℕ := 3

-- Define the mathematical function for the n-th term of the arithmetic sequence
def num_matchsticks (n : ℕ) : ℕ :=
  first_term + (n - 1) * common_difference

-- State the theorem to prove the number of matchsticks in the 20th stage
theorem matchsticks_in_20th_stage : num_matchsticks 20 = 61 :=
by
  -- Proof skipped
  sorry

end matchsticks_in_20th_stage_l1590_159019


namespace power_add_one_eq_twice_l1590_159052

theorem power_add_one_eq_twice (a b : ℕ) (h : 2^a = b) : 2^(a + 1) = 2 * b := by
  sorry

end power_add_one_eq_twice_l1590_159052


namespace prove_Praveen_present_age_l1590_159056

-- Definitions based on the conditions identified in a)
def PraveenAge (P : ℝ) := P + 10 = 3 * (P - 3)

-- The equivalent proof problem statement
theorem prove_Praveen_present_age : ∃ P : ℝ, PraveenAge P ∧ P = 9.5 :=
by
  sorry

end prove_Praveen_present_age_l1590_159056


namespace distinct_sets_count_l1590_159047

noncomputable def num_distinct_sets : ℕ :=
  let product : ℕ := 11 * 21 * 31 * 41 * 51 * 61
  728

theorem distinct_sets_count : 
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a * b * c = 11 * 21 * 31 * 41 * 51 * 61 ∧ num_distinct_sets = 728 :=
sorry

end distinct_sets_count_l1590_159047


namespace routes_from_A_to_B_in_4_by_3_grid_l1590_159029

-- Problem: Given a 4 by 3 rectangular grid, and movement allowing only right (R) or down (D),
-- prove that the number of different routes from point A to point B is 35.
def routes_4_by_3 : ℕ :=
  let n_moves := 3 + 4  -- Total moves required are 3 Rs and 4 Ds
  let r_moves := 3      -- Number of Right moves (R)
  Nat.choose (n_moves) (r_moves) -- Number of ways to choose 3 Rs from 7 moves

theorem routes_from_A_to_B_in_4_by_3_grid : routes_4_by_3 = 35 := by {
  sorry -- Proof omitted
}

end routes_from_A_to_B_in_4_by_3_grid_l1590_159029


namespace isosceles_triangle_perimeter_l1590_159033

theorem isosceles_triangle_perimeter {a b : ℝ} (h1 : a = 6) (h2 : b = 3) (h3 : a ≠ b) :
  (2 * b + a = 15) :=
by
  sorry

end isosceles_triangle_perimeter_l1590_159033


namespace domain_of_c_x_l1590_159096

theorem domain_of_c_x (k : ℝ) :
  (∀ x : ℝ, -5 * x ^ 2 + 3 * x + k ≠ 0) ↔ k < -9 / 20 := 
sorry

end domain_of_c_x_l1590_159096


namespace base_k_addition_is_ten_l1590_159024

theorem base_k_addition_is_ten :
  ∃ k : ℕ, (k > 4) ∧ (5 * k^3 + 3 * k^2 + 4 * k + 2 + 6 * k^3 + 4 * k^2 + 2 * k + 1 = 1 * k^4 + 4 * k^3 + 1 * k^2 + 6 * k + 3) ∧ k = 10 :=
by
  sorry

end base_k_addition_is_ten_l1590_159024


namespace boxes_to_fill_l1590_159004

theorem boxes_to_fill (total_boxes filled_boxes : ℝ) (h₁ : total_boxes = 25.75) (h₂ : filled_boxes = 17.5) : 
  total_boxes - filled_boxes = 8.25 := 
by
  sorry

end boxes_to_fill_l1590_159004


namespace new_person_weight_l1590_159031

theorem new_person_weight (W : ℝ) (N : ℝ)
  (h1 : ∀ avg_increase : ℝ, avg_increase = 2.5 → N = 55) 
  (h2 : ∀ original_weight : ℝ, original_weight = 35) 
  : N = 55 := 
by 
  sorry

end new_person_weight_l1590_159031


namespace last_two_digits_of_9_power_h_are_21_l1590_159018

def a := 1
def b := 2^a
def c := 3^b
def d := 4^c
def e := 5^d
def f := 6^e
def g := 7^f
def h := 8^g

theorem last_two_digits_of_9_power_h_are_21 : (9^h) % 100 = 21 := by
  sorry

end last_two_digits_of_9_power_h_are_21_l1590_159018


namespace arithmetic_sequence_common_difference_l1590_159046

variable (a_n : ℕ → ℝ)

theorem arithmetic_sequence_common_difference
  (h_arith : ∀ n, a_n (n + 1) = a_n n + d)
  (h_non_zero : d ≠ 0)
  (h_sum : a_n 1 + a_n 2 + a_n 3 = 9)
  (h_geom : a_n 2 ^ 2 = a_n 1 * a_n 5) :
  d = 2 :=
by
  sorry

end arithmetic_sequence_common_difference_l1590_159046


namespace Xiao_Ming_min_steps_l1590_159094

-- Problem statement: Prove that the minimum number of steps Xiao Ming needs to move from point A to point B is 5,
-- given his movement pattern and the fact that he can reach eight different positions from point C.

def min_steps_from_A_to_B : ℕ :=
  5

theorem Xiao_Ming_min_steps (A B C : Type) (f : A → B → C) : 
  (min_steps_from_A_to_B = 5) :=
by
  sorry

end Xiao_Ming_min_steps_l1590_159094


namespace diophantine_solution_range_l1590_159080

theorem diophantine_solution_range {a b c n : ℤ} (coprime_ab : Int.gcd a b = 1) :
  (∃ (x y : ℕ), a * x + b * y = c ∧ ∀ k : ℤ, k ≥ 1 → ∃ (x y : ℕ), a * (x + k * b) + b * (y - k * a) = c) → 
  ((n - 1) * a * b + a + b ≤ c ∧ c ≤ (n + 1) * a * b) :=
sorry

end diophantine_solution_range_l1590_159080


namespace fraction_addition_l1590_159086

theorem fraction_addition : (3/4) / (5/8) + (1/2) = 17/10 := by
  sorry

end fraction_addition_l1590_159086


namespace calories_per_candy_bar_l1590_159063

theorem calories_per_candy_bar (total_calories : ℕ) (number_of_bars : ℕ) 
  (h : total_calories = 341) (n : number_of_bars = 11) : (total_calories / number_of_bars = 31) :=
by
  sorry

end calories_per_candy_bar_l1590_159063


namespace price_reduction_l1590_159050

theorem price_reduction (x : ℝ) (h : 0 ≤ x ∧ x ≤ 1) : 150 * (1 - x) * (1 - x) = 96 :=
sorry

end price_reduction_l1590_159050


namespace factorial_expression_evaluation_l1590_159008

theorem factorial_expression_evaluation : (Real.sqrt ((Nat.factorial 5 * Nat.factorial 4) / Nat.factorial 2))^2 = 1440 :=
by
  sorry

end factorial_expression_evaluation_l1590_159008


namespace total_animals_in_shelter_l1590_159083

def initial_cats : ℕ := 15
def adopted_cats := initial_cats / 3
def replacement_cats := 2 * adopted_cats
def current_cats := initial_cats - adopted_cats + replacement_cats
def additional_dogs := 2 * current_cats
def total_animals := current_cats + additional_dogs

theorem total_animals_in_shelter : total_animals = 60 := by
  sorry

end total_animals_in_shelter_l1590_159083


namespace age_difference_l1590_159067

theorem age_difference (A B C : ℕ) (h : A + B = B + C + 18) : A = C + 18 :=
by
  sorry

end age_difference_l1590_159067


namespace height_of_water_in_cylinder_l1590_159043

theorem height_of_water_in_cylinder
  (r_cone : ℝ) (h_cone : ℝ) (r_cylinder : ℝ) (V_cone : ℝ) (V_cylinder : ℝ) (h_cylinder : ℝ) :
  r_cone = 15 → h_cone = 25 → r_cylinder = 20 →
  V_cone = (1 / 3) * π * r_cone^2 * h_cone →
  V_cylinder = V_cone → V_cylinder = π * r_cylinder^2 * h_cylinder →
  h_cylinder = 4.7 :=
by
  intros r_cone_eq h_cone_eq r_cylinder_eq V_cone_eq V_cylinder_eq volume_eq
  sorry

end height_of_water_in_cylinder_l1590_159043


namespace other_root_of_quadratic_l1590_159001

theorem other_root_of_quadratic (m : ℝ) (h : (2:ℝ) * (t:ℝ) = -6 ): 
  ∃ t, t = -3 :=
by
  sorry

end other_root_of_quadratic_l1590_159001


namespace ratio_length_to_width_l1590_159071

theorem ratio_length_to_width
  (w l : ℕ)
  (pond_length : ℕ)
  (field_length : ℕ)
  (pond_area : ℕ)
  (field_area : ℕ)
  (pond_to_field_area_ratio : ℚ)
  (field_length_given : field_length = 28)
  (pond_length_given : pond_length = 7)
  (pond_area_def : pond_area = pond_length * pond_length)
  (pond_to_field_area_ratio_def : pond_to_field_area_ratio = 1 / 8)
  (field_area_def : field_area = pond_area * 8)
  (field_area_calc : field_area = field_length * w) :
  (field_length / w) = 2 :=
by
  sorry

end ratio_length_to_width_l1590_159071


namespace gcd_m_n_l1590_159022

def m := 122^2 + 234^2 + 345^2 + 10
def n := 123^2 + 233^2 + 347^2 + 10

theorem gcd_m_n : Nat.gcd m n = 1 := by
  sorry

end gcd_m_n_l1590_159022


namespace geometric_sequence_sixth_term_l1590_159078

theorem geometric_sequence_sixth_term (a r : ℕ) (h₁ : a = 8) (h₂ : a * r ^ 3 = 64) : a * r ^ 5 = 256 :=
by
  -- to be filled (proof skipped)
  sorry

end geometric_sequence_sixth_term_l1590_159078


namespace dogs_in_academy_l1590_159025

noncomputable def numberOfDogs : ℕ :=
  let allSit := 60
  let allStay := 35
  let allFetch := 40
  let allRollOver := 45
  let sitStay := 20
  let sitFetch := 15
  let sitRollOver := 18
  let stayFetch := 10
  let stayRollOver := 13
  let fetchRollOver := 12
  let sitStayFetch := 11
  let sitStayFetchRoll := 8
  let none := 15
  118 -- final count of dogs in the academy

theorem dogs_in_academy : numberOfDogs = 118 :=
by
  sorry

end dogs_in_academy_l1590_159025


namespace ellipse_distance_CD_l1590_159059

theorem ellipse_distance_CD :
  ∃ (CD : ℝ), 
    (∀ (x y : ℝ),
    4 * (x - 2)^2 + 16 * y^2 = 64) → 
      CD = 2*Real.sqrt 5 :=
by sorry

end ellipse_distance_CD_l1590_159059


namespace equation_has_seven_real_solutions_l1590_159058

def f (x : ℝ) : ℝ := abs (x^2 - 1) - 1

theorem equation_has_seven_real_solutions (b c : ℝ) : 
  (c ≤ 0 ∧ 0 < b ∧ b < 1) ↔ 
  ∃ (x₁ x₂ x₃ x₄ x₅ x₆ x₇ : ℝ), 
  x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₁ ≠ x₅ ∧ x₁ ≠ x₆ ∧ x₁ ≠ x₇ ∧
  x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₂ ≠ x₅ ∧ x₂ ≠ x₆ ∧ x₂ ≠ x₇ ∧
  x₃ ≠ x₄ ∧ x₃ ≠ x₅ ∧ x₃ ≠ x₆ ∧ x₃ ≠ x₇ ∧
  x₄ ≠ x₅ ∧ x₄ ≠ x₆ ∧ x₄ ≠ x₇ ∧
  x₅ ≠ x₆ ∧ x₅ ≠ x₇ ∧
  x₆ ≠ x₇ ∧
  f x₁ ^ 2 - b * f x₁ + c = 0 ∧ f x₂ ^ 2 - b * f x₂ + c = 0 ∧
  f x₃ ^ 2 - b * f x₃ + c = 0 ∧ f x₄ ^ 2 - b * f x₄ + c = 0 ∧
  f x₅ ^ 2 - b * f x₅ + c = 0 ∧ f x₆ ^ 2 - b * f x₆ + c = 0 ∧
  f x₇ ^ 2 - b * f x₇ + c = 0 :=
sorry

end equation_has_seven_real_solutions_l1590_159058


namespace colorful_family_total_children_l1590_159041

theorem colorful_family_total_children (x : ℕ) (b : ℕ) :
  -- Initial equal number of white, blue, and striped children
  -- After some blue children become striped
  -- Total number of blue and white children was 10,
  -- Total number of white and striped children was 18
  -- We need to prove the total number of children is 21
  (x = 5) →
  (x + x = 10) →
  (10 + b = 18) →
  (3*x = 21) :=
by
  intros h1 h2 h3
  -- x initially represents the number of white, blue, and striped children
  -- We know x is 5 and satisfy the conditions
  sorry

end colorful_family_total_children_l1590_159041


namespace contractor_daily_amount_l1590_159027

theorem contractor_daily_amount
  (days_worked : ℕ) (total_days : ℕ) (fine_per_absent_day : ℝ)
  (total_amount : ℝ) (days_absent : ℕ) (amount_received : ℝ) :
  days_worked = total_days - days_absent →
  (total_amount = (days_worked * amount_received - days_absent * fine_per_absent_day)) →
  total_days = 30 →
  fine_per_absent_day = 7.50 →
  total_amount = 685 →
  days_absent = 2 →
  amount_received = 25 :=
by
  sorry

end contractor_daily_amount_l1590_159027


namespace trip_distance_1200_miles_l1590_159013

theorem trip_distance_1200_miles
    (D : ℕ)
    (H : D / 50 - D / 60 = 4) :
    D = 1200 :=
by
    sorry

end trip_distance_1200_miles_l1590_159013


namespace gcd_two_powers_l1590_159089

noncomputable def gcd_expression (m n : ℕ) : ℕ :=
  Int.gcd (2^m + 1) (2^n - 1)

theorem gcd_two_powers (m n : ℕ) (hm : m > 0) (hn : n > 0) (odd_n : n % 2 = 1) : 
  gcd_expression m n = 1 :=
by
  sorry

end gcd_two_powers_l1590_159089


namespace problem_conditions_l1590_159085

variables (a b : ℝ)
open Real

theorem problem_conditions (ha : a < 0) (hb : 0 < b) (hab : a + b > 0) :
  (a / b > -1) ∧ (abs a < abs b) ∧ (1 / a + 1 / b ≤ 0) ∧ ((a - 1) * (b - 1) < 1) := sorry

end problem_conditions_l1590_159085


namespace binom_squared_l1590_159066

theorem binom_squared :
  (Nat.choose 12 11) ^ 2 = 144 := 
by
  -- Mathematical steps would go here.
  sorry

end binom_squared_l1590_159066


namespace in_range_p_1_to_100_l1590_159082

def p (m n : ℤ) : ℤ :=
  2 * m^2 - 6 * m * n + 5 * n^2

-- Predicate that asserts k is in the range of p
def in_range_p (k : ℤ) : Prop :=
  ∃ m n : ℤ, p m n = k

-- Lean statement for the theorem
theorem in_range_p_1_to_100 :
  {k : ℕ | 1 ≤ k ∧ k ≤ 100 ∧ in_range_p k} = 
  {1, 2, 4, 5, 8, 9, 10, 13, 16, 17, 18, 20, 25, 26, 29, 32, 34, 36, 37, 40, 41, 45, 49, 50, 52, 53, 58, 61, 64, 65, 68, 72, 73, 74, 80, 81, 82, 85, 89, 90, 97, 98, 100} :=
  by
    sorry

end in_range_p_1_to_100_l1590_159082


namespace minimum_value_proof_l1590_159023

noncomputable def minimum_value (a b c : ℝ) (h : a + b + c = 6) : ℝ :=
  9 / a + 4 / b + 1 / c

theorem minimum_value_proof (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) (h₃ : a + b + c = 6) :
  (minimum_value a b c h₃) = 6 :=
sorry

end minimum_value_proof_l1590_159023


namespace mass_percentage_O_correct_l1590_159015

noncomputable def molar_mass_H : ℝ := 1.01
noncomputable def molar_mass_B : ℝ := 10.81
noncomputable def molar_mass_O : ℝ := 16.00

noncomputable def molar_mass_H3BO3 : ℝ := (3 * molar_mass_H) + (1 * molar_mass_B) + (3 * molar_mass_O)

noncomputable def mass_percentage_O_in_H3BO3 : ℝ := ((3 * molar_mass_O) / molar_mass_H3BO3) * 100

theorem mass_percentage_O_correct : abs (mass_percentage_O_in_H3BO3 - 77.59) < 0.01 := 
sorry

end mass_percentage_O_correct_l1590_159015


namespace sequence_formula_l1590_159073

theorem sequence_formula (a : ℕ → ℤ) (h1 : a 1 = 1)
  (h2 : ∀ n: ℕ, a (n + 1) = 2 * a n + n * (1 + 2^n)) :
  ∀ n : ℕ, a n = 2^(n - 2) * (n^2 - n + 6) - n - 1 :=
by intro n; sorry

end sequence_formula_l1590_159073


namespace weight_of_b_l1590_159053

variable (Wa Wb Wc: ℝ)

-- Conditions
def avg_weight_abc : Prop := (Wa + Wb + Wc) / 3 = 45
def avg_weight_ab : Prop := (Wa + Wb) / 2 = 40
def avg_weight_bc : Prop := (Wb + Wc) / 2 = 43

-- Theorem to prove
theorem weight_of_b (Wa Wb Wc: ℝ) (h_avg_abc : avg_weight_abc Wa Wb Wc)
  (h_avg_ab : avg_weight_ab Wa Wb) (h_avg_bc : avg_weight_bc Wb Wc) : Wb = 31 :=
by
  sorry

end weight_of_b_l1590_159053


namespace fraction_simplification_l1590_159000

theorem fraction_simplification (x : ℝ) (h : x = 0.5 * 106) : 18 / x = 18 / 53 := by
  rw [h]
  norm_num

end fraction_simplification_l1590_159000


namespace perfect_square_trinomial_m_l1590_159064

theorem perfect_square_trinomial_m (m : ℝ) :
  (∃ (a b : ℝ), ∀ x : ℝ, x^2 + (m-1)*x + 9 = (a*x + b)^2) ↔ (m = 7 ∨ m = -5) :=
sorry

end perfect_square_trinomial_m_l1590_159064


namespace simplify_and_evaluate_expression_l1590_159092

variable (x y : ℚ)

theorem simplify_and_evaluate_expression :
    x = 2 / 15 → y = 3 / 2 → 
    (2 * x + y)^2 - (3 * x - y)^2 + 5 * x * (x - y) = 1 :=
by 
  intros h1 h2
  subst h1
  subst h2
  sorry

end simplify_and_evaluate_expression_l1590_159092


namespace natural_number_40_times_smaller_l1590_159005

-- Define the sum of the first (n-1) natural numbers
def sum_natural_numbers (n : ℕ) := (n * (n - 1)) / 2

-- Define the proof statement
theorem natural_number_40_times_smaller (n : ℕ) (h : sum_natural_numbers n = 40 * n) : n = 81 :=
by {
  -- The proof is left as an exercise
  sorry
}

end natural_number_40_times_smaller_l1590_159005


namespace percentage_given_away_l1590_159035

theorem percentage_given_away
  (initial_bottles : ℕ)
  (drank_percentage : ℝ)
  (remaining_percentage : ℝ)
  (gave_away : ℝ):
  initial_bottles = 3 →
  drank_percentage = 0.90 →
  remaining_percentage = 0.70 →
  gave_away = initial_bottles - (drank_percentage * 1 + remaining_percentage) →
  (gave_away / 2) / 1 * 100 = 70 :=
by
  intros
  sorry

end percentage_given_away_l1590_159035


namespace absolute_value_sum_10_terms_l1590_159021

def sequence_sum (n : ℕ) : ℤ := (n^2 - 4 * n + 2)

def term (n : ℕ) : ℤ := sequence_sum n - sequence_sum (n - 1)

-- Prove that the sum of the absolute values of the first 10 terms is 66.
theorem absolute_value_sum_10_terms : 
  (|term 1| + |term 2| + |term 3| + |term 4| + |term 5| + 
   |term 6| + |term 7| + |term 8| + |term 9| + |term 10| = 66) := 
by 
  -- Skip the proof
  sorry

end absolute_value_sum_10_terms_l1590_159021


namespace system_inequalities_1_l1590_159037

theorem system_inequalities_1 (x : ℝ) (h1 : 2 * x ≥ x - 1) (h2 : 4 * x + 10 > x + 1) :
  x ≥ -1 :=
sorry

end system_inequalities_1_l1590_159037
