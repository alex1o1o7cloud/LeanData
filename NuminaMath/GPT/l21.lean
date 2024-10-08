import Mathlib

namespace obtuse_triangle_k_values_l21_21895

theorem obtuse_triangle_k_values (k : ℕ) (h : k > 0) :
  (∃ k, (5 < k ∧ k ≤ 12) ∨ (21 ≤ k ∧ k < 29)) → ∃ n : ℕ, n = 15 :=
by
  sorry

end obtuse_triangle_k_values_l21_21895


namespace divide_80_into_two_parts_l21_21012

theorem divide_80_into_two_parts :
  ∃ a b : ℕ, a + b = 80 ∧ b / 2 = a + 10 ∧ a = 20 ∧ b = 60 :=
by
  sorry

end divide_80_into_two_parts_l21_21012


namespace triangle_inequality_values_l21_21248

theorem triangle_inequality_values (x : ℕ) :
  x ≥ 2 ∧ x < 10 ↔ (x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5 ∨ x = 6 ∨ x = 7 ∨ x = 8 ∨ x = 9) :=
by sorry

end triangle_inequality_values_l21_21248


namespace problem1_problem2_l21_21200

noncomputable def tan_inv_3_value : ℝ := -4 / 5

theorem problem1 (α : ℝ) (h : Real.tan α = 3) :
  Real.cos α ^ 2 - 3 * Real.sin α * Real.cos α = tan_inv_3_value := 
sorry

noncomputable def f (θ : ℝ) : ℝ := 
  (2 * Real.cos θ ^ 3 + Real.sin (2 * Real.pi - θ) ^ 2 + 
   Real.sin (Real.pi / 2 + θ) - 3) / 
  (2 + 2 * Real.cos (Real.pi + θ) ^ 2 + Real.cos (-θ))

theorem problem2 :
  f (Real.pi / 3) = -1 / 2 :=
sorry

end problem1_problem2_l21_21200


namespace possible_values_of_m_l21_21307

theorem possible_values_of_m (m : ℕ) (h1 : 3 * m + 15 > 3 * m + 8) 
  (h2 : 3 * m + 8 > 4 * m - 4) (h3 : m > 11) : m = 11 := 
by
  sorry

end possible_values_of_m_l21_21307


namespace proportion_equal_l21_21174

theorem proportion_equal (x : ℝ) : (0.25 / x = 2 / 6) → x = 0.75 :=
by
  sorry

end proportion_equal_l21_21174


namespace jia_profits_1_yuan_l21_21652

-- Definition of the problem conditions
def initial_cost : ℝ := 1000
def profit_rate : ℝ := 0.1
def loss_rate : ℝ := 0.1
def resale_rate : ℝ := 0.9

-- Defined transactions with conditions
def jia_selling_price1 : ℝ := initial_cost * (1 + profit_rate)
def yi_selling_price_to_jia : ℝ := jia_selling_price1 * (1 - loss_rate)
def jia_selling_price2 : ℝ := yi_selling_price_to_jia * resale_rate

-- Final net income calculation
def jia_net_income : ℝ := -initial_cost + jia_selling_price1 - yi_selling_price_to_jia + jia_selling_price2

-- Lean statement to be proved
theorem jia_profits_1_yuan : jia_net_income = 1 := sorry

end jia_profits_1_yuan_l21_21652


namespace system_of_equations_solution_l21_21321

theorem system_of_equations_solution (x y z : ℝ) (h1 : x + y = 1) (h2 : x + z = 0) (h3 : y + z = -1) : 
    x = 1 ∧ y = 0 ∧ z = -1 := 
by 
  sorry

end system_of_equations_solution_l21_21321


namespace range_of_a_l21_21914

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 - a * x - 1 else a / x

def is_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f x ≤ f y

def func_increasing_on_R (a : ℝ) : Prop :=
  is_increasing_on (f a) Set.univ

theorem range_of_a (a : ℝ) : func_increasing_on_R a ↔ a < -2 :=
sorry

end range_of_a_l21_21914


namespace temperature_decrease_is_negative_l21_21681

-- Condition: A temperature rise of 3°C is denoted as +3°C.
def temperature_rise (c : Int) : String := if c > 0 then "+" ++ toString c ++ "°C" else toString c ++ "°C"

-- Specification: Prove a decrease of 4°C is denoted as -4°C.
theorem temperature_decrease_is_negative (h : temperature_rise 3 = "+3°C") : temperature_rise (-4) = "-4°C" :=
by
  -- Proof
  sorry

end temperature_decrease_is_negative_l21_21681


namespace quadratic_distinct_real_roots_l21_21115

-- Definitions
def is_quadratic_eq (a b c x : ℝ) (fx : ℝ) := a * x^2 + b * x + c = fx

-- Theorem statement
theorem quadratic_distinct_real_roots (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ is_quadratic_eq 1 (-2) m x₁ 0 ∧ is_quadratic_eq 1 (-2) m x₂ 0) → m < 1 :=
sorry -- Proof omitted

end quadratic_distinct_real_roots_l21_21115


namespace max_withdrawal_l21_21741

def initial_balance : ℕ := 500
def withdraw_amount : ℕ := 300
def add_amount : ℕ := 198
def remaining_balance (x : ℕ) : Prop := 
  x % 6 = 0 ∧ x ≤ initial_balance

theorem max_withdrawal : ∃(max_withdrawal_amount : ℕ), 
  max_withdrawal_amount = initial_balance - 498 :=
sorry

end max_withdrawal_l21_21741


namespace range_of_a_l21_21163

theorem range_of_a (a : ℝ) :
  (∃ x y : ℝ, (x^2 + (a^2 + 1) * x + a - 2 = 0 ∧ y^2 + (a^2 + 1) * y + a - 2 = 0)
    ∧ x > 1 ∧ y < -1) ↔ (-1 < a ∧ a < 0) := sorry

end range_of_a_l21_21163


namespace represent_2021_as_squares_l21_21345

theorem represent_2021_as_squares :
  ∃ n : ℕ, n = 505 → 2021 = (n + 1)^2 - (n - 1)^2 + 1^2 :=
by
  sorry

end represent_2021_as_squares_l21_21345


namespace intersection_A_B_l21_21563

def A : Set ℝ := {x | x ≤ 2*x + 1 ∧ 2*x + 1 ≤ 5}
def B : Set ℝ := {x | 0 < x ∧ x ≤ 3}

theorem intersection_A_B : 
  A ∩ B = {x | 0 < x ∧ x ≤ 2} :=
  sorry

end intersection_A_B_l21_21563


namespace simultaneous_eq_solution_l21_21534

theorem simultaneous_eq_solution (n : ℝ) (hn : n ≠ 1 / 2) : 
  ∃ (x y : ℝ), (y = (3 * n + 1) * x + 2) ∧ (y = (5 * n - 2) * x + 5) := 
sorry

end simultaneous_eq_solution_l21_21534


namespace lindy_total_distance_l21_21749

def meet_distance (d v_j v_c : ℕ) : ℕ :=
  d / (v_j + v_c)

def lindy_distance (v_l t : ℕ) : ℕ :=
  v_l * t

theorem lindy_total_distance
  (d : ℕ)
  (v_j : ℕ)
  (v_c : ℕ)
  (v_l : ℕ)
  (h1 : d = 360)
  (h2 : v_j = 5)
  (h3 : v_c = 7)
  (h4 : v_l = 12)
  :
  lindy_distance v_l (meet_distance d v_j v_c) = 360 :=
by
  sorry

end lindy_total_distance_l21_21749


namespace find_g_four_l21_21093

theorem find_g_four (g : ℝ → ℝ) (h : ∀ x : ℝ, g x + 3 * g (1 - x) = 4 * x ^ 2) : g 4 = 11 / 2 := 
by
  sorry

end find_g_four_l21_21093


namespace intersection_of_parabola_with_y_axis_l21_21023

theorem intersection_of_parabola_with_y_axis :
  ∃ y : ℝ, y = - (0 + 2)^2 + 6 ∧ (0, y) = (0, 2) :=
by
  sorry

end intersection_of_parabola_with_y_axis_l21_21023


namespace sum_of_distinct_integers_l21_21046

noncomputable def a : ℤ := 11
noncomputable def b : ℤ := 9
noncomputable def c : ℤ := 4
noncomputable def d : ℤ := 2
noncomputable def e : ℤ := 1

def condition : Prop := (6 - a) * (6 - b) * (6 - c) * (6 - d) * (6 - e) = 120
def distinct_integers : Prop := a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e

theorem sum_of_distinct_integers (h1 : condition) (h2 : distinct_integers) : a + b + c + d + e = 27 :=
by
  sorry

end sum_of_distinct_integers_l21_21046


namespace length_of_first_train_l21_21955

theorem length_of_first_train
    (speed_train1_kmph : ℝ) (speed_train2_kmph : ℝ) 
    (length_train2_m : ℝ) (cross_time_s : ℝ)
    (conv_factor : ℝ)         -- Conversion factor from kmph to m/s
    (relative_speed_ms : ℝ)   -- Relative speed in m/s 
    (distance_covered_m : ℝ)  -- Total distance covered in meters
    (length_train1_m : ℝ) : Prop :=
  speed_train1_kmph = 120 →
  speed_train2_kmph = 80 →
  length_train2_m = 210.04 →
  cross_time_s = 9 →
  conv_factor = 1000 / 3600 →
  relative_speed_ms = (200 * conv_factor) →
  distance_covered_m = (relative_speed_ms * cross_time_s) →
  length_train1_m = 290 →
  distance_covered_m = length_train1_m + length_train2_m

end length_of_first_train_l21_21955


namespace nurses_count_l21_21785

theorem nurses_count (D N : ℕ) (h1 : D + N = 456) (h2 : D * 11 = 8 * N) : N = 264 :=
by
  sorry

end nurses_count_l21_21785


namespace men_hours_per_day_l21_21721

theorem men_hours_per_day
  (H : ℕ)
  (men_days := 15 * 21 * H)
  (women_days := 21 * 20 * 9)
  (conversion_ratio := 3 / 2)
  (equivalent_man_hours := women_days * conversion_ratio)
  (same_work : men_days = equivalent_man_hours) :
  H = 8 :=
by
  sorry

end men_hours_per_day_l21_21721


namespace solve_quadratic_inequality_l21_21149

theorem solve_quadratic_inequality (a : ℝ) :
  (∀ x : ℝ, (a * x ^ 2 - (2 * a + 1) * x + 2 > 0 ↔
    if a = 0 then
      x < 2
    else if a > 0 then
      if a >= 1 / 2 then
        x < 1 / a ∨ x > 2
      else
        x < 2 ∨ x > 1 / a
    else
      x > 1 / a ∧ x < 2)) :=
sorry

end solve_quadratic_inequality_l21_21149


namespace avg_price_of_returned_tshirts_l21_21980

-- Define the conditions as Lean definitions
def avg_price_50_tshirts := 750
def num_tshirts := 50
def num_returned_tshirts := 7
def avg_price_remaining_43_tshirts := 720

-- The correct price of the 7 returned T-shirts
def correct_avg_price_returned := 6540 / 7

-- The proof statement
theorem avg_price_of_returned_tshirts :
  (num_tshirts * avg_price_50_tshirts - (num_tshirts - num_returned_tshirts) * avg_price_remaining_43_tshirts) / num_returned_tshirts = correct_avg_price_returned :=
by
  sorry

end avg_price_of_returned_tshirts_l21_21980


namespace total_paint_area_l21_21381

structure Room where
  length : ℕ
  width : ℕ
  height : ℕ

def livingRoom : Room := { length := 40, width := 40, height := 10 }
def bedroom : Room := { length := 12, width := 10, height := 10 }

def wallArea (room : Room) (n_walls : ℕ) : ℕ :=
  let longWallsArea := 2 * (room.length * room.height)
  let shortWallsArea := 2 * (room.width * room.height)
  if n_walls <= 2 then
    longWallsArea * n_walls / 2
  else if n_walls <= 4 then
    longWallsArea + (shortWallsArea * (n_walls - 2) / 2)
  else
    0

def totalWallArea (livingRoom : Room) (bedroom : Room) (n_livingRoomWalls n_bedroomWalls : ℕ) : ℕ :=
  wallArea livingRoom n_livingRoomWalls + wallArea bedroom n_bedroomWalls

theorem total_paint_area : totalWallArea livingRoom bedroom 3 4 = 1640 := by
  sorry

end total_paint_area_l21_21381


namespace abs_diff_squares_104_98_l21_21802

theorem abs_diff_squares_104_98 : abs ((104 : ℤ)^2 - (98 : ℤ)^2) = 1212 := by
  sorry

end abs_diff_squares_104_98_l21_21802


namespace weight_of_empty_jar_l21_21265

variable (W : ℝ) -- Weight of the empty jar
variable (w : ℝ) -- Weight of water for one-fifth of the jar

-- Conditions
variable (h1 : W + w = 560)
variable (h2 : W + 4 * w = 740)

-- Theorem statement
theorem weight_of_empty_jar (W w : ℝ) (h1 : W + w = 560) (h2 : W + 4 * w = 740) : W = 500 := 
by
  sorry

end weight_of_empty_jar_l21_21265


namespace range_of_m_l21_21484

theorem range_of_m (m : ℝ) : (∀ x : ℝ, m*x^2 + m*x + 1 > 0) ↔ (0 ≤ m ∧ m < 4) :=
by
  sorry

end range_of_m_l21_21484


namespace overall_support_percentage_l21_21066

def men_support_percentage : ℝ := 0.75
def women_support_percentage : ℝ := 0.70
def number_of_men : ℕ := 200
def number_of_women : ℕ := 800

theorem overall_support_percentage :
  ((men_support_percentage * ↑number_of_men + women_support_percentage * ↑number_of_women) / (↑number_of_men + ↑number_of_women) * 100) = 71 := 
by 
sorry

end overall_support_percentage_l21_21066


namespace suff_but_not_necc_condition_l21_21095

def x_sq_minus_1_pos (x : ℝ) : Prop := x^2 - 1 > 0
def x_minus_1_pos (x : ℝ) : Prop := x - 1 > 0

theorem suff_but_not_necc_condition : 
  (∀ x : ℝ, x_minus_1_pos x → x_sq_minus_1_pos x) ∧
  (∃ x : ℝ, x_sq_minus_1_pos x ∧ ¬ x_minus_1_pos x) :=
by 
  sorry

end suff_but_not_necc_condition_l21_21095


namespace inequality_l21_21809

theorem inequality (a b : ℝ) : a^2 + b^2 + 7/4 ≥ a * b + 2 * a + b / 2 :=
sorry

end inequality_l21_21809


namespace necessary_but_not_sufficient_l21_21705

theorem necessary_but_not_sufficient (a : ℝ) : 
  (a > 2 → a^2 > 2 * a) ∧ (a^2 > 2 * a → (a > 2 ∨ a < 0)) :=
by
  sorry

end necessary_but_not_sufficient_l21_21705


namespace hairstylist_earnings_per_week_l21_21737

theorem hairstylist_earnings_per_week :
  let cost_normal := 5
  let cost_special := 6
  let cost_trendy := 8
  let haircuts_normal := 5
  let haircuts_special := 3
  let haircuts_trendy := 2
  let days_per_week := 7
  let daily_earnings := cost_normal * haircuts_normal + cost_special * haircuts_special + cost_trendy * haircuts_trendy
  let weekly_earnings := daily_earnings * days_per_week
  weekly_earnings = 413 := sorry

end hairstylist_earnings_per_week_l21_21737


namespace tap_B_time_l21_21859

-- Define the capacities and time variables
variable (A_rate B_rate : ℝ) -- rates in percentage per hour
variable (T_A T_B : ℝ) -- time in hours

-- Define the conditions as hypotheses
def conditions : Prop :=
  (4 * (A_rate + B_rate) = 50) ∧ (2 * A_rate = 15)

-- Define the question and the target time
def target_time := 7

-- Define the goal to prove
theorem tap_B_time (h : conditions A_rate B_rate) : T_B = target_time := by
  sorry

end tap_B_time_l21_21859


namespace max_integer_value_fraction_l21_21182

theorem max_integer_value_fraction (x : ℝ) : 
  (∃ t : ℤ, t = 2 ∧ (∀ y : ℝ, y = (4*x^2 + 8*x + 21) / (4*x^2 + 8*x + 9) → y <= t)) :=
sorry

end max_integer_value_fraction_l21_21182


namespace cone_surface_area_ratio_l21_21053

noncomputable def sector_angle := 135
noncomputable def sector_area (B : ℝ) := B
noncomputable def cone (A : ℝ) (B : ℝ) := A

theorem cone_surface_area_ratio (A B : ℝ) (h_sector_angle: sector_angle = 135) (h_sector_area: sector_area B = B) (h_cone_formed: cone A B = A) :
  A / B = 11 / 8 :=
by
  sorry

end cone_surface_area_ratio_l21_21053


namespace min_elements_of_B_l21_21915

def A (k : ℝ) : Set ℝ :=
if k < 0 then {x | (k / 4 + 9 / (4 * k) + 3) < x ∧ x < 11 / 2}
else if k = 0 then {x | x < 11 / 2}
else if 0 < k ∧ k < 1 ∨ k > 9 then {x | x < 11 / 2 ∨ x > k / 4 + 9 / (4 * k) + 3}
else if 1 ≤ k ∧ k ≤ 9 then {x | x < k / 4 + 9 / (4 * k) + 3 ∨ x > 11 / 2}
else ∅

def B (k : ℝ) : Set ℤ := {x : ℤ | ↑x ∈ A k}

theorem min_elements_of_B (k : ℝ) (hk : k < 0) : 
  B k = {2, 3, 4, 5} :=
sorry

end min_elements_of_B_l21_21915


namespace paul_mowing_money_l21_21102

theorem paul_mowing_money (M : ℝ) 
  (h1 : 2 * M = 6) : 
  M = 3 :=
by 
  sorry

end paul_mowing_money_l21_21102


namespace determine_value_of_product_l21_21650

theorem determine_value_of_product (x : ℝ) (h : (x - 2) * (x + 2) = 2021) : (x - 1) * (x + 1) = 2024 := 
by 
  sorry

end determine_value_of_product_l21_21650


namespace correct_quotient_l21_21335

theorem correct_quotient (D : ℕ) (Q : ℕ) (h1 : D = 21 * Q) (h2 : D = 12 * 49) : Q = 28 := 
by
  sorry

end correct_quotient_l21_21335


namespace sum_x_y_is_9_l21_21267

-- Definitions of the conditions
variables (x y S : ℝ)
axiom h1 : x + y = S
axiom h2 : x - y = 3
axiom h3 : x^2 - y^2 = 27

-- The theorem to prove
theorem sum_x_y_is_9 : S = 9 :=
by
  -- Placeholder for the proof
  sorry

end sum_x_y_is_9_l21_21267


namespace simultaneous_equations_in_quadrant_I_l21_21583

theorem simultaneous_equations_in_quadrant_I (c : ℝ) :
  (∃ x y : ℝ, x - y = 3 ∧ c * x + y = 4 ∧ x > 0 ∧ y > 0) ↔ (-1 < c ∧ c < 4 / 3) :=
  sorry

end simultaneous_equations_in_quadrant_I_l21_21583


namespace smallest_n_for_y_n_integer_l21_21854

noncomputable def y (n : ℕ) : ℝ :=
  if n = 0 then 0 else
  if n = 1 then (5 : ℝ)^(1/3) else
  if n = 2 then ((5 : ℝ)^(1/3))^((5 : ℝ)^(1/3)) else
  y (n-1)^((5 : ℝ)^(1/3))

theorem smallest_n_for_y_n_integer : ∃ n : ℕ, y n = 5 ∧ ∀ m < n, y m ≠ ((⌊y m⌋:ℝ)) :=
by
  sorry

end smallest_n_for_y_n_integer_l21_21854


namespace find_coefficients_l21_21905

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem find_coefficients 
  (A B Q C P : V) 
  (hQ : Q = (5 / 7 : ℝ) • A + (2 / 7 : ℝ) • B)
  (hC : C = A + 2 • B)
  (hP : P = Q + C) : 
  ∃ s v : ℝ, P = s • A + v • B ∧ s = 12 / 7 ∧ v = 16 / 7 :=
by
  sorry

end find_coefficients_l21_21905


namespace fifth_grade_soccer_students_l21_21642

variable (T B Gnp GP S : ℕ)
variable (p : ℝ)

theorem fifth_grade_soccer_students
  (hT : T = 420)
  (hB : B = 296)
  (hp_percent : p = 86 / 100)
  (hGnp : Gnp = 89)
  (hpercent_boys_playing_soccer : (1 - p) * S = GP)
  (hpercent_girls_playing_soccer : GP = 35) :
  S = 250 := by
  sorry

end fifth_grade_soccer_students_l21_21642


namespace problem_solution_l21_21873

noncomputable def solve_equation (x : ℝ) : Prop :=
  x ≠ 4 ∧ (x + 36 / (x - 4) = -9)

theorem problem_solution : {x : ℝ | solve_equation x} = {0, -5} :=
by
  sorry

end problem_solution_l21_21873


namespace greatest_positive_integer_N_l21_21464

def condition (x : Int) (y : Int) : Prop :=
  (x^2 - x * y) % 1111 ≠ 0

theorem greatest_positive_integer_N :
  ∃ N : Nat, (∀ (x : Fin N) (y : Fin N), x ≠ y → condition x y) ∧ N = 1000 :=
by
  sorry

end greatest_positive_integer_N_l21_21464


namespace find_number_l21_21188

theorem find_number (x : ℕ) (h1 : x > 7) (h2 : x ≠ 8) : x = 9 := by
  sorry

end find_number_l21_21188


namespace max_k_value_l21_21113

theorem max_k_value :
  ∀ (a b c : ℝ), 0 ≤ a → 0 ≤ b → 0 ≤ c → a + b + c = a * b + b * c + c * a →
  (a + b + c) * (1 / (a + b) + 1 / (b + c) + 1 / (c + a) - 1) ≥ 1 :=
by
  intros a b c ha hb hc habc_eq
  sorry

end max_k_value_l21_21113


namespace number_of_friends_l21_21871

def initial_candies : ℕ := 10
def additional_candies : ℕ := 4
def total_candies : ℕ := initial_candies + additional_candies
def candies_per_friend : ℕ := 2

theorem number_of_friends : total_candies / candies_per_friend = 7 :=
by
  sorry

end number_of_friends_l21_21871


namespace shooting_competition_probabilities_l21_21119

theorem shooting_competition_probabilities (p_A_not_losing p_B_losing : ℝ)
  (h₁ : p_A_not_losing = 0.59)
  (h₂ : p_B_losing = 0.44) :
  (1 - p_B_losing = 0.56) ∧ (p_A_not_losing - p_B_losing = 0.15) :=
by
  sorry

end shooting_competition_probabilities_l21_21119


namespace irrational_roots_of_odd_coeffs_l21_21003

theorem irrational_roots_of_odd_coeffs (a b c : ℤ) (ha : a % 2 = 1) (hb : b % 2 = 1) (hc : c % 2 = 1) : 
  ¬ ∃ (x : ℚ), a * x^2 + b * x + c = 0 := 
sorry

end irrational_roots_of_odd_coeffs_l21_21003


namespace problemStatement_l21_21562

-- Define the set of values as a type
structure SetOfValues where
  k : ℤ
  b : ℤ

-- The given sets of values
def A : SetOfValues := ⟨2, 2⟩
def B : SetOfValues := ⟨2, -2⟩
def C : SetOfValues := ⟨-2, -2⟩
def D : SetOfValues := ⟨-2, 2⟩

-- Define the conditions for the function
def isValidSet (s : SetOfValues) : Prop :=
  s.k < 0 ∧ s.b > 0

-- The problem statement: Prove that D is a valid set
theorem problemStatement : isValidSet D := by
  sorry

end problemStatement_l21_21562


namespace smallest_k_l21_21255

theorem smallest_k (k : ℕ) (h1 : k > 1) (h2 : k % 19 = 1) (h3 : k % 7 = 1) (h4 : k % 3 = 1) : k = 400 :=
by
  sorry

end smallest_k_l21_21255


namespace quadratic_equation_properties_l21_21763

theorem quadratic_equation_properties (m : ℝ) (h : m < 4) (root_one : ℝ) (root_two : ℝ) 
  (eq1 : root_one + root_two = 4) (eq2 : root_one * root_two = m) (root_one_eq : root_one = -1) :
  m = -5 ∧ root_two = 5 ∧ (root_one ≠ root_two) :=
by
  -- Sorry is added to skip the proof because only the statement is needed.
  sorry

end quadratic_equation_properties_l21_21763


namespace consecutive_odd_integers_expressions_l21_21032

theorem consecutive_odd_integers_expressions
  {p q : ℤ} (hpq : p + 2 = q ∨ p - 2 = q) (hp_odd : p % 2 = 1) (hq_odd : q % 2 = 1) :
  (2 * p + 5 * q) % 2 = 1 ∧ (5 * p - 2 * q) % 2 = 1 ∧ (2 * p * q + 5) % 2 = 1 :=
  sorry

end consecutive_odd_integers_expressions_l21_21032


namespace total_pies_l21_21469

-- Define the number of each type of pie.
def apple_pies : Nat := 2
def pecan_pies : Nat := 4
def pumpkin_pies : Nat := 7

-- Prove the total number of pies.
theorem total_pies : apple_pies + pecan_pies + pumpkin_pies = 13 := by
  sorry

end total_pies_l21_21469


namespace lucinda_jelly_beans_l21_21326

theorem lucinda_jelly_beans (g l : ℕ) 
  (h₁ : g = 3 * l) 
  (h₂ : g - 20 = 4 * (l - 20)) : 
  g = 180 := 
by 
  sorry

end lucinda_jelly_beans_l21_21326


namespace congruence_is_sufficient_but_not_necessary_for_equal_area_l21_21649

-- Definition of conditions
def Congruent (Δ1 Δ2 : Type) : Prop := sorry -- Definition of congruent triangles
def EqualArea (Δ1 Δ2 : Type) : Prop := sorry -- Definition of triangles with equal area

-- Theorem statement
theorem congruence_is_sufficient_but_not_necessary_for_equal_area 
  (Δ1 Δ2 : Type) :
  (Congruent Δ1 Δ2 → EqualArea Δ1 Δ2) ∧ (¬ (EqualArea Δ1 Δ2 → Congruent Δ1 Δ2)) :=
sorry

end congruence_is_sufficient_but_not_necessary_for_equal_area_l21_21649


namespace area_of_PQ_square_l21_21096

theorem area_of_PQ_square (a b c : ℕ)
  (h1 : a^2 = 144)
  (h2 : b^2 = 169)
  (h3 : a^2 + c^2 = b^2) :
  c^2 = 25 :=
by
  sorry

end area_of_PQ_square_l21_21096


namespace leak_empties_cistern_in_24_hours_l21_21600

theorem leak_empties_cistern_in_24_hours (F L : ℝ) (h1: F = 1 / 8) (h2: F - L = 1 / 12) :
  1 / L = 24 := 
by {
  sorry
}

end leak_empties_cistern_in_24_hours_l21_21600


namespace initial_milk_amount_l21_21847

theorem initial_milk_amount (d : ℚ) (r : ℚ) (T : ℚ) 
  (hd : d = 0.4) 
  (hr : r = 0.69) 
  (h_remaining : r = (1 - d) * T) : 
  T = 1.15 := 
  sorry

end initial_milk_amount_l21_21847


namespace selling_price_with_increase_l21_21369

variable (a : ℝ)

theorem selling_price_with_increase (h : a > 0) : 1.1 * a = a + 0.1 * a := by
  -- Here you will add the proof, which we skip with sorry
  sorry

end selling_price_with_increase_l21_21369


namespace x_fourth_minus_inv_fourth_l21_21165

theorem x_fourth_minus_inv_fourth (x : ℝ) (h : x - 1/x = 5) : x^4 - 1/(x^4) = 727 :=
by
  sorry

end x_fourth_minus_inv_fourth_l21_21165


namespace find_values_of_p_l21_21981

def geometric_progression (p : ℝ) : Prop :=
  (2 * p)^2 = (4 * p + 5) * |p - 3|

theorem find_values_of_p :
  {p : ℝ | geometric_progression p} = {-1, 15 / 8} :=
by
  sorry

end find_values_of_p_l21_21981


namespace no_roots_in_disk_l21_21060

noncomputable def homogeneous_polynomial_deg2 (a b c : ℝ) (x y : ℝ) := a * x^2 + b * x * y + c * y^2
noncomputable def homogeneous_polynomial_deg3 (q : ℝ → ℝ → ℝ) (x y : ℝ) := q x y

theorem no_roots_in_disk 
  (a b c : ℝ) (h_poly_deg2 : ∀ x y, homogeneous_polynomial_deg2 a b c x y = a * x^2 + b * x * y + c * y^2)
  (q : ℝ → ℝ → ℝ) (h_poly_deg3 : ∀ x y, homogeneous_polynomial_deg3 q x y = q x y)
  (h_cond : b^2 < 4 * a * c) :
  ∃ k > 0, ∀ x y, x^2 + y^2 < k → homogeneous_polynomial_deg2 a b c x y ≠ homogeneous_polynomial_deg3 q x y ∨ (x = 0 ∧ y = 0) :=
sorry

end no_roots_in_disk_l21_21060


namespace part_I_part_II_l21_21440

noncomputable def f (a x : ℝ) : ℝ := a * Real.log x + ((2 * a^2) / x) + x

theorem part_I (a : ℝ) (h : a > 0) : 
  (∃ x : ℝ, x = 1 ∧ deriv (f a) x = -2) → a = 3 / 2 :=
sorry

theorem part_II (a : ℝ) (h : a = 3 / 2) : 
  (∀ x : ℝ, 0 < x ∧ x < 3 / 2 → deriv (f a) x < 0) ∧ 
  (∀ x : ℝ, x > 3 / 2 → deriv (f a) x > 0) :=
sorry

end part_I_part_II_l21_21440


namespace green_paint_amount_l21_21426

theorem green_paint_amount (T W B : ℕ) (hT : T = 69) (hW : W = 20) (hB : B = 34) : 
  T - (W + B) = 15 := 
by
  sorry

end green_paint_amount_l21_21426


namespace old_lamp_height_is_one_l21_21757

def new_lamp_height : ℝ := 2.3333333333333335
def height_difference : ℝ := 1.3333333333333333
def old_lamp_height : ℝ := new_lamp_height - height_difference

theorem old_lamp_height_is_one :
  old_lamp_height = 1 :=
by
  sorry

end old_lamp_height_is_one_l21_21757


namespace martin_total_distance_l21_21801

-- Define the conditions
def total_trip_time : ℕ := 8
def first_half_speed : ℕ := 70
def second_half_speed : ℕ := 85
def half_trip_time : ℕ := total_trip_time / 2

-- Define the total distance traveled 
def total_distance : ℕ := (first_half_speed * half_trip_time) + (second_half_speed * half_trip_time)

-- Statement to prove
theorem martin_total_distance : total_distance = 620 :=
by
  -- This is a placeholder to represent that a proof is needed
  -- Actual proof steps are omitted as instructed
  sorry

end martin_total_distance_l21_21801


namespace unique_x_inequality_l21_21401

theorem unique_x_inequality (a : ℝ) : (∀ x : ℝ, |x^2 + 2*a*x + 3*a| ≤ 2 → (a = 1 ∨ a = 2)) :=
by
  sorry

end unique_x_inequality_l21_21401


namespace original_profit_percentage_is_10_l21_21041

-- Define the conditions and the theorem
theorem original_profit_percentage_is_10
  (original_selling_price : ℝ)
  (price_reduction: ℝ)
  (additional_profit: ℝ)
  (profit_percentage: ℝ)
  (new_profit_percentage: ℝ)
  (new_selling_price: ℝ) :
  original_selling_price = 659.9999999999994 →
  price_reduction = 0.10 →
  additional_profit = 42 →
  profit_percentage = 30 →
  new_profit_percentage = 1.30 →
  new_selling_price = original_selling_price + additional_profit →
  ((original_selling_price / (original_selling_price / (new_profit_percentage * (1 - price_reduction)))) - 1) * 100 = 10 :=
by
  sorry

end original_profit_percentage_is_10_l21_21041


namespace f_increasing_l21_21852

noncomputable def f (x : Real) : Real := (2 * Real.exp x) / (1 + Real.exp x) + 1/2

theorem f_increasing : ∀ x y : Real, x < y → f x < f y := 
by
  -- the proof goes here
  sorry

end f_increasing_l21_21852


namespace prove_optionC_is_suitable_l21_21357

def OptionA := "Understanding the height of students in Class 7(1)"
def OptionB := "Companies recruiting and interviewing job applicants"
def OptionC := "Investigating the impact resistance of a batch of cars"
def OptionD := "Selecting the fastest runner in our school to participate in the city-wide competition"

def is_suitable_for_sampling_survey (option : String) : Prop :=
  option = OptionC

theorem prove_optionC_is_suitable :
  is_suitable_for_sampling_survey OptionC :=
by
  sorry

end prove_optionC_is_suitable_l21_21357


namespace isosceles_triangle_perimeter_l21_21745

theorem isosceles_triangle_perimeter {a b c : ℝ} (h1 : a = 4) (h2 : b = 8) 
  (isosceles : a = c ∨ b = c) (triangle_inequality : a + a > b) :
  a + b + c = 20 :=
by
  sorry

end isosceles_triangle_perimeter_l21_21745


namespace ratio_pentagon_side_length_to_rectangle_width_l21_21820

def pentagon_side_length (p : ℕ) (n : ℕ) := p / n
def rectangle_width (p : ℕ) (ratio : ℕ) := p / (2 * (1 + ratio))

theorem ratio_pentagon_side_length_to_rectangle_width :
  pentagon_side_length 60 5 / rectangle_width 80 3 = (6 : ℚ) / 5 :=
by {
  sorry
}

end ratio_pentagon_side_length_to_rectangle_width_l21_21820


namespace exists_line_with_two_colors_l21_21519

open Classical

/-- Given a grid with 1x1 squares where each vertex is painted one of four colors such that each 1x1 square's vertices are all different colors, 
    there exists a line in the grid with nodes of exactly two different colors. -/
theorem exists_line_with_two_colors 
  (A : Type)
  [Inhabited A]
  [DecidableEq A]
  (colors : Finset A) 
  (h_col : colors.card = 4) 
  (grid : ℤ × ℤ → A) 
  (h_diff_colors : ∀ (i j : ℤ), i ≠ j → ∀ (k l : ℤ), grid (i, k) ≠ grid (j, k) ∧ grid (i, l) ≠ grid (i, k)) :
  ∃ line : ℤ → ℤ × ℤ, ∃ a b : A, a ≠ b ∧ ∀ n : ℤ, grid (line n) = a ∨ grid (line n) = b :=
sorry

end exists_line_with_two_colors_l21_21519


namespace sum_of_positive_integers_eq_32_l21_21339

noncomputable def sum_of_integers (x y : ℕ) (h1 : x - y = 8) (h2 : x * y = 240) : ℕ :=
  x + y

theorem sum_of_positive_integers_eq_32 (x y : ℕ) (h1 : x - y = 8) (h2 : x * y = 240) : sum_of_integers x y h1 h2 = 32 :=
  sorry

end sum_of_positive_integers_eq_32_l21_21339


namespace q_work_alone_in_10_days_l21_21437

theorem q_work_alone_in_10_days (p_rate : ℝ) (q_rate : ℝ) (d : ℕ) (h1 : p_rate = 1 / 20)
                                    (h2 : q_rate = 1 / d) (h3 : 2 * (p_rate + q_rate) = 0.3) :
                                    d = 10 :=
by sorry

end q_work_alone_in_10_days_l21_21437


namespace only_C_forms_triangle_l21_21858

def triangle_sides (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem only_C_forms_triangle :
  ¬ triangle_sides 3 4 8 ∧
  ¬ triangle_sides 2 5 2 ∧
  triangle_sides 3 5 6 ∧
  ¬ triangle_sides 5 6 11 :=
by
  sorry

end only_C_forms_triangle_l21_21858


namespace remaining_days_to_finish_l21_21181

-- Define initial conditions and constants
def initial_play_hours_per_day : ℕ := 4
def initial_days : ℕ := 14
def completion_fraction : ℚ := 0.40
def increased_play_hours_per_day : ℕ := 7

-- Define the calculation for total initial hours played
def total_initial_hours_played : ℕ := initial_play_hours_per_day * initial_days

-- Define the total hours needed to complete the game
def total_hours_to_finish := total_initial_hours_played / completion_fraction

-- Define the remaining hours needed to finish the game
def remaining_hours := total_hours_to_finish - total_initial_hours_played

-- Prove that the remaining days to finish the game is 12
theorem remaining_days_to_finish : (remaining_hours / increased_play_hours_per_day) = 12 := by
  sorry -- Proof steps go here

end remaining_days_to_finish_l21_21181


namespace binary_sum_is_11_l21_21226

-- Define the binary numbers
def b1 : ℕ := 5  -- equivalent to 101 in binary
def b2 : ℕ := 6  -- equivalent to 110 in binary

-- Define the expected sum in decimal
def expected_sum : ℕ := 11

-- The theorem statement
theorem binary_sum_is_11 : b1 + b2 = expected_sum := by
  sorry

end binary_sum_is_11_l21_21226


namespace power_function_increasing_l21_21281

theorem power_function_increasing {α : ℝ} (hα : α = 1 ∨ α = 3 ∨ α = 1 / 2) :
  ∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x ≤ y → x ^ α ≤ y ^ α := 
sorry

end power_function_increasing_l21_21281


namespace max_observing_relations_lemma_l21_21405

/-- There are 24 robots on a plane, each with a 70-degree field of view. -/
def robots : ℕ := 24

/-- Definition of field of view for each robot. -/
def field_of_view : ℝ := 70

/-- Maximum number of observing relations. Observing is a one-sided relation. -/
def max_observing_relations := 468

/-- Theorem: The maximum number of observing relations among 24 robots,
each with a 70-degree field of view, is 468. -/
theorem max_observing_relations_lemma : max_observing_relations = 468 :=
by
  sorry

end max_observing_relations_lemma_l21_21405


namespace ratio_M_N_l21_21582

variable (M Q P N : ℝ)

-- Conditions
axiom h1 : M = 0.40 * Q
axiom h2 : Q = 0.25 * P
axiom h3 : N = 0.60 * P

theorem ratio_M_N : M / N = 1 / 6 :=
by
  sorry

end ratio_M_N_l21_21582


namespace new_person_weight_l21_21008

theorem new_person_weight (w : ℝ) (avg_increase : ℝ) (replaced_person_weight : ℝ) (num_people : ℕ) 
(H1 : avg_increase = 4.8) (H2 : replaced_person_weight = 62) (H3 : num_people = 12) : 
w = 119.6 :=
by
  -- We could provide the intermediate steps as definitions here but for the theorem statement, we just present the goal.
  sorry

end new_person_weight_l21_21008


namespace number_of_moles_of_NaCl_l21_21679

theorem number_of_moles_of_NaCl
  (moles_NaOH : ℕ)
  (moles_Cl2 : ℕ)
  (reaction : 2 * moles_NaOH + moles_Cl2 = 2 * moles_NaOH + 1) :
  2 * moles_Cl2 = 2 := by 
  sorry

end number_of_moles_of_NaCl_l21_21679


namespace parabola_has_one_x_intercept_l21_21104

-- Define the equation of the parabola
def parabola (y : ℝ) : ℝ :=
  -3 * y ^ 2 + 2 * y + 2

-- The theorem statement asserting there is exactly one x-intercept
theorem parabola_has_one_x_intercept : ∃! x : ℝ, ∃ y : ℝ, parabola y = x ∧ y = 0 := by
  sorry

end parabola_has_one_x_intercept_l21_21104


namespace find_x_l21_21937

theorem find_x (x y : ℕ) (h1 : x / y = 12 / 5) (h2 : y = 25) : x = 60 :=
sorry

end find_x_l21_21937


namespace f_1982_eq_660_l21_21922

def f : ℕ → ℕ := sorry

axiom h1 : ∀ m n : ℕ, f (m + n) - f m - f n = 0 ∨ f (m + n) - f m - f n = 1
axiom h2 : f 2 = 0
axiom h3 : f 3 > 0
axiom h4 : f 9999 = 3333

theorem f_1982_eq_660 : f 1982 = 660 := sorry

end f_1982_eq_660_l21_21922


namespace max_campaign_making_animals_prime_max_campaign_making_animals_nine_l21_21786

theorem max_campaign_making_animals_prime (n : ℕ) (h_prime : Nat.Prime n) (h_ge : n ≥ 3) : 
  ∃ k, k = (n - 1) / 2 :=
by
  sorry

theorem max_campaign_making_animals_nine : ∃ k, k = 4 :=
by
  sorry

end max_campaign_making_animals_prime_max_campaign_making_animals_nine_l21_21786


namespace board_division_condition_l21_21154

open Nat

theorem board_division_condition (n : ℕ) : 
  (∃ k : ℕ, n = 4 * k) ↔ 
  (∃ v h : ℕ, v = h ∧ (2 * v + 2 * h = n * n ∧ n % 2 = 0)) := 
sorry

end board_division_condition_l21_21154


namespace spherical_coord_plane_l21_21016

-- Let's define spherical coordinates and the condition theta = c.
structure SphericalCoordinates where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

def is_plane (c : ℝ) (p : SphericalCoordinates) : Prop :=
  p.θ = c

theorem spherical_coord_plane (c : ℝ) : 
  ∀ p : SphericalCoordinates, is_plane c p → True := 
by
  intros p hp
  sorry

end spherical_coord_plane_l21_21016


namespace tan_alpha_eq_three_sin_cos_l21_21235

theorem tan_alpha_eq_three_sin_cos (α : ℝ) (h : Real.tan α = 3) : 
  Real.sin α * Real.cos α = 3 / 10 :=
by 
  sorry

end tan_alpha_eq_three_sin_cos_l21_21235


namespace max_remainder_division_by_9_l21_21383

theorem max_remainder_division_by_9 : ∀ (r : ℕ), r < 9 → r ≤ 8 :=
by sorry

end max_remainder_division_by_9_l21_21383


namespace x_power_expression_l21_21453

theorem x_power_expression (x : ℝ) (h : x^3 - 3 * x = 5) : x^5 - 27 * x^2 = -22 * x^2 + 9 * x + 15 :=
by
  --proof goes here
  sorry

end x_power_expression_l21_21453


namespace b_cong_zero_l21_21694

theorem b_cong_zero (a b c m : ℤ) (h₀ : 1 < m) (h : ∀ (n : ℕ), (a ^ n + b * n + c) % m = 0) : b % m = 0 :=
  sorry

end b_cong_zero_l21_21694


namespace unique_positive_integer_n_l21_21555

-- Definitions based on conditions
def is_divisor (n a : ℕ) : Prop := a % n = 0

def is_perfect_square (m : ℕ) : Prop := ∃ k : ℕ, k * k = m

-- The main theorem statement
theorem unique_positive_integer_n : ∃ (n : ℕ), n > 0 ∧ is_divisor n 1989 ∧
    is_perfect_square (n^2 - 1989 / n) ∧ n = 13 :=
by
  sorry

end unique_positive_integer_n_l21_21555


namespace average_cost_of_fruit_l21_21223

variable (apples bananas oranges total_cost total_pieces avg_cost : ℕ)

theorem average_cost_of_fruit (h1 : apples = 12)
                              (h2 : bananas = 4)
                              (h3 : oranges = 4)
                              (h4 : total_cost = apples * 2 + bananas * 1 + oranges * 3)
                              (h5 : total_pieces = apples + bananas + oranges)
                              (h6 : avg_cost = total_cost / total_pieces) :
                              avg_cost = 2 :=
by sorry

end average_cost_of_fruit_l21_21223


namespace infinite_solutions_iff_a_eq_neg12_l21_21407

theorem infinite_solutions_iff_a_eq_neg12 {a : ℝ} : 
  (∀ x : ℝ, 4 * (3 * x - a) = 3 * (4 * x + 16)) ↔ a = -12 :=
by 
  sorry

end infinite_solutions_iff_a_eq_neg12_l21_21407


namespace complex_z_1000_l21_21832

open Complex

theorem complex_z_1000 (z : ℂ) (h : z + z⁻¹ = 2 * Real.cos (Real.pi * 5 / 180)) :
  z^(1000 : ℕ) + (z^(1000 : ℕ))⁻¹ = 2 * Real.cos (Real.pi * 20 / 180) :=
sorry

end complex_z_1000_l21_21832


namespace range_of_k_l21_21289

theorem range_of_k (k : ℝ) :
  (∀ x : ℝ, 1 < x → k * (Real.exp (k * x) + 1) - ((1 / x) + 1) * Real.log x > 0) ↔ k > 1 / Real.exp 1 := 
  sorry

end range_of_k_l21_21289


namespace exists_k_l21_21268

def satisfies_condition (a b : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, n ≥ 3 → (a n - a (n-1)) * (a n - a (n-2)) + (b n - b (n-1)) * (b n - b (n-2)) = 0

theorem exists_k (a b : ℕ → ℤ) 
  (h : satisfies_condition a b) : 
  ∃ k : ℕ, k > 0 ∧ a k = a (k + 2008) :=
sorry

end exists_k_l21_21268


namespace no_triangle_sides_exist_l21_21628

theorem no_triangle_sides_exist (x y z : ℝ) (h_triangle_sides : x > 0 ∧ y > 0 ∧ z > 0)
  (h_triangle_inequality : x < y + z ∧ y < x + z ∧ z < x + y) :
  x^3 + y^3 + z^3 ≠ (x + y) * (y + z) * (z + x) :=
sorry

end no_triangle_sides_exist_l21_21628


namespace ratio_alison_brittany_l21_21247

def kent_money : ℕ := 1000
def brooke_money : ℕ := 2 * kent_money
def brittany_money : ℕ := 4 * brooke_money
def alison_money : ℕ := 4000

theorem ratio_alison_brittany : alison_money * 2 = brittany_money :=
by
  sorry

end ratio_alison_brittany_l21_21247


namespace geometric_sequence_a1_l21_21293

theorem geometric_sequence_a1 (a : ℕ → ℝ) (q : ℝ) 
  (hq : 0 < q)
  (h1 : a 4 * a 8 = 2 * (a 5) ^ 2)
  (h2 : a 2 = 1) :
  a 1 = Real.sqrt 2 / 2 :=
sorry

end geometric_sequence_a1_l21_21293


namespace prob_A_wins_match_expected_games_won_variance_games_won_l21_21918

-- Definitions of probabilities
def prob_A_win := 0.6
def prob_B_win := 0.4

-- Prove that the probability of A winning the match is 0.648
theorem prob_A_wins_match : 
  prob_A_win * prob_A_win + 2 * prob_B_win * prob_A_win * prob_A_win = 0.648 :=
  sorry

-- Define the expected number of games won by A
noncomputable def expected_games_won_by_A := 
  0 * (prob_B_win * prob_B_win) + 1 * (2 * prob_A_win * prob_B_win * prob_B_win) + 
  2 * (prob_A_win * prob_A_win + 2 * prob_B_win * prob_A_win * prob_A_win)

-- Prove the expected number of games won by A is 1.5
theorem expected_games_won : 
  expected_games_won_by_A = 1.5 :=
  sorry

-- Define the variance of the number of games won by A
noncomputable def variance_games_won_by_A := 
  (prob_B_win * prob_B_win) * (0 - 1.5)^2 + 
  (2 * prob_A_win * prob_B_win * prob_B_win) * (1 - 1.5)^2 + 
  (prob_A_win * prob_A_win + 2 * prob_B_win * prob_A_win * prob_A_win) * (2 - 1.5)^2

-- Prove the variance of the number of games won by A is 0.57
theorem variance_games_won : 
  variance_games_won_by_A = 0.57 :=
  sorry

end prob_A_wins_match_expected_games_won_variance_games_won_l21_21918


namespace cost_per_serving_l21_21568

def pasta_cost : ℝ := 1.0
def sauce_cost : ℝ := 2.0
def meatballs_cost : ℝ := 5.0
def total_servings : ℝ := 8.0

theorem cost_per_serving : (pasta_cost + sauce_cost + meatballs_cost) / total_servings = 1.0 :=
by sorry

end cost_per_serving_l21_21568


namespace sum_of_obtuse_angles_l21_21543

theorem sum_of_obtuse_angles (A B : ℝ) (hA1 : A > π / 2) (hA2 : A < π)
  (hB1 : B > π / 2) (hB2 : B < π)
  (hSinA : Real.sin A = Real.sqrt 5 / 5)
  (hSinB : Real.sin B = Real.sqrt 10 / 10) :
  A + B = 7 * π / 4 := 
sorry

end sum_of_obtuse_angles_l21_21543


namespace difference_in_spending_l21_21670

-- Condition: original prices and discounts
def original_price_candy_bar : ℝ := 6
def discount_candy_bar : ℝ := 0.25
def original_price_chocolate : ℝ := 3
def discount_chocolate : ℝ := 0.10

-- The theorem to prove
theorem difference_in_spending : 
  (original_price_candy_bar * (1 - discount_candy_bar) - original_price_chocolate * (1 - discount_chocolate)) = 1.80 :=
by
  sorry

end difference_in_spending_l21_21670


namespace integral_abs_sin_from_0_to_2pi_l21_21558

theorem integral_abs_sin_from_0_to_2pi : ∫ x in (0 : ℝ)..(2 * Real.pi), |Real.sin x| = 4 := 
by
  sorry

end integral_abs_sin_from_0_to_2pi_l21_21558


namespace John_pays_more_than_Jane_l21_21977

theorem John_pays_more_than_Jane : 
  let original_price := 24.00000000000002
  let discount_rate := 0.10
  let tip_rate := 0.15
  let discount := discount_rate * original_price
  let discounted_price := original_price - discount
  let john_tip := tip_rate * original_price
  let jane_tip := tip_rate * discounted_price
  let john_total := discounted_price + john_tip
  let jane_total := discounted_price + jane_tip
  john_total - jane_total = 0.3600000000000003 :=
by
  sorry

end John_pays_more_than_Jane_l21_21977


namespace flower_bed_l21_21561

def planting_schemes (A B C D E F : Prop) : Prop :=
  A ≠ B ∧ B ≠ C ∧ D ≠ E ∧ E ≠ F ∧ A ≠ D ∧ B ≠ D ∧ B ≠ E ∧ C ≠ E ∧ C ≠ F ∧ D ≠ F

theorem flower_bed (A B C D E F : Prop) (plant_choices : Finset (Fin 6))
  (h_choice : plant_choices.card = 6)
  (h_different : ∀ x ∈ plant_choices, ∀ y ∈ plant_choices, x ≠ y → x ≠ y)
  (h_adj : planting_schemes A B C D E F) :
  ∃! planting_schemes, planting_schemes ∧ plant_choices.card = 13230 :=
by sorry

end flower_bed_l21_21561


namespace range_of_x_l21_21190

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + x + 1

-- State the theorem to prove the condition
theorem range_of_x (x : ℝ) : f (1 - x) + f (2 * x) > 2 ↔ x > -1 :=
by {
  sorry -- Proof placeholder
}

end range_of_x_l21_21190


namespace tan_11pi_over_6_l21_21068

theorem tan_11pi_over_6 :
  Real.tan (11 * Real.pi / 6) = -1 / Real.sqrt 3 :=
by
  sorry

end tan_11pi_over_6_l21_21068


namespace unique_positive_integer_n_l21_21921

theorem unique_positive_integer_n (n x : ℕ) (hx : x > 0) (hn : n = 2 ^ (2 * x - 1) - 5 * x - 3 ∧ n = (2 ^ (x-1) - 1) * (2 ^ x + 1)) : n = 2015 := by
  sorry

end unique_positive_integer_n_l21_21921


namespace max_slope_of_circle_l21_21114

theorem max_slope_of_circle (x y : ℝ) 
  (h : x^2 + y^2 - 6 * x - 6 * y + 12 = 0) : 
  ∃ k : ℝ, k = 3 + 2 * Real.sqrt 2 ∧ ∀ k' : ℝ, (x = 0 → k' = 0) ∧ (x ≠ 0 → y = k' * x → k' ≤ k) :=
sorry

end max_slope_of_circle_l21_21114


namespace increase_in_area_400ft2_l21_21244

theorem increase_in_area_400ft2 (l w : ℝ) (h₁ : l = 60) (h₂ : w = 20)
  (h₃ : 4 * (l + w) = 4 * (4 * (l + w) / 4 / 4 )):
  (4 * (l + w) / 4) ^ 2 - l * w = 400 := by
  sorry

end increase_in_area_400ft2_l21_21244


namespace rectangle_area_k_l21_21875

theorem rectangle_area_k (d : ℝ) (x : ℝ) (h_ratio : 5 * x > 0 ∧ 2 * x > 0) (h_diagonal : d^2 = (5 * x)^2 + (2 * x)^2) :
  ∃ k : ℝ, (∃ (h : k = 10 / 29), (5 * x) * (2 * x) = k * d^2) := by
  use 10 / 29
  sorry

end rectangle_area_k_l21_21875


namespace fifth_eq_l21_21026

theorem fifth_eq :
  (1 = 1) ∧
  (2 + 3 + 4 = 9) ∧
  (3 + 4 + 5 + 6 + 7 = 25) ∧
  (4 + 5 + 6 + 7 + 8 + 9 + 10 = 49) →
  5 + 6 + 7 + 8 + 9 + 10 + 11 + 12 + 13 = 81 :=
by
  intros
  sorry

end fifth_eq_l21_21026


namespace ellipse_center_x_coordinate_l21_21857

theorem ellipse_center_x_coordinate (C : ℝ × ℝ)
  (h1 : C.1 = 3)
  (h2 : 4 ≤ C.2 ∧ C.2 ≤ 12)
  (hx : ∃ F1 F2 : ℝ × ℝ, F1 = (3, 4) ∧ F2 = (3, 12)
    ∧ (F1.1 = F2.1 ∧ F1.2 < F2.2)
    ∧ C = ((F1.1 + F2.1)/2, (F1.2 + F2.2)/2))
  (tangent : ∀ P : ℝ × ℝ, (P.1 - 0) * (P.2 - 0) = 0)
  (ellipse : ∃ a b : ℝ, a > 0 ∧ b > 0
    ∧ ∀ P : ℝ × ℝ,
      (P.1 - C.1)^2/a^2 + (P.2 - C.2)^2/b^2 = 1) :
   C.1 = 3 := sorry

end ellipse_center_x_coordinate_l21_21857


namespace quadratic_roots_m_value_l21_21420

noncomputable def quadratic_roots_condition (m : ℝ) (x1 x2 : ℝ) : Prop :=
  (∀ a b c : ℝ, a = 1 ∧ b = 2 * (m + 1) ∧ c = m^2 - 1 → x1^2 + b * x1 + c = 0 ∧ x2^2 + b * x2 + c = 0) ∧ 
  (x1 - x2)^2 = 16 - x1 * x2

theorem quadratic_roots_m_value (m : ℝ) (x1 x2 : ℝ) (h : quadratic_roots_condition m x1 x2) : m = 1 :=
sorry

end quadratic_roots_m_value_l21_21420


namespace power_of_two_as_sum_of_squares_l21_21548

theorem power_of_two_as_sum_of_squares (n : ℕ) (h : n ≥ 3) :
  ∃ (x y : ℤ), x % 2 = 1 ∧ y % 2 = 1 ∧ (2^n = 7*x^2 + y^2) :=
by
  sorry

end power_of_two_as_sum_of_squares_l21_21548


namespace delta_max_success_ratio_l21_21427

/-- In a two-day math challenge, Gamma and Delta both attempted questions totalling 600 points. 
    Gamma scored 180 points out of 300 points attempted each day.
    Delta attempted a different number of points each day and their daily success ratios were less by both days than Gamma's, 
    whose overall success ratio was 3/5. Prove that the maximum possible two-day success ratio that Delta could have achieved was 359/600. -/
theorem delta_max_success_ratio :
  ∀ (x y z w : ℕ), (0 < x) ∧ (0 < y) ∧ (0 < z) ∧ (0 < w) ∧ (x ≤ (3 * y) / 5) ∧ (z ≤ (3 * w) / 5) ∧ (y + w = 600) ∧ (x + z < 360)
  → (x + z) / 600 ≤ 359 / 600 :=
by
  sorry

end delta_max_success_ratio_l21_21427


namespace probability_grunters_win_all_5_games_l21_21385

noncomputable def probability_grunters_win_game : ℚ := 4 / 5

theorem probability_grunters_win_all_5_games :
  (probability_grunters_win_game ^ 5) = 1024 / 3125 := 
  by 
  sorry

end probability_grunters_win_all_5_games_l21_21385


namespace largest_square_area_l21_21781

theorem largest_square_area (a b c : ℝ) 
  (h1 : c^2 = a^2 + b^2) 
  (h2 : a = b - 5) 
  (h3 : a^2 + b^2 + c^2 = 450) : 
  c^2 = 225 :=
by 
  sorry

end largest_square_area_l21_21781


namespace intersection_A_B_l21_21057

def A : Set ℤ := {-2, -1, 0, 1, 2}
def B : Set ℤ := { x | ∃ k : ℤ, x = 3 * k - 1 }

theorem intersection_A_B :
  A ∩ B = {-1, 2} :=
by
  sorry

end intersection_A_B_l21_21057


namespace find_a7_l21_21755

variable (a : ℕ → ℝ)

def arithmetic_sequence (d : ℝ) (a1 : ℝ) :=
  ∀ n, a n = a1 + (n - 1) * d

theorem find_a7
  (a : ℕ → ℝ)
  (d : ℝ)
  (a1 : ℝ)
  (h_arith : arithmetic_sequence a d a1)
  (h_a3 : a 3 = 7)
  (h_a5 : a 5 = 13):
  a 7 = 19 :=
by
  sorry

end find_a7_l21_21755


namespace find_common_ratio_l21_21774

noncomputable def geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (a 5 - a 1 = 15) ∧ (a 4 - a 2 = 6) → (q = 1/2 ∨ q = 2)

-- We declare this as a theorem statement
theorem find_common_ratio (a : ℕ → ℝ) (q : ℝ) : geometric_sequence_common_ratio a q :=
sorry

end find_common_ratio_l21_21774


namespace circle_area_difference_l21_21347

noncomputable def difference_of_circle_areas (C1 C2 : ℝ) : ℝ :=
  let π := Real.pi
  let r1 := C1 / (2 * π)
  let r2 := C2 / (2 * π)
  let A1 := π * r1 ^ 2
  let A2 := π * r2 ^ 2
  A2 - A1

theorem circle_area_difference :
  difference_of_circle_areas 396 704 = 26948.4 :=
by
  sorry

end circle_area_difference_l21_21347


namespace number_one_half_more_equals_twenty_five_percent_less_l21_21870

theorem number_one_half_more_equals_twenty_five_percent_less (n : ℤ) : 
    (80 - 0.25 * 80 = 60) → ((3 / 2 : ℚ) * n = 60) → (n = 40) :=
by
  intros h1 h2
  sorry

end number_one_half_more_equals_twenty_five_percent_less_l21_21870


namespace min_area_circle_tangent_l21_21311

theorem min_area_circle_tangent (h : ∀ (x : ℝ), x > 0 → y = 2 / x) : 
  ∃ (a b r : ℝ), (∀ (x : ℝ), x > 0 → 2 * a + b = 2 + 2 / x) ∧
  (∀ (x : ℝ), x > 0 → (x - 1)^2 + (y - 2)^2 = 5) :=
sorry

end min_area_circle_tangent_l21_21311


namespace total_bottles_per_day_l21_21367

def num_cases_per_day : ℕ := 7200
def bottles_per_case : ℕ := 10

theorem total_bottles_per_day : num_cases_per_day * bottles_per_case = 72000 := by
  sorry

end total_bottles_per_day_l21_21367


namespace Kayla_total_items_l21_21498

theorem Kayla_total_items (T_bars : ℕ) (T_cans : ℕ) (K_bars : ℕ) (K_cans : ℕ)
  (h1 : T_bars = 2 * K_bars) (h2 : T_cans = 2 * K_cans)
  (h3 : T_bars = 12) (h4 : T_cans = 18) : 
  K_bars + K_cans = 15 :=
by {
  -- In order to focus only on statement definition, we use sorry here
  sorry
}

end Kayla_total_items_l21_21498


namespace channel_bottom_width_l21_21286

theorem channel_bottom_width
  (area : ℝ)
  (top_width : ℝ)
  (depth : ℝ)
  (h_area : area = 880)
  (h_top_width : top_width = 14)
  (h_depth : depth = 80) :
  ∃ (b : ℝ), b = 8 ∧ area = (1/2) * (top_width + b) * depth := 
by
  sorry

end channel_bottom_width_l21_21286


namespace inequality_proof_l21_21387

theorem inequality_proof (a b c : ℝ) (h : a ^ 2 + b ^ 2 + c ^ 2 = 3) :
  (a ^ 2) / (2 + b + c ^ 2) + (b ^ 2) / (2 + c + a ^ 2) + (c ^ 2) / (2 + a + b ^ 2) ≥ (a + b + c) ^ 2 / 12 :=
by sorry

end inequality_proof_l21_21387


namespace farmhands_work_hours_l21_21889

def apples_per_pint (variety: String) : ℕ :=
  match variety with
  | "golden_delicious" => 20
  | "pink_lady" => 40
  | _ => 0

def total_apples_for_pints (pints: ℕ) : ℕ :=
  (apples_per_pint "golden_delicious") * pints + (apples_per_pint "pink_lady") * pints

def apples_picked_per_hour_per_farmhand : ℕ := 240

def num_farmhands : ℕ := 6

def total_apples_picked_per_hour : ℕ :=
  num_farmhands * apples_picked_per_hour_per_farmhand

def ratio_golden_to_pink : ℕ × ℕ := (1, 2)

def haley_cider_pints : ℕ := 120

def hours_worked (pints: ℕ) (picked_per_hour: ℕ): ℕ :=
  (total_apples_for_pints pints) / picked_per_hour

theorem farmhands_work_hours :
  hours_worked haley_cider_pints total_apples_picked_per_hour = 5 := by
  sorry

end farmhands_work_hours_l21_21889


namespace closest_integer_to_sqrt_11_l21_21951

theorem closest_integer_to_sqrt_11 : 
  ∀ (x : ℝ), (3 : ℝ) ≤ x → x ≤ 3.5 → x = 3 :=
by
  intro x hx h3_5
  sorry

end closest_integer_to_sqrt_11_l21_21951


namespace units_digit_2_pow_2130_l21_21841

theorem units_digit_2_pow_2130 : (Nat.pow 2 2130) % 10 = 4 :=
by sorry

end units_digit_2_pow_2130_l21_21841


namespace find_x_l21_21644

noncomputable def a : ℝ := Real.log 2 / Real.log 10
noncomputable def b : ℝ := 1 / a
noncomputable def log2_5 : ℝ := Real.log 5 / Real.log 2

theorem find_x (a₀ : a = 0.3010) : 
  ∃ x : ℝ, (log2_5 ^ 2 - a * log2_5 + x * b = 0) → 
  x = (log2_5 ^ 2 * 0.3010) :=
by
  sorry

end find_x_l21_21644


namespace times_faster_l21_21368

theorem times_faster (A B : ℝ) (h1 : A + B = 1 / 12) (h2 : A = 1 / 16) : 
  A / B = 3 :=
by
  sorry

end times_faster_l21_21368


namespace find_k_l21_21935

def f (x : ℝ) : ℝ := 3 * x ^ 2 - 2 * x + 8
def g (x : ℝ) (k : ℝ) : ℝ := x ^ 2 - k * x + 3

theorem find_k : 
  (f 5 - g 5 k = 12) → k = -53 / 5 :=
by
  intro hyp
  sorry

end find_k_l21_21935


namespace first_group_people_count_l21_21836

def group_ice_cream (P : ℕ) : Prop :=
  let total_days_per_person1 := P * 10
  let total_days_per_person2 := 5 * 16
  total_days_per_person1 = total_days_per_person2

theorem first_group_people_count 
  (P : ℕ) 
  (H1 : group_ice_cream P) : 
  P = 8 := 
sorry

end first_group_people_count_l21_21836


namespace marie_distance_biked_l21_21132

def biking_speed := 12.0 -- Speed in miles per hour
def biking_time := 2.583333333 -- Time in hours

theorem marie_distance_biked : biking_speed * biking_time = 31 := 
by 
  -- The proof steps go here
  sorry

end marie_distance_biked_l21_21132


namespace number_of_female_workers_l21_21923

theorem number_of_female_workers (M F : ℕ) (M_no F_no : ℝ) 
  (hM : M = 112)
  (h1 : M_no = 0.40 * M)
  (h2 : F_no = 0.25 * F)
  (h3 : M_no / (M_no + F_no) = 0.30)
  (h4 : F_no / (M_no + F_no) = 0.70)
  : F = 420 := 
by 
  sorry

end number_of_female_workers_l21_21923


namespace pupils_who_like_both_l21_21291

theorem pupils_who_like_both (total_pupils pizza_lovers burger_lovers : ℕ) (h1 : total_pupils = 200) (h2 : pizza_lovers = 125) (h3 : burger_lovers = 115) :
  (pizza_lovers + burger_lovers - total_pupils = 40) :=
by
  sorry

end pupils_who_like_both_l21_21291


namespace sum_of_solutions_l21_21580

theorem sum_of_solutions (x y : ℝ) (h₁ : y = 8) (h₂ : x^2 + y^2 = 144) : 
  ∃ x1 x2 : ℝ, (x1 = 4 * Real.sqrt 5 ∧ x2 = -4 * Real.sqrt 5) ∧ (x1 + x2 = 0) :=
by
  sorry

end sum_of_solutions_l21_21580


namespace range_of_m_l21_21966

theorem range_of_m 
  (m : ℝ)
  (f : ℝ → ℝ)
  (f_def : ∀ x, f x = x^3 + (m / 2 + 2) * x^2 - 2 * x)
  (f_prime : ℝ → ℝ)
  (f_prime_def : ∀ x, f_prime x = 3 * x^2 + (m + 4) * x - 2)
  (f_prime_at_1 : f_prime 1 < 0)
  (f_prime_at_2 : f_prime 2 < 0)
  (f_prime_at_3 : f_prime 3 > 0) :
  -37 / 3 < m ∧ m < -9 := 
  sorry

end range_of_m_l21_21966


namespace apple_juice_fraction_correct_l21_21804

def problem_statement : Prop :=
  let pitcher1_capacity := 800
  let pitcher2_capacity := 500
  let pitcher1_apple_fraction := 1 / 4
  let pitcher2_apple_fraction := 1 / 5
  let pitcher1_apple_volume := pitcher1_capacity * pitcher1_apple_fraction
  let pitcher2_apple_volume := pitcher2_capacity * pitcher2_apple_fraction
  let total_apple_volume := pitcher1_apple_volume + pitcher2_apple_volume
  let total_volume := pitcher1_capacity + pitcher2_capacity
  total_apple_volume / total_volume = 3 / 13

theorem apple_juice_fraction_correct : problem_statement := 
  sorry

end apple_juice_fraction_correct_l21_21804


namespace time_to_cross_lake_one_direction_l21_21823

-- Definitions for our conditions
def cost_per_hour := 10
def total_cost_round_trip := 80

-- Statement we want to prove
theorem time_to_cross_lake_one_direction : (total_cost_round_trip / cost_per_hour) / 2 = 4 :=
  by
  sorry

end time_to_cross_lake_one_direction_l21_21823


namespace determine_d_l21_21538

theorem determine_d (d c : ℕ) (hlcm : Nat.lcm 76 d = 456) (hhcf : Nat.gcd 76 d = c) : d = 24 :=
by
  sorry

end determine_d_l21_21538


namespace polynomial_remainder_l21_21984

theorem polynomial_remainder (c a b : ℤ) 
  (h1 : (16 * c + 8 * a + 2 * b = -12)) 
  (h2 : (81 * c - 27 * a - 3 * b = -85)) : 
  (a, b, c) = (5, 7, 1) :=
sorry

end polynomial_remainder_l21_21984


namespace solution_to_problem_l21_21097

def problem_statement : Prop :=
  (2.017 * 2016 - 10.16 * 201.7 = 2017)

theorem solution_to_problem : problem_statement :=
by
  sorry

end solution_to_problem_l21_21097


namespace nuts_per_student_l21_21618

theorem nuts_per_student (bags : ℕ) (nuts_per_bag : ℕ) (students : ℕ) (total_nuts : ℕ) (nuts_per_student : ℕ)
    (h1 : bags = 65)
    (h2 : nuts_per_bag = 15)
    (h3 : students = 13)
    (h4 : total_nuts = bags * nuts_per_bag)
    (h5 : nuts_per_student = total_nuts / students)
    : nuts_per_student = 75 :=
by
  sorry

end nuts_per_student_l21_21618


namespace representable_by_expression_l21_21479

theorem representable_by_expression (n : ℕ) :
  ∃ (x y z : ℕ), x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ (n = (x * y + y * z + z * x) / (x + y + z)) ↔ n ≠ 1 := by
  sorry

end representable_by_expression_l21_21479


namespace sqrt2_times_sqrt5_eq_sqrt10_l21_21222

theorem sqrt2_times_sqrt5_eq_sqrt10 : (Real.sqrt 2) * (Real.sqrt 5) = Real.sqrt 10 := 
by
  sorry

end sqrt2_times_sqrt5_eq_sqrt10_l21_21222


namespace minimum_seats_l21_21748

-- Condition: 150 seats in a row.
def seats : ℕ := 150

-- Assertion: The fewest number of seats that must be occupied so that any additional person seated must sit next to someone.
def minOccupiedSeats : ℕ := 50

theorem minimum_seats (s : ℕ) (m : ℕ) (h_seats : s = 150) (h_min : m = 50) :
  (∀ x, x = 150 → ∀ n, n ≥ 0 ∧ n ≤ m → 
    ∃ y, y ≥ 0 ∧ y ≤ x ∧ ∀ z, z = n + 1 → ∃ w, w ≥ 0 ∧ w ≤ x ∧ w = n ∨ w = n + 1) := 
sorry

end minimum_seats_l21_21748


namespace evaluate_ratio_l21_21647

theorem evaluate_ratio : (2^3002 * 3^3005 / 6^3003 : ℚ) = 9 / 2 := 
sorry

end evaluate_ratio_l21_21647


namespace train_speed_l21_21411

theorem train_speed (length_m : ℕ) (time_s : ℕ) (length_km : ℝ) (time_hr : ℝ) 
(length_conversion : length_km = (length_m : ℝ) / 1000)
(time_conversion : time_hr = (time_s : ℝ) / 3600)
(speed : ℝ) (speed_formula : speed = length_km / time_hr) :
  length_m = 300 → time_s = 18 → speed = 60 :=
by
  intros h1 h2
  rw [h1, h2] at *
  simp [length_conversion, time_conversion, speed_formula]
  norm_num
  sorry

end train_speed_l21_21411


namespace largest_square_side_length_largest_rectangle_dimensions_l21_21259

variable (a b : ℝ) (h : a > 0) (k : b > 0)

-- Part (a): Side length of the largest possible square
theorem largest_square_side_length (h : a > 0) (k : b > 0) :
  ∃ (s : ℝ), s = (a * b) / (a + b) := sorry

-- Part (b): Dimensions of the largest possible rectangle
theorem largest_rectangle_dimensions (h : a > 0) (k : b > 0) :
  ∃ (x y : ℝ), x = a / 2 ∧ y = b / 2 := sorry

end largest_square_side_length_largest_rectangle_dimensions_l21_21259


namespace arithmetic_sequence_problem_l21_21456

theorem arithmetic_sequence_problem (a : ℕ → ℕ) (S : ℕ → ℕ) (k : ℕ) 
  (h1 : ∀ n, S n = n * (a 1 + a n) / 2)
  (h2 : S 9 = 81)
  (h3 : a (k - 4) = 191)
  (h4 : S k = 10000) :
  k = 100 :=
by
  sorry

end arithmetic_sequence_problem_l21_21456


namespace percentage_employees_at_picnic_l21_21414

theorem percentage_employees_at_picnic (total_employees men_attend men_percentage women_attend women_percentage : ℝ)
  (h1 : men_attend = 0.20 * (men_percentage * total_employees))
  (h2 : women_attend = 0.40 * ((1 - men_percentage) * total_employees))
  (h3 : men_percentage = 0.30)
  : ((men_attend + women_attend) / total_employees) * 100 = 34 := by
sorry

end percentage_employees_at_picnic_l21_21414


namespace find_c_l21_21158

theorem find_c (c : ℝ) 
  (h : (⟨9, c⟩ : ℝ × ℝ) = (11/13 : ℝ) • ⟨-3, 2⟩) : 
  c = 19 :=
sorry

end find_c_l21_21158


namespace solve_rational_eq_l21_21944

theorem solve_rational_eq (x : ℝ) :
  (1 / (x^2 + 14*x - 36)) + (1 / (x^2 + 5*x - 14)) + (1 / (x^2 - 16*x - 36)) = 0 ↔ 
  x = 9 ∨ x = -4 ∨ x = 12 ∨ x = 3 :=
sorry

end solve_rational_eq_l21_21944


namespace range_of_f_l21_21364

noncomputable def f (x : ℝ) : ℝ := 4^x + 2^(x + 1) + 1

theorem range_of_f : Set.range f = {y : ℝ | y > 1} :=
by
  sorry

end range_of_f_l21_21364


namespace mul_3_6_0_5_l21_21969

theorem mul_3_6_0_5 : 3.6 * 0.5 = 1.8 :=
by
  sorry

end mul_3_6_0_5_l21_21969


namespace water_overflowed_calculation_l21_21947

/-- The water supply rate is 200 kilograms per hour. -/
def water_supply_rate : ℕ := 200

/-- The water tank capacity is 4000 kilograms. -/
def tank_capacity : ℕ := 4000

/-- The water runs for 24 hours. -/
def running_time : ℕ := 24

/-- Calculation for the kilograms of water that overflowed. -/
theorem water_overflowed_calculation :
  water_supply_rate * running_time - tank_capacity = 800 :=
by
  -- calculation skipped
  sorry

end water_overflowed_calculation_l21_21947


namespace num_distinct_units_digits_of_cubes_l21_21740

theorem num_distinct_units_digits_of_cubes : 
  ∃ S : Finset ℕ, (∀ d : ℕ, (d < 10) → (d^3 % 10) ∈ S) ∧ S.card = 10 := by
  sorry

end num_distinct_units_digits_of_cubes_l21_21740


namespace value_of_a_is_negative_one_l21_21783

-- Conditions
def I (a : ℤ) : Set ℤ := {2, 4, a^2 - a - 3}
def A (a : ℤ) : Set ℤ := {4, 1 - a}
def complement_I_A (a : ℤ) : Set ℤ := {x ∈ I a | x ∉ A a}

-- Theorem statement
theorem value_of_a_is_negative_one (a : ℤ) (h : complement_I_A a = {-1}) : a = -1 :=
by
  sorry

end value_of_a_is_negative_one_l21_21783


namespace h_at_3_l21_21762

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 + 3 * x + 5
noncomputable def g (x : ℝ) : ℝ := Real.sqrt (f x) + 1
noncomputable def h (x : ℝ) : ℝ := f (g x)

theorem h_at_3 : h 3 = 74 + 28 * Real.sqrt 2 :=
by
  sorry

end h_at_3_l21_21762


namespace prime_divisor_problem_l21_21507

theorem prime_divisor_problem (d r : ℕ) (h1 : d > 1) (h2 : Prime d)
  (h3 : 1274 % d = r) (h4 : 1841 % d = r) (h5 : 2866 % d = r) : d - r = 6 :=
by
  sorry

end prime_divisor_problem_l21_21507


namespace min_value_expression_l21_21516

theorem min_value_expression :
  ∀ (x y : ℝ), x > 0 → y > 0 → x + y = 1 → (∃ (c : ℝ), c = 16 ∧ ∀ z, z = (1 / x + 9 / y) → z ≥ c) :=
by
  sorry

end min_value_expression_l21_21516


namespace final_net_worth_l21_21022

noncomputable def initial_cash_A := (20000 : ℤ)
noncomputable def initial_cash_B := (22000 : ℤ)
noncomputable def house_value := (20000 : ℤ)
noncomputable def vehicle_value := (10000 : ℤ)

noncomputable def transaction_1_cash_A := initial_cash_A + 25000
noncomputable def transaction_1_cash_B := initial_cash_B - 25000

noncomputable def transaction_2_cash_A := transaction_1_cash_A - 12000
noncomputable def transaction_2_cash_B := transaction_1_cash_B + 12000

noncomputable def transaction_3_cash_A := transaction_2_cash_A + 18000
noncomputable def transaction_3_cash_B := transaction_2_cash_B - 18000

noncomputable def transaction_4_cash_A := transaction_3_cash_A + 9000
noncomputable def transaction_4_cash_B := transaction_3_cash_B + 9000

noncomputable def final_value_A := transaction_4_cash_A
noncomputable def final_value_B := transaction_4_cash_B + house_value + vehicle_value

theorem final_net_worth :
  final_value_A - initial_cash_A = 40000 ∧ final_value_B - initial_cash_B = 8000 :=
by
  sorry

end final_net_worth_l21_21022


namespace trapezoid_height_l21_21035

theorem trapezoid_height (AD BC : ℝ) (AB CD : ℝ) (h₁ : AD = 25) (h₂ : BC = 4) (h₃ : AB = 20) (h₄ : CD = 13) : ∃ h : ℝ, h = 12 :=
by
  -- Definitions
  let AD := 25
  let BC := 4
  let AB := 20
  let CD := 13
  
  sorry

end trapezoid_height_l21_21035


namespace factor_expression_l21_21515

theorem factor_expression (x : ℝ) :
  (16 * x^4 + 36 * x^2 - 9) - (4 * x^4 - 6 * x^2 - 9) = 6 * x^2 * (2 * x^2 + 7) :=
by
  sorry

end factor_expression_l21_21515


namespace zoe_spent_amount_l21_21807

theorem zoe_spent_amount :
  (3 * (8 + 2) = 30) :=
by sorry

end zoe_spent_amount_l21_21807


namespace distinct_integers_sum_l21_21465

theorem distinct_integers_sum {a b c d : ℤ} (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) 
  (h_product : a * b * c * d = 25) : a + b + c + d = 0 := by
  sorry

end distinct_integers_sum_l21_21465


namespace no_consecutive_days_played_l21_21723

theorem no_consecutive_days_played (john_interval mary_interval : ℕ) :
  john_interval = 16 ∧ mary_interval = 25 → 
  ¬ ∃ (n : ℕ), (n * john_interval + 1 = m * mary_interval ∨ n * john_interval = m * mary_interval + 1) :=
by
  sorry

end no_consecutive_days_played_l21_21723


namespace problem_proof_l21_21349

def problem_statement : Prop :=
  (1 / 4) * 8 * (1 / 16) * 32 * (1 / 64) * 128 * (1 / 256) * 512 * (1 / 1024) * 2048 * (1 / 4096) * 8192 = 64

theorem problem_proof : problem_statement := by
  sorry

end problem_proof_l21_21349


namespace quadratic_roots_expression_eq_zero_l21_21882

theorem quadratic_roots_expression_eq_zero
  (a b c : ℝ)
  (h : ∀ x : ℝ, a * x^2 + b * x + c = 0)
  (x1 x2 : ℝ)
  (hx1 : a * x1^2 + b * x1 + c = 0)
  (hx2 : a * x2^2 + b * x2 + c = 0)
  (s1 s2 s3 : ℝ)
  (h_s1 : s1 = x1 + x2)
  (h_s2 : s2 = x1^2 + x2^2)
  (h_s3 : s3 = x1^3 + x2^3) :
  a * s3 + b * s2 + c * s1 = 0 := sorry

end quadratic_roots_expression_eq_zero_l21_21882


namespace circle_radius_l21_21329

theorem circle_radius (x y : ℝ) : (x^2 - 4 * x + y^2 - 21 = 0) → (∃ r : ℝ, r = 5) :=
by
  sorry

end circle_radius_l21_21329


namespace intersect_single_point_l21_21577

theorem intersect_single_point (m : ℝ) : 
  (∃ x : ℝ, (m - 3) * x^2 - 4 * x + 2 = 0) ∧ ∀ x₁ x₂ : ℝ, 
  (m - 3) * x₁^2 - 4 * x₁ + 2 = 0 → (m - 3) * x₂^2 - 4 * x₂ + 2 = 0 → x₁ = x₂ ↔ m = 3 ∨ m = 5 := 
sorry

end intersect_single_point_l21_21577


namespace square_park_area_l21_21549

theorem square_park_area (side_length : ℝ) (h : side_length = 200) : side_length * side_length = 40000 := by
  sorry

end square_park_area_l21_21549


namespace part1_part2_l21_21531
noncomputable def f (x : ℝ) : ℝ := abs (x - 1) + abs (x - 2)

theorem part1 : {x : ℝ | f x ≥ 3} = {x | x ≤ 0} ∪ {x | x ≥ 3} :=
by
  sorry

theorem part2 (a : ℝ) : (∃ x : ℝ, f x ≤ -a^2 + a + 7) ↔ -2 ≤ a ∧ a ≤ 3 :=
by
  sorry

end part1_part2_l21_21531


namespace tan_alpha_problem_l21_21082

theorem tan_alpha_problem (α : ℝ) (h : Real.tan α = 3) : (4 * Real.sin α - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 5 / 7 := by
  sorry

end tan_alpha_problem_l21_21082


namespace smallest_n_Sn_pos_l21_21868

theorem smallest_n_Sn_pos {a : ℕ → ℤ} (S : ℕ → ℤ) 
  (h1 : ∀ n, S n = n * a 1 + (n * (n - 1) / 2) * (a 2 - a 1))
  (h2 : ∀ n, (n ≠ 5 → S n > S 5))
  (h3 : |a 5| > |a 6|) :
  ∃ n : ℕ, S n > 0 ∧ ∀ m < n, S m ≤ 0 :=
by 
  -- Actual proof steps would go here.
  sorry

end smallest_n_Sn_pos_l21_21868


namespace diving_assessment_l21_21120

theorem diving_assessment (total_athletes : ℕ) (selected_athletes : ℕ) (not_meeting_standard : ℕ) 
  (first_level_sample : ℕ) (first_level_total : ℕ) (athletes : Set ℕ) :
  total_athletes = 56 → 
  selected_athletes = 8 → 
  not_meeting_standard = 2 → 
  first_level_sample = 3 → 
  (∀ (A B C D E : ℕ), athletes = {A, B, C, D, E} → first_level_total = 5 → 
  (∃ proportion_standard number_first_level probability_E, 
    proportion_standard = (8 - 2) / 8 ∧  -- first part: proportion of athletes who met the standard
    number_first_level = 56 * (3 / 8) ∧ -- second part: number of first-level athletes
    probability_E = 4 / 10))           -- third part: probability of athlete E being chosen
:= sorry

end diving_assessment_l21_21120


namespace average_of_c_and_d_l21_21799

variable (c d e : ℝ)

theorem average_of_c_and_d
  (h1: (4 + 6 + 9 + c + d + e) / 6 = 20)
  (h2: e = c + 6) :
  (c + d) / 2 = 47.5 := by
sorry

end average_of_c_and_d_l21_21799


namespace boys_sitting_10_boys_sitting_11_l21_21086

def exists_two_boys_with_4_between (n : ℕ) : Prop :=
  ∃ (b : Finset ℕ), b.card = n ∧ ∀ (i j : ℕ) (h₁ : i ≠ j) (h₂ : i < 25) (h₃ : j < 25),
    (i + 5) % 25 = j

theorem boys_sitting_10 :
  ¬exists_two_boys_with_4_between 10 :=
sorry

theorem boys_sitting_11 :
  exists_two_boys_with_4_between 11 :=
sorry

end boys_sitting_10_boys_sitting_11_l21_21086


namespace find_D_l21_21906

noncomputable def Point : Type := ℝ × ℝ

-- Given points A, B, and C
def A : Point := (-2, 0)
def B : Point := (6, 8)
def C : Point := (8, 6)

-- Condition: AB parallel to DC and AD parallel to BC, which means it is a parallelogram
def is_parallelogram (A B C D : Point) : Prop :=
  ((B.1 - A.1, B.2 - A.2) = (D.1 - C.1, D.2 - C.2)) ∧
  ((C.1 - B.1, C.2 - B.2) = (D.1 - A.1, D.2 - A.2))

-- Proves that with given A, B, and C, D should be (0, -2)
theorem find_D : ∃ D : Point, is_parallelogram A B C D ∧ D = (0, -2) :=
  by sorry

end find_D_l21_21906


namespace find_a_minus_b_l21_21798

theorem find_a_minus_b
  (f : ℝ → ℝ)
  (a b : ℝ)
  (hf : ∀ x, f x = x^2 + 3 * a * x + 4)
  (h_even : ∀ x, f (-x) = f x)
  (hb_condition : b - 3 = -2 * b) :
  a - b = -1 :=
sorry

end find_a_minus_b_l21_21798


namespace find_chocolate_cakes_l21_21193

variable (C : ℕ)
variable (h1 : 12 * C + 6 * 22 = 168)

theorem find_chocolate_cakes : C = 3 :=
by
  -- this is the proof placeholder
  sorry

end find_chocolate_cakes_l21_21193


namespace m_divides_n_l21_21094

theorem m_divides_n 
  (m n : ℕ) 
  (hm_pos : 0 < m) 
  (hn_pos : 0 < n) 
  (h : 5 * m + n ∣ 5 * n + m) 
  : m ∣ n :=
sorry

end m_divides_n_l21_21094


namespace reciprocal_opposite_abs_val_l21_21243

theorem reciprocal_opposite_abs_val (a : ℚ) (h : a = -1 - 2/7) :
    (1 / a = -7/9) ∧ (-a = 1 + 2/7) ∧ (|a| = 1 + 2/7) := 
sorry

end reciprocal_opposite_abs_val_l21_21243


namespace euler_totient_divisibility_l21_21428

theorem euler_totient_divisibility (a n: ℕ) (h1 : a ≥ 2) : (n ∣ Nat.totient (a^n - 1)) :=
sorry

end euler_totient_divisibility_l21_21428


namespace triangle_area_solution_l21_21418

noncomputable def solve_for_x (x : ℝ) : Prop :=
  x > 0 ∧ (1 / 2 * x * 3 * x = 96) → x = 8

theorem triangle_area_solution : solve_for_x 8 :=
by
  sorry

end triangle_area_solution_l21_21418


namespace trapezoid_height_l21_21441

theorem trapezoid_height (a b : ℝ) (A : ℝ) (h : ℝ) : a = 5 → b = 9 → A = 56 → A = (1 / 2) * (a + b) * h → h = 8 :=
by 
  intros ha hb hA eqn
  sorry

end trapezoid_height_l21_21441


namespace tan_half_prod_eq_sqrt3_l21_21523

theorem tan_half_prod_eq_sqrt3 (a b : ℝ) (h : 7 * (Real.cos a + Real.cos b) + 3 * (Real.cos a * Real.cos b + 1) = 0) :
  ∃ (xy : ℝ), xy = Real.tan (a / 2) * Real.tan (b / 2) ∧ (xy = Real.sqrt 3 ∨ xy = -Real.sqrt 3) :=
by
  sorry

end tan_half_prod_eq_sqrt3_l21_21523


namespace max_min_fraction_l21_21129

-- Given condition
def circle_condition (x y : ℝ) : Prop := x^2 + y^2 + 2 * x - 4 * y + 1 = 0

-- Problem statement
theorem max_min_fraction (x y : ℝ) (h : circle_condition x y) :
  -20 / 21 ≤ y / (x - 4) ∧ y / (x - 4) ≤ 0 :=
sorry

end max_min_fraction_l21_21129


namespace initial_money_amount_l21_21867

theorem initial_money_amount 
  (X : ℝ) 
  (h : 0.70 * X = 350) : 
  X = 500 := 
sorry

end initial_money_amount_l21_21867


namespace toy_store_problem_l21_21275

variables (x y : ℕ)

theorem toy_store_problem (h1 : 8 * x + 26 * y + 33 * (31 - x - y) / 2 = 370)
                          (h2 : x + y + (31 - x - y) / 2 = 31) :
    x = 20 :=
sorry

end toy_store_problem_l21_21275


namespace range_of_m_l21_21078

theorem range_of_m (x : ℝ) (h₁ : 1/2 ≤ x) (h₂ : x ≤ 2) :
  2 - Real.log 2 ≤ -Real.log x + 3*x - x^2 ∧ -Real.log x + 3*x - x^2 ≤ 2 :=
sorry

end range_of_m_l21_21078


namespace original_money_in_wallet_l21_21901

-- Definitions based on the problem's conditions
def grandmother_gift : ℕ := 20
def aunt_gift : ℕ := 25
def uncle_gift : ℕ := 30
def cost_per_game : ℕ := 35
def number_of_games : ℕ := 3
def money_left : ℕ := 20

-- Calculations as specified in the solution
def birthday_money := grandmother_gift + aunt_gift + uncle_gift
def total_game_cost := cost_per_game * number_of_games
def total_money_before_purchase := total_game_cost + money_left

-- Proof that the original amount of money in Geoffrey's wallet
-- was €50 before he got the birthday money and made the purchase.
theorem original_money_in_wallet : total_money_before_purchase - birthday_money = 50 := by
  sorry

end original_money_in_wallet_l21_21901


namespace correct_factorization_l21_21747

theorem correct_factorization:
  (∃ a : ℝ, (a + 3) * (a - 3) = a ^ 2 - 9) ∧
  (∃ x : ℝ, x ^ 2 + x - 5 = x * (x + 1) - 5) ∧
  ¬ (∃ x : ℝ, x ^ 2 + 1 = x * (x + 1 / x)) ∧
  (∃ x : ℝ, x ^ 2 + 4 * x + 4 = (x + 2) ^ 2) →
  (∃ x : ℝ, x ^ 2 + 4 * x + 4 = (x + 2) ^ 2)
  := by
  sorry

end correct_factorization_l21_21747


namespace find_diameter_l21_21909

noncomputable def cost_per_meter : ℝ := 2
noncomputable def total_cost : ℝ := 188.49555921538757
noncomputable def circumference (c : ℝ) (p : ℝ) : ℝ := c / p
noncomputable def diameter (c : ℝ) : ℝ := c / Real.pi

theorem find_diameter :
  diameter (circumference total_cost cost_per_meter) = 30 := by
  sorry

end find_diameter_l21_21909


namespace find_coords_of_P_cond1_find_coords_of_P_cond2_find_coords_of_P_cond3_l21_21826

variables {m : ℝ} 
def point_on_y_axis (P : (ℝ × ℝ)) := P = (0, -3)
def point_distance_to_y_axis (P : (ℝ × ℝ)) := P = (6, 0) ∨ P = (-6, -6)
def point_in_third_quadrant_and_equidistant (P : (ℝ × ℝ)) := P = (-6, -6)

theorem find_coords_of_P_cond1 (P : ℝ × ℝ) (h : 2 * m + 4 = 0) : point_on_y_axis P ↔ P = (0, -3) :=
by {
  sorry
}

theorem find_coords_of_P_cond2 (P : ℝ × ℝ) (h : abs (2 * m + 4) = 6) : point_distance_to_y_axis P ↔ (P = (6, 0) ∨ P = (-6, -6)) :=
by {
  sorry
}

theorem find_coords_of_P_cond3 (P : ℝ × ℝ) (h1 : 2 * m + 4 < 0) (h2 : m - 1 < 0) (h3 : abs (2 * m + 4) = abs (m - 1)) : point_in_third_quadrant_and_equidistant P ↔ P = (-6, -6) :=
by {
  sorry
}

end find_coords_of_P_cond1_find_coords_of_P_cond2_find_coords_of_P_cond3_l21_21826


namespace ticket_number_l21_21059

-- Define the conditions and the problem
theorem ticket_number (x y z N : ℕ) (hx : 0 ≤ x ∧ x ≤ 9) (hy: 0 ≤ y ∧ y ≤ 9) (hz: 0 ≤ z ∧ z ≤ 9) 
(hN1: N = 100 * x + 10 * y + z) (hN2: N = 11 * (x + y + z)) : 
N = 198 :=
sorry

end ticket_number_l21_21059


namespace find_triplets_l21_21722

theorem find_triplets (x y z : ℕ) :
  (x^2 + y^2 = 3 * 2016^z + 77) →
  (x, y, z) = (77, 14, 1) ∨ (x, y, z) = (14, 77, 1) ∨ 
  (x, y, z) = (70, 35, 1) ∨ (x, y, z) = (35, 70, 1) ∨ 
  (x, y, z) = (8, 4, 0) ∨ (x, y, z) = (4, 8, 0) :=
by
  sorry

end find_triplets_l21_21722


namespace river_depth_ratio_l21_21865

-- Definitions based on the conditions
def depthMidMay : ℝ := 5
def increaseMidJune : ℝ := 10
def depthMidJune : ℝ := depthMidMay + increaseMidJune
def depthMidJuly : ℝ := 45

-- The theorem based on the question and correct answer
theorem river_depth_ratio : depthMidJuly / depthMidJune = 3 := by 
  -- Proof skipped for illustration purposes
  sorry

end river_depth_ratio_l21_21865


namespace find_square_sum_l21_21098

theorem find_square_sum (x y z : ℝ)
  (h1 : x^2 - 6 * y = 10)
  (h2 : y^2 - 8 * z = -18)
  (h3 : z^2 - 10 * x = -40) :
  x^2 + y^2 + z^2 = 50 :=
sorry

end find_square_sum_l21_21098


namespace mailman_distribution_l21_21251

theorem mailman_distribution 
    (total_mail_per_block : ℕ)
    (blocks : ℕ)
    (houses_per_block : ℕ)
    (h1 : total_mail_per_block = 32)
    (h2 : blocks = 55)
    (h3 : houses_per_block = 4) :
  total_mail_per_block / houses_per_block = 8 :=
by
  sorry

end mailman_distribution_l21_21251


namespace no5_battery_mass_l21_21373

theorem no5_battery_mass :
  ∃ (x y : ℝ), 2 * x + 2 * y = 72 ∧ 3 * x + 2 * y = 96 ∧ x = 24 :=
by
  sorry

end no5_battery_mass_l21_21373


namespace total_points_first_half_l21_21526

def geometric_sum (a r : ℕ) (n : ℕ) : ℕ :=
  a * (1 - r ^ n) / (1 - r)

def arithmetic_sum (a d : ℕ) (n : ℕ) : ℕ :=
  n * a + d * (n * (n - 1) / 2)

-- Given conditions:
variables (a r b d : ℕ)
variables (h1 : a = b)
variables (h2 : geometric_sum a r 4 = b + (b + d) + (b + 2 * d) + (b + 3 * d) + 2)
variables (h3 : a * (1 + r + r^2 + r^3) ≤ 120)
variables (h4 : b + (b + d) + (b + 2 * d) + (b + 3 * d) ≤ 120)

theorem total_points_first_half (a r b d : ℕ) (h1 : a = b) (h2 : a * (1 + r + r ^ 2 + r ^ 3) = b + (b + d) + (b + 2 * d) + (b + 3 * d) + 2)
  (h3 : a * (1 + r + r ^ 2 + r ^ 3) ≤ 120) (h4 : b + (b + d) + (b + 2 * d) + (b + 3 * d) ≤ 120) : 
  a + a * r + b + (b + d) = 45 :=
by
  sorry

end total_points_first_half_l21_21526


namespace Kim_drink_amount_l21_21049

namespace MathProof

-- Define the conditions
variable (milk_initial t_drinks k_drinks : ℚ)
variable (H1 : milk_initial = 3/4)
variable (H2 : t_drinks = 1/3 * milk_initial)
variable (H3 : k_drinks = 1/2 * (milk_initial - t_drinks))

-- Theorem statement
theorem Kim_drink_amount : k_drinks = 1/4 :=
by
  sorry -- Proof steps would go here, but we're just setting up the statement

end MathProof

end Kim_drink_amount_l21_21049


namespace sum_infinite_series_eq_l21_21471

theorem sum_infinite_series_eq : 
  ∑' n : ℕ, (n + 1) * (1 / 999 : ℝ) ^ n = 1000 / 998 := by
sorry

end sum_infinite_series_eq_l21_21471


namespace largest_fraction_addition_l21_21490

-- Definitions for the problem conditions
def proper_fraction (a b : ℕ) : Prop :=
  a < b

def denom_less_than (d : ℕ) (bound : ℕ) : Prop :=
  d < bound

-- Main statement of the problem
theorem largest_fraction_addition :
  ∃ (a b : ℕ), (b > 0) ∧ proper_fraction (b + 7 * a) (7 * b) ∧ denom_less_than b 5 ∧ (a / b : ℚ) <= 3/4 := 
sorry

end largest_fraction_addition_l21_21490


namespace solve_for_diamond_l21_21306

theorem solve_for_diamond (d : ℕ) (h : d * 5 + 3 = d * 6 + 2) : d = 1 :=
by
  sorry

end solve_for_diamond_l21_21306


namespace abs_diff_one_l21_21024

theorem abs_diff_one (a b : ℤ) (h : |a| + |b| = 1) : |a - b| = 1 := sorry

end abs_diff_one_l21_21024


namespace perpendicular_vectors_x_value_l21_21400

theorem perpendicular_vectors_x_value :
  let a := (4, 2)
  let b := (x, 3)
  a.1 * b.1 + a.2 * b.2 = 0 -> x = -3/2 :=
by
  intros
  sorry

end perpendicular_vectors_x_value_l21_21400


namespace common_difference_arithmetic_sequence_l21_21134

theorem common_difference_arithmetic_sequence (a b : ℝ) :
  ∃ d : ℝ, b = a + 6 * d ∧ d = (b - a) / 6 :=
by
  sorry

end common_difference_arithmetic_sequence_l21_21134


namespace find_base_b_l21_21187

theorem find_base_b (b : ℕ) : ( (2 * b + 5) ^ 2 = 6 * b ^ 2 + 5 * b + 5 ) → b = 9 := 
by 
  sorry  -- Proof is not required as per instruction

end find_base_b_l21_21187


namespace largest_common_value_less_than_1000_l21_21821

theorem largest_common_value_less_than_1000 :
  ∃ a : ℕ, a = 999 ∧ (∃ n m : ℕ, a = 4 + 5 * n ∧ a = 7 + 8 * m) ∧ a < 1000 :=
by
  sorry

end largest_common_value_less_than_1000_l21_21821


namespace rubies_in_chest_l21_21603

theorem rubies_in_chest (R : ℕ) (h₁ : 421 = R + 44) : R = 377 :=
by 
  sorry

end rubies_in_chest_l21_21603


namespace angles_terminal_side_equiv_l21_21547

theorem angles_terminal_side_equiv (k : ℤ) : (2 * k * Real.pi + Real.pi) % (2 * Real.pi) = (4 * k * Real.pi + Real.pi) % (2 * Real.pi) ∨ (2 * k * Real.pi + Real.pi) % (2 * Real.pi) = (4 * k * Real.pi - Real.pi) % (2 * Real.pi) :=
sorry

end angles_terminal_side_equiv_l21_21547


namespace value_of_bc_l21_21496

theorem value_of_bc (a b c d : ℝ) (h1 : a + b = 14) (h2 : c + d = 3) (h3 : a + d = 8) : b + c = 9 :=
sorry

end value_of_bc_l21_21496


namespace min_value_xy_expression_l21_21030

theorem min_value_xy_expression : ∃ x y : ℝ, (xy - 2)^2 + (x^2 + y^2) = 4 :=
by
  sorry

end min_value_xy_expression_l21_21030


namespace solve_for_x_l21_21142

theorem solve_for_x (x : ℤ) (h : (3012 + x)^2 = x^2) : x = -1506 := 
sorry

end solve_for_x_l21_21142


namespace pounds_of_oranges_l21_21943

noncomputable def price_of_pounds_oranges (E O : ℝ) (P : ℕ) : Prop :=
  let current_total_price := E
  let increased_total_price := 1.09 * E + 1.06 * (O * P)
  (increased_total_price - current_total_price) = 15

theorem pounds_of_oranges (E O : ℝ) (P : ℕ): 
  E = O * P ∧ 
  (price_of_pounds_oranges E O P) → 
  P = 100 := 
by
  sorry

end pounds_of_oranges_l21_21943


namespace incorrect_tripling_radius_l21_21209

-- Let r be the radius of a circle, and A be its area.
-- The claim is that tripling the radius quadruples the area.
-- We need to prove this claim is incorrect.

theorem incorrect_tripling_radius (r : ℝ) (A : ℝ) (π : ℝ) (hA : A = π * r^2) : 
    (π * (3 * r)^2) ≠ 4 * A :=
by
  sorry

end incorrect_tripling_radius_l21_21209


namespace angle_same_after_minutes_l21_21232

def angle_between_hands (H M : ℝ) : ℝ :=
  abs (30 * H - 5.5 * M)

theorem angle_same_after_minutes (x : ℝ) :
  x = 54 + 6 / 11 → 
  angle_between_hands (5 + (x / 60)) x = 150 :=
by
  sorry

end angle_same_after_minutes_l21_21232


namespace supremum_neg_frac_bound_l21_21676

noncomputable def supremum_neg_frac (a b : ℝ) : ℝ :=
  - (1 / (2 * a)) - (2 / b)

theorem supremum_neg_frac_bound (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  supremum_neg_frac a b ≤ - 9 / 2 :=
sorry

end supremum_neg_frac_bound_l21_21676


namespace carrie_worked_days_l21_21744

theorem carrie_worked_days (d : ℕ) 
  (h1: ∀ n : ℕ, d = n → (2 * 22 * n - 54 = 122)) : d = 4 :=
by
  -- The proof will go here.
  sorry

end carrie_worked_days_l21_21744


namespace balls_in_boxes_with_one_in_one_balls_in_boxes_with_two_empty_balls_in_boxes_with_three_empty_balls_in_boxes_A_not_less_B_l21_21491

noncomputable def ways_with_ball_in_box_one : Nat := 369
noncomputable def ways_with_two_empty_boxes : Nat := 360
noncomputable def ways_with_three_empty_boxes : Nat := 140
noncomputable def ways_ball_A_not_less_than_B : Nat := 375

theorem balls_in_boxes_with_one_in_one 
  (n_balls : Nat) (n_boxes : Nat) 
  (ball_1 : Nat) :
  n_balls = 4 → n_boxes = 5 → ball_1 = 1 → 
  ∃ ways, ways = ways_with_ball_in_box_one := 
sorry

theorem balls_in_boxes_with_two_empty 
  (n_balls : Nat) (n_boxes : Nat) 
  (empty_boxes : Nat) :
  n_balls = 4 → n_boxes = 5 → empty_boxes = 2 → 
  ∃ ways, ways = ways_with_two_empty_boxes := 
sorry

theorem balls_in_boxes_with_three_empty 
  (n_balls : Nat) (n_boxes : Nat) 
  (empty_boxes : Nat) :
  n_balls = 4 → n_boxes = 5 → empty_boxes = 3 → 
  ∃ ways, ways = ways_with_three_empty_boxes := 
sorry

theorem balls_in_boxes_A_not_less_B 
  (n_balls : Nat) (n_boxes : Nat) 
  (ball_A : Nat) (ball_B : Nat) :
  n_balls = 4 → n_boxes = 5 → ball_A ≠ ball_B →
  ∃ ways, ways = ways_ball_A_not_less_than_B := 
sorry

end balls_in_boxes_with_one_in_one_balls_in_boxes_with_two_empty_balls_in_boxes_with_three_empty_balls_in_boxes_A_not_less_B_l21_21491


namespace altered_solution_contains_correct_detergent_volume_l21_21717

-- Define the original and altered ratios.
def original_ratio : ℝ × ℝ × ℝ := (2, 25, 100)
def altered_ratio_bleach_to_detergent : ℝ × ℝ := (6, 25)
def altered_ratio_detergent_to_water : ℝ × ℝ := (25, 200)

-- Define the given condition about the amount of water in the altered solution.
def altered_solution_water_volume : ℝ := 300

-- Define a function for the total altered solution volume and detergent volume
noncomputable def altered_solution_detergent_volume (water_volume : ℝ) : ℝ :=
  let detergent_volume := (altered_ratio_detergent_to_water.1 * water_volume) / altered_ratio_detergent_to_water.2
  detergent_volume

-- The proof statement asserting the amount of detergent in the altered solution.
theorem altered_solution_contains_correct_detergent_volume :
  altered_solution_detergent_volume altered_solution_water_volume = 37.5 :=
by
  sorry

end altered_solution_contains_correct_detergent_volume_l21_21717


namespace product_of_numbers_l21_21218

theorem product_of_numbers (a b : ℕ) (hcf : ℕ := 12) (lcm : ℕ := 205) (ha : Nat.gcd a b = hcf) (hb : Nat.lcm a b = lcm) : a * b = 2460 := by
  sorry

end product_of_numbers_l21_21218


namespace increasing_iff_a_ge_half_l21_21886

noncomputable def f (a x : ℝ) : ℝ := (2 / 3) * x ^ 3 + (1 / 2) * (a - 1) * x ^ 2 + a * x + 1

theorem increasing_iff_a_ge_half (a : ℝ) :
  (∀ x, 1 < x ∧ x < 2 → (2 * x ^ 2 + (a - 1) * x + a) ≥ 0) ↔ a ≥ -1 / 2 :=
sorry

end increasing_iff_a_ge_half_l21_21886


namespace valid_third_side_l21_21578

theorem valid_third_side (a b c : ℝ) (h₁ : a = 3) (h₂ : b = 8) (h₃ : 5 < c) (h₄ : c < 11) : c = 8 := 
by 
  sorry

end valid_third_side_l21_21578


namespace cat_ratio_l21_21077

theorem cat_ratio (jacob_cats annie_cats melanie_cats : ℕ)
  (H1 : jacob_cats = 90)
  (H2 : annie_cats = jacob_cats / 3)
  (H3 : melanie_cats = 60) :
  melanie_cats / annie_cats = 2 := 
  by
  sorry

end cat_ratio_l21_21077


namespace tickets_count_l21_21736

theorem tickets_count (x y: ℕ) (h : 3 * x + 5 * y = 78) : 
  ∃ n : ℕ , n = 6 :=
sorry

end tickets_count_l21_21736


namespace contrapositive_squared_l21_21395

theorem contrapositive_squared (a : ℝ) : (a ≤ 0 → a^2 ≤ 0) ↔ (a > 0 → a^2 > 0) :=
by
  sorry

end contrapositive_squared_l21_21395


namespace student_correct_answers_l21_21855

theorem student_correct_answers (C W : ℕ) (h1 : C + W = 60) (h2 : 4 * C - W = 140) : C = 40 :=
by
  sorry

end student_correct_answers_l21_21855


namespace correct_choice_is_C_l21_21014

def is_opposite_number (a b : ℤ) : Prop := a + b = 0

def option_A : Prop := ¬is_opposite_number (2^3) (3^2)
def option_B : Prop := ¬is_opposite_number (-2) (-|-2|)
def option_C : Prop := is_opposite_number ((-3)^2) (-3^2)
def option_D : Prop := ¬is_opposite_number 2 (-(-2))

theorem correct_choice_is_C : option_C ∧ option_A ∧ option_B ∧ option_D :=
by
  sorry

end correct_choice_is_C_l21_21014


namespace evaluate_expression_l21_21320

theorem evaluate_expression : (527 * 527 - 526 * 528) = 1 :=
by
  sorry

end evaluate_expression_l21_21320


namespace Julie_work_hours_per_week_l21_21064

theorem Julie_work_hours_per_week 
  (hours_summer_per_week : ℕ)
  (weeks_summer : ℕ)
  (total_earnings_summer : ℕ)
  (planned_weeks_school_year : ℕ)
  (needed_income_school_year : ℕ)
  (hourly_wage : ℝ := total_earnings_summer / (hours_summer_per_week * weeks_summer))
  (total_hours_needed_school_year : ℝ := needed_income_school_year / hourly_wage)
  (hours_per_week_needed : ℝ := total_hours_needed_school_year / planned_weeks_school_year) :
  hours_summer_per_week = 60 →
  weeks_summer = 8 →
  total_earnings_summer = 6000 →
  planned_weeks_school_year = 40 →
  needed_income_school_year = 10000 →
  hours_per_week_needed = 20 :=
by 
  intros h1 h2 h3 h4 h5
  sorry

end Julie_work_hours_per_week_l21_21064


namespace MelAge_when_Katherine24_l21_21454

variable (Katherine Mel : ℕ)

-- Conditions
def isYounger (Mel Katherine : ℕ) : Prop :=
  Mel = Katherine - 3

def is24yearsOld (Katherine : ℕ) : Prop :=
  Katherine = 24

-- Statement to Prove
theorem MelAge_when_Katherine24 (Katherine Mel : ℕ) 
  (h1 : isYounger Mel Katherine) 
  (h2 : is24yearsOld Katherine) : 
  Mel = 21 := 
by 
  sorry

end MelAge_when_Katherine24_l21_21454


namespace nancy_total_spent_l21_21143

def crystal_cost : ℕ := 9
def metal_cost : ℕ := 10
def total_crystal_cost : ℕ := crystal_cost
def total_metal_cost : ℕ := 2 * metal_cost
def total_cost : ℕ := total_crystal_cost + total_metal_cost

theorem nancy_total_spent : total_cost = 29 := by
  sorry

end nancy_total_spent_l21_21143


namespace savings_after_purchase_l21_21686

theorem savings_after_purchase :
  let price_sweater := 30
  let price_scarf := 20
  let num_sweaters := 6
  let num_scarves := 6
  let savings := 500
  let total_cost := (num_sweaters * price_sweater) + (num_scarves * price_scarf)
  savings - total_cost = 200 :=
by
  sorry

end savings_after_purchase_l21_21686


namespace total_area_at_stage_4_l21_21138

/-- Define the side length of the square at a given stage -/
def side_length (n : ℕ) : ℕ := n + 2

/-- Define the area of the square at a given stage -/
def area (n : ℕ) : ℕ := (side_length n) ^ 2

/-- State the theorem -/
theorem total_area_at_stage_4 : 
  (area 0) + (area 1) + (area 2) + (area 3) = 86 :=
by
  -- proof goes here
  sorry

end total_area_at_stage_4_l21_21138


namespace total_texts_sent_l21_21431

theorem total_texts_sent (grocery_texts : ℕ) (response_texts_ratio : ℕ) (police_texts_percentage : ℚ) :
  grocery_texts = 5 →
  response_texts_ratio = 5 →
  police_texts_percentage = 0.10 →
  let response_texts := grocery_texts * response_texts_ratio
  let previous_texts := response_texts + grocery_texts
  let police_texts := previous_texts * police_texts_percentage
  response_texts + grocery_texts + police_texts = 33 :=
by
  sorry

end total_texts_sent_l21_21431


namespace problem_b_problem_c_problem_d_l21_21108

variable (a b : ℝ)

theorem problem_b (h : a * b > 0) :
  2 * (a^2 + b^2) ≥ (a + b)^2 :=
sorry

theorem problem_c (h : a * b > 0) :
  (b / a) + (a / b) ≥ 2 :=
sorry

theorem problem_d (h : a * b > 0) :
  (a + 1 / a) * (b + 1 / b) ≥ 4 :=
sorry

end problem_b_problem_c_problem_d_l21_21108


namespace true_propositions_l21_21691

-- Definitions according to conditions:
def p (x y : ℝ) : Prop := x > y → -x < -y
def q (x y : ℝ) : Prop := x > y → x^2 > y^2

-- Given that p is true and q is false.
axiom p_true {x y : ℝ} : p x y
axiom q_false {x y : ℝ} : ¬ q x y

-- Proving the actual propositions that are true:
theorem true_propositions (x y : ℝ) : 
  (p x y ∨ q x y) ∧ (p x y ∧ ¬ q x y) :=
by
  have h1 : p x y := p_true
  have h2 : ¬ q x y := q_false
  constructor
  · left; exact h1
  · constructor; assumption; assumption

end true_propositions_l21_21691


namespace probability_of_event_l21_21384

theorem probability_of_event (favorable unfavorable : ℕ) (h : favorable = 3) (h2 : unfavorable = 5) :
  (favorable / (favorable + unfavorable) : ℚ) = 3 / 8 :=
by
  sorry

end probability_of_event_l21_21384


namespace units_digit_periodic_10_l21_21666

theorem units_digit_periodic_10:
  ∀ n: ℕ, (n * (n + 1) * (n + 2)) % 10 = ((n + 10) * (n + 11) * (n + 12)) % 10 :=
by
  sorry

end units_digit_periodic_10_l21_21666


namespace compute_expression_l21_21130

noncomputable def a : ℝ := 125^(1/3)
noncomputable def b : ℝ := (-2/3)^0
noncomputable def c : ℝ := Real.log 8 / Real.log 2

theorem compute_expression : a - b - c = 1 := by
  sorry

end compute_expression_l21_21130


namespace cubic_expression_value_l21_21597

theorem cubic_expression_value (m : ℝ) (h : m^2 + m - 1 = 0) : m^3 + 2 * m^2 + 2010 = 2011 :=
by sorry

end cubic_expression_value_l21_21597


namespace find_same_goldfish_number_l21_21958

noncomputable def B (n : ℕ) : ℕ := 3 * 4^n
noncomputable def G (n : ℕ) : ℕ := 243 * 3^n

theorem find_same_goldfish_number : ∃ n, B n = G n :=
by sorry

end find_same_goldfish_number_l21_21958


namespace Fran_speed_l21_21575

-- Definitions needed for statements
def Joann_speed : ℝ := 15
def Joann_time : ℝ := 4
def Fran_time : ℝ := 5

-- Formalize the problem in Lean
theorem Fran_speed (Joann_distance : ℝ) (Fran_speed : ℝ) : 
  Joann_distance = Joann_speed * Joann_time →
  Fran_speed * Fran_time = Joann_distance →
  Fran_speed = 12 :=
by
  -- assume the conditions about distances
  intros h1 h2
  -- prove the goal
  sorry

end Fran_speed_l21_21575


namespace equilateral_triangle_l21_21280

theorem equilateral_triangle (a b c : ℝ) (h1 : b^2 = a * c) (h2 : 2 * b = a + c) : a = b ∧ b = c ∧ a = c := by
  sorry

end equilateral_triangle_l21_21280


namespace cos_alpha_value_l21_21978

open Real

theorem cos_alpha_value (α : ℝ) (h1 : sin (α + π / 3) = 3 / 5) (h2 : π / 6 < α ∧ α < 5 * π / 6) :
  cos α = (3 * sqrt 3 - 4) / 10 :=
by
  sorry

end cos_alpha_value_l21_21978


namespace complete_the_square_l21_21328

theorem complete_the_square (x : ℝ) :
  x^2 - 8 * x + 5 = 0 ↔ (x - 4)^2 = 11 :=
by
  sorry

end complete_the_square_l21_21328


namespace greatest_odd_integer_l21_21812

theorem greatest_odd_integer (x : ℕ) (h_odd : x % 2 = 1) (h_pos : x > 0) (h_ineq : x^2 < 50) : x = 7 :=
by sorry

end greatest_odd_integer_l21_21812


namespace initial_manufacturing_cost_l21_21240

theorem initial_manufacturing_cost
  (P : ℝ) -- selling price
  (initial_cost new_cost : ℝ)
  (initial_profit new_profit : ℝ)
  (h1 : initial_profit = 0.25 * P)
  (h2 : new_profit = 0.50 * P)
  (h3 : new_cost = 50)
  (h4 : new_profit = P - new_cost)
  (h5 : initial_profit = P - initial_cost) :
  initial_cost = 75 := 
by
  sorry

end initial_manufacturing_cost_l21_21240


namespace total_people_in_club_after_5_years_l21_21738

noncomputable def club_initial_people := 18
noncomputable def executives_per_year := 6
noncomputable def initial_regular_members := club_initial_people - executives_per_year

-- Define the function for regular members growth
noncomputable def regular_members_after_n_years (n : ℕ) : ℕ := initial_regular_members * 2 ^ n

-- Total people in the club after 5 years
theorem total_people_in_club_after_5_years : 
  club_initial_people + regular_members_after_n_years 5 - initial_regular_members = 390 :=
by
  sorry

end total_people_in_club_after_5_years_l21_21738


namespace total_fruits_purchased_l21_21033

-- Defining the costs of apples and bananas
def cost_per_apple : ℝ := 0.80
def cost_per_banana : ℝ := 0.70

-- Defining the total cost the customer spent
def total_cost : ℝ := 6.50

-- Defining the total number of fruits purchased as 9
theorem total_fruits_purchased (A B : ℕ) : 
  (cost_per_apple * A + cost_per_banana * B = total_cost) → 
  (A + B = 9) :=
by
  sorry

end total_fruits_purchased_l21_21033


namespace points_lie_on_ellipse_l21_21884

open Real

noncomputable def curve_points_all_lie_on_ellipse (s: ℝ) : Prop :=
  let x := 2 * cos s + 2 * sin s
  let y := 4 * (cos s - sin s)
  (x^2 / 8 + y^2 / 32 = 1)

-- Below statement defines the theorem we aim to prove:
theorem points_lie_on_ellipse (s: ℝ) : curve_points_all_lie_on_ellipse s :=
sorry -- This "sorry" is to indicate that the proof is omitted.

end points_lie_on_ellipse_l21_21884


namespace cone_from_sector_radius_l21_21180

theorem cone_from_sector_radius (r : ℝ) (slant_height : ℝ) : 
  (r = 9) ∧ (slant_height = 12) ↔ 
  (∃ (sector_angle : ℝ) (sector_radius : ℝ), 
    sector_angle = 270 ∧ sector_radius = 12 ∧ 
    slant_height = sector_radius ∧ 
    (2 * π * r = sector_angle / 360 * 2 * π * sector_radius)) :=
by
  sorry

end cone_from_sector_radius_l21_21180


namespace max_a_avoiding_lattice_points_l21_21668

def is_lattice_point (x y : ℤ) : Prop :=
  true  -- Placeholder for (x, y) being in lattice points.

def passes_through_lattice_point (m : ℚ) (x : ℤ) : Prop :=
  is_lattice_point x (⌊m * x + 2⌋)

theorem max_a_avoiding_lattice_points :
  ∀ {a : ℚ}, (∀ x : ℤ, (0 < x ∧ x ≤ 100) → ¬passes_through_lattice_point ((1 : ℚ) / 2) x ∧ ¬passes_through_lattice_point (a - 1) x) →
  a = 50 / 99 :=
by
  sorry

end max_a_avoiding_lattice_points_l21_21668


namespace num_orange_juice_l21_21544

-- Definitions based on the conditions in the problem
def O : ℝ := sorry -- To represent the number of bottles of orange juice
def A : ℝ := sorry -- To represent the number of bottles of apple juice
def cost_orange_juice : ℝ := 0.70
def cost_apple_juice : ℝ := 0.60
def total_cost : ℝ := 46.20
def total_bottles : ℝ := 70

-- Conditions used as definitions in Lean 4
axiom condition1 : O + A = total_bottles
axiom condition2 : cost_orange_juice * O + cost_apple_juice * A = total_cost

-- Proof statement with the correct answer
theorem num_orange_juice : O = 42 := by
  sorry

end num_orange_juice_l21_21544


namespace sum_of_coordinates_D_l21_21313

theorem sum_of_coordinates_D (x y : ℝ) 
  (M_midpoint : (4, 10) = ((8 + x) / 2, (6 + y) / 2)) : 
  x + y = 14 := 
by 
  sorry

end sum_of_coordinates_D_l21_21313


namespace find_percentage_l21_21169

theorem find_percentage (P N : ℕ) (h1 : N = 100) (h2 : (P : ℝ) / 100 * N = 50 / 100 * 40 + 10) :
  P = 30 :=
by
  sorry

end find_percentage_l21_21169


namespace probability_of_one_fork_one_spoon_one_knife_l21_21640

theorem probability_of_one_fork_one_spoon_one_knife 
  (num_forks : ℕ) (num_spoons : ℕ) (num_knives : ℕ) (total_pieces : ℕ)
  (h_forks : num_forks = 7) (h_spoons : num_spoons = 8) (h_knives : num_knives = 5)
  (h_total : total_pieces = num_forks + num_spoons + num_knives) :
  (∃ (prob : ℚ), prob = 14 / 57) :=
by
  sorry

end probability_of_one_fork_one_spoon_one_knife_l21_21640


namespace symmetrical_point_wrt_x_axis_l21_21950

theorem symmetrical_point_wrt_x_axis (x y : ℝ) (P_symmetrical : (ℝ × ℝ)) (hx : x = -1) (hy : y = 2) : 
  P_symmetrical = (x, -y) → P_symmetrical = (-1, -2) :=
by
  intros h
  rw [hx, hy] at h
  exact h

end symmetrical_point_wrt_x_axis_l21_21950


namespace third_month_sale_l21_21623

theorem third_month_sale
  (avg_sale : ℕ)
  (num_months : ℕ)
  (sales : List ℕ)
  (sixth_month_sale : ℕ)
  (total_sales_req : ℕ) :
  avg_sale = 6500 →
  num_months = 6 →
  sales = [6435, 6927, 7230, 6562] →
  sixth_month_sale = 4991 →
  total_sales_req = avg_sale * num_months →
  total_sales_req - (sales.sum + sixth_month_sale) = 6855 := by
  sorry

end third_month_sale_l21_21623


namespace sequence_periodic_of_period_9_l21_21297

theorem sequence_periodic_of_period_9 (a : ℕ → ℤ) (h : ∀ n, a (n + 2) = |a (n + 1)| - a n) (h_nonzero : ∃ n, a n ≠ 0) :
  ∃ m, ∃ k, m > 0 ∧ k > 0 ∧ (∀ n, a (n + m + k) = a (n + m)) ∧ k = 9 :=
by
  sorry

end sequence_periodic_of_period_9_l21_21297


namespace sets_of_three_teams_l21_21715

-- Definitions based on the conditions
def total_teams : ℕ := 20
def won_games : ℕ := 12
def lost_games : ℕ := 7

-- Main theorem to prove
theorem sets_of_three_teams : 
  (total_teams * (total_teams - 1) * (total_teams - 2)) / 6 / 2 = 570 := by
  sorry

end sets_of_three_teams_l21_21715


namespace total_cups_of_mushroom_soup_l21_21025

def cups_team_1 : ℕ := 90
def cups_team_2 : ℕ := 120
def cups_team_3 : ℕ := 70

theorem total_cups_of_mushroom_soup :
  cups_team_1 + cups_team_2 + cups_team_3 = 280 :=
  by sorry

end total_cups_of_mushroom_soup_l21_21025


namespace Brittany_age_after_vacation_l21_21817

variable (Rebecca_age Brittany_age vacation_years : ℕ)
variable (h1 : Rebecca_age = 25)
variable (h2 : Brittany_age = Rebecca_age + 3)
variable (h3 : vacation_years = 4)

theorem Brittany_age_after_vacation : 
    Brittany_age + vacation_years = 32 := 
by
  sorry

end Brittany_age_after_vacation_l21_21817


namespace arrow_in_48th_position_l21_21404

def arrow_sequence := ["→", "↔", "↓", "→", "↕"]

theorem arrow_in_48th_position :
  arrow_sequence[48 % arrow_sequence.length] = "↓" :=
by
  sorry

end arrow_in_48th_position_l21_21404


namespace backpack_prices_purchasing_plans_backpacks_given_away_l21_21380

-- Part 1: Prices of Type A and Type B backpacks
theorem backpack_prices (x y : ℝ) (h1 : x = 2 * y - 30) (h2 : 2 * x + 3 * y = 255) : x = 60 ∧ y = 45 :=
sorry

-- Part 2: Possible purchasing plans
theorem purchasing_plans (m : ℕ) (h1 : 8900 ≥ 50 * m + 40 * (200 - m)) (h2 : m > 87) : 
  m = 88 ∨ m = 89 ∨ m = 90 :=
sorry

-- Part 3: Number of backpacks given away
theorem backpacks_given_away (m n : ℕ) (total_A : ℕ := 89) (total_B : ℕ := 111) 
(h1 : m + n = 4) 
(h2 : 1250 = (total_A - if total_A > 10 then total_A / 10 else 0) * 60 + (total_B - if total_B > 10 then total_B / 10 else 0) * 45 - (50 * total_A + 40 * total_B)) :
m = 1 ∧ n = 3 := 
sorry

end backpack_prices_purchasing_plans_backpacks_given_away_l21_21380


namespace greatest_third_side_l21_21124

theorem greatest_third_side (a b : ℕ) (c : ℤ) (h₁ : a = 5) (h₂ : b = 10) (h₃ : 10 + 5 > c) (h₄ : 5 + c > 10) (h₅ : 10 + c > 5) : c = 14 :=
by sorry

end greatest_third_side_l21_21124


namespace total_cost_after_discounts_and_cashback_l21_21443

def iPhone_original_price : ℝ := 800
def iWatch_original_price : ℝ := 300
def iPhone_discount_rate : ℝ := 0.15
def iWatch_discount_rate : ℝ := 0.10
def cashback_rate : ℝ := 0.02

theorem total_cost_after_discounts_and_cashback :
  (iPhone_original_price * (1 - iPhone_discount_rate) + iWatch_original_price * (1 - iWatch_discount_rate)) * (1 - cashback_rate) = 931 :=
by sorry

end total_cost_after_discounts_and_cashback_l21_21443


namespace tablecloth_overhang_l21_21899

theorem tablecloth_overhang (d r l overhang1 overhang2 : ℝ) (h1 : d = 0.6) (h2 : r = d / 2) (h3 : l = 1) 
  (h4 : overhang1 = 0.5) (h5 : overhang2 = 0.3) :
  ∃ overhang3 overhang4 : ℝ, overhang3 = 0.33 ∧ overhang4 = 0.52 := 
sorry

end tablecloth_overhang_l21_21899


namespace trigonometric_identity_l21_21910

open Real

theorem trigonometric_identity (θ : ℝ) (h : π / 4 < θ ∧ θ < π / 2) :
  2 * cos θ + sqrt (1 - 2 * sin (π - θ) * cos θ) = sin θ + cos θ :=
sorry

end trigonometric_identity_l21_21910


namespace change_digit_correct_sum_l21_21734

theorem change_digit_correct_sum :
  ∃ d e, 
  d = 2 ∧ e = 8 ∧ 
  653479 + 938521 ≠ 1616200 ∧
  (658479 + 938581 = 1616200) ∧ 
  d + e = 10 := 
by {
  -- our proof goes here
  sorry
}

end change_digit_correct_sum_l21_21734


namespace monotonic_decreasing_interval_l21_21366

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x + 1 / x

theorem monotonic_decreasing_interval :
  ∀ x : ℝ, x > 0 → (∀ a b : ℝ, a < b → (f b ≤ f a → b ≤ (1 : ℝ) / 2)) :=
by sorry

end monotonic_decreasing_interval_l21_21366


namespace inverse_five_eq_two_l21_21250

-- Define the function f(x) = x^2 + 1 for x >= 0
def f (x : ℝ) : ℝ := x^2 + 1

-- Define the condition x >= 0
def nonneg (x : ℝ) : Prop := x ≥ 0

-- State the problem: proving that the inverse function f⁻¹(5) = 2
theorem inverse_five_eq_two : ∃ x : ℝ, nonneg x ∧ f x = 5 ∧ x = 2 :=
by
  sorry

end inverse_five_eq_two_l21_21250


namespace incorrect_conclusion_D_l21_21249

-- Define lines and planes
variables (l m n : Type) -- lines
variables (α β γ : Type) -- planes

-- Define the conditions
def intersection_planes (p1 p2 : Type) : Type := sorry
def perpendicular (a b : Type) : Prop := sorry

-- Given conditions for option D
axiom h1 : intersection_planes α β = m
axiom h2 : intersection_planes β γ = l
axiom h3 : intersection_planes γ α = n
axiom h4 : perpendicular l m
axiom h5 : perpendicular l n

-- Theorem stating that the conclusion of option D is incorrect
theorem incorrect_conclusion_D : ¬ perpendicular m n :=
by sorry

end incorrect_conclusion_D_l21_21249


namespace competition_participants_l21_21961

theorem competition_participants (N : ℕ)
  (h1 : (1 / 12) * N = 18) :
  N = 216 := 
by
  sorry

end competition_participants_l21_21961


namespace solid_triangle_front_view_l21_21021

def is_triangle_front_view (solid : ℕ) : Prop :=
  solid = 1 ∨ solid = 2 ∨ solid = 3 ∨ solid = 5

theorem solid_triangle_front_view (s : ℕ) (h : s = 1 ∨ s = 2 ∨ s = 3 ∨ s = 4 ∨ s = 5 ∨ s = 6):
  is_triangle_front_view s ↔ (s = 1 ∨ s = 2 ∨ s = 3 ∨ s = 5) :=
by
  sorry

end solid_triangle_front_view_l21_21021


namespace cubic_sum_l21_21930

theorem cubic_sum (a b c : ℤ) (h1 : a + b + c = 7) (h2 : a * b + a * c + b * c = 11) (h3 : a * b * c = -6) :
  a^3 + b^3 + c^3 = 223 :=
by
  sorry

end cubic_sum_l21_21930


namespace ants_in_third_anthill_l21_21360

-- Define the number of ants in the first anthill
def ants_first : ℕ := 100

-- Define the percentage reduction for each subsequent anthill
def percentage_reduction : ℕ := 20

-- Calculate the number of ants in the second anthill
def ants_second : ℕ := ants_first - (percentage_reduction * ants_first / 100)

-- Calculate the number of ants in the third anthill
def ants_third : ℕ := ants_second - (percentage_reduction * ants_second / 100)

-- Main theorem to prove that the number of ants in the third anthill is 64
theorem ants_in_third_anthill : ants_third = 64 := sorry

end ants_in_third_anthill_l21_21360


namespace alpha_div_3_range_l21_21605

theorem alpha_div_3_range (α : ℝ) (k : ℤ) 
  (h1 : Real.sin α > 0) 
  (h2 : Real.cos α < 0) 
  (h3 : Real.sin (α / 3) > Real.cos (α / 3)) :
  ∃ k : ℤ, (2 * k * Real.pi + Real.pi / 4 < α / 3 ∧ α / 3 < 2 * k * Real.pi + Real.pi / 3) ∨ 
            (2 * k * Real.pi + 5 * Real.pi / 6 < α / 3 ∧ α / 3 < 2 * k * Real.pi + Real.pi) :=
sorry

end alpha_div_3_range_l21_21605


namespace area_of_polygon_ABLFKJ_l21_21217

theorem area_of_polygon_ABLFKJ 
  (side_length : ℝ) (area_square : ℝ) (midpoint_l : ℝ) (area_triangle : ℝ)
  (remaining_area_each_square : ℝ) (total_area : ℝ)
  (h1 : side_length = 6)
  (h2 : area_square = side_length * side_length)
  (h3 : midpoint_l = side_length / 2)
  (h4 : area_triangle = 0.5 * side_length * midpoint_l)
  (h5 : remaining_area_each_square = area_square - 2 * area_triangle)
  (h6 : total_area = 3 * remaining_area_each_square)
  : total_area = 54 :=
by
  sorry

end area_of_polygon_ABLFKJ_l21_21217


namespace walking_speed_l21_21234

theorem walking_speed 
  (v : ℕ) -- v represents the man's walking speed in kmph
  (distance_formula : distance = speed * time)
  (distance_walking : distance = v * 9)
  (distance_running : distance = 24 * 3) : 
  v = 8 :=
by
  sorry

end walking_speed_l21_21234


namespace savings_correct_l21_21864

noncomputable def school_price_math : Float := 45
noncomputable def school_price_science : Float := 60
noncomputable def school_price_literature : Float := 35

noncomputable def discount_math : Float := 0.20
noncomputable def discount_science : Float := 0.25
noncomputable def discount_literature : Float := 0.15

noncomputable def tax_school : Float := 0.07
noncomputable def tax_alt : Float := 0.06
noncomputable def shipping_alt : Float := 10

noncomputable def alt_price_math : Float := (school_price_math * (1 - discount_math)) * (1 + tax_alt)
noncomputable def alt_price_science : Float := (school_price_science * (1 - discount_science)) * (1 + tax_alt)
noncomputable def alt_price_literature : Float := (school_price_literature * (1 - discount_literature)) * (1 + tax_alt)

noncomputable def total_alt_cost : Float := alt_price_math + alt_price_science + alt_price_literature + shipping_alt

noncomputable def school_price_math_tax : Float := school_price_math * (1 + tax_school)
noncomputable def school_price_science_tax : Float := school_price_science * (1 + tax_school)
noncomputable def school_price_literature_tax : Float := school_price_literature * (1 + tax_school)

noncomputable def total_school_cost : Float := school_price_math_tax + school_price_science_tax + school_price_literature_tax

noncomputable def savings : Float := total_school_cost - total_alt_cost

theorem savings_correct : savings = 22.40 := by
  sorry

end savings_correct_l21_21864


namespace functions_satisfying_equation_are_constants_l21_21166

theorem functions_satisfying_equation_are_constants (f g : ℝ → ℝ) :
  (∀ x y : ℝ, f (f (x + y)) = x * f y + g x) → ∃ k : ℝ, (∀ x : ℝ, f x = k) ∧ (∀ x : ℝ, g x = k * (1 - x)) :=
by
  sorry

end functions_satisfying_equation_are_constants_l21_21166


namespace car_speed_second_hour_l21_21678

theorem car_speed_second_hour (s1 s2 : ℝ) (h1 : s1 = 10) (h2 : (s1 + s2) / 2 = 35) : s2 = 60 := by
  sorry

end car_speed_second_hour_l21_21678


namespace sum_of_consecutive_numbers_with_lcm_168_l21_21934

theorem sum_of_consecutive_numbers_with_lcm_168 (a b c : ℕ) (h1 : b = a + 1) (h2 : c = b + 1) (h3 : Nat.lcm a (Nat.lcm b c) = 168) : a + b + c = 21 :=
sorry

end sum_of_consecutive_numbers_with_lcm_168_l21_21934


namespace range_of_a_l21_21361

def f (x : ℝ) : ℝ := x^2 - 2 * x

theorem range_of_a :
  (∃ (a : ℝ), (a ≤ -2 ∨ a ≥ 0) ∧ (∃ (x : ℝ) (hx : 2 ≤ x ∧ x ≤ 4), f x ≤ a^2 + 2 * a)) :=
by sorry

end range_of_a_l21_21361


namespace odd_function_property_l21_21495

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (4 - x^2)) / x

theorem odd_function_property (a : ℝ) (h_a : -2 ≤ a ∧ a ≤ 2) (h_fa : f a = -4) : f (-a) = 4 :=
by
  sorry

end odd_function_property_l21_21495


namespace molecular_weight_of_3_moles_of_Fe2_SO4_3_l21_21591

noncomputable def mol_weight_fe : ℝ := 55.845
noncomputable def mol_weight_s : ℝ := 32.065
noncomputable def mol_weight_o : ℝ := 15.999

noncomputable def mol_weight_fe2_so4_3 : ℝ :=
  (2 * mol_weight_fe) + (3 * (mol_weight_s + (4 * mol_weight_o)))

theorem molecular_weight_of_3_moles_of_Fe2_SO4_3 :
  3 * mol_weight_fe2_so4_3 = 1199.619 := by
  sorry

end molecular_weight_of_3_moles_of_Fe2_SO4_3_l21_21591


namespace problem_solution_l21_21266

variables (a : ℝ)

def p : Prop := ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0
def q : Prop := ∃ x_0 : ℝ, x_0^2 + (a-1)*x_0 + 1 < 0

theorem problem_solution (h₁ : p a ∨ q a) (h₂ : ¬(p a ∧ q a)) :
  -1 ≤ a ∧ a ≤ 1 ∨ a > 3 :=
sorry

end problem_solution_l21_21266


namespace total_laces_needed_l21_21806

variable (x : ℕ) -- Eva has x pairs of shoes
def long_laces_per_pair : ℕ := 3
def short_laces_per_pair : ℕ := 3
def laces_per_pair : ℕ := long_laces_per_pair + short_laces_per_pair

theorem total_laces_needed : 6 * x = 6 * x :=
by
  have h : laces_per_pair = 6 := rfl
  sorry

end total_laces_needed_l21_21806


namespace consecutive_product_solution_l21_21764

theorem consecutive_product_solution :
  ∀ (n : ℤ), (∃ a : ℤ, n^4 + 8 * n + 11 = a * (a + 1)) ↔ n = 1 :=
by
  sorry

end consecutive_product_solution_l21_21764


namespace number_of_grade11_students_l21_21155

-- Define the total number of students in the high school.
def total_students : ℕ := 900

-- Define the total number of students selected in the sample.
def sample_students : ℕ := 45

-- Define the number of Grade 10 students in the sample.
def grade10_students_sample : ℕ := 20

-- Define the number of Grade 12 students in the sample.
def grade12_students_sample : ℕ := 10

-- Prove the number of Grade 11 students in the school is 300.
theorem number_of_grade11_students :
  (sample_students - grade10_students_sample - grade12_students_sample) * (total_students / sample_students) = 300 :=
by
  sorry

end number_of_grade11_students_l21_21155


namespace carlos_goals_product_l21_21173

theorem carlos_goals_product :
  ∃ (g11 g12 : ℕ), g11 < 8 ∧ g12 < 8 ∧ 
  (33 + g11) % 11 = 0 ∧ 
  (33 + g11 + g12) % 12 = 0 ∧ 
  g11 * g12 = 49 := 
by
  sorry

end carlos_goals_product_l21_21173


namespace twins_ages_sum_equals_20_l21_21703

def sum_of_ages (A K : ℕ) := 2 * A + K

theorem twins_ages_sum_equals_20 (A K : ℕ) (h1 : A = A) (h2 : A * A * K = 256) : 
  sum_of_ages A K = 20 :=
by
  sorry

end twins_ages_sum_equals_20_l21_21703


namespace both_firms_participate_social_optimality_l21_21150

variables (α V IC : ℝ)

-- Conditions definitions
def expected_income_if_both_participate (α V : ℝ) : ℝ :=
  α * (1 - α) * V + 0.5 * (α^2) * V

def condition_for_both_participation (α V IC : ℝ) : Prop :=
  expected_income_if_both_participate α V - IC ≥ 0

-- Values for specific case
noncomputable def V_specific : ℝ := 24
noncomputable def α_specific : ℝ := 0.5
noncomputable def IC_specific : ℝ := 7

-- Proof problem statement
theorem both_firms_participate : condition_for_both_participation α_specific V_specific IC_specific := by
  sorry

-- Definitions for social welfare considerations
def total_profit_if_both_participate (α V IC : ℝ) : ℝ :=
  2 * (expected_income_if_both_participate α V - IC)

def expected_income_if_one_participates (α V IC : ℝ) : ℝ :=
  α * V - IC

def social_optimal (α V IC : ℝ) : Prop :=
  total_profit_if_both_participate α V IC < expected_income_if_one_participates α V IC

theorem social_optimality : social_optimal α_specific V_specific IC_specific := by
  sorry

end both_firms_participate_social_optimality_l21_21150


namespace total_fruits_in_baskets_l21_21805

def total_fruits (apples1 oranges1 bananas1 apples2 oranges2 bananas2 : ℕ) :=
  apples1 + oranges1 + bananas1 + apples2 + oranges2 + bananas2

theorem total_fruits_in_baskets :
  total_fruits 9 15 14 (9 - 2) (15 - 2) (14 - 2) = 70 :=
by
  sorry

end total_fruits_in_baskets_l21_21805


namespace min_inverse_ab_l21_21653

theorem min_inverse_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2 * b = 6) : 
  ∃ (m : ℝ), (m = 2 / 9) ∧ (∀ (a b : ℝ), a > 0 → b > 0 → a + 2 * b = 6 → 1/(a * b) ≥ m) :=
by
  sorry

end min_inverse_ab_l21_21653


namespace relationship_between_a_b_c_l21_21489

noncomputable def a := (3 / 5 : ℝ) ^ (2 / 5)
noncomputable def b := (2 / 5 : ℝ) ^ (3 / 5)
noncomputable def c := (2 / 5 : ℝ) ^ (2 / 5)

theorem relationship_between_a_b_c :
  a > c ∧ c > b :=
by
  sorry

end relationship_between_a_b_c_l21_21489


namespace parameter_range_exists_solution_l21_21416

theorem parameter_range_exists_solution :
  {a : ℝ | ∃ b : ℝ, ∃ x y : ℝ,
    x^2 + y^2 + 2 * a * (a + y - x) = 49 ∧
    y = 15 * Real.cos (x - b) - 8 * Real.sin (x - b)
  } = {a : ℝ | -24 ≤ a ∧ a ≤ 24} :=
sorry

end parameter_range_exists_solution_l21_21416


namespace smallest_area_is_10_l21_21998

noncomputable def smallest_square_area : ℝ :=
  let k₁ := 65
  let k₂ := -5
  10 * (9 + 4 * k₂)

theorem smallest_area_is_10 :
  smallest_square_area = 10 := by
  sorry

end smallest_area_is_10_l21_21998


namespace power_division_calculation_l21_21533

theorem power_division_calculation :
  ( ( 5^13 / 5^11 )^2 * 5^2 ) / 2^5 = 15625 / 32 :=
by
  sorry

end power_division_calculation_l21_21533


namespace compute_f_of_1_plus_g_of_3_l21_21461

def f (x : ℝ) : ℝ := 3 * x - 4
def g (x : ℝ) : ℝ := x^2 + 1

theorem compute_f_of_1_plus_g_of_3 : f (1 + g 3) = 29 := by 
  sorry

end compute_f_of_1_plus_g_of_3_l21_21461


namespace pond_fish_count_l21_21542

theorem pond_fish_count :
  (∃ (N : ℕ), (2 / 50 : ℚ) = (40 / N : ℚ)) → N = 1000 :=
by
  sorry

end pond_fish_count_l21_21542


namespace find_sum_of_angles_l21_21076

open Real

namespace math_problem

theorem find_sum_of_angles (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h1 : cos (α - β / 2) = sqrt 3 / 2)
  (h2 : sin (α / 2 - β) = -1 / 2) : α + β = 2 * π / 3 :=
sorry

end math_problem

end find_sum_of_angles_l21_21076


namespace find_c_value_l21_21685

theorem find_c_value 
  (a b c : ℝ)
  (h_a : a = 5 / 2)
  (h_b : b = 17)
  (roots : ∀ x : ℝ, x = (-b + Real.sqrt 23) / 5 ∨ x = (-b - Real.sqrt 23) / 5)
  (discrim_eq : ∀ c : ℝ, b ^ 2 - 4 * a * c = 23) :
  c = 26.6 := by
  sorry

end find_c_value_l21_21685


namespace smallest_is_57_l21_21458

noncomputable def smallest_of_four_numbers (a b c d : ℕ) : ℕ :=
  if h1 : a + b + c = 234 ∧ a + b + d = 251 ∧ a + c + d = 284 ∧ b + c + d = 299
  then Nat.min (Nat.min a b) (Nat.min c d)
  else 0

theorem smallest_is_57 (a b c d : ℕ) (h1 : a + b + c = 234) (h2 : a + b + d = 251)
  (h3 : a + c + d = 284) (h4 : b + c + d = 299) :
  smallest_of_four_numbers a b c d = 57 :=
sorry

end smallest_is_57_l21_21458


namespace triangle_inequality_sides_l21_21971

theorem triangle_inequality_sides
  (a b c : ℝ)
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a + b > c) (h5 : a + c > b) (h6 : b + c > a) :
  (a + b) * Real.sqrt (a * b) + (a + c) * Real.sqrt (a * c) + (b + c) * Real.sqrt (b * c) ≥ (a + b + c)^2 / 2 := 
by
  sorry

end triangle_inequality_sides_l21_21971


namespace volume_rect_prism_l21_21659

variables (a d h : ℝ)
variables (ha : a > 0) (hd : d > 0) (hh : h > 0)

theorem volume_rect_prism : a * d * h = adh :=
by
  sorry

end volume_rect_prism_l21_21659


namespace new_circle_radius_shaded_region_l21_21038

theorem new_circle_radius_shaded_region {r1 r2 : ℝ} 
    (h1 : r1 = 35) 
    (h2 : r2 = 24) : 
    ∃ r : ℝ, π * r^2 = π * (r1^2 - r2^2) ∧ r = Real.sqrt 649 := 
by
  sorry

end new_circle_radius_shaded_region_l21_21038


namespace pieces_of_meat_per_slice_eq_22_l21_21215

def number_of_pepperoni : Nat := 30
def number_of_ham : Nat := 2 * number_of_pepperoni
def number_of_sausage : Nat := number_of_pepperoni + 12
def total_meat : Nat := number_of_pepperoni + number_of_ham + number_of_sausage
def number_of_slices : Nat := 6

theorem pieces_of_meat_per_slice_eq_22 : total_meat / number_of_slices = 22 :=
by
  sorry

end pieces_of_meat_per_slice_eq_22_l21_21215


namespace total_packs_of_groceries_l21_21687

-- Definitions based on conditions
def packs_of_cookies : Nat := 4
def packs_of_cake : Nat := 22
def packs_of_chocolate : Nat := 16

-- The proof statement
theorem total_packs_of_groceries : packs_of_cookies + packs_of_cake + packs_of_chocolate = 42 :=
by
  -- Proof skipped using sorry
  sorry

end total_packs_of_groceries_l21_21687


namespace desert_area_2020_correct_desert_area_less_8_10_5_by_2023_l21_21063

-- Define initial desert area
def initial_desert_area : ℝ := 9 * 10^5

-- Define increase in desert area each year as observed
def yearly_increase (n : ℕ) : ℝ :=
  match n with
  | 1998 => 2000
  | 1999 => 4000
  | 2000 => 6001
  | 2001 => 7999
  | 2002 => 10001
  | _    => 0

-- Define arithmetic progression of increases
def common_difference : ℝ := 2000

-- Define desert area in 2020
def desert_area_2020 : ℝ :=
  initial_desert_area + 10001 + 18 * common_difference

-- Statement: Desert area by the end of 2020 is approximately 9.46 * 10^5 hm^2
theorem desert_area_2020_correct :
  desert_area_2020 = 9.46 * 10^5 :=
sorry

-- Define yearly transformation and desert increment with afforestation from 2003
def desert_area_with_afforestation (n : ℕ) : ℝ :=
  if n < 2003 then
    initial_desert_area + yearly_increase n
  else
    initial_desert_area + 10001 + (n - 2002) * (common_difference - 8000)

-- Statement: Desert area will be less than 8 * 10^5 hm^2 by the end of 2023
theorem desert_area_less_8_10_5_by_2023 :
  desert_area_with_afforestation 2023 < 8 * 10^5 :=
sorry

end desert_area_2020_correct_desert_area_less_8_10_5_by_2023_l21_21063


namespace peanut_raising_ratio_l21_21800

theorem peanut_raising_ratio
  (initial_peanuts : ℝ)
  (remove_peanuts_1 : ℝ)
  (add_raisins_1 : ℝ)
  (remove_mixture : ℝ)
  (add_raisins_2 : ℝ)
  (final_peanuts : ℝ)
  (final_raisins : ℝ)
  (ratio : ℝ) :
  initial_peanuts = 10 ∧
  remove_peanuts_1 = 2 ∧
  add_raisins_1 = 2 ∧
  remove_mixture = 2 ∧
  add_raisins_2 = 2 ∧
  final_peanuts = initial_peanuts - remove_peanuts_1 - (remove_mixture * (initial_peanuts - remove_peanuts_1) / (initial_peanuts - remove_peanuts_1 + add_raisins_1)) ∧
  final_raisins = add_raisins_1 - (remove_mixture * add_raisins_1 / (initial_peanuts - remove_peanuts_1 + add_raisins_1)) + add_raisins_2 ∧
  ratio = final_peanuts / final_raisins →
  ratio = 16 / 9 := by
  sorry

end peanut_raising_ratio_l21_21800


namespace field_length_l21_21634

theorem field_length 
  (w l : ℝ)
  (pond_area : ℝ := 25)
  (h1 : l = 2 * w)
  (h2 : pond_area = 25)
  (h3 : pond_area = (1 / 8) * (l * w)) :
  l = 20 :=
by
  sorry

end field_length_l21_21634


namespace square_root_of_25_squared_l21_21486

theorem square_root_of_25_squared :
  Real.sqrt (25 ^ 2) = 25 :=
sorry

end square_root_of_25_squared_l21_21486


namespace labor_cost_per_hour_l21_21831

theorem labor_cost_per_hour (total_repair_cost part_cost labor_hours : ℕ)
    (h1 : total_repair_cost = 2400)
    (h2 : part_cost = 1200)
    (h3 : labor_hours = 16) :
    (total_repair_cost - part_cost) / labor_hours = 75 := by
  sorry

end labor_cost_per_hour_l21_21831


namespace line_through_P_midpoint_l21_21537

noncomputable section

open Classical

variables (l l1 l2 : ℝ → ℝ → Prop) (P A B : ℝ × ℝ)

def line1 (x y : ℝ) := 2 * x - y - 2 = 0
def line2 (x y : ℝ) := x + y + 3 = 0

theorem line_through_P_midpoint (P A B : ℝ × ℝ)
  (hP : P = (3, 0))
  (hl1 : ∀ x y, line1 x y → l x y)
  (hl2 : ∀ x y, line2 x y → l x y)
  (hmid : (P.1 = (A.1 + B.1) / 2) ∧ (P.2 = (A.2 + B.2) / 2)) :
  ∃ k : ℝ, ∀ x y, (y = k * (x - 3)) ↔ (8 * x - y - 24 = 0) :=
by
  sorry

end line_through_P_midpoint_l21_21537


namespace red_pigment_weight_in_brown_paint_l21_21227

theorem red_pigment_weight_in_brown_paint :
  ∀ (M G : ℝ), 
    (M + G = 10) → 
    (0.5 * M + 0.3 * G = 4) →
    0.5 * M = 2.5 :=
by sorry

end red_pigment_weight_in_brown_paint_l21_21227


namespace money_per_percentage_point_l21_21791

theorem money_per_percentage_point
  (plates : ℕ) (total_states : ℕ) (total_amount : ℤ)
  (h_plates : plates = 40) (h_total_states : total_states = 50) (h_total_amount : total_amount = 160) :
  total_amount / (plates * 100 / total_states) = 2 :=
by
  -- Omitted steps of the proof
  sorry

end money_per_percentage_point_l21_21791


namespace sets_difference_M_star_N_l21_21771

def M (y : ℝ) : Prop := y ≤ 2

def N (y : ℝ) : Prop := 0 ≤ y ∧ y ≤ 3

def M_star_N (y : ℝ) : Prop := y < 0

theorem sets_difference_M_star_N : {y : ℝ | M y ∧ ¬ N y} = {y : ℝ | M_star_N y} :=
by {
  sorry
}

end sets_difference_M_star_N_l21_21771


namespace simplify_product_l21_21767

theorem simplify_product (b : ℤ) : (1 * 2 * b * 3 * b^2 * 4 * b^3 * 5 * b^4 * 6 * b^5) = 720 * b^15 := by
  sorry

end simplify_product_l21_21767


namespace n_fraction_of_sum_l21_21887

theorem n_fraction_of_sum (n S : ℝ) (h1 : n = S / 5) (h2 : S ≠ 0) :
  n = 1 / 6 * ((S + (S / 5))) :=
by
  sorry

end n_fraction_of_sum_l21_21887


namespace james_louise_age_sum_l21_21729

variables (J L : ℝ)

theorem james_louise_age_sum
  (h₁ : J = L + 9)
  (h₂ : J + 5 = 3 * (L - 3)) :
  J + L = 32 :=
by
  /- Proof goes here -/
  sorry

end james_louise_age_sum_l21_21729


namespace sahil_selling_price_l21_21236

def initial_cost : ℝ := 14000
def repair_cost : ℝ := 5000
def transportation_charges : ℝ := 1000
def profit_percent : ℝ := 50

noncomputable def total_cost : ℝ := initial_cost + repair_cost + transportation_charges
noncomputable def profit : ℝ := profit_percent / 100 * total_cost
noncomputable def selling_price : ℝ := total_cost + profit

theorem sahil_selling_price :
  selling_price = 30000 := by
  sorry

end sahil_selling_price_l21_21236


namespace frustum_surface_area_l21_21390

theorem frustum_surface_area (r r' l : ℝ) (h_r : r = 1) (h_r' : r' = 4) (h_l : l = 5) :
  π * r^2 + π * r'^2 + π * (r + r') * l = 42 * π :=
by
  rw [h_r, h_r', h_l]
  norm_num
  sorry

end frustum_surface_area_l21_21390


namespace distance_between_intersections_l21_21334

theorem distance_between_intersections (a : ℝ) (a_pos : 0 < a) : 
  |(Real.log a / Real.log 2) - (Real.log (a / 3) / Real.log 2)| = Real.log 3 / Real.log 2 :=
by
  sorry

end distance_between_intersections_l21_21334


namespace estimate_expr_l21_21315

theorem estimate_expr : 1 < (3 * Real.sqrt 2 - Real.sqrt 12) * Real.sqrt 3 ∧ (3 * Real.sqrt 2 - Real.sqrt 12) * Real.sqrt 3 < 2 := by
  sorry

end estimate_expr_l21_21315


namespace vertex_of_parabola_on_x_axis_l21_21972

theorem vertex_of_parabola_on_x_axis (c : ℝ) : 
  (∃ x : ℝ, (x^2 - 6*x + c = 0)) ↔ c = 9 :=
by
  sorry

end vertex_of_parabola_on_x_axis_l21_21972


namespace continuity_at_x0_l21_21707

noncomputable def f (x : ℝ) : ℝ := -4 * x^2 - 7

theorem continuity_at_x0 :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - 1| < δ → |f x - f 1| < ε :=
by
  sorry

end continuity_at_x0_l21_21707


namespace infinitely_many_n_l21_21655

-- Definition capturing the condition: equation \( (x + y + z)^3 = n^2 xyz \)
def equation (x y z n : ℕ) : Prop := (x + y + z)^3 = n^2 * x * y * z

-- The main statement: proving the existence of infinitely many positive integers n such that the equation has a solution
theorem infinitely_many_n :
  ∃ᶠ n : ℕ in at_top, ∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ equation x y z n :=
sorry

end infinitely_many_n_l21_21655


namespace right_triangle_area_l21_21567

theorem right_triangle_area (a : ℝ) (h : a > 2)
  (h_arith_seq : a - 2 > 0)
  (pythagorean : (a - 2)^2 + a^2 = (a + 2)^2) :
  (1 / 2) * (a - 2) * a = 24 :=
by
  sorry

end right_triangle_area_l21_21567


namespace ratio_area_A_to_C_l21_21851

noncomputable def side_length (perimeter : ℕ) : ℕ :=
  perimeter / 4

noncomputable def area (side : ℕ) : ℕ :=
  side * side

theorem ratio_area_A_to_C : 
  let A_perimeter := 16
  let B_perimeter := 40
  let C_perimeter := 2 * A_perimeter
  let side_A := side_length A_perimeter
  let side_C := side_length C_perimeter
  let area_A := area side_A
  let area_C := area side_C
  (area_A : ℚ) / area_C = 1 / 4 :=
by
  sorry

end ratio_area_A_to_C_l21_21851


namespace divide_equally_l21_21478

-- Define the input values based on the conditions.
def brother_strawberries := 3 * 15
def kimberly_strawberries := 8 * brother_strawberries
def parents_strawberries := kimberly_strawberries - 93
def total_strawberries := brother_strawberries + kimberly_strawberries + parents_strawberries
def family_members := 4

-- Define the theorem to prove the question.
theorem divide_equally : 
    (total_strawberries / family_members) = 168 :=
by
    -- (proof goes here)
    sorry

end divide_equally_l21_21478


namespace x_intercept_of_line_l21_21796

theorem x_intercept_of_line (x y : ℝ) : (4 * x + 7 * y = 28) ∧ (y = 0) → x = 7 :=
by
  sorry

end x_intercept_of_line_l21_21796


namespace exchange_rate_lire_l21_21946

theorem exchange_rate_lire (x : ℕ) (h : 2500 / 2 = x / 5) : x = 6250 :=
by
  sorry

end exchange_rate_lire_l21_21946


namespace jovana_total_shells_l21_21242

def initial_amount : ℕ := 5
def added_amount : ℕ := 23
def total_amount : ℕ := 28

theorem jovana_total_shells : initial_amount + added_amount = total_amount := by
  sorry

end jovana_total_shells_l21_21242


namespace quadratic_eq_has_two_distinct_real_roots_l21_21710

theorem quadratic_eq_has_two_distinct_real_roots (m : ℝ) : 
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (∀ x : ℝ, x^2 - 2*m*x - m - 1 = 0 ↔ x = x1 ∨ x = x2) :=
by
  sorry

end quadratic_eq_has_two_distinct_real_roots_l21_21710


namespace smallest_number_divisible_l21_21131

theorem smallest_number_divisible (n : ℕ) :
  (∀ d ∈ [4, 6, 8, 10, 12, 14, 16], (n - 16) % d = 0) ↔ n = 3376 :=
by {
  sorry
}

end smallest_number_divisible_l21_21131


namespace max_value_of_a_plus_b_l21_21620

theorem max_value_of_a_plus_b (a b : ℝ) (h₁ : a^2 + b^2 = 25) (h₂ : a ≤ 3) (h₃ : b ≥ 3) :
  a + b ≤ 7 :=
sorry

end max_value_of_a_plus_b_l21_21620


namespace three_x4_plus_two_x5_l21_21343

theorem three_x4_plus_two_x5 (x1 x2 x3 x4 x5 : ℤ)
  (h1 : 2 * x1 + x2 + x3 + x4 + x5 = 6)
  (h2 : x1 + 2 * x2 + x3 + x4 + x5 = 12)
  (h3 : x1 + x2 + 2 * x3 + x4 + x5 = 24)
  (h4 : x1 + x2 + x3 + 2 * x4 + x5 = 48)
  (h5 : x1 + x2 + x3 + x4 + 2 * x5 = 96) : 
  3 * x4 + 2 * x5 = 181 := 
sorry

end three_x4_plus_two_x5_l21_21343


namespace efficiency_ratio_l21_21010

variable {A B : ℝ}

theorem efficiency_ratio (hA : A = 1 / 30) (hAB : A + B = 1 / 20) : A / B = 2 :=
by
  sorry

end efficiency_ratio_l21_21010


namespace prob_part1_prob_part2_l21_21700

-- Define the probability that Person A hits the target
def pA : ℚ := 2 / 3

-- Define the probability that Person B hits the target
def pB : ℚ := 3 / 4

-- Define the number of shots
def nShotsA : ℕ := 3
def nShotsB : ℕ := 2

-- The problem posed to Person A
def probA_miss_at_least_once : ℚ := 1 - (pA ^ nShotsA)

-- The problem posed to Person A (exactly twice in 2 shots)
def probA_hits_exactly_twice : ℚ := pA ^ 2

-- The problem posed to Person B (exactly once in 2 shots)
def probB_hits_exactly_once : ℚ :=
  2 * (pB * (1 - pB))

-- The combined probability for Part 2
def combined_prob : ℚ := probA_hits_exactly_twice * probB_hits_exactly_once

theorem prob_part1 :
  probA_miss_at_least_once = 19 / 27 := by
  sorry

theorem prob_part2 :
  combined_prob = 1 / 6 := by
  sorry

end prob_part1_prob_part2_l21_21700


namespace female_students_count_l21_21731

theorem female_students_count 
  (total_average : ℝ)
  (male_count : ℕ)
  (male_average : ℝ)
  (female_average : ℝ)
  (female_count : ℕ) 
  (correct_female_count : female_count = 12)
  (h1 : total_average = 90)
  (h2 : male_count = 8)
  (h3 : male_average = 87)
  (h4 : female_average = 92) :
  total_average * (male_count + female_count) = male_count * male_average + female_count * female_average :=
by sorry

end female_students_count_l21_21731


namespace opposite_of_2023_l21_21290

theorem opposite_of_2023 : - 2023 = (-2023) := by
  sorry

end opposite_of_2023_l21_21290


namespace find_point_B_l21_21651

theorem find_point_B (A B : ℝ × ℝ) (a : ℝ × ℝ)
  (hA : A = (-1, -5)) 
  (ha : a = (2, 3)) 
  (hAB : B - A = 3 • a) : 
  B = (5, 4) := sorry

end find_point_B_l21_21651


namespace time_in_vancouver_l21_21055

theorem time_in_vancouver (toronto_time vancouver_time : ℕ) (h : toronto_time = 18 + 30 / 60) (h_diff : vancouver_time = toronto_time - 3) :
  vancouver_time = 15 + 30 / 60 :=
by
  sorry

end time_in_vancouver_l21_21055


namespace area_of_quadrilateral_ABDF_l21_21792

theorem area_of_quadrilateral_ABDF :
  let length := 40
  let width := 30
  let rectangle_area := length * width
  let B := (1/4 : ℝ) * length
  let F := (1/2 : ℝ) * width
  let area_BCD := (1/2 : ℝ) * (3/4 : ℝ) * length * width
  let area_EFD := (1/2 : ℝ) * F * length
  rectangle_area - area_BCD - area_EFD = 450 := sorry

end area_of_quadrilateral_ABDF_l21_21792


namespace Elaine_rent_percentage_l21_21625

variable (E : ℝ) (last_year_rent : ℝ) (this_year_rent : ℝ)

def Elaine_last_year_earnings (E : ℝ) : ℝ := E

def Elaine_last_year_rent (E : ℝ) : ℝ := 0.20 * E

def Elaine_this_year_earnings (E : ℝ) : ℝ := 1.25 * E

def Elaine_this_year_rent (E : ℝ) : ℝ := 0.30 * (1.25 * E)

theorem Elaine_rent_percentage 
  (E : ℝ) 
  (last_year_rent := Elaine_last_year_rent E)
  (this_year_rent := Elaine_this_year_rent E) :
  (this_year_rent / last_year_rent) * 100 = 187.5 := 
by sorry

end Elaine_rent_percentage_l21_21625


namespace max_integer_values_correct_l21_21598

noncomputable def max_integer_values (a b c : ℝ) : ℕ :=
  if a > 100 then 2 else 0

theorem max_integer_values_correct (a b c : ℝ) (h : a > 100) :
  max_integer_values a b c = 2 :=
by sorry

end max_integer_values_correct_l21_21598


namespace sufficient_but_not_necessary_l21_21611

def vectors_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

theorem sufficient_but_not_necessary (m n : ℝ) :
  vectors_parallel (m, 1) (n, 1) ↔ (m = n) := sorry

end sufficient_but_not_necessary_l21_21611


namespace decreasing_f_range_l21_21186

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then a^x else (a - 3) * x + 4 * a

theorem decreasing_f_range (a : ℝ) :
  (∀ x1 x2 : ℝ, x1 ≠ x2 → (f a x1 - f a x2) / (x1 - x2) < 0) → (0 < a ∧ a ≤ 1/4) :=
by
  sorry

end decreasing_f_range_l21_21186


namespace total_cats_and_kittens_received_l21_21052

theorem total_cats_and_kittens_received (total_adult_cats : ℕ) (percentage_female : ℕ) (fraction_with_kittens : ℚ) (kittens_per_litter : ℕ) 
  (h1 : total_adult_cats = 100) (h2 : percentage_female = 40) (h3 : fraction_with_kittens = 2 / 3) (h4 : kittens_per_litter = 3) :
  total_adult_cats + ((percentage_female * total_adult_cats / 100) * (fraction_with_kittens * total_adult_cats * kittens_per_litter) / 100) = 181 := by
  sorry

end total_cats_and_kittens_received_l21_21052


namespace determine_d_l21_21394

variables (u v : ℝ × ℝ × ℝ) -- defining u and v as 3D vectors

noncomputable def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (a.2.2 * b.2.1 - a.2.1 * b.2.2, a.1 * b.2.2 - a.2.2 * b.1 , a.2.1 * b.1 - a.1 * b.2.1)

noncomputable def dot_product (a b : ℝ × ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2.1 * b.2.1 + a.2.2 * b.2.2

noncomputable def i : ℝ × ℝ × ℝ := (1, 0, 0)
noncomputable def j : ℝ × ℝ × ℝ := (0, 1, 0)
noncomputable def k : ℝ × ℝ × ℝ := (0, 0, 1)

theorem determine_d (u : ℝ × ℝ × ℝ) (v : ℝ × ℝ × ℝ) :
  cross_product i (cross_product (u + v) i) +
  cross_product j (cross_product (u + v) j) +
  cross_product k (cross_product (u + v) k) =
  2 * (u + v) :=
sorry

end determine_d_l21_21394


namespace max_integer_value_of_expression_l21_21848

theorem max_integer_value_of_expression (x : ℝ) :
  ∃ M : ℤ, M = 15 ∧ ∀ y : ℝ, (4 * y^2 + 8 * y + 19) / (4 * y^2 + 8 * y + 5) ≤ M :=
sorry

end max_integer_value_of_expression_l21_21848


namespace max_possible_N_in_cities_l21_21447

theorem max_possible_N_in_cities (N : ℕ) (num_cities : ℕ) (roads : ℕ → List ℕ) :
  (num_cities = 110) →
  (∀ n, 1 ≤ n ∧ n ≤ N → List.length (roads n) = n) →
  N ≤ 107 :=
by
  sorry

end max_possible_N_in_cities_l21_21447


namespace condition_sufficiency_l21_21908

theorem condition_sufficiency (x : ℝ) :
  (2 ≤ x ∧ x ≤ 3) → (x < -3 ∨ x ≥ 1) ∧ (∃ x : ℝ, (x < -3 ∨ x ≥ 1) ∧ ¬(2 ≤ x ∧ x ≤ 3)) :=
by
  sorry

end condition_sufficiency_l21_21908


namespace average_of_remaining_two_l21_21121

theorem average_of_remaining_two (a1 a2 a3 a4 a5 : ℚ)
  (h1 : (a1 + a2 + a3 + a4 + a5) / 5 = 11)
  (h2 : (a1 + a2 + a3) / 3 = 4) :
  ((a4 + a5) / 2 = 21.5) :=
sorry

end average_of_remaining_two_l21_21121


namespace hyperbola_asymptotes_l21_21319

theorem hyperbola_asymptotes
    (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 2 = Real.sqrt (1 + (b^2) / (a^2))) :
    (∀ x y : ℝ, (y = x * Real.sqrt 3) ∨ (y = -x * Real.sqrt 3)) :=
by
  sorry

end hyperbola_asymptotes_l21_21319


namespace kevin_ends_with_604_cards_l21_21260

theorem kevin_ends_with_604_cards : 
  ∀ (initial_cards found_cards : ℕ), initial_cards = 65 → found_cards = 539 → initial_cards + found_cards = 604 :=
by
  intros initial_cards found_cards h_initial h_found
  sorry

end kevin_ends_with_604_cards_l21_21260


namespace gain_in_transaction_per_year_l21_21406

noncomputable def borrowing_interest (principal : ℕ) (rate : ℚ) (time : ℕ) : ℚ :=
  principal * rate * time

noncomputable def lending_interest (principal : ℕ) (rate : ℚ) (time : ℕ) : ℚ :=
  principal * rate * time

noncomputable def gain_per_year (borrow_principal : ℕ) (borrow_rate : ℚ) 
  (borrow_time : ℕ) (lend_principal : ℕ) (lend_rate : ℚ) (lend_time : ℕ) : ℚ :=
  (lending_interest lend_principal lend_rate lend_time - borrowing_interest borrow_principal borrow_rate borrow_time) / borrow_time

theorem gain_in_transaction_per_year :
  gain_per_year 4000 (4 / 100) 2 4000 (6 / 100) 2 = 80 := 
sorry

end gain_in_transaction_per_year_l21_21406


namespace john_marbles_l21_21048

theorem john_marbles : ∃ m : ℕ, (m ≡ 3 [MOD 7]) ∧ (m ≡ 2 [MOD 4]) ∧ m = 10 := by
  sorry

end john_marbles_l21_21048


namespace alarm_clock_shows_noon_in_14_minutes_l21_21888

-- Definitions based on given problem conditions
def clockRunsSlow (clock_time real_time : ℕ) : Prop :=
  clock_time = real_time * 56 / 60

def timeSinceSet : ℕ := 210 -- 3.5 hours in minutes
def correctClockShowsNoon : ℕ := 720 -- Noon in minutes (12*60)

-- Main statement to prove
theorem alarm_clock_shows_noon_in_14_minutes :
  ∃ minutes : ℕ, clockRunsSlow (timeSinceSet * 56 / 60) timeSinceSet ∧ correctClockShowsNoon - (480 + timeSinceSet * 56 / 60) = minutes ∧ minutes = 14 := 
by
  sorry

end alarm_clock_shows_noon_in_14_minutes_l21_21888


namespace find_k_l21_21900

theorem find_k (x y k : ℝ) (h1 : x + y = 5 * k) (h2 : x - 2 * y = -k) (h3 : 2 * x - y = 8) : k = 2 :=
by
  sorry

end find_k_l21_21900


namespace solution_set_of_bx2_ax_c_lt_zero_l21_21294

theorem solution_set_of_bx2_ax_c_lt_zero (a b c : ℝ) (h1 : a > 0) (h2 : b = a) (h3 : c = -6 * a) (h4 : ∀ x, ax^2 - bx + c < 0 ↔ -2 < x ∧ x < 3) :
  ∀ x, bx^2 + ax + c < 0 ↔ -3 < x ∧ x < 2 :=
by
  sorry

end solution_set_of_bx2_ax_c_lt_zero_l21_21294


namespace solution_is_thirteen_over_nine_l21_21920

noncomputable def check_solution (x : ℝ) : Prop :=
  (3 * x^2 / (x - 2) - (3 * x + 9) / 4 + (6 - 9 * x) / (x - 2) + 2 = 0) ∧
  (x^3 ≠ 3 * x + 1)

theorem solution_is_thirteen_over_nine :
  check_solution (13 / 9) :=
by
  sorry

end solution_is_thirteen_over_nine_l21_21920


namespace base15_mod_9_l21_21370

noncomputable def base15_to_decimal : ℕ :=
  2 * 15^3 + 6 * 15^2 + 4 * 15^1 + 3 * 15^0

theorem base15_mod_9 (n : ℕ) (h : n = base15_to_decimal) : n % 9 = 0 :=
sorry

end base15_mod_9_l21_21370


namespace quadratic_inequality_solution_l21_21926

theorem quadratic_inequality_solution
  (a : ℝ) :
  (∀ x : ℝ, (a-2)*x^2 + 2*(a-2)*x - 4 ≤ 0) ↔ -2 ≤ a ∧ a ≤ 2 :=
by
  sorry

end quadratic_inequality_solution_l21_21926


namespace enclosed_region_area_l21_21065

theorem enclosed_region_area :
  (∃ x y : ℝ, x ^ 2 + y ^ 2 - 6 * x + 8 * y = -9) →
  ∃ (r : ℝ), r ^ 2 = 16 ∧ ∀ (area : ℝ), area = π * 4 ^ 2 :=
by
  sorry

end enclosed_region_area_l21_21065


namespace geometric_sequence_term_number_l21_21125

theorem geometric_sequence_term_number 
  (a_n : ℕ → ℝ)
  (a1 : ℝ) (q : ℝ) (n : ℕ)
  (h1 : a1 = 1/2)
  (h2 : q = 1/2)
  (h3 : a_n n = 1/32)
  (h4 : ∀ n, a_n n = a1 * (q^(n-1))) :
  n = 5 := 
by
  sorry

end geometric_sequence_term_number_l21_21125


namespace sum_of_two_numbers_l21_21164

theorem sum_of_two_numbers (a b : ℕ) (h1 : (a + b) * (a - b) = 1996) (h2 : (a + b) % 2 = (a - b) % 2) (h3 : a + b > a - b) : a + b = 998 := 
sorry

end sum_of_two_numbers_l21_21164


namespace order_of_logs_l21_21619

noncomputable def a : ℝ := Real.log 6 / Real.log 3
noncomputable def b : ℝ := Real.log 12 / Real.log 6
noncomputable def c : ℝ := Real.log 16 / Real.log 8

theorem order_of_logs : a > b ∧ b > c := 
by
  sorry

end order_of_logs_l21_21619


namespace product_divisible_by_six_l21_21475

theorem product_divisible_by_six (a : ℤ) : 6 ∣ a * (a + 1) * (2 * a + 1) := 
sorry

end product_divisible_by_six_l21_21475


namespace solve_for_y_l21_21444

variable {y : ℚ}
def algebraic_expression_1 (y : ℚ) : ℚ := 4 * y + 8
def algebraic_expression_2 (y : ℚ) : ℚ := 8 * y - 7

theorem solve_for_y (h : algebraic_expression_1 y = - algebraic_expression_2 y) : y = -1 / 12 :=
by
  sorry

end solve_for_y_l21_21444


namespace smallest_rectangles_cover_square_l21_21261

theorem smallest_rectangles_cover_square :
  ∃ (n : ℕ), n = 8 ∧ ∀ (a : ℕ), ∀ (b : ℕ), (a = 2) ∧ (b = 4) → 
  ∃ (s : ℕ), s = 8 ∧ (s * s) / (a * b) = n :=
by
  sorry

end smallest_rectangles_cover_square_l21_21261


namespace tom_apple_fraction_l21_21225

theorem tom_apple_fraction (initial_oranges initial_apples oranges_sold_fraction oranges_remaining total_fruits_remaining apples_initial apples_sold_fraction : ℕ→ℚ) :
  initial_oranges = 40 →
  initial_apples = 70 →
  oranges_sold_fraction = 1 / 4 →
  oranges_remaining = initial_oranges - initial_oranges * oranges_sold_fraction →
  total_fruits_remaining = 65 →
  total_fruits_remaining = oranges_remaining + (initial_apples - initial_apples * apples_sold_fraction) →
  apples_sold_fraction = 1 / 2 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end tom_apple_fraction_l21_21225


namespace pencil_count_l21_21071

theorem pencil_count (a : ℕ) (h1 : 200 ≤ a) (h2 : a ≤ 300)
    (h3 : a % 10 = 7) (h4 : a % 12 = 9) : a = 237 ∨ a = 297 :=
by {
  sorry
}

end pencil_count_l21_21071


namespace prob_none_three_win_prob_at_least_two_not_win_l21_21168

-- Definitions for probabilities
def prob_win : ℚ := 1 / 6
def prob_not_win : ℚ := 1 - prob_win

-- Problem 1: Prove probability that none of the three students win
theorem prob_none_three_win : (prob_not_win ^ 3) = 125 / 216 := by
  sorry

-- Problem 2: Prove probability that at least two of the three students do not win
theorem prob_at_least_two_not_win : 1 - (3 * (prob_win ^ 2) * prob_not_win + prob_win ^ 3) = 25 / 27 := by
  sorry

end prob_none_three_win_prob_at_least_two_not_win_l21_21168


namespace no_integer_solutions_x_x_plus_1_eq_13y_plus_1_l21_21398

theorem no_integer_solutions_x_x_plus_1_eq_13y_plus_1 :
  ¬ ∃ x y : ℤ, x * (x + 1) = 13 * y + 1 :=
by sorry

end no_integer_solutions_x_x_plus_1_eq_13y_plus_1_l21_21398


namespace factorize_expr1_factorize_expr2_l21_21990

open BigOperators

/-- Given m and n, prove that m^3 n - 9 m n can be factorized as mn(m + 3)(m - 3). -/
theorem factorize_expr1 (m n : ℤ) : m^3 * n - 9 * m * n = n * m * (m + 3) * (m - 3) :=
sorry

/-- Given a, prove that a^3 + a - 2a^2 can be factorized as a(a - 1)^2. -/
theorem factorize_expr2 (a : ℤ) : a^3 + a - 2 * a^2 = a * (a - 1)^2 :=
sorry

end factorize_expr1_factorize_expr2_l21_21990


namespace percentage_increase_first_year_l21_21556

theorem percentage_increase_first_year (P : ℝ) (X : ℝ) 
  (h1 : P * (1 + X / 100) * 0.75 * 1.15 = P * 1.035) : 
  X = 20 :=
by
  sorry

end percentage_increase_first_year_l21_21556


namespace area_of_pentagon_m_n_l21_21896

noncomputable def m : ℤ := 12
noncomputable def n : ℤ := 11

theorem area_of_pentagon_m_n :
  let pentagon_area := (Real.sqrt m) + (Real.sqrt n)
  m + n = 23 :=
by
  have m_pos : m > 0 := by sorry
  have n_pos : n > 0 := by sorry
  sorry

end area_of_pentagon_m_n_l21_21896


namespace common_difference_arithmetic_sequence_l21_21656

theorem common_difference_arithmetic_sequence
  (S : ℕ → ℕ)
  (a : ℕ → ℕ)
  (d : ℕ) 
  (h₁ : ∀ n, S n = (n * (2 * a 0 + (n - 1) * d)) / 2) -- sum formula for arithmetic sequence
  (h₂ : 2 * S 3 = 3 * S 2 + 6) : 
  d = 2 := 
sorry

end common_difference_arithmetic_sequence_l21_21656


namespace certain_number_value_l21_21584

theorem certain_number_value
  (x : ℝ)
  (y : ℝ)
  (h1 : (28 + x + 42 + 78 + y) / 5 = 62)
  (h2 : (48 + 62 + 98 + 124 + x) / 5 = 78) : 
  y = 104 :=
by
  -- Proof goes here
  sorry

end certain_number_value_l21_21584


namespace tan_7pi_over_6_eq_1_over_sqrt_3_l21_21551

theorem tan_7pi_over_6_eq_1_over_sqrt_3 : 
  ∀ θ : ℝ, θ = (7 * Real.pi) / 6 → Real.tan θ = 1 / Real.sqrt 3 :=
by
  intros θ hθ
  rw [hθ]
  sorry  -- Proof to be completed

end tan_7pi_over_6_eq_1_over_sqrt_3_l21_21551


namespace polynomial_inequality_l21_21179

theorem polynomial_inequality (a b c : ℝ)
  (h1 : ∃ r1 r2 r3 : ℝ, (r1 ≠ r2 ∧ r1 ≠ r3 ∧ r2 ≠ r3) ∧ 
    (∀ t : ℝ, (t - r1) * (t - r2) * (t - r3) = t^3 + a*t^2 + b*t + c))
  (h2 : ¬ ∃ x : ℝ, (x^2 + x + 2013)^3 + a*(x^2 + x + 2013)^2 + b*(x^2 + x + 2013) + c = 0) :
  t^3 + a*2013^2 + b*2013 + c > 1 / 64 :=
sorry

end polynomial_inequality_l21_21179


namespace emily_age_l21_21389

theorem emily_age (A B C D E : ℕ) (h1 : A = B - 4) (h2 : B = C + 5) (h3 : D = C + 2) (h4 : E = A + D - B) (h5 : B = 20) : E = 13 :=
by sorry

end emily_age_l21_21389


namespace surface_area_hemisphere_l21_21365

theorem surface_area_hemisphere
  (r : ℝ)
  (h₁ : 4 * Real.pi * r^2 = 4 * Real.pi * r^2)
  (h₂ : Real.pi * r^2 = 3) :
  3 * Real.pi * r^2 = 9 :=
by
  sorry

end surface_area_hemisphere_l21_21365


namespace expand_polynomial_l21_21396

theorem expand_polynomial (t : ℝ) :
  (3 * t^3 - 4 * t^2 + 5 * t - 3) * (4 * t^2 - 2 * t + 1) = 12 * t^5 - 22 * t^4 + 31 * t^3 - 26 * t^2 + 11 * t - 3 := by
  sorry

end expand_polynomial_l21_21396


namespace possible_values_of_d_l21_21135

theorem possible_values_of_d (r s : ℝ) (c d : ℝ)
  (h1 : ∃ u, u = -r - s ∧ r * s + r * u + s * u = c)
  (h2 : ∃ v, v = -r - s - 8 ∧ (r - 3) * (s + 5) + (r - 3) * (u - 8) + (s + 5) * (u - 8) = c)
  (u_eq : u = -r - s)
  (v_eq : v = -r - s - 8)
  (polynomial_relation : d + 156 = -((r - 3) * (s + 5) * (u - 8))) : 
  d = -198 ∨ d = 468 := 
sorry

end possible_values_of_d_l21_21135


namespace min_sum_of_M_and_N_l21_21811

noncomputable def Alice (x : ℕ) : ℕ := 3 * x + 2
noncomputable def Bob (x : ℕ) : ℕ := 2 * x + 27

-- Define the result after 4 moves
noncomputable def Alice_4_moves (M : ℕ) : ℕ := Alice (Alice (Alice (Alice M)))
noncomputable def Bob_4_moves (N : ℕ) : ℕ := Bob (Bob (Bob (Bob N)))

theorem min_sum_of_M_and_N :
  ∃ (M N : ℕ), Alice_4_moves M = Bob_4_moves N ∧ M + N = 10 :=
sorry

end min_sum_of_M_and_N_l21_21811


namespace new_man_weight_l21_21210

theorem new_man_weight (avg_increase : ℝ) (crew_weight : ℝ) (new_man_weight : ℝ) 
(h_avg_increase : avg_increase = 1.8) (h_crew_weight : crew_weight = 53) :
  new_man_weight = crew_weight + 10 * avg_increase :=
by
  -- Here we will use the conditions to prove the theorem
  sorry

end new_man_weight_l21_21210


namespace area_of_third_face_l21_21449

-- Define the variables for the dimensions of the box: l, w, and h
variables (l w h: ℝ)

-- Given conditions
def face1_area := 120
def face2_area := 72
def volume := 720

-- The relationships between the dimensions and the given areas/volume
def face1_eq : Prop := l * w = face1_area
def face2_eq : Prop := w * h = face2_area
def volume_eq : Prop := l * w * h = volume

-- The statement we need to prove is that the area of the third face (l * h) is 60 cm² given the above equations
theorem area_of_third_face :
  face1_eq l w →
  face2_eq w h →
  volume_eq l w h →
  l * h = 60 :=
by
  intros h1 h2 h3
  sorry

end area_of_third_face_l21_21449


namespace work_done_in_one_day_l21_21356

theorem work_done_in_one_day (A_time B_time : ℕ) (hA : A_time = 4) (hB : B_time = A_time / 2) : 
  (1 / A_time + 1 / B_time) = (3 / 4) :=
by
  -- Here we are setting up the conditions as per our identified steps
  rw [hA, hB]
  -- The remaining steps to prove will be omitted as per instructions
  sorry

end work_done_in_one_day_l21_21356


namespace smallest_sphere_radius_l21_21759

theorem smallest_sphere_radius :
  ∃ (R : ℝ), (∀ (a b : ℝ), a = 14 → b = 12 → ∃ (h : ℝ), h = Real.sqrt (12^2 - (14 * Real.sqrt 2 / 2)^2) ∧ R = 7 * Real.sqrt 2 ∧ h ≤ R) :=
sorry

end smallest_sphere_radius_l21_21759


namespace angle_sum_of_roots_of_complex_eq_32i_l21_21139

noncomputable def root_angle_sum : ℝ :=
  let θ1 := 22.5
  let θ2 := 112.5
  let θ3 := 202.5
  let θ4 := 292.5
  θ1 + θ2 + θ3 + θ4

theorem angle_sum_of_roots_of_complex_eq_32i :
  root_angle_sum = 630 := by
  sorry

end angle_sum_of_roots_of_complex_eq_32i_l21_21139


namespace sum_of_slopes_eq_zero_l21_21029

theorem sum_of_slopes_eq_zero
  (p : ℝ) (a : ℝ) (hp : p > 0) (ha : a > 0)
  (P Q : ℝ × ℝ)
  (hP : P.2 ^ 2 = 2 * p * P.1)
  (hQ : Q.2 ^ 2 = 2 * p * Q.1)
  (hcollinear : ∃ m : ℝ, ∀ (x y : (ℝ × ℝ)), y = P ∨ y = Q ∨ y = (-a, 0) → y.2 = m * (y.1 + a)) :
  let k_AP := (P.2) / (P.1 - a)
  let k_AQ := (Q.2) / (Q.1 - a)
  k_AP + k_AQ = 0 := by
    sorry

end sum_of_slopes_eq_zero_l21_21029


namespace sum_of_values_of_M_l21_21421

theorem sum_of_values_of_M (M : ℝ) (h : M * (M - 8) = 12) :
  (∃ M1 M2 : ℝ, M^2 - 8 * M - 12 = 0 ∧ M1 + M2 = 8) :=
sorry

end sum_of_values_of_M_l21_21421


namespace count_words_200_l21_21995

theorem count_words_200 : 
  let single_word_numbers := 29
  let compound_words_21_to_99 := 144
  let compound_words_100_to_199 := 54 + 216
  single_word_numbers + compound_words_21_to_99 + compound_words_100_to_199 = 443 :=
by
  sorry

end count_words_200_l21_21995


namespace interval_probability_l21_21425

theorem interval_probability (x y z : ℝ) (h0x : 0 < x) (hy : y > 0) (hz : z > 0) (hx : x < y) (hy_mul_z_eq_half : y * z = 1 / 2) (hx_eq_third_y : x = 1 / 3 * y) :
  (x / y) = (2 / 3) :=
by
  -- Here you should provide the proof
  sorry

end interval_probability_l21_21425


namespace minimum_sum_of_dimensions_of_box_l21_21845

theorem minimum_sum_of_dimensions_of_box (a b c : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_vol : a * b * c = 2310) :
  a + b + c ≥ 52 :=
sorry

end minimum_sum_of_dimensions_of_box_l21_21845


namespace laura_garden_daisies_l21_21386

/-
Laura's Garden Problem: Given the ratio of daisies to tulips is 3:4,
Laura currently has 32 tulips, and she plans to add 24 more tulips,
prove that Laura will have 42 daisies in total after the addition to
maintain the same ratio.
-/

theorem laura_garden_daisies (daisies tulips add_tulips : ℕ) (ratio_d : ℕ) (ratio_t : ℕ)
    (h1 : ratio_d = 3) (h2 : ratio_t = 4) (h3 : tulips = 32) (h4 : add_tulips = 24)
    (new_tulips : ℕ := tulips + add_tulips) :
  daisies = 42 :=
by
  sorry

end laura_garden_daisies_l21_21386


namespace extreme_values_of_f_range_of_a_for_intersection_l21_21952

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x^2 - 9 * x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := 15 * x + a

theorem extreme_values_of_f :
  f (-1) = 5 ∧ f 3 = -27 :=
by {
  sorry
}

theorem range_of_a_for_intersection (a : ℝ) : 
  (-80 < a) ∧ (a < 28) ↔ ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ f x₁ = g x₁ a ∧ f x₂ = g x₂ a ∧ f x₃ = g x₃ a :=
by {
  sorry
}

end extreme_values_of_f_range_of_a_for_intersection_l21_21952


namespace value_of_ac_over_bd_l21_21271

theorem value_of_ac_over_bd (a b c d : ℝ) 
  (h1 : a = 4 * b)
  (h2 : b = 3 * c)
  (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 := 
by
  sorry

end value_of_ac_over_bd_l21_21271


namespace parallel_lines_implies_value_of_a_l21_21219

theorem parallel_lines_implies_value_of_a (a : ℝ) :
  (∀ x y : ℝ, ax + 2*y = 0 ∧ x + (a-1)*y + (a^2-1) = 0 → 
  (- a / 2) = - (1 / (a-1))) → a = 2 :=
sorry

end parallel_lines_implies_value_of_a_l21_21219


namespace determine_phi_l21_21043

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := Real.sin (2 * x + φ)

theorem determine_phi 
  (φ : ℝ)
  (H1 : ∀ x : ℝ, f x φ ≤ |f (π / 6) φ|)
  (H2 : f (π / 3) φ > f (π / 2) φ) :
  φ = π / 6 :=
sorry

end determine_phi_l21_21043


namespace simplify_expression_l21_21768

variable {s r : ℝ}

theorem simplify_expression :
  (2 * s^2 + 5 * r - 4) - (3 * s^2 + 9 * r - 7) = -s^2 - 4 * r + 3 := 
by
  sorry

end simplify_expression_l21_21768


namespace find_omega_l21_21509

theorem find_omega (ω : ℝ) (h₀ : ω > 0) (h₁ : (π / ω = π / 2)) : ω = 2 :=
sorry

end find_omega_l21_21509


namespace mapping_f_of_neg2_and_3_l21_21724

-- Define the mapping f
def f (x y : ℝ) : ℝ × ℝ := (x + y, x * y)

-- Define the given point
def p : ℝ × ℝ := (-2, 3)

-- Define the expected corresponding point
def expected_p : ℝ × ℝ := (1, -6)

-- The theorem stating the problem to be proved
theorem mapping_f_of_neg2_and_3 :
  f p.1 p.2 = expected_p := by
  sorry

end mapping_f_of_neg2_and_3_l21_21724


namespace general_term_of_series_l21_21512

def gen_term (a : ℕ → ℕ) : Prop :=
∀ n : ℕ, a n = if n = 1 then 2 else 6 * n - 5

def series_sum (S : ℕ → ℕ) : Prop :=
∀ n : ℕ, S n = 3 * n ^ 2 - 2 * n + 1

theorem general_term_of_series (a S : ℕ → ℕ) (h : series_sum S) :
  gen_term a ↔ (∀ n : ℕ, a n = if n = 1 then 2 else S n - S (n - 1)) :=
by sorry

end general_term_of_series_l21_21512


namespace ratio_of_poets_to_novelists_l21_21015

-- Define the conditions
def total_people : ℕ := 24
def novelists : ℕ := 15
def poets := total_people - novelists

-- Theorem asserting the ratio of poets to novelists
theorem ratio_of_poets_to_novelists (h1 : poets = total_people - novelists) : poets / novelists = 3 / 5 := by
  sorry

end ratio_of_poets_to_novelists_l21_21015


namespace determine_functions_l21_21850

noncomputable def satisfies_condition (f : ℕ → ℕ) : Prop :=
∀ (n p : ℕ), Prime p → (f n)^p % f p = n % f p

theorem determine_functions :
  ∀ (f : ℕ → ℕ),
  satisfies_condition f →
  f = id ∨
  (∀ p: ℕ, Prime p → f p = 1) ∨
  (f 2 = 2 ∧ (∀ p: ℕ, Prime p → p > 2 → f p = 1) ∧ ∀ n: ℕ, f n % 2 = n % 2) :=
by
  intros f h1
  sorry

end determine_functions_l21_21850


namespace max_rectangle_area_l21_21527

theorem max_rectangle_area (a b : ℝ) (h : 2 * a + 2 * b = 60) :
  a * b ≤ 225 :=
by
  sorry

end max_rectangle_area_l21_21527


namespace average_k_l21_21878

open Nat

def positive_integer_roots (a b : ℕ) : Prop :=
  a * b = 24 ∧ a + b = b + a

theorem average_k (k : ℕ) :
  (positive_integer_roots 1 24 ∨ 
  positive_integer_roots 2 12 ∨ 
  positive_integer_roots 3 8 ∨ 
  positive_integer_roots 4 6) →
  (k = 25 ∨ k = 14 ∨ k = 11 ∨ k = 10) →
  (25 + 14 + 11 + 10) / 4 = 15 := by
  sorry

end average_k_l21_21878


namespace part_1_part_2_l21_21085

-- Definitions based on given conditions
def a : ℕ → ℝ := λ n => 2 * n + 1
noncomputable def b : ℕ → ℝ := λ n => 1 / ((2 * n + 1)^2 - 1)
noncomputable def S : ℕ → ℝ := λ n => n ^ 2 + 2 * n
noncomputable def T : ℕ → ℝ := λ n => n / (4 * (n + 1))

-- Lean statement for proving the problem
theorem part_1 (n : ℕ) :
  ∀ a_3 a_5 a_7 : ℝ, 
  a 3 = a_3 → 
  a_3 = 7 →
  a_5 = a 5 →
  a_7 = a 7 →
  a_5 + a_7 = 26 →
  ∃ a_1 d : ℝ,
    (a 1 = a_1 + 0 * d) ∧
    (a 2 = a_1 + 1 * d) ∧
    (a 3 = a_1 + 2 * d) ∧
    (a 4 = a_1 + 3 * d) ∧
    (a 5 = a_1 + 4 * d) ∧
    (a 7 = a_1 + 6 * d) ∧
    (a n = a_1 + (n - 1) * d) ∧
    (S n = n^2 + 2*n) := sorry

theorem part_2 (n : ℕ) :
  ∀ a_n b_n : ℝ,
  b n = b_n →
  a n = a_n →
  1 / b n = a_n^2 - 1 →
  T n = τ →
  (T n = n / (4 * (n + 1))) := sorry

end part_1_part_2_l21_21085


namespace solve_for_x_l21_21505

theorem solve_for_x (x : ℚ) :
  (x^2 - 4*x + 3) / (x^2 - 7*x + 6) = (x^2 - 3*x - 10) / (x^2 - 2*x - 15) →
  x = -3 / 4 :=
by
  intro h
  sorry

end solve_for_x_l21_21505


namespace heartsuit_4_6_l21_21184

-- Define the operation \heartsuit
def heartsuit (x y : ℤ) : ℤ := 5 * x + 3 * y

-- Prove that 4 \heartsuit 6 = 38 under the given operation definition
theorem heartsuit_4_6 : heartsuit 4 6 = 38 := by
  -- Using the definition of \heartsuit
  -- Calculation is straightforward and skipped by sorry
  sorry

end heartsuit_4_6_l21_21184


namespace product_of_two_numbers_l21_21351

-- State the conditions and the proof problem
theorem product_of_two_numbers (x y : ℤ) (h_sum : x + y = 30) (h_diff : x - y = 6) :
  x * y = 216 :=
by
  -- Skipping the proof with 'sorry'
  sorry

end product_of_two_numbers_l21_21351


namespace chocolate_bars_percentage_l21_21746

noncomputable def total_chocolate_bars (milk dark almond white caramel : ℕ) : ℕ :=
  milk + dark + almond + white + caramel

noncomputable def percentage (count total : ℕ) : ℚ :=
  (count : ℚ) / (total : ℚ) * 100

theorem chocolate_bars_percentage :
  let milk := 36
  let dark := 21
  let almond := 40
  let white := 15
  let caramel := 28
  let total := total_chocolate_bars milk dark almond white caramel
  total = 140 ∧
  percentage milk total = 25.71 ∧
  percentage dark total = 15 ∧
  percentage almond total = 28.57 ∧
  percentage white total = 10.71 ∧
  percentage caramel total = 20 :=
by
  sorry

end chocolate_bars_percentage_l21_21746


namespace find_varphi_intervals_of_increase_l21_21893

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := Real.sin (2 * x + φ)

theorem find_varphi (φ : ℝ) (h1 : -Real.pi < φ) (h2 : φ < 0)
  (h3 : ∃ k : ℤ, 2 * (Real.pi / 8) + φ = (Real.pi / 2) + k * Real.pi) :
  φ = -3 * Real.pi / 4 :=
sorry

theorem intervals_of_increase (m : ℤ) :
  ∀ x : ℝ, (π / 8 + m * π ≤ x ∧ x ≤ 5 * π / 8 + m * π) ↔
  Real.sin (2 * x - 3 * π / 4) > 0 :=
sorry

end find_varphi_intervals_of_increase_l21_21893


namespace smallest_integer_l21_21808

theorem smallest_integer (x : ℤ) (h : 3 * (Int.natAbs x)^3 + 5 < 56) : x = -2 :=
sorry

end smallest_integer_l21_21808


namespace cos_A_eq_l21_21354

variable (A : Real) (A_interior_angle_tri_ABC : A > π / 2 ∧ A < π) (tan_A_eq_neg_two : Real.tan A = -2)

theorem cos_A_eq : Real.cos A = - (Real.sqrt 5) / 5 := by
  sorry

end cos_A_eq_l21_21354


namespace circles_intersect_if_and_only_if_l21_21080

noncomputable def circle1 (x y : ℝ) : Prop :=
  x^2 + y^2 - 2 * x + 10 * y + 1 = 0

noncomputable def circle2 (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2 * x + 2 * y - m = 0

theorem circles_intersect_if_and_only_if (m : ℝ) :
  (∃ x y : ℝ, circle1 x y ∧ circle2 x y m) ↔ -1 < m ∧ m < 79 :=
by {
  sorry
}

end circles_intersect_if_and_only_if_l21_21080


namespace sailboat_rental_cost_l21_21824

-- Define the conditions
def rental_per_hour_ski := 80
def hours_per_day := 3
def days := 2
def cost_ski := (hours_per_day * days * rental_per_hour_ski)
def additional_cost := 120

-- Statement to prove
theorem sailboat_rental_cost :
  ∃ (S : ℕ), cost_ski = S + additional_cost → S = 360 := by
  sorry

end sailboat_rental_cost_l21_21824


namespace find_m_l21_21658

-- Circle equation: x^2 + y^2 + 2x - 6y + 1 = 0
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 2 * x - 6 * y + 1 = 0

-- Line equation: x + m * y + 4 = 0
def line_eq (x y m : ℝ) : Prop := x + m * y + 4 = 0

-- Prove that the value of m such that the center of the circle lies on the line is -1
theorem find_m (m : ℝ) : 
  (∃ x y : ℝ, circle_eq x y ∧ (x, y) = (-1, 3) ∧ line_eq x y m) → m = -1 :=
by {
  sorry
}

end find_m_l21_21658


namespace unique_B_squared_l21_21432

variable (B : Matrix (Fin 2) (Fin 2) ℝ)

theorem unique_B_squared (h : B ^ 4 = 0) :
  ∃! B2 : Matrix (Fin 2) (Fin 2) ℝ, B ^ 2 = B2 :=
by sorry

end unique_B_squared_l21_21432


namespace complete_set_contains_all_rationals_l21_21621

theorem complete_set_contains_all_rationals (T : Set ℚ) (hT : ∀ (p q : ℚ), p / q ∈ T → p / (p + q) ∈ T ∧ q / (p + q) ∈ T) (r : ℚ) : 
  (r = 1 ∨ r = 1 / 2) → (∀ x : ℚ, 0 < x ∧ x < 1 → x ∈ T) :=
by
  sorry

end complete_set_contains_all_rationals_l21_21621


namespace clara_weight_l21_21839

-- Define the weights of Alice and Clara
variables (a c : ℕ)

-- Define the conditions given in the problem
def condition1 := a + c = 240
def condition2 := c - a = c / 3

-- The theorem to prove Clara's weight given the conditions
theorem clara_weight : condition1 a c → condition2 a c → c = 144 :=
by
  intros h1 h2
  sorry

end clara_weight_l21_21839


namespace paint_cost_contribution_l21_21727

theorem paint_cost_contribution
  (paint_cost_per_gallon : ℕ) 
  (coverage_per_gallon : ℕ) 
  (total_wall_area : ℕ) 
  (two_coats : ℕ) 
  : paint_cost_per_gallon = 45 → coverage_per_gallon = 400 → total_wall_area = 1600 → two_coats = 2 → 
    ((total_wall_area / coverage_per_gallon) * two_coats * paint_cost_per_gallon) / 2 = 180 :=
by
  intros h1 h2 h3 h4
  sorry

end paint_cost_contribution_l21_21727


namespace find_natural_numbers_l21_21879

theorem find_natural_numbers :
  ∃ (x y : ℕ), 
    x * y - (x + y) = Nat.gcd x y + Nat.lcm x y ∧ 
    ((x = 6 ∧ y = 3) ∨ (x = 6 ∧ y = 4) ∨ (x = 3 ∧ y = 6) ∨ (x = 4 ∧ y = 6)) := 
by 
  sorry

end find_natural_numbers_l21_21879


namespace least_possible_faces_combined_l21_21541

noncomputable def hasValidDiceConfiguration : Prop :=
  ∃ a b : ℕ,
  (∃ s8 s12 s13 : ℕ,
    (s8 = 3) ∧
    (s12 = 4) ∧
    (a ≥ 5 ∧ b = 6 ∧ (a + b = 11) ∧
      (2 * s12 = s8) ∧
      (2 * s8 = s13))
  )

theorem least_possible_faces_combined : hasValidDiceConfiguration :=
  sorry

end least_possible_faces_combined_l21_21541


namespace minimum_distance_square_l21_21122

/-- Given the equation of a circle centered at (2,3) with radius 1, find the minimum value of 
the function z = x^2 + y^2 -/
theorem minimum_distance_square (x y : ℝ) 
  (h : (x - 2)^2 + (y - 3)^2 = 1) : ∃ (z : ℝ), z = x^2 + y^2 ∧ z = 14 - 2 * Real.sqrt 13 :=
sorry

end minimum_distance_square_l21_21122


namespace pens_bought_l21_21171

theorem pens_bought
  (P : ℝ)
  (cost := 36 * P)
  (discount := 0.99 * P)
  (profit_percent := 0.1)
  (profit := (40 * discount) - cost)
  (profit_eq : profit = profit_percent * cost) :
  40 = 40 := 
by
  sorry

end pens_bought_l21_21171


namespace total_rings_is_19_l21_21885

-- Definitions based on the problem conditions
def rings_on_first_day : Nat := 8
def rings_on_second_day : Nat := 6
def rings_on_third_day : Nat := 5

-- Total rings calculation
def total_rings : Nat := rings_on_first_day + rings_on_second_day + rings_on_third_day

-- Proof statement
theorem total_rings_is_19 : total_rings = 19 := by
  -- Proof goes here
  sorry

end total_rings_is_19_l21_21885


namespace integral_value_l21_21987

-- Define the binomial coefficient
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Define the conditions of the problem
def a : ℝ := 2 -- This is derived from the problem condition

-- The main theorem statement
theorem integral_value :
  (∫ x in (0 : ℝ)..a, (Real.exp x + 2 * x)) = Real.exp 2 + 3 := by
  sorry

end integral_value_l21_21987


namespace value_is_correct_l21_21683

-- Define the number
def initial_number : ℝ := 4400

-- Define the value calculation in Lean
def value : ℝ := 0.15 * (0.30 * (0.50 * initial_number))

-- The theorem statement
theorem value_is_correct : value = 99 := by
  sorry

end value_is_correct_l21_21683


namespace sam_average_speed_l21_21672

theorem sam_average_speed :
  let total_time := 7 -- total time from 7 a.m. to 2 p.m.
  let rest_time := 1 -- rest period from 9 a.m. to 10 a.m.
  let effective_time := total_time - rest_time
  let total_distance := 200 -- total miles covered
  let avg_speed := total_distance / effective_time
  avg_speed = 33.3 :=
sorry

end sam_average_speed_l21_21672


namespace common_chord_length_common_chord_diameter_eq_circle_l21_21606

/-
Given two circles C1: x^2 + y^2 - 2x + 10y - 24 = 0 and C2: x^2 + y^2 + 2x + 2y - 8 = 0,
prove that 
1. The length of the common chord is 2 * sqrt(5).
2. The equation of the circle that has the common chord as its diameter is (x + 8/5)^2 + (y - 6/5)^2 = 36/5.
-/

-- Define the first circle C1
def C1 (x y : ℝ) : Prop := x^2 + y^2 - 2 * x + 10 * y - 24 = 0

-- Define the second circle C2
def C2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y - 8 = 0

-- Prove the length of the common chord
theorem common_chord_length : ∃ d : ℝ, d = 2 * Real.sqrt 5 :=
sorry

-- Prove the equation of the circle that has the common chord as its diameter
theorem common_chord_diameter_eq_circle : ∃ (x y : ℝ → ℝ), (x + 8/5)^2 + (y - 6/5)^2 = 36/5 :=
sorry

end common_chord_length_common_chord_diameter_eq_circle_l21_21606


namespace horner_method_value_v2_at_minus_one_l21_21198

noncomputable def f (x : ℝ) : ℝ :=
  x^6 - 5*x^5 + 6*x^4 - 3*x^3 + 1.8*x^2 + 0.35*x + 2

theorem horner_method_value_v2_at_minus_one :
  let a : ℝ := -1
  let v_0 := 1
  let v_1 := v_0 * a - 5
  let v_2 := v_1 * a + 6
  v_2 = 12 :=
by
  intros
  sorry

end horner_method_value_v2_at_minus_one_l21_21198


namespace quadratic_function_positive_difference_l21_21696

/-- Given a quadratic function y = ax^2 + bx + c, where the coefficient a
indicates a downward-opening parabola (a < 0) and the y-intercept is positive (c > 0),
prove that the expression (c - a) is always positive. -/
theorem quadratic_function_positive_difference (a b c : ℝ) (h1 : a < 0) (h2 : c > 0) : c - a > 0 := 
by
  sorry

end quadratic_function_positive_difference_l21_21696


namespace sale_price_lower_than_original_l21_21455

noncomputable def original_price (p : ℝ) : ℝ := 
  p

noncomputable def increased_price (p : ℝ) : ℝ := 
  1.30 * p

noncomputable def sale_price (p : ℝ) : ℝ := 
  0.75 * increased_price p

theorem sale_price_lower_than_original (p : ℝ) : 
  sale_price p = 0.975 * p := 
sorry

end sale_price_lower_than_original_l21_21455


namespace function_passes_through_point_l21_21054

theorem function_passes_through_point :
  (∃ (a : ℝ), a = 1 ∧ (∀ (x y : ℝ), y = a * x + a → y = x + 1)) →
  ∃ x y : ℝ, x = -2 ∧ y = -1 ∧ y = x + 1 :=
by
  sorry

end function_passes_through_point_l21_21054


namespace max_cookie_price_l21_21013

theorem max_cookie_price (k p : ℕ) :
  8 * k + 3 * p < 200 →
  4 * k + 5 * p > 150 →
  k ≤ 19 :=
sorry

end max_cookie_price_l21_21013


namespace possible_values_of_f2001_l21_21017

noncomputable def f : ℕ → ℝ := sorry

theorem possible_values_of_f2001 (f : ℕ → ℝ)
    (H : ∀ a b : ℕ, a > 1 → b > 1 → ∀ d : ℕ, d = Nat.gcd a b → 
           f (a * b) = f d * (f (a / d) + f (b / d))) :
    f 2001 = 0 ∨ f 2001 = 1/2 :=
sorry

end possible_values_of_f2001_l21_21017


namespace trig_identity_proof_l21_21462

theorem trig_identity_proof :
  Real.sin (30 * Real.pi / 180) * Real.sin (75 * Real.pi / 180) + 
  Real.sin (60 * Real.pi / 180) * Real.sin (15 * Real.pi / 180) =
  Real.sqrt 2 / 2 := 
by
  sorry

end trig_identity_proof_l21_21462


namespace repeating_decimal_fraction_l21_21572

theorem repeating_decimal_fraction (h : 0.02 = 2 / 99) : 2.06 = 68 / 33 :=
by
  sorry

end repeating_decimal_fraction_l21_21572


namespace car_traveled_miles_per_gallon_city_l21_21725

noncomputable def miles_per_gallon_city (H C G : ℝ) : Prop :=
  (C = H - 18) ∧ (462 = H * G) ∧ (336 = C * G)

theorem car_traveled_miles_per_gallon_city :
  ∃ H G, miles_per_gallon_city H 48 G :=
by
  sorry

end car_traveled_miles_per_gallon_city_l21_21725


namespace geom_sequence_product_l21_21148

noncomputable def geom_seq (a : ℕ → ℝ) :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geom_sequence_product (a : ℕ → ℝ) (h1 : geom_seq a) (h2 : a 0 * a 4 = 4) :
  a 0 * a 1 * a 2 * a 3 * a 4 = 32 ∨ a 0 * a 1 * a 2 * a 3 * a 4 = -32 :=
by
  sorry

end geom_sequence_product_l21_21148


namespace discs_angular_velocity_relation_l21_21322

variables {r1 r2 ω1 ω2 : ℝ} -- Radii and angular velocities

-- Conditions:
-- Discs have radii r1 and r2, and angular velocities ω1 and ω2, respectively.
-- Discs come to a halt after being brought into contact via friction.
-- Discs have identical thickness and are made of the same material.
-- Prove the required relation.

theorem discs_angular_velocity_relation
  (h1 : r1 > 0)
  (h2 : r2 > 0)
  (halt_contact : ω1 * r1^3 = ω2 * r2^3) :
  ω1 * r1^3 = ω2 * r2^3 :=
sorry

end discs_angular_velocity_relation_l21_21322


namespace move_right_by_three_units_l21_21434

theorem move_right_by_three_units :
  (-1 + 3 = 2) :=
  by { sorry }

end move_right_by_three_units_l21_21434


namespace certain_number_eq_1000_l21_21645

theorem certain_number_eq_1000 (x : ℝ) (h : 3500 - x / 20.50 = 3451.2195121951218) : x = 1000 := 
by
  sorry

end certain_number_eq_1000_l21_21645


namespace SummitAcademy_Contestants_l21_21846

theorem SummitAcademy_Contestants (s j : ℕ)
  (h1 : s > 0)
  (h2 : j > 0)
  (hs : (1 / 3 : ℚ) * s = (3 / 4 : ℚ) * j) :
  s = (9 / 4 : ℚ) * j :=
sorry

end SummitAcademy_Contestants_l21_21846


namespace arithmetic_computation_l21_21488

theorem arithmetic_computation : 65 * 1515 - 25 * 1515 = 60600 := by
  sorry

end arithmetic_computation_l21_21488


namespace max_value_of_expression_l21_21929

noncomputable def maximum_value (x y z : ℝ) := 8 * x + 3 * y + 10 * z

theorem max_value_of_expression :
  ∀ (x y z : ℝ), 9 * x^2 + 4 * y^2 + 25 * z^2 = 1 → maximum_value x y z ≤ (Real.sqrt 481) / 6 :=
by
  sorry

end max_value_of_expression_l21_21929


namespace question1_question2_l21_21550

def f (x : ℝ) : ℝ := abs (x - 5) - abs (x - 2)

theorem question1 :
  (∃ x : ℝ, f x ≤ m) ↔ m ≥ -3 :=
sorry

theorem question2 :
  { x : ℝ | x^2 - 8*x + 15 + f x ≤ 0 } = { x | 5 - Real.sqrt 3 ≤ x ∧ x ≤ 6 } :=
sorry

end question1_question2_l21_21550


namespace copper_zinc_mixture_mass_bounds_l21_21213

theorem copper_zinc_mixture_mass_bounds :
  ∀ (x y : ℝ) (D1 D2 : ℝ),
    (400 = x + y) →
    (50 = x / D1 + y / D2) →
    (8.8 ≤ D1 ∧ D1 ≤ 9) →
    (7.1 ≤ D2 ∧ D2 ≤ 7.2) →
    (200 ≤ x ∧ x ≤ 233) ∧ (167 ≤ y ∧ y ≤ 200) :=
sorry

end copper_zinc_mixture_mass_bounds_l21_21213


namespace credit_limit_l21_21635

theorem credit_limit (paid_tuesday : ℕ) (paid_thursday : ℕ) (remaining_payment : ℕ) (full_payment : ℕ) 
  (h1 : paid_tuesday = 15) 
  (h2 : paid_thursday = 23) 
  (h3 : remaining_payment = 62) 
  (h4 : full_payment = paid_tuesday + paid_thursday + remaining_payment) : 
  full_payment = 100 := 
by
  sorry

end credit_limit_l21_21635


namespace snickers_bars_needed_l21_21340

-- Definitions of the conditions
def points_needed : ℕ := 2000
def chocolate_bunny_points : ℕ := 100
def number_of_chocolate_bunnies : ℕ := 8
def snickers_points : ℕ := 25

-- Derived conditions
def points_from_bunnies : ℕ := number_of_chocolate_bunnies * chocolate_bunny_points
def remaining_points : ℕ := points_needed - points_from_bunnies

-- Statement to prove
theorem snickers_bars_needed : ∀ (n : ℕ), n = remaining_points / snickers_points → n = 48 :=
by 
  sorry

end snickers_bars_needed_l21_21340


namespace prism_base_shape_l21_21111

theorem prism_base_shape (n : ℕ) (hn : 3 * n = 12) : n = 4 := by
  sorry

end prism_base_shape_l21_21111


namespace square_distance_l21_21285

theorem square_distance (a b c d e f: ℝ) 
  (side_length : ℝ)
  (AB : a = 0 ∧ b = side_length)
  (BC : c = side_length ∧ d = 0)
  (BE_dist : (a - b)^2 + (b - b)^2 = 25)
  (AE_dist : a^2 + (c - b)^2 = 144)
  (DF_dist : (d)^2 + (d)^2 = 25)
  (CF_dist : (d - c)^2 + e^2 = 144) :
  (f - d)^2 + (e - a)^2 = 578 :=
by
  -- Required to bypass the proof steps
  sorry

end square_distance_l21_21285


namespace sum_of_consecutive_integers_between_ln20_l21_21435

theorem sum_of_consecutive_integers_between_ln20 : ∃ a b : ℤ, a < b ∧ b = a + 1 ∧ 1 ≤ a ∧ a + 1 ≤ 3 ∧ (a + b = 4) :=
by
  sorry

end sum_of_consecutive_integers_between_ln20_l21_21435


namespace percentage_of_men_speaking_french_l21_21779

theorem percentage_of_men_speaking_french {total_employees men women french_speaking_employees french_speaking_women french_speaking_men : ℕ}
    (h1 : total_employees = 100)
    (h2 : men = 60)
    (h3 : women = 40)
    (h4 : french_speaking_employees = 50)
    (h5 : french_speaking_women = 14)
    (h6 : french_speaking_men = french_speaking_employees - french_speaking_women)
    (h7 : french_speaking_men * 100 / men = 60) : true :=
by
  sorry

end percentage_of_men_speaking_french_l21_21779


namespace jack_runs_faster_than_paul_l21_21819

noncomputable def convert_km_hr_to_m_s (v : ℝ) : ℝ :=
  v * (1000 / 3600)

noncomputable def speed_difference : ℝ :=
  let v_J_km_hr := 20.62665  -- Jack's speed in km/hr
  let v_J_m_s := convert_km_hr_to_m_s v_J_km_hr  -- Jack's speed in m/s
  let distance := 1000  -- distance in meters
  let time_J := distance / v_J_m_s  -- Jack's time in seconds
  let time_P := time_J + 1.5  -- Paul's time in seconds
  let v_P_m_s := distance / time_P  -- Paul's speed in m/s
  let speed_diff_m_s := v_J_m_s - v_P_m_s  -- speed difference in m/s
  let speed_diff_km_hr := speed_diff_m_s * (3600 / 1000)  -- convert to km/hr
  speed_diff_km_hr

theorem jack_runs_faster_than_paul : speed_difference = 0.18225 :=
by
  -- Proof is omitted
  sorry

end jack_runs_faster_than_paul_l21_21819


namespace geom_seq_sum_l21_21001

theorem geom_seq_sum (q : ℝ) (a₃ a₄ a₅ : ℝ) : 
  0 < q ∧ 3 * (1 - q^3) / (1 - q) = 21 ∧ a₃ = 3 * q^2 ∧ a₄ = 3 * q^3 ∧ a₅ = 3 * q^4 
  -> a₃ + a₄ + a₅ = 84 := 
by 
  sorry

end geom_seq_sum_l21_21001


namespace max_additional_bags_correct_l21_21837

-- Definitions from conditions
def num_people : ℕ := 6
def bags_per_person : ℕ := 5
def weight_per_bag : ℕ := 50
def max_plane_capacity : ℕ := 6000

-- Derived definitions from conditions
def total_bags : ℕ := num_people * bags_per_person
def total_weight_of_bags : ℕ := total_bags * weight_per_bag
def remaining_capacity : ℕ := max_plane_capacity - total_weight_of_bags 
def max_additional_bags : ℕ := remaining_capacity / weight_per_bag

-- Theorem statement
theorem max_additional_bags_correct : max_additional_bags = 90 := by
  -- Proof skipped
  sorry

end max_additional_bags_correct_l21_21837


namespace find_A_l21_21371

theorem find_A (A : ℕ) (h : 59 = (A * 6) + 5) : A = 9 :=
by sorry

end find_A_l21_21371


namespace ellipses_have_equal_focal_length_l21_21350

-- Define ellipses and their focal lengths
def ellipse1_focal_length : ℝ := 8
def k_condition (k : ℝ) : Prop := 0 < k ∧ k < 9
def ellipse2_focal_length (k : ℝ) : ℝ := 8

-- The main statement
theorem ellipses_have_equal_focal_length (k : ℝ) (hk : k_condition k) :
  ellipse1_focal_length = ellipse2_focal_length k :=
sorry

end ellipses_have_equal_focal_length_l21_21350


namespace AM_GM_proof_equality_condition_l21_21233

variable (a b : ℝ)
variable (ha : 0 < a) (hb : 0 < b)

theorem AM_GM_proof : (a + b)^3 / (a^2 * b) ≥ 27 / 4 :=
sorry

theorem equality_condition : (a + b)^3 / (a^2 * b) = 27 / 4 ↔ a = 2 * b :=
sorry

end AM_GM_proof_equality_condition_l21_21233


namespace a1_minus_2a2_plus_3a3_minus_4a4_eq_48_l21_21838

theorem a1_minus_2a2_plus_3a3_minus_4a4_eq_48:
  ∀ (a a_1 a_2 a_3 a_4 : ℝ),
  (∀ x : ℝ, (1 + 2 * x) ^ 4 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4) →
  a_1 - 2 * a_2 + 3 * a_3 - 4 * a_4 = 48 :=
by
  sorry

end a1_minus_2a2_plus_3a3_minus_4a4_eq_48_l21_21838


namespace determine_X_with_7_gcd_queries_l21_21137

theorem determine_X_with_7_gcd_queries : 
  ∀ (X : ℕ), (X ≤ 100) → ∃ (f : Fin 7 → ℕ × ℕ), 
    (∀ i, (f i).1 < 100 ∧ (f i).2 < 100) ∧ (∃ (Y : Fin 7 → ℕ), 
      (∀ i, Y i = Nat.gcd (X + (f i).1) (f i).2) → 
        (∀ (X' : ℕ), (X' ≤ 100) → ((∀ i, Y i = Nat.gcd (X' + (f i).1) (f i).2) → X' = X))) :=
sorry

end determine_X_with_7_gcd_queries_l21_21137


namespace initial_men_count_l21_21677

theorem initial_men_count (M : ℕ) (A : ℕ) (H1 : 58 - (20 + 22) = 2 * M) : M = 8 :=
by
  sorry

end initial_men_count_l21_21677


namespace undefined_value_of_expression_l21_21123

theorem undefined_value_of_expression (a : ℝ) : (a^3 - 8 = 0) → (a = 2) := by
  sorry

end undefined_value_of_expression_l21_21123


namespace carl_additional_hours_per_week_l21_21481

def driving_hours_per_day : ℕ := 2

def days_per_week : ℕ := 7

def total_hours_two_weeks_after_promotion : ℕ := 40

def driving_hours_per_week_before_promotion : ℕ := driving_hours_per_day * days_per_week

def driving_hours_per_week_after_promotion : ℕ := total_hours_two_weeks_after_promotion / 2

def additional_hours_per_week : ℕ := driving_hours_per_week_after_promotion - driving_hours_per_week_before_promotion

theorem carl_additional_hours_per_week : 
  additional_hours_per_week = 6 :=
by
  -- Using plain arithmetic based on given definitions
  sorry

end carl_additional_hours_per_week_l21_21481


namespace part_a_part_b_l21_21438

variable (f : ℝ → ℝ)

-- Part (a)
theorem part_a (h : ∀ x y : ℝ, f (x + y) ≥ f x + y * f (f x)) :
  ∀ x : ℝ, f (f x) ≤ 0 :=
sorry

-- Part (b)
theorem part_b (h : ∀ x y : ℝ, f (x + y) ≥ f x + y * f (f x)) (h₀ : f 0 ≥ 0) :
  ∀ x : ℝ, f x = 0 :=
sorry

end part_a_part_b_l21_21438


namespace prob_one_mistake_eq_l21_21473

-- Define the probability of making a mistake on a single question
def prob_mistake : ℝ := 0.1

-- Define the probability of answering correctly on a single question
def prob_correct : ℝ := 1 - prob_mistake

-- Define the probability of answering all three questions correctly
def three_correct : ℝ := prob_correct ^ 3

-- Define the probability of making at least one mistake in three questions
def prob_at_least_one_mistake := 1 - three_correct

-- The theorem states that the above probability is equal to 1 - 0.9^3
theorem prob_one_mistake_eq :
  prob_at_least_one_mistake = 1 - (0.9 ^ 3) :=
by
  sorry

end prob_one_mistake_eq_l21_21473


namespace rented_room_percentage_l21_21070

theorem rented_room_percentage (total_rooms : ℕ) (h1 : 3 * total_rooms / 4 = 3 * total_rooms / 4) 
                               (h2 : 3 * total_rooms / 5 = 3 * total_rooms / 5) 
                               (h3 : 2 * (3 * total_rooms / 5) / 3 = 2 * (3 * total_rooms / 5) / 3) :
  (1 * (3 * total_rooms / 5) / 5) / (1 * total_rooms / 4) * 100 = 80 := by
  sorry

end rented_room_percentage_l21_21070


namespace circle_equation_exists_l21_21973

-- Define the necessary conditions
def tangent_to_x_axis (r b : ℝ) : Prop :=
  r^2 = b^2

def center_on_line (a b : ℝ) : Prop :=
  3 * a - b = 0

def intersects_formula (a b r : ℝ) : Prop :=
  2 * r^2 = (a - b)^2 + 14

-- Main theorem combining the conditions and proving the circles' equations
theorem circle_equation_exists (a b r : ℝ) :
  tangent_to_x_axis r b →
  center_on_line a b →
  intersects_formula a b r →
  ((x - 1)^2 + (y - 3)^2 = 9 ∨ (x + 1)^2 + (y + 3)^2 = 9) :=
by
  intros h_tangent h_center h_intersects
  sorry

end circle_equation_exists_l21_21973


namespace intersection_correct_l21_21898

open Set

def M := {x : ℝ | x^2 + x - 6 < 0}
def N := {x : ℝ | 1 ≤ x ∧ x ≤ 3}
def intersection := (M ∩ N) = {x : ℝ | 1 ≤ x ∧ x < 2}

theorem intersection_correct : intersection := by
  sorry

end intersection_correct_l21_21898


namespace range_of_function_is_correct_l21_21627

def range_of_quadratic_function : Set ℝ :=
  {y | ∃ x : ℝ, y = -x^2 - 6 * x - 5}

theorem range_of_function_is_correct :
  range_of_quadratic_function = {y | y ≤ 4} :=
by
  -- sorry allows skipping the actual proof step
  sorry

end range_of_function_is_correct_l21_21627


namespace initial_percentage_of_water_is_20_l21_21224

theorem initial_percentage_of_water_is_20 : 
  ∀ (P : ℝ) (total_initial_volume added_water total_final_volume final_percentage initial_water_percentage : ℝ), 
    total_initial_volume = 125 ∧ 
    added_water = 8.333333333333334 ∧ 
    total_final_volume = total_initial_volume + added_water ∧ 
    final_percentage = 25 ∧ 
    initial_water_percentage = (initial_water_percentage / total_initial_volume) * 100 ∧ 
    (final_percentage / 100) * total_final_volume = added_water + (initial_water_percentage / 100) * total_initial_volume → 
    initial_water_percentage = 20 := 
by 
  sorry

end initial_percentage_of_water_is_20_l21_21224


namespace horse_rent_problem_l21_21105

theorem horse_rent_problem (total_rent : ℝ) (b_payment : ℝ) (a_horses b_horses c_horses : ℝ) 
  (a_months b_months c_months : ℝ) (h_total_rent : total_rent = 870) (h_b_payment : b_payment = 360)
  (h_a_horses : a_horses = 12) (h_b_horses : b_horses = 16) (h_c_horses : c_horses = 18) 
  (h_b_months : b_months = 9) (h_c_months : c_months = 6) : 
  ∃ (a_months : ℝ), (a_horses * a_months * 2.5 + b_payment + c_horses * c_months * 2.5 = total_rent) :=
by
  use 8
  sorry

end horse_rent_problem_l21_21105


namespace average_monthly_increase_is_20_percent_l21_21760

-- Define the given conditions in Lean
def V_Jan : ℝ := 2 
def V_Mar : ℝ := 2.88 

-- Percentage increase each month over the previous month is the same
def consistent_growth_rate (x : ℝ) : Prop := 
  V_Jan * (1 + x)^2 = V_Mar

-- We need to prove that the monthly growth rate x is 0.2 (or 20%)
theorem average_monthly_increase_is_20_percent : 
  ∃ x : ℝ, consistent_growth_rate x ∧ x = 0.2 :=
by
  sorry

end average_monthly_increase_is_20_percent_l21_21760


namespace round_robin_10_players_l21_21810

theorem round_robin_10_players : @Nat.choose 10 2 = 45 := by
  sorry

end round_robin_10_players_l21_21810


namespace magic_triangle_max_sum_l21_21144

/-- In a magic triangle, each of the six consecutive whole numbers 11 to 16 is placed in one of the circles. 
    The sum, S, of the three numbers on each side of the triangle is the same. One of the sides must contain 
    three consecutive numbers. Prove that the largest possible value for S is 41. -/
theorem magic_triangle_max_sum :
  ∀ (a b c d e f : ℕ), 
  (a = 11 ∨ a = 12 ∨ a = 13 ∨ a = 14 ∨ a = 15 ∨ a = 16) ∧
  (b = 11 ∨ b = 12 ∨ b = 13 ∨ b = 14 ∨ b = 15 ∨ b = 16) ∧
  (c = 11 ∨ c = 12 ∨ c = 13 ∨ c = 14 ∨ c = 15 ∨ c = 16) ∧
  (d = 11 ∨ d = 12 ∨ d = 13 ∨ d = 14 ∨ d = 15 ∨ d = 16) ∧
  (e = 11 ∨ e = 12 ∨ e = 13 ∨ e = 14 ∨ e = 15 ∨ e = 16) ∧
  (f = 11 ∨ f = 12 ∨ f = 13 ∨ f = 14 ∨ f = 15 ∨ f = 16) ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
  d ≠ e ∧ d ≠ f ∧ e ≠ f ∧
  (a + b + c = S) ∧ (c + d + e = S) ∧ (e + f + a = S) ∧
  (∃ k, a = k ∧ b = k+1 ∧ c = k+2 ∨ b = k ∧ c = k+1 ∧ d = k+2 ∨ c = k ∧ d = k+1 ∧ e = k+2 ∨ d = k ∧ e = k+1 ∧ f = k+2) →
  S = 41 :=
by
  sorry

end magic_triangle_max_sum_l21_21144


namespace marble_probability_l21_21797

theorem marble_probability (W G R B : ℕ) (h_total : W + G + R + B = 84) 
  (h_white : W / 84 = 1 / 4) (h_green : G / 84 = 1 / 7) :
  (R + B) / 84 = 17 / 28 :=
by
  sorry

end marble_probability_l21_21797


namespace maximum_a_pos_integer_greatest_possible_value_of_a_l21_21815

theorem maximum_a_pos_integer (a : ℕ) (h : ∃ x : ℤ, x^2 + (a * x : ℤ) = -20) : a ≤ 21 :=
by
  sorry

theorem greatest_possible_value_of_a : ∃ (a : ℕ), (∀ b : ℕ, (∃ x : ℤ, x^2 + (b * x : ℤ) = -20) → b ≤ 21) ∧ 21 = a :=
by
  sorry

end maximum_a_pos_integer_greatest_possible_value_of_a_l21_21815


namespace taxi_fare_distance_condition_l21_21782

theorem taxi_fare_distance_condition (x : ℝ) (h1 : 7 + (max (x - 3) 0) * 2.4 = 19) : x ≤ 8 := 
by
  sorry

end taxi_fare_distance_condition_l21_21782


namespace taoqi_has_higher_utilization_rate_l21_21228

noncomputable def area_square (side_length : ℝ) : ℝ :=
  side_length * side_length

noncomputable def area_circle (radius : ℝ) : ℝ :=
  Real.pi * radius * radius

noncomputable def utilization_rate (cut_area : ℝ) (original_area : ℝ) : ℝ :=
  cut_area / original_area

noncomputable def tao_qi_utilization_rate : ℝ :=
  let side_length := 9
  let square_area := area_square side_length
  let radius := side_length / 2
  let circle_area := area_circle radius
  utilization_rate circle_area square_area

noncomputable def xiao_xiao_utilization_rate : ℝ :=
  let diameter := 9
  let radius := diameter / 2
  let large_circle_area := area_circle radius
  let small_circle_radius := diameter / 6
  let small_circle_area := area_circle small_circle_radius
  let total_small_circles_area := 7 * small_circle_area
  utilization_rate total_small_circles_area large_circle_area

-- Theorem statement reflecting the proof problem:
theorem taoqi_has_higher_utilization_rate :
  tao_qi_utilization_rate > xiao_xiao_utilization_rate := by sorry

end taoqi_has_higher_utilization_rate_l21_21228


namespace right_triangle_acute_angle_le_45_l21_21112

theorem right_triangle_acute_angle_le_45
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (hright : a^2 + b^2 = c^2):
  ∃ θ φ : ℝ, θ + φ = 90 ∧ (θ ≤ 45 ∨ φ ≤ 45) :=
by
  sorry

end right_triangle_acute_angle_le_45_l21_21112


namespace determine_all_functions_l21_21940

-- Define the natural numbers (ℕ) as positive integers
def is_perfect_square (x : ℕ) : Prop :=
  ∃ k : ℕ, k * k = x

theorem determine_all_functions (g : ℕ → ℕ) :
  (∀ m n : ℕ, is_perfect_square ((g m + n) * (m + g n))) →
  ∃ c : ℕ, ∀ n : ℕ, g n = n + c :=
by
  sorry

end determine_all_functions_l21_21940


namespace count_subsets_l21_21316

theorem count_subsets (S T : Set ℕ) (h1 : S = {1, 2, 3}) (h2 : T = {1, 2, 3, 4, 5, 6, 7}) :
  (∃ n : ℕ, n = 16 ∧ ∀ X, S ⊆ X ∧ X ⊆ T ↔ X ∈ { X | ∃ m : ℕ, m = 16 }) := 
sorry

end count_subsets_l21_21316


namespace classify_quadrilateral_l21_21569

structure Quadrilateral where
  sides : ℕ → ℝ 
  angle : ℕ → ℝ 
  diag_length : ℕ → ℝ 
  perpendicular_diagonals : Prop

def is_rhombus (q : Quadrilateral) : Prop :=
  (∀ i, q.sides i = q.sides 0) ∧ q.perpendicular_diagonals

def is_kite (q : Quadrilateral) : Prop :=
  (q.sides 1 = q.sides 2 ∧ q.sides 3 = q.sides 4) ∧ q.perpendicular_diagonals

def is_square (q : Quadrilateral) : Prop :=
  (∀ i, q.sides i = q.sides 0) ∧ (∀ i, q.angle i = 90) ∧ q.perpendicular_diagonals

theorem classify_quadrilateral (q : Quadrilateral) (h : q.perpendicular_diagonals) :
  is_rhombus q ∨ is_kite q ∨ is_square q :=
sorry

end classify_quadrilateral_l21_21569


namespace zero_integers_satisfy_conditions_l21_21629

noncomputable def satisfies_conditions (n : ℤ) : Prop :=
  ∃ k : ℤ, n * (25 - n) = k^2 * (25 - n)^2 ∧ n % 3 = 0

theorem zero_integers_satisfy_conditions :
  (∃ n : ℤ, satisfies_conditions n) → False := by
  sorry

end zero_integers_satisfy_conditions_l21_21629


namespace adjusted_area_difference_l21_21504

noncomputable def largest_circle_area (d : ℝ) : ℝ :=
  let r := d / 2
  r^2 * Real.pi

noncomputable def middle_circle_area (r : ℝ) : ℝ :=
  r^2 * Real.pi

noncomputable def smaller_circle_area (r : ℝ) : ℝ :=
  r^2 * Real.pi

theorem adjusted_area_difference (d_large r_middle r_small : ℝ) 
  (h_large : d_large = 30) (h_middle : r_middle = 10) (h_small : r_small = 5) :
  largest_circle_area d_large - middle_circle_area r_middle - smaller_circle_area r_small = 100 * Real.pi :=
by
  sorry

end adjusted_area_difference_l21_21504


namespace positive_integer_pairs_l21_21830

theorem positive_integer_pairs (a b : ℕ) (h_pos : 0 < a ∧ 0 < b) :
  (∃ k : ℕ, k > 0 ∧ a^2 = k * (2 * a * b^2 - b^3 + 1)) ↔ 
  ∃ l : ℕ, 0 < l ∧ ((a = l ∧ b = 2 * l) ∨ (a = 8 * l^4 - l ∧ b = 2 * l)) :=
by 
  sorry

end positive_integer_pairs_l21_21830


namespace exactly_one_correct_l21_21778

theorem exactly_one_correct (P_A P_B : ℚ) (hA : P_A = 1/5) (hB : P_B = 1/4) :
  P_A * (1 - P_B) + (1 - P_A) * P_B = 7/20 :=
by
  sorry

end exactly_one_correct_l21_21778


namespace find_b_l21_21983

-- Define the function f(x)
def f (x : ℝ) : ℝ := 5 * x - 7

-- State the theorem
theorem find_b (b : ℝ) : f b = 0 ↔ b = 7 / 5 := by
  sorry

end find_b_l21_21983


namespace ratio_of_sums_equiv_seven_eighths_l21_21713

variable (p q r u v w : ℝ)
variable (hp : 0 < p) (hq : 0 < q) (hr : 0 < r)
variable (hu : 0 < u) (hv : 0 < v) (hw : 0 < w)
variable (h1 : p^2 + q^2 + r^2 = 49)
variable (h2 : u^2 + v^2 + w^2 = 64)
variable (h3 : p * u + q * v + r * w = 56)

theorem ratio_of_sums_equiv_seven_eighths :
  (p + q + r) / (u + v + w) = 7 / 8 :=
by
  sorry

end ratio_of_sums_equiv_seven_eighths_l21_21713


namespace vasya_max_consecutive_liked_numbers_l21_21607

def is_liked_by_vasya (n : ℕ) : Prop :=
  ∀ d ∈ (n.digits 10), d ≠ 0 → n % d = 0

theorem vasya_max_consecutive_liked_numbers : 
  ∃ (seq : ℕ → ℕ), 
    (∀ n, seq n = n ∧ is_liked_by_vasya (seq n)) ∧
    (∀ m, seq m + 1 < seq (m + 1)) ∧ seq 12 - seq 0 + 1 = 13 :=
sorry

end vasya_max_consecutive_liked_numbers_l21_21607


namespace twenty_percent_greater_than_40_l21_21637

theorem twenty_percent_greater_than_40 (x : ℝ) (h : x = 40 + 0.2 * 40) : x = 48 := by
sorry

end twenty_percent_greater_than_40_l21_21637


namespace total_pizza_eaten_l21_21422

def don_pizzas : ℝ := 80
def daria_pizzas : ℝ := 2.5 * don_pizzas
def total_pizzas : ℝ := don_pizzas + daria_pizzas

theorem total_pizza_eaten : total_pizzas = 280 := by
  sorry

end total_pizza_eaten_l21_21422


namespace unfolded_paper_has_eight_holes_l21_21410

theorem unfolded_paper_has_eight_holes
  (T : Type)
  (equilateral_triangle : T)
  (midpoint : T → T → T)
  (vertex_fold : T → T → T)
  (holes_punched : T → ℕ)
  (first_fold_vertex midpoint_1 : T)
  (second_fold_vertex midpoint_2 : T)
  (holes_near_first_fold holes_near_second_fold : ℕ) :
  holes_punched (vertex_fold second_fold_vertex midpoint_2)
    = 8 := 
by sorry

end unfolded_paper_has_eight_holes_l21_21410


namespace correct_propositions_for_curve_C_l21_21510

def curve_C (k : ℝ) : Prop :=
  ∀ x y : ℝ, (x^2 / (4 - k) + y^2 / (k - 1) = 1)

theorem correct_propositions_for_curve_C (k : ℝ) :
  (∀ x y : ℝ, curve_C k) →
  ((∃ k, ((4 - k) * (k - 1) < 0) ↔ (k < 1 ∨ k > 4)) ∧
  ((1 < k ∧ k < (5 : ℝ) / 2) ↔
  (4 - k > k - 1 ∧ 4 - k > 0 ∧ k - 1 > 0))) :=
by {
  sorry
}

end correct_propositions_for_curve_C_l21_21510


namespace count_numbers_with_cube_root_lt_8_l21_21844

theorem count_numbers_with_cube_root_lt_8 : 
  ∀ n : ℕ, (n > 0) → (n < 8^3) → n ≤ 8^3 - 1 :=
by
  -- We need to prove that the count of such numbers is 511
  sorry

end count_numbers_with_cube_root_lt_8_l21_21844


namespace sufficient_but_not_necessary_condition_l21_21974

theorem sufficient_but_not_necessary_condition 
  (a : ℕ → ℤ) 
  (h : ∀ n, |a (n + 1)| < a n) : 
  (∀ n, a (n + 1) < a n) ∧ 
  ¬(∀ n, a (n + 1) < a n → |a (n + 1)| < a n) := 
by 
  sorry

end sufficient_but_not_necessary_condition_l21_21974


namespace exponent_zero_value_of_neg_3_raised_to_zero_l21_21986

theorem exponent_zero (x : ℤ) (hx : x ≠ 0) : x ^ 0 = 1 :=
by
  -- Proof goes here
  sorry

theorem value_of_neg_3_raised_to_zero : (-3 : ℤ) ^ 0 = 1 :=
by
  exact exponent_zero (-3) (by norm_num)

end exponent_zero_value_of_neg_3_raised_to_zero_l21_21986


namespace total_travel_cost_is_47100_l21_21088

-- Define the dimensions of the lawn
def lawn_length : ℝ := 200
def lawn_breadth : ℝ := 150

-- Define the roads' widths and their respective travel costs per sq m
def road1_width : ℝ := 12
def road1_travel_cost : ℝ := 4
def road2_width : ℝ := 15
def road2_travel_cost : ℝ := 5
def road3_width : ℝ := 10
def road3_travel_cost : ℝ := 3
def road4_width : ℝ := 20
def road4_travel_cost : ℝ := 6

-- Define the areas of the roads
def road1_area : ℝ := lawn_length * road1_width
def road2_area : ℝ := lawn_length * road2_width
def road3_area : ℝ := lawn_breadth * road3_width
def road4_area : ℝ := lawn_breadth * road4_width

-- Define the costs for the roads
def road1_cost : ℝ := road1_area * road1_travel_cost
def road2_cost : ℝ := road2_area * road2_travel_cost
def road3_cost : ℝ := road3_area * road3_travel_cost
def road4_cost : ℝ := road4_area * road4_travel_cost

-- Define the total cost
def total_cost : ℝ := road1_cost + road2_cost + road3_cost + road4_cost

-- The theorem statement
theorem total_travel_cost_is_47100 : total_cost = 47100 := by
  sorry

end total_travel_cost_is_47100_l21_21088


namespace sum_of_four_numbers_in_ratio_is_correct_l21_21518

variable (A B C D : ℝ)
variable (h_ratio : A / B = 2 / 3 ∧ B / C = 3 / 4 ∧ C / D = 4 / 5)
variable (h_biggest : D = 672)

theorem sum_of_four_numbers_in_ratio_is_correct :
  A + B + C + D = 1881.6 :=
by
  sorry

end sum_of_four_numbers_in_ratio_is_correct_l21_21518


namespace number_of_boys_in_class_l21_21697

theorem number_of_boys_in_class (n : ℕ) (h : 182 * n - 166 + 106 = 180 * n) : n = 30 :=
by {
  sorry
}

end number_of_boys_in_class_l21_21697


namespace distance_between_vertices_of_hyperbola_l21_21590

def hyperbola_equation (x y : ℝ) : Prop :=
  ∃ (c₁ c₂ : ℝ), c₁ = 4 ∧ c₂ = -4 ∧
    (c₁ * x^2 + 24 * x + c₂ * y^2 + 8 * y + 44 = 0)

theorem distance_between_vertices_of_hyperbola :
  (∀ x y : ℝ, hyperbola_equation x y) → (2 : ℝ) = 2 :=
by
  intro h
  sorry

end distance_between_vertices_of_hyperbola_l21_21590


namespace gwen_money_received_from_dad_l21_21646

variables (D : ℕ)

-- Conditions
def mom_received := 8
def mom_more_than_dad := 3

-- Question and required proof
theorem gwen_money_received_from_dad : 
  (mom_received = D + mom_more_than_dad) -> D = 5 := 
by
  sorry

end gwen_money_received_from_dad_l21_21646


namespace transformation_result_l21_21392

def f (x y : ℝ) : ℝ × ℝ := (y, x)
def g (x y : ℝ) : ℝ × ℝ := (-x, -y)

theorem transformation_result : g (f (-6) (7)).1 (f (-6) (7)).2 = (-7, 6) :=
by
  sorry

end transformation_result_l21_21392


namespace greatest_perimeter_of_triangle_l21_21436

theorem greatest_perimeter_of_triangle :
  ∃ (x : ℕ), (3 * x) + 15 = 57 ∧ 
  (x > 5 ∧ x < 15) ∧ 
  2 * x + x > 15 ∧ 
  x + 15 > 2 * x ∧ 
  2 * x + 15 > x := 
sorry

end greatest_perimeter_of_triangle_l21_21436


namespace anthony_total_pencils_l21_21592

theorem anthony_total_pencils :
  let original_pencils := 9
  let given_pencils := 56
  original_pencils + given_pencils = 65 := by
  sorry

end anthony_total_pencils_l21_21592


namespace unique_nonzero_solution_l21_21560

theorem unique_nonzero_solution (x : ℝ) (h : x ≠ 0) : (3 * x)^3 = (9 * x)^2 → x = 3 :=
by
  sorry

end unique_nonzero_solution_l21_21560


namespace quadratic_distinct_real_roots_range_l21_21803

open Real

theorem quadratic_distinct_real_roots_range (k : ℝ) :
    (∃ a b c : ℝ, a = k^2 ∧ b = 4 * k - 1 ∧ c = 4 ∧ (b^2 - 4 * a * c > 0) ∧ a ≠ 0) ↔ (k < 1 / 8 ∧ k ≠ 0) :=
by
  sorry

end quadratic_distinct_real_roots_range_l21_21803


namespace boats_meet_time_l21_21419

theorem boats_meet_time (v_A v_C current distance : ℝ) : 
  v_A = 7 → 
  v_C = 3 → 
  current = 2 → 
  distance = 20 → 
  (distance / (v_A + current + v_C - current) = 2 ∨
   distance / (v_A + current - (v_C + current)) = 5) := 
by 
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  -- Apply simplifications or calculations as necessary
  sorry

end boats_meet_time_l21_21419


namespace difference_of_squares_l21_21011

theorem difference_of_squares 
  (x y : ℝ) 
  (optionA := (-x + y) * (x + y))
  (optionB := (-x + y) * (x - y))
  (optionC := (x + 2) * (2 + x))
  (optionD := (2 * x + 3) * (3 * x - 2)) :
  optionA = -(x + y)^2 ∨ optionA = (x + y) * (y - x) :=
sorry

end difference_of_squares_l21_21011


namespace simplify_expression_l21_21853

theorem simplify_expression (a : ℝ) (h : a ≠ 1) : 1 - (1 / (1 + ((a + 1) / (1 - a)))) = (1 + a) / 2 := 
by
  sorry

end simplify_expression_l21_21853


namespace emily_necklaces_l21_21661

theorem emily_necklaces (n beads_per_necklace total_beads : ℕ) (h1 : beads_per_necklace = 8) (h2 : total_beads = 16) : n = total_beads / beads_per_necklace → n = 2 :=
by sorry

end emily_necklaces_l21_21661


namespace members_not_in_A_nor_B_l21_21282

variable (U A B : Finset ℕ) -- We define the sets as finite sets of natural numbers.
variable (hU_size : U.card = 190) -- Size of set U is 190.
variable (hB_size : (U ∩ B).card = 49) -- 49 items are in set B.
variable (hAB_size : (A ∩ U ∩ B).card = 23) -- 23 items are in both A and B.
variable (hA_size : (U ∩ A).card = 105) -- 105 items are in set A.

theorem members_not_in_A_nor_B :
  (U \ (A ∪ B)).card = 59 := sorry

end members_not_in_A_nor_B_l21_21282


namespace production_growth_rate_eq_l21_21497

theorem production_growth_rate_eq 
  (x : ℝ)
  (H : 100 + 100 * (1 + x) + 100 * (1 + x) ^ 2 = 364) : 
  100 + 100 * (1 + x) + 100 * (1 + x) ^ 2 = 364 :=
by {
  sorry
}

end production_growth_rate_eq_l21_21497


namespace min_odd_integers_l21_21146

theorem min_odd_integers 
  (a b c d e f : ℤ)
  (h1 : a + b = 30)
  (h2 : c + d = 15)
  (h3 : e + f = 17)
  (h4 : c + d + e + f = 32) :
  ∃ n : ℕ, (n = 2) ∧ (∃ odd_count, 
  odd_count = (if (a % 2 = 0) then 0 else 1) + 
                     (if (b % 2 = 0) then 0 else 1) + 
                     (if (c % 2 = 0) then 0 else 1) + 
                     (if (d % 2 = 0) then 0 else 1) + 
                     (if (e % 2 = 0) then 0 else 1) + 
                     (if (f % 2 = 0) then 0 else 1) ∧
  odd_count = 2) := sorry

end min_odd_integers_l21_21146


namespace perceived_temperature_difference_l21_21300

theorem perceived_temperature_difference (N : ℤ) (M L : ℤ)
  (h1 : M = L + N)
  (h2 : M - 11 - (L + 5) = 6 ∨ M - 11 - (L + 5) = -6) :
  N = 22 ∨ N = 10 := by
  sorry

end perceived_temperature_difference_l21_21300


namespace find_s_2_l21_21327

def t (x : ℝ) : ℝ := 4 * x - 6
def s (y : ℝ) : ℝ := y^2 + 5 * y - 7

theorem find_s_2 : s 2 = 7 := by
  sorry

end find_s_2_l21_21327


namespace convert_mps_to_kmph_l21_21925

theorem convert_mps_to_kmph (v_mps : ℝ) (conversion_factor : ℝ) : v_mps = 22 → conversion_factor = 3.6 → v_mps * conversion_factor = 79.2 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num

end convert_mps_to_kmph_l21_21925


namespace asymptote_of_hyperbola_l21_21949

theorem asymptote_of_hyperbola (x y : ℝ) (h : (x^2 / 16) - (y^2 / 25) = 1) : 
  y = (5 / 4) * x :=
sorry

end asymptote_of_hyperbola_l21_21949


namespace rectangle_area_l21_21536

theorem rectangle_area (L B : ℕ) 
  (h1 : L - B = 23)
  (h2 : 2 * L + 2 * B = 186) :
  L * B = 2030 := by
  sorry

end rectangle_area_l21_21536


namespace area_of_triangle_l21_21699

theorem area_of_triangle {a c : ℝ} (B : ℝ) (h1 : a = 1) (h2 : c = 2) (h3 : B = 60) :
    (1 / 2) * a * c * Real.sin (B * Real.pi / 180) = Real.sqrt 3 / 2 := by
  sorry

end area_of_triangle_l21_21699


namespace sum_of_first_four_terms_of_geometric_sequence_l21_21062

noncomputable def geometric_sum_first_four (a : ℕ → ℝ) (q : ℝ) : ℝ :=
  a 0 + a 1 + a 2 + a 3

theorem sum_of_first_four_terms_of_geometric_sequence 
  (a : ℕ → ℝ) (q : ℝ) (h1 : ∀ n, a (n + 1) = a n * q) 
  (h2 : q > 0) 
  (h3 : a 2 = 1) 
  (h4 : ∀ n, a (n + 2) + a (n + 1) = 6 * a n) :
  geometric_sum_first_four a q = 15 / 2 :=
sorry

end sum_of_first_four_terms_of_geometric_sequence_l21_21062


namespace monotonic_decreasing_intervals_l21_21270

theorem monotonic_decreasing_intervals (α : ℝ) (hα : α < 0) :
  (∀ x y : ℝ, x < y ∧ x < 0 ∧ y < 0 → x ^ α > y ^ α) ∧ 
  (∀ x y : ℝ, x < y ∧ 0 < x ∧ 0 < y → x ^ α > y ^ α) :=
by
  sorry

end monotonic_decreasing_intervals_l21_21270


namespace find_period_for_interest_l21_21263

noncomputable def period_for_compound_interest (P : ℝ) (r : ℝ) (n : ℕ) (A : ℝ) : ℝ :=
  (Real.log A - Real.log P) / (n * Real.log (1 + r / n))

theorem find_period_for_interest :
  period_for_compound_interest 8000 0.15 1 11109 = 2 := 
sorry

end find_period_for_interest_l21_21263


namespace extracurricular_books_l21_21754

theorem extracurricular_books (a b c d : ℕ) 
  (h1 : b + c + d = 110)
  (h2 : a + c + d = 108)
  (h3 : a + b + d = 104)
  (h4 : a + b + c = 119) :
  a = 37 ∧ b = 39 ∧ c = 43 ∧ d = 28 :=
by {
  -- Proof to be done here
  sorry
}

end extracurricular_books_l21_21754


namespace cubic_root_sum_eq_constant_term_divided_l21_21009

theorem cubic_root_sum_eq_constant_term_divided 
  (a b c : ℝ) 
  (h_roots : (24 * a^3 - 36 * a^2 + 14 * a - 1 = 0) 
           ∧ (24 * b^3 - 36 * b^2 + 14 * b - 1 = 0) 
           ∧ (24 * c^3 - 36 * c^2 + 14 * c - 1 = 0))
  (h_bounds : 0 < a ∧ a < 1 ∧ 0 < b ∧ b < 1 ∧ 0 < c ∧ c < 1) 
  : (1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c)) = (158 / 73) := 
sorry

end cubic_root_sum_eq_constant_term_divided_l21_21009


namespace depth_of_second_hole_l21_21207

theorem depth_of_second_hole :
  let workers1 := 45
  let hours1 := 8
  let depth1 := 30
  let man_hours1 := workers1 * hours1 -- 360 man-hours
  let workers2 := 45 + 35 -- 80 workers
  let hours2 := 6
  let man_hours2 := workers2 * hours2 -- 480 man-hours
  let depth2 := (man_hours2 * depth1) / man_hours1 -- value to solve for
  depth2 = 40 :=
by
  sorry

end depth_of_second_hole_l21_21207


namespace gopi_salary_turbans_l21_21106

-- Define the question and conditions as statements
def total_salary (turbans : ℕ) : ℕ := 90 + 30 * turbans
def servant_receives : ℕ := 60 + 30
def fraction_annual_salary : ℚ := 3 / 4

-- The theorem statement capturing the equivalent proof problem
theorem gopi_salary_turbans (T : ℕ) 
  (salary_eq : total_salary T = 90 + 30 * T)
  (servant_eq : servant_receives = 60 + 30)
  (fraction_eq : fraction_annual_salary = 3 / 4)
  (received_after_9_months : ℚ) :
  fraction_annual_salary * (90 + 30 * T : ℚ) = received_after_9_months → 
  received_after_9_months = 90 →
  T = 1 :=
sorry

end gopi_salary_turbans_l21_21106


namespace tangent_line_parabola_l21_21586

theorem tangent_line_parabola (a : ℝ) :
  (∀ x y : ℝ, y^2 = 32 * x → 4 * x + 3 * y + a = 0) → a = 18 :=
by
  sorry

end tangent_line_parabola_l21_21586


namespace tens_digit_of_9_pow_1010_l21_21714

theorem tens_digit_of_9_pow_1010 : (9 ^ 1010) % 100 = 1 :=
by sorry

end tens_digit_of_9_pow_1010_l21_21714


namespace lassie_original_bones_l21_21535

variable (B : ℕ) -- B is the number of bones Lassie started with

-- Conditions translated into Lean statements
def eats_half_on_saturday (B : ℕ) : ℕ := B / 2
def receives_ten_more_on_sunday (B : ℕ) : ℕ := eats_half_on_saturday B + 10
def total_bones_after_sunday (B : ℕ) : Prop := receives_ten_more_on_sunday B = 35

-- Proof goal: B is equal to 50 given the conditions
theorem lassie_original_bones :
  total_bones_after_sunday B → B = 50 :=
sorry

end lassie_original_bones_l21_21535


namespace fraction_of_boxes_loaded_by_day_crew_l21_21626

theorem fraction_of_boxes_loaded_by_day_crew
    (dayCrewBoxesPerWorker : ℚ)
    (dayCrewWorkers : ℚ)
    (nightCrewBoxesPerWorker : ℚ := (3 / 4) * dayCrewBoxesPerWorker)
    (nightCrewWorkers : ℚ := (3 / 4) * dayCrewWorkers) :
    (dayCrewBoxesPerWorker * dayCrewWorkers) / ((dayCrewBoxesPerWorker * dayCrewWorkers) + (nightCrewBoxesPerWorker * nightCrewWorkers)) = 16 / 25 :=
by
  sorry

end fraction_of_boxes_loaded_by_day_crew_l21_21626


namespace more_stable_performance_l21_21061

theorem more_stable_performance (S_A2 S_B2 : ℝ) (hA : S_A2 = 0.2) (hB : S_B2 = 0.09) (h : S_A2 > S_B2) : 
  "B" = "B" :=
by
  sorry

end more_stable_performance_l21_21061


namespace probability_no_defective_pencils_l21_21157

theorem probability_no_defective_pencils :
  let total_pencils := 9
  let defective_pencils := 2
  let total_ways_choose_3 := Nat.choose total_pencils 3
  let non_defective_pencils := total_pencils - defective_pencils
  let ways_choose_3_non_defective := Nat.choose non_defective_pencils 3
  (ways_choose_3_non_defective : ℚ) / total_ways_choose_3 = 5 / 12 :=
by
  sorry

end probability_no_defective_pencils_l21_21157


namespace percent_of_employed_females_l21_21402

theorem percent_of_employed_females (p e m f : ℝ) (h1 : e = 0.60 * p) (h2 : m = 0.15 * p) (h3 : f = e - m):
  (f / e) * 100 = 75 :=
by
  -- We place the proof here
  sorry

end percent_of_employed_females_l21_21402


namespace number_of_subsets_l21_21991

-- Defining the type of the elements
variable {α : Type*}

-- Statement of the problem in Lean 4
theorem number_of_subsets (s : Finset α) (h : s.card = n) : (Finset.powerset s).card = 2^n := 
sorry

end number_of_subsets_l21_21991


namespace fill_pipe_fraction_l21_21695

theorem fill_pipe_fraction (x : ℝ) (h : x = 1 / 2) : x = 1 / 2 :=
by
  sorry

end fill_pipe_fraction_l21_21695


namespace xy_value_l21_21424

theorem xy_value (x y : ℝ) (h : (x - 3)^2 + |y + 2| = 0) : x * y = -6 :=
by {
  sorry
}

end xy_value_l21_21424


namespace number_of_boys_is_10_l21_21514

-- Definitions based on given conditions
def num_children := 20
def has_blue_neighbor_clockwise (i : ℕ) : Prop := true -- Dummy predicate representing condition
def has_red_neighbor_counterclockwise (i : ℕ) : Prop := true -- Dummy predicate representing condition

axiom boys_and_girls_exist : ∃ b g : ℤ, b + g = num_children ∧ b > 0 ∧ g > 0

-- Theorem based on the problem statement
theorem number_of_boys_is_10 (b g : ℤ) 
  (total_children: b + g = num_children)
  (boys_exist: b > 0)
  (girls_exist: g > 0)
  (each_boy_has_blue_neighbor: ∀ i, has_blue_neighbor_clockwise i → true)
  (each_girl_has_red_neighbor: ∀ i, has_red_neighbor_counterclockwise i → true): 
  b = 10 :=
by
  sorry

end number_of_boys_is_10_l21_21514


namespace evaluate_difference_floor_squares_l21_21988

theorem evaluate_difference_floor_squares (x : ℝ) (h : x = 15.3) : ⌊x^2⌋ - ⌊x⌋^2 = 9 := by
  sorry

end evaluate_difference_floor_squares_l21_21988


namespace complement_of_P_in_U_l21_21399

def U : Set ℤ := {-1, 0, 1, 2}
def P : Set ℤ := {x | -Real.sqrt 2 < x ∧ x < Real.sqrt 2}
def compl_U (P : Set ℤ) : Set ℤ := {x ∈ U | x ∉ P}

theorem complement_of_P_in_U : compl_U P = {2} :=
by
  sorry

end complement_of_P_in_U_l21_21399


namespace odd_function_and_monotonic_decreasing_l21_21051

variable (f : ℝ → ℝ)

-- Given conditions:
axiom condition_1 : ∀ x y : ℝ, f (x + y) = f x + f y
axiom condition_2 : ∀ x : ℝ, x > 0 → f x < 0

-- Statement to prove:
theorem odd_function_and_monotonic_decreasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x1 x2 : ℝ, x1 > x2 → f x1 < f x2) := by
  sorry

end odd_function_and_monotonic_decreasing_l21_21051


namespace inequality_proof_l21_21662

theorem inequality_proof (a b : ℝ) (h1 : a ≠ b) (h2 : a + b = 2) : ab < 1 ∧ 1 < (a^2 + b^2) / 2 :=
by
  sorry

end inequality_proof_l21_21662


namespace initial_cats_count_l21_21739

theorem initial_cats_count :
  ∀ (initial_birds initial_puppies initial_spiders final_total initial_cats: ℕ),
    initial_birds = 12 →
    initial_puppies = 9 →
    initial_spiders = 15 →
    final_total = 25 →
    (initial_birds / 2 + initial_puppies - 3 + initial_spiders - 7 + initial_cats = final_total) →
    initial_cats = 5 := by
  intros initial_birds initial_puppies initial_spiders final_total initial_cats h1 h2 h3 h4 h5
  sorry

end initial_cats_count_l21_21739


namespace tangent_line_circle_l21_21500

theorem tangent_line_circle (m : ℝ) : 
  (∀ (x y : ℝ), x + y + m = 0 → x^2 + y^2 = m) → m = 2 :=
by
  sorry

end tangent_line_circle_l21_21500


namespace cheapest_lamp_cost_l21_21573

/--
Frank wants to buy a new lamp for his bedroom. The cost of the cheapest lamp is some amount, and the most expensive in the store is 3 times more expensive. Frank has $90, and if he buys the most expensive lamp available, he would have $30 remaining. Prove that the cost of the cheapest lamp is $20.
-/
theorem cheapest_lamp_cost (c most_expensive : ℝ) (h_cheapest_lamp : most_expensive = 3 * c) 
(h_frank_money : 90 - most_expensive = 30) : c = 20 := 
sorry

end cheapest_lamp_cost_l21_21573


namespace odd_operations_l21_21931

theorem odd_operations (a b : ℤ) (ha : ∃ k : ℤ, a = 2 * k + 1) (hb : ∃ j : ℤ, b = 2 * j + 1) :
  (∃ k : ℤ, (a * b) = 2 * k + 1) ∧ (∃ m : ℤ, a^2 = 2 * m + 1) :=
by {
  sorry
}

end odd_operations_l21_21931


namespace sherman_total_weekly_driving_time_l21_21220

def daily_commute_time : Nat := 1  -- 1 hour for daily round trip commute time
def work_days : Nat := 5  -- Sherman works 5 days a week
def weekend_day_driving_time : Nat := 2  -- 2 hours of driving each weekend day
def weekend_days : Nat := 2  -- There are 2 weekend days

theorem sherman_total_weekly_driving_time :
  daily_commute_time * work_days + weekend_day_driving_time * weekend_days = 9 := 
by
  sorry

end sherman_total_weekly_driving_time_l21_21220


namespace pen_distribution_l21_21160

theorem pen_distribution (x : ℕ) :
  8 * x + 3 = 12 * (x - 2) - 1 :=
sorry

end pen_distribution_l21_21160


namespace asymptotes_of_hyperbola_l21_21822

theorem asymptotes_of_hyperbola 
  (x y : ℝ)
  (h : x^2 / 4 - y^2 / 36 = 1) : 
  (y = 3 * x) ∨ (y = -3 * x) :=
sorry

end asymptotes_of_hyperbola_l21_21822


namespace expected_value_is_correct_l21_21084

noncomputable def expected_winnings : ℝ :=
  (1/12 : ℝ) * (9 + 8 + 7 + 6 + 5 + 1 + 2 + 3 + 4 + 5 + 6 + 7)

theorem expected_value_is_correct : expected_winnings = 5.25 := by
  sorry

end expected_value_is_correct_l21_21084


namespace cadence_total_earnings_l21_21079

noncomputable def total_earnings (old_years : ℕ) (old_monthly : ℕ) (new_increment : ℤ) (extra_months : ℕ) : ℤ :=
  let old_months := old_years * 12
  let old_earnings := old_monthly * old_months
  let new_monthly := old_monthly + ((old_monthly * new_increment) / 100)
  let new_months := old_months + extra_months
  let new_earnings := new_monthly * new_months
  old_earnings + new_earnings

theorem cadence_total_earnings :
  total_earnings 3 5000 20 5 = 426000 :=
by
  sorry

end cadence_total_earnings_l21_21079


namespace prob_correct_last_digit_no_more_than_two_attempts_prob_correct_last_digit_no_more_than_two_attempts_if_even_l21_21716

/-
Prove that if a person forgets the last digit of their 6-digit password, which can be any digit from 0 to 9,
the probability of pressing the correct last digit in no more than 2 attempts is 1/5.
-/

theorem prob_correct_last_digit_no_more_than_two_attempts :
  let correct_prob := 1 / 10 
  let incorrect_prob := 9 / 10 
  let second_attempt_prob := 1 / 9 
  correct_prob + (incorrect_prob * second_attempt_prob) = 1 / 5 :=
by
  sorry

/-
Prove that if a person forgets the last digit of their 6-digit password, but remembers that the last digit is an even number,
the probability of pressing the correct last digit in no more than 2 attempts is 2/5.
-/

theorem prob_correct_last_digit_no_more_than_two_attempts_if_even :
  let correct_prob := 1 / 5 
  let incorrect_prob := 4 / 5 
  let second_attempt_prob := 1 / 4 
  correct_prob + (incorrect_prob * second_attempt_prob) = 2 / 5 :=
by
  sorry

end prob_correct_last_digit_no_more_than_two_attempts_prob_correct_last_digit_no_more_than_two_attempts_if_even_l21_21716


namespace natural_number_pairs_int_l21_21720

theorem natural_number_pairs_int {
  a b : ℕ
} : 
  (∃ a b : ℕ, 
    (b^2 - a ≠ 0 ∧ (a^2 + b) % (b^2 - a) = 0) ∧ 
    (a^2 - b ≠ 0 ∧ (b^2 + a) % (a^2 - b) = 0)
  ) ↔ ((a, b) = (1, 2) ∨ (a, b) = (2, 1) ∨ (a, b) = (2, 2) ∨ (a, b) = (2, 3) ∨ (a, b) = (3, 2) ∨ (a, b) = (3, 3)) :=
by sorry

end natural_number_pairs_int_l21_21720


namespace cost_of_first_10_kgs_of_apples_l21_21612

theorem cost_of_first_10_kgs_of_apples 
  (l q : ℝ) 
  (h1 : 30 * l + 3 * q = 663) 
  (h2 : 30 * l + 6 * q = 726) : 
  10 * l = 200 :=
by
  -- Proof would follow here
  sorry

end cost_of_first_10_kgs_of_apples_l21_21612


namespace find_sum_l21_21587

variable {x y z w : ℤ}

-- Conditions: Consecutive integers and their sum condition
def consecutive_integers (x y z : ℤ) : Prop := y = x + 1 ∧ z = x + 2
def sum_is_150 (x y z : ℤ) : Prop := x + y + z = 150
def w_definition (w z x : ℤ) : Prop := w = 2 * z - x

-- Theorem statement
theorem find_sum (h1 : consecutive_integers x y z) (h2 : sum_is_150 x y z) (h3 : w_definition w z x) :
  x + y + z + w = 203 :=
sorry

end find_sum_l21_21587


namespace range_of_m_l21_21595

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x
noncomputable def g (m x : ℝ) : ℝ := m * x + 1
noncomputable def h (x : ℝ) : ℝ := (1 / x) - (2 * Real.log x / x)

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, (x ∈ Set.Icc (1 / Real.exp 1) (Real.exp 2)) ∧ (g m x = 2 - 2 * f x)) ↔
  (-2 * Real.exp (-3/2) ≤ m ∧ m ≤ 3 * Real.exp 1) :=
sorry

end range_of_m_l21_21595


namespace verify_chebyshev_polynomials_l21_21277

-- Define the Chebyshev polynomials of the first kind Tₙ(x)
def T : ℕ → ℝ → ℝ
| 0, x => 1
| 1, x => x
| (n+1), x => 2 * x * T n x - T (n-1) x

-- Define the Chebyshev polynomials of the second kind Uₙ(x)
def U : ℕ → ℝ → ℝ
| 0, x => 1
| 1, x => 2 * x
| (n+1), x => 2 * x * U n x - U (n-1) x

-- State the theorem to verify the Chebyshev polynomials initial conditions and recurrence relations
theorem verify_chebyshev_polynomials (n : ℕ) (x : ℝ) :
  T 0 x = 1 ∧ T 1 x = x ∧
  U 0 x = 1 ∧ U 1 x = 2 * x ∧
  (T (n+1) x = 2 * x * T n x - T (n-1) x) ∧
  (U (n+1) x = 2 * x * U n x - U (n-1) x) := sorry

end verify_chebyshev_polynomials_l21_21277


namespace stamp_total_cost_l21_21273

theorem stamp_total_cost :
  let price_A := 2
  let price_B := 3
  let price_C := 5
  let num_A := 150
  let num_B := 90
  let num_C := 60
  let discount_A := if num_A > 100 then 0.20 else 0
  let discount_B := if num_B > 50 then 0.15 else 0
  let discount_C := if num_C > 30 then 0.10 else 0
  let cost_A := num_A * price_A * (1 - discount_A)
  let cost_B := num_B * price_B * (1 - discount_B)
  let cost_C := num_C * price_C * (1 - discount_C)
  cost_A + cost_B + cost_C = 739.50 := sorry

end stamp_total_cost_l21_21273


namespace ratio_Jake_sister_l21_21092

theorem ratio_Jake_sister (Jake_weight : ℕ) (total_weight : ℕ) (weight_loss : ℕ) (expected_ratio : ℕ) :
  Jake_weight = 113 →
  total_weight = 153 →
  weight_loss = 33 →
  expected_ratio = 2 →
  (Jake_weight - weight_loss) / (total_weight - Jake_weight) = expected_ratio :=
by
  intros hJake hTotal hLoss hRatio
  sorry

end ratio_Jake_sister_l21_21092


namespace bike_covered_distance_l21_21272

theorem bike_covered_distance
  (time : ℕ) 
  (truck_distance : ℕ) 
  (speed_difference : ℕ) 
  (bike_speed truck_speed : ℕ)
  (h_time : time = 8)
  (h_truck_distance : truck_distance = 112)
  (h_speed_difference : speed_difference = 3)
  (h_truck_speed : truck_speed = truck_distance / time)
  (h_speed_relation : truck_speed = bike_speed + speed_difference) :
  bike_speed * time = 88 :=
by
  -- The proof is omitted
  sorry

end bike_covered_distance_l21_21272


namespace election_votes_l21_21409

theorem election_votes (T V : ℕ) 
    (hT : 8 * T = 11 * 20000) 
    (h_total_votes : T = 2500 + V + 20000) :
    V = 5000 :=
by
    sorry

end election_votes_l21_21409


namespace sum_of_extrema_l21_21056

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x^3

-- Main statement to prove
theorem sum_of_extrema :
  let a := -1
  let b := 1
  let f_min := f a
  let f_max := f b
  f_min + f_max = Real.exp 1 + Real.exp (-1) :=
by
  sorry

end sum_of_extrema_l21_21056


namespace free_endpoints_can_be_1001_l21_21007

variables (initial_segs : ℕ) (total_free_ends : ℕ) (k : ℕ)

-- Initial setup: one initial segment.
def initial_segment : ℕ := 1

-- Each time 5 segments are drawn from a point, the number of free ends increases by 4.
def free_ends_after_k_actions (k : ℕ) : ℕ := initial_segment + 4 * k

-- Question: Can the number of free endpoints be exactly 1001?
theorem free_endpoints_can_be_1001 : free_ends_after_k_actions 250 = 1001 := by
  sorry

end free_endpoints_can_be_1001_l21_21007


namespace percentage_shaded_is_18_75_l21_21559

-- conditions
def total_squares: ℕ := 16
def shaded_squares: ℕ := 3

-- claim to prove
theorem percentage_shaded_is_18_75 :
  ((shaded_squares : ℝ) / total_squares) * 100 = 18.75 := 
by
  sorry

end percentage_shaded_is_18_75_l21_21559


namespace larger_number_is_21_l21_21494

theorem larger_number_is_21 (x y : ℤ) (h1 : x + y = 35) (h2 : x - y = 7) : x = 21 := 
by 
  sorry

end larger_number_is_21_l21_21494


namespace inequality_solution_l21_21790

theorem inequality_solution :
  {x : Real | (2 * x - 5) * (x - 3) / x ≥ 0} = {x : Real | (x ∈ Set.Ioc 0 (5 / 2)) ∨ (x ∈ Set.Ici 3)} := 
sorry

end inequality_solution_l21_21790


namespace part_one_retail_wholesale_l21_21933

theorem part_one_retail_wholesale (x : ℕ) (wholesale : ℕ) : 
  70 * x + 40 * wholesale = 4600 ∧ x + wholesale = 100 → x = 20 ∧ wholesale = 80 :=
by
  sorry

end part_one_retail_wholesale_l21_21933


namespace rowing_upstream_speed_l21_21133

theorem rowing_upstream_speed (V_m V_down V_up V_s : ℝ) 
  (hVm : V_m = 40) 
  (hVdown : V_down = 60) 
  (hVdown_eq : V_down = V_m + V_s) 
  (hVup_eq : V_up = V_m - V_s) : 
  V_up = 20 := 
by
  sorry

end rowing_upstream_speed_l21_21133


namespace find_angle_B_l21_21314

open Real

theorem find_angle_B (A B : ℝ) 
  (h1 : 0 < B ∧ B < A ∧ A < π/2)
  (h2 : cos A = 1/7) 
  (h3 : cos (A - B) = 13/14) : 
  B = π/3 :=
sorry

end find_angle_B_l21_21314


namespace daughter_age_is_10_l21_21632

variable (D : ℕ)

-- Conditions
def father_current_age (D : ℕ) : ℕ := 4 * D
def father_age_in_20_years (D : ℕ) : ℕ := father_current_age D + 20
def daughter_age_in_20_years (D : ℕ) : ℕ := D + 20

-- Theorem statement
theorem daughter_age_is_10 :
  father_current_age D = 40 →
  father_age_in_20_years D = 2 * daughter_age_in_20_years D →
  D = 10 :=
by
  -- Here would be the proof steps to show that D = 10 given the conditions
  sorry

end daughter_age_is_10_l21_21632


namespace max_arithmetic_sum_l21_21262

def a1 : ℤ := 113
def d : ℤ := -4

def S (n : ℕ) : ℤ := n * (2 * a1 + (n - 1) * d) / 2

theorem max_arithmetic_sum : S 29 = 1653 :=
by
  sorry

end max_arithmetic_sum_l21_21262


namespace sum_even_and_multiples_of_5_l21_21874

def num_even_four_digit : ℕ :=
  let thousands := 9 -- thousands place cannot be zero
  let hundreds := 10
  let tens := 10
  let units := 5 -- even digits: {0, 2, 4, 6, 8}
  thousands * hundreds * tens * units

def num_multiples_of_5_four_digit : ℕ :=
  let thousands := 9 -- thousands place cannot be zero
  let hundreds := 10
  let tens := 10
  let units := 2 -- multiples of 5 digits: {0, 5}
  thousands * hundreds * tens * units

theorem sum_even_and_multiples_of_5 : num_even_four_digit + num_multiples_of_5_four_digit = 6300 := by
  sorry

end sum_even_and_multiples_of_5_l21_21874


namespace solve_equation_1_solve_equation_2_l21_21880

theorem solve_equation_1 (x : ℝ) :
  x^2 - 10 * x + 16 = 0 → x = 8 ∨ x = 2 :=
by
  sorry

theorem solve_equation_2 (x : ℝ) :
  x * (x - 3) = 6 - 2 * x → x = 3 ∨ x = -2 :=
by
  sorry

end solve_equation_1_solve_equation_2_l21_21880


namespace prize_amount_l21_21643

theorem prize_amount (P : ℝ) (n : ℝ) (a : ℝ) (b : ℝ) (c : ℝ)
  (h1 : n = 40)
  (h2 : a = 40)
  (h3 : b = (2 / 5) * P)
  (h4 : c = (3 / 5) * 40)
  (h5 : b / c = 120) :
  P = 7200 := 
sorry

end prize_amount_l21_21643


namespace sum_of_geometric_sequence_l21_21176

-- Consider a geometric sequence {a_n} with the first term a_1 = 1 and a common ratio of 1/3.
-- Let S_n denote the sum of the first n terms.
-- We need to prove that S_n = (3 - a_n) / 2, given the above conditions.
noncomputable def geometric_sequence_sum (n : ℕ) : ℝ :=
  let a_1 := 1
  let r := (1 : ℝ) / 3
  let a_n := r ^ (n - 1)
  (3 - a_n) / 2

theorem sum_of_geometric_sequence (n : ℕ) : geometric_sequence_sum n = 
  let a_1 := 1
  let r := (1 : ℝ) / 3
  let a_n := r ^ (n - 1)
  (3 - a_n) / 2 := sorry

end sum_of_geometric_sequence_l21_21176


namespace nina_running_distance_l21_21813

theorem nina_running_distance (x : ℝ) (hx : 2 * x + 0.67 = 0.83) : x = 0.08 := by
  sorry

end nina_running_distance_l21_21813


namespace simplify_power_of_product_l21_21522

theorem simplify_power_of_product (x y : ℝ) : (3 * x^2 * y^3)^2 = 9 * x^4 * y^6 :=
by
  -- hint: begin proof here
  sorry

end simplify_power_of_product_l21_21522


namespace find_triangle_sides_l21_21593

variable (a b c : ℕ)
variable (P : ℕ)
variable (R : ℚ := 65 / 8)
variable (r : ℕ := 4)

theorem find_triangle_sides (h1 : R = 65 / 8) (h2 : r = 4) (h3 : P = a + b + c) : 
  a = 13 ∧ b = 14 ∧ c = 15 :=
  sorry

end find_triangle_sides_l21_21593


namespace ratio_length_to_breadth_l21_21927

theorem ratio_length_to_breadth (l b : ℕ) (h1 : b = 14) (h2 : l * b = 588) : l / b = 3 :=
by
  sorry

end ratio_length_to_breadth_l21_21927


namespace sourav_distance_l21_21684

def D (t : ℕ) : ℕ := 20 * t

theorem sourav_distance :
  ∀ (t : ℕ), 20 * t = 25 * (t - 1) → 20 * t = 100 :=
by
  intros t h
  sorry

end sourav_distance_l21_21684


namespace min_abs_ab_l21_21118

theorem min_abs_ab (a b : ℤ) (h : 1009 * a + 2 * b = 1) : ∃ k : ℤ, |a * b| = 504 :=
by
  sorry

end min_abs_ab_l21_21118


namespace triangle_to_square_difference_l21_21348

noncomputable def number_of_balls_in_triangle (T : ℕ) : ℕ :=
  T * (T + 1) / 2

noncomputable def number_of_balls_in_square (S : ℕ) : ℕ :=
  S * S

theorem triangle_to_square_difference (T S : ℕ) 
  (h1 : number_of_balls_in_triangle T = 1176) 
  (h2 : number_of_balls_in_square S = 1600) :
  T - S = 8 :=
by
  sorry

end triangle_to_square_difference_l21_21348


namespace line_intersections_with_parabola_l21_21682

theorem line_intersections_with_parabola :
  ∃! (L : ℝ → ℝ) (l_count : ℕ),  
    l_count = 3 ∧
    (∀ x : ℝ, (L x) ∈ {x | (L 0 = 2) ∧ ∃ y, y * y = 8 * x ∧ L x = y}) := sorry

end line_intersections_with_parabola_l21_21682


namespace road_path_distance_l21_21279

theorem road_path_distance (d_AB d_AC d_BC d_BD : ℕ) 
  (h1 : d_AB = 9) (h2 : d_AC = 13) (h3 : d_BC = 8) (h4 : d_BD = 14) : A_to_D = 19 :=
by
  sorry

end road_path_distance_l21_21279


namespace find_s_when_t_is_64_l21_21938

theorem find_s_when_t_is_64 (s : ℝ) (t : ℝ) (h1 : t = 8 * s^3) (h2 : t = 64) : s = 2 :=
by
  -- Proof will be written here
  sorry

end find_s_when_t_is_64_l21_21938


namespace solution_set_condition_l21_21442

-- The assumptions based on the given conditions
variables (a b : ℝ)

noncomputable def inequality_system_solution_set (x : ℝ) : Prop :=
  (x + 2 * a > 4) ∧ (2 * x - b < 5)

theorem solution_set_condition (a b : ℝ) :
  (∀ x : ℝ, inequality_system_solution_set a b x ↔ 0 < x ∧ x < 2) →
  (a + b) ^ 2023 = 1 :=
by
  intro h
  sorry

end solution_set_condition_l21_21442


namespace AM_GM_inequality_l21_21719

theorem AM_GM_inequality (a : List ℝ) (h : ∀ x ∈ a, 0 < x) :
  (a.sum / a.length) ≥ a.prod ^ (1 / a.length) := 
sorry

end AM_GM_inequality_l21_21719


namespace ellipse_h_k_a_c_sum_l21_21709

theorem ellipse_h_k_a_c_sum :
  let h := -3
  let k := 1
  let a := 4
  let c := 2
  h + k + a + c = 4 :=
by
  let h := -3
  let k := 1
  let a := 4
  let c := 2
  show h + k + a + c = 4
  sorry

end ellipse_h_k_a_c_sum_l21_21709


namespace range_a_l21_21814

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2 * a * x + 3

theorem range_a (H : ∀ x1 x2 : ℝ, 2 < x1 → 2 < x2 → (f x1 a - f x2 a) / (x1 - x2) > 0) : a ≤ 2 := 
sorry

end range_a_l21_21814


namespace factorize_a_squared_plus_2a_l21_21579

theorem factorize_a_squared_plus_2a (a : ℝ) : a^2 + 2 * a = a * (a + 2) :=
  sorry

end factorize_a_squared_plus_2a_l21_21579


namespace quadratic_func_inequality_l21_21616

theorem quadratic_func_inequality (c : ℝ) (f : ℝ → ℝ) 
  (h_def : ∀ x, f x = x^2 + 4 * x + c)
  (h_increasing : ∀ x y, x ≤ y → -2 ≤ x → f x ≤ f y) :
  f 1 > f 0 ∧ f 0 > f (-2) :=
by
  sorry

end quadratic_func_inequality_l21_21616


namespace Liked_Both_Proof_l21_21448

section DessertProblem

variable (Total_Students Liked_Apple_Pie Liked_Chocolate_Cake Did_Not_Like_Either Liked_Both : ℕ)
variable (h1 : Total_Students = 50)
variable (h2 : Liked_Apple_Pie = 25)
variable (h3 : Liked_Chocolate_Cake = 20)
variable (h4 : Did_Not_Like_Either = 10)

theorem Liked_Both_Proof :
  Liked_Both = (Liked_Apple_Pie + Liked_Chocolate_Cake) - (Total_Students - Did_Not_Like_Either) :=
by
  sorry

end DessertProblem

end Liked_Both_Proof_l21_21448


namespace probability_at_least_one_female_is_five_sixths_l21_21485

-- Declare the total number of male and female students
def total_male_students := 6
def total_female_students := 4
def total_students := total_male_students + total_female_students
def selected_students := 3

-- Define the binomial coefficient function
def binomial_coefficient (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Total ways to select 3 students from 10 students
def total_ways_to_select_3 := binomial_coefficient total_students selected_students

-- Ways to select 3 male students from 6 male students
def ways_to_select_3_males := binomial_coefficient total_male_students selected_students

-- Probability of selecting at least one female student
def probability_of_at_least_one_female : ℚ := 1 - (ways_to_select_3_males / total_ways_to_select_3)

-- The theorem statement to be proved
theorem probability_at_least_one_female_is_five_sixths :
  probability_of_at_least_one_female = 5/6 := by
  sorry

end probability_at_least_one_female_is_five_sixths_l21_21485


namespace value_of_x_l21_21276

theorem value_of_x (x y z : ℤ) (h1 : x = y / 3) (h2 : y = z / 4) (h3 : z = 48) : x = 4 :=
by
  sorry

end value_of_x_l21_21276


namespace lines_parallel_to_skew_are_skew_or_intersect_l21_21827

-- Define skew lines conditions in space
def skew_lines (l1 l2 : ℝ → ℝ × ℝ × ℝ) : Prop :=
  ¬ (∀ t1 t2 : ℝ, l1 t1 = l2 t2) ∧ ¬ (∃ d : ℝ × ℝ × ℝ, ∀ t : ℝ, l1 t + d = l2 t)

-- Define parallel lines condition in space
def parallel_lines (m l : ℝ → ℝ × ℝ × ℝ) : Prop :=
  ∃ v : ℝ × ℝ × ℝ, ∀ t1 t2 : ℝ, m t1 = l t2 + v

-- Define the relationship to check between lines
def relationship (m1 m2 : ℝ → ℝ × ℝ × ℝ) : Prop :=
  (∃ t1 t2 : ℝ, m1 t1 = m2 t2) ∨ skew_lines m1 m2

-- The main theorem statement
theorem lines_parallel_to_skew_are_skew_or_intersect
  {l1 l2 m1 m2 : ℝ → ℝ × ℝ × ℝ}
  (h_skew: skew_lines l1 l2)
  (h_parallel_1: parallel_lines m1 l1)
  (h_parallel_2: parallel_lines m2 l2) :
  relationship m1 m2 :=
by
  sorry

end lines_parallel_to_skew_are_skew_or_intersect_l21_21827


namespace id_tags_divided_by_10_l21_21939

def uniqueIDTags (chars : List Char) (counts : Char → Nat) : Nat :=
  let permsWithoutRepetition := 
    Nat.factorial 7 / Nat.factorial (7 - 5)
  let repeatedCharTagCount := 10 * 10 * 6
  permsWithoutRepetition + repeatedCharTagCount

theorem id_tags_divided_by_10 :
  uniqueIDTags ['M', 'A', 'T', 'H', '2', '0', '3'] (fun c =>
    if c = 'M' then 1 else
    if c = 'A' then 1 else
    if c = 'T' then 1 else
    if c = 'H' then 1 else
    if c = '2' then 2 else
    if c = '0' then 1 else
    if c = '3' then 1 else 0) / 10 = 312 :=
by
  sorry

end id_tags_divided_by_10_l21_21939


namespace kim_fraction_of_shirts_given_l21_21256

open Nat

theorem kim_fraction_of_shirts_given (d : ℕ) (s_left : ℕ) (one_dozen := 12) 
  (original_shirts := 4 * one_dozen) 
  (given_shirts := original_shirts - s_left) 
  (fraction_given := given_shirts / original_shirts) 
  (hc1 : d = one_dozen) 
  (hc2 : s_left = 32) 
  : fraction_given = 1 / 3 := 
by 
  sorry

end kim_fraction_of_shirts_given_l21_21256


namespace print_papers_in_time_l21_21258

theorem print_papers_in_time :
  ∃ (n : ℕ), 35 * 15 * n = 500000 * 21 * n := by
  sorry

end print_papers_in_time_l21_21258


namespace compounding_frequency_l21_21546

variable (i : ℝ) (EAR : ℝ)

/-- Given the nominal annual rate (i = 6%) and the effective annual rate (EAR = 6.09%), 
    prove that the frequency of payment (n) is 4. -/
theorem compounding_frequency (h1 : i = 0.06) (h2 : EAR = 0.0609) : 
  ∃ n : ℕ, (1 + i / n)^n - 1 = EAR ∧ n = 4 := sorry

end compounding_frequency_l21_21546


namespace alligator_population_at_end_of_year_l21_21375

-- Define the conditions
def initial_population : ℕ := 4
def doubling_period_months : ℕ := 6
def total_months : ℕ := 12

-- Define the proof goal
theorem alligator_population_at_end_of_year (initial_population doubling_period_months total_months : ℕ)
  (h_init : initial_population = 4)
  (h_double : doubling_period_months = 6)
  (h_total : total_months = 12) :
  initial_population * (2 ^ (total_months / doubling_period_months)) = 16 := 
by
  sorry

end alligator_population_at_end_of_year_l21_21375


namespace range_of_m_l21_21397

-- Define the function and its properties
variable {f : ℝ → ℝ}
variable (increasing : ∀ x1 x2 : ℝ, x1 > x2 → f x1 > f x2)

theorem range_of_m (h: ∀ m : ℝ, f (2 * m) > f (-m + 9)) : 
  ∀ m : ℝ, m > 3 ↔ f (2 * m) > f (-m + 9) :=
by
  intros
  sorry

end range_of_m_l21_21397


namespace probability_of_green_l21_21151

open Classical

-- Define the total number of balls in each container
def balls_A := 12
def balls_B := 14
def balls_C := 12

-- Define the number of green balls in each container
def green_balls_A := 7
def green_balls_B := 6
def green_balls_C := 9

-- Define the probability of selecting each container
def prob_select_container := (1:ℚ) / 3

-- Define the probability of drawing a green ball from each container
def prob_green_A := green_balls_A / balls_A
def prob_green_B := green_balls_B / balls_B
def prob_green_C := green_balls_C / balls_C

-- Define the total probability of drawing a green ball
def total_prob_green := prob_select_container * prob_green_A +
                        prob_select_container * prob_green_B +
                        prob_select_container * prob_green_C

-- Create the proof statement
theorem probability_of_green : total_prob_green = 127 / 252 := 
by
  -- Skip the proof
  sorry

end probability_of_green_l21_21151


namespace expression_value_l21_21553

theorem expression_value (a b c d : ℝ) 
  (intersect1 : 4 = a * (2:ℝ)^2 + b * 2 + 1) 
  (intersect2 : 4 = (2:ℝ)^2 + c * 2 + d) 
  (hc : b + c = 1) : 
  4 * a + d = 1 := 
sorry

end expression_value_l21_21553


namespace christen_peeled_20_potatoes_l21_21450

-- Define the conditions and question
def homer_rate : ℕ := 3
def time_alone : ℕ := 4
def christen_rate : ℕ := 5
def total_potatoes : ℕ := 44

noncomputable def christen_potatoes : ℕ :=
  (total_potatoes - (homer_rate * time_alone)) / (homer_rate + christen_rate) * christen_rate

theorem christen_peeled_20_potatoes :
  christen_potatoes = 20 := by
  -- Proof steps would go here
  sorry

end christen_peeled_20_potatoes_l21_21450


namespace book_total_pages_l21_21221

-- Define the conditions given in the problem
def pages_per_night : ℕ := 12
def nights_to_finish : ℕ := 10

-- State that the total number of pages in the book is 120 given the conditions
theorem book_total_pages : (pages_per_night * nights_to_finish) = 120 :=
by sorry

end book_total_pages_l21_21221


namespace quadratic_rewrite_l21_21735

theorem quadratic_rewrite (x : ℝ) (b c : ℝ) : 
  (x^2 + 1560 * x + 2400 = (x + b)^2 + c) → 
  c / b = -300 :=
by
  sorry

end quadratic_rewrite_l21_21735


namespace sequence_problem_l21_21834

variable {n : ℕ}

-- We define the arithmetic sequence conditions
noncomputable def a_n : ℕ → ℕ
| n => 2 * n + 1

-- Conditions that the sequence must satisfy
axiom a_3_eq_7 : a_n 3 = 7
axiom a_5_a_7_eq_26 : a_n 5 + a_n 7 = 26

-- Define the sum of the sequence
noncomputable def S_n (n : ℕ) := n^2 + 2 * n

-- Define the sequence b_n
noncomputable def b_n (n : ℕ) := 1 / (a_n n ^ 2 - 1 : ℝ)

-- Define the sum of the sequence b_n
noncomputable def T_n (n : ℕ) := (n / (4 * (n + 1)) : ℝ)

-- The main theorem to prove
theorem sequence_problem :
  (a_n n = 2 * n + 1) ∧ (S_n n = n^2 + 2 * n) ∧ (T_n n = n / (4 * (n + 1))) :=
  sorry

end sequence_problem_l21_21834


namespace rectangle_diagonal_l21_21989

theorem rectangle_diagonal (l w : ℝ) (hl : l = 40) (hw : w = 40 * Real.sqrt 2) :
  Real.sqrt (l^2 + w^2) = 40 * Real.sqrt 3 :=
by
  rw [hl, hw]
  sorry

end rectangle_diagonal_l21_21989


namespace men_absent_l21_21540

theorem men_absent (original_men absent_men remaining_men : ℕ) (total_work : ℕ) 
  (h1 : original_men = 15) (h2 : total_work = original_men * 40) (h3 : 60 * remaining_men = total_work) : 
  remaining_men = original_men - absent_men → absent_men = 5 := 
by
  sorry

end men_absent_l21_21540


namespace A_wins_probability_is_3_over_4_l21_21493

def parity (n : ℕ) : Bool := n % 2 == 0

def number_of_dice_outcomes : ℕ := 36

def same_parity_outcome : ℕ := 18

def probability_A_wins : ℕ → ℕ → ℕ → ℚ
| total_outcomes, same_parity, different_parity =>
  (same_parity / total_outcomes : ℚ) * 1 + (different_parity / total_outcomes : ℚ) * (1 / 2)

theorem A_wins_probability_is_3_over_4 :
  probability_A_wins number_of_dice_outcomes same_parity_outcome (number_of_dice_outcomes - same_parity_outcome) = 3/4 :=
by
  sorry

end A_wins_probability_is_3_over_4_l21_21493


namespace boys_count_l21_21499

def total_pupils : ℕ := 485
def number_of_girls : ℕ := 232
def number_of_boys : ℕ := total_pupils - number_of_girls

theorem boys_count : number_of_boys = 253 := by
  -- The proof is omitted according to instruction
  sorry

end boys_count_l21_21499


namespace second_exponent_base_ends_in_1_l21_21674

theorem second_exponent_base_ends_in_1 
  (x : ℕ) 
  (h : ((1023 ^ 3923) + (x ^ 3921)) % 10 = 8) : 
  x % 10 = 1 := 
by sorry

end second_exponent_base_ends_in_1_l21_21674


namespace unique_sequence_l21_21295

/-- Define an infinite sequence of positive real numbers -/
def infinite_sequence (X : ℕ → ℝ) : Prop :=
  ∀ n, 0 < X n

/-- Define the recurrence relation for the sequence -/
def recurrence_relation (X : ℕ → ℝ) : Prop :=
  ∀ n, X (n + 2) = (1 / 2) * (1 / X (n + 1) + X n)

/-- Prove that the only infinite sequence satisfying the recurrence relation is the constant sequence 1 -/
theorem unique_sequence (X : ℕ → ℝ) (h_seq : infinite_sequence X) (h_recur : recurrence_relation X) :
  ∀ n, X n = 1 :=
by
  sorry

end unique_sequence_l21_21295


namespace cameron_gold_tokens_l21_21253

/-- Cameron starts with 90 red tokens and 60 blue tokens. 
  Booth 1 exchange: 3 red tokens for 1 gold token and 2 blue tokens.
  Booth 2 exchange: 2 blue tokens for 1 gold token and 1 red token.
  Cameron stops when fewer than 3 red tokens or 2 blue tokens remain.
  Prove that the number of gold tokens Cameron ends up with is 148.
-/
theorem cameron_gold_tokens :
  ∃ (x y : ℕ), 
    90 - 3 * x + y < 3 ∧
    60 + 2 * x - 2 * y < 2 ∧
    (x + y = 148) :=
  sorry

end cameron_gold_tokens_l21_21253


namespace least_number_to_add_l21_21985

theorem least_number_to_add (a b : ℤ) (d : ℤ) (h : a = 1054) (hb : b = 47) (hd : d = 27) :
  ∃ n : ℤ, (a + d) % b = 0 :=
by
  sorry

end least_number_to_add_l21_21985


namespace tracy_initial_candies_l21_21964

variable (x y : ℕ) (h1 : 2 ≤ y) (h2 : y ≤ 6)

theorem tracy_initial_candies :
  (x - (1/5 : ℚ) * x = (4/5 : ℚ) * x) ∧
  ((4/5 : ℚ) * x - (1/3 : ℚ) * (4/5 : ℚ) * x = (8/15 : ℚ) * x) ∧
  y - 10 * 2 + ((8/15 : ℚ) * x - 20) = 5 →
  x = 60 :=
by
  sorry

end tracy_initial_candies_l21_21964


namespace dogs_bunnies_ratio_l21_21795

theorem dogs_bunnies_ratio (total : ℕ) (dogs : ℕ) (bunnies : ℕ) (h1 : total = 375) (h2 : dogs = 75) (h3 : bunnies = total - dogs) : (75 / 75 : ℚ) / (300 / 75 : ℚ) = 1 / 4 := by
  sorry

end dogs_bunnies_ratio_l21_21795


namespace all_points_lie_on_line_l21_21183

theorem all_points_lie_on_line:
  ∀ (s : ℝ), s ≠ 0 → ∀ (x y : ℝ),
  x = (2 * s + 3) / s → y = (2 * s - 3) / s → x + y = 4 :=
by
  intros s hs x y hx hy
  sorry

end all_points_lie_on_line_l21_21183


namespace no_separation_sister_chromatids_first_meiotic_l21_21869

-- Definitions for the steps happening during the first meiotic division
def first_meiotic_division :=
  ∃ (prophase_I : Prop) (metaphase_I : Prop) (anaphase_I : Prop) (telophase_I : Prop),
    prophase_I ∧ metaphase_I ∧ anaphase_I ∧ telophase_I

def pairing_homologous_chromosomes (prophase_I : Prop) := prophase_I
def crossing_over (prophase_I : Prop) := prophase_I
def separation_homologous_chromosomes (anaphase_I : Prop) := anaphase_I
def separation_sister_chromatids (mitosis : Prop) (second_meiotic_division : Prop) :=
  mitosis ∨ second_meiotic_division

-- Theorem to prove that the separation of sister chromatids does not occur during the first meiotic division
theorem no_separation_sister_chromatids_first_meiotic
  (prophase_I metaphase_I anaphase_I telophase_I mitosis second_meiotic_division : Prop)
  (h1: first_meiotic_division)
  (h2 : pairing_homologous_chromosomes prophase_I)
  (h3 : crossing_over prophase_I)
  (h4 : separation_homologous_chromosomes anaphase_I)
  (h5 : separation_sister_chromatids mitosis second_meiotic_division) : 
  ¬ separation_sister_chromatids prophase_I anaphase_I :=
by
  sorry

end no_separation_sister_chromatids_first_meiotic_l21_21869


namespace relationship_between_a_b_c_l21_21430

noncomputable def a : ℝ := (1 / 2)^(1 / 3)
noncomputable def b : ℝ := Real.log 2 / Real.log (1 / 3)
noncomputable def c : ℝ := Real.log 3 / Real.log (1 / 2)

theorem relationship_between_a_b_c : c < b ∧ b < a :=
by
  sorry

end relationship_between_a_b_c_l21_21430


namespace bridge_length_l21_21525

theorem bridge_length
  (train_length : ℝ)
  (train_speed_km_hr : ℝ)
  (crossing_time_sec : ℝ)
  (train_speed_m_s : ℝ := train_speed_km_hr * 1000 / 3600)
  (total_distance : ℝ := train_speed_m_s * crossing_time_sec)
  (bridge_length : ℝ := total_distance - train_length)
  (train_length_val : train_length = 110)
  (train_speed_km_hr_val : train_speed_km_hr = 36)
  (crossing_time_sec_val : crossing_time_sec = 24.198064154867613) :
  bridge_length = 131.98064154867613 :=
by
  sorry

end bridge_length_l21_21525


namespace isosceles_triangle_perimeter_l21_21585

theorem isosceles_triangle_perimeter (a b c : ℕ) (h1 : a = 6) (h2 : b = 8) (h3 : a = c ∨ b = c) :
  a + b + c = 22 :=
by
  -- This part of the proof is simplified using the conditions
  sorry

end isosceles_triangle_perimeter_l21_21585


namespace solution_set_of_inequality_l21_21941

variable (a b x : ℝ)
variable (h1 : a < 0)

theorem solution_set_of_inequality (h : a * x + b < 0) : x > -b / a :=
sorry

end solution_set_of_inequality_l21_21941


namespace michael_digging_time_equals_700_l21_21750

-- Conditions defined
def digging_rate := 4
def father_depth := digging_rate * 400
def michael_depth := 2 * father_depth - 400
def time_for_michael := michael_depth / digging_rate

-- Statement to prove
theorem michael_digging_time_equals_700 : time_for_michael = 700 :=
by
  -- Here we would provide the proof steps, but we use sorry for now
  sorry

end michael_digging_time_equals_700_l21_21750


namespace mass_percentage_Cl_in_HClO2_is_51_78_l21_21758

noncomputable def molar_mass_H : ℝ := 1.01
noncomputable def molar_mass_Cl : ℝ := 35.45
noncomputable def molar_mass_O : ℝ := 16.00

noncomputable def molar_mass_HClO2 : ℝ :=
  molar_mass_H + molar_mass_Cl + 2 * molar_mass_O

noncomputable def mass_percentage_Cl_in_HClO2 : ℝ :=
  (molar_mass_Cl / molar_mass_HClO2) * 100

theorem mass_percentage_Cl_in_HClO2_is_51_78 :
  mass_percentage_Cl_in_HClO2 = 51.78 := 
sorry

end mass_percentage_Cl_in_HClO2_is_51_78_l21_21758


namespace lex_coins_total_l21_21145

def value_of_coins (dimes quarters : ℕ) : ℕ :=
  10 * dimes + 25 * quarters

def more_quarters_than_dimes (dimes quarters : ℕ) : Prop :=
  quarters > dimes

theorem lex_coins_total (dimes quarters : ℕ) (h : value_of_coins dimes quarters = 265) (h_more : more_quarters_than_dimes dimes quarters) : dimes + quarters = 13 :=
sorry

end lex_coins_total_l21_21145


namespace max_m_value_l21_21564

theorem max_m_value (a : ℚ) (m : ℚ) : (∀ x : ℤ, 0 < x ∧ x ≤ 50 → ¬ ∃ y : ℤ, y = m * x + 3) ∧ (1 / 2 < m) ∧ (m < a) → a = 26 / 51 :=
by sorry

end max_m_value_l21_21564


namespace calculate_value_of_expression_l21_21177

theorem calculate_value_of_expression :
  3.5 * 7.2 * (6.3 - 1.4) = 122.5 :=
  by
  sorry

end calculate_value_of_expression_l21_21177


namespace gcd_90_250_l21_21594

theorem gcd_90_250 : Nat.gcd 90 250 = 10 := by
  sorry

end gcd_90_250_l21_21594


namespace dot_product_equals_6_l21_21264

-- Define the vectors
def vec_a : ℝ × ℝ := (2, -1)
def vec_b : ℝ × ℝ := (-1, 2)

-- Define the scalar multiplication and addition
def scaled_added_vector : ℝ × ℝ := (2 * vec_a.1 + vec_b.1, 2 * vec_a.2 + vec_b.2)

-- Define the dot product
def dot_product : ℝ := scaled_added_vector.1 * vec_a.1 + scaled_added_vector.2 * vec_a.2

-- Assertion that the dot product is equal to 6
theorem dot_product_equals_6 : dot_product = 6 :=
by
  sorry

end dot_product_equals_6_l21_21264


namespace smallest_interesting_number_l21_21308

theorem smallest_interesting_number :
  ∃ n : ℕ, (∃ k1 : ℕ, 2 * n = k1^2) ∧ (∃ k2 : ℕ, 15 * n = k2^3) ∧ n = 1800 :=
by
  sorry

end smallest_interesting_number_l21_21308


namespace toothpaste_amount_in_tube_l21_21415

def dad_usage_per_brush : ℕ := 3
def mom_usage_per_brush : ℕ := 2
def kid_usage_per_brush : ℕ := 1
def brushes_per_day : ℕ := 3
def days : ℕ := 5

theorem toothpaste_amount_in_tube (dad_usage_per_brush mom_usage_per_brush kid_usage_per_brush brushes_per_day days : ℕ) : 
  dad_usage_per_brush * brushes_per_day * days + 
  mom_usage_per_brush * brushes_per_day * days + 
  (kid_usage_per_brush * brushes_per_day * days * 2) = 105 := 
  by sorry

end toothpaste_amount_in_tube_l21_21415


namespace intersection_correct_l21_21336

variable (x : ℝ)

def M : Set ℝ := { x | x^2 > 4 }
def N : Set ℝ := { x | x^2 - 3 * x ≤ 0 }
def NM_intersection : Set ℝ := { x | 2 < x ∧ x ≤ 3 }

theorem intersection_correct :
  {x | (M x) ∧ (N x)} = NM_intersection :=
sorry

end intersection_correct_l21_21336


namespace neg_of_forall_sin_ge_neg_one_l21_21152

open Real

theorem neg_of_forall_sin_ge_neg_one :
  (¬ (∀ x : ℝ, sin x ≥ -1)) ↔ (∃ x0 : ℝ, sin x0 < -1) := by
  sorry

end neg_of_forall_sin_ge_neg_one_l21_21152


namespace angle_same_terminal_side_315_l21_21229

theorem angle_same_terminal_side_315 (k : ℤ) : ∃ α, α = k * 360 + 315 ∧ α = -45 :=
by
  use -45
  sorry

end angle_same_terminal_side_315_l21_21229


namespace no_valid_six_digit_palindrome_years_l21_21924

noncomputable def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

noncomputable def is_six_digit_palindrome (n : ℕ) : Prop :=
  100000 ≤ n ∧ n ≤ 999999 ∧ is_palindrome n

noncomputable def is_four_digit_prime_palindrome (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ is_palindrome n ∧ is_prime n

noncomputable def is_two_digit_prime_palindrome (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ is_palindrome n ∧ is_prime n

theorem no_valid_six_digit_palindrome_years :
  ∀ N : ℕ, is_six_digit_palindrome N →
  ¬ ∃ (p q : ℕ), is_four_digit_prime_palindrome p ∧ is_two_digit_prime_palindrome q ∧ N = p * q := 
sorry

end no_valid_six_digit_palindrome_years_l21_21924


namespace david_moore_total_time_l21_21090

-- Given conditions
def david_work_rate := 1 / 12
def days_david_worked_alone := 6
def remaining_work_days_together := 3
def total_work := 1

-- Definition of total time taken for both to complete the job
def combined_total_time := 6

-- Proof problem statement in Lean
theorem david_moore_total_time :
  let d_work_done_alone := days_david_worked_alone * david_work_rate
  let remaining_work := total_work - d_work_done_alone
  let combined_work_rate := remaining_work / remaining_work_days_together
  let moore_work_rate := combined_work_rate - david_work_rate
  let new_combined_work_rate := david_work_rate + moore_work_rate
  total_work / new_combined_work_rate = combined_total_time := by
    sorry

end david_moore_total_time_l21_21090


namespace inequality_problem_l21_21379

variable (a b c : ℝ)

theorem inequality_problem (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b * c = 1) : 
  (a - 1 + 1 / b) * (b - 1 + 1 / c) * (c - 1 + 1 / a) ≤ 1 :=
sorry

end inequality_problem_l21_21379


namespace emma_final_amount_l21_21856

theorem emma_final_amount
  (initial_amount : ℕ)
  (furniture_cost : ℕ)
  (fraction_given_to_anna : ℚ)
  (amount_left : ℕ) :
  initial_amount = 2000 →
  furniture_cost = 400 →
  fraction_given_to_anna = 3 / 4 →
  amount_left = initial_amount - furniture_cost →
  amount_left - (fraction_given_to_anna * amount_left : ℚ) = 400 :=
by
  intros h_initial h_furniture h_fraction h_amount_left
  sorry

end emma_final_amount_l21_21856


namespace value_of_a_plus_c_l21_21050

-- Define the polynomials
def f (a b : ℝ) (x : ℝ) : ℝ := x^2 + a * x + b
def g (c d : ℝ) (x : ℝ) : ℝ := x^2 + c * x + d

-- Define the condition for the vertex of polynomial f being a root of g
def vertex_of_f_is_root_of_g (a b c d : ℝ) : Prop :=
  g c d (-a / 2) = 0

-- Define the condition for the vertex of polynomial g being a root of f
def vertex_of_g_is_root_of_f (a b c d : ℝ) : Prop :=
  f a b (-c / 2) = 0

-- Define the condition that both polynomials have the same minimum value
def same_minimum_value (a b c d : ℝ) : Prop :=
  f a b (-a / 2) = g c d (-c / 2)

-- Define the condition that the polynomials intersect at (100, -100)
def polynomials_intersect (a b c d : ℝ) : Prop :=
  f a b 100 = -100 ∧ g c d 100 = -100

-- Lean theorem statement for the problem
theorem value_of_a_plus_c (a b c d : ℝ) 
  (h1 : vertex_of_f_is_root_of_g a b c d)
  (h2 : vertex_of_g_is_root_of_f a b c d)
  (h3 : same_minimum_value a b c d)
  (h4 : polynomials_intersect a b c d) :
  a + c = -400 := 
sorry

end value_of_a_plus_c_l21_21050


namespace alyssa_cut_11_roses_l21_21103

theorem alyssa_cut_11_roses (initial_roses cut_roses final_roses : ℕ) 
  (h1 : initial_roses = 3) 
  (h2 : final_roses = 14) 
  (h3 : initial_roses + cut_roses = final_roses) : 
  cut_roses = 11 :=
by
  rw [h1, h2] at h3
  sorry

end alyssa_cut_11_roses_l21_21103


namespace man_arrived_earlier_l21_21614

-- Definitions of conditions as Lean variables
variables
  (usual_arrival_time_home : ℕ)  -- The usual arrival time at home
  (usual_drive_time : ℕ) -- The usual drive time for the wife to reach the station
  (early_arrival_difference : ℕ := 16) -- They arrived home 16 minutes earlier
  (man_walk_time : ℕ := 52) -- The man walked for 52 minutes

-- The proof statement
theorem man_arrived_earlier
  (usual_arrival_time_home : ℕ)
  (usual_drive_time : ℕ)
  (H : usual_arrival_time_home - man_walk_time <= usual_drive_time - early_arrival_difference)
  : man_walk_time = 52 :=
sorry

end man_arrived_earlier_l21_21614


namespace values_of_t_l21_21617

theorem values_of_t (x y z t : ℝ) 
  (h1 : 3 * x^2 + 3 * x * z + z^2 = 1)
  (h2 : 3 * y^2 + 3 * y * z + z^2 = 4)
  (h3 : x^2 - x * y + y^2 = t) : 
  t ≤ 10 :=
sorry

end values_of_t_l21_21617


namespace find_constants_a_b_l21_21141

def M : Matrix (Fin 2) (Fin 2) ℚ := ![
  ![3, 1],
  ![2, -2]
]

theorem find_constants_a_b :
  ∃ (a b : ℚ), (M⁻¹ = a • M + b • (1 : Matrix (Fin 2) (Fin 2) ℚ)) ∧
  a = 1/8 ∧ b = -1/8 :=
by
  sorry

end find_constants_a_b_l21_21141


namespace amount_spent_on_milk_l21_21451

-- Define conditions
def monthly_salary (S : ℝ) := 0.10 * S = 1800
def rent := 5000
def groceries := 4500
def education := 2500
def petrol := 2000
def miscellaneous := 700
def total_expenses (S : ℝ) := S - 1800
def known_expenses := rent + groceries + education + petrol + miscellaneous

-- Define the proof problem
theorem amount_spent_on_milk (S : ℝ) (milk : ℝ) :
  monthly_salary S →
  total_expenses S = known_expenses + milk →
  milk = 1500 :=
by
  sorry

end amount_spent_on_milk_l21_21451


namespace students_no_A_l21_21675

def total_students : Nat := 40
def students_A_chemistry : Nat := 10
def students_A_physics : Nat := 18
def students_A_both : Nat := 6

theorem students_no_A : (total_students - (students_A_chemistry + students_A_physics - students_A_both)) = 18 :=
by
  sorry

end students_no_A_l21_21675


namespace injective_functions_count_l21_21196

theorem injective_functions_count (m n : ℕ) (h_mn : m ≥ n) (h_n2 : n ≥ 2) :
  ∃ k, k = Nat.choose m n * (2^n - n - 1) :=
sorry

end injective_functions_count_l21_21196


namespace waiter_earned_in_tips_l21_21638

def waiter_customers := 7
def customers_didnt_tip := 5
def tip_per_customer := 3
def customers_tipped := waiter_customers - customers_didnt_tip
def total_earnings := customers_tipped * tip_per_customer

theorem waiter_earned_in_tips : total_earnings = 6 :=
by
  sorry

end waiter_earned_in_tips_l21_21638


namespace archer_expected_hits_l21_21903

noncomputable def binomial_expected_value (n : ℕ) (p : ℝ) : ℝ :=
  n * p

theorem archer_expected_hits :
  binomial_expected_value 10 0.9 = 9 :=
by
  sorry

end archer_expected_hits_l21_21903


namespace greatest_possible_perimeter_l21_21609

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem greatest_possible_perimeter :
  ∃ x : ℕ, 6 ≤ x ∧ x < 17 ∧ is_triangle x (2 * x) 17 ∧ (x + 2 * x + 17 = 65) := by
  sorry

end greatest_possible_perimeter_l21_21609


namespace remainder_b96_div_50_l21_21917

theorem remainder_b96_div_50 (b : ℕ → ℕ) (h : ∀ n, b n = 7^n + 9^n) : b 96 % 50 = 2 :=
by
  -- The proof is omitted.
  sorry

end remainder_b96_div_50_l21_21917


namespace binary_to_decimal_eq_l21_21589

theorem binary_to_decimal_eq :
  (1 * 2^7 + 1 * 2^6 + 0 * 2^5 + 0 * 2^4 + 1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0) = 205 :=
by
  sorry

end binary_to_decimal_eq_l21_21589


namespace boys_more_than_girls_l21_21631

-- Definitions of the conditions
def total_students : ℕ := 100
def boy_ratio : ℕ := 3
def girl_ratio : ℕ := 2

-- Statement of the problem
theorem boys_more_than_girls :
  (total_students * boy_ratio) / (boy_ratio + girl_ratio) - (total_students * girl_ratio) / (boy_ratio + girl_ratio) = 20 :=
by
  sorry

end boys_more_than_girls_l21_21631


namespace judgments_correct_l21_21463

variables {l m : Line} (a : Plane)

def is_perpendicular (l : Line) (a : Plane) : Prop := -- Definition of perpendicularity between a line and a plane
sorry

def is_parallel (l m : Line) : Prop := -- Definition of parallel lines
sorry

def is_contained_in (m : Line) (a : Plane) : Prop := -- Definition of a line contained in a plane
sorry

theorem judgments_correct 
  (hl : is_perpendicular l a)
  (hm : l ≠ m) :
  (∀ m, is_perpendicular m l → is_parallel m a) ∧ 
  (is_perpendicular m a → is_parallel m l) ∧
  (is_contained_in m a → is_perpendicular m l) ∧
  (is_parallel m l → is_perpendicular m a) :=
sorry

end judgments_correct_l21_21463


namespace plane_boat_ratio_l21_21298

theorem plane_boat_ratio (P B : ℕ) (h1 : P > B) (h2 : B ≤ 2) (h3 : P + B = 10) : P = 8 ∧ B = 2 ∧ P / B = 4 := by
  sorry

end plane_boat_ratio_l21_21298


namespace trapezium_area_l21_21004

theorem trapezium_area (a b area h : ℝ) (h1 : a = 20) (h2 : b = 15) (h3 : area = 245) :
  area = 1 / 2 * (a + b) * h → h = 14 :=
by
  sorry

end trapezium_area_l21_21004


namespace f_inequality_l21_21019

-- Definition of odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- f is an odd function
variable {f : ℝ → ℝ}
variable (h1 : is_odd_function f)

-- f has a period of 4
variable (h2 : ∀ x, f (x + 4) = f x)

-- f is monotonically increasing on [0, 2)
variable (h3 : ∀ x y, 0 ≤ x → x < y → y < 2 → f x < f y)

theorem f_inequality : f 3 < 0 ∧ 0 < f 1 :=
by 
  -- Place proof here
  sorry

end f_inequality_l21_21019


namespace determine_c_l21_21866

theorem determine_c (c d : ℝ) (hc : c < 0) (hd : d > 0) (hamp : ∀ x, y = c * Real.cos (d * x) → |y| ≤ 3) :
  c = -3 :=
sorry

end determine_c_l21_21866


namespace Ryan_spit_distance_correct_l21_21711

-- Definitions of given conditions
def Billy_spit_distance : ℝ := 30
def Madison_spit_distance : ℝ := Billy_spit_distance * 1.20
def Ryan_spit_distance : ℝ := Madison_spit_distance * 0.50

-- Goal statement
theorem Ryan_spit_distance_correct : Ryan_spit_distance = 18 := by
  -- proof would go here
  sorry

end Ryan_spit_distance_correct_l21_21711


namespace range_of_x_l21_21953

-- Define the even and increasing properties of the function
def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def is_increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, 0 ≤ x → x ≤ y → f x ≤ f y

-- The main theorem to be proven
theorem range_of_x (f : ℝ → ℝ) (h_even : is_even f) (h_incr : is_increasing_on_nonneg f) 
  (h_cond : ∀ x : ℝ, f (x - 1) < f (2 - x)) :
  ∀ x : ℝ, x < 3 / 2 :=
by
  sorry

end range_of_x_l21_21953


namespace circle_equation_is_correct_l21_21993

def center : Int × Int := (-3, 4)
def radius : Int := 3
def circle_standard_equation (x y : Int) : Int :=
  (x + 3)^2 + (y - 4)^2

theorem circle_equation_is_correct :
  circle_standard_equation x y = 9 :=
sorry

end circle_equation_is_correct_l21_21993


namespace base_conversion_sum_l21_21932

-- Definition of conversion from base 13 to base 10
def base13_to_base10 (n : ℕ) : ℕ :=
  3 * (13^2) + 4 * (13^1) + 5 * (13^0)

-- Definition of conversion from base 14 to base 10 where C = 12 and D = 13
def base14_to_base10 (m : ℕ) : ℕ :=
  4 * (14^2) + 12 * (14^1) + 13 * (14^0)

theorem base_conversion_sum :
  base13_to_base10 345 + base14_to_base10 (4 * 14^2 + 12 * 14 + 13) = 1529 := 
by
  sorry -- proof to be provided

end base_conversion_sum_l21_21932


namespace rook_placement_5x5_l21_21665

theorem rook_placement_5x5 :
  ∀ (board : Fin 5 → Fin 5) (distinct : Function.Injective board),
  ∃ (ways : Nat), ways = 120 := by
  sorry

end rook_placement_5x5_l21_21665


namespace sale_price_relative_to_original_l21_21732

variable (x : ℝ)

def increased_price (x : ℝ) := 1.30 * x
def sale_price (increased_price : ℝ) := 0.90 * increased_price

theorem sale_price_relative_to_original (x : ℝ) :
  sale_price (increased_price x) = 1.17 * x :=
by
  sorry

end sale_price_relative_to_original_l21_21732


namespace geometric_seq_a3_l21_21829

theorem geometric_seq_a3 (a : ℕ → ℝ) (r : ℝ)
  (h1 : a 6 = a 3 * r^3)
  (h2 : a 9 = a 3 * r^6)
  (h3 : a 6 = 6)
  (h4 : a 9 = 9) : 
  a 3 = 4 := 
sorry

end geometric_seq_a3_l21_21829


namespace min_abs_expr1_min_abs_expr2_l21_21892

theorem min_abs_expr1 (x : ℝ) : |x - 4| + |x + 2| ≥ 6 := sorry

theorem min_abs_expr2 (x : ℝ) : |(5 / 6) * x - 1| + |(1 / 2) * x - 1| + |(2 / 3) * x - 1| ≥ 1 / 2 := sorry

end min_abs_expr1_min_abs_expr2_l21_21892


namespace jerry_pick_up_trays_l21_21890

theorem jerry_pick_up_trays : 
  ∀ (trays_per_trip trips trays_from_second total),
  trays_per_trip = 8 →
  trips = 2 →
  trays_from_second = 7 →
  total = (trays_per_trip * trips) →
  (total - trays_from_second) = 9 :=
by
  intros trays_per_trip trips trays_from_second total
  intro h1 h2 h3 h4
  sorry

end jerry_pick_up_trays_l21_21890


namespace sum_of_roots_eq_4140_l21_21403

open Complex

noncomputable def sum_of_roots : ℝ :=
  let θ0 := 270 / 5;
  let θ1 := (270 + 360) / 5;
  let θ2 := (270 + 2 * 360) / 5;
  let θ3 := (270 + 3 * 360) / 5;
  let θ4 := (270 + 4 * 360) / 5;
  θ0 + θ1 + θ2 + θ3 + θ4

theorem sum_of_roots_eq_4140 : sum_of_roots = 4140 := by
  sorry

end sum_of_roots_eq_4140_l21_21403


namespace triangle_perimeter_l21_21968

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

noncomputable def perimeter (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  distance p1 p2 + distance p1 p3 + distance p2 p3

theorem triangle_perimeter :
  let p1 := (1, 4)
  let p2 := (-7, 0)
  let p3 := (1, 0)
  perimeter p1 p2 p3 = 4 * Real.sqrt 5 + 12 :=
by
  sorry

end triangle_perimeter_l21_21968


namespace chocolates_left_l21_21501

-- Definitions based on the conditions
def initially_bought := 3
def gave_away := 2
def additionally_bought := 3

-- Theorem statement to prove
theorem chocolates_left : initially_bought - gave_away + additionally_bought = 4 := by
  -- Proof skipped
  sorry

end chocolates_left_l21_21501


namespace men_per_table_l21_21948

theorem men_per_table 
  (num_tables : ℕ) 
  (women_per_table : ℕ) 
  (total_customers : ℕ) 
  (h1 : num_tables = 9) 
  (h2 : women_per_table = 7) 
  (h3 : total_customers = 90)
  : (total_customers - num_tables * women_per_table) / num_tables = 3 :=
by
  sorry

end men_per_table_l21_21948


namespace work_done_l21_21692

noncomputable def F (x : ℝ) : ℝ := 3 * x^2 - 2 * x + 3

theorem work_done (W : ℝ) (h : W = ∫ x in (1:ℝ)..(5:ℝ), F x) : W = 112 :=
by sorry

end work_done_l21_21692


namespace total_students_l21_21417

noncomputable def total_students_in_gym (F : ℕ) (T : ℕ) : Prop :=
  T = 26

theorem total_students (F T : ℕ) (h1 : 4 = T - F) (h2 : F / (F + 4) = 11 / 13) : total_students_in_gym F T :=
by sorry

end total_students_l21_21417


namespace trader_bags_correct_l21_21039

-- Definitions according to given conditions
def initial_bags := 55
def sold_bags := 23
def restocked_bags := 132

-- Theorem that encapsulates the problem's question and the proven answer
theorem trader_bags_correct :
  (initial_bags - sold_bags + restocked_bags) = 164 :=
by
  sorry

end trader_bags_correct_l21_21039


namespace find_overhead_expenses_l21_21960

noncomputable def overhead_expenses : ℝ := 35.29411764705882 / (1 + 0.1764705882352942)

theorem find_overhead_expenses (cost_price selling_price profit_percent : ℝ) (h_cp : cost_price = 225) (h_sp : selling_price = 300) (h_pp : profit_percent = 0.1764705882352942) :
  overhead_expenses = 30 :=
by
  sorry

end find_overhead_expenses_l21_21960


namespace part1_decreasing_on_pos_part2_t_range_l21_21997

noncomputable def f (x : ℝ) : ℝ := -x + 2 / x

theorem part1_decreasing_on_pos (x1 x2 : ℝ) (h1 : 0 < x1) (h2 : 0 < x2) (h3 : x1 < x2) : 
  f x1 > f x2 := by sorry

theorem part2_t_range (t : ℝ) (ht : ∀ x : ℝ, 1 ≤ x → f x ≤ (1 + t * x) / x) : 
  0 ≤ t := by sorry

end part1_decreasing_on_pos_part2_t_range_l21_21997


namespace proof_problem_l21_21994

variables (a b c : Line) (alpha beta gamma : Plane)

-- Define perpendicular relationship between line and plane
def perp_line_plane (l : Line) (p : Plane) : Prop := sorry

-- Define parallel relationship between lines
def parallel_lines (l1 l2 : Line) : Prop := sorry

-- Define parallel relationship between planes
def parallel_planes (p1 p2 : Plane) : Prop := sorry

-- Main theorem statement
theorem proof_problem 
  (h1 : perp_line_plane a alpha) 
  (h2 : perp_line_plane b beta) 
  (h3 : parallel_planes alpha beta) : 
  parallel_lines a b :=
sorry

end proof_problem_l21_21994


namespace local_minimum_f_is_1_maximum_local_minimum_g_is_1_l21_21363

noncomputable def f (x : ℝ) : ℝ := Real.exp (x - 1) - Real.log x

def local_minimum_value_f := 1

theorem local_minimum_f_is_1 : 
  ∃ x0 : ℝ, x0 > 0 ∧ (∀ x > 0, f x0 ≤ f x) ∧ f x0 = local_minimum_value_f :=
sorry

noncomputable def g (a x : ℝ) : ℝ := f x - a * (x - 1)

def maximum_value_local_minimum_g := 1

theorem maximum_local_minimum_g_is_1 :
  ∃ a x0 : ℝ, a = 0 ∧ x0 > 0 ∧ (∀ x > 0, g a x0 ≤ g a x) ∧ g a x0 = maximum_value_local_minimum_g :=
sorry

end local_minimum_f_is_1_maximum_local_minimum_g_is_1_l21_21363


namespace calculate_value_l21_21828

theorem calculate_value : 12 * ((1/3 : ℝ) + (1/4) - (1/12))⁻¹ = 24 :=
by
  sorry

end calculate_value_l21_21828


namespace remaining_watermelons_l21_21708

def initial_watermelons : ℕ := 4
def eaten_watermelons : ℕ := 3

theorem remaining_watermelons : initial_watermelons - eaten_watermelons = 1 :=
by sorry

end remaining_watermelons_l21_21708


namespace greatest_multiple_of_4_l21_21296

theorem greatest_multiple_of_4 (x : ℕ) (h1 : x % 4 = 0) (h2 : x > 0) (h3 : x^3 < 500) : x ≤ 4 :=
by sorry

end greatest_multiple_of_4_l21_21296


namespace depth_of_right_frustum_l21_21446

-- Definitions
def volume_cm3 := 190000 -- Volume in cubic centimeters (190 liters)
def top_edge := 60 -- Length of the top edge in centimeters
def bottom_edge := 40 -- Length of the bottom edge in centimeters
def expected_depth := 75 -- Expected depth in centimeters

-- The following is the statement of the proof
theorem depth_of_right_frustum 
  (V : ℝ) (A1 A2 : ℝ) (h : ℝ)
  (hV : V = 190 * 1000)
  (hA1 : A1 = top_edge * top_edge)
  (hA2 : A2 = bottom_edge * bottom_edge)
  (h_avg : 2 * A1 / (top_edge + bottom_edge) = 2 * A2 / (top_edge + bottom_edge))
  : h = expected_depth := 
sorry

end depth_of_right_frustum_l21_21446


namespace complement_intersection_l21_21970

open Set

variable (A B U : Set ℕ) 

theorem complement_intersection (hA : A = {1, 2, 3}) (hB : B = {2, 3, 4, 5}) (hU : U = A ∪ B) :
  (U \ A) ∩ B = {4, 5} :=
by sorry

end complement_intersection_l21_21970


namespace find_f_correct_l21_21532

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_con1 : ∀ x : ℝ, 2 * f x + f (-x) = 2 * x

theorem find_f_correct : ∀ x : ℝ, f x = 2 * x :=
by
  sorry

end find_f_correct_l21_21532


namespace find_a5_plus_a7_l21_21483

variable {a : ℕ → ℕ}

-- Assume a is a geometric sequence with common ratio q and first term a1.
def geometric_sequence (a : ℕ → ℕ) (a_1 : ℕ) (q : ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a_1 * q ^ n

-- Given conditions of the problem:
def conditions (a : ℕ → ℕ) : Prop :=
  a 2 + a 4 = 20 ∧ a 3 + a 5 = 40

-- The objective is to prove a_5 + a_7 = 160
theorem find_a5_plus_a7 (a : ℕ → ℕ) (a_1 q : ℕ) (h_geo : geometric_sequence a a_1 q) (h_cond : conditions a) : a 5 + a 7 = 160 :=
  sorry

end find_a5_plus_a7_l21_21483


namespace rationalize_denominator_l21_21784

def cbrt (x : ℝ) : ℝ := x^(1/3)

theorem rationalize_denominator :
  let a := cbrt 2
  let b := cbrt 27
  b = 3 -> ( 1 / (a + b)) = (cbrt 4 / (2 + 3 * cbrt 4))
:= by
  intro a
  intro b
  sorry

end rationalize_denominator_l21_21784


namespace factor_quadratic_l21_21058

theorem factor_quadratic (x : ℝ) : 
  (x^2 + 6 * x + 9 - 16 * x^4) = (-4 * x^2 + 2 * x + 3) * (4 * x^2 + 2 * x + 3) := 
by 
  sorry

end factor_quadratic_l21_21058


namespace find_unique_f_l21_21116

theorem find_unique_f (f : ℝ → ℝ) (h : ∀ x y z : ℝ, f (x * y) + f (x * z) ≥ f (x) * f (y * z) + 1) : 
    ∀ x : ℝ, f x = 1 :=
by
  sorry

end find_unique_f_l21_21116


namespace min_value_expression_l21_21324

theorem min_value_expression (x y z : ℝ) : ∃ v, v = 0 ∧ ∀ x y z : ℝ, x^2 + 2 * x * y + 3 * y^2 + 2 * x * z + 3 * z^2 ≥ v := 
by 
  use 0
  sorry

end min_value_expression_l21_21324


namespace theater_revenue_l21_21140

theorem theater_revenue 
  (seats : ℕ)
  (capacity_percentage : ℝ)
  (ticket_price : ℝ)
  (days : ℕ)
  (H1 : seats = 400)
  (H2 : capacity_percentage = 0.8)
  (H3 : ticket_price = 30)
  (H4 : days = 3)
  : (seats * capacity_percentage * ticket_price * days = 28800) :=
by
  sorry

end theater_revenue_l21_21140


namespace quadratic_factored_b_l21_21099

theorem quadratic_factored_b (b : ℤ) : 
  (∃ (m n p q : ℤ), 15 * x^2 + b * x + 30 = (m * x + n) * (p * x + q) ∧ m * p = 15 ∧ n * q = 30 ∧ m * q + n * p = b) ↔ b = 43 :=
by {
  sorry
}

end quadratic_factored_b_l21_21099


namespace proof_problem_l21_21439

def U : Set ℤ := {x | x^2 - x - 12 ≤ 0}
def A : Set ℤ := {-2, -1, 3}
def B : Set ℤ := {0, 1, 3, 4}

theorem proof_problem : (U \ A) ∩ B = {0, 1, 4} := 
by sorry

end proof_problem_l21_21439


namespace range_of_a_l21_21673

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0) ∧ (∃ x0 : ℝ, x0^2 + 2 * a * x0 + 2 - a = 0) ↔ a ≤ -2 ∨ a = 1 :=
by sorry

end range_of_a_l21_21673


namespace greatest_divisor_lemma_l21_21780

theorem greatest_divisor_lemma : ∃ (d : ℕ), d = Nat.gcd 1636 1852 ∧ d = 4 := by
  sorry

end greatest_divisor_lemma_l21_21780


namespace evaluate_expression_l21_21648

theorem evaluate_expression : 
  (1 / (2 - (1 / (2 - (1 / (2 - (1 / 3))))))) = 5 / 7 :=
by
  sorry

end evaluate_expression_l21_21648


namespace contest_sum_l21_21576

theorem contest_sum 
(A B C D E : ℕ) 
(h_sum : A + B + C + D + E = 35)
(h_right_E : B + C + D + E = 13)
(h_right_D : C + D + E = 31)
(h_right_A : B + C + D + E = 21)
(h_right_C : C + D + E = 7)
: D + B = 11 :=
sorry

end contest_sum_l21_21576


namespace find_positive_integers_l21_21566

theorem find_positive_integers (a b c : ℕ) (ha : a ≥ b) (hb : b ≥ c) :
  (∃ n₁ : ℕ, a^2 + 3 * b = n₁^2) ∧ 
  (∃ n₂ : ℕ, b^2 + 3 * c = n₂^2) ∧ 
  (∃ n₃ : ℕ, c^2 + 3 * a = n₃^2) →
  (a = 1 ∧ b = 1 ∧ c = 1) ∨ (a = 37 ∧ b = 25 ∧ c = 17) :=
by
  sorry

end find_positive_integers_l21_21566


namespace solve_fraction_equation_l21_21511

theorem solve_fraction_equation (x : ℝ) (h : (x + 6) / (x - 3) = 4) : x = 6 :=
sorry

end solve_fraction_equation_l21_21511


namespace sunday_to_saturday_ratio_l21_21391

theorem sunday_to_saturday_ratio : 
  ∀ (sold_friday sold_saturday sold_sunday total_sold : ℕ),
  sold_friday = 40 →
  sold_saturday = (2 * sold_friday - 10) →
  total_sold = 145 →
  total_sold = sold_friday + sold_saturday + sold_sunday →
  (sold_sunday : ℚ) / (sold_saturday : ℚ) = 1 / 2 :=
by
  intro sold_friday sold_saturday sold_sunday total_sold
  intros h_friday h_saturday h_total h_sum
  sorry

end sunday_to_saturday_ratio_l21_21391


namespace calc_m_l21_21777

theorem calc_m (m : ℤ) (h : (64 : ℝ)^(1 / 3) = 2^m) : m = 2 :=
sorry

end calc_m_l21_21777


namespace min_points_game_12_l21_21975

noncomputable def player_scores := (18, 22, 9, 29)

def avg_after_eleven_games (scores: ℕ × ℕ × ℕ × ℕ) := 
  let s₁ := 78 -- Sum of the points in 8th, 9th, 10th, 11th games
  (s₁: ℕ) / 4

def points_twelve_game_cond (n: ℕ) : Prop :=
  let total_points := 78 + n
  total_points > (20 * 12)

theorem min_points_game_12 (points_in_first_7_games: ℕ) (score_12th_game: ℕ) 
  (H1: avg_after_eleven_games player_scores > (points_in_first_7_games / 7)) 
  (H2: points_twelve_game_cond score_12th_game):
  score_12th_game = 30 := by
  sorry

end min_points_game_12_l21_21975


namespace depth_multiple_of_rons_height_l21_21211

theorem depth_multiple_of_rons_height (h d : ℕ) (Ron_height : h = 13) (water_depth : d = 208) : d = 16 * h := by
  sorry

end depth_multiple_of_rons_height_l21_21211


namespace eval_power_imaginary_unit_l21_21508

noncomputable def i : ℂ := Complex.I

theorem eval_power_imaginary_unit :
  i^20 + i^39 = 1 - i := by
  -- Skipping the proof itself, indicating it with "sorry"
  sorry

end eval_power_imaginary_unit_l21_21508


namespace mixed_doubles_selection_l21_21338

-- Given conditions
def num_male_players : ℕ := 5
def num_female_players : ℕ := 4

-- The statement to show the number of different ways to select two players is 20
theorem mixed_doubles_selection : (num_male_players * num_female_players) = 20 := by
  -- Proof to be filled in
  sorry

end mixed_doubles_selection_l21_21338


namespace polynomial_remainder_l21_21126

noncomputable def p (x : ℝ) : ℝ := 3 * x ^ 5 - 2 * x ^ 3 + 5 * x - 8
noncomputable def d (x : ℝ) : ℝ := x ^ 2 - 2 * x + 1
noncomputable def rem (x : ℝ) : ℝ := 14 * x - 14

theorem polynomial_remainder :
  ∀ x : ℝ, p x % d x = rem x := 
by
  sorry

end polynomial_remainder_l21_21126


namespace min_pounds_of_beans_l21_21881

theorem min_pounds_of_beans : 
  ∃ (b : ℕ), (∀ (r : ℝ), (r ≥ 8 + b / 3 ∧ r ≤ 3 * b) → b ≥ 3) :=
sorry

end min_pounds_of_beans_l21_21881


namespace arithmetic_seq_geom_seq_l21_21959

theorem arithmetic_seq_geom_seq {a : ℕ → ℝ} 
  (h1 : ∀ n, 0 < a n)
  (h2 : a 2 + a 3 + a 4 = 15)
  (h3 : (a 1 + 2) * (a 6 + 16) = (a 3 + 4) ^ 2) :
  a 10 = 19 :=
sorry

end arithmetic_seq_geom_seq_l21_21959


namespace condition1_not_sufficient_nor_necessary_condition2_necessary_l21_21698

variable (x y : ℝ)

-- ① Neither sufficient nor necessary
theorem condition1_not_sufficient_nor_necessary (h1 : x ≠ 1 ∧ y ≠ 2) : ¬ ((x ≠ 1 ∧ y ≠ 2) → x + y ≠ 3) ∧ ¬ (x + y ≠ 3 → x ≠ 1 ∧ y ≠ 2) := sorry

-- ② Necessary condition
theorem condition2_necessary (h2 : x ≠ 1 ∨ y ≠ 2) : x + y ≠ 3 → (x ≠ 1 ∨ y ≠ 2) := sorry

end condition1_not_sufficient_nor_necessary_condition2_necessary_l21_21698


namespace part1_solution_part2_solution_l21_21429

variables (x y m : ℤ)

-- Given the system of equations
def system_of_equations (x y m : ℤ) : Prop :=
  (2 * x - y = m) ∧ (3 * x + 2 * y = m + 7)

-- Part (1) m = 0, find x = 1, y = 2
theorem part1_solution : system_of_equations x y 0 → x = 1 ∧ y = 2 :=
sorry

-- Part (2) point A(-2,3) in the second quadrant with distances 3 and 2, find m = -7
def is_in_second_quadrant (x y : ℤ) : Prop :=
  x < 0 ∧ y > 0

def distance_to_axes (x y dx dy : ℤ) : Prop :=
  y = dy ∧ x = -dx

theorem part2_solution : is_in_second_quadrant x y →
  distance_to_axes x y 2 3 →
  system_of_equations x y m →
  m = -7 :=
sorry

end part1_solution_part2_solution_l21_21429


namespace elena_snow_removal_l21_21175

theorem elena_snow_removal :
  ∀ (length width depth : ℝ) (compaction_factor : ℝ), 
  length = 30 ∧ width = 3 ∧ depth = 0.75 ∧ compaction_factor = 0.90 → 
  (length * width * depth * compaction_factor = 60.75) :=
by
  intros length width depth compaction_factor h
  obtain ⟨length_eq, width_eq, depth_eq, compaction_factor_eq⟩ := h
  -- Proof steps go here
  sorry

end elena_snow_removal_l21_21175


namespace total_points_l21_21657

def jon_points (sam_points : ℕ) : ℕ := 2 * sam_points + 3
def sam_points (alex_points : ℕ) : ℕ := alex_points / 2
def jack_points (jon_points : ℕ) : ℕ := jon_points + 5
def tom_points (jon_points jack_points : ℕ) : ℕ := jon_points + jack_points - 4
def alex_points : ℕ := 18

theorem total_points : jon_points (sam_points alex_points) + 
                       jack_points (jon_points (sam_points alex_points)) + 
                       tom_points (jon_points (sam_points alex_points)) 
                       (jack_points (jon_points (sam_points alex_points))) + 
                       sam_points alex_points + 
                       alex_points = 117 :=
by sorry

end total_points_l21_21657


namespace right_triangle_345_l21_21680

theorem right_triangle_345 :
  ∃ (a b c : ℝ), a = 3 ∧ b = 4 ∧ c = 5 ∧ a^2 + b^2 = c^2 :=
by {
  -- Here, we should construct the proof later
  sorry
}

end right_triangle_345_l21_21680


namespace cylinder_surface_area_l21_21245

theorem cylinder_surface_area (side : ℝ) (h : ℝ) (r : ℝ) : 
  side = 2 ∧ h = side ∧ r = side → 
  (2 * Real.pi * r^2 + 2 * Real.pi * r * h) = 16 * Real.pi := 
by
  intro h
  sorry

end cylinder_surface_area_l21_21245


namespace root_in_interval_implies_a_in_range_l21_21630

theorem root_in_interval_implies_a_in_range {a : ℝ} (h : ∃ x : ℝ, x ≤ 1 ∧ 2^x - a^2 - a = 0) : 0 < a ∧ a ≤ 1 := sorry

end root_in_interval_implies_a_in_range_l21_21630


namespace length_ab_is_constant_l21_21599

noncomputable def length_AB_constant (p : ℝ) (hp : p > 0) : Prop :=
  let parabola := { P : ℝ × ℝ | P.1 ^ 2 = 2 * p * P.2 }
  let line := { P : ℝ × ℝ | P.2 = P.1 + p / 2 }
  (∃ A B : ℝ × ℝ, A ∈ parabola ∧ B ∈ parabola ∧ A ∈ line ∧ B ∈ line ∧ 
    dist A B = 4 * p)

theorem length_ab_is_constant (p : ℝ) (hp : p > 0) : length_AB_constant p hp :=
by {
  sorry
}

end length_ab_is_constant_l21_21599


namespace initial_deck_card_count_l21_21230

theorem initial_deck_card_count (r n : ℕ) (h1 : n = 2 * r) (h2 : n + 4 = 3 * r) : r + n = 12 := by
  sorry

end initial_deck_card_count_l21_21230


namespace log_inequality_region_l21_21452

theorem log_inequality_region (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hx1 : x ≠ 1) (hx2 : x ≠ y) :
  (0 < x ∧ x < 1 ∧ 0 < y ∧ y < x) 
  ∨ (1 < x ∧ y > x) ↔ (Real.log y / Real.log x ≥ Real.log (x * y) / Real.log (x / y)) :=
  sorry

end log_inequality_region_l21_21452


namespace blue_part_length_l21_21954

variable (total_length : ℝ) (black_part white_part blue_part : ℝ)

-- Conditions
axiom h1 : black_part = 1 / 8 * total_length
axiom h2 : white_part = 1 / 2 * (total_length - black_part)
axiom h3 : total_length = 8

theorem blue_part_length : blue_part = total_length - black_part - white_part :=
by
  sorry

end blue_part_length_l21_21954


namespace solve_for_x_l21_21639

theorem solve_for_x (x : ℝ) : (|2 * x + 8| = 4 - 3 * x) → x = -4 / 5 :=
  sorry

end solve_for_x_l21_21639


namespace range_of_a_l21_21299

noncomputable def f (a : ℝ) (x : ℝ) := Real.exp x - (x^2 / 2) - a * x - 1

theorem range_of_a (x : ℝ) (a : ℝ) (h : 1 ≤ x) : (0 ≤ f a x) → (a ≤ Real.exp 1 - 3 / 2) :=
by
  sorry

end range_of_a_l21_21299


namespace white_area_correct_l21_21902

def total_sign_area : ℕ := 8 * 20
def black_area_C : ℕ := 8 * 1 + 2 * (1 * 3)
def black_area_A : ℕ := 2 * (8 * 1) + 2 * (1 * 2)
def black_area_F : ℕ := 8 * 1 + 2 * (1 * 4)
def black_area_E : ℕ := 3 * (1 * 4)

def total_black_area : ℕ := black_area_C + black_area_A + black_area_F + black_area_E
def white_area : ℕ := total_sign_area - total_black_area

theorem white_area_correct : white_area = 98 :=
  by 
    sorry -- State the theorem without providing the proof.

end white_area_correct_l21_21902


namespace find_abc_l21_21457

variables {a b c : ℕ}

theorem find_abc (h1 : 1 < a) (h2 : a < b) (h3 : b < c) (h4 : abc ∣ ((a * b - 1) * (b * c - 1) * (c * a - 1))) : a = 2 ∧ b = 3 ∧ c = 5 :=
by {
    sorry
}

end find_abc_l21_21457


namespace find_x_l21_21860

theorem find_x (x : ℝ) (i : ℂ) (h : i * i = -1) (h1 : (1 - i) * (Complex.ofReal x + i) = 1 + i) : x = 0 :=
by sorry

end find_x_l21_21860


namespace m_minus_n_eq_2_l21_21979

theorem m_minus_n_eq_2 (m n : ℕ) (h1 : ∃ x : ℕ, m = 101 * x) (h2 : ∃ y : ℕ, n = 63 * y) (h3 : m + n = 2018) : m - n = 2 :=
sorry

end m_minus_n_eq_2_l21_21979


namespace ellipse_is_correct_l21_21775

-- Define the hyperbola equation
def hyperbola_eq (x y : ℝ) : Prop :=
  (x^2 / 4) - (y^2 / 12) = -1

-- Define the ellipse equation
def ellipse_eq (x y : ℝ) : Prop :=
  (x^2 / 4) + (y^2 / 16) = 1

-- Define the conditions
def ellipse_focus_vertex_of_hyperbola_vertex_and_focus (x y : ℝ) : Prop :=
  hyperbola_eq x y ∧ ellipse_eq x y

-- Theorem stating that the ellipse equation holds given the conditions
theorem ellipse_is_correct :
  ∀ (x y : ℝ), ellipse_focus_vertex_of_hyperbola_vertex_and_focus x y →
  ellipse_eq x y := by
  intros x y h
  sorry

end ellipse_is_correct_l21_21775


namespace range_of_k_l21_21047

noncomputable section

open Classical

variables {A B C k : ℝ}

def is_acute_triangle (A B C : ℝ) := A < 90 ∧ B < 90 ∧ C < 90

theorem range_of_k (hA : A = 60) (hBC : BC = 6) (h_acute : is_acute_triangle A B C) : 
  2 * Real.sqrt 3 < k ∧ k < 4 * Real.sqrt 3 :=
sorry

end range_of_k_l21_21047


namespace inequality_proof_l21_21928

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ((a^2 + a + 1) * (b^2 + b + 1) * (c^2 + c + 1)) / (a * b * c) ≥ 27 :=
by
  sorry

end inequality_proof_l21_21928


namespace speed_of_man_rowing_upstream_l21_21742

theorem speed_of_man_rowing_upstream (V_m V_downstream V_upstream : ℝ) 
  (H1 : V_m = 60) 
  (H2 : V_downstream = 65) 
  (H3 : V_upstream = V_m - (V_downstream - V_m)) : 
  V_upstream = 55 := 
by 
  subst H1 
  subst H2 
  rw [H3] 
  norm_num

end speed_of_man_rowing_upstream_l21_21742


namespace least_number_of_pennies_l21_21876

theorem least_number_of_pennies (a : ℕ) :
  (a ≡ 1 [MOD 7]) ∧ (a ≡ 0 [MOD 3]) → a = 15 := by
  sorry

end least_number_of_pennies_l21_21876


namespace small_paintings_completed_l21_21089

variable (S : ℕ)

def uses_paint : Prop :=
  3 * 3 + 2 * S = 17

theorem small_paintings_completed : uses_paint S → S = 4 := by
  intro h
  sorry

end small_paintings_completed_l21_21089


namespace band_to_orchestra_ratio_is_two_l21_21476

noncomputable def ratio_of_band_to_orchestra : ℤ :=
  let orchestra_students := 20
  let choir_students := 28
  let total_students := 88
  let band_students := (total_students - orchestra_students - choir_students)
  band_students / orchestra_students

theorem band_to_orchestra_ratio_is_two :
  let orchestra_students := 20
  let choir_students := 28
  let total_students := 88
  ratio_of_band_to_orchestra = 2 := by
  sorry

end band_to_orchestra_ratio_is_two_l21_21476


namespace xyz_value_l21_21942

-- We define the constants from the problem
variables {x y z : ℂ}

-- Here's the theorem statement in Lean 4.
theorem xyz_value :
  (x * y + 5 * y = -20) →
  (y * z + 5 * z = -20) →
  (z * x + 5 * x = -20) →
  x * y * z = 100 :=
by
  intros h1 h2 h3
  sorry

end xyz_value_l21_21942


namespace speed_of_current_is_6_l21_21312

noncomputable def speed_of_current : ℝ :=
  let Vm := 18  -- speed in still water in kmph
  let distance_m := 100  -- distance covered in meters
  let time_s := 14.998800095992323  -- time taken in seconds
  let distance_km := distance_m / 1000  -- converting distance to kilometers
  let time_h := time_s / 3600  -- converting time to hours
  let Vd := distance_km / time_h  -- speed downstream in kmph
  Vd - Vm  -- speed of the current

theorem speed_of_current_is_6 :
  speed_of_current = 6 := by
  sorry -- proof is skipped

end speed_of_current_is_6_l21_21312


namespace Alice_spent_19_percent_l21_21199

variable (A B A': ℝ)
def Bob_less_money_than_Alice (A B : ℝ) : Prop :=
  B = 0.9 * A

def Alice_less_money_than_Bob (B A' : ℝ) : Prop :=
  A' = 0.9 * B

theorem Alice_spent_19_percent (A B A' : ℝ) 
  (h1 : Bob_less_money_than_Alice A B)
  (h2 : Alice_less_money_than_Bob B A') :
  ((A - A') / A) * 100 = 19 :=
by
  sorry

end Alice_spent_19_percent_l21_21199


namespace total_weight_of_fish_l21_21288

-- Define the weights of fish caught by Peter, Ali, and Joey.
variables (P A J : ℕ)

-- Ali caught twice as much fish as Peter.
def condition1 := A = 2 * P

-- Joey caught 1 kg more fish than Peter.
def condition2 := J = P + 1

-- Ali caught 12 kg of fish.
def condition3 := A = 12

-- Prove the total weight of the fish caught by all three is 25 kg.
theorem total_weight_of_fish :
  condition1 P A → condition2 P J → condition3 A → P + A + J = 25 :=
by
  intros h1 h2 h3
  sorry

end total_weight_of_fish_l21_21288


namespace a_2005_l21_21005

noncomputable def a : ℕ → ℤ := sorry 

axiom a3 : a 3 = 5
axiom a5 : a 5 = 8
axiom exists_n : ∃ (n : ℕ), n > 0 ∧ a n + a (n + 1) + a (n + 2) = 7

theorem a_2005 : a 2005 = -6 := by {
  sorry
}

end a_2005_l21_21005


namespace total_apples_l21_21212

theorem total_apples (x : ℕ) : 
    (x - x / 5 - x / 12 - x / 8 - x / 20 - x / 4 - x / 7 - x / 30 - 4 * (x / 30) - 300 ≤ 50) -> 
    x = 3360 :=
by
    sorry

end total_apples_l21_21212


namespace inequality_am_gm_l21_21037

variable (a b c d : ℝ)

theorem inequality_am_gm (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h_sum : a + b + c + d = 1) :
  (b * c * d) / (1 - a)^2 + (a * c * d) / (1 - b)^2 + (a * b * d) / (1 - c)^2 + (a * b * c) / (1 - d)^2 ≤ 1 / 9  :=
by    
  sorry

end inequality_am_gm_l21_21037


namespace value_of_a_l21_21467

theorem value_of_a 
  (x y a : ℝ)
  (h1 : 2 * x + y = 3 * a)
  (h2 : x - 2 * y = 9 * a)
  (h3 : x + 3 * y = 24) :
  a = -4 :=
sorry

end value_of_a_l21_21467


namespace c_alone_finishes_job_in_7_5_days_l21_21231

theorem c_alone_finishes_job_in_7_5_days (A B C : ℝ) (h1 : A + B = 1 / 15) (h2 : A + B + C = 1 / 5) :
  1 / C = 7.5 :=
by
  -- The proof is omitted
  sorry

end c_alone_finishes_job_in_7_5_days_l21_21231


namespace nested_f_has_zero_l21_21571

def f (x : ℝ) : ℝ := x^2 + 2017 * x + 1

theorem nested_f_has_zero (n : ℕ) (hn : n ≥ 1) : ∃ x : ℝ, (Nat.iterate f n x) = 0 :=
by
  sorry

end nested_f_has_zero_l21_21571


namespace find_four_digit_numbers_l21_21965

theorem find_four_digit_numbers (a b c d : ℕ) : 
  (1000 ≤ 1000 * a + 100 * b + 10 * c + d) ∧ 
  (1000 * a + 100 * b + 10 * c + d ≤ 9999) ∧ 
  (1000 ≤ 1000 * d + 100 * c + 10 * b + a) ∧ 
  (1000 * d + 100 * c + 10 * b + a ≤ 9999) ∧
  (a + d = 9) ∧ 
  (b + c = 13) ∧
  (1001 * (a + d) + 110 * (b + c) = 19448) → 
  (1000 * a + 100 * b + 10 * c + d = 9949 ∨ 
   1000 * a + 100 * b + 10 * c + d = 9859 ∨ 
   1000 * a + 100 * b + 10 * c + d = 9769 ∨ 
   1000 * a + 100 * b + 10 * c + d = 9679 ∨ 
   1000 * a + 100 * b + 10 * c + d = 9589 ∨ 
   1000 * a + 100 * b + 10 * c + d = 9499) :=
sorry

end find_four_digit_numbers_l21_21965


namespace largest_circle_area_l21_21170

theorem largest_circle_area (PQ QR PR : ℝ)
  (h_right_triangle: PR^2 = PQ^2 + QR^2)
  (h_circle_areas_sum: π * (PQ/2)^2 + π * (QR/2)^2 + π * (PR/2)^2 = 338 * π) :
  π * (PR/2)^2 = 169 * π :=
by
  sorry

end largest_circle_area_l21_21170


namespace arlo_books_l21_21269

theorem arlo_books (total_items : ℕ) (books_ratio : ℕ) (pens_ratio : ℕ) (notebooks_ratio : ℕ) 
  (ratio_sum : ℕ) (items_per_part : ℕ) (parts_for_books : ℕ) (total_parts : ℕ) :
  total_items = 600 →
  books_ratio = 7 →
  pens_ratio = 3 →
  notebooks_ratio = 2 →
  total_parts = books_ratio + pens_ratio + notebooks_ratio →
  items_per_part = total_items / total_parts →
  parts_for_books = books_ratio →
  parts_for_books * items_per_part = 350 := by
  intros
  sorry

end arlo_books_l21_21269


namespace xy_value_l21_21330

theorem xy_value (x y : ℝ) (h : x ≠ y) (h_eq : x^2 + 2 / x^2 = y^2 + 2 / y^2) : 
  x * y = Real.sqrt 2 ∨ x * y = -Real.sqrt 2 :=
by
  sorry

end xy_value_l21_21330


namespace angle_terminal_side_equivalence_l21_21459

theorem angle_terminal_side_equivalence (k : ℤ) : 
    ∃ k : ℤ, 405 = k * 360 + 45 :=
by
  sorry

end angle_terminal_side_equivalence_l21_21459


namespace polynomial_arithmetic_sequence_roots_l21_21482

theorem polynomial_arithmetic_sequence_roots (p q : ℝ) (h : ∃ a b c d : ℝ, 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
  a + 3*(b - a) = b ∧ b + 3*(c - b) = c ∧ c + 3*(d - c) = d ∧ 
  (a^4 + p * a^2 + q = 0) ∧ (b^4 + p * b^2 + q = 0) ∧ 
  (c^4 + p * c^2 + q = 0) ∧ (d^4 + p * d^2 + q = 0)) :
  p ≤ 0 ∧ q = 0.09 * p^2 := 
sorry

end polynomial_arithmetic_sequence_roots_l21_21482


namespace increase_in_volume_eq_l21_21753

theorem increase_in_volume_eq (x : ℝ) (l w h : ℝ) (h₀ : l = 6) (h₁ : w = 4) (h₂ : h = 5) :
  (6 + x) * 4 * 5 = 6 * 4 * (5 + x) :=
by
  sorry

end increase_in_volume_eq_l21_21753


namespace growth_factor_condition_l21_21883

open BigOperators

theorem growth_factor_condition {n : ℕ} (h : ∏ i in Finset.range n, (i + 2) / (i + 1) = 50) : n = 49 := by
  sorry

end growth_factor_condition_l21_21883


namespace smallest_difference_of_factors_l21_21284

theorem smallest_difference_of_factors (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a * b = 2268) : 
  (a = 42 ∧ b = 54) ∨ (a = 54 ∧ b = 42) := sorry

end smallest_difference_of_factors_l21_21284


namespace sum_of_integers_is_eleven_l21_21202

theorem sum_of_integers_is_eleven (p q r s : ℤ) 
  (h1 : p - q + r = 7) 
  (h2 : q - r + s = 8) 
  (h3 : r - s + p = 4) 
  (h4 : s - p + q = 3) : 
  p + q + r + s = 11 := 
by
  sorry

end sum_of_integers_is_eleven_l21_21202


namespace find_fx_for_l21_21246

theorem find_fx_for {f : ℕ → ℤ} (h1 : f 0 = 1) (h2 : ∀ x, f (x + 1) = f x + 2 * x + 3) : f 2012 = 4052169 :=
by
  sorry

end find_fx_for_l21_21246


namespace find_z_l21_21153

-- Definitions based on the conditions from the problem
def x : ℤ := sorry
def y : ℤ := x - 1
def z : ℤ := x - 2
def condition1 : x > y ∧ y > z := by
  sorry

def condition2 : 2 * x + 3 * y + 3 * z = 5 * y + 11 := by
  sorry

-- Statement to prove
theorem find_z : z = 3 :=
by
  -- Use the conditions to prove the statement
  have h1 : x > y ∧ y > z := condition1
  have h2 : 2 * x + 3 * y + 3 * z = 5 * y + 11 := condition2
  sorry

end find_z_l21_21153


namespace total_time_to_fill_tank_l21_21388

noncomputable def pipe_filling_time : ℕ := 
  let tank_capacity := 2000
  let pipe_a_rate := 200
  let pipe_b_rate := 50
  let pipe_c_rate := 25
  let cycle_duration := 5
  let cycle_fill := (pipe_a_rate * 1 + pipe_b_rate * 2 - pipe_c_rate * 2)
  let num_cycles := tank_capacity / cycle_fill
  num_cycles * cycle_duration

theorem total_time_to_fill_tank : pipe_filling_time = 40 := 
by
  unfold pipe_filling_time
  sorry

end total_time_to_fill_tank_l21_21388


namespace chicken_feathers_after_crossing_l21_21283

def feathers_remaining_after_crossings (cars_dodged feathers_before pulling_factor : ℕ) : ℕ :=
  let feathers_lost := cars_dodged * pulling_factor
  feathers_before - feathers_lost

theorem chicken_feathers_after_crossing 
  (cars_dodged : ℕ := 23)
  (feathers_before : ℕ := 5263)
  (pulling_factor : ℕ := 2) :
  feathers_remaining_after_crossings cars_dodged feathers_before pulling_factor = 5217 :=
by
  sorry

end chicken_feathers_after_crossing_l21_21283


namespace problem_part_1_problem_part_2_l21_21849

theorem problem_part_1 (a b : ℝ) (h1 : a * 1^2 - 3 * 1 + 2 = 0) (h2 : a * b^2 - 3 * b + 2 = 0) (h3 : 1 + b = 3 / a) (h4 : 1 * b = 2 / a) : a = 1 ∧ b = 2 :=
sorry

theorem problem_part_2 (m : ℝ) (h5 : a = 1) (h6 : b = 2) : 
  (m = 2 → ∀ x, ¬ (x^2 - (m + 2) * x + 2 * m < 0)) ∧
  (m < 2 → ∀ x, x ∈ Set.Ioo m 2 ↔ x^2 - (m + 2) * x + 2 * m < 0) ∧
  (m > 2 → ∀ x, x ∈ Set.Ioo 2 m ↔ x^2 - (m + 2) * x + 2 * m < 0) :=
sorry

end problem_part_1_problem_part_2_l21_21849


namespace mike_taller_than_mark_l21_21147

def height_mark_feet : ℕ := 5
def height_mark_inches : ℕ := 3
def height_mike_feet : ℕ := 6
def height_mike_inches : ℕ := 1
def feet_to_inches : ℕ := 12

-- Calculate heights in inches.
def height_mark_total_inches : ℕ := height_mark_feet * feet_to_inches + height_mark_inches
def height_mike_total_inches : ℕ := height_mike_feet * feet_to_inches + height_mike_inches

-- Prove the height difference.
theorem mike_taller_than_mark : height_mike_total_inches - height_mark_total_inches = 10 :=
by
  sorry

end mike_taller_than_mark_l21_21147


namespace decimal_multiplication_l21_21766

theorem decimal_multiplication : (3.6 * 0.3 = 1.08) := by
  sorry

end decimal_multiplication_l21_21766


namespace cone_height_of_semicircular_sheet_l21_21040

theorem cone_height_of_semicircular_sheet (R h : ℝ) (h_cond: h = R) : h = R :=
by
  exact h_cond

end cone_height_of_semicircular_sheet_l21_21040


namespace expected_value_of_win_l21_21376

theorem expected_value_of_win :
  let prob := (1:ℝ)/8
  let win_value (n : ℕ) := (n ^ 3 : ℝ)
  (prob * win_value 1 + prob * win_value 2 + prob * win_value 3 + prob * win_value 4 + prob * win_value 5 + 
   prob * win_value 6 + prob * win_value 7 + prob * win_value 8 : ℝ) = 162 :=
by
  let prob := (1:ℝ)/8
  let win_value (n : ℕ) := (n ^ 3 : ℝ)
  have E : (prob * win_value 1 + prob * win_value 2 + prob * win_value 3 + prob * win_value 4 + prob * win_value 5 + 
            prob * win_value 6 + prob * win_value 7 + prob * win_value 8 : ℝ) = 162 := sorry
  exact E

end expected_value_of_win_l21_21376


namespace average_player_time_l21_21718

theorem average_player_time:
  let pg := 130
  let sg := 145
  let sf := 85
  let pf := 60
  let c := 180
  let total_secs := pg + sg + sf + pf + c
  let total_mins := total_secs / 60
  let num_players := 5
  let avg_mins_per_player := total_mins / num_players
  avg_mins_per_player = 2 :=
by
  sorry

end average_player_time_l21_21718


namespace Nell_initial_cards_l21_21191

theorem Nell_initial_cards (n : ℕ) (h1 : n - 136 = 106) : n = 242 := 
by
  sorry

end Nell_initial_cards_l21_21191


namespace intersection_complement_l21_21109

open Set

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {3, 4, 5}
def B : Set ℕ := {1, 3, 6}

theorem intersection_complement :
  A ∩ (U \ B) = {4, 5} := by
  sorry

end intersection_complement_l21_21109


namespace tenth_battery_replacement_in_january_l21_21957

theorem tenth_battery_replacement_in_january : ∀ (months_to_replace: ℕ) (start_month: ℕ), 
  months_to_replace = 4 → start_month = 1 → (4 * (10 - 1)) % 12 = 0 → start_month = 1 :=
by
  intros months_to_replace start_month h_replace h_start h_calc
  sorry

end tenth_battery_replacement_in_january_l21_21957


namespace find_digit_x_l21_21776

def base7_number (x : ℕ) : ℕ := 5 * 7^3 + 2 * 7^2 + x * 7 + 4

def is_divisible_by_19 (n : ℕ) : Prop := ∃ k : ℕ, n = 19 * k

theorem find_digit_x : is_divisible_by_19 (base7_number 4) :=
sorry

end find_digit_x_l21_21776


namespace exists_solution_for_lambda_9_l21_21769

theorem exists_solution_for_lambda_9 :
  ∃ x y : ℝ, (x^2 + y^2 = 8 * x + 6 * y) ∧ (9 * x^2 + y^2 = 6 * y) ∧ (y^2 + 9 = 9 * x + 6 * y + 9) :=
by
  sorry

end exists_solution_for_lambda_9_l21_21769


namespace radius_squared_l21_21701

-- Definitions of the conditions
def point_A := (2, -1)
def line_l1 (x y : ℝ) := x + y = 1
def line_l2 (x y : ℝ) := 2 * x + y = 0

-- Circle with center (h, k) and radius r
def circle_equation (h k r x y : ℝ) := (x - h) ^ 2 + (y - k) ^ 2 = r ^ 2

-- Prove statement: r^2 = 2 given the conditions
theorem radius_squared (h k r : ℝ) 
  (H1 : circle_equation h k r 2 (-1))
  (H2 : line_l1 h k)
  (H3 : line_l2 h k):
  r ^ 2 = 2 := sorry

end radius_squared_l21_21701


namespace recurring_decimal_to_fraction_l21_21318

theorem recurring_decimal_to_fraction :
  let x := 0.4 + 67 / (99 : ℝ)
  (∀ y : ℝ, y = x ↔ y = 463 / 990) := 
by
  sorry

end recurring_decimal_to_fraction_l21_21318


namespace pie_eating_contest_l21_21301

def a : ℚ := 7 / 8
def b : ℚ := 5 / 6
def difference : ℚ := 1 / 24

theorem pie_eating_contest : a - b = difference := 
sorry

end pie_eating_contest_l21_21301


namespace remainder_of_n_div_4_is_1_l21_21641

noncomputable def n : ℕ := sorry  -- We declare n as a noncomputable natural number to proceed with the proof complexity

theorem remainder_of_n_div_4_is_1 (n : ℕ) (h : (2 * n) % 4 = 2) : n % 4 = 1 :=
by
  sorry  -- skip the proof

end remainder_of_n_div_4_is_1_l21_21641


namespace sin_cos_product_l21_21216

theorem sin_cos_product (x : ℝ) (h₁ : 0 < x) (h₂ : x < π / 2) (h₃ : Real.sin x = 3 * Real.cos x) : 
  Real.sin x * Real.cos x = 3 / 10 :=
by
  sorry

end sin_cos_product_l21_21216


namespace system_solution_a_l21_21185

theorem system_solution_a (x y a : ℝ) (h1 : 3 * x + y = a) (h2 : 2 * x + 5 * y = 2 * a) (hx : x = 3) : a = 13 :=
by
  sorry

end system_solution_a_l21_21185


namespace fraction_unseated_l21_21872

theorem fraction_unseated :
  ∀ (tables seats_per_table seats_taken : ℕ),
  tables = 15 →
  seats_per_table = 10 →
  seats_taken = 135 →
  ((tables * seats_per_table - seats_taken : ℕ) / (tables * seats_per_table : ℕ) : ℚ) = 1 / 10 :=
by
  intros tables seats_per_table seats_taken h_tables h_seats_per_table h_seats_taken
  sorry

end fraction_unseated_l21_21872


namespace min_hypotenuse_of_right_triangle_l21_21031

theorem min_hypotenuse_of_right_triangle (a b c : ℝ) 
  (h1 : a^2 + b^2 = c^2) 
  (h2 : a + b + c = 6) : 
  c = 6 * (Real.sqrt 2 - 1) :=
sorry

end min_hypotenuse_of_right_triangle_l21_21031


namespace first_percentage_reduction_l21_21304

theorem first_percentage_reduction (P : ℝ) (x : ℝ) :
  (P - (x / 100) * P) * 0.4 = P * 0.3 → x = 25 := by
  sorry

end first_percentage_reduction_l21_21304


namespace four_digit_numbers_div_by_5_with_34_end_l21_21423

theorem four_digit_numbers_div_by_5_with_34_end : 
  ∃ (count : ℕ), count = 90 ∧
  ∀ (n : ℕ), (1000 ≤ n ∧ n < 10000) →
  (n % 100 = 34) →
  ((10 ∣ n) ∨ (5 ∣ n)) →
  (count = 90) :=
sorry

end four_digit_numbers_div_by_5_with_34_end_l21_21423


namespace point_not_in_fourth_quadrant_l21_21468

theorem point_not_in_fourth_quadrant (a : ℝ) :
  ¬ ((a - 3 > 0) ∧ (a + 3 < 0)) :=
by
  sorry

end point_not_in_fourth_quadrant_l21_21468


namespace find_large_number_l21_21539

theorem find_large_number (L S : ℕ) 
  (h1 : L - S = 1335) 
  (h2 : L = 6 * S + 15) : 
  L = 1599 := 
by 
  -- proof omitted
  sorry

end find_large_number_l21_21539


namespace volume_CO2_is_7_l21_21702

-- Definitions based on conditions
def Avogadro_law (V1 V2 : ℝ) : Prop := V1 = V2
def molar_ratio (V_CO2 V_O2 : ℝ) : Prop := V_CO2 = 1 / 2 * V_O2
def volume_O2 : ℝ := 14

-- Statement to be proved
theorem volume_CO2_is_7 : ∃ V_CO2 : ℝ, molar_ratio V_CO2 volume_O2 ∧ V_CO2 = 7 := by
  sorry

end volume_CO2_is_7_l21_21702


namespace box_contains_1_8_grams_child_ingests_0_1_grams_l21_21897

-- Define the conditions
def packet_weight : ℝ := 0.2
def packets_in_box : ℕ := 9
def half_a_packet : ℝ := 0.5

-- Prove that a box contains 1.8 grams of "acetaminophen"
theorem box_contains_1_8_grams : packets_in_box * packet_weight = 1.8 :=
by
  sorry

-- Prove that a child will ingest 0.1 grams of "acetaminophen" if they take half a packet
theorem child_ingests_0_1_grams : half_a_packet * packet_weight = 0.1 :=
by
  sorry

end box_contains_1_8_grams_child_ingests_0_1_grams_l21_21897


namespace value_of_expression_l21_21913

theorem value_of_expression (m n : ℤ) (hm : |m| = 3) (hn : |n| = 2) (hmn : m < n) :
  m^2 + m * n + n^2 = 7 ∨ m^2 + m * n + n^2 = 19 := by
  sorry

end value_of_expression_l21_21913


namespace problem_part_1_problem_part_2_problem_part_3_l21_21552

open Set

universe u

def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 8}
def B : Set ℝ := {x | 1 < x ∧ x < 6}
def C (a : ℝ) : Set ℝ := {x | x > a}
def U : Set ℝ := univ

theorem problem_part_1 : A ∪ B = {x | 1 < x ∧ x ≤ 8} :=
sorry

theorem problem_part_2 : (U \ A) ∩ B = {x | 1 < x ∧ x < 2} :=
sorry

theorem problem_part_3 (a : ℝ) (h : (A ∩ C a) ≠ ∅) : a < 8 :=
sorry

end problem_part_1_problem_part_2_problem_part_3_l21_21552


namespace a_b_sum_possible_values_l21_21728

theorem a_b_sum_possible_values (a b : ℝ) 
  (h1 : a^3 - 12 * a^2 + 9 * a - 18 = 0)
  (h2 : 9 * b^3 - 135 * b^2 + 450 * b - 1650 = 0) :
  a + b = 6 ∨ a + b = 14 :=
sorry

end a_b_sum_possible_values_l21_21728


namespace number_of_balls_sold_l21_21581

-- Definitions from conditions
def selling_price : ℕ := 720
def cost_price_per_ball : ℕ := 120
def loss : ℕ := 5 * cost_price_per_ball

-- Mathematically equivalent proof statement
theorem number_of_balls_sold (n : ℕ) (h : n * cost_price_per_ball - selling_price = loss) : n = 11 :=
  sorry

end number_of_balls_sold_l21_21581


namespace factor_expression_l21_21843

theorem factor_expression (x : ℝ) :
  (7 * x^6 + 36 * x^4 - 8) - (3 * x^6 - 4 * x^4 + 6) = 2 * (2 * x^6 + 20 * x^4 - 7) :=
  sorry

end factor_expression_l21_21843


namespace area_of_circle_segment_l21_21712

-- Definitions for the conditions in the problem
def circle_eq (x y : ℝ) : Prop := x^2 - 10 * x + y^2 = 9
def line_eq (x y : ℝ) : Prop := y = x - 5

-- The area of the portion of the circle that lies above the x-axis and to the left of the line y = x - 5
theorem area_of_circle_segment :
  let area_of_circle := 34 * Real.pi
  let portion_fraction := 1 / 8
  portion_fraction * area_of_circle = 4.25 * Real.pi :=
by
  sorry

end area_of_circle_segment_l21_21712


namespace find_M_plus_N_l21_21027

theorem find_M_plus_N (M N : ℕ) (h1 : (3:ℚ) / 5 = M / 45) (h2 : (3:ℚ) / 5 = 60 / N) : M + N = 127 :=
sorry

end find_M_plus_N_l21_21027


namespace kona_additional_miles_l21_21203

theorem kona_additional_miles 
  (d_apartment_to_bakery : ℕ := 9) 
  (d_bakery_to_grandmother : ℕ := 24) 
  (d_grandmother_to_apartment : ℕ := 27) : 
  (d_apartment_to_bakery + d_bakery_to_grandmother + d_grandmother_to_apartment) - (2 * d_grandmother_to_apartment) = 6 := 
by 
  sorry

end kona_additional_miles_l21_21203


namespace units_digit_of_m_squared_plus_two_to_the_m_is_seven_l21_21633

def m := 2016^2 + 2^2016

theorem units_digit_of_m_squared_plus_two_to_the_m_is_seven :
  (m^2 + 2^m) % 10 = 7 := by
sorry

end units_digit_of_m_squared_plus_two_to_the_m_is_seven_l21_21633


namespace probability_same_color_is_27_over_100_l21_21911

def num_sides_die1 := 20
def num_sides_die2 := 20

def maroon_die1 := 5
def teal_die1 := 6
def cyan_die1 := 7
def sparkly_die1 := 1
def silver_die1 := 1

def maroon_die2 := 4
def teal_die2 := 6
def cyan_die2 := 7
def sparkly_die2 := 1
def silver_die2 := 2

noncomputable def probability_same_color : ℚ :=
  (maroon_die1 * maroon_die2 + teal_die1 * teal_die2 + cyan_die1 * cyan_die2 + sparkly_die1 * sparkly_die2 + silver_die1 * silver_die2) /
  (num_sides_die1 * num_sides_die2)

theorem probability_same_color_is_27_over_100 :
  probability_same_color = 27 / 100 := 
sorry

end probability_same_color_is_27_over_100_l21_21911


namespace x_intercept_l21_21204

theorem x_intercept (x1 y1 x2 y2 : ℝ) (hx1 : x1 = 10) (hy1 : y1 = 3) (hx2 : x2 = -8) (hy2 : y2 = -6) : 
  ∃ x : ℝ, (y = 0) ∧ (∃ m : ℝ, m = (y2 - y1) / (x2 - x1) ∧ y1 - y = m * (x1 - x)) ∧ x = 4 :=
sorry

end x_intercept_l21_21204


namespace acute_not_greater_than_right_l21_21773

-- Definitions for conditions
def is_right_angle (α : ℝ) : Prop := α = 90
def is_acute_angle (α : ℝ) : Prop := α < 90

-- Statement to be proved
theorem acute_not_greater_than_right (α : ℝ) (h1 : is_right_angle 90) (h2 : is_acute_angle α) : ¬ (α > 90) :=
by
    sorry

end acute_not_greater_than_right_l21_21773


namespace evaluate_expression_l21_21688

open BigOperators

theorem evaluate_expression : 
  ∀ (x y : ℤ), x = -1 → y = 1 → 2 * (x^2 * y + x * y) - 3 * (x^2 * y - x * y) - 5 * x * y = -1 :=
by
  intros x y hx hy
  rw [hx, hy]
  sorry

end evaluate_expression_l21_21688


namespace find_P_coordinates_l21_21610

-- Given points A and B
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (0, 2)

-- The area of triangle PAB is 5
def areaPAB (P : ℝ × ℝ) : ℝ :=
  0.5 * abs (P.1 * (A.2 - B.2) + A.1 * (B.2 - P.2) + B.1 * (P.2 - A.2))

-- Point P lies on the x-axis
def on_x_axis (P : ℝ × ℝ) : Prop := P.2 = 0

theorem find_P_coordinates (P : ℝ × ℝ) :
  on_x_axis P → areaPAB P = 5 → (P = (-4, 0) ∨ P = (6, 0)) :=
by
  sorry

end find_P_coordinates_l21_21610


namespace solution_set_of_inequality_l21_21480

theorem solution_set_of_inequality (x : ℝ) :
  2 * |x - 1| - 1 < 0 ↔ (1 / 2) < x ∧ x < (3 / 2) :=
  sorry

end solution_set_of_inequality_l21_21480


namespace diff_in_set_l21_21006

variable (A : Set Int)
variable (ha : ∃ a ∈ A, a > 0)
variable (hb : ∃ b ∈ A, b < 0)
variable (h : ∀ {a b : Int}, a ∈ A → b ∈ A → (2 * a) ∈ A ∧ (a + b) ∈ A)

theorem diff_in_set (x y : Int) (hx : x ∈ A) (hy : y ∈ A) : (x - y) ∈ A :=
  sorry

end diff_in_set_l21_21006


namespace henry_correct_answers_l21_21206

theorem henry_correct_answers (c w : ℕ) (h1 : c + w = 15) (h2 : 6 * c - 3 * w = 45) : c = 10 :=
by
  sorry

end henry_correct_answers_l21_21206


namespace monotonicity_and_inequality_l21_21346

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - x^2

theorem monotonicity_and_inequality (a : ℝ) (p q : ℝ) (hp : 0 < p ∧ p < 1) (hq : 0 < q ∧ q < 1)
  (h_distinct: p ≠ q) (h_a : a ≥ 10) : 
  (f a (p + 1) - f a (q + 1)) / (p - q) > 1 := by
  sorry

end monotonicity_and_inequality_l21_21346


namespace find_students_with_equal_homework_hours_l21_21201

theorem find_students_with_equal_homework_hours :
  let Dan := 6
  let Joe := 3
  let Bob := 5
  let Susie := 4
  let Grace := 1
  (Joe + Grace = Dan ∨ Joe + Bob = Dan ∨ Bob + Grace = Dan ∨ Dan + Bob = Dan ∨ Susie + Grace = Dan) → 
  (Bob + Grace = Dan) := 
by 
  intros
  sorry

end find_students_with_equal_homework_hours_l21_21201


namespace max_area_14_5_l21_21162

noncomputable def rectangle_max_area (P D : ℕ) (x y : ℝ) : ℝ :=
  if (2 * x + 2 * y = P) ∧ (x^2 + y^2 = D^2) then x * y else 0

theorem max_area_14_5 :
  ∃ (x y : ℝ), (2 * x + 2 * y = 14) ∧ (x^2 + y^2 = 5^2) ∧ rectangle_max_area 14 5 x y = 12.25 :=
by
  sorry

end max_area_14_5_l21_21162


namespace next_in_sequence_is_65_by_19_l21_21074

section
  open Int

  -- Definitions for numerators
  def numerator_sequence : ℕ → ℤ
  | 0 => -3
  | 1 => 5
  | 2 => -9
  | 3 => 17
  | 4 => -33
  | (n + 5) => numerator_sequence n * (-2) + 1

  -- Definitions for denominators
  def denominator_sequence : ℕ → ℕ
  | 0 => 4
  | 1 => 7
  | 2 => 10
  | 3 => 13
  | 4 => 16
  | (n + 5) => denominator_sequence n + 3

  -- Next term in the sequence
  def next_term (n : ℕ) : ℚ :=
    (numerator_sequence (n + 5) : ℚ) / (denominator_sequence (n + 5) : ℚ)

  -- Theorem stating the next number in the sequence
  theorem next_in_sequence_is_65_by_19 :
    next_term 0 = 65 / 19 :=
  by
    unfold next_term
    simp [numerator_sequence, denominator_sequence]
    sorry
end

end next_in_sequence_is_65_by_19_l21_21074


namespace suff_cond_iff_lt_l21_21833

variable (a b : ℝ)

-- Proving that (a - b) a^2 < 0 is a sufficient but not necessary condition for a < b
theorem suff_cond_iff_lt (h : (a - b) * a^2 < 0) : a < b :=
by {
  sorry
}

end suff_cond_iff_lt_l21_21833


namespace inclination_angle_x_eq_one_l21_21530

noncomputable def inclination_angle_of_vertical_line (x : ℝ) : ℝ :=
if x = 1 then 90 else 0

theorem inclination_angle_x_eq_one :
  inclination_angle_of_vertical_line 1 = 90 :=
by
  sorry

end inclination_angle_x_eq_one_l21_21530


namespace inequality_proof_l21_21689

theorem inequality_proof
  (a b c : ℝ)
  (h1 : a ≤ b)
  (h2 : b ≤ c)
  (h3 : 0 < c)
  : a + b ≤ 2 * c ∧ 2 * c ≤ 3 * c :=
sorry

end inequality_proof_l21_21689


namespace tile_count_l21_21770

theorem tile_count (room_length room_width tile_length tile_width : ℝ)
  (h1 : room_length = 10)
  (h2 : room_width = 15)
  (h3 : tile_length = 1 / 4)
  (h4 : tile_width = 3 / 4) :
  (room_length * room_width) / (tile_length * tile_width) = 800 :=
by
  sorry

end tile_count_l21_21770


namespace checkered_rectangles_unique_gray_cells_l21_21167

noncomputable def num_checkered_rectangles (num_gray_cells : ℕ) (num_blue_cells : ℕ) (rects_per_blue_cell : ℕ)
    (num_red_cells : ℕ) (rects_per_red_cell : ℕ) : ℕ :=
    (num_blue_cells * rects_per_blue_cell) + (num_red_cells * rects_per_red_cell)

theorem checkered_rectangles_unique_gray_cells : num_checkered_rectangles 40 36 4 4 8 = 176 := 
sorry

end checkered_rectangles_unique_gray_cells_l21_21167


namespace length_of_PB_l21_21996

theorem length_of_PB 
  (AB BC : ℝ) 
  (PA PD PC PB : ℝ)
  (h1 : AB = 2 * BC) 
  (h2 : PA = 5) 
  (h3 : PD = 12) 
  (h4 : PC = 13) 
  (h5 : PA^2 + PB^2 = (AB^2 + BC^2) / 5) -- derived from question
  (h6 : PB^2 = ((2 * BC)^2) - PA^2) : 
  PB = 10.5 :=
by 
  -- We would insert proof steps here (not required as per instructions)
  sorry

end length_of_PB_l21_21996


namespace inequality_example_l21_21660

theorem inequality_example (a b : ℝ) (h : a - b > 0) : a + 1 > b + 1 :=
sorry

end inequality_example_l21_21660


namespace at_least_one_real_root_l21_21257

theorem at_least_one_real_root (a : ℝ) :
  (4*a)^2 - 4*(-4*a + 3) ≥ 0 ∨
  ((a - 1)^2 - 4*a^2) ≥ 0 ∨
  (2*a)^2 - 4*(-2*a) ≥ 0 := sorry

end at_least_one_real_root_l21_21257


namespace tank_empty_time_when_inlet_open_l21_21904

-- Define the conditions
def leak_empty_time : ℕ := 6
def tank_capacity : ℕ := 4320
def inlet_rate_per_minute : ℕ := 6

-- Calculate rates from conditions
def leak_rate_per_hour : ℕ := tank_capacity / leak_empty_time
def inlet_rate_per_hour : ℕ := inlet_rate_per_minute * 60

-- Proof Problem: Prove the time for the tank to empty when both leak and inlet are open
theorem tank_empty_time_when_inlet_open :
  tank_capacity / (leak_rate_per_hour - inlet_rate_per_hour) = 12 :=
by
  sorry

end tank_empty_time_when_inlet_open_l21_21904


namespace num_ways_arrange_l21_21861

open Finset

def valid_combinations : Finset (Finset Nat) :=
  { {2, 5, 11, 3}, {3, 5, 6, 2}, {3, 6, 11, 5}, {5, 6, 11, 2} }

theorem num_ways_arrange : valid_combinations.card = 4 :=
  by
    sorry  -- proof of the statement

end num_ways_arrange_l21_21861


namespace parallel_planes_x_plus_y_l21_21967

def planes_parallel (x y : ℝ) : Prop :=
  ∃ k : ℝ, (x = -k) ∧ (1 = k * y) ∧ (-2 = (1 / 2) * k)

theorem parallel_planes_x_plus_y (x y : ℝ) (h : planes_parallel x y) : x + y = 15 / 4 :=
sorry

end parallel_planes_x_plus_y_l21_21967


namespace find_number_l21_21393

variables (n : ℝ)

-- Condition: a certain number divided by 14.5 equals 173.
def condition_1 (n : ℝ) : Prop := n / 14.5 = 173

-- Condition: 29.94 ÷ 1.45 = 17.3.
def condition_2 : Prop := 29.94 / 1.45 = 17.3

-- Theorem: Prove that the number is 2508.5 given the conditions.
theorem find_number (h1 : condition_1 n) (h2 : condition_2) : n = 2508.5 :=
by 
  sorry

end find_number_l21_21393


namespace verify_formula_n1_l21_21788

theorem verify_formula_n1 (a : ℝ) (ha : a ≠ 1) : 1 + a = (a^3 - 1) / (a - 1) :=
by 
  sorry

end verify_formula_n1_l21_21788


namespace expand_binomials_l21_21344

theorem expand_binomials : 
  (x + 4) * (x - 9) = x^2 - 5*x - 36 := 
by 
  sorry

end expand_binomials_l21_21344


namespace consecutive_integer_cubes_sum_l21_21101

theorem consecutive_integer_cubes_sum : 
  ∀ (a : ℕ), 
  (a > 2) → 
  (a - 1) * a * (a + 1) * (a + 2) = 12 * ((a - 1) + a + (a + 1) + (a + 2)) →
  ((a - 1)^3 + a^3 + (a + 1)^3 + (a + 2)^3) = 224 :=
by
  intro a ha h
  sorry

end consecutive_integer_cubes_sum_l21_21101


namespace arithmetic_sequence_sum_l21_21278

theorem arithmetic_sequence_sum (S : ℕ → ℤ) (m : ℕ)
  (h1 : S (m - 1) = -2)
  (h2 : S m = 0)
  (h3 : S (m + 1) = 3) :
  m = 5 :=
sorry

end arithmetic_sequence_sum_l21_21278


namespace minimum_value_op_dot_fp_l21_21751

theorem minimum_value_op_dot_fp (x y : ℝ) (h_ellipse : x^2 / 2 + y^2 = 1) :
  let OP := (x, y)
  let FP := (x - 1, y)
  let dot_product := x * (x - 1) + y^2
  dot_product ≥ 1 / 2 :=
by
  sorry

end minimum_value_op_dot_fp_l21_21751


namespace not_necessarily_divisible_by_66_l21_21730

theorem not_necessarily_divisible_by_66 (m : ℤ) (h1 : ∃ k : ℤ, m = k * (k + 1) * (k + 2) * (k + 3) * (k + 4)) (h2 : 11 ∣ m) : ¬ (66 ∣ m) :=
sorry

end not_necessarily_divisible_by_66_l21_21730


namespace min_moves_to_break_chocolate_l21_21333

theorem min_moves_to_break_chocolate (n m : ℕ) (tiles : ℕ) (moves : ℕ) :
    (n = 4) → (m = 10) → (tiles = n * m) → (moves = tiles - 1) → moves = 39 :=
by
  intros hnm hn4 hm10 htm
  sorry

end min_moves_to_break_chocolate_l21_21333


namespace compute_fraction_power_l21_21962

theorem compute_fraction_power : 9 * (1 / 7)^4 = 9 / 2401 := by
  sorry

end compute_fraction_power_l21_21962


namespace total_caffeine_is_correct_l21_21159

def first_drink_caffeine := 250 -- milligrams
def first_drink_size := 12 -- ounces

def second_drink_caffeine_per_ounce := (first_drink_caffeine / first_drink_size) * 3
def second_drink_size := 8 -- ounces
def second_drink_caffeine := second_drink_caffeine_per_ounce * second_drink_size

def third_drink_concentration := 18 -- milligrams per milliliter
def third_drink_size := 150 -- milliliters
def third_drink_caffeine := third_drink_concentration * third_drink_size

def caffeine_pill_caffeine := first_drink_caffeine + second_drink_caffeine + third_drink_caffeine

def total_caffeine_consumed := first_drink_caffeine + second_drink_caffeine + third_drink_caffeine + caffeine_pill_caffeine

theorem total_caffeine_is_correct : total_caffeine_consumed = 6900 :=
by
  sorry

end total_caffeine_is_correct_l21_21159


namespace range_of_t_minus_1_over_t_minus_3_l21_21840

variable {f : ℝ → ℝ}

-- Function conditions: monotonically decreasing and odd
axiom f_mono_decreasing : ∀ x y : ℝ, x ≤ y → f y ≤ f x
axiom f_odd : ∀ x : ℝ, f (-x) = -f x

-- Condition on the real number t
variable {t : ℝ}
axiom f_condition : f (t^2 - 2 * t) + f (-3) > 0

-- Question: Prove the range of (t-1)/(t-3)
theorem range_of_t_minus_1_over_t_minus_3 (h : -1 < t ∧ t < 3) : 
  ((t - 1) / (t - 3)) < 1/2 :=
  sorry

end range_of_t_minus_1_over_t_minus_3_l21_21840


namespace determine_a_zeros_l21_21287

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x = 3 then a else 2 / |x - 3|

noncomputable def y (x : ℝ) (a : ℝ) : ℝ := f x a - 4

theorem determine_a_zeros (a : ℝ) : (∃ c d, c ≠ 3 ∧ d ≠ 3 ∧ c ≠ d ∧ y c a = 0 ∧ y d a = 0 ∧ y 3 a = 0) → a = 4 :=
sorry

end determine_a_zeros_l21_21287


namespace absolute_value_positive_l21_21067

theorem absolute_value_positive (a : ℝ) (h : a ≠ 0) : |a| > 0 := by
  sorry

end absolute_value_positive_l21_21067


namespace min_range_of_three_test_takers_l21_21353

-- Proposition: The minimum possible range in scores of the 3 test-takers
-- where the ranges of their scores in the 5 practice tests are 18, 26, and 32, is 76.
theorem min_range_of_three_test_takers (r1 r2 r3: ℕ) 
  (h1 : r1 = 18) (h2 : r2 = 26) (h3 : r3 = 32) : 
  (r1 + r2 + r3) = 76 := by
  sorry

end min_range_of_three_test_takers_l21_21353


namespace find_n_l21_21945

theorem find_n (n : ℕ) :
  (∀ k : ℕ, k > 0 → k^2 + (n / k^2) ≥ 1991) ∧ (∃ k : ℕ, k > 0 ∧ k^2 + (n / k^2) < 1992) ↔ 967 * 1024 ≤ n ∧ n < 968 * 1024 :=
by
  sorry

end find_n_l21_21945


namespace base_8_to_base_10_2671_to_1465_l21_21733

theorem base_8_to_base_10_2671_to_1465 :
  (2 * 8^3 + 6 * 8^2 + 7 * 8^1 + 1 * 8^0) = 1465 := by
  sorry

end base_8_to_base_10_2671_to_1465_l21_21733


namespace inverse_proposition_true_l21_21136

theorem inverse_proposition_true (x : ℝ) (h : x > 1 → x^2 > 1) : x^2 ≤ 1 → x ≤ 1 :=
by
  intros h₂
  sorry

end inverse_proposition_true_l21_21136


namespace part1_part2_l21_21274

-- Definition of the function f(x)
def f (x : ℝ) : ℝ := |2 * x| + |2 * x - 3|

-- Part 1: Proving the inequality solution
theorem part1 (x : ℝ) (h : f x ≤ 5) :
  -1/2 ≤ x ∧ x ≤ 2 :=
sorry

-- Part 2: Proving the range of m
theorem part2 (x₀ m : ℝ) (h1 : x₀ ∈ Set.Ici 1)
  (h2 : f x₀ + m ≤ x₀ + 3/x₀) :
  m ≤ 1 :=
sorry

end part1_part2_l21_21274


namespace equation_solutions_exist_l21_21000

theorem equation_solutions_exist (d x y : ℤ) (hx : Odd x) (hy : Odd y)
  (hxy : x^2 - d * y^2 = -4) : ∃ X Y : ℕ, X^2 - d * Y^2 = -1 :=
by
  sorry  -- Proof is omitted as per the instructions

end equation_solutions_exist_l21_21000


namespace liters_to_cubic_decimeters_eq_l21_21445

-- Define the condition for unit conversion
def liter_to_cubic_decimeter : ℝ :=
  1 -- since 1 liter = 1 cubic decimeter

-- Prove the equality for the given quantities
theorem liters_to_cubic_decimeters_eq :
  1.5 = 1.5 * liter_to_cubic_decimeter :=
by
  -- Proof to be filled in
  sorry

end liters_to_cubic_decimeters_eq_l21_21445


namespace num_cows_l21_21842

-- Define the context
variable (C H L Heads : ℕ)

-- Define the conditions
axiom condition1 : L = 2 * Heads + 8
axiom condition2 : L = 4 * C + 2 * H
axiom condition3 : Heads = C + H

-- State the goal
theorem num_cows : C = 4 := by
  sorry

end num_cows_l21_21842


namespace rachel_reading_pages_l21_21302

theorem rachel_reading_pages (M T : ℕ) (hM : M = 10) (hT : T = 23) : T - M = 3 := 
by
  rw [hM, hT]
  norm_num
  sorry

end rachel_reading_pages_l21_21302


namespace part1_part2_l21_21178

def is_regressive_sequence (x : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, ∃ m : ℕ, x n + x (n + 2) - x (n + 1) = x m

theorem part1 (a : ℕ → ℝ) (h : ∀ n : ℕ, a n = 3 ^ n) :
  ¬ is_regressive_sequence a := by
  sorry

theorem part2 (b : ℕ → ℝ) (h_reg : is_regressive_sequence b) (h_inc : ∀ n : ℕ, b n < b (n + 1)) :
  ∃ d : ℝ, ∀ n : ℕ, b (n + 1) = b n + d := by
  sorry

end part1_part2_l21_21178


namespace calculation_l21_21574

theorem calculation : (3 * 4 * 5) * ((1 / 3 : ℚ) + (1 / 4 : ℚ) - (1 / 5 : ℚ)) = 23 := by
  sorry

end calculation_l21_21574


namespace perimeter_of_garden_l21_21075

-- Define the area of the square garden
def area_square_garden : ℕ := 49

-- Define the relationship between q and p
def q_equals_p_plus_21 (q p : ℕ) : Prop := q = p + 21

-- Define the length of the side of the square garden
def side_length (area : ℕ) : ℕ := Nat.sqrt area

-- Define the perimeter of the square garden
def perimeter (side_length : ℕ) : ℕ := 4 * side_length

-- Define the perimeter of the square garden as a specific perimeter
def specific_perimeter (side_length : ℕ) : ℕ := perimeter side_length

-- Statement of the theorem
theorem perimeter_of_garden (q p : ℕ) (h1 : q = 49) (h2 : q_equals_p_plus_21 q p) : 
  specific_perimeter (side_length 49) = 28 := by
  sorry

end perimeter_of_garden_l21_21075


namespace min_value_of_2x_plus_4y_l21_21028

noncomputable def minimum_value (x y : ℝ) : ℝ := 2^x + 4^y

theorem min_value_of_2x_plus_4y (x y : ℝ) (h : x + 2 * y = 3) : minimum_value x y = 4 * Real.sqrt 2 :=
by
  sorry

end min_value_of_2x_plus_4y_l21_21028


namespace find_frac_sin_cos_l21_21100

theorem find_frac_sin_cos (α : ℝ) 
  (h : Real.sin (3 * Real.pi + α) = 2 * Real.sin (3 * Real.pi / 2 + α)) : 
  (Real.sin α - 4 * Real.cos α) / (5 * Real.sin α + 2 * Real.cos α) = -1 / 6 :=
by
  sorry

end find_frac_sin_cos_l21_21100


namespace tangent_line_ln_curve_l21_21002

theorem tangent_line_ln_curve (a : ℝ) :
  (∃ x y : ℝ, y = Real.log x + a ∧ x - y + 1 = 0 ∧ (∀ t : ℝ, t = x → (t - (Real.log t + a)) = -(1 - a))) → a = 2 :=
by
  sorry

end tangent_line_ln_curve_l21_21002


namespace complex_number_equality_l21_21513

def is_imaginary_unit (i : ℂ) : Prop :=
  i^2 = -1

theorem complex_number_equality (a b : ℝ) (i : ℂ) (h1 : is_imaginary_unit i) (h2 : (a + 4 * i) * i = b + i) : a + b = -3 :=
sorry

end complex_number_equality_l21_21513


namespace difference_of_squares_65_35_l21_21413

theorem difference_of_squares_65_35 :
  let a := 65
  let b := 35
  a^2 - b^2 = (a + b) * (a - b) :=
by
  sorry

end difference_of_squares_65_35_l21_21413


namespace first_day_exceeds_200_l21_21317

-- Bacteria population doubling function
def bacteria_population (n : ℕ) : ℕ := 4 * 3 ^ n

-- Prove the smallest day where bacteria count exceeds 200 is 4
theorem first_day_exceeds_200 : ∃ n : ℕ, bacteria_population n > 200 ∧ ∀ m < n, bacteria_population m ≤ 200 :=
by 
    -- Proof will be filled here
    sorry

end first_day_exceeds_200_l21_21317


namespace sum_of_products_l21_21128

theorem sum_of_products : 4 * 7 + 5 * 12 + 6 * 4 + 7 * 5 = 147 := by
  sorry

end sum_of_products_l21_21128


namespace num_balls_picked_l21_21916

-- Definitions based on the conditions
def numRedBalls : ℕ := 4
def numBlueBalls : ℕ := 3
def numGreenBalls : ℕ := 2
def totalBalls : ℕ := numRedBalls + numBlueBalls + numGreenBalls
def probFirstRed : ℚ := numRedBalls / totalBalls
def probSecondRed : ℚ := (numRedBalls - 1) / (totalBalls - 1)

-- Theorem stating the problem
theorem num_balls_picked :
  probFirstRed * probSecondRed = 1 / 6 → 
  (∃ (n : ℕ), n = 2) :=
by 
  sorry

end num_balls_picked_l21_21916


namespace range_of_function_l21_21433

theorem range_of_function :
  ∀ y : ℝ, ∃ x : ℝ, (x ≤ 1/2) ∧ (y = 2 * x - Real.sqrt (1 - 2 * x)) ↔ y ∈ Set.Iic 1 := 
by
  sorry

end range_of_function_l21_21433


namespace integral_part_odd_l21_21377

theorem integral_part_odd (n : ℕ) (hn : 0 < n) : 
  ∃ m : ℕ, (⌊(3 + Real.sqrt 5)^n⌋ = 2 * m + 1) := 
by
  -- Sorry used since the proof steps are not required in the task
  sorry

end integral_part_odd_l21_21377


namespace scientific_notation_of_2270000_l21_21596

theorem scientific_notation_of_2270000 : 
  (2270000 : ℝ) = 2.27 * 10^6 :=
sorry

end scientific_notation_of_2270000_l21_21596


namespace female_students_in_first_class_l21_21470

theorem female_students_in_first_class
  (females_in_second_class : ℕ)
  (males_in_first_class : ℕ)
  (males_in_second_class : ℕ)
  (males_in_third_class : ℕ)
  (females_in_third_class : ℕ)
  (extra_students : ℕ)
  (total_students_need_partners : ℕ)
  (total_males : ℕ := males_in_first_class + males_in_second_class + males_in_third_class)
  (total_females : ℕ := females_in_second_class + females_in_third_class)
  (females_in_first_class : ℕ)
  (females : ℕ := females_in_first_class + total_females) :
  (females_in_second_class = 18) →
  (males_in_first_class = 17) →
  (males_in_second_class = 14) →
  (males_in_third_class = 15) →
  (females_in_third_class = 17) →
  (extra_students = 2) →
  (total_students_need_partners = total_males - extra_students) →
  females = total_students_need_partners →
  females_in_first_class = 9 :=
by
  intros
  sorry

end female_students_in_first_class_l21_21470


namespace least_number_to_divisible_by_11_l21_21624

theorem least_number_to_divisible_by_11 (n : ℕ) (h : n = 11002) : ∃ k : ℕ, (n + k) % 11 = 0 ∧ ∀ m : ℕ, (n + m) % 11 = 0 → m ≥ k :=
by
  sorry

end least_number_to_divisible_by_11_l21_21624


namespace find_x_l21_21303

-- Define the conditions
def condition (x : ℕ) := (4 * x)^2 - 2 * x = 8062

-- State the theorem
theorem find_x : ∃ x : ℕ, condition x ∧ x = 134 := sorry

end find_x_l21_21303


namespace minimum_stamps_combination_l21_21352

theorem minimum_stamps_combination (c f : ℕ) (h : 3 * c + 4 * f = 30) :
  c + f = 8 :=
sorry

end minimum_stamps_combination_l21_21352


namespace number_less_than_neg_two_l21_21912

theorem number_less_than_neg_two : ∃ x : Int, x = -2 - 1 := 
by
  use -3
  sorry

end number_less_than_neg_two_l21_21912


namespace g_h_value_l21_21704

def g (x : ℕ) : ℕ := 3 * x^2 + 2
def h (x : ℕ) : ℕ := 5 * x^3 - 2

theorem g_h_value : g (h 2) = 4334 := by
  sorry

end g_h_value_l21_21704


namespace polynomial_factorization_l21_21816

theorem polynomial_factorization (m : ℝ) : 
  (∀ x : ℝ, (x - 7) * (x + 5) = x^2 + mx - 35) → m = -2 :=
by
  sorry

end polynomial_factorization_l21_21816


namespace johns_speed_final_push_l21_21726

-- Definitions for the given conditions
def john_behind_steve : ℝ := 14
def steve_speed : ℝ := 3.7
def john_ahead_steve : ℝ := 2
def john_final_push_time : ℝ := 32

-- Proving the statement
theorem johns_speed_final_push : 
  (∃ (v : ℝ), v * john_final_push_time = steve_speed * john_final_push_time + john_behind_steve + john_ahead_steve) -> 
  ∃ (v : ℝ), v = 4.2 :=
by
  sorry

end johns_speed_final_push_l21_21726


namespace sum_of_first_odd_numbers_l21_21506

theorem sum_of_first_odd_numbers (S1 S2 : ℕ) (n1 n2 : ℕ)
  (hS1 : S1 = n1^2) 
  (hS2 : S2 = n2^2) 
  (h1 : S1 = 2500)
  (h2 : S2 = 5625) : 
  n2 = 75 := by
  sorry

end sum_of_first_odd_numbers_l21_21506


namespace find_m_value_l21_21693

theorem find_m_value (x: ℝ) (m: ℝ) (hx: x > 2) (hm: m > 0) (h_min: ∀ y, (y = x + m / (x - 2)) → y ≥ 6) : m = 4 := 
sorry

end find_m_value_l21_21693


namespace amy_minimum_disks_l21_21036

theorem amy_minimum_disks :
  ∃ (d : ℕ), (d = 19) ∧ ( ∀ (f : ℕ), 
  (f = 40) ∧ ( ∀ (n m k : ℕ), 
  (n + m + k = f) ∧ ( ∀ (a b c : ℕ),
  (a = 8) ∧ (b = 15) ∧ (c = (f - a - b))
  ∧ ( ∀ (size_a size_b size_c : ℚ),
  (size_a = 0.6) ∧ (size_b = 0.55) ∧ (size_c = 0.45)
  ∧ ( ∀ (disk_space : ℚ),
  (disk_space = 1.44)
  ∧ ( ∀ (x y z : ℕ),
  (x = n * ⌈size_a / disk_space⌉) 
  ∧ (y = m * ⌈size_b / disk_space⌉) 
  ∧ (z = k * ⌈size_c / disk_space⌉)
  ∧ (x + y + z = d)) ∧ (size_a * a + size_b * b + size_c * c ≤ disk_space * d)))))) := sorry

end amy_minimum_disks_l21_21036


namespace is_factorization_l21_21654

-- Define the conditions
def A_transformation : Prop := (∀ x : ℝ, (x + 1) * (x - 1) = x ^ 2 - 1)
def B_transformation : Prop := (∀ m : ℝ, m ^ 2 + m - 4 = (m + 3) * (m - 2) + 2)
def C_transformation : Prop := (∀ x : ℝ, x ^ 2 + 2 * x = x * (x + 2))
def D_transformation : Prop := (∀ x : ℝ, 2 * x ^ 2 + 2 * x = 2 * x ^ 2 * (1 + (1 / x)))

-- The goal is to prove that transformation C is a factorization
theorem is_factorization : C_transformation :=
by
  sorry

end is_factorization_l21_21654


namespace units_digit_of_power_435_l21_21323

def units_digit_cycle (n : ℕ) : ℕ :=
  n % 2

def units_digit_of_four_powers (cycle : ℕ) : ℕ :=
  if cycle = 0 then 6 else 4

theorem units_digit_of_power_435 : 
  units_digit_of_four_powers (units_digit_cycle (3^5)) = 4 :=
by
  sorry

end units_digit_of_power_435_l21_21323


namespace initial_ratio_men_to_women_l21_21891

theorem initial_ratio_men_to_women (M W : ℕ) (h1 : (W - 3) * 2 = 24) (h2 : 14 - 2 = M) : M / gcd M W = 4 ∧ W / gcd M W = 5 := by 
  sorry

end initial_ratio_men_to_women_l21_21891


namespace odometer_reading_before_trip_l21_21359

-- Define the given conditions
def odometer_reading_lunch : ℝ := 372.0
def miles_traveled : ℝ := 159.7

-- Theorem to prove that the odometer reading before the trip was 212.3 miles
theorem odometer_reading_before_trip : odometer_reading_lunch - miles_traveled = 212.3 := by
  sorry

end odometer_reading_before_trip_l21_21359


namespace problem_l21_21636

noncomputable def F (x : ℝ) : ℝ :=
  (1 + x^2 - x^3) / (2 * x * (1 - x))

theorem problem (x : ℝ) (hx0 : x ≠ 0) (hx1 : x ≠ 1) :
  F x + F ((x - 1) / x) = 1 + x :=
by
  sorry

end problem_l21_21636


namespace minimum_photos_l21_21793

-- Given conditions:
-- 1. There are exactly three monuments.
-- 2. A group of 42 tourists visited the city.
-- 3. Each tourist took at most one photo of each of the three monuments.
-- 4. Any two tourists together had photos of all three monuments.

theorem minimum_photos (n m : ℕ) (h₁ : n = 42) (h₂ : m = 3) 
  (photographs : ℕ → ℕ → Bool) -- whether tourist i took a photo of monument j
  (h₃ : ∀ i : ℕ, i < n → ∀ j : ℕ, j < m → photographs i j = tt ∨ photographs i j = ff) -- each tourist took at most one photo of each monument
  (h₄ : ∀ i₁ i₂ : ℕ, i₁ < n → i₂ < n → i₁ ≠ i₂ → ∀ j : ℕ, j < m → photographs i₁ j = tt ∨ photographs i₂ j = tt) -- any two tourists together had photos of all three monuments
  : ∃ min_photos : ℕ, min_photos = 123 :=
by
  sorry

end minimum_photos_l21_21793


namespace max_area_l21_21963

noncomputable def PA : ℝ := 3
noncomputable def PB : ℝ := 4
noncomputable def PC : ℝ := 5
noncomputable def BC : ℝ := 6

theorem max_area (PA PB PC BC : ℝ) (hPA : PA = 3) (hPB : PB = 4) (hPC : PC = 5) (hBC : BC = 6) : 
  ∃ (A B C : Type) (area_ABC : ℝ), area_ABC = 19 := 
by 
  sorry

end max_area_l21_21963


namespace shooting_average_l21_21772

noncomputable def total_points (a b c d : ℕ) : ℕ :=
  (a * 10) + (b * 9) + (c * 8) + (d * 7)

noncomputable def average_points (total : ℕ) (shots : ℕ) : ℚ :=
  total / shots

theorem shooting_average :
  let a := 1
  let b := 4
  let c := 3
  let d := 2
  let shots := 10
  total_points a b c d = 84 ∧
  average_points (total_points a b c d) shots = 8.4 :=
by {
  sorry
}

end shooting_average_l21_21772


namespace total_tissues_used_l21_21310

-- Definitions based on the conditions
def initial_tissues := 97
def remaining_tissues := 47
def alice_tissues := 12
def bob_tissues := 2 * alice_tissues
def eve_tissues := alice_tissues - 3
def carol_tissues := initial_tissues - remaining_tissues
def friends_tissues := alice_tissues + bob_tissues + eve_tissues

-- The theorem to prove
theorem total_tissues_used : carol_tissues + friends_tissues = 95 := sorry

end total_tissues_used_l21_21310


namespace num_regular_soda_l21_21072

theorem num_regular_soda (t d r : ℕ) (h₁ : t = 17) (h₂ : d = 8) (h₃ : r = t - d) : r = 9 :=
by
  rw [h₁, h₂] at h₃
  exact h₃

end num_regular_soda_l21_21072


namespace cookie_radius_l21_21205

theorem cookie_radius (x y : ℝ) (h : x^2 + y^2 + 2 * x - 4 * y = 4) : 
  ∃ r : ℝ, (x + 1)^2 + (y - 2)^2 = r^2 ∧ r = 3 := by
  sorry

end cookie_radius_l21_21205


namespace john_income_l21_21765

theorem john_income 
  (john_tax_rate : ℝ) (ingrid_tax_rate : ℝ) (ingrid_income : ℝ) (combined_tax_rate : ℝ)
  (jt_30 : john_tax_rate = 0.30) (it_40 : ingrid_tax_rate = 0.40) (ii_72000 : ingrid_income = 72000) 
  (ctr_35625 : combined_tax_rate = 0.35625) :
  ∃ J : ℝ, (0.30 * J + ingrid_tax_rate * ingrid_income = combined_tax_rate * (J + ingrid_income)) ∧ (J = 56000) :=
by
  sorry

end john_income_l21_21765


namespace sqrt_expression_range_l21_21835

theorem sqrt_expression_range (x : ℝ) : x + 3 ≥ 0 ∧ x ≠ 0 ↔ x ≥ -3 ∧ x ≠ 0 :=
by
  sorry

end sqrt_expression_range_l21_21835


namespace gravitational_force_at_300000_l21_21358

-- Definitions and premises
def gravitational_force (d : ℝ) : ℝ := sorry

axiom inverse_square_law (d : ℝ) (f : ℝ) (k : ℝ) : f * d^2 = k

axiom surface_force : gravitational_force 5000 = 800

-- Goal: Prove the gravitational force at 300,000 miles
theorem gravitational_force_at_300000 : gravitational_force 300000 = 1 / 45 := sorry

end gravitational_force_at_300000_l21_21358


namespace velocity_at_t_10_time_to_reach_max_height_max_height_l21_21622

-- Define the height function H(t)
def H (t : ℝ) : ℝ := 200 * t - 4.9 * t^2

-- Define the velocity function v(t) as the derivative of H(t)
def v (t : ℝ) : ℝ := 200 - 9.8 * t

-- Theorem: The velocity of the body at t = 10 seconds
theorem velocity_at_t_10 : v 10 = 102 := by
  sorry

-- Theorem: The time to reach maximum height
theorem time_to_reach_max_height : (∃ t : ℝ, v t = 0 ∧ t = 200 / 9.8) := by
  sorry

-- Theorem: The maximum height the body will reach
theorem max_height : H (200 / 9.8) = 2040.425 := by
  sorry

end velocity_at_t_10_time_to_reach_max_height_max_height_l21_21622


namespace booth_makes_50_per_day_on_popcorn_l21_21545

-- Define the conditions as provided
def daily_popcorn_revenue (P : ℝ) : Prop :=
  let cotton_candy_revenue := 3 * P
  let total_days := 5
  let rent := 30
  let ingredients := 75
  let total_expenses := rent + ingredients
  let profit := 895
  let total_revenue_before_expenses := profit + total_expenses
  total_revenue_before_expenses = 20 * P 

theorem booth_makes_50_per_day_on_popcorn : daily_popcorn_revenue 50 :=
  by sorry

end booth_makes_50_per_day_on_popcorn_l21_21545


namespace difference_between_blue_and_red_balls_l21_21238

-- Definitions and conditions
def number_of_blue_balls := ℕ
def number_of_red_balls := ℕ
def difference_between_balls (m n : ℕ) := m - n

-- Problem statement: Prove that the difference between number_of_blue_balls and number_of_red_balls
-- can be any natural number greater than 1.
theorem difference_between_blue_and_red_balls (m n : ℕ) (h1 : m > n) (h2 : 
  let P_same := (n * (n - 1) + m * (m - 1)) / ((n + m) * (n + m - 1))
  let P_diff := 2 * (n * m) / ((n + m) * (n + m - 1))
  P_same = P_diff
  ) : ∃ a : ℕ, a > 1 ∧ a = m - n :=
by
  sorry

end difference_between_blue_and_red_balls_l21_21238


namespace geometric_sequence_common_ratio_l21_21337

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ)
  (h1 : S 2 = 2 * a 2 + 3)
  (h2 : S 3 = 2 * a 3 + 3)
  (h3 : ∀ n, S n = a 1 * (1 - q ^ n) / (1 - q)) : q = 2 := 
by
  sorry

end geometric_sequence_common_ratio_l21_21337


namespace gnomes_cannot_cross_l21_21863

theorem gnomes_cannot_cross :
  ∀ (gnomes : List ℕ), 
    (∀ g, g ∈ gnomes → g ∈ (List.range 100).map (λ x => x + 1)) →
    List.sum gnomes = 5050 → 
    ∀ (boat_capacity : ℕ), boat_capacity = 100 →
    ∀ (k : ℕ), (200 * (k + 1) - k^2 = 10100) → false :=
by
  intros gnomes H_weights H_sum boat_capacity H_capacity k H_equation
  sorry

end gnomes_cannot_cross_l21_21863


namespace negation_of_existence_implies_universal_l21_21362

theorem negation_of_existence_implies_universal (x : ℝ) :
  (∀ x : ℝ, ¬(x^2 ≤ |x|)) ↔ (∀ x : ℝ, x^2 > |x|) :=
by 
  sorry

end negation_of_existence_implies_universal_l21_21362


namespace smallest_value_l21_21608

theorem smallest_value (y : ℝ) (hy : 0 < y ∧ y < 1) :
  y^3 < y^2 ∧ y^3 < 3*y ∧ y^3 < (y)^(1/3:ℝ) ∧ y^3 < (1/y) :=
sorry

end smallest_value_l21_21608


namespace checkered_board_cut_l21_21378

def can_cut_equal_squares (n : ℕ) : Prop :=
  n % 5 = 0 ∧ n > 5

theorem checkered_board_cut (n : ℕ) (h : n % 5 = 0 ∧ n > 5) :
  ∃ m, n^2 = 5 * m :=
by
  sorry

end checkered_board_cut_l21_21378


namespace original_number_l21_21907

theorem original_number (x : ℤ) (h : 5 * x - 9 = 51) : x = 12 :=
sorry

end original_number_l21_21907


namespace amy_created_albums_l21_21355

theorem amy_created_albums (total_photos : ℕ) (photos_per_album : ℕ) 
  (h1 : total_photos = 180)
  (h2 : photos_per_album = 20) : 
  (total_photos / photos_per_album = 9) :=
by
  sorry

end amy_created_albums_l21_21355


namespace range_of_a_l21_21588

noncomputable def f : ℝ → ℝ := sorry

theorem range_of_a (h_odd : ∀ x, f (-x) = -f x) 
  (h_period : ∀ x, f (x + 3) = f x)
  (h1 : f 1 > 1) 
  (h2018 : f 2018 = (a : ℝ) ^ 2 - 5) : 
  -2 < a ∧ a < 2 :=
by
  sorry

end range_of_a_l21_21588


namespace kristin_runs_around_l21_21919

-- Definitions of the conditions.
def kristin_runs_faster (v_k v_s : ℝ) : Prop := v_k = 3 * v_s
def sarith_runs_times (S : ℕ) : Prop := S = 8
def field_length (c_field a_field : ℝ) : Prop := c_field = a_field / 2

-- The question is to prove Kristin runs around the field 12 times.
def kristin_runs_times (K : ℕ) : Prop := K = 12

-- The main theorem statement combining conditions to prove the question.
theorem kristin_runs_around :
  ∀ (v_k v_s c_field a_field : ℝ) (S K : ℕ),
    kristin_runs_faster v_k v_s →
    sarith_runs_times S →
    field_length c_field a_field →
    K = (S : ℝ) * (3 / 2) →
    kristin_runs_times K :=
by sorry

end kristin_runs_around_l21_21919


namespace triangle_height_l21_21214

theorem triangle_height (area : ℝ) (base : ℝ) (height : ℝ) 
  (h_area : area = 615) (h_base : base = 123) 
  (area_formula : area = (base * height) / 2) : height = 10 := 
by 
  sorry

end triangle_height_l21_21214


namespace rectangle_ratio_ratio_simplification_l21_21107

theorem rectangle_ratio (w : ℕ) (h : w + 10 = 10) (p : 2 * w + 2 * 10 = 30) :
  w = 5 := by
  sorry

theorem ratio_simplification (x y : ℕ) (h : x * 10 = y * 5) (rel_prime : Nat.gcd x y = 1) :
  (x, y) = (1, 2) := by
  sorry

end rectangle_ratio_ratio_simplification_l21_21107


namespace barrel_capacity_is_16_l21_21110

noncomputable def capacity_of_barrel (midway_tap_rate bottom_tap_rate used_bottom_tap_early_time assistant_use_time : Nat) : Nat :=
  let midway_draw := used_bottom_tap_early_time / midway_tap_rate
  let bottom_draw_assistant := assistant_use_time / bottom_tap_rate
  let total_extra_draw := midway_draw + bottom_draw_assistant
  2 * total_extra_draw

theorem barrel_capacity_is_16 :
  capacity_of_barrel 6 4 24 16 = 16 :=
by
  sorry

end barrel_capacity_is_16_l21_21110


namespace crayons_and_erasers_difference_l21_21794

theorem crayons_and_erasers_difference 
  (initial_crayons : ℕ) (initial_erasers : ℕ) (remaining_crayons : ℕ) 
  (h1 : initial_crayons = 601) (h2 : initial_erasers = 406) (h3 : remaining_crayons = 336) : 
  initial_erasers - remaining_crayons = 70 :=
by
  sorry

end crayons_and_erasers_difference_l21_21794


namespace no_solutions_for_inequalities_l21_21172

theorem no_solutions_for_inequalities (x y z t : ℝ) :
  |x| < |y - z + t| →
  |y| < |x - z + t| →
  |z| < |x - y + t| →
  |t| < |x - y + z| →
  False :=
by
  sorry

end no_solutions_for_inequalities_l21_21172


namespace find_coords_of_P_l21_21529

-- Definitions from the conditions
def line_eq (x y : ℝ) : Prop := x - y - 7 = 0
def is_midpoint (P Q M : ℝ × ℝ) : Prop := 
  M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- Coordinates given in the problem
def P : ℝ × ℝ := (-2, 1)

-- The proof goal
theorem find_coords_of_P : ∃ Q : ℝ × ℝ,
  is_midpoint P Q (1, -1) ∧ 
  line_eq Q.1 Q.2 :=
sorry

end find_coords_of_P_l21_21529


namespace representation_of_1_l21_21663

theorem representation_of_1 (x y z : ℕ) (h : 1 = 1/x + 1/y + 1/z) : 
  (x = 2 ∧ y = 3 ∧ z = 6) ∨ (x = 2 ∧ y = 4 ∧ z = 4) ∨ (x = 3 ∧ y = 3 ∧ z = 3) :=
by
  sorry

end representation_of_1_l21_21663


namespace total_carrots_l21_21690

def Joan_carrots : ℕ := 29
def Jessica_carrots : ℕ := 11

theorem total_carrots : Joan_carrots + Jessica_carrots = 40 := by
  sorry

end total_carrots_l21_21690


namespace find_speed_of_stream_l21_21487

theorem find_speed_of_stream (x : ℝ) (h1 : ∃ x, 1 / (39 - x) = 2 * (1 / (39 + x))) : x = 13 :=
by
sorry

end find_speed_of_stream_l21_21487


namespace problem_value_of_m_l21_21664

theorem problem_value_of_m (m : ℝ)
  (h1 : (m + 1) * x ^ (m ^ 2 - 3) = y)
  (h2 : m ^ 2 - 3 = 1)
  (h3 : m + 1 < 0) : 
  m = -2 := 
  sorry

end problem_value_of_m_l21_21664


namespace truck_distance_on_7_liters_l21_21521

-- Define the conditions
def truck_300_km_per_5_liters := 300
def liters_5 := 5
def liters_7 := 7
def expected_distance_7_liters := 420

-- The rate of distance (km per liter)
def rate := truck_300_km_per_5_liters / liters_5

-- Proof statement
theorem truck_distance_on_7_liters :
  rate * liters_7 = expected_distance_7_liters :=
  by
  sorry

end truck_distance_on_7_liters_l21_21521


namespace negation_of_existential_proposition_l21_21374

-- Define the propositions
def proposition (x : ℝ) := x^2 - 2 * x + 1 ≤ 0

-- Define the negation of the propositions
def negation_prop (x : ℝ) := x^2 - 2 * x + 1 > 0

-- Theorem to prove that the negation of the existential proposition is the universal proposition
theorem negation_of_existential_proposition
  (h : ¬ ∃ x : ℝ, proposition x) :
  ∀ x : ℝ, negation_prop x :=
by
  sorry

end negation_of_existential_proposition_l21_21374


namespace ratio_of_original_to_doubled_l21_21408

theorem ratio_of_original_to_doubled (x : ℕ) (h : x + 5 = 17) : (x / Nat.gcd x (2 * x)) = 1 ∧ ((2 * x) / Nat.gcd x (2 * x)) = 2 := 
by
  sorry

end ratio_of_original_to_doubled_l21_21408


namespace smaller_angle_measure_l21_21372

theorem smaller_angle_measure (x : ℝ) (h₁ : 5 * x + 3 * x = 180) : 3 * x = 67.5 :=
by
  sorry

end smaller_angle_measure_l21_21372


namespace gcd_three_numbers_l21_21192

def a : ℕ := 8650
def b : ℕ := 11570
def c : ℕ := 28980

theorem gcd_three_numbers : Nat.gcd (Nat.gcd a b) c = 10 :=
by 
  sorry

end gcd_three_numbers_l21_21192


namespace speed_in_kmh_l21_21020

def distance : ℝ := 550.044
def time : ℝ := 30
def conversion_factor : ℝ := 3.6

theorem speed_in_kmh : (distance / time) * conversion_factor = 66.00528 := 
by
  sorry

end speed_in_kmh_l21_21020


namespace birds_on_fence_l21_21237

theorem birds_on_fence :
  let i := 12           -- initial birds
  let added1 := 8       -- birds that land first
  let T := i + added1   -- total first stage birds
  
  let fly_away1 := 5
  let join1 := 3
  let W := T - fly_away1 + join1   -- birds after some fly away, others join
  
  let D := W * 2       -- birds doubles
  
  let fly_away2 := D * 0.25  -- 25% fly away
  let D_after_fly_away := D - fly_away2
  
  let return_birds := 2        -- 2.5 birds return, rounded down to 2
  let final_birds := D_after_fly_away + return_birds
  
  final_birds = 29 := 
by {
  sorry
}

end birds_on_fence_l21_21237


namespace minimum_benches_for_equal_occupancy_l21_21787

theorem minimum_benches_for_equal_occupancy (M : ℕ) :
  (∃ x y, x = y ∧ 8 * M = x ∧ 12 * M = y) ↔ M = 3 := by
  sorry

end minimum_benches_for_equal_occupancy_l21_21787


namespace sum_of_scores_l21_21825

/-- Prove that given the conditions on Bill, John, and Sue's scores, the total sum of the scores of the three students is 160. -/
theorem sum_of_scores (B J S : ℕ) (h1 : B = J + 20) (h2 : B = S / 2) (h3 : B = 45) : B + J + S = 160 :=
sorry

end sum_of_scores_l21_21825


namespace clock_spoke_angle_l21_21528

-- Define the parameters of the clock face and the problem.
def num_spokes := 10
def total_degrees := 360
def degrees_per_spoke := total_degrees / num_spokes
def position_3_oclock := 3 -- the third spoke
def halfway_45_oclock := 5 -- approximately the fifth spoke
def spokes_between := halfway_45_oclock - position_3_oclock
def smaller_angle := spokes_between * degrees_per_spoke
def expected_angle := 72

-- Statement of the problem
theorem clock_spoke_angle :
  smaller_angle = expected_angle := by
    -- Proof is omitted
    sorry

end clock_spoke_angle_l21_21528


namespace remainder_of_3_pow_800_mod_17_l21_21999

theorem remainder_of_3_pow_800_mod_17 :
    (3 ^ 800) % 17 = 1 :=
by
    sorry

end remainder_of_3_pow_800_mod_17_l21_21999


namespace percentage_B_of_C_l21_21752

theorem percentage_B_of_C 
  (A C B : ℝ)
  (h1 : A = (7 / 100) * C)
  (h2 : A = (50 / 100) * B) :
  B = (14 / 100) * C := 
sorry

end percentage_B_of_C_l21_21752


namespace total_spent_l21_21045

theorem total_spent (puppy_cost dog_food_cost treats_cost_per_bag toys_cost crate_cost bed_cost collar_leash_cost bags_of_treats discount_rate : ℝ) :
  puppy_cost = 20 →
  dog_food_cost = 20 →
  treats_cost_per_bag = 2.5 →
  toys_cost = 15 →
  crate_cost = 20 →
  bed_cost = 20 →
  collar_leash_cost = 15 →
  bags_of_treats = 2 →
  discount_rate = 0.2 →
  (dog_food_cost + treats_cost_per_bag * bags_of_treats + toys_cost + crate_cost + bed_cost + collar_leash_cost) * (1 - discount_rate) + puppy_cost = 96 :=
by sorry

end total_spent_l21_21045


namespace sin_inverse_equation_l21_21412

noncomputable def a := Real.arcsin (4/5)
noncomputable def b := Real.arctan 1
noncomputable def c := Real.arccos (1/3)
noncomputable def sin_a_plus_b_minus_c := Real.sin (a + b - c)

theorem sin_inverse_equation : sin_a_plus_b_minus_c = 11 / 15 := sorry

end sin_inverse_equation_l21_21412


namespace quadratic_decreasing_on_nonneg_real_l21_21044

theorem quadratic_decreasing_on_nonneg_real (a b c : ℝ) (h_a : a < 0) (h_b : b < 0) : 
  ∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x ≤ y → (a * x^2 + b * x + c) ≥ (a * y^2 + b * y + c) :=
by
  sorry

end quadratic_decreasing_on_nonneg_real_l21_21044


namespace janet_miles_per_day_l21_21601

def total_miles : ℕ := 72
def days : ℕ := 9
def miles_per_day : ℕ := 8

theorem janet_miles_per_day : total_miles / days = miles_per_day :=
by {
  sorry
}

end janet_miles_per_day_l21_21601


namespace man_speed_upstream_l21_21818

def man_speed_still_water : ℕ := 50
def speed_downstream : ℕ := 80

theorem man_speed_upstream : (man_speed_still_water - (speed_downstream - man_speed_still_water)) = 20 :=
by
  sorry

end man_speed_upstream_l21_21818


namespace christine_aquafaba_needed_l21_21520

-- Define the number of tablespoons per egg white
def tablespoons_per_egg_white : ℕ := 2

-- Define the number of egg whites per cake
def egg_whites_per_cake : ℕ := 8

-- Define the number of cakes
def number_of_cakes : ℕ := 2

-- Express the total amount of aquafaba needed
def aquafaba_needed : ℕ :=
  tablespoons_per_egg_white * egg_whites_per_cake * number_of_cakes

-- Statement asserting the amount of aquafaba needed is 32
theorem christine_aquafaba_needed : aquafaba_needed = 32 := by
  sorry

end christine_aquafaba_needed_l21_21520


namespace parallel_lines_l21_21341

theorem parallel_lines (a : ℝ) :
  (∀ x y : ℝ, ax + 2 * y + 3 * a = 0 → 3 * x + (a - 1) * y = a - 7) → a = 3 :=
by sorry

end parallel_lines_l21_21341


namespace value_of_7x_minus_3y_l21_21613

theorem value_of_7x_minus_3y (x y : ℚ) (h1 : 4 * x + y = 8) (h2 : 3 * x - 4 * y = 5) : 
  7 * x - 3 * y = 247 / 19 := 
sorry

end value_of_7x_minus_3y_l21_21613


namespace two_triangles_not_separable_by_plane_l21_21069

/-- Definition of a point in three-dimensional space -/
structure Point (α : Type) :=
(x : α)
(y : α)
(z : α)

/-- Definition of a segment joining two points -/
structure Segment (α : Type) :=
(p1 : Point α)
(p2 : Point α)

/-- Definition of a triangle formed by three points -/
structure Triangle (α : Type) :=
(a : Point α)
(b : Point α)
(c : Point α)

/-- Definition of a plane given by a normal vector and a point on the plane -/
structure Plane (α : Type) :=
(n : Point α)
(p : Point α)

/-- Definition of separation of two triangles by a plane -/
def separates (plane : Plane ℝ) (t1 t2 : Triangle ℝ) : Prop :=
  -- Placeholder for the actual separation condition
  sorry

/-- The theorem to be proved -/
theorem two_triangles_not_separable_by_plane (points : Fin 6 → Point ℝ) :
  ∃ t1 t2 : Triangle ℝ, ¬∃ plane : Plane ℝ, separates plane t1 t2 :=
sorry

end two_triangles_not_separable_by_plane_l21_21069


namespace max_integer_value_of_f_l21_21604

noncomputable def f (x : ℝ) : ℝ := (4 * x^2 + 8 * x + 21) / (4 * x^2 + 8 * x + 5)

theorem max_integer_value_of_f :
  ∃ n : ℤ, n = 17 ∧ ∀ x : ℝ, f x ≤ (n : ℝ) :=
by
  sorry

end max_integer_value_of_f_l21_21604


namespace complement_A_correct_l21_21331

def A : Set ℝ := {x | 1 - (8 / (x - 2)) < 0}

def complement_A : Set ℝ := {x | x ≤ 2 ∨ x ≥ 10}

theorem complement_A_correct : (Aᶜ = complement_A) :=
by {
  -- Placeholder for the necessary proof
  sorry
}

end complement_A_correct_l21_21331


namespace integers_solution_l21_21982

theorem integers_solution (a b : ℤ) (S D : ℤ) 
  (h1 : S = a + b) (h2 : D = a - b) (h3 : S / D = 3) (h4 : S * D = 300) : 
  ((a = 20 ∧ b = 10) ∨ (a = -20 ∧ b = -10)) :=
by
  sorry

end integers_solution_l21_21982


namespace count_ordered_pairs_squares_diff_l21_21554

theorem count_ordered_pairs_squares_diff (m n : ℕ) (h1 : m ≥ n) (h2 : m^2 - n^2 = 72) : 
∃ (a : ℕ), a = 3 :=
sorry

end count_ordered_pairs_squares_diff_l21_21554


namespace find_wrong_observation_value_l21_21081

theorem find_wrong_observation_value :
  ∃ (wrong_value : ℝ),
    let n := 50
    let mean_initial := 36
    let mean_corrected := 36.54
    let observation_incorrect := 48
    let sum_initial := n * mean_initial
    let sum_corrected := n * mean_corrected
    let difference := sum_corrected - sum_initial
    wrong_value = observation_incorrect - difference := sorry

end find_wrong_observation_value_l21_21081


namespace tony_water_trips_calculation_l21_21502

noncomputable def tony_drinks_water_after_every_n_trips (bucket_capacity_sand : ℤ) 
                                                        (sandbox_depth : ℤ) (sandbox_width : ℤ) 
                                                        (sandbox_length : ℤ) (sand_weight_cubic_foot : ℤ) 
                                                        (water_consumption : ℤ) (water_bottle_ounces : ℤ) 
                                                        (water_bottle_cost : ℤ) (money_with_tony : ℤ) 
                                                        (expected_change : ℤ) : ℤ :=
  let volume_sandbox := sandbox_depth * sandbox_width * sandbox_length
  let total_sand_weight := volume_sandbox * sand_weight_cubic_foot
  let trips_needed := total_sand_weight / bucket_capacity_sand
  let money_spent_on_water := money_with_tony - expected_change
  let water_bottles_bought := money_spent_on_water / water_bottle_cost
  let total_water_ounces := water_bottles_bought * water_bottle_ounces
  let drinking_sessions := total_water_ounces / water_consumption
  trips_needed / drinking_sessions

theorem tony_water_trips_calculation : 
  tony_drinks_water_after_every_n_trips 2 2 4 5 3 3 15 2 10 4 = 4 := 
by 
  sorry

end tony_water_trips_calculation_l21_21502


namespace stock_values_l21_21325

theorem stock_values (AA_invest : ℕ) (BB_invest : ℕ) (CC_invest : ℕ)
  (AA_first_year_increase : ℝ) (BB_first_year_decrease : ℝ) (CC_first_year_change : ℝ)
  (AA_second_year_decrease : ℝ) (BB_second_year_increase : ℝ) (CC_second_year_increase : ℝ)
  (A_final : ℝ) (B_final : ℝ) (C_final : ℝ) :
  AA_invest = 150 → BB_invest = 100 → CC_invest = 50 →
  AA_first_year_increase = 1.10 → BB_first_year_decrease = 0.70 → CC_first_year_change = 1 →
  AA_second_year_decrease = 0.95 → BB_second_year_increase = 1.10 → CC_second_year_increase = 1.08 →
  A_final = (AA_invest * AA_first_year_increase) * AA_second_year_decrease →
  B_final = (BB_invest * BB_first_year_decrease) * BB_second_year_increase →
  C_final = (CC_invest * CC_first_year_change) * CC_second_year_increase →
  C_final < B_final ∧ B_final < A_final :=
by
  intros
  sorry

end stock_values_l21_21325


namespace xy_value_l21_21602

noncomputable def x (y : ℝ) : ℝ := 36 * y

theorem xy_value (y : ℝ) (h1 : y = 0.16666666666666666) : x y * y = 1 :=
by
  rw [h1, x]
  sorry

end xy_value_l21_21602


namespace original_radius_new_perimeter_l21_21667

variable (r : ℝ)

theorem original_radius_new_perimeter (h : (π * (r + 5)^2 = 4 * π * r^2)) :
  r = 5 ∧ 2 * π * (r + 5) = 20 * π :=
by
  sorry

end original_radius_new_perimeter_l21_21667


namespace find_y_l21_21756

theorem find_y (y : ℝ) (h : |2 * y - 44| + |y - 24| = |3 * y - 66|) : y = 23 := 
by 
  sorry

end find_y_l21_21756


namespace hannah_sweatshirts_l21_21789

theorem hannah_sweatshirts (S : ℕ) (h1 : 15 * S + 2 * 10 = 65) : S = 3 := 
by
  sorry

end hannah_sweatshirts_l21_21789


namespace exchange_5_dollars_to_francs_l21_21292

-- Define the exchange rates
def dollar_to_lire (d : ℕ) : ℕ := d * 5000
def lire_to_francs (l : ℕ) : ℕ := (l / 1000) * 3

-- Define the main theorem
theorem exchange_5_dollars_to_francs : lire_to_francs (dollar_to_lire 5) = 75 :=
by
  sorry

end exchange_5_dollars_to_francs_l21_21292


namespace range_of_a_l21_21894

noncomputable def y (x : ℝ) (a : ℝ) : ℝ := Real.log x + a * x ^ 2 - 2 * x

noncomputable def y' (x : ℝ) (a : ℝ) : ℝ := 1 / x + 2 * a * x - 2

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 < x → y' x a ≥ 0) ↔ a ≥ 1 / 2 :=
by
  sorry

end range_of_a_l21_21894


namespace rationalize_denominator_l21_21156

theorem rationalize_denominator (cbrt : ℝ → ℝ) (h₁ : cbrt 81 = 3 * cbrt 3) :
  1 / (cbrt 3 + cbrt 81) = cbrt 9 / 12 :=
sorry

end rationalize_denominator_l21_21156


namespace number_added_to_x_is_2_l21_21091

/-- Prove that in a set of integers {x, x + y, x + 4, x + 7, x + 22}, 
    where the mean is 3 greater than the median, the number added to x 
    to get the second integer is 2. --/

theorem number_added_to_x_is_2 (x y : ℤ) (h_pos : 0 < x ∧ 0 < y) 
  (h_median : (x + 4) = ((x + y) + (x + (x + y) + (x + 4) + (x + 7) + (x + 22)) / 5 - 3)) : 
  y = 2 := by
  sorry

end number_added_to_x_is_2_l21_21091


namespace lamps_on_bridge_l21_21761

theorem lamps_on_bridge (bridge_length : ℕ) (lamp_spacing : ℕ) (num_intervals : ℕ) (num_lamps : ℕ) 
  (h1 : bridge_length = 30) 
  (h2 : lamp_spacing = 5)
  (h3 : num_intervals = bridge_length / lamp_spacing)
  (h4 : num_lamps = num_intervals + 1) :
  num_lamps = 7 := 
by
  sorry

end lamps_on_bridge_l21_21761


namespace chocolate_bars_sold_last_week_l21_21492

-- Definitions based on conditions
def initial_chocolate_bars : Nat := 18
def chocolate_bars_sold_this_week : Nat := 7
def chocolate_bars_needed_to_sell : Nat := 6

-- Define the number of chocolate bars sold so far
def chocolate_bars_sold_so_far : Nat := chocolate_bars_sold_this_week + chocolate_bars_needed_to_sell

-- Target statement to prove
theorem chocolate_bars_sold_last_week :
  initial_chocolate_bars - chocolate_bars_sold_so_far = 5 :=
by
  sorry

end chocolate_bars_sold_last_week_l21_21492


namespace coronavirus_diameter_in_meters_l21_21309

theorem coronavirus_diameter_in_meters (n : ℕ) (h₁ : 1 = (10 : ℤ) ^ 9) (h₂ : n = 125) :
  (n * 10 ^ (-9 : ℤ) : ℝ) = 1.25 * 10 ^ (-7 : ℤ) :=
by
  sorry

end coronavirus_diameter_in_meters_l21_21309


namespace Gilda_marbles_left_l21_21474

theorem Gilda_marbles_left (M : ℝ) (h1 : M > 0) :
  let remaining_after_pedro := M - 0.30 * M
  let remaining_after_ebony := remaining_after_pedro - 0.40 * remaining_after_pedro
  remaining_after_ebony / M * 100 = 42 :=
by
  sorry

end Gilda_marbles_left_l21_21474


namespace find_k_l21_21877

theorem find_k (x y z k : ℝ) (h1 : 5 / (x + y) = k / (x + z)) (h2 : k / (x + z) = 9 / (z - y)) : k = 14 :=
by
  sorry

end find_k_l21_21877


namespace roots_quadratic_eq_l21_21472

theorem roots_quadratic_eq :
  (∃ a b : ℝ, (a + b = 8) ∧ (a * b = 8) ∧ (a^2 + b^2 = 48)) :=
sorry

end roots_quadratic_eq_l21_21472


namespace linear_avoid_third_quadrant_l21_21305

theorem linear_avoid_third_quadrant (k b : ℝ) (h : ∀ x : ℝ, k * x + b ≥ 0 → k * x + b > 0 → (k * x + b ≥ 0) ∧ (x ≥ 0)) :
  k < 0 ∧ b ≥ 0 :=
by
  sorry

end linear_avoid_third_quadrant_l21_21305


namespace ambulance_reachable_area_l21_21239

theorem ambulance_reachable_area :
  let travel_time_minutes := 8
  let travel_time_hours := (travel_time_minutes : ℝ) / 60
  let speed_on_road := 60 -- speed in miles per hour
  let speed_off_road := 10 -- speed in miles per hour
  let distance_on_road := speed_on_road * travel_time_hours
  distance_on_road = 8 → -- this verifies the distance covered on road
  let area := (2 * distance_on_road) ^ 2
  area = 256 := sorry

end ambulance_reachable_area_l21_21239


namespace negation_of_P_l21_21524

-- Define the proposition P
def P : Prop := ∀ x : ℝ, x^2 + 2*x + 2 > 0

-- State the negation of P
theorem negation_of_P : ¬P ↔ ∃ x : ℝ, x^2 + 2*x + 2 ≤ 0 :=
by
  sorry

end negation_of_P_l21_21524


namespace fg_of_1_l21_21477

def f (x : ℤ) : ℤ := x + 3
def g (x : ℤ) : ℤ := x^3 - x^2 - 6

theorem fg_of_1 : f (g 1) = -3 := by
  sorry

end fg_of_1_l21_21477


namespace Julie_hours_per_week_school_l21_21241

noncomputable def summer_rate : ℚ := 4500 / (36 * 10)

noncomputable def school_rate : ℚ := summer_rate * 1.10

noncomputable def total_school_hours_needed : ℚ := 9000 / school_rate

noncomputable def hours_per_week_school : ℚ := total_school_hours_needed / 40

theorem Julie_hours_per_week_school : hours_per_week_school = 16.36 := by
  sorry

end Julie_hours_per_week_school_l21_21241


namespace zoey_finishes_on_wednesday_l21_21503

noncomputable def day_zoey_finishes (n : ℕ) : String :=
  let total_days := (n * (n + 1)) / 2
  match total_days % 7 with
  | 0 => "Sunday"
  | 1 => "Monday"
  | 2 => "Tuesday"
  | 3 => "Wednesday"
  | 4 => "Thursday"
  | 5 => "Friday"
  | 6 => "Saturday"
  | _ => "Error"

theorem zoey_finishes_on_wednesday : day_zoey_finishes 18 = "Wednesday" :=
by
  -- Calculate that Zoey takes 171 days to read 18 books
  -- Recall that 171 mod 7 = 3, so she finishes on "Wednesday"
  sorry

end zoey_finishes_on_wednesday_l21_21503


namespace expression_value_l21_21565

theorem expression_value : 
  (Nat.factorial 10) / (2 * (Finset.sum (Finset.range 11) id)) = 33080 := by
  sorry

end expression_value_l21_21565


namespace ferris_wheel_seat_capacity_l21_21976

-- Define the given conditions
def people := 16
def seats := 4

-- Define the problem and the proof goal
theorem ferris_wheel_seat_capacity : people / seats = 4 := by
  sorry

end ferris_wheel_seat_capacity_l21_21976


namespace factor_is_2_l21_21194

variable (x : ℕ) (f : ℕ)

theorem factor_is_2 (h₁ : x = 36)
                    (h₂ : ((f * (x + 10)) / 2) - 2 = 44) : f = 2 :=
by {
  sorry
}

end factor_is_2_l21_21194


namespace quotient_calc_l21_21557

theorem quotient_calc (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ)
  (h_dividend : dividend = 139)
  (h_divisor : divisor = 19)
  (h_remainder : remainder = 6)
  (h_formula : dividend - remainder = quotient * divisor):
  quotient = 7 :=
by {
  -- Insert proof here
  sorry
}

end quotient_calc_l21_21557


namespace anna_correct_percentage_l21_21671

theorem anna_correct_percentage :
  let test1_problems := 30
  let test1_score := 0.75
  let test2_problems := 50
  let test2_score := 0.85
  let test3_problems := 20
  let test3_score := 0.65
  let correct_test1 := test1_score * test1_problems
  let correct_test2 := test2_score * test2_problems
  let correct_test3 := test3_score * test3_problems
  let total_problems := test1_problems + test2_problems + test3_problems
  let total_correct := correct_test1 + correct_test2 + correct_test3
  (total_correct / total_problems) * 100 = 78 :=
by
  sorry

end anna_correct_percentage_l21_21671


namespace abc_inequality_l21_21117

theorem abc_inequality (a b c : ℝ) : a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 := 
sorry

end abc_inequality_l21_21117


namespace mrs_hilt_has_more_money_l21_21197

/-- Mrs. Hilt has two pennies, two dimes, and two nickels. 
    Jacob has four pennies, one nickel, and one dime. 
    Prove that Mrs. Hilt has $0.13 more than Jacob. -/
theorem mrs_hilt_has_more_money 
  (hilt_pennies hilt_dimes hilt_nickels : ℕ)
  (jacob_pennies jacob_dimes jacob_nickels : ℕ)
  (value_penny value_nickel value_dime : ℝ)
  (H1 : hilt_pennies = 2) (H2 : hilt_dimes = 2) (H3 : hilt_nickels = 2)
  (H4 : jacob_pennies = 4) (H5 : jacob_dimes = 1) (H6 : jacob_nickels = 1)
  (H7 : value_penny = 0.01) (H8 : value_nickel = 0.05) (H9 : value_dime = 0.10) :
  ((hilt_pennies * value_penny + hilt_dimes * value_dime + hilt_nickels * value_nickel) 
   - (jacob_pennies * value_penny + jacob_dimes * value_dime + jacob_nickels * value_nickel) 
   = 0.13) :=
by sorry

end mrs_hilt_has_more_money_l21_21197


namespace non_neg_reals_inequality_l21_21083

theorem non_neg_reals_inequality (a b c : ℝ)
  (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c)
  (h₃ : a + b + c ≤ 3) :
  (a / (1 + a^2) + b / (1 + b^2) + c / (1 + c^2) ≤ 3/2) ∧
  (3/2 ≤ (1 / (1 + a) + 1 / (1 + b) + 1 / (1 + c))) :=
by
  sorry

end non_neg_reals_inequality_l21_21083


namespace original_price_of_computer_l21_21254

theorem original_price_of_computer (P : ℝ) (h1 : 1.30 * P = 364) (h2 : 2 * P = 560) : P = 280 :=
by 
  -- The proof is skipped as per instruction
  sorry

end original_price_of_computer_l21_21254


namespace dan_has_13_limes_l21_21517

theorem dan_has_13_limes (picked_limes : ℕ) (given_limes : ℕ) (h1 : picked_limes = 9) (h2 : given_limes = 4) : 
  picked_limes + given_limes = 13 := 
by
  sorry

end dan_has_13_limes_l21_21517


namespace simplified_expression_value_l21_21087

noncomputable def expression (a b : ℝ) : ℝ :=
  3 * a ^ 2 - b ^ 2 - (a ^ 2 - 6 * a) - 2 * (-b ^ 2 + 3 * a)

theorem simplified_expression_value :
  expression (-1/2) 3 = 19 / 2 :=
by
  sorry

end simplified_expression_value_l21_21087


namespace total_amount_in_bank_l21_21332

-- Definition of the checks and their values
def checks_1mil : Nat := 25
def checks_100k : Nat := 8
def value_1mil : Nat := 1000000
def value_100k : Nat := 100000

-- The proof statement
theorem total_amount_in_bank 
  (total : Nat) 
  (h1 : checks_1mil * value_1mil = 25000000)
  (h2 : checks_100k * value_100k = 800000):
  total = 25000000 + 800000 :=
sorry

end total_amount_in_bank_l21_21332


namespace prove_final_value_is_111_l21_21460

theorem prove_final_value_is_111 :
  let initial_num := 16
  let doubled_num := initial_num * 2
  let added_five := doubled_num + 5
  let trebled_result := added_five * 3
  trebled_result = 111 :=
by
  sorry

end prove_final_value_is_111_l21_21460


namespace slope_range_l21_21936

open Real

theorem slope_range (k : ℝ) :
  (∃ b : ℝ, 
    ∃ x1 x2 x3 : ℝ,
      (x1 + x2 + x3 = 0) ∧
      (x1 ≥ 0) ∧ (x2 ≥ 0) ∧ (x3 < 0) ∧
      ((kx1 + b) = ((x1 + 1) / (|x1| + 1))) ∧
      ((kx2 + b) = ((x2 + 1) / (|x2| + 1))) ∧
      ((kx3 + b) = ((x3 + 1) / (|x3| + 1)))) →
  (0 < k ∧ k < (2 / 9)) :=
sorry

end slope_range_l21_21936


namespace value_of_y_at_64_l21_21862

theorem value_of_y_at_64 (x y k : ℝ) (h1 : y = k * x^(1/3)) (h2 : 8^(1/3) = 2) (h3 : y = 4 ∧ x = 8):
  y = 8 :=
by {
  sorry
}

end value_of_y_at_64_l21_21862


namespace num_from_1_to_200_not_squares_or_cubes_l21_21208

noncomputable def numNonPerfectSquaresAndCubes (n : ℕ) : ℕ :=
  let num_squares := 14
  let num_cubes := 5
  let num_sixth_powers := 2
  n - (num_squares + num_cubes - num_sixth_powers)

theorem num_from_1_to_200_not_squares_or_cubes : numNonPerfectSquaresAndCubes 200 = 183 := by
  sorry

end num_from_1_to_200_not_squares_or_cubes_l21_21208


namespace find_angle_l21_21127

theorem find_angle (A : ℝ) (deg_to_rad : ℝ) :
  (1/2 * Real.sin (A / 2 * deg_to_rad) + Real.cos (A / 2 * deg_to_rad) = 1) →
  (A = 360) :=
sorry

end find_angle_l21_21127


namespace sum_of_solutions_l21_21073

-- Define the initial condition
def initial_equation (x : ℝ) : Prop := (x - 8) ^ 2 = 49

-- Define the conclusion we want to reach
def sum_of_solutions_is_16 : Prop :=
  ∃ x1 x2 : ℝ, initial_equation x1 ∧ initial_equation x2 ∧ x1 ≠ x2 ∧ x1 + x2 = 16

theorem sum_of_solutions :
  sum_of_solutions_is_16 :=
by
  sorry

end sum_of_solutions_l21_21073


namespace necessary_and_sufficient_condition_l21_21992

theorem necessary_and_sufficient_condition (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : 
  (m + n > m * n) ↔ (m = 1 ∨ n = 1) := by
  sorry

end necessary_and_sufficient_condition_l21_21992


namespace chocolate_bar_cost_l21_21195

theorem chocolate_bar_cost (x : ℝ) (total_bars : ℕ) (bars_sold : ℕ) (total_amount_made : ℝ)
    (h1 : total_bars = 7)
    (h2 : bars_sold = total_bars - 4)
    (h3 : total_amount_made = 9)
    (h4 : total_amount_made = bars_sold * x) : x = 3 :=
sorry

end chocolate_bar_cost_l21_21195


namespace total_cost_of_vacation_l21_21743

variable (C : ℚ)

def cost_per_person_divided_among_3 := C / 3
def cost_per_person_divided_among_4 := C / 4
def per_person_difference := 40

theorem total_cost_of_vacation
  (h : cost_per_person_divided_among_3 C - cost_per_person_divided_among_4 C = per_person_difference) :
  C = 480 := by
  sorry

end total_cost_of_vacation_l21_21743


namespace distance_between_circle_centers_l21_21161

theorem distance_between_circle_centers
  (R r d : ℝ)
  (h1 : R = 7)
  (h2 : r = 4)
  (h3 : d = 5 + 1)
  (h_total_diameter : 5 + 8 + 1 = 14)
  (h_radius_R : R = 14 / 2)
  (h_radius_r : r = 8 / 2) : d = 6 := 
by sorry

end distance_between_circle_centers_l21_21161


namespace total_gallons_of_seed_l21_21956

-- Condition (1): The area of the football field is 8000 square meters.
def area_football_field : ℝ := 8000

-- Condition (2): Each square meter needs 4 times as much seed as fertilizer.
def seed_to_fertilizer_ratio : ℝ := 4

-- Condition (3): Carson uses 240 gallons of seed and fertilizer combined for every 2000 square meters.
def combined_usage_per_2000sqm : ℝ := 240
def area_unit : ℝ := 2000

-- Target: Prove that the total gallons of seed Carson uses for the entire field is 768 gallons.
theorem total_gallons_of_seed : seed_to_fertilizer_ratio * area_football_field / area_unit / (seed_to_fertilizer_ratio + 1) * combined_usage_per_2000sqm * (area_football_field / area_unit) = 768 :=
sorry

end total_gallons_of_seed_l21_21956


namespace number_of_masters_students_l21_21034

theorem number_of_masters_students (total_sample : ℕ) (ratio_assoc : ℕ) (ratio_undergrad : ℕ) (ratio_masters : ℕ) (ratio_doctoral : ℕ) 
(h1 : ratio_assoc = 5) (h2 : ratio_undergrad = 15) (h3 : ratio_masters = 9) (h4 : ratio_doctoral = 1) (h_total_sample : total_sample = 120) :
  (ratio_masters * total_sample) / (ratio_assoc + ratio_undergrad + ratio_masters + ratio_doctoral) = 36 :=
by
  sorry

end number_of_masters_students_l21_21034


namespace smallest_even_n_l21_21042

theorem smallest_even_n (n : ℕ) :
  (∃ n, 0 < n ∧ n % 2 = 0 ∧ (∀ k, 1 ≤ k → k ≤ n / 2 → k = 2213 ∨ k = 3323 ∨ k = 6121) ∧ (2^k * (k!)) % (2213 * 3323 * 6121) = 0) → n = 12242 :=
sorry

end smallest_even_n_l21_21042


namespace initial_dogs_count_is_36_l21_21669

-- Conditions
def initial_cats := 29
def adopted_dogs := 20
def additional_cats := 12
def total_pets := 57

-- Calculate total cats
def total_cats := initial_cats + additional_cats

-- Calculate initial dogs
def initial_dogs (initial_dogs : ℕ) : Prop :=
(initial_dogs - adopted_dogs) + total_cats = total_pets

-- Prove that initial dogs (D) is 36
theorem initial_dogs_count_is_36 : initial_dogs 36 :=
by
-- Here should contain the proof which is omitted
sorry

end initial_dogs_count_is_36_l21_21669


namespace find_x_l21_21466

theorem find_x (x : ℝ) : (x = 2 ∨ x = -2) ↔ (|x|^2 - 5 * |x| + 6 = 0 ∧ x^2 - 4 = 0) :=
by
  sorry

end find_x_l21_21466


namespace number_of_sets_A_l21_21615

/-- Given conditions about intersections and unions of set A, we want to find the number of 
  possible sets A that satisfy the given conditions. Specifically, prove the following:
  - A ∩ {-1, 0, 1} = {0, 1}
  - A ∪ {-2, 0, 2} = {-2, 0, 1, 2}
  Total number of such sets A is 4.
-/
theorem number_of_sets_A : ∃ (As : Finset (Finset ℤ)), 
  (∀ A ∈ As, A ∩ {-1, 0, 1} = {0, 1} ∧ A ∪ {-2, 0, 2} = {-2, 0, 1, 2}) ∧
  As.card = 4 := 
sorry

end number_of_sets_A_l21_21615


namespace solve_a_l21_21252

variable (a : ℝ)

theorem solve_a (h : ∃ b : ℝ, (9 * x^2 + 12 * x + a) = (3 * x + b) ^ 2) : a = 4 :=
by
   sorry

end solve_a_l21_21252


namespace num_aluminum_cans_l21_21018

def num_glass_bottles : ℕ := 10
def total_litter : ℕ := 18

theorem num_aluminum_cans : total_litter - num_glass_bottles = 8 :=
by
  sorry

end num_aluminum_cans_l21_21018


namespace composite_numbers_quotient_l21_21706

theorem composite_numbers_quotient :
  (14 * 15 * 16 * 18 * 20 * 21 * 22 * 24 * 25 * 26 : ℚ) /
  (27 * 28 * 30 * 32 * 33 * 34 * 35 * 36 * 38 * 39) =
  (14 * 15 * 16 * 18 * 20 * 21 * 22 * 24 * 25 * 26 : ℚ) / 
  (27 * 28 * 30 * 32 * 33 * 34 * 35 * 36 * 38 * 39) :=
by sorry

end composite_numbers_quotient_l21_21706


namespace floor_add_double_eq_15_4_l21_21342

theorem floor_add_double_eq_15_4 (r : ℝ) (h : (⌊r⌋ : ℝ) + 2 * r = 15.4) : r = 5.2 := 
sorry

end floor_add_double_eq_15_4_l21_21342


namespace correct_answer_l21_21189

def mary_initial_cards : ℝ := 18.0
def mary_bought_cards : ℝ := 40.0
def mary_left_cards : ℝ := 32.0
def mary_promised_cards (initial_cards : ℝ) (bought_cards : ℝ) (left_cards : ℝ) : ℝ :=
  initial_cards + bought_cards - left_cards

theorem correct_answer :
  mary_promised_cards mary_initial_cards mary_bought_cards mary_left_cards = 26.0 := by
  sorry

end correct_answer_l21_21189


namespace smallest_b_for_factorable_polynomial_l21_21570

theorem smallest_b_for_factorable_polynomial :
  ∃ (b : ℕ), b > 0 ∧ (∃ (p q : ℤ), x^2 + b * x + 1176 = (x + p) * (x + q) ∧ p * q = 1176 ∧ p + q = b) ∧ 
  (∀ (b' : ℕ), b' > 0 → (∃ (p' q' : ℤ), x^2 + b' * x + 1176 = (x + p') * (x + q') ∧ p' * q' = 1176 ∧ p' + q' = b') → b ≤ b') :=
sorry

end smallest_b_for_factorable_polynomial_l21_21570


namespace part1_proof_part2_proof_l21_21382

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := |x - 1| - |x - m|

theorem part1_proof : ∀ x, f x 2 ≥ 1 ↔ x ≥ 2 :=
by 
  sorry

theorem part2_proof : (∀ x : ℝ, f x m ≤ 5) → (-4 ≤ m ∧ m ≤ 6) :=
by
  sorry

end part1_proof_part2_proof_l21_21382
