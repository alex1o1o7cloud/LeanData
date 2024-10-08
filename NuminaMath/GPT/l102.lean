import Mathlib

namespace Joan_video_game_expense_l102_102477

theorem Joan_video_game_expense : 
  let basketball_price := 5.20
  let racing_price := 4.23
  let action_price := 7.12
  let discount_rate := 0.10
  let sales_tax_rate := 0.06
  let discounted_basketball_price := basketball_price * (1 - discount_rate)
  let discounted_racing_price := racing_price * (1 - discount_rate)
  let discounted_action_price := action_price * (1 - discount_rate)
  let total_cost_before_tax := discounted_basketball_price + discounted_racing_price + discounted_action_price
  let sales_tax := total_cost_before_tax * sales_tax_rate
  let total_cost := total_cost_before_tax + sales_tax
  total_cost = 15.79 :=
by
  sorry

end Joan_video_game_expense_l102_102477


namespace solve_triangle_l102_102537

noncomputable def triangle_side_lengths (a b c : ℝ) : Prop :=
  a = 10 ∧ b = 9 ∧ c = 17

theorem solve_triangle (a b c : ℝ) :
  (a ^ 2 - b ^ 2 = 19) ∧ 
  (126 + 52 / 60 + 12 / 3600 = 126.87) ∧ -- Converting the angle into degrees for simplicity
  (21.25 = 21.25)  -- Diameter given directly
  → triangle_side_lengths a b c :=
sorry

end solve_triangle_l102_102537


namespace ratio_of_cream_l102_102699

theorem ratio_of_cream (coffee_init : ℕ) (joe_coffee_drunk : ℕ) (cream_added : ℕ) (joann_total_drunk : ℕ) 
  (joann_coffee_init : ℕ := coffee_init)
  (joe_coffee_init : ℕ := coffee_init) (joann_cream_init : ℕ := cream_added)
  (joe_cream_init : ℕ := cream_added)
  (joann_drunk_cream_ratio : ℚ := joann_cream_init / (joann_coffee_init + joann_cream_init)) :
  (joe_cream_init / (joann_cream_init - joann_total_drunk * (joann_drunk_cream_ratio))) = 
  (6 / 5) := 
by
  sorry

end ratio_of_cream_l102_102699


namespace original_decimal_l102_102704

theorem original_decimal (x : ℝ) : (10 * x = x + 2.7) → x = 0.3 := 
by
    intro h
    sorry

end original_decimal_l102_102704


namespace fewer_twos_result_100_l102_102347

theorem fewer_twos_result_100 :
  (222 / 2) - (22 / 2) = 100 := by
  sorry

end fewer_twos_result_100_l102_102347


namespace positive_number_square_roots_l102_102975

theorem positive_number_square_roots (a : ℝ) (x : ℝ) (h1 : x = (a - 7)^2)
  (h2 : x = (2 * a + 1)^2) : x = 25 := by
sorry

end positive_number_square_roots_l102_102975


namespace find_y_value_l102_102163

theorem find_y_value : (12 ^ 2 * 6 ^ 4) / 432 = 432 := by
  sorry

end find_y_value_l102_102163


namespace increase_circumference_l102_102269

theorem increase_circumference (d1 d2 : ℝ) (increase : ℝ) (P : ℝ) : 
  increase = 2 * Real.pi → 
  P = Real.pi * increase → 
  P = 2 * Real.pi ^ 2 := 
by 
  intros h_increase h_P
  rw [h_P, h_increase]
  sorry

end increase_circumference_l102_102269


namespace proof_theorem_l102_102336

noncomputable def proof_problem (a b c : ℝ) := 
  (2 * b = a + c) ∧ 
  (2 / b = 1 / a + 1 / c ∨ 2 / a = 1 / b + 1 / c ∨ 2 / c = 1 / a + 1 / b) ∧ 
  (a ≠ 0) ∧ (b ≠ 0) ∧ (c ≠ 0)

theorem proof_theorem (a b c : ℝ) (h : proof_problem a b c) :
  (a = b ∧ b = c) ∨ 
  (∃ (x : ℝ), x ≠ 0 ∧ a = -4 * x ∧ b = -x ∧ c = 2 * x) :=
by
  sorry

end proof_theorem_l102_102336


namespace highway_length_on_map_l102_102003

theorem highway_length_on_map (total_length_km : ℕ) (scale : ℚ) (length_on_map_cm : ℚ) 
  (h1 : total_length_km = 155) (h2 : scale = 1 / 500000) :
  length_on_map_cm = 31 :=
by
  sorry

end highway_length_on_map_l102_102003


namespace sphere_surface_area_ratios_l102_102873

theorem sphere_surface_area_ratios
  (s : ℝ)
  (r1 : ℝ)
  (r2 : ℝ)
  (r3 : ℝ)
  (h1 : r1 = s / 4 * Real.sqrt 6)
  (h2 : r2 = s / 4 * Real.sqrt 2)
  (h3 : r3 = s / 12 * Real.sqrt 6) :
  (4 * Real.pi * r1^2) / (4 * Real.pi * r3^2) = 9 ∧
  (4 * Real.pi * r2^2) / (4 * Real.pi * r3^2) = 3 ∧
  (4 * Real.pi * r3^2) / (4 * Real.pi * r3^2) = 1 := 
by
  sorry

end sphere_surface_area_ratios_l102_102873


namespace divisible_by_units_digit_l102_102666

theorem divisible_by_units_digit :
  ∃ l : List ℕ, l = [21, 22, 24, 25] ∧ l.length = 4 := 
  sorry

end divisible_by_units_digit_l102_102666


namespace fractional_eq_has_positive_root_m_value_l102_102998

-- Define the conditions and the proof goal
theorem fractional_eq_has_positive_root_m_value (m x : ℝ) (h1 : x - 2 ≠ 0) (h2 : 2 - x ≠ 0) (h3 : ∃ x > 0, (m / (x - 2)) = ((1 - x) / (2 - x)) - 3) : m = 1 :=
by
  -- Proof goes here
  sorry

end fractional_eq_has_positive_root_m_value_l102_102998


namespace correct_multiplication_factor_l102_102753

theorem correct_multiplication_factor (x : ℕ) : ((139 * x) - 1251 = 139 * 34) → x = 43 := by
  sorry

end correct_multiplication_factor_l102_102753


namespace quadratic_function_properties_l102_102867

def f (x : ℝ) : ℝ := 3 * x^2 - 9 * x + 2

theorem quadratic_function_properties :
  (∃ x : ℝ, x > 0 ∧ ∀ y : ℝ, f x ≤ f y) ∧
  (∃ x : ℝ, x = 1.5 ∧ ∀ y : ℝ, f x ≤ f y) :=
by
  sorry

end quadratic_function_properties_l102_102867


namespace apps_more_than_files_l102_102345

theorem apps_more_than_files
  (initial_apps : ℕ)
  (initial_files : ℕ)
  (deleted_apps : ℕ)
  (deleted_files : ℕ)
  (remaining_apps : ℕ)
  (remaining_files : ℕ)
  (h1 : initial_apps - deleted_apps = remaining_apps)
  (h2 : initial_files - deleted_files = remaining_files)
  (h3 : initial_apps = 24)
  (h4 : initial_files = 9)
  (h5 : remaining_apps = 12)
  (h6 : remaining_files = 5) :
  remaining_apps - remaining_files = 7 :=
by {
  sorry
}

end apps_more_than_files_l102_102345


namespace valid_k_range_l102_102493

noncomputable def fx (k : ℝ) (x : ℝ) : ℝ :=
  k * x^2 + k * x + k + 3

theorem valid_k_range:
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 3 → fx k x ≥ 0) ↔ (k ≥ -3 / 13) :=
by
  sorry

end valid_k_range_l102_102493


namespace no_valid_n_l102_102052

theorem no_valid_n (n : ℕ) (h₁ : 100 ≤ n / 4) (h₂ : n / 4 ≤ 999) (h₃ : 100 ≤ 4 * n) (h₄ : 4 * n ≤ 999) : false := by
  sorry

end no_valid_n_l102_102052


namespace more_green_peaches_than_red_l102_102931

theorem more_green_peaches_than_red : 
  let red_peaches := 7
  let green_peaches := 8
  green_peaches - red_peaches = 1 := 
by
  let red_peaches := 7
  let green_peaches := 8
  show green_peaches - red_peaches = 1 
  sorry

end more_green_peaches_than_red_l102_102931


namespace area_of_smaller_circle_l102_102328

noncomputable def radius_of_smaller_circle (r : ℝ) : ℝ := r

noncomputable def radius_of_larger_circle (r : ℝ) : ℝ := 3 * r

noncomputable def length_PA := 5
noncomputable def length_AB := 5

theorem area_of_smaller_circle (r : ℝ) (h1 : radius_of_smaller_circle r = r)
  (h2 : radius_of_larger_circle r = 3 * r)
  (h3 : length_PA = 5) (h4 : length_AB = 5) :
  π * r^2 = (25 / 3) * π :=
  sorry

end area_of_smaller_circle_l102_102328


namespace part1_part2_l102_102089

-- (1) Prove that if 2 ∈ M and M is the solution set of ax^2 + 5x - 2 > 0, then a > -2.
theorem part1 (a : ℝ) (h : 2 * (a * 4 + 10) - 2 > 0) : a > -2 :=
sorry

-- (2) Given M = {x | 1/2 < x < 2} and M is the solution set of ax^2 + 5x - 2 > 0,
-- prove that the solution set of ax^2 - 5x + a^2 - 1 > 0 is -3 < x < 1/2
theorem part2 (a : ℝ) (h1 : ∀ x : ℝ, (1/2 < x ∧ x < 2) ↔ ax^2 + 5*x - 2 > 0) (h2 : a = -2) :
  ∀ x : ℝ, (-3 < x ∧ x < 1/2) ↔ (-2 * x^2 - 5 * x + 3 > 0) :=
sorry

end part1_part2_l102_102089


namespace subset_N_M_l102_102487

def M : Set ℝ := { x | ∃ (k : ℤ), x = k / 2 + 1 / 3 }
def N : Set ℝ := { x | ∃ (k : ℤ), x = k + 1 / 3 }

theorem subset_N_M : N ⊆ M := 
  sorry

end subset_N_M_l102_102487


namespace negation_of_statement_l102_102204

theorem negation_of_statement :
  ¬ (∃ x_0 : ℝ, x_0^2 + 2 * x_0 + 2 ≤ 0) ↔ ∀ x : ℝ, x^2 + 2 * x + 2 > 0 := by
  sorry

end negation_of_statement_l102_102204


namespace chuck_vs_dave_ride_time_l102_102139

theorem chuck_vs_dave_ride_time (D E : ℕ) (h1 : D = 10) (h2 : E = 65) (h3 : E = 13 * C / 10) :
  (C / D = 5) :=
by
  sorry

end chuck_vs_dave_ride_time_l102_102139


namespace line_equation_l102_102329

theorem line_equation (x y : ℝ) (h : (2, 3) ∈ {p : ℝ × ℝ | (∃ a, p.1 + p.2 = a) ∨ (∃ k, p.2 = k * p.1)}) :
  (3 * x - 2 * y = 0) ∨ (x + y - 5 = 0) :=
sorry

end line_equation_l102_102329


namespace intersection_sums_l102_102747

theorem intersection_sums :
  (∀ (x y : ℝ), (y = x^3 - 3 * x - 4) → (x + 3 * y = 3) → (∃ (x1 y1 x2 y2 x3 y3 : ℝ),
  (y1 = x1^3 - 3 * x1 - 4) ∧ (x1 + 3 * y1 = 3) ∧
  (y2 = x2^3 - 3 * x2 - 4) ∧ (x2 + 3 * y2 = 3) ∧
  (y3 = x3^3 - 3 * x3 - 4) ∧ (x3 + 3 * y3 = 3) ∧
  x1 + x2 + x3 = 8 / 3 ∧ y1 + y2 + y3 = 19 / 9)) :=
sorry

end intersection_sums_l102_102747


namespace jason_seashells_after_giving_l102_102647

-- Define the number of seashells Jason originally found
def original_seashells : ℕ := 49

-- Define the number of seashells Jason gave to Tim
def seashells_given : ℕ := 13

-- Prove that the number of seashells Jason now has is 36
theorem jason_seashells_after_giving : original_seashells - seashells_given = 36 :=
by
  -- This is where the proof would go
  sorry

end jason_seashells_after_giving_l102_102647


namespace solution_set_l102_102094

open Set Real

noncomputable def f : ℝ → ℝ := sorry

axiom f_odd : ∀ x : ℝ, f (-x) = - f x
axiom f_at_two : f 2 = 0
axiom f_cond : ∀ x : ℝ, 0 < x → x * (deriv (deriv f) x) + f x < 0

theorem solution_set :
  {x : ℝ | x * f x > 0} = Ioo (-2 : ℝ) 0 ∪ Ioo 0 2 :=
by
  sorry

end solution_set_l102_102094


namespace find_eq_thirteen_l102_102031

open Real

theorem find_eq_thirteen
  (a x b y c z : ℝ)
  (h1 : x / a + y / b + z / c = 5)
  (h2 : a / x + b / y + c / z = 6) :
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 13 := 
sorry

end find_eq_thirteen_l102_102031


namespace max_sum_abs_coeff_l102_102875

theorem max_sum_abs_coeff (a b c : ℝ) (f : ℝ → ℝ)
  (h1 : ∀ x, f x = a * x^2 + b * x + c)
  (h2 : |f 1| ≤ 1)
  (h3 : |f (1/2)| ≤ 1)
  (h4 : |f 0| ≤ 1) :
  |a| + |b| + |c| ≤ 17 :=
sorry

end max_sum_abs_coeff_l102_102875


namespace book_configurations_l102_102241

theorem book_configurations : 
  (∃ (configurations : Finset ℕ), configurations = {1, 2, 3, 4, 5, 6, 7} ∧ configurations.card = 7) 
  ↔ 
  (∃ (n : ℕ), n = 7) :=
by 
  sorry

end book_configurations_l102_102241


namespace remainder_of_875_div_by_170_l102_102534

theorem remainder_of_875_div_by_170 :
  ∃ r, (∀ x, x ∣ 680 ∧ x ∣ (875 - r) → x ≤ 170) ∧ 170 ∣ (875 - r) ∧ r = 25 :=
by
  sorry

end remainder_of_875_div_by_170_l102_102534


namespace mike_practices_hours_on_saturday_l102_102182

-- Definitions based on conditions
def weekday_hours : ℕ := 3
def weekdays_per_week : ℕ := 5
def total_hours : ℕ := 60
def weeks : ℕ := 3

def calculate_total_weekday_hours (weekday_hours weekdays_per_week weeks : ℕ) : ℕ :=
  weekday_hours * weekdays_per_week * weeks

def calculate_saturday_hours (total_hours total_weekday_hours weeks : ℕ) : ℕ :=
  (total_hours - total_weekday_hours) / weeks

-- Statement to prove
theorem mike_practices_hours_on_saturday :
  calculate_saturday_hours total_hours (calculate_total_weekday_hours weekday_hours weekdays_per_week weeks) weeks = 5 :=
by 
  sorry

end mike_practices_hours_on_saturday_l102_102182


namespace minimum_distance_l102_102658

theorem minimum_distance (m n : ℝ) (a : ℝ) (h1 : 2 ≤ a) (h2 : a ≤ 4) 
  (h3 : m * Real.sqrt (Real.log a - 1 / 4) + 2 * a + 1 / 2 * n = 0) : 
  Real.sqrt (m^2 + n^2) = 4 * Real.sqrt (Real.log 2) / Real.log 2 :=
sorry

end minimum_distance_l102_102658


namespace Q_at_1_eq_neg_1_l102_102491

def P (x : ℝ) : ℝ := 3 * x^3 - 5 * x^2 + 2 * x - 1

noncomputable def mean_coefficient : ℝ := (3 - 5 + 2 - 1) / 4

noncomputable def Q (x : ℝ) : ℝ := mean_coefficient * x^3 + mean_coefficient * x^2 + mean_coefficient * x + mean_coefficient

theorem Q_at_1_eq_neg_1 : Q 1 = -1 := by
  sorry

end Q_at_1_eq_neg_1_l102_102491


namespace neg_p_l102_102757

theorem neg_p : ∀ (m : ℝ), ∀ (x : ℝ), (x^2 + m*x + 1 ≠ 0) :=
by
  sorry

end neg_p_l102_102757


namespace chimney_height_theorem_l102_102518

noncomputable def chimney_height :=
  let BCD := 75 * Real.pi / 180
  let BDC := 60 * Real.pi / 180
  let CBD := 45 * Real.pi / 180
  let CD := 40
  let BC := CD * Real.sin BDC / Real.sin CBD
  let CE := 1
  let elevation := 30 * Real.pi / 180
  let AB := CE + (Real.tan elevation * BC)
  AB

theorem chimney_height_theorem : chimney_height = 1 + 20 * Real.sqrt 2 :=
by
  sorry

end chimney_height_theorem_l102_102518


namespace crushing_load_value_l102_102327

-- Define the given formula and values
def T : ℕ := 3
def H : ℕ := 9
def K : ℕ := 2

-- Given formula for L
def L (T H K : ℕ) : ℚ := 50 * T^5 / (K * H^3)

-- Prove that L = 8 + 1/3
theorem crushing_load_value :
  L T H K = 8 + 1 / 3 :=
by
  sorry

end crushing_load_value_l102_102327


namespace cashback_discount_percentage_l102_102795

noncomputable def iphoneOriginalPrice : ℝ := 800
noncomputable def iwatchOriginalPrice : ℝ := 300
noncomputable def iphoneDiscountRate : ℝ := 0.15
noncomputable def iwatchDiscountRate : ℝ := 0.10
noncomputable def finalPrice : ℝ := 931

noncomputable def iphoneDiscountedPrice : ℝ := iphoneOriginalPrice * (1 - iphoneDiscountRate)
noncomputable def iwatchDiscountedPrice : ℝ := iwatchOriginalPrice * (1 - iwatchDiscountRate)
noncomputable def totalDiscountedPrice : ℝ := iphoneDiscountedPrice + iwatchDiscountedPrice
noncomputable def cashbackAmount : ℝ := totalDiscountedPrice - finalPrice
noncomputable def cashbackRate : ℝ := (cashbackAmount / totalDiscountedPrice) * 100

theorem cashback_discount_percentage : cashbackRate = 2 := by
  sorry

end cashback_discount_percentage_l102_102795


namespace markers_blue_l102_102499

theorem markers_blue {total_markers red_markers blue_markers : ℝ} 
  (h_total : total_markers = 64.0) 
  (h_red : red_markers = 41.0) 
  (h_blue : blue_markers = total_markers - red_markers) : 
  blue_markers = 23.0 := 
by 
  sorry

end markers_blue_l102_102499


namespace rectangle_area_perimeter_l102_102654

/-- 
Given a rectangle with positive integer sides a and b,
let A be the area and P be the perimeter.

A = a * b
P = 2 * a + 2 * b

Prove that 100 cannot be expressed as A + P - 4.
-/
theorem rectangle_area_perimeter (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (A : ℕ) (P : ℕ)
  (hA : A = a * b) (hP : P = 2 * a + 2 * b) : 
  ¬ (A + P - 4 = 100) := 
sorry

end rectangle_area_perimeter_l102_102654


namespace Moscow_1975_p_q_r_equal_primes_l102_102531

theorem Moscow_1975_p_q_r_equal_primes (a b c : ℕ) (p q r : ℕ) 
  (hp : p = b^c + a) 
  (hq : q = a^b + c) 
  (hr : r = c^a + b) 
  (prime_p : Prime p) 
  (prime_q : Prime q) 
  (prime_r : Prime r) : 
  q = r :=
sorry

end Moscow_1975_p_q_r_equal_primes_l102_102531


namespace sum_of_seven_consecutive_integers_l102_102936

theorem sum_of_seven_consecutive_integers (n : ℤ) :
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) = 7 * n + 21 :=
  sorry

end sum_of_seven_consecutive_integers_l102_102936


namespace years_later_l102_102063

variables (R F Y : ℕ)

-- Conditions
def condition1 := F = 4 * R
def condition2 := F + Y = 5 * (R + Y) / 2
def condition3 := F + Y + 8 = 2 * (R + Y + 8)

-- The result to be proved
theorem years_later (R F Y : ℕ) (h1 : condition1 R F) (h2 : condition2 R F Y) (h3 : condition3 R F Y) : 
  Y = 8 := by
  sorry

end years_later_l102_102063


namespace kitchen_chairs_count_l102_102833

-- Define the conditions
def total_chairs : ℕ := 9
def living_room_chairs : ℕ := 3

-- Prove the number of kitchen chairs
theorem kitchen_chairs_count : total_chairs - living_room_chairs = 6 := by
  -- Proof goes here
  sorry

end kitchen_chairs_count_l102_102833


namespace min_value_x_plus_4_div_x_plus_1_l102_102423

theorem min_value_x_plus_4_div_x_plus_1 (x : ℝ) (h : x > -1) : ∃ m, m = 3 ∧ ∀ y, y = x + 4 / (x + 1) → y ≥ m :=
sorry

end min_value_x_plus_4_div_x_plus_1_l102_102423


namespace garden_least_cost_l102_102035

-- Define the costs per flower type
def cost_sunflower : ℝ := 0.75
def cost_tulip : ℝ := 2
def cost_marigold : ℝ := 1.25
def cost_orchid : ℝ := 4
def cost_violet : ℝ := 3.5

-- Define the areas of each section
def area_top_left : ℝ := 5 * 2
def area_bottom_left : ℝ := 5 * 5
def area_top_right : ℝ := 3 * 5
def area_bottom_right : ℝ := 3 * 4
def area_central_right : ℝ := 5 * 3

-- Calculate the total costs after assigning the most cost-effective layout
def total_cost : ℝ :=
  (area_top_left * cost_orchid) +
  (area_bottom_right * cost_violet) +
  (area_central_right * cost_tulip) +
  (area_bottom_left * cost_marigold) +
  (area_top_right * cost_sunflower)

-- Prove that the total cost is $154.50
theorem garden_least_cost : total_cost = 154.50 :=
by sorry

end garden_least_cost_l102_102035


namespace max_bicycle_distance_l102_102330

-- Define the properties of the tires
def front_tire_duration : ℕ := 5000
def rear_tire_duration : ℕ := 3000

-- Define the maximum distance the bicycle can travel
def max_distance : ℕ := 3750

-- The main statement to be proven (proof is not required)
theorem max_bicycle_distance 
  (swap_usage : ∀ (d1 d2 : ℕ), d1 + d2 <= front_tire_duration + rear_tire_duration) : 
  ∃ (x : ℕ), x = max_distance := 
sorry

end max_bicycle_distance_l102_102330


namespace find_k_l102_102586

theorem find_k
  (t k : ℝ)
  (h1 : t = 5 / 9 * (k - 32))
  (h2 : t = 20) :
  k = 68 := 
by
  sorry

end find_k_l102_102586


namespace little_john_gave_to_each_friend_l102_102516

noncomputable def little_john_total : ℝ := 10.50
noncomputable def sweets : ℝ := 2.25
noncomputable def remaining : ℝ := 3.85

theorem little_john_gave_to_each_friend :
  (little_john_total - sweets - remaining) / 2 = 2.20 :=
by
  sorry

end little_john_gave_to_each_friend_l102_102516


namespace sequence_is_geometric_not_arithmetic_l102_102388

noncomputable def a_n (n : ℕ) : ℕ :=
  if n = 1 then 1
  else 2^(n-1)

def S_n (n : ℕ) : ℕ :=
  2^n - 1

theorem sequence_is_geometric_not_arithmetic (n : ℕ) : 
  (∀ n ≥ 2, a_n n = S_n n - S_n (n - 1)) ∧
  (a_n 1 = 1) ∧
  (∃ r : ℕ, r > 1 ∧ ∀ n ≥ 1, a_n (n + 1) = r * a_n n) ∧
  ¬(∃ d : ℤ, ∀ n, (a_n (n + 1) : ℤ) = a_n n + d) :=
by
  sorry

end sequence_is_geometric_not_arithmetic_l102_102388


namespace sphere_radius_same_volume_l102_102511

noncomputable def tent_radius : ℝ := 3
noncomputable def tent_height : ℝ := 9

theorem sphere_radius_same_volume : 
  (4 / 3) * Real.pi * ( (20.25)^(1/3) )^3 = (1 / 3) * Real.pi * tent_radius^2 * tent_height :=
by
  sorry

end sphere_radius_same_volume_l102_102511


namespace sum_of_diffs_is_10_l102_102324

-- Define the number of fruits each person has
def Sharon_plums : ℕ := 7
def Allan_plums : ℕ := 10
def Dave_oranges : ℕ := 12

-- Define the differences in the number of fruits
def diff_Sharon_Allan : ℕ := Allan_plums - Sharon_plums
def diff_Sharon_Dave : ℕ := Dave_oranges - Sharon_plums
def diff_Allan_Dave : ℕ := Dave_oranges - Allan_plums

-- Define the sum of these differences
def sum_of_diffs : ℕ := diff_Sharon_Allan + diff_Sharon_Dave + diff_Allan_Dave

-- State the theorem to be proved
theorem sum_of_diffs_is_10 : sum_of_diffs = 10 := by
  sorry

end sum_of_diffs_is_10_l102_102324


namespace difference_in_students_specific_case_diff_l102_102027

-- Define the variables and conditions
variables (a b : ℕ)

-- Condition: a > b
axiom h1 : a > b

-- Definition of eighth grade students
def eighth_grade_students := (3 * a + b) * (2 * a + 2 * b)

-- Definition of seventh grade students
def seventh_grade_students := (2 * (a + b)) ^ 2

-- Theorem for the difference in the number of students
theorem difference_in_students : (eighth_grade_students a b) - (seventh_grade_students a b) = 2 * a^2 - 2 * b^2 :=
sorry

-- Theorem for the specific example when a = 10 and b = 2
theorem specific_case_diff : eighth_grade_students 10 2 - seventh_grade_students 10 2 = 192 :=
sorry

end difference_in_students_specific_case_diff_l102_102027


namespace area_of_ADFE_l102_102695

namespace Geometry

open Classical

noncomputable def area_triangle (A B C : Type) [Field A] (area_DBF area_BFC area_FCE : A) : A :=
  let total_area := area_DBF + area_BFC + area_FCE
  let area := (105 : A) / 4
  total_area + area

theorem area_of_ADFE (A B C D E F : Type) [Field A] 
  (area_DBF : A) (area_BFC : A) (area_FCE : A) : 
  area_DBF = 4 → area_BFC = 6 → area_FCE = 5 → 
  area_triangle A B C area_DBF area_BFC area_FCE = (15 : A) + (105 : A) / 4 := 
by 
  intros 
  sorry

end area_of_ADFE_l102_102695


namespace no_real_roots_of_f_l102_102107

def f (x : ℝ) : ℝ := (x + 1) * |x + 1| - x * |x| + 1

theorem no_real_roots_of_f :
  ∀ x : ℝ, f x ≠ 0 := by
  sorry

end no_real_roots_of_f_l102_102107


namespace find_ordered_pair_l102_102239

theorem find_ordered_pair (s h : ℝ) :
  (∀ (u : ℝ), ∃ (x y : ℝ), x = s + 3 * u ∧ y = -3 + h * u ∧ y = 4 * x + 2) →
  (s, h) = (-5 / 4, 12) :=
by
  sorry

end find_ordered_pair_l102_102239


namespace price_Ramesh_paid_l102_102945

-- Define the conditions
def labelled_price_sold (P : ℝ) := 1.10 * P
def discount_price_paid (P : ℝ) := 0.80 * P
def additional_costs := 125 + 250
def total_cost (P : ℝ) := discount_price_paid P + additional_costs

-- The main theorem stating that given the conditions,
-- the price Ramesh paid for the refrigerator is Rs. 13175.
theorem price_Ramesh_paid (P : ℝ) (H : labelled_price_sold P = 17600) :
  total_cost P = 13175 :=
by
  -- Providing a placeholder, as we do not need to provide the proof steps in the problem formulation
  sorry

end price_Ramesh_paid_l102_102945


namespace total_bowling_balls_l102_102056

theorem total_bowling_balls (red_balls : ℕ) (green_balls : ℕ) (h1 : red_balls = 30) (h2 : green_balls = red_balls + 6) : red_balls + green_balls = 66 :=
by
  sorry

end total_bowling_balls_l102_102056


namespace sqrt_meaningful_range_l102_102706

theorem sqrt_meaningful_range (x : ℝ) : x + 1 ≥ 0 ↔ x ≥ -1 :=
by sorry

end sqrt_meaningful_range_l102_102706


namespace find_ages_l102_102135

variables (H J A : ℕ)

def conditions := 
  H + J + A = 90 ∧ 
  H = 2 * J - 5 ∧ 
  H + J - 10 = A

theorem find_ages (h_cond : conditions H J A) : 
  H = 32 ∧ 
  J = 18 ∧ 
  A = 40 :=
sorry

end find_ages_l102_102135


namespace interval_proof_l102_102211

theorem interval_proof (x : ℝ) (h1 : 2 < 3 * x) (h2 : 3 * x < 3) (h3 : 2 < 4 * x) (h4 : 4 * x < 3) :
    (2 / 3) < x ∧ x < (3 / 4) :=
sorry

end interval_proof_l102_102211


namespace polynomial_characterization_l102_102113

theorem polynomial_characterization (P : ℝ → ℝ) :
  (∀ a b c : ℝ, ab + bc + ca = 0 → P (a - b) + P (b - c) + P (c - a) = 2 * P (a + b + c)) →
  ∃ (α β : ℝ), ∀ x : ℝ, P x = α * x^4 + β * x^2 :=
by
  sorry

end polynomial_characterization_l102_102113


namespace alex_needs_additional_coins_l102_102989

theorem alex_needs_additional_coins :
  let n := 15
  let current_coins := 63
  let target_sum := (n * (n + 1)) / 2
  let additional_coins := target_sum - current_coins
  additional_coins = 57 :=
by
  sorry

end alex_needs_additional_coins_l102_102989


namespace graph_symmetry_about_line_l102_102341

noncomputable def f (x : ℝ) : ℝ := 3 * Real.cos (2 * x - (Real.pi / 3))

theorem graph_symmetry_about_line (x : ℝ) : 
  ∀ x, f (2 * (Real.pi / 3) - x) = f x :=
by
  sorry

end graph_symmetry_about_line_l102_102341


namespace books_sold_on_Tuesday_l102_102176

theorem books_sold_on_Tuesday 
  (initial_stock : ℕ)
  (books_sold_Monday : ℕ)
  (books_sold_Wednesday : ℕ)
  (books_sold_Thursday : ℕ)
  (books_sold_Friday : ℕ)
  (books_not_sold : ℕ) :
  initial_stock = 800 →
  books_sold_Monday = 60 →
  books_sold_Wednesday = 20 →
  books_sold_Thursday = 44 →
  books_sold_Friday = 66 →
  books_not_sold = 600 →
  ∃ (books_sold_Tuesday : ℕ), books_sold_Tuesday = 10
:= by
  intros h_initial h_monday h_wednesday h_thursday h_friday h_not_sold
  sorry

end books_sold_on_Tuesday_l102_102176


namespace max_students_with_equal_distribution_l102_102062

theorem max_students_with_equal_distribution (pens pencils : ℕ) (h_pens : pens = 3540) (h_pencils : pencils = 2860) :
  gcd pens pencils = 40 :=
by
  rw [h_pens, h_pencils]
  -- Proof steps will go here
  sorry

end max_students_with_equal_distribution_l102_102062


namespace transformed_graph_equation_l102_102655

theorem transformed_graph_equation (x y x' y' : ℝ)
  (h1 : x' = 5 * x)
  (h2 : y' = 3 * y)
  (h3 : x^2 + y^2 = 1) :
  x'^2 / 25 + y'^2 / 9 = 1 :=
by
  sorry

end transformed_graph_equation_l102_102655


namespace possible_shapes_l102_102099

def is_valid_shapes (T S C : ℕ) : Prop :=
  T + S + C = 24 ∧ T = 7 * S

theorem possible_shapes :
  ∃ (T S C : ℕ), is_valid_shapes T S C ∧ 
    (T = 0 ∧ S = 0 ∧ C = 24) ∨
    (T = 7 ∧ S = 1 ∧ C = 16) ∨
    (T = 14 ∧ S = 2 ∧ C = 8) ∨
    (T = 21 ∧ S = 3 ∧ C = 0) :=
by
  sorry

end possible_shapes_l102_102099


namespace larger_number_1655_l102_102793

theorem larger_number_1655 (L S : ℕ) (h1 : L - S = 1325) (h2 : L = 5 * S + 5) : L = 1655 :=
by sorry

end larger_number_1655_l102_102793


namespace range_of_a_l102_102991

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 - (a^2 + a) * x + a^3 > 0 ↔ (x < a^2 ∨ x > a)) → (0 ≤ a ∧ a ≤ 1) :=
by
  intros h
  sorry

end range_of_a_l102_102991


namespace first_divisor_l102_102220

theorem first_divisor (d x : ℕ) (h1 : ∃ k : ℕ, x = k * d + 11) (h2 : ∃ m : ℕ, x = 9 * m + 2) : d = 3 :=
sorry

end first_divisor_l102_102220


namespace graph_passes_through_fixed_point_l102_102058

theorem graph_passes_through_fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∃ y : ℝ, y = a^0 + 3 ∧ (0, y) = (0, 4)) :=
by
  use 4
  have h : a^0 = 1 := by simp
  rw [h]
  simp
  sorry

end graph_passes_through_fixed_point_l102_102058


namespace margin_in_terms_of_selling_price_l102_102544

variable (C S n M : ℝ)

theorem margin_in_terms_of_selling_price (h : M = (2 * C) / n) : M = (2 * S) / (n + 2) :=
sorry

end margin_in_terms_of_selling_price_l102_102544


namespace pool_capacity_l102_102605

theorem pool_capacity
  (pump_removes : ∀ (x : ℝ), x > 0 → (2 / 3) * x / 7.5 = (4 / 15) * x)
  (working_time : 0.15 * 60 = 9)
  (remaining_water : ∀ (x : ℝ), x > 0 → x - (0.8 * x) = 25) :
  ∃ x : ℝ, x = 125 :=
by
  sorry

end pool_capacity_l102_102605


namespace value_of_f_minus_3_l102_102933

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a * Real.sin x + b * Real.tan x + x^3 + 1

theorem value_of_f_minus_3 (a b : ℝ) (h : f 3 a b = 7) : f (-3) a b = -5 := 
by
  sorry

end value_of_f_minus_3_l102_102933


namespace smallest_pos_int_ending_in_9_divisible_by_13_l102_102584

theorem smallest_pos_int_ending_in_9_divisible_by_13 :
  ∃ (n : ℕ), n % 10 = 9 ∧ n % 13 = 0 ∧ ∀ m, m % 10 = 9 ∧ m % 13 = 0 → n ≤ m :=
by
  sorry

end smallest_pos_int_ending_in_9_divisible_by_13_l102_102584


namespace lcm_5_6_8_9_l102_102590

theorem lcm_5_6_8_9 : Nat.lcm (Nat.lcm 5 6) (Nat.lcm 8 9) = 360 := by
  sorry

end lcm_5_6_8_9_l102_102590


namespace hyperbola_asymptotes_equation_l102_102012

noncomputable def hyperbola_asymptotes (x y : ℝ) : Prop :=
  (x^2 / 4 - y^2 / 9 = 1) → (y = (3 / 2) * x) ∨ (y = -(3 / 2) * x)

-- Now we assert the theorem that states this
theorem hyperbola_asymptotes_equation :
  ∀ (x y : ℝ), hyperbola_asymptotes x y :=
by
  intros x y
  unfold hyperbola_asymptotes
  -- proof here
  sorry

end hyperbola_asymptotes_equation_l102_102012


namespace new_cylinder_height_percentage_l102_102858

variables (r h h_new : ℝ)

theorem new_cylinder_height_percentage :
  (7 / 8) * π * r^2 * h = (3 / 5) * π * (1.25 * r)^2 * h_new →
  (h_new / h) = 14 / 15 :=
by
  intro h_volume_eq
  sorry

end new_cylinder_height_percentage_l102_102858


namespace new_releases_fraction_is_2_over_5_l102_102685

def fraction_new_releases (total_books : ℕ) (frac_historical_fiction : ℚ) (frac_new_historical_fiction : ℚ) (frac_new_non_historical_fiction : ℚ) : ℚ :=
  let num_historical_fiction := frac_historical_fiction * total_books
  let num_new_historical_fiction := frac_new_historical_fiction * num_historical_fiction
  let num_non_historical_fiction := total_books - num_historical_fiction
  let num_new_non_historical_fiction := frac_new_non_historical_fiction * num_non_historical_fiction
  let total_new_releases := num_new_historical_fiction + num_new_non_historical_fiction
  num_new_historical_fiction / total_new_releases

theorem new_releases_fraction_is_2_over_5 :
  ∀ (total_books : ℕ), total_books > 0 →
    fraction_new_releases total_books (40 / 100) (40 / 100) (40 / 100) = 2 / 5 :=
by 
  intro total_books h
  sorry

end new_releases_fraction_is_2_over_5_l102_102685


namespace S8_value_l102_102761

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {q : ℝ}

-- The sequence {a_n} is a geometric sequence with common ratio q
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n, a (n + 1) = q * a n

theorem S8_value 
  (h_geo : is_geometric_sequence a q)
  (h_S4 : S 4 = 3)
  (h_S12_S8 : S 12 - S 8 = 12) :
  S 8 = 9 := 
sorry

end S8_value_l102_102761


namespace min_time_to_pass_l102_102384

noncomputable def tunnel_length : ℝ := 2150
noncomputable def num_vehicles : ℝ := 55
noncomputable def vehicle_length : ℝ := 10
noncomputable def speed_limit : ℝ := 20
noncomputable def max_speed : ℝ := 40

noncomputable def distance_between_vehicles (x : ℝ) : ℝ :=
if 0 < x ∧ x ≤ 10 then 20 else
if 10 < x ∧ x ≤ 20 then (1/6) * x ^ 2 + (1/3) * x else
0

noncomputable def time_to_pass_through_tunnel (x : ℝ) : ℝ :=
if 0 < x ∧ x ≤ 10 then (2150 + 10 * 55 + 20 * (55 - 1)) / x else
if 10 < x ∧ x ≤ 20 then (2150 + 10 * 55 + ((1/6) * x^2 + (1/3) * x) * (55 - 1)) / x + 9 * x + 18 else
0

theorem min_time_to_pass : ∃ x : ℝ, (10 < x ∧ x ≤ 20) ∧ x = 17.3 ∧ time_to_pass_through_tunnel x = 329.4 :=
sorry

end min_time_to_pass_l102_102384


namespace two_fifths_in_fraction_l102_102807

theorem two_fifths_in_fraction : 
  (∃ (k : ℚ), k = (9/3) / (2/5) ∧ k = 15/2) :=
by 
  sorry

end two_fifths_in_fraction_l102_102807


namespace child_ticket_cost_l102_102774

theorem child_ticket_cost :
  ∃ x : ℤ, (9 * 11 = 7 * x + 50) ∧ x = 7 :=
by
  sorry

end child_ticket_cost_l102_102774


namespace cross_number_puzzle_hundreds_digit_l102_102930

theorem cross_number_puzzle_hundreds_digit :
  ∃ a b : ℕ, a ≥ 5 ∧ a ≤ 6 ∧ b = 3 ∧ (3^a / 100 = 7 ∨ 7^b / 100 = 7) :=
sorry

end cross_number_puzzle_hundreds_digit_l102_102930


namespace algebra_expr_value_l102_102952

theorem algebra_expr_value (x y : ℝ) (h : x - 2 * y = 3) : 4 * y + 1 - 2 * x = -5 :=
sorry

end algebra_expr_value_l102_102952


namespace necessary_not_sufficient_l102_102140

theorem necessary_not_sufficient (a b : ℝ) : (a > b - 1) ∧ ¬ (a > b - 1 → a > b) := 
sorry

end necessary_not_sufficient_l102_102140


namespace reading_speed_increase_factor_l102_102554

-- Define Tom's normal reading speed as a constant rate
def tom_normal_speed := 12 -- pages per hour

-- Define the time period
def hours := 2 -- hours

-- Define the number of pages read in the given time period
def pages_read := 72 -- pages

-- Calculate the expected pages read at normal speed in the given time
def expected_pages := tom_normal_speed * hours -- should be 24 pages

-- Define the calculated factor by which the reading speed has increased
def expected_factor := pages_read / expected_pages -- should be 3

-- Prove that the factor is indeed 3
theorem reading_speed_increase_factor :
  expected_factor = 3 := by
  sorry

end reading_speed_increase_factor_l102_102554


namespace domain_of_f_l102_102092

noncomputable def f (x : ℝ) : ℝ := (x + 3) / (x^2 - 5 * x + 6)

theorem domain_of_f :
  {x : ℝ | x^2 - 5 * x + 6 ≠ 0} = {x : ℝ | x < 2} ∪ {x : ℝ | 2 < x ∧ x < 3} ∪ {x : ℝ | x > 3} :=
by
  sorry

end domain_of_f_l102_102092


namespace dvds_rented_l102_102458

def total_cost : ℝ := 4.80
def cost_per_dvd : ℝ := 1.20

theorem dvds_rented : total_cost / cost_per_dvd = 4 := 
by
  sorry

end dvds_rented_l102_102458


namespace jeanne_should_buy_more_tickets_l102_102331

theorem jeanne_should_buy_more_tickets :
  let cost_ferris_wheel := 5
  let cost_roller_coaster := 4
  let cost_bumper_cars := 4
  let jeanne_current_tickets := 5
  let total_tickets_needed := cost_ferris_wheel + cost_roller_coaster + cost_bumper_cars
  let tickets_needed_to_buy := total_tickets_needed - jeanne_current_tickets
  tickets_needed_to_buy = 8 :=
by
  sorry

end jeanne_should_buy_more_tickets_l102_102331


namespace children_difference_l102_102639

-- Define the initial number of children on the bus
def initial_children : ℕ := 5

-- Define the number of children who got off the bus
def children_off : ℕ := 63

-- Define the number of children on the bus after more got on
def final_children : ℕ := 14

-- Define the number of children who got on the bus
def children_on : ℕ := (final_children + children_off) - initial_children

-- Prove the number of children who got on minus the number of children who got off is equal to 9
theorem children_difference :
  (children_on - children_off) = 9 :=
by
  -- Direct translation from the proof steps
  sorry

end children_difference_l102_102639


namespace hyperbola_asymptote_slope_proof_l102_102763

noncomputable def hyperbola_asymptote_slope : ℝ :=
  let foci_distance := Real.sqrt ((8 - 2)^2 + (3 - 3)^2)
  let c := foci_distance / 2
  let a := 2  -- Given that 2a = 4
  let b := Real.sqrt (c^2 - a^2)
  b / a

theorem hyperbola_asymptote_slope_proof :
  ∀ x y : ℝ, 
  (Real.sqrt ((x - 2)^2 + (y - 3)^2) - Real.sqrt ((x - 8)^2 + (y - 3)^2) = 4) →
  hyperbola_asymptote_slope = Real.sqrt 5 / 2 :=
by
  sorry

end hyperbola_asymptote_slope_proof_l102_102763


namespace rationalize_and_divide_l102_102146

theorem rationalize_and_divide :
  (8 / Real.sqrt 8 / 2) = Real.sqrt 2 :=
by
  sorry

end rationalize_and_divide_l102_102146


namespace find_N_l102_102580

theorem find_N (N p q : ℝ) 
  (h1 : N / p = 4) 
  (h2 : N / q = 18) 
  (h3 : p - q = 0.5833333333333334) :
  N = 3 := 
sorry

end find_N_l102_102580


namespace time_for_c_l102_102641

   variable (A B C : ℚ)

   -- Conditions
   def condition1 : Prop := (A + B = 1/6)
   def condition2 : Prop := (B + C = 1/8)
   def condition3 : Prop := (C + A = 1/12)

   -- Theorem to be proved
   theorem time_for_c (h1 : condition1 A B) (h2 : condition2 B C) (h3 : condition3 C A) :
     1 / C = 48 :=
   sorry
   
end time_for_c_l102_102641


namespace number_of_three_leaf_clovers_l102_102985

theorem number_of_three_leaf_clovers (total_leaves : ℕ) (three_leaf_clover : ℕ) (four_leaf_clover : ℕ) (n : ℕ)
  (h1 : total_leaves = 40) (h2 : three_leaf_clover = 3) (h3 : four_leaf_clover = 4) (h4: total_leaves = 3 * n + 4) :
  n = 12 :=
by
  sorry

end number_of_three_leaf_clovers_l102_102985


namespace x_range_condition_l102_102826

-- Define the inequality and conditions
def inequality (x : ℝ) : Prop := x^2 + 2 * x < 8

-- The range of x must be (-4, 2)
theorem x_range_condition (x : ℝ) : inequality x → x > -4 ∧ x < 2 :=
by
  intro h
  sorry

end x_range_condition_l102_102826


namespace inequality_solution_l102_102353

theorem inequality_solution (x : ℝ) : x ∈ Set.Ioo (-7 : ℝ) (7 : ℝ) ↔ (x^2 - 49) / (x + 7) < 0 :=
by 
  sorry

end inequality_solution_l102_102353


namespace number_of_bricks_l102_102937

noncomputable def bricklayer_one_hours : ℝ := 8
noncomputable def bricklayer_two_hours : ℝ := 12
noncomputable def reduction_rate : ℝ := 12
noncomputable def combined_hours : ℝ := 6

theorem number_of_bricks (y : ℝ) :
  ((combined_hours * ((y / bricklayer_one_hours) + (y / bricklayer_two_hours) - reduction_rate)) = y) →
  y = 288 :=
by sorry

end number_of_bricks_l102_102937


namespace min_max_expression_l102_102383

theorem min_max_expression (x : ℝ) (h : 2 ≤ x ∧ x ≤ 7) :
  ∃ (a : ℝ) (b : ℝ), a = 11 / 3 ∧ b = 87 / 16 ∧ 
  (∀ y, 2 ≤ y ∧ y ≤ 7 → 11 / 3 ≤ (y^2 + 4*y + 10) / (2*y + 2)) ∧
  (∀ y, 2 ≤ y ∧ y ≤ 7 → (y^2 + 4*y + 10) / (2*y + 2) ≤ 87 / 16) :=
sorry

end min_max_expression_l102_102383


namespace find_a8_l102_102039

theorem find_a8 (a : ℕ → ℝ) (h1 : ∀ n : ℕ, n > 0 → a (n + 1) / (n + 1) = a n / n) (h2 : a 5 = 15) : a 8 = 24 :=
sorry

end find_a8_l102_102039


namespace solve_for_x_l102_102398

theorem solve_for_x (x : ℝ) (h : 3 * x + 20 = (1 / 3) * (7 * x + 45)) : x = -7.5 :=
sorry

end solve_for_x_l102_102398


namespace cube_and_difference_of_squares_l102_102915

theorem cube_and_difference_of_squares (x : ℤ) (h : x^3 = 9261) : (x + 1) * (x - 1) = 440 :=
by {
  sorry
}

end cube_and_difference_of_squares_l102_102915


namespace age_difference_l102_102576

theorem age_difference 
  (a b : ℕ) 
  (h1 : 0 ≤ a ∧ a < 10) 
  (h2 : 0 ≤ b ∧ b < 10) 
  (h3 : 10 * a + b + 5 = 3 * (10 * b + a + 5)) : 
  (10 * a + b) - (10 * b + a) = 63 := 
by
  sorry

end age_difference_l102_102576


namespace unique_solution_condition_l102_102644

noncomputable def unique_solution_system (a b c x y z : ℝ) : Prop :=
  (a * x + b * y - b * z = c) ∧ 
  (a * y + b * x - b * z = c) ∧ 
  (a * z + b * y - b * x = c) → 
  (x = y ∧ y = z ∧ x = c / a)

theorem unique_solution_condition (a b c x y z : ℝ) 
  (h1 : a * x + b * y - b * z = c)
  (h2 : a * y + b * x - b * z = c)
  (h3 : a * z + b * y - b * x = c)
  (ha : a ≠ 0)
  (ha_b : a ≠ b)
  (ha_b' : a + b ≠ 0) :
  unique_solution_system a b c x y z :=
by 
  sorry

end unique_solution_condition_l102_102644


namespace slope_of_line_l102_102141

theorem slope_of_line {x y : ℝ} : 
  (∃ (x y : ℝ), 0 = 3 * x + 4 * y + 12) → ∀ (m : ℝ), m = -3/4 :=
by
  sorry

end slope_of_line_l102_102141


namespace solution_l102_102958

-- Define the discount conditions
def discount (price : ℕ) : ℕ :=
  if price > 22 then price * 7 / 10 else
  if price < 20 then price * 8 / 10 else
  price

-- Define the given book prices
def book_prices : List ℕ := [25, 18, 21, 35, 12, 10]

-- Calculate total cost using the discount function
def total_cost (prices : List ℕ) : ℕ :=
  prices.foldl (λ acc price => acc + discount price) 0

def problem_statement : Prop :=
  total_cost book_prices = 95

theorem solution : problem_statement :=
  by
  unfold problem_statement
  unfold total_cost
  simp [book_prices, discount]
  sorry

end solution_l102_102958


namespace sum_remainders_l102_102844

theorem sum_remainders (n : ℤ) (h : n % 20 = 14) : (n % 4) + (n % 5) = 6 :=
  by
  sorry

end sum_remainders_l102_102844


namespace parabola_vertex_coordinates_l102_102350

theorem parabola_vertex_coordinates :
  ∀ x : ℝ, -x^2 + 15 ≥ -x^2 + 15 :=
by
  sorry

end parabola_vertex_coordinates_l102_102350


namespace circle_radius_l102_102532

theorem circle_radius {C : ℝ → ℝ → Prop} (h1 : C 4 0) (h2 : C (-4) 0) : ∃ r : ℝ, r = 4 :=
by
  -- sorry for brevity
  sorry

end circle_radius_l102_102532


namespace sequence_equality_l102_102841

noncomputable def a (x : ℝ) (n : ℕ) : ℝ := 1 + x^(n+1) + x^(n+2)

theorem sequence_equality (x : ℝ) (hx : x = 0 ∨ x = 1 ∨ x = -1) (n : ℕ) (hn : n ≥ 3) :
  (a x n)^2 = (a x (n-1)) * (a x (n+1)) :=
by sorry

end sequence_equality_l102_102841


namespace diagonal_length_of_rhombus_l102_102344

-- Definitions for the conditions
def side_length_of_square : ℝ := 8
def area_of_square : ℝ := side_length_of_square ^ 2
def area_of_rhombus : ℝ := 64
def d2 : ℝ := 8
-- Question
theorem diagonal_length_of_rhombus (d1 : ℝ) : (d1 * d2) / 2 = area_of_rhombus ↔ d1 = 16 := by
  sorry

end diagonal_length_of_rhombus_l102_102344


namespace exists_no_zero_digits_divisible_by_2_pow_100_l102_102020

theorem exists_no_zero_digits_divisible_by_2_pow_100 :
  ∃ (N : ℕ), (2^100 ∣ N) ∧ (∀ d ∈ (N.digits 10), d ≠ 0) := sorry

end exists_no_zero_digits_divisible_by_2_pow_100_l102_102020


namespace incenter_sum_equals_one_l102_102732

noncomputable def incenter (A B C : Point) : Point := sorry -- Definition goes here

def side_length (A B C : Point) (a b c : ℝ) : Prop :=
  -- Definitions relating to side lengths go here
  sorry

theorem incenter_sum_equals_one (A B C I : Point) (a b c IA IB IC : ℝ) (h_incenter : I = incenter A B C)
    (h_sides : side_length A B C a b c) :
    (IA ^ 2 / (b * c)) + (IB ^ 2 / (a * c)) + (IC ^ 2 / (a * b)) = 1 :=
  sorry

end incenter_sum_equals_one_l102_102732


namespace other_root_l102_102950

theorem other_root (m : ℝ) (h : 1^2 + m*1 + 3 = 0) : 
  ∃ α : ℝ, (1 + α = -m ∧ 1 * α = 3) ∧ α = 3 := 
by 
  sorry

end other_root_l102_102950


namespace find_f2_l102_102568

noncomputable def f : ℝ → ℝ := sorry

axiom function_property : ∀ (x : ℝ), f (2^x) + x * f (2^(-x)) = 1

theorem find_f2 : f 2 = 0 :=
by
  sorry

end find_f2_l102_102568


namespace three_digit_numbers_with_4_and_5_correct_l102_102134

def count_three_digit_numbers_with_4_and_5 : ℕ :=
  48

theorem three_digit_numbers_with_4_and_5_correct :
  count_three_digit_numbers_with_4_and_5 = 48 :=
by
  sorry -- proof goes here

end three_digit_numbers_with_4_and_5_correct_l102_102134


namespace playground_ratio_l102_102193

theorem playground_ratio (L B : ℕ) (playground_area landscape_area : ℕ) 
  (h1 : B = 8 * L)
  (h2 : B = 480)
  (h3 : playground_area = 3200)
  (h4 : landscape_area = L * B) : 
  (playground_area : ℚ) / landscape_area = 1 / 9 :=
by
  sorry

end playground_ratio_l102_102193


namespace part1_solution_part2_solution_l102_102947

noncomputable def f (x a : ℝ) := |x + a| + |x - a|

theorem part1_solution : (∀ x : ℝ, f x 1 ≥ 4 ↔ x ∈ Set.Iic (-2) ∨ x ∈ Set.Ici 2) := by
  sorry

theorem part2_solution : (∀ x : ℝ, f x a ≥ 6 → a ∈ Set.Iic (-3) ∨ a ∈ Set.Ici 3) := by
  sorry

end part1_solution_part2_solution_l102_102947


namespace asian_games_tourists_scientific_notation_l102_102559

theorem asian_games_tourists_scientific_notation : 
  ∀ (n : ℕ), n = 18480000 → 1.848 * (10:ℝ) ^ 7 = (n : ℝ) :=
by
  intro n
  sorry

end asian_games_tourists_scientific_notation_l102_102559


namespace general_formula_l102_102633

theorem general_formula (a : ℕ → ℕ) (h₀ : a 1 = 1) (h₁ : ∀ n : ℕ, n > 0 → a (n + 1) = 2 * a n + 1) :
  ∀ n : ℕ, a (n + 1) = 2^(n + 1) - 1 :=
by
  sorry

end general_formula_l102_102633


namespace solve_inequality_l102_102251

noncomputable def satisfies_inequality (x : ℝ) : Prop :=
  (x ^ 3 - 3 * x ^ 2 + 2 * x) / (x ^ 2 - 3 * x + 2) ≤ 0 ∧
  x ≠ 1 ∧ x ≠ 2

theorem solve_inequality :
  {x : ℝ | satisfies_inequality x} = {x : ℝ | x ≤ 0 ∧ x ≠ 1 ∧ x ≠ 2} :=
  sorry

end solve_inequality_l102_102251


namespace find_mistaken_divisor_l102_102894

-- Define the conditions
def remainder : ℕ := 0
def quotient_correct : ℕ := 32
def divisor_correct : ℕ := 21
def quotient_mistaken : ℕ := 56
def dividend : ℕ := quotient_correct * divisor_correct + remainder

-- Prove the mistaken divisor
theorem find_mistaken_divisor : ∃ x : ℕ, dividend = quotient_mistaken * x + remainder ∧ x = 12 :=
by
  -- We leave this as an exercise to the prover
  sorry

end find_mistaken_divisor_l102_102894


namespace range_of_x_coordinate_of_Q_l102_102265

def Point := ℝ × ℝ

def parabola (P : Point) : Prop :=
  P.2 = P.1 ^ 2

def vector (P Q : Point) : Point :=
  (Q.1 - P.1, Q.2 - P.2)

def dot_product (u v : Point) : ℝ :=
  u.1 * v.1 + u.2 * v.2

def perpendicular (P Q R : Point) : Prop :=
  dot_product (vector P Q) (vector P R) = 0

theorem range_of_x_coordinate_of_Q:
  ∀ (A P Q: Point), 
    A = (-1, 1) →
    parabola P →
    parabola Q →
    perpendicular P A Q →
    (Q.1 ≤ -3 ∨ Q.1 ≥ 1) :=
by
  intros A P Q hA hParabP hParabQ hPerp
  sorry

end range_of_x_coordinate_of_Q_l102_102265


namespace part1_part2_l102_102274

-- Given conditions for part (Ⅰ)
variables {a_n : ℕ → ℝ} {S_n : ℕ → ℝ}

-- The general formula for the sequence {a_n}
theorem part1 (a3_eq : a_n 3 = 1 / 8)
  (arith_seq : S_n 2 + 1 / 16 = 2 * S_n 3 - S_n 4) :
  ∀ n, a_n n = (1 / 2)^n := sorry

-- Given conditions for part (Ⅱ)
variables {b_n : ℕ → ℝ} {T_n : ℕ → ℝ}

-- The sum of the first n terms of the sequence {b_n}
theorem part2 (h_general : ∀ n, a_n n = (1 / 2)^n)
  (b_formula : ∀ n, b_n n = a_n n * (Real.log (a_n n) / Real.log (1 / 2))) :
  ∀ n, T_n n = 2 - (n + 2) / 2^n := sorry

end part1_part2_l102_102274


namespace minimum_triangle_area_l102_102964

theorem minimum_triangle_area (r a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a = b) : 
  ∀ T, (T = (a + b) * r / 2) → T = 2 * r * r :=
by 
  sorry

end minimum_triangle_area_l102_102964


namespace trapezoid_area_ratio_l102_102196

theorem trapezoid_area_ratio (b h x : ℝ) 
  (base_relation : b + 150 = x)
  (area_ratio : (3 / 7) * h * (b + 75) = (1 / 2) * h * (b + x))
  (mid_segment : x = b + 150) 
  : ⌊x^3 / 1000⌋ = 142 :=
by
  sorry

end trapezoid_area_ratio_l102_102196


namespace pairwise_coprime_triples_l102_102438

open Nat

theorem pairwise_coprime_triples (a b c : ℕ) 
  (h1 : a.gcd b = 1) (h2 : a.gcd c = 1) (h3 : b.gcd c = 1)
  (h4 : (a + b) ∣ c) (h5 : (a + c) ∣ b) (h6 : (b + c) ∣ a) :
  { (a, b, c) | (a = 1 ∧ b = 1 ∧ (c = 1 ∨ c = 2)) ∨ (a = 1 ∧ b = 2 ∧ c = 3) } :=
by
  -- Proof omitted for conciseness
  sorry

end pairwise_coprime_triples_l102_102438


namespace number_of_subsets_of_five_element_set_is_32_l102_102893

theorem number_of_subsets_of_five_element_set_is_32 (M : Finset ℕ) (h : M.card = 5) :
    (2 : ℕ) ^ 5 = 32 :=
by
  sorry

end number_of_subsets_of_five_element_set_is_32_l102_102893


namespace sum_three_consecutive_integers_divisible_by_three_l102_102560

theorem sum_three_consecutive_integers_divisible_by_three (a : ℕ) (h : 1 < a) :
  (a - 1) + a + (a + 1) % 3 = 0 :=
by
  sorry

end sum_three_consecutive_integers_divisible_by_three_l102_102560


namespace ratio_shirt_to_coat_l102_102982

-- Define the given conditions
def total_cost := 600
def shirt_cost := 150

-- Define the coat cost based on the given conditions
def coat_cost := total_cost - shirt_cost

-- State the theorem to prove the ratio of shirt cost to coat cost is 1:3
theorem ratio_shirt_to_coat : (shirt_cost : ℚ) / (coat_cost : ℚ) = 1 / 3 :=
by
  -- The proof would go here
  sorry

end ratio_shirt_to_coat_l102_102982


namespace probability_identical_cubes_l102_102812

-- Definitions translating given conditions
def total_ways_to_paint_single_cube : Nat := 3^6
def total_ways_to_paint_three_cubes : Nat := total_ways_to_paint_single_cube^3

-- Cases counting identical painting schemes
def identical_painting_schemes : Nat :=
  let case_A := 3
  let case_B := 90
  let case_C := 540
  case_A + case_B + case_C

-- The main theorem stating the desired probability
theorem probability_identical_cubes :
  let total_ways := (387420489 : ℚ) -- 729^3
  let favorable_ways := (633 : ℚ)  -- sum of all cases (3 + 90 + 540)
  favorable_ways / total_ways = (211 / 129140163 : ℚ) :=
by
  sorry

end probability_identical_cubes_l102_102812


namespace multiple_of_6_is_multiple_of_3_l102_102701

theorem multiple_of_6_is_multiple_of_3 (n : ℕ) : (∃ k : ℕ, n = 6 * k) → (∃ m : ℕ, n = 3 * m) :=
by
  sorry

end multiple_of_6_is_multiple_of_3_l102_102701


namespace bottle_and_beverage_weight_l102_102592

theorem bottle_and_beverage_weight 
  (B : ℝ)  -- Weight of the bottle in kilograms
  (x : ℝ)  -- Original weight of the beverage in kilograms
  (h1 : B + 2 * x = 5)  -- Condition: double the beverage weight total
  (h2 : B + 4 * x = 9)  -- Condition: quadruple the beverage weight total
: x = 2 ∧ B = 1 := 
by
  sorry

end bottle_and_beverage_weight_l102_102592


namespace count_of_integer_values_not_satisfying_inequality_l102_102115

theorem count_of_integer_values_not_satisfying_inequality :
  ∃ n : ℕ, n = 8 ∧ ∀ x : ℤ, (3 * x^2 + 11 * x + 10 ≤ 17) ↔ (x = -7 ∨ x = -6 ∨ x = -5 ∨ x = -4 ∨ x = -3 ∨ x = -2 ∨ x = -1 ∨ x = 0) :=
by sorry

end count_of_integer_values_not_satisfying_inequality_l102_102115


namespace geometric_sequence_l102_102243

theorem geometric_sequence (a b c r : ℤ) (h1 : b = a * r) (h2 : c = a * r^2) (h3 : c = a + 56) : b = 21 :=
by sorry

end geometric_sequence_l102_102243


namespace sum_n_k_l102_102919

theorem sum_n_k (n k : ℕ) (h1 : 3 * (k + 1) = n - k) (h2 : 2 * (k + 2) = n - k - 1) : n + k = 13 := by
  sorry

end sum_n_k_l102_102919


namespace increasing_function_range_a_l102_102017

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x ≤ 0 then (a - 1) * x + 3 * a - 4 else a^x

theorem increasing_function_range_a (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₂ - f a x₁) / (x₂ - x₁) > 0) ↔ 1 < a ∧ a ≤ 5 / 3 :=
sorry

end increasing_function_range_a_l102_102017


namespace equation1_solution_equation2_solution_equation3_solution_l102_102215

theorem equation1_solution :
  ∀ x : ℝ, x^2 - 2 * x - 99 = 0 ↔ x = 11 ∨ x = -9 :=
by
  sorry

theorem equation2_solution :
  ∀ x : ℝ, x^2 + 5 * x = 7 ↔ x = (-5 - Real.sqrt 53) / 2 ∨ x = (-5 + Real.sqrt 53) / 2 :=
by
  sorry

theorem equation3_solution :
  ∀ x : ℝ, 4 * x * (2 * x + 1) = 3 * (2 * x + 1) ↔ x = -1/2 ∨ x = 3/4 :=
by
  sorry

end equation1_solution_equation2_solution_equation3_solution_l102_102215


namespace theo_eggs_needed_l102_102587

def customers_first_hour : ℕ := 5
def customers_second_hour : ℕ := 7
def customers_third_hour : ℕ := 3
def customers_fourth_hour : ℕ := 8
def eggs_per_3_egg_omelette : ℕ := 3
def eggs_per_4_egg_omelette : ℕ := 4

theorem theo_eggs_needed :
  (customers_first_hour * eggs_per_3_egg_omelette) +
  (customers_second_hour * eggs_per_4_egg_omelette) +
  (customers_third_hour * eggs_per_3_egg_omelette) +
  (customers_fourth_hour * eggs_per_4_egg_omelette) = 84 := by
  sorry

end theo_eggs_needed_l102_102587


namespace mother_returns_home_at_8_05_l102_102037

noncomputable
def xiaoMing_home_time : Nat := 7 * 60 -- 7:00 AM in minutes
def xiaoMing_speed : Nat := 40 -- in meters per minute
def mother_home_time : Nat := 7 * 60 + 20 -- 7:20 AM in minutes
def meet_point : Nat := 1600 -- in meters
def stay_time : Nat := 5 -- in minutes
def return_duration_by_bike : Nat := 20 -- in minutes

theorem mother_returns_home_at_8_05 :
    (xiaoMing_home_time + (meet_point / xiaoMing_speed) + stay_time + return_duration_by_bike) = (8 * 60 + 5) :=
by
    sorry

end mother_returns_home_at_8_05_l102_102037


namespace period_and_monotonic_interval_range_of_f_l102_102745

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 3 / 3) * Real.cos (2 * x) + Real.sin (x + Real.pi / 4) ^ 2

theorem period_and_monotonic_interval :
  (∃ T > 0, ∀ x, f (x + T) = f x) ∧ 
  (∃ k : ℤ, ∀ x, x ∈ Set.Icc (k * Real.pi - 5 * Real.pi / 12) (k * Real.pi + Real.pi / 12) →
    MonotoneOn f (Set.Icc (2 * k * Real.pi - Real.pi / 2) (2 * k * Real.pi + Real.pi / 2))) :=
sorry

theorem range_of_f (x : ℝ) (hx : x ∈ Set.Icc (-Real.pi / 12) (5 * Real.pi / 12)) :
  f x ∈ Set.Icc 0 (3 / 2) :=
sorry

end period_and_monotonic_interval_range_of_f_l102_102745


namespace math_crackers_initial_l102_102739

def crackers_initial (gave_each : ℕ) (left : ℕ) (num_friends : ℕ) : ℕ :=
  (gave_each * num_friends) + left

theorem math_crackers_initial :
  crackers_initial 7 17 3 = 38 :=
by
  -- The definition of crackers_initial and the theorem statement should be enough.
  -- The exact proof is left as a sorry placeholder.
  sorry

end math_crackers_initial_l102_102739


namespace lyndee_friends_count_l102_102546

-- Definitions
variables (total_chicken total_garlic_bread : ℕ)
variables (lyndee_chicken lyndee_garlic_bread : ℕ)
variables (friends_large_chicken_count : ℕ)
variables (friends_large_chicken : ℕ)
variables (friend_garlic_bread_per_friend : ℕ)

def remaining_chicken (total_chicken lyndee_chicken friends_large_chicken_count friends_large_chicken : ℕ) : ℕ :=
  total_chicken - (lyndee_chicken + friends_large_chicken_count * friends_large_chicken)

def remaining_garlic_bread (total_garlic_bread lyndee_garlic_bread : ℕ) : ℕ :=
  total_garlic_bread - lyndee_garlic_bread

def total_friends (friends_large_chicken_count remaining_chicken remaining_garlic_bread friend_garlic_bread_per_friend : ℕ) : ℕ :=
  friends_large_chicken_count + remaining_chicken + remaining_garlic_bread / friend_garlic_bread_per_friend

-- Theorem statement
theorem lyndee_friends_count : 
  total_chicken = 11 → 
  total_garlic_bread = 15 →
  lyndee_chicken = 1 →
  lyndee_garlic_bread = 1 →
  friends_large_chicken_count = 3 →
  friends_large_chicken = 2 →
  friend_garlic_bread_per_friend = 3 →
  total_friends friends_large_chicken_count 
                (remaining_chicken total_chicken lyndee_chicken friends_large_chicken_count friends_large_chicken)
                (remaining_garlic_bread total_garlic_bread lyndee_garlic_bread)
                friend_garlic_bread_per_friend = 7 := 
by 
  intros h1 h2 h3 h4 h5 h6 h7
  -- Proof omitted
  sorry

end lyndee_friends_count_l102_102546


namespace directrix_parabola_l102_102999

theorem directrix_parabola (x y : ℝ) (h : x^2 = 8 * y) : y = -2 :=
sorry

end directrix_parabola_l102_102999


namespace basketball_weight_l102_102876

theorem basketball_weight (b k : ℝ) (h1 : 6 * b = 4 * k) (h2 : 3 * k = 72) : b = 16 :=
by
  sorry

end basketball_weight_l102_102876


namespace vertical_line_divides_triangle_l102_102694

theorem vertical_line_divides_triangle (k : ℝ) :
  let triangle_area := 1 / 2 * |0 * (1 - 1) + 1 * (1 - 0) + 9 * (0 - 1)|
  let left_triangle_area := 1 / 2 * |0 * (1 - 1) + k * (1 - 0) + 1 * (0 - 1)|
  let right_triangle_area := triangle_area - left_triangle_area
  triangle_area = 4 
  ∧ left_triangle_area = 2
  ∧ right_triangle_area = 2
  ∧ (k = 5) ∨ (k = -3) → 
  k = 5 :=
by
  sorry

end vertical_line_divides_triangle_l102_102694


namespace tan_double_angle_l102_102943

theorem tan_double_angle (θ : ℝ) 
  (h1 : Real.sin θ = 4 / 5) 
  (h2 : Real.sin θ - Real.cos θ > 1) : 
  Real.tan (2 * θ) = 24 / 7 := 
sorry

end tan_double_angle_l102_102943


namespace largest_consecutive_sum_55_l102_102729

theorem largest_consecutive_sum_55 :
  ∃ n a : ℕ, (n * (a + (n - 1) / 2) = 55) ∧ (n = 10) ∧ (∀ m : ℕ, ∀ b : ℕ, (m * (b + (m - 1) / 2) = 55) → (m ≤ 10)) :=
by 
  sorry

end largest_consecutive_sum_55_l102_102729


namespace MissyTotalTVTime_l102_102569

theorem MissyTotalTVTime :
  let reality_shows := [28, 35, 42, 39, 29]
  let cartoons := [10, 10]
  let ad_breaks := [8, 6, 12]
  let total_time := reality_shows.sum + cartoons.sum + ad_breaks.sum
  total_time = 219 := by
{
  -- Lean proof logic goes here (proof not requested)
  sorry
}

end MissyTotalTVTime_l102_102569


namespace area_relation_l102_102771

open Real

noncomputable def S_OMN (a b c d θ : ℝ) : ℝ := 1 / 2 * abs (b * c - a * d) * sin θ
noncomputable def S_ABCD (a b c d θ : ℝ) : ℝ := 2 * abs (b * c - a * d) * sin θ

theorem area_relation (a b c d θ : ℝ) (hθ : 0 < θ ∧ θ < π / 2) : 
    4 * (S_OMN a b c d θ) = S_ABCD a b c d θ :=
by
  sorry

end area_relation_l102_102771


namespace probability_intersection_three_elements_l102_102060

theorem probability_intersection_three_elements (U : Finset ℕ) (hU : U = {1, 2, 3, 4, 5}) : 
  ∃ (p : ℚ), p = 5 / 62 :=
by
  sorry

end probability_intersection_three_elements_l102_102060


namespace largest_of_five_consecutive_sum_180_l102_102803

theorem largest_of_five_consecutive_sum_180 (n : ℕ) (h : n + (n + 1) + (n + 2) + (n + 3) + (n + 4) = 180) :
  n + 4 = 38 :=
by
  sorry

end largest_of_five_consecutive_sum_180_l102_102803


namespace certain_number_modulo_l102_102780

theorem certain_number_modulo (x : ℕ) : (57 * x) % 8 = 7 ↔ x = 1 := by
  sorry

end certain_number_modulo_l102_102780


namespace total_crayons_correct_l102_102880

-- Define the number of crayons each child has
def crayons_per_child : ℕ := 12

-- Define the number of children
def number_of_children : ℕ := 18

-- Define the total number of crayons
def total_crayons : ℕ := crayons_per_child * number_of_children

-- State the theorem
theorem total_crayons_correct : total_crayons = 216 :=
by
  -- Proof goes here
  sorry

end total_crayons_correct_l102_102880


namespace m_ducks_l102_102387

variable (M C K : ℕ)

theorem m_ducks :
  (M = C + 4) ∧
  (M = 2 * C + K + 3) ∧
  (M + C + K = 90) →
  M = 89 := by
  sorry

end m_ducks_l102_102387


namespace road_greening_cost_l102_102684

-- Define constants for the conditions
def l_total : ℕ := 1500
def cost_A : ℕ := 22
def cost_B : ℕ := 25

-- Define variables for the cost per stem
variables (x y : ℕ)

-- Define the conditions from Plan A and Plan B
def plan_A (x y : ℕ) : Prop := 2 * x + 3 * y = cost_A
def plan_B (x y : ℕ) : Prop := x + 5 * y = cost_B

-- System of equations to find x and y
def system_of_equations (x y : ℕ) : Prop := plan_A x y ∧ plan_B x y

-- Define the constraint for the length of road greened according to Plan B
def length_constraint (a : ℕ) : Prop := l_total - a ≥ 2 * a

-- Define the total cost function
def total_cost (a : ℕ) (x y : ℕ) : ℕ := 22 * a + (x + 5 * y) * (l_total - a)

-- Prove the cost per stem and the minimized cost
theorem road_greening_cost :
  (∃ x y, system_of_equations x y ∧ x = 5 ∧ y = 4) ∧
  (∃ a : ℕ, length_constraint a ∧ a = 500 ∧ total_cost a 5 4 = 36000) :=
by
  -- This is where the proof would go
  sorry

end road_greening_cost_l102_102684


namespace plane_equation_correct_l102_102552

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def vectorBC (B C : Point3D) : Point3D :=
  { x := C.x - B.x, y := C.y - B.y, z := C.z - B.z }

def planeEquation (n : Point3D) (A : Point3D) (P : Point3D) : ℝ :=
  n.x * (P.x - A.x) + n.y * (P.y - A.y) + n.z * (P.z - A.z)

theorem plane_equation_correct :
  let A := ⟨3, -3, -6⟩
  let B := ⟨1, 9, -5⟩
  let C := ⟨6, 6, -4⟩
  let n := vectorBC B C
  ∀ P, planeEquation n A P = 0 ↔ 5 * (P.x - A.x) - 3 * (P.y - A.y) + 1 * (P.z - A.z) - 18 = 0 :=
by
  sorry

end plane_equation_correct_l102_102552


namespace trig_inequality_2016_l102_102791

theorem trig_inequality_2016 :
  let a := Real.sin (Real.cos (2016 * Real.pi / 180))
  let b := Real.sin (Real.sin (2016 * Real.pi / 180))
  let c := Real.cos (Real.sin (2016 * Real.pi / 180))
  let d := Real.cos (Real.cos (2016 * Real.pi / 180))
  c > d ∧ d > b ∧ b > a := by
  sorry

end trig_inequality_2016_l102_102791


namespace min_value_func_l102_102558

noncomputable def func (x : ℝ) : ℝ := Real.sin x + Real.sqrt 3 * Real.cos x

theorem min_value_func : ∃ x : ℝ, func x = -2 :=
by
  existsi (Real.pi / 2 + Real.pi / 3)
  sorry

end min_value_func_l102_102558


namespace fresh_pineapples_left_l102_102145

namespace PineappleStore

def initial := 86
def sold := 48
def rotten := 9

theorem fresh_pineapples_left (initial sold rotten : ℕ) (h_initial : initial = 86) (h_sold : sold = 48) (h_rotten : rotten = 9) :
  initial - sold - rotten = 29 :=
by sorry

end PineappleStore

end fresh_pineapples_left_l102_102145


namespace arithmetic_sequence_sixth_term_l102_102697

theorem arithmetic_sequence_sixth_term (a d : ℤ) 
    (sum_first_five : a + (a + d) + (a + 2 * d) + (a + 3 * d) + (a + 4 * d) = 15)
    (fourth_term : a + 3 * d = 4) : a + 5 * d = 6 :=
by
  sorry

end arithmetic_sequence_sixth_term_l102_102697


namespace y_payment_is_approximately_272_73_l102_102306

noncomputable def calc_y_payment : ℝ :=
  let total_payment : ℝ := 600
  let percent_x_to_y : ℝ := 1.2
  total_payment / (percent_x_to_y + 1)

theorem y_payment_is_approximately_272_73
  (total_payment : ℝ)
  (percent_x_to_y : ℝ)
  (h1 : total_payment = 600)
  (h2 : percent_x_to_y = 1.2) :
  calc_y_payment = 272.73 :=
by
  sorry

end y_payment_is_approximately_272_73_l102_102306


namespace unique_positive_x_for_volume_l102_102872

variable (x : ℕ)

def prism_volume (x : ℕ) : ℕ :=
  (x + 5) * (x - 5) * (x ^ 2 + 25)

theorem unique_positive_x_for_volume {x : ℕ} (h : prism_volume x < 700) (h_pos : 0 < x) :
  ∃! x, (prism_volume x < 700) ∧ (x - 5 > 0) :=
by
  sorry

end unique_positive_x_for_volume_l102_102872


namespace vector_collinearity_l102_102848

variables (a b : ℝ × ℝ)

def collinear (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

theorem vector_collinearity : collinear (-1, 2) (1, -2) :=
by
  sorry

end vector_collinearity_l102_102848


namespace a5_value_l102_102408

def seq (a : ℕ → ℤ) (a1 : a 1 = 2) (rec : ∀ n, a (n + 1) = 2 * a n - 1) : Prop := True

theorem a5_value : 
  ∀ (a : ℕ → ℤ)
  (h1 : a 1 = 2)
  (recurrence : ∀ n, a (n + 1) = 2 * a n - 1),
  seq a h1 recurrence → a 5 = 17 :=
by
  intros a h1 recurrence seq_a
  sorry

end a5_value_l102_102408


namespace smallest_n_congruent_5n_eq_n5_mod_7_l102_102444

theorem smallest_n_congruent_5n_eq_n5_mod_7 : ∃ (n : ℕ), n > 0 ∧ (∀ m > 0, 5^m % 7 ≠ m^5 % 7 → m ≥ n) :=
by
  use 6
  -- Proof steps here which are skipped
  sorry

end smallest_n_congruent_5n_eq_n5_mod_7_l102_102444


namespace find_number_l102_102186

theorem find_number (x : ℝ) (h : (x / 6) * 12 = 9) : x = 4.5 :=
by
  sorry

end find_number_l102_102186


namespace problem_1_problem_2_l102_102302

noncomputable def A : Set ℝ := {x | 2 * x^2 - 7 * x + 3 ≤ 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | x + a < 0}

theorem problem_1 (a : ℝ) :
  a = -2 →
  A ∩ B a = {x | (1 / 2 : ℝ) ≤ x ∧ x < 2} :=
by
  intro ha
  sorry

theorem problem_2 (a : ℝ) :
  (A ∩ B a) = A → a < -3 :=
by
  intro h
  sorry

end problem_1_problem_2_l102_102302


namespace pow_div_pow_l102_102456

variable (a : ℝ)
variable (A B : ℕ)

theorem pow_div_pow (a : ℝ) (A B : ℕ) : a^A / a^B = a^(A - B) :=
  sorry

example : a^6 / a^2 = a^4 :=
  pow_div_pow a 6 2

end pow_div_pow_l102_102456


namespace team_leader_and_deputy_choice_l102_102924

def TeamLeaderSelection : Type := {x : Fin 5 // true}
def DeputyLeaderSelection (TL : TeamLeaderSelection) : Type := {x : Fin 5 // x ≠ TL.val}

theorem team_leader_and_deputy_choice : 
  (Σ TL : TeamLeaderSelection, DeputyLeaderSelection TL) → Fin 20 :=
by sorry

end team_leader_and_deputy_choice_l102_102924


namespace average_percentage_decrease_l102_102637

theorem average_percentage_decrease
  (original_price final_price : ℕ)
  (h_original_price : original_price = 2000)
  (h_final_price : final_price = 1280) :
  (original_price - final_price) / original_price * 100 / 2 = 18 :=
by 
  sorry

end average_percentage_decrease_l102_102637


namespace abs_square_implication_l102_102766

theorem abs_square_implication (a b : ℝ) (h : abs a > abs b) : a^2 > b^2 :=
by sorry

end abs_square_implication_l102_102766


namespace find_honeydews_left_l102_102101

theorem find_honeydews_left 
  (cantaloupe_price : ℕ)
  (honeydew_price : ℕ)
  (initial_cantaloupes : ℕ)
  (initial_honeydews : ℕ)
  (dropped_cantaloupes : ℕ)
  (rotten_honeydews : ℕ)
  (end_cantaloupes : ℕ)
  (total_revenue : ℕ)
  (honeydews_left : ℕ) :
  cantaloupe_price = 2 →
  honeydew_price = 3 →
  initial_cantaloupes = 30 →
  initial_honeydews = 27 →
  dropped_cantaloupes = 2 →
  rotten_honeydews = 3 →
  end_cantaloupes = 8 →
  total_revenue = 85 →
  honeydews_left = 9 :=
by
  sorry

end find_honeydews_left_l102_102101


namespace rationalize_denominator_l102_102129

theorem rationalize_denominator :
  (2 + Real.sqrt 5) / (2 - Real.sqrt 5) = -9 - 4 * Real.sqrt 5 :=
by
  -- Commutative field properties and algebraic manipulation will be used here.
  sorry

end rationalize_denominator_l102_102129


namespace leak_out_time_l102_102948

theorem leak_out_time (T_A T_full : ℝ) (h1 : T_A = 16) (h2 : T_full = 80) :
  ∃ T_B : ℝ, (1 / T_A - 1 / T_B = 1 / T_full) ∧ T_B = 80 :=
by {
  sorry
}

end leak_out_time_l102_102948


namespace min_sum_x8y4z_l102_102939

theorem min_sum_x8y4z (x y z : ℝ) (h : 4 / x + 2 / y + 1 / z = 1) : x + 8 * y + 4 * z ≥ 64 := 
sorry

end min_sum_x8y4z_l102_102939


namespace angle_terminal_side_on_non_negative_y_axis_l102_102800

theorem angle_terminal_side_on_non_negative_y_axis (P : ℝ × ℝ) (α : ℝ) (hP : P = (0, 3)) :
  α = some_angle_with_terminal_side_on_non_negative_y_axis := by
  sorry

end angle_terminal_side_on_non_negative_y_axis_l102_102800


namespace x_expression_l102_102825

noncomputable def f (t : ℝ) : ℝ := t / (1 - t)

theorem x_expression {x y : ℝ} (hx : x ≠ 1) (hy : y = f x) : x = y / (1 + y) :=
by {
  sorry
}

end x_expression_l102_102825


namespace two_point_five_one_million_in_scientific_notation_l102_102462

theorem two_point_five_one_million_in_scientific_notation :
  (2.51 * 10^6 : ℝ) = 2.51e6 := 
sorry

end two_point_five_one_million_in_scientific_notation_l102_102462


namespace number_of_types_of_sliced_meat_l102_102954

-- Define the constants and conditions
def varietyPackCostWithoutRush := 40.00
def rushDeliveryPercentage := 0.30
def costPerTypeWithRush := 13.00
def totalCostWithRush := varietyPackCostWithoutRush + (rushDeliveryPercentage * varietyPackCostWithoutRush)

-- Define the statement that needs to be proven
theorem number_of_types_of_sliced_meat :
  (totalCostWithRush / costPerTypeWithRush) = 4 := by
  sorry

end number_of_types_of_sliced_meat_l102_102954


namespace corrected_mean_l102_102485

theorem corrected_mean (incorrect_mean : ℕ) (num_observations : ℕ) (wrong_value actual_value : ℕ) : 
  (50 * 36 + (43 - 23)) / 50 = 36.4 :=
by
  sorry

end corrected_mean_l102_102485


namespace r_daily_earning_l102_102526

-- Definitions from conditions in the problem
def earnings_of_all (P Q R : ℕ) : Prop := 9 * (P + Q + R) = 1620
def earnings_p_and_r (P R : ℕ) : Prop := 5 * (P + R) = 600
def earnings_q_and_r (Q R : ℕ) : Prop := 7 * (Q + R) = 910

-- Theorem to prove the daily earnings of r
theorem r_daily_earning (P Q R : ℕ) 
    (h1 : earnings_of_all P Q R)
    (h2 : earnings_p_and_r P R)
    (h3 : earnings_q_and_r Q R) : 
    R = 70 := 
by 
  sorry

end r_daily_earning_l102_102526


namespace find_x_l102_102917

theorem find_x (x : ℕ) 
  (h : (744 + 745 + 747 + 748 + 749 + 752 + 752 + 753 + 755 + x) / 10 = 750) : 
  x = 1255 := 
sorry

end find_x_l102_102917


namespace john_cuts_his_grass_to_l102_102730

theorem john_cuts_his_grass_to (growth_rate monthly_cost annual_cost cut_height : ℝ)
  (h : ℝ) : 
  growth_rate = 0.5 ∧ monthly_cost = 100 ∧ annual_cost = 300 ∧ cut_height = 4 →
  h = 2 := by
  intros conditions
  sorry

end john_cuts_his_grass_to_l102_102730


namespace rectangle_dimensions_l102_102692

-- Definitions from conditions
def is_rectangle (length width : ℝ) : Prop :=
  3 * width = length ∧ 3 * width^2 = 8 * width

-- The theorem to prove
theorem rectangle_dimensions :
  ∃ (length width : ℝ), is_rectangle length width ∧ width = 8 / 3 ∧ length = 8 := by
  sorry

end rectangle_dimensions_l102_102692


namespace problem_solution_l102_102768

variable {f : ℕ → ℕ}
variable (h_mul : ∀ a b : ℕ, f (a + b) = f a * f b)
variable (h_one : f 1 = 2)

theorem problem_solution : 
  (f 2 / f 1) + (f 4 / f 3) + (f 6 / f 5) + (f 8 / f 7) + (f 10 / f 9) = 10 :=
by
  sorry

end problem_solution_l102_102768


namespace wrapping_paper_fraction_l102_102976

theorem wrapping_paper_fraction (s l : ℚ) (h1 : 4 * s + 2 * l = 5 / 12) (h2 : l = 2 * s) :
  s = 5 / 96 ∧ l = 5 / 48 :=
by
  sorry

end wrapping_paper_fraction_l102_102976


namespace toys_profit_l102_102888

theorem toys_profit (sp cp : ℕ) (x : ℕ) (h1 : sp = 25200) (h2 : cp = 1200) (h3 : 18 * cp + x * cp = sp) :
  x = 3 :=
by
  sorry

end toys_profit_l102_102888


namespace complex_modulus_squared_l102_102808

theorem complex_modulus_squared (z : ℂ) (h : z^2 + Complex.abs z^2 = 4 + 6 * Complex.I) : Complex.abs z^2 = 13 / 2 :=
by
  sorry

end complex_modulus_squared_l102_102808


namespace mean_values_are_two_l102_102651

noncomputable def verify_means (a b : ℝ) : Prop :=
  (a + b) / 2 = 2 ∧ 2 / ((1 / a) + (1 / b)) = 2

theorem mean_values_are_two (a b : ℝ) (h : verify_means a b) : a = 2 ∧ b = 2 :=
  sorry

end mean_values_are_two_l102_102651


namespace value_of_k_h_5_l102_102067

def h (x : ℝ) : ℝ := 4 * x + 6
def k (x : ℝ) : ℝ := 6 * x - 8

theorem value_of_k_h_5 : k (h 5) = 148 :=
by
  have h5 : h 5 = 4 * 5 + 6 := rfl
  simp [h5, h, k]
  sorry

end value_of_k_h_5_l102_102067


namespace minimum_sum_of_distances_squared_l102_102399

-- Define the points A and B
structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := -2, y := 0 }
def B : Point := { x := 2, y := 0 }

-- Define the moving point P on the circle
def on_circle (P : Point) : Prop :=
  (P.x - 3)^2 + (P.y - 4)^2 = 4

-- Distance squared between two points
def dist_squared (P Q : Point) : ℝ :=
  (P.x - Q.x)^2 + (P.y - Q.y)^2

-- Define the sum of squared distances from P to points A and B
def sum_distances_squared (P : Point) : ℝ :=
  dist_squared P A + dist_squared P B

-- Statement of the proof problem
theorem minimum_sum_of_distances_squared :
  ∃ P : Point, on_circle P ∧ sum_distances_squared P = 26 :=
sorry

end minimum_sum_of_distances_squared_l102_102399


namespace max_remainder_209_lt_120_l102_102237

theorem max_remainder_209_lt_120 : 
  ∃ n : ℕ, n < 120 ∧ (209 % n = 104) := 
sorry

end max_remainder_209_lt_120_l102_102237


namespace original_number_is_64_l102_102272

theorem original_number_is_64 (x : ℕ) : 500 + x = 9 * x - 12 → x = 64 :=
by
  sorry

end original_number_is_64_l102_102272


namespace Andrew_runs_2_miles_each_day_l102_102779

theorem Andrew_runs_2_miles_each_day
  (A : ℕ)
  (Peter_runs : ℕ := A + 3)
  (total_miles_after_5_days : 5 * (A + Peter_runs) = 35) :
  A = 2 :=
by
  sorry

end Andrew_runs_2_miles_each_day_l102_102779


namespace trig_identity_l102_102171

theorem trig_identity (x : ℝ) (h : Real.sin (π / 6 - x) = 1 / 2) :
  Real.sin (19 * π / 6 - x) + Real.sin (-2 * π / 3 + x) ^ 2 = 1 / 4 :=
by
  sorry

end trig_identity_l102_102171


namespace find_n_l102_102475

theorem find_n (n : ℕ) :
  Int.lcm n 16 = 52 ∧ Nat.gcd n 16 = 8 → n = 26 :=
by
  sorry

end find_n_l102_102475


namespace transform_graph_of_g_to_f_l102_102926

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x)^2 - Real.sqrt 3 * Real.sin (2 * x)
noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (2 * x) + 1

theorem transform_graph_of_g_to_f :
  ∀ (x : ℝ), f x = g (x + (5 * Real.pi) / 12) :=
by
  sorry

end transform_graph_of_g_to_f_l102_102926


namespace jacque_suitcase_weight_l102_102372

noncomputable def suitcase_weight_return (original_weight : ℝ)
                                         (perfume_weight_oz : ℕ → ℝ)
                                         (chocolate_weight_lb : ℝ)
                                         (soap_weight_oz : ℕ → ℝ)
                                         (jam_weight_oz : ℕ → ℝ)
                                         (sculpture_weight_kg : ℝ)
                                         (shirt_weight_g : ℕ → ℝ)
                                         (oz_to_lb : ℝ)
                                         (kg_to_lb : ℝ)
                                         (g_to_kg : ℝ) : ℝ :=
  original_weight +
  (perfume_weight_oz 5 / oz_to_lb) +
  chocolate_weight_lb +
  (soap_weight_oz 2 / oz_to_lb) +
  (jam_weight_oz 2 / oz_to_lb) +
  (sculpture_weight_kg * kg_to_lb) +
  ((shirt_weight_g 3 / g_to_kg) * kg_to_lb)

theorem jacque_suitcase_weight :
  suitcase_weight_return 12 
                        (fun n => n * 1.2) 
                        4 
                        (fun n => n * 5) 
                        (fun n => n * 8)
                        3.5 
                        (fun n => n * 300) 
                        16 
                        2.20462 
                        1000 
  = 27.70 :=
sorry

end jacque_suitcase_weight_l102_102372


namespace tan_double_angle_l102_102121

theorem tan_double_angle (theta : ℝ) (h : 2 * Real.sin theta + Real.cos theta = 0) :
  Real.tan (2 * theta) = - 4 / 3 :=
sorry

end tan_double_angle_l102_102121


namespace range_of_a_l102_102118

variable {x a : ℝ}

def p (x a : ℝ) : Prop := x > a
def q (x : ℝ) : Prop := x^2 + x - 2 > 0

theorem range_of_a 
  (h_sufficient : ∀ x, p x a → q x)
  (h_not_necessary : ∃ x, q x ∧ ¬ p x a) :
  a ≥ 1 :=
sorry

end range_of_a_l102_102118


namespace find_k_exists_p3_p5_no_number_has_p2_and_p4_l102_102674

def has_prop_pk (n k : ℕ) : Prop := ∃ lst : List ℕ, (∀ x ∈ lst, x > 1) ∧ (lst.length = k) ∧ (lst.prod = n)

theorem find_k_exists_p3_p5 :
  ∃ (k : ℕ), (k = 3) ∧ ∃ (n : ℕ), has_prop_pk n k ∧ has_prop_pk n (k + 2) :=
by {
  sorry
}

theorem no_number_has_p2_and_p4 :
  ¬ ∃ (n : ℕ), has_prop_pk n 2 ∧ has_prop_pk n 4 :=
by {
  sorry
}

end find_k_exists_p3_p5_no_number_has_p2_and_p4_l102_102674


namespace inequality_solution_nonempty_l102_102421

theorem inequality_solution_nonempty (a : ℝ) :
  (∃ x : ℝ, x ^ 2 - a * x - a ≤ -3) ↔ (a ≤ -6 ∨ a ≥ 2) :=
by
  sorry

end inequality_solution_nonempty_l102_102421


namespace boat_speed_still_water_l102_102088

def effective_upstream_speed (b c : ℝ) : ℝ := b - c
def effective_downstream_speed (b c : ℝ) : ℝ := b + c

theorem boat_speed_still_water :
  ∃ b c : ℝ, effective_upstream_speed b c = 9 ∧ effective_downstream_speed b c = 15 ∧ b = 12 :=
by {
  sorry
}

end boat_speed_still_water_l102_102088


namespace company_production_n_l102_102259

theorem company_production_n (n : ℕ) (P : ℕ) 
  (h1 : P = n * 50) 
  (h2 : (P + 90) / (n + 1) = 58) : n = 4 := by 
  sorry

end company_production_n_l102_102259


namespace max_value_of_M_l102_102786

def J (k : ℕ) := 10^(k + 3) + 256

def M (k : ℕ) := Nat.factors (J k) |>.count 2

theorem max_value_of_M (k : ℕ) (hk : k > 0) :
  M k = 8 := by
  sorry

end max_value_of_M_l102_102786


namespace P_at_3_l102_102870

noncomputable def P (x : ℝ) : ℝ := 1 * x^5 + 0 * x^4 + 0 * x^3 + 2 * x^2 + 1 * x + 4

theorem P_at_3 : P 3 = 268 := by
  sorry

end P_at_3_l102_102870


namespace find_W_l102_102069

noncomputable def volumeOutsideCylinder (r_cylinder r_sphere : ℝ) : ℝ :=
  let h := 2 * Real.sqrt (r_sphere^2 - r_cylinder^2)
  let V_sphere := (4 / 3) * Real.pi * r_sphere^3
  let V_cylinder := Real.pi * r_cylinder^2 * h
  V_sphere - V_cylinder

theorem find_W : 
  volumeOutsideCylinder 4 7 = (1372 / 3 - 32 * Real.sqrt 33) * Real.pi :=
by
  sorry

end find_W_l102_102069


namespace find_c_value_l102_102510

theorem find_c_value :
  ∃ c : ℝ, (∀ x y : ℝ, (x + 10) ^ 2 + (y + 4) ^ 2 = 169 ∧ (x - 3) ^ 2 + (y - 9) ^ 2 = 65 → x + y = c) ∧ c = 3 :=
sorry

end find_c_value_l102_102510


namespace probability_of_winning_plan1_is_2_over_5_probability_of_winning_plan2_is_11_over_36_choose_plan1_l102_102887

-- Definition of the total number of outcomes and outcomes where a player wins for Plan 1
def total_outcomes_plan1 := 15
def winning_outcomes_plan1 := 6
def probability_plan1 : ℚ := winning_outcomes_plan1 / total_outcomes_plan1

-- Definition of the total number of outcomes and outcomes where a player wins for Plan 2
def total_outcomes_plan2 := 36
def winning_outcomes_plan2 := 11
def probability_plan2 : ℚ := winning_outcomes_plan2 / total_outcomes_plan2

-- Statements to prove
theorem probability_of_winning_plan1_is_2_over_5 : probability_plan1 = 2 / 5 :=
by sorry

theorem probability_of_winning_plan2_is_11_over_36 : probability_plan2 = 11 / 36 :=
by sorry

theorem choose_plan1 : probability_plan1 > probability_plan2 :=
by sorry

end probability_of_winning_plan1_is_2_over_5_probability_of_winning_plan2_is_11_over_36_choose_plan1_l102_102887


namespace total_price_before_increase_l102_102439

-- Conditions
def original_price_candy_box (c_or: ℝ) := 10 = c_or * 1.25
def original_price_soda_can (s_or: ℝ) := 15 = s_or * 1.50

-- Goal
theorem total_price_before_increase :
  ∃ (c_or s_or : ℝ), original_price_candy_box c_or ∧ original_price_soda_can s_or ∧ c_or + s_or = 25 :=
by
  sorry

end total_price_before_increase_l102_102439


namespace triangle_angle_B_l102_102169

theorem triangle_angle_B (a b A B : ℝ) (h1 : a * Real.cos B = 3 * b * Real.cos A) (h2 : B = A - Real.pi / 6) : 
  B = Real.pi / 6 := by
  sorry

end triangle_angle_B_l102_102169


namespace pipe_cistern_problem_l102_102583

theorem pipe_cistern_problem:
  ∀ (rate_p rate_q : ℝ),
    rate_p = 1 / 10 →
    rate_q = 1 / 15 →
    ∀ (filled_in_4_minutes : ℝ),
      filled_in_4_minutes = 4 * (rate_p + rate_q) →
      ∀ (remaining : ℝ),
        remaining = 1 - filled_in_4_minutes →
        ∀ (time_to_fill : ℝ),
          time_to_fill = remaining / rate_q →
          time_to_fill = 5 :=
by
  intros rate_p rate_q Hp Hq filled_in_4_minutes H4 remaining Hr time_to_fill Ht
  sorry

end pipe_cistern_problem_l102_102583


namespace parallel_transitivity_l102_102443

variable (Line Plane : Type)
variable (m n : Line)
variable (α : Plane)

-- Definitions for parallelism
variable (parallel : Line → Line → Prop)
variable (parallelLinePlane : Line → Plane → Prop)

-- Conditions
variable (m_n_parallel : parallel m n)
variable (m_alpha_parallel : parallelLinePlane m α)
variable (n_outside_alpha : ¬ parallelLinePlane n α)

-- Proposition to be proved
theorem parallel_transitivity (m n : Line) (α : Plane) 
  (h1 : parallel m n) 
  (h2 : parallelLinePlane m α) 
  : parallelLinePlane n α :=
sorry 

end parallel_transitivity_l102_102443


namespace minimum_transportation_cost_l102_102203

theorem minimum_transportation_cost :
  ∀ (x : ℕ), 
    (17 - x) + (x - 3) = 12 → 
    (18 - x) + (17 - x) = 14 → 
    (200 * x + 19300 = 19900) → 
    (x = 3) 
:= by sorry

end minimum_transportation_cost_l102_102203


namespace amount_of_bill_correct_l102_102019

noncomputable def TD : ℝ := 360
noncomputable def BD : ℝ := 421.7142857142857
noncomputable def computeFV (TD BD : ℝ) := (TD * BD) / (BD - TD)

theorem amount_of_bill_correct :
  computeFV TD BD = 2460 := 
sorry

end amount_of_bill_correct_l102_102019


namespace beaver_hid_36_carrots_l102_102539

variable (x y : ℕ)

-- Conditions
def beaverCarrots := 4 * x
def bunnyCarrots := 6 * y

-- Given that both animals hid the same total number of carrots
def totalCarrotsEqual := beaverCarrots x = bunnyCarrots y

-- Bunny used 3 fewer burrows than the beaver
def bunnyBurrows := y = x - 3

-- The goal is to show the beaver hid 36 carrots
theorem beaver_hid_36_carrots (H1 : totalCarrotsEqual x y) (H2 : bunnyBurrows x y) : beaverCarrots x = 36 := by
  sorry

end beaver_hid_36_carrots_l102_102539


namespace exists_sum_pair_l102_102189

theorem exists_sum_pair (n : ℕ) (a b : List ℕ) (h₁ : ∀ x ∈ a, x < n) (h₂ : ∀ y ∈ b, y < n) 
  (h₃ : List.Nodup a) (h₄ : List.Nodup b) (h₅ : a.length + b.length ≥ n) : ∃ x ∈ a, ∃ y ∈ b, x + y = n := by
  sorry

end exists_sum_pair_l102_102189


namespace area_of_square_field_l102_102149

-- Definitions
def cost_per_meter : ℝ := 1.40
def total_cost : ℝ := 932.40
def gate_width : ℝ := 1.0

-- Problem Statement
theorem area_of_square_field (s : ℝ) (A : ℝ) 
  (h1 : (4 * s - 2 * gate_width) * cost_per_meter = total_cost)
  (h2 : A = s^2) : A = 27889 := 
  sorry

end area_of_square_field_l102_102149


namespace electronic_items_stock_l102_102600

-- Define the base statements
def all_in_stock (S : Type) (p : S → Prop) : Prop := ∀ x, p x
def some_not_in_stock (S : Type) (p : S → Prop) : Prop := ∃ x, ¬ p x

-- Define the main theorem statement
theorem electronic_items_stock (S : Type) (p : S → Prop) :
  ¬ all_in_stock S p → some_not_in_stock S p :=
by
  intros
  sorry

end electronic_items_stock_l102_102600


namespace complex_number_corresponding_to_OB_l102_102821

theorem complex_number_corresponding_to_OB :
  let OA : ℂ := 6 + 5 * Complex.I
  let AB : ℂ := 4 + 5 * Complex.I
  OB = OA + AB -> OB = 10 + 10 * Complex.I := by
  sorry

end complex_number_corresponding_to_OB_l102_102821


namespace range_a_empty_intersection_range_a_sufficient_condition_l102_102788

noncomputable def A (x : ℝ) : Prop := -10 < x ∧ x < 2
noncomputable def B (x : ℝ) (a : ℝ) : Prop := x ≥ 1 + a ∨ x ≤ 1 - a
noncomputable def A_inter_B_empty (a : ℝ) : Prop := ∀ x : ℝ, A x → ¬ B x a
noncomputable def neg_p (x : ℝ) : Prop := x ≥ 2 ∨ x ≤ -10
noncomputable def neg_p_implies_q (a : ℝ) : Prop := ∀ x : ℝ, neg_p x → B x a

theorem range_a_empty_intersection : (∀ x : ℝ, A x → ¬ B x 11) → 11 ≤ a := by
  sorry

theorem range_a_sufficient_condition : (∀ x : ℝ, neg_p x → B x 1) → 0 < a ∧ a ≤ 1 := by
  sorry

end range_a_empty_intersection_range_a_sufficient_condition_l102_102788


namespace minimum_seedlings_needed_l102_102621

theorem minimum_seedlings_needed (n : ℕ) (h1 : 75 ≤ n) (h2 : n ≤ 80) (H : 1200 * 100 / n = 1500) : n = 80 :=
sorry

end minimum_seedlings_needed_l102_102621


namespace identical_answers_l102_102881
-- Import necessary libraries

-- Define the entities and conditions
structure Person :=
  (name : String)
  (always_tells_truth : Bool)

def Fyodor : Person := { name := "Fyodor", always_tells_truth := true }
def Sasha : Person := { name := "Sasha", always_tells_truth := false }

def answer (p : Person) : String :=
  if p.always_tells_truth then "Yes" else "No"

-- The theorem statement
theorem identical_answers :
  answer Fyodor = answer Sasha :=
by
  -- Proof steps will be filled in later
  sorry

end identical_answers_l102_102881


namespace pos_int_solutions_l102_102229

-- defining the condition for a positive integer solution to the equation
def is_pos_int_solution (x y : Int) : Prop :=
  5 * x + 2 * y = 25 ∧ x > 0 ∧ y > 0

-- stating the theorem for positive integer solutions of the equation
theorem pos_int_solutions : 
  ∃ x y : Int, is_pos_int_solution x y ∧ ((x = 1 ∧ y = 10) ∨ (x = 3 ∧ y = 5)) :=
by
  sorry

end pos_int_solutions_l102_102229


namespace f_six_equals_twenty_two_l102_102073

-- Definitions as per conditions
variable (n : ℕ) (f : ℕ → ℕ)

-- Conditions of the problem
-- n is a natural number greater than or equal to 3
-- f(n) satisfies the properties defined in the given solution
axiom f_base : f 1 = 2
axiom f_recursion {k : ℕ} (hk : k ≥ 1) : f (k + 1) = f k + (k + 1)

-- Goal to prove
theorem f_six_equals_twenty_two : f 6 = 22 := sorry

end f_six_equals_twenty_two_l102_102073


namespace sum_series_equals_4_div_9_l102_102702

noncomputable def sum_series : ℝ :=
  ∑' k : ℕ, (k + 1 : ℝ) / 4^(k + 1)

theorem sum_series_equals_4_div_9 : sum_series = 4 / 9 := by
  sorry

end sum_series_equals_4_div_9_l102_102702


namespace geometric_sequence_sum_l102_102494

variable {a : ℕ → ℕ}

def is_geometric_sequence_with_common_product (k : ℕ) (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a n * a (n + 1) * a (n + 2) = k

theorem geometric_sequence_sum :
  is_geometric_sequence_with_common_product 27 a →
  a 1 = 1 →
  a 2 = 3 →
  (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 +
   a 11 + a 12 + a 13 + a 14 + a 15 + a 16 + a 17 + a 18) = 78 :=
by
  intros h_geom h_a1 h_a2
  sorry

end geometric_sequence_sum_l102_102494


namespace second_sum_is_1704_l102_102845

theorem second_sum_is_1704
    (total_sum : ℝ)
    (x : ℝ)
    (interest_rate_first_part : ℝ)
    (time_first_part : ℝ)
    (interest_rate_second_part : ℝ)
    (time_second_part : ℝ)
    (h1 : total_sum = 2769)
    (h2 : interest_rate_first_part = 3)
    (h3 : time_first_part = 8)
    (h4 : interest_rate_second_part = 5)
    (h5 : time_second_part = 3)
    (h6 : 24 * x / 100 = (total_sum - x) * 15 / 100) :
    total_sum - x = 1704 :=
  by
    sorry

end second_sum_is_1704_l102_102845


namespace jamie_minimum_4th_quarter_score_l102_102630

-- Define the conditions for Jamie's scores and the average requirement
def qualifying_score := 85
def first_quarter_score := 80
def second_quarter_score := 85
def third_quarter_score := 78

-- The function to determine the required score in the 4th quarter
def minimum_score_for_quarter (N : ℕ) := first_quarter_score + second_quarter_score + third_quarter_score + N ≥ 4 * qualifying_score

-- The main statement to be proved
theorem jamie_minimum_4th_quarter_score (N : ℕ) : minimum_score_for_quarter N ↔ N ≥ 97 :=
by
  sorry

end jamie_minimum_4th_quarter_score_l102_102630


namespace second_account_interest_rate_l102_102339

theorem second_account_interest_rate
  (investment1 : ℝ)
  (rate1 : ℝ)
  (interest1 : ℝ)
  (investment2 : ℝ)
  (interest2 : ℝ)
  (h1 : 4000 = investment1)
  (h2 : 0.08 = rate1)
  (h3 : 320 = interest1)
  (h4 : 7200 - 4000 = investment2)
  (h5 : interest1 = interest2) :
  interest2 / investment2 = 0.1 :=
by
  sorry

end second_account_interest_rate_l102_102339


namespace average_investment_per_km_in_scientific_notation_l102_102622

-- Definitions based on the conditions of the problem
def total_investment : ℝ := 29.6 * 10^9
def upgraded_distance : ℝ := 6000

-- A theorem to be proven
theorem average_investment_per_km_in_scientific_notation :
  (total_investment / upgraded_distance) = 4.9 * 10^6 :=
by
  sorry

end average_investment_per_km_in_scientific_notation_l102_102622


namespace inequality_implies_l102_102851

theorem inequality_implies:
  ∀ (x y : ℝ), (x > y) → (2 * x - 1 > 2 * y - 1) :=
by
  intro x y hxy
  sorry

end inequality_implies_l102_102851


namespace gcd_of_repeated_three_digit_numbers_l102_102561

theorem gcd_of_repeated_three_digit_numbers :
  ∀ n : ℕ, 100 ≤ n ∧ n ≤ 999 → Int.gcd 1001001 n = 1001001 :=
by
  -- proof omitted
  sorry

end gcd_of_repeated_three_digit_numbers_l102_102561


namespace yvon_combination_l102_102361

theorem yvon_combination :
  let num_notebooks := 4
  let num_pens := 5
  num_notebooks * num_pens = 20 :=
by
  sorry

end yvon_combination_l102_102361


namespace find_a_l102_102138

variable (a : ℝ)

def p (a : ℝ) : Set ℝ := {x | a-1 < x ∧ x < a+1}
def q : Set ℝ := {x | x^2 - 4*x + 3 ≥ 0}
def q_negation : Set ℝ := {x | 1 < x ∧ x < 3}

theorem find_a :
  (∀ x, q_negation x → p a x) → a = 2 := by
  sorry

end find_a_l102_102138


namespace sunny_weather_prob_correct_l102_102902

def rain_prob : ℝ := 0.45
def cloudy_prob : ℝ := 0.20
def sunny_prob : ℝ := 1 - rain_prob - cloudy_prob

theorem sunny_weather_prob_correct : sunny_prob = 0.35 := by
  sorry

end sunny_weather_prob_correct_l102_102902


namespace tangent_and_parallel_l102_102319

noncomputable def parabola1 (x : ℝ) (b1 c1 : ℝ) : ℝ := -x^2 + b1 * x + c1
noncomputable def parabola2 (x : ℝ) (b2 c2 : ℝ) : ℝ := -x^2 + b2 * x + c2
noncomputable def parabola3 (x : ℝ) (b3 c3 : ℝ) : ℝ := x^2 + b3 * x + c3

theorem tangent_and_parallel (b1 b2 b3 c1 c2 c3 : ℝ) :
  (b3 - b1)^2 = 8 * (c3 - c1) → (b3 - b2)^2 = 8 * (c3 - c2) →
  ((b2^2 - b1^2 + 2 * b3 * (b2 - b1)) / (4 * (b2 - b1))) = 
  ((4 * (c1 - c2) - 2 * b3 * (b1 - b2)) / (2 * (b2 - b1))) :=
by
  intros h1 h2
  sorry

end tangent_and_parallel_l102_102319


namespace slope_of_CD_l102_102281

-- Define the two circle equations
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6 * x + 4 * y - 20 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 10 * x - 2 * y + 40 = 0

-- Theorem statement
theorem slope_of_CD :
  ∃ C D : ℝ × ℝ,
    (circle1 C.1 C.2) ∧ (circle2 C.1 C.2) ∧ (circle1 D.1 D.2) ∧ (circle2 D.1 D.2) ∧
    (∃ m : ℝ, m = -2 / 3) := 
  sorry

end slope_of_CD_l102_102281


namespace cycling_problem_l102_102980

theorem cycling_problem (x : ℚ) (h1 : 25 * x + 15 * (7 - x) = 140) : x = 7 / 2 := 
sorry

end cycling_problem_l102_102980


namespace socks_probability_l102_102923

theorem socks_probability :
  let total_socks := 18
  let total_pairs := (total_socks.choose 2)
  let gray_socks := 12
  let white_socks := 6
  let gray_pairs := (gray_socks.choose 2)
  let white_pairs := (white_socks.choose 2)
  let same_color_pairs := gray_pairs + white_pairs
  same_color_pairs / total_pairs = (81 / 153) :=
by
  sorry

end socks_probability_l102_102923


namespace sum_of_xy_eq_20_l102_102638

theorem sum_of_xy_eq_20 (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hx_lt : x < 30) (hy_lt : y < 30)
    (hxy : x + y + x * y = 119) : x + y = 20 :=
sorry

end sum_of_xy_eq_20_l102_102638


namespace seats_still_available_l102_102232

theorem seats_still_available (total_seats : ℕ) (two_fifths_seats : ℕ) (one_tenth_seats : ℕ) 
  (h1 : total_seats = 500) 
  (h2 : two_fifths_seats = (2 * total_seats) / 5) 
  (h3 : one_tenth_seats = total_seats / 10) :
  total_seats - (two_fifths_seats + one_tenth_seats) = 250 :=
by 
  sorry

end seats_still_available_l102_102232


namespace largest_composite_sequence_l102_102173

theorem largest_composite_sequence (a b c d e f g : ℕ) (h₁ : a < b) (h₂ : b < c) (h₃ : c < d) (h₄ : d < e) (h₅ : e < f) (h₆ : f < g) 
  (h₇ : g < 50) (h₈ : a ≥ 10) (h₉ : g ≤ 32)
  (h₁₀ : ¬ Prime a) (h₁₁ : ¬ Prime b) (h₁₂ : ¬ Prime c) (h₁₃ : ¬ Prime d) 
  (h₁₄ : ¬ Prime e) (h₁₅ : ¬ Prime f) (h₁₆ : ¬ Prime g) :
  g = 32 :=
sorry

end largest_composite_sequence_l102_102173


namespace remainder_of_1999_pow_81_mod_7_eq_1_l102_102614

/-- 
  Prove the remainder R when 1999^81 is divided by 7 is equal to 1.
  Conditions:
  - number: 1999
  - divisor: 7
-/
theorem remainder_of_1999_pow_81_mod_7_eq_1 : (1999 ^ 81) % 7 = 1 := 
by 
  sorry

end remainder_of_1999_pow_81_mod_7_eq_1_l102_102614


namespace negation_example_l102_102083

variable {I : Set ℝ}

theorem negation_example (h : ∀ x ∈ I, x^3 - x^2 + 1 ≤ 0) : ¬(∀ x ∈ I, x^3 - x^2 + 1 ≤ 0) ↔ ∃ x ∈ I, x^3 - x^2 + 1 > 0 :=
by
  sorry

end negation_example_l102_102083


namespace fuel_at_40_min_fuel_l102_102235

section FuelConsumption

noncomputable def fuel_consumption (x : ℝ) : ℝ := (1 / 128000) * x^3 - (3 / 80) * x + 8

noncomputable def total_fuel (x : ℝ) : ℝ := (fuel_consumption x) * (100 / x)

theorem fuel_at_40 : total_fuel 40 = 17.5 :=
by sorry

theorem min_fuel : total_fuel 80 = 11.25 ∧ ∀ x, (0 < x ∧ x ≤ 120) → total_fuel x ≥ total_fuel 80 :=
by sorry

end FuelConsumption

end fuel_at_40_min_fuel_l102_102235


namespace total_volume_calculation_l102_102962

noncomputable def total_volume_of_four_cubes (edge_length_in_feet : ℝ) (conversion_factor : ℝ) : ℝ :=
  let edge_length_in_meters := edge_length_in_feet * conversion_factor
  let volume_of_one_cube := edge_length_in_meters^3
  4 * volume_of_one_cube

theorem total_volume_calculation :
  total_volume_of_four_cubes 5 0.3048 = 14.144 :=
by
  -- Proof needs to be filled in.
  sorry

end total_volume_calculation_l102_102962


namespace perpendicular_lines_condition_l102_102994

theorem perpendicular_lines_condition (m : ℝ) :
  (∀ x y : ℝ, 2 * x - y - 1 = 0 → m * x + y + 1 = 0 → False) ↔ m = 1 / 2 :=
by sorry

end perpendicular_lines_condition_l102_102994


namespace percentage_problem_l102_102380

theorem percentage_problem 
    (y : ℝ)
    (h₁ : 0.47 * 1442 = 677.74)
    (h₂ : (677.74 - (y / 100) * 1412) + 63 = 3) :
    y = 52.25 :=
by sorry

end percentage_problem_l102_102380


namespace rectangular_prism_dimensions_l102_102963

theorem rectangular_prism_dimensions (b l h : ℕ) 
  (h1 : l = 3 * b) 
  (h2 : l = 2 * h) 
  (h3 : l * b * h = 12168) :
  b = 14 ∧ l = 42 ∧ h = 21 :=
by
  -- The proof will go here
  sorry

end rectangular_prism_dimensions_l102_102963


namespace find_breadth_l102_102664

-- Define variables and constants
variables (SA l h w : ℝ)

-- Given conditions
axiom h1 : SA = 2400
axiom h2 : l = 15
axiom h3 : h = 16

-- Define the surface area equation for a cuboid 
def surface_area := 2 * (l * w + l * h + w * h)

-- Statement to prove
theorem find_breadth : surface_area l w h = SA → w = 30.97 := sorry

end find_breadth_l102_102664


namespace courtyard_width_l102_102907

theorem courtyard_width 
  (L : ℝ) (N : ℕ) (brick_length brick_width : ℝ) (courtyard_area : ℝ)
  (hL : L = 18)
  (hN : N = 30000)
  (hbrick_length : brick_length = 0.12)
  (hbrick_width : brick_width = 0.06)
  (hcourtyard_area : courtyard_area = (N : ℝ) * (brick_length * brick_width)) :
  (courtyard_area / L) = 12 :=
by
  sorry

end courtyard_width_l102_102907


namespace parabola_focus_l102_102096

theorem parabola_focus (a : ℝ) (h : a = 4) : 
  ∃ f : ℝ × ℝ, f = (0, 1 / (4 * a)) ∧ y = 4 * x^2 → f = (0, 1 / 16) := 
by {
  sorry
}

end parabola_focus_l102_102096


namespace primes_product_less_than_20_l102_102545

-- Define the primes less than 20
def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

-- Define the product of a list of natural numbers
def product (l : List ℕ) : ℕ := l.foldr (· * ·) 1

theorem primes_product_less_than_20 :
  product primes_less_than_20 = 9699690 :=
by
  sorry

end primes_product_less_than_20_l102_102545


namespace num_5_digit_numbers_is_six_l102_102267

-- Define that we have the digits 2, 45, and 68
def digits : List Nat := [2, 45, 68]

-- Function to generate all permutations of given digits
def permute : List Nat → List (List Nat)
| [] => [[]]
| (x::xs) =>
  List.join (List.map (λ ys =>
    List.map (λ zs => x :: zs) (permute xs)) (permute xs))

-- Calculate the number of distinct 5-digit numbers
def numberOf5DigitNumbers : Int := 
  (permute digits).length

-- Theorem to prove the number of distinct 5-digit numbers formed
theorem num_5_digit_numbers_is_six : numberOf5DigitNumbers = 6 := by
  sorry

end num_5_digit_numbers_is_six_l102_102267


namespace angle_triple_complement_l102_102553

theorem angle_triple_complement (x : ℝ) (h1 : x + (180 - x) = 180) (h2 : x = 3 * (180 - x)) : x = 135 := 
by
  sorry

end angle_triple_complement_l102_102553


namespace sufficient_condition_for_solution_l102_102065

theorem sufficient_condition_for_solution 
  (a : ℝ) (f g h : ℝ → ℝ) (h_a : 1 < a)
  (h_fg_h : ∀ x : ℝ, 0 ≤ f x + g x + h x) 
  (h_common_root : ∃ x : ℝ, f x = 0 ∧ g x = 0 ∧ h x = 0) : 
  ∃ x : ℝ, a^(f x) + a^(g x) + a^(h x) = 3 := 
by
  sorry

end sufficient_condition_for_solution_l102_102065


namespace root_in_interval_l102_102551

noncomputable def f (x : ℝ) : ℝ := x^3 + 2 * x - 1

theorem root_in_interval (k : ℤ) (h : ∃ x : ℝ, k < x ∧ x < k + 1 ∧ f x = 0) : k = 0 :=
by
  sorry

end root_in_interval_l102_102551


namespace quarts_of_water_needed_l102_102665

-- Definitions of conditions
def total_parts := 5 + 2 + 1
def total_gallons := 3
def quarts_per_gallon := 4
def water_parts := 5

-- Lean proof statement
theorem quarts_of_water_needed :
  (water_parts : ℚ) * ((total_gallons * quarts_per_gallon) / total_parts) = 15 / 2 :=
by sorry

end quarts_of_water_needed_l102_102665


namespace billy_reads_60_pages_per_hour_l102_102396

theorem billy_reads_60_pages_per_hour
  (free_time_per_day : ℕ)
  (days : ℕ)
  (video_games_time_percentage : ℝ)
  (books : ℕ)
  (pages_per_book : ℕ)
  (remaining_time_percentage : ℝ)
  (total_free_time := free_time_per_day * days)
  (time_playing_video_games := video_games_time_percentage * total_free_time)
  (time_reading := remaining_time_percentage * total_free_time)
  (total_pages := books * pages_per_book)
  (pages_per_hour := total_pages / time_reading) :
  free_time_per_day = 8 →
  days = 2 →
  video_games_time_percentage = 0.75 →
  remaining_time_percentage = 0.25 →
  books = 3 →
  pages_per_book = 80 →
  pages_per_hour = 60 :=
by
  intros
  sorry

end billy_reads_60_pages_per_hour_l102_102396


namespace opposite_of_neg_2_is_2_l102_102570

theorem opposite_of_neg_2_is_2 : -(-2) = 2 := 
by
  sorry

end opposite_of_neg_2_is_2_l102_102570


namespace domain_of_f_l102_102970

def condition1 (x : ℝ) : Prop := 4 - |x| ≥ 0
def condition2 (x : ℝ) : Prop := (x^2 - 5 * x + 6) / (x - 3) > 0

theorem domain_of_f (x : ℝ) :
  (condition1 x) ∧ (condition2 x) ↔ ((2 < x ∧ x < 3) ∨ (3 < x ∧ x ≤ 4)) :=
by
  sorry

end domain_of_f_l102_102970


namespace maximize_profit_l102_102515

noncomputable def R (x : ℝ) : ℝ := 
  if x ≤ 40 then
    40 * x - (1 / 2) * x^2
  else
    1500 - 25000 / x

noncomputable def cost (x : ℝ) : ℝ := 2 + 0.1 * x

noncomputable def f (x : ℝ) : ℝ := R x - cost x

theorem maximize_profit :
  ∃ x : ℝ, x = 50 ∧ f 50 = 300 := by
  sorry

end maximize_profit_l102_102515


namespace circumference_of_circle_of_given_area_l102_102157

theorem circumference_of_circle_of_given_area (A : ℝ) (h : A = 225 * Real.pi) : 
  ∃ C : ℝ, C = 2 * Real.pi * 15 :=
by
  let r := 15
  let C := 2 * Real.pi * r
  use C
  sorry

end circumference_of_circle_of_given_area_l102_102157


namespace power_function_point_l102_102689

theorem power_function_point (a : ℝ) (h : (2 : ℝ) ^ a = (1 / 2 : ℝ)) : a = -1 :=
by sorry

end power_function_point_l102_102689


namespace basketball_team_win_requirement_l102_102160

theorem basketball_team_win_requirement :
  ∀ (games_won_first_60 : ℕ) (total_games : ℕ) (win_percentage : ℚ) (remaining_games : ℕ),
    games_won_first_60 = 45 →
    total_games = 110 →
    win_percentage = 0.75 →
    remaining_games = 50 →
    ∃ games_won_remaining, games_won_remaining = 38 ∧
    (games_won_first_60 + games_won_remaining) / total_games = win_percentage :=
by
  intros
  sorry

end basketball_team_win_requirement_l102_102160


namespace sally_pens_proof_l102_102207

variable (p : ℕ)  -- define p as a natural number for pens each student received
variable (pensLeft : ℕ)  -- define pensLeft as a natural number for pens left after distributing to students

-- Function representing Sally giving pens to each student
def pens_after_giving_students (p : ℕ) : ℕ := 342 - 44 * p

-- Condition 1: Left half of the remainder in her locker
def locker_pens (p : ℕ) : ℕ := (pens_after_giving_students p) / 2

-- Condition 2: She took 17 pens home
def home_pens : ℕ := 17

-- Main proof statement
theorem sally_pens_proof :
  (locker_pens p + home_pens = pens_after_giving_students p) → p = 7 :=
by
  sorry

end sally_pens_proof_l102_102207


namespace coach_mike_change_l102_102364

theorem coach_mike_change (cost amount_given change : ℕ) 
    (h_cost : cost = 58) (h_amount_given : amount_given = 75) : 
    change = amount_given - cost → change = 17 := by
    sorry

end coach_mike_change_l102_102364


namespace ultramen_defeat_monster_in_5_minutes_l102_102983

theorem ultramen_defeat_monster_in_5_minutes :
  ∀ (attacksRequired : ℕ) (attackRate1 attackRate2 : ℕ),
    (attacksRequired = 100) →
    (attackRate1 = 12) →
    (attackRate2 = 8) →
    (attacksRequired / (attackRate1 + attackRate2) = 5) :=
by
  intros
  sorry

end ultramen_defeat_monster_in_5_minutes_l102_102983


namespace alyssa_turnips_l102_102280

theorem alyssa_turnips (k a t: ℕ) (h1: k = 6) (h2: t = 15) (h3: t = k + a) : a = 9 := 
by
  -- proof goes here
  sorry

end alyssa_turnips_l102_102280


namespace scientific_notation_9600000_l102_102119

theorem scientific_notation_9600000 :
  ∃ (a : ℝ) (n : ℤ), (1 ≤ |a| ∧ |a| < 10) ∧ 9600000 = a * 10 ^ n ∧ a = 9.6 ∧ n = 6 :=
by
  exists 9.6
  exists 6
  simp
  sorry

end scientific_notation_9600000_l102_102119


namespace geometric_progression_quadrilateral_exists_l102_102148

theorem geometric_progression_quadrilateral_exists :
  ∃ (a1 r : ℝ), a1 > 0 ∧ r > 0 ∧ 
  (1 + r + r^2 > r^3) ∧
  (1 + r + r^3 > r^2) ∧
  (1 + r^2 + r^3 > r) ∧
  (r + r^2 + r^3 > 1) := 
sorry

end geometric_progression_quadrilateral_exists_l102_102148


namespace range_of_a_l102_102772

noncomputable def f (x : ℝ) : ℝ := x + 1 / Real.exp x

theorem range_of_a (a : ℝ) : (∀ x : ℝ, f x > a * x) ↔ (1 - Real.exp 1) < a ∧ a ≤ 1 := 
by
  sorry

end range_of_a_l102_102772


namespace min_unsuccessful_placements_8x8_l102_102362

-- Define the board, the placement, and the unsuccessful condition
def is_unsuccessful_placement (board : ℕ → ℕ → ℤ) (i j : ℕ) : Prop :=
  (i < 7 ∧ j < 7 ∧ (board i j + board (i+1) j + board i (j+1) + board (i+1) (j+1)) ≠ 0)

-- Main theorem: The minimum number of unsuccessful placements is 36 on an 8x8 board
theorem min_unsuccessful_placements_8x8 (board : ℕ → ℕ → ℤ) (H : ∀ i j, board i j = 1 ∨ board i j = -1) :
  ∃ (n : ℕ), n = 36 ∧ (∀ m : ℕ, (∀ i j, is_unsuccessful_placement board i j → m < 36 ) → m = n) :=
sorry

end min_unsuccessful_placements_8x8_l102_102362


namespace tan_neg_405_eq_neg_1_l102_102206

theorem tan_neg_405_eq_neg_1 :
  Real.tan (Real.pi * -405 / 180) = -1 := 
sorry

end tan_neg_405_eq_neg_1_l102_102206


namespace cannot_determine_congruency_l102_102754

-- Define the congruency criteria for triangles
def SSS (a1 b1 c1 a2 b2 c2 : ℝ) : Prop := a1 = a2 ∧ b1 = b2 ∧ c1 = c2
def SAS (a1 b1 angle1 a2 b2 angle2 : ℝ) : Prop := a1 = a2 ∧ b1 = b2 ∧ angle1 = angle2
def ASA (angle1 b1 angle2 angle3 b2 angle4 : ℝ) : Prop := angle1 = angle2 ∧ b1 = b2 ∧ angle3 = angle4
def AAS (angle1 angle2 b1 angle3 angle4 b2 : ℝ) : Prop := angle1 = angle2 ∧ angle3 = angle4 ∧ b1 = b2
def HL (hyp1 leg1 hyp2 leg2 : ℝ) : Prop := hyp1 = hyp2 ∧ leg1 = leg2

-- Define the condition D, which states the equality of two corresponding sides and a non-included angle
def conditionD (a1 b1 angle1 a2 b2 angle2 : ℝ) : Prop := a1 = a2 ∧ b1 = b2 ∧ angle1 = angle2

-- The theorem to be proven
theorem cannot_determine_congruency (a1 b1 angle1 a2 b2 angle2 : ℝ) :
  conditionD a1 b1 angle1 a2 b2 angle2 → ¬(SSS a1 b1 0 a2 b2 0 ∨ SAS a1 b1 0 a2 b2 0 ∨ ASA 0 b1 0 0 b2 0 ∨ AAS 0 0 b1 0 0 b2 ∨ HL 0 0 0 0) :=
by
  sorry

end cannot_determine_congruency_l102_102754


namespace no_natural_n_divisible_by_2019_l102_102555

theorem no_natural_n_divisible_by_2019 :
  ∀ n : ℕ, ¬ 2019 ∣ (n^2 + n + 2) :=
by sorry

end no_natural_n_divisible_by_2019_l102_102555


namespace circle_area_in_sq_cm_l102_102301

theorem circle_area_in_sq_cm (diameter_meters : ℝ) (h : diameter_meters = 5) : 
  let radius_meters := diameter_meters / 2
  let area_square_meters := π * radius_meters^2
  let area_square_cm := area_square_meters * 10000
  area_square_cm = 62500 * π :=
by
  sorry

end circle_area_in_sq_cm_l102_102301


namespace total_cookies_l102_102011

theorem total_cookies (num_people : ℕ) (cookies_per_person : ℕ) (total_cookies : ℕ) 
  (h1: num_people = 4) (h2: cookies_per_person = 22) : total_cookies = 88 :=
by
  sorry

end total_cookies_l102_102011


namespace quadratic_rational_solutions_product_l102_102463

theorem quadratic_rational_solutions_product :
  ∃ (c₁ c₂ : ℕ), (7 * x^2 + 15 * x + c₁ = 0 ∧ 225 - 28 * c₁ = k^2 ∧ ∃ k : ℤ, k^2 = 225 - 28 * c₁) ∧
                 (7 * x^2 + 15 * x + c₂ = 0 ∧ 225 - 28 * c₂ = k^2 ∧ ∃ k : ℤ, k^2 = 225 - 28 * c₂) ∧
                 (c₁ = 1) ∧ (c₂ = 8) ∧ (c₁ * c₂ = 8) :=
by
  sorry

end quadratic_rational_solutions_product_l102_102463


namespace evaluate_expression_l102_102317

theorem evaluate_expression : 
  (900 * 900) / ((306 * 306) - (294 * 294)) = 112.5 := by
  sorry

end evaluate_expression_l102_102317


namespace convert_ternary_to_octal_2101211_l102_102236

def ternaryToOctal (n : List ℕ) : ℕ := 
  sorry

theorem convert_ternary_to_octal_2101211 :
  ternaryToOctal [2, 1, 0, 1, 2, 1, 1] = 444
  := sorry

end convert_ternary_to_octal_2101211_l102_102236


namespace uncle_welly_roses_l102_102048

theorem uncle_welly_roses :
  let roses_two_days_ago := 50
  let roses_yesterday := roses_two_days_ago + 20
  let roses_today := 2 * roses_two_days_ago
  roses_two_days_ago + roses_yesterday + roses_today = 220 :=
by
  let roses_two_days_ago := 50
  let roses_yesterday := roses_two_days_ago + 20
  let roses_today := 2 * roses_two_days_ago
  show roses_two_days_ago + roses_yesterday + roses_today = 220
  sorry

end uncle_welly_roses_l102_102048


namespace mouse_jump_less_than_frog_l102_102161

-- Definitions for the given conditions
def grasshopper_jump : ℕ := 25
def frog_jump : ℕ := grasshopper_jump + 32
def mouse_jump : ℕ := 31

-- The statement we need to prove
theorem mouse_jump_less_than_frog :
  frog_jump - mouse_jump = 26 :=
by
  -- The proof will be filled in here
  sorry

end mouse_jump_less_than_frog_l102_102161


namespace monotonic_intervals_l102_102946

noncomputable def y : ℝ → ℝ := λ x => x * Real.log x

theorem monotonic_intervals :
  (∀ x : ℝ, 0 < x → x < (1 / Real.exp 1) → y x < -1) ∧ 
  (∀ x : ℝ, (1 / Real.exp 1) < x → x < 5 → y x > 1) := 
by
  sorry -- Proof goes here.

end monotonic_intervals_l102_102946


namespace simplify_fraction_l102_102607

noncomputable def a : ℝ := 2
noncomputable def b : ℝ := Real.sqrt 3

theorem simplify_fraction :
    (1 / (a + b)) * (1 / (a - b)) = 1 := by
  sorry

end simplify_fraction_l102_102607


namespace find_original_cost_price_l102_102224

variables (P : ℝ) (A B C D E : ℝ)

-- Define the conditions as per the problem statement
def with_tax (P : ℝ) : ℝ := P * 1.10
def profit_60 (price : ℝ) : ℝ := price * 1.60
def profit_25 (price : ℝ) : ℝ := price * 1.25
def loss_15 (price : ℝ) : ℝ := price * 0.85
def profit_30 (price : ℝ) : ℝ := price * 1.30

-- The final price E is given.
def final_price (P : ℝ) : ℝ :=
  profit_30 
  (loss_15 
  (profit_25 
  (profit_60 
  (with_tax P))))

-- To find original cost price P given final price of Rs. 500.
theorem find_original_cost_price (h : final_price P = 500) : 
  P = 500 / 2.431 :=
by 
  sorry

end find_original_cost_price_l102_102224


namespace car_passing_problem_l102_102275

noncomputable def maxCarsPerHourDividedBy10 : ℕ :=
  let unit_length (n : ℕ) := 5 * (n + 1)
  let cars_passed_in_one_hour (n : ℕ) := 10000 * n / unit_length n
  Nat.div (2000) (10)

theorem car_passing_problem : maxCarsPerHourDividedBy10 = 200 :=
  by
  sorry

end car_passing_problem_l102_102275


namespace find_cost_price_l102_102038

/-- Statement: Given Mohit sold an article for $18000 and 
if he offered a discount of 10% on the selling price, he would have earned a profit of 8%, 
prove that the cost price (CP) of the article is $15000. -/

def discounted_price (sp : ℝ) := sp - (0.10 * sp)
def profit_price (cp : ℝ) := cp * 1.08

theorem find_cost_price (sp : ℝ) (discount: sp = 18000) (profit_discount: profit_price (discounted_price sp) = discounted_price sp):
    ∃ (cp : ℝ), cp = 15000 :=
by
    sorry

end find_cost_price_l102_102038


namespace probability_of_spade_or_king_in_two_draws_l102_102285

def total_cards : ℕ := 52
def spades_count : ℕ := 13
def kings_count : ℕ := 4
def king_of_spades_count : ℕ := 1
def spades_or_kings_count : ℕ := spades_count + kings_count - king_of_spades_count
def probability_not_spade_or_king : ℚ := (total_cards - spades_or_kings_count) / total_cards
def probability_both_not_spade_or_king : ℚ := probability_not_spade_or_king^2
def probability_at_least_one_spade_or_king : ℚ := 1 - probability_both_not_spade_or_king

theorem probability_of_spade_or_king_in_two_draws :
  probability_at_least_one_spade_or_king = 88 / 169 :=
sorry

end probability_of_spade_or_king_in_two_draws_l102_102285


namespace marble_problem_l102_102738

theorem marble_problem
  (x : ℕ) (h1 : 144 / x = 144 / (x + 2) + 1) :
  x = 16 :=
sorry

end marble_problem_l102_102738


namespace isosceles_triangles_with_perimeter_21_l102_102564

theorem isosceles_triangles_with_perimeter_21 : 
  ∃ n : ℕ, n = 5 ∧ (∀ (a b c : ℕ), a ≤ b ∧ b = c ∧ a + 2*b = 21 → 1 ≤ a ∧ a ≤ 10) :=
sorry

end isosceles_triangles_with_perimeter_21_l102_102564


namespace zoo_problem_l102_102102

theorem zoo_problem :
  let parrots := 8
  let snakes := 3 * parrots
  let monkeys := 2 * snakes
  let elephants := (parrots + snakes) / 2
  let zebras := monkeys - 35
  elephants - zebras = 3 :=
by
  sorry

end zoo_problem_l102_102102


namespace line_equation_l102_102249

noncomputable def line_intersects_at_point (a1 a2 b1 b2 c1 c2 : ℝ) (p : ℝ × ℝ) : Prop :=
  p.1 * a1 + p.2 * b1 = c1 ∧ p.1 * a2 + p.2 * b2 = c2

noncomputable def point_on_line (a b c : ℝ) (p : ℝ × ℝ) : Prop :=
  a * p.1 + b * p.2 = c

theorem line_equation
  (p : ℝ × ℝ)
  (h1 : line_intersects_at_point 3 2 2 3 5 5 p)
  (h2 : point_on_line 0 1 (-5) p)
  : ∃ a b c : ℝ,  a * p.1 + b * p.2 + (-5) = 0 :=
sorry

end line_equation_l102_102249


namespace speed_ratio_l102_102968

theorem speed_ratio (v1 v2 : ℝ) (t1 t2 : ℝ) (dist_before dist_after : ℝ) (total_dist : ℝ)
  (h1 : dist_before + dist_after = total_dist)
  (h2 : dist_before = 20)
  (h3 : dist_after = 20)
  (h4 : t2 = t1 + 11)
  (h5 : t2 = 22)
  (h6 : t1 = dist_before / v1)
  (h7 : t2 = dist_after / v2) :
  v1 / v2 = 2 := 
sorry

end speed_ratio_l102_102968


namespace elizabeth_time_l102_102502

-- Defining the conditions
def tom_time_minutes : ℕ := 120
def time_ratio : ℕ := 4

-- Proving Elizabeth's time
theorem elizabeth_time : tom_time_minutes / time_ratio = 30 := 
by
  sorry

end elizabeth_time_l102_102502


namespace value_of_a_star_b_l102_102209

variable (a b : ℤ)

def operation_star (a b : ℤ) : ℚ :=
  1 / a + 1 / b

theorem value_of_a_star_b (h1 : a + b = 7) (h2 : a * b = 12) :
  operation_star a b = 7 / 12 := by
  sorry

end value_of_a_star_b_l102_102209


namespace range_of_squared_sum_l102_102588

theorem range_of_squared_sum (x y : ℝ) (h : x^2 + 1 / y^2 = 2) : ∃ z, z = x^2 + y^2 ∧ z ≥ 1 / 2 :=
by
  sorry

end range_of_squared_sum_l102_102588


namespace hex_B2F_to_base10_l102_102984

theorem hex_B2F_to_base10 :
  let b := 11
  let two := 2
  let f := 15
  let base := 16
  (b * base^2 + two * base^1 + f * base^0) = 2863 :=
by
  sorry

end hex_B2F_to_base10_l102_102984


namespace quarters_remaining_l102_102714

-- Define the number of quarters Sally originally had
def initialQuarters : Nat := 760

-- Define the number of quarters Sally spent
def spentQuarters : Nat := 418

-- Prove that the number of quarters she has now is 342
theorem quarters_remaining : initialQuarters - spentQuarters = 342 :=
by
  sorry

end quarters_remaining_l102_102714


namespace gasoline_tank_capacity_l102_102227

theorem gasoline_tank_capacity (x : ℝ)
  (h1 : (7 / 8) * x - (1 / 2) * x = 12) : x = 32 := 
sorry

end gasoline_tank_capacity_l102_102227


namespace john_total_payment_l102_102111

-- Definitions of the conditions
def yearly_cost_first_8_years : ℕ := 10000
def yearly_cost_9_to_18_years : ℕ := 2 * yearly_cost_first_8_years
def university_tuition : ℕ := 250000
def total_cost := (8 * yearly_cost_first_8_years) + (10 * yearly_cost_9_to_18_years) + university_tuition

-- John pays half of the total cost
def johns_total_cost := total_cost / 2

-- Theorem stating the total cost John pays
theorem john_total_payment : johns_total_cost = 265000 := by
  sorry

end john_total_payment_l102_102111


namespace amare_fabric_needed_l102_102212

-- Definitions for the conditions
def fabric_per_dress_yards : ℝ := 5.5
def number_of_dresses : ℕ := 4
def fabric_owned_feet : ℝ := 7
def yard_to_feet : ℝ := 3

-- Total fabric needed in yards
def total_fabric_needed_yards : ℝ := fabric_per_dress_yards * number_of_dresses

-- Total fabric needed in feet
def total_fabric_needed_feet : ℝ := total_fabric_needed_yards * yard_to_feet

-- Fabric still needed
def fabric_still_needed : ℝ := total_fabric_needed_feet - fabric_owned_feet

-- Proof
theorem amare_fabric_needed : fabric_still_needed = 59 := by
  sorry

end amare_fabric_needed_l102_102212


namespace hyperbola_range_m_l102_102567

theorem hyperbola_range_m (m : ℝ) : 
  (∃ x y : ℝ, (x^2 / (16 - m)) + (y^2 / (9 - m)) = 1) → 9 < m ∧ m < 16 :=
by 
  sorry

end hyperbola_range_m_l102_102567


namespace computer_additions_per_hour_l102_102310

def operations_per_second : ℕ := 15000
def additions_per_second : ℕ := operations_per_second / 2
def seconds_per_hour : ℕ := 3600

theorem computer_additions_per_hour : 
  additions_per_second * seconds_per_hour = 27000000 := by
  sorry

end computer_additions_per_hour_l102_102310


namespace least_value_a_l102_102959

theorem least_value_a (a : ℤ) :
  (∃ a : ℤ, a ≥ 0 ∧ (a ^ 6) % 1920 = 0) → a = 8 ∧ (a ^ 6) % 1920 = 0 :=
by
  sorry

end least_value_a_l102_102959


namespace participants_are_multiple_of_7_l102_102906

theorem participants_are_multiple_of_7 (P : ℕ) (h1 : P % 2 = 0)
  (h2 : ∀ p, p = P / 2 → P + p / 7 = (4 * P) / 7)
  (h3 : (4 * P) / 7 * 7 = 4 * P) : ∃ k : ℕ, P = 7 * k := 
by
  sorry

end participants_are_multiple_of_7_l102_102906


namespace number_of_raised_beds_l102_102427

def length_feed := 8
def width_feet := 4
def height_feet := 1
def cubic_feet_per_bag := 4
def total_bags_needed := 16

theorem number_of_raised_beds :
  ∀ (length_feed width_feet height_feet : ℕ) (cubic_feet_per_bag total_bags_needed : ℕ),
    (length_feed * width_feet * height_feet) / cubic_feet_per_bag = 8 →
    total_bags_needed / (8 : ℕ) = 2 :=
by sorry

end number_of_raised_beds_l102_102427


namespace average_death_rate_l102_102966

-- Definitions of the given conditions
def birth_rate_two_seconds := 10
def net_increase_one_day := 345600
def seconds_per_day := 24 * 60 * 60 

-- Define the theorem to be proven
theorem average_death_rate :
  (birth_rate_two_seconds / 2) - (net_increase_one_day / seconds_per_day) = 1 :=
by 
  sorry

end average_death_rate_l102_102966


namespace lemon_juice_fraction_l102_102988

theorem lemon_juice_fraction :
  ∃ L : ℚ, 30 - 30 * L - (1 / 3) * (30 - 30 * L) = 6 ∧ L = 7 / 10 :=
sorry

end lemon_juice_fraction_l102_102988


namespace total_amount_received_l102_102868

theorem total_amount_received
  (total_books : ℕ := 500)
  (novels_price : ℕ := 8)
  (biographies_price : ℕ := 12)
  (science_books_price : ℕ := 10)
  (novels_discount : ℚ := 0.25)
  (biographies_discount : ℚ := 0.30)
  (science_books_discount : ℚ := 0.20)
  (sales_tax : ℚ := 0.05)
  (remaining_novels : ℕ := 60)
  (remaining_biographies : ℕ := 65)
  (remaining_science_books : ℕ := 50)
  (novel_ratio_sold : ℚ := 3/5)
  (biography_ratio_sold : ℚ := 2/3)
  (science_book_ratio_sold : ℚ := 7/10)
  (original_novels : ℕ := 150)
  (original_biographies : ℕ := 195)
  (original_science_books : ℕ := 167) -- Rounded from 166.67
  (sold_novels : ℕ := 90)
  (sold_biographies : ℕ := 130)
  (sold_science_books : ℕ := 117)
  (total_revenue_before_discount : ℚ := (90 * 8 + 130 * 12 + 117 * 10))
  (total_revenue_after_discount : ℚ := (720 * (1 - 0.25) + 1560 * (1 - 0.30) + 1170 * (1 - 0.20)))
  (total_revenue_after_tax : ℚ := (2568 * 1.05)) :
  total_revenue_after_tax = 2696.4 :=
by
  sorry

end total_amount_received_l102_102868


namespace evaluate_expression_l102_102529

theorem evaluate_expression : (∃ (x : Real), 6 < x ∧ x < 7 ∧ x = Real.sqrt 45) → (Int.floor (Real.sqrt 45))^2 + 2*Int.floor (Real.sqrt 45) + 1 = 49 := 
by
  sorry

end evaluate_expression_l102_102529


namespace present_age_of_B_l102_102471

theorem present_age_of_B 
  (a b : ℕ)
  (h1 : a + 10 = 2 * (b - 10))
  (h2 : a = b + 9) :
  b = 39 :=
by
  sorry

end present_age_of_B_l102_102471


namespace rhombus_second_diagonal_l102_102480

theorem rhombus_second_diagonal (perimeter : ℝ) (d1 : ℝ) (side : ℝ) (half_d2 : ℝ) (d2 : ℝ) :
  perimeter = 52 → d1 = 24 → side = 13 → (half_d2 = 5) → d2 = 2 * half_d2 → d2 = 10 :=
by
  sorry

end rhombus_second_diagonal_l102_102480


namespace positive_integers_square_less_than_three_times_l102_102314

theorem positive_integers_square_less_than_three_times (n : ℕ) (hn : 0 < n) (ineq : n^2 < 3 * n) : n = 1 ∨ n = 2 :=
by sorry

end positive_integers_square_less_than_three_times_l102_102314


namespace supplement_of_angle_l102_102890

theorem supplement_of_angle (θ : ℝ) 
  (h_complement: θ = 90 - 30) : 180 - θ = 120 :=
by
  sorry

end supplement_of_angle_l102_102890


namespace tan_30_degrees_correct_l102_102032

noncomputable def tan_30_degrees : ℝ := Real.tan (Real.pi / 6)

theorem tan_30_degrees_correct : tan_30_degrees = Real.sqrt 3 / 3 :=
by
  sorry

end tan_30_degrees_correct_l102_102032


namespace remainder_12345678901_mod_101_l102_102613

theorem remainder_12345678901_mod_101 : 12345678901 % 101 = 24 :=
by
  sorry

end remainder_12345678901_mod_101_l102_102613


namespace length_to_width_ratio_l102_102068

-- Define the conditions: perimeter and length
variable (P : ℕ) (l : ℕ) (w : ℕ)

-- Given conditions
def conditions : Prop := (P = 100) ∧ (l = 40) ∧ (P = 2 * l + 2 * w)

-- The proposition we want to prove
def ratio : Prop := l / w = 4

-- The main theorem
theorem length_to_width_ratio (h : conditions P l w) : ratio l w :=
by sorry

end length_to_width_ratio_l102_102068


namespace sum_of_four_consecutive_integers_prime_factor_l102_102297

theorem sum_of_four_consecutive_integers_prime_factor (n : ℤ) : ∃ p : ℤ, Prime p ∧ p = 2 ∧ ∀ n : ℤ, p ∣ ((n - 1) + n + (n + 1) + (n + 2)) := 
by 
  sorry

end sum_of_four_consecutive_integers_prime_factor_l102_102297


namespace part_I_part_II_l102_102496

def sequence_sn (n : ℕ) : ℚ := (3 / 2 : ℚ) * n^2 + (1 / 2 : ℚ) * n

def sequence_a (n : ℕ) : ℕ := 3 * n - 1

def sequence_b (n : ℕ) : ℚ := (1 / 2 : ℚ)^n

def sequence_C (n : ℕ) : ℚ := sequence_a (sequence_a n) + sequence_b (sequence_a n)

def sum_of_first_n_terms (f : ℕ → ℚ) (n : ℕ) : ℚ :=
  (Finset.range n).sum f

theorem part_I (n : ℕ) : sequence_a n = 3 * n - 1 ∧ sequence_b n = (1 / 2)^n :=
by {
  sorry
}

theorem part_II (n : ℕ) : sum_of_first_n_terms sequence_C n =
  (n * (9 * n + 1) / 2) - (2 / 7) * (1 / 8)^n + (2 / 7) :=
by {
  sorry
}

end part_I_part_II_l102_102496


namespace trigonometric_identity_l102_102790

theorem trigonometric_identity :
  (2 * Real.sin (10 * Real.pi / 180) - Real.cos (20 * Real.pi / 180)) / Real.cos (70 * Real.pi / 180) = - Real.sqrt 3 := 
by
  sorry

end trigonometric_identity_l102_102790


namespace problem1_problem2_l102_102836

theorem problem1 : 3 / Real.sqrt 3 + (Real.pi + Real.sqrt 3)^0 + abs (Real.sqrt 3 - 2) = 3 := 
by
  sorry

theorem problem2 : (3 * Real.sqrt 12 - 2 * Real.sqrt (1 / 3) + Real.sqrt 48) / Real.sqrt 3 = 28 / 3 :=
by
  sorry

end problem1_problem2_l102_102836


namespace find_initial_apples_l102_102365

def initial_apples (a b c : ℕ) : Prop :=
  b + c = a

theorem find_initial_apples (a b initial_apples : ℕ) (h : b + initial_apples = a) : initial_apples = 8 :=
by
  sorry

end find_initial_apples_l102_102365


namespace geometric_sequence_product_l102_102940

theorem geometric_sequence_product
    (a : ℕ → ℝ)
    (r : ℝ)
    (h₀ : a 1 = 1 / 9)
    (h₃ : a 4 = 3)
    (h_geom : ∀ n, a (n + 1) = a n * r) :
    (a 1) * (a 2) * (a 3) * (a 4) * (a 5) = 1 :=
sorry

end geometric_sequence_product_l102_102940


namespace transformed_graph_passes_point_l102_102079

theorem transformed_graph_passes_point (f : ℝ → ℝ) 
  (h₁ : f 1 = 3) :
  f (-1) + 1 = 4 :=
by
  sorry

end transformed_graph_passes_point_l102_102079


namespace sum_of_number_and_preceding_l102_102469

theorem sum_of_number_and_preceding (n : ℤ) (h : 6 * n - 2 = 100) : n + (n - 1) = 33 :=
by {
  sorry
}

end sum_of_number_and_preceding_l102_102469


namespace boat_fuel_cost_per_hour_l102_102278

variable (earnings_per_photo : ℕ)
variable (shark_frequency_minutes : ℕ)
variable (hunting_hours : ℕ)
variable (expected_profit : ℕ)

def cost_of_fuel_per_hour (earnings_per_photo shark_frequency_minutes hunting_hours expected_profit : ℕ) : ℕ :=
  sorry

theorem boat_fuel_cost_per_hour
  (h₁ : earnings_per_photo = 15)
  (h₂ : shark_frequency_minutes = 10)
  (h₃ : hunting_hours = 5)
  (h₄ : expected_profit = 200) :
  cost_of_fuel_per_hour earnings_per_photo shark_frequency_minutes hunting_hours expected_profit = 50 :=
  sorry

end boat_fuel_cost_per_hour_l102_102278


namespace median_is_70_74_l102_102854

-- Define the histogram data as given
def histogram : List (ℕ × ℕ) :=
  [(85, 5), (80, 15), (75, 18), (70, 22), (65, 20), (60, 10), (55, 10)]

-- Function to calculate the cumulative sum at each interval
def cumulativeSum (hist : List (ℕ × ℕ)) : List (ℕ × ℕ) :=
  hist.scanl (λ acc pair => (pair.1, acc.2 + pair.2)) (0, 0)

-- Function to find the interval where the median lies
def medianInterval (hist : List (ℕ × ℕ)) : ℕ :=
  let cumSum := cumulativeSum hist
  -- The median is the 50th and 51st scores
  let medianPos := 50
  -- Find the interval that contains the median position
  List.find? (λ pair => medianPos ≤ pair.2) cumSum |>.getD (0, 0) |>.1

-- The theorem stating that the median interval is 70-74
theorem median_is_70_74 : medianInterval histogram = 70 :=
  by sorry

end median_is_70_74_l102_102854


namespace joe_spent_255_minutes_l102_102879

-- Define the time taken to cut hair for women, men, and children
def time_per_woman : Nat := 50
def time_per_man : Nat := 15
def time_per_child : Nat := 25

-- Define the number of haircuts for each category
def women_haircuts : Nat := 3
def men_haircuts : Nat := 2
def children_haircuts : Nat := 3

-- Compute the total time spent cutting hair
def total_time_spent : Nat :=
  (women_haircuts * time_per_woman) +
  (men_haircuts * time_per_man) +
  (children_haircuts * time_per_child)

-- The theorem stating the total time spent is equal to 255 minutes
theorem joe_spent_255_minutes : total_time_spent = 255 := by
  sorry

end joe_spent_255_minutes_l102_102879


namespace algebraic_identity_l102_102914

theorem algebraic_identity (m : ℝ) (h : m^2 + m - 1 = 0) : m^3 + 2 * m^2 - 2001 = -2000 :=
by
  sorry

end algebraic_identity_l102_102914


namespace correct_oblique_projection_conclusions_l102_102794

def oblique_projection (shape : Type) : Type := shape

theorem correct_oblique_projection_conclusions :
  (oblique_projection Triangle = Triangle) ∧
  (oblique_projection Parallelogram = Parallelogram) ↔
  (oblique_projection Square ≠ Square) ∧
  (oblique_projection Rhombus ≠ Rhombus) :=
by
  sorry

end correct_oblique_projection_conclusions_l102_102794


namespace f_at_zero_f_on_negative_l102_102172

-- Define the odd function condition
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define the function f(x) for x > 0 condition
def f_on_positive (f : ℝ → ℝ) : Prop :=
  ∀ x, x > 0 → f x = x^2 + x - 1

-- Lean statement for the first proof: f(0) = 0
theorem f_at_zero (f : ℝ → ℝ) (h_odd : is_odd_function f) (h_positive : f_on_positive f) : f 0 = 0 :=
sorry

-- Lean statement for the second proof: for x < 0, f(x) = -x^2 + x + 1
theorem f_on_negative (f : ℝ → ℝ) (h_odd : is_odd_function f) (h_positive : f_on_positive f) :
  ∀ x, x < 0 → f x = -x^2 + x + 1 :=
sorry

end f_at_zero_f_on_negative_l102_102172


namespace find_smallest_angle_b1_l102_102578

-- Definitions and conditions
def smallest_angle_in_sector (b1 e : ℕ) (k : ℕ := 5) : Prop :=
  2 * b1 + (k - 1) * k * e = 360 ∧ b1 + 2 * e = 36

theorem find_smallest_angle_b1 (b1 e : ℕ) : smallest_angle_in_sector b1 e → b1 = 30 :=
  sorry

end find_smallest_angle_b1_l102_102578


namespace Borgnine_total_legs_l102_102359

def numChimps := 12
def numLions := 8
def numLizards := 5
def numTarantulas := 125

def chimpLegsEach := 2
def lionLegsEach := 4
def lizardLegsEach := 4
def tarantulaLegsEach := 8

def legsSeen := numChimps * chimpLegsEach +
                numLions * lionLegsEach +
                numLizards * lizardLegsEach

def legsToSee := numTarantulas * tarantulaLegsEach

def totalLegs := legsSeen + legsToSee

theorem Borgnine_total_legs : totalLegs = 1076 := by
  sorry

end Borgnine_total_legs_l102_102359


namespace find_unique_positive_integers_l102_102896

theorem find_unique_positive_integers (x y : ℕ) (hx : x > 0) (hy : y > 0) :
  3 ^ x + 7 = 2 ^ y → x = 2 ∧ y = 4 :=
by
  -- Proof will go here
  sorry

end find_unique_positive_integers_l102_102896


namespace TwentyFifthMultipleOfFour_l102_102892

theorem TwentyFifthMultipleOfFour (n : ℕ) (h : ∀ k, 0 <= k ∧ k <= 24 → n = 16 + 4 * k) : n = 112 :=
by
  sorry

end TwentyFifthMultipleOfFour_l102_102892


namespace sprouted_percentage_l102_102295

-- Define the initial conditions
def cherryPits := 80
def saplingsSold := 6
def saplingsLeft := 14

-- Define the calculation of the total saplings that sprouted
def totalSaplingsSprouted := saplingsSold + saplingsLeft

-- Define the percentage calculation
def percentageSprouted := (totalSaplingsSprouted / cherryPits) * 100

-- The theorem to be proved
theorem sprouted_percentage : percentageSprouted = 25 := by
  sorry

end sprouted_percentage_l102_102295


namespace average_salary_technicians_correct_l102_102770

section
variable (average_salary_all : ℝ)
variable (total_workers : ℕ)
variable (average_salary_rest : ℝ)
variable (num_technicians : ℕ)

noncomputable def average_salary_technicians
  (h1 : average_salary_all = 8000)
  (h2 : total_workers = 30)
  (h3 : average_salary_rest = 6000)
  (h4 : num_technicians = 10)
  : ℝ :=
  12000

theorem average_salary_technicians_correct
  (h1 : average_salary_all = 8000)
  (h2 : total_workers = 30)
  (h3 : average_salary_rest = 6000)
  (h4 : num_technicians = 10)
  : average_salary_technicians average_salary_all total_workers average_salary_rest num_technicians h1 h2 h3 h4 = 12000 :=
sorry

end

end average_salary_technicians_correct_l102_102770


namespace expand_expression_l102_102847

variable (x : ℝ)

theorem expand_expression : (9 * x + 4) * (2 * x ^ 2) = 18 * x ^ 3 + 8 * x ^ 2 :=
by sorry

end expand_expression_l102_102847


namespace initial_students_per_class_l102_102468

theorem initial_students_per_class
  (S : ℕ) 
  (parents chaperones left_students left_chaperones : ℕ)
  (teachers remaining_individuals : ℕ)
  (h1 : parents = 5)
  (h2 : chaperones = 2)
  (h3 : left_students = 10)
  (h4 : left_chaperones = 2)
  (h5 : teachers = 2)
  (h6 : remaining_individuals = 15)
  (h7 : 2 * S + parents + teachers - left_students - left_chaperones = remaining_individuals) :
  S = 10 :=
by
  sorry

end initial_students_per_class_l102_102468


namespace num_ways_to_distribute_balls_into_boxes_l102_102911

theorem num_ways_to_distribute_balls_into_boxes :
  let balls := 5
  let boxes := 3
  (boxes ^ balls = 243) := 
by
  sorry

end num_ways_to_distribute_balls_into_boxes_l102_102911


namespace garin_homework_pages_l102_102720

theorem garin_homework_pages (pages_per_day : ℕ) (days : ℕ) (total_pages : ℕ) : 
    pages_per_day = 19 → 
    days = 24 → 
    total_pages = pages_per_day * days → 
    total_pages = 456 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end garin_homework_pages_l102_102720


namespace decrease_in_demand_correct_l102_102181

noncomputable def proportionate_decrease_in_demand (p e : ℝ) : ℝ :=
  1 - (1 / (1 + e * p))

theorem decrease_in_demand_correct :
  proportionate_decrease_in_demand 0.20 1.5 = 0.23077 :=
by
  sorry

end decrease_in_demand_correct_l102_102181


namespace b_eq_6_l102_102348

theorem b_eq_6 (a b : ℤ) (h₁ : |a| = 1) (h₂ : ∀ x : ℝ, a * x^2 - 2 * x - b + 5 = 0 → x < 0) : b = 6 := 
by
  sorry

end b_eq_6_l102_102348


namespace rolls_in_package_l102_102105

theorem rolls_in_package (n : ℕ) :
  (9 : ℝ) = (n : ℝ) * (1 - 0.25) → n = 12 :=
by
  sorry

end rolls_in_package_l102_102105


namespace ratio_of_diagonals_to_sides_l102_102961

-- Define the given parameters and formula
def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- State the theorem
theorem ratio_of_diagonals_to_sides (n : ℕ) (h : n = 5) : 
  (num_diagonals n) / n = 1 :=
by
  -- Proof skipped
  sorry

end ratio_of_diagonals_to_sides_l102_102961


namespace regular_polygon_sides_l102_102263

theorem regular_polygon_sides (n : ℕ) (h : 0 < n) (h_angle : (n - 2) * 180 = 144 * n) :
  n = 10 :=
sorry

end regular_polygon_sides_l102_102263


namespace unique_coprime_solution_l102_102393

theorem unique_coprime_solution 
  (p : ℕ) (a b m r : ℕ) 
  (hp : Nat.Prime p) 
  (hp_odd : p % 2 = 1)
  (hp_nmid_ab : ¬ (p ∣ a * b))
  (hab_gt_m2 : a * b > m^2) :
  ∃! (x y : ℕ), Nat.Coprime x y ∧ (a * x^2 + b * y^2 = m * p ^ r) := 
sorry

end unique_coprime_solution_l102_102393


namespace prob_2_lt_X_le_4_l102_102013

-- Define the PMF of the random variable X
noncomputable def pmf_X (k : ℕ) : ℝ :=
  if h : k ≥ 1 then 1 / (2 ^ k) else 0

-- Define the probability that X lies in the range (2, 4]
noncomputable def P_2_lt_X_le_4 : ℝ :=
  pmf_X 3 + pmf_X 4

-- Theorem stating the probability of x lying in (2, 4) is 3/16.
theorem prob_2_lt_X_le_4 : P_2_lt_X_le_4 = 3 / 16 := 
by
  -- Provide proof here
  sorry

end prob_2_lt_X_le_4_l102_102013


namespace difference_of_two_numbers_l102_102326

theorem difference_of_two_numbers :
  ∃ S : ℕ, S * 16 + 15 = 1600 ∧ 1600 - S = 1501 :=
by
  sorry

end difference_of_two_numbers_l102_102326


namespace number_of_people_who_didnt_do_both_l102_102804

def total_graduates : ℕ := 73
def graduates_both : ℕ := 13

theorem number_of_people_who_didnt_do_both : total_graduates - graduates_both = 60 :=
by
  sorry

end number_of_people_who_didnt_do_both_l102_102804


namespace number_of_integer_solutions_l102_102859

theorem number_of_integer_solutions :
  ∃ (n : ℕ), 
  (∀ (x y : ℤ), 2 * x + 3 * y = 7 ∧ 5 * x + n * y = n ^ 2) ∧
  (n = 8) := 
sorry

end number_of_integer_solutions_l102_102859


namespace percentage_difference_l102_102445

variable (x y z : ℝ)

theorem percentage_difference (h1 : y = 1.75 * x) (h2 : z = 0.60 * y) :
  (1 - x / z) * 100 = 4.76 :=
by
  sorry

end percentage_difference_l102_102445


namespace inequality_solution_set_l102_102609

theorem inequality_solution_set :
  {x : ℝ | (3 - x) * (1 + x) > 0} = {x : ℝ | -1 < x ∧ x < 3} :=
sorry

end inequality_solution_set_l102_102609


namespace power_division_l102_102869

theorem power_division (a b : ℕ) (h : 64 = 8^2) : (8 ^ 15) / (64 ^ 7) = 8 := by
  sorry

end power_division_l102_102869


namespace soda_cost_l102_102877

variable {b s f : ℕ}

theorem soda_cost :
    5 * b + 3 * s + 2 * f = 520 ∧
    3 * b + 2 * s + f = 340 →
    s = 80 :=
by
  sorry

end soda_cost_l102_102877


namespace ten_times_six_x_plus_fourteen_pi_l102_102852

theorem ten_times_six_x_plus_fourteen_pi (x : ℝ) (Q : ℝ) (h : 5 * (3 * x + 7 * Real.pi) = Q) : 
  10 * (6 * x + 14 * Real.pi) = 4 * Q :=
by
  sorry

end ten_times_six_x_plus_fourteen_pi_l102_102852


namespace number_of_dots_on_A_number_of_dots_on_B_number_of_dots_on_C_number_of_dots_on_D_l102_102411

variable (A B C D : ℕ)
variable (dice : ℕ → ℕ)

-- Conditions
axiom dice_faces : ∀ {i : ℕ}, 1 ≤ i ∧ i ≤ 6 → ∃ j, dice i = j
axiom opposite_faces_sum : ∀ {i j : ℕ}, dice i + dice j = 7
axiom configuration : True -- Placeholder for the specific arrangement configuration

-- Questions and Proof Statements
theorem number_of_dots_on_A :
  A = 3 := sorry

theorem number_of_dots_on_B :
  B = 5 := sorry

theorem number_of_dots_on_C :
  C = 6 := sorry

theorem number_of_dots_on_D :
  D = 5 := sorry

end number_of_dots_on_A_number_of_dots_on_B_number_of_dots_on_C_number_of_dots_on_D_l102_102411


namespace goblin_treasure_l102_102034

theorem goblin_treasure : 
  (∃ d : ℕ, 8000 + 300 * d = 5000 + 500 * d) ↔ ∃ (d : ℕ), d = 15 :=
by
  sorry

end goblin_treasure_l102_102034


namespace find_sum_of_terms_l102_102796

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∃ (r : ℝ), ∀ n : ℕ, a (n + 1) = r * a n

def given_conditions (a : ℕ → ℝ) : Prop :=
geometric_sequence a ∧ (a 4 + a 7 = 2) ∧ (a 5 * a 6 = -8)

theorem find_sum_of_terms (a : ℕ → ℝ) (h : given_conditions a) : a 1 + a 10 = -7 :=
sorry

end find_sum_of_terms_l102_102796


namespace pseudocode_output_l102_102743

theorem pseudocode_output :
  let s := 0
  let t := 1
  let (s, t) := (List.range 3).foldl (fun (s, t) i => (s + (i + 1), t * (i + 1))) (s, t)
  let r := s * t
  r = 36 :=
by
  sorry

end pseudocode_output_l102_102743


namespace minimum_tangent_length_l102_102054

theorem minimum_tangent_length
  (a b : ℝ)
  (h_circle : ∀ x y : ℝ, x^2 + y^2 + 2 * x - 4 * y + 3 = 0)
  (h_symmetry : ∀ x y : ℝ, (x + 1)^2 + (y - 2)^2 = 4 → 2 * a * x + b * y + 6 = 0) :
  ∃ t : ℝ, t = 2 :=
by sorry

end minimum_tangent_length_l102_102054


namespace pool_capacity_l102_102219

-- Define the total capacity of the pool as a variable
variable (C : ℝ)

-- Define the conditions
def additional_water_needed (x : ℝ) : Prop :=
  x = 300

def increases_by_25_percent (x : ℝ) (y : ℝ) : Prop :=
  y = x * 0.25

-- State the proof problem
theorem pool_capacity :
  ∃ C : ℝ, additional_water_needed 300 ∧ increases_by_25_percent (0.75 * C) 300 ∧ C = 1200 :=
sorry

end pool_capacity_l102_102219


namespace JillTotalTaxPercentage_l102_102248

noncomputable def totalTaxPercentage : ℝ :=
  let totalSpending (beforeDiscount : ℝ) : ℝ := 100
  let clothingBeforeDiscount : ℝ := 0.4 * totalSpending 100
  let foodBeforeDiscount : ℝ := 0.2 * totalSpending 100
  let electronicsBeforeDiscount : ℝ := 0.1 * totalSpending 100
  let cosmeticsBeforeDiscount : ℝ := 0.2 * totalSpending 100
  let householdBeforeDiscount : ℝ := 0.1 * totalSpending 100

  let clothingDiscount : ℝ := 0.1 * clothingBeforeDiscount
  let foodDiscount : ℝ := 0.05 * foodBeforeDiscount
  let electronicsDiscount : ℝ := 0.15 * electronicsBeforeDiscount

  let clothingAfterDiscount := clothingBeforeDiscount - clothingDiscount
  let foodAfterDiscount := foodBeforeDiscount - foodDiscount
  let electronicsAfterDiscount := electronicsBeforeDiscount - electronicsDiscount
  
  let taxOnClothing := 0.06 * clothingAfterDiscount
  let taxOnFood := 0.0 * foodAfterDiscount
  let taxOnElectronics := 0.1 * electronicsAfterDiscount
  let taxOnCosmetics := 0.08 * cosmeticsBeforeDiscount
  let taxOnHousehold := 0.04 * householdBeforeDiscount

  let totalTaxPaid := taxOnClothing + taxOnFood + taxOnElectronics + taxOnCosmetics + taxOnHousehold
  (totalTaxPaid / totalSpending 100) * 100

theorem JillTotalTaxPercentage :
  totalTaxPercentage = 5.01 := by
  sorry

end JillTotalTaxPercentage_l102_102248


namespace pears_value_equivalence_l102_102843

-- Condition: $\frac{3}{4}$ of $16$ apples are worth $12$ pears
def apples_to_pears (a p : ℕ) : Prop :=
  (3 * 16 / 4 * a = 12 * p)

-- Question: How many pears (p) are equivalent in value to $\frac{2}{3}$ of $9$ apples?
def pears_equivalent_to_apples (p : ℕ) : Prop :=
  (2 * 9 / 3 * p = 6)

theorem pears_value_equivalence (p : ℕ) (a : ℕ) (h1 : apples_to_pears a p) (h2 : pears_equivalent_to_apples p) : 
  p = 6 :=
sorry

end pears_value_equivalence_l102_102843


namespace distance_in_interval_l102_102748

open Set Real

def distance_to_town (d : ℝ) : Prop :=
d < 8 ∧ 7 < d ∧ 6 < d

theorem distance_in_interval (d : ℝ) : distance_to_town d → d ∈ Ioo 7 8 :=
by
  intro h
  have d_in_Ioo_8 := h.left
  have d_in_Ioo_7 := h.right.left
  have d_in_Ioo_6 := h.right.right
  /- The specific steps for combining inequalities aren't needed for the final proof. -/
  sorry

end distance_in_interval_l102_102748


namespace expression_evaluation_l102_102202

theorem expression_evaluation :
  (3 * Real.sqrt 12 - 2 * Real.sqrt (1 / 3) + Real.sqrt 48) / (2 * Real.sqrt 3) + (Real.sqrt (1 / 3))^2 = 5 :=
by
  sorry

end expression_evaluation_l102_102202


namespace boxes_with_neither_l102_102473

-- Definitions translating the conditions from the problem
def total_boxes : Nat := 15
def boxes_with_markers : Nat := 8
def boxes_with_crayons : Nat := 4
def boxes_with_both : Nat := 3

-- The theorem statement to prove
theorem boxes_with_neither : total_boxes - (boxes_with_markers + boxes_with_crayons - boxes_with_both) = 6 := by
  -- Proof will go here
  sorry

end boxes_with_neither_l102_102473


namespace last_number_of_nth_row_sum_of_numbers_in_nth_row_position_of_2008_l102_102617

theorem last_number_of_nth_row (n : ℕ) : 
    let last_number := 2^n - 1
    last_number = 2^n - 1 := 
sorry

theorem sum_of_numbers_in_nth_row (n : ℕ) :
    let sum := (3 * 2^(n-3)) - 2^(n-2)
    sum = (3 * 2^(n-3)) - 2^(n-2) :=
sorry

theorem position_of_2008 : 
    let position := 985
    position = 985 :=
sorry

end last_number_of_nth_row_sum_of_numbers_in_nth_row_position_of_2008_l102_102617


namespace combined_salaries_l102_102178

variable (S_A S_B S_C S_D S_E : ℝ)

theorem combined_salaries 
    (h1 : S_C = 16000)
    (h2 : (S_A + S_B + S_C + S_D + S_E) / 5 = 9000) : 
    S_A + S_B + S_D + S_E = 29000 :=
by 
    sorry

end combined_salaries_l102_102178


namespace quadratic_roots_ratio_l102_102504

theorem quadratic_roots_ratio (r1 r2 p q n : ℝ) (h1 : p = r1 * r2) (h2 : q = -(r1 + r2)) (h3 : p ≠ 0) (h4 : q ≠ 0) (h5 : n ≠ 0) (h6 : r1 ≠ 0) (h7 : r2 ≠ 0) (h8 : x^2 + q * x + p = 0) (h9 : x^2 + p * x + n = 0) :
  n / q = -3 :=
by
  sorry

end quadratic_roots_ratio_l102_102504


namespace greatest_integer_floor_div_l102_102737

-- Define the parameters
def a : ℕ := 3^100 + 2^105
def b : ℕ := 3^96 + 2^101

-- Formulate the proof statement
theorem greatest_integer_floor_div (a b : ℕ) : 
  a = 3^100 + 2^105 →
  b = 3^96 + 2^101 →
  (a / b) = 16 := 
by
  intros ha hb
  sorry

end greatest_integer_floor_div_l102_102737


namespace exponential_monotonicity_example_l102_102501

theorem exponential_monotonicity_example (m n : ℕ) (a b : ℝ) (h1 : a = 0.2 ^ m) (h2 : b = 0.2 ^ n) (h3 : m > n) : a < b :=
by
  sorry

end exponential_monotonicity_example_l102_102501


namespace solve_equation_l102_102044

theorem solve_equation (x : ℚ) (h₁ : x ≠ 1) (h₂ : x ≠ -1) : 
  (2 * x / (x + 1) - 2 = 3 / (x^2 - 1)) → x = -0.5 := 
by
  sorry

end solve_equation_l102_102044


namespace correct_regression_equation_l102_102525

-- Problem Statement
def negatively_correlated (x y : ℝ) : Prop := sorry -- Define negative correlation for x, y
def sample_mean_x : ℝ := 3
def sample_mean_y : ℝ := 3.5
def regression_equation (b0 b1 : ℝ) (x : ℝ) : ℝ := b0 + b1 * x

theorem correct_regression_equation 
    (H_neg_corr : negatively_correlated x y) :
    regression_equation 9.5 (-2) sample_mean_x = sample_mean_y :=
by
    -- The proof will go here, skipping with sorry
    sorry

end correct_regression_equation_l102_102525


namespace possible_to_fill_array_l102_102174

open BigOperators

theorem possible_to_fill_array :
  ∃ (f : (Fin 10) × (Fin 10) → ℕ),
    (∀ i j : Fin 10, 
      (i ≠ 0 → f (i, j) ∣ f (i - 1, j) ∧ f (i, j) ≠ f (i - 1, j))) ∧
    (∀ i : Fin 10, ∃ n : ℕ, ∀ j : Fin 10, f (i, j) = n + j) :=
sorry

end possible_to_fill_array_l102_102174


namespace AM_GM_Inequality_four_vars_l102_102735

theorem AM_GM_Inequality_four_vars (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  a / b + b / c + c / d + d / a ≥ 4 :=
by sorry

end AM_GM_Inequality_four_vars_l102_102735


namespace quadratic_condition_l102_102696

theorem quadratic_condition (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - 2 * a * x + 3 > 0) ↔ (0 ≤ a ∧ a < 3) :=
sorry

end quadratic_condition_l102_102696


namespace find_y_l102_102097

theorem find_y (k p y : ℝ) (hk : k ≠ 0) (hp : p ≠ 0) 
  (h : (y - 2 * k)^2 - (y - 3 * k)^2 = 4 * k^2 - p) : 
  y = -(p + k^2) / (2 * k) :=
sorry

end find_y_l102_102097


namespace find_sum_of_squares_l102_102205

theorem find_sum_of_squares (a b c m : ℤ) (h1 : a + b + c = 0) (h2 : a * b + b * c + a * c = -2023) (h3 : a * b * c = -m) : a^2 + b^2 + c^2 = 4046 := by
  sorry

end find_sum_of_squares_l102_102205


namespace volume_ratio_octahedron_cube_l102_102711

theorem volume_ratio_octahedron_cube 
  (s : ℝ) -- edge length of the octahedron
  (h := s * Real.sqrt 2 / 2) -- height of one of the pyramids forming the octahedron
  (volume_O := s^3 * Real.sqrt 2 / 3) -- volume of the octahedron
  (a := (2 * s) / Real.sqrt 3) -- edge length of the cube
  (volume_C := (a ^ 3)) -- volume of the cube
  (diag_C : ℝ := 2 * s) -- diagonal of the cube
  (h_diag : diag_C = (a * Real.sqrt 3)) -- relation of diagonal to edge length of the cube
  (ratio := volume_O / volume_C) -- ratio of the volumes
  (desired_ratio := 3 / 8) -- given ratio in simplified form
  (m := 3) -- first part of the ratio
  (n := 8) -- second part of the ratio
  (rel_prime : Nat.gcd m n = 1) -- m and n are relatively prime
  (correct_ratio : ratio = desired_ratio) -- the ratio is correct
  : m + n = 11 :=
by
  sorry 

end volume_ratio_octahedron_cube_l102_102711


namespace S4_equals_15_l102_102895

noncomputable def S_n (q : ℝ) (n : ℕ) := (1 - q^n) / (1 - q)

theorem S4_equals_15 (q : ℝ) (n : ℕ) (h1 : S_n q 1 = 1) (h2 : S_n q 5 = 5 * S_n q 3 - 4) : 
  S_n q 4 = 15 :=
by
  sorry

end S4_equals_15_l102_102895


namespace grandma_contribution_l102_102242

def trip_cost : ℝ := 485
def candy_bar_profit : ℝ := 1.25
def candy_bars_sold : ℕ := 188
def amount_earned_from_selling_candy_bars : ℝ := candy_bars_sold * candy_bar_profit
def amount_grandma_gave : ℝ := trip_cost - amount_earned_from_selling_candy_bars

theorem grandma_contribution :
  amount_grandma_gave = 250 := by
  sorry

end grandma_contribution_l102_102242


namespace jane_original_number_l102_102776

theorem jane_original_number (x : ℝ) (h : 5 * (3 * x + 16) = 250) : x = 34 / 3 := 
sorry

end jane_original_number_l102_102776


namespace coin_ratio_l102_102882

theorem coin_ratio (coins_1r coins_50p coins_25p : ℕ) (value_1r value_50p value_25p : ℕ) :
  coins_1r = 120 → coins_50p = 120 → coins_25p = 120 →
  value_1r = coins_1r * 1 → value_50p = coins_50p * 50 → value_25p = coins_25p * 25 →
  value_1r + value_50p + value_25p = 210 →
  (coins_1r : ℚ) / (coins_50p + coins_25p : ℚ) = (1 / 1) :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end coin_ratio_l102_102882


namespace ne_of_P_l102_102575

-- Define the initial proposition P
def P : Prop := ∀ m : ℝ, (0 ≤ m → 4^m ≥ 4 * m)

-- Define the negation of P
def not_P : Prop := ∃ m : ℝ, (0 ≤ m ∧ 4^m < 4 * m)

-- The theorem we need to prove
theorem ne_of_P : ¬P ↔ not_P :=
by
  sorry

end ne_of_P_l102_102575


namespace ratio_of_fixing_times_is_two_l102_102597

noncomputable def time_per_shirt : ℝ := 1.5
noncomputable def number_of_shirts : ℕ := 10
noncomputable def number_of_pants : ℕ := 12
noncomputable def hourly_rate : ℝ := 30
noncomputable def total_cost : ℝ := 1530

theorem ratio_of_fixing_times_is_two :
  let total_hours := total_cost / hourly_rate
  let shirt_hours := number_of_shirts * time_per_shirt
  let pant_hours := total_hours - shirt_hours
  let time_per_pant := pant_hours / number_of_pants
  (time_per_pant / time_per_shirt) = 2 :=
by
  sorry

end ratio_of_fixing_times_is_two_l102_102597


namespace brianna_more_chocolates_than_alix_l102_102098

def Nick_ClosetA : ℕ := 10
def Nick_ClosetB : ℕ := 6
def Alix_ClosetA : ℕ := 3 * Nick_ClosetA
def Alix_ClosetB : ℕ := 3 * Nick_ClosetA
def Mom_Takes_From_AlixA : ℚ := (1/4:ℚ) * Alix_ClosetA
def Brianna_ClosetA : ℚ := 2 * (Nick_ClosetA + Alix_ClosetA - Mom_Takes_From_AlixA)
def Brianna_ClosetB_after : ℕ := 18
def Brianna_ClosetB : ℚ := Brianna_ClosetB_after / (0.8:ℚ)

def Brianna_Total : ℚ := Brianna_ClosetA + Brianna_ClosetB
def Alix_Total : ℚ := Alix_ClosetA + Alix_ClosetB
def Difference : ℚ := Brianna_Total - Alix_Total

theorem brianna_more_chocolates_than_alix : Difference = 35 := by
  sorry

end brianna_more_chocolates_than_alix_l102_102098


namespace circle_radius_of_complex_roots_l102_102424

theorem circle_radius_of_complex_roots (z : ℂ) (hz : (z - 1)^3 = 8 * z^3) : 
  ∃ r : ℝ, r = 1 / Real.sqrt 3 :=
by
  sorry

end circle_radius_of_complex_roots_l102_102424


namespace masha_can_pay_with_5_ruble_coins_l102_102402

theorem masha_can_pay_with_5_ruble_coins (p c n : ℤ) (h : 2 * p + c + 7 * n = 100) : (p + 3 * c + n) % 5 = 0 :=
  sorry

end masha_can_pay_with_5_ruble_coins_l102_102402


namespace pencil_length_l102_102170

theorem pencil_length :
  let purple := 1.5
  let black := 0.5
  let blue := 2
  purple + black + blue = 4 := by sorry

end pencil_length_l102_102170


namespace sum_of_numbers_l102_102756

theorem sum_of_numbers : 148 + 35 + 17 + 13 + 9 = 222 := 
by
  sorry

end sum_of_numbers_l102_102756


namespace remove_toothpicks_l102_102006

-- Definitions based on problem conditions
def toothpicks := 40
def triangles := 40
def initial_triangulation := True
def additional_condition := True

-- Statement to be proved
theorem remove_toothpicks :
  initial_triangulation ∧ additional_condition ∧ (triangles > 40) → ∃ (t: ℕ), t = 15 :=
by
  sorry

end remove_toothpicks_l102_102006


namespace ratio_of_ages_l102_102686

-- Given conditions
def present_age_sum (H J : ℕ) : Prop :=
  H + J = 43

def present_ages (H J : ℕ) : Prop := 
  H = 27 ∧ J = 16

def multiple_of_age (H J k : ℕ) : Prop :=
  H - 5 = k * (J - 5)

-- Prove that the ratio of Henry's age to Jill's age 5 years ago was 2:1
theorem ratio_of_ages (H J k : ℕ) 
  (h_sum : present_age_sum H J)
  (h_present : present_ages H J)
  (h_multiple : multiple_of_age H J k) :
  (H - 5) / (J - 5) = 2 :=
by
  sorry

end ratio_of_ages_l102_102686


namespace find_y_l102_102608

noncomputable def a := (3/5) * 2500
noncomputable def b := (2/7) * ((5/8) * 4000 + (1/4) * 3600 - (11/20) * 7200)
noncomputable def c (y : ℚ) := (3/10) * y
def result (a b c : ℚ) := a * b / c

theorem find_y : ∃ y : ℚ, result a b (c y) = 25000 ∧ y = -4/21 := 
by
  sorry

end find_y_l102_102608


namespace equivalent_fraction_l102_102657

theorem equivalent_fraction (b : ℕ) (h : b = 2024) :
  (b^3 - 2 * b^2 * (b + 1) + 3 * b * (b + 1)^2 - (b + 1)^3 + 4) / (b * (b + 1)) = 2022 := by
  rw [h]
  sorry

end equivalent_fraction_l102_102657


namespace polygon_interior_angles_l102_102579

theorem polygon_interior_angles {n : ℕ} (h : (n - 2) * 180 = 900) : n = 7 :=
by
  sorry

end polygon_interior_angles_l102_102579


namespace negation_of_proposition_l102_102004

theorem negation_of_proposition :
  (¬ ∃ x : ℝ, x ≤ -1 ∨ x ≥ 2) ↔ ∀ x : ℝ, -1 < x ∧ x < 2 :=
by
  sorry

end negation_of_proposition_l102_102004


namespace solve_for_A_l102_102391

theorem solve_for_A : 
  ∃ (A B : ℕ), (100 * A + 78) - (200 + 10 * B + 4) = 364 → A = 5 :=
by
  sorry

end solve_for_A_l102_102391


namespace correct_transformation_C_l102_102932

-- Define the conditions as given in the problem
def condition_A (x : ℝ) : Prop := 4 + x = 3 ∧ x = 3 - 4
def condition_B (x : ℝ) : Prop := (1 / 3) * x = 0 ∧ x = 0
def condition_C (y : ℝ) : Prop := 5 * y = -4 * y + 2 ∧ 5 * y + 4 * y = 2
def condition_D (a : ℝ) : Prop := (1 / 2) * a - 1 = 3 * a ∧ a - 2 = 6 * a

-- The theorem to prove that condition_C is correctly transformed
theorem correct_transformation_C : condition_C 1 := 
by sorry

end correct_transformation_C_l102_102932


namespace inequality_solution_l102_102784

theorem inequality_solution (x : ℝ) :
  (x + 2 > 3 * (2 - x) ∧ x < (x + 3) / 2) ↔ 1 < x ∧ x < 3 := sorry

end inequality_solution_l102_102784


namespace find_a_of_exponential_inverse_l102_102631

theorem find_a_of_exponential_inverse (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) (h₂ : ∀ x, a^x = 9 ↔ x = 2) : a = 3 := 
by
  sorry

end find_a_of_exponential_inverse_l102_102631


namespace pq_eqv_l102_102338

theorem pq_eqv (p q : Prop) : 
  ((¬ p ∧ ¬ q) ∧ (p ∨ q)) ↔ ((p ∧ ¬ q) ∨ (¬ p ∧ q)) :=
by
  sorry

end pq_eqv_l102_102338


namespace Lloyd_hourly_rate_is_3_5_l102_102810

/-!
Lloyd normally works 7.5 hours per day and earns a certain amount per hour.
For each hour he works in excess of 7.5 hours on a given day, he is paid 1.5 times his regular rate.
If Lloyd works 10.5 hours on a given day, he earns $42 for that day.
-/

variable (Lloyd_hourly_rate : ℝ)  -- regular hourly rate

def Lloyd_daily_earnings (total_hours : ℝ) (regular_hours : ℝ) (hourly_rate : ℝ) : ℝ :=
  let excess_hours := total_hours - regular_hours
  let excess_earnings := excess_hours * (1.5 * hourly_rate)
  let regular_earnings := regular_hours * hourly_rate
  excess_earnings + regular_earnings

-- Given conditions
axiom H1 : 7.5 = 7.5
axiom H2 : ∀ R : ℝ, Lloyd_hourly_rate = R
axiom H3 : ∀ R : ℝ, ∀ excess_hours : ℝ, Lloyd_hourly_rate + excess_hours = 1.5 * R
axiom H4 : Lloyd_daily_earnings 10.5 7.5 Lloyd_hourly_rate = 42

-- Prove Lloyd earns $3.50 per hour.
theorem Lloyd_hourly_rate_is_3_5 : Lloyd_hourly_rate = 3.5 :=
sorry

end Lloyd_hourly_rate_is_3_5_l102_102810


namespace cost_of_white_washing_l102_102814

-- Definitions for room dimensions, doors, windows, and cost per square foot
def length : ℕ := 25
def width : ℕ := 15
def height1 : ℕ := 12
def height2 : ℕ := 8
def door_height : ℕ := 6
def door_width : ℕ := 3
def window_height : ℕ := 4
def window_width : ℕ := 3
def cost_per_sq_ft : ℕ := 10
def ceiling_decoration_area : ℕ := 10

-- Definitions for the areas calculation
def area_walls_height1 : ℕ := 2 * (length * height1)
def area_walls_height2 : ℕ := 2 * (width * height2)
def total_wall_area : ℕ := area_walls_height1 + area_walls_height2

def area_one_door : ℕ := door_height * door_width
def total_doors_area : ℕ := 2 * area_one_door

def area_one_window : ℕ := window_height * window_width
def total_windows_area : ℕ := 3 * area_one_window

def adjusted_wall_area : ℕ := total_wall_area - total_doors_area - total_windows_area - ceiling_decoration_area

def total_cost : ℕ := adjusted_wall_area * cost_per_sq_ft

-- The theorem we want to prove
theorem cost_of_white_washing : total_cost = 7580 := by
  sorry

end cost_of_white_washing_l102_102814


namespace range_of_a_l102_102376

-- Defining the propositions P and Q 
def P (a : ℝ) : Prop := ∀ x : ℝ, a*x^2 + a*x + 1 > 0
def Q (a : ℝ) : Prop := ∃ x1 x2 : ℝ, x1^2 - x1 + a = 0 ∧ x2^2 - x2 + a = 0

-- Stating the theorem
theorem range_of_a (a : ℝ) :
  (P a ∧ ¬Q a) ∨ (¬P a ∧ Q a) ↔ a ∈ Set.Ioo (1/4 : ℝ) 4 ∪ Set.Iio 0 :=
sorry

end range_of_a_l102_102376


namespace roll_seven_dice_at_least_one_pair_no_three_l102_102435

noncomputable def roll_seven_dice_probability : ℚ :=
  let total_outcomes := (6^7 : ℚ)
  let one_pair_case := (6 * 21 * 120 : ℚ)
  let two_pairs_case := (15 * 21 * 10 * 24 : ℚ)
  let successful_outcomes := one_pair_case + two_pairs_case
  successful_outcomes / total_outcomes

theorem roll_seven_dice_at_least_one_pair_no_three :
  roll_seven_dice_probability = 315 / 972 :=
by
  unfold roll_seven_dice_probability
  -- detailed steps to show the proof would go here
  sorry

end roll_seven_dice_at_least_one_pair_no_three_l102_102435


namespace find_age_of_b_l102_102855

variables (A B C : ℕ)

def average_abc (A B C : ℕ) : Prop := (A + B + C) / 3 = 28
def average_ac (A C : ℕ) : Prop := (A + C) / 2 = 29

theorem find_age_of_b (h1 : average_abc A B C) (h2 : average_ac A C) : B = 26 :=
by
  sorry

end find_age_of_b_l102_102855


namespace compute_expression_l102_102185

theorem compute_expression (y : ℕ) (h : y = 3) : (y^8 + 10 * y^4 + 25) / (y^4 + 5) = 86 :=
by
  rw [h]
  sorry

end compute_expression_l102_102185


namespace ellipse_equation_parabola_equation_l102_102026

noncomputable def ellipse_standard_equation (a b c : ℝ) : Prop :=
  a = 6 → b = 2 * Real.sqrt 5 → c = 4 → 
  ((∀ x y : ℝ, (y^2 / 36) + (x^2 / 20) = 1))

noncomputable def parabola_standard_equation (focus_x focus_y : ℝ) : Prop :=
  focus_x = 3 → focus_y = 0 → 
  (∀ x y : ℝ, y^2 = 12 * x)

theorem ellipse_equation : ellipse_standard_equation 6 (2 * Real.sqrt 5) 4 := by
  sorry

theorem parabola_equation : parabola_standard_equation 3 0 := by
  sorry

end ellipse_equation_parabola_equation_l102_102026


namespace sin_x_eq_2ab_div_a2_plus_b2_l102_102860

theorem sin_x_eq_2ab_div_a2_plus_b2
  (a b : ℝ) (x : ℝ)
  (h_tan : Real.tan x = 2 * a * b / (a^2 - b^2))
  (h_pos : 0 < b) (h_lt : b < a) (h_x : 0 < x ∧ x < Real.pi / 2) :
  Real.sin x = 2 * a * b / (a^2 + b^2) :=
by sorry

end sin_x_eq_2ab_div_a2_plus_b2_l102_102860


namespace speed_ratio_thirteen_l102_102422

noncomputable section

def speed_ratio (vNikita vCar : ℝ) : ℝ := vCar / vNikita

theorem speed_ratio_thirteen :
  ∀ (vNikita vCar : ℝ),
  (65 * vNikita = 5 * vCar) →
  speed_ratio vNikita vCar = 13 :=
by
  intros vNikita vCar h
  unfold speed_ratio
  sorry

end speed_ratio_thirteen_l102_102422


namespace rhombus_shorter_diagonal_l102_102550

theorem rhombus_shorter_diagonal (perimeter : ℝ) (angle_ratio : ℝ) (side_length diagonal_length : ℝ)
  (h₁ : perimeter = 9.6) 
  (h₂ : angle_ratio = 1 / 2) 
  (h₃ : side_length = perimeter / 4) 
  (h₄ : diagonal_length = side_length) :
  diagonal_length = 2.4 := 
sorry

end rhombus_shorter_diagonal_l102_102550


namespace hannah_total_spending_l102_102835

def sweatshirt_price : ℕ := 15
def sweatshirt_quantity : ℕ := 3
def t_shirt_price : ℕ := 10
def t_shirt_quantity : ℕ := 2
def socks_price : ℕ := 5
def socks_quantity : ℕ := 4
def jacket_price : ℕ := 50
def discount_rate : ℚ := 0.10

noncomputable def total_cost_before_discount : ℕ :=
  (sweatshirt_quantity * sweatshirt_price) +
  (t_shirt_quantity * t_shirt_price) +
  (socks_quantity * socks_price) +
  jacket_price

noncomputable def total_cost_after_discount : ℚ :=
  total_cost_before_discount - (discount_rate * total_cost_before_discount)

theorem hannah_total_spending : total_cost_after_discount = 121.50 := by
  sorry

end hannah_total_spending_l102_102835


namespace max_view_angle_dist_l102_102652

theorem max_view_angle_dist (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) : ∃ (x : ℝ), x = Real.sqrt (b * (a + b)) := by
  sorry

end max_view_angle_dist_l102_102652


namespace find_x_l102_102612

-- Define vectors
def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (2, 5)
def c (x : ℝ) : ℝ × ℝ := (3, x)

-- Dot product function
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Compute 8a - b
def sum_vec : ℝ × ℝ :=
  (8 * a.1 - b.1, 8 * a.2 - b.2)

-- Prove that x = 4 given condition
theorem find_x (x : ℝ) (h : dot_product sum_vec (c x) = 30) : x = 4 :=
by
  sorry

end find_x_l102_102612


namespace sum_of_faces_l102_102536

variable (a d b c e f : ℕ)
variable (pos_a : a > 0) (pos_d : d > 0) (pos_b : b > 0) (pos_c : c > 0) 
variable (pos_e : e > 0) (pos_f : f > 0)
variable (h : a * b * e + a * b * f + a * c * e + a * c * f + d * b * e + d * b * f + d * c * e + d * c * f = 1176)

theorem sum_of_faces : a + d + b + c + e + f = 33 := by
  sorry

end sum_of_faces_l102_102536


namespace muffin_banana_costs_l102_102413

variable (m b : ℕ) -- Using natural numbers for non-negativity

theorem muffin_banana_costs (h : 3 * (3 * m + 5 * b) = 4 * m + 10 * b) : m = b :=
by
  sorry

end muffin_banana_costs_l102_102413


namespace lumberjack_question_l102_102216

def logs_per_tree (total_firewood : ℕ) (firewood_per_log : ℕ) (trees_chopped : ℕ) : ℕ :=
  total_firewood / firewood_per_log / trees_chopped

theorem lumberjack_question : logs_per_tree 500 5 25 = 4 := by
  sorry

end lumberjack_question_l102_102216


namespace running_current_each_unit_l102_102687

theorem running_current_each_unit (I : ℝ) (h1 : ∀i, i = 2 * I) (h2 : ∀i, i * 3 = 6 * I) (h3 : 6 * I = 240) : I = 40 :=
by
  sorry

end running_current_each_unit_l102_102687


namespace percentage_of_first_solution_l102_102103

theorem percentage_of_first_solution (P : ℕ) 
  (h1 : 28 * P / 100 + 12 * 80 / 100 = 40 * 45 / 100) : 
  P = 30 :=
sorry

end percentage_of_first_solution_l102_102103


namespace calculator_transform_implication_l102_102885

noncomputable def transform (x n S : ℕ) : Prop :=
  (S > x^n + 1)

theorem calculator_transform_implication (x n S : ℕ) (hx : 0 < x) (hn : 0 < n) (hS : 0 < S) 
  (h_transform: transform x n S) : S > x^n + x - 1 := by
  sorry

end calculator_transform_implication_l102_102885


namespace correct_pronoun_possessive_l102_102304

theorem correct_pronoun_possessive : 
  (∃ (pronoun : String), 
    pronoun = "whose" ∧ 
    pronoun = "whose" ∨ pronoun = "who" ∨ pronoun = "that" ∨ pronoun = "which") := 
by
  -- the proof would go here
  sorry

end correct_pronoun_possessive_l102_102304


namespace polynomial_abs_sum_roots_l102_102457

theorem polynomial_abs_sum_roots (p q r m : ℤ) (h1 : p + q + r = 0) (h2 : p * q + q * r + r * p = -2500) (h3 : p * q * r = -m) :
  |p| + |q| + |r| = 100 :=
sorry

end polynomial_abs_sum_roots_l102_102457


namespace fraction_B_compared_to_A_and_C_l102_102541

theorem fraction_B_compared_to_A_and_C
    (A B C : ℕ) 
    (h1 : A = (B + C) / 3) 
    (h2 : A = B + 35) 
    (h3 : A + B + C = 1260) : 
    (∃ x : ℚ, B = x * (A + C) ∧ x = 2 / 7) :=
by
  sorry

end fraction_B_compared_to_A_and_C_l102_102541


namespace det_abs_eq_one_l102_102744

variable {n : ℕ}
variable {A : Matrix (Fin n) (Fin n) ℤ}
variable {p q r : ℕ}
variable (hpq : p^2 = q^2 + r^2)
variable (hodd : Odd r)
variable (hA : p^2 • A ^ p^2 = q^2 • A ^ q^2 + r^2 • 1)

theorem det_abs_eq_one : |A.det| = 1 := by
  sorry

end det_abs_eq_one_l102_102744


namespace possible_integer_radii_l102_102863

theorem possible_integer_radii (r : ℕ) (h : r < 140) : 
  (3 * 2 * r * π = 2 * 140 * π) → ∃ rs : Finset ℕ, rs.card = 10 := by
  sorry

end possible_integer_radii_l102_102863


namespace unique_prime_range_start_l102_102662

theorem unique_prime_range_start (N : ℕ) (hN : N = 220) (h1 : ∀ n, N ≥ n → n ≥ 211 → ¬Prime n) (h2 : Prime 211) : N - 8 = 212 :=
by
  sorry

end unique_prime_range_start_l102_102662


namespace john_pays_percentage_of_srp_l102_102828

theorem john_pays_percentage_of_srp (P MP : ℝ) (h1 : P = 1.20 * MP) (h2 : MP > 0): 
  (0.60 * MP / P) * 100 = 50 :=
by
  sorry

end john_pays_percentage_of_srp_l102_102828


namespace boat_speed_in_still_water_l102_102245

theorem boat_speed_in_still_water (b s : ℝ) (h1 : b + s = 11) (h2 : b - s = 5) : b = 8 :=
by 
  sorry

end boat_speed_in_still_water_l102_102245


namespace egg_roll_ratio_l102_102981

-- Define the conditions as hypotheses 
variables (Matthew_eats Patrick_eats Alvin_eats : ℕ)

-- Define the specific conditions
def conditions : Prop :=
  (Matthew_eats = 6) ∧
  (Patrick_eats = Alvin_eats / 2) ∧
  (Alvin_eats = 4)

-- Define the ratio of Matthew's egg rolls to Patrick's egg rolls
def ratio (a b : ℕ) := a / b

-- State the theorem with the corresponding proof problem
theorem egg_roll_ratio : conditions Matthew_eats Patrick_eats Alvin_eats → ratio Matthew_eats Patrick_eats = 3 :=
by
  -- Proof is not required as mentioned. Adding sorry to skip the proof.
  sorry

end egg_roll_ratio_l102_102981


namespace tax_percentage_excess_income_l102_102159

theorem tax_percentage_excess_income :
  ∀ (rate : ℝ) (total_tax income : ℝ), 
  rate = 0.15 →
  total_tax = 8000 →
  income = 50000 →
  (total_tax - income * rate) / (income - 40000) = 0.2 :=
by
  intros rate total_tax income hrate htotal hincome
  -- proof omitted
  sorry

end tax_percentage_excess_income_l102_102159


namespace number_composite_l102_102256

theorem number_composite : ∃ a1 a2 : ℕ, a1 > 1 ∧ a2 > 1 ∧ 2^17 + 2^5 - 1 = a1 * a2 := 
by
  sorry

end number_composite_l102_102256


namespace probability_of_at_most_3_heads_l102_102929

-- Definitions based on conditions
def binom : ℕ → ℕ → ℕ
| 0, 0       => 1
| 0, (k + 1) => 0
| (n + 1), 0 => 1
| (n + 1), (k + 1) => binom n k + binom n (k + 1)

def favorable_outcomes : ℕ :=
binom 10 0 + binom 10 1 + binom 10 2 + binom 10 3

def total_outcomes : ℕ := 2 ^ 10

def probability (favorable : ℕ) (total : ℕ) : ℚ :=
favorable / total

-- Theorem statement
theorem probability_of_at_most_3_heads : 
  probability favorable_outcomes total_outcomes = 11 / 64 :=
sorry

end probability_of_at_most_3_heads_l102_102929


namespace largest_angle_in_triangle_l102_102087

theorem largest_angle_in_triangle 
  (A B C : ℝ)
  (h_sum_angles: 2 * A + 20 = 105)
  (h_triangle_sum: A + (A + 20) + C = 180)
  (h_A_ge_0: A ≥ 0)
  (h_B_ge_0: B ≥ 0)
  (h_C_ge_0: C ≥ 0) : 
  max A (max (A + 20) C) = 75 := 
by
  -- Placeholder proof
  sorry

end largest_angle_in_triangle_l102_102087


namespace stream_speed_l102_102177

variable (B S : ℝ)

def downstream_eq : Prop := B + S = 13
def upstream_eq : Prop := B - S = 5

theorem stream_speed (h1 : downstream_eq B S) (h2 : upstream_eq B S) : S = 4 :=
by
  sorry

end stream_speed_l102_102177


namespace find_y_value_l102_102370

theorem find_y_value (y : ℝ) (h : 12^2 * y^3 / 432 = 72) : y = 6 :=
by
  sorry

end find_y_value_l102_102370


namespace exists_language_spoken_by_at_least_three_l102_102133

noncomputable def smallestValue_n (k : ℕ) : ℕ :=
  2 * k + 3

theorem exists_language_spoken_by_at_least_three (k n : ℕ) (P : Fin n → Set ℕ) (K : ℕ → ℕ) :
  (n = smallestValue_n k) →
  (∀ i, (K i) ≤ k) →
  (∀ (x y z : Fin n), ∃ l, l ∈ P x ∧ l ∈ P y ∧ l ∈ P z ∨ l ∈ P y ∧ l ∈ P z ∨ l ∈ P z ∧ l ∈ P x ∨ l ∈ P x ∧ l ∈ P y) →
  ∃ l, ∃ (a b c : Fin n), l ∈ P a ∧ l ∈ P b ∧ l ∈ P c :=
by
  intros h1 h2 h3
  sorry

end exists_language_spoken_by_at_least_three_l102_102133


namespace average_of_first_12_is_14_l102_102734

-- Definitions based on given conditions
def average_of_25 := 19
def sum_of_25 := average_of_25 * 25

def average_of_last_12 := 17
def sum_of_last_12 := average_of_last_12 * 12

def result_13 := 103

-- Main proof statement to be checked
theorem average_of_first_12_is_14 (A : ℝ) (h1 : sum_of_25 = sum_of_25) (h2 : sum_of_last_12 = sum_of_last_12) (h3 : result_13 = 103) :
  (A * 12 + result_13 + sum_of_last_12 = sum_of_25) → (A = 14) :=
by
  sorry

end average_of_first_12_is_14_l102_102734


namespace geometric_progression_sum_l102_102801

theorem geometric_progression_sum (a q : ℝ) :
  (a + a * q^2 + a * q^4 = 63) →
  (a * q + a * q^3 = 30) →
  (a = 3 ∧ q = 2) ∨ (a = 48 ∧ q = 1 / 2) :=
by
  intro h1 h2
  sorry

end geometric_progression_sum_l102_102801


namespace total_pieces_of_junk_mail_l102_102464

def houses : ℕ := 6
def pieces_per_house : ℕ := 4

theorem total_pieces_of_junk_mail : houses * pieces_per_house = 24 :=
by 
  sorry

end total_pieces_of_junk_mail_l102_102464


namespace Sergey_full_years_l102_102935

def full_years (years months weeks days hours : ℕ) : ℕ :=
  years + months / 12 + (weeks * 7 + days) / 365

theorem Sergey_full_years 
  (years : ℕ)
  (months : ℕ)
  (weeks : ℕ)
  (days : ℕ)
  (hours : ℕ) :
  years = 36 →
  months = 36 →
  weeks = 36 →
  days = 36 →
  hours = 36 →
  full_years years months weeks days hours = 39 :=
by
  intros
  sorry

end Sergey_full_years_l102_102935


namespace business_transaction_loss_l102_102547

theorem business_transaction_loss (cost_price : ℝ) (final_price : ℝ) (markup_percent : ℝ) (reduction_percent : ℝ) : 
  (final_price = 96) ∧ (markup_percent = 0.2) ∧ (reduction_percent = 0.2) ∧ (cost_price * (1 + markup_percent) * (1 - reduction_percent) = final_price) → 
  (cost_price - final_price = -4) :=
by
sorry

end business_transaction_loss_l102_102547


namespace least_number_to_subtract_l102_102420

-- Define the problem and prove that this number, when subtracted, makes the original number divisible by 127.
theorem least_number_to_subtract (n : ℕ) (h₁ : n = 100203) (h₂ : 127 > 0) : 
  ∃ k : ℕ, (100203 - 72) = 127 * k :=
by
  sorry

end least_number_to_subtract_l102_102420


namespace sum_n_k_l102_102392

theorem sum_n_k (n k : ℕ) (h₁ : (x+1)^n = 2 * x^k + 3 * x^(k+1) + 4 * x^(k+2)) (h₂ : 3 * k + 3 = 2 * n - 2 * k)
  (h₃ : 4 * k + 8 = 3 * n - 3 * k - 3) : n + k = 47 := 
sorry

end sum_n_k_l102_102392


namespace sequence_term_sequence_sum_l102_102775

def a_seq (n : ℕ) : ℕ :=
  if n = 1 then 1 else 3^(n-1)

def S_n (n : ℕ) : ℕ :=
  (3^n - 1) / 2

theorem sequence_term (n : ℕ) (h : n ≥ 1) :
  a_seq n = 3^(n-1) :=
sorry

theorem sequence_sum (n : ℕ) :
  S_n n = (3^n - 1) / 2 :=
sorry

end sequence_term_sequence_sum_l102_102775


namespace find_f_l102_102290

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f (x : ℝ) :
  (∀ t : ℝ, t = (1 - x) / (1 + x) → f t = (1 - x^2) / (1 + x^2)) →
  f x = (2 * x) / (1 + x^2) :=
by
  intros h
  specialize h ((1 - x) / (1 + x))
  specialize h rfl
  exact sorry

end find_f_l102_102290


namespace trains_cross_time_l102_102486

noncomputable def timeToCross (length1 length2 speed1 speed2 : ℝ) : ℝ :=
  let speed1_mps := speed1 * (5 / 18)
  let speed2_mps := speed2 * (5 / 18)
  let relative_speed := speed1_mps + speed2_mps
  let total_length := length1 + length2
  total_length / relative_speed

theorem trains_cross_time
  (length1 length2 : ℝ)
  (speed1 speed2 : ℝ)
  (h_length1 : length1 = 250)
  (h_length2 : length2 = 250)
  (h_speed1 : speed1 = 90)
  (h_speed2 : speed2 = 110) :
  timeToCross length1 length2 speed1 speed2 = 9 := 
by sorry

end trains_cross_time_l102_102486


namespace evaluate_expression_l102_102415

variables {a b c d e : ℝ}

theorem evaluate_expression (a b c d e : ℝ) : a * b^c - d + e = a * (b^c - (d + e)) :=
by
  sorry

end evaluate_expression_l102_102415


namespace min_value_problem_l102_102606

theorem min_value_problem (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_sum : a + 3 * b = 1) :
    (1 / a) + (3 / b) ≥ 16 :=
sorry

end min_value_problem_l102_102606


namespace polygon_side_possibilities_l102_102871

theorem polygon_side_possibilities (n : ℕ) (h : (n-2) * 180 = 1620) :
  n = 10 ∨ n = 11 ∨ n = 12 :=
by
  sorry

end polygon_side_possibilities_l102_102871


namespace number_of_children_l102_102363

theorem number_of_children (C B : ℕ) (h1 : B = 2 * C) (h2 : B = 4 * (C - 390)) : C = 780 :=
by
  sorry

end number_of_children_l102_102363


namespace sum_of_x_y_possible_values_l102_102483

theorem sum_of_x_y_possible_values (x y : ℝ) (h : x^3 + 21 * x * y + y^3 = 343) : x + y = 7 ∨ x + y = -14 := 
sorry

end sum_of_x_y_possible_values_l102_102483


namespace first_half_speed_l102_102050

noncomputable def speed_first_half : ℝ := 21

theorem first_half_speed (total_distance first_half_distance second_half_distance second_half_speed total_time : ℝ)
  (h1 : total_distance = 224)
  (h2 : first_half_distance = total_distance / 2)
  (h3 : second_half_distance = total_distance / 2)
  (h4 : second_half_speed = 24)
  (h5 : total_time = 10)
  (h6 : total_time = first_half_distance / speed_first_half + second_half_distance / second_half_speed) :
  speed_first_half = 21 :=
sorry

end first_half_speed_l102_102050


namespace percentage_relationship_l102_102755

theorem percentage_relationship (a b : ℝ) (h : a = 1.2 * b) : ¬ (b = 0.8 * a) :=
by
  -- assumption: a = 1.2 * b
  -- goal: ¬ (b = 0.8 * a)
  sorry

end percentage_relationship_l102_102755


namespace pairs_sold_l102_102535

-- Define the given conditions
def initial_large_pairs : ℕ := 22
def initial_medium_pairs : ℕ := 50
def initial_small_pairs : ℕ := 24
def pairs_left : ℕ := 13

-- Translate to the equivalent proof problem
theorem pairs_sold : (initial_large_pairs + initial_medium_pairs + initial_small_pairs) - pairs_left = 83 := by
  sorry

end pairs_sold_l102_102535


namespace find_m_if_f_even_l102_102726

theorem find_m_if_f_even (m : ℝ) (f : ℝ → ℝ) : 
  (∀ x : ℝ, f x = x^4 + (m - 1) * x + 1) ∧ (∀ x : ℝ, f x = f (-x)) → m = 1 := 
by 
  sorry

end find_m_if_f_even_l102_102726


namespace remainder_of_M_div_by_51_is_zero_l102_102987

open Nat

noncomputable def M := 1234567891011121314151617181920212223242526272829303132333435363738394041424344454647484950

theorem remainder_of_M_div_by_51_is_zero :
  M % 51 = 0 :=
sorry

end remainder_of_M_div_by_51_is_zero_l102_102987


namespace value_op_and_add_10_l102_102264

def op_and (a b : ℤ) : ℤ := (a + b) * (a - b)

theorem value_op_and_add_10 : op_and 8 5 + 10 = 49 :=
by
  sorry

end value_op_and_add_10_l102_102264


namespace intersection_m_zero_range_of_m_l102_102886

def A (x : ℝ) : Prop := x^2 - 2 * x - 3 < 0
def B (x : ℝ) (m : ℝ) : Prop := (x - m + 1) * (x - m - 1) ≥ 0

theorem intersection_m_zero : 
  ∀ x : ℝ, A x → B x 0 ↔ (1 ≤ x ∧ x < 3) :=
sorry

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, A x → B x m) ∧ (∃ x : ℝ, B x m ∧ ¬A x) → (m ≤ -2 ∨ m ≥ 4) :=
sorry

end intersection_m_zero_range_of_m_l102_102886


namespace central_angle_measure_l102_102395

-- Given the problem definitions
variables (A : ℝ) (x : ℝ)

-- Condition: The probability of landing in the region is 1/8
def probability_condition : Prop :=
  (1 / 8 : ℝ) = (x / 360)

-- The final theorem to prove
theorem central_angle_measure (h : probability_condition x) : x = 45 := 
  sorry

end central_angle_measure_l102_102395


namespace ninas_money_l102_102030

theorem ninas_money (C M : ℝ) (h1 : 6 * C = M) (h2 : 8 * (C - 1.15) = M) : M = 27.6 := 
by
  sorry

end ninas_money_l102_102030


namespace total_cost_correct_l102_102891

-- Define the individual costs and quantities
def pumpkin_cost : ℝ := 2.50
def tomato_cost : ℝ := 1.50
def chili_pepper_cost : ℝ := 0.90

def pumpkin_quantity : ℕ := 3
def tomato_quantity : ℕ := 4
def chili_pepper_quantity : ℕ := 5

-- Define the total cost calculation
def total_cost : ℝ :=
  pumpkin_quantity * pumpkin_cost +
  tomato_quantity * tomato_cost +
  chili_pepper_quantity * chili_pepper_cost

-- Prove the total cost is $18.00
theorem total_cost_correct : total_cost = 18.00 := by
  sorry

end total_cost_correct_l102_102891


namespace scientific_notation_of_1206_million_l102_102465

theorem scientific_notation_of_1206_million :
  (1206 * 10^6 : ℝ) = 1.206 * 10^7 :=
by
  sorry

end scientific_notation_of_1206_million_l102_102465


namespace unique_acute_triangulation_l102_102623

-- Definitions for the proof problem
def is_convex (polygon : Type) : Prop := sorry
def is_acute_triangle (triangle : Type) : Prop := sorry
def is_triangulation (polygon : Type) (triangulation : List Type) : Prop := sorry
def is_acute_triangulation (polygon : Type) (triangulation : List Type) : Prop :=
  is_triangulation polygon triangulation ∧ ∀ triangle ∈ triangulation, is_acute_triangle triangle

-- Proposition to be proved
theorem unique_acute_triangulation (n : ℕ) (polygon : Type) 
  (h₁ : is_convex polygon) (h₂ : n ≥ 3) :
  ∃! triangulation : List Type, is_acute_triangulation polygon triangulation := 
sorry

end unique_acute_triangulation_l102_102623


namespace sum_of_cubes_l102_102187

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 11) (h2 : x * y = 12) : x^3 + y^3 = 935 := by
  sorry

end sum_of_cubes_l102_102187


namespace total_rent_paid_l102_102837

theorem total_rent_paid
  (weekly_rent : ℕ) (num_weeks : ℕ) 
  (hrent : weekly_rent = 388)
  (hweeks : num_weeks = 1359) :
  weekly_rent * num_weeks = 527292 := 
by
  sorry

end total_rent_paid_l102_102837


namespace division_and_multiplication_l102_102333

theorem division_and_multiplication (dividend divisor quotient factor product : ℕ) 
  (h₁ : dividend = 24) 
  (h₂ : divisor = 3) 
  (h₃ : quotient = dividend / divisor) 
  (h₄ : factor = 5) 
  (h₅ : product = quotient * factor) : 
  quotient = 8 ∧ product = 40 := 
by 
  sorry

end division_and_multiplication_l102_102333


namespace equivalent_proof_problem_l102_102865

def option_A : ℚ := 14 / 10
def option_B : ℚ := 1 + 2 / 5
def option_C : ℚ := 1 + 6 / 15
def option_D : ℚ := 1 + 3 / 8
def option_E : ℚ := 1 + 28 / 20
def target : ℚ := 7 / 5

theorem equivalent_proof_problem : option_D ≠ target :=
by {
  sorry
}

end equivalent_proof_problem_l102_102865


namespace remainder_of_504_divided_by_100_is_4_l102_102356

theorem remainder_of_504_divided_by_100_is_4 :
  (504 % 100) = 4 :=
by
  sorry

end remainder_of_504_divided_by_100_is_4_l102_102356


namespace nguyen_fabric_yards_l102_102085

open Nat

theorem nguyen_fabric_yards :
  let fabric_per_pair := 8.5
  let pairs_needed := 7
  let fabric_still_needed := 49
  let total_fabric_needed := pairs_needed * fabric_per_pair
  let fabric_already_have := total_fabric_needed - fabric_still_needed
  let yards_of_fabric := fabric_already_have / 3
  yards_of_fabric = 3.5 := by
    sorry

end nguyen_fabric_yards_l102_102085


namespace solve_inequality_system_l102_102585

theorem solve_inequality_system (x : ℝ) (h1 : x - 2 ≤ 0) (h2 : (x - 1) / 2 < x) : -1 < x ∧ x ≤ 2 := 
sorry

end solve_inequality_system_l102_102585


namespace max_value_g_l102_102663

def g (x : ℝ) : ℝ := 4 * x - x ^ 4

theorem max_value_g : ∃ x : ℝ, (0 ≤ x ∧ x ≤ 2 ∧ ∀ y : ℝ, (0 ≤ y ∧ y ≤ 2) → g y ≤ g x) ∧ g x = 3 :=
by
  sorry

end max_value_g_l102_102663


namespace joint_probability_l102_102839

noncomputable def P (A B : Prop) : ℝ := sorry
def A : Prop := sorry
def B : Prop := sorry

axiom prob_A : P A true = 0.005
axiom prob_B_given_A : P B true = 0.99

theorem joint_probability :
  P A B = 0.00495 :=
by sorry

end joint_probability_l102_102839


namespace fraction_of_boys_participated_l102_102799

-- Definitions based on given conditions
def total_students (B G : ℕ) : Prop := B + G = 800
def participating_girls (G : ℕ) : Prop := (3 / 4 : ℚ) * G = 150
def total_participants (P : ℕ) : Prop := P = 550
def participating_girls_count (PG : ℕ) : Prop := PG = 150

-- Definition of the fraction of participating boys
def fraction_participating_boys (X : ℚ) (B : ℕ) (PB : ℕ) : Prop := X * B = PB

-- The problem of proving the fraction of boys who participated
theorem fraction_of_boys_participated (B G PB : ℕ) (X : ℚ)
  (h1 : total_students B G)
  (h2 : participating_girls G)
  (h3 : total_participants 550)
  (h4 : participating_girls_count 150)
  (h5 : PB = 550 - 150) :
  fraction_participating_boys X B PB → X = 2 / 3 := by
  sorry

end fraction_of_boys_participated_l102_102799


namespace Rudolph_stop_signs_l102_102198

def distance : ℕ := 5 + 2
def stopSignsPerMile : ℕ := 2
def totalStopSigns : ℕ := distance * stopSignsPerMile

theorem Rudolph_stop_signs :
  totalStopSigns = 14 := 
  by sorry

end Rudolph_stop_signs_l102_102198


namespace find_b1_b7_b10_value_l102_102023

open Classical

theorem find_b1_b7_b10_value
  (a : ℕ → ℝ)
  (b : ℕ → ℝ)
  (h_arith_seq : ∀ n m : ℕ, a n + a m = 2 * a ((n + m) / 2))
  (h_geom_seq : ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r)
  (a3_condition : a 3 - 2 * (a 6)^2 + 3 * a 7 = 0)
  (b6_a6_eq : b 6 = a 6)
  (non_zero_seq : ∀ n : ℕ, a n ≠ 0) :
  b 1 * b 7 * b 10 = 8 := 
by 
  sorry

end find_b1_b7_b10_value_l102_102023


namespace minimum_boxes_cost_300_muffins_l102_102247

theorem minimum_boxes_cost_300_muffins :
  ∃ (L_used M_used S_used : ℕ), 
    L_used + M_used + S_used = 28 ∧ 
    (L_used = 10 ∧ M_used = 15 ∧ S_used = 3) ∧ 
    (L_used * 15 + M_used * 9 + S_used * 5 = 300) ∧ 
    (L_used * 5 + M_used * 3 + S_used * 2 = 101) ∧ 
    (L_used ≤ 10 ∧ M_used ≤ 15 ∧ S_used ≤ 25) :=
by
  -- The proof is omitted (theorem statement only).
  sorry

end minimum_boxes_cost_300_muffins_l102_102247


namespace mower_next_tangent_point_l102_102819

theorem mower_next_tangent_point (r_garden r_mower : ℝ) (h_garden : r_garden = 15) (h_mower : r_mower = 5) :
    ∃ θ : ℝ, θ = (2 * π * r_mower / (2 * π * r_garden)) * 360 ∧ θ = 120 :=
sorry

end mower_next_tangent_point_l102_102819


namespace problem_prove_ω_and_delta_l102_102618

noncomputable def f (ω φ x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

theorem problem_prove_ω_and_delta (ω φ : ℝ) (h_ω : ω > 0) (h_φ : abs φ < π / 2) 
    (h_sym_axis : ∀ x, f ω φ x = f ω φ (-(x + π))) 
    (h_center_sym : ∃ c : ℝ, (c = π / 2) ∧ (f ω φ c = 0)) 
    (h_monotone_increasing : ∀ x, -π ≤ x ∧ x ≤ -π / 2 → f ω φ x < f ω φ (x + 1)) :
    (ω = 1 / 3) ∧ (∀ δ : ℝ, (∀ x : ℝ, f ω φ (x + δ) = f ω φ (-x + δ)) → ∃ k : ℤ, δ = 2 * π + 3 * k * π) :=
by
  sorry

end problem_prove_ω_and_delta_l102_102618


namespace equal_roots_of_quadratic_l102_102440

theorem equal_roots_of_quadratic (k : ℝ) : 
  ( ∀ x : ℝ, 2 * k * x^2 + 7 * k * x + 2 = 0 → x = x ) ↔ k = 16 / 49 :=
by
  sorry

end equal_roots_of_quadratic_l102_102440


namespace spencer_walked_distance_l102_102653

/-- Define the distances involved -/
def total_distance := 0.8
def library_to_post_office := 0.1
def post_office_to_home := 0.4

/-- Define the distance from house to library as a variable to calculate -/
def house_to_library := total_distance - library_to_post_office - post_office_to_home

/-- The theorem states that Spencer walked 0.3 miles from his house to the library -/
theorem spencer_walked_distance : 
  house_to_library = 0.3 :=
by
  -- Proof omitted
  sorry

end spencer_walked_distance_l102_102653


namespace certain_number_minus_15_l102_102482

theorem certain_number_minus_15 (n : ℕ) (h : n / 10 = 6) : n - 15 = 45 :=
sorry

end certain_number_minus_15_l102_102482


namespace find_age_l102_102406

theorem find_age (A : ℤ) (h : 4 * (A + 4) - 4 * (A - 4) = A) : A = 32 :=
by sorry

end find_age_l102_102406


namespace average_speed_monkey_l102_102740

def monkeyDistance : ℝ := 2160
def monkeyTimeMinutes : ℝ := 30
def monkeyTimeSeconds : ℝ := monkeyTimeMinutes * 60

theorem average_speed_monkey :
  (monkeyDistance / monkeyTimeSeconds) = 1.2 := 
sorry

end average_speed_monkey_l102_102740


namespace min_varphi_symmetry_l102_102681

theorem min_varphi_symmetry (ϕ : ℝ) (hϕ : ϕ > 0) :
  (∃ k : ℤ, ϕ = (4 * Real.pi) / 3 - k * Real.pi ∧ ϕ > 0 ∧ (∀ x : ℝ, Real.cos (x - ϕ + (4 * Real.pi) / 3) = Real.cos (-x - ϕ + (4 * Real.pi) / 3))) 
  → ϕ = Real.pi / 3 :=
sorry

end min_varphi_symmetry_l102_102681


namespace total_prayers_in_a_week_l102_102995

def prayers_per_week (pastor_prayers : ℕ → ℕ) : ℕ :=
  (pastor_prayers 0) + (pastor_prayers 1) + (pastor_prayers 2) +
  (pastor_prayers 3) + (pastor_prayers 4) + (pastor_prayers 5) + (pastor_prayers 6)

def pastor_paul (day : ℕ) : ℕ :=
  if day = 6 then 40 else 20

def pastor_bruce (day : ℕ) : ℕ :=
  if day = 6 then 80 else 10

def pastor_caroline (day : ℕ) : ℕ :=
  if day = 6 then 30 else 10

theorem total_prayers_in_a_week :
  prayers_per_week pastor_paul + prayers_per_week pastor_bruce + prayers_per_week pastor_caroline = 390 :=
sorry

end total_prayers_in_a_week_l102_102995


namespace new_sequence_after_removal_is_geometric_l102_102750

theorem new_sequence_after_removal_is_geometric (a : ℕ → ℝ) (a₁ q : ℝ) (k : ℕ)
  (h_geo : ∀ n, a n = a₁ * q ^ n) :
  ∀ n, (a (n + k)) = a₁ * q ^ (n + k) :=
by
  sorry

end new_sequence_after_removal_is_geometric_l102_102750


namespace all_numbers_positive_l102_102993

noncomputable def condition (a : Fin 9 → ℝ) : Prop :=
  ∀ (S : Finset (Fin 9)), S.card = 4 → S.sum (a : Fin 9 → ℝ) < (Finset.univ \ S).sum (a : Fin 9 → ℝ)

theorem all_numbers_positive (a : Fin 9 → ℝ) (h : condition a) : ∀ i, 0 < a i :=
by
  sorry

end all_numbers_positive_l102_102993


namespace sum_divisible_by_49_l102_102725

theorem sum_divisible_by_49
  {x y z : ℤ} 
  (hx : x % 7 ≠ 0)
  (hy : y % 7 ≠ 0)
  (hz : z % 7 ≠ 0)
  (h : 7 ^ 3 ∣ (x ^ 7 + y ^ 7 + z ^ 7)) : 7^2 ∣ (x + y + z) :=
by
  sorry

end sum_divisible_by_49_l102_102725


namespace rationalize_denominator_l102_102656

theorem rationalize_denominator : (1 / (Real.sqrt 3 + 1) = (Real.sqrt 3 - 1) / 2) :=
by
  sorry

end rationalize_denominator_l102_102656


namespace total_balls_in_box_l102_102389

theorem total_balls_in_box :
  ∀ (W B R : ℕ), 
    W = 16 →
    B = W + 12 →
    R = 2 * B →
    W + B + R = 100 :=
by
  intros W B R hW hB hR
  sorry

end total_balls_in_box_l102_102389


namespace tens_digit_seven_last_digit_six_l102_102072

theorem tens_digit_seven_last_digit_six (n : ℕ) (h : ((n * n) / 10) % 10 = 7) :
  (n * n) % 10 = 6 :=
sorry

end tens_digit_seven_last_digit_six_l102_102072


namespace circle_value_l102_102417

theorem circle_value (c d s : ℝ) 
  (h1 : ∀ x y : ℝ, x^2 - 8*x + y^2 + 16*y = -100 ↔ (x - c)^2 + (y + d)^2 = s^2)
  (h2 : c = 4)
  (h3 : d = -8)
  (h4 : s = 2 * Real.sqrt 5) :
  c + d + s = -4 + 2 * Real.sqrt 5 := 
sorry

end circle_value_l102_102417


namespace sufficient_but_not_necessary_condition_for_q_l102_102650

theorem sufficient_but_not_necessary_condition_for_q (k : ℝ) :
  (∀ x : ℝ, x ≥ k → x^2 - x > 2) ∧ (∃ x : ℝ, x < k ∧ x^2 - x > 2) ↔ k > 2 :=
sorry

end sufficient_but_not_necessary_condition_for_q_l102_102650


namespace restaurant_table_difference_l102_102143

theorem restaurant_table_difference :
  ∃ (N O : ℕ), N + O = 40 ∧ 6 * N + 4 * O = 212 ∧ (N - O) = 12 :=
by
  sorry

end restaurant_table_difference_l102_102143


namespace candy_sampling_l102_102276

theorem candy_sampling (total_customers caught_sampling not_caught_sampling : ℝ) :
  caught_sampling = 0.22 * total_customers →
  not_caught_sampling = 0.12 * (total_customers * sampling_percent) →
  (sampling_percent * total_customers = caught_sampling / 0.78) :=
by
  intros h1 h2
  sorry

end candy_sampling_l102_102276


namespace nine_x_plus_twenty_seven_y_l102_102378

theorem nine_x_plus_twenty_seven_y (x y : ℤ) (h : 17 * x + 51 * y = 102) : 9 * x + 27 * y = 54 := 
by sorry

end nine_x_plus_twenty_seven_y_l102_102378


namespace percentage_of_students_wearing_blue_shirts_l102_102120

theorem percentage_of_students_wearing_blue_shirts :
  ∀ (total_students red_percent green_percent students_other_colors : ℕ),
  total_students = 800 →
  red_percent = 23 →
  green_percent = 15 →
  students_other_colors = 136 →
  ((total_students - students_other_colors) - (red_percent + green_percent) = 45) :=
by
  intros total_students red_percent green_percent students_other_colors h_total h_red h_green h_other
  have h_other_percent : (students_other_colors * 100 / total_students) = 17 :=
    sorry
  exact sorry

end percentage_of_students_wearing_blue_shirts_l102_102120


namespace gilled_mushrooms_count_l102_102315

def mushrooms_problem (G S : ℕ) : Prop :=
  (S = 9 * G) ∧ (G + S = 30) → (G = 3)

-- The theorem statement corresponding to the problem
theorem gilled_mushrooms_count (G S : ℕ) : mushrooms_problem G S :=
by {
  sorry
}

end gilled_mushrooms_count_l102_102315


namespace simplify_expression_l102_102762

theorem simplify_expression (b : ℝ) : (1 * 3 * b * 4 * b^2 * 5 * b^3 * 6 * b^4) = 360 * b^10 :=
by sorry

end simplify_expression_l102_102762


namespace gcd_8251_6105_l102_102900

theorem gcd_8251_6105 : Nat.gcd 8251 6105 = 37 := by
  sorry

end gcd_8251_6105_l102_102900


namespace percentage_deposit_l102_102577

theorem percentage_deposit (deposited : ℝ) (initial_amount : ℝ) (amount_deposited : ℝ) (P : ℝ) 
  (h1 : deposited = 750) 
  (h2 : initial_amount = 50000)
  (h3 : amount_deposited = 0.20 * (P / 100) * (0.25 * initial_amount))
  (h4 : amount_deposited = deposited) : 
  P = 30 := 
sorry

end percentage_deposit_l102_102577


namespace fathers_age_more_than_three_times_son_l102_102307

variable (F S x : ℝ)

theorem fathers_age_more_than_three_times_son :
  F = 27 →
  F = 3 * S + x →
  F + 3 = 2 * (S + 3) + 8 →
  x = 3 :=
by
  intros hF h1 h2
  sorry

end fathers_age_more_than_three_times_son_l102_102307


namespace square_of_binomial_l102_102057

theorem square_of_binomial (c : ℝ) (h : c = 3600) :
  ∃ a : ℝ, (x : ℝ) → (x + a)^2 = x^2 + 120 * x + c := by
  sorry

end square_of_binomial_l102_102057


namespace number_of_zeros_in_factorial_30_l102_102862

theorem number_of_zeros_in_factorial_30 :
  let count_factors (n k : Nat) : Nat := n / k
  count_factors 30 5 + count_factors 30 25 = 7 :=
by
  let count_factors (n k : Nat) : Nat := n / k
  sorry

end number_of_zeros_in_factorial_30_l102_102862


namespace distinct_lines_through_point_and_parabola_l102_102997

noncomputable def num_distinct_lines : ℕ :=
  let num_divisors (n : ℕ) : ℕ :=
    have factors := [2^5, 3^2, 7]
    factors.foldl (fun acc f => acc * (f + 1)) 1
  (num_divisors 2016) / 2 -- as each pair (x_1, x_2) corresponds twice

theorem distinct_lines_through_point_and_parabola :
  num_distinct_lines = 36 :=
by
  sorry

end distinct_lines_through_point_and_parabola_l102_102997


namespace pressure_force_correct_l102_102379

-- Define the conditions
noncomputable def base_length : ℝ := 4
noncomputable def vertex_depth : ℝ := 4
noncomputable def gamma : ℝ := 1000 -- density of water in kg/m^3
noncomputable def g : ℝ := 9.81 -- acceleration due to gravity in m/s^2

-- Define the calculation of the pressure force on the parabolic segment
noncomputable def pressure_force (base_length vertex_depth gamma g : ℝ) : ℝ :=
  19620 * (4 * ((2/3) * (4 : ℝ)^(3/2)) - ((2/5) * (4 : ℝ)^(5/2)))

-- State the theorem
theorem pressure_force_correct : pressure_force base_length vertex_depth gamma g = 167424 := 
by
  sorry

end pressure_force_correct_l102_102379


namespace max_a_condition_range_a_condition_l102_102349

-- Definitions of the functions f and g
def f (x a : ℝ) : ℝ := |2 * x - a| + a
def g (x : ℝ) : ℝ := |2 * x - 1|

-- Problem (I)
theorem max_a_condition (a : ℝ) :
  (∀ x, g x ≤ 5 → f x a ≤ 6) → a ≤ 1 :=
sorry

-- Problem (II)
theorem range_a_condition (a : ℝ) :
  (∀ x, f x a + g x ≥ 3) → a ≥ 2 :=
sorry

end max_a_condition_range_a_condition_l102_102349


namespace total_number_of_people_on_bus_l102_102648

theorem total_number_of_people_on_bus (boys girls : ℕ)
    (driver assistant teacher : ℕ) 
    (h1 : boys = 50)
    (h2 : girls = boys + (2 * boys / 5))
    (h3 : driver = 1)
    (h4 : assistant = 1)
    (h5 : teacher = 1) :
    (boys + girls + driver + assistant + teacher = 123) :=
by
    sorry

end total_number_of_people_on_bus_l102_102648


namespace distinct_students_27_l102_102075

variable (students_euler : ℕ) (students_fibonacci : ℕ) (students_gauss : ℕ) (overlap_euler_fibonacci : ℕ)

-- Conditions
def conditions : Prop := 
  students_euler = 12 ∧ 
  students_fibonacci = 10 ∧ 
  students_gauss = 11 ∧ 
  overlap_euler_fibonacci = 3

-- Question and correct answer
def distinct_students (students_euler students_fibonacci students_gauss overlap_euler_fibonacci : ℕ) : ℕ :=
  (students_euler + students_fibonacci + students_gauss) - overlap_euler_fibonacci

theorem distinct_students_27 : conditions students_euler students_fibonacci students_gauss overlap_euler_fibonacci →
  distinct_students students_euler students_fibonacci students_gauss overlap_euler_fibonacci = 27 :=
by
  sorry

end distinct_students_27_l102_102075


namespace number_of_pizzas_ordered_l102_102667

-- Definitions from conditions
def slices_per_pizza : Nat := 2
def total_slices : Nat := 28

-- Proof that the number of pizzas ordered is 14
theorem number_of_pizzas_ordered : total_slices / slices_per_pizza = 14 := by
  sorry

end number_of_pizzas_ordered_l102_102667


namespace estimate_m_value_l102_102371

-- Definition of polynomial P(x) and its roots related to the problem
noncomputable def P (x : ℂ) (a b c : ℂ) : ℂ := x^3 + a * x^2 + b * x + c

-- Statement of the problem in Lean 4
theorem estimate_m_value :
  ∀ (a b c : ℕ),
  a ≤ 100 ∧ b ≤ 100 ∧ c ≤ 100 ∧
  (∃ z1 z2 z3 : ℂ, z1 ≠ z2 ∧ z1 ≠ z3 ∧ z2 ≠ z3 ∧ 
  P z1 a b c = 0 ∧ P z2 a b c = 0 ∧ P z3 a b c = 0) →
  ∃ m : ℕ, m = 8097 :=
sorry

end estimate_m_value_l102_102371


namespace find_constants_l102_102234

open Set

variable {α : Type*} [LinearOrderedField α]

def Set_1 : Set α := {x | x^2 - 3*x + 2 = 0}

def Set_2 (a : α) : Set α := {x | x^2 - a*x + (a-1) = 0}

def Set_3 (m : α) : Set α := {x | x^2 - m*x + 2 = 0}

theorem find_constants (a m : α) :
  (Set_1 ∪ Set_2 a = Set_1) ∧ (Set_1 ∩ Set_2 a = Set_3 m) → 
  a = 3 ∧ m = 3 :=
by sorry

end find_constants_l102_102234


namespace find_a_l102_102992

theorem find_a (a x : ℝ) : 
  ((x + a)^2 / (3 * x + 65) = 2) 
  ∧ (∃ x1 x2 : ℝ,  x1 ≠ x2 ∧ (x1 = x2 + 22 ∨ x2 = x1 + 22 )) 
  → a = 3 := 
sorry

end find_a_l102_102992


namespace initial_avg_weight_proof_l102_102856

open Classical

variable (A B C D E : ℝ) (W : ℝ)

-- Given conditions
def initial_avg_weight_A_B_C : Prop := W = (A + B + C) / 3
def avg_with_D : Prop := (A + B + C + D) / 4 = 80
def E_weighs_D_plus_8 : Prop := E = D + 8
def avg_with_E_replacing_A : Prop := (B + C + D + E) / 4 = 79
def weight_of_A : Prop := A = 80

-- Question to prove
theorem initial_avg_weight_proof (h1 : initial_avg_weight_A_B_C W A B C)
                                 (h2 : avg_with_D A B C D)
                                 (h3 : E_weighs_D_plus_8 D E)
                                 (h4 : avg_with_E_replacing_A B C D E)
                                 (h5 : weight_of_A A) :
  W = 84 := by
  sorry

end initial_avg_weight_proof_l102_102856


namespace pq_logic_l102_102374

theorem pq_logic (p q : Prop) (h1 : p ∨ q) (h2 : ¬p) : ¬p ∧ q :=
by
  sorry

end pq_logic_l102_102374


namespace distance_traveled_by_second_hand_l102_102798

theorem distance_traveled_by_second_hand (r : ℝ) (minutes : ℝ) (h1 : r = 10) (h2 : minutes = 45) :
  (2 * Real.pi * r) * (minutes / 1) = 900 * Real.pi := by
  -- Given:
  -- r = length of the second hand = 10 cm
  -- minutes = 45
  -- To prove: distance traveled by the tip = 900π cm
  sorry

end distance_traveled_by_second_hand_l102_102798


namespace f_2008th_derivative_at_0_l102_102296

noncomputable def f (x : ℝ) : ℝ := (Real.sin (x / 4))^6 + (Real.cos (x / 4))^6

theorem f_2008th_derivative_at_0 : (deriv^[2008] f) 0 = 3 / 8 :=
sorry

end f_2008th_derivative_at_0_l102_102296


namespace average_percentage_reduction_l102_102213

theorem average_percentage_reduction (x : ℝ) (hx : 0 < x ∧ x < 1)
  (initial_price final_price : ℝ)
  (h_initial : initial_price = 25)
  (h_final : final_price = 16)
  (h_reduction : final_price = initial_price * (1-x)^2) :
  x = 0.2 :=
by {
  --". Convert fraction \( = x / y \)", proof is omitted
  sorry
}

end average_percentage_reduction_l102_102213


namespace polynomial_proof_l102_102194

variable (a b : ℝ)

-- Define the given monomial and the resulting polynomial 
def monomial := -3 * a ^ 2 * b
def result := 6 * a ^ 3 * b ^ 2 - 3 * a ^ 2 * b ^ 2 + 9 * a ^ 2 * b

-- Define the polynomial we want to prove
def poly := -2 * a * b + b - 3

-- Statement of the problem in Lean 4
theorem polynomial_proof :
  monomial * poly = result :=
by sorry

end polynomial_proof_l102_102194


namespace teacher_age_is_56_l102_102660

theorem teacher_age_is_56 (s t : ℝ) (h1 : s = 40 * 15) (h2 : s + t = 41 * 16) : t = 56 := by
  sorry

end teacher_age_is_56_l102_102660


namespace certain_number_equation_l102_102467

theorem certain_number_equation (x : ℤ) (h : 16 * x + 17 * x + 20 * x + 11 = 170) : x = 3 :=
by {
  sorry
}

end certain_number_equation_l102_102467


namespace abs_a_plus_2_always_positive_l102_102849

theorem abs_a_plus_2_always_positive (a : ℝ) : |a| + 2 > 0 := 
sorry

end abs_a_plus_2_always_positive_l102_102849


namespace area_of_octagon_in_square_l102_102254

theorem area_of_octagon_in_square (perimeter : ℝ) (side_length : ℝ) (area_square : ℝ)
  (segment_length : ℝ) (area_triangle : ℝ) (total_area_triangles : ℝ) :
  perimeter = 144 →
  side_length = perimeter / 4 →
  segment_length = side_length / 3 →
  area_triangle = (segment_length * segment_length) / 2 →
  total_area_triangles = 4 * area_triangle →
  area_square = side_length * side_length →
  (area_square - total_area_triangles) = 1008 :=
by
  sorry

end area_of_octagon_in_square_l102_102254


namespace fraction_people_over_65_l102_102401

theorem fraction_people_over_65 (T : ℕ) (F : ℕ) : 
  (3:ℚ) / 7 * T = 24 ∧ 50 < T ∧ T < 100 → T = 56 ∧ ∃ F : ℕ, (F / 56 : ℚ) = F / (T : ℚ) :=
by 
  sorry

end fraction_people_over_65_l102_102401


namespace triangle_angle_C_l102_102661

theorem triangle_angle_C (A B C : Real) (h1 : A - B = 10) (h2 : B = A / 2) :
  C = 150 :=
by
  -- Proof goes here
  sorry

end triangle_angle_C_l102_102661


namespace VehicleB_travel_time_l102_102636

theorem VehicleB_travel_time 
    (v_A v_B : ℝ)
    (d : ℝ)
    (h1 : d = 3 * (v_A + v_B))
    (h2 : 3 * v_A = d / 2)
    (h3 : ∀ t ≤ 3.5 , d - t * v_B - 0.5 * v_A = 0)
    : d / v_B = 7.2 :=
by
  sorry

end VehicleB_travel_time_l102_102636


namespace average_apples_per_hour_l102_102153

theorem average_apples_per_hour :
  (5.0 / 3.0) = 1.67 := 
sorry

end average_apples_per_hour_l102_102153


namespace polynomial_value_l102_102530

theorem polynomial_value (x : ℝ) (hx : x^2 - 4*x + 1 = 0) : 
  x^4 - 8*x^3 + 10*x^2 - 8*x + 1 = -56 - 32*Real.sqrt 3 ∨ 
  x^4 - 8*x^3 + 10*x^2 - 8*x + 1 = -56 + 32*Real.sqrt 3 :=
sorry

end polynomial_value_l102_102530


namespace max_min_magnitude_of_sum_l102_102419

open Real

-- Define the vectors a and b and their magnitudes
variables {a b : ℝ × ℝ}
variable (h_a : ‖a‖ = 5)
variable (h_b : ‖b‖ = 2)

-- Define the constant 7 and 3 for the max and min values
noncomputable def max_magnitude : ℝ := 7
noncomputable def min_magnitude : ℝ := 3

-- State the theorem
theorem max_min_magnitude_of_sum (h_a : ‖a‖ = 5) (h_b : ‖b‖ = 2) :
  ‖a + b‖ ≤ max_magnitude ∧ ‖a + b‖ ≥ min_magnitude :=
by {
  sorry -- Proof goes here
}

end max_min_magnitude_of_sum_l102_102419


namespace second_candidate_more_marks_30_l102_102033

noncomputable def total_marks : ℝ := 600
def passing_marks_approx : ℝ := 240

def candidate_marks (percentage : ℝ) (total : ℝ) : ℝ :=
  percentage * total

def more_marks (second_candidate : ℝ) (passing : ℝ) : ℝ :=
  second_candidate - passing

theorem second_candidate_more_marks_30 :
  more_marks (candidate_marks 0.45 total_marks) passing_marks_approx = 30 := by
  sorry

end second_candidate_more_marks_30_l102_102033


namespace range_of_k_l102_102594

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x^2 - x

theorem range_of_k :
  (∀ (x₁ x₂ : ℝ), x₁ ∈ Set.Icc (-1) 1 → x₂ ∈ Set.Icc (-1) 1 → |f x₁ - f x₂| ≤ k) → k ≥ Real.exp 1 - 1 :=
by
  sorry

end range_of_k_l102_102594


namespace math_problem_l102_102076

theorem math_problem (n a b : ℕ) (hn_pos : n > 0) (h1 : 3 * n + 1 = a^2) (h2 : 5 * n - 1 = b^2) :
  (∃ x y: ℕ, 7 * n + 13 = x * y ∧ 1 < x ∧ 1 < y) ∧
  (∃ p q: ℕ, 8 * (17 * n^2 + 3 * n) = p^2 + q^2) :=
  sorry

end math_problem_l102_102076


namespace cargo_arrival_day_l102_102321

-- Definitions based on conditions
def navigation_days : Nat := 21
def customs_days : Nat := 4
def warehouse_days_from_today : Nat := 2
def departure_days_ago : Nat := 30

-- Definition represents the total transit time
def total_transit_days : Nat := navigation_days + customs_days + warehouse_days_from_today

-- Theorem to prove the cargo always arrives at the rural warehouse 1 day after leaving the port in Vancouver
theorem cargo_arrival_day : 
  (departure_days_ago - total_transit_days + warehouse_days_from_today = 1) :=
by
  -- Placeholder for the proof
  sorry

end cargo_arrival_day_l102_102321


namespace expression_simplification_l102_102286

theorem expression_simplification : 2 + 1 / (3 + 1 / (2 + 2)) = 30 / 13 := 
by 
  sorry

end expression_simplification_l102_102286


namespace find_alpha_polar_equation_l102_102449

noncomputable def alpha := (3 * Real.pi) / 4

theorem find_alpha (P : ℝ × ℝ) (l : ℝ → ℝ × ℝ) (A B : ℝ × ℝ)
  (hP : P = (2, 1))
  (hl_def : ∀ t, l t = (2 + t * Real.cos alpha, 1 + t * Real.sin alpha))
  (hA : ∃ t, l t = (A.1, 0) ∧ A.1 > 0)
  (hB : ∃ t, l t = (0, B.2) ∧ B.2 > 0)
  (h_cond : dist P A * dist P B = 4) : alpha = (3 * Real.pi) / 4 :=
sorry

theorem polar_equation (l : ℝ → ℝ × ℝ)
  (hl_def : ∀ t, l t = (2 + t * Real.cos alpha, 1 + t * Real.sin alpha))
  (h_alpha : alpha = (3 * Real.pi) / 4)
  (h_polar : ∀ ρ θ, l ρ = (ρ * Real.cos θ, ρ * Real.sin θ))
  : ∀ ρ θ, ρ * (Real.cos θ + Real.sin θ) = 3 :=
sorry

end find_alpha_polar_equation_l102_102449


namespace pool_drain_rate_l102_102053

-- Define the dimensions and other conditions
def poolLength : ℝ := 150
def poolWidth : ℝ := 40
def poolDepth : ℝ := 10
def poolCapacityPercent : ℝ := 0.80
def drainTime : ℕ := 800

-- Define the problem statement
theorem pool_drain_rate :
  let fullVolume := poolLength * poolWidth * poolDepth
  let volumeAt80Percent := fullVolume * poolCapacityPercent
  let drainRate := volumeAt80Percent / drainTime
  drainRate = 60 :=
by
  sorry

end pool_drain_rate_l102_102053


namespace Marty_combinations_l102_102042

theorem Marty_combinations : 
  let colors := 4
  let decorations := 3
  colors * decorations = 12 :=
by
  sorry

end Marty_combinations_l102_102042


namespace albert_age_l102_102095

theorem albert_age
  (A : ℕ)
  (dad_age : ℕ)
  (h1 : dad_age = 48)
  (h2 : dad_age - 4 = 4 * (A - 4)) :
  A = 15 :=
by
  sorry

end albert_age_l102_102095


namespace tan_beta_value_l102_102513

theorem tan_beta_value (α β : ℝ) (h1 : Real.tan α = -3 / 4) (h2 : Real.tan (α + β) = 1) : Real.tan β = 7 :=
sorry

end tan_beta_value_l102_102513


namespace point_in_fourth_quadrant_l102_102498

theorem point_in_fourth_quadrant (m : ℝ) : (m-1 > 0 ∧ 2-m < 0) ↔ m > 2 :=
by
  sorry

end point_in_fourth_quadrant_l102_102498


namespace quadratic_points_range_l102_102629

theorem quadratic_points_range (a : ℝ) (y1 y2 y3 y4 : ℝ) :
  (∀ (x : ℝ), 
    (x = -4 → y1 = a * x^2 + 4 * a * x - 6) ∧ 
    (x = -3 → y2 = a * x^2 + 4 * a * x - 6) ∧ 
    (x = 0 → y3 = a * x^2 + 4 * a * x - 6) ∧ 
    (x = 2 → y4 = a * x^2 + 4 * a * x - 6)) →
  (∃! (y : ℝ), y > 0 ∧ (y = y1 ∨ y = y2 ∨ y = y3 ∨ y = y4)) →
  (a < -2 ∨ a > 1 / 2) :=
by
  sorry

end quadratic_points_range_l102_102629


namespace no_primes_in_range_l102_102818

theorem no_primes_in_range (n : ℕ) (hn : n > 2) : 
  ∀ k, n! + 2 < k ∧ k < n! + n + 1 → ¬Prime k := 
sorry

end no_primes_in_range_l102_102818


namespace find_c_l102_102217

open Real

theorem find_c (a b c d : ℕ) (M : ℝ) (h1 : a > 1) (h2 : b > 1) (h3 : c > 1) (h4 : d > 1) (hM : M ≠ 1) :
  (M ^ (1 / a) * (M ^ (1 / b) * (M ^ (1 / c) * (M ^ (1 / d))))) ^ (1 / a * b * c * d) = (M ^ 37) ^ (1 / 48) →
  c = 2 :=
by
  sorry

end find_c_l102_102217


namespace lcm_36_105_l102_102104

noncomputable def factorize_36 : List (ℕ × ℕ) := [(2, 2), (3, 2)]
noncomputable def factorize_105 : List (ℕ × ℕ) := [(3, 1), (5, 1), (7, 1)]

theorem lcm_36_105 : Nat.lcm 36 105 = 1260 :=
by
  have h_36 : 36 = 2^2 * 3^2 := by norm_num
  have h_105 : 105 = 3^1 * 5^1 * 7^1 := by norm_num
  sorry

end lcm_36_105_l102_102104


namespace random_event_is_B_l102_102956

variable (isCertain : Event → Prop)
variable (isImpossible : Event → Prop)
variable (isRandom : Event → Prop)

variable (A : Event)
variable (B : Event)
variable (C : Event)
variable (D : Event)

-- Here we set the conditions as definitions in Lean 4:
def condition_A : isCertain A := sorry
def condition_B : isRandom B := sorry
def condition_C : isCertain C := sorry
def condition_D : isImpossible D := sorry

-- The theorem we need to prove:
theorem random_event_is_B : isRandom B := 
by
-- adding sorry to skip the proof
sorry

end random_event_is_B_l102_102956


namespace ratio_ab_l102_102086

variable (x y a b : ℝ)
variable (h1 : 4 * x - 2 * y = a)
variable (h2 : 6 * y - 12 * x = b)
variable (h3 : b ≠ 0)

theorem ratio_ab : 4 * x - 2 * y = a ∧ 6 * y - 12 * x = b ∧ b ≠ 0 → a / b = -1 / 3 := by
  sorry

end ratio_ab_l102_102086


namespace change_of_b_l102_102659

variable {t b1 b2 C C_new : ℝ}

theorem change_of_b (hC : C = t * b1^4) 
                   (hC_new : C_new = 16 * C) 
                   (hC_new_eq : C_new = t * b2^4) : 
                   b2 = 2 * b1 :=
by
  sorry

end change_of_b_l102_102659


namespace dino_finances_l102_102718

def earnings_per_gig (hours: ℕ) (rate: ℕ) : ℕ := hours * rate

def dino_total_income : ℕ :=
  earnings_per_gig 20 10 + -- Earnings from the first gig
  earnings_per_gig 30 20 + -- Earnings from the second gig
  earnings_per_gig 5 40    -- Earnings from the third gig

def dino_expenses : ℕ := 500

def dino_net_income : ℕ :=
  dino_total_income - dino_expenses

theorem dino_finances : 
  dino_net_income = 500 :=
by
  -- Here, the actual proof would be constructed.
  sorry

end dino_finances_l102_102718


namespace triangle_sides_possible_k_l102_102716

noncomputable def f (x k : ℝ) : ℝ := x^2 - 4*x + 4 + k^2

theorem triangle_sides_possible_k (a b c k : ℝ) (ha : 0 ≤ a) (hb : a ≤ 3) (ha' : 0 ≤ b) (hb' : b ≤ 3) (ha'' : 0 ≤ c) (hb'' : c ≤ 3) :
  (f a k + f b k > f c k) ∧ (f a k + f c k > f b k) ∧ (f b k + f c k > f a k) ↔ k = 3 ∨ k = 4 :=
by
  sorry

end triangle_sides_possible_k_l102_102716


namespace validity_of_D_l102_102789

def binary_op (a b : ℕ) : ℕ := a^(b + 1)

theorem validity_of_D (a b n : ℕ) (ha : 0 < a) (hb : 0 < b) (hn : 0 < n) :
  binary_op (a^n) b = (binary_op a b)^n := 
by
  sorry

end validity_of_D_l102_102789


namespace initial_blue_balls_l102_102268

-- Define the problem conditions
variable (R B : ℕ) -- Number of red balls and blue balls originally in the box.

-- Condition 1: Blue balls are 17 more than red balls
axiom h1 : B = R + 17

-- Condition 2: Ball addition and removal scenario
noncomputable def total_balls_after_changes : ℕ :=
  (B + 57) + (R + 18) - 44

-- Condition 3: Total balls after all changes is 502
axiom h2 : total_balls_after_changes R B = 502

-- We need to prove the initial number of blue balls
theorem initial_blue_balls : B = 244 :=
by
  sorry

end initial_blue_balls_l102_102268


namespace find_a21_l102_102874

def seq_a (n : ℕ) : ℝ := sorry  -- This should define the sequence a_n
def seq_b (n : ℕ) : ℝ := sorry  -- This should define the sequence b_n

theorem find_a21 (h1 : seq_a 1 = 2)
  (h2 : ∀ n, seq_b n = seq_a (n + 1) / seq_a n)
  (h3 : ∀ n m, seq_b n = seq_b m * r^(n - m)) 
  (h4 : seq_b 10 * seq_b 11 = 2) :
  seq_a 21 = 2 ^ 11 :=
sorry

end find_a21_l102_102874


namespace kitten_length_l102_102150

theorem kitten_length (initial_length : ℕ) (doubled_length_1 : ℕ) (doubled_length_2 : ℕ) :
  initial_length = 4 →
  doubled_length_1 = 2 * initial_length →
  doubled_length_2 = 2 * doubled_length_1 →
  doubled_length_2 = 16 :=
by
  intros h1 h2 h3
  rw [h1] at h2
  rw [h2] at h3
  exact h3

end kitten_length_l102_102150


namespace number_of_ways_to_cut_pipe_l102_102627

theorem number_of_ways_to_cut_pipe : 
  (∃ (x y: ℕ), 2 * x + 3 * y = 15) ∧ 
  (∃! (x y: ℕ), 2 * x + 3 * y = 15) :=
by
  sorry

end number_of_ways_to_cut_pipe_l102_102627


namespace fraction_multiplication_division_l102_102180

-- We will define the fractions and state the equivalence
def fraction_1 : ℚ := 145 / 273
def fraction_2 : ℚ := 2 * (173 / 245) -- equivalent to 2 173/245
def fraction_3 : ℚ := 21 * (13 / 15) -- equivalent to 21 13/15

theorem fraction_multiplication_division :
  (frac1 * frac2 / frac3) = 7395 / 112504 := 
by sorry

end fraction_multiplication_division_l102_102180


namespace problem_proof_l102_102596

def delta (a b : ℕ) : ℕ := a^2 + b

theorem problem_proof :
  let x := 6
  let y := 8
  let z := 4
  let w := 2
  let u := 5^delta x y
  let v := 7^delta z w
  delta u v = 5^88 + 7^18 :=
by
  let x := 6
  let y := 8
  let z := 4
  let w := 2
  let u := 5^delta x y
  let v := 7^delta z w
  have h1: delta x y = 44 := by sorry
  have h2: delta z w = 18 := by sorry
  have hu: u = 5^44 := by sorry
  have hv: v = 7^18 := by sorry
  have hdelta: delta u v = 5^88 + 7^18 := by sorry
  exact hdelta

end problem_proof_l102_102596


namespace inequality_bound_l102_102724

theorem inequality_bound (a x : ℝ) (h1 : |a| ≤ 1) (h2 : |x| ≤ 1) :
  |x^2 - ax - a^2| ≤ 5 / 4 :=
sorry

end inequality_bound_l102_102724


namespace compute_expression_l102_102938

theorem compute_expression : 11 * (1 / 17) * 34 = 22 := 
sorry

end compute_expression_l102_102938


namespace smallest_b_for_perfect_square_l102_102942

theorem smallest_b_for_perfect_square : ∃ (b : ℤ), b > 4 ∧ (∃ (n : ℤ), 4 * b + 5 = n ^ 2) ∧ b = 5 := 
sorry

end smallest_b_for_perfect_square_l102_102942


namespace students_enrolled_both_english_and_german_l102_102842

def total_students : ℕ := 32
def enrolled_german : ℕ := 22
def only_english : ℕ := 10
def students_enrolled_at_least_one_subject := total_students

theorem students_enrolled_both_english_and_german :
  ∃ (e_g : ℕ), e_g = enrolled_german - only_english :=
by
  sorry

end students_enrolled_both_english_and_german_l102_102842


namespace pete_flag_total_circle_square_l102_102223

theorem pete_flag_total_circle_square : 
  let stars := 50
  let stripes := 13
  let circles := (stars / 2) - 3
  let squares := (stripes * 2) + 6
  circles + squares = 54 := 
by
  let stars := 50
  let stripes := 13
  let circles := (stars / 2) - 3
  let squares := (stripes * 2) + 6
  show circles + squares = 54
  sorry

end pete_flag_total_circle_square_l102_102223


namespace range_of_m_l102_102390

theorem range_of_m (x m : ℝ) :
  (∀ x, (x - 1) / 2 ≥ (x - 2) / 3 → 2 * x - m ≥ x → x ≥ m) ↔ m ≥ -1 := by
  sorry

end range_of_m_l102_102390


namespace slow_train_speed_l102_102727

/-- Given the conditions of two trains traveling towards each other and their meeting times,
     prove the speed of the slow train. -/
theorem slow_train_speed :
  let distance_AB := 901
  let slow_train_departure := 5 + 30 / 60 -- 5:30 AM in decimal hours
  let fast_train_departure := 9 + 30 / 60 -- 9:30 AM in decimal hours
  let meeting_time := 16 + 30 / 60 -- 4:30 PM in decimal hours
  let fast_train_speed := 58 -- speed in km/h
  let slow_train_time := meeting_time - slow_train_departure
  let fast_train_time := meeting_time - fast_train_departure
  let fast_train_distance := fast_train_speed * fast_train_time
  let slow_train_distance := distance_AB - fast_train_distance
  let slow_train_speed := slow_train_distance / slow_train_time
  slow_train_speed = 45 := sorry

end slow_train_speed_l102_102727


namespace kitchen_clock_correct_again_bedroom_clock_correct_again_both_clocks_same_time_again_l102_102287

/-- Conditions: -/
def kitchen_clock_gain_rate : ℝ := 1.5 -- minutes per hour
def bedroom_clock_lose_rate : ℝ := 0.5 -- minutes per hour
def synchronization_time : ℝ := 0 -- time in hours when both clocks were correct

/-- Problem 1: -/
theorem kitchen_clock_correct_again :
  ∃ t : ℝ, 1.5 * t = 720 :=
by {
  sorry
}

/-- Problem 2: -/
theorem bedroom_clock_correct_again :
  ∃ t : ℝ, 0.5 * t = 720 :=
by {
  sorry
}

/-- Problem 3: -/
theorem both_clocks_same_time_again :
  ∃ t : ℝ, 2 * t = 720 :=
by {
  sorry
}

end kitchen_clock_correct_again_bedroom_clock_correct_again_both_clocks_same_time_again_l102_102287


namespace compare_M_N_l102_102051

theorem compare_M_N (a b c : ℝ) (h1 : a > 0) (h2 : b < -2 * a) : 
  (|a - b + c| + |2 * a + b|) < (|a + b + c| + |2 * a - b|) :=
by
  sorry

end compare_M_N_l102_102051


namespace total_money_correct_l102_102447

-- Define the number of pennies and quarters Sam has
def pennies : ℕ := 9
def quarters : ℕ := 7

-- Define the value of one penny and one quarter
def penny_value : ℝ := 0.01
def quarter_value : ℝ := 0.25

-- Calculate the total value of pennies and quarters Sam has
def total_value : ℝ := pennies * penny_value + quarters * quarter_value

-- Proof problem: Prove that the total value of money Sam has is $1.84
theorem total_money_correct : total_value = 1.84 :=
sorry

end total_money_correct_l102_102447


namespace proof_l102_102108

def statement : Prop :=
  ∀ (a : ℝ),
    (¬ (∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ a^2 - 3 * a - x + 1 > 0) ∧
    ¬ (a^2 - 4 ≥ 0 ∧
    (∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ a^2 - 3 * a - x + 1 > 0)))
    → (1 ≤ a ∧ a < 2)

theorem proof : statement :=
by
  sorry

end proof_l102_102108


namespace badminton_members_count_l102_102028

def total_members := 30
def neither_members := 2
def both_members := 6

def members_play_badminton_and_tennis (B T : ℕ) : Prop :=
  B + T - both_members = total_members - neither_members

theorem badminton_members_count (B T : ℕ) (hbt : B = T) :
  members_play_badminton_and_tennis B T → B = 17 :=
by
  intros h
  sorry

end badminton_members_count_l102_102028


namespace geom_seq_sum_a3_a4_a5_l102_102258

-- Define the geometric sequence terms and sum condition
def geometric_seq (a1 q : ℕ) (n : ℕ) : ℕ :=
  a1 * q^(n - 1)

def sum_first_three (a1 q : ℕ) : ℕ :=
  a1 + a1 * q + a1 * q^2

-- Given conditions
def a1 : ℕ := 3
def S3 : ℕ := 21

-- Define the problem statement
theorem geom_seq_sum_a3_a4_a5 (q : ℕ) (h : sum_first_three a1 q = S3) (h_pos : ∀ n, geometric_seq a1 q n > 0) :
  geometric_seq a1 q 3 + geometric_seq a1 q 4 + geometric_seq a1 q 5 = 84 :=
by sorry

end geom_seq_sum_a3_a4_a5_l102_102258


namespace donna_paid_165_l102_102505

def original_price : ℝ := 200
def discount_rate : ℝ := 0.25
def tax_rate : ℝ := 0.1

def sale_price := original_price * (1 - discount_rate)
def tax := sale_price * tax_rate
def total_amount_paid := sale_price + tax

theorem donna_paid_165 : total_amount_paid = 165 := by
  sorry

end donna_paid_165_l102_102505


namespace karen_average_speed_correct_l102_102309

def karen_time_duration : ℚ := (22 : ℚ) / 3
def karen_distance : ℚ := 230

def karen_average_speed (distance : ℚ) (time : ℚ) : ℚ := distance / time

theorem karen_average_speed_correct :
  karen_average_speed karen_distance karen_time_duration = (31 + 4/11 : ℚ) :=
by
  sorry

end karen_average_speed_correct_l102_102309


namespace simplest_quadratic_radicals_same_type_l102_102916

theorem simplest_quadratic_radicals_same_type (m n : ℕ)
  (h : ∀ {a : ℕ}, (a = m - 1 → a = 2) ∧ (a = 4 * n - 1 → a = 7)) :
  m + n = 5 :=
sorry

end simplest_quadratic_radicals_same_type_l102_102916


namespace Auston_height_in_cm_l102_102990

theorem Auston_height_in_cm : 
  (60 : ℝ) * 2.54 = 152.4 :=
by sorry

end Auston_height_in_cm_l102_102990


namespace Liam_chapters_in_fourth_week_l102_102759

noncomputable def chapters_in_first_week (x : ℕ) : ℕ := x
noncomputable def chapters_in_second_week (x : ℕ) : ℕ := x + 3
noncomputable def chapters_in_third_week (x : ℕ) : ℕ := x + 6
noncomputable def chapters_in_fourth_week (x : ℕ) : ℕ := x + 9
noncomputable def total_chapters (x : ℕ) : ℕ := x + (x + 3) + (x + 6) + (x + 9)

theorem Liam_chapters_in_fourth_week : ∃ x : ℕ, total_chapters x = 50 → chapters_in_fourth_week x = 17 :=
by
  sorry

end Liam_chapters_in_fourth_week_l102_102759


namespace problem_statement_l102_102507

-- Definitions based on problem conditions
def p (a b c : ℝ) : Prop := a > b → (a * c^2 > b * c^2)

def q : Prop := ∃ x_0 : ℝ, (x_0 > 0) ∧ (x_0 - 1 + Real.log x_0 = 0)

-- Main theorem
theorem problem_statement : (¬ (∀ a b c : ℝ, p a b c)) ∧ q :=
by sorry

end problem_statement_l102_102507


namespace find_solution_l102_102709

open Nat

def binomial (n : ℕ) (k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.choose n k else 0

noncomputable def expression (n : ℕ) : ℕ :=
  1 + binomial n 1 + binomial n 2 + binomial n 3

theorem find_solution (n : ℕ) (h : n > 3) :
  expression n ∣ 2 ^ 2000 ↔ n = 7 ∨ n = 23 :=
by
  sorry

end find_solution_l102_102709


namespace cars_on_river_road_l102_102971

-- Define the given conditions
variables (B C : ℕ)
axiom ratio_condition : B = C / 13
axiom difference_condition : B = C - 60 

-- State the theorem to be proved
theorem cars_on_river_road : C = 65 :=
by
  -- proof would go here 
  sorry

end cars_on_river_road_l102_102971


namespace jiwon_distance_to_school_l102_102130

theorem jiwon_distance_to_school
  (taehong_distance_meters jiwon_distance_meters : ℝ)
  (taehong_distance_km : ℝ := 1.05)
  (h1 : taehong_distance_meters = jiwon_distance_meters + 460)
  (h2 : taehong_distance_meters = taehong_distance_km * 1000) :
  jiwon_distance_meters / 1000 = 0.59 := 
sorry

end jiwon_distance_to_school_l102_102130


namespace winner_more_votes_than_second_place_l102_102678

theorem winner_more_votes_than_second_place :
  ∃ (W S T F : ℕ), 
    F = 199 ∧
    W = S + (W - S) ∧
    W = T + 79 ∧
    W = F + 105 ∧
    W + S + T + F = 979 ∧
    W - S = 53 :=
by
  sorry

end winner_more_votes_than_second_place_l102_102678


namespace zero_not_in_range_of_g_l102_102257

noncomputable def g (x : ℝ) : ℤ :=
  if x > -3 then ⌈(Real.cos x) / (x + 3)⌉
  else if x < -3 then ⌊(Real.cos x) / (x + 3)⌋
  else 0 -- arbitrary value since it's undefined

theorem zero_not_in_range_of_g :
  ¬ (∃ x : ℝ, g x = 0) :=
by
  intro h
  sorry

end zero_not_in_range_of_g_l102_102257


namespace quadrilateral_offset_l102_102253

-- Define the problem statement
theorem quadrilateral_offset
  (d : ℝ) (x : ℝ) (y : ℝ) (A : ℝ)
  (h₀ : d = 10) 
  (h₁ : y = 3) 
  (h₂ : A = 50) :
  x = 7 :=
by
  -- Assuming the given conditions
  have h₃ : A = 1/2 * d * x + 1/2 * d * y :=
  by
    -- specific formula for area of the quadrilateral
    sorry
  
  -- Given A = 50, d = 10, y = 3, solve for x to show x = 7
  sorry

end quadrilateral_offset_l102_102253


namespace find_xy_l102_102461

noncomputable def xy_value (x y : ℝ) := x * y

theorem find_xy :
  ∃ x y : ℝ, (x + y = 2) ∧ (x^2 * y^3 + y^2 * x^3 = 32) ∧ xy_value x y = -8 :=
by
  sorry

end find_xy_l102_102461


namespace car_travel_distance_l102_102566

-- Definitions based on the problem
def arith_seq_sum (a1 d : ℤ) (n : ℕ) : ℤ :=
  n * a1 + d * (n * (n - 1)) / 2

-- Main statement to prove
theorem car_travel_distance : arith_seq_sum 40 (-12) 5 = 88 :=
by sorry

end car_travel_distance_l102_102566


namespace no_integer_solution_for_triples_l102_102291

theorem no_integer_solution_for_triples :
  ∀ (x y z : ℤ),
    x^2 - 2*x*y + 3*y^2 - z^2 = 17 →
    -x^2 + 4*y*z + z^2 = 28 →
    x^2 + 2*x*y + 5*z^2 = 42 →
    false :=
by
  intros x y z h1 h2 h3
  sorry

end no_integer_solution_for_triples_l102_102291


namespace complement_union_M_N_l102_102765

universe u

namespace complement_union

def U : Set (ℝ × ℝ) := { p | true }

def M : Set (ℝ × ℝ) := { p | (p.2 - 3) = (p.1 - 2) }

def N : Set (ℝ × ℝ) := { p | p.2 ≠ (p.1 + 1) }

theorem complement_union_M_N : (U \ (M ∪ N)) = { (2, 3) } := 
by 
  sorry

end complement_union

end complement_union_M_N_l102_102765


namespace sequence_general_term_l102_102599

theorem sequence_general_term (S : ℕ → ℕ) (a : ℕ → ℕ) (hS : ∀ n, S n = n^2) 
    (h_a₁ : S 1 = 1) (h_an : ∀ n, n ≥ 2 → a n = S n - S (n - 1)) : 
  ∀ n, a n = 2 * n - 1 := 
by
  sorry

end sequence_general_term_l102_102599


namespace cos_value_l102_102901

theorem cos_value (α : ℝ) (h : Real.sin (π / 6 - α) = 1 / 3) : Real.cos (2 * π / 3 + 2 * α) = -7 / 9 :=
by
  sorry

end cos_value_l102_102901


namespace folder_cost_l102_102090

theorem folder_cost (cost_pens : ℕ) (cost_notebooks : ℕ) (total_spent : ℕ) (folders : ℕ) :
  cost_pens = 3 → cost_notebooks = 12 → total_spent = 25 → folders = 2 →
  ∃ (cost_per_folder : ℕ), cost_per_folder = 5 :=
by
  intros
  sorry

end folder_cost_l102_102090


namespace f_zero_eq_one_f_pos_all_f_increasing_l102_102008

noncomputable def f : ℝ → ℝ := sorry

axiom f_nonzero : f 0 ≠ 0
axiom f_pos : ∀ x, 0 < x → 1 < f x
axiom f_mul : ∀ a b : ℝ, f (a + b) = f a * f b

theorem f_zero_eq_one : f 0 = 1 :=
sorry

theorem f_pos_all : ∀ x : ℝ, 0 < f x :=
sorry

theorem f_increasing : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂ :=
sorry

end f_zero_eq_one_f_pos_all_f_increasing_l102_102008


namespace min_value_correct_l102_102323

noncomputable def min_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : ℝ :=
  Real.sqrt ((a^2 + 2 * b^2) * (4 * a^2 + b^2)) / (a * b)

theorem min_value_correct (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  min_value a b ha hb ≥ 3 :=
sorry

end min_value_correct_l102_102323


namespace solve_for_x_l102_102719

theorem solve_for_x (x : ℚ) (h : x + 3 * x = 300 - (4 * x + 5 * x)) : x = 300 / 13 :=
by
  sorry

end solve_for_x_l102_102719


namespace count_three_digit_concave_numbers_l102_102074

def is_concave_number (a b c : ℕ) : Prop :=
  a > b ∧ c > b

theorem count_three_digit_concave_numbers : 
  (∃! n : ℕ, n = 240) := by
  sorry

end count_three_digit_concave_numbers_l102_102074


namespace sin_pi_minus_alpha_l102_102878

theorem sin_pi_minus_alpha (α : ℝ) (h : Real.sin α = 1 / 2) : Real.sin (π - α) = 1 / 2 :=
by
  sorry

end sin_pi_minus_alpha_l102_102878


namespace find_sum_of_numbers_l102_102979

theorem find_sum_of_numbers (x A B C : ℝ) (h1 : x > 0) (h2 : A = x) (h3 : B = 2 * x) (h4 : C = 3 * x) (h5 : A^2 + B^2 + C^2 = 2016) : A + B + C = 72 :=
sorry

end find_sum_of_numbers_l102_102979


namespace exists_a_satisfying_f_l102_102615

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 1 then -x^2 + 1 else x - 1

theorem exists_a_satisfying_f (a : ℝ) : 
  f (a + 1) = f a ↔ (a = -1/2 ∨ a = (-1 + Real.sqrt 5) / 2) :=
by
  sorry

end exists_a_satisfying_f_l102_102615


namespace function_takes_negative_values_l102_102905

def f (x a : ℝ) : ℝ := x^2 - a * x + 1

theorem function_takes_negative_values {a : ℝ} :
  (∃ x : ℝ, f x a < 0) ↔ (a > 2 ∨ a < -2) :=
by
  sorry

end function_takes_negative_values_l102_102905


namespace combined_stickers_l102_102589

def initial_stickers_june : ℕ := 76
def initial_stickers_bonnie : ℕ := 63
def birthday_stickers : ℕ := 25

theorem combined_stickers : 
  (initial_stickers_june + birthday_stickers) + (initial_stickers_bonnie + birthday_stickers) = 189 := 
by
  sorry

end combined_stickers_l102_102589


namespace candy_pieces_per_package_l102_102955

theorem candy_pieces_per_package (packages_gum : ℕ) (packages_candy : ℕ) (total_candies : ℕ) :
  packages_gum = 21 →
  packages_candy = 45 →
  total_candies = 405 →
  total_candies / packages_candy = 9 := by
  intros h1 h2 h3
  sorry

end candy_pieces_per_package_l102_102955


namespace rectangular_park_area_l102_102941

/-- Define the conditions for the rectangular park -/
def rectangular_park (w l : ℕ) : Prop :=
  l = 3 * w ∧ 2 * (w + l) = 72

/-- Prove that the area of the rectangular park is 243 square meters -/
theorem rectangular_park_area (w l : ℕ) (h : rectangular_park w l) : w * l = 243 := by
  sorry

end rectangular_park_area_l102_102941


namespace ramu_profit_percent_is_21_64_l102_102593

-- Define the costs and selling price as constants
def cost_of_car : ℕ := 42000
def cost_of_repairs : ℕ := 13000
def selling_price : ℕ := 66900

-- Define the total cost and profit
def total_cost : ℕ := cost_of_car + cost_of_repairs
def profit : ℕ := selling_price - total_cost

-- Define the profit percent formula
def profit_percent : ℚ := ((profit : ℚ) / (total_cost : ℚ)) * 100

-- State the theorem we want to prove
theorem ramu_profit_percent_is_21_64 : profit_percent = 21.64 := by
  sorry

end ramu_profit_percent_is_21_64_l102_102593


namespace jose_investment_l102_102512

theorem jose_investment (P T : ℝ) (X : ℝ) (months_tom months_jose : ℝ) (profit_total profit_jose profit_tom : ℝ) :
  T = 30000 →
  months_tom = 12 →
  months_jose = 10 →
  profit_total = 54000 →
  profit_jose = 30000 →
  profit_tom = profit_total - profit_jose →
  profit_tom / profit_jose = (T * months_tom) / (X * months_jose) →
  X = 45000 :=
by sorry

end jose_investment_l102_102512


namespace value_of_f_of_x_minus_3_l102_102084

theorem value_of_f_of_x_minus_3 (x : ℝ) (f : ℝ → ℝ) (h : ∀ y : ℝ, f y = y^2) : f (x - 3) = x^2 - 6*x + 9 :=
by
  sorry

end value_of_f_of_x_minus_3_l102_102084


namespace empty_seats_in_theater_l102_102710

theorem empty_seats_in_theater :
  let total_seats := 750
  let occupied_seats := 532
  total_seats - occupied_seats = 218 :=
by
  sorry

end empty_seats_in_theater_l102_102710


namespace tom_wins_with_smallest_n_l102_102147

def tom_and_jerry_game_proof_problem (n : ℕ) : Prop :=
  ∀ (pos : ℕ), pos ≥ 1 ∧ pos ≤ 2018 → 
  ∀ (move : ℕ), move ≥ 1 ∧ move ≤ n →
  (∃ n_min : ℕ, n_min ≤ n ∧ ∀ pos, (pos ≤ n_min ∨ pos > 2018 - n_min) → false)

theorem tom_wins_with_smallest_n : tom_and_jerry_game_proof_problem 1010 :=
sorry

end tom_wins_with_smallest_n_l102_102147


namespace smallest_abs_value_l102_102343

theorem smallest_abs_value : 
    ∀ (a b c d : ℝ), 
    a = -1/2 → b = -2/3 → c = 4 → d = -5 → 
    abs a < abs b ∧ abs a < abs c ∧ abs a < abs d := 
by
  intros a b c d ha hb hc hd
  rw [ha, hb, hc, hd]
  simp
  -- Proof omitted for brevity
  sorry

end smallest_abs_value_l102_102343


namespace positive_difference_two_solutions_abs_eq_15_l102_102949

theorem positive_difference_two_solutions_abs_eq_15 :
  ∀ (x1 x2 : ℝ), (|x1 - 3| = 15) ∧ (|x2 - 3| = 15) ∧ (x1 > x2) → (x1 - x2 = 30) :=
by
  intros x1 x2 h
  sorry

end positive_difference_two_solutions_abs_eq_15_l102_102949


namespace committee_selection_l102_102260

-- Definitions based on the conditions
def num_people := 12
def num_women := 7
def num_men := 5
def committee_size := 5
def min_women := 2

-- Binomial coefficient function
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Required number of ways to form the committee
def num_ways_5_person_committee_with_at_least_2_women : ℕ :=
  binom num_women min_women * binom (num_people - min_women) (committee_size - min_women)

-- Statement to be proven
theorem committee_selection : num_ways_5_person_committee_with_at_least_2_women = 2520 :=
by
  sorry

end committee_selection_l102_102260


namespace problem_statement_l102_102832

variable (m : ℝ) -- We declare m as a real number

theorem problem_statement (h : m + 1/m = 10) : m^2 + 1/m^2 + 4 = 102 := 
by 
  sorry -- The proof is omitted

end problem_statement_l102_102832


namespace determine_y_l102_102106

theorem determine_y (y : ℕ) : (8^5 + 8^5 + 2 * 8^5 = 2^y) → y = 17 := 
by {
  sorry
}

end determine_y_l102_102106


namespace completion_days_for_B_l102_102466

-- Conditions
def A_completion_days := 20
def B_completion_days (x : ℕ) := x
def project_completion_days := 20
def A_work_days := project_completion_days - 10
def B_work_days := project_completion_days
def A_work_rate := 1 / A_completion_days
def B_work_rate (x : ℕ) := 1 / B_completion_days x
def combined_work_rate (x : ℕ) := A_work_rate + B_work_rate x
def A_project_completed := A_work_days * A_work_rate
def B_project_remaining (x : ℕ) := 1 - A_project_completed
def B_project_completion (x : ℕ) := B_work_days * B_work_rate x

-- Proof statement
theorem completion_days_for_B (x : ℕ) 
  (h : B_project_completion x = B_project_remaining x ∧ combined_work_rate x > 0) :
  x = 40 :=
sorry

end completion_days_for_B_l102_102466


namespace real_nums_inequality_l102_102007

theorem real_nums_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h : a ^ 2000 + b ^ 2000 = a ^ 1998 + b ^ 1998) :
  a ^ 2 + b ^ 2 ≤ 2 :=
sorry

end real_nums_inequality_l102_102007


namespace quadratic_two_distinct_real_roots_l102_102690

theorem quadratic_two_distinct_real_roots : 
  ∀ (a b c : ℝ), a = 1 ∧ b = -5 ∧ c = 6 → 
  b^2 - 4 * a * c > 0 :=
by
  sorry

end quadratic_two_distinct_real_roots_l102_102690


namespace haley_more_than_josh_l102_102746

-- Definitions of the variables and conditions
variable (H : Nat) -- Number of necklaces Haley has
variable (J : Nat) -- Number of necklaces Jason has
variable (Jos : Nat) -- Number of necklaces Josh has

-- The conditions as assumptions
axiom h1 : H = 25
axiom h2 : H = J + 5
axiom h3 : Jos = J / 2

-- The theorem we want to prove based on these conditions
theorem haley_more_than_josh (H J Jos : Nat) (h1 : H = 25) (h2 : H = J + 5) (h3 : Jos = J / 2) : H - Jos = 15 := 
by 
  sorry

end haley_more_than_josh_l102_102746


namespace height_in_meters_l102_102823

theorem height_in_meters (h: 1 * 100 + 36 = 136) : 1.36 = 1 + 36 / 100 :=
by 
  -- proof steps will go here
  sorry

end height_in_meters_l102_102823


namespace equivalent_math_problem_l102_102548

noncomputable def P : ℝ := Real.sqrt 1011 + Real.sqrt 1012
noncomputable def Q : ℝ := - (Real.sqrt 1011 + Real.sqrt 1012)
noncomputable def R : ℝ := Real.sqrt 1011 - Real.sqrt 1012
noncomputable def S : ℝ := Real.sqrt 1012 - Real.sqrt 1011

theorem equivalent_math_problem :
  (P * Q)^2 * R * S = 8136957 :=
by
  sorry

end equivalent_math_problem_l102_102548


namespace arithmetic_sequence_condition_l102_102403

theorem arithmetic_sequence_condition (a : ℕ → ℝ) (h : 2 * (a 1 + a 3 + a 5) + 3 * (a 8 + a 10) = 36) : 
a 6 = 3 := 
by 
  sorry

end arithmetic_sequence_condition_l102_102403


namespace original_price_given_discounts_l102_102131

theorem original_price_given_discounts (p q d : ℝ) (h : d > 0) :
  ∃ x : ℝ, x * (1 + (p - q) / 100 - p * q / 10000) = d :=
by
  sorry

end original_price_given_discounts_l102_102131


namespace domain_f1_correct_f2_correct_f2_at_3_l102_102556

noncomputable def f1 (x : ℝ) : ℝ := Real.sqrt (4 - 2 * x) + 1 + 1 / (x + 1)

noncomputable def domain_f1 : Set ℝ := {x | 4 - 2 * x ≥ 0} \ (insert 1 (insert (-1) {}))

theorem domain_f1_correct : domain_f1 = { x | x ≤ 2 ∧ x ≠ 1 ∧ x ≠ -1 } :=
by
  sorry

noncomputable def f2 (x : ℝ) : ℝ := x^2 - 4 * x + 3

theorem f2_correct : ∀ x, f2 (x + 1) = x^2 - 2 * x :=
by
  sorry

theorem f2_at_3 : f2 3 = 0 :=
by
  sorry

end domain_f1_correct_f2_correct_f2_at_3_l102_102556


namespace minimum_f_l102_102351

def f (x : ℝ) : ℝ := |3 - x| + |x - 7|

theorem minimum_f : ∀ x : ℝ, min (f x) = 4 := sorry

end minimum_f_l102_102351


namespace rotated_parabola_eq_l102_102311

theorem rotated_parabola_eq :
  ∀ x y : ℝ, y = x^2 → ∃ y' x' : ℝ, (y' = (-x':ℝ)^2) := sorry

end rotated_parabola_eq_l102_102311


namespace A_walking_speed_l102_102632

-- Definition for the conditions
def A_speed (v : ℝ) : Prop := 
  ∃ (t : ℝ), 120 = 20 * (t - 6) ∧ 120 = v * t

-- The main theorem to prove the question
theorem A_walking_speed : ∀ (v : ℝ), A_speed v → v = 10 :=
by
  intros v h
  sorry

end A_walking_speed_l102_102632


namespace novel_writing_time_l102_102218

theorem novel_writing_time :
  ∀ (total_words : ℕ) (first_half_speed second_half_speed : ℕ),
    total_words = 50000 →
    first_half_speed = 600 →
    second_half_speed = 400 →
    (total_words / 2 / first_half_speed + total_words / 2 / second_half_speed : ℚ) = 104.17 :=
by
  -- No proof is required, placeholder using sorry
  sorry

end novel_writing_time_l102_102218


namespace correct_operation_B_l102_102142

theorem correct_operation_B (x : ℝ) : 
  x - 2 * x = -x :=
sorry

end correct_operation_B_l102_102142


namespace sum_of_five_distinct_integers_product_2022_l102_102722

theorem sum_of_five_distinct_integers_product_2022 :
  ∃ (a b c d e : ℤ), 
    a * b * c * d * e = 2022 ∧ 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
    c ≠ d ∧ c ≠ e ∧
    d ≠ e ∧ 
    (a + b + c + d + e = 342 ∨
     a + b + c + d + e = 338 ∨
     a + b + c + d + e = 336 ∨
     a + b + c + d + e = -332) :=
by 
  sorry

end sum_of_five_distinct_integers_product_2022_l102_102722


namespace mathe_matics_equals_2014_l102_102806

/-- 
Given the following mappings for characters in the word "MATHEMATICS":
M = 1, A = 8, T = 3, E = '+', I = 9, K = '-',
verify that the resulting numerical expression 183 + 1839 - 8 equals 2014.
-/
theorem mathe_matics_equals_2014 :
  183 + 1839 - 8 = 2014 :=
by
  sorry

end mathe_matics_equals_2014_l102_102806


namespace exists_solution_iff_l102_102691

theorem exists_solution_iff (m : ℝ) (x y : ℝ) :
  ((y = (3 * m + 2) * x + 1) ∧ (y = (5 * m - 4) * x + 5)) ↔ m ≠ 3 :=
by sorry

end exists_solution_iff_l102_102691


namespace arithmetic_result_l102_102288

/-- Define the constants involved in the arithmetic operation. -/
def a : ℕ := 999999999999
def b : ℕ := 888888888888
def c : ℕ := 111111111111

/-- The theorem stating that the given arithmetic operation results in the expected answer. -/
theorem arithmetic_result :
  a - b + c = 222222222222 :=
by
  sorry

end arithmetic_result_l102_102288


namespace total_coin_tosses_l102_102116

variable (heads : ℕ) (tails : ℕ)

theorem total_coin_tosses (h_head : heads = 9) (h_tail : tails = 5) : heads + tails = 14 := by
  sorry

end total_coin_tosses_l102_102116


namespace trinomial_ne_binomial_l102_102360

theorem trinomial_ne_binomial (a b c A B : ℝ) (h : a ≠ 0) : 
  ¬ ∀ x : ℝ, ax^2 + bx + c = Ax + B :=
by
  sorry

end trinomial_ne_binomial_l102_102360


namespace probability_exact_four_out_of_twelve_dice_is_approx_0_089_l102_102649

noncomputable def dice_probability_exact_four_six : ℝ :=
  let p := (1/6 : ℝ)
  let q := (5/6 : ℝ)
  (Nat.choose 12 4) * (p ^ 4) * (q ^ 8)

theorem probability_exact_four_out_of_twelve_dice_is_approx_0_089 :
  abs (dice_probability_exact_four_six - 0.089) < 0.001 :=
sorry

end probability_exact_four_out_of_twelve_dice_is_approx_0_089_l102_102649


namespace circumference_of_jogging_track_l102_102313

noncomputable def trackCircumference (Deepak_speed : ℝ) (Wife_speed : ℝ) (meet_time_minutes : ℝ) : ℝ :=
  let relative_speed := Deepak_speed + Wife_speed
  let meet_time_hours := meet_time_minutes / 60
  relative_speed * meet_time_hours

theorem circumference_of_jogging_track :
  trackCircumference 20 17 37 = 1369 / 60 :=
by
  sorry

end circumference_of_jogging_track_l102_102313


namespace ratio_of_x_y_l102_102080

theorem ratio_of_x_y (x y : ℚ) (h : (3 * x - 2 * y) / (2 * x + 3 * y + 1) = 4 / 5) : x / y = 22 / 7 :=
sorry

end ratio_of_x_y_l102_102080


namespace students_not_A_either_l102_102199

-- Given conditions as definitions
def total_students : ℕ := 40
def students_A_history : ℕ := 10
def students_A_math : ℕ := 18
def students_A_both : ℕ := 6

-- Statement to prove
theorem students_not_A_either : (total_students - (students_A_history + students_A_math - students_A_both)) = 18 := 
by
  sorry

end students_not_A_either_l102_102199


namespace div_by_64_l102_102179

theorem div_by_64 (n : ℕ) (h : n ≥ 1) : 64 ∣ (3^(2*n + 2) - 8*n - 9) :=
sorry

end div_by_64_l102_102179


namespace find_h3_l102_102977

noncomputable def h (x : ℝ) : ℝ :=
  ((x + 1) * (x^2 + 1) * (x^3 + 1) * (x^9 + 1) - 1) / (x^(3^3 - 1) - 1)

theorem find_h3 : h 3 = 3 := by
  sorry

end find_h3_l102_102977


namespace goods_purchase_solutions_l102_102634

theorem goods_purchase_solutions (a : ℕ) (h1 : 0 < a ∧ a ≤ 45) :
  ∃ x : ℝ, 45 - 20 * (x - 1) = a * x :=
by sorry

end goods_purchase_solutions_l102_102634


namespace distance_down_correct_l102_102022

-- Conditions
def rate_up : ℕ := 5  -- rate on the way up (miles per day)
def time_up : ℕ := 2  -- time to travel up (days)
def rate_factor : ℕ := 3 / 2  -- factor for the rate on the way down
def time_down := time_up  -- time to travel down is the same

-- Formula for computation
def distance_up : ℕ := rate_up * time_up
def rate_down : ℕ := rate_up * rate_factor
def distance_down : ℕ := rate_down * time_down

-- Theorem to be proved
theorem distance_down_correct : distance_down = 15 := by
  sorry

end distance_down_correct_l102_102022


namespace tree_planting_problem_l102_102824

variables (n t : ℕ)

theorem tree_planting_problem (h1 : 4 * n = t + 11) (h2 : 2 * n = t - 13) : n = 12 ∧ t = 37 :=
by
  sorry

end tree_planting_problem_l102_102824


namespace percentage_value_l102_102293

variables {P a b c : ℝ}

theorem percentage_value (h1 : (P / 100) * a = 12) (h2 : (12 / 100) * b = 6) (h3 : c = b / a) : c = P / 24 :=
by
  sorry

end percentage_value_l102_102293


namespace payment_difference_correct_l102_102713

noncomputable def initial_debt : ℝ := 12000

noncomputable def planA_interest_rate : ℝ := 0.08
noncomputable def planA_compounding_periods : ℕ := 2

noncomputable def planB_interest_rate : ℝ := 0.08

noncomputable def planA_payment_years : ℕ := 4
noncomputable def planA_remaining_years : ℕ := 4

noncomputable def planB_years : ℕ := 8

-- Amount accrued in Plan A after 4 years
noncomputable def planA_amount_after_first_period : ℝ :=
  initial_debt * (1 + planA_interest_rate / planA_compounding_periods) ^ (planA_compounding_periods * planA_payment_years)

-- Amount paid at the end of first period (two-thirds of total)
noncomputable def planA_first_payment : ℝ :=
  (2/3) * planA_amount_after_first_period

-- Remaining debt after first payment
noncomputable def planA_remaining_debt : ℝ :=
  planA_amount_after_first_period - planA_first_payment

-- Amount accrued on remaining debt after 8 years (second 4-year period)
noncomputable def planA_second_payment : ℝ :=
  planA_remaining_debt * (1 + planA_interest_rate / planA_compounding_periods) ^ (planA_compounding_periods * planA_remaining_years)

-- Total payment under Plan A
noncomputable def total_payment_planA : ℝ :=
  planA_first_payment + planA_second_payment

-- Total payment under Plan B
noncomputable def total_payment_planB : ℝ :=
  initial_debt * (1 + planB_interest_rate * planB_years)

-- Positive difference between payments
noncomputable def payment_difference : ℝ :=
  total_payment_planB - total_payment_planA

theorem payment_difference_correct :
  payment_difference = 458.52 :=
by
  sorry

end payment_difference_correct_l102_102713


namespace insurance_covers_80_percent_of_lenses_l102_102785

/--
James needs to get a new pair of glasses. 
His frames cost $200 and the lenses cost $500. 
Insurance will cover a certain percentage of the cost of lenses and he has a $50 off coupon for frames. 
Everything costs $250. 
Prove that the insurance covers 80% of the cost of the lenses.
-/

def frames_cost : ℕ := 200
def lenses_cost : ℕ := 500
def total_cost_after_discounts_and_insurance : ℕ := 250
def coupon : ℕ := 50

theorem insurance_covers_80_percent_of_lenses :
  ((frames_cost - coupon + lenses_cost - total_cost_after_discounts_and_insurance) * 100 / lenses_cost) = 80 := 
  sorry

end insurance_covers_80_percent_of_lenses_l102_102785


namespace mauve_red_paint_parts_l102_102175

noncomputable def parts_of_red_in_mauve : ℕ :=
let fuchsia_red_ratio := 5
let fuchsia_blue_ratio := 3
let total_fuchsia := 16
let added_blue := 14
let mauve_blue_ratio := 6

let total_fuchsia_parts := fuchsia_red_ratio + fuchsia_blue_ratio
let red_in_fuchsia := (fuchsia_red_ratio * total_fuchsia) / total_fuchsia_parts
let blue_in_fuchsia := (fuchsia_blue_ratio * total_fuchsia) / total_fuchsia_parts
let blue_in_mauve := blue_in_fuchsia + added_blue
let ratio_red_to_blue_in_mauve := red_in_fuchsia / blue_in_mauve
ratio_red_to_blue_in_mauve * mauve_blue_ratio

theorem mauve_red_paint_parts : parts_of_red_in_mauve = 3 :=
by sorry

end mauve_red_paint_parts_l102_102175


namespace cindy_olaf_earnings_l102_102158
noncomputable def total_earnings (apples grapes : ℕ) (price_apple price_grape : ℝ) : ℝ :=
  apples * price_apple + grapes * price_grape

theorem cindy_olaf_earnings :
  total_earnings 15 12 2 1.5 = 48 :=
by
  sorry

end cindy_olaf_earnings_l102_102158


namespace min_value_of_function_l102_102797

theorem min_value_of_function (p : ℝ) : 
  ∃ x : ℝ, (x^2 - 2 * p * x + 2 * p^2 + 2 * p - 1) = -2 := sorry

end min_value_of_function_l102_102797


namespace carlotta_total_time_l102_102018

-- Define the main function for calculating total time
def total_time (performance_time practicing_ratio tantrum_ratio : ℕ) : ℕ :=
  performance_time + (performance_time * practicing_ratio) + (performance_time * tantrum_ratio)

-- Define the conditions from the problem
def singing_time := 6
def practicing_per_minute := 3
def tantrums_per_minute := 5

-- The expected total time based on the conditions
def expected_total_time := 54

-- The theorem to prove the equivalence
theorem carlotta_total_time :
  total_time singing_time practicing_per_minute tantrums_per_minute = expected_total_time :=
by
  sorry

end carlotta_total_time_l102_102018


namespace simplify_expression_l102_102407

variable (x y z : ℝ)

-- Statement of the problem to be proved.
theorem simplify_expression :
  (15 * x + 45 * y - 30 * z) + (20 * x - 10 * y + 5 * z) - (5 * x + 35 * y - 15 * z) = 
  (30 * x - 10 * z) :=
by
  -- Placeholder for the actual proof
  sorry

end simplify_expression_l102_102407


namespace general_term_formula_l102_102543

-- Define the sequence as given in the conditions
def seq (n : ℕ) : ℚ := 
  match n with 
  | 0       => 1
  | 1       => 2 / 3
  | 2       => 1 / 2
  | 3       => 2 / 5
  | (n + 1) => sorry   -- This is just a placeholder, to be proved

-- State the theorem
theorem general_term_formula (n : ℕ) : seq n = 2 / (n + 1) := 
by {
  -- Proof will be provided here
  sorry
}

end general_term_formula_l102_102543


namespace youtube_dislikes_calculation_l102_102565

theorem youtube_dislikes_calculation :
  ∀ (l d_initial d_final : ℕ),
    l = 3000 →
    d_initial = (l / 2) + 100 →
    d_final = d_initial + 1000 →
    d_final = 2600 :=
by
  intros l d_initial d_final h_l h_d_initial h_d_final
  sorry

end youtube_dislikes_calculation_l102_102565


namespace cot_sum_simplified_l102_102472

noncomputable def cot (x : ℝ) : ℝ := (Real.cos x) / (Real.sin x)

theorem cot_sum_simplified : cot (π / 24) + cot (π / 8) = 96 / (π^2) := 
by 
  sorry

end cot_sum_simplified_l102_102472


namespace constant_term_expansion_l102_102184

-- auxiliary definitions and facts
def binomial_coeff (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def term_constant (n k : ℕ) (a b x : ℂ) : ℂ :=
  binomial_coeff n k * (a * x)^(n-k) * (b / x)^k

-- main theorem statement
theorem constant_term_expansion : ∀ (x : ℂ), (term_constant 8 4 (5 : ℂ) (2 : ℂ) x).re = 1120 :=
by
  intro x
  sorry

end constant_term_expansion_l102_102184


namespace constants_sum_l102_102165

theorem constants_sum (c d : ℝ) (f : ℝ → ℝ) 
  (h₁ : ∀ x : ℝ, f x = if x ≤ 5 then c * x + d else 10 - 2 * x) 
  (h₂ : ∀ x : ℝ, f (f x) = x) : c + d = 6.5 := 
by sorry

end constants_sum_l102_102165


namespace arithmetic_sequence_ratio_l102_102490

theorem arithmetic_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : ∀ n, S n = (n * (a 1 + a n)) / 2)
  (h2 : ∀ n, S n / a n = (n + 1) / 2) :
  (a 2 / a 3 = 2 / 3) :=
sorry

end arithmetic_sequence_ratio_l102_102490


namespace smallest_a_l102_102974

theorem smallest_a (a b c : ℚ)
  (h1 : a > 0)
  (h2 : b = -2 * a / 3)
  (h3 : c = a / 9 - 5 / 9)
  (h4 : (a + b + c).den = 1) : a = 5 / 4 :=
by
  sorry

end smallest_a_l102_102974


namespace one_head_two_tails_probability_l102_102162

noncomputable def probability_of_one_head_two_tails :=
  let total_outcomes := 8
  let favorable_outcomes := 3
  favorable_outcomes / total_outcomes

theorem one_head_two_tails_probability :
  probability_of_one_head_two_tails = 3 / 8 :=
by
  -- Proof would go here
  sorry

end one_head_two_tails_probability_l102_102162


namespace min_voters_for_tall_24_l102_102335

/-
There are 105 voters divided into 5 districts, each district divided into 7 sections, with each section having 3 voters.
A section is won by a majority vote. A district is won by a majority of sections. The contest is won by a majority of districts.
Tall won the contest. Prove that the minimum number of voters who could have voted for Tall is 24.
-/
noncomputable def min_voters_for_tall (total_voters districts sections voters_per_section : ℕ) (sections_needed_to_win_district districts_needed_to_win_contest : ℕ) : ℕ :=
  let voters_needed_per_section := voters_per_section / 2 + 1
  sections_needed_to_win_district * districts_needed_to_win_contest * voters_needed_per_section

theorem min_voters_for_tall_24 :
  min_voters_for_tall 105 5 7 3 4 3 = 24 :=
sorry

end min_voters_for_tall_24_l102_102335


namespace find_x_l102_102357

-- Definitions for the vectors a and b
def a : ℝ × ℝ := (3, 4)
def b : ℝ × ℝ := (2, 1)

-- Definition for the condition of parallel vectors
def are_parallel (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.2 = v1.2 * v2.1

-- Mathematical statement to prove
theorem find_x (x : ℝ) 
  (h_parallel : are_parallel (a.1 + x * b.1, a.2 + x * b.2) (a.1 - b.1, a.2 - b.2)) : 
  x = -1 :=
sorry

end find_x_l102_102357


namespace juniors_involved_in_sports_l102_102582

theorem juniors_involved_in_sports 
    (total_students : ℕ) (percentage_juniors : ℝ) (percentage_sports : ℝ) 
    (H1 : total_students = 500) 
    (H2 : percentage_juniors = 0.40) 
    (H3 : percentage_sports = 0.70) : 
    total_students * percentage_juniors * percentage_sports = 140 := 
by
  sorry

end juniors_involved_in_sports_l102_102582


namespace distance_between_parallel_lines_l102_102470

theorem distance_between_parallel_lines :
  let a := 4
  let b := -3
  let c1 := 2
  let c2 := -1
  let d := (abs (c1 - c2)) / (Real.sqrt (a^2 + b^2))
  d = 3 / 5 :=
by
  sorry

end distance_between_parallel_lines_l102_102470


namespace statement_1_equiv_statement_2_equiv_l102_102228

-- Statement 1
variable (A B C : Prop)

theorem statement_1_equiv : ((A ∨ B) → C) ↔ (A → C) ∧ (B → C) :=
by
  sorry

-- Statement 2
theorem statement_2_equiv : (A → (B ∧ C)) ↔ (A → B) ∧ (A → C) :=
by
  sorry

end statement_1_equiv_statement_2_equiv_l102_102228


namespace quadratic_has_one_solution_l102_102670

theorem quadratic_has_one_solution (q : ℚ) (hq : q ≠ 0) : 
  (∃ x, ∀ y, q*y^2 - 18*y + 8 = 0 → x = y) ↔ q = 81 / 8 :=
by
  sorry

end quadratic_has_one_solution_l102_102670


namespace sin_300_eq_neg_sqrt_three_div_two_l102_102405

theorem sin_300_eq_neg_sqrt_three_div_two :
  Real.sin (300 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_300_eq_neg_sqrt_three_div_two_l102_102405


namespace incorrect_statement_maximum_value_l102_102866

def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + b * x + c

theorem incorrect_statement_maximum_value :
  ∃ (a b c : ℝ), 
    (quadratic_function a b c 1 = -40) ∧
    (quadratic_function a b c (-1) = -8) ∧
    (quadratic_function a b c (-3) = 8) ∧
    (∀ (x_max : ℝ), (x_max = -b / (2 * a)) →
      (quadratic_function a b c x_max = 10) ∧
      (quadratic_function a b c x_max ≠ 8)) :=
by
  sorry

end incorrect_statement_maximum_value_l102_102866


namespace female_athletes_drawn_l102_102497

theorem female_athletes_drawn (total_athletes male_athletes female_athletes sample_size : ℕ)
  (h_total : total_athletes = male_athletes + female_athletes)
  (h_team : male_athletes = 48 ∧ female_athletes = 36)
  (h_sample_size : sample_size = 35) :
  (female_athletes * sample_size) / total_athletes = 15 :=
by
  sorry

end female_athletes_drawn_l102_102497


namespace area_smallest_region_enclosed_l102_102045

theorem area_smallest_region_enclosed {x y : ℝ} (circle_eq : x^2 + y^2 = 9) (abs_line_eq : y = |x|) :
  ∃ area, area = (9 * Real.pi) / 4 :=
by
  sorry

end area_smallest_region_enclosed_l102_102045


namespace population_reduction_l102_102132

theorem population_reduction (initial_population : ℕ) (final_population : ℕ) (left_percentage : ℝ)
    (bombardment_percentage : ℝ) :
    initial_population = 7145 →
    final_population = 4555 →
    left_percentage = 0.75 →
    bombardment_percentage = 100 - 84.96 →
    ∃ (x : ℝ), bombardment_percentage = (100 - x) := 
by
    sorry

end population_reduction_l102_102132


namespace square_side_lengths_l102_102041

theorem square_side_lengths (x y : ℝ) (h1 : x + y = 20) (h2 : x^2 - y^2 = 120) :
  (x = 13 ∧ y = 7) ∨ (x = 7 ∧ y = 13) :=
by {
  -- skip proof
  sorry
}

end square_side_lengths_l102_102041


namespace tan_pi_add_alpha_eq_two_l102_102857

theorem tan_pi_add_alpha_eq_two
  (α : ℝ)
  (h : Real.tan (Real.pi + α) = 2) :
  (2 * Real.sin α - Real.cos α) / (3 * Real.sin α + 2 * Real.cos α) = 3 / 8 :=
sorry

end tan_pi_add_alpha_eq_two_l102_102857


namespace minimum_monkeys_required_l102_102454

theorem minimum_monkeys_required (total_weight : ℕ) (weapon_max_weight : ℕ) (monkey_max_capacity : ℕ) 
  (num_monkeys : ℕ) (total_weapons : ℕ) 
  (H1 : total_weight = 600) 
  (H2 : weapon_max_weight = 30) 
  (H3 : monkey_max_capacity = 50) 
  (H4 : total_weapons = 600 / 30) 
  (H5 : num_monkeys = 23) : 
  num_monkeys ≤ (total_weapons * weapon_max_weight) / monkey_max_capacity :=
sorry

end minimum_monkeys_required_l102_102454


namespace unique_4_digit_number_l102_102233

theorem unique_4_digit_number (P E R U : ℕ) 
  (hP : 0 ≤ P ∧ P < 10)
  (hE : 0 ≤ E ∧ E < 10)
  (hR : 0 ≤ R ∧ R < 10)
  (hU : 0 ≤ U ∧ U < 10)
  (hPERU : 1000 ≤ (P * 1000 + E * 100 + R * 10 + U) ∧ (P * 1000 + E * 100 + R * 10 + U) < 10000) 
  (h_eq : (P * 1000 + E * 100 + R * 10 + U) = (P + E + R + U) ^ U) : 
  (P = 4) ∧ (E = 9) ∧ (R = 1) ∧ (U = 3) ∧ (P * 1000 + E * 100 + R * 10 + U = 4913) :=
sorry

end unique_4_digit_number_l102_102233


namespace average_age_of_students_l102_102320

variable (A : ℕ) -- We define A as a natural number representing average age

-- Define the conditions
def num_students : ℕ := 32
def staff_age : ℕ := 49
def new_average_age := A + 1

-- Definition of total age including the staff
def total_age_with_staff := 33 * new_average_age

-- Original condition stated as an equality
def condition : Prop := num_students * A + staff_age = total_age_with_staff

-- Theorem statement asserting that the average age A is 16 given the condition
theorem average_age_of_students : condition A → A = 16 :=
by sorry

end average_age_of_students_l102_102320


namespace appropriate_speech_length_l102_102969

-- Condition 1: Speech duration in minutes
def speech_duration_min : ℝ := 30
def speech_duration_max : ℝ := 45

-- Condition 2: Ideal rate of speech in words per minute
def ideal_rate : ℝ := 150

-- Question translated into Lean proof statement
theorem appropriate_speech_length (n : ℝ) (h : n = 5650) :
  speech_duration_min * ideal_rate ≤ n ∧ n ≤ speech_duration_max * ideal_rate :=
by
  sorry

end appropriate_speech_length_l102_102969


namespace pilot_speed_outbound_l102_102325

theorem pilot_speed_outbound (v : ℝ) (d : ℝ) (s_return : ℝ) (t_total : ℝ) 
    (return_time : ℝ := d / s_return) 
    (outbound_time : ℝ := t_total - return_time) 
    (speed_outbound : ℝ := d / outbound_time) :
  d = 1500 → s_return = 500 → t_total = 8 → speed_outbound = 300 :=
by
  intros hd hs ht
  sorry

end pilot_speed_outbound_l102_102325


namespace inverse_proportion_m_range_l102_102898

theorem inverse_proportion_m_range (m : ℝ) :
  (∀ x : ℝ, x < 0 → ∀ y1 y2 : ℝ, y1 = (1 - 2 * m) / x → y2 = (1 - 2 * m) / (x + 1) → y1 < y2) 
  ↔ (m > 1 / 2) :=
by sorry

end inverse_proportion_m_range_l102_102898


namespace find_all_pos_integers_l102_102394

theorem find_all_pos_integers (M : ℕ) (h1 : M > 0) (h2 : M < 10) :
  (5 ∣ (1989^M + M^1989)) ↔ (M = 1) ∨ (M = 4) :=
by
  sorry

end find_all_pos_integers_l102_102394


namespace M_subset_N_l102_102252

noncomputable def M_set : Set ℝ := { x | ∃ (k : ℤ), x = k / 4 + 1 / 4 }
noncomputable def N_set : Set ℝ := { x | ∃ (k : ℤ), x = k / 8 - 1 / 4 }

theorem M_subset_N : M_set ⊆ N_set :=
sorry

end M_subset_N_l102_102252


namespace sandwich_is_not_condiments_l102_102289

theorem sandwich_is_not_condiments (sandwich_weight condiments_weight : ℕ)
  (h1 : sandwich_weight = 150)
  (h2 : condiments_weight = 45) :
  (sandwich_weight - condiments_weight) / sandwich_weight * 100 = 70 := 
by sorry

end sandwich_is_not_condiments_l102_102289


namespace expr_comparison_l102_102773

-- Define the given condition
def eight_pow_2001 : ℝ := 8 * (64 : ℝ) ^ 1000

-- State the theorem
theorem expr_comparison : (65 : ℝ) ^ 1000 > eight_pow_2001 := by
  sorry

end expr_comparison_l102_102773


namespace number_of_disconnected_regions_l102_102226

theorem number_of_disconnected_regions (n : ℕ) (h : 2 ≤ n) : 
  ∀ R : ℕ → ℕ, (R 1 = 2) → 
  (∀ k, R k = k^2 - k + 2 → R (k + 1) = (k + 1)^2 - (k + 1) + 2) → 
  R n = n^2 - n + 2 :=
sorry

end number_of_disconnected_regions_l102_102226


namespace divisible_by_6_and_sum_15_l102_102078

theorem divisible_by_6_and_sum_15 (A B : ℕ) (h1 : A + B = 15) (h2 : (10 * A + B) % 6 = 0) :
  (A * B = 56) ∨ (A * B = 54) :=
by sorry

end divisible_by_6_and_sum_15_l102_102078


namespace chloe_profit_l102_102642

def cost_per_dozen : ℕ := 50
def sell_per_half_dozen : ℕ := 30
def total_dozens_sold : ℕ := 50

def total_cost (n: ℕ) : ℕ := n * cost_per_dozen
def total_revenue (n: ℕ) : ℕ := n * (sell_per_half_dozen * 2)
def profit (cost revenue : ℕ) : ℕ := revenue - cost

theorem chloe_profit : 
  profit (total_cost total_dozens_sold) (total_revenue total_dozens_sold) = 500 := 
by
  sorry

end chloe_profit_l102_102642


namespace find_values_l102_102016

noncomputable def equation_satisfaction (x y : ℝ) : Prop :=
  x^2 + (1 - y)^2 + (x - y)^2 = 1 / 3

theorem find_values (x y : ℝ) :
  equation_satisfaction x y → x = 1 / 3 ∧ y = 2 / 3 :=
by
  intro h
  sorry

end find_values_l102_102016


namespace sin_beta_value_sin2alpha_over_cos2alpha_plus_cos2alpha_value_l102_102261

open Real

noncomputable def problem_conditions (α β : ℝ) : Prop :=
  0 < α ∧ α < π / 2 ∧ 0 < β ∧ β < π / 2 ∧
  cos α = 3/5 ∧ cos (β + α) = 5/13

theorem sin_beta_value 
  {α β : ℝ} (h : problem_conditions α β) : 
  sin β = 16 / 65 :=
sorry

theorem sin2alpha_over_cos2alpha_plus_cos2alpha_value
  {α β : ℝ} (h : problem_conditions α β) : 
  (sin (2 * α)) / (cos α^2 + cos (2 * α)) = 12 :=
sorry

end sin_beta_value_sin2alpha_over_cos2alpha_plus_cos2alpha_value_l102_102261


namespace mediant_fraction_of_6_11_and_5_9_minimized_is_31_l102_102001

theorem mediant_fraction_of_6_11_and_5_9_minimized_is_31 
  (p q : ℕ) (h_pos : 0 < p ∧ 0 < q)
  (h_bounds : (6 : ℝ) / 11 < p / q ∧ p / q < 5 / 9)
  (h_min_q : ∀ r s : ℕ, (6 : ℝ) / 11 < r / s ∧ r / s < 5 / 9 → s ≥ q) :
  p + q = 31 :=
sorry

end mediant_fraction_of_6_11_and_5_9_minimized_is_31_l102_102001


namespace students_absent_afternoon_l102_102571

theorem students_absent_afternoon
  (morning_registered afternoon_registered total_students morning_absent : ℕ)
  (h_morning_registered : morning_registered = 25)
  (h_morning_absent : morning_absent = 3)
  (h_afternoon_registered : afternoon_registered = 24)
  (h_total_students : total_students = 42) :
  (afternoon_registered - (total_students - (morning_registered - morning_absent))) = 4 :=
by
  sorry

end students_absent_afternoon_l102_102571


namespace merchant_marked_price_l102_102481

-- Definitions
def list_price : ℝ := 100
def purchase_price (L : ℝ) : ℝ := 0.8 * L
def selling_price_with_discount (x : ℝ) : ℝ := 0.75 * x
def profit (purchase_price : ℝ) (selling_price : ℝ) : ℝ := selling_price - purchase_price
def desired_profit (selling_price : ℝ) : ℝ := 0.3 * selling_price

-- Statement to prove
theorem merchant_marked_price :
  ∃ (x : ℝ), 
    profit (purchase_price list_price) (selling_price_with_discount x) = desired_profit (selling_price_with_discount x) ∧
    x / list_price = 152.38 / 100 :=
sorry

end merchant_marked_price_l102_102481


namespace percentage_increase_visitors_l102_102549

theorem percentage_increase_visitors 
  (V_Oct : ℕ)
  (V_Nov V_Dec : ℕ)
  (h1 : V_Oct = 100)
  (h2 : V_Dec = V_Nov + 15)
  (h3 : V_Oct + V_Nov + V_Dec = 345) : 
  (V_Nov - V_Oct) * 100 / V_Oct = 15 := 
by 
  sorry

end percentage_increase_visitors_l102_102549


namespace crayons_per_row_correct_l102_102453

-- Declare the given conditions
def total_crayons : ℕ := 210
def num_rows : ℕ := 7

-- Define the expected number of crayons per row
def crayons_per_row : ℕ := 30

-- The desired proof statement: Prove that dividing total crayons by the number of rows yields the expected crayons per row.
theorem crayons_per_row_correct : total_crayons / num_rows = crayons_per_row :=
by sorry

end crayons_per_row_correct_l102_102453


namespace simplify_and_evaluate_expr_l102_102250

theorem simplify_and_evaluate_expr (x : ℝ) (h : x = Real.sqrt 2 - 1) : 
  ((x + 3) * (x - 3) - x * (x - 2)) = 2 * Real.sqrt 2 - 11 := by
  rw [h]
  sorry

end simplify_and_evaluate_expr_l102_102250


namespace quadratic_two_distinct_real_roots_l102_102373

theorem quadratic_two_distinct_real_roots (k : ℝ) :
    (∃ x : ℝ, 2 * k * x^2 + (8 * k + 1) * x + 8 * k = 0 ∧ 2 * k ≠ 0) →
    k > -1/16 ∧ k ≠ 0 :=
by
  intro h
  sorry

end quadratic_two_distinct_real_roots_l102_102373


namespace find_special_n_l102_102679

open Nat

theorem find_special_n (m : ℕ) (hm : m ≥ 3) :
  ∃ (n : ℕ), 
    (n = m^2 - 2) ∧ (∃ (k : ℕ), 1 ≤ k ∧ k < n ∧ 2 * (Nat.choose n k) = (Nat.choose n (k - 1) + Nat.choose n (k + 1))) :=
by
  sorry

end find_special_n_l102_102679


namespace prove_p_and_q_l102_102029

def p (m : ℝ) : Prop :=
  (∀ x : ℝ, x^2 + x + m > 0) → m > 1 / 4

def q (A B : ℝ) : Prop :=
  A > B ↔ Real.sin A > Real.sin B

theorem prove_p_and_q :
  (∀ m : ℝ, p m) ∧ (∀ A B : ℝ, q A B) :=
by
  sorry

end prove_p_and_q_l102_102029


namespace max_min_difference_l102_102503

variable (x y z : ℝ)

theorem max_min_difference :
  x + y + z = 3 →
  x^2 + y^2 + z^2 = 18 →
  (max z (-z)) - ((min z (-z))) = 6 :=
  by
    intros h1 h2
    sorry

end max_min_difference_l102_102503


namespace remaining_payment_l102_102059

theorem remaining_payment (deposit_percent : ℝ) (deposit_amount : ℝ) (total_percent : ℝ) (total_price : ℝ) :
  deposit_percent = 5 ∧ deposit_amount = 50 ∧ total_percent = 100 → total_price - deposit_amount = 950 :=
by {
  sorry
}

end remaining_payment_l102_102059


namespace sculpture_and_base_total_height_l102_102524

noncomputable def sculpture_height_ft : Nat := 2
noncomputable def sculpture_height_in : Nat := 10
noncomputable def base_height_in : Nat := 4
noncomputable def inches_per_foot : Nat := 12

theorem sculpture_and_base_total_height :
  (sculpture_height_ft * inches_per_foot + sculpture_height_in + base_height_in = 38) :=
by
  sorry

end sculpture_and_base_total_height_l102_102524


namespace polynomial_identity_equals_neg_one_l102_102698

theorem polynomial_identity_equals_neg_one
  (a a₁ a₂ a₃ : ℝ) :
  (∀ x : ℝ, (5 * x + 4)^3 = a + a₁ * x + a₂ * x^2 + a₃ * x^3) →
  (a + a₂) - (a₁ + a₃) = -1 :=
by
  intro h
  sorry

end polynomial_identity_equals_neg_one_l102_102698


namespace water_depth_when_upright_l102_102414
-- Import the entire Mathlib library

-- Define the conditions and question as a theorem
theorem water_depth_when_upright (height : ℝ) (diameter : ℝ) (horizontal_depth : ℝ) :
  height = 20 → diameter = 6 → horizontal_depth = 4 → water_depth = 5.3 :=
by
  intro h1 h2 h3
  -- The proof would go here, but we insert sorry to skip it
  sorry

end water_depth_when_upright_l102_102414


namespace f_eq_zero_of_le_zero_l102_102783

variable {R : Type*} [LinearOrderedField R]
variable {f : R → R}
variable (cond : ∀ x y : R, f (x + y) ≤ y * f x + f (f x))

theorem f_eq_zero_of_le_zero (x : R) (h : x ≤ 0) : f x = 0 :=
sorry

end f_eq_zero_of_le_zero_l102_102783


namespace minimum_difference_l102_102429

def even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k
def odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

theorem minimum_difference (x y z : ℤ) 
  (hx : even x) (hy : odd y) (hz : odd z)
  (hxy : x < y) (hyz : y < z) (hzx : z - x = 9) : y - x = 1 := 
sorry

end minimum_difference_l102_102429


namespace toby_breakfast_calories_l102_102522

noncomputable def calories_bread := 100
noncomputable def calories_peanut_butter_per_serving := 200
noncomputable def servings_peanut_butter := 2

theorem toby_breakfast_calories :
  1 * calories_bread + servings_peanut_butter * calories_peanut_butter_per_serving = 500 :=
by
  sorry

end toby_breakfast_calories_l102_102522


namespace impossible_result_l102_102322

noncomputable def f (a b : ℝ) (c : ℤ) (x : ℝ) : ℝ :=
  a * Real.sin x + b * x + c

theorem impossible_result (a b : ℝ) (c : ℤ) :
  ¬(f a b c 1 = 1 ∧ f a b c (-1) = 2) :=
by {
  sorry
}

end impossible_result_l102_102322


namespace solve_quadratic_eq_l102_102811

theorem solve_quadratic_eq (x : ℝ) : x^2 = 6 * x ↔ (x = 0 ∨ x = 6) := by
  sorry

end solve_quadratic_eq_l102_102811


namespace find_principal_l102_102787

noncomputable def principal_amount (P : ℝ) : Prop :=
  let r := 0.05
  let t := 2
  let SI := P * r * t
  let CI := P * (1 + r) ^ t - P
  CI - SI = 15

theorem find_principal : principal_amount 6000 :=
by
  simp [principal_amount]
  sorry

end find_principal_l102_102787


namespace balloons_popped_on_ground_l102_102377

def max_rate : Nat := 2
def max_time : Nat := 30
def zach_rate : Nat := 3
def zach_time : Nat := 40
def total_filled_balloons : Nat := 170

theorem balloons_popped_on_ground :
  (max_rate * max_time + zach_rate * zach_time) - total_filled_balloons = 10 :=
by
  sorry

end balloons_popped_on_ground_l102_102377


namespace find_table_height_l102_102864

theorem find_table_height (b r g h : ℝ) (h1 : h + b - g = 111) (h2 : h + r - b = 80) (h3 : h + g - r = 82) : h = 91 := 
by
  sorry

end find_table_height_l102_102864


namespace simplify_expression_l102_102817

variable {a b c : ℤ}

theorem simplify_expression (a b c : ℤ) : 3 * a - (4 * a - 6 * b - 3 * c) - 5 * (c - b) = -a + 11 * b - 2 * c :=
by
  sorry

end simplify_expression_l102_102817


namespace range_of_a_l102_102061

theorem range_of_a (a x : ℝ) (h_eq : 2 * x - 1 = x + a) (h_pos : x > 0) : a > -1 :=
sorry

end range_of_a_l102_102061


namespace tan_sum_identity_l102_102683

theorem tan_sum_identity (α : ℝ) (h : Real.tan α = 1 / 2) : Real.tan (α + π / 4) = 3 := 
by 
  sorry

end tan_sum_identity_l102_102683


namespace even_odd_decomposition_exp_l102_102671

variable (f g : ℝ → ℝ)

-- Conditions
def is_even (f : ℝ → ℝ) := ∀ x, f (-x) = f x
def is_odd (g : ℝ → ℝ) := ∀ x, g (-x) = -g x
def decomposition (f g : ℝ → ℝ) := ∀ x, f x + g x = Real.exp x

-- Main statement to prove
theorem even_odd_decomposition_exp (hf : is_even f) (hg : is_odd g) (hfg : decomposition f g) :
  f (Real.log 2) + g (Real.log (1 / 2)) = 1 / 2 := 
sorry

end even_odd_decomposition_exp_l102_102671


namespace boat_speed_in_still_water_l102_102244

-- Identifying the speeds of the boat in still water and the stream
variables (b s : ℝ)

-- Conditions stated in terms of equations
axiom boat_along_stream : b + s = 7
axiom boat_against_stream : b - s = 5

-- Prove that the boat speed in still water is 6 km/hr
theorem boat_speed_in_still_water : b = 6 :=
by
  sorry

end boat_speed_in_still_water_l102_102244


namespace fractional_eq_solution_range_l102_102521

theorem fractional_eq_solution_range (x m : ℝ) (h : (2 * x - m) / (x + 1) = 1) (hx : x < 0) : 
  m < -1 ∧ m ≠ -2 := 
by 
  sorry

end fractional_eq_solution_range_l102_102521


namespace invitation_methods_l102_102778

noncomputable def combination (n k : ℕ) : ℕ := Nat.choose n k

theorem invitation_methods (A B : Type) (students : Finset Type) (h : students.card = 10) :
  (∃ s : Finset Type, s.card = 6 ∧ A ∉ s ∧ B ∉ s) ∧ 
  (∃ t : Finset Type, t.card = 6 ∧ (A ∈ t ∨ B ∉ t)) →
  (combination 10 6 - combination 8 4 = 140) :=
by
  sorry

end invitation_methods_l102_102778


namespace correct_sum_is_826_l102_102342

theorem correct_sum_is_826 (ABC : ℕ)
  (h1 : 100 ≤ ABC ∧ ABC < 1000)  -- Ensuring ABC is a three-digit number
  (h2 : ∃ A B C : ℕ, ABC = 100 * A + 10 * B + C ∧ C = 6) -- Misread ones digit is 6
  (incorrect_sum : ℕ)
  (h3 : incorrect_sum = ABC + 57)  -- Sum obtained by Yoongi was 823
  (h4 : incorrect_sum = 823) : ABC + 57 + 3 = 826 :=  -- Correcting the sum considering the 6 to 9 error
by
  sorry

end correct_sum_is_826_l102_102342


namespace quadratic_inequality_solution_l102_102123

theorem quadratic_inequality_solution (x : ℝ) : 3 * x^2 - 5 * x - 8 > 0 ↔ x < -4/3 ∨ x > 2 :=
by
  sorry

end quadratic_inequality_solution_l102_102123


namespace parabola_directrix_value_l102_102346

noncomputable def parabola_p_value (p : ℝ) : Prop :=
(∀ y : ℝ, y^2 = 2 * p * (-2 - (-2)))

theorem parabola_directrix_value : parabola_p_value 4 :=
by
  -- proof steps here
  sorry

end parabola_directrix_value_l102_102346


namespace lines_intersection_l102_102478

theorem lines_intersection (n c : ℝ) : 
    (∀ x y : ℝ, y = n * x + 5 → y = 4 * x + c → (x, y) = (8, 9)) → 
    n + c = -22.5 := 
by
    intro h
    sorry

end lines_intersection_l102_102478


namespace sum_of_remainders_mod_500_l102_102436

theorem sum_of_remainders_mod_500 : 
  (5 ^ (5 ^ (5 ^ 5)) + 2 ^ (2 ^ (2 ^ 2))) % 500 = 49 := by
  sorry

end sum_of_remainders_mod_500_l102_102436


namespace additional_people_needed_l102_102912

theorem additional_people_needed (h₁ : ∀ p h : ℕ, (p * h = 40)) (h₂ : 5 * 8 = 40) : 7 - 5 = 2 :=
by
  sorry

end additional_people_needed_l102_102912


namespace find_fake_coin_in_two_weighings_l102_102680

theorem find_fake_coin_in_two_weighings (coins : Fin 8 → ℝ) (h : ∃ i : Fin 8, (∀ j ≠ i, coins i < coins j)) : 
  ∃! i : Fin 8, ∀ j ≠ i, coins i < coins j :=
by
  sorry

end find_fake_coin_in_two_weighings_l102_102680


namespace values_of_a_and_b_l102_102283

def is_root (a b x : ℝ) : Prop := x^2 - 2*a*x + b = 0

noncomputable def A : Set ℝ := {-1, 1}
noncomputable def B (a b : ℝ) : Set ℝ := {x | is_root a b x}

theorem values_of_a_and_b (a b : ℝ) (h_nonempty : Set.Nonempty (B a b)) (h_union : A ∪ B a b = A) :
  (a = 1 ∧ b = 1) ∨ (a = -1 ∧ b = 1) ∨ (a = 0 ∧ b = -1) :=
sorry

end values_of_a_and_b_l102_102283


namespace solve_for_x_l102_102294

theorem solve_for_x (x : ℝ) :
  (x - 2)^6 + (x - 6)^6 = 64 → x = 3 ∨ x = 5 :=
by
  intros h
  sorry

end solve_for_x_l102_102294


namespace find_ab_l102_102731

noncomputable def poly (x a b : ℝ) := x^4 + a * x^3 - 5 * x^2 + b * x - 6

theorem find_ab (a b : ℝ) (h : poly 2 a b = 0) : (a = 0 ∧ b = 4) :=
by
  sorry

end find_ab_l102_102731


namespace correct_avg_weight_of_class_l102_102829

theorem correct_avg_weight_of_class :
  ∀ (n : ℕ) (avg_wt : ℝ) (mis_A mis_B mis_C actual_A actual_B actual_C : ℝ),
  n = 30 →
  avg_wt = 60.2 →
  mis_A = 54 → actual_A = 64 →
  mis_B = 58 → actual_B = 68 →
  mis_C = 50 → actual_C = 60 →
  (n * avg_wt + (actual_A - mis_A) + (actual_B - mis_B) + (actual_C - mis_C)) / n = 61.2 :=
by
  intros n avg_wt mis_A mis_B mis_C actual_A actual_B actual_C h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end correct_avg_weight_of_class_l102_102829


namespace forty_percent_of_number_l102_102167

theorem forty_percent_of_number (N : ℝ) (h : (1 / 4) * (1 / 3) * (2 / 5) * N = 20) : 0.40 * N = 240 :=
by
  sorry

end forty_percent_of_number_l102_102167


namespace model_to_statue_ratio_inch_per_feet_model_inches_for_statue_feet_l102_102386

theorem model_to_statue_ratio_inch_per_feet (statue_height_ft : ℝ) (model_height_in : ℝ) :
  statue_height_ft = 120 → model_height_in = 6 → (120 / 6 = 20)
:= by
  intros h1 h2
  sorry

theorem model_inches_for_statue_feet (model_per_inch_feet : ℝ) :
  model_per_inch_feet = 20 → (30 / 20 = 1.5)
:= by
  intros h
  sorry

end model_to_statue_ratio_inch_per_feet_model_inches_for_statue_feet_l102_102386


namespace find_y_l102_102610

theorem find_y (y : ℕ) (hy1 : y % 9 = 0) (hy2 : y^2 > 200) (hy3 : y < 30) : y = 18 :=
sorry

end find_y_l102_102610


namespace range_of_a_l102_102972

noncomputable def piecewise_f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 2 then x^2 - 2 * a * x - 2 else x + (36 / x) - 6 * a

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, piecewise_f a x ≥ piecewise_f a 2) ↔ (2 ≤ a ∧ a ≤ 5) :=
by
  sorry

end range_of_a_l102_102972


namespace equal_intercepts_line_l102_102025

theorem equal_intercepts_line (x y : ℝ)
  (h1 : x + 2*y - 6 = 0) 
  (h2 : x - 2*y + 2 = 0) 
  (hx : x = 2) 
  (hy : y = 2) :
  (y = x) ∨ (x + y = 4) :=
sorry

end equal_intercepts_line_l102_102025


namespace smallest_integer_square_l102_102430

theorem smallest_integer_square (x : ℤ) (h : x^2 = 2 * x + 75) : x = -7 :=
  sorry

end smallest_integer_square_l102_102430


namespace inequality_of_f_log2015_l102_102024

noncomputable def f : ℝ → ℝ := sorry

theorem inequality_of_f_log2015 :
  (∀ x : ℝ, deriv f x > f x) →
  f (Real.log 2015) > 2015 * f 0 :=
by sorry

end inequality_of_f_log2015_l102_102024


namespace sin_A_value_l102_102913

variables {A B C a b c : ℝ}
variables {sin cos : ℝ → ℝ}

-- Conditions
axiom triangle_sides : ∀ (A B C: ℝ), ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0
axiom sin_cos_conditions : 3 * b * sin A = c * cos A + a * cos C

-- Proof statement
theorem sin_A_value (h : 3 * b * sin A = c * cos A + a * cos C) : sin A = 1 / 3 :=
by 
  sorry

end sin_A_value_l102_102913


namespace speed_in_still_water_l102_102197

-- Definitions of the conditions
def downstream_condition (v_m v_s : ℝ) : Prop := v_m + v_s = 6
def upstream_condition (v_m v_s : ℝ) : Prop := v_m - v_s = 3

-- The theorem to be proven
theorem speed_in_still_water (v_m v_s : ℝ) 
  (h1 : downstream_condition v_m v_s) 
  (h2 : upstream_condition v_m v_s) : v_m = 4.5 :=
by
  sorry

end speed_in_still_water_l102_102197


namespace remainder_proof_l102_102114

theorem remainder_proof : 1234567 % 12 = 7 := sorry

end remainder_proof_l102_102114


namespace min_function_value_l102_102809

theorem min_function_value (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (hxyz : x + y + z = 2) :
  (1/3 * x^3 + y^2 + z) = 13/12 :=
sorry

end min_function_value_l102_102809


namespace find_value_of_a_perpendicular_lines_l102_102246

theorem find_value_of_a_perpendicular_lines :
  ∃ (a : ℝ), (∀ (x y : ℝ), y = a * x - 2 → y = 2 * x + 1 → 
  (a * 2 = -1)) → a = -1/2 :=
by
  sorry

end find_value_of_a_perpendicular_lines_l102_102246


namespace problem_statement_l102_102277

variables {a b x : ℝ}

theorem problem_statement (h1 : x = a / b + 2) (h2 : a ≠ b) (h3 : b ≠ 0) : 
  (a + 2 * b) / (a - 2 * b) = x / (x - 4) := 
sorry

end problem_statement_l102_102277


namespace cricket_bat_profit_percentage_l102_102055

-- Definitions for the problem conditions
def selling_price : ℝ := 850
def profit : ℝ := 255
def cost_price : ℝ := selling_price - profit
def expected_profit_percentage : ℝ := 42.86

-- The theorem to be proven
theorem cricket_bat_profit_percentage : 
  (profit / cost_price) * 100 = expected_profit_percentage :=
by 
  sorry

end cricket_bat_profit_percentage_l102_102055


namespace total_videos_watched_l102_102112

variable (Ekon Uma Kelsey : ℕ)

theorem total_videos_watched
  (hKelsey : Kelsey = 160)
  (hKelsey_Ekon : Kelsey = Ekon + 43)
  (hEkon_Uma : Ekon = Uma - 17) :
  Kelsey + Ekon + Uma = 411 := by
  sorry

end total_videos_watched_l102_102112


namespace miles_from_second_friend_to_work_l102_102043
variable (distance_to_first_friend := 8)
variable (distance_to_second_friend := distance_to_first_friend / 2)
variable (total_distance_to_second_friend := distance_to_first_friend + distance_to_second_friend)
variable (distance_to_work := 3 * total_distance_to_second_friend)

theorem miles_from_second_friend_to_work :
  distance_to_work = 36 := 
by
  sorry

end miles_from_second_friend_to_work_l102_102043


namespace volume_units_correct_l102_102368

/-- Definition for the volume of a bottle of coconut juice in milliliters (200 milliliters). -/
def volume_of_coconut_juice := 200 

/-- Definition for the volume of an electric water heater in liters (50 liters). -/
def volume_of_electric_water_heater := 50 

/-- Prove that the volume of a bottle of coconut juice is measured in milliliters (200 milliliters)
    and the volume of an electric water heater is measured in liters (50 liters).
-/
theorem volume_units_correct :
  volume_of_coconut_juice = 200 ∧ volume_of_electric_water_heater = 50 :=
sorry

end volume_units_correct_l102_102368


namespace crayons_loss_l102_102273

def initial_crayons : ℕ := 479
def final_crayons : ℕ := 134
def crayons_lost : ℕ := initial_crayons - final_crayons

theorem crayons_loss :
  crayons_lost = 345 := by
  sorry

end crayons_loss_l102_102273


namespace evaluate_expression_l102_102385

theorem evaluate_expression (x y : ℝ) (h1 : x = 3) (h2 : y = 0) : y * (y - 3 * x) = 0 :=
by sorry

end evaluate_expression_l102_102385


namespace donation_to_second_orphanage_l102_102693

variable (total_donation : ℝ) (first_donation : ℝ) (third_donation : ℝ)

theorem donation_to_second_orphanage :
  total_donation = 650 ∧ first_donation = 175 ∧ third_donation = 250 →
  (total_donation - first_donation - third_donation = 225) := by
  sorry

end donation_to_second_orphanage_l102_102693


namespace quadratic_grid_fourth_column_l102_102628

theorem quadratic_grid_fourth_column 
  (grid : ℕ → ℕ → ℝ)
  (row_quadratic : ∀ i : ℕ, (∃ a b c : ℝ, ∀ n : ℕ, grid i n = a * n^2 + b * n + c))
  (col_quadratic : ∀ j : ℕ, j ≤ 3 → (∃ a b c : ℝ, ∀ n : ℕ, grid n j = a * n^2 + b * n + c)) :
  ∃ a b c : ℝ, ∀ n : ℕ, grid n 4 = a * n^2 + b * n + c := 
sorry

end quadratic_grid_fourth_column_l102_102628


namespace necessary_not_sufficient_to_form_triangle_l102_102707

-- Define the vectors and the condition
variables (a b c : ℝ × ℝ)

-- Define the condition that these vectors form a closed loop (triangle)
def forms_closed_loop (a b c : ℝ × ℝ) : Prop :=
  a + b + c = (0, 0)

-- Prove that the condition is necessary but not sufficient
theorem necessary_not_sufficient_to_form_triangle :
  forms_closed_loop a b c → ∃ (x : ℝ × ℝ), a ≠ x ∧ b ≠ -2 * x ∧ c ≠ x :=
sorry

end necessary_not_sufficient_to_form_triangle_l102_102707


namespace count_ordered_pairs_l102_102918

theorem count_ordered_pairs (d n : ℕ) (h₁ : d ≥ 35) (h₂ : n > 0) 
    (h₃ : 45 + 2 * n < 120)
    (h₄ : ∃ a b : ℕ, 10 * a + b = 30 + n ∧ 10 * b + a = 35 + n ∧ a ≤ 9 ∧ b ≤ 9) :
    ∃ k : ℕ, -- number of valid ordered pairs (d, n)
    sorry := sorry

end count_ordered_pairs_l102_102918


namespace poly_comp_eq_l102_102688

variable {K : Type*} [Field K]

theorem poly_comp_eq {Q1 Q2 : Polynomial K} (P : Polynomial K) (hP : ¬P.degree = 0) :
  Q1.comp P = Q2.comp P → Q1 = Q2 :=
by
  intro h
  sorry

end poly_comp_eq_l102_102688


namespace right_triangle_leg_squared_l102_102934

variable (a b c : ℝ)

theorem right_triangle_leg_squared (h1 : c = a + 2) (h2 : a^2 + b^2 = c^2) : b^2 = 4 * (a + 1) :=
by
  sorry

end right_triangle_leg_squared_l102_102934


namespace ordered_pair_solution_l102_102091

theorem ordered_pair_solution :
  ∃ (x y : ℚ), 
  (3 * x - 2 * y = (6 - 2 * x) + (6 - 2 * y)) ∧
  (x + 3 * y = (2 * x + 1) - (2 * y + 1)) ∧
  x = 12 / 5 ∧
  y = 12 / 25 :=
by
  sorry

end ordered_pair_solution_l102_102091


namespace union_is_equivalent_l102_102861

def A (x : ℝ) : Prop := x ^ 2 - x - 6 ≤ 0
def B (x : ℝ) : Prop := 0 < x ∧ x < 4

theorem union_is_equivalent (x : ℝ) :
  (A x ∨ B x) ↔ (-2 ≤ x ∧ x < 4) :=
sorry

end union_is_equivalent_l102_102861


namespace area_of_OBEC_is_25_l102_102619

noncomputable def area_OBEC : ℝ :=
  let A := (20 / 3, 0)
  let B := (0, 20)
  let C := (10, 0)
  let E := (5, 5)
  let O := (0, 0)
  let area_triangle (P Q R : ℝ × ℝ) : ℝ :=
    (1 / 2) * abs ((Q.1 - P.1) * (R.2 - P.2) - (R.1 - P.1) * (Q.2 - P.2))
  area_triangle O B E - area_triangle O E C

theorem area_of_OBEC_is_25 :
  area_OBEC = 25 := 
by
  sorry

end area_of_OBEC_is_25_l102_102619


namespace total_prep_time_is_8_l102_102117

-- Defining the conditions
def prep_vocab_sentence_eq := 3
def prep_analytical_writing := 2
def prep_quantitative_reasoning := 3

-- Stating the total preparation time
def total_prep_time := prep_vocab_sentence_eq + prep_analytical_writing + prep_quantitative_reasoning

-- The Lean statement of the mathematical proof problem
theorem total_prep_time_is_8 : total_prep_time = 8 := by
  sorry

end total_prep_time_is_8_l102_102117


namespace parentheses_removal_correct_l102_102777

theorem parentheses_removal_correct (x y : ℝ) : -(x^2 + y^2) = -x^2 - y^2 :=
by
  sorry

end parentheses_removal_correct_l102_102777


namespace x_plus_y_equals_22_l102_102070

theorem x_plus_y_equals_22 (x y : ℕ) (h1 : 2^x = 4^(y + 2)) (h2 : 27^y = 9^(x - 7)) : x + y = 22 := 
sorry

end x_plus_y_equals_22_l102_102070


namespace no_common_root_of_quadratics_l102_102366

theorem no_common_root_of_quadratics (a b c d : ℝ) (h : 0 < a ∧ a < b ∧ b < c ∧ c < d) :
  ¬ ∃ x₀ : ℝ, (x₀^2 + b * x₀ + c = 0 ∧ x₀^2 + a * x₀ + d = 0) := 
by
  sorry

end no_common_root_of_quadratics_l102_102366


namespace power_of_fraction_l102_102192

theorem power_of_fraction : ((1/3)^5 = (1/243)) :=
by
  sorry

end power_of_fraction_l102_102192


namespace total_dogs_l102_102831

axiom brown_dogs : ℕ
axiom white_dogs : ℕ
axiom black_dogs : ℕ

theorem total_dogs (b w bl : ℕ) (h1 : b = 20) (h2 : w = 10) (h3 : bl = 15) : (b + w + bl) = 45 :=
by {
  sorry
}

end total_dogs_l102_102831


namespace balance_blue_balls_l102_102736

variable (G Y W B : ℝ)

-- Define the conditions
def condition1 : 4 * G = 8 * B := sorry
def condition2 : 3 * Y = 8 * B := sorry
def condition3 : 4 * B = 3 * W := sorry

-- Prove the required balance of 3G + 4Y + 3W
theorem balance_blue_balls (h1 : 4 * G = 8 * B) (h2 : 3 * Y = 8 * B) (h3 : 4 * B = 3 * W) :
  3 * (2 * B) + 4 * (8 / 3 * B) + 3 * (4 / 3 * B) = 62 / 3 * B := by
  sorry

end balance_blue_balls_l102_102736


namespace circle_symmetric_about_line_l102_102721

theorem circle_symmetric_about_line :
  ∃ b : ℝ, (∀ (x y : ℝ), x^2 + y^2 + 2*x - 4*y + 4 = 0 → y = 2*x + b) → b = 4 :=
by
  sorry

end circle_symmetric_about_line_l102_102721


namespace least_number_division_remainder_4_l102_102409

theorem least_number_division_remainder_4 : 
  ∃ n : Nat, (n % 6 = 4) ∧ (n % 130 = 4) ∧ (n % 9 = 4) ∧ (n % 18 = 4) ∧ n = 2344 :=
by
  sorry

end least_number_division_remainder_4_l102_102409


namespace fraction_of_boys_among_attendees_l102_102446

def boys : ℕ := sorry
def girls : ℕ := boys
def teachers : ℕ := boys / 2

def boys_attending : ℕ := (4 * boys) / 5
def girls_attending : ℕ := girls / 2
def teachers_attending : ℕ := teachers / 10

theorem fraction_of_boys_among_attendees :
  (boys_attending : ℚ) / (boys_attending + girls_attending + teachers_attending) = 16 / 27 := sorry

end fraction_of_boys_among_attendees_l102_102446


namespace sufficient_condition_l102_102815

variable (a : ℝ)

theorem sufficient_condition (h : ∀ x : ℝ, -1 ≤ x → x ≤ 2 → x^2 - a ≥ 0) : a ≤ -1 := 
sorry

end sufficient_condition_l102_102815


namespace remainder_modulo_l102_102897

theorem remainder_modulo (n : ℤ) (h : n % 50 = 23) : (3 * n - 5) % 15 = 4 := 
by 
  sorry

end remainder_modulo_l102_102897


namespace combinations_count_l102_102316

theorem combinations_count:
  let valid_a (a: ℕ) := a < 1000 ∧ a % 29 = 7
  let valid_b (b: ℕ) := b < 1000 ∧ b % 47 = 22
  let valid_c (c: ℕ) (a b: ℕ) := c < 1000 ∧ c = (a + b) % 23 
  ∃ (a b c: ℕ), valid_a a ∧ valid_b b ∧ valid_c c a b :=
  sorry

end combinations_count_l102_102316


namespace largest_n_exists_l102_102889

theorem largest_n_exists :
  ∃ (n : ℕ), 
  (∀ (x y z : ℕ), n^2 = 2*x^2 + 2*y^2 + 2*z^2 + 4*x*y + 4*y*z + 4*z*x + 6*x + 6*y + 6*z - 14) → n = 9 :=
sorry

end largest_n_exists_l102_102889


namespace sum_of_roots_l102_102152

theorem sum_of_roots : 
  ∀ x1 x2 : ℝ, 
  (x1^2 + 2023*x1 = 2024 ∧ x2^2 + 2023*x2 = 2024) → 
  x1 + x2 = -2023 := 
by 
  sorry

end sum_of_roots_l102_102152


namespace xenia_weekly_earnings_l102_102021

theorem xenia_weekly_earnings
  (hours_week_1 : ℕ)
  (hours_week_2 : ℕ)
  (week2_additional_earnings : ℕ)
  (hours_week_3 : ℕ)
  (bonus_week_3 : ℕ)
  (hourly_wage : ℚ)
  (earnings_week_1 : ℚ)
  (earnings_week_2 : ℚ)
  (earnings_week_3 : ℚ)
  (total_earnings : ℚ) :
  hours_week_1 = 18 →
  hours_week_2 = 25 →
  week2_additional_earnings = 60 →
  hours_week_3 = 28 →
  bonus_week_3 = 30 →
  hourly_wage = (60 : ℚ) / (25 - 18) →
  earnings_week_1 = hours_week_1 * hourly_wage →
  earnings_week_2 = hours_week_2 * hourly_wage →
  earnings_week_2 = earnings_week_1 + 60 →
  earnings_week_3 = hours_week_3 * hourly_wage + 30 →
  total_earnings = earnings_week_1 + earnings_week_2 + earnings_week_3 →
  hourly_wage = (857 : ℚ) / 1000 ∧
  total_earnings = (63947 : ℚ) / 100
:= by
  intros h1 h2 h3 h4 h5 hw he1 he2 he2_60 he3 hte
  sorry

end xenia_weekly_earnings_l102_102021


namespace find_focus_parabola_l102_102292

theorem find_focus_parabola
  (x y : ℝ) 
  (h₁ : y = 9 * x^2 + 6 * x - 4) :
  ∃ (h k p : ℝ), (x + 1/3)^2 = 1/3 * (y + 5) ∧ 4 * p = 1/3 ∧ h = -1/3 ∧ k = -5 ∧ (h, k + p) = (-1/3, -59/12) :=
sorry

end find_focus_parabola_l102_102292


namespace hotel_loss_l102_102928

theorem hotel_loss (operations_expenses : ℝ) (payment_fraction : ℝ) (total_payment : ℝ) (loss : ℝ) 
  (hOpExp : operations_expenses = 100) 
  (hPayFr : payment_fraction = 3 / 4)
  (hTotalPay : total_payment = payment_fraction * operations_expenses) 
  (hLossCalc : loss = operations_expenses - total_payment) : 
  loss = 25 := 
by 
  sorry

end hotel_loss_l102_102928


namespace arcsin_cos_solution_l102_102643

theorem arcsin_cos_solution (x : ℝ) (h : -π/2 ≤ x/3 ∧ x/3 ≤ π/2) :
  x = 3*π/10 ∨ x = 3*π/8 := 
sorry

end arcsin_cos_solution_l102_102643


namespace female_democrats_count_l102_102154

theorem female_democrats_count :
  ∃ (F : ℕ) (M : ℕ),
    F + M = 750 ∧
    (F / 2) + (M / 4) = 250 ∧
    1 / 3 * 750 = 250 ∧
    F / 2 = 125 := sorry

end female_democrats_count_l102_102154


namespace quadratic_inequality_solution_l102_102155

theorem quadratic_inequality_solution (m : ℝ) : 
  (∀ x : ℝ, x^2 + m * x + 1 ≥ 0) ↔ (-2 ≤ m ∧ m ≤ 2) :=
by
  sorry

end quadratic_inequality_solution_l102_102155


namespace complete_square_solution_l102_102960

-- Define the initial equation 
def equation_to_solve (x : ℝ) : Prop := x^2 - 4 * x = 6

-- Define the transformed equation after completing the square
def transformed_equation (x : ℝ) : Prop := (x - 2)^2 = 10

-- Prove that solving the initial equation using completing the square results in the transformed equation
theorem complete_square_solution : 
  ∀ x : ℝ, equation_to_solve x → transformed_equation x := 
by
  -- Proof will be provided here
  sorry

end complete_square_solution_l102_102960


namespace unique_solution_fraction_l102_102166

theorem unique_solution_fraction (x : ℝ) :
  (2 * x^2 - 10 * x + 8 ≠ 0) → 
  (∃! (x : ℝ), (3 * x^2 - 15 * x + 12) / (2 * x^2 - 10 * x + 8) = x - 4) :=
by
  sorry

end unique_solution_fraction_l102_102166


namespace min_guests_at_banquet_l102_102920

-- Definitions based on conditions
def total_food : ℕ := 675
def vegetarian_food : ℕ := 195
def pescatarian_food : ℕ := 220
def carnivorous_food : ℕ := 260

def max_vegetarian_per_guest : ℚ := 3
def max_pescatarian_per_guest : ℚ := 2.5
def max_carnivorous_per_guest : ℚ := 4

-- Definition based on the question and the correct answer
def minimum_number_of_guests : ℕ := 218

-- Lean statement to prove the problem
theorem min_guests_at_banquet :
  195 / 3 + 220 / 2.5 + 260 / 4 = 218 :=
by sorry

end min_guests_at_banquet_l102_102920


namespace interval_width_and_count_l102_102834

def average_income_intervals := [3000, 4000, 5000, 6000, 7000]
def frequencies := [5, 9, 4, 2]

theorem interval_width_and_count:
  (average_income_intervals[1] - average_income_intervals[0] = 1000) ∧
  (frequencies.length = 4) :=
by
  sorry

end interval_width_and_count_l102_102834


namespace closest_point_on_line_l102_102908

open Real

theorem closest_point_on_line (x y : ℝ) (h_line : y = 4 * x - 3) (h_closest : ∀ p : ℝ × ℝ, (p.snd - -1)^2 + (p.fst - 2)^2 ≥ (y - -1)^2 + (x - 2)^2) :
  x = 10 / 17 ∧ y = 31 / 17 :=
sorry

end closest_point_on_line_l102_102908


namespace problem1_problem2_l102_102238

-- Problem 1
theorem problem1 (a b : ℝ) (h : a ≠ 0) : 
  (a - b^2 / a) / ((a^2 + 2 * a * b + b^2) / a) = (a - b) / (a + b) :=
by
  sorry

-- Problem 2
theorem problem2 (x : ℝ) : 
  (6 - 2 * x ≥ 4) ∧ ((1 + 2 * x) / 3 > x - 1) ↔ (x ≤ 1) :=
by
  sorry

end problem1_problem2_l102_102238


namespace eugene_total_pencils_l102_102144

-- Define the initial number of pencils Eugene has
def initial_pencils : ℕ := 51

-- Define the number of pencils Joyce gives to Eugene
def pencils_from_joyce : ℕ := 6

-- Define the expected total number of pencils
def expected_total_pencils : ℕ := 57

-- Theorem to prove the total number of pencils Eugene has
theorem eugene_total_pencils : initial_pencils + pencils_from_joyce = expected_total_pencils := 
by sorry

end eugene_total_pencils_l102_102144


namespace correct_formula_l102_102645

theorem correct_formula {x y : ℕ} : 
  (x = 0 ∧ y = 100) ∨
  (x = 1 ∧ y = 90) ∨
  (x = 2 ∧ y = 70) ∨
  (x = 3 ∧ y = 40) ∨
  (x = 4 ∧ y = 0) →
  y = 100 - 5 * x - 5 * x^2 :=
by
  sorry

end correct_formula_l102_102645


namespace gcd_72_108_150_l102_102071

theorem gcd_72_108_150 : Nat.gcd (Nat.gcd 72 108) 150 = 6 := by
  sorry

end gcd_72_108_150_l102_102071


namespace compare_abc_l102_102951

noncomputable def a : ℝ := 2^(4/3)
noncomputable def b : ℝ := 4^(2/5)
noncomputable def c : ℝ := 5^(2/3)

theorem compare_abc : c > a ∧ a > b := 
by
  sorry

end compare_abc_l102_102951


namespace solution_set_of_inequality_l102_102381

theorem solution_set_of_inequality : 
  {x : ℝ | |x|^3 - 2 * x^2 - 4 * |x| + 3 < 0} = 
  { x : ℝ | -3 < x ∧ x < -1 } ∪ { x : ℝ | 1 < x ∧ x < 3 } := 
by
  sorry

end solution_set_of_inequality_l102_102381


namespace average_chem_math_l102_102538

theorem average_chem_math (P C M : ℕ) (h : P + C + M = P + 180) : (C + M) / 2 = 90 :=
  sorry

end average_chem_math_l102_102538


namespace birds_flew_up_l102_102733

theorem birds_flew_up (initial_birds new_birds total_birds : ℕ) 
    (h_initial : initial_birds = 29) 
    (h_total : total_birds = 42) : 
    new_birds = total_birds - initial_birds := 
by 
    sorry

end birds_flew_up_l102_102733


namespace max_abcsum_l102_102492

theorem max_abcsum (a b c : ℕ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_eq : a * b^2 * c^3 = 1350) : 
  a + b + c ≤ 154 :=
sorry

end max_abcsum_l102_102492


namespace sum_of_x_coordinates_of_intersections_l102_102190

def g : ℝ → ℝ := sorry  -- Definition of g is unspecified but it consists of five line segments.

theorem sum_of_x_coordinates_of_intersections 
  (h1 : ∃ x1, g x1 = x1 - 2 ∧ (x1 = -2 ∨ x1 = 1 ∨ x1 = 4))
  (h2 : ∃ x2, g x2 = x2 - 2 ∧ (x2 = -2 ∨ x2 = 1 ∨ x2 = 4))
  (h3 : ∃ x3, g x3 = x3 - 2 ∧ (x3 = -2 ∨ x3 = 1 ∨ x3 = 4)) 
  (hx1x2x3 : x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3) :
  x1 + x2 + x3 = 3 := by
  -- Proof here
  sorry

end sum_of_x_coordinates_of_intersections_l102_102190


namespace number_of_clients_l102_102400

-- Definitions from the problem
def cars : ℕ := 18
def selections_per_client : ℕ := 3
def selections_per_car : ℕ := 3

-- Theorem statement: Prove that the number of clients is 18
theorem number_of_clients (total_cars : ℕ) (cars_selected_by_each_client : ℕ) (each_car_selected : ℕ)
  (h_cars : total_cars = cars)
  (h_select_each : cars_selected_by_each_client = selections_per_client)
  (h_selected_car : each_car_selected = selections_per_car) :
  (total_cars * each_car_selected) / cars_selected_by_each_client = 18 :=
by
  rw [h_cars, h_select_each, h_selected_car]
  sorry

end number_of_clients_l102_102400


namespace total_hours_charged_l102_102271

variables (K P M : ℕ)

theorem total_hours_charged (h1 : P = 2 * K) 
                            (h2 : P = (1/3 : ℚ) * (M : ℚ)) 
                            (h3 : M = K + 85) : 
  K + P + M = 153 := 
by 
  sorry

end total_hours_charged_l102_102271


namespace calculate_otimes_l102_102474

def otimes (a b : ℚ) : ℚ := (a + b) / (a - b)

theorem calculate_otimes :
  otimes (otimes 8 6) 12 = -19 / 5 := by
  sorry

end calculate_otimes_l102_102474


namespace Carol_optimal_choice_l102_102616

noncomputable def Alice_choices := Set.Icc 0 (1 : ℝ)
noncomputable def Bob_choices := Set.Icc (1 / 3) (3 / 4 : ℝ)

theorem Carol_optimal_choice : 
  ∀ (c : ℝ), c ∈ Set.Icc 0 1 → 
  (∃! c, c = 7 / 12) := 
sorry

end Carol_optimal_choice_l102_102616


namespace avg_ac_l102_102723

-- Define the ages of a, b, and c as variables A, B, and C
variables (A B C : ℕ)

-- Define the conditions
def avg_abc (A B C : ℕ) : Prop := (A + B + C) / 3 = 26
def age_b (B : ℕ) : Prop := B = 20

-- State the theorem to prove
theorem avg_ac {A B C : ℕ} (h1 : avg_abc A B C) (h2 : age_b B) : (A + C) / 2 = 29 := 
by sorry

end avg_ac_l102_102723


namespace expand_polynomial_l102_102425

theorem expand_polynomial :
  (x^2 - 3 * x + 3) * (x^2 + 3 * x + 1) = x^4 - 5 * x^2 + 6 * x + 3 :=
by
  sorry

end expand_polynomial_l102_102425


namespace costForFirstKgs_l102_102208

noncomputable def applePrice (l : ℝ) (q : ℝ) (x : ℝ) (totalWeight : ℝ) : ℝ :=
  if totalWeight <= x then l * totalWeight else l * x + q * (totalWeight - x)

theorem costForFirstKgs (l q x : ℝ) :
  l = 10 ∧ q = 11 ∧ (applePrice l q x 33 = 333) ∧ (applePrice l q x 36 = 366) ∧ (applePrice l q 15 15 = 150) → x = 30 := 
by
  sorry

end costForFirstKgs_l102_102208


namespace exists_sum_of_three_l102_102527

theorem exists_sum_of_three {a b c d : ℕ} 
  (h1 : Nat.Coprime a b) 
  (h2 : Nat.Coprime a c) 
  (h3 : Nat.Coprime a d)
  (h4 : Nat.Coprime b c) 
  (h5 : Nat.Coprime b d) 
  (h6 : Nat.Coprime c d) 
  (h7 : a * b + c * d = a * c - 10 * b * d) :
  ∃ x y z, (x = a ∨ x = b ∨ x = c ∨ x = d) ∧ 
           (y = a ∨ y = b ∨ y = c ∨ y = d) ∧ 
           (z = a ∨ z = b ∨ z = c ∨ z = d) ∧ 
           x ≠ y ∧ x ≠ z ∧ y ≠ z ∧ 
           (x = y + z ∨ y = x + z ∨ z = x + y) :=
by
  sorry

end exists_sum_of_three_l102_102527


namespace combinations_of_three_toppings_l102_102557

def number_of_combinations (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem combinations_of_three_toppings : number_of_combinations 10 3 = 120 := by
  sorry

end combinations_of_three_toppings_l102_102557


namespace tracy_additional_miles_l102_102758

def total_distance : ℕ := 1000
def michelle_distance : ℕ := 294
def twice_michelle_distance : ℕ := 2 * michelle_distance
def katie_distance : ℕ := michelle_distance / 3
def tracy_distance := total_distance - (michelle_distance + katie_distance)
def additional_miles := tracy_distance - twice_michelle_distance

-- The statement to prove:
theorem tracy_additional_miles : additional_miles = 20 := by
  sorry

end tracy_additional_miles_l102_102758


namespace total_blue_points_l102_102340

variables (a b c d : ℕ)

theorem total_blue_points (h1 : a * b = 56) (h2 : c * d = 50) (h3 : a + b = c + d) :
  a + b = 15 :=
sorry

end total_blue_points_l102_102340


namespace xyz_sum_l102_102603

theorem xyz_sum (x y z : ℝ) (h1 : y = 3 * x) (h2 : z = 4 * y) : x + y + z = 16 * x :=
by
  sorry

end xyz_sum_l102_102603


namespace at_least_one_true_l102_102230

-- Definitions (Conditions)
variables (p q : Prop)

-- Statement
theorem at_least_one_true (h : p ∨ q) : p ∨ q := by
  sorry

end at_least_one_true_l102_102230


namespace ratio_girls_to_boys_l102_102299

-- Define the number of students and conditions
def num_students : ℕ := 25
def girls_more_than_boys : ℕ := 3

-- Define the variables
variables (g b : ℕ)

-- Define the conditions
def total_students := g + b = num_students
def girls_boys_relationship := b = g - girls_more_than_boys

-- Lean theorem statement
theorem ratio_girls_to_boys (g b : ℕ) (h1 : total_students g b) (h2 : girls_boys_relationship g b) : (g : ℚ) / b = 14 / 11 :=
sorry

end ratio_girls_to_boys_l102_102299


namespace gcd_180_450_l102_102000

theorem gcd_180_450 : Nat.gcd 180 450 = 90 :=
by
  sorry

end gcd_180_450_l102_102000


namespace value_of_f_5_l102_102428

-- Define the function f
def f (x y : ℕ) : ℕ := 2 * x ^ 2 + y

-- Given conditions
variable (some_value : ℕ)
axiom h1 : f some_value 52 = 60
axiom h2 : f 5 52 = 102

-- Proof statement
theorem value_of_f_5 : f 5 52 = 102 := by
  sorry

end value_of_f_5_l102_102428


namespace five_more_than_three_in_pages_l102_102751

def pages := (List.range 512).map (λ n => n + 1)

def count_digit (d : Nat) (n : Nat) : Nat :=
  if n = 0 then 0
  else if n % 10 = d then 1 + count_digit d (n / 10)
  else count_digit d (n / 10)

def total_digit_count (d : Nat) (l : List Nat) : Nat :=
  l.foldl (λ acc x => acc + count_digit d x) 0

theorem five_more_than_three_in_pages :
  total_digit_count 5 pages - total_digit_count 3 pages = 22 := 
by 
  sorry

end five_more_than_three_in_pages_l102_102751


namespace abs_inequality_solution_l102_102624

theorem abs_inequality_solution (x : ℝ) : 
  (|x - 2| + |x + 3| < 8) → x ∈ Set.Ioo (-4.5) (3.5) :=
by
  sorry

end abs_inequality_solution_l102_102624


namespace find_c_l102_102816

theorem find_c (b c : ℝ) (h : (∀ x : ℝ, (x + 3) * (x + b) = x^2 + c * x + 12)) : 
  c = 7 := 
by {
  sorry
}

end find_c_l102_102816


namespace final_weight_is_200_l102_102303

def initial_weight : ℕ := 220
def percentage_lost : ℕ := 10
def weight_gained : ℕ := 2

theorem final_weight_is_200 :
  initial_weight - (initial_weight * percentage_lost / 100) + weight_gained = 200 := by
  sorry

end final_weight_is_200_l102_102303


namespace mean_greater_than_median_by_six_l102_102782

theorem mean_greater_than_median_by_six (x : ℕ) : 
  let mean := (x + (x + 2) + (x + 4) + (x + 7) + (x + 37)) / 5
  let median := x + 4
  mean - median = 6 :=
by
  sorry

end mean_greater_than_median_by_six_l102_102782


namespace distance_C_distance_BC_l102_102354

variable (A B C D : ℕ)

theorem distance_C
  (hA : A = 350)
  (hAB : A + B = 600)
  (hABCD : A + B + C + D = 1500)
  (hD : D = 275)
  : C = 625 :=
by
  sorry

theorem distance_BC
  (A B C D : ℕ)
  (hA : A = 350)
  (hAB : A + B = 600)
  (hABCD : A + B + C + D = 1500)
  (hD : D = 275)
  : B + C = 875 :=
by
  sorry

end distance_C_distance_BC_l102_102354


namespace find_coins_l102_102308

-- Definitions based on conditions
structure Wallet where
  coin1 : ℕ
  coin2 : ℕ
  h_total_value : coin1 + coin2 = 15
  h_not_five : coin1 ≠ 5 ∨ coin2 ≠ 5

-- Theorem statement based on the proof problem
theorem find_coins (w : Wallet) : (w.coin1 = 5 ∧ w.coin2 = 10) ∨ (w.coin1 = 10 ∧ w.coin2 = 5) := by
  sorry

end find_coins_l102_102308


namespace simplify_product_l102_102046

theorem simplify_product : 
  18 * (8 / 15) * (2 / 27) = 32 / 45 :=
by
  sorry

end simplify_product_l102_102046


namespace smallest_positive_integer_with_20_divisors_is_432_l102_102953

-- Define the condition that a number n has exactly 20 positive divisors
def has_exactly_20_divisors (n : ℕ) : Prop :=
  ∃ (a₁ a₂ : ℕ), a₁ + 1 = 5 ∧ a₂ + 1 = 4 ∧
                n = 2^a₁ * 3^a₂

-- The main statement to prove
theorem smallest_positive_integer_with_20_divisors_is_432 :
  ∀ n : ℕ, has_exactly_20_divisors n → n = 432 :=
sorry

end smallest_positive_integer_with_20_divisors_is_432_l102_102953


namespace soldiers_line_l102_102484

theorem soldiers_line (n x y z : ℕ) (h₁ : y = 6 * x) (h₂ : y = 7 * z)
                      (h₃ : n = x + y) (h₄ : n = 7 * x) (h₅ : n = 8 * z) : n = 98 :=
by 
  sorry

end soldiers_line_l102_102484


namespace max_cars_div_10_l102_102262

noncomputable def max_cars (car_length : ℕ) (distance_for_speed : ℕ → ℕ) (speed : ℕ → ℕ) : ℕ :=
  let k := 2000
  2000 -- Maximum number of cars passing the sensor

theorem max_cars_div_10 (car_length : ℕ) (distance_for_speed : ℕ → ℕ) (speed : ℕ → ℕ) :
  car_length = 5 →
  (∀ k : ℕ, distance_for_speed k = k) →
  (∀ k : ℕ, speed k = 10 * k) →
  (max_cars car_length distance_for_speed speed) = 2000 → 
  (max_cars car_length distance_for_speed speed) / 10 = 200 := by
  intros
  sorry

end max_cars_div_10_l102_102262


namespace simplify_expr_l102_102625

theorem simplify_expr (x : ℝ) : 
  2 * x * (4 * x ^ 3 - 3 * x + 1) - 7 * (x ^ 3 - x ^ 2 + 3 * x - 4) = 
  8 * x ^ 4 - 7 * x ^ 3 + x ^ 2 - 19 * x + 28 := 
by
  sorry

end simplify_expr_l102_102625


namespace sand_problem_l102_102451

-- Definitions based on conditions
def initial_sand := 1050
def sand_lost_first := 32
def sand_lost_second := 67
def sand_lost_third := 45
def sand_lost_fourth := 54

-- Total sand lost
def total_sand_lost := sand_lost_first + sand_lost_second + sand_lost_third + sand_lost_fourth

-- Sand remaining
def sand_remaining := initial_sand - total_sand_lost

-- Theorem stating the proof problem
theorem sand_problem : sand_remaining = 852 :=
by
-- Skipping proof as per instructions
sorry

end sand_problem_l102_102451


namespace Bobby_has_27_pairs_l102_102715

-- Define the number of shoes Becky has
variable (B : ℕ)

-- Define the number of shoes Bonny has as 13, with the relationship to Becky's shoes
def Bonny_shoes : Prop := 2 * B - 5 = 13

-- Define the number of shoes Bobby has given Becky's count
def Bobby_shoes := 3 * B

-- Prove that Bobby has 27 pairs of shoes given the conditions
theorem Bobby_has_27_pairs (hB : Bonny_shoes B) : Bobby_shoes B = 27 := 
by 
  sorry

end Bobby_has_27_pairs_l102_102715


namespace incorrect_fraction_addition_l102_102909

theorem incorrect_fraction_addition (a b x y : ℤ) (h1 : 0 < b) (h2 : 0 < y) (h3 : (a + x) * (b * y) = (a * y + b * x) * (b + y)) :
  ∃ k : ℤ, x = -a * k^2 ∧ y = b * k :=
by
  sorry

end incorrect_fraction_addition_l102_102909


namespace smallest_value_m_plus_n_l102_102978

theorem smallest_value_m_plus_n (m n : ℕ) (h : 3 * n^3 = 5 * m^2) : m + n = 60 :=
sorry

end smallest_value_m_plus_n_l102_102978


namespace express_c_in_terms_of_a_b_l102_102967

-- Defining the vectors
def vec (x y : ℝ) : ℝ × ℝ := (x, y)

-- Defining the given vectors
def a := vec 1 1
def b := vec 1 (-1)
def c := vec (-1) 2

-- The statement
theorem express_c_in_terms_of_a_b :
  c = (1/2) • a + (-3/2) • b :=
sorry

end express_c_in_terms_of_a_b_l102_102967


namespace inequality_solution_l102_102703

theorem inequality_solution {x : ℝ} : (1 / 2 - (x - 2) / 3 > 1) → (x < 1 / 2) :=
by {
  sorry
}

end inequality_solution_l102_102703


namespace required_sand_volume_is_five_l102_102769

noncomputable def length : ℝ := 10
noncomputable def depth_cm : ℝ := 50
noncomputable def depth_m : ℝ := depth_cm / 100  -- converting cm to m
noncomputable def width : ℝ := 2
noncomputable def total_volume : ℝ := length * depth_m * width
noncomputable def current_volume : ℝ := total_volume / 2
noncomputable def additional_sand : ℝ := total_volume - current_volume

theorem required_sand_volume_is_five : additional_sand = 5 :=
by sorry

end required_sand_volume_is_five_l102_102769


namespace total_triangles_in_geometric_figure_l102_102792

noncomputable def numberOfTriangles : ℕ :=
  let smallest_triangles := 3 + 2 + 1
  let medium_triangles := 2
  let large_triangle := 1
  smallest_triangles + medium_triangles + large_triangle

theorem total_triangles_in_geometric_figure : numberOfTriangles = 9 := by
  unfold numberOfTriangles
  sorry

end total_triangles_in_geometric_figure_l102_102792


namespace sin_cos_identity_l102_102049

theorem sin_cos_identity (x : ℝ) (h : Real.sin x = 5 * Real.cos x) : Real.sin x * Real.cos x = 5 / 26 := 
by
  sorry

end sin_cos_identity_l102_102049


namespace solution_existence_l102_102986

def problem_statement : Prop :=
  ∃ x : ℝ, (0.38 * 80) - (0.12 * x) = 11.2 ∧ x = 160

theorem solution_existence : problem_statement :=
  sorry

end solution_existence_l102_102986


namespace line_equation_l102_102910

theorem line_equation (a b : ℝ) (h_intercept_eq : a = b) (h_pass_through : 3 * a + 2 * b = 2 * a + 5) : (3 + 2 = 5) ↔ (a = 5 ∧ b = 5) :=
sorry

end line_equation_l102_102910


namespace number_of_books_in_library_l102_102010

def number_of_bookcases : ℕ := 28
def shelves_per_bookcase : ℕ := 6
def books_per_shelf : ℕ := 19

theorem number_of_books_in_library : number_of_bookcases * shelves_per_bookcase * books_per_shelf = 3192 :=
by
  sorry

end number_of_books_in_library_l102_102010


namespace range_of_a_l102_102904

theorem range_of_a (a : ℝ) (h : a > 0) :
  let A := {x : ℝ | x^2 + 2 * x - 8 > 0}
  let B := {x : ℝ | x^2 - 2 * a * x + 4 ≤ 0}
  (∃! x : ℤ, (x : ℝ) ∈ A ∩ B) → (13 / 6 ≤ a ∧ a < 5 / 2) :=
by
  sorry

end range_of_a_l102_102904


namespace total_toothpicks_needed_l102_102602

theorem total_toothpicks_needed (length width : ℕ) (hl : length = 50) (hw : width = 40) : 
  (length + 1) * width + (width + 1) * length = 4090 := 
by
  -- proof omitted, replace this line with actual proof
  sorry

end total_toothpicks_needed_l102_102602


namespace select_student_for_performance_and_stability_l102_102840

def average_score_A : ℝ := 6.2
def average_score_B : ℝ := 6.0
def average_score_C : ℝ := 5.8
def average_score_D : ℝ := 6.2

def variance_A : ℝ := 0.32
def variance_B : ℝ := 0.58
def variance_C : ℝ := 0.12
def variance_D : ℝ := 0.25

theorem select_student_for_performance_and_stability :
  (average_score_A ≤ average_score_D ∧ variance_D < variance_A) →
  (average_score_B < average_score_A ∧ average_score_B < average_score_D) →
  (average_score_C < average_score_A ∧ average_score_C < average_score_D) →
  "D" = "D" :=
by
  intros h₁ h₂ h₃
  exact rfl

end select_student_for_performance_and_stability_l102_102840


namespace dinner_cost_l102_102764

theorem dinner_cost (tax_rate : ℝ) (tip_rate : ℝ) (total_amount : ℝ) : 
  tax_rate = 0.12 → 
  tip_rate = 0.18 → 
  total_amount = 30 → 
  (total_amount / (1 + tax_rate + tip_rate)) = 23.08 :=
by
  intros h1 h2 h3
  sorry

end dinner_cost_l102_102764


namespace weight_12m_rod_l102_102742

-- Define the weight of a 6 meters long rod
def weight_of_6m_rod : ℕ := 7

-- Given the condition that the weight is proportional to the length
def weight_of_rod (length : ℕ) : ℕ := (length / 6) * weight_of_6m_rod

-- Prove the weight of a 12 meters long rod
theorem weight_12m_rod : weight_of_rod 12 = 14 := by
  -- Calculation skipped, proof required here
  sorry

end weight_12m_rod_l102_102742


namespace divisible_by_42_l102_102520

theorem divisible_by_42 (a : ℤ) : ∃ k : ℤ, a^7 - a = 42 * k := 
sorry

end divisible_by_42_l102_102520


namespace general_term_of_sequence_l102_102200

theorem general_term_of_sequence (a : ℕ → ℝ) (h₁ : a 1 = 3) (h₂ : ∀ n : ℕ, n > 0 → a (n + 1) = (a n) ^ 2) :
  ∀ n : ℕ, n > 0 → a n = 3 ^ (2 ^ (n - 1)) :=
by
  intros n hn
  sorry

end general_term_of_sequence_l102_102200


namespace correct_option_c_l102_102282

variable (a b c : ℝ)

def quadratic_function (x : ℝ) : ℝ := a * x^2 + b * x + c

axiom symmetry_axis : -b / (2 * a) = 1

theorem correct_option_c (h : b = -2 * a) : c > 2 * b :=
 sorry

end correct_option_c_l102_102282


namespace problem1_l102_102127

theorem problem1 (f : ℚ → ℚ) (a : Fin 7 → ℚ) (h₁ : ∀ x, f x = (1 - 3 * x) * (1 + x) ^ 5)
  (h₂ : ∀ x, f x = a 0 + a 1 * x + a 2 * x^2 + a 3 * x^3 + a 4 * x^4 + a 5 * x^5 + a 6 * x^6) :
  a 0 + (1/3) * a 1 + (1/3^2) * a 2 + (1/3^3) * a 3 + (1/3^4) * a 4 + (1/3^5) * a 5 + (1/3^6) * a 6 = 
  (1 - 3 * (1/3)) * (1 + (1/3))^5 :=
by sorry

end problem1_l102_102127


namespace completing_the_square_l102_102611

theorem completing_the_square {x : ℝ} : x^2 - 6*x - 5 = 0 ↔ (x - 3)^2 = 14 := 
sorry

end completing_the_square_l102_102611


namespace measure_angle_ACB_l102_102899

-- Definitions of angles and the conditions
def angle_ABD := 140
def angle_BAC := 105
def supplementary_angle (α β : ℕ) := α + β = 180
def angle_sum_property (α β γ : ℕ) := α + β + γ = 180

-- Theorem to prove the measure of angle ACB
theorem measure_angle_ACB (angle_ABD : ℕ) 
                         (angle_BAC : ℕ) 
                         (h1 : supplementary_angle angle_ABD 40)
                         (h2 : angle_sum_property 40 angle_BAC 35) :
  angle_sum_property 40 105 35 :=
sorry

end measure_angle_ACB_l102_102899


namespace sqrt_meaningful_range_l102_102064

theorem sqrt_meaningful_range (x : ℝ) : x + 2 ≥ 0 → x ≥ -2 :=
by 
  intro h
  linarith [h]

end sqrt_meaningful_range_l102_102064


namespace picnic_problem_l102_102620

variables (M W C A : ℕ)

theorem picnic_problem
  (H1 : M + W + C = 200)
  (H2 : A = C + 20)
  (H3 : M = 65)
  (H4 : A = M + W) :
  M - W = 20 :=
by sorry

end picnic_problem_l102_102620


namespace probability_one_no_GP_l102_102509

def num_pies : ℕ := 6
def growth_pies : ℕ := 2
def shrink_pies : ℕ := 4
def picked_pies : ℕ := 3
def total_outcomes : ℕ := Nat.choose num_pies picked_pies

def fav_outcomes : ℕ := Nat.choose shrink_pies 2 -- Choosing 2 out of the 4 SP

def probability_complementary : ℚ := fav_outcomes / total_outcomes
def probability : ℚ := 1 - probability_complementary

theorem probability_one_no_GP :
  probability = 0.4 := by
  sorry

end probability_one_no_GP_l102_102509


namespace train_cross_duration_l102_102830

noncomputable def train_length : ℝ := 250
noncomputable def train_speed_kmph : ℝ := 162
noncomputable def train_speed_mps : ℝ := train_speed_kmph * (1000 / 3600)
noncomputable def time_to_cross_pole : ℝ := train_length / train_speed_mps

theorem train_cross_duration :
  time_to_cross_pole = 250 / (162 * (1000 / 3600)) :=
by
  -- The detailed proof is omitted as per instructions
  sorry

end train_cross_duration_l102_102830


namespace min_a_plus_b_l102_102519

variable (a b : ℝ)
variable (ha_pos : a > 0)
variable (hb_pos : b > 0)
variable (h1 : a^2 - 12 * b ≥ 0)
variable (h2 : 9 * b^2 - 4 * a ≥ 0)

theorem min_a_plus_b (a b : ℝ) (ha_pos : a > 0) (hb_pos : b > 0)
  (h1 : a^2 - 12 * b ≥ 0) (h2 : 9 * b^2 - 4 * a ≥ 0) :
  a + b = 3.3442 := 
sorry

end min_a_plus_b_l102_102519


namespace major_axis_double_minor_axis_l102_102404

-- Define the radius of the right circular cylinder.
def cylinder_radius := 2

-- Define the minor axis length based on the cylinder's radius.
def minor_axis_length := 2 * cylinder_radius

-- Define the major axis length as double the minor axis length.
def major_axis_length := 2 * minor_axis_length

-- State the theorem to prove the major axis length.
theorem major_axis_double_minor_axis : major_axis_length = 8 := by
  sorry

end major_axis_double_minor_axis_l102_102404


namespace heptagon_labeling_impossible_l102_102005

/-- 
  Let a heptagon be given with vertices labeled by integers a₁, a₂, a₃, a₄, a₅, a₆, a₇.
  The following two conditions are imposed:
  1. For every pair of consecutive vertices (aᵢ, aᵢ₊₁) (with indices mod 7), 
     at least one of aᵢ and aᵢ₊₁ divides the other.
  2. For every pair of non-consecutive vertices (aᵢ, aⱼ) where i ≠ j ± 1 mod 7, 
     neither aᵢ divides aⱼ nor aⱼ divides aᵢ. 

  Prove that such a labeling is impossible.
-/
theorem heptagon_labeling_impossible :
  ¬ ∃ (a : Fin 7 → ℕ),
    (∀ i : Fin 7, a i ∣ a ((i + 1) % 7) ∨ a ((i + 1) % 7) ∣ a i) ∧
    (∀ {i j : Fin 7}, (i ≠ j + 1 % 7) → (i ≠ j + 6 % 7) → ¬ (a i ∣ a j) ∧ ¬ (a j ∣ a i)) :=
sorry

end heptagon_labeling_impossible_l102_102005


namespace simplify_fraction_l102_102591

variable {x y : ℝ}

theorem simplify_fraction (h : x ≠ y) : (x^6 - y^6) / (x^3 - y^3) = x^3 + y^3 := by
  sorry

end simplify_fraction_l102_102591


namespace sum_of_coefficients_l102_102712

def original_function (x : ℝ) : ℝ := 3 * x^2 - 2 * x + 4

def transformed_function (x : ℝ) : ℝ := 3 * (x + 2)^2 - 2 * (x + 2) + 4 + 5

theorem sum_of_coefficients : (3 : ℝ) + 10 + 17 = 30 :=
by
  sorry

end sum_of_coefficients_l102_102712


namespace exists_nonneg_poly_div_l102_102040

theorem exists_nonneg_poly_div (P : Polynomial ℝ) 
  (hP_pos : ∀ x : ℝ, x > 0 → P.eval x > 0) :
  ∃ (Q R : Polynomial ℝ), (∀ n, Q.coeff n ≥ 0) ∧ (∀ n, R.coeff n ≥ 0) ∧ (P = Q / R) := 
sorry

end exists_nonneg_poly_div_l102_102040


namespace basketball_team_wins_l102_102563

theorem basketball_team_wins (f : ℚ) (h1 : 40 + 40 * f + (40 + 40 * f) = 130) : f = 5 / 8 :=
by
  sorry

end basketball_team_wins_l102_102563


namespace marbles_per_customer_l102_102996

theorem marbles_per_customer
  (initial_marbles remaining_marbles customers marbles_per_customer : ℕ)
  (h1 : initial_marbles = 400)
  (h2 : remaining_marbles = 100)
  (h3 : customers = 20)
  (h4 : initial_marbles - remaining_marbles = customers * marbles_per_customer) :
  marbles_per_customer = 15 :=
by
  sorry

end marbles_per_customer_l102_102996


namespace prime_bounds_l102_102601

noncomputable def is_prime (p : ℕ) : Prop := 2 ≤ p ∧ ∀ d : ℕ, d ∣ p → d = 1 ∨ d = p

theorem prime_bounds (n : ℕ) (h1 : 2 ≤ n) 
  (h2 : ∀ k, 0 ≤ k → k ≤ Nat.sqrt (n / 3) → is_prime (k^2 + k + n)) : 
  ∀ k, 0 ≤ k → k ≤ n - 2 → is_prime (k^2 + k + n) :=
by
  sorry

end prime_bounds_l102_102601


namespace distinguishable_arrangements_l102_102240

theorem distinguishable_arrangements :
  let n := 9
  let n1 := 3
  let n2 := 2
  let n3 := 4
  (Nat.factorial n) / ((Nat.factorial n1) * (Nat.factorial n2) * (Nat.factorial n3)) = 1260 :=
by sorry

end distinguishable_arrangements_l102_102240


namespace slices_leftover_l102_102741

def total_initial_slices : ℕ := 12 * 2
def bob_slices : ℕ := 12 / 2
def tom_slices : ℕ := 12 / 3
def sally_slices : ℕ := 12 / 6
def jerry_slices : ℕ := 12 / 4
def total_slices_eaten : ℕ := bob_slices + tom_slices + sally_slices + jerry_slices

theorem slices_leftover : total_initial_slices - total_slices_eaten = 9 := by
  sorry

end slices_leftover_l102_102741


namespace statistical_hypothesis_independence_l102_102077

def independence_test_statistical_hypothesis (A B: Prop) (independence_test: Prop) : Prop :=
  (independence_test ∧ A ∧ B) → (A = B)

theorem statistical_hypothesis_independence (A B: Prop) (independence_test: Prop) :
  (independence_test ∧ A ∧ B) → (A = B) :=
by
  sorry

end statistical_hypothesis_independence_l102_102077


namespace solution_l102_102927

theorem solution (y q : ℝ) (h1 : |y - 3| = q) (h2 : y < 3) : y - 2 * q = 3 - 3 * q :=
by
  sorry

end solution_l102_102927


namespace problem_a_l102_102416

def part_a : Prop :=
  ∃ (tokens : Finset (Fin 4 × Fin 4)), 
    tokens.card = 7 ∧ 
    (∀ (rows : Finset (Fin 4)) (cols : Finset (Fin 4)), rows.card = 2 → cols.card = 2 → 
      ∃ (token : (Fin 4 × Fin 4)), token ∈ tokens ∧ token.1 ∉ rows ∧ token.2 ∉ cols)

theorem problem_a : part_a :=
  sorry

end problem_a_l102_102416


namespace product_of_dodecagon_l102_102562

open Complex

theorem product_of_dodecagon (Q : Fin 12 → ℂ) (h₁ : Q 0 = 2) (h₇ : Q 6 = 8) :
  (Q 0) * (Q 1) * (Q 2) * (Q 3) * (Q 4) * (Q 5) * (Q 6) * (Q 7) * (Q 8) * (Q 9) * (Q 10) * (Q 11) = 244140624 :=
sorry

end product_of_dodecagon_l102_102562


namespace rank_from_left_l102_102452

theorem rank_from_left (total_students : ℕ) (rank_from_right : ℕ) (h1 : total_students = 20) (h2 : rank_from_right = 13) : 
  (total_students - rank_from_right + 1 = 8) :=
by
  sorry

end rank_from_left_l102_102452


namespace find_certain_number_l102_102846

theorem find_certain_number 
  (num : ℝ)
  (h1 : num / 14.5 = 177)
  (h2 : 29.94 / 1.45 = 17.7) : 
  num = 2566.5 := 
by 
  sorry

end find_certain_number_l102_102846


namespace proof_by_contradiction_conditions_l102_102448

theorem proof_by_contradiction_conditions:
  (∃ (neg_conclusion known_conditions ax_thms_defs original_conclusion : Prop),
    (neg_conclusion ∧ known_conditions ∧ ax_thms_defs) → False)
:= sorry

end proof_by_contradiction_conditions_l102_102448


namespace fg_of_3_is_94_l102_102122

def g (x : ℕ) : ℕ := 4 * x + 5
def f (x : ℕ) : ℕ := 6 * x - 8

theorem fg_of_3_is_94 : f (g 3) = 94 := by
  sorry

end fg_of_3_is_94_l102_102122


namespace irrational_sum_floor_eq_iff_l102_102640

theorem irrational_sum_floor_eq_iff (a b c d : ℝ) (h_irr_a : ¬ ∃ (q : ℚ), a = q) 
                                     (h_irr_b : ¬ ∃ (q : ℚ), b = q) 
                                     (h_irr_c : ¬ ∃ (q : ℚ), c = q) 
                                     (h_irr_d : ¬ ∃ (q : ℚ), d = q) 
                                     (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
                                     (h_pos_c : 0 < c) (h_pos_d : 0 < d)
                                     (h_sum_ab : a + b = 1) :
  (c + d = 1) ↔ (∀ (n : ℕ), ⌊n * a⌋ + ⌊n * b⌋ = ⌊n * c⌋ + ⌊n * d⌋) :=
sorry

end irrational_sum_floor_eq_iff_l102_102640


namespace minimum_positive_announcements_l102_102397

theorem minimum_positive_announcements (x y : ℕ) (h : x * (x - 1) = 132) (positive_products negative_products : ℕ)
  (hp : positive_products = y * (y - 1)) (hn : negative_products = (x - y) * (x - y - 1)) 
  (h_sum : positive_products + negative_products = 132) : 
  y = 2 :=
by sorry

end minimum_positive_announcements_l102_102397


namespace smaller_solid_volume_l102_102442

noncomputable def cube_edge_length : ℝ := 2

def point (x y z : ℝ) : ℝ × ℝ × ℝ := (x, y, z)

def D := point 0 0 0
def M := point 1 2 0
def N := point 2 0 1

-- Define the condition for the plane that passes through D, M, and N
def plane (p r q : ℝ × ℝ × ℝ) (x y z : ℝ) : Prop :=
  let (px, py, pz) := p
  let (rx, ry, rz) := r
  let (qx, qy, qz) := q
  2 * x - 4 * y - 8 * z = 0

-- Predicate to test if point is on a plane
def on_plane (pt : ℝ × ℝ × ℝ) : Prop :=
  let (x, y, z) := pt
  plane D M N x y z

-- Volume of the smaller solid
theorem smaller_solid_volume :
  ∃ V : ℝ, V = 1 / 6 :=
by
  sorry

end smaller_solid_volume_l102_102442


namespace smallest_positive_integer_l102_102708

-- Definitions of the conditions
def condition1 (k : ℕ) : Prop := k % 10 = 9
def condition2 (k : ℕ) : Prop := k % 9 = 8
def condition3 (k : ℕ) : Prop := k % 8 = 7
def condition4 (k : ℕ) : Prop := k % 7 = 6
def condition5 (k : ℕ) : Prop := k % 6 = 5
def condition6 (k : ℕ) : Prop := k % 5 = 4
def condition7 (k : ℕ) : Prop := k % 4 = 3
def condition8 (k : ℕ) : Prop := k % 3 = 2
def condition9 (k : ℕ) : Prop := k % 2 = 1

-- Statement of the problem
theorem smallest_positive_integer : ∃ k : ℕ, 
  k > 0 ∧
  condition1 k ∧ 
  condition2 k ∧ 
  condition3 k ∧ 
  condition4 k ∧ 
  condition5 k ∧ 
  condition6 k ∧ 
  condition7 k ∧ 
  condition8 k ∧ 
  condition9 k ∧
  k = 2519 := 
sorry

end smallest_positive_integer_l102_102708


namespace gcd_of_459_and_357_l102_102523

theorem gcd_of_459_and_357 : Nat.gcd 459 357 = 51 := by
  sorry

end gcd_of_459_and_357_l102_102523


namespace train_passes_jogger_in_approx_36_seconds_l102_102093

noncomputable def jogger_speed_kmph : ℝ := 8
noncomputable def train_speed_kmph : ℝ := 55
noncomputable def distance_ahead_m : ℝ := 340
noncomputable def train_length_m : ℝ := 130

noncomputable def kmph_to_mps (speed_kmph : ℝ) : ℝ :=
  (speed_kmph * 1000) / 3600

noncomputable def jogger_speed_mps : ℝ :=
  kmph_to_mps jogger_speed_kmph

noncomputable def train_speed_mps : ℝ :=
  kmph_to_mps train_speed_kmph

noncomputable def relative_speed_mps : ℝ :=
  train_speed_mps - jogger_speed_mps

noncomputable def total_distance_m : ℝ :=
  distance_ahead_m + train_length_m

noncomputable def time_to_pass_jogger_s : ℝ :=
  total_distance_m / relative_speed_mps

theorem train_passes_jogger_in_approx_36_seconds : 
  abs (time_to_pass_jogger_s - 36) < 1 := 
sorry

end train_passes_jogger_in_approx_36_seconds_l102_102093


namespace travel_ways_l102_102136

theorem travel_ways (highways : ℕ) (railways : ℕ) (n : ℕ) :
  highways = 3 → railways = 2 → n = highways + railways → n = 5 :=
by
  intros h_eq r_eq n_eq
  rw [h_eq, r_eq] at n_eq
  exact n_eq

end travel_ways_l102_102136


namespace coin_difference_is_eight_l102_102156

theorem coin_difference_is_eight :
  let min_coins := 2  -- two 25-cent coins
  let max_coins := 10 -- ten 5-cent coins
  max_coins - min_coins = 8 :=
by
  sorry

end coin_difference_is_eight_l102_102156


namespace sum_of_cubes_l102_102225

theorem sum_of_cubes {a b c : ℝ} (h1 : a + b + c = 5) (h2 : a * b + a * c + b * c = 7) (h3 : a * b * c = -18) : 
  a^3 + b^3 + c^3 = 29 :=
by
  -- The proof part is intentionally left out.
  sorry

end sum_of_cubes_l102_102225


namespace natural_number_triplets_l102_102432

theorem natural_number_triplets :
  ∀ (a b c : ℕ), a^3 + b^3 + c^3 = (a * b * c)^2 → 
    (a = 3 ∧ b = 2 ∧ c = 1) ∨ (a = 3 ∧ b = 1 ∧ c = 2) ∨ 
    (a = 2 ∧ b = 3 ∧ c = 1) ∨ (a = 2 ∧ b = 1 ∧ c = 3) ∨ 
    (a = 1 ∧ b = 3 ∧ c = 2) ∨ (a = 1 ∧ b = 2 ∧ c = 3) := 
by
  sorry

end natural_number_triplets_l102_102432


namespace numWaysToChoosePairs_is_15_l102_102542

def numWaysToChoosePairs : ℕ :=
  let white := Nat.choose 5 2
  let brown := Nat.choose 3 2
  let blue := Nat.choose 2 2
  let black := Nat.choose 2 2
  white + brown + blue + black

theorem numWaysToChoosePairs_is_15 : numWaysToChoosePairs = 15 := by
  -- We will prove this theorem in actual proof
  sorry

end numWaysToChoosePairs_is_15_l102_102542


namespace initial_marbles_l102_102749

theorem initial_marbles (M : ℕ) (h1 : M + 9 = 104) : M = 95 := by
  sorry

end initial_marbles_l102_102749


namespace numberOfColoringWays_l102_102820

-- Define the problem parameters
def totalBalls : Nat := 5
def redBalls : Nat := 1
def blueBalls : Nat := 1
def yellowBalls : Nat := 2
def whiteBalls : Nat := 1

-- Show that the number of permutations of the multiset is 60
theorem numberOfColoringWays : (Nat.factorial totalBalls) / ((Nat.factorial redBalls) * (Nat.factorial blueBalls) * (Nat.factorial yellowBalls) * (Nat.factorial whiteBalls)) = 60 :=
  by
  simp [totalBalls, redBalls, blueBalls, yellowBalls, whiteBalls]
  sorry

end numberOfColoringWays_l102_102820


namespace special_hash_value_l102_102231

def special_hash (a b c d : ℝ) : ℝ :=
  d * b ^ 2 - 4 * a * c

theorem special_hash_value :
  special_hash 2 3 1 (1 / 2) = -3.5 :=
by
  -- Note: Insert proof here
  sorry

end special_hash_value_l102_102231


namespace t_shirt_cost_l102_102646

theorem t_shirt_cost (T : ℕ) 
  (h1 : 3 * T + 50 = 110) : T = 20 := 
by
  sorry

end t_shirt_cost_l102_102646


namespace f_at_10_l102_102675

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 2 * x^2 - 5 * x + 6

-- Prove that f(10) = 756
theorem f_at_10 : f 10 = 756 := by
  sorry

end f_at_10_l102_102675


namespace shaded_area_percentage_l102_102903

theorem shaded_area_percentage (n_shaded : ℕ) (n_total : ℕ) (hn_shaded : n_shaded = 21) (hn_total : n_total = 36) :
  ((n_shaded : ℚ) / (n_total : ℚ)) * 100 = 58.33 :=
by
  sorry

end shaded_area_percentage_l102_102903


namespace functional_equation_solution_l102_102802

theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, (∀ x y : ℝ, f x * f y + f (x + y) = x * y) →
  (∀ x : ℝ, f x = x - 1 ∨ f x = -x - 1) :=
by
  sorry

end functional_equation_solution_l102_102802


namespace interest_rate_bc_l102_102781

def interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * rate * time

def gain_b (interest_bc interest_ab : ℝ) : ℝ :=
  interest_bc - interest_ab

theorem interest_rate_bc :
  ∀ (principal : ℝ) (rate_ab rate_bc : ℝ) (time : ℕ) (gain : ℝ),
    principal = 3500 → rate_ab = 0.10 → time = 3 → gain = 525 →
    interest principal rate_ab time = 1050 →
    gain_b (interest principal rate_bc time) (interest principal rate_ab time) = gain →
    rate_bc = 0.15 :=
by
  intros principal rate_ab rate_bc time gain h_principal h_rate_ab h_time h_gain h_interest_ab h_gain_b
  sorry

end interest_rate_bc_l102_102781


namespace rectangle_area_l102_102284

theorem rectangle_area (a b : ℕ) 
  (h1 : 2 * (a + b) = 16)
  (h2 : a^2 + b^2 - 2 * a * b - 4 = 0) :
  a * b = 30 :=
by
  sorry

end rectangle_area_l102_102284


namespace simplify_expression_l102_102540

theorem simplify_expression : (-5) - (-4) + (-7) - (2) = -5 + 4 - 7 - 2 := 
by
  sorry

end simplify_expression_l102_102540


namespace original_number_l102_102573

theorem original_number (x y : ℕ) (h1 : x + y = 859560) (h2 : y = 859560 % 456) : x = 859376 ∧ 456 ∣ x :=
by
  sorry

end original_number_l102_102573


namespace prob_of_caps_given_sunglasses_l102_102036

theorem prob_of_caps_given_sunglasses (n_sunglasses n_caps n_both : ℕ) (P_sunglasses_given_caps : ℚ) 
  (h_nsunglasses : n_sunglasses = 80) (h_ncaps : n_caps = 45)
  (h_Psunglasses_given_caps : P_sunglasses_given_caps = 3/8)
  (h_nboth : n_both = P_sunglasses_given_caps * n_sunglasses) :
  (n_both / n_caps) = 2/3 := 
by
  sorry

end prob_of_caps_given_sunglasses_l102_102036


namespace average_speed_l102_102137

-- Define the conditions
def initial_reading : ℕ := 2552
def final_reading : ℕ := 2992
def day1_time : ℕ := 6
def day2_time : ℕ := 8

-- Theorem: Proving the average speed is 31 miles per hour.
theorem average_speed :
  final_reading - initial_reading = 440 ∧ day1_time + day2_time = 14 ∧ 
  (final_reading - initial_reading) / (day1_time + day2_time) = 31 :=
by
  sorry

end average_speed_l102_102137


namespace min_value_of_a_sq_plus_b_sq_over_a_minus_b_l102_102479

theorem min_value_of_a_sq_plus_b_sq_over_a_minus_b {a b : ℝ} (h1 : a > b) (h2 : a * b = 1) : 
  ∃ x, x = 2 * Real.sqrt 2 ∧ ∀ y, y = (a^2 + b^2) / (a - b) → y ≥ x :=
by {
  sorry
}

end min_value_of_a_sq_plus_b_sq_over_a_minus_b_l102_102479


namespace find_a_b_and_compare_y_values_l102_102375

-- Conditions
def quadratic (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x - 1
def linear (a : ℝ) (x : ℝ) : ℝ := a * x

-- Problem statement
theorem find_a_b_and_compare_y_values (a b y1 y2 y3 : ℝ) (h₀ : quadratic a b (-2) = 1) (h₁ : linear a (-2) = 1)
    (h2 : y1 = quadratic a b 2) (h3 : y2 = quadratic a b b) (h4 : y3 = quadratic a b (a - b)) :
  (a = -1/2) ∧ (b = -2) ∧ y1 < y3 ∧ y3 < y2 :=
by
  -- Placeholder for the proof
  sorry

end find_a_b_and_compare_y_values_l102_102375


namespace vegetarian_family_l102_102767

theorem vegetarian_family (eat_veg eat_non_veg eat_both : ℕ) (total_veg : ℕ) 
  (h1 : eat_non_veg = 8) (h2 : eat_both = 11) (h3 : total_veg = 26)
  : eat_veg = total_veg - eat_both := by
  sorry

end vegetarian_family_l102_102767


namespace total_people_in_line_l102_102382

theorem total_people_in_line (n : ℕ) (h : n = 5): n + 2 = 7 :=
by
  -- This is where the proof would normally go, but we omit it with "sorry"
  sorry

end total_people_in_line_l102_102382


namespace number_of_cows_l102_102188

-- Definitions
variables (a g e c : ℕ)
variables (six_two : 6 * e = 2 * a + 4 * g) (eight_two : 8 * e = 2 * a + 8 * g)

-- Theorem statement
theorem number_of_cows (a g e : ℕ) (six_two : 6 * e = 2 * a + 4 * g) (eight_two : 8 * e = 2 * a + 8 * g) :
  ∃ c : ℕ, c * e * 6 = 6 * a + 36 * g ∧ c = 5 :=
by
  sorry

end number_of_cows_l102_102188


namespace trirectangular_tetrahedron_max_volume_l102_102298

noncomputable def max_volume_trirectangular_tetrahedron (S : ℝ) : ℝ :=
  S^3 * (Real.sqrt 2 - 1)^3 / 162

theorem trirectangular_tetrahedron_max_volume
  (a b c : ℝ) (H : a > 0 ∧ b > 0 ∧ c > 0)
  (S : ℝ)
  (edge_sum :
    S = a + b + c + Real.sqrt (a^2 + b^2) + Real.sqrt (b^2 + c^2) + Real.sqrt (c^2 + a^2))
  : ∃ V, V = max_volume_trirectangular_tetrahedron S :=
by
  sorry

end trirectangular_tetrahedron_max_volume_l102_102298


namespace season_duration_l102_102965

theorem season_duration (total_games : ℕ) (games_per_month : ℕ) (h1 : total_games = 323) (h2 : games_per_month = 19) :
  (total_games / games_per_month) = 17 :=
by
  sorry

end season_duration_l102_102965


namespace value_of_expression_l102_102488

-- Defining the given conditions as Lean definitions
def x : ℚ := 2 / 3
def y : ℚ := 5 / 2

-- The theorem statement to prove that the given expression equals the correct answer
theorem value_of_expression : (1 / 3) * x^7 * y^6 = 125 / 261 :=
by
  sorry

end value_of_expression_l102_102488


namespace point_slope_form_l102_102853

theorem point_slope_form (k : ℝ) (p : ℝ × ℝ) (h_slope : k = 2) (h_point : p = (2, -3)) :
  (∃ l : ℝ → ℝ, ∀ x y : ℝ, y = l x ↔ y = 2 * (x - 2) + (-3)) := 
sorry

end point_slope_form_l102_102853


namespace verify_z_relationship_l102_102437

variable {x y z : ℝ}

theorem verify_z_relationship (h1 : x > y) (h2 : y > 1) :
  z = (x + 3) - 2 * (y - 5) → z = x - 2 * y + 13 :=
by
  intros
  sorry

end verify_z_relationship_l102_102437


namespace solve_cubic_eq_l102_102434

theorem solve_cubic_eq (x : ℝ) : x^3 + (2 - x)^3 = 8 ↔ x = 0 ∨ x = 2 := 
by 
  { sorry }

end solve_cubic_eq_l102_102434


namespace subtraction_example_l102_102015

theorem subtraction_example : 6102 - 2016 = 4086 := by
  sorry

end subtraction_example_l102_102015


namespace triangle_area_is_18_l102_102081

noncomputable def area_triangle : ℝ :=
  let vertices : List (ℝ × ℝ) := [(1, 2), (7, 6), (1, 8)]
  let base := (8 - 2) -- Length between (1, 2) and (1, 8)
  let height := (7 - 1) -- Perpendicular distance from (7, 6) to x = 1
  (1 / 2) * base * height

theorem triangle_area_is_18 : area_triangle = 18 := by
  sorry

end triangle_area_is_18_l102_102081


namespace find_F_l102_102352

theorem find_F (F C : ℝ) (h1 : C = 35) (h2 : C = (7/12) * (F - 40)) : F = 100 :=
by
  sorry

end find_F_l102_102352


namespace angie_bought_18_pretzels_l102_102334

theorem angie_bought_18_pretzels
  (B : ℕ := 12) -- Barry bought 12 pretzels
  (S : ℕ := B / 2) -- Shelly bought half as many pretzels as Barry
  (A : ℕ := 3 * S) -- Angie bought three times as many pretzels as Shelly
  : A = 18 := sorry

end angie_bought_18_pretzels_l102_102334


namespace equivalent_condition_for_continuity_l102_102305

theorem equivalent_condition_for_continuity {x c d : ℝ} (g : ℝ → ℝ) (h1 : g x = 5 * x - 3) (h2 : ∀ x, |g x - 1| < c → |x - 1| < d) (hc : c > 0) (hd : d > 0) : d ≤ c / 5 :=
sorry

end equivalent_condition_for_continuity_l102_102305


namespace stream_speed_is_2_l102_102598

variable (v : ℝ) -- Let v be the speed of the stream in km/h

-- Condition 1: Man's swimming speed in still water
def swimming_speed_still : ℝ := 6

-- Condition 2: It takes him twice as long to swim upstream as downstream
def condition : Prop := (swimming_speed_still + v) / (swimming_speed_still - v) = 2

theorem stream_speed_is_2 : condition v → v = 2 := by
  intro h
  -- Proof goes here
  sorry

end stream_speed_is_2_l102_102598


namespace k_range_condition_l102_102367

theorem k_range_condition (k : ℝ) :
    (∀ x : ℝ, x^2 - (2 * k - 6) * x + k - 3 > 0) ↔ (3 < k ∧ k < 4) :=
by
  sorry

end k_range_condition_l102_102367


namespace road_width_l102_102455

theorem road_width
  (road_length : ℝ) 
  (truckload_area : ℝ) 
  (truckload_cost : ℝ) 
  (sales_tax : ℝ) 
  (total_cost : ℝ) :
  road_length = 2000 ∧
  truckload_area = 800 ∧
  truckload_cost = 75 ∧
  sales_tax = 0.20 ∧
  total_cost = 4500 →
  ∃ width : ℝ, width = 20 :=
by
  sorry

end road_width_l102_102455


namespace sequence_general_formula_l102_102973

open Nat

noncomputable def seq (a : ℕ → ℝ) : Prop :=
∀ (n : ℕ), n > 0 → (n+1) * (a (n + 1))^2 - n * (a n)^2 + (a (n + 1)) * (a n) = 0

theorem sequence_general_formula :
  ∃ (a : ℕ → ℝ), seq a ∧ (a 1 = 1) ∧ (∀ (n : ℕ), n > 0 → a n = 1 / n) :=
by
  sorry

end sequence_general_formula_l102_102973


namespace divisibility_condition_l102_102124

theorem divisibility_condition (n : ℕ) : 
  13 ∣ (4 * 3^(2^n) + 3 * 4^(2^n)) ↔ Even n := 
sorry

end divisibility_condition_l102_102124


namespace find_article_cost_l102_102508

noncomputable def original_cost_price (C S : ℝ) :=
  (S = 1.25 * C) ∧
  (S - 6.30 = 1.04 * C)

theorem find_article_cost (C S : ℝ) (h : original_cost_price C S) : C = 30 :=
by sorry

end find_article_cost_l102_102508


namespace find_a_l102_102533

theorem find_a (x : ℝ) (hx1 : 0 < x)
  (hx2 : x + 1/x ≥ 2)
  (hx3 : x + 4/x^2 ≥ 3)
  (hx4 : x + 27/x^3 ≥ 4) :
  (x + a/x^4 ≥ 5) → a = 4^4 :=
sorry

end find_a_l102_102533


namespace find_parallel_line_l102_102668

def line1 : ℝ → ℝ → Prop := λ x y => 2 * x - 3 * y + 2 = 0
def line2 : ℝ → ℝ → Prop := λ x y => 3 * x - 4 * y + 2 = 0
def parallelLine : ℝ → ℝ → Prop := λ x y => 4 * x + y - 4 = 0

theorem find_parallel_line (x y : ℝ) (hx : line1 x y) (hy : line2 x y) : 
  ∃ c : ℝ, (λ x y => 4 * x + y + c = 0) (2:ℝ) (2:ℝ) ∧ 
          ∀ x' y', (λ x' y' => 4 * x' + y' + c = 0) x' y' ↔ 4 * x' + y' - 10 = 0 := 
sorry

end find_parallel_line_l102_102668


namespace james_bags_l102_102682

theorem james_bags (total_marbles : ℕ) (remaining_marbles : ℕ) (b : ℕ) (m : ℕ) 
  (h1 : total_marbles = 28) 
  (h2 : remaining_marbles = 21) 
  (h3 : m = total_marbles - remaining_marbles) 
  (h4 : b = total_marbles / m) : 
  b = 4 :=
by
  sorry

end james_bags_l102_102682


namespace value_of_k_l102_102255

noncomputable def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

theorem value_of_k
  (a d : ℝ)
  (a1_eq_1 : a = 1)
  (sum_9_eq_sum_4 : 9/2 * (2*a + 8*d) = 4/2 * (2*a + 3*d))
  (k : ℕ)
  (a_k_plus_a_4_eq_0 : arithmetic_sequence a d k + arithmetic_sequence a d 4 = 0) :
  k = 10 :=
by
  sorry

end value_of_k_l102_102255


namespace zinc_percentage_in_1_gram_antacid_l102_102433

theorem zinc_percentage_in_1_gram_antacid :
  ∀ (z1 z2 : ℕ → ℤ) (total_zinc : ℤ),
    z1 0 = 2 ∧ z2 0 = 2 ∧ z1 1 = 1 ∧ total_zinc = 650 ∧
    (z1 0) * 2 * 5 / 100 + (z2 1) * 3 = total_zinc / 100 →
    (z2 1) * 100 = 15 :=
by
  sorry

end zinc_percentage_in_1_gram_antacid_l102_102433


namespace day_53_days_from_thursday_is_monday_l102_102944

def day_of_week : Type := {n : ℤ // n % 7 = n}

def Thursday : day_of_week := ⟨4, by norm_num⟩
def Monday : day_of_week := ⟨1, by norm_num⟩

theorem day_53_days_from_thursday_is_monday : 
  (⟨(4 + 53) % 7, by norm_num⟩ : day_of_week) = Monday := 
by 
  sorry

end day_53_days_from_thursday_is_monday_l102_102944


namespace smallest_sum_3x3_grid_l102_102921

-- Define the given conditions
def numbers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9] -- List of numbers used in the grid
def total_sum : ℕ := 45 -- Total sum of numbers from 1 to 9
def grid_size : ℕ := 3 -- Size of the grid
def corners_ids : List Nat := [0, 2, 6, 8] -- Indices of the corners in the grid
def remaining_sum : ℕ := 25 -- Sum of the remaining 5 numbers (after excluding the corners)

-- Define the goal: Prove that the smallest sum s is achieved
theorem smallest_sum_3x3_grid : ∃ s : ℕ, 
  (∀ (r : Fin grid_size) (c : Fin grid_size),
    r + c = s) → (s = 12) :=
by
  sorry

end smallest_sum_3x3_grid_l102_102921


namespace modulus_Z_l102_102279

theorem modulus_Z (Z : ℂ) (h : Z * (2 - 3 * Complex.I) = 6 + 4 * Complex.I) : Complex.abs Z = 2 := 
sorry

end modulus_Z_l102_102279


namespace binary_1011_is_11_decimal_124_is_174_l102_102066

-- Define the conversion from binary to decimal
def binaryToDecimal (n : Nat) : Nat :=
  (n % 10) * 2^0 + ((n / 10) % 10) * 2^1 + ((n / 100) % 10) * 2^2 + ((n / 1000) % 10) * 2^3

-- Define the conversion from decimal to octal through division and remainder
noncomputable def decimalToOctal (n : Nat) : String := 
  let rec aux (n : Nat) (acc : List Nat) : List Nat :=
    if n = 0 then acc else aux (n / 8) ((n % 8) :: acc)
  (aux n []).foldr (fun d s => s ++ d.repr) ""

-- Prove that the binary number 1011 (base 2) equals the decimal number 11
theorem binary_1011_is_11 : binaryToDecimal 1011 = 11 := by
  sorry

-- Prove that the decimal number 124 equals the octal number 174 (base 8)
theorem decimal_124_is_174 : decimalToOctal 124 = "174" := by
  sorry

end binary_1011_is_11_decimal_124_is_174_l102_102066


namespace sum_of_three_numbers_eq_zero_l102_102957

theorem sum_of_three_numbers_eq_zero 
  (a b c : ℝ) 
  (h_sorted : a ≤ b ∧ b ≤ c) 
  (h_median : b = 10) 
  (h_mean_least : (a + b + c) / 3 = a + 20) 
  (h_mean_greatest : (a + b + c) / 3 = c - 10) 
  : a + b + c = 0 := 
by 
  sorry

end sum_of_three_numbers_eq_zero_l102_102957


namespace no_four_digit_numbers_divisible_by_11_l102_102128

theorem no_four_digit_numbers_divisible_by_11 (a b c d : ℕ) (h₁ : 1 ≤ a) (h₂ : a ≤ 9) 
(h₃ : 0 ≤ b) (h₄ : b ≤ 9) (h₅ : 0 ≤ c) (h₆ : c ≤ 9) (h₇ : 0 ≤ d) (h₈ : d ≤ 9) 
(h₉ : a + b + c + d = 10) (h₁₀ : a + c = b + d) : 
0 = 0 :=
sorry

end no_four_digit_numbers_divisible_by_11_l102_102128


namespace max_tan_alpha_l102_102604

theorem max_tan_alpha (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
    (h : Real.tan (α + β) = 9 * Real.tan β) : Real.tan α ≤ 4 / 3 :=
by
  sorry

end max_tan_alpha_l102_102604


namespace cubic_roots_sum_of_cubes_l102_102922

def cube_root (x : ℝ) : ℝ := x^(1/3)

theorem cubic_roots_sum_of_cubes :
  let α := cube_root 17
  let β := cube_root 73
  let γ := cube_root 137
  ∀ (a b c : ℝ),
    (a - α) * (a - β) * (a - γ) = 1/2 ∧
    (b - α) * (b - β) * (b - γ) = 1/2 ∧
    (c - α) * (c - β) * (c - γ) = 1/2 →
    a^3 + b^3 + c^3 = 228.5 :=
by {
  sorry
}

end cubic_roots_sum_of_cubes_l102_102922


namespace andy_cavities_l102_102760

def candy_canes_from_parents : ℕ := 2
def candy_canes_per_teacher : ℕ := 3
def number_of_teachers : ℕ := 4
def fraction_to_buy : ℚ := 1 / 7
def cavities_per_candies : ℕ := 4

theorem andy_cavities : (candy_canes_from_parents 
                         + candy_canes_per_teacher * number_of_teachers 
                         + (candy_canes_from_parents 
                         + candy_canes_per_teacher * number_of_teachers) * fraction_to_buy)
                         / cavities_per_candies = 4 := by
  sorry

end andy_cavities_l102_102760


namespace cans_left_to_be_loaded_l102_102100

def cartons_total : ℕ := 50
def cartons_loaded : ℕ := 40
def cans_per_carton : ℕ := 20

theorem cans_left_to_be_loaded : (cartons_total - cartons_loaded) * cans_per_carton = 200 := by
  sorry

end cans_left_to_be_loaded_l102_102100


namespace cleaner_used_after_30_minutes_l102_102426

-- Define function to calculate the total amount of cleaner used
def total_cleaner_used (time: ℕ) (rate1: ℕ) (time1: ℕ) (rate2: ℕ) (time2: ℕ) (rate3: ℕ) (time3: ℕ) : ℕ :=
  (rate1 * time1) + (rate2 * time2) + (rate3 * time3)

-- The main theorem statement
theorem cleaner_used_after_30_minutes : total_cleaner_used 30 2 15 3 10 4 5 = 80 := by
  -- insert proof here
  sorry

end cleaner_used_after_30_minutes_l102_102426


namespace total_tickets_sold_l102_102581

theorem total_tickets_sold (x y : ℕ) (h1 : 12 * x + 8 * y = 3320) (h2 : y = x + 240) : 
  x + y = 380 :=
by -- proof
  sorry

end total_tickets_sold_l102_102581


namespace mr_johnson_needs_additional_volunteers_l102_102805

-- Definitions for the given conditions
def math_classes := 5
def students_per_class := 4
def total_students := math_classes * students_per_class

def total_teachers := 10
def carpentry_skilled_teachers := 3

def total_parents := 15
def lighting_sound_experienced_parents := 6

def total_volunteers_needed := 100
def carpentry_volunteers_needed := 8
def lighting_sound_volunteers_needed := 10

-- Total current volunteers
def current_volunteers := total_students + total_teachers + total_parents

-- Volunteers with specific skills
def current_carpentry_skilled := carpentry_skilled_teachers
def current_lighting_sound_experienced := lighting_sound_experienced_parents

-- Additional volunteers needed
def additional_carpentry_needed :=
  carpentry_volunteers_needed - current_carpentry_skilled
def additional_lighting_sound_needed :=
  lighting_sound_volunteers_needed - current_lighting_sound_experienced

-- Total additional volunteer needed
def additional_volunteers_needed :=
  additional_carpentry_needed + additional_lighting_sound_needed

-- The theorem we need to prove:
theorem mr_johnson_needs_additional_volunteers :
  additional_volunteers_needed = 9 := by
  sorry

end mr_johnson_needs_additional_volunteers_l102_102805


namespace range_of_t_range_of_a_l102_102221

-- Proposition P: The curve equation represents an ellipse with foci on the x-axis
def propositionP (t : ℝ) : Prop := ∀ x y : ℝ, (x^2 / (4 - t) + y^2 / (t - 1) = 1)

-- Proof problem for t
theorem range_of_t (t : ℝ) (h : propositionP t) : 1 < t ∧ t < 5 / 2 := 
  sorry

-- Proposition Q: The inequality involving real number t
def propositionQ (t a : ℝ) : Prop := t^2 - (a + 3) * t + (a + 2) < 0

-- Proof problem for a
theorem range_of_a (a : ℝ) (h₁ : ∀ t : ℝ, propositionP t → propositionQ t a) 
                   (h₂ : ∃ t : ℝ, propositionQ t a ∧ ¬ propositionP t) :
  a > 1 / 2 :=
  sorry

end range_of_t_range_of_a_l102_102221


namespace transformed_expression_value_l102_102728

-- Defining the new operations according to the problem's conditions
def new_minus (a b : ℕ) : ℕ := a + b
def new_plus (a b : ℕ) : ℕ := a * b
def new_times (a b : ℕ) : ℕ := a / b
def new_div (a b : ℕ) : ℕ := a - b

-- Problem statement
theorem transformed_expression_value : new_minus 6 (new_plus 9 (new_times 8 (new_div 3 25))) = 5 :=
sorry

end transformed_expression_value_l102_102728


namespace f_is_odd_range_of_x_l102_102883

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation : ∀ x₁ x₂ : ℝ, f (x₁ + x₂) = f x₁ + f x₂
axiom f_3 : f 3 = 1
axiom f_increase_nonneg : ∀ x₁ x₂ : ℝ, (0 ≤ x₁ ∧ x₁ ≤ x₂) → f x₁ ≤ f x₂
axiom f_lt_2 : ∀ x : ℝ, f (x - 1) < 2

theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x :=
sorry

theorem range_of_x : {x : ℝ | f (x - 1) < 2} =
{s : ℝ | sorry } :=
sorry

end f_is_odd_range_of_x_l102_102883


namespace negation_of_exists_solution_l102_102183

theorem negation_of_exists_solution :
  ¬ (∃ c : ℝ, c > 0 ∧ ∃ x : ℝ, x^2 - x + c = 0) ↔ ∀ c : ℝ, c > 0 → ¬ (∃ x : ℝ, x^2 - x + c = 0) :=
by
  sorry

end negation_of_exists_solution_l102_102183


namespace complement_set_unique_l102_102635

-- Define the universal set U
def U : Set ℕ := {1,2,3,4,5,6,7,8}

-- Define the complement of B with respect to U
def complement_B : Set ℕ := {1,3}

-- The set B that we need to prove
def B : Set ℕ := {2,4,5,6,7,8}

-- State that B is the set with the given complement in U
theorem complement_set_unique (U : Set ℕ) (complement_B : Set ℕ) :
    (U \ complement_B = {2,4,5,6,7,8}) :=
by
    -- We need to prove B is the set {2,4,5,6,7,8}
    sorry

end complement_set_unique_l102_102635


namespace prove_incorrect_statement_l102_102517

-- Definitions based on given conditions
def isIrrational (x : ℝ) : Prop := ¬ ∃ a b : ℚ, x = a / b ∧ b ≠ 0
def isSquareRoot (x y : ℝ) : Prop := x * x = y
def hasSquareRoot (x : ℝ) : Prop := ∃ y : ℝ, isSquareRoot y x

-- Options translated into Lean
def optionA : Prop := ∀ x : ℝ, isIrrational x → ¬ hasSquareRoot x
def optionB (x : ℝ) : Prop := 0 < x → ∃ y : ℝ, y * y = x ∧ (-y) * (-y) = x
def optionC : Prop := isSquareRoot 0 0
def optionD (a : ℝ) : Prop := ∀ x : ℝ, x = -a → (x ^ 3 = - (a ^ 3))

-- The incorrect statement according to the solution
def incorrectStatement : Prop := optionA

-- The theorem to be proven
theorem prove_incorrect_statement : incorrectStatement :=
by
  -- Replace with the actual proof, currently a placeholder using sorry
  sorry

end prove_incorrect_statement_l102_102517


namespace swiss_slices_correct_l102_102266

-- Define the variables and conditions
variables (S : ℕ) (cheddar_slices : ℕ := 12) (total_cheddar_slices : ℕ := 84) (total_swiss_slices : ℕ := 84)

-- Define the statement to be proved
theorem swiss_slices_correct (H : total_cheddar_slices = total_swiss_slices) : S = 12 :=
sorry

end swiss_slices_correct_l102_102266


namespace power_division_l102_102201

theorem power_division (a b : ℕ) (h₁ : 64 = 8^2) (h₂ : a = 15) (h₃ : b = 7) : 8^a / 64^b = 8 :=
by
  -- Equivalent to 8^15 / 64^7 = 8, given that 64 = 8^2
  sorry

end power_division_l102_102201


namespace difference_between_min_and_max_l102_102418

noncomputable 
def minValue (x z : ℝ) (hx : x ≠ 0) (hz : z ≠ 0) : ℝ := 0

noncomputable
def maxValue (x z : ℝ) (hx : x ≠ 0) (hz : z ≠ 0) : ℝ := 1.5

theorem difference_between_min_and_max (x z : ℝ) (hx : x ≠ 0) (hz : z ≠ 0) :
  maxValue x z hx hz - minValue x z hx hz = 1.5 :=
by
  sorry

end difference_between_min_and_max_l102_102418


namespace monotonically_increasing_interval_l102_102164

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem monotonically_increasing_interval :
  ∀ x : ℝ, x > 1 / Real.exp 1 → (Real.log x + 1) > 0 :=
by
  intros x hx
  sorry

end monotonically_increasing_interval_l102_102164


namespace imo_hosting_arrangements_l102_102014

structure IMOCompetition where
  countries : Finset String
  continents : Finset String
  assignments : Finset (String × String)
  constraints : String → String
  assignments_must_be_unique : ∀ {c1 c2 : String} {cnt1 cnt2 : String},
                                 (c1, cnt1) ∈ assignments → (c2, cnt2) ∈ assignments → 
                                 constraints c1 ≠ constraints c2 → c1 ≠ c2
  no_consecutive_same_continent : ∀ {c1 c2 : String} {cnt1 cnt2 : String},
                                   (c1, cnt1) ∈ assignments → (c2, cnt2) ∈ assignments → 
                                   (c1, cnt1) ≠ (c2, cnt2) →
                                   constraints c1 ≠ constraints c2

def number_of_valid_arrangements (comp: IMOCompetition) : Nat := 240

theorem imo_hosting_arrangements (comp : IMOCompetition) :
  number_of_valid_arrangements comp = 240 := by
  sorry

end imo_hosting_arrangements_l102_102014


namespace exists_composite_power_sum_l102_102151

def is_composite (n : ℕ) : Prop := ∃ p q : ℕ, 1 < p ∧ 1 < q ∧ n = p * q 

theorem exists_composite_power_sum (a : ℕ) (h1 : 1 < a) (h2 : a ≤ 100) : 
  ∃ n, (n > 0) ∧ (n ≤ 6) ∧ is_composite (a ^ (2 ^ n) + 1) :=
by
  sorry

end exists_composite_power_sum_l102_102151


namespace number_of_primary_schools_l102_102450

theorem number_of_primary_schools (A B total : ℕ) (h1 : A = 2 * 400)
  (h2 : B = 2 * 340) (h3 : total = 1480) (h4 : total = A + B) :
  2 + 2 = 4 :=
by
  sorry

end number_of_primary_schools_l102_102450


namespace quadratic_fraction_formula_l102_102925

theorem quadratic_fraction_formula (p q α β : ℝ) 
  (h1 : α + β = p) 
  (h2 : α * β = 6) 
  (h3 : p^2 ≠ 12) 
  (h4 : ∃ x : ℝ, x^2 - p * x + q = 0) :
  (α + β) / (α^2 + β^2) = p / (p^2 - 12) :=
sorry

end quadratic_fraction_formula_l102_102925


namespace percentage_decrease_second_year_l102_102705

-- Define initial population
def initial_population : ℝ := 14999.999999999998

-- Define the population at the end of the first year after 12% increase
def population_end_year_1 : ℝ := initial_population * 1.12

-- Define the final population at the end of the second year
def final_population : ℝ := 14784.0

-- Define the proof statement
theorem percentage_decrease_second_year :
  ∃ D : ℝ, final_population = population_end_year_1 * (1 - D / 100) ∧ D = 12 :=
by
  sorry

end percentage_decrease_second_year_l102_102705


namespace largest_x_l102_102355

-- Define the condition of the problem.
def equation_holds (x : ℝ) : Prop :=
  (5 * x - 20) / (4 * x - 5) ^ 2 + (5 * x - 20) / (4 * x - 5) = 20

-- State the theorem to prove the largest value of x is 9/5.
theorem largest_x : ∃ x : ℝ, equation_holds x ∧ ∀ y : ℝ, equation_holds y → y ≤ 9 / 5 :=
by
  sorry

end largest_x_l102_102355


namespace undefined_expression_l102_102441

theorem undefined_expression (a : ℝ) : (a = 3 ∨ a = -3) ↔ (a^2 - 9 = 0) := 
by
  sorry

end undefined_expression_l102_102441


namespace squared_expression_l102_102595

variable (x : ℝ)

theorem squared_expression (h : x + 1/x = 3) : x^2 + 1/x^2 = 7 := by
  sorry

end squared_expression_l102_102595


namespace largest_common_value_l102_102460

/-- The largest value less than 300 that appears in both sequences 
    {7, 14, 21, 28, ...} and {5, 15, 25, 35, ...} -/
theorem largest_common_value (a : ℕ) (n m k : ℕ) :
  (a = 7 * (1 + n)) ∧ (a = 5 + 10 * m) ∧ (a < 300) ∧ (∀ k, (55 + 70 * k < 300) → (55 + 70 * k) ≤ a) 
  → a = 265 :=
by
  sorry

end largest_common_value_l102_102460


namespace min_value_expression_l102_102009

theorem min_value_expression :
  ∀ x : ℝ, (x^2 + 9) / Real.sqrt (x^2 + 5) ≥ (5 * Real.sqrt 6) / 3 :=
by
  sorry

end min_value_expression_l102_102009


namespace sector_radius_l102_102669

theorem sector_radius (r : ℝ) (h1 : r > 0) 
  (h2 : ∀ (l : ℝ), l = r → 
    (3 * r) / (1 / 2 * r^2) = 2) : r = 3 := 
sorry

end sector_radius_l102_102669


namespace find_x_l102_102002

theorem find_x (x y z : ℕ) (h_pos : 0 < x) (h_pos : 0 < y) (h_pos : 0 < z) (h_eq1 : x + y + z = 37) (h_eq2 : 5 * y = 6 * z) : x = 21 :=
sorry

end find_x_l102_102002


namespace total_animals_hunted_l102_102572

theorem total_animals_hunted :
  let sam_hunts := 6
  let rob_hunts := sam_hunts / 2
  let total_sam_rob := sam_hunts + rob_hunts
  let mark_hunts := total_sam_rob / 3
  let peter_hunts := mark_hunts * 3
  sam_hunts + rob_hunts + mark_hunts + peter_hunts = 21 :=
by
  sorry

end total_animals_hunted_l102_102572


namespace find_R_l102_102270

theorem find_R (a b Q R : ℕ) (ha_prime : Prime a) (hb_prime : Prime b) (h_distinct : a ≠ b)
  (h1 : a^2 - a * Q + R = 0) (h2 : b^2 - b * Q + R = 0) : R = 6 :=
sorry

end find_R_l102_102270


namespace reliefSuppliesCalculation_l102_102369

noncomputable def totalReliefSupplies : ℝ := 644

theorem reliefSuppliesCalculation
    (A_capacity : ℝ)
    (B_capacity : ℝ)
    (A_capacity_per_day : A_capacity = 64.4)
    (capacity_ratio : A_capacity = 1.75 * B_capacity)
    (additional_transport : ∃ t : ℝ, A_capacity * t - B_capacity * t = 138 ∧ A_capacity * t = 322) :
  totalReliefSupplies = 644 := by
  sorry

end reliefSuppliesCalculation_l102_102369


namespace tax_percentage_l102_102358

-- Definitions
def salary_before_taxes := 5000
def rent_expense_per_month := 1350
def total_late_rent_payments := 2 * rent_expense_per_month
def fraction_of_next_salary_after_taxes := (3 / 5 : ℚ)

-- Main statement to prove
theorem tax_percentage (T : ℚ) : 
  fraction_of_next_salary_after_taxes * (salary_before_taxes - (T / 100) * salary_before_taxes) = total_late_rent_payments → 
  T = 10 :=
by
  sorry

end tax_percentage_l102_102358


namespace remaining_area_correct_l102_102676

noncomputable def remaining_area_ABHFGD : ℝ :=
  let area_square_ABCD := 25
  let area_square_EFGD := 16
  let side_length_ABCD := Real.sqrt area_square_ABCD
  let side_length_EFGD := Real.sqrt area_square_EFGD
  let overlap_area := 8
  area_square_ABCD + area_square_EFGD - overlap_area

theorem remaining_area_correct :
  let area := remaining_area_ABHFGD
  area = 33 :=
by
  sorry

end remaining_area_correct_l102_102676


namespace iggy_wednesday_run_6_l102_102222

open Nat

noncomputable def iggy_miles_wednesday : ℕ :=
  let total_time := 4 * 60    -- Iggy spends 4 hours running (240 minutes)
  let pace := 10              -- Iggy runs 1 mile in 10 minutes
  let monday := 3
  let tuesday := 4
  let thursday := 8
  let friday := 3
  let total_miles_other_days := monday + tuesday + thursday + friday
  let total_time_other_days := total_miles_other_days * pace
  let wednesday_time := total_time - total_time_other_days
  wednesday_time / pace

theorem iggy_wednesday_run_6 :
  iggy_miles_wednesday = 6 := by
  sorry

end iggy_wednesday_run_6_l102_102222


namespace total_tires_parking_lot_l102_102300

-- Definitions for each condition in a)
def four_wheel_drive_cars := 30
def motorcycles := 20
def six_wheel_trucks := 10
def bicycles := 5
def unicycles := 3
def baby_strollers := 2

def extra_roof_tires := 4
def flat_bike_tires_removed := 3
def extra_unicycle_wheel := 1

def tires_per_car := 4 + 1
def tires_per_motorcycle := 2 + 2
def tires_per_truck := 6 + 1
def tires_per_bicycle := 2
def tires_per_unicycle := 1
def tires_per_stroller := 4

-- Define total tires calculation
def total_tires (four_wheel_drive_cars motorcycles six_wheel_trucks bicycles unicycles baby_strollers 
                 extra_roof_tires flat_bike_tires_removed extra_unicycle_wheel : ℕ) :=
  (four_wheel_drive_cars * tires_per_car + extra_roof_tires) +
  (motorcycles * tires_per_motorcycle) +
  (six_wheel_trucks * tires_per_truck) +
  (bicycles * tires_per_bicycle - flat_bike_tires_removed) +
  (unicycles * tires_per_unicycle + extra_unicycle_wheel) +
  (baby_strollers * tires_per_stroller)

-- The Lean statement for the proof problem
theorem total_tires_parking_lot : 
  total_tires four_wheel_drive_cars motorcycles six_wheel_trucks bicycles unicycles baby_strollers 
              extra_roof_tires flat_bike_tires_removed extra_unicycle_wheel = 323 :=
by 
  sorry

end total_tires_parking_lot_l102_102300


namespace total_apples_purchased_l102_102476

theorem total_apples_purchased (M : ℝ) (T : ℝ) (W : ℝ) 
    (hM : M = 15.5)
    (hT : T = 3.2 * M)
    (hW : W = 1.05 * T) :
    M + T + W = 117.18 := by
  sorry

end total_apples_purchased_l102_102476


namespace find_solutions_of_equation_l102_102822

theorem find_solutions_of_equation (m n : ℝ) 
  (h1 : ∀ x, (x - m)^2 + n = 0 ↔ (x = -1 ∨ x = 3)) :
  (∀ x, (x - 1)^2 + m^2 = 2 * m * (x - 1) - n ↔ (x = 0 ∨ x = 4)) :=
by
  sorry

end find_solutions_of_equation_l102_102822


namespace cards_given_l102_102717

/-- Martha starts with 3 cards. She ends up with 79 cards after receiving some from Emily. We need to prove that Emily gave her 76 cards. -/
theorem cards_given (initial_cards final_cards cards_given : ℕ) (h1 : initial_cards = 3) (h2 : final_cards = 79) (h3 : final_cards = initial_cards + cards_given) :
  cards_given = 76 :=
sorry

end cards_given_l102_102717


namespace star_operation_example_l102_102125

-- Define the operation ☆
def star (a b : ℚ) : ℚ := a - b + 1

-- The theorem to prove
theorem star_operation_example : star (star 2 3) 2 = -1 := by
  sorry

end star_operation_example_l102_102125


namespace find_baking_soda_boxes_l102_102489

-- Define the quantities and costs
def num_flour_boxes := 3
def cost_per_flour_box := 3
def num_egg_trays := 3
def cost_per_egg_tray := 10
def num_milk_liters := 7
def cost_per_milk_liter := 5
def baking_soda_cost_per_box := 3
def total_cost := 80

-- Define the total cost of flour, eggs, and milk
def total_flour_cost := num_flour_boxes * cost_per_flour_box
def total_egg_cost := num_egg_trays * cost_per_egg_tray
def total_milk_cost := num_milk_liters * cost_per_milk_liter

-- Define the total cost of non-baking soda items
def total_non_baking_soda_cost := total_flour_cost + total_egg_cost + total_milk_cost

-- Define the remaining cost for baking soda
def baking_soda_total_cost := total_cost - total_non_baking_soda_cost

-- Define the number of baking soda boxes
def num_baking_soda_boxes := baking_soda_total_cost / baking_soda_cost_per_box

theorem find_baking_soda_boxes : num_baking_soda_boxes = 2 :=
by
  sorry

end find_baking_soda_boxes_l102_102489


namespace max_food_per_guest_l102_102110

theorem max_food_per_guest (total_food : ℕ) (min_guests : ℕ)
    (H1 : total_food = 406) (H2 : min_guests = 163) :
    2 ≤ total_food / min_guests ∧ total_food / min_guests < 3 := by
  sorry

end max_food_per_guest_l102_102110


namespace eval_f_at_10_l102_102047

def f (x : ℚ) : ℚ := (6 * x + 3) / (x - 2)

theorem eval_f_at_10 : f 10 = 63 / 8 :=
by
  sorry

end eval_f_at_10_l102_102047


namespace polygon_interior_exterior_eq_l102_102168

theorem polygon_interior_exterior_eq (n : ℕ) (hn : 3 ≤ n)
  (interior_sum_eq_exterior_sum : (n - 2) * 180 = 360) : n = 4 := by
  sorry

end polygon_interior_exterior_eq_l102_102168


namespace carlos_wins_one_game_l102_102506

def games_Won_Laura : ℕ := 5
def games_Lost_Laura : ℕ := 4
def games_Won_Mike : ℕ := 7
def games_Lost_Mike : ℕ := 2
def games_Lost_Carlos : ℕ := 5
variable (C : ℕ) -- Carlos's wins

theorem carlos_wins_one_game :
  games_Won_Laura + games_Won_Mike + C = (games_Won_Laura + games_Lost_Laura + games_Won_Mike + games_Lost_Mike + C + games_Lost_Carlos) / 2 →
  C = 1 :=
by
  sorry

end carlos_wins_one_game_l102_102506


namespace parallelogram_sides_l102_102495

theorem parallelogram_sides (x y : ℝ) 
  (h1 : 5 * x - 7 = 14) 
  (h2 : 3 * y + 4 = 8 * y - 3) : 
  x + y = 5.6 :=
sorry

end parallelogram_sides_l102_102495


namespace find_a_value_l102_102191

noncomputable def prob_sum_equals_one (a : ℝ) : Prop :=
  a * (1/2 + 1/4 + 1/8 + 1/16) = 1

theorem find_a_value (a : ℝ) (h : prob_sum_equals_one a) : a = 16/15 :=
sorry

end find_a_value_l102_102191


namespace inverse_var_q_value_l102_102672

theorem inverse_var_q_value (p q : ℝ) (h1 : ∀ p q, (p * q = 400))
(p_init : p = 800) (q_init : q = 0.5) (new_p : p = 400) :
  q = 1 := by
  sorry

end inverse_var_q_value_l102_102672


namespace expression_value_l102_102332

theorem expression_value : (4 - 2) ^ 3 = 8 :=
by sorry

end expression_value_l102_102332


namespace probability_red_red_red_l102_102312

-- Definition of probability for picking three red balls without replacement
def total_balls := 21
def red_balls := 7
def blue_balls := 9
def green_balls := 5

theorem probability_red_red_red : 
  (red_balls / total_balls) * ((red_balls - 1) / (total_balls - 1)) * ((red_balls - 2) / (total_balls - 2)) = 1 / 38 := 
by sorry

end probability_red_red_red_l102_102312


namespace intersection_of_A_and_B_is_5_and_8_l102_102700

def A : Set ℕ := {4, 5, 6, 8}
def B : Set ℕ := {5, 7, 8, 9}

theorem intersection_of_A_and_B_is_5_and_8 : A ∩ B = {5, 8} :=
  by sorry

end intersection_of_A_and_B_is_5_and_8_l102_102700


namespace isabelle_weeks_needed_l102_102410

def total_ticket_cost : ℕ := 20 + 10 + 10
def total_savings : ℕ := 5 + 5
def weekly_earnings : ℕ := 3
def amount_needed : ℕ := total_ticket_cost - total_savings
def weeks_needed : ℕ := amount_needed / weekly_earnings

theorem isabelle_weeks_needed 
  (ticket_cost_isabelle : ℕ := 20)
  (ticket_cost_brother : ℕ := 10)
  (savings_brothers : ℕ := 5)
  (savings_isabelle : ℕ := 5)
  (earnings_weekly : ℕ := 3)
  (total_cost := ticket_cost_isabelle + 2 * ticket_cost_brother)
  (total_savings := savings_brothers + savings_isabelle)
  (needed_amount := total_cost - total_savings)
  (weeks := needed_amount / earnings_weekly) :
  weeks = 10 :=
  by
  sorry

end isabelle_weeks_needed_l102_102410


namespace find_remainder_l102_102126

-- Define the numbers
def a := 98134
def b := 98135
def c := 98136
def d := 98137
def e := 98138
def f := 98139

-- Theorem statement
theorem find_remainder :
  (a + b + c + d + e + f) % 9 = 3 :=
by {
  sorry
}

end find_remainder_l102_102126


namespace percentage_tax_raise_expecting_population_l102_102752

def percentage_affirmative_responses_tax : ℝ := 0.4
def percentage_affirmative_responses_money : ℝ := 0.3
def percentage_affirmative_responses_bonds : ℝ := 0.5
def percentage_affirmative_responses_gold : ℝ := 0.0

def fraction_liars : ℝ := 0.1
def fraction_economists : ℝ := 1 - fraction_liars

theorem percentage_tax_raise_expecting_population : 
  (percentage_affirmative_responses_tax - fraction_liars) = 0.3 :=
by
  sorry

end percentage_tax_raise_expecting_population_l102_102752


namespace loan_difference_l102_102827

noncomputable def future_value (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

noncomputable def monthly_compounding : ℝ :=
  future_value 8000 0.10 12 5

noncomputable def semi_annual_compounding : ℝ :=
  future_value 8000 0.10 2 5

noncomputable def interest_difference : ℝ :=
  monthly_compounding - semi_annual_compounding

theorem loan_difference (P : ℝ) (r : ℝ) (n_m n_s t : ℝ) :
    interest_difference = 745.02 := by sorry

end loan_difference_l102_102827


namespace a_and_c_can_complete_in_20_days_l102_102677

-- Define the work rates for the pairs given in the conditions.
variables {A B C : ℚ}

-- a and b together can complete the work in 12 days
axiom H1 : A + B = 1 / 12

-- b and c together can complete the work in 15 days
axiom H2 : B + C = 1 / 15

-- a, b, and c together can complete the work in 10 days
axiom H3 : A + B + C = 1 / 10

-- We aim to prove that a and c together can complete the work in 20 days,
-- hence their combined work rate should be 1 / 20.
theorem a_and_c_can_complete_in_20_days : A + C = 1 / 20 :=
by
  -- sorry will be used to skip the proof
  sorry

end a_and_c_can_complete_in_20_days_l102_102677


namespace find_AD_l102_102214

theorem find_AD
  (A B C D : Type)
  (BD BC CD AD : ℝ)
  (hBD : BD = 21)
  (hBC : BC = 30)
  (hCD : CD = 15)
  (hAngleBisect : true) -- Encode that D bisects the angle at C internally
  : AD = 35 := by
  sorry

end find_AD_l102_102214


namespace find_judy_rotation_l102_102195

-- Definition of the problem
def CarlaRotation := 480 % 360 -- This effectively becomes 120
def JudyRotation (y : ℕ) := (360 - 120) % 360 -- This should effectively be 240

-- Theorem stating the problem and solution
theorem find_judy_rotation (y : ℕ) (h : y < 360) : 360 - CarlaRotation = y :=
by 
  dsimp [CarlaRotation, JudyRotation] 
  sorry

end find_judy_rotation_l102_102195


namespace infinite_x_differs_from_two_kth_powers_l102_102626

theorem infinite_x_differs_from_two_kth_powers (k : ℕ) (h : k > 1) : 
  ∃ (f : ℕ → ℕ), (∀ n, f n = (2^(n+1))^k - (2^n)^k) ∧ (∀ n, ∀ a b : ℕ, ¬ f n = a^k + b^k) :=
sorry

end infinite_x_differs_from_two_kth_powers_l102_102626


namespace positive_difference_1010_1000_l102_102528

-- Define the arithmetic sequence
def arithmetic_sequence (a d n : ℕ) : ℕ :=
  a + (n - 1) * d

-- Define the specific terms
def a_1000 := arithmetic_sequence 5 7 1000
def a_1010 := arithmetic_sequence 5 7 1010

-- Proof statement
theorem positive_difference_1010_1000 : a_1010 - a_1000 = 70 :=
by
  sorry

end positive_difference_1010_1000_l102_102528


namespace pyramid_top_value_l102_102337

theorem pyramid_top_value 
  (p : ℕ) (q : ℕ) (z : ℕ) (m : ℕ) (n : ℕ) (left_mid : ℕ) (right_mid : ℕ) 
  (left_upper : ℕ) (right_upper : ℕ) (x_pre : ℕ) (x : ℕ) : 
  p = 20 → 
  q = 6 → 
  z = 44 → 
  m = p + 34 → 
  n = q + z → 
  left_mid = 17 + 29 → 
  right_mid = m + n → 
  left_upper = 36 + left_mid → 
  right_upper = right_mid + 42 → 
  x_pre = left_upper + 78 → 
  x = 2 * x_pre → 
  x = 320 :=
by
  intros
  sorry

end pyramid_top_value_l102_102337


namespace exists_product_sum_20000_l102_102210

theorem exists_product_sum_20000 :
  ∃ (k m : ℕ), 1 ≤ k ∧ k ≤ 999 ∧ 1 ≤ m ∧ m ≤ 999 ∧ k * (k + 1) + m * (m + 1) = 20000 :=
by 
  sorry

end exists_product_sum_20000_l102_102210


namespace avg_cost_equals_0_22_l102_102109

-- Definitions based on conditions
def num_pencils : ℕ := 150
def cost_pencils : ℝ := 24.75
def shipping_cost : ℝ := 8.50

-- Calculating total cost and average cost
noncomputable def total_cost : ℝ := cost_pencils + shipping_cost
noncomputable def avg_cost_per_pencil : ℝ := total_cost / num_pencils

-- Lean theorem statement
theorem avg_cost_equals_0_22 : avg_cost_per_pencil = 0.22 :=
by
  sorry

end avg_cost_equals_0_22_l102_102109


namespace rope_cutting_impossible_l102_102412

/-- 
Given a rope initially cut into 5 pieces, and then some of these pieces were each cut into 
5 parts, with this process repeated several times, it is not possible for the total 
number of pieces to be exactly 2019.
-/ 
theorem rope_cutting_impossible (n : ℕ) : 5 + 4 * n ≠ 2019 := 
sorry

end rope_cutting_impossible_l102_102412


namespace distance_between_parallel_sides_l102_102459

-- Define the givens
def length_side_a : ℝ := 24  -- length of one parallel side
def length_side_b : ℝ := 14  -- length of the other parallel side
def area_trapezium : ℝ := 342  -- area of the trapezium

-- We need to prove that the distance between parallel sides (h) is 18 cm
theorem distance_between_parallel_sides (h : ℝ)
  (H1 :  area_trapezium = (1/2) * (length_side_a + length_side_b) * h) :
  h = 18 :=
by sorry

end distance_between_parallel_sides_l102_102459


namespace grass_field_width_l102_102318

theorem grass_field_width (w : ℝ) (length_field : ℝ) (path_width : ℝ) (area_path : ℝ) :
  length_field = 85 → path_width = 2.5 → area_path = 1450 →
  (90 * (w + path_width * 2) - length_field * w = area_path) → w = 200 :=
by
  intros h_length_field h_path_width h_area_path h_eq
  sorry

end grass_field_width_l102_102318


namespace find_m_l102_102850

theorem find_m (x m : ℝ) (h1 : 4 * x + 2 * m = 5 * x + 1) (h2 : 3 * x = 6 * x - 1) : m = 2 / 3 :=
by
  sorry

end find_m_l102_102850


namespace find_m_l102_102514

-- Definitions
variable {A B C O H : Type}
variable {O_is_circumcenter : is_circumcenter O A B C}
variable {H_is_altitude_intersection : is_altitude_intersection H A B C}
variable (AH BH CH OA OB OC : ℝ)

-- Problem Statement
theorem find_m (h : AH * BH * CH = m * (OA * OB * OC)) : m = 1 :=
sorry

end find_m_l102_102514


namespace state_tax_percentage_l102_102431

theorem state_tax_percentage (weekly_salary federal_percent health_insurance life_insurance parking_fee final_paycheck : ℝ)
  (h_weekly_salary : weekly_salary = 450)
  (h_federal_percent : federal_percent = 1/3)
  (h_health_insurance : health_insurance = 50)
  (h_life_insurance : life_insurance = 20)
  (h_parking_fee : parking_fee = 10)
  (h_final_paycheck : final_paycheck = 184) :
  (36 / 450) * 100 = 8 :=
by
  sorry

end state_tax_percentage_l102_102431


namespace slower_pipe_fills_tank_in_200_minutes_l102_102500

noncomputable def slower_pipe_filling_time (F S : ℝ) (h1 : F = 4 * S) (h2 : F + S = 1 / 40) : ℝ :=
  1 / S

theorem slower_pipe_fills_tank_in_200_minutes (F S : ℝ) (h1 : F = 4 * S) (h2 : F + S = 1 / 40) :
  slower_pipe_filling_time F S h1 h2 = 200 :=
sorry

end slower_pipe_fills_tank_in_200_minutes_l102_102500


namespace surface_area_of_reassembled_solid_l102_102838

noncomputable def total_surface_area : ℕ :=
let height_E := 1/4
let height_F := 1/6
let height_G := 1/9 
let height_H := 1 - (height_E + height_F + height_G)
let face_area := 2 * 1
(face_area * 2)     -- Top and bottom surfaces
+ 2                -- Side surfaces (1 foot each side * 2 sides)
+ (face_area * 2)   -- Front and back surfaces 

theorem surface_area_of_reassembled_solid :
  total_surface_area = 10 :=
by
  sorry

end surface_area_of_reassembled_solid_l102_102838


namespace polar_equation_of_circle_c_range_of_op_oq_l102_102813

noncomputable def circle_param_eq (φ : ℝ) : ℝ × ℝ :=
  (1 + Real.cos φ, Real.sin φ)

noncomputable def line_kl_eq (θ : ℝ) : ℝ :=
  3 * Real.sqrt 3 / (Real.sin θ + Real.sqrt 3 * Real.cos θ)

theorem polar_equation_of_circle_c :
  ∀ θ : ℝ, ∃ ρ : ℝ, ρ = 2 * Real.cos θ :=
by sorry

theorem range_of_op_oq (θ₁ : ℝ) (hθ : 0 < θ₁ ∧ θ₁ < Real.pi / 2) :
  0 < (2 * Real.cos θ₁) * (3 * Real.sqrt 3 / (Real.sin θ₁ + Real.sqrt 3 * Real.cos θ₁)) ∧
  (2 * Real.cos θ₁) * (3 * Real.sqrt 3 / (Real.sin θ₁ + Real.sqrt 3 * Real.cos θ₁)) < 6 :=
by sorry

end polar_equation_of_circle_c_range_of_op_oq_l102_102813


namespace remainder_when_dividing_698_by_13_is_9_l102_102884

theorem remainder_when_dividing_698_by_13_is_9 :
  ∃ k m : ℤ, 242 = k * 13 + 8 ∧
             698 = m * 13 + 9 ∧
             (k + m) * 13 + 4 = 940 :=
by {
  sorry
}

end remainder_when_dividing_698_by_13_is_9_l102_102884


namespace maximize_distance_l102_102673

theorem maximize_distance (D_F D_R : ℕ) (x y : ℕ) (h1 : D_F = 21000) (h2 : D_R = 28000)
  (h3 : x + y ≤ D_F) (h4 : x + y ≤ D_R) :
  x + y = 24000 :=
sorry

end maximize_distance_l102_102673


namespace chef_earns_2_60_less_l102_102082

/--
At Joe's Steakhouse, the hourly wage for a chef is 20% greater than that of a dishwasher,
and the hourly wage of a dishwasher is half as much as the hourly wage of a manager.
If a manager's wage is $6.50 per hour, prove that a chef earns $2.60 less per hour than a manager.
-/
theorem chef_earns_2_60_less {w_manager w_dishwasher w_chef : ℝ} 
  (h1 : w_dishwasher = w_manager / 2)
  (h2 : w_chef = w_dishwasher * 1.20)
  (h3 : w_manager = 6.50) :
  w_manager - w_chef = 2.60 :=
by
  sorry

end chef_earns_2_60_less_l102_102082


namespace polynomial_value_l102_102574

theorem polynomial_value : 103^4 - 4 * 103^3 + 6 * 103^2 - 4 * 103 + 1 = 108243216 :=
by sorry

end polynomial_value_l102_102574
