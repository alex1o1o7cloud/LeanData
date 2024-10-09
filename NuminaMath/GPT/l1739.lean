import Mathlib

namespace determine_b_l1739_173978

noncomputable def has_exactly_one_real_solution (b : ℝ) : Prop :=
  ∃ x : ℝ, x^4 - b*x^3 - 3*b*x + b^2 - 2 = 0 ∧ ∀ y : ℝ, y ≠ x → y^4 - b*y^3 - 3*b*y + b^2 - 2 ≠ 0

theorem determine_b (b : ℝ) :
  has_exactly_one_real_solution b → b < 7 / 4 :=
by
  sorry

end determine_b_l1739_173978


namespace slope_intercept_parallel_line_l1739_173986

def is_parallel (m1 m2 : ℝ) : Prop :=
  m1 = m2

theorem slope_intercept_parallel_line (A : ℝ × ℝ) (hA₁ : A.1 = 3) (hA₂ : A.2 = 2) 
  (m : ℝ) (h_parallel : is_parallel m (-4)) : ∃ b : ℝ, ∀ x y : ℝ, y = -4 * x + b :=
by
  use 14
  intro x y
  sorry

end slope_intercept_parallel_line_l1739_173986


namespace certain_event_birthday_example_l1739_173940
-- Import the necessary library

-- Define the problem with conditions
def certain_event_people_share_birthday (num_days : ℕ) (num_people : ℕ) : Prop :=
  num_people > num_days

-- Define a specific instance based on the given problem
theorem certain_event_birthday_example : certain_event_people_share_birthday 365 400 :=
by
  sorry

end certain_event_birthday_example_l1739_173940


namespace range_of_m_l1739_173942

def P (m : ℝ) : Prop := m^2 - 4 > 0
def Q (m : ℝ) : Prop := 16 * (m - 2)^2 - 16 < 0

theorem range_of_m (m : ℝ) : ¬(P m ∧ Q m) ∧ (P m ∨ Q m) ↔ (m < -2 ∨ (1 < m ∧ m ≤ 2) ∨ m ≥ 3) :=
by
  sorry

end range_of_m_l1739_173942


namespace notebooks_to_sell_to_earn_profit_l1739_173919

-- Define the given conditions
def notebooks_purchased : ℕ := 2000
def cost_per_notebook : ℚ := 0.15
def selling_price_per_notebook : ℚ := 0.30
def desired_profit : ℚ := 120

-- Define the total cost
def total_cost := notebooks_purchased * cost_per_notebook

-- Define the total revenue needed
def total_revenue_needed := total_cost + desired_profit

-- Define the number of notebooks to be sold to achieve the total revenue
def notebooks_to_sell := total_revenue_needed / selling_price_per_notebook

-- Prove that the number of notebooks to be sold is 1400 to make a profit of $120
theorem notebooks_to_sell_to_earn_profit : notebooks_to_sell = 1400 := 
by {
  sorry
}

end notebooks_to_sell_to_earn_profit_l1739_173919


namespace hyperbola_foci_distance_l1739_173933

-- Define the hyperbola equation
def hyperbola_eq (x y : ℝ) : Prop := x^2 - 4 * x - 9 * y^2 - 18 * y = 56

-- Define the distance between the foci of the hyperbola
def distance_between_foci (d : ℝ) : Prop :=
  d = 2 * Real.sqrt (170 / 3)

-- The theorem stating that the distance between the foci of the given hyperbola
theorem hyperbola_foci_distance :
  ∃ d, hyperbola_eq x y → distance_between_foci d :=
by { sorry }

end hyperbola_foci_distance_l1739_173933


namespace simplify_fraction_l1739_173949

open Complex

theorem simplify_fraction :
  (7 + 9 * I) / (3 - 4 * I) = 2.28 + 2.2 * I := 
by {
    -- We know that this should be true based on the provided solution,
    -- but we will place a placeholder here for the actual proof.
    sorry
}

end simplify_fraction_l1739_173949


namespace women_in_third_group_l1739_173911

variables (m w : ℝ)

theorem women_in_third_group (h1 : 3 * m + 8 * w = 6 * m + 2 * w) (x : ℝ) (h2 : 2 * m + x * w = 0.5 * (3 * m + 8 * w)) :
  x = 4 :=
sorry

end women_in_third_group_l1739_173911


namespace john_apartment_number_l1739_173990

variable (k d m : ℕ)

theorem john_apartment_number (h1 : k = m) (h2 : d + m = 239) (h3 : 10 * (k - 1) + 1 ≤ d) (h4 : d ≤ 10 * k) : d = 217 := 
by 
  sorry

end john_apartment_number_l1739_173990


namespace ratio_sum_odd_even_divisors_l1739_173968

def M : ℕ := 33 * 38 * 58 * 462

theorem ratio_sum_odd_even_divisors : 
  let sum_odd_divisors := 
    (1 + 3 + 3^2) * (1 + 7) * (1 + 11 + 11^2) * (1 + 19) * (1 + 29)
  let sum_all_divisors := 
    (1 + 2 + 4 + 8) * (1 + 3 + 3^2) * (1 + 7) * (1 + 11 + 11^2) * (1 + 19) * (1 + 29)
  let sum_even_divisors := sum_all_divisors - sum_odd_divisors
  (sum_odd_divisors : ℚ) / sum_even_divisors = 1 / 14 :=
by sorry

end ratio_sum_odd_even_divisors_l1739_173968


namespace minimum_toothpicks_to_remove_l1739_173918

-- Conditions
def number_of_toothpicks : ℕ := 60
def largest_triangle_side : ℕ := 3
def smallest_triangle_side : ℕ := 1

-- Problem Statement
theorem minimum_toothpicks_to_remove (toothpicks_total : ℕ) (largest_side : ℕ) (smallest_side : ℕ) 
  (h1 : toothpicks_total = 60) 
  (h2 : largest_side = 3) 
  (h3 : smallest_side = 1) : 
  ∃ n : ℕ, n = 20 := by
  sorry

end minimum_toothpicks_to_remove_l1739_173918


namespace total_number_of_soccer_games_l1739_173951

theorem total_number_of_soccer_games (teams : ℕ)
  (regular_games_per_team : ℕ)
  (promotional_games_per_team : ℕ)
  (h1 : teams = 15)
  (h2 : regular_games_per_team = 14)
  (h3 : promotional_games_per_team = 2) :
  ((teams * regular_games_per_team) / 2 + (teams * promotional_games_per_team) / 2) = 120 :=
by
  sorry

end total_number_of_soccer_games_l1739_173951


namespace maximize_revenue_l1739_173923

def revenue_function (p : ℝ) : ℝ :=
  p * (200 - 6 * p)

theorem maximize_revenue :
  ∃ (p : ℝ), (p ≤ 30) ∧ (∀ q : ℝ, (q ≤ 30) → revenue_function p ≥ revenue_function q) ∧ p = 50 / 3 :=
by
  sorry

end maximize_revenue_l1739_173923


namespace average_wage_correct_l1739_173952

def male_workers : ℕ := 20
def female_workers : ℕ := 15
def child_workers : ℕ := 5

def male_wage : ℕ := 25
def female_wage : ℕ := 20
def child_wage : ℕ := 8

def total_amount_paid_per_day : ℕ := 
  (male_workers * male_wage) + (female_workers * female_wage) + (child_workers * child_wage)

def total_number_of_workers : ℕ := 
  male_workers + female_workers + child_workers

def average_wage_per_day : ℕ := 
  total_amount_paid_per_day / total_number_of_workers

theorem average_wage_correct : 
  average_wage_per_day = 21 := by 
  sorry

end average_wage_correct_l1739_173952


namespace cyclist_average_rate_l1739_173974

noncomputable def average_rate_round_trip (D : ℝ) : ℝ :=
  let time_to_travel := D / 10
  let time_to_return := D / 9
  let total_distance := 2 * D
  let total_time := time_to_travel + time_to_return
  (total_distance / total_time)

theorem cyclist_average_rate (D : ℝ) (hD : D > 0) :
  average_rate_round_trip D = 180 / 19 :=
by
  sorry

end cyclist_average_rate_l1739_173974


namespace find_acute_angle_of_parallel_vectors_l1739_173914

open Real

theorem find_acute_angle_of_parallel_vectors (x : ℝ) (hx1 : (sin x) * (1 / 2 * cos x) = 1 / 4) (hx2 : 0 < x ∧ x < π / 2) : x = π / 4 :=
by
  sorry

end find_acute_angle_of_parallel_vectors_l1739_173914


namespace perry_more_games_than_phil_l1739_173901

theorem perry_more_games_than_phil (dana_wins charlie_wins perry_wins : ℕ) :
  perry_wins = dana_wins + 5 →
  charlie_wins = dana_wins - 2 →
  charlie_wins + 3 = 12 →
  perry_wins - 12 = 4 :=
by
  sorry

end perry_more_games_than_phil_l1739_173901


namespace linear_eq_represents_plane_l1739_173930

theorem linear_eq_represents_plane (A B C : ℝ) (h : ¬ (A = 0 ∧ B = 0 ∧ C = 0)) :
  ∃ (P : ℝ × ℝ × ℝ → Prop), (∀ (x y z : ℝ), P (x, y, z) ↔ A * x + B * y + C * z = 0) ∧ 
  (P (0, 0, 0)) :=
by
  -- To be filled in with the proof steps
  sorry

end linear_eq_represents_plane_l1739_173930


namespace lines_intersect_at_l1739_173912

theorem lines_intersect_at :
  ∃ (x y : ℚ), 3 * y = -2 * x + 6 ∧ 7 * y = -3 * x - 4 ∧ x = 54 / 5 ∧ y = -26 / 5 := 
by
  sorry

end lines_intersect_at_l1739_173912


namespace find_a_values_l1739_173965

theorem find_a_values (a t t₁ t₂ : ℝ) :
  (t^2 + (a - 6) * t + (9 - 3 * a) = 0) ∧
  (t₁ = 4 * t₂) ∧
  (t₁ + t₂ = 6 - a) ∧
  (t₁ * t₂ = 9 - 3 * a)
  ↔ (a = -2 ∨ a = 2) := sorry

end find_a_values_l1739_173965


namespace martin_big_bell_rings_l1739_173909

theorem martin_big_bell_rings (B S : ℚ) (h1 : S = B / 3 + B^2 / 4) (h2 : S + B = 52) : B = 12 :=
by
  sorry

end martin_big_bell_rings_l1739_173909


namespace solve_inequality_system_l1739_173945

theorem solve_inequality_system (x : ℝ) :
  (8 * x - 3 ≤ 13) ∧ ((x - 1) / 3 - 2 < x - 1) → -2 < x ∧ x ≤ 2 :=
by
  intros h
  sorry

end solve_inequality_system_l1739_173945


namespace sum_of_relatively_prime_integers_l1739_173938

theorem sum_of_relatively_prime_integers (x y : ℕ) (h1 : 0 < x) (h2 : 0 < y)
  (h3 : x * y + x + y = 154) (h4 : Nat.gcd x y = 1) (h5 : x < 30) (h6 : y < 30) : 
  x + y = 34 :=
sorry -- proof

end sum_of_relatively_prime_integers_l1739_173938


namespace lucas_earnings_l1739_173980

-- Declare constants and definitions given in the problem
def dollars_per_window : ℕ := 3
def windows_per_floor : ℕ := 5
def floors : ℕ := 4
def penalty_amount : ℕ := 2
def days_per_period : ℕ := 4
def total_days : ℕ := 12

-- Definition of the number of total windows
def total_windows : ℕ := windows_per_floor * floors

-- Initial earnings before penalties
def initial_earnings : ℕ := total_windows * dollars_per_window

-- Number of penalty periods
def penalty_periods : ℕ := total_days / days_per_period

-- Total penalty amount
def total_penalty : ℕ := penalty_periods * penalty_amount

-- Final earnings after penalties
def final_earnings : ℕ := initial_earnings - total_penalty

-- Proof problem: correct amount Lucas' father will pay
theorem lucas_earnings : final_earnings = 54 :=
by
  sorry

end lucas_earnings_l1739_173980


namespace sally_cut_red_orchids_l1739_173958

-- Definitions and conditions
def initial_red_orchids := 9
def orchids_in_vase_after_cutting := 15

-- Problem statement
theorem sally_cut_red_orchids : (orchids_in_vase_after_cutting - initial_red_orchids) = 6 := by
  sorry

end sally_cut_red_orchids_l1739_173958


namespace day_in_43_days_is_wednesday_l1739_173905

-- Define a function to represent the day of the week after a certain number of days
def day_of_week (n : ℕ) : ℕ := n % 7

-- Use an enum or some notation to represent the days of the week, but this is implicit in our setup.
-- We assume the days are numbered from 0 to 6 with 0 representing Tuesday.
def Tuesday : ℕ := 0
def Wednesday : ℕ := 1

-- Theorem to prove that 43 days after Tuesday is a Wednesday
theorem day_in_43_days_is_wednesday : day_of_week (Tuesday + 43) = Wednesday :=
by
  sorry

end day_in_43_days_is_wednesday_l1739_173905


namespace smallest_consecutive_natural_number_sum_l1739_173999

theorem smallest_consecutive_natural_number_sum (a n : ℕ) (hn : n > 1) (h : n * a + (n * (n - 1)) / 2 = 2016) :
  ∃ a, a = 1 :=
by
  sorry

end smallest_consecutive_natural_number_sum_l1739_173999


namespace total_juice_sold_3_days_l1739_173941

def juice_sales_problem (V_L V_M V_S : ℕ) (d1 d2 d3 : ℕ) :=
  (d1 = V_L + 4 * V_M) ∧ 
  (d2 = 2 * V_L + 6 * V_S) ∧ 
  (d3 = V_L + 3 * V_M + 3 * V_S) ∧
  (d1 = d2) ∧
  (d2 = d3)

theorem total_juice_sold_3_days (V_L V_M V_S d1 d2 d3 : ℕ) 
  (h : juice_sales_problem V_L V_M V_S d1 d2 d3) 
  (h_VM : V_M = 3) 
  (h_VL : V_L = 6) : 
  3 * d1 = 54 := 
by 
  -- Proof will be filled in
  sorry

end total_juice_sold_3_days_l1739_173941


namespace problem_statement_l1739_173983

noncomputable def f (x : ℝ) : ℝ := x + 1 / x - Real.sqrt 2

theorem problem_statement (x : ℝ) (h₁ : x ∈ Set.Ioc (Real.sqrt 2 / 2) 1) :
  Real.sqrt 2 / 2 < f (f x) ∧ f (f x) < x :=
by
  sorry

end problem_statement_l1739_173983


namespace sequence_general_term_l1739_173994

theorem sequence_general_term 
  (x : ℕ → ℝ)
  (h1 : x 1 = 2)
  (h2 : x 2 = 3)
  (h3 : ∀ m ≥ 1, x (2*m+1) = x (2*m) + x (2*m-1))
  (h4 : ∀ m ≥ 2, x (2*m) = x (2*m-1) + 2*x (2*m-2)) :
  ∀ m, (x (2*m-1) = ((3 - Real.sqrt 2) / 4) * (2 + Real.sqrt 2) ^ m + ((3 + Real.sqrt 2) / 4) * (2 - Real.sqrt 2) ^ m ∧ 
          x (2*m) = ((1 + 2 * Real.sqrt 2) / 4) * (2 + Real.sqrt 2) ^ m + ((1 - 2 * Real.sqrt 2) / 4) * (2 - Real.sqrt 2) ^ m) :=
sorry

end sequence_general_term_l1739_173994


namespace river_flow_rate_l1739_173966

variables (d w : ℝ) (V : ℝ)

theorem river_flow_rate (h₁ : d = 4) (h₂ : w = 40) (h₃ : V = 10666.666666666666) :
  ((V / 60) / (d * w) * 3.6) = 4 :=
by sorry

end river_flow_rate_l1739_173966


namespace max_expression_value_l1739_173997

theorem max_expression_value : 
  ∃ a b c d e f : ℕ, 1 ≤ a ∧ a ≤ 6 ∧
                   1 ≤ b ∧ b ≤ 6 ∧
                   1 ≤ c ∧ c ≤ 6 ∧
                   1 ≤ d ∧ d ≤ 6 ∧
                   1 ≤ e ∧ e ≤ 6 ∧
                   1 ≤ f ∧ f ≤ 6 ∧
                   a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
                   b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
                   c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
                   d ≠ e ∧ d ≠ f ∧
                   e ≠ f ∧
                   (f * (a * d + b * c) / (b * d * e) = 14) :=
sorry

end max_expression_value_l1739_173997


namespace percentage_sold_is_80_l1739_173953

-- Definitions corresponding to conditions
def first_day_houses : Nat := 20
def items_per_house : Nat := 2
def total_items_sold : Nat := 104

-- Calculate the houses visited on the second day
def second_day_houses : Nat := 2 * first_day_houses

-- Calculate items sold on the first day
def items_sold_first_day : Nat := first_day_houses * items_per_house

-- Calculate items sold on the second day
def items_sold_second_day : Nat := total_items_sold - items_sold_first_day

-- Calculate houses sold to on the second day
def houses_sold_to_second_day : Nat := items_sold_second_day / items_per_house

-- Percentage calculation
def percentage_sold_second_day : Nat := (houses_sold_to_second_day * 100) / second_day_houses

-- Theorem proving that James sold to 80% of the houses on the second day
theorem percentage_sold_is_80 : percentage_sold_second_day = 80 := by
  sorry

end percentage_sold_is_80_l1739_173953


namespace fraction_to_decimal_l1739_173956

theorem fraction_to_decimal : (7 / 16 : ℝ) = 0.4375 := by
  sorry

end fraction_to_decimal_l1739_173956


namespace solve_for_x_l1739_173929

theorem solve_for_x (x : ℝ) (h : (2 * x - 3) ^ (x + 3) = 1) : 
  x = -3 ∨ x = 2 ∨ x = 1 := 
sorry

end solve_for_x_l1739_173929


namespace intersection_value_unique_l1739_173993

theorem intersection_value_unique (x : ℝ) :
  (∃ y : ℝ, y = 8 / (x^2 + 4) ∧ x + y = 2) → x = 0 :=
by
  sorry

end intersection_value_unique_l1739_173993


namespace am_gm_example_l1739_173934

theorem am_gm_example {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a * b * c = 1) :
  (a^2 + 1) * (b^3 + 2) * (c^5 + 4) ≥ 30 :=
sorry

end am_gm_example_l1739_173934


namespace triangle_angle_ratio_l1739_173954

theorem triangle_angle_ratio (a b c : ℝ) (h₁ : a + b + c = 180)
  (h₂ : b = 2 * a) (h₃ : c = 3 * a) : a = 30 ∧ b = 60 ∧ c = 90 :=
by
  sorry

end triangle_angle_ratio_l1739_173954


namespace number_of_new_bricks_l1739_173961

-- Definitions from conditions
def edge_length_original_brick : ℝ := 0.3
def edge_length_new_brick : ℝ := 0.5
def number_original_bricks : ℕ := 600

-- The classroom volume is unchanged, so we set up a proportion problem
-- Assuming the classroom is fully paved
theorem number_of_new_bricks :
  let volume_original_bricks := number_original_bricks * (edge_length_original_brick ^ 2)
  let volume_new_bricks := x * (edge_length_new_brick ^ 2)
  volume_original_bricks = volume_new_bricks → x = 216 := 
by
  sorry

end number_of_new_bricks_l1739_173961


namespace lesser_of_two_numbers_l1739_173957

theorem lesser_of_two_numbers (a b : ℕ) (h₁ : a + b = 55) (h₂ : a - b = 7) (h₃ : a > b) : b = 24 :=
by
  sorry

end lesser_of_two_numbers_l1739_173957


namespace jill_braids_dancers_l1739_173987

def dancers_on_team (braids_per_dancer : ℕ) (seconds_per_braid : ℕ) (total_time_seconds : ℕ) : ℕ :=
  total_time_seconds / seconds_per_braid / braids_per_dancer

theorem jill_braids_dancers (h1 : braids_per_dancer = 5) (h2 : seconds_per_braid = 30)
                             (h3 : total_time_seconds = 20 * 60) : 
  dancers_on_team braids_per_dancer seconds_per_braid total_time_seconds = 8 :=
by
  sorry

end jill_braids_dancers_l1739_173987


namespace distance_covered_l1739_173964

-- Definitions
def speed : ℕ := 150  -- Speed in km/h
def time : ℕ := 8  -- Time in hours

-- Proof statement
theorem distance_covered : speed * time = 1200 := 
by
  sorry

end distance_covered_l1739_173964


namespace circle_radius_l1739_173992

theorem circle_radius (a c r : ℝ) (h₁ : a = π * r^2) (h₂ : c = 2 * π * r) (h₃ : a + c = 100 * π) : 
  r = 9.05 := 
sorry

end circle_radius_l1739_173992


namespace range_of_a_l1739_173926

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x = 1 → a * x^2 + 2 * x + 1 < 0) ↔ a < -3 :=
by
  sorry

end range_of_a_l1739_173926


namespace triangle_inequality_inequality_l1739_173972

theorem triangle_inequality_inequality {a b c : ℝ}
  (h1 : a + b > c) (h2 : b + c > a) (h3 : a + c > b) :
  3 * (b + c - a) * (c + a - b) * (a + b - c) ≤ a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) :=
sorry

end triangle_inequality_inequality_l1739_173972


namespace initial_number_is_11_l1739_173988

theorem initial_number_is_11 :
  ∃ (N : ℤ), ∃ (k : ℤ), N - 11 = 17 * k ∧ N = 11 :=
by
  sorry

end initial_number_is_11_l1739_173988


namespace find_initial_amount_l1739_173922

-- Let x be the initial amount Mark paid for the Magic card
variable {x : ℝ}

-- Condition 1: The card triples in value, resulting in 3x
-- Condition 2: Mark makes a profit of 200
def initial_amount (x : ℝ) : Prop := (3 * x - x = 200)

-- Theorem: Prove that the initial amount x equals 100 given the conditions
theorem find_initial_amount (h : initial_amount x) : x = 100 := by
  sorry

end find_initial_amount_l1739_173922


namespace tangent_line_inv_g_at_0_l1739_173973

noncomputable def g (x : ℝ) := Real.log x

theorem tangent_line_inv_g_at_0 
  (h₁ : ∀ x, g x = Real.log x) 
  (h₂ : ∀ x, x > 0): 
  ∃ m b, (∀ x y, y = g⁻¹ x → y - m * x = b) ∧ 
         (m = 1) ∧ 
         (b = 1) ∧ 
         (∀ x y, x - y + 1 = 0) := 
by
  sorry

end tangent_line_inv_g_at_0_l1739_173973


namespace empty_boxes_count_l1739_173985

-- Definitions based on conditions:
def large_box_contains (B : Type) : ℕ := 1
def initial_small_boxes (B : Type) : ℕ := 10
def non_empty_boxes (B : Type) : ℕ := 6
def additional_smaller_boxes_in_non_empty (B : Type) (b : B) : ℕ := 10
def non_empty_small_boxes := 5

-- Proving that the number of empty boxes is 55 given the conditions:
theorem empty_boxes_count (B : Type) : 
  large_box_contains B = 1 ∧
  initial_small_boxes B = 10 ∧
  non_empty_boxes B = 6 ∧
  (∃ b : B, additional_smaller_boxes_in_non_empty B b = 10) →
  (initial_small_boxes B - non_empty_small_boxes + non_empty_small_boxes * additional_smaller_boxes_in_non_empty B) = 55 :=
by 
  sorry

end empty_boxes_count_l1739_173985


namespace smallest_x_absolute_value_l1739_173971

theorem smallest_x_absolute_value : ∃ x : ℤ, |x + 3| = 15 ∧ ∀ y : ℤ, |y + 3| = 15 → x ≤ y :=
sorry

end smallest_x_absolute_value_l1739_173971


namespace paper_pieces_l1739_173991

theorem paper_pieces (n : ℕ) (h1 : 20 = 2 * n - 8) : n^2 + 20 = 216 := 
by
  sorry

end paper_pieces_l1739_173991


namespace student_average_marks_l1739_173981

theorem student_average_marks 
(P C M : ℕ) 
(h1 : (P + M) / 2 = 90) 
(h2 : (P + C) / 2 = 70) 
(h3 : P = 65) : 
  (P + C + M) / 3 = 85 :=
  sorry

end student_average_marks_l1739_173981


namespace calc_difference_l1739_173976

theorem calc_difference :
  let a := (7/12 : ℚ) * 450
  let b := (3/5 : ℚ) * 320
  let c := (5/9 : ℚ) * 540
  let d := b + c
  d - a = 229.5 := by
  -- declare the variables and provide their values
  sorry

end calc_difference_l1739_173976


namespace candy_cases_total_l1739_173920

theorem candy_cases_total
  (choco_cases lolli_cases : ℕ)
  (h1 : choco_cases = 25)
  (h2 : lolli_cases = 55) : 
  (choco_cases + lolli_cases) = 80 := by
-- The proof is omitted as requested.
sorry

end candy_cases_total_l1739_173920


namespace f_at_one_f_decreasing_f_min_on_interval_l1739_173906

noncomputable def f : ℝ → ℝ := sorry

-- Conditions
axiom f_defined : ∀ x, 0 < x → ∃ y, f y = y
axiom f_eq : ∀ x1 x2, 0 < x1 → 0 < x2 → f (x1 / x2) = f x1 - f x2
axiom f_neg : ∀ x, 1 < x → f x < 0

-- Proof statements
theorem f_at_one : f 1 = 0 := sorry

theorem f_decreasing : ∀ x1 x2, 0 < x1 → 0 < x2 → x1 < x2 → f x1 > f x2 := sorry

axiom f_at_three : f 3 = -1

theorem f_min_on_interval : ∀ x, 2 ≤ x ∧ x ≤ 9 → f x ≥ -2 := sorry

end f_at_one_f_decreasing_f_min_on_interval_l1739_173906


namespace basketball_cards_per_box_l1739_173900

-- Given conditions
def num_basketball_boxes : ℕ := 9
def num_football_boxes := num_basketball_boxes - 3
def cards_per_football_box : ℕ := 20
def total_cards : ℕ := 255
def total_football_cards := num_football_boxes * cards_per_football_box

-- We want to prove that the number of cards in each basketball card box is 15
theorem basketball_cards_per_box :
  (total_cards - total_football_cards) / num_basketball_boxes = 15 := by
  sorry

end basketball_cards_per_box_l1739_173900


namespace ratio_of_men_to_women_l1739_173931

-- Define conditions
def avg_height_students := 180
def avg_height_female := 170
def avg_height_male := 185

-- This is the math proof problem statement
theorem ratio_of_men_to_women (M W : ℕ) (h1 : (M * avg_height_male + W * avg_height_female) = (M + W) * avg_height_students) : 
  M / W = 2 :=
sorry

end ratio_of_men_to_women_l1739_173931


namespace original_number_l1739_173943

theorem original_number (x : ℝ) (h : 1.40 * x = 1680) : x = 1200 :=
by {
  sorry -- We will skip the actual proof steps here.
}

end original_number_l1739_173943


namespace sally_sours_total_l1739_173989

theorem sally_sours_total (cherry_sours lemon_sours orange_sours total_sours : ℕ) 
    (h1 : cherry_sours = 32)
    (h2 : 5 * cherry_sours = 4 * lemon_sours)
    (h3 : orange_sours = total_sours / 4)
    (h4 : cherry_sours + lemon_sours + orange_sours = total_sours) : 
    total_sours = 96 :=
by
  rw [h1] at h2
  have h5 : lemon_sours = 40 := by linarith
  rw [h1, h5] at h4
  have h6 : orange_sours = total_sours / 4 := by assumption
  rw [h6] at h4
  have h7 : 72 + total_sours / 4 = total_sours := by linarith
  sorry

end sally_sours_total_l1739_173989


namespace fraction_problem_l1739_173927

theorem fraction_problem : 
  (  (1/4 - 1/5) / (1/3 - 1/4)  ) = 3/5 :=
by
  sorry

end fraction_problem_l1739_173927


namespace sum_of_distances_l1739_173917

theorem sum_of_distances (A B C D M P : ℝ × ℝ) 
    (hA : A = (0, 0))
    (hB : B = (4, 0))
    (hC : C = (4, 4))
    (hD : D = (0, 4))
    (hM : M = (2, 0))
    (hP : P = (0, 2)) :
    dist A M + dist A P = 4 :=
by
  sorry

end sum_of_distances_l1739_173917


namespace pythagorean_theorem_l1739_173959

-- Definitions from the conditions
variables {a b c : ℝ}
-- Assuming a right triangle with legs a, b and hypotenuse c
def is_right_triangle (a b c : ℝ) : Prop :=
  c^2 = a^2 + b^2

-- Statement of the theorem:
theorem pythagorean_theorem (a b c : ℝ) (h : is_right_triangle a b c) : c^2 = a^2 + b^2 :=
sorry

end pythagorean_theorem_l1739_173959


namespace trigonometric_value_existence_l1739_173984

noncomputable def can_be_value_of_tan (n : ℝ) : Prop :=
∃ θ : ℝ, Real.tan θ = n

noncomputable def can_be_value_of_cot (n : ℝ) : Prop :=
∃ θ : ℝ, 1 / Real.tan θ = n

def can_be_value_of_sin (n : ℝ) : Prop :=
|n| ≤ 1 ∧ ∃ θ : ℝ, Real.sin θ = n

def can_be_value_of_cos (n : ℝ) : Prop :=
|n| ≤ 1 ∧ ∃ θ : ℝ, Real.cos θ = n

def can_be_value_of_sec (n : ℝ) : Prop :=
|n| ≥ 1 ∧ ∃ θ : ℝ, 1 / Real.cos θ = n

def can_be_value_of_csc (n : ℝ) : Prop :=
|n| ≥ 1 ∧ ∃ θ : ℝ, 1 / Real.sin θ = n

theorem trigonometric_value_existence (n : ℝ) : 
  can_be_value_of_tan n ∧ 
  can_be_value_of_cot n ∧ 
  can_be_value_of_sin n ∧ 
  can_be_value_of_cos n ∧ 
  can_be_value_of_sec n ∧ 
  can_be_value_of_csc n := 
sorry

end trigonometric_value_existence_l1739_173984


namespace greatest_value_of_b_l1739_173963

noncomputable def solution : ℝ :=
  (3 + Real.sqrt 21) / 2

theorem greatest_value_of_b :
  ∀ b : ℝ, b^2 - 4 * b + 3 < -b + 6 → b ≤ solution :=
by
  intro b
  intro h
  sorry

end greatest_value_of_b_l1739_173963


namespace orange_pyramid_total_l1739_173916

theorem orange_pyramid_total :
  let base_length := 7
  let base_width := 9
  -- layer 1 -> dimensions (7, 9)
  -- layer 2 -> dimensions (6, 8)
  -- layer 3 -> dimensions (5, 6)
  -- layer 4 -> dimensions (4, 5)
  -- layer 5 -> dimensions (3, 3)
  -- layer 6 -> dimensions (2, 2)
  -- layer 7 -> dimensions (1, 1)
  (base_length * base_width) + ((base_length - 1) * (base_width - 1))
  + ((base_length - 2) * (base_width - 3)) + ((base_length - 3) * (base_width - 4))
  + ((base_length - 4) * (base_width - 6)) + ((base_length - 5) * (base_width - 7))
  + ((base_length - 6) * (base_width - 8)) = 175 := sorry

end orange_pyramid_total_l1739_173916


namespace solve_problem_l1739_173902

open Real

noncomputable def problem_statement : ℝ :=
  2 * log (sqrt 2) + (log 5 / log 2) * log 2

theorem solve_problem : problem_statement = 1 := by
  sorry

end solve_problem_l1739_173902


namespace fractional_expression_evaluation_l1739_173962

theorem fractional_expression_evaluation
  (m n r t : ℚ)
  (h1 : m / n = 4 / 3)
  (h2 : r / t = 9 / 14) :
  (3 * m * r - n * t) / (4 * n * t - 7 * m * r) = -11 / 14 := by
  sorry

end fractional_expression_evaluation_l1739_173962


namespace feeding_times_per_day_l1739_173915

theorem feeding_times_per_day (p f d : ℕ) (h₁ : p = 7) (h₂ : f = 105) (h₃ : d = 5) : 
  (f / d) / p = 3 := by
  sorry

end feeding_times_per_day_l1739_173915


namespace Xia_shared_stickers_l1739_173977

def stickers_shared (initial remaining sheets_per_sheet : ℕ) : ℕ :=
  initial - (remaining * sheets_per_sheet)

theorem Xia_shared_stickers :
  stickers_shared 150 5 10 = 100 :=
by
  sorry

end Xia_shared_stickers_l1739_173977


namespace arithmetic_sequence_a3_l1739_173913

theorem arithmetic_sequence_a3 (a : ℕ → ℝ) (h : a 2 + a 4 = 8) (h_seq : a 2 + a 4 = 2 * a 3) :
  a 3 = 4 :=
by
  sorry

end arithmetic_sequence_a3_l1739_173913


namespace option_b_is_same_type_l1739_173932

def polynomial_same_type (p1 p2 : ℕ → ℕ → ℕ) : Prop :=
  ∀ x y, (p1 x y = 1 → p2 x y = 1) ∧ (p2 x y = 1 → p1 x y = 1)

def ab_squared (a b : ℕ) := if a = 1 ∧ b = 2 then 1 else 0

def a_squared_b (a b : ℕ) := if a = 2 ∧ b = 1 then 1 else 0
def negative_two_ab_squared (a b : ℕ) := if a = 1 ∧ b = 2 then 1 else 0
def ab (a b : ℕ) := if a = 1 ∧ b = 1 then 1 else 0
def ab_squared_c (a b c : ℕ) := if a = 1 ∧ b = 2 ∧ c = 1 then 1 else 0

theorem option_b_is_same_type : polynomial_same_type ab_squared negative_two_ab_squared :=
by
  sorry

end option_b_is_same_type_l1739_173932


namespace max_varphi_l1739_173979

noncomputable def f (x φ : ℝ) : ℝ := 2 * Real.sin (2 * x + φ)
noncomputable def g (x φ : ℝ) : ℝ := 2 * Real.sin (2 * x + φ + (2 * Real.pi / 3))

theorem max_varphi (φ : ℝ) (h : φ < 0) (hE : ∀ x, g x φ = g (-x) φ) : φ = -Real.pi / 6 :=
by
  sorry

end max_varphi_l1739_173979


namespace percent_increase_perimeter_third_triangle_l1739_173995

noncomputable def side_length_first : ℝ := 4
noncomputable def side_length_second : ℝ := 2 * side_length_first
noncomputable def side_length_third : ℝ := 2 * side_length_second

noncomputable def perimeter (s : ℝ) : ℝ := 3 * s

noncomputable def percent_increase (initial_perimeter final_perimeter : ℝ) : ℝ := 
  ((final_perimeter - initial_perimeter) / initial_perimeter) * 100

theorem percent_increase_perimeter_third_triangle :
  percent_increase (perimeter side_length_first) (perimeter side_length_third) = 300 := 
sorry

end percent_increase_perimeter_third_triangle_l1739_173995


namespace hypotenuse_length_l1739_173924

-- Define the conditions
def right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

-- State the theorem using the conditions and correct answer
theorem hypotenuse_length : right_triangle 20 21 29 :=
by
  -- To be filled in by proof steps
  sorry

end hypotenuse_length_l1739_173924


namespace trapezium_height_l1739_173935

theorem trapezium_height (a b A h : ℝ) (ha : a = 12) (hb : b = 16) (ha_area : A = 196) :
  (A = 0.5 * (a + b) * h) → h = 14 :=
by
  intros h_eq
  rw [ha, hb, ha_area] at h_eq
  sorry

end trapezium_height_l1739_173935


namespace parabola_focus_l1739_173996

theorem parabola_focus (a : ℝ) (p : ℝ) (x y : ℝ) :
  a = -3 ∧ p = 6 →
  (y^2 = -2 * p * x) → 
  (y^2 = -12 * x) := 
by sorry

end parabola_focus_l1739_173996


namespace jasmine_laps_l1739_173921

theorem jasmine_laps (x : ℕ) :
  (∀ (x : ℕ), ∃ (y : ℕ), y = 60 * x) :=
by
  sorry

end jasmine_laps_l1739_173921


namespace find_discounts_l1739_173982

variables (a b c : ℝ)
variables (x y z : ℝ)

theorem find_discounts (h1 : 1.1 * a - x * a = 0.99 * a)
                       (h2 : 1.12 * b - y * b = 0.99 * b)
                       (h3 : 1.15 * c - z * c = 0.99 * c) : 
x = 0.11 ∧ y = 0.13 ∧ z = 0.16 := 
sorry

end find_discounts_l1739_173982


namespace grandma_can_give_cherry_exists_better_grand_strategy_l1739_173937

variable (Packet1 : Finset String) (Packet2 : Finset String) (Packet3 : Finset String)
variable (isCabbage : String → Prop) (isCherry : String → Prop)
variable (wholePie : String → Prop)

-- Conditions
axiom Packet1_cond : ∀ p ∈ Packet1, isCabbage p
axiom Packet2_cond : ∀ p ∈ Packet2, isCherry p
axiom Packet3_cond_cabbage : ∃ p ∈ Packet3, isCabbage p
axiom Packet3_cond_cherry : ∃ p ∈ Packet3, isCherry p

-- Question (a)
theorem grandma_can_give_cherry (h1 : ∃ p1 ∈ Packet3, wholePie p1 ∧ isCherry p1 ∨
    ∃ p2 ∈ Packet1, wholePie p2 ∧ (∃ q ∈ Packet2 ∪ Packet3, isCherry q ∧ wholePie q) ∨
    ∃ p3 ∈ Packet2, wholePie p3 ∧ isCherry p3) :
  ∃ grand_strategy, grand_strategy = (2 / 3) * (1 : ℝ) :=
by
  sorry

-- Question (b)
theorem exists_better_grand_strategy (h2 : ∃ p ∈ Packet3, wholePie p ∧ isCherry p ∨
    ∃ p2 ∈ Packet1, wholePie p2 ∧ (∃ q ∈ Packet2 ∪ Packet3, isCherry q ∧ wholePie q) ∨
    ∃ p3 ∈ Packet2, wholePie p3 ∧ isCherry p3) :
  ∃ grand_strategy, grand_strategy > (2 / 3) * (1 : ℝ) :=
by
  sorry

end grandma_can_give_cherry_exists_better_grand_strategy_l1739_173937


namespace number_of_truthful_dwarfs_l1739_173944

def total_dwarfs := 10
def hands_raised_vanilla := 10
def hands_raised_chocolate := 5
def hands_raised_fruit := 1
def total_hands_raised := hands_raised_vanilla + hands_raised_chocolate + hands_raised_fruit
def extra_hands := total_hands_raised - total_dwarfs
def liars := extra_hands
def truthful := total_dwarfs - liars

theorem number_of_truthful_dwarfs : truthful = 4 :=
by sorry

end number_of_truthful_dwarfs_l1739_173944


namespace mila_calculator_sum_l1739_173907

theorem mila_calculator_sum :
  let n := 60
  let calc1_start := 2
  let calc2_start := 0
  let calc3_start := -1
  calc1_start^(3^n) + calc2_start^2^(n) + (-calc3_start)^n = 2^(3^60) + 1 :=
by {
  sorry
}

end mila_calculator_sum_l1739_173907


namespace geometric_sequence_characterization_l1739_173939

theorem geometric_sequence_characterization (a : ℕ → ℝ) :
  (∀ n, a (n + 1) / a n = (a (n + 2) - a (n + 1)) / (a (n + 1) - a n)) →
  (∃ r : ℝ, ∀ n, a (n + 1) = r * a n) :=
by
  sorry

end geometric_sequence_characterization_l1739_173939


namespace Susan_roses_ratio_l1739_173967

theorem Susan_roses_ratio (total_roses given_roses vase_roses remaining_roses : ℕ) 
  (H1 : total_roses = 3 * 12)
  (H2 : vase_roses = total_roses - given_roses)
  (H3 : remaining_roses = vase_roses * 2 / 3)
  (H4 : remaining_roses = 12) :
  given_roses / gcd given_roses total_roses = 1 ∧ total_roses / gcd given_roses total_roses = 2 :=
by
  sorry

end Susan_roses_ratio_l1739_173967


namespace arithmetic_difference_l1739_173969

variables (p q r : ℝ)

theorem arithmetic_difference (h1 : (p + q) / 2 = 10) (h2 : (q + r) / 2 = 27) : r - p = 34 :=
by
  sorry

end arithmetic_difference_l1739_173969


namespace irrational_sum_root_l1739_173904

theorem irrational_sum_root
  (α : ℝ) (hα : Irrational α)
  (n : ℕ) (hn : 0 < n) :
  Irrational ((α + (α^2 - 1).sqrt)^(1/n : ℝ) + (α - (α^2 - 1).sqrt)^(1/n : ℝ)) := sorry

end irrational_sum_root_l1739_173904


namespace correct_statements_l1739_173928

-- Given the values of x and y on the parabola
def parabola (a b c : ℝ) (x : ℝ) := a * x^2 + b * x + c

-- Define the points on the parabola
def points_on_parabola (a b c : ℝ) : Prop :=
  parabola a b c (-1) = 3 ∧
  parabola a b c 0 = 0 ∧
  parabola a b c 1 = -1 ∧
  parabola a b c 2 = 0 ∧
  parabola a b c 3 = 3

-- Prove the correct statements
theorem correct_statements (a b c : ℝ) (h : points_on_parabola a b c) : 
  ¬(∃ x, parabola a b c x < 0 ∧ x < 0) ∧
  parabola a b c 2 = 0 :=
by 
  sorry

end correct_statements_l1739_173928


namespace ratio_fraction_l1739_173970

theorem ratio_fraction (A B C : ℕ) (h1 : 7 * B = 3 * A) (h2 : 6 * C = 5 * B) :
  (C : ℚ) / (A : ℚ) = 5 / 14 ∧ (A : ℚ) / (C : ℚ) = 14 / 5 :=
by
  sorry

end ratio_fraction_l1739_173970


namespace smallest_gcd_value_l1739_173975

theorem smallest_gcd_value (m n : ℕ) (h1 : m > 0) (h2 : n > 0) (h3 : Nat.gcd m n = 8) : Nat.gcd (8 * m) (12 * n) = 32 :=
by
  sorry

end smallest_gcd_value_l1739_173975


namespace exists_equal_sum_pairs_l1739_173998

theorem exists_equal_sum_pairs (n : ℕ) (hn : n > 2009) :
  ∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  a ≤ 2009 ∧ b ≤ 2009 ∧ c ≤ 2009 ∧ d ≤ 2009 ∧
  (1 / a + 1 / b : ℝ) = 1 / c + 1 / d :=
sorry

end exists_equal_sum_pairs_l1739_173998


namespace find_divisor_l1739_173925

-- Definitions from the conditions
def remainder : ℤ := 8
def quotient : ℤ := 43
def dividend : ℤ := 997
def is_prime (n : ℤ) : Prop := n ≠ 1 ∧ (∀ d : ℤ, d ∣ n → d = 1 ∨ d = n)

-- The proof problem statement
theorem find_divisor (d : ℤ) 
  (hd : is_prime d) 
  (hdiv : dividend = (d * quotient) + remainder) : 
  d = 23 := 
sorry

end find_divisor_l1739_173925


namespace largest_unrepresentable_n_l1739_173948

theorem largest_unrepresentable_n (a b : ℕ) (ha : 1 < a) (hb : 1 < b) : ∃ n, ¬ ∃ x y : ℕ, n = 7 * a + 5 * b ∧ n = 47 :=
  sorry

end largest_unrepresentable_n_l1739_173948


namespace unknown_number_l1739_173947

theorem unknown_number (x : ℝ) (h : 7^8 - 6/x + 9^3 + 3 + 12 = 95) : x = 1 / 960908.333 :=
sorry

end unknown_number_l1739_173947


namespace find_angle_CDB_l1739_173946

variables (A B C D E : Type)
variables [LinearOrderedField A] [LinearOrderedField B] [LinearOrderedField C] [LinearOrderedField D] [LinearOrderedField E]

noncomputable def angle := ℝ -- Define type for angles

variables (AB AD AC ACB ACD : angle)
variables (BAD BEA CDB : ℝ)

-- Define the given angles and conditions in Lean
axiom AB_eq_AD : AB = AD
axiom angle_ACD_eq_angle_ACB : AC = ACD
axiom angle_BAD_eq_140 : BAD = 140
axiom angle_BEA_eq_110 : BEA = 110

theorem find_angle_CDB (AB_eq_AD : AB = AD)
                       (angle_ACD_eq_angle_ACB : AC = ACD)
                       (angle_BAD_eq_140 : BAD = 140)
                       (angle_BEA_eq_110 : BEA = 110) :
                       CDB = 50 :=
by
  sorry

end find_angle_CDB_l1739_173946


namespace ratio_of_years_taught_l1739_173955

-- Definitions based on given conditions
def C : ℕ := 4
def A : ℕ := 2 * C
def total_years (S : ℕ) : Prop := C + A + S = 52

-- Proof statement
theorem ratio_of_years_taught (S : ℕ) (h : total_years S) : 
  S / A = 5 / 1 :=
by
  sorry

end ratio_of_years_taught_l1739_173955


namespace better_fit_model_l1739_173950

-- Define the residual sums of squares
def RSS_1 : ℝ := 152.6
def RSS_2 : ℝ := 159.8

-- Define the statement that the model with RSS_1 is the better fit
theorem better_fit_model : RSS_1 < RSS_2 → RSS_1 = 152.6 :=
by
  sorry

end better_fit_model_l1739_173950


namespace arithmetic_sequence_common_difference_l1739_173960

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℤ) (h₁ : a 2 = 9) (h₂ : a 5 = 33) :
  ∀ d : ℤ, (∀ n : ℕ, a n = a 1 + (n - 1) * d) → d = 8 :=
by
  -- We state the theorem and provide a "sorry" proof placeholder
  sorry

end arithmetic_sequence_common_difference_l1739_173960


namespace box_office_scientific_notation_l1739_173910

def billion : ℝ := 10^9
def box_office_revenue : ℝ := 57.44 * billion
def scientific_notation (n : ℝ) : ℝ × ℝ := (5.744, 10^10)

theorem box_office_scientific_notation :
  scientific_notation box_office_revenue = (5.744, 10^10) :=
by
  sorry

end box_office_scientific_notation_l1739_173910


namespace calculate_expression_l1739_173903

theorem calculate_expression : 5 * 401 + 4 * 401 + 3 * 401 + 400 = 5212 := by
  sorry

end calculate_expression_l1739_173903


namespace sin_3theta_over_sin_theta_l1739_173936

theorem sin_3theta_over_sin_theta (θ : ℝ) (h : Real.tan θ = Real.sqrt 2) : 
  Real.sin (3 * θ) / Real.sin θ = 1 / 3 :=
by
  sorry

end sin_3theta_over_sin_theta_l1739_173936


namespace intersection_A_B_l1739_173908

open Set

def A : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.log (x - 1)}
def B : Set ℝ := {x : ℝ | x^2 + 2 * x - 3 ≥ 0}

theorem intersection_A_B : A ∩ B = {x : ℝ | 1 < x} :=
by
  sorry

end intersection_A_B_l1739_173908
