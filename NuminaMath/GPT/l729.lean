import Mathlib

namespace remainder_3_mod_6_l729_72911

theorem remainder_3_mod_6 (n : ℕ) (h : n % 18 = 3) : n % 6 = 3 :=
by
    sorry

end remainder_3_mod_6_l729_72911


namespace distance_from_center_l729_72983

-- Define the circle equation as a predicate
def isCircle (x y : ℝ) : Prop :=
  x^2 + y^2 = 2 * x - 4 * y + 8

-- Define the center of the circle
def circleCenter : ℝ × ℝ := (1, -2)

-- Define the point in question
def point : ℝ × ℝ := (-3, 4)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the proof problem
theorem distance_from_center :
  ∀ (x y : ℝ), isCircle x y → distance circleCenter point = 2 * Real.sqrt 13 :=
by
  sorry

end distance_from_center_l729_72983


namespace attendance_changes_l729_72977

theorem attendance_changes :
  let m := 25  -- Monday attendance
  let t := 31  -- Tuesday attendance
  let w := 20  -- initial Wednesday attendance
  let th := 28  -- Thursday attendance
  let f := 22  -- Friday attendance
  let sa := 26  -- Saturday attendance
  let w_new := 30  -- corrected Wednesday attendance
  let initial_total := m + t + w + th + f + sa
  let new_total := m + t + w_new + th + f + sa
  let initial_mean := initial_total / 6
  let new_mean := new_total / 6
  let mean_increase := new_mean - initial_mean
  let initial_median := (25 + 26) / 2  -- median of [20, 22, 25, 26, 28, 31]
  let new_median := (26 + 28) / 2  -- median of [22, 25, 26, 28, 30, 31]
  let median_increase := new_median - initial_median
  mean_increase = 1.667 ∧ median_increase = 1.5 := by
sorry

end attendance_changes_l729_72977


namespace least_number_subtracted_l729_72996

theorem least_number_subtracted (n : ℕ) (x : ℕ) (h_pos : 0 < x) (h_init : n = 427398) (h_div : ∃ k : ℕ, (n - x) = 14 * k) : x = 6 :=
sorry

end least_number_subtracted_l729_72996


namespace smallest_positive_real_l729_72925

theorem smallest_positive_real (x : ℝ) (h₁ : ∃ y : ℝ, y > 0 ∧ ⌊y^2⌋ - y * ⌊y⌋ = 4) : x = 29 / 5 :=
by
  sorry

end smallest_positive_real_l729_72925


namespace remainder_two_when_divided_by_3_l729_72981

-- Define the main theorem stating that for any positive integer n,
-- n^3 + 3/2 * n^2 + 1/2 * n - 1 leaves a remainder of 2 when divided by 3.

theorem remainder_two_when_divided_by_3 (n : ℕ) (h : n > 0) : 
  (n^3 + (3 / 2) * n^2 + (1 / 2) * n - 1) % 3 = 2 := 
sorry

end remainder_two_when_divided_by_3_l729_72981


namespace inequality_l729_72956

-- Define the real variables p, q, r and the condition that their product is 1
variables {p q r : ℝ} (h : p * q * r = 1)

-- State the theorem
theorem inequality (h : p * q * r = 1) :
  (1 / (1 - p))^2 + (1 / (1 - q))^2 + (1 / (1 - r))^2 ≥ 1 := 
sorry

end inequality_l729_72956


namespace max_students_distributed_equally_l729_72940

theorem max_students_distributed_equally (pens pencils : ℕ) (h1 : pens = 3528) (h2 : pencils = 3920) : 
  Nat.gcd pens pencils = 392 := 
by 
  sorry

end max_students_distributed_equally_l729_72940


namespace sandbox_width_l729_72953

theorem sandbox_width (P : ℕ) (W L : ℕ) (h1 : P = 30) (h2 : L = 2 * W) (h3 : P = 2 * L + 2 * W) : W = 5 := 
sorry

end sandbox_width_l729_72953


namespace hallie_read_pages_third_day_more_than_second_day_l729_72922

theorem hallie_read_pages_third_day_more_than_second_day :
  ∀ (d1 d2 d3 d4 : ℕ),
  d1 = 63 →
  d2 = 2 * d1 →
  d4 = 29 →
  d1 + d2 + d3 + d4 = 354 →
  (d3 - d2) = 10 :=
by
  intros d1 d2 d3 d4 h1 h2 h4 h_sum
  sorry

end hallie_read_pages_third_day_more_than_second_day_l729_72922


namespace find_D_l729_72965

theorem find_D (A B D : ℕ) (h1 : (100 * A + 10 * B + D) * (A + B + D) = 1323) (h2 : A ≥ B) : D = 1 :=
sorry

end find_D_l729_72965


namespace car_speed_l729_72917

-- Define the given conditions
def distance := 800 -- in kilometers
def time := 5 -- in hours

-- Define the speed calculation
def speed (d : ℕ) (t : ℕ) : ℕ := d / t

-- State the theorem to be proved
theorem car_speed : speed distance time = 160 := by
  -- proof would go here
  sorry

end car_speed_l729_72917


namespace average_speed_is_20_mph_l729_72920

-- Defining the conditions
def distance1 := 40 -- miles
def speed1 := 20 -- miles per hour
def distance2 := 20 -- miles
def speed2 := 40 -- miles per hour
def distance3 := 30 -- miles
def speed3 := 15 -- miles per hour

-- Calculating total distance and total time
def total_distance := distance1 + distance2 + distance3
def time1 := distance1 / speed1 -- hours
def time2 := distance2 / speed2 -- hours
def time3 := distance3 / speed3 -- hours
def total_time := time1 + time2 + time3

-- Theorem statement
theorem average_speed_is_20_mph : (total_distance / total_time) = 20 := by
  sorry

end average_speed_is_20_mph_l729_72920


namespace angle_C_measure_l729_72993

theorem angle_C_measure
  (D C : ℝ)
  (h1 : C + D = 90)
  (h2 : C = 3 * D) :
  C = 67.5 :=
by
  sorry

end angle_C_measure_l729_72993


namespace find_length_l729_72934

-- Define the perimeter and breadth as constants
def P : ℕ := 950
def B : ℕ := 100

-- State the theorem
theorem find_length (L : ℕ) (H : 2 * (L + B) = P) : L = 375 :=
by sorry

end find_length_l729_72934


namespace even_goals_more_likely_l729_72950

theorem even_goals_more_likely (p₁ : ℝ) (q₁ : ℝ) 
  (h₁ : q₁ = 1 - p₁)
  (independent_halves : (p₁ * p₁ + q₁ * q₁) > (2 * p₁ * q₁)) :
  (p₁ * p₁ + q₁ * q₁) > (1 - (p₁ * p₁ + q₁ * q₁)) :=
by
  sorry

end even_goals_more_likely_l729_72950


namespace expected_non_allergic_l729_72990

theorem expected_non_allergic (p : ℝ) (n : ℕ) (h : p = 1 / 4) (hn : n = 300) : n * p = 75 :=
by sorry

end expected_non_allergic_l729_72990


namespace sum_c_d_eq_30_l729_72988

noncomputable def c_d_sum : ℕ :=
  let c : ℕ := 28
  let d : ℕ := 2
  c + d

theorem sum_c_d_eq_30 : c_d_sum = 30 :=
by {
  sorry
}

end sum_c_d_eq_30_l729_72988


namespace remainder_of_70_div_17_l729_72998

theorem remainder_of_70_div_17 : 70 % 17 = 2 :=
by
  sorry

end remainder_of_70_div_17_l729_72998


namespace same_terminal_side_angle_exists_l729_72967

theorem same_terminal_side_angle_exists :
  ∃ k : ℤ, -5 * π / 8 + 2 * k * π = 11 * π / 8 := 
by
  sorry

end same_terminal_side_angle_exists_l729_72967


namespace smallest_positive_integer_divisible_by_14_15_18_l729_72995

theorem smallest_positive_integer_divisible_by_14_15_18 : 
  ∃ n : ℕ, n > 0 ∧ (14 ∣ n) ∧ (15 ∣ n) ∧ (18 ∣ n) ∧ n = 630 :=
sorry

end smallest_positive_integer_divisible_by_14_15_18_l729_72995


namespace polynomial_divisibility_n_l729_72963

theorem polynomial_divisibility_n :
  ∀ (n : ℤ), (∀ x, x = 2 → 3 * x^2 - 4 * x + n = 0) → n = -4 :=
by
  intros n h
  have h2 : 3 * 2^2 - 4 * 2 + n = 0 := h 2 rfl
  linarith

end polynomial_divisibility_n_l729_72963


namespace oomyapeck_eyes_count_l729_72958

-- Define the various conditions
def number_of_people : ℕ := 3
def fish_per_person : ℕ := 4
def eyes_per_fish : ℕ := 2
def eyes_given_to_dog : ℕ := 2

-- Compute the total number of fish
def total_fish : ℕ := number_of_people * fish_per_person

-- Compute the total number of eyes from the total number of fish
def total_eyes : ℕ := total_fish * eyes_per_fish

-- Compute the number of eyes Oomyapeck eats
def eyes_eaten_by_oomyapeck : ℕ := total_eyes - eyes_given_to_dog

-- The proof statement
theorem oomyapeck_eyes_count : eyes_eaten_by_oomyapeck = 22 := by
  sorry

end oomyapeck_eyes_count_l729_72958


namespace hyperbola_properties_l729_72985

def hyperbola (x y : ℝ) : Prop := x^2 - 4 * y^2 = 1

theorem hyperbola_properties :
  (∀ x y : ℝ, hyperbola x y → (x + 2 * y = 0 ∨ x - 2 * y = 0)) ∧
  (2 * (1 / 2) = 1) := 
by
  sorry

end hyperbola_properties_l729_72985


namespace range_of_m_l729_72979

theorem range_of_m {x : ℝ} (m : ℝ) :
  (∀ x, |x - 1| + |x - 2| + |x - 3| ≥ m) ↔ m ≤ 2 :=
by
  sorry

end range_of_m_l729_72979


namespace multiply_203_197_square_neg_699_l729_72976

theorem multiply_203_197 : 203 * 197 = 39991 := by
  sorry

theorem square_neg_699 : (-69.9)^2 = 4886.01 := by
  sorry

end multiply_203_197_square_neg_699_l729_72976


namespace calculate_width_of_vessel_base_l729_72900

noncomputable def cube_edge : ℝ := 17
noncomputable def base_length : ℝ := 20
noncomputable def water_rise : ℝ := 16.376666666666665
noncomputable def cube_volume : ℝ := cube_edge ^ 3
noncomputable def base_area (W : ℝ) : ℝ := base_length * W
noncomputable def displaced_volume (W : ℝ) : ℝ := base_area W * water_rise

theorem calculate_width_of_vessel_base :
  ∃ W : ℝ, displaced_volume W = cube_volume ∧ W = 15 := by
  sorry

end calculate_width_of_vessel_base_l729_72900


namespace functional_equation_zero_l729_72931

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equation_zero (hx : ∀ x y : ℝ, f (x + y) = f x + f y) : f 0 = 0 :=
by
  sorry

end functional_equation_zero_l729_72931


namespace pump_without_leak_time_l729_72975

variables (P : ℝ) (effective_rate_with_leak : ℝ) (leak_rate : ℝ)
variable (pump_filling_time : ℝ)

-- Define the conditions
def conditions :=
  effective_rate_with_leak = 3/7 ∧
  leak_rate = 1/14 ∧
  pump_filling_time = P

-- Define the theorem
theorem pump_without_leak_time (h : conditions P effective_rate_with_leak leak_rate pump_filling_time) : 
  P = 2 :=
sorry

end pump_without_leak_time_l729_72975


namespace time_for_train_to_pass_jogger_l729_72933

noncomputable def time_to_pass (s_jogger s_train : ℝ) (d_headstart l_train : ℝ) : ℝ :=
  let speed_jogger := s_jogger * (1000 / 3600)
  let speed_train := s_train * (1000 / 3600)
  let relative_speed := speed_train - speed_jogger
  let total_distance := d_headstart + l_train
  total_distance / relative_speed

theorem time_for_train_to_pass_jogger :
  time_to_pass 12 60 360 180 = 40.48 :=
by
  sorry

end time_for_train_to_pass_jogger_l729_72933


namespace negation_of_proposition_l729_72924

theorem negation_of_proposition (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 - a * x + 1 < 0) ↔ (∀ x : ℝ, x^2 - a * x + 1 ≥ 0) :=
by sorry

end negation_of_proposition_l729_72924


namespace sum_a1_to_a5_l729_72932

noncomputable def f (x : ℝ) : ℝ := (1 + x) + (1 + x)^2 + (1 + x)^3 + (1 + x)^4 + (1 + x)^5
noncomputable def g (x : ℝ) (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ) : ℝ := a_0 + a_1 * (1 - x) + a_2 * (1 - x)^2 + a_3 * (1 - x)^3 + a_4 * (1 - x)^4 + a_5 * (1 - x)^5

theorem sum_a1_to_a5 (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ) :
  (∀ x : ℝ, f x = g x a_0 a_1 a_2 a_3 a_4 a_5) →
  (f 1 = g 1 a_0 a_1 a_2 a_3 a_4 a_5) →
  (f 0 = g 0 a_0 a_1 a_2 a_3 a_4 a_5) →
  a_0 = 62 →
  a_0 + a_1 + a_2 + a_3 + a_4 + a_5 = 5 →
  a_1 + a_2 + a_3 + a_4 + a_5 = -57 :=
by
  intro hf1 hf2 hf3 ha0 hsum
  sorry

end sum_a1_to_a5_l729_72932


namespace fred_dimes_l729_72982

theorem fred_dimes (initial_dimes borrowed_dimes : ℕ) (h1 : initial_dimes = 7) (h2 : borrowed_dimes = 3) :
  initial_dimes - borrowed_dimes = 4 :=
by
  sorry

end fred_dimes_l729_72982


namespace find_dimensions_l729_72944

-- Define the conditions
def perimeter (x y : ℕ) : Prop := (2 * (x + y) = 3996)
def divisible_parts (x y k : ℕ) : Prop := (x * y = 1998 * k) ∧ ∃ (k : ℕ), (k * 1998 = x * y) ∧ k ≠ 0

-- State the theorem
theorem find_dimensions (x y : ℕ) (k : ℕ) : perimeter x y ∧ divisible_parts x y k → (x = 1332 ∧ y = 666) ∨ (x = 666 ∧ y = 1332) :=
by
  -- This is where the proof would go.
  sorry

end find_dimensions_l729_72944


namespace range_of_H_l729_72941

def H (x : ℝ) : ℝ := 2 * |2 * x + 2| - 3 * |2 * x - 2|

theorem range_of_H : Set.range H = Set.Ici 8 := 
by 
  sorry

end range_of_H_l729_72941


namespace binomial_sum_l729_72984

theorem binomial_sum (n k : ℕ) (h : n = 10) (hk : k = 3) :
  Nat.choose n k + Nat.choose n (n - k) = 240 :=
by
  -- placeholder for actual proof
  sorry

end binomial_sum_l729_72984


namespace system_of_equations_l729_72906

-- Given conditions: Total number of fruits and total cost of the fruits purchased
def total_fruits := 1000
def total_cost := 999
def cost_of_sweet_fruit := (11 : ℚ) / 9
def cost_of_bitter_fruit := (4 : ℚ) / 7

-- Variables representing the number of sweet and bitter fruits
variables (x y : ℚ)

-- Problem statement in Lean 4
theorem system_of_equations :
  (x + y = total_fruits) ∧ (cost_of_sweet_fruit * x + cost_of_bitter_fruit * y = total_cost) ↔
  ((x + y = 1000) ∧ (11 / 9 * x + 4 / 7 * y = 999)) :=
by
  sorry

end system_of_equations_l729_72906


namespace remainder_by_19_l729_72999

theorem remainder_by_19 (N : ℤ) (k : ℤ) (h : N = 779 * k + 47) : N % 19 = 9 :=
by sorry

end remainder_by_19_l729_72999


namespace profit_percentage_is_12_36_l729_72904

noncomputable def calc_profit_percentage (SP CP : ℝ) : ℝ :=
  let Profit := SP - CP
  (Profit / CP) * 100

theorem profit_percentage_is_12_36
  (SP : ℝ) (h1 : SP = 100)
  (CP : ℝ) (h2 : CP = 0.89 * SP) :
  calc_profit_percentage SP CP = 12.36 :=
by
  sorry

end profit_percentage_is_12_36_l729_72904


namespace select_test_point_l729_72909

theorem select_test_point (x1 x2 : ℝ) (h1 : x1 = 2 + 0.618 * (4 - 2)) (h2 : x2 = 2 + 4 - x1) :
  (x1 > x2 → x3 = 4 - 0.618 * (4 - x1)) ∨ (x1 < x2 → x3 = 6 - x3) :=
  sorry

end select_test_point_l729_72909


namespace gnomes_and_ponies_l729_72951

theorem gnomes_and_ponies (g p : ℕ) (h1 : g + p = 15) (h2 : 2 * g + 4 * p = 36) : g = 12 ∧ p = 3 :=
by
  sorry

end gnomes_and_ponies_l729_72951


namespace bukvinsk_acquaintances_l729_72989

theorem bukvinsk_acquaintances (Martin Klim Inna Tamara Kamilla : Type) 
  (acquaints : Type → Type → Prop)
  (exists_same_letters : ∀ (x y : Type), acquaints x y ↔ ∃ S, (x = S ∧ y = S)) :
  (∃ (count_Martin : ℕ), count_Martin = 20) →
  (∃ (count_Klim : ℕ), count_Klim = 15) →
  (∃ (count_Inna : ℕ), count_Inna = 12) →
  (∃ (count_Tamara : ℕ), count_Tamara = 12) →
  (∃ (count_Kamilla : ℕ), count_Kamilla = 15) := by
  sorry

end bukvinsk_acquaintances_l729_72989


namespace odd_function_extended_l729_72974

noncomputable def f (x : ℝ) : ℝ := 
  if h : x ≥ 0 then 
    x * Real.log (x + 1)
  else 
    x * Real.log (-x + 1)

theorem odd_function_extended : (∀ x : ℝ, f (-x) = -f x) →
  (∀ x : ℝ, x ≥ 0 → f x = x * Real.log (x + 1)) →
  (∀ x : ℝ, x < 0 → f x = x * Real.log (-x + 1)) :=
by
  intros h_odd h_def_neg
  sorry

end odd_function_extended_l729_72974


namespace intersection_points_l729_72907

def line1 (x y : ℝ) : Prop := 2 * y - 3 * x = 4
def line2 (x y : ℝ) : Prop := x + 3 * y = 6
def line3 (x y : ℝ) : Prop := 6 * x - 9 * y = 12

theorem intersection_points :
  ∃ (x y : ℝ), line1 x y ∧ line2 x y ∧ ¬(x = x ∧ y = y) → 0 = 1 :=
sorry

end intersection_points_l729_72907


namespace restaurant_sales_decrease_l729_72949

-- Conditions
variable (Sales_August : ℝ := 42000)
variable (Sales_October : ℝ := 27000)
variable (a : ℝ) -- monthly average decrease rate as a decimal

-- Theorem statement
theorem restaurant_sales_decrease :
  42 * (1 - a)^2 = 27 := sorry

end restaurant_sales_decrease_l729_72949


namespace jessica_blueberry_pies_l729_72921

theorem jessica_blueberry_pies 
  (total_pies : ℕ)
  (ratio_apple : ℕ)
  (ratio_blueberry : ℕ)
  (ratio_cherry : ℕ)
  (h_total : total_pies = 36)
  (h_ratios : ratio_apple = 2)
  (h_ratios_b : ratio_blueberry = 5)
  (h_ratios_c : ratio_cherry = 3) : 
  total_pies * ratio_blueberry / (ratio_apple + ratio_blueberry + ratio_cherry) = 18 := 
by
  sorry

end jessica_blueberry_pies_l729_72921


namespace g_half_l729_72968

noncomputable def g : ℝ → ℝ := sorry

axiom g0 : g 0 = 0
axiom g1 : g 1 = 1
axiom g_non_decreasing : ∀ {x y : ℝ}, 0 ≤ x → x < y → y ≤ 1 → g x ≤ g y
axiom g_symmetry : ∀ {x : ℝ}, 0 ≤ x → x ≤ 1 → g (1 - x) = 1 - g x
axiom g_fraction : ∀ {x : ℝ}, 0 ≤ x → x ≤ 1 → g (x / 4) = g x / 3

theorem g_half : g (1 / 2) = 1 / 2 := sorry

end g_half_l729_72968


namespace sum_of_two_numbers_l729_72928

theorem sum_of_two_numbers (a b : ℕ) (h1 : a - b = 10) (h2 : a = 22) : a + b = 34 :=
sorry

end sum_of_two_numbers_l729_72928


namespace finding_value_of_expression_l729_72937

open Real

theorem finding_value_of_expression
  (a b : ℝ)
  (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_eq : 1/a - 1/b - 1/(a + b) = 0) :
  (b/a + a/b)^2 = 5 :=
sorry

end finding_value_of_expression_l729_72937


namespace hydrochloric_acid_solution_l729_72913

variable (V : ℝ) (pure_acid_added : ℝ) (initial_concentration : ℝ) (final_concentration : ℝ)

theorem hydrochloric_acid_solution :
  initial_concentration = 0.10 → 
  final_concentration = 0.15 → 
  pure_acid_added = 3.52941176471 → 
  0.10 * V + 3.52941176471 = 0.15 * (V + 3.52941176471) → 
  V = 60 :=
by
  intros h_initial h_final h_pure h_equation
  sorry

end hydrochloric_acid_solution_l729_72913


namespace find_cost_price_per_item_min_items_type_A_l729_72939

-- Definitions based on the conditions
def cost_A (x : ℝ) (y : ℝ) : Prop := 4 * x + 10 = 5 * y
def cost_B (x : ℝ) (y : ℝ) : Prop := 20 * x + 10 * y = 160

-- Proving the cost price per item of goods A and B
theorem find_cost_price_per_item : ∃ x y : ℝ, cost_A x y ∧ cost_B x y ∧ x = 5 ∧ y = 6 :=
by
  -- This is where the proof would go
  sorry

-- Additional conditions for part (2)
def profit_condition (a : ℕ) : Prop :=
  10 * (a - 30) + 8 * (200 - (a - 30)) - 5 * a - 6 * (200 - a) ≥ 640

-- Proving the minimum number of items of type A purchased
theorem min_items_type_A : ∃ a : ℕ, profit_condition a ∧ a ≥ 100 :=
by
  -- This is where the proof would go
  sorry

end find_cost_price_per_item_min_items_type_A_l729_72939


namespace solve_quadratic_problem_l729_72987

theorem solve_quadratic_problem :
  ∀ x : ℝ, (x^2 + 6 * x + 8 = -(x + 4) * (x + 7)) ↔ (x = -4 ∨ x = -4.5) := by
  sorry

end solve_quadratic_problem_l729_72987


namespace tiffany_final_lives_l729_72955

def initial_lives : ℕ := 43
def lost_lives : ℕ := 14
def gained_lives : ℕ := 27

theorem tiffany_final_lives : (initial_lives - lost_lives + gained_lives) = 56 := by
    sorry

end tiffany_final_lives_l729_72955


namespace option_d_correct_l729_72966

theorem option_d_correct (a b : ℝ) : 2 * a^2 * b - 4 * a^2 * b = -2 * a^2 * b :=
by
  sorry

end option_d_correct_l729_72966


namespace inscribed_square_area_l729_72926

-- Define the conditions and the problem
theorem inscribed_square_area
  (side_length : ℝ)
  (square_area : ℝ) :
  side_length = 24 →
  square_area = 576 :=
by
  sorry

end inscribed_square_area_l729_72926


namespace scientific_notation_of_220_billion_l729_72929

theorem scientific_notation_of_220_billion :
  220000000000 = 2.2 * 10^11 :=
by
  sorry

end scientific_notation_of_220_billion_l729_72929


namespace find_n_l729_72991

theorem find_n (n : ℕ) : 2^(2 * n) + 2^(2 * n) + 2^(2 * n) + 2^(2 * n) = 4^22 → n = 21 :=
by
  sorry

end find_n_l729_72991


namespace problem1_l729_72938

variable (m : ℤ)

theorem problem1 : m * (m - 3) + 3 * (3 - m) = (m - 3) ^ 2 := by
  sorry

end problem1_l729_72938


namespace walter_chore_days_l729_72908

-- Definitions for the conditions
variables (b w : ℕ)  -- b: days regular, w: days exceptionally well

-- Conditions
def days_eq : Prop := b + w = 15
def earnings_eq : Prop := 3 * b + 4 * w = 47

-- The theorem stating the proof problem
theorem walter_chore_days (hb : days_eq b w) (he : earnings_eq b w) : w = 2 :=
by
  -- We only need to state the theorem; the proof is omitted.
  sorry

end walter_chore_days_l729_72908


namespace sin_2_alpha_plus_pi_by_3_l729_72930

-- Define the statement to be proved
theorem sin_2_alpha_plus_pi_by_3 (α : ℝ) (hα : 0 < α ∧ α < π / 2)
  (hcos : Real.cos (α + π / 6) = 4 / 5) :
  Real.sin (2 * α + π / 3) = 24 / 25 := sorry

end sin_2_alpha_plus_pi_by_3_l729_72930


namespace initial_percentage_of_alcohol_l729_72945

variable (P : ℝ)
variables (x y : ℝ) (initial_percent replacement_percent replaced_quantity final_percent : ℝ)

def whisky_problem :=
  initial_percent = P ∧
  replacement_percent = 0.19 ∧
  replaced_quantity = 2/3 ∧
  final_percent = 0.26 ∧
  (P * (1 - replaced_quantity) + replacement_percent * replaced_quantity = final_percent)

theorem initial_percentage_of_alcohol :
  whisky_problem P 0.40 0.19 (2/3) 0.26 := sorry

end initial_percentage_of_alcohol_l729_72945


namespace bowl_capacity_percentage_l729_72954

theorem bowl_capacity_percentage
    (initial_half_full : ℕ)
    (added_water : ℕ)
    (total_water : ℕ)
    (full_capacity : ℕ)
    (percentage_filled : ℚ) :
    initial_half_full * 2 = full_capacity →
    initial_half_full + added_water = total_water →
    added_water = 4 →
    total_water = 14 →
    percentage_filled = (total_water * 100) / full_capacity →
    percentage_filled = 70 := 
by
    intros h1 h2 h3 h4 h5
    sorry

end bowl_capacity_percentage_l729_72954


namespace range_of_x_range_of_a_l729_72902

variable {x a : ℝ}

def p (x a : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def q (x : ℝ) : Prop := (x - 3) / (x - 2) < 0

theorem range_of_x (h1 : a = 1) (h2 : p x 1 ∧ q x) : 2 < x ∧ x < 3 := by
  sorry

theorem range_of_a (h : ∀ x, p x a → q x) : 1 ≤ a ∧ a ≤ 2 := by
  sorry

end range_of_x_range_of_a_l729_72902


namespace ratio_of_x_to_y_l729_72997

theorem ratio_of_x_to_y (x y : ℝ) (h : (3 * x - 2 * y) / (2 * x + y) = 5 / 4) : x / y = 13 / 2 :=
sorry

end ratio_of_x_to_y_l729_72997


namespace number_of_classmates_l729_72936

theorem number_of_classmates (total_apples : ℕ) (apples_per_classmate : ℕ) (people_in_class : ℕ) 
  (h1 : total_apples = 15) (h2 : apples_per_classmate = 5) (h3 : people_in_class = total_apples / apples_per_classmate) : 
  people_in_class = 3 :=
by sorry

end number_of_classmates_l729_72936


namespace ratio_of_expenditure_l729_72969

variable (A B AE BE : ℕ)

theorem ratio_of_expenditure (h1 : A = 2000) 
    (h2 : A / B = 5 / 4) 
    (h3 : A - AE = 800) 
    (h4: B - BE = 800) :
    AE / BE = 3 / 2 := by
  sorry

end ratio_of_expenditure_l729_72969


namespace smallest_five_digit_palindrome_div_4_thm_l729_72978

def is_palindrome (n : ℕ) : Prop :=
  n = (n % 10) * 10000 + ((n / 10) % 10) * 1000 + ((n / 100) % 10) * 100 + ((n / 1000) % 10) * 10 + (n / 10000)

def smallest_five_digit_palindrome_div_4 : ℕ :=
  18881

theorem smallest_five_digit_palindrome_div_4_thm :
  is_palindrome smallest_five_digit_palindrome_div_4 ∧
  10000 ≤ smallest_five_digit_palindrome_div_4 ∧
  smallest_five_digit_palindrome_div_4 < 100000 ∧
  smallest_five_digit_palindrome_div_4 % 4 = 0 ∧
  ∀ n, is_palindrome n ∧ 10000 ≤ n ∧ n < 100000 ∧ n % 4 = 0 → n ≥ smallest_five_digit_palindrome_div_4 :=
by
  sorry

end smallest_five_digit_palindrome_div_4_thm_l729_72978


namespace divisibility_by_91_l729_72962

theorem divisibility_by_91 (n : ℕ) : ∃ k : ℤ, 9^(n + 2) + 10^(2 * n + 1) = 91 * k := by
  sorry

end divisibility_by_91_l729_72962


namespace sin_alpha_two_alpha_plus_beta_l729_72914

variable {α β : ℝ}
variable (h₁ : 0 < α ∧ α < π / 2)
variable (h₂ : 0 < β ∧ β < π / 2)
variable (h₃ : Real.tan (α / 2) = 1 / 3)
variable (h₄ : Real.cos (α - β) = -4 / 5)

theorem sin_alpha (h₁ : 0 < α ∧ α < π / 2)
                  (h₃ : Real.tan (α / 2) = 1 / 3) :
                  Real.sin α = 3 / 5 :=
by
  sorry

theorem two_alpha_plus_beta (h₁ : 0 < α ∧ α < π / 2)
                            (h₂ : 0 < β ∧ β < π / 2)
                            (h₄ : Real.cos (α - β) = -4 / 5) :
                            2 * α + β = π :=
by
  sorry

end sin_alpha_two_alpha_plus_beta_l729_72914


namespace range_of_a_minus_b_l729_72973

theorem range_of_a_minus_b {a b : ℝ} (h₁ : -2 < a) (h₂ : a < 1) (h₃ : 0 < b) (h₄ : b < 4) : -6 < a - b ∧ a - b < 1 :=
by
  sorry -- The proof is skipped as per the instructions.

end range_of_a_minus_b_l729_72973


namespace product_of_two_numbers_l729_72952

theorem product_of_two_numbers (x y : ℝ) (h1 : x + y = 24) (h2 : x^2 + y^2 = 400) : x * y = 88 :=
by
  -- Proof goes here
  sorry

end product_of_two_numbers_l729_72952


namespace find_percentage_l729_72918

-- conditions
def N : ℕ := 160
def expected_percentage : ℕ := 35

-- statement to prove
theorem find_percentage (P : ℕ) (h : P / 100 * N = 50 / 100 * N - 24) : P = expected_percentage :=
sorry

end find_percentage_l729_72918


namespace sum_square_geq_one_third_l729_72948

variable (a b c : ℝ)

theorem sum_square_geq_one_third (h : a + b + c = 1) : 
  a^2 + b^2 + c^2 ≥ 1 / 3 := 
sorry

end sum_square_geq_one_third_l729_72948


namespace students_taking_neither_l729_72943

theorem students_taking_neither (total_students : ℕ)
    (students_math : ℕ) (students_physics : ℕ) (students_chemistry : ℕ)
    (students_math_physics : ℕ) (students_physics_chemistry : ℕ) (students_math_chemistry : ℕ)
    (students_all_three : ℕ) :
    total_students = 60 →
    students_math = 40 →
    students_physics = 30 →
    students_chemistry = 25 →
    students_math_physics = 18 →
    students_physics_chemistry = 10 →
    students_math_chemistry = 12 →
    students_all_three = 5 →
    (total_students - (students_math + students_physics + students_chemistry - students_math_physics - students_physics_chemistry - students_math_chemistry + students_all_three)) = 5 :=
by
  intros
  sorry

end students_taking_neither_l729_72943


namespace P_desert_but_not_Coffee_is_0_15_l729_72994

-- Define the relevant probabilities as constants
def P_desert_and_coffee := 0.60
def P_not_desert := 0.2500000000000001
def P_desert := 1 - P_not_desert
def P_desert_but_not_coffee := P_desert - P_desert_and_coffee

-- The theorem to prove that the probability of ordering dessert but not coffee is 0.15
theorem P_desert_but_not_Coffee_is_0_15 :
  P_desert_but_not_coffee = 0.15 :=
by 
  -- calculation steps can be filled in here eventually
  sorry

end P_desert_but_not_Coffee_is_0_15_l729_72994


namespace extreme_points_range_of_a_l729_72916

noncomputable def f (x a : ℝ) : ℝ := (x - 1) * Real.exp x - a * x^2

-- Problem 1: Extreme points
theorem extreme_points (a : ℝ) : 
  (a ≤ 0 → ∃! x, ∀ y, f y a ≤ f x a) ∧
  (0 < a ∧ a < 1/2 → ∃ x1 x2, x1 ≠ x2 ∧ ∀ y, f y a ≤ f x1 a ∨ f y a ≤ f x2 a) ∧
  (a = 1/2 → ∀ x y, f y a ≤ f x a → x = y) ∧
  (a > 1/2 → ∃ x1 x2, x1 ≠ x2 ∧ ∀ y, f y a ≤ f x1 a ∨ f y a ≤ f x2 a) :=
sorry

-- Problem 2: Range of values for 'a'
theorem range_of_a (a : ℝ) : 
  (∀ x, f x a + Real.exp x ≥ x^3 + x) ↔ (a ≤ Real.exp 1 - 2) :=
sorry

end extreme_points_range_of_a_l729_72916


namespace ninth_graders_science_only_l729_72947

theorem ninth_graders_science_only 
    (total_students : ℕ := 120)
    (science_students : ℕ := 80)
    (programming_students : ℕ := 75) 
    : (science_students - (science_students + programming_students - total_students)) = 45 :=
by
  sorry

end ninth_graders_science_only_l729_72947


namespace solution_set_of_inequality_l729_72957

variable (a b c : ℝ)

theorem solution_set_of_inequality 
  (h1 : a < 0)
  (h2 : b = a)
  (h3 : c = -2 * a)
  (h4 : ∀ x : ℝ, -2 < x ∧ x < 1 → ax^2 + bx + c > 0) :
  ∀ x : ℝ, (x ≤ -1 / 2 ∨ x ≥ 1) ↔ cx^2 + ax + b ≥ 0 :=
sorry

end solution_set_of_inequality_l729_72957


namespace evaluate_expression_l729_72927

theorem evaluate_expression :
  (3 + 1) * (3^3 + 1^3) * (3^9 + 1^9) = 2878848 :=
by
  sorry

end evaluate_expression_l729_72927


namespace john_brown_bags_l729_72935

theorem john_brown_bags :
  (∃ b : ℕ, 
     let total_macaroons := 12
     let weight_per_macaroon := 5
     let total_weight := total_macaroons * weight_per_macaroon
     let remaining_weight := 45
     let bag_weight := total_weight - remaining_weight
     let macaroons_per_bag := bag_weight / weight_per_macaroon
     total_macaroons / macaroons_per_bag = b
  ) → b = 4 :=
by
  sorry

end john_brown_bags_l729_72935


namespace prob_return_to_freezer_l729_72912

-- Define the probabilities of picking two pops of each flavor
def probability_same_flavor (total: ℕ) (pop1: ℕ) (pop2: ℕ) : ℚ :=
  (pop1 * pop2) / (total * (total - 1))

-- Definitions according to the problem conditions
def cherry_pops : ℕ := 4
def orange_pops : ℕ := 3
def lemon_lime_pops : ℕ := 4
def total_pops : ℕ := cherry_pops + orange_pops + lemon_lime_pops

-- Calculate the probability of picking two ice pops of the same flavor
def prob_cherry : ℚ := probability_same_flavor total_pops cherry_pops (cherry_pops - 1)
def prob_orange : ℚ := probability_same_flavor total_pops orange_pops (orange_pops - 1)
def prob_lemon_lime : ℚ := probability_same_flavor total_pops lemon_lime_pops (lemon_lime_pops - 1)

def prob_same_flavor : ℚ := prob_cherry + prob_orange + prob_lemon_lime
def prob_diff_flavor : ℚ := 1 - prob_same_flavor

-- Theorem stating the probability of needing to return to the freezer
theorem prob_return_to_freezer : prob_diff_flavor = 8 / 11 := by
  sorry

end prob_return_to_freezer_l729_72912


namespace triangle_area_l729_72910

theorem triangle_area :
  let A := (-3, 0)
  let B := (0, 2)
  let O := (0, 0)
  let area := 1 / 2 * |A.1 * (B.2 - O.2) + B.1 * (O.2 - A.2) + O.1 * (A.2 - B.2)|
  area = 3 := by
  let A := (-3, 0)
  let B := (0, 2)
  let O := (0, 0)
  let area := 1 / 2 * |A.1 * (B.2 - O.2) + B.1 * (O.2 - A.2) + O.1 * (A.2 - B.2)|
  sorry

end triangle_area_l729_72910


namespace A_empty_iff_A_single_element_iff_and_value_A_at_most_one_element_iff_l729_72972

-- Define the set A
def A (a : ℝ) : Set ℝ := {x | a * x^2 - 3 * x + 2 = 0}

-- Statement for (1)
theorem A_empty_iff (a : ℝ) : A a = ∅ ↔ a ∈ Set.Ioi 0 :=
sorry

-- Statement for (2)
theorem A_single_element_iff_and_value (a : ℝ) : 
  (∃ x, A a = {x}) ↔ (a = 0 ∨ a = 9 / 8) ∧ A a = {2 / 3} :=
sorry

-- Statement for (3)
theorem A_at_most_one_element_iff (a : ℝ) : 
  (∃ x, A a = {x} ∨ A a = ∅) ↔ (a = 0 ∨ a ∈ Set.Ici (9 / 8)) :=
sorry

end A_empty_iff_A_single_element_iff_and_value_A_at_most_one_element_iff_l729_72972


namespace patio_rows_before_rearrangement_l729_72923

theorem patio_rows_before_rearrangement (r c : ℕ) 
  (h1 : r * c = 160) 
  (h2 : (r + 4) * (c - 2) = 160)
  (h3 : ∃ k : ℕ, 5 * k = r)
  (h4 : ∃ l : ℕ, 5 * l = c) :
  r = 16 :=
by
  sorry

end patio_rows_before_rearrangement_l729_72923


namespace expected_worth_coin_flip_l729_72986

noncomputable def expected_worth : ℝ := 
  (1 / 3) * 6 + (2 / 3) * (-2) - 1

theorem expected_worth_coin_flip : expected_worth = -0.33 := 
by 
  unfold expected_worth
  norm_num
  sorry

end expected_worth_coin_flip_l729_72986


namespace factorization_problem_l729_72960

theorem factorization_problem (a b c : ℤ)
  (h1 : ∀ x : ℝ, x^2 + 7 * x + 12 = (x + a) * (x + b))
  (h2 : ∀ x : ℝ, x^2 - 8 * x - 20 = (x - b) * (x - c)) :
  a - b + c = -9 :=
sorry

end factorization_problem_l729_72960


namespace hexagons_after_cuts_l729_72942

theorem hexagons_after_cuts (rectangles_initial : ℕ) (cuts : ℕ) (sheets_total : ℕ)
  (initial_sides : ℕ) (additional_sides : ℕ) 
  (triangle_sides : ℕ) (hexagon_sides : ℕ) 
  (final_sides : ℕ) (number_of_hexagons : ℕ) :
  rectangles_initial = 15 →
  cuts = 60 →
  sheets_total = rectangles_initial + cuts →
  initial_sides = rectangles_initial * 4 →
  additional_sides = cuts * 4 →
  final_sides = initial_sides + additional_sides →
  triangle_sides = 3 →
  hexagon_sides = 6 →
  (sheets_total * 4 = final_sides) →
  number_of_hexagons = (final_sides - 225) / 3 →
  number_of_hexagons = 25 :=
by
  intros
  sorry

end hexagons_after_cuts_l729_72942


namespace minimum_weight_of_grass_seed_l729_72961

-- Definitions of cost and weights
def price_5_pound_bag : ℝ := 13.85
def price_10_pound_bag : ℝ := 20.43
def price_25_pound_bag : ℝ := 32.20
def max_weight : ℝ := 80
def min_cost : ℝ := 98.68

-- Lean proposition to prove the minimum weight given the conditions
theorem minimum_weight_of_grass_seed (w : ℝ) :
  w = 75 ↔ (w ≤ max_weight ∧
            ∃ (n5 n10 n25 : ℕ), 
              w = 5 * n5 + 10 * n10 + 25 * n25 ∧
              min_cost ≤ n5 * price_5_pound_bag + n10 * price_10_pound_bag + n25 * price_25_pound_bag ∧
              n5 * price_5_pound_bag + n10 * price_10_pound_bag + n25 * price_25_pound_bag ≤ min_cost) := 
by
  sorry

end minimum_weight_of_grass_seed_l729_72961


namespace exists_digit_combination_l729_72903

theorem exists_digit_combination (d1 d2 d3 d4 : ℕ) (H1 : 42 * (d1 * 10 + 8) = 2 * 1000 + d2 * 100 + d3 * 10 + d4) (H2: ∃ n: ℕ, n = 2 + d2 + d3 + d4 ∧ n % 2 = 1):
  d1 = 4 ∧ 42 * 48 = 2016 ∨ d1 = 6 ∧ 42 * 68 = 2856 :=
sorry

end exists_digit_combination_l729_72903


namespace dilation_image_l729_72915

theorem dilation_image (z : ℂ) (c : ℂ) (k : ℝ) (w : ℂ) (h₁ : c = 0 + 5 * I) 
  (h₂ : k = 3) (h₃ : w = 3 + 2 * I) : z = 9 - 4 * I :=
by
  -- Given conditions
  have hc : c = 0 + 5 * I := h₁
  have hk : k = 3 := h₂
  have hw : w = 3 + 2 * I := h₃

  -- Dilation formula
  let formula := (w - c) * k + c

  -- Prove the result
  -- sorry for now, the proof is not required as per instructions
  sorry

end dilation_image_l729_72915


namespace sin_alpha_second_quadrant_l729_72970

theorem sin_alpha_second_quadrant (α : ℝ) (h_α_quad_2 : π / 2 < α ∧ α < π) (h_cos_α : Real.cos α = -1 / 3) : Real.sin α = 2 * Real.sqrt 2 / 3 := 
sorry

end sin_alpha_second_quadrant_l729_72970


namespace tadpoles_more_than_fish_l729_72964

def fish_initial : ℕ := 100
def tadpoles_initial := 4 * fish_initial
def snails_initial : ℕ := 150
def fish_caught : ℕ := 12
def tadpoles_to_frogs := (2 * tadpoles_initial) / 3
def snails_crawled_away : ℕ := 20

theorem tadpoles_more_than_fish :
  let fish_now : ℕ := fish_initial - fish_caught
  let tadpoles_now : ℕ := tadpoles_initial - tadpoles_to_frogs
  fish_now < tadpoles_now ∧ tadpoles_now - fish_now = 46 :=
by
  sorry

end tadpoles_more_than_fish_l729_72964


namespace Sarah_total_weeds_l729_72992

noncomputable def Tuesday_weeds : ℕ := 25
noncomputable def Wednesday_weeds : ℕ := 3 * Tuesday_weeds
noncomputable def Thursday_weeds : ℕ := (1 / 5) * Tuesday_weeds
noncomputable def Friday_weeds : ℕ := (3 / 4) * Tuesday_weeds - 10

noncomputable def Total_weeds : ℕ := Tuesday_weeds + Wednesday_weeds + Thursday_weeds + Friday_weeds

theorem Sarah_total_weeds : Total_weeds = 113 := by
  sorry

end Sarah_total_weeds_l729_72992


namespace wins_per_girl_l729_72980

theorem wins_per_girl (a b c d : ℕ) (h1 : a + b = 8) (h2 : a + c = 10) (h3 : b + c = 12) (h4 : a + d = 12) (h5 : b + d = 14) (h6 : c + d = 16) : 
  a = 3 ∧ b = 5 ∧ c = 7 ∧ d = 9 :=
sorry

end wins_per_girl_l729_72980


namespace steve_marbles_l729_72971

-- Define the initial condition variables
variables (S Steve_initial Sam_initial Sally_initial Sarah_initial Steve_now : ℕ)

-- Conditions
def cond1 : Sam_initial = 2 * Steve_initial := by sorry
def cond2 : Sally_initial = Sam_initial - 5 := by sorry
def cond3 : Sarah_initial = Steve_initial + 3 := by sorry
def cond4 : Steve_now = Steve_initial + 3 := by sorry
def cond5 : Sam_initial - (3 + 3 + 4) = 6 := by sorry

-- Goal
theorem steve_marbles : Steve_now = 11 := by sorry

end steve_marbles_l729_72971


namespace exists_acute_triangle_side_lengths_l729_72901

-- Define the real numbers d_1, d_2, ..., d_12 in the interval (1, 12).
noncomputable def real_numbers_in_interval (d : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 12 → 1 < d n ∧ d n < 12

-- Define the condition for d_i, d_j, d_k to form an acute triangle
def forms_acuse_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2

-- The main theorem statement
theorem exists_acute_triangle_side_lengths (d : ℕ → ℝ) (h : real_numbers_in_interval d) :
  ∃ i j k, i ≠ j ∧ j ≠ k ∧ k ≠ i ∧ forms_acuse_triangle (d i) (d j) (d k) :=
sorry

end exists_acute_triangle_side_lengths_l729_72901


namespace solution_to_system_of_equations_l729_72959

theorem solution_to_system_of_equations :
  ∃ x y : ℤ, 4 * x - 3 * y = 11 ∧ 2 * x + y = 13 ∧ x = 5 ∧ y = 3 :=
by
  sorry

end solution_to_system_of_equations_l729_72959


namespace x_equals_1_over_16_l729_72905

-- Given conditions
def distance_center_to_tangents_intersection : ℚ := 3 / 8
def radius_of_circle : ℚ := 3 / 16
def distance_center_to_CD : ℚ := 1 / 2

-- Calculated total distance
def total_distance_center_to_C : ℚ := distance_center_to_tangents_intersection + radius_of_circle

-- Problem statement
theorem x_equals_1_over_16 (x : ℚ) 
    (h : total_distance_center_to_C = x + distance_center_to_CD) : 
    x = 1 / 16 := 
by
  -- Proof is omitted, based on the provided solution steps
  sorry

end x_equals_1_over_16_l729_72905


namespace solution_interval_l729_72919

theorem solution_interval (X₀ : ℝ) (h₀ : Real.log (X₀ + 1) = 2 / X₀) : 1 < X₀ ∧ X₀ < 2 :=
by
  admit -- to be proved

end solution_interval_l729_72919


namespace front_view_correct_l729_72946

section stack_problem

def column1 : List ℕ := [3, 2]
def column2 : List ℕ := [1, 4, 2]
def column3 : List ℕ := [5]
def column4 : List ℕ := [2, 1]

def tallest (l : List ℕ) : ℕ := l.foldr max 0

theorem front_view_correct :
  [tallest column1, tallest column2, tallest column3, tallest column4] = [3, 4, 5, 2] :=
sorry

end stack_problem

end front_view_correct_l729_72946
