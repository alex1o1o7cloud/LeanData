import Mathlib

namespace ellipse_sum_a_k_l2199_219918

theorem ellipse_sum_a_k {a b h k : ℝ}
  (foci1 foci2 : ℝ × ℝ)
  (point_on_ellipse : ℝ × ℝ)
  (h_center : h = (foci1.1 + foci2.1) / 2)
  (k_center : k = (foci1.2 + foci2.2) / 2)
  (distance1 : ℝ := Real.sqrt ((point_on_ellipse.1 - foci1.1)^2 + (point_on_ellipse.2 - foci1.2)^2))
  (distance2 : ℝ := Real.sqrt ((point_on_ellipse.1 - foci2.1)^2 + (point_on_ellipse.2 - foci2.2)^2))
  (major_axis_length : ℝ := distance1 + distance2)
  (h_a : a = major_axis_length / 2)
  (c := Real.sqrt ((foci2.1 - foci1.1)^2 + (foci2.2 - foci1.2)^2) / 2)
  (h_b : b^2 = a^2 - c^2) :
  a + k = (7 + Real.sqrt 13) / 2 := 
by
  sorry

end ellipse_sum_a_k_l2199_219918


namespace total_amount_paid_l2199_219986

-- Define the conditions of the problem
def cost_without_discount (quantity : ℕ) (unit_price : ℚ) : ℚ :=
  quantity * unit_price

def cost_with_discount (quantity : ℕ) (unit_price : ℚ) (discount_rate : ℚ) : ℚ :=
  let total_cost := cost_without_discount quantity unit_price
  total_cost - (total_cost * discount_rate)

-- Define each category's cost after discount
def pens_cost : ℚ := cost_with_discount 7 1.5 0.10
def notebooks_cost : ℚ := cost_without_discount 4 5
def water_bottles_cost : ℚ := cost_with_discount 2 8 0.30
def backpack_cost : ℚ := cost_with_discount 1 25 0.15
def socks_cost : ℚ := cost_with_discount 3 3 0.25

-- Prove the total amount paid is $68.65
theorem total_amount_paid : pens_cost + notebooks_cost + water_bottles_cost + backpack_cost + socks_cost = 68.65 := by
  sorry

end total_amount_paid_l2199_219986


namespace back_seat_can_hold_8_people_l2199_219999

def totalPeopleOnSides : ℕ :=
  let left_seats := 15
  let right_seats := left_seats - 3
  let people_per_seat := 3
  (left_seats + right_seats) * people_per_seat

def bus_total_capacity : ℕ := 89

def back_seat_capacity : ℕ :=
  bus_total_capacity - totalPeopleOnSides

theorem back_seat_can_hold_8_people : back_seat_capacity = 8 := by
  sorry

end back_seat_can_hold_8_people_l2199_219999


namespace determine_a_l2199_219916

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  if h : x = 3 then a else 2 / |x - 3|

theorem determine_a (a : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ 3 ∧ x2 ≠ 3 ∧ (f x1 a - 4 = 0) ∧ (f x2 a - 4 = 0) ∧ f 3 a - 4 = 0) →
  a = 4 :=
by
  sorry

end determine_a_l2199_219916


namespace recreation_percentage_l2199_219995

variable (W : ℝ) -- John's wages last week
variable (recreation_last_week : ℝ := 0.35 * W) -- Amount spent on recreation last week
variable (wages_this_week : ℝ := 0.70 * W) -- Wages this week
variable (recreation_this_week : ℝ := 0.25 * wages_this_week) -- Amount spent on recreation this week

theorem recreation_percentage :
  (recreation_this_week / recreation_last_week) * 100 = 50 := by
  sorry

end recreation_percentage_l2199_219995


namespace max_colored_nodes_without_cycle_in_convex_polygon_l2199_219926

def convex_polygon (n : ℕ) : Prop := n ≥ 3

def valid_diagonals (n : ℕ) : Prop := n = 2019

def no_three_diagonals_intersect_at_single_point (x : Type*) : Prop :=
  sorry -- You can provide a formal definition here based on combinatorial geometry.

def no_loops (n : ℕ) (k : ℕ) : Prop :=
  k ≤ (n * (n - 3)) / 2 - 1

theorem max_colored_nodes_without_cycle_in_convex_polygon :
  convex_polygon 2019 →
  valid_diagonals 2019 →
  no_three_diagonals_intersect_at_single_point ℝ →
  ∃ k, k = 2035151 ∧ no_loops 2019 k := 
by
  -- The proof would be constructed here.
  sorry

end max_colored_nodes_without_cycle_in_convex_polygon_l2199_219926


namespace percentage_of_children_speaking_only_Hindi_l2199_219917

/-
In a class of 60 children, 30% of children can speak only English,
20% can speak both Hindi and English, and 42 children can speak Hindi.
Prove that the percentage of children who can speak only Hindi is 50%.
-/
theorem percentage_of_children_speaking_only_Hindi :
  let total_children := 60
  let english_only := 0.30 * total_children
  let both_languages := 0.20 * total_children
  let hindi_only := 42 - both_languages
  (hindi_only / total_children) * 100 = 50 :=
by
  sorry

end percentage_of_children_speaking_only_Hindi_l2199_219917


namespace proposition_2_proposition_3_l2199_219900

theorem proposition_2 (a b : ℝ) (h: a > |b|) : a^2 > b^2 := 
sorry

theorem proposition_3 (a b : ℝ) (h: a > b) : a^3 > b^3 := 
sorry

end proposition_2_proposition_3_l2199_219900


namespace find_difference_of_max_and_min_values_l2199_219932

noncomputable def v (a b : Int) : Int := a * (-4) + b

theorem find_difference_of_max_and_min_values :
  let v0 := 3
  let v1 := v v0 12
  let v2 := v v1 6
  let v3 := v v2 10
  let v4 := v v3 (-8)
  (max (max (max (max v0 v1) v2) v3) v4) - (min (min (min (min v0 v1) v2) v3) v4) = 62 :=
by
  sorry

end find_difference_of_max_and_min_values_l2199_219932


namespace arccos_sqrt_3_over_2_eq_pi_over_6_l2199_219973

open Real

theorem arccos_sqrt_3_over_2_eq_pi_over_6 :
  ∀ (x : ℝ), x = (sqrt 3) / 2 → arccos x = π / 6 :=
by
  intro x
  sorry

end arccos_sqrt_3_over_2_eq_pi_over_6_l2199_219973


namespace limit_an_to_a_l2199_219948

theorem limit_an_to_a (ε : ℝ) (hε : ε > 0) : 
  ∃ (N : ℕ), ∀ (n : ℕ), n ≥ N →
  |(9 - (n^3 : ℝ)) / (1 + 2 * (n^3 : ℝ)) + 1/2| < ε :=
sorry

end limit_an_to_a_l2199_219948


namespace solve_equation_l2199_219946

theorem solve_equation (x : ℝ) (h1: (6 * x) ^ 18 = (12 * x) ^ 9) (h2 : x ≠ 0) : x = 1 / 3 := by
  sorry

end solve_equation_l2199_219946


namespace verify_value_of_sum_l2199_219934

noncomputable def value_of_sum (a b c d e f : ℕ) (values : Finset ℕ) : ℕ :=
if h : a ∈ values ∧ b ∈ values ∧ c ∈ values ∧ d ∈ values ∧ e ∈ values ∧ f ∈ values ∧
        a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
        b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
        c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
        d ≠ e ∧ d ≠ f ∧
        e ≠ f ∧
        a + b = c ∧
        b + c = d ∧
        c + e = f
then a + c + f
else 0

theorem verify_value_of_sum :
  ∃ (a b c d e f : ℕ) (values : Finset ℕ),
  values = {4, 12, 15, 27, 31, 39} ∧
  a ∈ values ∧ b ∈ values ∧ c ∈ values ∧ d ∈ values ∧ e ∈ values ∧ f ∈ values ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
  d ≠ e ∧ d ≠ f ∧
  e ≠ f ∧
  a + b = c ∧
  b + c = d ∧
  c + e = f ∧
  value_of_sum a b c d e f values = 73 :=
by
  sorry

end verify_value_of_sum_l2199_219934


namespace number_of_ways_to_divide_day_l2199_219944

theorem number_of_ways_to_divide_day (n m : ℕ) (hn : 0 < n) (hm : 0 < m) (h : n * m = 1440) : 
  ∃ (pairs : List (ℕ × ℕ)), (pairs.length = 36) ∧
  (∀ (p : ℕ × ℕ), p ∈ pairs → (p.1 * p.2 = 1440)) :=
sorry

end number_of_ways_to_divide_day_l2199_219944


namespace divide_equal_parts_l2199_219976

theorem divide_equal_parts (m n: ℕ) (h₁: (m + n) % 2 = 0) (h₂: gcd m n ∣ ((m + n) / 2)) : ∃ a b: ℕ, a = b ∧ a + b = m + n ∧ a ≤ m + n ∧ b ≤ m + n :=
sorry

end divide_equal_parts_l2199_219976


namespace intersection_points_l2199_219957

noncomputable def curve1 (x y : ℝ) : Prop := x^2 + 4 * y^2 = 1
noncomputable def curve2 (x y : ℝ) : Prop := 4 * x^2 + y^2 = 4

theorem intersection_points : 
  ∃ (points : Finset (ℝ × ℝ)), 
  (∀ p ∈ points, curve1 p.1 p.2 ∧ curve2 p.1 p.2) ∧ points.card = 2 := 
by 
  sorry

end intersection_points_l2199_219957


namespace find_a_l2199_219979

def lambda : Set ℝ := { x | ∃ (a b : ℤ), x = a + b * Real.sqrt 3 }

theorem find_a (a : ℤ) (x : ℝ)
  (h1 : x = 7 + a * Real.sqrt 3)
  (h2 : x ∈ lambda)
  (h3 : (1 / x) ∈ lambda) :
  a = 4 ∨ a = -4 :=
sorry

end find_a_l2199_219979


namespace num_people_on_boats_l2199_219912

-- Definitions based on the conditions
def boats := 5
def people_per_boat := 3

-- Theorem stating the problem to be solved
theorem num_people_on_boats : boats * people_per_boat = 15 :=
by sorry

end num_people_on_boats_l2199_219912


namespace smallest_cube_volume_l2199_219930

noncomputable def sculpture_height : ℝ := 15
noncomputable def sculpture_base_radius : ℝ := 8
noncomputable def cube_side_length : ℝ := 16

theorem smallest_cube_volume :
  ∀ (h r s : ℝ), 
    h = sculpture_height ∧
    r = sculpture_base_radius ∧
    s = cube_side_length →
    s ^ 3 = 4096 :=
by
  intros h r s 
  intro h_def
  sorry

end smallest_cube_volume_l2199_219930


namespace discount_rate_on_pony_jeans_l2199_219983

-- Define the conditions as Lean definitions
def fox_price : ℝ := 15
def pony_price : ℝ := 18
def total_savings : ℝ := 8.91
def total_discount_rate : ℝ := 22
def number_of_fox_pairs : ℕ := 3
def number_of_pony_pairs : ℕ := 2

-- Given definitions of the discount rates on Fox and Pony jeans
variable (F P : ℝ)

-- The system of equations based on the conditions
axiom sum_of_discount_rates : F + P = total_discount_rate
axiom savings_equation : 
  number_of_fox_pairs * (fox_price * F / 100) + number_of_pony_pairs * (pony_price * P / 100) = total_savings

-- The theorem to prove
theorem discount_rate_on_pony_jeans : P = 11 := by
  sorry

end discount_rate_on_pony_jeans_l2199_219983


namespace ellipse_focal_point_l2199_219988

theorem ellipse_focal_point (m : ℝ) (m_pos : m > 0)
  (h : ∃ f : ℝ × ℝ, f = (1, 0) ∧ ∀ x y : ℝ, (x^2 / 4) + (y^2 / m^2) = 1 → 
    (x - 1)^2 + y^2 = (x^2 / 4) + (y^2 / m^2)) :
  m = Real.sqrt 3 := 
sorry

end ellipse_focal_point_l2199_219988


namespace mass_percentage_H_in_chlorous_acid_l2199_219977

noncomputable def mass_percentage_H_in_HClO2 : ℚ :=
  let molar_mass_H : ℚ := 1.01
  let molar_mass_Cl : ℚ := 35.45
  let molar_mass_O : ℚ := 16.00
  let molar_mass_HClO2 : ℚ := molar_mass_H + molar_mass_Cl + 2 * molar_mass_O
  (molar_mass_H / molar_mass_HClO2) * 100

theorem mass_percentage_H_in_chlorous_acid :
  mass_percentage_H_in_HClO2 = 1.475 := by
  sorry

end mass_percentage_H_in_chlorous_acid_l2199_219977


namespace number_of_deleted_apps_l2199_219972

def initial_apps := 16
def remaining_apps := 8

def deleted_apps : ℕ := initial_apps - remaining_apps

theorem number_of_deleted_apps : deleted_apps = 8 := 
by
  unfold deleted_apps initial_apps remaining_apps
  rfl

end number_of_deleted_apps_l2199_219972


namespace fg_value_correct_l2199_219952

def f_table (x : ℕ) : ℕ :=
  if x = 1 then 3
  else if x = 3 then 7
  else if x = 5 then 9
  else if x = 7 then 13
  else if x = 9 then 17
  else 0  -- Default value to handle unexpected inputs

def g_table (x : ℕ) : ℕ :=
  if x = 1 then 54
  else if x = 3 then 9
  else if x = 5 then 25
  else if x = 7 then 19
  else if x = 9 then 44
  else 0  -- Default value to handle unexpected inputs

theorem fg_value_correct : f_table (g_table 3) = 17 := 
by sorry

end fg_value_correct_l2199_219952


namespace marbles_solution_l2199_219981

open Nat

def marbles_problem : Prop :=
  ∃ J_k J_j : Nat, (J_k = 3) ∧ (J_k = J_j - 4) ∧ (J_k + J_j = 10)

theorem marbles_solution : marbles_problem := by
  sorry

end marbles_solution_l2199_219981


namespace point_in_third_quadrant_l2199_219922

noncomputable def is_second_quadrant (a b : ℝ) : Prop :=
a < 0 ∧ b > 0

noncomputable def is_third_quadrant (a b : ℝ) : Prop :=
a < 0 ∧ b < 0

theorem point_in_third_quadrant (a b : ℝ) (h : is_second_quadrant a b) : is_third_quadrant a (-b) :=
by
  sorry

end point_in_third_quadrant_l2199_219922


namespace domain_of_sqrt_function_l2199_219935

noncomputable def domain_of_function : Set ℝ :=
  {x : ℝ | 3 - 2 * x - x^2 ≥ 0}

theorem domain_of_sqrt_function : domain_of_function = {x : ℝ | -3 ≤ x ∧ x ≤ 1} :=
by
  sorry

end domain_of_sqrt_function_l2199_219935


namespace fred_allowance_is_16_l2199_219927

def fred_weekly_allowance (A : ℕ) : Prop :=
  (A / 2) + 6 = 14

theorem fred_allowance_is_16 : ∃ A : ℕ, fred_weekly_allowance A ∧ A = 16 := 
by
  -- Proof can be filled here
  sorry

end fred_allowance_is_16_l2199_219927


namespace remaining_pieces_total_l2199_219939

noncomputable def initial_pieces : Nat := 16
noncomputable def kennedy_lost_pieces : Nat := 4 + 1 + 2
noncomputable def riley_lost_pieces : Nat := 1 + 1 + 1

theorem remaining_pieces_total : (initial_pieces - kennedy_lost_pieces) + (initial_pieces - riley_lost_pieces) = 22 := by
  sorry

end remaining_pieces_total_l2199_219939


namespace odd_function_f_neg_one_l2199_219923

open Real

noncomputable def f (x : ℝ) : ℝ :=
  if h : 0 < x ∧ x < 2 then 2^x else 0 -- Placeholder; actual implementation skipped for simplicity

theorem odd_function_f_neg_one :
  (∀ x, f (-x) = -f x) ∧ (∀ x, (0 < x ∧ x < 2) → f x = 2^x) → 
  f (-1) = -2 :=
by
  intros h
  let odd_property := h.1
  let condition_in_range := h.2
  sorry

end odd_function_f_neg_one_l2199_219923


namespace no_n_for_equal_sums_l2199_219909

theorem no_n_for_equal_sums (n : ℕ) (h : n ≠ 0) :
  let s1 := (3 * n^2 + 7 * n) / 2
  let s2 := (3 * n^2 + 37 * n) / 2
  s1 ≠ s2 :=
by
  let s1 := (3 * n^2 + 7 * n) / 2
  let s2 := (3 * n^2 + 37 * n) / 2
  sorry

end no_n_for_equal_sums_l2199_219909


namespace width_of_rectangle_11_l2199_219941

variable (L W : ℕ)

-- The conditions: 
-- 1. The perimeter is 48cm
-- 2. Width is 2 cm shorter than length
def is_rectangle (L W : ℕ) : Prop :=
  2 * L + 2 * W = 48 ∧ W = L - 2

-- The statement we need to prove
theorem width_of_rectangle_11 (L W : ℕ) (h : is_rectangle L W) : W = 11 :=
by
  sorry

end width_of_rectangle_11_l2199_219941


namespace repeating_decimal_sum_l2199_219911

/--
The number 3.17171717... can be written as a reduced fraction x/y where x = 314 and y = 99.
We aim to prove that the sum of x and y is 413.
-/
theorem repeating_decimal_sum : 
  let x := 314
  let y := 99
  (x + y) = 413 := 
by
  sorry

end repeating_decimal_sum_l2199_219911


namespace midpoint_of_hyperbola_segment_l2199_219955

theorem midpoint_of_hyperbola_segment :
  (∃ A B : ℝ × ℝ, (A.1 ^ 2 - (A.2 ^ 2 / 9) = 1) ∧ (B.1 ^ 2 - (B.2 ^ 2 / 9) = 1)
    ∧ (A.1 + B.1) / 2 = -1 ∧ (A.2 + B.2) / 2 = -4) :=
by
  sorry

end midpoint_of_hyperbola_segment_l2199_219955


namespace intersection_shape_is_rectangle_l2199_219905

noncomputable def curve1 (x y : ℝ) : Prop := x * y = 16
noncomputable def curve2 (x y : ℝ) : Prop := x^2 + y^2 = 34

theorem intersection_shape_is_rectangle (x y : ℝ) :
  (curve1 x y ∧ curve2 x y) → 
  ∃ p1 p2 p3 p4 : ℝ × ℝ,
    (curve1 p1.1 p1.2 ∧ curve1 p2.1 p2.2 ∧ curve1 p3.1 p3.2 ∧ curve1 p4.1 p4.2) ∧
    (curve2 p1.1 p1.2 ∧ curve2 p2.1 p2.2 ∧ curve2 p3.1 p3.2 ∧ curve2 p4.1 p4.2) ∧ 
    (dist p1 p2 = dist p3 p4 ∧ dist p2 p3 = dist p4 p1) ∧ 
    (∃ m : ℝ, p1.1 = p2.1 ∧ p3.1 = p4.1 ∧ p1.1 ≠ m ∧ p2.1 ≠ m) := sorry

end intersection_shape_is_rectangle_l2199_219905


namespace cows_in_group_l2199_219901

theorem cows_in_group (c h : ℕ) (h_condition : 4 * c + 2 * h = 2 * (c + h) + 16) : c = 8 :=
sorry

end cows_in_group_l2199_219901


namespace boxes_per_case_l2199_219936

theorem boxes_per_case (total_boxes : ℕ) (total_cases : ℕ) (h1 : total_boxes = 24) (h2 : total_cases = 3) : (total_boxes / total_cases) = 8 :=
by 
  sorry

end boxes_per_case_l2199_219936


namespace total_wheels_l2199_219992

def cars : Nat := 15
def bicycles : Nat := 3
def trucks : Nat := 8
def tricycles : Nat := 1
def wheels_per_car_or_truck : Nat := 4
def wheels_per_bicycle : Nat := 2
def wheels_per_tricycle : Nat := 3

theorem total_wheels : cars * wheels_per_car_or_truck + trucks * wheels_per_car_or_truck + bicycles * wheels_per_bicycle + tricycles * wheels_per_tricycle = 101 :=
by
  sorry

end total_wheels_l2199_219992


namespace exists_subset_with_property_l2199_219910

theorem exists_subset_with_property :
  ∃ X : Set Int, ∀ n : Int, ∃ (a b : X), a + 2 * b = n ∧ ∀ (a' b' : X), (a + 2 * b = n ∧ a' + 2 * b' = n) → (a = a' ∧ b = b') :=
sorry

end exists_subset_with_property_l2199_219910


namespace sum_m_n_eq_zero_l2199_219925

theorem sum_m_n_eq_zero (m n p : ℝ) (h1 : m * n + p^2 + 4 = 0) (h2 : m - n = 4) : m + n = 0 := 
  sorry

end sum_m_n_eq_zero_l2199_219925


namespace ratio_area_ADE_BCED_is_8_over_9_l2199_219958

noncomputable def ratio_area_ADE_BCED 
  (AB BC AC AD AE : ℝ)
  (hAB : AB = 30)
  (hBC : BC = 45)
  (hAC : AC = 54)
  (hAD : AD = 20)
  (hAE : AE = 24) : ℝ := 
  sorry

theorem ratio_area_ADE_BCED_is_8_over_9 
  (AB BC AC AD AE : ℝ)
  (hAB : AB = 30)
  (hBC : BC = 45)
  (hAC : AC = 54)
  (hAD : AD = 20)
  (hAE : AE = 24) :
  ratio_area_ADE_BCED AB BC AC AD AE hAB hBC hAC hAD hAE = 8 / 9 :=
  sorry

end ratio_area_ADE_BCED_is_8_over_9_l2199_219958


namespace hours_per_batch_l2199_219974

noncomputable section

def gallons_per_batch : ℕ := 3 / 2   -- 1.5 gallons expressed as a rational number
def ounces_per_gallon : ℕ := 128
def jack_consumption_per_2_days : ℕ := 96
def total_days : ℕ := 24
def time_spent_hours : ℕ := 120

def total_ounces : ℕ := gallons_per_batch * ounces_per_gallon
def total_ounces_consumed_24_days : ℕ := jack_consumption_per_2_days * (total_days / 2)
def number_of_batches : ℕ := total_ounces_consumed_24_days / total_ounces

theorem hours_per_batch :
  (time_spent_hours / number_of_batches) = 20 := by
  sorry

end hours_per_batch_l2199_219974


namespace action_figures_more_than_books_l2199_219924

variable (initialActionFigures : Nat) (newActionFigures : Nat) (books : Nat)

def totalActionFigures (initialActionFigures newActionFigures : Nat) : Nat :=
  initialActionFigures + newActionFigures

theorem action_figures_more_than_books :
  initialActionFigures = 5 → newActionFigures = 7 → books = 9 →
  totalActionFigures initialActionFigures newActionFigures - books = 3 :=
by
  intros h_initial h_new h_books
  rw [h_initial, h_new, h_books]
  sorry

end action_figures_more_than_books_l2199_219924


namespace rectangle_area_l2199_219947

theorem rectangle_area (x : ℝ) (w : ℝ) (h_diag : (3 * w) ^ 2 + w ^ 2 = x ^ 2) : 
  3 * w ^ 2 = (3 / 10) * x ^ 2 :=
by
  sorry

end rectangle_area_l2199_219947


namespace fraction_to_decimal_l2199_219968

theorem fraction_to_decimal : (5 / 16 : ℝ) = 0.3125 :=
by sorry

end fraction_to_decimal_l2199_219968


namespace trigonometric_expression_value_l2199_219994

noncomputable def trigonometric_expression (α : ℝ) : ℝ :=
  (|Real.tan α| / Real.tan α) + (Real.sin α / Real.sqrt ((1 - Real.cos (2 * α)) / 2))

theorem trigonometric_expression_value (α : ℝ) (h : Real.sin α = -Real.cos α) : 
  trigonometric_expression α = 0 ∨ trigonometric_expression α = -2 :=
by 
  sorry

end trigonometric_expression_value_l2199_219994


namespace sqrt_expression_eq_three_l2199_219967

theorem sqrt_expression_eq_three (h: (Real.sqrt 81) = 9) : Real.sqrt ((Real.sqrt 81 + Real.sqrt 81) / 2) = 3 :=
by 
  sorry

end sqrt_expression_eq_three_l2199_219967


namespace sufficient_but_not_necessary_l2199_219933

theorem sufficient_but_not_necessary (a b : ℝ) (h : a * b ≠ 0) : 
  (¬ (a = 0)) ∧ ¬ ((a ≠ 0) → (a * b ≠ 0)) :=
by {
  -- The proof will be constructed here and is omitted as per the instructions
  sorry
}

end sufficient_but_not_necessary_l2199_219933


namespace max_value_inequality_l2199_219965

theorem max_value_inequality
  (x1 x2 y1 y2 : ℝ)
  (h1 : x1^2 + y1^2 = 1)
  (h2 : x2^2 + y2^2 = 1)
  (h3 : x1 * x2 + y1 * y2 = ⅟2) :
  (|x1 + y1 - 1| / Real.sqrt 2) + (|x2 + y2 - 1| / Real.sqrt 2) ≤ 1 :=
by {
  sorry
}

end max_value_inequality_l2199_219965


namespace malcolm_brushes_teeth_l2199_219953

theorem malcolm_brushes_teeth :
  (∃ (M : ℕ), M = 180 ∧ (∃ (N : ℕ), N = 90 ∧ (M / N = 2))) :=
by
  sorry

end malcolm_brushes_teeth_l2199_219953


namespace coffee_bags_per_week_l2199_219945

def bags_morning : Nat := 3
def bags_afternoon : Nat := 3 * bags_morning
def bags_evening : Nat := 2 * bags_morning
def bags_per_day : Nat := bags_morning + bags_afternoon + bags_evening
def days_per_week : Nat := 7

theorem coffee_bags_per_week : bags_per_day * days_per_week = 126 := by
  sorry

end coffee_bags_per_week_l2199_219945


namespace max_value_of_f_l2199_219914

noncomputable def f (x : ℝ) : ℝ := 3 * x - x ^ 3

theorem max_value_of_f (a b : ℝ) (ha : ∀ x, f x ≤ b) (hfa : f a = b) : a - b = -1 :=
by
  sorry

end max_value_of_f_l2199_219914


namespace polynomial_sum_l2199_219937

def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 3
def g (x : ℝ) : ℝ := -3 * x^2 + 7 * x - 6
def h (x : ℝ) : ℝ := 3 * x^2 - 3 * x + 2
def j (x : ℝ) : ℝ := x^2 + x - 1

theorem polynomial_sum (x : ℝ) : f x + g x + h x + j x = 3 * x^2 + x - 2 := by
  sorry

end polynomial_sum_l2199_219937


namespace highest_power_of_3_l2199_219980

-- Define the integer M formed by concatenating the 3-digit numbers from 100 to 250
def M : ℕ := sorry  -- We should define it in a way that represents the concatenation

-- Define a proof that the highest power of 3 that divides M is 3^1
theorem highest_power_of_3 (n : ℕ) (h : M = n) : ∃ m : ℕ, 3^m ∣ n ∧ ¬ (3^(m + 1) ∣ n) ∧ m = 1 :=
by sorry  -- We will not provide proofs; we're only writing the statement

end highest_power_of_3_l2199_219980


namespace coeff_x2y2_in_expansion_l2199_219969

-- Define the coefficient of a specific term in the binomial expansion
def coeff_binom (n k : ℕ) (a b : ℤ) (x y : ℕ) : ℤ :=
  (Nat.choose n k) * (a ^ (n - k)) * (b ^ k)

theorem coeff_x2y2_in_expansion : coeff_binom 4 2 1 (-2) 2 2 = 24 := by
  sorry

end coeff_x2y2_in_expansion_l2199_219969


namespace cost_of_banana_l2199_219940

theorem cost_of_banana (B : ℝ) (apples bananas oranges total_pieces total_cost : ℝ) 
  (h1 : apples = 12) (h2 : bananas = 4) (h3 : oranges = 4) 
  (h4 : total_pieces = 20) (h5 : total_cost = 40)
  (h6 : 2 * apples + 3 * oranges + bananas * B = total_cost)
  : B = 1 :=
by
  sorry

end cost_of_banana_l2199_219940


namespace regular_18gon_symmetries_l2199_219907

theorem regular_18gon_symmetries :
  let L := 18
  let R := 20
  L + R = 38 := by
sorry

end regular_18gon_symmetries_l2199_219907


namespace trigonometric_identity_l2199_219954

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = 2) : 
  (Real.sin θ * Real.cos θ) / (1 + Real.sin θ ^ 2) = 2 / 9 := 
sorry

end trigonometric_identity_l2199_219954


namespace angle_measure_l2199_219943

theorem angle_measure (x : ℝ) (h : x + (3 * x - 10) = 180) : x = 47.5 := 
by
  sorry

end angle_measure_l2199_219943


namespace find_position_2002_l2199_219919

def T (n : ℕ) : ℕ := n * (n + 1) / 2
def a (n : ℕ) : ℕ := T n + 1

theorem find_position_2002 : ∃ row col : ℕ, 1 ≤ row ∧ 1 ≤ col ∧ (a (row - 1) + (col - 1) = 2002 ∧ row = 15 ∧ col = 49) := 
sorry

end find_position_2002_l2199_219919


namespace intersection_of_sets_l2199_219904

theorem intersection_of_sets :
  let A := {y : ℝ | ∃ x : ℝ, y = Real.sin x}
  let B := {y : ℝ | ∃ x : ℝ, y = Real.sqrt (-(x^2 - 4*x + 3))}
  A ∩ B = {y : ℝ | 0 ≤ y ∧ y ≤ 1} :=
by
  sorry

end intersection_of_sets_l2199_219904


namespace intersection_P_Q_correct_l2199_219921

-- Define sets P and Q based on given conditions
def is_in_P (x : ℝ) : Prop := x > 1
def is_in_Q (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 2

-- Define the intersection P ∩ Q and the correct answer
def P_inter_Q (x : ℝ) : Prop := is_in_P x ∧ is_in_Q x
def correct_ans (x : ℝ) : Prop := 1 < x ∧ x ≤ 2

-- Prove that P ∩ Q = (1, 2]
theorem intersection_P_Q_correct : ∀ x : ℝ, P_inter_Q x ↔ correct_ans x :=
by sorry

end intersection_P_Q_correct_l2199_219921


namespace number_of_wickets_last_match_l2199_219962

noncomputable def bowling_average : ℝ := 12.4
noncomputable def runs_taken_last_match : ℝ := 26
noncomputable def wickets_before_last_match : ℕ := 175
noncomputable def decrease_in_average : ℝ := 0.4
noncomputable def new_average : ℝ := bowling_average - decrease_in_average

theorem number_of_wickets_last_match (w : ℝ) :
  (175 + w) > 0 → 
  ((wickets_before_last_match * bowling_average + runs_taken_last_match) / (wickets_before_last_match + w) = new_average) →
  w = 8 := 
sorry

end number_of_wickets_last_match_l2199_219962


namespace division_problem_l2199_219978

theorem division_problem (x : ℕ) (h : x / 5 = 30 + x / 6) : x = 900 :=
sorry

end division_problem_l2199_219978


namespace find_number_being_divided_l2199_219961

theorem find_number_being_divided (divisor quotient remainder : ℕ) (h1: divisor = 15) (h2: quotient = 9) (h3: remainder = 1) : 
  divisor * quotient + remainder = 136 :=
by
  -- Simplification and computation would follow here
  sorry

end find_number_being_divided_l2199_219961


namespace perimeter_of_square_l2199_219998

theorem perimeter_of_square
  (length_rect : ℕ) (width_rect : ℕ) (area_rect : ℕ)
  (area_square : ℕ) (side_square : ℕ) (perimeter_square : ℕ) :
  (length_rect = 32) → (width_rect = 10) → 
  (area_rect = length_rect * width_rect) →
  (area_square = 5 * area_rect) →
  (side_square * side_square = area_square) →
  (perimeter_square = 4 * side_square) →
  perimeter_square = 160 :=
by
  intros h1 h2 h3 h4 h5 h6
  -- Proof would go here
  sorry

end perimeter_of_square_l2199_219998


namespace wife_catch_up_l2199_219938

/-- A man drives at a speed of 40 miles/hr.
His wife left 30 minutes late with a speed of 50 miles/hr.
Prove that they will meet 2 hours after the wife starts driving. -/
theorem wife_catch_up (t : ℝ) (speed_man speed_wife : ℝ) (late_time : ℝ) :
  speed_man = 40 →
  speed_wife = 50 →
  late_time = 0.5 →
  50 * t = 40 * (t + 0.5) →
  t = 2 :=
by
  intros h_man h_wife h_late h_eq
  -- Actual proof goes here. 
  -- (Skipping the proof as requested, leaving it as a placeholder)
  sorry

end wife_catch_up_l2199_219938


namespace transform_polynomial_l2199_219997

theorem transform_polynomial (x y : ℝ) 
  (h1 : y = x + 1 / x) 
  (h2 : x^4 - x^3 - 2 * x^2 - x + 1 = 0) : x^2 * (y^2 - y - 4) = 0 :=
sorry

end transform_polynomial_l2199_219997


namespace total_games_in_season_l2199_219949

theorem total_games_in_season {n : ℕ} {k : ℕ} (h1 : n = 25) (h2 : k = 15) :
  (n * (n - 1) / 2) * k = 4500 :=
by
  sorry

end total_games_in_season_l2199_219949


namespace brick_surface_area_l2199_219982

variable (X Y Z : ℝ)

#check 4 * X + 4 * Y + 2 * Z = 72 → 
       4 * X + 2 * Y + 4 * Z = 96 → 
       2 * X + 4 * Y + 4 * Z = 102 →
       2 * (X + Y + Z) = 54

theorem brick_surface_area (h1 : 4 * X + 4 * Y + 2 * Z = 72)
                           (h2 : 4 * X + 2 * Y + 4 * Z = 96)
                           (h3 : 2 * X + 4 * Y + 4 * Z = 102) :
                           2 * (X + Y + Z) = 54 := by
  sorry

end brick_surface_area_l2199_219982


namespace mean_score_of_seniors_l2199_219942

theorem mean_score_of_seniors (num_students : ℕ) (mean_score : ℚ) 
  (ratio_non_seniors_seniors : ℚ) (ratio_mean_seniors_non_seniors : ℚ) (total_score_seniors : ℚ) :
  num_students = 200 →
  mean_score = 80 →
  ratio_non_seniors_seniors = 1.25 →
  ratio_mean_seniors_non_seniors = 1.2 →
  total_score_seniors = 7200 →
  let num_seniors := (num_students : ℚ) / (1 + ratio_non_seniors_seniors)
  let mean_score_seniors := total_score_seniors / num_seniors
  mean_score_seniors = 80.9 :=
by 
  sorry

end mean_score_of_seniors_l2199_219942


namespace Jason_has_22_5_toys_l2199_219966

noncomputable def RachelToys : ℝ := 1
noncomputable def JohnToys : ℝ := RachelToys + 6.5
noncomputable def JasonToys : ℝ := 3 * JohnToys

theorem Jason_has_22_5_toys : JasonToys = 22.5 := sorry

end Jason_has_22_5_toys_l2199_219966


namespace polynomial_value_l2199_219963

-- Define the conditions as Lean definitions
def condition (x : ℝ) : Prop := x^2 + 2 * x + 1 = 4

-- State the theorem to be proved
theorem polynomial_value (x : ℝ) (h : condition x) : 2 * x^2 + 4 * x + 5 = 11 :=
by
  -- Proof goes here
  sorry

end polynomial_value_l2199_219963


namespace rectangle_perimeter_at_least_l2199_219996

theorem rectangle_perimeter_at_least (m : ℕ) (m_pos : 0 < m) :
  (∃ a b : ℝ, 0 < a ∧ 0 < b ∧ a * b ≥ 1 / (m * m) ∧ 2 * (a + b) ≥ 4 / m) := sorry

end rectangle_perimeter_at_least_l2199_219996


namespace andrew_age_l2199_219951

-- Definitions based on the conditions
variables (a g : ℝ)

-- The conditions
def condition1 : Prop := g = 9 * a
def condition2 : Prop := g - a = 63

-- The theorem we want to prove
theorem andrew_age (h1 : condition1 a g) (h2 : condition2 a g) : a = 63 / 8 :=
by
  intros
  sorry

end andrew_age_l2199_219951


namespace part1_part2_part3_l2199_219989

-- Definitions for the conditions
def not_divisible_by_2_or_3 (k : ℤ) : Prop :=
  ¬(k % 2 = 0 ∨ k % 3 = 0)

def form_6n1_or_6n5 (k : ℤ) : Prop :=
  ∃ (n : ℤ), k = 6 * n + 1 ∨ k = 6 * n + 5

-- Part 1
theorem part1 (k : ℤ) (h : not_divisible_by_2_or_3 k) : form_6n1_or_6n5 k :=
sorry

-- Part 2
def form_6n1 (a : ℤ) : Prop :=
  ∃ (n : ℤ), a = 6 * n + 1

def form_6n5 (a : ℤ) : Prop :=
  ∃ (n : ℤ), a = 6 * n + 5

theorem part2 (a b : ℤ) (ha : form_6n1 a ∨ form_6n5 a) (hb : form_6n1 b ∨ form_6n5 b) :
  form_6n1 (a * b) :=
sorry

-- Part 3
theorem part3 (a b : ℤ) (ha : form_6n1 a) (hb : form_6n5 b) :
  form_6n5 (a * b) :=
sorry

end part1_part2_part3_l2199_219989


namespace ratio_of_cans_l2199_219970

theorem ratio_of_cans (martha_cans : ℕ) (total_required : ℕ) (remaining_cans : ℕ) (diego_cans : ℕ) (ratio : ℚ) 
  (h1 : martha_cans = 90) 
  (h2 : total_required = 150) 
  (h3 : remaining_cans = 5) 
  (h4 : martha_cans + diego_cans = total_required - remaining_cans) 
  (h5 : ratio = (diego_cans : ℚ) / martha_cans) : 
  ratio = 11 / 18 := 
by
  sorry

end ratio_of_cans_l2199_219970


namespace highest_power_of_5_dividing_S_l2199_219931

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def f (n : ℕ) : ℤ :=
  if sum_of_digits n % 2 = 0 then n ^ 100 else -n ^ 100

def S : ℤ :=
  (Finset.range (10 ^ 100)).sum (λ n => f n)

theorem highest_power_of_5_dividing_S :
  ∃ m : ℕ, 5 ^ m ∣ S ∧ ∀ k : ℕ, 5 ^ (k + 1) ∣ S → k < 24 :=
by
  sorry

end highest_power_of_5_dividing_S_l2199_219931


namespace probability_single_draws_probability_two_different_colors_l2199_219928

-- Define probabilities for black, yellow and green as events A, B, and C respectively.
variables (A B C : ℝ)

-- Conditions based on the problem statement
axiom h1 : A + B = 5/9
axiom h2 : B + C = 2/3
axiom h3 : A + B + C = 1

-- Here is the statement to prove the calculated probabilities of single draws
theorem probability_single_draws : 
  A = 1/3 ∧ B = 2/9 ∧ C = 4/9 :=
sorry

-- Define the event of drawing two balls of the same color
variables (black yellow green : ℕ)
axiom balls_count : black + yellow + green = 9
axiom black_component : A = black / 9
axiom yellow_component : B = yellow / 9
axiom green_component : C = green / 9

-- Using the counts to infer the probability of drawing two balls of different colors
axiom h4 : black = 3
axiom h5 : yellow = 2
axiom h6 : green = 4

theorem probability_two_different_colors :
  (1 - (3/36 + 1/36 + 6/36)) = 13/18 :=
sorry

end probability_single_draws_probability_two_different_colors_l2199_219928


namespace circumference_greater_than_100_l2199_219908

def running_conditions (A B : ℝ) (C : ℝ) (P : ℝ) : Prop :=
  A > 0 ∧ B > 0 ∧ A ≠ B ∧ P = 0 ∧ C > 0

theorem circumference_greater_than_100 (A B C P : ℝ) (h : running_conditions A B C P):
  C > 100 :=
by
  sorry

end circumference_greater_than_100_l2199_219908


namespace inequality_holds_l2199_219993

theorem inequality_holds (x y : ℝ) (hx₀ : 0 < x) (hy₀ : 0 < y) (hxy : x + y = 1) :
  (1 / x^2 - 1) * (1 / y^2 - 1) ≥ 9 :=
sorry

end inequality_holds_l2199_219993


namespace train_A_total_distance_l2199_219956

variables (Speed_A : ℝ) (Time_meet : ℝ) (Total_Distance : ℝ)

def Distance_A_to_C (Speed_A Time_meet : ℝ) : ℝ := Speed_A * Time_meet
def Distance_B_to_C (Total_Distance Distance_A_to_C : ℝ) : ℝ := Total_Distance - Distance_A_to_C
def Additional_Distance_A (Speed_A Time_meet : ℝ) : ℝ := Speed_A * Time_meet
def Total_Distance_A (Distance_A_to_C Additional_Distance_A : ℝ) : ℝ :=
  Distance_A_to_C + Additional_Distance_A

theorem train_A_total_distance
  (h1 : Speed_A = 50)
  (h2 : Time_meet = 0.5)
  (h3 : Total_Distance = 120) :
  Total_Distance_A (Distance_A_to_C Speed_A Time_meet)
                   (Additional_Distance_A Speed_A Time_meet) = 50 :=
by 
  rw [Distance_A_to_C, Additional_Distance_A, Total_Distance_A]
  rw [h1, h2]
  norm_num

end train_A_total_distance_l2199_219956


namespace combined_mpg_l2199_219964

theorem combined_mpg (m : ℕ) (ray_mpg tom_mpg : ℕ) (h1 : m = 200) (h2 : ray_mpg = 40) (h3 : tom_mpg = 20) :
  (m / (m / (2 * ray_mpg) + m / (2 * tom_mpg))) = 80 / 3 :=
by
  sorry

end combined_mpg_l2199_219964


namespace average_income_BC_l2199_219903

theorem average_income_BC {A_income B_income C_income : ℝ}
  (hAB : (A_income + B_income) / 2 = 4050)
  (hAC : (A_income + C_income) / 2 = 4200)
  (hA : A_income = 3000) :
  (B_income + C_income) / 2 = 5250 :=
by sorry

end average_income_BC_l2199_219903


namespace second_number_exists_l2199_219960

theorem second_number_exists (x : ℕ) (h : 150 / x = 15) : x = 10 :=
sorry

end second_number_exists_l2199_219960


namespace instantaneous_velocity_at_t_5_l2199_219987

noncomputable def s (t : ℝ) : ℝ := (1/4) * t^4 - 3

theorem instantaneous_velocity_at_t_5 : 
  (deriv s 5) = 125 :=
by
  sorry

end instantaneous_velocity_at_t_5_l2199_219987


namespace minimum_value_of_angle_l2199_219929

theorem minimum_value_of_angle
  (α : ℝ)
  (h : ∃ x y : ℝ, (x, y) = (Real.sin (2 * Real.pi / 3), Real.cos (2 * Real.pi / 3))) :
  α = 11 * Real.pi / 6 :=
sorry

end minimum_value_of_angle_l2199_219929


namespace train_crosses_platform_l2199_219991

theorem train_crosses_platform :
  ∀ (L : ℕ), 
  (300 + L) / (50 / 3) = 48 → 
  L = 500 := 
by
  sorry

end train_crosses_platform_l2199_219991


namespace find_extrema_l2199_219975

-- Define the variables and the constraints
variables (x y z : ℝ)

-- Define the inequalities as conditions
def cond1 := -1 ≤ 2 * x + y - z ∧ 2 * x + y - z ≤ 8
def cond2 := 2 ≤ x - y + z ∧ x - y + z ≤ 9
def cond3 := -3 ≤ x + 2 * y - z ∧ x + 2 * y - z ≤ 7

-- Define the function f
def f (x y z : ℝ) := 7 * x + 5 * y - 2 * z

-- State the theorem that needs to be proved
theorem find_extrema :
  (∃ x y z, cond1 x y z ∧ cond2 x y z ∧ cond3 x y z) →
  (-6 ≤ f x y z ∧ f x y z ≤ 47) :=
by sorry

end find_extrema_l2199_219975


namespace moon_speed_conversion_l2199_219902

theorem moon_speed_conversion
  (speed_kps : ℝ)
  (seconds_per_hour : ℝ)
  (h1 : speed_kps = 0.2)
  (h2 : seconds_per_hour = 3600) :
  speed_kps * seconds_per_hour = 720 := by
  sorry

end moon_speed_conversion_l2199_219902


namespace value_of_x_l2199_219920

theorem value_of_x (n x : ℝ) (h1: x = 3 * n) (h2: 2 * n + 3 = 0.2 * 25) : x = 3 :=
by
  sorry

end value_of_x_l2199_219920


namespace area_correct_l2199_219913

noncomputable def area_of_30_60_90_triangle (hypotenuse : ℝ) (angle : ℝ) : ℝ :=
if hypotenuse = 10 ∧ angle = 30 then 25 * Real.sqrt 3 / 2 else 0

theorem area_correct {hypotenuse angle : ℝ} (h1 : hypotenuse = 10) (h2 : angle = 30) :
  area_of_30_60_90_triangle hypotenuse angle = 25 * Real.sqrt 3 / 2 :=
by
  sorry

end area_correct_l2199_219913


namespace lcm_of_8_12_15_l2199_219990

theorem lcm_of_8_12_15 : Nat.lcm 8 (Nat.lcm 12 15) = 120 :=
by
  -- This is where the proof steps would go
  sorry

end lcm_of_8_12_15_l2199_219990


namespace determinant_of_matrix_A_l2199_219984

noncomputable def matrix_A (x : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := 
  ![![x + 2, x + 1, x], 
    ![x, x + 2, x + 1], 
    ![x + 1, x, x + 2]]

theorem determinant_of_matrix_A (x : ℝ) :
  (matrix_A x).det = x^2 + 11 * x + 9 :=
by sorry

end determinant_of_matrix_A_l2199_219984


namespace average_words_per_hour_l2199_219950

-- Define the given conditions
variables (W : ℕ) (H : ℕ)

-- State constants for the known values
def words := 60000
def writing_hours := 100

-- Define theorem to prove the average words per hour during the writing phase
theorem average_words_per_hour (h : W = words) (h2 : H = writing_hours) : (W / H) = 600 := by
  sorry

end average_words_per_hour_l2199_219950


namespace digit_C_equals_one_l2199_219959

-- Define the scope of digits
def is_digit (n : ℕ) : Prop := n < 10

-- Define the equality for sums of digits
def sum_of_digits (A B C : ℕ) : Prop := A + B + C = 10

-- Main theorem to prove C = 1
theorem digit_C_equals_one (A B C : ℕ) (hA : is_digit A) (hB : is_digit B) (hC : is_digit C) (hSum : sum_of_digits A B C) : C = 1 :=
sorry

end digit_C_equals_one_l2199_219959


namespace Pima_investment_value_at_week6_l2199_219915

noncomputable def Pima_initial_investment : ℝ := 400
noncomputable def Pima_week1_gain : ℝ := 0.25
noncomputable def Pima_week1_addition : ℝ := 200
noncomputable def Pima_week2_gain : ℝ := 0.50
noncomputable def Pima_week2_withdrawal : ℝ := 150
noncomputable def Pima_week3_loss : ℝ := 0.10
noncomputable def Pima_week4_gain : ℝ := 0.20
noncomputable def Pima_week4_addition : ℝ := 100
noncomputable def Pima_week5_gain : ℝ := 0.05
noncomputable def Pima_week6_loss : ℝ := 0.15
noncomputable def Pima_week6_withdrawal : ℝ := 250
noncomputable def weekly_interest_rate : ℝ := 0.02

noncomputable def calculate_investment_value : ℝ :=
  let week0 := Pima_initial_investment
  let week1 := (week0 * (1 + Pima_week1_gain) * (1 + weekly_interest_rate)) + Pima_week1_addition
  let week2 := ((week1 * (1 + Pima_week2_gain) * (1 + weekly_interest_rate)) - Pima_week2_withdrawal)
  let week3 := (week2 * (1 - Pima_week3_loss) * (1 + weekly_interest_rate))
  let week4 := ((week3 * (1 + Pima_week4_gain) * (1 + weekly_interest_rate)) + Pima_week4_addition)
  let week5 := (week4 * (1 + Pima_week5_gain) * (1 + weekly_interest_rate))
  let week6 := ((week5 * (1 - Pima_week6_loss) * (1 + weekly_interest_rate)) - Pima_week6_withdrawal)
  week6

theorem Pima_investment_value_at_week6 : calculate_investment_value = 819.74 := 
  by
  sorry

end Pima_investment_value_at_week6_l2199_219915


namespace ab_bd_ratio_l2199_219906

-- Definitions based on the conditions
variables {A B C D : ℝ}
variables (h1 : A / B = 1 / 2) (h2 : B / C = 8 / 5)

-- Math equivalence proving AB/BD = 4/13 based on given conditions
theorem ab_bd_ratio
  (h1 : A / B = 1 / 2)
  (h2 : B / C = 8 / 5) :
  A / (B + C) = 4 / 13 :=
by
  sorry

end ab_bd_ratio_l2199_219906


namespace not_prime_for_large_n_l2199_219971

theorem not_prime_for_large_n {n : ℕ} (h : n > 1) : ¬ Prime (n^4 + n^2 + 1) :=
sorry

end not_prime_for_large_n_l2199_219971


namespace age_difference_two_children_l2199_219985

/-!
# Age difference between two children in a family

## Given:
- 10 years ago, the average age of a family of 4 members was 24 years.
- Two children have been born since then.
- The present average age of the family (now 6 members) is the same, 24 years.
- The present age of the youngest child (Y1) is 3 years.

## Prove:
The age difference between the two children is 2 years.
-/

theorem age_difference_two_children :
  let Y1 := 3
  let Y2 := 5
  let total_age_10_years_ago := 4 * 24
  let total_age_now := 6 * 24
  let increase_age_10_years := total_age_now - total_age_10_years_ago
  let increase_due_to_original_members := 4 * 10
  let increase_due_to_children := increase_age_10_years - increase_due_to_original_members
  Y1 + Y2 = increase_due_to_children
  → Y2 - Y1 = 2 :=
by
  intros
  sorry

end age_difference_two_children_l2199_219985
