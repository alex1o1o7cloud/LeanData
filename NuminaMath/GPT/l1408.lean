import Mathlib

namespace minimum_value_l1408_140881

theorem minimum_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 6) :
    6 ≤ (x^2 + 2*y^2) / (x + y) + (x^2 + 2*z^2) / (x + z) + (y^2 + 2*z^2) / (y + z) :=
by
  sorry

end minimum_value_l1408_140881


namespace spinsters_count_l1408_140838

variable (S C : ℕ)

-- defining the conditions
def ratio_condition (S C : ℕ) : Prop := 9 * S = 2 * C
def difference_condition (S C : ℕ) : Prop := C = S + 63

-- theorem to prove
theorem spinsters_count 
  (h1 : ratio_condition S C) 
  (h2 : difference_condition S C) : 
  S = 18 :=
sorry

end spinsters_count_l1408_140838


namespace power_inequality_l1408_140806

theorem power_inequality (a b c d : ℝ) (ha : 0 < a) (hab : a ≤ b) (hbc : b ≤ c) (hcd : c ≤ d) :
  a^b * b^c * c^d * d^a ≥ b^a * c^b * d^c * a^d := 
sorry

end power_inequality_l1408_140806


namespace inequality_proof_l1408_140813

variable {a b c d : ℝ}

theorem inequality_proof (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  64 * (abcd + 1) / (a + b + c + d)^2 ≤ a^2 + b^2 + c^2 + d^2 + 1 / a^2 + 1 / b^2 + 1 / c^2 + 1 / d^2 :=
by 
  sorry

end inequality_proof_l1408_140813


namespace monomial_properties_l1408_140895

def coefficient (m : ℝ) := -3
def degree (x_exp y_exp : ℕ) := x_exp + y_exp

theorem monomial_properties :
  ∀ (x_exp y_exp : ℕ), coefficient (-3) = -3 ∧ degree 2 1 = 3 :=
by
  sorry

end monomial_properties_l1408_140895


namespace quadratic_equation_solution_l1408_140890

noncomputable def findOrderPair (b d : ℝ) : Prop :=
  (b + d = 7) ∧ (b < d) ∧ (36 - 4 * b * d = 0)

theorem quadratic_equation_solution :
  ∃ b d : ℝ, findOrderPair b d ∧ (b, d) = ( (7 - Real.sqrt 13) / 2, (7 + Real.sqrt 13) / 2 ) :=
by
  sorry

end quadratic_equation_solution_l1408_140890


namespace material_needed_l1408_140842

-- Define the required conditions
def feet_per_tee_shirt : ℕ := 4
def number_of_tee_shirts : ℕ := 15

-- State the theorem and the proof obligation
theorem material_needed : feet_per_tee_shirt * number_of_tee_shirts = 60 := 
by 
  sorry

end material_needed_l1408_140842


namespace berry_circle_properties_l1408_140837

theorem berry_circle_properties :
  ∃ r : ℝ, (∀ x y : ℝ, x^2 + y^2 - 12 = 2 * x + 4 * y → r = Real.sqrt 17)
    ∧ (π * Real.sqrt 17 ^ 2 > 30) :=
by
  sorry

end berry_circle_properties_l1408_140837


namespace parabola_intersection_l1408_140873

theorem parabola_intersection :
  (∀ x y : ℝ, y = 3 * x^2 - 4 * x + 2 ↔ y = 9 * x^2 + 6 * x + 2) →
  (∃ x1 y1 x2 y2 : ℝ,
    (x1 = 0 ∧ y1 = 2) ∧ (x2 = -5 / 3 ∧ y2 = 17)) :=
by
  intro h
  sorry

end parabola_intersection_l1408_140873


namespace binary_ternary_conversion_l1408_140849

theorem binary_ternary_conversion (a b : ℕ) (h_b : b = 0 ∨ b = 1) (h_a : a = 0 ∨ a = 1 ∨ a = 2)
  (h_eq : 8 + 2 * b + 1 = 9 * a + 2) : 2 * a + b = 3 :=
by
  sorry

end binary_ternary_conversion_l1408_140849


namespace perimeter_of_triangle_l1408_140880

theorem perimeter_of_triangle (side_length : ℕ) (num_sides : ℕ) (h1 : side_length = 7) (h2 : num_sides = 3) : 
  num_sides * side_length = 21 :=
by
  sorry

end perimeter_of_triangle_l1408_140880


namespace find_de_over_ef_l1408_140884

-- Definitions based on problem conditions
variables {A B C D E F : Type} [AddCommGroup A] [AddCommGroup B] [AddCommGroup C] [AddCommGroup D] [AddCommGroup E] [AddCommGroup F] 
variables (a b c d e f : A) 
variables (α β γ δ : ℝ)

-- Conditions
-- AD:DB = 2:3
def d_def : A := (3 / 5) • a + (2 / 5) • b
-- BE:EC = 1:4
def e_def : A := (4 / 5) • b + (1 / 5) • c
-- Intersection F of DE and AC
def f_def : A := (5 • d) - (10 • e)

-- Target Proof
theorem find_de_over_ef (h_d: d = d_def a b) (h_e: e = e_def b c) (h_f: f = f_def d e):
  DE / EF = 1 / 5 := 
sorry

end find_de_over_ef_l1408_140884


namespace contrapositive_iff_l1408_140801

theorem contrapositive_iff (a b : ℤ) : (a > b → a - 5 > b - 5) ↔ (a - 5 ≤ b - 5 → a ≤ b) :=
by sorry

end contrapositive_iff_l1408_140801


namespace three_a_ge_two_b_plus_two_l1408_140872

theorem three_a_ge_two_b_plus_two (a b : ℕ)
  (h1 : 0 < a)
  (h2 : 0 < b)
  (h3 : (a! * b!) % (a! + b!) = 0) :
  3 * a ≥ 2 * b + 2 :=
sorry

end three_a_ge_two_b_plus_two_l1408_140872


namespace cone_height_circular_sector_l1408_140861

theorem cone_height_circular_sector (r : ℝ) (n : ℕ) (h : ℝ)
  (hr : r = 10)
  (hn : n = 3)
  (hradius : r > 0)
  (hcircumference : 2 * Real.pi * r / n = 2 * Real.pi * r / 3)
  : h = (20 * Real.sqrt 2) / 3 :=
by {
  sorry
}

end cone_height_circular_sector_l1408_140861


namespace total_cookies_l1408_140840

   -- Define the conditions
   def cookies_per_bag : ℕ := 41
   def number_of_bags : ℕ := 53

   -- Define the problem: Prove that the total number of cookies is 2173
   theorem total_cookies : cookies_per_bag * number_of_bags = 2173 :=
   by sorry
   
end total_cookies_l1408_140840


namespace net_distance_from_start_total_distance_driven_fuel_consumption_l1408_140893

def driving_distances : List Int := [14, -3, 7, -3, 11, -4, -3, 11, 6, -7, 9]

theorem net_distance_from_start : List.sum driving_distances = 38 := by
  sorry

theorem total_distance_driven : List.sum (List.map Int.natAbs driving_distances) = 78 := by
  sorry

theorem fuel_consumption (fuel_rate : Float) (total_distance : Nat) : total_distance = 78 → total_distance.toFloat * fuel_rate = 7.8 := by
  intros h_total_distance
  rw [h_total_distance]
  norm_num
  sorry

end net_distance_from_start_total_distance_driven_fuel_consumption_l1408_140893


namespace find_the_number_l1408_140839

-- Define the variables and conditions
variable (x z : ℝ)
variable (the_number : ℝ)

-- Condition: given that x = 1
axiom h1 : x = 1

-- Condition: given the equation
axiom h2 : 14 * (-x + z) + 18 = -14 * (x - z) - the_number

-- The theorem to prove
theorem find_the_number : the_number = -4 :=
by
  sorry

end find_the_number_l1408_140839


namespace jaylen_charge_per_yard_l1408_140867

def total_cost : ℝ := 250
def number_of_yards : ℝ := 6
def charge_per_yard : ℝ := 41.67

theorem jaylen_charge_per_yard :
  total_cost / number_of_yards = charge_per_yard :=
sorry

end jaylen_charge_per_yard_l1408_140867


namespace min_value_3x_4y_l1408_140820

noncomputable def minValue (x y : ℝ) : ℝ := 3 * x + 4 * y

theorem min_value_3x_4y 
  (x y : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (h : x + 3 * y = 5 * x * y) : 
  minValue x y ≥ 5 :=
sorry

end min_value_3x_4y_l1408_140820


namespace T_n_bounds_l1408_140824

noncomputable def a_n (n : ℕ) : ℕ := 2 * n + 1

noncomputable def S_n (n : ℕ) : ℕ := n * (n + 2)

noncomputable def b_n (n : ℕ) : ℚ := 
if n ≤ 4 then 2 * n + 1
else 1 / (n * (n + 2))

noncomputable def T_n (n : ℕ) : ℚ := 
if n ≤ 4 then S_n n
else (24 : ℚ) + (1 / 2) * (1 / 5 + 1 / 6 - 1 / (n + 1 : ℚ) - 1 / (n + 2 : ℚ))

theorem T_n_bounds (n : ℕ) : 3 ≤ T_n n ∧ T_n n < 24 + 11 / 60 := by
  sorry

end T_n_bounds_l1408_140824


namespace sequence_number_pair_l1408_140841

theorem sequence_number_pair (n m : ℕ) (h : m ≤ n) : (m, n - m + 1) = (m, n - m + 1) :=
by sorry

end sequence_number_pair_l1408_140841


namespace find_side_c_l1408_140810

noncomputable def triangle_side_c (A b S : ℝ) (c : ℝ) : Prop :=
  S = 0.5 * b * c * Real.sin A

theorem find_side_c :
  ∀ (c : ℝ), triangle_side_c (Real.pi / 3) 16 (64 * Real.sqrt 3) c → c = 16 :=
by
  sorry

end find_side_c_l1408_140810


namespace cos_120_eq_neg_one_half_l1408_140871

theorem cos_120_eq_neg_one_half : Real.cos (120 * Real.pi / 180) = -1 / 2 :=
by
  sorry

end cos_120_eq_neg_one_half_l1408_140871


namespace curve_is_circle_l1408_140817

theorem curve_is_circle (r θ : ℝ) (h : r = 1 / (Real.sin θ + Real.cos θ)) : 
  ∃ k : ℝ, ∀ x y : ℝ, (x^2 + y^2 = k^2) → 
    (r^2 = x^2 + y^2 ∧ ∃ (θ : ℝ), x/r = Real.cos θ ∧ y/r = Real.sin θ) :=
sorry

end curve_is_circle_l1408_140817


namespace ny_mets_fans_count_l1408_140809

variable (Y M R : ℕ) -- Variables representing number of fans
variable (k j : ℕ)   -- Helper variables for ratios

theorem ny_mets_fans_count :
  (Y = 3 * k) →
  (M = 2 * k) →
  (M = 4 * j) →
  (R = 5 * j) →
  (Y + M + R = 330) →
  (∃ (k j : ℕ), k = 2 * j) →
  M = 88 := sorry

end ny_mets_fans_count_l1408_140809


namespace solve_system_l1408_140825

theorem solve_system (x1 x2 x3 : ℝ) :
  (x1 - 2 * x2 + 3 * x3 = 5) ∧ 
  (2 * x1 + 3 * x2 - x3 = 7) ∧ 
  (3 * x1 + x2 + 2 * x3 = 12) 
  ↔ (x1, x2, x3) = (7 - 5 * x3, 1 - x3, x3) :=
by
  sorry

end solve_system_l1408_140825


namespace cubic_function_properties_l1408_140896

noncomputable def f (x : ℝ) : ℝ := x^3 - 6 * x^2 + 9 * x

theorem cubic_function_properties :
  (∀ (x : ℝ), deriv f x = 3 * x^2 - 12 * x + 9) ∧
  (f 1 = 4) ∧ 
  (deriv f 1 = 0) ∧
  (f 3 = 0) ∧ 
  (deriv f 3 = 0) ∧
  (f 0 = 0) :=
by
  sorry

end cubic_function_properties_l1408_140896


namespace set_intersection_complement_l1408_140870

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def P : Set ℕ := {1, 2, 3, 4}
def Q : Set ℕ := {3, 4, 5}

theorem set_intersection_complement :
  P ∩ (U \ Q) = {1, 2} :=
by
  sorry

end set_intersection_complement_l1408_140870


namespace statue_selling_price_l1408_140804

/-- Problem conditions -/
def original_cost : ℤ := 550
def profit_percentage : ℝ := 0.20

/-- Proof problem statement -/
theorem statue_selling_price : original_cost + profit_percentage * original_cost = 660 := by
  sorry

end statue_selling_price_l1408_140804


namespace evaluate_expression_l1408_140882

noncomputable def given_expression : ℝ :=
  |8 - 8 * (3 - 12)^2| - |5 - Real.sin 11| + |2^(4 - 2 * 3) / ((3^2) - 7)|

theorem evaluate_expression : given_expression = 634.125009794 := 
  sorry

end evaluate_expression_l1408_140882


namespace units_digit_fib_cycle_length_60_l1408_140858

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
| 0     => 0
| 1     => 1
| (n+2) => fib n + fib (n+1)

-- Define the function to get the units digit (mod 10)
def units_digit_fib (n : ℕ) : ℕ :=
  (fib n) % 10

-- State the theorem about the cycle length of the units digits in Fibonacci sequence
theorem units_digit_fib_cycle_length_60 :
  ∃ k, k = 60 ∧ ∀ n, units_digit_fib (n + k) = units_digit_fib n := sorry

end units_digit_fib_cycle_length_60_l1408_140858


namespace vertex_x_coordinate_of_quadratic_l1408_140865

-- Define the quadratic function
def quadratic_function (x : ℝ) : ℝ := x^2 - 8 * x + 15

-- Define the x-coordinate of the vertex
def vertex_x_coordinate (f : ℝ → ℝ) : ℝ := 4

-- The theorem to prove
theorem vertex_x_coordinate_of_quadratic :
  vertex_x_coordinate quadratic_function = 4 :=
by
  -- Proof skipped
  sorry

end vertex_x_coordinate_of_quadratic_l1408_140865


namespace minimum_area_convex_quadrilateral_l1408_140866

theorem minimum_area_convex_quadrilateral
  (S_AOB S_COD : ℝ) (h₁ : S_AOB = 4) (h₂ : S_COD = 9) :
  (∀ S_BOC S_AOD : ℝ, S_AOB * S_COD = S_BOC * S_AOD → 
    (S_AOB + S_BOC + S_COD + S_AOD) ≥ 25) := sorry

end minimum_area_convex_quadrilateral_l1408_140866


namespace cost_per_pumpkin_pie_l1408_140894

theorem cost_per_pumpkin_pie
  (pumpkin_pies : ℕ)
  (cherry_pies : ℕ)
  (cost_cherry_pie : ℕ)
  (total_profit : ℕ)
  (selling_price : ℕ)
  (total_revenue : ℕ)
  (total_cost : ℕ)
  (cost_pumpkin_pie : ℕ)
  (H1 : pumpkin_pies = 10)
  (H2 : cherry_pies = 12)
  (H3 : cost_cherry_pie = 5)
  (H4 : total_profit = 20)
  (H5 : selling_price = 5)
  (H6 : total_revenue = (pumpkin_pies + cherry_pies) * selling_price)
  (H7 : total_cost = total_revenue - total_profit)
  (H8 : total_cost = pumpkin_pies * cost_pumpkin_pie + cherry_pies * cost_cherry_pie) :
  cost_pumpkin_pie = 3 :=
by
  -- Placeholder for proof
  sorry

end cost_per_pumpkin_pie_l1408_140894


namespace min_cubes_l1408_140851

theorem min_cubes (a b c : ℕ) (h₁ : (a - 1) * (b - 1) * (c - 1) = 240) : a * b * c = 385 :=
  sorry

end min_cubes_l1408_140851


namespace cos_A_sin_B_eq_l1408_140855

theorem cos_A_sin_B_eq (A B : ℝ) (hA1 : 0 < A) (hA2 : A < π / 2) (hB1 : 0 < B) (hB2 : B < π / 2)
    (h : (4 + (Real.tan A)^2) * (5 + (Real.tan B)^2) = Real.sqrt 320 * Real.tan A * Real.tan B) :
    Real.cos A * Real.sin B = 1 / Real.sqrt 6 := sorry

end cos_A_sin_B_eq_l1408_140855


namespace students_like_both_l1408_140807

variable (total_students : ℕ) 
variable (students_like_sea : ℕ) 
variable (students_like_mountains : ℕ) 
variable (students_like_neither : ℕ) 

theorem students_like_both (h1 : total_students = 500)
                           (h2 : students_like_sea = 337)
                           (h3 : students_like_mountains = 289)
                           (h4 : students_like_neither = 56) :
  (students_like_sea + students_like_mountains - (total_students - students_like_neither)) = 182 :=
sorry

end students_like_both_l1408_140807


namespace positive_distinct_solutions_of_system_l1408_140843

variables {a b x y z : ℝ}

theorem positive_distinct_solutions_of_system
  (h1 : x + y + z = a)
  (h2 : x^2 + y^2 + z^2 = b^2)
  (h3 : xy = z^2) :
  (x > 0 ∧ y > 0 ∧ z > 0 ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z) ↔ (3 * b^2 > a^2 ∧ a^2 > b^2 ∧ a > 0) :=
by
  sorry

end positive_distinct_solutions_of_system_l1408_140843


namespace M_subset_N_cond_l1408_140830

theorem M_subset_N_cond (a : ℝ) (h : 0 < a) :
  (∀ p : ℝ × ℝ, p ∈ {p : ℝ × ℝ | p.fst^2 + p.snd^2 = a^2} → p ∈ {p : ℝ × ℝ | |p.fst + p.snd| + |p.fst - p.snd| ≤ 2}) ↔ (0 < a ∧ a ≤ 1) :=
sorry

end M_subset_N_cond_l1408_140830


namespace trigonometric_identity_l1408_140802

theorem trigonometric_identity (theta : ℝ) (h : Real.cos ((5 * Real.pi)/12 - theta) = 1/3) :
  Real.sin ((Real.pi)/12 + theta) = 1/3 :=
by
  sorry

end trigonometric_identity_l1408_140802


namespace price_of_thermometer_l1408_140832

noncomputable def thermometer_price : ℝ := 2

theorem price_of_thermometer
  (T : ℝ)
  (price_hot_water_bottle : ℝ := 6)
  (hot_water_bottles_sold : ℕ := 60)
  (total_sales : ℝ := 1200)
  (thermometers_sold : ℕ := 7 * hot_water_bottles_sold)
  (thermometers_sales : ℝ := total_sales - (price_hot_water_bottle * hot_water_bottles_sold)) :
  T = thermometer_price :=
by
  sorry

end price_of_thermometer_l1408_140832


namespace initial_interest_rate_l1408_140828

theorem initial_interest_rate 
  (r P : ℝ)
  (h1 : 20250 = P * r)
  (h2 : 22500 = P * (r + 5)) :
  r = 45 :=
by
  sorry

end initial_interest_rate_l1408_140828


namespace black_cards_taken_out_l1408_140836

theorem black_cards_taken_out (total_black_cards remaining_black_cards : ℕ)
  (h1 : total_black_cards = 26) (h2 : remaining_black_cards = 21) :
  total_black_cards - remaining_black_cards = 5 :=
by
  sorry

end black_cards_taken_out_l1408_140836


namespace toothpick_sequence_l1408_140889

theorem toothpick_sequence (a d n : ℕ) (h1 : a = 6) (h2 : d = 4) (h3 : n = 150) : a + (n - 1) * d = 602 := by
  sorry

end toothpick_sequence_l1408_140889


namespace triangle_side_length_c_l1408_140897

theorem triangle_side_length_c (a b : ℝ) (α β γ : ℝ) (h_angle_sum : α + β + γ = 180) (h_angle_eq : 3 * α + 2 * β = 180) (h_a : a = 2) (h_b : b = 3) : 
∃ c : ℝ, c = 4 :=
by
  sorry

end triangle_side_length_c_l1408_140897


namespace second_divisor_l1408_140818

theorem second_divisor (x : ℕ) : (282 % 31 = 3) ∧ (282 % x = 3) → x = 9 :=
by
  sorry

end second_divisor_l1408_140818


namespace road_signs_at_first_intersection_l1408_140803

theorem road_signs_at_first_intersection (x : ℕ) 
    (h1 : x + (x + x / 4) + 2 * (x + x / 4) + (2 * (x + x / 4) - 20) = 270) : 
    x = 40 := 
sorry

end road_signs_at_first_intersection_l1408_140803


namespace wendy_total_glasses_l1408_140826

noncomputable def small_glasses : ℕ := 50
noncomputable def large_glasses : ℕ := small_glasses + 10
noncomputable def total_glasses : ℕ := small_glasses + large_glasses

theorem wendy_total_glasses : total_glasses = 110 :=
by
  sorry

end wendy_total_glasses_l1408_140826


namespace substring_012_appears_148_times_l1408_140800

noncomputable def count_substring_012_in_base_3_concat (n : ℕ) : ℕ :=
  -- The function that counts the "012" substrings in the concatenated base-3 representations
  sorry

theorem substring_012_appears_148_times :
  count_substring_012_in_base_3_concat 728 = 148 :=
  sorry

end substring_012_appears_148_times_l1408_140800


namespace jake_not_drop_coffee_percentage_l1408_140863

-- Definitions for the conditions
def trip_probability : ℝ := 0.40
def drop_when_trip_probability : ℝ := 0.25

-- The question and proof statement
theorem jake_not_drop_coffee_percentage :
  100 * (1 - trip_probability * drop_when_trip_probability) = 90 :=
by
  sorry

end jake_not_drop_coffee_percentage_l1408_140863


namespace find_x_l1408_140811

-- We define the given condition in Lean
theorem find_x (x : ℝ) (h : 6 * x - 12 = -(4 + 2 * x)) : x = 1 :=
sorry

end find_x_l1408_140811


namespace find_x_l1408_140850

theorem find_x (x : ℝ) (h : 70 + 60 / (x / 3) = 71) : x = 180 :=
sorry

end find_x_l1408_140850


namespace sum_of_coefficients_l1408_140856

noncomputable def coeff_sum (x y z : ℝ) : ℝ :=
  let p := (x + 2*y - z)^8  
  -- extract and sum coefficients where exponent of x is 2 and exponent of y is not 1
  sorry

theorem sum_of_coefficients (x y z : ℝ) :
  coeff_sum x y z = 364 := by
  sorry

end sum_of_coefficients_l1408_140856


namespace parabola_vertex_b_l1408_140874

theorem parabola_vertex_b (a b c p : ℝ) (h₁ : p ≠ 0)
  (h₂ : ∀ x, (x = p → -p = a * (p^2) + b * p + c) ∧ (x = 0 → p = c)) :
  b = - (4 / p) :=
sorry

end parabola_vertex_b_l1408_140874


namespace trigonometric_identity_l1408_140898

theorem trigonometric_identity
  (α : ℝ)
  (h : Real.tan α = 2) :
  (4 * Real.sin α ^ 3 - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 2 / 5 :=
by
  sorry

end trigonometric_identity_l1408_140898


namespace old_record_was_300_points_l1408_140862

theorem old_record_was_300_points :
  let touchdowns_per_game := 4
  let points_per_touchdown := 6
  let games_in_season := 15
  let conversions := 6
  let points_per_conversion := 2
  let points_beat := 72
  let total_points := touchdowns_per_game * points_per_touchdown * games_in_season + conversions * points_per_conversion
  total_points - points_beat = 300 := 
by
  sorry

end old_record_was_300_points_l1408_140862


namespace average_episodes_per_year_is_16_l1408_140852

-- Define the number of years the TV show has been running
def years : Nat := 14

-- Define the number of seasons and episodes for each category
def seasons_8_15 : Nat := 8
def episodes_per_season_8_15 : Nat := 15
def seasons_4_20 : Nat := 4
def episodes_per_season_4_20 : Nat := 20
def seasons_2_12 : Nat := 2
def episodes_per_season_2_12 : Nat := 12

-- Define the total number of episodes
def total_episodes : Nat :=
  (seasons_8_15 * episodes_per_season_8_15) + 
  (seasons_4_20 * episodes_per_season_4_20) + 
  (seasons_2_12 * episodes_per_season_2_12)

-- Define the average number of episodes per year
def average_episodes_per_year : Nat :=
  total_episodes / years

-- State the theorem to prove the average number of episodes per year is 16
theorem average_episodes_per_year_is_16 : average_episodes_per_year = 16 :=
by
  sorry

end average_episodes_per_year_is_16_l1408_140852


namespace initial_cheerleaders_count_l1408_140854

theorem initial_cheerleaders_count (C : ℕ) 
  (initial_football_players : ℕ := 13) 
  (quit_football_players : ℕ := 10) 
  (quit_cheerleaders : ℕ := 4) 
  (remaining_people : ℕ := 15) 
  (initial_total : ℕ := initial_football_players + C) 
  (final_total : ℕ := (initial_football_players - quit_football_players) + (C - quit_cheerleaders)) :
  remaining_people = final_total → C = 16 :=
by intros h; sorry

end initial_cheerleaders_count_l1408_140854


namespace monthly_rent_requirement_l1408_140845

noncomputable def initial_investment : Float := 200000
noncomputable def annual_return_rate : Float := 0.06
noncomputable def annual_insurance_cost : Float := 4500
noncomputable def maintenance_percentage : Float := 0.15
noncomputable def required_monthly_rent : Float := 1617.65

theorem monthly_rent_requirement :
  let annual_return := initial_investment * annual_return_rate
  let annual_cost_with_insurance := annual_return + annual_insurance_cost
  let monthly_required_net := annual_cost_with_insurance / 12
  let rental_percentage_kept := 1 - maintenance_percentage
  let monthly_rental_full := monthly_required_net / rental_percentage_kept
  monthly_rental_full = required_monthly_rent := 
by
  sorry

end monthly_rent_requirement_l1408_140845


namespace total_tickets_sold_is_336_l1408_140823

-- Define the costs of the tickets
def cost_vip_ticket : ℕ := 45
def cost_ga_ticket : ℕ := 20

-- Define the total cost collected
def total_cost_collected : ℕ := 7500

-- Define the difference in the number of tickets sold
def vip_less_ga : ℕ := 276

-- Define the main theorem to be proved
theorem total_tickets_sold_is_336 (V G : ℕ) 
  (h1 : cost_vip_ticket * V + cost_ga_ticket * G = total_cost_collected)
  (h2 : V = G - vip_less_ga) : V + G = 336 :=
  sorry

end total_tickets_sold_is_336_l1408_140823


namespace intersection_complement_A_B_l1408_140847

def Universe : Set ℝ := Set.univ

def A : Set ℝ := {x | abs (x - 1) > 2}

def B : Set ℝ := {x | x^2 - 6*x + 8 < 0}

theorem intersection_complement_A_B :
  (Universe \ A) ∩ B = {x : ℝ | 2 < x ∧ x ≤ 3} :=
by
  sorry

end intersection_complement_A_B_l1408_140847


namespace fish_lives_longer_than_dog_l1408_140885

-- Definitions based on conditions
def hamster_lifespan : ℝ := 2.5
def dog_lifespan : ℝ := 4 * hamster_lifespan
def fish_lifespan : ℝ := 12

-- Theorem stating the desired proof
theorem fish_lives_longer_than_dog :
  fish_lifespan - dog_lifespan = 2 := 
sorry

end fish_lives_longer_than_dog_l1408_140885


namespace range_of_a_l1408_140812

noncomputable def f (x : ℝ) (a : ℝ) := Real.log (3 * x + a / x - 2)

theorem range_of_a (a : ℝ) :
  (∀ x y : ℝ, 1 ≤ x → x ≤ y → f x a ≤ f y a) ↔ (-1 < a ∧ a ≤ 3) := 
sorry

end range_of_a_l1408_140812


namespace files_remaining_l1408_140844

theorem files_remaining (music_files video_files deleted_files : ℕ) 
  (h_music : music_files = 13) 
  (h_video : video_files = 30) 
  (h_deleted : deleted_files = 10) : 
  (music_files + video_files - deleted_files) = 33 :=
by
  sorry

end files_remaining_l1408_140844


namespace cliff_collection_has_180_rocks_l1408_140835

noncomputable def cliffTotalRocks : ℕ :=
  let shiny_igneous_rocks := 40
  let total_igneous_rocks := shiny_igneous_rocks * 3 / 2
  let total_sedimentary_rocks := total_igneous_rocks * 2
  total_igneous_rocks + total_sedimentary_rocks

theorem cliff_collection_has_180_rocks :
  let shiny_igneous_rocks := 40
  let total_igneous_rocks := shiny_igneous_rocks * 3 / 2
  let total_sedimentary_rocks := total_igneous_rocks * 2
  total_igneous_rocks + total_sedimentary_rocks = 180 := sorry

end cliff_collection_has_180_rocks_l1408_140835


namespace train_distance_after_braking_l1408_140868

theorem train_distance_after_braking : 
  (∃ t : ℝ, (27 * t - 0.45 * t^2 = 0) ∧ (∀ s : ℝ, s = 27 * t - 0.45 * t^2) ∧ s = 405) :=
sorry

end train_distance_after_braking_l1408_140868


namespace factorize_binomial_square_l1408_140808

theorem factorize_binomial_square (x y : ℝ) : x^2 + 2*x*y + y^2 = (x + y)^2 :=
by
  sorry

end factorize_binomial_square_l1408_140808


namespace parallel_lines_condition_l1408_140831

theorem parallel_lines_condition (a : ℝ) :
  (∀ x y : ℝ, a * x + 2 * y - 4 = 0 → x + (a + 1) * y + 2 = 0) ↔ a = 1 :=
by sorry

end parallel_lines_condition_l1408_140831


namespace arithmetic_seq_inequality_l1408_140829

-- Definition for the sum of the first n terms of an arithmetic sequence
def sum_arith_seq (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * a₁ + n * (n - 1) / 2 * d

theorem arithmetic_seq_inequality (a₁ : ℕ) (d : ℕ) (n : ℕ) (h : d > 0) :
  sum_arith_seq a₁ d n + sum_arith_seq a₁ d (3 * n) > 2 * sum_arith_seq a₁ d (2 * n) := by
  sorry

end arithmetic_seq_inequality_l1408_140829


namespace milan_billed_minutes_l1408_140853

-- Variables corresponding to the conditions
variables (f r b : ℝ) (m : ℕ)

-- The conditions of the problem
def conditions : Prop :=
  f = 2 ∧ r = 0.12 ∧ b = 23.36 ∧ b = f + r * m

-- The theorem based on given conditions and aiming to prove that m = 178
theorem milan_billed_minutes (h : conditions f r b m) : m = 178 :=
sorry

end milan_billed_minutes_l1408_140853


namespace central_student_coins_l1408_140859

theorem central_student_coins (n_students: ℕ) (total_coins : ℕ)
  (equidistant_same : Prop)
  (coin_exchange : Prop):
  (n_students = 16) →
  (total_coins = 3360) →
  (equidistant_same) →
  (coin_exchange) →
  ∃ coins_in_center: ℕ, coins_in_center = 280 :=
by
  intros
  sorry

end central_student_coins_l1408_140859


namespace connie_grandma_birth_year_l1408_140875

theorem connie_grandma_birth_year :
  ∀ (B S G : ℕ),
  B = 1932 →
  S = 1936 →
  (S - B) * 2 = (S - G) →
  G = 1928 := 
by
  intros B S G hB hS hGap
  -- Proof goes here
  sorry

end connie_grandma_birth_year_l1408_140875


namespace max_value_of_y_l1408_140819

theorem max_value_of_y (x : ℝ) (h : x < 5/4) : 
  ∃ y : ℝ, y = 4 * x - 2 + 1 / (4 * x - 5) ∧ y ≤ 1 :=
sorry

end max_value_of_y_l1408_140819


namespace sum_of_three_consecutive_integers_is_21_l1408_140857

theorem sum_of_three_consecutive_integers_is_21 : 
  ∃ (n : ℤ), 3 * n = 21 :=
by
  sorry

end sum_of_three_consecutive_integers_is_21_l1408_140857


namespace percentage_reduction_in_price_of_oil_l1408_140878

theorem percentage_reduction_in_price_of_oil :
  ∀ P : ℝ, ∀ R : ℝ, P = 800 / (800 / R - 5) ∧ R = 40 →
  (P - R) / P * 100 = 25 := by
  -- Assumptions
  intros P R h
  have hP : P = 800 / (800 / R - 5) := h.1
  have hR : R = 40 := h.2
  -- Result to be proved
  sorry

end percentage_reduction_in_price_of_oil_l1408_140878


namespace num_values_of_n_l1408_140827

theorem num_values_of_n (a b c : ℕ) (h : 7 * a + 77 * b + 7777 * c = 8000) : 
  ∃ n : ℕ, (n = a + 2 * b + 4 * c) ∧ (110 * n ≤ 114300) ∧ ((8000 - 7 * a) % 70 = 7 * (10 * b + 111 * c) % 70) := 
sorry

end num_values_of_n_l1408_140827


namespace sequence_is_increasing_l1408_140821

def S (n : ℕ) : ℤ :=
  n^2 + 2 * n - 2

def a : ℕ → ℤ
| 0       => 0
| 1       => 1
| n + 1   => S (n + 1) - S n

theorem sequence_is_increasing : ∀ n m : ℕ, n < m → a n < a m :=
  sorry

end sequence_is_increasing_l1408_140821


namespace compute_f_g_f_l1408_140815

def f (x : ℤ) : ℤ := 2 * x + 4
def g (x : ℤ) : ℤ := 5 * x + 2

theorem compute_f_g_f (x : ℤ) : f (g (f 3)) = 108 := 
  by 
  sorry

end compute_f_g_f_l1408_140815


namespace goldfish_cost_discrete_points_l1408_140879

def goldfish_cost (n : ℕ) : ℝ :=
  0.25 * n + 5

theorem goldfish_cost_discrete_points :
  ∀ n : ℕ, 5 ≤ n ∧ n ≤ 20 → ∃ k : ℕ, goldfish_cost n = goldfish_cost k ∧ 5 ≤ k ∧ k ≤ 20 :=
by sorry

end goldfish_cost_discrete_points_l1408_140879


namespace find_m_for_split_l1408_140860

theorem find_m_for_split (m : ℕ) (h1 : m > 1) (h2 : ∃ k, k < m ∧ 2023 = (m^2 - m + 1) + 2*k) : m = 45 :=
sorry

end find_m_for_split_l1408_140860


namespace negation_of_P_equiv_l1408_140846

-- Define the proposition P
def P : Prop := ∀ x : ℝ, 2 * x^2 + 1 > 0

-- State the negation of P equivalently
theorem negation_of_P_equiv :
  ¬ P ↔ ∃ x : ℝ, 2 * x^2 + 1 ≤ 0 := 
sorry

end negation_of_P_equiv_l1408_140846


namespace simplify_expression_l1408_140833

variable (a b c d x y z : ℝ)

theorem simplify_expression :
  (cx * (b^2 * x^3 + 3 * a^2 * y^3 + c^2 * z^3) + dz * (a^2 * x^3 + 3 * c^2 * y^3 + b^2 * z^3)) / (cx + dz) =
  b^2 * x^3 + 3 * c^2 * y^3 + c^2 * z^3 :=
sorry

end simplify_expression_l1408_140833


namespace sanjay_homework_fraction_l1408_140892

theorem sanjay_homework_fraction (x : ℚ) :
  (2 * x + 1) / 3 + 4 / 15 = 1 ↔ x = 3 / 5 :=
by
  sorry

end sanjay_homework_fraction_l1408_140892


namespace possible_values_of_n_l1408_140899

-- Definitions for the problem
def side_ab (n : ℕ) := 3 * n + 3
def side_ac (n : ℕ) := 2 * n + 10
def side_bc (n : ℕ) := 2 * n + 16

-- Triangle inequality conditions
def triangle_inequality_1 (n : ℕ) : Prop := side_ab n + side_ac n > side_bc n
def triangle_inequality_2 (n : ℕ) : Prop := side_ab n + side_bc n > side_ac n
def triangle_inequality_3 (n : ℕ) : Prop := side_ac n + side_bc n > side_ab n

-- Angle condition simplified (since the more complex one was invalid)
def angle_condition (n : ℕ) : Prop := side_ac n > side_ab n

-- Combined valid n range
def valid_n_range (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 12

-- The theorem to prove
theorem possible_values_of_n (n : ℕ) : triangle_inequality_1 n ∧
                                        triangle_inequality_2 n ∧
                                        triangle_inequality_3 n ∧
                                        angle_condition n ↔
                                        valid_n_range n :=
by
  sorry

end possible_values_of_n_l1408_140899


namespace set_equality_implies_a_value_l1408_140886

theorem set_equality_implies_a_value (a : ℤ) : ({2, 3} : Set ℤ) = {2, 2 * a - 1} → a = 2 := 
by
  intro h
  sorry

end set_equality_implies_a_value_l1408_140886


namespace sum_of_tens_l1408_140891

theorem sum_of_tens (n : ℕ) (h : n = 100^10) : (n / 10) = 10^19 := by
  sorry

end sum_of_tens_l1408_140891


namespace find_specific_n_l1408_140877

theorem find_specific_n :
  ∀ (n : ℕ), (∃ (a b : ℤ), n^2 = a + b ∧ n^3 = a^2 + b^2) ↔ n = 0 ∨ n = 1 ∨ n = 2 :=
by {
  sorry
}

end find_specific_n_l1408_140877


namespace min_inverse_sum_l1408_140869

theorem min_inverse_sum (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : a + b = 4) : 1 ≤ (1/a) + (1/b) :=
by
  sorry

end min_inverse_sum_l1408_140869


namespace incorrect_statements_count_l1408_140888

-- Definitions of the statements
def statement1 : Prop := "The diameter perpendicular to the chord bisects the chord" = "incorrect"

def statement2 : Prop := "A circle is a symmetrical figure, and any diameter is its axis of symmetry" = "incorrect"

def statement3 : Prop := "Two arcs of equal length are congruent" = "incorrect"

-- Theorem stating that the number of incorrect statements is 3
theorem incorrect_statements_count : 
  (statement1 → False) → (statement2 → False) → (statement3 → False) → 3 = 3 :=
by sorry

end incorrect_statements_count_l1408_140888


namespace problem1_problem2_problem3_l1408_140805

-- Problem 1
theorem problem1 : -2.8 + (-3.6) + 3 - (-3.6) = 0.2 := 
by
  sorry

-- Problem 2
theorem problem2 : (-4) ^ 2010 * (-0.25) ^ 2009 + (-12) * (1 / 3 - 3 / 4 + 5 / 6) = -9 := 
by
  sorry

-- Problem 3
theorem problem3 : 13 * (16/60 : ℝ) * 5 - 19 * (12/60 : ℝ) / 6 = 13 * (8/60 : ℝ) + 50 := 
by
  sorry

end problem1_problem2_problem3_l1408_140805


namespace neg_cube_squared_l1408_140834

theorem neg_cube_squared (x : ℝ) : (-x^3) ^ 2 = x ^ 6 :=
by
  sorry

end neg_cube_squared_l1408_140834


namespace range_of_x_l1408_140822

noncomputable def f (x : ℝ) : ℝ := 2 * x + Real.sin x

theorem range_of_x (x : ℝ) (m : ℝ) (h : m ∈ Set.Icc (-2 : ℝ) 2) :
  f (m * x - 3) + f x < 0 → -3 < x ∧ x < 1 :=
sorry

end range_of_x_l1408_140822


namespace find_parallel_lines_l1408_140883

open Real

-- Definitions for the problem conditions
def line1 (a x y : ℝ) : Prop := x + 2 * a * y - 1 = 0
def line2 (a x y : ℝ) : Prop := (2 * a - 1) * x - a * y - 1 = 0

-- Definition of when two lines are parallel in ℝ²
def parallel (l1 l2 : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, (l1 x y → ∃ k, ∀ x' y', l2 x' y' → x = k * x' ∧ y = k * y')

-- Main theorem statement
theorem find_parallel_lines:
  ∀ a : ℝ, (parallel (line1 a) (line2 a)) → (a = 0 ∨ a = 1 / 4) :=
by sorry

end find_parallel_lines_l1408_140883


namespace symmetric_point_origin_l1408_140876

-- Define the coordinates of point A and the relation of symmetry about the origin
def A : ℝ × ℝ := (2, -1)
def symm_origin (P : ℝ × ℝ) : ℝ × ℝ := (-P.1, -P.2)

-- Theorem statement: Point B is the symmetric point of A about the origin
theorem symmetric_point_origin : symm_origin A = (-2, 1) :=
  sorry

end symmetric_point_origin_l1408_140876


namespace determine_x_l1408_140864

theorem determine_x (x : ℝ) :
  (x^2 - 6 * x + 8) / (x^2 - 9 * x + 14) = (x^2 - 8 * x + 15) / (x^2 - 10 * x + 24) →
  x = (13 + Real.sqrt 5) / 2 ∨ x = (13 - Real.sqrt 5) / 2 :=
by
  sorry

end determine_x_l1408_140864


namespace max_value_x_2y_2z_l1408_140814

theorem max_value_x_2y_2z (x y z : ℝ) (h : x^2 + y^2 + z^2 = 25) : x + 2*y + 2*z ≤ 15 :=
sorry

end max_value_x_2y_2z_l1408_140814


namespace number_of_zeros_of_f_l1408_140848

noncomputable def f (a x : ℝ) := x * Real.log x - a * x^2 - x

theorem number_of_zeros_of_f (a : ℝ) (h : |a| ≥ 1 / (2 * Real.exp 1)) :
  ∃! x, f a x = 0 :=
sorry

end number_of_zeros_of_f_l1408_140848


namespace area_large_square_l1408_140887

theorem area_large_square (a b c : ℝ) 
  (h1 : a^2 = b^2 + 32) 
  (h2 : 4*a = 4*c + 16) : a^2 = 100 := 
by {
  sorry
}

end area_large_square_l1408_140887


namespace jana_height_l1408_140816

theorem jana_height (jess_height : ℕ) (kelly_height : ℕ) (jana_height : ℕ) 
  (h1 : kelly_height = jess_height - 3) 
  (h2 : jana_height = kelly_height + 5) 
  (h3 : jess_height = 72) : 
  jana_height = 74 := 
by
  sorry

end jana_height_l1408_140816
