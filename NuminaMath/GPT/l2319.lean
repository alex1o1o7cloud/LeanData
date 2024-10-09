import Mathlib

namespace parabola_vertex_eq_l2319_231938

theorem parabola_vertex_eq : 
  ∃ (x y : ℝ), y = -3 * x^2 + 6 * x + 1 ∧ (x = 1) ∧ (y = 4) := 
by
  sorry

end parabola_vertex_eq_l2319_231938


namespace common_ratio_l2319_231915

theorem common_ratio (a S r : ℝ) (h1 : S = a / (1 - r))
  (h2 : ar^5 / (1 - r) = S / 81) : r = 1 / 3 :=
sorry

end common_ratio_l2319_231915


namespace percentage_equivalence_l2319_231943

theorem percentage_equivalence (x : ℝ) (h : 0.30 * 0.15 * x = 45) : 0.15 * 0.30 * x = 45 :=
sorry

end percentage_equivalence_l2319_231943


namespace sin_is_odd_l2319_231978

def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f x

theorem sin_is_odd : is_odd_function sin :=
by
  sorry

end sin_is_odd_l2319_231978


namespace allowance_calculation_l2319_231903

theorem allowance_calculation (A : ℝ)
  (h1 : (3 / 5) * A + (1 / 3) * (2 / 5) * A + 0.40 = A)
  : A = 1.50 :=
sorry

end allowance_calculation_l2319_231903


namespace pencils_pens_total_l2319_231934

theorem pencils_pens_total (x : ℕ) (h1 : 4 * x + 1 = 7 * (5 * x - 1)) : 4 * x + 5 * x = 45 :=
by
  sorry

end pencils_pens_total_l2319_231934


namespace polygon_sides_eq_14_l2319_231991

def num_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

theorem polygon_sides_eq_14 (n : ℕ) (h : n + num_diagonals n = 77) : n = 14 :=
by
  sorry

end polygon_sides_eq_14_l2319_231991


namespace arithmetic_sequence_sum_l2319_231965

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (S : ℕ → ℕ) 
  (h1 : ∀ n, n ≥ 2 → 2 * a n = a (n + 1) + a (n - 1))
  (h2 : S 3 = 6)
  (h3 : a 3 = 3) :
  S 2023 / 2023 = 1012 := by
  sorry

end arithmetic_sequence_sum_l2319_231965


namespace evaluateExpression_at_1_l2319_231970

noncomputable def evaluateExpression (x : ℝ) : ℝ :=
  (x^2 - 3 * x - 10) / (x - 5)

theorem evaluateExpression_at_1 : evaluateExpression 1 = 3 :=
by
  sorry

end evaluateExpression_at_1_l2319_231970


namespace intersection_A_B_l2319_231913

def A : Set ℝ := { x | ∃ y, y = Real.sqrt (x^2 - 2*x - 3) }
def B : Set ℝ := { x | ∃ y, y = Real.log x }

theorem intersection_A_B : A ∩ B = {x | x ∈ Set.Ici 3} :=
by
  sorry

end intersection_A_B_l2319_231913


namespace smaller_solution_l2319_231925

theorem smaller_solution (x : ℝ) (h : x^2 + 9 * x - 22 = 0) : x = -11 :=
sorry

end smaller_solution_l2319_231925


namespace f_increasing_f_odd_function_l2319_231952

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - 2 / (2^x + 1)

theorem f_increasing (a : ℝ) : ∀ (x1 x2 : ℝ), x1 < x2 → f a x1 < f a x2 :=
by
  sorry

theorem f_odd_function (a : ℝ) : f a 0 = 0 → (a = 1) :=
by
  sorry

end f_increasing_f_odd_function_l2319_231952


namespace sufficient_but_not_necessary_l2319_231920

theorem sufficient_but_not_necessary (x: ℝ) (hx: 0 < x ∧ x < 1) : 0 < x^2 ∧ x^2 < 1 ∧ (∀ y, 0 < y^2 ∧ y^2 < 1 → (y > 0 ∧ y < 1 ∨ y < 0 ∧ y > -1)) :=
by {
  sorry
}

end sufficient_but_not_necessary_l2319_231920


namespace problem_statement_l2319_231947

variable {f : ℝ → ℝ}
variable {a : ℝ}

def odd_function (f : ℝ → ℝ) :=
  ∀ x, f (-x) = -f x

def periodic_function (f : ℝ → ℝ) (p : ℝ) :=
  ∀ x, f (x + p) = f x

theorem problem_statement
  (h_odd : odd_function f)
  (h_periodic : periodic_function f 3)
  (h_f1 : f 1 < 1)
  (h_f2 : f 2 = a) :
  -1 < a ∧ a < 2 :=
sorry

end problem_statement_l2319_231947


namespace stockholm_to_malmo_road_distance_l2319_231928

-- Define constants based on the conditions
def map_distance_cm : ℕ := 120
def scale_factor : ℕ := 10
def road_distance_multiplier : ℚ := 1.15

-- Define the real distances based on the conditions
def straight_line_distance_km : ℕ :=
  map_distance_cm * scale_factor

def road_distance_km : ℚ :=
  straight_line_distance_km * road_distance_multiplier

-- Assert the final statement
theorem stockholm_to_malmo_road_distance :
  road_distance_km = 1380 := 
sorry

end stockholm_to_malmo_road_distance_l2319_231928


namespace miles_per_hour_l2319_231954

theorem miles_per_hour (total_distance : ℕ) (total_hours : ℕ) (h1 : total_distance = 81) (h2 : total_hours = 3) :
  total_distance / total_hours = 27 :=
by
  sorry

end miles_per_hour_l2319_231954


namespace find_m_values_l2319_231936

theorem find_m_values (m : ℕ) : (m - 3) ^ m = 1 ↔ m = 0 ∨ m = 2 ∨ m = 4 := sorry

end find_m_values_l2319_231936


namespace domain_of_f_x_minus_1_l2319_231977

theorem domain_of_f_x_minus_1 (f : ℝ → ℝ) (h : ∀ x, x^2 + 1 ∈ Set.Icc 1 10 → x ∈ Set.Icc (-3 : ℝ) 2) :
  Set.Icc 2 (11 : ℝ) ⊆ {x : ℝ | x - 1 ∈ Set.Icc 1 10} :=
by
  sorry

end domain_of_f_x_minus_1_l2319_231977


namespace tickets_needed_for_equal_distribution_l2319_231910

theorem tickets_needed_for_equal_distribution :
  ∃ k : ℕ, 865 + k ≡ 0 [MOD 9] ∧ k = 8 := sorry

end tickets_needed_for_equal_distribution_l2319_231910


namespace expected_red_light_l2319_231971

variables (n : ℕ) (p : ℝ)
def binomial_distribution : Type := sorry

noncomputable def expected_value (n : ℕ) (p : ℝ) : ℝ :=
n * p

theorem expected_red_light :
  expected_value 3 0.4 = 1.2 :=
by
  simp [expected_value]
  sorry

end expected_red_light_l2319_231971


namespace Mrs_Hilt_bought_two_cones_l2319_231968

def ice_cream_cone_cost : ℕ := 99
def total_spent : ℕ := 198

theorem Mrs_Hilt_bought_two_cones : total_spent / ice_cream_cone_cost = 2 :=
by
  sorry

end Mrs_Hilt_bought_two_cones_l2319_231968


namespace sin_390_eq_half_l2319_231902

theorem sin_390_eq_half : Real.sin (390 * Real.pi / 180) = 1 / 2 := by
  sorry

end sin_390_eq_half_l2319_231902


namespace fraction_of_menu_safely_eaten_l2319_231918

-- Given conditions
def VegetarianDishes := 6
def GlutenContainingVegetarianDishes := 5
def TotalDishes := 3 * VegetarianDishes

-- Derived information
def GlutenFreeVegetarianDishes := VegetarianDishes - GlutenContainingVegetarianDishes

-- Question: What fraction of the menu can Sarah safely eat?
theorem fraction_of_menu_safely_eaten : 
  (GlutenFreeVegetarianDishes / TotalDishes) = 1 / 18 :=
by
  sorry

end fraction_of_menu_safely_eaten_l2319_231918


namespace petya_purchase_cost_l2319_231953

theorem petya_purchase_cost (x : ℝ) 
  (h1 : ∃ shirt_cost : ℝ, x + shirt_cost = 2 * x)
  (h2 : ∃ boots_cost : ℝ, x + boots_cost = 5 * x)
  (h3 : ∃ shin_guards_cost : ℝ, x + shin_guards_cost = 3 * x) :
  ∃ total_cost : ℝ, total_cost = 8 * x :=
by 
  sorry

end petya_purchase_cost_l2319_231953


namespace total_pencils_crayons_l2319_231963

theorem total_pencils_crayons (r : ℕ) (p : ℕ) (c : ℕ) 
  (hp : p = 31) (hc : c = 27) (hr : r = 11) : 
  r * p + r * c = 638 := 
  by
  sorry

end total_pencils_crayons_l2319_231963


namespace line_parallel_through_point_l2319_231961

theorem line_parallel_through_point (P : ℝ × ℝ) (a b c : ℝ) (ha : a = 3) (hb : b = -4) (hc : c = 6) (hP : P = (4, -1)) :
  ∃ d : ℝ, (d = -16) ∧ (∀ x y : ℝ, a * x + b * y + d = 0 ↔ 3 * x - 4 * y - 16 = 0) :=
by
  sorry

end line_parallel_through_point_l2319_231961


namespace sum_of_pills_in_larger_bottles_l2319_231904

-- Definitions based on the conditions
def supplements := 5
def pills_in_small_bottles := 2 * 30
def pills_per_day := 5
def days_used := 14
def pills_remaining := 350
def total_pills_before := pills_remaining + (pills_per_day * days_used)
def total_pills_in_large_bottles := total_pills_before - pills_in_small_bottles

-- The theorem statement that needs to be proven
theorem sum_of_pills_in_larger_bottles : total_pills_in_large_bottles = 360 := 
by 
  -- Placeholder for the proof
  sorry

end sum_of_pills_in_larger_bottles_l2319_231904


namespace shorter_piece_length_l2319_231958

theorem shorter_piece_length (L : ℝ) (k : ℝ) (shorter_piece : ℝ) : 
  L = 28 ∧ k = 2.00001 / 5 ∧ L = shorter_piece + k * shorter_piece → 
  shorter_piece = 20 :=
by
  sorry

end shorter_piece_length_l2319_231958


namespace negation_of_p_l2319_231945

def proposition_p (n : ℕ) : Prop := 3^n ≥ n + 1

theorem negation_of_p : (∃ n0 : ℕ, 3^n0 < n0^2 + 1) :=
  by sorry

end negation_of_p_l2319_231945


namespace three_friends_at_least_50_mushrooms_l2319_231996

theorem three_friends_at_least_50_mushrooms (a : Fin 7 → ℕ) (h_sum : (Finset.univ.sum a) = 100) (h_different : Function.Injective a) :
  ∃ i j k : Fin 7, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ (a i + a j + a k) ≥ 50 :=
by
  sorry

end three_friends_at_least_50_mushrooms_l2319_231996


namespace janice_total_hours_worked_l2319_231957

-- Declare the conditions as definitions
def hourly_rate_first_40_hours : ℝ := 10
def hourly_rate_overtime : ℝ := 15
def first_40_hours : ℕ := 40
def total_pay : ℝ := 700

-- Define the main theorem
theorem janice_total_hours_worked (H : ℕ) (O : ℕ) : 
  H = first_40_hours + O ∧ (hourly_rate_first_40_hours * first_40_hours + hourly_rate_overtime * O = total_pay) → H = 60 :=
by
  sorry

end janice_total_hours_worked_l2319_231957


namespace pencils_to_sell_l2319_231927

/--
A store owner bought 1500 pencils at $0.10 each. 
Each pencil is sold for $0.25. 
He wants to make a profit of exactly $100. 
Prove that he must sell 1000 pencils to achieve this profit.
-/
theorem pencils_to_sell (total_pencils : ℕ) (cost_per_pencil : ℝ) (selling_price_per_pencil : ℝ) (desired_profit : ℝ)
  (h1 : total_pencils = 1500)
  (h2 : cost_per_pencil = 0.10)
  (h3 : selling_price_per_pencil = 0.25)
  (h4 : desired_profit = 100) :
  total_pencils * cost_per_pencil + desired_profit = 1000 * selling_price_per_pencil :=
by
  -- Since Lean code requires some proof content, we put sorry to skip it.
  sorry

end pencils_to_sell_l2319_231927


namespace find_b_plus_c_l2319_231922

theorem find_b_plus_c (a b c d : ℝ) 
    (h₁ : a + d = 6) 
    (h₂ : a * b + a * c + b * d + c * d = 40) : 
    b + c = 20 / 3 := 
sorry

end find_b_plus_c_l2319_231922


namespace minimum_reciprocal_sum_l2319_231987

theorem minimum_reciprocal_sum (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = 1) : 
  (∃ z : ℝ, (∀ x y : ℝ, 0 < x → 0 < y → x + 2 * y = 1 → z ≤ (1 / x + 2 / y)) ∧ z = 35 / 6) :=
  sorry

end minimum_reciprocal_sum_l2319_231987


namespace cosine_product_l2319_231984

-- Definitions for the conditions of the problem
variable (A B C D E : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E]
variables (circle : Set A) (inscribed_pentagon : Set A)
variables (AB BC CD DE AE : ℝ) (cosB cosACE : ℝ)

-- Conditions
axiom pentagon_inscribed_in_circle : inscribed_pentagon ⊆ circle
axiom AB_eq_3 : AB = 3
axiom BC_eq_3 : BC = 3
axiom CD_eq_3 : CD = 3
axiom DE_eq_3 : DE = 3
axiom AE_eq_2 : AE = 2

-- Theorem statement
theorem cosine_product :
  (1 - cosB) * (1 - cosACE) = (1 / 9) := 
sorry

end cosine_product_l2319_231984


namespace equation_solutions_35_implies_n_26_l2319_231946

theorem equation_solutions_35_implies_n_26 (n : ℕ) (h3x3y2z_eq_n : ∃ (s : Finset (ℕ × ℕ × ℕ)), (∀ t ∈ s, ∃ (x y z : ℕ), 
  t = (x, y, z) ∧ 3 * x + 3 * y + 2 * z = n ∧ x > 0 ∧ y > 0 ∧ z > 0) ∧ s.card = 35) : n = 26 := 
sorry

end equation_solutions_35_implies_n_26_l2319_231946


namespace paul_lost_crayons_l2319_231941

theorem paul_lost_crayons :
  let total := 229
  let given_away := 213
  let lost := total - given_away
  lost = 16 :=
by
  sorry

end paul_lost_crayons_l2319_231941


namespace mike_taller_than_mark_l2319_231969

-- Define the heights of Mark and Mike in terms of feet and inches
def mark_height_feet : ℕ := 5
def mark_height_inches : ℕ := 3
def mike_height_feet : ℕ := 6
def mike_height_inches : ℕ := 1

-- Define the conversion factor from feet to inches
def feet_to_inches : ℕ := 12

-- Conversion of heights to inches
def mark_total_height_in_inches : ℕ := mark_height_feet * feet_to_inches + mark_height_inches
def mike_total_height_in_inches : ℕ := mike_height_feet * feet_to_inches + mike_height_inches

-- Define the problem statement: proving Mike is 10 inches taller than Mark
theorem mike_taller_than_mark : mike_total_height_in_inches - mark_total_height_in_inches = 10 :=
by sorry

end mike_taller_than_mark_l2319_231969


namespace find_values_l2319_231960

theorem find_values (a b : ℝ) 
  (h1 : a + b = 10)
  (h2 : a - b = 4) 
  (h3 : a^2 + b^2 = 58) : 
  a^2 - b^2 = 40 ∧ ab = 21 := 
by 
  sorry

end find_values_l2319_231960


namespace total_apples_picked_l2319_231948

def Mike_apples : ℕ := 7
def Nancy_apples : ℕ := 3
def Keith_apples : ℕ := 6
def Jennifer_apples : ℕ := 5
def Tom_apples : ℕ := 8
def Stacy_apples : ℕ := 4

theorem total_apples_picked : 
  Mike_apples + Nancy_apples + Keith_apples + Jennifer_apples + Tom_apples + Stacy_apples = 33 :=
by
  sorry

end total_apples_picked_l2319_231948


namespace product_identity_l2319_231967

variable (x y : ℝ)

theorem product_identity :
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 :=
by
  sorry

end product_identity_l2319_231967


namespace sum_lent_is_10000_l2319_231905

theorem sum_lent_is_10000
  (P : ℝ)
  (r : ℝ := 0.075)
  (t : ℝ := 7)
  (I : ℝ := P - 4750) 
  (H1 : I = P * r * t) :
  P = 10000 :=
sorry

end sum_lent_is_10000_l2319_231905


namespace gcd_a2_14a_49_a_7_l2319_231982

theorem gcd_a2_14a_49_a_7 (a : ℤ) (k : ℤ) (h : a = 2100 * k) :
  Int.gcd (a^2 + 14*a + 49) (a + 7) = 7 := 
by
  sorry

end gcd_a2_14a_49_a_7_l2319_231982


namespace at_least_one_ge_two_l2319_231914

theorem at_least_one_ge_two (a b c : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) :
  a + 1 / b ≥ 2 ∨ b + 1 / c ≥ 2 ∨ c + 1 / a ≥ 2 := 
sorry

end at_least_one_ge_two_l2319_231914


namespace plane_speed_ratio_train_l2319_231901

def distance (speed time : ℝ) := speed * time

theorem plane_speed_ratio_train (x y z : ℝ)
  (h_train : distance x 20 = distance y 10)
  (h_wait_time : z > 5)
  (h_plane_meet_train : distance y (8/9) = distance x (z + 8/9)) :
  y = 10 * x :=
by {
  sorry
}

end plane_speed_ratio_train_l2319_231901


namespace log_relationship_l2319_231979

theorem log_relationship (a b : ℝ) (x : ℝ) (h₁ : 6 * (Real.log (x) / Real.log (a)) ^ 2 + 5 * (Real.log (x) / Real.log (b)) ^ 2 = 12 * (Real.log (x) ^ 2) / (Real.log (a) * Real.log (b))) :
  a = b^(5/3) ∨ a = b^(3/5) := by
  sorry

end log_relationship_l2319_231979


namespace coffee_table_price_l2319_231950

theorem coffee_table_price :
  let sofa := 1250
  let armchairs := 2 * 425
  let rug := 350
  let bookshelf := 200
  let subtotal_without_coffee_table := sofa + armchairs + rug + bookshelf
  let C := 429.24
  let total_before_discount_and_tax := subtotal_without_coffee_table + C
  let discounted_total := total_before_discount_and_tax * 0.90
  let final_invoice_amount := discounted_total * 1.06
  final_invoice_amount = 2937.60 :=
by
  sorry

end coffee_table_price_l2319_231950


namespace factory_toys_production_each_day_l2319_231994

theorem factory_toys_production_each_day 
  (weekly_production : ℕ)
  (days_worked_per_week : ℕ)
  (h1 : weekly_production = 4560)
  (h2 : days_worked_per_week = 4) : 
  (weekly_production / days_worked_per_week) = 1140 :=
  sorry

end factory_toys_production_each_day_l2319_231994


namespace least_homeowners_l2319_231926

theorem least_homeowners (M W : ℕ) (total_members : M + W = 150)
  (men_homeowners : ∃ n : ℕ, n = 10 * M / 100) 
  (women_homeowners : ∃ n : ℕ, n = 20 * W / 100) : 
  ∃ homeowners : ℕ, homeowners = 16 := 
sorry

end least_homeowners_l2319_231926


namespace even_decreasing_function_l2319_231983

noncomputable def f : ℝ → ℝ := sorry

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def is_decreasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x ≤ y → f y ≤ f x

theorem even_decreasing_function :
  is_even f →
  is_decreasing_on_nonneg f →
  f 1 > f (-2) ∧ f (-2) > f 3 :=
by
  sorry

end even_decreasing_function_l2319_231983


namespace wickets_before_last_match_l2319_231999

theorem wickets_before_last_match (R W : ℕ) 
  (initial_average : ℝ) (runs_last_match wickets_last_match : ℕ) (average_decrease : ℝ)
  (h_initial_avg : initial_average = 12.4)
  (h_last_match_runs : runs_last_match = 26)
  (h_last_match_wickets : wickets_last_match = 5)
  (h_avg_decrease : average_decrease = 0.4)
  (h_initial_runs_eq : R = initial_average * W)
  (h_new_average : (R + runs_last_match) / (W + wickets_last_match) = initial_average - average_decrease) :
  W = 85 :=
by
  sorry

end wickets_before_last_match_l2319_231999


namespace five_hash_neg_one_l2319_231911

def hash (x y : ℤ) : ℤ := x * (y + 2) + x * y

theorem five_hash_neg_one : hash 5 (-1) = 0 :=
by
  sorry

end five_hash_neg_one_l2319_231911


namespace ratio_long_side_brush_width_l2319_231997

theorem ratio_long_side_brush_width 
  (l : ℝ) (w : ℝ) (d : ℝ) (total_area : ℝ) (painted_area : ℝ) (b : ℝ) 
  (h1 : l = 9)
  (h2 : w = 4)
  (h3 : total_area = l * w)
  (h4 : total_area / 3 = painted_area)
  (h5 : d = Real.sqrt (l^2 + w^2))
  (h6 : d * b = painted_area) :
  l / b = (3 * Real.sqrt 97) / 4 :=
by
  sorry

end ratio_long_side_brush_width_l2319_231997


namespace volunteer_selection_count_l2319_231962

open Nat

theorem volunteer_selection_count :
  let boys : ℕ := 5
  let girls : ℕ := 2
  let total_ways := choose girls 1 * choose boys 2 + choose girls 2 * choose boys 1
  total_ways = 25 :=
by
  sorry

end volunteer_selection_count_l2319_231962


namespace tom_fractions_l2319_231912

theorem tom_fractions (packages : ℕ) (cars_per_package : ℕ) (cars_left : ℕ) (nephews : ℕ) :
  packages = 10 → 
  cars_per_package = 5 → 
  cars_left = 30 → 
  nephews = 2 → 
  ∃ fraction_given : ℚ, fraction_given = 1/5 :=
by
  intros
  sorry

end tom_fractions_l2319_231912


namespace difference_of_squares_l2319_231988

theorem difference_of_squares (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 8) : x^2 - y^2 = 160 :=
sorry

end difference_of_squares_l2319_231988


namespace ratio_twelfth_term_geometric_sequence_l2319_231906

theorem ratio_twelfth_term_geometric_sequence (G H : ℕ → ℝ) (n : ℕ) (a r b s : ℝ)
  (hG : ∀ n, G n = a * (r^n - 1) / (r - 1))
  (hH : ∀ n, H n = b * (s^n - 1) / (s - 1))
  (ratio_condition : ∀ n, G n / H n = (5 * n + 3) / (3 * n + 17)) :
  (a * r^11) / (b * s^11) = 2 / 5 :=
by 
  sorry

end ratio_twelfth_term_geometric_sequence_l2319_231906


namespace minimum_value_a_plus_2b_l2319_231966

theorem minimum_value_a_plus_2b {a b : ℝ} (ha : a > 0) (hb : b > 0) (h : 2 * a + b - a * b = 0) : a + 2 * b = 9 :=
by sorry

end minimum_value_a_plus_2b_l2319_231966


namespace percent_unionized_men_is_70_l2319_231980

open Real

def total_employees : ℝ := 100
def percent_men : ℝ := 0.5
def percent_unionized : ℝ := 0.6
def percent_women_nonunion : ℝ := 0.8
def percent_men_nonunion : ℝ := 0.2

def num_men := total_employees * percent_men
def num_unionized := total_employees * percent_unionized
def num_nonunion := total_employees - num_unionized
def num_men_nonunion := num_nonunion * percent_men_nonunion
def num_men_unionized := num_men - num_men_nonunion

theorem percent_unionized_men_is_70 :
  (num_men_unionized / num_unionized) * 100 = 70 := by
  sorry

end percent_unionized_men_is_70_l2319_231980


namespace tan_at_max_value_l2319_231931

theorem tan_at_max_value : 
  ∃ x₀, (∀ x, 3 * Real.sin x₀ - 4 * Real.cos x₀ ≥ 3 * Real.sin x - 4 * Real.cos x) → Real.tan x₀ = 3/4 := 
sorry

end tan_at_max_value_l2319_231931


namespace wall_length_to_height_ratio_l2319_231900

theorem wall_length_to_height_ratio
  (W H L : ℝ)
  (V : ℝ)
  (h1 : H = 6 * W)
  (h2 : L * H * W = V)
  (h3 : V = 86436)
  (h4 : W = 6.999999999999999) :
  L / H = 7 :=
by
  sorry

end wall_length_to_height_ratio_l2319_231900


namespace magician_identifies_card_l2319_231933

def Grid : Type := Fin 6 → Fin 6 → Nat

def choose_card (g : Grid) (c : Fin 6) (r : Fin 6) : Nat := g r c

def rearrange_columns_to_rows (s : List Nat) : Grid :=
  λ r c => s.get! (r.val * 6 + c.val)

theorem magician_identifies_card (g : Grid) (c1 : Fin 6) (r2 : Fin 6) :
  ∃ (card : Nat), (choose_card g c1 r2 = card) :=
  sorry

end magician_identifies_card_l2319_231933


namespace num_k_values_lcm_l2319_231998

-- Define prime factorizations of given numbers
def nine_pow_nine := 3^18
def twelve_pow_twelve := 2^24 * 3^12
def eighteen_pow_eighteen := 2^18 * 3^36

-- Number of values of k making eighteen_pow_eighteen the LCM of nine_pow_nine, twelve_pow_twelve, and k
def number_of_k_values : ℕ := 
  19 -- Based on calculations from the proof

theorem num_k_values_lcm :
  ∀ (k : ℕ), eighteen_pow_eighteen = Nat.lcm (Nat.lcm nine_pow_nine twelve_pow_twelve) k → ∃ n, n = number_of_k_values :=
  sorry -- Add the proof later

end num_k_values_lcm_l2319_231998


namespace shorter_piece_length_l2319_231986

theorem shorter_piece_length :
  ∃ (x : ℝ), x + 2 * x = 69 ∧ x = 23 :=
by
  sorry

end shorter_piece_length_l2319_231986


namespace value_at_2_l2319_231959

def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 2 * x

theorem value_at_2 : f 2 = 0 := by
  sorry

end value_at_2_l2319_231959


namespace inequality_proof_l2319_231919

variable {a b c : ℝ}

theorem inequality_proof (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h : a * b * c = 1) : 
  (1 / (a^2 * (b + c))) + (1 / (b^2 * (c + a))) + (1 / (c^2 * (a + b))) ≥ 3 / 2 :=
sorry

end inequality_proof_l2319_231919


namespace arithmetic_sequence_general_term_absolute_sum_first_19_terms_l2319_231937

theorem arithmetic_sequence_general_term (a : ℕ → ℤ) (h1 : ∀ n : ℕ, n > 0 → 2 * a (n + 1) = a n + a (n + 2))
  (h2 : a 1 + a 4 = 41) (h3 : a 3 + a 7 = 26) :
  ∀ n : ℕ, a n = 28 - 3 * n := 
sorry

theorem absolute_sum_first_19_terms (a : ℕ → ℤ) (h1 : ∀ n : ℕ, n > 0 → 2 * a (n + 1) = a n + a (n + 2))
  (h2 : a 1 + a 4 = 41) (h3 : a 3 + a 7 = 26) (an_eq : ∀ n : ℕ, a n = 28 - 3 * n) :
  |a 1| + |a 3| + |a 5| + |a 7| + |a 9| + |a 11| + |a 13| + |a 15| + |a 17| + |a 19| = 150 := 
sorry

end arithmetic_sequence_general_term_absolute_sum_first_19_terms_l2319_231937


namespace minimum_fraction_l2319_231975

theorem minimum_fraction (m n : ℕ) (hm : m > 0) (hn : n > 0) (h : m + 2 * n = 8) : 2 / m + 1 / n = 1 :=
by
  sorry

end minimum_fraction_l2319_231975


namespace simplify_cube_root_21952000_l2319_231976

noncomputable def cube_root (x : ℝ) : ℝ := x^(1/3)

theorem simplify_cube_root_21952000 : 
  cube_root 21952000 = 280 := 
by {
  sorry
}

end simplify_cube_root_21952000_l2319_231976


namespace number_of_allowed_pairs_l2319_231944

theorem number_of_allowed_pairs (total_books : ℕ) (prohibited_books : ℕ) : ℕ :=
  let total_pairs := (total_books * (total_books - 1)) / 2
  let prohibited_pairs := (prohibited_books * (prohibited_books - 1)) / 2
  total_pairs - prohibited_pairs

example : number_of_allowed_pairs 15 3 = 102 :=
by
  sorry

end number_of_allowed_pairs_l2319_231944


namespace nancy_threw_out_2_carrots_l2319_231990

theorem nancy_threw_out_2_carrots :
  ∀ (x : ℕ), 12 - x + 21 = 31 → x = 2 :=
by
  sorry

end nancy_threw_out_2_carrots_l2319_231990


namespace mom_younger_than_grandmom_l2319_231929

def cara_age : ℕ := 40
def cara_younger_mom : ℕ := 20
def grandmom_age : ℕ := 75

def mom_age : ℕ := cara_age + cara_younger_mom
def age_difference : ℕ := grandmom_age - mom_age

theorem mom_younger_than_grandmom : age_difference = 15 := by
  sorry

end mom_younger_than_grandmom_l2319_231929


namespace combination_5_3_eq_10_l2319_231992

-- Define the combination function according to its formula
noncomputable def combination (n k : ℕ) : ℕ :=
  (n.factorial) / (k.factorial * (n - k).factorial)

-- Theorem stating the required result
theorem combination_5_3_eq_10 : combination 5 3 = 10 := by
  sorry

end combination_5_3_eq_10_l2319_231992


namespace max_integer_is_twelve_l2319_231923

theorem max_integer_is_twelve
  (a b c d e : ℕ)
  (h1 : a < b)
  (h2 : b < c)
  (h3 : c < d)
  (h4 : d < e)
  (h5 : (a + b + c + d + e) / 5 = 9)
  (h6 : ((a - 9)^2 + (b - 9)^2 + (c - 9)^2 + (d - 9)^2 + (e - 9)^2) / 5 = 4) :
  e = 12 := sorry

end max_integer_is_twelve_l2319_231923


namespace strap_pieces_l2319_231995

/-
  Given the conditions:
  1. The sum of the lengths of the two straps is 64 cm.
  2. The longer strap is 48 cm longer than the shorter strap.
  
  Prove that the number of pieces of strap that equal the length of the shorter strap 
  that can be cut from the longer strap is 7.
-/

theorem strap_pieces (S L : ℕ) (h1 : S + L = 64) (h2 : L = S + 48) :
  L / S = 7 :=
by
  sorry

end strap_pieces_l2319_231995


namespace average_temperature_correct_l2319_231964

theorem average_temperature_correct (W T : ℝ) :
  (38 + W + T) / 3 = 32 →
  44 = 44 →
  38 = 38 →
  (W + T + 44) / 3 = 34 :=
by
  intros h1 h2 h3
  sorry

end average_temperature_correct_l2319_231964


namespace average_temperature_l2319_231917

def temperature_NY := 80
def temperature_MIA := temperature_NY + 10
def temperature_SD := temperature_MIA + 25

theorem average_temperature :
  (temperature_NY + temperature_MIA + temperature_SD) / 3 = 95 := 
sorry

end average_temperature_l2319_231917


namespace exactly_one_solves_problem_l2319_231951

theorem exactly_one_solves_problem (pA pB pC : ℝ) (hA : pA = 1 / 2) (hB : pB = 1 / 3) (hC : pC = 1 / 4) :
  (pA * (1 - pB) * (1 - pC) + (1 - pA) * pB * (1 - pC) + (1 - pA) * (1 - pB) * pC) = 11 / 24 :=
by
  sorry

end exactly_one_solves_problem_l2319_231951


namespace simplify_expression_l2319_231955

theorem simplify_expression :
  (625: ℝ)^(1/4) * (256: ℝ)^(1/3) = 20 := 
sorry

end simplify_expression_l2319_231955


namespace total_amount_received_is_1465_l2319_231993

-- defining the conditions
def principal_1 : ℝ := 4000
def principal_2 : ℝ := 8200
def rate_1 : ℝ := 0.11
def rate_2 : ℝ := rate_1 + 0.015

-- defining the interest from each account
def interest_1 := principal_1 * rate_1
def interest_2 := principal_2 * rate_2

-- stating the total amount received
def total_received := interest_1 + interest_2

-- proving the total amount received
theorem total_amount_received_is_1465 : total_received = 1465 := by
  -- proof goes here
  sorry

end total_amount_received_is_1465_l2319_231993


namespace distinct_arrangements_of_PHONE_l2319_231935

-- Condition: The word PHONE consists of 5 distinct letters
def distinctLetters := 5

-- Theorem: The number of distinct arrangements of the letters in the word PHONE
theorem distinct_arrangements_of_PHONE : Nat.factorial distinctLetters = 120 := sorry

end distinct_arrangements_of_PHONE_l2319_231935


namespace shaded_region_is_hyperbolas_l2319_231908

theorem shaded_region_is_hyperbolas (T : ℝ) (hT : T > 0) :
  (∃ (x y : ℝ), x * y = T / 4) ∧ (∃ (x y : ℝ), x * y = - (T / 4)) :=
by
  sorry

end shaded_region_is_hyperbolas_l2319_231908


namespace factor_by_resultant_l2319_231907

theorem factor_by_resultant (x f : ℤ) (h1 : x = 17) (h2 : (2 * x + 5) * f = 117) : f = 3 := 
by
  sorry

end factor_by_resultant_l2319_231907


namespace min_vertical_segment_length_l2319_231973

noncomputable def f₁ (x : ℝ) : ℝ := |x|
noncomputable def f₂ (x : ℝ) : ℝ := -x^2 - 4 * x - 3

theorem min_vertical_segment_length :
  ∃ m : ℝ, m = 3 ∧
            ∀ x : ℝ, abs (f₁ x - f₂ x) ≥ m :=
sorry

end min_vertical_segment_length_l2319_231973


namespace team_overall_progress_is_89_l2319_231974

def yard_changes : List Int := [-5, 9, -12, 17, -15, 24, -7]

def overall_progress (changes : List Int) : Int :=
  changes.sum

theorem team_overall_progress_is_89 :
  overall_progress yard_changes = 89 :=
by
  sorry

end team_overall_progress_is_89_l2319_231974


namespace mr_brown_no_calls_in_2020_l2319_231942

noncomputable def number_of_days_with_no_calls (total_days : ℕ) (calls_niece1 : ℕ) (calls_niece2 : ℕ) (calls_niece3 : ℕ) : ℕ := 
  let calls_2 := total_days / calls_niece1
  let calls_3 := total_days / calls_niece2
  let calls_4 := total_days / calls_niece3
  let calls_6 := total_days / (Nat.lcm calls_niece1 calls_niece2)
  let calls_12_ := total_days / (Nat.lcm calls_niece1 (Nat.lcm calls_niece2 calls_niece3))
  total_days - (calls_2 + calls_3 + calls_4 - calls_6 - calls_4 - (total_days / calls_niece2 / 4) + calls_12_)

theorem mr_brown_no_calls_in_2020 : number_of_days_with_no_calls 365 2 3 4 = 122 := 
  by 
    -- Proof steps would go here
    sorry

end mr_brown_no_calls_in_2020_l2319_231942


namespace find_satisfying_pairs_l2319_231939

theorem find_satisfying_pairs (n p : ℕ) (prime_p : Nat.Prime p) :
  n ≤ 2 * p ∧ (p - 1)^n + 1 ≡ 0 [MOD n^2] →
  (n = 1 ∧ Nat.Prime p) ∨ (n = 2 ∧ p = 2) ∨ (n = 3 ∧ p = 3) :=
by sorry

end find_satisfying_pairs_l2319_231939


namespace exists_divisible_by_2021_l2319_231949

def concat_numbers (n m : ℕ) : ℕ :=
  -- function to concatenate numbers from n to m
  sorry

theorem exists_divisible_by_2021 :
  ∃ (n m : ℕ), n > m ∧ m ≥ 1 ∧ 2021 ∣ concat_numbers n m :=
by
  sorry

end exists_divisible_by_2021_l2319_231949


namespace hyperbola_equation_l2319_231909

theorem hyperbola_equation (h1 : ∀ x y : ℝ, (x = 0 ∧ y = 0)) 
                           (h2 : ∀ a : ℝ, (2 * a = 4)) 
                           (h3 : ∀ c : ℝ, (c = 3)) : 
  ∃ b : ℝ, (b^2 = 5) ∧ (∀ x y : ℝ, (y^2 / 4) - (x^2 / b^2) = 1) :=
sorry

end hyperbola_equation_l2319_231909


namespace div_by_1963_iff_odd_l2319_231940

-- Define the given condition and statement
theorem div_by_1963_iff_odd (n : ℕ) :
  (1963 ∣ (82^n + 454 * 69^n)) ↔ (n % 2 = 1) :=
sorry

end div_by_1963_iff_odd_l2319_231940


namespace line_intersects_circle_l2319_231985

theorem line_intersects_circle (k : ℝ) :
  ∃ x y : ℝ, (kx - y - k +1 = 0) ∧ (x^2 + y^2 = 4) :=
sorry

end line_intersects_circle_l2319_231985


namespace find_number_l2319_231924

theorem find_number :
  let s := 2615 + 3895
  let d := 3895 - 2615
  let q := 3 * d
  let x := s * q + 65
  x = 24998465 :=
by
  let s := 2615 + 3895
  let d := 3895 - 2615
  let q := 3 * d
  let x := s * q + 65
  sorry

end find_number_l2319_231924


namespace running_time_constant_pace_l2319_231921

/-!
# Running Time Problem

We are given that the running pace is constant, it takes 30 minutes to run 5 miles,
and we need to find out how long it will take to run 2.5 miles.
-/

theorem running_time_constant_pace :
  ∀ (distance_to_store distance_to_cousin distance_run time_run : ℝ)
  (constant_pace : Prop),
  distance_to_store = 5 → time_run = 30 → distance_to_cousin = 2.5 →
  constant_pace → 
  time_run / distance_to_store * distance_to_cousin = 15 :=
by 
  intros distance_to_store distance_to_cousin distance_run time_run constant_pace 
         hds htr hdc hcp
  rw [hds, htr, hdc]
  exact sorry

end running_time_constant_pace_l2319_231921


namespace profit_at_15_is_correct_l2319_231981

noncomputable def profit (x : ℝ) : ℝ := (2 * x - 20) * (40 - x)

theorem profit_at_15_is_correct :
  profit 15 = 1250 := by
  sorry

end profit_at_15_is_correct_l2319_231981


namespace sum_ab_eq_negative_two_l2319_231916

def f (x : ℝ) := x^3 + 3 * x^2 + 6 * x + 4

theorem sum_ab_eq_negative_two (a b : ℝ) (h1 : f a = 14) (h2 : f b = -14) : a + b = -2 := 
by 
  sorry

end sum_ab_eq_negative_two_l2319_231916


namespace elizabeth_bananas_eaten_l2319_231972

theorem elizabeth_bananas_eaten (initial_bananas remaining_bananas eaten_bananas : ℕ) 
    (h1 : initial_bananas = 12) 
    (h2 : remaining_bananas = 8) 
    (h3 : eaten_bananas = initial_bananas - remaining_bananas) :
    eaten_bananas = 4 := 
sorry

end elizabeth_bananas_eaten_l2319_231972


namespace total_weekly_airflow_l2319_231930

-- Definitions from conditions
def fanA_airflow : ℝ := 10  -- liters per second
def fanA_time_per_day : ℝ := 10 * 60  -- converted to seconds (10 minutes * 60 seconds/minute)

def fanB_airflow : ℝ := 15  -- liters per second
def fanB_time_per_day : ℝ := 20 * 60  -- converted to seconds (20 minutes * 60 seconds/minute)

def fanC_airflow : ℝ := 25  -- liters per second
def fanC_time_per_day : ℝ := 30 * 60  -- converted to seconds (30 minutes * 60 seconds/minute)

def days_in_week : ℝ := 7

-- Theorem statement to be proven
theorem total_weekly_airflow : fanA_airflow * fanA_time_per_day * days_in_week +
                               fanB_airflow * fanB_time_per_day * days_in_week +
                               fanC_airflow * fanC_time_per_day * days_in_week = 483000 := 
by
  -- skip the proof
  sorry

end total_weekly_airflow_l2319_231930


namespace fixed_point_PQ_passes_l2319_231956

theorem fixed_point_PQ_passes (P Q : ℝ × ℝ) (x1 x2 : ℝ)
  (hP : P = (x1, x1^2))
  (hQ : Q = (x2, x2^2))
  (hC1 : x1 ≠ 0)
  (hC2 : x2 ≠ 0)
  (hSlopes : (x2 / x2^2 * (2 * x1)) = -2) :
  ∃ D : ℝ × ℝ, D = (0, 1) ∧
    ∀ (x y : ℝ), (y = x1^2 + (x1 - (1 / x1)) * (x - x1)) → ((x, y) = P ∨ (x, y) = Q) := sorry

end fixed_point_PQ_passes_l2319_231956


namespace count_squares_below_graph_l2319_231932

theorem count_squares_below_graph (x y: ℕ) (h_eq : 12 * x + 180 * y = 2160) (h_first_quadrant : x ≥ 0 ∧ y ≥ 0) :
  let total_squares := 180 * 12
  let diagonal_squares := 191
  let below_squares := total_squares - diagonal_squares
  below_squares = 1969 :=
by
  sorry

end count_squares_below_graph_l2319_231932


namespace critical_temperature_of_water_l2319_231989

/--
Given the following conditions:
1. The temperature at which solid, liquid, and gaseous water coexist is the triple point.
2. The temperature at which water vapor condenses is the condensation point.
3. The maximum temperature at which liquid water can exist.
4. The minimum temperature at which water vapor can exist.

Prove that the critical temperature of water is the maximum temperature at which liquid water can exist.
-/
theorem critical_temperature_of_water :
    ∀ (triple_point condensation_point maximum_liquid_temp minimum_vapor_temp critical_temp : ℝ), 
    (critical_temp = maximum_liquid_temp) ↔
    ((critical_temp ≠ triple_point) ∧ (critical_temp ≠ condensation_point) ∧ (critical_temp ≠ minimum_vapor_temp)) := 
  sorry

end critical_temperature_of_water_l2319_231989
