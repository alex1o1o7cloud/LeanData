import Mathlib

namespace pastries_sold_l749_74943

def initial_pastries : ℕ := 148
def pastries_left : ℕ := 45

theorem pastries_sold : initial_pastries - pastries_left = 103 := by
  sorry

end pastries_sold_l749_74943


namespace price_per_glass_second_day_l749_74910

theorem price_per_glass_second_day (O : ℝ) (P : ℝ) 
  (V1 : ℝ := 2 * O) -- Volume on the first day
  (V2 : ℝ := 3 * O) -- Volume on the second day
  (price_first_day : ℝ := 0.30) -- Price per glass on the first day
  (revenue_equal : V1 * price_first_day = V2 * P) :
  P = 0.20 := 
by
  -- skipping the proof
  sorry

end price_per_glass_second_day_l749_74910


namespace total_canvas_area_l749_74985

theorem total_canvas_area (rect_length rect_width tri1_base tri1_height tri2_base tri2_height : ℕ)
    (h1 : rect_length = 5) (h2 : rect_width = 8)
    (h3 : tri1_base = 3) (h4 : tri1_height = 4)
    (h5 : tri2_base = 4) (h6 : tri2_height = 6) :
    (rect_length * rect_width) + ((tri1_base * tri1_height) / 2) + ((tri2_base * tri2_height) / 2) = 58 := by
  sorry

end total_canvas_area_l749_74985


namespace max_electronic_thermometers_l749_74961

theorem max_electronic_thermometers :
  ∀ (x : ℕ), 10 * x + 3 * (53 - x) ≤ 300 → x ≤ 20 :=
by
  sorry

end max_electronic_thermometers_l749_74961


namespace comb_sum_C8_2_C8_3_l749_74929

open Nat

theorem comb_sum_C8_2_C8_3 : (Nat.choose 8 2) + (Nat.choose 8 3) = 84 :=
by
  sorry

end comb_sum_C8_2_C8_3_l749_74929


namespace average_of_ratios_l749_74930

theorem average_of_ratios (a b c : ℕ) (h1 : 2 * b = 3 * a) (h2 : 3 * c = 4 * a) (h3 : a = 28) : (a + b + c) / 3 = 42 := by
  -- skipping the proof
  sorry

end average_of_ratios_l749_74930


namespace remainder_n_plus_2023_mod_7_l749_74949

theorem remainder_n_plus_2023_mod_7 (n : ℤ) (h : n % 7 = 2) : (n + 2023) % 7 = 2 :=
by
  sorry

end remainder_n_plus_2023_mod_7_l749_74949


namespace sum_of_fractions_l749_74916

theorem sum_of_fractions : (1 / 6) + (2 / 9) + (1 / 3) = 13 / 18 := by
  sorry

end sum_of_fractions_l749_74916


namespace nesbitt_inequality_l749_74992

theorem nesbitt_inequality (a b c : ℝ) (h_pos1 : 0 < a) (h_pos2 : 0 < b) (h_pos3 : 0 < c) (h_abc: a * b * c = 1) :
  1 / (1 + 2 * a) + 1 / (1 + 2 * b) + 1 / (1 + 2 * c) ≥ 1 :=
sorry

end nesbitt_inequality_l749_74992


namespace find_b_of_perpendicular_bisector_l749_74954

theorem find_b_of_perpendicular_bisector :
  (∃ b : ℝ, (∀ x y : ℝ, x + y = b → x + y = 4 + 6)) → b = 10 :=
by
  sorry

end find_b_of_perpendicular_bisector_l749_74954


namespace find_natural_number_A_l749_74950

theorem find_natural_number_A (A : ℕ) : 
  (A * 1000 ≤ (A * (A + 1)) / 2 ∧ (A * (A + 1)) / 2 ≤ A * 1000 + 999) → A = 1999 :=
by
  sorry

end find_natural_number_A_l749_74950


namespace x_plus_2y_equals_2_l749_74981

theorem x_plus_2y_equals_2 (x y : ℝ) (h : |x + 3| + (2 * y - 5)^2 = 0) : x + 2 * y = 2 := 
sorry

end x_plus_2y_equals_2_l749_74981


namespace distance_between_first_and_last_tree_l749_74911

theorem distance_between_first_and_last_tree (n : ℕ) (d : ℕ) 
  (h1 : n = 10) 
  (h2 : d = 100) 
  (h3 : d / 5 = 20) :
  (20 * 9 = 180) :=
by
  sorry

end distance_between_first_and_last_tree_l749_74911


namespace consumption_increase_percentage_l749_74900

theorem consumption_increase_percentage
  (T C : ℝ)
  (H1 : 0.90 * (1 + X/100) = 0.9999999999999858) :
  X = 11.11111111110953 :=
by
  sorry

end consumption_increase_percentage_l749_74900


namespace min_x2_y2_z2_l749_74946

open Real

theorem min_x2_y2_z2 (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3 * x * y * z = 8) : x^2 + y^2 + z^2 ≥ 4 :=
sorry

end min_x2_y2_z2_l749_74946


namespace number_of_people_in_group_l749_74926

-- Definitions and conditions
def total_cost : ℕ := 94
def mango_juice_cost : ℕ := 5
def pineapple_juice_cost : ℕ := 6
def pineapple_cost_total : ℕ := 54

-- Theorem statement to prove
theorem number_of_people_in_group : 
  ∃ M P : ℕ, 
    mango_juice_cost * M + pineapple_juice_cost * P = total_cost ∧ 
    pineapple_juice_cost * P = pineapple_cost_total ∧ 
    M + P = 17 := 
by 
  sorry

end number_of_people_in_group_l749_74926


namespace addition_result_l749_74970

theorem addition_result : 148 + 32 + 18 + 2 = 200 :=
by
  sorry

end addition_result_l749_74970


namespace feasibility_orderings_l749_74936

theorem feasibility_orderings (a : ℝ) :
  (a ≠ 0) →
  (∀ a > 0, a < 2 * a ∧ 2 * a < 3 * a + 1) ∧
  ¬∃ a, a < 3 * a + 1 ∧ 3 * a + 1 < 2 * a ∧ 2 * a < 3 * a + 1 ∧ a ≠ 0 ∧ a > 0 ∧ a < -1 / 2 ∧ a < 0 ∧ a < -1 ∧ a < -1 / 2 ∧ a < -1 / 2 ∧ a < 0 :=
sorry

end feasibility_orderings_l749_74936


namespace john_height_in_feet_l749_74965

theorem john_height_in_feet (initial_height : ℕ) (growth_rate : ℕ) (months : ℕ) (inches_per_foot : ℕ) :
  initial_height = 66 → growth_rate = 2 → months = 3 → inches_per_foot = 12 → 
  (initial_height + growth_rate * months) / inches_per_foot = 6 := by
  intros h1 h2 h3 h4
  sorry

end john_height_in_feet_l749_74965


namespace distance_between_lines_l749_74927

-- Definitions from conditions in (a)
def l1 (x y : ℝ) := 3 * x + 4 * y - 7 = 0
def l2 (x y : ℝ) := 6 * x + 8 * y + 1 = 0

-- The proof goal from (c)
theorem distance_between_lines : 
  ∀ (x y : ℝ),
    (l1 x y) → 
    (l2 x y) →
      -- Distance between the lines is 3/2
      ( (|(-14) - 1| : ℝ) / (Real.sqrt (6^2 + 8^2)) ) = 3 / 2 :=
by
  sorry

end distance_between_lines_l749_74927


namespace total_people_at_zoo_l749_74952

theorem total_people_at_zoo (A K : ℕ) (ticket_price_adult : ℕ := 28) (ticket_price_kid : ℕ := 12) (total_sales : ℕ := 3864) (number_of_kids : ℕ := 203) :
  (ticket_price_adult * A + ticket_price_kid * number_of_kids = total_sales) → 
  (A + number_of_kids = 254) :=
by
  sorry

end total_people_at_zoo_l749_74952


namespace fraction_a_over_b_l749_74917

theorem fraction_a_over_b (a b : ℝ) (h : 2 * a = 3 * b) : a / b = 3 / 2 := by
  sorry

end fraction_a_over_b_l749_74917


namespace cyclists_equal_distance_l749_74939

theorem cyclists_equal_distance (v1 v2 v3 : ℝ) (t1 t2 t3 : ℝ) (d : ℝ)
  (h_v1 : v1 = 12) (h_v2 : v2 = 16) (h_v3 : v3 = 24)
  (h_one_riding : t1 + t2 + t3 = 3) 
  (h_dist_equal : v1 * t1 = v2 * t2 ∧ v2 * t2 = v3 * t3 ∧ v1 * t1 = d) :
  d = 16 :=
by
  sorry

end cyclists_equal_distance_l749_74939


namespace min_number_of_girls_l749_74941

theorem min_number_of_girls (d : ℕ) (students : ℕ) (boys : ℕ → ℕ) : 
  students = 20 ∧ ∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k → boys i ≠ boys j ∨ boys j ≠ boys k ∨ boys i ≠ boys k → d = 6 :=
by
  sorry

end min_number_of_girls_l749_74941


namespace tire_price_l749_74901

theorem tire_price (x : ℝ) (h1 : 2 * x + 5 = 185) : x = 90 :=
by
  sorry

end tire_price_l749_74901


namespace company_employees_count_l749_74907

theorem company_employees_count :
  ∃ E : ℕ, E = 80 + 100 - 30 + 20 := 
sorry

end company_employees_count_l749_74907


namespace no_such_natural_number_exists_l749_74984

theorem no_such_natural_number_exists :
  ¬ ∃ (n s : ℕ), n = 2014 * s + 2014 ∧ n % s = 2014 ∧ (n / s) = 2014 :=
by
  sorry

end no_such_natural_number_exists_l749_74984


namespace sum_and_product_of_roots_l749_74999

theorem sum_and_product_of_roots :
  let a := 1
  let b := -7
  let c := 12
  (∀ x: ℝ, x^2 - 7*x + 12 = 0 → (x = 3 ∨ x = 4)) →
  (-b/a = 7) ∧ (c/a = 12) := 
by
  sorry

end sum_and_product_of_roots_l749_74999


namespace arvin_first_day_km_l749_74976

theorem arvin_first_day_km :
  ∀ (x : ℕ), (∀ i : ℕ, (i < 5 → (i + x) < 6) → (x + 4 = 6)) → x = 2 :=
by sorry

end arvin_first_day_km_l749_74976


namespace prove_nat_number_l749_74978

theorem prove_nat_number (p : ℕ) (hp : Nat.Prime p) (n : ℕ) :
  n^2 = p^2 + 3*p + 9 → n = 7 :=
sorry

end prove_nat_number_l749_74978


namespace unique_real_y_l749_74991

def star (x y : ℝ) : ℝ := 5 * x - 4 * y + 2 * x * y

theorem unique_real_y (y : ℝ) : (∃! y : ℝ, star 4 y = 10) :=
  by {
    sorry
  }

end unique_real_y_l749_74991


namespace square_of_binomial_l749_74969

theorem square_of_binomial (k : ℝ) : (∃ a : ℝ, x^2 - 20 * x + k = (x - a)^2) → k = 100 :=
by {
  sorry
}

end square_of_binomial_l749_74969


namespace positive_integer_as_sum_of_distinct_factors_l749_74906

-- Defining that all elements of a list are factors of a given number
def AllFactorsOf (factors : List ℕ) (n : ℕ) : Prop :=
  ∀ f ∈ factors, f ∣ n

-- Defining that the sum of elements in the list equals a given number
def SumList (l : List ℕ) : ℕ :=
  l.foldl (· + ·) 0

-- Theorem statement
theorem positive_integer_as_sum_of_distinct_factors (n m : ℕ) (hn : 0 < n) (hm : 1 ≤ m ∧ m ≤ n!) :
  ∃ factors : List ℕ, factors.length ≤ n ∧ AllFactorsOf factors n! ∧ SumList factors = m := 
sorry

end positive_integer_as_sum_of_distinct_factors_l749_74906


namespace find_exponent_l749_74951

theorem find_exponent (x : ℕ) (h : 2^x + 2^x + 2^x + 2^x + 2^x = 2048) : x = 9 :=
sorry

end find_exponent_l749_74951


namespace speed_of_stream_l749_74966

theorem speed_of_stream 
  (b s : ℝ) 
  (h1 : 78 = (b + s) * 2) 
  (h2 : 50 = (b - s) * 2) 
  : s = 7 := 
sorry

end speed_of_stream_l749_74966


namespace weight_of_7_weights_l749_74996

theorem weight_of_7_weights :
  ∀ (w : ℝ), (16 * w + 0.6 = 17.88) → 7 * w = 7.56 :=
by
  intros w h
  sorry

end weight_of_7_weights_l749_74996


namespace exponent_is_23_l749_74977

theorem exponent_is_23 (k : ℝ) : (1/2: ℝ) ^ 23 * (1/81: ℝ) ^ k = (1/18: ℝ) ^ 23 → 23 = 23 := by
  intro h
  sorry

end exponent_is_23_l749_74977


namespace circle_radius_integer_l749_74931

theorem circle_radius_integer (r : ℤ)
  (center : ℝ × ℝ)
  (inside_point : ℝ × ℝ)
  (outside_point : ℝ × ℝ)
  (h1 : center = (-2, -3))
  (h2 : inside_point = (-2, 2))
  (h3 : outside_point = (5, -3))
  (h4 : (dist center inside_point : ℝ) < r)
  (h5 : (dist center outside_point : ℝ) > r) 
  : r = 6 :=
sorry

end circle_radius_integer_l749_74931


namespace vertex_parabola_l749_74990

theorem vertex_parabola (h k : ℝ) : 
  (∀ x : ℝ, -((x - 2)^2) + 3 = k) → (h = 2 ∧ k = 3) :=
by 
  sorry

end vertex_parabola_l749_74990


namespace number_of_customers_before_lunch_rush_l749_74922

-- Defining the total number of customers during the lunch rush
def total_customers_during_lunch_rush : ℕ := 49 + 2

-- Defining the number of additional customers during the lunch rush
def additional_customers : ℕ := 12

-- Target statement to prove
theorem number_of_customers_before_lunch_rush : total_customers_during_lunch_rush - additional_customers = 39 :=
  by sorry

end number_of_customers_before_lunch_rush_l749_74922


namespace complementary_angles_ratio_4_to_1_smaller_angle_l749_74959

theorem complementary_angles_ratio_4_to_1_smaller_angle :
  ∃ (θ : ℝ), (4 * θ + θ = 90) ∧ (θ = 18) :=
by
  sorry

end complementary_angles_ratio_4_to_1_smaller_angle_l749_74959


namespace discount_rate_for_1000_min_price_for_1_3_discount_l749_74980

def discounted_price (original_price : ℕ) : ℕ := 
  original_price * 80 / 100

def voucher_amount (discounted_price : ℕ) : ℕ :=
  if discounted_price < 400 then 30
  else if discounted_price < 500 then 60
  else if discounted_price < 700 then 100
  else if discounted_price < 900 then 130
  else 0 -- Can extend the rule as needed

def discount_rate (original_price : ℕ) : ℚ := 
  let total_discount := original_price * 20 / 100 + voucher_amount (discounted_price original_price)
  (total_discount : ℚ) / (original_price : ℚ)

theorem discount_rate_for_1000 : 
  discount_rate 1000 = 0.33 := 
by
  sorry

theorem min_price_for_1_3_discount :
  ∀ (x : ℕ), 500 ≤ x ∧ x ≤ 800 → 0.33 ≤ discount_rate x ↔ (625 ≤ x ∧ x ≤ 750) :=
by
  sorry

end discount_rate_for_1000_min_price_for_1_3_discount_l749_74980


namespace largest_perfect_square_factor_of_3780_l749_74998

theorem largest_perfect_square_factor_of_3780 :
  ∃ m : ℕ, (∃ k : ℕ, 3780 = k * m * m) ∧ m * m = 36 :=
by
  sorry

end largest_perfect_square_factor_of_3780_l749_74998


namespace number_of_distributions_room_receives_three_people_number_of_distributions_room_receives_at_least_one_person_l749_74982

-- Define the total number of people
def total_people : ℕ := 6

-- Define the number of rooms
def total_rooms : ℕ := 2

-- For the first question, define: each room must receive exact three people
def room_receives_three_people (n m : ℕ) : Prop :=
  n = 3 ∧ m = 3

-- For the second question, define: each room must receive at least one person
def room_receives_at_least_one_person (n m : ℕ) : Prop :=
  n ≥ 1 ∧ m ≥ 1

theorem number_of_distributions_room_receives_three_people :
  ∃ (ways : ℕ), ways = 20 :=
by
  sorry

theorem number_of_distributions_room_receives_at_least_one_person :
  ∃ (ways : ℕ), ways = 62 :=
by
  sorry

end number_of_distributions_room_receives_three_people_number_of_distributions_room_receives_at_least_one_person_l749_74982


namespace original_cost_proof_l749_74973

/-!
# Prove that the original cost of the yearly subscription to professional magazines is $940.
# Given conditions:
# 1. The company must make a 20% cut in the magazine budget.
# 2. After the cut, the company will spend $752.
-/

theorem original_cost_proof (x : ℝ)
  (h1 : 0.80 * x = 752) :
  x = 940 :=
by
  sorry

end original_cost_proof_l749_74973


namespace necessary_but_not_sufficient_l749_74955

theorem necessary_but_not_sufficient (a b : ℝ) : (a > b) → (a + 1 > b - 2) :=
by sorry

end necessary_but_not_sufficient_l749_74955


namespace spongebob_earnings_l749_74956

-- Define the conditions as variables and constants
def burgers_sold : ℕ := 30
def price_per_burger : ℝ := 2
def fries_sold : ℕ := 12
def price_per_fries : ℝ := 1.5

-- Define total earnings calculation
def earnings_from_burgers := burgers_sold * price_per_burger
def earnings_from_fries := fries_sold * price_per_fries
def total_earnings := earnings_from_burgers + earnings_from_fries

-- State the theorem we need to prove
theorem spongebob_earnings :
  total_earnings = 78 := by
    sorry

end spongebob_earnings_l749_74956


namespace circle_tangent_x_axis_at_origin_l749_74942

theorem circle_tangent_x_axis_at_origin (D E F : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + D * x + E * y + F = 0 → x = 0 ∧ y = 0) ↔ (D = 0 ∧ F = 0 ∧ E ≠ 0) :=
sorry

end circle_tangent_x_axis_at_origin_l749_74942


namespace investment_Q_correct_l749_74925

-- Define the investments of P and Q
def investment_P : ℝ := 40000
def investment_Q : ℝ := 60000

-- Define the profit share ratio
def profit_ratio_PQ : ℝ × ℝ := (2, 3)

-- State the theorem to prove
theorem investment_Q_correct :
  (investment_P / investment_Q = (profit_ratio_PQ.1 / profit_ratio_PQ.2)) → 
  investment_Q = 60000 := 
by 
  sorry

end investment_Q_correct_l749_74925


namespace tan_theta_value_l749_74979

open Real

theorem tan_theta_value
  (theta : ℝ)
  (h_quad : 3 * pi / 2 < theta ∧ theta < 2 * pi)
  (h_sin : sin theta = -sqrt 6 / 3) :
  tan theta = -sqrt 2 := by
  sorry

end tan_theta_value_l749_74979


namespace difference_of_cubes_l749_74989

theorem difference_of_cubes (m n : ℕ) (h1 : m > 0) (h2 : n > 0) (h3 : m^2 - n^2 = 43) : m^3 - n^3 = 1387 :=
by
  sorry

end difference_of_cubes_l749_74989


namespace average_fuel_consumption_correct_l749_74914

def distance_to_x : ℕ := 150
def distance_to_y : ℕ := 220
def fuel_to_x : ℕ := 20
def fuel_to_y : ℕ := 15

def total_distance : ℕ := distance_to_x + distance_to_y
def total_fuel_used : ℕ := fuel_to_x + fuel_to_y
def avg_fuel_consumption : ℚ := total_fuel_used / total_distance

theorem average_fuel_consumption_correct :
  avg_fuel_consumption = 0.0946 := by
  sorry

end average_fuel_consumption_correct_l749_74914


namespace combined_average_pieces_lost_l749_74957

theorem combined_average_pieces_lost
  (audrey_losses : List ℕ) (thomas_losses : List ℕ)
  (h_audrey : audrey_losses = [6, 8, 4, 7, 10])
  (h_thomas : thomas_losses = [5, 6, 3, 7, 11]) :
  (audrey_losses.sum + thomas_losses.sum : ℚ) / 5 = 13.4 := by 
  sorry

end combined_average_pieces_lost_l749_74957


namespace f_neg_a_eq_neg_2_l749_74909

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (1 + x^2) - x) + 1

-- Given condition: f(a) = 4
variable (a : ℝ)
axiom h_f_a : f a = 4

-- We need to prove that: f(-a) = -2
theorem f_neg_a_eq_neg_2 (a : ℝ) (h_f_a : f a = 4) : f (-a) = -2 :=
by
  sorry

end f_neg_a_eq_neg_2_l749_74909


namespace find_b_l749_74988

theorem find_b (a b : ℝ) (h1 : (1 : ℝ)^3 + a*(1)^2 + b*1 + a^2 = 10)
    (h2 : 3*(1 : ℝ)^2 + 2*a*(1) + b = 0) : b = -11 :=
sorry

end find_b_l749_74988


namespace solve_fractional_equation_l749_74962

theorem solve_fractional_equation {x : ℝ} (h1 : x ≠ -1) (h2 : x ≠ 0) :
  6 / (x + 1) = (x + 5) / (x * (x + 1)) ↔ x = 1 :=
by
  -- This proof is left as an exercise.
  sorry

end solve_fractional_equation_l749_74962


namespace ratio_of_John_to_Mary_l749_74920

-- Definitions based on conditions
variable (J M T : ℕ)
variable (hT : T = 60)
variable (hJ : J = T / 2)
variable (hAvg : (J + M + T) / 3 = 35)

-- Statement to prove
theorem ratio_of_John_to_Mary : J / M = 2 := by
  -- Proof goes here
  sorry

end ratio_of_John_to_Mary_l749_74920


namespace part_II_l749_74958

noncomputable def f (a x : ℝ) : ℝ := (1 / 2) * a * x ^ 2 + (a - 1) * x - Real.log x

theorem part_II (a : ℝ) (h : a > 0) :
  ∀ x > 0, f a x ≥ 2 - (3 / (2 * a)) :=
sorry

end part_II_l749_74958


namespace max_distance_from_origin_to_line_l749_74948

variable (k : ℝ)

def line (x y : ℝ) : Prop := k * x + y + 1 = 0

theorem max_distance_from_origin_to_line :
  ∃ k : ℝ, ∀ x y : ℝ, line k x y -> dist (0, 0) (x, y) ≤ 1 := 
sorry

end max_distance_from_origin_to_line_l749_74948


namespace smallest_value_inequality_l749_74904

variable (a b c d : ℝ)
variable (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)

theorem smallest_value_inequality :
  (a + b + c + d) * (1 / (a + b) + 1 / (a + c) + 1 / (b + d) + 1 / (c + d)) ≥ 8 :=
sorry

end smallest_value_inequality_l749_74904


namespace number_of_children_l749_74933

-- Define conditions
variable (A C : ℕ) (h1 : A + C = 280) (h2 : 60 * A + 25 * C = 14000)

-- Lean statement to prove the number of children
theorem number_of_children : C = 80 :=
by
  sorry

end number_of_children_l749_74933


namespace expected_value_of_win_l749_74971

noncomputable def win_amount (n : ℕ) : ℕ :=
  2 * n^2

noncomputable def expected_value : ℝ :=
  (1/8) * (win_amount 1 + win_amount 2 + win_amount 3 + win_amount 4 + win_amount 5 + win_amount 6 + win_amount 7 + win_amount 8)

theorem expected_value_of_win :
  expected_value = 51 := by
  sorry

end expected_value_of_win_l749_74971


namespace min_value_of_f_in_D_l749_74908

noncomputable def f (x y : ℝ) : ℝ := 6 * (x^2 + y^2) * (x + y) - 4 * (x^2 + x * y + y^2) - 3 * (x + y) + 5

def D (x y : ℝ) : Prop := x > 0 ∧ y > 0

theorem min_value_of_f_in_D : ∃ (x y : ℝ), D x y ∧ f x y = 2 ∧ (∀ (u v : ℝ), D u v → f u v ≥ 2) :=
by
  sorry

end min_value_of_f_in_D_l749_74908


namespace rational_mul_example_l749_74913

theorem rational_mul_example : ((19 + 15 / 16) * (-8)) = (-159 - 1 / 2) :=
by
  sorry

end rational_mul_example_l749_74913


namespace benzene_molecular_weight_l749_74995

theorem benzene_molecular_weight (w: ℝ) (h: 4 * w = 312) : w = 78 :=
by
  sorry

end benzene_molecular_weight_l749_74995


namespace determine_x_l749_74986

theorem determine_x (y : ℚ) (h : y = (36 + 249 / 999) / 100) :
  ∃ x : ℕ, y = x / 99900 ∧ x = 36189 :=
by
  sorry

end determine_x_l749_74986


namespace proof_speed_of_man_in_still_water_l749_74983

def speed_of_man_in_still_water (v_m v_s : ℝ) : Prop :=
  50 / 4 = v_m + v_s ∧ 30 / 6 = v_m - v_s

theorem proof_speed_of_man_in_still_water (v_m v_s : ℝ) :
  speed_of_man_in_still_water v_m v_s → v_m = 8.75 :=
by
  intro h
  sorry

end proof_speed_of_man_in_still_water_l749_74983


namespace ratio_of_areas_l749_74937

noncomputable def area (a b : ℕ) : ℚ := (a * b : ℚ) / 2

theorem ratio_of_areas :
  let GHI := (7, 24, 25)
  let JKL := (9, 40, 41)
  area 7 24 / area 9 40 = (7 : ℚ) / 15 :=
by
  sorry

end ratio_of_areas_l749_74937


namespace sara_total_money_eq_640_l749_74902

def days_per_week : ℕ := 5
def cakes_per_day : ℕ := 4
def price_per_cake : ℕ := 8
def weeks : ℕ := 4

theorem sara_total_money_eq_640 :
  (days_per_week * cakes_per_day * price_per_cake * weeks) = 640 := 
sorry

end sara_total_money_eq_640_l749_74902


namespace divide_talers_l749_74987

theorem divide_talers (loaves1 loaves2 : ℕ) (coins : ℕ) (loavesShared : ℕ) :
  loaves1 = 3 → loaves2 = 5 → coins = 8 → loavesShared = (loaves1 + loaves2) →
  (3 - loavesShared / 3) * coins / loavesShared = 1 ∧ (5 - loavesShared / 3) * coins / loavesShared = 7 := 
by
  intros h1 h2 h3 h4
  sorry

end divide_talers_l749_74987


namespace january_roses_l749_74945

theorem january_roses (r_october r_november r_december r_february r_january : ℕ)
  (h_october_november : r_november = r_october + 12)
  (h_november_december : r_december = r_november + 12)
  (h_december_january : r_january = r_december + 12)
  (h_january_february : r_february = r_january + 12) :
  r_january = 144 :=
by {
  -- The proof would go here.
  sorry
}

end january_roses_l749_74945


namespace force_for_wrenches_l749_74924

open Real

theorem force_for_wrenches (F : ℝ) (k : ℝ) :
  (F * 12 = 3600) → 
  (k = 3600) →
  (3600 / 8 = 450) →
  (3600 / 18 = 200) →
  true :=
by
  intro hF hk h8 h18
  trivial

end force_for_wrenches_l749_74924


namespace temperature_conversion_correct_l749_74964

noncomputable def f_to_c (T : ℝ) : ℝ := (T - 32) * (5 / 9)

theorem temperature_conversion_correct :
  f_to_c 104 = 40 :=
by
  sorry

end temperature_conversion_correct_l749_74964


namespace ratio_pat_mark_l749_74974

-- Conditions (as definitions)
variables (K P M : ℕ)
variables (h1 : P = 2 * K)  -- Pat charged twice as much time as Kate
variables (h2 : M = K + 80) -- Mark charged 80 more hours than Kate
variables (h3 : K + P + M = 144) -- Total hours charged is 144

theorem ratio_pat_mark (h1 : P = 2 * K) (h2 : M = K + 80) (h3 : K + P + M = 144) : 
  P / M = 1 / 3 :=
by
  sorry -- to be proved

end ratio_pat_mark_l749_74974


namespace salt_solution_percentage_l749_74915

theorem salt_solution_percentage
  (x : ℝ)
  (y : ℝ)
  (h1 : 600 + y = 1000)
  (h2 : 600 * x + y * 0.12 = 1000 * 0.084) :
  x = 0.06 :=
by
  -- The proof goes here.
  sorry

end salt_solution_percentage_l749_74915


namespace three_digit_perfect_squares_div_by_4_count_l749_74919

theorem three_digit_perfect_squares_div_by_4_count : 
  (∃ count : ℕ, count = 11 ∧ (∀ n : ℕ, 10 ≤ n ∧ n ≤ 31 → n^2 ≥ 100 ∧ n^2 ≤ 999 ∧ n^2 % 4 = 0)) :=
by
  sorry

end three_digit_perfect_squares_div_by_4_count_l749_74919


namespace fraction_satisfactory_is_two_thirds_l749_74932

-- Total number of students with satisfactory grades
def satisfactory_grades : ℕ := 3 + 7 + 4 + 2

-- Total number of students with unsatisfactory grades
def unsatisfactory_grades : ℕ := 4

-- Total number of students
def total_students : ℕ := satisfactory_grades + unsatisfactory_grades

-- Fraction of satisfactory grades
def fraction_satisfactory : ℚ := satisfactory_grades / total_students

theorem fraction_satisfactory_is_two_thirds :
  fraction_satisfactory = 2 / 3 := by
  sorry

end fraction_satisfactory_is_two_thirds_l749_74932


namespace president_and_committee_combination_l749_74928

theorem president_and_committee_combination (n : ℕ) (k : ℕ) (total : ℕ) :
  n = 10 ∧ k = 3 ∧ total = (10 * Nat.choose 9 3) → total = 840 :=
by
  intros
  sorry

end president_and_committee_combination_l749_74928


namespace find_a_plus_b_eq_102_l749_74994

theorem find_a_plus_b_eq_102 :
  ∃ (a b : ℕ), (1600^(1 / 2) - 24 = (a^(1 / 2) - b)^2) ∧ (a + b = 102) :=
by {
  sorry
}

end find_a_plus_b_eq_102_l749_74994


namespace michael_average_speed_l749_74935

-- Definitions of conditions
def motorcycle_speed := 20 -- mph
def motorcycle_time := 40 / 60 -- hours
def jogging_speed := 5 -- mph
def jogging_time := 60 / 60 -- hours

-- Define the total distance
def motorcycle_distance := motorcycle_speed * motorcycle_time
def jogging_distance := jogging_speed * jogging_time
def total_distance := motorcycle_distance + jogging_distance

-- Define the total time
def total_time := motorcycle_time + jogging_time

-- The proof statement to be proven
theorem michael_average_speed :
  total_distance / total_time = 11 := 
sorry

end michael_average_speed_l749_74935


namespace distinct_real_roots_of_quadratic_l749_74923

/-
Given a quadratic equation x^2 + 4x = 0,
prove that the equation has two distinct real roots.
-/

theorem distinct_real_roots_of_quadratic : 
  ∀ (a b c : ℝ), a = 1 → b = 4 → c = 0 → (b^2 - 4 * a * c) > 0 → 
  ∃ r₁ r₂ : ℝ, r₁ ≠ r₂ ∧ (r₁^2 + 4 * r₁ = 0) ∧ (r₂^2 + 4 * r₂ = 0) := 
by
  intros a b c ha hb hc hΔ
  sorry -- Proof to be provided later

end distinct_real_roots_of_quadratic_l749_74923


namespace total_initial_passengers_l749_74918

theorem total_initial_passengers (M W : ℕ) 
  (h1 : W = M / 3) 
  (h2 : M - 24 = W + 12) : 
  M + W = 72 :=
sorry

end total_initial_passengers_l749_74918


namespace krish_remaining_money_l749_74972

variable (initial_amount sweets stickers friends each_friend charity : ℝ)

theorem krish_remaining_money :
  initial_amount = 200.50 →
  sweets = 35.25 →
  stickers = 10.75 →
  friends = 4 →
  each_friend = 25.20 →
  charity = 15.30 →
  initial_amount - (sweets + stickers + friends * each_friend + charity) = 38.40 :=
by
  intros h_initial h_sweets h_stickers h_friends h_each_friend h_charity
  sorry

end krish_remaining_money_l749_74972


namespace g_f_g_1_equals_82_l749_74947

def f (x : ℤ) : ℤ := 2 * x + 2
def g (x : ℤ) : ℤ := 5 * x + 2
def x : ℤ := 1

theorem g_f_g_1_equals_82 : g (f (g x)) = 82 := by
  sorry

end g_f_g_1_equals_82_l749_74947


namespace sequence_term_divisible_by_n_l749_74963

theorem sequence_term_divisible_by_n (n : ℕ) (hn1 : 1 < n) (hn_odd : n % 2 = 1) :
  ∃ k : ℕ, 1 ≤ k ∧ k < n ∧ n ∣ (2^k - 1) :=
by
  sorry

end sequence_term_divisible_by_n_l749_74963


namespace mike_washed_cars_l749_74960

theorem mike_washed_cars 
    (total_work_time : ℕ := 4 * 60) 
    (wash_time : ℕ := 10)
    (oil_change_time : ℕ := 15) 
    (tire_change_time : ℕ := 30) 
    (num_oil_changes : ℕ := 6) 
    (num_tire_changes : ℕ := 2) 
    (remaining_time : ℕ := total_work_time - (num_oil_changes * oil_change_time + num_tire_changes * tire_change_time))
    (num_cars_washed : ℕ := remaining_time / wash_time) :
    num_cars_washed = 9 := by
  sorry

end mike_washed_cars_l749_74960


namespace value_range_f_l749_74953

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x + 2 * Real.cos x - Real.sin (2 * x) + 1

theorem value_range_f :
  ∀ x ∈ Set.Ico (-(5 * Real.pi) / 12) (Real.pi / 3), 
  f x ∈ Set.Icc ((3 : ℝ) / 2 - Real.sqrt 2) 3 :=
by
  sorry

end value_range_f_l749_74953


namespace minimum_value_shifted_function_l749_74940

def f (x a : ℝ) : ℝ := x^2 + 4 * x + 7 - a

theorem minimum_value_shifted_function (a : ℝ) (h : ∃ x, f x a = 2) :
  ∃ y, (∃ x, y = f (x - 2015) a) ∧ y = 2 :=
sorry

end minimum_value_shifted_function_l749_74940


namespace acute_angle_89_l749_74997

def is_acute_angle (angle : ℝ) : Prop := angle > 0 ∧ angle < 90

theorem acute_angle_89 :
  is_acute_angle 89 :=
by {
  -- proof details would go here, since only the statement is required
  sorry
}

end acute_angle_89_l749_74997


namespace compute_expression_l749_74944

theorem compute_expression (p q r : ℝ) 
  (h1 : p + q + r = 6) 
  (h2 : pq + qr + rp = 11) 
  (h3 : pqr = 12) : 
  (pq / r) + (qr / p) + (rp / q) = -23 / 12 := 
sorry

end compute_expression_l749_74944


namespace binomial_expansion_coefficient_l749_74921

theorem binomial_expansion_coefficient :
  let a_0 : ℚ := (1 + 2 * (0:ℚ))^5
  (1 + 2 * x)^5 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 →
  a_3 = 80 :=
by 
  sorry

end binomial_expansion_coefficient_l749_74921


namespace greatest_possible_n_l749_74938

theorem greatest_possible_n (n : ℤ) (h : 101 * n ^ 2 ≤ 8100) : n ≤ 8 :=
by
  -- Intentionally left uncommented.
  sorry

end greatest_possible_n_l749_74938


namespace correct_model_l749_74968

def average_homework_time_decrease (x : ℝ) : Prop :=
  100 * (1 - x) ^ 2 = 70

theorem correct_model (x : ℝ) : average_homework_time_decrease x := 
  sorry

end correct_model_l749_74968


namespace prism_volume_l749_74903

theorem prism_volume (x y z : ℝ) (h1 : x * y = 24) (h2 : y * z = 8) (h3 : x * z = 3) : 
  x * y * z = 24 :=
sorry

end prism_volume_l749_74903


namespace range_of_a_l749_74934

theorem range_of_a (a : ℝ) (p : ∀ x : ℝ, |x - 1| + |x + 1| ≥ 3 * a) (q : 0 < 2 * a - 1 ∧ 2 * a - 1 < 1) : 
  (1 / 2) < a ∧ a ≤ (2 / 3) :=
sorry

end range_of_a_l749_74934


namespace intersection_of_sets_l749_74975

def set_A (x : ℝ) := x + 1 ≤ 3
def set_B (x : ℝ) := 4 - x^2 ≤ 0

theorem intersection_of_sets : {x : ℝ | set_A x} ∩ {x : ℝ | set_B x} = {x : ℝ | x ≤ -2} ∪ {2} :=
by
  sorry

end intersection_of_sets_l749_74975


namespace total_toothpicks_needed_l749_74905

/-- The number of toothpicks needed to construct both a large and smaller equilateral triangle 
    side by side, given the large triangle has a base of 100 small triangles and the smaller triangle 
    has a base of 50 small triangles -/
theorem total_toothpicks_needed 
  (base_large : ℕ) (base_small : ℕ) (shared_boundary : ℕ) 
  (h1 : base_large = 100) (h2 : base_small = 50) (h3 : shared_boundary = base_small) :
  3 * (100 * 101 / 2) / 2 + 3 * (50 * 51 / 2) / 2 - shared_boundary = 9462 := 
sorry

end total_toothpicks_needed_l749_74905


namespace number_of_apps_needed_l749_74993

-- Definitions based on conditions
variable (cost_per_app : ℕ) (total_money : ℕ) (remaining_money : ℕ)

-- Assume the conditions given
axiom cost_app_eq : cost_per_app = 4
axiom total_money_eq : total_money = 66
axiom remaining_money_eq : remaining_money = 6

-- The goal is to determine the number of apps Lidia needs to buy
theorem number_of_apps_needed (n : ℕ) (h : total_money - remaining_money = cost_per_app * n) :
  n = 15 :=
by
  sorry

end number_of_apps_needed_l749_74993


namespace sufficient_but_not_necessary_condition_l749_74967

variable {α : Type*} (A B : Set α)

theorem sufficient_but_not_necessary_condition (h₁ : A ∩ B = A) (h₂ : A ≠ B) :
  (∀ x, x ∈ A → x ∈ B) ∧ ¬(∀ x, x ∈ B → x ∈ A) :=
by 
  sorry

end sufficient_but_not_necessary_condition_l749_74967


namespace expression_independent_of_alpha_l749_74912

theorem expression_independent_of_alpha
  (α : Real) (n : ℤ) (h : α ≠ (n * (π / 2)) + (π / 12)) :
  (1 - 2 * Real.sin (α - (3 * π / 2))^2 + (Real.sqrt 3) * Real.cos (2 * α + (3 * π / 2))) /
  (Real.sin (π / 6 - 2 * α)) = -2 := 
sorry

end expression_independent_of_alpha_l749_74912
