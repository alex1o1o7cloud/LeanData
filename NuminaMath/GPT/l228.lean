import Mathlib

namespace value_of_b_l228_22805

theorem value_of_b (a b : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 35 * 45 * b) : b = 105 :=
sorry

end value_of_b_l228_22805


namespace relationship_points_l228_22822

noncomputable def is_on_inverse_proportion (m x y : ℝ) : Prop :=
  y = (-m^2 - 2) / x

theorem relationship_points (a b c m : ℝ) :
  is_on_inverse_proportion m a (-1) ∧
  is_on_inverse_proportion m b 2 ∧
  is_on_inverse_proportion m c 3 →
  a > c ∧ c > b :=
by
  sorry

end relationship_points_l228_22822


namespace eight_xyz_le_one_equality_conditions_l228_22873

theorem eight_xyz_le_one (x y z : ℝ) (h : x^2 + y^2 + z^2 + 2 * x * y * z = 1) :
  8 * x * y * z ≤ 1 :=
sorry

theorem equality_conditions (x y z : ℝ) (h : x^2 + y^2 + z^2 + 2 * x * y * z = 1) :
  8 * x * y * z = 1 ↔ (x = 1/2 ∧ y = 1/2 ∧ z = 1/2) ∨
                   (x = -1/2 ∧ y = -1/2 ∧ z = 1/2) ∨
                   (x = -1/2 ∧ y = 1/2 ∧ z = -1/2) ∨
                   (x = 1/2 ∧ y = -1/2 ∧ z = -1/2) :=
sorry

end eight_xyz_le_one_equality_conditions_l228_22873


namespace number_of_boxwoods_l228_22864

variables (x : ℕ)
def charge_per_trim := 5
def charge_per_shape := 15
def number_of_shaped_boxwoods := 4
def total_charge := 210
def total_shaping_charge := number_of_shaped_boxwoods * charge_per_shape

theorem number_of_boxwoods (h : charge_per_trim * x + total_shaping_charge = total_charge) : x = 30 :=
by
  sorry

end number_of_boxwoods_l228_22864


namespace train_meetings_between_stations_l228_22869

theorem train_meetings_between_stations
  (travel_time : ℕ := 3 * 60 + 30) -- Travel time in minutes
  (first_departure : ℕ := 6 * 60) -- First departure time in minutes from 0 (midnight)
  (departure_interval : ℕ := 60) -- Departure interval in minutes
  (A_departure_time : ℕ := 9 * 60) -- Departure time from Station A at 9:00 AM in minutes
  :
  ∃ n : ℕ, n = 7 :=
by
  sorry

end train_meetings_between_stations_l228_22869


namespace problem_statement_l228_22804

namespace GeometricRelations

variables {Line Plane : Type} [Nonempty Line] [Nonempty Plane]

-- Define parallel and perpendicular relations
def parallel (l : Line) (p : Plane) : Prop := sorry
def perpendicular (l : Line) (p : Plane) : Prop := sorry

-- Given conditions
variables (m n : Line) (α β : Plane)

-- The theorem to be proven
theorem problem_statement 
  (h1 : perpendicular m β) 
  (h2 : parallel α β) : 
  perpendicular m α :=
sorry

end GeometricRelations

end problem_statement_l228_22804


namespace discount_difference_l228_22852

noncomputable def single_discount (amount : ℝ) (rate : ℝ) : ℝ :=
  amount * (1 - rate)

noncomputable def successive_discounts (amount : ℝ) (rates : List ℝ) : ℝ :=
  rates.foldl (λ acc rate => acc * (1 - rate)) amount

theorem discount_difference:
  let amount := 12000
  let single_rate := 0.35
  let successive_rates := [0.25, 0.08, 0.02]
  single_discount amount single_rate - successive_discounts amount successive_rates = 314.4 := 
  sorry

end discount_difference_l228_22852


namespace minimum_value_of_expression_l228_22827

noncomputable def min_value_expr (x y z : ℝ) : ℝ := (x + 3 * y) * (y + 3 * z) * (2 * x * z + 1)

theorem minimum_value_of_expression (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z = 1) :
  min_value_expr x y z = 24 * Real.sqrt 2 :=
sorry

end minimum_value_of_expression_l228_22827


namespace three_point_seven_five_minus_one_point_four_six_l228_22849

theorem three_point_seven_five_minus_one_point_four_six : 3.75 - 1.46 = 2.29 :=
by sorry

end three_point_seven_five_minus_one_point_four_six_l228_22849


namespace minimum_value_l228_22840

theorem minimum_value (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x * y * z = 128) : 
  ∃ (m : ℝ), (∀ (a b c : ℝ), 0 < a → 0 < b → 0 < c → a * b * c = 128 → (a^2 + 8 * a * b + 4 * b^2 + 8 * c^2) ≥ m) 
  ∧ m = 384 :=
sorry


end minimum_value_l228_22840


namespace cats_eat_fish_l228_22825

theorem cats_eat_fish (c d: ℕ) (h1 : 1 < c) (h2 : c < 10) (h3 : c * d = 91) : c + d = 20 := by
  sorry

end cats_eat_fish_l228_22825


namespace find_length_of_EF_l228_22847

-- Definitions based on conditions
noncomputable def AB : ℝ := 300
noncomputable def DC : ℝ := 180
noncomputable def BC : ℝ := 200
noncomputable def E_as_fraction_of_BC : ℝ := (3 / 5)

-- Derived definition based on given conditions
noncomputable def EB : ℝ := E_as_fraction_of_BC * BC
noncomputable def EC : ℝ := BC - EB
noncomputable def EF : ℝ := (EC / BC) * DC

-- The theorem we need to prove
theorem find_length_of_EF : EF = 72 := by
  sorry

end find_length_of_EF_l228_22847


namespace dot_product_of_a_and_b_l228_22875

noncomputable def vector_a (a b : ℝ × ℝ) (h1 : a + b = (1, -3)) (h2 : a - b = (3, 7)) : ℝ × ℝ := 
a

noncomputable def vector_b (a b : ℝ × ℝ) (h1 : a + b = (1, -3)) (h2 : a - b = (3, 7)) : ℝ × ℝ := 
b

theorem dot_product_of_a_and_b {a b : ℝ × ℝ} 
  (h1 : a + b = (1, -3)) 
  (h2 : a - b = (3, 7)) : 
  (a.1 * b.1 + a.2 * b.2) = -12 := 
sorry

end dot_product_of_a_and_b_l228_22875


namespace minimum_value_sum_l228_22853

theorem minimum_value_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (a / (b + 3 * c) + b / (8 * c + 4 * a) + 9 * c / (3 * a + 2 * b)) ≥ 47 / 48 :=
by sorry

end minimum_value_sum_l228_22853


namespace top_angle_isosceles_triangle_l228_22819

open Real

theorem top_angle_isosceles_triangle (A B C : ℝ) (abc_is_isosceles : (A = B ∨ B = C ∨ A = C))
  (angle_A : A = 40) : (B = 40 ∨ B = 100) :=
sorry

end top_angle_isosceles_triangle_l228_22819


namespace blackBurgerCost_l228_22828

def ArevaloFamilyBill (smokySalmonCost blackBurgerCost chickenKatsuCost totalBill : ℝ) : Prop :=
  smokySalmonCost = 40 ∧ chickenKatsuCost = 25 ∧ 
  totalBill = smokySalmonCost + blackBurgerCost + chickenKatsuCost + 
    0.15 * (smokySalmonCost + blackBurgerCost + chickenKatsuCost)

theorem blackBurgerCost (smokySalmonCost chickenKatsuCost change : ℝ) (B : ℝ) 
  (h1 : smokySalmonCost = 40) 
  (h2 : chickenKatsuCost = 25)
  (h3 : 100 - change = 92) 
  (h4 : ArevaloFamilyBill smokySalmonCost B chickenKatsuCost 92) : 
  B = 15 :=
sorry

end blackBurgerCost_l228_22828


namespace rabbit_toy_cost_l228_22881

theorem rabbit_toy_cost 
  (cost_pet_food : ℝ) 
  (cost_cage : ℝ) 
  (found_dollar : ℝ)
  (total_cost : ℝ) 
  (h1 : cost_pet_food = 5.79) 
  (h2 : cost_cage = 12.51)
  (h3 : found_dollar = 1.00)
  (h4 : total_cost = 24.81):
  ∃ (cost_rabbit_toy : ℝ), cost_rabbit_toy = 7.51 := by
  let cost_rabbit_toy := total_cost - (cost_pet_food + cost_cage) + found_dollar
  use cost_rabbit_toy
  sorry

end rabbit_toy_cost_l228_22881


namespace minimize_PA2_PB2_PC2_l228_22854

def point : Type := ℝ × ℝ

noncomputable def distance_sq (P Q : point) : ℝ := 
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2

noncomputable def PA_sq (P : point) : ℝ := distance_sq P (5, 0)
noncomputable def PB_sq (P : point) : ℝ := distance_sq P (0, 5)
noncomputable def PC_sq (P : point) : ℝ := distance_sq P (-4, -3)

noncomputable def circumcircle (P : point) : Prop := 
  P.1^2 + P.2^2 = 25

noncomputable def objective_function (P : point) : ℝ := 
  PA_sq P + PB_sq P + PC_sq P

theorem minimize_PA2_PB2_PC2 : ∃ P : point, circumcircle P ∧ 
  (∀ Q : point, circumcircle Q → objective_function P ≤ objective_function Q) :=
sorry

end minimize_PA2_PB2_PC2_l228_22854


namespace cab_speed_fraction_l228_22877

theorem cab_speed_fraction :
  ∀ (S R : ℝ),
    (75 * S = 90 * R) →
    (R / S = 5 / 6) :=
by
  intros S R h
  sorry

end cab_speed_fraction_l228_22877


namespace arithmetic_sequence_fifth_term_l228_22892

theorem arithmetic_sequence_fifth_term (a1 d : ℕ) (a_n : ℕ → ℕ) 
  (h_a1 : a1 = 2) (h_d : d = 1) (h_a_n : ∀ n : ℕ, a_n n = a1 + (n-1) * d) : 
  a_n 5 = 6 := 
    by
    -- Given the conditions, we need to prove a_n evaluated at 5 is equal to 6.
    sorry

end arithmetic_sequence_fifth_term_l228_22892


namespace greatest_roses_for_680_l228_22857

/--
Greatest number of roses that can be purchased for $680
given the following costs:
- $4.50 per individual rose
- $36 per dozen roses
- $50 per two dozen roses
--/
theorem greatest_roses_for_680 (cost_individual : ℝ) 
  (cost_dozen : ℝ) 
  (cost_two_dozen : ℝ) 
  (budget : ℝ) 
  (dozen : ℕ) 
  (two_dozen : ℕ) 
  (total_budget : ℝ) 
  (individual_cost : ℝ) 
  (dozen_cost : ℝ) 
  (two_dozen_cost : ℝ) 
  (roses_dozen : ℕ) 
  (roses_two_dozen : ℕ):
  individual_cost = 4.50 → dozen_cost = 36 → two_dozen_cost = 50 →
  budget = 680 → dozen = 12 → two_dozen = 24 →
  (∀ n : ℕ, n * two_dozen_cost ≤ budget → n * two_dozen + (budget - n * two_dozen_cost) / individual_cost ≤ total_budget) →
  total_budget = 318 := 
by
  sorry

end greatest_roses_for_680_l228_22857


namespace arithmetic_mean_difference_l228_22889

theorem arithmetic_mean_difference (p q r : ℝ)
  (h1 : (p + q) / 2 = 10)
  (h2 : (q + r) / 2 = 20) : 
  r - p = 20 := 
by sorry

end arithmetic_mean_difference_l228_22889


namespace parabola_point_distance_l228_22870

open Real

noncomputable def parabola_coords (y z: ℝ) : Prop :=
  y^2 = 12 * z

noncomputable def distance (x1 y1 x2 y2: ℝ) : ℝ :=
  sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

theorem parabola_point_distance (x y: ℝ) :
  parabola_coords y x ∧ distance x y 3 0 = 9 ↔ ( x = 6 ∧ (y = 6 * sqrt 2 ∨ y = -6 * sqrt 2)) :=
by
  sorry

end parabola_point_distance_l228_22870


namespace rectangle_dimensions_l228_22826

theorem rectangle_dimensions (w l : ℕ) 
  (h1 : l = 2 * w) 
  (h2 : 2 * (w + l) = 6 * w ^ 2) : 
  w = 1 ∧ l = 2 :=
by sorry

end rectangle_dimensions_l228_22826


namespace equal_sum_partition_l228_22830

theorem equal_sum_partition (n : ℕ) (a : Fin n.succ → ℕ)
  (h1 : a 0 = 1)
  (h2 : ∀ i : Fin n, a i ≤ a i.succ ∧ a i.succ ≤ 2 * a i)
  (h3 : (Finset.univ : Finset (Fin n.succ)).sum a % 2 = 0) :
  ∃ (partition : Finset (Fin n.succ)), 
    (partition.sum a = (partitionᶜ : Finset (Fin n.succ)).sum a) :=
by sorry

end equal_sum_partition_l228_22830


namespace bag_contains_twenty_cookies_l228_22810

noncomputable def cookies_in_bag 
  (total_calories : ℕ) 
  (calories_per_cookie : ℕ)
  (bags_in_box : ℕ)
  : ℕ :=
  total_calories / (calories_per_cookie * bags_in_box)

theorem bag_contains_twenty_cookies 
  (H1 : total_calories = 1600) 
  (H2 : calories_per_cookie = 20) 
  (H3 : bags_in_box = 4)
  : cookies_in_bag total_calories calories_per_cookie bags_in_box = 20 := 
by
  have h1 : total_calories = 1600 := H1
  have h2 : calories_per_cookie = 20 := H2
  have h3 : bags_in_box = 4 := H3
  sorry

end bag_contains_twenty_cookies_l228_22810


namespace cylinder_height_relationship_l228_22863

theorem cylinder_height_relationship
  (r1 h1 r2 h2 : ℝ)
  (vol_eq : π * r1^2 * h1 = π * r2^2 * h2)
  (radius_rel : r2 = (6 / 5) * r1) : h1 = (36 / 25) * h2 := 
sorry

end cylinder_height_relationship_l228_22863


namespace half_of_number_l228_22858

theorem half_of_number (N : ℕ) (h : (4 / 15 * 5 / 7 * N) - (4 / 9 * 2 / 5 * N) = 24) : N / 2 = 945 :=
by
  sorry

end half_of_number_l228_22858


namespace cyclist_distance_l228_22806

theorem cyclist_distance
  (v t d : ℝ)
  (h1 : d = v * t)
  (h2 : d = (v + 1) * (t - 0.5))
  (h3 : d = (v - 1) * (t + 1)) :
  d = 6 :=
by
  sorry

end cyclist_distance_l228_22806


namespace expression_is_five_l228_22844

-- Define the expression
def given_expression : ℤ := abs (abs (-abs (-2 + 1) - 2) + 2)

-- Prove that the expression equals 5
theorem expression_is_five : given_expression = 5 :=
by
  -- We skip the proof for now
  sorry

end expression_is_five_l228_22844


namespace cookies_initial_count_l228_22886

theorem cookies_initial_count (C : ℕ) (h1 : C / 8 = 8) : C = 64 :=
by
  sorry

end cookies_initial_count_l228_22886


namespace points_divisibility_l228_22838

theorem points_divisibility {k n : ℕ} (hkn : k ≤ n) (hpositive : 0 < n) 
  (hcondition : ∀ x : Fin n, (∃ m : ℕ, (∀ y : Fin n, x.val ≤ y.val → y.val ≤ x.val + 1 → True) ∧ m % k = 0)) :
  k ∣ n :=
sorry

end points_divisibility_l228_22838


namespace mean_of_smallest_and_largest_is_12_l228_22812

-- Definition of the condition: the mean of five consecutive even numbers is 12.
def mean_of_five_consecutive_even_numbers_is_12 (n : ℤ) : Prop :=
  ((n - 4) + (n - 2) + n + (n + 2) + (n + 4)) / 5 = 12

-- Theorem stating that the mean of the smallest and largest of these numbers is 12.
theorem mean_of_smallest_and_largest_is_12 (n : ℤ) 
  (h : mean_of_five_consecutive_even_numbers_is_12 n) : 
  (8 + (16 : ℤ)) / (2 : ℤ) = 12 := 
by
  sorry

end mean_of_smallest_and_largest_is_12_l228_22812


namespace P_gt_Q_l228_22816

theorem P_gt_Q (a : ℝ) : 
  let P := a^2 + 2*a
  let Q := 3*a - 1
  P > Q :=
by
  sorry

end P_gt_Q_l228_22816


namespace min_expression_value_l228_22883

theorem min_expression_value (m n : ℝ) (h : m - n^2 = 1) : ∃ min_val : ℝ, min_val = 4 ∧ (∀ x y, x - y^2 = 1 → m^2 + 2 * y^2 + 4 * x - 1 ≥ min_val) :=
by
  sorry

end min_expression_value_l228_22883


namespace calculate_sum_l228_22894

theorem calculate_sum (P r : ℝ) (h1 : 2 * P * r = 10200) (h2 : P * ((1 + r) ^ 2 - 1) = 11730) : P = 17000 :=
sorry

end calculate_sum_l228_22894


namespace oreo_shop_ways_l228_22868

theorem oreo_shop_ways (α β : ℕ) (products total_ways : ℕ) :
  let oreo_flavors := 6
  let milk_flavors := 4
  let total_flavors := oreo_flavors + milk_flavors
  (α + β = products) ∧ (products = 4) ∧ (total_ways = 2143) ∧ 
  (α ≤ 2 * total_flavors) ∧ (β ≤ 4 * oreo_flavors) →
  total_ways = 2143 :=
by sorry


end oreo_shop_ways_l228_22868


namespace range_of_a_l228_22845

theorem range_of_a (a : ℝ) : (∀ (x : ℝ), x > 0 → x / (x ^ 2 + 3 * x + 1) ≤ a) → a ≥ 1 / 5 :=
by
  sorry

end range_of_a_l228_22845


namespace complement_intersection_l228_22842

noncomputable def U : Set ℤ := {-1, 0, 2}
noncomputable def A : Set ℤ := {-1, 2}
noncomputable def B : Set ℤ := {0, 2}
noncomputable def C_U_A : Set ℤ := U \ A

theorem complement_intersection :
  (C_U_A ∩ B) = {0} :=
by {
  -- sorry to skip the proof part as per instruction
  sorry
}

end complement_intersection_l228_22842


namespace sum_of_possible_values_l228_22872

theorem sum_of_possible_values (M : ℝ) (h : M * (M + 4) = 12) : M + (if M = -6 then 2 else -6) = -4 :=
by
  sorry

end sum_of_possible_values_l228_22872


namespace find_g_minus_6_l228_22802

-- Define the function g on integers
def g : ℤ → ℤ := sorry

-- Conditions given in the problem
axiom cond1 : g 1 - 1 > 0
axiom cond2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x
axiom cond3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3

-- The proof we want to make (assertion)
theorem find_g_minus_6 : g (-6) = 723 := 
sorry

end find_g_minus_6_l228_22802


namespace discount_is_25_percent_l228_22862

noncomputable def discount_percentage (M : ℝ) (C : ℝ) (SP : ℝ) : ℝ :=
  ((M - SP) / M) * 100

theorem discount_is_25_percent (M : ℝ) (C : ℝ) (SP : ℝ) 
  (h1 : C = 0.64 * M) 
  (h2 : SP = C * 1.171875) : 
  discount_percentage M C SP = 25 := 
by 
  sorry

end discount_is_25_percent_l228_22862


namespace dividend_percentage_l228_22807

theorem dividend_percentage (investment_amount market_value : ℝ) (interest_rate : ℝ) 
  (h1 : investment_amount = 44) (h2 : interest_rate = 12) (h3 : market_value = 33) : 
  ((interest_rate / 100) * investment_amount / market_value) * 100 = 16 := 
by
  sorry

end dividend_percentage_l228_22807


namespace charlie_share_l228_22861

variable (A B C : ℝ)

theorem charlie_share :
  A = (1/3) * B →
  B = (1/2) * C →
  A + B + C = 10000 →
  C = 6000 :=
by
  intros hA hB hSum
  sorry

end charlie_share_l228_22861


namespace emily_journey_length_l228_22865

theorem emily_journey_length
  (y : ℝ)
  (h1 : y / 5 + 30 + y / 3 + y / 6 = y) :
  y = 100 :=
by
  sorry

end emily_journey_length_l228_22865


namespace smallest_x_mod_7_one_sq_l228_22808

theorem smallest_x_mod_7_one_sq (x : ℕ) (h : 1 < x) (hx : (x * x) % 7 = 1) : x = 6 :=
  sorry

end smallest_x_mod_7_one_sq_l228_22808


namespace amanda_pay_if_not_finished_l228_22813

-- Define Amanda's hourly rate and daily work hours.
def amanda_hourly_rate : ℝ := 50
def amanda_daily_hours : ℝ := 10

-- Define the percentage of pay Jose will withhold.
def withholding_percentage : ℝ := 0.20

-- Define Amanda's total pay if she finishes the sales report.
def amanda_total_pay : ℝ := amanda_hourly_rate * amanda_daily_hours

-- Define the amount withheld if she does not finish the sales report.
def withheld_amount : ℝ := amanda_total_pay * withholding_percentage

-- Define the amount Amanda will receive if she does not finish the sales report.
def amanda_final_pay_not_finished : ℝ := amanda_total_pay - withheld_amount

-- The theorem to prove:
theorem amanda_pay_if_not_finished : amanda_final_pay_not_finished = 400 := by
  sorry

end amanda_pay_if_not_finished_l228_22813


namespace negative_integer_solution_l228_22891

theorem negative_integer_solution (N : ℤ) (hN : N^2 + N = -12) : N = -3 ∨ N = -4 :=
sorry

end negative_integer_solution_l228_22891


namespace product_of_last_two_digits_l228_22841

theorem product_of_last_two_digits (n : ℤ) (A B : ℤ) :
  (n % 8 = 0) ∧ (A + B = 15) ∧ (n % 10 = B) ∧ (n / 10 % 10 = A) →
  A * B = 54 :=
by
-- Add proof here
sorry

end product_of_last_two_digits_l228_22841


namespace cubic_roots_sum_of_cubes_l228_22885

theorem cubic_roots_sum_of_cubes :
  ∀ (a b c : ℝ), 
  (∀ x : ℝ, 9 * x^3 + 14 * x^2 + 2047 * x + 3024 = 0 → (x = a ∨ x = b ∨ x = c)) →
  (a + b)^3 + (b + c)^3 + (c + a)^3 = -58198 / 729 :=
by
  intros a b c roota_eqn
  sorry

end cubic_roots_sum_of_cubes_l228_22885


namespace plot_length_l228_22896

-- Define the conditions
def rent_per_acre_per_month : ℝ := 30
def total_rent_per_month : ℝ := 300
def width_feet : ℝ := 1210
def area_acres : ℝ := 10
def square_feet_per_acre : ℝ := 43560

-- Prove that the length of the plot is 360 feet
theorem plot_length (h1 : rent_per_acre_per_month = 30)
                    (h2 : total_rent_per_month = 300)
                    (h3 : width_feet = 1210)
                    (h4 : area_acres = 10)
                    (h5 : square_feet_per_acre = 43560) :
  (area_acres * square_feet_per_acre) / width_feet = 360 := 
by {
  sorry
}

end plot_length_l228_22896


namespace range_of_values_for_a_l228_22898

theorem range_of_values_for_a 
  (f : ℝ → ℝ)
  (h1 : ∀ x > 0, f x = x - 1/x - a * Real.log x)
  (h2 : ∀ x > 0, (x^2 - a * x + 1) ≥ 0) : 
  a ≤ 2 :=
sorry

end range_of_values_for_a_l228_22898


namespace andrew_permit_rate_l228_22882

def permits_per_hour (a h_a H T : ℕ) : ℕ :=
  T / (H - (a * h_a))

theorem andrew_permit_rate :
  permits_per_hour 2 3 8 100 = 50 := by
  sorry

end andrew_permit_rate_l228_22882


namespace sqrt_two_irrational_l228_22818

theorem sqrt_two_irrational : ¬ ∃ (p q : ℕ), (q ≠ 0) ∧ (Nat.gcd p q = 1) ∧ (p ^ 2 = 2 * q ^ 2) := by
  sorry

end sqrt_two_irrational_l228_22818


namespace correct_conclusions_l228_22893

-- Definitions based on conditions
def condition_1 (x : ℝ) : Prop := x ≠ 0 → x + |x| > 0
def condition_3 (a b c : ℝ) (Δ : ℝ) : Prop := a > 0 ∧ Δ ≤ 0 ∧ Δ = b^2 - 4*a*c → 
  ∀ x, a*x^2 + b*x + c ≥ 0

-- Stating the proof problem
theorem correct_conclusions (x a b c Δ : ℝ) :
  (condition_1 x) ∧ (condition_3 a b c Δ) :=
sorry

end correct_conclusions_l228_22893


namespace amount_solution_y_correct_l228_22859

-- Define conditions
def solution_x_alcohol_percentage : ℝ := 0.10
def solution_y_alcohol_percentage : ℝ := 0.30
def volume_solution_x : ℝ := 300.0
def target_alcohol_percentage : ℝ := 0.18

-- Define the main question as a theorem
theorem amount_solution_y_correct (y : ℝ) :
  (30 + 0.3 * y = 0.18 * (300 + y)) → y = 200 :=
by
  sorry

end amount_solution_y_correct_l228_22859


namespace angle_A_measure_triangle_area_l228_22814

variable {a b c : ℝ} 
variable {A B C : ℝ} 
variable (triangle : a^2 = b^2 + c^2 - 2 * b * c * (Real.cos A))

theorem angle_A_measure (h : (b - c)^2 = a^2 - b * c) : A = Real.pi / 3 :=
sorry

theorem triangle_area 
  (h1 : a = 3) 
  (h2 : Real.sin C = 2 * Real.sin B) 
  (h3 : A = Real.pi / 3) 
  (hb : b = Real.sqrt 3)
  (hc : c = 2 * Real.sqrt 3) : 
  (1/2) * b * c * Real.sin A = (3 * Real.sqrt 3) / 2 :=
sorry

end angle_A_measure_triangle_area_l228_22814


namespace find_percentage_l228_22820

theorem find_percentage (P N : ℕ) (h₁ : N = 125) (h₂ : N = (P * N / 100) + 105) : P = 16 :=
by
  sorry

end find_percentage_l228_22820


namespace value_is_20_l228_22839

-- Define the conditions
def number : ℕ := 5
def value := number + 3 * number

-- State the theorem
theorem value_is_20 : value = 20 := by
  -- Proof goes here
  sorry

end value_is_20_l228_22839


namespace area_of_room_l228_22835

def length : ℝ := 12
def width : ℝ := 8

theorem area_of_room : length * width = 96 :=
by sorry

end area_of_room_l228_22835


namespace at_least_one_closed_l228_22880

theorem at_least_one_closed {T V : Set ℤ} (hT : T.Nonempty) (hV : V.Nonempty) (h_disjoint : ∀ x, x ∈ T → x ∉ V)
  (h_union : ∀ x, x ∈ T ∨ x ∈ V)
  (hT_closed : ∀ a b c, a ∈ T → b ∈ T → c ∈ T → a * b * c ∈ T)
  (hV_closed : ∀ x y z, x ∈ V → y ∈ V → z ∈ V → x * y * z ∈ V) :
  (∀ a b, a ∈ T → b ∈ T → a * b ∈ T) ∨ (∀ x y, x ∈ V → y ∈ V → x * y ∈ V) := sorry

end at_least_one_closed_l228_22880


namespace length_of_base_of_isosceles_triangle_l228_22851

noncomputable def length_congruent_sides : ℝ := 8
noncomputable def perimeter_triangle : ℝ := 26

theorem length_of_base_of_isosceles_triangle : 
  ∀ (b : ℝ), 
  2 * length_congruent_sides + b = perimeter_triangle → 
  b = 10 :=
by
  intros b h
  -- The proof is omitted.
  sorry

end length_of_base_of_isosceles_triangle_l228_22851


namespace A_share_of_profit_l228_22823

section InvestmentProfit

variables (capitalA capitalB : ℕ) -- initial capitals
variables (withdrawA advanceB : ℕ) -- changes after 8 months
variables (profit : ℕ) -- total profit

def investment_months (initial : ℕ) (final : ℕ) (first_period : ℕ) (second_period : ℕ) : ℕ :=
  initial * first_period + final * second_period

def ratio (a b : ℕ) : ℚ := (a : ℚ) / (b : ℚ)

def A_share (total_profit : ℕ) (ratioA ratioB : ℚ) : ℚ :=
  (ratioA / (ratioA + ratioB)) * total_profit

theorem A_share_of_profit :
  let capitalA := 3000
  let capitalB := 4000
  let withdrawA := 1000
  let advanceB := 1000
  let profit := 756
  let A_investment_months := investment_months capitalA (capitalA - withdrawA) 8 4
  let B_investment_months := investment_months capitalB (capitalB + advanceB) 8 4
  let ratioA := ratio A_investment_months B_investment_months
  let ratioB := ratio B_investment_months A_investment_months
  A_share profit ratioA ratioB = 288 := sorry

end InvestmentProfit

end A_share_of_profit_l228_22823


namespace triangle_area_eq_l228_22815

/-- In a triangle ABC, given that A = arccos(7/8), BC = a, and the altitude from vertex A 
     is equal to the sum of the other two altitudes, show that the area of triangle ABC 
     is (a^2 * sqrt(15)) / 4. -/
theorem triangle_area_eq (a : ℝ) (angle_A : ℝ) (h_angle : angle_A = Real.arccos (7/8))
    (BC : ℝ) (h_BC : BC = a) (H : ∀ (AC AB altitude_A altitude_C altitude_B : ℝ),
    AC = X → AB = Y → 
    altitude_A = (altitude_C + altitude_B) → 
    ∃ (S : ℝ), 
    S = (1/2) * X * Y * Real.sin angle_A ∧ 
    altitude_A = (2 * S / X) + (2 * S / Y) 
    → (X * Y) = 4 * (a^2) 
    → S = ((a^2 * Real.sqrt 15) / 4)) :
S = (a^2 * Real.sqrt 15) / 4 := sorry

end triangle_area_eq_l228_22815


namespace min_oranges_in_new_box_l228_22836

theorem min_oranges_in_new_box (m n : ℕ) (x : ℕ) (h1 : m + n ≤ 60) 
    (h2 : 59 * m = 60 * n + x) : x = 30 :=
sorry

end min_oranges_in_new_box_l228_22836


namespace find_x_value_l228_22855

theorem find_x_value (X : ℕ) 
  (top_left : ℕ := 2)
  (top_second : ℕ := 3)
  (top_last : ℕ := 4)
  (bottom_left : ℕ := 3)
  (bottom_middle : ℕ := 5) 
  (top_sum_eq: 2 + 3 + X + 4 = 9 + X)
  (bottom_sum_eq: 3 + 5 + (X + 1) = 9 + X) : 
  X = 1 := by 
  sorry

end find_x_value_l228_22855


namespace cost_price_article_l228_22888
-- Importing the required library

-- Definition of the problem
theorem cost_price_article
  (C S C_new S_new : ℝ)
  (h1 : S = 1.05 * C)
  (h2 : C_new = 0.95 * C)
  (h3 : S_new = S - 1)
  (h4 : S_new = 1.045 * C) :
  C = 200 :=
by
  -- The proof is omitted
  sorry

end cost_price_article_l228_22888


namespace solve_years_later_twice_age_l228_22801

-- Define the variables and the given conditions
def man_age (S: ℕ) := S + 25
def years_later_twice_age (S M: ℕ) (Y: ℕ) := (M + Y = 2 * (S + Y))

-- Given conditions
def present_age_son := 23
def present_age_man := man_age present_age_son

theorem solve_years_later_twice_age :
  ∃ Y, years_later_twice_age present_age_son present_age_man Y ∧ Y = 2 := by
  sorry

end solve_years_later_twice_age_l228_22801


namespace factorize_expression_1_factorize_expression_2_l228_22890

theorem factorize_expression_1 (m : ℤ) : 
  m^3 - 2 * m^2 - 4 * m + 8 = (m - 2)^2 * (m + 2) := 
sorry

theorem factorize_expression_2 (x y : ℤ) : 
  x^2 - 2 * x * y + y^2 - 9 = (x - y + 3) * (x - y - 3) :=
sorry

end factorize_expression_1_factorize_expression_2_l228_22890


namespace pistachio_shells_percentage_l228_22817

theorem pistachio_shells_percentage (total_pistachios : ℕ) (opened_shelled_pistachios : ℕ) (P : ℝ) :
  total_pistachios = 80 →
  opened_shelled_pistachios = 57 →
  (0.75 : ℝ) * (P / 100) * (total_pistachios : ℝ) = (opened_shelled_pistachios : ℝ) →
  P = 95 :=
by
  intros h_total h_opened h_equation
  sorry

end pistachio_shells_percentage_l228_22817


namespace factorize_polynomial_l228_22897

variable (x : ℝ)

theorem factorize_polynomial : 4 * x^3 - 8 * x^2 + 4 * x = 4 * x * (x - 1)^2 := 
by 
  sorry

end factorize_polynomial_l228_22897


namespace find_a_l228_22895

theorem find_a (f : ℝ → ℝ) (h1 : ∀ x, f (x + 1) = 3 * x + 2) (h2 : f a = 5) : a = 2 :=
sorry

end find_a_l228_22895


namespace minute_hand_coincides_hour_hand_11_times_l228_22879

noncomputable def number_of_coincidences : ℕ := 11

theorem minute_hand_coincides_hour_hand_11_times :
  ∀ (t : ℝ), (0 < t ∧ t < 12) → ∃(n : ℕ), (1 ≤ n ∧ n ≤ 11) ∧ t = (n * 1 + n * (5 / 11)) :=
sorry

end minute_hand_coincides_hour_hand_11_times_l228_22879


namespace sum_of_bases_l228_22850

theorem sum_of_bases (R1 R2 : ℕ)
  (h1 : ∀ F1 : ℚ, F1 = (4 * R1 + 8) / (R1 ^ 2 - 1) → F1 = (5 * R2 + 9) / (R2 ^ 2 - 1))
  (h2 : ∀ F2 : ℚ, F2 = (8 * R1 + 4) / (R1 ^ 2 - 1) → F2 = (9 * R2 + 5) / (R2 ^ 2 - 1)) :
  R1 + R2 = 24 :=
sorry

end sum_of_bases_l228_22850


namespace price_of_light_bulb_and_motor_l228_22831

theorem price_of_light_bulb_and_motor
  (x : ℝ) (motor_price : ℝ)
  (h1 : x + motor_price = 12)
  (h2 : 10 / x = 2 * 45 / (12 - x)) :
  x = 3 ∧ motor_price = 9 :=
sorry

end price_of_light_bulb_and_motor_l228_22831


namespace greatest_multiple_less_150_l228_22878

/-- Define the LCM of two natural numbers -/
def LCM (a b : ℕ) : ℕ := Nat.lcm a b

theorem greatest_multiple_less_150 (x y : ℕ) (h1 : x = 15) (h2 : y = 20) : 
  (∃ m : ℕ, LCM x y * m < 150 ∧ ∀ n : ℕ, LCM x y * n < 150 → LCM x y * n ≤ LCM x y * m) ∧ 
  (∃ m : ℕ, LCM x y * m = 120) :=
by
  sorry

end greatest_multiple_less_150_l228_22878


namespace total_quantities_l228_22871

theorem total_quantities (N : ℕ) (S S₃ S₂ : ℕ)
  (h1 : S = 12 * N)
  (h2 : S₃ = 12)
  (h3 : S₂ = 48)
  (h4 : S = S₃ + S₂) :
  N = 5 :=
by
  sorry

end total_quantities_l228_22871


namespace sum_of_cubes_of_consecutive_numbers_divisible_by_9_l228_22800

theorem sum_of_cubes_of_consecutive_numbers_divisible_by_9 (a : ℕ) (h : a > 1) : 
  9 ∣ ((a - 1)^3 + a^3 + (a + 1)^3) := 
by 
  sorry

end sum_of_cubes_of_consecutive_numbers_divisible_by_9_l228_22800


namespace valid_values_of_X_Y_l228_22867

-- Stating the conditions
def odd_combinations := 125
def even_combinations := 64
def revenue_diff (X Y : ℕ) := odd_combinations * X - even_combinations * Y = 5
def valid_limit (n : ℕ) := 0 < n ∧ n < 250

-- The theorem we want to prove
theorem valid_values_of_X_Y (X Y : ℕ) :
  revenue_diff X Y ∧ valid_limit X ∧ valid_limit Y ↔ (X = 41 ∧ Y = 80) ∨ (X = 105 ∧ Y = 205) :=
  sorry

end valid_values_of_X_Y_l228_22867


namespace probability_is_1_div_28_l228_22803

noncomputable def probability_valid_combinations : ℚ :=
  let total_combinations := Nat.choose 8 3
  let valid_combinations := 2
  valid_combinations / total_combinations

theorem probability_is_1_div_28 :
  probability_valid_combinations = 1 / 28 := by
  sorry

end probability_is_1_div_28_l228_22803


namespace arithmetic_mean_location_l228_22834

theorem arithmetic_mean_location (a b : ℝ) : 
    abs ((a + b) / 2 - a) = abs (b - (a + b) / 2) := 
by 
    sorry

end arithmetic_mean_location_l228_22834


namespace initial_worth_is_30_l228_22821

-- Definitions based on conditions
def numberOfCoinsLeft := 2
def amountLeft := 12

-- Definition of the value of each gold coin based on amount left and number of coins left
def valuePerCoin : ℕ := amountLeft / numberOfCoinsLeft

-- Define the total worth of sold coins
def soldCoinsWorth (coinsSold : ℕ) : ℕ := coinsSold * valuePerCoin

-- The total initial worth of Roman's gold coins
def totalInitialWorth : ℕ := amountLeft + soldCoinsWorth 3

-- The proof goal
theorem initial_worth_is_30 : totalInitialWorth = 30 :=
by
  sorry

end initial_worth_is_30_l228_22821


namespace new_student_info_l228_22843

-- Definitions of the information pieces provided by each classmate.
structure StudentInfo where
  last_name : String
  gender : String
  total_score : Nat
  specialty : String

def student_A : StudentInfo := {
  last_name := "Ji",
  gender := "Male",
  total_score := 260,
  specialty := "Singing"
}

def student_B : StudentInfo := {
  last_name := "Zhang",
  gender := "Female",
  total_score := 220,
  specialty := "Dancing"
}

def student_C : StudentInfo := {
  last_name := "Chen",
  gender := "Male",
  total_score := 260,
  specialty := "Singing"
}

def student_D : StudentInfo := {
  last_name := "Huang",
  gender := "Female",
  total_score := 220,
  specialty := "Drawing"
}

def student_E : StudentInfo := {
  last_name := "Zhang",
  gender := "Female",
  total_score := 240,
  specialty := "Singing"
}

-- The theorem we need to prove based on the given conditions.
theorem new_student_info :
  ∃ info : StudentInfo,
    info.last_name = "Huang" ∧
    info.gender = "Male" ∧
    info.total_score = 240 ∧
    info.specialty = "Dancing" :=
  sorry

end new_student_info_l228_22843


namespace expand_product_l228_22829

theorem expand_product (x : ℝ) :
  (3 * x + 4) * (2 * x - 5) = 6 * x^2 - 7 * x - 20 :=
sorry

end expand_product_l228_22829


namespace greatest_possible_value_l228_22866

theorem greatest_possible_value (x : ℝ) (hx : x^3 + (1 / x^3) = 9) : x + (1 / x) = 3 := by
  sorry

end greatest_possible_value_l228_22866


namespace element_in_set_l228_22874

def M : Set (ℤ × ℤ) := {(1, 2)}

theorem element_in_set : (1, 2) ∈ M :=
by
  sorry

end element_in_set_l228_22874


namespace train_platform_length_l228_22846

theorem train_platform_length (train_length : ℕ) (platform_crossing_time : ℕ) (pole_crossing_time : ℕ) (length_of_platform : ℕ) :
  train_length = 300 →
  platform_crossing_time = 27 →
  pole_crossing_time = 18 →
  ((train_length * platform_crossing_time / pole_crossing_time) = train_length + length_of_platform) →
  length_of_platform = 150 :=
by
  intros h_train_length h_platform_time h_pole_time h_eq
  -- Proof omitted
  sorry

end train_platform_length_l228_22846


namespace smaller_square_area_l228_22887

theorem smaller_square_area (A_L : ℝ) (h : A_L = 100) : ∃ A_S : ℝ, A_S = 50 := 
by
  sorry

end smaller_square_area_l228_22887


namespace min_value_proof_l228_22832

noncomputable def min_value : ℝ := 3 + 2 * Real.sqrt 2

theorem min_value_proof (m n : ℝ) (hm : 0 < m) (hn : 0 < n) (h : m + n = 1) :
  (1 / m + 2 / n) = min_value :=
sorry

end min_value_proof_l228_22832


namespace batsman_average_after_17th_inning_l228_22833

theorem batsman_average_after_17th_inning :
  ∀ (A : ℕ), (16 * A + 50) / 17 = A + 2 → A = 16 → A + 2 = 18 := by
  intros A h1 h2
  rw [h2] at h1
  linarith

end batsman_average_after_17th_inning_l228_22833


namespace farmer_land_l228_22824

theorem farmer_land (initial_land remaining_land : ℚ) (h1 : initial_land - initial_land / 10 = remaining_land) (h2 : remaining_land = 10) : initial_land = 100 / 9 := by
  sorry

end farmer_land_l228_22824


namespace shara_shells_l228_22899

def initial_shells : ℕ := 20
def first_vacation_day1_3 : ℕ := 5 * 3
def first_vacation_day4 : ℕ := 6
def second_vacation_day1_2 : ℕ := 4 * 2
def second_vacation_day3 : ℕ := 7
def third_vacation_day1 : ℕ := 8
def third_vacation_day2 : ℕ := 4
def third_vacation_day3_4 : ℕ := 3 * 2

def total_shells : ℕ :=
  initial_shells + 
  (first_vacation_day1_3 + first_vacation_day4) +
  (second_vacation_day1_2 + second_vacation_day3) + 
  (third_vacation_day1 + third_vacation_day2 + third_vacation_day3_4)

theorem shara_shells : total_shells = 74 :=
by
  sorry

end shara_shells_l228_22899


namespace coordinate_of_M_l228_22860

-- Definition and given conditions
def L : ℚ := 1 / 6
def P : ℚ := 1 / 12

def divides_into_three_equal_parts (L P M N : ℚ) : Prop :=
  M = L + (P - L) / 3 ∧ N = L + 2 * (P - L) / 3

theorem coordinate_of_M (M N : ℚ) 
  (h1 : divides_into_three_equal_parts L P M N) : 
  M = 1 / 9 := 
by 
  sorry
  
end coordinate_of_M_l228_22860


namespace Beth_bought_10_cans_of_corn_l228_22876

theorem Beth_bought_10_cans_of_corn (a b : ℕ) (h1 : b = 15 + 2 * a) (h2 : b = 35) : a = 10 := by
  sorry

end Beth_bought_10_cans_of_corn_l228_22876


namespace rotten_pineapples_l228_22837

theorem rotten_pineapples (initial sold fresh remaining rotten: ℕ) 
  (h1: initial = 86) 
  (h2: sold = 48) 
  (h3: fresh = 29) 
  (h4: remaining = initial - sold) 
  (h5: rotten = remaining - fresh) : 
  rotten = 9 := by 
  sorry

end rotten_pineapples_l228_22837


namespace cookie_percentage_increase_l228_22809

theorem cookie_percentage_increase (cookies_Monday cookies_Tuesday cookies_Wednesday total_cookies : ℕ) 
  (h1 : cookies_Monday = 5)
  (h2 : cookies_Tuesday = 2 * cookies_Monday)
  (h3 : total_cookies = cookies_Monday + cookies_Tuesday + cookies_Wednesday)
  (h4 : total_cookies = 29) :
  (100 * (cookies_Wednesday - cookies_Tuesday) / cookies_Tuesday = 40) := 
by
  sorry

end cookie_percentage_increase_l228_22809


namespace area_one_magnet_is_150_l228_22848

noncomputable def area_one_magnet : ℕ :=
  let length := 15
  let total_circumference := 70
  let combined_width := (total_circumference / 2 - length) / 2
  let width := combined_width
  length * width

theorem area_one_magnet_is_150 :
  area_one_magnet = 150 :=
by
  -- This will skip the actual proof for now
  sorry

end area_one_magnet_is_150_l228_22848


namespace cookout_kids_2004_l228_22856

variable (kids2005 kids2004 kids2006 : ℕ)

theorem cookout_kids_2004 :
  (kids2006 = 20) →
  (2 * kids2005 = 3 * kids2006) →
  (2 * kids2004 = kids2005) →
  kids2004 = 60 :=
by
  intros h1 h2 h3
  sorry

end cookout_kids_2004_l228_22856


namespace points_on_line_l228_22884

-- Define the two points the line connects
def P1 : (ℝ × ℝ) := (8, 10)
def P2 : (ℝ × ℝ) := (2, -2)

-- Define the candidate points
def A : (ℝ × ℝ) := (5, 4)
def E : (ℝ × ℝ) := (1, -4)

-- Define the line equation, given the slope and y-intercept
def line (x : ℝ) : ℝ := 2 * x - 6

theorem points_on_line :
  (A.snd = line A.fst) ∧ (E.snd = line E.fst) :=
by
  sorry

end points_on_line_l228_22884


namespace number_of_divisors_of_n_l228_22811

theorem number_of_divisors_of_n :
  let n : ℕ := (7^3) * (11^2) * (13^4)
  ∃ d : ℕ, d = 60 ∧ ∀ m : ℕ, m ∣ n ↔ ∃ l₁ l₂ l₃ : ℕ, l₁ ≤ 3 ∧ l₂ ≤ 2 ∧ l₃ ≤ 4 ∧ m = 7^l₁ * 11^l₂ * 13^l₃ := 
by
  sorry

end number_of_divisors_of_n_l228_22811
