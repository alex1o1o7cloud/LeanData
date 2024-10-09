import Mathlib

namespace geometric_sequence_fourth_term_l330_33078

theorem geometric_sequence_fourth_term :
  let a₁ := 3^(3/4)
  let a₂ := 3^(2/4)
  let a₃ := 3^(1/4)
  ∃ a₄, a₄ = 1 ∧ a₂ = a₁ * (a₃ / a₂) ∧ a₃ = a₂ * (a₄ / a₃) :=
by
  sorry

end geometric_sequence_fourth_term_l330_33078


namespace sum_ineq_l330_33040

theorem sum_ineq (x y z t : ℝ) (h₁ : x + y + z + t = 0) (h₂ : x^2 + y^2 + z^2 + t^2 = 1) :
  -1 ≤ x * y + y * z + z * t + t * x ∧ x * y + y * z + z * t + t * x ≤ 0 :=
by
  sorry

end sum_ineq_l330_33040


namespace quadratic_decreasing_conditions_l330_33020

theorem quadratic_decreasing_conditions (a : ℝ) :
  (∀ x : ℝ, 2 ≤ x → ∃ y : ℝ, y = ax^2 + 4*(a+1)*x - 3 ∧ (∀ z : ℝ, z ≥ x → y ≥ (ax^2 + 4*(a+1)*z - 3))) ↔ a ∈ Set.Iic (-1 / 2) :=
sorry

end quadratic_decreasing_conditions_l330_33020


namespace total_points_team_l330_33077

def T : ℕ := 4
def J : ℕ := 2 * T + 6
def S : ℕ := J / 2
def R : ℕ := T + J - 3
def A : ℕ := S + R + 4

theorem total_points_team : T + J + S + R + A = 66 := by
  sorry

end total_points_team_l330_33077


namespace polyhedra_impossible_l330_33011

noncomputable def impossible_polyhedra_projections (p1_outer : List (ℝ × ℝ)) (p1_inner : List (ℝ × ℝ))
                                                  (p2_outer : List (ℝ × ℝ)) (p2_inner : List (ℝ × ℝ)) : Prop :=
  -- Add definitions for the vertices labeling here 
  let vertices_outer := ["A", "B", "C", "D"]
  let vertices_inner := ["A1", "B1", "C1", "D1"]
  -- Add the conditions for projection (a) and (b) 
  p1_outer = [(0,0), (1,0), (1,1), (0,1)] ∧
  p1_inner = [(0.25,0.25), (0.75,0.25), (0.75,0.75), (0.25,0.75)] ∧
  p2_outer = [(0,0), (1,0), (1,1), (0,1)] ∧
  p2_inner = [(0.25,0.25), (0.75,0.25), (0.75,0.75), (0.25,0.75)] →
  -- Prove that the polyhedra corresponding to these projections are impossible.
  false

-- Now let's state the theorem
theorem polyhedra_impossible : impossible_polyhedra_projections [(0,0), (1,0), (1,1), (0,1)] 
                                                                [(0.25,0.25), (0.75,0.25), (0.75,0.75), (0.25,0.75)]
                                                                [(0,0), (1,0), (1,1), (0,1)]
                                                                [(0.25,0.25), (0.75,0.25), (0.75,0.75), (0.25,0.75)] := 
by {
  sorry
}

end polyhedra_impossible_l330_33011


namespace area_under_the_curve_l330_33058

theorem area_under_the_curve : 
  ∫ x in (0 : ℝ)..1, (x^2 + 1) = 4 / 3 := 
by
  sorry

end area_under_the_curve_l330_33058


namespace subtraction_of_fractions_l330_33046

theorem subtraction_of_fractions :
  1 + 1 / 2 - 3 / 5 = 9 / 10 := by
  sorry

end subtraction_of_fractions_l330_33046


namespace find_k_for_perfect_square_l330_33044

theorem find_k_for_perfect_square :
  ∃ k : ℤ, (k = 12 ∨ k = -12) ∧ (∀ n : ℤ, ∃ a b : ℤ, 4 * n^2 + k * n + 9 = (a * n + b)^2) :=
sorry

end find_k_for_perfect_square_l330_33044


namespace square_difference_l330_33073

theorem square_difference (x y : ℝ) 
  (h₁ : (x + y)^2 = 36) 
  (h₂ : x * y = 8) : 
  (x - y)^2 = 4 :=
by
  sorry

end square_difference_l330_33073


namespace total_boxes_l330_33007

variable (N_initial : ℕ) (N_nonempty : ℕ) (N_new_boxes : ℕ)

theorem total_boxes (h_initial : N_initial = 7) 
                     (h_nonempty : N_nonempty = 10)
                     (h_new_boxes : N_new_boxes = N_nonempty * 7) :
  N_initial + N_new_boxes = 77 :=
by 
  have : N_initial = 7 := h_initial
  have : N_new_boxes = N_nonempty * 7 := h_new_boxes
  have : N_nonempty = 10 := h_nonempty
  sorry

end total_boxes_l330_33007


namespace prism_volume_l330_33064

open Real

theorem prism_volume :
  ∃ (a b c : ℝ), a * b = 15 ∧ b * c = 10 ∧ c * a = 30 ∧ a * b * c = 30 * sqrt 5 :=
by
  sorry

end prism_volume_l330_33064


namespace flower_bouquet_violets_percentage_l330_33012

theorem flower_bouquet_violets_percentage
  (total_flowers yellow_flowers purple_flowers : ℕ)
  (yellow_daisies yellow_tulips purple_violets : ℕ)
  (h_yellow_flowers : yellow_flowers = (total_flowers / 2))
  (h_purple_flowers : purple_flowers = (total_flowers / 2))
  (h_yellow_daisies : yellow_daisies = (yellow_flowers / 5))
  (h_yellow_tulips : yellow_tulips = yellow_flowers - yellow_daisies)
  (h_purple_violets : purple_violets = (purple_flowers / 2)) :
  ((purple_violets : ℚ) / total_flowers) * 100 = 25 :=
by
  sorry

end flower_bouquet_violets_percentage_l330_33012


namespace sugar_required_in_new_recipe_l330_33085

theorem sugar_required_in_new_recipe
  (ratio_flour_water_sugar : ℕ × ℕ × ℕ)
  (double_ratio_flour_water : (ℕ → ℕ))
  (half_ratio_flour_sugar : (ℕ → ℕ))
  (new_water_cups : ℕ) :
  ratio_flour_water_sugar = (7, 2, 1) →
  double_ratio_flour_water 7 = 14 → 
  double_ratio_flour_water 2 = 4 →
  half_ratio_flour_sugar 7 = 7 →
  half_ratio_flour_sugar 1 = 2 →
  new_water_cups = 2 →
  (∃ sugar_cups : ℕ, sugar_cups = 1) :=
by
  sorry

end sugar_required_in_new_recipe_l330_33085


namespace percent_increase_decrease_l330_33083

theorem percent_increase_decrease (P y : ℝ) (h : (P * (1 + y / 100) * (1 - y / 100) = 0.90 * P)) :
    y = 31.6 :=
by
  sorry

end percent_increase_decrease_l330_33083


namespace solve_problem_l330_33019

theorem solve_problem (a : ℝ) (x : ℝ) (h1 : 3 * x + |a - 2| = -3) (h2 : 3 * x + 4 = 0) :
  (a = 3 ∨ a = 1) → ((a - 2) ^ 2010 - 2 * a + 1 = -4 ∨ (a - 2) ^ 2010 - 2 * a + 1 = 0) :=
by {
  sorry
}

end solve_problem_l330_33019


namespace parts_processed_per_day_l330_33071

-- Given conditions
variable (a : ℕ)

-- Goal: Prove the daily productivity of Master Wang given the conditions
theorem parts_processed_per_day (h1 : ∀ n, n = 8) (h2 : ∃ m, m = a + 3):
  (a + 3) / 8 = (a + 3) / 8 :=
by
  sorry

end parts_processed_per_day_l330_33071


namespace min_value_of_fraction_l330_33061

theorem min_value_of_fraction (m n : ℝ) (h1 : 2 * n + m = 4) (h2 : m > 0) (h3 : n > 0) : 
  (∀ n m, 2 * n + m = 4 ∧ m > 0 ∧ n > 0 → ∀ y, y = 2 / m + 1 / n → y ≥ 2) :=
by sorry

end min_value_of_fraction_l330_33061


namespace initial_percentage_decrease_l330_33038

theorem initial_percentage_decrease (x : ℝ) (P : ℝ) (h₀ : P > 0)
  (initial_decrease : ∀ (x : ℝ), P * (1 - x / 100) * 1.3 = P * 1.04) :
  x = 20 :=
by 
  sorry

end initial_percentage_decrease_l330_33038


namespace zero_in_interval_l330_33021

theorem zero_in_interval (a b : ℝ) (ha : 1 < a) (hb : 0 < b ∧ b < 1) :
  ∃ x : ℝ, -1 < x ∧ x < 0 ∧ (a^x + x - b = 0) :=
by {
  sorry
}

end zero_in_interval_l330_33021


namespace midpoint_one_seventh_one_ninth_l330_33052

theorem midpoint_one_seventh_one_ninth : 
  let a := (1 : ℚ) / 7
  let b := (1 : ℚ) / 9
  (a + b) / 2 = 8 / 63 := 
by
  sorry

end midpoint_one_seventh_one_ninth_l330_33052


namespace cost_of_each_item_l330_33001

theorem cost_of_each_item (initial_order items : ℕ) (price per_item_reduction additional_orders : ℕ) (reduced_order total_order reduced_price profit_per_item : ℕ) 
  (h1 : initial_order = 60)
  (h2 : price = 100)
  (h3 : per_item_reduction = 1)
  (h4 : additional_orders = 3)
  (h5 : reduced_price = price - price * 4 / 100)
  (h6 : total_order = initial_order + additional_orders * (price * 4 / 100))
  (h7 : reduced_order = total_order)
  (h8 : profit_per_item = price - per_item_reduction )
  (h9 : profit_per_item = 24)
  (h10 : items * profit_per_item = reduced_order * (profit_per_item - per_item_reduction)) :
  (price - profit_per_item = 76) :=
by sorry

end cost_of_each_item_l330_33001


namespace amount_transferred_l330_33032

def original_balance : ℕ := 27004
def remaining_balance : ℕ := 26935

theorem amount_transferred : original_balance - remaining_balance = 69 :=
by
  sorry

end amount_transferred_l330_33032


namespace women_count_l330_33069

/-- 
Initially, the men and women in a room were in the ratio of 4:5.
Then, 2 men entered the room and 3 women left the room.
The number of women then doubled.
There are now 14 men in the room.
Prove that the number of women currently in the room is 24.
-/
theorem women_count (x : ℕ) (h1 : 4 * x + 2 = 14) (h2 : 2 * (5 * x - 3) = n) : 
  n = 24 :=
by
  sorry

end women_count_l330_33069


namespace agatha_bike_budget_l330_33008

def total_initial : ℕ := 60
def cost_frame : ℕ := 15
def cost_front_wheel : ℕ := 25
def total_spent : ℕ := cost_frame + cost_front_wheel
def total_left : ℕ := total_initial - total_spent

theorem agatha_bike_budget : total_left = 20 := by
  sorry

end agatha_bike_budget_l330_33008


namespace calculate_expression_l330_33010

theorem calculate_expression : (3072 - 2993) ^ 2 / 121 = 49 :=
by
  sorry

end calculate_expression_l330_33010


namespace perimeter_original_rectangle_l330_33041

variable {L W : ℕ}

axiom area_original : L * W = 360
axiom area_changed : (L + 10) * (W - 6) = 360

theorem perimeter_original_rectangle : 2 * (L + W) = 76 :=
by
  sorry

end perimeter_original_rectangle_l330_33041


namespace p_sufficient_not_necessary_for_q_l330_33036

def p (x : ℝ) : Prop := 1 ≤ x ∧ x ≤ 3
def q (x : ℝ) : Prop := x ≠ 0

theorem p_sufficient_not_necessary_for_q :
  (∀ x, p x → q x) ∧ ¬(∀ x, q x → p x) :=
by
  sorry

end p_sufficient_not_necessary_for_q_l330_33036


namespace calculate_fraction_l330_33084

-- Define the fractions we are working with
def fraction1 : ℚ := 3 / 4
def fraction2 : ℚ := 15 / 5
def one_half : ℚ := 1 / 2

-- Define the main calculation
def main_fraction (f1 f2 one_half : ℚ) : ℚ := f1 * f2 - one_half

-- State the theorem
theorem calculate_fraction : main_fraction fraction1 fraction2 one_half = (7 / 4) := by
  sorry

end calculate_fraction_l330_33084


namespace car_selling_price_l330_33082

def car_material_cost : ℕ := 100
def car_production_per_month : ℕ := 4
def motorcycle_material_cost : ℕ := 250
def motorcycles_sold_per_month : ℕ := 8
def motorcycle_selling_price : ℤ := 50
def additional_motorcycle_profit : ℤ := 50

theorem car_selling_price (x : ℤ) :
  (motorcycles_sold_per_month * motorcycle_selling_price - motorcycle_material_cost)
  = (car_production_per_month * x - car_material_cost + additional_motorcycle_profit) →
  x = 50 :=
by
  sorry

end car_selling_price_l330_33082


namespace min_value_when_a_is_negative_one_max_value_bounds_l330_33030

-- Conditions
def f (a x : ℝ) : ℝ := a * x^2 + x
def a1 : ℝ := -1
def a : ℝ := -2
def a_lower_bound : ℝ := -2
def a_upper_bound : ℝ := 0
def interval : Set ℝ := Set.Icc 0 2

-- Part I: Minimum value when a = -1
theorem min_value_when_a_is_negative_one : 
  ∃ x ∈ interval, f a1 x = -2 := 
by
  sorry

-- Part II: Maximum value criterions
theorem max_value_bounds (a : ℝ) (H : a ∈ Set.Icc a_lower_bound a_upper_bound) :
  (∀ x ∈ interval, 
    (a ≥ -1/4 → f a ( -1 / (2 * a) ) = -1 / (4 * a)) 
    ∧ (a < -1/4 → f a 2 = 4 * a + 2 )) :=
by
  sorry

end min_value_when_a_is_negative_one_max_value_bounds_l330_33030


namespace cory_chairs_l330_33088

theorem cory_chairs (total_cost table_cost chair_cost C : ℕ) (h1 : total_cost = 135) (h2 : table_cost = 55) (h3 : chair_cost = 20) (h4 : total_cost = table_cost + chair_cost * C) : C = 4 := 
by 
  sorry

end cory_chairs_l330_33088


namespace exists_factorial_with_first_digits_2015_l330_33005

theorem exists_factorial_with_first_digits_2015 : ∃ n : ℕ, n > 0 ∧ (∃ k : ℕ, 2015 * (10^k) ≤ n! ∧ n! < 2016 * (10^k)) :=
sorry

end exists_factorial_with_first_digits_2015_l330_33005


namespace right_triangle_legs_l330_33045

theorem right_triangle_legs (m r x y : ℝ) 
  (h1 : m^2 = x^2 + y^2) 
  (h2 : r = (x + y - m) / 2) 
  (h3 : r ≤ m * (Real.sqrt 2 - 1) / 2) : 
  (x = (2 * r + m + Real.sqrt (m^2 - 4 * r^2 - 4 * r * m)) / 2) ∧ 
  (y = (2 * r + m - Real.sqrt (m^2 - 4 * r^2 - 4 * r * m)) / 2) :=
by 
  sorry

end right_triangle_legs_l330_33045


namespace necessary_but_not_sufficient_condition_l330_33098

-- Definitions
def represents_ellipse (m n : ℝ) (x y : ℝ) : Prop := 
  (x^2 / m + y^2 / n = 1)

-- Main theorem statement
theorem necessary_but_not_sufficient_condition 
    (m n x y : ℝ) (h_mn_pos : m * n > 0) :
    (represents_ellipse m n x y) → 
    (m ≠ n ∧ m > 0 ∧ n > 0 ∧ represents_ellipse m n x y) → 
    (m * n > 0) ∧ ¬(
    ∀ m n : ℝ, (m ≠ n ∧ m > 0 ∧ n > 0) →
    represents_ellipse m n x y
    ) :=
by
  sorry

end necessary_but_not_sufficient_condition_l330_33098


namespace find_angle_C_find_side_c_l330_33006

variable {A B C a b c : ℝ}
variable {AD CD area_ABD : ℝ}

-- Conditions for question 1
variable (h1 : c * Real.tan C = Real.sqrt 3 * (a * Real.cos B + b * Real.cos A))

-- Conditions for question 2
variable (h2 : AD = 4)
variable (h3 : CD = 4)
variable (h4 : area_ABD = 8 * Real.sqrt 3)
variable (h5 : C = Real.pi / 3)

-- Lean 4 statement for both parts of the problem
theorem find_angle_C (h1 : c * Real.tan C = Real.sqrt 3 * (a * Real.cos B + b * Real.cos A)) : 
  C = Real.pi / 3 :=
sorry

theorem find_side_c (h2 : AD = 4) (h3 : CD = 4) (h4 : area_ABD = 8 * Real.sqrt 3) (h5 : C = Real.pi / 3) : 
  c = 4 * Real.sqrt 7 :=
sorry

end find_angle_C_find_side_c_l330_33006


namespace identify_conic_section_is_hyperbola_l330_33051

theorem identify_conic_section_is_hyperbola :
  ∀ x y : ℝ, x^2 - 16 * y^2 - 10 * x + 4 * y + 36 = 0 →
  (∃ a b h c d k : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ h = 0 ∧ (x - c)^2 / a^2 - (y - d)^2 / b^2 = k) :=
by
  sorry

end identify_conic_section_is_hyperbola_l330_33051


namespace find_Finley_age_l330_33028

variable (Roger Jill Finley : ℕ)
variable (Jill_age : Jill = 20)
variable (Roger_age : Roger = 2 * Jill + 5)
variable (Finley_condition : 15 + (Roger - Jill) = Finley - 30)

theorem find_Finley_age : Finley = 55 :=
by
  sorry

end find_Finley_age_l330_33028


namespace suma_work_rate_l330_33026

theorem suma_work_rate (r s : ℝ) (hr : r = 1 / 5) (hrs : r + s = 1 / 4) : 1 / s = 20 := by
  sorry

end suma_work_rate_l330_33026


namespace max_perimeter_triangle_l330_33065

theorem max_perimeter_triangle (y : ℤ) (h1 : y < 16) (h2 : y > 2) : 
    7 + 9 + y = 31 → y = 15 := by
  sorry

end max_perimeter_triangle_l330_33065


namespace cube_identity_simplification_l330_33003

theorem cube_identity_simplification (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (h : x + y + z = 0) :
  (x^3 + y^3 + z^3 + 3 * x * y * z) / (x * y * z) = 6 :=
by
  sorry

end cube_identity_simplification_l330_33003


namespace meal_combinations_l330_33053

theorem meal_combinations (n : ℕ) (h : n = 12) : ∃ m : ℕ, m = 132 :=
by
  -- Initialize the variables for dishes chosen by Yann and Camille
  let yann_choices := n
  let camille_choices := n - 1
  
  -- Calculate the total number of combinations
  let total_combinations := yann_choices * camille_choices
  
  -- Assert the number of combinations is equal to 132
  use total_combinations
  exact sorry

end meal_combinations_l330_33053


namespace consecutive_numbers_product_l330_33048

theorem consecutive_numbers_product (a b c d : ℤ) 
  (h1 : b = a + 1) 
  (h2 : c = a + 2) 
  (h3 : d = a + 3) 
  (h4 : a + d = 109) : 
  b * c = 2970 := by
  sorry

end consecutive_numbers_product_l330_33048


namespace value_of_expression_l330_33089

theorem value_of_expression : 4 * (8 - 6) - 7 = 1 := by
  -- Calculation steps would go here
  sorry

end value_of_expression_l330_33089


namespace expected_digits_fair_icosahedral_die_l330_33035

noncomputable def expected_number_of_digits : ℝ :=
  let one_digit_count := 9
  let two_digit_count := 11
  let total_faces := 20
  let prob_one_digit := one_digit_count / total_faces
  let prob_two_digit := two_digit_count / total_faces
  (prob_one_digit * 1) + (prob_two_digit * 2)

theorem expected_digits_fair_icosahedral_die :
  expected_number_of_digits = 1.55 :=
by
  sorry

end expected_digits_fair_icosahedral_die_l330_33035


namespace michael_max_correct_answers_l330_33096

theorem michael_max_correct_answers (c w b : ℕ) 
  (h1 : c + w + b = 30) 
  (h2 : 4 * c - 3 * w = 72) : 
  c ≤ 21 := 
sorry

end michael_max_correct_answers_l330_33096


namespace shifted_sine_odd_function_l330_33081

theorem shifted_sine_odd_function (ϕ : ℝ) (hϕ : 0 < ϕ ∧ ϕ < π) :
  ∃ k : ℤ, ϕ = (2 * π / 3) + k * π ∧ 0 < (2 * π / 3) + k * π ∧ (2 * π / 3) + k * π < π :=
sorry

end shifted_sine_odd_function_l330_33081


namespace total_campers_rowing_and_hiking_l330_33024

def campers_morning_rowing : ℕ := 41
def campers_morning_hiking : ℕ := 4
def campers_afternoon_rowing : ℕ := 26

theorem total_campers_rowing_and_hiking :
  campers_morning_rowing + campers_morning_hiking + campers_afternoon_rowing = 71 :=
by
  -- We are skipping the proof since instructions specify only the statement is needed
  sorry

end total_campers_rowing_and_hiking_l330_33024


namespace eggs_remaining_l330_33009

-- Assign the given constants
def hens : ℕ := 3
def eggs_per_hen_per_day : ℕ := 3
def days_gone : ℕ := 7
def eggs_taken_by_neighbor : ℕ := 12
def eggs_dropped_by_myrtle : ℕ := 5

-- Calculate the expected number of eggs Myrtle should have
noncomputable def total_eggs :=
  hens * eggs_per_hen_per_day * days_gone - eggs_taken_by_neighbor - eggs_dropped_by_myrtle

-- Prove that the total number of eggs equals the correct answer
theorem eggs_remaining : total_eggs = 46 :=
by
  sorry

end eggs_remaining_l330_33009


namespace ratio_of_b_to_sum_a_c_l330_33087

theorem ratio_of_b_to_sum_a_c (a b c : ℕ) (h1 : a + b + c = 60) (h2 : a = 1/3 * (b + c)) (h3 : c = 35) : b = 1/5 * (a + c) :=
by
  sorry

end ratio_of_b_to_sum_a_c_l330_33087


namespace jellybeans_in_jar_now_l330_33068

def initial_jellybeans : ℕ := 90
def samantha_takes : ℕ := 24
def shelby_takes : ℕ := 12
def scarlett_takes : ℕ := 2 * shelby_takes
def scarlett_returns : ℕ := scarlett_takes / 2
def shannon_refills : ℕ := (samantha_takes + shelby_takes) / 2

theorem jellybeans_in_jar_now : 
  initial_jellybeans 
  - samantha_takes 
  - shelby_takes 
  + scarlett_returns
  + shannon_refills 
  = 84 := by
  sorry

end jellybeans_in_jar_now_l330_33068


namespace sum_modulo_9_l330_33094

theorem sum_modulo_9 :
  (2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 + 99999999) % 9 = 6 := 
by
  -- Skipping the detailed proof steps
  sorry

end sum_modulo_9_l330_33094


namespace min_green_beads_l330_33016

theorem min_green_beads (B R G : ℕ) (h : B + R + G = 80)
  (hB : ∀ i j : ℕ, (i < j ∧ j ≤ B → ∃ k, i < k ∧ k < j ∧ k ≤ R)) 
  (hR : ∀ i j : ℕ, (i < j ∧ j ≤ R → ∃ k, i < k ∧ k < j ∧ k ≤ G)) :
  G >= 27 := 
sorry

end min_green_beads_l330_33016


namespace probability_of_two_pairs_of_same_value_is_correct_l330_33067

def total_possible_outcomes := 6^6
def number_of_ways_to_form_pairs := 15
def choose_first_pair := 6
def choose_second_pair := 15
def choose_third_pair := 6
def choose_fourth_die := 4
def choose_fifth_die := 3

def successful_outcomes := number_of_ways_to_form_pairs *
                           choose_first_pair *
                           choose_second_pair *
                           choose_third_pair *
                           choose_fourth_die *
                           choose_fifth_die

def probability_of_two_pairs_of_same_value := (successful_outcomes : ℚ) / total_possible_outcomes

theorem probability_of_two_pairs_of_same_value_is_correct :
  probability_of_two_pairs_of_same_value = 25 / 72 :=
by
  -- proof omitted
  sorry

end probability_of_two_pairs_of_same_value_is_correct_l330_33067


namespace parabola_equivalence_l330_33076

theorem parabola_equivalence :
  ∃ (a : ℝ) (h k : ℝ),
    (a = -3 ∧ h = -1 ∧ k = 2) ∧
    ∀ (x : ℝ), (y = -3 * x^2 + 1) → (y = -3 * (x + 1)^2 + 2) :=
sorry

end parabola_equivalence_l330_33076


namespace tadpole_catch_l330_33031

variable (T : ℝ) (H1 : T * 0.25 = 45)

theorem tadpole_catch (T : ℝ) (H1 : T * 0.25 = 45) : T = 180 :=
sorry

end tadpole_catch_l330_33031


namespace range_of_a_l330_33002

noncomputable def set_A (a : ℝ) : Set ℝ := {x | x < a}
noncomputable def set_B : Set ℝ := {x | 1 < x ∧ x < 2}
noncomputable def complement_B : Set ℝ := {x | x ≤ 1 ∨ x ≥ 2 }

theorem range_of_a (a : ℝ) : (set_A a ∪ complement_B) = Set.univ ↔ 2 ≤ a := 
by 
  sorry

end range_of_a_l330_33002


namespace clothing_prices_and_purchase_plans_l330_33054

theorem clothing_prices_and_purchase_plans :
  ∃ (x y : ℕ) (a : ℤ), 
  x + y = 220 ∧
  6 * x = 5 * y ∧
  120 * a + 100 * (150 - a) ≤ 17000 ∧
  (90 ≤ a ∧ a ≤ 100) ∧
  x = 100 ∧
  y = 120 ∧
  (∀ b : ℤ, (90 ≤ b ∧ b ≤ 100) → 120 * b + 100 * (150 - b) ≥ 16800)
  :=
sorry

end clothing_prices_and_purchase_plans_l330_33054


namespace find_blue_shirts_l330_33004

-- Statements of the problem conditions
def total_shirts : ℕ := 23
def green_shirts : ℕ := 17

-- Definition that we want to prove
def blue_shirts : ℕ := total_shirts - green_shirts

-- Proof statement (no need to include the proof itself)
theorem find_blue_shirts : blue_shirts = 6 := by
  sorry

end find_blue_shirts_l330_33004


namespace number_of_books_about_trains_l330_33057

theorem number_of_books_about_trains
  (books_animals : ℕ)
  (books_outer_space : ℕ)
  (book_cost : ℕ)
  (total_spent : ℕ)
  (T : ℕ)
  (hyp1 : books_animals = 8)
  (hyp2 : books_outer_space = 6)
  (hyp3 : book_cost = 6)
  (hyp4 : total_spent = 102)
  (hyp5 : total_spent = (books_animals + books_outer_space + T) * book_cost)
  : T = 3 := by
  sorry

end number_of_books_about_trains_l330_33057


namespace crews_complete_job_l330_33018

-- Define the productivity rates for each crew
variables (x y z : ℝ)

-- Define the conditions derived from the problem
def condition1 : Prop := 1/(x + y) = 1/z - 3/5
def condition2 : Prop := 1/(x + z) = 1/y
def condition3 : Prop := 1/(y + z) = 2/(7 * x)

-- Target proof: the combined time for all three crews
def target_proof : Prop := 1/(x + y + z) = 4/3

-- Final Lean 4 statement combining all conditions and proof requirement
theorem crews_complete_job (x y z : ℝ) (h1 : condition1 x y z) (h2 : condition2 x y z) (h3 : condition3 x y z) : target_proof x y z :=
sorry

end crews_complete_job_l330_33018


namespace basketball_free_throws_l330_33042

theorem basketball_free_throws
  (a b x : ℕ)
  (h1 : 3 * b = 2 * a)
  (h2 : x = 2 * a)
  (h3 : 2 * a + 3 * b + x = 72)
  : x = 24 := by
  sorry

end basketball_free_throws_l330_33042


namespace fewest_handshakes_is_zero_l330_33086

noncomputable def fewest_handshakes (n k : ℕ) : ℕ :=
  if h : (n * (n - 1)) / 2 + k = 325 then k else 325

theorem fewest_handshakes_is_zero :
  ∃ n k : ℕ, (n * (n - 1)) / 2 + k = 325 ∧ 0 = fewest_handshakes n k :=
by
  sorry

end fewest_handshakes_is_zero_l330_33086


namespace nap_time_left_l330_33091

def train_ride_duration : ℕ := 9
def reading_time : ℕ := 2
def eating_time : ℕ := 1
def watching_movie_time : ℕ := 3

theorem nap_time_left :
  train_ride_duration - (reading_time + eating_time + watching_movie_time) = 3 :=
by
  -- Insert proof here
  sorry

end nap_time_left_l330_33091


namespace sports_club_membership_l330_33079

theorem sports_club_membership :
  (17 + 21 - 10 + 2 = 30) :=
by
  sorry

end sports_club_membership_l330_33079


namespace number_of_pairs_l330_33022

theorem number_of_pairs (n : ℕ) (h : n = 2835) :
  ∃ (count : ℕ), count = 20 ∧
  (∀ (x y : ℕ), (0 < x ∧ 0 < y ∧ x < y ∧ (x^2 + y^2) % (x + y) = 0 ∧ (x^2 + y^2) / (x + y) ∣ n) → count = 20) := 
sorry

end number_of_pairs_l330_33022


namespace chantel_bracelets_at_end_l330_33062

-- Definitions based on conditions
def bracelets_day1 := 4
def days1 := 7
def given_away1 := 8

def bracelets_day2 := 5
def days2 := 10
def given_away2 := 12

-- Computation based on conditions
def total_bracelets := days1 * bracelets_day1 - given_away1 + days2 * bracelets_day2 - given_away2

-- The proof statement
theorem chantel_bracelets_at_end : total_bracelets = 58 := by
  sorry

end chantel_bracelets_at_end_l330_33062


namespace mans_speed_against_current_l330_33025

theorem mans_speed_against_current
  (speed_with_current : ℝ)
  (speed_of_current : ℝ)
  (h1 : speed_with_current = 25)
  (h2 : speed_of_current = 2.5) :
  speed_with_current - 2 * speed_of_current = 20 := 
by
  sorry

end mans_speed_against_current_l330_33025


namespace percent_of_x_is_y_l330_33043

variables (x y : ℝ)

theorem percent_of_x_is_y (h : 0.30 * (x - y) = 0.20 * (x + y)) : y = 0.20 * x :=
by sorry

end percent_of_x_is_y_l330_33043


namespace solve_inequality_l330_33090

theorem solve_inequality (x : ℝ) : (x^2 - 50 * x + 625 ≤ 25) = (20 ≤ x ∧ x ≤ 30) :=
sorry

end solve_inequality_l330_33090


namespace value_of_six_inch_cube_l330_33013

-- Defining the conditions
def original_cube_weight : ℝ := 5 -- in pounds
def original_cube_value : ℝ := 600 -- in dollars
def original_cube_side : ℝ := 4 -- in inches

def new_cube_side : ℝ := 6 -- in inches

def cube_volume (side_length : ℝ) : ℝ := side_length ^ 3

-- Statement of the theorem
theorem value_of_six_inch_cube :
  cube_volume new_cube_side / cube_volume original_cube_side * original_cube_value = 2025 :=
by
  -- Here goes the proof
  sorry

end value_of_six_inch_cube_l330_33013


namespace keegan_total_school_time_l330_33037

-- Definition of the conditions
def keegan_classes : Nat := 7
def history_and_chemistry_time : ℝ := 1.5
def other_class_time : ℝ := 1.2

-- The theorem stating that given these conditions, Keegan spends 7.5 hours a day in school.
theorem keegan_total_school_time : 
  (history_and_chemistry_time + 5 * other_class_time) = 7.5 := 
by
  sorry

end keegan_total_school_time_l330_33037


namespace find_OC_l330_33039

noncomputable section

open Real

structure Point where
  x : ℝ
  y : ℝ

def OA (A : Point) : ℝ := sqrt (A.x^2 + A.y^2)
def OB (B : Point) : ℝ := sqrt (B.x^2 + B.y^2)
def OD (D : Point) : ℝ := sqrt (D.x^2 + D.y^2)
def ratio_of_lengths (A B : Point) : ℝ := OA A / OB B

def find_D (A B : Point) : Point :=
  let ratio := ratio_of_lengths A B
  { x := (A.x + ratio * B.x) / (1 + ratio),
    y := (A.y + ratio * B.y) / (1 + ratio) }

-- Given conditions
def A : Point := ⟨0, 1⟩
def B : Point := ⟨-3, 4⟩
def C_magnitude : ℝ := 2

-- Goal to prove
theorem find_OC : Point :=
  let D := find_D A B
  let D_length := OD D
  let scale := C_magnitude / D_length
  { x := D.x * scale,
    y := D.y * scale }

example : find_OC = ⟨-sqrt 10 / 5, 3 * sqrt 10 / 5⟩ := by
  sorry

end find_OC_l330_33039


namespace exists_perfect_square_intersection_l330_33049

theorem exists_perfect_square_intersection : ∃ n : ℕ, n > 1 ∧ ∃ k : ℕ, (2^n - n) = k^2 :=
by sorry

end exists_perfect_square_intersection_l330_33049


namespace laura_rental_cost_l330_33060

def rental_cost_per_day : ℝ := 30
def driving_cost_per_mile : ℝ := 0.25
def days_rented : ℝ := 3
def miles_driven : ℝ := 300

theorem laura_rental_cost : rental_cost_per_day * days_rented + driving_cost_per_mile * miles_driven = 165 := by
  sorry

end laura_rental_cost_l330_33060


namespace num_fish_when_discovered_l330_33017

open Nat

/-- Definition of the conditions given in the problem --/
def initial_fish := 60
def fish_per_day_eaten := 2
def additional_fish := 8
def weeks_before_addition := 2
def extra_week := 1

/-- The proof problem statement --/
theorem num_fish_when_discovered : 
  let days := (weeks_before_addition + extra_week) * 7
  let total_fish_eaten := days * fish_per_day_eaten
  let fish_after_addition := initial_fish + additional_fish
  let final_fish := fish_after_addition - total_fish_eaten
  final_fish = 26 := 
by
  let days := (weeks_before_addition + extra_week) * 7
  let total_fish_eaten := days * fish_per_day_eaten
  let fish_after_addition := initial_fish + additional_fish
  let final_fish := fish_after_addition - total_fish_eaten
  have h : final_fish = 26 := sorry
  exact h

end num_fish_when_discovered_l330_33017


namespace fox_cub_distribution_l330_33072

variable (m a x y : ℕ)
-- Assuming the system of equations given in the problem:
def fox_cub_system_of_equations (n : ℕ) : Prop :=
  ∀ (k : ℕ), 1 ≤ k ∧ k ≤ n →
    ((k * (m - 1) * a + x) = ((m + k - 1) * y))

theorem fox_cub_distribution (m a x y : ℕ) (h : fox_cub_system_of_equations m a x y n) :
  y = ((m-1) * a) ∧ x = ((m-1)^2 * a) :=
by
  sorry

end fox_cub_distribution_l330_33072


namespace quadratic_inequality_l330_33050

theorem quadratic_inequality (a : ℝ) (h : ∀ x : ℝ, a * x^2 + 2 * x + 1 > 0) : a ≥ 1 :=
sorry

end quadratic_inequality_l330_33050


namespace last_digit_of_3_pow_2012_l330_33095

-- Theorem: The last digit of 3^2012 is 1 given the cyclic pattern of last digits for powers of 3.
theorem last_digit_of_3_pow_2012 : (3 ^ 2012) % 10 = 1 :=
by
  sorry

end last_digit_of_3_pow_2012_l330_33095


namespace sum_of_ages_l330_33097

variable (S F : ℕ)

theorem sum_of_ages (h1 : F - 18 = 3 * (S - 18)) (h2 : F = 2 * S) : S + F = 108 := by
  sorry

end sum_of_ages_l330_33097


namespace remainder_of_polynomial_l330_33023

theorem remainder_of_polynomial :
  ∀ (x : ℂ), (x^4 + x^3 + x^2 + x + 1 = 0) → (x^55 + x^44 + x^33 + x^22 + x^11 + 1) % (x^4 + x^3 + x^2 + x + 1) = 2 :=
by
  intro x hx
  sorry

end remainder_of_polynomial_l330_33023


namespace find_annual_interest_rate_l330_33093

theorem find_annual_interest_rate (A P : ℝ) (n t : ℕ) (r : ℝ) :
  A = P * (1 + r / n)^(n * t) →
  A = 5292 →
  P = 4800 →
  n = 1 →
  t = 2 →
  r = 0.05 :=
by
  intros h₁ h₂ h₃ h₄ h₅
  sorry

end find_annual_interest_rate_l330_33093


namespace monthly_average_growth_rate_optimal_selling_price_for_desired_profit_l330_33063

/- Part 1: Monthly Average Growth Rate -/
theorem monthly_average_growth_rate (m : ℝ) (sale_april sale_june : ℝ) (h_apr_val : sale_april = 256) (h_june_val : sale_june = 400) :
  256 * (1 + m) ^ 2 = 400 → m = 0.25 :=
sorry

/- Part 2: Optimal Selling Price for Desired Profit -/
theorem optimal_selling_price_for_desired_profit (y : ℝ) (initial_price selling_price : ℝ) (sale_june : ℝ) (h_june_sale : sale_june = 400) (profit : ℝ) (h_profit : profit = 8400) :
  (y - 35) * (1560 - 20 * y) = 8400 → y = 50 :=
sorry

end monthly_average_growth_rate_optimal_selling_price_for_desired_profit_l330_33063


namespace cost_of_each_card_is_2_l330_33014

-- Define the conditions
def christmas_cards : ℕ := 20
def birthday_cards : ℕ := 15
def total_spent : ℝ := 70

-- Define the total number of cards
def total_cards : ℕ := christmas_cards + birthday_cards

-- Define the cost per card
noncomputable def cost_per_card : ℝ := total_spent / total_cards

-- The theorem
theorem cost_of_each_card_is_2 : cost_per_card = 2 := by
  sorry

end cost_of_each_card_is_2_l330_33014


namespace evaluate_expression_l330_33066

theorem evaluate_expression : (3 + 5 + 7) / (2 + 4 + 6) - (2 + 4 + 6) / (3 + 5 + 7) = 9 / 20 := 
by 
  sorry

end evaluate_expression_l330_33066


namespace three_digit_sum_reverse_eq_l330_33056

theorem three_digit_sum_reverse_eq :
  ∃ (a b c : ℕ) (ha : 0 ≤ a ∧ a ≤ 9) (hb : 0 ≤ b ∧ b ≤ 9) (hc : 0 ≤ c ∧ c ≤ 9),
    101 * (a + c) + 20 * b = 1777 ∧ (a, b, c) = (8, 7, 9) ∨ (a, b, c) = (9, 7, 8) :=
by
  sorry

end three_digit_sum_reverse_eq_l330_33056


namespace total_ounces_of_coffee_l330_33080

/-
Defining the given conditions
-/
def num_packages_10_oz : Nat := 5
def num_packages_5_oz : Nat := num_packages_10_oz + 2
def ounces_per_10_oz_pkg : Nat := 10
def ounces_per_5_oz_pkg : Nat := 5

/-
Statement to prove the total ounces of coffee
-/
theorem total_ounces_of_coffee :
  (num_packages_10_oz * ounces_per_10_oz_pkg + num_packages_5_oz * ounces_per_5_oz_pkg) = 85 := by
  sorry

end total_ounces_of_coffee_l330_33080


namespace product_of_two_numbers_l330_33059

theorem product_of_two_numbers :
  ∃ x y : ℝ, x + y = 16 ∧ x^2 + y^2 = 200 ∧ x * y = 28 :=
by
  sorry

end product_of_two_numbers_l330_33059


namespace units_digit_n_l330_33033

theorem units_digit_n (m n : ℕ) (h1 : m * n = 14^5) (h2 : m % 10 = 8) : n % 10 = 3 :=
sorry

end units_digit_n_l330_33033


namespace binary_digit_one_l330_33015
-- We import the necessary libraries

-- Define the problem and prove the statement as follows
def fractional_part_in_binary (x : ℝ) : ℕ → ℕ := sorry

def sqrt_fractional_binary (k : ℕ) (i : ℕ) : ℕ :=
  fractional_part_in_binary (Real.sqrt ((k : ℝ) * (k + 1))) i

theorem binary_digit_one {k n : ℕ} (hk : 0 < k) (hn : 0 < n) :
  ∃ i, n + 1 ≤ i ∧ i ≤ 2 * n + 1 ∧ sqrt_fractional_binary k i = 1 :=
sorry

end binary_digit_one_l330_33015


namespace values_of_a_l330_33099

open Set

noncomputable def A : Set ℝ := { x | x^2 - 2*x - 3 = 0 }
noncomputable def B (a : ℝ) : Set ℝ := if a = 0 then ∅ else { x | a * x = 1 }

theorem values_of_a (a : ℝ) : (B a ⊆ A) ↔ (a = -1 ∨ a = 0 ∨ a = 1/3) :=
by 
  sorry

end values_of_a_l330_33099


namespace f_at_2023_l330_33027

noncomputable def f (a x : ℝ) : ℝ := (a - x) / (a + 2 * x)

noncomputable def g (a x : ℝ) : ℝ := (f a (x - 2023)) + (1 / 2)

def is_odd (g : ℝ → ℝ) := ∀ x : ℝ, g (-x) = -g x

variable (a : ℝ)
variable (h_a : a ≠ 0)
variable (h_odd : is_odd (g a))

theorem f_at_2023 : f a 2023 = 1 / 4 :=
sorry

end f_at_2023_l330_33027


namespace triangle_inequality_l330_33075

variables {l_a l_b l_c m_a m_b m_c h_n m_n h_h_n m_m_p : ℝ}

-- Assuming some basic properties for the variables involved (all are positive in their respective triangle context)
axiom pos_l_a : 0 < l_a
axiom pos_l_b : 0 < l_b
axiom pos_l_c : 0 < l_c
axiom pos_m_a : 0 < m_a
axiom pos_m_b : 0 < m_b
axiom pos_m_c : 0 < m_c
axiom pos_h_n : 0 < h_n
axiom pos_m_n : 0 < m_n
axiom pos_h_h_n : 0 < h_h_n
axiom pos_m_m_p : 0 < m_m_p

theorem triangle_inequality :
  (h_n / m_n) + (h_n / h_h_n) + (l_c / m_m_p) > 1 :=
sorry

end triangle_inequality_l330_33075


namespace truncated_cone_surface_area_l330_33074

theorem truncated_cone_surface_area (R r : ℝ) (S : ℝ)
  (h1: S = 4 * Real.pi * (R^2 + R * r + r^2)) :
  2 * Real.pi * (R^2 + R * r + r^2) = S / 2 :=
by
  sorry

end truncated_cone_surface_area_l330_33074


namespace lcm_60_30_40_eq_120_l330_33034

theorem lcm_60_30_40_eq_120 : (Nat.lcm (Nat.lcm 60 30) 40) = 120 := 
sorry

end lcm_60_30_40_eq_120_l330_33034


namespace find_number_l330_33047

theorem find_number (x : ℝ) (h : 0.60 * x - 40 = 50) : x = 150 := 
by
  sorry

end find_number_l330_33047


namespace boxes_in_case_number_of_boxes_in_case_l330_33070

-- Definitions based on the conditions
def boxes_of_eggs : Nat := 5
def eggs_per_box : Nat := 3
def total_eggs : Nat := 15

-- Proposition
theorem boxes_in_case (boxes_of_eggs : Nat) (eggs_per_box : Nat) (total_eggs : Nat) : Nat :=
  if boxes_of_eggs * eggs_per_box = total_eggs then boxes_of_eggs else 0

-- Assertion that needs to be proven
theorem number_of_boxes_in_case : boxes_in_case boxes_of_eggs eggs_per_box total_eggs = 5 :=
by sorry

end boxes_in_case_number_of_boxes_in_case_l330_33070


namespace p_necessary_not_sufficient_for_q_l330_33000

def p (x : ℝ) : Prop := abs x = -x
def q (x : ℝ) : Prop := x^2 ≥ -x

theorem p_necessary_not_sufficient_for_q : 
  (∀ x, q x → p x) ∧ ¬ (∀ x, p x → q x) :=
by
  sorry

end p_necessary_not_sufficient_for_q_l330_33000


namespace problem1_problem2_problem3_l330_33055

-- Definitions of transformations and final sequence S
def transformation (A : List ℕ) : List ℕ := 
  match A with
  | x :: y :: xs => (x + y) :: transformation (y :: xs)
  | _ => []

def nth_transform (A : List ℕ) (n : ℕ) : List ℕ :=
  Nat.iterate (λ L => transformation L) n A

def final_sequence (A : List ℕ) : ℕ :=
  match nth_transform A (A.length - 1) with
  | [x] => x
  | _ => 0

-- Proof Statements

theorem problem1 : final_sequence [1, 2, 3] = 8 := sorry

theorem problem2 (n : ℕ) : final_sequence (List.range (n+1)) = (n + 2) * 2 ^ (n - 1) := sorry

theorem problem3 (A B : List ℕ) (h : A = List.range (B.length)) (h_perm : B.permutations.contains A) : 
  final_sequence B = final_sequence A := by
  sorry

end problem1_problem2_problem3_l330_33055


namespace initial_value_divisible_by_456_l330_33029

def initial_value := 374
def to_add := 82
def divisor := 456

theorem initial_value_divisible_by_456 : (initial_value + to_add) % divisor = 0 := by
  sorry

end initial_value_divisible_by_456_l330_33029


namespace boat_downstream_distance_l330_33092

-- Given conditions
def speed_boat_still_water : ℕ := 25
def speed_stream : ℕ := 5
def travel_time_downstream : ℕ := 3

-- Proof statement: The distance travelled downstream is 90 km
theorem boat_downstream_distance :
  speed_boat_still_water + speed_stream * travel_time_downstream = 90 :=
by
  -- omitting the actual proof steps
  sorry

end boat_downstream_distance_l330_33092
