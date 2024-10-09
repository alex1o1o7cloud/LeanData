import Mathlib

namespace ratio_second_to_third_l1301_130120

-- Define the three numbers A, B, C, and their conditions.
variables (A B C : ℕ)

-- Conditions derived from the problem statement.
def sum_condition : Prop := A + B + C = 98
def ratio_condition : Prop := 3 * A = 2 * B
def second_number_value : Prop := B = 30

-- The main theorem stating the problem to prove.
theorem ratio_second_to_third (h1 : sum_condition A B C) (h2 : ratio_condition A B) (h3 : second_number_value B) :
  B = 30 ∧ A = 20 ∧ C = 48 → B / C = 5 / 8 :=
by
  sorry

end ratio_second_to_third_l1301_130120


namespace Jason_reroll_exactly_two_dice_probability_l1301_130101

noncomputable def probability_reroll_two_dice : ℚ :=
  let favorable_outcomes := 5 * 3 + 1 * 3 + 5 * 3
  let total_possibilities := 6^3
  favorable_outcomes / total_possibilities

theorem Jason_reroll_exactly_two_dice_probability : probability_reroll_two_dice = 5 / 9 := 
  sorry

end Jason_reroll_exactly_two_dice_probability_l1301_130101


namespace part_a_part_b_l1301_130112

-- Let p_k represent the probability that at the moment of completing the first collection, the second collection is missing exactly k crocodiles.
def p (k : ℕ) : ℝ := sorry

-- The conditions 
def totalCrocodiles : ℕ := 10
def probabilityEachEgg : ℝ := 0.1

-- Problems to prove:

-- (a) Prove that p_1 = p_2
theorem part_a : p 1 = p 2 := sorry

-- (b) Prove that p_2 > p_3 > p_4 > ... > p_10
theorem part_b : ∀ k, 2 ≤ k ∧ k < totalCrocodiles → p k > p (k + 1) := sorry

end part_a_part_b_l1301_130112


namespace students_more_than_pets_l1301_130178

-- Definitions for the conditions
def number_of_classrooms := 5
def students_per_classroom := 22
def rabbits_per_classroom := 3
def hamsters_per_classroom := 2

-- Total number of students in all classrooms
def total_students := number_of_classrooms * students_per_classroom

-- Total number of pets in all classrooms
def total_pets := number_of_classrooms * (rabbits_per_classroom + hamsters_per_classroom)

-- The theorem to prove
theorem students_more_than_pets : 
  total_students - total_pets = 85 :=
by
  sorry

end students_more_than_pets_l1301_130178


namespace gordon_total_cost_l1301_130111

noncomputable def DiscountA (price : ℝ) : ℝ :=
if price > 22.00 then price * 0.70 else price

noncomputable def DiscountB (price : ℝ) : ℝ :=
if 10.00 < price ∧ price <= 20.00 then price * 0.80 else price

noncomputable def DiscountC (price : ℝ) : ℝ :=
if price < 10.00 then price * 0.85 else price

noncomputable def apply_discount (price : ℝ) : ℝ :=
if price > 22.00 then DiscountA price
else if price > 10.00 then DiscountB price
else DiscountC price

noncomputable def total_price (prices : List ℝ) : ℝ :=
(prices.map apply_discount).sum

noncomputable def total_with_tax_and_fee (prices : List ℝ) (tax_rate extra_fee : ℝ) : ℝ :=
let total := total_price prices
let tax := total * tax_rate
total + tax + extra_fee

theorem gordon_total_cost :
  total_with_tax_and_fee
    [25.00, 18.00, 21.00, 35.00, 12.00, 10.00, 8.50, 23.00, 6.00, 15.50, 30.00, 9.50]
    0.05 2.00
  = 171.27 :=
  sorry

end gordon_total_cost_l1301_130111


namespace find_smallest_a_l1301_130135
open Real

noncomputable def a_min := 2 / 9

theorem find_smallest_a (a b c : ℝ)
  (h1 : (1/4, -9/8) = (1/4, a * (1/4) * (1/4) - 9/8))
  (h2 : ∃ n : ℤ, a + b + c = n)
  (h3 : a > 0)
  (h4 : b = - a / 2)
  (h5 : c = a / 16 - 9 / 8): 
  a = a_min :=
by {
  -- Lean code equivalent to the provided mathematical proof will be placed here.
  sorry
}

end find_smallest_a_l1301_130135


namespace smallest_flash_drives_l1301_130189

theorem smallest_flash_drives (total_files : ℕ) (flash_drive_space: ℝ)
  (files_size : ℕ → ℝ)
  (h1 : total_files = 40)
  (h2 : flash_drive_space = 2.0)
  (h3 : ∀ n, (n < 4 → files_size n = 1.2) ∧ 
              (4 ≤ n ∧ n < 20 → files_size n = 0.9) ∧ 
              (20 ≤ n → files_size n = 0.6)) :
  ∃ min_flash_drives, min_flash_drives = 20 :=
sorry

end smallest_flash_drives_l1301_130189


namespace find_number_l1301_130142

theorem find_number (x : ℝ) (h : (x / 6) * 12 = 15) : x = 7.5 :=
sorry

end find_number_l1301_130142


namespace find_x_l1301_130187

theorem find_x (x y : ℤ) (h1 : y = 3) (h2 : x + 3 * y = 10) : x = 1 :=
by
  sorry

end find_x_l1301_130187


namespace percentage_B_D_l1301_130143

variables (A B C D : ℝ)

-- Conditions as hypotheses
theorem percentage_B_D
  (h1 : B = 1.71 * A)
  (h2 : C = 1.80 * A)
  (h3 : D = 1.90 * B)
  (h4 : B = 1.62 * C)
  (h5 : A = 0.65 * D)
  (h6 : C = 0.55 * D) : 
  B = 1.1115 * D :=
sorry

end percentage_B_D_l1301_130143


namespace intersection_M_N_l1301_130173

def M : Set ℤ := { -2, -1, 0, 1, 2 }
def N : Set ℤ := {x | x^2 - x - 6 ≥ 0}

theorem intersection_M_N :
  M ∩ N = { -2 } :=
by
  sorry

end intersection_M_N_l1301_130173


namespace cake_eating_classmates_l1301_130186

theorem cake_eating_classmates (n : ℕ) :
  (Alex_ate : ℚ := 1/11) → (Alena_ate : ℚ := 1/14) → 
  (12 ≤ n ∧ n ≤ 13) :=
by
  sorry

end cake_eating_classmates_l1301_130186


namespace point_on_hyperbola_l1301_130129

theorem point_on_hyperbola (p r : ℝ) (h1 : p > 0) (h2 : r > 0)
  (h_el : ∀ (x y : ℝ), x^2 / 4 + y^2 / 2 = 1)
  (h_par : ∀ (x y : ℝ), y^2 = 2 * p * x)
  (h_circum : ∀ (a b c : ℝ), a = 2 * r - 2 * p) :
  r^2 - p^2 = 1 := sorry

end point_on_hyperbola_l1301_130129


namespace number_of_students_l1301_130150

theorem number_of_students (n : ℕ) (h1 : n < 60) (h2 : n % 6 = 4) (h3 : n % 8 = 5) : n = 46 := by
  sorry

end number_of_students_l1301_130150


namespace annual_decrease_rate_l1301_130107

theorem annual_decrease_rate (P₀ P₂ : ℝ) (r : ℝ) (h₀ : P₀ = 8000) (h₂ : P₂ = 5120) :
  P₂ = P₀ * (1 - r / 100) ^ 2 → r = 20 :=
by
  intros h
  have h₀' : P₀ = 8000 := h₀
  have h₂' : P₂ = 5120 := h₂
  sorry

end annual_decrease_rate_l1301_130107


namespace divisors_of_2700_l1301_130147

def prime_factors_2700 : ℕ := 2^2 * 3^3 * 5^2

def number_of_positive_divisors (n : ℕ) (a b c : ℕ) : ℕ :=
  (a + 1) * (b + 1) * (c + 1)

theorem divisors_of_2700 : number_of_positive_divisors 2700 2 3 2 = 36 := by
  sorry

end divisors_of_2700_l1301_130147


namespace integers_multiples_of_d_l1301_130148

theorem integers_multiples_of_d (d m n : ℕ) 
  (h1 : 2 ≤ m) 
  (h2 : 1 ≤ n) 
  (gcd_m_n : Nat.gcd m n = d) 
  (gcd_m_4n1 : Nat.gcd m (4 * n + 1) = 1) : 
  m % d = 0 :=
sorry

end integers_multiples_of_d_l1301_130148


namespace simplify_expression_l1301_130175

variable (z : ℝ)

theorem simplify_expression: (4 - 5 * z^2) - (2 + 7 * z^2 - z) = 2 - 12 * z^2 + z :=
by sorry

end simplify_expression_l1301_130175


namespace count_possible_integer_values_l1301_130171

theorem count_possible_integer_values :
  ∃ n : ℕ, (∀ x : ℤ, (25 < x ∧ x < 55) ↔ (26 ≤ x ∧ x ≤ 54)) ∧ n = 29 := by
  sorry

end count_possible_integer_values_l1301_130171


namespace minimize_expression_l1301_130156

theorem minimize_expression (x : ℝ) (h : 0 < x) : 
  x = 9 ↔ (∀ y : ℝ, 0 < y → x + 81 / x ≤ y + 81 / y) :=
sorry

end minimize_expression_l1301_130156


namespace point_coordinates_l1301_130163

def point : Type := ℝ × ℝ

def x_coordinate (P : point) : ℝ := P.1

def y_coordinate (P : point) : ℝ := P.2

theorem point_coordinates (P : point) (h1 : x_coordinate P = -3) (h2 : abs (y_coordinate P) = 5) :
  P = (-3, 5) ∨ P = (-3, -5) :=
by
  sorry

end point_coordinates_l1301_130163


namespace min_value_of_function_l1301_130119

open Real

theorem min_value_of_function (x : ℝ) (h : x > 2) : (∃ a : ℝ, (∀ y : ℝ, y = (4 / (x - 2) + x) → y ≥ a) ∧ a = 6) :=
sorry

end min_value_of_function_l1301_130119


namespace volume_of_box_l1301_130172

-- Defining the initial parameters of the problem
def length_sheet := 48
def width_sheet := 36
def side_length_cut_square := 3

-- Define the transformed dimensions after squares are cut off
def length_box := length_sheet - 2 * side_length_cut_square
def width_box := width_sheet - 2 * side_length_cut_square
def height_box := side_length_cut_square

-- The target volume of the box
def target_volume := 3780

-- Prove that the volume of the box is equal to the target volume
theorem volume_of_box : length_box * width_box * height_box = target_volume := by
  -- Calculate the expected volume
  -- Expected volume = 42 m * 30 m * 3 m
  -- Which equals 3780 m³
  sorry

end volume_of_box_l1301_130172


namespace probability_max_min_difference_is_five_l1301_130139

theorem probability_max_min_difference_is_five : 
  let total_outcomes := 6 ^ 4
  let outcomes_without_1 := 5 ^ 4
  let outcomes_without_6 := 5 ^ 4
  let outcomes_without_1_and_6 := 4 ^ 4
  total_outcomes - 2 * outcomes_without_1 + outcomes_without_1_and_6 = 302 →
  (302 : ℚ) / total_outcomes = 151 / 648 :=
by
  intros
  sorry

end probability_max_min_difference_is_five_l1301_130139


namespace triangle_equilateral_of_angles_and_intersecting_segments_l1301_130108

theorem triangle_equilateral_of_angles_and_intersecting_segments
    (A B C : Type) (angle_A : ℝ) (intersect_at_one_point : Prop)
    (angle_M_bisects : Prop) (N_is_median : Prop) (L_is_altitude : Prop) :
  angle_A = 60 ∧ angle_M_bisects ∧ N_is_median ∧ L_is_altitude ∧ intersect_at_one_point → 
  ∀ (angle_B angle_C : ℝ), angle_B = 60 ∧ angle_C = 60 := 
by
  intro h
  sorry

end triangle_equilateral_of_angles_and_intersecting_segments_l1301_130108


namespace total_perimeter_l1301_130132

/-- 
A rectangular plot where the long sides are three times the length of the short sides. 
One short side is 80 feet. Prove the total perimeter is 640 feet.
-/
theorem total_perimeter (s : ℕ) (h : s = 80) : 8 * s = 640 :=
  by sorry

end total_perimeter_l1301_130132


namespace power_exponent_multiplication_l1301_130140

variable (a : ℝ)

theorem power_exponent_multiplication : (a^3)^2 = a^6 := sorry

end power_exponent_multiplication_l1301_130140


namespace min_radius_of_circumcircle_l1301_130128

theorem min_radius_of_circumcircle {a b : ℝ} (ha : a = 3) (hb : b = 4) : 
∃ R : ℝ, R = 2.5 ∧ (∃ c : ℝ, c = Real.sqrt (a^2 + b^2) ∧ a^2 + b^2 = c^2 ∧ 2 * R = c) :=
by 
  sorry

end min_radius_of_circumcircle_l1301_130128


namespace mixture_ratio_l1301_130153

variables (p q : ℝ)

theorem mixture_ratio 
  (h1 : (5/8) * p + (1/4) * q = 0.5)
  (h2 : (3/8) * p + (3/4) * q = 0.5) : 
  p / q = 1 := 
by 
  sorry

end mixture_ratio_l1301_130153


namespace suyeong_ran_distance_l1301_130130

theorem suyeong_ran_distance 
  (circumference : ℝ) 
  (laps : ℕ) 
  (h_circumference : circumference = 242.7)
  (h_laps : laps = 5) : 
  (circumference * laps = 1213.5) := 
  by sorry

end suyeong_ran_distance_l1301_130130


namespace percentage_increase_l1301_130115

theorem percentage_increase 
  (distance : ℝ) (time_q : ℝ) (time_y : ℝ) 
  (speed_q : ℝ) (speed_y : ℝ) 
  (percentage_increase : ℝ) 
  (h_distance : distance = 80)
  (h_time_q : time_q = 2)
  (h_time_y : time_y = 1.3333333333333333)
  (h_speed_q : speed_q = distance / time_q)
  (h_speed_y : speed_y = distance / time_y)
  (h_faster : speed_y > speed_q)
  : percentage_increase = ((speed_y - speed_q) / speed_q) * 100 :=
by
  sorry

end percentage_increase_l1301_130115


namespace gain_percent_l1301_130105

theorem gain_percent (CP SP : ℝ) (hCP : CP = 20) (hSP : SP = 35) : 
  (SP - CP) / CP * 100 = 75 :=
by
  rw [hCP, hSP]
  sorry

end gain_percent_l1301_130105


namespace value_of_expression_l1301_130161

theorem value_of_expression (a b : ℝ) (h : a - b = 1) : a^2 - b^2 - 2 * b = 1 := 
by
  sorry

end value_of_expression_l1301_130161


namespace reading_rate_l1301_130122

-- Definitions based on conditions
def one_way_trip_time : ℕ := 4
def round_trip_time : ℕ := 2 * one_way_trip_time
def read_book_time : ℕ := 2 * round_trip_time
def book_pages : ℕ := 4000

-- The theorem to prove Juan's reading rate is 250 pages per hour.
theorem reading_rate : book_pages / read_book_time = 250 := by
  sorry

end reading_rate_l1301_130122


namespace car_avg_mpg_B_to_C_is_11_11_l1301_130199

noncomputable def avg_mpg_B_to_C (D : ℝ) : ℝ :=
  let avg_mpg_total := 42.857142857142854
  let x := (100 : ℝ) / 9
  let total_distance := (3 / 2) * D
  let total_gallons := (D / 40) + (D / (2 * x))
  (total_distance / total_gallons)

/-- Prove the car's average miles per gallon from town B to town C is 100/9 mpg. -/
theorem car_avg_mpg_B_to_C_is_11_11 (D : ℝ) (h1 : D > 0):
  avg_mpg_B_to_C D = 100 / 9 :=
by
  sorry

end car_avg_mpg_B_to_C_is_11_11_l1301_130199


namespace tobias_downloads_l1301_130184

theorem tobias_downloads : 
  ∀ (m : ℕ), (∀ (price_per_app total_spent : ℝ), 
  price_per_app = 2.00 + 2.00 * 0.10 ∧ 
  total_spent = 52.80 → 
  m = total_spent / price_per_app) → 
  m = 24 := 
  sorry

end tobias_downloads_l1301_130184


namespace peter_total_distance_is_six_l1301_130160

def total_distance_covered (d : ℝ) :=
  let first_part_time := (2/3) * d / 4
  let second_part_time := (1/3) * d / 5
  (first_part_time + second_part_time) = 1.4

theorem peter_total_distance_is_six :
  ∃ d : ℝ, total_distance_covered d ∧ d = 6 := 
by
  -- Proof can be filled here
  sorry

end peter_total_distance_is_six_l1301_130160


namespace optimal_selling_price_minimize_loss_l1301_130159

theorem optimal_selling_price_minimize_loss 
  (C : ℝ) (h1 : 17 * C = 720 + 5 * C) 
  (h2 : ∀ x : ℝ, x * (1 - 0.1) = 720 * 0.9)
  (h3 : ∀ y : ℝ, y * (1 + 0.05) = 648 * 1.05)
  (selling_price : ℝ)
  (optimal_selling_price : selling_price = 60) :
  selling_price = C :=
by 
  sorry

end optimal_selling_price_minimize_loss_l1301_130159


namespace remainder_div_357_l1301_130106

theorem remainder_div_357 (N : ℤ) (h : N % 17 = 2) : N % 357 = 2 :=
sorry

end remainder_div_357_l1301_130106


namespace roger_current_money_l1301_130182

noncomputable def roger_initial_money : ℕ := 16
noncomputable def roger_birthday_money : ℕ := 28
noncomputable def roger_game_spending : ℕ := 25

theorem roger_current_money : roger_initial_money + roger_birthday_money - roger_game_spending = 19 := by
  sorry

end roger_current_money_l1301_130182


namespace sequence_general_formula_l1301_130196

theorem sequence_general_formula (a : ℕ → ℝ) (h1 : a 1 = 3) (h2 : ∀ n : ℕ, n ≥ 1 → a (n + 1) = 3 * a n - 4) :
  ∀ n : ℕ, n ≥ 1 → a n = 3^(n - 1) + 2 :=
sorry

end sequence_general_formula_l1301_130196


namespace range_of_x_coordinate_l1301_130134

def is_on_line (A : ℝ × ℝ) : Prop := A.1 + A.2 = 6

def is_on_circle (C : ℝ × ℝ) : Prop := (C.1 - 1)^2 + (C.2 - 1)^2 = 4

def angle_BAC_is_60_degrees (A B C : ℝ × ℝ) : Prop :=
  -- This definition is simplified as an explanation. Angle computation in Lean might be more intricate.
  sorry 

theorem range_of_x_coordinate (A : ℝ × ℝ) (B C : ℝ × ℝ)
  (hA_on_line : is_on_line A)
  (hB_on_circle : is_on_circle B)
  (hC_on_circle : is_on_circle C)
  (h_angle_BAC : angle_BAC_is_60_degrees A B C) :
  1 ≤ A.1 ∧ A.1 ≤ 5 :=
sorry

end range_of_x_coordinate_l1301_130134


namespace rectangle_perimeter_l1301_130138

theorem rectangle_perimeter (a b : ℚ) (ha : ¬ a.den = 1) (hb : ¬ b.den = 1) (hab : a ≠ b) (h : (a - 2) * (b - 2) = -7) : 2 * (a + b) = 20 :=
by
  sorry

end rectangle_perimeter_l1301_130138


namespace calculate_f_50_l1301_130164

noncomputable def f (x : ℝ) : ℝ := sorry

theorem calculate_f_50 (f : ℝ → ℝ) (h_fun : ∀ x y : ℝ, f (x * y) = y * f x) (h_f2 : f 2 = 10) :
  f 50 = 250 :=
by
  sorry

end calculate_f_50_l1301_130164


namespace find_x_l1301_130116

noncomputable def x : ℝ :=
  0.49

theorem find_x (h : (Real.sqrt 1.21) / (Real.sqrt 0.81) + (Real.sqrt 0.81) / (Real.sqrt x) = 2.507936507936508) : 
  x = 0.49 :=
sorry

end find_x_l1301_130116


namespace johns_actual_marks_l1301_130174

def actual_marks (T : ℝ) (x : ℝ) (incorrect : ℝ) (students : ℕ) (avg_increase : ℝ) :=
  (incorrect = 82) ∧ (students = 80) ∧ (avg_increase = 1/2) ∧
  ((T + incorrect) / students = (T + x) / students + avg_increase)

theorem johns_actual_marks (T : ℝ) :
  ∃ x : ℝ, actual_marks T x 82 80 (1/2) ∧ x = 42 :=
by
  sorry

end johns_actual_marks_l1301_130174


namespace passengers_in_7_buses_l1301_130183

theorem passengers_in_7_buses (passengers_total buses_total_given buses_required : ℕ) 
    (h1 : passengers_total = 456) 
    (h2 : buses_total_given = 12) 
    (h3 : buses_required = 7) :
    (passengers_total / buses_total_given) * buses_required = 266 := 
sorry

end passengers_in_7_buses_l1301_130183


namespace richmond_tigers_revenue_l1301_130195

theorem richmond_tigers_revenue
  (total_tickets : ℕ)
  (first_half_tickets : ℕ)
  (catA_first_half : ℕ)
  (catB_first_half : ℕ)
  (catC_first_half : ℕ)
  (priceA : ℕ)
  (priceB : ℕ)
  (priceC : ℕ)
  (catA_second_half : ℕ)
  (catB_second_half : ℕ)
  (catC_second_half : ℕ)
  (total_revenue_second_half : ℕ)
  (h_total_tickets : total_tickets = 9570)
  (h_first_half_tickets : first_half_tickets = 3867)
  (h_catA_first_half : catA_first_half = 1350)
  (h_catB_first_half : catB_first_half = 1150)
  (h_catC_first_half : catC_first_half = 1367)
  (h_priceA : priceA = 50)
  (h_priceB : priceB = 40)
  (h_priceC : priceC = 30)
  (h_catA_second_half : catA_second_half = 1350)
  (h_catB_second_half : catB_second_half = 1150)
  (h_catC_second_half : catC_second_half = 1367)
  (h_total_revenue_second_half : total_revenue_second_half = 154510)
  :
  catA_second_half * priceA + catB_second_half * priceB + catC_second_half * priceC = total_revenue_second_half :=
by
  sorry

end richmond_tigers_revenue_l1301_130195


namespace triangle_area_l1301_130141

theorem triangle_area :
  let a := 4
  let c := 5
  let b := Real.sqrt (c^2 - a^2)
  (1 / 2) * a * b = 6 :=
by sorry

end triangle_area_l1301_130141


namespace train_length_l1301_130152

theorem train_length
  (speed_kmph : ℕ) (time_s : ℕ)
  (h1 : speed_kmph = 72)
  (h2 : time_s = 12) :
  speed_kmph * (1000 / 3600 : ℕ) * time_s = 240 :=
by
  sorry

end train_length_l1301_130152


namespace multiple_of_cans_of_corn_l1301_130181

theorem multiple_of_cans_of_corn (peas corn : ℕ) (h1 : peas = 35) (h2 : corn = 10) (h3 : peas = 10 * x + 15) : x = 2 := 
by
  sorry

end multiple_of_cans_of_corn_l1301_130181


namespace bowling_ball_weight_l1301_130180

theorem bowling_ball_weight :
  ∃ (b : ℝ) (c : ℝ),
    8 * b = 5 * c ∧
    4 * c = 100 ∧
    b = 15.625 :=
by 
  sorry

end bowling_ball_weight_l1301_130180


namespace sonia_and_joss_time_spent_moving_l1301_130194

def total_time_spent_moving (fill_time_per_trip drive_time_per_trip trips : ℕ) :=
  (fill_time_per_trip + drive_time_per_trip) * trips

def total_time_in_hours (total_time_in_minutes : ℕ) : ℚ :=
  total_time_in_minutes / 60

theorem sonia_and_joss_time_spent_moving :
  total_time_in_hours (total_time_spent_moving 15 30 6) = 4.5 :=
by
  sorry

end sonia_and_joss_time_spent_moving_l1301_130194


namespace book_pages_count_l1301_130176

theorem book_pages_count :
  (∀ n : ℕ, n = 4 → 42 * n = 168) ∧
  (∀ n : ℕ, n = 2 → 50 * n = 100) ∧
  (∀ p1 p2 : ℕ, p1 = 168 ∧ p2 = 100 → p1 + p2 = 268) ∧
  (∀ p : ℕ, p = 268 → p + 30 = 298) →
  298 = 298 := by
  sorry

end book_pages_count_l1301_130176


namespace collinear_k_perpendicular_k_l1301_130136

def vector := ℝ × ℝ

def a : vector := (1, 3)
def b : vector := (3, -4)

def collinear (u v : vector) : Prop :=
  u.1 * v.2 = u.2 * v.1

def perpendicular (u v : vector) : Prop :=
  u.1 * v.1 + u.2 * v.2 = 0

def k_vector_a_minus_b (k : ℝ) (a b : vector) : vector :=
  (k * a.1 - b.1, k * a.2 - b.2)

def a_plus_b (a b : vector) : vector :=
  (a.1 + b.1, a.2 + b.2)

theorem collinear_k (k : ℝ) : collinear (k_vector_a_minus_b k a b) (a_plus_b a b) ↔ k = -1 :=
sorry

theorem perpendicular_k (k : ℝ) : perpendicular (k_vector_a_minus_b k a b) (a_plus_b a b) ↔ k = 16 :=
sorry

end collinear_k_perpendicular_k_l1301_130136


namespace sum_of_percentages_l1301_130114

theorem sum_of_percentages :
  let percent1 := 7.35 / 100
  let percent2 := 13.6 / 100
  let percent3 := 21.29 / 100
  let num1 := 12658
  let num2 := 18472
  let num3 := 29345
  let result := percent1 * num1 + percent2 * num2 + percent3 * num3
  result = 9689.9355 :=
by
  sorry

end sum_of_percentages_l1301_130114


namespace equipment_unit_prices_purchasing_scenarios_l1301_130121

theorem equipment_unit_prices
  (x : ℝ)
  (price_A_eq_price_B_minus_10 : ∀ y, ∃ z, z = y + 10)
  (eq_purchases_equal_cost_A : ∀ n : ℕ, 300 / x = n)
  (eq_purchases_equal_cost_B : ∀ n : ℕ, 360 / (x + 10) = n) :
  x = 50 ∧ (x + 10) = 60 :=
by
  sorry

theorem purchasing_scenarios
  (m n : ℕ)
  (price_A : ℝ := 50)
  (price_B : ℝ := 60)
  (budget : ℝ := 1000)
  (purchase_eq_budget : 50 * m + 60 * n = 1000)
  (pos_integers : m > 0 ∧ n > 0) :
  (m = 14 ∧ n = 5) ∨ (m = 8 ∧ n = 10) ∨ (m = 2 ∧ n = 15) :=
by
  sorry

end equipment_unit_prices_purchasing_scenarios_l1301_130121


namespace sum_of_powers_of_two_l1301_130145

theorem sum_of_powers_of_two (n : ℕ) (h : 1 ≤ n ∧ n ≤ 511) : 
  ∃ (S : Finset ℕ), S ⊆ ({2^8, 2^7, 2^6, 2^5, 2^4, 2^3, 2^2, 2^1, 2^0} : Finset ℕ) ∧ 
  S.sum id = n :=
by
  sorry

end sum_of_powers_of_two_l1301_130145


namespace latest_time_temp_decreasing_l1301_130168

theorem latest_time_temp_decreasing (t : ℝ) 
  (h1 : -t^2 + 12 * t + 55 = 82) 
  (h2 : ∀ t0 : ℝ, -2 * t0 + 12 < 0 → t > t0) : 
  t = 6 + (3 * Real.sqrt 28 / 2) :=
sorry

end latest_time_temp_decreasing_l1301_130168


namespace ravi_jump_height_l1301_130192

theorem ravi_jump_height (j1 j2 j3 : ℕ) (average : ℕ) (ravi_jump_height : ℕ) (h : j1 = 23 ∧ j2 = 27 ∧ j3 = 28) 
  (ha : average = (j1 + j2 + j3) / 3) (hr : ravi_jump_height = 3 * average / 2) : ravi_jump_height = 39 :=
by
  sorry

end ravi_jump_height_l1301_130192


namespace ellipse_with_foci_on_x_axis_l1301_130167

theorem ellipse_with_foci_on_x_axis {a : ℝ} (h1 : a - 5 > 0) (h2 : 2 > 0) (h3 : a - 5 > 2) :
  a > 7 :=
by
  sorry

end ellipse_with_foci_on_x_axis_l1301_130167


namespace min_value_of_m_squared_plus_n_squared_l1301_130124

theorem min_value_of_m_squared_plus_n_squared (m n : ℝ) 
  (h : 4 * m - 3 * n - 5 * Real.sqrt 2 = 0) : m^2 + n^2 = 2 :=
sorry

end min_value_of_m_squared_plus_n_squared_l1301_130124


namespace intersection_sum_x_coordinates_mod_17_l1301_130146

theorem intersection_sum_x_coordinates_mod_17 :
  ∃ x : ℤ, (∃ y₁ y₂ : ℤ, (y₁ ≡ 7 * x + 3 [ZMOD 17]) ∧ (y₂ ≡ 13 * x + 4 [ZMOD 17]))
       ∧ x ≡ 14 [ZMOD 17]  :=
by
  sorry

end intersection_sum_x_coordinates_mod_17_l1301_130146


namespace roller_coaster_costs_7_tickets_l1301_130127

-- Define the number of tickets for the Ferris wheel, log ride, and the initial and additional tickets Zach needs.
def ferris_wheel_tickets : ℕ := 2
def log_ride_tickets : ℕ := 1
def initial_tickets : ℕ := 1
def additional_tickets : ℕ := 9

-- Define the total number of tickets Zach needs.
def total_tickets : ℕ := initial_tickets + additional_tickets

-- Define the number of tickets needed for the Ferris wheel and log ride together.
def combined_tickets_needed : ℕ := ferris_wheel_tickets + log_ride_tickets

-- Define the number of tickets the roller coaster costs.
def roller_coaster_tickets : ℕ := total_tickets - combined_tickets_needed

-- The theorem stating what we need to prove.
theorem roller_coaster_costs_7_tickets :
  roller_coaster_tickets = 7 :=
by sorry

end roller_coaster_costs_7_tickets_l1301_130127


namespace mass_percentage_H_in_C4H8O2_l1301_130100

theorem mass_percentage_H_in_C4H8O2 (molar_mass_C : Real := 12.01) 
                                    (molar_mass_H : Real := 1.008) 
                                    (molar_mass_O : Real := 16.00) 
                                    (num_C_atoms : Nat := 4)
                                    (num_H_atoms : Nat := 8)
                                    (num_O_atoms : Nat := 2) :
    (num_H_atoms * molar_mass_H) / ((num_C_atoms * molar_mass_C) + (num_H_atoms * molar_mass_H) + (num_O_atoms * molar_mass_O)) * 100 = 9.15 :=
by
  sorry

end mass_percentage_H_in_C4H8O2_l1301_130100


namespace smallest_integer_to_make_y_perfect_square_l1301_130109

-- Define y as given in the problem
def y : ℕ :=
  2^33 * 3^54 * 4^45 * 5^76 * 6^57 * 7^38 * 8^69 * 9^10

-- Smallest integer n such that (y * n) is a perfect square
theorem smallest_integer_to_make_y_perfect_square
  : ∃ n : ℕ, (∀ k : ℕ, y * n = k * k) ∧ (∀ m : ℕ, (∀ k : ℕ, y * m = k * k) → n ≤ m) := 
sorry

end smallest_integer_to_make_y_perfect_square_l1301_130109


namespace units_digit_of_product_l1301_130179

def is_units_digit (n : ℕ) (d : ℕ) : Prop := n % 10 = d

theorem units_digit_of_product : 
  is_units_digit (6 * 8 * 9 * 10 * 12) 0 := 
by
  sorry

end units_digit_of_product_l1301_130179


namespace count_implications_l1301_130131

theorem count_implications (p q r : Prop) :
  ((p ∧ q ∧ ¬r → ((q → p) → ¬r)) ∧ 
   (¬p ∧ ¬q ∧ ¬r → ((q → p) → ¬r)) ∧ 
   (p ∧ ¬q ∧ r → ¬ ((q → p) → ¬r)) ∧ 
   (¬p ∧ q ∧ ¬r → ((q → p) → ¬r))) →
   (3 = 3) := sorry

end count_implications_l1301_130131


namespace tan_double_angle_identity_l1301_130177

theorem tan_double_angle_identity (theta : ℝ) (h1 : 0 < theta ∧ theta < Real.pi / 2)
  (h2 : Real.sin theta - Real.cos theta = Real.sqrt 5 / 5) :
  Real.tan (2 * theta) = -(4 / 3) := 
by
  sorry

end tan_double_angle_identity_l1301_130177


namespace trig_identity_l1301_130118

theorem trig_identity (α : ℝ) :
  (Real.cos (α - 35 * Real.pi / 180) * Real.cos (25 * Real.pi / 180 + α) +
   Real.sin (α - 35 * Real.pi / 180) * Real.sin (25 * Real.pi / 180 + α)) = 1 / 2 :=
by
  sorry

end trig_identity_l1301_130118


namespace tile_size_l1301_130137

theorem tile_size (length width : ℕ) (total_tiles : ℕ) 
  (h_length : length = 48) 
  (h_width : width = 72) 
  (h_total_tiles : total_tiles = 96) : 
  ((length * width) / total_tiles) = 36 := 
by
  sorry

end tile_size_l1301_130137


namespace smallest_square_l1301_130158

theorem smallest_square 
  (a b : ℕ) 
  (h1 : 15 * a + 16 * b = m ^ 2) 
  (h2 : 16 * a - 15 * b = n ^ 2)
  (hm : m > 0) 
  (hn : n > 0) : 
  min (15 * a + 16 * b) (16 * a - 15 * b) = 481 ^ 2 := 
sorry

end smallest_square_l1301_130158


namespace incenter_ineq_l1301_130191

open Real

-- Definitions of the incenter and angle bisector intersection points
def incenter (A B C : Point) : Point := sorry
def angle_bisector_intersect (A B C I : Point) (angle_vertex : Point) : Point := sorry
def AI (A I : Point) : ℝ := sorry
def AA' (A A' : Point) : ℝ := sorry
def BI (B I : Point) : ℝ := sorry
def BB' (B B' : Point) : ℝ := sorry
def CI (C I : Point) : ℝ := sorry
def CC' (C C' : Point) : ℝ := sorry

-- Statement of the problem
theorem incenter_ineq 
    (A B C I A' B' C' : Point)
    (h1 : I = incenter A B C)
    (h2 : A' = angle_bisector_intersect A B C I A)
    (h3 : B' = angle_bisector_intersect A B C I B)
    (h4 : C' = angle_bisector_intersect A B C I C) :
    (1/4 : ℝ) < (AI A I * BI B I * CI C I) / (AA' A A' * BB' B B' * CC' C C') ∧ 
    (AI A I * BI B I * CI C I) / (AA' A A' * BB' B B' * CC' C C') ≤ (8/27 : ℝ) :=
sorry

end incenter_ineq_l1301_130191


namespace circle_center_l1301_130193

theorem circle_center (a b : ℝ)
  (passes_through_point : (a - 0)^2 + (b - 9)^2 = r^2)
  (is_tangent : (a - 3)^2 + (b - 9)^2 = r^2 ∧ b = 6 * (a - 3) + 9 ∧ (b - 9) / (a - 3) = -1/6) :
  a = 3/2 ∧ b = 37/4 := 
by 
  sorry

end circle_center_l1301_130193


namespace base6_addition_problem_l1301_130125

theorem base6_addition_problem (X Y : ℕ) (h1 : Y + 3 = X) (h2 : X + 2 = 7) : X + Y = 7 := 
by
  sorry

end base6_addition_problem_l1301_130125


namespace largest_stamps_per_page_l1301_130104

-- Definitions of the conditions
def stamps_book1 : ℕ := 1260
def stamps_book2 : ℕ := 1470

-- Statement to be proven: The largest number of stamps per page (gcd of 1260 and 1470)
theorem largest_stamps_per_page : Nat.gcd stamps_book1 stamps_book2 = 210 :=
by
  sorry

end largest_stamps_per_page_l1301_130104


namespace angle_bisector_ratio_l1301_130113

theorem angle_bisector_ratio (XY XZ YZ : ℝ) (hXY : XY = 8) (hXZ : XZ = 6) (hYZ : YZ = 4) :
  ∃ (Q : Point) (YQ QV : ℝ), YQ / QV = 2 :=
by
  sorry

end angle_bisector_ratio_l1301_130113


namespace base7_addition_l1301_130149

theorem base7_addition : (21 : ℕ) + 254 = 505 :=
by sorry

end base7_addition_l1301_130149


namespace distance_between_M_and_focus_l1301_130103

theorem distance_between_M_and_focus
  (θ : ℝ)
  (x y : ℝ)
  (M : ℝ × ℝ := (1/2, 0))
  (F : ℝ × ℝ := (0, 1/2))
  (hx : x = 2 * Real.cos θ)
  (hy : y = 1 + Real.cos (2 * θ)) :
  Real.sqrt ((M.1 - F.1)^2 + (M.2 - F.2)^2) = Real.sqrt 2 / 2 :=
by
  sorry

end distance_between_M_and_focus_l1301_130103


namespace number_of_rabbits_is_38_l1301_130110

-- Conditions: 
def ducks : ℕ := 52
def chickens : ℕ := 78
def condition (ducks rabbits chickens : ℕ) : Prop := 
  chickens = ducks + rabbits - 12

-- Statement: Prove that the number of rabbits is 38
theorem number_of_rabbits_is_38 : ∃ R : ℕ, condition ducks R chickens ∧ R = 38 := by
  sorry

end number_of_rabbits_is_38_l1301_130110


namespace solve_problem_l1301_130166

def question : ℝ := -7.8
def answer : ℕ := 22

theorem solve_problem : 2 * (⌊|question|⌋) + (|⌊question⌋|) = answer := by
  sorry

end solve_problem_l1301_130166


namespace area_comparison_l1301_130144

noncomputable def area_difference_decagon (s : ℝ) : ℝ := 
  let R := s / (2 * Real.sin (Real.pi / 10))
  let r := s / (2 * Real.tan (Real.pi / 10))
  Real.pi * (R^2 - r^2)

noncomputable def area_difference_nonagon (s : ℝ) : ℝ := 
  let R := s / (2 * Real.sin (Real.pi / 9))
  let r := s / (2 * Real.tan (Real.pi / 9))
  Real.pi * (R^2 - r^2)

theorem area_comparison :
  (area_difference_decagon 3 > area_difference_nonagon 3) :=
sorry

end area_comparison_l1301_130144


namespace drive_photos_storage_l1301_130123

theorem drive_photos_storage (photo_size: ℝ) (num_photos_with_videos: ℕ) (photo_storage_with_videos: ℝ) (video_size: ℝ) (num_videos_with_photos: ℕ) : 
  num_photos_with_videos * photo_size + num_videos_with_photos * video_size = 3000 → 
  (3000 / photo_size) = 2000 :=
by
  sorry

end drive_photos_storage_l1301_130123


namespace total_sum_is_2696_l1301_130126

def numbers := (100, 4900)

def harmonic_mean (a b : ℕ) : ℕ :=
  2 * a * b / (a + b)

def arithmetic_mean (a b : ℕ) : ℕ :=
  (a + b) / 2

theorem total_sum_is_2696 : 
  harmonic_mean numbers.1 numbers.2 + arithmetic_mean numbers.1 numbers.2 = 2696 :=
by
  sorry

end total_sum_is_2696_l1301_130126


namespace Ed_lost_marble_count_l1301_130197

variable (D : ℕ) -- Number of marbles Doug has

noncomputable def Ed_initial := D + 19 -- Ed initially had D + 19 marbles
noncomputable def Ed_now := D + 8 -- Ed now has D + 8 marbles
noncomputable def Ed_lost := Ed_initial D - Ed_now D -- Ed lost Ed_initial - Ed_now marbles

theorem Ed_lost_marble_count : Ed_lost D = 11 := by 
  sorry

end Ed_lost_marble_count_l1301_130197


namespace least_common_multiple_1260_980_l1301_130154

def LCM (a b : ℕ) : ℕ :=
  a * b / Nat.gcd a b

theorem least_common_multiple_1260_980 : LCM 1260 980 = 8820 := by
  sorry

end least_common_multiple_1260_980_l1301_130154


namespace cost_of_child_ticket_l1301_130165

theorem cost_of_child_ticket
  (total_seats : ℕ)
  (adult_ticket_price : ℕ)
  (num_children : ℕ)
  (total_revenue : ℕ)
  (H1 : total_seats = 250)
  (H2 : adult_ticket_price = 6)
  (H3 : num_children = 188)
  (H4 : total_revenue = 1124) :
  let num_adults := total_seats - num_children
  let revenue_from_adults := num_adults * adult_ticket_price
  let cost_of_child_ticket := (total_revenue - revenue_from_adults) / num_children
  cost_of_child_ticket = 4 :=
by
  sorry

end cost_of_child_ticket_l1301_130165


namespace each_persons_share_l1301_130162

def total_bill : ℝ := 211.00
def number_of_people : ℕ := 5
def tip_rate : ℝ := 0.15

theorem each_persons_share :
  (total_bill * (1 + tip_rate)) / number_of_people = 48.53 := 
by sorry

end each_persons_share_l1301_130162


namespace find_a_l1301_130170

theorem find_a (a : ℝ) (f : ℝ → ℝ) (f' : ℝ → ℝ) 
  (hf : ∀ x, f x = a * x^3 + 3 * x^2 + 2)
  (hf' : ∀ x, f' x = 3 * a * x^2 + 6 * x) 
  (h : f' (-1) = 4) : 
  a = (10 : ℝ) / 3 := 
sorry

end find_a_l1301_130170


namespace books_left_over_l1301_130151

theorem books_left_over (n_boxes : ℕ) (books_per_box : ℕ) (new_box_capacity : ℕ) :
  n_boxes = 1575 → books_per_box = 45 → new_box_capacity = 46 →
  (n_boxes * books_per_box) % new_box_capacity = 15 :=
by
  intro h1 h2 h3
  rw [h1, h2, h3]
  -- Actual proof steps would go here
  sorry

end books_left_over_l1301_130151


namespace total_growing_space_is_correct_l1301_130117

def garden_bed_area (length : ℕ) (width : ℕ) (count : ℕ) : ℕ :=
  length * width * count

def total_growing_space : ℕ :=
  garden_bed_area 5 4 3 +
  garden_bed_area 6 3 4 +
  garden_bed_area 7 5 2 +
  garden_bed_area 8 4 1

theorem total_growing_space_is_correct :
  total_growing_space = 234 := by
  sorry

end total_growing_space_is_correct_l1301_130117


namespace sandwiches_provided_now_l1301_130157

-- Define the initial number of sandwich kinds
def initialSandwichKinds : ℕ := 23

-- Define the number of sold out sandwich kinds
def soldOutSandwichKinds : ℕ := 14

-- Define the proof that the actual number of sandwich kinds provided now
theorem sandwiches_provided_now : initialSandwichKinds - soldOutSandwichKinds = 9 :=
by
  -- The proof goes here
  sorry

end sandwiches_provided_now_l1301_130157


namespace modulus_of_product_l1301_130190

namespace ComplexModule

open Complex

-- Definition of the complex numbers z1 and z2
def z1 : ℂ := 1 + I
def z2 : ℂ := 2 - I

-- Definition of their product z1z2
def z1z2 : ℂ := z1 * z2

-- Statement we need to prove (the modulus of z1z2 is √10)
theorem modulus_of_product : abs z1z2 = Real.sqrt 10 := by
  sorry

end ComplexModule

end modulus_of_product_l1301_130190


namespace chimes_1000_on_march_7_l1301_130185

theorem chimes_1000_on_march_7 : 
  ∀ (initial_time : Nat) (start_date : Nat) (chimes_before_noon : Nat) 
  (chimes_per_day : Nat) (target_chime : Nat) (final_date : Nat),
  initial_time = 10 * 60 + 15 ∧
  start_date = 26 ∧
  chimes_before_noon = 25 ∧
  chimes_per_day = 103 ∧
  target_chime = 1000 ∧
  final_date = start_date + (target_chime - chimes_before_noon) / chimes_per_day ∧
  (target_chime - chimes_before_noon) % chimes_per_day ≤ chimes_per_day
  → final_date = 7 := 
by
  intros
  sorry

end chimes_1000_on_march_7_l1301_130185


namespace phi_varphi_difference_squared_l1301_130198

theorem phi_varphi_difference_squared :
  ∀ (Φ φ : ℝ), (Φ ≠ φ) → (Φ^2 - 2*Φ - 1 = 0) → (φ^2 - 2*φ - 1 = 0) →
  (Φ - φ)^2 = 8 :=
by
  intros Φ φ distinct hΦ hφ
  sorry

end phi_varphi_difference_squared_l1301_130198


namespace boat_speed_in_still_water_l1301_130102

theorem boat_speed_in_still_water:
  ∀ (V_b : ℝ) (V_s : ℝ) (D : ℝ),
    V_s = 3 → 
    (D = (V_b + V_s) * 1) → 
    (D = (V_b - V_s) * 1.5) → 
    V_b = 15 :=
by
  intros V_b V_s D V_s_eq H_downstream H_upstream
  sorry

end boat_speed_in_still_water_l1301_130102


namespace pool_depth_multiple_l1301_130169

theorem pool_depth_multiple
  (johns_pool : ℕ)
  (sarahs_pool : ℕ)
  (h1 : johns_pool = 15)
  (h2 : sarahs_pool = 5)
  (h3 : johns_pool = x * sarahs_pool + 5) :
  x = 2 := by
  sorry

end pool_depth_multiple_l1301_130169


namespace Natalia_Tuesday_distance_l1301_130188

theorem Natalia_Tuesday_distance :
  ∃ T : ℕ, (40 + T + T / 2 + (40 + T / 2) = 180) ∧ T = 33 :=
by
  existsi 33
  -- proof can be filled here
  sorry

end Natalia_Tuesday_distance_l1301_130188


namespace ticket_cost_before_rally_l1301_130133

-- We define the variables and constants given in the problem
def total_attendance : ℕ := 750
def tickets_before_rally : ℕ := 475
def tickets_at_door : ℕ := total_attendance - tickets_before_rally
def cost_at_door : ℝ := 2.75
def total_receipts : ℝ := 1706.25

-- Problem statement: Prove that the cost of each ticket bought before the rally (x) is 2 dollars.
theorem ticket_cost_before_rally (x : ℝ) 
  (h₁ : tickets_before_rally * x + tickets_at_door * cost_at_door = total_receipts) :
  x = 2 :=
by
  sorry

end ticket_cost_before_rally_l1301_130133


namespace minimum_value_of_f_l1301_130155

noncomputable def f (x : ℝ) : ℝ := (Real.cos (2 * x) + 2 * Real.sin x)

theorem minimum_value_of_f : ∃ x : ℝ, (∀ y : ℝ, f x ≤ f y) ∧ f x = -3 := 
  sorry

end minimum_value_of_f_l1301_130155
