import Mathlib

namespace six_digit_count_div_by_217_six_digit_count_div_by_218_l1791_179117

-- Definitions for the problem
def six_digit_format (n : ℕ) : Prop :=
  ∃ a b : ℕ, (0 ≤ a ∧ a < 10) ∧ (0 ≤ b ∧ b < 10) ∧ n = 100001 * a + 10010 * b + 100 * a + 10 * b + a

def divisible_by (n : ℕ) (divisor : ℕ) : Prop :=
  n % divisor = 0

-- Problem Part a: How many six-digit numbers of the form are divisible by 217
theorem six_digit_count_div_by_217 :
  ∃ count : ℕ, count = 3 ∧ ∀ n : ℕ, six_digit_format n → divisible_by n 217  → (n = 313131 ∨ n = 626262 ∨ n = 939393) :=
sorry

-- Problem Part b: How many six-digit numbers of the form are divisible by 218
theorem six_digit_count_div_by_218 :
  ∀ n : ℕ, six_digit_format n → divisible_by n 218 → false :=
sorry

end six_digit_count_div_by_217_six_digit_count_div_by_218_l1791_179117


namespace no_naturals_satisfy_divisibility_condition_l1791_179166

theorem no_naturals_satisfy_divisibility_condition :
  ∀ (a b c : ℕ), ¬ (2013 * (a * b + b * c + c * a) ∣ a^2 + b^2 + c^2) :=
by
  sorry

end no_naturals_satisfy_divisibility_condition_l1791_179166


namespace curve_trajectory_a_eq_1_curve_fixed_point_a_ne_1_l1791_179179

noncomputable def curve (x y a : ℝ) : ℝ :=
  x^2 + y^2 - 2 * a * x + 2 * (a - 2) * y + 2 

theorem curve_trajectory_a_eq_1 :
  ∃! (x y : ℝ), curve x y 1 = 0 ∧ x = 1 ∧ y = 1 := by
  sorry

theorem curve_fixed_point_a_ne_1 (a : ℝ) (ha : a ≠ 1) :
  curve 1 1 a = 0 := by
  sorry

end curve_trajectory_a_eq_1_curve_fixed_point_a_ne_1_l1791_179179


namespace intersection_of_A_and_B_l1791_179124

def A : Set ℕ := {1, 2, 3, 4, 5}
def B : Set ℕ := {1, 2, 4, 6}

theorem intersection_of_A_and_B : A ∩ B = {1, 2, 4} := by
  sorry

end intersection_of_A_and_B_l1791_179124


namespace interval_length_correct_l1791_179171

def sin_log_interval_sum : ℝ := sorry

theorem interval_length_correct :
  sin_log_interval_sum = 2^π / (1 + 2^π) :=
by
  -- Definitions
  let is_valid_x (x : ℝ) := x < 1 ∧ x > 0 ∧ (Real.sin (Real.log x / Real.log 2)) < 0
  
  -- Assertion
  sorry

end interval_length_correct_l1791_179171


namespace balcony_height_l1791_179189

-- Definitions for conditions given in the problem

def final_position := 0 -- y, since the ball hits the ground
def initial_velocity := 5 -- v₀ in m/s
def time_elapsed := 3 -- t in seconds
def gravity := 10 -- g in m/s²

theorem balcony_height : 
  ∃ h₀ : ℝ, final_position = h₀ + initial_velocity * time_elapsed - (1/2) * gravity * time_elapsed^2 ∧ h₀ = 30 := 
by 
  sorry

end balcony_height_l1791_179189


namespace find_a_b_l1791_179175

noncomputable def A : Set ℝ := { x : ℝ | -1 < x ∧ x < 3 }
noncomputable def B : Set ℝ := { x : ℝ | -3 < x ∧ x < 2 }
noncomputable def sol_set (a b : ℝ) : Set ℝ := { x : ℝ | x^2 + a * x + b < 0 }

theorem find_a_b :
  (sol_set (-2) (3 - 6)) = A ∩ B → (-1) + (-2) = -3 :=
by
  intros h1
  sorry

end find_a_b_l1791_179175


namespace expr_containing_x_to_y_l1791_179193

theorem expr_containing_x_to_y (x y : ℝ) (h : 2 * x - y = 3) : y = 2 * x - 3 :=
by
  -- proof steps would be here
  sorry

end expr_containing_x_to_y_l1791_179193


namespace find_initial_red_balloons_l1791_179174

-- Define the initial state of balloons and the assumption.
def initial_blue_balloons : ℕ := 4
def red_balloons_after_inflation (R : ℕ) : ℕ := R + 2
def blue_balloons_after_inflation : ℕ := initial_blue_balloons + 2
def total_balloons (R : ℕ) : ℕ := red_balloons_after_inflation R + blue_balloons_after_inflation

-- Define the likelihood condition.
def likelihood_red (R : ℕ) : Prop := (red_balloons_after_inflation R : ℚ) / (total_balloons R : ℚ) = 0.4

-- Statement of the problem.
theorem find_initial_red_balloons (R : ℕ) (h : likelihood_red R) : R = 2 := by
  sorry

end find_initial_red_balloons_l1791_179174


namespace handshakes_mod_500_l1791_179107

theorem handshakes_mod_500 : 
  let n := 10
  let k := 3
  let M := 199584 -- total number of ways calculated from the problem
  (n = 10) -> (k = 3) -> (M % 500 = 84) :=
by
  intros
  sorry

end handshakes_mod_500_l1791_179107


namespace total_production_l1791_179161

variable (x : ℕ) -- total units produced by 4 machines in 6 days
variable (R : ℕ) -- rate of production per machine per day

-- Condition 1: 4 machines can produce x units in 6 days
axiom rate_definition : 4 * R * 6 = x

-- Question: Prove the total amount of product produced by 16 machines in 3 days is 2x
theorem total_production : 16 * R * 3 = 2 * x :=
by 
  sorry

end total_production_l1791_179161


namespace cell_phones_sold_l1791_179177

theorem cell_phones_sold (init_samsung init_iphone final_samsung final_iphone defective_samsung defective_iphone : ℕ)
    (h1 : init_samsung = 14) 
    (h2 : init_iphone = 8) 
    (h3 : final_samsung = 10) 
    (h4 : final_iphone = 5) 
    (h5 : defective_samsung = 2) 
    (h6 : defective_iphone = 1) : 
    init_samsung - defective_samsung - final_samsung + 
    init_iphone - defective_iphone - final_iphone = 4 := 
by
  sorry

end cell_phones_sold_l1791_179177


namespace hockey_league_games_l1791_179187

def num_teams : ℕ := 18
def encounters_per_pair : ℕ := 10
def num_games (n : ℕ) (k : ℕ) : ℕ := (n * (n - 1)) / 2 * k

theorem hockey_league_games :
  num_games num_teams encounters_per_pair = 1530 :=
by
  sorry

end hockey_league_games_l1791_179187


namespace value_of_a6_l1791_179120

theorem value_of_a6 
  (S : ℕ → ℤ)
  (a : ℕ → ℤ) 
  (hS : ∀ n, S n = 3 * n^2 - 5 * n)
  (ha : ∀ n, n ≥ 2 → a n = S n - S (n - 1)) 
  (h1 : a 1 = S 1):
  a 6 = 28 :=
sorry

end value_of_a6_l1791_179120


namespace range_of_p_l1791_179119

def A (x : ℝ) : Prop := -2 < x ∧ x < 5
def B (p : ℝ) (x : ℝ) : Prop := p + 1 < x ∧ x < 2 * p - 1

theorem range_of_p (p : ℝ) :
  (∀ x, A x ∨ B p x → A x) ↔ p ≤ 3 :=
by
  sorry

end range_of_p_l1791_179119


namespace sum_at_simple_interest_l1791_179141

theorem sum_at_simple_interest
  (P R : ℝ)  -- P is the principal amount, R is the rate of interest
  (H1 : (9 * P * (R + 5) / 100 - 9 * P * R / 100 = 1350)) :
  P = 3000 :=
by
  sorry

end sum_at_simple_interest_l1791_179141


namespace square_perimeter_l1791_179109

theorem square_perimeter (s : ℝ) (h : s^2 = 625) : 4 * s = 100 := 
by {
  sorry
}

end square_perimeter_l1791_179109


namespace builder_installed_windows_l1791_179135

-- Conditions
def total_windows : ℕ := 14
def hours_per_window : ℕ := 8
def remaining_hours : ℕ := 48

-- Definition for the problem statement
def installed_windows := total_windows - remaining_hours / hours_per_window

-- The hypothesis we need to prove
theorem builder_installed_windows : installed_windows = 8 := by
  sorry

end builder_installed_windows_l1791_179135


namespace grogg_possible_cubes_l1791_179105

theorem grogg_possible_cubes (a b c : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_prob : (a - 2) * (b - 2) * (c - 2) / (a * b * c) = 1 / 5) :
  a * b * c = 120 ∨ a * b * c = 160 ∨ a * b * c = 240 ∨ a * b * c = 360 := 
sorry

end grogg_possible_cubes_l1791_179105


namespace disease_given_positive_l1791_179176

-- Definitions and conditions extracted from the problem
def Pr_D : ℚ := 1 / 200
def Pr_Dc : ℚ := 1 - Pr_D
def Pr_T_D : ℚ := 1
def Pr_T_Dc : ℚ := 0.05

-- Derived probabilites from given conditions
def Pr_T : ℚ := Pr_T_D * Pr_D + Pr_T_Dc * Pr_Dc

-- Statement for the probability using Bayes' theorem
theorem disease_given_positive :
  (Pr_T_D * Pr_D) / Pr_T = 20 / 219 :=
sorry

end disease_given_positive_l1791_179176


namespace triangle_third_side_l1791_179180

theorem triangle_third_side (x : ℝ) (h1 : x > 2) (h2 : x < 6) : x = 5 :=
sorry

end triangle_third_side_l1791_179180


namespace arithmetic_sequence_product_l1791_179185

theorem arithmetic_sequence_product 
  (b : ℕ → ℤ) 
  (h_arith : ∀ n, b n = b 0 + (n : ℤ) * (b 1 - b 0))
  (h_inc : ∀ n, b n ≤ b (n + 1))
  (h4_5 : b 4 * b 5 = 21) : 
  b 3 * b 6 = -779 ∨ b 3 * b 6 = -11 := 
by 
  sorry

end arithmetic_sequence_product_l1791_179185


namespace find_x_l1791_179157

theorem find_x (h : 0.60 / x = 6 / 2) : x = 0.20 :=
by
  sorry

end find_x_l1791_179157


namespace y_completes_work_in_seventy_days_l1791_179160

def work_days (mahesh_days : ℕ) (mahesh_work_days : ℕ) (rajesh_days : ℕ) (y_days : ℕ) : Prop :=
  let mahesh_rate := (1:ℝ) / mahesh_days
  let rajesh_rate := (1:ℝ) / rajesh_days
  let work_done_by_mahesh := mahesh_rate * mahesh_work_days
  let remaining_work := (1:ℝ) - work_done_by_mahesh
  let rajesh_remaining_work_days := remaining_work / rajesh_rate
  let y_rate := (1:ℝ) / y_days
  y_rate = rajesh_rate

theorem y_completes_work_in_seventy_days :
  work_days 35 20 30 70 :=
by
  -- This is where the proof would go
  sorry

end y_completes_work_in_seventy_days_l1791_179160


namespace proportion_option_B_true_l1791_179172

theorem proportion_option_B_true {a b c d : ℚ} (h : a / b = c / d) : 
  (a + c) / c = (b + d) / d := 
by 
  sorry

end proportion_option_B_true_l1791_179172


namespace ab_plus_b_l1791_179134

theorem ab_plus_b (A B : ℤ) (h1 : A * B = 10) (h2 : 3 * A + 7 * B = 51) : A * B + B = 12 :=
by
  sorry

end ab_plus_b_l1791_179134


namespace find_k_of_symmetry_l1791_179145

noncomputable def f (x k : ℝ) := Real.sin (2 * x) + k * Real.cos (2 * x)

theorem find_k_of_symmetry (k : ℝ) :
  (∃ x, x = (Real.pi / 6) ∧ f x k = f (Real.pi / 6 - x) k) →
  k = Real.sqrt 3 / 3 :=
sorry

end find_k_of_symmetry_l1791_179145


namespace lunch_customers_is_127_l1791_179163

-- Define the conditions based on the given problem
def breakfast_customers : ℕ := 73
def dinner_customers : ℕ := 87
def total_customers_on_saturday : ℕ := 574
def total_customers_on_friday : ℕ := total_customers_on_saturday / 2

-- Define the variable representing the lunch customers
variable (L : ℕ)

-- State the proposition we want to prove
theorem lunch_customers_is_127 :
  breakfast_customers + L + dinner_customers = total_customers_on_friday → L = 127 := by {
  sorry
}

end lunch_customers_is_127_l1791_179163


namespace intersection_A_complement_B_range_of_a_l1791_179162

-- Define sets A and B with their respective conditions
def U : Set ℝ := Set.univ
def A (a : ℝ) : Set ℝ := {x | (x - 2) * (x - (3 * a + 1)) < 0}
def B (a : ℝ) : Set ℝ := {x | (x - 2 * a) / (x - (a^2 + 1)) < 0}

-- Question 1: Prove the intersection when a = 2
theorem intersection_A_complement_B (a : ℝ) (h : a = 2) : 
  A a ∩ (U \ B a) = {x | 2 < x ∧ x ≤ 4} ∪ {x | 5 ≤ x ∧ x < 7} :=
by sorry

-- Question 2: Find the range of a such that A ∪ B = A given a ≠ 1
theorem range_of_a (a : ℝ) (h : a ≠ 1) : 
  (A a ∪ B a = A a) ↔ (1 < a ∧ a ≤ 3 ∨ a = -1) :=
by sorry

end intersection_A_complement_B_range_of_a_l1791_179162


namespace cat_food_percentage_l1791_179139

theorem cat_food_percentage (D C : ℝ) (h1 : 7 * D + 4 * C = 8 * D) (h2 : 4 * C = D) : 
  (C / (7 * D + D)) * 100 = 3.125 := by
  sorry

end cat_food_percentage_l1791_179139


namespace sides_of_triangle_l1791_179147

-- Definitions from conditions
variables (a b c : ℕ) (r bk kc : ℕ)
def is_tangent_split : Prop := bk = 8 ∧ kc = 6
def inradius : Prop := r = 4

-- Main theorem statement
theorem sides_of_triangle (h1 : is_tangent_split bk kc) (h2 : inradius r) : a + 6 = 13 ∧ a + 8 = 15 ∧ b = 14 := by
  sorry

end sides_of_triangle_l1791_179147


namespace trapezoid_area_is_correct_l1791_179173

noncomputable def isosceles_trapezoid_area : ℝ :=
  let a : ℝ := 12
  let b : ℝ := 24 - 12 * Real.sqrt 2
  let h : ℝ := 6 * Real.sqrt 2
  (24 + b) / 2 * h

theorem trapezoid_area_is_correct :
  let a := 12
  let b := 24 - 12 * Real.sqrt 2
  let h := 6 * Real.sqrt 2
  (24 + b) / 2 * h = 144 * Real.sqrt 2 - 72 :=
by
  sorry

end trapezoid_area_is_correct_l1791_179173


namespace polynomial_real_roots_abs_c_geq_2_l1791_179170

-- Definition of the polynomial P(x)
def P (x : ℝ) (a b c : ℝ) : ℝ := x^6 + a*x^5 + b*x^4 + c*x^3 + b*x^2 + a*x + 1

-- Statement of the problem: Given that P(x) has six distinct real roots, prove |c| ≥ 2
theorem polynomial_real_roots_abs_c_geq_2 (a b c : ℝ) :
  (∃ r1 r2 r3 r4 r5 r6 : ℝ, r1 ≠ r2 ∧ r1 ≠ r3 ∧ r1 ≠ r4 ∧ r1 ≠ r5 ∧ r1 ≠ r6 ∧
                           r2 ≠ r3 ∧ r2 ≠ r4 ∧ r2 ≠ r5 ∧ r2 ≠ r6 ∧
                           r3 ≠ r4 ∧ r3 ≠ r5 ∧ r3 ≠ r6 ∧
                           r4 ≠ r5 ∧ r4 ≠ r6 ∧
                           r5 ≠ r6 ∧
                           P r1 a b c = 0 ∧ P r2 a b c = 0 ∧ P r3 a b c = 0 ∧
                           P r4 a b c = 0 ∧ P r5 a b c = 0 ∧ P r6 a b c = 0) →
  |c| ≥ 2 := by
  sorry

end polynomial_real_roots_abs_c_geq_2_l1791_179170


namespace total_area_of_rug_l1791_179148

theorem total_area_of_rug :
  let length_rect := 6
  let width_rect := 4
  let base_parallelogram := 3
  let height_parallelogram := 4
  let area_rect := length_rect * width_rect
  let area_parallelogram := base_parallelogram * height_parallelogram
  let total_area := area_rect + 2 * area_parallelogram
  total_area = 48 := by sorry

end total_area_of_rug_l1791_179148


namespace sufficient_condition_not_necessary_condition_l1791_179188

variable (a b : ℝ)

theorem sufficient_condition (hab : (a - b) * a^2 < 0) : a < b :=
by
  sorry

theorem not_necessary_condition (h : a < b) : (a - b) * a^2 < 0 ∨ (a - b) * a^2 = 0 :=
by
  sorry

end sufficient_condition_not_necessary_condition_l1791_179188


namespace alpha_value_l1791_179155

theorem alpha_value (m : ℝ) (α : ℝ) (h : m * 8 ^ α = 1 / 4) : α = -2 / 3 :=
by
  sorry

end alpha_value_l1791_179155


namespace proposition_2_proposition_4_l1791_179110

-- Definitions from conditions.
def circle_M (x y q : ℝ) : Prop := (x + Real.cos q)^2 + (y - Real.sin q)^2 = 1
def line_l (y k x : ℝ) : Prop := y = k * x

-- Prove that the line l and circle M always intersect for any real k and q.
theorem proposition_2 : ∀ (k q : ℝ), ∃ (x y : ℝ), circle_M x y q ∧ line_l y k x := sorry

-- Prove that for any real k, there exists a real q such that the line l is tangent to the circle M.
theorem proposition_4 : ∀ (k : ℝ), ∃ (q x y : ℝ), circle_M x y q ∧ line_l y k x ∧
  (abs (Real.sin q + k * Real.cos q) = 1 / Real.sqrt (1 + k^2)) := sorry

end proposition_2_proposition_4_l1791_179110


namespace max_value_of_y_l1791_179106

open Real

noncomputable def y (x : ℝ) := 1 + 1 / (x^2 + 2*x + 2)

theorem max_value_of_y : ∃ x : ℝ, y x = 2 :=
sorry

end max_value_of_y_l1791_179106


namespace number_of_adults_had_meal_l1791_179108

theorem number_of_adults_had_meal (A : ℝ) :
  let num_children_food : ℝ := 63
  let food_for_adults : ℝ := 70
  let food_for_children : ℝ := 90
  (food_for_children - A * (food_for_children / food_for_adults) = num_children_food) →
  A = 21 :=
by
  intros num_children_food food_for_adults food_for_children h
  have h2 : 90 - A * (90 / 70) = 63 := h
  sorry

end number_of_adults_had_meal_l1791_179108


namespace initial_speed_l1791_179169

variable (v : ℝ)
variable (h1 : (v / 2) + 2 * v = 75)

theorem initial_speed (v : ℝ) (h1 : (v / 2) + 2 * v = 75) : v = 30 :=
sorry

end initial_speed_l1791_179169


namespace stratified_sampling_l1791_179181

-- Conditions
def total_students : ℕ := 1200
def freshmen : ℕ := 300
def sophomores : ℕ := 400
def juniors : ℕ := 500
def sample_size : ℕ := 60
def probability : ℚ := sample_size / total_students

-- Number of students to be sampled from each grade
def freshmen_sampled : ℚ := freshmen * probability
def sophomores_sampled : ℚ := sophomores * probability
def juniors_sampled : ℚ := juniors * probability

-- Theorem to prove
theorem stratified_sampling :
  freshmen_sampled = 15 ∧ sophomores_sampled = 20 ∧ juniors_sampled = 25 :=
by
  -- The actual proof would go here
  sorry

end stratified_sampling_l1791_179181


namespace sequence_2011_l1791_179138

theorem sequence_2011 :
  ∀ (a : ℕ → ℤ), (a 1 = 1) →
                  (a 2 = 2) →
                  (∀ n : ℕ, a (n + 2) = a (n + 1) - a n) →
                  a 2011 = 1 :=
by {
  -- Insert proof here
  sorry
}

end sequence_2011_l1791_179138


namespace sum_first_2014_terms_l1791_179121

def sequence_is_arithmetic (a : ℕ → ℕ) :=
  ∀ n : ℕ, a (n + 1) = a n + a 2

def first_arithmetic_sum (a : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ) :=
  S n = (n * (n - 1)) / 2

theorem sum_first_2014_terms (a : ℕ → ℕ) (S : ℕ → ℕ) 
  (h1 : sequence_is_arithmetic a) 
  (h2 : a 3 = 2) : 
  S 2014 = 1007 * 2013 :=
sorry

end sum_first_2014_terms_l1791_179121


namespace area_of_square_is_1225_l1791_179111

-- Given some basic definitions and conditions
variable (s : ℝ) -- side of the square which is the radius of the circle
variable (length : ℝ := (2 / 5) * s)
variable (breadth : ℝ := 10)
variable (area_rectangle : ℝ := length * breadth)

-- Statement to prove
theorem area_of_square_is_1225 
  (h1 : length = (2 / 5) * s)
  (h2 : breadth = 10)
  (h3 : area_rectangle = 140) : 
  s^2 = 1225 := by
    sorry

end area_of_square_is_1225_l1791_179111


namespace distance_between_clocks_centers_l1791_179197

variable (M m : ℝ)

theorem distance_between_clocks_centers :
  ∃ (c : ℝ), (|c| = (1/2) * (M + m)) := by
  sorry

end distance_between_clocks_centers_l1791_179197


namespace ajay_walks_distance_l1791_179184

theorem ajay_walks_distance (speed : ℝ) (time : ℝ) (distance : ℝ) 
  (h_speed : speed = 3) 
  (h_time : time = 16.666666666666668) : 
  distance = speed * time :=
by
  sorry

end ajay_walks_distance_l1791_179184


namespace mass_of_man_l1791_179132

-- Definitions of the given conditions
def boat_length : Float := 3.0
def boat_breadth : Float := 2.0
def sink_depth : Float := 0.01 -- 1 cm converted to meters
def water_density : Float := 1000.0 -- Density of water in kg/m³

-- Define the proof goal as the mass of the man
theorem mass_of_man : Float :=
by
  let volume_displaced := boat_length * boat_breadth * sink_depth
  let weight_displaced := volume_displaced * water_density
  exact weight_displaced

end mass_of_man_l1791_179132


namespace perfect_square_pairs_l1791_179122

theorem perfect_square_pairs (x y : ℕ) (a b : ℤ) :
  (x^2 + 8 * ↑y = a^2 ∧ y^2 - 8 * ↑x = b^2) →
  (∃ n : ℕ, x = n ∧ y = n + 2) ∨ (x = 7 ∧ y = 15) ∨ (x = 33 ∧ y = 17) ∨ (x = 45 ∧ y = 23) :=
by
  sorry

end perfect_square_pairs_l1791_179122


namespace factorize_expression_l1791_179186

theorem factorize_expression (x y : ℝ) : 
  x * y^2 - 6 * x * y + 9 * x = x * (y - 3)^2 := 
by sorry

end factorize_expression_l1791_179186


namespace digit_proportions_l1791_179140

theorem digit_proportions (n : ℕ) :
  (∃ (n1 n2 n5 nother : ℕ),
    n1 = n / 2 ∧
    n2 = n / 5 ∧
    n5 = n / 5 ∧
    nother = n / 10 ∧
    n1 + n2 + n5 + nother = n) ↔ n = 10 :=
by
  sorry

end digit_proportions_l1791_179140


namespace right_triangle_and_inverse_l1791_179178

theorem right_triangle_and_inverse :
  30 * 30 + 272 * 272 = 278 * 278 ∧ (∃ (n : ℕ), 0 ≤ n ∧ n < 4079 ∧ (550 * n) % 4079 = 1) :=
by
  sorry

end right_triangle_and_inverse_l1791_179178


namespace susan_spaces_to_win_l1791_179101

def spaces_in_game : ℕ := 48
def first_turn_movement : ℤ := 8
def second_turn_movement : ℤ := 2 - 5
def third_turn_movement : ℤ := 6

def total_movement : ℤ :=
  first_turn_movement + second_turn_movement + third_turn_movement

def spaces_to_win (spaces_in_game : ℕ) (total_movement : ℤ) : ℤ :=
  spaces_in_game - total_movement

theorem susan_spaces_to_win : spaces_to_win spaces_in_game total_movement = 37 := by
  sorry

end susan_spaces_to_win_l1791_179101


namespace camper_ratio_l1791_179143

theorem camper_ratio (total_campers : ℕ) (G : ℕ) (B : ℕ)
  (h1: total_campers = 96) 
  (h2: G = total_campers / 3) 
  (h3: B = total_campers - G) 
  : B / total_campers = 2 / 3 :=
  by
    sorry

end camper_ratio_l1791_179143


namespace mrs_blue_expected_tomato_yield_l1791_179127

-- Definitions for conditions
def steps_length := 3 -- each step measures 3 feet
def length_steps := 18 -- 18 steps in length
def width_steps := 25 -- 25 steps in width
def yield_per_sq_ft := 3 / 4 -- three-quarters of a pound per square foot

-- Define the total expected yield in pounds
def expected_yield : ℝ :=
  let length_ft := length_steps * steps_length
  let width_ft := width_steps * steps_length
  let area := length_ft * width_ft
  area * yield_per_sq_ft

-- The goal statement
theorem mrs_blue_expected_tomato_yield : expected_yield = 3037.5 := by
  sorry

end mrs_blue_expected_tomato_yield_l1791_179127


namespace probability_at_least_two_defective_probability_at_most_one_defective_l1791_179149

variable (P_no_defective : ℝ)
variable (P_one_defective : ℝ)
variable (P_two_defective : ℝ)
variable (P_all_defective : ℝ)

theorem probability_at_least_two_defective (hP_no_defective : P_no_defective = 0.18)
                                          (hP_one_defective : P_one_defective = 0.53)
                                          (hP_two_defective : P_two_defective = 0.27)
                                          (hP_all_defective : P_all_defective = 0.02) :
  P_two_defective + P_all_defective = 0.29 :=
  by sorry

theorem probability_at_most_one_defective (hP_no_defective : P_no_defective = 0.18)
                                          (hP_one_defective : P_one_defective = 0.53)
                                          (hP_two_defective : P_two_defective = 0.27)
                                          (hP_all_defective : P_all_defective = 0.02) :
  P_no_defective + P_one_defective = 0.71 :=
  by sorry

end probability_at_least_two_defective_probability_at_most_one_defective_l1791_179149


namespace xyz_poly_identity_l1791_179154

theorem xyz_poly_identity (x y z : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0)
  (h4 : x + y + z = 0) (h5 : xy + xz + yz ≠ 0) :
  (x^6 + y^6 + z^6) / (xyz * (xy + xz + yz)) = 6 :=
by
  sorry

end xyz_poly_identity_l1791_179154


namespace decreased_revenue_l1791_179168

variable (T C : ℝ)
def Revenue (tax consumption : ℝ) : ℝ := tax * consumption

theorem decreased_revenue (hT_new : T_new = 0.9 * T) (hC_new : C_new = 1.1 * C) :
  Revenue T_new C_new = 0.99 * (Revenue T C) := 
sorry

end decreased_revenue_l1791_179168


namespace total_rainfall_l1791_179194

-- Given conditions
def sunday_rainfall : ℕ := 4
def monday_rainfall : ℕ := sunday_rainfall + 3
def tuesday_rainfall : ℕ := 2 * monday_rainfall

-- Question: Total rainfall over the 3 days
theorem total_rainfall : sunday_rainfall + monday_rainfall + tuesday_rainfall = 25 := by
  sorry

end total_rainfall_l1791_179194


namespace truthfulness_count_l1791_179115

-- Define variables to represent the number of warriors and their response counts
def num_warriors : Nat := 33
def yes_sword : Nat := 13
def yes_spear : Nat := 15
def yes_axe : Nat := 20
def yes_bow : Nat := 27

-- Define the total number of "Yes" answers
def total_yes_answers : Nat := yes_sword + yes_spear + yes_axe + yes_bow

theorem truthfulness_count :
  ∃ x : Nat, x + 3 * (num_warriors - x) = total_yes_answers ∧ x = 12 :=
by
  sorry

end truthfulness_count_l1791_179115


namespace remaining_money_is_correct_l1791_179133

def initial_amount : ℕ := 53
def cost_toy_car : ℕ := 11
def number_toy_cars : ℕ := 2
def cost_scarf : ℕ := 10
def cost_beanie : ℕ := 14
def remaining_money : ℕ := 
  initial_amount - (cost_toy_car * number_toy_cars) - cost_scarf - cost_beanie

theorem remaining_money_is_correct : remaining_money = 7 := by
  sorry

end remaining_money_is_correct_l1791_179133


namespace not_square_difference_formula_l1791_179102

theorem not_square_difference_formula (x y : ℝ) : ¬ ∃ (a b : ℝ), (x - y) * (-x + y) = (a + b) * (a - b) := 
sorry

end not_square_difference_formula_l1791_179102


namespace ratio_of_areas_of_triangles_l1791_179164

-- Define the given conditions
variables {X Y Z T : Type}
variable (distance_XY : ℝ)
variable (distance_XZ : ℝ)
variable (distance_YZ : ℝ)
variable (is_angle_bisector : Prop)

-- Define the correct answer as a goal
theorem ratio_of_areas_of_triangles (h1 : distance_XY = 15)
    (h2 : distance_XZ = 25)
    (h3 : distance_YZ = 34)
    (h4 : is_angle_bisector) : 
    -- Ratio of the areas of triangle XYT to triangle XZT
    ∃ (ratio : ℝ), ratio = 3 / 5 :=
by
  -- This is where the proof would go, omitted with "sorry"
  sorry

end ratio_of_areas_of_triangles_l1791_179164


namespace mountain_bike_cost_l1791_179156

theorem mountain_bike_cost (savings : ℕ) (lawns : ℕ) (lawn_rate : ℕ) (newspapers : ℕ) (paper_rate : ℕ) (dogs : ℕ) (dog_rate : ℕ) (remaining : ℕ) (total_earned : ℕ) (total_before_purchase : ℕ) (cost : ℕ) : 
  savings = 1500 ∧ lawns = 20 ∧ lawn_rate = 20 ∧ newspapers = 600 ∧ paper_rate = 40 ∧ dogs = 24 ∧ dog_rate = 15 ∧ remaining = 155 ∧ 
  total_earned = (lawns * lawn_rate) + (newspapers * paper_rate / 100) + (dogs * dog_rate) ∧
  total_before_purchase = savings + total_earned ∧
  cost = total_before_purchase - remaining →
  cost = 2345 := by
  sorry

end mountain_bike_cost_l1791_179156


namespace square_measurement_error_l1791_179192

theorem square_measurement_error (S S' : ℝ) (error_percentage : ℝ)
  (area_error_percentage : ℝ) (h1 : area_error_percentage = 2.01) :
  error_percentage = 1 :=
by
  sorry

end square_measurement_error_l1791_179192


namespace intersection_P_Q_l1791_179137

def P : Set ℝ := { x : ℝ | 2 ≤ x ∧ x < 4 }
def Q : Set ℝ := { x : ℝ | 3 ≤ x }

theorem intersection_P_Q :
  P ∩ Q = { x : ℝ | 3 ≤ x ∧ x < 4 } :=
by
  sorry  -- Proof step will be provided here

end intersection_P_Q_l1791_179137


namespace greatest_third_term_arithmetic_seq_l1791_179125

theorem greatest_third_term_arithmetic_seq (a d : ℤ) (h1: a > 0) (h2: d ≥ 0) (h3: 5 * a + 10 * d = 65) : 
  a + 2 * d = 13 := 
by 
  sorry

end greatest_third_term_arithmetic_seq_l1791_179125


namespace repeating_decimal_sum_is_one_l1791_179136

noncomputable def repeating_decimal_sum : ℝ :=
  let x := (1/3 : ℝ)
  let y := (2/3 : ℝ)
  x + y

theorem repeating_decimal_sum_is_one : repeating_decimal_sum = 1 := by
  sorry

end repeating_decimal_sum_is_one_l1791_179136


namespace minimum_value_of_absolute_sum_l1791_179129

theorem minimum_value_of_absolute_sum (x : ℝ) :
  ∃ y : ℝ, (∀ x : ℝ, y ≤ |x + 1| + |x + 2| + |x + 3| + |x + 4| + |x + 5|) ∧ y = 6 :=
sorry

end minimum_value_of_absolute_sum_l1791_179129


namespace range_of_a_not_empty_solution_set_l1791_179167

theorem range_of_a_not_empty_solution_set :
  {a : ℝ | ∃ x : ℝ, (a^2 - 4) * x^2 + (a + 2) * x - 1 ≥ 0} =
  {a : ℝ | a ∈ {a : ℝ | a < -2} ∪ {a : ℝ | a ≥ 6 / 5}} :=
sorry

end range_of_a_not_empty_solution_set_l1791_179167


namespace function_inequality_l1791_179152

noncomputable def f : ℝ → ℝ
| x => if x < 1 then (x + 1)^2 else 4 - Real.sqrt (x - 1)

theorem function_inequality : 
  {x : ℝ | f x ≥ x} = {x : ℝ | x ≤ -2} ∪ {x : ℝ | 0 ≤ x ∧ x ≤ 10} :=
by
  sorry

end function_inequality_l1791_179152


namespace value_of_expression_l1791_179116

-- Definitions based on the conditions
def a : ℕ := 15
def b : ℕ := 3

-- The theorem to prove
theorem value_of_expression : a^2 + 2 * a * b + b^2 = 324 := by
  -- Skipping the proof as per instructions
  sorry

end value_of_expression_l1791_179116


namespace joshua_skittles_l1791_179159

theorem joshua_skittles (eggs : ℝ) (skittles_per_friend : ℝ) (friends : ℝ) (h1 : eggs = 6.0) (h2 : skittles_per_friend = 40.0) (h3 : friends = 5.0) : skittles_per_friend * friends = 200.0 := 
by 
  sorry

end joshua_skittles_l1791_179159


namespace max_value_of_y_l1791_179153

noncomputable def max_y (x y : ℝ) : ℝ :=
  if h : x^2 + y^2 = 20*x + 54*y then y else 0

theorem max_value_of_y (x y : ℝ) (h : x^2 + y^2 = 20*x + 54*y) :
  max_y x y ≤ 27 + Real.sqrt 829 :=
sorry

end max_value_of_y_l1791_179153


namespace intersection_M_N_l1791_179144

def M : Set ℝ := {y | ∃ x, x ∈ Set.Icc (-5) 5 ∧ y = 2 * Real.sin x}
def N : Set ℝ := {x | ∃ y, y = Real.log (x - 1) / Real.log 2}

theorem intersection_M_N : {x | 1 < x ∧ x ≤ 2} = {x | x ∈ M ∩ N} :=
by sorry

end intersection_M_N_l1791_179144


namespace total_students_l1791_179199

theorem total_students (a : ℕ) (h1: (71 * ((3480 - 69 * a) / 2) + 69 * (a - (3480 - 69 * a) / 2)) = 3480) : a = 50 :=
by
  -- Proof to be provided here
  sorry

end total_students_l1791_179199


namespace find_r_minus2_l1791_179142

noncomputable def p : ℤ → ℤ := sorry
def r : ℤ → ℤ := sorry

-- Conditions given in the problem
axiom p_minus1 : p (-1) = 2
axiom p_3 : p (3) = 5
axiom p_minus4 : p (-4) = -3

-- Definition of r(x) when p(x) is divided by (x + 1)(x - 3)(x + 4)
axiom r_def : ∀ x, p x = (x + 1) * (x - 3) * (x + 4) * (sorry : ℤ → ℤ) + r x

-- Our goal to prove
theorem find_r_minus2 : r (-2) = 32 / 7 :=
sorry

end find_r_minus2_l1791_179142


namespace y_coordinate_equidistant_l1791_179104

theorem y_coordinate_equidistant :
  ∃ y : ℝ, (∀ ptC ptD : ℝ × ℝ, ptC = (-3, 0) → ptD = (4, 5) → 
    dist (0, y) ptC = dist (0, y) ptD) ∧ y = 16 / 5 :=
by
  sorry

end y_coordinate_equidistant_l1791_179104


namespace smallest_integer_gcd_6_l1791_179103

theorem smallest_integer_gcd_6 : ∃ n : ℕ, n > 100 ∧ gcd n 18 = 6 ∧ ∀ m : ℕ, (m > 100 ∧ gcd m 18 = 6) → m ≥ n :=
by
  let n := 114
  have h1 : n > 100 := sorry
  have h2 : gcd n 18 = 6 := sorry
  have h3 : ∀ m : ℕ, (m > 100 ∧ gcd m 18 = 6) → m ≥ n := sorry
  exact ⟨n, h1, h2, h3⟩

end smallest_integer_gcd_6_l1791_179103


namespace smallest_among_given_numbers_l1791_179114

theorem smallest_among_given_numbers :
  let a := abs (-3)
  let b := -2
  let c := 0
  let d := Real.pi
  b < a ∧ b < c ∧ b < d := by
  sorry

end smallest_among_given_numbers_l1791_179114


namespace quadratic_residue_property_l1791_179126

theorem quadratic_residue_property (p : ℕ) [hp : Fact (Nat.Prime p)] (a : ℕ)
  (h : ∃ t : ℤ, ∃ k : ℤ, k * k = p * t + a) : (a ^ ((p - 1) / 2)) % p = 1 :=
sorry

end quadratic_residue_property_l1791_179126


namespace total_students_l1791_179196

-- Define the problem statement in Lean 4
theorem total_students (n : ℕ) (h1 : n < 400)
  (h2 : n % 17 = 15) (h3 : n % 19 = 10) : n = 219 :=
sorry

end total_students_l1791_179196


namespace fraction_to_decimal_l1791_179165

theorem fraction_to_decimal : (5 : ℚ) / 16 = 0.3125 := sorry

end fraction_to_decimal_l1791_179165


namespace remainder_n_l1791_179182

-- Definitions for the conditions
/-- m is a positive integer leaving a remainder of 2 when divided by 6 -/
def m (m : ℕ) : Prop := m % 6 = 2

/-- The remainder when m - n is divided by 6 is 5 -/
def mn_remainder (m n : ℕ) : Prop := (m - n) % 6 = 5

-- Theorem statement
theorem remainder_n (m n : ℕ) (h1 : m % 6 = 2) (h2 : (m - n) % 6 = 5) (h3 : m > n) :
  n % 6 = 4 :=
by
  sorry

end remainder_n_l1791_179182


namespace fraction_b_not_whole_l1791_179131

-- Defining the fractions as real numbers
def fraction_a := 60 / 12
def fraction_b := 60 / 8
def fraction_c := 60 / 5
def fraction_d := 60 / 4
def fraction_e := 60 / 3

-- Defining what it means to be a whole number
def is_whole_number (x : ℝ) : Prop := ∃ (n : ℤ), x = n

-- Theorem stating that fraction_b is not a whole number
theorem fraction_b_not_whole : ¬ is_whole_number fraction_b := 
by 
-- proof to be filled in
sorry

end fraction_b_not_whole_l1791_179131


namespace correct_number_of_conclusions_l1791_179150

def y (x : ℝ) := -5 * x + 1

def conclusion1 := y (-1) = 5
def conclusion2 := ∃ x1 x2 x3 : ℝ, y x1 > 0 ∧ y x2 > 0 ∧ y (x3) < 0 ∧ (x1 < 0) ∧ (x2 > 0) ∧ (x3 < x2)
def conclusion3 := ∀ x : ℝ, x > 1 → y x < 0
def conclusion4 := ∀ x1 x2 : ℝ, x1 < x2 → y x1 < y x2

-- We want to prove that exactly 2 of these conclusions are correct
theorem correct_number_of_conclusions : (¬ conclusion1 ∧ conclusion2 ∧ conclusion3 ∧ ¬ conclusion4) :=
by
  sorry

end correct_number_of_conclusions_l1791_179150


namespace jefferson_high_school_ninth_graders_l1791_179123

theorem jefferson_high_school_ninth_graders (total_students science_students arts_students students_taking_both : ℕ):
  total_students = 120 →
  science_students = 85 →
  arts_students = 65 →
  students_taking_both = 150 - 120 →
  science_students - students_taking_both = 55 :=
by
  sorry

end jefferson_high_school_ninth_graders_l1791_179123


namespace fruit_total_l1791_179130

noncomputable def fruit_count_proof : Prop :=
  let oranges := 6
  let apples := oranges - 2
  let bananas := 3 * apples
  let peaches := bananas / 2
  oranges + apples + bananas + peaches = 28

theorem fruit_total : fruit_count_proof :=
by {
  sorry
}

end fruit_total_l1791_179130


namespace greatest_number_of_problems_missed_l1791_179183

theorem greatest_number_of_problems_missed 
    (total_problems : ℕ) (passing_percentage : ℝ) (max_missed : ℕ) :
    total_problems = 40 →
    passing_percentage = 0.85 →
    max_missed = total_problems - ⌈total_problems * passing_percentage⌉ →
    max_missed = 6 :=
by
  intros h1 h2 h3
  sorry

end greatest_number_of_problems_missed_l1791_179183


namespace total_albums_l1791_179113

theorem total_albums (Adele Bridget Katrina Miriam : ℕ) 
  (h₁ : Adele = 30)
  (h₂ : Bridget = Adele - 15)
  (h₃ : Katrina = 6 * Bridget)
  (h₄ : Miriam = 5 * Katrina) : Adele + Bridget + Katrina + Miriam = 585 := 
by
  sorry

end total_albums_l1791_179113


namespace pizza_eaten_after_six_trips_l1791_179100

noncomputable def fraction_eaten : ℚ :=
  let first_trip := 1 / 3
  let second_trip := 1 / (3 ^ 2)
  let third_trip := 1 / (3 ^ 3)
  let fourth_trip := 1 / (3 ^ 4)
  let fifth_trip := 1 / (3 ^ 5)
  let sixth_trip := 1 / (3 ^ 6)
  first_trip + second_trip + third_trip + fourth_trip + fifth_trip + sixth_trip

theorem pizza_eaten_after_six_trips : fraction_eaten = 364 / 729 :=
by sorry

end pizza_eaten_after_six_trips_l1791_179100


namespace jackie_free_time_correct_l1791_179158

noncomputable def jackie_free_time : ℕ :=
  let total_hours_in_a_day := 24
  let hours_working := 8
  let hours_exercising := 3
  let hours_sleeping := 8
  let total_activity_hours := hours_working + hours_exercising + hours_sleeping
  total_hours_in_a_day - total_activity_hours

theorem jackie_free_time_correct : jackie_free_time = 5 := by
  sorry

end jackie_free_time_correct_l1791_179158


namespace monthly_savings_correct_l1791_179195

-- Define the gross salaries before any deductions
def ivan_salary_gross : ℝ := 55000
def vasilisa_salary_gross : ℝ := 45000
def vasilisa_mother_salary_gross : ℝ := 18000
def vasilisa_father_salary_gross : ℝ := 20000
def son_scholarship_state : ℝ := 3000
def son_scholarship_non_state_gross : ℝ := 15000

-- Tax rate definition
def tax_rate : ℝ := 0.13

-- Net income calculations using the tax rate
def net_income (gross_income : ℝ) : ℝ := gross_income * (1 - tax_rate)

def ivan_salary_net : ℝ := net_income ivan_salary_gross
def vasilisa_salary_net : ℝ := net_income vasilisa_salary_gross
def vasilisa_mother_salary_net : ℝ := net_income vasilisa_mother_salary_gross
def vasilisa_father_salary_net : ℝ := net_income vasilisa_father_salary_gross
def son_scholarship_non_state_net : ℝ := net_income son_scholarship_non_state_gross

-- Monthly expenses total
def monthly_expenses : ℝ := 40000 + 20000 + 5000 + 5000 + 2000 + 2000

-- Net incomes for different periods
def total_net_income_before_01_05_2018 : ℝ :=
  ivan_salary_net + vasilisa_salary_net + vasilisa_mother_salary_net + vasilisa_father_salary_net + son_scholarship_state

def total_net_income_01_05_2018_to_31_08_2018 : ℝ :=
  ivan_salary_net + vasilisa_salary_net + 10000 + vasilisa_father_salary_net + son_scholarship_state

def total_net_income_from_01_09_2018 : ℝ :=
  ivan_salary_net + vasilisa_salary_net + 10000 + vasilisa_father_salary_net + (son_scholarship_state + son_scholarship_non_state_net)

-- Savings calculations for different periods
def monthly_savings_before_01_05_2018 : ℝ :=
  total_net_income_before_01_05_2018 - monthly_expenses

def monthly_savings_01_05_2018_to_31_08_2018 : ℝ :=
  total_net_income_01_05_2018_to_31_08_2018 - monthly_expenses

def monthly_savings_from_01_09_2018 : ℝ :=
  total_net_income_from_01_09_2018 - monthly_expenses

-- Theorem to be proved
theorem monthly_savings_correct :
  monthly_savings_before_01_05_2018 = 49060 ∧
  monthly_savings_01_05_2018_to_31_08_2018 = 43400 ∧
  monthly_savings_from_01_09_2018 = 56450 :=
by
  sorry

end monthly_savings_correct_l1791_179195


namespace woman_work_rate_l1791_179191

theorem woman_work_rate (M W : ℝ) (h1 : 10 * M + 15 * W = 1 / 8) (h2 : M = 1 / 100) : W = 1 / 600 :=
by 
  sorry

end woman_work_rate_l1791_179191


namespace james_muffins_baked_l1791_179190

theorem james_muffins_baked (arthur_muffins : ℝ) (factor : ℝ) (h1 : arthur_muffins = 115.0) (h2 : factor = 12.0) :
  (arthur_muffins / factor) = 9.5833 :=
by 
  -- using the conditions given, we would proceed to prove the result:
  -- sorry is used to indicate that the proof is omitted here
  sorry

end james_muffins_baked_l1791_179190


namespace total_profit_amount_l1791_179151

-- Definitions representing the conditions:
def ratio_condition (P_X P_Y : ℝ) : Prop :=
  P_X / P_Y = (1 / 2) / (1 / 3)

def difference_condition (P_X P_Y : ℝ) : Prop :=
  P_X - P_Y = 160

-- The proof problem statement:
theorem total_profit_amount (P_X P_Y : ℝ) (h1 : ratio_condition P_X P_Y) (h2 : difference_condition P_X P_Y) :
  P_X + P_Y = 800 := by
  sorry

end total_profit_amount_l1791_179151


namespace remainder_5_pow_100_mod_18_l1791_179146

theorem remainder_5_pow_100_mod_18 : (5 ^ 100) % 18 = 13 := 
by
  -- We will skip the proof since only the statement is required.
  sorry

end remainder_5_pow_100_mod_18_l1791_179146


namespace given_conditions_imply_f_neg3_gt_f_neg2_l1791_179128

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

theorem given_conditions_imply_f_neg3_gt_f_neg2
  {f : ℝ → ℝ}
  (h_even : is_even_function f)
  (h_comparison : f 2 < f 3) :
  f (-3) > f (-2) :=
by
  sorry

end given_conditions_imply_f_neg3_gt_f_neg2_l1791_179128


namespace solution_set_of_abs_inequality_l1791_179118

theorem solution_set_of_abs_inequality : 
  {x : ℝ | abs (x - 1) - abs (x - 5) < 2} = {x : ℝ | x < 4} := 
by 
  sorry

end solution_set_of_abs_inequality_l1791_179118


namespace find_n_l1791_179198

theorem find_n (a b c : ℕ) (n : ℕ) (h₁ : a^2 + b^2 = c^2) (h₂ : n > 2) 
    (h₃ : (a^n + b^n + c^n)^2 = 2 * (a^(2*n) + b^(2*n) + c^(2*n))) : n = 4 := 
sorry

end find_n_l1791_179198


namespace number_of_children_l1791_179112

def weekly_husband : ℕ := 335
def weekly_wife : ℕ := 225
def weeks_in_six_months : ℕ := 24
def amount_per_child : ℕ := 1680

theorem number_of_children : (weekly_husband + weekly_wife) * weeks_in_six_months / 2 / amount_per_child = 4 := by
  sorry

end number_of_children_l1791_179112
