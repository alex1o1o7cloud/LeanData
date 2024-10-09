import Mathlib

namespace lines_parallel_or_coincident_l2040_204036

/-- Given lines l₁ and l₂ with certain properties,
    prove that they are either parallel or coincident. -/
theorem lines_parallel_or_coincident
  (P Q : ℝ × ℝ)
  (hP : P = (-2, -1))
  (hQ : Q = (3, -6))
  (h_slope1 : ∀ θ, θ = 135 → Real.tan (θ * (Real.pi / 180)) = -1)
  (h_slope2 : (Q.2 - P.2) / (Q.1 - P.1) = -1) : 
  true :=
by sorry

end lines_parallel_or_coincident_l2040_204036


namespace simplify_cube_root_l2040_204035

theorem simplify_cube_root (a : ℝ) (h : 0 ≤ a) : (a * a^(1/2))^(1/3) = a^(1/2) :=
sorry

end simplify_cube_root_l2040_204035


namespace multiplicative_inverse_l2040_204040

def A : ℕ := 123456
def B : ℕ := 171428
def mod_val : ℕ := 1000000
def sum_A_B : ℕ := A + B
def N : ℕ := 863347

theorem multiplicative_inverse : (sum_A_B * N) % mod_val = 1 :=
by
  -- diverting proof with sorry since proof steps aren't the focus
  sorry

end multiplicative_inverse_l2040_204040


namespace original_faculty_members_l2040_204018

theorem original_faculty_members (X : ℝ) (H0 : X > 0) 
  (H1 : 0.75 * X ≤ X)
  (H2 : ((0.75 * X + 35) * 1.10 * 0.80 = 195)) :
  X = 253 :=
by {
  sorry
}

end original_faculty_members_l2040_204018


namespace jaeho_got_most_notebooks_l2040_204016

-- Define the number of notebooks each friend received
def notebooks_jaehyuk : ℕ := 12
def notebooks_kyunghwan : ℕ := 3
def notebooks_jaeho : ℕ := 15

-- Define the statement proving that Jaeho received the most notebooks
theorem jaeho_got_most_notebooks : notebooks_jaeho > notebooks_jaehyuk ∧ notebooks_jaeho > notebooks_kyunghwan :=
by {
  sorry -- this is where the proof would go
}

end jaeho_got_most_notebooks_l2040_204016


namespace pyramid_edges_sum_l2040_204047

noncomputable def sum_of_pyramid_edges (s : ℝ) (h : ℝ) : ℝ :=
  let diagonal := s * Real.sqrt 2
  let half_diagonal := diagonal / 2
  let slant_height := Real.sqrt (half_diagonal^2 + h^2)
  4 * s + 4 * slant_height

theorem pyramid_edges_sum
  (s : ℝ) (h : ℝ)
  (hs : s = 15)
  (hh : h = 15) :
  sum_of_pyramid_edges s h = 135 :=
sorry

end pyramid_edges_sum_l2040_204047


namespace age_of_youngest_child_l2040_204089

theorem age_of_youngest_child (x : ℕ) 
  (h : x + (x + 4) + (x + 8) + (x + 12) + (x + 16) + (x + 20) + (x + 24) = 112) :
  x = 4 :=
sorry

end age_of_youngest_child_l2040_204089


namespace monotonically_decreasing_interval_range_of_f_l2040_204002

noncomputable def f (x : ℝ) : ℝ := (1/2) ^ (abs (x - 1))

theorem monotonically_decreasing_interval :
  ∀ x₁ x₂ : ℝ, 1 < x₁ → x₁ < x₂ → f x₁ > f x₂ := by sorry

theorem range_of_f :
  Set.range f = {y : ℝ | 0 < y ∧ y ≤ 1 } := by sorry

end monotonically_decreasing_interval_range_of_f_l2040_204002


namespace lavinias_son_older_than_daughter_l2040_204037

def katies_daughter_age := 12
def lavinias_daughter_age := katies_daughter_age - 10
def lavinias_son_age := 2 * katies_daughter_age

theorem lavinias_son_older_than_daughter :
  lavinias_son_age - lavinias_daughter_age = 22 :=
by
  sorry

end lavinias_son_older_than_daughter_l2040_204037


namespace balance_rearrangement_vowels_at_end_l2040_204045

theorem balance_rearrangement_vowels_at_end : 
  let vowels := ['A', 'A', 'E'];
  let consonants := ['B', 'L', 'N', 'C'];
  (Nat.factorial 3 / Nat.factorial 2) * Nat.factorial 4 = 72 :=
by
  sorry

end balance_rearrangement_vowels_at_end_l2040_204045


namespace triangle_area_l2040_204067

-- Define the given conditions
def perimeter : ℝ := 60
def inradius : ℝ := 2.5

-- Prove the area of the triangle using the given inradius and perimeter
theorem triangle_area (p : ℝ) (r : ℝ) (h1 : p = 60) (h2 : r = 2.5) :
  (r * (p / 2)) = 75 := 
by
  rw [h1, h2]
  sorry

end triangle_area_l2040_204067


namespace last_integer_in_geometric_sequence_l2040_204046

theorem last_integer_in_geometric_sequence (a : ℕ) (r : ℚ) (h_a : a = 2048000) (h_r : r = 1/2) : 
  ∃ n : ℕ, (a : ℚ) * (r^n : ℚ) = 125 := 
by
  sorry

end last_integer_in_geometric_sequence_l2040_204046


namespace range_of_m_l2040_204068

theorem range_of_m (m : ℝ) : (∀ x : ℝ, |2009 * x + 1| ≥ |m - 1| - 2) → -1 ≤ m ∧ m ≤ 3 :=
by
  intro h
  sorry

end range_of_m_l2040_204068


namespace math_problem_l2040_204071

variable {x y z : ℝ}
variable (hx : x > 0) (hy : y > 0) (hz : z > 0)
variable (h : x^2 + y^2 + z^2 = 1)

theorem math_problem : 
  (x / (1 - x^2)) + (y / (1 - y^2)) + (z / (1 - z^2)) ≥ 3 * Real.sqrt 3 / 2 :=
sorry

end math_problem_l2040_204071


namespace maggie_earnings_l2040_204006

def subscriptions_to_parents := 4
def subscriptions_to_grandfather := 1
def subscriptions_to_next_door_neighbor := 2
def subscriptions_to_another_neighbor := 2 * subscriptions_to_next_door_neighbor
def subscription_rate := 5

theorem maggie_earnings : 
  (subscriptions_to_parents + subscriptions_to_grandfather + subscriptions_to_next_door_neighbor + subscriptions_to_another_neighbor) * subscription_rate = 55 := 
by
  sorry

end maggie_earnings_l2040_204006


namespace average_marks_l2040_204092

theorem average_marks {n : ℕ} (h1 : 5 * 74 + 104 = n * 79) : n = 6 :=
by
  sorry

end average_marks_l2040_204092


namespace possible_denominators_count_l2040_204055

variable (a b c : ℕ)
-- Conditions
def is_digit (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 9
def no_two_zeros (a b c : ℕ) : Prop := ¬(a = 0 ∧ b = 0) ∧ ¬(b = 0 ∧ c = 0) ∧ ¬(a = 0 ∧ c = 0)
def none_is_eight (a b c : ℕ) : Prop := a ≠ 8 ∧ b ≠ 8 ∧ c ≠ 8

-- Theorem
theorem possible_denominators_count : 
  is_digit a ∧ is_digit b ∧ is_digit c ∧ no_two_zeros a b c ∧ none_is_eight a b c →
  ∃ denoms : Finset ℕ, denoms.card = 7 ∧ ∀ d ∈ denoms, 999 % d = 0 :=
by
  sorry

end possible_denominators_count_l2040_204055


namespace sector_COD_area_ratio_l2040_204093

-- Define the given angles
def angle_AOC : ℝ := 30
def angle_DOB : ℝ := 45
def angle_AOB : ℝ := 180

-- Define the full circle angle
def full_circle_angle : ℝ := 360

-- Calculate the angle COD
def angle_COD : ℝ := angle_AOB - angle_AOC - angle_DOB

-- State the ratio of the area of sector COD to the area of the circle
theorem sector_COD_area_ratio :
  angle_COD / full_circle_angle = 7 / 24 := by
  sorry

end sector_COD_area_ratio_l2040_204093


namespace log_sum_equality_l2040_204038

noncomputable def log_base_5 (x : ℝ) := Real.log x / Real.log 5

theorem log_sum_equality :
  2 * log_base_5 10 + log_base_5 0.25 = 2 :=
by
  sorry -- proof goes here

end log_sum_equality_l2040_204038


namespace chickens_cheaper_than_eggs_l2040_204073

-- Define the initial costs of the chickens
def initial_cost_chicken1 : ℝ := 25
def initial_cost_chicken2 : ℝ := 30
def initial_cost_chicken3 : ℝ := 22
def initial_cost_chicken4 : ℝ := 35

-- Define the weekly feed costs for the chickens
def weekly_feed_cost_chicken1 : ℝ := 1.50
def weekly_feed_cost_chicken2 : ℝ := 1.30
def weekly_feed_cost_chicken3 : ℝ := 1.10
def weekly_feed_cost_chicken4 : ℝ := 0.90

-- Define the weekly egg production for the chickens
def weekly_egg_prod_chicken1 : ℝ := 4
def weekly_egg_prod_chicken2 : ℝ := 3
def weekly_egg_prod_chicken3 : ℝ := 5
def weekly_egg_prod_chicken4 : ℝ := 2

-- Define the cost of a dozen eggs at the store
def cost_per_dozen_eggs : ℝ := 2

-- Define total initial costs, total weekly feed cost, and weekly savings
def total_initial_cost : ℝ := initial_cost_chicken1 + initial_cost_chicken2 + initial_cost_chicken3 + initial_cost_chicken4
def total_weekly_feed_cost : ℝ := weekly_feed_cost_chicken1 + weekly_feed_cost_chicken2 + weekly_feed_cost_chicken3 + weekly_feed_cost_chicken4
def weekly_savings : ℝ := cost_per_dozen_eggs

-- Define the condition for the number of weeks (W) when the chickens become cheaper
def breakeven_weeks : ℝ := 40

theorem chickens_cheaper_than_eggs (W : ℕ) :
  total_initial_cost + W * total_weekly_feed_cost = W * weekly_savings :=
sorry

end chickens_cheaper_than_eggs_l2040_204073


namespace triangle_area_correct_l2040_204029

noncomputable def triangle_area_given_conditions (a b c : ℝ) (A : ℝ) : ℝ :=
  if h : a = c + 4 ∧ b = c + 2 ∧ Real.cos A = -1/2 then
  1/2 * b * c * Real.sin A
  else 0

theorem triangle_area_correct :
  ∀ (a b c : ℝ), ∀ A : ℝ, a = c + 4 → b = c + 2 → Real.cos A = -1/2 → 
  triangle_area_given_conditions a b c A = 15 * Real.sqrt 3 / 4 :=
by
  intros a b c A ha hb hc
  simp [triangle_area_given_conditions, ha, hb, hc]
  sorry

end triangle_area_correct_l2040_204029


namespace total_loaves_served_l2040_204062

-- Given conditions
def wheat_bread := 0.5
def white_bread := 0.4

-- Proof that total loaves served is 0.9
theorem total_loaves_served : wheat_bread + white_bread = 0.9 :=
by sorry

end total_loaves_served_l2040_204062


namespace students_calculation_l2040_204042

variable (students_boys students_playing_soccer students_not_playing_soccer girls_not_playing_soccer : ℕ)
variable (percentage_boys_play_soccer : ℚ)

def students_not_playing_sum (students_boys_not_playing : ℕ) : ℕ :=
  students_boys_not_playing + girls_not_playing_soccer

def total_students (students_not_playing_sum students_playing_soccer : ℕ) : ℕ :=
  students_not_playing_sum + students_playing_soccer

theorem students_calculation 
  (H1 : students_boys = 312)
  (H2 : students_playing_soccer = 250)
  (H3 : percentage_boys_play_soccer = 0.86)
  (H4 : girls_not_playing_soccer = 73)
  (H5 : percentage_boys_play_soccer * students_playing_soccer = 215)
  (H6 : students_boys - 215 = 97)
  (H7 : students_not_playing_sum 97 = 170)
  (H8 : total_students 170 250 = 420) : ∃ total, total = 420 :=
by 
  existsi total_students 170 250
  exact H8

end students_calculation_l2040_204042


namespace expression_for_f_l2040_204048

theorem expression_for_f (f : ℤ → ℤ) (h : ∀ x : ℤ, f (x + 1) = x^2 - x - 2) : ∀ x : ℤ, f x = x^2 - 3 * x := 
by
  sorry

end expression_for_f_l2040_204048


namespace parabola_passes_through_point_l2040_204072

theorem parabola_passes_through_point {x y : ℝ} (h_eq : y = (1/2) * x^2 - 2) :
  (x = 2 ∧ y = 0) :=
by
  sorry

end parabola_passes_through_point_l2040_204072


namespace unique_solution_for_equation_l2040_204034

theorem unique_solution_for_equation (a b c d : ℝ) 
  (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) (h₄ : 0 < d)
  (h : ∀ x : ℝ, (a * x + b) ^ 2016 + (x ^ 2 + c * x + d) ^ 1008 = 8 * (x - 2) ^ 2016) :
  a = 2 ^ (1 / 672) ∧ b = -2 * 2 ^ (1 / 672) ∧ c = -4 ∧ d = 4 :=
by
  sorry

end unique_solution_for_equation_l2040_204034


namespace emily_original_salary_l2040_204023

def original_salary_emily (num_employees : ℕ) (original_employee_salary new_employee_salary new_salary_emily : ℕ) : ℕ :=
  new_salary_emily + (new_employee_salary - original_employee_salary) * num_employees

theorem emily_original_salary :
  original_salary_emily 10 20000 35000 850000 = 1000000 :=
by
  sorry

end emily_original_salary_l2040_204023


namespace length_of_bridge_l2040_204001

theorem length_of_bridge (train_length : ℕ) (train_speed_kmph : ℕ) (cross_time_sec : ℕ) (bridge_length: ℕ):
  train_length = 110 →
  train_speed_kmph = 45 →
  cross_time_sec = 30 →
  bridge_length = 265 :=
by
  intros h1 h2 h3
  sorry

end length_of_bridge_l2040_204001


namespace measure_8_cm_measure_5_cm_1_measure_5_cm_2_l2040_204005

theorem measure_8_cm:
  ∃ n : ℕ, n * (11 - 7) = 8 := by
  sorry

theorem measure_5_cm_1:
  ∃ x : ℕ, ∃ y : ℕ, x * ((11 - 7) * 2) - y * 7 = 5 := by
  sorry

theorem measure_5_cm_2:
  3 * 11 - 4 * 7 = 5 := by
  sorry

end measure_8_cm_measure_5_cm_1_measure_5_cm_2_l2040_204005


namespace largest_multiple_of_18_with_8_and_0_digits_l2040_204060

theorem largest_multiple_of_18_with_8_and_0_digits :
  ∃ m : ℕ, (∀ d ∈ (m.digits 10), d = 8 ∨ d = 0) ∧ (m % 18 = 0) ∧ (m = 8888888880) ∧ (m / 18 = 493826048) :=
by sorry

end largest_multiple_of_18_with_8_and_0_digits_l2040_204060


namespace star_difference_l2040_204030

def star (x y : ℤ) : ℤ := x * y + 3 * x - y

theorem star_difference : (star 7 4) - (star 4 7) = 12 := by
  sorry

end star_difference_l2040_204030


namespace non_parallel_lines_implies_unique_solution_l2040_204084

variable (a1 b1 c1 a2 b2 c2 : ℝ)

def system_of_equations (x y : ℝ) := a1 * x + b1 * y = c1 ∧ a2 * x + b2 * y = c2

def lines_not_parallel := a1 * b2 ≠ a2 * b1

theorem non_parallel_lines_implies_unique_solution :
  lines_not_parallel a1 b1 a2 b2 → ∃! (x y : ℝ), system_of_equations a1 b1 c1 a2 b2 c2 x y :=
sorry

end non_parallel_lines_implies_unique_solution_l2040_204084


namespace symmetric_axis_parabola_l2040_204011

theorem symmetric_axis_parabola (h k : ℝ) (x : ℝ) :
  (∀ x, y = (x - h)^2 + k) → h = 2 → (x = 2) :=
by
  sorry

end symmetric_axis_parabola_l2040_204011


namespace solve_natural_a_l2040_204012

theorem solve_natural_a (a : ℕ) : 
  (∃ n : ℕ, a^2 + a + 1589 = n^2) ↔ (a = 43 ∨ a = 28 ∨ a = 316 ∨ a = 1588) :=
sorry

end solve_natural_a_l2040_204012


namespace arithmetic_sequence_middle_term_l2040_204033

theorem arithmetic_sequence_middle_term :
  let a1 := 3^2
  let a3 := 3^4
  let y := (a1 + a3) / 2
  y = 45 :=
by
  let a1 := (3:ℕ)^2
  let a3 := (3:ℕ)^4
  let y := (a1 + a3) / 2
  have : a1 = 9 := by norm_num
  have : a3 = 81 := by norm_num
  have : y = 45 := by norm_num
  exact this

end arithmetic_sequence_middle_term_l2040_204033


namespace simplify_expression_l2040_204020

variable {a b : ℝ}

theorem simplify_expression {a b : ℝ} (h : |2 - a + b| + (ab + 1)^2 = 0) :
  (4 * a - 5 * b - a * b) - (2 * a - 3 * b + 5 * a * b) = 10 := by
  sorry

end simplify_expression_l2040_204020


namespace problem1_l2040_204056

theorem problem1 (x : ℝ) (hx : x > 0) : (x + 1/x = 2) ↔ (x = 1) :=
by
  sorry

end problem1_l2040_204056


namespace pocket_money_calculation_l2040_204095

theorem pocket_money_calculation
  (a b c d e : ℝ)
  (h1 : (a + b + c + d + e) / 5 = 2300)
  (h2 : (a + b) / 2 = 3000)
  (h3 : (b + c) / 2 = 2100)
  (h4 : (c + d) / 2 = 2750)
  (h5 : a = b + 800) :
  d = 3900 :=
by
  sorry

end pocket_money_calculation_l2040_204095


namespace bin101_to_decimal_l2040_204058

-- Define the binary representation of 101 (base 2)
def bin101 : ℕ := 1 * 2^2 + 0 * 2^1 + 1 * 2^0

-- State the theorem that asserts the decimal value of 101 (base 2) is 5
theorem bin101_to_decimal : bin101 = 5 := by
  sorry

end bin101_to_decimal_l2040_204058


namespace find_last_number_l2040_204076

-- Definitions for the conditions
def avg_first_three (A B C : ℕ) : ℕ := (A + B + C) / 3
def avg_last_three (B C D : ℕ) : ℕ := (B + C + D) / 3
def sum_first_last (A D : ℕ) : ℕ := A + D

-- Proof problem statement
theorem find_last_number (A B C D : ℕ) 
  (h1 : avg_first_three A B C = 6)
  (h2 : avg_last_three B C D = 5)
  (h3 : sum_first_last A D = 11) : D = 4 :=
sorry

end find_last_number_l2040_204076


namespace quadratic_distinct_roots_example_l2040_204019

theorem quadratic_distinct_roots_example {b c : ℝ} (hb : b = 1) (hc : c = 0) :
    (b^2 - 4 * c) > 0 := by
  sorry

end quadratic_distinct_roots_example_l2040_204019


namespace find_ratio_of_sums_l2040_204078

variables {S : ℕ → ℝ} {a : ℕ → ℝ} {d : ℝ}

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

def sum_first_n_terms (S : ℕ → ℝ) (a : ℕ → ℝ) :=
  ∀ n, S n = n * (a 1 + a n) / 2

def ratio_condition (a : ℕ → ℝ) :=
  a 6 / a 5 = 9 / 11

theorem find_ratio_of_sums (seq : ∃ d, arithmetic_sequence a d)
    (sum_prop : sum_first_n_terms S a)
    (ratio_prop : ratio_condition a) :
  S 11 / S 9 = 1 :=
sorry

end find_ratio_of_sums_l2040_204078


namespace number_div_mult_l2040_204051

theorem number_div_mult (n : ℕ) (h : n = 4) : (n / 6) * 12 = 8 :=
by
  sorry

end number_div_mult_l2040_204051


namespace shaded_area_proof_l2040_204024

-- Given Definitions
def rectangle_area (length : ℕ) (width : ℕ) : ℕ := length * width
def triangle_area (base : ℕ) (height : ℕ) : ℕ := (base * height) / 2

-- Conditions
def grid_area : ℕ :=
  rectangle_area 2 3 + rectangle_area 3 4 + rectangle_area 4 5

def unshaded_triangle_area : ℕ := triangle_area 12 4

-- Question
def shaded_area : ℕ := grid_area - unshaded_triangle_area

-- Proof statement
theorem shaded_area_proof : shaded_area = 14 := by
  sorry

end shaded_area_proof_l2040_204024


namespace probability_same_gender_l2040_204070

theorem probability_same_gender :
  let males := 3
  let females := 2
  let total := males + females
  let total_ways := Nat.choose total 2
  let male_ways := Nat.choose males 2
  let female_ways := Nat.choose females 2
  let same_gender_ways := male_ways + female_ways
  let probability := (same_gender_ways : ℚ) / total_ways
  probability = 2 / 5 :=
by
  sorry

end probability_same_gender_l2040_204070


namespace betty_oranges_l2040_204059

theorem betty_oranges (boxes: ℕ) (oranges_per_box: ℕ) (h1: boxes = 3) (h2: oranges_per_box = 8) : boxes * oranges_per_box = 24 :=
by
  -- proof omitted
  sorry

end betty_oranges_l2040_204059


namespace max_sum_x_y_l2040_204065

theorem max_sum_x_y (x y : ℝ) (h1 : x^2 + y^2 = 7) (h2 : x^3 + y^3 = 10) : x + y ≤ 4 :=
sorry

end max_sum_x_y_l2040_204065


namespace prove_total_number_of_apples_l2040_204061

def avg_price (light_price heavy_price : ℝ) (light_proportion heavy_proportion : ℝ) : ℝ :=
  light_proportion * light_price + heavy_proportion * heavy_price

def weighted_avg_price (prices proportions : List ℝ) : ℝ :=
  (List.map (λ ⟨p, prop⟩ => p * prop) (List.zip prices proportions)).sum

noncomputable def total_num_apples (total_earnings weighted_price : ℝ) : ℝ :=
  total_earnings / weighted_price

theorem prove_total_number_of_apples : 
  let light_proportion := 0.6
  let heavy_proportion := 0.4
  let prices := [avg_price 0.4 0.6 light_proportion heavy_proportion, 
                 avg_price 0.1 0.15 light_proportion heavy_proportion,
                 avg_price 0.25 0.35 light_proportion heavy_proportion,
                 avg_price 0.15 0.25 light_proportion heavy_proportion,
                 avg_price 0.2 0.3 light_proportion heavy_proportion,
                 avg_price 0.05 0.1 light_proportion heavy_proportion]
  let proportions := [0.4, 0.2, 0.15, 0.1, 0.1, 0.05]
  let weighted_avg := weighted_avg_price prices proportions
  total_num_apples 120 weighted_avg = 392 :=
by
  sorry

end prove_total_number_of_apples_l2040_204061


namespace complement_set_solution_l2040_204096

open Set Real

theorem complement_set_solution :
  let M := {x : ℝ | (1 + x) / (1 - x) > 0}
  compl M = {x : ℝ | x ≤ -1 ∨ x ≥ 1} :=
by
  sorry

end complement_set_solution_l2040_204096


namespace costume_total_cost_l2040_204010

variable (friends : ℕ) (cost_per_costume : ℕ) 

theorem costume_total_cost (h1 : friends = 8) (h2 : cost_per_costume = 5) : friends * cost_per_costume = 40 :=
by {
  sorry -- We omit the proof, as instructed.
}

end costume_total_cost_l2040_204010


namespace officeEmployees_l2040_204032

noncomputable def totalEmployees 
  (averageSalaryAll : ℝ) 
  (averageSalaryOfficers : ℝ) 
  (averageSalaryManagers : ℝ) 
  (averageSalaryWorkers : ℝ) 
  (numOfficers : ℕ) 
  (numManagers : ℕ) 
  (numWorkers : ℕ) : ℕ := 
  if (numOfficers * averageSalaryOfficers + numManagers * averageSalaryManagers + numWorkers * averageSalaryWorkers) 
      = (numOfficers + numManagers + numWorkers) * averageSalaryAll 
  then numOfficers + numManagers + numWorkers 
  else 0

theorem officeEmployees
  (averageSalaryAll : ℝ)
  (averageSalaryOfficers : ℝ)
  (averageSalaryManagers : ℝ)
  (averageSalaryWorkers : ℝ)
  (numOfficers : ℕ)
  (numManagers : ℕ)
  (numWorkers : ℕ) :
  averageSalaryAll = 720 →
  averageSalaryOfficers = 1320 →
  averageSalaryManagers = 840 →
  averageSalaryWorkers = 600 →
  numOfficers = 10 →
  numManagers = 20 →
  (numOfficers * averageSalaryOfficers + numManagers * averageSalaryManagers + numWorkers * averageSalaryWorkers) 
    = (numOfficers + numManagers + numWorkers) * averageSalaryAll →
  totalEmployees averageSalaryAll averageSalaryOfficers averageSalaryManagers averageSalaryWorkers numOfficers numManagers numWorkers = 100 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  rw [h2, h3, h4, h5, h6] at h7
  rw [h1]
  simp [totalEmployees, h7]
  sorry

end officeEmployees_l2040_204032


namespace find_c_of_binomial_square_l2040_204014

theorem find_c_of_binomial_square (c : ℝ) (h : ∃ d : ℝ, (9*x^2 - 24*x + c = (3*x + d)^2)) : c = 16 := sorry

end find_c_of_binomial_square_l2040_204014


namespace candy_bar_cost_correct_l2040_204015

-- Definitions based on conditions
def candy_bar_cost := 3
def chocolate_cost := candy_bar_cost + 5
def total_cost := chocolate_cost + candy_bar_cost

-- Assertion to be proved
theorem candy_bar_cost_correct :
  total_cost = 11 → candy_bar_cost = 3 :=
by
  intro h
  simp [total_cost, chocolate_cost, candy_bar_cost] at h
  sorry

end candy_bar_cost_correct_l2040_204015


namespace exists_positive_int_n_l2040_204074

theorem exists_positive_int_n (p a k : ℕ) 
  (hp : Nat.Prime p) (ha : 0 < a) (hk1 : p^a < k) (hk2 : k < 2 * p^a) :
  ∃ n : ℕ, n < p^(2 * a) ∧ (Nat.choose n k ≡ n [MOD p^a]) ∧ (n ≡ k [MOD p^a]) :=
sorry

end exists_positive_int_n_l2040_204074


namespace multiples_of_6_or_8_under_201_not_both_l2040_204039

theorem multiples_of_6_or_8_under_201_not_both : 
  ∃ (n : ℕ), n = 42 ∧ 
    (∀ x : ℕ, x < 201 → ((x % 6 = 0 ∨ x % 8 = 0) ∧ x % 24 ≠ 0) → x ∈ Finset.range 201) :=
by
  sorry

end multiples_of_6_or_8_under_201_not_both_l2040_204039


namespace simplify_fraction_l2040_204000

theorem simplify_fraction :
  (1 / (1 / (Real.sqrt 2 + 1) + 1 / (Real.sqrt 5 - 2))) =
  ((Real.sqrt 2 + Real.sqrt 5 - 1) / (6 + 2 * Real.sqrt 10)) :=
by
  sorry

end simplify_fraction_l2040_204000


namespace percentage_saving_l2040_204007

theorem percentage_saving 
  (p_coat p_pants : ℝ)
  (d_coat d_pants : ℝ)
  (h_coat : p_coat = 100)
  (h_pants : p_pants = 50)
  (h_d_coat : d_coat = 0.30)
  (h_d_pants : d_pants = 0.40) :
  (p_coat * d_coat + p_pants * d_pants) / (p_coat + p_pants) = 0.333 :=
by
  sorry

end percentage_saving_l2040_204007


namespace sin_45_degrees_l2040_204027

noncomputable def Q := (Real.sqrt 2 / 2, Real.sqrt 2 / 2)

theorem sin_45_degrees : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := 
by sorry

end sin_45_degrees_l2040_204027


namespace sufficient_not_necessary_condition_l2040_204063

-- Definitions of propositions p and q
def p (x : ℝ) : Prop := abs (x + 1) ≤ 4
def q (x : ℝ) : Prop := x^2 < 5 * x - 6

-- Definitions of negations of p and q
def not_p (x : ℝ) : Prop := x < -5 ∨ x > 3
def not_q (x : ℝ) : Prop := x ≤ 2 ∨ x ≥ 3

-- The theorem to prove
theorem sufficient_not_necessary_condition (x : ℝ) :
  (¬ p x → ¬ q x) ∧ (¬ q x → ¬ p x → False) := 
by
  sorry

end sufficient_not_necessary_condition_l2040_204063


namespace Moscow_Olympiad_1958_problem_l2040_204097

theorem Moscow_Olympiad_1958_problem :
  ∀ n : ℤ, 1155 ^ 1958 + 34 ^ 1958 ≠ n ^ 2 := 
by 
  sorry

end Moscow_Olympiad_1958_problem_l2040_204097


namespace zero_intersections_l2040_204043

noncomputable def Line : Type := sorry  -- Define Line as a type
noncomputable def is_skew (a b : Line) : Prop := sorry  -- Predicate for skew lines
noncomputable def is_common_perpendicular (EF a b : Line) : Prop := sorry  -- Predicate for common perpendicular
noncomputable def is_parallel (l EF : Line) : Prop := sorry  -- Predicate for parallel lines
noncomputable def count_intersections (l a b : Line) : ℕ := sorry  -- Function to count intersections

theorem zero_intersections (EF a b l : Line) 
  (h_skew : is_skew a b) 
  (h_common_perpendicular : is_common_perpendicular EF a b)
  (h_parallel : is_parallel l EF) : 
  count_intersections l a b = 0 := 
sorry

end zero_intersections_l2040_204043


namespace middle_circle_radius_l2040_204053

theorem middle_circle_radius 
  (r1 r3 : ℝ) 
  (geometric_sequence: ∃ r2 : ℝ, r2 ^ 2 = r1 * r3) 
  (r1_val : r1 = 5) 
  (r3_val : r3 = 20) 
  : ∃ r2 : ℝ, r2 = 10 := 
by
  sorry

end middle_circle_radius_l2040_204053


namespace johann_ate_ten_oranges_l2040_204025

variable (x : ℕ)
variable (y : ℕ)

def johann_initial_oranges := 60

def johann_remaining_after_eating := johann_initial_oranges - x

def johann_remaining_after_theft := (johann_remaining_after_eating / 2)

def johann_remaining_after_return := johann_remaining_after_theft + 5

theorem johann_ate_ten_oranges (h : johann_remaining_after_return = 30) : x = 10 :=
by
  sorry

end johann_ate_ten_oranges_l2040_204025


namespace gcd_of_45_and_75_l2040_204044

def gcd_problem : Prop :=
  gcd 45 75 = 15

theorem gcd_of_45_and_75 : gcd_problem :=
by {
  sorry
}

end gcd_of_45_and_75_l2040_204044


namespace cubes_difference_l2040_204069

theorem cubes_difference (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 25) : a^3 - b^3 = 99 := 
sorry

end cubes_difference_l2040_204069


namespace handshake_problem_l2040_204085

noncomputable def total_handshakes (num_companies : ℕ) (repr_per_company : ℕ) : ℕ :=
    let total_people := num_companies * repr_per_company
    let possible_handshakes_per_person := total_people - repr_per_company
    (total_people * possible_handshakes_per_person) / 2

theorem handshake_problem : total_handshakes 4 4 = 96 :=
by
  sorry

end handshake_problem_l2040_204085


namespace m_n_value_l2040_204052

theorem m_n_value (m n : ℝ)
  (h1 : m * (-1/2)^2 + n * (-1/2) - 1/m < 0)
  (h2 : m * 2^2 + n * 2 - 1/m < 0)
  (h3 : m < 0)
  (h4 : (-1/2 + 2 = -n/m))
  (h5 : (-1/2) * 2 = -1/m^2) :
  m - n = -5/2 :=
sorry

end m_n_value_l2040_204052


namespace fraction_equality_l2040_204041

variable (a_n b_n : ℕ → ℝ)
variable (S_n T_n : ℕ → ℝ)

-- Conditions
axiom S_T_ratio (n : ℕ) : T_n n ≠ 0 → S_n n / T_n n = (2 * n + 1) / (4 * n - 2)
axiom Sn_def (n : ℕ) : S_n n = n / 2 * (2 * a_n 0 + (n - 1) * (a_n 1 - a_n 0))
axiom Tn_def (n : ℕ) : T_n n = n / 2 * (2 * b_n 0 + (n - 1) * (b_n 1 - b_n 0))
axiom an_def (n : ℕ) : a_n n = a_n 0 + n * (a_n 1 - a_n 0)
axiom bn_def (n : ℕ) : b_n n = b_n 0 + n * (b_n 1 - b_n 0)

-- Proof statement
theorem fraction_equality :
  (b_n 3 + b_n 18) ≠ 0 → (b_n 6 + b_n 15) ≠ 0 →
  (a_n 10 / (b_n 3 + b_n 18) + a_n 11 / (b_n 6 + b_n 15)) = (41 / 78) :=
by
  sorry

end fraction_equality_l2040_204041


namespace arithmetic_sequence_sum_l2040_204028

theorem arithmetic_sequence_sum (a b : ℤ) (h1 : 10 - 3 = 7)
  (h2 : a = 10 + 7) (h3 : b = 24 + 7) : a + b = 48 :=
by
  sorry

end arithmetic_sequence_sum_l2040_204028


namespace total_red_marbles_l2040_204066

theorem total_red_marbles (jessica_marbles sandy_marbles alex_marbles : ℕ) (dozen : ℕ)
  (h_jessica : jessica_marbles = 3 * dozen)
  (h_sandy : sandy_marbles = 4 * jessica_marbles)
  (h_alex : alex_marbles = jessica_marbles + 2 * dozen)
  (h_dozen : dozen = 12) :
  jessica_marbles + sandy_marbles + alex_marbles = 240 :=
by
  sorry

end total_red_marbles_l2040_204066


namespace n1_prime_n2_not_prime_l2040_204088

def n1 := 1163
def n2 := 16424
def N := 19101112
def N_eq : N = n1 * n2 := by decide

theorem n1_prime : Prime n1 := 
sorry

theorem n2_not_prime : ¬ Prime n2 :=
sorry

end n1_prime_n2_not_prime_l2040_204088


namespace mini_bottles_needed_to_fill_jumbo_l2040_204017

def mini_bottle_capacity : ℕ := 45
def jumbo_bottle_capacity : ℕ := 600

-- The problem statement expressed as a Lean theorem.
theorem mini_bottles_needed_to_fill_jumbo :
  (jumbo_bottle_capacity + mini_bottle_capacity - 1) / mini_bottle_capacity = 14 :=
by
  sorry

end mini_bottles_needed_to_fill_jumbo_l2040_204017


namespace num_quarters_l2040_204091

theorem num_quarters (n q : ℕ) (avg_initial avg_new : ℕ) 
  (h1 : avg_initial = 10) 
  (h2 : avg_new = 12) 
  (h3 : avg_initial * n + 10 = avg_new * (n + 1)) :
  q = 1 :=
by {
  sorry
}

end num_quarters_l2040_204091


namespace find_m_l2040_204064

-- Definitions for the sets A and B
def A (m : ℝ) : Set ℝ := {3, 4, 4 * m - 4}
def B (m : ℝ) : Set ℝ := {3, m^2}

-- Problem statement
theorem find_m {m : ℝ} (h : B m ⊆ A m) : m = -2 :=
sorry

end find_m_l2040_204064


namespace michael_needs_flour_l2040_204098

-- Define the given conditions
def total_flour : ℕ := 8
def measuring_cup : ℚ := 1/4
def scoops_to_remove : ℕ := 8

-- Prove the amount of flour Michael needs is 6 cups
theorem michael_needs_flour : 
  (total_flour - (scoops_to_remove * measuring_cup)) = 6 := 
by
  sorry

end michael_needs_flour_l2040_204098


namespace correct_calculation_l2040_204003

variable (a : ℝ)

theorem correct_calculation : (2 * a ^ 3) ^ 3 = 8 * a ^ 9 :=
by sorry

end correct_calculation_l2040_204003


namespace speed_of_stream_l2040_204099

theorem speed_of_stream
  (V S : ℝ)
  (h1 : 27 = 9 * (V - S))
  (h2 : 81 = 9 * (V + S)) :
  S = 3 :=
by
  sorry

end speed_of_stream_l2040_204099


namespace janet_time_per_post_l2040_204021

/-- Janet gets paid $0.25 per post she checks. She earns $90 per hour. 
    Prove that it takes her 10 seconds to check a post. -/
theorem janet_time_per_post
  (payment_per_post : ℕ → ℝ)
  (hourly_pay : ℝ)
  (posts_checked_hourly : ℕ)
  (secs_per_post : ℝ) :
  payment_per_post 1 = 0.25 →
  hourly_pay = 90 →
  hourly_pay = payment_per_post (posts_checked_hourly) →
  secs_per_post = 10 :=
sorry

end janet_time_per_post_l2040_204021


namespace sally_seashells_l2040_204013

variable (M : ℝ)

theorem sally_seashells : 
  (1.20 * (M + M / 2) = 54) → M = 30 := 
by
  sorry

end sally_seashells_l2040_204013


namespace find_divisor_l2040_204026

noncomputable def divisor_of_nearest_divisible (a b : ℕ) (d : ℕ) : ℕ :=
  if h : b % d = 0 ∧ (b - a < d) then d else 0

theorem find_divisor (a b : ℕ) (d : ℕ) (h1 : b = 462) (h2 : a = 457)
  (h3 : b % d = 0) (h4 : b - a < d) :
  d = 5 :=
sorry

end find_divisor_l2040_204026


namespace initial_puppies_correct_l2040_204087

def initial_puppies (total_puppies_after: ℝ) (bought_puppies: ℝ) : ℝ :=
  total_puppies_after - bought_puppies

theorem initial_puppies_correct : initial_puppies (4.2 * 5.0) 3.0 = 18.0 := by
  sorry

end initial_puppies_correct_l2040_204087


namespace max_pieces_l2040_204009

theorem max_pieces (plywood_width plywood_height piece_width piece_height : ℕ)
  (h_plywood : plywood_width = 22) (h_plywood_height : plywood_height = 15)
  (h_piece : piece_width = 3) (h_piece_height : piece_height = 5) :
  (plywood_width * plywood_height) / (piece_width * piece_height) = 22 := by
  sorry

end max_pieces_l2040_204009


namespace remainder_division_l2040_204090

def polynomial (x : ℝ) : ℝ := x^4 - 4 * x^2 + 7 * x - 8

theorem remainder_division : polynomial 3 = 58 :=
by
  sorry

end remainder_division_l2040_204090


namespace zaim_larger_part_l2040_204086

theorem zaim_larger_part (x y : ℕ) (h_sum : x + y = 20) (h_prod : x * y = 96) : max x y = 12 :=
by
  -- The proof goes here
  sorry

end zaim_larger_part_l2040_204086


namespace last_digit_of_a2009_div_a2006_is_6_l2040_204079
open Nat

def ratio_difference_sequence (a : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n : ℕ, n > 0 → a (n + 2) * a n = (a (n + 1)) ^ 2 + d * a (n + 1)

theorem last_digit_of_a2009_div_a2006_is_6
  (a : ℕ → ℕ)
  (h1 : a 1 = 1)
  (h2 : a 2 = 1)
  (h3 : a 3 = 2)
  (d : ℕ)
  (h4 : ratio_difference_sequence a d) :
  (a 2009 / a 2006) % 10 = 6 :=
by
  sorry

end last_digit_of_a2009_div_a2006_is_6_l2040_204079


namespace max_sum_after_swap_l2040_204054

section
variables (a1 a2 a3 b1 b2 b3 c1 c2 c3 : ℕ)
  (h1 : 100 * a1 + 10 * b1 + c1 + 100 * a2 + 10 * b2 + c2 + 100 * a3 + 10 * b3 + c3 = 2019)
  (h2 : 1 ≤ a1 ∧ a1 ≤ 9 ∧ 0 ≤ b1 ∧ b1 ≤ 9 ∧ 0 ≤ c1 ∧ c1 ≤ 9)
  (h3 : 1 ≤ a2 ∧ a2 ≤ 9 ∧ 0 ≤ b2 ∧ b2 ≤ 9 ∧ 0 ≤ c2 ∧ c2 ≤ 9)
  (h4 : 1 ≤ a3 ∧ a3 ≤ 9 ∧ 0 ≤ b3 ∧ b3 ≤ 9 ∧ 0 ≤ c3 ∧ c3 ≤ 9)

theorem max_sum_after_swap : 100 * c1 + 10 * b1 + a1 + 100 * c2 + 10 * b2 + a2 + 100 * c3 + 10 * b3 + a3 ≤ 2118 := 
  sorry

end

end max_sum_after_swap_l2040_204054


namespace cos_alpha_beta_half_l2040_204082

open Real

theorem cos_alpha_beta_half (α β : ℝ)
  (h1 : cos (α - β / 2) = -1 / 3)
  (h2 : sin (α / 2 - β) = 1 / 4)
  (h3 : 3 * π / 2 < α ∧ α < 2 * π)
  (h4 : π / 2 < β ∧ β < π) :
  cos ((α + β) / 2) = -(2 * sqrt 2 + sqrt 15) / 12 :=
by
  sorry

end cos_alpha_beta_half_l2040_204082


namespace basketball_success_rate_l2040_204077

theorem basketball_success_rate (p : ℝ) (h : 1 - p^2 = 16 / 25) : p = 3 / 5 :=
sorry

end basketball_success_rate_l2040_204077


namespace min_ab_value_l2040_204080

theorem min_ab_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : (1 / a) + (4 / b) = 1) : ab ≥ 16 :=
by
  sorry

end min_ab_value_l2040_204080


namespace percent_of_total_is_correct_l2040_204031

theorem percent_of_total_is_correct :
  (6.620000000000001 / 100 * 1000 = 66.2) :=
by
  sorry

end percent_of_total_is_correct_l2040_204031


namespace largest_whole_number_l2040_204083

theorem largest_whole_number (x : ℕ) : 8 * x < 120 → x ≤ 14 :=
by
  intro h
  -- prove the main statement here
  sorry

end largest_whole_number_l2040_204083


namespace area_difference_l2040_204022

-- Define the original and new rectangle dimensions
def original_rect_area (length width : ℕ) : ℕ := length * width
def new_rect_area (length width : ℕ) : ℕ := (length - 2) * (width + 2)

-- Define the problem statement
theorem area_difference (a : ℕ) : new_rect_area a 5 - original_rect_area a 5 = 2 * a - 14 :=
by
  -- Insert proof here
  sorry

end area_difference_l2040_204022


namespace lioness_hyena_age_ratio_l2040_204008

variables {k H : ℕ}

-- Conditions
def lioness_age (lioness_age hyena_age : ℕ) : Prop := ∃ k, lioness_age = k * hyena_age
def lioness_is_12 (lioness_age : ℕ) : Prop := lioness_age = 12
def baby_age (mother_age baby_age : ℕ) : Prop := baby_age = mother_age / 2
def baby_ages_sum_in_5_years (baby_l_age baby_h_age sum : ℕ) : Prop := 
  (baby_l_age + 5) + (baby_h_age + 5) = sum

-- The statement to be proved
theorem lioness_hyena_age_ratio (H : ℕ)
  (h1 : lioness_age 12 H) 
  (h2 : baby_age 12 6) 
  (h3 : baby_age H (H / 2)) 
  (h4 : baby_ages_sum_in_5_years 6 (H / 2) 19) : 12 / H = 2 := 
sorry

end lioness_hyena_age_ratio_l2040_204008


namespace triangle_side_b_eq_l2040_204057

   variable (a b c : Real) (A B C : Real)
   variable (cos_A sin_A : Real)
   variable (area : Real)
   variable (π : Real := Real.pi)

   theorem triangle_side_b_eq :
     cos_A = 1 / 3 →
     B = π / 6 →
     a = 4 * Real.sqrt 2 →
     sin_A = 2 * Real.sqrt 2 / 3 →
     b = (a * sin_B / sin_A) →
     b = 3 := sorry
   
end triangle_side_b_eq_l2040_204057


namespace ratio_of_ages_l2040_204004

theorem ratio_of_ages (S : ℕ) (M : ℕ) (h1 : S = 18) (h2 : M = S + 20) :
  (M + 2) / (S + 2) = 2 :=
by
  sorry

end ratio_of_ages_l2040_204004


namespace solution_set_abs_inequality_l2040_204050

theorem solution_set_abs_inequality :
  {x : ℝ | |x + 1| > 1} = {x : ℝ | x < -2} ∪ {x : ℝ | x > 0} :=
sorry

end solution_set_abs_inequality_l2040_204050


namespace months_decreasing_l2040_204075

noncomputable def stock_decrease (m : ℕ) : Prop :=
  2 * m + 2 * 8 = 18

theorem months_decreasing (m : ℕ) (h : stock_decrease m) : m = 1 :=
by
  exact sorry

end months_decreasing_l2040_204075


namespace find_alpha_l2040_204094

noncomputable def isochronous_growth (k α x₁ x₂ y₁ y₂ : ℝ) : Prop :=
  y₁ = k * x₁^α ∧
  y₂ = k * x₂^α ∧
  x₂ = 16 * x₁ ∧
  y₂ = 8 * y₁

theorem find_alpha (k x₁ x₂ y₁ y₂ : ℝ) (h : isochronous_growth k (3/4) x₁ x₂ y₁ y₂) : 3/4 = 3/4 :=
by 
  sorry

end find_alpha_l2040_204094


namespace algebra_simplification_l2040_204049

theorem algebra_simplification (a b : ℤ) (h : ∀ x : ℤ, x^2 - 6 * x + b = (x - a)^2 - 1) : b - a = 5 := by
  sorry

end algebra_simplification_l2040_204049


namespace circumcircle_diameter_of_triangle_l2040_204081

theorem circumcircle_diameter_of_triangle (a b c : ℝ) (A B C : ℝ) 
  (h_a : a = 1) 
  (h_B : B = π/4) 
  (h_area : (1/2) * a * c * Real.sin B = 2) : 
  (2 * b = 5 * Real.sqrt 2) := 
sorry

end circumcircle_diameter_of_triangle_l2040_204081
