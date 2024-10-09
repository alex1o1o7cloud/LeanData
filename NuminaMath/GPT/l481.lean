import Mathlib

namespace tan_difference_l481_48123

theorem tan_difference (x y : ℝ) (hx : Real.tan x = 3) (hy : Real.tan y = 2) :
  Real.tan (x - y) = 1 / 7 := 
  sorry

end tan_difference_l481_48123


namespace triangle_lengths_ce_l481_48153

theorem triangle_lengths_ce (AE BE CE : ℝ) (angle_AEB angle_BEC angle_CED : ℝ) (h1 : angle_AEB = 30)
  (h2 : angle_BEC = 45) (h3 : angle_CED = 45) (h4 : AE = 30) (h5 : BE = AE / 2) (h6 : CE = BE) : CE = 15 :=
by sorry

end triangle_lengths_ce_l481_48153


namespace lines_intersect_at_point_l481_48179

theorem lines_intersect_at_point :
  ∃ (x y : ℝ), (3 * x + 4 * y + 7 = 0) ∧ (x - 2 * y - 1 = 0) ∧ (x = -1) ∧ (y = -1) :=
by
  sorry

end lines_intersect_at_point_l481_48179


namespace function_has_one_zero_l481_48118

-- Define the function f
def f (x m : ℝ) : ℝ := (m - 1) * x^2 + 2 * (m + 1) * x - 1

-- State the theorem
theorem function_has_one_zero (m : ℝ) :
  (∃! x : ℝ, f x m = 0) ↔ m = 0 ∨ m = -3 := 
sorry

end function_has_one_zero_l481_48118


namespace alcohol_mixture_l481_48180

theorem alcohol_mixture:
  ∃ (x y z: ℝ), 
    0.10 * x + 0.30 * y + 0.50 * z = 157.5 ∧
    x + y + z = 450 ∧
    x = y ∧
    x = 112.5 ∧
    y = 112.5 ∧
    z = 225 :=
sorry

end alcohol_mixture_l481_48180


namespace regular_admission_ticket_price_l481_48154

theorem regular_admission_ticket_price
  (n : ℕ) (t : ℕ) (p : ℕ)
  (n_r n_s r : ℕ)
  (H1 : n_r = 3 * n_s)
  (H2 : n_s + n_r = n)
  (H3 : n_r * r + n_s * p = t)
  (H4 : n = 3240)
  (H5 : t = 22680)
  (H6 : p = 4) : 
  r = 8 :=
by sorry

end regular_admission_ticket_price_l481_48154


namespace simon_sand_dollars_l481_48116

theorem simon_sand_dollars (S G P : ℕ) (h1 : G = 3 * S) (h2 : P = 5 * G) (h3 : S + G + P = 190) : S = 10 := by
  sorry

end simon_sand_dollars_l481_48116


namespace smallest_constant_N_l481_48168

theorem smallest_constant_N (a b c : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) (h₄ : a + b > c) (h₅ : b + c > a) (h₆ : c + a > b) :
  (a^2 + b^2 + c^2) / (a * b + b * c + c * a) > 1 :=
by
  sorry

end smallest_constant_N_l481_48168


namespace cube_diagonal_length_l481_48199

theorem cube_diagonal_length (s : ℝ) 
    (h₁ : 6 * s^2 = 54) 
    (h₂ : 12 * s = 36) :
    ∃ d : ℝ, d = 3 * Real.sqrt 3 ∧ d = Real.sqrt (3 * s^2) :=
by
  sorry

end cube_diagonal_length_l481_48199


namespace inequality_solution_set_l481_48155

theorem inequality_solution_set (x : ℝ) : 
  (x + 2) / (x - 1) ≤ 0 ↔ -2 ≤ x ∧ x < 1 := 
sorry

end inequality_solution_set_l481_48155


namespace man_monthly_salary_l481_48135

theorem man_monthly_salary (S E : ℝ) (h1 : 0.20 * S = S - 1.20 * E) (h2 : E = 0.80 * S) :
  S = 6000 :=
by
  sorry

end man_monthly_salary_l481_48135


namespace sample_size_calculation_l481_48162

/--
A factory produces three different models of products: A, B, and C. The ratio of their quantities is 2:3:5.
Using stratified sampling, a sample of size n is drawn, and it contains 16 units of model A.
We need to prove that the sample size n is 80.
-/
theorem sample_size_calculation
  (k : ℕ)
  (hk : 2 * k = 16)
  (n : ℕ)
  (hn : n = (2 + 3 + 5) * k) :
  n = 80 :=
by
  sorry

end sample_size_calculation_l481_48162


namespace sum_of_sequences_l481_48100

-- Define the sequences and their type
def seq1 : List ℕ := [2, 12, 22, 32, 42]
def seq2 : List ℕ := [10, 20, 30, 40, 50]

-- The property we wish to prove
theorem sum_of_sequences : seq1.sum + seq2.sum = 260 :=
by
  sorry

end sum_of_sequences_l481_48100


namespace response_rate_increase_approx_l481_48129

theorem response_rate_increase_approx :
  let original_customers := 80
  let original_respondents := 7
  let redesigned_customers := 63
  let redesigned_respondents := 9
  let original_response_rate := (original_respondents : ℝ) / original_customers * 100
  let redesigned_response_rate := (redesigned_respondents : ℝ) / redesigned_customers * 100
  let percentage_increase := (redesigned_response_rate - original_response_rate) / original_response_rate * 100
  abs (percentage_increase - 63.24) < 0.01 := by
  sorry

end response_rate_increase_approx_l481_48129


namespace intersection_of_A_and_B_l481_48121

def setA (x : Real) : Prop := -1 < x ∧ x < 3
def setB (x : Real) : Prop := -2 < x ∧ x < 2

theorem intersection_of_A_and_B : {x : Real | setA x} ∩ {x : Real | setB x} = {x : Real | -1 < x ∧ x < 2} := 
by
  sorry

end intersection_of_A_and_B_l481_48121


namespace exist_equilateral_triangle_on_parallel_lines_l481_48111

-- Define the concept of lines and points in a relation to them
def Line := ℝ → ℝ -- For simplicity, let's assume lines are functions

-- Define the points A1, A2, A3
structure Point :=
(x : ℝ)
(y : ℝ)

-- Define the concept of parallel lines
def parallel (D1 D2 : Line) : Prop :=
  ∀ x y, D1 x - D2 x = D1 y - D2 y

axiom D1 : Line
axiom D2 : Line
axiom D3 : Line

-- Ensure the lines are parallel
axiom parallel_D1_D2 : parallel D1 D2
axiom parallel_D2_D3 : parallel D2 D3

-- Main statement to prove
theorem exist_equilateral_triangle_on_parallel_lines :
  ∃ (A1 A2 A3 : Point), 
    (A1.y = D1 A1.x) ∧ 
    (A2.y = D2 A2.x) ∧ 
    (A3.y = D3 A3.x) ∧ 
    ((A1.x - A2.x)^2 + (A1.y - A2.y)^2 = (A2.x - A3.x)^2 + (A2.y - A3.y)^2) ∧ 
    ((A2.x - A3.x)^2 + (A2.y - A3.y)^2 = (A3.x - A1.x)^2 + (A3.y - A1.y)^2) := sorry

end exist_equilateral_triangle_on_parallel_lines_l481_48111


namespace remaining_area_l481_48177

-- Given a regular hexagon and a rhombus composed of two equilateral triangles.
-- Hexagon area is 135 square centimeters.

variable (hexagon_area : ℝ) (rhombus_area : ℝ)
variable (is_regular_hexagon : Prop) (is_composed_of_two_equilateral_triangles : Prop)

-- The conditions
def correct_hexagon_area := hexagon_area = 135
def rhombus_is_composed := is_composed_of_two_equilateral_triangles = true
def hexagon_is_regular := is_regular_hexagon = true

-- Goal: Remaining area after cutting out the rhombus should be 75 square centimeters
theorem remaining_area : 
  correct_hexagon_area hexagon_area →
  hexagon_is_regular is_regular_hexagon →
  rhombus_is_composed is_composed_of_two_equilateral_triangles →
  hexagon_area - rhombus_area = 75 :=
by
  sorry

end remaining_area_l481_48177


namespace tank_filling_time_l481_48182

noncomputable def netWaterPerCycle (rateA rateB rateC : ℕ) : ℕ := rateA + rateB - rateC

noncomputable def totalTimeToFill (tankCapacity rateA rateB rateC cycleDuration : ℕ) : ℕ :=
  let netWater := netWaterPerCycle rateA rateB rateC
  let cyclesNeeded := tankCapacity / netWater
  cyclesNeeded * cycleDuration

theorem tank_filling_time :
  totalTimeToFill 750 40 30 20 3 = 45 :=
by
  -- replace "sorry" with the actual proof if required
  sorry

end tank_filling_time_l481_48182


namespace crackers_initial_count_l481_48130

theorem crackers_initial_count (friends : ℕ) (crackers_per_friend : ℕ) (total_crackers : ℕ) :
  (friends = 4) → (crackers_per_friend = 2) → (total_crackers = friends * crackers_per_friend) → total_crackers = 8 :=
by intros h_friends h_crackers_per_friend h_total_crackers
   rw [h_friends, h_crackers_per_friend] at h_total_crackers
   exact h_total_crackers

end crackers_initial_count_l481_48130


namespace quarterly_production_growth_l481_48188

theorem quarterly_production_growth (P_A P_Q2 : ℕ) (x : ℝ)
  (hA : P_A = 500000)
  (hQ2 : P_Q2 = 1820000) :
  50 + 50 * (1 + x) + 50 * (1 + x)^2 = 182 :=
by 
  sorry

end quarterly_production_growth_l481_48188


namespace roshini_spent_on_sweets_l481_48110

theorem roshini_spent_on_sweets
  (initial_amount : Real)
  (amount_given_per_friend : Real)
  (num_friends : Nat)
  (total_amount_given : Real)
  (amount_spent_on_sweets : Real) :
  initial_amount = 10.50 →
  amount_given_per_friend = 3.40 →
  num_friends = 2 →
  total_amount_given = amount_given_per_friend * num_friends →
  amount_spent_on_sweets = initial_amount - total_amount_given →
  amount_spent_on_sweets = 3.70 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end roshini_spent_on_sweets_l481_48110


namespace expand_fraction_product_l481_48128

-- Define the variable x and the condition that x ≠ 0
variable (x : ℝ) (h : x ≠ 0)

-- State the theorem
theorem expand_fraction_product (h : x ≠ 0) :
  3 / 7 * (7 / x^2 + 7 * x - 7 / x) = 3 / x^2 + 3 * x - 3 / x :=
sorry

end expand_fraction_product_l481_48128


namespace binom_26_6_l481_48144

theorem binom_26_6 (h₁ : Nat.choose 25 5 = 53130) (h₂ : Nat.choose 25 6 = 177100) :
  Nat.choose 26 6 = 230230 :=
by
  sorry

end binom_26_6_l481_48144


namespace picture_books_count_l481_48196

-- Definitions based on the given conditions
def total_books : ℕ := 35
def fiction_books : ℕ := 5
def non_fiction_books : ℕ := fiction_books + 4
def autobiographies : ℕ := 2 * fiction_books
def total_non_picture_books : ℕ := fiction_books + non_fiction_books + autobiographies
def picture_books : ℕ := total_books - total_non_picture_books

-- Statement of the problem
theorem picture_books_count : picture_books = 11 :=
by sorry

end picture_books_count_l481_48196


namespace car_speed_constant_l481_48119

theorem car_speed_constant (v : ℝ) : 
  (1 / (v / 3600) - 1 / (80 / 3600) = 2) → v = 3600 / 47 := 
by
  sorry

end car_speed_constant_l481_48119


namespace area_of_triangle_PQR_l481_48185

-- Define the vertices P, Q, and R
def P : (Int × Int) := (-3, 2)
def Q : (Int × Int) := (1, 7)
def R : (Int × Int) := (3, -1)

-- Define the formula for the area of a triangle given vertices
def triangle_area (A B C : Int × Int) : Real :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

-- Define the statement to prove
theorem area_of_triangle_PQR : triangle_area P Q R = 21 := 
  sorry

end area_of_triangle_PQR_l481_48185


namespace berries_ratio_l481_48139

theorem berries_ratio (total_berries : ℕ) (stacy_berries : ℕ) (ratio_stacy_steve : ℕ)
  (h_total : total_berries = 1100) (h_stacy : stacy_berries = 800)
  (h_ratio : stacy_berries = 4 * ratio_stacy_steve) :
  ratio_stacy_steve / (total_berries - stacy_berries - ratio_stacy_steve) = 2 :=
by {
  sorry
}

end berries_ratio_l481_48139


namespace last_two_digits_2005_power_1989_l481_48143

theorem last_two_digits_2005_power_1989 : (2005 ^ 1989) % 100 = 25 :=
by
  sorry

end last_two_digits_2005_power_1989_l481_48143


namespace find_S9_l481_48158

variables {a : ℕ → ℝ} (S : ℕ → ℝ)

-- Condition: an arithmetic sequence with the sum of first n terms S_n.
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (d : ℝ), ∀ n : ℕ, a (n + 1) = a n + d

-- Given condition: a_3 + a_4 + a_5 + a_6 + a_7 = 20.
def given_condition (a : ℕ → ℝ) : Prop :=
  a 3 + a 4 + a 5 + a 6 + a 7 = 20

-- The sum of the first n terms.
def sum_terms (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = (n / 2 : ℝ) * (a 1 + a n)

-- Prove that S_9 = 36.
theorem find_S9 (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h_arithmetic_sequence : arithmetic_sequence a) 
  (h_given_condition : given_condition a)
  (h_sum_terms : sum_terms S a) : 
  S 9 = 36 :=
sorry

end find_S9_l481_48158


namespace probability_of_usable_gas_pipe_l481_48174

theorem probability_of_usable_gas_pipe (x y : ℝ)
  (hx : 75 ≤ x) 
  (hy : 75 ≤ y)
  (hxy : x + y ≤ 225) :
  (∃ x y, 0 < x ∧ 0 < y ∧ x < 300 ∧ y < 300 ∧ x + y > 75 ∧ (300 - x - y) ≥ 75) → 
  ((150 * 150) / (300 * 300 / 2) = (1 / 4)) :=
by {
  sorry
}

end probability_of_usable_gas_pipe_l481_48174


namespace total_revenue_proof_l481_48176

-- Define constants for the problem
def original_price_per_case : ℝ := 25
def first_group_customers : ℕ := 8
def first_group_cases_per_customer : ℕ := 3
def first_group_discount_percentage : ℝ := 0.15
def second_group_customers : ℕ := 4
def second_group_cases_per_customer : ℕ := 2
def second_group_discount_percentage : ℝ := 0.10
def third_group_customers : ℕ := 8
def third_group_cases_per_customer : ℕ := 1

-- Calculate the prices after discount
def discounted_price_first_group : ℝ := original_price_per_case * (1 - first_group_discount_percentage)
def discounted_price_second_group : ℝ := original_price_per_case * (1 - second_group_discount_percentage)
def regular_price : ℝ := original_price_per_case

-- Calculate the total revenue
def total_revenue_first_group : ℝ := first_group_customers * first_group_cases_per_customer * discounted_price_first_group
def total_revenue_second_group : ℝ := second_group_customers * second_group_cases_per_customer * discounted_price_second_group
def total_revenue_third_group : ℝ := third_group_customers * third_group_cases_per_customer * regular_price

def total_revenue : ℝ := total_revenue_first_group + total_revenue_second_group + total_revenue_third_group

-- Prove that the total revenue is $890
theorem total_revenue_proof : total_revenue = 890 := by
  sorry

end total_revenue_proof_l481_48176


namespace find_multiple_l481_48126

-- Defining the conditions
variables (A B k : ℕ)

-- Given conditions
def sum_condition : Prop := A + B = 77
def bigger_number_condition : Prop := A = 42

-- Using the conditions and aiming to prove that k = 5
theorem find_multiple
  (h1 : sum_condition A B)
  (h2 : bigger_number_condition A) :
  6 * B = k * A → k = 5 :=
by
  sorry

end find_multiple_l481_48126


namespace famous_figures_mathematicians_l481_48142

-- List of figures encoded as integers for simplicity
def Bill_Gates := 1
def Gauss := 2
def Liu_Xiang := 3
def Nobel := 4
def Chen_Jingrun := 5
def Chen_Xingshen := 6
def Gorky := 7
def Einstein := 8

-- Set of mathematicians encoded as a set of integers
def mathematicians : Set ℕ := {2, 5, 6}

-- Correct answer set
def correct_answer_set : Set ℕ := {2, 5, 6}

-- The statement to prove
theorem famous_figures_mathematicians:
  mathematicians = correct_answer_set :=
by sorry

end famous_figures_mathematicians_l481_48142


namespace bus_trip_length_l481_48161

theorem bus_trip_length (v T : ℝ) 
    (h1 : 2 * v + (T - 2 * v) * (3 / (2 * v)) + 1 = T / v + 5)
    (h2 : 2 + 30 / v + (T - (2 * v + 30)) * (3 / (2 * v)) + 1 = T / v + 4) : 
    T = 180 :=
    sorry

end bus_trip_length_l481_48161


namespace queenie_worked_4_days_l481_48169

-- Conditions
def daily_earning : ℕ := 150
def overtime_rate : ℕ := 5
def overtime_hours : ℕ := 4
def total_pay : ℕ := 770

-- Question
def number_of_days_worked (d : ℕ) : Prop := 
  daily_earning * d + overtime_rate * overtime_hours * d = total_pay

-- Theorem statement
theorem queenie_worked_4_days : ∃ d : ℕ, number_of_days_worked d ∧ d = 4 := 
by 
  use 4
  unfold number_of_days_worked 
  sorry

end queenie_worked_4_days_l481_48169


namespace minimum_amount_spent_on_boxes_l481_48103

theorem minimum_amount_spent_on_boxes
  (box_length : ℕ) (box_width : ℕ) (box_height : ℕ) 
  (cost_per_box : ℝ) (total_volume_needed : ℕ) :
  box_length = 20 →
  box_width = 20 →
  box_height = 12 →
  cost_per_box = 0.50 →
  total_volume_needed = 2400000 →
  (total_volume_needed / (box_length * box_width * box_height) * cost_per_box) = 250 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  sorry

end minimum_amount_spent_on_boxes_l481_48103


namespace smallest_composite_proof_l481_48151

noncomputable def smallest_composite_no_prime_factors_less_than_15 : ℕ :=
  289

theorem smallest_composite_proof :
  smallest_composite_no_prime_factors_less_than_15 = 289 :=
by
  sorry

end smallest_composite_proof_l481_48151


namespace polynomial_divisibility_l481_48124

theorem polynomial_divisibility (a b c : ℝ) :
  (∀ x : ℝ, (x ^ 4 + a * x ^ 2 + b * x + c) = (x - 1) ^ 3 * (x + 1) →
  a = 0 ∧ b = 2 ∧ c = -1) :=
by
  intros x h
  sorry

end polynomial_divisibility_l481_48124


namespace min_value_expression_l481_48102

variable {a b c : ℝ}

theorem min_value_expression (h1 : a < b) (h2 : a > 0) (h3 : b^2 - 4 * a * c ≤ 0) : 
  ∃ m : ℝ, m = 3 ∧ (∀ x : ℝ, ((a + b + c) / (b - a)) ≥ m) := 
sorry

end min_value_expression_l481_48102


namespace sum_of_cubes_of_roots_l481_48149

theorem sum_of_cubes_of_roots:
  (∀ r s t : ℝ, (r + s + t = 8) ∧ (r * s + s * t + t * r = 9) ∧ (r * s * t = 2) → r^3 + s^3 + t^3 = 344) :=
by
  intros r s t h
  have h1 := h.1
  have h2 := h.2.1
  have h3 := h.2.2
  sorry

end sum_of_cubes_of_roots_l481_48149


namespace pests_eaten_by_frogs_in_week_l481_48125

-- Definitions
def pests_per_day_per_frog : ℕ := 80
def days_per_week : ℕ := 7
def number_of_frogs : ℕ := 5

-- Proposition to prove
theorem pests_eaten_by_frogs_in_week : (pests_per_day_per_frog * days_per_week * number_of_frogs) = 2800 := 
by sorry

end pests_eaten_by_frogs_in_week_l481_48125


namespace randy_brother_ate_l481_48186

-- Definitions
def initial_biscuits : ℕ := 32
def biscuits_from_father : ℕ := 13
def biscuits_from_mother : ℕ := 15
def remaining_biscuits : ℕ := 40

-- Theorem to prove
theorem randy_brother_ate : 
  initial_biscuits + biscuits_from_father + biscuits_from_mother - remaining_biscuits = 20 :=
by
  sorry

end randy_brother_ate_l481_48186


namespace no_nat_numbers_m_n_m_squared_eq_n_squared_plus_2018_l481_48189

theorem no_nat_numbers_m_n_m_squared_eq_n_squared_plus_2018 (m n : ℕ) : ¬ (m^2 = n^2 + 2018) :=
sorry

end no_nat_numbers_m_n_m_squared_eq_n_squared_plus_2018_l481_48189


namespace bicycle_price_l481_48127

theorem bicycle_price (P : ℝ) (h : 0.2 * P = 200) : P = 1000 := 
by
  sorry

end bicycle_price_l481_48127


namespace union_complement_eq_l481_48170

def U : Set ℕ := {0, 1, 2, 3}
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {0, 2, 3}

theorem union_complement_eq : M ∪ (U \ N) = {0, 1, 2} := by
  sorry

end union_complement_eq_l481_48170


namespace like_terms_eq_l481_48171

theorem like_terms_eq : 
  ∀ (x y : ℕ), 
  (x + 2 * y = 3) → 
  (2 * x + y = 9) → 
  (x + y = 4) :=
by
  intros x y h1 h2
  sorry

end like_terms_eq_l481_48171


namespace simplify_expression_l481_48106

theorem simplify_expression (b c : ℝ) : 
  (2 * 3 * b * 4 * b^2 * 5 * b^3 * 6 * b^4 * 7 * c^2 = 5040 * b^10 * c^2) :=
by sorry

end simplify_expression_l481_48106


namespace length_of_train_is_110_l481_48117

-- Define the speeds and time as constants
def speed_train_kmh := 90
def speed_man_kmh := 9
def time_pass_seconds := 4

-- Define the conversion factor from km/h to m/s
def kmh_to_mps (kmh : ℕ) : ℚ := (kmh : ℚ) * (5 / 18)

-- Calculate relative speed in m/s
def relative_speed_mps : ℚ := kmh_to_mps (speed_train_kmh + speed_man_kmh)

-- Define the length of the train in meters
def length_of_train : ℚ := relative_speed_mps * time_pass_seconds

-- The theorem to prove: The length of the train is 110 meters
theorem length_of_train_is_110 : length_of_train = 110 := 
by sorry

end length_of_train_is_110_l481_48117


namespace trajectory_of_midpoint_l481_48141

theorem trajectory_of_midpoint {x y : ℝ} :
  (∃ Mx My : ℝ, (Mx + 3)^2 + My^2 = 4 ∧ (2 * x - 3 = Mx) ∧ (2 * y = My)) →
  x^2 + y^2 = 1 :=
by
  intro h
  sorry

end trajectory_of_midpoint_l481_48141


namespace evaluate_expression_l481_48197

theorem evaluate_expression (b : ℝ) (hb : b = -3) : 
  (3 * b⁻¹ + b⁻¹ / 3) / b = 10 / 27 :=
by 
  -- Lean code typically begins the proof block here
  sorry  -- The proof itself is omitted

end evaluate_expression_l481_48197


namespace inequality_condition_l481_48138

theorem inequality_condition (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2*a*x + a > 0) ↔ (0 < a ∧ a < 1) ∨ (False) := 
sorry

end inequality_condition_l481_48138


namespace solve_system_eqn_l481_48172

theorem solve_system_eqn (x y : ℚ) (h₁ : 3*y - 4*x = 8) (h₂ : 2*y + x = -1) :
  x = -19/11 ∧ y = 4/11 :=
by
  sorry

end solve_system_eqn_l481_48172


namespace zumish_12_words_remainder_l481_48156

def zumishWords n :=
  if n < 2 then (0, 0, 0)
  else if n == 2 then (4, 4, 4)
  else let (a, b, c) := zumishWords (n - 1)
       (2 * (a + c) % 1000, 2 * a % 1000, 2 * b % 1000)

def countZumishWords (n : Nat) :=
  let (a, b, c) := zumishWords n
  (a + b + c) % 1000

theorem zumish_12_words_remainder :
  countZumishWords 12 = 322 :=
by
  intros
  sorry

end zumish_12_words_remainder_l481_48156


namespace circle_through_point_and_tangent_to_lines_l481_48150

theorem circle_through_point_and_tangent_to_lines :
  ∃ h k,
     ((h, k) = (4 / 5, 3 / 5) ∨ (h, k) = (4, -1)) ∧ 
     ((x - h)^2 + (y - k)^2 = 5) :=
by
  let P := (3, 1)
  let l1 := fun x y => x + 2 * y + 3 
  let l2 := fun x y => x + 2 * y - 7 
  sorry

end circle_through_point_and_tangent_to_lines_l481_48150


namespace ratio_of_middle_angle_l481_48190

theorem ratio_of_middle_angle (A B C : ℝ) 
  (h1 : A + B + C = 180)
  (h2 : C = 5 * A)
  (h3 : A = 20) :
  B / A = 3 :=
by
  sorry

end ratio_of_middle_angle_l481_48190


namespace car_rental_cost_l481_48157

theorem car_rental_cost (D R M P C : ℝ) (hD : D = 5) (hR : R = 30) (hM : M = 500) (hP : P = 0.25) 
(hC : C = (R * D) + (P * M)) : C = 275 :=
by
  rw [hD, hR, hM, hP] at hC
  sorry

end car_rental_cost_l481_48157


namespace bird_count_l481_48184

def initial_birds : ℕ := 12
def new_birds : ℕ := 8
def total_birds : ℕ := initial_birds + new_birds

theorem bird_count : total_birds = 20 := by
  sorry

end bird_count_l481_48184


namespace rearrange_infinite_decimal_l481_48148

-- Define the set of digits
def Digit : Type := Fin 10

-- Define the classes of digits
def Class1 (d : Digit) (dec : ℕ → Digit) : Prop :=
  ∃ n : ℕ, ∀ m : ℕ, m > n → dec m ≠ d

def Class2 (d : Digit) (dec : ℕ → Digit) : Prop :=
  ∀ n : ℕ, ∃ m : ℕ, m > n ∧ dec m = d

-- The statement to prove
theorem rearrange_infinite_decimal (dec : ℕ → Digit) (h : ∃ d : Digit, ¬ Class1 d dec) :
  ∃ rearranged : ℕ → Digit, (Class1 d rearranged ∧ Class2 d rearranged) →
  ∃ r : ℚ, ∃ n : ℕ, ∀ m ≥ n, rearranged m = rearranged (m + n) :=
sorry

end rearrange_infinite_decimal_l481_48148


namespace intersection_of_A_and_B_l481_48191

def set_A : Set ℝ := {x : ℝ | x^2 - 5 * x + 6 > 0}
def set_B : Set ℝ := {x : ℝ | x < 1}

theorem intersection_of_A_and_B : set_A ∩ set_B = {x : ℝ | x < 1} :=
sorry

end intersection_of_A_and_B_l481_48191


namespace calculate_expression_l481_48115

theorem calculate_expression:
  202.2 * 89.8 - 20.22 * 186 + 2.022 * 3570 - 0.2022 * 16900 = 18198 :=
by
  sorry

end calculate_expression_l481_48115


namespace union_of_sets_l481_48113

def A : Set ℕ := {2, 3}
def B : Set ℕ := {1, 2}

theorem union_of_sets : A ∪ B = {1, 2, 3} := by
  sorry

end union_of_sets_l481_48113


namespace repeating_seventy_two_exceeds_seventy_two_l481_48112

noncomputable def repeating_decimal (n d : ℕ) : ℚ := n / d

theorem repeating_seventy_two_exceeds_seventy_two :
  repeating_decimal 72 99 - (72 / 100) = (2 / 275) := 
sorry

end repeating_seventy_two_exceeds_seventy_two_l481_48112


namespace handshake_max_l481_48145

theorem handshake_max (N : ℕ) (hN : N > 4) (pN pNm1 : ℕ) 
    (hpN : pN ≠ pNm1) (h1 : ∃ p1, pN ≠ p1) (h2 : ∃ p2, pNm1 ≠ p2) :
    ∀ (i : ℕ), i ≤ N - 2 → i ≤ N - 2 :=
sorry

end handshake_max_l481_48145


namespace find_k_l481_48195

theorem find_k (x y k : ℝ) (h₁ : x = 2) (h₂ : y = -1) (h₃ : y - k * x = 7) : k = -4 :=
by
  sorry

end find_k_l481_48195


namespace opposite_sides_of_line_l481_48163

theorem opposite_sides_of_line (m : ℝ) 
  (ha : (m + 0 - 1) * (2 + m - 1) < 0): 
  -1 < m ∧ m < 1 :=
sorry

end opposite_sides_of_line_l481_48163


namespace symmetric_point_l481_48178

theorem symmetric_point (A B C : ℝ) (hA : A = Real.sqrt 7) (hB : B = 1) :
  C = 2 - Real.sqrt 7 ↔ (A + C) / 2 = B :=
by
  sorry

end symmetric_point_l481_48178


namespace croissant_to_orange_ratio_l481_48147

-- Define the conditions as given in the problem
variables (c o : ℝ)
variable (emily_expenditure : ℝ)
variable (lucas_expenditure : ℝ)

-- Given conditions of expenditures
axiom emily_expenditure_is : emily_expenditure = 5 * c + 4 * o
axiom lucas_expenditure_is : lucas_expenditure = 3 * emily_expenditure
axiom lucas_expenditure_as_purchased : lucas_expenditure = 4 * c + 10 * o

-- Prove the ratio of the cost of a croissant to an orange
theorem croissant_to_orange_ratio : (c / o) = 2 / 11 :=
by sorry

end croissant_to_orange_ratio_l481_48147


namespace inequality_int_part_l481_48164

theorem inequality_int_part (a : ℝ) (n : ℕ) (h1 : 1 ≤ a) (h2 : (0 : ℝ) ≤ n ∧ (n : ℝ) ≤ a) : 
  ⌊a⌋ > (n / (n + 1 : ℝ)) * a := 
by 
  sorry

end inequality_int_part_l481_48164


namespace trapezium_other_side_length_l481_48136

theorem trapezium_other_side_length 
  (side1 : ℝ) (perpendicular_distance : ℝ) (area : ℝ) (side1_val : side1 = 5) 
  (perpendicular_distance_val : perpendicular_distance = 6) (area_val : area = 27) : 
  ∃ other_side : ℝ, other_side = 4 :=
by
  sorry

end trapezium_other_side_length_l481_48136


namespace two_digit_numbers_with_5_as_second_last_digit_l481_48133

theorem two_digit_numbers_with_5_as_second_last_digit:
  ∀ N : ℕ, (10 ≤ N ∧ N ≤ 99) → (∃ k : ℤ, (N * k) % 100 / 10 = 5) ↔ ¬(N % 20 = 0) :=
by
  sorry

end two_digit_numbers_with_5_as_second_last_digit_l481_48133


namespace possible_ratios_of_distances_l481_48134

theorem possible_ratios_of_distances (a b : ℝ) (h : a > b) (h1 : ∃ points : Fin 4 → ℝ × ℝ, 
  ∀ (i j : Fin 4), i ≠ j → 
  (dist (points i) (points j) = a ∨ dist (points i) (points j) = b )) :
  a / b = Real.sqrt 2 ∨ 
  a / b = (1 + Real.sqrt 5) / 2 ∨ 
  a / b = Real.sqrt 3 ∨ 
  a / b = Real.sqrt (2 + Real.sqrt 3) :=
by 
  sorry

end possible_ratios_of_distances_l481_48134


namespace pumps_time_to_empty_pool_l481_48146

theorem pumps_time_to_empty_pool :
  (1 / (1 / 6 + 1 / 9) * 60) = 216 :=
by
  norm_num
  sorry

end pumps_time_to_empty_pool_l481_48146


namespace train_length_is_correct_l481_48152

noncomputable def length_of_train (time_in_seconds : ℝ) (relative_speed : ℝ) : ℝ :=
  relative_speed * time_in_seconds

noncomputable def relative_speed_in_mps (speed_of_train_kmph : ℝ) (speed_of_man_kmph : ℝ) : ℝ :=
  (speed_of_train_kmph + speed_of_man_kmph) * (1000 / 3600)

theorem train_length_is_correct :
  let speed_of_train_kmph := 65.99424046076315
  let speed_of_man_kmph := 6
  let time_in_seconds := 6
  length_of_train time_in_seconds (relative_speed_in_mps speed_of_train_kmph speed_of_man_kmph) = 119.9904 := by
  sorry

end train_length_is_correct_l481_48152


namespace new_person_weight_is_55_l481_48198

variable (W : ℝ) -- Total weight of the original 8 people
variable (new_person_weight : ℝ) -- Weight of the new person
variable (avg_increase : ℝ := 2.5) -- The average weight increase

-- Given conditions
def condition (W new_person_weight : ℝ) : Prop :=
  new_person_weight = W + (8 * avg_increase) + 35 - W

-- The proof statement
theorem new_person_weight_is_55 (W : ℝ) : (new_person_weight = 55) :=
by
  sorry

end new_person_weight_is_55_l481_48198


namespace altitude_difference_l481_48166

theorem altitude_difference 
  (alt_A : ℤ) (alt_B : ℤ) (alt_C : ℤ)
  (hA : alt_A = -102) (hB : alt_B = -80) (hC : alt_C = -25) :
  (max (max alt_A alt_B) alt_C) - (min (min alt_A alt_B) alt_C) = 77 := 
by 
  sorry

end altitude_difference_l481_48166


namespace max_k_value_condition_l481_48105

theorem max_k_value_condition (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  ∃ k, k = 100 ∧ (∀ k < 100, ∃ (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c), 
   (k * a * b * c / (a + b + c) <= (a + b)^2 + (a + b + 4 * c)^2)) :=
sorry

end max_k_value_condition_l481_48105


namespace no_nonzero_ints_l481_48101

theorem no_nonzero_ints (A B : ℤ) (hA : A ≠ 0) (hB : B ≠ 0) :
  (A ∣ (A + B) ∨ B ∣ (A - B)) → false :=
sorry

end no_nonzero_ints_l481_48101


namespace marble_game_solution_l481_48109

theorem marble_game_solution (B R : ℕ) (h1 : B + R = 21) (h2 : (B * (B - 1)) / (21 * 20) = 1 / 2) : B^2 + R^2 = 261 :=
by
  sorry

end marble_game_solution_l481_48109


namespace alice_zoe_difference_l481_48131

-- Definitions of the conditions
def AliceApples := 8
def ZoeApples := 2

-- Theorem statement to prove the difference in apples eaten
theorem alice_zoe_difference : AliceApples - ZoeApples = 6 := by
  -- Proof
  sorry

end alice_zoe_difference_l481_48131


namespace room_width_l481_48120

theorem room_width (length : ℝ) (total_cost : ℝ) (rate_per_sqm : ℝ) (width : ℝ)
  (h_length : length = 5.5)
  (h_total_cost : total_cost = 15400)
  (h_rate_per_sqm : rate_per_sqm = 700)
  (h_area : total_cost = rate_per_sqm * (length * width)) :
  width = 4 := 
sorry

end room_width_l481_48120


namespace age_of_person_A_l481_48122

-- Definitions corresponding to the conditions
variables (x y z : ℕ)
axiom sum_of_ages : x + y = 70
axiom age_difference_A_B : x - z = y
axiom age_difference_B_A_half : y - z = x / 2

-- The proof statement that needs to be proved
theorem age_of_person_A : x = 42 := by 
  -- This is where the proof would go
  sorry

end age_of_person_A_l481_48122


namespace arithmetic_sequence_and_sum_l481_48165

noncomputable def a_n (n : ℕ) : ℤ := 2 * n + 10

def S_n (n : ℕ) : ℤ := n * (12 + 2 * n + 10) / 2

theorem arithmetic_sequence_and_sum :
    (a_n 10 = 30) ∧ 
    (a_n 20 = 50) ∧ 
    (∀ n, S_n n = 11 * n + n^2) ∧ 
    (S_n 3 = 42) :=
by {
    -- a_n 10 = 2 * 10 + 10 = 30
    -- a_n 20 = 2 * 20 + 10 = 50
    -- S_n n = n * (2n + 22) / 2 = 11n + n^2
    -- S_n 3 = 3 * 14 = 42
    sorry
}

end arithmetic_sequence_and_sum_l481_48165


namespace max_volume_of_pyramid_l481_48192

theorem max_volume_of_pyramid
  (a b c : ℝ)
  (h1 : a + b + c = 9)
  (h2 : ∀ (α β : ℝ), α = 30 ∧ β = 45)
  : ∃ V, V = (9 * Real.sqrt 2) / 4 ∧ V = (1/6) * (Real.sqrt 2 / 2) * a * b * c :=
by
  sorry

end max_volume_of_pyramid_l481_48192


namespace hexagon_coloring_l481_48173

def valid_coloring_hexagon : Prop :=
  ∃ (A B C D E F : Fin 8), 
    A ≠ B ∧ A ≠ C ∧ B ≠ C ∧ A ≠ D ∧ B ≠ D ∧ C ≠ D ∧
    B ≠ E ∧ C ≠ E ∧ D ≠ E ∧ A ≠ F ∧ C ≠ F ∧ E ≠ F

theorem hexagon_coloring : ∃ (n : Nat), valid_coloring_hexagon ∧ n = 20160 := 
sorry

end hexagon_coloring_l481_48173


namespace problem_solution_l481_48108

theorem problem_solution (x : ℝ) (h : x^2 - 8*x - 3 = 0) : (x - 1) * (x - 3) * (x - 5) * (x - 7) = 180 :=
by sorry

end problem_solution_l481_48108


namespace A_times_B_is_correct_l481_48194

noncomputable def A : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 2}
noncomputable def B : Set ℝ := {x : ℝ | x ≥ 0}

noncomputable def A_union_B : Set ℝ := {x : ℝ | x ≥ 0}
noncomputable def A_inter_B : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 2}

noncomputable def A_times_B : Set ℝ := {x : ℝ | x ∈ A_union_B ∧ x ∉ A_inter_B}

theorem A_times_B_is_correct :
  A_times_B = {x : ℝ | x > 2} := sorry

end A_times_B_is_correct_l481_48194


namespace tangent_line_at_2_m_range_for_three_roots_l481_48114

def f (x : ℝ) : ℝ := 2 * x ^ 3 - 3 * x ^ 2 + 3

theorem tangent_line_at_2 :
  ∃ k b, k = 12 ∧ b = -17 ∧ (∀ x, 12 * x - (k * (x - 2) + f 2) = b) :=
by
  sorry

theorem m_range_for_three_roots :
  {m : ℝ | ∃ x₀ x₁ x₂, x₀ < x₁ ∧ x₁ < x₂ ∧ f x₀ + m = 0 ∧ f x₁ + m = 0 ∧ f x₂ + m = 0} = 
  {m : ℝ | -3 < m ∧ m < -2} :=
by
  sorry

end tangent_line_at_2_m_range_for_three_roots_l481_48114


namespace libby_quarters_left_after_payment_l481_48160

noncomputable def quarters_needed (usd_target : ℝ) (usd_per_quarter : ℝ) : ℝ := 
  usd_target / usd_per_quarter

noncomputable def quarters_left (initial_quarters : ℝ) (used_quarters : ℝ) : ℝ := 
  initial_quarters - used_quarters

theorem libby_quarters_left_after_payment
  (initial_quarters : ℝ) (usd_target : ℝ) (usd_per_quarter : ℝ) 
  (h_initial : initial_quarters = 160) 
  (h_usd_target : usd_target = 35) 
  (h_usd_per_quarter : usd_per_quarter = 0.25) : 
  quarters_left initial_quarters (quarters_needed usd_target usd_per_quarter) = 20 := 
by
  sorry

end libby_quarters_left_after_payment_l481_48160


namespace distinct_positive_roots_l481_48187

noncomputable def f (a x : ℝ) : ℝ := x^4 - x^3 + 8 * a * x^2 - a * x + a^2

theorem distinct_positive_roots (a : ℝ) :
  0 < a ∧ a < 1/24 → (∀ x1 x2 x3 x4 : ℝ, f a x1 = 0 ∧ 0 < x1 ∧ f a x2 = 0 ∧ 0 < x2 ∧ f a x3 = 0 ∧ 0 < x3 ∧ f a x4 = 0 ∧ 0 < x4 ∧ 
  x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4) ↔ (1/25 < a ∧ a < 1/24) :=
sorry

end distinct_positive_roots_l481_48187


namespace subtraction_of_twos_from_ones_l481_48193

theorem subtraction_of_twos_from_ones (n : ℕ) : 
  let ones := (10^n - 1) * 10^n + (10^n - 1)
  let twos := 2 * (10^n - 1)
  ones - twos = (10^n - 1) * (10^n - 1) :=
by
  sorry

end subtraction_of_twos_from_ones_l481_48193


namespace intersection_l481_48132

def A : Set ℝ := { x | -2 < x ∧ x < 3 }
def B : Set ℝ := { x | x > -1 }

theorem intersection (x : ℝ) : x ∈ (A ∩ B) ↔ -1 < x ∧ x < 3 := by
  sorry

end intersection_l481_48132


namespace complex_expression_equality_l481_48107

-- Define the basic complex number properties and operations.
def i : ℂ := Complex.I -- Define the imaginary unit

theorem complex_expression_equality (a b : ℤ) :
  (3 - 4 * i) * ((-4 + 2 * i) ^ 2) = -28 - 96 * i :=
by
  -- Syntactical proof placeholders
  sorry

end complex_expression_equality_l481_48107


namespace light_flash_fraction_l481_48175

theorem light_flash_fraction (flash_interval : ℕ) (total_flashes : ℕ) (seconds_in_hour : ℕ) (fraction_of_hour : ℚ) :
  flash_interval = 6 →
  total_flashes = 600 →
  seconds_in_hour = 3600 →
  fraction_of_hour = 1 →
  (total_flashes * flash_interval) / seconds_in_hour = fraction_of_hour := by
  sorry

end light_flash_fraction_l481_48175


namespace moles_CO2_formed_l481_48104

-- Define the conditions based on the problem statement
def moles_HCl := 1
def moles_NaHCO3 := 1

-- Define the reaction equation in equivalence terms
def chemical_equation (hcl : Nat) (nahco3 : Nat) : Nat :=
  if hcl = 1 ∧ nahco3 = 1 then 1 else 0

-- State the proof problem
theorem moles_CO2_formed : chemical_equation moles_HCl moles_NaHCO3 = 1 :=
by
  -- The proof goes here
  sorry

end moles_CO2_formed_l481_48104


namespace mass_percentage_of_O_in_CaCO3_l481_48167

-- Assuming the given conditions as definitions
def molar_mass_Ca : ℝ := 40.08
def molar_mass_C : ℝ := 12.01
def molar_mass_O : ℝ := 16.00
def formula_CaCO3 : (ℕ × ℝ) := (1, molar_mass_Ca) -- 1 atom of Calcium
def formula_CaCO3_C : (ℕ × ℝ) := (1, molar_mass_C) -- 1 atom of Carbon
def formula_CaCO3_O : (ℕ × ℝ) := (3, molar_mass_O) -- 3 atoms of Oxygen

-- Desired result
def mass_percentage_O_CaCO3 : ℝ := 47.95

-- The theorem statement to be proven
theorem mass_percentage_of_O_in_CaCO3 :
  let molar_mass_CaCO3 := formula_CaCO3.2 + formula_CaCO3_C.2 + (formula_CaCO3_O.1 * formula_CaCO3_O.2)
  let mass_percentage_O := (formula_CaCO3_O.1 * formula_CaCO3_O.2 / molar_mass_CaCO3) * 100
  mass_percentage_O = mass_percentage_O_CaCO3 :=
by
  sorry

end mass_percentage_of_O_in_CaCO3_l481_48167


namespace inverse_proportionality_l481_48183

theorem inverse_proportionality (a b c k a1 a2 b1 b2 c1 c2 : ℝ)
    (h1 : a * b * c = k)
    (h2 : a1 / a2 = 3 / 4)
    (h3 : b1 = 2 * b2)
    (h4 : c1 ≠ 0 ∧ c2 ≠ 0) :
    c1 / c2 = 2 / 3 :=
sorry

end inverse_proportionality_l481_48183


namespace circle_radius_increase_l481_48159

-- Defining the problem conditions and the resulting proof
theorem circle_radius_increase (r n : ℝ) (h : π * (r + n)^2 = 3 * π * r^2) : r = n * (Real.sqrt 3 - 1) / 2 :=
sorry  -- Proof is left as an exercise

end circle_radius_increase_l481_48159


namespace ducks_cows_problem_l481_48140

theorem ducks_cows_problem (D C : ℕ) (h : 2 * D + 4 * C = 2 * (D + C) + 24) : C = 12 := 
  sorry

end ducks_cows_problem_l481_48140


namespace gwen_spent_zero_l481_48181

theorem gwen_spent_zero 
  (m : ℕ) 
  (d : ℕ) 
  (S : ℕ) 
  (h1 : m = 8) 
  (h2 : d = 5)
  (h3 : (m - S) = (d - S) + 3) : 
  S = 0 :=
by
  sorry

end gwen_spent_zero_l481_48181


namespace johns_total_amount_l481_48137

def amount_from_grandpa : ℕ := 30
def multiplier : ℕ := 3
def amount_from_grandma : ℕ := amount_from_grandpa * multiplier
def total_amount : ℕ := amount_from_grandpa + amount_from_grandma

theorem johns_total_amount :
  total_amount = 120 :=
by
  sorry

end johns_total_amount_l481_48137
