import Mathlib

namespace a_values_in_terms_of_x_l2092_209227

open Real

-- Definitions for conditions
variables (a b x y : ℝ)
variables (h1 : a^3 - b^3 = 27 * x^3)
variables (h2 : a - b = y)
variables (h3 : y = 2 * x)

-- Theorem to prove
theorem a_values_in_terms_of_x : 
  (a = x + 5 * x / sqrt 6) ∨ (a = x - 5 * x / sqrt 6) :=
sorry

end a_values_in_terms_of_x_l2092_209227


namespace dot_product_is_4_l2092_209215

-- Define vectors a and b
def a (x : ℝ) : ℝ × ℝ := (2, x)
def b : ℝ × ℝ := (1, -1)

-- Define the condition that a is parallel to (a + b)
def is_parallel (u v : ℝ × ℝ) : Prop := 
  (u.1 * v.2 - u.2 * v.1) = 0

theorem dot_product_is_4 (x : ℝ) (h_parallel : is_parallel (a x) (a x + b)) : 
  (a x).1 * b.1 + (a x).2 * b.2 = 4 :=
sorry

end dot_product_is_4_l2092_209215


namespace value_of_ab_l2092_209213

theorem value_of_ab (a b : ℤ) (h1 : ∀ x : ℤ, -1 < x ∧ x < 1 → (2 * x < a + 1) ∧ (x > 2 * b + 3)) :
  (a + 1) * (b - 1) = -6 :=
by
  sorry

end value_of_ab_l2092_209213


namespace parabola_intersection_value_l2092_209245

theorem parabola_intersection_value (a : ℝ) (h : a^2 - a - 1 = 0) : a^2 - a + 2014 = 2015 :=
by
  sorry

end parabola_intersection_value_l2092_209245


namespace sin_double_angle_l2092_209254

variable {α : Real}

theorem sin_double_angle (h : Real.tan α = 2) : Real.sin (2 * α) = 4 / 5 := 
sorry

end sin_double_angle_l2092_209254


namespace cylinder_height_l2092_209257

variable (r h : ℝ) (SA : ℝ)

theorem cylinder_height (h : ℝ) (r : ℝ) (SA : ℝ) (h_eq : h = 2) (r_eq : r = 3) (SA_eq : SA = 30 * Real.pi) :
  SA = 2 * Real.pi * r ^ 2 + 2 * Real.pi * r * h → h = 2 :=
by
  intros
  sorry

end cylinder_height_l2092_209257


namespace net_percentage_change_is_correct_l2092_209253

def initial_price : Float := 100.0

def price_after_first_year (initial: Float) := initial * (1 - 0.05)

def price_after_second_year (price1: Float) := price1 * (1 + 0.10)

def price_after_third_year (price2: Float) := price2 * (1 + 0.04)

def price_after_fourth_year (price3: Float) := price3 * (1 - 0.03)

def price_after_fifth_year (price4: Float) := price4 * (1 + 0.08)

def final_price := price_after_fifth_year (price_after_fourth_year (price_after_third_year (price_after_second_year (price_after_first_year initial_price))))

def net_percentage_change (initial final: Float) := ((final - initial) / initial) * 100

theorem net_percentage_change_is_correct :
  net_percentage_change initial_price final_price = 13.85 := by
  sorry

end net_percentage_change_is_correct_l2092_209253


namespace value_of_P_dot_Q_l2092_209260

def P : Set ℝ := {x | Real.log x / Real.log 2 < 1}
def Q : Set ℝ := {x | abs (x - 2) < 1}
def P_dot_Q (P Q : Set ℝ) : Set ℝ := {x | x ∈ P ∧ x ∉ Q}

theorem value_of_P_dot_Q : P_dot_Q P Q = {x | 0 < x ∧ x ≤ 1} := by
  sorry

end value_of_P_dot_Q_l2092_209260


namespace sum_of_possible_values_of_g_l2092_209289

def f (x : ℝ) : ℝ := x^2 - 9 * x + 20
def g (x : ℝ) : ℝ := 3 * x - 4

theorem sum_of_possible_values_of_g :
  let x1 := (9 + 3 * Real.sqrt 5) / 2
  let x2 := (9 - 3 * Real.sqrt 5) / 2
  g x1 + g x2 = 19 :=
by
  sorry

end sum_of_possible_values_of_g_l2092_209289


namespace solve_inequality_l2092_209205

theorem solve_inequality (x : ℝ) : (|x - 3| + |x - 5| ≥ 4) ↔ (x ≤ 2 ∨ x ≥ 6) :=
by
  sorry

end solve_inequality_l2092_209205


namespace greatest_y_value_l2092_209208

theorem greatest_y_value (x y : ℤ) (h : x * y + 3 * x + 2 * y = -2) : y ≤ 1 :=
sorry

end greatest_y_value_l2092_209208


namespace min_objective_value_l2092_209250

theorem min_objective_value (x y : ℝ) 
  (h1 : x + y ≥ 2) 
  (h2 : x - y ≤ 2) 
  (h3 : y ≥ 1) : ∃ (z : ℝ), z = x + 3 * y ∧ z = 4 :=
by
  -- Provided proof omitted
  sorry

end min_objective_value_l2092_209250


namespace largest_n_l2092_209279

variable {a : ℕ → ℤ}
variable {S : ℕ → ℤ}

-- Conditions
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 1) - a n = a 2 - a 1

axiom a1_gt_zero : a 1 > 0
axiom a2011_a2012_sum_gt_zero : a 2011 + a 2012 > 0
axiom a2011_a2012_prod_lt_zero : a 2011 * a 2012 < 0

-- Sum of first n terms of an arithmetic sequence
def sequence_sum (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  n * (a 1 + a n) / 2

-- Problem statement to prove
theorem largest_n (H : is_arithmetic_sequence a) :
  ∀ n, (sequence_sum a 4022 > 0) ∧ (sequence_sum a 4023 < 0) → n = 4022 := by
  sorry

end largest_n_l2092_209279


namespace mono_sum_eq_five_l2092_209282

-- Conditions
def term1 (x y : ℝ) (m : ℕ) : ℝ := x^2 * y^m
def term2 (x y : ℝ) (n : ℕ) : ℝ := x^n * y^3

def is_monomial_sum (x y : ℝ) (m n : ℕ) : Prop :=
  term1 x y m + term2 x y n = x^(2:ℕ) * y^(3:ℕ)

-- Theorem stating the result
theorem mono_sum_eq_five (x y : ℝ) (m n : ℕ) (h : is_monomial_sum x y m n) : m + n = 5 :=
by
  sorry

end mono_sum_eq_five_l2092_209282


namespace butternut_wood_figurines_l2092_209233

theorem butternut_wood_figurines (B : ℕ) (basswood_blocks : ℕ) (aspen_blocks : ℕ) (butternut_blocks : ℕ) 
  (basswood_figurines_per_block : ℕ) (aspen_figurines_per_block : ℕ) (total_figurines : ℕ) 
  (h_basswood_blocks : basswood_blocks = 15)
  (h_aspen_blocks : aspen_blocks = 20)
  (h_butternut_blocks : butternut_blocks = 20)
  (h_basswood_figurines_per_block : basswood_figurines_per_block = 3)
  (h_aspen_figurines_per_block : aspen_figurines_per_block = 2 * basswood_figurines_per_block)
  (h_total_figurines : total_figurines = 245) :
  B = 4 :=
by
  -- Definitions based on the given conditions
  let basswood_figurines := basswood_blocks * basswood_figurines_per_block
  let aspen_figurines := aspen_blocks * aspen_figurines_per_block
  let figurines_from_butternut := total_figurines - basswood_figurines - aspen_figurines
  -- Calculate the number of figurines per block of butternut wood
  let butternut_figurines_per_block := figurines_from_butternut / butternut_blocks
  -- The objective is to prove that the number of figurines per block of butternut wood is 4
  exact sorry

end butternut_wood_figurines_l2092_209233


namespace find_m_n_l2092_209293

theorem find_m_n (x m n : ℤ) : (x + 2) * (x + 3) = x^2 + m * x + n → m = 5 ∧ n = 6 :=
by {
    sorry
}

end find_m_n_l2092_209293


namespace total_people_after_four_years_l2092_209236

-- Define initial conditions
def initial_total_people : Nat := 9
def board_members : Nat := 3
def regular_members_initial : Nat := initial_total_people - board_members
def years : Nat := 4

-- Define the function for regular members over the years
def regular_members (n : Nat) : Nat :=
  if n = 0 then 
    regular_members_initial
  else 
    2 * regular_members (n - 1)

theorem total_people_after_four_years :
  regular_members years = 96 := 
sorry

end total_people_after_four_years_l2092_209236


namespace trajectory_of_G_l2092_209203

-- Define the ellipse
def ellipse (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

-- Define the trajectory equation
def trajectory (x y : ℝ) : Prop :=
  9 * x^2 / 4 + 3 * y^2 = 1

-- State the theorem
theorem trajectory_of_G (P G : ℝ × ℝ) (hP : ellipse P.1 P.2) (hG_relation : ∃ k : ℝ, k = 2 ∧ P = (3 * G.1, 3 * G.2)) :
  trajectory G.1 G.2 :=
by
  sorry

end trajectory_of_G_l2092_209203


namespace area_EFCD_l2092_209275

-- Defining the geometrical setup and measurements of the trapezoid
variables (AB CD AD BC : ℝ) (h1 : AB = 10) (h2 : CD = 30) (h_altitude : ∃ h : ℝ, h = 18)

-- Defining the midpoints E and F of AD and BC respectively
variables (E F : ℝ) (h_E : E = AD / 2) (h_F : F = BC / 2)

-- Define the intersection of diagonals and the ratio condition
variables (AC BD G : ℝ) (h_ratio : ∃ r : ℝ, r = 1/2)

-- Proving the area of quadrilateral EFCD
theorem area_EFCD : EFCD_area = 225 :=
sorry

end area_EFCD_l2092_209275


namespace inequality_solution_set_nonempty_l2092_209238

-- Define the statement
theorem inequality_solution_set_nonempty (m : ℝ) : 
  (∃ x : ℝ, |x + 1| + |x - 1| < m) ↔ m > 2 :=
by
  sorry

end inequality_solution_set_nonempty_l2092_209238


namespace alice_ride_top_speed_l2092_209287

-- Define the conditions
variables (x y : Real) -- x is the hours at 25 mph, y is the hours at 15 mph.
def distance_eq : Prop := 25 * x + 15 * y + 10 * (9 - x - y) = 162
def time_eq : Prop := x + y ≤ 9

-- Define the final answer
def final_answer : Prop := x = 2.7

-- The statement to prove
theorem alice_ride_top_speed : distance_eq x y ∧ time_eq x y → final_answer x := sorry

end alice_ride_top_speed_l2092_209287


namespace work_done_in_five_days_l2092_209210

-- Define the work rates of A, B, and C
def work_rate_A : ℚ := 1 / 11
def work_rate_B : ℚ := 1 / 5
def work_rate_C : ℚ := 1 / 55

-- Define the work done in a cycle of 2 days
def work_one_cycle : ℚ := (work_rate_A + work_rate_B) + (work_rate_A + work_rate_C)

-- The total work needed to be done is 1
def total_work : ℚ := 1

-- The number of days in a cycle of 2 days
def days_per_cycle : ℕ := 2

-- Proving that the work will be done in exactly 5 days
theorem work_done_in_five_days :
  ∃ n : ℕ, n = 5 →
  n * (work_rate_A + work_rate_B) + (n-1) * (work_rate_A + work_rate_C) = total_work :=
by
  -- Sorry to skip the detailed proof steps
  sorry

end work_done_in_five_days_l2092_209210


namespace simplify_and_evaluate_expression_l2092_209297

variable (x y : ℚ)

theorem simplify_and_evaluate_expression (hx : x = 1) (hy : y = 1 / 2) :
  (3 * x + 2 * y) * (3 * x - 2 * y) - (x - y) ^ 2 = 31 / 4 :=
by
  rw [hx, hy]
  sorry

end simplify_and_evaluate_expression_l2092_209297


namespace distance_3_units_l2092_209261

theorem distance_3_units (x : ℤ) (h : |x + 2| = 3) : x = -5 ∨ x = 1 := by
  sorry

end distance_3_units_l2092_209261


namespace sum_of_infinite_perimeters_l2092_209278

noncomputable def geometric_series_sum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem sum_of_infinite_perimeters (a : ℝ) :
  let first_perimeter := 3 * a
  let common_ratio := (1/3 : ℝ)
  let S := geometric_series_sum first_perimeter common_ratio 0
  S = (9 * a / 2) :=
by
  sorry

end sum_of_infinite_perimeters_l2092_209278


namespace expression_behavior_l2092_209222

theorem expression_behavior (x : ℝ) (h1 : -3 < x) (h2 : x < 2) :
  ¬∃ m, ∀ y : ℝ, (h3 : -3 < y) → (h4 : y < 2) → (x ≠ 1) → (y ≠ 1) → 
    (m <= (y^2 - 3*y + 3) / (y - 1)) ∧ 
    (m >= (y^2 - 3*y + 3) / (y - 1)) :=
sorry

end expression_behavior_l2092_209222


namespace min_bottles_to_fill_large_bottle_l2092_209272

theorem min_bottles_to_fill_large_bottle (large_bottle_ml : Nat) (small_bottle1_ml : Nat) (small_bottle2_ml : Nat) (total_bottles : Nat) :
  large_bottle_ml = 800 ∧ small_bottle1_ml = 45 ∧ small_bottle2_ml = 60 ∧ total_bottles = 14 →
  ∃ x y : Nat, x * small_bottle1_ml + y * small_bottle2_ml = large_bottle_ml ∧ x + y = total_bottles :=
by
  intro h
  sorry

end min_bottles_to_fill_large_bottle_l2092_209272


namespace total_rainfall_l2092_209223

theorem total_rainfall :
  let monday := 0.12962962962962962
  let tuesday := 0.35185185185185186
  let wednesday := 0.09259259259259259
  let thursday := 0.25925925925925924
  let friday := 0.48148148148148145
  let saturday := 0.2222222222222222
  let sunday := 0.4444444444444444
  (monday + tuesday + wednesday + thursday + friday + saturday + sunday) = 1.9814814814814815 :=
by
  -- proof to be filled here
  sorry

end total_rainfall_l2092_209223


namespace distinct_roots_implies_m_greater_than_half_find_m_given_condition_l2092_209244

-- Define the quadratic equation with a free parameter m
def quadratic_eq (x : ℝ) (m : ℝ) : Prop :=
  x^2 - 4 * x - 2 * m + 5 = 0

-- Prove that if the quadratic equation has distinct roots, then m > 1/2
theorem distinct_roots_implies_m_greater_than_half (m : ℝ) :
  (∃ x₁ x₂ : ℝ, quadratic_eq x₁ m ∧ quadratic_eq x₂ m ∧ x₁ ≠ x₂) →
  m > 1 / 2 :=
by
  sorry

-- Given that x₁ and x₂ satisfy both the quadratic equation and the sum-product condition, find the value of m
theorem find_m_given_condition (m : ℝ) (x₁ x₂ : ℝ) :
  quadratic_eq x₁ m ∧ quadratic_eq x₂ m ∧ x₁ ≠ x₂ ∧ (x₁ * x₂ + x₁ + x₂ = m^2 + 6) → 
  m = 1 :=
by
  sorry

end distinct_roots_implies_m_greater_than_half_find_m_given_condition_l2092_209244


namespace identify_a_b_l2092_209270

theorem identify_a_b (a b : ℝ) (h : ∀ x y : ℝ, (⌊a * x + b * y⌋ + ⌊b * x + a * y⌋ = (a + b) * ⌊x + y⌋)) : 
  (a = 0 ∧ b = 0) ∨ (a = 1 ∧ b = 1) :=
sorry

end identify_a_b_l2092_209270


namespace A_can_finish_remaining_work_in_6_days_l2092_209291

-- Condition: A can finish the work in 18 days
def A_work_rate := 1 / 18

-- Condition: B can finish the work in 15 days
def B_work_rate := 1 / 15

-- Given B worked for 10 days
def B_days_worked := 10

-- Calculation of the remaining work
def remaining_work := 1 - B_days_worked * B_work_rate

-- Calculation of the time for A to finish the remaining work
def A_remaining_days := remaining_work / A_work_rate

-- The theorem to prove
theorem A_can_finish_remaining_work_in_6_days : A_remaining_days = 6 := 
by 
  -- The proof is not required, so we use sorry to skip it.
  sorry

end A_can_finish_remaining_work_in_6_days_l2092_209291


namespace angle_A_is_equilateral_l2092_209219

namespace TriangleProof

variables {A B C : ℝ} {a b c : ℝ}

-- Given condition (a+b+c)(a-b-c) + 3bc = 0
def condition1 (a b c : ℝ) : Prop := (a + b + c) * (a - b - c) + 3 * b * c = 0

-- Given condition a = 2c * cos B
def condition2 (a c B : ℝ) : Prop := a = 2 * c * Real.cos B

-- Prove that if (a+b+c)(a-b-c) + 3bc = 0, then A = π / 3
theorem angle_A (h1 : condition1 a b c) : A = Real.pi / 3 :=
sorry

-- Prove that if a = 2c * cos B and A = π / 3, then ∆ ABC is an equilateral triangle
theorem is_equilateral (h2 : condition2 a c B) (hA : A = Real.pi / 3) : 
  b = c ∧ a = b ∧ B = C :=
sorry

end TriangleProof

end angle_A_is_equilateral_l2092_209219


namespace Shara_savings_l2092_209295

theorem Shara_savings (P : ℝ) (d : ℝ) (paid : ℝ):
  d = 0.08 → paid = 184 → P = 200 → (P * (1 - d) = paid) → (P - paid = 16) :=
by
  intros hd hpaid hP heq
  -- It follows from the conditions given
  sorry

end Shara_savings_l2092_209295


namespace percentage_increase_l2092_209226

def initialProductivity := 120
def totalArea := 1440
def daysInitialProductivity := 2
def daysAheadOfSchedule := 2

theorem percentage_increase :
  let originalDays := totalArea / initialProductivity
  let daysWithIncrease := originalDays - daysAheadOfSchedule
  let daysWithNewProductivity := daysWithIncrease - daysInitialProductivity
  let remainingArea := totalArea - (daysInitialProductivity * initialProductivity)
  let newProductivity := remainingArea / daysWithNewProductivity
  let increase := ((newProductivity - initialProductivity) / initialProductivity) * 100
  increase = 25 :=
by
  sorry

end percentage_increase_l2092_209226


namespace ocean_depth_l2092_209263

theorem ocean_depth (t : ℕ) (v : ℕ) (h : ℕ)
  (h_t : t = 8)
  (h_v : v = 1500) :
  h = 6000 :=
by
  sorry

end ocean_depth_l2092_209263


namespace quadratic_solution_difference_l2092_209206

theorem quadratic_solution_difference : 
  ∃ a b : ℝ, (a^2 - 12 * a + 20 = 0) ∧ (b^2 - 12 * b + 20 = 0) ∧ (a > b) ∧ (a - b = 8) :=
by
  sorry

end quadratic_solution_difference_l2092_209206


namespace tree_height_increase_l2092_209214

-- Definitions given in the conditions
def h0 : ℝ := 4
def h (t : ℕ) (x : ℝ) : ℝ := h0 + t * x

-- Proof statement
theorem tree_height_increase (x : ℝ) :
  h 6 x = (4 / 3) * h 4 x + h 4 x → x = 2 :=
by
  intro h6_eq
  rw [h, h] at h6_eq
  norm_num at h6_eq
  sorry

end tree_height_increase_l2092_209214


namespace male_students_in_grade_l2092_209229

-- Define the total number of students and the number of students in the sample
def total_students : ℕ := 1200
def sample_students : ℕ := 30

-- Define the number of female students in the sample
def female_students_sample : ℕ := 14

-- Calculate the number of male students in the sample
def male_students_sample := sample_students - female_students_sample

-- State the main theorem
theorem male_students_in_grade :
  (male_students_sample : ℕ) * total_students / sample_students = 640 :=
by
  -- placeholder for calculations based on provided conditions
  sorry

end male_students_in_grade_l2092_209229


namespace savings_if_together_l2092_209252

def window_price : ℕ := 100

def free_windows_for_six_purchased : ℕ := 2

def windows_needed_Dave : ℕ := 9
def windows_needed_Doug : ℕ := 10

def total_individual_cost (windows_purchased : ℕ) : ℕ :=
  100 * windows_purchased

def total_cost_with_deal (windows_purchased: ℕ) : ℕ :=
  let sets_of_6 := windows_purchased / 6
  let remaining_windows := windows_purchased % 6
  100 * (sets_of_6 * 6 + remaining_windows)

def combined_savings (windows_needed_Dave: ℕ) (windows_needed_Doug: ℕ) : ℕ :=
  let total_windows := windows_needed_Dave + windows_needed_Doug
  total_individual_cost windows_needed_Dave 
  + total_individual_cost windows_needed_Doug 
  - total_cost_with_deal total_windows

theorem savings_if_together : combined_savings windows_needed_Dave windows_needed_Doug = 400 :=
by
  sorry

end savings_if_together_l2092_209252


namespace garden_length_l2092_209209

theorem garden_length (w l : ℝ) (h1: l = 2 * w) (h2 : 2 * l + 2 * w = 180) : l = 60 := 
by
  sorry

end garden_length_l2092_209209


namespace best_purchase_option_l2092_209224

-- Define the prices and discount conditions for each store
def technik_city_price_before_discount : ℝ := 2000 + 4000
def technomarket_price_before_discount : ℝ := 1500 + 4800

def technik_city_discount : ℝ := technik_city_price_before_discount * 0.10
def technomarket_bonus : ℝ := technomarket_price_before_discount * 0.20

def technik_city_final_price : ℝ := technik_city_price_before_discount - technik_city_discount
def technomarket_final_price : ℝ := technomarket_price_before_discount

-- The theorem stating the ultimate proof problem
theorem best_purchase_option : technik_city_final_price < technomarket_final_price :=
by
  -- Replace 'sorry' with the actual proof if required
  sorry

end best_purchase_option_l2092_209224


namespace infinite_series_sum_l2092_209211

noncomputable def partial_sum (n : ℕ) : ℚ := (2 * n - 1) / (n * (n + 1) * (n + 2))

theorem infinite_series_sum : (∑' n, partial_sum (n + 1)) = 3 / 4 :=
by
  sorry

end infinite_series_sum_l2092_209211


namespace passengers_on_third_plane_l2092_209220

theorem passengers_on_third_plane (
  P : ℕ
) (h1 : 600 - 2 * 50 = 500) -- Speed of the first plane
  (h2 : 600 - 2 * 60 = 480) -- Speed of the second plane
  (h_avg : (500 + 480 + (600 - 2 * P)) / 3 = 500) -- Average speed condition
  : P = 40 := by sorry

end passengers_on_third_plane_l2092_209220


namespace rational_sign_product_l2092_209240

theorem rational_sign_product (a b c : ℚ) (h : |a| / a + |b| / b + |c| / c = 1) : abc / |abc| = -1 := 
by
  -- Proof to be provided
  sorry

end rational_sign_product_l2092_209240


namespace coefficient_of_m5n4_in_expansion_l2092_209248

def binomial_coefficient (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem coefficient_of_m5n4_in_expansion : binomial_coefficient 9 5 = 126 := by
  sorry

end coefficient_of_m5n4_in_expansion_l2092_209248


namespace scientific_notation_of_18500000_l2092_209243

-- Definition of scientific notation function
def scientific_notation (n : ℕ) : string := sorry

-- Problem statement
theorem scientific_notation_of_18500000 : 
  scientific_notation 18500000 = "1.85 × 10^7" :=
sorry

end scientific_notation_of_18500000_l2092_209243


namespace enough_cat_food_for_six_days_l2092_209259

variable (B S : ℝ)

theorem enough_cat_food_for_six_days :
  (B > S) →
  (B < 2 * S) →
  (B + 2 * S = 2 * ((B + 2 * S) / 2)) →
  ¬ (4 * B + 4 * S ≥ 6 * ((B + 2 * S) / 2)) :=
by
  -- Proof logic goes here.
  sorry

end enough_cat_food_for_six_days_l2092_209259


namespace find_a_value_l2092_209274

theorem find_a_value (a : ℝ) (x : ℝ) :
  (a + 1) * x^2 + (a^2 + 1) + 8 * x = 9 →
  a + 1 ≠ 0 →
  a^2 + 1 = 9 →
  a = 2 * Real.sqrt 2 :=
by
  intro h1 h2 h3
  sorry

end find_a_value_l2092_209274


namespace m_plus_n_is_23_l2092_209292

noncomputable def find_m_plus_n : ℕ := 
  let A := 12
  let B := 4
  let C := 3
  let D := 3

  -- Declare the radius of E
  let radius_E : ℚ := (21 / 2)
  
  -- Let radius_E be written as m / n where m and n are relatively prime
  let (m : ℕ) := 21
  let (n : ℕ) := 2

  -- Calculate m + n
  m + n

theorem m_plus_n_is_23 : find_m_plus_n = 23 :=
by
  -- Proof is omitted
  sorry

end m_plus_n_is_23_l2092_209292


namespace sangwoo_gave_away_notebooks_l2092_209249

variables (n : ℕ)

theorem sangwoo_gave_away_notebooks
  (h1 : 12 - n + 34 - 3 * n = 30) :
  n = 4 :=
by
  sorry

end sangwoo_gave_away_notebooks_l2092_209249


namespace find_a_prove_f_pos_l2092_209232

noncomputable def f (x a : ℝ) : ℝ := (x - a) * Real.log x + (1 / 2) * x

theorem find_a (a x0 : ℝ) (hx0 : x0 > 0) (h_tangent : (x0 - a) * Real.log x0 + (1 / 2) * x0 = (1 / 2) * x0 ∧ Real.log x0 - a / x0 + 3 / 2 = 1 / 2) :
  a = 1 :=
sorry

theorem prove_f_pos (a : ℝ) (h_range : 1 / (2 * Real.exp 1) < a ∧ a < 2 * Real.sqrt (Real.exp 1)) (x : ℝ) (hx : x > 0) :
  f x a > 0 :=
sorry

end find_a_prove_f_pos_l2092_209232


namespace book_pages_total_l2092_209247

-- Define the conditions as hypotheses
def total_pages (P : ℕ) : Prop :=
  let read_first_day := P / 2
  let read_second_day := P / 4
  let read_third_day := P / 6
  let read_total := read_first_day + read_second_day + read_third_day
  let remaining_pages := P - read_total
  remaining_pages = 20

-- The proof statement
theorem book_pages_total (P : ℕ) (h : total_pages P) : P = 240 := sorry

end book_pages_total_l2092_209247


namespace sequence_periodicity_a5_a2019_l2092_209267

theorem sequence_periodicity_a5_a2019 (a : ℕ → ℝ) (h1 : a 1 = 1) (h2 : ∀ n : ℕ, 0 < n → a n * a (n + 2) = 3 * a (n + 1)) :
  a 5 * a 2019 = 27 :=
sorry

end sequence_periodicity_a5_a2019_l2092_209267


namespace minimum_value_ineq_l2092_209299

noncomputable def problem_statement (a b c : ℝ) (h : a + b + c = 3) : Prop :=
  (a > 0) ∧ (b > 0) ∧ (c > 0) → (1 / (a + b) + 1 / (b + c) + 1 / (c + a) ≥ 3 / 2)

theorem minimum_value_ineq (a b c : ℝ) (h : a + b + c = 3) : problem_statement a b c h :=
  sorry

end minimum_value_ineq_l2092_209299


namespace black_to_brown_ratio_l2092_209212

-- Definitions of the given conditions
def total_shoes : ℕ := 66
def brown_shoes : ℕ := 22
def black_shoes : ℕ := total_shoes - brown_shoes

-- Lean 4 problem statement: Prove the ratio of black shoes to brown shoes is 2:1
theorem black_to_brown_ratio :
  (black_shoes / Nat.gcd black_shoes brown_shoes) = 2 ∧ (brown_shoes / Nat.gcd black_shoes brown_shoes) = 1 := by
sorry

end black_to_brown_ratio_l2092_209212


namespace identical_machine_production_l2092_209283

-- Definitions based on given conditions
def machine_production_rate (machines : ℕ) (rate : ℕ) :=
  rate / machines

def bottles_in_minute (machines : ℕ) (rate_per_machine : ℕ) :=
  machines * rate_per_machine

def total_bottles (bottle_rate_per_minute : ℕ) (minutes : ℕ) :=
  bottle_rate_per_minute * minutes

-- Theorem to prove based on the question == answer given conditions
theorem identical_machine_production :
  ∀ (machines_initial machines_final : ℕ) (bottles_per_minute : ℕ) (minutes : ℕ),
    machines_initial = 6 →
    machines_final = 12 →
    bottles_per_minute = 270 →
    minutes = 4 →
    total_bottles (bottles_in_minute machines_final (machine_production_rate machines_initial bottles_per_minute)) minutes = 2160 := by
  intros
  sorry

end identical_machine_production_l2092_209283


namespace tangent_parallel_x_axis_coordinates_l2092_209230

theorem tangent_parallel_x_axis_coordinates :
  ∃ (x y : ℝ), (y = x^2 - 3 * x) ∧ (2 * x - 3 = 0) ∧ (x = 3 / 2) ∧ (y = -9 / 4) :=
by
  use (3 / 2)
  use (-9 / 4)
  sorry

end tangent_parallel_x_axis_coordinates_l2092_209230


namespace semicircle_radius_l2092_209264

noncomputable def radius_of_inscribed_semicircle (BD height : ℝ) (h_base : BD = 20) (h_height : height = 21) : ℝ :=
  let AB := Real.sqrt (21^2 + 10^2)
  let s := 2 * Real.sqrt 541
  let area := 20 * 21
  (area) / (s * 2)

theorem semicircle_radius (BD height : ℝ) (h_base : BD = 20) (h_height : height = 21)
  : radius_of_inscribed_semicircle BD height h_base h_height = 210 / Real.sqrt 541 :=
sorry

end semicircle_radius_l2092_209264


namespace percentage_of_non_technicians_l2092_209258

theorem percentage_of_non_technicians (total_workers technicians non_technicians permanent_technicians permanent_non_technicians temporary_workers : ℝ)
  (h1 : technicians = 0.5 * total_workers)
  (h2 : non_technicians = total_workers - technicians)
  (h3 : permanent_technicians = 0.5 * technicians)
  (h4 : permanent_non_technicians = 0.5 * non_technicians)
  (h5 : temporary_workers = 0.5 * total_workers) :
  (non_technicians / total_workers) * 100 = 50 :=
by
  -- Proof is omitted
  sorry

end percentage_of_non_technicians_l2092_209258


namespace largest_square_side_length_l2092_209221

theorem largest_square_side_length (smallest_square_side next_square_side : ℕ) (h1 : smallest_square_side = 1) 
(h2 : next_square_side = smallest_square_side + 6) :
  ∃ x : ℕ, x = 7 :=
by
  existsi 7
  sorry

end largest_square_side_length_l2092_209221


namespace average_age_of_team_l2092_209231

def total_age (A : ℕ) (N : ℕ) := A * N
def wicket_keeper_age (A : ℕ) := A + 3
def remaining_players_age (A : ℕ) (N : ℕ) (W : ℕ) := (total_age A N) - (A + W)

theorem average_age_of_team
  (A : ℕ)
  (N : ℕ)
  (H1 : N = 11)
  (H2 : A = 28)
  (W : ℕ)
  (H3 : W = wicket_keeper_age A)
  (H4 : (wicket_keeper_age A) = A + 3)
  : (remaining_players_age A N W) / (N - 2) = A - 1 :=
by
  rw [H1, H2, H3, H4]; sorry

end average_age_of_team_l2092_209231


namespace refreshment_stand_distance_l2092_209255

theorem refreshment_stand_distance 
  (A B S : ℝ) -- Positions of the camps and refreshment stand
  (dist_A_highway : A = 400) -- Distance from the first camp to the highway
  (dist_B_A : B = 700) -- Distance from the second camp directly across the highway
  (equidistant : ∀ x, S = x ∧ dist (S, A) = dist (S, B)) : 
  S = 500 := -- Distance from the refreshment stand to each camp is 500 meters
sorry

end refreshment_stand_distance_l2092_209255


namespace polynomial_two_distinct_negative_real_roots_l2092_209286

theorem polynomial_two_distinct_negative_real_roots :
  ∀ (p : ℝ), 
  (∃ (x1 x2 : ℝ), x1 < 0 ∧ x2 < 0 ∧ x1 ≠ x2 ∧ 
    (x1^4 + p*x1^3 + 3*x1^2 + p*x1 + 4 = 0) ∧ 
    (x2^4 + p*x2^3 + 3*x2^2 + p*x2 + 4 = 0)) ↔ 
  (p ≤ -2 ∨ p ≥ 2) :=
by
  sorry

end polynomial_two_distinct_negative_real_roots_l2092_209286


namespace prime_square_minus_one_divisible_by_24_l2092_209265

theorem prime_square_minus_one_divisible_by_24 (p : ℕ) (h_prime : Prime p) (h_gt_3 : p > 3) : 
  ∃ k : ℕ, p^2 - 1 = 24 * k := by
sorry

end prime_square_minus_one_divisible_by_24_l2092_209265


namespace lines_parallel_l2092_209277

def line1 (x y : ℝ) : Prop := x - y + 2 = 0
def line2 (x y : ℝ) : Prop := x - y + 1 = 0

theorem lines_parallel : 
  (∀ x y, line1 x y ↔ y = x + 2) ∧ 
  (∀ x y, line2 x y ↔ y = x + 1) ∧ 
  ∃ m₁ m₂ c₁ c₂, (∀ x y, (y = m₁ * x + c₁) ↔ line1 x y) ∧ (∀ x y, (y = m₂ * x + c₂) ↔ line2 x y) ∧ m₁ = m₂ ∧ c₁ ≠ c₂ :=
by
  sorry

end lines_parallel_l2092_209277


namespace arun_completes_work_alone_in_70_days_l2092_209296

def arun_days (A : ℕ) : Prop :=
  ∃ T : ℕ, (A > 0) ∧ (T > 0) ∧ 
           (∀ (work_done_by_arun_in_1_day work_done_by_tarun_in_1_day : ℝ),
            work_done_by_arun_in_1_day = 1 / A ∧
            work_done_by_tarun_in_1_day = 1 / T ∧
            (work_done_by_arun_in_1_day + work_done_by_tarun_in_1_day = 1 / 10) ∧
            (4 * (work_done_by_arun_in_1_day + work_done_by_tarun_in_1_day) = 4 / 10) ∧
            (42 * work_done_by_arun_in_1_day = 6 / 10) )

theorem arun_completes_work_alone_in_70_days : arun_days 70 :=
  sorry

end arun_completes_work_alone_in_70_days_l2092_209296


namespace hidden_dots_are_32_l2092_209200

theorem hidden_dots_are_32 
  (visible_faces : List ℕ)
  (h_visible : visible_faces = [1, 2, 3, 4, 4, 5, 6, 6])
  (num_dice : ℕ)
  (h_num_dice : num_dice = 3)
  (faces_per_die : List ℕ)
  (h_faces_per_die : faces_per_die = [1, 2, 3, 4, 5, 6]) :
  63 - visible_faces.sum = 32 := by
  sorry

end hidden_dots_are_32_l2092_209200


namespace max_value_of_expression_achieve_max_value_l2092_209268

theorem max_value_of_expression : 
  ∀ x : ℝ, -3 * x ^ 2 + 18 * x - 4 ≤ 77 :=
by
  -- Placeholder proof
  sorry

theorem achieve_max_value : 
  ∃ x : ℝ, -3 * x ^ 2 + 18 * x - 4 = 77 :=
by
  -- Placeholder proof
  sorry

end max_value_of_expression_achieve_max_value_l2092_209268


namespace perpendicular_vectors_x_value_l2092_209237

theorem perpendicular_vectors_x_value 
  (x : ℝ) (a b : ℝ × ℝ) (hₐ : a = (1, -2)) (hᵦ : b = (3, x)) (h_perpendicular : a.1 * b.1 + a.2 * b.2 = 0) : 
  x = 3 / 2 :=
by
  -- The proof is not required, hence we use 'sorry'
  sorry

end perpendicular_vectors_x_value_l2092_209237


namespace sin_sum_triangle_inequality_l2092_209202

theorem sin_sum_triangle_inequality (A B C : ℝ) (h : A + B + C = Real.pi) :
  Real.sin A + Real.sin B + Real.sin C ≤ (3 * Real.sqrt 3) / 2 :=
sorry

end sin_sum_triangle_inequality_l2092_209202


namespace maximize_profit_l2092_209281

-- Define the variables
variables (x y a b : ℝ)
variables (P : ℝ)

-- Define the conditions and the proof goal
theorem maximize_profit
  (h1 : x + 3 * y = 240)
  (h2 : 2 * x + y = 130)
  (h3 : a + b = 100)
  (h4 : a ≥ 4 * b)
  (ha : a = 80)
  (hb : b = 20) :
  x = 30 ∧ y = 70 ∧ P = (40 * a + 90 * b) - (30 * a + 70 * b) := 
by
  -- We assume the solution steps are solved correctly as provided
  sorry

end maximize_profit_l2092_209281


namespace initial_investment_calculation_l2092_209207

-- Define the conditions
def r : ℝ := 0.10
def n : ℕ := 1
def t : ℕ := 2
def A : ℝ := 6050.000000000001
def one : ℝ := 1

-- The goal is to prove that the initial principal P is 5000 under these conditions
theorem initial_investment_calculation (P : ℝ) : P = 5000 :=
by
  have interest_compounded : ℝ := (one + r / n) ^ (n * t)
  have total_amount : ℝ := P * interest_compounded
  sorry

end initial_investment_calculation_l2092_209207


namespace polynomial_root_divisibility_l2092_209204

noncomputable def p (x : ℤ) (a b c : ℤ) : ℤ := x^3 + a * x^2 + b * x + c

theorem polynomial_root_divisibility (a b c : ℤ) (h : ∃ u v : ℤ, p 0 a b c = (u * v * u * v)) :
  2 * (p (-1) a b c) ∣ (p 1 a b c + p (-1) a b c - 2 * (1 + p 0 a b c)) :=
sorry

end polynomial_root_divisibility_l2092_209204


namespace smallest_integer_with_eight_factors_l2092_209228

theorem smallest_integer_with_eight_factors : ∃ N : ℕ, (∀ p q : ℕ, N = p ^ 1 * q ^ 3 ∨ N = p ^ 3 * q ^ 1) ∧ ∀ M : ℕ, (∀ p q : ℕ, N = p ^ 1 * q ^ 3 ∨ N = p ^ 3 * q ^ 1) → N ≤ M :=
sorry -- Proof not provided.

end smallest_integer_with_eight_factors_l2092_209228


namespace gambler_final_amount_l2092_209276

-- Define initial amount of money
def initial_amount := 100

-- Define the multipliers
def win_multiplier := 4 / 3
def loss_multiplier := 2 / 3
def double_win_multiplier := 5 / 3

-- Define the gambler scenario (WWLWLWLW)
def scenario := [double_win_multiplier, win_multiplier, loss_multiplier, win_multiplier, loss_multiplier, win_multiplier, loss_multiplier, win_multiplier]

-- Function to compute final amount given initial amount, number of wins and losses, and the scenario
def final_amount (initial: ℚ) (multipliers: List ℚ) : ℚ :=
  multipliers.foldl (· * ·) initial

-- Prove that the final amount after all multipliers are applied is approximately equal to 312.12
theorem gambler_final_amount : abs (final_amount initial_amount scenario - 312.12) < 0.01 :=
by
  sorry

end gambler_final_amount_l2092_209276


namespace denominator_expression_l2092_209269

theorem denominator_expression (x y a b E : ℝ)
  (h1 : x / y = 3)
  (h2 : (2 * a - x) / E = 3)
  (h3 : a / b = 4.5) : E = 3 * b - y :=
sorry

end denominator_expression_l2092_209269


namespace simplify_sqrt8_minus_sqrt2_l2092_209284

theorem simplify_sqrt8_minus_sqrt2 :
  (Real.sqrt 8 - Real.sqrt 2 = Real.sqrt 2) :=
sorry

end simplify_sqrt8_minus_sqrt2_l2092_209284


namespace find_m_l2092_209234

theorem find_m (m : ℕ) (hm_pos : m > 0)
  (h1 : Nat.lcm 40 m = 120)
  (h2 : Nat.lcm m 45 = 180) : m = 60 := sorry

end find_m_l2092_209234


namespace x_proportionality_find_x_value_l2092_209225

theorem x_proportionality (m n : ℝ) (x z : ℝ) (h1 : ∀ y, x = m * y^4) (h2 : ∀ z, y = n / z^2) (h3 : x = 4) (h4 : z = 8) :
  ∃ k, ∀ z : ℝ, x = k / z^8 := 
sorry

theorem find_x_value (m n : ℝ) (k : ℝ) (h1 : ∀ y, x = m * y^4) (h2 : ∀ z, y = n / z^2) (h5 : k = 67108864) :
  ∀ z, (z = 32 → x = 1 / 16) :=
sorry

end x_proportionality_find_x_value_l2092_209225


namespace ab_value_l2092_209235

theorem ab_value (a b : ℝ) (h1 : a - b = 4) (h2 : a^2 + b^2 = 80) : a * b = 32 := by
  sorry

end ab_value_l2092_209235


namespace loss_percentage_on_first_book_l2092_209218

variable (C1 C2 SP L : ℝ)
variable (total_cost : ℝ := 540)
variable (C1_value : ℝ := 315)
variable (gain_percentage : ℝ := 0.19)
variable (common_selling_price : ℝ := 267.75)

theorem loss_percentage_on_first_book :
  C1 = C1_value →
  C2 = total_cost - C1 →
  SP = 1.19 * C2 →
  SP = C1 - (L / 100 * C1) →
  L = 15 :=
sorry

end loss_percentage_on_first_book_l2092_209218


namespace p_iff_q_l2092_209290

theorem p_iff_q (a b : ℝ) : (a > b) ↔ (a^3 > b^3) :=
sorry

end p_iff_q_l2092_209290


namespace quadratic_inequality_solution_l2092_209271

theorem quadratic_inequality_solution (x : ℝ) : (x^2 - x - 12 < 0) ↔ (-3 < x ∧ x < 4) :=
by sorry

end quadratic_inequality_solution_l2092_209271


namespace intersection_M_N_l2092_209239

def M : Set ℝ := { x | |x - 2| ≤ 1 }
def N : Set ℝ := { x | x^2 - x - 6 ≥ 0 }

theorem intersection_M_N : M ∩ N = {3} := by
  sorry

end intersection_M_N_l2092_209239


namespace solve_quadratic_l2092_209246

theorem solve_quadratic (x : ℝ) (h : x^2 - 4 = 0) : x = 2 ∨ x = -2 :=
by sorry

end solve_quadratic_l2092_209246


namespace range_of_a_l2092_209262

theorem range_of_a (a : ℝ) : (¬ ∃ x : ℝ, a*x^2 - 2*a*x + 3 ≤ 0) ↔ (0 ≤ a ∧ a < 3) := 
sorry

end range_of_a_l2092_209262


namespace candy_difference_l2092_209273

def given_away : ℕ := 6
def left : ℕ := 5
def difference : ℕ := given_away - left

theorem candy_difference :
  difference = 1 :=
by
  sorry

end candy_difference_l2092_209273


namespace cricket_initial_overs_l2092_209294

/-- Prove that the number of initial overs played was 10. -/
theorem cricket_initial_overs 
  (target : ℝ) 
  (initial_run_rate : ℝ) 
  (remaining_run_rate : ℝ) 
  (remaining_overs : ℕ)
  (h_target : target = 282)
  (h_initial_run_rate : initial_run_rate = 4.6)
  (h_remaining_run_rate : remaining_run_rate = 5.9)
  (h_remaining_overs : remaining_overs = 40) 
  : ∃ x : ℝ, x = 10 := 
by
  sorry

end cricket_initial_overs_l2092_209294


namespace martin_bell_ringing_l2092_209251

theorem martin_bell_ringing (B S : ℕ) (hB : B = 36) (hS : S = B / 3 + 4) : S + B = 52 :=
sorry

end martin_bell_ringing_l2092_209251


namespace paula_aunt_gave_her_total_money_l2092_209216

theorem paula_aunt_gave_her_total_money :
  let shirt_price := 11
  let pants_price := 13
  let shirts_bought := 2
  let money_left := 74
  let total_spent := shirts_bought * shirt_price + pants_price
  total_spent + money_left = 109 :=
by
  let shirt_price := 11
  let pants_price := 13
  let shirts_bought := 2
  let money_left := 74
  let total_spent := shirts_bought * shirt_price + pants_price
  show total_spent + money_left = 109
  sorry

end paula_aunt_gave_her_total_money_l2092_209216


namespace equal_circle_radius_l2092_209298

theorem equal_circle_radius (r R : ℝ) (h1: r > 0) (h2: R > 0)
  : ∃ x : ℝ, x = r * R / (R + r) :=
by 
  sorry

end equal_circle_radius_l2092_209298


namespace minimum_perimeter_area_l2092_209241

-- Define the focus point F of the parabola and point A
def F : ℝ × ℝ := (1, 0)  -- Focus for the parabola y² = 4x is (1, 0)
def A : ℝ × ℝ := (5, 4)

-- Parabola definition as a set of points (x, y) such that y² = 4x
def is_on_parabola (B : ℝ × ℝ) : Prop := B.2 * B.2 = 4 * B.1

-- The area of triangle ABF
def triangle_area (A B F : ℝ × ℝ) : ℝ := 
  0.5 * abs ((A.1 - B.1) * (A.2 - F.2) - (A.1 - F.1) * (A.2 - B.2))

-- Statement: The area of ∆ABF is 2 when the perimeter of ∆ABF is minimum
theorem minimum_perimeter_area (B : ℝ × ℝ) (hB : is_on_parabola B) 
  (hA_B_perimeter_min : ∀ (C : ℝ × ℝ), is_on_parabola C → 
                        (dist A C + dist C F ≥ dist A B + dist B F)) : 
  triangle_area A B F = 2 := 
sorry

end minimum_perimeter_area_l2092_209241


namespace q_value_l2092_209266

-- Define the conditions and the problem statement
theorem q_value (a b m p q : ℚ) (h1 : a * b = 3) 
  (h2 : (a + 1 / b) * (b + 1 / a) = q) : 
  q = 16 / 3 :=
by
  sorry

end q_value_l2092_209266


namespace average_is_equal_l2092_209256

theorem average_is_equal (x : ℝ) :
  (1 / 3) * (2 * x + 4 + 5 * x + 3 + 3 * x + 8) = 3 * x - 5 → 
  x = -30 :=
by
  sorry

end average_is_equal_l2092_209256


namespace sequence_a_n_correctness_l2092_209288

theorem sequence_a_n_correctness (a : ℕ → ℚ) (h₀ : a 1 = 1)
  (h₁ : ∀ n : ℕ, n > 0 → 2 * a (n + 1) = 2 * a n + 1) : a 2 = 1.5 := by
  sorry

end sequence_a_n_correctness_l2092_209288


namespace center_of_hyperbola_l2092_209242

theorem center_of_hyperbola :
  (∃ h k : ℝ, ∀ x y : ℝ, (3*y + 3)^2 / 49 - (2*x - 5)^2 / 9 = 1 ↔ x = h ∧ y = k) → 
  h = 5 / 2 ∧ k = -1 :=
by
  sorry

end center_of_hyperbola_l2092_209242


namespace find_dividend_l2092_209285

def dividend (divisor quotient remainder : ℕ) : ℕ :=
  (divisor * quotient) + remainder

theorem find_dividend (divisor quotient remainder : ℕ) (h_divisor : divisor = 16) (h_quotient : quotient = 8) (h_remainder : remainder = 4) :
  dividend divisor quotient remainder = 132 :=
by
  sorry

end find_dividend_l2092_209285


namespace rectangular_solid_surface_area_l2092_209201

theorem rectangular_solid_surface_area
  (length : ℕ) (width : ℕ) (depth : ℕ)
  (h_length : length = 9) (h_width : width = 8) (h_depth : depth = 5) :
  2 * (length * width + width * depth + length * depth) = 314 := 
  by
  sorry

end rectangular_solid_surface_area_l2092_209201


namespace part_a_winner_part_b_winner_part_c_winner_a_part_c_winner_b_l2092_209280

-- Define the game rules and conditions for the proof
def takeMatches (total_matches : Nat) (taken_matches : Nat) : Nat :=
  total_matches - taken_matches

-- Part (a) statement
theorem part_a_winner (total_matches : Nat) (m : Nat) : 
  (total_matches = 25) → (m = 3) → True := 
  sorry

-- Part (b) statement
theorem part_b_winner (total_matches : Nat) (m : Nat) : 
  (total_matches = 25) → (m = 3) → True := 
  sorry

-- Part (c) generalized statement for game type (a)
theorem part_c_winner_a (n : Nat) (m : Nat) : 
  (total_matches = 2 * n + 1) → True :=
  sorry

-- Part (c) generalized statement for game type (b)
theorem part_c_winner_b (n : Nat) (m : Nat) : 
  (total_matches = 2 * n + 1) → True :=
  sorry

end part_a_winner_part_b_winner_part_c_winner_a_part_c_winner_b_l2092_209280


namespace compute_f_2_neg3_neg1_l2092_209217

def f (p q r : ℤ) : ℚ := (r + p : ℚ) / (r - q + 1 : ℚ)

theorem compute_f_2_neg3_neg1 : f 2 (-3) (-1) = 1 / 3 := 
by
  sorry

end compute_f_2_neg3_neg1_l2092_209217
