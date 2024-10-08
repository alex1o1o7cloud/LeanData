import Mathlib

namespace min_value_x_plus_y_l233_233222

theorem min_value_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1/x + 9/y = 1) : x + y ≥ 16 :=
sorry

end min_value_x_plus_y_l233_233222


namespace total_copies_l233_233012

-- Conditions: Defining the rates of two copy machines and the time duration
def rate1 : ℕ := 35 -- rate in copies per minute for the first machine
def rate2 : ℕ := 65 -- rate in copies per minute for the second machine
def time : ℕ := 30 -- time in minutes

-- The theorem stating that the total number of copies made by both machines in 30 minutes is 3000
theorem total_copies : rate1 * time + rate2 * time = 3000 := by
  sorry

end total_copies_l233_233012


namespace asymptote_hole_sum_l233_233112

noncomputable def number_of_holes (f : ℝ → ℝ) : ℕ := -- Assume a definition to count holes
sorry

noncomputable def number_of_vertical_asymptotes (f : ℝ → ℝ) : ℕ := -- Assume a definition to count vertical asymptotes
sorry

noncomputable def number_of_horizontal_asymptotes (f : ℝ → ℝ) : ℕ := -- Assume a definition to count horizontal asymptotes
sorry

noncomputable def number_of_oblique_asymptotes (f : ℝ → ℝ) : ℕ := -- Assume a definition to count oblique asymptotes
sorry

theorem asymptote_hole_sum :
  let f := λ x => (x^2 + 4*x + 3) / (x^3 - 2*x^2 - x + 2)
  let a := number_of_holes f
  let b := number_of_vertical_asymptotes f
  let c := number_of_horizontal_asymptotes f
  let d := number_of_oblique_asymptotes f
  a + 2 * b + 3 * c + 4 * d = 8 :=
by
  sorry

end asymptote_hole_sum_l233_233112


namespace taqeesha_grade_correct_l233_233110

-- Definitions for conditions
def total_score_of_24_students := 24 * 82
def total_score_of_25_students (T: ℕ) := 25 * 84
def taqeesha_grade := 132

-- Theorem statement forming the proof problem
theorem taqeesha_grade_correct
    (h1: total_score_of_24_students + taqeesha_grade = total_score_of_25_students taqeesha_grade): 
    taqeesha_grade = 132 :=
by
  sorry

end taqeesha_grade_correct_l233_233110


namespace runner_time_second_half_l233_233093

theorem runner_time_second_half (v : ℝ) (h1 : 20 / v + 4 = 40 / v) : 40 / v = 8 :=
by
  sorry

end runner_time_second_half_l233_233093


namespace rectangle_ratio_l233_233840

theorem rectangle_ratio (t a b : ℝ) (h₀ : b = 2 * a) (h₁ : (t + 2 * a) ^ 2 = 3 * t ^ 2) : b / a = 2 :=
by
  sorry

end rectangle_ratio_l233_233840


namespace customers_in_each_car_l233_233199

-- Conditions given in the problem
def cars : ℕ := 10
def purchases_sports : ℕ := 20
def purchases_music : ℕ := 30

-- Total purchases are equal to the total number of customers
def total_purchases : ℕ := purchases_sports + purchases_music
def total_customers (C : ℕ) : ℕ := cars * C

-- Lean statement to prove that the number of customers in each car is 5
theorem customers_in_each_car : (∃ C : ℕ, total_customers C = total_purchases) ∧ (∀ C : ℕ, total_customers C = total_purchases → C = 5) :=
by
  sorry

end customers_in_each_car_l233_233199


namespace largest_and_smallest_correct_l233_233623

noncomputable def largest_and_smallest (x y : ℝ) (hx : x < 0) (hy : -1 < y ∧ y < 0) : ℝ × ℝ :=
  if hx_y : x * y > 0 then
    if hx_y_sq : x * y * y > x then
      (x * y, x)
    else
      sorry
  else
    sorry

theorem largest_and_smallest_correct {x y : ℝ} (hx : x < 0) (hy : -1 < y ∧ y < 0) :
  largest_and_smallest x y hx hy = (x * y, x) :=
by {
  sorry
}

end largest_and_smallest_correct_l233_233623


namespace inverse_function_correct_l233_233854

noncomputable def f (x : ℝ) : ℝ := 3 - 7 * x

noncomputable def g (x : ℝ) : ℝ := (3 - x) / 7

theorem inverse_function_correct : ∀ x : ℝ, f (g x) = x ∧ g (f x) = x :=
by
  intro x
  sorry

end inverse_function_correct_l233_233854


namespace symmetric_points_power_l233_233818

theorem symmetric_points_power 
  (a b : ℝ) 
  (h1 : 2 * a = 8) 
  (h2 : 2 = a + b) :
  a^b = 1/16 := 
by sorry

end symmetric_points_power_l233_233818


namespace each_cut_piece_weight_l233_233660

theorem each_cut_piece_weight (L : ℕ) (W : ℕ) (c : ℕ) 
  (hL : L = 20) (hW : W = 150) (hc : c = 2) : (L / c) * W = 1500 := by
  sorry

end each_cut_piece_weight_l233_233660


namespace moles_of_ca_oh_2_l233_233986

-- Define the chemical reaction
def ca_o := 1
def h_2_o := 1
def ca_oh_2 := ca_o + h_2_o

-- Prove the result of the reaction
theorem moles_of_ca_oh_2 :
  ca_oh_2 = 1 := by sorry

end moles_of_ca_oh_2_l233_233986


namespace sum_of_numbers_gt_1_1_equals_3_9_l233_233794

noncomputable def sum_of_elements_gt_1_1 : Float :=
  let numbers := [1.4, 9 / 10, 1.2, 0.5, 13 / 10]
  let numbers_gt_1_1 := List.filter (fun x => x > 1.1) numbers
  List.sum numbers_gt_1_1

theorem sum_of_numbers_gt_1_1_equals_3_9 :
  sum_of_elements_gt_1_1 = 3.9 := by
  sorry

end sum_of_numbers_gt_1_1_equals_3_9_l233_233794


namespace parallel_line_with_intercept_sum_l233_233129

theorem parallel_line_with_intercept_sum (c : ℝ) :
  (∀ x y : ℝ, 2 * x + 3 * y + 5 = 0 → 2 * x + 3 * y + c = 0) ∧ 
  (-c / 3 - c / 2 = 6) → 
  (10 * x + 15 * y - 36 = 0) :=
by
  sorry

end parallel_line_with_intercept_sum_l233_233129


namespace inequality_holds_l233_233328

theorem inequality_holds (a b : ℕ) (ha : a > 1) (hb : b > 2) : a ^ b + 1 ≥ b * (a + 1) :=
sorry

end inequality_holds_l233_233328


namespace smartphone_cost_l233_233117

theorem smartphone_cost :
  let current_savings : ℕ := 40
  let weekly_saving : ℕ := 15
  let num_months : ℕ := 2
  let weeks_in_month : ℕ := 4 
  let total_weeks := num_months * weeks_in_month
  let total_savings := weekly_saving * total_weeks
  let total_money := current_savings + total_savings
  total_money = 160 := by
  sorry

end smartphone_cost_l233_233117


namespace number_of_palindromes_divisible_by_6_l233_233831

theorem number_of_palindromes_divisible_by_6 :
  let is_palindrome (n : ℕ) : Prop := n / 1000 = n % 10 ∧ (n / 100 % 10) = (n / 10 % 10)
  let valid_digits (n : ℕ) : Prop := n >= 1000 ∧ n < 10000
  let divisible_6 (n : ℕ) : Prop := n % 6 = 0
  (Finset.filter (λ n => is_palindrome n ∧ valid_digits n ∧ divisible_6 n) (Finset.range 10000)).card = 13 :=
by
  -- We define what it means to be a palindrome between 1000 and 10000
  let is_palindrome (n : ℕ) : Prop := n / 1000 = n % 10 ∧ n / 100 % 10 = n / 10 % 10
  
  -- We define a valid number between 1000 and 10000
  let valid_digits (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000
  
  -- We define what it means to be divisible by 6
  let divisible_6 (n : ℕ) : Prop := n % 6 = 0

  -- Filtering the range 10000 within valid four-digit palindromes and checking for multiples of 6
  exact sorry

end number_of_palindromes_divisible_by_6_l233_233831


namespace a_divides_b_l233_233830

theorem a_divides_b (a b : ℕ) (h_pos : 0 < a ∧ 0 < b)
    (h : ∀ n : ℕ, a^n ∣ b^(n+1)) : a ∣ b :=
by
  sorry

end a_divides_b_l233_233830


namespace marble_ratio_l233_233044

theorem marble_ratio (A V X : ℕ) 
  (h1 : A + 5 = V - 5)
  (h2 : V + X = (A - X) + 30) : X / 5 = 2 :=
by
  sorry

end marble_ratio_l233_233044


namespace general_formula_for_sequence_l233_233448

def sequence_terms (a : ℕ → ℚ) : Prop :=
∀ n : ℕ, n > 0 → a n = 1 / (n * (n + 1))

def seq_conditions (a : ℕ → ℚ) (S : ℕ → ℚ) : Prop :=
a 1 = 1 / 2 ∧ (∀ n : ℕ, n > 0 → S n = n^2 * a n)

theorem general_formula_for_sequence :
  ∃ a S : ℕ → ℚ, seq_conditions a S ∧ sequence_terms a := by
  sorry

end general_formula_for_sequence_l233_233448


namespace animath_workshop_lists_l233_233666

/-- The 79 trainees of the Animath workshop each choose an activity for the free afternoon 
among 5 offered activities. It is known that:
- The swimming pool was at least as popular as soccer.
- The students went shopping in groups of 5.
- No more than 4 students played cards.
- At most one student stayed in their room.
We write down the number of students who participated in each activity.
How many different lists could we have written? --/
theorem animath_workshop_lists :
  ∃ (l : ℕ), l = Nat.choose 81 2 := 
sorry

end animath_workshop_lists_l233_233666


namespace bond_paper_cost_l233_233385

/-!
# Bond Paper Cost Calculation

This theorem calculates the total cost to buy the required amount of each type of bond paper, given the specified conditions.
-/

def cost_of_ream (sheets_per_ream : ℤ) (cost_per_ream : ℤ) (required_sheets : ℤ) : ℤ :=
  let reams_needed := (required_sheets + sheets_per_ream - 1) / sheets_per_ream
  reams_needed * cost_per_ream

theorem bond_paper_cost :
  let total_sheets := 5000
  let required_A := 2500
  let required_B := 1500
  let remaining_sheets := total_sheets - required_A - required_B
  let cost_A := cost_of_ream 500 27 required_A
  let cost_B := cost_of_ream 400 24 required_B
  let cost_C := cost_of_ream 300 18 remaining_sheets
  cost_A + cost_B + cost_C = 303 := 
by
  sorry

end bond_paper_cost_l233_233385


namespace macy_miles_left_l233_233801

theorem macy_miles_left (goal : ℕ) (daily_miles : ℕ) (days_run : ℕ) 
  (H1 : goal = 24) 
  (H2 : daily_miles = 3) 
  (H3 : days_run = 6) 
  : goal - daily_miles * days_run = 6 := 
by 
  sorry

end macy_miles_left_l233_233801


namespace total_clowns_l233_233958

def num_clown_mobiles : Nat := 5
def clowns_per_mobile : Nat := 28

theorem total_clowns : num_clown_mobiles * clowns_per_mobile = 140 := by
  sorry

end total_clowns_l233_233958


namespace original_average_l233_233913

theorem original_average (n : ℕ) (k : ℕ) (new_avg : ℝ) 
  (h1 : n = 35) 
  (h2 : k = 5) 
  (h3 : new_avg = 125) : 
  (new_avg / k) = 25 :=
by
  rw [h2, h3]
  simp
  sorry

end original_average_l233_233913


namespace equidistant_point_quadrants_l233_233505

theorem equidistant_point_quadrants :
  ∀ (x y : ℝ), 3 * x + 5 * y = 15 → (|x| = |y| → (x > 0 → y > 0 ∧ x = y ∧ y = x) ∧ (x < 0 → y > 0 ∧ x = -y ∧ -x = y)) := 
by
  sorry

end equidistant_point_quadrants_l233_233505


namespace last_three_digits_7_pow_105_l233_233190

theorem last_three_digits_7_pow_105 : (7^105) % 1000 = 783 :=
  sorry

end last_three_digits_7_pow_105_l233_233190


namespace sequence_terms_are_integers_l233_233633

theorem sequence_terms_are_integers (a : ℕ → ℕ)
  (h0 : a 0 = 1) 
  (h1 : a 1 = 2) 
  (h_recurrence : ∀ n : ℕ, (n + 3) * a (n + 2) = (6 * n + 9) * a (n + 1) - n * a n) :
  ∀ n : ℕ, ∃ k : ℤ, a n = k := 
by
  -- Initialize the proof
  sorry

end sequence_terms_are_integers_l233_233633


namespace P_is_in_third_quadrant_l233_233346

noncomputable def point : Type := (ℝ × ℝ)

def P : point := (-3, -4)

def is_in_third_quadrant (p : point) : Prop :=
  p.1 < 0 ∧ p.2 < 0

theorem P_is_in_third_quadrant : is_in_third_quadrant P :=
by {
  -- Prove that P is in the third quadrant
  sorry
}

end P_is_in_third_quadrant_l233_233346


namespace triangle_side_length_difference_l233_233752

theorem triangle_side_length_difference (x : ℤ) :
  (2 < x ∧ x < 16) → (∀ y : ℤ, (2 < y ∧ y < 16) → (3 ≤ y) ∧ (y ≤ 15)) →
  (∀ z : ℤ, (3 ≤ z ∨ z ≤ 15) → (15 - 3 = 12)) := by
  sorry

end triangle_side_length_difference_l233_233752


namespace village_population_rate_decrease_l233_233988

/--
Village X has a population of 78,000, which is decreasing at a certain rate \( R \) per year.
Village Y has a population of 42,000, which is increasing at the rate of 800 per year.
In 18 years, the population of the two villages will be equal.
We aim to prove that the rate of decrease in population per year for Village X is 1200.
-/
theorem village_population_rate_decrease (R : ℝ) 
  (hx : 78000 - 18 * R = 42000 + 18 * 800) : 
  R = 1200 :=
by
  sorry

end village_population_rate_decrease_l233_233988


namespace correct_calculation_l233_233653

-- Definitions of the conditions
def condition1 : Prop := 3 + Real.sqrt 3 ≠ 3 * Real.sqrt 3
def condition2 : Prop := 2 * Real.sqrt 3 + Real.sqrt 3 = 3 * Real.sqrt 3
def condition3 : Prop := 2 * Real.sqrt 3 - Real.sqrt 3 ≠ 2
def condition4 : Prop := Real.sqrt 3 + Real.sqrt 2 ≠ Real.sqrt 5

-- Proposition using the conditions to state the correct calculation
theorem correct_calculation (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) : 
  2 * Real.sqrt 3 + Real.sqrt 3 = 3 * Real.sqrt 3 :=
by
  exact h2

end correct_calculation_l233_233653


namespace logan_television_hours_l233_233304

-- Definitions
def minutes_in_an_hour : ℕ := 60
def logan_minutes_watched : ℕ := 300
def logan_hours_watched : ℕ := logan_minutes_watched / minutes_in_an_hour

-- Theorem statement
theorem logan_television_hours : logan_hours_watched = 5 := by
  sorry

end logan_television_hours_l233_233304


namespace roots_formula_l233_233826

theorem roots_formula (x₁ x₂ p : ℝ)
  (h₁ : x₁ + x₂ = 6 * p)
  (h₂ : x₁ * x₂ = p^2)
  (h₃ : ∀ x, x ^ 2 - 6 * p * x + p ^ 2 = 0 → x = x₁ ∨ x = x₂) :
  (1 / (x₁ + p) + 1 / (x₂ + p) = 1 / p) :=
by
  sorry

end roots_formula_l233_233826


namespace solve_equation_l233_233433

theorem solve_equation : ∀ x : ℝ, (10 - x) ^ 2 = 4 * x ^ 2 ↔ x = 10 / 3 ∨ x = -10 :=
by
  intros x
  sorry

end solve_equation_l233_233433


namespace Carlson_max_jars_l233_233167

theorem Carlson_max_jars (n a : ℕ) (hn : 13 * n = 5 * (8 * n + 9 * a)) : ∃ k : ℕ, k ≤ 23 := 
sorry

end Carlson_max_jars_l233_233167


namespace valid_ways_to_assign_volunteers_l233_233329

noncomputable def validAssignments : ℕ := 
  (Nat.choose 5 2) * (Nat.choose 3 2) + (Nat.choose 5 1) * (Nat.choose 4 2)

theorem valid_ways_to_assign_volunteers : validAssignments = 60 := 
  by
    simp [validAssignments]
    sorry

end valid_ways_to_assign_volunteers_l233_233329


namespace height_difference_l233_233363

theorem height_difference
  (a b : ℝ)
  (parabola_eq : ∀ x, y = x^2 + 1)
  (circle_center : b = 2 * a^2 + 1 / 2) :
  b - (a^2 + 1) = a^2 - 1 / 2 :=
by {
  sorry
}

end height_difference_l233_233363


namespace find_S2017_l233_233250

-- Setting up the given conditions and sequences
def a1 : ℤ := -2014
def S (n : ℕ) : ℚ := n * a1 + (n * (n - 1) / 2) * 2 -- Using the provided sum formula

theorem find_S2017
  (h1 : a1 = -2014)
  (h2 : (S 2014) / 2014 - (S 2008) / 2008 = 6) :
  S 2017 = 4034 := 
sorry

end find_S2017_l233_233250


namespace number_of_tangent_small_circles_l233_233813

-- Definitions from the conditions
def central_radius : ℝ := 2
def small_radius : ℝ := 1

-- The proof problem statement
theorem number_of_tangent_small_circles : 
  ∃ n : ℕ, (∀ i j k : ℕ, i ≠ j ∧ j ≠ k ∧ k ≠ i → 
    dist (3 * central_radius) (3 * small_radius) = 3) ∧ n = 3 :=
by
  sorry

end number_of_tangent_small_circles_l233_233813


namespace geoff_total_spending_l233_233688

def price_day1 : ℕ := 60
def pairs_day1 : ℕ := 2
def price_per_pair_day1 : ℕ := price_day1 / pairs_day1

def multiplier_day2 : ℕ := 3
def price_per_pair_day2 : ℕ := price_per_pair_day1 * 3 / 2
def discount_day2 : Real := 0.10
def cost_before_discount_day2 : ℕ := multiplier_day2 * price_per_pair_day2
def cost_after_discount_day2 : Real := cost_before_discount_day2 * (1 - discount_day2)

def multiplier_day3 : ℕ := 5
def price_per_pair_day3 : ℕ := price_per_pair_day1 * 2
def sales_tax_day3 : Real := 0.08
def cost_before_tax_day3 : ℕ := multiplier_day3 * price_per_pair_day3
def cost_after_tax_day3 : Real := cost_before_tax_day3 * (1 + sales_tax_day3)

def total_cost : Real := price_day1 + cost_after_discount_day2 + cost_after_tax_day3

theorem geoff_total_spending : total_cost = 505.50 := by
  sorry

end geoff_total_spending_l233_233688


namespace age_of_youngest_child_l233_233143

theorem age_of_youngest_child (x : ℕ) (h : x + (x + 3) + (x + 6) + (x + 9) + (x + 12) = 70) : x = 8 :=
by
  sorry

end age_of_youngest_child_l233_233143


namespace ball_maximum_height_l233_233458
-- Import necessary libraries

-- Define the height function
def ball_height (t : ℝ) : ℝ := -5 * t^2 + 50 * t + 20

-- Proposition asserting that the maximum height of the ball is 145 meters
theorem ball_maximum_height : ∃ t : ℝ, ball_height t = 145 :=
  sorry

end ball_maximum_height_l233_233458


namespace quadratic_to_vertex_form_l233_233785

theorem quadratic_to_vertex_form : ∃ m n : ℝ, (∀ x : ℝ, x^2 - 8*x + 3 = 0 ↔ (x - m)^2 = n) ∧ m + n = 17 :=
by sorry

end quadratic_to_vertex_form_l233_233785


namespace solve_quadratic_1_solve_quadratic_2_l233_233728

-- 1. Prove that the solutions to the equation x^2 - 4x - 1 = 0 are x = 2 + sqrt(5) and x = 2 - sqrt(5)
theorem solve_quadratic_1 (x : ℝ) : x ^ 2 - 4 * x - 1 = 0 ↔ x = 2 + Real.sqrt 5 ∨ x = 2 - Real.sqrt 5 :=
sorry

-- 2. Prove that the solutions to the equation 3(x - 1)^2 = 2(x - 1) are x = 1 and x = 5/3
theorem solve_quadratic_2 (x : ℝ) : 3 * (x - 1) ^ 2 = 2 * (x - 1) ↔ x = 1 ∨ x = 5 / 3 :=
sorry

end solve_quadratic_1_solve_quadratic_2_l233_233728


namespace total_wheels_eq_90_l233_233226

def total_wheels (num_bicycles : Nat) (wheels_per_bicycle : Nat) (num_tricycles : Nat) (wheels_per_tricycle : Nat) :=
  num_bicycles * wheels_per_bicycle + num_tricycles * wheels_per_tricycle

theorem total_wheels_eq_90 : total_wheels 24 2 14 3 = 90 :=
by
  sorry

end total_wheels_eq_90_l233_233226


namespace circumscribed_sphere_surface_area_l233_233597

noncomputable def surface_area_of_circumscribed_sphere_from_volume (V : ℝ) : ℝ :=
  let s := V^(1/3 : ℝ)
  let d := s * Real.sqrt 3
  4 * Real.pi * (d / 2) ^ 2

theorem circumscribed_sphere_surface_area (V : ℝ) (h : V = 27) : surface_area_of_circumscribed_sphere_from_volume V = 27 * Real.pi :=
by
  rw [h]
  unfold surface_area_of_circumscribed_sphere_from_volume
  sorry

end circumscribed_sphere_surface_area_l233_233597


namespace prism_edges_l233_233835

theorem prism_edges (V F E n : ℕ) (h1 : V + F + E = 44) (h2 : V = 2 * n) (h3 : F = n + 2) (h4 : E = 3 * n) : E = 21 := by
  sorry

end prism_edges_l233_233835


namespace maximum_value_m_l233_233078

def f (x : ℝ) : ℝ := x^2 + 2*x + 1

noncomputable def exists_t_and_max_m (m : ℝ) : Prop :=
  ∃ t : ℝ, ∀ x : ℝ, 1 ≤ x ∧ x ≤ m → f (x + t) ≤ x

theorem maximum_value_m : ∃ m : ℝ, exists_t_and_max_m m ∧ (∀ m' : ℝ, exists_t_and_max_m m' → m' ≤ 4) :=
by
  sorry

end maximum_value_m_l233_233078


namespace triangle_B_is_right_triangle_l233_233565

theorem triangle_B_is_right_triangle :
  let a := 1
  let b := 2
  let c := Real.sqrt 3
  a^2 + c^2 = b^2 :=
by
  sorry

end triangle_B_is_right_triangle_l233_233565


namespace Lyle_friends_sandwich_juice_l233_233775

/-- 
Lyle wants to buy himself and his friends a sandwich and a pack of juice. 
A sandwich costs $0.30 while a pack of juice costs $0.20. Given Lyle has $2.50, 
prove that he can buy sandwiches and juice for 4 of his friends.
-/
theorem Lyle_friends_sandwich_juice :
  let sandwich_cost := 0.30
  let juice_cost := 0.20
  let total_money := 2.50
  let total_cost_one_set := sandwich_cost + juice_cost
  let total_sets := total_money / total_cost_one_set
  total_sets - 1 = 4 :=
by
  sorry

end Lyle_friends_sandwich_juice_l233_233775


namespace hypotenuse_length_l233_233268

open Real

-- Definitions corresponding to the conditions
def right_triangle_vertex_length (ADC_length : ℝ) (AEC_length : ℝ) (x : ℝ) : Prop :=
  0 < x ∧ x < π / 2 ∧ ADC_length = sqrt 3 * sin x ∧ AEC_length = sin x

def trisect_hypotenuse (BD : ℝ) (DE : ℝ) (EC : ℝ) (c : ℝ) : Prop :=
  BD = c / 3 ∧ DE = c / 3 ∧ EC = c / 3

-- Main theorem definition
theorem hypotenuse_length (x hypotenuse ADC_length AEC_length : ℝ) :
  right_triangle_vertex_length ADC_length AEC_length x →
  trisect_hypotenuse (hypotenuse / 3) (hypotenuse / 3) (hypotenuse / 3) hypotenuse →
  hypotenuse = sqrt 3 * sin x :=
by
  intros h₁ h₂
  sorry

end hypotenuse_length_l233_233268


namespace pythagorean_theorem_sets_l233_233699

theorem pythagorean_theorem_sets :
  ¬ (4 ^ 2 + 5 ^ 2 = 6 ^ 2) ∧
  (1 ^ 2 + (Real.sqrt 3) ^ 2 = 2 ^ 2) ∧
  ¬ (5 ^ 2 + 6 ^ 2 = 7 ^ 2) ∧
  ¬ (1 ^ 2 + (Real.sqrt 2) ^ 2 = 3 ^ 2) :=
by {
  sorry
}

end pythagorean_theorem_sets_l233_233699


namespace sum_six_seven_l233_233377

variable (a : ℕ → ℝ)
variable (d : ℝ)

axiom arithmetic_sequence : ∀ (n : ℕ), a (n + 1) = a n + d
axiom sum_condition : a 2 + a 5 + a 8 + a 11 = 48

theorem sum_six_seven : a 6 + a 7 = 24 :=
by
  -- Using given axioms and properties of arithmetic sequence
  sorry

end sum_six_seven_l233_233377


namespace obtuse_vertex_angle_is_135_l233_233821

-- Define the obtuse scalene triangle with the given properties
variables {a b c : ℝ} (triangle : Triangle ℝ)
variables (φ : ℝ) (h_obtuse : φ > 90 ∧ φ < 180) (h_scalene : a ≠ b ∧ b ≠ c ∧ c ≠ a)
variables (h_side_relation : a^2 + b^2 = 2 * c^2) (h_sine_obtuse : Real.sin φ = Real.sqrt 2 / 2)

-- The measure of the obtuse vertex angle is 135 degrees
theorem obtuse_vertex_angle_is_135 :
  φ = 135 := by
  sorry

end obtuse_vertex_angle_is_135_l233_233821


namespace smallest_constant_inequality_l233_233007

open Real

theorem smallest_constant_inequality (x y z w : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w > 0) :
    sqrt (x / (y + z + w)) + sqrt (y / (x + z + w)) + sqrt (z / (x + y + w)) + sqrt (w / (x + y + z)) ≤ 2 := by
  sorry

end smallest_constant_inequality_l233_233007


namespace product_of_integers_l233_233949

theorem product_of_integers (X Y Z W : ℚ) (h_sum : X + Y + Z + W = 100)
  (h_relation : X + 5 = Y - 5 ∧ Y - 5 = 3 * Z ∧ 3 * Z = W / 3) :
  X * Y * Z * W = 29390625 / 256 := by
  sorry

end product_of_integers_l233_233949


namespace complete_set_of_events_l233_233867

-- Define the range of numbers on a die
def die_range := {n : ℕ | 1 ≤ n ∧ n ≤ 6}

-- Define what an outcome is
def outcome := { p : ℕ × ℕ | p.1 ∈ die_range ∧ p.2 ∈ die_range }

-- The theorem stating the complete set of outcomes
theorem complete_set_of_events : outcome = { p : ℕ × ℕ | 1 ≤ p.1 ∧ p.1 ≤ 6 ∧ 1 ≤ p.2 ∧ p.2 ≤ 6 } :=
by sorry

end complete_set_of_events_l233_233867


namespace same_profit_and_loss_selling_price_l233_233940

theorem same_profit_and_loss_selling_price (CP SP : ℝ) (h₁ : CP = 49) (h₂ : (CP - 42) = (SP - CP)) : SP = 56 :=
by 
  sorry

end same_profit_and_loss_selling_price_l233_233940


namespace students_move_bricks_l233_233744

variable (a b c : ℕ)

theorem students_move_bricks (h : a * b * c ≠ 0) : 
  (by let efficiency := (c : ℚ) / (a * b);
      let total_work := (a : ℚ);
      let required_time := total_work / efficiency;
      exact required_time = (a^2 * b) / (c^2)) := sorry

end students_move_bricks_l233_233744


namespace max_gcd_of_consecutive_terms_l233_233704

-- Given conditions
def a (n : ℕ) : ℕ := 2 * (n.factorial) + n

-- Theorem statement
theorem max_gcd_of_consecutive_terms : ∃ (d : ℕ), ∀ n ≥ 0, d ≤ gcd (a n) (a (n + 1)) ∧ d = 1 := by sorry

end max_gcd_of_consecutive_terms_l233_233704


namespace sin_14pi_over_5_eq_sin_36_degree_l233_233573

noncomputable def sin_14pi_over_5 : ℝ :=
  Real.sin (14 * Real.pi / 5)

noncomputable def sin_36_degree : ℝ :=
  Real.sin (36 * Real.pi / 180)

theorem sin_14pi_over_5_eq_sin_36_degree :
  sin_14pi_over_5 = sin_36_degree :=
sorry

end sin_14pi_over_5_eq_sin_36_degree_l233_233573


namespace find_expression_l233_233135

variables (x y z : ℝ) (ω : ℂ)

theorem find_expression
  (h1 : x ≠ -1) (h2 : y ≠ -1) (h3 : z ≠ -1)
  (h4 : ω^3 = 1) (h5 : ω ≠ 1)
  (h6 : (1 / (x + ω) + 1 / (y + ω) + 1 / (z + ω) = ω)) :
  1 / (x + 1) + 1 / (y + 1) + 1 / (z + 1) = -1 / 3 :=
sorry

end find_expression_l233_233135


namespace probability_of_odd_sum_given_even_product_l233_233904

-- Define a function to represent the probability of an event given the conditions
noncomputable def conditional_probability_odd_sum_even_product (dice : Fin 5 → Fin 8) : ℚ :=
  if h : (∃ i, (dice i).val % 2 = 0)  -- At least one die is even (product is even)
  then (1/2) / (31/32)  -- Probability of odd sum given even product
  else 0  -- If product is not even (not possible under conditions)

theorem probability_of_odd_sum_given_even_product :
  ∀ (dice : Fin 5 → Fin 8),
  conditional_probability_odd_sum_even_product dice = 16/31 :=
sorry  -- Proof omitted

end probability_of_odd_sum_given_even_product_l233_233904


namespace imaginary_part_of_z_l233_233118

-- Let 'z' be the complex number \(\frac {2i}{1-i}\)
noncomputable def z : ℂ := (2 * Complex.I) / (1 - Complex.I)

theorem imaginary_part_of_z :
  z.im = 1 :=
sorry

end imaginary_part_of_z_l233_233118


namespace min_value_a_plus_one_over_a_minus_one_l233_233642

theorem min_value_a_plus_one_over_a_minus_one (a : ℝ) (h : a > 1) : 
  a + 1 / (a - 1) ≥ 3 ∧ (a = 2 → a + 1 / (a - 1) = 3) :=
by
  -- Translate the mathematical proof problem into a Lean 4 theorem statement.
  sorry

end min_value_a_plus_one_over_a_minus_one_l233_233642


namespace length_of_bridge_l233_233859

/-- What is the length of a bridge (in meters), which a train 156 meters long and travelling at 45 km/h can cross in 40 seconds? -/
theorem length_of_bridge (train_length: ℕ) (train_speed_kmh: ℕ) (time_seconds: ℕ) (bridge_length: ℕ) :
  train_length = 156 →
  train_speed_kmh = 45 →
  time_seconds = 40 →
  bridge_length = 344 :=
by {
  sorry
}

end length_of_bridge_l233_233859


namespace net_rate_25_dollars_per_hour_l233_233293

noncomputable def net_rate_of_pay (hours : ℕ) (speed : ℕ) (mileage : ℕ) (rate_per_mile : ℚ) (diesel_cost_per_gallon : ℚ) : ℚ :=
  let distance := hours * speed
  let diesel_used := distance / mileage
  let earnings := rate_per_mile * distance
  let diesel_cost := diesel_cost_per_gallon * diesel_used
  let net_earnings := earnings - diesel_cost
  net_earnings / hours

theorem net_rate_25_dollars_per_hour :
  net_rate_of_pay 4 45 15 (0.75 : ℚ) (3.00 : ℚ) = 25 :=
by
  -- Proof is omitted
  sorry

end net_rate_25_dollars_per_hour_l233_233293


namespace find_n_l233_233437

-- Define that Amy bought and sold 15n avocados.
def bought_sold_avocados (n : ℕ) := 15 * n

-- Define the profit function.
def calculate_profit (n : ℕ) : ℤ := 
  let total_cost := 10 * n
  let total_earnings := 12 * n
  total_earnings - total_cost

theorem find_n (n : ℕ) (profit : ℤ) (h1 : profit = 100) (h2 : profit = calculate_profit n) : n = 50 := 
by 
  sorry

end find_n_l233_233437


namespace minimum_time_to_replace_shades_l233_233650

theorem minimum_time_to_replace_shades :
  ∀ (C : ℕ) (S : ℕ) (T : ℕ) (E : ℕ),
  ((C = 60) ∧ (S = 4) ∧ (T = 5) ∧ (E = 48)) →
  ((C * S * T) / E = 25) :=
by
  intros C S T E h
  rcases h with ⟨hC, hS, hT, hE⟩
  sorry

end minimum_time_to_replace_shades_l233_233650


namespace members_in_third_shift_l233_233036

-- Defining the given conditions
def total_first_shift : ℕ := 60
def percent_first_shift_pension : ℝ := 0.20

def total_second_shift : ℕ := 50
def percent_second_shift_pension : ℝ := 0.40

variable (T : ℕ)
def percent_third_shift_pension : ℝ := 0.10

def percent_total_pension_program : ℝ := 0.24

noncomputable def number_of_members_third_shift : ℕ :=
  T

-- Using the conditions to declare the theorem
theorem members_in_third_shift :
  ((60 * 0.20) + (50 * 0.40) + (number_of_members_third_shift T * percent_third_shift_pension)) / (60 + 50 + number_of_members_third_shift T) = percent_total_pension_program →
  number_of_members_third_shift T = 40 :=
sorry

end members_in_third_shift_l233_233036


namespace power_division_identity_l233_233667

theorem power_division_identity : (8 ^ 15) / (64 ^ 6) = 512 := by
  have h64 : 64 = 8 ^ 2 := by
    sorry
  have h_exp_rule : ∀ (a m n : ℕ), (a ^ m) ^ n = a ^ (m * n) := by
    sorry
  
  rw [h64]
  rw [h_exp_rule]
  sorry

end power_division_identity_l233_233667


namespace distribute_books_into_bags_l233_233342

def number_of_ways_to_distribute_books (books : Finset ℕ) (bags : ℕ) : ℕ :=
  if (books.card = 5) ∧ (bags = 3) then 51 else 0

theorem distribute_books_into_bags :
  number_of_ways_to_distribute_books (Finset.range 5) 3 = 51 := by
  sorry

end distribute_books_into_bags_l233_233342


namespace xyz_values_l233_233323

theorem xyz_values (x y z : ℝ)
  (h1 : x * y - 5 * y = 20)
  (h2 : y * z - 5 * z = 20)
  (h3 : z * x - 5 * x = 20) :
  x * y * z = 340 ∨ x * y * z = -62.5 := 
by sorry

end xyz_values_l233_233323


namespace find_digits_l233_233030

theorem find_digits (A B C D : ℕ) (h1 : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
(h2 : 1 ≤ A ∧ A ≤ 9)
(h3 : 1 ≤ B ∧ B ≤ 9)
(h4 : 1 ≤ C ∧ C ≤ 9)
(h5 : 1 ≤ D ∧ D ≤ 9)
(h6 : (10 * A + B) * (10 * C + B) = 111 * D)
(h7 : (10 * A + B) < (10 * C + B)) :
A = 2 ∧ B = 7 ∧ C = 3 ∧ D = 9 :=
sorry

end find_digits_l233_233030


namespace total_children_on_playground_l233_233609

theorem total_children_on_playground (boys girls : ℕ) (hb : boys = 27) (hg : girls = 35) : boys + girls = 62 :=
  by
  -- Proof goes here
  sorry

end total_children_on_playground_l233_233609


namespace range_of_a_l233_233488

variables {f : ℝ → ℝ} (a : ℝ)

-- Even function definition
def even_function (f : ℝ → ℝ) : Prop :=
∀ x, f x = f (-x)

-- Monotonically increasing on (-∞, 0)
def mono_increasing_on_neg (f : ℝ → ℝ) : Prop :=
∀ x y, x < y → y < 0 → f x ≤ f y

-- Problem statement
theorem range_of_a
  (h_even : even_function f)
  (h_mono_neg : mono_increasing_on_neg f)
  (h_inequality : f (2 ^ |a - 1|) > f 4) :
  -1 < a ∧ a < 3 :=
sorry

end range_of_a_l233_233488


namespace truth_values_of_p_and_q_l233_233332

variable (p q : Prop)

theorem truth_values_of_p_and_q
  (h1 : ¬ (p ∧ q))
  (h2 : (¬ p ∨ q)) :
  ¬ p ∧ (q ∨ ¬ q) :=
by {
  sorry
}

end truth_values_of_p_and_q_l233_233332


namespace sum_of_coordinates_B_l233_233817

theorem sum_of_coordinates_B 
  (M : ℝ × ℝ)
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)
  (hM_def : M = (-3, 2))
  (hA_def : A = (-8, 5))
  (hM_midpoint : M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) :
  B.1 + B.2 = 1 := 
sorry

end sum_of_coordinates_B_l233_233817


namespace even_func_min_value_l233_233981

theorem even_func_min_value (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)
  (h_neq_a : a ≠ 1) (h_neq_b : b ≠ 1) (h_even : ∀ x : ℝ, a^x + b^x = a^(-x) + b^(-x)) :
  ab = 1 → (∃ y : ℝ, y = (1 / a + 4 / b) ∧ y = 4) :=
by
  sorry

end even_func_min_value_l233_233981


namespace ticket_1000_wins_probability_l233_233774

-- Define the total number of tickets
def n_tickets := 1000

-- Define the number of odd tickets
def n_odd_tickets := 500

-- Define the number of relevant tickets (ticket 1000 + odd tickets)
def n_relevant_tickets := 501

-- Define the probability that ticket number 1000 wins a prize
def win_probability : ℚ := 1 / n_relevant_tickets

-- State the theorem
theorem ticket_1000_wins_probability : win_probability = 1 / 501 :=
by
  -- The proof would go here
  sorry

end ticket_1000_wins_probability_l233_233774


namespace broadcasting_methods_count_l233_233871

-- Defining the given conditions
def num_commercials : ℕ := 4 -- number of different commercial advertisements
def num_psa : ℕ := 2 -- number of different public service advertisements
def total_slots : ℕ := 6 -- total number of slots for commercials

-- The assertion we want to prove
theorem broadcasting_methods_count : 
  (num_psa * (total_slots - num_commercials - 1) * (num_commercials.factorial)) = 48 :=
by sorry

end broadcasting_methods_count_l233_233871


namespace polynomial_divisibility_l233_233555

theorem polynomial_divisibility (m : ℤ) : (4 * m + 5) ^ 2 - 9 ∣ 8 := by
  sorry

end polynomial_divisibility_l233_233555


namespace minimum_value_f_on_neg_ab_l233_233521

theorem minimum_value_f_on_neg_ab
  (f : ℝ → ℝ)
  (a b : ℝ)
  (h1 : a < b)
  (h2 : b < 0)
  (odd_f : ∀ x : ℝ, f (-x) = -f (x))
  (decreasing_f : ∀ x y : ℝ, 0 < x ∧ x < y → f y < f x)
  (range_ab : ∀ y : ℝ, a ≤ y ∧ y ≤ b → -3 ≤ f y ∧ f y ≤ 4) :
  ∀ x : ℝ, -b ≤ x ∧ x ≤ -a → -4 ≤ f x ∧ f x ≤ 3 := 
sorry

end minimum_value_f_on_neg_ab_l233_233521


namespace condition_1_condition_2_l233_233244

theorem condition_1 (m : ℝ) : (m^2 - 2*m - 15 > 0) ↔ (m < -3 ∨ m > 5) :=
sorry

theorem condition_2 (m : ℝ) : (2*m^2 + 3*m - 9 = 0) ∧ (7*m + 21 ≠ 0) ↔ (m = 3/2) :=
sorry

end condition_1_condition_2_l233_233244


namespace function_has_one_root_l233_233558

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x - 2

theorem function_has_one_root : ∃! x : ℝ, f x = 0 :=
by
  -- Indicate that we haven't included the proof
  sorry

end function_has_one_root_l233_233558


namespace sum_of_three_consecutive_odd_integers_l233_233459

-- Define the variables and conditions
variables (a : ℤ) (h1 : (a + (a + 4) = 100))

-- Define the statement that needs to be proved
theorem sum_of_three_consecutive_odd_integers (ha : a = 48) : a + (a + 2) + (a + 4) = 150 := by
  sorry

end sum_of_three_consecutive_odd_integers_l233_233459


namespace winning_strategy_for_A_winning_strategy_for_B_no_winning_strategy_l233_233745

def game (n : ℕ) : Prop :=
  ∃ A_winning_strategy B_winning_strategy neither_winning_strategy,
    (n ≥ 8 → A_winning_strategy) ∧
    (n ≤ 5 → B_winning_strategy) ∧
    (n = 6 ∨ n = 7 → neither_winning_strategy)

theorem winning_strategy_for_A (n : ℕ) (h : n ≥ 8) :
  game n :=
sorry

theorem winning_strategy_for_B (n : ℕ) (h : n ≤ 5) :
  game n :=
sorry

theorem no_winning_strategy (n : ℕ) (h : n = 6 ∨ n = 7) :
  game n :=
sorry

end winning_strategy_for_A_winning_strategy_for_B_no_winning_strategy_l233_233745


namespace simplify_fraction_l233_233761

theorem simplify_fraction : (3 ^ 2016 - 3 ^ 2014) / (3 ^ 2016 + 3 ^ 2014) = 4 / 5 :=
by
  sorry

end simplify_fraction_l233_233761


namespace increased_volume_l233_233359

theorem increased_volume (l w h : ℕ) 
  (volume_eq : l * w * h = 4500) 
  (surface_area_eq : l * w + l * h + w * h = 900) 
  (edges_sum_eq : l + w + h = 54) :
  (l + 1) * (w + 1) * (h + 1) = 5455 := 
by 
  sorry

end increased_volume_l233_233359


namespace polygon_has_twelve_sides_l233_233674

theorem polygon_has_twelve_sides
  (sum_exterior_angles : ℝ)
  (sum_interior_angles : ℝ → ℝ)
  (n : ℝ)
  (h1 : sum_exterior_angles = 360)
  (h2 : ∀ n, sum_interior_angles n = 180 * (n - 2))
  (h3 : ∀ n, sum_interior_angles n = 5 * sum_exterior_angles) :
  n = 12 :=
by
  sorry

end polygon_has_twelve_sides_l233_233674


namespace sum_geq_three_implies_one_geq_two_l233_233484

theorem sum_geq_three_implies_one_geq_two (a b : ℕ) (h : a + b ≥ 3) : a ≥ 2 ∨ b ≥ 2 :=
by { sorry }

end sum_geq_three_implies_one_geq_two_l233_233484


namespace second_negative_integer_l233_233616

theorem second_negative_integer (n : ℤ) (h : -11 * n + 5 = 93) : n = -8 :=
by
  sorry

end second_negative_integer_l233_233616


namespace computation_l233_233948

theorem computation :
  52 * 46 + 104 * 52 = 7800 := by
  sorry

end computation_l233_233948


namespace abs_linear_combination_l233_233511

theorem abs_linear_combination (a b : ℝ) :
  (∀ x y : ℝ, |a * x + b * y| + |b * x + a * y| = |x| + |y|) →
  (a = 1 ∧ b = 0) ∨ (a = 0 ∧ b = 1) ∨ (a = 0 ∧ b = -1) ∨ (a = -1 ∧ b = 0) :=
by {
  sorry
}

end abs_linear_combination_l233_233511


namespace perimeter_of_photo_l233_233357

theorem perimeter_of_photo 
  (frame_width : ℕ)
  (frame_area : ℕ)
  (outer_edge_length : ℕ)
  (photo_perimeter : ℕ) :
  frame_width = 2 → 
  frame_area = 48 → 
  outer_edge_length = 10 →
  photo_perimeter = 16 :=
by
  intros h1 h2 h3
  -- proof goes here
  sorry

end perimeter_of_photo_l233_233357


namespace time_to_fill_tank_with_leak_l233_233302

-- Definitions based on the given conditions:
def rate_of_pipe_A := 1 / 6 -- Pipe A fills the tank in 6 hours
def rate_of_leak := 1 / 12 -- The leak empties the tank in 12 hours
def combined_rate := rate_of_pipe_A - rate_of_leak -- Combined rate with leak

-- The proof problem: Prove the time taken to fill the tank with the leak present is 12 hours.
theorem time_to_fill_tank_with_leak : 
  (1 / combined_rate) = 12 := by
    -- Proof goes here...
    sorry

end time_to_fill_tank_with_leak_l233_233302


namespace cyclic_sum_inequality_l233_233796

theorem cyclic_sum_inequality (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b * c = 1) :
    (ab / (ab + a^5 + b^5)) + (bc / (bc + b^5 + c^5)) + (ca / (ca + c^5 + a^5)) ≤ 1 := by
  sorry

end cyclic_sum_inequality_l233_233796


namespace parallelogram_side_lengths_l233_233572

theorem parallelogram_side_lengths (x y : ℝ) (h1 : 3 * x + 6 = 12) (h2 : 5 * y - 2 = 10) : x + y = 22 / 5 :=
by
  sorry

end parallelogram_side_lengths_l233_233572


namespace find_angle_C_find_side_c_l233_233314

noncomputable def triangle_angle_C (a b c : ℝ) (C : ℝ) (A : ℝ) : Prop := 
a * Real.cos C = c * Real.sin A

theorem find_angle_C (a b c : ℝ) (C : ℝ) (A : ℝ)
  (h1 : triangle_angle_C a b c C A)
  (h2 : 0 < A) : C = Real.pi / 3 := 
sorry

noncomputable def triangle_side_c (a b c : ℝ) (C : ℝ) : Prop := 
(∃ (area : ℝ), area = 6 ∧ b = 4 ∧ c * c = a * a + b * b - 2 * a * b * Real.cos C)

theorem find_side_c (a b c : ℝ) (C : ℝ) 
  (h1 : triangle_side_c a b c C) : c = 2 * Real.sqrt 7 :=
sorry

end find_angle_C_find_side_c_l233_233314


namespace smaller_angle_at_9_15_l233_233500

theorem smaller_angle_at_9_15 (h_degree : ℝ) (m_degree : ℝ) (smaller_angle : ℝ) :
  (h_degree = 277.5) → (m_degree = 90) → (smaller_angle = 172.5) :=
by
  sorry

end smaller_angle_at_9_15_l233_233500


namespace michelle_sandwiches_l233_233576

def sandwiches_left (total : ℕ) (given_to_coworker : ℕ) (kept : ℕ) : ℕ :=
  total - given_to_coworker - kept

theorem michelle_sandwiches : sandwiches_left 20 4 (4 * 2) = 8 :=
by
  sorry

end michelle_sandwiches_l233_233576


namespace max_candy_received_l233_233468

theorem max_candy_received (students : ℕ) (candies : ℕ) (min_candy_per_student : ℕ) 
    (h_students : students = 40) (h_candies : candies = 200) (h_min_candy : min_candy_per_student = 2) :
    ∃ max_candy : ℕ, max_candy = 122 := by
  sorry

end max_candy_received_l233_233468


namespace percent_university_diploma_no_job_choice_l233_233291

theorem percent_university_diploma_no_job_choice
    (total_people : ℕ)
    (P1 : 10 * total_people / 100 = total_people / 10)
    (P2 : 20 * total_people / 100 = total_people / 5)
    (P3 : 30 * total_people / 100 = 3 * total_people / 10) :
  25 = (20 * total_people / (80 * total_people / 100)) :=
by
  sorry

end percent_university_diploma_no_job_choice_l233_233291


namespace seashells_in_six_weeks_l233_233822

def jar_weekly_update (week : Nat) (jarA : Nat) (jarB : Nat) : Nat × Nat :=
  if week % 3 = 0 then (jarA / 2, jarB / 2)
  else (jarA + 20, jarB * 2)

def total_seashells_after_weeks (initialA : Nat) (initialB : Nat) (weeks : Nat) : Nat :=
  let rec update (w : Nat) (jA : Nat) (jB : Nat) :=
    match w with
    | 0 => jA + jB
    | n + 1 =>
      let (newA, newB) := jar_weekly_update n jA jB
      update n newA newB
  update weeks initialA initialB

theorem seashells_in_six_weeks :
  total_seashells_after_weeks 50 30 6 = 97 :=
sorry

end seashells_in_six_weeks_l233_233822


namespace hike_distance_l233_233472

theorem hike_distance :
  let total_distance := 0.7
  let car_to_stream := 0.2
  let meadow_to_campsite := 0.1
  let stream_to_meadow := total_distance - (car_to_stream + meadow_to_campsite)
  stream_to_meadow = 0.4 :=
by
  let total_distance := 0.7
  let car_to_stream := 0.2
  let meadow_to_campsite := 0.1
  let stream_to_meadow := total_distance - (car_to_stream + meadow_to_campsite)
  show stream_to_meadow = 0.4
  sorry

end hike_distance_l233_233472


namespace diaz_age_twenty_years_later_l233_233605

theorem diaz_age_twenty_years_later (D S : ℕ) (h₁ : 10 * D - 40 = 10 * S + 20) (h₂ : S = 30) : D + 20 = 56 :=
sorry

end diaz_age_twenty_years_later_l233_233605


namespace problem_part_1_problem_part_2_l233_233034

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (Real.sin x, 3 / 2)
noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (Real.cos x, -1)

noncomputable def tan_2x_when_parallel (x : ℝ) (h : (Real.sin x + 3 / 2 * Real.cos x = 0)) : Prop :=
    Real.tan (2 * x) = 12 / 5

noncomputable def f (x : ℝ) : ℝ :=
  let a := vector_a x
  let b := vector_b x
  (a.1 + b.1) * b.1 + (a.2 + b.2) * b.2

def range_f_on_interval : Prop :=
  ∀ x ∈ Set.Icc (-Real.pi / 2) 0, -Real.sqrt 2 / 2 ≤ f x ∧ f x ≤ 1 / 2

theorem problem_part_1 (x : ℝ) (h : (Real.sin x + 3 / 2 * Real.cos x = 0)) : Real.tan (2 * x) = 12 / 5 :=
by
  sorry

theorem problem_part_2 : range_f_on_interval :=
by
  sorry

end problem_part_1_problem_part_2_l233_233034


namespace total_gain_is_19200_l233_233105

noncomputable def total_annual_gain_of_partnership (x : ℝ) (A_share : ℝ) (B_investment_after : ℕ) (C_investment_after : ℕ) : ℝ :=
  let A_investment_time := 12
  let B_investment_time := 12 - B_investment_after
  let C_investment_time := 12 - C_investment_after
  let proportional_sum := x * A_investment_time + 2 * x * B_investment_time + 3 * x * C_investment_time
  let individual_proportion := proportional_sum / A_investment_time
  3 * A_share

theorem total_gain_is_19200 (x A_share : ℝ) (B_investment_after C_investment_after : ℕ) :
  A_share = 6400 →
  B_investment_after = 6 →
  C_investment_after = 8 →
  total_annual_gain_of_partnership x A_share B_investment_after C_investment_after = 19200 :=
by
  intros hA hB hC
  have x_pos : x > 0 := by sorry   -- Additional assumptions if required
  have A_share_pos : A_share > 0 := by sorry -- Additional assumptions if required
  sorry

end total_gain_is_19200_l233_233105


namespace arcsin_arccos_interval_l233_233138

open Real
open Set

theorem arcsin_arccos_interval (x y : ℝ) (h : x^2 + y^2 = 1) : 
  ∃ t ∈ Icc (-3 * π / 2) (π / 2), 2 * arcsin x - arccos y = t := 
sorry

end arcsin_arccos_interval_l233_233138


namespace aldehyde_formula_l233_233038

-- Define the problem starting with necessary variables
variables (n : ℕ)

-- Given conditions
def general_formula_aldehyde (n : ℕ) : String :=
  "CₙH_{2n}O"

def mass_percent_hydrogen (n : ℕ) : ℚ :=
  (2 * n) / (14 * n + 16)

-- Given the percentage of hydrogen in the aldehyde
def given_hydrogen_percent : ℚ := 0.12

-- The main theorem
theorem aldehyde_formula :
  (exists n : ℕ, mass_percent_hydrogen n = given_hydrogen_percent ∧ n = 6) ->
  general_formula_aldehyde 6 = "C₆H_{12}O" :=
by
  sorry

end aldehyde_formula_l233_233038


namespace percentage_of_sum_l233_233171

theorem percentage_of_sum (x y P : ℝ) (h1 : 0.50 * (x - y) = (P / 100) * (x + y)) (h2 : y = 0.25 * x) : P = 30 :=
by
  sorry

end percentage_of_sum_l233_233171


namespace sphere_surface_area_l233_233547

theorem sphere_surface_area
  (V : ℝ)
  (r : ℝ)
  (h : ℝ)
  (R : ℝ)
  (V_cone : V = (2 * π) / 3)
  (r_cone_base : r = 1)
  (cone_height : h = 2 * V / (π * r^2))
  (sphere_radius : R^2 - (R - h)^2 = r^2):
  4 * π * R^2 = 25 * π / 4 :=
by
  sorry

end sphere_surface_area_l233_233547


namespace additional_cost_per_kg_l233_233009

theorem additional_cost_per_kg (l a : ℝ) 
  (h1 : 30 * l + 3 * a = 333) 
  (h2 : 30 * l + 6 * a = 366) 
  (h3 : 15 * l = 150) 
  : a = 11 := 
by
  sorry

end additional_cost_per_kg_l233_233009


namespace perpendicular_line_through_point_l233_233145

open Real

theorem perpendicular_line_through_point (B : ℝ × ℝ) (x y : ℝ) (c : ℝ)
  (hB : B = (3, 0)) (h_perpendicular : 2 * x + y - 5 = 0) :
  x - 2 * y + 3 = 0 :=
sorry

end perpendicular_line_through_point_l233_233145


namespace x_increase_80_percent_l233_233230

noncomputable def percentage_increase (x1 x2 : ℝ) : ℝ :=
  ((x2 / x1) - 1) * 100

theorem x_increase_80_percent
  (x1 y1 x2 y2 : ℝ)
  (h1 : x1 * y1 = x2 * y2)
  (h2 : y2 = (5 / 9) * y1) :
  percentage_increase x1 x2 = 80 :=
by
  sorry

end x_increase_80_percent_l233_233230


namespace polynomial_factorization_l233_233469

noncomputable def factorize_polynomial (a b : ℝ) : ℝ :=
  -3 * a^3 * b + 6 * a^2 * b^2 - 3 * a * b^3

theorem polynomial_factorization (a b : ℝ) : 
  factorize_polynomial a b = -3 * a * b * (a - b)^2 := 
by
  sorry

end polynomial_factorization_l233_233469


namespace merchant_gross_profit_l233_233136

theorem merchant_gross_profit :
  ∃ S : ℝ, (42 + 0.30 * S = S) ∧ ((0.80 * S) - 42 = 6) :=
by
  sorry

end merchant_gross_profit_l233_233136


namespace proof_triangle_inequality_l233_233578

noncomputable def proof_statement (a b c: ℝ) (h: a + b > c ∧ b + c > a ∧ c + a > b) : Prop :=
  a * b * c ≥ (-a + b + c) * (a - b + c) * (a + b - c)

-- Proof statement without the proof
theorem proof_triangle_inequality (a b c: ℝ) (h: a + b > c ∧ b + c > a ∧ c + a > b) : 
  proof_statement a b c h :=
sorry

end proof_triangle_inequality_l233_233578


namespace square_area_from_isosceles_triangle_l233_233289

theorem square_area_from_isosceles_triangle:
  ∀ (b h : ℝ) (Side_of_Square : ℝ), b = 2 ∧ h = 3 ∧ Side_of_Square = (6 / 5) 
  → (Side_of_Square ^ 2) = (36 / 25) := 
by
  intro b h Side_of_Square
  rintro ⟨hb, hh, h_side⟩
  sorry

end square_area_from_isosceles_triangle_l233_233289


namespace train_length_l233_233037

noncomputable def speed_km_hr : ℝ := 60
noncomputable def time_seconds : ℝ := 36
noncomputable def speed_m_s := speed_km_hr * (5/18 : ℝ)
noncomputable def distance := speed_m_s * time_seconds

-- Theorem statement
theorem train_length : distance = 600.12 := by
  sorry

end train_length_l233_233037


namespace lcm_of_two_numbers_l233_233873

theorem lcm_of_two_numbers (a b : ℕ) (h_ratio : a * 3 = b * 2) (h_sum : a + b = 30) : Nat.lcm a b = 18 :=
  sorry

end lcm_of_two_numbers_l233_233873


namespace total_strawberries_weight_is_72_l233_233580

-- Define the weights
def Marco_strawberries_weight := 19
def dad_strawberries_weight := Marco_strawberries_weight + 34 

-- The total weight of their strawberries
def total_strawberries_weight := Marco_strawberries_weight + dad_strawberries_weight

-- Prove that the total weight is 72 pounds
theorem total_strawberries_weight_is_72 : total_strawberries_weight = 72 := by
  sorry

end total_strawberries_weight_is_72_l233_233580


namespace negation_of_proposition_l233_233229

open Classical

theorem negation_of_proposition :
  (∃ x : ℝ, x^2 + 2 * x + 5 ≤ 0) ↔ ¬(∀ x : ℝ, x^2 + 2 * x + 5 > 0) := by
  sorry

end negation_of_proposition_l233_233229


namespace find_swimming_speed_l233_233183

variable (S : ℝ)

def is_average_speed (x y avg : ℝ) : Prop :=
  avg = 2 * x * y / (x + y)

theorem find_swimming_speed
  (running_speed : ℝ := 7)
  (average_speed : ℝ := 4)
  (h : is_average_speed S running_speed average_speed) :
  S = 2.8 :=
by sorry

end find_swimming_speed_l233_233183


namespace max_volume_cube_max_volume_parallelepiped_l233_233405

variables {a b c : ℝ}

-- Problem (a): Cube with the maximum volume entirely contained in the tetrahedron
theorem max_volume_cube (h : a > 0 ∧ b > 0 ∧ c > 0) :
  ∃ s : ℝ, s = (a * b * c) / (a * b + b * c + a * c) := sorry

-- Problem (b): Rectangular parallelepiped with the maximum volume entirely contained in the tetrahedron
theorem max_volume_parallelepiped (h : a > 0 ∧ b > 0 ∧ c > 0) :
  ∃ (x y z : ℝ),
  (x = a / 3 ∧ y = b / 3 ∧ z = c / 3) ∧
  (x * y * z = (a * b * c) / 27) := sorry

end max_volume_cube_max_volume_parallelepiped_l233_233405


namespace trapezoid_perimeter_l233_233872

theorem trapezoid_perimeter
  (AB CD AD BC : ℝ)
  (h1 : AB = 5)
  (h2 : CD = 5)
  (h3 : AD = 16)
  (h4 : BC = 8) :
  AB + BC + CD + AD = 34 :=
by
  sorry

end trapezoid_perimeter_l233_233872


namespace find_original_number_l233_233092

/-- The difference between a number increased by 18.7% and the same number decreased by 32.5% is 45. -/
theorem find_original_number (w : ℝ) (h : 1.187 * w - 0.675 * w = 45) : w = 45 / 0.512 :=
by
  sorry

end find_original_number_l233_233092


namespace calculate_product1_calculate_square_l233_233383

theorem calculate_product1 : 100.2 * 99.8 = 9999.96 :=
by
  sorry

theorem calculate_square : 103^2 = 10609 :=
by
  sorry

end calculate_product1_calculate_square_l233_233383


namespace paint_faces_l233_233313

def cuboid_faces : ℕ := 6
def number_of_cuboids : ℕ := 8 
def total_faces_painted : ℕ := cuboid_faces * number_of_cuboids

theorem paint_faces (h1 : cuboid_faces = 6) (h2 : number_of_cuboids = 8) : total_faces_painted = 48 := by
  -- conditions are defined above
  sorry

end paint_faces_l233_233313


namespace cameron_list_count_l233_233675

theorem cameron_list_count : 
  (∃ (n m : ℕ), n = 900 ∧ m = 27000 ∧ (∀ k : ℕ, (30 * k) ≥ n ∧ (30 * k) ≤ m → ∃ count : ℕ, count = 871)) :=
by
  sorry

end cameron_list_count_l233_233675


namespace fish_count_l233_233168

variables
  (x g s r : ℕ)
  (h1 : x - g = (2 / 3 : ℚ) * x - 1)
  (h2 : x - r = (2 / 3 : ℚ) * x + 4)
  (h3 : x = g + s + r)

theorem fish_count :
  s - g = 2 :=
by
  sorry

end fish_count_l233_233168


namespace find_f2_of_conditions_l233_233301

theorem find_f2_of_conditions (f g : ℝ → ℝ) (h_odd : ∀ x, f (-x) = -f x) 
                              (h_g : ∀ x, g x = f x + 9) 
                              (h_g_val : g (-2) = 3) : 
                              f 2 = 6 :=
by 
  sorry

end find_f2_of_conditions_l233_233301


namespace John_took_more_chickens_than_Ray_l233_233372

theorem John_took_more_chickens_than_Ray
  (r m j : ℕ)
  (h1 : r = 10)
  (h2 : r = m - 6)
  (h3 : j = m + 5) : j - r = 11 :=
by
  sorry

end John_took_more_chickens_than_Ray_l233_233372


namespace fraction_over_65_l233_233899

def num_people_under_21 := 33
def fraction_under_21 := 3 / 7
def total_people (N : ℕ) := N > 50 ∧ N < 100
def num_people (N : ℕ) := num_people_under_21 = fraction_under_21 * N

theorem fraction_over_65 (N : ℕ) : 
  total_people N → num_people N → N = 77 ∧ ∃ x, (x / 77) = x / 77 :=
by
  intro hN hnum
  sorry

end fraction_over_65_l233_233899


namespace ordered_triples_count_l233_233369

def similar_prisms_count (b : ℕ) (c : ℕ) (a : ℕ) := 
  (a ≤ c ∧ c ≤ b ∧ 
   ∃ (x y z : ℕ), x ≤ z ∧ z ≤ y ∧ y = b ∧ 
   x < a ∧ y < b ∧ z < c ∧ 
   ((x : ℚ) / a = (y : ℚ) / b ∧ (y : ℚ) / b = (z : ℚ) / c))

theorem ordered_triples_count : 
  ∃ (n : ℕ), n = 24 ∧ ∀ a c, similar_prisms_count 2000 c a → a < c :=
sorry

end ordered_triples_count_l233_233369


namespace single_reduction_equivalent_l233_233643

theorem single_reduction_equivalent (P : ℝ) (h1 : P > 0) :
  let final_price := 0.75 * P - 0.7 * (0.75 * P)
  let single_reduction := (P - final_price) / P
  single_reduction * 100 = 77.5 := 
by
  sorry

end single_reduction_equivalent_l233_233643


namespace pages_read_first_day_l233_233863

-- Alexa is reading a Nancy Drew mystery with 95 pages.
def total_pages : ℕ := 95

-- She read 58 pages the next day.
def pages_read_second_day : ℕ := 58

-- She has 19 pages left to read.
def pages_left_to_read : ℕ := 19

-- How many pages did she read on the first day?
theorem pages_read_first_day : total_pages - pages_read_second_day - pages_left_to_read = 18 := by
  -- Proof is omitted as instructed
  sorry

end pages_read_first_day_l233_233863


namespace melanie_total_dimes_l233_233994

/-- Melanie had 7 dimes in her bank. Her dad gave her 8 dimes. Her mother gave her 4 dimes. -/
def initial_dimes : ℕ := 7
def dimes_from_dad : ℕ := 8
def dimes_from_mom : ℕ := 4

/-- How many dimes does Melanie have now? -/
theorem melanie_total_dimes : initial_dimes + dimes_from_dad + dimes_from_mom = 19 := by
  sorry

end melanie_total_dimes_l233_233994


namespace time_after_3108_hours_l233_233058

/-- The current time is 3 o'clock. On a 12-hour clock, 
 what time will it be 3108 hours from now? -/
theorem time_after_3108_hours : (3 + 3108) % 12 = 3 := 
by
  sorry

end time_after_3108_hours_l233_233058


namespace sample_size_proof_l233_233534

-- Conditions
def investigate_height_of_students := "To investigate the height of junior high school students in Rui State City in early 2016, 200 students were sampled for the survey."

-- Definition of sample size based on the condition
def sample_size_condition (students_sampled : ℕ) : ℕ := students_sampled

-- Prove the sample size is 200 given the conditions
theorem sample_size_proof : sample_size_condition 200 = 200 := 
by
  sorry

end sample_size_proof_l233_233534


namespace least_number_subtracted_l233_233094

theorem least_number_subtracted (n m : ℕ) (h₁ : m = 2590) (h₂ : n = 2590 - 16) :
  (n % 9 = 6) ∧ (n % 11 = 6) ∧ (n % 13 = 6) :=
by
  sorry

end least_number_subtracted_l233_233094


namespace negation_of_existential_proposition_l233_233367

theorem negation_of_existential_proposition : 
  (¬ ∃ x : ℝ, x^2 - x + 3 = 0) = (∀ x : ℝ, x^2 - x + 3 ≠ 0) :=
by
  sorry

end negation_of_existential_proposition_l233_233367


namespace ratio_of_areas_l233_233132

noncomputable def side_length_C := 24 -- cm
noncomputable def side_length_D := 54 -- cm
noncomputable def ratio_areas := (side_length_C / side_length_D) ^ 2

theorem ratio_of_areas : ratio_areas = 16 / 81 := sorry

end ratio_of_areas_l233_233132


namespace number_of_ways_to_choose_one_top_and_one_bottom_l233_233506

theorem number_of_ways_to_choose_one_top_and_one_bottom :
  let number_of_hoodies := 5
  let number_of_sweatshirts := 4
  let number_of_jeans := 3
  let number_of_slacks := 5
  let total_tops := number_of_hoodies + number_of_sweatshirts
  let total_bottoms := number_of_jeans + number_of_slacks
  total_tops * total_bottoms = 72 := 
by
  sorry

end number_of_ways_to_choose_one_top_and_one_bottom_l233_233506


namespace equivalent_expression_l233_233176

-- Define the conditions and the statement that needs to be proven
theorem equivalent_expression (x : ℝ) (h : x^2 - 2 * x + 1 = 0) : 2 * x^2 - 4 * x = -2 := 
  by
    sorry

end equivalent_expression_l233_233176


namespace negation_of_universal_proposition_l233_233753

theorem negation_of_universal_proposition :
  ¬ (∀ x : ℝ, x^2 > 0) ↔ ∃ x : ℝ, x^2 ≤ 0 :=
sorry

end negation_of_universal_proposition_l233_233753


namespace solomon_sale_price_l233_233706

def original_price : ℝ := 500
def discount_rate : ℝ := 0.10
def sale_price := original_price * (1 - discount_rate)

theorem solomon_sale_price : sale_price = 450 := by
  sorry

end solomon_sale_price_l233_233706


namespace number_of_players_in_association_l233_233039

-- Define the variables and conditions based on the given problem
def socks_cost : ℕ := 6
def tshirt_cost := socks_cost + 8
def hat_cost := tshirt_cost - 3
def total_expenditure : ℕ := 4950
def cost_per_player := 2 * (socks_cost + tshirt_cost + hat_cost)

-- The statement to prove
theorem number_of_players_in_association :
  total_expenditure / cost_per_player = 80 := by
  sorry

end number_of_players_in_association_l233_233039


namespace james_total_cost_l233_233614

def courseCost (units: Nat) (cost_per_unit: Nat) : Nat :=
  units * cost_per_unit

def totalCostForFall : Nat :=
  courseCost 12 60 + courseCost 8 45

def totalCostForSpring : Nat :=
  let science_cost := courseCost 10 60
  let science_scholarship := science_cost / 2
  let humanities_cost := courseCost 10 45
  (science_cost - science_scholarship) + humanities_cost

def totalCostForSummer : Nat :=
  courseCost 6 80 + courseCost 4 55

def totalCostForWinter : Nat :=
  let science_cost := courseCost 6 80
  let science_scholarship := 3 * science_cost / 4
  let humanities_cost := courseCost 4 55
  (science_cost - science_scholarship) + humanities_cost

def totalAmountSpent : Nat :=
  totalCostForFall + totalCostForSpring + totalCostForSummer + totalCostForWinter

theorem james_total_cost: totalAmountSpent = 2870 :=
  by sorry

end james_total_cost_l233_233614


namespace smallest_product_bdf_l233_233809

theorem smallest_product_bdf 
  (a b c d e f : ℕ) 
  (h1 : (a + 1) * (c * e) = a * c * e + 3 * b * d * f)
  (h2 : a * (c + 1) * e = a * c * e + 4 * b * d * f)
  (h3 : a * c * (e + 1) = a * c * e + 5 * b * d * f) : 
  b * d * f = 60 := 
sorry

end smallest_product_bdf_l233_233809


namespace new_person_weight_l233_233659

theorem new_person_weight (W : ℝ) (old_weight : ℝ) (increase_per_person : ℝ) (num_persons : ℕ)
  (h1 : old_weight = 68)
  (h2 : increase_per_person = 5.5)
  (h3 : num_persons = 5)
  (h4 : W = old_weight + increase_per_person * num_persons) :
  W = 95.5 :=
by
  sorry

end new_person_weight_l233_233659


namespace line_intersects_x_axis_at_l233_233296

theorem line_intersects_x_axis_at (a b : ℝ) (h1 : a = 12) (h2 : b = 2)
  (c d : ℝ) (h3 : c = 6) (h4 : d = 6) : 
  ∃ x : ℝ, (x, 0) = (15, 0) := 
by
  -- proof needed here
  sorry

end line_intersects_x_axis_at_l233_233296


namespace eliza_ironing_hours_l233_233684

theorem eliza_ironing_hours (h : ℕ) 
  (blouse_minutes : ℕ := 15) 
  (dress_minutes : ℕ := 20) 
  (hours_ironing_blouses : ℕ := h)
  (hours_ironing_dresses : ℕ := 3)
  (total_clothes : ℕ := 17) :
  ((60 / blouse_minutes) * hours_ironing_blouses) + ((60 / dress_minutes) * hours_ironing_dresses) = total_clothes →
  hours_ironing_blouses = 2 := 
sorry

end eliza_ironing_hours_l233_233684


namespace adam_simon_distance_100_l233_233921

noncomputable def time_to_be_100_apart (x : ℝ) : Prop :=
  let distance_adam := 10 * x
  let distance_simon_east := 10 * x * (Real.sqrt 2 / 2)
  let distance_simon_south := 10 * x * (Real.sqrt 2 / 2)
  let total_eastward_separation := abs (distance_adam - distance_simon_east)
  let resultant_distance := Real.sqrt (total_eastward_separation^2 + distance_simon_south^2)
  resultant_distance = 100

theorem adam_simon_distance_100 : ∃ (x : ℝ), time_to_be_100_apart x ∧ x = 2 * Real.sqrt 2 := 
by
  sorry

end adam_simon_distance_100_l233_233921


namespace book_arrangement_count_l233_233652

-- Define the conditions
def total_books : ℕ := 6
def identical_books : ℕ := 3
def different_books : ℕ := total_books - identical_books

-- Prove the number of arrangements
theorem book_arrangement_count : (total_books.factorial / identical_books.factorial) = 120 := by
  sorry

end book_arrangement_count_l233_233652


namespace shop_weekly_earnings_l233_233788

theorem shop_weekly_earnings
  (price_women: ℕ := 18)
  (price_men: ℕ := 15)
  (time_open_hours: ℕ := 12)
  (minutes_per_hour: ℕ := 60)
  (weekly_days: ℕ := 7)
  (sell_rate_women: ℕ := 30)
  (sell_rate_men: ℕ := 40) :
  (time_open_hours * (minutes_per_hour / sell_rate_women) * price_women +
   time_open_hours * (minutes_per_hour / sell_rate_men) * price_men) * weekly_days = 4914 := 
sorry

end shop_weekly_earnings_l233_233788


namespace socks_selection_l233_233015

theorem socks_selection :
  (Nat.choose 7 3) - (Nat.choose 6 3) = 15 :=
by sorry

end socks_selection_l233_233015


namespace intersection_of_sets_l233_233748

def setA : Set ℝ := { x | x^2 - 4*x + 3 < 0 }
def setB : Set ℝ := { x | 2*x - 3 > 0 }

theorem intersection_of_sets : setA ∩ setB = { x : ℝ | x > 3/2 ∧ x < 3 } :=
  by sorry

end intersection_of_sets_l233_233748


namespace seafoam_azure_ratio_l233_233767

-- Define the conditions
variables (P S A : ℕ) 

-- Purple Valley has one-quarter as many skirts as Seafoam Valley
axiom h1 : P = S / 4

-- Azure Valley has 60 skirts
axiom h2 : A = 60

-- Purple Valley has 10 skirts
axiom h3 : P = 10

-- The goal is to prove the ratio of Seafoam Valley skirts to Azure Valley skirts is 2 to 3
theorem seafoam_azure_ratio : S / A = 2 / 3 :=
by 
  sorry

end seafoam_azure_ratio_l233_233767


namespace point_below_line_l233_233444

theorem point_below_line {a : ℝ} (h : 2 * a - 3 < 3) : a < 3 :=
by {
  sorry
}

end point_below_line_l233_233444


namespace molecular_weight_BaBr2_l233_233102

theorem molecular_weight_BaBr2 (w: ℝ) (h: w = 2376) : w / 8 = 297 :=
by
  sorry

end molecular_weight_BaBr2_l233_233102


namespace problem1_problem2_problem3_problem4_problem5_problem6_l233_233740

section
variables {a b : ℝ}

-- Problem 1
theorem problem1 (h : a + b > 0) : a^5 * b^2 + a^4 * b^3 ≥ 0 :=
sorry

-- Problem 2
theorem problem2 (h : a + b > 0) : ¬ (a^4 * b^3 + a^3 * b^4 ≥ 0) :=
sorry

-- Problem 3
theorem problem3 (h : a + b > 0) : a^21 + b^21 > 0 :=
sorry

-- Problem 4
theorem problem4 (h : a + b > 0) : (a + 2) * (b + 2) > a * b :=
sorry

-- Problem 5
theorem problem5 (h : a + b > 0) : ¬ (a - 3) * (b - 3) < a * b :=
sorry

-- Problem 6
theorem problem6 (h : a + b > 0) : ¬ (a + 2) * (b + 3) > a * b + 5 :=
sorry

end

end problem1_problem2_problem3_problem4_problem5_problem6_l233_233740


namespace range_of_independent_variable_of_sqrt_l233_233086

theorem range_of_independent_variable_of_sqrt (x : ℝ) : (2 * x - 3 ≥ 0) ↔ (x ≥ 3 / 2) := sorry

end range_of_independent_variable_of_sqrt_l233_233086


namespace milford_age_in_3_years_l233_233694

theorem milford_age_in_3_years (current_age_eustace : ℕ) (current_age_milford : ℕ) :
  (current_age_eustace = 2 * current_age_milford) → 
  (current_age_eustace + 3 = 39) → 
  current_age_milford + 3 = 21 :=
by
  intros h1 h2
  sorry

end milford_age_in_3_years_l233_233694


namespace wrapping_cube_wrapping_prism_a_wrapping_prism_b_l233_233151

theorem wrapping_cube (ways_cube : ℕ) :
  ways_cube = 3 :=
  sorry

theorem wrapping_prism_a (ways_prism_a : ℕ) (a : ℝ) :
  (ways_prism_a = 5) ↔ (a > 0) :=
  sorry

theorem wrapping_prism_b (ways_prism_b : ℕ) (b : ℝ) :
  (ways_prism_b = 7) ↔ (b > 0) :=
  sorry

end wrapping_cube_wrapping_prism_a_wrapping_prism_b_l233_233151


namespace length_AC_and_area_OAC_l233_233163

open Real EuclideanGeometry

def ellipse (x y : ℝ) : Prop :=
  x^2 + 2 * y^2 = 2

def line_1 (x y : ℝ) : Prop :=
  y = x + 1

def line_2 (B : ℝ × ℝ) (P : ℝ × ℝ) : Prop :=
  B.fst = 3 * P.fst ∧ B.snd = 3 * P.snd

theorem length_AC_and_area_OAC 
  (A C : ℝ × ℝ) 
  (B P : ℝ × ℝ) 
  (O : ℝ × ℝ := (0, 0)) 
  (h1 : ellipse A.fst A.snd) 
  (h2 : ellipse C.fst C.snd) 
  (h3 : line_1 A.fst A.snd) 
  (h4 : line_1 C.fst C.snd) 
  (h5 : line_2 B P) 
  (h6 : (P.fst = (A.fst + C.fst) / 2) ∧ (P.snd = (A.snd + C.snd) / 2)) : 
  |(dist A C)| = 4/3 * sqrt 2 ∧
  (1/2 * abs (A.fst * C.snd - C.fst * A.snd)) = 4/9 := sorry

end length_AC_and_area_OAC_l233_233163


namespace second_divisor_correct_l233_233908

noncomputable def smallest_num: Nat := 1012
def known_divisors := [12, 18, 21, 28]
def lcm_divisors: Nat := 252 -- This is the LCM of 12, 18, 21, and 28.
def result: Nat := 14

theorem second_divisor_correct :
  ∃ (d : Nat), d ≠ 12 ∧ d ≠ 18 ∧ d ≠ 21 ∧ d ≠ 28 ∧ d ≠ 252 ∧ (smallest_num - 4) % d = 0 ∧ d = result :=
by
  sorry

end second_divisor_correct_l233_233908


namespace goods_train_length_is_420_l233_233594

/-- The man's train speed in km/h. -/
def mans_train_speed_kmph : ℝ := 64

/-- The goods train speed in km/h. -/
def goods_train_speed_kmph : ℝ := 20

/-- The time taken for the trains to pass each other in seconds. -/
def passing_time_s : ℝ := 18

/-- The relative speed of two trains traveling in opposite directions in m/s. -/
noncomputable def relative_speed_mps : ℝ := 
  (mans_train_speed_kmph + goods_train_speed_kmph) * 1000 / 3600

/-- The length of the goods train in meters. -/
noncomputable def goods_train_length_m : ℝ := relative_speed_mps * passing_time_s

/-- The theorem stating the length of the goods train is 420 meters. -/
theorem goods_train_length_is_420 :
  goods_train_length_m = 420 :=
sorry

end goods_train_length_is_420_l233_233594


namespace ab_inequality_smaller_than_fourth_sum_l233_233607

theorem ab_inequality_smaller_than_fourth_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a * b) / (a + b + 2 * c) + (b * c) / (b + c + 2 * a) + (c * a) / (c + a + 2 * b) ≤ (1 / 4) * (a + b + c) := 
by
  sorry

end ab_inequality_smaller_than_fourth_sum_l233_233607


namespace amount_deducted_from_third_l233_233461

theorem amount_deducted_from_third
  (x : ℝ) 
  (h1 : ((x + (x + 1) + (x + 2) + (x + 3) + (x + 4) + (x + 5) + (x + 6) + (x + 7) + (x + 8) + (x + 9)) / 10 = 16)) 
  (h2 : (( (x - 9) + ((x + 1) - 8) + ((x + 2) - d) + (x + 3) + (x + 4) + (x + 5) + (x + 6) + (x + 7) + (x + 8) + (x + 9) ) / 10 = 11.5)) :
  d = 13.5 :=
by
  sorry

end amount_deducted_from_third_l233_233461


namespace no_int_x_divisible_by_169_l233_233431

theorem no_int_x_divisible_by_169 (x : ℤ) : ¬ (169 ∣ (x^2 + 5 * x + 16)) := by
  sorry

end no_int_x_divisible_by_169_l233_233431


namespace labourer_savings_l233_233471

theorem labourer_savings
  (monthly_expenditure_first_6_months : ℕ)
  (monthly_expenditure_next_4_months : ℕ)
  (monthly_income : ℕ)
  (total_expenditure_first_6_months : ℕ)
  (total_income_first_6_months : ℕ)
  (debt_incurred : ℕ)
  (total_expenditure_next_4_months : ℕ)
  (total_income_next_4_months : ℕ)
  (money_saved : ℕ) :
  monthly_expenditure_first_6_months = 85 →
  monthly_expenditure_next_4_months = 60 →
  monthly_income = 78 →
  total_expenditure_first_6_months = 6 * monthly_expenditure_first_6_months →
  total_income_first_6_months = 6 * monthly_income →
  debt_incurred = total_expenditure_first_6_months - total_income_first_6_months →
  total_expenditure_next_4_months = 4 * monthly_expenditure_next_4_months →
  total_income_next_4_months = 4 * monthly_income →
  money_saved = total_income_next_4_months - (total_expenditure_next_4_months + debt_incurred) →
  money_saved = 30 :=
by
  intro h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end labourer_savings_l233_233471


namespace area_tripled_radius_increase_l233_233011

theorem area_tripled_radius_increase (m r : ℝ) (h : (r + m)^2 = 3 * r^2) :
  r = m * (1 + Real.sqrt 3) / 2 :=
sorry

end area_tripled_radius_increase_l233_233011


namespace shirt_cost_l233_233836

variables (J S : ℝ)

theorem shirt_cost :
  (3 * J + 2 * S = 69) ∧
  (2 * J + 3 * S = 86) →
  S = 24 :=
by
  sorry

end shirt_cost_l233_233836


namespace complement_of_S_in_U_l233_233621

variable (U : Set ℕ)
variable (S : Set ℕ)

theorem complement_of_S_in_U (hU : U = {1, 2, 3, 4}) (hS : S = {1, 3}) : U \ S = {2, 4} := by
  sorry

end complement_of_S_in_U_l233_233621


namespace largest_possible_A_l233_233719

theorem largest_possible_A (A B C : ℕ) (h1 : 10 = A * B + C) (h2 : B = C) : A ≤ 9 :=
by sorry

end largest_possible_A_l233_233719


namespace intersection_A_B_eq_B_l233_233668

variable (a : ℝ) (A : Set ℝ) (B : Set ℝ)

def satisfies_quadratic (a : ℝ) (x : ℝ) : Prop := x^2 - a*x + 1 = 0

def set_A : Set ℝ := {1, 2, 3}

def set_B (a : ℝ) : Set ℝ := {x | satisfies_quadratic a x}

theorem intersection_A_B_eq_B (a : ℝ) (h : a ∈ set_A) : 
  (∀ x, x ∈ set_B a → x ∈ set_A) → (∃ x, x ∈ set_A ∧ satisfies_quadratic a x) →
  a = 2 :=
sorry

end intersection_A_B_eq_B_l233_233668


namespace wire_cut_problem_l233_233206

-- Conditions
variable (x y : ℝ)
variable (h1 : x = y)
variable (hx : x > 0) -- Assuming positive lengths for the wire pieces

-- Statement to prove
theorem wire_cut_problem : x / y = 1 :=
by sorry

end wire_cut_problem_l233_233206


namespace sophie_saves_money_by_using_wool_balls_l233_233420

def cost_of_dryer_sheets_per_year (loads_per_week : ℕ) (sheets_per_load : ℕ)
                                  (weeks_per_year : ℕ) (sheets_per_box : ℕ)
                                  (cost_per_box : ℝ) : ℝ :=
  let sheets_per_year := loads_per_week * sheets_per_load * weeks_per_year
  let boxes_per_year := sheets_per_year / sheets_per_box
  boxes_per_year * cost_per_box

theorem sophie_saves_money_by_using_wool_balls :
  cost_of_dryer_sheets_per_year 4 1 52 104 5.50 = 11.00 :=
by simp only [cost_of_dryer_sheets_per_year]; sorry

end sophie_saves_money_by_using_wool_balls_l233_233420


namespace sum_of_reciprocals_of_roots_l233_233274

theorem sum_of_reciprocals_of_roots (r1 r2 : ℚ) (h_sum : r1 + r2 = 17) (h_prod : r1 * r2 = 6) :
  1 / r1 + 1 / r2 = 17 / 6 :=
sorry

end sum_of_reciprocals_of_roots_l233_233274


namespace power_function_passes_through_1_1_l233_233054

theorem power_function_passes_through_1_1 (n : ℝ) : (1 : ℝ) ^ n = 1 :=
by
  -- Proof will go here
  sorry

end power_function_passes_through_1_1_l233_233054


namespace evaluate_expression_l233_233247

theorem evaluate_expression :
  (↑(2 ^ (6 / 4))) ^ 8 = 4096 :=
by sorry

end evaluate_expression_l233_233247


namespace time_to_fill_pool_l233_233060

theorem time_to_fill_pool :
  let R1 := 1
  let R2 := 1 / 2
  let R3 := 1 / 3
  let R4 := 1 / 4
  let R_total := R1 + R2 + R3 + R4
  let T := 1 / R_total
  T = 12 / 25 := 
by
  sorry

end time_to_fill_pool_l233_233060


namespace total_students_in_circle_l233_233075

theorem total_students_in_circle (N : ℕ) (h1 : ∃ (students : Finset ℕ), students.card = N)
  (h2 : ∃ (a b : ℕ), a = 6 ∧ b = 16 ∧ b - a = N / 2): N = 18 :=
by
  sorry

end total_students_in_circle_l233_233075


namespace value_of_expression_l233_233827

theorem value_of_expression (x y : ℝ) (h₁ : x = 3) (h₂ : y = 4) :
  (x^3 + 3*y^3) / 9 = 24.33 :=
by
  sorry

end value_of_expression_l233_233827


namespace geometric_seq_comparison_l233_233891

def geometric_seq_positive (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ (n : ℕ), a (n+1) = a n * q

theorem geometric_seq_comparison (a : ℕ → ℝ) (q : ℝ) (h1 : geometric_seq_positive a q) (h2 : q ≠ 1) (h3 : ∀ n, a n > 0) (h4 : q > 0) :
  a 0 + a 7 > a 3 + a 4 :=
sorry

end geometric_seq_comparison_l233_233891


namespace bruce_can_buy_11_bags_l233_233221

-- Defining the total initial amount
def initial_amount : ℕ := 200

-- Defining the quantities and prices of items
def packs_crayons   : ℕ := 5
def price_crayons   : ℕ := 5
def total_crayons   : ℕ := packs_crayons * price_crayons

def books          : ℕ := 10
def price_books    : ℕ := 5
def total_books    : ℕ := books * price_books

def calculators    : ℕ := 3
def price_calc     : ℕ := 5
def total_calc     : ℕ := calculators * price_calc

-- Total cost of all items
def total_cost : ℕ := total_crayons + total_books + total_calc

-- Calculating the change Bruce will have after buying the items
def change : ℕ := initial_amount - total_cost

-- Cost of each bag
def price_bags : ℕ := 10

-- Number of bags Bruce can buy with the change
def num_bags : ℕ := change / price_bags

-- Proposition stating the main problem
theorem bruce_can_buy_11_bags : num_bags = 11 := by
  sorry

end bruce_can_buy_11_bags_l233_233221


namespace walnut_swap_exists_l233_233762

theorem walnut_swap_exists (n : ℕ) (h_n : n = 2021) :
  ∃ k : ℕ, k < n ∧ ∃ a b : ℕ, a < k ∧ k < b :=
by
  sorry

end walnut_swap_exists_l233_233762


namespace polygon_sides_of_interior_angle_l233_233730

theorem polygon_sides_of_interior_angle (n : ℕ) (h : ∀ i : Fin n, (∃ (x : ℝ), x = (180 - 144) / 1) → (360 / (180 - 144)) = n) : n = 10 :=
sorry

end polygon_sides_of_interior_angle_l233_233730


namespace supplement_complement_l233_233071

theorem supplement_complement (angle1 angle2 : ℝ) 
  (h_complementary : angle1 + angle2 = 90) : 
   180 - angle1 = 90 + angle2 := by
  sorry

end supplement_complement_l233_233071


namespace students_walk_home_fraction_l233_233676

theorem students_walk_home_fraction :
  (1 - (3 / 8 + 2 / 5 + 1 / 8 + 5 / 100)) = (1 / 20) :=
by 
  -- The detailed proof is complex and would require converting these fractions to a common denominator,
  -- performing the arithmetic operations carefully and using Lean's rational number properties. Thus,
  -- the full detailed proof can be written with further steps, but here we insert 'sorry' to focus on the statement.
  sorry

end students_walk_home_fraction_l233_233676


namespace complex_root_seventh_power_l233_233933

theorem complex_root_seventh_power (r : ℂ) (h1 : r^7 = 1) (h2 : r ≠ 1) :
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^5 - 1) * (r^6 - 1) = 2 := by
  sorry

end complex_root_seventh_power_l233_233933


namespace total_spent_is_correct_l233_233589

def trumpet : ℝ := 149.16
def music_tool : ℝ := 9.98
def song_book : ℝ := 4.14
def trumpet_maintenance_accessories : ℝ := 21.47
def valve_oil_original : ℝ := 8.20
def valve_oil_discount_rate : ℝ := 0.20
def valve_oil_discounted : ℝ := valve_oil_original * (1 - valve_oil_discount_rate)
def band_t_shirt : ℝ := 14.95
def sales_tax_rate : ℝ := 0.065

def total_before_tax : ℝ :=
  trumpet + music_tool + song_book + trumpet_maintenance_accessories + valve_oil_discounted + band_t_shirt

def sales_tax : ℝ := total_before_tax * sales_tax_rate

def total_amount_spent : ℝ := total_before_tax + sales_tax

theorem total_spent_is_correct : total_amount_spent = 219.67 := by
  sorry

end total_spent_is_correct_l233_233589


namespace product_of_roots_l233_233162

variable {x1 x2 : ℝ}

theorem product_of_roots (h : ∀ x, -x^2 + 3*x = 0 → (x = x1 ∨ x = x2)) :
  x1 * x2 = 0 :=
by
  sorry

end product_of_roots_l233_233162


namespace largest_k_value_l233_233554

theorem largest_k_value (a b c d : ℕ) (k : ℝ)
  (h1 : a + b = c + d)
  (h2 : 2 * (a * b) = c * d)
  (h3 : a ≥ b) :
  (∀ k', (∀ a b (h1_b : a + b = c + d)
              (h2_b : 2 * a * b = c * d)
              (h3_b : a ≥ b), (a : ℝ) / (b : ℝ) ≥ k') → k' ≤ k) → k = 3 + 2 * Real.sqrt 2 :=
sorry

end largest_k_value_l233_233554


namespace circumradius_eq_l233_233089

noncomputable def circumradius (r : ℂ) (t1 t2 t3 : ℂ) : ℂ :=
  (2 * r ^ 4) / Complex.abs ((t1 + t2) * (t2 + t3) * (t3 + t1))

theorem circumradius_eq (r t1 t2 t3 : ℂ) (h_pos_r : r ≠ 0) :
  circumradius r t1 t2 t3 = (2 * r ^ 4) / Complex.abs ((t1 + t2) * (t2 + t3) * (t3 + t1)) :=
  by sorry

end circumradius_eq_l233_233089


namespace length_of_platform_l233_233606

theorem length_of_platform 
  (speed_kmph : ℕ)
  (time_cross_platform : ℕ)
  (time_cross_man : ℕ)
  (speed_mps : ℕ)
  (length_of_train : ℕ)
  (distance_platform : ℕ)
  (length_of_platform : ℕ) :
  speed_kmph = 72 →
  time_cross_platform = 30 →
  time_cross_man = 16 →
  speed_mps = speed_kmph * 1000 / 3600 →
  length_of_train = speed_mps * time_cross_man →
  distance_platform = speed_mps * time_cross_platform →
  length_of_platform = distance_platform - length_of_train →
  length_of_platform = 280 := by
  sorry

end length_of_platform_l233_233606


namespace part1_arithmetic_sequence_part2_minimum_value_Sn_l233_233501

-- Define the given conditions
variables {S : ℕ → ℝ}
variables {a : ℕ → ℝ}
variables {n : ℕ}

-- Given condition
axiom condition_1 (n : ℕ) : (2 * S n) / n + n = 2 * (a n) + 1

-- Prove that the sequence is arithmetic
theorem part1_arithmetic_sequence :
  ∀ n, (a (n+1) = a n + 1) := 
  sorry

-- Additional conditions for part 2
axiom geometric_sequence_condition (a4 a7 a9 : ℝ) : a 7 ^ 2 = a 4 * a 9
axiom a4_def : a 4 = a 1 + 3
axiom a7_def : a 7 = a 1 + 6
axiom a9_def : a 9 = a 1 + 8

-- Prove the minimum value of S_n
theorem part2_minimum_value_Sn :
  S 12 = -78 ∧ S 13 = -78 :=
  sorry

end part1_arithmetic_sequence_part2_minimum_value_Sn_l233_233501


namespace tan_ratio_l233_233864

theorem tan_ratio (x : ℝ) (h : Real.tan (x + Real.pi / 4) = 2) : 
  Real.tan x / Real.tan (2 * x) = 4 / 9 :=
sorry

end tan_ratio_l233_233864


namespace sum_mod_five_l233_233875

theorem sum_mod_five {n : ℕ} (h_pos : 0 < n) :
  (1^n + 2^n + 3^n + 4^n) % 5 = 0 ↔ ¬ (∃ k : ℕ, n = 4 * k) :=
sorry

end sum_mod_five_l233_233875


namespace possible_values_y_l233_233693

theorem possible_values_y (x : ℝ) (h : x^2 + 9 * (x / (x - 3))^2 = 90) :
  ∃ y : ℝ, (y = 0 ∨ y = 41 ∨ y = 144) ∧ y = (x - 3)^2 * (x + 4) / (2 * x - 5) :=
by sorry

end possible_values_y_l233_233693


namespace no_strategy_for_vasya_tolya_l233_233663

-- This definition encapsulates the conditions and question
def players_game (coins : ℕ) : Prop :=
  ∀ p v t : ℕ, 
    (1 ≤ p ∧ p ≤ 4) ∧ (1 ≤ v ∧ v ≤ 2) ∧ (1 ≤ t ∧ t ≤ 2) →
    (∃ (n : ℕ), coins = 5 * n)

-- Theorem formalizing the problem's conclusion
theorem no_strategy_for_vasya_tolya (n : ℕ) (h : n = 300) : 
  ¬ ∀ (v t : ℕ), 
     (1 ≤ v ∧ v ≤ 2) ∧ (1 ≤ t ∧ t ≤ 2) →
     players_game (n - v - t) :=
by
  intro h
  sorry -- Skip the proof, as it is not required

end no_strategy_for_vasya_tolya_l233_233663


namespace adam_lessons_on_monday_l233_233025

theorem adam_lessons_on_monday :
  (∃ (time_monday time_tuesday time_wednesday : ℝ) (n_monday_lessons : ℕ),
    time_tuesday = 3 ∧
    time_wednesday = 2 * time_tuesday ∧
    time_monday + time_tuesday + time_wednesday = 12 ∧
    n_monday_lessons = time_monday / 0.5 ∧
    n_monday_lessons = 6) :=
by
  sorry

end adam_lessons_on_monday_l233_233025


namespace age_of_50th_student_l233_233992

theorem age_of_50th_student (avg_50_students : ℝ) (total_students : ℕ)
                           (avg_15_students : ℝ) (group_1_count : ℕ)
                           (avg_15_students_2 : ℝ) (group_2_count : ℕ)
                           (avg_10_students : ℝ) (group_3_count : ℕ)
                           (avg_9_students : ℝ) (group_4_count : ℕ) :
                           avg_50_students = 20 → total_students = 50 →
                           avg_15_students = 18 → group_1_count = 15 →
                           avg_15_students_2 = 22 → group_2_count = 15 →
                           avg_10_students = 25 → group_3_count = 10 →
                           avg_9_students = 24 → group_4_count = 9 →
                           ∃ (age_50th_student : ℝ), age_50th_student = 66 := by
                           sorry

end age_of_50th_student_l233_233992


namespace custom_op_eval_l233_233893

-- Define the custom operation
def custom_op (a b : ℤ) : ℤ := 5 * a + 2 * b - 1

-- State the required proof problem
theorem custom_op_eval : custom_op (-4) 6 = -9 := 
by
  -- use sorry to skip the proof
  sorry

end custom_op_eval_l233_233893


namespace percentage_problem_l233_233219

theorem percentage_problem (X : ℝ) (h : 0.28 * X + 0.45 * 250 = 224.5) : X = 400 :=
sorry

end percentage_problem_l233_233219


namespace first_candidate_votes_percentage_l233_233014

theorem first_candidate_votes_percentage 
( total_votes : ℕ ) 
( second_candidate_votes : ℕ ) 
( P : ℕ ) 
( h1 : total_votes = 2400 ) 
( h2 : second_candidate_votes = 480 ) 
( h3 : (P/100 : ℝ) * total_votes + second_candidate_votes = total_votes ) : 
  P = 80 := 
sorry

end first_candidate_votes_percentage_l233_233014


namespace sum_of_reciprocals_l233_233355

theorem sum_of_reciprocals
  (m n p : ℕ)
  (HCF_mnp : Nat.gcd (Nat.gcd m n) p = 26)
  (LCM_mnp : Nat.lcm (Nat.lcm m n) p = 6930)
  (sum_mnp : m + n + p = 150) :
  (1 / (m : ℚ) + 1 / (n : ℚ) + 1 / (p : ℚ) = 1 / 320166) :=
by
  sorry

end sum_of_reciprocals_l233_233355


namespace find_t_l233_233912

-- Given a quadratic equation
def quadratic_eq (x : ℝ) := 4 * x ^ 2 - 16 * x - 200

-- Completing the square to find t
theorem find_t : ∃ q t : ℝ, (x : ℝ) → (quadratic_eq x = 0) → (x + q) ^ 2 = t ∧ t = 54 :=
by
  sorry

end find_t_l233_233912


namespace S_shaped_growth_curve_varied_growth_rate_l233_233285

theorem S_shaped_growth_curve_varied_growth_rate :
  ∀ (population_growth : ℝ → ℝ), 
    (∃ t1 t2 : ℝ, t1 < t2 ∧ 
      (∃ r : ℝ, r = population_growth t1 / t1 ∧ r ≠ population_growth t2 / t2)) 
    → 
    ∀ t3 t4 : ℝ, t3 < t4 → (population_growth t3 / t3) ≠ (population_growth t4 / t4) :=
by
  sorry

end S_shaped_growth_curve_varied_growth_rate_l233_233285


namespace range_of_a_l233_233207

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x ^ 2 + 4 * x + 1

theorem range_of_a (a : ℝ) : (∀ x : ℝ, f a (f a x) ≥ 0) ↔ a ≥ 3 :=
sorry

end range_of_a_l233_233207


namespace number_of_sons_l233_233374

noncomputable def land_area_hectares : ℕ := 3
noncomputable def hectare_to_m2 : ℕ := 10000
noncomputable def profit_per_section_per_3months : ℕ := 500
noncomputable def section_area_m2 : ℕ := 750
noncomputable def profit_per_son_per_year : ℕ := 10000
noncomputable def months_in_year : ℕ := 12
noncomputable def months_per_season : ℕ := 3

theorem number_of_sons :
  let total_land_area_m2 := land_area_hectares * hectare_to_m2
  let yearly_profit_per_section := profit_per_section_per_3months * (months_in_year / months_per_season)
  let number_of_sections := total_land_area_m2 / section_area_m2
  let total_yearly_profit := number_of_sections * yearly_profit_per_section
  let n := total_yearly_profit / profit_per_son_per_year
  n = 8 :=
by
  sorry

end number_of_sons_l233_233374


namespace fraction_of_students_getting_F_l233_233174

theorem fraction_of_students_getting_F
  (students_A students_B students_C students_D passing_fraction : ℚ) 
  (hA : students_A = 1/4)
  (hB : students_B = 1/2)
  (hC : students_C = 1/8)
  (hD : students_D = 1/12)
  (hPassing : passing_fraction = 0.875) :
  (1 - (students_A + students_B + students_C + students_D)) = 1/24 :=
by
  sorry

end fraction_of_students_getting_F_l233_233174


namespace Susan_has_10_dollars_left_l233_233128

def initial_amount : ℝ := 80
def food_expense : ℝ := 15
def rides_expense : ℝ := 3 * food_expense
def games_expense : ℝ := 10
def total_expense : ℝ := food_expense + rides_expense + games_expense
def remaining_amount : ℝ := initial_amount - total_expense

theorem Susan_has_10_dollars_left : remaining_amount = 10 := by
  sorry

end Susan_has_10_dollars_left_l233_233128


namespace total_amount_shared_l233_233042

theorem total_amount_shared (a b c : ℕ) (h_ratio : a = 3 * b / 5 ∧ c = 9 * b / 5) (h_b : b = 50) : a + b + c = 170 :=
by sorry

end total_amount_shared_l233_233042


namespace median_ratio_within_bounds_l233_233253

def median_ratio_limits (α : ℝ) (hα : 0 < α ∧ α < π) : Prop :=
  ∀ (s_c s_b : ℝ), s_b = 1 → (1 / 2) ≤ (s_c / s_b) ∧ (s_c / s_b) ≤ 2

theorem median_ratio_within_bounds (α : ℝ) (hα : 0 < α ∧ α < π) : 
  median_ratio_limits α hα :=
by
  sorry

end median_ratio_within_bounds_l233_233253


namespace task1_task2_task3_l233_233877

noncomputable def f (x a : ℝ) := x^2 - 4 * x + a + 3
noncomputable def g (x m : ℝ) := m * x + 5 - 2 * m

theorem task1 (a m : ℝ) (h₁ : a = -3) (h₂ : m = 0) :
  (∃ x : ℝ, f x a - g x m = 0) ↔ x = -1 ∨ x = 5 :=
sorry

theorem task2 (a : ℝ) :
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f x a = 0) ↔ -8 ≤ a ∧ a ≤ 0 :=
sorry

theorem task3 (m : ℝ) :
  (∀ x₁ : ℝ, 1 ≤ x₁ ∧ x₁ ≤ 4 → ∃ x₂ : ℝ, 1 ≤ x₂ ∧ x₂ ≤ 4 ∧ f x₁ 0 = g x₂ m) ↔ m ≤ -3 ∨ 6 ≤ m :=
sorry

end task1_task2_task3_l233_233877


namespace combined_time_l233_233333

def time_pulsar : ℕ := 10
def time_polly : ℕ := 3 * time_pulsar
def time_petra : ℕ := time_polly / 6

theorem combined_time : time_pulsar + time_polly + time_petra = 45 := 
by 
  -- proof steps will go here
  sorry

end combined_time_l233_233333


namespace division_remainder_is_7_l233_233651

theorem division_remainder_is_7 (d q D r : ℕ) (hd : d = 21) (hq : q = 14) (hD : D = 301) (h_eq : D = d * q + r) : r = 7 :=
by
  sorry

end division_remainder_is_7_l233_233651


namespace smallest_digit_to_make_divisible_by_9_l233_233463

theorem smallest_digit_to_make_divisible_by_9 : ∃ d : ℕ, d < 10 ∧ (5 + 2 + 8 + d + 4 + 6) % 9 = 0 ∧ ∀ d' : ℕ, d' < d → (5 + 2 + 8 + d' + 4 + 6) % 9 ≠ 0 := 
by 
  sorry

end smallest_digit_to_make_divisible_by_9_l233_233463


namespace integer_pairs_solution_l233_233182

theorem integer_pairs_solution (x y : ℤ) (k : ℤ) :
  2 * x^2 - 6 * x * y + 3 * y^2 = -1 ↔
  ∃ n : ℤ, x = (2 + Real.sqrt 3)^k / 2 ∨ x = -(2 + Real.sqrt 3)^k / 2 ∧
           y = x + (2 + Real.sqrt 3)^k / (2 * Real.sqrt 3) ∨ 
           y = x - (2 + Real.sqrt 3)^k / (2 * Real.sqrt 3) :=
sorry

end integer_pairs_solution_l233_233182


namespace kids_played_on_monday_l233_233006

theorem kids_played_on_monday (m t a : Nat) (h1 : t = 7) (h2 : a = 19) (h3 : a = m + t) : m = 12 := 
by 
  sorry

end kids_played_on_monday_l233_233006


namespace problem1_problem2_prob_dist_problem2_expectation_l233_233824

noncomputable def probability_A_wins_match_B_wins_once (pA pB : ℚ) : ℚ :=
  (pB * pA * pA) + (pA * pB * pA * pA)

theorem problem1 : probability_A_wins_match_B_wins_once (2/3) (1/3) = 20/81 :=
  by sorry

noncomputable def P_X (x : ℕ) (pA pB : ℚ) : ℚ :=
  match x with
  | 2 => pA^2 + pB^2
  | 3 => pB * pA^2 + pA * pB^2
  | 4 => (pA * pB * pA * pA) + (pB * pA * pB * pB)
  | 5 => (pB * pA * pB * pA) + (pA * pB * pA * pB)
  | _ => 0

theorem problem2_prob_dist : 
  P_X 2 (2/3) (1/3) = 5/9 ∧
  P_X 3 (2/3) (1/3) = 2/9 ∧
  P_X 4 (2/3) (1/3) = 10/81 ∧
  P_X 5 (2/3) (1/3) = 8/81 :=
  by sorry

noncomputable def E_X (pA pB : ℚ) : ℚ :=
  2 * (P_X 2 pA pB) + 3 * (P_X 3 pA pB) + 
  4 * (P_X 4 pA pB) + 5 * (P_X 5 pA pB)

theorem problem2_expectation : E_X (2/3) (1/3) = 224/81 :=
  by sorry

end problem1_problem2_prob_dist_problem2_expectation_l233_233824


namespace sequence_an_form_sum_cn_terms_l233_233983

theorem sequence_an_form (a_n S_n : ℕ → ℕ) (b_n : ℕ → ℕ) (h : ∀ n : ℕ, a_n n = 3/4 * S_n n + 2) :
  ∀ n : ℕ, b_n n = 2 * n + 1 :=
sorry 

theorem sum_cn_terms (a_n S_n : ℕ → ℕ) (b_n : ℕ → ℕ) (c_n : ℕ → ℕ) (T_n : ℕ → ℕ)
    (h : ∀ n : ℕ, a_n n = 3/4 * S_n n + 2)
    (hb : ∀ n : ℕ, b_n n = 2 * n + 1)
    (hc : ∀ n : ℕ, c_n n = 1 / (b_n n * b_n (n + 1))) :
  ∀ n : ℕ, T_n n = n / (3 * (2 * n + 3)) :=
sorry

end sequence_an_form_sum_cn_terms_l233_233983


namespace solve_trig_equation_l233_233157

open Real

theorem solve_trig_equation (x : ℝ) (n : ℤ) :
  (2 * tan (6 * x) ^ 4 + 4 * sin (4 * x) * sin (8 * x) - cos (8 * x) - cos (16 * x) + 2) / sqrt (cos x - sqrt 3 * sin x) = 0 
  ∧ cos x - sqrt 3 * sin x > 0 →
  ∃ (k : ℤ), x = 2 * π * k ∨ x = -π / 6 + 2 * π * k ∨ x = -π / 3 + 2 * π * k ∨ x = -π / 2 + 2 * π * k ∨ x = -2 * π / 3 + 2 * π * k :=
sorry

end solve_trig_equation_l233_233157


namespace train_stop_duration_l233_233263

theorem train_stop_duration (speed_without_stoppages speed_with_stoppages : ℕ) (h1 : speed_without_stoppages = 45) (h2 : speed_with_stoppages = 42) :
  ∃ t : ℕ, t = 4 :=
by
  sorry

end train_stop_duration_l233_233263


namespace bounded_expression_l233_233041

theorem bounded_expression (x y : ℝ) (h : 3 * x^2 + 2 * y^2 ≤ 6) : 2 * x + y ≤ Real.sqrt 11 :=
sorry

end bounded_expression_l233_233041


namespace prove_inequality_l233_233768

noncomputable def a : ℝ := Real.sin (33 * Real.pi / 180)
noncomputable def b : ℝ := Real.cos (55 * Real.pi / 180)
noncomputable def c : ℝ := Real.tan (55 * Real.pi / 180)

theorem prove_inequality : c > b ∧ b > a :=
by
  -- Proof goes here
  sorry

end prove_inequality_l233_233768


namespace xiaoming_age_l233_233811

theorem xiaoming_age
  (x x' : ℕ) 
  (h₁ : ∃ f : ℕ, f = 4 * x) 
  (h₂ : (x + 25) + (4 * x + 25) = 100) : 
  x = 10 :=
by
  obtain ⟨f, hf⟩ := h₁
  sorry

end xiaoming_age_l233_233811


namespace deepak_wife_speed_l233_233065

-- Definitions and conditions
def track_circumference_km : ℝ := 0.66
def deepak_speed_kmh : ℝ := 4.5
def time_to_meet_hr : ℝ := 0.08

-- Theorem statement
theorem deepak_wife_speed
  (track_circumference_km : ℝ)
  (deepak_speed_kmh : ℝ)
  (time_to_meet_hr : ℝ)
  (deepak_distance : ℝ := deepak_speed_kmh * time_to_meet_hr)
  (wife_distance : ℝ := track_circumference_km - deepak_distance)
  (wife_speed_kmh : ℝ := wife_distance / time_to_meet_hr) : 
  wife_speed_kmh = 3.75 :=
sorry

end deepak_wife_speed_l233_233065


namespace max_type_a_workers_l233_233317

theorem max_type_a_workers (x y : ℕ) (h1 : x + y = 150) (h2 : y ≥ 3 * x) : x ≤ 37 :=
sorry

end max_type_a_workers_l233_233317


namespace apples_chosen_l233_233119

def total_fruits : ℕ := 12
def bananas : ℕ := 4
def oranges : ℕ := 5
def total_other_fruits := bananas + oranges

theorem apples_chosen : total_fruits - total_other_fruits = 3 :=
by sorry

end apples_chosen_l233_233119


namespace m_div_x_l233_233516

variable (a b k : ℝ)
variable (ha : a = 4 * k)
variable (hb : b = 5 * k)
variable (k_pos : k > 0)

def x := a * 1.25
def m := b * 0.20

theorem m_div_x : m / x = 1 / 5 := by
  sorry

end m_div_x_l233_233516


namespace largest_two_digit_number_divisible_by_6_ending_in_4_l233_233487

theorem largest_two_digit_number_divisible_by_6_ending_in_4 :
  ∃ n : ℕ, (10 ≤ n ∧ n < 100) ∧ (n % 6 = 0) ∧ (n % 10 = 4) ∧ n = 84 :=
by
  existsi 84
  sorry

end largest_two_digit_number_divisible_by_6_ending_in_4_l233_233487


namespace number_of_female_democrats_l233_233252

variables (F M D_f : ℕ)

def total_participants := F + M = 660
def female_democrats := D_f = F / 2
def male_democrats := (F / 2) + (M / 4) = 220

theorem number_of_female_democrats 
  (h1 : total_participants F M) 
  (h2 : female_democrats F D_f) 
  (h3 : male_democrats F M) : 
  D_f = 110 := by
  sorry

end number_of_female_democrats_l233_233252


namespace tamika_greater_probability_l233_233969

-- Definitions for the conditions
def tamika_results : Set ℕ := {11 * 12, 11 * 13, 12 * 13}
def carlos_result : ℕ := 2 + 3 + 4

-- Theorem stating the problem
theorem tamika_greater_probability : 
  (∀ r ∈ tamika_results, r > carlos_result) → (1 : ℚ) = 1 := 
by
  intros h
  sorry

end tamika_greater_probability_l233_233969


namespace range_of_k_l233_233146

theorem range_of_k (k : ℤ) (x : ℤ) 
  (h1 : -4 * x - k ≤ 0) 
  (h2 : x = -1 ∨ x = -2) : 
  8 ≤ k ∧ k < 12 :=
sorry

end range_of_k_l233_233146


namespace actual_average_height_l233_233645

theorem actual_average_height 
  (incorrect_avg_height : ℝ)
  (num_students : ℕ)
  (incorrect_height : ℝ)
  (correct_height : ℝ)
  (actual_avg_height : ℝ) :
  incorrect_avg_height = 175 →
  num_students = 20 →
  incorrect_height = 151 →
  correct_height = 111 →
  actual_avg_height = 173 :=
by
  sorry

end actual_average_height_l233_233645


namespace original_number_l233_233991

theorem original_number (x : ℤ) (h : (x + 4) % 23 = 0) : x = 19 :=
sorry

end original_number_l233_233991


namespace complex_root_of_unity_prod_l233_233508

theorem complex_root_of_unity_prod (r : ℂ) (h₁ : r^6 = 1) (h₂ : r ≠ 1) :
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) = 6 :=
by
  sorry

end complex_root_of_unity_prod_l233_233508


namespace solve_system_l233_233934

theorem solve_system (x y z a : ℝ) 
  (h1 : x + y + z = a) 
  (h2 : x^2 + y^2 + z^2 = a^2) 
  (h3 : x^3 + y^3 + z^3 = a^3) : 
  (x = 0 ∧ y = 0 ∧ z = a) ∨ 
  (x = 0 ∧ y = a ∧ z = 0) ∨ 
  (x = a ∧ y = 0 ∧ z = 0) := 
sorry

end solve_system_l233_233934


namespace product_of_x_values_l233_233073

noncomputable def find_product_of_x : ℚ :=
  let x1 := -20
  let x2 := -20 / 7
  (x1 * x2)

theorem product_of_x_values :
  (∃ x : ℚ, abs (20 / x + 4) = 3) ->
  find_product_of_x = 400 / 7 :=
by
  sorry

end product_of_x_values_l233_233073


namespace lateral_area_cone_l233_233843

-- Define the cone problem with given conditions
def radius : ℝ := 5
def slant_height : ℝ := 10

-- Given these conditions, prove the lateral area is 50π
theorem lateral_area_cone (r : ℝ) (l : ℝ) (h_r : r = 5) (h_l : l = 10) : (1/2) * 2 * Real.pi * r * l = 50 * Real.pi :=
by 
  -- import useful mathematical tools
  sorry

end lateral_area_cone_l233_233843


namespace smallest_of_seven_consecutive_even_numbers_l233_233095

theorem smallest_of_seven_consecutive_even_numbers (a b c d e f g : ℤ)
  (h₁ : a + b + c + d + e + f + g = 700)
  (h₂ : b = a + 2)
  (h₃ : c = a + 4)
  (h₄ : d = a + 6)
  (h₅ : e = a + 8)
  (h₆ : f = a + 10)
  (h₇ : g = a + 12)
  : a = 94 :=
by
  -- Proof is omitted, this is just the statement.
  sorry

end smallest_of_seven_consecutive_even_numbers_l233_233095


namespace radius_of_circle_l233_233498

-- Define the given circle equation as a condition
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 2*y - 7 = 0

theorem radius_of_circle : ∀ x y : ℝ, circle_equation x y → ∃ r : ℝ, r = 3 :=
by
  sorry

end radius_of_circle_l233_233498


namespace billy_initial_lemon_heads_l233_233996

theorem billy_initial_lemon_heads (n f : ℕ) (h_friends : f = 6) (h_eat : n = 12) :
  f * n = 72 := 
by
  -- Proceed by proving the statement using Lean
  sorry

end billy_initial_lemon_heads_l233_233996


namespace Peter_total_distance_l233_233417

theorem Peter_total_distance 
  (total_time : ℝ) 
  (speed1 speed2 fraction1 fraction2 : ℝ) 
  (h_time : total_time = 1.4) 
  (h_speed1 : speed1 = 4) 
  (h_speed2 : speed2 = 5) 
  (h_fraction1 : fraction1 = 2/3) 
  (h_fraction2 : fraction2 = 1/3) 
  (D : ℝ) : 
  (fraction1 * D / speed1 + fraction2 * D / speed2 = total_time) → D = 6 :=
by
  intros h_eq
  sorry

end Peter_total_distance_l233_233417


namespace west_movement_80_eq_neg_80_l233_233587

-- Define conditions
def east_movement (distance : ℤ) : ℤ := distance

-- Prove that moving westward is represented correctly
theorem west_movement_80_eq_neg_80 : east_movement (-80) = -80 :=
by
  -- Theorem proof goes here
  sorry

end west_movement_80_eq_neg_80_l233_233587


namespace rational_product_sum_l233_233692

theorem rational_product_sum (x y : ℚ) 
  (h1 : x * y < 0) 
  (h2 : x + y < 0) : 
  |y| < |x| ∧ y < 0 ∧ x > 0 ∨ |x| < |y| ∧ x < 0 ∧ y > 0 :=
by
  sorry

end rational_product_sum_l233_233692


namespace every_positive_integer_displayable_l233_233592

-- Definitions based on the conditions of the problem
def flip_switch_up (n : ℕ) : ℕ := n + 1
def flip_switch_down (n : ℕ) : ℕ := n - 1
def press_red_button (n : ℕ) : ℕ := n * 3
def press_yellow_button (n : ℕ) : ℕ := if n % 3 = 0 then n / 3 else n
def press_green_button (n : ℕ) : ℕ := n * 5
def press_blue_button (n : ℕ) : ℕ := if n % 5 = 0 then n / 5 else n

-- Prove that every positive integer can appear on the calculator display
theorem every_positive_integer_displayable : ∀ n : ℕ, n > 0 → 
  ∃ m : ℕ, m = n ∧
    (m = flip_switch_up m ∨ m = flip_switch_down m ∨ 
     m = press_red_button m ∨ m = press_yellow_button m ∨ 
     m = press_green_button m ∨ m = press_blue_button m) := 
sorry

end every_positive_integer_displayable_l233_233592


namespace intersection_A_B_l233_233942

-- Define sets A and B
def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {1, 3, 5, 7}

-- The theorem stating the intersection of A and B
theorem intersection_A_B : A ∩ B = {1, 3} :=
by
  sorry -- proof is skipped as instructed

end intersection_A_B_l233_233942


namespace problem1_problem2_l233_233217

-- Problem 1
theorem problem1 (x: ℚ) (h: x + 1 / 4 = 7 / 4) : x = 3 / 2 :=
by sorry

-- Problem 2
theorem problem2 (x: ℚ) (h: 2 / 3 + x = 3 / 4) : x = 1 / 12 :=
by sorry

end problem1_problem2_l233_233217


namespace math_books_count_l233_233853

theorem math_books_count (M H : ℕ) :
  M + H = 90 →
  4 * M + 5 * H = 396 →
  H = 90 - M →
  M = 54 :=
by
  intro h1 h2 h3
  sorry

end math_books_count_l233_233853


namespace stamp_collection_cost_l233_233648

def cost_brazil_per_stamp : ℝ := 0.08
def cost_peru_per_stamp : ℝ := 0.05
def num_brazil_stamps_60s : ℕ := 7
def num_peru_stamps_60s : ℕ := 4
def num_brazil_stamps_70s : ℕ := 12
def num_peru_stamps_70s : ℕ := 6

theorem stamp_collection_cost :
  num_brazil_stamps_60s * cost_brazil_per_stamp +
  num_peru_stamps_60s * cost_peru_per_stamp +
  num_brazil_stamps_70s * cost_brazil_per_stamp +
  num_peru_stamps_70s * cost_peru_per_stamp =
  2.02 :=
by
  -- Skipping proof steps.
  sorry

end stamp_collection_cost_l233_233648


namespace confectioner_pastry_l233_233113

theorem confectioner_pastry (P : ℕ) (h : P / 28 - 6 = P / 49) : P = 378 :=
sorry

end confectioner_pastry_l233_233113


namespace log_sqrt2_bounds_l233_233257

theorem log_sqrt2_bounds :
  10^3 = 1000 →
  10^4 = 10000 →
  2^11 = 2048 →
  2^12 = 4096 →
  2^13 = 8192 →
  2^14 = 16384 →
  3 / 22 < Real.log 2 / Real.log 10 / 2 ∧ Real.log 2 / Real.log 10 / 2 < 1 / 7 :=
by
  sorry

end log_sqrt2_bounds_l233_233257


namespace line_intersects_circle_l233_233316

theorem line_intersects_circle (α : ℝ) (r : ℝ) (hα : true) (hr : r > 0) :
  (∃ x y : ℝ, (x * Real.cos α + y * Real.sin α = 1) ∧ (x^2 + y^2 = r^2)) → r > 1 :=
by
  sorry

end line_intersects_circle_l233_233316


namespace mandy_pieces_eq_fifteen_l233_233736

-- Define the initial chocolate pieces
def total_pieces := 60

-- Define Michael's share
def michael_share := total_pieces / 2

-- Define the remainder after Michael's share
def remainder_after_michael := total_pieces - michael_share

-- Define Paige's share
def paige_share := remainder_after_michael / 2

-- Define the remainder after Paige's share
def mandy_share := remainder_after_michael - paige_share

-- Theorem to prove Mandy gets 15 pieces
theorem mandy_pieces_eq_fifteen : mandy_share = 15 :=
by
  sorry

end mandy_pieces_eq_fifteen_l233_233736


namespace correct_average_wrong_reading_l233_233703

theorem correct_average_wrong_reading
  (initial_average : ℕ) (list_length : ℕ) (wrong_number : ℕ) (correct_number : ℕ) (correct_average : ℕ) 
  (h1 : initial_average = 18)
  (h2 : list_length = 10)
  (h3 : wrong_number = 26)
  (h4 : correct_number = 66)
  (h5 : correct_average = 22) :
  correct_average = ((initial_average * list_length) - wrong_number + correct_number) / list_length :=
sorry

end correct_average_wrong_reading_l233_233703


namespace at_least_one_not_less_than_four_l233_233886

theorem at_least_one_not_less_than_four 
( m n t : ℝ ) 
( h_m : 0 < m ) 
( h_n : 0 < n ) 
( h_t : 0 < t ) : 
∃ a, ( a = m + 4 / n ∨ a = n + 4 / t ∨ a = t + 4 / m ) ∧ 4 ≤ a :=
sorry

end at_least_one_not_less_than_four_l233_233886


namespace lines_through_origin_l233_233466

theorem lines_through_origin (n : ℕ) (h : 0 < n) :
    ∃ S : Finset (ℤ × ℤ), 
    (∀ xy : ℤ × ℤ, xy ∈ S ↔ (0 ≤ xy.1 ∧ xy.1 ≤ n ∧ 0 ≤ xy.2 ∧ xy.2 ≤ n ∧ Int.gcd xy.1 xy.2 = 1)) ∧
    S.card ≥ n^2 / 4 := 
sorry

end lines_through_origin_l233_233466


namespace complex_ratio_identity_l233_233518

variable {x y : ℂ}

theorem complex_ratio_identity :
  ( (x + y) / (x - y) - (x - y) / (x + y) = 3 ) →
  ( (x^4 + y^4) / (x^4 - y^4) - (x^4 - y^4) / (x^4 + y^4) = 49 / 600) :=
by
  sorry

end complex_ratio_identity_l233_233518


namespace missing_number_l233_233524

theorem missing_number (m x : ℕ) (h : 744 + 745 + 747 + 748 + 749 + 752 + 752 + 753 + m + x = 750 * 10)  
  (hx : x = 755) : m = 805 := by 
  sorry

end missing_number_l233_233524


namespace new_class_mean_l233_233898

theorem new_class_mean 
  (n1 n2 : ℕ) (mean1 mean2 : ℚ) 
  (h1 : n1 = 24) (h2 : n2 = 8) 
  (h3 : mean1 = 85/100) (h4 : mean2 = 90/100) :
  (n1 * mean1 + n2 * mean2) / (n1 + n2) = 345/400 :=
by
  rw [h1, h2, h3, h4]
  sorry

end new_class_mean_l233_233898


namespace airplane_seats_l233_233496

theorem airplane_seats (s : ℝ)
  (h1 : 0.30 * s = 0.30 * s)
  (h2 : (3 / 5) * s = (3 / 5) * s)
  (h3 : 36 + 0.30 * s + (3 / 5) * s = s) : s = 360 :=
by
  sorry

end airplane_seats_l233_233496


namespace trigonometric_identity_l233_233945

theorem trigonometric_identity : (1 / Real.cos (70 * Real.pi / 180) - Real.sqrt 3 / Real.sin (70 * Real.pi / 180)) = 1 - Real.sqrt 3 :=
by
  sorry

end trigonometric_identity_l233_233945


namespace angle_B_eq_pi_div_3_l233_233892

variables {A B C : ℝ} {a b c : ℝ}

/-- Given an acute triangle ABC, where sides a, b, c are opposite the angles A, B, and C respectively, 
    and given the condition b cos C + sqrt 3 * b sin C = a + c, prove that B = π / 3. -/
theorem angle_B_eq_pi_div_3 
  (h : ∀ (A B C : ℝ), 0 < A ∧ A < π / 2  ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2 ∧ A + B + C = π)
  (cond : b * Real.cos C + Real.sqrt 3 * b * Real.sin C = a + c) :
  B = π / 3 := 
sorry

end angle_B_eq_pi_div_3_l233_233892


namespace range_of_f_l233_233454

noncomputable def f (x : ℝ) : ℝ := Real.arcsin x + Real.arccos x + Real.arctan (2 * x)

theorem range_of_f :
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → (f x ≥ (Real.pi / 2 - Real.arctan 2) ∧ f x ≤ (Real.pi / 2 + Real.arctan 2)) :=
by
  sorry

end range_of_f_l233_233454


namespace border_area_correct_l233_233200

noncomputable def area_of_border (poster_height poster_width border_width : ℕ) : ℕ :=
  let framed_height := poster_height + 2 * border_width
  let framed_width := poster_width + 2 * border_width
  (framed_height * framed_width) - (poster_height * poster_width)

theorem border_area_correct :
  area_of_border 12 16 4 = 288 :=
by
  rfl

end border_area_correct_l233_233200


namespace magic_8_ball_probability_l233_233856

def probability_positive (p_pos : ℚ) (questions : ℕ) (positive_responses : ℕ) : ℚ :=
  (Nat.choose questions positive_responses : ℚ) * (p_pos ^ positive_responses) * ((1 - p_pos) ^ (questions - positive_responses))

theorem magic_8_ball_probability :
  probability_positive (1/3) 7 3 = 560 / 2187 :=
by
  sorry

end magic_8_ball_probability_l233_233856


namespace dice_probability_l233_233656

def prob_at_least_one_one : ℚ :=
  let total_outcomes := 36
  let no_1_outcomes := 25
  let favorable_outcomes := total_outcomes - no_1_outcomes
  let probability := favorable_outcomes / total_outcomes
  probability

theorem dice_probability :
  prob_at_least_one_one = 11 / 36 :=
by
  sorry

end dice_probability_l233_233656


namespace find_breadth_of_plot_l233_233515

-- Define the conditions
def length_of_plot (breadth : ℝ) := 3 * breadth
def area_of_plot := 2028

-- Define what we want to prove
theorem find_breadth_of_plot (breadth : ℝ) (h1 : length_of_plot breadth * breadth = area_of_plot) : breadth = 26 :=
sorry

end find_breadth_of_plot_l233_233515


namespace compute_fraction_power_l233_233447

theorem compute_fraction_power :
  8 * (2 / 7)^4 = 128 / 2401 :=
by
  sorry

end compute_fraction_power_l233_233447


namespace simple_interest_rate_l233_233672

theorem simple_interest_rate (P A : ℝ) (T : ℕ) (R : ℝ) 
  (P_pos : P = 800) (A_pos : A = 950) (T_pos : T = 5) :
  R = 3.75 :=
by
  sorry

end simple_interest_rate_l233_233672


namespace neg_mul_neg_pos_mul_neg_neg_l233_233979

theorem neg_mul_neg_pos (a b : Int) (ha : a < 0) (hb : b < 0) : a * b > 0 :=
sorry

theorem mul_neg_neg : (-1) * (-3) = 3 := 
by
  have h1 : -1 < 0 := by norm_num
  have h2 : -3 < 0 := by norm_num
  have h_pos := neg_mul_neg_pos (-1) (-3) h1 h2
  linarith

end neg_mul_neg_pos_mul_neg_neg_l233_233979


namespace wayne_took_cards_l233_233368

-- Let's define the problem context
variable (initial_cards : ℕ := 76)
variable (remaining_cards : ℕ := 17)

-- We need to show that Wayne took away 59 cards
theorem wayne_took_cards (x : ℕ) (h : x = initial_cards - remaining_cards) : x = 59 :=
by
  sorry

end wayne_took_cards_l233_233368


namespace opposite_of_neg_2023_l233_233669

theorem opposite_of_neg_2023 : -( -2023 ) = 2023 := by
  sorry

end opposite_of_neg_2023_l233_233669


namespace total_cats_received_l233_233415

-- Defining the constants and conditions
def total_adult_cats := 150
def fraction_female_cats := 2 / 3
def fraction_litters := 2 / 5
def kittens_per_litter := 5

-- Defining the proof problem
theorem total_cats_received :
  let number_female_cats := (fraction_female_cats * total_adult_cats : ℤ)
  let number_litters := (fraction_litters * number_female_cats : ℤ)
  let number_kittens := number_litters * kittens_per_litter
  number_female_cats + number_kittens + (total_adult_cats - number_female_cats) = 350 := 
by
  sorry

end total_cats_received_l233_233415


namespace Harriet_age_now_l233_233013

variable (P H: ℕ)

theorem Harriet_age_now (P : ℕ) (H : ℕ) (h1 : P + 4 = 2 * (H + 4)) (h2 : P = 60 / 2) : H = 13 := by
  sorry

end Harriet_age_now_l233_233013


namespace fraction_meaningful_iff_x_ne_pm1_l233_233045

theorem fraction_meaningful_iff_x_ne_pm1 (x : ℝ) : (x^2 - 1 ≠ 0) ↔ (x ≠ 1 ∧ x ≠ -1) :=
by
  sorry

end fraction_meaningful_iff_x_ne_pm1_l233_233045


namespace number_of_parallelograms_l233_233236

-- Problem's condition
def side_length (n : ℕ) : Prop := n > 0

-- Required binomial coefficient (combination formula)
def binom (n k : ℕ) : ℕ := n.choose k

-- Total number of parallelograms in the tiling
theorem number_of_parallelograms (n : ℕ) (h : side_length n) : 
  3 * binom (n + 2) 4 = 3 * (n+2).choose 4 :=
by
  sorry

end number_of_parallelograms_l233_233236


namespace only_possible_b_l233_233147

theorem only_possible_b (b : ℕ) (h : ∃ a k l : ℕ, k ≠ l ∧ (b > 0) ∧ (a > 0) ∧ (b ^ (k + l)) ∣ (a ^ k + b ^ l) ∧ (b ^ (k + l)) ∣ (a ^ l + b ^ k)) : 
  b = 1 :=
sorry

end only_possible_b_l233_233147


namespace xy_product_l233_233126

theorem xy_product (x y z : ℝ) (h : x^2 + y^2 = x * y * (z + 1 / z)) :
  x = y * z ∨ y = x * z := 
by
  sorry

end xy_product_l233_233126


namespace vanessa_savings_remaining_l233_233280

-- Conditions
def initial_investment : ℝ := 50000
def annual_interest_rate : ℝ := 0.035
def investment_duration : ℕ := 3
def conversion_rate : ℝ := 0.85
def cost_per_toy : ℝ := 75

-- Given the above conditions, prove the remaining amount in euros after buying as many toys as possible is 16.9125
theorem vanessa_savings_remaining
  (P : ℝ := initial_investment)
  (r : ℝ := annual_interest_rate)
  (t : ℕ := investment_duration)
  (c : ℝ := conversion_rate)
  (e : ℝ := cost_per_toy) :
  (((P * (1 + r)^t) * c) - (e * (⌊(P * (1 + r)^3 * 0.85) / e⌋))) = 16.9125 :=
sorry

end vanessa_savings_remaining_l233_233280


namespace simplify_expression_l233_233710

theorem simplify_expression (x : ℝ) : 
  3 - 5*x - 6*x^2 + 9 + 11*x - 12*x^2 - 15 + 17*x + 18*x^2 - 2*x^3 = -2*x^3 + 23*x - 3 :=
by
  sorry

end simplify_expression_l233_233710


namespace value_of_h_otimes_h_otimes_h_l233_233210

variable (h x y : ℝ)

-- Define the new operation
def otimes (x y : ℝ) := x^3 - x * y + y^2

-- Prove that h ⊗ (h ⊗ h) = h^6 - h^4 + h^3
theorem value_of_h_otimes_h_otimes_h :
  otimes h (otimes h h) = h^6 - h^4 + h^3 := by
  sorry

end value_of_h_otimes_h_otimes_h_l233_233210


namespace time_spent_watching_tv_excluding_breaks_l233_233538

-- Definitions based on conditions
def total_hours_watched : ℕ := 5
def breaks : List ℕ := [10, 15, 20, 25]

-- Conversion constants
def minutes_per_hour : ℕ := 60

-- Derived definitions
def total_minutes_watched : ℕ := total_hours_watched * minutes_per_hour
def total_break_minutes : ℕ := breaks.sum

-- The main theorem
theorem time_spent_watching_tv_excluding_breaks :
  total_minutes_watched - total_break_minutes = 230 := by
  sorry

end time_spent_watching_tv_excluding_breaks_l233_233538


namespace find_b_l233_233103

-- Define complex numbers z1 and z2
def z1 (b : ℝ) : Complex := Complex.mk 3 (-b)

def z2 : Complex := Complex.mk 1 (-2)

-- Statement that needs to be proved
theorem find_b (b : ℝ) (h : (z1 b / z2).re = 0) : b = -3 / 2 :=
by
  -- proof goes here
  sorry

end find_b_l233_233103


namespace alpha_squared_plus_3alpha_plus_beta_equals_2023_l233_233422

-- Definitions and conditions
variables (α β : ℝ)
-- α and β are roots of the quadratic equation x² + 2x - 2025 = 0
def is_root_of_quadratic_1 : Prop := α^2 + 2 * α - 2025 = 0
def is_root_of_quadratic_2 : Prop := β^2 + 2 * β - 2025 = 0
-- Vieta's formula gives us α + β = -2
def sum_of_roots : Prop := α + β = -2

-- Theorem (statement) we want to prove
theorem alpha_squared_plus_3alpha_plus_beta_equals_2023 (h1 : is_root_of_quadratic_1 α)
                                                      (h2 : is_root_of_quadratic_2 β)
                                                      (h3 : sum_of_roots α β) :
                                                      α^2 + 3 * α + β = 2023 :=
by
  sorry

end alpha_squared_plus_3alpha_plus_beta_equals_2023_l233_233422


namespace find_a_b_l233_233239

def z := Complex.ofReal 3 + Complex.I * 4
def z_conj := Complex.ofReal 3 - Complex.I * 4

theorem find_a_b 
  (a b : ℝ) 
  (h : z + Complex.ofReal a * z_conj + Complex.I * b = Complex.ofReal 9) : 
  a = 2 ∧ b = 4 := 
by 
  sorry

end find_a_b_l233_233239


namespace smallest_b_1111_is_square_l233_233079

theorem smallest_b_1111_is_square : 
  ∃ b : ℕ, b > 0 ∧ (∀ n : ℕ, (b^3 + b^2 + b + 1 = n^2 → b = 7)) :=
by
  sorry

end smallest_b_1111_is_square_l233_233079


namespace edge_length_of_cubical_box_l233_233141

noncomputable def volume_of_cube (edge_length_cm : ℝ) : ℝ :=
  edge_length_cm ^ 3

noncomputable def number_of_cubes : ℝ := 8000
noncomputable def edge_of_small_cube_cm : ℝ := 5

noncomputable def total_volume_of_cubes_cm3 : ℝ :=
  volume_of_cube edge_of_small_cube_cm * number_of_cubes

noncomputable def volume_of_box_cm3 : ℝ := total_volume_of_cubes_cm3
noncomputable def edge_length_of_box_m : ℝ :=
  (volume_of_box_cm3)^(1 / 3) / 100

theorem edge_length_of_cubical_box :
  edge_length_of_box_m = 1 := by 
  sorry

end edge_length_of_cubical_box_l233_233141


namespace ned_initial_lives_l233_233697

variable (lost_lives : ℕ) (current_lives : ℕ) 
variable (initial_lives : ℕ)

theorem ned_initial_lives (h_lost: lost_lives = 13) (h_current: current_lives = 70) :
  initial_lives = current_lives + lost_lives := by
  sorry

end ned_initial_lives_l233_233697


namespace different_ways_to_eat_spaghetti_l233_233837

-- Define the conditions
def red_spaghetti := 5
def blue_spaghetti := 5
def total_spaghetti := 6

-- This is the proof statement
theorem different_ways_to_eat_spaghetti : 
  ∃ (ways : ℕ), ways = 62 ∧ 
  (∃ r b : ℕ, r ≤ red_spaghetti ∧ b ≤ blue_spaghetti ∧ r + b = total_spaghetti) := 
sorry

end different_ways_to_eat_spaghetti_l233_233837


namespace cost_of_jeans_l233_233928

    variable (J S : ℝ)

    def condition1 := 3 * J + 6 * S = 104.25
    def condition2 := 4 * J + 5 * S = 112.15

    theorem cost_of_jeans (h1 : condition1 J S) (h2 : condition2 J S) : J = 16.85 := by
      sorry
    
end cost_of_jeans_l233_233928


namespace range_a_sub_b_mul_c_l233_233224

theorem range_a_sub_b_mul_c (a b c : ℝ) (h1 : -1 < a) (h2 : a < b) (h3 : b < 1) (h4 : 2 < c) (h5 : c < 3) :
  -6 < (a - b) * c ∧ (a - b) * c < 0 :=
by
  -- We need to prove the range of (a - b) * c is within (-6, 0)
  sorry

end range_a_sub_b_mul_c_l233_233224


namespace rent_cost_l233_233234

-- Definitions based on conditions
def daily_supplies_cost : ℕ := 12
def price_per_pancake : ℕ := 2
def pancakes_sold_per_day : ℕ := 21

-- Proving the daily rent cost
theorem rent_cost (total_sales : ℕ) (rent : ℕ) :
  total_sales = pancakes_sold_per_day * price_per_pancake →
  rent = total_sales - daily_supplies_cost →
  rent = 30 :=
by
  intro h_total_sales h_rent
  sorry

end rent_cost_l233_233234


namespace problems_left_to_grade_l233_233088

theorem problems_left_to_grade 
  (problems_per_worksheet : ℕ)
  (total_worksheets : ℕ)
  (graded_worksheets : ℕ)
  (h1 : problems_per_worksheet = 2)
  (h2 : total_worksheets = 14)
  (h3 : graded_worksheets = 7) : 
  (total_worksheets - graded_worksheets) * problems_per_worksheet = 14 :=
by 
  sorry

end problems_left_to_grade_l233_233088


namespace total_savings_l233_233101

def liam_oranges : ℕ := 40
def liam_price_per_two : ℝ := 2.50
def claire_oranges : ℕ := 30
def claire_price_per_one : ℝ := 1.20

theorem total_savings : (liam_oranges / 2 * liam_price_per_two) + (claire_oranges * claire_price_per_one) = 86 := by
  sorry

end total_savings_l233_233101


namespace dara_jane_age_ratio_l233_233284

theorem dara_jane_age_ratio :
  ∀ (min_age : ℕ) (jane_current_age : ℕ) (dara_years_til_min_age : ℕ) (d : ℕ) (j : ℕ),
  min_age = 25 →
  jane_current_age = 28 →
  dara_years_til_min_age = 14 →
  d = 17 →
  j = 34 →
  d = dara_years_til_min_age - 14 + 6 →
  j = jane_current_age + 6 →
  (d:ℚ) / j = 1 / 2 := 
by
  intros
  sorry

end dara_jane_age_ratio_l233_233284


namespace sum_of_second_and_third_of_four_consecutive_even_integers_l233_233266

-- Definitions of conditions
variables (n : ℤ)  -- Assume n is an integer

-- Statement of problem
theorem sum_of_second_and_third_of_four_consecutive_even_integers (h : 2 * n + 6 = 160) :
  (n + 2) + (n + 4) = 160 :=
by
  sorry

end sum_of_second_and_third_of_four_consecutive_even_integers_l233_233266


namespace loan_duration_l233_233786

theorem loan_duration (P R SI : ℝ) (hP : P = 20000) (hR : R = 12) (hSI : SI = 7200) : 
  ∃ T : ℝ, T = 3 :=
by
  sorry

end loan_duration_l233_233786


namespace value_of_x_minus_y_squared_l233_233848

theorem value_of_x_minus_y_squared (x y : ℝ) (hx : x^2 = 4) (hy : y^2 = 9) : 
  ((x - y)^2 = 1) ∨ ((x - y)^2 = 25) :=
sorry

end value_of_x_minus_y_squared_l233_233848


namespace original_price_l233_233525

theorem original_price (P : ℝ) (S : ℝ) (h1 : S = 1.3 * P) (h2 : S = P + 650) : P = 2166.67 :=
by
  sorry

end original_price_l233_233525


namespace apples_first_year_l233_233586

theorem apples_first_year (A : ℕ) 
  (second_year_prod : ℕ := 2 * A + 8)
  (third_year_prod : ℕ := 3 * (2 * A + 8) / 4)
  (total_prod : ℕ := A + second_year_prod + third_year_prod) :
  total_prod = 194 → A = 40 :=
by
  sorry

end apples_first_year_l233_233586


namespace student_A_incorrect_l233_233919

def is_on_circle (center : ℝ × ℝ) (radius : ℝ) (point : ℝ × ℝ) : Prop :=
  let (cx, cy) := center
  let (px, py) := point
  (px - cx)^2 + (py - cy)^2 = radius^2

def center : ℝ × ℝ := (2, -3)
def radius : ℝ := 5
def point_A : ℝ × ℝ := (-2, -1)
def point_D : ℝ × ℝ := (5, 1)

theorem student_A_incorrect :
  ¬ is_on_circle center radius point_A ∧ is_on_circle center radius point_D :=
by
  sorry

end student_A_incorrect_l233_233919


namespace greatest_divisible_by_13_l233_233197

def is_distinct_nonzero_digits (A B C : ℕ) : Prop :=
  0 < A ∧ A < 10 ∧ 0 < B ∧ B < 10 ∧ 0 < C ∧ C < 10 ∧ A ≠ B ∧ B ≠ C ∧ A ≠ C

def number (A B C : ℕ) : ℕ :=
  10000 * A + 1000 * B + 100 * C + 10 * B + A

theorem greatest_divisible_by_13 :
  ∃ (A B C : ℕ), is_distinct_nonzero_digits A B C ∧ number A B C % 13 = 0 ∧ number A B C = 96769 :=
sorry

end greatest_divisible_by_13_l233_233197


namespace smaller_tablet_diagonal_l233_233581

theorem smaller_tablet_diagonal :
  ∀ (A_large A_small : ℝ)
    (d : ℝ),
    A_large = (8 / Real.sqrt 2) ^ 2 →
    A_small = (d / Real.sqrt 2) ^ 2 →
    A_large = A_small + 7.5 →
    d = 7
:= by
  intros A_large A_small d h1 h2 h3
  sorry

end smaller_tablet_diagonal_l233_233581


namespace number_of_bowls_l233_233957

theorem number_of_bowls (n : ℕ) (h : 8 * 12 = 96) (avg_increase : 6 * n = 96) : n = 16 :=
by {
  sorry
}

end number_of_bowls_l233_233957


namespace determinant_triangle_l233_233537

theorem determinant_triangle (A B C : ℝ) (h : A + B + C = Real.pi) :
  Matrix.det ![![Real.cos A ^ 2, Real.tan A, 1],
               ![Real.cos B ^ 2, Real.tan B, 1],
               ![Real.cos C ^ 2, Real.tan C, 1]] = 0 := by
  sorry

end determinant_triangle_l233_233537


namespace total_hike_time_l233_233709

-- Define the conditions
def distance_to_mount_overlook : ℝ := 12
def pace_to_mount_overlook : ℝ := 4
def pace_return : ℝ := 6

-- Prove the total time for the hike
theorem total_hike_time :
  (distance_to_mount_overlook / pace_to_mount_overlook) +
  (distance_to_mount_overlook / pace_return) = 5 := 
sorry

end total_hike_time_l233_233709


namespace smallest_possible_value_of_sum_l233_233133

theorem smallest_possible_value_of_sum (a b : ℤ) (h1 : a > 6) (h2 : ∃ a' b', a' - b' = 4) : a + b < 11 := 
sorry

end smallest_possible_value_of_sum_l233_233133


namespace chess_tournament_ratio_l233_233742

theorem chess_tournament_ratio:
  ∃ n : ℕ, (n * (n - 1)) / 2 = 231 ∧ (n - 1) = 21 := 
sorry

end chess_tournament_ratio_l233_233742


namespace lines_parallel_iff_l233_233492

theorem lines_parallel_iff (a : ℝ) : (∀ x y : ℝ, x + 2*a*y - 1 = 0 ∧ (2*a - 1)*x - a*y - 1 = 0 → x = 1 ∧ x = -1 ∨ ∃ (slope : ℝ), slope = - (1 / (2 * a)) ∧ slope = (2 * a - 1) / a) ↔ (a = 0 ∨ a = 1/4) :=
by
  sorry

end lines_parallel_iff_l233_233492


namespace find_actual_number_of_children_l233_233553

theorem find_actual_number_of_children (B C : ℕ) (h1 : B = 2 * C) (h2 : B = 4 * (C - 420)) : C = 840 := 
by
  sorry

end find_actual_number_of_children_l233_233553


namespace least_prime_b_l233_233783

-- Define what it means for an angle to be a right triangle angle sum
def isRightTriangleAngleSum (a b : ℕ) : Prop := a + b = 90

-- Define what it means for a number to be prime
def isPrime (n : ℕ) : Prop := Nat.Prime n

-- Formalize the goal: proving that the smallest possible b is 7
theorem least_prime_b (a b : ℕ) (h1 : isRightTriangleAngleSum a b) (h2 : isPrime a) (h3 : isPrime b) (h4 : a > b) : b = 7 :=
sorry

end least_prime_b_l233_233783


namespace no_such_functions_exist_l233_233539

theorem no_such_functions_exist (f g : ℝ → ℝ) :
  ¬ (∀ x y : ℝ, x ≠ y → |f x - f y| + |g x - g y| > 1) :=
sorry

end no_such_functions_exist_l233_233539


namespace sequence_6th_term_l233_233180

theorem sequence_6th_term 
    (a₁ a₂ a₃ a₄ a₅ a₆ : ℚ)
    (h₁ : a₁ = 3)
    (h₅ : a₅ = 54)
    (h₂ : a₂ = (a₁ + a₃) / 3)
    (h₃ : a₃ = (a₂ + a₄) / 3)
    (h₄ : a₄ = (a₃ + a₅) / 3)
    (h₆ : a₅ = (a₄ + a₆) / 3) :
    a₆ = 1133 / 7 :=
by
  sorry

end sequence_6th_term_l233_233180


namespace land_area_of_each_section_l233_233443

theorem land_area_of_each_section (n : ℕ) (total_area : ℕ) (h1 : n = 3) (h2 : total_area = 7305) :
  total_area / n = 2435 :=
by {
  sorry
}

end land_area_of_each_section_l233_233443


namespace simplify_expression_l233_233394

theorem simplify_expression :
  (2021^3 - 3 * 2021^2 * 2022 + 4 * 2021 * 2022^2 - 2022^3 + 2) / (2021 * 2022) = 
  1 + (1 / 2021) :=
by
  sorry

end simplify_expression_l233_233394


namespace inequality_example_l233_233620

theorem inequality_example (a b c : ℝ) (hac : a ≠ 0) (hbc : b ≠ 0) (hcc : c ≠ 0) :
  (a^4) / (4 * a^4 + b^4 + c^4) + (b^4) / (a^4 + 4 * b^4 + c^4) + (c^4) / (a^4 + b^4 + 4 * c^4) ≤ 1 / 2 :=
sorry

end inequality_example_l233_233620


namespace sum_of_perimeters_l233_233130

theorem sum_of_perimeters (x y : Real) 
  (h1 : x^2 + y^2 = 85)
  (h2 : x^2 - y^2 = 45) :
  4 * (Real.sqrt 65 + 2 * Real.sqrt 5) = 4 * x + 4 * y := by
  sorry

end sum_of_perimeters_l233_233130


namespace product_of_two_real_numbers_sum_three_times_product_l233_233290

variable (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)

theorem product_of_two_real_numbers_sum_three_times_product
    (h : x + y = 3 * x * y) :
  x * y = (x + y) / 3 :=
sorry

end product_of_two_real_numbers_sum_three_times_product_l233_233290


namespace cost_price_percentage_l233_233261

variable (SP CP : ℝ)

-- Assumption that the profit percent is 25%
axiom profit_percent : 25 = ((SP - CP) / CP) * 100

-- The statement to prove
theorem cost_price_percentage : CP / SP = 0.8 := by
  sorry

end cost_price_percentage_l233_233261


namespace max_m_l233_233408

theorem max_m : ∃ m A B : ℤ, (AB = 90 ∧ m = 5 * B + A) ∧ (∀ m' A' B', (A' * B' = 90 ∧ m' = 5 * B' + A') → m' ≤ 451) ∧ m = 451 :=
by
  sorry

end max_m_l233_233408


namespace peggy_records_l233_233478

theorem peggy_records (R : ℕ) (h : 4 * R - (3 * R + R / 2) = 100) : R = 200 :=
sorry

end peggy_records_l233_233478


namespace grandson_age_l233_233312

theorem grandson_age (M S G : ℕ) (h1 : M = 2 * S) (h2 : S = 2 * G) (h3 : M + S + G = 140) : G = 20 :=
by 
  sorry

end grandson_age_l233_233312


namespace cody_initial_tickets_l233_233319

def initial_tickets (lost : ℝ) (spent : ℝ) (left : ℝ) : ℝ :=
  lost + spent + left

theorem cody_initial_tickets : initial_tickets 6.0 25.0 18.0 = 49.0 := by
  sorry

end cody_initial_tickets_l233_233319


namespace unique_integer_sequence_exists_l233_233166

open Nat

def a (n : ℕ) : ℤ := sorry

theorem unique_integer_sequence_exists :
  ∃ (a : ℕ → ℤ), a 1 = 1 ∧ a 2 > 1 ∧ (∀ n ≥ 1, (a (n+1))^3 + 1 = a n * a (n+2)) ∧
  (∀ b, (b 1 = 1) → (b 2 > 1) → (∀ n ≥ 1, (b (n+1))^3 + 1 = b n * b (n+2)) → b = a) :=
by
  sorry

end unique_integer_sequence_exists_l233_233166


namespace company_pays_each_man_per_hour_l233_233685

theorem company_pays_each_man_per_hour
  (men : ℕ) (hours_per_job : ℕ) (jobs : ℕ) (total_pay : ℕ)
  (completion_time : men * hours_per_job = 1)
  (total_jobs_time : jobs * hours_per_job = 5)
  (total_earning : total_pay = 150) :
  (total_pay / (jobs * men * hours_per_job)) = 10 :=
sorry

end company_pays_each_man_per_hour_l233_233685


namespace negation_equivalence_l233_233510

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x > 0 → x^2 + x > 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 + x ≤ 0) :=
by
  sorry

end negation_equivalence_l233_233510


namespace correct_sampling_methods_l233_233725

theorem correct_sampling_methods :
  (let num_balls := 1000
   let red_box := 500
   let blue_box := 200
   let yellow_box := 300
   let sample_balls := 100
   let num_students := 20
   let selected_students := 3
   let q1_method := "stratified"
   let q2_method := "simple_random"
   q1_method = "stratified" ∧ q2_method = "simple_random") := sorry

end correct_sampling_methods_l233_233725


namespace negation_proof_l233_233338

theorem negation_proof :
  (¬ (∀ x : ℝ, x^2 - x + 1 > 0)) ↔ (∃ x : ℝ, x^2 - x + 1 ≤ 0) := sorry

end negation_proof_l233_233338


namespace lcm_150_294_l233_233712

theorem lcm_150_294 : Nat.lcm 150 294 = 7350 := by
  sorry

end lcm_150_294_l233_233712


namespace simplify_expression_l233_233298

theorem simplify_expression :
  (∃ (a b c d : ℝ), 
   a = 14 * Real.sqrt 2 ∧ 
   b = 12 * Real.sqrt 2 ∧ 
   c = 8 * Real.sqrt 2 ∧ 
   d = 12 * Real.sqrt 2 ∧ 
   ((a / b) + (c / d) = 11 / 6)) :=
by 
  use 14 * Real.sqrt 2, 12 * Real.sqrt 2, 8 * Real.sqrt 2, 12 * Real.sqrt 2
  simp
  sorry

end simplify_expression_l233_233298


namespace solve_for_x_l233_233460

variable (x : ℝ)

theorem solve_for_x (h : 5 * x - 3 = 17) : x = 4 := sorry

end solve_for_x_l233_233460


namespace find_k_from_polynomial_l233_233409

theorem find_k_from_polynomial :
  ∃ (k : ℝ),
  (∃ (x₁ x₂ x₃ x₄ : ℝ), 
    x₁ * x₂ * x₃ * x₄ = -1984 ∧
    x₁ * x₂ + x₁ * x₃ + x₁ * x₄ + x₂ * x₃ + x₂ * x₄ + x₃ * x₄ = k ∧
    x₁ + x₂ + x₃ + x₄ = 18 ∧
    (x₁ * x₂ = -32 ∨ x₁ * x₃ = -32 ∨ x₁ * x₄ = -32 ∨ x₂ * x₃ = -32 ∨ x₂ * x₄ = -32 ∨ x₃ * x₄ = -32))
  → k = 86 :=
by
  sorry

end find_k_from_polynomial_l233_233409


namespace water_leaving_rate_l233_233690

-- Definitions: Volume of water and time taken
def volume_of_water : ℕ := 300
def time_taken : ℕ := 25

-- Theorem statement: Rate of water leaving the tank
theorem water_leaving_rate : (volume_of_water / time_taken) = 12 := 
by sorry

end water_leaving_rate_l233_233690


namespace train_speed_approximation_l233_233391

theorem train_speed_approximation (train_speed_mph : ℝ) (seconds : ℝ) :
  (40 : ℝ) * train_speed_mph * 1 / 60 = seconds → seconds = 27 := 
  sorry

end train_speed_approximation_l233_233391


namespace alpha_beta_power_eq_sum_power_for_large_p_l233_233750

theorem alpha_beta_power_eq_sum_power_for_large_p (α β : ℂ) (p : ℕ) (hp : p ≥ 5)
  (hαβ : ∀ x : ℂ, 2 * x^4 - 6 * x^3 + 11 * x^2 - 6 * x - 4 = 0 → x = α ∨ x = β) :
  α^p + β^p = (α + β)^p :=
sorry

end alpha_beta_power_eq_sum_power_for_large_p_l233_233750


namespace ratio_of_voters_l233_233043

open Real

theorem ratio_of_voters (X Y : ℝ) (h1 : 0.64 * X + 0.46 * Y = 0.58 * (X + Y)) : X / Y = 2 :=
by
  sorry

end ratio_of_voters_l233_233043


namespace probability_at_least_one_shows_one_is_correct_l233_233149

/-- Two fair 8-sided dice are rolled. What is the probability that at least one of the dice shows a 1? -/
def probability_at_least_one_shows_one : ℚ :=
  let total_outcomes := 8 * 8
  let neither_one := 7 * 7
  let at_least_one := total_outcomes - neither_one
  at_least_one / total_outcomes

theorem probability_at_least_one_shows_one_is_correct :
  probability_at_least_one_shows_one = 15 / 64 :=
by
  unfold probability_at_least_one_shows_one
  sorry

end probability_at_least_one_shows_one_is_correct_l233_233149


namespace decimal_to_base9_l233_233707

theorem decimal_to_base9 (n : ℕ) (h : n = 1729) : 
  (2 * 9^3 + 3 * 9^2 + 3 * 9^1 + 1 * 9^0) = n :=
by sorry

end decimal_to_base9_l233_233707


namespace michael_lap_time_l233_233691

theorem michael_lap_time (T : ℝ) :
  (∀ (lap_time_donovan : ℝ), lap_time_donovan = 45 → (9 * T) / lap_time_donovan + 1 = 9 → T = 40) :=
by
  intro lap_time_donovan
  intro h1
  intro h2
  sorry

end michael_lap_time_l233_233691


namespace horner_value_x_neg2_l233_233457

noncomputable def horner (x : ℝ) : ℝ :=
  (((((x - 5) * x + 6) * x + 0) * x + 1) * x + 0.3) * x + 2

theorem horner_value_x_neg2 : horner (-2) = -40 :=
by
  sorry

end horner_value_x_neg2_l233_233457


namespace trajectory_of_C_is_ellipse_l233_233776

theorem trajectory_of_C_is_ellipse :
  ∀ (C : ℝ × ℝ),
  ((C.1 + 4)^2 + C.2^2).sqrt + ((C.1 - 4)^2 + C.2^2).sqrt = 10 →
  (C.2 ≠ 0) →
  (C.1^2 / 25 + C.2^2 / 9 = 1) :=
by
  intros C h1 h2
  sorry

end trajectory_of_C_is_ellipse_l233_233776


namespace nine_chapters_problem_l233_233249

variable (m n : ℕ)

def horses_condition_1 : Prop := m + n = 100
def horses_condition_2 : Prop := 3 * m + n / 3 = 100

theorem nine_chapters_problem (h1 : horses_condition_1 m n) (h2 : horses_condition_2 m n) :
  (m + n = 100 ∧ 3 * m + n / 3 = 100) :=
by
  exact ⟨h1, h2⟩

end nine_chapters_problem_l233_233249


namespace problem_l233_233450

variables {A B C A1 B1 C1 A0 B0 C0 : Type}

-- Define the acute triangle and constructions
axiom acute_triangle (ABC : Type) : Prop
axiom circumcircle (ABC : Type) (A1 B1 C1 : Type) : Prop
axiom extended_angle_bisectors (ABC : Type) (A0 B0 C0 : Type) : Prop

-- Define the points according to the problem statement
axiom intersections_A0 (ABC : Type) (A0 : Type) : Prop
axiom intersections_B0 (ABC : Type) (B0 : Type) : Prop
axiom intersections_C0 (ABC : Type) (C0 : Type) : Prop

-- Define the areas of triangles and hexagon
axiom area_triangle_A0B0C0 (ABC : Type) (A0 B0 C0 : Type) : ℝ
axiom area_hexagon_AC1B_A1CB1 (ABC : Type) (A1 B1 C1 : Type) : ℝ
axiom area_triangle_ABC (ABC : Type) : ℝ

-- Problem: Prove the area relationships
theorem problem
  (ABC: Type)
  (h1 : acute_triangle ABC)
  (h2 : circumcircle ABC A1 B1 C1)
  (h3 : extended_angle_bisectors ABC A0 B0 C0)
  (h4 : intersections_A0 ABC A0)
  (h5 : intersections_B0 ABC B0)
  (h6 : intersections_C0 ABC C0):
  area_triangle_A0B0C0 ABC A0 B0 C0 = 2 * area_hexagon_AC1B_A1CB1 ABC A1 B1 C1 ∧
  area_triangle_A0B0C0 ABC A0 B0 C0 ≥ 4 * area_triangle_ABC ABC :=
sorry

end problem_l233_233450


namespace kim_points_correct_l233_233399

-- Definitions of given conditions
def points_easy : ℕ := 2
def points_average : ℕ := 3
def points_hard : ℕ := 5

def correct_easy : ℕ := 6
def correct_average : ℕ := 2
def correct_hard : ℕ := 4

-- Definition of total points calculation
def kim_total_points : ℕ :=
  (correct_easy * points_easy) +
  (correct_average * points_average) +
  (correct_hard * points_hard)

-- Theorem stating that Kim's total points are 38
theorem kim_points_correct : kim_total_points = 38 := by
  -- Proof placeholder
  sorry

end kim_points_correct_l233_233399


namespace average_weight_of_class_l233_233203

theorem average_weight_of_class (students_a students_b : ℕ) (avg_weight_a avg_weight_b : ℝ)
  (h_students_a : students_a = 24)
  (h_students_b : students_b = 16)
  (h_avg_weight_a : avg_weight_a = 40)
  (h_avg_weight_b : avg_weight_b = 35) :
  ((students_a * avg_weight_a + students_b * avg_weight_b) / (students_a + students_b)) = 38 := 
by
  sorry

end average_weight_of_class_l233_233203


namespace compute_expression_l233_233705

variable {R : Type*} [LinearOrderedField R]

theorem compute_expression (r s t : R)
  (h_eq_root: ∀ x, x^3 - 4 * x^2 + 4 * x - 6 = 0)
  (h1: r + s + t = 4)
  (h2: r * s + r * t + s * t = 4)
  (h3: r * s * t = 6) :
  r * s / t + s * t / r + t * r / s = -16 / 3 :=
sorry

end compute_expression_l233_233705


namespace find_a_l233_233116

-- The conditions converted to Lean definitions
variable (a : ℝ)
variable (α : ℝ)
variable (point_on_terminal_side : a ≠ 0 ∧ (∃ α, tan α = -1 / 2 ∧ ∀ y : ℝ, y = -1 → a = 2 * y) )

-- The theorem statement
theorem find_a (H : point_on_terminal_side): a = 2 := by
  sorry

end find_a_l233_233116


namespace final_concentration_of_milk_l233_233400

variable (x : ℝ) (total_vol : ℝ) (initial_milk : ℝ)
axiom x_value : x = 33.333333333333336
axiom total_volume : total_vol = 100
axiom initial_milk_vol : initial_milk = 36

theorem final_concentration_of_milk :
  let first_removal := x / total_vol * initial_milk
  let remaining_milk_after_first := initial_milk - first_removal
  let second_removal := x / total_vol * remaining_milk_after_first
  let final_milk := remaining_milk_after_first - second_removal
  (final_milk / total_vol) * 100 = 16 :=
by {
  sorry
}

end final_concentration_of_milk_l233_233400


namespace game_24_set1_game_24_set2_l233_233930

-- Equivalent proof problem for set {3, 2, 6, 7}
theorem game_24_set1 (a b c d : ℚ) (h₁ : a = 3) (h₂ : b = 2) (h₃ : c = 6) (h₄ : d = 7) :
  ((d / b) * c + a) = 24 := by
  subst_vars
  sorry

-- Equivalent proof problem for set {3, 4, -6, 10}
theorem game_24_set2 (a b c d : ℚ) (h₁ : a = 3) (h₂ : b = 4) (h₃ : c = -6) (h₄ : d = 10) :
  ((b + c + d) * a) = 24 := by
  subst_vars
  sorry

end game_24_set1_game_24_set2_l233_233930


namespace largest_possible_a_l233_233632

theorem largest_possible_a 
  (a b c d : ℕ) 
  (h1 : a < 2 * b)
  (h2 : b < 3 * c)
  (h3 : c < 2 * d)
  (h4 : d < 100) :
  a ≤ 1179 :=
sorry

end largest_possible_a_l233_233632


namespace equation_has_one_negative_and_one_zero_root_l233_233613

theorem equation_has_one_negative_and_one_zero_root :
  ∃ x y : ℝ, x < 0 ∧ y = 0 ∧ 3^x + x^2 + 2 * x - 1 = 0 ∧ 3^y + y^2 + 2 * y - 1 = 0 :=
sorry

end equation_has_one_negative_and_one_zero_root_l233_233613


namespace combined_pumps_fill_time_l233_233531

theorem combined_pumps_fill_time (small_pump_time large_pump_time : ℝ) (h1 : small_pump_time = 4) (h2 : large_pump_time = 1/2) : 
  let small_pump_rate := 1 / small_pump_time
  let large_pump_rate := 1 / large_pump_time
  let combined_rate := small_pump_rate + large_pump_rate
  (1 / combined_rate) = 4 / 9 :=
by
  -- Definitions of rates
  let small_pump_rate := 1 / small_pump_time
  let large_pump_rate := 1 / large_pump_time
  let combined_rate := small_pump_rate + large_pump_rate
  
  -- Using placeholder for the proof.
  sorry

end combined_pumps_fill_time_l233_233531


namespace acute_angle_probability_correct_l233_233956

noncomputable def acute_angle_probability (n : ℕ) (n_ge_4 : n ≥ 4) : ℝ :=
  (n * (n - 2)) / (2 ^ (n-1))

theorem acute_angle_probability_correct (n : ℕ) (h : n ≥ 4) (P : Fin n → ℝ) -- P represents points on the circle
    (uniformly_distributed : ∀ i, P i ∈ Set.Icc (0 : ℝ) 1) : 
    acute_angle_probability n h = (n * (n - 2)) / (2 ^ (n-1)) := 
  sorry

end acute_angle_probability_correct_l233_233956


namespace cost_for_Greg_l233_233773

theorem cost_for_Greg (N P M : ℝ)
(Bill : 13 * N + 26 * P + 19 * M = 25)
(Paula : 27 * N + 18 * P + 31 * M = 31) :
  24 * N + 120 * P + 52 * M = 88 := 
sorry

end cost_for_Greg_l233_233773


namespace liters_per_bottle_l233_233746

-- Condition statements
def price_per_liter : ℕ := 1
def total_cost : ℕ := 12
def num_bottles : ℕ := 6

-- Desired result statement
theorem liters_per_bottle : (total_cost / price_per_liter) / num_bottles = 2 := by
  sorry

end liters_per_bottle_l233_233746


namespace oranges_per_box_calculation_l233_233040

def total_oranges : ℕ := 2650
def total_boxes : ℕ := 265

theorem oranges_per_box_calculation (h : total_oranges % total_boxes = 0) : total_oranges / total_boxes = 10 :=
by {
  sorry
}

end oranges_per_box_calculation_l233_233040


namespace two_cards_totaling_15_probability_l233_233339

theorem two_cards_totaling_15_probability :
  let total_cards := 52
  let valid_numbers := [5, 6, 7]
  let combinations := 3 * 4 * 4 / (total_cards * (total_cards - 1))
  let prob := combinations
  prob = 8 / 442 :=
by
  sorry

end two_cards_totaling_15_probability_l233_233339


namespace product_of_two_numbers_l233_233295

theorem product_of_two_numbers (a b : ℕ) (h1 : Nat.lcm a b = 72) (h2 : Nat.gcd a b = 8) :
  a * b = 576 :=
by
  sorry

end product_of_two_numbers_l233_233295


namespace odd_function_periodic_value_l233_233820

noncomputable def f : ℝ → ℝ := sorry  -- Define f

theorem odd_function_periodic_value:
  (∀ x, f (-x) = - f x) →  -- f is odd
  (∀ x, f (x + 3) = f x) → -- f has period 3
  f 1 = 2014 →            -- given f(1) = 2014
  f 2013 + f 2014 + f 2015 = 0 := by
  intros h_odd h_period h_f1
  sorry

end odd_function_periodic_value_l233_233820


namespace cos_alpha_minus_pi_over_4_l233_233662

theorem cos_alpha_minus_pi_over_4 (α : ℝ) (hα1 : 0 < α) (hα2 : α < π / 2) (h_tan : Real.tan α = 2) :
  Real.cos (α - π / 4) = (3 * Real.sqrt 10) / 10 := 
  sorry

end cos_alpha_minus_pi_over_4_l233_233662


namespace part1_part2_l233_233278

def setA (a : ℝ) := {x : ℝ | a - 1 ≤ x ∧ x ≤ 3 - 2 * a}
def setB := {x : ℝ | x^2 - 2 * x - 8 ≤ 0}

theorem part1 (a : ℝ) : (setA a ∪ setB = setB) ↔ (-(1 / 2) ≤ a) :=
sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, x ∈ setB ↔ x ∈ setA a) ↔ (a ≤ -1) :=
sorry

end part1_part2_l233_233278


namespace graph_is_line_l233_233618

theorem graph_is_line : {p : ℝ × ℝ | (p.1 - p.2)^2 = 2 * (p.1^2 + p.2^2)} = {p : ℝ × ℝ | p.2 = -p.1} :=
by 
  sorry

end graph_is_line_l233_233618


namespace correct_statements_about_C_l233_233208

-- Conditions: Curve C is defined by the equation x^4 + y^2 = 1
def C (x y : ℝ) : Prop := x^4 + y^2 = 1

-- Prove the properties of curve C
theorem correct_statements_about_C :
  (-- 1. Symmetric about the x-axis
    (∀ x y : ℝ, C x y → C x (-y)) ∧
    -- 2. Symmetric about the y-axis
    (∀ x y : ℝ, C x y → C (-x) y) ∧
    -- 3. Symmetric about the origin
    (∀ x y : ℝ, C x y → C (-x) (-y)) ∧
    -- 6. A closed figure with an area greater than π
    (∃ (area : ℝ), area > π)) := sorry

end correct_statements_about_C_l233_233208


namespace find_n_l233_233810

theorem find_n (a b c : ℝ) (h : a^2 + b^2 = c^2) (n : ℕ) (hn : n > 2) : 
  (a^n + b^n + c^n)^2 = 2 * (a^(2*n) + b^(2*n) + c^(2*n)) → n = 4 :=
by
  sorry

end find_n_l233_233810


namespace number_of_moles_of_methanol_formed_l233_233330

def ch4_to_co2 : ℚ := 1
def o2_to_co2 : ℚ := 2
def co2_prod_from_ch4 (ch4 : ℚ) : ℚ := ch4 * ch4_to_co2 / o2_to_co2

def co2_to_ch3oh : ℚ := 1
def h2_to_ch3oh : ℚ := 3
def ch3oh_prod_from_co2 (co2 h2 : ℚ) : ℚ :=
  min (co2 / co2_to_ch3oh) (h2 / h2_to_ch3oh)

theorem number_of_moles_of_methanol_formed :
  (ch3oh_prod_from_co2 (co2_prod_from_ch4 5) 10) = 10/3 :=
by
  sorry

end number_of_moles_of_methanol_formed_l233_233330


namespace pure_gala_trees_l233_233751

theorem pure_gala_trees (T F G : ℝ) (h1 : F + 0.10 * T = 221)
  (h2 : F = 0.75 * T) : G = T - F - 0.10 * T := 
by 
  -- We define G and show it equals 39
  have eq : T = F / 0.75 := by sorry
  have G_eq : G = T - F - 0.10 * T := by sorry 
  exact G_eq

end pure_gala_trees_l233_233751


namespace probability_of_no_shaded_square_l233_233237

noncomputable def rectangles_without_shaded_square_probability : ℚ :=
  let n := 502 * 1003
  let m := 502 ^ 2
  1 - (m : ℚ) / n 

theorem probability_of_no_shaded_square : rectangles_without_shaded_square_probability = 501 / 1003 :=
  sorry

end probability_of_no_shaded_square_l233_233237


namespace abs_inequality_solution_l233_233242

theorem abs_inequality_solution (x : ℝ) :
  |x + 2| + |x - 2| ≤ 4 ↔ -2 ≤ x ∧ x ≤ 2 :=
by
  sorry

end abs_inequality_solution_l233_233242


namespace interest_problem_l233_233975

theorem interest_problem
  (P : ℝ)
  (h : P * 0.04 * 5 = P * 0.05 * 4) : 
  (P * 0.04 * 5) = 20 := 
by 
  sorry

end interest_problem_l233_233975


namespace min_value_expression_l233_233926

theorem min_value_expression (x y z : ℝ) (h1 : x > 1) (h2 : y > 1) (h3 : z > 1) : ∃ C, C = 12 ∧
  ∀ (x y z : ℝ), x > 1 → y > 1 → z > 1 → (x^2 / (y - 1) + y^2 / (z - 1) + z^2 / (x - 1)) ≥ C := by
  sorry

end min_value_expression_l233_233926


namespace tan_half_sum_pi_over_four_l233_233911

-- Define the problem conditions
variable (α : ℝ)
variable (h_cos : Real.cos α = -4 / 5)
variable (h_quad : α > π ∧ α < 3 * π / 2)

-- Define the theorem to prove
theorem tan_half_sum_pi_over_four (α : ℝ) (h_cos : Real.cos α = -4 / 5) (h_quad : α > π ∧ α < 3 * π / 2) :
  Real.tan (π / 4 + α / 2) = -1 / 2 := sorry

end tan_half_sum_pi_over_four_l233_233911


namespace parabola_directrix_is_x_eq_1_l233_233306

noncomputable def parabola_directrix (y : ℝ) : ℝ :=
  -1 / 4 * y^2

theorem parabola_directrix_is_x_eq_1 :
  ∀ x y, x = parabola_directrix y → x = 1 :=
by
  sorry

end parabola_directrix_is_x_eq_1_l233_233306


namespace hudson_daily_burger_spending_l233_233004

-- Definitions based on conditions
def total_spent := 465
def days_in_december := 31

-- Definition of the question
def amount_spent_per_day := total_spent / days_in_december

-- The theorem to prove
theorem hudson_daily_burger_spending : amount_spent_per_day = 15 := by
  sorry

end hudson_daily_burger_spending_l233_233004


namespace monkey_reach_top_in_20_hours_l233_233584

-- Defining the conditions
def tree_height : ℕ := 21
def hop_distance : ℕ := 3
def slip_distance : ℕ := 2

-- Defining the net distance gain per hour
def net_gain_per_hour : ℕ := hop_distance - slip_distance

-- Proof statement
theorem monkey_reach_top_in_20_hours :
  ∃ t : ℕ, t = 20 ∧ 20 * net_gain_per_hour + hop_distance = tree_height :=
by
  sorry

end monkey_reach_top_in_20_hours_l233_233584


namespace game_last_at_most_moves_l233_233526

theorem game_last_at_most_moves
  (n : Nat)
  (positions : Fin n → Fin (n + 1))
  (cards : Fin n → Fin (n + 1))
  (move : (k l : Fin n) → (h1 : k < l) → (h2 : k < cards k) → (positions l = cards k) → Fin n)
  : True :=
sorry

end game_last_at_most_moves_l233_233526


namespace min_pos_int_k_l233_233262

noncomputable def minimum_k (x0 : ℝ) : ℝ := (x0 * (Real.log x0 + 1)) / (x0 - 2)

theorem min_pos_int_k : ∃ k : ℝ, (∀ x0 : ℝ, x0 > 2 → k > minimum_k x0) ∧ k = 5 := 
by
  sorry

end min_pos_int_k_l233_233262


namespace scalene_triangle_process_l233_233482

theorem scalene_triangle_process (a b c : ℝ) 
  (h1: a > 0) (h2: b > 0) (h3: c > 0) 
  (h4: a + b > c) (h5: b + c > a) (h6: a + c > b) : 
  ¬(∃ k : ℝ, (k > 0) ∧ 
    ((k * a = a + b - c) ∧ 
     (k * b = b + c - a) ∧ 
     (k * c = a + c - b))) ∧ 
  (∀ n: ℕ, n > 0 → (a + b - c)^n + (b + c - a)^n + (a + c - b)^n < 1) :=
by
  sorry

end scalene_triangle_process_l233_233482


namespace sufficient_condition_a_gt_1_l233_233276

variable (a : ℝ)

theorem sufficient_condition_a_gt_1 (h : a > 1) : a^2 > 1 :=
by sorry

end sufficient_condition_a_gt_1_l233_233276


namespace fractional_eq_solution_l233_233579

theorem fractional_eq_solution (k : ℝ) :
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ 1 → (3 / x + 6 / (x - 1) - (x + k) / (x * (x - 1)) = 0)) →
  k ≠ -3 ∧ k ≠ 5 :=
by 
  sorry

end fractional_eq_solution_l233_233579


namespace correct_option_is_B_l233_233097

noncomputable def correct_calculation (x : ℝ) : Prop :=
  (x ≠ 1) → (x ≠ 0) → (x ≠ -1) → (-2 / (2 * x - 2) = 1 / (1 - x))

theorem correct_option_is_B (x : ℝ) : correct_calculation x := by
  intros hx1 hx2 hx3
  sorry

end correct_option_is_B_l233_233097


namespace factorize_x_cube_minus_9x_l233_233841

-- Lean statement: Prove that x^3 - 9x = x(x+3)(x-3)
theorem factorize_x_cube_minus_9x (x : ℝ) : x^3 - 9 * x = x * (x + 3) * (x - 3) :=
by
  sorry

end factorize_x_cube_minus_9x_l233_233841


namespace round_trip_and_car_percent_single_trip_and_motorcycle_percent_l233_233067

noncomputable def totalPassengers := 100
noncomputable def roundTripPercent := 35
noncomputable def singleTripPercent := 100 - roundTripPercent

noncomputable def roundTripCarPercent := 40
noncomputable def roundTripMotorcyclePercent := 15
noncomputable def roundTripNoVehiclePercent := 60

noncomputable def singleTripCarPercent := 25
noncomputable def singleTripMotorcyclePercent := 10
noncomputable def singleTripNoVehiclePercent := 45

theorem round_trip_and_car_percent : 
  ((roundTripCarPercent / 100) * (roundTripPercent / 100) * totalPassengers) = 14 :=
by
  sorry

theorem single_trip_and_motorcycle_percent :
  ((singleTripMotorcyclePercent / 100) * (singleTripPercent / 100) * totalPassengers) = 6 :=
by
  sorry

end round_trip_and_car_percent_single_trip_and_motorcycle_percent_l233_233067


namespace smallest_disk_cover_count_l233_233513

theorem smallest_disk_cover_count (D : ℝ) (r : ℝ) (n : ℕ) 
  (hD : D = 1) (hr : r = 1 / 2) : n = 7 :=
by
  sorry

end smallest_disk_cover_count_l233_233513


namespace find_x_l233_233489

theorem find_x (a x : ℝ) (ha : 1 < a) (hx : 0 < x)
  (h : (3 * x)^(Real.log 3 / Real.log a) - (4 * x)^(Real.log 4 / Real.log a) = 0) : 
  x = 1 / 4 := 
by 
  sorry

end find_x_l233_233489


namespace count_4_digit_numbers_divisible_by_13_l233_233334

theorem count_4_digit_numbers_divisible_by_13 : 
  let count := (9962 - 1014) / 13 + 1
  1000 ≤ 1014 ∧ 9962 ≤ 9999 →
  count = 689 :=
  by
    sorry

end count_4_digit_numbers_divisible_by_13_l233_233334


namespace additional_pass_combinations_l233_233664

def original_combinations : ℕ := 4 * 2 * 3 * 3
def new_combinations : ℕ := 6 * 2 * 4 * 3
def additional_combinations : ℕ := new_combinations - original_combinations

theorem additional_pass_combinations : additional_combinations = 72 := by
  sorry

end additional_pass_combinations_l233_233664


namespace train_around_probability_train_present_when_alex_arrives_l233_233797

noncomputable def trainArrivalTime : Set ℝ := Set.Icc 15 45
noncomputable def trainWaitTime (t : ℝ) : Set ℝ := Set.Icc t (t + 15)
noncomputable def alexArrivalTime : Set ℝ := Set.Icc 0 60

theorem train_around (t : ℝ) (h : t ∈ trainArrivalTime) :
  ∀ (x : ℝ), x ∈ alexArrivalTime → x ∈ trainWaitTime t ↔ 15 ≤ t ∧ t ≤ 45 ∧ t ≤ x ∧ x ≤ t + 15 :=
sorry

theorem probability_train_present_when_alex_arrives :
  let total_area := 60 * 60
  let favorable_area := 1 / 2 * (15 + 15) * 15
  (favorable_area / total_area) = 1 / 16 :=
sorry

end train_around_probability_train_present_when_alex_arrives_l233_233797


namespace sum_of_three_numbers_l233_233542

theorem sum_of_three_numbers 
  (a b c : ℝ) 
  (h1 : b = 10) 
  (h2 : (a + b + c) / 3 = a + 20) 
  (h3 : (a + b + c) / 3 = c - 25) : 
  a + b + c = 45 := 
by 
  sorry

end sum_of_three_numbers_l233_233542


namespace best_fitting_regression_line_l233_233935

theorem best_fitting_regression_line
  (R2_A : ℝ) (R2_B : ℝ) (R2_C : ℝ) (R2_D : ℝ)
  (h_A : R2_A = 0.27)
  (h_B : R2_B = 0.85)
  (h_C : R2_C = 0.96)
  (h_D : R2_D = 0.5) :
  R2_C = 0.96 :=
by
  -- Proof goes here
  sorry

end best_fitting_regression_line_l233_233935


namespace gran_age_indeterminate_l233_233932

theorem gran_age_indeterminate
(gran_age : ℤ) -- Let Gran's age be denoted by gran_age
(guess1 : ℤ := 75) -- The first grandchild guessed 75
(guess2 : ℤ := 78) -- The second grandchild guessed 78
(guess3 : ℤ := 81) -- The third grandchild guessed 81
-- One guess is mistaken by 1 year
(h1 : (abs (gran_age - guess1) = 1) ∨ (abs (gran_age - guess2) = 1) ∨ (abs (gran_age - guess3) = 1))
-- Another guess is mistaken by 2 years
(h2 : (abs (gran_age - guess1) = 2) ∨ (abs (gran_age - guess2) = 2) ∨ (abs (gran_age - guess3) = 2))
-- Another guess is mistaken by 4 years
(h3 : (abs (gran_age - guess1) = 4) ∨ (abs (gran_age - guess2) = 4) ∨ (abs (gran_age - guess3) = 4)) :
  False := sorry

end gran_age_indeterminate_l233_233932


namespace cubic_no_negative_roots_l233_233172

noncomputable def cubic_eq (x : ℝ) : ℝ := x^3 - 9 * x^2 + 23 * x - 15

theorem cubic_no_negative_roots {x : ℝ} : cubic_eq x = 0 → 0 ≤ x := sorry

end cubic_no_negative_roots_l233_233172


namespace inverse_proportion_graph_l233_233220

theorem inverse_proportion_graph (m n : ℝ) (h : n = -2 / m) : m = -2 / n :=
by
  sorry

end inverse_proportion_graph_l233_233220


namespace functional_equation_solution_l233_233504

theorem functional_equation_solution (f : ℝ → ℝ) (h : ∀ x y : ℝ, f (f (f x + f y)) = f x + y) : ∀ x : ℝ, f x = x :=
by
  sorry

end functional_equation_solution_l233_233504


namespace necessary_but_not_sufficient_condition_l233_233228

theorem necessary_but_not_sufficient_condition (a : ℝ) :
  (a < 2 → a^2 < 2 * a) ∧ (a^2 < 2 * a → 0 < a ∧ a < 2) :=
sorry

end necessary_but_not_sufficient_condition_l233_233228


namespace parabola_directrix_l233_233277

theorem parabola_directrix (y : ℝ) : (∃ p : ℝ, x = (1 / (4 * p)) * y^2 ∧ p = 2) → x = -2 :=
by
  sorry

end parabola_directrix_l233_233277


namespace set_equiv_l233_233384

-- Definition of the set A according to the conditions
def A : Set ℚ := { z : ℚ | ∃ p q : ℕ, z = p / (q : ℚ) ∧ p + q = 5 ∧ p > 0 ∧ q > 0 }

-- The target set we want to prove A is equal to
def target_set : Set ℚ := { 1/4, 2/3, 3/2, 4 }

-- The theorem to prove that both sets are equal
theorem set_equiv : A = target_set :=
by
  sorry -- Proof goes here

end set_equiv_l233_233384


namespace remainder_when_7x_div_9_l233_233406

theorem remainder_when_7x_div_9 (x : ℕ) (h : x % 9 = 5) : (7 * x) % 9 = 8 :=
sorry

end remainder_when_7x_div_9_l233_233406


namespace discount_savings_difference_l233_233083

def cover_price : ℝ := 30
def discount_amount : ℝ := 5
def discount_percentage : ℝ := 0.25

theorem discount_savings_difference :
  let price_after_discount := cover_price - discount_amount
  let price_after_percentage_first := cover_price * (1 - discount_percentage)
  let new_price_after_percentage := price_after_discount * (1 - discount_percentage)
  let new_price_after_discount := price_after_percentage_first - discount_amount
  (new_price_after_percentage - new_price_after_discount) * 100 = 125 :=
by
  sorry

end discount_savings_difference_l233_233083


namespace books_at_end_l233_233920

-- Define the conditions
def initialBooks : ℕ := 98
def checkoutsWednesday : ℕ := 43
def returnsThursday : ℕ := 23
def checkoutsThursday : ℕ := 5
def returnsFriday : ℕ := 7

-- Define the final number of books and the theorem to prove
def finalBooks : ℕ := initialBooks - checkoutsWednesday + returnsThursday - checkoutsThursday + returnsFriday

-- Prove that the final number of books is 80
theorem books_at_end : finalBooks = 80 := by
  sorry

end books_at_end_l233_233920


namespace market_value_of_share_l233_233772

-- Definitions from the conditions
def nominal_value : ℝ := 48
def dividend_rate : ℝ := 0.09
def desired_interest_rate : ℝ := 0.12

-- The proof problem (theorem statement) in Lean 4
theorem market_value_of_share : (nominal_value * dividend_rate / desired_interest_rate * 100) = 36 := 
by
  sorry

end market_value_of_share_l233_233772


namespace sequence_solution_l233_233308

theorem sequence_solution :
  ∃ (a : ℕ → ℕ) (b : ℕ → ℝ),
    a 1 = 2 ∧
    (∀ n, b n = (a (n + 1)) / (a n)) ∧
    b 10 * b 11 = 2 →
    a 21 = 2 ^ 11 :=
by
  sorry

end sequence_solution_l233_233308


namespace faith_earnings_correct_l233_233512

variable (pay_per_hour : ℝ) (regular_hours_per_day : ℝ) (work_days_per_week : ℝ) (overtime_hours_per_day : ℝ)
variable (overtime_rate_multiplier : ℝ)

def total_earnings (pay_per_hour : ℝ) (regular_hours_per_day : ℝ) (work_days_per_week : ℝ) 
                   (overtime_hours_per_day : ℝ) (overtime_rate_multiplier : ℝ) : ℝ :=
  let regular_hours := regular_hours_per_day * work_days_per_week
  let overtime_hours := overtime_hours_per_day * work_days_per_week
  let overtime_pay_rate := pay_per_hour * overtime_rate_multiplier
  let regular_earnings := pay_per_hour * regular_hours
  let overtime_earnings := overtime_pay_rate * overtime_hours
  regular_earnings + overtime_earnings

theorem faith_earnings_correct : 
  total_earnings 13.5 8 5 2 1.5 = 742.50 :=
by
  -- This is where the proof would go, but it's omitted as per the instructions
  sorry

end faith_earnings_correct_l233_233512


namespace magic_square_y_minus_x_l233_233331

theorem magic_square_y_minus_x :
  ∀ (x y : ℝ), 
    (x - 2 = 2 * y + y) ∧ (x - 2 = -2 + y + 6) →
    y - x = -6 :=
by 
  intros x y h
  sorry

end magic_square_y_minus_x_l233_233331


namespace other_type_jelly_amount_l233_233976

-- Combined total amount of jelly
def total_jelly := 6310

-- Amount of one type of jelly
def type_one_jelly := 4518

-- Amount of the other type of jelly
def type_other_jelly := total_jelly - type_one_jelly

theorem other_type_jelly_amount :
  type_other_jelly = 1792 :=
by
  sorry

end other_type_jelly_amount_l233_233976


namespace arithmetic_sequence_formula_l233_233115

theorem arithmetic_sequence_formula (a : ℕ → ℤ) (d : ℤ) :
  (a 3 = 4) → (d = -2) → ∀ n : ℕ, a n = 10 - 2 * n :=
by
  intros h1 h2 n
  sorry

end arithmetic_sequence_formula_l233_233115


namespace rug_overlap_area_l233_233421

theorem rug_overlap_area (A S S2 S3 : ℝ) 
  (hA : A = 200)
  (hS : S = 138)
  (hS2 : S2 = 24)
  (h1 : ∃ (S1 : ℝ), S1 + S2 + S3 = S)
  (h2 : ∃ (S1 : ℝ), S1 + 2 * S2 + 3 * S3 = A) : S3 = 19 :=
by
  sorry

end rug_overlap_area_l233_233421


namespace part1_part2_l233_233955

noncomputable def f (a x : ℝ) : ℝ := Real.log x - a * x - (a - 1) / x

theorem part1 (a : ℝ) (x : ℝ) (h1 : a ≥ 1) (h2 : x > 0) : f a x ≤ -1 :=
sorry

theorem part2 (a : ℝ) (θ : ℝ) (h1 : a ≥ 1) (h2 : 0 ≤ θ) (h3 : θ ≤ Real.pi / 2) : 
  f a (1 - Real.sin θ) ≤ f a (1 + Real.sin θ) :=
sorry

end part1_part2_l233_233955


namespace dealer_is_cheating_l233_233099

variable (w a : ℝ)
noncomputable def measured_weight (w : ℝ) (a : ℝ) : ℝ :=
  (a * w + w / a) / 2

theorem dealer_is_cheating (h : a > 0) : measured_weight w a ≥ w :=
by
  sorry

end dealer_is_cheating_l233_233099


namespace total_books_is_correct_l233_233718

-- Definitions based on the conditions
def initial_books_benny : Nat := 24
def books_given_to_sandy : Nat := 10
def books_tim : Nat := 33

-- Definition based on the computation in the solution
def books_benny_now := initial_books_benny - books_given_to_sandy
def total_books : Nat := books_benny_now + books_tim

-- The statement to be proven
theorem total_books_is_correct : total_books = 47 := by
  sorry

end total_books_is_correct_l233_233718


namespace ed_more_marbles_than_doug_initially_l233_233411

noncomputable def ed_initial_marbles := 37
noncomputable def doug_marbles := 5

theorem ed_more_marbles_than_doug_initially :
  ed_initial_marbles - doug_marbles = 32 := by
  sorry

end ed_more_marbles_than_doug_initially_l233_233411


namespace stock_status_after_limit_moves_l233_233879

theorem stock_status_after_limit_moves (initial_value : ℝ) (h₁ : initial_value = 1)
  (limit_up_factor : ℝ) (h₂ : limit_up_factor = 1 + 0.10)
  (limit_down_factor : ℝ) (h₃ : limit_down_factor = 1 - 0.10) :
  (limit_up_factor^5 * limit_down_factor^5) < initial_value :=
by
  sorry

end stock_status_after_limit_moves_l233_233879


namespace baseball_card_decrease_l233_233791

noncomputable def percentDecrease (V : ℝ) (P : ℝ) : ℝ :=
  V * (P / 100)

noncomputable def valueAfterDecrease (V : ℝ) (D : ℝ) : ℝ :=
  V - D

theorem baseball_card_decrease (V : ℝ) (H1 : V > 0) :
  let D1 := percentDecrease V 50
  let V1 := valueAfterDecrease V D1
  let D2 := percentDecrease V1 10
  let V2 := valueAfterDecrease V1 D2
  let totalDecrease := V - V2
  totalDecrease / V * 100 = 55 := sorry

end baseball_card_decrease_l233_233791


namespace distance_between_cities_l233_233793

-- Conditions Initialization
variable (A B : ℝ)
variable (S : ℕ)
variable (x : ℕ)
variable (gcd_values : ℕ)

-- Conditions:
-- 1. The distance between cities A and B is an integer.
-- 2. Markers are placed every kilometer.
-- 3. At each marker, one side has distance to city A, and the other has distance to city B.
-- 4. The GCDs observed were only 1, 3, and 13.

-- Given these conditions, we prove the distance is exactly 39 kilometers.
theorem distance_between_cities (h1 : gcd_values = 1 ∨ gcd_values = 3 ∨ gcd_values = 13)
    (h2 : S = Nat.lcm 1 (Nat.lcm 3 13)) :
    S = 39 := by 
  sorry

end distance_between_cities_l233_233793


namespace probability_red_jelly_bean_l233_233345

variable (r b g : Nat) (eaten_green eaten_blue : Nat)

theorem probability_red_jelly_bean
    (h_r : r = 15)
    (h_b : b = 20)
    (h_g : g = 16)
    (h_eaten_green : eaten_green = 1)
    (h_eaten_blue : eaten_blue = 1)
    (h_total : r + b + g = 51)
    (h_remaining_total : r + (b - eaten_blue) + (g - eaten_green) = 49) :
    (r : ℚ) / 49 = 15 / 49 :=
by
  sorry

end probability_red_jelly_bean_l233_233345


namespace positive_difference_l233_233160

theorem positive_difference (x y : ℚ) (h1 : x + y = 40) (h2 : 3 * y - 4 * x = 20) : y - x = 80 / 7 :=
by
  sorry

end positive_difference_l233_233160


namespace maria_miles_after_second_stop_l233_233523

theorem maria_miles_after_second_stop (total_distance : ℕ)
    (h1 : total_distance = 360)
    (distance_first_stop : ℕ)
    (h2 : distance_first_stop = total_distance / 2)
    (remaining_distance_after_first_stop : ℕ)
    (h3 : remaining_distance_after_first_stop = total_distance - distance_first_stop)
    (distance_second_stop : ℕ)
    (h4 : distance_second_stop = remaining_distance_after_first_stop / 4)
    (remaining_distance_after_second_stop : ℕ)
    (h5 : remaining_distance_after_second_stop = remaining_distance_after_first_stop - distance_second_stop) :
    remaining_distance_after_second_stop = 135 := by
  sorry

end maria_miles_after_second_stop_l233_233523


namespace inequality_hold_l233_233195

variables (x y : ℝ)

theorem inequality_hold (h : x^4 + y^4 ≥ 2) : |x^12 - y^12| + 2 * x^6 * y^6 ≥ 2 := 
sorry

end inequality_hold_l233_233195


namespace original_savings_l233_233916

-- Define original savings as a variable
variable (S : ℝ)

-- Define the condition that 1/4 of the savings equals 200
def tv_cost_condition : Prop := (1 / 4) * S = 200

-- State the theorem that if the condition is satisfied, then the original savings are 800
theorem original_savings (h : tv_cost_condition S) : S = 800 :=
by
  sorry

end original_savings_l233_233916


namespace part1_l233_233241

theorem part1 (f : ℝ → ℝ) (m n : ℝ) (cond1 : m + n > 0) (cond2 : ∀ x, f x = |x - m| + |x + n|) (cond3 : ∀ x, f x ≥ m + n) (minimum : ∃ x, f x = 2) :
    m + n = 2 := sorry

end part1_l233_233241


namespace divisible_by_91_l233_233847

theorem divisible_by_91 (n : ℕ) : 91 ∣ (5^n * (5^n + 1) - 6^n * (3^n + 2^n)) := 
by 
  sorry

end divisible_by_91_l233_233847


namespace parabola_max_value_l233_233987

theorem parabola_max_value 
  (y : ℝ → ℝ) 
  (h : ∀ x, y x = - (x + 1)^2 + 3) : 
  ∃ x, y x = 3 ∧ ∀ x', y x' ≤ 3 :=
by
  sorry

end parabola_max_value_l233_233987


namespace sum_of_reciprocals_of_squares_l233_233546

theorem sum_of_reciprocals_of_squares (a b : ℕ) (h : a * b = 11) :
  (1 / (a:ℚ)^2) + (1 / (b:ℚ)^2) = 122 / 121 :=
sorry

end sum_of_reciprocals_of_squares_l233_233546


namespace breakfast_cost_l233_233371

def muffin_cost : ℕ := 2
def fruit_cup_cost : ℕ := 3

def francis_muffins : ℕ := 2
def francis_fruit_cups : ℕ := 2

def kiera_muffins : ℕ := 2
def kiera_fruit_cup : ℕ := 1

theorem breakfast_cost :
  muffin_cost * francis_muffins + fruit_cup_cost * francis_fruit_cups
  + muffin_cost * kiera_muffins + fruit_cup_cost * kiera_fruit_cup = 17 :=
by
  -- skipping proof
  sorry

end breakfast_cost_l233_233371


namespace volume_of_polyhedron_l233_233735

open Real

-- Define the conditions
def square_side : ℝ := 100  -- in cm, equivalent to 1 meter
def rectangle_length : ℝ := 40  -- in cm
def rectangle_width : ℝ := 20  -- in cm
def trapezoid_leg_length : ℝ := 130  -- in cm

-- Define the question as a theorem statement
theorem volume_of_polyhedron :
  ∃ V : ℝ, V = 552 :=
sorry

end volume_of_polyhedron_l233_233735


namespace moles_of_H2O_formed_l233_233064

def NH4NO3 (n : ℕ) : Prop := n = 1
def NaOH (n : ℕ) : Prop := ∃ m : ℕ, m = n
def H2O (n : ℕ) : Prop := n = 1

theorem moles_of_H2O_formed :
  ∀ (n : ℕ), NH4NO3 1 → NaOH n → H2O 1 := 
by
  intros n hNH4NO3 hNaOH
  exact sorry

end moles_of_H2O_formed_l233_233064


namespace solution_set_l233_233474

noncomputable def truncated_interval (x : ℝ) (n : ℤ) : Prop :=
n ≤ x ∧ x < n + 1

theorem solution_set (x : ℝ) (hx : ∃ n : ℤ, n > 0 ∧ truncated_interval x n) :
  2 ≤ x ∧ x < 8 :=
sorry

end solution_set_l233_233474


namespace isosceles_trapezoid_problem_l233_233792

variable (AB CD AD BC : ℝ)
variable (x : ℝ)

noncomputable def p_squared (AB CD AD BC : ℝ) (x : ℝ) : ℝ :=
  if AB = 100 ∧ CD = 25 ∧ AD = x ∧ BC = x then 1875 else 0

theorem isosceles_trapezoid_problem (h₁ : AB = 100)
                                    (h₂ : CD = 25)
                                    (h₃ : AD = x)
                                    (h₄ : BC = x) :
  p_squared AB CD AD BC x = 1875 := by
  sorry

end isosceles_trapezoid_problem_l233_233792


namespace coordinates_of_point_l233_233615

theorem coordinates_of_point (a : ℝ) (P : ℝ × ℝ) (hy : P = (a^2 - 1, a + 1)) (hx : (a^2 - 1) = 0) :
  P = (0, 2) ∨ P = (0, 0) :=
sorry

end coordinates_of_point_l233_233615


namespace actual_price_of_good_l233_233549

theorem actual_price_of_good (P : ℝ) 
  (hp : 0.684 * P = 6500) : P = 9502.92 :=
by 
  sorry

end actual_price_of_good_l233_233549


namespace probability_of_spade_or_king_l233_233907

open Classical

-- Pack of cards containing 52 cards
def total_cards := 52

-- Number of spades in the deck
def num_spades := 13

-- Number of kings in the deck
def num_kings := 4

-- Number of overlap (king of spades)
def num_king_of_spades := 1

-- Total favorable outcomes
def total_favorable_outcomes := num_spades + num_kings - num_king_of_spades

-- Probability of drawing a spade or a king
def probability_spade_or_king := (total_favorable_outcomes : ℚ) / total_cards

theorem probability_of_spade_or_king : probability_spade_or_king = 4 / 13 := by
  sorry

end probability_of_spade_or_king_l233_233907


namespace equal_candies_l233_233493

theorem equal_candies
  (sweet_math_per_box : ℕ := 12)
  (geometry_nuts_per_box : ℕ := 15)
  (sweet_math_boxes : ℕ := 5)
  (geometry_nuts_boxes : ℕ := 4) :
  sweet_math_boxes * sweet_math_per_box = geometry_nuts_boxes * geometry_nuts_per_box := 
  by
  sorry

end equal_candies_l233_233493


namespace find_y_l233_233814

theorem find_y : ∃ y : ℚ, y + 2/3 = 1/4 - (2/5) * 2 ∧ y = -511/420 :=
by
  sorry

end find_y_l233_233814


namespace max_police_officers_needed_l233_233282

theorem max_police_officers_needed : 
  let streets := 10
  let non_parallel := true
  let curved_streets := 2
  let additional_intersections_per_curved := 3 
  streets = 10 ∧ 
  non_parallel = true ∧ 
  curved_streets = 2 ∧ 
  additional_intersections_per_curved = 3 → 
  ( (streets * (streets - 1) / 2) + (curved_streets * additional_intersections_per_curved) ) = 51 :=
by
  intros
  sorry

end max_police_officers_needed_l233_233282


namespace negation_of_prop_l233_233057

theorem negation_of_prop (P : Prop) :
  (¬ ∀ x > 0, x - 1 ≥ Real.log x) ↔ ∃ x > 0, x - 1 < Real.log x :=
by
  sorry

end negation_of_prop_l233_233057


namespace work_hours_together_l233_233063

theorem work_hours_together (t : ℚ) :
  (1 / 9) * (9 : ℚ) = 1 ∧ (1 / 12) * (12 : ℚ) = 1 ∧
  (7 / 36) * t + (1 / 9) * (15 / 4) = 1 → t = 3 :=
by
  sorry

end work_hours_together_l233_233063


namespace perpendicular_lines_condition_l233_233993

theorem perpendicular_lines_condition (m : ℝ) : (m = -1) ↔ ∀ (x y : ℝ), (x + y = 0) ∧ (x + m * y = 0) → 
  ((m ≠ 0) ∧ (-1) * (-1 / m) = 1) :=
by 
  sorry

end perpendicular_lines_condition_l233_233993


namespace chess_program_ratio_l233_233739

theorem chess_program_ratio {total_students chess_program_absent : ℕ}
  (h_total : total_students = 24)
  (h_absent : chess_program_absent = 4)
  (h_half : chess_program_absent * 2 = chess_program_absent + chess_program_absent) :
  (chess_program_absent * 2 : ℚ) / total_students = 1 / 3 :=
by
  sorry

end chess_program_ratio_l233_233739


namespace find_constants_monotonicity_l233_233185

noncomputable def f (x a b : ℝ) := (x^2 + a * x) * Real.exp x + b

theorem find_constants (a b : ℝ) (h_tangent : (f 0 a b = 1) ∧ (deriv (f · a b) 0 = -2)) :
  a = -2 ∧ b = 1 := by
  sorry

theorem monotonicity (a b : ℝ) (h_constants : a = -2 ∧ b = 1) :
  (∀ x : ℝ, (Real.exp x * (x^2 - 2) > 0 → x > Real.sqrt 2 ∨ x < -Real.sqrt 2)) ∧
  (∀ x : ℝ, (Real.exp x * (x^2 - 2) < 0 → -Real.sqrt 2 < x ∧ x < Real.sqrt 2)) := by
  sorry

end find_constants_monotonicity_l233_233185


namespace circle_center_l233_233960

theorem circle_center : ∃ (a b : ℝ), (∀ x y : ℝ, x^2 + y^2 - 2 * x - 4 * y - 4 = 0 ↔ (x - a)^2 + (y - b)^2 = 9) ∧ a = 1 ∧ b = 2 :=
sorry

end circle_center_l233_233960


namespace count_perfect_squares_l233_233000

theorem count_perfect_squares (N : Nat) :
  ∃ k : Nat, k = 1666 ∧ ∀ m, (∃ n, m = n * n ∧ m < 10^8 ∧ 36 ∣ m) ↔ (m = 36 * k ^ 2 ∧ k < 10^4) :=
sorry

end count_perfect_squares_l233_233000


namespace total_limes_picked_l233_233681

def Fred_limes : ℕ := 36
def Alyssa_limes : ℕ := 32
def Nancy_limes : ℕ := 35
def David_limes : ℕ := 42
def Eileen_limes : ℕ := 50

theorem total_limes_picked :
  Fred_limes + Alyssa_limes + Nancy_limes + David_limes + Eileen_limes = 195 :=
by
  sorry

end total_limes_picked_l233_233681


namespace quadratic_root_expression_l233_233959

theorem quadratic_root_expression (a b : ℝ) 
  (h : ∀ x : ℝ, x^2 + x - 2023 = 0 → (x = a ∨ x = b)) 
  (ha_neq_b : a ≠ b) :
  a^2 + 2*a + b = 2022 :=
sorry

end quadratic_root_expression_l233_233959


namespace min_sum_a_b_l233_233008

theorem min_sum_a_b {a b : ℝ} (h₀ : 0 < a) (h₁ : 0 < b)
  (h₂ : 1/a + 9/b = 1) : a + b ≥ 16 := 
sorry

end min_sum_a_b_l233_233008


namespace stamps_per_book_type2_eq_15_l233_233159

-- Defining the conditions
def num_books_type1 : ℕ := 4
def stamps_per_book_type1 : ℕ := 10
def num_books_type2 : ℕ := 6
def total_stamps : ℕ := 130

-- Stating the theorem to prove the number of stamps in each book of the second type is 15
theorem stamps_per_book_type2_eq_15 : 
  ∀ (x : ℕ), 
    (num_books_type1 * stamps_per_book_type1 + num_books_type2 * x = total_stamps) → 
    x = 15 :=
by
  sorry

end stamps_per_book_type2_eq_15_l233_233159


namespace fraction_squared_0_0625_implies_value_l233_233240

theorem fraction_squared_0_0625_implies_value (x : ℝ) (hx : x^2 = 0.0625) : x = 0.25 :=
sorry

end fraction_squared_0_0625_implies_value_l233_233240


namespace correct_propositions_l233_233131

namespace ProofProblem

-- Define Curve C
def curve_C (x y t : ℝ) : Prop :=
  (x^2 / (4 - t)) + (y^2 / (t - 1)) = 1

-- Proposition ①
def proposition_1 (t : ℝ) : Prop :=
  ¬(1 < t ∧ t < 4 ∧ t ≠ 5 / 2)

-- Proposition ②
def proposition_2 (t : ℝ) : Prop :=
  t > 4 ∨ t < 1

-- Proposition ③
def proposition_3 (t : ℝ) : Prop :=
  t ≠ 5 / 2

-- Proposition ④
def proposition_4 (t : ℝ) : Prop :=
  1 < t ∧ t < (5 / 2)

-- The theorem we need to prove
theorem correct_propositions (t : ℝ) :
  (proposition_1 t = false) ∧
  (proposition_2 t = true) ∧
  (proposition_3 t = false) ∧
  (proposition_4 t = true) :=
by
  sorry

end ProofProblem

end correct_propositions_l233_233131


namespace remainder_division_1614_254_eq_90_l233_233881

theorem remainder_division_1614_254_eq_90 :
  ∀ (x : ℕ) (R : ℕ),
    1614 - x = 1360 →
    x * 6 + R = 1614 →
    0 ≤ R →
    R < x →
    R = 90 := 
by
  intros x R h_diff h_div h_nonneg h_lt
  sorry

end remainder_division_1614_254_eq_90_l233_233881


namespace probability_club_then_spade_l233_233714

/--
   Two cards are dealt at random from a standard deck of 52 cards.
   Prove that the probability that the first card is a club (♣) and the second card is a spade (♠) is 13/204.
-/
theorem probability_club_then_spade :
  let total_cards := 52
  let clubs := 13
  let spades := 13
  let first_card_club_prob := (clubs : ℚ) / total_cards
  let second_card_spade_prob := (spades : ℚ) / (total_cards - 1)
  first_card_club_prob * second_card_spade_prob = 13 / 204 :=
by
  sorry

end probability_club_then_spade_l233_233714


namespace non_adjective_primes_sum_l233_233807

-- We will define the necessary components as identified from our problem

def is_adjective_prime (p : ℕ) [Fact (Nat.Prime p)] : Prop :=
  ∃ a : ℕ → ℕ, ∀ n : ℕ,
    a 0 % p = (1 + (1 / a 1) % p) ∧
    a 1 % p = (1 + (1 / (1 + (1 / a 2) % p)) % p) ∧
    a 2 % p = (1 + (1 / (1 + (1 / (1 + (1 / a 3) % p))) % p))

def is_not_adjective_prime (p : ℕ) [Fact (Nat.Prime p)] : Prop :=
  ¬ is_adjective_prime p

def first_three_non_adjective_primes_sum : ℕ :=
  3 + 7 + 23

theorem non_adjective_primes_sum :
  first_three_non_adjective_primes_sum = 33 := 
  sorry

end non_adjective_primes_sum_l233_233807


namespace common_chord_length_l233_233003

theorem common_chord_length (x y : ℝ) : 
    (x^2 + y^2 = 4) → 
    (x^2 + y^2 - 4*x + 4*y - 12 = 0) → 
    ∃ l : ℝ, l = 2 * Real.sqrt 2 :=
by
  intros h1 h2
  sorry

end common_chord_length_l233_233003


namespace cylindrical_coords_of_point_l233_233412

theorem cylindrical_coords_of_point :
  ∃ (r θ z : ℝ), r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧
                 r = Real.sqrt (3^2 + 3^2) ∧
                 θ = Real.arctan (3 / 3) ∧
                 z = 4 ∧
                 (3, 3, 4) = (r * Real.cos θ, r * Real.sin θ, z) :=
by
  sorry

end cylindrical_coords_of_point_l233_233412


namespace boxes_remaining_to_sell_l233_233085

-- Define the conditions
def first_customer_boxes : ℕ := 5 
def second_customer_boxes : ℕ := 4 * first_customer_boxes
def third_customer_boxes : ℕ := second_customer_boxes / 2
def fourth_customer_boxes : ℕ := 3 * third_customer_boxes
def final_customer_boxes : ℕ := 10
def sales_goal : ℕ := 150

-- Total boxes sold
def total_boxes_sold : ℕ := first_customer_boxes + second_customer_boxes + third_customer_boxes + fourth_customer_boxes + final_customer_boxes

-- Boxes left to sell to hit the sales goal
def boxes_left_to_sell : ℕ := sales_goal - total_boxes_sold

-- Prove the number of boxes left to sell is 75
theorem boxes_remaining_to_sell : boxes_left_to_sell = 75 :=
by
  -- Step to prove goes here
  sorry

end boxes_remaining_to_sell_l233_233085


namespace petrol_price_increase_l233_233336

theorem petrol_price_increase
  (P P_new : ℝ)
  (C : ℝ)
  (h1 : P * C = P_new * (C * 0.7692307692307693))
  (h2 : C * (1 - 0.23076923076923073) = C * 0.7692307692307693) :
  ((P_new - P) / P) * 100 = 30 := 
  sorry

end petrol_price_increase_l233_233336


namespace calculate_dividend_l233_233931

def divisor : ℕ := 21
def quotient : ℕ := 14
def remainder : ℕ := 7
def expected_dividend : ℕ := 301

theorem calculate_dividend : (divisor * quotient + remainder = expected_dividend) := 
by
  sorry

end calculate_dividend_l233_233931


namespace distinct_positive_integer_triplets_l233_233403

theorem distinct_positive_integer_triplets (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) (hprod : a * b * c = 72^3) : 
  ∃ n, n = 1482 :=
by
  sorry

end distinct_positive_integer_triplets_l233_233403


namespace proof_problem_l233_233059

variable {R : Type*} [LinearOrderedField R]

theorem proof_problem 
  (a1 a2 a3 b1 b2 b3 : R)
  (h1 : a1 < a2) (h2 : a2 < a3) (h3 : b1 < b2) (h4 : b2 < b3)
  (h_sum : a1 + a2 + a3 = b1 + b2 + b3)
  (h_pair_sum : a1 * a2 + a1 * a3 + a2 * a3 = b1 * b2 + b1 * b3 + b2 * b3)
  (h_a1_lt_b1 : a1 < b1) :
  (b2 < a2) ∧ (a3 < b3) ∧ (a1 * a2 * a3 < b1 * b2 * b3) ∧ ((1 - a1) * (1 - a2) * (1 - a3) > (1 - b1) * (1 - b2) * (1 - b3)) :=
by {
  sorry
}

end proof_problem_l233_233059


namespace find_a_b_sum_l233_233202

def star (a b : ℕ) : ℕ := a^b + a * b

theorem find_a_b_sum (a b : ℕ) (h1 : 2 ≤ a) (h2 : 2 ≤ b) (h3 : star a b = 24) : a + b = 6 :=
  sorry

end find_a_b_sum_l233_233202


namespace no_such_n_exists_l233_233520

noncomputable def is_partitionable (s : Finset ℕ) : Prop :=
  ∃ (A B : Finset ℕ), A ∪ B = s ∧ A ∩ B = ∅ ∧ (A.prod id = B.prod id)

theorem no_such_n_exists :
  ¬ ∃ n : ℕ, n > 0 ∧ is_partitionable {n, n+1, n+2, n+3, n+4, n+5} :=
by
  sorry

end no_such_n_exists_l233_233520


namespace abs_neg_four_squared_plus_six_l233_233982

theorem abs_neg_four_squared_plus_six : |(-4^2 + 6)| = 10 := by
  -- We skip the proof steps according to the instruction
  sorry

end abs_neg_four_squared_plus_six_l233_233982


namespace distance_to_valley_l233_233627

theorem distance_to_valley (car_speed_kph : ℕ) (time_seconds : ℕ) (sound_speed_mps : ℕ) 
  (car_speed_mps : ℕ) (distance_by_car : ℕ) (distance_by_sound : ℕ) 
  (total_distance_equation : 2 * x + distance_by_car = distance_by_sound) : x = 640 :=
by
  have car_speed_kph := 72
  have time_seconds := 4
  have sound_speed_mps := 340
  have car_speed_mps := car_speed_kph * 1000 / 3600
  have distance_by_car := time_seconds * car_speed_mps
  have distance_by_sound := time_seconds * sound_speed_mps
  have total_distance_equation := (2 * x + distance_by_car = distance_by_sound)
  exact sorry

end distance_to_valley_l233_233627


namespace range_of_a_l233_233216

theorem range_of_a (a : ℝ) : ((1 - a)^2 + (1 + a)^2 < 4) ↔ (-1 < a ∧ a < 1) := 
sorry

end range_of_a_l233_233216


namespace totalMoney_l233_233300

noncomputable def totalAmount (x : ℝ) : ℝ := 15 * x

theorem totalMoney (x : ℝ) (h : 1.8 * x = 9) : totalAmount x = 75 :=
by sorry

end totalMoney_l233_233300


namespace hollow_circles_in_2001_pattern_l233_233326

theorem hollow_circles_in_2001_pattern :
  let pattern_length := 9
  let hollow_in_pattern := 3
  let total_circles := 2001
  let complete_patterns := total_circles / pattern_length
  let remaining_circles := total_circles % pattern_length
  let hollow_in_remaining := if remaining_circles >= 3 then 1 else 0
  let total_hollow := complete_patterns * hollow_in_pattern + hollow_in_remaining
  total_hollow = 667 :=
by
  sorry

end hollow_circles_in_2001_pattern_l233_233326


namespace range_of_a_l233_233349

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, ax^2 + ax + 3 > 0) ↔ (0 ≤ a ∧ a < 12) :=
by
  sorry

end range_of_a_l233_233349


namespace M_even_comp_M_composite_comp_M_prime_not_div_l233_233805

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_composite (n : ℕ) : Prop :=  ∃ d : ℕ, d > 1 ∧ d < n ∧ d ∣ n
def M (n : ℕ) : ℕ := 2^n - 1

theorem M_even_comp (n : ℕ) (h1 : n ≠ 2) (h2 : is_even n) : is_composite (M n) :=
sorry

theorem M_composite_comp (n : ℕ) (h : is_composite n) : is_composite (M n) :=
sorry

theorem M_prime_not_div (p : ℕ) (h : Nat.Prime p) : ¬ (p ∣ M p) :=
sorry

end M_even_comp_M_composite_comp_M_prime_not_div_l233_233805


namespace smallest_positive_period_of_f_minimum_value_of_f_in_interval_l233_233315

noncomputable def vec_a (x : ℝ) : ℝ × ℝ := (Real.cos (2 * x), Real.sin (2 * x))
noncomputable def vec_b : ℝ × ℝ := (Real.sqrt 3, 1)
noncomputable def f (x m : ℝ) : ℝ := (vec_a x).1 * vec_b.1 + (vec_a x).2 * vec_b.2 + m

theorem smallest_positive_period_of_f :
  ∀ (x : ℝ) (m : ℝ), ∀ p : ℝ, p > 0 → (∀ x : ℝ, f (x + p) m = f x m) → p = Real.pi := 
sorry

theorem minimum_value_of_f_in_interval :
  ∀ (x m : ℝ), (0 ≤ x ∧ x ≤ Real.pi / 2) → ∃ m : ℝ, (∀ x : ℝ, f x m ≥ 5) ∧ m = 5 + Real.sqrt 3 :=
sorry

end smallest_positive_period_of_f_minimum_value_of_f_in_interval_l233_233315


namespace how_many_buckets_did_Eden_carry_l233_233978

variable (E : ℕ) -- Natural number representing buckets Eden carried
variable (M : ℕ) -- Natural number representing buckets Mary carried
variable (I : ℕ) -- Natural number representing buckets Iris carried

-- Conditions based on the problem
axiom Mary_Carry_More : M = E + 3
axiom Iris_Carry_Less : I = M - 1
axiom Total_Buckets : E + M + I = 34

theorem how_many_buckets_did_Eden_carry (h1 : M = E + 3) (h2 : I = M - 1) (h3 : E + M + I = 34) :
  E = 29 / 3 := by
  sorry

end how_many_buckets_did_Eden_carry_l233_233978


namespace sqrt_x_plus_sqrt_inv_x_l233_233121

theorem sqrt_x_plus_sqrt_inv_x (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = 50) : 
  (Real.sqrt x + 1 / Real.sqrt x) = Real.sqrt 52 := 
by
  sorry

end sqrt_x_plus_sqrt_inv_x_l233_233121


namespace shanghai_expo_visitors_l233_233686

theorem shanghai_expo_visitors :
  505000 = 5.05 * 10^5 :=
by
  sorry

end shanghai_expo_visitors_l233_233686


namespace equation_proof_l233_233971

theorem equation_proof :
  (40 + 5 * 12) / (180 / 3^2) + Real.sqrt 49 = 12 := 
by 
  sorry

end equation_proof_l233_233971


namespace g_h_2_equals_584_l233_233140

def g (x : ℝ) : ℝ := 3 * x^2 - 4
def h (x : ℝ) : ℝ := -2 * x^3 + 2

theorem g_h_2_equals_584 : g (h 2) = 584 := 
by 
  sorry

end g_h_2_equals_584_l233_233140


namespace remainder_when_divided_by_15_l233_233137

theorem remainder_when_divided_by_15 (N : ℕ) (h1 : N % 60 = 49) : N % 15 = 4 :=
by
  sorry

end remainder_when_divided_by_15_l233_233137


namespace find_y_values_l233_233789

theorem find_y_values (x : ℝ) (h : x^2 + 9 * (x / (x - 3))^2 = 90) :
  ∃ y, (y = 0 ∨ y = 144 ∨ y = -24) ∧ y = (x - 3)^2 * (x + 4) / (2 * x - 5) :=
by sorry

end find_y_values_l233_233789


namespace vertex_in_first_quadrant_l233_233426

theorem vertex_in_first_quadrant (a : ℝ) (h : a > 1) : 
  let x_vertex := (a + 1) / 2
  let y_vertex := (a + 3)^2 / 4
  x_vertex > 0 ∧ y_vertex > 0 := 
by
  sorry

end vertex_in_first_quadrant_l233_233426


namespace tomato_puree_water_percentage_l233_233373

theorem tomato_puree_water_percentage :
  (∀ (juice_purity water_percentage : ℝ), 
    (juice_purity = 0.90) → 
    (20 * juice_purity = 18) →
    (2.5 - 2) = 0.5 →
    (2.5 * water_percentage - 0.5) = 0 →
    water_percentage = 0.20) :=
by
  intros juice_purity water_percentage h1 h2 h3 h4
  sorry

end tomato_puree_water_percentage_l233_233373


namespace min_blocks_for_wall_l233_233828

-- Definitions based on conditions
def length_of_wall := 120
def height_of_wall := 6
def block_height := 1
def block_lengths := [1, 3]
def blocks_third_row := 3

-- Function to calculate the total blocks given the constraints from the conditions
noncomputable def min_blocks_needed : Nat := 164 + 80

-- Theorem assertion that the minimum number of blocks required is 244
theorem min_blocks_for_wall : min_blocks_needed = 244 := by
  -- The proof would go here
  sorry

end min_blocks_for_wall_l233_233828


namespace even_diagonal_moves_l233_233158

def King_Moves (ND D : ℕ) :=
  ND + D = 63 ∧ ND % 2 = 0

theorem even_diagonal_moves (ND D : ℕ) (traverse_board : King_Moves ND D) : D % 2 = 0 :=
by
  sorry

end even_diagonal_moves_l233_233158


namespace Marc_watch_episodes_l233_233527

theorem Marc_watch_episodes : ∀ (episodes per_day : ℕ), episodes = 50 → per_day = episodes / 10 → (episodes / per_day) = 10 :=
by
  intros episodes per_day h1 h2
  sorry

end Marc_watch_episodes_l233_233527


namespace inequality_solution_set_range_of_m_l233_233890

-- Proof Problem 1
theorem inequality_solution_set :
  {x : ℝ | -2 < x ∧ x < 4} = { x : ℝ | 2 * x^2 - 4 * x - 16 < 0 } :=
sorry

-- Proof Problem 2
theorem range_of_m (f : ℝ → ℝ) (m : ℝ) :
  (∀ x > 2, f x ≥ (m + 2) * x - m - 15) ↔ m ≤ 2 :=
  sorry

end inequality_solution_set_range_of_m_l233_233890


namespace middle_number_is_eight_l233_233844

theorem middle_number_is_eight
    (x y z : ℕ)
    (h1 : x + y = 14)
    (h2 : x + z = 20)
    (h3 : y + z = 22) :
    y = 8 := by
  sorry

end middle_number_is_eight_l233_233844


namespace cars_meet_first_time_l233_233049

-- Definitions based on conditions
def car (t : ℕ) (v : ℕ) : ℕ := t * v
def car_meet (t : ℕ) (v1 v2 : ℕ) : Prop := ∃ n, v1 * t + v2 * t = n

-- Given conditions
variables (v_A v_B v_C v_D : ℕ) (pairwise_different : v_A ≠ v_B ∧ v_B ≠ v_C ∧ v_C ≠ v_D ∧ v_D ≠ v_A)
variables (t1 t2 t3 : ℕ) (time_AC : t1 = 7) (time_BD : t1 = 7) (time_AB : t2 = 53)
variables (condition1 : car_meet t1 v_A v_C) (condition2 : car_meet t1 v_B v_D)
variables (condition3 : ∃ k, (v_A - v_B) * t2 = k)

-- Theorem statement
theorem cars_meet_first_time : ∃ t, (t = 371) := sorry

end cars_meet_first_time_l233_233049


namespace complex_number_pure_imaginary_l233_233402

theorem complex_number_pure_imaginary (a : ℝ) 
  (h1 : ∃ a : ℝ, (a^2 - 2*a - 3 = 0) ∧ (a + 1 ≠ 0)) 
  : a = 3 := sorry

end complex_number_pure_imaginary_l233_233402


namespace find_dividend_l233_233846

noncomputable def divisor := (-14 : ℚ) / 3
noncomputable def quotient := (-286 : ℚ) / 5
noncomputable def remainder := (19 : ℚ) / 9
noncomputable def dividend := 269 + (2 / 45 : ℚ)

theorem find_dividend :
  dividend = (divisor * quotient) + remainder := by
  sorry

end find_dividend_l233_233846


namespace smallest_apples_l233_233353

theorem smallest_apples (A : ℕ) (h1 : A % 9 = 2) (h2 : A % 10 = 2) (h3 : A % 11 = 2) (h4 : A > 2) : A = 992 :=
sorry

end smallest_apples_l233_233353


namespace trigonometric_expression_l233_233780

theorem trigonometric_expression (x : ℝ) (h : Real.tan x = -1/2) : 
  Real.sin x ^ 2 + 3 * Real.sin x * Real.cos x - 1 = -2 := 
sorry

end trigonometric_expression_l233_233780


namespace find_R_value_l233_233292

noncomputable def x (Q : ℝ) : ℝ := Real.sqrt (Q / 2 + Real.sqrt (Q / 2))
noncomputable def y (Q : ℝ) : ℝ := Real.sqrt (Q / 2 - Real.sqrt (Q / 2))
noncomputable def R (Q : ℝ) : ℝ := (x Q)^6 + (y Q)^6 / 40

theorem find_R_value (Q : ℝ) : R Q = 10 :=
sorry

end find_R_value_l233_233292


namespace P_3_eq_seven_eighths_P_4_ne_fifteen_sixteenths_P_decreasing_P_recurrence_l233_233568

open ProbabilityTheory

section
/-- Probability of not getting three consecutive heads -/
def P (n : ℕ) : ℚ := sorry

theorem P_3_eq_seven_eighths : P 3 = 7 / 8 := sorry

theorem P_4_ne_fifteen_sixteenths : P 4 ≠ 15 / 16 := sorry

theorem P_decreasing (n : ℕ) (h : 2 ≤ n) : P (n + 1) < P n := sorry

theorem P_recurrence (n : ℕ) (h : 4 ≤ n) : P n = (1 / 2) * P (n - 1) + (1 / 4) * P (n - 2) + (1 / 8) * P (n - 3) := sorry
end

end P_3_eq_seven_eighths_P_4_ne_fifteen_sixteenths_P_decreasing_P_recurrence_l233_233568


namespace sum_of_variables_l233_233974

theorem sum_of_variables (x y z : ℝ) (h : x^2 + y^2 + z^2 - 2*x + 4*y - 6*z + 14 = 0) : 
  x + y + z = 2 :=
sorry

end sum_of_variables_l233_233974


namespace most_likely_number_of_red_balls_l233_233733

-- Define the total number of balls and the frequency of picking red balls as given in the conditions
def total_balls : ℕ := 20
def frequency_red : ℝ := 0.8

-- State the equivalent proof problem: Prove that the most likely number of red balls is 16
theorem most_likely_number_of_red_balls : frequency_red * (total_balls : ℝ) = 16 := by
  sorry

end most_likely_number_of_red_balls_l233_233733


namespace watermelon_sales_correct_l233_233702

def total_watermelons_sold 
  (customers_one_melon : ℕ) 
  (customers_three_melons : ℕ) 
  (customers_two_melons : ℕ) : ℕ :=
  (customers_one_melon * 1) + (customers_three_melons * 3) + (customers_two_melons * 2)

theorem watermelon_sales_correct :
  total_watermelons_sold 17 3 10 = 46 := by
  sorry

end watermelon_sales_correct_l233_233702


namespace sequence_term_condition_l233_233560

theorem sequence_term_condition (n : ℕ) : (n^2 - 8 * n + 15 = 3) ↔ (n = 2 ∨ n = 6) :=
by 
  sorry

end sequence_term_condition_l233_233560


namespace nth_term_is_4037_l233_233533

noncomputable def arithmetic_sequence_nth_term (n : ℕ) : ℤ :=
7 + (n - 1) * 6

theorem nth_term_is_4037 {n : ℕ} : arithmetic_sequence_nth_term 673 = 4037 :=
by
  sorry

end nth_term_is_4037_l233_233533


namespace ribbon_per_box_l233_233272

theorem ribbon_per_box (ribbon_total ribbon_each : ℚ) (n : ℕ) (hn : n = 5) (h : ribbon_total = 5 / 12) :
  ribbon_each = ribbon_total / n ↔ ribbon_each = 1 / 12 :=
by
  sorry

end ribbon_per_box_l233_233272


namespace ivan_years_l233_233068

theorem ivan_years (years months weeks days hours : ℕ) (h1 : years = 48) (h2 : months = 48)
    (h3 : weeks = 48) (h4 : days = 48) (h5 : hours = 48) :
    (53 : ℕ) = (years + (months / 12) + ((weeks * 7 + days) / 365) + ((hours / 24) / 365)) := by
  sorry

end ivan_years_l233_233068


namespace box_office_collection_l233_233233

open Nat

/-- Define the total tickets sold -/
def total_tickets : ℕ := 1500

/-- Define the price of an adult ticket -/
def price_adult_ticket : ℕ := 12

/-- Define the price of a student ticket -/
def price_student_ticket : ℕ := 6

/-- Define the number of student tickets sold -/
def student_tickets : ℕ := 300

/-- Define the number of adult tickets sold -/
def adult_tickets : ℕ := total_tickets - student_tickets

/-- Define the revenue from adult tickets -/
def revenue_adult_tickets : ℕ := adult_tickets * price_adult_ticket

/-- Define the revenue from student tickets -/
def revenue_student_tickets : ℕ := student_tickets * price_student_ticket

/-- Define the total amount collected -/
def total_amount_collected : ℕ := revenue_adult_tickets + revenue_student_tickets

/-- Theorem to prove the total amount collected at the box office -/
theorem box_office_collection : total_amount_collected = 16200 := by
  sorry

end box_office_collection_l233_233233


namespace num_children_got_off_l233_233737

-- Define the original number of children on the bus
def original_children : ℕ := 43

-- Define the number of children left after some got off the bus
def children_left : ℕ := 21

-- Define the number of children who got off the bus as the difference between original_children and children_left
def children_got_off : ℕ := original_children - children_left

-- State the theorem that the number of children who got off the bus is 22
theorem num_children_got_off : children_got_off = 22 :=
by
  -- Proof steps would go here, but are omitted
  sorry

end num_children_got_off_l233_233737


namespace distance_AK_l233_233779

noncomputable def A : ℝ × ℝ := (0, 0)
noncomputable def B : ℝ × ℝ := (0, -1)
noncomputable def C : ℝ × ℝ := (1, 0)
noncomputable def D : ℝ × ℝ := (Real.sqrt 2 / 2, Real.sqrt 2 / 2)

-- Define the line equations
noncomputable def line_AB (x : ℝ) : Prop := x = 0
noncomputable def line_CD (x y : ℝ) : Prop := y = (Real.sqrt 2) / (2 - Real.sqrt 2) * (x - 1)

-- Define the intersection point K
noncomputable def K : ℝ × ℝ := (0, -(Real.sqrt 2 + 1))

-- Define the distance function
noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

-- Prove the desired distance
theorem distance_AK : distance A K = Real.sqrt 2 + 1 :=
by
  -- Proof details are omitted
  sorry

end distance_AK_l233_233779


namespace general_term_of_sequence_l233_233795

noncomputable def harmonic_mean {n : ℕ} (p : Fin n → ℝ) : ℝ :=
  n / (Finset.univ.sum (fun i => p i))

theorem general_term_of_sequence (a : ℕ → ℝ) (h : ∀ n : ℕ, harmonic_mean (fun i : Fin n => a (i + 1)) = 1 / (2 * n - 1))
    (h₂ : ∀ n : ℕ, (Finset.range n).sum a = 2 * n^2 - n) :
  ∀ n : ℕ, a n = 4 * n - 3 := by
  sorry

end general_term_of_sequence_l233_233795


namespace solve_for_x_l233_233023

theorem solve_for_x (x : ℚ) : (2/5 : ℚ) - (1/4 : ℚ) = 1/x → x = 20/3 :=
by
  intro h
  sorry

end solve_for_x_l233_233023


namespace pentagon_triangle_area_percentage_l233_233380

def is_equilateral_triangle (s : ℝ) (area : ℝ) : Prop :=
  area = (s^2 * Real.sqrt 3) / 4

def is_square (s : ℝ) (area : ℝ) : Prop :=
  area = s^2

def pentagon_area (square_area triangle_area : ℝ) : ℝ :=
  square_area + triangle_area

noncomputable def percentage (triangle_area pentagon_area : ℝ) : ℝ :=
  (triangle_area / pentagon_area) * 100

theorem pentagon_triangle_area_percentage (s : ℝ) (h₁ : s > 0) :
  let square_area := s^2
  let triangle_area := (s^2 * Real.sqrt 3) / 4
  let pentagon_total_area := pentagon_area square_area triangle_area
  let triangle_percentage := percentage triangle_area pentagon_total_area
  triangle_percentage = (100 * (4 * Real.sqrt 3 - 3) / 13) :=
by
  sorry

end pentagon_triangle_area_percentage_l233_233380


namespace Mr_Kishore_saved_10_percent_l233_233943

-- Define the costs and savings
def rent : ℕ := 5000
def milk : ℕ := 1500
def groceries : ℕ := 4500
def education : ℕ := 2500
def petrol : ℕ := 2000
def miscellaneous : ℕ := 6100
def savings : ℕ := 2400

-- Define the total expenses
def total_expenses : ℕ := rent + milk + groceries + education + petrol + miscellaneous

-- Define the total monthly salary
def total_monthly_salary : ℕ := total_expenses + savings

-- Define the percentage saved
def percentage_saved : ℕ := (savings * 100) / total_monthly_salary

-- The statement to prove
theorem Mr_Kishore_saved_10_percent : percentage_saved = 10 := by
  sorry

end Mr_Kishore_saved_10_percent_l233_233943


namespace average_eq_solution_l233_233324

theorem average_eq_solution (x : ℝ) :
  (1 / 3) * ((2 * x + 4) + (4 * x + 6) + (5 * x + 3)) = 3 * x + 5 → x = 1 :=
by
  sorry

end average_eq_solution_l233_233324


namespace evaluate_expression_l233_233107

theorem evaluate_expression : 4 * 6 * 8 + 24 / 4 - 2 = 196 := by
  sorry

end evaluate_expression_l233_233107


namespace abs_of_negative_l233_233902

theorem abs_of_negative (a : ℝ) (h : a < 0) : |a| = -a :=
sorry

end abs_of_negative_l233_233902


namespace correct_calculation_l233_233858

theorem correct_calculation (x : ℝ) : (-2 * x^2)^3 = -8 * x^6 :=
by sorry

end correct_calculation_l233_233858


namespace problem_a_problem_b_problem_c_problem_d_l233_233393

-- Problem a
theorem problem_a (a : ℝ) : (a + 1) * (a - 1) = a^2 - 1 :=
by sorry

-- Problem b
theorem problem_b (a : ℝ) : (2 * a + 3) * (2 * a - 3) = 4 * a^2 - 9 :=
by sorry

-- Problem c
theorem problem_c (m n : ℝ) : (m^3 - n^5) * (n^5 + m^3) = m^6 - n^10 :=
by sorry

-- Problem d
theorem problem_d (m n : ℝ) : (3 * m^2 - 5 * n^2) * (3 * m^2 + 5 * n^2) = 9 * m^4 - 25 * n^4 :=
by sorry

end problem_a_problem_b_problem_c_problem_d_l233_233393


namespace g_of_1_equals_3_l233_233880

theorem g_of_1_equals_3 (f g : ℝ → ℝ)
  (hf_odd : ∀ x, f (-x) = -f x)
  (hg_even : ∀ x, g (-x) = g x)
  (h1 : f (-1) + g 1 = 2)
  (h2 : f 1 + g (-1) = 4) :
  g 1 = 3 :=
sorry

end g_of_1_equals_3_l233_233880


namespace Bills_age_proof_l233_233165

variable {b t : ℚ}

theorem Bills_age_proof (h1 : b = 4 * t / 3) (h2 : b + 30 = 9 * (t + 30) / 8) : b = 24 := by 
  sorry

end Bills_age_proof_l233_233165


namespace value_of_k_l233_233938

theorem value_of_k (x k : ℝ) (h : (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 8)) (hk : k ≠ 0) : k = 8 :=
sorry

end value_of_k_l233_233938


namespace proof_problem_l233_233439

noncomputable def problem_statement (m : ℕ) : Prop :=
  ∀ pairs : List (ℕ × ℕ),
  (∀ (x y : ℕ), (x, y) ∈ pairs ↔ x^2 - 3 * y^2 + 2 = 16 * m ∧ 2 * y ≤ x - 1) →
  pairs.length % 2 = 0 ∨ pairs.length = 0

theorem proof_problem (m : ℕ) (hm : m > 0) : problem_statement m :=
by
  sorry

end proof_problem_l233_233439


namespace distance_from_circle_center_to_line_l233_233390

-- Define the center of the circle
def circle_center : ℝ × ℝ := (1, -2)

-- Define the equation of the line
def line_eq (x y : ℝ) : ℝ := 2 * x + y - 5

-- Define the distance function from a point to a line
noncomputable def distance_to_line (p : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  |a * p.1 + b * p.2 + c| / Real.sqrt (a ^ 2 + b ^ 2)

-- Define the actual proof problem
theorem distance_from_circle_center_to_line : 
  distance_to_line circle_center 2 1 (-5) = Real.sqrt 5 :=
by
  sorry

end distance_from_circle_center_to_line_l233_233390


namespace certain_number_l233_233903

theorem certain_number (a x : ℝ) (h1 : a / x * 2 = 12) (h2 : x = 0.1) : a = 0.6 := 
by
  sorry

end certain_number_l233_233903


namespace monikaTotalSpending_l233_233462

-- Define the conditions as constants
def mallSpent : ℕ := 250
def movieCost : ℕ := 24
def movieCount : ℕ := 3
def beanCost : ℚ := 1.25
def beanCount : ℕ := 20

-- Define the theorem to prove the total spending
theorem monikaTotalSpending : mallSpent + (movieCost * movieCount) + (beanCost * beanCount) = 347 :=
by
  sorry

end monikaTotalSpending_l233_233462


namespace player_match_count_l233_233254

open Real

theorem player_match_count (n : ℕ) : 
  (∃ T, T = 32 * n ∧ (T + 98) / (n + 1) = 38) → n = 10 :=
by
  sorry

end player_match_count_l233_233254


namespace chloe_boxes_of_clothing_l233_233860

theorem chloe_boxes_of_clothing (total_clothing pieces_per_box : ℕ) (h1 : total_clothing = 32) (h2 : pieces_per_box = 2 + 6) :
  ∃ B : ℕ, B = total_clothing / pieces_per_box ∧ B = 4 :=
by
  -- Proof can be filled in here
   sorry

end chloe_boxes_of_clothing_l233_233860


namespace fraction_product_l233_233734

theorem fraction_product : 
  (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) = 3 / 8 :=
by
  sorry

end fraction_product_l233_233734


namespace find_numbers_l233_233754

theorem find_numbers (S P : ℝ) (x₁ x₂ y₁ y₂ : ℝ) (h₁ : x₁ + y₁ = S) (h₂ : x₁ * y₁ = P) (h₃ : x₂ + y₂ = S) (h₄ : x₂ * y₂ = P) :
  x₁ = (S + Real.sqrt (S^2 - 4 * P)) / 2 ∧ y₁ = S - x₁ ∧
  x₂ = (S - Real.sqrt (S^2 - 4 * P)) / 2 ∧ y₂ = S - x₂ := 
by
  sorry

end find_numbers_l233_233754


namespace original_price_l233_233980

-- Definitions based on the problem conditions
variables (P : ℝ)

def john_payment (P : ℝ) : ℝ :=
  0.9 * P + 0.15 * P

def jane_payment (P : ℝ) : ℝ :=
  0.9 * P + 0.15 * (0.9 * P)

def price_difference (P : ℝ) : ℝ :=
  john_payment P - jane_payment P

theorem original_price (h : price_difference P = 0.51) : P = 34 := 
by
  sorry

end original_price_l233_233980


namespace hexagon_area_l233_233604

-- Definition of an equilateral triangle with a given perimeter.
def is_equilateral_triangle (P Q R : Type) [MetricSpace P] [MetricSpace Q] [MetricSpace R] :=
  ∀ (a b c : ℝ), a = b ∧ b = c ∧ a + b + c = 42 ∧ ∀ (angle : ℝ), angle = 60

-- Statement of the problem
theorem hexagon_area (P Q R P' Q' R' : Type) [MetricSpace P] [MetricSpace Q] [MetricSpace R]
  [MetricSpace P'] [MetricSpace Q'] [MetricSpace R']
  (h1 : is_equilateral_triangle P Q R) :
  ∃ (area : ℝ), area = 49 * Real.sqrt 3 := 
sorry

end hexagon_area_l233_233604


namespace slope_of_line_with_sine_of_angle_l233_233700

theorem slope_of_line_with_sine_of_angle (α : ℝ) 
  (hα₁ : 0 ≤ α) (hα₂ : α < Real.pi) 
  (h_sin : Real.sin α = Real.sqrt 3 / 2) : 
  ∃ k : ℝ, k = Real.tan α ∧ (k = Real.sqrt 3 ∨ k = -Real.sqrt 3) :=
by
  sorry

end slope_of_line_with_sine_of_angle_l233_233700


namespace percentage_of_bags_not_sold_l233_233914

theorem percentage_of_bags_not_sold
  (initial_stock : ℕ)
  (sold_monday : ℕ)
  (sold_tuesday : ℕ)
  (sold_wednesday : ℕ)
  (sold_thursday : ℕ)
  (sold_friday : ℕ)
  (h_initial : initial_stock = 600)
  (h_monday : sold_monday = 25)
  (h_tuesday : sold_tuesday = 70)
  (h_wednesday : sold_wednesday = 100)
  (h_thursday : sold_thursday = 110)
  (h_friday : sold_friday = 145) : 
  (initial_stock - (sold_monday + sold_tuesday + sold_wednesday + sold_thursday + sold_friday)) * 100 / initial_stock = 25 :=
by
  sorry

end percentage_of_bags_not_sold_l233_233914


namespace roman_coins_left_l233_233806

theorem roman_coins_left (X Y : ℕ) (h1 : X * Y = 50) (h2 : (X - 7) * Y = 28) : X - 7 = 8 :=
by
  sorry

end roman_coins_left_l233_233806


namespace inscribed_triangle_area_is_12_l233_233503

noncomputable def area_of_triangle_in_inscribed_circle 
  (a b c : ℝ) 
  (h_ratio : a = 2 * b ∧ c = 2 * a) 
  (h_radius : ∀ R, R = 4) 
  (h_inscribed : ∃ x, a = 2 * x ∧ b = 3 * x ∧ c = 4 * x ∧ c = 2 * 4) : 
  ℝ := 
1 / 2 * (2 * (4 / 2)) * (3 * (4 / 2))

theorem inscribed_triangle_area_is_12 
  (a b c : ℝ) 
  (h_ratio : a = 2 * b ∧ c = 2 * a) 
  (h_radius : ∀ R, R = 4) 
  (h_inscribed : ∃ x, a = 2 * x ∧ b = 3 * x ∧ c = 4 * x ∧ c = 2 * 4) :
  area_of_triangle_in_inscribed_circle a b c h_ratio h_radius h_inscribed = 12 :=
sorry

end inscribed_triangle_area_is_12_l233_233503


namespace expression_divisible_by_1897_l233_233680

theorem expression_divisible_by_1897 (n : ℕ) :
  1897 ∣ (2903^n - 803^n - 464^n + 261^n) :=
sorry

end expression_divisible_by_1897_l233_233680


namespace second_number_l233_233061

theorem second_number (A B : ℝ) (h1 : 0.50 * A = 0.40 * B + 180) (h2 : A = 456) : B = 120 := 
by
  sorry

end second_number_l233_233061


namespace minimum_value_of_f_range_of_a_l233_233419

noncomputable def f (x : ℝ) := x * Real.log x
noncomputable def g (x a : ℝ) := -x^2 + a * x - 3

theorem minimum_value_of_f :
  ∃ x_min : ℝ, ∀ x : ℝ, 0 < x → f x ≥ -1/Real.exp 1 := sorry -- This statement asserts that the minimum value of f(x) is -1/e.

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 < x → 2 * f x ≥ g x a) → a ≤ 4 := sorry -- This statement asserts that if 2f(x) ≥ g(x) for all x > 0, then a is at most 4.

end minimum_value_of_f_range_of_a_l233_233419


namespace regular_triangular_prism_properties_l233_233766

-- Regular triangular pyramid defined
structure RegularTriangularPyramid (height : ℝ) (base_side : ℝ)

-- Regular triangular prism defined
structure RegularTriangularPrism (height : ℝ) (base_side : ℝ) (lateral_area : ℝ)

-- Given data
def pyramid := RegularTriangularPyramid 15 12
def prism_lateral_area := 120

-- Statement of the problem
theorem regular_triangular_prism_properties (h_prism : ℝ) (ratio_lateral_area : ℚ) :
  (h_prism = 10 ∨ h_prism = 5) ∧ (ratio_lateral_area = 1/9 ∨ ratio_lateral_area = 4/9) :=
sorry

end regular_triangular_prism_properties_l233_233766


namespace no_integer_roots_of_quadratic_l233_233251

theorem no_integer_roots_of_quadratic (n : ℤ) : 
  ¬ ∃ (x : ℤ), x^2 - 16 * n * x + 7^5 = 0 := by
  sorry

end no_integer_roots_of_quadratic_l233_233251


namespace number_of_ping_pong_balls_l233_233619

def sales_tax_rate : ℝ := 0.16

def total_cost_with_tax (B x : ℝ) : ℝ := B * x * (1 + sales_tax_rate)

def total_cost_without_tax (B x : ℝ) : ℝ := (B + 3) * x

theorem number_of_ping_pong_balls
  (B x : ℝ) (h₁ : total_cost_with_tax B x = total_cost_without_tax B x) :
  B = 18.75 := 
sorry

end number_of_ping_pong_balls_l233_233619


namespace find_ax_plus_a_negx_l233_233909

theorem find_ax_plus_a_negx
  (a : ℝ) (x : ℝ)
  (h₁ : a > 0)
  (h₂ : a^(x/2) + a^(-x/2) = 5) :
  a^x + a^(-x) = 23 :=
by
  sorry

end find_ax_plus_a_negx_l233_233909


namespace factor_poly_PQ_sum_l233_233829

theorem factor_poly_PQ_sum (P Q : ℝ) (h : (∀ x : ℝ, (x^2 + 3 * x + 4) * (x^2 + -3 * x + 4) = x^4 + P * x^2 + Q)) : P + Q = 15 :=
by
  sorry

end factor_poly_PQ_sum_l233_233829


namespace trains_clear_time_l233_233186

theorem trains_clear_time
  (length_train1 : ℕ) (length_train2 : ℕ)
  (speed_train1_kmph : ℕ) (speed_train2_kmph : ℕ)
  (conversion_factor : ℕ) -- 5/18 as a rational number (for clarity)
  (approx_rel_speed : ℚ) -- Approximate relative speed 
  (total_distance : ℕ) 
  (total_time : ℚ) :
  length_train1 = 160 →
  length_train2 = 280 →
  speed_train1_kmph = 42 →
  speed_train2_kmph = 30 →
  conversion_factor = 5 / 18 →
  approx_rel_speed = (42 * (5 / 18) + 30 * (5 / 18)) →
  total_distance = length_train1 + length_train2 →
  total_time = total_distance / approx_rel_speed →
  total_time = 22 := 
by
  sorry

end trains_clear_time_l233_233186


namespace count_no_carry_pairs_l233_233177

theorem count_no_carry_pairs : 
  ∃ n, n = 1125 ∧ ∀ (a b : ℕ), (2000 ≤ a ∧ a < 2999 ∧ b = a + 1) → 
  (∀ i, (0 ≤ i ∧ i < 4) → ((a / (10 ^ i) % 10 + b / (10 ^ i) % 10) < 10)) := sorry

end count_no_carry_pairs_l233_233177


namespace shaded_areas_sum_l233_233485

theorem shaded_areas_sum (triangle_area : ℕ) (parts : ℕ)
  (h1 : triangle_area = 18)
  (h2 : parts = 9) :
  3 * (triangle_area / parts) = 6 :=
by
  sorry

end shaded_areas_sum_l233_233485


namespace one_fourth_of_8_point_8_is_fraction_l233_233528

theorem one_fourth_of_8_point_8_is_fraction:
  (1 / 4) * 8.8 = 11 / 5 :=
by sorry

end one_fourth_of_8_point_8_is_fraction_l233_233528


namespace platform_length_is_150_l233_233080

noncomputable def length_of_platform
  (train_length : ℝ)
  (time_to_cross_platform : ℝ)
  (time_to_cross_pole : ℝ)
  (L : ℝ) : Prop :=
  train_length + L = (train_length / time_to_cross_pole) * time_to_cross_platform

theorem platform_length_is_150 :
  length_of_platform 300 27 18 150 :=
by 
  -- Proof omitted, but the statement is ready for proving
  sorry

end platform_length_is_150_l233_233080


namespace equal_parts_count_l233_233812

def scale_length_in_inches : ℕ := (7 * 12) + 6
def part_length_in_inches : ℕ := 18
def number_of_parts (total_length part_length : ℕ) : ℕ := total_length / part_length

theorem equal_parts_count :
  number_of_parts scale_length_in_inches part_length_in_inches = 5 :=
by
  sorry

end equal_parts_count_l233_233812


namespace theo_cookies_per_sitting_l233_233122

-- Definitions from conditions
def sittings_per_day : ℕ := 3
def days_per_month : ℕ := 20
def cookies_in_3_months : ℕ := 2340

-- Calculation based on conditions
def sittings_per_month : ℕ := sittings_per_day * days_per_month
def sittings_in_3_months : ℕ := sittings_per_month * 3

-- Target statement
theorem theo_cookies_per_sitting :
  cookies_in_3_months / sittings_in_3_months = 13 :=
sorry

end theo_cookies_per_sitting_l233_233122


namespace fran_threw_away_80_pct_l233_233855

-- Definitions based on the conditions
def initial_votes_game_of_thrones := 10
def initial_votes_twilight := 12
def initial_votes_art_of_deal := 20
def altered_votes_twilight := initial_votes_twilight / 2
def new_total_votes := 2 * initial_votes_game_of_thrones

-- Theorem we are proving
theorem fran_threw_away_80_pct :
  ∃ x, x = 80 ∧
    new_total_votes = initial_votes_game_of_thrones + altered_votes_twilight + (initial_votes_art_of_deal * (1 - x / 100)) := by
  sorry

end fran_threw_away_80_pct_l233_233855


namespace translation_of_civilisation_l233_233543

def translation (word : String) (translation : String) : Prop :=
translation = "civilization"

theorem translation_of_civilisation (word : String) :
  word = "civilisation" → translation word "civilization" :=
by sorry

end translation_of_civilisation_l233_233543


namespace range_of_u_l233_233850

def satisfies_condition (x y : ℝ) : Prop := (x^2 / 3) + y^2 = 1

def u (x y : ℝ) : ℝ := |2 * x + y - 4| + |3 - x - 2 * y|

theorem range_of_u {x y : ℝ} (h : satisfies_condition x y) : ∀ u, 1 ≤ u ∧ u ≤ 13 :=
sorry

end range_of_u_l233_233850


namespace initial_short_bushes_l233_233939

theorem initial_short_bushes (B : ℕ) (H1 : B + 20 = 57) : B = 37 :=
by
  sorry

end initial_short_bushes_l233_233939


namespace value_of_f_m_plus_one_depends_on_m_l233_233720

def f (x a : ℝ) : ℝ := x^2 - x + a

theorem value_of_f_m_plus_one_depends_on_m (m a : ℝ) (h : f (-m) a < 0) :
  (∃ m, f (m + 1) a < 0) ∧ (∃ m, f (m + 1) a > 0) :=
by
  sorry

end value_of_f_m_plus_one_depends_on_m_l233_233720


namespace simpsons_paradox_example_l233_233499

theorem simpsons_paradox_example :
  ∃ n1 n2 a1 a2 b1 b2,
    n1 = 10 ∧ a1 = 3 ∧ b1 = 2 ∧
    n2 = 90 ∧ a2 = 45 ∧ b2 = 488 ∧
    ((a1 : ℝ) / n1 > (b1 : ℝ) / n1) ∧
    ((a2 : ℝ) / n2 > (b2 : ℝ) / n2) ∧
    ((a1 + a2 : ℝ) / (n1 + n2) < (b1 + b2 : ℝ) / (n1 + n2)) :=
by
  use 10, 90, 3, 45, 2, 488
  simp
  sorry

end simpsons_paradox_example_l233_233499


namespace find_a_for_perfect_square_trinomial_l233_233294

theorem find_a_for_perfect_square_trinomial (a : ℝ) :
  (∃ b : ℝ, x^2 - 8*x + a = (x - b)^2) ↔ a = 16 :=
by sorry

end find_a_for_perfect_square_trinomial_l233_233294


namespace focal_length_of_hyperbola_l233_233999

theorem focal_length_of_hyperbola (a b p: ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (p_pos : 0 < p) :
  (∃ (F V : ℝ × ℝ), 4 = dist F V ∧ F = (2, 0) ∧ V = (-2, 0)) ∧
  (∃ (P : ℝ × ℝ), P = (-2, -1) ∧ (∃ (d : ℝ), d = d / 2 ∧ P = (d, 0))) →
  2 * (Real.sqrt (a^2 + b^2)) = 2 * Real.sqrt 5 := 
sorry

end focal_length_of_hyperbola_l233_233999


namespace binom_identity1_binom_identity2_l233_233585

section Combinatorics

variable (n k m : ℕ)

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ :=
  if k > n then 0 else Nat.choose n k

-- Prove the identity: C(n, k) + C(n, k-1) = C(n+1, k)
theorem binom_identity1 : binomial n k + binomial n (k-1) = binomial (n+1) k :=
  sorry

-- Using the identity, prove: C(n, m) + C(n-1, m) + ... + C(n-10, m) = C(n+1, m+1) - C(n-10, m+1)
theorem binom_identity2 :
  (binomial n m + binomial (n-1) m + binomial (n-2) m + binomial (n-3) m
   + binomial (n-4) m + binomial (n-5) m + binomial (n-6) m + binomial (n-7) m
   + binomial (n-8) m + binomial (n-9) m + binomial (n-10) m)
   = binomial (n+1) (m+1) - binomial (n-10) (m+1) :=
  sorry

end Combinatorics

end binom_identity1_binom_identity2_l233_233585


namespace max_sum_of_four_integers_with_product_360_l233_233544

theorem max_sum_of_four_integers_with_product_360 :
  ∃ a b c d : ℕ, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d ∧ a * b * c * d = 360 ∧ a + b + c + d = 66 :=
sorry

end max_sum_of_four_integers_with_product_360_l233_233544


namespace toys_lost_l233_233475

theorem toys_lost (initial_toys found_in_closet total_after_finding : ℕ) 
  (h1 : initial_toys = 40) 
  (h2 : found_in_closet = 9) 
  (h3 : total_after_finding = 43) : 
  initial_toys - (total_after_finding - found_in_closet) = 9 :=
by 
  sorry

end toys_lost_l233_233475


namespace decreasing_interval_l233_233360

noncomputable def f (x : ℝ) := Real.exp (abs (x - 1))

theorem decreasing_interval : ∀ x y : ℝ, x ≤ y → y ≤ 1 → f y ≤ f x :=
by
  sorry

end decreasing_interval_l233_233360


namespace principal_amount_l233_233153

theorem principal_amount (r : ℝ) (n : ℕ) (t : ℕ) (A : ℝ) :
    r = 0.12 → n = 2 → t = 20 →
    ∃ P : ℝ, A = P * (1 + r / n)^(n * t) :=
by
  intros hr hn ht
  have P := A / (1 + r / n)^(n * t)
  use P
  sorry

end principal_amount_l233_233153


namespace smallest_number_divisible_by_6_in_permutations_list_l233_233347

def is_divisible_by_6 (n : ℕ) : Prop :=
  (∃ k : ℕ, n = 6 * k)

noncomputable def permutations_5_digits := 
  [1, 2, 3, 4, 5].permutations.map (λ l => l.foldl (λ acc x => 10 * acc + x) 0)

theorem smallest_number_divisible_by_6_in_permutations_list :
  ∃ n ∈ permutations_5_digits, is_divisible_by_6 n ∧ (∀ m ∈ permutations_5_digits, is_divisible_by_6 m → n ≤ m) :=
sorry

end smallest_number_divisible_by_6_in_permutations_list_l233_233347


namespace regular_vs_diet_sodas_l233_233495

theorem regular_vs_diet_sodas :
  let regular_cola := 67
  let regular_lemon := 45
  let regular_orange := 23
  let diet_cola := 9
  let diet_lemon := 32
  let diet_orange := 12
  let regular_sodas := regular_cola + regular_lemon + regular_orange
  let diet_sodas := diet_cola + diet_lemon + diet_orange
  regular_sodas - diet_sodas = 82 := sorry

end regular_vs_diet_sodas_l233_233495


namespace factorizations_of_2079_l233_233029

theorem factorizations_of_2079 :
  ∃ (a b : ℕ), 10 ≤ a ∧ a ≤ 99 ∧ 10 ≤ b ∧ b ≤ 99 ∧ a * b = 2079 ∧ (a, b) = (21, 99) ∨ (a, b) = (33, 63) :=
sorry

end factorizations_of_2079_l233_233029


namespace find_second_number_l233_233287

theorem find_second_number (a b c : ℕ) 
  (h1 : a + b + c = 550) 
  (h2 : a = 2 * b) 
  (h3 : c = a / 3) :
  b = 150 :=
by
  sorry

end find_second_number_l233_233287


namespace part1_part2_l233_233637

noncomputable def f (a x : ℝ) : ℝ := (Real.exp x) - a * x ^ 2 - x

theorem part1 {a : ℝ} : (∀ x y: ℝ, x < y → f a x ≤ f a y) ↔ (a = 1 / 2) :=
sorry

theorem part2 {a : ℝ} (h1 : a > 1 / 2):
  ∃ (x1 x2 : ℝ), (x1 < x2) ∧ (f a x2 < 1 + (Real.sin x2 - x2) / 2) :=
sorry

end part1_part2_l233_233637


namespace sequence_property_l233_233269

noncomputable def seq (n : ℕ) : ℕ := 
if n = 0 then 1 else 
if n = 1 then 3 else 
seq (n-2) + 3 * 2^(n-2)

theorem sequence_property {n : ℕ} (h_pos : n > 0) :
(∀ n : ℕ, n > 0 → seq (n + 2) ≤ seq n + 3 * 2^n) →
(∀ n : ℕ, n > 0 → seq (n + 1) ≥ 2 * seq n + 1) →
seq n = 2^n - 1 := 
sorry

end sequence_property_l233_233269


namespace triangle_inequality_l233_233617

variables {A B C P D E F : Type} -- Variables representing points in the plane.
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace P]
variables (PD PE PF PA PB PC : ℝ) -- Distances corresponding to the points.

-- Condition stating P lies inside or on the boundary of triangle ABC
axiom P_in_triangle_ABC : ∀ (A B C P : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace P], 
  (PD > 0 ∧ PE > 0 ∧ PF > 0 ∧ PA > 0 ∧ PB > 0 ∧ PC > 0)

-- Objective statement to prove
theorem triangle_inequality (PD PE PF PA PB PC : ℝ) 
  (h1 : PA ≥ 0) 
  (h2 : PB ≥ 0) 
  (h3 : PC ≥ 0) 
  (h4 : PD ≥ 0) 
  (h5 : PE ≥ 0) 
  (h6 : PF ≥ 0) :
  PA + PB + PC ≥ 2 * (PD + PE + PF) := 
sorry -- Proof to be provided later.

end triangle_inequality_l233_233617


namespace jonah_fish_count_l233_233569

theorem jonah_fish_count :
  let initial_fish := 14
  let added_fish := 2
  let eaten_fish := 6
  let removed_fish := 2
  let new_fish := 3
  initial_fish + added_fish - eaten_fish - removed_fish + new_fish = 11 := 
by
  sorry

end jonah_fish_count_l233_233569


namespace inequality_am_gm_l233_233430

theorem inequality_am_gm (a b : ℝ) (p q : ℝ) (h1: a > 0) (h2: b > 0) (h3: p > 1) (h4: q > 1) (h5 : 1/p + 1/q = 1) : 
  a^(1/p) * b^(1/q) ≤ a/p + b/q :=
by
  sorry

end inequality_am_gm_l233_233430


namespace solve_for_y_l233_233509

theorem solve_for_y (x y : ℝ) (h : (x + y)^5 - x^5 + y = 0) : y = 0 :=
sorry

end solve_for_y_l233_233509


namespace sub_fraction_l233_233936

theorem sub_fraction (a b c d : ℚ) (h1 : a = 5) (h2 : b = 9) (h3 : c = 1) (h4 : d = 6) : (a / b) - (c / d) = 7 / 18 := 
by
  sorry

end sub_fraction_l233_233936


namespace max_value_of_expression_l233_233851

theorem max_value_of_expression
  (x y z : ℝ)
  (h : 9 * x^2 + 4 * y^2 + 25 * z^2 = 1) :
  11 * x + 3 * y + 8 * z ≤ 3.1925 :=
sorry

end max_value_of_expression_l233_233851


namespace evaluate_x2_plus_y2_l233_233350

theorem evaluate_x2_plus_y2 (x y : ℝ) (h₁ : 3 * x + 2 * y = 20) (h₂ : 4 * x + 2 * y = 26) : x^2 + y^2 = 37 := by
  sorry

end evaluate_x2_plus_y2_l233_233350


namespace abigail_initial_money_l233_233424

variable (X : ℝ) -- Let X be the initial amount of money

def spent_on_food (X : ℝ) := 0.60 * X
def remaining_after_food (X : ℝ) := X - spent_on_food X
def spent_on_phone (X : ℝ) := 0.25 * remaining_after_food X
def remaining_after_phone (X : ℝ) := remaining_after_food X - spent_on_phone X
def final_amount (X : ℝ) := remaining_after_phone X - 20

theorem abigail_initial_money
    (food_spent : spent_on_food X = 0.60 * X)
    (phone_spent : spent_on_phone X = 0.10 * X)
    (remaining_after_entertainment : final_amount X = 40) :
    X = 200 :=
by
    sorry

end abigail_initial_money_l233_233424


namespace average_score_is_67_l233_233787

def scores : List ℕ := [55, 67, 76, 82, 55]
def num_of_subjects : ℕ := List.length scores
def total_score : ℕ := List.sum scores
def average_score : ℕ := total_score / num_of_subjects

theorem average_score_is_67 : average_score = 67 := by
  sorry

end average_score_is_67_l233_233787


namespace farmer_land_acres_l233_233184

theorem farmer_land_acres
  (initial_ratio_corn : Nat)
  (initial_ratio_sugar_cane : Nat)
  (initial_ratio_tobacco : Nat)
  (new_ratio_corn : Nat)
  (new_ratio_sugar_cane : Nat)
  (new_ratio_tobacco : Nat)
  (additional_tobacco_acres : Nat)
  (total_land_acres : Nat) :
  initial_ratio_corn = 5 →
  initial_ratio_sugar_cane = 2 →
  initial_ratio_tobacco = 2 →
  new_ratio_corn = 2 →
  new_ratio_sugar_cane = 2 →
  new_ratio_tobacco = 5 →
  additional_tobacco_acres = 450 →
  total_land_acres = 1350 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end farmer_land_acres_l233_233184


namespace find_ratio_l233_233227

-- Define the geometric sequence properties and conditions
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
 ∀ n : ℕ, a (n + 1) = a n * q

variables {a : ℕ → ℝ} {q : ℝ}

-- Conditions stated in the problem
axiom h₁ : a 5 * a 11 = 3
axiom h₂ : a 3 + a 13 = 4

-- The goal is to find the values of a_15 / a_5
theorem find_ratio (h₁ : a 5 * a 11 = 3) (h₂ : a 3 + a 13 = 4) :
  ∃ r : ℝ, r = a 15 / a 5 ∧ (r = 3 ∨ r = 1 / 3) :=
sorry

end find_ratio_l233_233227


namespace charles_average_speed_l233_233756

theorem charles_average_speed
  (total_distance : ℕ)
  (half_distance : ℕ)
  (second_half_speed : ℕ)
  (total_time : ℕ)
  (first_half_distance second_half_distance : ℕ)
  (time_for_second_half : ℕ)
  (time_for_first_half : ℕ)
  (first_half_speed : ℕ)
  (h1 : total_distance = 3600)
  (h2 : half_distance = total_distance / 2)
  (h3 : first_half_distance = half_distance)
  (h4 : second_half_distance = half_distance)
  (h5 : second_half_speed = 180)
  (h6 : total_time = 30)
  (h7 : time_for_second_half = second_half_distance / second_half_speed)
  (h8 : time_for_first_half = total_time - time_for_second_half)
  (h9 : first_half_speed = first_half_distance / time_for_first_half) :
  first_half_speed = 90 := by
  sorry

end charles_average_speed_l233_233756


namespace find_constant_a_l233_233677

theorem find_constant_a (S : ℕ → ℝ) (a : ℝ) :
  (∀ n, S n = (1/2) * 3^(n+1) - a) →
  a = 3/2 :=
sorry

end find_constant_a_l233_233677


namespace total_cards_in_box_l233_233062

-- Definitions based on conditions
def xiaoMingCountsFaster (m h : ℕ) := 6 * h = 4 * m
def xiaoHuaForgets (h1 h2 : ℕ) := h1 + h2 = 112
def finalCardLeft (t : ℕ) := t - 1 = 112

-- Main theorem stating that the total number of cards is 353
theorem total_cards_in_box : ∃ N : ℕ, 
    (∃ m h1 h2 : ℕ,
        xiaoMingCountsFaster m h1 ∧
        xiaoHuaForgets h1 h2 ∧
        finalCardLeft N) ∧
    N = 353 :=
sorry

end total_cards_in_box_l233_233062


namespace red_car_speed_l233_233923

noncomputable def speed_blue : ℕ := 80
noncomputable def speed_green : ℕ := 8 * speed_blue
noncomputable def speed_red : ℕ := 2 * speed_green

theorem red_car_speed : speed_red = 1280 := by
  unfold speed_red
  unfold speed_green
  unfold speed_blue
  sorry

end red_car_speed_l233_233923


namespace positional_relationship_l233_233322

variables {Point Line Plane : Type}
variables (a b : Line) (α : Plane)

-- Condition 1: Line a is parallel to Plane α
def line_parallel_to_plane (a : Line) (α : Plane) : Prop := sorry

-- Condition 2: Line b is contained within Plane α
def line_contained_within_plane (b : Line) (α : Plane) : Prop := sorry

-- The positional relationship between line a and line b is either parallel or skew
def lines_parallel_or_skew (a b : Line) : Prop := sorry

theorem positional_relationship (ha : line_parallel_to_plane a α) (hb : line_contained_within_plane b α) :
  lines_parallel_or_skew a b :=
sorry

end positional_relationship_l233_233322


namespace ratio_of_professionals_l233_233670

-- Define the variables and conditions as stated in the problem.
variables (e d l : ℕ)

-- The condition about the average ages leading to the given equation.
def avg_age_condition : Prop := (40 * e + 50 * d + 60 * l) / (e + d + l) = 45

-- The statement to prove that given the average age condition, the ratio is 1:1:3.
theorem ratio_of_professionals (h : avg_age_condition e d l) : e = d + 3 * l :=
sorry

end ratio_of_professionals_l233_233670


namespace original_stations_l233_233769

theorem original_stations (m n : ℕ) (h : n > 1) (h_equation : n * (2 * m + n - 1) = 58) : m = 14 :=
by
  -- proof omitted
  sorry

end original_stations_l233_233769


namespace final_amount_after_two_years_l233_233588

open BigOperators

/-- Given an initial amount A0 and a percentage increase p, calculate the amount after n years -/
def compound_increase (A0 : ℝ) (p : ℝ) (n : ℕ) : ℝ :=
  (A0 * (1 + p)^n)

theorem final_amount_after_two_years (A0 : ℝ) (p : ℝ) (A2 : ℝ) :
  A0 = 1600 ∧ p = 1 / 8 ∧ compound_increase 1600 (1 / 8) 2 = 2025 :=
  sorry

end final_amount_after_two_years_l233_233588


namespace number_of_routes_from_A_to_B_l233_233005

-- Define the grid dimensions
def grid_rows : ℕ := 3
def grid_columns : ℕ := 2

-- Define the total number of steps needed to travel from A to B
def total_steps : ℕ := grid_rows + grid_columns

-- Define the number of right moves (R) and down moves (D)
def right_moves : ℕ := grid_rows
def down_moves : ℕ := grid_columns

-- Calculate the number of different routes using combination formula
def number_of_routes : ℕ := Nat.choose total_steps right_moves

-- The main statement to be proven
theorem number_of_routes_from_A_to_B : number_of_routes = 10 :=
by sorry

end number_of_routes_from_A_to_B_l233_233005


namespace perfect_square_trinomial_l233_233884

theorem perfect_square_trinomial (m : ℝ) : (∃ b : ℝ, (x^2 - 6 * x + m) = (x + b) ^ 2) → m = 9 :=
by
  sorry

end perfect_square_trinomial_l233_233884


namespace range_of_m_l233_233514

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, (m / (2*x - 1) + 3 = 0) ∧ (x > 0)) ↔ (m < 3 ∧ m ≠ 0) :=
by
  sorry

end range_of_m_l233_233514


namespace chemical_reaction_proof_l233_233196

-- Define the given number of moles for each reactant
def moles_NaOH : ℕ := 4
def moles_NH4Cl : ℕ := 3

-- Define the balanced chemical equation stoichiometry
def stoichiometry_ratio_NaOH_NH4Cl : ℕ := 1

-- Define the product formation based on the limiting reactant
theorem chemical_reaction_proof
  (moles_NaOH : ℕ)
  (moles_NH4Cl : ℕ)
  (stoichiometry_ratio_NaOH_NH4Cl : ℕ)
  (h1 : moles_NaOH = 4)
  (h2 : moles_NH4Cl = 3)
  (h3 : stoichiometry_ratio_NaOH_NH4Cl = 1):
  (3 = 3 * 1) ∧
  (3 = 3 * 1) ∧
  (3 = 3 * 1) ∧
  (3 = moles_NH4Cl) ∧
  (1 = moles_NaOH - moles_NH4Cl) :=
by {
  -- Provide assumptions based on the problem
  sorry
}

end chemical_reaction_proof_l233_233196


namespace range_of_f_l233_233211

noncomputable def f (x : ℝ) : ℝ :=
  (Real.exp (3 * x) - 2) / (Real.exp (3 * x) + 2)

theorem range_of_f (x : ℝ) : -1 < f x ∧ f x < 1 :=
by
  sorry

end range_of_f_l233_233211


namespace centroid_value_l233_233682

-- Define the points P, Q, R
def P : ℝ × ℝ := (4, 3)
def Q : ℝ × ℝ := (-1, 6)
def R : ℝ × ℝ := (7, -2)

-- Define the coordinates of the centroid S
noncomputable def S : ℝ × ℝ := 
  ( (4 + (-1) + 7) / 3, (3 + 6 + (-2)) / 3 )

-- Statement to prove
theorem centroid_value : 
  let x := (4 + (-1) + 7) / 3
  let y := (3 + 6 + (-2)) / 3
  8 * x + 3 * y = 101 / 3 :=
by
  let x := (4 + (-1) + 7) / 3
  let y := (3 + 6 + (-2)) / 3
  have h: 8 * x + 3 * y = 101 / 3 := sorry
  exact h

end centroid_value_l233_233682


namespace equation_of_line_l233_233566

theorem equation_of_line (P : ℝ × ℝ) (A : ℝ) (m : ℝ) (hP : P = (-3, 4)) (hA : A = 3) (hm : m = 1) :
  ((2 * P.1 + 3 * P.2 - 6 = 0) ∨ (8 * P.1 + 3 * P.2 + 12 = 0)) :=
by 
  sorry

end equation_of_line_l233_233566


namespace time_taken_l233_233016

-- Define the function T which takes the number of cats, the number of rats, and returns the time in minutes
def T (n m : ℕ) : ℕ := if n = m then 4 else sorry

-- The theorem states that, given n cats and n rats, the time taken is 4 minutes
theorem time_taken (n : ℕ) : T n n = 4 :=
by simp [T]

end time_taken_l233_233016


namespace smallest_number_of_pencils_l233_233494

theorem smallest_number_of_pencils
  (P : ℕ)
  (h5 : P % 5 = 2)
  (h9 : P % 9 = 2)
  (h11 : P % 11 = 2)
  (hP_gt2 : P > 2) :
  P = 497 :=
by
  sorry

end smallest_number_of_pencils_l233_233494


namespace find_v5_l233_233937

noncomputable def sequence (v : ℕ → ℝ) : Prop :=
  ∀ n, v (n + 2) = 3 * v (n + 1) + v n + 1

theorem find_v5 :
  ∃ (v : ℕ → ℝ), sequence v ∧ v 3 = 11 ∧ v 6 = 242 ∧ v 5 = 73.5 :=
by
  sorry

end find_v5_l233_233937


namespace greening_investment_growth_l233_233984

-- Define initial investment in 2020 and investment in 2022.
def investment_2020 : ℝ := 20000
def investment_2022 : ℝ := 25000

-- Define the average growth rate x
variable (x : ℝ)

-- The mathematically equivalent proof problem:
theorem greening_investment_growth : 
  20 * (1 + x) ^ 2 = 25 :=
sorry

end greening_investment_growth_l233_233984


namespace system_of_equations_solution_l233_233731

theorem system_of_equations_solution :
  ∀ (a b : ℝ),
  (-2 * a + b^2 = Real.cos (π * a + b^2) - 1 ∧ b^2 = Real.cos (2 * π * a + b^2) - 1) ↔ (a = 0 ∧ b = 0) ∨ (a = 1 ∧ b = 0) :=
by
  intro a b
  sorry

end system_of_equations_solution_l233_233731


namespace alice_bob_meet_after_six_turns_l233_233279

/-
Alice and Bob play a game involving a circle whose circumference
is divided by 12 equally-spaced points. The points are numbered
clockwise, from 1 to 12. Both start on point 12. Alice moves clockwise
and Bob, counterclockwise. In a turn of the game, Alice moves 5 points 
clockwise and Bob moves 9 points counterclockwise. The game ends when they stop on
the same point. 
-/
theorem alice_bob_meet_after_six_turns (k : ℕ) :
  (5 * k) % 12 = (12 - (9 * k) % 12) % 12 -> k = 6 :=
by
  sorry

end alice_bob_meet_after_six_turns_l233_233279


namespace total_trees_now_l233_233286

-- Definitions from conditions
def initial_mahogany_trees : ℕ := 50
def initial_narra_trees : ℕ := 30
def total_fallen_trees : ℕ := 5

-- Additional definitions capturing relations
def fell_narra_trees (x : ℕ) : Prop := x + (x + 1) = total_fallen_trees
def new_narra_trees_planted (x : ℕ) : ℕ := 2 * x
def new_mahogany_trees_planted (x : ℕ) : ℕ := 3 * (x + 1)

-- Final goal
theorem total_trees_now (x : ℕ) (h : fell_narra_trees x) :
  initial_mahogany_trees + initial_narra_trees
  - total_fallen_trees
  + new_narra_trees_planted x
  + new_mahogany_trees_planted x = 88 := by
  sorry

end total_trees_now_l233_233286


namespace martin_ring_fraction_l233_233552

theorem martin_ring_fraction (f : ℚ) :
  (36 + (36 * f + 4) = 52) → (f = 1 / 3) :=
by
  intro h
  -- Solution steps would go here
  sorry

end martin_ring_fraction_l233_233552


namespace marbles_problem_l233_233708

theorem marbles_problem
  (cindy_original : ℕ)
  (lisa_original : ℕ)
  (h1 : cindy_original = 20)
  (h2 : cindy_original = lisa_original + 5)
  (marbles_given : ℕ)
  (h3 : marbles_given = 12) :
  (lisa_original + marbles_given) - (cindy_original - marbles_given) = 19 :=
by
  sorry

end marbles_problem_l233_233708


namespace initial_birds_l233_233432

theorem initial_birds (B : ℕ) (h : B + 13 = 42) : B = 29 :=
sorry

end initial_birds_l233_233432


namespace find_m_of_parallel_vectors_l233_233134

theorem find_m_of_parallel_vectors (m : ℝ) 
  (a : ℝ × ℝ := (1, 2)) 
  (b : ℝ × ℝ := (m, m + 1))
  (parallel : a.1 * b.2 = a.2 * b.1) :
  m = 1 :=
by
  -- We assume a parallel condition and need to prove m = 1
  sorry

end find_m_of_parallel_vectors_l233_233134


namespace f_at_7_l233_233348

-- Define the function f and its properties
axiom f : ℝ → ℝ
axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom periodic_f : ∀ x : ℝ, f (x + 4) = f x
axiom values_f : ∀ x : ℝ, 0 < x ∧ x < 2 → f x = 2 * x^2

-- Prove that f(7) = -2
theorem f_at_7 : f 7 = -2 :=
by
  sorry

end f_at_7_l233_233348


namespace smallest_solution_x4_minus_50x2_plus_625_eq_0_l233_233644

theorem smallest_solution_x4_minus_50x2_plus_625_eq_0 :
  ∃ (x : ℝ), (x * x * x * x - 50 * x * x + 625 = 0) ∧ (∀ y, (y * y * y * y - 50 * y * y + 625 = 0) → x ≤ y) :=
sorry

end smallest_solution_x4_minus_50x2_plus_625_eq_0_l233_233644


namespace points_same_color_separed_by_two_l233_233212

theorem points_same_color_separed_by_two (circle : Fin 239 → Bool) : 
  ∃ i j : Fin 239, i ≠ j ∧ (i + 2) % 239 = j ∧ circle i = circle j :=
by
  sorry

end points_same_color_separed_by_two_l233_233212


namespace betty_age_l233_233857

theorem betty_age (A M B : ℕ) (h1 : A = 2 * M) (h2 : A = 4 * B) (h3 : M = A - 22) : B = 11 :=
by
  sorry

end betty_age_l233_233857


namespace equations_of_line_l233_233885

variables (x y : ℝ)

-- Given conditions
def passes_through_point (P : ℝ × ℝ) (x y : ℝ) := (x, y) = P

def has_equal_intercepts_on_axes (f : ℝ → ℝ) :=
  ∃ z : ℝ, z ≠ 0 ∧ f z = 0 ∧ f 0 = z

-- The proof problem statement
theorem equations_of_line (P : ℝ × ℝ) (hP : passes_through_point P 2 (-3)) (h : has_equal_intercepts_on_axes (λ x => -x / (x / 2))) :
  (x + y + 1 = 0) ∨ (3 * x + 2 * y = 0) := 
sorry

end equations_of_line_l233_233885


namespace pasta_needed_for_family_reunion_l233_233429

-- Conditions definition
def original_pasta : ℝ := 2
def original_servings : ℕ := 7
def family_reunion_people : ℕ := 35

-- Proof statement
theorem pasta_needed_for_family_reunion : 
  (family_reunion_people / original_servings) * original_pasta = 10 := 
by 
  sorry

end pasta_needed_for_family_reunion_l233_233429


namespace angle_ratio_l233_233389

theorem angle_ratio (BP BQ BM: ℝ) (ABC: ℝ) (quadrisect : BP = ABC/4 ∧ BQ = ABC)
  (bisect : BM = (3/4) * ABC / 2):
  (BM / (ABC / 4 + ABC / 4)) = 1 / 6 := by
    sorry

end angle_ratio_l233_233389


namespace robert_ate_more_chocolates_l233_233997

-- Define the number of chocolates eaten by Robert and Nickel
def robert_chocolates : ℕ := 12
def nickel_chocolates : ℕ := 3

-- State the problem as a theorem to prove
theorem robert_ate_more_chocolates :
  robert_chocolates - nickel_chocolates = 9 :=
by
  sorry

end robert_ate_more_chocolates_l233_233997


namespace total_appetizers_l233_233561

theorem total_appetizers (hotdogs cheese_pops chicken_nuggets mini_quiches stuffed_mushrooms total_portions : Nat)
  (h1 : hotdogs = 60)
  (h2 : cheese_pops = 40)
  (h3 : chicken_nuggets = 80)
  (h4 : mini_quiches = 100)
  (h5 : stuffed_mushrooms = 50)
  (h6 : total_portions = hotdogs + cheese_pops + chicken_nuggets + mini_quiches + stuffed_mushrooms) :
  total_portions = 330 :=
by sorry

end total_appetizers_l233_233561


namespace proportion_of_solution_x_in_mixture_l233_233031

-- Definitions for the conditions in given problem
def solution_x_contains_perc_a : ℚ := 0.20
def solution_y_contains_perc_a : ℚ := 0.30
def solution_z_contains_perc_a : ℚ := 0.40

def solution_y_to_z_ratio : ℚ := 3 / 2
def final_mixture_perc_a : ℚ := 0.25

-- Proving the proportion of solution x in the mixture equals 9/14
theorem proportion_of_solution_x_in_mixture
  (x y z : ℚ) (k : ℚ) (hx : x = 9 * k) (hy : y = 3 * k) (hz : z = 2 * k) :
  solution_x_contains_perc_a * x + solution_y_contains_perc_a * y + solution_z_contains_perc_a * z
  = final_mixture_perc_a * (x + y + z) →
  x / (x + y + z) = 9 / 14 :=
by
  intros h
  -- leaving the proof as a placeholder
  sorry

end proportion_of_solution_x_in_mixture_l233_233031


namespace sin_double_angle_l233_233530

theorem sin_double_angle (α : ℝ) (h : Real.tan (Real.pi + α) = 2) : Real.sin (2 * α) = 4 / 5 := 
by 
  sorry

end sin_double_angle_l233_233530


namespace green_peaches_are_six_l233_233047

/-- There are 5 red peaches in the basket. -/
def red_peaches : ℕ := 5

/-- There are 14 yellow peaches in the basket. -/
def yellow_peaches : ℕ := 14

/-- There are total of 20 green and yellow peaches in the basket. -/
def green_and_yellow_peaches : ℕ := 20

/-- The number of green peaches is calculated as the difference between the total number of green and yellow peaches and the number of yellow peaches. -/
theorem green_peaches_are_six :
  (green_and_yellow_peaches - yellow_peaches) = 6 :=
by
  sorry

end green_peaches_are_six_l233_233047


namespace lines_coplanar_l233_233273

/-
Given:
- Line 1 parameterized as (2 + s, 4 - k * s, -1 + k * s)
- Line 2 parameterized as (2 * t, 2 + t, 3 - t)
Prove: If these lines are coplanar, then k = -1/2
-/
theorem lines_coplanar (k : ℚ) (s t : ℚ)
  (line1 : ℚ × ℚ × ℚ := (2 + s, 4 - k * s, -1 + k * s))
  (line2 : ℚ × ℚ × ℚ := (2 * t, 2 + t, 3 - t))
  (coplanar : ∃ (s t : ℚ), line1 = line2) :
  k = -1 / 2 := 
sorry

end lines_coplanar_l233_233273


namespace fg_of_2_l233_233876

def f (x : ℤ) : ℤ := 4 * x + 3
def g (x : ℤ) : ℤ := x ^ 3 + 1

theorem fg_of_2 : f (g 2) = 39 := by
  sorry

end fg_of_2_l233_233876


namespace percentage_decrease_l233_233479

theorem percentage_decrease (original_price new_price decrease: ℝ) (h₁: original_price = 2400) (h₂: new_price = 1200) (h₃: decrease = original_price - new_price): 
  decrease / original_price * 100 = 50 :=
by
  rw [h₁, h₂] at h₃ -- Update the decrease according to given prices
  sorry -- Left as a placeholder for the actual proof

end percentage_decrease_l233_233479


namespace find_x_l233_233634

theorem find_x (x : ℝ) (h : x + 2.75 + 0.158 = 2.911) : x = 0.003 :=
sorry

end find_x_l233_233634


namespace fraction_is_one_fifth_l233_233661

theorem fraction_is_one_fifth (f : ℚ) (h1 : f * 50 - 4 = 6) : f = 1 / 5 :=
by
  sorry

end fraction_is_one_fifth_l233_233661


namespace intersection_of_curves_l233_233639

theorem intersection_of_curves (x : ℝ) (y : ℝ) (h₁ : y = 9 / (x^2 + 3)) (h₂ : x + y = 3) : x = 0 :=
sorry

end intersection_of_curves_l233_233639


namespace hyperbola_problem_l233_233925

-- Given the conditions of the hyperbola
def hyperbola (x y: ℝ) (b: ℝ) : Prop := (x^2) / 4 - (y^2) / (b^2) = 1 ∧ b > 0

-- Asymptote condition
def asymptote (b: ℝ) : Prop := (b / 2) = (Real.sqrt 6 / 2)

-- Foci, point P condition
def foci_and_point (PF1 PF2: ℝ) : Prop := PF1 / PF2 = 3 / 1 ∧ PF1 - PF2 = 4

-- Math proof problem
theorem hyperbola_problem (b PF1 PF2: ℝ) (P: ℝ × ℝ) :
  hyperbola P.1 P.2 b ∧ asymptote b ∧ foci_and_point PF1 PF2 →
  |PF1 + PF2| = 2 * Real.sqrt 10 :=
by
  sorry

end hyperbola_problem_l233_233925


namespace find_constant_l233_233804

-- Define the variables: t, x, y, and the constant
variable (t x y constant : ℝ)

-- Conditions
def x_def : x = constant - 2 * t :=
  by sorry

def y_def : y = 2 * t - 2 :=
  by sorry

def x_eq_y_at_t : t = 0.75 → x = y :=
  by sorry

-- Proposition: Prove that the constant in the equation for x is 1
theorem find_constant (ht : t = 0.75) (hx : x = constant - 2 * t) (hy : y = 2 * t - 2) (he : x = y) :
  constant = 1 :=
  by sorry

end find_constant_l233_233804


namespace total_dress_designs_l233_233722

def num_colors := 5
def num_patterns := 6
def num_sizes := 3

theorem total_dress_designs : num_colors * num_patterns * num_sizes = 90 :=
by
  sorry

end total_dress_designs_l233_233722


namespace geometric_then_sum_geometric_l233_233223

variable {a b c d : ℝ}

def geometric_sequence (a b c d : ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ b = a * r ∧ c = b * r ∧ d = c * r

def forms_geometric_sequence (x y z : ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ y = x * r ∧ z = y * r

theorem geometric_then_sum_geometric (h : geometric_sequence a b c d) :
  forms_geometric_sequence (a + b) (b + c) (c + d) :=
sorry

end geometric_then_sum_geometric_l233_233223


namespace part1_k_real_part2_find_k_l233_233590

-- Part 1: Discriminant condition
theorem part1_k_real (k : ℝ) (h : x^2 + (2*k - 1)*x + k^2 - 1 = 0) : k ≤ 5 / 4 :=
by
  sorry

-- Part 2: Given additional conditions, find k
theorem part2_find_k (x1 x2 k : ℝ) (h_eq : x^2 + (2 * k - 1) * x + k^2 - 1 = 0)
  (h1 : x1 + x2 = 1 - 2 * k) (h2 : x1 * x2 = k^2 - 1) (h3 : x1^2 + x2^2 = 16 + x1 * x2) : k = -2 :=
by
  sorry

end part1_k_real_part2_find_k_l233_233590


namespace hawks_total_points_l233_233953

/-- 
  Define the number of points per touchdown 
  and the number of touchdowns scored by the Hawks. 
-/
def points_per_touchdown : ℕ := 7
def touchdowns : ℕ := 3

/-- 
  Prove that the total number of points the Hawks have is 21. 
-/
theorem hawks_total_points : touchdowns * points_per_touchdown = 21 :=
by
  sorry

end hawks_total_points_l233_233953


namespace decimal_arithmetic_l233_233070

theorem decimal_arithmetic : 0.45 - 0.03 + 0.008 = 0.428 := by
  sorry

end decimal_arithmetic_l233_233070


namespace valerie_light_bulbs_deficit_l233_233434

theorem valerie_light_bulbs_deficit :
  let small_price := 8.75
  let medium_price := 11.25
  let large_price := 15.50
  let xsmall_price := 6.10
  let budget := 120
  
  let lamp_A_cost := 2 * small_price
  let lamp_B_cost := 3 * medium_price
  let lamp_C_cost := large_price
  let lamp_D_cost := 4 * xsmall_price
  let lamp_E_cost := 2 * large_price
  let lamp_F_cost := small_price + medium_price

  let total_cost := lamp_A_cost + lamp_B_cost + lamp_C_cost + lamp_D_cost + lamp_E_cost + lamp_F_cost

  total_cost - budget = 22.15 :=
by
  sorry

end valerie_light_bulbs_deficit_l233_233434


namespace solve_equation_l233_233091

theorem solve_equation :
  {x : ℝ | (x + 1) * (x + 3) = x + 1} = {-1, -2} :=
sorry

end solve_equation_l233_233091


namespace minimum_choir_members_l233_233749

def choir_members_min (n : ℕ) : Prop :=
  (n % 8 = 0) ∧ 
  (n % 9 = 0) ∧ 
  (n % 10 = 0) ∧ 
  (n % 11 = 0)

theorem minimum_choir_members : ∃ n, choir_members_min n ∧ (∀ m, choir_members_min m → n ≤ m) :=
sorry

end minimum_choir_members_l233_233749


namespace mrs_sheridan_final_cats_l233_233570

def initial_cats : ℝ := 17.5
def given_away_cats : ℝ := 6.2
def returned_cats : ℝ := 2.8
def additional_given_away_cats : ℝ := 1.3

theorem mrs_sheridan_final_cats : 
  initial_cats - given_away_cats + returned_cats - additional_given_away_cats = 12.8 :=
by
  sorry

end mrs_sheridan_final_cats_l233_233570


namespace smallest_gcd_l233_233191

theorem smallest_gcd (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (H1 : Nat.gcd x y = 270) (H2 : Nat.gcd x z = 105) : Nat.gcd y z = 15 :=
sorry

end smallest_gcd_l233_233191


namespace triangle_sequence_relation_l233_233770

theorem triangle_sequence_relation (b d c k : ℤ) (h₁ : b % d = 0) (h₂ : c % k = 0) (h₃ : b^2 + (b + 2*d)^2 = (c + 6*k)^2) :
  c = 0 :=
sorry

end triangle_sequence_relation_l233_233770


namespace soccer_points_l233_233425

def total_points (wins draws losses : ℕ) (points_per_win points_per_draw points_per_loss : ℕ) : ℕ :=
  wins * points_per_win + draws * points_per_draw + losses * points_per_loss

theorem soccer_points : total_points 14 4 2 3 1 0 = 46 :=
by
  sorry

end soccer_points_l233_233425


namespace factorize_difference_of_squares_l233_233777

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 36 = (x + 6) * (x - 6) :=
by 
  sorry

end factorize_difference_of_squares_l233_233777


namespace molecular_weight_of_one_mole_l233_233657

-- Definitions as Conditions
def total_molecular_weight := 960
def number_of_moles := 5

-- The theorem statement
theorem molecular_weight_of_one_mole :
  total_molecular_weight / number_of_moles = 192 :=
by
  sorry

end molecular_weight_of_one_mole_l233_233657


namespace celsius_equals_fahrenheit_l233_233631

-- Define the temperature scales.
def celsius_to_fahrenheit (T_C : ℝ) : ℝ := 1.8 * T_C + 32

-- The Lean statement for the problem.
theorem celsius_equals_fahrenheit : ∃ (T : ℝ), T = celsius_to_fahrenheit T ↔ T = -40 :=
by
  sorry -- Proof is not required, just the statement.

end celsius_equals_fahrenheit_l233_233631


namespace exists_equilateral_triangle_l233_233179

variables {d1 d2 d3 : AffineSubspace ℝ (EuclideanSpace ℝ (Fin 2))}

theorem exists_equilateral_triangle (hne1 : d1 ≠ d2) (hne2 : d2 ≠ d3) (hne3 : d1 ≠ d3) : 
  ∃ (A1 A2 A3 : EuclideanSpace ℝ (Fin 2)), 
  (A1 ∈ d1 ∧ A2 ∈ d2 ∧ A3 ∈ d3) ∧ 
  dist A1 A2 = dist A2 A3 ∧ dist A2 A3 = dist A3 A1 := 
sorry

end exists_equilateral_triangle_l233_233179


namespace find_multiplier_l233_233972

theorem find_multiplier (N x : ℕ) (h₁ : N = 12) (h₂ : N * x - 3 = (N - 7) * 9) : x = 4 :=
by
  sorry

end find_multiplier_l233_233972


namespace arithmetic_sequence_nth_term_l233_233258

theorem arithmetic_sequence_nth_term (x n : ℝ) 
  (h1 : 3*x - 4 = a1)
  (h2 : 7*x - 14 = a2)
  (h3 : 4*x + 6 = a3)
  (h4 : a_n = 3012) :
n = 392 :=
  sorry

end arithmetic_sequence_nth_term_l233_233258


namespace third_divisor_is_11_l233_233375

theorem third_divisor_is_11 (n : ℕ) (x : ℕ) : 
  n = 200 ∧ (n - 20) % 15 = 0 ∧ (n - 20) % 30 = 0 ∧ (n - 20) % x = 0 ∧ (n - 20) % 60 = 0 → 
  x = 11 :=
by
  sorry

end third_divisor_is_11_l233_233375


namespace problem1_problem2_l233_233051

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.sin x * (Real.sqrt 3 * Real.cos x + Real.sin x) - 2

-- Problem 1
theorem problem1 (α : ℝ) (hα1 : Real.sin α = -1 / 2) (hα2 : Real.cos α = Real.sqrt 3 / 2) :
  f α = -3 := sorry

-- Problem 2
theorem problem2 (h0 : ∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≥ -2) :
  ∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = -2 := sorry

end problem1_problem2_l233_233051


namespace symmetric_point_coordinates_l233_233021

theorem symmetric_point_coordinates (M : ℝ × ℝ) (N : ℝ × ℝ) (hM : M = (1, -2)) (h_sym : N = (-M.1, -M.2)) :
  N = (-1, 2) :=
by sorry

end symmetric_point_coordinates_l233_233021


namespace parts_repetition_cycle_l233_233225

noncomputable def parts_repetition_condition (t : ℕ) : Prop := sorry
def parts_initial_condition : Prop := sorry

theorem parts_repetition_cycle :
  parts_initial_condition →
  parts_repetition_condition 2 ∧
  parts_repetition_condition 4 ∧
  parts_repetition_condition 38 ∧
  parts_repetition_condition 76 :=
sorry


end parts_repetition_cycle_l233_233225


namespace largest_n_exists_l233_233784

theorem largest_n_exists :
  ∃ (n : ℕ), (∀ (x : ℕ → ℝ), (∀ i j : ℕ, 1 ≤ i → i < j → j ≤ n → (1 + x i * x j)^2 ≤ 0.99 * (1 + x i^2) * (1 + x j^2))) ∧ n = 31 :=
sorry

end largest_n_exists_l233_233784


namespace find_number_of_Persians_l233_233765

variable (P : ℕ)  -- Number of Persian cats Jamie owns
variable (M : ℕ := 2)  -- Number of Maine Coons Jamie owns (given by conditions)
variable (G_P : ℕ := P / 2)  -- Number of Persian cats Gordon owns, which is half of Jamie's
variable (G_M : ℕ := M + 1)  -- Number of Maine Coons Gordon owns, one more than Jamie's
variable (H_P : ℕ := 0)  -- Number of Persian cats Hawkeye owns, which is 0
variable (H_M : ℕ := G_M - 1)  -- Number of Maine Coons Hawkeye owns, one less than Gordon's

theorem find_number_of_Persians (sum_cats : P + M + G_P + G_M + H_P + H_M = 13) : 
  P = 4 :=
by
  -- Proof can be filled in here
  sorry

end find_number_of_Persians_l233_233765


namespace remainder_1234567_div_256_l233_233977

/--
  Given the numbers 1234567 and 256, prove that the remainder when
  1234567 is divided by 256 is 57.
-/
theorem remainder_1234567_div_256 :
  1234567 % 256 = 57 :=
by
  sorry

end remainder_1234567_div_256_l233_233977


namespace sum_of_center_coordinates_l233_233545

theorem sum_of_center_coordinates (x y : ℝ) :
    (x^2 + y^2 - 6*x + 8*y = 18) → (x = 3) → (y = -4) → x + y = -1 := 
by
    intro h1 hx hy
    rw [hx, hy]
    norm_num

end sum_of_center_coordinates_l233_233545


namespace sector_angle_sector_max_area_l233_233055

-- Part (1)
theorem sector_angle (r l : ℝ) (α : ℝ) :
  2 * r + l = 10 → (1 / 2) * l * r = 4 → α = l / r → α = 1 / 2 :=
by
  intro h1 h2 h3
  sorry

-- Part (2)
theorem sector_max_area (r l : ℝ) (α S : ℝ) :
  2 * r + l = 40 → α = l / r → S = (1 / 2) * l * r →
  (∀ r' l' α' S', 2 * r' + l' = 40 → α' = l' / r' → S' = (1 / 2) * l' * r' → S ≤ S') →
  r = 10 ∧ α = 2 ∧ S = 100 :=
by
  intro h1 h2 h3 h4
  sorry

end sector_angle_sector_max_area_l233_233055


namespace upstream_distance_calc_l233_233678

noncomputable def speed_in_still_water : ℝ := 10.5
noncomputable def downstream_distance : ℝ := 45
noncomputable def downstream_time : ℝ := 3
noncomputable def upstream_time : ℝ := 3

theorem upstream_distance_calc : 
  ∃ (d v : ℝ), (10.5 + v) * downstream_time = downstream_distance ∧ 
               v = 4.5 ∧ 
               d = (10.5 - v) * upstream_time ∧ 
               d = 18 :=
by
  sorry

end upstream_distance_calc_l233_233678


namespace second_integer_is_ninety_point_five_l233_233924

theorem second_integer_is_ninety_point_five
  (n : ℝ)
  (first_integer fourth_integer : ℝ)
  (h1 : first_integer = n - 2)
  (h2 : fourth_integer = n + 1)
  (h_sum : first_integer + fourth_integer = 180) :
  n = 90.5 :=
by
  -- sorry to skip the proof
  sorry

end second_integer_is_ninety_point_five_l233_233924


namespace fraction_meaningful_l233_233320

-- Define the condition about the denominator not being zero.
def denominator_condition (x : ℝ) : Prop := x + 2 ≠ 0

-- The proof problem statement.
theorem fraction_meaningful (x : ℝ) : denominator_condition x ↔ x ≠ -2 :=
by
  -- Ensure that the Lean environment is aware this is a theorem statement.
  sorry -- Proof is omitted as instructed.

end fraction_meaningful_l233_233320


namespace find_solution_l233_233232

theorem find_solution (x : ℝ) (h : (5 + x / 3)^(1/3) = -4) : x = -207 :=
sorry

end find_solution_l233_233232


namespace gcd_pair_sum_ge_prime_l233_233878

theorem gcd_pair_sum_ge_prime
  (n : ℕ)
  (h_prime: Prime (2*n - 1))
  (a : Fin n → ℕ)
  (h_distinct: ∀ i j : Fin n, i ≠ j → a i ≠ a j) :
  ∃ i j : Fin n, i ≠ j ∧ (a i + a j) / Nat.gcd (a i) (a j) ≥ 2*n - 1 := sorry

end gcd_pair_sum_ge_prime_l233_233878


namespace find_matrix_N_l233_233486

def matrix2x2 := ℚ × ℚ × ℚ × ℚ

def apply_matrix (M : matrix2x2) (v : ℚ × ℚ) : ℚ × ℚ :=
  let (a, b, c, d) := M;
  let (x, y) := v;
  (a * x + b * y, c * x + d * y)

theorem find_matrix_N : ∃ (N : matrix2x2), 
  apply_matrix N (3, 1) = (5, -1) ∧ 
  apply_matrix N (1, -2) = (0, 6) ∧ 
  N = (10/7, 5/7, 4/7, -19/7) :=
by {
  sorry
}

end find_matrix_N_l233_233486


namespace sin_alpha_of_terminal_side_l233_233413

theorem sin_alpha_of_terminal_side (α : ℝ) (P : ℝ × ℝ) 
  (hP : P = (5, 12)) :
  Real.sin α = 12 / 13 := sorry

end sin_alpha_of_terminal_side_l233_233413


namespace distance_scientific_notation_l233_233833

theorem distance_scientific_notation :
  55000000 = 5.5 * 10^7 :=
sorry

end distance_scientific_notation_l233_233833


namespace values_of_a_for_single_root_l233_233922

theorem values_of_a_for_single_root (a : ℝ) :
  (∃ (x : ℝ), ax^2 - 4 * x + 2 = 0) ∧ (∀ (x1 x2 : ℝ), ax^2 - 4 * x1 + 2 = 0 → ax^2 - 4 * x2 + 2 = 0 → x1 = x2) ↔ a = 0 ∨ a = 2 :=
sorry

end values_of_a_for_single_root_l233_233922


namespace cookies_per_child_l233_233109

theorem cookies_per_child 
  (total_cookies : ℕ) 
  (children : ℕ) 
  (x : ℚ) 
  (adults_fraction : total_cookies * x = total_cookies / 4) 
  (remaining_cookies : total_cookies - total_cookies * x = 180) 
  (correct_fraction : x = 1 / 4) 
  (correct_children : children = 6) :
  (total_cookies - total_cookies * x) / children = 30 := by
  sorry

end cookies_per_child_l233_233109


namespace largest_x_undefined_largest_solution_l233_233517

theorem largest_x_undefined (x : ℝ) :
  (10 * x ^ 2 - 85 * x + 10 = 0) → x = 10 ∨ x = 1 / 10 :=
by
  sorry

theorem largest_solution (x : ℝ) :
  (10 * x ^ 2 - 85 * x + 10 = 0 → x ≤ 10) :=
by
  sorry

end largest_x_undefined_largest_solution_l233_233517


namespace centroid_coordinates_satisfy_l233_233053

noncomputable def P : ℝ × ℝ := (2, 5)
noncomputable def Q : ℝ × ℝ := (-1, 3)
noncomputable def R : ℝ × ℝ := (4, -2)

noncomputable def S : ℝ × ℝ := (
  (P.1 + Q.1 + R.1) / 3,
  (P.2 + Q.2 + R.2) / 3
)

theorem centroid_coordinates_satisfy :
  4 * S.1 + 3 * S.2 = 38 / 3 :=
by
  -- Proof will be added here
  sorry

end centroid_coordinates_satisfy_l233_233053


namespace total_dogs_equation_l233_233238

/-- Definition of the number of boxes and number of dogs per box. --/
def num_boxes : ℕ := 7
def dogs_per_box : ℕ := 4

/-- The total number of dogs --/
theorem total_dogs_equation : num_boxes * dogs_per_box = 28 := by 
  sorry

end total_dogs_equation_l233_233238


namespace max_value_of_expression_l233_233104

theorem max_value_of_expression (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h_sum : x + y + z = 1) :
  x^4 * y^3 * z^2 ≤ 1024 / 14348907 :=
sorry

end max_value_of_expression_l233_233104


namespace total_yardage_progress_l233_233281

def teamA_moves : List Int := [-5, 8, -3, 6]
def teamB_moves : List Int := [4, -2, 9, -7]

theorem total_yardage_progress :
  (teamA_moves.sum + teamB_moves.sum) = 10 :=
by
  sorry

end total_yardage_progress_l233_233281


namespace number_of_men_l233_233861

theorem number_of_men (M : ℕ) (h : M * 40 = 20 * 68) : M = 34 :=
by
  sorry

end number_of_men_l233_233861


namespace exists_geometric_arithmetic_progressions_l233_233874

theorem exists_geometric_arithmetic_progressions (n : ℕ) (hn : n > 3) :
  ∃ (x y : ℕ → ℕ),
  (∀ m < n, x (m + 1) = (1 + ε)^m ∧ y (m + 1) = (1 + (m + 1) * ε - δ)) ∧
  ∀ m < n, x m < y m ∧ y m < x (m + 1) :=
by
  sorry

end exists_geometric_arithmetic_progressions_l233_233874


namespace num_diagonals_tetragon_l233_233201

def num_diagonals_in_polygon (n : ℕ) : ℕ :=
  n * (n - 3) / 2

theorem num_diagonals_tetragon : num_diagonals_in_polygon 4 = 2 := by
  sorry

end num_diagonals_tetragon_l233_233201


namespace jelly_bean_remaining_l233_233288

theorem jelly_bean_remaining (J : ℕ) (P : ℕ) (taken_last_4_each : ℕ) (taken_first_each : ℕ) 
 (taken_last_total : ℕ) (taken_first_total : ℕ) (taken_total : ℕ) (remaining : ℕ) :
  J = 8000 →
  P = 10 →
  taken_last_4_each = 400 →
  taken_first_each = 2 * taken_last_4_each →
  taken_last_total = 4 * taken_last_4_each →
  taken_first_total = 6 * taken_first_each →
  taken_total = taken_last_total + taken_first_total →
  remaining = J - taken_total →
  remaining = 1600 :=
by
  intros
  sorry  

end jelly_bean_remaining_l233_233288


namespace cards_given_to_Jeff_l233_233343

-- Definitions according to the conditions
def initial_cards : Nat := 304
def remaining_cards : Nat := 276

-- The proof problem
theorem cards_given_to_Jeff : initial_cards - remaining_cards = 28 :=
by
  sorry

end cards_given_to_Jeff_l233_233343


namespace find_x_given_ratio_constant_l233_233541

theorem find_x_given_ratio_constant (x y : ℚ) (k : ℚ)
  (h1 : ∀ x y, (2 * x - 5) / (y + 20) = k)
  (h2 : (2 * 7 - 5) / (6 + 20) = k)
  (h3 : y = 21) :
  x = 499 / 52 :=
by
  sorry

end find_x_given_ratio_constant_l233_233541


namespace cos_135_eq_neg_sqrt2_div_2_l233_233264

theorem cos_135_eq_neg_sqrt2_div_2 : 
  Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt2_div_2_l233_233264


namespace one_plane_halves_rect_prism_l233_233445

theorem one_plane_halves_rect_prism :
  ∀ (T : Type) (a b c : ℝ)
  (x y z : ℝ) 
  (black_prisms_volume white_prisms_volume : ℝ),
  (black_prisms_volume = (x * y * z + x * (b - y) * (c - z) + (a - x) * y * (c - z) + (a - x) * (b - y) * z)) ∧
  (white_prisms_volume = ((a - x) * (b - y) * (c - z) + (a - x) * y * z + x * (b - y) * z + x * y * (c - z))) ∧
  (black_prisms_volume = white_prisms_volume) →
  (x = a / 2 ∨ y = b / 2 ∨ z = c / 2) :=
by
  sorry

end one_plane_halves_rect_prism_l233_233445


namespace intersection_eq_l233_233352

open Set

def setA : Set ℤ := {x | x ≥ -4}
def setB : Set ℤ := {x | x ≤ 3}

theorem intersection_eq : (setA ∩ setB) = {x | -4 ≤ x ∧ x ≤ 3} := by
  sorry

end intersection_eq_l233_233352


namespace find_r_and_s_l233_233600

theorem find_r_and_s (r s : ℝ) :
  (∀ m : ℝ, ¬(∃ x : ℝ, x^2 + 10 * x = m * (x - 10) + 5) ↔ r < m ∧ m < s) →
  r + s = 60 :=
sorry

end find_r_and_s_l233_233600


namespace absent_children_l233_233723

theorem absent_children (A : ℕ) (h1 : 2 * 610 = (610 - A) * 4) : A = 305 := 
by sorry

end absent_children_l233_233723


namespace vasya_can_construct_polyhedron_l233_233641

-- Definition of a polyhedron using given set of shapes
-- where the original set of shapes can form a polyhedron
def original_set_can_form_polyhedron (squares triangles : ℕ) : Prop :=
  squares = 1 ∧ triangles = 4

-- Transformation condition: replacing 2 triangles with 2 squares
def replacement_condition (initial_squares initial_triangles replaced_squares replaced_triangles : ℕ) : Prop :=
  initial_squares + 2 = replaced_squares ∧ initial_triangles - 2 = replaced_triangles

-- Proving that new set of shapes can form a polyhedron
theorem vasya_can_construct_polyhedron :
  ∃ (new_squares new_triangles : ℕ),
    (original_set_can_form_polyhedron 1 4)
    ∧ (replacement_condition 1 4 new_squares new_triangles)
    ∧ (new_squares = 3 ∧ new_triangles = 2) :=
by
  sorry

end vasya_can_construct_polyhedron_l233_233641


namespace length_of_each_part_l233_233741

-- Definitions from the conditions
def total_length_in_inches : ℕ := 6 * 12 + 8
def number_of_parts : ℕ := 4

-- Proof statement
theorem length_of_each_part : total_length_in_inches / number_of_parts = 20 :=
by
  sorry

end length_of_each_part_l233_233741


namespace like_terms_mn_eq_neg1_l233_233076

variable (m n : ℤ)

theorem like_terms_mn_eq_neg1
  (hx : m + 3 = 4)
  (hy : n + 3 = 1) :
  m + n = -1 :=
sorry

end like_terms_mn_eq_neg1_l233_233076


namespace sum_of_areas_of_triangles_l233_233156

noncomputable def triangle_sum_of_box (a b c : ℝ) :=
  let face_triangles_area := 4 * ((a * b + a * c + b * c) / 2)
  let perpendicular_triangles_area := 4 * ((a * c + b * c) / 2)
  let oblique_triangles_area := 8 * Real.sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c))
  face_triangles_area + perpendicular_triangles_area + oblique_triangles_area

theorem sum_of_areas_of_triangles :
  triangle_sum_of_box 2 3 4 = 168 + k * Real.sqrt p := sorry

end sum_of_areas_of_triangles_l233_233156


namespace sum_of_money_l233_233540

theorem sum_of_money (A B C : ℝ) (hB : B = 0.65 * A) (hC : C = 0.40 * A) (hC_value : C = 32) :
  A + B + C = 164 :=
by
  sorry

end sum_of_money_l233_233540


namespace double_espresso_cost_l233_233028

-- Define the cost of coffee, days, and total spent as constants
def iced_coffee : ℝ := 2.5
def total_days : ℝ := 20
def total_spent : ℝ := 110

-- Define the cost of double espresso as variable E
variable (E : ℝ)

-- The proposition to prove
theorem double_espresso_cost : (total_days * (E + iced_coffee) = total_spent) → (E = 3) :=
by
  sorry

end double_espresso_cost_l233_233028


namespace cups_needed_correct_l233_233803

-- Define the conditions
def servings : ℝ := 18.0
def cups_per_serving : ℝ := 2.0

-- Define the total cups needed calculation
def total_cups (servings : ℝ) (cups_per_serving : ℝ) : ℝ :=
  servings * cups_per_serving

-- State the proof problem
theorem cups_needed_correct :
  total_cups servings cups_per_serving = 36.0 :=
by
  sorry

end cups_needed_correct_l233_233803


namespace expIConjugate_l233_233665

open Complex

-- Define the given condition
def expICondition (θ φ : ℝ) : Prop :=
  Complex.exp (Complex.I * θ) + Complex.exp (Complex.I * φ) = (1/3 : ℂ) + (2/5 : ℂ) * Complex.I

-- The theorem we want to prove
theorem expIConjugate (θ φ : ℝ) (h : expICondition θ φ) : 
  Complex.exp (-Complex.I * θ) + Complex.exp (-Complex.I * φ) = (1/3 : ℂ) - (2/5 : ℂ) * Complex.I :=
sorry

end expIConjugate_l233_233665


namespace line_tangent_to_circle_l233_233990

noncomputable def circle_diameter : ℝ := 13
noncomputable def distance_from_center_to_line : ℝ := 6.5

theorem line_tangent_to_circle :
  ∀ (d r : ℝ), d = 13 → r = 6.5 → r = d/2 → distance_from_center_to_line = r → 
  (distance_from_center_to_line = r) := 
by
  intros d r hdiam hdist hradius hdistance
  sorry

end line_tangent_to_circle_l233_233990


namespace base7_to_base10_div_l233_233646

theorem base7_to_base10_div (x y : ℕ) (h : 546 = x * 10^2 + y * 10 + 9) : (x + y + 9) / 21 = 6 / 7 :=
by {
  sorry
}

end base7_to_base10_div_l233_233646


namespace find_radius_l233_233235

theorem find_radius (r : ℝ) :
  (135 * r * Real.pi) / 180 = 3 * Real.pi → r = 4 :=
by
  sorry

end find_radius_l233_233235


namespace george_speed_second_segment_l233_233381

theorem george_speed_second_segment 
  (distance_total : ℝ)
  (speed_normal : ℝ)
  (distance_first : ℝ)
  (speed_first : ℝ) : 
  distance_total = 1 ∧ 
  speed_normal = 3 ∧ 
  distance_first = 0.5 ∧ 
  speed_first = 2 →
  (distance_first / speed_first + 0.5 * speed_second = 1 / speed_normal → speed_second = 6) :=
sorry

end george_speed_second_segment_l233_233381


namespace three_colored_flag_l233_233743

theorem three_colored_flag (colors : Finset ℕ) (h : colors.card = 6) : 
  (∃ top middle bottom : ℕ, top ≠ middle ∧ top ≠ bottom ∧ middle ≠ bottom ∧ 
                            top ∈ colors ∧ middle ∈ colors ∧ bottom ∈ colors) → 
  colors.card * (colors.card - 1) * (colors.card - 2) = 120 :=
by 
  intro h_exists
  exact sorry

end three_colored_flag_l233_233743


namespace multiply_105_95_l233_233781

theorem multiply_105_95 : 105 * 95 = 9975 :=
by
  sorry

end multiply_105_95_l233_233781


namespace cage_cost_correct_l233_233696

def cost_of_cat_toy : Real := 10.22
def total_cost_of_purchases : Real := 21.95
def cost_of_cage : Real := total_cost_of_purchases - cost_of_cat_toy

theorem cage_cost_correct : cost_of_cage = 11.73 := by
  sorry

end cage_cost_correct_l233_233696


namespace jill_third_month_days_l233_233418

theorem jill_third_month_days :
  ∀ (days : ℕ),
    (earnings_first_month : ℕ) = 10 * 30 →
    (earnings_second_month : ℕ) = 20 * 30 →
    (total_earnings : ℕ) = 1200 →
    (total_earnings_two_months : ℕ) = earnings_first_month + earnings_second_month →
    (earnings_third_month : ℕ) = total_earnings - total_earnings_two_months →
    earnings_third_month = 300 →
    days = earnings_third_month / 20 →
    days = 15 := 
sorry

end jill_third_month_days_l233_233418


namespace geom_seq_property_l233_233952

noncomputable def a_n : ℕ → ℝ := sorry  -- The definition of the geometric sequence

theorem geom_seq_property (a_n : ℕ → ℝ) (h : a_n 6 + a_n 8 = 4) :
  a_n 8 * (a_n 4 + 2 * a_n 6 + a_n 8) = 16 := by
sorry

end geom_seq_property_l233_233952


namespace maximum_x_minus_y_l233_233087

theorem maximum_x_minus_y (x y : ℝ) (h : 3 * (x^2 + y^2) = x + y) : x - y ≤ 2 * Real.sqrt 3 / 3 := by
  sorry

end maximum_x_minus_y_l233_233087


namespace find_k_l233_233603

-- Define the function y = kx
def linear_function (k x : ℝ) : ℝ := k * x

-- Define the point P(3,1)
def P : ℝ × ℝ := (3, 1)

theorem find_k (k : ℝ) (h : linear_function k 3 = 1) : k = 1 / 3 :=
by
  sorry

end find_k_l233_233603


namespace sum_of_squares_of_rates_l233_233724

variable (b j s : ℤ) -- rates in km/h
-- conditions
def ed_condition : Prop := 3 * b + 4 * j + 2 * s = 86
def sue_condition : Prop := 5 * b + 2 * j + 4 * s = 110

theorem sum_of_squares_of_rates (b j s : ℤ) (hEd : ed_condition b j s) (hSue : sue_condition b j s) : 
  b^2 + j^2 + s^2 = 3349 := 
sorry

end sum_of_squares_of_rates_l233_233724


namespace eliana_total_steps_l233_233026

noncomputable def day1_steps : ℕ := 200 + 300
noncomputable def day2_steps : ℕ := 2 * day1_steps
noncomputable def day3_steps : ℕ := day1_steps + day2_steps + 100

theorem eliana_total_steps : day3_steps = 1600 := by
  sorry

end eliana_total_steps_l233_233026


namespace probability_XiaoYu_group_A_l233_233255

theorem probability_XiaoYu_group_A :
  ∀ (students : Fin 48) (groups : Fin 4) (groupAssignment : Fin 48 → Fin 4)
    (student : Fin 48) (groupA : Fin 4),
    (∀ (s : Fin 48), ∃ (g : Fin 4), groupAssignment s = g) → 
    (∀ (g : Fin 4), ∃ (count : ℕ), (0 < count ∧ count ≤ 12) ∧
       (∃ (groupMembers : List (Fin 48)), groupMembers.length = count ∧
        (∀ (m : Fin 48), m ∈ groupMembers → groupAssignment m = g))) →
    (groupAssignment student = groupA) →
  ∃ (p : ℚ), p = (1/4) ∧ ∀ (s : Fin 48), groupAssignment s = groupA → p = (1/4) :=
by
  sorry

end probability_XiaoYu_group_A_l233_233255


namespace angle_equality_l233_233154

variables {Point Circle : Type}
variables (K O1 O2 P1 P2 Q1 Q2 M1 M2 : Point)
variables (W1 W2 : Circle)
variables (midpoint : Point → Point → Point)
variables (is_center : Point → Circle → Prop)
variables (intersects_at : Circle → Circle → Point → Prop)
variables (common_tangent_points : Circle → Circle → (Point × Point) × (Point × Point) → Prop)
variables (intersect_circle_at : Circle → Line → Point → Point → Prop)
variables (angle : Point → Point → Point → ℝ) -- to denote the angle measure between three points

-- Conditions
axiom K_intersection : intersects_at W1 W2 K
axiom O1_center : is_center O1 W1
axiom O2_center : is_center O2 W2
axiom tangents_meet_at : common_tangent_points W1 W2 ((P1, Q1), (P2, Q2))
axiom M1_midpoint : M1 = midpoint P1 Q1
axiom M2_midpoint : M2 = midpoint P2 Q2

-- The statement to prove
theorem angle_equality : angle O1 K O2 = angle M1 K M2 := 
  sorry

end angle_equality_l233_233154


namespace three_digit_number_with_units5_and_hundreds3_divisible_by_9_l233_233483

theorem three_digit_number_with_units5_and_hundreds3_divisible_by_9 :
  ∃ n : ℕ, ∃ x : ℕ, n = 305 + 10 * x ∧ (n % 9) = 0 ∧ n = 315 := by
sorry

end three_digit_number_with_units5_and_hundreds3_divisible_by_9_l233_233483


namespace number_of_friends_l233_233571

-- Define the initial amount of money John had
def initial_money : ℝ := 20.10 

-- Define the amount spent on sweets
def sweets_cost : ℝ := 1.05 

-- Define the amount given to each friend
def money_per_friend : ℝ := 1.00 

-- Define the amount of money left after giving to friends
def final_money : ℝ := 17.05 

-- Define a theorem to find the number of friends John gave money to
theorem number_of_friends (init_money sweets_cost money_per_friend final_money : ℝ) : 
  (init_money - sweets_cost - final_money) / money_per_friend = 2 :=
by
  sorry

end number_of_friends_l233_233571


namespace ethanol_concentration_l233_233882

theorem ethanol_concentration
  (w1 : ℕ) (c1 : ℝ) (w2 : ℕ) (c2 : ℝ)
  (hw1 : w1 = 400) (hc1 : c1 = 0.30)
  (hw2 : w2 = 600) (hc2 : c2 = 0.80) :
  (c1 * w1 + c2 * w2) / (w1 + w2) = 0.60 := 
by
  sorry

end ethanol_concentration_l233_233882


namespace shaded_area_correct_l233_233396

-- Definition of the grid dimensions
def grid_width : ℕ := 15
def grid_height : ℕ := 5

-- Definition of the heights of the shaded regions in segments
def shaded_height (x : ℕ) : ℕ :=
if x < 4 then 2
else if x < 9 then 3
else if x < 13 then 4
else if x < 15 then 5
else 0

-- Definition for the area of the entire grid
def grid_area : ℝ := grid_width * grid_height

-- Definition for the area of the unshaded triangle
def unshaded_triangle_area : ℝ := 0.5 * grid_width * grid_height

-- Definition for the area of the shaded region
def shaded_area : ℝ := grid_area - unshaded_triangle_area

-- The theorem to be proved
theorem shaded_area_correct : shaded_area = 37.5 :=
by
  sorry

end shaded_area_correct_l233_233396


namespace dealer_cannot_prevent_goal_l233_233522

theorem dealer_cannot_prevent_goal (m n : ℕ) :
  (m + n) % 4 = 0 :=
sorry

end dealer_cannot_prevent_goal_l233_233522


namespace average_speed_jeffrey_l233_233189
-- Import the necessary Lean library.

-- Initial conditions in the problem, restated as Lean definitions.
def distance_jog (d : ℝ) : Prop := d = 3
def speed_jog (s : ℝ) : Prop := s = 4
def distance_walk (d : ℝ) : Prop := d = 4
def speed_walk (s : ℝ) : Prop := s = 3

-- Target statement to prove using Lean.
theorem average_speed_jeffrey :
  ∀ (dj sj dw sw : ℝ), distance_jog dj → speed_jog sj → distance_walk dw → speed_walk sw →
    (dj + dw) / ((dj / sj) + (dw / sw)) = 3.36 := 
  by
    intros dj sj dw sw hj hs hw hw
    sorry

end average_speed_jeffrey_l233_233189


namespace boys_variance_greater_than_girls_l233_233567

noncomputable def variance (scores : List ℝ) : ℝ :=
  let mean := (List.sum scores) / (scores.length : ℝ)
  List.sum (scores.map (λ x => (x - mean) ^ 2)) / (scores.length : ℝ)

def boys_scores : List ℝ := [86, 94, 88, 92, 90]
def girls_scores : List ℝ := [88, 93, 93, 88, 93]

theorem boys_variance_greater_than_girls :
  variance boys_scores > variance girls_scores :=
by
  sorry

end boys_variance_greater_than_girls_l233_233567


namespace time_to_return_l233_233155

-- Given conditions
def distance : ℝ := 1000
def return_speed : ℝ := 142.85714285714286

-- Goal to prove
theorem time_to_return : distance / return_speed = 7 := 
by
  sorry

end time_to_return_l233_233155


namespace ceil_minus_eq_zero_l233_233187

theorem ceil_minus_eq_zero (x : ℝ) (h : ⌈x⌉ - ⌊x⌋ = 0) : ⌈x⌉ - x = 0 :=
sorry

end ceil_minus_eq_zero_l233_233187


namespace luke_earning_problem_l233_233127

variable (WeedEarning Weeks SpendPerWeek MowingEarning : ℤ)

theorem luke_earning_problem
  (h1 : WeedEarning = 18)
  (h2 : Weeks = 9)
  (h3 : SpendPerWeek = 3)
  (h4 : MowingEarning + WeedEarning = Weeks * SpendPerWeek) :
  MowingEarning = 9 := by
  sorry

end luke_earning_problem_l233_233127


namespace inscribed_sphere_radius_base_height_l233_233655

noncomputable def radius_of_inscribed_sphere (r base_radius height : ℝ) := 
  r = (30 / (Real.sqrt 5 + 1)) * (Real.sqrt 5 - 1) 

theorem inscribed_sphere_radius_base_height (r : ℝ) (b d : ℝ) (base_radius height : ℝ) 
  (h_base: base_radius = 15) (h_height: height = 30) 
  (h_radius: radius_of_inscribed_sphere r base_radius height) 
  (h_expr: r = b * (Real.sqrt d) - b) : 
  b + d = 12.5 :=
sorry

end inscribed_sphere_radius_base_height_l233_233655


namespace fill_cistern_l233_233275

theorem fill_cistern (p_rate q_rate : ℝ) (total_time first_pipe_time : ℝ) (remaining_fraction : ℝ): 
  p_rate = 1/12 → q_rate = 1/15 → total_time = 2 → remaining_fraction = 7/10 → 
  (remaining_fraction / q_rate) = 10.5 :=
by
  sorry

end fill_cistern_l233_233275


namespace ladder_distance_from_wall_l233_233563

theorem ladder_distance_from_wall (h a b : ℕ) (h_hyp : h = 13) (h_wall : a = 12) :
  a^2 + b^2 = h^2 → b = 5 :=
by
  intros h_eq
  sorry

end ladder_distance_from_wall_l233_233563


namespace six_nine_op_l233_233946

variable (m n : ℚ)

def op (x y : ℚ) : ℚ := m^2 * x + n * y - 1

theorem six_nine_op :
  (op m n 2 3 = 3) →
  (op m n 6 9 = 11) :=
by
  intro h
  sorry

end six_nine_op_l233_233946


namespace combined_weight_of_candles_l233_233842

theorem combined_weight_of_candles :
  (∀ (candles: ℕ), ∀ (weight_per_candle: ℕ),
    (weight_per_candle = 8 + 1) →
    (candles = 10 - 3) →
    (candles * weight_per_candle = 63)) :=
by
  intros
  sorry

end combined_weight_of_candles_l233_233842


namespace solve_equations_l233_233299

theorem solve_equations (x : ℝ) :
  (3 * x^2 = 27 → x = 3 ∨ x = -3) ∧
  (2 * x^2 + x = 55 → x = 5 ∨ x = -5.5) ∧
  (2 * x^2 + 18 = 15 * x → x = 6 ∨ x = 1.5) :=
by
  sorry

end solve_equations_l233_233299


namespace subtract_two_decimals_l233_233562

theorem subtract_two_decimals : 3.75 - 1.46 = 2.29 := by
  sorry

end subtract_two_decimals_l233_233562


namespace total_cost_28_oranges_avg_cost_per_orange_cost_6_oranges_l233_233965

-- Initial conditions
def cost_4_oranges : Nat := 12
def cost_7_oranges : Nat := 28
def total_oranges : Nat := 28

-- Calculate the total cost for 28 oranges
theorem total_cost_28_oranges
  (x y : Nat) 
  (h1 : 4 * x + 7 * y = total_oranges) 
  (h2 : total_oranges = 28) 
  (h3 : x = 7) 
  (h4 : y = 0) : 
  7 * cost_4_oranges = 84 := 
by sorry

-- Calculate the average cost per orange
theorem avg_cost_per_orange 
  (total_cost : Nat) 
  (h1 : total_cost = 84)
  (h2 : total_oranges = 28) : 
  total_cost / total_oranges = 3 := 
by sorry

-- Calculate the cost for 6 oranges
theorem cost_6_oranges 
  (avg_cost : Nat)
  (h1 : avg_cost = 3)
  (n : Nat) 
  (h2 : n = 6) : 
  n * avg_cost = 18 := 
by sorry

end total_cost_28_oranges_avg_cost_per_orange_cost_6_oranges_l233_233965


namespace find_growth_rate_calculate_fourth_day_donation_l233_233435

-- Define the conditions
def first_day_donation : ℝ := 3000
def third_day_donation : ℝ := 4320
def growth_rate (x : ℝ) : Prop := (1 + x)^2 = third_day_donation / first_day_donation

-- Since the problem states growth rate for second and third day is the same,
-- we need to find that rate which is equivalent to solving the above proposition for x.

theorem find_growth_rate : ∃ x : ℝ, growth_rate x ∧ x = 0.2 := by
  sorry

-- Calculate the fourth day's donation based on the growth rate found.
def fourth_day_donation (third_day : ℝ) (growth_rate : ℝ) : ℝ :=
  third_day * (1 + growth_rate)

theorem calculate_fourth_day_donation : 
  ∀ x : ℝ, growth_rate x → x = 0.2 → fourth_day_donation third_day_donation x = 5184 := by 
  sorry

end find_growth_rate_calculate_fourth_day_donation_l233_233435


namespace incorrect_statement_isosceles_trapezoid_l233_233624

-- Define the properties of an isosceles trapezoid
structure IsoscelesTrapezoid (a b c d : ℝ) :=
  (parallel_bases : a = c ∨ b = d)  -- Bases are parallel
  (equal_diagonals : a = b) -- Diagonals are equal
  (equal_angles : ∀ α β : ℝ, α = β)  -- Angles on the same base are equal
  (axisymmetric : ∀ x : ℝ, x = -x)  -- Is an axisymmetric figure

-- Prove that the statement "The two bases of an isosceles trapezoid are parallel and equal" is incorrect
theorem incorrect_statement_isosceles_trapezoid (a b c d : ℝ) (h : IsoscelesTrapezoid a b c d) :
  ¬ (a = c ∧ b = d) :=
sorry

end incorrect_statement_isosceles_trapezoid_l233_233624


namespace min_value_abs_expression_l233_233679

theorem min_value_abs_expression {p x : ℝ} (hp1 : 0 < p) (hp2 : p < 15) (hx1 : p ≤ x) (hx2 : x ≤ 15) :
  |x - p| + |x - 15| + |x - p - 15| = 15 :=
sorry

end min_value_abs_expression_l233_233679


namespace polygon_diagonals_l233_233887

theorem polygon_diagonals (D n : ℕ) (hD : D = 20) (hFormula : D = n * (n - 3) / 2) :
  n = 8 :=
by
  -- The proof goes here
  sorry

end polygon_diagonals_l233_233887


namespace max_n_value_l233_233361

theorem max_n_value (a b c : ℝ) (n : ℕ) (h1 : a > b) (h2 : b > c) (h3 : 1/(a - b) + 1/(b - c) ≥ n / (a - c)) :
  n ≤ 4 := 
sorry

end max_n_value_l233_233361


namespace problem_solution_l233_233192

section
variables (a b : ℝ)

-- Definition of the \* operation
def star_op (a b : ℝ) : ℝ := (a + 1) * (b - 1)

-- Definition of a^{*2} as a \* a
def star_square (a : ℝ) : ℝ := star_op a a

-- Define the specific problem instance with x = 2
def problem_expr : ℝ := star_op 3 (star_square 2) - star_op 2 2 + 1

-- Theorem stating the correct answer
theorem problem_solution : problem_expr = 6 := by
  -- Proof steps, marked as 'sorry'
  sorry

end

end problem_solution_l233_233192


namespace part1_part2_l233_233683

def z1 (a : ℝ) : Complex := Complex.mk 2 a
def z2 : Complex := Complex.mk 3 (-4)

-- Part 1: Prove that the product of z1 and z2 equals 10 - 5i when a = 1.
theorem part1 : z1 1 * z2 = Complex.mk 10 (-5) :=
by
  -- proof to be filled in
  sorry

-- Part 2: Prove that a = 4 when z1 + z2 is a real number.
theorem part2 (a : ℝ) (h : (z1 a + z2).im = 0) : a = 4 :=
by
  -- proof to be filled in
  sorry

end part1_part2_l233_233683


namespace find_a_value_l233_233564

theorem find_a_value (a x y : ℝ) (h1 : x = 4) (h2 : y = 5) (h3 : a * x - 2 * y = 2) : a = 3 :=
by
  sorry

end find_a_value_l233_233564


namespace distance_from_A_to_y_axis_is_2_l233_233256

-- Define the point A
def point_A : ℝ × ℝ := (-2, 1)

-- Define the distance function to the y-axis
def distance_to_y_axis (p : ℝ × ℝ) : ℝ :=
  abs p.1

-- The theorem to prove
theorem distance_from_A_to_y_axis_is_2 : distance_to_y_axis point_A = 2 :=
by
  sorry

end distance_from_A_to_y_axis_is_2_l233_233256


namespace max_piece_length_l233_233401

theorem max_piece_length (a b c : ℕ) (h1 : a = 60) (h2 : b = 75) (h3 : c = 90) :
  Nat.gcd (Nat.gcd a b) c = 15 :=
by 
  sorry

end max_piece_length_l233_233401


namespace problem1_problem2_l233_233022

theorem problem1 : 3 * Real.sqrt 3 - Real.sqrt 8 + Real.sqrt 2 - Real.sqrt 27 = -Real.sqrt 2 := 
by sorry

theorem problem2 : (Real.sqrt 5 - Real.sqrt 3) * (Real.sqrt 5 + Real.sqrt 3) = 2 := 
by sorry

end problem1_problem2_l233_233022


namespace determine_BD_l233_233476

variable (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
variables (AB BC CD DA : ℝ)
variables (BD : ℝ)

-- Setting up the conditions:
axiom AB_eq_5 : AB = 5
axiom BC_eq_17 : BC = 17
axiom CD_eq_5 : CD = 5
axiom DA_eq_9 : DA = 9
axiom BD_is_integer : ∃ (n : ℤ), BD = n

theorem determine_BD : BD = 13 :=
by
  sorry

end determine_BD_l233_233476


namespace graph_passes_through_point_l233_233838

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x-2) + 1

theorem graph_passes_through_point (a : ℝ) (h_pos : a > 0) (h_neq : a ≠ 1) : f a 2 = 2 :=
by
  sorry

end graph_passes_through_point_l233_233838


namespace necessary_and_sufficient_condition_l233_233778

variable {A B : Prop}

theorem necessary_and_sufficient_condition (h1 : A → B) (h2 : B → A) : A ↔ B := 
by 
  sorry

end necessary_and_sufficient_condition_l233_233778


namespace potato_slice_length_l233_233602

theorem potato_slice_length (x : ℕ) (h1 : 600 = x + (x + 50)) : x + 50 = 325 :=
by
  sorry

end potato_slice_length_l233_233602


namespace Jeanine_has_more_pencils_than_Clare_l233_233231

def number_pencils_Jeanine_bought : Nat := 18
def number_pencils_Clare_bought := number_pencils_Jeanine_bought / 2
def number_pencils_given_to_Abby := number_pencils_Jeanine_bought / 3
def number_pencils_Jeanine_now := number_pencils_Jeanine_bought - number_pencils_given_to_Abby 

theorem Jeanine_has_more_pencils_than_Clare :
  number_pencils_Jeanine_now - number_pencils_Clare_bought = 3 := by
  sorry

end Jeanine_has_more_pencils_than_Clare_l233_233231


namespace snakes_in_cage_l233_233473

theorem snakes_in_cage (snakes_hiding : Nat) (snakes_not_hiding : Nat) (total_snakes : Nat) 
  (h : snakes_hiding = 64) (nh : snakes_not_hiding = 31) : 
  total_snakes = snakes_hiding + snakes_not_hiding := by
  sorry

end snakes_in_cage_l233_233473


namespace find_positive_n_l233_233259

def arithmetic_sequence (a d : ℤ) (n : ℤ) := a + (n - 1) * d

def sum_of_first_n_terms (a d : ℤ) (n : ℕ) : ℤ := (n * (2 * a + (n - 1) * d)) / 2

theorem find_positive_n :
  ∃ (n : ℕ), n > 0 ∧ ∀ a d : ℤ, a = -12 → sum_of_first_n_terms a d 13 = 0 → arithmetic_sequence a d n > 0 ∧ n = 8 := 
sorry

end find_positive_n_l233_233259


namespace minimize_b_plus_c_l233_233598

theorem minimize_b_plus_c (a b c : ℝ) (h1 : 0 < a)
  (h2 : ∀ x, (y : ℝ) = a * x^2 + b * x + c)
  (h3 : ∀ x, (yr : ℝ) = a * (x + 2)^2 + (a - 1)^2) :
  a = 1 :=
by
  sorry

end minimize_b_plus_c_l233_233598


namespace selling_price_l233_233181

def cost_price : ℝ := 76.92
def profit_rate : ℝ := 0.30

theorem selling_price : cost_price * (1 + profit_rate) = 100.00 := by
  sorry

end selling_price_l233_233181


namespace rectangular_solid_surface_area_l233_233265

noncomputable def is_prime (n : ℕ) : Prop := sorry

theorem rectangular_solid_surface_area (l w h : ℕ) (hl : is_prime l) (hw : is_prime w) (hh : is_prime h) (volume_eq_437 : l * w * h = 437) :
  2 * (l * w + w * h + h * l) = 958 :=
sorry

end rectangular_solid_surface_area_l233_233265


namespace base_six_digits_unique_l233_233002

theorem base_six_digits_unique (b : ℕ) (h : (b-1)^2*(b-2) = 100) : b = 6 :=
by
  sorry

end base_six_digits_unique_l233_233002


namespace problem_statement_l233_233246

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 6)

theorem problem_statement :
  f (5 * Real.pi / 24) = Real.sqrt 2 ∧
  ∀ x, f x ≥ 1 ↔ ∃ k : ℤ, k * Real.pi + Real.pi / 6 ≤ x ∧ x ≤ k * Real.pi + Real.pi / 2 :=
by
  sorry

end problem_statement_l233_233246


namespace sum_and_product_of_roots_cube_l233_233427

theorem sum_and_product_of_roots_cube (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : 
  x^3 + y^3 = 1008 := 
by {
  sorry
}

end sum_and_product_of_roots_cube_l233_233427


namespace four_spheres_max_intersections_l233_233759

noncomputable def max_intersection_points (n : Nat) : Nat :=
  if h : n > 0 then n * 2 else 0

theorem four_spheres_max_intersections : max_intersection_points 4 = 8 := by
  sorry

end four_spheres_max_intersections_l233_233759


namespace range_of_m_l233_233453

noncomputable def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, f x < |m - 2|) ↔ m < 0 ∨ m > 4 := 
sorry

end range_of_m_l233_233453


namespace rectangle_length_from_square_thread_l233_233108

theorem rectangle_length_from_square_thread (side_of_square width_of_rectangle : ℝ) (same_thread : Bool) 
  (h1 : side_of_square = 20) (h2 : width_of_rectangle = 14) (h3 : same_thread) : 
  ∃ length_of_rectangle : ℝ, length_of_rectangle = 26 := 
by
  sorry

end rectangle_length_from_square_thread_l233_233108


namespace game_ends_in_65_rounds_l233_233310

noncomputable def player_tokens_A : Nat := 20
noncomputable def player_tokens_B : Nat := 19
noncomputable def player_tokens_C : Nat := 18
noncomputable def player_tokens_D : Nat := 17

def rounds_until_game_ends (A B C D : Nat) : Nat :=
  -- Implementation to count the rounds will go here, but it is skipped for this statement-only task
  sorry

theorem game_ends_in_65_rounds : rounds_until_game_ends player_tokens_A player_tokens_B player_tokens_C player_tokens_D = 65 :=
  sorry

end game_ends_in_65_rounds_l233_233310


namespace initial_candies_proof_l233_233556

noncomputable def initial_candies (n : ℕ) := 
  ∃ c1 c2 c3 c4 c5 : ℕ, 
    c5 = 1 ∧
    c5 = n * 1 / 6 ∧
    c4 = n * 5 / 6 ∧
    c3 = n * 4 / 5 ∧
    c2 = n * 3 / 4 ∧
    c1 = n * 2 / 3 ∧
    n = 2 * c1

theorem initial_candies_proof (n : ℕ) : initial_candies n → n = 720 :=
  by
    sorry

end initial_candies_proof_l233_233556


namespace lines_coplanar_iff_k_eq_neg2_l233_233989

noncomputable def line1 (s k : ℝ) : ℝ × ℝ × ℝ :=
(2 + s, 4 - k * s, 2 + k * s)

noncomputable def line2 (t : ℝ) : ℝ × ℝ × ℝ :=
(t, 2 + 2 * t, 3 - t)

theorem lines_coplanar_iff_k_eq_neg2 :
  (∃ s t : ℝ, line1 s k = line2 t) → k = -2 :=
by
  sorry

end lines_coplanar_iff_k_eq_neg2_l233_233989


namespace find_x_l233_233077

variables (x : ℝ)

theorem find_x : (x / 4) * 12 = 9 → x = 3 :=
by
  sorry

end find_x_l233_233077


namespace union_of_sets_l233_233407

theorem union_of_sets (x y : ℕ) (A B : Set ℕ) (h1 : A = {x, y}) (h2 : B = {x + 1, 5}) (h3 : A ∩ B = {2}) : A ∪ B = {1, 2, 5} :=
sorry

end union_of_sets_l233_233407


namespace max_sum_of_factors_l233_233760

theorem max_sum_of_factors (A B C : ℕ) (h1 : A ≠ B) (h2 : B ≠ C) (h3 : A ≠ C) (h4 : 0 < A) (h5 : 0 < B) (h6 : 0 < C) (h7 : A * B * C = 3003) :
  A + B + C ≤ 45 :=
sorry

end max_sum_of_factors_l233_233760


namespace train_speed_l233_233599

theorem train_speed :
  ∀ (length : ℝ) (time : ℝ),
    length = 135 ∧ time = 3.4711508793582233 →
    (length / time) * 3.6 = 140.0004 :=
by
  sorry

end train_speed_l233_233599


namespace value_for_real_value_for_pure_imaginary_l233_233635

def is_real (z : ℂ) : Prop := z.im = 0
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

def value_conditions (k : ℝ) : ℂ := ⟨k^2 - 3*k - 4, k^2 - 5*k - 6⟩

theorem value_for_real (k : ℝ) : is_real (value_conditions k) ↔ (k = 6 ∨ k = -1) :=
by
  sorry

theorem value_for_pure_imaginary (k : ℝ) : is_pure_imaginary (value_conditions k) ↔ (k = 4) :=
by
  sorry

end value_for_real_value_for_pure_imaginary_l233_233635


namespace share_of_y_l233_233027

theorem share_of_y (A y z : ℝ)
  (hx : y = 0.45 * A)
  (hz : z = 0.30 * A)
  (h_total : A + y + z = 140) :
  y = 36 := by
  sorry

end share_of_y_l233_233027


namespace solve_3x_5y_eq_7_l233_233387

theorem solve_3x_5y_eq_7 :
  ∃ (x y k : ℤ), (3 * x + 5 * y = 7) ∧ (x = 4 + 5 * k) ∧ (y = -1 - 3 * k) :=
by 
  sorry

end solve_3x_5y_eq_7_l233_233387


namespace extra_large_yellow_curlers_l233_233941

def total_curlers : ℕ := 120
def small_pink_curlers : ℕ := total_curlers / 5
def medium_blue_curlers : ℕ := 2 * small_pink_curlers
def large_green_curlers : ℕ := total_curlers / 4

theorem extra_large_yellow_curlers : 
  total_curlers - small_pink_curlers - medium_blue_curlers - large_green_curlers = 18 :=
by
  sorry

end extra_large_yellow_curlers_l233_233941


namespace product_of_first_two_numbers_l233_233998

theorem product_of_first_two_numbers (A B C : ℕ) (h_coprime: Nat.gcd A B = 1 ∧ Nat.gcd B C = 1 ∧ Nat.gcd A C = 1)
  (h_product: B * C = 1073) (h_sum: A + B + C = 85) : A * B = 703 :=
sorry

end product_of_first_two_numbers_l233_233998


namespace sum_reciprocals_of_partial_fractions_l233_233915

noncomputable def f (s : ℝ) : ℝ := s^3 - 20 * s^2 + 125 * s - 500

theorem sum_reciprocals_of_partial_fractions :
  ∀ (p q r A B C : ℝ),
    p ≠ q ∧ q ≠ r ∧ r ≠ p ∧
    f p = 0 ∧ f q = 0 ∧ f r = 0 ∧
    (∀ s, s ≠ p ∧ s ≠ q ∧ s ≠ r → 
      (1 / f s = A / (s - p) + B / (s - q) + C / (s - r))) →
    1 / A + 1 / B + 1 / C = 720 :=
sorry

end sum_reciprocals_of_partial_fractions_l233_233915


namespace min_value_f_l233_233267

noncomputable def f (x : ℝ) : ℝ := (x^2 - x + 3) / (x - 1)

theorem min_value_f : ∀ (x : ℝ), x ≥ 3 → ∃ m : ℝ, m = 9/2 ∧ ∀ y : ℝ, f y ≥ m :=
by
  sorry

end min_value_f_l233_233267


namespace binary_multiplication_l233_233321

theorem binary_multiplication :
  let a := 0b1101
  let b := 0b111
  let product := 0b1000111
  a * b = product :=
by 
  let a := 0b1101
  let b := 0b111
  let product := 0b1000111
  sorry

end binary_multiplication_l233_233321


namespace domain_of_f_range_of_f_monotonic_increasing_interval_of_f_l233_233260

open Real

noncomputable def f (x : ℝ) : ℝ := log (9 - x^2)

theorem domain_of_f : Set.Ioo (-3 : ℝ) 3 = {x : ℝ | -3 < x ∧ x < 3} :=
by
  sorry

theorem range_of_f : ∃ y : ℝ, y ∈ Set.Iic (2 * log 3) :=
by
  sorry

theorem monotonic_increasing_interval_of_f : 
  {x : ℝ | -3 < x} ∩ {x : ℝ | 0 ≥ x} = Set.Ioc (-3 : ℝ) 0 :=
by
  sorry

end domain_of_f_range_of_f_monotonic_increasing_interval_of_f_l233_233260


namespace range_of_a_l233_233894

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then x^2 + 2 * x else -(x^2 + 2 * x)

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x : ℝ, x ≥ 0 → f x = x^2 + 2 * x) →
  f (2 - a^2) > f a ↔ -2 < a ∧ a < 1 :=
by
  sorry

end range_of_a_l233_233894


namespace SunshinePumpkinsCount_l233_233297

def MoonglowPumpkins := 14
def SunshinePumpkins := 3 * MoonglowPumpkins + 12

theorem SunshinePumpkinsCount : SunshinePumpkins = 54 :=
by
  -- proof goes here
  sorry

end SunshinePumpkinsCount_l233_233297


namespace triangle_inequality_l233_233386

theorem triangle_inequality 
  (a b c : ℝ) -- lengths of the sides of the triangle
  (α β γ : ℝ) -- angles of the triangle in radians opposite to sides a, b, c
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)  -- positivity of sides
  (hα : 0 < α ∧ α < π) (hβ : 0 < β ∧ β < π) (hγ : 0 < γ ∧ γ < π) -- positivity and range of angles
  (h_sum : α + β + γ = π) -- angle sum property of a triangle
: 
  b / Real.sin (γ + α / 3) + c / Real.sin (β + α / 3) > (2 / 3) * (a / Real.sin (α / 3)) :=
sorry

end triangle_inequality_l233_233386


namespace pizza_payment_l233_233033

theorem pizza_payment (n : ℕ) (cost : ℕ) (total : ℕ) 
  (h1 : n = 3) 
  (h2 : cost = 8) 
  (h3 : total = n * cost) : 
  total = 24 :=
by 
  rw [h1, h2] at h3 
  exact h3

end pizza_payment_l233_233033


namespace meaning_of_sum_of_squares_l233_233995

theorem meaning_of_sum_of_squares (a b : ℝ) : a ^ 2 + b ^ 2 ≠ 0 ↔ (a ≠ 0 ∨ b ≠ 0) :=
by
  sorry

end meaning_of_sum_of_squares_l233_233995


namespace max_value_of_quadratic_l233_233497

theorem max_value_of_quadratic :
  ∀ (x : ℝ), ∃ y : ℝ, y = -3 * x^2 + 18 ∧
  (∀ x' : ℝ, -3 * x'^2 + 18 ≤ y) := by
  sorry

end max_value_of_quadratic_l233_233497


namespace find_coefficients_l233_233695

-- Define the polynomial
def poly (a b : ℤ) (x : ℚ) : ℚ := a * x^4 + b * x^3 + 40 * x^2 - 20 * x + 8

-- Define the factor
def factor (x : ℚ) : ℚ := 3 * x^2 - 2 * x + 2

-- States that for a given polynomial and factor, the resulting (a, b) pair is (-51, 25)
theorem find_coefficients :
  ∃ a b c d : ℤ, 
  (∀ x, poly a b x = (factor x) * (c * x^2 + d * x + 4)) ∧ 
  a = -51 ∧ 
  b = 25 :=
by sorry

end find_coefficients_l233_233695


namespace expression_evaluation_l233_233111

theorem expression_evaluation :
  5 * 402 + 4 * 402 + 3 * 402 + 401 = 5225 := by
  sorry

end expression_evaluation_l233_233111


namespace equidistant_point_on_y_axis_l233_233636

theorem equidistant_point_on_y_axis :
  ∃ (y : ℝ), 0 < y ∧ 
  (dist (0, y) (-3, 0) = dist (0, y) (-2, 5)) ∧ 
  y = 2 :=
by
  sorry

end equidistant_point_on_y_axis_l233_233636


namespace sum_of_coefficients_l233_233150

theorem sum_of_coefficients (a b c : ℝ) (w : ℂ) (h_roots : ∃ w : ℂ, (∃ i : ℂ, i^2 = -1) ∧ 
  (x + ax^2 + bx + c)^3 = (w + 3*im)* (w + 9*im)*(2*w - 4)) :
  a + b + c = -136 :=
sorry

end sum_of_coefficients_l233_233150


namespace find_x_l233_233802

theorem find_x (u : ℕ) (h₁ : u = 90) (w : ℕ) (h₂ : w = u + 10)
                (z : ℕ) (h₃ : z = w + 25) (y : ℕ) (h₄ : y = z + 15)
                (x : ℕ) (h₅ : x = y + 3) : x = 143 :=
by {
  -- Proof will be included here
  sorry
}

end find_x_l233_233802


namespace gain_percentage_is_8_l233_233764

variable (C S : ℝ) (D : ℝ)
variable (h1 : 20 * C * (1 - D / 100) = 12 * S)
variable (h2 : D ≥ 5 ∧ D ≤ 25)

theorem gain_percentage_is_8 :
  (12 * S * 1.08 - 20 * C * (1 - D / 100)) / (20 * C * (1 - D / 100)) * 100 = 8 :=
by
  sorry

end gain_percentage_is_8_l233_233764


namespace max_value_of_function_l233_233123

noncomputable def function (x : ℝ) : ℝ := (Real.cos x) ^ 2 - Real.sin x

theorem max_value_of_function : ∃ x : ℝ, function x = 5 / 4 :=
by
  sorry

end max_value_of_function_l233_233123


namespace chalkboard_area_l233_233574

def width : Float := 3.5
def length : Float := 2.3 * width
def area : Float := length * width

theorem chalkboard_area : area = 28.175 :=
by 
  sorry

end chalkboard_area_l233_233574


namespace logic_problem_l233_233608

variable (p q : Prop)

theorem logic_problem (h₁ : p ∨ q) (h₂ : ¬ p) : ¬ p ∧ q :=
by
  sorry

end logic_problem_l233_233608


namespace liu_xing_statement_incorrect_l233_233120

-- Definitions of the initial statistics of the classes
def avg_score_class_91 : ℝ := 79.5
def avg_score_class_92 : ℝ := 80.2

-- Definitions of corrections applied
def correction_gain_class_91 : ℝ := 0.6 * 3
def correction_loss_class_91 : ℝ := 0.2 * 3
def correction_gain_class_92 : ℝ := 0.5 * 3
def correction_loss_class_92 : ℝ := 0.3 * 3

-- Definitions of corrected averages
def corrected_avg_class_91 : ℝ := avg_score_class_91 + correction_gain_class_91 - correction_loss_class_91
def corrected_avg_class_92 : ℝ := avg_score_class_92 + correction_gain_class_92 - correction_loss_class_92

-- Proof statement
theorem liu_xing_statement_incorrect : corrected_avg_class_91 ≤ corrected_avg_class_92 :=
by {
  -- Additional hints and preliminary calculations could be done here.
  sorry
}

end liu_xing_statement_incorrect_l233_233120


namespace gina_expenditure_l233_233671

noncomputable def gina_total_cost : ℝ :=
  let regular_classes_cost := 12 * 450
  let lab_classes_cost := 6 * 550
  let textbooks_cost := 3 * 150
  let online_resources_cost := 4 * 95
  let facilities_fee := 200
  let lab_fee := 6 * 75
  let total_cost := regular_classes_cost + lab_classes_cost + textbooks_cost + online_resources_cost + facilities_fee + lab_fee
  let scholarship_amount := 0.5 * regular_classes_cost
  let discount_amount := 0.25 * lab_classes_cost
  let adjusted_cost := total_cost - scholarship_amount - discount_amount
  let interest := 0.04 * adjusted_cost
  adjusted_cost + interest

theorem gina_expenditure : gina_total_cost = 5881.20 :=
by
  sorry

end gina_expenditure_l233_233671


namespace biking_days_in_week_l233_233152

def onurDistancePerDay : ℕ := 250
def hanilDistanceMorePerDay : ℕ := 40
def weeklyDistance : ℕ := 2700

theorem biking_days_in_week : (weeklyDistance / (onurDistancePerDay + hanilDistanceMorePerDay + onurDistancePerDay)) = 5 :=
by
  sorry

end biking_days_in_week_l233_233152


namespace total_kids_receive_macarons_l233_233491

theorem total_kids_receive_macarons :
  let mitch_good := 18
  let joshua := 26 -- 20 + 6
  let joshua_good := joshua - 3
  let miles := joshua * 2
  let miles_good := miles
  let renz := (3 * miles) / 4 - 1
  let renz_good := renz - 4
  let leah_good := 35 - 5
  let total_good := mitch_good + joshua_good + miles_good + renz_good + leah_good 
  let kids_with_3_macarons := 10
  let macaron_per_3 := kids_with_3_macarons * 3
  let remaining_macarons := total_good - macaron_per_3
  let kids_with_2_macarons := remaining_macarons / 2
  kids_with_3_macarons + kids_with_2_macarons = 73 :=
by 
  sorry

end total_kids_receive_macarons_l233_233491


namespace net_income_difference_l233_233638

-- Define Terry's and Jordan's daily income and working days
def terryDailyIncome : ℝ := 24
def terryWorkDays : ℝ := 7
def jordanDailyIncome : ℝ := 30
def jordanWorkDays : ℝ := 6

-- Define the tax rate
def taxRate : ℝ := 0.10

-- Calculate weekly gross incomes
def terryGrossWeeklyIncome : ℝ := terryDailyIncome * terryWorkDays
def jordanGrossWeeklyIncome : ℝ := jordanDailyIncome * jordanWorkDays

-- Calculate tax deductions
def terryTaxDeduction : ℝ := taxRate * terryGrossWeeklyIncome
def jordanTaxDeduction : ℝ := taxRate * jordanGrossWeeklyIncome

-- Calculate net weekly incomes
def terryNetWeeklyIncome : ℝ := terryGrossWeeklyIncome - terryTaxDeduction
def jordanNetWeeklyIncome : ℝ := jordanGrossWeeklyIncome - jordanTaxDeduction

-- Calculate the difference
def incomeDifference : ℝ := jordanNetWeeklyIncome - terryNetWeeklyIncome

-- The theorem to be proven
theorem net_income_difference :
  incomeDifference = 10.80 :=
by
  sorry

end net_income_difference_l233_233638


namespace find_missing_number_l233_233849

theorem find_missing_number (x : ℤ) : x + 64 = 16 → x = -48 := by
  intro h
  linarith

end find_missing_number_l233_233849


namespace total_clothing_l233_233862

def num_boxes : ℕ := 4
def scarves_per_box : ℕ := 2
def mittens_per_box : ℕ := 6

theorem total_clothing :
  num_boxes * scarves_per_box + num_boxes * mittens_per_box = 32 :=
by
  sorry

end total_clothing_l233_233862


namespace sin_neg_135_eq_neg_sqrt_2_over_2_l233_233142

theorem sin_neg_135_eq_neg_sqrt_2_over_2 :
  Real.sin (-135 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_neg_135_eq_neg_sqrt_2_over_2_l233_233142


namespace sum_of_largest_smallest_angles_l233_233480

noncomputable section

def sides_ratio (a b c : ℝ) : Prop := a / 5 = b / 7 ∧ b / 7 = c / 8

theorem sum_of_largest_smallest_angles (a b c : ℝ) (θA θB θC : ℝ) 
  (h1 : sides_ratio a b c) 
  (h2 : a^2 + b^2 - c^2 = 2 * a * b * Real.cos θC)
  (h3 : b^2 + c^2 - a^2 = 2 * b * c * Real.cos θA)
  (h4 : c^2 + a^2 - b^2 = 2 * c * a * Real.cos θB)
  (h5 : θA + θB + θC = 180) :
  θA + θC = 120 :=
sorry

end sum_of_largest_smallest_angles_l233_233480


namespace hypotenuse_of_right_triangle_l233_233944

theorem hypotenuse_of_right_triangle (a b : ℕ) (h_a : a = 8) (h_b : b = 15) : 
  ∃ c : ℕ, c = 17 ∧ c^2 = a^2 + b^2 :=
by
  sorry

end hypotenuse_of_right_triangle_l233_233944


namespace Jake_weight_l233_233465

variables (J S : ℝ)

theorem Jake_weight (h1 : 0.8 * J = 2 * S) (h2 : J + S = 168) : J = 120 :=
  sorry

end Jake_weight_l233_233465


namespace gcd_of_three_numbers_l233_233593

-- Definition of the numbers we are interested in
def a : ℕ := 9118
def b : ℕ := 12173
def c : ℕ := 33182

-- Statement of the problem to prove GCD
theorem gcd_of_three_numbers : Int.gcd (Int.gcd a b) c = 47 := 
sorry  -- Proof skipped

end gcd_of_three_numbers_l233_233593


namespace polygon_perimeter_exposure_l233_233910

theorem polygon_perimeter_exposure:
  let triangle_sides := 3
  let square_sides := 4
  let pentagon_sides := 5
  let hexagon_sides := 6
  let heptagon_sides := 7
  let octagon_sides := 8
  let nonagon_sides := 9
  let exposure_triangle_nonagon := triangle_sides + nonagon_sides - 2
  let other_polygons_adjacency := 2 * 5
  let exposure_other_polygons := square_sides + pentagon_sides + hexagon_sides + heptagon_sides + octagon_sides - other_polygons_adjacency
  exposure_triangle_nonagon + exposure_other_polygons = 30 :=
by sorry

end polygon_perimeter_exposure_l233_233910


namespace pictures_per_album_l233_233519

theorem pictures_per_album (phone_pics camera_pics albums pics_per_album : ℕ)
  (h1 : phone_pics = 7) (h2 : camera_pics = 13) (h3 : albums = 5)
  (h4 : pics_per_album * albums = phone_pics + camera_pics) :
  pics_per_album = 4 :=
by
  sorry

end pictures_per_album_l233_233519


namespace percent_increase_second_half_century_l233_233470

variable (P : ℝ) -- Initial population
variable (x : ℝ) -- Percentage increase in the second half of the century

noncomputable def population_first_half_century := 3 * P
noncomputable def population_end_century := P + 11 * P

theorem percent_increase_second_half_century :
  3 * P + (x / 100) * (3 * P) = 12 * P → x = 300 :=
by
  intro h
  sorry

end percent_increase_second_half_century_l233_233470


namespace triangle_in_base_7_l233_233905

theorem triangle_in_base_7 (triangle : ℕ) 
  (h1 : (triangle + 6) % 7 = 0) : 
  triangle = 1 := 
sorry

end triangle_in_base_7_l233_233905


namespace number_of_real_values_of_p_l233_233626

theorem number_of_real_values_of_p :
  ∃ p_values : Finset ℝ, (∀ p ∈ p_values, ∀ x, x^2 - 2 * p * x + 3 * p = 0 → (x = p)) ∧ Finset.card p_values = 2 :=
by
  sorry

end number_of_real_values_of_p_l233_233626


namespace white_cats_count_l233_233074

theorem white_cats_count (total_cats : ℕ) (black_cats : ℕ) (gray_cats : ℕ) (white_cats : ℕ)
  (h1 : total_cats = 15)
  (h2 : black_cats = 10)
  (h3 : gray_cats = 3)
  (h4 : total_cats = black_cats + gray_cats + white_cats) : 
  white_cats = 2 := 
  by
    -- proof or sorry here
    sorry

end white_cats_count_l233_233074


namespace positive_difference_between_two_numbers_l233_233270

variable (x y : ℝ)

theorem positive_difference_between_two_numbers 
  (h₁ : x + y = 40)
  (h₂ : 3 * y - 4 * x = 20) :
  |y - x| = 100 / 7 :=
by
  sorry

end positive_difference_between_two_numbers_l233_233270


namespace balcony_more_than_orchestra_l233_233917

variables (O B : ℕ) (H1 : O + B = 380) (H2 : 12 * O + 8 * B = 3320)

theorem balcony_more_than_orchestra : B - O = 240 :=
by sorry

end balcony_more_than_orchestra_l233_233917


namespace exsphere_identity_l233_233865

-- Given definitions for heights and radii
variables {h1 h2 h3 h4 r1 r2 r3 r4 : ℝ}

-- Definition of the relationship that needs to be proven
theorem exsphere_identity 
  (h1 h2 h3 h4 r1 r2 r3 r4 : ℝ) :
  2 * (1 / h1 + 1 / h2 + 1 / h3 + 1 / h4) = 1 / r1 + 1 / r2 + 1 / r3 + 1 / r4 := 
sorry

end exsphere_identity_l233_233865


namespace computation_result_l233_233716

theorem computation_result :
  (3 + 6 - 12 + 24 + 48 - 96 + 192 - 384) / (6 + 12 - 24 + 48 + 96 - 192 + 384 - 768) = 1 / 2 :=
by
  sorry

end computation_result_l233_233716


namespace obtuse_angle_between_line_and_plane_l233_233870

-- Define the problem conditions
def is_obtuse_angle (θ : ℝ) : Prop := θ > 90 ∧ θ < 180

-- Define what we are proving
theorem obtuse_angle_between_line_and_plane (θ : ℝ) (h1 : θ = angle_between_line_and_plane) :
  is_obtuse_angle θ :=
sorry

end obtuse_angle_between_line_and_plane_l233_233870


namespace quadratic_inequality_sufficient_necessary_l233_233050

theorem quadratic_inequality_sufficient_necessary (a : ℝ) : 
  (∀ x : ℝ, x^2 - a * x + a > 0) ↔ 0 < a ∧ a < 4 :=
by
  -- proof skipped
  sorry

end quadratic_inequality_sufficient_necessary_l233_233050


namespace part1_zero_of_f_a_neg1_part2_range_of_a_l233_233416

noncomputable def f (a x : ℝ) := a * x^2 + 2 * x - 2 - a

theorem part1_zero_of_f_a_neg1 : 
  f (-1) 1 = 0 :=
by 
  sorry

theorem part2_range_of_a (a : ℝ) :
  a ≤ 0 →
  (∃ x : ℝ, 0 < x ∧ x ≤ 1 ∧ f a x = 0) ∧ (∀ x : ℝ, 0 < x ∧ x ≤ 1 → f a x = 0 → x = 1) ↔ 
  (-1 ≤ a ∧ a ≤ 0) ∨ (a ≤ -2) :=
by 
  sorry

end part1_zero_of_f_a_neg1_part2_range_of_a_l233_233416


namespace remainder_of_7_pow_145_mod_9_l233_233577

theorem remainder_of_7_pow_145_mod_9 : (7 ^ 145) % 9 = 7 := by
  sorry

end remainder_of_7_pow_145_mod_9_l233_233577


namespace next_terms_arithmetic_seq_next_terms_alternating_seq_next_terms_interwoven_seq_next_terms_geometric_seq_l233_233757

-- Part (a)
theorem next_terms_arithmetic_seq : ∀ (a₀ a₁ a₂ a₃ a₄ a₅ d: ℕ), 
  a₀ = 3 → a₁ = 7 → a₂ = 11 → a₃ = 15 → a₄ = 19 → a₅ = 23 → d = 4 →
  (a₅ + d = 27) ∧ (a₅ + 2*d = 31) :=
by intros; sorry


-- Part (b)
theorem next_terms_alternating_seq : ∀ (a₀ a₁ a₂ a₃ a₄ a₅ : ℕ),
  a₀ = 9 → a₁ = 1 → a₂ = 7 → a₃ = 1 → a₄ = 5 → a₅ = 1 →
  a₄ - 2 = 3 ∧ a₁ = 1 :=
by intros; sorry


-- Part (c)
theorem next_terms_interwoven_seq : ∀ (a₀ a₁ a₂ a₃ a₄ a₅ d: ℕ),
  a₀ = 4 → a₁ = 5 → a₂ = 8 → a₃ = 9 → a₄ = 12 → a₅ = 13 → d = 4 →
  (a₄ + d = 16) ∧ (a₅ + d = 17) :=
by intros; sorry


-- Part (d)
theorem next_terms_geometric_seq : ∀ (a₀ a₁ a₂ a₃ a₄ a₅: ℕ), 
  a₀ = 1 → a₁ = 2 → a₂ = 4 → a₃ = 8 → a₄ = 16 → a₅ = 32 →
  (a₅ * 2 = 64) ∧ (a₅ * 4 = 128) :=
by intros; sorry

end next_terms_arithmetic_seq_next_terms_alternating_seq_next_terms_interwoven_seq_next_terms_geometric_seq_l233_233757


namespace exponent_evaluation_problem_l233_233248

theorem exponent_evaluation_problem (m : ℕ) : 
  (m^2 * m^3 ≠ m^6) → 
  (m^2 + m^4 ≠ m^6) → 
  ((m^3)^3 ≠ m^6) → 
  (m^7 / m = m^6) :=
by
  intros hA hB hC
  -- Provide the proof here
  sorry

end exponent_evaluation_problem_l233_233248


namespace abcde_sum_to_628_l233_233550

theorem abcde_sum_to_628 (a b c d e : ℕ) (h_distinct : (a = 1 ∨ a = 2 ∨ a = 3 ∨ a = 4 ∨ a = 5) ∧ 
                                                 (b = 1 ∨ b = 2 ∨ b = 3 ∨ b = 4 ∨ b = 5) ∧ 
                                                 (c = 1 ∨ c = 2 ∨ c = 3 ∨ c = 4 ∨ c = 5) ∧ 
                                                 (d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 5) ∧ 
                                                 (e = 1 ∨ e = 2 ∨ e = 3 ∨ e = 4 ∨ e = 5) ∧
                                                 a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
                                                 b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
                                                 c ≠ d ∧ c ≠ e ∧
                                                 d ≠ e)
  (h1 : b ≤ d)
  (h2 : c ≥ a)
  (h3 : a ≤ e)
  (h4 : b ≥ e)
  (h5 : d ≠ 5) :
  a^b + c^d + e = 628 := sorry

end abcde_sum_to_628_l233_233550


namespace emily_weight_l233_233311

theorem emily_weight (H_weight : ℝ) (difference : ℝ) (h : H_weight = 87) (d : difference = 78) : 
  ∃ E_weight : ℝ, E_weight = 9 := 
by
  sorry

end emily_weight_l233_233311


namespace residue_7_1234_mod_13_l233_233169

theorem residue_7_1234_mod_13 :
  (7 : ℤ) ^ 1234 % 13 = 12 :=
by
  -- given conditions as definitions
  have h1 : (7 : ℤ) % 13 = 7 := by norm_num
  
  -- auxiliary calculations
  have h2 : (49 : ℤ) % 13 = 10 := by norm_num
  have h3 : (100 : ℤ) % 13 = 9 := by norm_num
  have h4 : (81 : ℤ) % 13 = 3 := by norm_num
  have h5 : (729 : ℤ) % 13 = 1 := by norm_num

  -- the actual problem we want to prove
  sorry

end residue_7_1234_mod_13_l233_233169


namespace roots_of_quadratic_l233_233529

noncomputable def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem roots_of_quadratic (m : ℝ) :
  let a := 1
  let b := (3 * m - 1)
  let c := (2 * m^2 - m)
  discriminant a b c ≥ 0 :=
by
  sorry

end roots_of_quadratic_l233_233529


namespace not_p_equiv_exists_leq_sin_l233_233950

-- Define the conditions as a Lean proposition
def p : Prop := ∀ x : ℝ, x > Real.sin x

-- State the problem as a theorem to be proved
theorem not_p_equiv_exists_leq_sin : ¬p = ∃ x : ℝ, x ≤ Real.sin x := 
by sorry

end not_p_equiv_exists_leq_sin_l233_233950


namespace evaluate_expression_l233_233834

noncomputable def expression (a : ℚ) : ℚ := 
  (a / (a - 1)) / ((a + 1) / (a^2 - 1)) - (1 - 2 * a)

theorem evaluate_expression (a : ℚ) (ha : a = -1/3) : expression a = -2 :=
by 
  rw [expression, ha]
  sorry

end evaluate_expression_l233_233834


namespace trapezoid_diagonals_l233_233148

theorem trapezoid_diagonals (AD BC : ℝ) (angle_DAB angle_BCD : ℝ)
  (hAD : AD = 8) (hBC : BC = 6) (h_angle_DAB : angle_DAB = 90)
  (h_angle_BCD : angle_BCD = 120) :
  ∃ AC BD : ℝ, AC = 4 * Real.sqrt 3 ∧ BD = 2 * Real.sqrt 19 :=
by
  sorry

end trapezoid_diagonals_l233_233148


namespace songs_downloaded_later_l233_233442

-- Definition that each song has a size of 5 MB
def song_size : ℕ := 5

-- Definition that the new songs will occupy 140 MB of memory space
def total_new_song_memory : ℕ := 140

-- Prove that the number of songs Kira downloaded later on that day is 28
theorem songs_downloaded_later (x : ℕ) (h : song_size * x = total_new_song_memory) : x = 28 :=
by
  sorry

end songs_downloaded_later_l233_233442


namespace water_percentage_in_tomato_juice_l233_233964

-- Definitions from conditions
def tomato_juice_volume := 80 -- in liters
def tomato_puree_volume := 10 -- in liters
def tomato_puree_water_percentage := 20 -- in percent (20%)

-- Need to prove percentage of water in tomato juice is 20%
theorem water_percentage_in_tomato_juice : 
  (100 - tomato_puree_water_percentage) * tomato_puree_volume / tomato_juice_volume = 20 :=
by
  -- Skip the proof
  sorry

end water_percentage_in_tomato_juice_l233_233964


namespace eq1_eq2_eq3_eq4_l233_233066

theorem eq1 : ∀ x : ℝ, x = 6 → 3 * x - 8 = x + 4 := by
  intros x hx
  rw [hx]
  sorry

theorem eq2 : ∀ x : ℝ, x = -2 → 1 - 3 * (x + 1) = 2 * (1 - 0.5 * x) := by
  intros x hx
  rw [hx]
  sorry

theorem eq3 : ∀ x : ℝ, x = -20 → (1 / 6) * (3 * x - 6) = (2 / 5) * x - 3 := by
  intros x hx
  rw [hx]
  sorry

theorem eq4 : ∀ y : ℝ, y = -1 → (3 * y - 1) / 4 - 1 = (5 * y - 7) / 6 := by
  intros y hy
  rw [hy]
  sorry

end eq1_eq2_eq3_eq4_l233_233066


namespace y_intercept_of_line_l233_233382

theorem y_intercept_of_line :
  ∃ y, (∀ x : ℝ, 2 * x - 3 * y = 6) ∧ (y = -2) :=
sorry

end y_intercept_of_line_l233_233382


namespace inequality_proof_l233_233340

theorem inequality_proof (a b c d : ℝ) : 
  0 ≤ a → 0 ≤ b → 0 ≤ c → 0 ≤ d → a + b = 2 → c + d = 2 → 
  (a^2 + c^2) * (a^2 + d^2) * (b^2 + c^2) * (b^2 + d^2) ≤ 25 := 
by 
  intros ha hb hc hd hab hcd
  sorry

end inequality_proof_l233_233340


namespace slices_left_for_tomorrow_is_four_l233_233845

def initial_slices : ℕ := 12
def lunch_slices : ℕ := initial_slices / 2
def remaining_slices_after_lunch : ℕ := initial_slices - lunch_slices
def dinner_slices : ℕ := remaining_slices_after_lunch / 3
def slices_left_for_tomorrow : ℕ := remaining_slices_after_lunch - dinner_slices

theorem slices_left_for_tomorrow_is_four : slices_left_for_tomorrow = 4 := by
  sorry

end slices_left_for_tomorrow_is_four_l233_233845


namespace remainder_of_division_l233_233816

theorem remainder_of_division (x y R : ℕ) 
  (h1 : y = 1782)
  (h2 : y - x = 1500)
  (h3 : y = 6 * x + R) :
  R = 90 :=
by
  sorry

end remainder_of_division_l233_233816


namespace total_games_played_l233_233098

theorem total_games_played (points_per_game_winner : ℕ) (points_per_game_loser : ℕ) (jack_games_won : ℕ)
  (jill_total_points : ℕ) (total_games : ℕ)
  (h1 : points_per_game_winner = 2)
  (h2 : points_per_game_loser = 1)
  (h3 : jack_games_won = 4)
  (h4 : jill_total_points = 10)
  (h5 : ∀ games_won_by_jill : ℕ, jill_total_points = games_won_by_jill * points_per_game_winner +
           (jack_games_won * points_per_game_loser)) :
  total_games = jack_games_won + (jill_total_points - jack_games_won * points_per_game_loser) / points_per_game_winner := by
  sorry

end total_games_played_l233_233098


namespace triangle_side_a_l233_233640

theorem triangle_side_a (a : ℝ) : 2 < a ∧ a < 8 → a = 7 :=
by
  sorry

end triangle_side_a_l233_233640


namespace kylie_beads_total_l233_233947

def number_necklaces_monday : Nat := 10
def number_necklaces_tuesday : Nat := 2
def number_bracelets_wednesday : Nat := 5
def number_earrings_wednesday : Nat := 7

def beads_per_necklace : Nat := 20
def beads_per_bracelet : Nat := 10
def beads_per_earring : Nat := 5

theorem kylie_beads_total :
  (number_necklaces_monday + number_necklaces_tuesday) * beads_per_necklace + 
  number_bracelets_wednesday * beads_per_bracelet + 
  number_earrings_wednesday * beads_per_earring = 325 := 
by
  sorry

end kylie_beads_total_l233_233947


namespace real_roots_determinant_l233_233032

variable (a b c k : ℝ)
variable (k_pos : k > 0)
variable (a_nonzero : a ≠ 0) 
variable (b_nonzero : b ≠ 0)
variable (c_nonzero : c ≠ 0)
variable (k_nonzero : k ≠ 0)

theorem real_roots_determinant : 
  ∃! x : ℝ, (Matrix.det ![![x, k * c, -k * b], ![-k * c, x, k * a], ![k * b, -k * a, x]] = 0) :=
sorry

end real_roots_determinant_l233_233032


namespace determinant_of_trig_matrix_l233_233072

theorem determinant_of_trig_matrix (α β : ℝ) : 
  Matrix.det ![
    ![Real.sin α, Real.cos α], 
    ![Real.cos β, Real.sin β]
  ] = -Real.cos (α - β) :=
by sorry

end determinant_of_trig_matrix_l233_233072


namespace max_min_product_of_three_l233_233397

open List

theorem max_min_product_of_three (s : List Int) (h : s = [-1, -2, 3, 4]) : 
  ∃ (max min : Int), 
    max = 8 ∧ min = -24 ∧ 
    (∀ a b c, a ∈ s → b ∈ s → c ∈ s → a ≠ b → b ≠ c → a ≠ c → a * b * c ≤ max) ∧
    (∀ a b c, a ∈ s → b ∈ s → c ∈ s → a ≠ b → b ≠ c → a ≠ c → a * b * c ≥ min) := 
by
  sorry

end max_min_product_of_three_l233_233397


namespace Micheal_work_rate_l233_233100

theorem Micheal_work_rate 
    (M A : ℕ) 
    (h1 : 1 / M + 1 / A = 1 / 20)
    (h2 : 9 / 200 = 1 / A) : M = 200 :=
by
    sorry

end Micheal_work_rate_l233_233100


namespace four_digit_numbers_divisible_by_11_and_5_with_sum_12_l233_233557

theorem four_digit_numbers_divisible_by_11_and_5_with_sum_12:
  ∀ a b c d : ℕ, (a + b + c + d = 12) ∧ ((a + c) - (b + d)) % 11 = 0 ∧ (d = 0 ∨ d = 5) →
  false :=
by
  intro a b c d
  intro h
  sorry

end four_digit_numbers_divisible_by_11_and_5_with_sum_12_l233_233557


namespace calculate_expression_l233_233535

theorem calculate_expression : 5 * 7 + 10 * 4 - 36 / 3 + 6 * 3 = 81 :=
by
  -- Proof steps would be included here if they were needed, but the proof is left as sorry for now.
  sorry

end calculate_expression_l233_233535


namespace average_score_of_male_students_standard_deviation_of_all_students_l233_233799

def students : ℕ := 5
def total_average_score : ℝ := 80
def male_student_variance : ℝ := 150
def female_student1_score : ℝ := 85
def female_student2_score : ℝ := 75
def male_student_average_score : ℝ := 80 -- From solution step (1)
def total_standard_deviation : ℝ := 10 -- From solution step (2)

theorem average_score_of_male_students :
  (students = 5) →
  (total_average_score = 80) →
  (male_student_variance = 150) →
  (female_student1_score = 85) →
  (female_student2_score = 75) →
  male_student_average_score = 80 :=
by sorry

theorem standard_deviation_of_all_students :
  (students = 5) →
  (total_average_score = 80) →
  (male_student_variance = 150) →
  (female_student1_score = 85) →
  (female_student2_score = 75) →
  total_standard_deviation = 10 :=
by sorry

end average_score_of_male_students_standard_deviation_of_all_students_l233_233799


namespace samara_tire_spending_l233_233869

theorem samara_tire_spending :
  ∀ (T : ℕ), 
    (2457 = 25 + 79 + T + 1886) → 
    T = 467 :=
by intros T h
   sorry

end samara_tire_spending_l233_233869


namespace calculate_expression_l233_233446

theorem calculate_expression : (50 - (5020 - 520) + (5020 - (520 - 50))) = 100 := 
by
  sorry

end calculate_expression_l233_233446


namespace ticket_cost_difference_l233_233507

theorem ticket_cost_difference
  (num_adults : ℕ) (num_children : ℕ)
  (cost_adult_ticket : ℕ) (cost_child_ticket : ℕ)
  (h1 : num_adults = 9)
  (h2 : num_children = 7)
  (h3 : cost_adult_ticket = 11)
  (h4 : cost_child_ticket = 7) :
  num_adults * cost_adult_ticket - num_children * cost_child_ticket = 50 := 
by
  sorry

end ticket_cost_difference_l233_233507


namespace exposed_surface_area_l233_233832

theorem exposed_surface_area (r h : ℝ) (π : ℝ) (sphere_surface_area : ℝ) (cylinder_lateral_surface_area : ℝ) 
  (cond1 : r = 10) (cond2 : h = 5) (cond3 : sphere_surface_area = 4 * π * r^2) 
  (cond4 : cylinder_lateral_surface_area = 2 * π * r * h) :
  let hemisphere_curved_surface_area := sphere_surface_area / 2
  let hemisphere_base_area := π * r^2
  let total_surface_area := hemisphere_curved_surface_area + hemisphere_base_area + cylinder_lateral_surface_area
  total_surface_area = 400 * π :=
by
  sorry

end exposed_surface_area_l233_233832


namespace smallest_x_l233_233612

theorem smallest_x :
  ∃ (x : ℕ), x % 4 = 3 ∧ x % 5 = 4 ∧ x % 6 = 5 ∧ ∀ y : ℕ, (y % 4 = 3 ∧ y % 5 = 4 ∧ y % 6 = 5) → y ≥ x := 
sorry

end smallest_x_l233_233612


namespace sally_took_home_pens_l233_233658

theorem sally_took_home_pens
    (initial_pens : ℕ)
    (students : ℕ)
    (pens_per_student : ℕ)
    (locker_fraction : ℕ)
    (total_pens_given : ℕ)
    (remainder : ℕ)
    (locker_pens : ℕ)
    (home_pens : ℕ) :
    initial_pens = 5230 →
    students = 89 →
    pens_per_student = 58 →
    locker_fraction = 2 →
    total_pens_given = students * pens_per_student →
    remainder = initial_pens - total_pens_given →
    locker_pens = remainder / locker_fraction →
    home_pens = locker_pens →
    home_pens = 34 :=
by {
  sorry
}

end sally_took_home_pens_l233_233658


namespace Louisa_travel_distance_l233_233591

variables (D : ℕ)

theorem Louisa_travel_distance : 
  (200 / 50 + 3 = D / 50) → D = 350 :=
by
  intros h
  sorry

end Louisa_travel_distance_l233_233591


namespace keaton_earns_yearly_l233_233601

/-- Keaton's total yearly earnings from oranges and apples given the harvest cycles and prices. -/
theorem keaton_earns_yearly : 
  let orange_harvest_cycle := 2
  let orange_harvest_price := 50
  let apple_harvest_cycle := 3
  let apple_harvest_price := 30
  let months_in_a_year := 12
  
  let orange_harvests_per_year := months_in_a_year / orange_harvest_cycle
  let apple_harvests_per_year := months_in_a_year / apple_harvest_cycle
  
  let orange_yearly_earnings := orange_harvests_per_year * orange_harvest_price
  let apple_yearly_earnings := apple_harvests_per_year * apple_harvest_price
    
  orange_yearly_earnings + apple_yearly_earnings = 420 :=
by
  sorry

end keaton_earns_yearly_l233_233601


namespace second_number_division_l233_233966

theorem second_number_division (d x r : ℕ) (h1 : d = 16) (h2 : 25 % d = r) (h3 : 105 % d = r) (h4 : r = 9) : x % d = r → x = 41 :=
by 
  simp [h1, h2, h3, h4] 
  sorry

end second_number_division_l233_233966


namespace integer_solutions_of_system_l233_233188

theorem integer_solutions_of_system :
  {x : ℤ | - 2 * x + 7 < 10 ∧ (7 * x + 1) / 5 - 1 ≤ x} = {-1, 0, 1, 2} :=
by
  sorry

end integer_solutions_of_system_l233_233188


namespace necessary_condition_ac_eq_bc_l233_233968

theorem necessary_condition_ac_eq_bc {a b c : ℝ} (hc : c ≠ 0) : (ac = bc ↔ a = b) := by
  sorry

end necessary_condition_ac_eq_bc_l233_233968


namespace cos_half_pi_plus_alpha_correct_l233_233839

noncomputable def cos_half_pi_plus_alpha
  (α : ℝ)
  (h1 : Real.tan α = -3/4)
  (h2 : α ∈ Set.Ioo (3 * Real.pi / 2) (2 * Real.pi)) : Real :=
  Real.cos (Real.pi / 2 + α)

theorem cos_half_pi_plus_alpha_correct
  (α : ℝ)
  (h1 : Real.tan α = -3/4)
  (h2 : α ∈ Set.Ioo (3 * Real.pi / 2) (2 * Real.pi)) :
  cos_half_pi_plus_alpha α h1 h2 = 3/5 := by
  sorry

end cos_half_pi_plus_alpha_correct_l233_233839


namespace inequality_abc_l233_233271

theorem inequality_abc (a b c : ℝ) (n : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b * c = 1) (h5 : 2 ≤ n) :
  (a / (b + c)^(1/(n:ℝ)) + b / (c + a)^(1/(n:ℝ)) + c / (a + b)^(1/(n:ℝ)) ≥ 3 / 2^(1/(n:ℝ))) :=
by sorry

end inequality_abc_l233_233271


namespace range_of_a_l233_233366

theorem range_of_a (a : ℝ) : (∃ x : ℝ, a * x^2 + x + 1 < 0) ↔ (a < 1/4) := 
sorry

end range_of_a_l233_233366


namespace eq_sum_of_factorial_fractions_l233_233018

theorem eq_sum_of_factorial_fractions (b2 b3 b5 b6 b7 b8 : ℤ)
  (h2 : 0 ≤ b2 ∧ b2 < 2)
  (h3 : 0 ≤ b3 ∧ b3 < 3)
  (h5 : 0 ≤ b5 ∧ b5 < 5)
  (h6 : 0 ≤ b6 ∧ b6 < 6)
  (h7 : 0 ≤ b7 ∧ b7 < 7)
  (h8 : 0 ≤ b8 ∧ b8 < 8)
  (h_eq : (3 / 8 : ℚ) = (b2 / (2 * 1) + b3 / (3 * 2 * 1) + b5 / (5 * 4 * 3 * 2 * 1) +
                          b6 / (6 * 5 * 4 * 3 * 2 * 1) + b7 / (7 * 6 * 5 * 4 * 3 * 2 * 1) +
                          b8 / (8 * 7 * 6 * 5 * 4 * 3 * 2 * 1) : ℚ)) :
  b2 + b3 + b5 + b6 + b7 + b8 = 12 :=
by
  sorry

end eq_sum_of_factorial_fractions_l233_233018


namespace hiker_distance_l233_233536

-- Prove that the length of the path d is 90 miles
theorem hiker_distance (x t d : ℝ) (h1 : d = x * t)
                             (h2 : d = (x + 1) * (3 / 4) * t)
                             (h3 : d = (x - 1) * (t + 3)) :
  d = 90 := 
sorry

end hiker_distance_l233_233536


namespace walter_coins_value_l233_233214

theorem walter_coins_value :
  let pennies : ℕ := 2
  let nickels : ℕ := 2
  let dimes : ℕ := 1
  let quarters : ℕ := 1
  let half_dollars : ℕ := 1
  let penny_value : ℕ := 1
  let nickel_value : ℕ := 5
  let dime_value : ℕ := 10
  let quarter_value : ℕ := 25
  let half_dollar_value : ℕ := 50
  (pennies * penny_value + nickels * nickel_value + dimes * dime_value + quarters * quarter_value + half_dollars * half_dollar_value) = 97 := 
sorry

end walter_coins_value_l233_233214


namespace total_marks_by_category_l233_233687

theorem total_marks_by_category 
  (num_candidates_A : ℕ) (num_candidates_B : ℕ) (num_candidates_C : ℕ)
  (avg_marks_A : ℕ) (avg_marks_B : ℕ) (avg_marks_C : ℕ) 
  (hA : num_candidates_A = 30) (hB : num_candidates_B = 25) (hC : num_candidates_C = 25)
  (h_avg_A : avg_marks_A = 35) (h_avg_B : avg_marks_B = 42) (h_avg_C : avg_marks_C = 46) :
  (num_candidates_A * avg_marks_A = 1050) ∧
  (num_candidates_B * avg_marks_B = 1050) ∧
  (num_candidates_C * avg_marks_C = 1150) := 
by
  sorry

end total_marks_by_category_l233_233687


namespace geometric_sequence_sum_range_l233_233344

theorem geometric_sequence_sum_range (a b c : ℝ) 
  (h1 : ∃ q : ℝ, q ≠ 0 ∧ a = b * q ∧ c = b / q) 
  (h2 : a + b + c = 1) : 
  a + c ∈ (Set.Icc (2 / 3 : ℝ) 1 \ Set.Iio 1) ∪ (Set.Ioo 1 2) :=
sorry

end geometric_sequence_sum_range_l233_233344


namespace greatest_three_digit_multiple_of_17_l233_233395

theorem greatest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n ≤ 999 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m → m ≤ n) :=
sorry

end greatest_three_digit_multiple_of_17_l233_233395


namespace continued_fraction_l233_233906

theorem continued_fraction {w x y : ℕ} (hw : 0 < w) (hx : 0 < x) (hy : 0 < y)
  (h_eq : (97:ℚ) / 19 = w + 1 / (x + 1 / y)) : w + x + y = 16 :=
sorry

end continued_fraction_l233_233906


namespace simplify_expression_to_fraction_l233_233502

theorem simplify_expression_to_fraction : 
  (1 / (1 / (1/2)^2 + 1 / (1/2)^3 + 1 / (1/2)^4 + 1 / (1/2)^5)) = 1/60 :=
by 
  have h1 : 1 / (1/2)^2 = 4 := by sorry
  have h2 : 1 / (1/2)^3 = 8 := by sorry
  have h3 : 1 / (1/2)^4 = 16 := by sorry
  have h4 : 1 / (1/2)^5 = 32 := by sorry
  have h5 : 4 + 8 + 16 + 32 = 60 := by sorry
  have h6 : 1 / 60 = 1/60 := by sorry
  sorry

end simplify_expression_to_fraction_l233_233502


namespace julian_owes_jenny_l233_233954

-- Define the initial debt and the additional borrowed amount
def initial_debt : ℕ := 20
def additional_borrowed : ℕ := 8

-- Define the total debt
def total_debt : ℕ := initial_debt + additional_borrowed

-- Statement of the problem: Prove that total_debt equals 28
theorem julian_owes_jenny : total_debt = 28 :=
by
  sorry

end julian_owes_jenny_l233_233954


namespace jack_helped_hours_l233_233441

-- Definitions based on the problem's conditions
def sam_rate : ℕ := 6  -- Sam assembles 6 widgets per hour
def tony_rate : ℕ := 2  -- Tony assembles 2 widgets per hour
def jack_rate : ℕ := sam_rate  -- Jack assembles at the same rate as Sam
def total_widgets : ℕ := 68  -- The total number of widgets assembled by all three

-- Statement to prove
theorem jack_helped_hours : 
  ∃ h : ℕ, (sam_rate * h) + (tony_rate * h) + (jack_rate * h) = total_widgets ∧ h = 4 := 
  by
  -- The proof is not necessary; we only need the statement
  sorry

end jack_helped_hours_l233_233441


namespace scalene_triangle_area_l233_233364

theorem scalene_triangle_area (outer_triangle_area : ℝ) (hexagon_area : ℝ) (num_scalene_triangles : ℕ)
  (h1 : outer_triangle_area = 25) (h2 : hexagon_area = 4) (h3 : num_scalene_triangles = 6) : 
  (outer_triangle_area - hexagon_area) / num_scalene_triangles = 3.5 :=
by
  sorry

end scalene_triangle_area_l233_233364


namespace find_c_l233_233895

theorem find_c (x c : ℚ) (h1 : 3 * x + 5 = 1) (h2 : c * x + 15 = 3) : c = 9 :=
by sorry

end find_c_l233_233895


namespace part1_cos_A_part2_c_l233_233575

-- We define a triangle with sides a, b, c opposite to angles A, B, C respectively.
variables (a b c : ℝ) (A B C : ℝ)
-- Given conditions for the problem:
variable (h1 : 3 * a * Real.cos A = c * Real.cos B + b * Real.cos C)
variable (h_cos_sum : Real.cos B + Real.cos C = (2 * Real.sqrt 3) / 3)
variable (ha : a = 2 * Real.sqrt 3)

-- The first part of the problem statement proving cos A = 1/3 given the conditions.
theorem part1_cos_A : Real.cos A = 1 / 3 :=
by
  sorry

-- The second part of the problem statement proving c = 3 given the conditions.
theorem part2_c : c = 3 :=
by
  sorry

end part1_cos_A_part2_c_l233_233575


namespace percentage_flowering_plants_l233_233951

variable (P : ℝ)

theorem percentage_flowering_plants (h : 5 * (1 / 4) * (P / 100) * 80 = 40) : P = 40 :=
by
  -- This is where the proof would go, but we will use sorry to skip it for now
  sorry

end percentage_flowering_plants_l233_233951


namespace minimize_circumscribed_sphere_radius_l233_233376

noncomputable def cylinder_surface_area (r h : ℝ) : ℝ :=
  2 * Real.pi * r^2 + 2 * Real.pi * r * h

noncomputable def circumscribed_sphere_radius (r h : ℝ) : ℝ :=
  (r^2 + (1 / 2 * h)^2).sqrt

theorem minimize_circumscribed_sphere_radius (r : ℝ) (h : ℝ) (hr : cylinder_surface_area r h = 16 * Real.pi) : 
  r^2 = 8 * Real.sqrt 5 / 5 :=
sorry

end minimize_circumscribed_sphere_radius_l233_233376


namespace non_deg_ellipse_condition_l233_233967

theorem non_deg_ellipse_condition (k : ℝ) : k > -19 ↔ 
  (∃ x y : ℝ, 3 * x^2 + 7 * y^2 - 12 * x + 14 * y = k) :=
sorry

end non_deg_ellipse_condition_l233_233967


namespace center_number_is_4_l233_233815

-- Define the numbers and the 3x3 grid
inductive Square
| center | top_middle | left_middle | right_middle | bottom_middle

-- Define the properties of the problem
def isConsecutiveAdjacent (a b : ℕ) : Prop := 
  (a + 1 = b ∨ a = b + 1)

-- The condition to check the sum of edge squares
def sum_edge_squares (grid : Square → ℕ) : Prop := 
  grid Square.top_middle + grid Square.left_middle + grid Square.right_middle + grid Square.bottom_middle = 28

-- The condition that the center square number is even
def even_center (grid : Square → ℕ) : Prop := 
  grid Square.center % 2 = 0

-- The main theorem statement
theorem center_number_is_4 (grid : Square → ℕ) :
  (∀ i j : Square, i ≠ j → isConsecutiveAdjacent (grid i) (grid j)) → 
  (grid Square.top_middle + grid Square.left_middle + grid Square.right_middle + grid Square.bottom_middle = 28) →
  (grid Square.center % 2 = 0) →
  grid Square.center = 4 :=
by sorry

end center_number_is_4_l233_233815


namespace car_travel_inequality_l233_233365

variable (x : ℕ)

theorem car_travel_inequality (hx : 8 * (x + 19) > 2200) : 8 * (x + 19) > 2200 :=
by
  sorry

end car_travel_inequality_l233_233365


namespace compute_usage_difference_l233_233052

theorem compute_usage_difference
  (usage_last_week : ℕ)
  (usage_per_day : ℕ)
  (days_in_week : ℕ)
  (h1 : usage_last_week = 91)
  (h2 : usage_per_day = 8)
  (h3 : days_in_week = 7) :
  (usage_last_week - usage_per_day * days_in_week) = 35 :=
  sorry

end compute_usage_difference_l233_233052


namespace total_wage_is_75_l233_233020

noncomputable def wages_total (man_wage : ℕ) : ℕ :=
  let men := 5
  let women := (5 : ℕ)
  let boys := 8
  (man_wage * men) + (man_wage * men) + (man_wage * men)

theorem total_wage_is_75
  (W : ℕ)
  (man_wage : ℕ := 5)
  (h1 : 5 = W) 
  (h2 : W = 8) 
  : wages_total man_wage = 75 := by
  sorry

end total_wage_is_75_l233_233020


namespace pyramid_side_length_l233_233758

-- Definitions for our conditions
def area_of_lateral_face : ℝ := 150
def slant_height : ℝ := 25

-- Theorem statement
theorem pyramid_side_length (A : ℝ) (h : ℝ) (s : ℝ) (hA : A = area_of_lateral_face) (hh : h = slant_height) :
  A = (1 / 2) * s * h → s = 12 :=
by
  intro h_eq
  rw [hA, hh, area_of_lateral_face, slant_height] at h_eq
  -- Steps to verify s = 12
  sorry

end pyramid_side_length_l233_233758


namespace average_marbles_of_other_colors_l233_233897

theorem average_marbles_of_other_colors
  (clear_percentage : ℝ) (black_percentage : ℝ) (total_marbles_taken : ℕ)
  (h1 : clear_percentage = 0.4) (h2 : black_percentage = 0.2) :
  (total_marbles_taken : ℝ) * (1 - clear_percentage - black_percentage) = 2 :=
by
  sorry

end average_marbles_of_other_colors_l233_233897


namespace volleyball_team_girls_l233_233404

theorem volleyball_team_girls (B G : ℕ) (h1 : B + G = 30) (h2 : 1 / 3 * G + B = 20) : G = 15 :=
sorry

end volleyball_team_girls_l233_233404


namespace num_of_3_digit_nums_with_one_even_digit_l233_233918

def is_even (n : Nat) : Bool :=
  n % 2 == 0

def count_3_digit_nums_with_exactly_one_even_digit : Nat :=
  let even_digits := [0, 2, 4, 6, 8]
  let odd_digits := [1, 3, 5, 7, 9]
  -- Case 1: A is even, B and C are odd
  let case1 := 4 * 5 * 5
  -- Case 2: B is even, A and C are odd
  let case2 := 5 * 5 * 5
  -- Case 3: C is even, A and B are odd
  let case3 := 5 * 5 * 5
  case1 + case2 + case3

theorem num_of_3_digit_nums_with_one_even_digit : count_3_digit_nums_with_exactly_one_even_digit = 350 := by
  sorry

end num_of_3_digit_nums_with_one_even_digit_l233_233918


namespace spot_area_l233_233001

/-- Proving the area of the accessible region outside the doghouse -/
theorem spot_area
  (pentagon_side : ℝ)
  (rope_length : ℝ)
  (accessible_area : ℝ) 
  (h1 : pentagon_side = 1) 
  (h2 : rope_length = 3)
  (h3 : accessible_area = (37 * π) / 5) :
  accessible_area = (π * (rope_length^2) * (288 / 360)) + 2 * (π * (pentagon_side^2) * (36 / 360)) := 
  sorry

end spot_area_l233_233001


namespace purchasing_methods_count_l233_233889

theorem purchasing_methods_count :
  ∃ n, n = 6 ∧
    ∃ (x y : ℕ), 
      60 * x + 70 * y ≤ 500 ∧
      x ≥ 3 ∧
      y ≥ 2 :=
sorry

end purchasing_methods_count_l233_233889


namespace number_of_medium_boxes_l233_233194

def large_box_tape := 4
def medium_box_tape := 2
def small_box_tape := 1
def label_tape := 1

def num_large_boxes := 2
def num_small_boxes := 5
def total_tape := 44

theorem number_of_medium_boxes :
  let tape_used_large_boxes := num_large_boxes * (large_box_tape + label_tape)
  let tape_used_small_boxes := num_small_boxes * (small_box_tape + label_tape)
  let tape_used_medium_boxes := total_tape - (tape_used_large_boxes + tape_used_small_boxes)
  let medium_box_total_tape := medium_box_tape + label_tape
  let num_medium_boxes := tape_used_medium_boxes / medium_box_total_tape
  num_medium_boxes = 8 :=
by
  sorry

end number_of_medium_boxes_l233_233194


namespace calculate_expr_l233_233962

theorem calculate_expr (h1 : Real.sin (30 * Real.pi / 180) = 1 / 2)
    (h2 : Real.cos (30 * Real.pi / 180) = Real.sqrt (3) / 2) :
    3 * Real.tan (30 * Real.pi / 180) + 6 * Real.sin (30 * Real.pi / 180) = 3 + Real.sqrt 3 :=
  sorry

end calculate_expr_l233_233962


namespace parking_monthly_charge_l233_233218

theorem parking_monthly_charge :
  ∀ (M : ℕ), (52 * 10 - 12 * M = 100) → M = 35 :=
by
  intro M h
  sorry

end parking_monthly_charge_l233_233218


namespace solution_replacement_concentration_l233_233164

theorem solution_replacement_concentration :
  ∀ (init_conc replaced_fraction new_conc replaced_conc : ℝ),
    init_conc = 0.45 → replaced_fraction = 0.5 → replaced_conc = 0.25 → new_conc = 35 →
    (init_conc - replaced_fraction * init_conc + replaced_fraction * replaced_conc) * 100 = new_conc :=
by
  intro init_conc replaced_fraction new_conc replaced_conc
  intros h_init h_frac h_replaced h_new
  rw [h_init, h_frac, h_replaced, h_new]
  sorry

end solution_replacement_concentration_l233_233164


namespace find_sinD_l233_233124

variable (DE DF : ℝ)

-- Conditions
def area_of_triangle (DE DF : ℝ) (sinD : ℝ) : Prop :=
  1 / 2 * DE * DF * sinD = 72

def geometric_mean (DE DF : ℝ) : Prop :=
  Real.sqrt (DE * DF) = 15

theorem find_sinD (DE DF sinD : ℝ) (h1 : area_of_triangle DE DF sinD) (h2 : geometric_mean DE DF) :
  sinD = 16 / 25 :=
by 
  -- Proof goes here
  sorry

end find_sinD_l233_233124


namespace average_payment_is_460_l233_233114

theorem average_payment_is_460 :
  let n := 52
  let first_payment := 410
  let extra := 65
  let num_first_payments := 12
  let num_rest_payments := n - num_first_payments
  let rest_payment := first_payment + extra
  (num_first_payments * first_payment + num_rest_payments * rest_payment) / n = 460 := by
  sorry

end average_payment_is_460_l233_233114


namespace length_of_FD_l233_233213

-- Define the conditions
def is_square (ABCD : ℝ) (side_length : ℝ) : Prop :=
  side_length = 8 ∧ ABCD = 4 * side_length

def point_E (x : ℝ) : Prop :=
  x = 8 / 3

def point_F (CD : ℝ) (x : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ 8

-- State the theorem
theorem length_of_FD (side_length : ℝ) (x : ℝ) (CD ED FD : ℝ) :
  is_square 4 side_length → 
  point_E ED → 
  point_F CD x → 
  FD = 20 / 9 :=
by
  sorry

end length_of_FD_l233_233213


namespace only_negative_integer_among_list_l233_233035

namespace NegativeIntegerProblem

def is_integer (n : ℚ) : Prop := ∃ m : ℤ, n = m

theorem only_negative_integer_among_list :
  (∃ x, x ∈ [0, -1, 2, -1.5] ∧ (x < 0) ∧ is_integer x) ↔ (x = -1) :=
by
  sorry

end NegativeIntegerProblem

end only_negative_integer_among_list_l233_233035


namespace find_inner_circle_radius_of_trapezoid_l233_233125

noncomputable def radius_of_inner_circle (k m n p : ℤ) : ℝ :=
  (-k + m * Real.sqrt n) / p

def is_equivalent (a b : ℝ) : Prop := a = b

theorem find_inner_circle_radius_of_trapezoid :
  ∃ (r : ℝ), is_equivalent r (radius_of_inner_circle 123 104 3 29) :=
by
  let r := radius_of_inner_circle 123 104 3 29
  have h1 :  (4^2 + (Real.sqrt (r^2 + 8 * r))^2 = (r + 4)^2) := sorry
  have h2 :  (3^2 + (Real.sqrt (r^2 + 6 * r))^2 = (r + 3)^2) := sorry
  have height_eq : Real.sqrt 13 = (Real.sqrt (r^2 + 6 * r) + Real.sqrt (r^2 + 8 * r)) := sorry
  use r
  exact sorry

end find_inner_circle_radius_of_trapezoid_l233_233125


namespace invertible_elements_mod_8_l233_233689

theorem invertible_elements_mod_8 :
  {x : ℤ | (x * x) % 8 = 1} = {1, 3, 5, 7} :=
by
  sorry

end invertible_elements_mod_8_l233_233689


namespace range_of_real_roots_l233_233630

theorem range_of_real_roots (a : ℝ) :
  (∃ x : ℝ, x^2 + 4*a*x - 4*a + 3 = 0) ∨
  (∃ x : ℝ, x^2 + (a-1)*x + a^2 = 0) ∨
  (∃ x : ℝ, x^2 + 2*a*x - 2*a = 0) ↔
  a >= -1 ∨ a <= -3/2 :=
  sorry

end range_of_real_roots_l233_233630


namespace real_solution_2015x_equation_l233_233973

theorem real_solution_2015x_equation (k : ℝ) :
  (∃ x : ℝ, (4 * 2015^x - 2015^(-x)) / (2015^x - 3 * 2015^(-x)) = k) ↔ (k < 1/3 ∨ k > 4) := 
by sorry

end real_solution_2015x_equation_l233_233973


namespace greatest_b_no_minus_six_in_range_l233_233423

open Real

theorem greatest_b_no_minus_six_in_range :
  ∃ (b : ℤ), (b = 8) → (¬ ∃ x : ℝ, x^2 + (b : ℝ) * x + 15 = -6) :=
by {
  -- We need to find the largest integer b such that -6 is not in the range of f(x) = x^2 + bx + 15
  sorry
}

end greatest_b_no_minus_six_in_range_l233_233423


namespace sale_price_correct_l233_233490

variable (x : ℝ)

-- Conditions
def decreased_price (x : ℝ) : ℝ :=
  0.9 * x

def final_sale_price (decreased_price : ℝ) : ℝ :=
  0.7 * decreased_price

-- Proof statement
theorem sale_price_correct : final_sale_price (decreased_price x) = 0.63 * x := by
  sorry

end sale_price_correct_l233_233490


namespace problem_statement_l233_233337

theorem problem_statement (m n : ℝ) :
  (m^2 - 1840 * m + 2009 = 0) → (n^2 - 1840 * n + 2009 = 0) → 
  (m^2 - 1841 * m + 2009) * (n^2 - 1841 * n + 2009) = 2009 := 
by
  intros h1 h2
  sorry

end problem_statement_l233_233337


namespace intersection_sets_l233_233414

theorem intersection_sets (x : ℝ) :
  let M := {x | 2 * x - x^2 ≥ 0 }
  let N := {x | -1 < x ∧ x < 1}
  M ∩ N = {x | 0 ≤ x ∧ x < 1} :=
by
  sorry

end intersection_sets_l233_233414


namespace tank_volume_ratio_l233_233341

variable {V1 V2 : ℝ}

theorem tank_volume_ratio
  (h1 : 3 / 4 * V1 = 5 / 8 * V2) :
  V1 / V2 = 5 / 6 :=
sorry

end tank_volume_ratio_l233_233341


namespace blue_socks_count_l233_233628

theorem blue_socks_count (total_socks : ℕ) (two_thirds_white : ℕ) (one_third_blue : ℕ) 
  (h1 : total_socks = 180) 
  (h2 : two_thirds_white = (2 / 3) * total_socks) 
  (h3 : one_third_blue = total_socks - two_thirds_white) : 
  one_third_blue = 60 :=
by
  sorry

end blue_socks_count_l233_233628


namespace O_l233_233090

theorem O'Hara_triple_49_16_y : 
  (∃ y : ℕ, (49 : ℕ).sqrt + (16 : ℕ).sqrt = y) → y = 11 :=
by
  sorry

end O_l233_233090


namespace Bulgaria_f_1992_divisibility_l233_233596

def f (m n : ℕ) : ℕ := m^(3^(4 * n) + 6) - m^(3^(4 * n) + 4) - m^5 + m^3

theorem Bulgaria_f_1992_divisibility (n : ℕ) (m : ℕ) :
  ( ∀ m : ℕ, m > 0 → f m n ≡ 0 [MOD 1992] ) ↔ ( n % 2 = 1 ) :=
by
  sorry

end Bulgaria_f_1992_divisibility_l233_233596


namespace ermias_balls_more_is_5_l233_233161

-- Define the conditions
def time_per_ball : ℕ := 20
def alexia_balls : ℕ := 20
def total_time : ℕ := 900

-- Define Ermias's balls
def ermias_balls_more (x : ℕ) : ℕ := alexia_balls + x

-- Alexia's total inflation time
def alexia_total_time : ℕ := alexia_balls * time_per_ball

-- Ermias's total inflation time given x more balls than Alexia
def ermias_total_time (x : ℕ) : ℕ := (ermias_balls_more x) * time_per_ball

-- Total time taken by both Alexia and Ermias
def combined_time (x : ℕ) : ℕ := alexia_total_time + ermias_total_time x

-- Proven that Ermias inflated 5 more balls than Alexia given the total time condition
theorem ermias_balls_more_is_5 : (∃ x : ℕ, combined_time x = total_time) := 
by {
  sorry
}

end ermias_balls_more_is_5_l233_233161


namespace aluminum_iodide_mass_produced_l233_233056

theorem aluminum_iodide_mass_produced
  (mass_Al : ℝ) -- the mass of Aluminum used
  (molar_mass_Al : ℝ) -- molar mass of Aluminum
  (molar_mass_AlI3 : ℝ) -- molar mass of Aluminum Iodide
  (reaction_eq : ∀ (moles_Al : ℝ) (moles_AlI3 : ℝ), 2 * moles_Al = 2 * moles_AlI3) -- reaction equation which indicates a 1:1 molar ratio
  (mass_Al_value : mass_Al = 25.0) 
  (molar_mass_Al_value : molar_mass_Al = 26.98) 
  (molar_mass_AlI3_value : molar_mass_AlI3 = 407.68) :
  ∃ mass_AlI3 : ℝ, mass_AlI3 = 377.52 := by
  sorry

end aluminum_iodide_mass_produced_l233_233056


namespace cross_section_quadrilateral_is_cylinder_l233_233388

-- Definition of the solids
inductive Solid
| cone
| cylinder
| sphere

-- Predicate for the cross-section being a quadrilateral
def is_quadrilateral_cross_section (solid : Solid) : Prop :=
  match solid with
  | Solid.cylinder => true
  | Solid.cone     => false
  | Solid.sphere   => false

-- Main theorem statement
theorem cross_section_quadrilateral_is_cylinder (s : Solid) :
  is_quadrilateral_cross_section s → s = Solid.cylinder :=
by
  cases s
  . simp [is_quadrilateral_cross_section]
  . simp [is_quadrilateral_cross_section]
  . simp [is_quadrilateral_cross_section]

end cross_section_quadrilateral_is_cylinder_l233_233388


namespace roots_sum_squares_l233_233611

theorem roots_sum_squares (a b c : ℝ) (h₁ : Polynomial.eval a (3 * X^3 - 2 * X^2 + 5 * X + 15) = 0)
  (h₂ : Polynomial.eval b (3 * X^3 - 2 * X^2 + 5 * X + 15) = 0)
  (h₃ : Polynomial.eval c (3 * X^3 - 2 * X^2 + 5 * X + 15) = 0) :
  a^2 + b^2 + c^2 = -26 / 9 :=
sorry

end roots_sum_squares_l233_233611


namespace functional_eq_is_odd_function_l233_233245

theorem functional_eq_is_odd_function (f : ℝ → ℝ)
  (hf_nonzero : ∃ x : ℝ, f x ≠ 0)
  (hf_eq : ∀ a b : ℝ, f (a * b) = a * f b + b * f a) :
  ∀ x : ℝ, f (-x) = -f x :=
by
  sorry

end functional_eq_is_odd_function_l233_233245


namespace min_value_expression_l233_233438

theorem min_value_expression : 
  ∃ x y : ℝ, (∀ a b : ℝ, 2 * a^2 + 3 * a * b + 4 * b^2 + 5 ≥ 5) ∧ (2 * x^2 + 3 * x * y + 4 * y^2 + 5 = 5) := 
by 
sorry

end min_value_expression_l233_233438


namespace middle_part_division_l233_233467

theorem middle_part_division 
  (x : ℝ) 
  (x_pos : x > 0) 
  (H : x + (1 / 4) * x + (1 / 8) * x = 96) :
  (1 / 4) * x = 17 + 21 / 44 :=
by
  sorry

end middle_part_division_l233_233467


namespace nate_reading_percentage_l233_233069

-- Given conditions
def total_pages := 400
def pages_to_read := 320

-- Calculate the number of pages he has already read
def pages_read := total_pages - pages_to_read

-- Prove the percentage of the book Nate has finished reading
theorem nate_reading_percentage : (pages_read / total_pages) * 100 = 20 := by
  sorry

end nate_reading_percentage_l233_233069


namespace cyclist_avg_speed_l233_233410

theorem cyclist_avg_speed (d : ℝ) (h1 : d > 0) :
  let t_1 := d / 17
  let t_2 := d / 23
  let total_time := t_1 + t_2
  let total_distance := 2 * d
  (total_distance / total_time) = 19.55 :=
by
  -- Proof steps here
  sorry

end cyclist_avg_speed_l233_233410


namespace greater_segment_difference_l233_233825

theorem greater_segment_difference :
  ∀ (L1 L2 : ℝ), L1 = 7 ∧ L1^2 - L2^2 = 32 → L1 - L2 = 7 - Real.sqrt 17 :=
by
  intros L1 L2 h
  sorry

end greater_segment_difference_l233_233825


namespace sufficient_condition_B_is_proper_subset_of_A_l233_233362

def A : Set ℝ := {x | x^2 + x = 6}
def B (m : ℝ) : Set ℝ := {-1 / m}

theorem sufficient_condition_B_is_proper_subset_of_A (m : ℝ) : 
  m = -1/2 → B m ⊆ A ∧ B m ≠ A :=
by
  sorry

end sufficient_condition_B_is_proper_subset_of_A_l233_233362


namespace no_integer_roots_l233_233354

theorem no_integer_roots (n : ℕ) (p : Fin (2*n + 1) → ℤ)
  (non_zero : ∀ i, p i ≠ 0)
  (sum_non_zero : (Finset.univ.sum (λ i => p i)) ≠ 0) :
  ∃ P : ℤ → ℤ, ∀ x : ℤ, P x ≠ 0 → x > 1 ∨ x < -1 := sorry

end no_integer_roots_l233_233354


namespace original_price_l233_233017

theorem original_price (p q: ℝ) (h₁ : p ≠ 0) (h₂ : q ≠ 0) : 
  let x := 20000 / (10000^2 - (p^2 + q^2) * 10000 + p^2 * q^2)
  (x : ℝ) * (1 - p^2 / 10000) * (1 - q^2 / 10000) = 2 :=
by
  sorry

end original_price_l233_233017


namespace find_actual_average_height_l233_233283

noncomputable def actualAverageHeight (avg_height : ℕ) (num_boys : ℕ) (wrong_height : ℕ) (actual_height : ℕ) : Float :=
  let incorrect_total := avg_height * num_boys
  let difference := wrong_height - actual_height
  let correct_total := incorrect_total - difference
  (Float.ofInt correct_total) / (Float.ofNat num_boys)

theorem find_actual_average_height (avg_height num_boys wrong_height actual_height : ℕ) :
  avg_height = 185 ∧ num_boys = 35 ∧ wrong_height = 166 ∧ actual_height = 106 →
  actualAverageHeight avg_height num_boys wrong_height actual_height = 183.29 := by
  intros h
  have h_avg := h.1
  have h_num := h.2.1
  have h_wrong := h.2.2.1
  have h_actual := h.2.2.2
  rw [h_avg, h_num, h_wrong, h_actual]
  sorry

end find_actual_average_height_l233_233283


namespace cream_strawberry_prices_l233_233081

noncomputable def price_flavor_B : ℝ := 30
noncomputable def price_flavor_A : ℝ := 40

theorem cream_strawberry_prices (x y : ℝ) 
  (h1 : y = x + 10) 
  (h2 : 800 / y = 600 / x) : 
  x = price_flavor_B ∧ y = price_flavor_A :=
by 
  sorry

end cream_strawberry_prices_l233_233081


namespace Mark_same_color_opposite_foot_l233_233963

variable (shoes : Finset (Σ _ : Fin (14), Bool))

def same_color_opposite_foot_probability (shoes : Finset (Σ _ : Fin (14), Bool)) : ℚ := 
  let total_shoes : ℚ := 28
  let num_black_pairs := 7
  let num_brown_pairs := 4
  let num_gray_pairs := 2
  let num_white_pairs := 1
  let black_pair_prob  := (14 / total_shoes) * (7 / (total_shoes - 1))
  let brown_pair_prob  := (8 / total_shoes) * (4 / (total_shoes - 1))
  let gray_pair_prob   := (4 / total_shoes) * (2 / (total_shoes - 1))
  let white_pair_prob  := (2 / total_shoes) * (1 / (total_shoes - 1))
  black_pair_prob + brown_pair_prob + gray_pair_prob + white_pair_prob

theorem Mark_same_color_opposite_foot (shoes : Finset (Σ _ : Fin (14), Bool)) :
  same_color_opposite_foot_probability shoes = 35 / 189 := 
sorry

end Mark_same_color_opposite_foot_l233_233963


namespace triangle_ABC_is_right_triangle_l233_233738

theorem triangle_ABC_is_right_triangle (A B C : ℝ) (hA : A = 68) (hB : B = 22) :
  A + B + C = 180 → C = 90 :=
by
  intro hABC
  sorry

end triangle_ABC_is_right_triangle_l233_233738


namespace christen_potatoes_and_total_time_l233_233625

-- Variables representing the given conditions
variables (homer_rate : ℕ) (christen_rate : ℕ) (initial_potatoes : ℕ) 
(homer_time_alone : ℕ) (total_time : ℕ)

-- Specific values for the given problem
def homerRate := 4
def christenRate := 6
def initialPotatoes := 60
def homerTimeAlone := 5

-- Function to calculate the number of potatoes peeled by Homer alone
def potatoesPeeledByHomerAlone :=
  homerRate * homerTimeAlone

-- Function to calculate the number of remaining potatoes
def remainingPotatoes :=
  initialPotatoes - potatoesPeeledByHomerAlone

-- Function to calculate the total peeling rate when Homer and Christen are working together
def combinedRate :=
  homerRate + christenRate

-- Function to calculate the time taken to peel the remaining potatoes
def timePeelingTogether :=
  remainingPotatoes / combinedRate

-- Function to calculate the total time spent peeling potatoes
def totalTime :=
  homerTimeAlone + timePeelingTogether

-- Function to calculate the number of potatoes peeled by Christen
def potatoesPeeledByChristen :=
  christenRate * timePeelingTogether

/- The theorem to be proven: Christen peeled 24 potatoes, and it took 9 minutes to peel all the potatoes. -/
theorem christen_potatoes_and_total_time :
  (potatoesPeeledByChristen = 24) ∧ (totalTime = 9) :=
by {
  sorry
}

end christen_potatoes_and_total_time_l233_233625


namespace intersection_M_N_l233_233559

noncomputable def M : Set ℝ := {x | x^2 - x ≤ 0}
noncomputable def N : Set ℝ := {x | x < 1}

theorem intersection_M_N : M ∩ N = {x | 0 ≤ x ∧ x < 1} :=
by
  sorry

end intersection_M_N_l233_233559


namespace last_digit_of_2_pow_2004_l233_233358

theorem last_digit_of_2_pow_2004 : (2 ^ 2004) % 10 = 6 := 
by {
  sorry
}

end last_digit_of_2_pow_2004_l233_233358


namespace derivative_u_l233_233464

noncomputable def u (x : ℝ) : ℝ :=
  let z := Real.sin x
  let y := x^2
  Real.exp (z - 2 * y)

theorem derivative_u (x : ℝ) :
  deriv u x = Real.exp (Real.sin x - 2 * x^2) * (Real.cos x - 4 * x) :=
by
  sorry

end derivative_u_l233_233464


namespace calc_g_f_neg_2_l233_233305

def f (x : ℝ) : ℝ := x^3 - 4 * x + 3
def g (x : ℝ) : ℝ := 2 * x^2 + 2 * x + 1

theorem calc_g_f_neg_2 : g (f (-2)) = 25 := by
  sorry

end calc_g_f_neg_2_l233_233305


namespace total_amount_of_money_l233_233096

def one_rupee_note_value := 1
def five_rupee_note_value := 5
def ten_rupee_note_value := 10

theorem total_amount_of_money (n : ℕ) 
  (h : 3 * n = 90) : n * one_rupee_note_value + n * five_rupee_note_value + n * ten_rupee_note_value = 480 :=
by
  sorry

end total_amount_of_money_l233_233096


namespace avg_annual_growth_rate_l233_233961

variable (x : ℝ)

/-- Initial GDP in 2020 is 43903.89 billion yuan and GDP in 2022 is 53109.85 billion yuan. 
    Prove that the average annual growth rate x satisfies the equation 43903.89 * (1 + x)^2 = 53109.85 -/
theorem avg_annual_growth_rate (x : ℝ) :
  43903.89 * (1 + x)^2 = 53109.85 :=
sorry

end avg_annual_growth_rate_l233_233961


namespace complex_number_quadrant_l233_233649

open Complex

theorem complex_number_quadrant (z : ℂ) (h : (1 + 2 * Complex.I) / z = Complex.I) : 
  (0 < z.re) ∧ (0 < z.im) :=
by
  -- sorry to skip the actual proof
  sorry

end complex_number_quadrant_l233_233649


namespace smallest_base_l233_233771

theorem smallest_base (k b : ℕ) (h_k : k = 6) : 64 ^ k > b ^ 16 ↔ b < 5 :=
by
  have h1 : 64 ^ k = 2 ^ (6 * k) := by sorry
  have h2 : 2 ^ (6 * k) > b ^ 16 := by sorry
  exact sorry

end smallest_base_l233_233771


namespace tapA_fill_time_l233_233436

-- Define the conditions
def fillTapA (t : ℕ) := 1 / t
def fillTapB := 1 / 40
def fillCombined (t : ℕ) := 9 * (fillTapA t + fillTapB)
def fillRemaining := 23 * fillTapB

-- Main theorem statement
theorem tapA_fill_time : ∀ (t : ℕ), fillCombined t + fillRemaining = 1 → t = 45 := by
  sorry

end tapA_fill_time_l233_233436


namespace number_division_remainder_l233_233175

theorem number_division_remainder (N k m : ℤ) (h1 : N = 281 * k + 160) (h2 : N = D * m + 21) : D = 139 :=
by sorry

end number_division_remainder_l233_233175


namespace sum_of_multiples_of_4_between_34_and_135_l233_233866

theorem sum_of_multiples_of_4_between_34_and_135 :
  let first := 36
  let last := 132
  let n := (last - first) / 4 + 1
  let sum := n * (first + last) / 2
  sum = 2100 := 
by
  sorry

end sum_of_multiples_of_4_between_34_and_135_l233_233866


namespace equal_intercepts_condition_l233_233726

theorem equal_intercepts_condition (a b c : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) :
  (a = b ∨ c = 0) ↔ (c = 0 ∨ (c ≠ 0 ∧ a = b)) :=
by sorry

end equal_intercepts_condition_l233_233726


namespace dante_walk_time_l233_233782

-- Define conditions and problem
variables (T R : ℝ)

-- Conditions as per the problem statement
def wind_in_favor_condition : Prop := 0.8 * T = 15
def wind_against_condition : Prop := 1.25 * T = 7
def total_walk_time_condition : Prop := 15 + 7 = 22
def total_time_away_condition : Prop := 32 - 22 = 10
def lake_park_restaurant_condition : Prop := 0.8 * R = 10

-- Proof statement
theorem dante_walk_time :
  wind_in_favor_condition T ∧
  wind_against_condition T ∧
  total_walk_time_condition ∧
  total_time_away_condition ∧
  lake_park_restaurant_condition R →
  R = 12.5 :=
by
  intros
  sorry

end dante_walk_time_l233_233782


namespace two_digit_numbers_reverse_square_condition_l233_233082

theorem two_digit_numbers_reverse_square_condition :
  ∀ (a b : ℕ), 0 ≤ a ∧ a < 10 ∧ 0 ≤ b ∧ b < 10 →
  (∃ n : ℕ, 10 * a + b + 10 * b + a = n^2) ↔ 
  (10 * a + b = 29 ∨ 10 * a + b = 38 ∨ 10 * a + b = 47 ∨ 10 * a + b = 56 ∨ 
   10 * a + b = 65 ∨ 10 * a + b = 74 ∨ 10 * a + b = 83 ∨ 10 * a + b = 92) :=
by {
  sorry
}

end two_digit_numbers_reverse_square_condition_l233_233082


namespace tom_killed_enemies_l233_233927

-- Define the number of points per enemy
def points_per_enemy : ℝ := 10

-- Define the bonus threshold and bonus factor
def bonus_threshold : ℝ := 100
def bonus_factor : ℝ := 1.5

-- Define the total score achieved by Tom
def total_score : ℝ := 2250

-- Define the number of enemies killed by Tom
variable (E : ℝ)

-- The proof goal
theorem tom_killed_enemies 
  (h1 : E ≥ bonus_threshold)
  (h2 : bonus_factor * points_per_enemy * E = total_score) : 
  E = 150 :=
sorry

end tom_killed_enemies_l233_233927


namespace simplify_and_evaluate_expr_l233_233755

theorem simplify_and_evaluate_expr :
  let x := (Real.sqrt 5 + 1) / 2 
  let y := (Real.sqrt 5 - 1) / 2 
  (x - 2*y)^2 + x*(5*y - x) - 4*y^2 = 1 :=
by
  let x := (Real.sqrt 5 + 1) / 2 
  let y := (Real.sqrt 5 - 1) / 2 
  sorry

end simplify_and_evaluate_expr_l233_233755


namespace jake_last_10_shots_l233_233819

-- conditions
variable (total_shots_initially : ℕ) (shots_made_initially : ℕ) (percentage_initial : ℝ)
variable (total_shots_finally : ℕ) (shots_made_finally : ℕ) (percentage_final : ℝ)

axiom initial_conditions : shots_made_initially = percentage_initial * total_shots_initially
axiom final_conditions : shots_made_finally = percentage_final * total_shots_finally
axiom shots_difference : total_shots_finally - total_shots_initially = 10

-- prove that Jake made 7 out of the last 10 shots
theorem jake_last_10_shots : total_shots_initially = 30 → 
                             percentage_initial = 0.60 →
                             total_shots_finally = 40 → 
                             percentage_final = 0.62 →
                             shots_made_finally - shots_made_initially = 7 :=
by
  -- proofs to be filled in
  sorry

end jake_last_10_shots_l233_233819


namespace maximize_profit_constraints_l233_233622

variable (a1 a2 b1 b2 d1 d2 c1 c2 x y z : ℝ)

theorem maximize_profit_constraints (a1 a2 b1 b2 d1 d2 c1 c2 x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  (a1 * x + a2 * y ≤ c1) ∧ (b1 * x + b2 * y ≤ c2) :=
sorry

end maximize_profit_constraints_l233_233622


namespace soccer_game_points_ratio_l233_233583

theorem soccer_game_points_ratio :
  ∃ B1 A1 A2 B2 : ℕ,
    A1 = 8 ∧
    B2 = 8 ∧
    A2 = 6 ∧
    (A1 + B1 + A2 + B2 = 26) ∧
    (B1 / A1 = 1 / 2) := by
  sorry

end soccer_game_points_ratio_l233_233583


namespace sawing_steel_bar_time_l233_233106

theorem sawing_steel_bar_time (pieces : ℕ) (time_per_cut : ℕ) : 
  pieces = 6 → time_per_cut = 2 → (pieces - 1) * time_per_cut = 10 := 
by
  intros
  sorry

end sawing_steel_bar_time_l233_233106


namespace dot_product_theorem_l233_233451

open Real

namespace VectorProof

-- Define the vectors m and n
def m := (2, 5)
def n (t : ℝ) := (-5, t)

-- Define the condition that m is perpendicular to n
def perpendicular (t : ℝ) : Prop := (2 * -5) + (5 * t) = 0

-- Function to calculate the dot product
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Define the vectors m+n and m-2n
def vector_add (t : ℝ) : ℝ × ℝ := (m.1 + (n t).1, m.2 + (n t).2)
def vector_sub (t : ℝ) : ℝ × ℝ := (m.1 - 2 * (n t).1, m.2 - 2 * (n t).2)

-- The theorem to prove
theorem dot_product_theorem : ∀ (t : ℝ), perpendicular t → dot_product (vector_add t) (vector_sub t) = -29 :=
by
  intros t ht
  sorry

end VectorProof

end dot_product_theorem_l233_233451


namespace Frank_has_four_one_dollar_bills_l233_233440

noncomputable def Frank_one_dollar_bills : ℕ :=
  let total_money := 4 * 5 + 2 * 10 + 20 -- Money from five, ten, and twenty dollar bills
  let peanuts_cost := 10 - 4 -- Cost of peanuts (given $10 and received $4 in change)
  let one_dollar_bills_value := 54 - total_money -- Total money Frank has - money from large bills
  (one_dollar_bills_value : ℕ)

theorem Frank_has_four_one_dollar_bills 
   (five_dollar_bills : ℕ := 4) 
   (ten_dollar_bills : ℕ := 2)
   (twenty_dollar_bills : ℕ := 1)
   (peanut_price : ℚ := 3)
   (change : ℕ := 4)
   (total_money : ℕ := 50)
   (total_money_incl_change : ℚ := 54):
   Frank_one_dollar_bills = 4 := by
  sorry

end Frank_has_four_one_dollar_bills_l233_233440


namespace quadratic_standard_form_l233_233378

theorem quadratic_standard_form :
  ∀ x : ℝ, (2 * x - 1) ^ 2 = (x + 1) * (3 * x + 4) →
  (∃ a b c : ℝ, a = 1 ∧ b = -11 ∧ c = -3 ∧ a * x^2 + b * x + c = 0) :=
by
  intro x
  intro h
  sorry

end quadratic_standard_form_l233_233378


namespace initial_water_percentage_l233_233243

noncomputable def initial_percentage_of_water : ℚ :=
  20

theorem initial_water_percentage
  (initial_volume : ℚ := 125)
  (added_water : ℚ := 8.333333333333334)
  (final_volume : ℚ := initial_volume + added_water)
  (desired_percentage : ℚ := 25)
  (desired_amount_of_water : ℚ := desired_percentage / 100 * final_volume)
  (initial_amount_of_water : ℚ := desired_amount_of_water - added_water) :
  (initial_amount_of_water / initial_volume * 100 = initial_percentage_of_water) :=
by
  sorry

end initial_water_percentage_l233_233243


namespace nails_per_plank_l233_233595

theorem nails_per_plank {total_nails planks : ℕ} (h1 : total_nails = 4) (h2 : planks = 2) :
  total_nails / planks = 2 := by
  sorry

end nails_per_plank_l233_233595


namespace find_a_l233_233215

def star (a b : ℕ) : ℕ := 3 * a - b ^ 2

theorem find_a (a : ℕ) (b : ℕ) (h : star a b = 14) : a = 10 :=
by sorry

end find_a_l233_233215


namespace valid_common_ratios_count_l233_233046

noncomputable def num_valid_common_ratios (a₁ : ℝ) (q : ℝ) : ℝ :=
  let a₅ := a₁ * q^4
  let a₃ := a₁ * q^2
  if 2 * a₅ = 4 * a₁ + (-2) * a₃ then 1 else 0

theorem valid_common_ratios_count (a₁ : ℝ) : 
  (num_valid_common_ratios a₁ 1) + (num_valid_common_ratios a₁ (-1)) = 2 :=
by sorry

end valid_common_ratios_count_l233_233046


namespace probability_not_red_l233_233398

theorem probability_not_red (h : odds_red = 1 / 3) : probability_not_red_card = 3 / 4 :=
by
  sorry

end probability_not_red_l233_233398


namespace volume_between_concentric_spheres_l233_233698

theorem volume_between_concentric_spheres
  (r1 r2 : ℝ) (h_r1 : r1 = 5) (h_r2 : r2 = 10) :
  (4 / 3 * Real.pi * r2^3 - 4 / 3 * Real.pi * r1^3) = (3500 / 3) * Real.pi :=
by
  rw [h_r1, h_r2]
  sorry

end volume_between_concentric_spheres_l233_233698


namespace cannot_achieve_1970_minuses_l233_233673

theorem cannot_achieve_1970_minuses :
  ∃ (x y : ℕ), x ≤ 100 ∧ y ≤ 100 ∧ (x - 50) * (y - 50) = 1515 → false :=
by
  sorry

end cannot_achieve_1970_minuses_l233_233673


namespace scientific_notation_for_70_million_l233_233763

-- Define the parameters for the problem
def scientific_notation (x : ℕ) (a : ℝ) (n : ℤ) : Prop :=
  x = a * 10 ^ n ∧ 1 ≤ |a| ∧ |a| < 10

-- Problem statement
theorem scientific_notation_for_70_million :
  scientific_notation 70000000 7.0 7 :=
by
  sorry

end scientific_notation_for_70_million_l233_233763


namespace rectangular_sheet_integers_l233_233048

noncomputable def at_least_one_integer (a b : ℝ) : Prop :=
  ∃ i : ℤ, a = i ∨ b = i

theorem rectangular_sheet_integers (a b : ℝ)
  (h_positive_a : a > 0)
  (h_positive_b : b > 0)
  (h_cut_lines : ∀ x y : ℝ, (∃ k : ℤ, x = k ∧ y = 1 ∨ y = k ∧ x = 1) → (∃ z : ℤ, x = z ∨ y = z)) :
  at_least_one_integer a b :=
sorry

end rectangular_sheet_integers_l233_233048


namespace northton_time_capsule_depth_l233_233449

def southton_depth : ℕ := 15

def northton_depth : ℕ := 4 * southton_depth + 12

theorem northton_time_capsule_depth : northton_depth = 72 := by
  sorry

end northton_time_capsule_depth_l233_233449


namespace commute_time_absolute_difference_l233_233204

theorem commute_time_absolute_difference 
  (x y : ℝ)
  (h1 : (x + y + 10 + 11 + 9) / 5 = 10)
  (h2 : ((x - 10)^2 + (y - 10)^2 + 0^2 + 1^2 + (-1)^2) / 5 = 2) :
  |x - y| = 4 :=
by sorry

end commute_time_absolute_difference_l233_233204


namespace shaded_area_of_intersections_l233_233800

theorem shaded_area_of_intersections (r : ℝ) (n : ℕ) (intersect_origin : Prop) (radius_5 : r = 5) (four_circles : n = 4) : 
  ∃ (area : ℝ), area = 100 * Real.pi - 200 :=
by
  sorry

end shaded_area_of_intersections_l233_233800


namespace find_m_l233_233024

-- Define the conditions
def function_is_decreasing (m : ℝ) : Prop := 
  (m^2 - m - 1 = 1) ∧ (1 - m < 0)

-- The proof problem: prove m = 2 given the conditions
theorem find_m (m : ℝ) (h : function_is_decreasing m) : m = 2 := 
by
  sorry -- Proof to be filled in

end find_m_l233_233024


namespace cyclist_final_speed_l233_233139

def u : ℝ := 16
def a : ℝ := 0.5
def t : ℕ := 7200

theorem cyclist_final_speed : 
  (u + a * t) * 3.6 = 13017.6 := by
  sorry

end cyclist_final_speed_l233_233139


namespace not_sufficient_nor_necessary_condition_l233_233481

def is_even_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = f x

def is_increasing_for_nonpositive (f : ℝ → ℝ) : Prop :=
∀ x y, x ≤ 0 → y ≤ 0 → x < y → f x < f y

theorem not_sufficient_nor_necessary_condition
  {f : ℝ → ℝ}
  (hf_even : is_even_function f)
  (hf_incr : is_increasing_for_nonpositive f)
  (x : ℝ) :
  (6/5 < x ∧ x < 2) → ¬((1 < x ∧ x < 7/4) ↔ (f (Real.log (2 * x - 2) / Real.log 2) > f (Real.log (2 / 3) / Real.log (1 / 2)))) :=
sorry

end not_sufficient_nor_necessary_condition_l233_233481


namespace negation_exists_implication_l233_233173

theorem negation_exists_implication (x : ℝ) : (¬ ∃ y > 0, y^2 - 2*y - 3 ≤ 0) ↔ ∀ y > 0, y^2 - 2*y - 3 > 0 :=
by
  sorry

end negation_exists_implication_l233_233173


namespace solve_abs_eqn_l233_233452

theorem solve_abs_eqn (y : ℝ) : (|y - 4| + 3 * y = 15) ↔ (y = 19 / 4) :=
by
  sorry

end solve_abs_eqn_l233_233452


namespace min_editors_at_conference_l233_233713

variable (x E : ℕ)

theorem min_editors_at_conference (h1 : x ≤ 26) 
    (h2 : 100 = 35 + E + x) 
    (h3 : 2 * x ≤ 100 - 35 - E + x) : 
    E ≥ 39 :=
by
  sorry

end min_editors_at_conference_l233_233713


namespace shots_per_puppy_l233_233351

-- Definitions
def num_pregnant_dogs : ℕ := 3
def puppies_per_dog : ℕ := 4
def cost_per_shot : ℕ := 5
def total_shot_cost : ℕ := 120

-- Total number of puppies
def total_puppies : ℕ := num_pregnant_dogs * puppies_per_dog

-- Total number of shots
def total_shots : ℕ := total_shot_cost / cost_per_shot

-- The theorem to prove
theorem shots_per_puppy : total_shots / total_puppies = 2 :=
by
  sorry

end shots_per_puppy_l233_233351


namespace combined_percentage_grade4_l233_233711

-- Definitions based on the given conditions
def Pinegrove_total_students : ℕ := 120
def Maplewood_total_students : ℕ := 180

def Pinegrove_grade4_percentage : ℕ := 10
def Maplewood_grade4_percentage : ℕ := 20

theorem combined_percentage_grade4 :
  let combined_total_students := Pinegrove_total_students + Maplewood_total_students
  let Pinegrove_grade4_students := Pinegrove_grade4_percentage * Pinegrove_total_students / 100
  let Maplewood_grade4_students := Maplewood_grade4_percentage * Maplewood_total_students / 100 
  let combined_grade4_students := Pinegrove_grade4_students + Maplewood_grade4_students
  (combined_grade4_students * 100 / combined_total_students) = 16 := by
  sorry

end combined_percentage_grade4_l233_233711


namespace problem1_problem2_l233_233084

theorem problem1 (a b x y : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < x ∧ 0 < y) : 
  (a^2 / x + b^2 / y) ≥ ((a + b)^2 / (x + y)) ∧ (a * y = b * x → (a^2 / x + b^2 / y) = ((a + b)^2 / (x + y))) :=
sorry

theorem problem2 (x : ℝ) (h : 0 < x ∧ x < 1 / 2) :
  (∀ x, 0 < x ∧ x < 1 / 2 → ((2 / x + 9 / (1 - 2 * x)) ≥ 25)) ∧ (2 * (1 - 2 * (1 / 5)) = 9 * (1 / 5) → (2 / (1 / 5) + 9 / (1 - 2 * (1 / 5)) = 25)) :=
sorry

end problem1_problem2_l233_233084


namespace james_muffins_baked_l233_233392

-- Define the number of muffins Arthur baked
def muffinsArthur : ℕ := 115

-- Define the multiplication factor
def multiplicationFactor : ℕ := 12

-- Define the number of muffins James baked
def muffinsJames : ℕ := muffinsArthur * multiplicationFactor

-- The theorem that needs to be proved
theorem james_muffins_baked : muffinsJames = 1380 :=
by
  sorry

end james_muffins_baked_l233_233392


namespace perfect_square_condition_l233_233701

-- Definitions from conditions
def is_integer (x : ℝ) : Prop := ∃ k : ℤ, x = k

def is_perfect_square (n : ℤ) : Prop := ∃ m : ℤ, n = m^2

-- Theorem statement
theorem perfect_square_condition (n : ℤ) (h1 : 0 < n) (h2 : is_integer (2 + 2 * Real.sqrt (1 + 12 * (n: ℝ)^2))) : 
  is_perfect_square n :=
by
  sorry

end perfect_square_condition_l233_233701


namespace guards_can_protect_point_l233_233798

-- Define the conditions of the problem as Lean definitions
def guardVisionRadius : ℝ := 100

-- Define the proof statement
theorem guards_can_protect_point :
  ∃ (num_guards : ℕ), num_guards * 45 = 360 ∧ guardVisionRadius = 100 :=
by
  sorry

end guards_can_protect_point_l233_233798


namespace remainder_when_dividing_by_y_minus_4_l233_233985

def g (y : ℤ) : ℤ := y^5 - 8 * y^4 + 12 * y^3 + 25 * y^2 - 40 * y + 24

theorem remainder_when_dividing_by_y_minus_4 : g 4 = 8 :=
by
  sorry

end remainder_when_dividing_by_y_minus_4_l233_233985


namespace rice_bag_weight_l233_233970

theorem rice_bag_weight (r f : ℕ) (total_weight : ℕ) (h1 : 20 * r + 50 * f = 2250) (h2 : r = 2 * f) : r = 50 := 
by
  sorry

end rice_bag_weight_l233_233970


namespace coprime_gcd_l233_233309

theorem coprime_gcd (a b : ℤ) (h : Int.gcd a b = 1) : Int.gcd (2 * a + b) (a * (a + b)) = 1 := 
sorry

end coprime_gcd_l233_233309


namespace Mike_age_l233_233610

-- We define the ages of Mike and Barbara
variables (M B : ℕ)

-- Conditions extracted from the problem
axiom h1 : B = M / 2
axiom h2 : M - B = 8

-- The theorem to prove
theorem Mike_age : M = 16 :=
by sorry

end Mike_age_l233_233610


namespace math_score_is_75_l233_233428

def average_of_four_subjects (s1 s2 s3 s4 : ℕ) : ℕ := (s1 + s2 + s3 + s4) / 4
def total_of_four_subjects (s1 s2 s3 s4 : ℕ) : ℕ := s1 + s2 + s3 + s4
def average_of_five_subjects (s1 s2 s3 s4 s5 : ℕ) : ℕ := (s1 + s2 + s3 + s4 + s5) / 5
def total_of_five_subjects (s1 s2 s3 s4 s5 : ℕ) : ℕ := s1 + s2 + s3 + s4 + s5

theorem math_score_is_75 (s1 s2 s3 s4 : ℕ) (h1 : average_of_four_subjects s1 s2 s3 s4 = 90)
                            (h2 : average_of_five_subjects s1 s2 s3 s4 s5 = 87) :
  s5 = 75 :=
by
  sorry

end math_score_is_75_l233_233428


namespace remainder_sum_of_numbers_l233_233477

theorem remainder_sum_of_numbers :
  ((123450 + 123451 + 123452 + 123453 + 123454 + 123455) % 7) = 5 :=
by
  sorry

end remainder_sum_of_numbers_l233_233477


namespace min_ones_count_in_100_numbers_l233_233808

def sum_eq_product (l : List ℕ) : Prop :=
  l.sum = l.prod

theorem min_ones_count_in_100_numbers : ∀ l : List ℕ, l.length = 100 → sum_eq_product l → l.count 1 ≥ 95 :=
by sorry

end min_ones_count_in_100_numbers_l233_233808


namespace acute_triangle_tangent_difference_range_l233_233629

theorem acute_triangle_tangent_difference_range {A B C a b c : ℝ} 
    (h1 : a^2 + b^2 > c^2) (h2 : b^2 + c^2 > a^2) (h3 : c^2 + a^2 > b^2)
    (hb2_minus_ha2_eq_ac : b^2 - a^2 = a * c) :
    1 < (1 / Real.tan A - 1 / Real.tan B) ∧ (1 / Real.tan A - 1 / Real.tan B) < (2 * Real.sqrt 3 / 3) :=
by
  sorry

end acute_triangle_tangent_difference_range_l233_233629


namespace longest_side_range_of_obtuse_triangle_l233_233144

theorem longest_side_range_of_obtuse_triangle (a b c : ℝ) (h₁ : a = 1) (h₂ : b = 2) :
  a^2 + b^2 < c^2 → (Real.sqrt 5 < c ∧ c < 3) ∨ c = 2 :=
by
  sorry

end longest_side_range_of_obtuse_triangle_l233_233144


namespace range_of_t_l233_233852

noncomputable def f (t : ℝ) (x : ℝ) : ℝ :=
if x ≤ 0 then x^2 + 2 * t * x + t^2 else x + 1 / x + t

theorem range_of_t (t : ℝ) :
  (∀ x : ℝ, f t 0 ≤ f t x) ↔ (0 ≤ t ∧ t ≤ 2) :=
by sorry

end range_of_t_l233_233852


namespace problem_eqn_l233_233178

theorem problem_eqn (a b c : ℝ) :
  (∃ r₁ r₂ : ℝ, r₁^2 + 3 * r₁ - 1 = 0 ∧ r₂^2 + 3 * r₂ - 1 = 0) ∧
  (∀ x : ℝ, (x^2 + 3 * x - 1 = 0) → (x^4 + a * x^2 + b * x + c = 0)) →
  a + b + 4 * c = -7 :=
by
  sorry

end problem_eqn_l233_233178


namespace domain_f_2x_plus_1_eq_l233_233715

-- Conditions
def domain_fx_plus_1 : Set ℝ := {x : ℝ | -2 < x ∧ x < -1}

-- Question and Correct Answer
theorem domain_f_2x_plus_1_eq :
  (∃ (x : ℝ), x ∈ domain_fx_plus_1) →
  {x : ℝ | -1 < x ∧ x < -1/2} = {x : ℝ | (2*x + 1 ∈ domain_fx_plus_1)} :=
by
  sorry

end domain_f_2x_plus_1_eq_l233_233715


namespace total_pencils_correct_l233_233729

def initial_pencils : ℕ := 245
def added_pencils : ℕ := 758
def total_pencils : ℕ := initial_pencils + added_pencils

theorem total_pencils_correct : total_pencils = 1003 := 
by
  sorry

end total_pencils_correct_l233_233729


namespace ferris_wheel_seats_l233_233548

theorem ferris_wheel_seats (total_people seats_capacity : ℕ) (h1 : total_people = 8) (h2 : seats_capacity = 3) : 
  Nat.ceil ((total_people : ℚ) / (seats_capacity : ℚ)) = 3 := 
by
  sorry

end ferris_wheel_seats_l233_233548


namespace find_positive_k_l233_233456

noncomputable def polynomial_with_equal_roots (k: ℚ) : Prop := 
  ∃ a b : ℚ, a ≠ b ∧ 2 * a + b = -3 ∧ 2 * a * b + a^2 = -50 ∧ k = -2 * a^2 * b

theorem find_positive_k : ∃ k : ℚ, polynomial_with_equal_roots k ∧ 0 < k ∧ k = 950 / 27 :=
by
  sorry

end find_positive_k_l233_233456


namespace distinct_collections_proof_l233_233888

noncomputable def distinct_collections_count : ℕ := 240

theorem distinct_collections_proof : distinct_collections_count = 240 := by
  sorry

end distinct_collections_proof_l233_233888


namespace students_per_group_correct_l233_233551

def total_students : ℕ := 850
def number_of_teachers : ℕ := 23
def students_per_group : ℕ := total_students / number_of_teachers

theorem students_per_group_correct : students_per_group = 36 := sorry

end students_per_group_correct_l233_233551


namespace participants_in_sports_activities_l233_233303

theorem participants_in_sports_activities:
  ∃ (x y z : ℤ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x + y + z = 3 ∧
  let a := 10 * x + 6
  let b := 10 * y + 6
  let c := 10 * z + 6
  a + b + c = 48 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (a = 6 ∧ b = 16 ∧ c = 26 ∨ a = 6 ∧ b = 26 ∧ c = 16 ∨ a = 16 ∧ b = 6 ∧ c = 26 ∨ a = 16 ∧ b = 26 ∧ c = 6 ∨ a = 26 ∧ b = 6 ∧ c = 16 ∨ a = 26 ∧ b = 16 ∧ c = 6)
  :=
by {
  sorry
}

end participants_in_sports_activities_l233_233303


namespace lara_harvest_raspberries_l233_233193

-- Define measurements of the garden
def length : ℕ := 10
def width : ℕ := 7

-- Define planting and harvesting constants
def plants_per_sq_ft : ℕ := 5
def raspberries_per_plant : ℕ := 12

-- Calculate expected number of raspberries
theorem lara_harvest_raspberries :  length * width * plants_per_sq_ft * raspberries_per_plant = 4200 := 
by sorry

end lara_harvest_raspberries_l233_233193


namespace triangle_area_example_l233_233901

def point : Type := (ℝ × ℝ)

def triangle_area (A B C : point) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem triangle_area_example : 
  triangle_area (0, 0) (0, 6) (8, 10) = 24 :=
by
  sorry

end triangle_area_example_l233_233901


namespace combined_6th_grade_percentage_l233_233717

noncomputable def percentage_of_6th_graders 
  (parkPercent : Fin 7 → ℚ) 
  (riversidePercent : Fin 7 → ℚ) 
  (totalParkside : ℕ) 
  (totalRiverside : ℕ) 
  : ℚ := 
    let num6thParkside := parkPercent 6 * totalParkside
    let num6thRiverside := riversidePercent 6 * totalRiverside
    let total6thGraders := num6thParkside + num6thRiverside
    let totalStudents := totalParkside + totalRiverside
    (total6thGraders / totalStudents) * 100

theorem combined_6th_grade_percentage :
  let parkPercent := ![(14.0 : ℚ) / 100, 13 / 100, 16 / 100, 15 / 100, 12 / 100, 15 / 100, 15 / 100]
  let riversidePercent := ![(13.0 : ℚ) / 100, 16 / 100, 13 / 100, 15 / 100, 14 / 100, 15 / 100, 14 / 100]
  percentage_of_6th_graders parkPercent riversidePercent 150 250 = 15 := 
  by
  sorry

end combined_6th_grade_percentage_l233_233717


namespace angle_B_is_60_l233_233325

theorem angle_B_is_60 (A B C : ℝ) (h_seq : 2 * B = A + C) (h_sum : A + B + C = 180) : B = 60 := 
by 
  sorry

end angle_B_is_60_l233_233325


namespace tangent_line_at_1_1_is_5x_plus_y_minus_6_l233_233727

noncomputable def f : ℝ → ℝ :=
  λ x => x^3 - 4*x^2 + 4

def tangent_line_equation (x₀ y₀ m : ℝ) : ℝ → ℝ → Prop :=
  λ x y => y - y₀ = m * (x - x₀)

theorem tangent_line_at_1_1_is_5x_plus_y_minus_6 : 
  tangent_line_equation 1 1 (-5) = (λ x y => 5 * x + y - 6 = 0) := 
by
  sorry

end tangent_line_at_1_1_is_5x_plus_y_minus_6_l233_233727


namespace position_of_term_in_sequence_l233_233335

theorem position_of_term_in_sequence 
    (a : ℕ → ℕ)
    (h1 : a 1 = 1)
    (h2 : ∀ n, a (n + 1) - a n = 7 * n) :
    ∃ n, a n = 35351 ∧ n = 101 :=
by
  sorry

end position_of_term_in_sequence_l233_233335


namespace greatest_length_of_pieces_l233_233198

/-- Alicia has three ropes with lengths of 28 inches, 42 inches, and 70 inches.
She wants to cut these ropes into equal length pieces for her art project, and she doesn't want any leftover pieces.
Prove that the greatest length of each piece she can cut is 7 inches. -/
theorem greatest_length_of_pieces (a b c : ℕ) (h1 : a = 28) (h2 : b = 42) (h3 : c = 70) :
  ∃ (d : ℕ), d > 0 ∧ d ∣ a ∧ d ∣ b ∧ d ∣ c ∧ ∀ e : ℕ, e > 0 ∧ e ∣ a ∧ e ∣ b ∧ e ∣ c → e ≤ d := sorry

end greatest_length_of_pieces_l233_233198


namespace roots_of_poly_l233_233019

noncomputable def poly (x : ℝ) : ℝ := x^3 - 4 * x^2 - x + 4

theorem roots_of_poly :
  (poly 1 = 0) ∧ (poly (-1) = 0) ∧ (poly 4 = 0) ∧
  (∀ x, poly x = 0 → x = 1 ∨ x = -1 ∨ x = 4) :=
by
  sorry

end roots_of_poly_l233_233019


namespace quadratic_intersects_x_axis_l233_233747

theorem quadratic_intersects_x_axis (a b : ℝ) (h : a ≠ 0) :
    ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a * x1^2 + b * x1 - (b^2 / (4 * a)) = 0 ∧ a * x2^2 + b * x2 - (b^2 / (4 * a)) = 0 := by
  sorry

end quadratic_intersects_x_axis_l233_233747


namespace find_original_number_l233_233883

theorem find_original_number (h1 : 268 * 74 = 19732) (h2 : 2.68 * x = 1.9832) : x = 0.74 :=
sorry

end find_original_number_l233_233883


namespace vertex_angle_isosceles_l233_233582

theorem vertex_angle_isosceles (a b c : ℝ)
  (isosceles: (a = b ∨ b = c ∨ c = a))
  (angle_sum : a + b + c = 180)
  (one_angle_is_70 : a = 70 ∨ b = 70 ∨ c = 70) :
  a = 40 ∨ a = 70 ∨ b = 40 ∨ b = 70 ∨ c = 40 ∨ c = 70 :=
by sorry

end vertex_angle_isosceles_l233_233582


namespace milk_percentage_after_adding_water_l233_233318

theorem milk_percentage_after_adding_water
  (initial_total_volume : ℚ) (initial_milk_percentage : ℚ)
  (additional_water_volume : ℚ) :
  initial_total_volume = 60 → initial_milk_percentage = 0.84 → additional_water_volume = 18.75 →
  (50.4 / (initial_total_volume + additional_water_volume) * 100 = 64) :=
by
  intros h1 h2 h3
  rw [h1, h3]
  simp
  sorry

end milk_percentage_after_adding_water_l233_233318


namespace parabola_directrix_l233_233647

variable (a : ℝ)

theorem parabola_directrix (h1 : ∀ x : ℝ, y = a * x^2) (h2 : y = -1/4) : a = 1 :=
sorry

end parabola_directrix_l233_233647


namespace root_interval_k_l233_233170

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 6

theorem root_interval_k (k : ℤ) (h_cont : Continuous f) (h_mono : Monotone f)
  (h1 : f 2 < 0) (h2 : f 3 > 0) : k = 4 :=
by
  -- The proof part is omitted as per instruction.
  sorry

end root_interval_k_l233_233170


namespace largest_divisor_of_five_consecutive_odds_l233_233900

theorem largest_divisor_of_five_consecutive_odds (n : ℕ) (hn : n % 2 = 0) :
    ∃ d, d = 15 ∧ ∀ m, (m = (n + 3) * (n + 5) * (n + 7) * (n + 9) * (n + 11)) → d ∣ m :=
sorry

end largest_divisor_of_five_consecutive_odds_l233_233900


namespace decorative_plate_painted_fraction_l233_233370

noncomputable def fraction_painted_area (total_area painted_area : ℕ) : ℚ :=
  painted_area / total_area

theorem decorative_plate_painted_fraction :
  let side_length := 4
  let total_area := side_length * side_length
  let painted_smaller_squares := 6
  fraction_painted_area total_area painted_smaller_squares = 3 / 8 :=
by
  sorry

end decorative_plate_painted_fraction_l233_233370


namespace Shara_shells_total_l233_233823

def initial_shells : ℕ := 20
def shells_per_day : ℕ := 5
def days : ℕ := 3
def shells_fourth_day : ℕ := 6

theorem Shara_shells_total : (initial_shells + (shells_per_day * days) + shells_fourth_day) = 41 := by
  sorry

end Shara_shells_total_l233_233823


namespace factor_of_M_l233_233327

theorem factor_of_M (a b c : ℕ) (h₀ : a ≠ 0) (h₁ : a < 10) (h₂ : b < 10) (h₃ : c < 10) : 
  1 ∣ (101010 * a + 10001 * b + 100 * c) :=
sorry

end factor_of_M_l233_233327


namespace find_kn_l233_233356

theorem find_kn (k n : ℕ) (h_pos_k : k > 0) (h_pos_n : n > 0) (h_eq : k^2 - 2016 = 3^n) : k = 45 ∧ n = 2 :=
by
  sorry

end find_kn_l233_233356


namespace roger_total_distance_l233_233868

theorem roger_total_distance :
  let morning_ride_miles := 2
  let evening_ride_miles := 5 * morning_ride_miles
  let next_day_morning_ride_km := morning_ride_miles * 1.6
  let next_day_ride_km := 2 * next_day_morning_ride_km
  let next_day_ride_miles := next_day_ride_km / 1.6
  morning_ride_miles + evening_ride_miles + next_day_ride_miles = 16 :=
by
  sorry

end roger_total_distance_l233_233868


namespace original_six_digit_number_l233_233209

theorem original_six_digit_number :
  ∃ a b c d e : ℕ, 
  (100000 + 10000 * a + 1000 * b + 100 * c + 10 * d + e = 142857) ∧ 
  (100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + 1 = 64 * (100000 + 10000 * a + 1000 * b + 100 * c + 10 * d + e)) :=
by
  sorry

end original_six_digit_number_l233_233209


namespace action_figure_prices_l233_233721

noncomputable def prices (x y z w : ℝ) : Prop :=
  12 * x + 8 * y + 5 * z + 10 * w = 220 ∧
  x / 4 = y / 3 ∧
  x / 4 = z / 2 ∧
  x / 4 = w / 1

theorem action_figure_prices :
  ∃ x y z w : ℝ, prices x y z w ∧
    x = 220 / 23 ∧
    y = (3 / 4) * (220 / 23) ∧
    z = (1 / 2) * (220 / 23) ∧
    w = (1 / 4) * (220 / 23) :=
  sorry

end action_figure_prices_l233_233721


namespace no_n_exists_11_div_mod_l233_233379

theorem no_n_exists_11_div_mod (n : ℕ) (h1 : n > 0) (h2 : 3^5 ≡ 1 [MOD 11]) (h3 : 4^5 ≡ 1 [MOD 11]) : ¬ (11 ∣ (3^n + 4^n)) := 
sorry

end no_n_exists_11_div_mod_l233_233379


namespace ways_to_divide_day_l233_233654

theorem ways_to_divide_day (n m : ℕ+) : n * m = 86400 → 96 = 96 :=
by
  sorry

end ways_to_divide_day_l233_233654


namespace arithmetic_expression_evaluation_l233_233929

theorem arithmetic_expression_evaluation : (8 / 2 - 3 * 2 + 5^2 / 5) = 3 := by
  sorry

end arithmetic_expression_evaluation_l233_233929


namespace points_four_units_away_l233_233010

theorem points_four_units_away (x : ℚ) (h : |x| = 4) : x = -4 ∨ x = 4 := 
by 
  sorry

end points_four_units_away_l233_233010


namespace problem_solution_l233_233532

noncomputable def x : ℝ := 3 / 0.15
noncomputable def y : ℝ := 3 / 0.25
noncomputable def z : ℝ := 0.30 * y

theorem problem_solution : x - y + z = 11.6 := sorry

end problem_solution_l233_233532


namespace dividend_is_correct_l233_233790

def quotient : ℕ := 20
def divisor : ℕ := 66
def remainder : ℕ := 55

def dividend := (divisor * quotient) + remainder

theorem dividend_is_correct : dividend = 1375 := by
  sorry

end dividend_is_correct_l233_233790


namespace simplify_abs_expression_l233_233455

theorem simplify_abs_expression (a b c : ℝ) (h1 : a > 0) (h2 : b < 0) (h3 : c = 0) :
  |a - c| + |c - b| - |a - b| = 0 := 
by
  sorry

end simplify_abs_expression_l233_233455


namespace distance_between_centers_of_intersecting_circles_l233_233205

theorem distance_between_centers_of_intersecting_circles
  {r R d : ℝ} (hrR : r < R) (hr : 0 < r) (hR : 0 < R)
  (h_intersect : d < r + R ∧ d > R - r) :
  R - r < d ∧ d < r + R := by
  sorry

end distance_between_centers_of_intersecting_circles_l233_233205


namespace lcm_ac_least_value_l233_233896

theorem lcm_ac_least_value (a b c : ℕ) (h1 : Nat.lcm a b = 20) (h2 : Nat.lcm b c = 24) : 
  Nat.lcm a c = 30 :=
sorry

end lcm_ac_least_value_l233_233896


namespace geometric_sequence_frac_l233_233732

variable (a : ℕ → ℝ)
variable (q : ℝ)
variable (h_geometric : ∃ q > 0, ∀ n, a (n+1) = a n * q)
variable (h_decreasing : ∀ n, a (n+1) < a n)
variable (h1 : a 2 * a 8 = 6)
variable (h2 : a 4 + a 6 = 5)

theorem geometric_sequence_frac (h_geometric : ∃ q > 0, ∀ n, a (n+1) = a n * q)
                                (h_decreasing : ∀ n, a (n+1) < a n)
                                (h1 : a 2 * a 8 = 6)
                                (h2 : a 4 + a 6 = 5) :
                                a 3 / a 7 = 9 / 4 :=
by sorry

end geometric_sequence_frac_l233_233732


namespace volume_sphere_gt_cube_l233_233307

theorem volume_sphere_gt_cube (a r : ℝ) (h : 6 * a^2 = 4 * π * r^2) : 
  (4 / 3) * π * r^3 > a^3 :=
by sorry

end volume_sphere_gt_cube_l233_233307
