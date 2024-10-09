import Mathlib

namespace measure_of_smaller_angle_l348_34879

noncomputable def complementary_angle_ratio_smaller (x : ℝ) (h : 4 * x + x = 90) : ℝ :=
x

theorem measure_of_smaller_angle (x : ℝ) (h : 4 * x + x = 90) : complementary_angle_ratio_smaller x h = 18 :=
sorry

end measure_of_smaller_angle_l348_34879


namespace sum_of_first_six_terms_of_geom_seq_l348_34840

theorem sum_of_first_six_terms_of_geom_seq :
  let a := (1 : ℝ) / 4
  let r := (1 : ℝ) / 4
  let S6 := a * (1 - r^6) / (1 - r)
  S6 = 4095 / 12288 := by
sorry

end sum_of_first_six_terms_of_geom_seq_l348_34840


namespace indeterminate_equation_solution_l348_34830

theorem indeterminate_equation_solution (x y : ℝ) (n : ℕ) :
  (x^2 + (x + 1)^2 = y^2) ↔ 
  (x = 1/4 * ((1 + Real.sqrt 2)^(2*n + 1) + (1 - Real.sqrt 2)^(2*n + 1) - 2) ∧ 
   y = 1/(2 * Real.sqrt 2) * ((1 + Real.sqrt 2)^(2*n + 1) - (1 - Real.sqrt 2)^(2*n + 1))) := 
sorry

end indeterminate_equation_solution_l348_34830


namespace find_old_weight_l348_34872

variable (avg_increase : ℝ) (num_persons : ℕ) (W_new : ℝ) (total_increase : ℝ) (W_old : ℝ)

theorem find_old_weight (h1 : avg_increase = 3.5) 
                        (h2 : num_persons = 7) 
                        (h3 : W_new = 99.5) 
                        (h4 : total_increase = num_persons * avg_increase) 
                        (h5 : W_new = W_old + total_increase) 
                        : W_old = 75 :=
by
  sorry

end find_old_weight_l348_34872


namespace donny_spending_l348_34813

theorem donny_spending :
  (15 + 28 + 13) / 2 = 28 :=
by
  sorry

end donny_spending_l348_34813


namespace negation_of_p_l348_34832

theorem negation_of_p : 
  (¬(∀ x : ℝ, |x| < 0)) ↔ (∃ x : ℝ, |x| ≥ 0) :=
by {
  sorry
}

end negation_of_p_l348_34832


namespace jesse_pencils_l348_34878

def initial_pencils : ℕ := 78
def pencils_given : ℕ := 44
def final_pencils : ℕ := initial_pencils - pencils_given

theorem jesse_pencils :
  final_pencils = 34 :=
by
  -- Proof goes here
  sorry

end jesse_pencils_l348_34878


namespace geometric_locus_points_l348_34865

theorem geometric_locus_points :
  (∀ x y : ℝ, (y^2 = x^2) ↔ (y = x ∨ y = -x)) ∧
  (∀ x : ℝ, (x^2 - 2 * x + 1 = 0) ↔ (x = 1)) ∧
  (∀ x y : ℝ, (x^2 + y^2 = 4 * (y - 1)) ↔ (x = 0 ∧ y = 2)) ∧
  (∀ x y : ℝ, (x^2 - 2 * x * y + y^2 = -1) ↔ false) :=
by
  sorry

end geometric_locus_points_l348_34865


namespace prod_eq_diff_squares_l348_34863

variable (a b : ℝ)

theorem prod_eq_diff_squares :
  ( (1 / 4 * a + b) * (b - 1 / 4 * a) = b^2 - (1 / 16 * a^2) ) :=
by
  sorry

end prod_eq_diff_squares_l348_34863


namespace max_count_larger_than_20_l348_34802

noncomputable def max_larger_than_20 (int_list : List Int) : Nat :=
  (int_list.filter (λ n => n > 20)).length

theorem max_count_larger_than_20 (a1 a2 a3 a4 a5 a6 a7 a8 : Int)
  (h_sum : a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 = 10) :
  ∃ (k : Nat), k = 7 ∧ max_larger_than_20 [a1, a2, a3, a4, a5, a6, a7, a8] = k :=
sorry

end max_count_larger_than_20_l348_34802


namespace union_M_N_eq_l348_34856

def M : Set ℝ := {x | x^2 - 4 * x < 0}
def N : Set ℝ := {0, 4}

theorem union_M_N_eq : M ∪ N = Set.Icc 0 4 := 
  by
    sorry

end union_M_N_eq_l348_34856


namespace maplewood_total_population_l348_34837

-- Define the number of cities
def num_cities : ℕ := 25

-- Define the bounds for the average population
def lower_bound : ℕ := 5200
def upper_bound : ℕ := 5700

-- Define the average population, calculated as the midpoint of the bounds
def average_population : ℕ := (lower_bound + upper_bound) / 2

-- Define the total population as the product of the number of cities and the average population
def total_population : ℕ := num_cities * average_population

-- Theorem statement to prove the total population is 136,250
theorem maplewood_total_population : total_population = 136250 := by
  -- Insert formal proof here
  sorry

end maplewood_total_population_l348_34837


namespace squirrels_in_tree_l348_34858

-- Definitions based on the conditions
def nuts : Nat := 2
def squirrels : Nat := nuts + 2

-- Theorem stating the main proof problem
theorem squirrels_in_tree : squirrels = 4 := by
  -- Proof steps would go here, but we're adding sorry to skip them
  sorry

end squirrels_in_tree_l348_34858


namespace bus_tour_total_sales_l348_34842

noncomputable def total_sales (total_tickets sold_senior_tickets : Nat) (cost_senior_ticket cost_regular_ticket : Nat) : Nat :=
  let sold_regular_tickets := total_tickets - sold_senior_tickets
  let sales_senior := sold_senior_tickets * cost_senior_ticket
  let sales_regular := sold_regular_tickets * cost_regular_ticket
  sales_senior + sales_regular

theorem bus_tour_total_sales :
  total_sales 65 24 10 15 = 855 := by
    sorry

end bus_tour_total_sales_l348_34842


namespace pears_picking_total_l348_34826

theorem pears_picking_total :
  let Jason_day1 := 46
  let Keith_day1 := 47
  let Mike_day1 := 12
  let Alicia_day1 := 28
  let Tina_day1 := 33
  let Nicola_day1 := 52

  let Jason_day2 := Jason_day1 / 2
  let Keith_day2 := Keith_day1 / 2
  let Mike_day2 := Mike_day1 / 2
  let Alicia_day2 := 2 * Alicia_day1
  let Tina_day2 := 2 * Tina_day1
  let Nicola_day2 := 2 * Nicola_day1

  let Jason_day3 := (Jason_day1 + Jason_day2) / 2
  let Keith_day3 := (Keith_day1 + Keith_day2) / 2
  let Mike_day3 := (Mike_day1 + Mike_day2) / 2
  let Alicia_day3 := (Alicia_day1 + Alicia_day2) / 2
  let Tina_day3 := (Tina_day1 + Tina_day2) / 2
  let Nicola_day3 := (Nicola_day1 + Nicola_day2) / 2

  let Jason_total := Jason_day1 + Jason_day2 + Jason_day3
  let Keith_total := Keith_day1 + Keith_day2 + Keith_day3
  let Mike_total := Mike_day1 + Mike_day2 + Mike_day3
  let Alicia_total := Alicia_day1 + Alicia_day2 + Alicia_day3
  let Tina_total := Tina_day1 + Tina_day2 + Tina_day3
  let Nicola_total := Nicola_day1 + Nicola_day2 + Nicola_day3

  let overall_total := Jason_total + Keith_total + Mike_total + Alicia_total + Tina_total + Nicola_total

  overall_total = 747 := by
  intro Jason_day1 Jason_day2 Jason_day3 Jason_total
  intro Keith_day1 Keith_day2 Keith_day3 Keith_total
  intro Mike_day1 Mike_day2 Mike_day3 Mike_total
  intro Alicia_day1 Alicia_day2 Alicia_day3 Alicia_total
  intro Tina_day1 Tina_day2 Tina_day3 Tina_total
  intro Nicola_day1 Nicola_day2 Nicola_day3 Nicola_total

  sorry

end pears_picking_total_l348_34826


namespace participated_in_both_l348_34839

-- Define the conditions
def total_students := 40
def math_competition := 31
def physics_competition := 20
def not_participating := 8

-- Define number of students participated in both competitions
def both_competitions := 59 - total_students

-- Theorem statement
theorem participated_in_both : both_competitions = 19 := 
sorry

end participated_in_both_l348_34839


namespace rectangle_y_value_l348_34810

theorem rectangle_y_value (y : ℝ) (h1 : -2 < 6) (h2 : y > 2) 
    (h3 : 8 * (y - 2) = 64) : y = 10 :=
by
  sorry

end rectangle_y_value_l348_34810


namespace graph_is_empty_l348_34829

theorem graph_is_empty : ∀ (x y : ℝ), 3 * x^2 + y^2 - 9 * x - 4 * y + 17 ≠ 0 :=
by
  intros x y
  sorry

end graph_is_empty_l348_34829


namespace number_to_add_l348_34801

theorem number_to_add (a m : ℕ) (h₁ : a = 7844213) (h₂ : m = 549) :
  ∃ n, (a + n) % m = 0 ∧ n = m - (a % m) :=
by
  sorry

end number_to_add_l348_34801


namespace total_cars_in_group_l348_34852

theorem total_cars_in_group (C : ℕ)
  (h1 : 37 ≤ C)
  (h2 : ∃ n ≥ 51, n ≤ C)
  (h3 : ∃ n ≤ 49, n + 51 = C - 37) :
  C = 137 :=
by
  sorry

end total_cars_in_group_l348_34852


namespace solve_equation_l348_34833

theorem solve_equation (x : ℝ) : 
  (4 * (1 - x)^2 = 25) ↔ (x = -3 / 2 ∨ x = 7 / 2) := 
by 
  sorry

end solve_equation_l348_34833


namespace tangent_line_parallel_to_x_axis_l348_34844

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

noncomputable def f_derivative (x : ℝ) : ℝ := (1 - Real.log x) / (x^2)

theorem tangent_line_parallel_to_x_axis :
  ∀ x₀ : ℝ, 
  f_derivative x₀ = 0 → 
  f x₀ = 1 / Real.exp 1 :=
by
  intro x₀ h_deriv_zero
  sorry

end tangent_line_parallel_to_x_axis_l348_34844


namespace min_value_fraction_l348_34845

theorem min_value_fraction (a b : ℝ) (h₀ : a > b) (h₁ : a * b = 1) :
  ∃ c, c = (2 * Real.sqrt 2) ∧ (a^2 + b^2) / (a - b) ≥ c :=
by sorry

end min_value_fraction_l348_34845


namespace problem_solution_l348_34871

-- Definitions of the arithmetic sequence a_n and its common difference and first term
variables (a d : ℝ)

-- Definitions of arithmetic sequence conditions
def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ :=
  a + (n - 1) * d

-- Required conditions for the proof
variables (h1 : d ≠ 0) (h2 : a ≠ 0)
variables (h3 : arithmetic_sequence a d 2 * arithmetic_sequence a d 8 = (arithmetic_sequence a d 4) ^ 2)

-- The target theorem to prove
theorem problem_solution : 
  (a + (a + 4 * d) + (a + 8 * d)) / ((a + d) + (a + 2 * d)) = 3 :=
sorry

end problem_solution_l348_34871


namespace prob_no_1_or_6_l348_34870

theorem prob_no_1_or_6 :
  ∀ (a b c : ℕ), (1 ≤ a ∧ a ≤ 6) ∧ (1 ≤ b ∧ b ≤ 6) ∧ (1 ≤ c ∧ c ≤ 6) →
  (8 / 27 : ℝ) = (4 / 6) * (4 / 6) * (4 / 6) :=
by
  intros a b c h
  sorry

end prob_no_1_or_6_l348_34870


namespace impossible_coins_l348_34887

theorem impossible_coins (p1 p2 : ℝ) (hp1 : 0 ≤ p1 ∧ p1 ≤ 1) (hp2 : 0 ≤ p2 ∧ p2 ≤ 1) :
  (1 - p1) * (1 - p2) = p1 * p2 ∧ p1 * (1 - p2) + p2 * (1 - p1) = p1 * p2 → false :=
by 
  sorry

end impossible_coins_l348_34887


namespace current_population_is_15336_l348_34828

noncomputable def current_population : ℝ :=
  let growth_rate := 1.28
  let future_population : ℝ := 25460.736
  let years := 2
  future_population / (growth_rate ^ years)

theorem current_population_is_15336 :
  current_population = 15536 := sorry

end current_population_is_15336_l348_34828


namespace katie_added_new_songs_l348_34881

-- Definitions for the conditions
def initial_songs := 11
def deleted_songs := 7
def current_songs := 28

-- Definition of the expected answer
def new_songs_added := current_songs - (initial_songs - deleted_songs)

-- Statement of the problem in Lean
theorem katie_added_new_songs : new_songs_added = 24 :=
by
  sorry

end katie_added_new_songs_l348_34881


namespace geometric_progression_common_ratio_l348_34876

-- Define the problem conditions in Lean 4
theorem geometric_progression_common_ratio (a : ℕ → ℝ) (r : ℝ) (n : ℕ)
  (h_pos : ∀ n, a n > 0) 
  (h_rel : ∀ n, a n = (a (n + 1) + a (n + 2)) / 2 + 2 ) : 
  r = 1 :=
sorry

end geometric_progression_common_ratio_l348_34876


namespace hiker_total_distance_l348_34835

theorem hiker_total_distance :
  let day1_distance := 18
  let day1_speed := 3
  let day2_speed := day1_speed + 1
  let day1_time := day1_distance / day1_speed
  let day2_time := day1_time - 1
  let day2_distance := day2_speed * day2_time
  let day3_speed := 5
  let day3_time := 3
  let day3_distance := day3_speed * day3_time
  let total_distance := day1_distance + day2_distance + day3_distance
  total_distance = 53 :=
by
  sorry

end hiker_total_distance_l348_34835


namespace inverse_proportion_quadrants_l348_34857

theorem inverse_proportion_quadrants (k : ℝ) (h : k ≠ 0) :
  ∀ (x y : ℝ), y = k^2 / x → (x > 0 → y > 0) ∧ (x < 0 → y < 0) :=
by
  sorry

end inverse_proportion_quadrants_l348_34857


namespace at_least_two_equal_l348_34874

theorem at_least_two_equal
  {a b c d : ℝ}
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h₁ : a + b + (1 / (a * b)) = c + d + (1 / (c * d)))
  (h₂ : (1 / a) + (1 / b) + (a * b) = (1 / c) + (1 / d) + (c * d)) :
  a = c ∨ a = d ∨ b = c ∨ b = d ∨ a = b ∨ c = d := by
  sorry

end at_least_two_equal_l348_34874


namespace clear_time_is_approximately_7_point_1_seconds_l348_34831

-- Constants for the lengths of the trains in meters
def length_train1 : ℕ := 121
def length_train2 : ℕ := 165

-- Constants for the speeds of the trains in km/h
def speed_train1 : ℕ := 80
def speed_train2 : ℕ := 65

-- Kilometer to meter conversion
def km_to_meter (km : ℕ) : ℕ := km * 1000

-- Hour to second conversion
def hour_to_second (h : ℕ) : ℕ := h * 3600

-- Relative speed of the trains in meters per second
noncomputable def relative_speed_m_per_s : ℕ := 
  (km_to_meter (speed_train1 + speed_train2)) / hour_to_second 1

-- Total distance to be covered in meters
def total_distance : ℕ := length_train1 + length_train2

-- Time to be completely clear of each other in seconds
noncomputable def clear_time : ℝ := total_distance / (relative_speed_m_per_s : ℝ)

theorem clear_time_is_approximately_7_point_1_seconds :
  abs (clear_time - 7.1) < 0.01 :=
by
  sorry

end clear_time_is_approximately_7_point_1_seconds_l348_34831


namespace find_angle_APB_l348_34817

-- Definitions based on conditions
def r1 := 2 -- Radius of semicircle SAR
def r2 := 3 -- Radius of semicircle RBT

def angle_AO1S := 70
def angle_BO2T := 40

def angle_AO1R := 180 - angle_AO1S
def angle_BO2R := 180 - angle_BO2T

def angle_PA := 90
def angle_PB := 90

-- Statement of the theorem
theorem find_angle_APB : angle_PA + angle_AO1R + angle_BO2R + angle_PB + 110 = 540 :=
by
  -- Unused in proof: added only to state theorem 
  have _ := angle_PA
  have _ := angle_AO1R
  have _ := angle_BO2R
  have _ := angle_PB
  have _ := 110
  sorry

end find_angle_APB_l348_34817


namespace find_kn_l348_34861

section
variables (k n : ℝ)

def system_infinite_solutions (k n : ℝ) :=
  ∃ (y : ℝ → ℝ) (x : ℝ → ℝ),
  (∀ y, k * y + x y + n = 0) ∧
  (∀ y, |y - 2| + |y + 1| + |1 - y| + |y + 2| + x y = 0)

theorem find_kn :
  { (k, n) | system_infinite_solutions k n } = {(4, 0), (-4, 0), (2, 4), (-2, 4), (0, 6)} :=
sorry
end

end find_kn_l348_34861


namespace olivia_correct_answers_l348_34895

theorem olivia_correct_answers (c w : ℕ) (h1 : c + w = 15) (h2 : 4 * c - 3 * w = 25) : c = 10 :=
by
  sorry

end olivia_correct_answers_l348_34895


namespace beaker_volume_l348_34803

theorem beaker_volume {a b c d e f g h i j : ℝ} (h₁ : a = 7) (h₂ : b = 4) (h₃ : c = 5)
                      (h₄ : d = 4) (h₅ : e = 6) (h₆ : f = 8) (h₇ : g = 7)
                      (h₈ : h = 3) (h₉ : i = 9) (h₁₀ : j = 6) :
  (a + b + c + d + e + f + g + h + i + j) / 5 = 11.8 :=
by
  sorry

end beaker_volume_l348_34803


namespace area_of_trapezoid_MBCN_l348_34818

variables {AB BC MN : ℝ}
variables {Area_ABCD Area_MBCN : ℝ}
variables {Height : ℝ}

-- Given conditions
def cond1 : Area_ABCD = 40 := sorry
def cond2 : AB = 8 := sorry
def cond3 : BC = 5 := sorry
def cond4 : MN = 2 := sorry
def cond5 : Height = 5 := sorry

-- Define the theorem to be proven
theorem area_of_trapezoid_MBCN : 
  Area_ABCD = AB * BC → MN + BC = 6 → Height = 5 →
  Area_MBCN = (1/2) * (MN + BC) * Height → 
  Area_MBCN = 15 :=
by
  intros h1 h2 h3 h4
  sorry

end area_of_trapezoid_MBCN_l348_34818


namespace person_B_days_l348_34850

theorem person_B_days (A_days : ℕ) (combined_work : ℚ) (x : ℕ) : 
  A_days = 30 → combined_work = (1 / 6) → 3 * (1 / 30 + 1 / x) = combined_work → x = 45 :=
by
  intros hA hCombined hEquation
  sorry

end person_B_days_l348_34850


namespace find_k_l348_34834

theorem find_k (x k : ℤ) (h : 2 * k - x = 2) (hx : x = -4) : k = -1 :=
by
  rw [hx] at h
  -- Substituting x = -4 into the equation
  sorry  -- Skipping further proof steps

end find_k_l348_34834


namespace geometric_sequence_sum_l348_34848

variable (a : ℕ → ℝ)

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r, ∀ n, a (n + 1) = a n * r

theorem geometric_sequence_sum (h1 : geometric_sequence a)
  (h2 : a 1 > 0)
  (h3 : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 36) :
  a 3 + a 5 = 6 :=
sorry

end geometric_sequence_sum_l348_34848


namespace blue_paint_needed_l348_34866

theorem blue_paint_needed (total_cans : ℕ) (blue_ratio : ℕ) (yellow_ratio : ℕ)
  (h_ratio: blue_ratio = 5) (h_yellow_ratio: yellow_ratio = 3) (h_total: total_cans = 45) : 
  ⌊total_cans * (blue_ratio : ℝ) / (blue_ratio + yellow_ratio)⌋ = 28 :=
by
  sorry

end blue_paint_needed_l348_34866


namespace exists_infinitely_many_N_l348_34816

open Set

-- Conditions: Definition of the initial set S_0 and recursive sets S_n
variable {S_0 : Set ℕ} (h0 : Set.Finite S_0) -- S_0 is a finite set of positive integers
variable (S : ℕ → Set ℕ) 
(has_S : ∀ n, ∀ a, a ∈ S (n+1) ↔ (a-1 ∈ S n ∧ a ∉ S n ∨ a-1 ∉ S n ∧ a ∈ S n))

-- Main theorem: Proving the existence of infinitely many integers N such that 
-- S_N = S_0 ∪ {N + a : a ∈ S_0}
theorem exists_infinitely_many_N : 
  ∃ᶠ N in at_top, S N = S_0 ∪ {n | ∃ a ∈ S_0, n = N + a} := 
sorry

end exists_infinitely_many_N_l348_34816


namespace tangent_circle_distance_proof_l348_34882

noncomputable def tangent_circle_distance (R r : ℝ) (tangent_type : String) : ℝ :=
  if tangent_type = "external" then R + r else R - r

theorem tangent_circle_distance_proof (R r : ℝ) (tangent_type : String) (hR : R = 4) (hr : r = 3) :
  tangent_circle_distance R r tangent_type = 7 ∨ tangent_circle_distance R r tangent_type = 1 := by
  sorry

end tangent_circle_distance_proof_l348_34882


namespace number_of_minibuses_l348_34849

theorem number_of_minibuses (total_students : ℕ) (capacity : ℕ) (h : total_students = 48) (h_capacity : capacity = 8) : 
  ∃ minibuses, minibuses = (total_students + capacity - 1) / capacity ∧ minibuses = 7 :=
by
  have h1 : (48 + 8 - 1) = 55 := by simp [h, h_capacity]
  have h2 : 55 / 8 = 6 := by simp [h, h_capacity]
  use 7
  sorry

end number_of_minibuses_l348_34849


namespace maximal_n_for_sequence_l348_34809

theorem maximal_n_for_sequence
  (a : ℕ → ℤ)
  (n : ℕ)
  (h1 : ∀ i : ℕ, 1 ≤ i ∧ i ≤ n - 2 → a i + a (i + 1) + a (i + 2) > 0)
  (h2 : ∀ i : ℕ, 1 ≤ i ∧ i ≤ n - 4 → a i + a (i + 1) + a (i + 2) + a (i + 3) + a (i + 4) < 0)
  : n ≤ 9 :=
sorry

end maximal_n_for_sequence_l348_34809


namespace original_price_sarees_l348_34847

theorem original_price_sarees (P : ℝ) (h : 0.85 * 0.80 * P = 272) : P = 400 :=
by
  sorry

end original_price_sarees_l348_34847


namespace apples_kilos_first_scenario_l348_34846

noncomputable def cost_per_kilo_oranges : ℝ := 29
noncomputable def cost_per_kilo_apples : ℝ := 29
noncomputable def cost_first_scenario : ℝ := 419
noncomputable def cost_second_scenario : ℝ := 488
noncomputable def kilos_oranges_first_scenario : ℝ := 6
noncomputable def kilos_oranges_second_scenario : ℝ := 5
noncomputable def kilos_apples_second_scenario : ℝ := 7

theorem apples_kilos_first_scenario
  (O A : ℝ) 
  (cost1 cost2 : ℝ) 
  (k_oranges1 k_oranges2 k_apples2 : ℝ) 
  (hO : O = 29) (hA : A = 29) 
  (hCost1 : k_oranges1 * O + x * A = cost1) 
  (hCost2 : k_oranges2 * O + k_apples2 * A = cost2) 
  : x = 8 :=
by
  have hO : O = 29 := sorry
  have hA : A = 29 := sorry
  have h1 : k_oranges1 * O + x * A = cost1 := sorry
  have h2 : k_oranges2 * O + k_apples2 * A = cost2 := sorry
  sorry

end apples_kilos_first_scenario_l348_34846


namespace range_of_f_lt_0_l348_34822

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def decreasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f x > f y

variable (f : ℝ → ℝ)
variable (h_odd : is_odd f)
variable (h_decreasing : decreasing_on f (Set.Iic 0))
variable (h_at_2 : f 2 = 0)

theorem range_of_f_lt_0 : ∀ x, x ∈ (Set.Ioo (-2) 0 ∪ Set.Ioi 2) → f x < 0 := by
  sorry

end range_of_f_lt_0_l348_34822


namespace volume_of_sphere_l348_34827

theorem volume_of_sphere
  (a b c : ℝ)
  (h1 : a * b * c = 4 * Real.sqrt 6)
  (h2 : a * b = 2 * Real.sqrt 3)
  (h3 : b * c = 4 * Real.sqrt 3)
  (O_radius : ℝ := Real.sqrt (a^2 + b^2 + c^2) / 2) :
  4 / 3 * Real.pi * O_radius^3 = 32 * Real.pi / 3 := by
  sorry

end volume_of_sphere_l348_34827


namespace line_intersects_circle_chord_min_length_l348_34806

-- Define the circle C
def C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 25

-- Define the line L based on parameter m
def L (m x y : ℝ) : Prop := (2 * m + 1) * x + (m + 1) * y - 7 * m - 4 = 0

-- Prove that for any real number m, line L intersects circle C at two points.
theorem line_intersects_circle (m : ℝ) : 
  ∃ x y₁ y₂ : ℝ, y₁ ≠ y₂ ∧ C x y₁ ∧ C x y₂ ∧ L m x y₁ ∧ L m x y₂ :=
sorry

-- Prove the equation of line L in slope-intercept form when the chord cut by circle C has minimum length.
theorem chord_min_length : ∃ (m : ℝ), ∀ x y : ℝ, 
  L m x y ↔ y = 2 * x - 5 :=
sorry

end line_intersects_circle_chord_min_length_l348_34806


namespace range_of_a_for_inequality_l348_34884

theorem range_of_a_for_inequality (a : ℝ) : (∀ x : ℝ, x^2 + a * |x| + 1 ≥ 0) ↔ (a ≥ -2) :=
sorry

end range_of_a_for_inequality_l348_34884


namespace quadratic_min_value_l348_34825

theorem quadratic_min_value : ∀ x : ℝ, x^2 - 6 * x + 13 ≥ 4 := 
by 
  sorry

end quadratic_min_value_l348_34825


namespace minimum_value_of_a_plus_2b_l348_34883

theorem minimum_value_of_a_plus_2b 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (h : 2 * a + b = a * b - 1) 
  : a + 2 * b = 5 + 2 * Real.sqrt 6 :=
sorry

end minimum_value_of_a_plus_2b_l348_34883


namespace unique_triple_solution_l348_34807

theorem unique_triple_solution {x y z : ℤ} (hx : x > 1) (hy : y > 1) (hz : z > 1)
  (H1 : x ∣ y * z - 1) (H2 : y ∣ z * x - 1) (H3 : z ∣ x * y - 1) :
  (x, y, z) = (5, 3, 2) :=
sorry

end unique_triple_solution_l348_34807


namespace gcd_lcm_product_180_l348_34853

theorem gcd_lcm_product_180 (a b : ℕ) (g l : ℕ) (ha : a > 0) (hb : b > 0) (hg : g > 0) (hl : l > 0) 
  (h₁ : g = gcd a b) (h₂ : l = lcm a b) (h₃ : g * l = 180):
  ∃(n : ℕ), n = 8 :=
by
  sorry

end gcd_lcm_product_180_l348_34853


namespace find_b_l348_34805

theorem find_b (b c : ℝ) : 
  (-11 : ℝ) = (-1)^2 + (-1) * b + c ∧ 
  17 = 3^2 + 3 * b + c ∧ 
  6 = 2^2 + 2 * b + c → 
  b = 14 / 3 :=
by
  sorry

end find_b_l348_34805


namespace difference_in_biking_distance_l348_34838

def biking_rate_alberto : ℕ := 18  -- miles per hour
def biking_rate_bjorn : ℕ := 20    -- miles per hour

def start_time_alberto : ℕ := 9    -- a.m.
def start_time_bjorn : ℕ := 10     -- a.m.

def end_time : ℕ := 15            -- 3 p.m. in 24-hour format

def biking_duration_alberto : ℕ := end_time - start_time_alberto
def biking_duration_bjorn : ℕ := end_time - start_time_bjorn

def distance_alberto : ℕ := biking_rate_alberto * biking_duration_alberto
def distance_bjorn : ℕ := biking_rate_bjorn * biking_duration_bjorn

theorem difference_in_biking_distance : 
  (distance_alberto - distance_bjorn) = 8 := by
  sorry

end difference_in_biking_distance_l348_34838


namespace find_m_l348_34868

theorem find_m (x y m : ℤ) (h1 : 3 * x + 4 * y = 7) (h2 : 5 * x - 4 * y = m) (h3 : x + y = 0) : m = -63 := by
  sorry

end find_m_l348_34868


namespace sum_of_roots_of_y_squared_eq_36_l348_34891

theorem sum_of_roots_of_y_squared_eq_36 :
  (∀ y : ℝ, y^2 = 36 → y = 6 ∨ y = -6) → (6 + (-6) = 0) :=
by
  sorry

end sum_of_roots_of_y_squared_eq_36_l348_34891


namespace no_real_solutions_for_identical_lines_l348_34811

theorem no_real_solutions_for_identical_lines :
  ¬∃ (a d : ℝ), (∀ x y : ℝ, 5 * x + a * y + d = 0 ↔ 2 * d * x - 3 * y + 8 = 0) :=
by
  sorry

end no_real_solutions_for_identical_lines_l348_34811


namespace system_of_equations_solution_l348_34854

theorem system_of_equations_solution :
  ∃ x y z : ℝ, x + y = 1 ∧ y + z = 2 ∧ z + x = 3 ∧ x = 1 ∧ y = 0 ∧ z = 2 :=
by
  sorry

end system_of_equations_solution_l348_34854


namespace geometric_series_properties_l348_34867

theorem geometric_series_properties (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) :
  a 3 = 3 ∧ a 10 = 384 → 
  q = 2 ∧ 
  (∀ n, a n = (3 / 4) * 2 ^ (n - 1)) ∧ 
  (∀ n, S n = (3 / 4) * (2 ^ n - 1)) :=
by
  intro h
  -- Proofs will go here, if necessary.
  sorry

end geometric_series_properties_l348_34867


namespace probability_one_defective_l348_34855

def total_bulbs : ℕ := 20
def defective_bulbs : ℕ := 4
def non_defective_bulbs : ℕ := total_bulbs - defective_bulbs
def probability_non_defective_both : ℚ := (16 / 20) * (15 / 19)
def probability_at_least_one_defective : ℚ := 1 - probability_non_defective_both

theorem probability_one_defective :
  probability_at_least_one_defective = 7 / 19 :=
by
  sorry

end probability_one_defective_l348_34855


namespace correct_calculation_l348_34894

theorem correct_calculation (a b : ℝ) : 
  (¬ (2 * (a - 1) = 2 * a - 1)) ∧ 
  (3 * a^2 - 2 * a^2 = a^2) ∧ 
  (¬ (3 * a^2 - 2 * a^2 = 1)) ∧ 
  (¬ (3 * a + 2 * b = 5 * a * b)) :=
by
  sorry

end correct_calculation_l348_34894


namespace black_lambs_correct_l348_34821

-- Define the total number of lambs
def total_lambs : ℕ := 6048

-- Define the number of white lambs
def white_lambs : ℕ := 193

-- Define the number of black lambs
def black_lambs : ℕ := total_lambs - white_lambs

-- The goal is to prove that the number of black lambs is 5855
theorem black_lambs_correct : black_lambs = 5855 := by
  sorry

end black_lambs_correct_l348_34821


namespace family_members_l348_34875

variable (p : ℝ) (i : ℝ) (c : ℝ)

theorem family_members (h1 : p = 1.6) (h2 : i = 0.25) (h3 : c = 16) :
  (c / (2 * (p * (1 + i)))) = 4 := by
  sorry

end family_members_l348_34875


namespace curves_tangent_at_m_eq_two_l348_34804

-- Definitions of the ellipsoid and hyperbola equations.
def ellipse (x y : ℝ) : Prop := x^2 + 2 * y^2 = 2
def hyperbola (x y m : ℝ) : Prop := x^2 - m * (y + 1)^2 = 1

-- The proposition to be proved.
theorem curves_tangent_at_m_eq_two :
  ∃ m : ℝ, (∀ x y : ℝ, ellipse x y ∧ hyperbola x y m → m = 2) :=
sorry

end curves_tangent_at_m_eq_two_l348_34804


namespace commute_time_difference_l348_34869

theorem commute_time_difference (x y : ℝ) 
  (h1 : x + y = 39)
  (h2 : (x - 10)^2 + (y - 10)^2 = 10) :
  |x - y| = 4 :=
by
  sorry

end commute_time_difference_l348_34869


namespace compute_expression_l348_34860

theorem compute_expression : 1 + 6 * 2 - 3 + 5 * 4 / 2 = 20 :=
by sorry

end compute_expression_l348_34860


namespace minimum_cuts_for_10_pieces_l348_34890

theorem minimum_cuts_for_10_pieces :
  ∃ n : ℕ, (n * (n + 1)) / 2 ≥ 10 ∧ ∀ m < n, (m * (m + 1)) / 2 < 10 := sorry

end minimum_cuts_for_10_pieces_l348_34890


namespace odd_function_alpha_l348_34859
open Real

noncomputable def f (x : ℝ) : ℝ :=
  cos x * (sin x + sqrt 3 * cos x) - sqrt 3 / 2

noncomputable def g (x : ℝ) (α : ℝ) : ℝ :=
  f (x + α)

theorem odd_function_alpha (α : ℝ) (a : α > 0) :
  (∀ x : ℝ, g x α = - g (-x) α) ↔ 
  ∃ k : ℕ, α = (2 * k - 1) * π / 6 := sorry

end odd_function_alpha_l348_34859


namespace solve_for_y_l348_34819

theorem solve_for_y (x y : ℝ) (hx : x > 1) (hy : y > 1) (h1 : 1 / x + 1 / y = 1) (h2 : x * y = 9) :
  y = (9 + 3 * Real.sqrt 5) / 2 :=
by
  sorry

end solve_for_y_l348_34819


namespace add_fractions_l348_34864

theorem add_fractions (x : ℝ) (h : x ≠ 1) : (1 / (x - 1) + 3 / (x - 1)) = (4 / (x - 1)) :=
by
  sorry

end add_fractions_l348_34864


namespace tom_age_ratio_l348_34886

-- Define the variables and conditions
variables (T N : ℕ)

-- Condition 1: Tom's current age is twice the sum of his children's ages
def children_sum_current : ℤ := T / 2

-- Condition 2: Tom's age N years ago was three times the sum of their ages then
def children_sum_past : ℤ := (T / 2) - 2 * N

-- Main theorem statement proving the ratio T/N = 10 assuming given conditions
theorem tom_age_ratio (h1 : T = 2 * (T / 2)) 
                      (h2 : T - N = 3 * ((T / 2) - 2 * N)) : 
                      T / N = 10 :=
sorry

end tom_age_ratio_l348_34886


namespace cardinals_home_runs_second_l348_34851

-- Define the conditions
def cubs_home_runs_third : ℕ := 2
def cubs_home_runs_fifth : ℕ := 1
def cubs_home_runs_eighth : ℕ := 2
def cubs_total_home_runs := cubs_home_runs_third + cubs_home_runs_fifth + cubs_home_runs_eighth
def cubs_more_than_cardinals : ℕ := 3
def cardinals_home_runs_fifth : ℕ := 1

-- Define the proof problem
theorem cardinals_home_runs_second :
  (cubs_total_home_runs = cardinals_total_home_runs + cubs_more_than_cardinals) →
  (cardinals_total_home_runs - cardinals_home_runs_fifth = 1) :=
sorry

end cardinals_home_runs_second_l348_34851


namespace average_speed_before_increase_l348_34814

-- Definitions for the conditions
def t_before := 12   -- Travel time before the speed increase in hours
def t_after := 10    -- Travel time after the speed increase in hours
def speed_diff := 20 -- Speed difference between before and after in km/h

-- Variable for the speed before increase
variable (s_before : ℕ) -- Average speed before the speed increase in km/h

-- Definitions for the speeds
def s_after := s_before + speed_diff -- Average speed after the speed increase in km/h

-- Equations derived from the problem conditions
def dist_eqn_before := s_before * t_before
def dist_eqn_after := s_after * t_after

-- The proof problem stated in Lean
theorem average_speed_before_increase : dist_eqn_before = dist_eqn_after → s_before = 100 := by
  sorry

end average_speed_before_increase_l348_34814


namespace james_used_5_containers_l348_34800

-- Conditions
def initial_balls : ℕ := 100
def balls_given_away : ℕ := initial_balls / 2
def remaining_balls : ℕ := initial_balls - balls_given_away
def balls_per_container : ℕ := 10

-- Question (statement of the theorem to prove)
theorem james_used_5_containers : (remaining_balls / balls_per_container) = 5 := by
  sorry

end james_used_5_containers_l348_34800


namespace complement_of_M_l348_34841

def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x^2 - x > 0}

theorem complement_of_M :
  (U \ M) = {x | 0 ≤ x ∧ x ≤ 1} :=
sorry

end complement_of_M_l348_34841


namespace calculate_length_X_l348_34808

theorem calculate_length_X 
  (X : ℝ)
  (h1 : 3 + X + 4 = 5 + 7 + X)
  : X = 5 :=
sorry

end calculate_length_X_l348_34808


namespace company_blocks_l348_34885

noncomputable def number_of_blocks (workers_per_block total_budget gift_cost : ℕ) : ℕ :=
  (total_budget / gift_cost) / workers_per_block

theorem company_blocks :
  number_of_blocks 200 6000 2 = 15 :=
by
  sorry

end company_blocks_l348_34885


namespace sin_B_plus_pi_over_6_eq_l348_34898

noncomputable def sin_b_plus_pi_over_6 (B : ℝ) : ℝ :=
  Real.sin B * (Real.sqrt 3 / 2) + (Real.sqrt (1 - (Real.sin B) ^ 2)) * (1 / 2)

theorem sin_B_plus_pi_over_6_eq :
  ∀ (A B : ℝ) (b c : ℝ),
    A = (2 * Real.pi / 3) →
    b = 1 →
    (1 / 2 * b * c * Real.sin A) = Real.sqrt 3 →
    c = 2 →
    sin_b_plus_pi_over_6 B = (2 * Real.sqrt 7 / 7) :=
by
  intros A B b c hA hb hArea hc
  sorry

end sin_B_plus_pi_over_6_eq_l348_34898


namespace trigonometric_inequality_l348_34877

open Real

theorem trigonometric_inequality
  (x y z : ℝ)
  (h1 : 0 < x)
  (h2 : x < y)
  (h3 : y < z)
  (h4 : z < π / 2) :
  π / 2 + 2 * sin x * cos y + 2 * sin y * cos z > sin (2 * x) + sin (2 * y) + sin (2 * z) :=
by
  sorry

end trigonometric_inequality_l348_34877


namespace geometric_sequence_collinear_vectors_l348_34824

theorem geometric_sequence_collinear_vectors (a : ℕ → ℝ) (q : ℝ)
  (h_geometric : ∀ n, a (n + 1) = q * a n)
  (a2 a3 : ℝ)
  (h_a2 : a 2 = a2)
  (h_a3 : a 3 = a3)
  (h_parallel : 3 * a2 = 2 * a3) :
  (a2 + a 4) / (a3 + a 5) = 2 / 3 := 
by
  sorry

end geometric_sequence_collinear_vectors_l348_34824


namespace proof_problem_l348_34888

variables (x y b z a : ℝ)

def condition1 : Prop := x * y + x^2 = b
def condition2 : Prop := (1 / x^2) - (1 / y^2) = a
def z_def : Prop := z = x + y

theorem proof_problem (x y b z a : ℝ) (h1 : condition1 x y b) (h2 : condition2 x y a) (hz : z_def x y z) : (x + y) ^ 2 = z ^ 2 :=
by {
  sorry
}

end proof_problem_l348_34888


namespace repeating_decimal_calculation_l348_34880

theorem repeating_decimal_calculation :
  2 * (8 / 9 - 2 / 9 + 4 / 9) = 20 / 9 :=
by
  -- sorry proof will be inserted here.
  sorry

end repeating_decimal_calculation_l348_34880


namespace solve_for_x_l348_34836

noncomputable def solution_x : ℝ := -1011.5

theorem solve_for_x (x : ℝ) (h : (2023 + x)^2 = x^2) : x = solution_x :=
by sorry

end solve_for_x_l348_34836


namespace arithmetic_sequence_problem_l348_34843

theorem arithmetic_sequence_problem
  (a : ℕ → ℤ)  -- the arithmetic sequence
  (S : ℕ → ℤ)  -- the sum of the first n terms
  (m : ℕ)      -- the m in question
  (h1 : a (m - 1) + a (m + 1) - a m ^ 2 = 0)
  (h2 : S (2 * m - 1) = 18) :
  m = 5 := 
sorry

end arithmetic_sequence_problem_l348_34843


namespace regular_hexagon_area_l348_34899

noncomputable def dist (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem regular_hexagon_area 
  (A C : ℝ × ℝ)
  (hA : A = (0, 0))
  (hC : C = (8, 2))
  (h_eq_side_length : ∀ x y : ℝ × ℝ, dist A.1 A.2 C.1 C.2 = dist x.1 x.2 y.1 y.2) :
  hexagon_area = 34 * Real.sqrt 3 :=
by
  -- sorry indicates the proof is omitted
  sorry

end regular_hexagon_area_l348_34899


namespace greatest_number_of_sets_l348_34889

-- Definitions based on conditions
def whitney_tshirts := 5
def whitney_buttons := 24
def whitney_stickers := 12
def buttons_per_set := 2
def stickers_per_set := 1

-- The statement to prove the greatest number of identical sets Whitney can make
theorem greatest_number_of_sets : 
  ∃ max_sets : ℕ, 
  max_sets = whitney_tshirts ∧ 
  max_sets ≤ (whitney_buttons / buttons_per_set) ∧
  max_sets ≤ (whitney_stickers / stickers_per_set) :=
sorry

end greatest_number_of_sets_l348_34889


namespace number_of_shampoos_l348_34823

-- Define necessary variables in conditions
def h := 10 -- time spent hosing in minutes
def t := 55 -- total time spent cleaning in minutes
def p := 15 -- time per shampoo in minutes

-- State the theorem
theorem number_of_shampoos (h t p : Nat) (h_val : h = 10) (t_val : t = 55) (p_val : p = 15) :
    (t - h) / p = 3 := by
  -- Proof to be filled in
  sorry

end number_of_shampoos_l348_34823


namespace triangle_angle_A_triangle_length_b_l348_34897

variable (A B C : ℝ)
variable (a b c : ℝ)
variable (m n : ℝ × ℝ)
variable (S : ℝ)

theorem triangle_angle_A (h1 : a = 7) (h2 : c = 8) (h3 : m = (1, 7 * a)) (h4 : n = (-4 * a, Real.sin C))
  (h5 : m.1 * n.1 + m.2 * n.2 = 0) : 
  A = Real.pi / 6 := 
  sorry

theorem triangle_length_b (h1 : a = 7) (h2 : c = 8) (h3 : (7 * 8 * Real.sin B) / 2 = 16 * Real.sqrt 3) :
  b = Real.sqrt 97 :=
  sorry

end triangle_angle_A_triangle_length_b_l348_34897


namespace math_problem_l348_34812

theorem math_problem (a b c d m : ℝ) (h1 : a = -b) (h2 : a ≠ 0) (h3 : c * d = 1)
  (h4 : m = -1 ∨ m = 3) : (a + b) * (c / d) + m * c * d + (b / a) = 2 ∨ (a + b) * (c / d) + m * c * d + (b / a) = -2 :=
by
  sorry

end math_problem_l348_34812


namespace total_candy_pieces_l348_34892

theorem total_candy_pieces : 
  (brother_candy = 6) → 
  (wendy_boxes = 2) → 
  (pieces_per_box = 3) → 
  (brother_candy + (wendy_boxes * pieces_per_box) = 12) 
  := 
  by 
    intros brother_candy wendy_boxes pieces_per_box 
    sorry

end total_candy_pieces_l348_34892


namespace investment_ratio_l348_34893

theorem investment_ratio (X_investment Y_investment : ℕ) (hX : X_investment = 5000) (hY : Y_investment = 15000) : 
  X_investment * 3 = Y_investment :=
by
  sorry

end investment_ratio_l348_34893


namespace tangent_line_eqn_when_a_zero_min_value_f_when_a_zero_range_of_a_for_x_ge_zero_exp_x_ln_x_plus_one_gt_x_sq_l348_34896

noncomputable def f (a : ℝ) (x : ℝ) := Real.exp x - 1 - x - a * x ^ 2

theorem tangent_line_eqn_when_a_zero :
  (∀ x, y = f 0 x → y - (Real.exp 1 - 2) = (Real.exp 1 - 1) * (x - 1)) :=
sorry

theorem min_value_f_when_a_zero :
  (∀ x : ℝ, f 0 x >= f 0 0) := 
sorry

theorem range_of_a_for_x_ge_zero (a : ℝ) :
  (∀ x : ℝ, 0 ≤ x → f a x ≥ 0) → (a ≤ 1/2) :=
sorry

theorem exp_x_ln_x_plus_one_gt_x_sq (x : ℝ) :
  x > 0 → ((Real.exp x - 1) * Real.log (x + 1) > x ^ 2) :=
sorry

end tangent_line_eqn_when_a_zero_min_value_f_when_a_zero_range_of_a_for_x_ge_zero_exp_x_ln_x_plus_one_gt_x_sq_l348_34896


namespace find_a_10_l348_34862

-- Definitions and conditions from the problem
def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

def S (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  (n * (a 1 + a n)) / 2

variable (a : ℕ → ℕ)

-- Conditions given
axiom a_3 : a 3 = 3
axiom S_3 : S a 3 = 6
axiom arithmetic_seq : is_arithmetic_sequence a

-- Proof problem statement
theorem find_a_10 : a 10 = 10 := 
sorry

end find_a_10_l348_34862


namespace average_speed_l348_34820

def dist1 : ℝ := 60
def dist2 : ℝ := 30
def time : ℝ := 2

theorem average_speed : (dist1 + dist2) / time = 45 := by
  sorry

end average_speed_l348_34820


namespace knife_value_l348_34815

def sheep_sold (n : ℕ) : ℕ := n * n

def valid_units_digits (m : ℕ) : Bool :=
  (m ^ 2 = 16) ∨ (m ^ 2 = 36)

theorem knife_value (n : ℕ) (k : ℕ) (m : ℕ) (H1 : sheep_sold n = n * n) (H2 : n = 10 * k + m) (H3 : valid_units_digits m = true) :
  2 = 2 :=
by
  sorry

end knife_value_l348_34815


namespace evaluate_powers_of_i_l348_34873

noncomputable def imag_unit := Complex.I

theorem evaluate_powers_of_i :
  (imag_unit^11 + imag_unit^16 + imag_unit^21 + imag_unit^26 + imag_unit^31) = -imag_unit :=
by
  sorry

end evaluate_powers_of_i_l348_34873
