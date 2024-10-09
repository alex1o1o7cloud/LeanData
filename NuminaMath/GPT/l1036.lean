import Mathlib

namespace reporters_cover_local_politics_l1036_103694

-- Definitions of percentages and total reporters
def total_reporters : ℕ := 100
def politics_coverage_percent : ℕ := 20 -- Derived from 100 - 80
def politics_reporters : ℕ := (politics_coverage_percent * total_reporters) / 100
def not_local_politics_percent : ℕ := 40
def local_politics_percent : ℕ := 60 -- Derived from 100 - 40
def local_politics_reporters : ℕ := (local_politics_percent * politics_reporters) / 100

theorem reporters_cover_local_politics :
  (local_politics_reporters * 100) / total_reporters = 12 :=
by
  exact sorry

end reporters_cover_local_politics_l1036_103694


namespace profit_per_meter_is_15_l1036_103655

def sellingPrice (meters : ℕ) : ℕ := 
    if meters = 85 then 8500 else 0

def costPricePerMeter : ℕ := 85

def totalCostPrice (meters : ℕ) : ℕ := 
    meters * costPricePerMeter

def totalProfit (meters : ℕ) (sellingPrice : ℕ) (costPrice : ℕ) : ℕ := 
    sellingPrice - costPrice

def profitPerMeter (profit : ℕ) (meters : ℕ) : ℕ := 
    profit / meters

theorem profit_per_meter_is_15 : profitPerMeter (totalProfit 85 (sellingPrice 85) (totalCostPrice 85)) 85 = 15 := 
by sorry

end profit_per_meter_is_15_l1036_103655


namespace fencing_required_l1036_103637

theorem fencing_required (L W : ℝ) (hL : L = 20) (hArea : L * W = 60) : (L + 2 * W) = 26 := 
by
  sorry

end fencing_required_l1036_103637


namespace f_periodic_with_period_4a_l1036_103685

-- Definitions 'f' and 'g' (functions on real numbers), and the given conditions:
variables {a : ℝ} (f g : ℝ → ℝ)
-- Condition on a: a ≠ 0
variable (ha : a ≠ 0)

-- Given conditions
variable (hf0 : f 0 = 1) (hga : g a = 1) (h_odd_g : ∀ x : ℝ, g x = -g (-x))

-- Functional equation
variable (h_func_eq : ∀ x y : ℝ, f (x - y) = f x * f y + g x * g y)

-- The theorem stating that f is periodic with period 4a
theorem f_periodic_with_period_4a : ∀ x : ℝ, f (x + 4 * a) = f x :=
by
  sorry

end f_periodic_with_period_4a_l1036_103685


namespace simplify_and_evaluate_l1036_103607

variable (a : ℝ)
variable (b : ℝ)

theorem simplify_and_evaluate (h : b = -1/3) : (a + b)^2 - a * (2 * b + a) = 1/9 :=
by
  rw [h]
  sorry

end simplify_and_evaluate_l1036_103607


namespace local_minimum_at_two_l1036_103625

def f (x : ℝ) : ℝ := x^3 - 12 * x

theorem local_minimum_at_two : ∃ a : ℝ, a = 2 ∧ ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, (0 < |x - a| ∧ |x - a| < δ) → f x > f a :=
by sorry

end local_minimum_at_two_l1036_103625


namespace find_series_sum_l1036_103638

noncomputable def series_sum (s : ℝ) : ℝ := ∑' n : ℕ, (n+1) * s^(4*n + 3)

theorem find_series_sum (s : ℝ) (h : s^4 - s - 1/2 = 0) : series_sum s = -4 := by
  sorry

end find_series_sum_l1036_103638


namespace speed_on_way_home_l1036_103634

theorem speed_on_way_home (d : ℝ) (v_up : ℝ) (v_avg : ℝ) (v_home : ℝ) 
  (h1 : v_up = 110) 
  (h2 : v_avg = 91)
  (h3 : 91 = (2 * d) / (d / 110 + d / v_home)) : 
  v_home = 10010 / 129 := 
sorry

end speed_on_way_home_l1036_103634


namespace factorization_correct_l1036_103629

theorem factorization_correct (c : ℝ) : (x : ℝ) → x^2 - x + c = (x + 2) * (x - 3) → c = -6 := by
  intro x h
  sorry

end factorization_correct_l1036_103629


namespace circle_equation_from_parabola_l1036_103642

theorem circle_equation_from_parabola :
  let F := (2, 0)
  let A := (2, 4)
  let B := (2, -4)
  let diameter := 8
  let center := F
  let radius_squared := diameter^2 / 4
  (x - center.1)^2 + y^2 = radius_squared :=
by
  sorry

end circle_equation_from_parabola_l1036_103642


namespace perpendicular_lines_a_l1036_103692

theorem perpendicular_lines_a (a : ℝ) :
  (∀ x y : ℝ, (a * x + (1 + a) * y = 3) ∧ ((a + 1) * x + (3 - 2 * a) * y = 2) → 
     a = -1 ∨ a = 3) :=
by
  sorry

end perpendicular_lines_a_l1036_103692


namespace azalea_profit_l1036_103679

def num_sheep : Nat := 200
def wool_per_sheep : Nat := 10
def price_per_pound : Nat := 20
def shearer_cost : Nat := 2000

theorem azalea_profit : 
  (num_sheep * wool_per_sheep * price_per_pound) - shearer_cost = 38000 := 
by
  sorry

end azalea_profit_l1036_103679


namespace solve_linear_eq_l1036_103664

theorem solve_linear_eq : (∃ x : ℝ, 2 * x - 1 = 0) ↔ (∃ x : ℝ, x = 1/2) :=
by
  sorry

end solve_linear_eq_l1036_103664


namespace max_positive_root_eq_l1036_103676

theorem max_positive_root_eq (b c : ℝ) (h_b : |b| ≤ 3) (h_c : |c| ≤ 3) : 
  ∃ x, x = (3 + Real.sqrt 21) / 2 ∧ x^2 + b * x + c = 0 ∧ x ≥ 0 :=
by
  sorry

end max_positive_root_eq_l1036_103676


namespace time_difference_for_x_miles_l1036_103658

def time_old_shoes (n : Nat) : Int := 10 * n
def time_new_shoes (n : Nat) : Int := 13 * n
def time_difference_for_5_miles : Int := time_new_shoes 5 - time_old_shoes 5

theorem time_difference_for_x_miles (x : Nat) (h : time_difference_for_5_miles = 15) : 
  time_new_shoes x - time_old_shoes x = 3 * x := 
by
  sorry

end time_difference_for_x_miles_l1036_103658


namespace system_solution_l1036_103613

theorem system_solution (x y z : ℚ) 
  (h1 : x + y + x * y = 19) 
  (h2 : y + z + y * z = 11) 
  (h3 : z + x + z * x = 14) :
    (x = 4 ∧ y = 3 ∧ z = 2) ∨ (x = -6 ∧ y = -5 ∧ z = -4) :=
by
  sorry

end system_solution_l1036_103613


namespace compare_abc_l1036_103687

theorem compare_abc (a b c : Real) (h1 : a = Real.sqrt 3) (h2 : b = Real.log 2) (h3 : c = Real.logb 3 (Real.sin (Real.pi / 6))) :
  a > b ∧ b > c :=
by
  sorry

end compare_abc_l1036_103687


namespace actual_plot_area_l1036_103635

noncomputable def area_of_triangle_in_acres : Real :=
  let base_cm : Real := 8
  let height_cm : Real := 5
  let area_cm2 : Real := 0.5 * base_cm * height_cm
  let conversion_factor_cm2_to_km2 : Real := 25
  let area_km2 : Real := area_cm2 * conversion_factor_cm2_to_km2
  let conversion_factor_km2_to_acres : Real := 247.1
  area_km2 * conversion_factor_km2_to_acres

theorem actual_plot_area :
  area_of_triangle_in_acres = 123550 :=
by
  sorry

end actual_plot_area_l1036_103635


namespace prob_exactly_one_hits_is_one_half_prob_at_least_one_hits_is_two_thirds_l1036_103630

def person_A_hits : ℚ := 1 / 2
def person_B_hits : ℚ := 1 / 3

def person_A_misses : ℚ := 1 - person_A_hits
def person_B_misses : ℚ := 1 - person_B_hits

def exactly_one_hits : ℚ := (person_A_hits * person_B_misses) + (person_B_hits * person_A_misses)
def at_least_one_hits : ℚ := 1 - (person_A_misses * person_B_misses)

theorem prob_exactly_one_hits_is_one_half : exactly_one_hits = 1 / 2 := sorry

theorem prob_at_least_one_hits_is_two_thirds : at_least_one_hits = 2 / 3 := sorry

end prob_exactly_one_hits_is_one_half_prob_at_least_one_hits_is_two_thirds_l1036_103630


namespace translation_up_by_one_l1036_103681

def initial_function (x : ℝ) : ℝ := x^2

def translated_function (x : ℝ) : ℝ := x^2 + 1

theorem translation_up_by_one (x : ℝ) : translated_function x = initial_function x + 1 :=
by sorry

end translation_up_by_one_l1036_103681


namespace total_paper_clips_l1036_103650

/-
Given:
- The number of cartons: c = 3
- The number of boxes: b = 4
- The number of bags: p = 2
- The number of paper clips in each carton: paper_clips_per_carton = 300
- The number of paper clips in each box: paper_clips_per_box = 550
- The number of paper clips in each bag: paper_clips_per_bag = 1200

Prove that the total number of paper clips is 5500.
-/

theorem total_paper_clips :
  let c := 3
  let paper_clips_per_carton := 300
  let b := 4
  let paper_clips_per_box := 550
  let p := 2
  let paper_clips_per_bag := 1200
  (c * paper_clips_per_carton + b * paper_clips_per_box + p * paper_clips_per_bag) = 5500 :=
by
  sorry

end total_paper_clips_l1036_103650


namespace number_of_dogs_per_box_l1036_103643

-- Definition of the problem
def num_boxes : ℕ := 7
def total_dogs : ℕ := 28

-- Statement of the theorem to prove
theorem number_of_dogs_per_box (x : ℕ) (h : num_boxes * x = total_dogs) : x = 4 :=
by
  sorry

end number_of_dogs_per_box_l1036_103643


namespace A_work_days_l1036_103615

theorem A_work_days (x : ℝ) (h1 : 1 / 15 + 1 / x = 1 / 8.571428571428571) : x = 20 :=
by
  sorry

end A_work_days_l1036_103615


namespace enumerate_A_l1036_103698

-- Define the set A according to the given conditions
def A : Set ℕ := {X : ℕ | 8 % (6 - X) = 0}

-- The equivalent proof problem
theorem enumerate_A : A = {2, 4, 5} :=
by sorry

end enumerate_A_l1036_103698


namespace Moe_has_least_amount_of_money_l1036_103673

variables {B C F J M Z : ℕ}

theorem Moe_has_least_amount_of_money
  (h1 : Z > F) (h2 : F > B) (h3 : Z > C) (h4 : B > M) (h5 : C > M) (h6 : Z > J) (h7 : J > M) :
  ∀ x, x ≠ M → x > M :=
by {
  sorry
}

end Moe_has_least_amount_of_money_l1036_103673


namespace derivative_y_at_1_l1036_103696

-- Define the function y = x^2 + 2
def f (x : ℝ) : ℝ := x^2 + 2

-- Define the proposition that the derivative at x=1 is 2
theorem derivative_y_at_1 : deriv f 1 = 2 :=
by sorry

end derivative_y_at_1_l1036_103696


namespace triangular_region_area_l1036_103609

noncomputable def area_of_triangle (f g h : ℝ → ℝ) : ℝ :=
  let (x1, y1) := (-3, f (-3))
  let (x2, y2) := (7/3, g (7/3))
  let (x3, y3) := (15/11, f (15/11))
  let base := abs (x2 - x1)
  let height := abs (y3 - 2)
  (1/2) * base * height

theorem triangular_region_area :
  let f x := (2/3) * x + 4
  let g x := -3 * x + 9
  let h x := (2 : ℝ)
  area_of_triangle f g h = 256/33 :=  -- Given conditions
by
  sorry  -- Proof to be supplied

end triangular_region_area_l1036_103609


namespace michael_students_l1036_103610

theorem michael_students (M N : ℕ) (h1 : M = 5 * N) (h2 : M + N + 300 = 3500) : M = 2667 := 
by 
  -- This to be filled later
  sorry

end michael_students_l1036_103610


namespace theater_total_cost_l1036_103678

theorem theater_total_cost 
  (cost_orchestra : ℕ) (cost_balcony : ℕ)
  (total_tickets : ℕ) (ticket_difference : ℕ)
  (O B : ℕ)
  (h1 : cost_orchestra = 12)
  (h2 : cost_balcony = 8)
  (h3 : total_tickets = 360)
  (h4 : ticket_difference = 140)
  (h5 : O + B = total_tickets)
  (h6 : B = O + ticket_difference) :
  12 * O + 8 * B = 3320 :=
by
  sorry

end theater_total_cost_l1036_103678


namespace total_games_played_l1036_103602

-- Define the function for combinations
def combination (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

-- Given conditions
def teams : ℕ := 20
def games_per_pair : ℕ := 10

-- Proposition stating the target result
theorem total_games_played : 
  (combination teams 2 * games_per_pair) = 1900 :=
by
  sorry

end total_games_played_l1036_103602


namespace pats_stick_length_correct_l1036_103618

noncomputable def jane_stick_length : ℕ := 22
noncomputable def sarah_stick_length : ℕ := jane_stick_length + 24
noncomputable def uncovered_pats_stick : ℕ := sarah_stick_length / 2
noncomputable def covered_pats_stick : ℕ := 7
noncomputable def total_pats_stick : ℕ := uncovered_pats_stick + covered_pats_stick

theorem pats_stick_length_correct : total_pats_stick = 30 := by
  sorry

end pats_stick_length_correct_l1036_103618


namespace find_multiplier_l1036_103652

variable {a b : ℝ} 

theorem find_multiplier (h1 : 3 * a = x * b) (h2 : a ≠ 0 ∧ b ≠ 0) (h3 : a / 4 = b / 3) : x = 4 :=
by
  sorry

end find_multiplier_l1036_103652


namespace twice_not_square_l1036_103614

theorem twice_not_square (m : ℝ) : 2 * m ≠ m * m := by
  sorry

end twice_not_square_l1036_103614


namespace total_students_high_school_l1036_103691

theorem total_students_high_school (students_first_grade : ℕ) (total_sample : ℕ) 
  (sample_second_grade : ℕ) (sample_third_grade : ℕ) (total_students : ℕ) 
  (h1 : students_first_grade = 600) (h2 : total_sample = 45) 
  (h3 : sample_second_grade = 20) (h4 : sample_third_grade = 10)
  (h5 : 15 = total_sample - sample_second_grade - sample_third_grade) 
  (h6 : 15 * total_students = students_first_grade * total_sample) :
  total_students = 1800 :=
sorry

end total_students_high_school_l1036_103691


namespace product_of_x_y_l1036_103626

theorem product_of_x_y (x y : ℝ) (h1 : -3 * x + 4 * y = 28) (h2 : 3 * x - 2 * y = 8) : x * y = 264 :=
by
  sorry

end product_of_x_y_l1036_103626


namespace find_divisor_l1036_103623

theorem find_divisor (x : ℝ) (h : 1152 / x - 189 = 3) : x = 6 :=
by
  sorry

end find_divisor_l1036_103623


namespace student_count_incorrect_l1036_103606

theorem student_count_incorrect :
  ∀ k : ℕ, 2012 ≠ 18 + 17 * k :=
by
  intro k
  sorry

end student_count_incorrect_l1036_103606


namespace custom_op_3_7_l1036_103605

-- Define the custom operation (a # b)
def custom_op (a b : ℕ) : ℕ := a * b - b + b^2

-- State the theorem that proves the result
theorem custom_op_3_7 : custom_op 3 7 = 63 := by
  sorry

end custom_op_3_7_l1036_103605


namespace arithmetic_sequence_general_term_l1036_103611

theorem arithmetic_sequence_general_term
    (a : ℕ → ℤ)
    (h_arith_seq : ∀ n, a (n + 1) - a n = a 1 - a 0)
    (h_mean_26 : (a 2 + a 6) / 2 = 5)
    (h_mean_37 : (a 3 + a 7) / 2 = 7) :
    ∀ n, a n = 2 * n - 3 := 
by
  sorry

end arithmetic_sequence_general_term_l1036_103611


namespace range_of_m_exists_l1036_103683

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 3 * x

-- Proof problem statement
theorem range_of_m_exists (m : ℝ) (h : m ∈ Set.Icc (-2 : ℝ) (0 : ℝ)) : 
  ∃ x ∈ Set.Icc (0 : ℝ) (1 : ℝ), f x = m := 
by
  sorry

end range_of_m_exists_l1036_103683


namespace smallest_k_for_inequality_l1036_103636

theorem smallest_k_for_inequality :
  ∃ k : ℕ, (∀ m : ℕ, m < k → 64^m ≤ 7) ∧ 64^k > 7 :=
by
  sorry

end smallest_k_for_inequality_l1036_103636


namespace three_digit_multiple_l1036_103684

open Classical

theorem three_digit_multiple (n : ℕ) (h₁ : n % 2 = 0) (h₂ : n % 5 = 0) (h₃ : n % 3 = 0) (h₄ : 100 ≤ n) (h₅ : n < 1000) :
  120 ≤ n ∧ n ≤ 990 :=
by
  sorry

end three_digit_multiple_l1036_103684


namespace raccoon_hid_nuts_l1036_103633

theorem raccoon_hid_nuts :
  ∃ (r p : ℕ), r + p = 25 ∧ (p = r - 3) ∧ 5 * r = 6 * p ∧ 5 * r = 70 :=
by
  sorry

end raccoon_hid_nuts_l1036_103633


namespace ratio_of_y_and_z_l1036_103699

variable (x y z : ℝ)

theorem ratio_of_y_and_z (h1 : x + y = 2 * x + z) (h2 : x - 2 * y = 4 * z) (h3 : x + y + z = 21) : y / z = -5 := 
by 
  sorry

end ratio_of_y_and_z_l1036_103699


namespace find_a6_l1036_103641

variable {a : ℕ → ℝ}
variable {q : ℝ}
variable {a₁ : ℝ}

/-- The sequence is a geometric sequence -/
axiom geom_seq (n : ℕ) : a n = a₁ * q ^ (n - 1)

/-- The sum of the first three terms is 168 -/
axiom sum_of_first_three_terms : a₁ + a₁ * q + a₁ * q ^ 2 = 168

/-- The difference between the 2nd and the 5th terms is 42 -/
axiom difference_a2_a5 : a₁ * q - a₁ * q ^ 4 = 42

theorem find_a6 : a 6 = 3 :=
by
  -- Proof goes here
  sorry

end find_a6_l1036_103641


namespace decreasing_interval_0_pi_over_4_l1036_103648

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := Real.cos (x + φ)

theorem decreasing_interval_0_pi_over_4 (φ : ℝ) (hφ1 : 0 < |φ| ∧ |φ| < Real.pi / 2)
  (hodd : ∀ x : ℝ, f (x + Real.pi / 4) φ = -f (-x + Real.pi / 4) φ) :
  ∀ x : ℝ, 0 < x ∧ x < Real.pi / 4 → f x φ > f (x + 1e-6) φ :=
by sorry

end decreasing_interval_0_pi_over_4_l1036_103648


namespace Alejandra_overall_score_l1036_103640

theorem Alejandra_overall_score :
  let score1 := (60/100 : ℝ) * 20
  let score2 := (75/100 : ℝ) * 30
  let score3 := (85/100 : ℝ) * 40
  let total_score := score1 + score2 + score3
  let total_questions := 90
  let overall_percentage := (total_score / total_questions) * 100
  round overall_percentage = 77 :=
by
  sorry

end Alejandra_overall_score_l1036_103640


namespace bus_arrival_time_at_first_station_l1036_103639

noncomputable def time_to_first_station (start_time end_time first_station_to_work: ℕ) : ℕ :=
  (end_time - start_time) - first_station_to_work

theorem bus_arrival_time_at_first_station :
  time_to_first_station 360 540 140 = 40 :=
by
  -- provide the proof here, which has been omitted per the instructions
  sorry

end bus_arrival_time_at_first_station_l1036_103639


namespace percentage_of_water_in_juice_l1036_103649

-- Define the initial condition for tomato puree water percentage
def puree_water_percentage : ℝ := 0.20

-- Define the volume of tomato puree produced from tomato juice
def volume_puree : ℝ := 3.75

-- Define the volume of tomato juice used to produce the puree
def volume_juice : ℝ := 30

-- Given conditions and definitions, prove the percentage of water in tomato juice
theorem percentage_of_water_in_juice :
  ((volume_juice - (volume_puree - puree_water_percentage * volume_puree)) / volume_juice) * 100 = 90 :=
by sorry

end percentage_of_water_in_juice_l1036_103649


namespace ratio_of_apples_l1036_103680

/-- The store sold 32 red apples and the combined amount of red and green apples sold was 44. -/
theorem ratio_of_apples (R G : ℕ) (h1 : R = 32) (h2 : R + G = 44) : R / 4 = 8 ∧ G / 4 = 3 :=
by {
  -- Placeholder for the proof
  sorry
}

end ratio_of_apples_l1036_103680


namespace triangle_angle_inequality_l1036_103672

open Real

theorem triangle_angle_inequality (α β γ α₁ β₁ γ₁ : ℝ) 
  (h1 : α + β + γ = π)
  (h2 : α₁ + β₁ + γ₁ = π) :
  (cos α₁ / sin α) + (cos β₁ / sin β) + (cos γ₁ / sin γ) 
  ≤ (cos α / sin α) + (cos β / sin β) + (cos γ / sin γ) :=
sorry

end triangle_angle_inequality_l1036_103672


namespace cost_of_books_purchasing_plans_l1036_103662

theorem cost_of_books (x y : ℕ) (h1 : 4 * x + 2 * y = 480) (h2 : 2 * x + 3 * y = 520) : x = 50 ∧ y = 140 :=
by
  -- proof can be filled in later
  sorry

theorem purchasing_plans (a b : ℕ) (h_total_cost : 50 * a + 140 * (20 - a) ≤ 1720) (h_quantity : a ≤ 2 * (20 - b)) : (a = 12 ∧ b = 8) ∨ (a = 13 ∧ b = 7) :=
by
  -- proof can be filled in later
  sorry

end cost_of_books_purchasing_plans_l1036_103662


namespace find_pair_not_satisfying_equation_l1036_103659

theorem find_pair_not_satisfying_equation :
  ¬ (187 * 314 - 104 * 565 = 41) :=
by
  sorry

end find_pair_not_satisfying_equation_l1036_103659


namespace cube_decomposition_l1036_103665

theorem cube_decomposition (n s : ℕ) (h1 : n > s) (h2 : n^3 - s^3 = 152) : n = 6 := 
by
  sorry

end cube_decomposition_l1036_103665


namespace point_on_line_iff_l1036_103612

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- Given points O, A, B, and X in a vector space V, prove that X lies on the line AB if and only if
there exists a scalar t such that the position vector of X is a linear combination of the position vectors
of A and B with respect to O. -/
theorem point_on_line_iff (O A B X : V) :
  (∃ t : ℝ, X - O = t • (A - O) + (1 - t) • (B - O)) ↔ (∃ t : ℝ, ∃ (t : ℝ), X - O = (1 - t) • (A - O) + t • (B - O)) :=
sorry

end point_on_line_iff_l1036_103612


namespace problem_l1036_103631

theorem problem (x y : ℕ) (hy : y > 3) (h : x^2 + y^4 = 2 * ((x-6)^2 + (y+1)^2)) : x^2 + y^4 = 1994 := by
  sorry

end problem_l1036_103631


namespace equality_of_expressions_l1036_103616

theorem equality_of_expressions (a b c : ℝ) (h : a = b + c + 2) : 
  a + b * c = (a + b) * (a + c) ↔ a = 0 ∨ a = 1 :=
by sorry

end equality_of_expressions_l1036_103616


namespace shelter_cats_incoming_l1036_103608

theorem shelter_cats_incoming (x : ℕ) (h : x + x / 2 - 3 + 5 - 1 = 19) : x = 12 :=
by
  sorry

end shelter_cats_incoming_l1036_103608


namespace comm_ring_of_center_condition_l1036_103601

variable {R : Type*} [Ring R]

def in_center (x : R) : Prop := ∀ y : R, (x * y = y * x)

def is_commutative (R : Type*) [Ring R] : Prop := ∀ a b : R, a * b = b * a

theorem comm_ring_of_center_condition (h : ∀ x : R, in_center (x^2 - x)) : is_commutative R :=
sorry

end comm_ring_of_center_condition_l1036_103601


namespace second_part_of_ratio_l1036_103622

-- Define the conditions
def ratio_percent := 20
def first_part := 4

-- Define the proof statement using the conditions
theorem second_part_of_ratio (ratio_percent : ℕ) (first_part : ℕ) : 
  ∃ second_part : ℕ, (first_part * 100) = ratio_percent * second_part :=
by
  -- Let the second part be 20 and verify the condition
  use 20
  -- Clear the proof (details are not required)
  sorry

end second_part_of_ratio_l1036_103622


namespace ticket_count_l1036_103667

theorem ticket_count (x y : ℕ) 
  (h1 : x + y = 35)
  (h2 : 24 * x + 18 * y = 750) : 
  x = 20 ∧ y = 15 :=
by
  sorry

end ticket_count_l1036_103667


namespace minimum_value_expression_l1036_103693

theorem minimum_value_expression {a b c : ℝ} :
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 13 → 
  (∃ x, x = (a^2 + b^3 + c^4 + 2019) / (10 * b + 123 * c + 26) ∧ ∀ y, y ≤ x) →
  x = 4 :=
by
  sorry

end minimum_value_expression_l1036_103693


namespace second_place_team_wins_l1036_103689
open Nat

def points (wins ties : Nat) : Nat :=
  2 * wins + ties

def avg_points (p1 p2 p3 : Nat) : Nat :=
  (p1 + p2 + p3) / 3

def first_place_points := points 12 4
def elsa_team_points := points 8 10

def second_place_wins (w : Nat) : Nat :=
  w

def second_place_points (w : Nat) : Nat :=
  points w 1

theorem second_place_team_wins :
  ∃ (W : Nat), avg_points first_place_points (second_place_points W) elsa_team_points = 27 ∧ W = 13 :=
by sorry

end second_place_team_wins_l1036_103689


namespace find_a_l1036_103670

variable {a b c : ℝ}

theorem find_a 
  (h1 : (a + b + c) ^ 2 = 3 * (a ^ 2 + b ^ 2 + c ^ 2))
  (h2 : a + b + c = 12) : 
  a = 4 := 
sorry

end find_a_l1036_103670


namespace weight_loss_percentage_l1036_103617

variables (W : ℝ) (x : ℝ)

def weight_loss_challenge :=
  W - W * x / 100 + W * 2 / 100 = W * 86.7 / 100

theorem weight_loss_percentage (h : weight_loss_challenge W x) : x = 15.3 :=
by sorry

end weight_loss_percentage_l1036_103617


namespace xiao_li_hits_bullseye_14_times_l1036_103644

theorem xiao_li_hits_bullseye_14_times
  (initial_rifle_bullets : ℕ := 10)
  (initial_pistol_bullets : ℕ := 14)
  (reward_per_bullseye_rifle : ℕ := 2)
  (reward_per_bullseye_pistol : ℕ := 4)
  (xiao_wang_bullseyes : ℕ := 30)
  (total_bullets : ℕ := initial_rifle_bullets + xiao_wang_bullseyes * reward_per_bullseye_rifle) :
  ∃ (xiao_li_bullseyes : ℕ), total_bullets = initial_pistol_bullets + xiao_li_bullseyes * reward_per_bullseye_pistol ∧ xiao_li_bullseyes = 14 :=
by sorry

end xiao_li_hits_bullseye_14_times_l1036_103644


namespace no_positive_integer_solutions_l1036_103671

theorem no_positive_integer_solutions :
  ¬ ∃ (x y : ℕ) (h1 : x > 0) (h2 : y > 0), 21 * x * y = 7 - 3 * x - 4 * y :=
by
  sorry

end no_positive_integer_solutions_l1036_103671


namespace multiples_sum_squared_l1036_103627

theorem multiples_sum_squared :
  let a := 4
  let b := 4
  ((a + b)^2) = 64 :=
by
  sorry

end multiples_sum_squared_l1036_103627


namespace mrs_McGillicuddy_student_count_l1036_103690

theorem mrs_McGillicuddy_student_count :
  let morning_registered := 25
  let morning_absent := 3
  let early_afternoon_registered := 24
  let early_afternoon_absent := 4
  let late_afternoon_registered := 30
  let late_afternoon_absent := 5
  let evening_registered := 35
  let evening_absent := 7
  let morning_present := morning_registered - morning_absent
  let early_afternoon_present := early_afternoon_registered - early_afternoon_absent
  let late_afternoon_present := late_afternoon_registered - late_afternoon_absent
  let evening_present := evening_registered - evening_absent
  let total_present := morning_present + early_afternoon_present + late_afternoon_present + evening_present
  total_present = 95 :=
by
  sorry

end mrs_McGillicuddy_student_count_l1036_103690


namespace neg_one_power_zero_l1036_103653

theorem neg_one_power_zero : (-1: ℤ)^0 = 1 := 
sorry

end neg_one_power_zero_l1036_103653


namespace ice_bag_cost_correct_l1036_103682

def total_cost_after_discount (cost_small cost_large : ℝ) (num_bags num_small : ℕ) (discount_rate : ℝ) : ℝ :=
  let num_large := num_bags - num_small
  let total_cost_before_discount := num_small * cost_small + num_large * cost_large
  let discount := discount_rate * total_cost_before_discount
  total_cost_before_discount - discount

theorem ice_bag_cost_correct :
  total_cost_after_discount 0.80 1.46 30 18 0.12 = 28.09 :=
by
  sorry

end ice_bag_cost_correct_l1036_103682


namespace intersect_point_l1036_103647

-- Definitions as per conditions
def f (x : ℝ) (b : ℝ) : ℝ := 4 * x + b
def f_inv (x : ℝ) (a : ℝ) : ℝ := a -- We define inverse as per given (4, a)

-- Variables for the conditions
variables (a b : ℤ)

-- Theorems to prove the conditions match the answers
theorem intersect_point : ∃ a b : ℤ, f 4 b = a ∧ f_inv 4 a = 4 ∧ a = 4 := by
  sorry

end intersect_point_l1036_103647


namespace solve_for_z_l1036_103677

theorem solve_for_z (i : ℂ) (z : ℂ) (h : 3 - 5 * i * z = -2 + 5 * i * z) (h_i : i^2 = -1) :
  z = -i / 2 :=
by {
  sorry
}

end solve_for_z_l1036_103677


namespace problems_completed_l1036_103669

theorem problems_completed (p t : ℕ) (h1 : p > 15) (h2 : pt = (2 * p - 6) * (t - 3)) : p * t = 216 := 
by
  sorry

end problems_completed_l1036_103669


namespace volume_of_cube_l1036_103656

-- Define the conditions
def surface_area (a : ℝ) : ℝ := 6 * a^2
def side_length (a : ℝ) (SA : ℝ) : Prop := SA = 6 * a^2
def volume (a : ℝ) : ℝ := a^3

-- State the theorem
theorem volume_of_cube (a : ℝ) (SA : surface_area a = 150) : volume a = 125 := 
sorry

end volume_of_cube_l1036_103656


namespace probability_unit_square_not_touch_central_2x2_square_l1036_103654

-- Given a 6x6 checkerboard with a marked 2x2 square at the center,
-- prove that the probability of choosing a unit square that does not touch
-- the marked 2x2 square is 2/3.

theorem probability_unit_square_not_touch_central_2x2_square : 
    let total_squares := 36
    let touching_squares := 12
    let squares_not_touching := total_squares - touching_squares
    (squares_not_touching : ℚ) / (total_squares : ℚ) = 2 / 3 := by
  sorry

end probability_unit_square_not_touch_central_2x2_square_l1036_103654


namespace problem_l1036_103628

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry

-- Conditions
def condition1 : a + b = 1 := sorry
def condition2 : a^2 + b^2 = 3 := sorry
def condition3 : a^3 + b^3 = 4 := sorry
def condition4 : a^4 + b^4 = 7 := sorry

-- Question and proof
theorem problem : a^10 + b^10 = 123 :=
by
  have h1 : a + b = 1 := condition1
  have h2 : a^2 + b^2 = 3 := condition2
  have h3 : a^3 + b^3 = 4 := condition3
  have h4 : a^4 + b^4 = 7 := condition4
  sorry

end problem_l1036_103628


namespace factorize_a_squared_plus_2a_l1036_103688

theorem factorize_a_squared_plus_2a (a : ℝ) : a^2 + 2*a = a * (a + 2) :=
sorry

end factorize_a_squared_plus_2a_l1036_103688


namespace mary_avg_speed_round_trip_l1036_103661

theorem mary_avg_speed_round_trip :
  let distance_to_school := 1.5 -- in km
  let time_to_school := 45 / 60 -- in hours (converted from minutes)
  let time_back_home := 15 / 60 -- in hours (converted from minutes)
  let total_distance := 2 * distance_to_school
  let total_time := time_to_school + time_back_home
  let avg_speed := total_distance / total_time
  avg_speed = 3 := by
  -- Definitions used directly appear in the conditions.
  -- Each condition used:
  -- Mary lives 1.5 km -> distance_to_school = 1.5
  -- Time to school 45 minutes -> time_to_school = 45 / 60
  -- Time back home 15 minutes -> time_back_home = 15 / 60
  -- Route is same -> total_distance = 2 * distance_to_school, total_time = time_to_school + time_back_home
  -- Proof to show avg_speed = 3
  sorry

end mary_avg_speed_round_trip_l1036_103661


namespace right_handed_players_total_l1036_103668

theorem right_handed_players_total (total_players throwers : ℕ) (non_throwers: ℕ := total_players - throwers)
  (left_handed_non_throwers : ℕ := non_throwers / 3)
  (right_handed_non_throwers : ℕ := non_throwers - left_handed_non_throwers)
  (all_throwers_right_handed : throwers = 37)
  (total_players_55 : total_players = 55)
  (one_third_left_handed : left_handed_non_throwers = non_throwers / 3)
  (right_handed_total: ℕ := throwers + right_handed_non_throwers)
  : right_handed_total = 49 := by
  sorry

end right_handed_players_total_l1036_103668


namespace jared_yearly_earnings_l1036_103663

theorem jared_yearly_earnings (monthly_pay_diploma : ℕ) (multiplier : ℕ) (months_in_year : ℕ)
  (h1 : monthly_pay_diploma = 4000) (h2 : multiplier = 3) (h3 : months_in_year = 12) :
  (monthly_pay_diploma * multiplier * months_in_year) = 144000 :=
by
  -- The proof goes here
  sorry

end jared_yearly_earnings_l1036_103663


namespace scrap_cookie_radius_l1036_103603

theorem scrap_cookie_radius (r: ℝ) (r_cookies: ℝ) (A_scrap: ℝ) (r_large: ℝ) (A_large: ℝ) (A_total_small: ℝ):
  r_cookies = 1.5 ∧
  r_large = r_cookies + 2 * r_cookies ∧
  A_large = π * r_large^2 ∧
  A_total_small = 8 * (π * r_cookies^2) ∧
  A_scrap = A_large - A_total_small ∧
  A_scrap = π * r^2
  → r = r_cookies
  :=
by
  intro h
  rcases h with ⟨hcookies, hrlarge, halarge, hatotalsmall, hascrap, hpi⟩
  sorry

end scrap_cookie_radius_l1036_103603


namespace solve_for_y_l1036_103632

theorem solve_for_y (y : ℝ) (h : y ≠ 2) :
  (7 * y / (y - 2) - 4 / (y - 2) = 3 / (y - 2)) → y = 1 :=
by
  intro h_eq
  sorry

end solve_for_y_l1036_103632


namespace original_bales_correct_l1036_103675

-- Definitions
def total_bales_now : Nat := 54
def bales_stacked_today : Nat := 26
def bales_originally_in_barn : Nat := total_bales_now - bales_stacked_today

-- Theorem statement
theorem original_bales_correct :
  bales_originally_in_barn = 28 :=
by {
  -- We will prove this later
  sorry
}

end original_bales_correct_l1036_103675


namespace christina_age_half_in_five_years_l1036_103620

theorem christina_age_half_in_five_years (C Y : ℕ) 
  (h1 : C + 5 = Y / 2)
  (h2 : 21 = 3 * C / 5) :
  Y = 80 :=
sorry

end christina_age_half_in_five_years_l1036_103620


namespace plan_y_more_cost_effective_l1036_103657

theorem plan_y_more_cost_effective (m : Nat) : 2500 + 7 * m < 15 * m → 313 ≤ m :=
by
  intro h
  sorry

end plan_y_more_cost_effective_l1036_103657


namespace locus_of_point_P_l1036_103624

noncomputable def ellipse_locus
  (r : ℝ) (u v : ℝ) : Prop :=
  ∃ x1 y1 : ℝ,
    (x1^2 + y1^2 = r^2) ∧ (u - x1)^2 + v^2 = y1^2

theorem locus_of_point_P {r u v : ℝ} :
  (ellipse_locus r u v) ↔ ((u^2 / (2 * r^2)) + (v^2 / r^2) ≤ 1) :=
by sorry

end locus_of_point_P_l1036_103624


namespace calculate_expression_l1036_103660

theorem calculate_expression : (235 - 2 * 3 * 5) * 7 / 5 = 287 := 
by
  sorry

end calculate_expression_l1036_103660


namespace total_jokes_after_eight_days_l1036_103686

def jokes_counted (start_jokes : ℕ) (n : ℕ) : ℕ :=
  -- Sum of initial jokes until the nth day by doubling each day
  start_jokes * (2 ^ n - 1)

theorem total_jokes_after_eight_days (jessy_jokes : ℕ) (alan_jokes : ℕ) (tom_jokes : ℕ) (emily_jokes : ℕ)
  (total_days : ℕ) (days_per_week : ℕ) :
  total_days = 5 → days_per_week = 8 →
  jessy_jokes = 11 → alan_jokes = 7 → tom_jokes = 5 → emily_jokes = 3 →
  (jokes_counted jessy_jokes (days_per_week - total_days) +
   jokes_counted alan_jokes (days_per_week - total_days) +
   jokes_counted tom_jokes (days_per_week - total_days) +
   jokes_counted emily_jokes (days_per_week - total_days)) = 806 :=
by
  intros
  sorry

end total_jokes_after_eight_days_l1036_103686


namespace find_m_independent_quadratic_term_l1036_103604

def quadratic_poly (m : ℝ) (x : ℝ) : ℝ :=
  -3 * x^2 + m * x^2 - x + 3

theorem find_m_independent_quadratic_term (m : ℝ) :
  (∀ x, quadratic_poly m x = -x + 3) → m = 3 :=
by 
  sorry

end find_m_independent_quadratic_term_l1036_103604


namespace lola_dora_allowance_l1036_103621

variable (total_cost deck_cost sticker_cost sticker_count packs_each : ℕ)
variable (allowance : ℕ)

theorem lola_dora_allowance 
  (h1 : deck_cost = 10)
  (h2 : sticker_cost = 2)
  (h3 : packs_each = 2)
  (h4 : sticker_count = 2 * packs_each)
  (h5 : total_cost = deck_cost + sticker_count * sticker_cost)
  (h6 : total_cost = 18) :
  allowance = 9 :=
sorry

end lola_dora_allowance_l1036_103621


namespace largest_of_five_consecutive_integers_with_product_15120_eq_9_l1036_103600

theorem largest_of_five_consecutive_integers_with_product_15120_eq_9 :
  ∃ n : ℕ, (n + 0) * (n + 1) * (n + 2) * (n + 3) * (n + 4) = 15120 ∧ n + 4 = 9 :=
by
  sorry

end largest_of_five_consecutive_integers_with_product_15120_eq_9_l1036_103600


namespace sufficient_condition_for_one_positive_and_one_negative_root_l1036_103666

theorem sufficient_condition_for_one_positive_and_one_negative_root (a : ℝ) (h₀ : a ≠ 0) :
  a < -1 ↔ (∃ x y : ℝ, (a * x^2 + 2 * x + 1 = 0) ∧ (a * y^2 + 2 * y + 1 = 0) ∧ x > 0 ∧ y < 0) :=
by {
  sorry
}

end sufficient_condition_for_one_positive_and_one_negative_root_l1036_103666


namespace adam_total_spending_l1036_103651

def first_laptop_cost : ℤ := 500
def second_laptop_cost : ℤ := 3 * first_laptop_cost
def total_cost : ℤ := first_laptop_cost + second_laptop_cost

theorem adam_total_spending : total_cost = 2000 := by
  sorry

end adam_total_spending_l1036_103651


namespace yellow_balls_count_l1036_103674

theorem yellow_balls_count (total_balls white_balls green_balls red_balls purple_balls : ℕ)
  (h_total : total_balls = 100)
  (h_white : white_balls = 20)
  (h_green : green_balls = 30)
  (h_red : red_balls = 37)
  (h_purple : purple_balls = 3)
  (h_prob : ((white_balls + green_balls + (total_balls - white_balls - green_balls - red_balls - purple_balls)) / total_balls : ℝ) = 0.6) :
  (total_balls - white_balls - green_balls - red_balls - purple_balls = 10) :=
by {
  sorry
}

end yellow_balls_count_l1036_103674


namespace quadrilateral_side_inequality_quadrilateral_side_inequality_if_intersect_l1036_103695

variable (a b c d : ℝ)
variable (angle_B angle_D : ℝ)
variable (d_intersect_circle : Prop)

-- Condition that angles B and D sum up to more than 180 degrees.
def angle_condition : Prop := angle_B + angle_D > 180

-- Condition for sides of the convex quadrilateral
def side_condition1 : Prop := a + c > b + d

-- Condition for the circle touching sides a, b, and c
def circle_tangent : Prop := True -- Placeholder as no function to verify this directly in Lean

theorem quadrilateral_side_inequality (h1 : angle_condition angle_B angle_D) 
                                      (h2 : circle_tangent) 
                                      (h3 : ¬ d_intersect_circle) 
                                      : a + c > b + d :=
  sorry

theorem quadrilateral_side_inequality_if_intersect (h1 : angle_condition angle_B angle_D) 
                                                   (h2 : circle_tangent) 
                                                   (h3 : d_intersect_circle) 
                                                   : a + c < b + d :=
  sorry

end quadrilateral_side_inequality_quadrilateral_side_inequality_if_intersect_l1036_103695


namespace largest_k_l1036_103697

theorem largest_k (k n : ℕ) (h1 : 2^11 = (k * (2 * n + k + 1)) / 2) : k = 1 := sorry

end largest_k_l1036_103697


namespace outer_circle_radius_l1036_103645

theorem outer_circle_radius (C_inner : ℝ) (w : ℝ) (r_outer : ℝ) (h1 : C_inner = 440) (h2 : w = 14) :
  r_outer = (440 / (2 * Real.pi)) + 14 :=
by 
  have h_r_inner : r_outer = (440 / (2 * Real.pi)) + 14 := by sorry
  exact h_r_inner

end outer_circle_radius_l1036_103645


namespace temperature_range_l1036_103619

-- Define the problem conditions
def highest_temp := 26
def lowest_temp := 12

-- The theorem stating the range of temperature change
theorem temperature_range : ∀ t : ℝ, lowest_temp ≤ t ∧ t ≤ highest_temp :=
by sorry

end temperature_range_l1036_103619


namespace angle_between_vectors_l1036_103646

def vector (α : Type) [Field α] := (α × α)

theorem angle_between_vectors
    (a : vector ℝ)
    (b : vector ℝ)
    (ha : a = (4, 0))
    (hb : b = (-1, Real.sqrt 3)) :
  let dot_product (v w : vector ℝ) : ℝ := (v.1 * w.1 + v.2 * w.2)
  let norm (v : vector ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)
  let cos_theta := dot_product a b / (norm a * norm b)
  ∀ theta, Real.cos theta = cos_theta → theta = 2 * Real.pi / 3 :=
by
  sorry

end angle_between_vectors_l1036_103646
