import Mathlib

namespace fractional_part_of_blue_square_four_changes_l901_90135

theorem fractional_part_of_blue_square_four_changes 
  (initial_area : ℝ)
  (f : ℝ → ℝ)
  (h_f : ∀ (a : ℝ), f a = (8 / 9) * a) :
  (f^[4]) initial_area / initial_area = 4096 / 6561 :=
by
  sorry

end fractional_part_of_blue_square_four_changes_l901_90135


namespace mario_hibiscus_l901_90144

def hibiscus_flowers (F : ℕ) : Prop :=
  let F2 := 2 * F
  let F3 := 4 * F2
  F + F2 + F3 = 22 → F = 2

theorem mario_hibiscus (F : ℕ) : hibiscus_flowers F :=
  sorry

end mario_hibiscus_l901_90144


namespace solve_for_x_l901_90147

theorem solve_for_x (x : ℕ) : (3 : ℝ)^(27^x) = (27 : ℝ)^(3^x) → x = 0 :=
by
  sorry

end solve_for_x_l901_90147


namespace cube_probability_l901_90187

def prob_same_color_vertical_faces : ℕ := sorry

theorem cube_probability :
  prob_same_color_vertical_faces = 1 / 27 := 
sorry

end cube_probability_l901_90187


namespace weekly_milk_consumption_l901_90194

def milk_weekday : Nat := 3
def milk_saturday := 2 * milk_weekday
def milk_sunday := 3 * milk_weekday

theorem weekly_milk_consumption : (5 * milk_weekday) + milk_saturday + milk_sunday = 30 := by
  sorry

end weekly_milk_consumption_l901_90194


namespace problem_equiv_none_of_these_l901_90139

variable {x y : ℝ}

theorem problem_equiv_none_of_these (hx : x ≠ 0) (hx3 : x ≠ 3) (hy : y ≠ 0) (hy5 : y ≠ 5) :
  (3 / x + 2 / y = 1 / 3) →
  ¬(3 * x + 2 * y = x * y) ∧
  ¬(y = 3 * x / (5 - y)) ∧
  ¬(x / 3 + y / 2 = 3) ∧
  ¬(3 * y / (y - 5) = x) :=
sorry

end problem_equiv_none_of_these_l901_90139


namespace sum_of_surface_areas_of_two_smaller_cuboids_l901_90130

theorem sum_of_surface_areas_of_two_smaller_cuboids
  (L W H : ℝ) (hL : L = 3) (hW : W = 2) (hH : H = 1) :
  ∃ S, (S = 26 ∨ S = 28 ∨ S = 34) ∧ (∀ l w h, (l = L / 2 ∨ w = W / 2 ∨ h = H / 2) →
  (S = 2 * 2 * (l * W + w * H + h * L))) :=
by
  sorry

end sum_of_surface_areas_of_two_smaller_cuboids_l901_90130


namespace youngest_sibling_is_42_l901_90119

-- Definitions for the problem conditions
def consecutive_even_integers (a : ℤ) := [a, a + 2, a + 4, a + 6]
def sum_of_ages_is_180 (ages : List ℤ) := ages.sum = 180

-- Main statement
theorem youngest_sibling_is_42 (a : ℤ) 
  (h1 : sum_of_ages_is_180 (consecutive_even_integers a)) :
  a = 42 := 
sorry

end youngest_sibling_is_42_l901_90119


namespace unique_solution_to_equation_l901_90112

theorem unique_solution_to_equation (x y z : ℤ) 
    (h : 5 * x^3 + 11 * y^3 + 13 * z^3 = 0) : x = 0 ∧ y = 0 ∧ z = 0 :=
by
  sorry

end unique_solution_to_equation_l901_90112


namespace total_results_count_l901_90175

theorem total_results_count (N : ℕ) (S : ℕ) 
  (h1 : S = 50 * N) 
  (h2 : (12 * 14) + (12 * 17) = 372)
  (h3 : S = 372 + 878) : N = 25 := 
by 
  sorry

end total_results_count_l901_90175


namespace tan_diff_l901_90154

theorem tan_diff (α β : ℝ) (hα : Real.tan α = 3) (hβ : Real.tan β = 2) : Real.tan (α - β) = 1 / 7 := by
  sorry

end tan_diff_l901_90154


namespace sum_of_reciprocals_squares_l901_90182

theorem sum_of_reciprocals_squares (a b : ℕ) (h : a * b = 17) :
  (1 : ℚ) / (a * a) + 1 / (b * b) = 290 / 289 :=
sorry

end sum_of_reciprocals_squares_l901_90182


namespace polar_to_rectangular_l901_90156

open Real

theorem polar_to_rectangular (r θ : ℝ) (h_r : r = 8) (h_θ : θ = π / 4) :
    (r * cos θ, r * sin θ) = (4 * sqrt 2, 4 * sqrt 2) :=
by
  rw [h_r, h_θ]
  rw [cos_pi_div_four, sin_pi_div_four]
  norm_num
  field_simp [sqrt_eq_rpow]
  sorry

end polar_to_rectangular_l901_90156


namespace police_coverage_l901_90159

-- Define the intersections and streets
inductive Intersection : Type
| A | B | C | D | E | F | G | H | I | J | K

open Intersection

-- Define the streets
def Streets : List (List Intersection) :=
  [ [A, B, C, D],    -- Horizontal street 1
    [E, F, G],       -- Horizontal street 2
    [H, I, J, K],    -- Horizontal street 3
    [A, E, H],       -- Vertical street 1
    [B, F, I],       -- Vertical street 2
    [D, G, J],       -- Vertical street 3
    [H, F, C],       -- Diagonal street 1
    [C, G, K]        -- Diagonal street 2
  ]

-- Define the set of intersections where police officers are 
def policeIntersections : List Intersection := [B, G, H]

-- State the theorem to be proved
theorem police_coverage : 
  ∀ (street : List Intersection), street ∈ Streets → 
  ∃ (i : Intersection), i ∈ policeIntersections ∧ i ∈ street := 
sorry

end police_coverage_l901_90159


namespace integer_rational_ratio_l901_90199

open Real

theorem integer_rational_ratio (a b : ℤ) (h : (a : ℝ) + sqrt b = sqrt (15 + sqrt 216)) : (a : ℚ) / b = 1 / 2 := 
by 
  -- Omitted proof 
  sorry

end integer_rational_ratio_l901_90199


namespace roxy_garden_problem_l901_90148

variable (initial_flowering : ℕ)
variable (multiplier : ℕ)
variable (bought_flowering : ℕ)
variable (bought_fruiting : ℕ)
variable (given_flowering : ℕ)
variable (given_fruiting : ℕ)

def initial_fruiting (initial_flowering : ℕ) (multiplier : ℕ) : ℕ :=
  initial_flowering * multiplier

def saturday_flowering (initial_flowering : ℕ) (bought_flowering : ℕ) : ℕ :=
  initial_flowering + bought_flowering

def saturday_fruiting (initial_fruiting : ℕ) (bought_fruiting : ℕ) : ℕ :=
  initial_fruiting + bought_fruiting

def sunday_flowering (saturday_flowering : ℕ) (given_flowering : ℕ) : ℕ :=
  saturday_flowering - given_flowering

def sunday_fruiting (saturday_fruiting : ℕ) (given_fruiting : ℕ) : ℕ :=
  saturday_fruiting - given_fruiting

def total_plants_remaining (sunday_flowering : ℕ) (sunday_fruiting : ℕ) : ℕ :=
  sunday_flowering + sunday_fruiting

theorem roxy_garden_problem 
  (h1 : initial_flowering = 7)
  (h2 : multiplier = 2)
  (h3 : bought_flowering = 3)
  (h4 : bought_fruiting = 2)
  (h5 : given_flowering = 1)
  (h6 : given_fruiting = 4) :
  total_plants_remaining 
    (sunday_flowering 
      (saturday_flowering initial_flowering bought_flowering) 
      given_flowering) 
    (sunday_fruiting 
      (saturday_fruiting 
        (initial_fruiting initial_flowering multiplier) 
        bought_fruiting) 
      given_fruiting) = 21 := 
  sorry

end roxy_garden_problem_l901_90148


namespace probability_value_expr_is_7_l901_90107

theorem probability_value_expr_is_7 : 
  let num_ones : ℕ := 15
  let num_ops : ℕ := 14
  let target_value : ℤ := 7
  let total_ways := 2 ^ num_ops
  let favorable_ways := (Nat.choose num_ops 11)  -- Ways to choose positions for +1's
  let prob := (favorable_ways : ℝ) / total_ways
  prob = 91 / 4096 := sorry

end probability_value_expr_is_7_l901_90107


namespace new_job_hourly_wage_l901_90110

def current_job_weekly_earnings : ℝ := 8 * 10
def new_job_hours_per_week : ℝ := 4
def new_job_bonus : ℝ := 35
def new_job_expected_additional_wage : ℝ := 15

theorem new_job_hourly_wage (W : ℝ) 
  (h_current_job : current_job_weekly_earnings = 80)
  (h_new_job : new_job_hours_per_week * W + new_job_bonus = current_job_weekly_earnings + new_job_expected_additional_wage) : 
  W = 15 :=
by 
  sorry

end new_job_hourly_wage_l901_90110


namespace angle_triple_supplement_l901_90122

theorem angle_triple_supplement (x : ℝ) (h : x = 3 * (180 - x)) : x = 135 :=
by sorry

end angle_triple_supplement_l901_90122


namespace logical_equivalence_l901_90180

theorem logical_equivalence (P Q R : Prop) :
  ((¬ P ∧ ¬ Q) → ¬ R) ↔ (R → (P ∨ Q)) :=
by sorry

end logical_equivalence_l901_90180


namespace project_completion_time_l901_90161

theorem project_completion_time
  (x y z : ℝ)
  (h1 : x + y = 1 / 2)
  (h2 : y + z = 1 / 4)
  (h3 : z + x = 1 / 2.4) :
  (1 / x) = 3 :=
by
  sorry

end project_completion_time_l901_90161


namespace sqrt_8_plus_sqrt_2_minus_sqrt_18_sqrt_3_minus_2_squared_l901_90146

theorem sqrt_8_plus_sqrt_2_minus_sqrt_18 :
  (Real.sqrt 8 + Real.sqrt 2 - Real.sqrt 18 = 0) :=
sorry

theorem sqrt_3_minus_2_squared :
  ((Real.sqrt 3 - 2) ^ 2 = 7 - 4 * Real.sqrt 3) :=
sorry

end sqrt_8_plus_sqrt_2_minus_sqrt_18_sqrt_3_minus_2_squared_l901_90146


namespace bill_drew_12_triangles_l901_90117

theorem bill_drew_12_triangles 
  (T : ℕ)
  (total_lines : T * 3 + 8 * 4 + 4 * 5 = 88) : 
  T = 12 :=
sorry

end bill_drew_12_triangles_l901_90117


namespace sum_outer_equal_sum_inner_l901_90121

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def reverse_digits (n : ℕ) : ℕ :=
  let a := n / 1000
  let b := (n % 1000) / 100
  let c := (n % 100) / 10
  let d := n % 10
  1000 * d + 100 * c + 10 * b + a

theorem sum_outer_equal_sum_inner (M N : ℕ) (a b c d : ℕ) 
  (h1 : is_four_digit M)
  (h2 : M = 1000 * a + 100 * b + 10 * c + d) 
  (h3 : N = reverse_digits M) 
  (h4 : M + N % 101 = 0) 
  (h5 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0) : 
  a + d = b + c :=
  sorry

end sum_outer_equal_sum_inner_l901_90121


namespace smallest_positive_period_of_sin_2x_l901_90183

noncomputable def period_of_sine (B : ℝ) : ℝ := (2 * Real.pi) / B

theorem smallest_positive_period_of_sin_2x :
  period_of_sine 2 = Real.pi := sorry

end smallest_positive_period_of_sin_2x_l901_90183


namespace number_of_ninth_graders_l901_90170

def num_students_total := 50
def num_students_7th (x : Int) := 2 * x - 1
def num_students_8th (x : Int) := x

theorem number_of_ninth_graders (x : Int) :
  num_students_7th x + num_students_8th x + (51 - 3 * x) = num_students_total := by
  sorry

end number_of_ninth_graders_l901_90170


namespace required_speed_remaining_l901_90142

theorem required_speed_remaining (total_distance : ℕ) (total_time : ℕ) (initial_speed : ℕ) (initial_time : ℕ) 
  (h1 : total_distance = 24) (h2 : total_time = 8) (h3 : initial_speed = 4) (h4 : initial_time = 4) :
  (total_distance - initial_speed * initial_time) / (total_time - initial_time) = 2 := by
  sorry

end required_speed_remaining_l901_90142


namespace cuboid_volume_l901_90114

theorem cuboid_volume (length width height : ℕ) (h_length : length = 4) (h_width : width = 4) (h_height : height = 6) : (length * width * height = 96) :=
by 
  -- Sorry places a placeholder for the actual proof
  sorry

end cuboid_volume_l901_90114


namespace evaluate_expression_l901_90158

theorem evaluate_expression (a b c d : ℝ) 
  (h1 : a^2 + b^2 + c^2 + d^2 = 2018) 
  (h2 : 3 * a + 8 * b + 24 * c + 37 * d = 2018) : 
  3 * b + 8 * c + 24 * d + 37 * a = 1215 :=
by 
  sorry

end evaluate_expression_l901_90158


namespace opposite_event_of_hitting_at_least_once_is_missing_both_times_l901_90165

theorem opposite_event_of_hitting_at_least_once_is_missing_both_times
  (A B : Prop) :
  ¬(A ∨ B) ↔ (¬A ∧ ¬B) :=
by
  sorry

end opposite_event_of_hitting_at_least_once_is_missing_both_times_l901_90165


namespace distance_between_pulley_centers_l901_90102

theorem distance_between_pulley_centers (R1 R2 CD : ℝ) (R1_pos : R1 = 10) (R2_pos : R2 = 6) (CD_pos : CD = 30) :
  ∃ AB : ℝ, AB = 2 * Real.sqrt 229 :=
by
  sorry

end distance_between_pulley_centers_l901_90102


namespace boys_girls_relation_l901_90113

theorem boys_girls_relation (b g : ℕ) :
  (∃ b, 3 + (b - 1) * 2 = g) → b = (g - 1) / 2 :=
by
  intro h
  sorry

end boys_girls_relation_l901_90113


namespace max_possible_salary_l901_90162

-- Definition of the conditions
def num_players : ℕ := 25
def min_salary : ℕ := 20000
def total_salary_cap : ℕ := 800000

-- The theorem we want to prove: the maximum possible salary for a single player is $320,000
theorem max_possible_salary (total_salary_cap : ℕ) (num_players : ℕ) (min_salary : ℕ) :
  total_salary_cap - (num_players - 1) * min_salary = 320000 :=
by sorry

end max_possible_salary_l901_90162


namespace range_of_a_l901_90126

theorem range_of_a (f : ℝ → ℝ) (a : ℝ):
  (∀ x, f x = f (-x)) →
  (∀ x y, 0 ≤ x → x < y → f x ≤ f y) →
  (∀ x, 1/2 ≤ x ∧ x ≤ 1 → f (a * x + 1) ≤ f (x - 2)) →
  -2 ≤ a ∧ a ≤ 0 := 
sorry

end range_of_a_l901_90126


namespace stapler_machines_l901_90151

theorem stapler_machines (x : ℝ) :
  (∃ (x : ℝ), x > 0) ∧
  ((∀ r1 r2 : ℝ, (r1 = 800 / 6) → (r2 = 800 / x) → (r1 + r2 = 800 / 3)) ↔
    (1 / 6 + 1 / x = 1 / 3)) :=
by sorry

end stapler_machines_l901_90151


namespace Q_root_l901_90171

def Q (x : ℝ) : ℝ := x^3 - 6 * x^2 + 12 * x - 11

theorem Q_root : Q (3^(1 / 3 : ℝ) + 2) = 0 := sorry

end Q_root_l901_90171


namespace simplify_expression_l901_90198

theorem simplify_expression : (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1 / 2 :=
by
  sorry

end simplify_expression_l901_90198


namespace problem1_problem2_l901_90108

open Real

variables {α β γ : ℝ}

theorem problem1 (α β : ℝ) :
  abs (cos (α + β)) ≤ abs (cos α) + abs (sin β) ∧
  abs (sin (α + β)) ≤ abs (cos α) + abs (cos β) :=
sorry

theorem problem2 (h : α + β + γ = 0) :
  abs (cos α) + abs (cos β) + abs (cos γ) ≥ 1 :=
sorry

end problem1_problem2_l901_90108


namespace problem_2003_divisibility_l901_90163

theorem problem_2003_divisibility :
  let N := (List.range' 1 1001).prod + (List.range' 1002 1001).prod
  N % 2003 = 0 := by
  sorry

end problem_2003_divisibility_l901_90163


namespace knights_on_red_chairs_l901_90129

variable (K L Kr Lb : ℕ)
variable (h1 : K + L = 20)
variable (h2 : Kr + Lb = 10)
variable (h3 : Kr = L - Lb)

/-- Given the conditions:
1. There are 20 seats with knights and liars such that K + L = 20.
2. Half of the individuals claim to be sitting on blue chairs, and half on red chairs such that Kr + Lb = 10.
3. Knights on red chairs (Kr) must be equal to liars minus liars on blue chairs (Lb).
Prove that the number of knights now sitting on red chairs is 5. -/
theorem knights_on_red_chairs : Kr = 5 :=
by
  sorry

end knights_on_red_chairs_l901_90129


namespace tangent_line_and_area_l901_90123

noncomputable def tangent_line_equation (t : ℝ) : String := 
  "x + e^t * y - t - 1 = 0"

noncomputable def area_triangle_MON (t : ℝ) : ℝ :=
  (t + 1)^2 / (2 * Real.exp t)

theorem tangent_line_and_area (t : ℝ) (ht : t > 0) :
  tangent_line_equation t = "x + e^t * y - t - 1 = 0" ∧
  area_triangle_MON t = (t + 1)^2 / (2 * Real.exp t) := by
  sorry

end tangent_line_and_area_l901_90123


namespace fraction_of_trunks_l901_90109

theorem fraction_of_trunks (h1 : 0.38 ≤ 1) (h2 : 0.63 ≤ 1) : 
  0.63 - 0.38 = 0.25 :=
by
  sorry

end fraction_of_trunks_l901_90109


namespace find_acute_angle_as_pi_over_4_l901_90169
open Real

-- Definitions from the problem's conditions
variables (x : ℝ)
def is_acute (x : ℝ) : Prop := 0 < x ∧ x < π / 2
def trig_eq (x : ℝ) : Prop := (sin x) ^ 3 + (cos x) ^ 3 = sqrt 2 / 2

-- The math proof problem statement
theorem find_acute_angle_as_pi_over_4 (h_acute : is_acute x) (h_trig_eq : trig_eq x) : x = π / 4 := 
sorry

end find_acute_angle_as_pi_over_4_l901_90169


namespace time_bob_cleans_room_l901_90166

variable (timeAlice : ℕ) (fractionBob : ℚ)

-- Definitions based on conditions from the problem
def timeAliceCleaningRoom : ℕ := 40
def fractionOfTimeBob : ℚ := 3 / 8

-- Prove the time it takes Bob to clean his room
theorem time_bob_cleans_room : (timeAliceCleaningRoom * fractionOfTimeBob : ℚ) = 15 := 
by
  sorry

end time_bob_cleans_room_l901_90166


namespace taxi_range_l901_90185

noncomputable def fare (x : ℝ) : ℝ :=
  if x <= 3 then 
    5
  else if x <= 10 then
    5 + (x - 3) * 2
  else
    5 + 7 * 2 + (x - 10) * 3

theorem taxi_range (x : ℝ) (h : fare x + 1 = 38) : 15 < x ∧ x ≤ 16 := 
  sorry

end taxi_range_l901_90185


namespace fraction_subtraction_l901_90140

theorem fraction_subtraction : 
  (2 + 4 + 6 + 8) / (1 + 3 + 5 + 7) - (1 + 3 + 5 + 7) / (2 + 4 + 6 + 8) 
  = 9 / 20 := by
  sorry

end fraction_subtraction_l901_90140


namespace Alton_profit_l901_90191

variable (earnings_per_day : ℕ)
variable (days_per_week : ℕ)
variable (rent_per_week : ℕ)

theorem Alton_profit (h1 : earnings_per_day = 8) (h2 : days_per_week = 7) (h3 : rent_per_week = 20) :
  earnings_per_day * days_per_week - rent_per_week = 36 := 
by sorry

end Alton_profit_l901_90191


namespace distinct_solutions_subtraction_eq_two_l901_90155

theorem distinct_solutions_subtraction_eq_two :
  ∃ p q : ℝ, (p ≠ q) ∧ (p > q) ∧ ((6 * p - 18) / (p^2 + 4 * p - 21) = p + 3) ∧ ((6 * q - 18) / (q^2 + 4 * q - 21) = q + 3) ∧ (p - q = 2) :=
by
  have p := -3
  have q := -5
  exists p, q
  sorry

end distinct_solutions_subtraction_eq_two_l901_90155


namespace profit_percent_l901_90160

theorem profit_percent (cost_price : ℝ) (selling_price : ℝ) (marked_price : ℝ) (n_pens : ℕ) 
  (h1 : n_pens = 60) (h2 : marked_price = 1) (h3 : cost_price = (46 : ℝ) / (60 : ℝ)) 
  (h4 : selling_price = 0.99 * marked_price) : 
  (selling_price - cost_price) / cost_price * 100 = 29.11 :=
by
  sorry

end profit_percent_l901_90160


namespace total_remaining_books_l901_90190

-- Define the initial conditions as constants
def total_books_crazy_silly_school : ℕ := 14
def read_books_crazy_silly_school : ℕ := 8
def total_books_mystical_adventures : ℕ := 10
def read_books_mystical_adventures : ℕ := 5
def total_books_sci_fi_universe : ℕ := 18
def read_books_sci_fi_universe : ℕ := 12

-- Define the remaining books calculation
def remaining_books_crazy_silly_school : ℕ :=
  total_books_crazy_silly_school - read_books_crazy_silly_school

def remaining_books_mystical_adventures : ℕ :=
  total_books_mystical_adventures - read_books_mystical_adventures

def remaining_books_sci_fi_universe : ℕ :=
  total_books_sci_fi_universe - read_books_sci_fi_universe

-- Define the proof statement
theorem total_remaining_books : 
  remaining_books_crazy_silly_school + remaining_books_mystical_adventures + remaining_books_sci_fi_universe = 17 := by
  sorry

end total_remaining_books_l901_90190


namespace find_c_in_parabola_l901_90141

theorem find_c_in_parabola (b c : ℝ) (h₁ : 2 = (-1) ^ 2 + b * (-1) + c) (h₂ : 2 = 3 ^ 2 + b * 3 + c) : c = -1 :=
sorry

end find_c_in_parabola_l901_90141


namespace necessary_but_not_sufficient_l901_90104

theorem necessary_but_not_sufficient (A B : Prop) (h : A → B) : ¬ (B → A) :=
sorry

end necessary_but_not_sufficient_l901_90104


namespace minimum_value_of_expression_l901_90188

noncomputable def f (x : ℝ) : ℝ := 16^x - 2^x + x^2 + 1

theorem minimum_value_of_expression : ∃ (x : ℝ), f x = 1 ∧ ∀ y : ℝ, f y ≥ 1 := 
sorry

end minimum_value_of_expression_l901_90188


namespace cos_diff_angle_l901_90178

theorem cos_diff_angle
  (α : ℝ) (hα1 : 0 < α) (hα2 : α < π / 2) 
  (h : 3 * Real.sin α = Real.tan α) :
  Real.cos (α - π / 4) = (4 + Real.sqrt 2) / 6 :=
sorry

end cos_diff_angle_l901_90178


namespace probability_three_cards_l901_90128

theorem probability_three_cards (S : Type) [Fintype S]
  (deck : Finset S) (n : ℕ) (hn : n = 52)
  (hearts : Finset S) (spades : Finset S)
  (tens: Finset S)
  (hhearts_count : ∃ k, hearts.card = k ∧ k = 13)
  (hspades_count : ∃ k, spades.card = k ∧ k = 13)
  (htens_count : ∃ k, tens.card = k ∧ k = 4)
  (hdeck_partition : ∀ x ∈ deck, x ∈ hearts ∨ x ∈ spades ∨ x ∈ tens ∨ (x ∉ hearts ∧ x ∉ spades ∧ x ∉ tens)) :
  (12 / 52 * 13 / 51 * 4 / 50 + 1 / 52 * 13 / 51 * 3 / 50 = 221 / 44200) :=
by {
  sorry
}

end probability_three_cards_l901_90128


namespace sum_of_solutions_l901_90184

theorem sum_of_solutions : ∀ x : ℚ, (4 * x + 6) * (3 * x - 8) = 0 → 
  (x = -3 / 2 ∨ x = 8 / 3) → 
  (-3 / 2 + 8 / 3) = 7 / 6 :=
by
  intros x h1 h2
  sorry

end sum_of_solutions_l901_90184


namespace inequality_proof_l901_90173

theorem inequality_proof (a b : ℝ) : a^2 + b^2 + 2 * (a - 1) * (b - 1) ≥ 1 :=
by
  sorry

end inequality_proof_l901_90173


namespace graph_passes_through_point_l901_90168

noncomputable def exponential_shift (a : ℝ) (x : ℝ) := a^(x - 2)

theorem graph_passes_through_point (a : ℝ) (h : a > 0) (h1 : a ≠ 1) : exponential_shift a 2 = 1 :=
by
  unfold exponential_shift
  sorry

end graph_passes_through_point_l901_90168


namespace product_of_B_coordinates_l901_90153

theorem product_of_B_coordinates :
  (∃ (x y : ℝ), (1 / 3 * x + 2 / 3 * 4 = 1 ∧ 1 / 3 * y + 2 / 3 * 2 = 7) ∧ x * y = -85) :=
by
  sorry

end product_of_B_coordinates_l901_90153


namespace sum_interior_angles_l901_90105

theorem sum_interior_angles (n : ℕ) (h : 180 * (n - 2) = 3240) : 180 * ((n + 3) - 2) = 3780 := by
  sorry

end sum_interior_angles_l901_90105


namespace intersection_point_finv_l901_90189

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := 4 * x + b

theorem intersection_point_finv (a b : ℤ) : 
  (∀ x : ℝ, f (f x b) b = x) → 
  (∀ y : ℝ, f (f y b) b = y) → 
  (f (-4) b = a) → 
  (f a b = -4) → 
  a = -4 := 
by
  intros
  sorry

end intersection_point_finv_l901_90189


namespace total_students_in_high_school_l901_90195

theorem total_students_in_high_school 
  (num_freshmen : ℕ)
  (num_sample : ℕ) 
  (num_sophomores : ℕ)
  (num_seniors : ℕ)
  (freshmen_drawn : ℕ)
  (sampling_ratio : ℕ)
  (total_students : ℕ)
  (h1 : num_freshmen = 600)
  (h2 : num_sample = 45)
  (h3 : num_sophomores = 20)
  (h4 : num_seniors = 10)
  (h5 : freshmen_drawn = 15)
  (h6 : sampling_ratio = 40)
  (h7 : freshmen_drawn * sampling_ratio = num_freshmen)
  : total_students = 1800 :=
sorry

end total_students_in_high_school_l901_90195


namespace arc_length_l901_90131

theorem arc_length 
  (a : ℝ) 
  (α β : ℝ) 
  (hα : 0 < α) 
  (hβ : 0 < β) 
  (h1 : α + β < π) 
  :  ∃ l : ℝ, l = (a * (π - α - β) * (Real.sin α) * (Real.sin β)) / (Real.sin (α + β)) :=
sorry

end arc_length_l901_90131


namespace difference_of_decimal_and_fraction_l901_90174

theorem difference_of_decimal_and_fraction :
  0.127 - (1 / 8) = 0.002 := 
by
  sorry

end difference_of_decimal_and_fraction_l901_90174


namespace cancel_terms_valid_equation_l901_90115

theorem cancel_terms_valid_equation {m n : ℕ} 
  (x : Fin n → ℕ) (y : Fin m → ℕ) 
  (h_sum_eq : (Finset.univ.sum x) = (Finset.univ.sum y))
  (h_sum_lt : (Finset.univ.sum x) < (m * n)) : 
  ∃ x' : Fin n → ℕ, ∃ y' : Fin m → ℕ, 
    (Finset.univ.sum x' = Finset.univ.sum y') ∧ x' ≠ x ∧ y' ≠ y :=
sorry

end cancel_terms_valid_equation_l901_90115


namespace total_cost_to_replace_floor_l901_90124

def removal_cost : ℝ := 50
def cost_per_sqft : ℝ := 1.25
def room_dimensions : (ℝ × ℝ) := (8, 7)

theorem total_cost_to_replace_floor :
  removal_cost + (cost_per_sqft * (room_dimensions.1 * room_dimensions.2)) = 120 := by
  sorry

end total_cost_to_replace_floor_l901_90124


namespace sum_even_and_odd_numbers_up_to_50_l901_90181

def sum_even_numbers (n : ℕ) : ℕ :=
  (2 + 50) * n / 2

def sum_odd_numbers (n : ℕ) : ℕ :=
  (1 + 49) * n / 2

theorem sum_even_and_odd_numbers_up_to_50 : 
  sum_even_numbers 25 + sum_odd_numbers 25 = 1275 :=
by
  sorry

end sum_even_and_odd_numbers_up_to_50_l901_90181


namespace input_value_for_output_16_l901_90172

theorem input_value_for_output_16 (x : ℝ) (y : ℝ) (h1 : x < 0 → y = (x + 1)^2) (h2 : x ≥ 0 → y = (x - 1)^2) (h3 : y = 16) : x = 5 ∨ x = -5 := by
  sorry

end input_value_for_output_16_l901_90172


namespace matrix_determinant_zero_l901_90138

theorem matrix_determinant_zero (a b : ℝ) : 
  Matrix.det ![
    ![1, Real.sin (2 * a), Real.sin a],
    ![Real.sin (2 * a), 1, Real.sin b],
    ![Real.sin a, Real.sin b, 1]
  ] = 0 := 
by 
  sorry

end matrix_determinant_zero_l901_90138


namespace length_of_bridge_correct_l901_90111

noncomputable def length_of_bridge (speed_kmh : ℝ) (time_min : ℝ) : ℝ :=
  let speed_mpm := (speed_kmh * 1000) / 60  -- Convert speed from km/hr to m/min
  speed_mpm * time_min  -- Length of the bridge in meters

theorem length_of_bridge_correct :
  length_of_bridge 10 10 = 1666.7 :=
by
  sorry

end length_of_bridge_correct_l901_90111


namespace equal_intercepts_l901_90100

theorem equal_intercepts (a : ℝ) (h : ∃p, (a * p, 0) = (0, a - 2)) : a = 1 ∨ a = 2 :=
sorry

end equal_intercepts_l901_90100


namespace value_of_y_l901_90177

theorem value_of_y (x y : ℝ) (h1 : x = 2) (h2 : x^(2*y) = 4) : y = 1 :=
by
  sorry

end value_of_y_l901_90177


namespace exp_arbitrarily_large_l901_90120

theorem exp_arbitrarily_large (a : ℝ) (h : a > 1) : ∀ y > 0, ∃ x > 0, a^x > y := by
  sorry

end exp_arbitrarily_large_l901_90120


namespace roots_are_distinct_and_negative_l901_90192

theorem roots_are_distinct_and_negative : 
  (∀ x : ℝ, x^2 + m * x + 1 = 0 → ∃! (x1 x2 : ℝ), x1 < 0 ∧ x2 < 0 ∧ x1 ≠ x2) ↔ m > 2 :=
by
  sorry

end roots_are_distinct_and_negative_l901_90192


namespace sum_cubes_l901_90164

variables (a b : ℝ)
noncomputable def calculate_sum_cubes (a b : ℝ) : ℝ :=
a^3 + b^3

theorem sum_cubes (h1 : a + b = 11) (h2 : a * b = 21) : calculate_sum_cubes a b = 638 :=
by
  sorry

end sum_cubes_l901_90164


namespace sin_double_angle_l901_90134

noncomputable def r := Real.sqrt 5
noncomputable def sin_α := -2 / r
noncomputable def cos_α := 1 / r
noncomputable def sin_2α := 2 * sin_α * cos_α

theorem sin_double_angle (α : ℝ) :
  (∃ P : ℝ × ℝ, P = (1, -2) ∧ ∃ α : ℝ, true) → sin_2α = -4 / 5 :=
by
  sorry

end sin_double_angle_l901_90134


namespace which_two_students_donated_l901_90176

theorem which_two_students_donated (A B C D : Prop) 
  (h1 : A ∨ D) 
  (h2 : ¬(A ∧ D)) 
  (h3 : (A ∧ B) ∨ (A ∧ D) ∨ (B ∧ D))
  (h4 : ¬(A ∧ B ∧ D)) 
  : B ∧ D :=
sorry

end which_two_students_donated_l901_90176


namespace problem_solution_l901_90179

def is_desirable_n (n : ℕ) : Prop :=
  ∃ (r b : ℕ), n = r + b ∧ r^2 - r*b + b^2 = 2007 ∧ 3 ∣ r ∧ 3 ∣ b

theorem problem_solution :
  ∀ n : ℕ, (is_desirable_n n → n = 69 ∨ n = 84) :=
by
  sorry

end problem_solution_l901_90179


namespace range_of_a_l901_90197

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x > 1 → x > a) ∧ (∃ x : ℝ, x > a ∧ x ≤ 1) → a < 1 :=
by
  sorry

end range_of_a_l901_90197


namespace trig_eq_solutions_l901_90106

theorem trig_eq_solutions (x : ℝ) (h : 0 ≤ x ∧ x ≤ 2 * Real.pi) :
  3 * Real.sin x = 1 + Real.cos (2 * x) ↔ x = Real.pi / 6 ∨ x = 5 * Real.pi / 6 :=
by
  sorry

end trig_eq_solutions_l901_90106


namespace sum_consecutive_even_l901_90116

theorem sum_consecutive_even (m : ℤ) : m + (m + 2) + (m + 4) + (m + 6) + (m + 8) + (m + 10) = 6 * m + 30 :=
by
  sorry

end sum_consecutive_even_l901_90116


namespace all_lucky_years_l901_90186

def is_lucky_year (y : ℕ) : Prop :=
  ∃ m d : ℕ, 1 ≤ m ∧ m ≤ 12 ∧ 1 ≤ d ∧ d ≤ 31 ∧ (m * d = y % 100)

theorem all_lucky_years :
  (is_lucky_year 2024) ∧ (is_lucky_year 2025) ∧ (is_lucky_year 2026) ∧ (is_lucky_year 2027) ∧ (is_lucky_year 2028) :=
sorry

end all_lucky_years_l901_90186


namespace agent_takes_19_percent_l901_90103

def agentPercentage (copies_sold : ℕ) (advance_copies : ℕ) (price_per_copy : ℕ) (steve_earnings : ℕ) : ℕ :=
  let total_earnings := copies_sold * price_per_copy
  let agent_earnings := total_earnings - steve_earnings
  let percentage_agent := 100 * agent_earnings / total_earnings
  percentage_agent

theorem agent_takes_19_percent :
  agentPercentage 1000000 100000 2 1620000 = 19 :=
by 
  sorry

end agent_takes_19_percent_l901_90103


namespace find_n_for_integer_roots_l901_90136

theorem find_n_for_integer_roots (n : ℤ):
    (∃ x y : ℤ, x ≠ y ∧ x^2 + (n+1)*x + (2*n - 1) = 0 ∧ y^2 + (n+1)*y + (2*n - 1) = 0) →
    (n = 1 ∨ n = 5) :=
sorry

end find_n_for_integer_roots_l901_90136


namespace total_votes_4500_l901_90125

theorem total_votes_4500 (V : ℝ) 
  (h : 0.60 * V - 0.40 * V = 900) : V = 4500 :=
by
  sorry

end total_votes_4500_l901_90125


namespace distance_Reims_to_Chaumont_l901_90193

noncomputable def distance_Chalons_Vitry : ℝ := 30
noncomputable def distance_Vitry_Chaumont : ℝ := 80
noncomputable def distance_Chaumont_SaintQuentin : ℝ := 236
noncomputable def distance_SaintQuentin_Reims : ℝ := 86
noncomputable def distance_Reims_Chalons : ℝ := 40

theorem distance_Reims_to_Chaumont :
  distance_Reims_Chalons + 
  distance_Chalons_Vitry + 
  distance_Vitry_Chaumont = 150 :=
sorry

end distance_Reims_to_Chaumont_l901_90193


namespace find_limit_of_hours_l901_90157

def regular_rate : ℝ := 16
def overtime_rate (r : ℝ) : ℝ := r * 1.75
def total_compensation : ℝ := 920
def total_hours : ℝ := 50

theorem find_limit_of_hours : 
  ∃ (L : ℝ), 
    total_compensation = (regular_rate * L) + ((overtime_rate regular_rate) * (total_hours - L)) →
    L = 40 :=
by
  sorry

end find_limit_of_hours_l901_90157


namespace percentage_markup_l901_90145

theorem percentage_markup (SP CP : ℕ) (h1 : SP = 8340) (h2 : CP = 6672) :
  ((SP - CP) / CP * 100) = 25 :=
by
  -- Before proving, we state our assumptions
  sorry

end percentage_markup_l901_90145


namespace percent_problem_l901_90133

theorem percent_problem
  (X : ℝ)
  (h1 : 0.28 * 400 = 112)
  (h2 : 0.45 * X + 112 = 224.5) :
  X = 250 := 
sorry

end percent_problem_l901_90133


namespace g_function_ratio_l901_90132

theorem g_function_ratio (g : ℝ → ℝ) (h : ∀ c d : ℝ, c^3 * g d = d^3 * g c) (hg3 : g 3 ≠ 0) :
  (g 6 - g 2) / g 3 = 208 / 27 := 
by
  sorry

end g_function_ratio_l901_90132


namespace odd_and_increasing_function_l901_90101

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f (x)

def is_increasing (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f (x) ≤ f (y)

def function_D (x : ℝ) : ℝ := x * abs x

theorem odd_and_increasing_function : 
  (is_odd function_D) ∧ (is_increasing function_D) :=
sorry

end odd_and_increasing_function_l901_90101


namespace rhombus_diagonal_length_l901_90143

theorem rhombus_diagonal_length (side : ℝ) (shorter_diagonal : ℝ) 
  (h1 : side = 51) (h2 : shorter_diagonal = 48) : 
  ∃ longer_diagonal : ℝ, longer_diagonal = 90 :=
by
  sorry

end rhombus_diagonal_length_l901_90143


namespace find_numbers_l901_90167

theorem find_numbers (a b c : ℝ) (x y z: ℝ) (h1 : x + y = z + a) (h2 : x + z = y + b) (h3 : y + z = x + c) :
    x = (a + b - c) / 2 ∧ y = (a - b + c) / 2 ∧ z = (-a + b + c) / 2 := by
  sorry

end find_numbers_l901_90167


namespace geometric_series_sum_l901_90149

theorem geometric_series_sum :
  (1 / 5 - 1 / 25 + 1 / 125 - 1 / 625 + 1 / 3125) = 521 / 3125 :=
by
  sorry

end geometric_series_sum_l901_90149


namespace find_m_of_lcm_conditions_l901_90196

theorem find_m_of_lcm_conditions (m : ℕ) (h_pos : 0 < m)
  (h1 : Int.lcm 18 m = 54)
  (h2 : Int.lcm m 45 = 180) : m = 36 :=
sorry

end find_m_of_lcm_conditions_l901_90196


namespace ninth_term_arithmetic_sequence_l901_90152

-- Definitions based on conditions:
def first_term : ℚ := 5 / 6
def seventeenth_term : ℚ := 5 / 8

-- Here is the main statement we need to prove:
theorem ninth_term_arithmetic_sequence : (first_term + 8 * ((seventeenth_term - first_term) / 16) = 15 / 16) :=
by
  sorry

end ninth_term_arithmetic_sequence_l901_90152


namespace percentage_loss_15_l901_90127

theorem percentage_loss_15
  (sold_at_loss : ℝ)
  (sold_at_profit : ℝ)
  (percentage_profit : ℝ)
  (cost_price : ℝ)
  (percentage_loss : ℝ)
  (H1 : sold_at_loss = 12)
  (H2 : sold_at_profit = 14.823529411764707)
  (H3 : percentage_profit = 5)
  (H4 : cost_price = sold_at_profit / (1 + percentage_profit / 100))
  (H5 : percentage_loss = (cost_price - sold_at_loss) / cost_price * 100) :
  percentage_loss = 15 :=
by
  sorry

end percentage_loss_15_l901_90127


namespace base12_addition_example_l901_90137

theorem base12_addition_example : 
  (5 * 12^2 + 2 * 12^1 + 8 * 12^0) + (2 * 12^2 + 7 * 12^1 + 3 * 12^0) = (7 * 12^2 + 9 * 12^1 + 11 * 12^0) :=
by sorry

end base12_addition_example_l901_90137


namespace factorable_polynomial_l901_90150

theorem factorable_polynomial (a b : ℝ) :
  (∀ x y : ℝ, ∃ u v p q : ℝ, (x + uy + v) * (x + py + q) = x * (x + 4) + a * (y^2 - 1) + 2 * b * y) ↔
  (a + 2)^2 + b^2 = 4 :=
  sorry

end factorable_polynomial_l901_90150


namespace negation_of_universal_proposition_l901_90118

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x > 2 → x^3 - 8 > 0)) ↔ (∃ x : ℝ, x > 2 ∧ x^3 - 8 ≤ 0) :=
by
  sorry

end negation_of_universal_proposition_l901_90118
