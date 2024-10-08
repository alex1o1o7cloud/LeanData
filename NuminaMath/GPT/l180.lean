import Mathlib

namespace arithmetic_sequence_transformation_l180_180714

theorem arithmetic_sequence_transformation (a : ℕ → ℝ) (d c : ℝ) (h : ∀ n, a (n + 1) = a n + d) (hc : c ≠ 0) :
  ∀ n, (c * a (n + 1)) - (c * a n) = c * d := 
by
  sorry

end arithmetic_sequence_transformation_l180_180714


namespace simplify_and_evaluate_expression_l180_180995

theorem simplify_and_evaluate_expression (a : ℝ) (h : a^2 + 2 * a - 8 = 0) :
  ((a^2 - 4) / (a^2 - 4 * a + 4) - a / (a - 2)) / (a^2 + 2 * a) / (a - 2) = 1 / 4 := 
by 
  sorry

end simplify_and_evaluate_expression_l180_180995


namespace cannot_form_right_triangle_l180_180524

theorem cannot_form_right_triangle :
  ¬ (6^2 + 7^2 = 8^2) :=
by
  sorry

end cannot_form_right_triangle_l180_180524


namespace total_seconds_eq_250200_l180_180707

def bianca_hours : ℝ := 12.5
def celeste_hours : ℝ := 2 * bianca_hours
def mcclain_hours : ℝ := celeste_hours - 8.5
def omar_hours : ℝ := bianca_hours + 3

def total_hours : ℝ := bianca_hours + celeste_hours + mcclain_hours + omar_hours
def hour_to_seconds : ℝ := 3600
def total_seconds : ℝ := total_hours * hour_to_seconds

theorem total_seconds_eq_250200 : total_seconds = 250200 := by
  sorry

end total_seconds_eq_250200_l180_180707


namespace vector_sum_solve_for_m_n_l180_180388

-- Define the vectors
def a : ℝ × ℝ := (3, 2)
def b : ℝ × ℝ := (-1, 2)
def c : ℝ × ℝ := (4, 1)

-- Problem 1: Vector sum
theorem vector_sum : 3 • a + b - 2 • c = (0, 6) :=
by sorry

-- Problem 2: Solving for m and n
theorem solve_for_m_n (m n : ℝ) (hm : a = m • b + n • c) :
  m = 5 / 9 ∧ n = 8 / 9 :=
by sorry

end vector_sum_solve_for_m_n_l180_180388


namespace number_of_rows_seating_exactly_9_students_l180_180437

theorem number_of_rows_seating_exactly_9_students (x : ℕ) : 
  ∀ y z, x * 9 + y * 5 + z * 8 = 55 → x % 5 = 1 ∧ x % 8 = 7 → x = 3 :=
by sorry

end number_of_rows_seating_exactly_9_students_l180_180437


namespace nuts_distributive_problem_l180_180139

theorem nuts_distributive_problem (x y : ℕ) (h1 : 70 ≤ x + y) (h2 : x + y ≤ 80) (h3 : (3 / 4 : ℚ) * x + (1 / 5 : ℚ) * (y + (1 / 4 : ℚ) * x) = (x : ℚ) + 1) :
  x = 36 ∧ y = 41 :=
by
  sorry

end nuts_distributive_problem_l180_180139


namespace intersection_ab_correct_l180_180667

noncomputable def set_A : Set ℝ := { x : ℝ | x > 1/3 }
def set_B : Set ℝ := { x : ℝ | ∃ y : ℝ, x^2 + y^2 = 4 ∧ y ≥ -2 ∧ y ≤ 2 }
def intersection_AB : Set ℝ := { x : ℝ | 1/3 < x ∧ x ≤ 2 }

theorem intersection_ab_correct : set_A ∩ set_B = intersection_AB := 
by 
  -- proof omitted
  sorry

end intersection_ab_correct_l180_180667


namespace negation_of_p_l180_180061

-- Define the proposition p
def p : Prop := ∃ n : ℕ, 2^n > 100

-- Goal is to show the negation of p
theorem negation_of_p : (¬ p) = (∀ n : ℕ, 2^n ≤ 100) :=
by
  sorry

end negation_of_p_l180_180061


namespace roots_square_sum_eq_l180_180846

theorem roots_square_sum_eq (r s t p q : ℝ) 
  (h1 : r + s + t = p) 
  (h2 : r * s + r * t + s * t = q) 
  (h3 : r * s * t = r) :
  r^2 + s^2 + t^2 = p^2 - 2 * q :=
by
  sorry

end roots_square_sum_eq_l180_180846


namespace composite_10201_base_n_composite_10101_base_n_l180_180487

-- 1. Prove that 10201_n is composite given n > 2
theorem composite_10201_base_n (n : ℕ) (h : n > 2) : 
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n^4 + 2*n^2 + 1 := 
sorry

-- 2. Prove that 10101_n is composite given n > 2.
theorem composite_10101_base_n (n : ℕ) (h : n > 2) : 
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n^4 + n^2 + 1 := 
sorry

end composite_10201_base_n_composite_10101_base_n_l180_180487


namespace major_axis_length_l180_180938

-- Define the radius of the cylinder
def cylinder_radius : ℝ := 2

-- Define the relationship given in the problem
def major_axis_ratio : ℝ := 1.6

-- Define the calculation for minor axis
def minor_axis : ℝ := 2 * cylinder_radius

-- Define the calculation for major axis
def major_axis : ℝ := major_axis_ratio * minor_axis

-- The theorem statement
theorem major_axis_length:
  major_axis = 6.4 :=
by 
  sorry -- Proof to be provided later

end major_axis_length_l180_180938


namespace find_r_cubed_l180_180936

theorem find_r_cubed (r : ℝ) (h : (r + 1/r)^2 = 5) : r^3 + 1/r^3 = 2 * Real.sqrt 5 :=
by
  sorry

end find_r_cubed_l180_180936


namespace cost_of_fencing_correct_l180_180806

noncomputable def cost_of_fencing (d : ℝ) (r : ℝ) : ℝ :=
  Real.pi * d * r

theorem cost_of_fencing_correct : cost_of_fencing 30 5 = 471 :=
by
  sorry

end cost_of_fencing_correct_l180_180806


namespace carrots_picked_by_mother_l180_180164

-- Define the conditions
def faye_picked : ℕ := 23
def good_carrots : ℕ := 12
def bad_carrots : ℕ := 16

-- Define the problem of the total number of carrots
def total_carrots : ℕ := good_carrots + bad_carrots

-- Define the mother's picked carrots
def mother_picked (total_faye : ℕ) (total : ℕ) := total - total_faye

-- State the theorem
theorem carrots_picked_by_mother (faye_picked : ℕ) (total_carrots : ℕ) : mother_picked faye_picked total_carrots = 5 := by
  sorry

end carrots_picked_by_mother_l180_180164


namespace dvd_sold_168_l180_180176

/-- 
Proof that the number of DVDs sold (D) is 168 given the conditions:
1) D = 1.6 * C
2) D + C = 273 
-/
theorem dvd_sold_168 (C D : ℝ) (h1 : D = 1.6 * C) (h2 : D + C = 273) : D = 168 := 
sorry

end dvd_sold_168_l180_180176


namespace total_frogs_in_pond_l180_180971

def frogsOnLilyPads : ℕ := 5
def frogsOnLogs : ℕ := 3
def babyFrogsOnRock : ℕ := 2 * 12 -- Two dozen

theorem total_frogs_in_pond : frogsOnLilyPads + frogsOnLogs + babyFrogsOnRock = 32 :=
by
  sorry

end total_frogs_in_pond_l180_180971


namespace days_to_complete_work_l180_180005

theorem days_to_complete_work :
  ∀ (M B: ℝ) (D: ℝ),
    (M = 2 * B)
    → (13 * M + 24 * B) * 4 = (12 * M + 16 * B) * D
    → D = 5 :=
by
  intros M B D h1 h2
  sorry

end days_to_complete_work_l180_180005


namespace find_k_l180_180734

theorem find_k (a b c k : ℤ)
  (g : ℤ → ℤ)
  (h1 : ∀ x, g x = a * x^2 + b * x + c)
  (h2 : g 2 = 0)
  (h3 : 60 < g 6 ∧ g 6 < 70)
  (h4 : 90 < g 9 ∧ g 9 < 100)
  (h5 : 10000 * k < g 50 ∧ g 50 < 10000 * (k + 1)) :
  k = 0 :=
sorry

end find_k_l180_180734


namespace find_number_x_l180_180639

theorem find_number_x (x : ℝ) (h : 2500 - x / 20.04 = 2450) : x = 1002 :=
by
  -- Proof can be written here, but skipped by using sorry
  sorry

end find_number_x_l180_180639


namespace find_a_l180_180694

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 + 2

theorem find_a (a : ℝ) (h : deriv (f a) (-1) = 4) : a = 10 / 3 :=
by {
  sorry
}

end find_a_l180_180694


namespace sphere_surface_area_quadruple_l180_180257

theorem sphere_surface_area_quadruple (r : ℝ) :
  (4 * π * (2 * r)^2) = 4 * (4 * π * r^2) :=
by
  sorry

end sphere_surface_area_quadruple_l180_180257


namespace solution_triple_root_system_l180_180569

theorem solution_triple_root_system (x y z : ℝ) :
  (x - 1) * (y - 1) * (z - 1) = x * y * z - 1 ∧
  (x - 2) * (y - 2) * (z - 2) = x * y * z - 2 →
  x = 1 ∧ y = 1 ∧ z = 1 :=
by
  intro h
  sorry

end solution_triple_root_system_l180_180569


namespace cost_per_metre_of_carpet_l180_180653

theorem cost_per_metre_of_carpet :
  (length_of_room = 18) →
  (breadth_of_room = 7.5) →
  (carpet_width = 0.75) →
  (total_cost = 810) →
  (cost_per_metre = 4.5) :=
by
  intros length_of_room breadth_of_room carpet_width total_cost
  sorry

end cost_per_metre_of_carpet_l180_180653


namespace trajectory_is_parabola_l180_180485

theorem trajectory_is_parabola (C : ℝ × ℝ) (M : ℝ × ℝ) (l : ℝ → ℝ)
  (hM : M = (0, 3)) (hl : ∀ y, l y = -3)
  (h : dist C M = |C.2 + 3|) : C.1^2 = 12 * C.2 := by
  sorry

end trajectory_is_parabola_l180_180485


namespace sequence_problem_l180_180851

theorem sequence_problem
  (a1 a2 b1 b2 b3 : ℝ)
  (h1 : 1 + a1 + a1 = a1 + a1)
  (h2 : b1 * b1 = b2)
  (h3 : 4 = b2 * b2):
  (a1 + a2) / b2 = 2 :=
by
  -- The proof would go here
  sorry

end sequence_problem_l180_180851


namespace find_b_eq_neg_three_l180_180082

theorem find_b_eq_neg_three (b : ℝ) (h : (2 - b) / 5 = -(2 * b + 1) / 5) : b = -3 :=
by
  sorry

end find_b_eq_neg_three_l180_180082


namespace commission_rate_l180_180101

theorem commission_rate (old_salary new_base_salary sale_amount : ℝ) (required_sales : ℕ) (condition: (old_salary = 75000) ∧ (new_base_salary = 45000) ∧ (sale_amount = 750) ∧ (required_sales = 267)) :
  ∃ commission_rate : ℝ, abs (commission_rate - 0.14981) < 0.0001 :=
by
  sorry

end commission_rate_l180_180101


namespace find_k_l180_180339

variable (m n k : ℚ)

def line_eq (x y : ℚ) : Prop := x - (5/2 : ℚ) * y + 1 = 0

theorem find_k (h1 : line_eq m n) (h2 : line_eq (m + 1/2) (n + 1/k)) : k = 3/5 := by
  sorry

end find_k_l180_180339


namespace containers_needed_l180_180159

-- Define the conditions: 
def weight_in_pounds : ℚ := 25 / 2
def ounces_per_pound : ℚ := 16
def ounces_per_container : ℚ := 50

-- Define the total weight in ounces
def total_weight_in_ounces := weight_in_pounds * ounces_per_pound

-- Theorem statement: Number of containers.
theorem containers_needed : total_weight_in_ounces / ounces_per_container = 4 := 
by
  -- Write the proof here
  sorry

end containers_needed_l180_180159


namespace simplify_expression_l180_180415

variables {R : Type*} [CommRing R]

theorem simplify_expression (x y : R) : (3 * x^2 * y^3)^4 = 81 * x^8 * y^12 := by
  sorry

end simplify_expression_l180_180415


namespace line_does_not_pass_second_quadrant_l180_180625

theorem line_does_not_pass_second_quadrant (a : ℝ) :
  (∀ x y : ℝ, (a + 1) * x + y + 2 - a = 0 → ¬(x < 0 ∧ y > 0)) ↔ a ≤ -1 :=
by
  sorry

end line_does_not_pass_second_quadrant_l180_180625


namespace teacher_engineer_ratio_l180_180696

-- Define the context with the given conditions
variable (t e : ℕ)

-- Conditions
def avg_age (t e : ℕ) : Prop := (40 * t + 55 * e) / (t + e) = 45

-- The statement to be proved
theorem teacher_engineer_ratio
  (h : avg_age t e) :
  t / e = 2 := sorry

end teacher_engineer_ratio_l180_180696


namespace ab_value_l180_180977

theorem ab_value (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end ab_value_l180_180977


namespace more_oil_l180_180796

noncomputable def original_price (P : ℝ) :=
  P - 0.3 * P = 70

noncomputable def amount_of_oil_before (P : ℝ) :=
  700 / P

noncomputable def amount_of_oil_after :=
  700 / 70

theorem more_oil (P : ℝ) (h1 : original_price P) :
  (amount_of_oil_after - amount_of_oil_before P) = 3 :=
  sorry

end more_oil_l180_180796


namespace intersection_of_sets_l180_180661

-- Define the sets M and N
def M : Set ℝ := { x | 2 < x ∧ x < 3 }
def N : Set ℝ := { x | 2 < x ∧ x ≤ 5 / 2 }

-- State the theorem to prove
theorem intersection_of_sets : M ∩ N = { x | 2 < x ∧ x ≤ 5 / 2 } :=
by 
  sorry

end intersection_of_sets_l180_180661


namespace num_common_elements_1000_multiples_5_9_l180_180381

def multiples_up_to (n k : ℕ) : ℕ := n / k

def num_common_elements_in_sets (k m n : ℕ) : ℕ :=
  multiples_up_to n (Nat.lcm k m)

theorem num_common_elements_1000_multiples_5_9 :
  num_common_elements_in_sets 5 9 5000 = 111 :=
by
  -- The proof is omitted as per instructions
  sorry

end num_common_elements_1000_multiples_5_9_l180_180381


namespace gwen_math_problems_l180_180755

-- Problem statement
theorem gwen_math_problems (m : ℕ) (science_problems : ℕ := 11) (problems_finished_at_school : ℕ := 24) (problems_left_for_homework : ℕ := 5) 
  (h1 : m + science_problems = problems_finished_at_school + problems_left_for_homework) : m = 18 := 
by {
  sorry
}

end gwen_math_problems_l180_180755


namespace smaller_cylinder_diameter_l180_180513

theorem smaller_cylinder_diameter
  (vol_large : ℝ)
  (height_large : ℝ)
  (diameter_large : ℝ)
  (height_small : ℝ)
  (ratio : ℝ)
  (π : ℝ)
  (volume_large_eq : vol_large = π * (diameter_large / 2)^2 * height_large)  -- Volume formula for the larger cylinder
  (ratio_eq : ratio = 74.07407407407408) -- Given ratio
  (height_large_eq : height_large = 10)  -- Given height of the larger cylinder
  (diameter_large_eq : diameter_large = 20)  -- Given diameter of the larger cylinder
  (height_small_eq : height_small = 6)  -- Given height of smaller cylinders):
  :
  ∃ (diameter_small : ℝ), diameter_small = 3 := 
by
  sorry

end smaller_cylinder_diameter_l180_180513


namespace three_g_two_plus_two_g_neg_four_l180_180004

def g (x : ℝ) : ℝ := 2 * x ^ 2 - 2 * x + 11

theorem three_g_two_plus_two_g_neg_four : 3 * g 2 + 2 * g (-4) = 147 := by
  sorry

end three_g_two_plus_two_g_neg_four_l180_180004


namespace additional_time_due_to_leak_l180_180494

theorem additional_time_due_to_leak 
  (normal_time_per_barrel : ℕ)
  (leak_time_per_barrel : ℕ)
  (barrels : ℕ)
  (normal_duration : normal_time_per_barrel = 3)
  (leak_duration : leak_time_per_barrel = 5)
  (barrels_needed : barrels = 12) :
  (leak_time_per_barrel * barrels - normal_time_per_barrel * barrels) = 24 := 
by
  sorry

end additional_time_due_to_leak_l180_180494


namespace find_value_x_y_cube_l180_180820

variables (x y k c m : ℝ)

theorem find_value_x_y_cube
  (h1 : x^3 * y^3 = k)
  (h2 : 1 / x^3 + 1 / y^3 = c)
  (h3 : x + y = m) :
  (x + y)^3 = c * k + 3 * k^(1/3) * m :=
by
  sorry

end find_value_x_y_cube_l180_180820


namespace mary_needs_more_sugar_l180_180051

def recipe_sugar := 14
def sugar_already_added := 2
def sugar_needed := recipe_sugar - sugar_already_added

theorem mary_needs_more_sugar : sugar_needed = 12 := by
  sorry

end mary_needs_more_sugar_l180_180051


namespace calculate_b_l180_180013

open Real

theorem calculate_b (b : ℝ) (h : ∫ x in e..b, 2 / x = 6) : b = exp 4 := 
sorry

end calculate_b_l180_180013


namespace A_beats_B_by_40_meters_l180_180855

-- Definitions based on conditions
def distance_A := 1000 -- Distance in meters
def time_A := 240      -- Time in seconds
def time_diff := 10      -- Time difference in seconds

-- Intermediate calculations
def velocity_A : ℚ := distance_A / time_A
def time_B := time_A + time_diff
def velocity_B : ℚ := distance_A / time_B

-- Distance B covers in 240 seconds
def distance_B_in_240 : ℚ := velocity_B * time_A

-- Proof goal
theorem A_beats_B_by_40_meters : (distance_A - distance_B_in_240 = 40) :=
by
  -- Insert actual steps to prove here
  sorry

end A_beats_B_by_40_meters_l180_180855


namespace parametric_curve_intersects_l180_180703

noncomputable def curve_crosses_itself : Prop :=
  let t1 := Real.sqrt 11
  let t2 := -Real.sqrt 11
  let x (t : ℝ) := t^3 - t + 1
  let y (t : ℝ) := t^3 - 11*t + 11
  (x t1 = 10 * Real.sqrt 11 + 1) ∧ (y t1 = 11) ∧
  (x t2 = 10 * Real.sqrt 11 + 1) ∧ (y t2 = 11)

theorem parametric_curve_intersects : curve_crosses_itself :=
by
  sorry

end parametric_curve_intersects_l180_180703


namespace fraction_inequality_l180_180370

variable (a b c : ℝ)

theorem fraction_inequality (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : c > a) (h5 : a > b) :
  (a / (c - a)) > (b / (c - b)) := 
sorry

end fraction_inequality_l180_180370


namespace john_splits_profit_correctly_l180_180359

-- Conditions
def total_cookies : ℕ := 6 * 12
def revenue_per_cookie : ℝ := 1.5
def cost_per_cookie : ℝ := 0.25
def amount_per_charity : ℝ := 45

-- Computations based on conditions
def total_revenue : ℝ := total_cookies * revenue_per_cookie
def total_cost : ℝ := total_cookies * cost_per_cookie
def total_profit : ℝ := total_revenue - total_cost

-- Proof statement
theorem john_splits_profit_correctly : total_profit / amount_per_charity = 2 := by
  sorry

end john_splits_profit_correctly_l180_180359


namespace cubic_root_expression_l180_180722

theorem cubic_root_expression (u v w : ℂ) (huvwx : u * v * w ≠ 0)
  (h1 : u^3 - 6 * u^2 + 11 * u - 6 = 0)
  (h2 : v^3 - 6 * v^2 + 11 * v - 6 = 0)
  (h3 : w^3 - 6 * w^2 + 11 * w - 6 = 0) :
  (u * v / w) + (v * w / u) + (w * u / v) = 49 / 6 :=
sorry

end cubic_root_expression_l180_180722


namespace initial_weight_of_load_l180_180254

variable (W : ℝ)
variable (h : 0.8 * 0.9 * W = 36000)

theorem initial_weight_of_load :
  W = 50000 :=
by
  sorry

end initial_weight_of_load_l180_180254


namespace find_number_l180_180948

def condition (x : ℤ) : Prop := 3 * (x + 8) = 36

theorem find_number (x : ℤ) (h : condition x) : x = 4 := by
  sorry

end find_number_l180_180948


namespace initial_birds_count_l180_180712

theorem initial_birds_count (B : ℕ) (h1 : 6 = B + 3 + 1) : B = 2 :=
by
  -- Placeholder for the proof, we are not required to provide it here.
  sorry

end initial_birds_count_l180_180712


namespace quadratic_solution_eq_l180_180633

noncomputable def p : ℝ :=
  (8 + Real.sqrt 364) / 10

noncomputable def q : ℝ :=
  (8 - Real.sqrt 364) / 10

theorem quadratic_solution_eq (p q : ℝ) (h₁ : 5 * p^2 - 8 * p - 15 = 0) (h₂ : 5 * q^2 - 8 * q - 15 = 0) : 
  (p - q) ^ 2 = 14.5924 :=
sorry

end quadratic_solution_eq_l180_180633


namespace bisecting_chord_line_eqn_l180_180537

theorem bisecting_chord_line_eqn :
  ∀ (x1 y1 x2 y2 : ℝ),
  y1 ^ 2 = 16 * x1 →
  y2 ^ 2 = 16 * x2 →
  (x1 + x2) / 2 = 2 →
  (y1 + y2) / 2 = 1 →
  ∃ (a b c : ℝ), a = 8 ∧ b = -1 ∧ c = -15 ∧
  ∀ (x y : ℝ), y = 8 * x - 15 → a * x + b * y + c = 0 :=
by 
  sorry

end bisecting_chord_line_eqn_l180_180537


namespace percentage_meetings_correct_l180_180441

def work_day_hours : ℕ := 10
def minutes_in_hour : ℕ := 60
def total_work_day_minutes := work_day_hours * minutes_in_hour

def lunch_break_minutes : ℕ := 30
def effective_work_day_minutes := total_work_day_minutes - lunch_break_minutes

def first_meeting_minutes : ℕ := 60
def second_meeting_minutes := 3 * first_meeting_minutes
def total_meeting_minutes := first_meeting_minutes + second_meeting_minutes

def percentage_of_day_spent_in_meetings := (total_meeting_minutes * 100) / effective_work_day_minutes

theorem percentage_meetings_correct : percentage_of_day_spent_in_meetings = 42 := 
by
  sorry

end percentage_meetings_correct_l180_180441


namespace area_of_rectangle_is_270_l180_180856

noncomputable def side_of_square := Real.sqrt 2025

noncomputable def radius_of_circle := side_of_square

noncomputable def length_of_rectangle := (2/5 : ℝ) * radius_of_circle

noncomputable def initial_breadth_of_rectangle := (1/2 : ℝ) * length_of_rectangle + 5

noncomputable def breadth_of_rectangle := if (length_of_rectangle + initial_breadth_of_rectangle) % 3 = 0 
                                          then initial_breadth_of_rectangle 
                                          else initial_breadth_of_rectangle + 1

noncomputable def area_of_rectangle := length_of_rectangle * breadth_of_rectangle

theorem area_of_rectangle_is_270 :
  area_of_rectangle = 270 := by
  sorry

end area_of_rectangle_is_270_l180_180856


namespace expression_evaluation_l180_180133

theorem expression_evaluation (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : x^2 = 1 / y^2) :
  (x^2 - 4 / x^2) * (y^2 + 4 / y^2) = x^4 - 16 / x^4 :=
by
  sorry

end expression_evaluation_l180_180133


namespace pie_cost_correct_l180_180337

-- Define the initial and final amounts of money Mary had.
def initial_amount : ℕ := 58
def final_amount : ℕ := 52

-- Define the cost of the pie as the difference between initial and final amounts.
def pie_cost : ℕ := initial_amount - final_amount

-- State the theorem that given the initial and final amounts, the cost of the pie is 6.
theorem pie_cost_correct : pie_cost = 6 := by 
  sorry

end pie_cost_correct_l180_180337


namespace mystical_words_count_l180_180265

-- We define a function to count words given the conditions
def count_possible_words : ℕ := 
  let total_words : ℕ := (20^1 - 19^1) + (20^2 - 19^2) + (20^3 - 19^3) + (20^4 - 19^4) + (20^5 - 19^5)
  total_words

theorem mystical_words_count : count_possible_words = 755761 :=
by 
  unfold count_possible_words
  sorry

end mystical_words_count_l180_180265


namespace product_of_fractions_l180_180326

theorem product_of_fractions : (2 / 9) * (5 / 11) = 10 / 99 := 
by
  sorry

end product_of_fractions_l180_180326


namespace probability_of_intersecting_diagonals_l180_180038

def num_vertices := 8
def total_diagonals := (num_vertices * (num_vertices - 3)) / 2
def total_ways_to_choose_two_diagonals := Nat.choose total_diagonals 2
def ways_to_choose_4_vertices := Nat.choose num_vertices 4
def number_of_intersecting_pairs := ways_to_choose_4_vertices
def probability_intersecting_diagonals := (number_of_intersecting_pairs : ℚ) / (total_ways_to_choose_two_diagonals : ℚ)

theorem probability_of_intersecting_diagonals :
  probability_intersecting_diagonals = 7 / 19 := by
  sorry

end probability_of_intersecting_diagonals_l180_180038


namespace trigonometric_identity_l180_180314

variable (A B C a b c : ℝ)
variable (h_triangle : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2)
variable (h_sides : a > 0 ∧ b > 0 ∧ c > 0)
variable (h_sum_angles : A + B + C = π)
variable (h_condition : (c / b) + (b / c) = (5 * Real.cos A) / 2)

theorem trigonometric_identity 
  (h_triangle_acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2) 
  (h_sides_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum_angles_eq : A + B + C = π) 
  (h_given : (c / b) + (b / c) = (5 * Real.cos A) / 2) : 
  (Real.tan A / Real.tan B) + (Real.tan A / Real.tan C) = 1/2 :=
by
  sorry

end trigonometric_identity_l180_180314


namespace ganesh_average_speed_l180_180616

noncomputable def averageSpeed (D : ℝ) : ℝ :=
  let time_uphill := D / 60
  let time_downhill := D / 36
  let total_time := time_uphill + time_downhill
  let total_distance := 2 * D
  total_distance / total_time

theorem ganesh_average_speed (D : ℝ) (hD : D > 0) : averageSpeed D = 45 := by
  sorry

end ganesh_average_speed_l180_180616


namespace finance_charge_rate_l180_180075

theorem finance_charge_rate (original_balance total_payment finance_charge_rate : ℝ)
    (h1 : original_balance = 150)
    (h2 : total_payment = 153)
    (h3 : finance_charge_rate = ((total_payment - original_balance) / original_balance) * 100) :
    finance_charge_rate = 2 :=
by
  rw [h1, h2] at h3
  norm_num at h3
  exact h3

end finance_charge_rate_l180_180075


namespace josie_animal_counts_l180_180867

/-- Josie counted 80 antelopes, 34 more rabbits than antelopes, 42 fewer hyenas than 
the total number of antelopes and rabbits combined, some more wild dogs than hyenas, 
and the number of leopards was half the number of rabbits. The total number of animals 
Josie counted was 605. Prove that the difference between the number of wild dogs 
and hyenas Josie counted is 50. -/
theorem josie_animal_counts :
  ∃ (antelopes rabbits hyenas wild_dogs leopards : ℕ),
    antelopes = 80 ∧
    rabbits = antelopes + 34 ∧
    hyenas = (antelopes + rabbits) - 42 ∧
    leopards = rabbits / 2 ∧
    (antelopes + rabbits + hyenas + wild_dogs + leopards = 605) ∧
    wild_dogs - hyenas = 50 := 
by
  sorry

end josie_animal_counts_l180_180867


namespace factorize_expression_l180_180980

variable {a b : ℕ}

theorem factorize_expression (h : 6 * a^2 * b - 3 * a * b = 3 * a * b * (2 * a - 1)) : 6 * a^2 * b - 3 * a * b = 3 * a * b * (2 * a - 1) :=
by sorry

end factorize_expression_l180_180980


namespace parabola_value_f_l180_180540

theorem parabola_value_f (d e f : ℝ) :
  (∀ y : ℝ, x = d * y ^ 2 + e * y + f) →
  (∀ x y : ℝ, (x + 3) = d * (y - 1) ^ 2) →
  (x = -1 ∧ y = 3) →
  y = 0 →
  f = -2.5 :=
sorry

end parabola_value_f_l180_180540


namespace arnold_total_protein_l180_180596

-- Definitions and conditions
def collagen_protein_per_scoop : ℕ := 18 / 2
def protein_powder_protein_per_scoop : ℕ := 21
def steak_protein : ℕ := 56

def collagen_scoops : ℕ := 1
def protein_powder_scoops : ℕ := 1
def steaks : ℕ := 1

-- Statement of the theorem/problem
theorem arnold_total_protein : 
  (collagen_protein_per_scoop * collagen_scoops) + 
  (protein_powder_protein_per_scoop * protein_powder_scoops) + 
  (steak_protein * steaks) = 86 :=
by
  sorry

end arnold_total_protein_l180_180596


namespace total_mail_l180_180392

def monday_mail := 65
def tuesday_mail := monday_mail + 10
def wednesday_mail := tuesday_mail - 5
def thursday_mail := wednesday_mail + 15

theorem total_mail : 
  monday_mail + tuesday_mail + wednesday_mail + thursday_mail = 295 := by
  sorry

end total_mail_l180_180392


namespace max_value_of_ex1_ex2_l180_180908

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 0 then exp x else -(x^3)

-- Define the function g
noncomputable def g (x a : ℝ) : ℝ := 
  f (f x) - a

-- Define the condition that g(x) = 0 has two distinct zeros
def has_two_distinct_zeros (a : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ g x1 a = 0 ∧ g x2 a = 0

-- Define the target function h
noncomputable def h (m : ℝ) : ℝ := 
  m^3 * exp (-m)

-- Statement of the final proof
theorem max_value_of_ex1_ex2 (a : ℝ) (hpos : 0 < a) (zeros : has_two_distinct_zeros a) :
  (∃ x1 x2 : ℝ, e^x1 * e^x2 = (27 : ℝ) / (exp 3) ∧ g x1 a = 0 ∧ g x2 a = 0) :=
sorry

end max_value_of_ex1_ex2_l180_180908


namespace find_d_l180_180358

noncomputable def f (x : ℝ) (a b c d : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + c*x + d

theorem find_d (a b c d : ℝ) (roots_negative_integers : ∀ x, f x a b c d = 0 → x < 0) (sum_is_2023 : a + b + c + d = 2023) :
  d = 17020 :=
sorry

end find_d_l180_180358


namespace gcd_13924_27018_l180_180907

theorem gcd_13924_27018 : Int.gcd 13924 27018 = 2 := 
  by
    sorry

end gcd_13924_27018_l180_180907


namespace find_abc_l180_180863

theorem find_abc (a b c : ℝ) 
  (h1 : 2 * b = a + c)  -- a, b, c form an arithmetic sequence
  (h2 : a + b + c = 12) -- The sum of a, b, and c is 12
  (h3 : (b + 2)^2 = (a + 2) * (c + 5)) -- a+2, b+2, and c+5 form a geometric sequence
: (a = 1 ∧ b = 4 ∧ c = 7) ∨ (a = 10 ∧ b = 4 ∧ c = -2) :=
sorry

end find_abc_l180_180863


namespace village_Y_initial_population_l180_180992

def population_X := 76000
def decrease_rate_X := 1200
def increase_rate_Y := 800
def years := 17

def population_X_after_17_years := population_X - decrease_rate_X * years
def population_Y_after_17_years (P : Nat) := P + increase_rate_Y * years

theorem village_Y_initial_population (P : Nat) (h : population_Y_after_17_years P = population_X_after_17_years) : P = 42000 :=
by
  sorry

end village_Y_initial_population_l180_180992


namespace smallest_n_l180_180429

theorem smallest_n (n : ℕ) : 
  (∃ k : ℕ, 4 * n = k^2) ∧ (∃ l : ℕ, 5 * n = l^5) ↔ n = 625 :=
by sorry

end smallest_n_l180_180429


namespace bob_correct_answers_l180_180298

-- Define the variables, c for correct answers, w for incorrect answers, total problems 15, score 54
variables (c w : ℕ)

-- Define the conditions
axiom total_problems : c + w = 15
axiom total_score : 6 * c - 3 * w = 54

-- Prove that the number of correct answers is 11
theorem bob_correct_answers : c = 11 :=
by
  -- Here, you would provide the proof, but for the sake of the statement, we'll use sorry.
  sorry

end bob_correct_answers_l180_180298


namespace emily_points_l180_180641

theorem emily_points (r1 r2 r3 r4 r5 m4 m5 l : ℤ)
  (h1 : r1 = 16)
  (h2 : r2 = 33)
  (h3 : r3 = 21)
  (h4 : r4 = 10)
  (h5 : r5 = 4)
  (hm4 : m4 = 2)
  (hm5 : m5 = 3)
  (hl : l = 48) :
  r1 + r2 + r3 + r4 * m4 + r5 * m5 - l = 54 := by
  sorry

end emily_points_l180_180641


namespace productivity_increase_is_233_33_percent_l180_180957

noncomputable def productivity_increase :
  Real :=
  let B := 1 -- represents the base number of bears made per week
  let H := 1 -- represents the base number of hours worked per week
  let P := B / H -- base productivity in bears per hour

  let B1 := 1.80 * B -- bears per week with first assistant
  let H1 := 0.90 * H -- hours per week with first assistant
  let P1 := B1 / H1 -- productivity with first assistant

  let B2 := 1.60 * B -- bears per week with second assistant
  let H2 := 0.80 * H -- hours per week with second assistant
  let P2 := B2 / H2 -- productivity with second assistant

  let B_both := B1 + B2 - B -- total bears with both assistants
  let H_both := H1 * H2 / H -- total hours with both assistants
  let P_both := B_both / H_both -- productivity with both assistants

  (P_both / P - 1) * 100

theorem productivity_increase_is_233_33_percent :
  productivity_increase = 233.33 :=
by
  sorry

end productivity_increase_is_233_33_percent_l180_180957


namespace prime_gt_p_l180_180036

theorem prime_gt_p (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hgt : q > 5) (hdiv : q ∣ 2^p + 3^p) : q > p := 
sorry

end prime_gt_p_l180_180036


namespace restoration_of_axes_l180_180701

theorem restoration_of_axes (parabola : ℝ → ℝ) (h : ∀ x, parabola x = x^2) : 
  ∃ (origin : ℝ × ℝ) (x_axis y_axis : ℝ × ℝ → Prop), 
    (∀ x, x_axis (x, 0)) ∧ 
    (∀ y, y_axis (0, y)) ∧ 
    origin = (0, 0) := 
sorry

end restoration_of_axes_l180_180701


namespace f_2_2_eq_7_f_3_3_eq_61_f_4_4_can_be_evaluated_l180_180603

noncomputable def f : ℕ → ℕ → ℕ
| 0, y => y + 1
| (x + 1), 0 => f x 1
| (x + 1), (y + 1) => f x (f (x + 1) y)

theorem f_2_2_eq_7 : f 2 2 = 7 :=
sorry

theorem f_3_3_eq_61 : f 3 3 = 61 :=
sorry

theorem f_4_4_can_be_evaluated : ∃ n, f 4 4 = n :=
sorry

end f_2_2_eq_7_f_3_3_eq_61_f_4_4_can_be_evaluated_l180_180603


namespace fuel_cost_is_50_cents_l180_180880

-- Define the capacities of the tanks
def small_tank_capacity : ℕ := 60
def large_tank_capacity : ℕ := 60 * 3 / 2 -- 50% larger than small tank

-- Define the number of planes
def number_of_small_planes : ℕ := 2
def number_of_large_planes : ℕ := 2

-- Define the service charge per plane
def service_charge_per_plane : ℕ := 100
def total_service_charge : ℕ :=
  service_charge_per_plane * (number_of_small_planes + number_of_large_planes)

-- Define the total cost to fill all planes
def total_cost : ℕ := 550

-- Define the total fuel capacity
def total_fuel_capacity : ℕ :=
  number_of_small_planes * small_tank_capacity + number_of_large_planes * large_tank_capacity

-- Define the total fuel cost
def total_fuel_cost : ℕ := total_cost - total_service_charge

-- Define the fuel cost per liter
def fuel_cost_per_liter : ℕ :=
  total_fuel_cost / total_fuel_capacity

theorem fuel_cost_is_50_cents :
  fuel_cost_per_liter = 50 / 100 := by
sorry

end fuel_cost_is_50_cents_l180_180880


namespace square_side_length_equals_5_sqrt_pi_l180_180573

theorem square_side_length_equals_5_sqrt_pi :
  ∃ s : ℝ, ∃ r : ℝ, (r = 5) ∧ (s = 2 * r) ∧ (s ^ 2 = 25 * π) ∧ (s = 5 * Real.sqrt π) :=
by
  sorry

end square_side_length_equals_5_sqrt_pi_l180_180573


namespace GouguPrinciple_l180_180651

-- Definitions according to conditions
def volumes_not_equal (A B : Type) : Prop := sorry -- p: volumes of A and B are not equal
def cross_sections_not_equal (A B : Type) : Prop := sorry -- q: cross-sectional areas of A and B are not always equal

-- The theorem to be proven
theorem GouguPrinciple (A B : Type) (h1 : volumes_not_equal A B) : cross_sections_not_equal A B :=
sorry

end GouguPrinciple_l180_180651


namespace inequality_proof_l180_180637

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a / Real.sqrt (a^2 + 8 * b * c)) + 
  (b / Real.sqrt (b^2 + 8 * c * a)) + 
  (c / Real.sqrt (c^2 + 8 * a * b)) ≥ 1 :=
by
  sorry

end inequality_proof_l180_180637


namespace continuity_sum_l180_180244

noncomputable def piecewise_function (x : ℝ) (a b c : ℝ) : ℝ :=
if h : x > 1 then a * (2 * x + 1) + 2
else if h' : -1 <= x && x <= 1 then b * x + 3
else 3 * x - c

theorem continuity_sum (a b c : ℝ) (h_cont1 : 3 * a = b + 1) (h_cont2 : c = 3 * a + 1) :
  a + c = 4 * a + 1 :=
by
  sorry

end continuity_sum_l180_180244


namespace supplements_delivered_l180_180017

-- Define the conditions as given in the problem
def total_medicine_boxes : ℕ := 760
def vitamin_boxes : ℕ := 472

-- Define the number of supplement boxes
def supplement_boxes : ℕ := total_medicine_boxes - vitamin_boxes

-- State the theorem to be proved
theorem supplements_delivered : supplement_boxes = 288 :=
by
  -- The actual proof is not required, so we use "sorry"
  sorry

end supplements_delivered_l180_180017


namespace prove_collinear_prove_perpendicular_l180_180747

noncomputable def vec_a : ℝ × ℝ := (1, 3)
noncomputable def vec_b : ℝ × ℝ := (3, -4)

def collinear (k : ℝ) : Prop :=
  let v1 := (k * 1 - 3, k * 3 + 4)
  let v2 := (1 + 3, 3 - 4)
  v1.1 * v2.2 = v1.2 * v2.1

def perpendicular (k : ℝ) : Prop :=
  let v1 := (k * 1 - 3, k * 3 + 4)
  let v2 := (1 + 3, 3 - 4)
  v1.1 * v2.1 + v1.2 * v2.2 = 0

theorem prove_collinear : collinear (-1) :=
by
  sorry

theorem prove_perpendicular : perpendicular (16) :=
by
  sorry

end prove_collinear_prove_perpendicular_l180_180747


namespace ratio_of_ages_l180_180340

theorem ratio_of_ages (F C : ℕ) (h1 : F = C) (h2 : F = 75) :
  (C + 5 * 15) / (F + 15) = 5 / 3 :=
by
  sorry

end ratio_of_ages_l180_180340


namespace harold_car_payment_l180_180802

variables (C : ℝ)

noncomputable def harold_income : ℝ := 2500
noncomputable def rent : ℝ := 700
noncomputable def groceries : ℝ := 50
noncomputable def remaining_after_retirement : ℝ := 1300

-- Harold's utility cost is half his car payment
noncomputable def utilities (C : ℝ) : ℝ := C / 2

-- Harold's total expenses.
noncomputable def total_expenses (C : ℝ) : ℝ := rent + C + utilities C + groceries

-- Proving that Harold’s car payment \(C\) can be calculated with the remaining money
theorem harold_car_payment : (2500 - total_expenses C = 1300) → (C = 300) :=
by 
  sorry

end harold_car_payment_l180_180802


namespace least_number_to_subtract_l180_180192

theorem least_number_to_subtract (n m : ℕ) (h : n = 56783421) (d : m = 569) : (n % m) = 56783421 % 569 := 
by sorry

end least_number_to_subtract_l180_180192


namespace dogwood_trees_tomorrow_l180_180132

def initial_dogwood_trees : Nat := 7
def trees_planted_today : Nat := 3
def final_total_dogwood_trees : Nat := 12

def trees_after_today : Nat := initial_dogwood_trees + trees_planted_today
def trees_planted_tomorrow : Nat := final_total_dogwood_trees - trees_after_today

theorem dogwood_trees_tomorrow :
  trees_planted_tomorrow = 2 :=
by
  sorry

end dogwood_trees_tomorrow_l180_180132


namespace quadratic_solution_l180_180354

theorem quadratic_solution :
  ∀ x : ℝ, x^2 = 4 ↔ x = 2 ∨ x = -2 := sorry

end quadratic_solution_l180_180354


namespace unique_mod_inverse_l180_180717

theorem unique_mod_inverse (a n : ℤ) (coprime : Int.gcd a n = 1) : 
  ∃! b : ℤ, (a * b) % n = 1 % n := 
sorry

end unique_mod_inverse_l180_180717


namespace swimmers_pass_each_other_l180_180966

/-- Two swimmers in a 100-foot pool, one swimming at 4 feet per second, the other at 3 feet per second,
    continuously for 12 minutes, pass each other exactly 32 times. -/
theorem swimmers_pass_each_other 
  (pool_length : ℕ) 
  (time : ℕ) 
  (rate1 : ℕ)
  (rate2 : ℕ)
  (meet_times : ℕ)
  (hp : pool_length = 100) 
  (ht : time = 720) -- 12 minutes = 720 seconds
  (hr1 : rate1 = 4) 
  (hr2 : rate2 = 3)
  : meet_times = 32 := 
sorry

end swimmers_pass_each_other_l180_180966


namespace line_point_relation_l180_180840

theorem line_point_relation (x1 y1 x2 y2 a1 b1 c1 a2 b2 c2 : ℝ)
  (h1 : a1 * x1 + b1 * y1 = c1)
  (h2 : a2 * x2 + b2 * y2 = c2)
  (h3 : a1 + b1 = c1)
  (h4 : a2 + b2 = 2 * c2)
  (h5 : dist (x1, y1) (x2, y2) ≥ (Real.sqrt 2) / 2) :
  c1 / a1 + a2 / c2 = 3 := 
sorry

end line_point_relation_l180_180840


namespace surface_area_of_sphere_l180_180028

theorem surface_area_of_sphere (a b c : ℝ) (h1 : a = 1) (h2 : b = 2) (h3 : c = 2)
  (h4 : ∀ d, d = Real.sqrt (a^2 + b^2 + c^2)) : 
  4 * Real.pi * (d / 2)^2 = 9 * Real.pi :=
by
  sorry

end surface_area_of_sphere_l180_180028


namespace f_even_function_l180_180849

def f (x : ℝ) : ℝ := x^2 + 1

theorem f_even_function : ∀ x : ℝ, f x = f (-x) :=
by
  intro x
  show f x = f (-x)
  sorry

end f_even_function_l180_180849


namespace greatest_whole_number_difference_l180_180723

theorem greatest_whole_number_difference (x y : ℤ) (hx1 : 7 < x) (hx2 : x < 9) (hy1 : 9 < y) (hy2 : y < 15) : y - x = 6 :=
by
  sorry

end greatest_whole_number_difference_l180_180723


namespace average_speed_of_train_l180_180893

theorem average_speed_of_train
  (distance1 : ℝ) (time1 : ℝ) (stop_time : ℝ) (distance2 : ℝ) (time2 : ℝ)
  (h1 : distance1 = 240) (h2 : time1 = 3) (h3 : stop_time = 0.5)
  (h4 : distance2 = 450) (h5 : time2 = 5) :
  (distance1 + distance2) / (time1 + stop_time + time2) = 81.18 := 
sorry

end average_speed_of_train_l180_180893


namespace crayons_total_cost_l180_180439

theorem crayons_total_cost :
  let packs_initial := 4
  let packs_to_buy := 2
  let cost_per_pack := 2.5
  let total_packs := packs_initial + packs_to_buy
  let total_cost := total_packs * cost_per_pack
  total_cost = 15 :=
by
  sorry

end crayons_total_cost_l180_180439


namespace min_pos_solution_eqn_l180_180866

theorem min_pos_solution_eqn (x : ℝ) (h : (⌊x^2⌋ : ℤ) - (⌊x⌋ : ℤ)^2 = 25) : x = 7 * Real.sqrt 3 :=
sorry

end min_pos_solution_eqn_l180_180866


namespace birch_trees_count_l180_180123

-- Definitions based on the conditions
def total_trees : ℕ := 4000
def percentage_spruce : ℕ := 10
def percentage_pine : ℕ := 13

def count_spruce : ℕ := (percentage_spruce * total_trees) / 100
def count_pine : ℕ := (percentage_pine * total_trees) / 100
def count_oak : ℕ := count_spruce + count_pine

def count_birch : ℕ := total_trees - (count_spruce + count_pine + count_oak)

-- The theorem to be proven
theorem birch_trees_count :
  count_birch = 2160 := by
  sorry

end birch_trees_count_l180_180123


namespace max_distance_proof_area_of_coverage_ring_proof_l180_180881

noncomputable def maxDistanceFromCenterToRadars : ℝ :=
  24 / Real.sin (Real.pi / 7)

noncomputable def areaOfCoverageRing : ℝ :=
  960 * Real.pi / Real.tan (Real.pi / 7)

theorem max_distance_proof :
  ∀ (r n : ℕ) (width : ℝ),  n = 7 → r = 26 → width = 20 → 
  maxDistanceFromCenterToRadars = 24 / Real.sin (Real.pi / 7) :=
by
  intros r n width hn hr hwidth
  sorry

theorem area_of_coverage_ring_proof :
  ∀ (r n : ℕ) (width : ℝ), n = 7 → r = 26 → width = 20 → 
  areaOfCoverageRing = 960 * Real.pi / Real.tan (Real.pi / 7) :=
by
  intros r n width hn hr hwidth
  sorry

end max_distance_proof_area_of_coverage_ring_proof_l180_180881


namespace min_value_frac_sin_cos_l180_180848

open Real

theorem min_value_frac_sin_cos (α : ℝ) (hα1 : 0 < α) (hα2 : α < π / 2) :
  ∃ m : ℝ, (∀ x : ℝ, x = (1 / (sin α)^2 + 3 / (cos α)^2) → x ≥ m) ∧ m = 4 + 2 * sqrt 3 :=
by
  have h_sin_cos : sin α ≠ 0 ∧ cos α ≠ 0 := sorry -- This is an auxiliary lemma in the process, a proof is required.
  sorry

end min_value_frac_sin_cos_l180_180848


namespace set_intersection_correct_l180_180027

def set_A := {x : ℝ | x + 1 > 0}
def set_B := {x : ℝ | x - 3 < 0}
def set_intersection := {x : ℝ | -1 < x ∧ x < 3}

theorem set_intersection_correct : (set_A ∩ set_B) = set_intersection :=
by
  sorry

end set_intersection_correct_l180_180027


namespace sphere_radius_eq_three_of_volume_eq_surface_area_l180_180739

theorem sphere_radius_eq_three_of_volume_eq_surface_area
  (r : ℝ) 
  (h1 : (4 / 3) * Real.pi * r^3 = 4 * Real.pi * r^2) : 
  r = 3 :=
sorry

end sphere_radius_eq_three_of_volume_eq_surface_area_l180_180739


namespace cost_of_milkshake_is_correct_l180_180800

-- Definitions related to the problem conditions
def initial_amount : ℕ := 15
def spent_on_cupcakes : ℕ := initial_amount * (1 / 3)
def remaining_after_cupcakes : ℕ := initial_amount - spent_on_cupcakes
def spent_on_sandwich : ℕ := remaining_after_cupcakes * (20 / 100)
def remaining_after_sandwich : ℕ := remaining_after_cupcakes - spent_on_sandwich
def remaining_after_milkshake : ℕ := 4
def cost_of_milkshake : ℕ := remaining_after_sandwich - remaining_after_milkshake

-- The theorem stating the equivalent proof problem
theorem cost_of_milkshake_is_correct :
  cost_of_milkshake = 4 :=
sorry

end cost_of_milkshake_is_correct_l180_180800


namespace line_x_intercept_l180_180198

-- Define the given points
def Point1 : ℝ × ℝ := (10, 3)
def Point2 : ℝ × ℝ := (-10, -7)

-- Define the x-intercept problem
theorem line_x_intercept (x : ℝ) : 
  ∃ m b : ℝ, (Point1.2 = m * Point1.1 + b) ∧ (Point2.2 = m * Point2.1 + b) ∧ (0 = m * x + b) → x = 4 :=
by
  sorry

end line_x_intercept_l180_180198


namespace savings_per_month_l180_180931

-- Define the monthly earnings, total needed for car, and total earnings
def monthly_earnings : ℤ := 4000
def total_needed_for_car : ℤ := 45000
def total_earnings : ℤ := 360000

-- Define the number of months it takes to save the required amount using total earnings and monthly earnings
def number_of_months : ℤ := total_earnings / monthly_earnings

-- Define the monthly savings based on the total needed and number of months
def monthly_savings : ℤ := total_needed_for_car / number_of_months

-- Prove that the monthly savings is £500
theorem savings_per_month : monthly_savings = 500 := by
  -- Placeholder for the proof
  sorry

end savings_per_month_l180_180931


namespace kids_waiting_for_swings_l180_180677

theorem kids_waiting_for_swings (x : ℕ) (h1 : 2 * 60 = 120) 
  (h2 : ∀ y, y = 2 → (y * x = 2 * x)) 
  (h3 : 15 * (2 * x) = 30 * x)
  (h4 : 120 * x - 30 * x = 270) : x = 3 :=
sorry

end kids_waiting_for_swings_l180_180677


namespace max_value_of_function_l180_180493

noncomputable def y (x : ℝ) : ℝ := 
  Real.sin x - Real.cos x - Real.sin x * Real.cos x

theorem max_value_of_function :
  ∃ x : ℝ, y x = (1 / 2) + Real.sqrt 2 :=
sorry

end max_value_of_function_l180_180493


namespace meetings_percentage_l180_180514

-- Define all the conditions given in the problem
def first_meeting := 60 -- duration of first meeting in minutes
def second_meeting := 2 * first_meeting -- duration of second meeting in minutes
def third_meeting := first_meeting / 2 -- duration of third meeting in minutes
def total_meeting_time := first_meeting + second_meeting + third_meeting -- total meeting time
def total_workday := 10 * 60 -- total workday time in minutes

-- Statement to prove that the percentage of workday spent in meetings is 35%
def percent_meetings : Prop := (total_meeting_time / total_workday) * 100 = 35

theorem meetings_percentage :
  percent_meetings :=
by
  sorry

end meetings_percentage_l180_180514


namespace evaluate_expression_l180_180785

theorem evaluate_expression (b : ℕ) (h : b = 2) : b^3 * b^4 = 128 := by
  sorry

end evaluate_expression_l180_180785


namespace max_xyz_eq_one_l180_180157

noncomputable def max_xyz (x y z : ℝ) : ℝ :=
  if h_cond : 0 < x ∧ 0 < y ∧ 0 < z ∧ (x * y + z ^ 2 = (x + z) * (y + z)) ∧ (x + y + z = 3) then
    x * y * z
  else
    0

theorem max_xyz_eq_one : ∀ (x y z : ℝ), 0 < x → 0 < y → 0 < z → 
  (x * y + z ^ 2 = (x + z) * (y + z)) → (x + y + z = 3) → max_xyz x y z ≤ 1 :=
by
  intros x y z hx hy hz h1 h2
  -- Proof is omitted here
  sorry

end max_xyz_eq_one_l180_180157


namespace smallest_total_cashews_l180_180131

noncomputable def first_monkey_final (c1 c2 c3 : ℕ) : ℕ :=
  (2 * c1) / 3 + c2 / 6 + (4 * c3) / 18

noncomputable def second_monkey_final (c1 c2 c3 : ℕ) : ℕ :=
  c1 / 6 + (2 * c2) / 6 + (4 * c3) / 18

noncomputable def third_monkey_final (c1 c2 c3 : ℕ) : ℕ :=
  c1 / 6 + (2 * c2) / 6 + c3 / 9

theorem smallest_total_cashews : ∃ (c1 c2 c3 : ℕ), ∃ y : ℕ,
  3 * y = first_monkey_final c1 c2 c3 ∧
  2 * y = second_monkey_final c1 c2 c3 ∧
  y = third_monkey_final c1 c2 c3 ∧
  c1 + c2 + c3 = 630 :=
sorry

end smallest_total_cashews_l180_180131


namespace difference_of_areas_l180_180834

-- Defining the side length of the square
def square_side_length : ℝ := 8

-- Defining the side lengths of the rectangle
def rectangle_length : ℝ := 10
def rectangle_width : ℝ := 5

-- Defining the area functions
def area_of_square (side_length : ℝ) : ℝ := side_length * side_length
def area_of_rectangle (length : ℝ) (width : ℝ) : ℝ := length * width

-- Stating the theorem
theorem difference_of_areas :
  area_of_square square_side_length - area_of_rectangle rectangle_length rectangle_width = 14 :=
by
  sorry

end difference_of_areas_l180_180834


namespace monotonically_increasing_f_l180_180841

open Set Filter Topology

noncomputable def f (x : ℝ) : ℝ := x / (x + 1)

theorem monotonically_increasing_f : MonotoneOn f (Ioi 0) :=
sorry

end monotonically_increasing_f_l180_180841


namespace ganesh_ram_together_l180_180087

theorem ganesh_ram_together (G R S : ℝ) (h1 : G + R + S = 1 / 16) (h2 : S = 1 / 48) : (G + R) = 1 / 24 :=
by
  sorry

end ganesh_ram_together_l180_180087


namespace downstream_distance_correct_l180_180402

-- Definitions based on the conditions
def still_water_speed : ℝ := 22
def stream_speed : ℝ := 5
def travel_time : ℝ := 3

-- The effective speed downstream is the sum of the still water speed and the stream speed
def effective_speed_downstream : ℝ := still_water_speed + stream_speed

-- The distance covered downstream is the product of effective speed and travel time
def downstream_distance : ℝ := effective_speed_downstream * travel_time

-- The theorem to be proven
theorem downstream_distance_correct : downstream_distance = 81 := by
  sorry

end downstream_distance_correct_l180_180402


namespace cube_volume_increase_l180_180197

variable (a : ℝ) (h : a ≥ 0)

theorem cube_volume_increase :
  ((2 * a) ^ 3) = 8 * (a ^ 3) :=
by sorry

end cube_volume_increase_l180_180197


namespace factor_difference_of_squares_196_l180_180599

theorem factor_difference_of_squares_196 (x : ℝ) : x^2 - 196 = (x - 14) * (x + 14) := by
  sorry

end factor_difference_of_squares_196_l180_180599


namespace flower_problem_l180_180353

theorem flower_problem
  (O : ℕ) 
  (total : ℕ := 105)
  (pink_purple : ℕ := 30)
  (red := 2 * O)
  (yellow := 2 * O - 5)
  (pink := pink_purple / 2)
  (purple := pink)
  (H1 : pink + purple = pink_purple)
  (H2 : pink_purple = 30)
  (H3 : pink = purple)
  (H4 : O + red + yellow + pink + purple = total)
  (H5 : total = 105):
  O = 16 := 
by 
  sorry

end flower_problem_l180_180353


namespace equal_roots_implies_c_value_l180_180012

theorem equal_roots_implies_c_value (c : ℝ) 
  (h : ∃ x : ℝ, (x^2 + 6 * x - c = 0) ∧ (2 * x + 6 = 0)) :
  c = -9 :=
sorry

end equal_roots_implies_c_value_l180_180012


namespace minimum_value_am_bn_l180_180347

theorem minimum_value_am_bn (a b m n : ℝ) (hp_a : a > 0)
    (hp_b : b > 0) (hp_m : m > 0) (hp_n : n > 0) (ha_b : a + b = 1)
    (hm_n : m * n = 2) :
    (am + bn) * (bm + an) ≥ 3/2 := by
  sorry

end minimum_value_am_bn_l180_180347


namespace min_length_intersection_l180_180115

def set_with_length (a b : ℝ) := {x : ℝ | a ≤ x ∧ x ≤ b}
def length_of_set (a b : ℝ) := b - a
def M (m : ℝ) := set_with_length m (m + 3/4)
def N (n : ℝ) := set_with_length (n - 1/3) n

theorem min_length_intersection (m n : ℝ) (h₁ : 0 ≤ m) (h₂ : m + 3/4 ≤ 1) (h₃ : 0 ≤ n - 1/3) (h₄ : n ≤ 1) : 
  length_of_set (max m (n - 1/3)) (min (m + 3/4) n) = 1/12 :=
by
  sorry

end min_length_intersection_l180_180115


namespace real_solutions_to_system_l180_180655

theorem real_solutions_to_system :
  ∃ (s : Finset (ℝ × ℝ × ℝ × ℝ)), 
    (∀ (x y z w : ℝ), 
    (x = z + w + 2*z*w*x) ∧ 
    (y = w + x + 2*w*x*y) ∧ 
    (z = x + y + 2*x*y*z) ∧ 
    (w = y + z + 2*y*z*w) ↔ 
    (x, y, z, w) ∈ s) ∧
    (s.card = 15) :=
sorry

end real_solutions_to_system_l180_180655


namespace tan_C_in_triangle_l180_180053

theorem tan_C_in_triangle
  (A B C : ℝ)
  (cos_A : Real.cos A = 4/5)
  (tan_A_minus_B : Real.tan (A - B) = -1/2) :
  Real.tan C = 11/2 := 
sorry

end tan_C_in_triangle_l180_180053


namespace ron_l180_180847

-- Definitions for the given problem conditions
def cost_of_chocolate_bar : ℝ := 1.5
def s'mores_per_chocolate_bar : ℕ := 3
def number_of_scouts : ℕ := 15
def s'mores_per_scout : ℕ := 2

-- Proof that Ron will spend $15.00 on chocolate bars
theorem ron's_chocolate_bar_cost :
  (number_of_scouts * s'mores_per_scout / s'mores_per_chocolate_bar) * cost_of_chocolate_bar = 15 :=
by
  sorry

end ron_l180_180847


namespace calculate_final_amount_l180_180105

def calculate_percentage (percentage : ℝ) (amount : ℝ) : ℝ :=
  percentage * amount

theorem calculate_final_amount :
  let A := 3000
  let B := 0.20
  let C := 0.35
  let D := 0.05
  D * (C * (B * A)) = 10.50 := by
    sorry

end calculate_final_amount_l180_180105


namespace tile_ratio_l180_180761

theorem tile_ratio (original_black_tiles : ℕ) (original_white_tiles : ℕ) (original_width : ℕ) (original_height : ℕ) (border_width : ℕ) (border_height : ℕ) :
  original_black_tiles = 10 ∧ original_white_tiles = 22 ∧ original_width = 8 ∧ original_height = 4 ∧ border_width = 2 ∧ border_height = 2 →
  (original_black_tiles + ( (original_width + 2 * border_width) * (original_height + 2 * border_height) - original_width * original_height ) ) / original_white_tiles = 19 / 11 :=
by
  -- sorry to skip the proof
  sorry

end tile_ratio_l180_180761


namespace triangle_shape_l180_180916

-- Let there be a triangle ABC with sides opposite to angles A, B, and C being a, b, and c respectively
variables (A B C : ℝ) (a b c : ℝ) (b_ne_1 : b ≠ 1)
          (h1 : (log (b) (C / A)) = (log (sqrt (b)) (2)))
          (h2 : (log (b) (sin B / sin A)) = (log (sqrt (b)) (2)))

-- Define the theorem that states the shape of the triangle
theorem triangle_shape : A = π / 6 ∧ B = π / 2 ∧ C = π / 3 ∧ (A + B + C = π) :=
by
  -- Proof is provided in the solution, skipping proof here
  sorry

end triangle_shape_l180_180916


namespace carnival_tickets_l180_180688

theorem carnival_tickets (total_tickets friends : ℕ) (equal_share : ℕ)
  (h1 : friends = 6)
  (h2 : total_tickets = 234)
  (h3 : total_tickets % friends = 0)
  (h4 : equal_share = total_tickets / friends) : 
  equal_share = 39 := 
by
  sorry

end carnival_tickets_l180_180688


namespace probability_white_black_l180_180466

variable (a b : ℕ)

theorem probability_white_black (h_a_pos : 0 < a) (h_b_pos : 0 < b) :
  (2 * a * b) / (a + b) / (a + b - 1) = (2 * (a * b) : ℝ) / ((a + b) * (a + b - 1): ℝ) :=
by sorry

end probability_white_black_l180_180466


namespace compare_sums_of_sines_l180_180634

theorem compare_sums_of_sines {A B C : ℝ} 
  (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
  (h_sum : A + B + C = π) :
  (if A < π / 2 ∧ B < π / 2 ∧ C < π / 2 then
    (1 / (Real.sin (2 * A)) + 1 / (Real.sin (2 * B)) + 1 / (Real.sin (2 * C))
      ≥ 1 / (Real.sin A) + 1 / (Real.sin B) + 1 / (Real.sin C))
  else
    (1 / (Real.sin (2 * A)) + 1 / (Real.sin (2 * B)) + 1 / (Real.sin (2 * C))
      < 1 / (Real.sin A) + 1 / (Real.sin B) + 1 / (Real.sin C))) :=
sorry

end compare_sums_of_sines_l180_180634


namespace math_problem_l180_180398

theorem math_problem 
  (x y : ℝ) 
  (h1 : x + y = -5) 
  (h2 : x * y = 3) :
  x * Real.sqrt (y / x) + y * Real.sqrt (x / y) = -2 * Real.sqrt 3 := 
sorry

end math_problem_l180_180398


namespace number_is_correct_l180_180720

theorem number_is_correct : (1 / 8) + 0.675 = 0.800 := 
by
  sorry

end number_is_correct_l180_180720


namespace S_rational_iff_divides_l180_180693

-- Definition of "divides" for positive integers
def divides (m k : ℕ) : Prop := ∃ j : ℕ, k = m * j

-- Definition of the series S(m, k)
noncomputable def S (m k : ℕ) : ℝ := 
  ∑' n, 1 / (n * (m * n + k))

-- Proof statement
theorem S_rational_iff_divides (m k : ℕ) (hm : 0 < m) (hk : 0 < k) : 
  (∃ r : ℚ, S m k = r) ↔ divides m k :=
sorry

end S_rational_iff_divides_l180_180693


namespace randy_wipes_days_l180_180991

theorem randy_wipes_days (wipes_per_pack : ℕ) (packs_needed : ℕ) (wipes_per_walk : ℕ) (walks_per_day : ℕ) (total_wipes : ℕ) (wipes_per_day : ℕ) (days_needed : ℕ) 
(h1 : wipes_per_pack = 120)
(h2 : packs_needed = 6)
(h3 : wipes_per_walk = 4)
(h4 : walks_per_day = 2)
(h5 : total_wipes = packs_needed * wipes_per_pack)
(h6 : wipes_per_day = wipes_per_walk * walks_per_day)
(h7 : days_needed = total_wipes / wipes_per_day) : 
days_needed = 90 :=
by sorry

end randy_wipes_days_l180_180991


namespace product_sqrt_50_l180_180266

theorem product_sqrt_50 (a b : ℕ) (h₁ : a = 7) (h₂ : b = 8) (h₃ : a^2 < 50) (h₄ : 50 < b^2) : a * b = 56 := by
  sorry

end product_sqrt_50_l180_180266


namespace nextSimultaneousRingingTime_l180_180482

-- Define the intervals
def townHallInterval := 18
def universityTowerInterval := 24
def fireStationInterval := 30

-- Define the start time (in minutes from 00:00)
def startTime := 8 * 60 -- 8:00 AM

-- Define the least common multiple function
def lcm (a b : ℕ) : ℕ := Nat.lcm a b

-- Prove the next simultaneous ringing time
theorem nextSimultaneousRingingTime : 
  let lcmIntervals := lcm (lcm townHallInterval universityTowerInterval) fireStationInterval 
  startTime + lcmIntervals = 14 * 60 := -- 14:00 equals 2:00 PM in minutes
by
  -- You can replace the proof with the actual detailed proof.
  sorry

end nextSimultaneousRingingTime_l180_180482


namespace quadratic_equation_unique_solution_l180_180168

theorem quadratic_equation_unique_solution (a b x k : ℝ) (h : a = 8) (h₁ : b = 36) (h₂ : k = 40.5) : 
  (8*x^2 + 36*x + 40.5 = 0) ∧ x = -2.25 :=
by {
  sorry
}

end quadratic_equation_unique_solution_l180_180168


namespace balls_left_l180_180431

-- Define the conditions
def initial_balls : ℕ := 10
def removed_balls : ℕ := 3

-- The main statement to prove
theorem balls_left : initial_balls - removed_balls = 7 := by sorry

end balls_left_l180_180431


namespace problem1_problem2_l180_180445

def f (x a : ℝ) := x^2 + 2 * a * x + 2

theorem problem1 (a : ℝ) (h : a = -1) : 
  (∀ x : ℝ, -5 ≤ x ∧ x ≤ 5 → f x a ≤ 37) ∧ (∃ x : ℝ, -5 ≤ x ∧ x ≤ 5 ∧ f x a = 37) ∧
  (∀ x : ℝ, -5 ≤ x ∧ x ≤ 5 → f x a ≥ 1) ∧ (∃ x : ℝ, -5 ≤ x ∧ x ≤ 5 ∧ f x a = 1) :=
by
  sorry

theorem problem2 (a : ℝ) : 
  (∀ x1 x2 : ℝ, -5 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ 5 → f x1 a > f x2 a) ↔ a ≤ -5 :=
by
  sorry

end problem1_problem2_l180_180445


namespace proof1_proof2_l180_180568

noncomputable def a (n : ℕ) : ℝ := (n^2 + 1) * 3^n

def recurrence_relation : Prop :=
  ∀ n, a (n + 3) - 9 * a (n + 2) + 27 * a (n + 1) - 27 * a n = 0

noncomputable def series_sum (x : ℝ) : ℝ :=
  ∑' n, a n * x^n

def series_evaluation (x : ℝ) : Prop :=
  series_sum x = (1 - 3*x + 18*x^2) / (1 - 3*x)^3

theorem proof1 : recurrence_relation := 
  by sorry

theorem proof2 : ∀ x : ℝ, series_evaluation x := 
  by sorry

end proof1_proof2_l180_180568


namespace container_volume_ratio_l180_180357

variable (A B C : ℝ)

theorem container_volume_ratio (h1 : (4 / 5) * A = (3 / 5) * B) (h2 : (3 / 5) * B = (3 / 4) * C) :
  A / C = 15 / 16 :=
sorry

end container_volume_ratio_l180_180357


namespace expected_value_is_one_third_l180_180969

noncomputable def expected_value_of_winnings : ℚ :=
  let p1 := (1/6 : ℚ)
  let p2 := (1/6 : ℚ)
  let p3 := (1/6 : ℚ)
  let p4 := (1/6 : ℚ)
  let p5 := (1/6 : ℚ)
  let p6 := (1/6 : ℚ)
  let winnings1 := (5 : ℚ)
  let winnings2 := (5 : ℚ)
  let winnings3 := (0 : ℚ)
  let winnings4 := (0 : ℚ)
  let winnings5 := (-4 : ℚ)
  let winnings6 := (-4 : ℚ)
  (p1 * winnings1 + p2 * winnings2 + p3 * winnings3 + p4 * winnings4 + p5 * winnings5 + p6 * winnings6)

theorem expected_value_is_one_third : expected_value_of_winnings = 1 / 3 := by
  sorry

end expected_value_is_one_third_l180_180969


namespace find_number_l180_180913

theorem find_number (x : ℝ) : 61 + x * 12 / (180 / 3) = 62 → x = 5 :=
by
  sorry

end find_number_l180_180913


namespace father_son_skating_ratio_l180_180533

theorem father_son_skating_ratio (v_f v_s : ℝ) (h1 : v_f > v_s) (h2 : (v_f + v_s) / (v_f - v_s) = 5) :
  v_f / v_s = 1.5 :=
sorry

end father_son_skating_ratio_l180_180533


namespace initial_concentration_is_40_l180_180709

noncomputable def initial_concentration_fraction : ℝ := 1 / 3
noncomputable def replaced_solution_concentration : ℝ := 25
noncomputable def resulting_concentration : ℝ := 35
noncomputable def initial_concentration := 40

theorem initial_concentration_is_40 (C : ℝ) (h1 : C = (3 / 2) * (resulting_concentration - (initial_concentration_fraction * replaced_solution_concentration))) :
  C = initial_concentration :=
by sorry

end initial_concentration_is_40_l180_180709


namespace number_of_even_factors_of_n_l180_180905

def n : ℕ := 2^4 * 3^3 * 5 * 7^2

theorem number_of_even_factors_of_n : 
  (∃ k : ℕ, n = 2^4 * 3^3 * 5 * 7^2 ∧ k = 96) → 
  ∃ count : ℕ, 
    count = 96 ∧ 
    (∀ m : ℕ, 
      (m ∣ n ∧ m % 2 = 0) ↔ 
      (∃ a b c d : ℕ, 1 ≤ a ∧ a ≤ 4 ∧ 0 ≤ b ∧ b ≤ 3 ∧ 0 ≤ c ∧ c ≤ 1 ∧ 0 ≤ d ∧ d ≤ 2 ∧ m = 2^a * 3^b * 5^c * 7^d)) :=
by
  sorry

end number_of_even_factors_of_n_l180_180905


namespace shekar_marks_math_l180_180240

theorem shekar_marks_math (M : ℕ) (science : ℕ) (social_studies : ℕ) (english : ℕ) 
(biology : ℕ) (average : ℕ) (num_subjects : ℕ) 
(h_science : science = 65)
(h_social : social_studies = 82)
(h_english : english = 67)
(h_biology : biology = 55)
(h_average : average = 69)
(h_num_subjects : num_subjects = 5) :
M + science + social_studies + english + biology = average * num_subjects →
M = 76 :=
by
  sorry

end shekar_marks_math_l180_180240


namespace factorize_difference_of_squares_l180_180870

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 4 = (x + 2) * (x - 2) := 
sorry

end factorize_difference_of_squares_l180_180870


namespace angela_more_marbles_l180_180457

/--
Albert has three times as many marbles as Angela.
Allison has 28 marbles.
Albert and Allison have 136 marbles together.
Prove that Angela has 8 more marbles than Allison.
-/
theorem angela_more_marbles 
  (albert_angela : ℕ) 
  (angela: ℕ) 
  (albert: ℕ) 
  (allison: ℕ) 
  (h_albert_is_three_times_angela : albert = 3 * angela) 
  (h_allison_is_28 : allison = 28) 
  (h_albert_allison_is_136 : albert + allison = 136) 
  : angela - allison = 8 := 
by
  sorry

end angela_more_marbles_l180_180457


namespace tips_multiple_l180_180382

variable (A T : ℝ) (x : ℝ)
variable (h1 : T = 7 * A)
variable (h2 : T / 4 = x * A)

theorem tips_multiple (A T : ℝ) (x : ℝ) (h1 : T = 7 * A) (h2 : T / 4 = x * A) : x = 1.75 := by
  sorry

end tips_multiple_l180_180382


namespace total_number_of_songs_is_30_l180_180894

-- Define the number of country albums and pop albums
def country_albums : ℕ := 2
def pop_albums : ℕ := 3

-- Define the number of songs per album
def songs_per_album : ℕ := 6

-- Define the total number of albums
def total_albums : ℕ := country_albums + pop_albums

-- Define the total number of songs
def total_songs : ℕ := total_albums * songs_per_album

-- Prove that the total number of songs is 30
theorem total_number_of_songs_is_30 : total_songs = 30 := 
sorry

end total_number_of_songs_is_30_l180_180894


namespace fifth_term_sequence_l180_180155

theorem fifth_term_sequence : 2^5 * 1 * 3 * 5 * 7 * 9 = 6 * 7 * 8 * 9 * 10 := 
by 
  sorry

end fifth_term_sequence_l180_180155


namespace divisor_is_13_l180_180395

theorem divisor_is_13 (N D : ℕ) (h1 : N = 32) (h2 : (N - 6) / D = 2) : D = 13 := by
  sorry

end divisor_is_13_l180_180395


namespace joel_age_when_dad_twice_l180_180925

theorem joel_age_when_dad_twice (x joel_age dad_age: ℕ) (h₁: joel_age = 12) (h₂: dad_age = 47) 
(h₃: dad_age + x = 2 * (joel_age + x)) : joel_age + x = 35 :=
by
  rw [h₁, h₂] at h₃ 
  sorry

end joel_age_when_dad_twice_l180_180925


namespace sum_of_legs_of_larger_triangle_l180_180982

theorem sum_of_legs_of_larger_triangle 
  (area_small area_large : ℝ)
  (hypotenuse_small : ℝ)
  (A : area_small = 10)
  (B : area_large = 250)
  (C : hypotenuse_small = 13) : 
  ∃ a b : ℝ, (a + b = 35) := 
sorry

end sum_of_legs_of_larger_triangle_l180_180982


namespace A_share_of_annual_gain_l180_180823

-- Definitions based on the conditions
def investment_A (x : ℝ) : ℝ := 12 * x
def investment_B (x : ℝ) : ℝ := 12 * x
def investment_C (x : ℝ) : ℝ := 12 * x
def total_investment (x : ℝ) : ℝ := investment_A x + investment_B x + investment_C x
def annual_gain : ℝ := 15000

-- Theorem based on the question and correct answer
theorem A_share_of_annual_gain (x : ℝ) : (investment_A x / total_investment x) * annual_gain = 5000 :=
by
  sorry

end A_share_of_annual_gain_l180_180823


namespace circle_tangent_values_l180_180219

theorem circle_tangent_values (m : ℝ) :
  (∀ x y : ℝ, ((x - m)^2 + (y + 2)^2 = 9) → ((x + 1)^2 + (y - m)^2 = 4)) → 
  m = 2 ∨ m = -5 :=
by
  sorry

end circle_tangent_values_l180_180219


namespace hcf_36_84_l180_180965

def highestCommonFactor (a b : ℕ) : ℕ := Nat.gcd a b

theorem hcf_36_84 : highestCommonFactor 36 84 = 12 := by
  sorry

end hcf_36_84_l180_180965


namespace find_circle_equation_l180_180106

-- Define the conditions on the circle
def passes_through_points (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  ∃ c : ℝ × ℝ, ∃ r : ℝ, (c = center ∧ r = radius) ∧ 
  dist (0, 2) c = r ∧ dist (0, 4) c = r

def lies_on_line (center : ℝ × ℝ) : Prop :=
  2 * center.1 - center.2 - 1 = 0

-- Define the problem
theorem find_circle_equation :
  ∃ (center : ℝ × ℝ) (radius : ℝ), 
  passes_through_points center radius ∧ lies_on_line center ∧ 
  (∀ x y : ℝ, (x - center.1)^2 + (y - center.2)^2 = radius^2 
  ↔ (x - 2)^2 + (y - 3)^2 = 5) :=
sorry

end find_circle_equation_l180_180106


namespace binomial_133_133_l180_180716

theorem binomial_133_133 : @Nat.choose 133 133 = 1 := by   
sorry

end binomial_133_133_l180_180716


namespace find_larger_number_l180_180721

-- Definitions based on the conditions
def larger_number (L S : ℕ) : Prop :=
  L - S = 1365 ∧ L = 6 * S + 20

-- The theorem to prove
theorem find_larger_number (L S : ℕ) (h : larger_number L S) : L = 1634 :=
by
  sorry  -- Proof would go here

end find_larger_number_l180_180721


namespace smallest_possible_value_l180_180435

-- Definitions and conditions provided
def x_plus_4_y_minus_4_eq_zero (x y : ℝ) : Prop := (x + 4) * (y - 4) = 0

-- Main theorem to state
theorem smallest_possible_value (x y : ℝ) (h : x_plus_4_y_minus_4_eq_zero x y) : x^2 + y^2 = 32 :=
sorry

end smallest_possible_value_l180_180435


namespace quadratic_root_condition_l180_180242

theorem quadratic_root_condition (b c : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 + b*x1 + c = 0) ∧ (x2^2 + b*x2 + c = 0)) ↔ (b^2 - 4*c ≥ 0) :=
by
  sorry

end quadratic_root_condition_l180_180242


namespace factorize_problem_1_factorize_problem_2_l180_180111

theorem factorize_problem_1 (a b : ℝ) : -3 * a ^ 3 + 12 * a ^ 2 * b - 12 * a * b ^ 2 = -3 * a * (a - 2 * b) ^ 2 := 
sorry

theorem factorize_problem_2 (m n : ℝ) : 9 * (m + n) ^ 2 - (m - n) ^ 2 = 4 * (2 * m + n) * (m + 2 * n) := 
sorry

end factorize_problem_1_factorize_problem_2_l180_180111


namespace solve_for_y_l180_180571

theorem solve_for_y : ∃ y : ℝ, (5 * y^(1/3) - 3 * (y / y^(2/3)) = 10 + y^(1/3)) ↔ y = 1000 := by
  sorry

end solve_for_y_l180_180571


namespace find_two_digit_number_l180_180627

theorem find_two_digit_number (n s p : ℕ) (h1 : n = 4 * s) (h2 : n = 3 * p) : n = 24 := 
  sorry

end find_two_digit_number_l180_180627


namespace area_regular_octagon_l180_180209

theorem area_regular_octagon (AB BC: ℝ) (hAB: AB = 2) (hBC: BC = 2) :
  let side_length := 2 * Real.sqrt 2
  let triangle_area := (AB * AB) / 2
  let total_triangle_area := 4 * triangle_area
  let side_length_rect := 4 + 2 * Real.sqrt 2
  let rect_area := side_length_rect * side_length_rect
  let octagon_area := rect_area - total_triangle_area
  octagon_area = 16 + 8 * Real.sqrt 2 :=
by sorry

end area_regular_octagon_l180_180209


namespace max_cursed_roads_l180_180534

theorem max_cursed_roads (cities roads N kingdoms : ℕ) (h1 : cities = 1000) (h2 : roads = 2017)
  (h3 : cities = 1 → cities = 1000 → N ≤ 1024 → kingdoms = 7 → True) :
  max_N = 1024 :=
by
  sorry

end max_cursed_roads_l180_180534


namespace sum_xyz_eq_11sqrt5_l180_180136

noncomputable def x : ℝ :=
sorry

noncomputable def y : ℝ :=
sorry

noncomputable def z : ℝ :=
sorry

axiom pos_x : x > 0
axiom pos_y : y > 0
axiom pos_z : z > 0

axiom xy_eq_30 : x * y = 30
axiom xz_eq_60 : x * z = 60
axiom yz_eq_90 : y * z = 90

theorem sum_xyz_eq_11sqrt5 : x + y + z = 11 * Real.sqrt 5 :=
sorry

end sum_xyz_eq_11sqrt5_l180_180136


namespace max_x_plus_2y_l180_180135

theorem max_x_plus_2y (x y : ℝ) (h : 3 * (x^2 + y^2) = x + y) : x + 2 * y ≤ 3 :=
sorry

end max_x_plus_2y_l180_180135


namespace Sherry_catches_train_within_5_minutes_l180_180271

-- Defining the probabilities given in the conditions
def P_A : ℝ := 0.75  -- Probability of train arriving
def P_N : ℝ := 0.75  -- Probability of Sherry not noticing the train

-- Event that no train arrives combined with event that train arrives but not noticed
def P_not_catch_in_a_minute : ℝ := 1 - P_A + P_A * P_N

-- Generalizing to 5 minutes
def P_not_catch_in_5_minutes : ℝ := P_not_catch_in_a_minute ^ 5

-- Probability Sherry catches the train within 5 minutes
def P_C : ℝ := 1 - P_not_catch_in_5_minutes

theorem Sherry_catches_train_within_5_minutes : P_C = 1 - (13 / 16) ^ 5 := by
  sorry

end Sherry_catches_train_within_5_minutes_l180_180271


namespace find_q_l180_180942

theorem find_q (a b m p q : ℚ) 
  (h1 : ∀ x, x^2 - m * x + 3 = (x - a) * (x - b)) 
  (h2 : a * b = 3) 
  (h3 : (x^2 - p * x + q) = (x - (a + 1/b)) * (x - (b + 1/a))) : 
  q = 16 / 3 := 
by sorry

end find_q_l180_180942


namespace rate_of_current_l180_180458

theorem rate_of_current (c : ℝ) (h1 : 7.5 = (20 + c) * 0.3) : c = 5 :=
by
  sorry

end rate_of_current_l180_180458


namespace gcd_polynomial_l180_180404

theorem gcd_polynomial (b : ℕ) (h : 570 ∣ b) : Nat.gcd (5 * b^3 + 2 * b^2 + 5 * b + 95) b = 95 :=
by
  sorry

end gcd_polynomial_l180_180404


namespace math_problem_l180_180920

theorem math_problem :
    (50 + 5 * (12 / (180 / 3))^2) * Real.sin (Real.pi / 6) = 25.1 :=
by
  sorry

end math_problem_l180_180920


namespace krishan_money_l180_180026

theorem krishan_money (R G K : ℕ) 
  (h_ratio1 : R * 17 = G * 7) 
  (h_ratio2 : G * 17 = K * 7) 
  (h_R : R = 735) : 
  K = 4335 := 
sorry

end krishan_money_l180_180026


namespace baker_made_cakes_l180_180408

-- Conditions
def cakes_sold : ℕ := 145
def cakes_left : ℕ := 72

-- Question and required proof
theorem baker_made_cakes : (cakes_sold + cakes_left = 217) :=
by
  sorry

end baker_made_cakes_l180_180408


namespace taxi_fare_l180_180080

theorem taxi_fare (x : ℝ) (h : x > 6) : 
  let starting_price := 6
  let mid_distance_fare := (6 - 2) * 2.4
  let long_distance_fare := (x - 6) * 3.6
  let total_fare := starting_price + mid_distance_fare + long_distance_fare
  total_fare = 3.6 * x - 6 :=
by
  sorry

end taxi_fare_l180_180080


namespace inequality_solution_l180_180981

noncomputable def solve_inequality (m : ℝ) (m_lt_neg2 : m < -2) : Set ℝ :=
  if h : m = -3 then {x | 1 < x}
  else if h' : -3 < m then {x | x < m / (m + 3) ∨ 1 < x}
  else {x | 1 < x ∧ x < m / (m + 3)}

theorem inequality_solution (m : ℝ) (m_lt_neg2 : m < -2) :
  (solve_inequality m m_lt_neg2) = 
    if m = -3 then {x | 1 < x}
    else if -3 < m then {x | x < m / (m + 3) ∨ 1 < x}
    else {x | 1 < x ∧ x < m / (m + 3)} :=
sorry

end inequality_solution_l180_180981


namespace max_value_of_z_l180_180348

theorem max_value_of_z (k : ℝ) (x y : ℝ)
  (h1 : x + 2 * y - 1 ≥ 0)
  (h2 : x - y ≥ 0)
  (h3 : 0 ≤ x)
  (h4 : x ≤ k)
  (h5 : ∀ x y, x + 2 * y - 1 ≥ 0 ∧ x - y ≥ 0 ∧ 0 ≤ x ∧ x ≤ k → x + k * y ≥ -2) :
  ∃ (x y : ℝ), x + k * y = 20 := 
by
  sorry

end max_value_of_z_l180_180348


namespace resulting_solid_vertices_l180_180319

theorem resulting_solid_vertices (s1 s2 : ℕ) (orig_vertices removed_cubes : ℕ) :
  s1 = 5 → s2 = 2 → orig_vertices = 8 → removed_cubes = 8 → 
  (orig_vertices - removed_cubes + removed_cubes * (4 * 3 - 3)) = 40 := by
  intros h1 h2 h3 h4
  simp [h1, h2, h3, h4]
  sorry

end resulting_solid_vertices_l180_180319


namespace transportation_cost_l180_180323

theorem transportation_cost 
  (cost_per_kg : ℝ) 
  (weight_communication : ℝ) 
  (weight_sensor : ℝ) 
  (extra_sensor_cost_percentage : ℝ) 
  (cost_communication : ℝ)
  (basic_cost_sensor : ℝ)
  (extra_cost_sensor : ℝ)
  (total_cost : ℝ) : 
  cost_per_kg = 25000 → 
  weight_communication = 0.5 → 
  weight_sensor = 0.3 → 
  extra_sensor_cost_percentage = 0.10 →
  cost_communication = weight_communication * cost_per_kg →
  basic_cost_sensor = weight_sensor * cost_per_kg →
  extra_cost_sensor = extra_sensor_cost_percentage * basic_cost_sensor →
  total_cost = cost_communication + basic_cost_sensor + extra_cost_sensor →
  total_cost = 20750 :=
by sorry

end transportation_cost_l180_180323


namespace sum_of_first_ten_primes_ending_in_3_is_671_l180_180094

noncomputable def sum_of_first_ten_primes_ending_in_3 : ℕ :=
  3 + 13 + 23 + 43 + 53 + 73 + 83 + 103 + 113 + 163

theorem sum_of_first_ten_primes_ending_in_3_is_671 :
  sum_of_first_ten_primes_ending_in_3 = 671 :=
by
  sorry

end sum_of_first_ten_primes_ending_in_3_is_671_l180_180094


namespace sqrt_of_8_l180_180799

-- Definition of square root
def isSquareRoot (x : ℝ) (a : ℝ) : Prop := x * x = a

-- Theorem statement: The square root of 8 is ±√8
theorem sqrt_of_8 :
  ∃ x : ℝ, isSquareRoot x 8 ∧ (x = Real.sqrt 8 ∨ x = -Real.sqrt 8) :=
by
  sorry

end sqrt_of_8_l180_180799


namespace total_chestnuts_weight_l180_180697

def eunsoo_kg := 2
def eunsoo_g := 600
def mingi_g := 3700

theorem total_chestnuts_weight :
  (eunsoo_kg * 1000 + eunsoo_g + mingi_g) = 6300 :=
by
  sorry

end total_chestnuts_weight_l180_180697


namespace fraction_to_decimal_l180_180869

theorem fraction_to_decimal (numerator : ℚ) (denominator : ℚ) (h : numerator = 5 ∧ denominator = 40) : 
  (numerator / denominator) = 0.125 :=
sorry

end fraction_to_decimal_l180_180869


namespace find_second_number_l180_180384

theorem find_second_number (a : ℕ) (c : ℕ) (x : ℕ) : 
  3 * a + 3 * x + 3 * c + 11 = 170 → a = 16 → c = 20 → x = 17 := 
by
  intros h1 h2 h3
  rw [h2, h3] at h1
  simp at h1
  sorry

end find_second_number_l180_180384


namespace min_value_l180_180433

theorem min_value (a b c : ℤ) (h : a > b ∧ b > c) :
  ∃ x, x = (a + b + c) / (a - b - c) ∧ 
       x + (a - b - c) / (a + b + c) = 2 := sorry

end min_value_l180_180433


namespace enemies_left_undefeated_l180_180592

theorem enemies_left_undefeated (points_per_enemy points_earned total_enemies : ℕ) 
  (h1 : points_per_enemy = 3)
  (h2 : total_enemies = 6)
  (h3 : points_earned = 12) : 
  (total_enemies - points_earned / points_per_enemy) = 2 :=
by
  sorry

end enemies_left_undefeated_l180_180592


namespace maximized_area_using_squares_l180_180793

theorem maximized_area_using_squares (a b c : ℝ) : a^2 + b^2 + c^2 ≥ a * b + b * c + c * a :=
  by sorry

end maximized_area_using_squares_l180_180793


namespace total_ticket_sales_l180_180086

def ticket_price : Type := 
  ℕ → ℕ

def total_individual_sales (student_count adult_count child_count senior_count : ℕ) (prices : ticket_price) : ℝ :=
  (student_count * prices 6 + adult_count * prices 8 + child_count * prices 4 + senior_count * prices 7)

def total_group_sales (group_student_count group_adult_count group_child_count group_senior_count : ℕ) (prices : ticket_price) : ℝ :=
  let total_price := (group_student_count * prices 6 + group_adult_count * prices 8 + group_child_count * prices 4 + group_senior_count * prices 7)
  if (group_student_count + group_adult_count + group_child_count + group_senior_count) > 10 then 
    total_price - 0.10 * total_price 
  else 
    total_price

theorem total_ticket_sales
  (prices : ticket_price)
  (student_count adult_count child_count senior_count : ℕ)
  (group_student_count group_adult_count group_child_count group_senior_count : ℕ)
  (total_sales : ℝ) :
  student_count = 20 →
  adult_count = 12 →
  child_count = 15 →
  senior_count = 10 →
  group_student_count = 5 →
  group_adult_count = 8 →
  group_child_count = 10 →
  group_senior_count = 9 →
  prices 6 = 6 →
  prices 8 = 8 →
  prices 4 = 4 →
  prices 7 = 7 →
  total_sales = (total_individual_sales student_count adult_count child_count senior_count prices) + (total_group_sales group_student_count group_adult_count group_child_count group_senior_count prices) →
  total_sales = 523.30 := by
  sorry

end total_ticket_sales_l180_180086


namespace couple_ticket_cost_l180_180912

variable (x : ℝ)

def single_ticket_cost : ℝ := 20
def total_sales : ℝ := 2280
def total_attendees : ℕ := 128
def couple_tickets_sold : ℕ := 16

theorem couple_ticket_cost :
  96 * single_ticket_cost + 16 * x = total_sales →
  x = 22.5 :=
by
  sorry

end couple_ticket_cost_l180_180912


namespace find_speed_from_p_to_q_l180_180166

noncomputable def speed_from_p_to_q (v : ℝ) (d : ℝ) : Prop :=
  let return_speed := 1.5 * v
  let avg_speed := 75
  let total_distance := 2 * d
  let total_time := d / v + d / return_speed
  avg_speed = total_distance / total_time

theorem find_speed_from_p_to_q (v : ℝ) (d : ℝ) : speed_from_p_to_q v d → v = 62.5 :=
by
  intro h
  sorry

end find_speed_from_p_to_q_l180_180166


namespace area_of_connected_colored_paper_l180_180293

noncomputable def side_length : ℕ := 30
noncomputable def overlap : ℕ := 7
noncomputable def sheets : ℕ := 6
noncomputable def total_length : ℕ := side_length + (sheets - 1) * (side_length - overlap)
noncomputable def width : ℕ := side_length

theorem area_of_connected_colored_paper : total_length * width = 4350 := by
  sorry

end area_of_connected_colored_paper_l180_180293


namespace correct_sentence_l180_180798

-- Define an enumeration for different sentences
inductive Sentence
| A : Sentence
| B : Sentence
| C : Sentence
| D : Sentence

-- Define a function stating properties of each sentence
def sentence_property (s : Sentence) : Bool :=
  match s with
  | Sentence.A => false  -- "The chromosomes from dad are more than from mom" is false
  | Sentence.B => false  -- "The chromosomes in my cells and my brother's cells are exactly the same" is false
  | Sentence.C => true   -- "Each pair of homologous chromosomes is provided by both parents" is true
  | Sentence.D => false  -- "Each pair of homologous chromosomes in my brother's cells are the same size" is false

-- The theorem to prove that Sentence.C is the correct one
theorem correct_sentence : sentence_property Sentence.C = true :=
by
  unfold sentence_property
  rfl

end correct_sentence_l180_180798


namespace vector_add_sub_l180_180308

open Matrix

section VectorProof

/-- Define the vectors a, b, and c. -/
def a : Matrix (Fin 2) (Fin 1) ℤ := ![![3], ![-6]]
def b : Matrix (Fin 2) (Fin 1) ℤ := ![![-1], ![5]]
def c : Matrix (Fin 2) (Fin 1) ℤ := ![![5], ![-20]]

/-- State the proof problem. -/
theorem vector_add_sub :
  2 • a + 4 • b - c = ![![-3], ![28]] :=
by
  sorry

end VectorProof

end vector_add_sub_l180_180308


namespace shift_parabola_left_l180_180555

theorem shift_parabola_left (x : ℝ) : (x + 1)^2 = y ↔ x^2 = y :=
sorry

end shift_parabola_left_l180_180555


namespace arrange_f_values_l180_180214

noncomputable def f : ℝ → ℝ := sorry -- Assuming the actual definition is not necessary

-- The function f is even
def even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

-- The function f is strictly decreasing on (-∞, 0)
def strictly_decreasing_on_negative (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2 : ℝ, x1 < 0 ∧ x2 < 0 ∧ x1 ≠ x2 → (x1 < x2 ↔ f x1 > f x2)

theorem arrange_f_values (f : ℝ → ℝ)
  (h_even : even_function f)
  (h_decreasing : strictly_decreasing_on_negative f) :
  f 3 < f (-2) ∧ f (-2) < f 1 :=
by
  -- The actual proof would go here.
  sorry

end arrange_f_values_l180_180214


namespace red_balls_removed_to_certain_event_l180_180211

theorem red_balls_removed_to_certain_event (total_balls red_balls yellow_balls : ℕ) (m : ℕ)
  (total_balls_eq : total_balls = 8)
  (red_balls_eq : red_balls = 3)
  (yellow_balls_eq : yellow_balls = 5)
  (certain_event_A : ∀ remaining_red_balls remaining_yellow_balls,
    remaining_red_balls = red_balls - m → remaining_yellow_balls = yellow_balls →
    remaining_red_balls = 0) : m = 3 :=
by
  sorry

end red_balls_removed_to_certain_event_l180_180211


namespace remainder_7_pow_4_div_100_l180_180147

theorem remainder_7_pow_4_div_100 : (7 ^ 4) % 100 = 1 := 
by
  sorry

end remainder_7_pow_4_div_100_l180_180147


namespace solve_chris_age_l180_180921

/-- 
The average of Amy's, Ben's, and Chris's ages is 12. Six years ago, Chris was the same age as Amy is now. In 3 years, Ben's age will be 3/4 of Amy's age at that time. 
How old is Chris now? 
-/
def chris_age : Prop := 
  ∃ (a b c : ℤ), 
    (a + b + c = 36) ∧
    (c - 6 = a) ∧ 
    (b + 3 = 3 * (a + 3) / 4) ∧
    (c = 17)

theorem solve_chris_age : chris_age := 
  by
    sorry

end solve_chris_age_l180_180921


namespace cos_135_eq_neg_sqrt2_div_2_l180_180309

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = - Real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt2_div_2_l180_180309


namespace simplify_expression_l180_180606

theorem simplify_expression (x y : ℝ) : ((3 * x + 22) + (150 * y + 22)) = (3 * x + 150 * y + 44) :=
by
  sorry

end simplify_expression_l180_180606


namespace max_cos_a_correct_l180_180204

noncomputable def max_cos_a (a b : ℝ) (h : Real.cos (a + b) = Real.cos a + Real.cos b) : ℝ :=
  Real.sqrt 3 - 1

theorem max_cos_a_correct (a b : ℝ) (h : Real.cos (a + b) = Real.cos a + Real.cos b) :
  max_cos_a a b h = Real.sqrt 3 - 1 :=
sorry

end max_cos_a_correct_l180_180204


namespace negation_of_proposition_l180_180328

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x^2 = 1 → x = 1 ∨ x = -1) ↔ ∀ x : ℝ, x^2 ≠ 1 → x ≠ 1 ∧ x ≠ -1 := by
sorry

end negation_of_proposition_l180_180328


namespace silvia_order_total_cost_l180_180744

theorem silvia_order_total_cost :
  let quiche_price : ℝ := 15
  let croissant_price : ℝ := 3
  let biscuit_price : ℝ := 2
  let quiche_count : ℝ := 2
  let croissant_count : ℝ := 6
  let biscuit_count : ℝ := 6
  let discount_rate : ℝ := 0.10
  let pre_discount_total : ℝ := (quiche_price * quiche_count) + (croissant_price * croissant_count) + (biscuit_price * biscuit_count)
  let discount_amount : ℝ := pre_discount_total * discount_rate
  let post_discount_total : ℝ := pre_discount_total - discount_amount
  pre_discount_total > 50 → post_discount_total = 54 :=
by
  sorry

end silvia_order_total_cost_l180_180744


namespace polynomial_evaluation_l180_180220

theorem polynomial_evaluation (x : ℝ) (h₁ : 0 < x) (h₂ : x^2 - 2 * x - 15 = 0) :
  x^3 - 2 * x^2 - 8 * x + 16 = 51 :=
sorry

end polynomial_evaluation_l180_180220


namespace perfect_square_iff_all_perfect_squares_l180_180488

theorem perfect_square_iff_all_perfect_squares
  (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0):
  (∃ k : ℕ, (xy + 1) * (yz + 1) * (zx + 1) = k^2) ↔
  (∃ a b c : ℕ, xy + 1 = a^2 ∧ yz + 1 = b^2 ∧ zx + 1 = c^2) := 
sorry

end perfect_square_iff_all_perfect_squares_l180_180488


namespace sufficient_but_not_necessary_condition_l180_180276

theorem sufficient_but_not_necessary_condition
  (p q r : Prop)
  (h_p_sufficient_q : p → q)
  (h_r_necessary_q : q → r)
  (h_p_not_necessary_q : ¬ (q → p))
  (h_r_not_sufficient_q : ¬ (r → q)) :
  (p → r) ∧ ¬ (r → p) :=
by
  sorry

end sufficient_but_not_necessary_condition_l180_180276


namespace number_of_players_l180_180608

-- Definitions based on conditions in the problem
def cost_of_gloves : ℕ := 6
def cost_of_helmet : ℕ := cost_of_gloves + 7
def cost_of_cap : ℕ := 3
def total_expenditure : ℕ := 2968

-- Total cost for one player
def cost_per_player : ℕ := 2 * (cost_of_gloves + cost_of_helmet) + cost_of_cap

-- Statement to prove: number of players
theorem number_of_players : total_expenditure / cost_per_player = 72 := 
by
  sorry

end number_of_players_l180_180608


namespace evaluate_g_at_4_l180_180666

def g (x : ℕ) := 5 * x + 2

theorem evaluate_g_at_4 : g 4 = 22 := by
  sorry

end evaluate_g_at_4_l180_180666


namespace min_value_range_l180_180909

noncomputable def f (a x : ℝ) := x^2 + a * x

theorem min_value_range (a : ℝ) :
  (∃x : ℝ, ∀y : ℝ, f a (f a x) ≥ f a (f a y)) ∧ (∀x : ℝ, f a x ≥ f a (-a / 2)) →
  a ≤ 0 ∨ a ≥ 2 := sorry

end min_value_range_l180_180909


namespace train_stop_time_per_hour_l180_180509

theorem train_stop_time_per_hour
    (v1 : ℕ) (v2 : ℕ)
    (h1 : v1 = 45)
    (h2 : v2 = 33) : ∃ (t : ℕ), t = 16 := by
  -- including the proof steps here is unnecessary, so we use sorry
  sorry

end train_stop_time_per_hour_l180_180509


namespace total_value_of_item_l180_180030

theorem total_value_of_item (V : ℝ) 
  (h1 : 0.07 * (V - 1000) = 109.20) : 
  V = 2560 :=
sorry

end total_value_of_item_l180_180030


namespace range_of_a_l180_180684

def line_intersects_circle (a : ℝ) : Prop :=
  let distance_from_center_to_line := |1 - a| / Real.sqrt 2
  distance_from_center_to_line ≤ Real.sqrt 2

theorem range_of_a :
  {a : ℝ | line_intersects_circle a} = {a : ℝ | -1 ≤ a ∧ a ≤ 3} :=
by
  sorry

end range_of_a_l180_180684


namespace expression_divisibility_l180_180582

theorem expression_divisibility (x y : ℝ) : 
  ∃ P : ℝ, (x^2 - x * y + y^2)^3 + (x^2 + x * y + y^2)^3 = (2 * x^2 + 2 * y^2) * P := 
by 
  sorry

end expression_divisibility_l180_180582


namespace max_a_b_l180_180385

theorem max_a_b (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (h_eq : a^2 + b^2 = 1) : a + b ≤ Real.sqrt 2 := sorry

end max_a_b_l180_180385


namespace choose_questions_l180_180342

theorem choose_questions (q : ℕ) (last : ℕ) (total : ℕ) (chosen : ℕ) 
  (condition : q ≥ 3) 
  (n : last = 5) 
  (m : total = 10) 
  (k : chosen = 6) : 
  ∃ (ways : ℕ), ways = 155 := 
by
  sorry

end choose_questions_l180_180342


namespace divisor_is_11_l180_180329

noncomputable def least_subtracted_divisor : Nat := 11

def problem_condition (D : Nat) (x : Nat) : Prop :=
  2000 - x = 1989 ∧ (2000 - x) % D = 0

theorem divisor_is_11 (D : Nat) (x : Nat) (h : problem_condition D x) : D = least_subtracted_divisor :=
by
  sorry

end divisor_is_11_l180_180329


namespace rhombus_area_l180_180950

theorem rhombus_area
  (side_length : ℝ)
  (h₀ : side_length = 2 * Real.sqrt 3)
  (tri_a_base : ℝ)
  (tri_b_base : ℝ)
  (h₁ : tri_a_base = side_length)
  (h₂ : tri_b_base = side_length) :
  ∃ rhombus_area : ℝ,
    rhombus_area = 8 * Real.sqrt 3 - 12 :=
by
  sorry

end rhombus_area_l180_180950


namespace reduced_price_l180_180241

theorem reduced_price (
  P R : ℝ)
  (h1 : R = 0.70 * P)
  (h2 : 9 = 900 / R - 900 / P)
  (h3 : P = 42.8571) :
  R = 30 :=
by {
  sorry
}

end reduced_price_l180_180241


namespace solve_for_x_l180_180473

theorem solve_for_x (x : ℝ) (h : (4 / 7) * (1 / 9) * x = 14) : x = 220.5 :=
by
  sorry

end solve_for_x_l180_180473


namespace penny_dime_halfdollar_same_probability_l180_180792

def probability_same_penny_dime_halfdollar : ℚ :=
  let total_outcomes := 2 ^ 5
  let successful_outcomes := 2 * 2 * 2
  successful_outcomes / total_outcomes

theorem penny_dime_halfdollar_same_probability :
  probability_same_penny_dime_halfdollar = 1 / 4 :=
by 
  sorry

end penny_dime_halfdollar_same_probability_l180_180792


namespace complement_of_A_in_U_l180_180584

def U : Set ℕ := {1, 2, 3, 4}
def A : Set ℕ := {1, 3}

theorem complement_of_A_in_U : U \ A = {2, 4} := 
by
  sorry

end complement_of_A_in_U_l180_180584


namespace total_students_went_to_concert_l180_180928

/-- There are 12 buses and each bus took 57 students. We want to find out the total number of students who went to the concert. -/
theorem total_students_went_to_concert (num_buses : ℕ) (students_per_bus : ℕ) (total_students : ℕ) 
  (h1 : num_buses = 12) (h2 : students_per_bus = 57) (h3 : total_students = num_buses * students_per_bus) : 
  total_students = 684 := 
by
  sorry

end total_students_went_to_concert_l180_180928


namespace solve_for_x_l180_180997

theorem solve_for_x (x : ℝ) (h : (3 * x + 15)^2 = 3 * (4 * x + 40)) :
  x = -5 / 3 ∨ x = -7 :=
sorry

end solve_for_x_l180_180997


namespace sin_ratio_in_triangle_l180_180741

theorem sin_ratio_in_triangle
  {A B C : ℝ} {a b c : ℝ}
  (h : (b + c) / (c + a) = 4 / 5 ∧ (c + a) / (a + b) = 5 / 6) :
  (Real.sin A + Real.sin C) / Real.sin B = 2 :=
sorry

end sin_ratio_in_triangle_l180_180741


namespace locus_of_point_C_l180_180812

structure Point :=
  (x : ℝ)
  (y : ℝ)

def is_isosceles_triangle (A B C : Point) : Prop := 
  let AB := (A.x - B.x)^2 + (A.y - B.y)^2
  let AC := (A.x - C.x)^2 + (A.y - C.y)^2
  AB = AC

def circle_eqn (C : Point) : Prop :=
  C.x^2 + C.y^2 - 3 * C.x + C.y = 2

def not_points (C : Point) : Prop :=
  (C ≠ {x := 3, y := -2}) ∧ (C ≠ {x := 0, y := 1})

theorem locus_of_point_C :
  ∀ (A B C : Point),
    A = {x := 3, y := -2} →
    B = {x := 0, y := 1} →
    is_isosceles_triangle A B C →
    circle_eqn C ∧ not_points C :=
by
  intros A B C hA hB hIso
  sorry

end locus_of_point_C_l180_180812


namespace find_a_parallel_find_a_perpendicular_l180_180733

open Real

def line_parallel (p1 p2 q1 q2 : (ℝ × ℝ)) : Prop :=
  let k1 := (q2.2 - q1.2) / (q2.1 - q1.1)
  let k2 := (p2.2 - p1.2) / (p2.1 - p1.1)
  k1 = k2

def line_perpendicular (p1 p2 q1 q2 : (ℝ × ℝ)) : Prop :=
  let k1 := (q2.2 - q1.2) / (q2.1 - q1.1)
  let k2 := (p2.2 - p1.2) / (p2.1 - p1.1)
  k1 * k2 = -1

theorem find_a_parallel (a : ℝ) :
  line_parallel (3, a) (a-1, 2) (1, 2) (-2, a+2) ↔ a = 1 ∨ a = 6 :=
by sorry

theorem find_a_perpendicular (a : ℝ) :
  line_perpendicular (3, a) (a-1, 2) (1, 2) (-2, a+2) ↔ a = 3 ∨ a = -4 :=
by sorry

end find_a_parallel_find_a_perpendicular_l180_180733


namespace count_five_letter_words_l180_180930

theorem count_five_letter_words : (26 ^ 4 = 456976) :=
by {
    sorry
}

end count_five_letter_words_l180_180930


namespace gotham_street_termite_ridden_not_collapsing_l180_180277

def fraction_termite_ridden := 1 / 3
def fraction_collapsing_given_termite_ridden := 4 / 7
def fraction_not_collapsing := 3 / 21

theorem gotham_street_termite_ridden_not_collapsing
  (h1: fraction_termite_ridden = 1 / 3)
  (h2: fraction_collapsing_given_termite_ridden = 4 / 7) :
  fraction_termite_ridden * (1 - fraction_collapsing_given_termite_ridden) = fraction_not_collapsing :=
sorry

end gotham_street_termite_ridden_not_collapsing_l180_180277


namespace money_distribution_l180_180172

theorem money_distribution (A B C : ℝ) (h1 : A + B + C = 1000) (h2 : B + C = 600) (h3 : C = 300) : A + C = 700 := by
  sorry

end money_distribution_l180_180172


namespace marks_of_A_l180_180018

variable (a b c d e : ℕ)

theorem marks_of_A:
  (a + b + c = 144) →
  (a + b + c + d = 188) →
  (e = d + 3) →
  (b + c + d + e = 192) →
  a = 43 := 
by 
  intros h1 h2 h3 h4
  sorry

end marks_of_A_l180_180018


namespace smallest_number_ending_in_9_divisible_by_13_l180_180778

theorem smallest_number_ending_in_9_divisible_by_13 :
  ∃ (n : ℕ), (n % 10 = 9) ∧ (13 ∣ n) ∧ (∀ (m : ℕ), (m % 10 = 9) ∧ (13 ∣ m) ∧ (m < n) -> false) :=
sorry

end smallest_number_ending_in_9_divisible_by_13_l180_180778


namespace simplify_fraction_l180_180832

variable (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0)

theorem simplify_fraction : (1 / a) + (1 / b) - (2 * a + b) / (2 * a * b) = 1 / (2 * a) :=
by
  sorry

end simplify_fraction_l180_180832


namespace class_student_numbers_l180_180253

theorem class_student_numbers (a b c d : ℕ) 
    (h_avg : (a + b + c + d) / 4 = 46)
    (h_diff_ab : a - b = 4)
    (h_diff_bc : b - c = 3)
    (h_diff_cd : c - d = 2)
    (h_max_a : a > b ∧ a > c ∧ a > d) : 
    a = 51 ∧ b = 47 ∧ c = 44 ∧ d = 42 := 
by 
  sorry

end class_student_numbers_l180_180253


namespace sum_abcd_eq_neg_46_div_3_l180_180994

theorem sum_abcd_eq_neg_46_div_3
  (a b c d : ℝ) 
  (h : a + 2 = b + 3 ∧ b + 3 = c + 4 ∧ c + 4 = d + 5 ∧ d + 5 = a + b + c + d + 15) :
  a + b + c + d = -46 / 3 := 
by sorry

end sum_abcd_eq_neg_46_div_3_l180_180994


namespace total_students_in_class_l180_180362

theorem total_students_in_class (female_students : ℕ) (male_students : ℕ) (total_students : ℕ) 
  (h1 : female_students = 13) 
  (h2 : male_students = 3 * female_students) 
  (h3 : total_students = female_students + male_students) : 
    total_students = 52 := 
by
  sorry

end total_students_in_class_l180_180362


namespace difference_max_min_eq_2log2_minus_1_l180_180986

noncomputable def f (x : ℝ) : ℝ := x * Real.log x - x

theorem difference_max_min_eq_2log2_minus_1 :
  let M := max (f (1 / 2)) (max (f 1) (f 2))
  let N := min (f (1 / 2)) (min (f 1) (f 2))
  M - N = 2 * Real.log 2 - 1 :=
by
  let M := max (f (1 / 2)) (max (f 1) (f 2))
  let N := min (f (1 / 2)) (min (f 1) (f 2))
  sorry

end difference_max_min_eq_2log2_minus_1_l180_180986


namespace order_A_C_B_l180_180311

noncomputable def A (a b : ℝ) : ℝ := Real.log ((a + b) / 2)
noncomputable def B (a b : ℝ) : ℝ := Real.sqrt (Real.log a * Real.log b)
noncomputable def C (a b : ℝ) : ℝ := (Real.log a + Real.log b) / 2

theorem order_A_C_B (a b : ℝ) (h1 : 1 < b) (h2 : b < a) :
  A a b > C a b ∧ C a b > B a b :=
by 
  sorry

end order_A_C_B_l180_180311


namespace percentage_A_to_B_l180_180690

variable (A B : ℕ)
variable (total : ℕ := 570)
variable (B_amount : ℕ := 228)

theorem percentage_A_to_B :
  (A + B = total) →
  B = B_amount →
  (A = total - B_amount) →
  ((A / B_amount : ℚ) * 100 = 150) :=
sorry

end percentage_A_to_B_l180_180690


namespace find_x_l180_180079

def custom_op (a b : ℤ) : ℤ := 2 * a + 3 * b

theorem find_x : ∃ x : ℤ, custom_op 5 (custom_op 7 x) = -4 ∧ x = -56 / 9 := by
  sorry

end find_x_l180_180079


namespace possible_values_of_m_l180_180184

def f (x a m : ℝ) := abs (x - a) + m * abs (x + a)

theorem possible_values_of_m {a m : ℝ} (h1 : 0 < m) (h2 : m < 1) 
  (h3 : ∀ x : ℝ, f x a m ≥ 2)
  (h4 : a ≤ -5 ∨ a ≥ 5) : m = 1 / 5 :=
by 
  sorry

end possible_values_of_m_l180_180184


namespace tan_15_degrees_theta_range_valid_max_f_value_l180_180516

-- Define the dot product condition
def dot_product_condition (AB BC : ℝ) (θ : ℝ) : Prop :=
  AB * BC * (Real.cos θ) = 6

-- Define the sine inequality condition
def sine_inequality_condition (AB BC : ℝ) (θ : ℝ) : Prop :=
  6 * (2 - Real.sqrt 3) ≤ AB * BC * (Real.sin θ) ∧ AB * BC * (Real.sin θ) ≤ 6 * Real.sqrt 3

-- Define the maximum value function
noncomputable def f (θ : ℝ) : ℝ :=
  (1 - Real.sqrt 2 * Real.cos (2 * θ - Real.pi / 4)) / (Real.sin θ)

-- Proof that tan 15 degrees is equal to 2 - sqrt(3)
theorem tan_15_degrees : Real.tan (Real.pi / 12) = 2 - Real.sqrt 3 := 
  by sorry

-- Proof for the range of θ
theorem theta_range_valid (AB BC : ℝ) (θ : ℝ) 
  (h1 : dot_product_condition AB BC θ)
  (h2 : sine_inequality_condition AB BC θ) : 
  (Real.pi / 12) ≤ θ ∧ θ ≤ (Real.pi / 3) := 
  by sorry

-- Proof for the maximum value of the function
theorem max_f_value (θ : ℝ) 
  (h : (Real.pi / 12) ≤ θ ∧ θ ≤ (Real.pi / 3)) : 
  f θ ≤ Real.sqrt 3 - 1 := 
  by sorry

end tan_15_degrees_theta_range_valid_max_f_value_l180_180516


namespace kite_minimum_area_correct_l180_180593

noncomputable def minimumKiteAreaAndSum (r : ℕ) (OP : ℕ) (h₁ : r = 60) (h₂ : OP < r) : ℕ × ℝ :=
  let d₁ := 2 * r
  let d₂ := 2 * Real.sqrt (r^2 - OP^2)
  let area := (d₁ * d₂) / 2
  (120 + 119, area)

theorem kite_minimum_area_correct {r OP : ℕ} (h₁ : r = 60) (h₂ : OP < r) :
  minimumKiteAreaAndSum r OP h₁ h₂ = (239, 120 * Real.sqrt 119) :=
by simp [minimumKiteAreaAndSum, h₁, h₂] ; sorry

end kite_minimum_area_correct_l180_180593


namespace minimum_students_ans_q1_correctly_l180_180073

variable (Total Students Q1 Q2 Q1_and_Q2 : ℕ)
variable (did_not_take_test: Student → Bool)

-- Given Conditions
def total_students := 40
def students_ans_q2_correctly := 29
def students_not_taken_test := 10
def students_ans_both_correctly := 29

theorem minimum_students_ans_q1_correctly (H1: Q2 - students_not_taken_test == 30)
                                           (H2: Q1_and_Q2 + students_not_taken_test == total_students)
                                           (H3: Q1_and_Q2 == students_ans_q2_correctly):
  Q1 ≥ 29 := by
  sorry

end minimum_students_ans_q1_correctly_l180_180073


namespace log_lt_x_squared_for_x_gt_zero_l180_180541

theorem log_lt_x_squared_for_x_gt_zero (x : ℝ) (h : x > 0) : Real.log (1 + x) < x^2 :=
sorry

end log_lt_x_squared_for_x_gt_zero_l180_180541


namespace water_added_l180_180179

theorem water_added (initial_fullness : ℝ) (fullness_after : ℝ) (capacity : ℝ) 
  (h_initial : initial_fullness = 0.30) (h_after : fullness_after = 3/4) (h_capacity : capacity = 100) : 
  fullness_after * capacity - initial_fullness * capacity = 45 := 
by 
  sorry

end water_added_l180_180179


namespace number_of_distinct_arrangements_l180_180896

-- Given conditions: There are 7 items and we need to choose 4 out of these 7.
def binomial_coefficient (n k : ℕ) : ℕ :=
  (n.choose k)

-- Given condition: Calculate the number of sequences of arranging 4 selected items.
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- The statement in Lean 4 to prove that the number of distinct arrangements is 840.
theorem number_of_distinct_arrangements : binomial_coefficient 7 4 * factorial 4 = 840 :=
by
  sorry

end number_of_distinct_arrangements_l180_180896


namespace mod_exp_sub_l180_180406

theorem mod_exp_sub (a b k : ℕ) (h₁ : a ≡ 6 [MOD 7]) (h₂ : b ≡ 4 [MOD 7]) :
  (a ^ k - b ^ k) % 7 = 2 :=
sorry

end mod_exp_sub_l180_180406


namespace boys_in_class_l180_180757

theorem boys_in_class (g b : ℕ) 
  (h_ratio : 4 * g = 3 * b) (h_total : g + b = 28) : b = 16 :=
by
  sorry

end boys_in_class_l180_180757


namespace oranges_taken_l180_180897

theorem oranges_taken (initial_oranges remaining_oranges taken_oranges : ℕ) 
  (h1 : initial_oranges = 60) 
  (h2 : remaining_oranges = 25) 
  (h3 : taken_oranges = initial_oranges - remaining_oranges) : 
  taken_oranges = 35 :=
by
  -- Proof is omitted, as instructed.
  sorry

end oranges_taken_l180_180897


namespace rainy_days_l180_180407

theorem rainy_days (n R NR : ℤ) 
  (h1 : n * R + 4 * NR = 26)
  (h2 : 4 * NR - n * R = 14)
  (h3 : R + NR = 7) : 
  R = 2 := 
sorry

end rainy_days_l180_180407


namespace charges_are_equal_l180_180554

variable (a : ℝ)  -- original price for both travel agencies

def charge_A (a : ℝ) : ℝ := a + 2 * 0.7 * a
def charge_B (a : ℝ) : ℝ := 3 * 0.8 * a

theorem charges_are_equal : charge_A a = charge_B a :=
by
  sorry

end charges_are_equal_l180_180554


namespace maximize_expression_l180_180387

theorem maximize_expression :
  ∀ (a b c d e : ℕ),
    a ≠ b → a ≠ c → a ≠ d → a ≠ e → b ≠ c → b ≠ d → b ≠ e → c ≠ d → c ≠ e → d ≠ e →
    (a = 1 ∨ a = 2 ∨ a = 3 ∨ a = 4 ∨ a = 6) → 
    (b = 1 ∨ b = 2 ∨ b = 3 ∨ b = 4 ∨ b = 6) →
    (c = 1 ∨ c = 2 ∨ c = 3 ∨ c = 4 ∨ c = 6) →
    (d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 6) →
    (e = 1 ∨ e = 2 ∨ e = 3 ∨ e = 4 ∨ e = 6) →
    ((a : ℚ) / 2 + (d : ℚ) / e * (c / b)) ≤ 9 :=
by
  sorry

end maximize_expression_l180_180387


namespace outer_boundary_diameter_l180_180759

theorem outer_boundary_diameter (fountain_diameter garden_width path_width : ℝ) 
(h1 : fountain_diameter = 12) 
(h2 : garden_width = 10) 
(h3 : path_width = 6) : 
2 * ((fountain_diameter / 2) + garden_width + path_width) = 44 :=
by
  -- Sorry, proof not needed for this statement
  sorry

end outer_boundary_diameter_l180_180759


namespace number_of_intersections_l180_180917

theorem number_of_intersections : ∃ (a_values : Finset ℚ), 
  ∀ a ∈ a_values, ∀ x y, y = 2 * x + a ∧ y = x^2 + 3 * a^2 ∧ x = 0 → 
  2 = a_values.card :=
by 
  sorry

end number_of_intersections_l180_180917


namespace no_integer_solutions_l180_180258

theorem no_integer_solutions (p : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) (hq : Nat.Prime (2*p + 1)) :
  ∀ (x y z : ℤ), x^p + 2 * y^p + 5 * z^p = 0 → x = 0 ∧ y = 0 ∧ z = 0 :=
by
  sorry

end no_integer_solutions_l180_180258


namespace no_solution_lines_parallel_l180_180486

theorem no_solution_lines_parallel (m : ℝ) :
  (∀ t s : ℝ, (1 + 5 * t = 4 - 2 * s) ∧ (-3 + 2 * t = 1 + m * s) → false) ↔ m = -4 / 5 :=
by
  sorry

end no_solution_lines_parallel_l180_180486


namespace find_y_plus_one_over_y_l180_180782

variable (y : ℝ)

theorem find_y_plus_one_over_y (h : y^3 + (1/y)^3 = 110) : y + 1/y = 5 :=
by
  sorry

end find_y_plus_one_over_y_l180_180782


namespace evaluate_expression_l180_180758

theorem evaluate_expression : 5 - 7 * (8 - 3^2) * 4 = 33 :=
by
  sorry

end evaluate_expression_l180_180758


namespace ground_beef_sold_ratio_l180_180644

variable (beef_sold_Thursday : ℕ) (beef_sold_Saturday : ℕ) (avg_sold_per_day : ℕ) (days : ℕ)

theorem ground_beef_sold_ratio (h₁ : beef_sold_Thursday = 210)
                             (h₂ : beef_sold_Saturday = 150)
                             (h₃ : avg_sold_per_day = 260)
                             (h₄ : days = 3) :
  let total_sold := avg_sold_per_day * days
  let beef_sold_Friday := total_sold - beef_sold_Thursday - beef_sold_Saturday
  (beef_sold_Friday : ℕ) / (beef_sold_Thursday : ℕ) = 2 := by
  sorry

end ground_beef_sold_ratio_l180_180644


namespace two_b_leq_a_plus_c_l180_180335

variable (t a b c : ℝ)

theorem two_b_leq_a_plus_c (ht : t > 1)
  (h : 2 / Real.log t / Real.log b = 1 / Real.log t / Real.log a + 1 / Real.log t / Real.log c) :
  2 * b ≤ a + c := by sorry

end two_b_leq_a_plus_c_l180_180335


namespace modular_units_l180_180813

theorem modular_units (U N S : ℕ) 
  (h1 : N = S / 4)
  (h2 : (S : ℚ) / (S + U * N) = 0.14285714285714285) : 
  U = 24 :=
by
  sorry

end modular_units_l180_180813


namespace pradeep_passing_percentage_l180_180859

-- Define the constants based on the conditions
def totalMarks : ℕ := 550
def marksObtained : ℕ := 200
def marksFailedBy : ℕ := 20

-- Calculate the passing marks
def passingMarks : ℕ := marksObtained + marksFailedBy

-- Define the percentage calculation as a noncomputable function
noncomputable def requiredPercentageToPass : ℚ := (passingMarks / totalMarks) * 100

-- The theorem to prove
theorem pradeep_passing_percentage :
  requiredPercentageToPass = 40 := 
sorry

end pradeep_passing_percentage_l180_180859


namespace largest_divisor_n4_minus_n2_l180_180368

theorem largest_divisor_n4_minus_n2 (n : ℤ) : 12 ∣ (n^4 - n^2) :=
by
  sorry

end largest_divisor_n4_minus_n2_l180_180368


namespace problem_g3_1_l180_180984

theorem problem_g3_1 (a : ℝ) : 
  (2002^3 + 4 * 2002^2 + 6006) / (2002^2 + 2002) = a ↔ a = 2005 := 
sorry

end problem_g3_1_l180_180984


namespace total_points_of_three_players_l180_180900

-- Definitions based on conditions
def points_tim : ℕ := 30
def points_joe : ℕ := points_tim - 20
def points_ken : ℕ := 2 * points_tim

-- Theorem statement for the total points scored by the three players
theorem total_points_of_three_players :
  points_tim + points_joe + points_ken = 100 :=
by
  -- Proof is to be provided
  sorry

end total_points_of_three_players_l180_180900


namespace set_intersection_l180_180025

noncomputable def SetA : Set ℝ := {x | Real.sqrt (x - 1) < Real.sqrt 2}
noncomputable def SetB : Set ℝ := {x | x^2 - 6 * x + 8 < 0}

theorem set_intersection :
  SetA ∩ SetB = {x | 2 < x ∧ x < 3} := by
  sorry

end set_intersection_l180_180025


namespace product_of_five_consecutive_integers_divisible_by_120_l180_180414

theorem product_of_five_consecutive_integers_divisible_by_120 (n : ℤ) : 
  120 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by 
  sorry

end product_of_five_consecutive_integers_divisible_by_120_l180_180414


namespace determine_value_of_y_l180_180367

variable (s y : ℕ)
variable (h_pos : s > 30)
variable (h_eq : s * s = (s - 15) * (s + y))

theorem determine_value_of_y (h_pos : s > 30) (h_eq : s * s = (s - 15) * (s + y)) : 
  y = 15 * s / (s + 15) :=
by
  sorry

end determine_value_of_y_l180_180367


namespace probability_point_in_sphere_eq_2pi_div_3_l180_180629

open Real Topology

noncomputable def volume_of_region := 4 * 2 * 2

noncomputable def volume_of_sphere_radius_2 : ℝ :=
  (4 / 3) * π * (2 ^ 3)

noncomputable def probability_in_sphere : ℝ :=
  volume_of_sphere_radius_2 / volume_of_region

theorem probability_point_in_sphere_eq_2pi_div_3 :
  probability_in_sphere = (2 * π) / 3 :=
by
  sorry

end probability_point_in_sphere_eq_2pi_div_3_l180_180629


namespace total_fish_l180_180497

-- Definition of the number of fish Lilly has
def lilly_fish : Nat := 10

-- Definition of the number of fish Rosy has
def rosy_fish : Nat := 8

-- Statement to prove
theorem total_fish : lilly_fish + rosy_fish = 18 := 
by
  -- The proof is omitted
  sorry

end total_fish_l180_180497


namespace geometric_sequence_condition_l180_180126

variable (a_1 : ℝ) (q : ℝ)

noncomputable def geometric_sum (n : ℕ) : ℝ :=
if q = 1 then a_1 * n else a_1 * (1 - q^n) / (1 - q)

theorem geometric_sequence_condition (a_1 : ℝ) (q : ℝ) :
  (a_1 > 0) ↔ (geometric_sum a_1 q 2017 > 0) :=
by sorry

end geometric_sequence_condition_l180_180126


namespace line_segments_cannot_form_triangle_l180_180670

theorem line_segments_cannot_form_triangle (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : a 7 = 21)
    (h3 : ∀ n, a n < a (n+1)) (h4 : ∀ i j k, a i + a j ≤ a k) :
    a 6 = 13 :=
    sorry

end line_segments_cannot_form_triangle_l180_180670


namespace chef_earns_less_than_manager_l180_180932

noncomputable def hourly_wage_manager : ℝ := 8.5
noncomputable def hourly_wage_dishwasher : ℝ := hourly_wage_manager / 2
noncomputable def hourly_wage_chef : ℝ := hourly_wage_dishwasher * 1.2
noncomputable def daily_bonus : ℝ := 5
noncomputable def overtime_multiplier : ℝ := 1.5
noncomputable def tax_rate : ℝ := 0.15

noncomputable def manager_hours : ℝ := 10
noncomputable def dishwasher_hours : ℝ := 6
noncomputable def chef_hours : ℝ := 12
noncomputable def standard_hours : ℝ := 8

noncomputable def compute_earnings (hourly_wage : ℝ) (hours_worked : ℝ) : ℝ :=
  let regular_hours := min standard_hours hours_worked
  let overtime_hours := max 0 (hours_worked - standard_hours)
  let regular_pay := regular_hours * hourly_wage
  let overtime_pay := overtime_hours * hourly_wage * overtime_multiplier
  let total_earnings_before_tax := regular_pay + overtime_pay + daily_bonus
  total_earnings_before_tax * (1 - tax_rate)

noncomputable def manager_earnings : ℝ := compute_earnings hourly_wage_manager manager_hours
noncomputable def dishwasher_earnings : ℝ := compute_earnings hourly_wage_dishwasher dishwasher_hours
noncomputable def chef_earnings : ℝ := compute_earnings hourly_wage_chef chef_hours

theorem chef_earns_less_than_manager : manager_earnings - chef_earnings = 18.78 := by
  sorry

end chef_earns_less_than_manager_l180_180932


namespace enrique_shredder_pages_l180_180508

theorem enrique_shredder_pages (total_contracts : ℕ) (num_times : ℕ) (pages_per_time : ℕ) :
  total_contracts = 2132 ∧ num_times = 44 → pages_per_time = 48 :=
by
  intros h
  sorry

end enrique_shredder_pages_l180_180508


namespace domain_is_correct_l180_180312

def domain_of_function (x : ℝ) : Prop :=
  (3 - x ≥ 0) ∧ (x + 1 ≠ 0) ∧ (x + 2 > 0)

theorem domain_is_correct :
  { x : ℝ | domain_of_function x } = { x : ℝ | -2 < x ∧ x ≤ 3 ∧ x ≠ -1 } :=
by
  sorry

end domain_is_correct_l180_180312


namespace event_complementary_and_mutually_exclusive_l180_180162

def students : Finset (String × String) := 
  { ("boy", "1"), ("boy", "2"), ("boy", "3"), ("girl", "1"), ("girl", "2") }

def event_at_least_one_girl (s : Finset (String × String)) : Prop :=
  ∃ x ∈ s, (x.1 = "girl")

def event_all_boys (s : Finset (String × String)) : Prop :=
  ∀ x ∈ s, (x.1 = "boy")

def two_students (s : Finset (String × String)) : Prop :=
  s.card = 2

theorem event_complementary_and_mutually_exclusive :
  ∀ s: Finset (String × String), two_students s → 
  (event_at_least_one_girl s ↔ ¬ event_all_boys s) ∧ 
  (event_all_boys s ↔ ¬ event_at_least_one_girl s) :=
sorry

end event_complementary_and_mutually_exclusive_l180_180162


namespace height_of_water_a_height_of_water_b_height_of_water_c_l180_180217

noncomputable def edge_length : ℝ := 10  -- Edge length of the cube in cm.
noncomputable def angle_deg : ℝ := 20   -- Angle in degrees.

noncomputable def volume_a : ℝ := 100  -- Volume in cm^3 for case a)
noncomputable def height_a : ℝ := 2.53  -- Height in cm for case a)

noncomputable def volume_b : ℝ := 450  -- Volume in cm^3 for case b)
noncomputable def height_b : ℝ := 5.94  -- Height in cm for case b)

noncomputable def volume_c : ℝ := 900  -- Volume in cm^3 for case c)
noncomputable def height_c : ℝ := 10.29  -- Height in cm for case c)

theorem height_of_water_a :
  ∀ (edge_length angle_deg volume_a : ℝ), volume_a = 100 → height_a = 2.53 := by 
  sorry

theorem height_of_water_b :
  ∀ (edge_length angle_deg volume_b : ℝ), volume_b = 450 → height_b = 5.94 := by 
  sorry

theorem height_of_water_c :
  ∀ (edge_length angle_deg volume_c : ℝ), volume_c = 900 → height_c = 10.29 := by 
  sorry

end height_of_water_a_height_of_water_b_height_of_water_c_l180_180217


namespace minimum_value_of_abs_phi_l180_180369

theorem minimum_value_of_abs_phi (φ : ℝ) :
  (∃ k : ℤ, φ = k * π - (13 * π) / 6) → 
  ∃ φ_min : ℝ, 0 ≤ φ_min ∧ φ_min = abs φ ∧ φ_min = π / 6 :=
by
  sorry

end minimum_value_of_abs_phi_l180_180369


namespace least_positive_value_tan_inv_k_l180_180023

theorem least_positive_value_tan_inv_k 
  (a b : ℝ) 
  (x : ℝ) 
  (h1 : Real.tan x = a / b) 
  (h2 : Real.tan (2 * x) = 2 * b / (a + 2 * b)) 
  : x = Real.arctan 1 := 
sorry

end least_positive_value_tan_inv_k_l180_180023


namespace division_of_fractions_l180_180583

theorem division_of_fractions :
  (5 / 6) / (9 / 10) = 25 / 27 :=
by sorry

end division_of_fractions_l180_180583


namespace max_side_length_triangle_l180_180386

def triangle_with_max_side_length (a b c : ℕ) (ha : a ≠ b ∧ b ≠ c ∧ c ≠ a) (hper : a + b + c = 30) : Prop :=
  a > b ∧ a > c ∧ a = 14

theorem max_side_length_triangle : ∃ a b c : ℕ, 
  a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ a + b + c = 30 ∧ a > b ∧ a > c ∧ a = 14 :=
sorry

end max_side_length_triangle_l180_180386


namespace evaluate_fraction_l180_180890

theorem evaluate_fraction :
  (0.5^2 + 0.05^3) / 0.005^3 = 2000100 := by
  sorry

end evaluate_fraction_l180_180890


namespace ratio_of_pipe_lengths_l180_180901

theorem ratio_of_pipe_lengths (L S : ℕ) (h1 : L + S = 177) (h2 : L = 118) (h3 : ∃ k : ℕ, L = k * S) : L / S = 2 := 
by 
  sorry

end ratio_of_pipe_lengths_l180_180901


namespace cuboid_distance_properties_l180_180186

theorem cuboid_distance_properties (cuboid : Type) :
  (∃ P : cuboid → ℝ, ∀ V1 V2 : cuboid, P V1 = P V2) ∧
  ¬ (∃ Q : cuboid → ℝ, ∀ E1 E2 : cuboid, Q E1 = Q E2) ∧
  ¬ (∃ R : cuboid → ℝ, ∀ F1 F2 : cuboid, R F1 = R F2) := 
sorry

end cuboid_distance_properties_l180_180186


namespace count_three_digit_integers_with_tens_7_divisible_by_25_l180_180551

theorem count_three_digit_integers_with_tens_7_divisible_by_25 :
  ∃ n, n = 33 ∧ ∃ k1 k2 : ℕ, 175 = 25 * k1 ∧ 975 = 25 * k2 ∧ (k2 - k1 + 1 = n) :=
by
  sorry

end count_three_digit_integers_with_tens_7_divisible_by_25_l180_180551


namespace difference_of_squares_650_550_l180_180465

theorem difference_of_squares_650_550 : 650^2 - 550^2 = 120000 :=
by sorry

end difference_of_squares_650_550_l180_180465


namespace norma_found_cards_l180_180934

/-- Assume Norma originally had 88.0 cards. -/
def original_cards : ℝ := 88.0

/-- Assume Norma now has a total of 158 cards. -/
def total_cards : ℝ := 158

/-- Prove that Norma found 70 cards. -/
theorem norma_found_cards : total_cards - original_cards = 70 := 
by
  sorry

end norma_found_cards_l180_180934


namespace solve_for_n_l180_180822

theorem solve_for_n (n : ℕ) (h1 : n > 2)
  (h2 : 6 * (n - 2) ^ 2 = 12 * 12 * (n - 2)) :
  n = 26 :=
by {
  sorry
}

end solve_for_n_l180_180822


namespace percentage_reduction_in_price_l180_180691

theorem percentage_reduction_in_price (P R : ℝ) (hR : R = 2.953846153846154)
  (h_condition : ∃ P, 65 / 12 * R = 40 - 24 / P) :
  ((P - R) / P) * 100 = 33.3 := by
  sorry

end percentage_reduction_in_price_l180_180691


namespace least_value_of_p_plus_q_l180_180575

theorem least_value_of_p_plus_q (p q : ℕ) (hp : 1 < p) (hq : 1 < q) (h : 17 * (p + 1) = 28 * (q + 1)) : p + q = 135 :=
  sorry

end least_value_of_p_plus_q_l180_180575


namespace average_monthly_growth_rate_l180_180390

variable (x : ℝ)

-- Conditions
def turnover_January : ℝ := 36
def turnover_March : ℝ := 48

-- Theorem statement that corresponds to the problem's conditions and question
theorem average_monthly_growth_rate :
  turnover_January * (1 + x)^2 = turnover_March :=
sorry

end average_monthly_growth_rate_l180_180390


namespace solve_for_k_l180_180218

theorem solve_for_k (x y : ℤ) (h₁ : x = 1) (h₂ : y = k) (h₃ : 2 * x + y = 6) : k = 4 :=
by 
  sorry

end solve_for_k_l180_180218


namespace add_sub_decimals_l180_180685

theorem add_sub_decimals :
  (0.513 + 0.0067 - 0.048 = 0.4717) :=
by
  sorry

end add_sub_decimals_l180_180685


namespace pow_ge_double_plus_one_l180_180272

theorem pow_ge_double_plus_one (n : ℕ) (h : n ≥ 3) : 2^n ≥ 2 * (n + 1) :=
sorry

end pow_ge_double_plus_one_l180_180272


namespace time_to_cross_first_platform_l180_180409

noncomputable def train_length : ℝ := 30
noncomputable def first_platform_length : ℝ := 180
noncomputable def second_platform_length : ℝ := 250
noncomputable def time_second_platform : ℝ := 20

noncomputable def train_speed : ℝ :=
(train_length + second_platform_length) / time_second_platform

noncomputable def time_first_platform : ℝ :=
(train_length + first_platform_length) / train_speed

theorem time_to_cross_first_platform :
  time_first_platform = 15 :=
by
  sorry

end time_to_cross_first_platform_l180_180409


namespace hillside_camp_boys_percentage_l180_180668

theorem hillside_camp_boys_percentage (B G : ℕ) 
  (h1 : B + G = 60) 
  (h2 : G = 6) : (B: ℕ) / 60 * 100 = 90 :=
by
  sorry

end hillside_camp_boys_percentage_l180_180668


namespace nanometers_to_scientific_notation_l180_180246

   theorem nanometers_to_scientific_notation :
     (0.000000001 : Float) = 1 * 10 ^ (-9) :=
   by
     sorry
   
end nanometers_to_scientific_notation_l180_180246


namespace algebraic_expression_value_l180_180631

theorem algebraic_expression_value (x y : ℝ) 
  (h1 : x - y = -2) 
  (h2 : 2 * x + y = -1) : 
  (x - y)^2 - (x - 2 * y) * (x + 2 * y) = 7 :=
by {
  sorry
}

end algebraic_expression_value_l180_180631


namespace perimeter_to_side_ratio_l180_180825

variable (a b c h_a r : ℝ) (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < h_a ∧ 0 < r ∧ a + b > c ∧ a + c > b ∧ b + c > a)

theorem perimeter_to_side_ratio (P : ℝ) (hP : P = a + b + c) :
  P / a = h_a / r := by
  sorry

end perimeter_to_side_ratio_l180_180825


namespace sum_xy_sum_inv_squared_geq_nine_four_l180_180039

variable {x y z : ℝ}

theorem sum_xy_sum_inv_squared_geq_nine_four (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x * y + y * z + z * x) * (1 / (x + y)^2 + 1 / (y + z)^2 + 1 / (z + x)^2) ≥ 9 / 4 :=
by sorry

end sum_xy_sum_inv_squared_geq_nine_four_l180_180039


namespace find_theta_l180_180565

variable (x : ℝ) (θ : ℝ) (k : ℤ)

def condition := (3 - 3^(-|x - 3|))^2 = 3 - Real.cos θ

theorem find_theta (h : condition x θ) : ∃ k : ℤ, θ = (2 * k + 1) * Real.pi :=
by
  sorry

end find_theta_l180_180565


namespace find_c_l180_180857

theorem find_c (a b c : ℝ) (h1 : ∃ x y : ℝ, x = a * (y - 2)^2 + 3 ∧ (x,y) = (3,2))
  (h2 : (1 : ℝ) = a * ((4 : ℝ) - 2)^2 + 3) : c = 1 :=
sorry

end find_c_l180_180857


namespace a6_equals_8_l180_180379

-- Defining Sn as given in the condition
def S (n : ℕ) : ℤ :=
  if n = 0 then 0
  else n^2 - 3*n

-- Defining a_n in terms of the differences stated in the solution
def a (n : ℕ) : ℤ := S n - S (n-1)

-- The problem statement to prove
theorem a6_equals_8 : a 6 = 8 :=
by
  sorry

end a6_equals_8_l180_180379


namespace find_5_minus_a_l180_180868

-- Define the problem conditions as assumptions
variable (a b : ℤ)
variable (h1 : 5 + a = 6 - b)
variable (h2 : 3 + b = 8 + a)

-- State the theorem we want to prove
theorem find_5_minus_a : 5 - a = 7 :=
by
  sorry

end find_5_minus_a_l180_180868


namespace compute_expression_l180_180471

variables (a b c : ℝ)

theorem compute_expression (h1 : a - b = 2) (h2 : a + c = 6) : 
  (2 * a + b + c) - 2 * (a - b - c) = 12 :=
by
  sorry

end compute_expression_l180_180471


namespace spring_length_5kg_weight_l180_180563

variable {x y : ℝ}

-- Given conditions
def spring_length_no_weight : y = 6 := sorry
def spring_length_4kg_weight : y = 7.2 := sorry

-- The problem: to find the length of the spring for 5 kilograms
theorem spring_length_5kg_weight :
  (∃ (k b : ℝ), (∀ x, y = k * x + b) ∧ (b = 6) ∧ (4 * k + b = 7.2)) →
  y = 0.3 * 5 + 6 :=
  sorry

end spring_length_5kg_weight_l180_180563


namespace area_of_square_l180_180100

-- Conditions: Points A (5, -2) and B (5, 3) are adjacent corners of a square.
def A : ℝ × ℝ := (5, -2)
def B : ℝ × ℝ := (5, 3)

-- The statement to prove that the area of the square formed by these points is 25.
theorem area_of_square : (∃ s : ℝ, s = Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)) → s^2 = 25 :=
sorry

end area_of_square_l180_180100


namespace ann_age_l180_180918

variable (A T : ℕ)

-- Condition 1: Tom is currently two times older than Ann
def tom_older : Prop := T = 2 * A

-- Condition 2: The sum of their ages 10 years later will be 38
def age_sum_later : Prop := (A + 10) + (T + 10) = 38

-- Theorem: Ann's current age
theorem ann_age (h1 : tom_older A T) (h2 : age_sum_later A T) : A = 6 :=
by
  sorry

end ann_age_l180_180918


namespace bryan_total_earnings_l180_180200

-- Declare the data given in the problem:
def num_emeralds : ℕ := 3
def num_rubies : ℕ := 2
def num_sapphires : ℕ := 3

def price_emerald : ℝ := 1785
def price_ruby : ℝ := 2650
def price_sapphire : ℝ := 2300

-- Calculate the total earnings from each type of stone:
def total_emeralds : ℝ := num_emeralds * price_emerald
def total_rubies : ℝ := num_rubies * price_ruby
def total_sapphires : ℝ := num_sapphires * price_sapphire

-- Calculate the overall total earnings:
def total_earnings : ℝ := total_emeralds + total_rubies + total_sapphires

-- Prove that Bryan got 17555 dollars in total:
theorem bryan_total_earnings : total_earnings = 17555 := by
  simp [total_earnings, total_emeralds, total_rubies, total_sapphires, num_emeralds, num_rubies, num_sapphires, price_emerald, price_ruby, price_sapphire]
  sorry

end bryan_total_earnings_l180_180200


namespace find_x_l180_180831

variables (a b c d x y : ℚ)

noncomputable def modified_fraction (a b x y : ℚ) := (a + x) / (b + y)

theorem find_x (h1 : a ≠ b) (h2 : b ≠ 0) (h3 : modified_fraction a b x y = c / d) :
  x = (b * c - a * d + y * c) / d :=
by
  sorry

end find_x_l180_180831


namespace cheapest_salon_option_haily_l180_180742

theorem cheapest_salon_option_haily : 
  let gustran_haircut := 45
  let gustran_facial := 22
  let gustran_nails := 30
  let gustran_foot_spa := 15
  let gustran_massage := 50
  let gustran_total := gustran_haircut + gustran_facial + gustran_nails + gustran_foot_spa + gustran_massage
  let gustran_discount := 0.20
  let gustran_final := gustran_total * (1 - gustran_discount)

  let barbara_nails := 40
  let barbara_haircut := 30
  let barbara_facial := 28
  let barbara_foot_spa := 18
  let barbara_massage := 45
  let barbara_total :=
      barbara_nails + barbara_haircut + (barbara_facial * 0.5) + barbara_foot_spa + (barbara_massage * 0.5)

  let fancy_haircut := 34
  let fancy_facial := 30
  let fancy_nails := 20
  let fancy_foot_spa := 25
  let fancy_massage := 60
  let fancy_total := fancy_haircut + fancy_facial + fancy_nails + fancy_foot_spa + fancy_massage
  let fancy_discount := 15
  let fancy_final := fancy_total - fancy_discount

  let avg_haircut := (gustran_haircut + barbara_haircut + fancy_haircut) / 3
  let avg_facial := (gustran_facial + barbara_facial + fancy_facial) / 3
  let avg_nails := (gustran_nails + barbara_nails + fancy_nails) / 3
  let avg_foot_spa := (gustran_foot_spa + barbara_foot_spa + fancy_foot_spa) / 3
  let avg_massage := (gustran_massage + barbara_massage + fancy_massage) / 3

  let luxury_haircut := avg_haircut * 1.10
  let luxury_facial := avg_facial * 1.10
  let luxury_nails := avg_nails * 1.10
  let luxury_foot_spa := avg_foot_spa * 1.10
  let luxury_massage := avg_massage * 1.10
  let luxury_total := luxury_haircut + luxury_facial + luxury_nails + luxury_foot_spa + luxury_massage
  let luxury_discount := 20
  let luxury_final := luxury_total - luxury_discount

  gustran_final > barbara_total ∧ barbara_total < fancy_final ∧ barbara_total < luxury_final := 
by 
  sorry

end cheapest_salon_option_haily_l180_180742


namespace kerry_age_l180_180167

theorem kerry_age (cost_per_box : ℝ) (boxes_bought : ℕ) (candles_per_box : ℕ) (cakes : ℕ) 
  (total_cost : ℝ) (total_candles : ℕ) (candles_per_cake : ℕ) (age : ℕ) :
  cost_per_box = 2.5 →
  boxes_bought = 2 →
  candles_per_box = 12 →
  cakes = 3 →
  total_cost = 5 →
  total_cost = boxes_bought * cost_per_box →
  total_candles = boxes_bought * candles_per_box →
  candles_per_cake = total_candles / cakes →
  age = candles_per_cake →
  age = 8 :=
by
  intros
  sorry

end kerry_age_l180_180167


namespace sum_of_side_lengths_l180_180223

theorem sum_of_side_lengths (p q r : ℕ) (h : p = 8 ∧ q = 1 ∧ r = 5) 
    (area_ratio : 128 / 50 = 64 / 25) 
    (side_length_ratio : 8 / 5 = Real.sqrt (128 / 50)) :
    p + q + r = 14 := 
by 
  sorry

end sum_of_side_lengths_l180_180223


namespace Anne_is_15_pounds_heavier_l180_180807

def Anne_weight : ℕ := 67
def Douglas_weight : ℕ := 52

theorem Anne_is_15_pounds_heavier : Anne_weight - Douglas_weight = 15 := by
  sorry

end Anne_is_15_pounds_heavier_l180_180807


namespace boarders_joined_l180_180475

theorem boarders_joined (initial_boarders : ℕ) (initial_day_scholars : ℕ) (final_ratio_num : ℕ) (final_ratio_denom : ℕ) (new_boarders : ℕ)
  (initial_ratio_boarders_to_day_scholars : initial_boarders * 16 = 7 * initial_day_scholars)
  (initial_boarders_eq : initial_boarders = 560)
  (final_ratio : (initial_boarders + new_boarders) * 2 = final_day_scholars)
  (day_scholars_eq : initial_day_scholars = 1280) : 
  new_boarders = 80 := by
  sorry

end boarders_joined_l180_180475


namespace value_of_x_l180_180243

theorem value_of_x (x : ℝ) : (12 - x)^3 = x^3 → x = 12 :=
by
  sorry

end value_of_x_l180_180243


namespace graph_passes_quadrants_l180_180303

theorem graph_passes_quadrants {x y : ℝ} (h : y = -x - 2) :
  -- Statement that the graph passes through the second, third, and fourth quadrants.
  (∃ (x : ℝ), x > 0 ∧ (∃ (y : ℝ), y < 0 ∧ y = -x - 2)) ∧
  (∃ (x : ℝ), x < 0 ∧ (∃ (y : ℝ), y < 0 ∧ y = -x - 2)) ∧
  (∃ (x : ℝ), x > 0 ∧ (∃ (y : ℝ), y > 0 ∧ y = -x - 2)) :=
by
  sorry

end graph_passes_quadrants_l180_180303


namespace modulus_of_complex_l180_180185

open Complex

theorem modulus_of_complex (z : ℂ) (h : z = 1 - (1 / Complex.I)) : Complex.abs z = Real.sqrt 2 :=
by
  sorry

end modulus_of_complex_l180_180185


namespace n_cubed_plus_two_not_divisible_by_nine_l180_180865

theorem n_cubed_plus_two_not_divisible_by_nine (n : ℕ) : ¬ (9 ∣ n^3 + 2) :=
sorry

end n_cubed_plus_two_not_divisible_by_nine_l180_180865


namespace weight_of_five_single_beds_l180_180032

-- Define the problem conditions and the goal
theorem weight_of_five_single_beds :
  ∃ S D : ℝ, (2 * S + 4 * D = 100) ∧ (D = S + 10) → (5 * S = 50) :=
by
  sorry

end weight_of_five_single_beds_l180_180032


namespace solution_set_for_a1_find_a_if_min_value_is_4_l180_180760

noncomputable def f (a x : ℝ) : ℝ := |2 * x - 1| + |a * x - 5|

theorem solution_set_for_a1 : 
  { x : ℝ | f 1 x ≥ 9 } = { x : ℝ | x ≤ -1 ∨ x > 5 } :=
sorry

theorem find_a_if_min_value_is_4 :
  ∃ a : ℝ, (0 < a ∧ a < 5) ∧ (∀ x : ℝ, f a x ≥ 4) ∧ (∃ x : ℝ, f a x = 4) ∧ a = 2 :=
sorry

end solution_set_for_a1_find_a_if_min_value_is_4_l180_180760


namespace find_last_even_number_l180_180177

theorem find_last_even_number (n : ℕ) (h : 4 * (n * (n + 1) * (2 * n + 1) / 6) = 560) : 2 * n = 14 :=
by
  sorry

end find_last_even_number_l180_180177


namespace baby_plants_produced_l180_180083

theorem baby_plants_produced (baby_plants_per_time: ℕ) (times_per_year: ℕ) (years: ℕ) (total_babies: ℕ) :
  baby_plants_per_time = 2 ∧ times_per_year = 2 ∧ years = 4 ∧ total_babies = baby_plants_per_time * times_per_year * years → 
  total_babies = 16 :=
by
  intro h
  rcases h with ⟨h1, h2, h3, h4⟩
  sorry

end baby_plants_produced_l180_180083


namespace find_x_l180_180460

theorem find_x (x : ℤ) (h : (4 + x) / (6 + x) = (2 + x) / (3 + x)) : x = 0 :=
by
  sorry

end find_x_l180_180460


namespace distinct_solutions_sub_l180_180598

open Nat Real

theorem distinct_solutions_sub (p q : Real) (hpq_distinct : p ≠ q) (h_eqn_p : (p - 4) * (p + 4) = 17 * p - 68) (h_eqn_q : (q - 4) * (q + 4) = 17 * q - 68) (h_p_gt_q : p > q) : p - q = 9 := 
sorry

end distinct_solutions_sub_l180_180598


namespace value_of_a_if_lines_are_parallel_l180_180588

theorem value_of_a_if_lines_are_parallel (a : ℝ) :
  (∀ (x y : ℝ), x + a*y - 7 = 0 → (a+1)*x + 2*y - 14 = 0) → a = -2 :=
sorry

end value_of_a_if_lines_are_parallel_l180_180588


namespace sin_cos_quotient_l180_180189

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x
noncomputable def f_prime (x : ℝ) : ℝ := Real.cos x - Real.sin x

theorem sin_cos_quotient 
  (x : ℝ)
  (h : f_prime x = 3 * f x) 
  : (Real.sin x ^ 2 - 3) / (Real.cos x ^ 2 + 1) = -14 / 9 := 
by 
  sorry

end sin_cos_quotient_l180_180189


namespace average_score_l180_180183

theorem average_score (m n : ℝ) (hm : m ≥ 0) (hn : n ≥ 0) :
  (20 * m + 23 * n) / (20 + 23) = 20 / 43 * m + 23 / 43 * n := sorry

end average_score_l180_180183


namespace luke_total_points_l180_180672

theorem luke_total_points (rounds : ℕ) (points_per_round : ℕ) (total_points : ℕ) 
  (h1 : rounds = 177) (h2 : points_per_round = 46) : 
  total_points = 8142 := by
  have h : total_points = rounds * points_per_round := by sorry
  rw [h1, h2] at h
  exact h

end luke_total_points_l180_180672


namespace ratio_of_female_to_male_officers_on_duty_l180_180617

theorem ratio_of_female_to_male_officers_on_duty 
    (p : ℝ) (T : ℕ) (F : ℕ) 
    (hp : p = 0.19) (hT : T = 152) (hF : F = 400) : 
    (76 / 76) = 1 :=
by
  sorry

end ratio_of_female_to_male_officers_on_duty_l180_180617


namespace sequence_a4_l180_180295

theorem sequence_a4 (S : ℕ → ℚ) (a : ℕ → ℚ) 
  (hS : ∀ n, S n = (n + 1) / (n + 2))
  (hS0 : S 0 = a 0)
  (hSn : ∀ n, S (n + 1) = S n + a (n + 1)) :
  a 4 = 1 / 30 := 
sorry

end sequence_a4_l180_180295


namespace relationship_p_q_no_linear_term_l180_180692

theorem relationship_p_q_no_linear_term (p q : ℝ) :
  (∀ x : ℝ, (x^2 - p * x + q) * (x - 3) = x^3 + (-p - 3) * x^2 + (3 * p + q) * x - 3 * q) 
  → (3 * p + q = 0) → (q + 3 * p = 0) :=
by
  intro h_expansion coeff_zero
  sorry

end relationship_p_q_no_linear_term_l180_180692


namespace quintuple_sum_not_less_than_l180_180605

theorem quintuple_sum_not_less_than (a : ℝ) : 5 * (a + 3) ≥ 6 :=
by
  -- Insert proof here
  sorry

end quintuple_sum_not_less_than_l180_180605


namespace correct_reaction_for_phosphoric_acid_l180_180020

-- Define the reactions
def reaction_A := "H₂ + 2OH⁻ - 2e⁻ = 2H₂O"
def reaction_B := "H₂ - 2e⁻ = 2H⁺"
def reaction_C := "O₂ + 4H⁺ + 4e⁻ = 2H₂O"
def reaction_D := "O₂ + 2H₂O + 4e⁻ = 4OH⁻"

-- Define the condition that the electrolyte used is phosphoric acid
def electrolyte := "phosphoric acid"

-- Define the correct reaction
def correct_negative_electrode_reaction := reaction_B

-- Theorem to state that given the conditions above, the correct reaction is B
theorem correct_reaction_for_phosphoric_acid :
  (∃ r, r = reaction_B ∧ electrolyte = "phosphoric acid") :=
by
  sorry

end correct_reaction_for_phosphoric_acid_l180_180020


namespace cube_roof_ratio_proof_l180_180273

noncomputable def cube_roof_edge_ratio : Prop :=
  ∃ (a b : ℝ), (∃ isosceles_triangles symmetrical_trapezoids : ℝ, isosceles_triangles = 2 ∧ symmetrical_trapezoids = 2)
  ∧ (∀ edge : ℝ, edge = a)
  ∧ (∀ face1 face2 : ℝ, face1 = face2)
  ∧ b = (Real.sqrt 5 - 1) / 2 * a

theorem cube_roof_ratio_proof : cube_roof_edge_ratio :=
sorry

end cube_roof_ratio_proof_l180_180273


namespace twenty_percent_l180_180990

-- Given condition
def condition (X : ℝ) : Prop := 0.4 * X = 160

-- Theorem to show that 20% of X equals 80 given the condition
theorem twenty_percent (X : ℝ) (h : condition X) : 0.2 * X = 80 :=
by sorry

end twenty_percent_l180_180990


namespace possible_point_counts_l180_180903

theorem possible_point_counts (r b g : ℕ) (d_RB d_RG d_BG : ℕ) :
    r + b + g = 15 →
    r * b * d_RB = 51 →
    r * g * d_RG = 39 →
    b * g * d_BG = 1 →
    (r = 13 ∧ b = 1 ∧ g = 1) ∨ (r = 8 ∧ b = 4 ∧ g = 3) :=
by {
    sorry
}

end possible_point_counts_l180_180903


namespace statement_I_statement_II_statement_III_statement_IV_statement_V_statement_VI_statement_VII_statement_VIII_statement_IX_statement_X_statement_XI_statement_XII_l180_180604

-- Definitions of conditions
structure Polygon (n : ℕ) :=
  (sides : Fin n → ℝ)
  (angles : Fin n → ℝ)

def circumscribed (P : Polygon n) : Prop := sorry -- Definition of circumscribed
def inscribed (P : Polygon n) : Prop := sorry -- Definition of inscribed
def equal_sides (P : Polygon n) : Prop := ∀ i j, P.sides i = P.sides j
def equal_angles (P : Polygon n) : Prop := ∀ i j, P.angles i = P.angles j

-- The statements to be proved
theorem statement_I : ∀ P : Polygon n, circumscribed P → equal_sides P → equal_angles P := sorry

theorem statement_II : ∃ P : Polygon n, inscribed P ∧ equal_sides P ∧ ¬equal_angles P := sorry

theorem statement_III : ∃ P : Polygon n, circumscribed P ∧ equal_angles P ∧ ¬equal_sides P := sorry

theorem statement_IV : ∀ P : Polygon n, inscribed P → equal_angles P → equal_sides P := sorry

theorem statement_V : ∀ (P : Polygon 5), circumscribed P → equal_sides P → equal_angles P := sorry

theorem statement_VI : ∀ (P : Polygon 6), circumscribed P → equal_sides P → equal_angles P := sorry

theorem statement_VII : ∀ (P : Polygon 5), inscribed P → equal_sides P → equal_angles P := sorry

theorem statement_VIII : ∃ (P : Polygon 6), inscribed P ∧ equal_sides P ∧ ¬equal_angles P := sorry

theorem statement_IX : ∀ (P : Polygon 5), circumscribed P → equal_angles P → equal_sides P := sorry

theorem statement_X : ∃ (P : Polygon 6), circumscribed P ∧ equal_angles P ∧ ¬equal_sides P := sorry

theorem statement_XI : ∀ (P : Polygon 5), inscribed P → equal_angles P → equal_sides P := sorry

theorem statement_XII : ∀ (P : Polygon 6), inscribed P → equal_angles P → equal_sides P := sorry

end statement_I_statement_II_statement_III_statement_IV_statement_V_statement_VI_statement_VII_statement_VIII_statement_IX_statement_X_statement_XI_statement_XII_l180_180604


namespace find_x_l180_180355

noncomputable def solution_x (m n y : ℝ) (m_gt_3n : m > 3 * n) : ℝ :=
  (n * m) / (m + n)

theorem find_x (m n y : ℝ) (m_gt_3n : m > 3 * n) :
  let initial_acid := m * (m / 100)
  let final_volume := m + (solution_x m n y m_gt_3n) + y
  let final_acid := (m - n) / 100 * final_volume
  initial_acid = final_acid → 
  solution_x m n y m_gt_3n = (n * m) / (m + n) :=
by sorry

end find_x_l180_180355


namespace total_time_iggy_runs_correct_l180_180332

noncomputable def total_time_iggy_runs : ℝ :=
  let monday_time := 3 * (10 + 1 + 0.5);
  let tuesday_time := 5 * (9 + 1 + 1);
  let wednesday_time := 7 * (12 - 2 + 2);
  let thursday_time := 10 * (8 + 2 + 4);
  let friday_time := 4 * (10 + 0.25);
  monday_time + tuesday_time + wednesday_time + thursday_time + friday_time

theorem total_time_iggy_runs_correct : total_time_iggy_runs = 354.5 := by
  sorry

end total_time_iggy_runs_correct_l180_180332


namespace remainder_when_divided_by_x_minus_4_l180_180837

-- Define the polynomial function
def f (x : ℝ) : ℝ := x^5 - 8 * x^4 + 15 * x^3 + 20 * x^2 - 5 * x - 20

-- State the problem as a theorem
theorem remainder_when_divided_by_x_minus_4 : 
    (f 4 = 216) := 
by 
    -- Calculation goes here
    sorry

end remainder_when_divided_by_x_minus_4_l180_180837


namespace sum_of_interior_angles_heptagon_l180_180590

theorem sum_of_interior_angles_heptagon (n : ℕ) (h : n = 7) : (n - 2) * 180 = 900 := by
  sorry

end sum_of_interior_angles_heptagon_l180_180590


namespace min_questions_to_determine_product_50_numbers_l180_180296

/-- Prove that to uniquely determine the product of 50 numbers each either +1 or -1 
arranged on the circumference of a circle by asking for the product of three 
consecutive numbers, one must ask a minimum of 50 questions. -/
theorem min_questions_to_determine_product_50_numbers : 
  ∀ (a : ℕ → ℤ), (∀ i, a i = 1 ∨ a i = -1) → 
  (∀ i, ∃ b : ℤ, b = a i * a (i+1) * a (i+2)) → 
  ∃ n, n = 50 :=
by
  sorry

end min_questions_to_determine_product_50_numbers_l180_180296


namespace find_unknown_rate_l180_180959

variable {x : ℝ}

theorem find_unknown_rate (h : (3 * 100 + 1 * 150 + 2 * x) / 6 = 150) : x = 225 :=
by 
  sorry

end find_unknown_rate_l180_180959


namespace smallest_next_divisor_l180_180352

noncomputable def is_even (n : ℕ) : Prop := n % 2 = 0

noncomputable def is_4_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

noncomputable def has_divisor_323 (n : ℕ) : Prop := 323 ∣ n

theorem smallest_next_divisor (n : ℕ) (h1 : is_even n) (h2 : is_4_digit n) (h3 : has_divisor_323 n) :
  ∃ m : ℕ, m > 323 ∧ m ∣ n ∧ (∀ k : ℕ, k > 323 ∧ k < m → ¬ k ∣ n) ∧ m = 340 :=
sorry

end smallest_next_divisor_l180_180352


namespace halfway_between_one_eighth_and_one_tenth_l180_180210

theorem halfway_between_one_eighth_and_one_tenth :
  (1 / 8 + 1 / 10) / 2 = 9 / 80 :=
by
  sorry

end halfway_between_one_eighth_and_one_tenth_l180_180210


namespace largest_integral_value_l180_180505

theorem largest_integral_value (y : ℤ) (h1 : 0 < y) (h2 : (1 : ℚ)/4 < y / 7) (h3 : y / 7 < 7 / 11) : y = 4 :=
sorry

end largest_integral_value_l180_180505


namespace line_l_equation_symmetrical_line_equation_l180_180549

theorem line_l_equation (x y : ℝ) (h₁ : 3 * x + 4 * y - 2 = 0) (h₂ : 2 * x + y + 2 = 0) :
  2 * x + y + 2 = 0 :=
sorry

theorem symmetrical_line_equation (x y : ℝ) :
  (2 * x + y + 2 = 0) → (2 * x + y - 2 = 0) :=
sorry

end line_l_equation_symmetrical_line_equation_l180_180549


namespace angle_ACE_is_38_l180_180451

noncomputable def measure_angle_ACE (A B C D E : Type) : Prop :=
  let angle_ABC := 55
  let angle_BCA := 38
  let angle_BAC := 87
  let angle_ABD := 125
  (angle_ABC + angle_ABD = 180) → -- supplementary condition
  (angle_BAC = 87) → -- given angle at BAC
  (let angle_ACB := 180 - angle_BAC - angle_ABC;
   angle_ACB = angle_BCA ∧  -- derived angle at BCA
   angle_ACB = 38) → -- target angle
  (angle_BCA = 38) -- final result that needs to be proven

theorem angle_ACE_is_38 {A B C D E : Type} :
  measure_angle_ACE A B C D E :=
by
  sorry

end angle_ACE_is_38_l180_180451


namespace ellipse_focal_length_l180_180953

theorem ellipse_focal_length (k : ℝ) :
  (∀ x y : ℝ, x^2 / k + y^2 / 2 = 1) →
  (∃ c : ℝ, 2 * c = 2 ∧ (k = 1 ∨ k = 3)) :=
by
  -- Given condition: equation of ellipse and focal length  
  intro h  
  sorry

end ellipse_focal_length_l180_180953


namespace num_integer_solutions_eq_3_l180_180973

theorem num_integer_solutions_eq_3 :
  ∃ (S : Finset (ℤ × ℤ)), 
  (∀ (x y : ℤ), ((2 * x^2) + (x * y) + (y^2) - x + 2 * y + 1 = 0 ↔ (x, y) ∈ S)) ∧ 
  S.card = 3 :=
sorry

end num_integer_solutions_eq_3_l180_180973


namespace slant_height_base_plane_angle_l180_180049

noncomputable def angle_between_slant_height_and_base_plane (R : ℝ) : ℝ :=
  Real.arcsin ((Real.sqrt 13 - 1) / 3)

theorem slant_height_base_plane_angle (R : ℝ) (h : R = R) : angle_between_slant_height_and_base_plane R = Real.arcsin ((Real.sqrt 13 - 1) / 3) :=
by
  -- Here we assume that the mathematical conditions and transformations hold true.
  -- According to the solution steps provided:
  -- We found that γ = arcsin ((sqrt(13) - 1) / 3)
  sorry

end slant_height_base_plane_angle_l180_180049


namespace red_sea_glass_pieces_l180_180318

theorem red_sea_glass_pieces (R : ℕ) 
    (h_bl : ∃ g : ℕ, g = 12) 
    (h_rose_red : ∃ r_b : ℕ, r_b = 9)
    (h_rose_blue : ∃ b : ℕ, b = 11) 
    (h_dorothy_red : 2 * (R + 9) + 3 * 11 = 57) : R = 3 :=
  by
    sorry

end red_sea_glass_pieces_l180_180318


namespace equation_of_parallel_line_l180_180803

theorem equation_of_parallel_line 
  (l : ℝ → ℝ) 
  (passes_through : l 0 = 7) 
  (parallel_to : ∀ x : ℝ, l x = -4 * x + (l 0)) :
  ∀ x : ℝ, l x = -4 * x + 7 :=
by
  sorry

end equation_of_parallel_line_l180_180803


namespace ratio_of_sum_l180_180263

theorem ratio_of_sum (a b c : ℚ) (h1 : b / a = 3) (h2 : c / b = 4) : 
  (2 * a + 3 * b) / (b + 2 * c) = 11 / 27 := 
by
  sorry

end ratio_of_sum_l180_180263


namespace rabbit_catch_up_time_l180_180297

theorem rabbit_catch_up_time :
  let rabbit_speed := 25 -- miles per hour
  let cat_speed := 20 -- miles per hour
  let head_start := 15 / 60 -- hours, which is 0.25 hours
  let initial_distance := cat_speed * head_start
  let relative_speed := rabbit_speed - cat_speed
  initial_distance / relative_speed = 1 := by
  sorry

end rabbit_catch_up_time_l180_180297


namespace total_population_increase_l180_180828
-- Import the required library

-- Define the conditions for Region A and Region B
def regionA_births_0_14 (time: ℕ) := time / 20
def regionA_births_15_64 (time: ℕ) := time / 30
def regionB_births_0_14 (time: ℕ) := time / 25
def regionB_births_15_64 (time: ℕ) := time / 35

-- Define the total number of people in each age group for both regions
def regionA_population_0_14 := 2000
def regionA_population_15_64 := 6000
def regionB_population_0_14 := 1500
def regionB_population_15_64 := 5000

-- Define the total time in seconds
def total_time := 25 * 60

-- Proof statement
theorem total_population_increase : 
  regionA_population_0_14 * regionA_births_0_14 total_time +
  regionA_population_15_64 * regionA_births_15_64 total_time +
  regionB_population_0_14 * regionB_births_0_14 total_time +
  regionB_population_15_64 * regionB_births_15_64 total_time = 227 := 
by sorry

end total_population_increase_l180_180828


namespace students_in_all_classes_l180_180114

theorem students_in_all_classes (total_students : ℕ) (students_photography : ℕ) (students_music : ℕ) (students_theatre : ℕ) (students_dance : ℕ) (students_at_least_two : ℕ) (students_in_all : ℕ) :
  total_students = 30 →
  students_photography = 15 →
  students_music = 18 →
  students_theatre = 12 →
  students_dance = 10 →
  students_at_least_two = 18 →
  students_in_all = 4 :=
by
  intros
  sorry

end students_in_all_classes_l180_180114


namespace overall_gain_percentage_correct_l180_180419

structure Transaction :=
  (buy_prices : List ℕ)
  (sell_prices : List ℕ)

def overallGainPercentage (trans : Transaction) : ℚ :=
  let total_cost := (trans.buy_prices.foldl (· + ·) 0 : ℚ)
  let total_sell := (trans.sell_prices.foldl (· + ·) 0 : ℚ)
  (total_sell - total_cost) / total_cost * 100

theorem overall_gain_percentage_correct
  (trans : Transaction)
  (h_buy_prices : trans.buy_prices = [675, 850, 920])
  (h_sell_prices : trans.sell_prices = [1080, 1100, 1000]) :
  overallGainPercentage trans = 30.06 := by
  sorry

end overall_gain_percentage_correct_l180_180419


namespace find_S17_l180_180449

-- Definitions based on the conditions
variables (a : ℕ → ℝ) (S : ℕ → ℝ)
variables (a1 : ℝ) (d : ℝ)

-- Conditions from the problem restated in Lean
axiom arithmetic_sequence : ∀ n, a n = a1 + (n - 1) * d
axiom sum_of_n_terms : ∀ n, S n = n / 2 * (2 * a1 + (n - 1) * d)
axiom arithmetic_subseq : 2 * a 7 = a 5 + 3

-- Theorem to prove
theorem find_S17 : S 17 = 51 :=
by sorry

end find_S17_l180_180449


namespace area_of_inscribed_rectangle_l180_180468

open Real

theorem area_of_inscribed_rectangle (r l w : ℝ) (h_radius : r = 7) 
  (h_ratio : l / w = 3) (h_width : w = 2 * r) : l * w = 588 :=
by
  sorry

end area_of_inscribed_rectangle_l180_180468


namespace arnold_danny_age_l180_180983

theorem arnold_danny_age (x : ℕ) (h : (x + 1) * (x + 1) = x * x + 13) : x = 6 :=
by {
  sorry
}

end arnold_danny_age_l180_180983


namespace cookies_with_five_cups_l180_180098

theorem cookies_with_five_cups (cookies_per_four_cups : ℕ) (flour_for_four_cups : ℕ) (flour_for_five_cups : ℕ) (h : 24 / 4 = cookies_per_four_cups / 5) :
  cookies_per_four_cups = 30 :=
by
  sorry

end cookies_with_five_cups_l180_180098


namespace find_c_l180_180680

theorem find_c (a b c n : ℝ) (h : n = (2 * a * b * c) / (c - a)) : c = (n * a) / (n - 2 * a * b) :=
by
  sorry

end find_c_l180_180680


namespace cannot_be_20182017_l180_180401

theorem cannot_be_20182017 (a b : ℤ) (h : a * b * (a + b) = 20182017) : False :=
by
  sorry

end cannot_be_20182017_l180_180401


namespace arithmetic_sequences_diff_l180_180546

theorem arithmetic_sequences_diff
  (a : ℕ → ℤ)
  (b : ℕ → ℤ)
  (d_a d_b : ℤ)
  (ha : ∀ n, a n = 3 + n * d_a)
  (hb : ∀ n, b n = -3 + n * d_b)
  (h : a 19 - b 19 = 16) :
  a 10 - b 10 = 11 := by
    sorry

end arithmetic_sequences_diff_l180_180546


namespace number_of_valid_sequences_l180_180014

-- Define the sequence and conditions
def is_valid_sequence (a : Fin 9 → ℝ) : Prop :=
  a 0 = 1 ∧ a 8 = 1 ∧
  ∀ i : Fin 8, (a (i + 1) / a i) ∈ ({2, 1, -1/2} : Set ℝ)

-- The main problem statement
theorem number_of_valid_sequences : ∃ n, n = 491 ∧ ∀ a : Fin 9 → ℝ, is_valid_sequence a ↔ n = 491 := 
sorry

end number_of_valid_sequences_l180_180014


namespace common_difference_of_arithmetic_sequence_l180_180372

theorem common_difference_of_arithmetic_sequence (a_n : ℕ → ℤ) (S_n : ℕ → ℤ) (n d a_2 S_3 a_4 : ℤ) 
  (h1 : a_2 + S_3 = -4) (h2 : a_4 = 3)
  (h3 : ∀ n, S_n = n * (a_n + (a_n + (n - 1) * d)) / 2)
  : d = 2 := by
  sorry

end common_difference_of_arithmetic_sequence_l180_180372


namespace monotonic_increase_l180_180640

noncomputable def f (x : ℝ) : ℝ := Real.log (x - 2)

theorem monotonic_increase : ∀ x1 x2 : ℝ, 2 < x1 → x1 < x2 → f x1 < f x2 :=
by
  sorry

end monotonic_increase_l180_180640


namespace cupboard_slots_l180_180658

theorem cupboard_slots (shelves_from_top shelves_from_bottom slots_from_left slots_from_right : ℕ)
  (h_top : shelves_from_top = 1)
  (h_bottom : shelves_from_bottom = 3)
  (h_left : slots_from_left = 0)
  (h_right : slots_from_right = 6) :
  (shelves_from_top + 1 + shelves_from_bottom) * (slots_from_left + 1 + slots_from_right) = 35 := by
  sorry

end cupboard_slots_l180_180658


namespace students_neither_math_physics_l180_180943

theorem students_neither_math_physics (total_students math_students physics_students both_students : ℕ) 
  (h1 : total_students = 120)
  (h2 : math_students = 80)
  (h3 : physics_students = 50)
  (h4 : both_students = 15) : 
  total_students - (math_students - both_students + physics_students - both_students + both_students) = 5 :=
by
  -- Each of the hypotheses are used exactly as given in the conditions.
  -- We omit the proof as requested.
  sorry

end students_neither_math_physics_l180_180943


namespace problem1_coefficient_of_x_problem2_maximum_coefficient_term_l180_180338

-- Problem 1: Coefficient of x term
theorem problem1_coefficient_of_x (n : ℕ) 
  (A : ℕ := (3 + 1)^n) 
  (B : ℕ := 2^n) 
  (h1 : A + B = 272) 
  : true :=  -- Replacing true with actual condition
by sorry

-- Problem 2: Term with maximum coefficient
theorem problem2_maximum_coefficient_term (n : ℕ)
  (h : 1 + n + (n * (n - 1)) / 2 = 79) 
  : true :=  -- Replacing true with actual condition
by sorry

end problem1_coefficient_of_x_problem2_maximum_coefficient_term_l180_180338


namespace center_of_symmetry_l180_180225

-- Define the given conditions
def has_axis_symmetry_x (F : Set (ℝ × ℝ)) : Prop := 
  ∀ (x y : ℝ), (x, y) ∈ F → (-x, y) ∈ F

def has_axis_symmetry_y (F : Set (ℝ × ℝ)) : Prop :=
  ∀ (x y : ℝ), (x, y) ∈ F → (x, -y) ∈ F
  
-- Define the central proof goal
theorem center_of_symmetry (F : Set (ℝ × ℝ)) (H1: has_axis_symmetry_x F) (H2: has_axis_symmetry_y F) :
  ∀ (x y : ℝ), (x, y) ∈ F → (-x, -y) ∈ F :=
sorry

end center_of_symmetry_l180_180225


namespace inequality_solution_l180_180158

theorem inequality_solution (x : ℝ) : 
  (3 - (1 / (3 * x + 4)) < 5) ↔ (x < -4 / 3) := 
by
  sorry

end inequality_solution_l180_180158


namespace radius_of_circle_l180_180447

theorem radius_of_circle (r : ℝ) (C : ℝ) (A : ℝ) (h1 : 3 * C = 2 * A) 
  (h2 : C = 2 * Real.pi * r) (h3 : A = Real.pi * r^2) : 
  r = 3 :=
by 
  sorry

end radius_of_circle_l180_180447


namespace exponent_property_l180_180762

theorem exponent_property : 3000 * 3000^2500 = 3000^2501 := 
by sorry

end exponent_property_l180_180762


namespace problem_statement_l180_180071

theorem problem_statement (f : ℝ → ℝ) (a b c m : ℝ)
  (h_cond1 : ∀ x, f x = -x^2 + a * x + b)
  (h_range : ∀ y, y ∈ Set.range f ↔ y ≤ 0)
  (h_ineq_sol : ∀ x, ((-x^2 + a * x + b > c - 1) ↔ (m - 4 < x ∧ x < m + 1))) :
  (b = -(1/4) * (2 * m - 3)^2) ∧ (c = -(21 / 4)) := sorry

end problem_statement_l180_180071


namespace ticket_price_reduction_l180_180294

theorem ticket_price_reduction
    (original_price : ℝ := 50)
    (increase_in_tickets : ℝ := 1 / 3)
    (increase_in_revenue : ℝ := 1 / 4)
    (x : ℝ)
    (reduced_price : ℝ)
    (new_tickets : ℝ := x * (1 + increase_in_tickets))
    (original_revenue : ℝ := x * original_price)
    (new_revenue : ℝ := new_tickets * reduced_price) :
    new_revenue = (1 + increase_in_revenue) * original_revenue →
    reduced_price = original_price - (original_price / 2) :=
    sorry

end ticket_price_reduction_l180_180294


namespace maximum_ab_expression_l180_180945

open Function Real

theorem maximum_ab_expression {a b : ℝ} (h : 0 < a ∧ 0 < b ∧ 5 * a + 6 * b < 110) :
  ab * (110 - 5 * a - 6 * b) ≤ 1331000 / 810 :=
sorry

end maximum_ab_expression_l180_180945


namespace xyz_sum_neg1_l180_180915

theorem xyz_sum_neg1 (x y z : ℝ) (h : (x + 1)^2 + |y - 2| = -(2 * x - z)^2) : x + y + z = -1 :=
sorry

end xyz_sum_neg1_l180_180915


namespace sequence_bounded_l180_180562

theorem sequence_bounded (a : ℕ → ℝ) (h_pos : ∀ n, 0 < a n)
  (h_dep : ∀ k n m l, k + n = m + l → (a k + a n) / (1 + a k * a n) = (a m + a l) / (1 + a m * a l)) :
  ∃ m M : ℝ, ∀ n, m ≤ a n ∧ a n ≤ M :=
sorry

end sequence_bounded_l180_180562


namespace slower_train_speed_l180_180845

-- Defining the conditions

def length_of_each_train := 80 -- in meters
def faster_train_speed := 52 -- in km/hr
def time_to_pass := 36 -- in seconds

-- Main statement: 
theorem slower_train_speed (v : ℝ) : 
    let relative_speed := (faster_train_speed - v) * (1000 / 3600) -- converting relative speed from km/hr to m/s
    let total_distance := 2 * length_of_each_train
    let speed_equals_distance_over_time := total_distance / time_to_pass 
    (relative_speed = speed_equals_distance_over_time) -> v = 36 :=
by
  intros
  sorry

end slower_train_speed_l180_180845


namespace min_sum_of_factors_l180_180181

theorem min_sum_of_factors (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_prod : a * b * c = 2310) : 
  a + b + c ≥ 42 :=
sorry

end min_sum_of_factors_l180_180181


namespace scientific_notation_example_l180_180291

theorem scientific_notation_example : 10500 = 1.05 * 10^4 :=
by
  sorry

end scientific_notation_example_l180_180291


namespace greatest_possible_price_per_notebook_l180_180962

theorem greatest_possible_price_per_notebook (budget entrance_fee : ℝ) (notebooks : ℕ) (tax_rate : ℝ) (price_per_notebook : ℝ) :
  budget = 160 ∧ entrance_fee = 5 ∧ notebooks = 18 ∧ tax_rate = 0.05 ∧ price_per_notebook * notebooks * (1 + tax_rate) ≤ (budget - entrance_fee) →
  price_per_notebook = 8 :=
by
  sorry

end greatest_possible_price_per_notebook_l180_180962


namespace emery_reading_days_l180_180902

theorem emery_reading_days (S E : ℕ) (h1 : E = S / 5) (h2 : (E + S) / 2 = 60) : E = 20 := by
  sorry

end emery_reading_days_l180_180902


namespace minimum_participants_l180_180809

theorem minimum_participants (x y z n : ℕ) 
  (hx : x + 1 + 2 * x = n)
  (hy : y + 1 + 3 * y = n)
  (hz : z + 1 + 4 * z = n) :
  n = 61 :=
by sorry

end minimum_participants_l180_180809


namespace largest_prime_17p_625_l180_180911

theorem largest_prime_17p_625 (p : ℕ) (h_prime : Nat.Prime p) (h_sqrt : ∃ q, 17 * p + 625 = q^2) : p = 67 :=
by
  sorry

end largest_prime_17p_625_l180_180911


namespace a_range_l180_180195

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2
def g (x a : ℝ) : ℝ := 2 * x + a

theorem a_range :
  (∃ (x1 x2 : ℝ), x1 ∈ Set.Icc (1 / 2) 2 ∧ x2 ∈ Set.Icc (1 / 2) 2 ∧ f x1 = g x2 a) ↔ -5 ≤ a ∧ a ≤ 0 := 
by 
  sorry

end a_range_l180_180195


namespace expression_value_l180_180740

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_add_prop (a b : ℝ) : f (a + b) = f a * f b
axiom f_one_val : f 1 = 2

theorem expression_value : 
  (f 1 ^ 2 + f 2) / f 1 + 
  (f 2 ^ 2 + f 4) / f 3 +
  (f 3 ^ 2 + f 6) / f 5 + 
  (f 4 ^ 2 + f 8) / f 7 
  = 16 := 
sorry

end expression_value_l180_180740


namespace problem_I_problem_II_problem_III_l180_180336

noncomputable def f (a x : ℝ) := a * x * Real.exp x
noncomputable def f' (a x : ℝ) := a * (1 + x) * Real.exp x

theorem problem_I (a : ℝ) (h : a ≠ 0) :
  (if a > 0 then ∀ x, (f' a x > 0 ↔ x > -1) ∧ (f' a x < 0 ↔ x < -1)
  else ∀ x, (f' a x > 0 ↔ x < -1) ∧ (f' a x < 0 ↔ x > -1)) :=
sorry

theorem problem_II (h : ∃ a : ℝ, a = 1) :
  ∃ (x : ℝ) (y : ℝ), x = -1 ∧ f 1 (-1) = -1 / Real.exp 1 ∧ ¬ ∃ y, ∀ x, y = f 1 x ∧ (f' 1 x) < 0 :=
sorry

theorem problem_III (h : ∃ m : ℝ, f 1 m = e * m * Real.exp m ∧ f' 1 m = e * (1 + m) * Real.exp m) :
  ∃ a : ℝ, a = 1 / 2 :=
sorry

end problem_I_problem_II_problem_III_l180_180336


namespace max_sin_a_l180_180143

theorem max_sin_a (a b : ℝ)
  (h1 : b = Real.pi / 2 - a)
  (h2 : Real.cos (a + b) = Real.cos a + Real.cos b) :
  Real.sin a ≤ Real.sqrt 2 / 2 :=
sorry

end max_sin_a_l180_180143


namespace inscribed_circle_ratio_l180_180876

theorem inscribed_circle_ratio (a b c u v : ℕ) (h_triangle : a = 10 ∧ b = 24 ∧ c = 26) 
    (h_tangent_segments : u < v) (h_side_sum : u + v = a) : u / v = 2 / 3 :=
by
    sorry

end inscribed_circle_ratio_l180_180876


namespace smallest_n_gt_15_l180_180360

theorem smallest_n_gt_15 (n : ℕ) : n ≡ 4 [MOD 6] → n ≡ 3 [MOD 7] → n > 15 → n = 52 :=
by
  sorry

end smallest_n_gt_15_l180_180360


namespace eval_power_l180_180008

theorem eval_power (h : 81 = 3^4) : 81^(5/4) = 243 := by
  sorry

end eval_power_l180_180008


namespace minimum_value_m_l180_180871

theorem minimum_value_m (x0 : ℝ) : (∃ x0 : ℝ, |x0 + 1| + |x0 - 1| ≤ m) → m = 2 :=
by
  sorry

end minimum_value_m_l180_180871


namespace proof_supplies_proof_transportation_cost_proof_min_cost_condition_l180_180713

open Real

noncomputable def supplies_needed (a b : ℕ) := a = 200 ∧ b = 300

noncomputable def transportation_cost (x : ℝ) := 60 ≤ x ∧ x ≤ 260 ∧ ∀ w : ℝ, w = 10 * x + 10200

noncomputable def min_cost_condition (m x : ℝ) := 
  (0 < m ∧ m ≤ 8) ∧ (∀ w : ℝ, (10 - m) * x + 10200 ≥ 10320)

theorem proof_supplies : ∃ a b : ℕ, supplies_needed a b := 
by
  use 200, 300
  sorry

theorem proof_transportation_cost : ∃ x : ℝ, transportation_cost x := 
by
  use 60
  sorry

theorem proof_min_cost_condition : ∃ m x : ℝ, min_cost_condition m x := 
by
  use 8, 60
  sorry

end proof_supplies_proof_transportation_cost_proof_min_cost_condition_l180_180713


namespace tire_circumference_l180_180292

theorem tire_circumference (rpm : ℕ) (speed_kmh : ℕ) (C : ℝ) (h_rpm : rpm = 400) (h_speed_kmh : speed_kmh = 48) :
  (C = 2) :=
by
  -- sorry statement to assume the solution for now
  sorry

end tire_circumference_l180_180292


namespace circle_radius_increase_l180_180427

variable (r n : ℝ) -- declare variables r and n as real numbers

theorem circle_radius_increase (h : 2 * π * (r + n) = 2 * (2 * π * r)) : r = n :=
by
  sorry

end circle_radius_increase_l180_180427


namespace scientific_notation_l180_180899

theorem scientific_notation (n : ℕ) (h : n = 27000000) : 
  ∃ (m : ℝ) (e : ℤ), n = m * (10 : ℝ) ^ e ∧ m = 2.7 ∧ e = 7 :=
by 
  use 2.7 
  use 7
  sorry

end scientific_notation_l180_180899


namespace perpendicular_lines_solve_a_l180_180646

theorem perpendicular_lines_solve_a (a : ℝ) :
  (3 * a + 2) * (5 * a - 2) + (1 - 4 * a) * (a + 4) = 0 → a = 0 ∨ a = 12 / 11 :=
by 
  sorry

end perpendicular_lines_solve_a_l180_180646


namespace positive_real_inequality_l180_180862

theorem positive_real_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / Real.sqrt (a^2 + 8 * b * c)) + (b / Real.sqrt (b^2 + 8 * c * a)) + (c / Real.sqrt (c^2 + 8 * a * b)) ≥ 1 :=
by
  sorry

end positive_real_inequality_l180_180862


namespace helen_needed_gas_l180_180490

-- Definitions based on conditions
def cuts_per_month_routine_1 : ℕ := 2 -- Cuts per month for March, April, September, October
def cuts_per_month_routine_2 : ℕ := 4 -- Cuts per month for May, June, July, August
def months_routine_1 : ℕ := 4 -- Number of months with routine 1
def months_routine_2 : ℕ := 4 -- Number of months with routine 2
def gas_per_fill : ℕ := 2 -- Gallons of gas used every 4th cut
def cuts_per_fill : ℕ := 4 -- Number of cuts per fill

-- Total number of cuts in routine 1 months
def total_cuts_routine_1 : ℕ := cuts_per_month_routine_1 * months_routine_1

-- Total number of cuts in routine 2 months
def total_cuts_routine_2 : ℕ := cuts_per_month_routine_2 * months_routine_2

-- Total cuts from March to October
def total_cuts : ℕ := total_cuts_routine_1 + total_cuts_routine_2

-- Total fills needed from March to October
def total_fills : ℕ := total_cuts / cuts_per_fill

-- Total gallons of gas needed
def total_gal_of_gas : ℕ := total_fills * gas_per_fill

-- The statement to prove
theorem helen_needed_gas : total_gal_of_gas = 12 :=
by
  -- This would be replaced by our solution steps.
  sorry

end helen_needed_gas_l180_180490


namespace apple_slices_count_l180_180007

theorem apple_slices_count :
  let boxes := 7
  let apples_per_box := 7
  let slices_per_apple := 8
  let total_apples := boxes * apples_per_box
  let total_slices := total_apples * slices_per_apple
  total_slices = 392 :=
by
  sorry

end apple_slices_count_l180_180007


namespace stored_bales_correct_l180_180440

theorem stored_bales_correct :
  let initial_bales := 28
  let new_bales := 54
  let stored_bales := new_bales - initial_bales
  stored_bales = 26 :=
by
  let initial_bales := 28
  let new_bales := 54
  let stored_bales := new_bales - initial_bales
  show stored_bales = 26
  sorry

end stored_bales_correct_l180_180440


namespace min_value_fraction_l180_180279

theorem min_value_fraction (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) : 
  ∃ c, (c = 9) ∧ (∀ x y, (x > 0 ∧ y > 0 ∧ x + y = 1) → (1/x + 4/y ≥ c)) :=
by
  sorry

end min_value_fraction_l180_180279


namespace value_of_a_l180_180504

theorem value_of_a (a : ℝ) (h : (1 : ℝ)^2 - 2 * (1 : ℝ) + a = 0) : a = 1 := 
by 
  sorry

end value_of_a_l180_180504


namespace vendor_profit_l180_180594

theorem vendor_profit {s₁ s₂ c₁ c₂ : ℝ} (h₁ : s₁ = 80) (h₂ : s₂ = 80) (profit₁ : s₁ = c₁ * 1.60) (loss₂ : s₂ = c₂ * 0.80) 
: (s₁ + s₂) - (c₁ + c₂) = 10 := by 
  sorry

end vendor_profit_l180_180594


namespace point_in_second_quadrant_l180_180635

variable (m : ℝ)

-- Defining the conditions
def x_negative (m : ℝ) := 3 - m < 0
def y_positive (m : ℝ) := m - 1 > 0

theorem point_in_second_quadrant (h1 : x_negative m) (h2 : y_positive m) : m > 3 :=
by
  sorry

end point_in_second_quadrant_l180_180635


namespace new_student_weight_l180_180570

theorem new_student_weight :
  let avg_weight_29 := 28
  let num_students_29 := 29
  let avg_weight_30 := 27.4
  let num_students_30 := 30
  let total_weight_29 := avg_weight_29 * num_students_29
  let total_weight_30 := avg_weight_30 * num_students_30
  let new_student_weight := total_weight_30 - total_weight_29
  new_student_weight = 10 :=
by
  sorry

end new_student_weight_l180_180570


namespace cat_mouse_position_258_l180_180613

-- Define the cycle positions for the cat
def cat_position (n : ℕ) : String :=
  match n % 4 with
  | 0 => "top left"
  | 1 => "top right"
  | 2 => "bottom right"
  | _ => "bottom left"

-- Define the cycle positions for the mouse
def mouse_position (n : ℕ) : String :=
  match n % 8 with
  | 0 => "top middle"
  | 1 => "top right"
  | 2 => "right middle"
  | 3 => "bottom right"
  | 4 => "bottom middle"
  | 5 => "bottom left"
  | 6 => "left middle"
  | _ => "top left"

theorem cat_mouse_position_258 : 
  cat_position 258 = "top right" ∧ mouse_position 258 = "top right" := by
  sorry

end cat_mouse_position_258_l180_180613


namespace int_product_negative_max_negatives_l180_180550

theorem int_product_negative_max_negatives (n : ℤ) (hn : n ≤ 9) (hp : n % 2 = 1) :
  ∃ m : ℤ, n + m = m ∧ m ≥ 0 :=
by
  use 9
  sorry

end int_product_negative_max_negatives_l180_180550


namespace liked_both_desserts_l180_180963

noncomputable def total_students : ℕ := 50
noncomputable def apple_pie_lovers : ℕ := 22
noncomputable def chocolate_cake_lovers : ℕ := 20
noncomputable def neither_dessert_lovers : ℕ := 17
noncomputable def both_desserts_lovers : ℕ := 9

theorem liked_both_desserts :
  (total_students - neither_dessert_lovers) + both_desserts_lovers = apple_pie_lovers + chocolate_cake_lovers - both_desserts_lovers :=
by
  sorry

end liked_both_desserts_l180_180963


namespace combined_weight_cats_l180_180904

-- Define the weights of the cats
def weight_cat1 := 2
def weight_cat2 := 7
def weight_cat3 := 4

-- Prove the combined weight of the three cats is 13 pounds
theorem combined_weight_cats :
  weight_cat1 + weight_cat2 + weight_cat3 = 13 := by
  sorry

end combined_weight_cats_l180_180904


namespace largest_value_of_n_l180_180725

noncomputable def largest_n_under_200000 : ℕ :=
  if h : 199999 < 200000 ∧ (8 * (199999 - 3)^5 - 2 * 199999^2 + 18 * 199999 - 36) % 7 = 0 then 199999 else 0

theorem largest_value_of_n (n : ℕ) :
  n < 200000 → (8 * (n - 3)^5 - 2 * n^2 + 18 * n - 36) % 7 = 0 → n = 199999 :=
by sorry

end largest_value_of_n_l180_180725


namespace quadratic_function_range_l180_180346

theorem quadratic_function_range (f : ℝ → ℝ) (a : ℝ)
  (h_quad : ∃ p q r : ℝ, ∀ x, f x = p * x^2 + q * x + r)
  (h_sym : ∀ x, f (2 + x) = f (2 - x))
  (h_cond : f a ≤ f 0 ∧ f 0 < f 1) :
  a ≤ 0 ∨ a ≥ 4 :=
sorry

end quadratic_function_range_l180_180346


namespace jessica_flowers_problem_l180_180145

theorem jessica_flowers_problem
(initial_roses initial_daisies : ℕ)
(thrown_roses thrown_daisies : ℕ)
(current_roses current_daisies : ℕ)
(cut_roses cut_daisies : ℕ)
(h_initial_roses : initial_roses = 21)
(h_initial_daisies : initial_daisies = 17)
(h_thrown_roses : thrown_roses = 34)
(h_thrown_daisies : thrown_daisies = 25)
(h_current_roses : current_roses = 15)
(h_current_daisies : current_daisies = 10)
(h_cut_roses : cut_roses = (thrown_roses - initial_roses) + current_roses)
(h_cut_daisies : cut_daisies = (thrown_daisies - initial_daisies) + current_daisies) :
thrown_roses + thrown_daisies - (cut_roses + cut_daisies) = 13 := by
  sorry

end jessica_flowers_problem_l180_180145


namespace necessary_sufficient_condition_l180_180107

theorem necessary_sufficient_condition 
  (a b : ℝ) : 
  a * |a + b| < |a| * (a + b) ↔ (a < 0 ∧ b > -a) :=
sorry

end necessary_sufficient_condition_l180_180107


namespace solve_missing_figure_l180_180019

theorem solve_missing_figure (x : ℝ) (h : 0.25/100 * x = 0.04) : x = 16 :=
by
  sorry

end solve_missing_figure_l180_180019


namespace pencils_more_than_200_on_saturday_l180_180467

theorem pencils_more_than_200_on_saturday 
    (p : ℕ → ℕ) 
    (h_start : p 1 = 3)
    (h_next_day : ∀ n, p (n + 1) = (p n + 2) * 2) 
    : p 6 > 200 :=
by
  -- Proof steps can be filled in here.
  sorry

end pencils_more_than_200_on_saturday_l180_180467


namespace balance_pitcher_with_saucers_l180_180794

-- Define the weights of the cup (C), pitcher (P), and saucer (S)
variables (C P S : ℝ)

-- Conditions provided in the problem
axiom cond1 : 2 * C + 2 * P = 14 * S
axiom cond2 : P = C + S

-- The statement to prove
theorem balance_pitcher_with_saucers : P = 4 * S :=
by
  sorry

end balance_pitcher_with_saucers_l180_180794


namespace inequality_must_be_true_l180_180481

theorem inequality_must_be_true (a b : ℝ) (h : a > b ∧ b > 0) :
  a + 1 / b > b + 1 / a :=
sorry

end inequality_must_be_true_l180_180481


namespace chris_money_l180_180472

-- Define conditions
def grandmother_gift : Nat := 25
def aunt_uncle_gift : Nat := 20
def parents_gift : Nat := 75
def total_after_birthday : Nat := 279

-- Define the proof problem to show Chris had $159 before his birthday
theorem chris_money (x : Nat) (h : x + grandmother_gift + aunt_uncle_gift + parents_gift = total_after_birthday) :
  x = 159 :=
by
  -- Leave the proof blank
  sorry

end chris_money_l180_180472


namespace ascending_order_l180_180671

theorem ascending_order (a b : ℝ) (ha : a < 0) (hb1 : -1 < b) (hb2 : b < 0) : a < a * b^2 ∧ a * b^2 < a * b :=
by
  sorry

end ascending_order_l180_180671


namespace problem1_asymptotes_problem2_equation_l180_180643

-- Problem 1: Asymptotes of a hyperbola
theorem problem1_asymptotes (a : ℝ) (x y : ℝ) (hx : (y + a) ^ 2 - (x - a) ^ 2 = 2 * a)
  (hpt : 3 = x ∧ 1 = y) : 
  (y = x - 2 * a) ∨ (y = - x) := 
by 
  sorry

-- Problem 2: Equation of a hyperbola
theorem problem2_equation (a b c : ℝ) (x y : ℝ) 
  (hasymptote : y = x + 1 ∨ y = - (x + 1))  (hfocal : 2 * c = 4)
  (hc_squared : c ^ 2 = a ^ 2 + b ^ 2) (ha_eq_b : a = b): 
  y^2 - (x + 1)^2 = 2 := 
by 
  sorry

end problem1_asymptotes_problem2_equation_l180_180643


namespace walkway_time_l180_180001

theorem walkway_time {v_p v_w : ℝ} 
  (cond1 : 60 = (v_p + v_w) * 30) 
  (cond2 : 60 = (v_p - v_w) * 120) 
  : 60 / v_p = 48 := 
by
  sorry

end walkway_time_l180_180001


namespace visitors_inversely_proportional_l180_180560

theorem visitors_inversely_proportional (k : ℝ) (v₁ v₂ t₁ t₂ : ℝ) (h1 : v₁ * t₁ = k) (h2 : t₁ = 20) (h3 : v₁ = 150) (h4 : t₂ = 30) : v₂ = 100 :=
by
  -- This is a placeholder line; the actual proof would go here.
  sorry

end visitors_inversely_proportional_l180_180560


namespace parameterization_properties_l180_180923

theorem parameterization_properties (a b c d : ℚ)
  (h1 : a * (-1) + b = -3)
  (h2 : c * (-1) + d = 5)
  (h3 : a * 2 + b = 4)
  (h4 : c * 2 + d = 15) :
  a^2 + b^2 + c^2 + d^2 = 790 / 9 :=
sorry

end parameterization_properties_l180_180923


namespace quadratic_distinct_real_roots_iff_l180_180538

theorem quadratic_distinct_real_roots_iff (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ (∀ (z : ℝ), z^2 - 2 * (m - 2) * z + m^2 = (z - x) * (z - y))) ↔ m < 1 :=
by
  sorry

end quadratic_distinct_real_roots_iff_l180_180538


namespace find_second_sum_l180_180010

def total_sum : ℝ := 2691
def interest_rate_first : ℝ := 0.03
def interest_rate_second : ℝ := 0.05
def time_first : ℝ := 8
def time_second : ℝ := 3

theorem find_second_sum (x second_sum : ℝ) 
  (H : x + second_sum = total_sum)
  (H_interest : x * interest_rate_first * time_first = second_sum * interest_rate_second * time_second) :
  second_sum = 1656 :=
sorry

end find_second_sum_l180_180010


namespace functional_relationship_inversely_proportional_l180_180498

-- Definitions based on conditions
def table_data : List (ℝ × ℝ) := [(100, 1.00), (200, 0.50), (400, 0.25), (500, 0.20)]

-- The main conjecture to be proved
theorem functional_relationship_inversely_proportional (y x : ℝ) (h : (x, y) ∈ table_data) : y = 100 / x :=
sorry

end functional_relationship_inversely_proportional_l180_180498


namespace coeff_of_nxy_n_l180_180601

theorem coeff_of_nxy_n {n : ℕ} (degree_eq : 1 + n = 10) : n = 9 :=
by
  sorry

end coeff_of_nxy_n_l180_180601


namespace pipe_p_fills_cistern_in_12_minutes_l180_180074

theorem pipe_p_fills_cistern_in_12_minutes :
  (∃ (t : ℝ), 
    ∀ (q_fill_rate p_fill_rate : ℝ), 
      q_fill_rate = 1 / 15 ∧ 
      t > 0 ∧ 
      (4 * (1 / t + q_fill_rate) + 6 * q_fill_rate = 1) → t = 12) :=
sorry

end pipe_p_fills_cistern_in_12_minutes_l180_180074


namespace benjamin_billboards_l180_180228

theorem benjamin_billboards (B : ℕ) (h1 : 20 + 23 + B = 60) : B = 17 :=
by
  sorry

end benjamin_billboards_l180_180228


namespace inequality_solution_l180_180964

theorem inequality_solution (x : ℝ) : 
  (x^2 - 9) / (x^2 - 4) > 0 ↔ (x < -3 ∨ x > 3) := 
by 
  sorry

end inequality_solution_l180_180964


namespace total_worksheets_l180_180858

theorem total_worksheets (worksheets_graded : ℕ) (problems_per_worksheet : ℕ) (problems_remaining : ℕ)
  (h1 : worksheets_graded = 7)
  (h2 : problems_per_worksheet = 2)
  (h3 : problems_remaining = 14): 
  worksheets_graded + (problems_remaining / problems_per_worksheet) = 14 := 
by 
  sorry

end total_worksheets_l180_180858


namespace cone_surface_area_volume_ineq_l180_180056

theorem cone_surface_area_volume_ineq
  (A V r a m : ℝ)
  (hA : A = π * r * (r + a))
  (hV : V = (1/3) * π * r^2 * m)
  (hPythagoras : a^2 = r^2 + m^2) :
  A^3 ≥ 72 * π * V^2 := 
by
  sorry

end cone_surface_area_volume_ineq_l180_180056


namespace n_squared_divisible_by_36_l180_180922

theorem n_squared_divisible_by_36 (n : ℕ) (h1 : 0 < n) (h2 : 6 ∣ n) : 36 ∣ n^2 := 
sorry

end n_squared_divisible_by_36_l180_180922


namespace product_remainder_l180_180491

theorem product_remainder (a b c d : ℕ) (ha : a % 7 = 2) (hb : b % 7 = 3) (hc : c % 7 = 4) (hd : d % 7 = 5) :
  (a * b * c * d) % 7 = 1 :=
by
  sorry

end product_remainder_l180_180491


namespace algebraic_expression_value_l180_180442

noncomputable def a : ℝ := 1 + Real.sqrt 2
noncomputable def b : ℝ := 1 - Real.sqrt 2

theorem algebraic_expression_value :
  let a := 1 + Real.sqrt 2
  let b := 1 - Real.sqrt 2
  a^2 - a * b + b^2 = 7 := by
  sorry

end algebraic_expression_value_l180_180442


namespace range_of_m_l180_180924

def f (x : ℝ) : ℝ := x^3 + x

theorem range_of_m (m : ℝ) (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < Real.pi / 2) :
  (f (m * Real.sin θ) + f (1 - m) > 0) ↔ (m ≤ 1) :=
sorry

end range_of_m_l180_180924


namespace multiply_add_square_l180_180769

theorem multiply_add_square : 15 * 28 + 42 * 15 + 15^2 = 1275 :=
by
  sorry

end multiply_add_square_l180_180769


namespace binary_to_decimal_l180_180968

theorem binary_to_decimal : (1 * 2^2 + 0 * 2^1 + 1 * 2^0 = 5) :=
by
  sorry

end binary_to_decimal_l180_180968


namespace negation_of_proposition_l180_180084

variable (x : ℝ)

theorem negation_of_proposition (h : ∃ x : ℝ, x^2 + x - 1 < 0) : ¬ (∀ x : ℝ, x^2 + x - 1 ≥ 0) :=
sorry

end negation_of_proposition_l180_180084


namespace grid_segments_divisible_by_4_l180_180956

-- Definition: square grid where each cell has a side length of 1
structure SquareGrid (n : ℕ) :=
  (segments : ℕ)

-- Condition: Function to calculate the total length of segments in the grid
def total_length {n : ℕ} (Q : SquareGrid n) : ℕ := Q.segments

-- Lean 4 statement: Prove that for any grid, the total length is divisible by 4
theorem grid_segments_divisible_by_4 {n : ℕ} (Q : SquareGrid n) :
  total_length Q % 4 = 0 :=
sorry

end grid_segments_divisible_by_4_l180_180956


namespace largest_angle_measures_203_l180_180770

-- Define the angles of the hexagon
def angle1 (x : ℚ) : ℚ := x + 2
def angle2 (x : ℚ) : ℚ := 2 * x + 1
def angle3 (x : ℚ) : ℚ := 3 * x
def angle4 (x : ℚ) : ℚ := 4 * x - 1
def angle5 (x : ℚ) : ℚ := 5 * x + 2
def angle6 (x : ℚ) : ℚ := 6 * x - 2

-- Define the sum of interior angles for a hexagon
def hexagon_angle_sum : ℚ := 720

-- Prove that the largest angle is equal to 203 degrees given the conditions
theorem largest_angle_measures_203 (x : ℚ) (h : angle1 x + angle2 x + angle3 x + angle4 x + angle5 x + angle6 x = hexagon_angle_sum) :
  (6 * x - 2) = 203 := by
  sorry

end largest_angle_measures_203_l180_180770


namespace systematic_sampling_removal_count_l180_180974

-- Define the conditions
def total_population : Nat := 1252
def sample_size : Nat := 50

-- Define the remainder after division
def remainder := total_population % sample_size

-- Proof statement
theorem systematic_sampling_removal_count :
  remainder = 2 := by
    sorry

end systematic_sampling_removal_count_l180_180974


namespace base4_to_base10_conversion_l180_180232

theorem base4_to_base10_conversion : (2 * 4^4 + 1 * 4^3 + 0 * 4^2 + 1 * 4^1 + 2 * 4^0 = 582) :=
by {
  -- The proof is omitted
  sorry
}

end base4_to_base10_conversion_l180_180232


namespace repunit_polynomial_characterization_l180_180252

noncomputable def is_repunit (n : ℕ) : Prop :=
  ∃ k : ℕ, n = (10^k - 1) / 9

def polynomial_condition (f : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, is_repunit n → is_repunit (f n)

theorem repunit_polynomial_characterization :
  ∀ (f : ℕ → ℕ), polynomial_condition f ↔
  ∃ m r : ℕ, m ≥ 0 ∧ r ≥ 1 - m ∧ ∀ n : ℕ, f n = (10^r * (9 * n + 1)^m - 1) / 9 :=
by
  sorry

end repunit_polynomial_characterization_l180_180252


namespace suitable_value_for_x_evaluates_to_neg1_l180_180985

noncomputable def given_expression (x : ℝ) : ℝ :=
  (x^3 + 2 * x^2) / (x^2 - 4 * x + 4) / (4 * x + 8) - 1 / (x - 2)

theorem suitable_value_for_x_evaluates_to_neg1 : 
  given_expression (-6) = -1 :=
by
  sorry

end suitable_value_for_x_evaluates_to_neg1_l180_180985


namespace cannot_contain_point_1997_0_l180_180425

variable {m b : ℝ}

theorem cannot_contain_point_1997_0 (h : m * b > 0) : ¬ (0 = 1997 * m + b) := sorry

end cannot_contain_point_1997_0_l180_180425


namespace strawberries_weight_l180_180405

theorem strawberries_weight (marco_weight dad_increase : ℕ) (h_marco: marco_weight = 30) (h_diff: marco_weight = dad_increase + 13) : marco_weight + (marco_weight - 13) = 47 :=
by
  sorry

end strawberries_weight_l180_180405


namespace relationship_bx_l180_180542

variable {a b t x : ℝ}

-- Given conditions
variable (h1 : b > a)
variable (h2 : a > 1)
variable (h3 : t > 0)
variable (h4 : a ^ x = a + t)

theorem relationship_bx (h1 : b > a) (h2 : a > 1) (h3 : t > 0) (h4 : a ^ x = a + t) : b ^ x > b + t :=
by
  sorry

end relationship_bx_l180_180542


namespace compare_expression_solve_inequality_l180_180887

-- Part (1) Problem Statement in Lean 4
theorem compare_expression (x : ℝ) (h : x ≥ -1) : 
  x^3 + 1 ≥ x^2 + x ∧ (x^3 + 1 = x^2 + x ↔ x = 1 ∨ x = -1) :=
by sorry

-- Part (2) Problem Statement in Lean 4
theorem solve_inequality (x a : ℝ) (ha : a < 0) : 
  (x^2 - a * x - 6 * a^2 > 0) ↔ (x < 3 * a ∨ x > -2 * a) :=
by sorry

end compare_expression_solve_inequality_l180_180887


namespace eva_fruit_diet_l180_180349

noncomputable def dietary_requirements : Prop :=
  ∃ (days_in_week : ℕ) (days_in_month : ℕ) (apples : ℕ) (bananas : ℕ) (pears : ℕ) (oranges : ℕ),
    days_in_week = 7 ∧
    days_in_month = 30 ∧
    apples = 2 * days_in_week ∧
    bananas = days_in_week / 2 ∧
    pears = 4 ∧
    oranges = days_in_month / 3 ∧
    apples = 14 ∧
    bananas = 4 ∧
    pears = 4 ∧
    oranges = 10

theorem eva_fruit_diet : dietary_requirements :=
sorry

end eva_fruit_diet_l180_180349


namespace integer_roots_of_polynomial_l180_180146

theorem integer_roots_of_polynomial (a₂ a₁ : ℤ) :
  {x : ℤ | (x^3 + a₂ * x^2 + a₁ * x - 18 = 0)} ⊆ {-18, -9, -6, -3, -2, -1, 1, 2, 3, 6, 9, 18} :=
by sorry

end integer_roots_of_polynomial_l180_180146


namespace star_eq_zero_iff_x_eq_5_l180_180044

/-- Define the operation * on real numbers -/
def star (a b : ℝ) : ℝ := a^2 - 2*a*b + b^2

/-- Proposition stating that x = 5 is the solution to (x - 4) * 1 = 0 -/
theorem star_eq_zero_iff_x_eq_5 (x : ℝ) : (star (x-4) 1 = 0) ↔ x = 5 :=
by
  sorry

end star_eq_zero_iff_x_eq_5_l180_180044


namespace contrapositive_l180_180998

theorem contrapositive (a b : ℕ) : (a = 0 → ab = 0) → (ab ≠ 0 → a ≠ 0) :=
by
  sorry

end contrapositive_l180_180998


namespace min_cost_speed_l180_180520

noncomputable def fuel_cost (v : ℝ) : ℝ := (1/200) * v^3

theorem min_cost_speed 
  (v : ℝ) 
  (u : ℝ) 
  (other_costs : ℝ) 
  (h1 : u = (1/200) * v^3) 
  (h2 : u = 40) 
  (h3 : v = 20) 
  (h4 : other_costs = 270) 
  (b : ℝ) 
  : ∃ v_min, v_min = 30 ∧ 
    ∀ (v : ℝ), (0 < v ∧ v ≤ b) → 
    ((fuel_cost v / v + other_costs / v) ≥ (fuel_cost v_min / v_min + other_costs / v_min)) := 
sorry

end min_cost_speed_l180_180520


namespace find_rs_l180_180239

noncomputable def r : ℝ := sorry
noncomputable def s : ℝ := sorry
def cond1 := r > 0 ∧ s > 0
def cond2 := r^2 + s^2 = 1
def cond3 := r^4 + s^4 = (3 : ℝ) / 4

theorem find_rs (h1 : cond1) (h2 : cond2) (h3 : cond3) : r * s = Real.sqrt 2 / 4 :=
by sorry

end find_rs_l180_180239


namespace quadratic_distinct_real_roots_l180_180421

open Real

theorem quadratic_distinct_real_roots (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ k * x ^ 2 - 2 * x - 1 = 0 ∧ k * y ^ 2 - 2 * y - 1 = 0) ↔ k > -1 ∧ k ≠ 0 :=
by
  sorry

end quadratic_distinct_real_roots_l180_180421


namespace calculate_triangle_area_l180_180383

-- Define the side lengths of the triangle.
def side1 : ℕ := 13
def side2 : ℕ := 13
def side3 : ℕ := 24

-- Define the area calculation.
noncomputable def triangle_area : ℕ := 60

-- Statement of the theorem we wish to prove.
theorem calculate_triangle_area :
  ∃ (a b c : ℕ) (area : ℕ), a = side1 ∧ b = side2 ∧ c = side3 ∧ area = triangle_area :=
sorry

end calculate_triangle_area_l180_180383


namespace find_notebook_price_l180_180522

noncomputable def notebook_and_pencil_prices : Prop :=
  ∃ (x y : ℝ),
    5 * x + 4 * y = 16.5 ∧
    2 * x + 2 * y = 7 ∧
    x = 2.5

theorem find_notebook_price : notebook_and_pencil_prices :=
  sorry

end find_notebook_price_l180_180522


namespace max_rectangle_area_l180_180076

-- Define the perimeter and side lengths of the rectangle
def perimeter := 60
def L (x : ℝ) := x
def W (x : ℝ) := 30 - x

-- Define the area of the rectangle
def area (x : ℝ) := L x * W x

-- State the theorem to prove the maximum area is 225 square feet
theorem max_rectangle_area : ∃ x : ℝ, 0 ≤ x ∧ x ≤ 30 ∧ area x = 225 :=
by
  sorry

end max_rectangle_area_l180_180076


namespace find_first_episode_l180_180532

variable (x : ℕ)
variable (w y z : ℕ)
variable (total_minutes: ℕ)
variable (h1 : w = 62)
variable (h2 : y = 65)
variable (h3 : z = 55)
variable (h4 : total_minutes = 240)

theorem find_first_episode :
  x + w + y + z = total_minutes → x = 58 := 
by
  intro h
  rw [h1, h2, h3, h4] at h
  linarith

end find_first_episode_l180_180532


namespace pelicans_among_non_egrets_is_47_percent_l180_180787

-- Definitions for the percentage of each type of bird.
def pelican_percentage : ℝ := 0.4
def cormorant_percentage : ℝ := 0.2
def egret_percentage : ℝ := 0.15
def osprey_percentage : ℝ := 0.25

-- Calculate the percentage of pelicans among the non-egret birds.
theorem pelicans_among_non_egrets_is_47_percent :
  (pelican_percentage / (1 - egret_percentage)) * 100 = 47 :=
by
  -- Detailed proof goes here
  sorry

end pelicans_among_non_egrets_is_47_percent_l180_180787


namespace sum_in_base5_correct_l180_180110

-- Defining the integers
def num1 : ℕ := 210
def num2 : ℕ := 72

-- Summing the integers
def sum : ℕ := num1 + num2

-- Converting the resulting sum to base 5
def to_base5 (n : ℕ) : String :=
  let rec aux (n : ℕ) (acc : List Char) : List Char :=
    if n < 5 then Char.ofNat (n + 48) :: acc
    else aux (n / 5) (Char.ofNat (n % 5 + 48) :: acc)
  String.mk (aux n [])

-- The expected sum in base 5
def expected_sum_base5 : String := "2062"

-- The Lean theorem to be proven
theorem sum_in_base5_correct : to_base5 sum = expected_sum_base5 :=
by
  sorry

end sum_in_base5_correct_l180_180110


namespace red_apples_count_l180_180860

-- Definitions based on conditions
def green_apples : ℕ := 2
def yellow_apples : ℕ := 14
def total_apples : ℕ := 19

-- Definition of red apples as a theorem to be proven
theorem red_apples_count :
  green_apples + yellow_apples + red_apples = total_apples → red_apples = 3 :=
by
  -- You would need to prove this using Lean
  sorry

end red_apples_count_l180_180860


namespace solutions_equation1_solutions_equation2_l180_180092

-- Definition for the first equation
def equation1 (x : ℝ) : Prop := 4 * x^2 - 9 = 0

-- Definition for the second equation
def equation2 (x : ℝ) : Prop := 2 * x^2 - 3 * x - 5 = 0

theorem solutions_equation1 (x : ℝ) :
  equation1 x ↔ (x = 3 / 2 ∨ x = -3 / 2) := 
  by sorry

theorem solutions_equation2 (x : ℝ) :
  equation2 x ↔ (x = 1 ∨ x = 5 / 2) := 
  by sorry

end solutions_equation1_solutions_equation2_l180_180092


namespace alpha_value_l180_180301

theorem alpha_value (α : ℝ) (h : (α * (α - 1) * (-1 : ℝ)^(α - 2)) = 4) : α = -4 :=
by
  sorry

end alpha_value_l180_180301


namespace not_enough_info_sweets_l180_180506

theorem not_enough_info_sweets
    (S : ℕ)         -- Initial number of sweet cookies.
    (initial_salty : ℕ := 6)  -- Initial number of salty cookies given as 6.
    (eaten_sweets : ℕ := 20)   -- Number of sweet cookies Paco ate.
    (eaten_salty : ℕ := 34)    -- Number of salty cookies Paco ate.
    (diff_eaten : eaten_salty - eaten_sweets = 14) -- Paco ate 14 more salty cookies than sweet cookies.
    : (∃ S', S' = S) → False :=  -- Conclusion: Not enough information to determine initial number of sweet cookies S.
by
  sorry

end not_enough_info_sweets_l180_180506


namespace selling_price_A_count_purchasing_plans_refund_amount_l180_180088

-- Problem 1
theorem selling_price_A (last_revenue this_revenue last_price this_price cars_sold : ℝ) 
    (last_revenue_eq : last_revenue = 1) (this_revenue_eq : this_revenue = 0.9)
    (diff_eq : last_price = this_price + 1)
    (same_cars : cars_sold ≠ 0) :
    this_price = 9 := by
  sorry

-- Problem 2
theorem count_purchasing_plans (cost_A cost_B total_cars min_cost max_cost : ℝ)
    (cost_A_eq : cost_A = 0.75) (cost_B_eq : cost_B = 0.6)
    (total_cars_eq : total_cars = 15) (min_cost_eq : min_cost = 0.99)
    (max_cost_eq : max_cost = 1.05) :
    ∃ n : ℕ, n = 5 := by
  sorry

-- Problem 3
theorem refund_amount (refund_A refund_B revenue_A revenue_B cost_A cost_B total_profits a : ℝ)
    (revenue_B_eq : revenue_B = 0.8) (cost_A_eq : cost_A = 0.75)
    (cost_B_eq : cost_B = 0.6) (total_profits_eq : total_profits = 30 - 15 * a) :
    a = 0.5 := by
  sorry

end selling_price_A_count_purchasing_plans_refund_amount_l180_180088


namespace petes_average_speed_l180_180250

-- Definitions of the conditions
def map_distance : ℝ := 5 -- in inches
def driving_time : ℝ := 6.5 -- in hours
def map_scale : ℝ := 0.01282051282051282 -- in inches per mile

-- Theorem statement: If the conditions are given, then the average speed is 60 miles per hour
theorem petes_average_speed :
  (map_distance / map_scale) / driving_time = 60 :=
by
  -- The proof will go here
  sorry

end petes_average_speed_l180_180250


namespace roots_quadratic_sum_product_l180_180619

theorem roots_quadratic_sum_product :
  (∀ x1 x2 : ℝ, (∀ x, x^2 - 4 * x + 3 = 0 → x = x1 ∨ x = x2) → (x1 + x2 - x1 * x2 = 1)) :=
by
  sorry

end roots_quadratic_sum_product_l180_180619


namespace solve_inequality_system_l180_180507

theorem solve_inequality_system (x : ℝ) :
  (5 * x - 1 > 3 * (x + 1)) ∧ (x - 1 ≤ 7 - x) ↔ (2 < x ∧ x ≤ 4) :=
by
  sorry

end solve_inequality_system_l180_180507


namespace projection_of_vector_l180_180951

open Real EuclideanSpace

noncomputable def vector_projection (a b : ℝ × ℝ) : ℝ × ℝ :=
  ((a.1 * b.1 + a.2 * b.2) / (b.1^2 + b.2^2)) • b

theorem projection_of_vector : 
  vector_projection (6, -3) (3, 0) = (6, 0) := 
by 
  sorry

end projection_of_vector_l180_180951


namespace complex_computation_l180_180607

theorem complex_computation : 3 * 3^4 - 27^63 / 27^61 = -486 := by
  sorry

end complex_computation_l180_180607


namespace find_fifth_month_sale_l180_180285

theorem find_fifth_month_sale (
  a1 a2 a3 a4 a6 : ℕ
) (avg_sales : ℕ)
  (h1 : a1 = 5420)
  (h2 : a2 = 5660)
  (h3 : a3 = 6200)
  (h4 : a4 = 6350)
  (h6 : a6 = 7070)
  (avg_condition : avg_sales = 6200)
  (total_condition : (a1 + a2 + a3 + a4 + a6 + (6500)) / 6 = avg_sales)
  : (∃ a5 : ℕ, a5 = 6500 ∧ (a1 + a2 + a3 + a4 + a5 + a6) / 6 = avg_sales) :=
by {
  sorry
}

end find_fifth_month_sale_l180_180285


namespace compound_interest_l180_180213

theorem compound_interest 
  (P : ℝ) (r : ℝ) (t : ℕ) : P = 500 → r = 0.02 → t = 3 → (P * (1 + r)^t) - P = 30.60 :=
by
  intros P_invest rate years
  simp [P_invest, rate, years]
  sorry

end compound_interest_l180_180213


namespace rational_comparison_correct_l180_180426

-- Definitions based on conditions 
def positive_gt_zero (a : ℚ) : Prop := 0 < a
def negative_lt_zero (a : ℚ) : Prop := a < 0
def positive_gt_negative (a b : ℚ) : Prop := positive_gt_zero a ∧ negative_lt_zero b ∧ a > b
def negative_comparison (a b : ℚ) : Prop := negative_lt_zero a ∧ negative_lt_zero b ∧ abs a > abs b ∧ a < b

-- Theorem to prove
theorem rational_comparison_correct :
  (0 < - (1 / 2)) = false ∧
  ((4 / 5) < - (6 / 7)) = false ∧
  ((9 / 8) > (8 / 9)) = true ∧
  (-4 > -3) = false :=
by
  -- Mark the proof as unfinished.
  sorry

end rational_comparison_correct_l180_180426


namespace rectangle_side_greater_than_twelve_l180_180089

theorem rectangle_side_greater_than_twelve (a b : ℕ) (h1 : a ≠ b) (h2 : a * b = 6 * (a + b)) : a > 12 ∨ b > 12 :=
sorry

end rectangle_side_greater_than_twelve_l180_180089


namespace dimes_in_piggy_bank_l180_180543

variable (q d : ℕ)

def total_coins := q + d = 100
def total_amount := 25 * q + 10 * d = 1975

theorem dimes_in_piggy_bank (h1 : total_coins q d) (h2 : total_amount q d) : d = 35 := by
  sorry

end dimes_in_piggy_bank_l180_180543


namespace find_m_l180_180212

theorem find_m (m : ℝ) : (∀ x : ℝ, 0 < x → x < 2 → - (1/2)*x^2 + 2*x > -m*x) ↔ m = -1 := 
sorry

end find_m_l180_180212


namespace Alexei_finished_ahead_of_Sergei_by_1_9_km_l180_180000

noncomputable def race_distance : ℝ := 10
noncomputable def v_A : ℝ := 1  -- speed of Alexei
noncomputable def v_V : ℝ := 0.9 * v_A  -- speed of Vitaly
noncomputable def v_S : ℝ := 0.81 * v_A  -- speed of Sergei

noncomputable def distance_Alexei_finished_Ahead_of_Sergei : ℝ :=
race_distance - (0.81 * race_distance)

theorem Alexei_finished_ahead_of_Sergei_by_1_9_km :
  distance_Alexei_finished_Ahead_of_Sergei = 1.9 :=
by
  simp [race_distance, v_A, v_V, v_S, distance_Alexei_finished_Ahead_of_Sergei]
  sorry

end Alexei_finished_ahead_of_Sergei_by_1_9_km_l180_180000


namespace exponentiation_power_rule_l180_180235

theorem exponentiation_power_rule (a : ℝ) : (a^2)^3 = a^6 :=
by sorry

end exponentiation_power_rule_l180_180235


namespace length_of_picture_frame_l180_180142

theorem length_of_picture_frame (P W : ℕ) (hP : P = 30) (hW : W = 10) : ∃ L : ℕ, 2 * (L + W) = P ∧ L = 5 :=
by
  sorry

end length_of_picture_frame_l180_180142


namespace M_inter_N_is_1_2_l180_180436

-- Definitions based on given conditions
def M : Set ℝ := { y | ∃ x : ℝ, x > 0 ∧ y = 2^x }
def N : Set ℝ := { x | 0 ≤ x ∧ x ≤ 2 }

-- Prove intersection of M and N is (1, 2]
theorem M_inter_N_is_1_2 :
  M ∩ N = { x | 1 < x ∧ x ≤ 2 } :=
by
  sorry

end M_inter_N_is_1_2_l180_180436


namespace tangent_value_of_k_k_range_l180_180300

noncomputable def f (x : Real) : Real := Real.exp (2 * x)
def g (k x : Real) : Real := k * x + 1

theorem tangent_value_of_k (k : Real) :
  (∃ t : Real, f t = g k t ∧ deriv f t = deriv (g k) t) → k = 2 :=
by
  sorry

theorem k_range (k : Real) (h : k > 0) :
  (∃ m : Real, m > 0 ∧ ∀ x : Real, 0 < x → x < m → |f x - g k x| > 2 * x) → 4 < k :=
by
  sorry

end tangent_value_of_k_k_range_l180_180300


namespace polynomial_coeffs_identity_l180_180815

theorem polynomial_coeffs_identity : 
  (∀ a b c : ℝ, (2 * x^4 + x^3 - 41 * x^2 + 83 * x - 45 = 
                (a * x^2 + b * x + c) * (x^2 + 4 * x + 9))
                  → a = 2 ∧ b = -7 ∧ c = -5) :=
by
  intros a b c h
  have h₁ : a = 2 := 
    sorry-- prove that a = 2
  have h₂ : b = -7 := 
    sorry-- prove that b = -7
  have h₃ : c = -5 := 
    sorry-- prove that c = -5
  exact ⟨h₁, h₂, h₃⟩

end polynomial_coeffs_identity_l180_180815


namespace equation_1_solutions_equation_2_solutions_l180_180305

-- Equation 1: Proving solutions for (x+8)(x+1) = -12
theorem equation_1_solutions (x : ℝ) :
  (x + 8) * (x + 1) = -12 ↔ x = -4 ∨ x = -5 :=
sorry

-- Equation 2: Proving solutions for (2x-3)^2 = 5(2x-3)
theorem equation_2_solutions (x : ℝ) :
  (2 * x - 3) ^ 2 = 5 * (2 * x - 3) ↔ x = 3 / 2 ∨ x = 4 :=
sorry

end equation_1_solutions_equation_2_solutions_l180_180305


namespace tangerines_in_one_box_l180_180484

theorem tangerines_in_one_box (total_tangerines boxes remaining_tangerines tangerines_per_box : ℕ) 
  (h1 : total_tangerines = 29)
  (h2 : boxes = 8)
  (h3 : remaining_tangerines = 5)
  (h4 : total_tangerines - remaining_tangerines = boxes * tangerines_per_box) :
  tangerines_per_box = 3 :=
by 
  sorry

end tangerines_in_one_box_l180_180484


namespace find_coordinates_of_P_l180_180875

-- Define the conditions
variable (x y : ℝ)
def in_second_quadrant := x < 0 ∧ y > 0
def distance_to_x_axis := abs y = 7
def distance_to_y_axis := abs x = 3

-- Define the statement to be proved in Lean 4
theorem find_coordinates_of_P :
  in_second_quadrant x y ∧ distance_to_x_axis y ∧ distance_to_y_axis x → (x, y) = (-3, 7) :=
by
  sorry

end find_coordinates_of_P_l180_180875


namespace work_problem_l180_180043

theorem work_problem (x : ℝ) (h1 : x > 0) 
                      (h2 : (2 * (1 / 4 + 1 / x) + 2 * (1 / x) = 1)) : 
                      x = 8 := sorry

end work_problem_l180_180043


namespace spicy_hot_noodles_plates_l180_180954

theorem spicy_hot_noodles_plates (total_plates lobster_rolls seafood_noodles spicy_hot_noodles : ℕ) :
  total_plates = 55 →
  lobster_rolls = 25 →
  seafood_noodles = 16 →
  spicy_hot_noodles = total_plates - (lobster_rolls + seafood_noodles) →
  spicy_hot_noodles = 14 := by
  intros h_total h_lobster h_seafood h_eq
  rw [h_total, h_lobster, h_seafood] at h_eq
  exact h_eq

end spicy_hot_noodles_plates_l180_180954


namespace line_through_diameter_l180_180724

theorem line_through_diameter (P : ℝ × ℝ) (hP : P = (2, 1)) (h_circle : ∀ x y : ℝ, (x - 1)^2 + y^2 = 4) :
  ∃ a b c : ℝ, a * P.1 + b * P.2 + c = 0 ∧ a = 1 ∧ b = -1 ∧ c = -1 :=
by
  exists 1, -1, -1
  sorry

end line_through_diameter_l180_180724


namespace find_number_l180_180771

theorem find_number (x : ℝ) (h : 7 * x = 50.68) : x = 7.24 :=
sorry

end find_number_l180_180771


namespace maximize_S_n_l180_180315

def a1 : ℚ := 5
def d : ℚ := -5 / 7

def S_n (n : ℕ) : ℚ :=
  (n * (2 * a1 + (n - 1) * d)) / 2

theorem maximize_S_n :
  (∃ n : ℕ, (S_n n ≥ S_n (n - 1)) ∧ (S_n n ≥ S_n (n + 1))) →
  (n = 7 ∨ n = 8) :=
sorry

end maximize_S_n_l180_180315


namespace rectangle_semicircle_area_split_l180_180939

open Real

/-- The main problem statement -/
theorem rectangle_semicircle_area_split 
  (A B D C N U T : ℝ)
  (AU_AN_UAlengths : AU = 84 ∧ AN = 126 ∧ UB = 168)
  (area_ratio : ∃ (ℓ : ℝ), ∃ (N U T : ℝ), 1 / 2 = area_differ / (area_left + area_right))
  (DA_calculation : DA = 63 * sqrt 6) :
  63 + 6 = 69
:=
sorry

end rectangle_semicircle_area_split_l180_180939


namespace equivalent_statements_l180_180775

variables (P Q : Prop)

theorem equivalent_statements (h : P → Q) : 
  ((¬ Q → ¬ P) ∧ (¬ P ∨ Q)) ↔ (P → Q) := by
sorry

end equivalent_statements_l180_180775


namespace tangent_line_proof_minimum_a_proof_l180_180892

noncomputable def f (x : ℝ) := 2 * Real.log x - 3 * x^2 - 11 * x

def tangent_equation_correct : Prop :=
  let y := f 1
  let slope := (2 / 1 - 6 * 1 - 11)
  (slope = -15) ∧ (y = -14) ∧ (∀ x y, y = -15 * (x - 1) + -14 ↔ 15 * x + y - 1 = 0)

def minimum_a_correct : Prop :=
  ∃ a : ℤ, 
    (∀ x, f x ≤ (a - 3) * x^2 + (2 * a - 13) * x - 2) ↔ (a = 2)

theorem tangent_line_proof : tangent_equation_correct := sorry

theorem minimum_a_proof : minimum_a_correct := sorry

end tangent_line_proof_minimum_a_proof_l180_180892


namespace earnings_per_weed_is_six_l180_180559

def flower_bed_weeds : ℕ := 11
def vegetable_patch_weeds : ℕ := 14
def grass_weeds : ℕ := 32
def grass_weeds_half : ℕ := grass_weeds / 2
def soda_cost : ℕ := 99
def money_left : ℕ := 147
def total_weeds : ℕ := flower_bed_weeds + vegetable_patch_weeds + grass_weeds_half
def total_money : ℕ := money_left + soda_cost

theorem earnings_per_weed_is_six :
  total_money / total_weeds = 6 :=
by
  sorry

end earnings_per_weed_is_six_l180_180559


namespace scientific_notation_of_0_000000023_l180_180009

theorem scientific_notation_of_0_000000023 : 
  0.000000023 = 2.3 * 10^(-8) :=
sorry

end scientific_notation_of_0_000000023_l180_180009


namespace geometric_sum_4_terms_l180_180830

theorem geometric_sum_4_terms 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h1 : a 2 = 9) 
  (h2 : a 5 = 243) 
  (hq : ∀ n, a (n + 1) = a n * q) 
  : a 1 * (1 - q^4) / (1 - q) = 120 := 
sorry

end geometric_sum_4_terms_l180_180830


namespace cube_faces_sum_eq_neg_3_l180_180021

theorem cube_faces_sum_eq_neg_3 
    (a b c d e f : ℤ)
    (h1 : a = -3)
    (h2 : b = a + 1)
    (h3 : c = b + 1)
    (h4 : d = c + 1)
    (h5 : e = d + 1)
    (h6 : f = e + 1)
    (h7 : a + f = b + e)
    (h8 : b + e = c + d) :
  a + b + c + d + e + f = -3 := sorry

end cube_faces_sum_eq_neg_3_l180_180021


namespace fraction_eggs_used_for_cupcakes_l180_180544

theorem fraction_eggs_used_for_cupcakes:
  ∀ (total_eggs crepes_fraction remaining_eggs after_cupcakes_eggs used_for_cupcakes_fraction: ℚ),
  total_eggs = 36 →
  crepes_fraction = 1 / 4 →
  after_cupcakes_eggs = 9 →
  used_for_cupcakes_fraction = 2 / 3 →
  (total_eggs * (1 - crepes_fraction) - after_cupcakes_eggs) / (total_eggs * (1 - crepes_fraction)) = used_for_cupcakes_fraction :=
by
  intros
  sorry

end fraction_eggs_used_for_cupcakes_l180_180544


namespace intersection_A_B_l180_180645

-- Define the sets A and B based on the given conditions
def A := { x : ℝ | (1 / 9) ≤ (3:ℝ)^x ∧ (3:ℝ)^x ≤ 1 }
def B := { x : ℝ | x^2 < 1 }

-- State the theorem for the intersection of sets A and B
theorem intersection_A_B :
  A ∩ B = { x : ℝ | -1 < x ∧ x ≤ 0 } :=
by
  sorry

end intersection_A_B_l180_180645


namespace find_BC_length_l180_180888

theorem find_BC_length
  (area : ℝ) (AB AC : ℝ)
  (h_area : area = 10 * Real.sqrt 3)
  (h_AB : AB = 5)
  (h_AC : AC = 8) :
  ∃ BC : ℝ, BC = 7 :=
by
  sorry

end find_BC_length_l180_180888


namespace correct_limiting_reagent_and_yield_l180_180077

noncomputable def balanced_reaction_theoretical_yield : Prop :=
  let Fe2O3_initial : ℕ := 4
  let CaCO3_initial : ℕ := 10
  let moles_Fe2O3_needed_for_CaCO3 := Fe2O3_initial * (6 / 2)
  let limiting_reagent := if CaCO3_initial < moles_Fe2O3_needed_for_CaCO3 then true else false
  let theoretical_yield := (CaCO3_initial * (3 / 6))
  limiting_reagent = true ∧ theoretical_yield = 5

theorem correct_limiting_reagent_and_yield : balanced_reaction_theoretical_yield :=
by
  sorry

end correct_limiting_reagent_and_yield_l180_180077


namespace mathd_inequality_l180_180203

theorem mathd_inequality (x y z : ℝ) (hx : x ≥ 1) (hy : y ≥ 1) (hz : z ≥ 1) : 
  (x^3 + 2*y^2 + 3*z) * (4*y^3 + 5*z^2 + 6*x) * (7*z^3 + 8*x^2 + 9*y) ≥ 720 * (x * y + y * z + z * x) :=
by
  sorry

end mathd_inequality_l180_180203


namespace robert_saves_5_dollars_l180_180066

theorem robert_saves_5_dollars :
  let original_price := 50
  let promotion_c_discount (price : ℕ) := price * 20 / 100
  let promotion_d_discount (price : ℕ) := 15
  let cost_promotion_c := original_price + (original_price - promotion_c_discount original_price)
  let cost_promotion_d := original_price + (original_price - promotion_d_discount original_price)
  (cost_promotion_c - cost_promotion_d) = 5 :=
by
  sorry

end robert_saves_5_dollars_l180_180066


namespace max_students_l180_180156

-- Definitions for the conditions
noncomputable def courses := ["Mathematics", "Physics", "Biology", "Music", "History", "Geography"]

def most_preferred (ranking : List String) : Prop :=
  "Mathematics" ∈ (ranking.take 2) ∨ "Mathematics" ∈ (ranking.take 3)

def least_preferred (ranking : List String) : Prop :=
  "Music" ∉ ranking.drop (ranking.length - 2)

def preference_constraints (ranking : List String) : Prop :=
  ranking.indexOf "History" < ranking.indexOf "Geography" ∧
  ranking.indexOf "Physics" < ranking.indexOf "Biology"

def all_rankings_unique (rankings : List (List String)) : Prop :=
  ∀ (r₁ r₂ : List String), r₁ ≠ r₂ → r₁ ∈ rankings → r₂ ∈ rankings → r₁ ≠ r₂

-- The goal statement
theorem max_students : 
  ∃ (rankings : List (List String)), 
  (∀ r ∈ rankings, most_preferred r) ∧
  (∀ r ∈ rankings, least_preferred r) ∧
  (∀ r ∈ rankings, preference_constraints r) ∧
  all_rankings_unique rankings ∧
  rankings.length = 44 :=
sorry

end max_students_l180_180156


namespace product_of_roots_l180_180854

theorem product_of_roots :
  (∃ r s t : ℝ, (r + s + t) = 15 ∧ (r*s + s*t + r*t) = 50 ∧ (r*s*t) = -35) ∧ (∀ x : ℝ, x^3 - 15*x^2 + 50*x + 35 = (x - r) * (x - s) * (x - t)) :=
sorry

end product_of_roots_l180_180854


namespace age_equation_correct_l180_180058

-- Define the main theorem
theorem age_equation_correct (x : ℕ) (h1 : ∀ (b : ℕ), b = 2 * x) (h2 : ∀ (b4 s4 : ℕ), b4 = b - 4 ∧ s4 = x - 4 ∧ b4 = 3 * s4) : 
  2 * x - 4 = 3 * (x - 4) :=
by
  sorry

end age_equation_correct_l180_180058


namespace find_base_a_l180_180547

theorem find_base_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1)
  (h3 : (if a < 1 then a + a^2 else a^2 + a) = 12) : a = 3 := 
sorry

end find_base_a_l180_180547


namespace binomial_30_3_l180_180576

theorem binomial_30_3 : Nat.choose 30 3 = 4060 := by
  sorry

end binomial_30_3_l180_180576


namespace num_three_digit_integers_divisible_by_12_l180_180448

theorem num_three_digit_integers_divisible_by_12 : 
  (∃ (count : ℕ), count = 3 ∧ 
    (∀ (n : ℕ), 100 ≤ n ∧ n ≤ 999 → 
      (∀ d : ℕ, d ∈ [n / 100, (n / 10) % 10, n % 10] → 4 < d) ∧ 
      n % 12 = 0 → 
      count = count + 1)) := 
sorry

end num_three_digit_integers_divisible_by_12_l180_180448


namespace john_age_l180_180117

/-!
# John’s Current Age Proof
Given the following condition:
1. 9 years from now, John will be 3 times as old as he was 11 years ago.
Prove that John is currently 21 years old.
-/

def john_current_age (x : ℕ) : Prop :=
  (x + 9 = 3 * (x - 11)) → (x = 21)

-- Proof Statement
theorem john_age : john_current_age 21 :=
by
  sorry

end john_age_l180_180117


namespace f_odd_solve_inequality_l180_180364

noncomputable def f : ℝ → ℝ := sorry

axiom f_additive : ∀ x y : ℝ, f (x + y) = f x + f y
axiom f_positive : ∀ x : ℝ, x > 0 → f x > 0

theorem f_odd : ∀ x : ℝ, f (-x) = -f x := 
by
  sorry

theorem solve_inequality : {a : ℝ | f (a-4) + f (2*a+1) < 0} = {a | a < 1} := 
by
  sorry

end f_odd_solve_inequality_l180_180364


namespace sum_of_digits_l180_180233

theorem sum_of_digits (a b c d : ℕ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
                      (h_range : 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9 ∧ 1 ≤ d ∧ d ≤ 9)
                      (h_product : a * b * c * d = 810) :
  a + b + c + d = 23 := sorry

end sum_of_digits_l180_180233


namespace pet_shop_ways_l180_180743

theorem pet_shop_ways (puppies : ℕ) (kittens : ℕ) (turtles : ℕ)
  (h_puppies : puppies = 10) (h_kittens : kittens = 8) (h_turtles : turtles = 5) : 
  (puppies * kittens * turtles = 400) :=
by
  sorry

end pet_shop_ways_l180_180743


namespace geometric_sequence_sum_l180_180199

-- Define the geometric sequence {a_n}
def a (n : ℕ) : ℕ := 2 * (1 ^ (n - 1))

-- Define the sum of the first n terms, s_n
def s (n : ℕ) : ℕ := (Finset.range n).sum (a)

-- The transformed sequence {a_n + 1} assumed also geometric
def b (n : ℕ) : ℕ := a n + 1

-- Lean theorem that s_n = 2n
theorem geometric_sequence_sum (n : ℕ) (h₁ : a 1 = 2) (h₂ : ∀ n, (b (n + 1)) * (b (n + 1)) = (b n * b (n + 2))) : 
  s n = 2 * n :=
sorry

end geometric_sequence_sum_l180_180199


namespace rational_sum_p_q_l180_180780

noncomputable def x := (Real.sqrt 5 - 1) / 2

theorem rational_sum_p_q :
  ∃ (p q : ℚ), x^3 + p * x + q = 0 ∧ p + q = -1 := by
  sorry

end rational_sum_p_q_l180_180780


namespace A_receives_more_than_B_l180_180280

variable (A B C : ℝ)

axiom h₁ : A = 1/3 * (B + C)
axiom h₂ : B = 2/7 * (A + C)
axiom h₃ : A + B + C = 720

theorem A_receives_more_than_B : A - B = 20 :=
by
  sorry

end A_receives_more_than_B_l180_180280


namespace max_value_of_f_l180_180461

noncomputable def f (x a b : ℝ) := (1 - x ^ 2) * (x ^ 2 + a * x + b)

theorem max_value_of_f (a b : ℝ) (x : ℝ) 
  (h1 : (∀ x, f x 8 15 = (1 - x^2) * (x^2 + 8*x + 15)))
  (h2 : ∀ x, f x a b = f (-(x + 4)) a b) :
  ∃ m, (∀ x, f x 8 15 ≤ m) ∧ m = 16 :=
by
  sorry

end max_value_of_f_l180_180461


namespace geometric_sequence_problem_l180_180838

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ m n p q : ℕ, m + n = p + q → a m * a n = a p * a q

def condition (a : ℕ → ℝ) : Prop :=
a 4 + a 8 = -3

theorem geometric_sequence_problem (a : ℕ → ℝ) (h1 : geometric_sequence a) (h2 : condition a) :
  a 6 * (a 2 + 2 * a 6 + a 10) = 9 :=
sorry

end geometric_sequence_problem_l180_180838


namespace solve_inequality_l180_180914

theorem solve_inequality (x : ℝ) : (2 ≤ |3 * x - 6| ∧ |3 * x - 6| ≤ 12) ↔ (x ∈ Set.Icc (-2 : ℝ) (4 / 3) ∨ x ∈ Set.Icc (8 / 3) (6 : ℝ)) :=
sorry

end solve_inequality_l180_180914


namespace distance_between_A_and_mrs_A_l180_180070

-- Define the initial conditions
def speed_mr_A : ℝ := 30 -- Mr. A's speed in kmph
def speed_mrs_A : ℝ := 10 -- Mrs. A's speed in kmph
def speed_bee : ℝ := 60 -- The bee's speed in kmph
def distance_bee_traveled : ℝ := 180 -- Distance traveled by the bee in km

-- Define the proven statement
theorem distance_between_A_and_mrs_A : 
  distance_bee_traveled / speed_bee * (speed_mr_A + speed_mrs_A) = 120 := 
by 
  sorry

end distance_between_A_and_mrs_A_l180_180070


namespace share_apples_l180_180585

theorem share_apples (h : 9 / 3 = 3) : true :=
sorry

end share_apples_l180_180585


namespace find_g_of_5_l180_180256

theorem find_g_of_5 (g : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, g (x * y) = g x * g y) 
  (h2 : g 1 = 2) : 
  g 5 = 32 := 
by 
  sorry

end find_g_of_5_l180_180256


namespace incorrect_statement_B_l180_180929

axiom statement_A : ¬ (0 > 0 ∨ 0 < 0)
axiom statement_C : ∀ (q : ℚ), (∃ (m : ℤ), q = m) ∨ (∃ (a b : ℤ), b ≠ 0 ∧ q = a / b)
axiom statement_D : abs (0 : ℚ) = 0

theorem incorrect_statement_B : ¬ (∀ (q : ℚ), abs q ≥ 1 → abs 1 = abs q) := sorry

end incorrect_statement_B_l180_180929


namespace total_points_l180_180895

theorem total_points (zach_points ben_points : ℝ) (h₁ : zach_points = 42.0) (h₂ : ben_points = 21.0) : zach_points + ben_points = 63.0 :=
  by sorry

end total_points_l180_180895


namespace arithmetic_sequence_problem_l180_180579

variable {a : ℕ → ℝ} {d : ℝ} -- Declare the sequence and common difference

-- Define the arithmetic sequence property
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions
def given_conditions (a : ℕ → ℝ) (d : ℝ) : Prop :=
  a 5 + a 10 = 12 ∧ arithmetic_sequence a d

-- Main theorem statement
theorem arithmetic_sequence_problem (a : ℕ → ℝ) (d : ℝ) 
  (h : given_conditions a d) :
  3 * a 7 + a 9 = 24 :=
sorry

end arithmetic_sequence_problem_l180_180579


namespace determine_f_value_l180_180065

-- Define initial conditions
def parabola_eqn (d e f : ℝ) (y : ℝ) : ℝ := d * y^2 + e * y + f
def vertex : (ℝ × ℝ) := (2, -3)
def point_on_parabola : (ℝ × ℝ) := (7, 0)

-- Prove that f = 7 given the conditions
theorem determine_f_value (d e f : ℝ) :
  (parabola_eqn d e f (vertex.snd) = vertex.fst) ∧
  (parabola_eqn d e f (point_on_parabola.snd) = point_on_parabola.fst) →
  f = 7 := 
by
  sorry 

end determine_f_value_l180_180065


namespace sarah_age_l180_180767

variable (s m : ℕ)

theorem sarah_age (h1 : s = m - 18) (h2 : s + m = 50) : s = 16 :=
by {
  -- The proof will go here
  sorry
}

end sarah_age_l180_180767


namespace tonya_payment_l180_180844

def original_balance : ℝ := 150.00
def new_balance : ℝ := 120.00

noncomputable def payment_amount : ℝ := original_balance - new_balance

theorem tonya_payment :
  payment_amount = 30.00 :=
by
  sorry

end tonya_payment_l180_180844


namespace john_total_skateboarded_distance_l180_180738

noncomputable def total_skateboarded_distance (to_park: ℕ) (back_home: ℕ) : ℕ :=
  to_park + back_home

theorem john_total_skateboarded_distance :
  total_skateboarded_distance 10 10 = 20 :=
by
  sorry

end john_total_skateboarded_distance_l180_180738


namespace range_of_function_l180_180861

theorem range_of_function : 
  ∀ y : ℝ, (∃ x : ℝ, y = x / (1 + x^2)) ↔ (-1 / 2 ≤ y ∧ y ≤ 1 / 2) := 
by sorry

end range_of_function_l180_180861


namespace problem_equivalent_l180_180413

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^5 + a * x^3 + b * x - 8

theorem problem_equivalent (a b : ℝ) (h : f (-2) a b = 10) : f 2 a b = -10 :=
by
  sorry

end problem_equivalent_l180_180413


namespace initial_volume_of_solution_l180_180283

variable (V : ℝ)
variables (h1 : 0.10 * V = 0.08 * (V + 16))
variables (V_correct : V = 64)

theorem initial_volume_of_solution : V = 64 := by
  sorry

end initial_volume_of_solution_l180_180283


namespace circle_repr_eq_l180_180660

theorem circle_repr_eq (a : ℝ) : (∃ x y : ℝ, x^2 + y^2 - 2*x + 4*y + 1 + a = 0) ↔ a < 4 :=
by
  sorry

end circle_repr_eq_l180_180660


namespace fixed_point_coordinates_l180_180483

theorem fixed_point_coordinates (k : ℝ) (M : ℝ × ℝ) (h : ∀ k : ℝ, M.2 - 2 = k * (M.1 + 1)) :
  M = (-1, 2) :=
sorry

end fixed_point_coordinates_l180_180483


namespace min_sum_of_ab_l180_180531

theorem min_sum_of_ab (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 2 * a + 3 * b = a * b) :
  a + b ≥ 5 + 2 * Real.sqrt 6 :=
sorry

end min_sum_of_ab_l180_180531


namespace peanut_butter_candy_count_l180_180632

theorem peanut_butter_candy_count (B G P : ℕ) 
  (hB : B = 43)
  (hG : G = B + 5)
  (hP : P = 4 * G) :
  P = 192 := by
  sorry

end peanut_butter_candy_count_l180_180632


namespace isosceles_right_triangle_legs_are_26_cm_and_hypotenuse_is_26_sqrt2_cm_l180_180801

def isosceles_right_triangle_initial_leg_length (x : ℝ) (h : ℝ) : Prop :=
  x + 4 * ((x + 4) / 2) ^ 2 = x * x / 2 + 112 

def isosceles_right_triangle_legs_correct (a b : ℝ) (h : ℝ) : Prop :=
  a = 26 ∧ b = 26 * Real.sqrt 2

theorem isosceles_right_triangle_legs_are_26_cm_and_hypotenuse_is_26_sqrt2_cm :
  ∃ (x : ℝ) (h : ℝ), isosceles_right_triangle_initial_leg_length x h ∧ 
                       isosceles_right_triangle_legs_correct x (x * Real.sqrt 2) h := 
by
  sorry

end isosceles_right_triangle_legs_are_26_cm_and_hypotenuse_is_26_sqrt2_cm_l180_180801


namespace tan_alpha_l180_180112

variable (α : ℝ)
variable (H_cos : Real.cos α = 12/13)
variable (H_quadrant : 3 * Real.pi / 2 < α ∧ α < 2 * Real.pi)

theorem tan_alpha :
  Real.tan α = -5/12 :=
sorry

end tan_alpha_l180_180112


namespace graph_transformation_l180_180600

theorem graph_transformation (a b c : ℝ) (h1 : c = 1) (h2 : a + b + c = -2) (h3 : a - b + c = 2) :
  (∀ x, cx^2 + 2 * bx + a = (x - 2)^2 - 5) := 
sorry

end graph_transformation_l180_180600


namespace perimeter_of_square_B_l180_180201

theorem perimeter_of_square_B
  (perimeter_A : ℝ)
  (h_perimeter_A : perimeter_A = 36)
  (area_ratio : ℝ)
  (h_area_ratio : area_ratio = 1 / 3)
  : ∃ (perimeter_B : ℝ), perimeter_B = 12 * Real.sqrt 3 :=
by
  sorry

end perimeter_of_square_B_l180_180201


namespace board_officer_election_l180_180910

def num_ways_choose_officers (total_members : ℕ) (elect_officers : ℕ) : ℕ :=
  -- This will represent the number of ways to choose 4 officers given 30 members
  -- with the conditions on Alice, Bob, Chris, and Dana.
  if total_members = 30 ∧ elect_officers = 4 then
    358800 + 7800 + 7800 + 24
  else
    0

theorem board_officer_election : num_ways_choose_officers 30 4 = 374424 :=
by {
  -- Proof would go here
  sorry
}

end board_officer_election_l180_180910


namespace number_of_plain_lemonade_sold_l180_180334

theorem number_of_plain_lemonade_sold
  (price_per_plain_lemonade : ℝ)
  (earnings_strawberry_lemonade : ℝ)
  (earnings_more_plain_than_strawberry : ℝ)
  (P : ℝ)
  (H1 : price_per_plain_lemonade = 0.75)
  (H2 : earnings_strawberry_lemonade = 16)
  (H3 : earnings_more_plain_than_strawberry = 11)
  (H4 : price_per_plain_lemonade * P = earnings_strawberry_lemonade + earnings_more_plain_than_strawberry) :
  P = 36 :=
by
  sorry

end number_of_plain_lemonade_sold_l180_180334


namespace cos_value_l180_180842

theorem cos_value (A : ℝ) (h : Real.sin (π + A) = 1/2) : Real.cos (3*π/2 - A) = 1/2 :=
sorry

end cos_value_l180_180842


namespace distribution_of_collection_items_l180_180321

-- Declaring the collections
structure Collection where
  stickers : Nat
  baseball_cards : Nat
  keychains : Nat
  stamps : Nat

-- Defining the individual collections based on the conditions
def Karl : Collection := { stickers := 25, baseball_cards := 15, keychains := 5, stamps := 10 }
def Ryan : Collection := { stickers := Karl.stickers + 20, baseball_cards := Karl.baseball_cards - 10, keychains := Karl.keychains + 2, stamps := Karl.stamps }
def Ben_scenario1 : Collection := { stickers := Ryan.stickers - 10, baseball_cards := (Ryan.baseball_cards / 2), keychains := Karl.keychains * 2, stamps := Karl.stamps + 5 }

-- Total number of items in the collection
def total_items_scenario1 :=
  Karl.stickers + Karl.baseball_cards + Karl.keychains + Karl.stamps +
  Ryan.stickers + Ryan.baseball_cards + Ryan.keychains + Ryan.stamps +
  Ben_scenario1.stickers + Ben_scenario1.baseball_cards + Ben_scenario1.keychains + Ben_scenario1.stamps

-- The proof statement
theorem distribution_of_collection_items :
  total_items_scenario1 = 184 ∧ total_items_scenario1 % 4 = 0 → (184 / 4 = 46) := 
by
  sorry

end distribution_of_collection_items_l180_180321


namespace angus_caught_4_more_l180_180766

theorem angus_caught_4_more (
  angus ollie patrick: ℕ
) (
  h1: ollie = angus - 7
) (
  h2: ollie = 5
) (
  h3: patrick = 8
) : (angus - patrick) = 4 := 
sorry

end angus_caught_4_more_l180_180766


namespace minimum_value_of_expression_l180_180310

noncomputable def min_expression_value (a b : ℝ) : ℝ :=
  1 / (1 + a) + 4 / (2 + b)

theorem minimum_value_of_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + 3 * b = 7) : 
  min_expression_value a b ≥ (13 + 4 * Real.sqrt 3) / 14 :=
by
  sorry

end minimum_value_of_expression_l180_180310


namespace expected_value_of_8_sided_die_l180_180093

-- Define the expected value function for the given win calculation rule.
def expected_value := (1/8 : ℚ) * (7 + 6 + 5 + 4 + 3 + 2 + 1 + 0)

-- Formal statement of the proof problem.
theorem expected_value_of_8_sided_die : 
  expected_value = 3.50 :=
by
  sorry

end expected_value_of_8_sided_die_l180_180093


namespace determine_functions_l180_180927

noncomputable def f : (ℝ → ℝ) := sorry

theorem determine_functions (f : ℝ → ℝ)
  (h_domain: ∀ x, 0 < x → 0 < f x)
  (h_eq: ∀ w x y z, 0 < w → 0 < x → 0 < y → 0 < z → w * x = y * z →
    (f w)^2 + (f x)^2 = (f (y^2) + f (z^2)) * (w^2 + x^2) / (y^2 + z^2)) :
  (∀ x, 0 < x → (f x = x ∨ f x = 1 / x)) :=
by
  intros x hx
  sorry

end determine_functions_l180_180927


namespace geometric_sum_formula_l180_180499

noncomputable def geometric_sequence_sum (n : ℕ) : ℕ :=
  sorry

theorem geometric_sum_formula (a : ℕ → ℕ)
  (h_geom : ∀ n, a (n + 1) = 2 * a n)
  (h_a1_a2 : a 0 + a 1 = 3)
  (h_a1_a2_a3 : a 0 * a 1 * a 2 = 8) :
  geometric_sequence_sum n = 2^n - 1 :=
sorry

end geometric_sum_formula_l180_180499


namespace distance_between_foci_of_hyperbola_l180_180042

open Real

-- Definitions based on the given conditions
def asymptote1 (x : ℝ) : ℝ := x + 3
def asymptote2 (x : ℝ) : ℝ := -x + 5
def hyperbola_passes_through (x y : ℝ) : Prop := x = 4 ∧ y = 6
noncomputable def hyperbola_centre : (ℝ × ℝ) := (1, 4)

-- Definition of the hyperbola and the proof problem
theorem distance_between_foci_of_hyperbola (x y : ℝ) (hx : asymptote1 x = y) (hy : asymptote2 x = y) (hpass : hyperbola_passes_through 4 6) :
  2 * (sqrt (5 + 5)) = 2 * sqrt 10 :=
by
  sorry

end distance_between_foci_of_hyperbola_l180_180042


namespace monica_studied_32_67_hours_l180_180756

noncomputable def monica_total_study_time : ℚ :=
  let monday := 1
  let tuesday := 2 * monday
  let wednesday := 2
  let thursday := 3 * wednesday
  let friday := thursday / 2
  let total_weekday := monday + tuesday + wednesday + thursday + friday
  let saturday := total_weekday
  let sunday := saturday / 3
  total_weekday + saturday + sunday

theorem monica_studied_32_67_hours :
  monica_total_study_time = 32.67 := by
  sorry

end monica_studied_32_67_hours_l180_180756


namespace copies_made_in_half_hour_l180_180119

theorem copies_made_in_half_hour :
  let copies_per_minute_machine1 := 40
  let copies_per_minute_machine2 := 55
  let time_minutes := 30
  (copies_per_minute_machine1 * time_minutes) + (copies_per_minute_machine2 * time_minutes) = 2850 := by
    sorry

end copies_made_in_half_hour_l180_180119


namespace angle_at_3_15_l180_180090

-- Define the measurements and conditions
def hour_hand_position (hour min : ℕ) : ℝ := 
  30 * hour + 0.5 * min

def minute_hand_position (min : ℕ) : ℝ := 
  6 * min

def angle_between_hands (hour min : ℕ) : ℝ := 
  abs (minute_hand_position min - hour_hand_position hour min)

-- Theorem statement in Lean 4
theorem angle_at_3_15 : angle_between_hands 3 15 = 7.5 :=
by sorry

end angle_at_3_15_l180_180090


namespace initial_bales_l180_180944

theorem initial_bales (bales_initially bales_added bales_now : ℕ)
  (h₀ : bales_added = 26)
  (h₁ : bales_now = 54)
  (h₂ : bales_now = bales_initially + bales_added) :
  bales_initially = 28 :=
by
  sorry

end initial_bales_l180_180944


namespace value_of_x_l180_180399

theorem value_of_x (x : ℝ) (h : x = -x) : x = 0 := 
by 
  sorry

end value_of_x_l180_180399


namespace log_product_eq_one_sixth_log_y_x_l180_180835

variable (x y : ℝ) (hx : 0 < x) (hy : 0 < y)

theorem log_product_eq_one_sixth_log_y_x :
  (Real.log x ^ 2 / Real.log (y ^ 5)) * 
  (Real.log (y ^ 3) / Real.log (x ^ 4)) *
  (Real.log (x ^ 4) / Real.log (y ^ 3)) *
  (Real.log (y ^ 5) / Real.log (x ^ 3)) *
  (Real.log (x ^ 3) / Real.log (y ^ 4)) = 
  (1 / 6) * (Real.log x / Real.log y) := 
sorry

end log_product_eq_one_sixth_log_y_x_l180_180835


namespace minimum_empty_cells_face_move_minimum_empty_cells_diagonal_move_l180_180779

-- Definition for Problem Part (a)
def box_dimensions := (3, 5, 7)
def initial_cockchafers := 3 * 5 * 7 -- or 105

-- Defining the theorem for part (a)
theorem minimum_empty_cells_face_move (d : (ℕ × ℕ × ℕ)) (n : ℕ) :
  d = box_dimensions →
  n = initial_cockchafers →
  ∃ k ≥ 1, k = 1 :=
by
  intros hdim hn
  sorry

-- Definition for Problem Part (b)
def row_odd_cells := 2 * 5 * 7  
def row_even_cells := 1 * 5 * 7  

-- Defining the theorem for part (b)
theorem minimum_empty_cells_diagonal_move (r_odd r_even : ℕ) :
  r_odd = row_odd_cells →
  r_even = row_even_cells →
  ∃ m ≥ 35, m = 35 :=
by
  intros ho he
  sorry

end minimum_empty_cells_face_move_minimum_empty_cells_diagonal_move_l180_180779


namespace find_parallel_and_perpendicular_lines_through_A_l180_180833

def point_A : ℝ × ℝ := (2, 2)

def line_l (x y : ℝ) : Prop := 3 * x + 4 * y - 20 = 0

def parallel_line_l1 (x y : ℝ) : Prop := 3 * x + 4 * y - 14 = 0

def perpendicular_line_l2 (x y : ℝ) : Prop := 4 * x - 3 * y - 2 = 0

theorem find_parallel_and_perpendicular_lines_through_A :
  (∀ x y, line_l x y → parallel_line_l1 x y) ∧
  (∀ x y, line_l x y → perpendicular_line_l2 x y) :=
by
  sorry

end find_parallel_and_perpendicular_lines_through_A_l180_180833


namespace complementary_angle_l180_180302

theorem complementary_angle (α : ℝ) (h : α = 35 + 30 / 60) : 90 - α = 54 + 30 / 60 :=
by
  sorry

end complementary_angle_l180_180302


namespace polynomial_value_at_2_l180_180057

-- Define the polynomial f(x)
def f (x : ℝ) : ℝ := x^6 - 12*x^5 + 60*x^4 - 160*x^3 + 240*x^2 - 192*x + 64

-- Define the transformation rules for each v_i according to Horner's Rule
def v0 : ℝ := 1
def v1 (x : ℝ) : ℝ := (v0 * x) - 12
def v2 (x : ℝ) : ℝ := (v1 x * x) + 60
def v3 (x : ℝ) : ℝ := (v2 x * x) - 160

-- State the theorem to be proven
theorem polynomial_value_at_2 : v3 2 = -80 := 
by 
  -- Since this is just a Lean 4 statement, we include sorry to defer proof
  sorry

end polynomial_value_at_2_l180_180057


namespace range_of_a_l180_180113

structure PropositionP (a : ℝ) : Prop :=
  (h : 2 * a + 1 > 5)

structure PropositionQ (a : ℝ) : Prop :=
  (h : -1 ≤ a ∧ a ≤ 3)

theorem range_of_a (a : ℝ) (hp : PropositionP a ∨ PropositionQ a) (hq : ¬(PropositionP a ∧ PropositionQ a)) :
  (-1 ≤ a ∧ a ≤ 2) ∨ (a > 3) :=
sorry

end range_of_a_l180_180113


namespace how_many_years_younger_is_C_compared_to_A_l180_180459

variables (a b c d : ℕ)

def condition1 : Prop := a + b = b + c + 13
def condition2 : Prop := b + d = c + d + 7
def condition3 : Prop := a + d = 2 * c - 12

theorem how_many_years_younger_is_C_compared_to_A
  (h1 : condition1 a b c)
  (h2 : condition2 b c d)
  (h3 : condition3 a c d) : a = c + 13 :=
sorry

end how_many_years_younger_is_C_compared_to_A_l180_180459


namespace contrapositive_example_l180_180512

theorem contrapositive_example (x : ℝ) (h : -2 < x ∧ x < 2) : x^2 < 4 :=
sorry

end contrapositive_example_l180_180512


namespace find_x_l180_180403

def bin_op (p1 p2 : ℤ × ℤ) : ℤ × ℤ :=
  (p1.1 - 2 * p2.1, p1.2 + 2 * p2.2)

theorem find_x :
  ∃ x y : ℤ, 
  bin_op (2, -4) (1, -3) = bin_op (x, y) (2, 1) ∧ x = 4 :=
by
  sorry

end find_x_l180_180403


namespace green_chameleon_increase_l180_180140

variable (initial_yellow: ℕ) (initial_green: ℕ)

-- Number of sunny and cloudy days
def sunny_days: ℕ := 18
def cloudy_days: ℕ := 12

-- Change in yellow chameleons
def delta_yellow: ℕ := 5

theorem green_chameleon_increase 
  (initial_yellow: ℕ) (initial_green: ℕ) 
  (sunny_days: ℕ := 18) 
  (cloudy_days: ℕ := 12) 
  (delta_yellow: ℕ := 5): 
  (initial_green + 11) = initial_green + delta_yellow + (sunny_days - cloudy_days) := 
by
  sorry

end green_chameleon_increase_l180_180140


namespace larger_pie_crust_flour_l180_180960

theorem larger_pie_crust_flour
  (p1 p2 : ℕ)
  (f1 f2 c : ℚ)
  (h1 : p1 = 36)
  (h2 : p2 = 24)
  (h3 : f1 = 1 / 8)
  (h4 : p1 * f1 = c)
  (h5 : p2 * f2 = c)
  : f2 = 3 / 16 :=
sorry

end larger_pie_crust_flour_l180_180960


namespace triangle_inequality_l180_180374

theorem triangle_inequality (a : ℝ) :
  (3/2 < a) ∧ (a < 5) ↔ ((4 * a + 1 - (3 * a - 1) < 12 - a) ∧ (4 * a + 1 + (3 * a - 1) > 12 - a)) := 
by 
  sorry

end triangle_inequality_l180_180374


namespace similar_iff_condition_l180_180751

-- Define the similarity of triangles and the necessary conditions.
variables {α : Type*} [LinearOrderedField α]
variables (a b c a' b' c' : α)

-- Statement of the problem in Lean 4
theorem similar_iff_condition : 
  (∃ z w : α, a' = a * z + w ∧ b' = b * z + w ∧ c' = c * z + w) ↔ 
  (a' * (b - c) + b' * (c - a) + c' * (a - b) = 0) :=
sorry

end similar_iff_condition_l180_180751


namespace box_volume_l180_180621

theorem box_volume
  (l w h : ℝ)
  (A1 : l * w = 36)
  (A2 : w * h = 18)
  (A3 : l * h = 8) :
  l * w * h = 102 := 
sorry

end box_volume_l180_180621


namespace marcus_sees_7_l180_180456

variable (marcus humphrey darrel : ℕ)
variable (humphrey_sees : humphrey = 11)
variable (darrel_sees : darrel = 9)
variable (average_is_9 : (marcus + humphrey + darrel) / 3 = 9)

theorem marcus_sees_7 : marcus = 7 :=
by
  -- Needs proof
  sorry

end marcus_sees_7_l180_180456


namespace train_length_approx_90_l180_180416

noncomputable def speed_in_m_per_s := (124 : ℝ) * (1000 / 3600)

noncomputable def time_in_s := (2.61269421026963 : ℝ)

noncomputable def length_of_train := speed_in_m_per_s * time_in_s

theorem train_length_approx_90 : abs (length_of_train - 90) < 1e-9 :=
  by
  sorry

end train_length_approx_90_l180_180416


namespace four_times_angle_triangle_l180_180731

theorem four_times_angle_triangle (A B C : ℕ) 
  (h1 : A + B + C = 180) 
  (h2 : A = 40)
  (h3 : (A = 4 * C) ∨ (B = 4 * C) ∨ (C = 4 * A)) : 
  (B = 130 ∧ C = 10) ∨ (B = 112 ∧ C = 28) :=
by
  sorry

end four_times_angle_triangle_l180_180731


namespace sum_of_eight_numbers_l180_180288

theorem sum_of_eight_numbers (avg : ℝ) (S : ℝ) (h1 : avg = 5.3) (h2 : avg = S / 8) : S = 42.4 :=
by
  sorry

end sum_of_eight_numbers_l180_180288


namespace A_union_B_subset_B_A_intersection_B_subset_B_l180_180446

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 3 * x - 10 <= 0}
def B (m : ℝ) : Set ℝ := {x | m - 4 <= x ∧ x <= 3 * m + 2}

-- Problem 1: Prove the range of m if A ∪ B = B
theorem A_union_B_subset_B (m : ℝ) : (A ∪ B m = B m) → (1 ≤ m ∧ m ≤ 2) :=
by
  sorry

-- Problem 2: Prove the range of m if A ∩ B = B
theorem A_intersection_B_subset_B (m : ℝ) : (A ∩ B m = B m) → (m < -3) :=
by
  sorry

end A_union_B_subset_B_A_intersection_B_subset_B_l180_180446


namespace students_basketball_cricket_l180_180811

theorem students_basketball_cricket (A B: ℕ) (AB: ℕ):
  A = 12 →
  B = 8 →
  AB = 3 →
  (A + B - AB) = 17 :=
by
  intros
  sorry

end students_basketball_cricket_l180_180811


namespace number_of_possible_outcomes_l180_180774

theorem number_of_possible_outcomes : 
  ∃ n : ℕ, n = 30 ∧
  ∀ (total_shots successful_shots consecutive_hits : ℕ),
  total_shots = 8 ∧ successful_shots = 3 ∧ consecutive_hits = 2 →
  n = 30 := 
by
  sorry

end number_of_possible_outcomes_l180_180774


namespace always_composite_l180_180363

theorem always_composite (p : ℕ) (hp : Nat.Prime p) : ¬Nat.Prime (p^2 + 35) ∧ ¬Nat.Prime (p^2 + 55) :=
by
  sorry

end always_composite_l180_180363


namespace mass_percentage_ca_in_compound_l180_180615

noncomputable def mass_percentage_ca_in_cac03 : ℝ :=
  let mm_ca := 40.08
  let mm_c := 12.01
  let mm_o := 16.00
  let mm_caco3 := mm_ca + mm_c + 3 * mm_o
  (mm_ca / mm_caco3) * 100

theorem mass_percentage_ca_in_compound (mp : ℝ) (h : mp = mass_percentage_ca_in_cac03) : mp = 40.04 := by
  sorry

end mass_percentage_ca_in_compound_l180_180615


namespace two_digit_sum_reverse_l180_180434

theorem two_digit_sum_reverse (a b : ℕ) (h1 : 1 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9) (h : (10 * a + b) - (10 * b + a) = 7 * (a + b)) :
  (10 * a + b) + (10 * b + a) = 99 :=
by
  sorry

end two_digit_sum_reverse_l180_180434


namespace smaller_circle_radius_l180_180099

theorem smaller_circle_radius :
  ∀ (R r : ℝ), R = 10 ∧ (4 * r = 2 * R) → r = 5 :=
by
  intro R r
  intro h
  have h1 : R = 10 := h.1
  have h2 : 4 * r = 2 * R := h.2
  -- Use the conditions to eventually show r = 5
  sorry

end smaller_circle_radius_l180_180099


namespace students_solved_only_B_l180_180118

variable (A B C : Prop)
variable (n x y b c d : ℕ)

-- Conditions given in the problem
axiom h1 : n = 25
axiom h2 : x + y + b + c + d = n
axiom h3 : b + d = 2 * (c + d)
axiom h4 : x = y + 1
axiom h5 : x + b + c = 2 * (b + c)

-- Theorem to be proved
theorem students_solved_only_B : b = 6 :=
by
  sorry

end students_solved_only_B_l180_180118


namespace ratio_of_projection_l180_180238

theorem ratio_of_projection (x y : ℝ)
  (h : ∀ (x y : ℝ), (∃ x y : ℝ, 
  (3/25 * x + 4/25 * y = x) ∧ (4/25 * x + 12/25 * y = y))) : x / y = 2 / 11 :=
sorry

end ratio_of_projection_l180_180238


namespace elaine_earnings_increase_l180_180428

variable (E : ℝ) -- Elaine's earnings last year
variable (P : ℝ) -- Percentage increase in earnings

-- Conditions
variable (rent_last_year : ℝ := 0.20 * E)
variable (earnings_this_year : ℝ := E * (1 + P / 100))
variable (rent_this_year : ℝ := 0.30 * earnings_this_year)
variable (multiplied_rent_last_year : ℝ := 1.875 * rent_last_year)

-- Theorem to be proven
theorem elaine_earnings_increase (h : rent_this_year = multiplied_rent_last_year) : P = 25 :=
by
  sorry

end elaine_earnings_increase_l180_180428


namespace triangle_perimeter_l180_180284

theorem triangle_perimeter (area : ℝ) (leg1 : ℝ) (leg2 : ℝ) (hypotenuse : ℝ) 
  (h1 : area = 150)
  (h2 : leg1 = 30)
  (h3 : 0 < leg2)
  (h4 : hypotenuse = (leg1^2 + leg2^2).sqrt)
  (hArea : area = 0.5 * leg1 * leg2)
  : hypotenuse = 10 * Real.sqrt 10 ∧ leg2 = 10 ∧ (leg1 + leg2 + hypotenuse = 40 + 10 * Real.sqrt 10) := 
by
  sorry

end triangle_perimeter_l180_180284


namespace max_green_socks_l180_180365

theorem max_green_socks (g y : ℕ) (h_t : g + y ≤ 2000) (h_prob : (g * (g - 1) + y * (y - 1) = (g + y) * (g + y - 1) / 3)) :
  g ≤ 19 := by
  sorry

end max_green_socks_l180_180365


namespace find_ABC_l180_180206

noncomputable def problem (A B C : ℕ) : Prop :=
  A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧
  A < 8 ∧ B < 8 ∧ C < 6 ∧
  (A * 8 + B + C = 8 * 2 + C) ∧
  (A * 8 + B + B * 8 + A = C * 8 + C) ∧
  (100 * A + 10 * B + C = 246)

theorem find_ABC : ∃ A B C : ℕ, problem A B C := sorry

end find_ABC_l180_180206


namespace original_number_l180_180095

theorem original_number (x : ℝ) (h1 : 268 * x = 19832) (h2 : 2.68 * x = 1.9832) : x = 74 :=
sorry

end original_number_l180_180095


namespace number_of_outcomes_for_champions_l180_180148

def num_events : ℕ := 3
def num_competitors : ℕ := 6
def total_possible_outcomes : ℕ := num_competitors ^ num_events

theorem number_of_outcomes_for_champions :
  total_possible_outcomes = 216 :=
by
  sorry

end number_of_outcomes_for_champions_l180_180148


namespace derivative_ln_div_x_l180_180108

noncomputable def f (x : ℝ) := (Real.log x) / x

theorem derivative_ln_div_x (x : ℝ) (h : x ≠ 0) : deriv f x = (1 - Real.log x) / (x^2) :=
by
  sorry

end derivative_ln_div_x_l180_180108


namespace cube_volume_in_cubic_yards_l180_180480

def volume_in_cubic_feet := 64
def cubic_feet_per_cubic_yard := 27

theorem cube_volume_in_cubic_yards : 
  volume_in_cubic_feet / cubic_feet_per_cubic_yard = 64 / 27 :=
by
  sorry

end cube_volume_in_cubic_yards_l180_180480


namespace average_interest_rate_l180_180614

theorem average_interest_rate 
  (total : ℝ)
  (rate1 rate2 yield1 yield2 : ℝ)
  (amount1 amount2 : ℝ)
  (h_total : total = amount1 + amount2)
  (h_rate1 : rate1 = 0.03)
  (h_rate2 : rate2 = 0.07)
  (h_yield_equal : yield1 = yield2)
  (h_yield1 : yield1 = rate1 * amount1)
  (h_yield2 : yield2 = rate2 * amount2) :
  (yield1 + yield2) / total = 0.042 :=
by
  sorry

end average_interest_rate_l180_180614


namespace monotonicity_f_inequality_proof_l180_180207

noncomputable def f (x : ℝ) : ℝ := Real.log x - x + 1

theorem monotonicity_f :
  (∀ x : ℝ, 0 < x ∧ x < 1 → f x < f (x + ε)) ∧ (∀ x : ℝ, 1 < x → f (x - ε) > f x) := 
sorry

theorem inequality_proof (x : ℝ) (hx : 1 < x) :
  1 < (x - 1) / Real.log x ∧ (x - 1) / Real.log x < x :=
sorry

end monotonicity_f_inequality_proof_l180_180207


namespace right_triangle_area_l180_180149

theorem right_triangle_area {a r R : ℝ} (hR : R = (5 / 2) * r) (h_leg : ∃ BC, BC = a) :
  (∃ area, area = (2 * a^2 / 3) ∨ area = (3 * a^2 / 8)) :=
sorry

end right_triangle_area_l180_180149


namespace min_value_collinear_l180_180768

theorem min_value_collinear (x y : ℝ) (h₁ : 2 * x + 3 * y = 3) (h₂ : 0 < x) (h₃ : 0 < y) : 
  (3 / x + 2 / y) = 8 :=
sorry

end min_value_collinear_l180_180768


namespace range_of_a_l180_180072

variable (a : ℝ)
def p : Prop := ∀ x : ℝ, 1 ≤ x ∧ x ≤ 12 → x^2 - a ≥ 0
def q : Prop := ∃ x₀ : ℝ, x₀^2 + (a - 1) * x₀ + 1 < 0

theorem range_of_a (hpq : p a ∨ q a) (hpnq : ¬p a ∧ ¬q a) : 
  (-1 ≤ a ∧ a ≤ 1) ∨ (a > 3) :=
sorry

end range_of_a_l180_180072


namespace exists_k_such_that_n_eq_k_2010_l180_180495

theorem exists_k_such_that_n_eq_k_2010 (m n : ℕ) (h_pos_m : 0 < m) (h_pos_n : 0 < n)
  (h : m * n ∣ m ^ 2010 + n ^ 2010 + n) : ∃ k : ℕ, 0 < k ∧ n = k ^ 2010 := by
  sorry

end exists_k_such_that_n_eq_k_2010_l180_180495


namespace expected_plain_zongzi_picked_l180_180933

-- Definitions and conditions:
def total_zongzi := 10
def red_bean_zongzi := 3
def meat_zongzi := 3
def plain_zongzi := 4

def choose (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Define the probabilities
def P_X_0 : ℚ := (choose 6 2 : ℚ) / choose 10 2
def P_X_1 : ℚ := (choose 6 1 * choose 4 1 : ℚ) / choose 10 2
def P_X_2 : ℚ := (choose 4 2 : ℚ) / choose 10 2

-- Expected value of X
def E_X : ℚ := 0 * P_X_0 + 1 * P_X_1 + 2 * P_X_2

theorem expected_plain_zongzi_picked : E_X = 4 / 5 := by
  -- Using the definition of E_X and the respective probabilities
  unfold E_X P_X_0 P_X_1 P_X_2
  -- Use the given formula to calculate the values
  -- Remaining steps would show detailed calculations leading to the answer
  sorry

end expected_plain_zongzi_picked_l180_180933


namespace puppy_sleep_duration_l180_180752

-- Definitions based on the given conditions
def connor_sleep_hours : ℕ := 6
def luke_sleep_hours : ℕ := connor_sleep_hours + 2
def puppy_sleep_hours : ℕ := 2 * luke_sleep_hours

-- Theorem stating the puppy's sleep duration
theorem puppy_sleep_duration : puppy_sleep_hours = 16 :=
by
  -- ( Proof goes here )
  sorry

end puppy_sleep_duration_l180_180752


namespace part1_part2_l180_180776

theorem part1 : 2 * (-1)^3 - (-2)^2 / 4 + 10 = 7 := by
  sorry

theorem part2 : abs (-3) - (-6 + 4) / (-1 / 2)^3 + (-1)^2013 = -14 := by
  sorry

end part1_part2_l180_180776


namespace probability_of_darkness_l180_180350

theorem probability_of_darkness (rev_per_min : ℕ) (stay_in_dark_time : ℕ) (revolution_time : ℕ) (stay_fraction : ℕ → ℚ) :
  rev_per_min = 2 →
  stay_in_dark_time = 10 →
  revolution_time = 60 / rev_per_min →
  stay_fraction stay_in_dark_time / revolution_time = 1 / 3 :=
by
  sorry

end probability_of_darkness_l180_180350


namespace cos_identity_example_l180_180961

theorem cos_identity_example (α : ℝ) (h : Real.sin (Real.pi / 6 + α) = 3 / 5) : Real.cos (Real.pi / 3 - α) = 3 / 5 := by
  sorry

end cos_identity_example_l180_180961


namespace find_x7_l180_180754

-- Definitions for the conditions
def seq (x : ℕ → ℕ) : Prop :=
  (x 6 = 144) ∧ ∀ n, (n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4) → (x (n + 3) = x (n + 2) * (x (n + 1) + x n))

-- Theorem statement to prove x_7 = 3456
theorem find_x7 (x : ℕ → ℕ) (h : seq x) : x 7 = 3456 := sorry

end find_x7_l180_180754


namespace skating_probability_given_skiing_l180_180737

theorem skating_probability_given_skiing (P_A P_B P_A_or_B : ℝ)
    (h1 : P_A = 0.6) (h2 : P_B = 0.5) (h3 : P_A_or_B = 0.7) : 
    (P_A_or_B = P_A + P_B - P_A * P_B) → 
    ((P_A * P_B) / P_B = 0.8) := 
    by
        intros
        sorry

end skating_probability_given_skiing_l180_180737


namespace penny_difference_l180_180122

variables (p : ℕ)

/-- Liam and Mia have certain numbers of fifty-cent coins. This theorem proves the difference 
    in their total value in pennies. 
-/
theorem penny_difference:
  (3 * p + 2) * 50 - (2 * p + 7) * 50 = 50 * p - 250 :=
by
  sorry

end penny_difference_l180_180122


namespace avg_weight_of_22_boys_l180_180877

theorem avg_weight_of_22_boys:
  let total_boys := 30
  let avg_weight_8 := 45.15
  let avg_weight_total := 48.89
  let total_weight_8 := 8 * avg_weight_8
  let total_weight_all := total_boys * avg_weight_total
  ∃ A : ℝ, A = 50.25 ∧ 22 * A + total_weight_8 = total_weight_all :=
by {
  sorry 
}

end avg_weight_of_22_boys_l180_180877


namespace cos_difference_simplification_l180_180120

theorem cos_difference_simplification :
  let x := Real.cos (20 * Real.pi / 180)
  let y := Real.cos (40 * Real.pi / 180)
  x - y = -1 / (2 * Real.sqrt 5) :=
sorry

end cos_difference_simplification_l180_180120


namespace question_l180_180574

-- Let x and y be real numbers.
variables (x y : ℝ)

-- Proposition A: x + y ≠ 8
def PropA : Prop := x + y ≠ 8

-- Proposition B: x ≠ 2 ∨ y ≠ 6
def PropB : Prop := x ≠ 2 ∨ y ≠ 6

-- We need to prove that PropA is a sufficient but not necessary condition for PropB.
theorem question : (PropA x y → PropB x y) ∧ ¬ (PropB x y → PropA x y) :=
sorry

end question_l180_180574


namespace max_ratio_of_mean_70_l180_180764

theorem max_ratio_of_mean_70 (x y : ℕ) (hx : 10 ≤ x ∧ x ≤ 99) (hy : 10 ≤ y ∧ y ≤ 99) (hmean : (x + y) / 2 = 70) : (x / y ≤ 99 / 41) :=
sorry

end max_ratio_of_mean_70_l180_180764


namespace triangle_inequality_difference_l180_180626

theorem triangle_inequality_difference :
  ∀ (x : ℕ), (x + 8 > 10) → (x + 10 > 8) → (8 + 10 > x) →
    (17 - 3 = 14) :=
by
  intros x hx1 hx2 hx3
  sorry

end triangle_inequality_difference_l180_180626


namespace find_x_l180_180331

theorem find_x (x : ℝ) : 0.5 * x + (0.3 * 0.2) = 0.26 ↔ x = 0.4 := by
  sorry

end find_x_l180_180331


namespace g_at_10_l180_180826

noncomputable def g : ℕ → ℝ := sorry

axiom g_zero : g 0 = 2
axiom g_one : g 1 = 1
axiom g_func_eq (m n : ℕ) (h : m ≥ n) : 
  g (m + n) + g (m - n) = (g (2 * m) + g (2 * n)) / 2 + 2

theorem g_at_10 : g 10 = 102 := sorry

end g_at_10_l180_180826


namespace greatest_possible_sum_of_visible_numbers_l180_180970

theorem greatest_possible_sum_of_visible_numbers :
  ∀ (numbers : ℕ → ℕ) (Cubes : Fin 4 → ℤ), 
  (numbers 0 = 1) → (numbers 1 = 3) → (numbers 2 = 9) → (numbers 3 = 27) → (numbers 4 = 81) → (numbers 5 = 243) →
  (Cubes 0 = (16 - 2) * (243 + 81 + 27 + 9 + 3)) → 
  (Cubes 1 = (16 - 2) * (243 + 81 + 27 + 9 + 3)) →
  (Cubes 2 = (16 - 2) * (243 + 81 + 27 + 9 + 3)) ->
  (Cubes 3 = 16 * (243 + 81 + 27 + 9 + 3)) ->
  (Cubes 0 + Cubes 1 + Cubes 2 + Cubes 3 = 1452) :=
by 
  sorry

end greatest_possible_sum_of_visible_numbers_l180_180970


namespace units_digit_2_pow_2015_l180_180496

theorem units_digit_2_pow_2015 : ∃ u : ℕ, (2 ^ 2015 % 10) = u ∧ u = 8 := 
by
  sorry

end units_digit_2_pow_2015_l180_180496


namespace cows_count_l180_180423

theorem cows_count (D C : ℕ) (h_legs : 2 * D + 4 * C = 2 * (D + C) + 24) : C = 12 := 
  sorry

end cows_count_l180_180423


namespace hourly_rate_is_7_l180_180783

-- Define the fixed fee, the total payment, and the number of hours
def fixed_fee : ℕ := 17
def total_payment : ℕ := 80
def num_hours : ℕ := 9

-- Define the function calculating the hourly rate based on the given conditions
def hourly_rate (fixed_fee total_payment num_hours : ℕ) : ℕ :=
  (total_payment - fixed_fee) / num_hours

-- Prove that the hourly rate is 7 dollars per hour
theorem hourly_rate_is_7 :
  hourly_rate fixed_fee total_payment num_hours = 7 := 
by 
  -- proof is skipped
  sorry

end hourly_rate_is_7_l180_180783


namespace range_of_m_l180_180979

theorem range_of_m (m : ℝ) :
  ¬(1^2 + 2*1 - m > 0) ∧ (2^2 + 2*2 - m > 0) ↔ (3 ≤ m ∧ m < 8) :=
by
  sorry

end range_of_m_l180_180979


namespace eggs_in_each_group_l180_180022

theorem eggs_in_each_group (eggs marbles groups : ℕ) 
  (h_eggs : eggs = 15)
  (h_groups : groups = 3) 
  (h_marbles : marbles = 4) :
  eggs / groups = 5 :=
by sorry

end eggs_in_each_group_l180_180022


namespace hare_height_l180_180191

theorem hare_height (camel_height_ft : ℕ) (hare_height_in_inches : ℕ) :
  (camel_height_ft = 28) ∧ (hare_height_in_inches * 24 = camel_height_ft * 12) → hare_height_in_inches = 14 :=
by
  sorry

end hare_height_l180_180191


namespace arccos_neg_one_l180_180322

theorem arccos_neg_one : Real.arccos (-1) = Real.pi := by
  sorry

end arccos_neg_one_l180_180322


namespace linear_function_quadrants_l180_180558

theorem linear_function_quadrants (k : ℝ) :
  (∀ x y : ℝ, y = (k + 1) * x + k - 2 → 
    ((x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0) ∨ (x > 0 ∧ y < 0))) ↔ (-1 < k ∧ k < 2) := 
sorry

end linear_function_quadrants_l180_180558


namespace arithmetic_sequence_sum_l180_180097

-- Define the arithmetic sequence and the given conditions
def arithmetic_seq (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Define the specific values for the sequence a_1 = 2 and a_2 + a_3 = 13
variables {a : ℕ → ℤ} (d : ℤ)
axiom h1 : a 1 = 2
axiom h2 : a 2 + a 3 = 13

-- Conclude the value of a_4 + a_5 + a_6
theorem arithmetic_sequence_sum : a 4 + a 5 + a 6 = 42 :=
sorry

end arithmetic_sequence_sum_l180_180097


namespace area_of_rectangular_park_l180_180993

theorem area_of_rectangular_park
  (l w : ℕ) 
  (h_perimeter : 2 * l + 2 * w = 80)
  (h_length : l = 3 * w) :
  l * w = 300 :=
sorry

end area_of_rectangular_park_l180_180993


namespace min_right_triangles_cover_equilateral_triangle_l180_180041

theorem min_right_triangles_cover_equilateral_triangle :
  let side_length_equilateral := 12
  let legs_right_triangle := 1
  let area_equilateral := (Real.sqrt 3 / 4) * side_length_equilateral ^ 2
  let area_right_triangle := (1 / 2) * legs_right_triangle * legs_right_triangle
  let triangles_needed := area_equilateral / area_right_triangle
  triangles_needed = 72 * Real.sqrt 3 := 
by 
  sorry

end min_right_triangles_cover_equilateral_triangle_l180_180041


namespace correct_option_l180_180324

theorem correct_option (x y a b : ℝ) :
  ((x + 2 * y) ^ 2 ≠ x ^ 2 + 4 * y ^ 2) ∧
  ((-2 * (a ^ 3)) ^ 2 = 4 * (a ^ 6)) ∧
  (-6 * (a ^ 2) * (b ^ 5) + a * b ^ 2 ≠ -6 * a * (b ^ 3)) ∧
  (2 * (a ^ 2) * 3 * (a ^ 3) ≠ 6 * (a ^ 6)) :=
by
  sorry

end correct_option_l180_180324


namespace triangle_perimeter_l180_180715

variable (y : ℝ)

theorem triangle_perimeter (h₁ : 2 * y > y) (h₂ : y > 0) :
  ∃ (P : ℝ), P = 2 * y + y * Real.sqrt 2 :=
sorry

end triangle_perimeter_l180_180715


namespace box_area_relation_l180_180578

theorem box_area_relation (a b c : ℕ) (h : a = b + c + 10) :
  (a * b) * (b * c) * (c * a) = (2 * (b + c) + 10)^2 := 
sorry

end box_area_relation_l180_180578


namespace paint_replacement_l180_180817

theorem paint_replacement :
  ∀ (original_paint new_paint : ℝ), 
  original_paint = 100 →
  new_paint = 0.10 * (original_paint - 0.5 * original_paint) + 0.20 * (0.5 * original_paint) →
  new_paint / original_paint = 0.15 :=
by
  intros original_paint new_paint h_orig h_new
  sorry

end paint_replacement_l180_180817


namespace three_Z_five_l180_180652

def Z (a b : ℤ) : ℤ := b + 7 * a - 3 * a^2

theorem three_Z_five : Z 3 5 = -1 := by
  sorry

end three_Z_five_l180_180652


namespace sum_of_missing_digits_l180_180732

-- Define the problem's conditions
def add_digits (a b c d e f g h : ℕ) := 
a + b = 18 ∧ b + c + d = 21

-- Prove the sum of the missing digits equals 7
theorem sum_of_missing_digits (a b c d e f g h : ℕ) (h1 : add_digits a b c d e f g h) : a + c = 7 := 
sorry

end sum_of_missing_digits_l180_180732


namespace radius_increase_l180_180906

theorem radius_increase (C₁ C₂ : ℝ) (C₁_eq : C₁ = 30) (C₂_eq : C₂ = 40) :
  let r₁ := C₁ / (2 * Real.pi)
  let r₂ := C₂ / (2 * Real.pi)
  r₂ - r₁ = 5 / Real.pi :=
by
  simp [C₁_eq, C₂_eq]
  sorry

end radius_increase_l180_180906


namespace smallest_natural_number_l180_180795

open Nat

theorem smallest_natural_number (n : ℕ) :
  (n + 1) % 4 = 0 ∧ (n + 1) % 6 = 0 ∧ (n + 1) % 10 = 0 ∧ (n + 1) % 12 = 0 →
  n = 59 :=
by
  sorry

end smallest_natural_number_l180_180795


namespace solution_l180_180125

noncomputable def problem (x : ℝ) : Prop :=
  0 < x ∧ (1/2 * (4 * x^2 - 1) = (x^2 - 50 * x - 20) * (x^2 + 25 * x + 10))

theorem solution (x : ℝ) (h : problem x) : x = 26 + Real.sqrt 677 :=
by
  sorry

end solution_l180_180125


namespace f_at_five_l180_180366

-- Define the function f with the property given in the condition
axiom f : ℝ → ℝ
axiom f_prop : ∀ x : ℝ, f (3 * x - 1) = x^2 + x + 1

-- Prove that f(5) = 7 given the properties above
theorem f_at_five : f 5 = 7 :=
by
  sorry

end f_at_five_l180_180366


namespace sin_pow_cos_pow_sum_l180_180727

namespace ProofProblem

-- Define the condition
def trig_condition (x : ℝ) : Prop :=
  3 * (Real.sin x)^3 + (Real.cos x)^3 = 3

-- State the theorem
theorem sin_pow_cos_pow_sum (x : ℝ) (h : trig_condition x) : Real.sin x ^ 2018 + Real.cos x ^ 2018 = 1 :=
by
  sorry

end ProofProblem

end sin_pow_cos_pow_sum_l180_180727


namespace train_crosses_pole_in_l180_180947

noncomputable def train_crossing_time (length : ℝ) (speed_km_hr : ℝ) : ℝ :=
  let speed_m_s := speed_km_hr * (5.0 / 18.0)
  length / speed_m_s

theorem train_crosses_pole_in : train_crossing_time 175 180 = 3.5 :=
by
  -- Proof would be here, but for now, it is omitted.
  sorry

end train_crosses_pole_in_l180_180947


namespace quadratic_solution_difference_l180_180610

theorem quadratic_solution_difference (x : ℝ) :
  ∀ x : ℝ, (x^2 - 5*x + 15 = x + 55) → (∃ a b : ℝ, a ≠ b ∧ x^2 - 6*x - 40 = 0 ∧ abs (a - b) = 14) :=
by
  sorry

end quadratic_solution_difference_l180_180610


namespace probability_of_b_l180_180160

noncomputable def P : ℕ → ℝ := sorry

axiom P_a : P 0 = 0.15
axiom P_a_and_b : P 1 = 0.15
axiom P_neither_a_nor_b : P 2 = 0.6

theorem probability_of_b : P 3 = 0.4 := 
by
  sorry

end probability_of_b_l180_180160


namespace probability_of_selecting_at_least_one_female_l180_180675

open BigOperators

noncomputable def prob_at_least_one_female_selected : ℚ :=
  let total_choices := Nat.choose 10 3
  let all_males_choices := Nat.choose 6 3
  1 - (all_males_choices / total_choices : ℚ)

theorem probability_of_selecting_at_least_one_female :
  prob_at_least_one_female_selected = 5 / 6 := by
  sorry

end probability_of_selecting_at_least_one_female_l180_180675


namespace soda_cost_132_cents_l180_180552

theorem soda_cost_132_cents
  (b s : ℕ)
  (h1 : 3 * b + 2 * s + 30 = 510)
  (h2 : 2 * b + 3 * s = 540) 
  : s = 132 :=
by
  sorry

end soda_cost_132_cents_l180_180552


namespace solution_set_of_inequality_l180_180975

theorem solution_set_of_inequality (x : ℝ) : 
  (1 / x ≤ 1 ↔ (0 < x ∧ x < 1) ∨ (1 ≤ x)) :=
  sorry

end solution_set_of_inequality_l180_180975


namespace problem_lean_l180_180229

theorem problem_lean (x y : ℝ) (h₁ : (|x + 2| ≥ 0) ∧ (|y - 4| ≥ 0)) : 
  (|x + 2| = 0 ∧ |y - 4| = 0) → x + y - 3 = -1 :=
by sorry

end problem_lean_l180_180229


namespace bathroom_width_l180_180134

def length : ℝ := 4
def area : ℝ := 8
def width : ℝ := 2

theorem bathroom_width :
  area = length * width :=
by
  sorry

end bathroom_width_l180_180134


namespace find_product_of_offsets_l180_180345

theorem find_product_of_offsets
  (a b c : ℝ)
  (h1 : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h2 : a * b + a + b = 99)
  (h3 : b * c + b + c = 99)
  (h4 : c * a + c + a = 99) :
  (a + 1) * (b + 1) * (c + 1) = 1000 := by
  sorry

end find_product_of_offsets_l180_180345


namespace Seokjin_paper_count_l180_180872

theorem Seokjin_paper_count (Jimin_paper : ℕ) (h1 : Jimin_paper = 41) (h2 : ∀ x : ℕ, Seokjin_paper = Jimin_paper - 1) : Seokjin_paper = 40 :=
by {
  sorry
}

end Seokjin_paper_count_l180_180872


namespace positive_divisors_of_x_l180_180852

theorem positive_divisors_of_x (x : ℕ) (h : ∀ d : ℕ, d ∣ x^3 → d = 1 ∨ d = x^3 ∨ d ∣ x^2) : (∀ d : ℕ, d ∣ x → d = 1 ∨ d = x ∨ d ∣ p) :=
by
  sorry

end positive_divisors_of_x_l180_180852


namespace find_annual_interest_rate_l180_180736

theorem find_annual_interest_rate (P0 P1 P2 : ℝ) (r1 r : ℝ) :
  P0 = 12000 →
  r1 = 10 →
  P1 = P0 * (1 + (r1 / 100) / 2) →
  P1 = 12600 →
  P2 = 13260 →
  P1 * (1 + (r / 200)) = P2 →
  r = 10.476 :=
by
  intros hP0 hr1 hP1 hP1val hP2 hP1P2
  sorry

end find_annual_interest_rate_l180_180736


namespace f_of_x_l180_180972

variable (f : ℝ → ℝ)

theorem f_of_x (x : ℝ) (h : f (x - 1 / x) = x^2 + 1 / x^2) : f x = x^2 + 2 :=
sorry

end f_of_x_l180_180972


namespace f_pos_for_all_x_g_le_ax_plus_1_for_a_eq_1_l180_180745

noncomputable def f (x : ℝ) : ℝ := Real.exp x - (x + 1)^2 / 2
noncomputable def g (x : ℝ) : ℝ := 2 * Real.log (x + 1) + Real.exp (-x)

theorem f_pos_for_all_x (x : ℝ) (hx : x > -1) : f x > 0 := by
  sorry

theorem g_le_ax_plus_1_for_a_eq_1 (a : ℝ) (ha : a > 0) : (∀ x : ℝ, -1 < x → g x ≤ a * x + 1) ↔ a = 1 := by
  sorry

end f_pos_for_all_x_g_le_ax_plus_1_for_a_eq_1_l180_180745


namespace initial_tests_count_l180_180152

theorem initial_tests_count (n S : ℕ)
  (h1 : S = 35 * n)
  (h2 : (S - 20) / (n - 1) = 40) :
  n = 4 := 
sorry

end initial_tests_count_l180_180152


namespace even_n_square_mod_8_odd_n_square_mod_8_odd_n_fourth_mod_8_l180_180333

open Int
open Nat

theorem even_n_square_mod_8 (n : ℤ) (h : n % 2 = 0) : (n^2 % 8 = 0) ∨ (n^2 % 8 = 4) := sorry

theorem odd_n_square_mod_8 (n : ℤ) (h : n % 2 = 1) : n^2 % 8 = 1 := sorry

theorem odd_n_fourth_mod_8 (n : ℤ) (h : n % 2 = 1) : n^4 % 8 = 1 := sorry

end even_n_square_mod_8_odd_n_square_mod_8_odd_n_fourth_mod_8_l180_180333


namespace part1_part2_l180_180230

noncomputable def f (a x : ℝ) : ℝ := a - 1/x - Real.log x

theorem part1 (a : ℝ) :
  a = 2 → ∃ m b : ℝ, (∀ x : ℝ, f a x = x * m + b) ∧ (∀ y : ℝ, f a 1 = y → b = y ∧ m = 0) :=
by
  sorry

theorem part2 (a : ℝ) :
  (∃! x : ℝ, f a x = 0) → a = 1 :=
by
  sorry

end part1_part2_l180_180230


namespace total_books_of_gwen_l180_180033

theorem total_books_of_gwen 
  (mystery_shelves : ℕ) (picture_shelves : ℕ) (books_per_shelf : ℕ)
  (h1 : mystery_shelves = 3) (h2 : picture_shelves = 5) (h3 : books_per_shelf = 9) : 
  mystery_shelves * books_per_shelf + picture_shelves * books_per_shelf = 72 :=
by
  -- Given:
  -- 1. Gwen had 3 shelves of mystery books.
  -- 2. Each shelf had 9 books.
  -- 3. Gwen had 5 shelves of picture books.
  -- 4. Each shelf had 9 books.
  -- Prove:
  -- The total number of books Gwen had is 72.
  sorry

end total_books_of_gwen_l180_180033


namespace jenny_house_value_l180_180753

/-- Jenny's property tax rate is 2% -/
def property_tax_rate : ℝ := 0.02

/-- Her house's value increases by 25% due to the new high-speed rail project -/
noncomputable def house_value_increase_rate : ℝ := 0.25

/-- Jenny can afford to spend $15,000/year on property tax -/
def max_affordable_tax : ℝ := 15000

/-- Jenny can make improvements worth $250,000 to her house -/
def improvement_value : ℝ := 250000

/-- Current worth of Jenny's house -/
noncomputable def current_house_worth : ℝ := 500000

theorem jenny_house_value :
  property_tax_rate * (current_house_worth + improvement_value) = max_affordable_tax :=
by
  sorry

end jenny_house_value_l180_180753


namespace sphere_triangle_distance_l180_180455

theorem sphere_triangle_distance
  (P X Y Z : Type)
  (radius : ℝ)
  (h1 : radius = 15)
  (dist_XY : ℝ)
  (h2 : dist_XY = 6)
  (dist_YZ : ℝ)
  (h3 : dist_YZ = 8)
  (dist_ZX : ℝ)
  (h4 : dist_ZX = 10)
  (distance_from_P_to_triangle : ℝ)
  (h5 : distance_from_P_to_triangle = 10 * Real.sqrt 2) :
  let a := 10
  let b := 2
  let c := 1
  let result := a + b + c
  result = 13 :=
by
  sorry

end sphere_triangle_distance_l180_180455


namespace maximum_value_of_a_l180_180978

theorem maximum_value_of_a (a b c d : ℕ) (h1 : a < 3 * b) (h2 : b < 4 * c) (h3 : c < 5 * d) (h4 : d < 50) (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d) : a ≤ 2924 := 
sorry

end maximum_value_of_a_l180_180978


namespace simplify_expression_l180_180059

variable (x : ℝ)

theorem simplify_expression :
  2 * x * (4 * x^2 - 3 * x + 1) - 4 * (2 * x^2 - 3 * x + 5) =
  8 * x^3 - 14 * x^2 + 14 * x - 20 := 
  sorry

end simplify_expression_l180_180059


namespace extremum_value_l180_180526

noncomputable def f (x a b : ℝ) : ℝ := x^3 + 3 * a * x^2 + b * x + a^2

theorem extremum_value (a b : ℝ) (h1 : (3 - 6 * a + b = 0)) (h2 : (-1 + 3 * a - b + a^2 = 0)) :
  a - b = -7 :=
by
  sorry

end extremum_value_l180_180526


namespace modulus_of_z_l180_180060

-- Definitions of the problem conditions
def z := Complex.mk 1 (-1)

-- Statement of the math proof problem
theorem modulus_of_z : Complex.abs z = Real.sqrt 2 := by
  sorry -- Proof placeholder

end modulus_of_z_l180_180060


namespace initial_ducks_count_l180_180234

theorem initial_ducks_count (D : ℕ) 
  (h1 : ∃ (G : ℕ), G = 2 * D - 10) 
  (h2 : ∃ (D_new : ℕ), D_new = D + 4) 
  (h3 : ∃ (G_new : ℕ), G_new = 2 * D - 20) 
  (h4 : ∀ (D_new G_new : ℕ), G_new = D_new + 1) : 
  D = 25 := by
  sorry

end initial_ducks_count_l180_180234


namespace max_truthful_students_l180_180085

def count_students (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

theorem max_truthful_students : count_students 2015 = 2031120 :=
by sorry

end max_truthful_students_l180_180085


namespace initial_blue_balls_proof_l180_180772

-- Define the main problem parameters and condition
def initial_jars (total_balls initial_blue_balls removed_blue probability remaining_balls : ℕ) :=
  total_balls = 18 ∧
  removed_blue = 3 ∧
  remaining_balls = total_balls - removed_blue ∧
  probability = 1/5 → 
  (initial_blue_balls - removed_blue) / remaining_balls = probability

-- Define the proof problem
theorem initial_blue_balls_proof (total_balls initial_blue_balls removed_blue probability remaining_balls : ℕ) :
  initial_jars total_balls initial_blue_balls removed_blue probability remaining_balls →
  initial_blue_balls = 6 :=
by
  sorry

end initial_blue_balls_proof_l180_180772


namespace part1_part2_l180_180529

noncomputable def f (x a b : ℝ) : ℝ := |x - a| + |x + b|

theorem part1 (a b : ℝ) (h₀ : a = 1) (h₁ : b = 2) :
  {x : ℝ | f x a b ≤ 5} = {x : ℝ | -3 ≤ x ∧ x ≤ 2} :=
by
  sorry

theorem part2 (a b : ℝ) (h_min_value : ∀ x : ℝ, f x a b ≥ 3) :
  a + b = 3 → (a > 0 ∧ b > 0) →
  (∃ a b : ℝ, a = b ∧ a + b = 3 ∧ (a = b → f x a b = 3)) →
  (∀ a b : ℝ, (a^2/b + b^2/a) ≥ 3) :=
by
  sorry

end part1_part2_l180_180529


namespace odd_prime_does_not_divide_odd_nat_number_increment_l180_180341

theorem odd_prime_does_not_divide_odd_nat_number_increment (p n : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) (hn_odd : n % 2 = 1) :
  ¬ (p * n + 1 ∣ p ^ p - 1) :=
by
  sorry

end odd_prime_does_not_divide_odd_nat_number_increment_l180_180341


namespace avg_displacement_per_man_l180_180765

-- Problem definition as per the given conditions
def num_men : ℕ := 50
def tank_length : ℝ := 40  -- 40 meters
def tank_width : ℝ := 20   -- 20 meters
def rise_in_water_level : ℝ := 0.25  -- 25 cm -> 0.25 meters

-- Given the conditions, we need to prove the average displacement per man
theorem avg_displacement_per_man :
  (tank_length * tank_width * rise_in_water_level) / num_men = 4 := by
  sorry

end avg_displacement_per_man_l180_180765


namespace two_students_exist_l180_180282

theorem two_students_exist (scores : Fin 49 → Fin 8 × Fin 8 × Fin 8) :
  ∃ (i j : Fin 49), i ≠ j ∧ (scores i).1 ≥ (scores j).1 ∧ (scores i).2.1 ≥ (scores j).2.1 ∧ (scores i).2.2 ≥ (scores j).2.2 := 
by
  sorry

end two_students_exist_l180_180282


namespace find_largest_natural_number_l180_180411

theorem find_largest_natural_number :
  ∃ n : ℕ, (forall m : ℕ, m > n -> m ^ 300 ≥ 3 ^ 500) ∧ n ^ 300 < 3 ^ 500 :=
  sorry

end find_largest_natural_number_l180_180411


namespace negation_of_all_students_are_punctual_l180_180824

variable (Student : Type)
variable (student : Student → Prop)
variable (punctual : Student → Prop)

theorem negation_of_all_students_are_punctual :
  ¬ (∀ x, student x → punctual x) ↔ (∃ x, student x ∧ ¬ punctual x) := by
  sorry

end negation_of_all_students_are_punctual_l180_180824


namespace expenditure_record_l180_180124

/-- Lean function to represent the condition and the proof problem -/
theorem expenditure_record (income expenditure : Int) (h_income : income = 500) (h_recorded_income : income = 500) (h_expenditure : expenditure = 200) : expenditure = -200 := 
by
  sorry

end expenditure_record_l180_180124


namespace gain_percentage_of_watch_l180_180249

theorem gain_percentage_of_watch :
  let CP := 1076.923076923077
  let S1 := CP * 0.90
  let S2 := S1 + 140
  let gain_percentage := ((S2 - CP) / CP) * 100
  gain_percentage = 3 := by
  sorry

end gain_percentage_of_watch_l180_180249


namespace ratatouille_cost_per_quart_l180_180453

theorem ratatouille_cost_per_quart:
  let eggplant_weight := 5.5
  let eggplant_price := 2.20
  let zucchini_weight := 3.8
  let zucchini_price := 1.85
  let tomatoes_weight := 4.6
  let tomatoes_price := 3.75
  let onions_weight := 2.7
  let onions_price := 1.10
  let basil_weight := 1.0
  let basil_price_per_quarter := 2.70
  let bell_peppers_weight := 0.75
  let bell_peppers_price := 3.15
  let yield_quarts := 4.5
  let eggplant_cost := eggplant_weight * eggplant_price
  let zucchini_cost := zucchini_weight * zucchini_price
  let tomatoes_cost := tomatoes_weight * tomatoes_price
  let onions_cost := onions_weight * onions_price
  let basil_cost := basil_weight * (basil_price_per_quarter * 4)
  let bell_peppers_cost := bell_peppers_weight * bell_peppers_price
  let total_cost := eggplant_cost + zucchini_cost + tomatoes_cost + onions_cost + basil_cost + bell_peppers_cost
  let cost_per_quart := total_cost / yield_quarts
  cost_per_quart = 11.67 :=
by
  sorry

end ratatouille_cost_per_quart_l180_180453


namespace jon_buys_2_coffees_each_day_l180_180777

-- Define the conditions
def cost_per_coffee : ℕ := 2
def total_spent : ℕ := 120
def days_in_april : ℕ := 30

-- Define the total number of coffees bought
def total_coffees_bought : ℕ := total_spent / cost_per_coffee

-- Prove that Jon buys 2 coffees each day
theorem jon_buys_2_coffees_each_day : total_coffees_bought / days_in_april = 2 := by
  sorry

end jon_buys_2_coffees_each_day_l180_180777


namespace average_sqft_per_person_texas_l180_180669

theorem average_sqft_per_person_texas :
  let population := 17000000
  let area_sqmiles := 268596
  let usable_land_percentage := 0.8
  let sqfeet_per_sqmile := 5280 * 5280
  let total_sqfeet := area_sqmiles * sqfeet_per_sqmile
  let usable_sqfeet := usable_land_percentage * total_sqfeet
  let avg_sqfeet_per_person := usable_sqfeet / population
  352331 <= avg_sqfeet_per_person ∧ avg_sqfeet_per_person < 500000 :=
by
  sorry

end average_sqft_per_person_texas_l180_180669


namespace find_value_l180_180052

theorem find_value (x : ℝ) (h : Real.cos x - 3 * Real.sin x = 2) : 2 * Real.sin x + 3 * Real.cos x = -7 / 3 := 
sorry

end find_value_l180_180052


namespace union_of_A_and_B_l180_180839

open Set

def A : Set ℝ := {x | x > 2}
def B : Set ℝ := {x | -1 < x ∧ x < 4}

theorem union_of_A_and_B : A ∪ B = {x | x > -1} :=
by sorry

end union_of_A_and_B_l180_180839


namespace no_sum_of_consecutive_integers_to_420_l180_180216

noncomputable def perfect_square (n : ℕ) : Prop :=
∃ k : ℕ, k * k = n

def sum_sequence (n a : ℕ) : ℕ :=
n * a + n * (n - 1) / 2

theorem no_sum_of_consecutive_integers_to_420 
  (h1 : 420 > 0)
  (h2 : ∀ (n a : ℕ), n ≥ 2 → sum_sequence n a = 420 → perfect_square a)
  (h3 : ∃ n a, n ≥ 2 ∧ sum_sequence n a = 420 ∧ perfect_square a) :
  false :=
by
  sorry

end no_sum_of_consecutive_integers_to_420_l180_180216


namespace total_value_after_3_years_l180_180561

noncomputable def value_after_years (initial_value : ℝ) (depreciation_rate : ℝ) (years : ℕ) : ℝ :=
  initial_value * (1 - depreciation_rate) ^ years

def machine1_initial_value : ℝ := 2500
def machine1_depreciation_rate : ℝ := 0.05
def machine2_initial_value : ℝ := 3500
def machine2_depreciation_rate : ℝ := 0.07
def machine3_initial_value : ℝ := 4500
def machine3_depreciation_rate : ℝ := 0.04
def years : ℕ := 3

theorem total_value_after_3_years :
  value_after_years machine1_initial_value machine1_depreciation_rate years +
  value_after_years machine2_initial_value machine2_depreciation_rate years +
  value_after_years machine3_initial_value machine3_depreciation_rate years = 8940 :=
by
  sorry

end total_value_after_3_years_l180_180561


namespace binom_mult_eq_6720_l180_180539

theorem binom_mult_eq_6720 :
  (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 :=
by
  sorry

end binom_mult_eq_6720_l180_180539


namespace triangle_right_angled_solve_system_quadratic_roots_real_l180_180710

-- Problem 1
theorem triangle_right_angled (a b c : ℝ) (h : a^2 + b^2 + c^2 - 6 * a - 8 * b - 10 * c + 50 = 0) :
  (a = 3) ∧ (b = 4) ∧ (c = 5) ∧ (a^2 + b^2 = c^2) :=
sorry

-- Problem 2
theorem solve_system (x y : ℝ) (h1 : 3 * x + 4 * y = 30) (h2 : 5 * x + 3 * y = 28) :
  (x = 2) ∧ (y = 6) :=
sorry

-- Problem 3
theorem quadratic_roots_real (m : ℝ) :
  (∃ x : ℝ, ∃ y : ℝ, 3 * x^2 + 4 * x + m = 0 ∧ 3 * y^2 + 4 * y + m = 0) ↔ (m ≤ 4 / 3) :=
sorry

end triangle_right_angled_solve_system_quadratic_roots_real_l180_180710


namespace bucket_capacity_l180_180589

theorem bucket_capacity (jack_buckets_per_trip : ℕ)
                        (jill_buckets_per_trip : ℕ)
                        (jack_trip_ratio : ℝ)
                        (jill_trips : ℕ)
                        (tank_capacity : ℝ)
                        (bucket_capacity : ℝ)
                        (h1 : jack_buckets_per_trip = 2)
                        (h2 : jill_buckets_per_trip = 1)
                        (h3 : jack_trip_ratio = 3 / 2)
                        (h4 : jill_trips = 30)
                        (h5 : tank_capacity = 600) :
  bucket_capacity = 5 :=
by 
  sorry

end bucket_capacity_l180_180589


namespace sally_fries_count_l180_180730

theorem sally_fries_count (sally_initial_fries mark_initial_fries : ℕ) 
  (mark_gave_fraction : ℤ) 
  (h_sally_initial : sally_initial_fries = 14) 
  (h_mark_initial : mark_initial_fries = 36) 
  (h_mark_give : mark_gave_fraction = 1 / 3) :
  sally_initial_fries + (mark_initial_fries * mark_gave_fraction).natAbs = 26 :=
by
  sorry

end sally_fries_count_l180_180730


namespace intersection_correct_l180_180515

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {0, 2, 3, 4}

theorem intersection_correct : A ∩ B = {2, 3} := sorry

end intersection_correct_l180_180515


namespace perimeter_ABFCDE_l180_180545

theorem perimeter_ABFCDE 
  (ABCD_perimeter : ℝ)
  (ABCD : ℝ)
  (triangle_BFC : ℝ -> ℝ)
  (translate_BFC : ℝ -> ℝ)
  (ABFCDE : ℝ -> ℝ -> ℝ)
  (h1 : ABCD_perimeter = 40)
  (h2 : ABCD = ABCD_perimeter / 4)
  (h3 : triangle_BFC ABCD = 10 * Real.sqrt 2)
  (h4 : translate_BFC (10 * Real.sqrt 2) = 10 * Real.sqrt 2)
  (h5 : ABFCDE ABCD (10 * Real.sqrt 2) = 40 + 20 * Real.sqrt 2)
  : ABFCDE ABCD (10 * Real.sqrt 2) = 40 + 20 * Real.sqrt 2 := 
by 
  sorry

end perimeter_ABFCDE_l180_180545


namespace find_a_b_and_tangent_lines_l180_180819

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^3 + a * x^2 + b * x + 1

theorem find_a_b_and_tangent_lines (a b : ℝ) :
  (3 * (-2 / 3)^2 + 2 * a * (-2 / 3) + b = 0) ∧
  (3 * 1^2 + 2 * a * 1 + b = 0) →
  a = -1 / 2 ∧ b = -2 ∧
  (∀ t : ℝ, f t a b = (t^3 + (a - 1 / 2) * t^2 - 2 * t + 1) → 
     (f t a b - (3 * t^2 - t - 2) * (0 - t) = 1) →
       (3 * t^2 - t - 2 = (t * (3 * (t - t))) ) → 
          ((2 * 0 + f 0 a b) = 1) ∨ (33 * 0 + 16 * 1 - 16 = 1)) :=
sorry

end find_a_b_and_tangent_lines_l180_180819


namespace diving_competition_scores_l180_180050

theorem diving_competition_scores (A B C D E : ℝ) (hA : 1 ≤ A ∧ A ≤ 10)
  (hB : 1 ≤ B ∧ B ≤ 10) (hC : 1 ≤ C ∧ C ≤ 10) (hD : 1 ≤ D ∧ D ≤ 10) 
  (hE : 1 ≤ E ∧ E ≤ 10) (degree_of_difficulty : ℝ) (h_diff : degree_of_difficulty = 3.2)
  (point_value : ℝ) (h_point_value : point_value = 79.36) :
  A = max A (max B (max C (max D E))) →
  E = min A (min B (min C (min D E))) →
  (B + C + D) = (point_value / degree_of_difficulty) :=
by sorry

end diving_competition_scores_l180_180050


namespace inequality_satisfaction_l180_180454

theorem inequality_satisfaction (k n : ℕ) (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (1 + y^n / x^k) ≥ ((1 + y)^n / (1 + x)^k) ↔ 
    (k = 0) ∨ (n = 0) ∨ (0 = k ∧ 0 = n) ∨ (k ≥ n - 1 ∧ n ≥ 1) :=
by sorry

end inequality_satisfaction_l180_180454


namespace ball_placement_count_l180_180248

-- Definitions for the balls and their numbering
inductive Ball
| b1
| b2
| b3
| b4

-- Definitions for the boxes and their numbering
inductive Box
| box1
| box2
| box3

-- Function that checks if an assignment is valid given the conditions.
def isValidAssignment (assignment : Ball → Box) : Prop :=
  assignment Ball.b1 ≠ Box.box1 ∧ assignment Ball.b3 ≠ Box.box3

-- Main statement to prove
theorem ball_placement_count : 
  ∃ (assignments : Finset (Ball → Box)), 
    (∀ f ∈ assignments, isValidAssignment f) ∧ assignments.card = 14 := 
sorry

end ball_placement_count_l180_180248


namespace meal_cost_l180_180609

/-- 
    Define the cost of a meal consisting of one sandwich, one cup of coffee, and one piece of pie 
    given the costs of two different meals.
-/
theorem meal_cost (s c p : ℝ) (h1 : 2 * s + 5 * c + p = 5) (h2 : 3 * s + 8 * c + p = 7) :
    s + c + p = 3 :=
by
  sorry

end meal_cost_l180_180609


namespace prob_two_sunny_days_l180_180015

-- Define the probability of rain and sunny
def probRain : ℚ := 3 / 4
def probSunny : ℚ := 1 / 4

-- Define the problem statement
theorem prob_two_sunny_days : (10 * (probSunny^2) * (probRain^3)) = 135 / 512 := 
by
  sorry

end prob_two_sunny_days_l180_180015


namespace first_machine_rate_l180_180262

theorem first_machine_rate (x : ℝ) (h : (x + 55) * 30 = 2400) : x = 25 :=
by
  sorry

end first_machine_rate_l180_180262


namespace tangent_line_through_origin_max_value_on_interval_min_value_on_interval_l180_180659
noncomputable def f (x : ℝ) : ℝ := x^2 / Real.exp x

theorem tangent_line_through_origin (x y : ℝ) :
  (∃ a : ℝ, (x, y) = (a, f a) ∧ (0, 0) = (0, 0) ∧ y - f a = ((2 * a - a^2) / Real.exp a) * (x - a)) →
  y = x / Real.exp 1 :=
sorry

theorem max_value_on_interval : ∃ (x : ℝ), x = 9 / Real.exp 3 :=
  sorry

theorem min_value_on_interval : ∃ (x : ℝ), x = 0 :=
  sorry

end tangent_line_through_origin_max_value_on_interval_min_value_on_interval_l180_180659


namespace square_side_length_l180_180818

theorem square_side_length (A : ℝ) (π : ℝ) (s : ℝ) (area_circle_eq : A = 100)
  (area_circle_eq_perimeter_square : A = 4 * s) : s = 25 := by
  sorry

end square_side_length_l180_180818


namespace exponential_monotone_l180_180580

theorem exponential_monotone {a b : ℝ} (ha : a > 0) (hb : b > 0) (h : Real.exp a + 2 * a = Real.exp b + 3 * b) : a < b :=
sorry

end exponential_monotone_l180_180580


namespace automobile_travel_distance_5_minutes_l180_180935

variable (a r : ℝ)

theorem automobile_travel_distance_5_minutes (h0 : r ≠ 0) :
  let distance_in_feet := (2 * a) / 5
  let time_in_seconds := 300
  (distance_in_feet / r) * time_in_seconds / 3 = 40 * a / r :=
by
  sorry

end automobile_travel_distance_5_minutes_l180_180935


namespace angle_AFE_is_80_degrees_l180_180208

-- Defining the setup and given conditions
def point := ℝ × ℝ  -- defining a 2D point
noncomputable def A : point := (0, 0)
noncomputable def B : point := (1, 0)
noncomputable def C : point := (1, 1)
noncomputable def D : point := (0, 1)
noncomputable def E : point := (-1, 1.732)  -- Place E such that angle CDE ≈ 130 degrees

-- Conditions
def angle_CDE := 130
def DF_over_DE := 2  -- DF = 2 * DE
noncomputable def F : point := (0.5, 1)  -- This is an example position; real positioning depends on more details

-- Proving that the angle AFE is 80 degrees
theorem angle_AFE_is_80_degrees :
  ∃ (AFE : ℝ), AFE = 80 := sorry

end angle_AFE_is_80_degrees_l180_180208


namespace total_cost_is_63_l180_180602

-- Define the original price, markdown percentage, and sales tax percentage
def original_price : ℝ := 120
def markdown_percentage : ℝ := 0.50
def sales_tax_percentage : ℝ := 0.05

-- Calculate the reduced price
def reduced_price : ℝ := original_price * (1 - markdown_percentage)

-- Calculate the sales tax on the reduced price
def sales_tax : ℝ := reduced_price * sales_tax_percentage

-- Calculate the total cost
noncomputable def total_cost : ℝ := reduced_price + sales_tax

-- Theorem stating that the total cost of the aquarium is $63
theorem total_cost_is_63 : total_cost = 63 := by
  sorry

end total_cost_is_63_l180_180602


namespace cryptarithm_solution_l180_180987

theorem cryptarithm_solution (A B C D : ℕ) (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (h_adjacent : A = C + 1 ∨ A = C - 1)
  (h_diff : B = D + 2 ∨ B = D - 2) :
  1000 * A + 100 * B + 10 * C + D = 5240 :=
sorry

end cryptarithm_solution_l180_180987


namespace range_of_k_l180_180553

variables (k : ℝ)

def vector_a (k : ℝ) : ℝ × ℝ := (-k, 4)
def vector_b (k : ℝ) : ℝ × ℝ := (k, k + 3)

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem range_of_k (h : 0 < dot_product (vector_a k) (vector_b k)) : 
  -2 < k ∧ k < 0 ∨ 0 < k ∧ k < 6 :=
sorry

end range_of_k_l180_180553


namespace primes_x_y_eq_l180_180245

theorem primes_x_y_eq 
  {p q x y : ℕ} (hp : Nat.Prime p) (hq : Nat.Prime q)
  (hx : 0 < x) (hy : 0 < y)
  (hp_lt_x : x < p) (hq_lt_y : y < q)
  (h : (p : ℚ) / x + (q : ℚ) / y = (p * y + q * x) / (x * y)) :
  x = y :=
sorry

end primes_x_y_eq_l180_180245


namespace scientific_notation_of_400000_l180_180919

theorem scientific_notation_of_400000 :
  (400000: ℝ) = 4 * 10^5 :=
by 
  sorry

end scientific_notation_of_400000_l180_180919


namespace smallest_possible_sum_l180_180687

theorem smallest_possible_sum (A B C D : ℤ) 
  (h1 : A + B = 2 * C)
  (h2 : B * D = C * C)
  (h3 : 3 * C = 7 * B)
  (h4 : 0 < A ∧ 0 < B ∧ 0 < C ∧ 0 < D) : 
  A + B + C + D = 76 :=
sorry

end smallest_possible_sum_l180_180687


namespace group_size_l180_180054

noncomputable def total_cost : ℤ := 13500
noncomputable def cost_per_person : ℤ := 900

theorem group_size : total_cost / cost_per_person = 15 :=
by {
  sorry
}

end group_size_l180_180054


namespace product_of_millions_l180_180278

-- Define the conditions
def a := 5 * (10 : ℝ) ^ 6
def b := 8 * (10 : ℝ) ^ 6

-- State the proof problem
theorem product_of_millions : (a * b) = 40 * (10 : ℝ) ^ 12 := 
by
  sorry

end product_of_millions_l180_180278


namespace intersection_A_B_l180_180287

noncomputable def A : Set ℝ := {x | 2 * x^2 - 3 * x - 2 ≤ 0}
noncomputable def B : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_A_B :
  A ∩ B = {0, 1, 2} := by
  sorry

end intersection_A_B_l180_180287


namespace smallest_n_mod5_l180_180996

theorem smallest_n_mod5 :
  ∃ n : ℕ, n > 0 ∧ 6^n % 5 = n^6 % 5 ∧ ∀ m : ℕ, m > 0 ∧ 6^m % 5 = m^6 % 5 → n ≤ m :=
by
  sorry

end smallest_n_mod5_l180_180996


namespace cross_fraction_eq1_cross_fraction_eq2_cross_fraction_eq3_l180_180662

-- Problem 1
theorem cross_fraction_eq1 (x : ℝ) : (x + 12 / x = -7) → 
  ∃ (x₁ x₂ : ℝ), (x₁ = -3 ∧ x₂ = -4 ∧ x = x₁ ∨ x = x₂) :=
sorry

-- Problem 2
theorem cross_fraction_eq2 (a b : ℝ) 
    (h1 : a * b = -6) 
    (h2 : a + b = -5) : (a ≠ 0 ∧ b ≠ 0) →
    (b / a + a / b + 1 = -31 / 6) :=
sorry

-- Problem 3
theorem cross_fraction_eq3 (k x₁ x₂ : ℝ)
    (hk : k > 2)
    (hx1 : x₁ = 2022 * k - 2022)
    (hx2 : x₂ = k + 1) :
    (x₁ > x₂) →
    (x₁ + 4044) / x₂ = 2022 :=
sorry

end cross_fraction_eq1_cross_fraction_eq2_cross_fraction_eq3_l180_180662


namespace shaded_area_z_shape_l180_180735

theorem shaded_area_z_shape (L W s1 s2 : ℕ) (hL : L = 6) (hW : W = 4) (hs1 : s1 = 2) (hs2 : s2 = 1) :
  (L * W - (s1 * s1 + s2 * s2)) = 19 := by
  sorry

end shaded_area_z_shape_l180_180735


namespace find_k_l180_180430

-- Given conditions
def p (x : ℝ) : ℝ := 2 * x + 3
def q (k : ℝ) (x : ℝ) : ℝ := k * x + k
def intersection (x y : ℝ) : Prop := y = p x ∧ ∃ k, y = q k x

-- Proof that based on the intersection at (1, 5), k evaluates to 5/2
theorem find_k : ∃ k : ℝ, intersection 1 5 → k = 5 / 2 := by
  sorry

end find_k_l180_180430


namespace atomic_weight_of_nitrogen_l180_180853

-- Definitions from conditions
def molecular_weight := 53.0
def hydrogen_weight := 1.008
def chlorine_weight := 35.45
def hydrogen_atoms := 4
def chlorine_atoms := 1

-- The proof goal
theorem atomic_weight_of_nitrogen : 
  53.0 - (4.0 * 1.008) - 35.45 = 13.518 :=
by
  sorry

end atomic_weight_of_nitrogen_l180_180853


namespace time_to_eat_potatoes_l180_180572

theorem time_to_eat_potatoes (rate : ℕ → ℕ → ℝ) (potatoes : ℕ → ℕ → ℝ) 
    (minutes : ℕ) (hours : ℝ) (total_potatoes : ℕ) : 
    rate 3 20 = 9 / 1 -> potatoes 27 9 = 3 := 
by
  intro h1
  -- You can add intermediate steps here as optional comments for clarity during proof construction
  /- 
  Given: 
  rate 3 20 = 9 -> Jason's rate of eating potatoes is 9 potatoes per hour
  time = potatoes / rate -> 27 potatoes / 9 potatoes/hour = 3 hours
  -/
  sorry

end time_to_eat_potatoes_l180_180572


namespace max_ab_correct_l180_180500

noncomputable def max_ab (k : ℝ) (a b: ℝ) : ℝ :=
if k = -3 then 9 else sorry

theorem max_ab_correct (k : ℝ) (a b: ℝ)
  (h1 : (-3 ≤ k ∧ k ≤ 1))
  (h2 : a + b = 2 * k)
  (h3 : a^2 + b^2 = k^2 - 2 * k + 3) :
  max_ab k a b = 9 :=
sorry

end max_ab_correct_l180_180500


namespace validate_shots_statistics_l180_180174

-- Define the scores and their frequencies
def scores : List ℕ := [6, 7, 8, 9, 10]
def times : List ℕ := [4, 10, 11, 9, 6]

-- Condition 1: Calculate the mode
def mode := 8

-- Condition 2: Calculate the median
def median := 8

-- Condition 3: Calculate the 35th percentile
def percentile_35 := ¬(35 * 40 / 100 = 7)

-- Condition 4: Calculate the average
def average := 8.075

theorem validate_shots_statistics :
  mode = 8
  ∧ median = 8
  ∧ percentile_35
  ∧ average = 8.075 :=
by
  sorry

end validate_shots_statistics_l180_180174


namespace find_x_l180_180536

theorem find_x (x : ℚ) : |x + 3| = |x - 4| → x = 1/2 := 
by 
-- Add appropriate content here
sorry

end find_x_l180_180536


namespace range_of_a_l180_180188

theorem range_of_a (M N : Set ℝ) (a : ℝ) 
(hM : M = {x : ℝ | x < 2}) 
(hN : N = {x : ℝ | x < a}) 
(hSubset : M ⊆ N) : 
  2 ≤ a := 
sorry

end range_of_a_l180_180188


namespace distance_between_centers_l180_180063

-- Declare radii of the circles and the shortest distance between points on the circles
def R := 28
def r := 12
def d := 10

-- Define the problem to prove the distance between the centers
theorem distance_between_centers (R r d : ℝ) (hR : R = 28) (hr : r = 12) (hd : d = 10) : 
  ∀ OO1 : ℝ, OO1 = 6 :=
by sorry

end distance_between_centers_l180_180063


namespace cubic_polynomial_value_at_3_and_neg3_l180_180170

variable (Q : ℝ → ℝ)
variable (a b c d m : ℝ)
variable (h1 : Q 1 = 5 * m)
variable (h0 : Q 0 = 2 * m)
variable (h_1 : Q (-1) = 6 * m)
variable (hQ : ∀ x, Q x = a * x^3 + b * x^2 + c * x + d)

theorem cubic_polynomial_value_at_3_and_neg3 :
  Q 3 + Q (-3) = 67 * m := by
  -- sorry is used to skip the proof
  sorry

end cubic_polynomial_value_at_3_and_neg3_l180_180170


namespace quadratic_factorization_value_of_a_l180_180695

theorem quadratic_factorization_value_of_a (a : ℝ) :
  (∀ x : ℝ, 2 * x^2 - 8 * x + a = 0 ↔ 2 * (x - 2)^2 = 4) → a = 4 :=
by
  intro h
  sorry

end quadratic_factorization_value_of_a_l180_180695


namespace calculate_f_sum_l180_180595

noncomputable def f (n : ℕ) := Real.log (3 * n^2) / Real.log 3003

theorem calculate_f_sum :
  f 7 + f 11 + f 13 = 2 :=
by
  sorry

end calculate_f_sum_l180_180595


namespace three_pow_1000_mod_seven_l180_180138

theorem three_pow_1000_mod_seven : (3 ^ 1000) % 7 = 4 := 
by 
  -- proof omitted
  sorry

end three_pow_1000_mod_seven_l180_180138


namespace revenue_from_full_price_tickets_l180_180786

noncomputable def full_price_ticket_revenue (f h p : ℕ) : ℕ := f * p

theorem revenue_from_full_price_tickets (f h p : ℕ) (total_tickets total_revenue : ℕ) 
  (tickets_eq : f + h = total_tickets)
  (revenue_eq : f * p + h * (p / 2) = total_revenue) 
  (total_tickets_value : total_tickets = 180)
  (total_revenue_value : total_revenue = 2652) :
  full_price_ticket_revenue f h p = 984 :=
by {
  sorry
}

end revenue_from_full_price_tickets_l180_180786


namespace border_area_is_198_l180_180683

-- We define the dimensions of the picture and the border width
def picture_height : ℝ := 12
def picture_width : ℝ := 15
def border_width : ℝ := 3

-- We compute the entire framed height and width
def framed_height : ℝ := picture_height + 2 * border_width
def framed_width : ℝ := picture_width + 2 * border_width

-- We compute the area of the picture and framed area
def picture_area : ℝ := picture_height * picture_width
def framed_area : ℝ := framed_height * framed_width

-- We compute the area of the border
def border_area : ℝ := framed_area - picture_area

-- Now we pose the theorem to prove the area of the border is 198 square inches
theorem border_area_is_198 : border_area = 198 := by
  sorry

end border_area_is_198_l180_180683


namespace greatest_three_digit_multiple_of_17_l180_180638

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

theorem greatest_three_digit_multiple_of_17 : ∃ n, is_three_digit n ∧ 17 ∣ n ∧ ∀ k, is_three_digit k ∧ 17 ∣ k → k ≤ n :=
by
  sorry

end greatest_three_digit_multiple_of_17_l180_180638


namespace integer_cubed_fraction_l180_180116

theorem integer_cubed_fraction
  (a b : ℕ)
  (hab : 0 < b ∧ 0 < a)
  (h : (a^2 + b^2) % (a - b)^2 = 0) :
  (a^3 + b^3) % (a - b)^3 = 0 :=
by sorry

end integer_cubed_fraction_l180_180116


namespace number_of_parrots_in_each_cage_l180_180109

theorem number_of_parrots_in_each_cage (num_cages : ℕ) (total_birds : ℕ) (parrots_per_cage parakeets_per_cage : ℕ)
    (h1 : num_cages = 9)
    (h2 : parrots_per_cage = parakeets_per_cage)
    (h3 : total_birds = 36)
    (h4 : total_birds = num_cages * (parrots_per_cage + parakeets_per_cage)) :
  parrots_per_cage = 2 :=
by
  sorry

end number_of_parrots_in_each_cage_l180_180109


namespace domain_h_l180_180648

noncomputable def h (x : ℝ) : ℝ := (3 * x - 1) / Real.sqrt (x - 5)

theorem domain_h (x : ℝ) : h x = (3 * x - 1) / Real.sqrt (x - 5) → (x > 5) :=
by
  intro hx
  have hx_nonneg : x - 5 >= 0 := sorry
  have sqrt_nonzero : Real.sqrt (x - 5) ≠ 0 := sorry
  sorry

end domain_h_l180_180648


namespace first_step_of_testing_circuit_broken_l180_180591

-- Definitions based on the problem
def circuit_broken : Prop := true
def binary_search_method : Prop := true
def test_first_step_at_midpoint : Prop := true

-- The theorem stating the first step in testing a broken circuit using the binary search method
theorem first_step_of_testing_circuit_broken (h1 : circuit_broken) (h2 : binary_search_method) :
  test_first_step_at_midpoint :=
sorry

end first_step_of_testing_circuit_broken_l180_180591


namespace lunch_choices_l180_180882

theorem lunch_choices (chickens drinks : ℕ) (h1 : chickens = 3) (h2 : drinks = 2) : chickens * drinks = 6 :=
by
  sorry

end lunch_choices_l180_180882


namespace false_props_l180_180464

-- Definitions for conditions
def prop1 :=
  ∀ (a b c d : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ (a * d = b * c) → 
  (a / b = b / c ∧ b / c = c / d)

def prop2 :=
  ∀ (a : ℕ), (∃ k : ℕ, a = 2 * k) → (a % 2 = 0)

def prop3 :=
  ∀ (A : ℝ), (A > 30) → (Real.sin (A * Real.pi / 180) > 1 / 2)

-- Theorem statement
theorem false_props : (¬ prop1) ∧ (¬ prop3) :=
by sorry

end false_props_l180_180464


namespace recurring_to_fraction_l180_180037

theorem recurring_to_fraction : ∀ (x : ℚ), x = 0.5656 ∧ 100 * x = 56.5656 → x = 56 / 99 :=
by
  intros x hx
  sorry

end recurring_to_fraction_l180_180037


namespace min_value_sin_cos_l180_180477

open Real

theorem min_value_sin_cos (x : ℝ) : 
  ∃ x : ℝ, sin x ^ 4 + 2 * cos x ^ 4 = 2 / 3 ∧ ∀ x : ℝ, sin x ^ 4 + 2 * cos x ^ 4 ≥ 2 / 3 :=
sorry

end min_value_sin_cos_l180_180477


namespace eldest_child_age_l180_180045

variable (y m e : Nat)

theorem eldest_child_age :
  (m - y = 3) →
  (e = 3 * y) →
  (e = y + m + 2) →
  (e = 15) :=
by
  intros h1 h2 h3
  sorry

end eldest_child_age_l180_180045


namespace intersection_of_S_and_T_l180_180681

-- Define S and T based on given conditions
def S : Set ℝ := { x | x^2 + 2 * x = 0 }
def T : Set ℝ := { x | x^2 - 2 * x = 0 }

-- Prove the intersection of S and T
theorem intersection_of_S_and_T : S ∩ T = {0} :=
sorry

end intersection_of_S_and_T_l180_180681


namespace trigonometric_inequality_l180_180267

theorem trigonometric_inequality (x : ℝ) (h1 : 0 < x) (h2 : x < π / 2) :
  0 < (1 / (Real.sin x)^2) - (1 / x^2) ∧ (1 / (Real.sin x)^2) - (1 / x^2) < 1 := 
sorry

end trigonometric_inequality_l180_180267


namespace price_of_rice_packet_l180_180657

-- Definitions based on conditions
def initial_amount : ℕ := 500
def wheat_flour_price : ℕ := 25
def wheat_flour_quantity : ℕ := 3
def soda_price : ℕ := 150
def remaining_balance : ℕ := 235
def total_spending (P : ℕ) : ℕ := initial_amount - remaining_balance

-- Theorem to prove
theorem price_of_rice_packet (P : ℕ) (h: 2 * P + wheat_flour_quantity * wheat_flour_price + soda_price = total_spending P) : P = 20 :=
sorry

end price_of_rice_packet_l180_180657


namespace amount_returned_l180_180665

theorem amount_returned (deposit_usd : ℝ) (exchange_rate : ℝ) (h1 : deposit_usd = 10000) (h2 : exchange_rate = 58.15) : 
  deposit_usd * exchange_rate = 581500 := 
by 
  sorry

end amount_returned_l180_180665


namespace find_a_l180_180517

def has_root_greater_than_zero (a : ℝ) : Prop :=
  ∃ x : ℝ, x > 0 ∧ ((3 * x - 1) / (x - 3) = a / (3 - x) - 1)

theorem find_a (a : ℝ) : has_root_greater_than_zero a → a = -8 :=
by
  sorry

end find_a_l180_180517


namespace square_fold_distance_l180_180062

noncomputable def distance_from_A (area : ℝ) (visible_equal : Bool) : ℝ :=
  if area = 18 ∧ visible_equal then 2 * Real.sqrt 6 else 0

theorem square_fold_distance (area : ℝ) (visible_equal : Bool) :
  area = 18 → visible_equal → distance_from_A area visible_equal = 2 * Real.sqrt 6 :=
by
  sorry

end square_fold_distance_l180_180062


namespace evaluate_expression_l180_180597

theorem evaluate_expression : (1 / (2 + (1 / (3 + (1 / 4))))) = (13 / 30) :=
by
  sorry

end evaluate_expression_l180_180597


namespace number_of_paths_l180_180864

-- Define the coordinates and the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.choose n k

def E := (0, 7)
def F := (4, 5)
def G := (9, 0)

-- Define the number of steps required for each path segment
def steps_to_F := 6
def steps_to_G := 10

-- Capture binomial coefficients for the calculated path segments
def paths_E_to_F := binomial steps_to_F 4
def paths_F_to_G := binomial steps_to_G 5

-- Prove the total number of paths from E to G through F
theorem number_of_paths : paths_E_to_F * paths_F_to_G = 3780 :=
by rw [paths_E_to_F, paths_F_to_G]; sorry

end number_of_paths_l180_180864


namespace inequality_solution_l180_180068

variable {a b : ℝ}

theorem inequality_solution
  (h1 : a < 0)
  (h2 : -1 < b ∧ b < 0) :
  ab > ab^2 ∧ ab^2 > a := 
sorry

end inequality_solution_l180_180068


namespace problem_l180_180418

noncomputable def trajectory_C (x y : ℝ) : Prop :=
  y^2 = -8 * x

theorem problem (P : ℝ × ℝ) (k : ℝ) (h : -1 < k ∧ k < 0) 
  (H1 : P.1 = -2 ∨ P.1 = 2)
  (H2 : trajectory_C P.1 P.2) :
  ∃ Q : ℝ × ℝ, Q.1 < -6 :=
  sorry

end problem_l180_180418


namespace negation_universal_proposition_l180_180521

theorem negation_universal_proposition :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x : ℝ, x^2 < 0) :=
by
  sorry

end negation_universal_proposition_l180_180521


namespace system_solution_l180_180165

theorem system_solution (x y c d : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hd : d ≠ 0)
  (h1 : 8 * x - 6 * y = c) (h2 : 10 * y - 15 * x = d) : 
  c / d = -2 / 5 :=
by
  sorry

end system_solution_l180_180165


namespace no_unboxed_products_l180_180247

-- Definitions based on the conditions
def big_box_capacity : ℕ := 50
def small_box_capacity : ℕ := 40
def total_products : ℕ := 212

-- Theorem statement proving the least number of unboxed products
theorem no_unboxed_products (big_box_capacity small_box_capacity total_products : ℕ) : 
  (total_products - (total_products / big_box_capacity) * big_box_capacity) % small_box_capacity = 0 :=
by
  sorry

end no_unboxed_products_l180_180247


namespace incorrect_statement_l180_180452

def population : ℕ := 13000
def sample_size : ℕ := 500
def academic_performance (n : ℕ) : Type := sorry

def statement_A (ap : Type) : Prop := 
  ap = academic_performance population

def statement_B (ap : Type) : Prop := 
  ∀ (u : ℕ), u ≤ population → ap = academic_performance 1

def statement_C (ap : Type) : Prop := 
  ap = academic_performance sample_size

def statement_D : Prop := 
  sample_size = 500

theorem incorrect_statement : ¬ (statement_B (academic_performance 1)) :=
sorry

end incorrect_statement_l180_180452


namespace larger_of_two_numbers_l180_180255

noncomputable def larger_number (HCF LCM A B : ℕ) : ℕ :=
  if HCF = 23 ∧ LCM = 23 * 9 * 10 ∧ A * B = HCF * LCM ∧ (A = 10 ∧ B = 23 * 9 ∨ B = 10 ∧ A = 23 * 9)
  then max A B
  else 0

theorem larger_of_two_numbers : larger_number (23) (23 * 9 * 10) 230 207 = 230 := by
  sorry

end larger_of_two_numbers_l180_180255


namespace find_ax5_plus_by5_l180_180566

theorem find_ax5_plus_by5 (a b x y : ℝ)
  (h1 : a * x + b * y = 3)
  (h2 : a * x^2 + b * y^2 = 7)
  (h3 : a * x^3 + b * y^3 = 16)
  (h4 : a * x^4 + b * y^4 = 42) :
  a * x^5 + b * y^5 = 20 := 
sorry

end find_ax5_plus_by5_l180_180566


namespace mean_value_of_quadrilateral_angles_l180_180371

theorem mean_value_of_quadrilateral_angles 
  (a1 a2 a3 a4 : ℝ) 
  (h_sum : a1 + a2 + a3 + a4 = 360) :
  (a1 + a2 + a3 + a4) / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l180_180371


namespace jillian_largest_apartment_l180_180187

noncomputable def largest_apartment_size (budget : ℝ) (rate : ℝ) : ℝ :=
  budget / rate

theorem jillian_largest_apartment : largest_apartment_size 720 1.20 = 600 := by
  sorry

end jillian_largest_apartment_l180_180187


namespace avg_diff_condition_l180_180163

variable (a b c : ℝ)

theorem avg_diff_condition (h1 : (a + b) / 2 = 110) (h2 : (b + c) / 2 = 150) : a - c = -80 :=
by
  sorry

end avg_diff_condition_l180_180163


namespace exists_k_in_octahedron_l180_180940

theorem exists_k_in_octahedron
  (x0 y0 z0 : ℚ)
  (h : ∀ n : ℤ, x0 + y0 + z0 ≠ n ∧ 
                 x0 + y0 - z0 ≠ n ∧ 
                 x0 - y0 + z0 ≠ n ∧ 
                 x0 - y0 - z0 ≠ n) :
  ∃ k : ℕ, ∃ (xk yk zk : ℚ), 
    k ≠ 0 ∧ 
    xk = k * x0 ∧ 
    yk = k * y0 ∧ 
    zk = k * z0 ∧
    ∀ n : ℤ, 
      (xk + yk + zk < ↑n → xk + yk + zk > ↑(n - 1)) ∧ 
      (xk + yk - zk < ↑n → xk + yk - zk > ↑(n - 1)) ∧ 
      (xk - yk + zk < ↑n → xk - yk + zk > ↑(n - 1)) ∧ 
      (xk - yk - zk < ↑n → xk - yk - zk > ↑(n - 1)) :=
sorry

end exists_k_in_octahedron_l180_180940


namespace glens_speed_is_37_l180_180127

/-!
# Problem Statement
Glen and Hannah drive at constant speeds toward each other on a highway. Glen drives at a certain speed G km/h. At some point, they pass by each other, and keep driving away from each other, maintaining their constant speeds. 
Glen is 130 km away from Hannah at 6 am and again at 11 am. Hannah is driving at 15 kilometers per hour.
Prove that Glen's speed is 37 km/h.
-/

def glens_speed (G : ℝ) : Prop :=
  ∃ G: ℝ, 
    (∃ H_speed : ℝ, H_speed = 15) ∧ -- Hannah's speed
    (∃ distance : ℝ, distance = 130) ∧ -- distance at 6 am and 11 am
    G + 15 = 260 / 5 -- derived equation from conditions

theorem glens_speed_is_37 : glens_speed 37 :=
by {
  sorry -- proof to be filled in
}

end glens_speed_is_37_l180_180127


namespace who_scored_full_marks_l180_180519

-- Define students and their statements
inductive Student
| A | B | C

open Student

def scored_full_marks (s : Student) : Prop :=
  match s with
  | A => true
  | B => true
  | C => true

def statement_A : Prop := scored_full_marks A
def statement_B : Prop := ¬ scored_full_marks C
def statement_C : Prop := statement_B

-- Given conditions
def exactly_one_lied (a b c : Prop) : Prop :=
  (a ∧ ¬ b ∧ ¬ c) ∨ (¬ a ∧ b ∧ ¬ c) ∨ (¬ a ∧ ¬ b ∧ c)

-- Main proof statement: Prove that B scored full marks
theorem who_scored_full_marks (h : exactly_one_lied statement_A statement_B statement_C) : scored_full_marks B :=
sorry

end who_scored_full_marks_l180_180519


namespace apples_per_pie_l180_180548

-- Definitions of given conditions
def total_apples : ℕ := 75
def handed_out_apples : ℕ := 19
def remaining_apples : ℕ := total_apples - handed_out_apples
def pies_made : ℕ := 7

-- Statement of the problem to be proved
theorem apples_per_pie : remaining_apples / pies_made = 8 := by
  sorry

end apples_per_pie_l180_180548


namespace news_spread_time_l180_180682

theorem news_spread_time (n : ℕ) (m : ℕ) :
  (2^m < n ∧ n < 2^(m+k+1) ∧ (n % 2 = 1) ∧ n % 2 = 1) →
  ∃ t : ℕ, t = (if n % 2 = 1 then m+2 else m+1) := 
sorry

end news_spread_time_l180_180682


namespace more_balloons_allan_l180_180748

theorem more_balloons_allan (allan_balloons : ℕ) (jake_initial_balloons : ℕ) (jake_bought_balloons : ℕ) 
  (h1 : allan_balloons = 6) (h2 : jake_initial_balloons = 2) (h3 : jake_bought_balloons = 3) :
  allan_balloons = jake_initial_balloons + jake_bought_balloons + 1 := 
by 
  -- Assuming Jake's total balloons after purchase
  let jake_total_balloons := jake_initial_balloons + jake_bought_balloons
  -- The proof would involve showing that Allan's balloons are one more than Jake's total balloons
  sorry

end more_balloons_allan_l180_180748


namespace total_emails_vacation_l180_180137

def day_1_emails : ℕ := 16
def day_2_emails : ℕ := day_1_emails / 2
def day_3_emails : ℕ := day_2_emails / 2
def day_4_emails : ℕ := day_3_emails / 2

def total_emails : ℕ := day_1_emails + day_2_emails + day_3_emails + day_4_emails

theorem total_emails_vacation : total_emails = 30 := by
  -- Use "sorry" to skip the proof as per instructions.
  sorry

end total_emails_vacation_l180_180137


namespace correct_equation_l180_180420

theorem correct_equation:
  (∀ x y : ℝ, -5 * (x - y) = -5 * x + 5 * y) ∧ 
  (∀ a c : ℝ, ¬ (-2 * (-a + c) = -2 * a - 2 * c)) ∧ 
  (∀ x y z : ℝ, ¬ (3 - (x + y + z) = -x + y - z)) ∧ 
  (∀ a b : ℝ, ¬ (3 * (a + 2 * b) = 3 * a + 2 * b)) :=
by
  sorry

end correct_equation_l180_180420


namespace depth_of_well_l180_180941

theorem depth_of_well
  (d : ℝ)
  (h1 : ∃ t1 t2 : ℝ, 18 * t1^2 = d ∧ t2 = d / 1150 ∧ t1 + t2 = 8) :
  d = 33.18 :=
sorry

end depth_of_well_l180_180941


namespace problem_l180_180182

theorem problem (a b : ℕ) (ha : 2^a ∣ 180) (h2 : ∀ n, 2^n ∣ 180 → n ≤ a) (hb : 5^b ∣ 180) (h5 : ∀ n, 5^n ∣ 180 → n ≤ b) : (1 / 3) ^ (b - a) = 3 := by
  sorry

end problem_l180_180182


namespace gcd_polynomial_l180_180492

theorem gcd_polynomial (b : ℤ) (h : 2142 ∣ b) : Int.gcd (b^2 + 11 * b + 28) (b + 6) = 2 :=
sorry

end gcd_polynomial_l180_180492


namespace jean_pages_written_l180_180173

theorem jean_pages_written:
  (∀ d : ℕ, 150 * d = 900 → d * 2 = 12) :=
by
  sorry

end jean_pages_written_l180_180173


namespace simplify_expression_l180_180463

theorem simplify_expression : (90 / 150) * (35 / 21) = 1 :=
by
  -- Insert proof here 
  sorry

end simplify_expression_l180_180463


namespace parallelogram_diagonal_length_l180_180356

-- Define a structure to represent a parallelogram
structure Parallelogram :=
  (side_length : ℝ) 
  (diagonal_length : ℝ)
  (perpendicular : Bool)

-- State the theorem about the relationship between the diagonals in a parallelogram
theorem parallelogram_diagonal_length (a b : ℝ) (P : Parallelogram) (h₀ : P.side_length = a) (h₁ : P.diagonal_length = b) (h₂ : P.perpendicular = true) : 
  ∃ (AC : ℝ), AC = Real.sqrt (4 * a^2 + b^2) :=
by
  sorry

end parallelogram_diagonal_length_l180_180356


namespace calculate_v2_using_horner_method_l180_180344

def f (x : ℕ) : ℕ := x^5 + 5 * x^4 + 10 * x^3 + 10 * x^2 + 5 * x + 1

def horner_step (x b a : ℕ) := a * x + b

def horner_eval (coeffs : List ℕ) (x : ℕ) : ℕ :=
coeffs.foldr (horner_step x) 0

theorem calculate_v2_using_horner_method :
  horner_eval [1, 5, 10, 10, 5, 1] 2 = 24 :=
by
  -- This is the theorem statement, the proof is not required as per instructions
  sorry

end calculate_v2_using_horner_method_l180_180344


namespace range_of_m_l180_180377

open Set

def setM (m : ℝ) : Set ℝ := { x | x ≤ m }
def setP : Set ℝ := { x | x ≥ -1 }

theorem range_of_m (m : ℝ) (h : setM m ∩ setP = ∅) : m < -1 := sorry

end range_of_m_l180_180377


namespace number_of_ways_to_distribute_balls_l180_180304

theorem number_of_ways_to_distribute_balls : 
  ∃ n : ℕ, n = 81 ∧ n = 3^4 := 
by sorry

end number_of_ways_to_distribute_balls_l180_180304


namespace ratio_of_side_lengths_l180_180330

theorem ratio_of_side_lengths (t p : ℕ) (h1 : 3 * t = 30) (h2 : 5 * p = 30) : t / p = 5 / 3 :=
by
  sorry

end ratio_of_side_lengths_l180_180330


namespace find_q_revolutions_per_minute_l180_180814

variable (p_rpm : ℕ) (q_rpm : ℕ) (t : ℕ)

def revolutions_per_minute_q : Prop :=
  (p_rpm = 10) → (t = 4) → (q_rpm = (10 / 60 * 4 + 2) * 60 / 4) → (q_rpm = 120)

theorem find_q_revolutions_per_minute (p_rpm q_rpm t : ℕ) :
  revolutions_per_minute_q p_rpm q_rpm t :=
by
  unfold revolutions_per_minute_q
  sorry

end find_q_revolutions_per_minute_l180_180814


namespace kyle_money_l180_180805

theorem kyle_money (dave_money : ℕ) (kyle_initial : ℕ) (kyle_remaining : ℕ)
  (h1 : dave_money = 46)
  (h2 : kyle_initial = 3 * dave_money - 12)
  (h3 : kyle_remaining = kyle_initial - kyle_initial / 3) :
  kyle_remaining = 84 :=
by
  -- Define Dave's money and provide the assumption
  let dave_money := 46
  have h1 : dave_money = 46 := rfl

  -- Define Kyle's initial money based on Dave's money
  let kyle_initial := 3 * dave_money - 12
  have h2 : kyle_initial = 3 * dave_money - 12 := rfl

  -- Define Kyle's remaining money after spending one third on snowboarding
  let kyle_remaining := kyle_initial - kyle_initial / 3
  have h3 : kyle_remaining = kyle_initial - kyle_initial / 3 := rfl

  -- Now we prove that Kyle's remaining money is 84
  sorry -- Proof steps omitted

end kyle_money_l180_180805


namespace edward_candy_purchase_l180_180989

theorem edward_candy_purchase (whack_a_mole_tickets : ℕ) (skee_ball_tickets : ℕ) (candy_cost : ℕ) 
  (h1 : whack_a_mole_tickets = 3) (h2 : skee_ball_tickets = 5) (h3 : candy_cost = 4) :
  (whack_a_mole_tickets + skee_ball_tickets) / candy_cost = 2 := 
by 
  sorry

end edward_candy_purchase_l180_180989


namespace average_without_ivan_l180_180469

theorem average_without_ivan
  (total_friends : ℕ := 5)
  (avg_all : ℝ := 55)
  (ivan_amount : ℝ := 43)
  (remaining_friends : ℕ := total_friends - 1)
  (total_amount : ℝ := total_friends * avg_all)
  (remaining_amount : ℝ := total_amount - ivan_amount)
  (new_avg : ℝ := remaining_amount / remaining_friends) :
  new_avg = 58 := 
sorry

end average_without_ivan_l180_180469


namespace book_pages_total_l180_180224

theorem book_pages_total
  (pages_read_first_day : ℚ) (total_pages : ℚ) (pages_read_second_day : ℚ)
  (rem_read_ratio : ℚ) (read_ratio_mult : ℚ)
  (book_ratio: ℚ) (read_pages_ratio: ℚ)
  (read_second_day_ratio: ℚ):
  pages_read_first_day = 1 / 6 →
  pages_read_second_day = 42 →
  rem_read_ratio = 3 →
  read_ratio_mult = (2 / 6) →
  book_ratio = 3 / 5 →
  read_pages_ratio = 2 / 5 →
  read_second_day_ratio = (2 / 5 - 1 / 6) →
  total_pages = pages_read_second_day / read_second_day_ratio  →
  total_pages = 126 :=
by sorry

end book_pages_total_l180_180224


namespace g_2002_value_l180_180663

noncomputable def g : ℕ → ℤ := sorry

theorem g_2002_value :
  (∀ a b n : ℕ, a + b = 2^n → g a + g b = n^3) →
  (g 2 + g 46 = 180) →
  g 2002 = 1126 := 
by
  intros h1 h2
  sorry

end g_2002_value_l180_180663


namespace new_rectangle_area_l180_180313

theorem new_rectangle_area :
  let a := 3
  let b := 4
  let diagonal := Real.sqrt (a^2 + b^2)
  let sum_of_sides := a + b
  let area := diagonal * sum_of_sides
  area = 35 :=
by
  sorry

end new_rectangle_area_l180_180313


namespace find_range_of_m_l180_180698

def has_two_distinct_negative_real_roots (m : ℝ) : Prop := 
  let Δ := m^2 - 4
  Δ > 0 ∧ -m > 0

def inequality_holds_for_all_real (m : ℝ) : Prop :=
  let Δ := (4 * (m - 2))^2 - 16
  Δ < 0

def problem_statement (m : ℝ) : Prop :=
  (has_two_distinct_negative_real_roots m ∨ inequality_holds_for_all_real m) ∧ 
  ¬(has_two_distinct_negative_real_roots m ∧ inequality_holds_for_all_real m)

theorem find_range_of_m (m : ℝ) : problem_statement m ↔ ((1 < m ∧ m ≤ 2) ∨ (3 ≤ m)) :=
by
  sorry

end find_range_of_m_l180_180698


namespace expected_balls_in_original_position_proof_l180_180664

-- Define the problem conditions as Lean definitions
def n_balls : ℕ := 10

def probability_not_moved_by_one_rotation : ℚ := 7 / 10

def probability_not_moved_by_two_rotations : ℚ := (7 / 10) * (7 / 10)

def expected_balls_in_original_position : ℚ := n_balls * probability_not_moved_by_two_rotations

-- The statement representing the proof problem
theorem expected_balls_in_original_position_proof :
  expected_balls_in_original_position = 4.9 :=
  sorry

end expected_balls_in_original_position_proof_l180_180664


namespace otimes_calculation_l180_180391

def otimes (x y : ℝ) : ℝ := x^2 + y^2

theorem otimes_calculation (x : ℝ) : otimes x (otimes x x) = x^2 + 4 * x^4 :=
by
  sorry

end otimes_calculation_l180_180391


namespace highest_financial_backing_l180_180525

-- Let x be the lowest level of financial backing
-- Define the five levels of backing as x, 6x, 36x, 216x, 1296x
-- Given that the total raised is $200,000

theorem highest_financial_backing (x : ℝ) 
  (h₁: 50 * x + 20 * 6 * x + 12 * 36 * x + 7 * 216 * x + 4 * 1296 * x = 200000) : 
  1296 * x = 35534 :=
sorry

end highest_financial_backing_l180_180525


namespace inequality_for_positive_n_and_x_l180_180069

theorem inequality_for_positive_n_and_x (n : ℕ) (x : ℝ) (hn : n > 0) (hx : x > 0) :
  (x^(2 * n - 1) - 1) / (2 * n - 1) ≤ (x^(2 * n) - 1) / (2 * n) :=
by sorry

end inequality_for_positive_n_and_x_l180_180069


namespace parabola_solution_l180_180518

noncomputable def parabola_coefficients (a b c : ℝ) : Prop :=
  (6 : ℝ) = a * (5 : ℝ)^2 + b * (5 : ℝ) + c ∧
  0 = a * (3 : ℝ)^2 + b * (3 : ℝ) + c

theorem parabola_solution :
  ∃ (a b c : ℝ), parabola_coefficients a b c ∧ (a + b + c = 6) :=
by {
  -- definitions and constraints based on problem conditions
  sorry
}

end parabola_solution_l180_180518


namespace volume_of_cut_pyramid_l180_180130

theorem volume_of_cut_pyramid
  (base_length : ℝ)
  (slant_length : ℝ)
  (cut_height : ℝ)
  (original_base_area : ℝ)
  (original_height : ℝ)
  (new_base_area : ℝ)
  (volume : ℝ)
  (h_base_length : base_length = 8 * Real.sqrt 2)
  (h_slant_length : slant_length = 10)
  (h_cut_height : cut_height = 3)
  (h_original_base_area : original_base_area = (base_length ^ 2) / 2)
  (h_original_height : original_height = Real.sqrt (slant_length ^ 2 - (base_length / Real.sqrt 2) ^ 2))
  (h_new_base_area : new_base_area = original_base_area / 4)
  (h_volume : volume = (1 / 3) * new_base_area * cut_height) :
  volume = 32 :=
by
  sorry

end volume_of_cut_pyramid_l180_180130


namespace particular_solution_exists_l180_180804

noncomputable def general_solution (C : ℝ) (x : ℝ) : ℝ := C * x + 1

def differential_equation (x y y' : ℝ) : Prop := x * y' = y - 1

def initial_condition (y : ℝ) : Prop := y = 5

theorem particular_solution_exists :
  (∀ C x y, y = general_solution C x → differential_equation x y (C : ℝ)) →
  (∃ C, initial_condition (general_solution C 1)) →
  (∀ x, ∃ y, y = general_solution 4 x) :=
by
  intros h1 h2
  sorry

end particular_solution_exists_l180_180804


namespace total_cost_of_trip_l180_180523

def totalDistance (d1 d2 d3 d4 : ℕ) : ℕ :=
  d1 + d2 + d3 + d4

def gallonsUsed (distance miles_per_gallon : ℕ) : ℕ :=
  distance / miles_per_gallon

def totalCost (gallons : ℕ) (cost_per_gallon : ℕ) : ℕ :=
  gallons * cost_per_gallon

theorem total_cost_of_trip :
  (totalDistance 10 6 5 9 = 30) →
  (gallonsUsed 30 15 = 2) →
  totalCost 2 35 = 700 :=
by
  sorry

end total_cost_of_trip_l180_180523


namespace math_problem_l180_180611

theorem math_problem
  (x₁ x₂ x₃ x₄ x₅ x₆ x₇ : ℝ)
  (h1 : x₁ + 4 * x₂ + 9 * x₃ + 16 * x₄ + 25 * x₅ + 36 * x₆ + 49 * x₇ = 1)
  (h2 : 4 * x₁ + 9 * x₂ + 16 * x₃ + 25 * x₄ + 36 * x₅ + 49 * x₆ + 64 * x₇ = 12)
  (h3 : 9 * x₁ + 16 * x₂ + 25 * x₃ + 36 * x₄ + 49 * x₅ + 64 * x₆ + 81 * x₇ = 123) :
  16 * x₁ + 25 * x₂ + 36 * x₃ + 49 * x₄ + 64 * x₅ + 81 * x₆ + 100 * x₇ = 334 := by
  sorry

end math_problem_l180_180611


namespace last_student_calls_out_l180_180343

-- Define the transformation rules as a function
def next_student (n : ℕ) : ℕ :=
  if n < 10 then n + 8 else (n % 10) + 7

-- Define the sequence generation function
noncomputable def student_number : ℕ → ℕ
| 0       => 1  -- the 1st student starts with number 1
| (n + 1) => next_student (student_number n)

-- The main theorem to prove
theorem last_student_calls_out (n : ℕ) : student_number 2013 = 12 :=
sorry

end last_student_calls_out_l180_180343


namespace number_of_possible_ceil_values_l180_180976

theorem number_of_possible_ceil_values (x : ℝ) (h : ⌈x⌉ = 15) : 
  (∃ (n : ℕ), 196 < x^2 ∧ x^2 ≤ 225 → n = 29) := by
sorry

end number_of_possible_ceil_values_l180_180976


namespace total_protest_days_l180_180810

theorem total_protest_days (d1 : ℕ) (increase_percent : ℕ) (d2 : ℕ) (total_days : ℕ) (h1 : d1 = 4) (h2 : increase_percent = 25) (h3 : d2 = d1 + (d1 * increase_percent / 100)) : total_days = d1 + d2 → total_days = 9 :=
by
  intros
  sorry

end total_protest_days_l180_180810


namespace k_value_tangent_l180_180289

-- Defining the equations
def line (k : ℝ) (x y : ℝ) : Prop := 3 * x + 5 * y + k = 0
def parabola (x y : ℝ) : Prop := y^2 = 24 * x

-- The main theorem stating that k must be 50 for the line to be tangent to the parabola
theorem k_value_tangent (k : ℝ) : (∀ x y : ℝ, line k x y → parabola x y → True) → k = 50 :=
by 
  -- The proof can be constructed based on the discriminant condition provided in the problem
  sorry

end k_value_tangent_l180_180289


namespace line_is_tangent_to_circle_l180_180708

theorem line_is_tangent_to_circle
  (θ : Real)
  (l : ℝ → ℝ → Prop) (C : ℝ → ℝ → Prop)
  (h_l : ∀ x y, l x y ↔ x * Real.sin θ + 2 * y * Real.cos θ = 1)
  (h_C : ∀ x y, C x y ↔ x^2 + y^2 = 1) :
  (∀ x y, l x y ↔ x = 1 ∨ x = -1) ↔
  (∃ x y, C x y ∧ ∀ x y, l x y → Real.sqrt ((x * Real.sin θ + 2 * y * Real.cos θ - 1)^2 / (Real.sin θ^2 + 4 * Real.cos θ^2)) = 1) :=
sorry

end line_is_tangent_to_circle_l180_180708


namespace hotel_room_mistake_l180_180029

theorem hotel_room_mistake (a b c : ℕ) (ha : 1 ≤ a ∧ a ≤ 9) (hb : 0 ≤ b ∧ b ≤ 9) (hc : 0 ≤ c ∧ c ≤ 9) :
  100 * a + 10 * b + c = (a + 1) * (b + 1) * c → false := by sorry

end hotel_room_mistake_l180_180029


namespace range_of_f_l180_180612

def f (x : Int) : Int :=
  x + 1

def domain : Set Int :=
  {-1, 1, 2}

theorem range_of_f :
  Set.image f domain = {0, 2, 3} :=
by
  sorry

end range_of_f_l180_180612


namespace minimize_side_length_of_triangle_l180_180808

-- Define a triangle with sides a, b, and c and angle C
structure Triangle :=
  (a b c : ℝ)
  (C : ℝ) -- angle C in radians
  (area : ℝ) -- area of the triangle

-- Define the conditions for the problem
def conditions (T : Triangle) : Prop :=
  T.area > 0 ∧ T.C > 0 ∧ T.C < Real.pi

-- Define the desired result
def min_side_length (T : Triangle) : Prop :=
  T.a = T.b ∧ T.a = Real.sqrt ((2 * T.area) / Real.sin T.C)

-- The theorem to be proven
theorem minimize_side_length_of_triangle (T : Triangle) (h : conditions T) : min_side_length T :=
  sorry

end minimize_side_length_of_triangle_l180_180808


namespace max_pencils_thrown_out_l180_180417

theorem max_pencils_thrown_out (n : ℕ) : (n % 7 ≤ 6) :=
by
  sorry

end max_pencils_thrown_out_l180_180417


namespace min_value_of_expression_l180_180967

theorem min_value_of_expression (x y : ℝ) : 
  ∃ m : ℝ, m = (xy - 1)^2 + (x + y)^2 ∧ (∀ x y : ℝ, (xy - 1)^2 + (x + y)^2 ≥ m) := 
sorry

end min_value_of_expression_l180_180967


namespace initial_eggs_ben_l180_180706

-- Let's define the conditions from step a):
def eggs_morning := 4
def eggs_afternoon := 3
def eggs_left := 13

-- Define the total eggs Ben ate
def eggs_eaten := eggs_morning + eggs_afternoon

-- Now we define the initial eggs Ben had
def initial_eggs := eggs_left + eggs_eaten

-- The theorem that states the initial number of eggs
theorem initial_eggs_ben : initial_eggs = 20 :=
  by sorry

end initial_eggs_ben_l180_180706


namespace cakes_count_l180_180003

theorem cakes_count (x y : ℕ) 
  (price_fruit price_chocolate total_cost : ℝ) 
  (avg_price : ℝ) 
  (H1 : price_fruit = 4.8)
  (H2 : price_chocolate = 6.6)
  (H3 : total_cost = 167.4)
  (H4 : avg_price = 6.2)
  (H5 : price_fruit * x + price_chocolate * y = total_cost)
  (H6 : total_cost / (x + y) = avg_price) : 
  x = 6 ∧ y = 21 := 
by
  sorry

end cakes_count_l180_180003


namespace lindsey_savings_in_october_l180_180380

-- Definitions based on conditions
def savings_september := 50
def savings_november := 11
def spending_video_game := 87
def final_amount_left := 36
def mom_gift := 25

-- The theorem statement
theorem lindsey_savings_in_october (X : ℕ) 
  (h1 : savings_september + X + savings_november > 75) 
  (total_savings := savings_september + X + savings_november + mom_gift) 
  (final_condition : total_savings - spending_video_game = final_amount_left) : 
  X = 37 :=
by
  sorry

end lindsey_savings_in_october_l180_180380


namespace interval1_increase_decrease_interval2_increasing_interval3_increase_decrease_l180_180251

section
open Real

noncomputable def interval1 (x : ℝ) : Real := log (1 - x ^ 2)
noncomputable def interval2 (x : ℝ) : Real := x * (1 + 2 * sqrt x)
noncomputable def interval3 (x : ℝ) : Real := log (abs x)

-- Function 1: p = ln(1 - x^2)
theorem interval1_increase_decrease :
  (∀ x : ℝ, -1 < x ∧ x < 0 → deriv interval1 x > 0) ∧
  (∀ x : ℝ, 0 < x ∧ x < 1 → deriv interval1 x < 0) := by
  sorry

-- Function 2: z = x(1 + 2√x)
theorem interval2_increasing :
  ∀ x : ℝ, x ≥ 0 → deriv interval2 x > 0 := by
  sorry

-- Function 3: y = ln|x|
theorem interval3_increase_decrease :
  (∀ x : ℝ, x < 0 → deriv interval3 x < 0) ∧
  (∀ x : ℝ, x > 0 → deriv interval3 x > 0) := by
  sorry

end

end interval1_increase_decrease_interval2_increasing_interval3_increase_decrease_l180_180251


namespace value_of_a_plus_b_l180_180891

theorem value_of_a_plus_b (a b c : ℤ) 
    (h1 : a + b + c = 11)
    (h2 : a + b - c = 19)
    : a + b = 15 := 
by
    -- Mathematical details skipped
    sorry

end value_of_a_plus_b_l180_180891


namespace binary_division_correct_l180_180642

def b1100101 := 0b1100101
def b1101 := 0b1101
def b101 := 0b101
def expected_result := 0b11111010

theorem binary_division_correct : ((b1100101 * b1101) / b101) = expected_result :=
by {
  sorry
}

end binary_division_correct_l180_180642


namespace triangular_region_area_l180_180788

theorem triangular_region_area :
  let x_intercept := 4
  let y_intercept := 6
  let area := (1 / 2) * x_intercept * y_intercept
  area = 12 :=
by
  sorry

end triangular_region_area_l180_180788


namespace three_digit_numbers_count_l180_180394

theorem three_digit_numbers_count : 
  ∃ (count : ℕ), count = 3 ∧ 
  ∀ (n : ℕ), (100 ≤ n ∧ n < 1000) ∧ 
             (n / 100 = 9) ∧ 
             (∃ a b c, n = 100 * a + 10 * b + c ∧ a + b + c = 27) ∧ 
             (n % 2 = 0) → count = 3 :=
sorry

end three_digit_numbers_count_l180_180394


namespace expression_eval_l180_180081

theorem expression_eval :
  -14 - (-2) ^ 3 * (1 / 4) - 16 * (1 / 2 - 1 / 4 + 3 / 8) = -22 := by
  sorry

end expression_eval_l180_180081


namespace captain_age_l180_180316

-- Definitions
def num_team_members : ℕ := 11
def total_team_age : ℕ := 11 * 24
def total_age_remainder := 9 * (24 - 1)
def combined_age_of_captain_and_keeper := total_team_age - total_age_remainder

-- The actual proof statement
theorem captain_age (C : ℕ) (W : ℕ) 
  (hW : W = C + 5)
  (h_total_team : total_team_age = 264)
  (h_total_remainders : total_age_remainder = 207)
  (h_combined_age : combined_age_of_captain_and_keeper = 57) :
  C = 26 :=
by sorry

end captain_age_l180_180316


namespace totalBirdsOnFence_l180_180006

/-
Statement: Given initial birds and additional birds joining, the total number
           of birds sitting on the fence is 10.
Conditions:
1. Initially, there are 4 birds.
2. 6 more birds join them.
3. There are 46 storks on the fence, but they do not affect the number of birds.
-/

def initialBirds : Nat := 4
def additionalBirds : Nat := 6

theorem totalBirdsOnFence : initialBirds + additionalBirds = 10 := by
  sorry

end totalBirdsOnFence_l180_180006


namespace hockey_games_in_season_l180_180623

-- Define the conditions
def games_per_month : Nat := 13
def season_months : Nat := 14

-- Define the total number of hockey games in the season
def total_games_in_season (games_per_month : Nat) (season_months : Nat) : Nat :=
  games_per_month * season_months

-- Define the theorem to prove
theorem hockey_games_in_season :
  total_games_in_season games_per_month season_months = 182 :=
by
  -- Proof omitted
  sorry

end hockey_games_in_season_l180_180623


namespace difference_eq_neg_subtrahend_implies_minuend_zero_l180_180128

theorem difference_eq_neg_subtrahend_implies_minuend_zero {x y : ℝ} (h : x - y = -y) : x = 0 :=
sorry

end difference_eq_neg_subtrahend_implies_minuend_zero_l180_180128


namespace triangle_area_is_4_l180_180577

variable {PQ RS : ℝ} -- lengths of PQ and RS respectively
variable {area_PQRS area_PQS : ℝ} -- areas of the trapezoid and triangle respectively

-- Given conditions
@[simp]
def trapezoid_area_is_12 (area_PQRS : ℝ) : Prop :=
  area_PQRS = 12

@[simp]
def RS_is_twice_PQ (PQ RS : ℝ) : Prop :=
  RS = 2 * PQ

-- To prove: the area of triangle PQS is 4 given the conditions
theorem triangle_area_is_4 (h1 : trapezoid_area_is_12 area_PQRS)
                          (h2 : RS_is_twice_PQ PQ RS)
                          (h3 : area_PQRS = 3 * area_PQS) : area_PQS = 4 :=
by
  sorry

end triangle_area_is_4_l180_180577


namespace fans_with_all_vouchers_l180_180196

theorem fans_with_all_vouchers (total_fans : ℕ) 
    (soda_interval : ℕ) (popcorn_interval : ℕ) (hotdog_interval : ℕ)
    (h1 : soda_interval = 60) (h2 : popcorn_interval = 80) (h3 : hotdog_interval = 100)
    (h4 : total_fans = 4500)
    (h5 : Nat.lcm soda_interval (Nat.lcm popcorn_interval hotdog_interval) = 1200) :
    (total_fans / Nat.lcm soda_interval (Nat.lcm popcorn_interval hotdog_interval)) = 3 := 
by
    sorry

end fans_with_all_vouchers_l180_180196


namespace female_athletes_in_sample_l180_180193

theorem female_athletes_in_sample (total_athletes : ℕ) (male_athletes : ℕ) (sample_size : ℕ)
  (total_athletes_eq : total_athletes = 98)
  (male_athletes_eq : male_athletes = 56)
  (sample_size_eq : sample_size = 28)
  : (sample_size * (total_athletes - male_athletes) / total_athletes) = 12 :=
by
  sorry

end female_athletes_in_sample_l180_180193


namespace first_term_to_common_difference_ratio_l180_180153

theorem first_term_to_common_difference_ratio (a d : ℝ) 
  (h : (14 / 2) * (2 * a + 13 * d) = 3 * (7 / 2) * (2 * a + 6 * d)) :
  a / d = 4 :=
by
  sorry

end first_term_to_common_difference_ratio_l180_180153


namespace factorize_poly_l180_180816

theorem factorize_poly (x : ℝ) : 4 * x^3 - x = x * (2 * x + 1) * (2 * x - 1) := by
  sorry

end factorize_poly_l180_180816


namespace max_marks_is_400_l180_180373

-- Given conditions
def passing_mark (M : ℝ) : ℝ := 0.30 * M
def student_marks : ℝ := 80
def marks_failed_by : ℝ := 40
def pass_marks : ℝ := student_marks + marks_failed_by

-- Statement to prove
theorem max_marks_is_400 (M : ℝ) (h : passing_mark M = pass_marks) : M = 400 :=
by sorry

end max_marks_is_400_l180_180373


namespace simplified_equation_equivalent_l180_180836

theorem simplified_equation_equivalent  (x : ℝ) :
    (x / 0.3 = 1 + (1.2 - 0.3 * x) / 0.2) ↔ (10 * x / 3 = 1 + (12 - 3 * x) / 2) :=
by sorry

end simplified_equation_equivalent_l180_180836


namespace bailey_dog_treats_l180_180375

-- Definitions based on conditions
def total_charges_per_card : Nat := 5
def number_of_cards : Nat := 4
def chew_toys : Nat := 2
def rawhide_bones : Nat := 10

-- Total number of items bought
def total_items : Nat := total_charges_per_card * number_of_cards

-- Definition of the number of dog treats
def dog_treats : Nat := total_items - (chew_toys + rawhide_bones)

-- Theorem to prove the number of dog treats
theorem bailey_dog_treats : dog_treats = 8 := by
  -- Proof is skipped with sorry
  sorry

end bailey_dog_treats_l180_180375


namespace fraction_simplification_l180_180678

theorem fraction_simplification :
  (3 / (2 - (3 / 4))) = 12 / 5 := 
by
  sorry

end fraction_simplification_l180_180678


namespace sine_central_angle_of_circular_sector_eq_4_5_l180_180261

theorem sine_central_angle_of_circular_sector_eq_4_5
  (R : Real)
  (α : Real)
  (h : π * R ^ 2 * Real.sin α = 2 * π * R ^ 2 * (1 - Real.cos α)) :
  Real.sin α = 4 / 5 := by
  sorry

end sine_central_angle_of_circular_sector_eq_4_5_l180_180261


namespace geometric_sequence_sum_is_9_l180_180618

theorem geometric_sequence_sum_is_9 {a : ℕ → ℝ} (q : ℝ) 
  (h3a7 : a 3 * a 7 = 8) 
  (h4a6 : a 4 + a 6 = 6)
  (h_geom : ∀ n, a (n + 1) = a n * q) : a 2 + a 8 = 9 :=
sorry

end geometric_sequence_sum_is_9_l180_180618


namespace original_price_l180_180884

theorem original_price (P : ℝ) 
  (h1 : 1.40 * P = P + 700) : P = 1750 :=
by sorry

end original_price_l180_180884


namespace negation_equivalence_l180_180171

-- Declare the condition for real solutions of a quadratic equation
def has_real_solutions (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + a * x + 1 = 0

-- Define the proposition p
def prop_p : Prop :=
  ∀ a : ℝ, a ≥ 0 → has_real_solutions a

-- Define the negation of p
def neg_prop_p : Prop :=
  ∃ a : ℝ, a ≥ 0 ∧ ¬ has_real_solutions a

-- The theorem stating the equivalence of p's negation to its formulated negation.
theorem negation_equivalence : neg_prop_p = ¬ prop_p := by
  sorry

end negation_equivalence_l180_180171


namespace cost_of_notebooks_and_markers_l180_180024

theorem cost_of_notebooks_and_markers 
  (x y : ℝ) 
  (h1 : 3 * x + 2 * y = 7.30) 
  (h2 : 5 * x + 3 * y = 11.65) : 
  2 * x + y = 4.35 :=
by
  sorry

end cost_of_notebooks_and_markers_l180_180024


namespace least_pawns_required_l180_180443

theorem least_pawns_required (n k : ℕ) (h1 : n > 0) (h2 : k > 0) (h3 : 2 * k > n) (h4 : 3 * k ≤ 2 * n) : 
  ∃ (m : ℕ), m = 4 * (n - k) :=
sorry

end least_pawns_required_l180_180443


namespace coin_prob_not_unique_l180_180046

theorem coin_prob_not_unique (p : ℝ) (w : ℝ) (h1 : 0 ≤ p ∧ p ≤ 1) (h2 : w = 144 / 625) :
  ¬ ∃! p, (∃ w, w = 10 * p^3 * (1 - p)^2 ∧ w = 144 / 625) :=
by
  sorry

end coin_prob_not_unique_l180_180046


namespace total_operations_l180_180141

-- Define the process of iterative multiplication and division as described in the problem
def process (start : Nat) : Nat :=
  let m1 := 3 * start
  let m2 := 3 * m1
  let m3 := 3 * m2
  let m4 := 3 * m3
  let m5 := 3 * m4
  let d1 := m5 / 2
  let d2 := d1 / 2
  let d3 := d2 / 2
  let d4 := d3 / 2
  let d5 := d4 / 2
  let d6 := d5 / 2
  let d7 := d6 / 2
  d7

theorem total_operations : process 1 = 1 ∧ 5 + 7 = 12 :=
by
  sorry

end total_operations_l180_180141


namespace r_minus_p_value_l180_180656

theorem r_minus_p_value (p q r : ℝ)
  (h₁ : (p + q) / 2 = 10)
  (h₂ : (q + r) / 2 = 22) :
  r - p = 24 :=
by
  sorry

end r_minus_p_value_l180_180656


namespace melinda_doughnuts_picked_l180_180502

theorem melinda_doughnuts_picked :
  (∀ d h_coffee m_coffee : ℕ, d = 3 → h_coffee = 4 → m_coffee = 6 →
    ∀ cost_d cost_h cost_m : ℝ, cost_d = 0.45 → 
    cost_h = 4.91 → cost_m = 7.59 → 
    ∃ m_doughnuts : ℕ, cost_m - m_coffee * ((cost_h - d * cost_d) / h_coffee) = m_doughnuts * cost_d) → 
  ∃ n : ℕ, n = 5 := 
by sorry

end melinda_doughnuts_picked_l180_180502


namespace michael_truck_meetings_2_times_l180_180422

/-- Michael walks at a rate of 6 feet per second on a straight path. 
Trash pails are placed every 240 feet along the path. 
A garbage truck traveling at 12 feet per second in the same direction stops for 40 seconds at each pail. 
When Michael passes a pail, he sees the truck, which is 240 feet ahead, just leaving the next pail. 
Prove that Michael and the truck will meet exactly 2 times. -/

def michael_truck_meetings (v_michael v_truck d_pail t_stop init_michael init_truck : ℕ) : ℕ := sorry

theorem michael_truck_meetings_2_times :
  michael_truck_meetings 6 12 240 40 0 240 = 2 := 
  sorry

end michael_truck_meetings_2_times_l180_180422


namespace sheetrock_width_l180_180378

theorem sheetrock_width (l A w : ℕ) (h_length : l = 6) (h_area : A = 30) (h_formula : A = l * w) : w = 5 :=
by
  -- Placeholder for the proof
  sorry

end sheetrock_width_l180_180378


namespace jayden_planes_l180_180883

theorem jayden_planes (W : ℕ) (wings_per_plane : ℕ) (total_wings : W = 108) (wpp_pos : wings_per_plane = 2) :
  ∃ n : ℕ, n = W / wings_per_plane ∧ n = 54 :=
by
  sorry

end jayden_planes_l180_180883


namespace percentage_students_receive_valentine_l180_180424

/-- Given the conditions:
  1. There are 30 students.
  2. Mo wants to give a Valentine to some percentage of them.
  3. Each Valentine costs $2.
  4. Mo has $40.
  5. Mo will spend 90% of his money on Valentines.
Prove that the percentage of students receiving a Valentine is 60%.
-/
theorem percentage_students_receive_valentine :
  let total_students := 30
  let valentine_cost := 2
  let total_money := 40
  let spent_percentage := 0.90
  ∃ (cards : ℕ), 
    let money_spent := total_money * spent_percentage
    let cards_bought := money_spent / valentine_cost
    let percentage_students := (cards_bought / total_students) * 100
    percentage_students = 60 := 
by
  sorry

end percentage_students_receive_valentine_l180_180424


namespace correct_calculation_l180_180886

theorem correct_calculation (x : ℝ) (h : (x / 2) + 45 = 85) : (2 * x) - 45 = 115 :=
by {
  -- Note: Proof steps are not needed, 'sorry' is used to skip the proof
  sorry
}

end correct_calculation_l180_180886


namespace expression_value_l180_180260

theorem expression_value (a b m n : ℚ) 
  (ha : a = -7/4) 
  (hb : b = -2/3) 
  (hmn : m + n = 0) : 
  4 * a / b + 3 * (m + n) = 21 / 2 :=
by 
  sorry

end expression_value_l180_180260


namespace san_antonio_bus_passes_4_austin_buses_l180_180048

theorem san_antonio_bus_passes_4_austin_buses :
  ∀ (hourly_austin_buses : ℕ → ℕ) (every_50_minute_san_antonio_buses : ℕ → ℕ) (trip_time : ℕ),
    (∀ h : ℕ, hourly_austin_buses (h) = (h * 60)) →
    (∀ m : ℕ, every_50_minute_san_antonio_buses (m) = (m * 60 + 50)) →
    trip_time = 240 →
    ∃ num_buses_passed : ℕ, num_buses_passed = 4 :=
by
  sorry

end san_antonio_bus_passes_4_austin_buses_l180_180048


namespace percentage_error_in_area_l180_180104

theorem percentage_error_in_area (s : ℝ) (h : s > 0) :
  let s' := 1.02 * s
  let A := s ^ 2
  let A' := s' ^ 2
  let error := A' - A
  let percent_error := (error / A) * 100
  percent_error = 4.04 := by
  sorry

end percentage_error_in_area_l180_180104


namespace cos_alpha_minus_11pi_div_12_eq_neg_2_div_3_l180_180064

theorem cos_alpha_minus_11pi_div_12_eq_neg_2_div_3
  (α : ℝ)
  (h : Real.sin (7 * Real.pi / 12 + α) = 2 / 3) :
  Real.cos (α - 11 * Real.pi / 12) = -(2 / 3) :=
by
  sorry

end cos_alpha_minus_11pi_div_12_eq_neg_2_div_3_l180_180064


namespace baker_remaining_cakes_l180_180462

def initial_cakes : ℝ := 167.3
def sold_cakes : ℝ := 108.2
def remaining_cakes : ℝ := initial_cakes - sold_cakes

theorem baker_remaining_cakes : remaining_cakes = 59.1 := by
  sorry

end baker_remaining_cakes_l180_180462


namespace total_clouds_count_l180_180999

def carson_clouds := 12
def little_brother_clouds := 5 * carson_clouds
def older_sister_clouds := carson_clouds / 2
def cousin_clouds := 2 * (older_sister_clouds + carson_clouds)

theorem total_clouds_count : carson_clouds + little_brother_clouds + older_sister_clouds + cousin_clouds = 114 := by
  sorry

end total_clouds_count_l180_180999


namespace unique_value_of_W_l180_180750

theorem unique_value_of_W (T O W F U R : ℕ) (h1 : T = 8) (h2 : O % 2 = 0) (h3 : ∀ x y, x ≠ y → x = O → y = T → x ≠ O) :
  (T + T) * 10^2 + (W + W) * 10 + (O + O) = F * 10^3 + O * 10^2 + U * 10 + R → W = 3 :=
by
  sorry

end unique_value_of_W_l180_180750


namespace closest_perfect_square_to_325_is_324_l180_180011

theorem closest_perfect_square_to_325_is_324 :
  ∃ n : ℕ, n^2 = 324 ∧ (∀ m : ℕ, m * m ≠ 325) ∧
    (n = 18 ∧ (∀ k : ℕ, (k*k < 325 ∧ (325 - k*k) > 325 - 324) ∨ 
               (k*k > 325 ∧ (k*k - 325) > 361 - 325))) :=
by
  sorry

end closest_perfect_square_to_325_is_324_l180_180011


namespace age_of_b_l180_180791

theorem age_of_b (A B C : ℕ) (h₁ : (A + B + C) / 3 = 25) (h₂ : (A + C) / 2 = 29) : B = 17 := 
by
  sorry

end age_of_b_l180_180791


namespace part1_part2_l180_180528

theorem part1 (m : ℝ) (h_m_not_zero : m ≠ 0) : m ≤ 4 / 3 :=
by
  -- The proof would go here, but we are inserting sorry to skip it.
  sorry

theorem part2 (m : ℕ) (h_m_range : m = 1) :
  ∃ x1 x2 : ℝ, (m * x1^2 - 4 * x1 + 3 = 0) ∧ (m * x2^2 - 4 * x2 + 3 = 0) ∧ x1 = 1 ∧ x2 = 3 :=
by
  -- The proof would go here, but we are inserting sorry to skip it.
  sorry

end part1_part2_l180_180528


namespace fifth_number_in_21st_row_l180_180702

theorem fifth_number_in_21st_row : 
  let nth_odd_number (n : ℕ) := 2 * n - 1 
  let sum_first_n_rows (n : ℕ) := n * (n + (n - 1))
  nth_odd_number 405 = 809 := 
by
  sorry

end fifth_number_in_21st_row_l180_180702


namespace sampling_method_is_systematic_l180_180567

-- Definition of the conditions
def factory_produces_product := True  -- Assuming the factory is always producing
def uses_conveyor_belt := True  -- Assuming the conveyor belt is always in use
def samples_taken_every_10_minutes := True  -- Sampling at specific intervals

-- Definition corresponding to the systematic sampling
def systematic_sampling := True

-- Theorem: Prove that given the conditions, the sampling method is systematic sampling.
theorem sampling_method_is_systematic :
  factory_produces_product → uses_conveyor_belt → samples_taken_every_10_minutes → systematic_sampling :=
by
  intros _ _ _
  trivial

end sampling_method_is_systematic_l180_180567


namespace system_inconsistent_l180_180215

-- Define the coefficient matrix and the augmented matrices.
def A : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![1, -2, 3], ![2, 3, -1], ![3, 1, 2]]

def B1 : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![5, -2, 3], ![7, 3, -1], ![10, 1, 2]]

-- Calculate the determinants.
noncomputable def delta : ℤ := A.det
noncomputable def delta1 : ℤ := B1.det

-- The main theorem statement: the system is inconsistent if Δ = 0 and Δ1 ≠ 0.
theorem system_inconsistent (h₁ : delta = 0) (h₂ : delta1 ≠ 0) : False :=
sorry

end system_inconsistent_l180_180215


namespace john_total_replacement_cost_l180_180674

def cost_to_replace_all_doors
  (num_bedroom_doors : ℕ)
  (num_outside_doors : ℕ)
  (cost_outside_door : ℕ)
  (cost_bedroom_door : ℕ) : ℕ :=
  let total_cost_outside_doors := num_outside_doors * cost_outside_door
  let total_cost_bedroom_doors := num_bedroom_doors * cost_bedroom_door
  total_cost_outside_doors + total_cost_bedroom_doors

theorem john_total_replacement_cost :
  let num_bedroom_doors := 3
  let num_outside_doors := 2
  let cost_outside_door := 20
  let cost_bedroom_door := cost_outside_door / 2
  cost_to_replace_all_doors num_bedroom_doors num_outside_doors cost_outside_door cost_bedroom_door = 70 := by
  sorry

end john_total_replacement_cost_l180_180674


namespace value_of_m_has_positive_root_l180_180878

theorem value_of_m_has_positive_root (x m : ℝ) (hx : x ≠ 3) :
    ((x + 5) / (x - 3) = 2 - m / (3 - x)) → x > 0 → m = 8 := 
sorry

end value_of_m_has_positive_root_l180_180878


namespace cream_ratio_l180_180205

theorem cream_ratio (j : ℝ) (jo : ℝ) (jc : ℝ) (joc : ℝ) (jdrank : ℝ) (jodrank : ℝ) :
  j = 15 ∧ jo = 15 ∧ jc = 3 ∧ joc = 2.5 ∧ jdrank = 0 ∧ jodrank = 0.5 →
  j + jc - jdrank = jc ∧ jo + jc - jodrank = joc →
  (jc / joc) = (6 / 5) :=
  by
  sorry

end cream_ratio_l180_180205


namespace remainder_of_3_pow_2023_mod_7_l180_180784

theorem remainder_of_3_pow_2023_mod_7 :
  3 ^ 2023 % 7 = 3 :=
by
  sorry

end remainder_of_3_pow_2023_mod_7_l180_180784


namespace train_length_is_50_meters_l180_180129

theorem train_length_is_50_meters
  (L : ℝ)
  (equal_length : ∀ (a b : ℝ), a = L ∧ b = L → a + b = 2 * L)
  (speed_faster_train : ℝ := 46) -- km/hr
  (speed_slower_train : ℝ := 36) -- km/hr
  (relative_speed : ℝ := speed_faster_train - speed_slower_train)
  (relative_speed_km_per_sec : ℝ := relative_speed / 3600) -- converting km/hr to km/sec
  (time : ℝ := 36) -- seconds
  (distance_covered : ℝ := 2 * L)
  (distance_eq : distance_covered = relative_speed_km_per_sec * time):
  L = 50 / 1000 :=
by 
  -- We will prove it as per the derived conditions
  sorry

end train_length_is_50_meters_l180_180129


namespace transform_f_to_shift_left_l180_180194

theorem transform_f_to_shift_left (f : ℝ → ℝ) :
  ∀ x : ℝ, f (2 * x - 1) = f (2 * (x - 1) + 1) := by
  sorry

end transform_f_to_shift_left_l180_180194


namespace find_x_l180_180396

variable (c d : ℝ)

theorem find_x (x : ℝ) (h : x^2 + 4 * c^2 = (3 * d - x)^2) : 
  x = (9 * d^2 - 4 * c^2) / (6 * d) :=
sorry

end find_x_l180_180396


namespace minimum_p_for_required_profit_l180_180510

noncomputable def profit (x p : ℝ) : ℝ := p * x - (0.5 * x^2 - 2 * x - 10)
noncomputable def max_profit (p : ℝ) : ℝ := (p + 2)^2 / 2 + 10

theorem minimum_p_for_required_profit : ∀ (p : ℝ), 3 * max_profit p >= 126 → p >= 6 :=
by
  intro p
  unfold max_profit
  -- Given:  3 * ((p + 2)^2 / 2 + 10) >= 126
  sorry

end minimum_p_for_required_profit_l180_180510


namespace fgf_3_is_299_l180_180789

def f (x : ℕ) : ℕ := 5 * x + 4
def g (x : ℕ) : ℕ := 3 * x + 2
def h : ℕ := 3

theorem fgf_3_is_299 : f (g (f h)) = 299 :=
by
  sorry

end fgf_3_is_299_l180_180789


namespace inscribed_circle_radius_range_l180_180556

noncomputable def r_range (AD DB : ℝ) (angle_A : ℝ) : Set ℝ :=
  { r | 0 < r ∧ r < (-3 + Real.sqrt 33) / 2 }

theorem inscribed_circle_radius_range (AD DB : ℝ) (angle_A : ℝ) (h1 : AD = 2 * Real.sqrt 3) 
    (h2 : DB = Real.sqrt 3) (h3 : angle_A > 60) : 
    r_range AD DB angle_A = { r | 0 < r ∧ r < (-3 + Real.sqrt 33) / 2 } :=
by
  sorry

end inscribed_circle_radius_range_l180_180556


namespace matrix_addition_l180_180699

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℤ := ![![1, 0], ![-1, 2]]
def B : Matrix (Fin 2) (Fin 2) ℤ := ![![2, 4], ![1, -3]]
def C : Matrix (Fin 2) (Fin 2) ℤ := ![![3, 4], ![0, -1]]

theorem matrix_addition : A + B = C := by
    sorry

end matrix_addition_l180_180699


namespace max_cos_alpha_l180_180898

open Real

-- Define the condition as a hypothesis
def cos_sum_eq (α β : ℝ) : Prop :=
  cos (α + β) = cos α + cos β

-- State the maximum value theorem
theorem max_cos_alpha (α β : ℝ) (h : cos_sum_eq α β) : ∃ α, cos α = sqrt 3 - 1 :=
by
  sorry   -- Proof is omitted

#check max_cos_alpha

end max_cos_alpha_l180_180898


namespace no_natural_number_n_exists_l180_180327

theorem no_natural_number_n_exists (n : ℕ) :
  ¬ ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ x^2 + y^2 = 2 * n * (n + 1) * (n + 2) * (n + 3) + 12 := 
sorry

end no_natural_number_n_exists_l180_180327


namespace ellipse_foci_distance_l180_180438

theorem ellipse_foci_distance (a b : ℝ) (ha : a = 10) (hb : b = 8) :
  2 * Real.sqrt (a^2 - b^2) = 12 :=
by
  rw [ha, hb]
  -- Proof follows here, but we skip it using sorry.
  sorry

end ellipse_foci_distance_l180_180438


namespace lcm_of_denominators_l180_180299

theorem lcm_of_denominators (x : ℕ) [NeZero x] : Nat.lcm (Nat.lcm x (2 * x)) (3 * x^2) = 6 * x^2 :=
by
  sorry

end lcm_of_denominators_l180_180299


namespace abs_x_plus_abs_y_eq_one_area_l180_180647

theorem abs_x_plus_abs_y_eq_one_area : 
  (∃ (A : ℝ), ∀ (x y : ℝ), |x| + |y| = 1 → A = 2) :=
sorry

end abs_x_plus_abs_y_eq_one_area_l180_180647


namespace planes_are_perpendicular_l180_180269

-- Define the normal vectors
def N1 : List ℝ := [2, 3, -4]
def N2 : List ℝ := [5, -2, 1]

-- Define the dot product function
def dotProduct (v1 v2 : List ℝ) : ℝ :=
  List.zipWith (fun a b => a * b) v1 v2 |>.sum

-- State the theorem
theorem planes_are_perpendicular :
  dotProduct N1 N2 = 0 :=
by
  sorry

end planes_are_perpendicular_l180_180269


namespace cheapest_store_for_60_balls_l180_180161

def cost_store_A (n : ℕ) (price_per_ball : ℕ) (free_per_10 : ℕ) : ℕ :=
  if n < 10 then n * price_per_ball
  else (n / 10) * 10 * price_per_ball + (n % 10) * price_per_ball * (n / (10 + free_per_10))

def cost_store_B (n : ℕ) (discount : ℕ) (price_per_ball : ℕ) : ℕ :=
  n * (price_per_ball - discount)

def cost_store_C (n : ℕ) (price_per_ball : ℕ) (cashback_threshold cashback_amt : ℕ) : ℕ :=
  let initial_cost := n * price_per_ball
  let cashback := (initial_cost / cashback_threshold) * cashback_amt
  initial_cost - cashback

theorem cheapest_store_for_60_balls
  (price_per_ball discount free_per_10 cashback_threshold cashback_amt : ℕ) :
  cost_store_A 60 price_per_ball free_per_10 = 1250 →
  cost_store_B 60 discount price_per_ball = 1200 →
  cost_store_C 60 price_per_ball cashback_threshold cashback_amt = 1290 →
  min (cost_store_A 60 price_per_ball free_per_10) (min (cost_store_B 60 discount price_per_ball) (cost_store_C 60 price_per_ball cashback_threshold cashback_amt))
  = 1200 :=
by
  sorry

end cheapest_store_for_60_balls_l180_180161


namespace least_number_to_add_l180_180700

-- Definition of LCM for given primes
def lcm_of_primes : ℕ := 5 * 7 * 11 * 13 * 17 * 19

theorem least_number_to_add (n : ℕ) : 
  (5432 + n) % 5 = 0 ∧ 
  (5432 + n) % 7 = 0 ∧ 
  (5432 + n) % 11 = 0 ∧ 
  (5432 + n) % 13 = 0 ∧ 
  (5432 + n) % 17 = 0 ∧ 
  (5432 + n) % 19 = 0 ↔ 
  n = 1611183 :=
by sorry

end least_number_to_add_l180_180700


namespace questions_for_second_project_l180_180729

open Nat

theorem questions_for_second_project (days_per_week : ℕ) (first_project_q : ℕ) (questions_per_day : ℕ) 
  (total_questions : ℕ) (second_project_q : ℕ) 
  (h1 : days_per_week = 7)
  (h2 : first_project_q = 518)
  (h3 : questions_per_day = 142)
  (h4 : total_questions = days_per_week * questions_per_day)
  (h5 : second_project_q = total_questions - first_project_q) :
  second_project_q = 476 :=
by
  -- we assume the solution steps as correct
  sorry

end questions_for_second_project_l180_180729


namespace relationship_y1_y2_y3_l180_180236

theorem relationship_y1_y2_y3 :
  ∀ (y1 y2 y3 : ℝ), y1 = 6 ∧ y2 = 3 ∧ y3 = -2 → y1 > y2 ∧ y2 > y3 :=
by 
  intros y1 y2 y3 h
  sorry

end relationship_y1_y2_y3_l180_180236


namespace shortest_side_of_similar_triangle_l180_180180

theorem shortest_side_of_similar_triangle (h1 : ∀ (a b c : ℝ), a^2 + b^2 = c^2)
  (h2 : 15^2 + b^2 = 34^2) (h3 : ∃ (k : ℝ), k = 68 / 34) :
  ∃ s : ℝ, s = 2 * Real.sqrt 931 :=
by
  sorry

end shortest_side_of_similar_triangle_l180_180180


namespace base10_to_base7_of_804_l180_180829

def base7 (n : ℕ) : ℕ :=
  let d3 := n / 343
  let r3 := n % 343
  let d2 := r3 / 49
  let r2 := r3 % 49
  let d1 := r2 / 7
  let d0 := r2 % 7
  d3 * 1000 + d2 * 100 + d1 * 10 + d0

theorem base10_to_base7_of_804 :
  base7 804 = 2226 :=
by
  -- Proof to be filled in.
  sorry

end base10_to_base7_of_804_l180_180829


namespace graph_is_hyperbola_l180_180746

def graph_equation (x y : ℝ) : Prop := x^2 - 16 * y^2 - 8 * x + 64 = 0

theorem graph_is_hyperbola : ∃ (a b : ℝ), ∀ x y : ℝ, graph_equation x y ↔ (x - a)^2 / 48 - y^2 / 3 = -1 :=
by
  sorry

end graph_is_hyperbola_l180_180746


namespace find_f_2_l180_180150

theorem find_f_2 (f : ℕ → ℕ) (h : ∀ x, f (x + 1) = 2 * x + 3) : f 2 = 5 :=
sorry

end find_f_2_l180_180150


namespace find_minimum_value_l180_180096

open Real

noncomputable def g (x : ℝ) : ℝ := 
  x + x / (x^2 + 2) + x * (x + 3) / (x^2 + 3) + 3 * (x + 1) / (x * (x^2 + 3))

theorem find_minimum_value (x : ℝ) (hx : x > 0) : g x ≥ 4 := 
sorry

end find_minimum_value_l180_180096


namespace haley_extra_tickets_l180_180268

theorem haley_extra_tickets (cost_per_ticket : ℤ) (tickets_bought_for_self_and_friends : ℤ) (total_spent : ℤ) 
    (h1 : cost_per_ticket = 4) (h2 : tickets_bought_for_self_and_friends = 3) (h3 : total_spent = 32) : 
    (total_spent / cost_per_ticket) - tickets_bought_for_self_and_friends = 5 :=
by
  sorry

end haley_extra_tickets_l180_180268


namespace rhombus_area_l180_180264

def d1 : ℝ := 10
def d2 : ℝ := 30

theorem rhombus_area (d1 d2 : ℝ) : (d1 * d2) / 2 = 150 := by
  sorry

end rhombus_area_l180_180264


namespace smallest_n_l180_180949

theorem smallest_n :
  ∃ n : ℕ, n > 0 ∧ 2000 * n % 21 = 0 ∧ ∀ m : ℕ, m > 0 ∧ 2000 * m % 21 = 0 → n ≤ m :=
sorry

end smallest_n_l180_180949


namespace c_positive_when_others_negative_l180_180711

variables {a b c d e f : ℤ}

theorem c_positive_when_others_negative (h_ab_cdef_lt_0 : a * b + c * d * e * f < 0)
  (h_a_neg : a < 0) (h_b_neg : b < 0) (h_d_neg : d < 0) (h_e_neg : e < 0) (h_f_neg : f < 0) 
  : c > 0 :=
sorry

end c_positive_when_others_negative_l180_180711


namespace toothpicks_15_l180_180781

def toothpicks (n : ℕ) : ℕ :=
  match n with
  | 0 => 0  -- Not used, placeholder for 1-based indexing.
  | 1 => 3
  | k+1 => let p := toothpicks k
           2 + if k % 2 = 0 then 1 else 0 + p

theorem toothpicks_15 : toothpicks 15 = 38 :=
by
  sorry

end toothpicks_15_l180_180781


namespace arithmetic_sequence_30th_term_l180_180679

-- Defining the initial term and the common difference of the arithmetic sequence
def a : ℕ := 3
def d : ℕ := 4

-- Defining the general formula for the n-th term of the arithmetic sequence
def a_n (n : ℕ) : ℕ := a + (n - 1) * d

-- Theorem stating that the 30th term of the given sequence is 119
theorem arithmetic_sequence_30th_term : a_n 30 = 119 := 
sorry

end arithmetic_sequence_30th_term_l180_180679


namespace apples_in_each_bag_l180_180636

variable (x : ℕ)
variable (total_children : ℕ)
variable (eaten_apples : ℕ)
variable (sold_apples : ℕ)
variable (remaining_apples : ℕ)

theorem apples_in_each_bag
  (h1 : total_children = 5)
  (h2 : eaten_apples = 2 * 4)
  (h3 : sold_apples = 7)
  (h4 : remaining_apples = 60)
  (h5 : total_children * x - eaten_apples - sold_apples = remaining_apples) :
  x = 15 :=
by
  sorry

end apples_in_each_bag_l180_180636


namespace fruit_box_assignment_proof_l180_180286

-- Definitions of the boxes with different fruits
inductive Fruit | Apple | Pear | Orange | Banana
open Fruit

-- Define a function representing the placement of fruits in the boxes
def box_assignment := ℕ → Fruit

-- Conditions based on the problem statement
def conditions (assign : box_assignment) : Prop :=
  assign 1 ≠ Orange ∧
  assign 2 ≠ Pear ∧
  (assign 1 = Banana → assign 3 ≠ Apple ∧ assign 3 ≠ Pear) ∧
  assign 4 ≠ Apple

-- The correct assignment of fruits to boxes
def correct_assignment (assign : box_assignment) : Prop :=
  assign 1 = Banana ∧
  assign 2 = Apple ∧
  assign 3 = Orange ∧
  assign 4 = Pear

-- Theorem statement
theorem fruit_box_assignment_proof : ∃ assign : box_assignment, conditions assign ∧ correct_assignment assign :=
sorry

end fruit_box_assignment_proof_l180_180286


namespace total_investment_amount_l180_180221

theorem total_investment_amount 
    (x : ℝ) 
    (h1 : 6258.0 * 0.08 + x * 0.065 = 678.87) : 
    x + 6258.0 = 9000.0 :=
sorry

end total_investment_amount_l180_180221


namespace line_equation_l180_180624

theorem line_equation (x y : ℝ) : 
  (∃ (m c : ℝ), m = 3 ∧ c = 4 ∧ y = m * x + c) ↔ 3 * x - y + 4 = 0 := by
  sorry

end line_equation_l180_180624


namespace tub_volume_ratio_l180_180958

theorem tub_volume_ratio (C D : ℝ) 
  (h₁ : 0 < C) 
  (h₂ : 0 < D)
  (h₃ : (3/4) * C = (2/3) * D) : 
  C / D = 8 / 9 := 
sorry

end tub_volume_ratio_l180_180958


namespace correct_options_l180_180320

-- Definitions of conditions in Lean 
def is_isosceles (T : Triangle) : Prop := sorry -- Define isosceles triangle
def is_right_angle (T : Triangle) : Prop := sorry -- Define right-angled triangle
def similar (T₁ T₂ : Triangle) : Prop := sorry -- Define similarity of triangles
def equal_vertex_angle (T₁ T₂ : Triangle) : Prop := sorry -- Define equal vertex angle
def equal_base_angle (T₁ T₂ : Triangle) : Prop := sorry -- Define equal base angle

-- Theorem statement to verify correct options (2) and (4)
theorem correct_options {T₁ T₂ : Triangle} :
  (is_right_angle T₁ ∧ is_right_angle T₂ ∧ is_isosceles T₁ ∧ is_isosceles T₂ → similar T₁ T₂) ∧ 
  (equal_vertex_angle T₁ T₂ ∧ is_isosceles T₁ ∧ is_isosceles T₂ → similar T₁ T₂) :=
sorry -- proof not required

end correct_options_l180_180320


namespace range_of_slope_angle_l180_180620

theorem range_of_slope_angle (l : ℝ → ℝ) (theta : ℝ) 
    (h_line_eqn : ∀ x y, l x = y ↔ x - y * Real.sin theta + 2 = 0) : 
    ∃ α : ℝ, α ∈ Set.Icc (Real.pi / 4) (3 * Real.pi / 4) :=
sorry

end range_of_slope_angle_l180_180620


namespace compare_31_17_compare_33_63_compare_82_26_compare_29_80_l180_180190

-- Definition and proof obligation for each comparison

theorem compare_31_17 : 31^11 < 17^14 := sorry

theorem compare_33_63 : 33^75 > 63^60 := sorry

theorem compare_82_26 : 82^33 > 26^44 := sorry

theorem compare_29_80 : 29^31 > 80^23 := sorry

end compare_31_17_compare_33_63_compare_82_26_compare_29_80_l180_180190


namespace kite_area_correct_l180_180955

-- Define the coordinates of the vertices
def vertex1 : (ℤ × ℤ) := (3, 0)
def vertex2 : (ℤ × ℤ) := (0, 5)
def vertex3 : (ℤ × ℤ) := (3, 7)
def vertex4 : (ℤ × ℤ) := (6, 5)

-- Define the area of a kite using the Shoelace formula for a quadrilateral
-- with given vertices
def kite_area (v1 v2 v3 v4 : ℤ × ℤ) : ℤ :=
  let (x1, y1) := v1
  let (x2, y2) := v2
  let (x3, y3) := v3
  let (x4, y4) := v4
  (abs ((x1 * y2 + x2 * y3 + x3 * y4 + x4 * y1) - (y1 * x2 + y2 * x3 + y3 * x4 + y4 * x1))) / 2

theorem kite_area_correct : kite_area vertex1 vertex2 vertex3 vertex4 = 21 := 
  sorry

end kite_area_correct_l180_180955


namespace sum_of_real_solutions_l180_180704

theorem sum_of_real_solutions:
  (∃ (s : ℝ), ∀ x : ℝ, 
    (x - 3) / (x^2 + 6 * x + 2) = (x - 6) / (x^2 - 12 * x) → 
    s = 106 / 9) :=
  sorry

end sum_of_real_solutions_l180_180704


namespace range_of_a_minimum_value_of_b_l180_180103

def is_fixed_point (f : ℝ → ℝ) (x₀ : ℝ) : Prop := f x₀ = x₀

noncomputable def f (a b x : ℝ) : ℝ := a * x^2 + (2 * b - 1) * x + b - 2
noncomputable def g (a x : ℝ) : ℝ := -x + a / (3 * a^2 - 2 * a + 1)

theorem range_of_a (h : ∀ b : ℝ, ∃ x1 x2 : ℝ, is_fixed_point (f a b) x1 ∧ is_fixed_point (f a b) x2) : 0 < a ∧ a < 4 :=
sorry

theorem minimum_value_of_b (hx1 : is_fixed_point (f a b) x₁) (hx2 : is_fixed_point (f a b) x₂)
  (hm : g a ((x₁ + x₂) / 2) = (x₁ + x₂) / 2) (ha : 0 < a ∧ a < 4) : b ≥ 3/4 :=
sorry

end range_of_a_minimum_value_of_b_l180_180103


namespace determine_function_l180_180274

theorem determine_function (f : ℕ → ℕ) :
  (∀ a b c d : ℕ, 2 * a * b = c^2 + d^2 → f (a + b) = f a + f b + f c + f d) →
  ∃ k : ℕ, ∀ n : ℕ, f n = k * n^2 :=
by
  sorry

end determine_function_l180_180274


namespace range_of_a_l180_180790

noncomputable def f (a x : ℝ) := (Real.exp x - a * x^2) 

theorem range_of_a (a : ℝ) :
  (∀ (x : ℝ), 0 ≤ x → f a x ≥ x + 1) ↔ a ∈ Set.Iic (1/2) :=
by
  sorry

end range_of_a_l180_180790


namespace marcia_minutes_worked_l180_180874

/--
If Marcia worked for 5 hours on her science project,
then she worked for 300 minutes.
-/
theorem marcia_minutes_worked (hours : ℕ) (h : hours = 5) : (hours * 60) = 300 := by
  sorry

end marcia_minutes_worked_l180_180874


namespace valid_permutations_l180_180397

theorem valid_permutations (a : Fin 101 → ℕ) :
  (∀ k, a k ≥ 2 ∧ a k ≤ 102 ∧ (∃ j, a j = k + 2)) →
  (∀ k, a (k + 1) % (k + 1) = 0) →
  (∃ cycles : List (List ℕ), cycles = [[1, 102], [1, 2, 102], [1, 3, 102], [1, 6, 102], [1, 17, 102], [1, 34, 102], 
                                       [1, 51, 102], [1, 2, 6, 102], [1, 2, 34, 102], [1, 3, 6, 102], [1, 3, 51, 102], 
                                       [1, 17, 34, 102], [1, 17, 51, 102]]) :=
sorry

end valid_permutations_l180_180397


namespace oranges_left_to_be_sold_l180_180728

-- Defining the initial conditions
def seven_dozen_oranges : ℕ := 7 * 12
def reserved_for_friend (total : ℕ) : ℕ := total / 4
def remaining_after_reserve (total reserved : ℕ) : ℕ := total - reserved
def sold_yesterday (remaining : ℕ) : ℕ := 3 * remaining / 7
def remaining_after_sale (remaining sold : ℕ) : ℕ := remaining - sold
def remaining_after_rotten (remaining : ℕ) : ℕ := remaining - 4

-- Statement to prove
theorem oranges_left_to_be_sold (total reserved remaining sold final : ℕ) :
  total = seven_dozen_oranges →
  reserved = reserved_for_friend total →
  remaining = remaining_after_reserve total reserved →
  sold = sold_yesterday remaining →
  final = remaining_after_sale remaining sold - 4 →
  final = 32 :=
by
  sorry

end oranges_left_to_be_sold_l180_180728


namespace no_integer_b_for_four_integer_solutions_l180_180581

theorem no_integer_b_for_four_integer_solutions :
  ∀ (b : ℤ), ¬ ∃ x1 x2 x3 x4 : ℤ, 
    (x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4) ∧
    (∀ x : ℤ, (x^2 + b*x + 1 ≤ 0) ↔ (x = x1 ∨ x = x2 ∨ x = x3 ∨ x = x4)) :=
by sorry

end no_integer_b_for_four_integer_solutions_l180_180581


namespace jack_additional_sweets_is_correct_l180_180749

/-- Initial number of sweets --/
def initial_sweets : ℕ := 22

/-- Sweets taken by Paul --/
def sweets_taken_by_paul : ℕ := 7

/-- Jack's total sweets taken --/
def jack_total_sweets_taken : ℕ := initial_sweets - sweets_taken_by_paul

/-- Half of initial sweets --/
def half_initial_sweets : ℕ := initial_sweets / 2

/-- Additional sweets taken by Jack --/
def additional_sweets_taken_by_jack : ℕ := jack_total_sweets_taken - half_initial_sweets

theorem jack_additional_sweets_is_correct : additional_sweets_taken_by_jack = 4 := by
  sorry

end jack_additional_sweets_is_correct_l180_180749


namespace people_dislike_both_radio_and_music_l180_180470

theorem people_dislike_both_radio_and_music :
  let total_people := 1500
  let dislike_radio_percent := 0.35
  let dislike_both_percent := 0.20
  let dislike_radio := dislike_radio_percent * total_people
  let dislike_both := dislike_both_percent * dislike_radio
  dislike_both = 105 :=
by
  sorry

end people_dislike_both_radio_and_music_l180_180470


namespace rational_equation_solutions_l180_180937

open Real

theorem rational_equation_solutions :
  (∃ x : ℝ, (x ≠ 1 ∧ x ≠ -1) ∧ ((x^2 - 6*x + 9) / (x - 1) - (3 - x) / (x^2 - 1) = 0)) →
  ∃ S : Finset ℝ, S.card = 2 ∧ ∀ x ∈ S, (x ≠ 1 ∧ x ≠ -1) :=
by
  sorry

end rational_equation_solutions_l180_180937


namespace combined_sum_is_115_over_3_l180_180040

def geometric_series_sum (a : ℚ) (r : ℚ) : ℚ :=
  if h : abs r < 1 then a / (1 - r) else 0

def arithmetic_series_sum (a : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  n * (2 * a + (n - 1) * d) / 2

noncomputable def combined_series_sum : ℚ :=
  let geo_sum := geometric_series_sum 5 (-1/2)
  let arith_sum := arithmetic_series_sum 3 2 5
  geo_sum + arith_sum

theorem combined_sum_is_115_over_3 : combined_series_sum = 115 / 3 := 
  sorry

end combined_sum_is_115_over_3_l180_180040


namespace compare_trig_values_l180_180270

noncomputable def a : ℝ := Real.sin (2 * Real.pi / 7)
noncomputable def b : ℝ := Real.tan (5 * Real.pi / 7)
noncomputable def c : ℝ := Real.cos (5 * Real.pi / 7)

theorem compare_trig_values :
  (0 < 2 * Real.pi / 7 ∧ 2 * Real.pi / 7 < Real.pi / 2) →
  (Real.pi / 2 < 5 * Real.pi / 7 ∧ 5 * Real.pi / 7 < 3 * Real.pi / 4) →
  b < c ∧ c < a :=
by
  intro h1 h2
  sorry

end compare_trig_values_l180_180270


namespace incident_reflected_eqs_l180_180361

theorem incident_reflected_eqs {x y : ℝ} :
  (∃ (A B : ℝ × ℝ), A = (2, 3) ∧ B = (1, 1) ∧ 
   (∀ (P : ℝ × ℝ), (P = A ∨ P = B → (P.1 + P.2 + 1 = 0) → false)) ∧
   (∃ (line_inc line_ref : ℝ × ℝ × ℝ),
     line_inc = (5, -4, 2) ∧
     line_ref = (4, -5, 1))) :=
sorry

end incident_reflected_eqs_l180_180361


namespace total_vehicles_is_120_l180_180530

def num_trucks : ℕ := 20
def num_tanks : ℕ := 5 * num_trucks
def total_vehicles : ℕ := num_tanks + num_trucks

theorem total_vehicles_is_120 : total_vehicles = 120 :=
by
  sorry

end total_vehicles_is_120_l180_180530


namespace solution_set_of_inequality_l180_180501

theorem solution_set_of_inequality : {x : ℝ | -3 < x ∧ x < 1} = {x : ℝ | x^2 + 2 * x < 3} :=
sorry

end solution_set_of_inequality_l180_180501


namespace original_number_of_laborers_l180_180673

theorem original_number_of_laborers 
(L : ℕ) (h1 : L * 15 = (L - 5) * 20) : L = 15 :=
sorry

end original_number_of_laborers_l180_180673


namespace snow_volume_l180_180078

-- Define the dimensions of the sidewalk and the snow depth
def length : ℝ := 20
def width : ℝ := 2
def depth : ℝ := 0.5

-- Define the volume calculation
def volume (l w d : ℝ) : ℝ := l * w * d

-- The theorem to prove
theorem snow_volume : volume length width depth = 20 := 
by
  sorry

end snow_volume_l180_180078


namespace largest_divisor_consecutive_odd_l180_180410

theorem largest_divisor_consecutive_odd (m n : ℤ) (h : ∃ k : ℤ, m = 2 * k + 1 ∧ n = 2 * k - 1) :
  ∃ d : ℤ, d = 8 ∧ ∀ m n : ℤ, (∃ k : ℤ, m = 2 * k + 1 ∧ n = 2 * k - 1) → d ∣ (m^2 - n^2) :=
by
  sorry

end largest_divisor_consecutive_odd_l180_180410


namespace cost_of_song_book_l180_180926

def cost_of_trumpet : ℝ := 145.16
def total_amount_spent : ℝ := 151.00

theorem cost_of_song_book : (total_amount_spent - cost_of_trumpet) = 5.84 := by
  sorry

end cost_of_song_book_l180_180926


namespace garden_snake_is_10_inches_l180_180047

-- Define the conditions from the problem statement
def garden_snake_length (garden_snake boa_constrictor : ℝ) : Prop :=
  boa_constrictor = 7 * garden_snake

def boa_constrictor_length (boa_constrictor : ℝ) : Prop :=
  boa_constrictor = 70

-- Prove the length of the garden snake
theorem garden_snake_is_10_inches : ∃ (garden_snake : ℝ), garden_snake_length garden_snake 70 ∧ garden_snake = 10 :=
by {
  sorry
}

end garden_snake_is_10_inches_l180_180047


namespace find_xyz_l180_180628

theorem find_xyz (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x + 1/y = 5) (h2 : y + 1/z = 2) (h3 : z + 1/x = 3) :
  x * y * z = 10 + 3 * Real.sqrt 11 := by
  sorry

end find_xyz_l180_180628


namespace casey_saves_by_paying_monthly_l180_180511

theorem casey_saves_by_paying_monthly :
  let weekly_rate := 280
  let monthly_rate := 1000
  let weeks_in_a_month := 4
  let number_of_months := 3
  let total_weeks := number_of_months * weeks_in_a_month
  let total_cost_weekly := total_weeks * weekly_rate
  let total_cost_monthly := number_of_months * monthly_rate
  let savings := total_cost_weekly - total_cost_monthly
  savings = 360 :=
by
  sorry

end casey_saves_by_paying_monthly_l180_180511


namespace fraction_transformation_l180_180763

theorem fraction_transformation (a b : ℝ) (h : a ≠ b) : 
  (-a) / (a - b) = a / (b - a) :=
sorry

end fraction_transformation_l180_180763


namespace find_largest_even_integer_l180_180527

-- Define the sum of the first 30 positive even integers
def sum_first_30_even : ℕ := 2 * (30 * 31 / 2)

-- Assume five consecutive even integers and their sum
def consecutive_even_sum (m : ℕ) : ℕ := (m - 8) + (m - 6) + (m - 4) + (m - 2) + m

-- Statement of the theorem to be proven
theorem find_largest_even_integer : ∃ (m : ℕ), consecutive_even_sum m = sum_first_30_even ∧ m = 190 :=
by
  sorry

end find_largest_even_integer_l180_180527


namespace length_of_AB_l180_180226

-- Definitions of the given entities
def is_on_parabola (A : ℝ × ℝ) : Prop := A.2^2 = 4 * A.1
def focus : ℝ × ℝ := (1, 0)
def line_through_focus (l : ℝ × ℝ → Prop) : Prop := l focus

-- The theorem we need to prove
theorem length_of_AB (A B : ℝ × ℝ) (l : ℝ × ℝ → Prop)
  (h1 : is_on_parabola A)
  (h2 : is_on_parabola B)
  (h3 : line_through_focus l)
  (h4 : l A)
  (h5 : l B)
  (h6 : A.1 + B.1 = 10 / 3) :
  dist A B = 16 / 3 :=
sorry

end length_of_AB_l180_180226


namespace total_dots_not_visible_eq_54_l180_180689

theorem total_dots_not_visible_eq_54 :
  let die_sum := 21
  let num_dice := 4
  let total_sum := num_dice * die_sum
  let visible_sum := 1 + 2 + 3 + 4 + 4 + 5 + 5 + 6
  total_sum - visible_sum = 54 :=
by
  let die_sum := 21
  let num_dice := 4
  let total_sum := num_dice * die_sum
  let visible_sum := 1 + 2 + 3 + 4 + 4 + 5 + 5 + 6
  show total_sum - visible_sum = 54
  sorry

end total_dots_not_visible_eq_54_l180_180689


namespace maximum_value_of_f_l180_180317

theorem maximum_value_of_f (x : ℝ) (h : x^4 + 36 ≤ 13 * x^2) : 
  ∃ (m : ℝ), m = 18 ∧ ∀ (x : ℝ), (x^4 + 36 ≤ 13 * x^2) → (x^3 - 3 * x ≤ m) :=
sorry

end maximum_value_of_f_l180_180317


namespace find_x_l180_180676

theorem find_x (x : ℝ) 
  (a : ℝ × ℝ := (2*x - 1, x + 3)) 
  (b : ℝ × ℝ := (x, 2*x + 1))
  (c : ℝ × ℝ := (1, 2))
  (h : (a.1 - b.1) * c.1 + (a.2 - b.2) * c.2 = 0) :
  x = 3 :=
  sorry

end find_x_l180_180676


namespace inspectors_in_group_B_l180_180306

theorem inspectors_in_group_B
  (a b : ℕ)  -- a: number of original finished products, b: daily production
  (A_inspectors := 8)  -- Number of inspectors in group A
  (total_days := 5) -- Group B inspects in 5 days
  (inspects_same_speed : (2 * a + 2 * 2 * b) * total_days/A_inspectors = (2 * a + 2 * 5 * b) * (total_days/3))
  : ∃ (B_inspectors : ℕ), B_inspectors = 12 := 
by
  sorry

end inspectors_in_group_B_l180_180306


namespace percent_students_own_cats_l180_180281

theorem percent_students_own_cats 
  (total_students : ℕ) (cat_owners : ℕ) (h1 : total_students = 300) (h2 : cat_owners = 45) :
  (cat_owners : ℚ) / total_students * 100 = 15 := 
by
  sorry

end percent_students_own_cats_l180_180281


namespace heloise_total_pets_l180_180952

-- Define initial data
def ratio_dogs_to_cats := (10, 17)
def dogs_given_away := 10
def dogs_remaining := 60

-- Definition of initial number of dogs based on conditions
def initial_dogs := dogs_remaining + dogs_given_away

-- Definition based on ratio of dogs to cats
def dogs_per_set := ratio_dogs_to_cats.1
def cats_per_set := ratio_dogs_to_cats.2

-- Compute the number of sets of dogs
def sets_of_dogs := initial_dogs / dogs_per_set

-- Compute the number of cats
def initial_cats := sets_of_dogs * cats_per_set

-- Definition of the total number of pets
def total_pets := dogs_remaining + initial_cats

-- Lean statement for the proof
theorem heloise_total_pets :
  initial_dogs = 70 ∧
  sets_of_dogs = 7 ∧
  initial_cats = 119 ∧
  total_pets = 179 :=
by
  -- The statements to be proved are listed as conjunctions (∧)
  sorry

end heloise_total_pets_l180_180952


namespace Andy_and_Carlos_tie_for_first_l180_180151

def AndyLawnArea (A : ℕ) := 3 * A
def CarlosLawnArea (A : ℕ) := A / 4
def BethMowingRate := 90
def CarlosMowingRate := BethMowingRate / 3
def AndyMowingRate := BethMowingRate * 4

theorem Andy_and_Carlos_tie_for_first (A : ℕ) (hA_nonzero : 0 < A) :
  (AndyLawnArea A / AndyMowingRate) = (CarlosLawnArea A / CarlosMowingRate) ∧
  (AndyLawnArea A / AndyMowingRate) < (A / BethMowingRate) :=
by
  unfold AndyLawnArea CarlosLawnArea BethMowingRate CarlosMowingRate AndyMowingRate
  sorry

end Andy_and_Carlos_tie_for_first_l180_180151


namespace circle_tangent_to_x_axis_at_origin_l180_180102

theorem circle_tangent_to_x_axis_at_origin
  (D E F : ℝ)
  (h : ∀ (x y : ℝ), x^2 + y^2 + D*x + E*y + F = 0 → 
      (x = 0 → y = 0)) :
  D = 0 ∧ E ≠ 0 ∧ F = 0 :=
sorry

end circle_tangent_to_x_axis_at_origin_l180_180102


namespace profit_margin_in_terms_of_retail_price_l180_180412

theorem profit_margin_in_terms_of_retail_price
  (k c P_R : ℝ) (h1 : ∀ C, P = k * C) (h2 : ∀ C, P_R = c * (P + C)) :
  P = (k / (c * (k + 1))) * P_R :=
by sorry

end profit_margin_in_terms_of_retail_price_l180_180412


namespace eval_f_at_5_l180_180489

def f (x : ℝ) : ℝ := 2 * x^7 - 9 * x^6 + 5 * x^5 - 49 * x^4 - 5 * x^3 + 2 * x^2 + x + 1

theorem eval_f_at_5 : f 5 = 56 := 
 by 
   sorry

end eval_f_at_5_l180_180489


namespace cost_of_article_l180_180169

-- Definitions for conditions
def gain_340 (C G : ℝ) : Prop := 340 = C + G
def gain_360 (C G : ℝ) : Prop := 360 = C + G + 0.05 * C

-- Theorem to be proven
theorem cost_of_article (C G : ℝ) (h1 : gain_340 C G) (h2 : gain_360 C G) : C = 400 :=
by sorry

end cost_of_article_l180_180169


namespace find_line_m_l180_180889

noncomputable def reflect_point_across_line 
  (P : ℝ × ℝ) (a b c : ℝ) : ℝ × ℝ :=
  let line_vector := (a, b)
  let scaling_factor := -2 * ((a * P.1 + b * P.2 + c) / (a^2 + b^2))
  ((P.1 + scaling_factor * a), (P.2 + scaling_factor * b))

theorem find_line_m (P P'' : ℝ × ℝ) (a b : ℝ) (c : ℝ := 0)
  (h₁ : P = (2, -3))
  (h₂ : a * 1 + b * 4 = 0)
  (h₃ : P'' = (1, 4))
  (h₄ : reflect_point_across_line (reflect_point_across_line P a b c) a b c = P'') :
  4 * P''.1 - P''.2 = 0 :=
by
  sorry

end find_line_m_l180_180889


namespace find_f1_plus_g1_l180_180630

variable (f g : ℝ → ℝ)

-- Conditions
def even_function (h : ℝ → ℝ) := ∀ x : ℝ, h x = h (-x)
def odd_function (h : ℝ → ℝ) := ∀ x : ℝ, h x = -h (-x)
def function_relation := ∀ x : ℝ, f x - g x = x^3 + x^2 + 1

-- Mathematically equivalent proof problem
theorem find_f1_plus_g1
  (hf_even : even_function f)
  (hg_odd : odd_function g)
  (h_relation : function_relation f g) :
  f 1 + g 1 = 1 := by
  sorry

end find_f1_plus_g1_l180_180630


namespace largest_inscribed_square_size_l180_180535

noncomputable def side_length_of_largest_inscribed_square : ℝ :=
  6 - 2 * Real.sqrt 3

theorem largest_inscribed_square_size (side_length_of_square : ℝ)
  (equi_triangles_shared_side : ℝ)
  (vertexA_of_square : ℝ)
  (vertexB_of_square : ℝ)
  (vertexC_of_square : ℝ)
  (vertexD_of_square : ℝ)
  (vertexF_of_triangles : ℝ)
  (vertexG_of_triangles : ℝ) :
  side_length_of_square = 12 →
  equi_triangles_shared_side = vertexB_of_square - vertexA_of_square →
  vertexF_of_triangles = vertexD_of_square - vertexC_of_square →
  vertexG_of_triangles = vertexB_of_square - vertexA_of_square →
  side_length_of_largest_inscribed_square = 6 - 2 * Real.sqrt 3 :=
sorry

end largest_inscribed_square_size_l180_180535


namespace find_f_13_l180_180586

def periodic (f : ℝ → ℝ) (p : ℝ) := ∀ x, f (x + p) = f x

theorem find_f_13 (f : ℝ → ℝ) 
  (h_period : periodic f 1.5) 
  (h_val : f 1 = 20) 
  : f 13 = 20 :=
by
  sorry

end find_f_13_l180_180586


namespace find_smaller_number_l180_180649

theorem find_smaller_number (x y : ℕ) (h1 : x = 2 * y - 3) (h2 : x + y = 51) : y = 18 :=
sorry

end find_smaller_number_l180_180649


namespace rhombus_area_l180_180016

theorem rhombus_area (d1 d2 : ℝ) (h_d1 : d1 = 25) (h_d2 : d2 = 50) :
  (d1 * d2) / 2 = 625 := 
by
  sorry

end rhombus_area_l180_180016


namespace stratified_sampling_female_students_l180_180144

-- Definitions from conditions
def male_students : ℕ := 800
def female_students : ℕ := 600
def drawn_male_students : ℕ := 40
def total_students : ℕ := 1400

-- Proof statement
theorem stratified_sampling_female_students : 
  (female_students * drawn_male_students) / male_students = 30 :=
by
  -- substitute and simplify
  sorry

end stratified_sampling_female_students_l180_180144


namespace log_function_domain_l180_180035

theorem log_function_domain :
  { x : ℝ | x^2 - 2 * x - 3 > 0 } = { x | x > 3 } ∪ { x | x < -1 } :=
by {
  sorry
}

end log_function_domain_l180_180035


namespace possible_m_value_l180_180432

variable (a b m t : ℝ)
variable (h_a : a ≠ 0)
variable (h1 : ∃ t, ∀ x, ax^2 - bx ≥ -1 ↔ (x ≤ t - 1 ∨ x ≥ -3 - t))
variable (h2 : a * m^2 - b * m = 2)

theorem possible_m_value : m = 1 :=
sorry

end possible_m_value_l180_180432


namespace correct_equation_l180_180237

theorem correct_equation (a b : ℝ) : 
  (a + b)^2 = a^2 + 2 * a * b + b^2 := by
  sorry

end correct_equation_l180_180237


namespace refills_count_l180_180879

variable (spent : ℕ) (cost : ℕ)

theorem refills_count (h1 : spent = 40) (h2 : cost = 10) : spent / cost = 4 := 
by
  sorry

end refills_count_l180_180879


namespace solution_in_quadrant_I_l180_180351

theorem solution_in_quadrant_I (k : ℝ) :
  ∃ x y : ℝ, (2 * x - y = 5 ∧ k * x + 2 * y = 4 ∧ x > 0 ∧ y > 0) ↔ -4 < k ∧ k < 8 / 5 :=
by
  sorry

end solution_in_quadrant_I_l180_180351


namespace calculate_expression_l180_180479

-- Definitions based on the conditions
def opposite (a b : ℤ) : Prop := a + b = 0
def reciprocal (c d : ℝ) : Prop := c * d = 1
def negative_abs_two (m : ℝ) : Prop := m = -2

-- The main statement to be proved
theorem calculate_expression (a b : ℤ) (c d m : ℝ) 
  (h1 : opposite a b) 
  (h2 : reciprocal c d) 
  (h3 : negative_abs_two m) : 
  m + c * d + a + b + (c * d) ^ 2010 = 0 := 
by
  sorry

end calculate_expression_l180_180479


namespace remainder_of_8_pow_6_plus_1_mod_7_l180_180705

theorem remainder_of_8_pow_6_plus_1_mod_7 :
  (8^6 + 1) % 7 = 2 := by
  sorry

end remainder_of_8_pow_6_plus_1_mod_7_l180_180705


namespace problem1_problem2_l180_180719

theorem problem1 : ((- (5 : ℚ) / 6) + 2 / 3) / (- (7 / 12)) * (7 / 2) = 1 := 
sorry

theorem problem2 : ((1 - 1 / 6) * (-3) - (- (11 / 6)) / (- (22 / 3))) = - (11 / 4) := 
sorry

end problem1_problem2_l180_180719


namespace coffee_consumption_l180_180850

variables (h w g : ℝ)

theorem coffee_consumption (k : ℝ) 
  (H1 : ∀ h w g, h * g = k * w)
  (H2 : h = 8 ∧ g = 4.5 ∧ w = 2)
  (H3 : h = 4 ∧ w = 3) : g = 13.5 :=
by {
  sorry
}

end coffee_consumption_l180_180850


namespace minimum_rectangle_length_l180_180175

theorem minimum_rectangle_length (a x y : ℝ) (h : x * y = a^2) : x ≥ a ∨ y ≥ a :=
sorry

end minimum_rectangle_length_l180_180175


namespace Ayla_call_duration_l180_180587

theorem Ayla_call_duration
  (charge_per_minute : ℝ)
  (monthly_bill : ℝ)
  (customers_per_week : ℕ)
  (weeks_in_month : ℕ)
  (calls_duration : ℝ)
  (h_charge : charge_per_minute = 0.05)
  (h_bill : monthly_bill = 600)
  (h_customers : customers_per_week = 50)
  (h_weeks_in_month : weeks_in_month = 4)
  (h_calls_duration : calls_duration = (monthly_bill / charge_per_minute) / (customers_per_week * weeks_in_month)) :
  calls_duration = 60 :=
by 
  sorry

end Ayla_call_duration_l180_180587


namespace solution_set_f_gt_5_range_m_f_ge_abs_2m1_l180_180686

noncomputable def f (x : ℝ) : ℝ := abs (2 * x - 1) + abs (x + 3)

theorem solution_set_f_gt_5 :
  {x : ℝ | f x > 5} = {x : ℝ | x < -1} ∪ {x : ℝ | x > 1} :=
by sorry

theorem range_m_f_ge_abs_2m1 :
  (∀ x : ℝ, f x ≥ abs (2 * m + 1)) ↔ -9/4 ≤ m ∧ m ≤ 5/4 :=
by sorry

end solution_set_f_gt_5_range_m_f_ge_abs_2m1_l180_180686


namespace parabola_constant_l180_180773

theorem parabola_constant (b c : ℝ)
  (h₁ : -20 = 2 * (-2)^2 + b * (-2) + c)
  (h₂ : 24 = 2 * 2^2 + b * 2 + c) : 
  c = -6 := 
by 
  sorry

end parabola_constant_l180_180773


namespace negate_proposition_l180_180154

def p (x : ℝ) : Prop := x^2 + x - 6 > 0
def q (x : ℝ) : Prop := x > 2 ∨ x < -3

def neg_p (x : ℝ) : Prop := x^2 + x - 6 ≤ 0
def neg_q (x : ℝ) : Prop := -3 ≤ x ∧ x ≤ 2

theorem negate_proposition (x : ℝ) :
  (¬ (p x → q x)) ↔ (neg_p x → neg_q x) :=
by unfold p q neg_p neg_q; apply sorry

end negate_proposition_l180_180154


namespace bus_speeds_l180_180885

theorem bus_speeds (d t : ℝ) (s₁ s₂ : ℝ)
  (h₀ : d = 48)
  (h₁ : t = 1 / 6) -- 10 minutes in hours
  (h₂ : s₂ = s₁ - 4)
  (h₃ : d / s₂ - d / s₁ = t) :
  s₁ = 36 ∧ s₂ = 32 := 
sorry

end bus_speeds_l180_180885


namespace total_feed_per_week_l180_180178

-- Define the conditions
def daily_feed_per_pig : ℕ := 10
def number_of_pigs : ℕ := 2
def days_per_week : ℕ := 7

-- Theorem statement
theorem total_feed_per_week : daily_feed_per_pig * number_of_pigs * days_per_week = 140 := 
  sorry

end total_feed_per_week_l180_180178


namespace total_students_correct_l180_180843

def students_in_general_hall : ℕ := 30
def students_in_biology_hall : ℕ := 2 * students_in_general_hall
def combined_students_general_biology : ℕ := students_in_general_hall + students_in_biology_hall
def students_in_math_hall : ℕ := (3 * combined_students_general_biology) / 5
def total_students_in_all_halls : ℕ := students_in_general_hall + students_in_biology_hall + students_in_math_hall

theorem total_students_correct : total_students_in_all_halls = 144 := by
  -- Proof omitted, it should be
  sorry

end total_students_correct_l180_180843


namespace scientific_notation_example_l180_180946

theorem scientific_notation_example :
  110000 = 1.1 * 10^5 :=
by {
  sorry
}

end scientific_notation_example_l180_180946


namespace evaluate_expression_l180_180290

theorem evaluate_expression : 
  (3^1002 + 7^1003)^2 - (3^1002 - 7^1003)^2 = 1372 * 10^1003 := 
by sorry

end evaluate_expression_l180_180290


namespace factor_poly_l180_180055

theorem factor_poly (x : ℝ) : (75 * x^3 - 300 * x^7) = 75 * x^3 * (1 - 4 * x^4) :=
by sorry

end factor_poly_l180_180055


namespace sum_is_220_l180_180654

def second_number := 60
def first_number := 2 * second_number
def third_number := first_number / 3
def sum_of_numbers := first_number + second_number + third_number

theorem sum_is_220 : sum_of_numbers = 220 :=
by
  sorry

end sum_is_220_l180_180654


namespace permutation_average_sum_l180_180275

theorem permutation_average_sum :
  let p := 286
  let q := 11
  p + q = 297 :=
by
  sorry

end permutation_average_sum_l180_180275


namespace grassy_area_percentage_l180_180002

noncomputable def percentage_grassy_area (park_area path1_area path2_area intersection_area : ℝ) : ℝ :=
  let covered_by_paths := path1_area + path2_area - intersection_area
  let grassy_area := park_area - covered_by_paths
  (grassy_area / park_area) * 100

theorem grassy_area_percentage (park_area : ℝ) (path1_area : ℝ) (path2_area : ℝ) (intersection_area : ℝ) 
  (h1 : park_area = 4000) (h2 : path1_area = 400) (h3 : path2_area = 250) (h4 : intersection_area = 25) : 
  percentage_grassy_area park_area path1_area path2_area intersection_area = 84.375 :=
by
  rw [percentage_grassy_area, h1, h2, h3, h4]
  simp
  sorry

end grassy_area_percentage_l180_180002


namespace compare_fx_l180_180307

noncomputable def f (x : ℝ) (b : ℝ) (c : ℝ) : ℝ :=
  x^2 - b * x + c

theorem compare_fx (b c : ℝ) (x : ℝ) (h1 : ∀ x : ℝ, f (1 - x) b c = f (1 + x) b c) (h2 : f 0 b c = 3) :
  f (2^x) b c ≤ f (3^x) b c :=
by
  sorry

end compare_fx_l180_180307


namespace alex_seashells_l180_180202

theorem alex_seashells (mimi_seashells kyle_seashells leigh_seashells alex_seashells : ℕ) 
    (h1 : mimi_seashells = 2 * 12) 
    (h2 : kyle_seashells = 2 * mimi_seashells) 
    (h3 : leigh_seashells = kyle_seashells / 3) 
    (h4 : alex_seashells = 3 * leigh_seashells) : 
  alex_seashells = 48 := by
  sorry

end alex_seashells_l180_180202


namespace vertex_of_parabola_l180_180564

theorem vertex_of_parabola (a b c : ℝ) :
  (∀ x y : ℝ, (x = -2 ∧ y = 5) ∨ (x = 4 ∧ y = 5) ∨ (x = 2 ∧ y = 2) →
    y = a * x^2 + b * x + c) →
  (∃ x_vertex : ℝ, x_vertex = 1) :=
by
  sorry

end vertex_of_parabola_l180_180564


namespace value_of_m_l180_180444

theorem value_of_m (m : ℝ) :
  (∀ A B : ℝ × ℝ, A = (m + 1, -2) → B = (3, m - 1) → (A.snd = B.snd) → m = -1) :=
by
  intros A B hA hB h_parallel
  -- Apply the given conditions and assumptions to prove the value of m.
  sorry

end value_of_m_l180_180444


namespace intersection_of_M_and_N_l180_180231

def M : Set ℝ := {x | -3 < x ∧ x < 1}
def N : Set ℝ := {-3, -2, -1, 0, 1}

theorem intersection_of_M_and_N : M ∩ N = {-2, -1, 0} := sorry

end intersection_of_M_and_N_l180_180231


namespace calculate_p_op_l180_180988

def op (x y : ℝ) := x * y^2 - x

theorem calculate_p_op (p : ℝ) : op p (op p p) = p^7 - 2*p^5 + p^3 - p :=
by
  sorry

end calculate_p_op_l180_180988


namespace dice_sum_impossible_l180_180821

theorem dice_sum_impossible (a b c d : ℕ) (h1 : a * b * c * d = 216)
  (ha : 1 ≤ a ∧ a ≤ 6) (hb : 1 ≤ b ∧ b ≤ 6) 
  (hc : 1 ≤ c ∧ c ≤ 6) (hd : 1 ≤ d ∧ d ≤ 6) : 
  a + b + c + d ≠ 18 :=
sorry

end dice_sum_impossible_l180_180821


namespace hyperbola_focal_distance_distance_focus_to_asymptote_l180_180650

theorem hyperbola_focal_distance :
  let a := 1
  let b := Real.sqrt 3
  let c := Real.sqrt (a ^ 2 + b ^ 2)
  (2 * c = 4) :=
by sorry

theorem distance_focus_to_asymptote :
  let a := 1
  let b := Real.sqrt 3
  let c := Real.sqrt (a ^ 2 + b ^ 2)
  let focus := (c, 0)
  let A := -Real.sqrt 3
  let B := 1
  let C := 0
  let distance := (|A * focus.fst + B * focus.snd + C|) / Real.sqrt (A ^ 2 + B ^ 2)
  (distance = Real.sqrt 3) :=
by sorry

end hyperbola_focal_distance_distance_focus_to_asymptote_l180_180650


namespace probability_of_same_length_segments_l180_180797

-- Definitions directly from the conditions
def total_elements : ℕ := 6 + 9

-- Probability calculations
def probability_same_length : ℚ :=
  ((6 / total_elements) * (5 / (total_elements - 1))) + 
  ((9 / total_elements) * (8 / (total_elements - 1)))

-- Proof statement
theorem probability_of_same_length_segments : probability_same_length = 17 / 35 := by
  unfold probability_same_length
  sorry

end probability_of_same_length_segments_l180_180797


namespace possible_degrees_of_remainder_l180_180450

theorem possible_degrees_of_remainder (p : Polynomial ℝ) (h : p = 3 * X^3 - 5 * X^2 + 2 * X - 8) :
  ∃ d : Finset ℕ, d = {0, 1, 2} :=
by
  sorry

end possible_degrees_of_remainder_l180_180450


namespace average_apples_per_guest_l180_180474

theorem average_apples_per_guest
  (servings_per_pie : ℕ)
  (pies : ℕ)
  (apples_per_serving : ℚ)
  (total_guests : ℕ)
  (red_delicious_proportion : ℚ)
  (granny_smith_proportion : ℚ)
  (total_servings := pies * servings_per_pie)
  (total_apples := total_servings * apples_per_serving)
  (total_red_delicious := (red_delicious_proportion / (red_delicious_proportion + granny_smith_proportion)) * total_apples)
  (total_granny_smith := (granny_smith_proportion / (red_delicious_proportion + granny_smith_proportion)) * total_apples)
  (average_apples_per_guest := total_apples / total_guests) :
  servings_per_pie = 8 →
  pies = 3 →
  apples_per_serving = 1.5 →
  total_guests = 12 →
  red_delicious_proportion = 2 →
  granny_smith_proportion = 1 →
  average_apples_per_guest = 3 :=
by
  intros;
  sorry

end average_apples_per_guest_l180_180474


namespace solve_equation_l180_180393

theorem solve_equation (x : ℝ) : 
  3 * x * (x - 1) = 2 * x - 2 ↔ x = 1 ∨ x = 2 / 3 :=
by
  sorry

end solve_equation_l180_180393


namespace product_of_possible_values_l180_180227

theorem product_of_possible_values (N : ℤ) (M L : ℤ) 
(h1 : M = L + N)
(h2 : M - 3 = L + N - 3)
(h3 : L + 5 = L + 5)
(h4 : |(L + N - 3) - (L + 5)| = 4) :
N = 12 ∨ N = 4 → (12 * 4 = 48) :=
by sorry

end product_of_possible_values_l180_180227


namespace at_least_three_double_marked_l180_180476

noncomputable def grid := Matrix (Fin 10) (Fin 20) ℕ -- 10x20 matrix with natural numbers

def is_red_marked (g : grid) (i : Fin 10) (j : Fin 20) : Prop :=
  ∃ (k₁ k₂ : Fin 20), k₁ ≠ k₂ ∧ (g i k₁) ≤ g i j ∧ (g i k₂) ≤ g i j ∧ ∀ (k : Fin 20), (k ≠ k₁ ∧ k ≠ k₂) → g i k ≤ g i j

def is_blue_marked (g : grid) (i : Fin 10) (j : Fin 20) : Prop :=
  ∃ (k₁ k₂ : Fin 10), k₁ ≠ k₂ ∧ (g k₁ j) ≤ g i j ∧ (g k₂ j) ≤ g i j ∧ ∀ (k : Fin 10), (k ≠ k₁ ∧ k ≠ k₂) → g k j ≤ g i j

def is_double_marked (g : grid) (i : Fin 10) (j : Fin 20) : Prop :=
  is_red_marked g i j ∧ is_blue_marked g i j

theorem at_least_three_double_marked (g : grid) :
  (∃ (i₁ i₂ i₃ : Fin 10) (j₁ j₂ j₃ : Fin 20), i₁ ≠ i₂ ∧ i₂ ≠ i₃ ∧ i₃ ≠ i₁ ∧ 
    j₁ ≠ j₂ ∧ j₂ ≠ j₃ ∧ j₃ ≠ j₁ ∧ is_double_marked g i₁ j₁ ∧ is_double_marked g i₂ j₂ ∧ is_double_marked g i₃ j₃) :=
sorry

end at_least_three_double_marked_l180_180476


namespace cannot_determine_if_counterfeit_coin_is_lighter_or_heavier_l180_180726

/-- 
Vasiliy has 2019 coins, one of which is counterfeit (differing in weight). 
Using balance scales without weights and immediately paying out identified genuine coins, 
it is impossible to determine whether the counterfeit coin is lighter or heavier.
-/
theorem cannot_determine_if_counterfeit_coin_is_lighter_or_heavier 
  (num_coins : ℕ)
  (num_counterfeit : ℕ)
  (balance_scale : Bool → Bool → Bool)
  (immediate_payment : Bool → Bool) :
  num_coins = 2019 →
  num_counterfeit = 1 →
  (∀ coins_w1 coins_w2, balance_scale coins_w1 coins_w2 = (coins_w1 = coins_w2)) →
  (∀ coin_p coin_q, (immediate_payment coin_p = true) → ¬ coin_p = coin_q) →
  ¬ ∃ (is_lighter_or_heavier : Bool), true :=
by
  intro h1 h2 h3 h4
  sorry

end cannot_determine_if_counterfeit_coin_is_lighter_or_heavier_l180_180726


namespace grunters_win_4_out_of_6_l180_180376

/-- The Grunters have a probability of winning any given game as 60% --/
def p : ℚ := 3 / 5

/-- The Grunters have a probability of losing any given game as 40% --/
def q : ℚ := 1 - p

/-- The binomial coefficient for choosing exactly 4 wins out of 6 games --/
def binomial_6_4 : ℚ := Nat.choose 6 4

/-- The probability that the Grunters win exactly 4 out of the 6 games --/
def prob_4_wins : ℚ := binomial_6_4 * (p ^ 4) * (q ^ 2)

/--
The probability that the Grunters win exactly 4 out of the 6 games is
exactly $\frac{4860}{15625}$.
--/
theorem grunters_win_4_out_of_6 : prob_4_wins = 4860 / 15625 := by
  sorry

end grunters_win_4_out_of_6_l180_180376


namespace geometric_sequence_term_6_l180_180478

-- Define the geometric sequence conditions
def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
∀ n, a (n + 1) = a n * r

variables 
  (a : ℕ → ℝ) -- the geometric sequence
  (r : ℝ) -- common ratio, which is 2
  (h_r : r = 2)
  (h_pos : ∀ n, 0 < a n)
  (h_condition : a 4 * a 10 = 16)

-- The proof statement
theorem geometric_sequence_term_6 :
  a 6 = 2 :=
sorry

end geometric_sequence_term_6_l180_180478


namespace no_real_roots_quadratic_eq_l180_180259

theorem no_real_roots_quadratic_eq :
  ¬ ∃ x : ℝ, 7 * x^2 - 4 * x + 6 = 0 :=
by sorry

end no_real_roots_quadratic_eq_l180_180259


namespace combination_identity_l180_180718

-- Lean statement defining the proof problem
theorem combination_identity : Nat.choose 12 5 + Nat.choose 12 6 = Nat.choose 13 6 :=
  sorry

end combination_identity_l180_180718


namespace find_certain_number_l180_180503

theorem find_certain_number (n : ℕ)
  (h1 : 3153 + 3 = 3156)
  (h2 : 3156 % 9 = 0)
  (h3 : 3156 % 70 = 0)
  (h4 : 3156 % 25 = 0) :
  3156 % 37 = 0 :=
by
  sorry

end find_certain_number_l180_180503


namespace polynomial_mod_p_zero_l180_180121

def is_zero_mod_p (p : ℕ) [Fact (Nat.Prime p)] (f : (List ℕ → ℤ)) : Prop :=
  ∀ (x : List ℕ), f x % p = 0

theorem polynomial_mod_p_zero
  (p : ℕ) [Fact (Nat.Prime p)]
  (n : ℕ) 
  (f : (List ℕ → ℤ)) 
  (h : ∀ (x : List ℕ), f x % p = 0) 
  (g : (List ℕ → ℤ)) :
  (∀ (x : List ℕ), g x % p = 0) := sorry

end polynomial_mod_p_zero_l180_180121


namespace intersection_with_unit_circle_l180_180389

theorem intersection_with_unit_circle (α : ℝ) : 
    let x := Real.cos (α - Real.pi / 2)
    let y := Real.sin (α - Real.pi / 2)
    (x, y) = (Real.sin α, -Real.cos α) :=
by
  sorry

end intersection_with_unit_circle_l180_180389


namespace statement_II_and_IV_true_l180_180031

-- Definitions based on the problem's conditions
def AllNewEditions (P : Type) (books : P → Prop) := ∀ x, books x

-- Condition that the statement "All books in the library are new editions." is false
def NotAllNewEditions (P : Type) (books : P → Prop) := ¬ (AllNewEditions P books)

-- Statements to analyze
def SomeBookNotNewEdition (P : Type) (books : P → Prop) := ∃ x, ¬ books x
def NotAllBooksNewEditions (P : Type) (books : P → Prop) := ∃ x, ¬ books x

-- The theorem to prove
theorem statement_II_and_IV_true 
  (P : Type) 
  (books : P → Prop) 
  (h : NotAllNewEditions P books) : 
  SomeBookNotNewEdition P books ∧ NotAllBooksNewEditions P books :=
  by
    sorry

end statement_II_and_IV_true_l180_180031


namespace two_p_plus_q_l180_180222

variable {p q : ℚ}  -- Variables are rationals

theorem two_p_plus_q (h : p / q = 5 / 4) : 2 * p + q = 7 * q / 2 :=
by sorry

end two_p_plus_q_l180_180222


namespace sum_series_eq_one_quarter_l180_180325

noncomputable def series_term (n : ℕ) : ℝ :=
  3^n / (1 + 3^n + 3^(n + 1) + 3^(2 * n + 1))

theorem sum_series_eq_one_quarter : 
  (∑' n, series_term (n + 1)) = 1 / 4 :=
by
  sorry

end sum_series_eq_one_quarter_l180_180325


namespace negation_P_l180_180067

-- Define the condition that x is a real number
variable (x : ℝ)

-- Define the proposition P
def P := ∀ (x : ℝ), x ≥ 2

-- Define the negation of P
def not_P := ∃ (x : ℝ), x < 2

-- Theorem stating the equivalence of the negation of P
theorem negation_P : ¬P ↔ not_P := by
  sorry

end negation_P_l180_180067


namespace exists_integers_a_b_l180_180873

theorem exists_integers_a_b : 
  ∃ (a b : ℤ), 2003 < a + b * (Real.sqrt 2) ∧ a + b * (Real.sqrt 2) < 2003.01 :=
by
  sorry

end exists_integers_a_b_l180_180873


namespace find_value_of_f2_plus_g3_l180_180034

def f (x : ℝ) : ℝ := 3 * x - 2
def g (x : ℝ) : ℝ := x^2 + 2

theorem find_value_of_f2_plus_g3 : f (2 + g 3) = 37 :=
by
  simp [f, g]
  norm_num
  done

end find_value_of_f2_plus_g3_l180_180034


namespace sin_double_angle_of_tangent_l180_180622

theorem sin_double_angle_of_tangent (α : ℝ) (h : Real.tan (π + α) = 2) : Real.sin (2 * α) = 4 / 5 := by
  sorry

end sin_double_angle_of_tangent_l180_180622


namespace proof_problem_l180_180091

-- Proposition B: ∃ x ∈ ℝ, x^2 - 3*x + 3 < 0
def propB : Prop := ∃ x : ℝ, x^2 - 3 * x + 3 < 0

-- Proposition D: ∀ x ∈ ℝ, x^2 - a*x + 1 = 0 has real solutions
def propD (a : ℝ) : Prop := ∀ x : ℝ, ∃ (x1 x2 : ℝ), x^2 - a * x + 1 = 0

-- Negation of Proposition B: ∀ x ∈ ℝ, x^2 - 3 * x + 3 ≥ 0
def neg_propB : Prop := ∀ x : ℝ, x^2 - 3 * x + 3 ≥ 0

-- Negation of Proposition D: ∃ a ∈ ℝ, ∃ x ∈ ℝ, ∄ (x1 x2 : ℝ), x^2 - a * x + 1 = 0
def neg_propD : Prop := ∃ a : ℝ, ∀ x : ℝ, ¬ ∃ (x1 x2 : ℝ), x^2 - a * x + 1 = 0 

-- The main theorem combining the results based on the solutions.
theorem proof_problem : neg_propB ∧ neg_propD :=
by
  sorry

end proof_problem_l180_180091


namespace sales_fifth_month_l180_180557

theorem sales_fifth_month (s1 s2 s3 s4 s6 s5 : ℝ) (target_avg total_sales : ℝ)
  (h1 : s1 = 4000)
  (h2 : s2 = 6524)
  (h3 : s3 = 5689)
  (h4 : s4 = 7230)
  (h6 : s6 = 12557)
  (h_avg : target_avg = 7000)
  (h_total_sales : total_sales = 42000) :
  s5 = 6000 :=
by
  sorry

end sales_fifth_month_l180_180557


namespace max_tulips_l180_180400

theorem max_tulips (r y n : ℕ) (hr : r = y + 1) (hc : 50 * y + 31 * r ≤ 600) :
  r + y = 2 * y + 1 → n = 2 * y + 1 → n = 15 :=
by
  -- Given the constraints, we need to identify the maximum n given the costs and relationships.
  sorry

end max_tulips_l180_180400


namespace onlyD_is_PythagoreanTriple_l180_180827

def isPythagoreanTriple (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

def validTripleA := ¬ isPythagoreanTriple 12 15 18
def validTripleB := isPythagoreanTriple 3 4 5 ∧ (¬ (3 = 3 ∧ 4 = 4 ∧ 5 = 5)) -- Since 0.3, 0.4, 0.5 not integers
def validTripleC := ¬ isPythagoreanTriple 15 25 30 -- Conversion of 1.5, 2.5, 3 to integers
def validTripleD := isPythagoreanTriple 12 16 20

theorem onlyD_is_PythagoreanTriple : validTripleA ∧ validTripleB ∧ validTripleC ∧ validTripleD :=
by {
  sorry
}

end onlyD_is_PythagoreanTriple_l180_180827
