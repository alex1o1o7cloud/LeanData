import Mathlib

namespace NUMINAMATH_GPT_edward_initial_money_l378_37882

variable (spent_books : ℕ) (spent_pens : ℕ) (money_left : ℕ)

theorem edward_initial_money (h_books : spent_books = 6) 
                             (h_pens : spent_pens = 16)
                             (h_left : money_left = 19) : 
                             spent_books + spent_pens + money_left = 41 := by
  sorry

end NUMINAMATH_GPT_edward_initial_money_l378_37882


namespace NUMINAMATH_GPT_number_of_questionnaires_from_unit_D_l378_37837

theorem number_of_questionnaires_from_unit_D 
  (a d : ℕ) 
  (total : ℕ) 
  (samples : ℕ → ℕ) 
  (h_seq : samples 0 = a ∧ samples 1 = a + d ∧ samples 2 = a + 2 * d ∧ samples 3 = a + 3 * d)
  (h_total : samples 0 + samples 1 + samples 2 + samples 3 = total)
  (h_stratified : ∀ (i : ℕ), i < 4 → samples i * 100 / total = 20 → i = 1) 
  : samples 3 = 40 := sorry

end NUMINAMATH_GPT_number_of_questionnaires_from_unit_D_l378_37837


namespace NUMINAMATH_GPT_measure_of_U_is_120_l378_37856

variable {α β γ δ ε ζ : ℝ}
variable (h1 : α = γ) (h2 : α = ζ) (h3 : β + δ = 180) (h4 : ε + ζ = 180)

noncomputable def measure_of_U : ℝ :=
  let total_sum := 720
  have sum_of_angles : α + β + γ + δ + ζ + ε = total_sum := by
    sorry
  have subs_suppl_G_R : β + δ = 180 := h3
  have subs_suppl_E_U : ε + ζ = 180 := h4
  have congruent_F_I_U : α = γ ∧ α = ζ := ⟨h1, h2⟩
  let α : ℝ := sorry
  α

theorem measure_of_U_is_120 : measure_of_U h1 h2 h3 h4 = 120 :=
  sorry

end NUMINAMATH_GPT_measure_of_U_is_120_l378_37856


namespace NUMINAMATH_GPT_ones_digit_of_largest_power_of_3_dividing_factorial_l378_37877

theorem ones_digit_of_largest_power_of_3_dividing_factorial (n : ℕ) (h : 27 = 3^3) : 
  (fun x => x % 10) (3^13) = 3 := by
  sorry

end NUMINAMATH_GPT_ones_digit_of_largest_power_of_3_dividing_factorial_l378_37877


namespace NUMINAMATH_GPT_find_g1_gneg1_l378_37876

variables {f g : ℝ → ℝ}

theorem find_g1_gneg1 (h1 : ∀ x y : ℝ, f (x - y) = f x * g y - g x * f y)
                      (h2 : f (-2) = f 1 ∧ f 1 ≠ 0) :
  g 1 + g (-1) = -1 :=
sorry

end NUMINAMATH_GPT_find_g1_gneg1_l378_37876


namespace NUMINAMATH_GPT_insufficient_pharmacies_l378_37854

/-- Define the grid size and conditions --/
def grid_size : ℕ := 9
def total_intersections : ℕ := 100
def pharmacy_walking_distance : ℕ := 3
def number_of_pharmacies : ℕ := 12

/-- Prove that 12 pharmacies are not enough for the given conditions. --/
theorem insufficient_pharmacies : ¬ (number_of_pharmacies ≥ grid_size + 3) → 
  ∃(pharmacies : ℕ), pharmacies = number_of_pharmacies ∧
  ∀(x y : ℕ), (x ≤ grid_size ∧ y ≤ grid_size) → 
  ¬ exists (p : ℕ × ℕ), p.1 ≤ x + pharmacy_walking_distance ∧
  p.2 ≤ y + pharmacy_walking_distance ∧ 
  p.1 < total_intersections ∧ 
  p.2 < total_intersections := 
by sorry

end NUMINAMATH_GPT_insufficient_pharmacies_l378_37854


namespace NUMINAMATH_GPT_part1_part2_l378_37878

def f (x a : ℝ) : ℝ := abs (x - 1) + abs (x - a)

theorem part1 (x : ℝ) (h : f x 2 ≥ 2) : x ≤ 1/2 ∨ x ≥ 2.5 := by
  sorry

theorem part2 (a : ℝ) (h_even : ∀ x : ℝ, f (-x) a = f x a) : a = -1 := by
  sorry

end NUMINAMATH_GPT_part1_part2_l378_37878


namespace NUMINAMATH_GPT_impossibility_of_transition_l378_37821

theorem impossibility_of_transition 
  {a b c : ℤ}
  (h1 : a = 2)
  (h2 : b = 2)
  (h3 : c = 2) :
  ¬(∃ x y z : ℤ, x = 19 ∧ y = 1997 ∧ z = 1999 ∧
    (∃ n : ℕ, ∀ i < n, ∃ a' b' c' : ℤ, 
      if i = 0 then a' = 2 ∧ b' = 2 ∧ c' = 2 
      else (a', b', c') = 
        if i % 3 = 0 then (b + c - 1, b, c)
        else if i % 3 = 1 then (a, a + c - 1, c)
        else (a, b, a + b - 1) 
  )) :=
sorry

end NUMINAMATH_GPT_impossibility_of_transition_l378_37821


namespace NUMINAMATH_GPT_spike_hunts_20_crickets_per_day_l378_37869

/-- Spike the bearded dragon hunts 5 crickets every morning -/
def spike_morning_crickets : ℕ := 5

/-- Spike hunts three times the morning amount in the afternoon and evening -/
def spike_afternoon_evening_multiplier : ℕ := 3

/-- Total number of crickets Spike hunts per day -/
def spike_total_crickets_per_day : ℕ := spike_morning_crickets + spike_morning_crickets * spike_afternoon_evening_multiplier

/-- Prove that the total number of crickets Spike hunts per day is 20 -/
theorem spike_hunts_20_crickets_per_day : spike_total_crickets_per_day = 20 := 
by
  sorry

end NUMINAMATH_GPT_spike_hunts_20_crickets_per_day_l378_37869


namespace NUMINAMATH_GPT_percentage_trucks_returned_l378_37826

theorem percentage_trucks_returned (total_trucks rented_trucks returned_trucks : ℕ)
  (h1 : total_trucks = 24)
  (h2 : rented_trucks = total_trucks)
  (h3 : returned_trucks ≥ 12)
  (h4 : returned_trucks ≤ total_trucks) :
  (returned_trucks / rented_trucks) * 100 = 50 :=
by sorry

end NUMINAMATH_GPT_percentage_trucks_returned_l378_37826


namespace NUMINAMATH_GPT_heavy_cream_cost_l378_37811

theorem heavy_cream_cost
  (cost_strawberries : ℕ)
  (cost_raspberries : ℕ)
  (total_cost : ℕ)
  (cost_heavy_cream : ℕ) :
  (cost_strawberries = 3 * 2) →
  (cost_raspberries = 5 * 2) →
  (total_cost = 20) →
  (cost_heavy_cream = total_cost - (cost_strawberries + cost_raspberries)) →
  cost_heavy_cream = 4 :=
by
  sorry

end NUMINAMATH_GPT_heavy_cream_cost_l378_37811


namespace NUMINAMATH_GPT_range_of_m_l378_37859

theorem range_of_m (x y m : ℝ) (h1 : 2 / x + 1 / y = 1) (h2 : x + y = 2 + 2 * m) : -4 < m ∧ m < 2 :=
sorry

end NUMINAMATH_GPT_range_of_m_l378_37859


namespace NUMINAMATH_GPT_simplify_fraction_l378_37838

theorem simplify_fraction (x y : ℕ) : (x + y)^3 / (x + y) = (x + y)^2 := by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l378_37838


namespace NUMINAMATH_GPT_total_amount_shared_l378_37884

theorem total_amount_shared (a b c d : ℝ) (h1 : a = (1/3) * (b + c + d)) 
    (h2 : b = (2/7) * (a + c + d)) (h3 : c = (4/9) * (a + b + d)) 
    (h4 : d = (5/11) * (a + b + c)) (h5 : a = b + 20) (h6 : c = d - 15) 
    (h7 : (a + b + c + d) % 10 = 0) : a + b + c + d = 1330 :=
by
  sorry

end NUMINAMATH_GPT_total_amount_shared_l378_37884


namespace NUMINAMATH_GPT_dennis_floor_l378_37806

theorem dennis_floor :
  ∃ d c b f e: ℕ, 
  (d = c + 2) ∧ 
  (c = b + 1) ∧ 
  (c = f / 4) ∧ 
  (f = 16) ∧ 
  (e = d / 2) ∧ 
  (d = 6) :=
by
  sorry

end NUMINAMATH_GPT_dennis_floor_l378_37806


namespace NUMINAMATH_GPT_ratio_of_speeds_l378_37839

variables (v_A v_B v_C : ℝ)

-- Conditions definitions
def condition1 : Prop := v_A - v_B = 5
def condition2 : Prop := v_A + v_C = 15

-- Theorem statement (the mathematically equivalent proof problem)
theorem ratio_of_speeds (h1 : condition1 v_A v_B) (h2 : condition2 v_A v_C) : (v_A / v_B) = 3 :=
sorry

end NUMINAMATH_GPT_ratio_of_speeds_l378_37839


namespace NUMINAMATH_GPT_shorter_leg_of_right_triangle_l378_37804

theorem shorter_leg_of_right_triangle {a b : ℕ} (h : a^2 + b^2 = 65^2) (ha : a ≤ b) : a = 25 :=
by sorry

end NUMINAMATH_GPT_shorter_leg_of_right_triangle_l378_37804


namespace NUMINAMATH_GPT_correct_average_is_26_l378_37881

noncomputable def initial_average : ℕ := 20
noncomputable def number_of_numbers : ℕ := 10
noncomputable def incorrect_number : ℕ := 26
noncomputable def correct_number : ℕ := 86
noncomputable def incorrect_total_sum : ℕ := initial_average * number_of_numbers
noncomputable def correct_total_sum : ℕ := incorrect_total_sum + (correct_number - incorrect_number)
noncomputable def correct_average : ℕ := correct_total_sum / number_of_numbers

theorem correct_average_is_26 :
  correct_average = 26 := by
  sorry

end NUMINAMATH_GPT_correct_average_is_26_l378_37881


namespace NUMINAMATH_GPT_smallest_n_for_multiple_of_7_l378_37801

theorem smallest_n_for_multiple_of_7 (x y : ℤ) (h1 : x % 7 = -1 % 7) (h2 : y % 7 = 2 % 7) :
  ∃ n : ℕ, n > 0 ∧ (x^2 + x * y + y^2 + n) % 7 = 0 ∧ n = 4 :=
sorry

end NUMINAMATH_GPT_smallest_n_for_multiple_of_7_l378_37801


namespace NUMINAMATH_GPT_length_of_platform_l378_37868

noncomputable def train_length : ℝ := 450
noncomputable def signal_pole_time : ℝ := 18
noncomputable def platform_time : ℝ := 39

theorem length_of_platform : 
  ∃ (L : ℝ), 
    (train_length / signal_pole_time = (train_length + L) / platform_time) → 
    L = 525 := 
by
  sorry

end NUMINAMATH_GPT_length_of_platform_l378_37868


namespace NUMINAMATH_GPT_area_of_triangle_DEF_l378_37843

theorem area_of_triangle_DEF :
  let s := 2
  let hexagon_area := (3 * Real.sqrt 3 / 2) * s^2
  let radius := s
  let distance_between_centers := 2 * radius
  let side_of_triangle_DEF := distance_between_centers
  let triangle_area := (Real.sqrt 3 / 4) * side_of_triangle_DEF^2
  triangle_area = 4 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_area_of_triangle_DEF_l378_37843


namespace NUMINAMATH_GPT_john_age_l378_37858

/-
Problem statement:
John is 24 years younger than his dad. The sum of their ages is 68 years.
We need to prove that John is 22 years old.
-/

theorem john_age:
  ∃ (j d : ℕ), (j = d - 24 ∧ j + d = 68) → j = 22 :=
by
  sorry

end NUMINAMATH_GPT_john_age_l378_37858


namespace NUMINAMATH_GPT_solve_inequality_l378_37885

theorem solve_inequality (x : ℝ) :
  (x ^ 2 - 2 * x - 3) * (x ^ 2 - 4 * x + 4) < 0 ↔ (-1 < x ∧ x < 3 ∧ x ≠ 2) := by
  sorry

end NUMINAMATH_GPT_solve_inequality_l378_37885


namespace NUMINAMATH_GPT_geometric_progression_term_count_l378_37860

theorem geometric_progression_term_count
  (q : ℝ) (b4 : ℝ) (S : ℝ) (b1 : ℝ)
  (h1 : q = 1 / 3)
  (h2 : b4 = b1 * (q ^ 3))
  (h3 : S = b1 * (1 - q ^ 5) / (1 - q))
  (h4 : b4 = 1 / 54)
  (h5 : S = 121 / 162) :
  5 = 5 := sorry

end NUMINAMATH_GPT_geometric_progression_term_count_l378_37860


namespace NUMINAMATH_GPT_everton_college_payment_l378_37851

theorem everton_college_payment :
  let num_sci_calculators := 20
  let num_graph_calculators := 25
  let price_sci_calculator := 10
  let price_graph_calculator := 57
  let total_payment := num_sci_calculators * price_sci_calculator + num_graph_calculators * price_graph_calculator
  total_payment = 1625 :=
by
  let num_sci_calculators := 20
  let num_graph_calculators := 25
  let price_sci_calculator := 10
  let price_graph_calculator := 57
  let total_payment := num_sci_calculators * price_sci_calculator + num_graph_calculators * price_graph_calculator
  sorry

end NUMINAMATH_GPT_everton_college_payment_l378_37851


namespace NUMINAMATH_GPT_base7_to_base10_conversion_l378_37805

theorem base7_to_base10_conversion (n : ℤ) (h : n = 2 * 7^2 + 4 * 7^1 + 6 * 7^0) : n = 132 := by
  sorry

end NUMINAMATH_GPT_base7_to_base10_conversion_l378_37805


namespace NUMINAMATH_GPT_common_chord_line_l378_37817

-- Define the first circle equation
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 6*y + 4 = 0

-- Define the second circle equation
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y - 3 = 0

-- Definition of the line equation for the common chord
def line (x y : ℝ) : Prop := 2*x - 2*y + 7 = 0

theorem common_chord_line (x y : ℝ) (h1 : circle1 x y) (h2 : circle2 x y) : line x y :=
by
  sorry

end NUMINAMATH_GPT_common_chord_line_l378_37817


namespace NUMINAMATH_GPT_yuan_exchange_l378_37897

theorem yuan_exchange : 
  ∃ (n : ℕ), n = 5 ∧ ∀ (x y : ℕ), x + 5 * y = 20 → x ≥ 0 ∧ y ≥ 0 :=
by {
  sorry
}

end NUMINAMATH_GPT_yuan_exchange_l378_37897


namespace NUMINAMATH_GPT_chris_sick_weeks_l378_37800

theorem chris_sick_weeks :
  ∀ (h1 : ∀ w : ℕ, w = 4 → 2 * w = 8),
    ∀ (h2 : ∀ h w : ℕ, h = 20 → ∀ m : ℕ, 2 * (w * m) = 160),
    ∀ (h3 : ∀ h : ℕ, h = 180 → 180 - 160 = 20),
    ∀ (h4 : ∀ h w : ℕ, h = 20 → w = 20 → 20 / 20 = 1),
    180 - 160 = (20 / 20) * 20 :=
by
  intros
  sorry

end NUMINAMATH_GPT_chris_sick_weeks_l378_37800


namespace NUMINAMATH_GPT_no_solution_for_x6_eq_2y2_plus_2_l378_37831

theorem no_solution_for_x6_eq_2y2_plus_2 :
  ¬ ∃ (x y : ℤ), x^6 = 2 * y^2 + 2 :=
sorry

end NUMINAMATH_GPT_no_solution_for_x6_eq_2y2_plus_2_l378_37831


namespace NUMINAMATH_GPT_range_of_m_l378_37895

noncomputable def f (x m : ℝ) : ℝ := (1 / 4) * x^4 - (2 / 3) * x^3 + m

theorem range_of_m (m : ℝ) : (∀ x : ℝ, f x m + (1 / 3) ≥ 0) ↔ m ≥ 1 := 
sorry

end NUMINAMATH_GPT_range_of_m_l378_37895


namespace NUMINAMATH_GPT_robot_handling_capacity_l378_37809

variables (x : ℝ) (A B : ℝ)

def robot_speed_condition1 : Prop :=
  A = B + 30

def robot_speed_condition2 : Prop :=
  1000 / A = 800 / B

theorem robot_handling_capacity
  (h1 : robot_speed_condition1 A B)
  (h2 : robot_speed_condition2 A B) :
  B = 120 ∧ A = 150 :=
by
  sorry

end NUMINAMATH_GPT_robot_handling_capacity_l378_37809


namespace NUMINAMATH_GPT_geometric_sequence_reciprocals_sum_l378_37898

theorem geometric_sequence_reciprocals_sum :
  ∃ (a : ℕ → ℝ) (q : ℝ), 
    (a 1 = 2) ∧ 
    (a 1 + a 3 + a 5 = 14) ∧ 
    (∀ n : ℕ, a (n + 1) = a n * q) → 
      (1 / a 1 + 1 / a 3 + 1 / a 5 = 7 / 8) :=
sorry

end NUMINAMATH_GPT_geometric_sequence_reciprocals_sum_l378_37898


namespace NUMINAMATH_GPT_value_of_2_Z_6_l378_37803

def Z (a b : ℝ) : ℝ := b + 10 * a - a^2

theorem value_of_2_Z_6 : Z 2 6 = 22 :=
by
  sorry

end NUMINAMATH_GPT_value_of_2_Z_6_l378_37803


namespace NUMINAMATH_GPT_rectangle_breadth_l378_37822

theorem rectangle_breadth (l b : ℕ) (hl : l = 15) (h : l * b = 15 * b) (h2 : l - b = 10) : b = 5 := 
sorry

end NUMINAMATH_GPT_rectangle_breadth_l378_37822


namespace NUMINAMATH_GPT_find_ac_find_a_and_c_l378_37887

variables (A B C a b c : ℝ)

-- Condition: Angles A, B, C form an arithmetic sequence.
def arithmetic_sequence := 2 * B = A + C

-- Condition: Area of the triangle is sqrt(3)/2.
def area_triangle := (1/2) * a * c * (Real.sin B) = (Real.sqrt 3) / 2

-- Condition: b = sqrt(3)
def b_sqrt3 := b = Real.sqrt 3

-- Goal 1: To prove that ac = 2.
theorem find_ac (h1 : arithmetic_sequence A B C) (h2 : area_triangle a c B) : a * c = 2 :=
sorry

-- Goal 2: To prove a = 2 and c = 1 given the additional condition.
theorem find_a_and_c (h1 : arithmetic_sequence A B C) (h2 : area_triangle a c B) (h3 : b_sqrt3 b) (h4 : a > c) : a = 2 ∧ c = 1 :=
sorry

end NUMINAMATH_GPT_find_ac_find_a_and_c_l378_37887


namespace NUMINAMATH_GPT_dishes_combinations_is_correct_l378_37836

-- Define the number of dishes
def num_dishes : ℕ := 15

-- Define the number of appetizers
def num_appetizers : ℕ := 5

-- Compute the total number of combinations
def combinations_of_dishes : ℕ :=
  num_dishes * num_dishes * num_appetizers

-- The theorem that states the total number of combinations is 1125
theorem dishes_combinations_is_correct :
  combinations_of_dishes = 1125 := by
  sorry

end NUMINAMATH_GPT_dishes_combinations_is_correct_l378_37836


namespace NUMINAMATH_GPT_number_difference_l378_37852

theorem number_difference:
  ∀ (number : ℝ), 0.30 * number = 63.0000000000001 →
  (3 / 7) * number - 0.40 * number = 6.00000000000006 := by
  sorry

end NUMINAMATH_GPT_number_difference_l378_37852


namespace NUMINAMATH_GPT_cattle_train_left_6_hours_before_l378_37844

theorem cattle_train_left_6_hours_before 
  (Vc : ℕ) (Vd : ℕ) (T : ℕ) 
  (h1 : Vc = 56)
  (h2 : Vd = Vc - 33)
  (h3 : 12 * Vd + 12 * Vc + T * Vc = 1284) : 
  T = 6 := 
by
  sorry

end NUMINAMATH_GPT_cattle_train_left_6_hours_before_l378_37844


namespace NUMINAMATH_GPT_prism_surface_area_l378_37847

theorem prism_surface_area (P : ℝ) (h : ℝ) (S : ℝ) (s: ℝ) 
  (hP : P = 4)
  (hh : h = 2) 
  (hs : s = 1) 
  (h_surf_top : S = s * s) 
  (h_lat : S = 8) : 
  S = 10 := 
sorry

end NUMINAMATH_GPT_prism_surface_area_l378_37847


namespace NUMINAMATH_GPT_side_lengths_are_10_and_50_l378_37810

-- Define variables used in the problem
variables {s t : ℕ}

-- Define the conditions
def condition1 (s t : ℕ) : Prop := 4 * s = 20 * t
def condition2 (s t : ℕ) : Prop := s + t = 60

-- Prove that given the conditions, the side lengths of the squares are 10 and 50
theorem side_lengths_are_10_and_50 (s t : ℕ) (h1 : condition1 s t) (h2 : condition2 s t) : (s = 50 ∧ t = 10) ∨ (s = 10 ∧ t = 50) :=
by sorry

end NUMINAMATH_GPT_side_lengths_are_10_and_50_l378_37810


namespace NUMINAMATH_GPT_events_per_coach_l378_37857

theorem events_per_coach {students events_per_student coaches events total_participations total_events : ℕ} 
  (h1 : students = 480) 
  (h2 : events_per_student = 4) 
  (h3 : (students * events_per_student) = total_participations) 
  (h4 : ¬ students * events_per_student ≠ total_participations)
  (h5 : total_participations = 1920) 
  (h6 : (total_participations / 20) = total_events) 
  (h7 : ¬ total_participations / 20 ≠ total_events)
  (h8 : total_events = 96)
  (h9 : coaches = 16) :
  (total_events / coaches) = 6 := sorry

end NUMINAMATH_GPT_events_per_coach_l378_37857


namespace NUMINAMATH_GPT_simplify_expr_at_sqrt6_l378_37830

noncomputable def simplifyExpression (x : ℝ) : ℝ :=
  (1 / (Real.sqrt (3 + x) * Real.sqrt (x + 2)) + 1 / (Real.sqrt (3 - x) * Real.sqrt (x - 2))) /
  (1 / (Real.sqrt (3 + x) * Real.sqrt (x + 2)) - 1 / (Real.sqrt (3 - x) * Real.sqrt (x - 2)))

theorem simplify_expr_at_sqrt6 : simplifyExpression (Real.sqrt 6) = - (Real.sqrt 6) / 2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expr_at_sqrt6_l378_37830


namespace NUMINAMATH_GPT_ratio_h_w_l378_37855

-- Definitions from conditions
variables (h w : ℝ)
variables (XY YZ : ℝ)
variables (h_pos : 0 < h) (w_pos : 0 < w) -- heights and widths are positive
variables (XY_pos : 0 < XY) (YZ_pos : 0 < YZ) -- segment lengths are positive

-- Given that in the right-angled triangle ∆XYZ, YZ = 2 * XY
axiom YZ_eq_2XY : YZ = 2 * XY

-- Prove that h / w = 3 / 8
theorem ratio_h_w (H : XY / YZ = 4 * h / (3 * w)) : h / w = 3 / 8 :=
by {
  -- Use the axioms and given conditions here to prove H == ratio
  sorry
}

end NUMINAMATH_GPT_ratio_h_w_l378_37855


namespace NUMINAMATH_GPT_percentage_gain_is_20_percent_l378_37892

theorem percentage_gain_is_20_percent (manufacturing_cost transportation_cost total_shoes selling_price : ℝ)
(h1 : manufacturing_cost = 220)
(h2 : transportation_cost = 500)
(h3 : total_shoes = 100)
(h4 : selling_price = 270) :
  let cost_per_shoe := manufacturing_cost + transportation_cost / total_shoes
  let profit_per_shoe := selling_price - cost_per_shoe
  let percentage_gain := (profit_per_shoe / cost_per_shoe) * 100
  percentage_gain = 20 :=
by
  sorry

end NUMINAMATH_GPT_percentage_gain_is_20_percent_l378_37892


namespace NUMINAMATH_GPT_problem_solution_l378_37842

theorem problem_solution (x m : ℝ) (h1 : x ≠ 0) (h2 : x / (x^2 - m*x + 1) = 1) :
  x^3 / (x^6 - m^3 * x^3 + 1) = 1 / (3 * m^2 - 2) :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l378_37842


namespace NUMINAMATH_GPT_base_4_last_digit_of_389_l378_37875

theorem base_4_last_digit_of_389 : (389 % 4) = 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_base_4_last_digit_of_389_l378_37875


namespace NUMINAMATH_GPT_find_2x_2y_2z_l378_37872

theorem find_2x_2y_2z (x y z : ℝ) 
  (h1 : y + z = 10 - 2 * x)
  (h2 : x + z = -12 - 4 * y)
  (h3 : x + y = 5 - 2 * z) : 
  2 * x + 2 * y + 2 * z = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_2x_2y_2z_l378_37872


namespace NUMINAMATH_GPT_proper_subsets_B_l378_37889

theorem proper_subsets_B (A B : Set ℝ) (a : ℝ)
  (hA : A = {x | x^2 + 2*x + 1 = 0})
  (hA_singleton : A = {a})
  (hB : B = {x | x^2 + a*x = 0}) :
  a = -1 ∧ 
  B = {0, 1} ∧
  (∀ S, S ∈ ({∅, {0}, {1}} : Set (Set ℝ)) ↔ S ⊂ B) :=
by
  -- Proof not provided, only statement required.
  sorry

end NUMINAMATH_GPT_proper_subsets_B_l378_37889


namespace NUMINAMATH_GPT_children_playing_both_sports_l378_37863

variable (total_children : ℕ) (T : ℕ) (S : ℕ) (N : ℕ)

theorem children_playing_both_sports 
  (h1 : total_children = 38) 
  (h2 : T = 19) 
  (h3 : S = 21) 
  (h4 : N = 10) : 
  (T + S) - (total_children - N) = 12 := 
by
  sorry

end NUMINAMATH_GPT_children_playing_both_sports_l378_37863


namespace NUMINAMATH_GPT_map_distance_correct_l378_37841

noncomputable def distance_on_map : ℝ :=
  let speed := 60  -- miles per hour
  let time := 6.5  -- hours
  let scale := 0.01282051282051282 -- inches per mile
  let actual_distance := speed * time -- in miles
  actual_distance * scale -- convert to inches

theorem map_distance_correct :
  distance_on_map = 5 :=
by 
  sorry

end NUMINAMATH_GPT_map_distance_correct_l378_37841


namespace NUMINAMATH_GPT_fraction_of_juniors_equals_seniors_l378_37827

theorem fraction_of_juniors_equals_seniors (J S : ℕ) (h1 : 0 < J) (h2 : 0 < S) (h3 : J * 7 = 4 * (J + S)) : J / S = 4 / 3 :=
sorry

end NUMINAMATH_GPT_fraction_of_juniors_equals_seniors_l378_37827


namespace NUMINAMATH_GPT_disjoint_sets_condition_l378_37814

theorem disjoint_sets_condition (A B : Set ℕ) (h_disjoint: Disjoint A B) (h_union: A ∪ B = Set.univ) :
  ∀ n : ℕ, ∃ a b : ℕ, a > n ∧ b > n ∧ a ≠ b ∧ 
             ((a ∈ A ∧ b ∈ A ∧ a + b ∈ A) ∨ (a ∈ B ∧ b ∈ B ∧ a + b ∈ B)) := 
by
  sorry

end NUMINAMATH_GPT_disjoint_sets_condition_l378_37814


namespace NUMINAMATH_GPT_divides_2pow18_minus_1_l378_37828

theorem divides_2pow18_minus_1 (n : ℕ) : 20 ≤ n ∧ n < 30 ∧ (n ∣ 2^18 - 1) ↔ (n = 19 ∨ n = 27) := by
  sorry

end NUMINAMATH_GPT_divides_2pow18_minus_1_l378_37828


namespace NUMINAMATH_GPT_abs_neg_two_l378_37891

theorem abs_neg_two : abs (-2) = 2 :=
by
  sorry

end NUMINAMATH_GPT_abs_neg_two_l378_37891


namespace NUMINAMATH_GPT_arithmetic_sequence_squares_l378_37866

theorem arithmetic_sequence_squares (a b c : ℝ) :
  (1 / (a + b) - 1 / (b + c) = 1 / (c + a) - 1 / (b + c)) →
  (2 * b^2 = a^2 + c^2) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_squares_l378_37866


namespace NUMINAMATH_GPT_train_speed_l378_37813

theorem train_speed 
  (length : ℝ)
  (time : ℝ)
  (relative_speed : ℝ)
  (conversion_factor : ℝ)
  (h_length : length = 120)
  (h_time : time = 4)
  (h_relative_speed : relative_speed = 60)
  (h_conversion_factor : conversion_factor = 3.6) :
  (relative_speed / 2) * conversion_factor = 108 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_l378_37813


namespace NUMINAMATH_GPT_gcd_18_30_l378_37886

-- Define the two numbers
def num1 : ℕ := 18
def num2 : ℕ := 30

-- State the theorem to find the gcd
theorem gcd_18_30 : Nat.gcd num1 num2 = 6 := by
  sorry

end NUMINAMATH_GPT_gcd_18_30_l378_37886


namespace NUMINAMATH_GPT_purely_imaginary_iff_l378_37899

theorem purely_imaginary_iff (a : ℝ) :
  (a^2 - a - 2 = 0 ∧ (|a - 1| - 1 ≠ 0)) ↔ a = -1 :=
by
  sorry

end NUMINAMATH_GPT_purely_imaginary_iff_l378_37899


namespace NUMINAMATH_GPT_son_age_l378_37894

theorem son_age:
  ∃ S M : ℕ, 
  (M = S + 20) ∧ 
  (M + 2 = 2 * (S + 2)) ∧ 
  (S = 18) := 
by
  sorry

end NUMINAMATH_GPT_son_age_l378_37894


namespace NUMINAMATH_GPT_radius_of_regular_polygon_l378_37829

theorem radius_of_regular_polygon :
  ∃ (p : ℝ), 
        (∀ n : ℕ, 3 ≤ n → (n : ℝ) = 6) ∧ 
        (∀ s : ℝ, s = 2 → s = 2) → 
        (∀ i : ℝ, i = 720 → i = 720) →
        (∀ e : ℝ, e = 360 → e = 360) →
        p = 2 :=
by
  sorry

end NUMINAMATH_GPT_radius_of_regular_polygon_l378_37829


namespace NUMINAMATH_GPT_not_all_inequalities_hold_l378_37802

theorem not_all_inequalities_hold (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
(hlt_a : a < 1) (hlt_b : b < 1) (hlt_c : c < 1) :
  ¬(a * (1 - b) > 1 / 4 ∧ b * (1 - c) > 1 / 4 ∧ c * (1 - a) > 1 / 4) :=
by
  sorry

end NUMINAMATH_GPT_not_all_inequalities_hold_l378_37802


namespace NUMINAMATH_GPT_car_arrives_first_and_earlier_l378_37850

-- Define the conditions
def total_intersections : ℕ := 11
def total_blocks : ℕ := 12
def green_time : ℕ := 3
def red_time : ℕ := 1
def car_block_time : ℕ := 1
def bus_block_time : ℕ := 2

-- Define the functions that compute the travel times
def car_travel_time (blocks : ℕ) : ℕ :=
  (blocks / 3) * (green_time + red_time) + (blocks % 3 * car_block_time)

def bus_travel_time (blocks : ℕ) : ℕ :=
  blocks * bus_block_time

-- Define the theorem to prove
theorem car_arrives_first_and_earlier :
  car_travel_time total_blocks < bus_travel_time total_blocks ∧
  bus_travel_time total_blocks - car_travel_time total_blocks = 9 := 
by
  sorry

end NUMINAMATH_GPT_car_arrives_first_and_earlier_l378_37850


namespace NUMINAMATH_GPT_a8_value_l378_37846

variable {an : ℕ → ℕ}

def S (n : ℕ) : ℕ := n ^ 2

theorem a8_value : an 8 = S 8 - S 7 := by
  sorry

end NUMINAMATH_GPT_a8_value_l378_37846


namespace NUMINAMATH_GPT_a2_a3_equals_20_l378_37879

-- Sequence definition
def a_n (n : ℕ) : ℕ :=
  if n % 2 = 1 then 3 * n + 1 else 2 * n - 2

-- Proof that a_2 * a_3 = 20
theorem a2_a3_equals_20 :
  a_n 2 * a_n 3 = 20 :=
by
  sorry

end NUMINAMATH_GPT_a2_a3_equals_20_l378_37879


namespace NUMINAMATH_GPT_james_drinks_per_day_l378_37880

-- condition: James buys 5 packs of sodas, each contains 12 sodas
def num_packs : Nat := 5
def sodas_per_pack : Nat := 12
def sodas_bought : Nat := num_packs * sodas_per_pack

-- condition: James already had 10 sodas
def sodas_already_had : Nat := 10

-- condition: James finishes all the sodas in 1 week (7 days)
def days_in_week : Nat := 7

-- total sodas
def total_sodas : Nat := sodas_bought + sodas_already_had

-- number of sodas james drinks per day
def sodas_per_day : Nat := 10

-- proof problem
theorem james_drinks_per_day : (total_sodas / days_in_week) = sodas_per_day :=
  sorry

end NUMINAMATH_GPT_james_drinks_per_day_l378_37880


namespace NUMINAMATH_GPT_fib_inequality_l378_37815

def Fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => Fib n + Fib (n + 1)

theorem fib_inequality {n : ℕ} (h : 2 ≤ n) : Fib (n + 5) > 10 * Fib n :=
  sorry

end NUMINAMATH_GPT_fib_inequality_l378_37815


namespace NUMINAMATH_GPT_circle_center_and_radius_sum_l378_37818

theorem circle_center_and_radius_sum :
  let a := -4
  let b := -8
  let r := Real.sqrt 17
  a + b + r = -12 + Real.sqrt 17 :=
by
  sorry

end NUMINAMATH_GPT_circle_center_and_radius_sum_l378_37818


namespace NUMINAMATH_GPT_num_kids_eq_3_l378_37873

def mom_eyes : ℕ := 1
def dad_eyes : ℕ := 3
def kid_eyes : ℕ := 4
def total_eyes : ℕ := 16

theorem num_kids_eq_3 : ∃ k : ℕ, 1 + 3 + 4 * k = 16 ∧ k = 3 := by
  sorry

end NUMINAMATH_GPT_num_kids_eq_3_l378_37873


namespace NUMINAMATH_GPT_correct_mean_251_l378_37871

theorem correct_mean_251
  (n : ℕ) (incorrect_mean : ℕ) (wrong_val : ℕ) (correct_val : ℕ)
  (h1 : n = 30) (h2 : incorrect_mean = 250) (h3 : wrong_val = 135) (h4 : correct_val = 165) :
  ((incorrect_mean * n + (correct_val - wrong_val)) / n) = 251 :=
by
  sorry

end NUMINAMATH_GPT_correct_mean_251_l378_37871


namespace NUMINAMATH_GPT_fill_time_first_and_fourth_taps_l378_37874

noncomputable def pool_filling_time (m x y z u : ℝ) (h₁ : 2 * (x + y) = m) (h₂ : 3 * (y + z) = m) (h₃ : 4 * (z + u) = m) : ℝ :=
  m / (x + u)

theorem fill_time_first_and_fourth_taps (m x y z u : ℝ) (h₁ : 2 * (x + y) = m) (h₂ : 3 * (y + z) = m) (h₃ : 4 * (z + u) = m) :
  pool_filling_time m x y z u h₁ h₂ h₃ = 12 / 5 :=
sorry

end NUMINAMATH_GPT_fill_time_first_and_fourth_taps_l378_37874


namespace NUMINAMATH_GPT_rectangular_field_perimeter_l378_37824

theorem rectangular_field_perimeter (A L : ℝ) (h1 : A = 300) (h2 : L = 15) : 
  let W := A / L 
  let P := 2 * (L + W)
  P = 70 := by
  sorry

end NUMINAMATH_GPT_rectangular_field_perimeter_l378_37824


namespace NUMINAMATH_GPT_exists_natural_multiple_of_2015_with_digit_sum_2015_l378_37808

-- Definition of sum of digits function
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Proposition that we need to prove
theorem exists_natural_multiple_of_2015_with_digit_sum_2015 :
  ∃ n : ℕ, (2015 ∣ n) ∧ sum_of_digits n = 2015 :=
sorry

end NUMINAMATH_GPT_exists_natural_multiple_of_2015_with_digit_sum_2015_l378_37808


namespace NUMINAMATH_GPT_midpoint_integer_of_five_points_l378_37845

theorem midpoint_integer_of_five_points 
  (P : Fin 5 → ℤ × ℤ) 
  (distinct : Function.Injective P) :
  ∃ i j : Fin 5, i ≠ j ∧ (P i).1 + (P j).1 % 2 = 0 ∧ (P i).2 + (P j).2 % 2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_midpoint_integer_of_five_points_l378_37845


namespace NUMINAMATH_GPT_find_x_l378_37888

-- Define that x is a real number and positive
variable (x : ℝ)
variable (hx_pos : 0 < x)

-- Define the floor function and the main equation
variable (hx_eq : ⌊x⌋ * x = 90)

theorem find_x (h : ⌊x⌋ * x = 90) (hx_pos : 0 < x) : ⌊x⌋ = 9 ∧ x = 10 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l378_37888


namespace NUMINAMATH_GPT_batsman_average_after_11th_inning_l378_37840

theorem batsman_average_after_11th_inning (x : ℝ) (h : 10 * x + 110 = 11 * (x + 5)) : 
    (10 * x + 110) / 11 = 60 := by
  sorry

end NUMINAMATH_GPT_batsman_average_after_11th_inning_l378_37840


namespace NUMINAMATH_GPT_kim_initial_classes_l378_37820

-- Necessary definitions for the problem
def hours_per_class := 2
def total_hours_after_dropping := 6
def classes_after_dropping := total_hours_after_dropping / hours_per_class
def initial_classes := classes_after_dropping + 1

theorem kim_initial_classes : initial_classes = 4 :=
by
  -- Proof will be derived here
  sorry

end NUMINAMATH_GPT_kim_initial_classes_l378_37820


namespace NUMINAMATH_GPT_min_cos_C_l378_37849

theorem min_cos_C (a b c : ℝ) (A B C : ℝ) (h1 : a^2 + b^2 = (5 / 2) * c^2) 
  (h2 : ∃ (A B C : ℝ), a ≠ b ∧ 
    c = (a ^ 2 + b ^ 2 - 2 * a * b * (Real.cos C))) : 
  ∃ (C : ℝ), Real.cos C = 3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_min_cos_C_l378_37849


namespace NUMINAMATH_GPT_family_total_weight_gain_l378_37870

def orlando_gain : ℕ := 5
def jose_gain : ℕ := 2 * orlando_gain + 2
def fernando_gain : ℕ := (jose_gain / 2) - 3
def total_weight_gain : ℕ := orlando_gain + jose_gain + fernando_gain

theorem family_total_weight_gain : total_weight_gain = 20 := by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_family_total_weight_gain_l378_37870


namespace NUMINAMATH_GPT_valid_votes_other_candidate_l378_37835

theorem valid_votes_other_candidate (total_votes : ℕ) (invalid_percentage : ℕ) (candidate1_percentage : ℕ) (valid_votes_other_candidate : ℕ) : 
  total_votes = 7500 → 
  invalid_percentage = 20 → 
  candidate1_percentage = 55 → 
  valid_votes_other_candidate = 2700 :=
by
  sorry

end NUMINAMATH_GPT_valid_votes_other_candidate_l378_37835


namespace NUMINAMATH_GPT_basketball_lineup_count_l378_37834

theorem basketball_lineup_count :
  (∃ (players : Finset ℕ), players.card = 15) → 
  ∃ centers power_forwards small_forwards shooting_guards point_guards sixth_men : ℕ,
  ∃ b : Fin (15) → Fin (15),
  15 * 14 * 13 * 12 * 11 * 10 = 360360 
:= by sorry

end NUMINAMATH_GPT_basketball_lineup_count_l378_37834


namespace NUMINAMATH_GPT_study_group_members_l378_37867

theorem study_group_members (x : ℕ) (h : x * (x - 1) = 90) : x = 10 :=
sorry

end NUMINAMATH_GPT_study_group_members_l378_37867


namespace NUMINAMATH_GPT_find_a_2018_l378_37848

noncomputable def a : ℕ → ℕ
| n => if n > 0 then 2 * n else sorry

theorem find_a_2018 (a : ℕ → ℕ) 
  (h : ∀ m n : ℕ, m > 0 ∧ n > 0 → a m + a n = a (m + n)) 
  (h1 : a 1 = 2) : a 2018 = 4036 := by
  sorry

end NUMINAMATH_GPT_find_a_2018_l378_37848


namespace NUMINAMATH_GPT_puppy_food_cost_l378_37862

theorem puppy_food_cost :
  let puppy_cost : ℕ := 10
  let days_in_week : ℕ := 7
  let total_number_of_weeks : ℕ := 3
  let cups_per_day : ℚ := 1 / 3
  let cups_per_bag : ℚ := 3.5
  let cost_per_bag : ℕ := 2
  let total_days := total_number_of_weeks * days_in_week
  let total_cups := total_days * cups_per_day
  let total_bags := total_cups / cups_per_bag
  let food_cost := total_bags * cost_per_bag
  let total_cost := puppy_cost + food_cost
  total_cost = 14 := by
  sorry

end NUMINAMATH_GPT_puppy_food_cost_l378_37862


namespace NUMINAMATH_GPT_problem1_problem2_l378_37890

-- Define variables
variables {x y m : ℝ}
variables (h1 : x + y > 0) (h2 : xy ≠ 0)

-- Problem (1): Prove that x^3 + y^3 ≥ x^2 y + y^2 x
theorem problem1 (h1 : x + y > 0) (h2 : xy ≠ 0) : x^3 + y^3 ≥ x^2 * y + y^2 * x :=
sorry

-- Problem (2): Given the conditions, the range of m is [-6, 2]
theorem problem2 (h1 : x + y > 0) (h2 : xy ≠ 0) (h3 : (x / y^2) + (y / x^2) ≥ (m / 2) * ((1 / x) + (1 / y))) : m ∈ Set.Icc (-6 : ℝ) 2 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l378_37890


namespace NUMINAMATH_GPT_calculation_proof_l378_37853

theorem calculation_proof : 
  2 * Real.tan (Real.pi / 3) - (-2023) ^ 0 + (1 / 2) ^ (-1 : ℤ) + abs (Real.sqrt 3 - 1) = 3 * Real.sqrt 3 := 
by
  sorry

end NUMINAMATH_GPT_calculation_proof_l378_37853


namespace NUMINAMATH_GPT_sum_m_n_l378_37865

theorem sum_m_n (m n : ℤ) (h1 : m^2 - n^2 = 18) (h2 : m - n = 9) : m + n = 2 := 
by
  sorry

end NUMINAMATH_GPT_sum_m_n_l378_37865


namespace NUMINAMATH_GPT_corresponding_angles_equal_l378_37883

-- Definition of corresponding angles (this should be previously defined, so here we assume it is just a predicate)
def CorrespondingAngles (a b : Angle) : Prop := sorry

-- The main theorem to be proven
theorem corresponding_angles_equal (a b : Angle) (h : CorrespondingAngles a b) : a = b := 
sorry

end NUMINAMATH_GPT_corresponding_angles_equal_l378_37883


namespace NUMINAMATH_GPT_avg_rest_students_l378_37861

/- Definitions based on conditions -/
def total_students : ℕ := 28
def students_scored_95 : ℕ := 4
def students_scored_0 : ℕ := 3
def avg_whole_class : ℚ := 47.32142857142857
def total_marks_95 : ℚ := students_scored_95 * 95
def total_marks_0 : ℚ := students_scored_0 * 0
def marks_whole_class : ℚ := total_students * avg_whole_class
def rest_students : ℕ := total_students - students_scored_95 - students_scored_0

/- Theorem to prove the average of the rest students given the conditions -/
theorem avg_rest_students : (total_marks_95 + total_marks_0 + rest_students * 45) = marks_whole_class :=
by
  sorry

end NUMINAMATH_GPT_avg_rest_students_l378_37861


namespace NUMINAMATH_GPT_fertilizer_needed_per_acre_l378_37833

-- Definitions for the conditions
def horse_daily_fertilizer : ℕ := 5 -- Each horse produces 5 gallons of fertilizer per day.
def horses : ℕ := 80 -- Janet has 80 horses.
def days : ℕ := 25 -- It takes 25 days until all her fields are fertilized.
def total_acres : ℕ := 20 -- Janet's farmland is 20 acres.

-- Calculated intermediate values
def total_fertilizer : ℕ := horse_daily_fertilizer * horses * days -- Total fertilizer produced
def fertilizer_per_acre : ℕ := total_fertilizer / total_acres -- Fertilizer needed per acre

-- Theorem to prove
theorem fertilizer_needed_per_acre : fertilizer_per_acre = 500 := by
  sorry

end NUMINAMATH_GPT_fertilizer_needed_per_acre_l378_37833


namespace NUMINAMATH_GPT_total_profit_is_8800_l378_37896

variable (A B C : Type) [CommRing A] [CommRing B] [CommRing C]

variable (investment_A investment_B investment_C : ℝ)
variable (total_profit : ℝ)

-- Conditions
def A_investment_three_times_B (investment_A investment_B : ℝ) : Prop :=
  investment_A = 3 * investment_B

def B_invest_two_thirds_C (investment_B investment_C : ℝ) : Prop :=
  investment_B = 2 / 3 * investment_C

def B_share_is_1600 (investment_B total_profit : ℝ) : Prop :=
  1600 = (2 / 11) * total_profit

theorem total_profit_is_8800 :
  A_investment_three_times_B investment_A investment_B →
  B_invest_two_thirds_C investment_B investment_C →
  B_share_is_1600 investment_B total_profit →
  total_profit = 8800 :=
by
  intros
  sorry

end NUMINAMATH_GPT_total_profit_is_8800_l378_37896


namespace NUMINAMATH_GPT_min_books_borrowed_l378_37825

theorem min_books_borrowed 
    (h1 : 12 * 1 = 12) 
    (h2 : 10 * 2 = 20) 
    (h3 : 2 = 2) 
    (h4 : 32 = 32) 
    (h5 : (32 * 2 = 64))
    (h6 : ∀ x, x ≤ 11) :
    ∃ (x : ℕ), (8 * x = 32) ∧ x ≤ 11 := 
  sorry

end NUMINAMATH_GPT_min_books_borrowed_l378_37825


namespace NUMINAMATH_GPT_geometric_sequence_from_second_term_l378_37812

open Nat

-- Define the sequence S_n
def S (n : ℕ) : ℕ := 
  match n with
  | 0 => 0 -- to handle the 0th term which is typically not used here
  | 1 => 1
  | 2 => 2
  | n + 3 => 3 * S (n + 2) - 2 * S (n + 1) -- given recurrence relation

-- Define the sequence a_n
def a (n : ℕ) : ℕ := 
  match n with
  | 0 => 0 -- Define a_0 as 0 since it's not used in the problem
  | 1 => 1 -- a1
  | n + 2 => S (n + 2) - S (n + 1) -- a_n = S_n - S_(n-1)

theorem geometric_sequence_from_second_term :
  ∀ n ≥ 2, a (n + 1) = 2 * a n := by
  -- Proof step not provided
  sorry

end NUMINAMATH_GPT_geometric_sequence_from_second_term_l378_37812


namespace NUMINAMATH_GPT_find_m_l378_37864

def triangle (x y : ℤ) := x * y + x + y

theorem find_m (m : ℤ) (h : triangle 2 m = -16) : m = -6 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l378_37864


namespace NUMINAMATH_GPT_origami_papers_per_cousin_l378_37832

/-- Haley has 48 origami papers and 6 cousins. Each cousin should receive the same number of papers. -/
theorem origami_papers_per_cousin : ∀ (total_papers : ℕ) (number_of_cousins : ℕ),
  total_papers = 48 → number_of_cousins = 6 → total_papers / number_of_cousins = 8 :=
by
  intros total_papers number_of_cousins
  sorry

end NUMINAMATH_GPT_origami_papers_per_cousin_l378_37832


namespace NUMINAMATH_GPT_simplify_fraction_l378_37816

theorem simplify_fraction : (3 ^ 100 + 3 ^ 98) / (3 ^ 100 - 3 ^ 98) = 5 / 4 := 
by sorry

end NUMINAMATH_GPT_simplify_fraction_l378_37816


namespace NUMINAMATH_GPT_sum_of_legs_of_right_triangle_l378_37807

theorem sum_of_legs_of_right_triangle (y : ℤ) (hyodd : y % 2 = 1) (hyp : y ^ 2 + (y + 2) ^ 2 = 17 ^ 2) :
  y + (y + 2) = 24 :=
sorry

end NUMINAMATH_GPT_sum_of_legs_of_right_triangle_l378_37807


namespace NUMINAMATH_GPT_gilbert_herb_plants_count_l378_37893

variable (initial_basil : Nat) (initial_parsley : Nat) (initial_mint : Nat)
variable (dropped_basil_seeds : Nat) (rabbit_ate_all_mint : Bool)

def total_initial_plants (initial_basil initial_parsley initial_mint : Nat) : Nat :=
  initial_basil + initial_parsley + initial_mint

def total_plants_after_dropping_seeds (initial_basil initial_parsley initial_mint dropped_basil_seeds : Nat) : Nat :=
  total_initial_plants initial_basil initial_parsley initial_mint + dropped_basil_seeds

def total_plants_after_rabbit (initial_basil initial_parsley initial_mint dropped_basil_seeds : Nat) (rabbit_ate_all_mint : Bool) : Nat :=
  if rabbit_ate_all_mint then 
    total_plants_after_dropping_seeds initial_basil initial_parsley initial_mint dropped_basil_seeds - initial_mint 
  else 
    total_plants_after_dropping_seeds initial_basil initial_parsley initial_mint dropped_basil_seeds

theorem gilbert_herb_plants_count
  (h1 : initial_basil = 3)
  (h2 : initial_parsley = 1)
  (h3 : initial_mint = 2)
  (h4 : dropped_basil_seeds = 1)
  (h5 : rabbit_ate_all_mint = true) :
  total_plants_after_rabbit initial_basil initial_parsley initial_mint dropped_basil_seeds rabbit_ate_all_mint = 5 := by
  sorry

end NUMINAMATH_GPT_gilbert_herb_plants_count_l378_37893


namespace NUMINAMATH_GPT_handshakes_at_gathering_l378_37819

noncomputable def total_handshakes : Nat :=
  let twins := 16
  let triplets := 15
  let handshakes_among_twins := twins * 14 / 2
  let handshakes_among_triplets := 0
  let cross_handshakes := twins * triplets
  handshakes_among_twins + handshakes_among_triplets + cross_handshakes

theorem handshakes_at_gathering : total_handshakes = 352 := 
by
  -- By substituting the values, we can solve and show that the total handshakes equal to 352.
  sorry

end NUMINAMATH_GPT_handshakes_at_gathering_l378_37819


namespace NUMINAMATH_GPT_work_days_by_a_l378_37823

-- Given
def work_days_by_b : ℕ := 10  -- B can do the work alone in 10 days
def combined_work_days : ℕ := 5  -- A and B together can do the work in 5 days

-- Question: In how many days can A do the work alone?
def days_for_a_work_alone : ℕ := 10  -- The correct answer from the solution

-- Proof statement
theorem work_days_by_a (x : ℕ) : 
  ((1 : ℝ) / (x : ℝ) + (1 : ℝ) / (work_days_by_b : ℝ) = (1 : ℝ) / (combined_work_days : ℝ)) → 
  x = days_for_a_work_alone :=
by 
  sorry

end NUMINAMATH_GPT_work_days_by_a_l378_37823
