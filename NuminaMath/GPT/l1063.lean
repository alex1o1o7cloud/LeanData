import Mathlib

namespace NUMINAMATH_GPT_total_journey_distance_l1063_106354

variable (D : ℝ) (T : ℝ) (v₁ : ℝ) (v₂ : ℝ)

theorem total_journey_distance :
  T = 10 → 
  v₁ = 21 → 
  v₂ = 24 → 
  (T = (D / (2 * v₁)) + (D / (2 * v₂))) → 
  D = 224 :=
by
  intros hT hv₁ hv₂ hDistance
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_total_journey_distance_l1063_106354


namespace NUMINAMATH_GPT_total_people_in_cars_by_end_of_race_l1063_106348

-- Define the initial conditions and question
def initial_num_cars : ℕ := 20
def initial_num_passengers_per_car : ℕ := 2
def initial_num_drivers_per_car : ℕ := 1
def extra_passengers_per_car : ℕ := 1

-- Define the number of people per car initially
def initial_people_per_car : ℕ := initial_num_passengers_per_car + initial_num_drivers_per_car

-- Define the number of people per car after gaining extra passenger
def final_people_per_car : ℕ := initial_people_per_car + extra_passengers_per_car

-- The statement to be proven
theorem total_people_in_cars_by_end_of_race : initial_num_cars * final_people_per_car = 80 := by
  -- Prove the theorem
  sorry

end NUMINAMATH_GPT_total_people_in_cars_by_end_of_race_l1063_106348


namespace NUMINAMATH_GPT_mary_cut_roses_l1063_106315

theorem mary_cut_roses (initial_roses add_roses total_roses : ℕ) (h1 : initial_roses = 6) (h2 : total_roses = 16) (h3 : total_roses = initial_roses + add_roses) : add_roses = 10 :=
by
  sorry

end NUMINAMATH_GPT_mary_cut_roses_l1063_106315


namespace NUMINAMATH_GPT_unique_combinations_bathing_suits_l1063_106383

theorem unique_combinations_bathing_suits
  (men_styles : ℕ) (men_sizes : ℕ) (men_colors : ℕ)
  (women_styles : ℕ) (women_sizes : ℕ) (women_colors : ℕ)
  (h_men_styles : men_styles = 5) (h_men_sizes : men_sizes = 3) (h_men_colors : men_colors = 4)
  (h_women_styles : women_styles = 4) (h_women_sizes : women_sizes = 4) (h_women_colors : women_colors = 5) :
  men_styles * men_sizes * men_colors + women_styles * women_sizes * women_colors = 140 :=
by
  sorry

end NUMINAMATH_GPT_unique_combinations_bathing_suits_l1063_106383


namespace NUMINAMATH_GPT_has_exactly_one_zero_point_l1063_106347

noncomputable def f (x a b : ℝ) : ℝ := (x - 1) * Real.exp x - a * x^2 + b

theorem has_exactly_one_zero_point
  (a b : ℝ) 
  (h1 : (1/2 < a ∧ a ≤ Real.exp 2 / 2 ∧ b > 2 * a) ∨ (0 < a ∧ a < 1/2 ∧ b ≤ 2 * a)) :
  ∃! x : ℝ, f x a b = 0 := 
sorry

end NUMINAMATH_GPT_has_exactly_one_zero_point_l1063_106347


namespace NUMINAMATH_GPT_volume_of_sphere_l1063_106313

noncomputable def cuboid_volume (a b c : ℝ) := a * b * c

noncomputable def sphere_volume (r : ℝ) := (4/3) * Real.pi * r^3

theorem volume_of_sphere
  (a b c : ℝ) 
  (sphere_radius : ℝ)
  (h1 : a = 1)
  (h2 : b = Real.sqrt 3)
  (h3 : c = 2)
  (h4 : sphere_radius = Real.sqrt (a^2 + b^2 + c^2) / 2)
  : sphere_volume sphere_radius = (8 * Real.sqrt 2 / 3) * Real.pi := 
by
  sorry

end NUMINAMATH_GPT_volume_of_sphere_l1063_106313


namespace NUMINAMATH_GPT_number_of_subsets_of_three_element_set_l1063_106356

theorem number_of_subsets_of_three_element_set :
  ∃ (S : Finset ℕ), S.card = 3 ∧ S.powerset.card = 8 :=
sorry

end NUMINAMATH_GPT_number_of_subsets_of_three_element_set_l1063_106356


namespace NUMINAMATH_GPT_golden_section_AP_length_l1063_106338

noncomputable def golden_ratio : ℝ := (1 + Real.sqrt 5) / 2

noncomputable def golden_ratio_recip : ℝ := (Real.sqrt 5 - 1) / 2

theorem golden_section_AP_length (AB : ℝ) (P : ℝ) 
  (h1 : AB = 2) (h2 : P = golden_ratio_recip * AB) : 
  P = Real.sqrt 5 - 1 :=
by
  sorry

end NUMINAMATH_GPT_golden_section_AP_length_l1063_106338


namespace NUMINAMATH_GPT_shaded_area_of_octagon_l1063_106377

def side_length := 12
def octagon_area := 288

theorem shaded_area_of_octagon (s : ℕ) (h0 : s = side_length):
  (2 * s * s - 2 * s * s / 2) * 2 / 2 = octagon_area :=
by
  skip
  sorry

end NUMINAMATH_GPT_shaded_area_of_octagon_l1063_106377


namespace NUMINAMATH_GPT_lily_remaining_money_l1063_106305

def initial_amount := 55
def spent_on_shirt := 7
def spent_at_second_shop := 3 * spent_on_shirt
def total_spent := spent_on_shirt + spent_at_second_shop
def remaining_amount := initial_amount - total_spent

theorem lily_remaining_money : remaining_amount = 27 :=
by
  sorry

end NUMINAMATH_GPT_lily_remaining_money_l1063_106305


namespace NUMINAMATH_GPT_shortest_chord_intercept_l1063_106302

theorem shortest_chord_intercept (m : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 = 3 → x + m * y - m - 1 = 0 → m = 1) :=
sorry

end NUMINAMATH_GPT_shortest_chord_intercept_l1063_106302


namespace NUMINAMATH_GPT_min_value_of_expression_l1063_106320

theorem min_value_of_expression (a b : ℝ) (h_pos_b : 0 < b) (h_eq : 2 * a + b = 1) : 
  42 + b^2 + 1 / (a * b) ≥ 17 / 2 := 
sorry

end NUMINAMATH_GPT_min_value_of_expression_l1063_106320


namespace NUMINAMATH_GPT_number_reduced_by_10_eq_0_09_three_point_two_four_increased_to_three_two_four_zero_l1063_106372

theorem number_reduced_by_10_eq_0_09 : ∃ (x : ℝ), x / 10 = 0.09 ∧ x = 0.9 :=
sorry

theorem three_point_two_four_increased_to_three_two_four_zero : ∃ (y : ℝ), 3.24 * y = 3240 ∧ y = 1000 :=
sorry

end NUMINAMATH_GPT_number_reduced_by_10_eq_0_09_three_point_two_four_increased_to_three_two_four_zero_l1063_106372


namespace NUMINAMATH_GPT_koala_fiber_intake_l1063_106331

theorem koala_fiber_intake (x : ℝ) (h1 : 0.3 * x = 12) : x = 40 := 
by 
  sorry

end NUMINAMATH_GPT_koala_fiber_intake_l1063_106331


namespace NUMINAMATH_GPT_solve_for_x_l1063_106307

theorem solve_for_x (x : ℝ) (h : (2 + x) / (4 + x) = (3 + x) / (7 + x)) : x = -1 :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_for_x_l1063_106307


namespace NUMINAMATH_GPT_tangent_line_to_curve_perpendicular_l1063_106368

noncomputable def perpendicular_tangent_line (x y : ℝ) : Prop :=
  y = x^4 ∧ (4*x - y - 3 = 0)

theorem tangent_line_to_curve_perpendicular {x y : ℝ} (h : y = x^4 ∧ (4*x - y - 3 = 0)) :
  ∃ (x y : ℝ), (x+4*y-8=0) ∧ (4*x - y - 3 = 0) :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_to_curve_perpendicular_l1063_106368


namespace NUMINAMATH_GPT_sarah_problem_sum_l1063_106330

theorem sarah_problem_sum (x y : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 100 ≤ y ∧ y < 1000) (h : 1000 * x + y = 9 * x * y) :
  x + y = 126 :=
sorry

end NUMINAMATH_GPT_sarah_problem_sum_l1063_106330


namespace NUMINAMATH_GPT_acute_triangle_conditions_l1063_106329

-- Definitions exclusively from the conditions provided.
def condition_A (AB AC : ℝ) : Prop :=
  AB * AC > 0

def condition_B (sinA sinB sinC : ℝ) : Prop :=
  sinA / sinB = 4 / 5 ∧ sinA / sinC = 4 / 6 ∧ sinB / sinC = 5 / 6

def condition_C (cosA cosB cosC : ℝ) : Prop :=
  cosA * cosB * cosC > 0

def condition_D (tanA tanB : ℝ) : Prop :=
  tanA * tanB = 2

-- Prove which conditions guarantee that triangle ABC is acute.
theorem acute_triangle_conditions (AB AC sinA sinB sinC cosA cosB cosC tanA tanB : ℝ) :
  (condition_B sinA sinB sinC ∨ condition_C cosA cosB cosC ∨ condition_D tanA tanB) →
  (∀ (A B C : ℝ), A < π / 2 ∧ B < π / 2 ∧ C < π / 2) :=
sorry

end NUMINAMATH_GPT_acute_triangle_conditions_l1063_106329


namespace NUMINAMATH_GPT_smallest_positive_integer_modulo_l1063_106323

theorem smallest_positive_integer_modulo {n : ℕ} (h : 19 * n ≡ 546 [MOD 13]) : n = 11 := by
  sorry

end NUMINAMATH_GPT_smallest_positive_integer_modulo_l1063_106323


namespace NUMINAMATH_GPT_unattainable_value_l1063_106359

theorem unattainable_value : ∀ x : ℝ, x ≠ -4/3 → (y = (2 - x) / (3 * x + 4) → y ≠ -1/3) :=
by
  intro x hx h
  rw [eq_comm] at h
  sorry

end NUMINAMATH_GPT_unattainable_value_l1063_106359


namespace NUMINAMATH_GPT_find_a1_and_d_l1063_106366

-- Defining the arithmetic sequence and its properties
def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Given conditions
def conditions (a : ℕ → ℤ) (a_1 : ℤ) (d : ℤ) : Prop :=
  (a 4 + a 5 + a 6 + a 7 = 56) ∧ (a 4 * a 7 = 187) ∧ (a 1 = a_1) ∧ is_arithmetic_sequence a d

-- Proving the solution
theorem find_a1_and_d :
  ∃ (a : ℕ → ℤ) (a_1 d : ℤ),
    conditions a a_1 d ∧ ((a_1 = 5 ∧ d = 2) ∨ (a_1 = 23 ∧ d = -2)) :=
by
  sorry

end NUMINAMATH_GPT_find_a1_and_d_l1063_106366


namespace NUMINAMATH_GPT_length_percentage_increase_l1063_106390

/--
Given that the area of a rectangle is 460 square meters and the breadth is 20 meters,
prove that the percentage increase in length compared to the breadth is 15%.
-/
theorem length_percentage_increase (A : ℝ) (b : ℝ) (l : ℝ) (hA : A = 460) (hb : b = 20) (hl : l = A / b) :
  ((l - b) / b) * 100 = 15 :=
by
  sorry

end NUMINAMATH_GPT_length_percentage_increase_l1063_106390


namespace NUMINAMATH_GPT_bicycle_speed_l1063_106360

theorem bicycle_speed (x : ℝ) :
  (10 / x = 10 / (2 * x) + 1 / 3) → x = 15 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_bicycle_speed_l1063_106360


namespace NUMINAMATH_GPT_f_monotonic_f_odd_find_a_k_range_l1063_106399
open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2^x / (2^x + 1) + a

-- (1) Prove the monotonicity of the function f
theorem f_monotonic (a : ℝ) : ∀ {x y : ℝ}, x < y → f a x < f a y := sorry

-- (2) If f is an odd function, find the value of the real number a
theorem f_odd_find_a : ∀ a : ℝ, (∀ x : ℝ, f a (-x) = -f a x) → a = -1/2 := sorry

-- (3) Under the condition in (2), if the inequality holds for all x ∈ ℝ, find the range of values for k
theorem k_range (k : ℝ) :
  (∀ x : ℝ, f (-1/2) (x^2 - 2*x) + f (-1/2) (2*x^2 - k) > 0) → k < -1/3 := sorry

end NUMINAMATH_GPT_f_monotonic_f_odd_find_a_k_range_l1063_106399


namespace NUMINAMATH_GPT_correct_statement_l1063_106353

variable {a b : Type} -- Let a and b be types representing lines
variable {α β : Type} -- Let α and β be types representing planes

-- Define parallel relations for lines and planes
def parallel (L P : Type) : Prop := sorry

-- Define the subset relation for lines in planes
def subset (L P : Type) : Prop := sorry

-- Now state the theorem corresponding to the correct answer
theorem correct_statement (h1 : parallel α β) (h2 : subset a α) : parallel a β :=
sorry

end NUMINAMATH_GPT_correct_statement_l1063_106353


namespace NUMINAMATH_GPT_boat_speed_in_still_water_l1063_106351

theorem boat_speed_in_still_water : 
  ∀ (V_b V_s : ℝ), 
  V_b + V_s = 15 → 
  V_b - V_s = 5 → 
  V_b = 10 :=
by
  intros V_b V_s h1 h2
  have h3 : 2 * V_b = 20 := by linarith
  linarith

end NUMINAMATH_GPT_boat_speed_in_still_water_l1063_106351


namespace NUMINAMATH_GPT_tony_combined_lift_weight_l1063_106300

noncomputable def tony_exercises :=
  let curl_weight := 90 -- pounds.
  let military_press_weight := 2 * curl_weight -- pounds.
  let squat_weight := 5 * military_press_weight -- pounds.
  let bench_press_weight := 1.5 * military_press_weight -- pounds.
  squat_weight + bench_press_weight

theorem tony_combined_lift_weight :
  tony_exercises = 1170 := by
  -- Here we will include the necessary proof steps
  sorry

end NUMINAMATH_GPT_tony_combined_lift_weight_l1063_106300


namespace NUMINAMATH_GPT_time_to_cross_bridge_l1063_106335

noncomputable def train_length := 300  -- in meters
noncomputable def train_speed_kmph := 72  -- in km/h
noncomputable def bridge_length := 1500  -- in meters

-- Define the conversion from km/h to m/s
noncomputable def train_speed_mps := (train_speed_kmph * 1000) / 3600  -- in m/s

-- Define the total distance to be traveled
noncomputable def total_distance := train_length + bridge_length  -- in meters

-- Define the time to cross the bridge
noncomputable def time_to_cross := total_distance / train_speed_mps  -- in seconds

theorem time_to_cross_bridge : time_to_cross = 90 := by
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_time_to_cross_bridge_l1063_106335


namespace NUMINAMATH_GPT_number_multiplied_value_l1063_106322

theorem number_multiplied_value (x : ℝ) :
  (4 / 6) * x = 8 → x = 12 :=
by
  sorry

end NUMINAMATH_GPT_number_multiplied_value_l1063_106322


namespace NUMINAMATH_GPT_achieve_100_with_fewer_threes_example1_achieve_100_with_fewer_threes_example2_l1063_106332

theorem achieve_100_with_fewer_threes_example1 :
  ((333 / 3) - (33 / 3) = 100) :=
by
  sorry

theorem achieve_100_with_fewer_threes_example2 :
  ((33 * 3) + (3 / 3) = 100) :=
by
  sorry

end NUMINAMATH_GPT_achieve_100_with_fewer_threes_example1_achieve_100_with_fewer_threes_example2_l1063_106332


namespace NUMINAMATH_GPT_age_of_teacher_l1063_106310

variables (age_students : ℕ) (age_all : ℕ) (teacher_age : ℕ)

def avg_age_students := 15
def num_students := 10
def num_people := 11
def avg_age_people := 16

theorem age_of_teacher
  (h1 : age_students = num_students * avg_age_students)
  (h2 : age_all = num_people * avg_age_people)
  (h3 : age_all = age_students + teacher_age) : teacher_age = 26 :=
by
  sorry

end NUMINAMATH_GPT_age_of_teacher_l1063_106310


namespace NUMINAMATH_GPT_evaluate_fraction_l1063_106397

-- Let's restate the problem in Lean
theorem evaluate_fraction :
  (∃ q, (2024 / 2023 - 2023 / 2024) = 4047 / q) :=
by
  -- Substitute a = 2023
  let a := 2023
  -- Provide the value we expect for q to hold in the reduced fraction.
  use (a * (a + 1)) -- The expected denominator
  -- The proof for the theorem is omitted here
  sorry

end NUMINAMATH_GPT_evaluate_fraction_l1063_106397


namespace NUMINAMATH_GPT_Barkley_bones_l1063_106387

def bones_per_month : ℕ := 10
def months : ℕ := 5
def bones_received : ℕ := bones_per_month * months
def bones_buried : ℕ := 42
def bones_available : ℕ := 8

theorem Barkley_bones :
  bones_received - bones_buried = bones_available := by sorry

end NUMINAMATH_GPT_Barkley_bones_l1063_106387


namespace NUMINAMATH_GPT_newspapers_ratio_l1063_106339

theorem newspapers_ratio :
  (∀ (j m : ℕ), j = 234 → m = 4 * j + 936 → (m / 4) / j = 2) :=
by
  sorry

end NUMINAMATH_GPT_newspapers_ratio_l1063_106339


namespace NUMINAMATH_GPT_ratio_of_small_square_to_shaded_area_l1063_106321

theorem ratio_of_small_square_to_shaded_area :
  let small_square_area := 2 * 2
  let large_square_area := 5 * 5
  let shaded_area := (large_square_area / 2) - (small_square_area / 2)
  (small_square_area : ℚ) / shaded_area = 8 / 21 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_small_square_to_shaded_area_l1063_106321


namespace NUMINAMATH_GPT_find_multiple_sales_l1063_106373

theorem find_multiple_sales 
  (A : ℝ) 
  (M : ℝ)
  (h : M * A = 0.35294117647058826 * (11 * A + M * A)) 
  : M = 6 :=
sorry

end NUMINAMATH_GPT_find_multiple_sales_l1063_106373


namespace NUMINAMATH_GPT_mixed_operations_with_rationals_l1063_106362

theorem mixed_operations_with_rationals :
  let a := 1 / 4
  let b := 1 / 2
  let c := 2 / 3
  (a - b + c) * (-12) = -8 :=
by
  sorry

end NUMINAMATH_GPT_mixed_operations_with_rationals_l1063_106362


namespace NUMINAMATH_GPT_number_of_women_more_than_men_l1063_106308

variables (M W : ℕ)

def ratio_condition : Prop := M * 3 = 2 * W
def total_condition : Prop := M + W = 20
def correct_answer : Prop := W - M = 4

theorem number_of_women_more_than_men 
  (h1 : ratio_condition M W) 
  (h2 : total_condition M W) : 
  correct_answer M W := 
by 
  sorry

end NUMINAMATH_GPT_number_of_women_more_than_men_l1063_106308


namespace NUMINAMATH_GPT_right_triangle_has_one_right_angle_l1063_106364

def is_right_angle (θ : ℝ) : Prop := θ = 90

def sum_of_triangle_angles (α β γ : ℝ) : Prop := α + β + γ = 180

def right_triangle (α β γ : ℝ) : Prop := is_right_angle α ∨ is_right_angle β ∨ is_right_angle γ

theorem right_triangle_has_one_right_angle (α β γ : ℝ) :
  right_triangle α β γ → sum_of_triangle_angles α β γ →
  (is_right_angle α ∧ ¬is_right_angle β ∧ ¬is_right_angle γ) ∨
  (¬is_right_angle α ∧ is_right_angle β ∧ ¬is_right_angle γ) ∨
  (¬is_right_angle α ∧ ¬is_right_angle β ∧ is_right_angle γ) :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_has_one_right_angle_l1063_106364


namespace NUMINAMATH_GPT_shell_count_l1063_106381

theorem shell_count (initial_shells : ℕ) (ed_limpet : ℕ) (ed_oyster : ℕ) (ed_conch : ℕ) (jacob_extra : ℕ)
  (h1 : initial_shells = 2)
  (h2 : ed_limpet = 7) 
  (h3 : ed_oyster = 2) 
  (h4 : ed_conch = 4) 
  (h5 : jacob_extra = 2) : 
  (initial_shells + ed_limpet + ed_oyster + ed_conch + (ed_limpet + ed_oyster + ed_conch + jacob_extra)) = 30 := 
by 
  sorry

end NUMINAMATH_GPT_shell_count_l1063_106381


namespace NUMINAMATH_GPT_tom_average_speed_l1063_106301

theorem tom_average_speed 
  (d1 d2 : ℝ) (s1 s2 t1 t2 : ℝ)
  (h_d1 : d1 = 30) 
  (h_d2 : d2 = 50) 
  (h_s1 : s1 = 30) 
  (h_s2 : s2 = 50) 
  (h_t1 : t1 = d1 / s1) 
  (h_t2 : t2 = d2 / s2)
  (h_total_distance : d1 + d2 = 80) 
  (h_total_time : t1 + t2 = 2) :
  (d1 + d2) / (t1 + t2) = 40 := 
by {
  sorry
}

end NUMINAMATH_GPT_tom_average_speed_l1063_106301


namespace NUMINAMATH_GPT_janet_total_earnings_l1063_106346

-- Definitions based on conditions from step a)
def hourly_wage := 70
def hours_worked := 20
def rate_per_pound := 20
def weight_sculpture1 := 5
def weight_sculpture2 := 7

-- Statement for the proof problem
theorem janet_total_earnings : 
  let earnings_from_extermination := hourly_wage * hours_worked
  let earnings_from_sculpture1 := rate_per_pound * weight_sculpture1
  let earnings_from_sculpture2 := rate_per_pound * weight_sculpture2
  earnings_from_extermination + earnings_from_sculpture1 + earnings_from_sculpture2 = 1640 := 
by
  sorry

end NUMINAMATH_GPT_janet_total_earnings_l1063_106346


namespace NUMINAMATH_GPT_net_effect_on_sale_l1063_106394

variable (P Q : ℝ) -- Price and Quantity

theorem net_effect_on_sale :
  let reduced_price := 0.40 * P
  let increased_quantity := 2.50 * Q
  let price_after_tax := 0.44 * P
  let price_after_discount := 0.418 * P
  let final_revenue := price_after_discount * increased_quantity 
  let original_revenue := P * Q
  final_revenue / original_revenue = 1.045 :=
by
  sorry

end NUMINAMATH_GPT_net_effect_on_sale_l1063_106394


namespace NUMINAMATH_GPT_gerald_total_pieces_eq_672_l1063_106350

def pieces_per_table : Nat := 12
def pieces_per_chair : Nat := 8
def num_tables : Nat := 24
def num_chairs : Nat := 48

def total_pieces : Nat := pieces_per_table * num_tables + pieces_per_chair * num_chairs

theorem gerald_total_pieces_eq_672 : total_pieces = 672 :=
by
  sorry

end NUMINAMATH_GPT_gerald_total_pieces_eq_672_l1063_106350


namespace NUMINAMATH_GPT_percent_motorists_no_ticket_l1063_106355

theorem percent_motorists_no_ticket (M : ℝ) :
  (0.14285714285714285 * M - 0.10 * M) / (0.14285714285714285 * M) * 100 = 30 :=
by
  sorry

end NUMINAMATH_GPT_percent_motorists_no_ticket_l1063_106355


namespace NUMINAMATH_GPT_part_I_part_II_l1063_106358

noncomputable def f (x : ℝ) : ℝ := 2 * |x + 1| + |2 * x - 1|

theorem part_I (x : ℝ) : 
  (f x > f 1) ↔ (x < -3/2 ∨ x > 1) :=
by
  -- We leave the proof as sorry to indicate it needs to be filled in
  sorry

theorem part_II (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  (∀ x : ℝ, f x ≥ 1/m + 1/n) → m + n ≥ 4/3 :=
by
  -- We leave the proof as sorry to indicate it needs to be filled in
  sorry

end NUMINAMATH_GPT_part_I_part_II_l1063_106358


namespace NUMINAMATH_GPT_find_a_plus_b_l1063_106376

noncomputable def f (a b : ℝ) (x : ℝ) := a * x + b

noncomputable def h (x : ℝ) := 3 * x + 2

theorem find_a_plus_b (a b : ℝ) (x : ℝ) (h_condition : ∀ x, h (f a b x) = 4 * x - 1) :
  a + b = 1 / 3 := 
by
  sorry

end NUMINAMATH_GPT_find_a_plus_b_l1063_106376


namespace NUMINAMATH_GPT_sqrt_factorial_squared_l1063_106314

theorem sqrt_factorial_squared :
  (Real.sqrt ((Nat.factorial 5) * (Nat.factorial 4))) ^ 2 = 2880 :=
by sorry

end NUMINAMATH_GPT_sqrt_factorial_squared_l1063_106314


namespace NUMINAMATH_GPT_not_necessarily_prime_sum_l1063_106352

theorem not_necessarily_prime_sum (nat_ordered_sequence : ℕ → ℕ) :
  (∀ n1 n2 n3 : ℕ, n1 < n2 → n2 < n3 → nat_ordered_sequence n1 + nat_ordered_sequence n2 + nat_ordered_sequence n3 ≠ prime) :=
sorry

end NUMINAMATH_GPT_not_necessarily_prime_sum_l1063_106352


namespace NUMINAMATH_GPT_max_red_socks_l1063_106386

-- Define r (red socks), b (blue socks), t (total socks), with the given constraints
def socks_problem (r b t : ℕ) : Prop :=
  t = r + b ∧
  t ≤ 2023 ∧
  (2 * r * (r - 1) + 2 * b * (b - 1)) = 2 * 5 * t * (t - 1)

-- State the theorem that the maximum number of red socks is 990
theorem max_red_socks : ∃ r b t, socks_problem r b t ∧ r = 990 :=
sorry

end NUMINAMATH_GPT_max_red_socks_l1063_106386


namespace NUMINAMATH_GPT_heaviest_weight_is_aq3_l1063_106327

variable (a q : ℝ) (h : 0 < a) (hq : 1 < q)

theorem heaviest_weight_is_aq3 :
  let w1 := a
  let w2 := a * q
  let w3 := a * q^2
  let w4 := a * q^3
  w4 > w3 ∧ w4 > w2 ∧ w4 > w1 ∧ w1 + w4 > w2 + w3 :=
by
  sorry

end NUMINAMATH_GPT_heaviest_weight_is_aq3_l1063_106327


namespace NUMINAMATH_GPT_set_A_roster_l1063_106357

def is_nat_not_greater_than_4 (x : ℕ) : Prop := x ≤ 4

def A : Set ℕ := {x | is_nat_not_greater_than_4 x}

theorem set_A_roster : A = {0, 1, 2, 3, 4} := by
  sorry

end NUMINAMATH_GPT_set_A_roster_l1063_106357


namespace NUMINAMATH_GPT_find_y_l1063_106334

def is_divisible_by (x y : ℕ) : Prop := x % y = 0

def ends_with_digit (x : ℕ) (d : ℕ) : Prop :=
  x % 10 = d

theorem find_y (y : ℕ) :
  (y > 0) ∧
  is_divisible_by y 4 ∧
  is_divisible_by y 5 ∧
  is_divisible_by y 7 ∧
  is_divisible_by y 13 ∧
  ¬ is_divisible_by y 8 ∧
  ¬ is_divisible_by y 15 ∧
  ¬ is_divisible_by y 50 ∧
  ends_with_digit y 0
  → y = 1820 :=
sorry

end NUMINAMATH_GPT_find_y_l1063_106334


namespace NUMINAMATH_GPT_solve_diamond_l1063_106378

theorem solve_diamond (d : ℕ) (hd : d < 10) (h : d * 9 + 6 = d * 10 + 3) : d = 3 :=
sorry

end NUMINAMATH_GPT_solve_diamond_l1063_106378


namespace NUMINAMATH_GPT_slope_magnitude_l1063_106367

-- Definitions based on given conditions
def parabola : Set (ℝ × ℝ) := { p | ∃ x y, p = (x, y) ∧ y^2 = 4 * x }
def line (k m : ℝ) : Set (ℝ × ℝ) := { p | ∃ x y, p = (x, y) ∧ y = k * x + m }
def focus : ℝ × ℝ := (1, 0)
def intersects (l p : Set (ℝ × ℝ)) : Prop := ∃ x1 y1 x2 y2, (x1, y1) ∈ l ∧ (x1, y1) ∈ p ∧ (x2, y2) ∈ l ∧ (x2, y2) ∈ p ∧ (x1, y1) ≠ (x2, y2)

theorem slope_magnitude (k m : ℝ) (h_k_nonzero : k ≠ 0) 
  (h_intersects : intersects (line k m) parabola) 
  (h_AF_2FB : ∀ x1 y1 x2 y2, (x1, y1) ∈ line k m → (x1, y1) ∈ parabola → 
                          (x2, y2) ∈ line k m → (x2, y2) ∈ parabola → 
                          (1 - x1 = 2 * (x2 - 1)) ∧ (-y1 = 2 * y2)) :
  |k| = 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_slope_magnitude_l1063_106367


namespace NUMINAMATH_GPT_ordered_pair_solution_l1063_106319

theorem ordered_pair_solution :
  ∃ (x y : ℤ), x + y = (6 - x) + (6 - y) ∧ x - y = (x - 2) + (y - 2) ∧ (x, y) = (2, 4) :=
by
  sorry

end NUMINAMATH_GPT_ordered_pair_solution_l1063_106319


namespace NUMINAMATH_GPT_neg_fraction_comparison_l1063_106342

theorem neg_fraction_comparison : - (4 / 5 : ℝ) > - (5 / 6 : ℝ) :=
by {
  -- sorry to skip the proof
  sorry
}

end NUMINAMATH_GPT_neg_fraction_comparison_l1063_106342


namespace NUMINAMATH_GPT_perfect_square_expression_l1063_106391

theorem perfect_square_expression (n : ℕ) : ∃ t : ℕ, n^2 - 4 * n + 11 = t^2 ↔ n = 5 :=
by
  sorry

end NUMINAMATH_GPT_perfect_square_expression_l1063_106391


namespace NUMINAMATH_GPT_contradiction_prop_l1063_106396

theorem contradiction_prop (p : Prop) : 
  (∃ x : ℝ, x < -1 ∧ x^2 - x + 1 < 0) → (∀ x : ℝ, x < -1 → x^2 - x + 1 ≥ 0) :=
sorry

end NUMINAMATH_GPT_contradiction_prop_l1063_106396


namespace NUMINAMATH_GPT_distance_covered_l1063_106337

/-- 
Given the following conditions:
1. The speed of Abhay (A) is 5 km/h.
2. The time taken by Abhay to cover a distance is 2 hours more than the time taken by Sameer.
3. If Abhay doubles his speed, then he would take 1 hour less than Sameer.
Prove that the distance (D) they are covering is 30 kilometers.
-/
theorem distance_covered (D S : ℝ) (A : ℝ) (hA : A = 5) 
  (h1 : D / A = D / S + 2) 
  (h2 : D / (2 * A) = D / S - 1) : 
  D = 30 := by
    sorry

end NUMINAMATH_GPT_distance_covered_l1063_106337


namespace NUMINAMATH_GPT_problem1_problem2_l1063_106363

-- Proof Problem for (1)
theorem problem1 : -15 - (-5) + 6 = -4 := sorry

-- Proof Problem for (2)
theorem problem2 : 81 / (-9 / 5) * (5 / 9) = -25 := sorry

end NUMINAMATH_GPT_problem1_problem2_l1063_106363


namespace NUMINAMATH_GPT_range_of_a_l1063_106384

noncomputable def geometric_seq (r : ℝ) (n : ℕ) (a₁ : ℝ) : ℝ := a₁ * r ^ (n - 1)

theorem range_of_a (a : ℝ) :
  (∃ a_seq b_seq : ℕ → ℝ, a_seq 1 = a ∧ (∀ n, b_seq n = (a_seq n - 2) / (a_seq n - 1)) ∧ (∀ n, a_seq n > a_seq (n+1)) ∧ (∀ n, b_seq (n + 1) = geometric_seq (2/3) (n + 1) (b_seq 1))) → 2 < a :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1063_106384


namespace NUMINAMATH_GPT_inverse_function_shift_l1063_106388

-- Conditions
variable {f : ℝ → ℝ} {f_inv : ℝ → ℝ}
variable (hf : ∀ x : ℝ, f_inv (f x) = x ∧ f (f_inv x) = x)
variable (point_B : f 3 = -1)

-- Proof statement
theorem inverse_function_shift :
  f_inv (-3 + 2) = 3 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_inverse_function_shift_l1063_106388


namespace NUMINAMATH_GPT_circle_value_in_grid_l1063_106349

theorem circle_value_in_grid :
  ∃ (min_circle_val : ℕ), min_circle_val = 21 ∧ (∀ (max_circle_val : ℕ), ∃ (L : ℕ), L > max_circle_val) :=
by
  sorry

end NUMINAMATH_GPT_circle_value_in_grid_l1063_106349


namespace NUMINAMATH_GPT_white_marbles_in_C_equals_15_l1063_106303

variables (A_red A_yellow B_green B_yellow C_yellow : ℕ) (w : ℕ)

-- Conditions from the problem
def conditions : Prop :=
  A_red = 4 ∧ A_yellow = 2 ∧
  B_green = 6 ∧ B_yellow = 1 ∧
  C_yellow = 9 ∧
  (A_red - A_yellow = 2) ∧
  (B_green - B_yellow = 5) ∧
  (w - C_yellow = 6)

-- Proving w = 15 given the conditions
theorem white_marbles_in_C_equals_15 (h : conditions A_red A_yellow B_green B_yellow C_yellow w) : w = 15 :=
  sorry

end NUMINAMATH_GPT_white_marbles_in_C_equals_15_l1063_106303


namespace NUMINAMATH_GPT_janet_savings_l1063_106328

def wall1_area := 5 * 8 -- wall 1 area
def wall2_area := 7 * 8 -- wall 2 area
def wall3_area := 6 * 9 -- wall 3 area
def total_area := wall1_area + wall2_area + wall3_area
def tiles_per_square_foot := 4
def total_tiles := total_area * tiles_per_square_foot

def turquoise_tile_cost := 13
def turquoise_labor_cost := 6
def total_cost_turquoise := (total_tiles * turquoise_tile_cost) + (total_area * turquoise_labor_cost)

def purple_tile_cost := 11
def purple_labor_cost := 8
def total_cost_purple := (total_tiles * purple_tile_cost) + (total_area * purple_labor_cost)

def orange_tile_cost := 15
def orange_labor_cost := 5
def total_cost_orange := (total_tiles * orange_tile_cost) + (total_area * orange_labor_cost)

def least_expensive_option := total_cost_purple
def most_expensive_option := total_cost_orange

def savings := most_expensive_option - least_expensive_option

theorem janet_savings : savings = 1950 := by
  sorry

end NUMINAMATH_GPT_janet_savings_l1063_106328


namespace NUMINAMATH_GPT_smallest_successive_number_l1063_106309

theorem smallest_successive_number :
  ∃ n : ℕ, n * (n + 1) * (n + 2) = 1059460 ∧ ∀ m : ℕ, m * (m + 1) * (m + 2) = 1059460 → n ≤ m :=
sorry

end NUMINAMATH_GPT_smallest_successive_number_l1063_106309


namespace NUMINAMATH_GPT_find_z_l1063_106389

open Complex

theorem find_z (z : ℂ) (h1 : (z + 2 * I).im = 0) (h2 : ((z / (2 - I)).im = 0)) : z = 4 - 2 * I :=
by
  sorry

end NUMINAMATH_GPT_find_z_l1063_106389


namespace NUMINAMATH_GPT_range_of_m_l1063_106393

theorem range_of_m {m : ℝ} (h1 : ∀ (x : ℝ), 1 < x → (2 / (x - m) < 2 / (x - m + 1)))
                   (h2 : ∃ (a : ℝ), -1 ≤ a ∧ a ≤ 1 ∧ ∀ (x1 x2 : ℝ), x1 + x2 = a ∧ x1 * x2 = -2 → m^2 + 5 * m - 3 ≥ abs (x1 - x2))
                   (h3 : ¬(∀ (x : ℝ), 1 < x → (2 / (x - m) < 2 / (x - m + 1))) ∧
                           (∃ (a : ℝ), -1 ≤ a ∧ a ≤ 1 ∧ ∀ (x1 x2 : ℝ), x1 + x2 = a ∧ x1 * x2 = -2 → m^2 + 5 * m - 3 ≥ abs (x1 - x2))) :
  m > 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1063_106393


namespace NUMINAMATH_GPT_max_sections_with_five_lines_l1063_106371

def sections (n : ℕ) : ℕ :=
  if n = 0 then 1 else
  n * (n + 1) / 2 + 1

theorem max_sections_with_five_lines : sections 5 = 16 := by
  sorry

end NUMINAMATH_GPT_max_sections_with_five_lines_l1063_106371


namespace NUMINAMATH_GPT_kids_joined_in_l1063_106326

-- Define the given conditions
def original : ℕ := 14
def current : ℕ := 36

-- State the goal
theorem kids_joined_in : (current - original = 22) :=
by
  sorry

end NUMINAMATH_GPT_kids_joined_in_l1063_106326


namespace NUMINAMATH_GPT_population_increase_rate_is_20_percent_l1063_106375

noncomputable def population_increase_rate 
  (initial_population final_population : ℕ) : ℕ :=
  ((final_population - initial_population) * 100) / initial_population

theorem population_increase_rate_is_20_percent :
  population_increase_rate 2000 2400 = 20 :=
by
  unfold population_increase_rate
  sorry

end NUMINAMATH_GPT_population_increase_rate_is_20_percent_l1063_106375


namespace NUMINAMATH_GPT_total_payment_correct_l1063_106374

noncomputable def calculate_total_payment : ℝ :=
  let original_price_vase := 200
  let discount_vase := 0.35 * original_price_vase
  let sale_price_vase := original_price_vase - discount_vase
  let tax_vase := 0.10 * sale_price_vase

  let original_price_teacups := 300
  let discount_teacups := 0.20 * original_price_teacups
  let sale_price_teacups := original_price_teacups - discount_teacups
  let tax_teacups := 0.08 * sale_price_teacups

  let original_price_plate := 500
  let sale_price_plate := original_price_plate
  let tax_plate := 0.10 * sale_price_plate

  (sale_price_vase + tax_vase) + (sale_price_teacups + tax_teacups) + (sale_price_plate + tax_plate)

theorem total_payment_correct : calculate_total_payment = 952.20 :=
by sorry

end NUMINAMATH_GPT_total_payment_correct_l1063_106374


namespace NUMINAMATH_GPT_turtles_in_lake_l1063_106317

-- Definitions based on conditions
def total_turtles : ℝ := 100
def percent_female : ℝ := 0.6
def percent_male : ℝ := 0.4
def percent_striped_male : ℝ := 0.25
def striped_turtle_babies : ℝ := 4
def percent_babies : ℝ := 0.4

-- Statement to prove
theorem turtles_in_lake : 
  (total_turtles * percent_male * percent_striped_male / percent_babies = striped_turtle_babies) →
  total_turtles = 100 :=
by
  sorry

end NUMINAMATH_GPT_turtles_in_lake_l1063_106317


namespace NUMINAMATH_GPT_expand_polynomial_identity_l1063_106336

variable {x : ℝ}

theorem expand_polynomial_identity : (7 * x + 5) * (5 * x ^ 2 - 2 * x + 4) = 35 * x ^ 3 + 11 * x ^ 2 + 18 * x + 20 := by
    sorry

end NUMINAMATH_GPT_expand_polynomial_identity_l1063_106336


namespace NUMINAMATH_GPT_pascal_triangle_contains_53_once_l1063_106344

theorem pascal_triangle_contains_53_once
  (h_prime : Nat.Prime 53) :
  ∃! n, ∃ k, n ≥ k ∧ n > 0 ∧ k > 0 ∧ Nat.choose n k = 53 := by
  sorry

end NUMINAMATH_GPT_pascal_triangle_contains_53_once_l1063_106344


namespace NUMINAMATH_GPT_isabella_hair_growth_l1063_106340

theorem isabella_hair_growth :
  ∀ (initial final : ℤ), initial = 18 → final = 24 → final - initial = 6 :=
by
  intros initial final h_initial h_final
  rw [h_initial, h_final]
  exact rfl
-- sorry

end NUMINAMATH_GPT_isabella_hair_growth_l1063_106340


namespace NUMINAMATH_GPT_intersection_with_y_axis_l1063_106365

-- Define the given function
def f (x : ℝ) := x^2 + x - 2

-- Prove that the intersection point with the y-axis is (0, -2)
theorem intersection_with_y_axis : f 0 = -2 :=
by {
  sorry
}

end NUMINAMATH_GPT_intersection_with_y_axis_l1063_106365


namespace NUMINAMATH_GPT_max_capacity_tank_l1063_106306

-- Definitions of the conditions
def water_loss_1 := 32000 * 5
def water_loss_2 := 10000 * 10
def total_loss := water_loss_1 + water_loss_2
def water_added := 40000 * 3
def missing_water := 140000

-- Definition of the maximum capacity
def max_capacity := total_loss + water_added + missing_water

-- The theorem to prove
theorem max_capacity_tank : max_capacity = 520000 := by
  sorry

end NUMINAMATH_GPT_max_capacity_tank_l1063_106306


namespace NUMINAMATH_GPT_cos_60_eq_sqrt3_div_2_l1063_106324

theorem cos_60_eq_sqrt3_div_2 : Real.cos (60 * Real.pi / 180) = Real.sqrt 3 / 2 := sorry

end NUMINAMATH_GPT_cos_60_eq_sqrt3_div_2_l1063_106324


namespace NUMINAMATH_GPT_star_neg5_4_star_neg3_neg6_l1063_106392

-- Definition of the new operation
def star (a b : ℤ) : ℤ := 2 * a * b - b / 2

-- The first proof problem
theorem star_neg5_4 : star (-5) 4 = -42 := by sorry

-- The second proof problem
theorem star_neg3_neg6 : star (-3) (-6) = 39 := by sorry

end NUMINAMATH_GPT_star_neg5_4_star_neg3_neg6_l1063_106392


namespace NUMINAMATH_GPT_prove_ab_l1063_106304

theorem prove_ab 
  (a b : ℝ)
  (h1 : a + b = 4)
  (h2 : a^2 + b^2 = 6) : 
  a * b = 5 :=
by
  sorry

end NUMINAMATH_GPT_prove_ab_l1063_106304


namespace NUMINAMATH_GPT_max_sum_at_1008_l1063_106361

noncomputable def sum_sequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n * (a 1 + a n)) / 2

theorem max_sum_at_1008 (a : ℕ → ℝ) : 
  sum_sequence a 2015 > 0 → 
  sum_sequence a 2016 < 0 → 
  ∃ n, n = 1008 ∧ ∀ m, sum_sequence a m ≤ sum_sequence a 1008 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_max_sum_at_1008_l1063_106361


namespace NUMINAMATH_GPT_Sabrina_pencils_l1063_106318

variable (S : ℕ) (J : ℕ)

theorem Sabrina_pencils (h1 : S + J = 50) (h2 : J = 2 * S + 8) :
  S = 14 :=
by
  sorry

end NUMINAMATH_GPT_Sabrina_pencils_l1063_106318


namespace NUMINAMATH_GPT_number_of_young_teachers_selected_l1063_106316

theorem number_of_young_teachers_selected 
  (total_teachers elderly_teachers middle_aged_teachers young_teachers sample_size : ℕ)
  (h_total: total_teachers = 200)
  (h_elderly: elderly_teachers = 25)
  (h_middle_aged: middle_aged_teachers = 75)
  (h_young: young_teachers = 100)
  (h_sample_size: sample_size = 40)
  : young_teachers * sample_size / total_teachers = 20 := 
sorry

end NUMINAMATH_GPT_number_of_young_teachers_selected_l1063_106316


namespace NUMINAMATH_GPT_compute_fraction_power_l1063_106345

theorem compute_fraction_power :
  8 * (1 / 4) ^ 4 = 1 / 32 := 
by
  sorry

end NUMINAMATH_GPT_compute_fraction_power_l1063_106345


namespace NUMINAMATH_GPT_total_operation_time_correct_l1063_106311

def accessories_per_doll := 2 + 3 + 1 + 5
def number_of_dolls := 12000
def time_per_doll := 45
def time_per_accessory := 10
def total_accessories := number_of_dolls * accessories_per_doll
def time_for_dolls := number_of_dolls * time_per_doll
def time_for_accessories := total_accessories * time_per_accessory
def total_combined_time := time_for_dolls + time_for_accessories

theorem total_operation_time_correct :
  total_combined_time = 1860000 :=
by
  sorry

end NUMINAMATH_GPT_total_operation_time_correct_l1063_106311


namespace NUMINAMATH_GPT_b_share_of_earnings_l1063_106312

-- Definitions derived from conditions
def work_rate_a := 1 / 6
def work_rate_b := 1 / 8
def work_rate_c := 1 / 12
def total_earnings := 1170

-- Mathematically equivalent Lean statement
theorem b_share_of_earnings : 
  (work_rate_b / (work_rate_a + work_rate_b + work_rate_c)) * total_earnings = 390 := 
by
  sorry

end NUMINAMATH_GPT_b_share_of_earnings_l1063_106312


namespace NUMINAMATH_GPT_sum_of_odd_integers_l1063_106379

theorem sum_of_odd_integers (n : ℕ) (h1 : 4970 = n * (1 + n)) : (n ^ 2 = 4900) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_odd_integers_l1063_106379


namespace NUMINAMATH_GPT_exp_add_exp_nat_mul_l1063_106343

noncomputable def Exp (z : ℝ) : ℝ := Real.exp z

theorem exp_add (a b x : ℝ) :
  Exp ((a + b) * x) = Exp (a * x) * Exp (b * x) := sorry

theorem exp_nat_mul (x : ℝ) (k : ℕ) :
  Exp (k * x) = (Exp x) ^ k := sorry

end NUMINAMATH_GPT_exp_add_exp_nat_mul_l1063_106343


namespace NUMINAMATH_GPT_jeanne_additional_tickets_l1063_106380

-- Define the costs
def ferris_wheel_cost : ℕ := 5
def roller_coaster_cost : ℕ := 4
def bumper_cars_cost : ℕ := 4
def jeanne_tickets : ℕ := 5

-- Calculate the total cost
def total_cost : ℕ := ferris_wheel_cost + roller_coaster_cost + bumper_cars_cost

-- Define the proof problem
theorem jeanne_additional_tickets : total_cost - jeanne_tickets = 8 :=
by sorry

end NUMINAMATH_GPT_jeanne_additional_tickets_l1063_106380


namespace NUMINAMATH_GPT_problem_part1_problem_part2_l1063_106369

-- Statement for Part (1)
theorem problem_part1 (a b : ℝ) (h_sol : {x : ℝ | ax^2 - 3 * x + 2 > 0} = {x : ℝ | x < 1 ∨ x > 2}) :
  a = 1 ∧ b = 2 := sorry

-- Statement for Part (2)
theorem problem_part2 (x y k : ℝ) (hx_pos : x > 0) (hy_pos : y > 0) 
  (h_eq : (1 / x) + (2 / y) = 1) (h_ineq : 2 * x + y ≥ k^2 + k + 2) :
  -3 ≤ k ∧ k ≤ 2 := sorry

end NUMINAMATH_GPT_problem_part1_problem_part2_l1063_106369


namespace NUMINAMATH_GPT_jeans_price_increase_l1063_106370

theorem jeans_price_increase (manufacturing_cost customer_price : ℝ) 
  (h1 : customer_price = 1.40 * (1.40 * manufacturing_cost))
  : (customer_price - manufacturing_cost) / manufacturing_cost * 100 = 96 :=
by sorry

end NUMINAMATH_GPT_jeans_price_increase_l1063_106370


namespace NUMINAMATH_GPT_circle_radius_is_7_5_l1063_106341

noncomputable def radius_of_circle (side_length : ℝ) : ℝ := sorry

theorem circle_radius_is_7_5 :
  radius_of_circle 12 = 7.5 := sorry

end NUMINAMATH_GPT_circle_radius_is_7_5_l1063_106341


namespace NUMINAMATH_GPT_stampsLeftover_l1063_106395

-- Define the number of stamps each person has
def oliviaStamps : ℕ := 52
def parkerStamps : ℕ := 66
def quinnStamps : ℕ := 23

-- Define the album's capacity in stamps
def albumCapacity : ℕ := 15

-- Define the total number of leftovers
def totalLeftover : ℕ := (oliviaStamps + parkerStamps + quinnStamps) % albumCapacity

-- Define the theorem we want to prove
theorem stampsLeftover : totalLeftover = 6 := by
  sorry

end NUMINAMATH_GPT_stampsLeftover_l1063_106395


namespace NUMINAMATH_GPT_calc_theoretical_yield_l1063_106385
-- Importing all necessary libraries

-- Define the molar masses
def molar_mass_NaNO3 : ℝ := 85

-- Define the initial moles
def initial_moles_NH4NO3 : ℝ := 2
def initial_moles_NaOH : ℝ := 2

-- Define the final yield percentage
def yield_percentage : ℝ := 0.85

-- State the proof problem
theorem calc_theoretical_yield :
  let moles_NaNO3 := (2 : ℝ) * 2 * yield_percentage
  let grams_NaNO3 := moles_NaNO3 * molar_mass_NaNO3
  grams_NaNO3 = 289 :=
by 
  sorry

end NUMINAMATH_GPT_calc_theoretical_yield_l1063_106385


namespace NUMINAMATH_GPT_total_pieces_of_mail_l1063_106398

-- Definitions based on given conditions
def pieces_each_friend_delivers : ℕ := 41
def pieces_johann_delivers : ℕ := 98
def number_of_friends : ℕ := 2

-- Theorem statement to prove the total number of pieces of mail delivered
theorem total_pieces_of_mail :
  (number_of_friends * pieces_each_friend_delivers) + pieces_johann_delivers = 180 := 
by
  -- proof would go here
  sorry

end NUMINAMATH_GPT_total_pieces_of_mail_l1063_106398


namespace NUMINAMATH_GPT_smallest_six_consecutive_number_exists_max_value_N_perfect_square_l1063_106325

-- Definition of 'six-consecutive numbers'
def is_six_consecutive (a b c d : ℕ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧
  b ≠ d ∧ c ≠ d ∧ (a + b) * (c + d) = 60

-- Definition of the function F
def F (a b c d : ℕ) : ℤ :=
  let p := (10 * a + c) - (10 * b + d)
  let q := (10 * a + d) - (10 * b + c)
  q - p

-- Exists statement for the smallest six-consecutive number
theorem smallest_six_consecutive_number_exists :
  ∃ (a b c d : ℕ), is_six_consecutive a b c d ∧ (1000 * a + 100 * b + 10 * c + d) = 1369 := 
sorry

-- Exists statement for the maximum N such that F(N) is perfect square
theorem max_value_N_perfect_square :
  ∃ (a b c d : ℕ), is_six_consecutive a b c d ∧ 
  (1000 * a + 100 * b + 10 * c + d) = 9613 ∧
  ∃ (k : ℤ), F a b c d = k ^ 2 := 
sorry

end NUMINAMATH_GPT_smallest_six_consecutive_number_exists_max_value_N_perfect_square_l1063_106325


namespace NUMINAMATH_GPT_inequality_0_lt_a_lt_1_l1063_106333

theorem inequality_0_lt_a_lt_1 (a : ℝ) (h : 0 < a ∧ a < 1) : 
  (1 / a) + (4 / (1 - a)) ≥ 9 :=
by
  sorry

end NUMINAMATH_GPT_inequality_0_lt_a_lt_1_l1063_106333


namespace NUMINAMATH_GPT_solve_expression_l1063_106382

theorem solve_expression :
  ( (12.05 * 5.4 + 0.6) / (2.3 - 1.8) * (7/3) - (4.07 * 3.5 + 0.45) ^ 2) = 90.493 := 
by 
  sorry

end NUMINAMATH_GPT_solve_expression_l1063_106382
