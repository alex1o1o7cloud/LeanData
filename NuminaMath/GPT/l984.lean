import Mathlib

namespace union_of_M_and_N_l984_98447

namespace SetOperations

def M : Set ℕ := {1, 2, 4}
def N : Set ℕ := {1, 3, 4}

theorem union_of_M_and_N :
  M ∪ N = {1, 2, 3, 4} :=
sorry

end SetOperations

end union_of_M_and_N_l984_98447


namespace inequality_of_positive_numbers_l984_98415

theorem inequality_of_positive_numbers (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) : 
  (a^2 * b^2 + b^2 * c^2 + c^2 * a^2) / (a + b + c) ≥ a * b * c :=
sorry

end inequality_of_positive_numbers_l984_98415


namespace compute_difference_of_squares_l984_98464

theorem compute_difference_of_squares :
  (23 + 15) ^ 2 - (23 - 15) ^ 2 = 1380 := by
  sorry

end compute_difference_of_squares_l984_98464


namespace find_g_5_l984_98426

variable (g : ℝ → ℝ)

axiom func_eqn : ∀ x y : ℝ, x * g y = y * g x
axiom g_10 : g 10 = 15

theorem find_g_5 : g 5 = 7.5 :=
by
  sorry

end find_g_5_l984_98426


namespace sqrt_two_irrational_l984_98473

theorem sqrt_two_irrational : ¬ ∃ (a b : ℕ), b ≠ 0 ∧ (a / b) ^ 2 = 2 :=
by
  sorry

end sqrt_two_irrational_l984_98473


namespace composite_divisor_bound_l984_98489

theorem composite_divisor_bound (n : ℕ) (hn : ¬Prime n ∧ 1 < n) : 
  ∃ a : ℕ, 1 < a ∧ a ≤ Int.sqrt (n : ℤ) ∧ a ∣ n :=
sorry

end composite_divisor_bound_l984_98489


namespace angleC_is_36_l984_98494

theorem angleC_is_36 
  (p q r : ℝ)  -- fictitious types for lines, as Lean needs a type here
  (A B C : ℝ)  -- Angles as Real numbers
  (hpq : p = q)  -- Line p is parallel to line q (represented equivalently for Lean)
  (h : A = 1/4 * B)
  (hr : B + C = 180)
  (vert_opposite : C = A) :
  C = 36 := 
by
  sorry

end angleC_is_36_l984_98494


namespace max_acceptable_ages_l984_98409

noncomputable def acceptable_ages (avg_age std_dev : ℕ) : ℕ :=
  let lower_limit := avg_age - 2 * std_dev
  let upper_limit := avg_age + 2 * std_dev
  upper_limit - lower_limit + 1

theorem max_acceptable_ages : acceptable_ages 40 10 = 41 :=
by
  sorry

end max_acceptable_ages_l984_98409


namespace right_triangle_construction_condition_l984_98425

theorem right_triangle_construction_condition
  (b s : ℝ) 
  (h_b_pos : b > 0)
  (h_s_pos : s > 0)
  (h_perimeter : ∃ (AC BC AB : ℝ), AC = b ∧ AC + BC + AB = 2 * s ∧ (AC^2 + BC^2 = AB^2)) :
  b < s := 
sorry

end right_triangle_construction_condition_l984_98425


namespace solution_set_of_inequality_l984_98487

theorem solution_set_of_inequality :
  {x : ℝ | (x + 1) / x ≤ 3} = {x : ℝ | x < 0} ∪ {x : ℝ | x ≥ 1 / 2} :=
by
  sorry

end solution_set_of_inequality_l984_98487


namespace max_area_of_triangle_l984_98452

theorem max_area_of_triangle (a b c : ℝ) 
  (h1 : ∀ (a b c : ℝ), S = a^2 - (b - c)^2)
  (h2 : b + c = 8) : 
  S ≤ 64 / 17 :=
sorry

end max_area_of_triangle_l984_98452


namespace apple_selling_price_l984_98485

theorem apple_selling_price (CP SP Loss : ℝ) (h₀ : CP = 18) (h₁ : Loss = (1/6) * CP) (h₂ : SP = CP - Loss) : SP = 15 :=
  sorry

end apple_selling_price_l984_98485


namespace MinValue_x3y2z_l984_98427

theorem MinValue_x3y2z (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h : 1/x + 1/y + 1/z = 6) : x^3 * y^2 * z ≥ 1 / 108 :=
by
  sorry

end MinValue_x3y2z_l984_98427


namespace native_answer_l984_98429

-- Define properties to represent native types
inductive NativeType
| normal
| zombie
| half_zombie

-- Define the function that determines the response of a native
def response (native : NativeType) : String :=
  match native with
  | NativeType.normal => "да"
  | NativeType.zombie => "да"
  | NativeType.half_zombie => "да"

-- Define the main theorem
theorem native_answer (native : NativeType) : response native = "да" :=
by sorry

end native_answer_l984_98429


namespace function_passes_through_point_l984_98406

noncomputable def func_graph (a : ℝ) (x : ℝ) : ℝ := a ^ (x - 1) + 2

theorem function_passes_through_point (a : ℝ) (h0 : a > 0) (h1 : a ≠ 1) :
  func_graph a 1 = 3 :=
by
  -- Proof logic is omitted
  sorry

end function_passes_through_point_l984_98406


namespace function_increasing_l984_98408

noncomputable def f (a x : ℝ) := x^2 + a * x + 1 / x

theorem function_increasing (a : ℝ) :
  (∀ x, (1 / 3) < x → 0 ≤ (2 * x + a - 1 / x^2)) → a ≥ 25 / 3 :=
by
  sorry

end function_increasing_l984_98408


namespace students_solved_both_l984_98446

theorem students_solved_both (total_students solved_set_problem solved_function_problem both_problems_wrong: ℕ) 
  (h1: total_students = 50)
  (h2 : solved_set_problem = 40)
  (h3 : solved_function_problem = 31)
  (h4 : both_problems_wrong = 4) :
  (solved_set_problem + solved_function_problem - x + both_problems_wrong = total_students) → x = 25 := by
  sorry

end students_solved_both_l984_98446


namespace bus_time_one_way_l984_98467

-- define conditions
def walk_time_one_way := 5 -- 5 minutes for one walk
def total_annual_travel_time_hours := 365 -- 365 hours per year
def work_days_per_year := 365 -- works every day

-- convert annual travel time from hours to minutes
def total_annual_travel_time_minutes := total_annual_travel_time_hours * 60

-- calculate total daily travel time
def total_daily_travel_time := total_annual_travel_time_minutes / work_days_per_year

-- walking time per day
def total_daily_walking_time := (walk_time_one_way * 4)

-- total bus travel time per day
def total_daily_bus_time := total_daily_travel_time - total_daily_walking_time

-- one-way bus time
theorem bus_time_one_way : total_daily_bus_time / 2 = 20 := by
  sorry

end bus_time_one_way_l984_98467


namespace consecutive_log_sum_l984_98472

theorem consecutive_log_sum : 
  ∃ c d: ℤ, (c + 1 = d) ∧ (c < Real.logb 5 125) ∧ (Real.logb 5 125 < d) ∧ (c + d = 5) :=
sorry

end consecutive_log_sum_l984_98472


namespace hike_up_time_eq_l984_98423

variable (t : ℝ)
variable (h_rate_up : ℝ := 4)
variable (h_rate_down : ℝ := 6)
variable (total_time : ℝ := 3)

theorem hike_up_time_eq (h_rate_up_eq : h_rate_up = 4) 
                        (h_rate_down_eq : h_rate_down = 6) 
                        (total_time_eq : total_time = 3) 
                        (dist_eq : h_rate_up * t = h_rate_down * (total_time - t)) :
  t = 9 / 5 := by
  sorry

end hike_up_time_eq_l984_98423


namespace find_m_n_l984_98440

theorem find_m_n : ∃ (m n : ℕ), 2^n + 1 = m^2 ∧ m = 3 ∧ n = 3 :=
by {
  sorry
}

end find_m_n_l984_98440


namespace taco_cost_l984_98477

theorem taco_cost (T E : ℝ) (h1 : 2 * T + 3 * E = 7.80) (h2 : 3 * T + 5 * E = 12.70) : T = 0.90 := 
by 
  sorry

end taco_cost_l984_98477


namespace intersect_at_single_point_l984_98410

theorem intersect_at_single_point :
  (∃ (x y : ℝ), y = 3 * x + 5 ∧ y = -5 * x + 20 ∧ y = 4 * x + p) → p = 25 / 8 :=
by
  sorry

end intersect_at_single_point_l984_98410


namespace ratio_of_areas_l984_98474

theorem ratio_of_areas (x y l : ℝ)
  (h1 : 2 * (x + 3 * y) = 2 * (l + y))
  (h2 : 2 * x + l = 3 * y) :
  (x * 3 * y) / (l * y) = 3 / 7 :=
by
  -- Proof will be provided here
  sorry

end ratio_of_areas_l984_98474


namespace chemistry_more_than_physics_l984_98417

variables (M P C x : ℤ)

-- Condition 1: The total marks in mathematics and physics is 50
def condition1 : Prop := M + P = 50

-- Condition 2: The average marks in mathematics and chemistry together is 35
def condition2 : Prop := (M + C) / 2 = 35

-- Condition 3: The score in chemistry is some marks more than that in physics
def condition3 : Prop := C = P + x

theorem chemistry_more_than_physics :
  condition1 M P ∧ condition2 M C ∧ (∃ x : ℤ, condition3 P C x ∧ x = 20) :=
sorry

end chemistry_more_than_physics_l984_98417


namespace problem_l984_98445

theorem problem (a b c d e : ℤ) 
  (h1 : a - b + c - e = 7)
  (h2 : b - c + d + e = 8)
  (h3 : c - d + a - e = 4)
  (h4 : d - a + b + e = 3) :
  a + b + c + d + e = 22 := by
  sorry

end problem_l984_98445


namespace james_tylenol_daily_intake_l984_98444

def tylenol_per_tablet : ℕ := 375
def tablets_per_dose : ℕ := 2
def hours_per_dose : ℕ := 6
def hours_per_day : ℕ := 24

theorem james_tylenol_daily_intake :
  (hours_per_day / hours_per_dose) * (tablets_per_dose * tylenol_per_tablet) = 3000 := by
  sorry

end james_tylenol_daily_intake_l984_98444


namespace coplanar_values_l984_98455

namespace CoplanarLines

-- Define parametric equations of the lines
def line1 (t : ℝ) (m : ℝ) : ℝ × ℝ × ℝ := (3 + 2 * t, 2 - t, 5 + m * t)
def line2 (u : ℝ) (m : ℝ) : ℝ × ℝ × ℝ := (4 - m * u, 5 + 3 * u, 6 + 2 * u)

-- Define coplanarity condition
def coplanar_condition (m : ℝ) : Prop :=
  ∃ t u : ℝ, line1 t m = line2 u m

-- Theorem to prove the specific values of m for coplanarity
theorem coplanar_values (m : ℝ) : coplanar_condition m ↔ (m = -13/9 ∨ m = 1) :=
sorry

end CoplanarLines

end coplanar_values_l984_98455


namespace total_ticket_cost_l984_98430

theorem total_ticket_cost :
  ∀ (A : ℝ), 
  -- Conditions
  (6 : ℝ) * (5 : ℝ) + (2 : ℝ) * A = 50 :=
by
  sorry

end total_ticket_cost_l984_98430


namespace largest_decimal_number_l984_98493

theorem largest_decimal_number :
  max (0.9123 : ℝ) (max (0.9912 : ℝ) (max (0.9191 : ℝ) (max (0.9301 : ℝ) (0.9091 : ℝ)))) = 0.9912 :=
by
  sorry

end largest_decimal_number_l984_98493


namespace pineapple_total_cost_correct_l984_98411

-- Define the conditions
def pineapple_cost : ℝ := 1.25
def num_pineapples : ℕ := 12
def shipping_cost : ℝ := 21.00

-- Calculate total cost
noncomputable def total_pineapple_cost : ℝ := pineapple_cost * num_pineapples
noncomputable def total_cost : ℝ := total_pineapple_cost + shipping_cost
noncomputable def cost_per_pineapple : ℝ := total_cost / num_pineapples

-- The proof problem
theorem pineapple_total_cost_correct : cost_per_pineapple = 3 := by
  -- The proof will be filled in here
  sorry

end pineapple_total_cost_correct_l984_98411


namespace planted_fraction_l984_98418

theorem planted_fraction (a b c : ℕ) (x h : ℝ) 
  (h_right_triangle : a = 5 ∧ b = 12)
  (h_hypotenuse : c = 13)
  (h_square_dist : x = 3) : 
  (h * ((a * b) - (x^2))) / (a * b / 2) = (7 : ℝ) / 10 :=
by
  sorry

end planted_fraction_l984_98418


namespace max_streetlights_l984_98492

theorem max_streetlights {road_length streetlight_length : ℝ} 
  (h1 : road_length = 1000)
  (h2 : streetlight_length = 1)
  (fully_illuminated : ∀ (n : ℕ), (n * streetlight_length) < road_length)
  : ∃ max_n, max_n = 1998 ∧ (∀ n, n > max_n → (∃ i, streetlight_length * i > road_length)) :=
sorry

end max_streetlights_l984_98492


namespace algebraic_expression_value_zero_l984_98400

theorem algebraic_expression_value_zero (a b : ℝ) (h : a - b = 2) : (a^3 - 2 * a^2 * b + a * b^2 - 4 * a = 0) :=
sorry

end algebraic_expression_value_zero_l984_98400


namespace midpoint_to_plane_distance_l984_98442

noncomputable def distance_to_plane (A B P: ℝ) (dA dB: ℝ) : ℝ :=
if h : A = B then |dA|
else if h1 : dA + dB = (2 : ℝ) * (dA + dB) / 2 then (dA + dB) / 2
else if h2 : |dB - dA| = (2 : ℝ) * |dB - dA| / 2 then |dB - dA| / 2
else 0

theorem midpoint_to_plane_distance
  (α : Type*)
  (A B P: ℝ)
  {dA dB : ℝ}
  (h_dA : dA = 3)
  (h_dB : dB = 5) :
  distance_to_plane A B P dA dB = 4 ∨ distance_to_plane A B P dA dB = 1 :=
by sorry

end midpoint_to_plane_distance_l984_98442


namespace transformed_area_l984_98428

noncomputable def area_transformation (f : ℝ → ℝ) (x1 x2 x3 : ℝ)
  (h : (1 / 2 * ((x2 - x1) * ((3 * f x3) - (3 * f x1))) - 1 / 2 * ((x3 - x2) * ((3 * f x1) - (3 * f x2)))) = 27) : Prop :=
  1 / 2 * ((0.5 * x2 - 0.5 * x1) * (3 * f (2 * x3) - 3 * f (2 * x1)) - 1 / 2 * (0.5 * x3 - 0.5 * x2) * (3 * f (2 * x1) - 3 * f (2 * x2))) = 40.5

theorem transformed_area
  (f : ℝ → ℝ) (x1 x2 x3 : ℝ)
  (h : 1 / 2 * ((x2 - x1) * (f x3 - f x1) - (x3 - x2) * (f x1 - f x2)) = 27) :
  1 / 2 * ((0.5 * x2 - 0.5 * x1) * (3 * f (2 * x3) - 3 * f (2 * x1)) - 1 / 2 * (0.5 * x3 - 0.5 * x2) * (3 * f (2 * x1) - 3 * f (2 * x2))) = 40.5 := sorry

end transformed_area_l984_98428


namespace sum_sequence_conjecture_l984_98488

theorem sum_sequence_conjecture (a : ℕ → ℚ) (S : ℕ → ℚ) :
  (∀ n : ℕ+, a n = (8 * n) / ((2 * n - 1) ^ 2 * (2 * n + 1) ^ 2)) →
  (∀ n : ℕ+, S n = (S n + a (n + 1))) →
  (∀ n : ℕ+, S 1 = 8 / 9) →
  (∀ n : ℕ+, S n = ((2 * n + 1) ^ 2 - 1) / (2 * n + 1) ^ 2) :=
by {
  sorry
}

end sum_sequence_conjecture_l984_98488


namespace max_difference_and_max_value_of_multiple_of_5_l984_98443

theorem max_difference_and_max_value_of_multiple_of_5:
  ∀ (N : ℕ), 
  (∃ (d : ℕ), d = 0 ∨ d = 5 ∧ N = 740 + d) →
  (∃ (diff : ℕ), diff = 5) ∧ (∃ (max_num : ℕ), max_num = 745) :=
by
  intro N
  rintro ⟨d, (rfl | rfl), rfl⟩
  apply And.intro
  use 5
  use 745
  sorry

end max_difference_and_max_value_of_multiple_of_5_l984_98443


namespace no_positive_integer_satisfies_conditions_l984_98448

theorem no_positive_integer_satisfies_conditions : 
  ¬ ∃ (n : ℕ), (100 ≤ n / 4) ∧ (n / 4 ≤ 999) ∧ (100 ≤ 4 * n) ∧ (4 * n ≤ 999) :=
by
  -- Proof will go here.
  sorry

end no_positive_integer_satisfies_conditions_l984_98448


namespace unique_solution_triple_l984_98416

theorem unique_solution_triple (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (xy / z : ℚ) + (yz / x) + (zx / y) = 3 → (x = 1 ∧ y = 1 ∧ z = 1) := 
by 
  sorry

end unique_solution_triple_l984_98416


namespace deepak_walking_speed_l984_98414

noncomputable def speed_deepak (circumference: ℕ) (wife_speed_kmph: ℚ) (meet_time_min: ℚ) : ℚ :=
  let meet_time_hr := meet_time_min / 60
  let wife_speed_mpm := wife_speed_kmph * 1000 / 60
  let distance_wife := wife_speed_mpm * meet_time_min
  let distance_deepak := circumference - distance_wife
  let deepak_speed_mpm := distance_deepak / meet_time_min
  deepak_speed_mpm * 60 / 1000

theorem deepak_walking_speed
  (circumference: ℕ) 
  (wife_speed_kmph: ℚ)
  (meet_time_min: ℚ)
  (H1: circumference = 627)
  (H2: wife_speed_kmph = 3.75)
  (H3: meet_time_min = 4.56) :
  speed_deepak circumference wife_speed_kmph meet_time_min = 4.5 :=
by
  sorry

end deepak_walking_speed_l984_98414


namespace arccos_sin_eq_l984_98486

open Real

-- Definitions from the problem conditions
noncomputable def radians := π / 180

-- The theorem we need to prove
theorem arccos_sin_eq : arccos (sin 3) = 3 - (π / 2) :=
by
  sorry

end arccos_sin_eq_l984_98486


namespace find_a_l984_98490

def diamond (a b : ℝ) : ℝ := 3 * a - b^2

theorem find_a (a : ℝ) (h : diamond a 6 = 15) : a = 17 :=
by
  sorry

end find_a_l984_98490


namespace divide_54_degree_angle_l984_98439

theorem divide_54_degree_angle :
  ∃ (angle_div : ℝ), angle_div = 54 / 3 :=
by
  sorry

end divide_54_degree_angle_l984_98439


namespace fraction_half_l984_98412

theorem fraction_half {A : ℕ} (h : 8 * (A + 8) - 8 * (A - 8) = 128) (age_eq : A = 64) :
  (64 : ℚ) / (128 : ℚ) = 1 / 2 :=
by
  sorry

end fraction_half_l984_98412


namespace solution_of_inequality_is_correct_l984_98469

-- Inequality condition (x-1)/(2x+1) ≤ 0
def inequality (x : ℝ) : Prop := (x - 1) / (2 * x + 1) ≤ 0 

-- Conditions
def condition1 (x : ℝ) : Prop := (x - 1) * (2 * x + 1) ≤ 0
def condition2 (x : ℝ) : Prop := 2 * x + 1 ≠ 0

-- Combined condition
def combined_condition (x : ℝ) : Prop := condition1 x ∧ condition2 x

-- Solution set
def solution_set : Set ℝ := { x | -1/2 < x ∧ x ≤ 1 }

-- Theorem statement
theorem solution_of_inequality_is_correct :
  ∀ x : ℝ, inequality x ↔ combined_condition x ∧ x ∈ solution_set :=
by
  sorry

end solution_of_inequality_is_correct_l984_98469


namespace factor_tree_value_l984_98436

theorem factor_tree_value :
  ∀ (X Y Z F G : ℕ),
  X = Y * Z → 
  Y = 7 * F → 
  F = 2 * 5 → 
  Z = 11 * G → 
  G = 7 * 3 → 
  X = 16170 := 
by
  intros X Y Z F G
  sorry

end factor_tree_value_l984_98436


namespace marching_band_total_weight_l984_98458

noncomputable def total_weight : ℕ :=
  let trumpet_weight := 5
  let clarinet_weight := 5
  let trombone_weight := 10
  let tuba_weight := 20
  let drum_weight := 15
  let trumpets := 6
  let clarinets := 9
  let trombones := 8
  let tubas := 3
  let drummers := 2
  (trumpets + clarinets) * trumpet_weight + trombones * trombone_weight + tubas * tuba_weight + drummers * drum_weight

theorem marching_band_total_weight : total_weight = 245 := by
  sorry

end marching_band_total_weight_l984_98458


namespace length_of_platform_l984_98465

/--
Problem statement:
A train 450 m long running at 108 km/h crosses a platform in 25 seconds.
Prove that the length of the platform is 300 meters.

Given:
- The train is 450 meters long.
- The train's speed is 108 km/h.
- The train crosses the platform in 25 seconds.

To prove:
The length of the platform is 300 meters.
-/
theorem length_of_platform :
  let train_length := 450
  let train_speed := 108 * (1000 / 3600) -- converting km/h to m/s
  let crossing_time := 25
  let total_distance_covered := train_speed * crossing_time
  let platform_length := total_distance_covered - train_length
  platform_length = 300 := by
  sorry

end length_of_platform_l984_98465


namespace angle_B_in_triangle_is_pi_over_6_l984_98462

theorem angle_B_in_triangle_is_pi_over_6
  (a b c : ℝ)
  (A B C : ℝ)
  (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0)
  (h₄ : A + B + C = π)
  (h₅ : b * (Real.cos C) / (Real.cos B) + c = (2 * Real.sqrt 3 / 3) * a) :
  B = π / 6 :=
by sorry

end angle_B_in_triangle_is_pi_over_6_l984_98462


namespace max_blocks_l984_98405

theorem max_blocks (box_height box_width box_length : ℝ) 
  (typeA_height typeA_width typeA_length typeB_height typeB_width typeB_length : ℝ) 
  (h_box : box_height = 8) (w_box : box_width = 10) (l_box : box_length = 12) 
  (h_typeA : typeA_height = 3) (w_typeA : typeA_width = 2) (l_typeA : typeA_length = 4) 
  (h_typeB : typeB_height = 4) (w_typeB : typeB_width = 3) (l_typeB : typeB_length = 5) : 
  max (⌊box_height / typeA_height⌋ * ⌊box_width / typeA_width⌋ * ⌊box_length / typeA_length⌋)
      (⌊box_height / typeB_height⌋ * ⌊box_width / typeB_width⌋ * ⌊box_length / typeB_length⌋) = 30 := 
  by
  sorry

end max_blocks_l984_98405


namespace length_of_AB_l984_98434

-- Define the ellipse equation
def ellipse (x y : ℝ) : Prop := (x^2) / 25 + (y^2) / 16 = 1

-- Define the line perpendicular to the x-axis passing through the right focus of the ellipse
def line_perpendicular_y_axis_through_focus (y : ℝ) : Prop := true

-- Define the right focus of the ellipse
def right_focus : ℝ × ℝ := (3, 0)

-- Statement to prove the length of the line segment AB
theorem length_of_AB : 
  ∃ A B : ℝ × ℝ, 
  (ellipse A.1 A.2 ∧ ellipse B.1 B.2) ∧ 
  (A.1 = 3 ∧ B.1 = 3) ∧
  (|A.2 - B.2| = 2 * 16 / 5) :=
sorry

end length_of_AB_l984_98434


namespace salary_increase_percentage_l984_98497

variable {P : ℝ} (initial_salary : P > 0)

def salary_after_first_year (P: ℝ) : ℝ :=
  P * 1.12

def salary_after_second_year (P: ℝ) : ℝ :=
  (salary_after_first_year P) * 1.12

def salary_after_third_year (P: ℝ) : ℝ :=
  (salary_after_second_year P) * 1.15

theorem salary_increase_percentage (P: ℝ) (h: P > 0) : 
  (salary_after_third_year P - P) / P * 100 = 44 :=
by 
  sorry

end salary_increase_percentage_l984_98497


namespace sqrt_one_over_four_eq_pm_half_l984_98495

theorem sqrt_one_over_four_eq_pm_half : Real.sqrt (1 / 4) = 1 / 2 ∨ Real.sqrt (1 / 4) = - (1 / 2) := by
  sorry

end sqrt_one_over_four_eq_pm_half_l984_98495


namespace cyclist_speed_l984_98403

theorem cyclist_speed (c d : ℕ) (h1 : d = c + 5) (hc : c ≠ 0) (hd : d ≠ 0)
    (H1 : ∀ tC tD : ℕ, 80 = c * tC → 120 = d * tD → tC = tD) : c = 10 := by
  sorry

end cyclist_speed_l984_98403


namespace extra_men_needed_l984_98459

theorem extra_men_needed (total_length : ℝ) (total_days : ℕ) (initial_men : ℕ) (completed_length : ℝ) (days_passed : ℕ) 
  (remaining_length := total_length - completed_length)
  (remaining_days := total_days - days_passed)
  (current_rate := completed_length / days_passed)
  (required_rate := remaining_length / remaining_days)
  (rate_increase := required_rate / current_rate)
  (total_men_needed := initial_men * rate_increase)
  (extra_men_needed := ⌈total_men_needed⌉ - initial_men) :
  total_length = 15 → 
  total_days = 300 → 
  initial_men = 35 → 
  completed_length = 2.5 → 
  days_passed = 100 → 
  extra_men_needed = 53 :=
by
-- Prove that given the conditions, the number of extra men needed is 53
sorry

end extra_men_needed_l984_98459


namespace ratio_of_construction_paper_packs_l984_98419

-- Definitions for conditions
def marie_glue_sticks : Nat := 15
def marie_construction_paper : Nat := 30
def allison_total_items : Nat := 28
def allison_additional_glue_sticks : Nat := 8

-- Define the main quantity to prove
def allison_glue_sticks : Nat := marie_glue_sticks + allison_additional_glue_sticks
def allison_construction_paper : Nat := allison_total_items - allison_glue_sticks

-- The ratio should be of type Rat or Nat
theorem ratio_of_construction_paper_packs : (marie_construction_paper : Nat) / allison_construction_paper = 6 / 1 := by
  -- This is a placeholder for the actual proof
  sorry

end ratio_of_construction_paper_packs_l984_98419


namespace yellow_balls_l984_98422

theorem yellow_balls (total_balls : ℕ) (prob_yellow : ℚ) (x : ℕ) :
  total_balls = 40 ∧ prob_yellow = 0.30 → (x : ℚ) = 12 := 
by 
  sorry

end yellow_balls_l984_98422


namespace other_coin_value_l984_98482

-- Condition definitions
def total_coins : ℕ := 36
def dime_count : ℕ := 26
def total_value_dollars : ℝ := 3.10
def dime_value : ℝ := 0.10

-- Derived definitions
def total_dimes_value : ℝ := dime_count * dime_value
def remaining_value : ℝ := total_value_dollars - total_dimes_value
def other_coin_count : ℕ := total_coins - dime_count

-- Proof statement
theorem other_coin_value : (remaining_value / other_coin_count) = 0.05 := by
  sorry

end other_coin_value_l984_98482


namespace probability_blue_face_eq_one_third_l984_98454

-- Define the necessary conditions
def numberOfFaces : Nat := 12
def numberOfBlueFaces : Nat := 4

-- Define the term representing the probability
def probabilityOfBlueFace : ℚ := numberOfBlueFaces / numberOfFaces

-- The theorem to prove that the probability is 1/3
theorem probability_blue_face_eq_one_third :
  probabilityOfBlueFace = (1 : ℚ) / 3 :=
  by
  sorry

end probability_blue_face_eq_one_third_l984_98454


namespace Tammy_runs_10_laps_per_day_l984_98475

theorem Tammy_runs_10_laps_per_day
  (total_distance_per_week : ℕ)
  (track_length : ℕ)
  (days_per_week : ℕ)
  (h1 : total_distance_per_week = 3500)
  (h2 : track_length = 50)
  (h3 : days_per_week = 7) :
  (total_distance_per_week / track_length) / days_per_week = 10 := by
  sorry

end Tammy_runs_10_laps_per_day_l984_98475


namespace max_area_of_triangle_l984_98424

-- Define the problem conditions and the maximum area S
theorem max_area_of_triangle
  (A B C : ℝ)
  (a b c S : ℝ)
  (h1 : 4 * S = a^2 - (b - c)^2)
  (h2 : b + c = 8) :
  S ≤ 8 :=
sorry

end max_area_of_triangle_l984_98424


namespace f_at_4_l984_98457

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry
noncomputable def g_inv : ℝ → ℝ := sorry

axiom h1 : ∀ x : ℝ, f (x-1) = g_inv (x-3)
axiom h2 : ∀ x : ℝ, g_inv (g x) = x
axiom h3 : ∀ x : ℝ, g (g_inv x) = x
axiom h4 : g 5 = 2005

theorem f_at_4 : f 4 = 2008 :=
by
  sorry

end f_at_4_l984_98457


namespace q_can_do_work_in_10_days_l984_98435

theorem q_can_do_work_in_10_days (R_p R_q R_pq: ℝ)
  (h1 : R_p = 1 / 15)
  (h2 : R_pq = 1 / 6)
  (h3 : R_p + R_q = R_pq) :
  1 / R_q = 10 :=
by
  -- Proof steps go here.
  sorry

end q_can_do_work_in_10_days_l984_98435


namespace greatest_int_satisfying_inequality_l984_98421

theorem greatest_int_satisfying_inequality : 
  ∃ m : ℤ, (∀ x : ℤ, x - 5 > 4 * x - 1 → x ≤ -2) ∧ (∀ k : ℤ, k < -2 → k - 5 > 4 * k - 1) :=
by
  sorry

end greatest_int_satisfying_inequality_l984_98421


namespace range_of_x_l984_98481

def odot (a b : ℝ) : ℝ := a * b + 2 * a + b

theorem range_of_x :
  {x : ℝ | odot x (x - 2) < 0} = {x : ℝ | -2 < x ∧ x < 1} := 
by sorry

end range_of_x_l984_98481


namespace candy_initial_count_l984_98499

theorem candy_initial_count (candy_given_first candy_given_second candy_given_third candy_bought candy_eaten candy_left initial_candy : ℕ) 
    (h1 : candy_given_first = 18) 
    (h2 : candy_given_second = 12)
    (h3 : candy_given_third = 25)
    (h4 : candy_bought = 10)
    (h5 : candy_eaten = 7)
    (h6 : candy_left = 16)
    (h_initial : candy_left + candy_eaten = initial_candy - candy_bought - candy_given_first - candy_given_second - candy_given_third):
    initial_candy = 68 := 
by 
  sorry

end candy_initial_count_l984_98499


namespace complement_A_eq_B_subset_complement_A_l984_98437

-- Definitions of sets A and B
def A : Set ℝ := {x | x^2 + 4 * x > 0 }
def B (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < a + 1 }

-- The universal set U is the set of all real numbers
def U : Set ℝ := Set.univ

-- Complement of A in U
def complement_U_A : Set ℝ := {x | -4 ≤ x ∧ x ≤ 0}

-- Proof statement for part (1)
theorem complement_A_eq : complement_U_A = {x | -4 ≤ x ∧ x ≤ 0} :=
  sorry 

-- Proof statement for part (2)
theorem B_subset_complement_A (a : ℝ) : B a ⊆ complement_U_A ↔ -3 ≤ a ∧ a ≤ -1 :=
  sorry 

end complement_A_eq_B_subset_complement_A_l984_98437


namespace smallest_positive_integer_l984_98420

theorem smallest_positive_integer (k : ℕ) :
  (∃ k : ℕ, ((2^4 ∣ 1452 * k) ∧ (3^3 ∣ 1452 * k) ∧ (13^3 ∣ 1452 * k))) → 
  k = 676 := 
sorry

end smallest_positive_integer_l984_98420


namespace complement_U_A_l984_98491

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {3, 4, 5}

theorem complement_U_A :
  U \ A = {1, 2, 6} := by
  sorry

end complement_U_A_l984_98491


namespace complex_product_l984_98461

theorem complex_product (a b c d : ℤ) (i : ℂ) (h : i^2 = -1) :
  (6 - 7 * i) * (3 + 6 * i) = 60 + 15 * i :=
  by
    -- proof statements would go here
    sorry

end complex_product_l984_98461


namespace solve_for_x_l984_98483

theorem solve_for_x (x : ℝ) (y : ℝ) (h : y = 1) (h1 : y = 1 / (4 * x^2 + 2 * x + 1)) : 
  x = 0 ∨ x = -1/2 :=
by
  sorry

end solve_for_x_l984_98483


namespace other_ticket_price_l984_98451

theorem other_ticket_price (total_tickets : ℕ) (total_sales : ℝ) (cheap_tickets : ℕ) (cheap_price : ℝ) (expensive_tickets : ℕ) (expensive_price : ℝ) :
  total_tickets = 380 →
  total_sales = 1972.50 →
  cheap_tickets = 205 →
  cheap_price = 4.50 →
  expensive_tickets = 380 - 205 →
  205 * 4.50 + expensive_tickets * expensive_price = 1972.50 →
  expensive_price = 6.00 :=
by
  intros
  -- proof will be filled here
  sorry

end other_ticket_price_l984_98451


namespace find_r_s_l984_98479

theorem find_r_s (r s : ℚ) :
  (-3)^5 - 2*(-3)^4 + 3*(-3)^3 - r*(-3)^2 + s*(-3) - 8 = 0 ∧
  2^5 - 2*(2^4) + 3*(2^3) - r*(2^2) + s*2 - 8 = 0 →
  (r, s) = (-482/15, -1024/15) :=
by
  sorry

end find_r_s_l984_98479


namespace find_x_l984_98432

theorem find_x (x : ℤ) (h : 2 * x = (26 - x) + 19) : x = 15 :=
by
  sorry

end find_x_l984_98432


namespace jane_mean_score_l984_98413

def quiz_scores : List ℕ := [85, 90, 95, 80, 100]

def total_scores : ℕ := quiz_scores.length

def sum_scores : ℕ := quiz_scores.sum

def mean_score : ℕ := sum_scores / total_scores

theorem jane_mean_score : mean_score = 90 := by
  sorry

end jane_mean_score_l984_98413


namespace backpack_pencil_case_combinations_l984_98496

theorem backpack_pencil_case_combinations (backpacks pencil_cases : Fin 2) : 
  (backpacks * pencil_cases) = 4 :=
by 
  sorry

end backpack_pencil_case_combinations_l984_98496


namespace parabola_through_origin_l984_98471

theorem parabola_through_origin {a b c : ℝ} :
  (c = 0 ↔ ∀ x, (0, 0) = (x, a * x^2 + b * x + c)) :=
sorry

end parabola_through_origin_l984_98471


namespace distinct_remainders_sum_quotient_l984_98449

theorem distinct_remainders_sum_quotient :
  let sq_mod_7 (n : Nat) := (n * n) % 7
  let distinct_remainders := List.eraseDup ([sq_mod_7 1, sq_mod_7 2, sq_mod_7 3, sq_mod_7 4, sq_mod_7 5])
  let s := List.sum distinct_remainders
  s / 7 = 1 :=
by
  sorry

end distinct_remainders_sum_quotient_l984_98449


namespace cos_B_find_b_l984_98404

theorem cos_B (a b c : ℝ) (h1 : 2 * b = a + c) (h2 : 7 * a = 3 * c) :
  Real.cos (Real.arccos ((a^2 + c^2 - b^2) / (2 * a * c))) = 11 / 14 := by
  sorry

theorem find_b (a b c : ℝ) (h1 : 2 * b = a + c) (h2 : 7 * a = 3 * c)
  (area : ℝ := 15 * Real.sqrt 3 / 4)
  (h3 : (1/2) * a * c * Real.sin (Real.arccos ((a^2 + c^2 - b^2) / (2 * a * c))) = area) :
  b = 5 := by
  sorry

end cos_B_find_b_l984_98404


namespace smallest_four_digit_number_l984_98453

noncomputable def smallest_four_digit_solution : ℕ := 1011

theorem smallest_four_digit_number (x : ℕ) (h1 : 5 * x ≡ 25 [MOD 20]) (h2 : 3 * x + 10 ≡ 19 [MOD 7]) (h3 : x + 3 ≡ 2 * x [MOD 12]) :
  x = smallest_four_digit_solution :=
by
  sorry

end smallest_four_digit_number_l984_98453


namespace digits_count_of_special_numbers_l984_98433

theorem digits_count_of_special_numbers
  (n : ℕ)
  (h1 : 8^n = 28672) : n = 5 := 
by
  sorry

end digits_count_of_special_numbers_l984_98433


namespace q_l984_98450

-- Definitions for the problem conditions
def slips := 50
def numbers := 12
def slips_per_number := 5
def drawn_slips := 5
def binom := Nat.choose -- Lean function for binomial coefficients

-- Define the probabilities p' and q'
def p' := 12 / (binom slips drawn_slips)
def favorable_q' := (binom numbers 2) * (binom slips_per_number 3) * (binom slips_per_number 2)
def q' := favorable_q' / (binom slips drawn_slips)

-- The statement we need to prove
theorem q'_over_p'_equals_550 : q' / p' = 550 :=
by sorry

end q_l984_98450


namespace pipe_fill_time_without_leak_l984_98460

theorem pipe_fill_time_without_leak (T : ℝ) (h1 : (1 / 9 : ℝ) = 1 / T - 1 / 4.5) : T = 3 := 
by
  sorry

end pipe_fill_time_without_leak_l984_98460


namespace feasible_test_for_rhombus_l984_98401

def is_rhombus (paper : Type) : Prop :=
  true -- Placeholder for the actual definition of a rhombus

def method_A (paper : Type) : Prop :=
  -- Placeholder for the condition "Measure if the four internal angles are equal"
  true

def method_B (paper : Type) : Prop :=
  -- Placeholder for the condition "Measure if the two diagonals are equal"
  true

def method_C (paper : Type) : Prop :=
  -- Placeholder for the condition "Measure if the distance from the intersection of the two diagonals to the four vertices is equal"
  true

def method_D (paper : Type) : Prop :=
  -- Placeholder for the condition "Fold the paper along the two diagonals separately and see if the parts on both sides of the diagonals coincide completely each time"
  true

theorem feasible_test_for_rhombus (paper : Type) : is_rhombus paper → method_D paper :=
by
  intro h_rhombus
  sorry

end feasible_test_for_rhombus_l984_98401


namespace sandy_paid_for_pants_l984_98407

-- Define the costs and change as constants
def cost_of_shirt : ℝ := 8.25
def amount_paid_with : ℝ := 20.00
def change_received : ℝ := 2.51

-- Define the amount paid for pants
def amount_paid_for_pants : ℝ := 9.24

-- The theorem stating the problem
theorem sandy_paid_for_pants : 
  amount_paid_with - (cost_of_shirt + change_received) = amount_paid_for_pants := 
by 
  -- proof is required here
  sorry

end sandy_paid_for_pants_l984_98407


namespace hyperbola_eccentricity_l984_98470

theorem hyperbola_eccentricity (a b c : ℝ) (h_a : a > 0) (h_b : b > 0)
  (h_hyperbola: ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1)
  (h_asymptotes_l1: ∀ x : ℝ, y = (b / a) * x)
  (h_asymptotes_l2: ∀ x : ℝ, y = -(b / a) * x)
  (h_focus: c^2 = a^2 + b^2)
  (h_symmetric: ∀ m : ℝ, m = -c / 2 ∧ (m, (b * c) / (2 * a)) ∈ { p : ℝ × ℝ | p.2 = -(b / a) * p.1 }) :
  (c / a) = 2 := sorry

end hyperbola_eccentricity_l984_98470


namespace sum_of_products_l984_98466

theorem sum_of_products {a b c : ℝ}
  (h1 : a ^ 2 + b ^ 2 + c ^ 2 = 138)
  (h2 : a + b + c = 20) :
  a * b + b * c + c * a = 131 := 
by
  sorry

end sum_of_products_l984_98466


namespace super_k_teams_l984_98478

theorem super_k_teams (n : ℕ) (h : n * (n - 1) / 2 = 45) : n = 10 :=
sorry

end super_k_teams_l984_98478


namespace compute_expression_l984_98441

theorem compute_expression : 3034 - (1002 / 200.4) = 3029 := by
  sorry

end compute_expression_l984_98441


namespace problem_statement_l984_98438

theorem problem_statement (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) 
  (h4 : x^3 + (1 / (y + 2016)) = y^3 + (1 / (z + 2016))) 
  (h5 : y^3 + (1 / (z + 2016)) = z^3 + (1 / (x + 2016))) : 
  (x^3 + y^3 + z^3) / (x * y * z) = 3 :=
by
  sorry

end problem_statement_l984_98438


namespace electricity_consumption_l984_98484

variable (x y : ℝ)

-- y = 0.55 * x
def electricity_fee := 0.55 * x

-- if y = 40.7 then x should be 74
theorem electricity_consumption :
  (∃ x, electricity_fee x = 40.7) → (x = 74) :=
by
  sorry

end electricity_consumption_l984_98484


namespace perpendicular_condition_l984_98476

-- Definitions of lines
def line1 (x y : ℝ) : Prop := x + y = 0
def line2 (x y : ℝ) (a : ℝ) : Prop := x - a * y = 0

-- Theorem: Prove that a = 1 is a necessary and sufficient condition for the lines
-- line1 and line2 to be perpendicular.
theorem perpendicular_condition (a : ℝ) : 
  (∀ x y : ℝ, line1 x y → line2 x y a) ↔ (a = 1) :=
sorry

end perpendicular_condition_l984_98476


namespace find_c_l984_98431

variables {α : Type*} [LinearOrderedField α]

def p (x : α) : α := 3 * x - 9
def q (x : α) (c : α) : α := 4 * x - c

-- We aim to prove that if p(q(3,c)) = 6, then c = 7
theorem find_c (c : α) : p (q 3 c) = 6 → c = 7 :=
by
  sorry

end find_c_l984_98431


namespace min_value_y_l984_98463

theorem min_value_y (x : ℝ) (h : x > 5 / 4) : 
  ∃ y, y = 4*x - 1 + 1 / (4*x - 5) ∧ y ≥ 6 :=
by
  sorry

end min_value_y_l984_98463


namespace initial_number_of_people_l984_98468

theorem initial_number_of_people (X : ℕ) (h : ((X - 10) + 15 = 17)) : X = 12 :=
by
  sorry

end initial_number_of_people_l984_98468


namespace solve_z_l984_98498

variable (z : ℂ) -- Define the variable z in the complex number system
variable (i : ℂ) -- Define the variable i in the complex number system

-- State the conditions: 2 - 3i * z = 4 + 5i * z and i^2 = -1
axiom cond1 : 2 - 3 * i * z = 4 + 5 * i * z
axiom cond2 : i^2 = -1

-- The theorem to prove: z = i / 4
theorem solve_z : z = i / 4 :=
by
  sorry

end solve_z_l984_98498


namespace extra_profit_is_60000_l984_98456

theorem extra_profit_is_60000 (base_house_cost special_house_cost base_house_price special_house_price : ℝ) :
  (special_house_cost = base_house_cost + 100000) →
  (special_house_price = 1.5 * base_house_price) →
  (base_house_price = 320000) →
  (special_house_price - base_house_price - 100000 = 60000) :=
by
  -- Definitions and conditions
  intro h1 h2 h3
  -- Placeholder for the eventual proof
  sorry

end extra_profit_is_60000_l984_98456


namespace students_with_dog_and_cat_only_l984_98480

theorem students_with_dog_and_cat_only
  (U : Finset (ℕ)) -- Universe of students
  (D C B : Finset (ℕ)) -- Sets of students with dogs, cats, and birds respectively
  (hU : U.card = 50)
  (hD : D.card = 30)
  (hC : C.card = 35)
  (hB : B.card = 10)
  (hIntersection : (D ∩ C ∩ B).card = 5) :
  ((D ∩ C) \ B).card = 25 := 
sorry

end students_with_dog_and_cat_only_l984_98480


namespace least_number_divisible_by_12_leaves_remainder_4_is_40_l984_98402

theorem least_number_divisible_by_12_leaves_remainder_4_is_40 :
  ∃ n : ℕ, (∀ k : ℕ, n = 12 * k + 4) ∧ (∀ m : ℕ, (∀ k : ℕ, m = 12 * k + 4) → n ≤ m) ∧ n = 40 :=
by
  sorry

end least_number_divisible_by_12_leaves_remainder_4_is_40_l984_98402
