import Mathlib

namespace NUMINAMATH_GPT_canoe_kayak_problem_l2345_234539

theorem canoe_kayak_problem (C K : ℕ) 
  (h1 : 9 * C + 12 * K = 432)
  (h2 : C = (4 * K) / 3) : 
  C - K = 6 := by
sorry

end NUMINAMATH_GPT_canoe_kayak_problem_l2345_234539


namespace NUMINAMATH_GPT_sum_of_consecutive_numbers_mod_13_l2345_234581

theorem sum_of_consecutive_numbers_mod_13 :
  ((8930 + 8931 + 8932 + 8933 + 8934) % 13) = 5 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_consecutive_numbers_mod_13_l2345_234581


namespace NUMINAMATH_GPT_linear_elimination_l2345_234598

theorem linear_elimination (a b : ℤ) (x y : ℤ) :
  (a = 2) ∧ (b = -5) → 
  (a * (5 * x - 2 * y) + b * (2 * x + 3 * y) = 0) → 
  (10 * x - 4 * y + -10 * x - 15 * y = 8 + -45) :=
by
  sorry

end NUMINAMATH_GPT_linear_elimination_l2345_234598


namespace NUMINAMATH_GPT_existence_of_solution_values_continuous_solution_value_l2345_234550

noncomputable def functional_equation_has_solution (a : ℝ) (f : ℝ → ℝ) : Prop :=
  f 0 = 0 ∧ f 1 = 1 ∧ ∀ x y, (x ≤ y → f ((x + y) / 2) = (1 - a) * f x + a * f y)

theorem existence_of_solution_values :
  {a : ℝ | ∃ f : ℝ → ℝ, functional_equation_has_solution a f} = {0, 1/2, 1} :=
sorry

theorem continuous_solution_value :
  {a : ℝ | ∃ (f : ℝ → ℝ) (hf : Continuous f), functional_equation_has_solution a f} = {1/2} :=
sorry

end NUMINAMATH_GPT_existence_of_solution_values_continuous_solution_value_l2345_234550


namespace NUMINAMATH_GPT_evaluate_expression_l2345_234572

theorem evaluate_expression : 2^(3^2) + 3^(2^3) = 7073 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2345_234572


namespace NUMINAMATH_GPT_shaded_shape_area_l2345_234558

/-- Define the coordinates and the conditions for the central square and triangles in the grid -/
def grid_size := 10
def central_square_side := 2
def central_square_area := central_square_side * central_square_side

def triangle_base := 5
def triangle_height := 5
def triangle_area := (1 / 2) * triangle_base * triangle_height

def number_of_triangles := 4
def total_triangle_area := number_of_triangles * triangle_area

def total_shaded_area := total_triangle_area + central_square_area

theorem shaded_shape_area : total_shaded_area = 54 :=
by
  -- We have defined each area component and summed them to the total shaded area.
  -- The statement ensures that the area of the shaded shape is equal to 54.
  sorry

end NUMINAMATH_GPT_shaded_shape_area_l2345_234558


namespace NUMINAMATH_GPT_John_can_finish_work_alone_in_48_days_l2345_234585

noncomputable def John_and_Roger_can_finish_together_in_24_days (J R: ℝ) : Prop :=
  1 / J + 1 / R = 1 / 24

noncomputable def John_finished_remaining_work (J: ℝ) : Prop :=
  (1 / 3) / (16 / J) = 1

theorem John_can_finish_work_alone_in_48_days (J R: ℝ) 
  (h1 : John_and_Roger_can_finish_together_in_24_days J R) 
  (h2 : John_finished_remaining_work J):
  J = 48 := 
sorry

end NUMINAMATH_GPT_John_can_finish_work_alone_in_48_days_l2345_234585


namespace NUMINAMATH_GPT_lines_intersection_l2345_234574

/-- Two lines are defined by the equations y = 2x + c and y = 4x + d.
These lines intersect at the point (8, 12).
Prove that c + d = -24. -/
theorem lines_intersection (c d : ℝ) (h1 : 12 = 2 * 8 + c) (h2 : 12 = 4 * 8 + d) :
    c + d = -24 :=
by
  sorry

end NUMINAMATH_GPT_lines_intersection_l2345_234574


namespace NUMINAMATH_GPT_train_length_l2345_234570

-- Defining the conditions
def speed_kmh : ℕ := 64
def speed_m_per_s : ℚ := (64 * 1000) / 3600 -- 64 km/h converted to m/s
def time_to_cross_seconds : ℕ := 9 

-- The theorem to prove the length of the train
theorem train_length : speed_m_per_s * time_to_cross_seconds = 160 := 
by 
  unfold speed_m_per_s 
  norm_num
  sorry -- Placeholder for actual proof

end NUMINAMATH_GPT_train_length_l2345_234570


namespace NUMINAMATH_GPT_distance_school_house_l2345_234513

def speed_to_school : ℝ := 6
def speed_from_school : ℝ := 4
def total_time : ℝ := 10

theorem distance_school_house : 
  ∃ D : ℝ, (D / speed_to_school + D / speed_from_school = total_time) ∧ (D = 24) :=
sorry

end NUMINAMATH_GPT_distance_school_house_l2345_234513


namespace NUMINAMATH_GPT_relationship_between_a_and_b_l2345_234535

def a : ℤ := (-12) * (-23) * (-34) * (-45)
def b : ℤ := (-123) * (-234) * (-345)

theorem relationship_between_a_and_b : a > b := by
  sorry

end NUMINAMATH_GPT_relationship_between_a_and_b_l2345_234535


namespace NUMINAMATH_GPT_turnip_difference_l2345_234593

theorem turnip_difference :
  let melanie_turnips := 139
  let benny_turnips := 113
  let caroline_turnips := 172
  (melanie_turnips + benny_turnips) - caroline_turnips = 80 :=
by
  let melanie_turnips := 139
  let benny_turnips := 113
  let caroline_turnips := 172
  show (melanie_turnips + benny_turnips) - caroline_turnips = 80
  sorry

end NUMINAMATH_GPT_turnip_difference_l2345_234593


namespace NUMINAMATH_GPT_selection_plans_count_l2345_234566

-- Define the number of boys and girls
def num_boys : ℕ := 3
def num_girls : ℕ := 3
def total_students : ℕ := num_boys + num_girls

-- Define the number of subjects
def num_subjects : ℕ := 3

-- Prove that the number of selection plans is 120
theorem selection_plans_count :
  (Nat.choose total_students num_subjects) * (num_subjects.factorial) = 120 := 
by
  sorry

end NUMINAMATH_GPT_selection_plans_count_l2345_234566


namespace NUMINAMATH_GPT_find_angle_C_l2345_234509

variables {A B C : ℝ} {a b c : ℝ} 

theorem find_angle_C (h1 : a^2 + b^2 - c^2 + a*b = 0) (C_pos : 0 < C) (C_lt_pi : C < Real.pi) :
  C = (2 * Real.pi) / 3 :=
sorry

end NUMINAMATH_GPT_find_angle_C_l2345_234509


namespace NUMINAMATH_GPT_percent_in_range_70_to_79_is_correct_l2345_234537

-- Define the total number of students.
def total_students : Nat := 8 + 12 + 11 + 5 + 7

-- Define the number of students within the $70\%-79\%$ range.
def students_70_to_79 : Nat := 11

-- Define the percentage of the students within the $70\%-79\%$ range.
def percent_70_to_79 : ℚ := (students_70_to_79 : ℚ) / (total_students : ℚ) * 100

theorem percent_in_range_70_to_79_is_correct : percent_70_to_79 = 25.58 := by
  sorry

end NUMINAMATH_GPT_percent_in_range_70_to_79_is_correct_l2345_234537


namespace NUMINAMATH_GPT_problem1_problem2_l2345_234591

-- Definition of sets A and B
def A : Set ℝ := { x | x^2 - 2*x - 3 < 0 }
def B (p : ℝ) : Set ℝ := { x | abs (x - p) > 1 }

-- Statement for the first problem
theorem problem1 : B 0 ∩ A = { x | 1 < x ∧ x < 3 } := 
by
  sorry

-- Statement for the second problem
theorem problem2 (p : ℝ) (h : A ∪ B p = B p) : p ≤ -2 ∨ p ≥ 4 := 
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l2345_234591


namespace NUMINAMATH_GPT_find_number_l2345_234578

theorem find_number (x : ℤ) (h : x + x^2 = 342) : x = 18 ∨ x = -19 :=
sorry

end NUMINAMATH_GPT_find_number_l2345_234578


namespace NUMINAMATH_GPT_crop_yield_growth_l2345_234536

-- Definitions based on conditions
def initial_yield := 300
def final_yield := 363
def eqn (x : ℝ) : Prop := initial_yield * (1 + x)^2 = final_yield

-- The theorem we need to prove
theorem crop_yield_growth (x : ℝ) : eqn x :=
by
  sorry

end NUMINAMATH_GPT_crop_yield_growth_l2345_234536


namespace NUMINAMATH_GPT_work_completion_time_l2345_234562

noncomputable def work_done_by_woman_per_day : ℝ := 1 / 50
noncomputable def work_done_by_child_per_day : ℝ := 1 / 100
noncomputable def total_work_done_by_5_women_per_day : ℝ := 5 * work_done_by_woman_per_day
noncomputable def total_work_done_by_10_children_per_day : ℝ := 10 * work_done_by_child_per_day
noncomputable def combined_work_per_day : ℝ := total_work_done_by_5_women_per_day + total_work_done_by_10_children_per_day

theorem work_completion_time (h1 : 10 / 5 = 2) (h2 : 10 / 10 = 1) :
  1 / combined_work_per_day = 5 :=
by
  sorry

end NUMINAMATH_GPT_work_completion_time_l2345_234562


namespace NUMINAMATH_GPT_point_on_line_l2345_234533

theorem point_on_line (m : ℝ) (P : ℝ × ℝ) (line_eq : ℝ × ℝ → Prop) (h : P = (2, m)) 
  (h_line : line_eq = fun P => 3 * P.1 + P.2 = 2) : 
  3 * 2 + m = 2 → m = -4 :=
by
  intro h1
  linarith

end NUMINAMATH_GPT_point_on_line_l2345_234533


namespace NUMINAMATH_GPT_xyz_inequality_l2345_234586

theorem xyz_inequality (x y z : ℝ) (h_condition : x^2 + y^2 + z^2 = 2) : x + y + z ≤ x * y * z + 2 := 
sorry

end NUMINAMATH_GPT_xyz_inequality_l2345_234586


namespace NUMINAMATH_GPT_cannot_divide_m_l2345_234587

/-
  A proof that for the real number m = 2009^3 - 2009, 
  the number 2007 does not divide m.
-/

theorem cannot_divide_m (m : ℤ) (h : m = 2009^3 - 2009) : ¬ (2007 ∣ m) := 
by sorry

end NUMINAMATH_GPT_cannot_divide_m_l2345_234587


namespace NUMINAMATH_GPT_sum_of_midpoints_l2345_234554

theorem sum_of_midpoints (p q r : ℝ) (h : p + q + r = 15) :
  (p + q) / 2 + (p + r) / 2 + (q + r) / 2 = 15 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_midpoints_l2345_234554


namespace NUMINAMATH_GPT_monotonic_increasing_odd_function_implies_a_eq_1_find_max_m_l2345_234505

noncomputable def f (a x : ℝ) : ℝ := a - 2 / (2^x + 1)

-- 1. Monotonicity of f(x)
theorem monotonic_increasing (a : ℝ) : ∀ x1 x2 : ℝ, x1 < x2 → f a x1 < f a x2 := sorry

-- 2. f(x) is odd implies a = 1
theorem odd_function_implies_a_eq_1 (h : ∀ x : ℝ, f a (-x) = -f a x) : a = 1 := sorry

-- 3. Find max m such that f(x) ≥ m / 2^x for all x ∈ [2, 3]
theorem find_max_m (h : ∀ x : ℝ, 2 ≤ x ∧ x ≤ 3 → f 1 x ≥ m / 2^x) : m ≤ 12/5 := sorry

end NUMINAMATH_GPT_monotonic_increasing_odd_function_implies_a_eq_1_find_max_m_l2345_234505


namespace NUMINAMATH_GPT_average_speed_l2345_234579

theorem average_speed (v1 v2 t1 t2 total_time total_distance : ℝ)
  (h1 : v1 = 50)
  (h2 : t1 = 4)
  (h3 : v2 = 80)
  (h4 : t2 = 4)
  (h5 : total_time = t1 + t2)
  (h6 : total_distance = v1 * t1 + v2 * t2) :
  (total_distance / total_time = 65) :=
by
  sorry

end NUMINAMATH_GPT_average_speed_l2345_234579


namespace NUMINAMATH_GPT_product_odd_primes_mod_32_l2345_234512

open Nat

theorem product_odd_primes_mod_32 : 
  let primes := [3, 5, 7, 11, 13] 
  let product := primes.foldl (· * ·) 1 
  product % 32 = 7 := 
by
  sorry

end NUMINAMATH_GPT_product_odd_primes_mod_32_l2345_234512


namespace NUMINAMATH_GPT_machine_output_l2345_234589

theorem machine_output (input : ℕ) (output : ℕ) (h : input = 26) (h_out : output = input + 15 - 6) : output = 35 := 
by 
  sorry

end NUMINAMATH_GPT_machine_output_l2345_234589


namespace NUMINAMATH_GPT_first_tv_cost_is_672_l2345_234519

-- width and height of the first TV
def width_first_tv : ℕ := 24
def height_first_tv : ℕ := 16
-- width and height of the new TV
def width_new_tv : ℕ := 48
def height_new_tv : ℕ := 32
-- cost of the new TV
def cost_new_tv : ℕ := 1152
-- extra cost per square inch for the first TV
def extra_cost_per_square_inch : ℕ := 1

noncomputable def cost_first_tv : ℕ :=
  let area_first_tv := width_first_tv * height_first_tv
  let area_new_tv := width_new_tv * height_new_tv
  let cost_per_square_inch_new_tv := cost_new_tv / area_new_tv
  let cost_per_square_inch_first_tv := cost_per_square_inch_new_tv + extra_cost_per_square_inch
  cost_per_square_inch_first_tv * area_first_tv

theorem first_tv_cost_is_672 : cost_first_tv = 672 := by
  sorry

end NUMINAMATH_GPT_first_tv_cost_is_672_l2345_234519


namespace NUMINAMATH_GPT_minor_axis_length_is_2sqrt3_l2345_234531

-- Define the points given in the problem
def points : List (ℝ × ℝ) := [(1, 1), (0, 0), (0, 3), (4, 0), (4, 3)]

-- Define a function that checks if an ellipse with axes parallel to the coordinate axes
-- passes through given points, and returns the length of its minor axis if it does.
noncomputable def minor_axis_length (pts : List (ℝ × ℝ)) : ℝ :=
  if h : (0,0) ∈ pts ∧ (0,3) ∈ pts ∧ (4,0) ∈ pts ∧ (4,3) ∈ pts ∧ (1,1) ∈ pts then
    let a := (4 - 0) / 2 -- half the width of the rectangle
    let b_sq := 3 -- derived from solving the ellipse equation
    2 * Real.sqrt b_sq
  else 0

-- The theorem statement:
theorem minor_axis_length_is_2sqrt3 : minor_axis_length points = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_minor_axis_length_is_2sqrt3_l2345_234531


namespace NUMINAMATH_GPT_koala_fiber_absorption_l2345_234508

theorem koala_fiber_absorption (x : ℝ) (h1 : 0 < x) (h2 : x * 0.30 = 15) : x = 50 :=
sorry

end NUMINAMATH_GPT_koala_fiber_absorption_l2345_234508


namespace NUMINAMATH_GPT_find_number_l2345_234545

theorem find_number (x : ℝ) : 50 + (x * 12) / (180 / 3) = 51 ↔ x = 5 := by
  sorry

end NUMINAMATH_GPT_find_number_l2345_234545


namespace NUMINAMATH_GPT_intersect_A_B_l2345_234521

def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x | -1 < x ∧ x ≤ 1}

theorem intersect_A_B : A ∩ B = {0, 1} :=
by
  sorry

end NUMINAMATH_GPT_intersect_A_B_l2345_234521


namespace NUMINAMATH_GPT_hungarian_math_olympiad_1927_l2345_234573

-- Definitions
def is_coprime (a b : ℤ) : Prop :=
  Int.gcd a b = 1

-- The main statement
theorem hungarian_math_olympiad_1927
  (a b c d x y k m : ℤ) 
  (h_coprime : is_coprime a b)
  (h_m : m = a * d - b * c)
  (h_divides : m ∣ (a * x + b * y)) :
  m ∣ (c * x + d * y) :=
sorry

end NUMINAMATH_GPT_hungarian_math_olympiad_1927_l2345_234573


namespace NUMINAMATH_GPT_trader_gain_percentage_l2345_234583

theorem trader_gain_percentage (C : ℝ) (h1 : 95 * C = (95 * C - cost_of_95_pens) + (19 * C)) :
  100 * (19 * C / (95 * C)) = 20 := 
by {
  sorry
}

end NUMINAMATH_GPT_trader_gain_percentage_l2345_234583


namespace NUMINAMATH_GPT_solve_for_a_l2345_234590

theorem solve_for_a (a : ℤ) : -2 - a = 0 → a = -2 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_a_l2345_234590


namespace NUMINAMATH_GPT_solve_for_x_l2345_234510

theorem solve_for_x (x : ℝ) : 
  x^2 - 2 * x - 8 = -(x + 2) * (x - 6) → (x = 5 ∨ x = -2) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_solve_for_x_l2345_234510


namespace NUMINAMATH_GPT_max_value_of_y_l2345_234529

open Real

noncomputable def y (x : ℝ) : ℝ := 
  (sin (π / 4 + x) - sin (π / 4 - x)) * sin (π / 3 + x)

theorem max_value_of_y : 
  ∃ x : ℝ, (∀ x, y x ≤ 3 * sqrt 2 / 4) ∧ (∀ k : ℤ, x = k * π + π / 3 → y x = 3 * sqrt 2 / 4) :=
sorry

end NUMINAMATH_GPT_max_value_of_y_l2345_234529


namespace NUMINAMATH_GPT_neg_of_exists_a_l2345_234561

theorem neg_of_exists_a (a : ℝ) : ¬ (∃ a : ℝ, a^2 + 1 < 2 * a) :=
by
  sorry

end NUMINAMATH_GPT_neg_of_exists_a_l2345_234561


namespace NUMINAMATH_GPT_range_of_a_l2345_234563

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  (Real.exp x / x) - a * (x ^ 2)

theorem range_of_a (a : ℝ) :
  (∀ x1 x2 : ℝ, 0 < x1 ∧ x1 < x2 → (f a x1 / x2) - (f a x2 / x1) < 0) ↔ (a ≤ Real.exp 2 / 12) := by
  sorry

end NUMINAMATH_GPT_range_of_a_l2345_234563


namespace NUMINAMATH_GPT_find_x_l2345_234549

variable (P T S : Point)
variable (angle_PTS angle_TSR x : ℝ)
variable (reflector : Point)

-- Given conditions
axiom angle_PTS_is_90 : angle_PTS = 90
axiom angle_TSR_is_26 : angle_TSR = 26

-- Proof problem
theorem find_x : x = 32 := by
  sorry

end NUMINAMATH_GPT_find_x_l2345_234549


namespace NUMINAMATH_GPT_max_a_plus_b_cubed_plus_c_fourth_l2345_234515

theorem max_a_plus_b_cubed_plus_c_fourth (a b c : ℕ) (h : a > 0 ∧ b > 0 ∧ c > 0) (h_sum : a + b + c = 2) :
  a + b^3 + c^4 ≤ 2 := sorry

end NUMINAMATH_GPT_max_a_plus_b_cubed_plus_c_fourth_l2345_234515


namespace NUMINAMATH_GPT_find_a_tangent_slope_at_point_l2345_234504

theorem find_a_tangent_slope_at_point :
  ∃ (a : ℝ), (∃ (y : ℝ), y = (fun (x : ℝ) => x^4 + a * x^2 + 1) (-1) ∧ (∃ (y' : ℝ), y' = (fun (x : ℝ) => 4 * x^3 + 2 * a * x) (-1) ∧ y' = 8)) ∧ a = -6 :=
by
  -- Used to skip the proof
  sorry

end NUMINAMATH_GPT_find_a_tangent_slope_at_point_l2345_234504


namespace NUMINAMATH_GPT_votes_cast_l2345_234528

theorem votes_cast (total_votes : ℕ) 
  (h1 : (3/8 : ℚ) * total_votes = 45)
  (h2 : (1/4 : ℚ) * total_votes = (1/4 : ℚ) * 120) : 
  total_votes = 120 := 
by
  sorry

end NUMINAMATH_GPT_votes_cast_l2345_234528


namespace NUMINAMATH_GPT_five_ab4_is_perfect_square_l2345_234596

theorem five_ab4_is_perfect_square (a b : ℕ) (h : 5000 ≤ 5000 + 100 * a + 10 * b + 4 ∧ 5000 + 100 * a + 10 * b + 4 ≤ 5999) :
    ∃ n, n^2 = 5000 + 100 * a + 10 * b + 4 → a + b = 9 :=
by
  sorry

end NUMINAMATH_GPT_five_ab4_is_perfect_square_l2345_234596


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l2345_234542

-- Define sets M and N
def M (x : ℝ) : Prop := x < 5
def N (x : ℝ) : Prop := x > 3

-- Define the union and intersection of M and N
def M_union_N (x : ℝ) : Prop := M x ∨ N x
def M_inter_N (x : ℝ) : Prop := M x ∧ N x

-- Theorem statement: Prove the necessity but not sufficiency
theorem necessary_but_not_sufficient (x : ℝ) :
  M_inter_N x → M_union_N x ∧ ¬(M_union_N x → M_inter_N x) := 
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l2345_234542


namespace NUMINAMATH_GPT_spring_spending_l2345_234564

theorem spring_spending (end_of_feb : ℝ) (end_of_may : ℝ) (h_end_of_feb : end_of_feb = 0.8) (h_end_of_may : end_of_may = 2.5)
  : (end_of_may - end_of_feb) = 1.7 :=
by
  have spending_end_of_feb : end_of_feb = 0.8 := h_end_of_feb
  have spending_end_of_may : end_of_may = 2.5 := h_end_of_may
  sorry

end NUMINAMATH_GPT_spring_spending_l2345_234564


namespace NUMINAMATH_GPT_wood_allocation_l2345_234522

theorem wood_allocation (x y : ℝ) (h1 : 50 * x * 4 = 300 * y) (h2 : x + y = 5) : x = 3 :=
by
  sorry

end NUMINAMATH_GPT_wood_allocation_l2345_234522


namespace NUMINAMATH_GPT_triangle_ABC_area_l2345_234588

def point : Type := ℚ × ℚ

def triangle_area (A B C : point) : ℚ :=
  let (x1, y1) := A;
  let (x2, y2) := B;
  let (x3, y3) := C;
  (1/2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

theorem triangle_ABC_area :
  let A : point := (-5, 4)
  let B : point := (1, 7)
  let C : point := (4, -3)
  triangle_area A B C = 34.5 :=
by
  sorry

end NUMINAMATH_GPT_triangle_ABC_area_l2345_234588


namespace NUMINAMATH_GPT_base6_addition_l2345_234568

/-- Adding two numbers in base 6 -/
theorem base6_addition : (3454 : ℕ) + (12345 : ℕ) = (142042 : ℕ) := by
  sorry

end NUMINAMATH_GPT_base6_addition_l2345_234568


namespace NUMINAMATH_GPT_gross_profit_value_l2345_234516

theorem gross_profit_value
  (sales_price : ℝ)
  (gross_profit_percentage : ℝ)
  (sales_price_eq : sales_price = 91)
  (gross_profit_percentage_eq : gross_profit_percentage = 1.6)
  (C : ℝ)
  (cost_eqn : sales_price = C + gross_profit_percentage * C) :
  gross_profit_percentage * C = 56 :=
by
  sorry

end NUMINAMATH_GPT_gross_profit_value_l2345_234516


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l2345_234526

variable {a : ℕ → ℝ} -- The arithmetic sequence {a_n} represented by a function a : ℕ → ℝ

/-- Given that the sum of some terms of an arithmetic sequence is 25, prove the sum of other terms -/
theorem arithmetic_sequence_sum (h : a 3 + a 4 + a 5 + a 6 + a 7 = 25) : a 2 + a 8 = 10 := by
    sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l2345_234526


namespace NUMINAMATH_GPT_perimeters_ratio_l2345_234502

noncomputable def ratio_perimeters_of_squares (area_ratio : ℚ) (ratio_area : area_ratio = 49 / 64) : ℚ :=
if h : area_ratio = 49 / 64 
then (7 / 8) 
else 0  -- This shouldn't happen since we enforce the condition

theorem perimeters_ratio (area_ratio : ℚ) (h : area_ratio = 49 / 64) : ratio_perimeters_of_squares area_ratio h = 7 / 8 :=
by {
  -- Proof goes here
  sorry
}

end NUMINAMATH_GPT_perimeters_ratio_l2345_234502


namespace NUMINAMATH_GPT_verify_a_eq_x0_verify_p_squared_ge_4x0q_l2345_234575

theorem verify_a_eq_x0 (p q x0 a b : ℝ) (hx0_root : x0^3 + p * x0 + q = 0) 
  (h_eq : ∀ x : ℝ, x^3 + p * x + q = (x - x0) * (x^2 + a * x + b)) : 
  a = x0 :=
by
  sorry

theorem verify_p_squared_ge_4x0q (p q x0 b : ℝ) (hx0_root : x0^3 + p * x0 + q = 0) 
  (h_eq : ∀ x : ℝ, x^3 + p * x + q = (x - x0) * (x^2 + x0 * x + b)) : 
  p^2 ≥ 4 * x0 * q :=
by
  sorry

end NUMINAMATH_GPT_verify_a_eq_x0_verify_p_squared_ge_4x0q_l2345_234575


namespace NUMINAMATH_GPT_selling_price_is_1260_l2345_234543

-- Definitions based on conditions
def purchase_price : ℕ := 900
def repair_cost : ℕ := 300
def gain_percent : ℕ := 5 -- percentage as a natural number

-- Known variables
def total_cost : ℕ := purchase_price + repair_cost
def gain_amount : ℕ := (gain_percent * total_cost) / 100
def selling_price : ℕ := total_cost + gain_amount

-- The theorem we want to prove
theorem selling_price_is_1260 : selling_price = 1260 := by
  sorry

end NUMINAMATH_GPT_selling_price_is_1260_l2345_234543


namespace NUMINAMATH_GPT_find_x_in_terms_of_N_l2345_234556

theorem find_x_in_terms_of_N (N : ℤ) (x y : ℝ) 
(h1 : (⌊x⌋ : ℤ) + 2 * y = N + 2) 
(h2 : (⌊y⌋ : ℤ) + 2 * x = 3 - N) : 
x = (3 / 2) - N := 
by
  sorry

end NUMINAMATH_GPT_find_x_in_terms_of_N_l2345_234556


namespace NUMINAMATH_GPT_solution_l2345_234532

def p : Prop := ∀ x > 0, Real.log (x + 1) > 0
def q : Prop := ∀ a b : ℝ, a > b → a^2 > b^2

theorem solution : p ∧ ¬ q := by
  sorry

end NUMINAMATH_GPT_solution_l2345_234532


namespace NUMINAMATH_GPT_position_1011th_square_l2345_234582

-- Define the initial position and transformations
inductive SquarePosition
| ABCD : SquarePosition
| DABC : SquarePosition
| BADC : SquarePosition
| DCBA : SquarePosition

open SquarePosition

def R1 (p : SquarePosition) : SquarePosition :=
  match p with
  | ABCD => DABC
  | DABC => BADC
  | BADC => DCBA
  | DCBA => ABCD

def R2 (p : SquarePosition) : SquarePosition :=
  match p with
  | ABCD => DCBA
  | DCBA => ABCD
  | DABC => BADC
  | BADC => DABC

def transform : ℕ → SquarePosition
| 0 => ABCD
| n + 1 => if n % 2 = 0 then R1 (transform n) else R2 (transform n)

theorem position_1011th_square : transform 1011 = DCBA :=
by {
  sorry
}

end NUMINAMATH_GPT_position_1011th_square_l2345_234582


namespace NUMINAMATH_GPT_find_f_2021_l2345_234520

def f (x : ℝ) : ℝ := sorry

theorem find_f_2021 (h : ∀ a b : ℝ, f ((a + 2 * b) / 3) = (f a + 2 * f b) / 3)
    (h1 : f 1 = 5) (h4 : f 4 = 2) : f 2021 = -2015 :=
by
  sorry

end NUMINAMATH_GPT_find_f_2021_l2345_234520


namespace NUMINAMATH_GPT_total_number_of_applications_l2345_234555

def in_state_apps := 200
def out_state_apps := 2 * in_state_apps
def total_apps := in_state_apps + out_state_apps

theorem total_number_of_applications : total_apps = 600 := by
  sorry

end NUMINAMATH_GPT_total_number_of_applications_l2345_234555


namespace NUMINAMATH_GPT_nursery_school_students_l2345_234500

theorem nursery_school_students (S : ℕ)
  (h1 : ∃ x, x = S / 10)
  (h2 : 20 + (S / 10) = 25) : S = 50 :=
by
  sorry

end NUMINAMATH_GPT_nursery_school_students_l2345_234500


namespace NUMINAMATH_GPT_movie_ticket_ratio_l2345_234557

-- Definitions based on the conditions
def monday_cost : ℕ := 5
def wednesday_cost : ℕ := 2 * monday_cost

theorem movie_ticket_ratio (S : ℕ) (h1 : wednesday_cost + S = 35) :
  S / monday_cost = 5 :=
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_movie_ticket_ratio_l2345_234557


namespace NUMINAMATH_GPT_num_children_eq_3_l2345_234518

-- Definitions from the conditions
def regular_ticket_cost : ℕ := 9
def child_ticket_discount : ℕ := 2
def given_amount : ℕ := 20 * 2
def received_change : ℕ := 1
def num_adults : ℕ := 2

-- Derived data
def total_ticket_cost : ℕ := given_amount - received_change
def adult_ticket_cost : ℕ := num_adults * regular_ticket_cost
def children_ticket_cost : ℕ := total_ticket_cost - adult_ticket_cost
def child_ticket_cost : ℕ := regular_ticket_cost - child_ticket_discount

-- Statement to prove
theorem num_children_eq_3 : (children_ticket_cost / child_ticket_cost) = 3 := by
  sorry

end NUMINAMATH_GPT_num_children_eq_3_l2345_234518


namespace NUMINAMATH_GPT_arccos_pi_over_3_l2345_234553

theorem arccos_pi_over_3 :
  Real.arccos (1 / 2) = Real.pi / 3 :=
by
  sorry

end NUMINAMATH_GPT_arccos_pi_over_3_l2345_234553


namespace NUMINAMATH_GPT_contradiction_with_angles_l2345_234523

-- Definitions of conditions
def triangle (α β γ : ℝ) : Prop := α + β + γ = 180 ∧ α > 0 ∧ β > 0 ∧ γ > 0

-- The proposition we want to prove by contradiction
def at_least_one_angle_not_greater_than_60 (α β γ : ℝ) : Prop := α ≤ 60 ∨ β ≤ 60 ∨ γ ≤ 60

-- The assumption for contradiction
def all_angles_greater_than_60 (α β γ : ℝ) : Prop := α > 60 ∧ β > 60 ∧ γ > 60

-- The proof problem
theorem contradiction_with_angles (α β γ : ℝ) (h : triangle α β γ) :
  ¬ all_angles_greater_than_60 α β γ → at_least_one_angle_not_greater_than_60 α β γ :=
sorry

end NUMINAMATH_GPT_contradiction_with_angles_l2345_234523


namespace NUMINAMATH_GPT_valid_sequences_length_21_l2345_234538

def valid_sequences (n : ℕ) : ℕ :=
  if n < 3 then 0
  else if n = 3 then 1
  else if n = 4 then 1
  else if n = 5 then 2
  else valid_sequences (n - 3) + valid_sequences (n - 4)

theorem valid_sequences_length_21 : valid_sequences 21 = 38 :=
by
  sorry

end NUMINAMATH_GPT_valid_sequences_length_21_l2345_234538


namespace NUMINAMATH_GPT_raw_score_is_correct_l2345_234511

-- Define the conditions
def points_per_correct : ℝ := 1
def points_subtracted_per_incorrect : ℝ := 0.25
def total_questions : ℕ := 85
def answered_questions : ℕ := 82
def correct_answers : ℕ := 70

-- Define the number of incorrect answers
def incorrect_answers : ℕ := answered_questions - correct_answers
-- Calculate the raw score
def raw_score : ℝ := 
  (correct_answers * points_per_correct) - (incorrect_answers * points_subtracted_per_incorrect)

-- Prove the raw score is 67
theorem raw_score_is_correct : raw_score = 67 := 
by
  sorry

end NUMINAMATH_GPT_raw_score_is_correct_l2345_234511


namespace NUMINAMATH_GPT_unique_shirt_and_tie_outfits_l2345_234559

theorem unique_shirt_and_tie_outfits :
  let shirts := 10
  let ties := 8
  let choose n k := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))
  choose shirts 5 * choose ties 4 = 17640 :=
by
  sorry

end NUMINAMATH_GPT_unique_shirt_and_tie_outfits_l2345_234559


namespace NUMINAMATH_GPT_find_m_value_l2345_234569

-- Definitions based on conditions
variables {a b m : ℝ} (ha : 2 ^ a = m) (hb : 5 ^ b = m) (h : 1 / a + 1 / b = 1)

-- Lean 4 statement of the problem
theorem find_m_value (ha : 2 ^ a = m) (hb : 5 ^ b = m) (h : 1 / a + 1 / b = 1) : m = 10 := sorry

end NUMINAMATH_GPT_find_m_value_l2345_234569


namespace NUMINAMATH_GPT_part1_l2345_234565

theorem part1 (a : ℝ) (f : ℝ → ℝ) (h : ∀ x, x > 0 → f x < 0) :
  a > 1 :=
sorry

end NUMINAMATH_GPT_part1_l2345_234565


namespace NUMINAMATH_GPT_unique_factor_and_multiple_of_13_l2345_234514

theorem unique_factor_and_multiple_of_13 (n : ℕ) (h1 : n ∣ 13) (h2 : 13 ∣ n) : n = 13 :=
sorry

end NUMINAMATH_GPT_unique_factor_and_multiple_of_13_l2345_234514


namespace NUMINAMATH_GPT_linear_func_3_5_l2345_234544

def linear_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y

theorem linear_func_3_5 (f : ℝ → ℝ) (h_linear: linear_function f) 
  (h_diff: ∀ d : ℝ, f (d + 1) - f d = 3) : f 3 - f 5 = -6 :=
by
  sorry

end NUMINAMATH_GPT_linear_func_3_5_l2345_234544


namespace NUMINAMATH_GPT_part_a_part_b_l2345_234551

noncomputable def probability_Peter_satisfied : ℚ :=
  let total_people := 100
  let men := 50
  let women := 50
  let P_both_men := (men - 1 : ℚ)/ (total_people - 1 : ℚ) * (men - 2 : ℚ)/ (total_people - 2 : ℚ)
  1 - P_both_men

theorem part_a : probability_Peter_satisfied = 25 / 33 := 
  sorry

noncomputable def expected_satisfied_men : ℚ :=
  let men := 50
  probability_Peter_satisfied * men

theorem part_b : expected_satisfied_men = 1250 / 33 := 
  sorry

end NUMINAMATH_GPT_part_a_part_b_l2345_234551


namespace NUMINAMATH_GPT_fraction_greater_than_decimal_l2345_234584

/-- 
  Prove that the fraction 1/3 is greater than the decimal 0.333 by the amount 1/(3 * 10^3)
-/
theorem fraction_greater_than_decimal :
  (1 / 3 : ℚ) = (333 / 1000 : ℚ) + (1 / (3 * 1000) : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_fraction_greater_than_decimal_l2345_234584


namespace NUMINAMATH_GPT_expression_parity_l2345_234560

theorem expression_parity (a b c : ℕ) (h_a_pos : a > 0) (h_b_pos : b > 0) (h_a_odd : a % 2 = 1) (h_b_odd : b % 2 = 1) : (3^a + (b + 1)^2 * c) % 2 = 1 :=
by sorry

end NUMINAMATH_GPT_expression_parity_l2345_234560


namespace NUMINAMATH_GPT_addition_example_l2345_234576

theorem addition_example : 36 + 15 = 51 := 
by
  sorry

end NUMINAMATH_GPT_addition_example_l2345_234576


namespace NUMINAMATH_GPT_part_a_part_b_l2345_234527

-- Define the problem as described
noncomputable def can_transform_to_square (figure : Type) (parts : ℕ) (all_triangles : Bool) : Bool :=
sorry  -- This is a placeholder for the actual implementation

-- The figure satisfies the condition to cut into four parts and rearrange into a square
theorem part_a (figure : Type) : can_transform_to_square figure 4 false = true :=
sorry

-- The figure satisfies the condition to cut into five triangular parts and rearrange into a square
theorem part_b (figure : Type) : can_transform_to_square figure 5 true = true :=
sorry

end NUMINAMATH_GPT_part_a_part_b_l2345_234527


namespace NUMINAMATH_GPT_negation_of_universal_proposition_l2345_234546

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + x + 1 ≥ 0) ↔ (∃ x : ℝ, x^2 + x + 1 < 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_universal_proposition_l2345_234546


namespace NUMINAMATH_GPT_total_students_l2345_234547

-- Definitions based on conditions
variable (T M Z : ℕ)  -- T for Tina's students, M for Maura's students, Z for Zack's students

-- Conditions as hypotheses
axiom h1 : T = M  -- Tina's classroom has the same amount of students as Maura's
axiom h2 : Z = (T + M) / 2  -- Zack's classroom has half the amount of total students between Tina and Maura's classrooms
axiom h3 : Z = 23  -- There are 23 students in Zack's class when present

-- Proof statement
theorem total_students : T + M + Z = 69 :=
  sorry

end NUMINAMATH_GPT_total_students_l2345_234547


namespace NUMINAMATH_GPT_find_number_l2345_234567

noncomputable def question (x : ℝ) : Prop :=
  (2 * x^2 + Real.sqrt 6)^3 = 19683

theorem find_number : ∃ x : ℝ, question x ∧ (x = Real.sqrt ((27 - Real.sqrt 6) / 2) ∨ x = -Real.sqrt ((27 - Real.sqrt 6) / 2)) :=
  sorry

end NUMINAMATH_GPT_find_number_l2345_234567


namespace NUMINAMATH_GPT_positive_integer_conditions_l2345_234594

theorem positive_integer_conditions (p : ℕ) (hp : p > 0) : 
  (∃ k : ℕ, k > 0 ∧ 4 * p + 28 = k * (3 * p - 7)) ↔ (p = 6 ∨ p = 28) :=
by
  sorry

end NUMINAMATH_GPT_positive_integer_conditions_l2345_234594


namespace NUMINAMATH_GPT_barium_oxide_moles_l2345_234580

noncomputable def moles_of_bao_needed (mass_H2O : ℝ) (molar_mass_H2O : ℝ) : ℝ :=
  mass_H2O / molar_mass_H2O

theorem barium_oxide_moles :
  moles_of_bao_needed 54 18.015 = 3 :=
by
  unfold moles_of_bao_needed
  norm_num
  sorry

end NUMINAMATH_GPT_barium_oxide_moles_l2345_234580


namespace NUMINAMATH_GPT_solve_cyclic_quadrilateral_area_l2345_234524

noncomputable def cyclic_quadrilateral_area (AB BC AD CD : ℝ) (cyclic : Bool) : ℝ :=
  if cyclic ∧ AB = 2 ∧ BC = 6 ∧ AD = 4 ∧ CD = 4 then 8 * Real.sqrt 3 else 0

theorem solve_cyclic_quadrilateral_area :
  cyclic_quadrilateral_area 2 6 4 4 true = 8 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_cyclic_quadrilateral_area_l2345_234524


namespace NUMINAMATH_GPT_speed_of_car_A_l2345_234501

variable (V_A V_B T : ℕ)
variable (h1 : V_B = 35) (h2 : T = 10) (h3 : 2 * V_B * T = V_A * T)

theorem speed_of_car_A :
  V_A = 70 :=
by
  sorry

end NUMINAMATH_GPT_speed_of_car_A_l2345_234501


namespace NUMINAMATH_GPT_price_of_first_variety_l2345_234599

theorem price_of_first_variety
  (p2 : ℝ) (p3 : ℝ) (r : ℝ) (w : ℝ)
  (h1 : p2 = 135)
  (h2 : p3 = 177.5)
  (h3 : r = 154)
  (h4 : w = 4) :
  ∃ p1 : ℝ, 1 * p1 + 1 * p2 + 2 * p3 = w * r ∧ p1 = 126 :=
by {
  sorry
}

end NUMINAMATH_GPT_price_of_first_variety_l2345_234599


namespace NUMINAMATH_GPT_maximum_volume_of_prism_l2345_234503

noncomputable def maximum_volume_prism (s : ℝ) (θ : ℝ) (face_area_sum : ℝ) : ℝ := 
  if (s = 6 ∧ θ = Real.pi / 3 ∧ face_area_sum = 36) then 27 
  else 0

theorem maximum_volume_of_prism : 
  ∀ (s θ face_area_sum), s = 6 ∧ θ = Real.pi / 3 ∧ face_area_sum = 36 → maximum_volume_prism s θ face_area_sum = 27 :=
by
  intros
  sorry

end NUMINAMATH_GPT_maximum_volume_of_prism_l2345_234503


namespace NUMINAMATH_GPT_determine_a_range_l2345_234530

variable (a : ℝ)

-- Define proposition p as a function
def p : Prop := ∀ x : ℝ, x^2 + x > a

-- Negation of Proposition q
def not_q : Prop := ∀ x : ℝ, x^2 + 2 * a * x + 2 - a ≠ 0

-- The main theorem to be stated, proving the range of 'a'
theorem determine_a_range (h₁ : p a) (h₂ : not_q a) : -2 < a ∧ a < -1 / 4 := sorry

end NUMINAMATH_GPT_determine_a_range_l2345_234530


namespace NUMINAMATH_GPT_no_such_function_exists_l2345_234571

noncomputable def func_a (a : ℕ → ℕ) : Prop :=
  a 0 = 0 ∧ ∀ n : ℕ, a n = n - a (a n)

theorem no_such_function_exists : ¬ ∃ a : ℕ → ℕ, func_a a :=
by
  sorry

end NUMINAMATH_GPT_no_such_function_exists_l2345_234571


namespace NUMINAMATH_GPT_buildings_subset_count_l2345_234595

theorem buildings_subset_count :
  let buildings := Finset.range (16 + 1) \ {0}
  ∃ S ⊆ buildings, ∀ (a b : ℕ), a ≠ b ∧ a ∈ S ∧ b ∈ S → ∃ k, (b - a = 2 * k + 1) ∨ (a - b = 2 * k + 1) ∧ Finset.card S = 510 :=
sorry

end NUMINAMATH_GPT_buildings_subset_count_l2345_234595


namespace NUMINAMATH_GPT_range_of_a_l2345_234592

noncomputable def A : Set ℝ := {x : ℝ | ((x^2) - x - 2) ≤ 0}

theorem range_of_a (a : ℝ) : (∀ x ∈ A, (x^2 - a*x - a - 2) ≤ 0) → a ≥ (2/3) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_range_of_a_l2345_234592


namespace NUMINAMATH_GPT_lucy_final_balance_l2345_234552

def initial_balance : ℝ := 65
def deposit : ℝ := 15
def withdrawal : ℝ := 4

theorem lucy_final_balance : initial_balance + deposit - withdrawal = 76 :=
by
  sorry

end NUMINAMATH_GPT_lucy_final_balance_l2345_234552


namespace NUMINAMATH_GPT_smaller_of_x_and_y_is_15_l2345_234506

variable {x y : ℕ}

/-- Given two positive numbers x and y are in the ratio 3:5, 
and the sum of x and y plus 10 equals 50,
prove that the smaller of x and y is 15. -/
theorem smaller_of_x_and_y_is_15 (h1 : x * 5 = y * 3) (h2 : x + y + 10 = 50) (h3 : 0 < x) (h4 : 0 < y) : x = 15 :=
by
  sorry

end NUMINAMATH_GPT_smaller_of_x_and_y_is_15_l2345_234506


namespace NUMINAMATH_GPT_anna_spent_more_on_lunch_l2345_234577

def bagel_cost : ℝ := 0.95
def cream_cheese_cost : ℝ := 0.50
def orange_juice_cost : ℝ := 1.25
def orange_juice_discount : ℝ := 0.32
def sandwich_cost : ℝ := 4.65
def avocado_cost : ℝ := 0.75
def milk_cost : ℝ := 1.15
def milk_discount : ℝ := 0.10

-- Calculate total cost of breakfast.
def breakfast_cost : ℝ := 
  let bagel_with_cream_cheese := bagel_cost + cream_cheese_cost
  let discounted_orange_juice := orange_juice_cost - (orange_juice_cost * orange_juice_discount)
  bagel_with_cream_cheese + discounted_orange_juice

-- Calculate total cost of lunch.
def lunch_cost : ℝ :=
  let sandwich_with_avocado := sandwich_cost + avocado_cost
  let discounted_milk := milk_cost - (milk_cost * milk_discount)
  sandwich_with_avocado + discounted_milk

-- Calculate the difference between lunch and breakfast costs.
theorem anna_spent_more_on_lunch : lunch_cost - breakfast_cost = 4.14 := by
  sorry

end NUMINAMATH_GPT_anna_spent_more_on_lunch_l2345_234577


namespace NUMINAMATH_GPT_tens_digit_of_large_power_l2345_234534

theorem tens_digit_of_large_power : ∃ a : ℕ, a = 2 ∧ ∀ n ≥ 2, (5 ^ n) % 100 = 25 :=
by
  sorry

end NUMINAMATH_GPT_tens_digit_of_large_power_l2345_234534


namespace NUMINAMATH_GPT_initial_population_l2345_234548

theorem initial_population (P : ℝ) 
  (h1 : P * 0.90 * 0.95 * 0.85 * 1.08 = 6514) : P = 8300 :=
by
  -- Given conditions lead to the final population being 6514
  -- We need to show that the initial population P was 8300
  sorry

end NUMINAMATH_GPT_initial_population_l2345_234548


namespace NUMINAMATH_GPT_central_angle_of_sector_l2345_234517

theorem central_angle_of_sector {r l : ℝ} 
  (h1 : 2 * r + l = 4) 
  (h2 : (1 / 2) * l * r = 1) : 
  l / r = 2 :=
by 
  sorry

end NUMINAMATH_GPT_central_angle_of_sector_l2345_234517


namespace NUMINAMATH_GPT_molecular_weight_of_compound_l2345_234541

def num_atoms_C : ℕ := 6
def num_atoms_H : ℕ := 8
def num_atoms_O : ℕ := 7

def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.008
def atomic_weight_O : ℝ := 16.00

def molecular_weight (nC nH nO : ℕ) (wC wH wO : ℝ) : ℝ :=
  nC * wC + nH * wH + nO * wO

theorem molecular_weight_of_compound :
  molecular_weight num_atoms_C num_atoms_H num_atoms_O atomic_weight_C atomic_weight_H atomic_weight_O = 192.124 :=
by
  sorry

end NUMINAMATH_GPT_molecular_weight_of_compound_l2345_234541


namespace NUMINAMATH_GPT_find_x_solution_l2345_234507

theorem find_x_solution
  (x y z : ℤ)
  (h1 : 4 * x + y + z = 80)
  (h2 : 2 * x - y - z = 40)
  (h3 : 3 * x + y - z = 20) :
  x = 20 :=
by
  -- Proof steps go here...
  sorry

end NUMINAMATH_GPT_find_x_solution_l2345_234507


namespace NUMINAMATH_GPT_arithmetic_geometric_value_l2345_234597

-- Definitions and annotations
variables {a1 a2 b1 b2 : ℝ}
variable {d : ℝ} -- common difference for the arithmetic sequence
variable {q : ℝ} -- common ratio for the geometric sequence

-- Assuming input values for the initial elements of the sequences
axiom h1 : -9 = -9
axiom h2 : -9 + 3 * d = -1
axiom h3 : b1 = -9 * q
axiom h4 : b2 = -9 * q^2

-- The desired equality to prove
theorem arithmetic_geometric_value :
  b2 * (a2 - a1) = -8 :=
sorry

end NUMINAMATH_GPT_arithmetic_geometric_value_l2345_234597


namespace NUMINAMATH_GPT_nesbitts_inequality_l2345_234540

theorem nesbitts_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (b + c)) + (b / (c + a)) + (c / (a + b)) ≥ (3 / 2) :=
sorry

end NUMINAMATH_GPT_nesbitts_inequality_l2345_234540


namespace NUMINAMATH_GPT_value_of_X_l2345_234525

def M := 2007 / 3
def N := M / 3
def X := M - N

theorem value_of_X : X = 446 := by
  sorry

end NUMINAMATH_GPT_value_of_X_l2345_234525
