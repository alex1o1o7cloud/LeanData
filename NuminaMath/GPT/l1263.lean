import Mathlib

namespace NUMINAMATH_GPT_equivalent_statements_l1263_126317

variable (P Q : Prop)

theorem equivalent_statements :
  (P → Q) ↔ ((¬ Q → ¬ P) ∧ (¬ P ∨ Q)) :=
by
  sorry

end NUMINAMATH_GPT_equivalent_statements_l1263_126317


namespace NUMINAMATH_GPT_students_in_class_l1263_126345

theorem students_in_class (x : ℕ) (S : ℕ)
  (h1 : S = 3 * (S / x) + 24)
  (h2 : S = 4 * (S / x) - 26) : 3 * x + 24 = 4 * x - 26 :=
by
  sorry

end NUMINAMATH_GPT_students_in_class_l1263_126345


namespace NUMINAMATH_GPT_average_marks_class_l1263_126340

theorem average_marks_class (total_students : ℕ)
  (students_98 : ℕ) (score_98 : ℕ)
  (students_0 : ℕ) (score_0 : ℕ)
  (remaining_avg : ℝ)
  (h1 : total_students = 40)
  (h2 : students_98 = 6)
  (h3 : score_98 = 98)
  (h4 : students_0 = 9)
  (h5 : score_0 = 0)
  (h6 : remaining_avg = 57) :
  ( (( students_98 * score_98) + (students_0 * score_0) + ((total_students - students_98 - students_0) * remaining_avg)) / total_students ) = 50.325 :=
by 
  -- This is where the proof steps would go
  sorry

end NUMINAMATH_GPT_average_marks_class_l1263_126340


namespace NUMINAMATH_GPT_shortest_player_height_correct_l1263_126328

def tallest_player_height : Real := 77.75
def height_difference : Real := 9.5
def shortest_player_height : Real := 68.25

theorem shortest_player_height_correct :
  tallest_player_height - height_difference = shortest_player_height :=
by
  sorry

end NUMINAMATH_GPT_shortest_player_height_correct_l1263_126328


namespace NUMINAMATH_GPT_tank_capacity_l1263_126371

theorem tank_capacity (C : ℝ) : 
  (0.5 * C = 0.9 * C - 45) → C = 112.5 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_tank_capacity_l1263_126371


namespace NUMINAMATH_GPT_equivalent_fractions_l1263_126320

variable {x y a c : ℝ}

theorem equivalent_fractions (h_nonzero_c : c ≠ 0) (h_transform : x = (a / c) * y) :
  (x + a) / (y + c) = a / c :=
by
  sorry

end NUMINAMATH_GPT_equivalent_fractions_l1263_126320


namespace NUMINAMATH_GPT_cistern_water_depth_l1263_126324

theorem cistern_water_depth
  (length width : ℝ) 
  (wet_surface_area : ℝ)
  (h : ℝ) 
  (hl : length = 7)
  (hw : width = 4)
  (ha : wet_surface_area = 55.5)
  (h_eq : 28 + 22 * h = wet_surface_area) 
  : h = 1.25 := 
  by 
  sorry

end NUMINAMATH_GPT_cistern_water_depth_l1263_126324


namespace NUMINAMATH_GPT_vector_parallel_eq_l1263_126390

theorem vector_parallel_eq (k : ℝ) (a b : ℝ × ℝ) 
  (h_a : a = (k, 2)) (h_b : b = (1, 1)) (h_parallel : (∃ c : ℝ, a = (c * 1, c * 1))) : k = 2 := by
  sorry

end NUMINAMATH_GPT_vector_parallel_eq_l1263_126390


namespace NUMINAMATH_GPT_smallest_invariant_number_l1263_126336

def operation (n : ℕ) : ℕ :=
  let q := n / 10
  let r := n % 10
  q + 2 * r

def is_invariant (n : ℕ) : Prop :=
  operation n = n

theorem smallest_invariant_number : ∃ n : ℕ, is_invariant n ∧ n = 10^99 + 1 :=
by
  sorry

end NUMINAMATH_GPT_smallest_invariant_number_l1263_126336


namespace NUMINAMATH_GPT_find_a_b_find_max_m_l1263_126330

-- Define the function
def f (x : ℝ) : ℝ := abs (2 * x + 1) + abs (3 * x - 2)

-- Conditions
def solution_set_condition (x a : ℝ) : Prop := (-4 * a / 5 ≤ x ∧ x ≤ 3 * a / 5)
def eq_five_condition (x : ℝ) : Prop := f x ≤ 5

-- Prove the values of a and b
theorem find_a_b (a b : ℝ) : (∀ x : ℝ, eq_five_condition x ↔ solution_set_condition x a) → (a = 1 ∧ b = 2) :=
by
  sorry

-- Prove that |x - a| + |x + b| >= m^2 - 3m and find the maximum value of m
theorem find_max_m (a b m : ℝ) : (a = 1 ∧ b = 2) →
  (∀ x : ℝ, abs (x - a) + abs (x + b) ≥ m^2 - 3 * m) →
  m ≤ (3 + Real.sqrt 21) / 2 :=
by
  sorry


end NUMINAMATH_GPT_find_a_b_find_max_m_l1263_126330


namespace NUMINAMATH_GPT_john_payment_and_hourly_rate_l1263_126383

variable (court_hours : ℕ) (prep_hours : ℕ) (upfront_fee : ℕ) 
variable (total_payment : ℕ) (brother_contribution_factor : ℕ)
variable (hourly_rate : ℚ) (john_payment : ℚ)

axiom condition1 : upfront_fee = 1000
axiom condition2 : court_hours = 50
axiom condition3 : prep_hours = 2 * court_hours
axiom condition4 : total_payment = 8000
axiom condition5 : brother_contribution_factor = 2

theorem john_payment_and_hourly_rate :
  (john_payment = total_payment / brother_contribution_factor + upfront_fee) ∧
  (hourly_rate = (total_payment - upfront_fee) / (court_hours + prep_hours)) :=
by
  sorry

end NUMINAMATH_GPT_john_payment_and_hourly_rate_l1263_126383


namespace NUMINAMATH_GPT_find_P_l1263_126348

variable (a b c d P : ℝ)

theorem find_P 
  (h1 : (a + b + c + d) / 4 = 8) 
  (h2 : (a + b + c + d + P) / 5 = P) : 
  P = 8 := 
by 
  sorry

end NUMINAMATH_GPT_find_P_l1263_126348


namespace NUMINAMATH_GPT_kids_on_Monday_l1263_126310

-- Defining the conditions
def kidsOnTuesday : ℕ := 10
def difference : ℕ := 8

-- Formulating the theorem to prove the number of kids Julia played with on Monday
theorem kids_on_Monday : kidsOnTuesday + difference = 18 := by
  sorry

end NUMINAMATH_GPT_kids_on_Monday_l1263_126310


namespace NUMINAMATH_GPT_problem_I_solution_set_l1263_126318

def f1 (x : ℝ) : ℝ := |2 * x| + |x - 1| -- since a = -1

theorem problem_I_solution_set :
  {x : ℝ | f1 x ≤ 4} = Set.Icc (-1 : ℝ) ((5 : ℝ) / 3) :=
sorry

end NUMINAMATH_GPT_problem_I_solution_set_l1263_126318


namespace NUMINAMATH_GPT_reflect_triangle_final_position_l1263_126354

variables {x1 x2 x3 y1 y2 y3 : ℝ}

-- Definition of reflection in x-axis and y-axis
def reflect_x (x y : ℝ) : ℝ × ℝ := (x, -y)
def reflect_y (x y : ℝ) : ℝ × ℝ := (-x, y)

theorem reflect_triangle_final_position (x1 x2 x3 y1 y2 y3 : ℝ) :
  (reflect_y (reflect_x x1 y1).1 (reflect_x x1 y1).2) = (-x1, -y1) ∧
  (reflect_y (reflect_x x2 y2).1 (reflect_x x2 y2).2) = (-x2, -y2) ∧
  (reflect_y (reflect_x x3 y3).1 (reflect_x x3 y3).2) = (-x3, -y3) :=
by
  sorry

end NUMINAMATH_GPT_reflect_triangle_final_position_l1263_126354


namespace NUMINAMATH_GPT_original_purchase_price_first_commodity_l1263_126353

theorem original_purchase_price_first_commodity (x y : ℝ) 
  (h1 : 1.07 * (x + y) = 827) 
  (h2 : x = y + 127) : 
  x = 450.415 :=
  sorry

end NUMINAMATH_GPT_original_purchase_price_first_commodity_l1263_126353


namespace NUMINAMATH_GPT_real_solutions_quadratic_l1263_126366

theorem real_solutions_quadratic (a : ℝ) : 
  (∃ x : ℝ, x^2 - 4 * x + a = 0) ↔ a ≤ 4 :=
by sorry

end NUMINAMATH_GPT_real_solutions_quadratic_l1263_126366


namespace NUMINAMATH_GPT_vector_subtraction_result_l1263_126349

-- definition of vectors as pairs of integers
def OA : ℝ × ℝ := (1, -2)
def OB : ℝ × ℝ := (-3, 1)

-- definition of vector subtraction for pairs of reals
def vector_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 - v2.1, v1.2 - v2.2)

-- definition of the vector AB as the subtraction of OB and OA
def AB : ℝ × ℝ := vector_sub OB OA

-- statement to assert the expected result
theorem vector_subtraction_result : AB = (-4, 3) :=
by
  -- this is where the proof would go, but we use sorry to skip it
  sorry

end NUMINAMATH_GPT_vector_subtraction_result_l1263_126349


namespace NUMINAMATH_GPT_petya_numbers_l1263_126302

-- Define the arithmetic sequence property
def arithmetic_seq (a d : ℕ) : ℕ → ℕ
| 0     => a
| (n+1) => a + (n + 1) * d

-- Given conditions
theorem petya_numbers (a d : ℕ) : 
  (arithmetic_seq a d 0 = 6) ∧
  (arithmetic_seq a d 1 = 15) ∧
  (arithmetic_seq a d 2 = 24) ∧
  (arithmetic_seq a d 3 = 33) ∧
  (arithmetic_seq a d 4 = 42) :=
sorry

end NUMINAMATH_GPT_petya_numbers_l1263_126302


namespace NUMINAMATH_GPT_geometric_sequence_sum_l1263_126396

variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)

-- Conditions
axiom a1 : (a 1) = 1
axiom a2 : ∀ (n : ℕ), n ≥ 2 → 2 * a (n + 1) + 2 * a (n - 1) = 5 * a n
axiom increasing : ∀ (n m : ℕ), n < m → a n < a m

-- Target
theorem geometric_sequence_sum : S 5 = 31 := by
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l1263_126396


namespace NUMINAMATH_GPT_exists_ten_positive_integers_l1263_126367

theorem exists_ten_positive_integers :
  ∃ (a : ℕ → ℕ), (∀ i j, i ≠ j → ¬ (a i ∣ a j))
  ∧ (∀ i j, (a i)^2 ∣ a j) :=
sorry

end NUMINAMATH_GPT_exists_ten_positive_integers_l1263_126367


namespace NUMINAMATH_GPT_Wendy_total_glasses_l1263_126378

theorem Wendy_total_glasses (small large : ℕ)
  (h1 : small = 50)
  (h2 : large = small + 10) :
  small + large = 110 :=
by
  sorry

end NUMINAMATH_GPT_Wendy_total_glasses_l1263_126378


namespace NUMINAMATH_GPT_domain_sqrt_quot_l1263_126316

noncomputable def domain_of_function (f : ℝ → ℝ) : Set ℝ := {x : ℝ | f x ≠ 0}

theorem domain_sqrt_quot (x : ℝ) :
  (x + 1 ≥ 0) ∧ (x ≠ 0) ↔ (x ∈ {x : ℝ | -1 ≤ x ∧ x < 0} ∪ {x : ℝ | x > 0}) :=
by
  sorry

end NUMINAMATH_GPT_domain_sqrt_quot_l1263_126316


namespace NUMINAMATH_GPT_nathan_banana_payment_l1263_126395

theorem nathan_banana_payment
  (bunches_8 : ℕ)
  (cost_per_bunch_8 : ℝ)
  (bunches_7 : ℕ)
  (cost_per_bunch_7 : ℝ)
  (discount : ℝ)
  (total_payment : ℝ) :
  bunches_8 = 6 →
  cost_per_bunch_8 = 2.5 →
  bunches_7 = 5 →
  cost_per_bunch_7 = 2.2 →
  discount = 0.10 →
  total_payment = 6 * 2.5 + 5 * 2.2 - 0.10 * (6 * 2.5 + 5 * 2.2) →
  total_payment = 23.40 :=
by
  intros
  sorry

end NUMINAMATH_GPT_nathan_banana_payment_l1263_126395


namespace NUMINAMATH_GPT_all_faces_rhombuses_l1263_126388

variable {R : Type} [LinearOrderedCommRing R]

structure Parallelepiped (R : Type) :=
  (a b c : R)

def parallelogram_area {R : Type} [LinearOrderedCommRing R] (x y : R) : R :=
  x * y

def is_rhombus (x y : R) : Prop :=
  x = y

theorem all_faces_rhombuses (P : Parallelepiped R)
  (h1: parallelogram_area P.a P.b = parallelogram_area P.b P.c)
  (h2: parallelogram_area P.b P.c = parallelogram_area P.a P.c)
  (h3: parallelogram_area P.a P.b = parallelogram_area P.a P.c) :
  is_rhombus P.a P.b ∧ is_rhombus P.b P.c ∧ is_rhombus P.a P.c :=
  sorry

end NUMINAMATH_GPT_all_faces_rhombuses_l1263_126388


namespace NUMINAMATH_GPT_Alan_finish_time_third_task_l1263_126398

theorem Alan_finish_time_third_task :
  let start_time := 480 -- 8:00 AM in minutes from midnight
  let finish_time_second_task := 675 -- 11:15 AM in minutes from midnight
  let total_tasks_time := 195 -- Total time spent on first two tasks
  let first_task_time := 65 -- Time taken for the first task calculated as per the solution
  let second_task_time := 130 -- Time taken for the second task calculated as per the solution
  let third_task_time := 65 -- Time taken for the third task
  let finish_time_third_task := 740 -- 12:20 PM in minutes from midnight
  start_time + total_tasks_time + third_task_time = finish_time_third_task :=
by
  -- proof here
  sorry

end NUMINAMATH_GPT_Alan_finish_time_third_task_l1263_126398


namespace NUMINAMATH_GPT_problem_statement_l1263_126365

variables (a b c : ℝ)

-- The conditions given in the problem
def condition1 : Prop := a^2 + b^2 - 4 * a ≤ 1
def condition2 : Prop := b^2 + c^2 - 8 * b ≤ -3
def condition3 : Prop := c^2 + a^2 - 12 * c ≤ -26

-- The theorem we need to prove
theorem problem_statement (h1 : condition1 a b) (h2 : condition2 b c) (h3 : condition3 c a) : (a + b) ^ c = 27 :=
by sorry

end NUMINAMATH_GPT_problem_statement_l1263_126365


namespace NUMINAMATH_GPT_larry_final_channels_l1263_126359

def initial_channels : Int := 150
def removed_channels : Int := 20
def replacement_channels : Int := 12
def reduced_channels : Int := 10
def sports_package_channels : Int := 8
def supreme_sports_package_channels : Int := 7

theorem larry_final_channels :
  initial_channels 
  - removed_channels 
  + replacement_channels 
  - reduced_channels 
  + sports_package_channels 
  + supreme_sports_package_channels 
  = 147 := by
  rfl  -- Reflects the direct computation as per the problem

end NUMINAMATH_GPT_larry_final_channels_l1263_126359


namespace NUMINAMATH_GPT_a_2019_value_l1263_126303

noncomputable def a_sequence (n : ℕ) : ℝ :=
  if n = 0 then 0  -- not used, a_0 is irrelevant
  else if n = 1 then 1 / 2
  else a_sequence (n - 1) + 1 / (2 ^ (n - 1))

theorem a_2019_value :
  a_sequence 2019 = 3 / 2 - 1 / (2 ^ 2018) :=
by
  sorry

end NUMINAMATH_GPT_a_2019_value_l1263_126303


namespace NUMINAMATH_GPT_difference_in_pups_l1263_126399

theorem difference_in_pups :
  let huskies := 5
  let pitbulls := 2
  let golden_retrievers := 4
  let pups_per_husky := 3
  let pups_per_pitbull := 3
  let total_adults := huskies + pitbulls + golden_retrievers
  let total_pups := total_adults + 30
  let total_husky_pups := huskies * pups_per_husky
  let total_pitbull_pups := pitbulls * pups_per_pitbull
  let H := pups_per_husky
  let D := (total_pups - total_husky_pups - total_pitbull_pups - 3 * golden_retrievers) / golden_retrievers
  D = 2 := sorry

end NUMINAMATH_GPT_difference_in_pups_l1263_126399


namespace NUMINAMATH_GPT_interest_earned_l1263_126308

theorem interest_earned :
  let P : ℝ := 1500
  let r : ℝ := 0.02
  let n : ℕ := 3
  let A : ℝ := P * (1 + r) ^ n
  let interest : ℝ := A - P
  interest = 92 := 
by
  sorry

end NUMINAMATH_GPT_interest_earned_l1263_126308


namespace NUMINAMATH_GPT_problem_statement_l1263_126362

noncomputable def f (x : ℝ) : ℝ := (1 + x) / (2 - x)

noncomputable def f_iter : ℕ → ℝ → ℝ
| 0, x => x
| n + 1, x => f (f_iter n x)

variable (x : ℝ)

theorem problem_statement
  (h : f_iter 13 x = f_iter 31 x) :
  f_iter 16 x = (x - 1) / x :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1263_126362


namespace NUMINAMATH_GPT_calc_g_x_plus_2_minus_g_x_l1263_126375

def g (x : ℝ) : ℝ := 3 * x^2 + 5 * x + 4

theorem calc_g_x_plus_2_minus_g_x (x : ℝ) : g (x + 2) - g x = 12 * x + 22 := 
by 
  sorry

end NUMINAMATH_GPT_calc_g_x_plus_2_minus_g_x_l1263_126375


namespace NUMINAMATH_GPT_tagged_fish_ratio_l1263_126322

theorem tagged_fish_ratio (tagged_first_catch : ℕ) (total_second_catch : ℕ) (tagged_second_catch : ℕ) 
  (approx_total_fish : ℕ) (h1 : tagged_first_catch = 60) 
  (h2 : total_second_catch = 50) 
  (h3 : tagged_second_catch = 2) 
  (h4 : approx_total_fish = 1500) :
  tagged_second_catch / total_second_catch = 1 / 25 := by
  sorry

end NUMINAMATH_GPT_tagged_fish_ratio_l1263_126322


namespace NUMINAMATH_GPT_burritos_in_each_box_l1263_126311

theorem burritos_in_each_box (B : ℕ) (h1 : 3 * B - B - 30 = 10) : B = 20 :=
by
  sorry

end NUMINAMATH_GPT_burritos_in_each_box_l1263_126311


namespace NUMINAMATH_GPT_cyclic_sum_inequality_l1263_126347

theorem cyclic_sum_inequality (x y z : ℝ) (hp : x > 0 ∧ y > 0 ∧ z > 0) (h : x + y + z = 3) : 
  (y^2 * z^2 + z^2 * x^2 + x^2 * y^2) < (3 + x * y + y * z + z * x) := by
  sorry

end NUMINAMATH_GPT_cyclic_sum_inequality_l1263_126347


namespace NUMINAMATH_GPT_max_elevation_l1263_126305

def particle_elevation (t : ℝ) : ℝ := 200 * t - 20 * t^2 + 50

theorem max_elevation : ∃ t : ℝ, particle_elevation t = 550 :=
by {
  sorry
}

end NUMINAMATH_GPT_max_elevation_l1263_126305


namespace NUMINAMATH_GPT_average_visitors_on_sundays_l1263_126301

theorem average_visitors_on_sundays 
  (avg_other_days : ℕ) (avg_per_day : ℕ) (days_in_month : ℕ) (sundays : ℕ) (S : ℕ)
  (h_avg_other_days : avg_other_days = 240)
  (h_avg_per_day : avg_per_day = 310)
  (h_days_in_month : days_in_month = 30)
  (h_sundays : sundays = 5) :
  (sundays * S + (days_in_month - sundays) * avg_other_days = avg_per_day * days_in_month) → 
  S = 660 :=
by
  intros h
  rw [h_avg_other_days, h_avg_per_day, h_days_in_month, h_sundays] at h
  sorry

end NUMINAMATH_GPT_average_visitors_on_sundays_l1263_126301


namespace NUMINAMATH_GPT_percent_games_lost_l1263_126325

def games_ratio (won lost : ℕ) : Prop :=
  won * 3 = lost * 7

def total_games (won lost : ℕ) : Prop :=
  won + lost = 50

def percentage_lost (lost total : ℕ) : ℕ :=
  lost * 100 / total

theorem percent_games_lost (won lost : ℕ) (h1 : games_ratio won lost) (h2 : total_games won lost) : 
  percentage_lost lost 50 = 30 := 
by
  sorry

end NUMINAMATH_GPT_percent_games_lost_l1263_126325


namespace NUMINAMATH_GPT_simplify_expression_1_simplify_expression_2_l1263_126391

-- Problem 1
theorem simplify_expression_1 (a b : ℤ) : a + 2 * b + 3 * a - 2 * b = 4 * a :=
by
  sorry

-- Problem 2
theorem simplify_expression_2 (m n : ℤ) (h_m : m = 2) (h_n : n = 1) :
  (2 * m ^ 2 - 3 * m * n + 8) - (5 * m * n - 4 * m ^ 2 + 8) = 8 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_1_simplify_expression_2_l1263_126391


namespace NUMINAMATH_GPT_problem_statement_l1263_126334

noncomputable def percent_of_y (y : ℝ) (z : ℂ) : ℝ :=
  ((6 * y + 3 * z * Complex.I) / 20 + (3 * y + 4 * z * Complex.I) / 10).re

theorem problem_statement (y : ℝ) (z : ℂ) (hy : y > 0) : percent_of_y y z = 0.6 * y :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1263_126334


namespace NUMINAMATH_GPT_cakes_difference_l1263_126358

theorem cakes_difference :
  let bought := 154
  let sold := 91
  bought - sold = 63 :=
by
  let bought := 154
  let sold := 91
  show bought - sold = 63
  sorry

end NUMINAMATH_GPT_cakes_difference_l1263_126358


namespace NUMINAMATH_GPT_last_two_digits_of_sum_l1263_126327

-- Define factorial, and factorials up to 50 specifically for our problem.
def fac : ℕ → ℕ
  | 0     => 1
  | (n+1) => (n+1) * fac n

-- Sum the last two digits of factorials from 1 to 50
def last_two_digits_sum : ℕ :=
  (fac 1 % 100 + fac 2 % 100 + fac 3 % 100 + fac 4 % 100 + fac 5 % 100 + 
   fac 6 % 100 + fac 7 % 100 + fac 8 % 100 + fac 9 % 100) % 100

theorem last_two_digits_of_sum : last_two_digits_sum = 13 := by
  sorry

end NUMINAMATH_GPT_last_two_digits_of_sum_l1263_126327


namespace NUMINAMATH_GPT_part_a_l1263_126351

-- Lean 4 statement equivalent to Part (a)
theorem part_a (n : ℕ) (x : ℝ) (hn : 0 < n) (hx : n^2 ≤ x) : 
  n * Real.sqrt (x - n^2) ≤ x / 2 := 
sorry

-- Lean 4 statement equivalent to Part (b)
noncomputable def find_xyz : ℕ × ℕ × ℕ :=
  ((2, 8, 18) : ℕ × ℕ × ℕ)

end NUMINAMATH_GPT_part_a_l1263_126351


namespace NUMINAMATH_GPT_greatest_integer_of_negative_fraction_l1263_126360

-- Define the original fraction
def original_fraction : ℚ := -19 / 5

-- Define the greatest integer function
def greatest_integer_less_than (q : ℚ) : ℤ :=
  Int.floor q

-- The proof problem statement:
theorem greatest_integer_of_negative_fraction :
  greatest_integer_less_than original_fraction = -4 :=
sorry

end NUMINAMATH_GPT_greatest_integer_of_negative_fraction_l1263_126360


namespace NUMINAMATH_GPT_father_l1263_126386

-- Conditions definitions
def man's_current_age (F : ℕ) : ℕ := (2 / 5) * F
def man_after_5_years (M F : ℕ) : Prop := M + 5 = (1 / 2) * (F + 5)

-- Main statement to prove
theorem father's_age (F : ℕ) (h₁ : man's_current_age F = (2 / 5) * F)
  (h₂ : ∀ M, man_after_5_years M F → M = (2 / 5) * F + 5): F = 25 :=
sorry

end NUMINAMATH_GPT_father_l1263_126386


namespace NUMINAMATH_GPT_find_age_of_second_person_l1263_126329

variable (T A X : ℝ)

def average_original_group (T A : ℝ) : Prop :=
  T = 7 * A

def average_with_39 (T A : ℝ) : Prop :=
  T + 39 = 8 * (A + 2)

def average_with_second_person (T A X : ℝ) : Prop :=
  T + X = 8 * (A - 1) 

theorem find_age_of_second_person (T A X : ℝ) 
  (h1 : average_original_group T A)
  (h2 : average_with_39 T A)
  (h3 : average_with_second_person T A X) :
  X = 15 :=
sorry

end NUMINAMATH_GPT_find_age_of_second_person_l1263_126329


namespace NUMINAMATH_GPT_cistern_length_l1263_126344

variable (L : ℝ) (width water_depth total_area : ℝ)

theorem cistern_length
  (h_width : width = 8)
  (h_water_depth : water_depth = 1.5)
  (h_total_area : total_area = 134) :
  11 * L + 24 = total_area → L = 10 :=
by
  intro h_eq
  have h_eq1 : 11 * L = 110 := by
    linarith
  have h_L : L = 10 := by
    linarith
  exact h_L

end NUMINAMATH_GPT_cistern_length_l1263_126344


namespace NUMINAMATH_GPT_perpendicular_condition_sufficient_not_necessary_l1263_126335

theorem perpendicular_condition_sufficient_not_necessary (m : ℝ) :
  (∀ x y : ℝ, m * x + (2 * m - 1) * y + 1 = 0) →
  (∀ x y : ℝ, 3 * x + m * y + 3 = 0) →
  (∀ a b : ℝ, m = -1 → (∃ c d : ℝ, 3 / a = 1 / b)) →
  (m = -1 → (m = -1 → (3 / (-m / (2 * m - 1)) * m) / 2 - (3 / m) = -1)) :=
by sorry

end NUMINAMATH_GPT_perpendicular_condition_sufficient_not_necessary_l1263_126335


namespace NUMINAMATH_GPT_max_sum_non_zero_nats_l1263_126393

theorem max_sum_non_zero_nats (O square : ℕ) (hO : O ≠ 0) (hsquare : square ≠ 0) :
  (O / 11 < 7 / square) ∧ (7 / square < 4 / 5) → O + square = 77 :=
by 
  sorry -- Proof omitted as requested

end NUMINAMATH_GPT_max_sum_non_zero_nats_l1263_126393


namespace NUMINAMATH_GPT_range_of_a_l1263_126361

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x ≤ 2 then |x - 2 * a| else x + 1 / (x - 2) + a

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, f a 2 ≤ f a x) : 1 ≤ a ∧ a ≤ 6 := 
sorry

end NUMINAMATH_GPT_range_of_a_l1263_126361


namespace NUMINAMATH_GPT_area_unpainted_region_l1263_126326

theorem area_unpainted_region
  (width_board_1 : ℝ)
  (width_board_2 : ℝ)
  (cross_angle_degrees : ℝ)
  (unpainted_area : ℝ)
  (h1 : width_board_1 = 5)
  (h2 : width_board_2 = 7)
  (h3 : cross_angle_degrees = 45)
  (h4 : unpainted_area = (49 * Real.sqrt 2) / 2) : 
  unpainted_area = (width_board_2 * ((width_board_1 * Real.sqrt 2) / 2)) / 2 :=
sorry

end NUMINAMATH_GPT_area_unpainted_region_l1263_126326


namespace NUMINAMATH_GPT_correct_operation_l1263_126392

theorem correct_operation (a b : ℝ) : 
  (a^2 + a^4 ≠ a^6) ∧
  ((a - b)^2 ≠ a^2 - b^2) ∧
  ((a^2 * b)^3 = a^6 * b^3) ∧
  (a^6 / a^6 ≠ a) :=
by
  sorry

end NUMINAMATH_GPT_correct_operation_l1263_126392


namespace NUMINAMATH_GPT_sufficient_condition_ab_greater_than_1_l1263_126337

theorem sufficient_condition_ab_greater_than_1 (a b : ℝ) (h₁ : a > 1) (h₂ : b > 1) : ab > 1 := 
  sorry

end NUMINAMATH_GPT_sufficient_condition_ab_greater_than_1_l1263_126337


namespace NUMINAMATH_GPT_elements_in_M_l1263_126323

def is_element_of_M (x y : ℕ) : Prop :=
  x + y ≤ 1

def M : Set (ℕ × ℕ) :=
  {p | is_element_of_M p.fst p.snd}

theorem elements_in_M :
  M = { (0,0), (0,1), (1,0) } :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_elements_in_M_l1263_126323


namespace NUMINAMATH_GPT_sunil_investment_l1263_126356

noncomputable def total_amount (P : ℝ) : ℝ :=
  let r1 := 0.025  -- 5% per annum compounded semi-annually
  let r2 := 0.03   -- 6% per annum compounded semi-annually
  let A2 := P * (1 + r1) * (1 + r1)
  let A3 := (A2 + 0.5 * P) * (1 + r2)
  let A4 := A3 * (1 + r2)
  A4

theorem sunil_investment (P : ℝ) : total_amount P = 1.645187625 * P :=
by
  sorry

end NUMINAMATH_GPT_sunil_investment_l1263_126356


namespace NUMINAMATH_GPT_natural_number_property_l1263_126385

theorem natural_number_property (N k : ℕ) (hk : k > 0)
    (h1 : 10^(k-1) ≤ N) (h2 : N < 10^k) (h3 : N * 10^(k-1) ≤ N^2) (h4 : N^2 ≤ N * 10^k) :
    N = 10^(k-1) := 
sorry

end NUMINAMATH_GPT_natural_number_property_l1263_126385


namespace NUMINAMATH_GPT_possible_values_x2_y2_z2_l1263_126352

theorem possible_values_x2_y2_z2 {x y z : ℤ}
    (h1 : x + y + z = 3)
    (h2 : x^3 + y^3 + z^3 = 3) : (x^2 + y^2 + z^2 = 3) ∨ (x^2 + y^2 + z^2 = 57) :=
by sorry

end NUMINAMATH_GPT_possible_values_x2_y2_z2_l1263_126352


namespace NUMINAMATH_GPT_polynomial_at_five_l1263_126357

theorem polynomial_at_five (P : ℝ → ℝ) 
  (hP_degree : ∃ (a b c d : ℝ), ∀ x : ℝ, P x = a*x^3 + b*x^2 + c*x + d)
  (hP1 : P 1 = 1 / 3)
  (hP2 : P 2 = 1 / 7)
  (hP3 : P 3 = 1 / 13)
  (hP4 : P 4 = 1 / 21) :
  P 5 = -3 / 91 :=
sorry

end NUMINAMATH_GPT_polynomial_at_five_l1263_126357


namespace NUMINAMATH_GPT_quadratic_roots_l1263_126382

theorem quadratic_roots : ∀ (x : ℝ), x^2 + 5 * x - 4 = 0 ↔ x = (-5 + Real.sqrt 41) / 2 ∨ x = (-5 - Real.sqrt 41) / 2 := 
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_l1263_126382


namespace NUMINAMATH_GPT_feet_perpendiculars_concyclic_l1263_126339

variables {S A B C D O M N P Q : Type} 

-- Given conditions
variables (is_convex_quadrilateral : convex_quadrilateral A B C D)
variables (diagonals_perpendicular : ∀ (AC BD : Line), perpendicular AC BD)
variables (foot_perpendicular : ∀ (O : Point), intersection_point O = foot (perpendicular_from S (base_quadrilateral A B C D)))

-- Define the proof statement
theorem feet_perpendiculars_concyclic
  (h1 : is_convex_quadrilateral A B C D)
  (h2 : diagonals_perpendicular AC BD)
  (h3 : foot_perpendicular O) :
  concyclic (feet_perpendicular_pts O (face S A B)) (feet_perpendicular_pts O (face S B C)) 
            (feet_perpendicular_pts O (face S C D)) (feet_perpendicular_pts O (face S D A)) := sorry

end NUMINAMATH_GPT_feet_perpendiculars_concyclic_l1263_126339


namespace NUMINAMATH_GPT_find_the_number_l1263_126309

theorem find_the_number (x : ℤ) (h : 2 + x = 6) : x = 4 :=
sorry

end NUMINAMATH_GPT_find_the_number_l1263_126309


namespace NUMINAMATH_GPT_father_son_age_relationship_l1263_126343

theorem father_son_age_relationship 
    (F S X : ℕ) 
    (h1 : F = 27) 
    (h2 : F = 3 * S + 3) 
    : X = 11 ∧ F + X > 2 * (S + X) :=
by
  sorry

end NUMINAMATH_GPT_father_son_age_relationship_l1263_126343


namespace NUMINAMATH_GPT_concert_ticket_to_motorcycle_ratio_l1263_126341

theorem concert_ticket_to_motorcycle_ratio (initial_amount spend_motorcycle remaining_amount : ℕ)
  (h_initial : initial_amount = 5000)
  (h_spend_motorcycle : spend_motorcycle = 2800)
  (amount_left := initial_amount - spend_motorcycle)
  (h_remaining : remaining_amount = 825)
  (h_amount_left : ∃ C : ℕ, amount_left - C - (1/4 : ℚ) * (amount_left - C) = remaining_amount) :
  ∃ C : ℕ, (C / amount_left) = (1 / 2 : ℚ) := sorry

end NUMINAMATH_GPT_concert_ticket_to_motorcycle_ratio_l1263_126341


namespace NUMINAMATH_GPT_approximate_probability_hit_shot_l1263_126300

-- Define the data from the table
def shots : List ℕ := [10, 50, 100, 150, 200, 500, 1000, 2000]
def hits : List ℕ := [9, 40, 70, 108, 143, 361, 721, 1440]
def hit_rates : List ℚ := [0.9, 0.8, 0.7, 0.72, 0.715, 0.722, 0.721, 0.72]

-- State the theorem that the stabilized hit rate is approximately 0.72
theorem approximate_probability_hit_shot : 
  ∃ (p : ℚ), p = 0.72 ∧ 
  ∀ (n : ℕ), n ∈ [150, 200, 500, 1000, 2000] → 
     ∃ (r : ℚ), r = 0.72 ∧ 
     r = (hits.get ⟨shots.indexOf n, sorry⟩ : ℚ) / n := sorry

end NUMINAMATH_GPT_approximate_probability_hit_shot_l1263_126300


namespace NUMINAMATH_GPT_markup_percent_based_on_discounted_price_l1263_126314

-- Defining the conditions
def original_price : ℝ := 1
def discount_percent : ℝ := 0.2
def discounted_price : ℝ := original_price * (1 - discount_percent)

-- The proof problem statement
theorem markup_percent_based_on_discounted_price :
  (original_price - discounted_price) / discounted_price = 0.25 :=
sorry

end NUMINAMATH_GPT_markup_percent_based_on_discounted_price_l1263_126314


namespace NUMINAMATH_GPT_solve_for_k_l1263_126364

theorem solve_for_k (x y k : ℕ) (hx : 0 < x) (hy : 0 < y) 
  (h : (1 / 2)^(25 * x) * (1 / 81)^k = 1 / (18 ^ (25 * y))) :
  k = 25 * y / 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_k_l1263_126364


namespace NUMINAMATH_GPT_find_x_plus_y_l1263_126312

variables {x y : ℝ}

def f (t : ℝ) : ℝ := t^2003 + 2002 * t

theorem find_x_plus_y (hx : f (x - 1) = -1) (hy : f (y - 2) = 1) : x + y = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_x_plus_y_l1263_126312


namespace NUMINAMATH_GPT_pens_sold_l1263_126368

variable (C S : ℝ)
variable (n : ℕ)

-- Define conditions
def condition1 : Prop := 10 * C = n * S
def condition2 : Prop := S = 1.5 * C

-- Define the statement to be proved
theorem pens_sold (h1 : condition1 C S n) (h2 : condition2 C S) : n = 6 := by
  -- leave the proof steps to be filled in
  sorry

end NUMINAMATH_GPT_pens_sold_l1263_126368


namespace NUMINAMATH_GPT_students_exam_percentage_l1263_126319

theorem students_exam_percentage 
  (total_students : ℕ) 
  (avg_assigned_day : ℚ) 
  (avg_makeup_day : ℚ)
  (overall_avg : ℚ) 
  (h_total : total_students = 100)
  (h_avg_assigned_day : avg_assigned_day = 0.60) 
  (h_avg_makeup_day : avg_makeup_day = 0.80) 
  (h_overall_avg : overall_avg = 0.66) : 
  ∃ x : ℚ, x = 70 / 100 :=
by
  sorry

end NUMINAMATH_GPT_students_exam_percentage_l1263_126319


namespace NUMINAMATH_GPT_total_sections_l1263_126333

theorem total_sections (boys girls : ℕ) (h1 : boys = 408) (h2 : girls = 288) : 
  (boys / Nat.gcd boys girls) + (girls / Nat.gcd boys girls) = 29 :=
by
  sorry

end NUMINAMATH_GPT_total_sections_l1263_126333


namespace NUMINAMATH_GPT_simplify_fraction_expression_l1263_126376

variable (a b : ℝ)
variable (h_pos_a : 0 < a) (h_pos_b : 0 < b)
variable (h_eq : a^3 - b^3 = a - b)

theorem simplify_fraction_expression : (a / b) + (b / a) + (1 / (a * b)) = 2 := by
  sorry

end NUMINAMATH_GPT_simplify_fraction_expression_l1263_126376


namespace NUMINAMATH_GPT_AM_GM_inequality_l1263_126315

theorem AM_GM_inequality (a b : ℝ) (n : ℕ) (ha : 0 < a) (hb : 0 < b) (hn : 0 < n) :
  (a^n + b^n) / 2 ≥ ((a + b) / 2) ^ n :=
by
  sorry

end NUMINAMATH_GPT_AM_GM_inequality_l1263_126315


namespace NUMINAMATH_GPT_vector_subtraction_proof_l1263_126389

theorem vector_subtraction_proof (a b : ℝ × ℝ) (ha : a = (3, 2)) (hb : b = (0, -1)) :
    3 • b - a = (-3, -5) := by
  sorry

end NUMINAMATH_GPT_vector_subtraction_proof_l1263_126389


namespace NUMINAMATH_GPT_remainder_1493824_div_4_l1263_126380

theorem remainder_1493824_div_4 : 1493824 % 4 = 0 :=
by
  sorry

end NUMINAMATH_GPT_remainder_1493824_div_4_l1263_126380


namespace NUMINAMATH_GPT_inequality_solution_set_l1263_126306

open Set

noncomputable def rational_expression (x : ℝ) : ℝ := (x^2 - 16) / (x^2 + 10*x + 25)

theorem inequality_solution_set :
  {x : ℝ | rational_expression x < 0} = Ioo (-4 : ℝ) 4 :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_set_l1263_126306


namespace NUMINAMATH_GPT_big_sale_commission_l1263_126370

theorem big_sale_commission (avg_increase : ℝ) (new_avg : ℝ) (num_sales : ℕ) 
  (prev_avg := new_avg - avg_increase)
  (total_prev := prev_avg * (num_sales - 1))
  (total_new := new_avg * num_sales)
  (C := total_new - total_prev) :
  avg_increase = 150 → new_avg = 250 → num_sales = 6 → C = 1000 :=
by
  intros 
  sorry

end NUMINAMATH_GPT_big_sale_commission_l1263_126370


namespace NUMINAMATH_GPT_inequality_abc_ad_bc_bd_cd_l1263_126342

theorem inequality_abc_ad_bc_bd_cd (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (1 / (a * b) + 1 / (a * c) + 1 / (a * d) + 1 / (b * c) + 1 / (b * d) + 1 / (c * d)) 
  ≤ (3 / 8) * (1 / a + 1 / b + 1 / c + 1 / d) ^ 2 := sorry

end NUMINAMATH_GPT_inequality_abc_ad_bc_bd_cd_l1263_126342


namespace NUMINAMATH_GPT_binomial_square_l1263_126355

variable (c : ℝ)

theorem binomial_square (h : ∃ a : ℝ, (x^2 - 164 * x + c) = (x + a)^2) : c = 6724 := sorry

end NUMINAMATH_GPT_binomial_square_l1263_126355


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1263_126331

theorem solution_set_of_inequality (x : ℝ) :
  |x^2 - 2| < 2 ↔ (-2 < x ∧ x < 0) ∨ (0 < x ∧ x < 2) :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1263_126331


namespace NUMINAMATH_GPT_range_of_a_l1263_126384

variables (a : ℝ) (x : ℝ) (x0 : ℝ)

def proposition_P (a : ℝ) : Prop :=
  ∀ x, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0

def proposition_Q (a : ℝ) : Prop :=
  ∃ x0, x0^2 + 2 * a * x0 + 2 - a = 0

theorem range_of_a (a : ℝ) :
  (proposition_P a ∧ proposition_Q a) → a ∈ {a : ℝ | a ≤ -2} ∪ {a : ℝ | a = 1} :=
by {
  sorry -- Proof goes here.
}

end NUMINAMATH_GPT_range_of_a_l1263_126384


namespace NUMINAMATH_GPT_domain_of_g_l1263_126313

theorem domain_of_g (x y : ℝ) : 
  (∃ g : ℝ, g = 1 / (x^2 + (x - y)^2 + y^2)) ↔ (x, y) ≠ (0, 0) :=
by sorry

end NUMINAMATH_GPT_domain_of_g_l1263_126313


namespace NUMINAMATH_GPT_find_fraction_l1263_126369

variable (F N : ℚ)

-- Defining the conditions
def condition1 : Prop := (1 / 3) * F * N = 18
def condition2 : Prop := (3 / 10) * N = 64.8

-- Proof statement
theorem find_fraction (h1 : condition1 F N) (h2 : condition2 N) : F = 1 / 4 := by 
  sorry

end NUMINAMATH_GPT_find_fraction_l1263_126369


namespace NUMINAMATH_GPT_combined_work_time_l1263_126381

-- Define the time taken by Paul and Rose to complete the work individually
def paul_days : ℕ := 80
def rose_days : ℕ := 120

-- Define the work rates of Paul and Rose
def paul_rate := 1 / (paul_days : ℚ)
def rose_rate := 1 / (rose_days : ℚ)

-- Define the combined work rate
def combined_rate := paul_rate + rose_rate

-- Statement to prove: Together they can complete the work in 48 days.
theorem combined_work_time : combined_rate = 1 / 48 := by 
  sorry

end NUMINAMATH_GPT_combined_work_time_l1263_126381


namespace NUMINAMATH_GPT_prod_ab_eq_three_l1263_126307

theorem prod_ab_eq_three (a b : ℝ) (h₁ : a - b = 5) (h₂ : a^2 + b^2 = 31) : a * b = 3 := 
sorry

end NUMINAMATH_GPT_prod_ab_eq_three_l1263_126307


namespace NUMINAMATH_GPT_area_enclosed_by_circle_l1263_126373

theorem area_enclosed_by_circle : Π (x y : ℝ), x^2 + y^2 + 8 * x - 6 * y = -9 → 
  ∃ A, A = 7 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_area_enclosed_by_circle_l1263_126373


namespace NUMINAMATH_GPT_soaking_time_l1263_126374

theorem soaking_time (time_per_grass_stain : ℕ) (time_per_marinara_stain : ℕ) 
    (number_of_grass_stains : ℕ) (number_of_marinara_stains : ℕ) : 
    time_per_grass_stain = 4 ∧ time_per_marinara_stain = 7 ∧ 
    number_of_grass_stains = 3 ∧ number_of_marinara_stains = 1 →
    (time_per_grass_stain * number_of_grass_stains + time_per_marinara_stain * number_of_marinara_stains) = 19 :=
by
  sorry

end NUMINAMATH_GPT_soaking_time_l1263_126374


namespace NUMINAMATH_GPT_four_thirds_of_number_is_36_l1263_126338

theorem four_thirds_of_number_is_36 (x : ℝ) (h : (4 / 3) * x = 36) : x = 27 :=
  sorry

end NUMINAMATH_GPT_four_thirds_of_number_is_36_l1263_126338


namespace NUMINAMATH_GPT_maximum_squares_formation_l1263_126346

theorem maximum_squares_formation (total_matchsticks : ℕ) (triangles : ℕ) (used_for_triangles : ℕ) (remaining_matchsticks : ℕ) (squares : ℕ):
  total_matchsticks = 24 →
  triangles = 6 →
  used_for_triangles = 13 →
  remaining_matchsticks = total_matchsticks - used_for_triangles →
  squares = remaining_matchsticks / 4 →
  squares = 4 :=
by
  sorry

end NUMINAMATH_GPT_maximum_squares_formation_l1263_126346


namespace NUMINAMATH_GPT_mother_daughter_ages_l1263_126350

theorem mother_daughter_ages :
  ∃ (x y : ℕ), (y = x + 22) ∧ (2 * x = (x + 22) - x) ∧ (x = 11) ∧ (y = 33) :=
by
  sorry

end NUMINAMATH_GPT_mother_daughter_ages_l1263_126350


namespace NUMINAMATH_GPT_sin_cos_ratio_l1263_126387

open Real

theorem sin_cos_ratio
  (θ : ℝ)
  (h : (sin θ + cos θ) / (sin θ - cos θ) = 2) :
  sin θ * cos θ = 3 / 10 := 
by
  sorry

end NUMINAMATH_GPT_sin_cos_ratio_l1263_126387


namespace NUMINAMATH_GPT_find_angle_A_find_area_l1263_126394

-- Definition for angle A
theorem find_angle_A (a b c : ℝ) (A B C : ℝ) 
  (h1 : a * Real.cos C + c * Real.cos A = 2 * b * Real.cos A)
  (h_tria : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_A : 0 < A ∧ A < Real.pi) :
  A = Real.pi / 3 :=
by
  sorry

-- Definition for area of triangle ABC
theorem find_area (a b c : ℝ) (A : ℝ)
  (h_a : a = Real.sqrt 7) 
  (h_b : b = 2)
  (h_A : A = Real.pi / 3) 
  (h_c : c = 3) :
  (1/2) * b * c * Real.sin A = (3 * Real.sqrt 3) / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_angle_A_find_area_l1263_126394


namespace NUMINAMATH_GPT_percentage_of_students_attend_chess_class_l1263_126321

-- Definitions based on the conditions
def total_students : ℕ := 1000
def swimming_students : ℕ := 125
def chess_to_swimming_ratio : ℚ := 1 / 2

-- Problem statement
theorem percentage_of_students_attend_chess_class :
  ∃ P : ℚ, (P / 100) * total_students / 2 = swimming_students → P = 25 := by
  sorry

end NUMINAMATH_GPT_percentage_of_students_attend_chess_class_l1263_126321


namespace NUMINAMATH_GPT_sum_coordinates_l1263_126363

theorem sum_coordinates (x : ℝ) : 
  let C := (x, 8)
  let D := (-x, 8)
  (C.1 + C.2 + D.1 + D.2) = 16 := 
by
  sorry

end NUMINAMATH_GPT_sum_coordinates_l1263_126363


namespace NUMINAMATH_GPT_nth_equation_l1263_126304

theorem nth_equation (n : ℕ) : (2 * n + 1)^2 - (2 * n - 1)^2 = 8 * n := by
  sorry

end NUMINAMATH_GPT_nth_equation_l1263_126304


namespace NUMINAMATH_GPT_modulo_inverse_product_l1263_126372

open Int 

theorem modulo_inverse_product (n : ℕ) (a b c : ℤ) 
  (hn : 0 < n) 
  (ha : a * a.gcd n = 1) 
  (hb : b * b.gcd n = 1) 
  (hc : c * c.gcd n = 1) 
  (hab : (a * b) % n = 1) 
  (hac : (c * a) % n = 1) : 
  ((a * b) * c) % n = c % n :=
by
  sorry

end NUMINAMATH_GPT_modulo_inverse_product_l1263_126372


namespace NUMINAMATH_GPT_C_share_of_rent_l1263_126377

-- Define the given conditions
def A_ox_months : ℕ := 10 * 7
def B_ox_months : ℕ := 12 * 5
def C_ox_months : ℕ := 15 * 3
def total_rent : ℕ := 175
def total_ox_months : ℕ := A_ox_months + B_ox_months + C_ox_months
def cost_per_ox_month := total_rent / total_ox_months

-- The goal is to prove that C's share of the rent is Rs. 45
theorem C_share_of_rent : C_ox_months * cost_per_ox_month = 45 := by
  -- Adding sorry to skip the proof
  sorry

end NUMINAMATH_GPT_C_share_of_rent_l1263_126377


namespace NUMINAMATH_GPT_people_in_room_after_2019_minutes_l1263_126332

theorem people_in_room_after_2019_minutes :
  ∀ (P : Nat → Int), 
    P 0 = 0 -> 
    (∀ t, P (t+1) = P t + 2 ∨ P (t+1) = P t - 1) -> 
    P 2019 ≠ 2018 :=
by
  intros P hP0 hP_changes
  sorry

end NUMINAMATH_GPT_people_in_room_after_2019_minutes_l1263_126332


namespace NUMINAMATH_GPT_total_snow_volume_l1263_126379

-- Definitions and conditions set up from part (a)
def driveway_length : ℝ := 30
def driveway_width : ℝ := 3
def section1_length : ℝ := 10
def section1_depth : ℝ := 1
def section2_length : ℝ := driveway_length - section1_length
def section2_depth : ℝ := 0.5

-- The theorem corresponding to part (c)
theorem total_snow_volume : 
  (section1_length * driveway_width * section1_depth) +
  (section2_length * driveway_width * section2_depth) = 60 :=
by 
  -- Proof is omitted as required
  sorry

end NUMINAMATH_GPT_total_snow_volume_l1263_126379


namespace NUMINAMATH_GPT_least_n_factorial_6930_l1263_126397

theorem least_n_factorial_6930 (n : ℕ) (h : n! % 6930 = 0) : n ≥ 11 := by
  sorry

end NUMINAMATH_GPT_least_n_factorial_6930_l1263_126397
