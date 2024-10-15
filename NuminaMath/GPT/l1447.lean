import Mathlib

namespace NUMINAMATH_GPT_parallelogram_base_length_l1447_144706

theorem parallelogram_base_length (A h : ℕ) (hA : A = 32) (hh : h = 8) : (A / h) = 4 := by
  sorry

end NUMINAMATH_GPT_parallelogram_base_length_l1447_144706


namespace NUMINAMATH_GPT_triangles_in_figure_l1447_144765

-- Define the conditions of the problem.
def bottom_row_small := 4
def next_row_small := 3
def following_row_small := 2
def topmost_row_small := 1

def small_triangles := bottom_row_small + next_row_small + following_row_small + topmost_row_small

def medium_triangles := 3
def large_triangle := 1

def total_triangles := small_triangles + medium_triangles + large_triangle

-- Lean proof statement that the total number of triangles is 14
theorem triangles_in_figure : total_triangles = 14 :=
by
  unfold total_triangles
  unfold small_triangles
  unfold bottom_row_small next_row_small following_row_small topmost_row_small
  unfold medium_triangles large_triangle
  sorry

end NUMINAMATH_GPT_triangles_in_figure_l1447_144765


namespace NUMINAMATH_GPT_f_pi_over_4_l1447_144760

noncomputable def f (ω φ x : ℝ) : ℝ := 2 * Real.cos (ω * x + φ)

theorem f_pi_over_4 (ω φ : ℝ) (h : ω ≠ 0) 
  (symm : ∀ x, f ω φ (π / 4 + x) = f ω φ (π / 4 - x)) : 
  f ω φ (π / 4) = 2 ∨ f ω φ (π / 4) = -2 := 
by 
  sorry

end NUMINAMATH_GPT_f_pi_over_4_l1447_144760


namespace NUMINAMATH_GPT_find_x_l1447_144744

theorem find_x (x : ℝ) :
  (1 / 3) * ((2 * x + 8) + (7 * x + 3) + (3 * x + 9)) = 5 * x^2 - 8 * x + 2 ↔ 
  x = (36 + Real.sqrt 2136) / 30 ∨ x = (36 - Real.sqrt 2136) / 30 := 
sorry

end NUMINAMATH_GPT_find_x_l1447_144744


namespace NUMINAMATH_GPT_initial_money_l1447_144798

-- Define the conditions
def spent_toy_truck : ℕ := 3
def spent_pencil_case : ℕ := 2
def money_left : ℕ := 5

-- Define the total money spent
def total_spent := spent_toy_truck + spent_pencil_case

-- Theorem statement
theorem initial_money (I : ℕ) (h : total_spent + money_left = I) : I = 10 :=
sorry

end NUMINAMATH_GPT_initial_money_l1447_144798


namespace NUMINAMATH_GPT_find_length_PQ_l1447_144740

noncomputable def length_of_PQ (PQ PR : ℝ) (ST SU : ℝ) (angle_PQPR angle_STSU : ℝ) : ℝ :=
if (angle_PQPR = 120 ∧ angle_STSU = 120 ∧ PR / SU = 8 / 9) then 
  2 
else 
  0

theorem find_length_PQ :
  let PQ := 4 
  let PR := 8
  let ST := 9
  let SU := 18
  let PQ_crop := 2
  let angle_PQPR := 120
  let angle_STSU := 120
  length_of_PQ PQ PR ST SU angle_PQPR angle_STSU = PQ_crop :=
by
  sorry

end NUMINAMATH_GPT_find_length_PQ_l1447_144740


namespace NUMINAMATH_GPT_complement_intersection_l1447_144779

open Set

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4}

-- Define the set A
def A : Set ℕ := {1, 2}

-- Define the set B
def B : Set ℕ := {2, 3, 4}

-- Statement to be proven
theorem complement_intersection :
  (U \ A) ∩ B = {3, 4} :=
sorry

end NUMINAMATH_GPT_complement_intersection_l1447_144779


namespace NUMINAMATH_GPT_Reeya_fifth_subject_score_l1447_144783

theorem Reeya_fifth_subject_score 
  (a1 a2 a3 a4 : ℕ) (avg : ℕ) (subjects : ℕ) (a1_eq : a1 = 55) (a2_eq : a2 = 67) (a3_eq : a3 = 76) 
  (a4_eq : a4 = 82) (avg_eq : avg = 73) (subjects_eq : subjects = 5) :
  ∃ a5 : ℕ, (a1 + a2 + a3 + a4 + a5) / subjects = avg ∧ a5 = 85 :=
by
  sorry

end NUMINAMATH_GPT_Reeya_fifth_subject_score_l1447_144783


namespace NUMINAMATH_GPT_rita_saving_l1447_144745

theorem rita_saving
  (num_notebooks : ℕ)
  (price_per_notebook : ℝ)
  (discount_rate : ℝ) :
  num_notebooks = 7 →
  price_per_notebook = 3 →
  discount_rate = 0.15 →
  (num_notebooks * price_per_notebook) - (num_notebooks * (price_per_notebook * (1 - discount_rate))) = 3.15 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end NUMINAMATH_GPT_rita_saving_l1447_144745


namespace NUMINAMATH_GPT_largest_divisor_of_odd_sequence_for_even_n_l1447_144719

theorem largest_divisor_of_odd_sequence_for_even_n (n : ℕ) (h : n % 2 = 0) : 
  ∃ d : ℕ, d = 105 ∧ ∀ k : ℕ, k = (n+1)*(n+3)*(n+5)*(n+7)*(n+9)*(n+11)*(n+13) → 105 ∣ k :=
sorry

end NUMINAMATH_GPT_largest_divisor_of_odd_sequence_for_even_n_l1447_144719


namespace NUMINAMATH_GPT_pet_store_problem_l1447_144702

noncomputable def num_ways_to_buy_pets (puppies kittens hamsters birds : ℕ) (people : ℕ) : ℕ :=
  (puppies * kittens * hamsters * birds) * (people.factorial)

theorem pet_store_problem :
  num_ways_to_buy_pets 12 10 5 3 4 = 43200 :=
by
  sorry

end NUMINAMATH_GPT_pet_store_problem_l1447_144702


namespace NUMINAMATH_GPT_zero_points_sum_gt_one_l1447_144724

noncomputable def f (x : ℝ) : ℝ := Real.log x + (1 / (2 * x))

noncomputable def g (x m : ℝ) : ℝ := f x - m

theorem zero_points_sum_gt_one (x₁ x₂ m : ℝ) (h₁ : x₁ < x₂) 
  (hx₁ : g x₁ m = 0) (hx₂ : g x₂ m = 0) : 
  x₁ + x₂ > 1 := 
  by
    sorry

end NUMINAMATH_GPT_zero_points_sum_gt_one_l1447_144724


namespace NUMINAMATH_GPT_fixed_point_on_line_find_m_values_l1447_144754

-- Define the conditions and set up the statements to prove

/-- 
Condition 1: Line equation 
-/
def line_eq (m x y : ℝ) : Prop := (2 * m + 1) * x + (m + 1) * y - 7 * m - 4 = 0

/-- 
Condition 2: Circle equation 
-/
def circle_eq (x y : ℝ) : Prop := (x - 1) ^ 2 + (y - 2) ^ 2 = 25

/-- 
Question (1): Fixed point (3,1) is always on the line
-/
theorem fixed_point_on_line (m : ℝ) : line_eq m 3 1 := by
  sorry

/-- 
Question (2): Finding the values of m for the given chord length
-/
theorem find_m_values (m : ℝ) (h_chord : ∀x y : ℝ, circle_eq x y → line_eq m x y → (x - y)^2 = 6) : 
  m = -1/2 ∨ m = 1/2 := by
  sorry

end NUMINAMATH_GPT_fixed_point_on_line_find_m_values_l1447_144754


namespace NUMINAMATH_GPT_round_trip_time_l1447_144714

theorem round_trip_time 
  (d1 d2 d3 : ℝ) 
  (s1 s2 s3 t : ℝ) 
  (h1 : d1 = 18) 
  (h2 : d2 = 18) 
  (h3 : d3 = 36) 
  (h4 : s1 = 12) 
  (h5 : s2 = 10) 
  (h6 : s3 = 9) 
  (h7 : t = (d1 / s1) + (d2 / s2) + (d3 / s3)) :
  t = 7.3 :=
by
  sorry

end NUMINAMATH_GPT_round_trip_time_l1447_144714


namespace NUMINAMATH_GPT_find_B_divisible_by_6_l1447_144710

theorem find_B_divisible_by_6 (B : ℕ) : (5170 + B) % 6 = 0 ↔ (B = 2 ∨ B = 8) :=
by
  -- Conditions extracted from the problem are directly used here:
  sorry -- Proof would be here

end NUMINAMATH_GPT_find_B_divisible_by_6_l1447_144710


namespace NUMINAMATH_GPT_negation_of_exists_l1447_144741

theorem negation_of_exists (x : ℝ) : x^2 + 2 * x + 2 > 0 := sorry

end NUMINAMATH_GPT_negation_of_exists_l1447_144741


namespace NUMINAMATH_GPT_distance_between_parallel_lines_l1447_144700

theorem distance_between_parallel_lines (r d : ℝ) :
  let c₁ := 36
  let c₂ := 36
  let c₃ := 40
  let expr1 := (324 : ℝ) + (1 / 4) * d^2
  let expr2 := (400 : ℝ) + d^2
  let radius_eq1 := r^2 = expr1
  let radius_eq2 := r^2 = expr2
  radius_eq1 ∧ radius_eq2 → d = Real.sqrt (304 / 3) :=
by
  sorry

end NUMINAMATH_GPT_distance_between_parallel_lines_l1447_144700


namespace NUMINAMATH_GPT_teacher_engineer_ratio_l1447_144769

theorem teacher_engineer_ratio 
  (t e : ℕ) -- t is the number of teachers and e is the number of engineers.
  (h1 : (40 * t + 55 * e) / (t + e) = 45)
  : t = 2 * e :=
by
  sorry

end NUMINAMATH_GPT_teacher_engineer_ratio_l1447_144769


namespace NUMINAMATH_GPT_rate_of_interest_l1447_144782

noncomputable def compound_interest (P r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r / 100) ^ (n : ℝ)

theorem rate_of_interest (P : ℝ) (r : ℝ) (A : ℕ → ℝ) :
  A 2 = compound_interest P r 2 →
  A 3 = compound_interest P r 3 →
  A 2 = 2420 →
  A 3 = 2662 →
  r = 10 :=
by
  sorry

end NUMINAMATH_GPT_rate_of_interest_l1447_144782


namespace NUMINAMATH_GPT_reading_minutes_per_disc_l1447_144796

-- Define the total reading time
def total_reading_time := 630

-- Define the maximum capacity per disc
def max_capacity_per_disc := 80

-- Define the allowable unused space
def max_unused_space := 4

-- Define the effective capacity of each disc
def effective_capacity_per_disc := max_capacity_per_disc - max_unused_space

-- Define the number of discs needed, rounded up as a ceiling function
def number_of_discs := Nat.ceil (total_reading_time / effective_capacity_per_disc)

-- Theorem statement: Each disc will contain 70 minutes of reading if all conditions are met
theorem reading_minutes_per_disc : ∀ (total_reading_time : ℕ) (max_capacity_per_disc : ℕ) (max_unused_space : ℕ)
  (effective_capacity_per_disc := max_capacity_per_disc - max_unused_space) 
  (number_of_discs := Nat.ceil (total_reading_time / effective_capacity_per_disc)), 
  number_of_discs = 9 → total_reading_time / number_of_discs = 70 :=
by
  sorry

end NUMINAMATH_GPT_reading_minutes_per_disc_l1447_144796


namespace NUMINAMATH_GPT_tan_double_angle_l1447_144735

theorem tan_double_angle (α : Real) (h1 : Real.sin α - Real.cos α = 4 / 3) (h2 : α ∈ Set.Icc (Real.pi / 2) (3 * Real.pi / 4)) :
  Real.tan (2 * α) = (7 * Real.sqrt 2) / 8 :=
by
  sorry

end NUMINAMATH_GPT_tan_double_angle_l1447_144735


namespace NUMINAMATH_GPT_total_time_is_60_l1447_144757

def emma_time : ℕ := 20
def fernando_time : ℕ := 2 * emma_time
def total_time : ℕ := emma_time + fernando_time

theorem total_time_is_60 : total_time = 60 := by
  sorry

end NUMINAMATH_GPT_total_time_is_60_l1447_144757


namespace NUMINAMATH_GPT_man_rate_in_still_water_l1447_144751

theorem man_rate_in_still_water (Vm Vs : ℝ) :
  Vm + Vs = 20 ∧ Vm - Vs = 8 → Vm = 14 :=
by
  sorry

end NUMINAMATH_GPT_man_rate_in_still_water_l1447_144751


namespace NUMINAMATH_GPT_first_day_of_month_is_thursday_l1447_144785

theorem first_day_of_month_is_thursday :
  (27 - 7 - 7 - 7 + 1) % 7 = 4 :=
by
  sorry

end NUMINAMATH_GPT_first_day_of_month_is_thursday_l1447_144785


namespace NUMINAMATH_GPT_almond_butter_ratio_l1447_144759

theorem almond_butter_ratio
  (peanut_cost almond_cost batch_extra almond_per_batch : ℝ)
  (h1 : almond_cost = 3 * peanut_cost)
  (h2 : peanut_cost = 3)
  (h3 : almond_per_batch = batch_extra)
  (h4 : batch_extra = 3) :
  almond_per_batch / almond_cost = 1 / 3 := sorry

end NUMINAMATH_GPT_almond_butter_ratio_l1447_144759


namespace NUMINAMATH_GPT_sara_frosting_total_l1447_144758

def cakes_baked_each_day : List Nat := [7, 12, 8, 10, 15]
def cakes_eaten_by_Carol : List Nat := [4, 6, 3, 2, 3]
def cans_per_cake_each_day : List Nat := [2, 3, 4, 3, 2]

def total_frosting_cans_needed : Nat :=
  let remaining_cakes := List.zipWith (· - ·) cakes_baked_each_day cakes_eaten_by_Carol
  let required_cans := List.zipWith (· * ·) remaining_cakes cans_per_cake_each_day
  required_cans.foldl (· + ·) 0

theorem sara_frosting_total : total_frosting_cans_needed = 92 := by
  sorry

end NUMINAMATH_GPT_sara_frosting_total_l1447_144758


namespace NUMINAMATH_GPT_kevin_food_expense_l1447_144726

theorem kevin_food_expense
    (total_budget : ℕ)
    (samuel_ticket : ℕ)
    (samuel_food_drinks : ℕ)
    (kevin_ticket : ℕ)
    (kevin_drinks : ℕ)
    (kevin_total_exp : ℕ) :
    total_budget = 20 →
    samuel_ticket = 14 →
    samuel_food_drinks = 6 →
    kevin_ticket = 14 →
    kevin_drinks = 2 →
    kevin_total_exp = 20 →
    kevin_food = 4 :=
by
  sorry

end NUMINAMATH_GPT_kevin_food_expense_l1447_144726


namespace NUMINAMATH_GPT_min_sum_factors_l1447_144774

theorem min_sum_factors (a b c : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_prod : a * b * c = 2310) : a + b + c = 40 :=
sorry

end NUMINAMATH_GPT_min_sum_factors_l1447_144774


namespace NUMINAMATH_GPT_unique_solution_tan_eq_sin_cos_l1447_144708

theorem unique_solution_tan_eq_sin_cos :
  ∃! x, 0 ≤ x ∧ x ≤ Real.arccos 0.1 ∧ Real.tan x = Real.sin (Real.cos x) :=
sorry

end NUMINAMATH_GPT_unique_solution_tan_eq_sin_cos_l1447_144708


namespace NUMINAMATH_GPT_range_of_b_l1447_144723

noncomputable def f (x : ℝ) : ℝ :=
  if x < -1 / 2 then (2 * x + 1) / (x ^ 2) else x + 1

def g (x : ℝ) : ℝ := x ^ 2 - 4 * x - 4

-- The main theorem to prove the range of b
theorem range_of_b (a b : ℝ) (h : f a + g b = 0) : b ∈ Set.Icc (-1) 5 := by
  sorry

end NUMINAMATH_GPT_range_of_b_l1447_144723


namespace NUMINAMATH_GPT_calc_expression_l1447_144718

theorem calc_expression : 5 + 2 * (8 - 3) = 15 :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_calc_expression_l1447_144718


namespace NUMINAMATH_GPT_always_positive_sum_l1447_144775

def f : ℝ → ℝ := sorry  -- assuming f(x) is provided elsewhere

theorem always_positive_sum (f : ℝ → ℝ)
    (h1 : ∀ x, f x = -f (2 - x))
    (h2 : ∀ x, x < 1 → f (x) < f (x + 1))
    (x1 x2 : ℝ)
    (h3 : x1 + x2 > 2)
    (h4 : (x1 - 1) * (x2 - 1) < 0) :
  f x1 + f x2 > 0 :=
by {
  sorry
}

end NUMINAMATH_GPT_always_positive_sum_l1447_144775


namespace NUMINAMATH_GPT_number_of_rows_seating_10_is_zero_l1447_144737

theorem number_of_rows_seating_10_is_zero :
  ∀ (y : ℕ) (total_people : ℕ) (total_rows : ℕ),
    (∀ (r : ℕ), r * 9 + (total_rows - r) * 10 = total_people) →
    total_people = 54 →
    total_rows = 6 →
    y = 0 :=
by
  sorry

end NUMINAMATH_GPT_number_of_rows_seating_10_is_zero_l1447_144737


namespace NUMINAMATH_GPT_solve_expr_l1447_144739

theorem solve_expr (x : ℝ) (h : x = 3) : x^6 - 6 * x^2 = 675 := by
  sorry

end NUMINAMATH_GPT_solve_expr_l1447_144739


namespace NUMINAMATH_GPT_lateral_surface_area_cut_off_l1447_144767

theorem lateral_surface_area_cut_off {a b c d : ℝ} (h₁ : a = 4) (h₂ : b = 25) 
(h₃ : c = (2/5 : ℝ)) (h₄ : d = 2 * (4 / 25) * b) : 
4 + 10 + (1/4 * b) = 20.25 :=
by
  sorry

end NUMINAMATH_GPT_lateral_surface_area_cut_off_l1447_144767


namespace NUMINAMATH_GPT_meatballs_fraction_each_son_eats_l1447_144792

theorem meatballs_fraction_each_son_eats
  (f1 f2 f3 : ℝ)
  (h1 : ∃ f1 f2 f3, f1 + f2 + f3 = 2)
  (meatballs_initial : ∀ n, n = 3) :
  f1 = 2/3 ∧ f2 = 2/3 ∧ f3 = 2/3 := by
  sorry

end NUMINAMATH_GPT_meatballs_fraction_each_son_eats_l1447_144792


namespace NUMINAMATH_GPT_greatest_three_digit_multiple_23_l1447_144794

theorem greatest_three_digit_multiple_23 : 
  ∃ n : ℕ, n < 1000 ∧ n % 23 = 0 ∧ (∀ m : ℕ, m < 1000 ∧ m % 23 = 0 → m ≤ n) ∧ n = 989 :=
sorry

end NUMINAMATH_GPT_greatest_three_digit_multiple_23_l1447_144794


namespace NUMINAMATH_GPT_jake_correct_speed_l1447_144731

noncomputable def distance (d t : ℝ) : Prop :=
  d = 50 * (t + 4/60) ∧ d = 70 * (t - 4/60)

noncomputable def correct_speed (d t : ℝ) : ℝ :=
  d / t

theorem jake_correct_speed (d t : ℝ) (h1 : distance d t) : correct_speed d t = 58 :=
by
  sorry

end NUMINAMATH_GPT_jake_correct_speed_l1447_144731


namespace NUMINAMATH_GPT_maximum_pencils_l1447_144747

-- Define the problem conditions
def red_pencil_cost := 27
def blue_pencil_cost := 23
def max_total_cost := 940
def max_diff := 10

-- Define the main theorem
theorem maximum_pencils (x y : ℕ) 
  (h1 : red_pencil_cost * x + blue_pencil_cost * y ≤ max_total_cost)
  (h2 : y - x ≤ max_diff)
  (hx_min : ∀ z : ℕ, z < x → red_pencil_cost * z + blue_pencil_cost * (z + max_diff) > max_total_cost):
  x = 14 ∧ y = 24 ∧ x + y = 38 := 
  sorry

end NUMINAMATH_GPT_maximum_pencils_l1447_144747


namespace NUMINAMATH_GPT_binom_11_1_l1447_144797

theorem binom_11_1 : Nat.choose 11 1 = 11 :=
by
  sorry

end NUMINAMATH_GPT_binom_11_1_l1447_144797


namespace NUMINAMATH_GPT_contrapositive_l1447_144722

theorem contrapositive (x : ℝ) : (x > 1 → x^2 + x > 2) ↔ (x^2 + x ≤ 2 → x ≤ 1) :=
sorry

end NUMINAMATH_GPT_contrapositive_l1447_144722


namespace NUMINAMATH_GPT_battery_usage_minutes_l1447_144763

theorem battery_usage_minutes (initial_battery final_battery : ℝ) (initial_minutes : ℝ) (rate_of_usage : ℝ) :
  initial_battery - final_battery = rate_of_usage * initial_minutes →
  initial_battery = 100 →
  final_battery = 68 →
  initial_minutes = 60 →
  rate_of_usage = 8 / 15 →
  ∃ additional_minutes : ℝ, additional_minutes = 127.5 :=
by
  intros
  sorry

end NUMINAMATH_GPT_battery_usage_minutes_l1447_144763


namespace NUMINAMATH_GPT_tetrahedron_distance_sum_l1447_144778

theorem tetrahedron_distance_sum (S₁ S₂ S₃ S₄ H₁ H₂ H₃ H₄ V k : ℝ) 
  (h1 : S₁ = k) (h2 : S₂ = 2 * k) (h3 : S₃ = 3 * k) (h4 : S₄ = 4 * k)
  (V_eq : (1 / 3) * S₁ * H₁ + (1 / 3) * S₂ * H₂ + (1 / 3) * S₃ * H₃ + (1 / 3) * S₄ * H₄ = V) :
  1 * H₁ + 2 * H₂ + 3 * H₃ + 4 * H₄ = (3 * V) / k :=
by
  sorry

end NUMINAMATH_GPT_tetrahedron_distance_sum_l1447_144778


namespace NUMINAMATH_GPT_annual_concert_tickets_l1447_144711

theorem annual_concert_tickets (S NS : ℕ) (h1 : S + NS = 150) (h2 : 5 * S + 8 * NS = 930) : NS = 60 :=
by
  sorry

end NUMINAMATH_GPT_annual_concert_tickets_l1447_144711


namespace NUMINAMATH_GPT_find_m_l1447_144701

def g (n : ℤ) : ℤ :=
  if n % 2 = 1 then n + 5
  else if n % 3 = 0 then n / 3
  else n / 2

theorem find_m (m : ℤ) (h_odd : m % 2 = 1) (h_g : g (g (g m)) = 16) : m = 59 ∨ m = 91 :=
by sorry

end NUMINAMATH_GPT_find_m_l1447_144701


namespace NUMINAMATH_GPT_S7_eq_14_l1447_144773

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

variables (a : ℕ → ℤ) (S : ℕ → ℤ) (h_arith_seq : arithmetic_sequence a)
variables (h_a3 : a 3 = 0) (h_a6_plus_a7 : a 6 + a 7 = 14)

theorem S7_eq_14 : S 7 = 14 := sorry

end NUMINAMATH_GPT_S7_eq_14_l1447_144773


namespace NUMINAMATH_GPT_packet_b_average_height_l1447_144721

theorem packet_b_average_height (x y R_A R_B H_A H_B : ℝ)
  (h_RA : R_A = 2 * x + y)
  (h_RB : R_B = 3 * x - y)
  (h_x : x = 10)
  (h_y : y = 6)
  (h_HA : H_A = 192)
  (h_20percent : H_A = H_B + 0.20 * H_B) :
  H_B = 160 := 
sorry

end NUMINAMATH_GPT_packet_b_average_height_l1447_144721


namespace NUMINAMATH_GPT_complex_in_fourth_quadrant_l1447_144738

theorem complex_in_fourth_quadrant (m : ℝ) :
  (m^2 - 8*m + 15 > 0) ∧ (m^2 - 5*m - 14 < 0) →
  (-2 < m ∧ m < 3) ∨ (5 < m ∧ m < 7) :=
sorry

end NUMINAMATH_GPT_complex_in_fourth_quadrant_l1447_144738


namespace NUMINAMATH_GPT_sum_of_reciprocal_squares_leq_reciprocal_product_square_l1447_144717

theorem sum_of_reciprocal_squares_leq_reciprocal_product_square (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h_sum : a + b + c + d = 3) : 
  1 / a^2 + 1 / b^2 + 1 / c^2 + 1 / d^2 ≤ 1 / (a^2 * b^2 * c^2 * d^2) :=
sorry

end NUMINAMATH_GPT_sum_of_reciprocal_squares_leq_reciprocal_product_square_l1447_144717


namespace NUMINAMATH_GPT_fib_ratio_bound_l1447_144781

def fib : ℕ → ℕ
| 0     => 0
| 1     => 1
| (n+2) => fib (n+1) + fib n

theorem fib_ratio_bound {a b n : ℕ} (h1: b > 0) (h2: fib (n-1) > 0)
  (h3: (fib n) * b > (fib (n-1)) * a)
  (h4: (fib (n+1)) * b < (fib n) * a) :
  b ≥ fib (n+1) :=
sorry

end NUMINAMATH_GPT_fib_ratio_bound_l1447_144781


namespace NUMINAMATH_GPT_dolls_total_correct_l1447_144727

def Jazmin_dolls : Nat := 1209
def Geraldine_dolls : Nat := 2186
def total_dolls : Nat := Jazmin_dolls + Geraldine_dolls

theorem dolls_total_correct : total_dolls = 3395 := by
  sorry

end NUMINAMATH_GPT_dolls_total_correct_l1447_144727


namespace NUMINAMATH_GPT_series_sum_l1447_144709

noncomputable def sum_series : Real :=
  ∑' n: ℕ, (4 * (n + 1) + 2) / (3 : ℝ)^(n + 1)

theorem series_sum : sum_series = 3 := by
  sorry

end NUMINAMATH_GPT_series_sum_l1447_144709


namespace NUMINAMATH_GPT_sum_a2_a4_a6_l1447_144764

-- Define the arithmetic sequence with a positive common difference
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∃ (d : ℝ), d > 0 ∧ ∀ n, a (n + 1) = a n + d

-- Define that a_1 and a_7 are roots of the quadratic equation x^2 - 10x + 16 = 0
def roots_condition (a : ℕ → ℝ) : Prop :=
(a 1) * (a 7) = 16 ∧ (a 1) + (a 7) = 10

-- The main theorem we want to prove
theorem sum_a2_a4_a6 (a : ℕ → ℝ) (h1 : is_arithmetic_sequence a) (h2 : roots_condition a) :
  a 2 + a 4 + a 6 = 15 :=
sorry

end NUMINAMATH_GPT_sum_a2_a4_a6_l1447_144764


namespace NUMINAMATH_GPT_both_selected_prob_l1447_144787

def ram_prob : ℚ := 6 / 7
def ravi_prob : ℚ := 1 / 5

theorem both_selected_prob : ram_prob * ravi_prob = 6 / 35 := 
by
  sorry

end NUMINAMATH_GPT_both_selected_prob_l1447_144787


namespace NUMINAMATH_GPT_find_coordinates_l1447_144742

def A : Prod ℤ ℤ := (-3, 2)
def move_right (p : Prod ℤ ℤ) : Prod ℤ ℤ := (p.fst + 1, p.snd)
def move_down (p : Prod ℤ ℤ) : Prod ℤ ℤ := (p.fst, p.snd - 2)

theorem find_coordinates :
  move_down (move_right A) = (-2, 0) :=
by
  sorry

end NUMINAMATH_GPT_find_coordinates_l1447_144742


namespace NUMINAMATH_GPT_B_can_finish_work_in_6_days_l1447_144752

theorem B_can_finish_work_in_6_days :
  (A_work_alone : ℕ) → (A_work_before_B : ℕ) → (A_B_together : ℕ) → (B_days_alone : ℕ) → 
  (A_work_alone = 12) → (A_work_before_B = 3) → (A_B_together = 3) → B_days_alone = 6 :=
by
  intros A_work_alone A_work_before_B A_B_together B_days_alone
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_B_can_finish_work_in_6_days_l1447_144752


namespace NUMINAMATH_GPT_min_value_xyz_l1447_144789

theorem min_value_xyz (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h_prod : x * y * z = 8) : 
  x + 3 * y + 6 * z ≥ 18 :=
sorry

end NUMINAMATH_GPT_min_value_xyz_l1447_144789


namespace NUMINAMATH_GPT_remainder_2021_2025_mod_17_l1447_144703

theorem remainder_2021_2025_mod_17 : 
  (2021 * 2022 * 2023 * 2024 * 2025) % 17 = 0 :=
by 
  -- Proof omitted for brevity
  sorry

end NUMINAMATH_GPT_remainder_2021_2025_mod_17_l1447_144703


namespace NUMINAMATH_GPT_swimming_speed_l1447_144705

theorem swimming_speed (v : ℝ) (water_speed : ℝ) (swim_time : ℝ) (distance : ℝ) :
  water_speed = 8 →
  swim_time = 8 →
  distance = 16 →
  distance = (v - water_speed) * swim_time →
  v = 10 := 
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_swimming_speed_l1447_144705


namespace NUMINAMATH_GPT_symmetric_line_equation_l1447_144712

theorem symmetric_line_equation (x y : ℝ) : 
  3 * x - 4 * y + 5 = 0 → (3 * x + 4 * y - 5 = 0) :=
by
sorry

end NUMINAMATH_GPT_symmetric_line_equation_l1447_144712


namespace NUMINAMATH_GPT_number_of_6mb_pictures_l1447_144713

theorem number_of_6mb_pictures
    (n : ℕ)             -- initial number of pictures
    (size_old : ℕ)      -- size of old pictures in megabytes
    (size_new : ℕ)      -- size of new pictures in megabytes
    (total_capacity : ℕ)  -- total capacity of the memory card in megabytes
    (h1 : n = 3000)      -- given memory card can hold 3000 pictures
    (h2 : size_old = 8)  -- each old picture is 8 megabytes
    (h3 : size_new = 6)  -- each new picture is 6 megabytes
    (h4 : total_capacity = n * size_old)  -- total capacity calculated from old pictures
    : total_capacity / size_new = 4000 :=  -- the number of new pictures that can be held
by
  sorry

end NUMINAMATH_GPT_number_of_6mb_pictures_l1447_144713


namespace NUMINAMATH_GPT_total_profit_correct_l1447_144730

def natasha_money : ℤ := 60
def carla_money : ℤ := natasha_money / 3
def cosima_money : ℤ := carla_money / 2
def sergio_money : ℤ := (3 * cosima_money) / 2

def natasha_items : ℤ := 4
def carla_items : ℤ := 6
def cosima_items : ℤ := 5
def sergio_items : ℤ := 3

def natasha_profit_margin : ℚ := 0.10
def carla_profit_margin : ℚ := 0.15
def cosima_sergio_profit_margin : ℚ := 0.12

def natasha_item_cost : ℚ := (natasha_money : ℚ) / natasha_items
def carla_item_cost : ℚ := (carla_money : ℚ) / carla_items
def cosima_item_cost : ℚ := (cosima_money : ℚ) / cosima_items
def sergio_item_cost : ℚ := (sergio_money : ℚ) / sergio_items

def natasha_profit : ℚ := natasha_items * natasha_item_cost * natasha_profit_margin
def carla_profit : ℚ := carla_items * carla_item_cost * carla_profit_margin
def cosima_profit : ℚ := cosima_items * cosima_item_cost * cosima_sergio_profit_margin
def sergio_profit : ℚ := sergio_items * sergio_item_cost * cosima_sergio_profit_margin

def total_profit : ℚ := natasha_profit + carla_profit + cosima_profit + sergio_profit

theorem total_profit_correct : total_profit = 11.99 := 
by sorry

end NUMINAMATH_GPT_total_profit_correct_l1447_144730


namespace NUMINAMATH_GPT_work_duration_17_333_l1447_144771

def work_done (rate: ℚ) (days: ℕ) : ℚ := rate * days

def combined_work_done (rate1: ℚ) (rate2: ℚ) (days: ℕ) : ℚ :=
  (rate1 + rate2) * days

def total_work_done (rate1: ℚ) (rate2: ℚ) (rate3: ℚ) (days: ℚ) : ℚ :=
  (rate1 + rate2 + rate3) * days

noncomputable def total_days_work_last (rate_p rate_q rate_r: ℚ) : ℚ :=
  have work_p := 8 * rate_p
  have work_pq := combined_work_done rate_p rate_q 4
  have remaining_work := 1 - (work_p + work_pq)
  have days_all_together := remaining_work / (rate_p + rate_q + rate_r)
  8 + 4 + days_all_together

theorem work_duration_17_333 (rate_p rate_q rate_r: ℚ) : total_days_work_last rate_p rate_q rate_r = 17.333 :=
  by 
  have hp := 1/40
  have hq := 1/24
  have hr := 1/30
  sorry -- proof omitted

end NUMINAMATH_GPT_work_duration_17_333_l1447_144771


namespace NUMINAMATH_GPT_measure_angle_y_l1447_144748

theorem measure_angle_y
  (triangle_angles : ∀ {A B C : ℝ}, (A = 45 ∧ B = 45 ∧ C = 90) ∨ (A = 45 ∧ B = 90 ∧ C = 45) ∨ (A = 90 ∧ B = 45 ∧ C = 45))
  (p q : ℝ) (hpq : p = q) :
  ∃ (y : ℝ), y = 90 :=
by
  sorry

end NUMINAMATH_GPT_measure_angle_y_l1447_144748


namespace NUMINAMATH_GPT_prob_B_draws_given_A_draws_black_fairness_l1447_144777

noncomputable def event_A1 : Prop := true  -- A draws the red ball
noncomputable def event_A2 : Prop := true  -- B draws the red ball
noncomputable def event_A3 : Prop := true  -- C draws the red ball

noncomputable def prob_A1 : ℝ := 1 / 3
noncomputable def prob_not_A1 : ℝ := 2 / 3
noncomputable def prob_A2_given_not_A1 : ℝ := 1 / 2

theorem prob_B_draws_given_A_draws_black : (prob_not_A1 * prob_A2_given_not_A1) / prob_not_A1 = 1 / 2 := by
  sorry

theorem fairness :
  let prob_A1 := 1 / 3
  let prob_A2 := prob_not_A1 * prob_A2_given_not_A1
  let prob_A3 := prob_not_A1 * prob_A2_given_not_A1 * 1
  prob_A1 = prob_A2 ∧ prob_A2 = prob_A3 := by
  sorry

end NUMINAMATH_GPT_prob_B_draws_given_A_draws_black_fairness_l1447_144777


namespace NUMINAMATH_GPT_reflected_point_correct_l1447_144776

-- Defining the original point coordinates
def original_point : ℝ × ℝ := (3, -5)

-- Defining the transformation function
def reflect_across_y_axis (point : ℝ × ℝ) : ℝ × ℝ :=
  (-point.1, point.2)

-- Proving the point after reflection is as expected
theorem reflected_point_correct : reflect_across_y_axis original_point = (-3, -5) :=
by
  sorry

end NUMINAMATH_GPT_reflected_point_correct_l1447_144776


namespace NUMINAMATH_GPT_complex_number_purely_imaginary_l1447_144762

theorem complex_number_purely_imaginary (m : ℝ) 
  (h1 : m^2 - 5 * m + 6 = 0) 
  (h2 : m^2 - 3 * m ≠ 0) : 
  m = 2 :=
sorry

end NUMINAMATH_GPT_complex_number_purely_imaginary_l1447_144762


namespace NUMINAMATH_GPT_earnings_correct_l1447_144732

-- Define the initial number of roses, the number of roses left, and the price per rose.
def initial_roses : ℕ := 13
def roses_left : ℕ := 4
def price_per_rose : ℕ := 4

-- Calculate the number of roses sold.
def roses_sold : ℕ := initial_roses - roses_left

-- Calculate the total earnings.
def earnings : ℕ := roses_sold * price_per_rose

-- Prove that the earnings are 36 dollars.
theorem earnings_correct : earnings = 36 := by
  sorry

end NUMINAMATH_GPT_earnings_correct_l1447_144732


namespace NUMINAMATH_GPT_sum_evaluation_l1447_144720

theorem sum_evaluation : 5 * 399 + 4 * 399 + 3 * 399 + 398 = 5186 :=
by
  sorry

end NUMINAMATH_GPT_sum_evaluation_l1447_144720


namespace NUMINAMATH_GPT_james_veg_consumption_l1447_144728

-- Define the given conditions in Lean
def asparagus_per_day : ℝ := 0.25
def broccoli_per_day : ℝ := 0.25
def days_in_week : ℝ := 7
def weeks : ℝ := 2
def kale_per_week : ℝ := 3

-- Define the amount of vegetables (initial, doubled, and added kale)
def initial_veg_per_day := asparagus_per_day + broccoli_per_day
def initial_veg_per_week := initial_veg_per_day * days_in_week
def double_veg_per_week := initial_veg_per_week * weeks
def total_veg_per_week_after_kale := double_veg_per_week + kale_per_week

-- Statement of the proof problem
theorem james_veg_consumption :
  total_veg_per_week_after_kale = 10 := by 
  sorry

end NUMINAMATH_GPT_james_veg_consumption_l1447_144728


namespace NUMINAMATH_GPT_imaginary_part_z_is_correct_l1447_144733

open Complex

noncomputable def problem_conditions (z : ℂ) : Prop :=
  (3 - 4 * Complex.I) * z = Complex.abs (4 + 3 * Complex.I)

theorem imaginary_part_z_is_correct (z : ℂ) (hz : problem_conditions z) :
  z.im = 4 / 5 :=
sorry

end NUMINAMATH_GPT_imaginary_part_z_is_correct_l1447_144733


namespace NUMINAMATH_GPT_john_days_off_l1447_144734

def streams_per_week (earnings_per_week : ℕ) (rate_per_hour : ℕ) : ℕ := earnings_per_week / rate_per_hour

def streaming_sessions (hours_per_week : ℕ) (hours_per_session : ℕ) : ℕ := hours_per_week / hours_per_session

def days_off_per_week (total_days : ℕ) (streaming_days : ℕ) : ℕ := total_days - streaming_days

theorem john_days_off (hours_per_session : ℕ) (hourly_rate : ℕ) (weekly_earnings : ℕ) (total_days : ℕ) :
  hours_per_session = 4 → 
  hourly_rate = 10 → 
  weekly_earnings = 160 → 
  total_days = 7 → 
  days_off_per_week total_days (streaming_sessions (streams_per_week weekly_earnings hourly_rate) hours_per_session) = 3 := 
by
  intros
  sorry

end NUMINAMATH_GPT_john_days_off_l1447_144734


namespace NUMINAMATH_GPT_quadratic_pos_in_interval_l1447_144791

theorem quadratic_pos_in_interval (m n : ℤ)
  (h2014 : (2014:ℤ)^2 + m * 2014 + n > 0)
  (h2015 : (2015:ℤ)^2 + m * 2015 + n > 0) :
  ∀ x : ℝ, 2014 ≤ x ∧ x ≤ 2015 → (x^2 + (m:ℝ) * x + (n:ℝ)) > 0 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_pos_in_interval_l1447_144791


namespace NUMINAMATH_GPT_train_A_length_l1447_144768

theorem train_A_length
  (speed_A : ℕ)
  (speed_B : ℕ)
  (time_to_cross : ℕ)
  (len_A : ℕ)
  (h1 : speed_A = 54) 
  (h2 : speed_B = 36) 
  (h3 : time_to_cross = 15)
  (h4 : len_A = (speed_A + speed_B) * 1000 / 3600 * time_to_cross) :
  len_A = 375 :=
sorry

end NUMINAMATH_GPT_train_A_length_l1447_144768


namespace NUMINAMATH_GPT_correct_choice_l1447_144743

-- Define the structures and options
inductive Structure
| Sequential
| Conditional
| Loop
| Module

def option_A : List Structure :=
  [Structure.Sequential, Structure.Module, Structure.Conditional]

def option_B : List Structure :=
  [Structure.Sequential, Structure.Loop, Structure.Module]

def option_C : List Structure :=
  [Structure.Sequential, Structure.Conditional, Structure.Loop]

def option_D : List Structure :=
  [Structure.Module, Structure.Conditional, Structure.Loop]

-- Define the correct structures
def basic_structures : List Structure :=
  [Structure.Sequential, Structure.Conditional, Structure.Loop]

-- The theorem statement
theorem correct_choice : option_C = basic_structures :=
  by
    sorry  -- Proof would go here

end NUMINAMATH_GPT_correct_choice_l1447_144743


namespace NUMINAMATH_GPT_L_shaped_region_area_l1447_144790

noncomputable def area_L_shaped_region (length full_width : ℕ) (sub_length sub_width : ℕ) : ℕ :=
  let area_full_rect := length * full_width
  let small_width := length - sub_length
  let small_height := full_width - sub_width
  let area_small_rect := small_width * small_height
  area_full_rect - area_small_rect

theorem L_shaped_region_area :
  area_L_shaped_region 10 7 3 4 = 49 :=
by sorry

end NUMINAMATH_GPT_L_shaped_region_area_l1447_144790


namespace NUMINAMATH_GPT_forty_percent_of_number_l1447_144746

/--
Given that (1/4) * (1/3) * (2/5) * N = 30, prove that 0.40 * N = 360.
-/
theorem forty_percent_of_number {N : ℝ} (h : (1/4 : ℝ) * (1/3) * (2/5) * N = 30) : 0.40 * N = 360 := 
by
  sorry

end NUMINAMATH_GPT_forty_percent_of_number_l1447_144746


namespace NUMINAMATH_GPT_point_inside_circle_implies_range_l1447_144750

theorem point_inside_circle_implies_range (a : ℝ) : 
  (1 - a)^2 + (1 + a)^2 < 4 → -1 < a ∧ a < 1 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_point_inside_circle_implies_range_l1447_144750


namespace NUMINAMATH_GPT_fans_received_all_offers_l1447_144704

theorem fans_received_all_offers :
  let hotdog_freq := 90
  let soda_freq := 45
  let popcorn_freq := 60
  let stadium_capacity := 4500
  let lcm_freq := Nat.lcm (Nat.lcm hotdog_freq soda_freq) popcorn_freq
  (stadium_capacity / lcm_freq) = 25 :=
by
  sorry

end NUMINAMATH_GPT_fans_received_all_offers_l1447_144704


namespace NUMINAMATH_GPT_golden_triangle_expression_l1447_144736

noncomputable def t : ℝ := (Real.sqrt 5 - 1) / 2

theorem golden_triangle_expression :
  t = (Real.sqrt 5 - 1) / 2 →
  (1 - 2 * (Real.sin (27 * Real.pi / 180))^2) / (2 * t * Real.sqrt (4 - t^2)) = 1 / 4 :=
by
  intro h_t
  have h1 : t = (Real.sqrt 5 - 1) / 2 := h_t
  sorry

end NUMINAMATH_GPT_golden_triangle_expression_l1447_144736


namespace NUMINAMATH_GPT_equal_share_is_168_l1447_144715

namespace StrawberryProblem

def brother_baskets : ℕ := 3
def strawberries_per_basket : ℕ := 15
def brother_strawberries : ℕ := brother_baskets * strawberries_per_basket

def kimberly_multiplier : ℕ := 8
def kimberly_strawberries : ℕ := kimberly_multiplier * brother_strawberries

def parents_difference : ℕ := 93
def parents_strawberries : ℕ := kimberly_strawberries - parents_difference

def total_strawberries : ℕ := kimberly_strawberries + brother_strawberries + parents_strawberries
def total_people : ℕ := 4

def equal_share : ℕ := total_strawberries / total_people

theorem equal_share_is_168 :
  equal_share = 168 := by
  -- We state that for the given problem conditions,
  -- the total number of strawberries divided equally among the family members results in 168 strawberries per person.
  sorry

end StrawberryProblem

end NUMINAMATH_GPT_equal_share_is_168_l1447_144715


namespace NUMINAMATH_GPT_axis_symmetry_shifted_graph_l1447_144755

open Real

theorem axis_symmetry_shifted_graph :
  ∀ k : ℤ, ∃ x : ℝ, (y = 2 * sin (2 * x)) ∧
  y = 2 * sin (2 * (x + π / 12)) ↔
  x = k * π / 2 + π / 6 :=
sorry

end NUMINAMATH_GPT_axis_symmetry_shifted_graph_l1447_144755


namespace NUMINAMATH_GPT_two_digit_sum_condition_l1447_144749

theorem two_digit_sum_condition (x y : ℕ) (hx : 1 ≤ x) (hx9 : x ≤ 9) (hy : 0 ≤ y) (hy9 : y ≤ 9)
    (h : (x + 1) + (y + 2) - 10 = 2 * (x + y)) :
    (x = 6 ∧ y = 8) ∨ (x = 5 ∧ y = 9) :=
sorry

end NUMINAMATH_GPT_two_digit_sum_condition_l1447_144749


namespace NUMINAMATH_GPT_pyramid_structure_l1447_144725

variables {d e f a b c h i j g : ℝ}

theorem pyramid_structure (h_val : h = 16)
                         (i_val : i = 48)
                         (j_val : j = 72)
                         (g_val : g = 8)
                         (d_def : d = b * a)
                         (e_def1 : e = b * c) 
                         (e_def2 : e = d * a)
                         (f_def : f = c * a)
                         (h_def : h = d * b)
                         (i_def : i = d * a)
                         (j_def : j = e * c)
                         (g_def : g = f * c) : 
   a = 3 ∧ b = 1 ∧ c = 1.5 :=
by sorry

end NUMINAMATH_GPT_pyramid_structure_l1447_144725


namespace NUMINAMATH_GPT_avg_age_of_coaches_l1447_144799

theorem avg_age_of_coaches (n_girls n_boys n_coaches : ℕ)
  (avg_age_girls avg_age_boys avg_age_members : ℕ)
  (h_girls : n_girls = 30)
  (h_boys : n_boys = 15)
  (h_coaches : n_coaches = 5)
  (h_avg_age_girls : avg_age_girls = 18)
  (h_avg_age_boys : avg_age_boys = 19)
  (h_avg_age_members : avg_age_members = 20) :
  (n_girls * avg_age_girls + n_boys * avg_age_boys + n_coaches * 35) / (n_girls + n_boys + n_coaches) = avg_age_members :=
by sorry

end NUMINAMATH_GPT_avg_age_of_coaches_l1447_144799


namespace NUMINAMATH_GPT_total_cost_l1447_144772

/-- Sam initially has s yellow balloons.
He gives away a of these balloons to Fred.
Mary has m yellow balloons.
Each balloon costs c dollars.
Determine the total cost for the remaining balloons that Sam and Mary jointly have.
Given: s = 6.0, a = 5.0, m = 7.0, c = 9.0 dollars.
Expected result: the total cost is 72.0 dollars.
-/
theorem total_cost (s a m c : ℝ) (h_s : s = 6.0) (h_a : a = 5.0) (h_m : m = 7.0) (h_c : c = 9.0) :
  (s - a + m) * c = 72.0 := 
by
  rw [h_s, h_a, h_m, h_c]
  -- At this stage, the proof would involve showing the expression is 72.0, but since no proof is required:
  sorry

end NUMINAMATH_GPT_total_cost_l1447_144772


namespace NUMINAMATH_GPT_number_of_girl_students_l1447_144770

theorem number_of_girl_students (total_third_graders : ℕ) (boy_students : ℕ) (girl_students : ℕ) 
  (h1 : total_third_graders = 123) (h2 : boy_students = 66) (h3 : total_third_graders = boy_students + girl_students) :
  girl_students = 57 :=
by
  sorry

end NUMINAMATH_GPT_number_of_girl_students_l1447_144770


namespace NUMINAMATH_GPT_minimum_area_of_square_on_parabola_l1447_144753

theorem minimum_area_of_square_on_parabola :
  ∃ (A B C : ℝ × ℝ), 
  (∃ (x₁ x₂ x₃ : ℝ), (A = (x₁, x₁^2)) ∧ (B = (x₂, x₂^2)) ∧ (C = (x₃, x₃^2)) 
  ∧ x₁ < x₂ ∧ x₂ < x₃ 
  ∧ ∀ S : ℝ, (S = (1 + (x₃ + x₂)^2) * ((x₂ - x₃) - (x₃ - x₂))^2) → S ≥ 2) :=
sorry

end NUMINAMATH_GPT_minimum_area_of_square_on_parabola_l1447_144753


namespace NUMINAMATH_GPT_quilt_shaded_fraction_l1447_144788

theorem quilt_shaded_fraction :
  let total_squares := 16
  let shaded_squares := 8
  let fully_shaded := 4
  let half_shaded := 4
  let shaded_area := fully_shaded + half_shaded * 1 / 2
  shaded_area / total_squares = 3 / 8 :=
by
  sorry

end NUMINAMATH_GPT_quilt_shaded_fraction_l1447_144788


namespace NUMINAMATH_GPT_quadratic_zeros_l1447_144795

theorem quadratic_zeros : ∀ x : ℝ, (x = 3 ∨ x = -1) ↔ (x^2 - 2*x - 3 = 0) := by
  intro x
  sorry

end NUMINAMATH_GPT_quadratic_zeros_l1447_144795


namespace NUMINAMATH_GPT_price_per_glass_first_day_l1447_144761

variables (O G : ℝ) (P1 : ℝ)

theorem price_per_glass_first_day (H1 : G * P1 = 1.5 * G * 0.40) : 
  P1 = 0.60 :=
by sorry

end NUMINAMATH_GPT_price_per_glass_first_day_l1447_144761


namespace NUMINAMATH_GPT_train_cross_time_l1447_144707

noncomputable def speed_kmh := 72
noncomputable def speed_mps : ℝ := speed_kmh * (1000 / 3600)
noncomputable def length_train := 180
noncomputable def length_bridge := 270
noncomputable def total_distance := length_train + length_bridge
noncomputable def time_to_cross := total_distance / speed_mps

theorem train_cross_time :
  time_to_cross = 22.5 := 
sorry

end NUMINAMATH_GPT_train_cross_time_l1447_144707


namespace NUMINAMATH_GPT_problem1_problem2_l1447_144786

theorem problem1 : (- (2 : ℤ) ^ 3 / 8 - (1 / 4 : ℚ) * ((-2)^2)) = -2 :=
by {
    sorry
}

theorem problem2 : ((- (1 / 12 : ℚ) - 1 / 16 + 3 / 4 - 1 / 6) * -48) = -21 :=
by {
    sorry
}

end NUMINAMATH_GPT_problem1_problem2_l1447_144786


namespace NUMINAMATH_GPT_sin_identity_l1447_144766

theorem sin_identity (α : ℝ) (hα : α = Real.pi / 7) : 
  1 / Real.sin α = 1 / Real.sin (2 * α) + 1 / Real.sin (3 * α) := 
  by 
  sorry

end NUMINAMATH_GPT_sin_identity_l1447_144766


namespace NUMINAMATH_GPT_simplify_polynomial_l1447_144793

theorem simplify_polynomial (s : ℝ) :
  (2 * s ^ 2 + 5 * s - 3) - (2 * s ^ 2 + 9 * s - 6) = -4 * s + 3 :=
by 
  sorry

end NUMINAMATH_GPT_simplify_polynomial_l1447_144793


namespace NUMINAMATH_GPT_number_of_even_factors_l1447_144784

theorem number_of_even_factors {n : ℕ} (h : n = 2^4 * 3^3 * 7) : 
  ∃ (count : ℕ), count = 32 ∧ ∀ k, (k ∣ n) → k % 2 = 0 → count = 32 :=
by
  sorry

end NUMINAMATH_GPT_number_of_even_factors_l1447_144784


namespace NUMINAMATH_GPT_employee_salaries_l1447_144780

theorem employee_salaries 
  (x y z : ℝ)
  (h1 : x + y + z = 638)
  (h2 : x = 1.20 * y)
  (h3 : z = 0.80 * y) :
  x = 255.20 ∧ y = 212.67 ∧ z = 170.14 :=
sorry

end NUMINAMATH_GPT_employee_salaries_l1447_144780


namespace NUMINAMATH_GPT_log_expression_simplification_l1447_144756

open Real

theorem log_expression_simplification (p q r s t z : ℝ) :
  log (p / q) + log (q / r) + log (r / s) - log (p * t / (s * z)) = log (z / t) :=
  sorry

end NUMINAMATH_GPT_log_expression_simplification_l1447_144756


namespace NUMINAMATH_GPT_rear_revolutions_l1447_144729

variable (r_r : ℝ)  -- radius of the rear wheel
variable (r_f : ℝ)  -- radius of the front wheel
variable (n_f : ℕ)  -- number of revolutions of the front wheel
variable (n_r : ℕ)  -- number of revolutions of the rear wheel

-- Condition: radius of the front wheel is 2 times the radius of the rear wheel.
axiom front_radius : r_f = 2 * r_r

-- Condition: the front wheel makes 10 revolutions.
axiom front_revolutions : n_f = 10

-- Theorem statement to prove
theorem rear_revolutions : n_r = 20 :=
sorry

end NUMINAMATH_GPT_rear_revolutions_l1447_144729


namespace NUMINAMATH_GPT_find_y_given_conditions_l1447_144716

theorem find_y_given_conditions (x y : ℝ) (h₁ : 3 * x^2 = y - 6) (h₂ : x = 4) : y = 54 :=
  sorry

end NUMINAMATH_GPT_find_y_given_conditions_l1447_144716
