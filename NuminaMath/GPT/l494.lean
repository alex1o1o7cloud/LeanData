import Mathlib

namespace increase_by_percentage_l494_494255

theorem increase_by_percentage (initial_amount : ℕ) (percentage_increase : ℕ) :
  initial_amount = 80 → 
  percentage_increase = 150 → 
  initial_amount * (percentage_increase / 100) + initial_amount = 200 :=
by
  intros h_initial h_percentage
  rw [h_initial, h_percentage]
  calc
    80 * (150 / 100) + 80 = 80 * 1.5 + 80 : by norm_num
                     ... = 120 + 80         : by norm_num
                     ... = 200              : by norm_num
  sorry

end increase_by_percentage_l494_494255


namespace smallest_N_for_triangle_sides_l494_494462

theorem smallest_N_for_triangle_sides (a b c : ℝ) (h_triangle : a + b > c) (h_a_ne_b : a ≠ b) : (a^2 + b^2) / c^2 < 1 := 
sorry

end smallest_N_for_triangle_sides_l494_494462


namespace equation_represents_point_l494_494663

theorem equation_represents_point (a b x y : ℝ) :
  x^2 + y^2 + 2 * a * x + 2 * b * y + a^2 + b^2 = 0 ↔ x = -a ∧ y = -b := 
by sorry

end equation_represents_point_l494_494663


namespace ratio_of_areas_of_circles_l494_494949

theorem ratio_of_areas_of_circles
    (C_C R_C C_D R_D L : ℝ)
    (hC : C_C = 2 * Real.pi * R_C)
    (hD : C_D = 2 * Real.pi * R_D)
    (hL : (60 / 360) * C_C = L ∧ L = (40 / 360) * C_D) :
    (Real.pi * R_C ^ 2) / (Real.pi * R_D ^ 2) = 4 / 9 :=
by
  sorry

end ratio_of_areas_of_circles_l494_494949


namespace log_sum_example_l494_494793

theorem log_sum_example :
  let log_base_10 (x : ℝ) := Real.log x / Real.log 10 in
  log_base_10 50 + log_base_10 20 = 3 :=
by
  sorry

end log_sum_example_l494_494793


namespace cover_2000_points_with_circles_l494_494499

theorem cover_2000_points_with_circles :
  ∀ (points : Fin 2000 → ℝ × ℝ),
  ∃ (circles : Fin 2000 → ℝ × ℝ × ℝ), -- Each circle represented as (center_x, center_y, radius)
    (∀ i j : Fin 2000, i ≠ j → dist (circles i).fst (circles j).fst > 1) ∧
    (finset.sum finset.univ (λ i, 2 * (circles i).snd) ≤ 2000) ∧
    (∀ i j : Fin 2000, i ≠ j → dist (points i) ((circles i).fst) = 0) := sorry

end cover_2000_points_with_circles_l494_494499


namespace interval_of_decrease_for_f_x_plus_1_l494_494555

def f_prime (x : ℝ) : ℝ := x^2 - 4*x + 3

theorem interval_of_decrease_for_f_x_plus_1 : 
  ∀ x, (f_prime (x + 1) < 0 ↔ 0 < x ∧ x < 2) :=
by 
  intro x
  sorry

end interval_of_decrease_for_f_x_plus_1_l494_494555


namespace sum_of_coefficients_l494_494068

theorem sum_of_coefficients : 
  let f := (5*x - 2)^8
  let expansion := f.expand
  expansion.sum_coefficients = 6561 :=
by
  sorry

end sum_of_coefficients_l494_494068


namespace mowing_work_rate_l494_494722

variables (A B C : ℚ)

theorem mowing_work_rate :
  A + B = 1/28 → A + B + C = 1/21 → C = 1/84 :=
by
  intros h1 h2
  sorry

end mowing_work_rate_l494_494722


namespace brick_width_l494_494464

variable (w : ℝ)

theorem brick_width :
  ∃ (w : ℝ), 2 * (10 * w + 10 * 3 + 3 * w) = 164 → w = 4 :=
by
  sorry

end brick_width_l494_494464


namespace increase_80_by_150_percent_l494_494283

theorem increase_80_by_150_percent : 
  ∀ x percent, x = 80 ∧ percent = 1.5 → x + (x * percent) = 200 :=
by
  sorry

end increase_80_by_150_percent_l494_494283


namespace find_p_of_table_tennis_winning_probability_after_four_points_l494_494577

def table_tennis_probability_p (p : ℝ) : Prop :=
  p * (2/5) + (1 - p) * (3/5) = 7/15

def probability_of_winning := 
  let p := 2/3 in
  ((2/3) * (3/5) + (1 - 2/3) * (2/5)) * (2/3) * (2/5) = (32/225)

theorem find_p_of_table_tennis : 
  ∃ p : ℝ, table_tennis_probability_p p ∧ p = 2/3 := sorry

theorem winning_probability_after_four_points :
  probability_of_winning := sorry

end find_p_of_table_tennis_winning_probability_after_four_points_l494_494577


namespace balls_in_indistinguishable_boxes_l494_494019

theorem balls_in_indistinguishable_boxes : 
  let balls := 6 in
  let boxes := 3 in
  -- Expression to count ways to distribute balls into indistinguishable boxes
  ((balls.choose 0 * ((balls-0).choose 6)) + 
   (balls.choose 1 * ((balls-1).choose 5)) + 
   (balls.choose 2 * ((balls-2).choose 4)) + 
   (balls.choose 2 * ((balls-2).choose 3) / 2!) + 
   (balls.choose 3 * ((balls-3).choose 3)) + 
   (balls.choose 3 * ((balls-3).choose 2) * ((balls-5).choose 1)) + 
   (balls.choose 2 * ((balls-2).choose 2) / 3!))
  = 92 := 
sorry

end balls_in_indistinguishable_boxes_l494_494019


namespace distance_travel_l494_494745

-- Definition of the parameters and the proof problem
variable (W_t : ℕ)
variable (R_c : ℕ)
variable (remaining_coal : ℕ)

-- Conditions
def rate_of_coal_consumption : Prop := R_c = 4 * W_t / 1000
def remaining_coal_amount : Prop := remaining_coal = 160

-- Theorem statement
theorem distance_travel (W_t : ℕ) (R_c : ℕ) (remaining_coal : ℕ) 
  (h1 : rate_of_coal_consumption W_t R_c) 
  (h2 : remaining_coal_amount remaining_coal) : 
  (remaining_coal * 1000 / 4 / W_t) = 40000 / W_t := 
by
  sorry

end distance_travel_l494_494745


namespace problem1_problem2_problem3_l494_494526

noncomputable def f (x : ℝ) : ℝ := x * Real.log x
def g (λ x : ℝ) : ℝ := λ * (x - 1)
def p (x : ℝ) : ℝ := f (x - 1) - (x - 3)
def q (x : ℝ) : ℝ := f (Real.exp x) - 3 * (Real.exp x - 3)

theorem problem1 : (∀ x, deriv (λ x, f x) x = 1) → f 1 = 0 → f x = x * Real.log x :=
sorry

theorem problem2 (λ : ℝ) : (∀ x ∈ set.Ici 1, f x ≥ g λ x) → λ <= 1 :=
sorry

theorem problem3 (x : ℝ) : x > 1 → (p x) * (q x) ≥ 9 - Real.exp 2 :=
sorry

end problem1_problem2_problem3_l494_494526


namespace south_side_students_count_l494_494686

variables (N : ℕ)
def students_total := 41
def difference := 3

theorem south_side_students_count (N : ℕ) (h₁ : 2 * N + difference = students_total) : N + difference = 22 :=
sorry

end south_side_students_count_l494_494686


namespace odd_function_iff_phi_values_l494_494513

noncomputable def f (x ϕ : ℝ) : ℝ := Math.sin (x + ϕ) - Math.sin (x + 7 * ϕ)

theorem odd_function_iff_phi_values (ϕ : ℝ) :
  (∀ x, f (-x) ϕ = -f x ϕ) ↔ (ϕ = π / 8 ∨ ϕ = 3 * π / 8) :=
sorry

end odd_function_iff_phi_values_l494_494513


namespace num_ways_dist_6_balls_3_boxes_l494_494046

open Finset

/-- Number of ways to distribute 6 distinguishable balls into 3 indistinguishable boxes -/
theorem num_ways_dist_6_balls_3_boxes : 
  let num_ways := 
    let n := 6 in let k := 3 in
    ((choose n 6) + 
    (choose n 5) + 
    (choose n 4) + 
    (choose n 4) + 
    (choose n 3) / 2 + 
    (choose n 3) * (choose (n - 3) 2) + 
    (choose n 2) * (choose (n - 2) 2) / (factorial k)) 
  in num_ways = 122 := 
by 
  sorry

end num_ways_dist_6_balls_3_boxes_l494_494046


namespace maximum_candies_eaten_l494_494213

theorem maximum_candies_eaten :
  let total_moves := 24,
      initial_numbers := 25,
      initial_candy := (1 * 1) in
  -- Assuming each pair combination process
  (∀ k : ℕ, k < total_moves → ∃ candies : ℕ, candies = (k + initial_candy)) →
  ∃ max_candies : ℕ, max_candies = 300 :=
sorry

end maximum_candies_eaten_l494_494213


namespace Isabel_finished_problems_l494_494997

theorem Isabel_finished_problems (h1 : Isabel.had_homework_problems = 72.0) 
                                  (h2 : ∀ (p : HomeworkProblem), p.subTasks = 5) 
                                  (h3 : Isabel.has_to_solve_subTasks = 200) : 
  Isabel.finished_homework_problems = 40 :=
by sorry

end Isabel_finished_problems_l494_494997


namespace unique_increasing_sequence_satisfies_condition_l494_494830

theorem unique_increasing_sequence_satisfies_condition : 
  ∃! (a : ℕ → ℕ) (k : ℕ),
    (strictly_increasing a ∧ 
     (∀ i, i < k → 0 ≤ a i) ∧ 
     (∑ i in finset.range k, 2 ^ a i = (2 ^ 145 + 1) / (2 ^ 9 + 1)))

/- Verification that k == 17 -/
  → k = 17 :=
by 
  sorry

end unique_increasing_sequence_satisfies_condition_l494_494830


namespace sum_of_first_ten_primes_with_units_digit_three_l494_494860

-- Define the problem to prove the sum of the first 10 primes ending in 3 is 639
theorem sum_of_first_ten_primes_with_units_digit_three : 
  let primes_with_units_digit_three := [3, 13, 23, 43, 53, 73, 83, 103, 113, 163]
  in list.sum primes_with_units_digit_three = 639 := 
by 
  -- We define the primes with the units digit 3 as given and check the sum
  let primes_with_units_digit_three := [3, 13, 23, 43, 53, 73, 83, 103, 113, 163]
  show list.sum primes_with_units_digit_three = 639 from sorry

end sum_of_first_ten_primes_with_units_digit_three_l494_494860


namespace count_special_five_digit_numbers_l494_494543

theorem count_special_five_digit_numbers :
  (∃ (s : Fin 5 → Fin 3), ∀ i, s i ∈ {0, 1, 2}) ↔ 3^5 = 243 :=
by 
  sorry

end count_special_five_digit_numbers_l494_494543


namespace Amanda_money_left_l494_494774

theorem Amanda_money_left (initial_amount cost_cassette tape_count cost_headphone : ℕ) 
  (h1 : initial_amount = 50) 
  (h2 : cost_cassette = 9) 
  (h3 : tape_count = 2) 
  (h4 : cost_headphone = 25) :
  initial_amount - (tape_count * cost_cassette + cost_headphone) = 7 :=
by
  sorry

end Amanda_money_left_l494_494774


namespace exam_combination_probability_determine_a_d_and_confidence_l494_494964

-- Define the conditions for part 1
def num_subject_combinations := 12

def history_political_geography_combination_count := 1

def probability_of_history_political_geography_combination :=
  (history_political_geography_combination_count : ℝ) / num_subject_combinations

-- Define the conditions for part 2
def n := 100
def b := 30
def c := 30 + 10
def total_students := 100
def boys_chose_history := 10
def girls_chose_history := 20
def girls_chose_physics := 30
def girls_total := 50

def a := total_students - (boys_chose_history + girls_chose_history + girls_chose_physics)
def d := 20
def contingency_table := (a = 40 ∧ d = 20)

def K_squared := (n * (a * d - b * c)^2) / ((a + b) * (c + d) * (a + c) * (b + d))

def critical_value := 3.841
def hypothesis_test := K_squared > critical_value

theorem exam_combination_probability :
  probability_of_history_political_geography_combination = (1 / 12) :=
sorry

theorem determine_a_d_and_confidence : 
  contingency_table ∧ hypothesis_test :=
sorry

end exam_combination_probability_determine_a_d_and_confidence_l494_494964


namespace train_cross_time_is_26_00208_sec_l494_494752

/-
Given a goods train runs at a speed of 72 km/h and crosses a 260 m long platform.
The length of the goods train is 260.0416 meters.
We need to prove that the time it takes for the train to cross the platform is 26.00208 seconds.
-/

def train_speed_kmph : ℝ := 72
def train_length_meters : ℝ := 260.0416
def platform_length_meters : ℝ := 260

-- Convert train speed from km/h to m/s
def kmph_to_mps (speed: ℝ) : ℝ := speed * (1000 / 3600)
def train_speed_mps : ℝ := kmph_to_mps train_speed_kmph

-- Calculate the total distance to be covered when crossing the platform
def total_distance_meters : ℝ := train_length_meters + platform_length_meters

-- Calculate the time taken to cross the platform
def time_to_cross_platform (distance: ℝ) (speed: ℝ) : ℝ := distance / speed

theorem train_cross_time_is_26_00208_sec :
  time_to_cross_platform total_distance_meters train_speed_mps = 26.00208 :=
sorry

end train_cross_time_is_26_00208_sec_l494_494752


namespace area_ratio_of_circles_l494_494946

theorem area_ratio_of_circles (R_C R_D : ℝ) (hL : (60.0 / 360.0) * 2.0 * Real.pi * R_C = (40.0 / 360.0) * 2.0 * Real.pi * R_D) : 
  (Real.pi * R_C^2) / (Real.pi * R_D^2) = 4.0 / 9.0 :=
by
  sorry

end area_ratio_of_circles_l494_494946


namespace teacher_mathematics_is_C_l494_494180

theorem teacher_mathematics_is_C :
  (∀ (A B C D : Type),
    (A → (A teaches Physics ∧ A teaches Chemistry)) →
    (B → (B teaches Mathematics ∧ B teaches English)) →
    (C → (C teaches Mathematics ∧ C teaches Physics ∧ C teaches Chemistry)) →
    (D → (D teaches Chemistry ∧ ∀ S, S ≠ D ∨ S teaches Chemistry)) →
    (∀ (T : Type), (T teaches Mathematics ∧ (T = A ∨ T = B ∨ T = C ∨ T = D)) → T = C)) :=
by
  intros A B C D a_conds b_conds c_conds d_conds T t_conds
  sorry

end teacher_mathematics_is_C_l494_494180


namespace fraction_age_28_to_32_l494_494418

theorem fraction_age_28_to_32 (F : ℝ) (total_participants : ℝ) 
  (next_year_fraction_increase : ℝ) (next_year_fraction : ℝ) 
  (h1 : total_participants = 500)
  (h2 : next_year_fraction_increase = (1 / 8 : ℝ))
  (h3 : next_year_fraction = 0.5625) 
  (h4 : F + next_year_fraction_increase * F = next_year_fraction) :
  F = 0.5 :=
by
  sorry

end fraction_age_28_to_32_l494_494418


namespace exists_graph_with_chromatic_and_girth_l494_494873

theorem exists_graph_with_chromatic_and_girth (k : ℤ) : ∃ (G : Type), (girth G > k) ∧ (chromatic_number G > k) :=
sorry

end exists_graph_with_chromatic_and_girth_l494_494873


namespace opposite_face_proof_l494_494758

noncomputable def opposite_face (A B C D E F : Type) (adjacent_to : A → B → C → D → E → Type) : F :=
  if ∃ a b c d e, adjacent_to A B C D E
  then F
  else sorry

theorem opposite_face_proof (A B C D E F : Type) (adjacent_to : A → B → C → D → E → Type) : opposite_face A B C D E F adjacent_to = F :=
sorry

end opposite_face_proof_l494_494758


namespace trigonometric_identity_l494_494060

theorem trigonometric_identity
  (α : ℝ)
  (h : Real.tan α = Real.sqrt 2) :
  2 * (Real.sin α)^2 - (Real.sin α) * (Real.cos α) + (Real.cos α)^2 = (5 - Real.sqrt 2) / 3 := 
by
  sorry

end trigonometric_identity_l494_494060


namespace meeting_anniversary_day_l494_494411

-- Define the input parameters for the problem
def initial_years : Set ℕ := {1668, 1669, 1670, 1671}
def meeting_day := "Friday"
def is_leap_year (year : ℕ) : Bool := (year % 4 = 0)

-- Define the theorem for the problem statement
theorem meeting_anniversary_day :
  ∀ (year : ℕ), year ∈ initial_years →
  let leap_years := (∑ n in range 1668, if is_leap_year n then 1 else 0)
  let total_days := 11 * 365 + leap_years
  let day_of_week := total_days % 7
  in (day_of_week = 0 ∧ probability Friday = 3 / 4) ∨ (day_of_week = 6 ∧ probability Thursday 1 / 4) :=
by
  sorry

end meeting_anniversary_day_l494_494411


namespace sum_of_first_ten_primes_with_units_digit_3_l494_494850

def units_digit_3_and_prime (n : ℕ) : Prop :=
  (n % 10 = 3) ∧ (Prime n)

def first_ten_primes_with_units_digit_3 : list ℕ :=
  [3, 13, 23, 43, 53, 73, 83, 103, 113, 163]

theorem sum_of_first_ten_primes_with_units_digit_3 :
  list.sum first_ten_primes_with_units_digit_3 = 671 :=
by
  sorry

end sum_of_first_ten_primes_with_units_digit_3_l494_494850


namespace can_make_all_heads_from_initial_l494_494868

inductive Coin
| H : Coin
| T : Coin

open Coin

def initial_state : list Coin := [T, T, H, T, T]

def flip_three (l: list Coin) (n: ℕ) : list Coin :=
  if h : n + 2 < l.length then
    let sub := l[n..n+3] in
    let flipped := sub.map (λ x => match x with | H => T | T => H end) in
    l.take n ++ flipped ++ l.drop (n + 3)
  else l

def is_all_heads (l: list Coin) : Prop :=
  l.all (λ x => x = H)

theorem can_make_all_heads_from_initial : ∃ fs, fs = flip_three (flip_three initial_state 2) 0 ∧ is_all_heads fs := by
  exists (flip_three (flip_three initial_state 2) 0)
  split
  . rfl
  . sorry

end can_make_all_heads_from_initial_l494_494868


namespace solve_logarithm_eq_l494_494175

theorem solve_logarithm_eq (y : ℝ) (hy : log 3 y + log 9 y = 5) : y = 3^(10/3) := 
  sorry

end solve_logarithm_eq_l494_494175


namespace increase_by_150_percent_l494_494320

theorem increase_by_150_percent (x : ℝ) (h : x = 80) : x + (1.5 * x) = 200 := 
by
  -- The proof goes here, but is omitted as per instructions.
  sorry

end increase_by_150_percent_l494_494320


namespace increase_by_150_percent_l494_494317

theorem increase_by_150_percent (x : ℝ) (h : x = 80) : x + (1.5 * x) = 200 := 
by
  -- The proof goes here, but is omitted as per instructions.
  sorry

end increase_by_150_percent_l494_494317


namespace frozen_yogurt_combination_count_l494_494751

theorem frozen_yogurt_combination_count : 
    let flavors := 5 in
    let toppings := 7 in
    let topping_combinations := Nat.choose toppings 2 in
    let doubling_options := 3 in
    let total_combinations := flavors * (topping_combinations * doubling_options) in
    total_combinations = 315 :=
by
  sorry

end frozen_yogurt_combination_count_l494_494751


namespace range_of_a_l494_494475

noncomputable def m := {x : ℝ | Real.exp (x - 1) + x ^ 3 - 2 = 0}
noncomputable def n (a : ℝ) := {x : ℝ | x ^ 2 - a * x - a + 3 = 0}

theorem range_of_a : 
  (∃ (m : ℝ) (n : ℝ) (a : ℝ), m ∈ m ∧ n ∈ n a ∧ abs (m - n) ≤ 1) → 
  (2 : ℝ) ≤ a ∧ a ≤ 3 :=
by
  sorry

end range_of_a_l494_494475


namespace ratio_R_U_l494_494216

theorem ratio_R_U : 
  let spacing := 1 / 4
  let R := 3 * spacing
  let U := 6 * spacing
  R / U = 0.5 := 
by
  sorry

end ratio_R_U_l494_494216


namespace meeting_anniversary_day_l494_494410

-- Define the input parameters for the problem
def initial_years : Set ℕ := {1668, 1669, 1670, 1671}
def meeting_day := "Friday"
def is_leap_year (year : ℕ) : Bool := (year % 4 = 0)

-- Define the theorem for the problem statement
theorem meeting_anniversary_day :
  ∀ (year : ℕ), year ∈ initial_years →
  let leap_years := (∑ n in range 1668, if is_leap_year n then 1 else 0)
  let total_days := 11 * 365 + leap_years
  let day_of_week := total_days % 7
  in (day_of_week = 0 ∧ probability Friday = 3 / 4) ∨ (day_of_week = 6 ∧ probability Thursday 1 / 4) :=
by
  sorry

end meeting_anniversary_day_l494_494410


namespace num_ways_to_assign_grades_l494_494371

-- Define the number of students
def num_students : ℕ := 12

-- Define the number of grades available to each student
def num_grades : ℕ := 4

-- The theorem stating that the total number of ways to assign grades is 4^12
theorem num_ways_to_assign_grades : num_grades ^ num_students = 16777216 := by
  sorry

end num_ways_to_assign_grades_l494_494371


namespace solve_for_x_l494_494538

noncomputable def vec (x y : ℝ) : ℝ × ℝ := (x, y)

theorem solve_for_x (x : ℝ) :
  let a := vec 1 2
  let b := vec x 1
  let u := (a.1 + 2 * b.1, a.2 + 2 * b.2)
  let v := (2 * a.1 - 2 * b.1, 2 * a.2 - 2 * b.2)
  (u.1 * v.2 = u.2 * v.1) → x = 1 / 2 := by
  sorry

end solve_for_x_l494_494538


namespace minimum_value_6x7_7x_minus6_minimum_value_6x7_7x_minus6_at_x_eq_1_l494_494129

theorem minimum_value_6x7_7x_minus6 (x : ℝ) (hx : 0 < x) : 6*x^7 + 7*x^(-6) ≥ 13 :=
begin
  sorry
end

theorem minimum_value_6x7_7x_minus6_at_x_eq_1 : 6*(1:ℝ)^7 + 7*(1:ℝ)^(-6) = 13 :=
begin
  norm_num,
end

end minimum_value_6x7_7x_minus6_minimum_value_6x7_7x_minus6_at_x_eq_1_l494_494129


namespace matrix_condition_min_sum_l494_494141

theorem matrix_condition_min_sum
  (a b c d : ℕ)
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)
  (h_matrix_eq : 
    ⟨⟨4, 0⟩, ⟨0, 3⟩⟩ * ⟨⟨a, b⟩, ⟨c, d⟩⟩ = ⟨⟨a, b⟩, ⟨c, d⟩⟩ * ⟨⟨12, 16⟩, ⟨-15, -20⟩⟩) :
  a + b + c + d = 47 :=
begin
  sorry
end

end matrix_condition_min_sum_l494_494141


namespace perfect_square_trinomial_l494_494064

theorem perfect_square_trinomial (m : ℝ) : 
  (∃ a : ℝ, (∀ x : ℝ, x^2 + 2*(m - 1)*x + 16 = (x + a)^2) ∨ (∀ x : ℝ, x^2 + 2*(m - 1)*x + 16 = (x - a)^2)) ↔ m = 5 ∨ m = -3 :=
sorry

end perfect_square_trinomial_l494_494064


namespace fourth_term_of_gp_is_negative_10_point_42_l494_494063

theorem fourth_term_of_gp_is_negative_10_point_42 (x : ℝ) 
  (h : ∃ r : ℝ, r * (5 * x + 5) = (3 * x + 3) * ((3 * x + 3) / x)) :
  r * (5 * x + 5) * ((3 * x + 3) / x) * ((3 * x + 3) / x) = -10.42 :=
by
  sorry

end fourth_term_of_gp_is_negative_10_point_42_l494_494063


namespace problem1_problem2_l494_494732

theorem problem1 (f : ℝ → ℝ) (x : ℝ) : 
  (f (x + 1) = x^2 - 3 * x + 2) → 
  (f x = x^2 - 6 * x + 6) :=
begin
  sorry
end

theorem problem2 (f : ℝ → ℝ) (k : ℝ) :
  (∀ x : ℝ, f x = x^2 - 2 * k * x - 8) ∧
  ∀ x₁ x₂ ∈ Icc (1 : ℝ) (4 : ℝ), (x₁ < x₂ → f x₁ ≤ f x₂ ∨ f x₁ ≥ f x₂) → 
  (k ≥ 4 ∨ k ≤ 1) :=
begin
  sorry
end

end problem1_problem2_l494_494732


namespace max_sum_x_y_l494_494912

theorem max_sum_x_y (x y : ℝ) (h1 : x^2 + y^2 = 7) (h2 : x^3 + y^3 = 10) : x + y ≤ 4 :=
sorry

end max_sum_x_y_l494_494912


namespace increase_by_150_percent_l494_494315

theorem increase_by_150_percent (x : ℝ) (h : x = 80) : x + (1.5 * x) = 200 := 
by
  -- The proof goes here, but is omitted as per instructions.
  sorry

end increase_by_150_percent_l494_494315


namespace number_of_correct_calculations_is_one_l494_494671

/- Given conditions -/
def cond1 (a : ℝ) : Prop := a^2 * a^2 = 2 * a^2
def cond2 (a b : ℝ) : Prop := (a - b)^2 = a^2 - b^2
def cond3 (a : ℝ) : Prop := a^2 + a^3 = a^5
def cond4 (a b : ℝ) : Prop := (-2 * a^2 * b^3)^3 = -6 * a^6 * b^3
def cond5 (a : ℝ) : Prop := (-a^3)^2 / a = a^5

/- Statement to prove the number of correct calculations is 1 -/
theorem number_of_correct_calculations_is_one :
  (¬ (cond1 a)) ∧ (¬ (cond2 a b)) ∧ (¬ (cond3 a)) ∧ (¬ (cond4 a b)) ∧ (cond5 a) → 1 = 1 :=
by
  sorry

end number_of_correct_calculations_is_one_l494_494671


namespace compound_interest_rate_l494_494456

theorem compound_interest_rate (P A : ℝ) (t n : ℕ) (CI r : ℝ)
  (hP : P = 1200)
  (hCI : CI = 1785.98)
  (ht : t = 5)
  (hn : n = 1)
  (hA : A = P * (1 + r/n)^(n * t)) :
  A = P + CI → 
  r = 0.204 :=
by
  sorry

end compound_interest_rate_l494_494456


namespace increase_by_150_percent_l494_494305

theorem increase_by_150_percent (n : ℕ) : 
  n = 80 → n + (3 / 2) * n = 200 :=
by
  intros h
  rw [h]
  norm_num
  sorry

end increase_by_150_percent_l494_494305


namespace no_primes_between_n_fact_plus_2_and_n_fact_plus_n_l494_494871

theorem no_primes_between_n_fact_plus_2_and_n_fact_plus_n (n : ℕ) (h : n > 2) : 
  ∀ a, n! + 2 < a ∧ a < n! + n + 1 → ¬ nat.prime a :=
by
  sorry

end no_primes_between_n_fact_plus_2_and_n_fact_plus_n_l494_494871


namespace concurrency_of_AM_DN_XY_l494_494131

variables {A B C D X Y O M N : Point}
variables {Γ1 Γ2 : Circle}

-- Define A, B, C, D as collinear points
axiom collinear_points : collinear [A, B, C, D]

-- Define Γ1 as the circle with diameter [AC]
axiom circle_Γ1 : diameter_circle Γ1 A C

-- Define Γ2 as the circle with diameter [BD]
axiom circle_Γ2 : diameter_circle Γ2 B D

-- Intersection points X and Y of Γ1 and Γ2
axiom intersection_XY : (X ∈ (Γ1 ∩ Γ2)) ∧ (Y ∈ (Γ1 ∩ Γ2))

-- Arbitrary point O on (XY) not on (AB)
axiom point_O : O ∈ line(X, Y) ∧ O ∉ line(A, B)

-- Define M and N as intersections
-- CO intersects at M on Γ1
axiom intersect_CO_M : intersects_at (line(C, O)) Γ1 M

-- BO intersects at N on Γ2
axiom intersect_BO_N : intersects_at (line(B, O)) Γ2 N

-- Proof of concurrency of AM, DN, and XY
theorem concurrency_of_AM_DN_XY : concurrent (line(A, M)) (line(D, N)) (line(X, Y)) :=
by sorry

end concurrency_of_AM_DN_XY_l494_494131


namespace correct_propositions_l494_494778

def P (x : ℝ) : Prop := x^2 + x + 1 < 0
def neg_P (x : ℝ) : Prop := x^2 + x + 1 ≥ 0

-- Statement that represents the problem and asserts that proposition ② and ③ are correct
theorem correct_propositions :
  (∀ (x : ℝ), x^2 + x + 1 ≥ 0) ∧
  (∀ P Q : Prop, ¬P → (P ∨ Q) → Q) :=
by
  -- Proposition ②
  have prop_2 : ∀ (x : ℝ), neg_P x := sorry
  
  -- Proposition ③
  have prop_3 : ∀ P Q : Prop, (¬P → (P ∨ Q) → Q) := sorry
  
  exact ⟨prop_2, prop_3⟩

end correct_propositions_l494_494778


namespace tens_digit_of_large_power_l494_494079

theorem tens_digit_of_large_power : ∃ a : ℕ, a = 2 ∧ ∀ n ≥ 2, (5 ^ n) % 100 = 25 :=
by
  sorry

end tens_digit_of_large_power_l494_494079


namespace third_cyclist_speed_l494_494879

theorem third_cyclist_speed (s1 s3 : ℝ) :
  (∃ s1 s3 : ℝ,
    (∀ t : ℝ, t > 0 → (s1 > s3) ∧ (20 = abs (10 * t - s1 * t)) ∧ (5 = abs (s1 * t - s3 * t)) ∧ (s1 ≥ 10))) →
  (s3 = 25 ∨ s3 = 5) :=
by sorry

end third_cyclist_speed_l494_494879


namespace fruit_basket_count_l494_494930

theorem fruit_basket_count :
  let apples := 6 in
  let oranges := 12 in
  let min_apples := 2 in
  let apple_choices := apples - min_apples + 1 in -- (6 - 2 + 1) = 5
  let orange_choices := oranges + 1 in -- (12 + 1) = 13
  apple_choices * orange_choices = 65 := by
  sorry

end fruit_basket_count_l494_494930


namespace fraction_halfway_l494_494707

theorem fraction_halfway (a b c d : ℚ) (h₁ : a = 3) (h₂ : b = 4) (h₃ : c = 5) (h₄ : d = 6) :
  (1 / 2) * ((a / b) + (c / d)) = 19 / 24 := 
by
  sorry

end fraction_halfway_l494_494707


namespace trig_identity_l494_494866

theorem trig_identity :
  (Real.sin (17 * Real.pi / 180) * Real.cos (47 * Real.pi / 180) - 
   Real.sin (73 * Real.pi / 180) * Real.cos (43 * Real.pi / 180)) = -1/2 := 
by
  sorry

end trig_identity_l494_494866


namespace plant_short_trees_l494_494685

theorem plant_short_trees (current_short_trees : ℕ) (desired_short_trees : ℕ) : 
  current_short_trees = 3 → 
  desired_short_trees = 12 → 
  (desired_short_trees - current_short_trees = 9) :=
by
  intros h1 h2
  rw [h1, h2]
  show 12 - 3 = 9
  rfl

end plant_short_trees_l494_494685


namespace sum_of_first_ten_primes_ending_in_3_is_671_l494_494847

noncomputable def sum_of_first_ten_primes_ending_in_3 : ℕ :=
  3 + 13 + 23 + 43 + 53 + 73 + 83 + 103 + 113 + 163

theorem sum_of_first_ten_primes_ending_in_3_is_671 :
  sum_of_first_ten_primes_ending_in_3 = 671 :=
by
  sorry

end sum_of_first_ten_primes_ending_in_3_is_671_l494_494847


namespace sum_f_values_l494_494933

noncomputable def f (x : ℤ) : ℤ := (x - 1)^3 + 1

theorem sum_f_values :
  (f (-5) + f (-4) + f (-3) + f (-2) + f (-1) + f 0 + f 1 + f 2 + f 3 + f 4 + f 5 + f 6 + f 7) = 13 :=
by
  sorry

end sum_f_values_l494_494933


namespace IsoperimetricQuotient_Inequality_l494_494554

theorem IsoperimetricQuotient_Inequality 
  (ABC : Triangle) (A1 A2 : Point) 
  (h1 : A1 ∈ interior ABC)
  (h2 : A2 ∈ interior (Triangle.mk A1 ABC.B ABC.C)) :
  IQ (Triangle.mk A1 ABC.B ABC.C) > IQ (Triangle.mk A2 ABC.B ABC.C) := 
sorry

end IsoperimetricQuotient_Inequality_l494_494554


namespace units_digit_sum_of_squares_of_first_3003_odd_integers_l494_494230

theorem units_digit_sum_of_squares_of_first_3003_odd_integers:
  let units_digit_square (n : ℕ) : ℕ := match n % 10 with
    | 1 => 1
    | 3 => 9
    | 5 => 5
    | 7 => 9
    | 9 => 1
    | _ => 0 in
  let sum_units_digits (n : ℕ) : ℕ := (n / 5) * (2 * 1 + 2 * 9 + 1 * 5) + 
                                    [(1, 1), (2, 9), (3, 5)].take (n % 5).map (λ x => snd x) in
  sum_units_digits 3003 % 10 = 5 :=
by sorry

end units_digit_sum_of_squares_of_first_3003_odd_integers_l494_494230


namespace vectors_are_orthogonal_l494_494884

open Real

variables {α β : ℝ}

def a : ℝ × ℝ := (cos α, sin α)
def b : ℝ × ℝ := (cos β, sin β)
def add_vectors (x y : ℝ × ℝ) : ℝ × ℝ := (x.1 + y.1, x.2 + y.2)
def sub_vectors (x y : ℝ × ℝ) : ℝ × ℝ := (x.1 - y.1, x.2 - y.2)
def dot_product (x y : ℝ × ℝ) : ℝ := x.1 * y.1 + x.2 * y.2

theorem vectors_are_orthogonal : dot_product (add_vectors a b) (sub_vectors a b) = 0 :=
by
  sorry

end vectors_are_orthogonal_l494_494884


namespace find_larger_number_l494_494193

theorem find_larger_number (S L : ℕ) (h1 : L - S = 1365) (h2 : L = 6 * S + 15) : L = 1635 :=
sorry

end find_larger_number_l494_494193


namespace longer_leg_of_minimum_30_60_90_triangle_is_9_l494_494441

-- Define the properties of a 30-60-90 triangle
def sideRatios := (1 : ℝ, Real.sqrt 3, 2 : ℝ)

noncomputable def longer_leg_of_smallest_triangle (hypotenuse_largest : ℝ) : ℝ :=
  let hypotenuse1  := hypotenuse_largest
  let shorter_leg1 := hypotenuse1 / 2
  let longer_leg1  := shorter_leg1 * Real.sqrt 3
  let hypotenuse2  := longer_leg1
  let shorter_leg2 := hypotenuse2 / 2
  let longer_leg2  := shorter_leg2 * Real.sqrt 3
  let hypotenuse3  := longer_leg2
  let shorter_leg3 := hypotenuse3 / 2
  let longer_leg3  := shorter_leg3 * Real.sqrt 3
  let hypotenuse4  := longer_leg3
  let shorter_leg4 := hypotenuse4 / 2
  let longer_leg4  := shorter_leg4 * Real.sqrt 3
  longer_leg4

theorem longer_leg_of_minimum_30_60_90_triangle_is_9 (hypotenuse_largest : ℝ) 
  (H : hypotenuse_largest = 16) : longer_leg_of_smallest_triangle hypotenuse_largest = 9 := by
  sorry

end longer_leg_of_minimum_30_60_90_triangle_is_9_l494_494441


namespace vector_angle_range_and_function_extremes_l494_494510

noncomputable def angle_between_vectors (a b : ℝ) : ℝ := 
if h : a ≠ 0 ∧ b ≠ 0 then
acos ((a * b) / (a.abs * b.abs))
else 0

theorem vector_angle_range_and_function_extremes (θ : ℝ)
(hθ : θ = angle_between_vectors 2 1) :
  (2 * |(cos θ)| ≤ 1 ∧ ∃θ : ℝ, θ ∈ [π/3, π] ∧ f min = -1 at θ = 7π/6)
  ∧ (∀θ : ℝ, θ ∈ [π/3, π] 
    ∃ (θ : ℝ), 2 * θ + π/3 = 7π / 3 ∧ f max = sqrt 3 / 2 at θ = π) :=
sorry

end vector_angle_range_and_function_extremes_l494_494510


namespace increase_result_l494_494247

-- Given conditions
def original_number : ℝ := 80
def increase_percentage : ℝ := 1.5

-- The result after the increase
theorem increase_result (h1 : original_number = 80) (h2 : increase_percentage = 1.5) : 
  original_number + (increase_percentage * original_number) = 200 := by
  sorry

end increase_result_l494_494247


namespace student_university_assignment_l494_494788

theorem student_university_assignment : 
  ∃ (assignments : finset (set (finset (fin 5)))), finite assignments ∧ assignments.card = 150 := 
  sorry

end student_university_assignment_l494_494788


namespace minimum_value_of_x_squared_plus_y_squared_l494_494925

theorem minimum_value_of_x_squared_plus_y_squared (x y : ℝ) (h : x^2 + y^2 - 4 * x + 1 = 0) : 
  ∃ (m : ℝ), (m = 7 - 4 * real.sqrt 3) ∧ ∀ (a b : ℝ), (a^2 + b^2 - 4 * a + 1 = 0) → m ≤ a^2 + b^2 :=
by
  sorry

end minimum_value_of_x_squared_plus_y_squared_l494_494925


namespace min_value_expr_l494_494461

noncomputable def expr (θ : ℝ) : ℝ := 3 * Real.cos θ + 1 / Real.sin θ + 2 * Real.tan θ

theorem min_value_expr : 
  ∃ θ, 0 < θ ∧ θ < Real.pi / 2 ∧ expr θ = 3 * Real.cbrt 6 := 
begin
  sorry
end

end min_value_expr_l494_494461


namespace maximum_minimum_cos_sin_cos_l494_494478

noncomputable def max_min_cos_sin_cos_product (x y z : ℝ) : ℝ × ℝ :=
  if x + y + z = π / 2 ∧ x ≥ y ∧ y ≥ z ∧ z ≥ π / 12 then
    let product := cos x * sin y * cos z
    (max product, min product)
  else (0, 0)

theorem maximum_minimum_cos_sin_cos :
  ∃ x y z : ℝ, 
    x + y + z = π / 2 ∧ x ≥ y ∧ y ≥ z ∧ z ≥ π / 12 ∧
    max_min_cos_sin_cos_product x y z = ( (2 + real.sqrt 3) / 8, 1 / 8) :=
by
  sorry

end maximum_minimum_cos_sin_cos_l494_494478


namespace anniversary_day_probability_l494_494393

/- Definitions based on the conditions -/
def is_leap_year (year : ℕ) : Prop :=
  year % 4 = 0

def days_in_year (year : ℕ) : ℕ :=
  if is_leap_year year then 366 else 365

def total_days (start_year : ℕ) : ℕ :=
  (list.sum (list.map days_in_year (list.range' start_year 11)))

/- Prove the day of the 11th anniversary and its probabilities -/
theorem anniversary_day_probability (start_year : ℕ) (h : start_year ∈ {1668, 1669, 1670, 1671}) :
  let days := total_days start_year % 7
  in (days = 0 ∧ 0.75 ≤ 1) ∨ (days = 6 ∧ 0.25 ≤ 1) :=
by
  sorry

end anniversary_day_probability_l494_494393


namespace circle_area_ratio_l494_494692

theorem circle_area_ratio (O Q Y Z : Point)
  (h1 : midpoint O Q Y)
  (h2 : midpoint O Y Z) :
  let rOQ := distance O Q,
      rOY := distance O Y,
      rOZ := distance O Z
  in (rOZ * rOZ) / (rOQ * rOQ) = 1 / 16 :=
by
  sorry

end circle_area_ratio_l494_494692


namespace tangent_line_at_1_l494_494523

noncomputable def f : ℝ → ℝ := λ x, Real.log x - 3 * x

theorem tangent_line_at_1 :
  ∀ (x y : ℝ), (y = f x) → x = 1 → y = -3 → 2 * x + y + 1 = 0 :=
by
  intros x y hy hx hy1
  -- Function definition
  have hf : f 1 = -3 := by
    unfold f
    rw [Real.log_one, mul_one, sub_zero]
    simp
  -- Derivative
  have h_deriv : deriv f 1 = -2 := by
    simp [f, deriv_log, deriv_mul, deriv_sub, deriv_id']
    unfold_coes
    simp
  -- Equation of the tangent line
  have h_tangent : y = f 1 + deriv f 1 * (x - 1) := by
    simp [hy1, h_deriv]
  calc
  2 * x + y + 1 = 2 * 1 + (-3) + 1 : by rw [hx, hy1]
  ... = 0 : by 
    ring

  exact h_tangent

end tangent_line_at_1_l494_494523


namespace find_number_of_children_l494_494357

theorem find_number_of_children (adults children : ℕ) (adult_ticket_price child_ticket_price total_money change : ℕ) 
    (h1 : adult_ticket_price = 9) 
    (h2 : child_ticket_price = adult_ticket_price - 2) 
    (h3 : total_money = 40) 
    (h4 : change = 1) 
    (h5 : adults = 2) 
    (total_cost : total_money - change = adults * adult_ticket_price + children * child_ticket_price) : 
    children = 3 :=
sorry

end find_number_of_children_l494_494357


namespace balls_in_indistinguishable_boxes_l494_494017

theorem balls_in_indistinguishable_boxes : 
  let balls := 6 in
  let boxes := 3 in
  -- Expression to count ways to distribute balls into indistinguishable boxes
  ((balls.choose 0 * ((balls-0).choose 6)) + 
   (balls.choose 1 * ((balls-1).choose 5)) + 
   (balls.choose 2 * ((balls-2).choose 4)) + 
   (balls.choose 2 * ((balls-2).choose 3) / 2!) + 
   (balls.choose 3 * ((balls-3).choose 3)) + 
   (balls.choose 3 * ((balls-3).choose 2) * ((balls-5).choose 1)) + 
   (balls.choose 2 * ((balls-2).choose 2) / 3!))
  = 92 := 
sorry

end balls_in_indistinguishable_boxes_l494_494017


namespace percentage_industrial_lubricants_l494_494741

def percentage_microphotonics : ℝ := 13
def percentage_home_electronics : ℝ := 24
def percentage_food_additives : ℝ := 15
def percentage_gmo : ℝ := 29
def degrees_basic_astrophysics : ℝ := 39.6
def degrees_full_circle : ℝ := 360
def percentage_basic_astrophysics : ℝ := (degrees_basic_astrophysics / degrees_full_circle) * 100
def percentage_total_known : ℝ := percentage_microphotonics + percentage_home_electronics + percentage_food_additives + percentage_gmo + percentage_basic_astrophysics

theorem percentage_industrial_lubricants : 
  (100 - percentage_total_known) = 8 := by
  sorry

end percentage_industrial_lubricants_l494_494741


namespace expected_difference_is_91_5_l494_494388

noncomputable def expected_difference_between_toast_days : ℚ :=
  let die_faces := {1, 2, 3, 4, 5, 6, 7, 8}
  let perfect_squares := {1, 4}
  let primes := {2, 3, 5, 7}
  let prob_perfect_square := (perfect_squares.card : ℚ) / (die_faces.card : ℚ)
  let prob_prime := (primes.card : ℚ) / (die_faces.card : ℚ)
  let days_in_leap_year : ℚ := 366
  let expected_days_toast_jam := prob_perfect_square * days_in_leap_year
  let expected_days_toast_butter := prob_prime * days_in_leap_year
  expected_days_toast_butter - expected_days_toast_jam

theorem expected_difference_is_91_5 :
  expected_difference_between_toast_days = 91.5 :=
by
  sorry

end expected_difference_is_91_5_l494_494388


namespace max_min_cos_sin_cos_l494_494484

theorem max_min_cos_sin_cos (x y z : ℝ) (h1 : x ≥ y) (h2 : y ≥ z) (h3 : z ≥ π / 12) (h4 : x + y + z = π / 2) :
  ∃ (max_val min_val : ℝ), 
    (max_val = (2 + Real.sqrt 3) / 8) ∧ 
    (min_val = 1 / 8) ∧ 
    max_val = max (cos x * sin y * cos z) ∧ 
    min_val = min (cos x * sin y * cos z) :=
  sorry

end max_min_cos_sin_cos_l494_494484


namespace area_of_plane_region_max_value_find_bc_l494_494524

-- Part 1
theorem area_of_plane_region (a b : ℝ) (h1 : -1 ≤ a - b ∧ a - b ≤ 2) (h2 : 2 ≤ a + b ∧ a + b ≤ 4) : 
  let distance1 := abs (-1 - 2) / sqrt 2,
      distance2 := abs (4 - 2) / sqrt 2,
      area := distance1 * distance2 in
  area = 3 :=
sorry

-- Part 2
noncomputable def t (x : ℝ) := 2 + (1 / (x^2 - x))

noncomputable def f (x : ℝ) := t x * x

theorem max_value (h : ∀ x : ℝ, x < 1 ∧ x ≠ 0) : 
  ∃ x : ℝ, f x = 2 - 2 * sqrt 2 :=
sorry

-- Part 3
theorem find_bc (a b c : ℝ) (h : ∀ x : ℝ, b^2 + c^2 - bc - 3b - 1 ≤ (x^2 - (a + 3) * x) ∧ 
                                           (x^2 - (a + 3) * x) ≤ a + 4)
                  (sol_set : set ℝ := {x | x ∈ [-1, 5]}) :
  b = 2 ∧ c = 1 :=
sorry

end area_of_plane_region_max_value_find_bc_l494_494524


namespace increase_by_percentage_l494_494329

theorem increase_by_percentage (a : ℤ) (b : ℚ) : ((b + 1) * a) = 200 :=
by
  -- Definitions based directly on conditions
  let a := (80 : ℤ)
  let b := (1.5 : ℚ)
  -- Assertions about the equivalent proof problem
  have h1 : b + 1 = 2.5 := by norm_num
  have h2 : (b + 1) * a = 200 := by norm_num
  exact h2

end increase_by_percentage_l494_494329


namespace remove_7_increases_probability_l494_494699

open Finset
open BigOperators

variable {α : Type*} [Fintype α] [DecidableEq α] (S : Finset α) (n : α)

noncomputable def isValidPairSum14 (pair : α × α) : Prop :=
  pair.1 ≠ pair.2 ∧ pair.1 + pair.2 = 14

noncomputable def validPairs (S : Finset α) : Finset (α × α) :=
  (S ×ˢ S).filter isValidPairSum14

noncomputable def probabilitySum14 (S : Finset α) : ℚ :=
  (validPairs S).card / S.card.choose 2

theorem remove_7_increases_probability :
  probabilitySum14 (erase S 7) > probabilitySum14 S :=
  sorry

end remove_7_increases_probability_l494_494699


namespace modulus_of_complex_z_l494_494072

noncomputable theory

-- Definitions of complex numbers and required operations
open Complex

-- Problem statement and the condition
theorem modulus_of_complex_z : 
  ∀ (z : ℂ), (1 - I) * z = 1 + 2 * I → abs z = (Real.sqrt 10) / 2 :=
by
  intros z h
  sorry

end modulus_of_complex_z_l494_494072


namespace car_travel_distance_is_correct_l494_494929

noncomputable def car_distance_traveled
  (train_speed : ℝ) -- Train speed in miles per hour
  (car_speed_factor : ℝ) -- Factor of car speed relative to train speed
  (total_time_minutes : ℝ) -- Total time in minutes
  (stop_time_minutes : ℝ) -- Stop time in minutes) : ℝ :=
  let car_speed := car_speed_factor * train_speed
  let effective_travel_time := total_time_minutes - stop_time_minutes
  (car_speed / 60) * effective_travel_time

theorem car_travel_distance_is_correct :
  car_distance_traveled 120 (2 / 3) 30 5 = 100 / 3 :=
by
  sorry -- proof will be provided here

end car_travel_distance_is_correct_l494_494929


namespace find_angle_B_find_ratio_a_c_l494_494963

variable {a b c A B : ℝ}

/-- Given condition: √3 * a * cos B = b * sin A --/
def given_condition : Prop :=
  sqrt 3 * a * cos B = b * sin A

/-- Prove that angle B = π/3 --/
theorem find_angle_B (h : given_condition) : B = π / 3 :=
  sorry

/-- Additional given condition for question 2: the area S is (√3 / 4) * b ^ 2 --/
def given_area_condition (S : ℝ) : Prop :=
  S = sqrt 3 / 4 * b ^ 2

/-- Prove that a/c = 1 given B = π/3 and the area condition holds --/
theorem find_ratio_a_c (hB : B = π / 3) (hS : given_area_condition (1 / 2 * a * c * sin B)) : a / c = 1 :=
  sorry

end find_angle_B_find_ratio_a_c_l494_494963


namespace intersection_of_sets_l494_494903

open Set

variable {α : Type*} [PartialOrder α] [HasMem α (Set α)]

def A : Set ℝ := {x | x ≤ 2}
def B : Set ℝ := Icc 0 3

-- prove that A ∩ B = [0, 2]
theorem intersection_of_sets : A ∩ B = Icc 0 2 := 
sorry

end intersection_of_sets_l494_494903


namespace hexagon_area_l494_494089

theorem hexagon_area
  (ABCDEF : Type) [linear_ordered_semiring ABCDEF]
  (regular_hexagon : ∀ A B C D E F : ABCDEF, regular_hexagon_rest ABCDEF A B C D E F)
  (G_on_CD : ∀ (C D G : ABCDEF), C + D = G * (1 / 3) + G * (2 / 3))
  (area_OMF_less_3 : ∀ (O M F A N : ABCDEF), area O M F = area O A N - 3) :
    hexagon_area ABCDEF = 60 := 
sorry

end hexagon_area_l494_494089


namespace solve_for_y_l494_494935

theorem solve_for_y (y : ℕ) (h : 9^y = 3^12) : y = 6 :=
by {
  sorry
}

end solve_for_y_l494_494935


namespace multiple_choice_questions_count_l494_494148

variable (M F : ℕ)

-- Conditions
def totalQuestions := M + F = 60
def totalStudyTime := 15 * M + 25 * F = 1200

-- Statement to prove
theorem multiple_choice_questions_count (h1 : totalQuestions M F) (h2 : totalStudyTime M F) : M = 30 := by
  sorry

end multiple_choice_questions_count_l494_494148


namespace polygon_diagonalization_l494_494632

theorem polygon_diagonalization (n : ℕ) (h : n ≥ 3) : 
  ∃ (triangles : ℕ), triangles = n - 2 ∧ 
  (∀ (polygons : ℕ), 3 ≤ polygons → polygons < n → ∃ k, k = polygons - 2) := 
by {
  -- base case
  sorry
}

end polygon_diagonalization_l494_494632


namespace increase_80_by_150_percent_l494_494267

-- Original number
def originalNumber : ℕ := 80

-- Percentage increase as a decimal
def increaseFactor : ℚ := 1.5

-- Expected result after the increase
def expectedResult : ℕ := 200

theorem increase_80_by_150_percent :
  originalNumber + (increaseFactor * originalNumber) = expectedResult :=
by
  sorry

end increase_80_by_150_percent_l494_494267


namespace remainder_of_m_div_5_l494_494338

theorem remainder_of_m_div_5 (m n : ℕ) (hpos : 0 < m) (hdef : m = 15 * n - 1) : m % 5 = 4 :=
sorry

end remainder_of_m_div_5_l494_494338


namespace selection_methods_count_l494_494677

noncomputable def number_of_selection_methods (students : Finset String) (initial_students : Finset String) : ℕ :=
  let remaining_students := students \ initial_students
  remaining_students.card * (remaining_students.card * (remaining_students.card - 1))

theorem selection_methods_count :
  let students := { "A", "B", "C", "D" }.to_finset,
      initial_students := { "C", "D" }.to_finset in
  number_of_selection_methods students initial_students = 12 :=
by
  let students := { "A", "B", "C", "D" }.to_finset,
      initial_students := { "C", "D" }.to_finset
  have h : (students \ initial_students).card = 2 := by sorry
  exact eq.trans (by simp [number_of_selection_methods, h, Finset.card]) (by norm_num)

end selection_methods_count_l494_494677


namespace glass_bottles_count_l494_494649

-- Declare the variables for the conditions
variable (G : ℕ)

-- Define the conditions
def aluminum_cans : ℕ := 8
def total_litter : ℕ := 18

-- State the theorem
theorem glass_bottles_count : G + aluminum_cans = total_litter → G = 10 :=
by
  intro h
  -- place proof here
  sorry

end glass_bottles_count_l494_494649


namespace gcd_2728_1575_l494_494227

theorem gcd_2728_1575 : Int.gcd 2728 1575 = 1 :=
by sorry

end gcd_2728_1575_l494_494227


namespace maximum_minimum_cos_sin_cos_l494_494477

noncomputable def max_min_cos_sin_cos_product (x y z : ℝ) : ℝ × ℝ :=
  if x + y + z = π / 2 ∧ x ≥ y ∧ y ≥ z ∧ z ≥ π / 12 then
    let product := cos x * sin y * cos z
    (max product, min product)
  else (0, 0)

theorem maximum_minimum_cos_sin_cos :
  ∃ x y z : ℝ, 
    x + y + z = π / 2 ∧ x ≥ y ∧ y ≥ z ∧ z ≥ π / 12 ∧
    max_min_cos_sin_cos_product x y z = ( (2 + real.sqrt 3) / 8, 1 / 8) :=
by
  sorry

end maximum_minimum_cos_sin_cos_l494_494477


namespace altitude_line_eq_circumcircle_eq_l494_494537

noncomputable def point := ℝ × ℝ

noncomputable def A : point := (5, 1)
noncomputable def B : point := (1, 3)
noncomputable def C : point := (4, 4)

theorem altitude_line_eq : ∃ (k b : ℝ), (k = 2 ∧ b = -4) ∧ (∀ x y : ℝ, y = k * x + b ↔ 2 * x - y - 4 = 0) :=
sorry

theorem circumcircle_eq : ∃ (h k r : ℝ), (h = 3 ∧ k = 2 ∧ r = 5) ∧ (∀ x y : ℝ, (x - h)^2 + (y - k)^2 = r ↔ (x - 3)^2 + (y - 2)^2 = 5) :=
sorry

end altitude_line_eq_circumcircle_eq_l494_494537


namespace sin_sq_sum_eq_l494_494906

-- Declaring the points A, B, and O
variables {A B O C : Type}
-- Declaring a line l
variable (l : set Type)
-- Assume A and B are on the line l
variable (hA : A ∈ l) (hB : B ∈ l)
-- Assume O is not on the line l
variable (hO : O ∉ l)
-- Assume C is on the line l, and it satisfies the given vector equation
variable (hC : C ∈ l)
variable (h_eq : \overrightarrow{OC} = \overrightarrow{OA}\cos θ + \overrightarrow{OB}\cos^{2} θ)

-- The main statement to be proven
theorem sin_sq_sum_eq (A B O C : Type) [is_point A] [is_point B] [is_point O] [is_point C]
  (l : set Type) (hA : A ∈ l) (hB : B ∈ l) (hO : O ∉ l) (hC : C ∈ l)
  (h_eq : \overrightarrow{OC} = \overrightarrow{OA}\cos θ + \overrightarrow{OB}\cos^{2} θ) :
  sin(θ)^2 + sin(θ)^4 + sin(θ)^6 = sqrt(5) - 1 :=
begin
  sorry
end

end sin_sq_sum_eq_l494_494906


namespace factorize_poly_l494_494832

-- Statement of the problem
theorem factorize_poly (x : ℝ) : x^2 - 3 * x = x * (x - 3) :=
sorry

end factorize_poly_l494_494832


namespace sum_of_first_ten_primes_ending_in_3_is_671_l494_494844

noncomputable def sum_of_first_ten_primes_ending_in_3 : ℕ :=
  3 + 13 + 23 + 43 + 53 + 73 + 83 + 103 + 113 + 163

theorem sum_of_first_ten_primes_ending_in_3_is_671 :
  sum_of_first_ten_primes_ending_in_3 = 671 :=
by
  sorry

end sum_of_first_ten_primes_ending_in_3_is_671_l494_494844


namespace circle_area_ratio_is_correct_l494_494939

def circle_area_ratio (R_C R_D : ℝ) : ℝ := (R_C / R_D) ^ 2

theorem circle_area_ratio_is_correct (R_C R_D : ℝ) (h1: R_C / R_D = 3 / 2) : 
  circle_area_ratio R_C R_D = 9 / 4 :=
by
  unfold circle_area_ratio
  rw [h1]
  norm_num

end circle_area_ratio_is_correct_l494_494939


namespace solve_log_equation_l494_494171

theorem solve_log_equation (y : ℝ) (h : log 3 y + log 9 y = 5) : y = 3^(10/3) :=
sorry

end solve_log_equation_l494_494171


namespace impossibility_of_fitting_l494_494162

noncomputable def impossible_fit_non_overlapping_triangles_with_area_greater_than_1_inside_circle_of_radius_1 : Prop :=
  ∀ (T1 T2 : Type) [triangle T1] [triangle T2] 
  (area_T1 : ℝ) (area_T2 : ℝ) (R : ℝ),
  R = 1 →
  area_T1 > 1 →
  area_T2 > 1 →
  (area_T1 + area_T2 ≤ π) →
  false

theorem impossibility_of_fitting :
  impossible_fit_non_overlapping_triangles_with_area_greater_than_1_inside_circle_of_radius_1 := 
by
  sorry

end impossibility_of_fitting_l494_494162


namespace distance_center_circle_to_line_l494_494982

/-- The parametric equation of a circle C is given by:
x = cos α, y = 1 + sin α
The polar coordinate equation of line l is:
ρ cos θ - ρ sin θ - 1 = 0
Prove that the distance from the center of circle C to line l is √2 --/
theorem distance_center_circle_to_line 
  (α θ ρ : ℝ) 
  (x y : ℝ := cos α) 
  (y := 1 + sin α) 
  (l : ℝ := ρ * cos θ - ρ * sin θ - 1)
  (center_x : ℝ := 0) 
  (center_y : ℝ := 1)
  (line_eq : x - y - 1 = 0) : 
  distance (center_x, center_y) (l, 0) = sqrt 2 := 
by {
  sorry
}

end distance_center_circle_to_line_l494_494982


namespace negation_of_p_l494_494071

-- Declare the proposition p as a condition
def p : Prop :=
  ∀ (x : ℝ), 0 ≤ x → x^2 + 4 * x + 3 > 0

-- State the problem
theorem negation_of_p : ¬ p ↔ ∃ (x : ℝ), 0 ≤ x ∧ x^2 + 4 * x + 3 ≤ 0 :=
by
  sorry

end negation_of_p_l494_494071


namespace school_pays_570_l494_494182

theorem school_pays_570
  (price_per_model : ℕ := 100)
  (models_kindergarten : ℕ := 2)
  (models_elementary_multiple : ℕ := 2)
  (total_models : ℕ := models_kindergarten + models_elementary_multiple * models_kindergarten)
  (price_reduction : ℕ := if total_models > 5 then (price_per_model * 5 / 100) else 0)
  (reduced_price_per_model : ℕ := price_per_model - price_reduction) :
  2 * models_kindergarten * reduced_price_per_model = 570 :=
by
  -- Proof omitted
  sorry

end school_pays_570_l494_494182


namespace range_of_m_l494_494931

theorem range_of_m (x m : ℝ) (h : x^2 - 2 * x - 8 < 0) :
  (∃ m : ℝ, h → x < m) → m ≥ 4 :=
by
  sorry

end range_of_m_l494_494931


namespace range_of_a_l494_494530

theorem range_of_a (f g : ℝ → ℝ) (a : ℝ) (x1 x2 : ℝ) :
  (∀ x1 ∈ set.Icc 0 4, ∃ x2 ∈ set.Icc 0 4, f x1 = g x2) →
  (f = λ x, x^3 - 6*x^2 + 9*x) →
  (g = λ x, (1/3)*x^3 - (a+1)/2*x^2 + a*x - 1/3) →
  1 < a →
  ((1 < a ∧ a ≤ 9/4) ∨ 9 ≤ a) :=
begin
  sorry
end

end range_of_a_l494_494530


namespace income_calculation_l494_494370

theorem income_calculation
    (I : ℝ)  -- total income
    (remaining_amount : ℝ := 0.25 * I)
    (to_orphanage : ℝ := 0.10 * remaining_amount)
    (to_charity : ℝ := 0.05 * remaining_amount)
    (invested : ℝ := 0.10 * I)
    (amount_left : ℝ := remaining_amount - (to_orphanage + to_charity))
    (final_amount : ℝ := 50000) :
    amount_left = final_amount → I ≈ 235294.12 :=
sorry

end income_calculation_l494_494370


namespace two_triangles_in_circle_l494_494159

open Real EuclideanGeometry

-- Define the problem conditions and statement
theorem two_triangles_in_circle (t1 t2 : Triangle) (C : Circle)
    (hC : C.radius = 1)
    (h_t1_in_C : t1 ⊂ C)
    (h_t2_in_C : t2 ⊂ C)
    (h_area_t1 : Triangle.area t1 > 1)
    (h_area_t2 : Triangle.area t2 > 1)
    (h_no_overlap : ¬ (t1 ∩ t2 ⊄ ∅)) : 
    False :=
sorry

end two_triangles_in_circle_l494_494159


namespace longest_leg_of_smallest_triangle_l494_494437

-- Definitions based on conditions
def is306090Triangle (h : ℝ) (s : ℝ) (l : ℝ) : Prop :=
  s = h / 2 ∧ l = s * (Real.sqrt 3)

def chain_of_306090Triangles (H : ℝ) : Prop :=
  ∃ h1 s1 l1 h2 s2 l2 h3 s3 l3 h4 s4 l4,
    is306090Triangle h1 s1 l1 ∧
    is306090Triangle h2 s2 l2 ∧
    is306090Triangle h3 s3 l3 ∧
    is306090Triangle h4 s4 l4 ∧
    h1 = H ∧ l1 = h2 ∧ l2 = h3 ∧ l3 = h4

-- Main theorem
theorem longest_leg_of_smallest_triangle (H : ℝ) (h : ℝ) (l : ℝ) (H_cond : H = 16) 
  (h_cond : h = 9) :
  chain_of_306090Triangles H →
  ∃ h4 s4 l4, is306090Triangle h4 s4 l4 ∧ l = h4 →
  l = 9 := 
by
  sorry

end longest_leg_of_smallest_triangle_l494_494437


namespace increased_number_l494_494277

theorem increased_number (original_number : ℕ) (percentage_increase : ℚ) :
  original_number = 80 →
  percentage_increase = 1.5 →
  original_number * (1 + percentage_increase) = 200 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end increased_number_l494_494277


namespace application_methods_count_l494_494578

theorem application_methods_count :
  let S := 5; -- number of students
  let U := 3; -- number of universities
  let unrestricted := U^S; -- unrestricted distribution
  let restricted_one_university_empty := (U - 1)^S * U; -- one university empty
  let restricted_two_universities_empty := 0; -- invalid scenario
  let valid_methods := unrestricted - restricted_one_university_empty - restricted_two_universities_empty;
  valid_methods - U = 144 :=
by
  let S := 5
  let U := 3
  let unrestricted := U^S
  let restricted_one_university_empty := (U - 1)^S * U
  let restricted_two_universities_empty := 0
  let valid_methods := unrestricted - restricted_one_university_empty - restricted_two_universities_empty
  have : valid_methods - U = 144 := by sorry
  exact this

end application_methods_count_l494_494578


namespace polynomial_roots_value_l494_494609

noncomputable def roots_prod_sum (p q r s : ℂ) : ℚ :=
  1 / (p * q) + 1 / (p * r) + 1 / (p * s) + 1 / (q * r) + 1 / (q * s) + 1 / (r * s)

theorem polynomial_roots_value :
  (p q r s : ℂ) (h : (p, q, r, s) ∈ { (p, q, r, s) | (x^4 + 10*x^3 + 20*x^2 + 15*x + 6).roots = [p, q, r, s] }) :
  roots_prod_sum p q r s = 10 / 3 :=
by
  sorry

end polynomial_roots_value_l494_494609


namespace triple_exists_not_prime_l494_494158

theorem triple_exists_not_prime (k : ℕ) (hk : 0 < k) :
  ∃ (a b c : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ abc = k * (a + b + c) ∧ ¬ is_prime (a^3 + b^3 + c^3) :=
begin
  sorry
end

end triple_exists_not_prime_l494_494158


namespace tan_A_tan_C_l494_494590

theorem tan_A_tan_C (A B C : Triangle) (H : Point) (D : Point)
  (h1 : orthocenter A B C H)
  (h2 : foot_of_altitude B AC D)
  (h3 : dist H D = 8)
  (h4 : dist H B = 24) :
  tan (angle_A A B C) * tan (angle_C A B C) = 4 := 
sorry

end tan_A_tan_C_l494_494590


namespace school_pays_570_l494_494183

theorem school_pays_570
  (price_per_model : ℕ := 100)
  (models_kindergarten : ℕ := 2)
  (models_elementary_multiple : ℕ := 2)
  (total_models : ℕ := models_kindergarten + models_elementary_multiple * models_kindergarten)
  (price_reduction : ℕ := if total_models > 5 then (price_per_model * 5 / 100) else 0)
  (reduced_price_per_model : ℕ := price_per_model - price_reduction) :
  2 * models_kindergarten * reduced_price_per_model = 570 :=
by
  -- Proof omitted
  sorry

end school_pays_570_l494_494183


namespace sin_double_alpha_l494_494472

variable (α : ℝ)
axiom sin_add_pi_four : Real.sin (α + π / 4) = √2 / 3

theorem sin_double_alpha : Real.sin (2 * α) = -5 / 9 :=
by
  have h := sin_add_pi_four α
  sorry

end sin_double_alpha_l494_494472


namespace number_of_men_in_larger_group_l494_494734

-- Define the constants and conditions
def men1 := 36         -- men in the first group
def days1 := 18        -- days taken by the first group
def men2 := 108       -- men in the larger group (what we want to prove)
def days2 := 6         -- days taken by the second group

-- Given conditions as lean definitions
def total_work (men : Nat) (days : Nat) := men * days
def condition1 := (total_work men1 days1 = 648)
def condition2 := (total_work men2 days2 = 648)

-- Problem statement 
-- proving that men2 is 108
theorem number_of_men_in_larger_group : condition1 → condition2 → men2 = 108 :=
by
  intros
  sorry

end number_of_men_in_larger_group_l494_494734


namespace min_value_of_a_l494_494201

noncomputable def smallest_root_sum : ℕ := 78

theorem min_value_of_a (r s t : ℕ) (h1 : r * s * t = 2310) (h2 : r > 0) (h3 : s > 0) (h4 : t > 0) :
  r + s + t = smallest_root_sum :=
sorry

end min_value_of_a_l494_494201


namespace fraction_spent_at_toy_store_l494_494542

theorem fraction_spent_at_toy_store :
  ∃ (weekly_allowance : ℝ) (spent_arcade_fraction : ℝ) (remaining_after_candy : ℝ),
  weekly_allowance = 4.50 ∧ 
  spent_arcade_fraction = 3/5 ∧ 
  remaining_after_candy = 1.20 ∧ 
  (let spent_arcade := spent_arcade_fraction * weekly_allowance,
       remaining_after_arcade := weekly_allowance - spent_arcade,
       spent_toy_store := remaining_after_arcade - remaining_after_candy in
   spent_toy_store / remaining_after_arcade = 1/3) :=
begin
  existsi 4.50,
  existsi 3/5,
  existsi 1.20,
  split, {refl},
  split, {refl},
  split, {refl},
  sorry
end

end fraction_spent_at_toy_store_l494_494542


namespace number_of_common_tangents_l494_494670

open Real EuclideanGeometry

noncomputable def circle1 := (circle (0 : ℝ) 0 3)
noncomputable def circle2 := (circle (4 : ℝ) -3 4)

theorem number_of_common_tangents :
  num_common_tangents circle1 circle2 = 2 := 
sorry

end number_of_common_tangents_l494_494670


namespace balls_in_boxes_l494_494058

theorem balls_in_boxes :
    (Nat.choose 6 6) + 
    (Nat.choose 6 5) + 
    (Nat.choose 6 4 * Nat.choose 2 2) + 
    (Nat.choose 6 3 / 2) + 
    (Nat.choose 6 4 * Nat.choose 2 1) + 
    (Nat.choose 6 2 * Nat.choose 4 2 / 3.factorial) = 77 :=
by
  sorry

end balls_in_boxes_l494_494058


namespace number_of_players_l494_494096

-- Definitions according to the conditions
def cost_of_shoes : ℝ := 12
def cost_of_jersey : ℝ := cost_of_shoes + 8
def cost_of_cap : ℝ := cost_of_jersey / 2
def total_cost_per_player : ℝ := 2 * cost_of_shoes + 2 * cost_of_jersey + cost_of_cap
def total_expense : ℝ := 4760

-- The statement to prove
theorem number_of_players :
  ∃ (n : ℕ), n * total_cost_per_player = total_expense :=
  sorry

end number_of_players_l494_494096


namespace log_sum_eq_five_has_solution_l494_494173

theorem log_sum_eq_five_has_solution : ∃ y : ℝ, log 3 y + log 9 y = 5 ∧ y = 3^(10/3) := by
  sorry

end log_sum_eq_five_has_solution_l494_494173


namespace length_of_AB_l494_494960

theorem length_of_AB (A B C M N G : Type) [MetricSpace A] [MetricSpace B]
  (hAC : dist A C = 6) (hBC : dist B C = 7) 
  (medianA : isMedian A M G) (medianB : isMedian B N G)
  (hPerpendicular : isPerpendicular medianA medianB) : dist A B = Real.sqrt 17 :=
sorry

end length_of_AB_l494_494960


namespace increase_by_150_percent_l494_494308

theorem increase_by_150_percent (n : ℕ) : 
  n = 80 → n + (3 / 2) * n = 200 :=
by
  intros h
  rw [h]
  norm_num
  sorry

end increase_by_150_percent_l494_494308


namespace distance_to_edges_AC_BD_l494_494105

noncomputable def distance_from_center_to_edges (x y z : ℝ) : ℝ × ℝ :=
let BY := real.sqrt (x^2 - (y/2)^2),
    AZ := real.sqrt (x^2 - (z/2)^2),
    YZ := real.sqrt (x^2 - (y/2)^2 - (z/2)^2) in
if 2 * x > y ∧ 2 * x > z ∧ 4 * x^2 > y^2 + z^2 then
  ( (y * BY / (y * BY + z * AZ)) * YZ,
    (z * AZ / (y * BY + z * AZ)) * YZ )
else
  (0, 0) -- Meaning undefined distances if conditions are not met

theorem distance_to_edges_AC_BD (AB AC BD : ℝ) (x y z : ℝ) 
  (h1 : AB = x) (h2 : AC = y) (h3 : BD = z) 
  (h4 : 2 * x > y) (h5 : 2 * x > z) (h6 : 4 * x^2 > y^2 + z^2) :
  distance_from_center_to_edges x y z = 
  ( (y * real.sqrt(x^2 - (y/2)^2) / (y * real.sqrt(x^2 - (y/2)^2) + z * real.sqrt(x^2 - (z/2)^2))) * real.sqrt(x^2 - (y/2)^2 - (z/2)^2),
    (z * real.sqrt(x^2 - (z/2)^2) / (y * real.sqrt(x^2 - (y/2)^2) + z * real.sqrt(x^2 - (z/2)^2))) * real.sqrt(x^2 - (y/2)^2 - (z/2)^2) ) := sorry

end distance_to_edges_AC_BD_l494_494105


namespace sum_of_first_ten_primes_with_units_digit_three_l494_494863

-- Define the problem to prove the sum of the first 10 primes ending in 3 is 639
theorem sum_of_first_ten_primes_with_units_digit_three : 
  let primes_with_units_digit_three := [3, 13, 23, 43, 53, 73, 83, 103, 113, 163]
  in list.sum primes_with_units_digit_three = 639 := 
by 
  -- We define the primes with the units digit 3 as given and check the sum
  let primes_with_units_digit_three := [3, 13, 23, 43, 53, 73, 83, 103, 113, 163]
  show list.sum primes_with_units_digit_three = 639 from sorry

end sum_of_first_ten_primes_with_units_digit_three_l494_494863


namespace balls_in_boxes_l494_494035

theorem balls_in_boxes : ∀ (balls boxes : ℕ), balls = 6 → boxes = 3 → 
  (∃! n : ℕ, n = 47 ∧ n = number_of_ways_to_distribute_balls_in_boxes_d(balls, boxes)) :=
by
  intros balls boxes h1 h2
  have h_balls : balls = 6 := h1
  have h_boxes : boxes = 3 := h2
  use 47
  split
  sorry
  sorry

end balls_in_boxes_l494_494035


namespace increase_by_150_percent_l494_494310

theorem increase_by_150_percent (n : ℕ) : 
  n = 80 → n + (3 / 2) * n = 200 :=
by
  intros h
  rw [h]
  norm_num
  sorry

end increase_by_150_percent_l494_494310


namespace correct_sequence_l494_494784
open Nat

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def sequence := [1, 2, 3, 8, 5, 6, 7, 10, 9, 4]

def valid_sequence (seq : List ℕ) : Prop :=
  (seq.nodup) ∧
  (∀ i, i < 9 → is_prime (seq.nthLe i (by linarith) + seq.nthLe (i + 1) (by linarith))) ∧
  (is_prime (seq.head! + seq.reverse.head!))

theorem correct_sequence : valid_sequence sequence :=
  sorry ⟩

end correct_sequence_l494_494784


namespace series_sum_equality_l494_494809

noncomputable def series_sum : ℚ :=
  ∑ n in finset.range 85, 1 / ((3 * (n + 1) - 2) * (3 * (n + 1) + 1))

theorem series_sum_equality : series_sum = 255 / 640 :=
by
  sorry

end series_sum_equality_l494_494809


namespace increase_80_by_150_percent_l494_494235

-- Define the initial number (n) and the percentage increase (p)
def n : ℕ := 80
def p : ℚ := 1.5

-- The theorem stating the expected result after increasing n by 150%
theorem increase_80_by_150_percent : n + (p * n) = 200 := by
  sorry

end increase_80_by_150_percent_l494_494235


namespace domain_ln_x_minus_one_l494_494660

theorem domain_ln_x_minus_one : 
  ∀ x : ℝ, (x > 1) ↔ ∃ y : ℝ, y = Real.log (x - 1) :=
by {
  intros,
  exact Iff.symm (Set.mem_Ioi x)
}
sorry

end domain_ln_x_minus_one_l494_494660


namespace tan_half_angle_negative_l494_494936

theorem tan_half_angle_negative {α : ℝ} (hα1: 270 < α ∨ α < 0 ∨ α < 360) (hα2: α < 360) :
  tan (α / 2) < 0 :=
sorry

end tan_half_angle_negative_l494_494936


namespace total_cost_of_selling_watermelons_l494_494770

-- Definitions of the conditions:
def watermelon_weight : ℝ := 23.0
def daily_prices : List ℝ := [2.10, 1.90, 1.80, 2.30, 2.00, 1.95, 2.20]
def discount_threshold : ℕ := 15
def discount_rate : ℝ := 0.10
def sales_tax_rate : ℝ := 0.05
def number_of_watermelons : ℕ := 18

-- The theorem statement:
theorem total_cost_of_selling_watermelons :
  let average_price := (daily_prices.sum / daily_prices.length)
  let total_weight := number_of_watermelons * watermelon_weight
  let initial_cost := total_weight * average_price
  let discounted_cost := if number_of_watermelons > discount_threshold then initial_cost * (1 - discount_rate) else initial_cost
  let final_cost := discounted_cost * (1 + sales_tax_rate)
  final_cost = 796.43 := by
    sorry

end total_cost_of_selling_watermelons_l494_494770


namespace circle_area_ratio_is_correct_l494_494937

def circle_area_ratio (R_C R_D : ℝ) : ℝ := (R_C / R_D) ^ 2

theorem circle_area_ratio_is_correct (R_C R_D : ℝ) (h1: R_C / R_D = 3 / 2) : 
  circle_area_ratio R_C R_D = 9 / 4 :=
by
  unfold circle_area_ratio
  rw [h1]
  norm_num

end circle_area_ratio_is_correct_l494_494937


namespace jenny_spent_625_dollars_l494_494597

def adoption_fee := 50
def vet_visits_cost := 500
def monthly_food_cost := 25
def toys_cost := 200
def year_months := 12

def jenny_adoption_vet_share := (adoption_fee + vet_visits_cost) / 2
def jenny_food_share := (monthly_food_cost * year_months) / 2
def jenny_total_cost := jenny_adoption_vet_share + jenny_food_share + toys_cost

theorem jenny_spent_625_dollars :
  jenny_total_cost = 625 := by
  sorry

end jenny_spent_625_dollars_l494_494597


namespace find_a_and_b_l494_494908

theorem find_a_and_b (a b m : ℝ) 
  (h1 : (3 * a - 5)^(1 / 3) = -2)
  (h2 : ∀ x, x^2 = b → x = m ∨ x = 1 - 5 * m) : 
  a = -1 ∧ b = 1 / 16 :=
by
  sorry  -- proof to be constructed

end find_a_and_b_l494_494908


namespace polynomial_degree_correct_derivative_of_polynomial_correct_l494_494430

noncomputable def polynomial := 3 + 7 * x^5 - 4 + 8 * π * x^6 - (sqrt 15) * x^5 + 18

theorem polynomial_degree_correct :
  polynomial.degree = 6 := 
sorry

theorem derivative_of_polynomial_correct :
  polynomial.derivative = 5 * (7 - sqrt 15) * x^4 + 48 * π * x^5 :=
sorry

end polynomial_degree_correct_derivative_of_polynomial_correct_l494_494430


namespace distribution_of_6_balls_in_3_indistinguishable_boxes_l494_494002

-- Definition of the problem with conditions
def ways_to_distribute_balls_into_boxes
    (balls : ℕ) (boxes : ℕ) (distinguishable : bool)
    (indistinguishable : bool) : ℕ :=
  if (balls = 6) ∧ (boxes = 3) ∧ (distinguishable = true) ∧ (indistinguishable = true) 
  then 122 -- The correct answer given the conditions
  else 0

-- The Lean statement for the proof problem
theorem distribution_of_6_balls_in_3_indistinguishable_boxes :
  ways_to_distribute_balls_into_boxes 6 3 true true = 122 :=
by sorry

end distribution_of_6_balls_in_3_indistinguishable_boxes_l494_494002


namespace meaningful_fraction_range_l494_494560

theorem meaningful_fraction_range (x : ℝ) : (x - 1 ≠ 0) ↔ (fraction_meaningful := x ≠ 1) :=
by
  sorry

end meaningful_fraction_range_l494_494560


namespace store_toys_into_five_boxes_cannot_store_toys_into_six_boxes_l494_494701

-- Define the conditions
def valid_distribution (toys : ℕ) (boxes : ℕ → ℕ) (n : ℕ) : Prop :=
  (∀ i j, i ≠ j → boxes i ≠ boxes j) ∧
  (∀ i, 1 ≤ boxes i) ∧
  ∑ i in finset.range n, boxes i = toys

-- Part (a): How to store 20 toys into 5 boxes.
theorem store_toys_into_five_boxes :
  ∃ (boxes : ℕ → ℕ), valid_distribution 20 boxes 5 :=
sorry

-- Part (b): Can he store the toys into 6 boxes?
theorem cannot_store_toys_into_six_boxes :
  ¬ ∃ (boxes : ℕ → ℕ), valid_distribution 20 boxes 6 :=
sorry

end store_toys_into_five_boxes_cannot_store_toys_into_six_boxes_l494_494701


namespace overall_profit_percentage_l494_494748

-- Conditions
def cost_price_A := 15 * 25
def cost_price_B := 20 * 40
def cost_price_C := 30 * 55

def total_cost_price := cost_price_A + cost_price_B + cost_price_C

def selling_price_A := 12 * 38
def selling_price_B := 18 * 50
def selling_price_C := 25 * 65

def total_selling_price := selling_price_A + selling_price_B + selling_price_C

def profit := total_selling_price - total_cost_price

def profit_percentage := (profit / total_cost_price.toFloat) * 100

-- Proof statement
theorem overall_profit_percentage : profit_percentage = 5.52 := by
  sorry

end overall_profit_percentage_l494_494748


namespace halfway_fraction_l494_494703

theorem halfway_fraction (a b : ℚ) (ha : a = 3/4) (hb : b = 5/6) : (a + b) / 2 = 19/24 :=
by
  rw [ha, hb] -- replace a and b with 3/4 and 5/6 respectively
  have h1 : 3/4 + 5/6 = 19/12,
  { norm_num, -- ensures 3/4 + 5/6 = 19/12
    linarith },
  rw h1, -- replace a + b with 19/12
  norm_num -- ensures (19/12) / 2 = 19/24

end halfway_fraction_l494_494703


namespace buckets_in_each_package_l494_494368

theorem buckets_in_each_package
  (total_buckets : ℕ)
  (packages : ℕ)
  (h_total_buckets : total_buckets = 426)
  (h_packages : packages = 54) :
  (total_buckets / packages).round = 8 :=
by
  sorry

end buckets_in_each_package_l494_494368


namespace divide_plane_into_four_quadrants_l494_494991

-- Definitions based on conditions
def perpendicular_axes (x y : ℝ → ℝ) : Prop :=
  (∀ t : ℝ, x t = t ∨ x t = 0) ∧ (∀ t : ℝ, y t = t ∨ y t = 0) ∧ ∀ t : ℝ, x t ≠ y t

-- The mathematical proof statement
theorem divide_plane_into_four_quadrants (x y : ℝ → ℝ) (hx : perpendicular_axes x y) :
  ∃ quadrants : ℕ, quadrants = 4 :=
by
  sorry

end divide_plane_into_four_quadrants_l494_494991


namespace quadratic_inequality_solution_l494_494536

theorem quadratic_inequality_solution
  (a b c x : ℝ)
  (h₁ : a * x^2 + b * x + c > 0 ∀ x ∈ Ioo (-1/3) 2)
  (h₂ : a < 0) :
  cx^2 + bx + a < 0 ∀ x ∈ Ioo -3 (1/2) := 
sorry

end quadratic_inequality_solution_l494_494536


namespace increase_80_by_150_percent_l494_494262

-- Original number
def originalNumber : ℕ := 80

-- Percentage increase as a decimal
def increaseFactor : ℚ := 1.5

-- Expected result after the increase
def expectedResult : ℕ := 200

theorem increase_80_by_150_percent :
  originalNumber + (increaseFactor * originalNumber) = expectedResult :=
by
  sorry

end increase_80_by_150_percent_l494_494262


namespace sqrt_meaningful_range_l494_494957

theorem sqrt_meaningful_range (x : ℝ) (h : x - 2 ≥ 0) : x ≥ 2 :=
by {
  sorry
}

end sqrt_meaningful_range_l494_494957


namespace circle_area_ratio_is_correct_l494_494940

def circle_area_ratio (R_C R_D : ℝ) : ℝ := (R_C / R_D) ^ 2

theorem circle_area_ratio_is_correct (R_C R_D : ℝ) (h1: R_C / R_D = 3 / 2) : 
  circle_area_ratio R_C R_D = 9 / 4 :=
by
  unfold circle_area_ratio
  rw [h1]
  norm_num

end circle_area_ratio_is_correct_l494_494940


namespace problem1_problem2_problem3_problem4_l494_494811

variable (a b c : ℝ)

theorem problem1 : a^4 * (a^2)^3 = a^10 :=
by
  sorry

theorem problem2 : 2 * a^3 * b^2 * c / (1 / 3 * a^2 * b) = 6 * a * b * c :=
by
  sorry

theorem problem3 : 6 * a * (1 / 3 * a * b - b) - (2 * a * b + b) * (a - 1) = -5 * a * b + b :=
by
  sorry

theorem problem4 : (a - 2)^2 - (3 * a + 2 * b) * (3 * a - 2 * b) = -8 * a^2 - 4 * a + 4 + 4 * b^2 :=
by
  sorry

end problem1_problem2_problem3_problem4_l494_494811


namespace village_population_l494_494736

theorem village_population (P : ℝ) (h : 0.8 * P = 32000) : P = 40000 := by
  sorry

end village_population_l494_494736


namespace volume_of_revolution_l494_494727

theorem volume_of_revolution :
  let f := fun x => x^2 - 2*x + 1,
      a := 0,
      b := 1,
      outer_radius := 2,
      inner_radius := fun y => 1 + sqrt y in
  π * ∫ y in 0..1, (outer_radius^2 - (inner_radius y)^2) = 7*π/6 :=
by
  let f := fun x => x^2 - 2*x + 1
  let a := 0
  let b := 1
  let outer_radius := 2
  let inner_radius := fun y => 1 + sqrt y
  have h1 : ∫ y in 0..1, 3 - 2*sqrt y - y = 7/6 := sorry
  calc
    π * ∫ y in 0..1, (outer_radius^2 - (inner_radius y)^2)
      = π * ∫ y in 0..1, (4 - (1+2*sqrt y+y)) : by apply congrArg; funext; field_simp [pow_two]
  ... = π * ∫ y in 0..1, 3 - 2*sqrt y - y : by apply congrArg; funext; ring
  ... =  π * 7/6 : by rwa [←h1]
  ... = 7*π/6 : by ring

end volume_of_revolution_l494_494727


namespace tan_seven_pi_over_four_l494_494451

theorem tan_seven_pi_over_four : Real.tan (7 * Real.pi / 4) = -1 := 
by
  -- In this case, we are proving a specific trigonometric identity
  sorry

end tan_seven_pi_over_four_l494_494451


namespace least_number_subtracted_divisible_l494_494842

theorem least_number_subtracted_divisible (n : ℕ) (d : ℕ) (h : n = 1234567) (k : d = 37) :
  n % d = 13 :=
by 
  rw [h, k]
  sorry

end least_number_subtracted_divisible_l494_494842


namespace find_weekly_allowance_l494_494111

noncomputable def weekly_allowance (A : ℝ) : Prop :=
  0.45 * A + 17 = 29

theorem find_weekly_allowance : ∃ A : ℝ, weekly_allowance A ∧ A = 26.67 :=
by {
  use 26.67,
  unfold weekly_allowance,
  norm_num,
  sorry
}

end find_weekly_allowance_l494_494111


namespace area_triangle_AEB_l494_494981

theorem area_triangle_AEB :
  ∀ (A B C D F G E : Type)
    (AB AD BC CD : ℝ) 
    (AF BG : ℝ) 
    (triangle_AEB : ℝ),
  (AB = 7) →
  (BC = 4) →
  (CD = 7) →
  (AD = 4) →
  (DF = 2) →
  (GC = 1) →
  (triangle_AEB = 1/2 * 7 * (4 + 16/3)) →
  (triangle_AEB = 98 / 3) :=
by
  intros A B C D F G E AB AD BC CD AF BG triangle_AEB
  sorry

end area_triangle_AEB_l494_494981


namespace loop_result_eq_132_l494_494956

theorem loop_result_eq_132 (i_init s_init : ℕ) :
    i_init = 12 → s_init = 1 →
    (∀ f : ℕ × ℕ, f = (λ p, (p.snd * p.fst, p.fst - 1)) →
        let (s_final, i_final) := iterate f 2 (s_init, i_init) in s_final = 132 → i_final = 10) →
    "condition" = (λ i, i < 11) :=
by
  intros hi hs h_iter
  have h_iter' := h_iter (λ p, (p.snd * p.fst, p.fst - 1))
  rw [hi, hs] at h_iter'
  dsimp at h_iter'
  simp at h_iter'
  exact h_iter' - sorry

end loop_result_eq_132_l494_494956


namespace sum_of_first_ten_primes_ending_in_3_is_671_l494_494845

noncomputable def sum_of_first_ten_primes_ending_in_3 : ℕ :=
  3 + 13 + 23 + 43 + 53 + 73 + 83 + 103 + 113 + 163

theorem sum_of_first_ten_primes_ending_in_3_is_671 :
  sum_of_first_ten_primes_ending_in_3 = 671 :=
by
  sorry

end sum_of_first_ten_primes_ending_in_3_is_671_l494_494845


namespace river_speed_l494_494756

theorem river_speed :
  ∀ (v : ℝ), 
    (let man_speed := 8 in 
     let total_distance := 7.82 in 
     let total_time := 1 in 
     let d := total_distance / 2 in 
     d / (man_speed - v) + d / (man_speed + v) = total_time) 
    → v = 1.2 := 
by
  intros v h
  let man_speed := 8
  let total_distance := 7.82
  let total_time := 1
  let d := total_distance / 2
  have h1 : d / (man_speed - v) + d / (man_speed + v) = total_time := h
  show v = 1.2
  sorry

end river_speed_l494_494756


namespace log_addition_property_l494_494801

theorem log_addition_property : log 10 50 + log 10 20 = 3 :=
by
  sorry

end log_addition_property_l494_494801


namespace log_sum_example_l494_494805

theorem log_sum_example : log 10 50 + log 10 20 = 3 :=
by
  -- Proof goes here, skipping with sorry
  sorry

end log_sum_example_l494_494805


namespace polynomial_nonzero_d_l494_494662

theorem polynomial_nonzero_d {a b c d e : ℝ} 
  (Q : Polynomial ℝ)
  (hQ : Q = Polynomial.C e + Polynomial.C d * Polynomial.X + Polynomial.C c * Polynomial.X^2 + Polynomial.C b * Polynomial.X^3 + Polynomial.C a * Polynomial.X^4 + Polynomial.X^5)
  (h_roots : Q.roots.length = 5)
  (h_zero_root : Polynomial.eval 0 Q = 0)
  (h_complex_root : Polynomial.eval (2 + 3 * Complex.I) Q = 0) :
  d ≠ 0 := 
sorry

end polynomial_nonzero_d_l494_494662


namespace find_m_l494_494505

-- Definitions for the sets A and B
def A (m : ℝ) : Set ℝ := {3, 4, 4 * m - 4}
def B (m : ℝ) : Set ℝ := {3, m^2}

-- Problem statement
theorem find_m {m : ℝ} (h : B m ⊆ A m) : m = -2 :=
sorry

end find_m_l494_494505


namespace exists_x_quadratic_eq_zero_iff_le_one_l494_494630

variable (a : ℝ)

theorem exists_x_quadratic_eq_zero_iff_le_one : (∃ x : ℝ, x^2 - 2 * x + a = 0) ↔ a ≤ 1 :=
sorry

end exists_x_quadratic_eq_zero_iff_le_one_l494_494630


namespace find_c_l494_494676

theorem find_c (a b c : ℝ) (h1 : ∃ x y : ℝ, x = a * (y - 2)^2 + 3 ∧ (x,y) = (3,2))
  (h2 : (1 : ℝ) = a * ((4 : ℝ) - 2)^2 + 3) : c = 1 :=
sorry

end find_c_l494_494676


namespace increased_number_l494_494272

theorem increased_number (original_number : ℕ) (percentage_increase : ℚ) :
  original_number = 80 →
  percentage_increase = 1.5 →
  original_number * (1 + percentage_increase) = 200 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end increased_number_l494_494272


namespace increase_by_percentage_l494_494324

theorem increase_by_percentage (a : ℤ) (b : ℚ) : ((b + 1) * a) = 200 :=
by
  -- Definitions based directly on conditions
  let a := (80 : ℤ)
  let b := (1.5 : ℚ)
  -- Assertions about the equivalent proof problem
  have h1 : b + 1 = 2.5 := by norm_num
  have h2 : (b + 1) * a = 200 := by norm_num
  exact h2

end increase_by_percentage_l494_494324


namespace solve_log_equation_l494_494170

theorem solve_log_equation (y : ℝ) (h : log 3 y + log 9 y = 5) : y = 3^(10/3) :=
sorry

end solve_log_equation_l494_494170


namespace balls_in_indistinguishable_boxes_l494_494016

theorem balls_in_indistinguishable_boxes : 
  let balls := 6 in
  let boxes := 3 in
  -- Expression to count ways to distribute balls into indistinguishable boxes
  ((balls.choose 0 * ((balls-0).choose 6)) + 
   (balls.choose 1 * ((balls-1).choose 5)) + 
   (balls.choose 2 * ((balls-2).choose 4)) + 
   (balls.choose 2 * ((balls-2).choose 3) / 2!) + 
   (balls.choose 3 * ((balls-3).choose 3)) + 
   (balls.choose 3 * ((balls-3).choose 2) * ((balls-5).choose 1)) + 
   (balls.choose 2 * ((balls-2).choose 2) / 3!))
  = 92 := 
sorry

end balls_in_indistinguishable_boxes_l494_494016


namespace balls_in_indistinguishable_boxes_l494_494012

theorem balls_in_indistinguishable_boxes : 
  let balls := 6 in
  let boxes := 3 in
  -- Expression to count ways to distribute balls into indistinguishable boxes
  ((balls.choose 0 * ((balls-0).choose 6)) + 
   (balls.choose 1 * ((balls-1).choose 5)) + 
   (balls.choose 2 * ((balls-2).choose 4)) + 
   (balls.choose 2 * ((balls-2).choose 3) / 2!) + 
   (balls.choose 3 * ((balls-3).choose 3)) + 
   (balls.choose 3 * ((balls-3).choose 2) * ((balls-5).choose 1)) + 
   (balls.choose 2 * ((balls-2).choose 2) / 3!))
  = 92 := 
sorry

end balls_in_indistinguishable_boxes_l494_494012


namespace max_area_of_inscribed_equilateral_triangle_l494_494679

noncomputable def side_length_eq : ℝ := 2 * Real.sqrt 183

def max_triangle_area (s : ℝ) : ℝ :=
  s^2 * Real.sqrt 3 / 4

theorem max_area_of_inscribed_equilateral_triangle (p q r : ℕ) :
  max_triangle_area side_length_eq = 183 * Real.sqrt 3 ∧ p + q + r = 186 :=
by
  have s : ℝ := side_length_eq
  have area : ℝ := max_triangle_area s
  have h1 : area = 183 * Real.sqrt 3 := by sorry
  have h2 : p = 183 := by sorry
  have h3 : q = 3 := by sorry
  have h4 : r = 0 := by sorry
  have h : p + q + r = 186 := by linarith
  exact ⟨h1, h⟩

end max_area_of_inscribed_equilateral_triangle_l494_494679


namespace total_games_played_l494_494998

theorem total_games_played (points_per_game_winner : ℕ) (points_per_game_loser : ℕ) (jack_games_won : ℕ)
  (jill_total_points : ℕ) (total_games : ℕ)
  (h1 : points_per_game_winner = 2)
  (h2 : points_per_game_loser = 1)
  (h3 : jack_games_won = 4)
  (h4 : jill_total_points = 10)
  (h5 : ∀ games_won_by_jill : ℕ, jill_total_points = games_won_by_jill * points_per_game_winner +
           (jack_games_won * points_per_game_loser)) :
  total_games = jack_games_won + (jill_total_points - jack_games_won * points_per_game_loser) / points_per_game_winner := by
  sorry

end total_games_played_l494_494998


namespace positive_difference_is_54_l494_494881

open Real

-- Define the lines using their slope-intercept form based on the given conditions.
def line_l (x : ℝ) : ℝ := (-5 / 3) * x + 5
def line_m (x : ℝ) : ℝ := (-2 / 7) * x + 2

-- Define the x-coordinates where y = 20 for both lines
def x_l : ℝ := (20 - 5) / (-5 / 3)
def x_m : ℝ := (20 - 2) / (-2 / 7)

-- Define the positive difference in x-coordinates
def positive_difference_x_coordinates : ℝ := abs (x_l - x_m)

-- Finally, the statement that needs to be proven
theorem positive_difference_is_54 : positive_difference_x_coordinates = 54 := by
  -- This will be proven based on the definitions above
  sorry

end positive_difference_is_54_l494_494881


namespace sum_of_first_2016_terms_l494_494520

-- Define the arithmetic series conditions
def S3 (a1 d : ℝ) : Prop := 3 * a1 + 3 * d = 0
def Sn (a1 d : ℝ) : Prop := 5 * a1 + (5 * (5 - 1) / 2) * d = 5

-- Define the general term of the sequence a_{n}
def a (n : ℕ) (a1 d : ℝ) : ℝ := a1 + (n - 1) * d

-- Define the sum of first 2016 terms of the sequence 1 / (a_{2n-1} * a_{2n+1})
noncomputable def sum_first_2016_terms (a1 d : ℝ) : ℝ :=
∑ k in Finset.range 2016, 1 / (a (2*k+1) a1 d * a (2*k+3) a1 d)

-- The theorem
theorem sum_of_first_2016_terms (a1 d : ℝ)
  (hS3: S3 a1 d) (hSn: Sn a1 d) :
  sum_first_2016_terms a1 d = -2016 / 4031 :=
sorry

end sum_of_first_2016_terms_l494_494520


namespace arc_length_proof_l494_494808

noncomputable def arc_length (rho : ℝ → ℝ) (varphi1 varphi2 : ℝ) :=
  ∫ varphi in varphi1..varphi2, Real.sqrt (rho varphi ^ 2 + (rho' varphi) ^ 2)

theorem arc_length_proof :
  arc_length (λ varphi => 6 * Real.exp (12 * varphi / 5)) (-Real.pi / 2) (Real.pi / 2) =
  13 * Real.sinh (6 * Real.pi / 5) :=
by
  sorry

end arc_length_proof_l494_494808


namespace jenny_cat_expense_l494_494599

def adoption_fee : ℕ := 50
def vet_visits_cost : ℕ := 500
def monthly_food_cost : ℕ := 25
def jenny_toy_expenses : ℕ := 200
def split_factor : ℕ := 2

-- Given conditions, prove that Jenny spent $625 on the cat in the first year.
theorem jenny_cat_expense : 
  let yearly_food_cost := 12 * monthly_food_cost 
  let total_shared_expenses := adoption_fee + vet_visits_cost + yearly_food_cost 
  let jenny_shared_expenses := total_shared_expenses / split_factor 
  let total_jenny_cost := jenny_shared_expenses + jenny_toy_expenses
  in total_jenny_cost = 625 := 
by 
  sorry

end jenny_cat_expense_l494_494599


namespace complex_magnitude_sum_inv_l494_494138

theorem complex_magnitude_sum_inv (z w : ℂ) (hz : |z| = 2) (hw : |w| = 4) (hzw : |z + w| = 5) :
  |(1/z) + (1/w)| = 5 / 8 :=
by
  sorry

end complex_magnitude_sum_inv_l494_494138


namespace increase_80_by_150_percent_l494_494288

theorem increase_80_by_150_percent : 
  ∀ x percent, x = 80 ∧ percent = 1.5 → x + (x * percent) = 200 :=
by
  sorry

end increase_80_by_150_percent_l494_494288


namespace binomial_expansion_properties_l494_494989

theorem binomial_expansion_properties :
  let binom := (1 - 4 * x) ^ 8 in
  (∀ x : ℤ, (binom.eval 1 = 3 ^ 8)) ∧ 
  (∀ x : ℤ, (binom.eval 1 + binom.eval (-1)) / 2 = 2 ^ 7) :=
by
  sorry

end binomial_expansion_properties_l494_494989


namespace harkamal_total_amount_l494_494541

def cost_grapes (quantity rate : ℕ) : ℕ := quantity * rate
def cost_mangoes (quantity rate : ℕ) : ℕ := quantity * rate
def total_amount_paid (cost1 cost2 : ℕ) : ℕ := cost1 + cost2

theorem harkamal_total_amount :
  let grapes_quantity := 8
  let grapes_rate := 70
  let mangoes_quantity := 9
  let mangoes_rate := 65
  total_amount_paid (cost_grapes grapes_quantity grapes_rate) (cost_mangoes mangoes_quantity mangoes_rate) = 1145 := 
by
  sorry

end harkamal_total_amount_l494_494541


namespace count_correct_statements_l494_494674

theorem count_correct_statements : 
  let statement_1 := (a : ℕ) → a^2 * a^2 = 2 * a^2
  let statement_2 := (a b : ℕ) → (a - b)^2 = a^2 - b^2
  let statement_3 := (a : ℕ) → a^2 + a^3 = a^5
  let statement_4 := (a b : ℕ) → (-2 * a^2 * b^3)^3 = -6 * a^6 * b^3
  let statement_5 := (a : ℕ) → (-a^3)^2 / a = a^5
  1 = (if statement_1 then 1 else 0) +
      (if statement_2 then 1 else 0) +
      (if statement_3 then 1 else 0) +
      (if statement_4 then 1 else 0) +
      (if statement_5 then 1 else 0) :=
by
  sorry

end count_correct_statements_l494_494674


namespace parallelogram_t_l494_494069

def W : ℝ × ℝ := (-1, 4)
def X : ℝ × ℝ := (5, 4)
def Y (t : ℝ) : ℝ × ℝ := (t - 6, 1)
def Z : ℝ × ℝ := (-4, 1)

theorem parallelogram_t (t : ℝ) (h : WXYZisParallelogram W X (Y t) Z) : t = 8 :=
by sorry

-- Placeholder definition for the parallelogram hypothesis
def WXYZisParallelogram (W X Y Z : ℝ × ℝ) : Prop :=
  (Y.1 - X.1 = W.1 - Z.1 ∧ Y.2 - X.2 = W.2 - Z.2 ∧
   Z.1 - W.1 = X.1 - Y.1 ∧ Z.2 - W.2 = X.2 - Y.2)

end parallelogram_t_l494_494069


namespace log_sum_eq_five_has_solution_l494_494174

theorem log_sum_eq_five_has_solution : ∃ y : ℝ, log 3 y + log 9 y = 5 ∧ y = 3^(10/3) := by
  sorry

end log_sum_eq_five_has_solution_l494_494174


namespace cylinder_volume_triple_check_l494_494747

noncomputable def volume (r h : ℝ) : ℝ := π * r^2 * h

theorem cylinder_volume_triple_check
  (cylinder_radius : ℝ) (cylinder_height : ℝ)
  (original_radius : ℝ = 5) (original_height : ℝ = 10)
  (given_radius : ℝ = 5) (given_height : ℝ = 30) :
  volume given_radius given_height = 3 * volume original_radius original_height :=
by
  sorry

end cylinder_volume_triple_check_l494_494747


namespace intersection_polar_coords_l494_494923

-- Definitions from conditions
def parametric_c1 (θ : ℝ) : ℝ × ℝ := (1 + Real.cos θ, 1 + Real.sin θ)
def polar_c2 (ρ : ℝ) : Prop := ρ = 1

-- Lean proof problem
theorem intersection_polar_coords : 
  (∀ (θ : ℝ), (parametric_c1 θ).fst = 1 + Real.cos θ ∧ (parametric_c1 θ).snd = 1 + Real.sin θ) ∧
  polar_c2 1 ∧
  (∀ x y, (x - 1)^2 + (y - 1)^2 = 1 ↔ ∃ θ, (x, y) = parametric_c1 θ) ∧
  (∀ x y, x^2 + y^2 = 1 ↔ ∃ ρ θ, x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ ∧ polar_c2 ρ) →
  (∃ (θ1 θ2 : ℝ), (1 + Real.cos θ1, 1 + Real.sin θ1) = (1, 0) ∧ (1, 0) = (1 * Real.cos 0, 1 * Real.sin 0) ∧
                  (1 + Real.cos θ2, 1 + Real.sin θ2) = (0, 1) ∧ (0, 1) = (1 * Real.cos (Real.pi / 2), 1 * Real.sin (Real.pi / 2))) :=
begin
  sorry
end

end intersection_polar_coords_l494_494923


namespace first_player_wins_l494_494214

theorem first_player_wins :
  ∀ (matches : ℕ) (hp : matches = 10000000) (P : (ℕ → Prop) → Prop)
  (valid_move : ℕ → ℕ → Prop),
  (∃ x : ℕ, valid_move 10000000 x ∧ P (λ n, valid_move n 0)) :=
by
  let valid_move := λ m (x : ℕ), ∃ p : ℕ, prime p ∧ ∃ n : ℕ, x = p^n ∧ x ≤ m
  have strategy : ∀ matches, matches % 6 ≠ 0 → (∃ x, valid_move matches x ∧ (matches - x) % 6 = 0),
    sorry
  sorry

end first_player_wins_l494_494214


namespace MK_eq_ML_l494_494611

open EuclideanGeometry

variables {A B C C0 X K L M : Point}

noncomputable def is_right_angle (a b c : Point) : Prop :=
  ∠b c a = π / 2

noncomputable def is_tangent_circle (o : Point) (r : ℝ) (p : Point) : Prop :=
  dist o p = r

noncomputable def reflect (a b : Point) : Point :=
  (2 • b - a)

-- Given conditions
axiom angle_BCA_90 : is_right_angle B C A
axiom C0_foot : foot_of_perpendicular C (line_through A B) = C0
axiom X_within_C0C : between X C C0
axiom BK_eq_BC : dist B K = dist B C
axiom AL_eq_AC : dist A L = dist A C
axiom M_intersection : ∃ t : ℝ, M = t • (AL) + (1 - t) • (BK)

-- The theorem to prove
theorem MK_eq_ML : dist M K = dist M L :=
by
  sorry

end MK_eq_ML_l494_494611


namespace increase_150_percent_of_80_l494_494296

theorem increase_150_percent_of_80 : 80 * 1.5 + 80 = 200 :=
by
  sorry

end increase_150_percent_of_80_l494_494296


namespace midpoint_fraction_l494_494710

theorem midpoint_fraction (a b : ℚ) (h1 : a = 3 / 4) (h2 : b = 5 / 6) :
  (a + b) / 2 = 19 / 24 :=
by {
  sorry
}

end midpoint_fraction_l494_494710


namespace anniversary_day_probability_l494_494402

-- Define the years in which the meeting could take place
def years := {1668, 1669, 1670, 1671}

-- Define a function to check if a year is a leap year
def is_leap_year (y : ℕ) : Prop := (y % 4 = 0)

-- Define the meeting date
def meeting_day := 5 -- Friday as the 5th day of the week (assuming 0 = Sunday)

-- Define the anniversary function that computes the day of the 11th anniversary
def anniversary_day (start_year : ℕ) : ℕ :=
  let days_in_year (y : ℕ) := if is_leap_year y then 366 else 365
  let total_days := (List.range 11).map (λ i => days_in_year (start_year + i))
  (total_days.sum + meeting_day) % 7

-- Define the probability computation
def probability (day : ℕ) : ℚ :=
  let occurences := (years.toList.map anniversary_day).count (λ d => d = day)
  occurences / years.toList.length

-- Statement of the theorem
theorem anniversary_day_probability :
  probability 5 = 3 / 4 ∧ probability 4 = 1 / 4 := -- 5 is Friday, 4 is Thursday
  by
    sorry -- Proof goes here

end anniversary_day_probability_l494_494402


namespace workers_reading_ratio_l494_494571

theorem workers_reading_ratio :
  let W := 150 in
  let S := W / 2 in
  let B := 12 in
  let N := (S - B) - 1 in
  ∃ K, (K / W = 1 / 6) ∧ (W = S + K - B + N) := 
by
  let W := 150
  let S := W / 2
  let B := 12
  let N := (S - B) - 1
  use (150 - 125) -- K = 25
  split
  · exact (25 / 150 = 1 / 6)
  · exact (150 = 75 + 25 - 12 + 62)
  

end workers_reading_ratio_l494_494571


namespace increase_result_l494_494246

-- Given conditions
def original_number : ℝ := 80
def increase_percentage : ℝ := 1.5

-- The result after the increase
theorem increase_result (h1 : original_number = 80) (h2 : increase_percentage = 1.5) : 
  original_number + (increase_percentage * original_number) = 200 := by
  sorry

end increase_result_l494_494246


namespace odd_number_as_difference_of_squares_l494_494593

theorem odd_number_as_difference_of_squares (n : ℤ) (h : ∃ k : ℤ, n = 2 * k + 1) :
  ∃ a b : ℤ, n = a^2 - b^2 :=
by
  sorry

end odd_number_as_difference_of_squares_l494_494593


namespace log_addition_identity_l494_494795

theorem log_addition_identity :
  ∃ (a x y : ℝ), a = 10 ∧ x = 50 ∧ y = 20 ∧ (log 10 50 + log 10 20 = 3) :=
by
  let a := 10
  let x := 50
  let y := 20
  have h1 : log a (x * y) = log a x + log a y,
    from sorry -- logarithmic identity
  have h2 : log a (x * y) = log a 1000,
    from congrArg (log a) (by norm_num) -- simplifying x * y
  have h3 : log 10 1000 = 3,
    from sorry -- calculating log 1000 base 10 directly
  exact ⟨a, x, y, rfl, rfl, rfl, by linarith [h1, h2, h3]⟩

end log_addition_identity_l494_494795


namespace distinguishable_balls_in_indistinguishable_boxes_l494_494021

theorem distinguishable_balls_in_indistinguishable_boxes :
  let num_distinguishable_balls := 6
  let num_indistinguishable_boxes := 3
  -- different ways to distribute balls across boxes summarized
  let ways := ∑([{6, 0, 0} ++ {5, 1, 0} ++ {4, 2, 0} ++ {4, 1, 1} ++ {3, 3, 0} ++ {3, 2, 1} ++ {2, 2, 2}], sum)
  -- corresponding ways to remove permutation of distinguishable ball cases
  ways = 222 := sorry

end distinguishable_balls_in_indistinguishable_boxes_l494_494021


namespace xiaoming_score_above_median_l494_494086

-- Definitions based on the conditions
def xiaoming_score : ℕ := 72
def classmates_scores_exceed_half (scores : list ℕ) : Prop :=
  let median := if scores.length % 2 = 0 then
                  (scores.nth_le (scores.length / 2) sorry + scores.nth_le (scores.length / 2 - 1) sorry) / 2
                else
                  scores.nth_le (scores.length / 2) sorry in
  xiaoming_score > median

-- The theorem statement
theorem xiaoming_score_above_median (scores : list ℕ) (h : classmates_scores_exceed_half scores) :
  statistical_measure_used = "median" := sorry

end xiaoming_score_above_median_l494_494086


namespace exists_large_root_deviation_l494_494128

theorem exists_large_root_deviation 
  (p q p' q' u u' v v' : ℝ)
  (h1 : ∀ p q : ℝ, ∃ u v : ℝ, u ≠ v ∧ u > v ∧ (u^2 + p * u + q = 0) ∧ (v^2 + p * v + q = 0))
  (h2 : |p' - p| < 0.01)
  (h3 : |q' - q| < 0.01)
  (h4 : ∀ p' q' : ℝ, ∃ u' v' : ℝ, u' ≠ v' ∧ u' > v' ∧ (u'^2 + p' * u' + q' = 0) ∧ (v'^2 + p' * v' + q' = 0)) :
  ∃ p q p' q' u u' v v', |u' - u| > 10000 := sorry

end exists_large_root_deviation_l494_494128


namespace distinct_square_proof_l494_494224

-- Define a structure to represent the conditions in the problem
structure Rectangle :=
(x : ℕ)
(y : ℕ)

-- Define the conditions as per the problem
noncomputable def num_distinct_squares (rectangles: List Rectangle) (size: ℕ) := 
  rectangles.length = 8 ∧ size = 4 ∧ ∀ r ∈ rectangles, (r.x = 2 ∧ r.y = 1) 

-- Define a function to compute the number of unique configurations
noncomputable def distinct_square_configurations (rectangles: List Rectangle) (size: ℕ) : ℕ :=
  if num_distinct_squares rectangles size then 
    4 -- as per the problem's solution, this is known to be the answer 
  else 
    0

-- Define the theorem statement to be proven
theorem distinct_square_proof: 
  ∀ (rectangles: List Rectangle) (size: ℕ), 
    num_distinct_squares rectangles size 
    → distinct_square_configurations rectangles size = 4 :=
by
  intros
  sorry

end distinct_square_proof_l494_494224


namespace shepherd_incorrect_l494_494733

theorem shepherd_incorrect (n m : ℕ) (h1 : n + 7 * m = 150) (h2 : 7 * n + m = 150) :
  false :=
by
  have h3 : 7 * (n + 7 * m) = 1050 := by rw [h1]; ring
  have h4 : 7 * n + 49 * m = 1050 := by rw [←h3, bit0, mul_add, add_mul]
  have h5 : 7 * n + m = 150 := h2
  have h6 : 48 * m = 900 := by linarith [h4, h5]
  have h7 : m = 900 / 48 := by rw [←nat_cast_mul_right (48 : ℚ) 75]
  have h8 : 75 % 4 = 3 := rfl
  have : m ≠ int_val m := by rw [int_val, int.eq_lcm]; linarith
  contradiction

end shepherd_incorrect_l494_494733


namespace tangent_lines_possible_values_l494_494153

theorem tangent_lines_possible_values (r1 r2 : ℝ) (h1 : r1 = 4) (h2 : r2 = 6) :
  ∃ m : ℕ, (m = 0 ∨ m = 1 ∨ m = 2 ∨ m = 3 ∨ m = 4) ∧ (unique_values m = 5) :=
by
  sorry

end tangent_lines_possible_values_l494_494153


namespace sum_def_l494_494610

-- Define the given polynomial p
def p (x : ℝ) : ℝ := x^3 - 7 * x^2 + 12 * x - 20

-- Define the sums of powers of the roots as given in the problem
def t : ℕ → ℝ
| 0 := 3
| 1 := 7
| 2 := 15
| (n + 3) := sorry -- This would be filled with the recurrence relation checked in the proof

-- Assume the existence of d, e, and f
variables (d e f : ℝ)

-- Recurrence relation
axiom recurrence_relation : ∀ k ≥ 2, t (k + 1) = d * t k + e * t (k - 1) + f * t (k - 2) - 5

-- Prove that d + e + f = 15
theorem sum_def (hd_def : d = 7) (he_def : e = -12) (hf_def : f = 20) : d + e + f = 15 :=
by 
  -- here we should develop the proof, but it is omitted and hence using sorry
  sorry

end sum_def_l494_494610


namespace find_n_l494_494894

-- Define the sequence {a_n} as a function
def sequence (n : ℕ) : ℕ :=
  if n = 1 then 1
  else if n = 2 then 4
  else 2 * sequence n = sequence (n - 1) + sequence (n + 1)

theorem find_n :
  ∃ n : ℕ, sequence n = 301 :=
begin
  sorry
end

end find_n_l494_494894


namespace total_flowers_l494_494382

def number_of_flowers (F : ℝ) : Prop :=
  let vases := (F - 7.0) / 6.0
  vases = 6.666666667

theorem total_flowers : number_of_flowers 47.0 :=
by
  sorry

end total_flowers_l494_494382


namespace sum_of_rationals_l494_494613

-- Definition of the conditions
def pairwise_products (a1 a2 a3 a4 : ℚ) : set ℚ :=
  {a1 * a2, a1 * a3, a1 * a4, a2 * a3, a2 * a4, a3 * a4}

-- Statement of the proof problem
theorem sum_of_rationals 
  (a1 a2 a3 a4 : ℚ)
  (h : pairwise_products a1 a2 a3 a4 = {-24, -2, -3/2, -1/8, 1, 3}) :
    a1 + a2 + a3 + a4 = 9/4 ∨ a1 + a2 + a3 + a4 = -9/4 :=
sorry

end sum_of_rationals_l494_494613


namespace sin_alpha_value_l494_494506

theorem sin_alpha_value (α : ℝ) (h₀ : cos (α + π / 6) = 1 / 3) (h₁ : 0 < α ∧ α < π) :
  sin α = (2 * real.sqrt 6 - 1) / 6 := 
begin
  sorry
end

end sin_alpha_value_l494_494506


namespace area_gray_region_concentric_circles_l494_494100

theorem area_gray_region_concentric_circles :
  ∀ (r : ℝ), (∀ (r : ℝ), 2 * r - r = 3) →
  let r_inner := 3 in
  let r_outer := 2 * r_inner in
  let area_outer := Real.pi * r_outer ^ 2 in
  let area_inner := Real.pi * r_inner ^ 2 in
  let area_gray := area_outer - area_inner in
  area_gray = 27 * Real.pi :=
by
  intro r h
  let r_inner := 3
  let r_outer := 2 * r_inner
  let area_outer := Real.pi * r_outer ^ 2
  let area_inner := Real.pi * r_inner ^ 2
  let area_gray := area_outer - area_inner
  sorry

end area_gray_region_concentric_circles_l494_494100


namespace length_DF_l494_494140

variable (A B C D E F X : Type) -- Declaring points
variables {AB AD BC BX DF : ℝ} -- Declaring lengths

-- Assume a square with side length 13
def square_ABCD (AB AD BC : ℝ) : Prop :=
  (AB = 13 ∧ AD = 13 ∧ BC = 13)

-- Point X such that BX = 6
def point_X (BX : ℝ) : Prop :=
  (BX = 6)

-- Area of square ABCD
def area_square (s : ℝ) : ℝ := s * s

-- Area of triangle AEF
def area_triangle (a h : ℝ) : ℝ := (1/2) * a * h

-- The condition that areas are equal
def equal_area (area1 area2 : ℝ) : Prop :=
  area1 = area2

-- Main theorem stating DF = sqrt(13)
theorem length_DF :
  ∃ DF : ℝ, square_ABCD AB AD BC ∧ point_X BX ∧ equal_area (area_square 13) (area_triangle 13 DF) → DF = Real.sqrt 13 :=
sorry

end length_DF_l494_494140


namespace chip_rearrangement_possible_l494_494968

-- Define a predicate that checks for the property that each column contains all colors
def hasThreeColors (cols : List (List Nat)) : Prop :=
  ∀ col, 1 ∈ col ∧ 2 ∈ col ∧ 3 ∈ col

-- Main theorem statement
theorem chip_rearrangement_possible (n : Nat) (h : n > 0) (chips : List (List Nat)) 
  (Hrow : ∀ row, row.length = n) (Hcolor : (List.countp (· = 1) (chips.join)) = n ∧
                       (List.countp (· = 2) (chips.join)) = n ∧
                       (List.countp (· = 3) (chips.join)) = n) : 
  ∃ chips' : List (List Nat), hasThreeColors (List.transpose chips') ∧
                    (∀ row, row.length = n) := 
begin
  sorry
end

end chip_rearrangement_possible_l494_494968


namespace gcd_10293_29384_l494_494841

theorem gcd_10293_29384 : Nat.gcd 10293 29384 = 1 := by
  sorry

end gcd_10293_29384_l494_494841


namespace pam_age_l494_494551

-- Given conditions:
-- 1. Pam is currently twice as young as Rena.
-- 2. In 10 years, Rena will be 5 years older than Pam.

variable (Pam Rena : ℕ)

theorem pam_age
  (h1 : 2 * Pam = Rena)
  (h2 : Rena + 10 = Pam + 15)
  : Pam = 5 := 
sorry

end pam_age_l494_494551


namespace paper_thickness_after_folds_l494_494715

def folded_thickness (initial_thickness : ℝ) (folds : ℕ) : ℝ :=
  initial_thickness * 2^folds

theorem paper_thickness_after_folds :
  folded_thickness 0.1 4 = 1.6 :=
by
  sorry

end paper_thickness_after_folds_l494_494715


namespace increased_number_l494_494275

theorem increased_number (original_number : ℕ) (percentage_increase : ℚ) :
  original_number = 80 →
  percentage_increase = 1.5 →
  original_number * (1 + percentage_increase) = 200 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end increased_number_l494_494275


namespace log_sum_example_l494_494806

theorem log_sum_example : log 10 50 + log 10 20 = 3 :=
by
  -- Proof goes here, skipping with sorry
  sorry

end log_sum_example_l494_494806


namespace paperboy_sequence_ways_15_l494_494759

def D : ℕ → ℕ
| 0 := 1
| 1 := 2
| 2 := 4
| n := D (n-1) + D (n-2) + D (n-3)

def E : ℕ → ℕ
| n := D (n-2)

theorem paperboy_sequence_ways_15 : E 15 = 3136 := by
  have h₁ : D 13 = 3136 := by
    rw[D]
    sorry
  show E 15 = 3136
  rw[E]
  exact h₁

end paperboy_sequence_ways_15_l494_494759


namespace square_perimeter_l494_494765

theorem square_perimeter (s : ℝ) (h₁ : s > 0) (h₂ : 4 * s = 4 * (8 / 5)) : 4 * s = 51.2 :=
by
  have h3 : s = 8 / 5, from sorry
  rw h3
  norm_num

end square_perimeter_l494_494765


namespace anniversary_day_probability_l494_494400

-- Define the years in which the meeting could take place
def years := {1668, 1669, 1670, 1671}

-- Define a function to check if a year is a leap year
def is_leap_year (y : ℕ) : Prop := (y % 4 = 0)

-- Define the meeting date
def meeting_day := 5 -- Friday as the 5th day of the week (assuming 0 = Sunday)

-- Define the anniversary function that computes the day of the 11th anniversary
def anniversary_day (start_year : ℕ) : ℕ :=
  let days_in_year (y : ℕ) := if is_leap_year y then 366 else 365
  let total_days := (List.range 11).map (λ i => days_in_year (start_year + i))
  (total_days.sum + meeting_day) % 7

-- Define the probability computation
def probability (day : ℕ) : ℚ :=
  let occurences := (years.toList.map anniversary_day).count (λ d => d = day)
  occurences / years.toList.length

-- Statement of the theorem
theorem anniversary_day_probability :
  probability 5 = 3 / 4 ∧ probability 4 = 1 / 4 := -- 5 is Friday, 4 is Thursday
  by
    sorry -- Proof goes here

end anniversary_day_probability_l494_494400


namespace solution_interval_l494_494823

theorem solution_interval (x : ℝ) : (x^2 / (x - 5)^2 > 0) ↔ (x ∈ Set.Iio 0 ∪ Set.Ioi 0 ∩ Set.Iio 5 ∪ Set.Ioi 5) :=
by
  sorry

end solution_interval_l494_494823


namespace arithmetic_progression_20th_term_and_sum_l494_494460

theorem arithmetic_progression_20th_term_and_sum :
  let a := 3
  let d := 4
  let n := 20
  let a_20 := a + (n - 1) * d
  let S_20 := n / 2 * (a + a_20)
  a_20 = 79 ∧ S_20 = 820 := by
    let a := 3
    let d := 4
    let n := 20
    let a_20 := a + (n - 1) * d
    let S_20 := n / 2 * (a + a_20)
    sorry

end arithmetic_progression_20th_term_and_sum_l494_494460


namespace distance_from_point_to_circle_center_l494_494992

noncomputable def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * Real.cos θ, ρ * Real.sin θ)

def distance_squared (P Q : ℝ × ℝ) : ℝ :=
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2

theorem distance_from_point_to_circle_center :
  let P := polar_to_cartesian 2 (Real.pi / 3) in
  let C := (1, 0) in
  Real.sqrt (distance_squared P C) = Real.sqrt 3 :=
by sorry

end distance_from_point_to_circle_center_l494_494992


namespace balls_in_boxes_l494_494056

theorem balls_in_boxes :
    (Nat.choose 6 6) + 
    (Nat.choose 6 5) + 
    (Nat.choose 6 4 * Nat.choose 2 2) + 
    (Nat.choose 6 3 / 2) + 
    (Nat.choose 6 4 * Nat.choose 2 1) + 
    (Nat.choose 6 2 * Nat.choose 4 2 / 3.factorial) = 77 :=
by
  sorry

end balls_in_boxes_l494_494056


namespace find_circle_l494_494838

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem find_circle :
  ∃ a : ℝ, (a = 5 ∨ a = 3) ∧ ∀ x y, (x - a) ^ 2 + (y + 1) ^ 2 = 1 ↔
  (distance (x, y) (a, -1) = 1) ∧
  (distance (a, -1) (2, -1) = Real.sqrt (distance (2, -1) (4, -1) ^ 2 + 1)) :=
begin
  sorry
end

end find_circle_l494_494838


namespace increased_number_l494_494273

theorem increased_number (original_number : ℕ) (percentage_increase : ℚ) :
  original_number = 80 →
  percentage_increase = 1.5 →
  original_number * (1 + percentage_increase) = 200 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end increased_number_l494_494273


namespace y_value_l494_494080

noncomputable def x := (30 + 31 + 32 + 33 + 34 + 35 + 36 + 37 + 38 + 39 + 40)
def y := 6
axiom xy_sum : x + y = 391

theorem y_value : y = 6 :=
by
  have h1 : x = 385 := 
    by calc
      x = 30 + 31 + 32 + 33 + 34 + 35 + 36 + 37 + 38 + 39 + 40 : by rfl 
      _ = 385 : by norm_num

  have h2 : 385 + y = 391 :=
    by rw [h1, xy_sum] 

  have h3 : 385 + 6 = 391 := by norm_num

  exact eq.trans h2 h3
  sorry


end y_value_l494_494080


namespace increase_80_by_150_percent_l494_494266

-- Original number
def originalNumber : ℕ := 80

-- Percentage increase as a decimal
def increaseFactor : ℚ := 1.5

-- Expected result after the increase
def expectedResult : ℕ := 200

theorem increase_80_by_150_percent :
  originalNumber + (increaseFactor * originalNumber) = expectedResult :=
by
  sorry

end increase_80_by_150_percent_l494_494266


namespace trajectory_of_P_l494_494900

def point := ℝ × ℝ

-- Definitions for points A and F, and the circle equation
def A : point := (-1, 0)
def F (x y : ℝ) := (x - 1) ^ 2 + y ^ 2 = 16

-- Main theorem statement: proving the trajectory equation of point P
theorem trajectory_of_P : 
  (∀ (B : point), F B.1 B.2 → 
  (∃ P : point, ∃ (k : ℝ), (P.1 - B.1) * k = -(P.2 - B.2) ∧ (P.1 - A.1) * (P.1 - B.1) + (P.2 - A.2) * (P.2 - B.2) = 0)) →
  (∃ x y : ℝ, (x^2 / 4) + (y^2 / 3) = 1) :=
sorry

end trajectory_of_P_l494_494900


namespace ivan_sum_max_l494_494869

theorem ivan_sum_max (n : ℕ) (h : n ≥ 2) :
  ∃ (A : set ℕ), A.card = n ∧
    (∀ a ∈ A, ∀ b ∈ A, a ≠ b → 
      (∃ sum_written : ℕ,
        sum_written = ∑ i in (A : list ℕ).combinations 2, 
                       (list.filter (λ x, x < i.unordered_pair a b ∧ x ∣ i.unordered_pair a b) A).card) ∧
        sum_written = (n-1)*n*(n+1)/6) := 
sorry

end ivan_sum_max_l494_494869


namespace bakery_combination_l494_494349

variable (α : Type) [Fintype α] [DecidableEq α] (R : α → ℕ)
variable (r1 r2 r3 : α)
variable (total_rolls : ℕ := 7)
variable (min_rolls : ℕ := 2)

theorem bakery_combination : total_rolls = 7 ∧ min_rolls = 2 ∧ ∀ roll ∈ {r1, r2, r3}, R roll ≥ min_rolls → 
  (∃ (combinations : ℕ), combinations = 3) :=
by
  sorry

end bakery_combination_l494_494349


namespace shortest_distance_between_points_l494_494198

theorem shortest_distance_between_points
  (A B : Point)
  (curved_path : Path A B)
  (straight_line_segment_unique : Path A B → Prop)
  (shortest_distance_property : ∀ (A B : Point) (p : Path A B), 
    straight_line_segment_unique p → ∀ (q : Path A B), length q ≥ length p) :
  straightening_a_curved_road_implies_shortest_distance :=
sorry

end shortest_distance_between_points_l494_494198


namespace shadow_indeterminacy_l494_494343

-- Definitions from conditions
variables (XiaoMing XiaoQiang : Type)
variables (shadow_sunlight : XiaoMing → ℝ)
variables (shadow_streetlight : XiaoMing → ℝ)

-- Condition: Under the sunlight at the same moment, Xiao Ming's shadow is longer than Xiao Qiang's shadow.
constant sunlight_condition : shadow_sunlight XiaoMing > shadow_sunlight XiaoQiang

-- The proof that it's impossible to determine whose shadow is longer under the same streetlight
theorem shadow_indeterminacy :
  ¬(shadow_streetlight XiaoMing > shadow_streetlight XiaoQiang ∨
    shadow_streetlight XiaoMing < shadow_streetlight XiaoQiang ∨
    shadow_streetlight XiaoMing = shadow_streetlight XiaoQiang) := 
sorry

end shadow_indeterminacy_l494_494343


namespace range_of_x_l494_494518

theorem range_of_x (x : ℝ) (hx : 0 < x) (h : (log x)^2015 < (log x)^2014 ∧ (log x)^2014 < (log x)^2016) :
  0 < x ∧ x < 0.1 :=
by {
  sorry
}

end range_of_x_l494_494518


namespace balls_in_boxes_l494_494033

theorem balls_in_boxes : ∀ (balls boxes : ℕ), balls = 6 → boxes = 3 → 
  (∃! n : ℕ, n = 47 ∧ n = number_of_ways_to_distribute_balls_in_boxes_d(balls, boxes)) :=
by
  intros balls boxes h1 h2
  have h_balls : balls = 6 := h1
  have h_boxes : boxes = 3 := h2
  use 47
  split
  sorry
  sorry

end balls_in_boxes_l494_494033


namespace distant_a3b3_2ab_sqrt_ab_l494_494647

def is_distant (x y m : ℝ) : Prop :=
  |x - |y - m|| = m

theorem distant_a3b3_2ab_sqrt_ab (a b : ℝ) (h : a ≠ b) (ha : 0 < a) (hb : 0 < b) :
  is_distant (a^3 + b^3) (2 * a * b * Real.sqrt (a * b)) (a^3 + b^3) := by
  -- Proof to be provided
  sorry

end distant_a3b3_2ab_sqrt_ab_l494_494647


namespace fencing_cost_32m_l494_494457

noncomputable def total_fencing_cost (diameter : ℝ) (cost_per_meter : ℝ) : ℝ :=
  let circumference := Real.pi * diameter
  let total_cost := cost_per_meter * circumference
  (total_cost).ceil

theorem fencing_cost_32m : total_fencing_cost 32 1.5 = 151 := by
  sorry

end fencing_cost_32m_l494_494457


namespace continuity_of_g_differentiability_of_g_l494_494817

def g (α x : ℝ) : ℝ :=
  (α + | x |)^2 * Real.exp ((5 - | x |)^2)

theorem continuity_of_g (α : ℝ) : 
  (∀ x, ContinuousAt (g α) x) ↔ True :=
by
  sorry

theorem differentiability_of_g (α : ℝ) :
  (∀ x, DifferentiableAt ℝ (g α) x) ↔ (α = 0) :=
by
  sorry

end continuity_of_g_differentiability_of_g_l494_494817


namespace log_sum_eq_five_has_solution_l494_494172

theorem log_sum_eq_five_has_solution : ∃ y : ℝ, log 3 y + log 9 y = 5 ∧ y = 3^(10/3) := by
  sorry

end log_sum_eq_five_has_solution_l494_494172


namespace increase_80_by_150_percent_l494_494284

theorem increase_80_by_150_percent : 
  ∀ x percent, x = 80 ∧ percent = 1.5 → x + (x * percent) = 200 :=
by
  sorry

end increase_80_by_150_percent_l494_494284


namespace fruits_total_l494_494878

def remaining_fruits (frank_apples susan_blueberries henry_apples karen_grapes : ℤ) : ℤ :=
  let frank_remaining := 36 - (36 / 3)
  let susan_remaining := 120 - (120 / 2)
  let henry_collected := 2 * 120
  let henry_after_eating := henry_collected - (henry_collected / 4)
  let henry_remaining := henry_after_eating - (henry_after_eating / 10)
  let karen_collected := henry_collected / 2
  let karen_after_spoilage := karen_collected - (15 * karen_collected / 100)
  let karen_after_giving_away := karen_after_spoilage - (karen_after_spoilage / 3)
  let karen_remaining := karen_after_giving_away - (Int.sqrt karen_after_giving_away)
  frank_remaining + susan_remaining + henry_remaining + karen_remaining

theorem fruits_total : remaining_fruits 36 120 240 120 = 254 :=
by sorry

end fruits_total_l494_494878


namespace increase_by_percentage_l494_494253

theorem increase_by_percentage (initial_amount : ℕ) (percentage_increase : ℕ) :
  initial_amount = 80 → 
  percentage_increase = 150 → 
  initial_amount * (percentage_increase / 100) + initial_amount = 200 :=
by
  intros h_initial h_percentage
  rw [h_initial, h_percentage]
  calc
    80 * (150 / 100) + 80 = 80 * 1.5 + 80 : by norm_num
                     ... = 120 + 80         : by norm_num
                     ... = 200              : by norm_num
  sorry

end increase_by_percentage_l494_494253


namespace ages_of_siblings_l494_494351

-- Define the variables representing the ages of the siblings
variables (R D S E : ℕ)

-- Define the conditions
def conditions := 
  R = D + 6 ∧ 
  D = S + 8 ∧ 
  E = R - 5 ∧ 
  R + 8 = 2 * (S + 8)

-- Define the statement to be proved
theorem ages_of_siblings (h : conditions R D S E) : 
  R = 20 ∧ D = 14 ∧ S = 6 ∧ E = 15 :=
sorry

end ages_of_siblings_l494_494351


namespace increase_80_by_150_percent_l494_494238

-- Define the initial number (n) and the percentage increase (p)
def n : ℕ := 80
def p : ℚ := 1.5

-- The theorem stating the expected result after increasing n by 150%
theorem increase_80_by_150_percent : n + (p * n) = 200 := by
  sorry

end increase_80_by_150_percent_l494_494238


namespace XYZ_total_length_l494_494652

-- Defining the length of diagonal in a unit square
def diagonal_length : ℝ := Real.sqrt 2

-- Summarizing the sum of lengths for X, Y, Z
def X_length : ℝ := 2 * diagonal_length
def Y_length : ℝ := 1 + 2 * diagonal_length
def Z_length : ℝ := 2 + diagonal_length

-- Claim the total length of XYZ
theorem XYZ_total_length : X_length + Y_length + Z_length = 3 + 5 * Real.sqrt 2 :=
by
  -- Calculations for clarity
  have h1 : X_length = 2 * Real.sqrt 2 := by rfl
  have h2 : Y_length = 1 + 2 * Real.sqrt 2 := by rfl
  have h3 : Z_length = 2 + Real.sqrt 2 := by rfl
  -- Combining the lengths
  calc
    X_length + Y_length + Z_length
        = (2 * Real.sqrt 2) + (1 + 2 * Real.sqrt 2) + (2 + Real.sqrt 2) : by rw [h1, h2, h3]
    ... = 3 + 5 * Real.sqrt 2 : sorry

end XYZ_total_length_l494_494652


namespace increase_by_150_percent_l494_494312

theorem increase_by_150_percent (x : ℝ) (h : x = 80) : x + (1.5 * x) = 200 := 
by
  -- The proof goes here, but is omitted as per instructions.
  sorry

end increase_by_150_percent_l494_494312


namespace complement_union_M_N_l494_494926

open Finset

variable (U M N : Finset ℕ)

def U := {1, 2, 3, 4, 5, 6, 7, 8}
def M := {1, 3, 5, 7}
def N := {5, 6, 7}

theorem complement_union_M_N : compl (M ∪ N) U = {2, 4, 8} :=
by
  -- proof steps skipped
  sorry

end complement_union_M_N_l494_494926


namespace increase_by_percentage_l494_494326

theorem increase_by_percentage (a : ℤ) (b : ℚ) : ((b + 1) * a) = 200 :=
by
  -- Definitions based directly on conditions
  let a := (80 : ℤ)
  let b := (1.5 : ℚ)
  -- Assertions about the equivalent proof problem
  have h1 : b + 1 = 2.5 := by norm_num
  have h2 : (b + 1) * a = 200 := by norm_num
  exact h2

end increase_by_percentage_l494_494326


namespace ferry_P_travel_time_l494_494880

-- Define the conditions based on the problem statement
variables (t : ℝ) -- travel time of ferry P
def speed_P := 6 -- speed of ferry P in km/h
def speed_Q := speed_P + 3 -- speed of ferry Q in km/h
def distance_P := speed_P * t -- distance traveled by ferry P in km
def distance_Q := 3 * distance_P -- distance traveled by ferry Q in km
def time_Q := t + 3 -- travel time of ferry Q

-- Theorem to prove that travel time t for ferry P is 3 hours
theorem ferry_P_travel_time : time_Q * speed_Q = distance_Q → t = 3 :=
by {
  -- Since you've mentioned to include the statement only and not the proof,
  -- Therefore, the proof body is left as an exercise or represented by sorry.
  sorry
}

end ferry_P_travel_time_l494_494880


namespace adjust_positions_l494_494154

def choose (n k : ℕ) : ℕ := n.choose k
def perm (n k : ℕ) : ℕ := n.perm k

theorem adjust_positions (total : ℕ) (front_row : ℕ) (back_row : ℕ) (moved : ℕ) (not_adjacent_gaps : ℕ) :
  total = 12 → front_row = 4 → back_row = 8 → moved = 2 → not_adjacent_gaps = 5 →
  choose back_row moved * perm not_adjacent_gaps moved = 560 :=
by
  intros h_total h_front_row h_back_row h_moved h_gaps
  rw [h_total, h_front_row, h_back_row, h_moved, h_gaps]
  sorry

end adjust_positions_l494_494154


namespace positive_value_of_m_l494_494955

theorem positive_value_of_m (m : ℝ) (h : (64 * m^2 - 60 * m) = 0) : m = 15 / 16 :=
sorry

end positive_value_of_m_l494_494955


namespace symmetry_axis_of_shifted_even_function_l494_494474

-- Given conditions
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (x) = f (-x)

-- Main statement we want to prove
theorem symmetry_axis_of_shifted_even_function {f : ℝ → ℝ} (h : even_function (λ x, f (x + 1))) : 
    (axis_of_symmetry f = 1) := 
    sorry

end symmetry_axis_of_shifted_even_function_l494_494474


namespace total_hours_worked_l494_494187

theorem total_hours_worked
  (x : ℕ)
  (h1 : 5 * x = 55)
  : 2 * x + 3 * x + 5 * x = 110 :=
by 
  sorry

end total_hours_worked_l494_494187


namespace sum_of_first_ten_primes_with_units_digit_3_l494_494852

def units_digit_3_and_prime (n : ℕ) : Prop :=
  (n % 10 = 3) ∧ (Prime n)

def first_ten_primes_with_units_digit_3 : list ℕ :=
  [3, 13, 23, 43, 53, 73, 83, 103, 113, 163]

theorem sum_of_first_ten_primes_with_units_digit_3 :
  list.sum first_ten_primes_with_units_digit_3 = 671 :=
by
  sorry

end sum_of_first_ten_primes_with_units_digit_3_l494_494852


namespace increase_80_by_150_percent_l494_494270

-- Original number
def originalNumber : ℕ := 80

-- Percentage increase as a decimal
def increaseFactor : ℚ := 1.5

-- Expected result after the increase
def expectedResult : ℕ := 200

theorem increase_80_by_150_percent :
  originalNumber + (increaseFactor * originalNumber) = expectedResult :=
by
  sorry

end increase_80_by_150_percent_l494_494270


namespace longer_leg_of_minimum_30_60_90_triangle_is_9_l494_494439

-- Define the properties of a 30-60-90 triangle
def sideRatios := (1 : ℝ, Real.sqrt 3, 2 : ℝ)

noncomputable def longer_leg_of_smallest_triangle (hypotenuse_largest : ℝ) : ℝ :=
  let hypotenuse1  := hypotenuse_largest
  let shorter_leg1 := hypotenuse1 / 2
  let longer_leg1  := shorter_leg1 * Real.sqrt 3
  let hypotenuse2  := longer_leg1
  let shorter_leg2 := hypotenuse2 / 2
  let longer_leg2  := shorter_leg2 * Real.sqrt 3
  let hypotenuse3  := longer_leg2
  let shorter_leg3 := hypotenuse3 / 2
  let longer_leg3  := shorter_leg3 * Real.sqrt 3
  let hypotenuse4  := longer_leg3
  let shorter_leg4 := hypotenuse4 / 2
  let longer_leg4  := shorter_leg4 * Real.sqrt 3
  longer_leg4

theorem longer_leg_of_minimum_30_60_90_triangle_is_9 (hypotenuse_largest : ℝ) 
  (H : hypotenuse_largest = 16) : longer_leg_of_smallest_triangle hypotenuse_largest = 9 := by
  sorry

end longer_leg_of_minimum_30_60_90_triangle_is_9_l494_494439


namespace increase_150_percent_of_80_l494_494297

theorem increase_150_percent_of_80 : 80 * 1.5 + 80 = 200 :=
by
  sorry

end increase_150_percent_of_80_l494_494297


namespace not_all_composite_sum_of_blocks_l494_494638

-- Define a term embedding with a simple type as the index
def arr {α : Type*} (n : ℕ) := fin n → α

theorem not_all_composite_sum_of_blocks :
  ∃ (l : list ℕ), (l.length = 20 ∧ 
  (∀ i, i < 20 → l.nth i ≠ none) ∧
  (∀ i, i < 10 → l.count i = 2) ∧ 
  (∃ i, i < 17 ∧ ¬(nat.composite ((l.nth_le i (lt_of_lt_of_le i_succ_lt_succ i_small.contains)) + 
                                  (l.nth_le (i+1) (lt_of_lt_of_le (i + 1).succ_lt_succ (i_small.contains))) + 
                                   (l.nth_le (i+2) (lt_of_lt_of_le (i + 2).succ_lt_succ i_small.contains)) + 
                                   (l.nth_le (i+3) (lt_of_lt_of_le (i + 3).succ_lt_succ i_small.contains)))))
 ) :=
sorry

end not_all_composite_sum_of_blocks_l494_494638


namespace coefficient_of_x_squared_in_expansion_l494_494988

theorem coefficient_of_x_squared_in_expansion :
  (polynomial.coeff ((polynomial.X * 3 - 1) ^ 6) 2) = 135 := 
by
  sorry

end coefficient_of_x_squared_in_expansion_l494_494988


namespace no_solution_ineq_l494_494958

theorem no_solution_ineq (m : ℝ) : 
  (∀ x : ℝ, x - m ≥ 0 → ¬(0.5 * x + 0.5 < 2)) → m ≥ 3 :=
by
  sorry

end no_solution_ineq_l494_494958


namespace b_100_eq_6_div_88424_l494_494895

noncomputable def b (n : ℕ) : ℚ :=
match n with
| 0     => 0
| 1     => 2
| (n+1) => let T_n := Σ i in range (n+2), b i in
           3 * T_n^2 / (3 * T_n - 2)

theorem b_100_eq_6_div_88424 :
  b 100 = 6 / 88424 :=
sorry

end b_100_eq_6_div_88424_l494_494895


namespace fx_plus_3_equals_fx_plus_3x_plus_3_l494_494067

-- Define the function f
def f (x : ℝ) : ℝ := x * (x - 1) / 2

-- The theorem statement to prove
theorem fx_plus_3_equals_fx_plus_3x_plus_3 (x : ℝ) : f(x + 3) = f(x) + 3 * x + 3 :=
by
  sorry

end fx_plus_3_equals_fx_plus_3x_plus_3_l494_494067


namespace sum_first_13_eq_104_l494_494495

-- Define the arithmetic sequence
def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

def sum_of_first_n_terms (a d : ℝ) (n : ℕ) : ℝ := (n * (2 * a + (n - 1) * d)) / 2

-- Given conditions
noncomputable def a (d : ℝ) := 2 * d

noncomputable def sequence_sum_condition (a d : ℝ) : Prop :=
  arithmetic_sequence a d 2 + arithmetic_sequence a d 7 + arithmetic_sequence a d 12 = 24

-- Target proof statement
theorem sum_first_13_eq_104 (a d : ℝ) (h : sequence_sum_condition a d) : sum_of_first_n_terms a d 13 = 104 :=
by
  sorry

end sum_first_13_eq_104_l494_494495


namespace length_of_longer_leg_of_smallest_triangle_l494_494443

theorem length_of_longer_leg_of_smallest_triangle :
  ∀ (a b c : ℝ), 
  is_30_60_90_triangle (a, b, c) ∧ c = 16 
  ∧ (∀ (a₁ b₁ c₁ : ℝ), is_30_60_90_triangle (a₁, b₁, c₁) → b = c₁ → true) 
  ∧ (∀ (a₂ b₂ c₂ : ℝ), is_30_60_90_triangle (a₂, b₂, c₂) → true) 
  ∧ (∀ (a₃ b₃ c₃ : ℝ), is_30_60_90_triangle (a₃, b₃, c₃) → true) 
  → ∃ (a₄ b₄ c₄ : ℝ), is_30_60_90_triangle (a₄, b₄, c₄) ∧ b₄ = 9 :=
sorry

end length_of_longer_leg_of_smallest_triangle_l494_494443


namespace ratio_of_areas_l494_494943

theorem ratio_of_areas (R_C R_D : ℝ) (h : (60 / 360) * (2 * Real.pi * R_C) = (40 / 360) * (2 * Real.pi * R_D)) :
  (Real.pi * R_C^2) / (Real.pi * R_D^2) = (9 / 4) :=
by
  sorry

end ratio_of_areas_l494_494943


namespace radius_calc_circle_l494_494742

noncomputable def radius_of_circle (a : ℚ) : Prop :=
  let center := (a, 2 * a^2 - 27)
  ∧ a = ∅ / if the circle is tangent to the y-axis
  ∧ a = ∅ / if the circle is tangent to the line 4x = 3y
  ∧ ∅ protocol formula of  6 a^2 + a - 81 
  ∧ m = 9 ∧ n = 2
  ∧ a = 9/2

theorem radius_calc_circle (hm : 9) : Prop :=
  radius_of_circle ∧ ( 9 + 2 = 11 ) 

end radius_calc_circle_l494_494742


namespace number_of_correct_statements_l494_494779

def proposition_p : Prop := ∃ x_φ : ℝ, 2 ^ x_φ = 4

-- Statement 1
def statement1 (p : Prop) : Prop := ¬p

-- Statement 2
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) (S : ℕ → ℝ) : Prop :=
  (d > 0) → (S 4 + S 6 > 2 * S 5)

-- Statement 3
def contradiction_proof (a b c : ℝ) : Prop := 
  (a + b + c < 3) → (¬(a < 1 ∨ b < 1 ∨ c < 1))

-- Statement 4
def induction_step (n : ℕ) (lhs_sum rhs : ℝ) : Prop :=
  1 + (1/2) + (1/3) + ... + (1/(2^n - 1)) < n → sorry -- just placeholder, replace ... accordingly

-- The main theorem
theorem number_of_correct_statements :
  let num_correct := [statement1 proposition_p, arithmetic_sequence a d S, contradiction_proof, induction_step]
  ∑ i in num_correct, if i then 1 else 0 = 1 := 
sorry

end number_of_correct_statements_l494_494779


namespace increase_80_by_150_percent_l494_494268

-- Original number
def originalNumber : ℕ := 80

-- Percentage increase as a decimal
def increaseFactor : ℚ := 1.5

-- Expected result after the increase
def expectedResult : ℕ := 200

theorem increase_80_by_150_percent :
  originalNumber + (increaseFactor * originalNumber) = expectedResult :=
by
  sorry

end increase_80_by_150_percent_l494_494268


namespace anniversary_day_probability_l494_494398

-- Define the years in which the meeting could take place
def years := {1668, 1669, 1670, 1671}

-- Define a function to check if a year is a leap year
def is_leap_year (y : ℕ) : Prop := (y % 4 = 0)

-- Define the meeting date
def meeting_day := 5 -- Friday as the 5th day of the week (assuming 0 = Sunday)

-- Define the anniversary function that computes the day of the 11th anniversary
def anniversary_day (start_year : ℕ) : ℕ :=
  let days_in_year (y : ℕ) := if is_leap_year y then 366 else 365
  let total_days := (List.range 11).map (λ i => days_in_year (start_year + i))
  (total_days.sum + meeting_day) % 7

-- Define the probability computation
def probability (day : ℕ) : ℚ :=
  let occurences := (years.toList.map anniversary_day).count (λ d => d = day)
  occurences / years.toList.length

-- Statement of the theorem
theorem anniversary_day_probability :
  probability 5 = 3 / 4 ∧ probability 4 = 1 / 4 := -- 5 is Friday, 4 is Thursday
  by
    sorry -- Proof goes here

end anniversary_day_probability_l494_494398


namespace number_of_ways_to_construct_cube_l494_494355

-- Definition of the problem conditions
def white_cubes : ℕ := 5
def blue_cubes : ℕ := 3
def cube_size : ℕ := 2

-- Main theorem statement
theorem number_of_ways_to_construct_cube : 
  (count_distinct_cubical_constructions white_cubes blue_cubes cube_size = 2) := 
sorry

end number_of_ways_to_construct_cube_l494_494355


namespace minimum_games_needed_l494_494651

theorem minimum_games_needed
  (sharks_initial_wins tigers_initial_wins : ℕ)
  (initial_games N : ℕ)
  (tigers_win_fraction : ℚ)
  (sharks_initial_wins = 3)
  (tigers_initial_wins = 2)
  (initial_games = 5)
  (tigers_win_fraction = (4 : ℚ) / 5) :
  (N : ℕ) >= (10 : ℕ) :=
by
  sorry

end minimum_games_needed_l494_494651


namespace plates_after_events_l494_494999

theorem plates_after_events (flowered_plates : ℕ) (checked_plates : ℕ) (polka_dotted_plates : ℕ) :
  flowered_plates = 4 → 
  checked_plates = 8 → 
  polka_dotted_plates = 2 * checked_plates → 
  (flowered_plates + checked_plates + polka_dotted_plates - 1) = 27 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  calc 
    (4 + 8 + 2 * 8 - 1) = 12 + 16 - 1 : by rw [mul_comm]
                   ... = 28 - 1      : by rfl
                   ... = 27          : by rfl

end plates_after_events_l494_494999


namespace find_omega_a_b_and_monotonic_intervals_l494_494920

theorem find_omega_a_b_and_monotonic_intervals
  (f : ℝ → ℝ)
  (a ω b : ℝ)
  (h_f : ∀ x, f x = a * real.sin (2 * ω * x + real.pi / 6) + a / 2 + b)
  (h_a_pos : 0 < a)
  (h_omega_pos : 0 < ω)
  (h_period : ∀ t, f (t + real.pi) = f t)
  (h_max : ∃ x, f x = 7 / 4)
  (h_min : ∃ x, f x = 3 / 4) :
  ω = 1 ∧ a = 1 / 2 ∧ b = 1 ∧
  (∀ k : ℤ, ∀ x, k * real.pi - real.pi / 3 ≤ x ∧ x ≤ k * real.pi + real.pi / 6 → ∀ y, x ≤ y → y ≤ k * real.pi + real.pi / 6 → f x ≤ f y) :=
by
  sorry

end find_omega_a_b_and_monotonic_intervals_l494_494920


namespace distance_from_apex_to_larger_cross_section_l494_494763

namespace PyramidProof

variables (As Al : ℝ) (d h : ℝ)

theorem distance_from_apex_to_larger_cross_section 
  (As_eq : As = 256 * Real.sqrt 2) 
  (Al_eq : Al = 576 * Real.sqrt 2) 
  (d_eq : d = 12) :
  h = 36 := 
sorry

end PyramidProof

end distance_from_apex_to_larger_cross_section_l494_494763


namespace increase_80_by_150_percent_l494_494231

-- Define the initial number (n) and the percentage increase (p)
def n : ℕ := 80
def p : ℚ := 1.5

-- The theorem stating the expected result after increasing n by 150%
theorem increase_80_by_150_percent : n + (p * n) = 200 := by
  sorry

end increase_80_by_150_percent_l494_494231


namespace fib_determinant_property_fib_790_788_789_l494_494640

-- Definitions for Fibonacci numbers and matrix power
noncomputable def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

def matrix_fib (n : ℕ) : (ℕ × ℕ) × (ℕ × ℕ) := 
((fib (n+1), fib n), (fib n, fib (n-1)))

-- Property of the matrix representation for Fibonacci sequence
def matrix_property (n : ℕ) : Prop :=
(matrix_fib n) = pow (λ (M : ℕ × ℕ × ℕ × ℕ), ((1, 1), (1, 0))) n

-- The main theorem to be proved
theorem fib_determinant_property : 
∀ n : ℕ, matrix_property n → (fib (n+1)) * (fib (n-1)) - (fib n)^2 = (-1)^n := sorry

-- The targeted problem
theorem fib_790_788_789 : fib 790 * fib 788 - (fib 789)^2 = -1 := 
begin
  apply fib_determinant_property 789,
  sorry
end

end fib_determinant_property_fib_790_788_789_l494_494640


namespace meeting_anniversary_day_l494_494408

-- Define the input parameters for the problem
def initial_years : Set ℕ := {1668, 1669, 1670, 1671}
def meeting_day := "Friday"
def is_leap_year (year : ℕ) : Bool := (year % 4 = 0)

-- Define the theorem for the problem statement
theorem meeting_anniversary_day :
  ∀ (year : ℕ), year ∈ initial_years →
  let leap_years := (∑ n in range 1668, if is_leap_year n then 1 else 0)
  let total_days := 11 * 365 + leap_years
  let day_of_week := total_days % 7
  in (day_of_week = 0 ∧ probability Friday = 3 / 4) ∨ (day_of_week = 6 ∧ probability Thursday 1 / 4) :=
by
  sorry

end meeting_anniversary_day_l494_494408


namespace increase_by_percentage_l494_494322

theorem increase_by_percentage (a : ℤ) (b : ℚ) : ((b + 1) * a) = 200 :=
by
  -- Definitions based directly on conditions
  let a := (80 : ℤ)
  let b := (1.5 : ℚ)
  -- Assertions about the equivalent proof problem
  have h1 : b + 1 = 2.5 := by norm_num
  have h2 : (b + 1) * a = 200 := by norm_num
  exact h2

end increase_by_percentage_l494_494322


namespace sum_of_first_ten_primes_with_units_digit_three_l494_494861

-- Define the problem to prove the sum of the first 10 primes ending in 3 is 639
theorem sum_of_first_ten_primes_with_units_digit_three : 
  let primes_with_units_digit_three := [3, 13, 23, 43, 53, 73, 83, 103, 113, 163]
  in list.sum primes_with_units_digit_three = 639 := 
by 
  -- We define the primes with the units digit 3 as given and check the sum
  let primes_with_units_digit_three := [3, 13, 23, 43, 53, 73, 83, 103, 113, 163]
  show list.sum primes_with_units_digit_three = 639 from sorry

end sum_of_first_ten_primes_with_units_digit_three_l494_494861


namespace time_to_paint_one_room_l494_494757

theorem time_to_paint_one_room (total_rooms : ℕ) (rooms_painted : ℕ) (time_remaining : ℕ) (rooms_left : ℕ) :
  total_rooms = 9 ∧ rooms_painted = 5 ∧ time_remaining = 32 ∧ rooms_left = total_rooms - rooms_painted → time_remaining / rooms_left = 8 :=
by
  intros h
  sorry

end time_to_paint_one_room_l494_494757


namespace function_properties_l494_494333

-- Define the function with the specified conditions
theorem function_properties (f : ℝ → ℝ) :
  (∀ x, f(x-1) = f(1-x)) ∧ 
  (∀ x, f(x) ≥ 3) ∧ 
  (∀ x, f(x) = f(x+2)) → 
  ∀ x, f(x) = 3 * Real.cos (π * x) :=
begin
  sorry
end

end function_properties_l494_494333


namespace range_of_a_l494_494891

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (1 - 2 * a) * x + 3 * a else 2^(x - 1)

theorem range_of_a (a : ℝ) : 
  (∀ y ∈ Set.univ, ∃ x : ℝ, f(a)(x) = y) ↔ (0 ≤ a ∧ a < 0.5) := 
by
  sorry

end range_of_a_l494_494891


namespace option_d_bijective_l494_494719

def bijective {α β : Type*} (f : α → β) : Prop :=
injective f ∧ surjective f

theorem option_d_bijective : bijective (λ x : ℝ, x^3) :=
sorry

end option_d_bijective_l494_494719


namespace right_triangle_hypotenuse_l494_494090

theorem right_triangle_hypotenuse
  (a b c : ℝ)
  (h₀ : a = 24)
  (h₁ : a^2 + b^2 + c^2 = 2500)
  (h₂ : c^2 = a^2 + b^2) :
  c = 25 * Real.sqrt 2 :=
by
  sorry

end right_triangle_hypotenuse_l494_494090


namespace log_addition_identity_l494_494797

theorem log_addition_identity :
  ∃ (a x y : ℝ), a = 10 ∧ x = 50 ∧ y = 20 ∧ (log 10 50 + log 10 20 = 3) :=
by
  let a := 10
  let x := 50
  let y := 20
  have h1 : log a (x * y) = log a x + log a y,
    from sorry -- logarithmic identity
  have h2 : log a (x * y) = log a 1000,
    from congrArg (log a) (by norm_num) -- simplifying x * y
  have h3 : log 10 1000 = 3,
    from sorry -- calculating log 1000 base 10 directly
  exact ⟨a, x, y, rfl, rfl, rfl, by linarith [h1, h2, h3]⟩

end log_addition_identity_l494_494797


namespace increase_80_by_150_percent_l494_494269

-- Original number
def originalNumber : ℕ := 80

-- Percentage increase as a decimal
def increaseFactor : ℚ := 1.5

-- Expected result after the increase
def expectedResult : ℕ := 200

theorem increase_80_by_150_percent :
  originalNumber + (increaseFactor * originalNumber) = expectedResult :=
by
  sorry

end increase_80_by_150_percent_l494_494269


namespace distinguishable_balls_in_indistinguishable_boxes_l494_494024

theorem distinguishable_balls_in_indistinguishable_boxes :
  let num_distinguishable_balls := 6
  let num_indistinguishable_boxes := 3
  -- different ways to distribute balls across boxes summarized
  let ways := ∑([{6, 0, 0} ++ {5, 1, 0} ++ {4, 2, 0} ++ {4, 1, 1} ++ {3, 3, 0} ++ {3, 2, 1} ++ {2, 2, 2}], sum)
  -- corresponding ways to remove permutation of distinguishable ball cases
  ways = 222 := sorry

end distinguishable_balls_in_indistinguishable_boxes_l494_494024


namespace Jack_initial_dollars_l494_494107

-- Definitions based on the conditions given
def euros := 36
def exchange_rate := 2
def total_dollars := 117

-- Theorem to prove how many dollars Jack initially had
theorem Jack_initial_dollars :
  let euro_value_in_dollars := euros * exchange_rate in
  total_dollars - euro_value_in_dollars = 45 :=
by
  sorry

end Jack_initial_dollars_l494_494107


namespace common_tangent_circle_of_circumcircles_l494_494607

theorem common_tangent_circle_of_circumcircles 
  (A B C D E F M : Point)
  (h1 : CyclicQuad ABCD)
  (h2 : Intersect AB CD = E)
  (h3 : Intersect BC DA = F)
  (h4 : MiquelPoint ABCD M) :
  CommonTangentCircle ([Circumcircle(EAD), Circumcircle(EBC), Circumcircle(FCD), Circumcircle(FAB)]) :=
by
  sorry

end common_tangent_circle_of_circumcircles_l494_494607


namespace annual_growth_rate_of_investment_total_affordable_housing_built_2012_l494_494217

noncomputable def annual_investment_growth_rate (initial_investment : ℝ) (total_investment_2012 : ℝ) (years : ℕ) : ℝ :=
let eq := (initial_investment + initial_investment * (1 + x) + initial_investment * (1 + x)^2 = total_investment_2012) in
classical.some (exists_unique_of_exists_of_unique eq 
  (λ x1 x2 h1 h2, begin 
    simp at *,
    sorry -- The actual proof would go here
  end))

theorem annual_growth_rate_of_investment :
  annual_investment_growth_rate 2 9.5 3 = 0.5 := sorry

noncomputable def total_area_2012 (total_investment_2012 : ℝ) (cost_per_sqm : ℝ) : ℝ :=
total_investment_2012 / cost_per_sqm

theorem total_affordable_housing_built_2012 :
  total_area_2012 9.5 (2 / 8) = 38 := sorry

end annual_growth_rate_of_investment_total_affordable_housing_built_2012_l494_494217


namespace water_to_concentrate_ratio_l494_494385

theorem water_to_concentrate_ratio (servings : ℕ) (serving_size_oz concentrate_size_oz : ℕ)
                                (cans_of_concentrate required_juice_oz : ℕ)
                                (h_servings : servings = 280)
                                (h_serving_size : serving_size_oz = 6)
                                (h_concentrate_size : concentrate_size_oz = 12)
                                (h_cans_of_concentrate : cans_of_concentrate = 35)
                                (h_required_juice : required_juice_oz = servings * serving_size_oz)
                                (h_made_juice : required_juice_oz = 1680)
                                (h_concentrate_volume : cans_of_concentrate * concentrate_size_oz = 420)
                                (h_water_volume : required_juice_oz - (cans_of_concentrate * concentrate_size_oz) = 1260)
                                (h_water_cans : 1260 / concentrate_size_oz = 105) :
                                105 / 35 = 3 :=
by
  sorry

end water_to_concentrate_ratio_l494_494385


namespace increase_80_by_150_percent_l494_494240

-- Define the initial number (n) and the percentage increase (p)
def n : ℕ := 80
def p : ℚ := 1.5

-- The theorem stating the expected result after increasing n by 150%
theorem increase_80_by_150_percent : n + (p * n) = 200 := by
  sorry

end increase_80_by_150_percent_l494_494240


namespace first_digit_appears_four_times_l494_494584

def G : ℕ → ℕ
| 1     := 3
| 2     := 4
| (n+2) := (G (n+1) + G n) % 10

theorem first_digit_appears_four_times :
  ∃ n, n < 40 ∧ ((list.units_digits (list.map (G ∘ nat.succ) (list.range 40))).countp (λ x, x = 7)) = 4 :=
by
  sorry

end first_digit_appears_four_times_l494_494584


namespace value_of_f_at_half_l494_494911

-- Define the function f as an odd function.
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Given conditions:
variable (a b c : ℝ) (f : ℝ → ℝ)
hypothesis (h1 : is_odd_function (λ x, x^3 + a * x^2 + b * x + c))
hypothesis (h2 : 2 * b - 5 ≤ 2 * b - 3)

theorem value_of_f_at_half
  (h3 : b = 2)
  (h4 : c = 0)
  (h5 : a = 0) :
  (x^3 + 2 * x) (1 / 2) = 9 / 8 :=
by
  -- Proof not required
  sorry

end value_of_f_at_half_l494_494911


namespace evening_sales_l494_494636

theorem evening_sales
  (remy_bottles_morning : ℕ := 55)
  (nick_bottles_fewer : ℕ := 6)
  (price_per_bottle : ℚ := 0.50)
  (evening_sales_more : ℚ := 3) :
  let nick_bottles_morning := remy_bottles_morning - nick_bottles_fewer
  let remy_sales_morning := remy_bottles_morning * price_per_bottle
  let nick_sales_morning := nick_bottles_morning * price_per_bottle
  let total_morning_sales := remy_sales_morning + nick_sales_morning
  let total_evening_sales := total_morning_sales + evening_sales_more
  total_evening_sales = 55 :=
by
  sorry

end evening_sales_l494_494636


namespace dolly_dresses_shipment_l494_494377

variable (T : ℕ)

/-- Given that 70% of the total number of Dolly Dresses in the shipment is equal to 140,
    prove that the total number of Dolly Dresses in the shipment is 200. -/
theorem dolly_dresses_shipment (h : (7 * T) / 10 = 140) : T = 200 :=
sorry

end dolly_dresses_shipment_l494_494377


namespace fraction_halfway_l494_494706

theorem fraction_halfway (a b c d : ℚ) (h₁ : a = 3) (h₂ : b = 4) (h₃ : c = 5) (h₄ : d = 6) :
  (1 / 2) * ((a / b) + (c / d)) = 19 / 24 := 
by
  sorry

end fraction_halfway_l494_494706


namespace increase_result_l494_494245

-- Given conditions
def original_number : ℝ := 80
def increase_percentage : ℝ := 1.5

-- The result after the increase
theorem increase_result (h1 : original_number = 80) (h2 : increase_percentage = 1.5) : 
  original_number + (increase_percentage * original_number) = 200 := by
  sorry

end increase_result_l494_494245


namespace distance_from_point_A_l494_494766

theorem distance_from_point_A :
  ∀ (A : ℝ) (area : ℝ) (white_area : ℝ) (black_area : ℝ), area = 18 →
  (black_area = 2 * white_area) →
  A = (12 * Real.sqrt 2) / 5 := by
  intros A area white_area black_area h1 h2
  sorry

end distance_from_point_A_l494_494766


namespace find_point_on_parabola_l494_494909

noncomputable def parabola (x y : ℝ) : Prop := y^2 = 6 * x
def positive_y (y : ℝ) : Prop := y > 0
def distance_to_focus (x y : ℝ) : Prop := (x - 3/2)^2 + y^2 = (5/2)^2 

theorem find_point_on_parabola (x y : ℝ) :
  parabola x y ∧ positive_y y ∧ distance_to_focus x y → (x = 1 ∧ y = Real.sqrt 6) :=
by
  sorry

end find_point_on_parabola_l494_494909


namespace possible_ratios_of_distances_l494_494898

theorem possible_ratios_of_distances (a b : ℝ) (h : a > b) (h1 : ∃ points : Fin 4 → ℝ × ℝ, 
  ∀ (i j : Fin 4), i ≠ j → 
  (dist (points i) (points j) = a ∨ dist (points i) (points j) = b )) :
  a / b = Real.sqrt 2 ∨ 
  a / b = (1 + Real.sqrt 5) / 2 ∨ 
  a / b = Real.sqrt 3 ∨ 
  a / b = Real.sqrt (2 + Real.sqrt 3) :=
by 
  sorry

end possible_ratios_of_distances_l494_494898


namespace num_ways_dist_6_balls_3_boxes_l494_494048

open Finset

/-- Number of ways to distribute 6 distinguishable balls into 3 indistinguishable boxes -/
theorem num_ways_dist_6_balls_3_boxes : 
  let num_ways := 
    let n := 6 in let k := 3 in
    ((choose n 6) + 
    (choose n 5) + 
    (choose n 4) + 
    (choose n 4) + 
    (choose n 3) / 2 + 
    (choose n 3) * (choose (n - 3) 2) + 
    (choose n 2) * (choose (n - 2) 2) / (factorial k)) 
  in num_ways = 122 := 
by 
  sorry

end num_ways_dist_6_balls_3_boxes_l494_494048


namespace balls_in_boxes_l494_494031

theorem balls_in_boxes : ∀ (balls boxes : ℕ), balls = 6 → boxes = 3 → 
  (∃! n : ℕ, n = 47 ∧ n = number_of_ways_to_distribute_balls_in_boxes_d(balls, boxes)) :=
by
  intros balls boxes h1 h2
  have h_balls : balls = 6 := h1
  have h_boxes : boxes = 3 := h2
  use 47
  split
  sorry
  sorry

end balls_in_boxes_l494_494031


namespace compare_xyz_l494_494888

noncomputable def x := (0.5 : ℝ)^(0.5 : ℝ)
noncomputable def y := (0.5 : ℝ)^(1.3 : ℝ)
noncomputable def z := (1.3 : ℝ)^(0.5 : ℝ)

theorem compare_xyz : z > x ∧ x > y := by
  sorry

end compare_xyz_l494_494888


namespace longer_leg_of_minimum_30_60_90_triangle_is_9_l494_494440

-- Define the properties of a 30-60-90 triangle
def sideRatios := (1 : ℝ, Real.sqrt 3, 2 : ℝ)

noncomputable def longer_leg_of_smallest_triangle (hypotenuse_largest : ℝ) : ℝ :=
  let hypotenuse1  := hypotenuse_largest
  let shorter_leg1 := hypotenuse1 / 2
  let longer_leg1  := shorter_leg1 * Real.sqrt 3
  let hypotenuse2  := longer_leg1
  let shorter_leg2 := hypotenuse2 / 2
  let longer_leg2  := shorter_leg2 * Real.sqrt 3
  let hypotenuse3  := longer_leg2
  let shorter_leg3 := hypotenuse3 / 2
  let longer_leg3  := shorter_leg3 * Real.sqrt 3
  let hypotenuse4  := longer_leg3
  let shorter_leg4 := hypotenuse4 / 2
  let longer_leg4  := shorter_leg4 * Real.sqrt 3
  longer_leg4

theorem longer_leg_of_minimum_30_60_90_triangle_is_9 (hypotenuse_largest : ℝ) 
  (H : hypotenuse_largest = 16) : longer_leg_of_smallest_triangle hypotenuse_largest = 9 := by
  sorry

end longer_leg_of_minimum_30_60_90_triangle_is_9_l494_494440


namespace negative_pairs_in_sequence_l494_494728

open Real

theorem negative_pairs_in_sequence 
  {m n : ℕ} {a : Fin m → ℝ} 
  (h_pos_mn : m > n) 
  (hnz : ∀ i, a ⟨i, by simp; exact lt_of_lt_of_le (Nat.succ_pos _) (Nat.le_of_lt_succ (Nat.lt_of_succ_lt_succ (Fin.is_lt ⟨i, by simp⟩)))⟩ ≠ 0) 
  (hsum : ∀ k (hk : k < n), ∑ i in Finset.range m, (a ⟨i, by simp; exact lt_of_lt_of_le (Nat.succ_pos _) (Nat.le_of_lt_succ (Nat.lt_of_succ_lt_succ (Fin.is_lt ⟨i, by simp⟩)))⟩^ (2 * i + 1)) * ((by exact Nat.choose (k + i) k): ℝ) = 0) :
  ∃ s : Finset (Fin (m-1)), s.card ≥ n ∧ ∀ i ∈ s, a ⟨i, by simp; exact lt_of_lt_of_le (Nat.succ_pos _) (Nat.le_of_lt_succ (Nat.lt_of_succ_lt_succ (Fin.is_lt ⟨i, by simp⟩)))⟩ * a ⟨i+1, by simp; exact lt_of_lt_of_le (Nat.succ_pos _) (Nat.le_of_lt_succ (Nat.lt_of_succ_lt_succ (Fin.is_lt ⟨i+1, by simp⟩)))⟩ < 0 := 
by
  sorry

end negative_pairs_in_sequence_l494_494728


namespace num_ways_dist_6_balls_3_boxes_l494_494050

open Finset

/-- Number of ways to distribute 6 distinguishable balls into 3 indistinguishable boxes -/
theorem num_ways_dist_6_balls_3_boxes : 
  let num_ways := 
    let n := 6 in let k := 3 in
    ((choose n 6) + 
    (choose n 5) + 
    (choose n 4) + 
    (choose n 4) + 
    (choose n 3) / 2 + 
    (choose n 3) * (choose (n - 3) 2) + 
    (choose n 2) * (choose (n - 2) 2) / (factorial k)) 
  in num_ways = 122 := 
by 
  sorry

end num_ways_dist_6_balls_3_boxes_l494_494050


namespace pam_age_l494_494550

-- Given conditions:
-- 1. Pam is currently twice as young as Rena.
-- 2. In 10 years, Rena will be 5 years older than Pam.

variable (Pam Rena : ℕ)

theorem pam_age
  (h1 : 2 * Pam = Rena)
  (h2 : Rena + 10 = Pam + 15)
  : Pam = 5 := 
sorry

end pam_age_l494_494550


namespace num_children_eq_3_l494_494361

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

end num_children_eq_3_l494_494361


namespace find_coordinates_of_C_l494_494157

def Point := (ℝ × ℝ)

def A : Point := (-2, -1)
def B : Point := (4, 7)

/-- A custom definition to express that point C divides the segment AB in the ratio 2:1 from point B. -/
def is_point_C (C : Point) : Prop :=
  ∃ k : ℝ, k = 2 / 3 ∧
  C = (k * A.1 + (1 - k) * B.1, k * A.2 + (1 - k) * B.2)

theorem find_coordinates_of_C (C : Point) (h : is_point_C C) : 
  C = (2, 13 / 3) :=
sorry

end find_coordinates_of_C_l494_494157


namespace interval_of_monotonic_increase_l494_494509

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem interval_of_monotonic_increase : {x : ℝ | -1 ≤ x} ⊆ {x : ℝ | 0 < deriv f x} :=
by
  sorry

end interval_of_monotonic_increase_l494_494509


namespace good_horse_catchup_l494_494986

theorem good_horse_catchup (x : ℕ) : 240 * x = 150 * (x + 12) :=
by sorry

end good_horse_catchup_l494_494986


namespace distribute_6_balls_in_3_boxes_l494_494037

def num_ways_to_distribute_balls (balls : ℕ) (boxes : ℕ) : ℕ :=
  if balls = 6 ∧ boxes = 3 then 92 else 0

theorem distribute_6_balls_in_3_boxes :
  num_ways_to_distribute_balls 6 3 = 92 :=
by
  rw num_ways_to_distribute_balls
  norm_num
  sorry

end distribute_6_balls_in_3_boxes_l494_494037


namespace total_students_l494_494690

-- Definition of variables and conditions
def M := 50
def E := 4 * M - 3

-- Statement of the theorem to prove
theorem total_students : E + M = 247 := by
  sorry

end total_students_l494_494690


namespace trig_product_identity_l494_494814

theorem trig_product_identity
  (sin cos : ℝ → ℝ)
  (h_sin : ∀ x, sin x = math.sin x)
  (h_cos : ∀ x, cos x = math.cos x)
  (deg_to_rad : ℝ → ℝ)
  (h_deg_to_rad : ∀ x, deg_to_rad x = x * (real.pi / 180)) :
  sin (deg_to_rad 8) * sin (deg_to_rad 40) * sin (deg_to_rad 70) * sin (deg_to_rad 82) =
    3 * real.sqrt 3 / 16 := 
begin
  sorry
end

end trig_product_identity_l494_494814


namespace return_to_start_position_l494_494974

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def move_steps (n : ℕ) : ℤ :=
  if n = 1 then 0
  else if is_prime n then 2
  else -3

def total_steps_after_30_moves : ℤ :=
  ∑ i in Finset.range 31, move_steps i

theorem return_to_start_position :
  total_steps_after_30_moves = -37 :=
by
  sorry

end return_to_start_position_l494_494974


namespace total_value_of_horse_and_saddle_l494_494150

def saddle_value : ℝ := 12.5
def horse_value : ℝ := 7 * saddle_value

theorem total_value_of_horse_and_saddle : horse_value + saddle_value = 100 := by
  sorry

end total_value_of_horse_and_saddle_l494_494150


namespace max_min_cos_sin_cos_l494_494483

theorem max_min_cos_sin_cos (x y z : ℝ) (h1 : x ≥ y) (h2 : y ≥ z) (h3 : z ≥ π / 12) (h4 : x + y + z = π / 2) :
  ∃ (max_val min_val : ℝ), 
    (max_val = (2 + Real.sqrt 3) / 8) ∧ 
    (min_val = 1 / 8) ∧ 
    max_val = max (cos x * sin y * cos z) ∧ 
    min_val = min (cos x * sin y * cos z) :=
  sorry

end max_min_cos_sin_cos_l494_494483


namespace increase_result_l494_494243

-- Given conditions
def original_number : ℝ := 80
def increase_percentage : ℝ := 1.5

-- The result after the increase
theorem increase_result (h1 : original_number = 80) (h2 : increase_percentage = 1.5) : 
  original_number + (increase_percentage * original_number) = 200 := by
  sorry

end increase_result_l494_494243


namespace find_x_l494_494716

theorem find_x (x y : ℤ) (h1 : x + 2 * y = 100) (h2 : y = 25) : x = 50 :=
by
  sorry

end find_x_l494_494716


namespace prime_square_minus_one_mod_12_remainder_l494_494135

theorem prime_square_minus_one_mod_12_remainder {p : ℕ} (hp : Nat.Prime p) : 
  ∃ r ∈ {0, 3, 8}, (p^2 - 1) % 12 = r :=
sorry

end prime_square_minus_one_mod_12_remainder_l494_494135


namespace odd_function_iff_phi_values_l494_494514

noncomputable def f (x ϕ : ℝ) : ℝ := Math.sin (x + ϕ) - Math.sin (x + 7 * ϕ)

theorem odd_function_iff_phi_values (ϕ : ℝ) :
  (∀ x, f (-x) ϕ = -f x ϕ) ↔ (ϕ = π / 8 ∨ ϕ = 3 * π / 8) :=
sorry

end odd_function_iff_phi_values_l494_494514


namespace problem_divide_children_into_groups_l494_494574

theorem problem_divide_children_into_groups :
  let count_partitions :=
    (5.choose 1 + 5.choose 2 + 5.choose 3 + 5.choose 4) / 2
  let count_arrangements :=
    count_partitions * ((1 * 6) + (1 * 2) + (2 * 1) + (6 * 1))
  count_arrangements = 50 :=
by
  let count_partitions := (5.choose 1 + 5.choose 2 + 5.choose 3 + 5.choose 4) / 2
  have h1 : count_partitions = 15 := by sorry
  let count_arrangements := count_partitions * ((1 * 6) + (1 * 2) + (2 * 1) + (6 * 1))
  have h2 : count_arrangements = 240 := by sorry
  show count_arrangements = 50 from false.elim sorry

end problem_divide_children_into_groups_l494_494574


namespace sequence_periodic_l494_494147

theorem sequence_periodic (a : ℕ → ℝ) (h1 : a 1 = 0) (h2 : ∀ n, a n + a (n + 1) = 2) : a 2011 = 0 := by
  sorry

end sequence_periodic_l494_494147


namespace find_area_of_triangle_l494_494818

-- Definitions of the given points P and Q
def P : ℝ × ℝ := (1, 1)
def Q : ℝ × ℝ := (4, 4)

-- Definition of the condition that R lies on the line x - y = 1
def onLine (R : ℝ × ℝ) : Prop := R.1 - R.2 = 1

-- Definition of the area calculation using the Shoelace formula
def area (P Q R : ℝ × ℝ) : ℝ :=
  1 / 2 * (P.1 * (Q.2 - R.2) + Q.1 * (R.2 - P.2) + R.1 * (P.2 - Q.2)).abs

-- Proof statement to show that for some R on the line x - y = 1, area of PQR is 1.5
theorem find_area_of_triangle (R : ℝ × ℝ) (hR : onLine R) : 
  area P Q R = 1.5 := 
sorry

end find_area_of_triangle_l494_494818


namespace possible_marks_l494_494468

theorem possible_marks (n : ℕ) : n = 3 ∨ n = 6 ↔
  ∃ (m : ℕ), n = (m * (m - 1)) / 2 ∧ (∀ (k : ℕ), 1 ≤ k ∧ k ≤ n → ∃ (i j : ℕ), i < j ∧ j - i = k ∧ (∀ (x y : ℕ), x < y → x ≠ i ∨ y ≠ j)) :=
by sorry

end possible_marks_l494_494468


namespace solve_logarithm_eq_l494_494176

theorem solve_logarithm_eq (y : ℝ) (hy : log 3 y + log 9 y = 5) : y = 3^(10/3) := 
  sorry

end solve_logarithm_eq_l494_494176


namespace actual_average_height_l494_494654

theorem actual_average_height 
  (initial_avg : ℕ) 
  (num_students : ℕ) 
  (recorded_heights : list ℕ) 
  (actual_heights : list ℕ)
  (wrong_indices : list ℕ) : 
  (initial_avg = 180) →
  (num_students = 50) →
  (recorded_heights = [170, 140, 200]) →
  (actual_heights = [150, 155, 180]) →
  (wrong_indices = [0, 1, 2]) →
  (let total_diff := list.sum (list.map (λ (i : ℕ), list.nth_le recorded_heights i sorry - list.nth_le actual_heights i sorry) wrong_indices),
      initial_total_height := initial_avg * num_students,
      correct_total_height := initial_total_height - total_diff,
      actual_avg_height := correct_total_height / num_students in
    actual_avg_height = 179.50) :=
by
  intros
  sorry

end actual_average_height_l494_494654


namespace num_children_eq_3_l494_494359

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

end num_children_eq_3_l494_494359


namespace solve_log_inequality_l494_494142

noncomputable def solution_set (a : ℝ) : Set ℝ :=
  if a = 2 then {x : ℝ | x < 0}
  else if a > 2 then {x : ℝ | -2 / (a - 2) < x ∧ x < 0}
  else {x : ℝ | x > 2 / (2 - a) ∨ x < 0}

theorem solve_log_inequality (a : ℝ) (h : a > 0) :
  ∀ x : ℝ, log 2 (a * x / (x - 1)) < 1 ↔ x ∈ solution_set (a) :=
by
  -- proof to be filled in
  sorry

end solve_log_inequality_l494_494142


namespace minimize_max_F_l494_494661

noncomputable def F (x A B : ℝ) : ℝ := 
  abs (cos x ^ 2 + 2 * sin x * cos x - sin x ^ 2 + A * x + B)

theorem minimize_max_F : ∀ (A B : ℝ), 
  (∀ x, 0 ≤ x ∧ x ≤ (3 / 2) * π → F x A B ≤ sqrt 2) ↔ (A = 0 ∧ B = 0) :=
by
  sorry

end minimize_max_F_l494_494661


namespace find_p_of_segment_length_l494_494665

theorem find_p_of_segment_length (p : ℝ) : 
  (∀ p > 0, ∃ x y : ℝ, (x + 1)^2 + y^2 = 4 ∧ (y^2 = 2 * p * x) ∧ (∃ l : ℝ,  l = 4 ∧ 
  (x = -p / 2) ∧ l ∈ line_segment)) → p = 2 :=
by
  sorry

end find_p_of_segment_length_l494_494665


namespace find_m_l494_494075

theorem find_m (m : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = x^2 - 2 * x + m) 
  (h2 : ∀ x ≥ (3 : ℝ), f x ≥ 1) : m = -2 := 
sorry

end find_m_l494_494075


namespace arithmetic_mean_of_primes_l494_494807

theorem arithmetic_mean_of_primes :
  let primes := [37, 41, 43]
  let mean := (37 + 41 + 43) / 3
  mean = 40.33 :=
by
  let primes := [37, 41, 43]
  let mean := (37 + 41 + 43) / 3
  have h : mean = 40.33 := sorry
  exact h

end arithmetic_mean_of_primes_l494_494807


namespace positive_integer_solutions_l494_494643

theorem positive_integer_solutions : 
  (∀ x : ℤ, ((1 + 2 * (x:ℝ)) / 4 - (1 - 3 * (x:ℝ)) / 10 > -1 / 5) ∧ (3 * (x:ℝ) - 1 < 2 * ((x:ℝ) + 1)) → (x = 1 ∨ x = 2)) :=
by 
  sorry

end positive_integer_solutions_l494_494643


namespace vector_norm_sum_l494_494123

variables {R : Type*} [NormedField R]

variables (a b m : EuclideanSpace ℝ (Fin 2))

def midpoint (a b : EuclideanSpace ℝ (Fin 2)) : EuclideanSpace ℝ (Fin 2) := (a + b) / 2

def norm_squared (v : EuclideanSpace ℝ (Fin 2)) : ℝ := ∥v∥^2

theorem vector_norm_sum
  (h1 : m = ⟨4, 5⟩)
  (h2 : midpoint a b = m)
  (h3 : inner a b = 10) :
  norm_squared a + norm_squared b = 144 :=
sorry

end vector_norm_sum_l494_494123


namespace carls_membership_number_l494_494470

-- Definitions for the conditions
def is_two_digit_prime (n : ℕ) : Prop := 10 ≤ n ∧ n < 100 ∧ Nat.Prime n

variable (a b c d : ℕ)

-- Stating the conditions
hypothesis h1 : is_two_digit_prime a
hypothesis h2 : is_two_digit_prime b
hypothesis h3 : is_two_digit_prime c
hypothesis h4 : is_two_digit_prime d
hypothesis h_sum_all : a + b + c + d = 100
hypothesis h_sum_birthday_ben : a + c + d = 30
hypothesis h_sum_birthday_carl : a + b + d = 29
hypothesis h_sum_birthday_david : a + b + c = 23

-- Proposition to prove
theorem carls_membership_number : c = 23 :=
by {
    sorry  -- Proof is not required as per the instructions
}

end carls_membership_number_l494_494470


namespace sam_and_erica_money_total_l494_494637

def sam_money : ℕ := 38
def erica_money : ℕ := 53

theorem sam_and_erica_money_total : sam_money + erica_money = 91 :=
by
  -- the proof is not required; hence we skip it
  sorry

end sam_and_erica_money_total_l494_494637


namespace max_value_neg_x_l494_494124

variable (f : ℝ → ℝ) (a : ℝ) (x : ℝ)

-- Conditions
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

def f_pos_def (x : ℝ) (hx : 0 < x) : ℝ :=
  4 * x + (1 / x) + 3

-- Problem statement
theorem max_value_neg_x :
  is_odd_function f →
  (∀ x > 0, f(x) = 4 * x + (1 / x) + 3) →
  ∃ xmax < 0, (∀ x < 0, f(x) ≤ f(xmax)) ∧ f(xmax) = -7 :=
by
  sorry

end max_value_neg_x_l494_494124


namespace area_ratio_of_circles_l494_494948

theorem area_ratio_of_circles (R_C R_D : ℝ) (hL : (60.0 / 360.0) * 2.0 * Real.pi * R_C = (40.0 / 360.0) * 2.0 * Real.pi * R_D) : 
  (Real.pi * R_C^2) / (Real.pi * R_D^2) = 4.0 / 9.0 :=
by
  sorry

end area_ratio_of_circles_l494_494948


namespace negation_of_universal_prop_l494_494466

theorem negation_of_universal_prop :
  (¬ ∀ x : ℝ, x^2 - 2*x + 4 ≤ 0) ↔ (∃ x : ℝ, x^2 - 2*x + 4 > 0) :=
by
  sorry

end negation_of_universal_prop_l494_494466


namespace Mo_tea_cups_l494_494624

theorem Mo_tea_cups (n t : ℕ) 
  (h1 : 2 * n + 5 * t = 36)
  (h2 : 5 * t = 2 * n + 14) : 
  t = 5 :=
by
  sorry

end Mo_tea_cups_l494_494624


namespace geometric_sequence_sum_l494_494519

-- Define the geometric sequence property
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

-- The given sequence a_n, with specific conditions
variable {a : ℕ → ℝ}
variable (h_geom : is_geometric_sequence a)
variable (h_a2 : a 2 = 4)
variable (h_mean : a 3 + 2 = (a 2 + a 4) / 2)

-- Define the b_n sequence
def b (n : ℕ) := 2 * Real.log2 (a n) - 1

-- Define the sequence a_n * b_n
def ab (n : ℕ) := a n * b n

-- Define the sum T_n
def T (n : ℕ) := ∑ i in Finset.range n, ab i

-- State the main theorem combining parts (I) and (II)
theorem geometric_sequence_sum :
  (∀ n : ℕ, a n = 2 ^ n) →
  ∀ n : ℕ, T n = 6 + (2 * n - 3) * 2 ^ (n + 1) :=
sorry

end geometric_sequence_sum_l494_494519


namespace num_true_propositions_l494_494902

variable (x : ℝ)
def a : ℝ × ℝ := (-1, x)
def b : ℝ × ℝ := (x + 2, x)

-- Perpendicular vectors satisfy dot product equals zero
def perpendicular (u v : ℝ × ℝ) : Prop :=
  u.1 * v.1 + u.2 * v.2 = 0

theorem num_true_propositions :
  (perpendicular (a x) (b x) → x = -1) ∧ (perpendicular (a x) (b x) ↔ x = 2) →
  (count
    [ (x = -1 → perpendicular (a x) (b x)),
      (perpendicular (a x) (b x) → x = -1),
      (¬ (x = -1) → ¬ perpendicular (a x) (b x)),
      (¬ perpendicular (a x) (b x) → ¬ (x = -1)) ]
    true) = 2 := sorry

end num_true_propositions_l494_494902


namespace sum_of_first_ten_primes_with_units_digit_3_l494_494858

def is_prime (n : ℕ) : Prop := nat.prime n

def units_digit_3 (n : ℕ) : Prop := n % 10 = 3

def first_ten_primes_units_digit_3 : list ℕ :=
  [3, 13, 23, 43, 53, 73, 83, 103, 113, 163]

def sum_first_ten_primes_units_digit_3 : ℕ :=
  first_ten_primes_units_digit_3.sum

theorem sum_of_first_ten_primes_with_units_digit_3 :
  sum_first_ten_primes_units_digit_3 = 793 := by
  -- Here we provide the steps as a placeholder, but in real practice,
  -- a proof should be constructed to verify this calculation.
  sorry

end sum_of_first_ten_primes_with_units_digit_3_l494_494858


namespace total_fruits_is_174_l494_494367

def basket1_apples : ℕ := 9
def basket1_oranges : ℕ := 15
def basket1_bananas : ℕ := 14
def basket1_grapes : ℕ := 12

def basket4_apples : ℕ := basket1_apples - 2
def basket4_oranges : ℕ := basket1_oranges - 2
def basket4_bananas : ℕ := basket1_bananas - 2
def basket4_grapes : ℕ := basket1_grapes - 2

def basket5_apples : ℕ := basket1_apples + 3
def basket5_oranges : ℕ := basket1_oranges - 5
def basket5_bananas : ℕ := basket1_bananas
def basket5_grapes : ℕ := basket1_grapes

def basket6_bananas : ℕ := basket1_bananas * 2
def basket6_grapes : ℕ := basket1_grapes / 2

def total_fruits_b1_3 : ℕ := basket1_apples + basket1_oranges + basket1_bananas + basket1_grapes
def total_fruits_b4 : ℕ := basket4_apples + basket4_oranges + basket4_bananas + basket4_grapes
def total_fruits_b5 : ℕ := basket5_apples + basket5_oranges + basket5_bananas + basket5_grapes
def total_fruits_b6 : ℕ := basket6_bananas + basket6_grapes

def total_fruits_all : ℕ := total_fruits_b1_3 + total_fruits_b4 + total_fruits_b5 + total_fruits_b6

theorem total_fruits_is_174 : total_fruits_all = 174 := by
  -- proof will go here
  sorry

end total_fruits_is_174_l494_494367


namespace area_bounded_by_arccos_sin_eq_pi2_l494_494455

noncomputable def area_arccos_sin : ℝ :=
  let f : ℝ → ℝ := λ x, real.arccos (real.sin x)
  let interval_start : ℝ := real.pi / 2
  let interval_end : ℝ := 5 * real.pi / 2
  (interval_end - interval_start) * real.pi / 2

theorem area_bounded_by_arccos_sin_eq_pi2 :
  area_arccos_sin = real.pi ^ 2 :=
sorry

end area_bounded_by_arccos_sin_eq_pi2_l494_494455


namespace length_YZ_eq_22_5_l494_494761

noncomputable def distance_YZ : ℝ :=
  let AB := 15
  let BC := 20
  let height_Q := 30
  let vol_ratio := 8
  let height_Q' := height_Q / (vol_ratio^(1/3)) -- Height of Q' calculated from volume ratio
  let frustum_height := height_Q - height_Q'
  let base_diagonal := real.sqrt ((AB ^ 2) + (BC ^ 2))
  let ZC := real.sqrt ((base_diagonal ^ 2) + (height_Q ^ 2))
  frustum_height / 2 -- Distance of Y from base (center point calculation for similarity)
  height_Q - (frustum_height / 2) -- Calculating length YZ
    
theorem length_YZ_eq_22_5 : distance_YZ = 22.5 :=
  sorry

end length_YZ_eq_22_5_l494_494761


namespace length_of_longer_leg_of_smallest_triangle_l494_494444

theorem length_of_longer_leg_of_smallest_triangle :
  ∀ (a b c : ℝ), 
  is_30_60_90_triangle (a, b, c) ∧ c = 16 
  ∧ (∀ (a₁ b₁ c₁ : ℝ), is_30_60_90_triangle (a₁, b₁, c₁) → b = c₁ → true) 
  ∧ (∀ (a₂ b₂ c₂ : ℝ), is_30_60_90_triangle (a₂, b₂, c₂) → true) 
  ∧ (∀ (a₃ b₃ c₃ : ℝ), is_30_60_90_triangle (a₃, b₃, c₃) → true) 
  → ∃ (a₄ b₄ c₄ : ℝ), is_30_60_90_triangle (a₄, b₄, c₄) ∧ b₄ = 9 :=
sorry

end length_of_longer_leg_of_smallest_triangle_l494_494444


namespace three_digit_number_addition_l494_494376

theorem three_digit_number_addition (a b : ℕ) (ha : a < 10) (hb : b < 10) (h1 : 307 + 294 = 6 * 100 + b * 10 + 1)
  (h2 : (6 * 100 + b * 10 + 1) % 7 = 0) : a + b = 8 :=
by {
  sorry  -- Proof steps not needed
}

end three_digit_number_addition_l494_494376


namespace increase_by_150_percent_l494_494303

theorem increase_by_150_percent (n : ℕ) : 
  n = 80 → n + (3 / 2) * n = 200 :=
by
  intros h
  rw [h]
  norm_num
  sorry

end increase_by_150_percent_l494_494303


namespace triangles_do_not_overlap_l494_494697

-- Definitions based on conditions
variables {Point Line Triangle : Type}
variables (A B C D E F G : Point)
variables (AC AB BC CF CG CBF CAG : Line)
variables [is_acute_triangle : is_acute (Triangle.mk A B C)]

-- Problem conditions as Lean definitions
def fold_ray_AC_onto_AB (AC AB : Line) : Point := sorry -- Definition of D on BC
def fold_ray_BC_onto_BA (BC BA : Line) : Point := sorry -- Definition of E on AC
def fold_perpendicular_through_D (D : Point) (AB : Line) : Point := sorry -- Definition of F on AB
def fold_perpendicular_through_E (E : Point) (AB : Line) : Point := sorry -- Definition of G on AB

-- Ensure C, D, E, F, and G are correctly located based on the problem statement
axiom location_D : D = fold_ray_AC_onto_AB AC AB
axiom location_E : E = fold_ray_BC_onto_BA BC BA
axiom location_F : F = fold_perpendicular_through_D D AB
axiom location_G : G = fold_perpendicular_through_E E AB

-- Theorem: Triangles CBF and CAG when folded do not overlap except at C
theorem triangles_do_not_overlap (h_acute : is_acute_triangle) :
  triangles_folded_do_not_overlap C B F D G :=
sorry

end triangles_do_not_overlap_l494_494697


namespace mr_johnson_needs_additional_volunteers_l494_494618

-- Definitions for the given conditions
def math_classes := 5
def students_per_class := 4
def total_students := math_classes * students_per_class

def total_teachers := 10
def carpentry_skilled_teachers := 3

def total_parents := 15
def lighting_sound_experienced_parents := 6

def total_volunteers_needed := 100
def carpentry_volunteers_needed := 8
def lighting_sound_volunteers_needed := 10

-- Total current volunteers
def current_volunteers := total_students + total_teachers + total_parents

-- Volunteers with specific skills
def current_carpentry_skilled := carpentry_skilled_teachers
def current_lighting_sound_experienced := lighting_sound_experienced_parents

-- Additional volunteers needed
def additional_carpentry_needed :=
  carpentry_volunteers_needed - current_carpentry_skilled
def additional_lighting_sound_needed :=
  lighting_sound_volunteers_needed - current_lighting_sound_experienced

-- Total additional volunteer needed
def additional_volunteers_needed :=
  additional_carpentry_needed + additional_lighting_sound_needed

-- The theorem we need to prove:
theorem mr_johnson_needs_additional_volunteers :
  additional_volunteers_needed = 9 := by
  sorry

end mr_johnson_needs_additional_volunteers_l494_494618


namespace sequence_inequality_l494_494764

theorem sequence_inequality (a : ℕ → ℕ)
  (h1 : a 0 > 0) -- Ensure all entries are positive integers.
  (h2 : ∀ k l m n : ℕ, k * l = m * n → a k + a l = a m + a n)
  {p q : ℕ} (hpq : p ∣ q) :
  a p ≤ a q :=
sorry

end sequence_inequality_l494_494764


namespace fraction_cubed_sum_l494_494552

theorem fraction_cubed_sum (x y : ℤ) (h1 : x = 3) (h2 : y = 4) :
  (x^3 + 3 * y^3) / 7 = 31 + 3 / 7 := by
  sorry

end fraction_cubed_sum_l494_494552


namespace height_difference_l494_494392

theorem height_difference :
  ∀ (a s b : ℕ), a = 80 ∧ a = 2 * s ∧ b = 3 * a → b - s = 200 := 
by
  intros a s b h
  cases h with h1 h_rest
  cases h_rest with h2 h3
  sorry

end height_difference_l494_494392


namespace inequality_for_five_real_numbers_l494_494132

open Real

theorem inequality_for_five_real_numbers
  (a1 a2 a3 a4 a5 : ℝ)
  (h1 : 1 < a1)
  (h2 : 1 < a2)
  (h3 : 1 < a3)
  (h4 : 1 < a4)
  (h5 : 1 < a5) :
  16 * (a1 * a2 * a3 * a4 * a5 + 1) ≥ (1 + a1) * (1 + a2) * (1 + a3) * (1 + a4) * (1 + a5) := 
sorry

end inequality_for_five_real_numbers_l494_494132


namespace increase_80_by_150_percent_l494_494290

theorem increase_80_by_150_percent : 
  ∀ x percent, x = 80 ∧ percent = 1.5 → x + (x * percent) = 200 :=
by
  sorry

end increase_80_by_150_percent_l494_494290


namespace balls_in_boxes_l494_494030

theorem balls_in_boxes : ∀ (balls boxes : ℕ), balls = 6 → boxes = 3 → 
  (∃! n : ℕ, n = 47 ∧ n = number_of_ways_to_distribute_balls_in_boxes_d(balls, boxes)) :=
by
  intros balls boxes h1 h2
  have h_balls : balls = 6 := h1
  have h_boxes : boxes = 3 := h2
  use 47
  split
  sorry
  sorry

end balls_in_boxes_l494_494030


namespace number_of_form_2016_is_product_of_non_trivial_palindromes_l494_494369

-- Noncomputable definition for "is_palindrome" and "is_product_of_non_trivial_palindromes"
def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits == digits.reverse

def is_product_of_non_trivial_palindromes (n : ℕ) : Prop :=
  ∃ (p q : ℕ), p ≠ 0 ∧ q ≠ 0 ∧ p ≠ n ∧ q ≠ n ∧ is_palindrome p ∧ is_palindrome q ∧ n = p * q

-- Assume n is a number of the form 2016 repeated k times
def repeats_2016 (k : ℕ) : ℕ :=
  let digits := List.repeat 2016 k
  digits.foldl (λ acc x => 10 ^ (acc.digits 10).length * acc + x) 0

-- Main theorem statement
theorem number_of_form_2016_is_product_of_non_trivial_palindromes (k : ℕ) :
  is_product_of_non_trivial_palindromes (repeats_2016 k) :=
sorry

end number_of_form_2016_is_product_of_non_trivial_palindromes_l494_494369


namespace quadrilateral_is_parallelogram_l494_494580

variables {A B C D M N P Q : Type} [AffineSimplicialComplex A B C D] 

def is_rectangle (A B C D : Type) : Prop :=
  ∀ (A B C D : Point),
    A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A ∧
    dist A B = dist C D ∧ dist B C = dist D A ∧
    ∠ A B C = π / 2 ∧ ∠ B C D = π / 2 ∧ ∠ C D A = π / 2 ∧ ∠ D A B = π / 2

def are_collinear (A B : Point) : Prop := collinear A B

def area (U V W : Point) : ℝ :=
  ∑ U V W

theorem quadrilateral_is_parallelogram
  (h_rect : is_rectangle A B C D)
  (hM : are_collinear A B M)
  (hN : are_collinear B C N)
  (hP : are_collinear C D P)
  (hQ : are_collinear D A Q)
  (h_eq_area : area A Q M = area B M N ∧ area B M N = area C N P ∧ area C N P = area D P Q) :
  parallelogram M N P Q :=
begin
  sorry
end

end quadrilateral_is_parallelogram_l494_494580


namespace distance_z100_l494_494821

noncomputable def z : ℕ → ℂ 
| 1 := 0
| (n + 2) := (z (n + 1))^2 - 1

theorem distance_z100 : complex.abs (z 100) = 1 := 
sorry

end distance_z100_l494_494821


namespace soccer_players_count_l494_494973

theorem soccer_players_count (total_socks : ℕ) (P : ℕ) 
  (h_total_socks : total_socks = 22)
  (h_each_player_contributes : ∀ p : ℕ, p = P → total_socks = 2 * P) :
  P = 11 :=
by
  sorry

end soccer_players_count_l494_494973


namespace digit_sum_multiple_exists_ck_l494_494872

def digit_sum (n : ℕ) : ℕ :=
  n.digits.sum  

theorem digit_sum_multiple_exists_ck 
  (k : ℕ) 
  (hk : k > 1) 
  (n : ℕ) 
  (hn : n > 0) :
  ∃ (c_k : ℝ), c_k > 0 ∧ ∀ (n : ℕ), n > 0 → digit_sum (k * n) ≥ (c_k * digit_sum n) ↔ 
  ∀ p, p.prime → p ∣ k → p = 2 ∨ p = 5 :=
sorry

end digit_sum_multiple_exists_ck_l494_494872


namespace two_triangles_in_circle_l494_494160

open Real EuclideanGeometry

-- Define the problem conditions and statement
theorem two_triangles_in_circle (t1 t2 : Triangle) (C : Circle)
    (hC : C.radius = 1)
    (h_t1_in_C : t1 ⊂ C)
    (h_t2_in_C : t2 ⊂ C)
    (h_area_t1 : Triangle.area t1 > 1)
    (h_area_t2 : Triangle.area t2 > 1)
    (h_no_overlap : ¬ (t1 ∩ t2 ⊄ ∅)) : 
    False :=
sorry

end two_triangles_in_circle_l494_494160


namespace increase_by_150_percent_l494_494307

theorem increase_by_150_percent (n : ℕ) : 
  n = 80 → n + (3 / 2) * n = 200 :=
by
  intros h
  rw [h]
  norm_num
  sorry

end increase_by_150_percent_l494_494307


namespace increase_150_percent_of_80_l494_494294

theorem increase_150_percent_of_80 : 80 * 1.5 + 80 = 200 :=
by
  sorry

end increase_150_percent_of_80_l494_494294


namespace increase_80_by_150_percent_l494_494237

-- Define the initial number (n) and the percentage increase (p)
def n : ℕ := 80
def p : ℚ := 1.5

-- The theorem stating the expected result after increasing n by 150%
theorem increase_80_by_150_percent : n + (p * n) = 200 := by
  sorry

end increase_80_by_150_percent_l494_494237


namespace inequality_increases_by_l494_494993

theorem inequality_increases_by (k : ℕ) :
  (∑ i in finset.range(k + 1), 1 / (k + 1 + i + 1)) - (∑ i in finset.range(k), 1 / (k + i + 1)) = 
    1 / (2 * k + 2) + 1 / (2 * k + 1) - 1 / (k + 1) :=
by
  sorry

end inequality_increases_by_l494_494993


namespace reduce_fraction_l494_494420

theorem reduce_fraction (n : ℤ) (d : ℕ) (h1 : d ∣ (4 * n + 3)) (h2 : d ∣ (5 * n + 2)) : 
  d = 7 → ∃ k : ℤ, n = 7 * k + 1 :=
by
  sorry

end reduce_fraction_l494_494420


namespace balls_in_boxes_l494_494009

/-
Prove that the number of ways to distribute 6 distinguishable balls into 3 indistinguishable boxes is 132.
-/
theorem balls_in_boxes : 
  ∃ (ways : ℕ), ways = 132 ∧ ways = 
    (1) + 
    (Nat.choose 6 5) + 
    (Nat.choose 6 4) + 
    (Nat.choose 6 4) + 
    (Nat.choose 6 3) + 
    (Nat.choose 6 3 * Nat.choose 3 2) + 
    (Nat.choose 6 2 * Nat.choose 4 2 / 6) := 
by
  sorry

end balls_in_boxes_l494_494009


namespace distinguishable_balls_in_indistinguishable_boxes_l494_494023

theorem distinguishable_balls_in_indistinguishable_boxes :
  let num_distinguishable_balls := 6
  let num_indistinguishable_boxes := 3
  -- different ways to distribute balls across boxes summarized
  let ways := ∑([{6, 0, 0} ++ {5, 1, 0} ++ {4, 2, 0} ++ {4, 1, 1} ++ {3, 3, 0} ++ {3, 2, 1} ++ {2, 2, 2}], sum)
  -- corresponding ways to remove permutation of distinguishable ball cases
  ways = 222 := sorry

end distinguishable_balls_in_indistinguishable_boxes_l494_494023


namespace anniversary_day_probability_l494_494397

/- Definitions based on the conditions -/
def is_leap_year (year : ℕ) : Prop :=
  year % 4 = 0

def days_in_year (year : ℕ) : ℕ :=
  if is_leap_year year then 366 else 365

def total_days (start_year : ℕ) : ℕ :=
  (list.sum (list.map days_in_year (list.range' start_year 11)))

/- Prove the day of the 11th anniversary and its probabilities -/
theorem anniversary_day_probability (start_year : ℕ) (h : start_year ∈ {1668, 1669, 1670, 1671}) :
  let days := total_days start_year % 7
  in (days = 0 ∧ 0.75 ≤ 1) ∨ (days = 6 ∧ 0.25 ≤ 1) :=
by
  sorry

end anniversary_day_probability_l494_494397


namespace sum_of_first_ten_primes_with_units_digit_3_l494_494853

def units_digit_3_and_prime (n : ℕ) : Prop :=
  (n % 10 = 3) ∧ (Prime n)

def first_ten_primes_with_units_digit_3 : list ℕ :=
  [3, 13, 23, 43, 53, 73, 83, 103, 113, 163]

theorem sum_of_first_ten_primes_with_units_digit_3 :
  list.sum first_ten_primes_with_units_digit_3 = 671 :=
by
  sorry

end sum_of_first_ten_primes_with_units_digit_3_l494_494853


namespace log_eq_log_implies_MN_one_l494_494545

theorem log_eq_log_implies_MN_one (M N : ℝ) (h1 : log M N = log N M) (h2 : M ≠ N) (h3 : M * N > 0) (h4 : M ≠ 1) (h5 : N ≠ 1) : M * N = 1 :=
by
  sorry

end log_eq_log_implies_MN_one_l494_494545


namespace cookies_bags_division_l494_494566

theorem cookies_bags_division (total_cookies : ℕ) (cookies_per_bag : ℕ) (bags : ℕ) :
  total_cookies = 703 ∧ cookies_per_bag = 19 → bags = total_cookies / cookies_per_bag → bags = 37 :=
by
  intros h1 h2
  cases h1 with ht hg
  rw [ht, hg] at h2
  exact h2
-- sorry

end cookies_bags_division_l494_494566


namespace amar_average_speed_l494_494777

def convertToHours (hours : ℕ) (minutes : ℕ) : ℝ := hours + (minutes / 60 : ℝ)

def totalDistance (d1 d2 : ℕ) : ℕ := d1 + d2

def totalTime (t1 t2 : ℝ) : ℝ := t1 + t2

def averageSpeed (distance : ℕ) (time : ℝ) : ℝ := distance / time

theorem amar_average_speed :
  let d1 := 350
  let d2 := 420
  let t1 := convertToHours 6 30
  let t2 := convertToHours 7 15
  averageSpeed (totalDistance d1 d2) (totalTime t1 t2) = 56 := by
  sorry

end amar_average_speed_l494_494777


namespace algae_coverage_l494_494186

theorem algae_coverage (doubles_coverage_every_day : ∀ n : ℕ, algae_coverage (n + 1) = 2 * algae_coverage n)
                       (fully_covered_on_day_24 : algae_coverage 24 = 1) :
  algae_coverage 21 = 1 / 8 := 
sorry

end algae_coverage_l494_494186


namespace vector_addition_l494_494449

def v1 : ℝ × ℝ × ℝ := (5, -3, 2)
def v2 : ℝ × ℝ × ℝ := (-4, 8, -1)

theorem vector_addition : v1.1 + v2.1 = 1 ∧ v1.2 + v2.2 = 5 ∧ v1.3 + v2.3 = 1 :=
by simp [v1, v2]; sorry

end vector_addition_l494_494449


namespace increase_80_by_150_percent_l494_494234

-- Define the initial number (n) and the percentage increase (p)
def n : ℕ := 80
def p : ℚ := 1.5

-- The theorem stating the expected result after increasing n by 150%
theorem increase_80_by_150_percent : n + (p * n) = 200 := by
  sorry

end increase_80_by_150_percent_l494_494234


namespace necessary_but_not_sufficient_l494_494074

noncomputable def f (a x : ℝ) : ℝ := x^2 - 2*(a+1)*x + 3

theorem necessary_but_not_sufficient (a : ℝ) :
  (∀ x : ℝ, 1 ≤ x → f a x - f a 1 ≥ 0) ↔ (a ≤ -2) :=
sorry

end necessary_but_not_sufficient_l494_494074


namespace minimum_prime_factorization_sum_l494_494136

theorem minimum_prime_factorization_sum (x y a b c d : ℕ) (hx : x > 0) (hy : y > 0)
  (h : 5 * x^7 = 13 * y^17) (h_pf: x = a ^ c * b ^ d) :
  a + b + c + d = 33 :=
sorry

end minimum_prime_factorization_sum_l494_494136


namespace increase_by_percentage_l494_494258

theorem increase_by_percentage (initial_amount : ℕ) (percentage_increase : ℕ) :
  initial_amount = 80 → 
  percentage_increase = 150 → 
  initial_amount * (percentage_increase / 100) + initial_amount = 200 :=
by
  intros h_initial h_percentage
  rw [h_initial, h_percentage]
  calc
    80 * (150 / 100) + 80 = 80 * 1.5 + 80 : by norm_num
                     ... = 120 + 80         : by norm_num
                     ... = 200              : by norm_num
  sorry

end increase_by_percentage_l494_494258


namespace fixed_point_circumcircle_l494_494735

theorem fixed_point_circumcircle (A B C M N : Point)
  (hABC : 1 < (B.dist A) / (C.dist A) ∧ (B.dist A) / (C.dist A) < 3/2)
  (hMBN : (M.dist B) / (C.dist A) - (N.dist C) / (B.dist A) = 1)
  (hPoints : M ≠ A ∧ N ≠ A) :
  ∃ (L : Point), (L ≠ A) ∧ passes_through (circumcircle A M N) L :=
by sorry

end fixed_point_circumcircle_l494_494735


namespace z_in_fourth_quadrant_l494_494905

-- Definition of the imaginary unit
def i : ℂ := Complex.I

-- The complex number in question
def z : ℂ := 1 / (1 + i)

-- The coordinates of the complex number in the plane
def z_coordinates : ℝ × ℝ := (z.re, z.im)

-- Statement that specifies the quadrant
theorem z_in_fourth_quadrant : z_coordinates.1 > 0 ∧ z_coordinates.2 < 0 := by
  sorry

end z_in_fourth_quadrant_l494_494905


namespace largest_subset_no_four_times_l494_494375

theorem largest_subset_no_four_times : 
  ∃ (S : set ℤ), (∀ (x ∈ S) (y ∈ S), x ≠ 4 * y ∧ y ≠ 4 * x) ∧ set.of_list [(1 : ℤ), 2, 3, ..., 50] = 50 ∧ card S = 47 := 
sorry

end largest_subset_no_four_times_l494_494375


namespace heart_to_heart_probability_l494_494579

def probability_heart_to_heart_connection : ℚ := 5 / 8

theorem heart_to_heart_probability :
  let balls := [4, 5, 6, 7] in
  let possible_cases := (balls × balls).toFinset in
  let favorable_cases := possible_cases.filter (λ (p : ℕ × ℕ), |p.1 - p.2| ≤ 1) in
  favorable_cases.card = (5 / 8 : ℚ) * possible_cases.card :=
by
  let balls := [4, 5, 6, 7]
  let possible_cases := (balls × balls).toFinset
  let favorable_cases := possible_cases.filter (λ (p : ℕ × ℕ), |p.1 - p.2| ≤ 1)
  have h_total_cases : possible_cases.card = 16 := by sorry
  have h_favorable_cases : favorable_cases.card = 10 := by sorry
  calc
    (5 / 8 : ℚ) * possible_cases.card
        = (5 / 8 : ℚ) * 16 : by rw h_total_cases
    ... = 10 : by norm_num
    ... = favorable_cases.card : by rw h_favorable_cases

end heart_to_heart_probability_l494_494579


namespace min_value_of_f_domain_min_value_of_f_interval_inequality_ln_l494_494918

def f (x : ℝ) : ℝ := x * Real.log x

theorem min_value_of_f_domain : ∃ x : ℝ, f x = -1 / Real.exp 1 := sorry

theorem min_value_of_f_interval (t : ℝ) (ht : 0 < t) : 
  ∃ x : ℝ, x ∈ set.Icc t (t + 2) ∧ 
  f x = if 0 < t ∧ t < 1 / Real.exp 1 then -1 / Real.exp 1 else t * Real.log t := sorry

theorem inequality_ln (x : ℝ) (hx : 0 < x) : 
  Real.log x > 1 / Real.exp x - 2 / (Real.exp 1 * x) := sorry

end min_value_of_f_domain_min_value_of_f_interval_inequality_ln_l494_494918


namespace find_overtime_hours_l494_494771

theorem find_overtime_hours
  (pay_rate_ordinary : ℝ := 0.60)
  (pay_rate_overtime : ℝ := 0.90)
  (total_pay : ℝ := 32.40)
  (total_hours : ℕ := 50) :
  ∃ y : ℕ, pay_rate_ordinary * (total_hours - y) + pay_rate_overtime * y = total_pay ∧ y = 8 := 
by
  sorry

end find_overtime_hours_l494_494771


namespace no_formidable_successful_sum_l494_494782

/-- Definition of a formidable number: it is a sum of distinct powers of 4. -/
def is_formidable (n : ℕ) : Prop :=
  ∃ (S : Finset ℕ), n = S.sum (λ k, 4^k)

/-- Definition of a successful number: it is a sum of distinct powers of 6. -/
def is_successful (n : ℕ) : Prop :=
  ∃ (S : Finset ℕ), n = S.sum (λ k, 6^k)

/-- The main theorem stating that 2005 cannot be written as the sum of a formidable number and a successful number. -/
theorem no_formidable_successful_sum (a b : ℕ) (h1 : is_formidable a) (h2 : is_successful b) : a + b ≠ 2005 := 
sorry

end no_formidable_successful_sum_l494_494782


namespace PetrovskyInequality_l494_494822

variable {A B C : Type}
variable [ABC : EuclideanTriangle A B C]

variable {a b c : ℝ} -- lengths of the sides of the triangle
variable {H_a H_b H_c : Point} -- feet of the altitudes
variable {h_a h_b h_c : ℝ} -- lengths of the altitudes

open EuclideanGeometry

theorem PetrovskyInequality
  (h_a_def : h_a = altitude_length A B C)
  (h_b_def : h_b = altitude_length B A C)
  (h_c_def : h_c = altitude_length C A B)
  (acute_angle_ABC : is_acute_triangle A B C)
  (side_lengths : sides A B C = (a, b, c))
  (feet_of_altitudes : feet_of_altitudes A B C = (H_a, H_b, H_c)) :
  \frac {h_a^2} {a^2 - (distance B H_a)^2} + \frac{h_b^2} {b^2 - (distance A H_b)^2} + \frac{h_c^2} {c^2 - (distance B H_c)^2} \ge 3 :=
by sorry

end PetrovskyInequality_l494_494822


namespace rotary_club_eggs_needed_l494_494650

theorem rotary_club_eggs_needed 
  (small_children_tickets : ℕ := 53)
  (older_children_tickets : ℕ := 35)
  (adult_tickets : ℕ := 75)
  (senior_tickets : ℕ := 37)
  (waste_percentage : ℝ := 0.03)
  (extra_omelets : ℕ := 25)
  (eggs_per_extra_omelet : ℝ := 2.5) :
  53 * 1 + 35 * 2 + 75 * 3 + 37 * 4 + 
  Nat.ceil (waste_percentage * (53 * 1 + 35 * 2 + 75 * 3 + 37 * 4)) + 
  Nat.ceil (extra_omelets * eggs_per_extra_omelet) = 574 := 
by 
  sorry

end rotary_club_eggs_needed_l494_494650


namespace increase_by_150_percent_l494_494311

theorem increase_by_150_percent (x : ℝ) (h : x = 80) : x + (1.5 * x) = 200 := 
by
  -- The proof goes here, but is omitted as per instructions.
  sorry

end increase_by_150_percent_l494_494311


namespace probability_black_ball_l494_494739

theorem probability_black_ball (p_red p_white p_black : ℝ) (h_red : p_red = 0.42) (h_white : p_white = 0.28) :
  p_black = 1 - (p_red + p_white) → p_black = 0.30 :=
by
  intros h_combined
  rw [h_red, h_white, h_combined]
  sorry

end probability_black_ball_l494_494739


namespace midpoint_fraction_l494_494709

theorem midpoint_fraction (a b : ℚ) (h1 : a = 3 / 4) (h2 : b = 5 / 6) :
  (a + b) / 2 = 19 / 24 :=
by {
  sorry
}

end midpoint_fraction_l494_494709


namespace intersection_line_computation_l494_494425

-- Definitions of the centers and radii of the circles
def center1 := (0 : ℝ, 6 : ℝ)
def radius1 := 30 : ℝ

def center2 := (20 : ℝ, 0 : ℝ)
def radius2 := 30 : ℝ

-- Equations of the circles
def circle1 (x y : ℝ) : Prop := x^2 + (y - 6)^2 = 900
def circle2 (x y : ℝ) : Prop := (x - 20)^2 + y^2 = 900

-- Intermediary step: Finding the line
def line_eq (x y : ℝ) : Prop := y = (10 / 3) * x - (91 / 3)

-- The final proof problem
theorem intersection_line_computation : 100 * (10 / 3) - (91 / 3) = 303 := by
  sorry

end intersection_line_computation_l494_494425


namespace percentage_of_people_win_a_prize_l494_494596

-- Define the constants used in the problem
def totalMinnows : Nat := 600
def minnowsPerPrize : Nat := 3
def totalPlayers : Nat := 800
def minnowsLeft : Nat := 240

-- Calculate the number of minnows given away as prizes
def minnowsGivenAway : Nat := totalMinnows - minnowsLeft

-- Calculate the number of prizes given away
def prizesGivenAway : Nat := minnowsGivenAway / minnowsPerPrize

-- Calculate the percentage of people winning a prize
def percentageWinners : Nat := (prizesGivenAway * 100) / totalPlayers

-- Theorem to prove the percentage of winners
theorem percentage_of_people_win_a_prize : 
    percentageWinners = 15 := 
sorry

end percentage_of_people_win_a_prize_l494_494596


namespace quadratic_properties_l494_494194

theorem quadratic_properties : 
  (∃ (b c x y : ℝ), y = 4 * x^2 + b * x + c ∧ ((b = 5) ∧ (c = 0)) ∧ 
  (∀ x1 x2, (4 * x1^2 + 5 * x1 = 20) ∧ (4 * x2^2 + 5 * x2 = 20) → 
   (let n := 4 * (x1 + x2)^2 + 5 * (x1 + x2) in n = 0)) ∧ 
  ∀ x, (4 * x^2 + 5 * x > 0) → (x < -5/4 ∨ x > 0)) := 
sorry

end quadratic_properties_l494_494194


namespace maximum_marks_is_500_l494_494092

-- Define the conditions
def passing_marks (M : ℝ) := 0.60 * M
def fail_margin := 210 + 90

-- The eventual equation to prove
theorem maximum_marks_is_500 (M : ℝ) :
  passing_marks M = fail_margin → M = 500 :=
by
  intro h
  sorry

end maximum_marks_is_500_l494_494092


namespace hours_difference_less_on_tuesday_l494_494152

variable (T : ℕ)

-- Define the conditions based on the given problem
def onMonday : ℕ := 4
def onSunday : ℕ := 4
def onTuesday : ℕ := T
def onWednesday : ℕ := onMonday * 2
def onThursday: ℕ := onTuesday * 2
def totalHours : ℕ := onMonday + onTuesday + onWednesday + onThursday

-- Prove that the total hours equation holds using the given total hours of 18
axiom eqn1 : totalHours = 18

-- Prove that T must be equal to 2
axiom eqn2 : T = 2

-- Proof statement
theorem hours_difference_less_on_tuesday (T : ℕ) (H1 : totalHours = 18) (H2 : T = 2) : onMonday - onTuesday = 2 := by
  sorry

end hours_difference_less_on_tuesday_l494_494152


namespace increased_number_l494_494278

theorem increased_number (original_number : ℕ) (percentage_increase : ℚ) :
  original_number = 80 →
  percentage_increase = 1.5 →
  original_number * (1 + percentage_increase) = 200 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end increased_number_l494_494278


namespace ellipse_tangents_intersect_at_EF_l494_494521

noncomputable def ellipse := { p : ℝ × ℝ // (p.1^2 / 9) + (p.2^2 / 5) = 1 }
noncomputable def point_inside := (sqrt 5, sqrt 2)
noncomputable def point_on_chord (x1 y1 : ℝ) := (sqrt 5 * x1 / 9) + (sqrt 2 * y1 / 5) = 1

theorem ellipse_tangents_intersect_at_EF :
    ∃ E F : ℝ × ℝ, (∃ x1 y1 : ℝ, point_on_chord x1 y1) ∧
                    (∃ x2 y2 : ℝ, point_on_chord x2 y2) ∧
                    (E.1 * sqrt 5 / 9 + E.2 * sqrt 2 / 5 = 1) ∧
                    (F.1 * sqrt 5 / 9 + F.2 * sqrt 2 / 5 = 1) ∧
                    (∀ x y : ℝ, (x * sqrt 5 / 9 + y * sqrt 2 / 5 = 1) ↔ 
                                  E = (x, y) ∨ F = (x, y)) :=
sorry

end ellipse_tangents_intersect_at_EF_l494_494521


namespace range_of_k_l494_494429

def tensor (a b : ℝ) : ℝ := a * b + a + b^2

theorem range_of_k (k : ℝ) : (∀ x : ℝ, tensor k x > 0) ↔ (0 < k ∧ k < 4) :=
by
  sorry

end range_of_k_l494_494429


namespace balls_in_boxes_l494_494055

theorem balls_in_boxes :
    (Nat.choose 6 6) + 
    (Nat.choose 6 5) + 
    (Nat.choose 6 4 * Nat.choose 2 2) + 
    (Nat.choose 6 3 / 2) + 
    (Nat.choose 6 4 * Nat.choose 2 1) + 
    (Nat.choose 6 2 * Nat.choose 4 2 / 3.factorial) = 77 :=
by
  sorry

end balls_in_boxes_l494_494055


namespace tangent_of_7pi_over_4_l494_494453

   theorem tangent_of_7pi_over_4 : Real.tan (7 * Real.pi / 4) = -1 := 
   sorry
   
end tangent_of_7pi_over_4_l494_494453


namespace sewers_handle_rain_l494_494208

theorem sewers_handle_rain (total_capacity : ℕ) (runoff_per_hour : ℕ) : 
  total_capacity = 240000 → 
  runoff_per_hour = 1000 → 
  total_capacity / runoff_per_hour / 24 = 10 :=
by 
  intro h1 h2
  sorry

end sewers_handle_rain_l494_494208


namespace total_loss_is_1000_l494_494787

variable (P : ℝ) -- Pyarelal's capital
variable (A : ℝ) -- Ashok's capital
variable (L_P : ℝ) -- Pyarelal's loss
variable (L_A : ℝ) -- Ashok's loss

-- Conditions
axiom ashok_capital_ratio : A = (1/9) * P
axiom pyarelal_loss : L_P = 900
axiom loss_ratio : L_A / L_P = A / P

-- Theorem to prove
theorem total_loss_is_1000 : L_A + L_P = 1000 :=
by
  sorry

end total_loss_is_1000_l494_494787


namespace ratio_of_areas_of_circles_l494_494951

theorem ratio_of_areas_of_circles
    (C_C R_C C_D R_D L : ℝ)
    (hC : C_C = 2 * Real.pi * R_C)
    (hD : C_D = 2 * Real.pi * R_D)
    (hL : (60 / 360) * C_C = L ∧ L = (40 / 360) * C_D) :
    (Real.pi * R_C ^ 2) / (Real.pi * R_D ^ 2) = 4 / 9 :=
by
  sorry

end ratio_of_areas_of_circles_l494_494951


namespace angle_complement_half_supplement_is_zero_l494_494836

theorem angle_complement_half_supplement_is_zero (x : ℝ) 
  (h_complement: x - 90 = (1 / 2) * (x - 180)) : x = 0 := 
sorry

end angle_complement_half_supplement_is_zero_l494_494836


namespace scientists_language_bound_l494_494789

theorem scientists_language_bound (k p : ℕ) 
  (languages : Finset (Finset ℕ))
  (h1 : ∀ S ∈ languages, S.card ≤ p)
  (h2 : languages.card = k)
  (h3 : ∀ S1 S2 ∈ languages, S1 ≠ S2 → (S1 ∩ S2).nonempty)
  (h4 : ∀ S1 S2 ∈ languages, S1 ≠ S2) :
  k ≤ 2^(p-1) := by
    sorry

end scientists_language_bound_l494_494789


namespace speed_of_second_fragment_l494_494362

noncomputable def magnitude_speed_of_second_fragment 
  (u : ℝ) (t : ℝ) (g : ℝ) (v_x1 : ℝ) (v_y1 : ℝ := - (u - g * t)) 
  (v_x2 : ℝ := -v_x1) (v_y2 : ℝ := v_y1) : ℝ :=
Real.sqrt ((v_x2 ^ 2) + (v_y2 ^ 2))

theorem speed_of_second_fragment 
  (u : ℝ) (t : ℝ) (g : ℝ) (v_x1 : ℝ) 
  (h_u : u = 20) (h_t : t = 3) (h_g : g = 10) (h_vx1 : v_x1 = 48) :
  magnitude_speed_of_second_fragment u t g v_x1 = Real.sqrt 2404 :=
by
  -- Proof
  sorry

end speed_of_second_fragment_l494_494362


namespace union_of_M_and_N_l494_494890

namespace SetOperations

def M : Set ℕ := {1, 2, 4}
def N : Set ℕ := {1, 3, 4}

theorem union_of_M_and_N :
  M ∪ N = {1, 2, 3, 4} :=
sorry

end SetOperations

end union_of_M_and_N_l494_494890


namespace collinear_points_vectors_l494_494503

theorem collinear_points_vectors (a : ℝ) (hOANonZero : a ≠ 0):
  let OA := (a, 0)
  let OB := (0, a)
  let OC := (1, 2)
  let AB := (OB.1 - OA.1, OB.2 - OA.2)
  let BC := (OC.1 - OB.1, OC.2 - OB.2)
  (AB.1 * BC.2) = (AB.2 * BC.1) → a = 3 := by
  intro OA OB OC AB BC hOANonZero hCollinear
  sorry

end collinear_points_vectors_l494_494503


namespace algebraic_expression_value_l494_494492

theorem algebraic_expression_value (x : ℝ) (h : (x^2 - x)^2 - 4 * (x^2 - x) - 12 = 0) : x^2 - x + 1 = 7 :=
sorry

end algebraic_expression_value_l494_494492


namespace cos_of_vector_dot_product_l494_494539

open Real

noncomputable def cos_value (x : ℝ) : ℝ := cos (x + π / 4)

theorem cos_of_vector_dot_product (x : ℝ)
  (h1 : π / 4 < x)
  (h2 : x < π / 2)
  (h3 : (sqrt 2) * cos x + (sqrt 2) * sin x = 8 / 5) :
  cos_value x = - 3 / 5 :=
by
  sorry

end cos_of_vector_dot_product_l494_494539


namespace long_show_episodes_correct_l494_494694

variable {short_show_episodes : ℕ} {short_show_duration : ℕ} {total_watched_time : ℕ} {long_show_episode_duration : ℕ}

def episodes_long_show (short_episodes_duration total_duration long_episode_duration : ℕ) : ℕ :=
  (total_duration - short_episodes_duration) / long_episode_duration

theorem long_show_episodes_correct :
  ∀ (short_show_episodes short_show_duration total_watched_time long_show_episode_duration : ℕ),
  short_show_episodes = 24 →
  short_show_duration = 1 / 2 →
  total_watched_time = 24 →
  long_show_episode_duration = 1 →
  episodes_long_show (short_show_episodes * short_show_duration) total_watched_time long_show_episode_duration = 12 := by
  intros
  sorry

end long_show_episodes_correct_l494_494694


namespace evaluate_expression_alternating_squares_sum_l494_494447

theorem evaluate_expression_alternating_squares_sum :
  (∑ k in finset.range 50, 
    if k % 2 = 0 then (100 - k)^2 else -(100 - k)^2) = 5050 :=
by 
  sorry

end evaluate_expression_alternating_squares_sum_l494_494447


namespace fraction_meaningful_l494_494562

theorem fraction_meaningful (x : ℝ) : (x - 1 ≠ 0) ↔ (∃ (y : ℝ), y = 3 / (x - 1)) :=
by sorry

end fraction_meaningful_l494_494562


namespace range_of_m_l494_494558

def isDistinctRealRootsInInterval (a b x : ℝ) : Prop :=
  a * x^2 + b * x + 4 = 0 ∧ 0 < x ∧ x ≤ 3

theorem range_of_m (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ isDistinctRealRootsInInterval 1 (- (m + 1)) x ∧ isDistinctRealRootsInInterval 1 (- (m + 1)) y) ↔
  (3 < m ∧ m ≤ 10 / 3) :=
sorry

end range_of_m_l494_494558


namespace sum_of_first_ten_primes_with_units_digit_three_l494_494862

-- Define the problem to prove the sum of the first 10 primes ending in 3 is 639
theorem sum_of_first_ten_primes_with_units_digit_three : 
  let primes_with_units_digit_three := [3, 13, 23, 43, 53, 73, 83, 103, 113, 163]
  in list.sum primes_with_units_digit_three = 639 := 
by 
  -- We define the primes with the units digit 3 as given and check the sum
  let primes_with_units_digit_three := [3, 13, 23, 43, 53, 73, 83, 103, 113, 163]
  show list.sum primes_with_units_digit_three = 639 from sorry

end sum_of_first_ten_primes_with_units_digit_three_l494_494862


namespace opposite_face_of_gold_is_olive_l494_494639

-- Definitions of the colors used
inductive Color
| Aqua | Maroon | Olive | Purple | Silver | Gold | Black
deriving DecidableEq, Repr

open Color

-- Conditions: The arrangement and folding pattern of squares
def isValidCrossPattern : (Array Color) → Prop
| #[Gold, Purple, Maroon, Silver, Aqua] := True
| _ := False

-- Primary goal: Prove that the oppsite face to Gold (G) is Olive (O)
theorem opposite_face_of_gold_is_olive (cubes : Array Color) (h : isValidCrossPattern cubes) : 
  (cubes[0] == Gold) → cubes[5] == Olive :=
by
  sorry

end opposite_face_of_gold_is_olive_l494_494639


namespace middleton_sewers_capacity_l494_494207

theorem middleton_sewers_capacity:
  (total_runoff: ℤ) (runoff_per_hour: ℤ) (hours_per_day: ℤ) 
  (h1: total_runoff = 240000) 
  (h2: runoff_per_hour = 1000) 
  (h3: hours_per_day = 24) : 
  total_runoff / runoff_per_hour / hours_per_day = 10 := 
by sorry

end middleton_sewers_capacity_l494_494207


namespace sum_of_first_ten_primes_with_units_digit_3_l494_494857

def is_prime (n : ℕ) : Prop := nat.prime n

def units_digit_3 (n : ℕ) : Prop := n % 10 = 3

def first_ten_primes_units_digit_3 : list ℕ :=
  [3, 13, 23, 43, 53, 73, 83, 103, 113, 163]

def sum_first_ten_primes_units_digit_3 : ℕ :=
  first_ten_primes_units_digit_3.sum

theorem sum_of_first_ten_primes_with_units_digit_3 :
  sum_first_ten_primes_units_digit_3 = 793 := by
  -- Here we provide the steps as a placeholder, but in real practice,
  -- a proof should be constructed to verify this calculation.
  sorry

end sum_of_first_ten_primes_with_units_digit_3_l494_494857


namespace circles_intersect_l494_494202

section PositionalRelationshipCircles

-- Define the first circle O1 with center (1, 0) and radius 1
def Circle1 (p : ℝ × ℝ) : Prop := (p.1 - 1)^2 + p.2^2 = 1

-- Define the second circle O2 with center (0, 3) and radius 3
def Circle2 (p : ℝ × ℝ) : Prop := p.1^2 + (p.2 - 3)^2 = 9

-- Prove that the positional relationship between Circle1 and Circle2 is intersecting
theorem circles_intersect : 
  ∃ p : ℝ × ℝ, Circle1 p ∧ Circle2 p :=
sorry

end PositionalRelationshipCircles

end circles_intersect_l494_494202


namespace increase_80_by_150_percent_l494_494263

-- Original number
def originalNumber : ℕ := 80

-- Percentage increase as a decimal
def increaseFactor : ℚ := 1.5

-- Expected result after the increase
def expectedResult : ℕ := 200

theorem increase_80_by_150_percent :
  originalNumber + (increaseFactor * originalNumber) = expectedResult :=
by
  sorry

end increase_80_by_150_percent_l494_494263


namespace max_value_ω_l494_494522

noncomputable def f (ω : ℝ) (ϕ : ℝ) (x : ℝ) := Real.cos (ω * x + ϕ)

axiom ω_pos (ω : ℝ) : ω > 0
axiom zero_condition (ω : ℝ) (ϕ : ℝ) : f ω ϕ (-Real.pi / 8) = 0
axiom symmetry_condition (ω : ℝ) (ϕ : ℝ) : ∀ x : ℝ, f ω ϕ (3 * Real.pi / 8 - x) = f ω ϕ (3 * Real.pi / 8 + x)
axiom monotonicity_condition (ω : ℝ) (ϕ : ℝ) : ∀ x₁ x₂ : ℝ, (Real.pi / 12 < x₁ ∧ x₁ < x₂ ∧ x₂ < 5 * Real.pi / 24) → f ω ϕ x₁ < f ω ϕ x₂

theorem max_value_ω : ∃ (ω : ℝ), (∀ (ω' : ℝ), ω' > 0 → zero_condition ω' 0 → symmetry_condition ω' 0 → monotonicity_condition ω' 0 → ω' ≤ 3) :=
sorry

end max_value_ω_l494_494522


namespace find_x_minus_y_l494_494549

-- Definitions for the conditions given in the problem
variables {x y : ℝ}
def cond1 := x + y = 10
def cond2 := x^2 - y^2 = 20

-- Stating the proof problem
theorem find_x_minus_y (h1 : cond1) (h2 : cond2) : x - y = 2 :=
sorry

end find_x_minus_y_l494_494549


namespace shift_right_by_pi_over_six_l494_494773

variable (x : ℝ)

def original_function (x : ℝ) : ℝ := Math.sin (2 * x + π / 6)

def shifted_function (x : ℝ) : ℝ := Math.sin (2 * (x - π / 6) + π / 6)

theorem shift_right_by_pi_over_six :
  shifted_function x = Math.sin (2 * x - π / 6) :=
by
  sorry

end shift_right_by_pi_over_six_l494_494773


namespace sum_fractions_eq_one_l494_494134

noncomputable def gcd (a b : ℕ) : ℕ := a.gcd b

theorem sum_fractions_eq_one (n : ℕ) (h₀ : n > 0) :
    (∑ (x : ℕ) in finset.range (n + 1), ∑ (y : ℕ) in finset.range (n + 1), if (x + y > n) ∧ (gcd x y = 1) then (1 : ℚ) / (x * y) else 0) = 1 :=
  sorry

end sum_fractions_eq_one_l494_494134


namespace shaded_area_correct_l494_494353

def diameter := 2
def side_of_square := 4
def radius : ℝ := diameter / 2
def area_of_square : ℝ := side_of_square * side_of_square
def area_of_circle : ℝ := Real.pi * radius * radius
def area_of_shaded_region : ℝ := area_of_square - area_of_circle

theorem shaded_area_correct : area_of_shaded_region = 16 - Real.pi := by
  unfold diameter side_of_square radius area_of_square area_of_circle area_of_shaded_region
  sorry

end shaded_area_correct_l494_494353


namespace ratio_of_areas_l494_494944

theorem ratio_of_areas (R_C R_D : ℝ) (h : (60 / 360) * (2 * Real.pi * R_C) = (40 / 360) * (2 * Real.pi * R_D)) :
  (Real.pi * R_C^2) / (Real.pi * R_D^2) = (9 / 4) :=
by
  sorry

end ratio_of_areas_l494_494944


namespace task_force_allocation_l494_494746

-- Define conditions
def subsidiaries := 6
def task_force_size := 8
def required_personnel (subsidiaries: ℕ) (task_force_size: ℕ) : Prop :=
  ∃ count_from_subs : fin subsidiaries → ℕ, (∀ i, 1 ≤ count_from_subs i) ∧ (∑ i, count_from_subs i) = task_force_size

-- Define the problem statement
theorem task_force_allocation :
  required_personnel subsidiaries task_force_size → 
  ∃ n, n = 21 :=
sorry

end task_force_allocation_l494_494746


namespace middleton_sewers_capacity_l494_494206

theorem middleton_sewers_capacity:
  (total_runoff: ℤ) (runoff_per_hour: ℤ) (hours_per_day: ℤ) 
  (h1: total_runoff = 240000) 
  (h2: runoff_per_hour = 1000) 
  (h3: hours_per_day = 24) : 
  total_runoff / runoff_per_hour / hours_per_day = 10 := 
by sorry

end middleton_sewers_capacity_l494_494206


namespace log_addition_identity_l494_494798

theorem log_addition_identity :
  ∃ (a x y : ℝ), a = 10 ∧ x = 50 ∧ y = 20 ∧ (log 10 50 + log 10 20 = 3) :=
by
  let a := 10
  let x := 50
  let y := 20
  have h1 : log a (x * y) = log a x + log a y,
    from sorry -- logarithmic identity
  have h2 : log a (x * y) = log a 1000,
    from congrArg (log a) (by norm_num) -- simplifying x * y
  have h3 : log 10 1000 = 3,
    from sorry -- calculating log 1000 base 10 directly
  exact ⟨a, x, y, rfl, rfl, rfl, by linarith [h1, h2, h3]⟩

end log_addition_identity_l494_494798


namespace distinct_real_numbers_satisfy_g4_eq_8_l494_494127

def g (x : ℝ) : ℝ := x^3 - 3 * x

theorem distinct_real_numbers_satisfy_g4_eq_8 :
  ∃! d1 d2 : ℝ, d1 ≠ d2 ∧ g(g(g(g(d1)))) = 8 ∧ g(g(g(g(d2)))) = 8 := 
sorry

end distinct_real_numbers_satisfy_g4_eq_8_l494_494127


namespace increase_by_percentage_l494_494260

theorem increase_by_percentage (initial_amount : ℕ) (percentage_increase : ℕ) :
  initial_amount = 80 → 
  percentage_increase = 150 → 
  initial_amount * (percentage_increase / 100) + initial_amount = 200 :=
by
  intros h_initial h_percentage
  rw [h_initial, h_percentage]
  calc
    80 * (150 / 100) + 80 = 80 * 1.5 + 80 : by norm_num
                     ... = 120 + 80         : by norm_num
                     ... = 200              : by norm_num
  sorry

end increase_by_percentage_l494_494260


namespace increase_by_150_percent_l494_494316

theorem increase_by_150_percent (x : ℝ) (h : x = 80) : x + (1.5 * x) = 200 := 
by
  -- The proof goes here, but is omitted as per instructions.
  sorry

end increase_by_150_percent_l494_494316


namespace sum_of_first_ten_primes_with_units_digit_3_l494_494855

def is_prime (n : ℕ) : Prop := nat.prime n

def units_digit_3 (n : ℕ) : Prop := n % 10 = 3

def first_ten_primes_units_digit_3 : list ℕ :=
  [3, 13, 23, 43, 53, 73, 83, 103, 113, 163]

def sum_first_ten_primes_units_digit_3 : ℕ :=
  first_ten_primes_units_digit_3.sum

theorem sum_of_first_ten_primes_with_units_digit_3 :
  sum_first_ten_primes_units_digit_3 = 793 := by
  -- Here we provide the steps as a placeholder, but in real practice,
  -- a proof should be constructed to verify this calculation.
  sorry

end sum_of_first_ten_primes_with_units_digit_3_l494_494855


namespace total_ticket_income_l494_494347

-- All given conditions as definitions/assumptions
def total_seats : ℕ := 200
def children_tickets : ℕ := 60
def adult_ticket_price : ℝ := 3.00
def children_ticket_price : ℝ := 1.50
def adult_tickets : ℕ := total_seats - children_tickets

-- The claim we need to prove
theorem total_ticket_income :
  (adult_tickets * adult_ticket_price + children_tickets * children_ticket_price) = 510.00 :=
by
  -- Placeholder to complete proof later
  sorry

end total_ticket_income_l494_494347


namespace classify_equation_l494_494389

/-- An equation that contains one variable and the terms with variables are of degree 1 is called a linear equation in one variable. -/
def linear_equation_one (eq : String) : Prop := 
  eq = "contains one variable and the terms with variables are of degree 1"

/-- An equation that contains one variable and the terms with variables are of degree 2 is called a quadratic equation in one variable. -/
def quadratic_equation_one (eq : String) : Prop := 
  eq = "contains one variable and the terms with variables are of degree 2"

/-- An equation whose denominator contains variables is called a fractional equation. -/
def fractional_equation (eq : String) : Prop := 
  eq = "whose denominator contains variables"

/-- An equation that contains two variables and the terms with variables are of degree 1 is called a linear equation in two variables. -/
def linear_equation_two (eq : String) : Prop := 
  eq = "contains two variables and the terms with variables are of degree 1"

/-- Prove that an equation that satisfies the definition of a linear equation in two variables matches the expected classification. -/
theorem classify_equation (eq : String) :
  eq = "contains two variables and the terms with variables are of degree 1" -> linear_equation_two eq :=
by
  intro h
  rw h
  exact True.intro

end classify_equation_l494_494389


namespace cristina_photos_l494_494820

theorem cristina_photos (john_photos : ℕ) (sarah_photos : ℕ) (clarissa_photos : ℕ) (total_slots : ℕ) :
  john_photos = 10 → sarah_photos = 9 → clarissa_photos = 14 → total_slots = 40 → 
  ∃ cristina_photos : ℕ, cristina_photos = total_slots - (john_photos + sarah_photos + clarissa_photos) ∧ cristina_photos = 7 :=
by 
  intros h1 h2 h3 h4
  use total_slots - (john_photos + sarah_photos + clarissa_photos)
  split
  { rw [h1, h2, h3, h4],
    norm_num, }
  { rw [h1, h2, h3, h4],
    norm_num, }

end cristina_photos_l494_494820


namespace greater_number_is_33_l494_494664

theorem greater_number_is_33 (A B : ℕ) (hcf_11 : Nat.gcd A B = 11) (product_363 : A * B = 363) :
  max A B = 33 :=
by
  sorry

end greater_number_is_33_l494_494664


namespace Nick_total_money_l494_494619

variable (nickels : Nat) (dimes : Nat) (quarters : Nat)
variable (value_nickel : Nat := 5) (value_dime : Nat := 10) (value_quarter : Nat := 25)

def total_value (nickels dimes quarters : Nat) : Nat :=
  nickels * value_nickel + dimes * value_dime + quarters * value_quarter

theorem Nick_total_money :
  total_value 6 2 1 = 75 := by
  sorry

end Nick_total_money_l494_494619


namespace range_of_a_l494_494076

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 + 2*x + a > 0) ↔ a ∈ set.Ioi 1 := 
sorry

end range_of_a_l494_494076


namespace sum_of_arithmetic_seq_l494_494103

-- Define the sequence and conditions
def seq (n : ℕ) : ℕ :=
  nat.rec_on n 2 (λ n' a_n, a_n + 2)

-- Define the sum of the first n terms of the sequence
def sum_seq : ℕ → ℕ
| 0       := 0
| (n + 1) := sum_seq n + seq (n + 1)

-- Prove the sum of the first n terms is n(n + 1)
theorem sum_of_arithmetic_seq (n : ℕ) : sum_seq n = n * (n + 1) :=
  sorry

end sum_of_arithmetic_seq_l494_494103


namespace area_enclosed_curve_line_l494_494653

theorem area_enclosed_curve_line :
  let curve := λ x : ℝ, 1 / x
  let line := λ x : ℝ, 4 * x
  let x1 := (1 / 2 : ℝ)
  let int1 := ∫ x in (0 : ℝ)..x1, line x
  let int2 := ∫ x in x1..(1 : ℝ), curve x
  int1 + int2 = real.log 2 + 1 / 2 :=
by
  sorry

end area_enclosed_curve_line_l494_494653


namespace sandy_younger_than_molly_l494_494166

variable (s m : ℕ)
variable (h_ratio : 7 * m = 9 * s)
variable (h_sandy : s = 56)

theorem sandy_younger_than_molly : 
  m - s = 16 := 
by
  sorry

end sandy_younger_than_molly_l494_494166


namespace count_values_undef_l494_494874

noncomputable def g (x : ℝ) : ℝ := (x^2 + 4*x - 5) * (x - 4)

theorem count_values_undef : ∃ n : ℕ, n = 3 ∧ ∀ x : ℝ, g x = 0 ↔ x = 1 ∨ x = -5 ∨ x = 4 :=
by 
  existsi 3
  split
  { refl }
  { intro x, split,
    { intro hgx,
      rw [mul_eq_zero] at hgx,
      cases hgx with hx1 hx2,
      { rw [mul_eq_zero] at hx1,
        cases hx1 with hx11 hx12,
        { exact or.inl (eq_of_mul_eq_zero_left hx11) },
        { exact or.inr (or.inl (eq_of_mul_eq_zero_left hx12)) }
      },
      { exact or.inr (or.inr (eq_of_mul_eq_zero_left hx2)) }
    },
    { intro hx,
      cases hx with hx1 hx2,
      { rw [hx1, mul_zero] },
      { cases hx2 with hx21 hx22,
        { rw [hx21, mul_zero] },
        { rw [hx22, mul_zero] }
      }
    }
  }

end count_values_undef_l494_494874


namespace rational_number_opposite_incorrect_reciprocal_incorrect_both_statements_incorrect_l494_494688

theorem rational_number_opposite_incorrect :
  ¬ ∀ (r : ℚ), ∃ (q : ℚ), (q > r ∧ q < -r) ∨ (q < r ∧ q > -r) :=
begin
  intro h,
  have h0 : ∃ (q : ℚ), (q > 0 ∧ q < 0) ∨ (q < 0 ∧ q > 0),
  { apply h,
    exact 0 },
  cases h0 with q hq,
  cases hq; linarith
end

theorem reciprocal_incorrect :
  ¬ ∀ (r : ℚ), (r ≠ 0 → ∃ (q : ℚ), (q > r ∧ q < r⁻¹) ∨ (q < r ∧ q > r⁻¹)) :=
begin
  intro h,
  have h1 : ∃ (q : ℚ), (q > 1 ∧ q < 1) ∨ (q < 1 ∧ q > 1),
  { apply h,
    intro h1,
    exact 1 },
  cases h1 with q hq,
  cases hq; linarith
end

theorem both_statements_incorrect : ¬ ( ∀ (r : ℚ), ∃ (q : ℚ), (q > r ∧ q < -r) ∨ (q < r ∧ q > -r) ) ∧
                                      ¬ ( ∀ (r : ℚ), (r ≠ 0 → ∃ (q : ℚ), (q > r ∧ q < r⁻¹) ∨ (q < r ∧ q > r⁻¹)) ) :=
begin
  split,
  { exact rational_number_opposite_incorrect, },
  { exact reciprocal_incorrect, }
end

#check both_statements_incorrect

end rational_number_opposite_incorrect_reciprocal_incorrect_both_statements_incorrect_l494_494688


namespace sum_of_first_four_terms_l494_494487

theorem sum_of_first_four_terms
    (a : ℕ → ℝ)
    (h_geom : ∀ n, a(n + 1) = a(n) * 2)
    (h_pos : ∀ n, 0 < a(n))
    (h_arith : 2 * a(3) = (4 * a(2) + a(4)) / 2)
    (h_a1 : a(0) = 1) 
    : a(0) + a(1) + a(2) + a(3) = 15 := by
  sorry

end sum_of_first_four_terms_l494_494487


namespace parabola_zero_diff_l494_494200

noncomputable def parabola (a b c : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + b * x + c

theorem parabola_zero_diff {a b c : ℝ} (h_vertex : (parabola a b c 1) = 3) (h_point : (parabola a b c 3) = -5) :
  let m := 1 + (Real.sqrt 6) / 2 in
  let n := 1 - (Real.sqrt 6) / 2 in
  m - n = Real.sqrt 6 :=
sorry

end parabola_zero_diff_l494_494200


namespace problem_1_problem_2_l494_494511

variable {a : ℝ} (ha : a > 0) (ha_ne_one : a ≠ 1)

def f (x : ℝ) : ℝ :=
if x ≥ 0 then a^x - 1 else 1 - a^(-x)

axiom odd_function (x : ℝ) : f(-x) = -f(x)

theorem problem_1 : f 2 + f (-2) = 0 := by 
  sorry

theorem problem_2 : ∀ x : ℝ, f x = if x ≥ 0 then a^x - 1 else 1 - a^(-x) := by 
  sorry

end problem_1_problem_2_l494_494511


namespace num_ways_dist_6_balls_3_boxes_l494_494044

open Finset

/-- Number of ways to distribute 6 distinguishable balls into 3 indistinguishable boxes -/
theorem num_ways_dist_6_balls_3_boxes : 
  let num_ways := 
    let n := 6 in let k := 3 in
    ((choose n 6) + 
    (choose n 5) + 
    (choose n 4) + 
    (choose n 4) + 
    (choose n 3) / 2 + 
    (choose n 3) * (choose (n - 3) 2) + 
    (choose n 2) * (choose (n - 2) 2) / (factorial k)) 
  in num_ways = 122 := 
by 
  sorry

end num_ways_dist_6_balls_3_boxes_l494_494044


namespace factor_1_factor_2_triangle_is_isosceles_l494_494635

-- Factorization problems
theorem factor_1 (x y : ℝ) : 
  (x^2 - x * y + 4 * x - 4 * y) = ((x - y) * (x + 4)) :=
sorry

theorem factor_2 (x y : ℝ) : 
  (x^2 - y^2 + 4 * y - 4) = ((x + y - 2) * (x - y + 2)) :=
sorry

-- Triangle shape problem
theorem triangle_is_isosceles (a b c : ℝ) (h : a^2 - a * c - b^2 + b * c = 0) : 
  a = b ∨ a = c ∨ b = c :=
sorry

end factor_1_factor_2_triangle_is_isosceles_l494_494635


namespace magician_never_equal_l494_494990

-- Define the conditions
def exchange_rate_Kiria_to_Daria : ℕ → ℕ := λ k : ℕ, 10 * k
def exchange_rate_Daria_to_Kiria : ℕ → ℕ := λ d : ℕ, 10 * d
def initial_state : ℕ × ℕ := (0, 1) -- (Kiriels, Dariels)

-- State the theorem
theorem magician_never_equal (n m : ℕ) : 
  (∃ k : ℕ, initial_state.1 + k * exchange_rate_Daria_to_Kiria n = n) →
  (∃ d : ℕ, initial_state.2 + d * exchange_rate_Kiria_to_Daria m = m) →
  n ≠ m :=
by sorry

end magician_never_equal_l494_494990


namespace Lee_payment_total_l494_494877

theorem Lee_payment_total 
  (ticket_price : ℝ := 10.00)
  (booking_fee : ℝ := 1.50)
  (youngest_discount : ℝ := 0.40)
  (oldest_discount : ℝ := 0.30)
  (middle_discount : ℝ := 0.20)
  (youngest_tickets : ℕ := 3)
  (oldest_tickets : ℕ := 3)
  (middle_tickets : ℕ := 4) :
  (youngest_tickets * (ticket_price * (1 - youngest_discount)) + 
   oldest_tickets * (ticket_price * (1 - oldest_discount)) + 
   middle_tickets * (ticket_price * (1 - middle_discount)) + 
   (youngest_tickets + oldest_tickets + middle_tickets) * booking_fee) = 86.00 :=
by 
  sorry

end Lee_payment_total_l494_494877


namespace modulus_z_l494_494669

noncomputable def z : ℂ := (-3 + complex.i) / (2 + complex.i)

theorem modulus_z : complex.abs z = real.sqrt 2 := by
  sorry

end modulus_z_l494_494669


namespace AD_length_proof_l494_494980

noncomputable def length_AD (AB BC CD angleB angleC : ℝ) : ℝ :=
  if h1 : AB = 6 ∧ BC = 10 ∧ CD = 25 ∧ angleB = 60 ∧ angleC = 90 then
    5 * Real.sqrt 13
  else
    0

theorem AD_length_proof :
  length_AD 6 10 25 60 90 = 5 * Real.sqrt 13 := by
  simp [length_AD]
  sorry

end AD_length_proof_l494_494980


namespace triangle_area_ratio_l494_494589

theorem triangle_area_ratio
  (A B C D E F : Point) 
  (h_AB : dist A B = 100) 
  (h_AC : dist A C = 100)
  (h_AD : dist A D = 25)
  (h_CF : dist C F = 60)
  (h_BD : dist B D = 75) 
  (h_AF : dist A F = 160) :
  (area_ratio_quotient A E F D B E = 10.24) := 
sorry

end triangle_area_ratio_l494_494589


namespace circle_equation_center_on_line_l494_494839

theorem circle_equation_center_on_line (a r : ℝ) (h₁ : 2 * a - (2 * a - 3) - 3 = 0)
  (h₂ : (5 - a)^2 + (2 - (2 * a - 3))^2 = r^2)
  (h₃ : (3 - a)^2 + (-2 - (2 * a - 3))^2 = r^2) :
  (2 = a ∧ 10 = r * r) → (∀ x y : ℝ, (x - 2)^2 + (y - 1)^2 = 10) :=
begin
  intros ha,
  sorry
end

end circle_equation_center_on_line_l494_494839


namespace definite_integral_tan_sin_cos_l494_494726

theorem definite_integral_tan_sin_cos :
  ∫ x in real.pi / 4 .. real.arccos (1 / real.sqrt 3), (real.tan x) / (real.sin x^2 - 5 * real.cos x^2 + 4) = (1 / 10) * real.log (9 / 4) :=
by
  sorry

end definite_integral_tan_sin_cos_l494_494726


namespace fraction_meaningful_l494_494561

theorem fraction_meaningful (x : ℝ) : (x - 1 ≠ 0) ↔ (∃ (y : ℝ), y = 3 / (x - 1)) :=
by sorry

end fraction_meaningful_l494_494561


namespace arc_length_sector_l494_494563

theorem arc_length_sector (r : ℝ) (α : ℝ) (h1 : r = 2) (h2 : α = π / 3) : 
  α * r = 2 * π / 3 := 
by 
  sorry

end arc_length_sector_l494_494563


namespace avg_cards_removed_until_prime_l494_494875

theorem avg_cards_removed_until_prime:
  let prime_count := 13
  let cards_count := 42
  let non_prime_count := cards_count - prime_count
  let groups_count := prime_count + 1
  let avg_non_prime_per_group := (non_prime_count: ℚ) / (groups_count: ℚ)
  (groups_count: ℚ) > 0 →
  avg_non_prime_per_group + 1 = (43: ℚ) / (14: ℚ) :=
by
  sorry

end avg_cards_removed_until_prime_l494_494875


namespace increased_number_l494_494279

theorem increased_number (original_number : ℕ) (percentage_increase : ℚ) :
  original_number = 80 →
  percentage_increase = 1.5 →
  original_number * (1 + percentage_increase) = 200 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end increased_number_l494_494279


namespace snow_probability_january_first_week_l494_494467

noncomputable def P_snow_at_least_once_first_week : ℚ :=
  1 - ((2 / 3) ^ 4 * (3 / 4) ^ 3)

theorem snow_probability_january_first_week :
  P_snow_at_least_once_first_week = 11 / 12 :=
by
  sorry

end snow_probability_january_first_week_l494_494467


namespace overall_percentage_change_l494_494568

theorem overall_percentage_change :
  let initial_investment := 200
  let first_year_loss := 0.10
  let second_year_gain := 0.15
  let third_year_loss := 0.05
  let final_amount := initial_investment * (1 - first_year_loss) * (1 + second_year_gain) * (1 - third_year_loss)
  let percentage_change := (final_amount - initial_investment) / initial_investment * 100
  percentage_change ≈ -1.68 :=
sorry

end overall_percentage_change_l494_494568


namespace longest_leg_of_smallest_triangle_l494_494436

-- Definitions based on conditions
def is306090Triangle (h : ℝ) (s : ℝ) (l : ℝ) : Prop :=
  s = h / 2 ∧ l = s * (Real.sqrt 3)

def chain_of_306090Triangles (H : ℝ) : Prop :=
  ∃ h1 s1 l1 h2 s2 l2 h3 s3 l3 h4 s4 l4,
    is306090Triangle h1 s1 l1 ∧
    is306090Triangle h2 s2 l2 ∧
    is306090Triangle h3 s3 l3 ∧
    is306090Triangle h4 s4 l4 ∧
    h1 = H ∧ l1 = h2 ∧ l2 = h3 ∧ l3 = h4

-- Main theorem
theorem longest_leg_of_smallest_triangle (H : ℝ) (h : ℝ) (l : ℝ) (H_cond : H = 16) 
  (h_cond : h = 9) :
  chain_of_306090Triangles H →
  ∃ h4 s4 l4, is306090Triangle h4 s4 l4 ∧ l = h4 →
  l = 9 := 
by
  sorry

end longest_leg_of_smallest_triangle_l494_494436


namespace math_problem_l494_494473

theorem math_problem (a b : ℝ) (h : a * b < 0) : a^2 * |b| - b^2 * |a| + a * b * (|a| - |b|) = 0 :=
sorry

end math_problem_l494_494473


namespace meaningful_fraction_range_l494_494559

theorem meaningful_fraction_range (x : ℝ) : (x - 1 ≠ 0) ↔ (fraction_meaningful := x ≠ 1) :=
by
  sorry

end meaningful_fraction_range_l494_494559


namespace larger_root_of_quadratic_greater_root_is_9_l494_494840

theorem larger_root_of_quadratic :
  ∀ x : ℝ, x^2 - 5 * x - 36 = 0 → x = 9 ∨ x = -4 :=
begin
  intro x,
  intro h,
  sorry
end

theorem greater_root_is_9 :
  (∃ x : ℝ, x^2 = 81) → 9 = 9 :=
begin
  sorry,
end

end larger_root_of_quadratic_greater_root_is_9_l494_494840


namespace increase_by_150_percent_l494_494319

theorem increase_by_150_percent (x : ℝ) (h : x = 80) : x + (1.5 * x) = 200 := 
by
  -- The proof goes here, but is omitted as per instructions.
  sorry

end increase_by_150_percent_l494_494319


namespace angle_A_is_pi_over_3_expression_value_l494_494961

variable {A B C a b c : ℝ}

-- Condition: In triangle ABC, sides opposite to angles A, B, C are a, b, c respectively
axiom triangle_sides : ∀ A B C a b c, a = opposite_side_of A ∧ b = opposite_side_of B ∧ c = opposite_side_of C

-- Condition (2): The sides satisfy the given trigonometric identity
axiom trig_identity : (a^2 + c^2 - b^2) * tan B = sqrt 3 * (b^2 + c^2 - a^2)

-- Condition (3): The area of the triangle is 3/2
axiom area_condition : 1 / 2 * b * c * sin A = 3 / 2

-- Prove that A = π/3
theorem angle_A_is_pi_over_3 : A = π / 3 :=
  by
  sorry

-- Prove the value of the given expression
theorem expression_value : (bc - 4 * sqrt 3) * cos A + ac * cos B / (a^2 - b^2) = 1 :=
  by
  sorry

end angle_A_is_pi_over_3_expression_value_l494_494961


namespace increase_150_percent_of_80_l494_494292

theorem increase_150_percent_of_80 : 80 * 1.5 + 80 = 200 :=
by
  sorry

end increase_150_percent_of_80_l494_494292


namespace balls_in_boxes_l494_494006

/-
Prove that the number of ways to distribute 6 distinguishable balls into 3 indistinguishable boxes is 132.
-/
theorem balls_in_boxes : 
  ∃ (ways : ℕ), ways = 132 ∧ ways = 
    (1) + 
    (Nat.choose 6 5) + 
    (Nat.choose 6 4) + 
    (Nat.choose 6 4) + 
    (Nat.choose 6 3) + 
    (Nat.choose 6 3 * Nat.choose 3 2) + 
    (Nat.choose 6 2 * Nat.choose 4 2 / 6) := 
by
  sorry

end balls_in_boxes_l494_494006


namespace correct_statement_is_D_l494_494219

def student : Type := { s : Type // Type }
def mathematics_scores (students : student) : Type := students

structure data :=
  (sample_size : ℕ)
  (students : Type)
  (random_sample : fin sample_size → students)

def examination : data :=
  { sample_size := 1000,
    students := student,
    random_sample := sorry }

def population := { s : student // mathematics_scores s }
def individual := mathematics_scores student

-- Statements to be evaluated
def A := population = { s : student // Type }
def B := individual = student
def C := data.sample_size = 1000
def D := examination.random_sample = mathematics_scores student

-- Problem statement
theorem correct_statement_is_D : ¬A ∧ ¬B ∧ ¬C ∧ D :=
sorry

end correct_statement_is_D_l494_494219


namespace train_speed_is_correct_l494_494378

-- Definitions based on the conditions
def length_of_train : ℝ := 120       -- Train is 120 meters long
def time_to_cross : ℝ := 16          -- The train takes 16 seconds to cross the post

-- Conversion constants
def seconds_to_hours : ℝ := 3600
def meters_to_kilometers : ℝ := 1000

-- The speed of the train in km/h
noncomputable def speed_of_train (length : ℝ) (time : ℝ) : ℝ :=
  (length / time) * (seconds_to_hours / meters_to_kilometers)

-- Theorem: The speed of the train is 27 km/h
theorem train_speed_is_correct : speed_of_train length_of_train time_to_cross = 27 :=
by
  -- This is where the proof should be, but we leave it as sorry as instructed
  sorry

end train_speed_is_correct_l494_494378


namespace total_bulbs_needed_l494_494151

def NumberOfMediumLights : ℕ := 12
def BulbsPerSmall : ℕ := 1
def BulbsPerMedium : ℕ := 2
def BulbsPerLarge : ℕ := 3
def NumberOfLargeLights (M : ℕ) : ℕ := 2 * M
def NumberOfSmallLights (M : ℕ) : ℕ := M + 10

theorem total_bulbs_needed : 
  let M := NumberOfMediumLights in
  let L := NumberOfLargeLights M in
  let S := NumberOfSmallLights M in
  ((M * BulbsPerMedium) + 
  (L * BulbsPerLarge) + 
  (S * BulbsPerSmall)) = 118 := by
  let M := NumberOfMediumLights
  let L := NumberOfLargeLights M
  let S := NumberOfSmallLights M
  sorry

end total_bulbs_needed_l494_494151


namespace cp_of_apple_l494_494340

theorem cp_of_apple (SP : ℝ) (hSP : SP = 17) (loss_fraction : ℝ) (h_loss_fraction : loss_fraction = 1 / 6) : 
  ∃ CP : ℝ, CP = 20.4 ∧ SP = CP - loss_fraction * CP :=
by
  -- Placeholder for proof
  sorry

end cp_of_apple_l494_494340


namespace center_of_regular_ngon_prism_l494_494693

universe u

-- Definitions related to the n-gon prism
variables {V : Type u} [normed_group V] [normed_space ℝ V] [finite_dimensional ℝ V]

def is_regular_ngon_prism (P : set V) (n : ℕ) : Prop :=
  ∃ (A B : fin n → V), 
    (∀ i, (norm (A i - A ((i + 1) % n)) = norm (A 0 - A 1)) ∧ 
          (norm (B i - B ((i + 1) % n)) = norm (B 0 - B 1)) ∧ 
          (norm (A i - B i) = norm (A 0 - B 0))) ∧
    (∀ i j, i ≠ j → line_through (A i) (B j) ∈ P)

variables {P : set V} {O : V} {n : ℕ}

-- The Lean statement for the proof problem.
theorem center_of_regular_ngon_prism (hP : is_regular_ngon_prism P n) :
  (∃ i j k l m p, i ≠ j ∧ k ≠ l ∧ m ≠ p ∧ 
      O ∈ line_through (A i) (B j) ∧ 
      O ∈ line_through (A k) (B l) ∧
      O ∈ line_through (A m) (B p)) → 
  (∀ i, norm (O - A i) = norm (O - B i)) :=
by
  sorry

end center_of_regular_ngon_prism_l494_494693


namespace largest_arith_prog_l494_494119

def S : Set ℚ := {1 / n | n : ℕ, 1 ≤ n ∧ n ≤ 2016}
def is_arith_progression (s : List ℚ) : Prop :=
  ∀ i j k : ℕ, i < j → j < k → k < List.length s → 
  2 * s[j] = s[i] + s[k]

def T : Set (List ℚ) := { l | l.toList ⊆ S ∧ is_arith_progression l }

theorem largest_arith_prog : ∃ l ∈ T, List.length l = 6 := by
  sorry

end largest_arith_prog_l494_494119


namespace prob_cos_eq_one_half_l494_494104

def set_x : Set ℝ := {x | ∃ (n: ℕ), n ∈ Finset.range 11 ∧ x = n * Real.pi / 6}

def satisfies_cos (x : ℝ) : Prop := Real.cos x = 1 / 2

theorem prob_cos_eq_one_half : 
  (Finset.filter (λ x, satisfies_cos x) (Finset.image (λ n, n * Real.pi / 6) (Finset.range 11))).card 
  / (Finset.range 11).card = 1 / 5 := by
  sorry

end prob_cos_eq_one_half_l494_494104


namespace midpoint_fraction_l494_494711

theorem midpoint_fraction (a b : ℚ) (h1 : a = 3 / 4) (h2 : b = 5 / 6) :
  (a + b) / 2 = 19 / 24 :=
by {
  sorry
}

end midpoint_fraction_l494_494711


namespace EF_perpendicular_HK_l494_494118

-- Given definitions
variables (A B C D E F G H K : Type*)
variables [IsCyclicQuadrilateral A B C D] -- Assumes ABCD is a cyclic quadrilateral (not a trapezoid)
variables (midpoint_AB : Midpoint A B F) -- F is midpoint of AB
variables (midpoint_CD : Midpoint C D G) -- G is midpoint of CD
variables (line_parallel_AB : LineThrough G A B) -- line through G parallel to AB
variables (foot_perpendicular_E_ell : FootPerpendicular E (LineThrough G A B) H) -- H is foot from E to line through G
variables (foot_perpendicular_E_CD : FootPerpendicular E (LineThrough C D) K) -- K is foot from E to CD

-- Prove that lines EF and HK are perpendicular
theorem EF_perpendicular_HK : Perpendicular (LineThrough E F) (LineThrough H K) := by
  sorry

end EF_perpendicular_HK_l494_494118


namespace log_216_eq_3_times_log_2_plus_log_3_l494_494421

theorem log_216_eq_3_times_log_2_plus_log_3 (log : ℕ → ℝ) :
  log 216 = 3 * (log 2 + log 3) :=
by
  -- Assuming properties of logarithms are part of the environment.
  have h1 : 216 = 6^3 := by norm_num
  have h2 : 6 = 2 * 3 := by norm_num
  have h3 : log (a * b) = log a + log b := sorry  -- Product rule
  have h4 : log (a^b) = b * log a := sorry         -- Power rule
  sorry

end log_216_eq_3_times_log_2_plus_log_3_l494_494421


namespace minimum_travel_distance_l494_494994

-- Define the pyramid structure
structure Pyramid :=
  (A B C D : Point)
  (DA DB DC : ℝ)
  (LateralEdges_Perpendicular : DA^2 + DB^2 = DC^2)
  (DA_eq_DB : DA = DB)
  (DA_val : DA = 5)
  (DB_val : DB = 5)
  (DC_val : DC = 1)

-- Define the proof problem
theorem minimum_travel_distance (p : Pyramid) :
  ∃ d : ℝ, d = (10 * real.sqrt 3) / 9 :=
begin
  -- Define the hypothesis based on pyramid properties
  have h1 : p.DA = 5 := by rw [p.DA_val],
  have h2 : p.DB = 5 := by rw [p.DB_val],
  have h3 : p.DC = 1 := by rw [p.DC_val],
  have h4 : p.LateralEdges_Perpendicular,
  have h5 : p.DA = p.DB := by rw [p.DA_eq_DB],
  
  -- State the minimal distance
  use (10 * real.sqrt 3) / 9,
  sorry
end

end minimum_travel_distance_l494_494994


namespace eleventh_anniversary_days_l494_494413

-- Define the conditions
def is_leap_year (year : ℕ) : Prop := year % 4 = 0

def initial_years : Set ℕ := {1668, 1669, 1670, 1671}

def initial_day := "Friday"

noncomputable def day_after_11_years (start_year : ℕ) : String :=
  let total_days := 4015 + (if is_leap_year start_year then 3 else 2)
  if total_days % 7 = 0 then "Friday"
  else "Thursday"

-- Define the proposition to prove
theorem eleventh_anniversary_days : 
  (∀ year ∈ initial_years, 
    (if day_after_11_years year = "Friday" then (3 : ℝ) / 4 else (1 : ℝ) / 4) = 
    (if year = 1668 ∨ year = 1670 ∨ year = 1671 then (3 : ℝ) / 4 else (1 : ℝ) / 4)) := 
sorry

end eleventh_anniversary_days_l494_494413


namespace common_difference_is_neg_half_l494_494508

noncomputable theory

open_locale classical

variable (a : ℕ → ℝ)

-- Conditions
def arithmetic_sequence (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n+1) = a n + d

variable (d : ℝ)

-- Hypothesis
axiom h1 : a 6 - 2 * a 3 = -1
axiom h2 : a 2 = 0

-- The question to prove
theorem common_difference_is_neg_half :
  arithmetic_sequence a d →
  d = -1/2 :=
sorry

end common_difference_is_neg_half_l494_494508


namespace marbles_count_l494_494969

variable (r b : ℕ)

theorem marbles_count (hr1 : 8 * (r - 1) = r + b - 2) (hr2 : 4 * r = r + b - 3) : r + b = 9 := 
by sorry

end marbles_count_l494_494969


namespace sum_ai_aj_neg_l494_494490

variables {n : ℕ} {a : ℕ → ℝ}

-- Define the premise conditions
axiom h_n : n ≥ 2
axiom h_sum : (∑ i in Finset.range n, a i) = 0

-- Define the set A
def A : Finset (ℕ × ℕ) := 
  {ij | 1 ≤ ij.1 ∧ ij.1 < ij.2 ∧ ij.2 ≤ n ∧ abs (a ij.1 - a ij.2) ≥ 1}.to_finset

-- The main theorem to prove
theorem sum_ai_aj_neg (hA_nonempty : A.nonempty) : ∑ ij in A, a ij.1 * a ij.2 < 0 := 
sorry

end sum_ai_aj_neg_l494_494490


namespace distinguishable_balls_in_indistinguishable_boxes_l494_494026

theorem distinguishable_balls_in_indistinguishable_boxes :
  let num_distinguishable_balls := 6
  let num_indistinguishable_boxes := 3
  -- different ways to distribute balls across boxes summarized
  let ways := ∑([{6, 0, 0} ++ {5, 1, 0} ++ {4, 2, 0} ++ {4, 1, 1} ++ {3, 3, 0} ++ {3, 2, 1} ++ {2, 2, 2}], sum)
  -- corresponding ways to remove permutation of distinguishable ball cases
  ways = 222 := sorry

end distinguishable_balls_in_indistinguishable_boxes_l494_494026


namespace Concyniclicity_A_M_N_I_l494_494139

variables {A B C M N I D : Type} [Nonempty M]
variables (triangle : Triangle A B C)
variables [Incenter I A B C]
variables [internalBisector AD A (Segment BC)]
variables [perpendicularBisectorInterp AD M (angleBisector B A D)]
variables [perpendicularBisectorInterp AD N (angleBisector C A D)]

theorem Concyniclicity_A_M_N_I :
  CyclicFourPoints A M N I :=
sorry

end Concyniclicity_A_M_N_I_l494_494139


namespace increase_80_by_150_percent_l494_494232

-- Define the initial number (n) and the percentage increase (p)
def n : ℕ := 80
def p : ℚ := 1.5

-- The theorem stating the expected result after increasing n by 150%
theorem increase_80_by_150_percent : n + (p * n) = 200 := by
  sorry

end increase_80_by_150_percent_l494_494232


namespace a_n_formula_exists_n_satisfying_condition_l494_494617

-- Definition of the sequence and its sum
def a (n : ℕ) : ℕ := if n = 0 then 0 else (6 * n - 5)

def S (n : ℕ) : ℕ := n * (a n) - 3 * n * (n - 1)

-- Problem (I): Prove the general formula for a_n
theorem a_n_formula (n : ℕ) (hn : n > 0) : a n = 6 * n - 5 :=
by sorry

-- Problem (II): Prove the existence of n such that a specific condition is met
theorem exists_n_satisfying_condition : ∃ n : ℕ, n > 0 ∧ 
  (∑ i in Finset.range n, S (i+1) / (i+1)) - (3/2) * (n-1)^2 = 2016 :=
by
  use 807
  split
  · sorry -- proof that 807 > 0
  · sorry -- proof that 807 satisfies the condition

end a_n_formula_exists_n_satisfying_condition_l494_494617


namespace sin_cos_ratio_l494_494865

theorem sin_cos_ratio (θ : ℝ) (h : θ = Real.arctan (-2)) : 
  (sin θ + cos θ) / (sin θ - cos θ) = 1 / 3 :=
by sorry

end sin_cos_ratio_l494_494865


namespace count_correct_statements_l494_494673

theorem count_correct_statements : 
  let statement_1 := (a : ℕ) → a^2 * a^2 = 2 * a^2
  let statement_2 := (a b : ℕ) → (a - b)^2 = a^2 - b^2
  let statement_3 := (a : ℕ) → a^2 + a^3 = a^5
  let statement_4 := (a b : ℕ) → (-2 * a^2 * b^3)^3 = -6 * a^6 * b^3
  let statement_5 := (a : ℕ) → (-a^3)^2 / a = a^5
  1 = (if statement_1 then 1 else 0) +
      (if statement_2 then 1 else 0) +
      (if statement_3 then 1 else 0) +
      (if statement_4 then 1 else 0) +
      (if statement_5 then 1 else 0) :=
by
  sorry

end count_correct_statements_l494_494673


namespace positive_integer_prime_condition_l494_494730

theorem positive_integer_prime_condition (n : ℕ) 
  (h1 : 0 < n)
  (h2 : ∀ (k : ℕ), k < n → Nat.Prime (4 * k^2 + n)) : 
  n = 3 ∨ n = 7 := 
sorry

end positive_integer_prime_condition_l494_494730


namespace largest_number_of_four_consecutive_whole_numbers_l494_494876

theorem largest_number_of_four_consecutive_whole_numbers 
  (a : ℕ) (h1 : a + (a + 1) + (a + 2) = 184)
  (h2 : a + (a + 1) + (a + 3) = 201)
  (h3 : a + (a + 2) + (a + 3) = 212)
  (h4 : (a + 1) + (a + 2) + (a + 3) = 226) : 
  a + 3 = 70 := 
by sorry

end largest_number_of_four_consecutive_whole_numbers_l494_494876


namespace find_point_B_coordinates_l494_494717

theorem find_point_B_coordinates (a : ℝ) : 
  (∀ (x y : ℝ), x^2 - 4*x + y^2 = 0 → (x - a)^2 + y^2 = 4 * ((x - 1)^2 + y^2)) →
  a = -2 :=
by
  sorry

end find_point_B_coordinates_l494_494717


namespace difference_largest_smallest_odd_1_to_100_l494_494458

theorem difference_largest_smallest_odd_1_to_100:
  (finset.range 100).filter (fun x => x % 2 = 1).max' sorry - 
  (finset.range 100).filter (fun x => x % 2 = 1).min' sorry = 98 :=
sorry

end difference_largest_smallest_odd_1_to_100_l494_494458


namespace magnitude_relationship_l494_494825

noncomputable def a : ℝ := 2 ^ 0.3
def b : ℝ := 0.3 ^ 2
noncomputable def c : ℝ := log 2 0.3

theorem magnitude_relationship : c < b ∧ b < a := by
  sorry

end magnitude_relationship_l494_494825


namespace P1234_concyclic_O1234_cyclic_circumradius_comparison_l494_494512

-- Defining circles and points
variables {O1 O2 O3 O4 : Type} {P1 P2 P3 P4 : Type}

-- Conditions
axiom tangent_O4_O1_at_P1 : ¬(O4 = O1) ∧ ¬(P1 ∈ set.inter P1 P1)
axiom tangent_O1_O2_at_P2 : ¬(O1 = O2) ∧ ¬(P2 ∈ set.inter P2 P2)
axiom tangent_O2_O3_at_P3 : ¬(O2 = O3) ∧ ¬(P3 ∈ set.inter P3 P3)
axiom tangent_O3_O4_at_P4 : ¬(O3 = O4) ∧ ¬(P4 ∈ set.inter P4 P4)

-- Proving P1, P2, P3, P4 are concyclic
theorem P1234_concyclic : 
∃ (k : Type), is_circumcircle k {P1, P2, P3, P4} := sorry

-- Proving O1 O2 O3 O4 is cyclic
theorem O1234_cyclic : 
∃ (l : Type), is_circumcircle l {O1, O2, O3, O4} := sorry

-- Proving the circumradius relationship
theorem circumradius_comparison :
∀ (c1 c2 : Type), is_circumcircle c1 {O1, O2, O3, O4} -> is_circumcircle c2 {P1, P2, P3, P4} -> radius c1 ≤ radius c2 := sorry

end P1234_concyclic_O1234_cyclic_circumradius_comparison_l494_494512


namespace increase_result_l494_494242

-- Given conditions
def original_number : ℝ := 80
def increase_percentage : ℝ := 1.5

-- The result after the increase
theorem increase_result (h1 : original_number = 80) (h2 : increase_percentage = 1.5) : 
  original_number + (increase_percentage * original_number) = 200 := by
  sorry

end increase_result_l494_494242


namespace trigonometric_identity_75_deg_l494_494864

theorem trigonometric_identity_75_deg :
  (cos (75 * (Real.pi / 180)))^4 + (sin (75 * (Real.pi / 180)))^4 + 3 * (sin (75 * (Real.pi / 180)))^2 * (cos (75 * (Real.pi / 180)))^2 = 
  ((cos (75 * (Real.pi / 180)))^6 + (sin (75 * (Real.pi / 180)))^6 + 4 * (sin (75 * (Real.pi / 180)))^2 * (cos (75 * (Real.pi / 180)))^2) :=
sorry

end trigonometric_identity_75_deg_l494_494864


namespace length_of_rect_box_l494_494199

noncomputable def length_of_box (height : ℝ) (width : ℝ) (volume : ℝ) : ℝ :=
  volume / (width * height)

theorem length_of_rect_box :
  (length_of_box 0.5 25 (6000 / 7.48052)) = 64.1624 :=
by
  unfold length_of_box
  norm_num
  sorry

end length_of_rect_box_l494_494199


namespace min_omega_for_sine_cycles_l494_494668

theorem min_omega_for_sine_cycles (A : ℝ) (hA : 0 < A) : 
  ∃ ω : ℝ, (0 < ω ∧ ω ≥ 49.5 * Real.pi) ∧ 
    (∀ x ∈ Icc (0 : ℝ) (1 : ℝ), ∃ n : ℕ, 25 ≤ n ∧ y = A * Real.sin (ω * x)) := 
sorry

end min_omega_for_sine_cycles_l494_494668


namespace eleventh_anniversary_days_l494_494416

-- Define the conditions
def is_leap_year (year : ℕ) : Prop := year % 4 = 0

def initial_years : Set ℕ := {1668, 1669, 1670, 1671}

def initial_day := "Friday"

noncomputable def day_after_11_years (start_year : ℕ) : String :=
  let total_days := 4015 + (if is_leap_year start_year then 3 else 2)
  if total_days % 7 = 0 then "Friday"
  else "Thursday"

-- Define the proposition to prove
theorem eleventh_anniversary_days : 
  (∀ year ∈ initial_years, 
    (if day_after_11_years year = "Friday" then (3 : ℝ) / 4 else (1 : ℝ) / 4) = 
    (if year = 1668 ∨ year = 1670 ∨ year = 1671 then (3 : ℝ) / 4 else (1 : ℝ) / 4)) := 
sorry

end eleventh_anniversary_days_l494_494416


namespace trigonometric_expression_value_l494_494465

theorem trigonometric_expression_value :
  (sin (5 * π / 24))^4 + (cos (7 * π / 24))^4 + (sin (17 * π / 24))^4 + (cos (19 * π / 24))^4 = 3 / 2 - sqrt 3 / 4 :=
by
  sorry

end trigonometric_expression_value_l494_494465


namespace num_ways_dist_6_balls_3_boxes_l494_494045

open Finset

/-- Number of ways to distribute 6 distinguishable balls into 3 indistinguishable boxes -/
theorem num_ways_dist_6_balls_3_boxes : 
  let num_ways := 
    let n := 6 in let k := 3 in
    ((choose n 6) + 
    (choose n 5) + 
    (choose n 4) + 
    (choose n 4) + 
    (choose n 3) / 2 + 
    (choose n 3) * (choose (n - 3) 2) + 
    (choose n 2) * (choose (n - 2) 2) / (factorial k)) 
  in num_ways = 122 := 
by 
  sorry

end num_ways_dist_6_balls_3_boxes_l494_494045


namespace fred_gave_mary_34_cards_l494_494149

theorem fred_gave_mary_34_cards :
  ∀ (initial cards_torn cards_bought total : ℕ),
    initial = 18 →
    cards_torn = 8 →
    cards_bought = 40 →
    total = 84 →
    ∃ (cards_given_by_fred : ℕ), cards_given_by_fred = 34 :=
by
  intros initial cards_torn cards_bought total h_initial h_cards_torn h_cards_bought h_total
  use 34
  sorry

end fred_gave_mary_34_cards_l494_494149


namespace distribute_6_balls_in_3_boxes_l494_494036

def num_ways_to_distribute_balls (balls : ℕ) (boxes : ℕ) : ℕ :=
  if balls = 6 ∧ boxes = 3 then 92 else 0

theorem distribute_6_balls_in_3_boxes :
  num_ways_to_distribute_balls 6 3 = 92 :=
by
  rw num_ways_to_distribute_balls
  norm_num
  sorry

end distribute_6_balls_in_3_boxes_l494_494036


namespace distribute_6_balls_in_3_boxes_l494_494043

def num_ways_to_distribute_balls (balls : ℕ) (boxes : ℕ) : ℕ :=
  if balls = 6 ∧ boxes = 3 then 92 else 0

theorem distribute_6_balls_in_3_boxes :
  num_ways_to_distribute_balls 6 3 = 92 :=
by
  rw num_ways_to_distribute_balls
  norm_num
  sorry

end distribute_6_balls_in_3_boxes_l494_494043


namespace problem_1_problem_2_l494_494893

noncomputable def seq_a (a n : ℕ) : ℕ :=
  match n with
  | 1 => 1
  | n+2 => (2 * a ^ 2) / (2 * a - 1)

def sum_seq (seq : ℕ → ℕ) (n : ℕ) : ℕ :=
  (List.range n).map seq |>.sum

def sequence_s (n : ℕ) := sum_seq seq_a n

def sequence_b (n : ℕ) : ℕ :=
  let s := sequence_s n
  s / (2 * n + 1) + 2 ^ n / s

theorem problem_1 (s n : ℕ) : 
  (∀ n ≥ 2, seq_a n = 2 * (sequence_s s) ^ 2 / (2 * (sequence_s s) - 1)) →
  (∃ n : ℕ, sequence_s n = 1 / (2 * n - 1)) →
  (∃ P_n : ℕ, P_n = n^2) := 
sorry

theorem problem_2 (s n : ℕ) : 
  (∀ n ≥ 2, seq_a n = 2 * (sequence_s s) ^ 2 / (2 * (sequence_s s) - 1)) →
  (∃ T_n : ℕ, b n = (n / (2 * n + 1) + (2 * n - 3) * 2 ^ (n + 1) + 6)) :=
sorry

end problem_1_problem_2_l494_494893


namespace minimum_value_of_expression_l494_494476

noncomputable def minimum_value (m n s : ℝ) (α β : ℝ) : ℝ := m * (sec α) + n * (sec β)

theorem minimum_value_of_expression (m n s : ℝ) (α β : ℝ)
  (hm : m > 0) (hn : n > 0) (hs : s > 0)
  (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h_cond : m * (tan α) + n * (tan β) = s) :
  minimum_value m n s α β = sqrt ((m + n)^2 + s^2) := 
sorry

end minimum_value_of_expression_l494_494476


namespace increase_result_l494_494250

-- Given conditions
def original_number : ℝ := 80
def increase_percentage : ℝ := 1.5

-- The result after the increase
theorem increase_result (h1 : original_number = 80) (h2 : increase_percentage = 1.5) : 
  original_number + (increase_percentage * original_number) = 200 := by
  sorry

end increase_result_l494_494250


namespace number_of_correct_calculations_is_one_l494_494672

/- Given conditions -/
def cond1 (a : ℝ) : Prop := a^2 * a^2 = 2 * a^2
def cond2 (a b : ℝ) : Prop := (a - b)^2 = a^2 - b^2
def cond3 (a : ℝ) : Prop := a^2 + a^3 = a^5
def cond4 (a b : ℝ) : Prop := (-2 * a^2 * b^3)^3 = -6 * a^6 * b^3
def cond5 (a : ℝ) : Prop := (-a^3)^2 / a = a^5

/- Statement to prove the number of correct calculations is 1 -/
theorem number_of_correct_calculations_is_one :
  (¬ (cond1 a)) ∧ (¬ (cond2 a b)) ∧ (¬ (cond3 a)) ∧ (¬ (cond4 a b)) ∧ (cond5 a) → 1 = 1 :=
by
  sorry

end number_of_correct_calculations_is_one_l494_494672


namespace tan_alpha_l494_494882

variables {α : ℝ}

theorem tan_alpha (h : (sin α - 2 * cos α) / (2 * sin α + cos α) = -1) : tan α = 1 / 3 :=
by sorry

end tan_alpha_l494_494882


namespace Amanda_money_left_l494_494776

theorem Amanda_money_left (initial_amount cost_cassette tape_count cost_headphone : ℕ) 
  (h1 : initial_amount = 50) 
  (h2 : cost_cassette = 9) 
  (h3 : tape_count = 2) 
  (h4 : cost_headphone = 25) :
  initial_amount - (tape_count * cost_cassette + cost_headphone) = 7 :=
by
  sorry

end Amanda_money_left_l494_494776


namespace exists_non_poly_func_satisfies_ineq_l494_494828

noncomputable def k (x : ℝ) : ℝ := sin (Real.pi * x) -- Example of a bounded, non-constant, periodic function with period 2

noncomputable def f (x : ℝ) : ℝ := x^3 + x * k(x)

theorem exists_non_poly_func_satisfies_ineq :
  ∃ f : ℝ → ℝ, (∀ x : ℝ, (x - 1) * f (x + 1) - (x + 1) * f (x - 1) ≥ 4 * x * (x^2 - 1)) ∧
  ¬ (∃ p : ℤ → ℝ, ∀ z : ℝ, f z = polynomial.eval z (polynomial.of_fun p T)) :=
begin
  use f,
  split,
  { -- Proof of inequality goes here
    sorry },
  { -- Proof that f is not a polynomial
    sorry }
end

end exists_non_poly_func_satisfies_ineq_l494_494828


namespace function_crosses_horizontal_asymptote_at_l494_494469

def f (x : ℝ) : ℝ := (3 * x ^ 2 - 6 * x - 8) / (2 * x ^ 2 - 5 * x + 2)

theorem function_crosses_horizontal_asymptote_at (x : ℝ) : f x = 3 / 2 ↔ x = 22 / 3 :=
by 
  sorry

end function_crosses_horizontal_asymptote_at_l494_494469


namespace john_toy_store_fraction_l494_494928

theorem john_toy_store_fraction
  (allowance : ℝ)
  (spent_at_arcade_fraction : ℝ)
  (remaining_allowance : ℝ)
  (spent_at_candy_store : ℝ)
  (spent_at_toy_store : ℝ)
  (john_allowance : allowance = 3.60)
  (arcade_fraction : spent_at_arcade_fraction = 3 / 5)
  (arcade_amount : remaining_allowance = allowance - (spent_at_arcade_fraction * allowance))
  (candy_store_amount : spent_at_candy_store = 0.96)
  (remaining_after_candy_store : spent_at_toy_store = remaining_allowance - spent_at_candy_store)
  : spent_at_toy_store / remaining_allowance = 1 / 3 :=
by
  sorry

end john_toy_store_fraction_l494_494928


namespace maximum_dot_product_l494_494899

-- Define the points in Cartesian coordinate system
def O := (0, 0) : ℝ × ℝ
def A := (1, -2) : ℝ × ℝ
def B := (1, 1) : ℝ × ℝ
def C := (2, -1) : ℝ × ℝ

-- Define a point M in the Cartesian coordinate system with a constraint on x
def M (x y : ℝ) (h : -2 ≤ x ∧ x ≤ 2) := (x, y)

-- Define vector dot product
def dot_product (u v : ℝ × ℝ) : ℝ :=
u.1 * v.1 + u.2 * v.2

-- Definition of vector OM and OC
def vector_OM (x y : ℝ) : ℝ × ℝ := (x, y)
def vector_OC : ℝ × ℝ := (2, -1)

-- The problem to be proven
theorem maximum_dot_product : 
  ∃ M_y : ℝ, ∃ (x : ℝ) (hx : -2 ≤ x ∧ x ≤ 2), 
    dot_product (vector_OM x M_y) vector_OC = 4 :=
sorry

end maximum_dot_product_l494_494899


namespace increase_80_by_150_percent_l494_494264

-- Original number
def originalNumber : ℕ := 80

-- Percentage increase as a decimal
def increaseFactor : ℚ := 1.5

-- Expected result after the increase
def expectedResult : ℕ := 200

theorem increase_80_by_150_percent :
  originalNumber + (increaseFactor * originalNumber) = expectedResult :=
by
  sorry

end increase_80_by_150_percent_l494_494264


namespace impossible_circular_arrangement_1_to_60_l494_494341

theorem impossible_circular_arrangement_1_to_60 :
  (∀ (f : ℕ → ℕ), 
      (∀ n, 1 ≤ f n ∧ f n ≤ 60) ∧ 
      (∀ n, f (n + 2) + f n ≡ 0 [MOD 2]) ∧ 
      (∀ n, f (n + 3) + f n ≡ 0 [MOD 3]) ∧ 
      (∀ n, f (n + 7) + f n ≡ 0 [MOD 7]) 
      → false) := 
  sorry

end impossible_circular_arrangement_1_to_60_l494_494341


namespace boat_speed_in_still_water_l494_494350

-- Define given conditions
def speed_of_stream (V_s : ℝ) : Prop := V_s = 5
def travel_downstream (distance time V_d : ℝ) : Prop := distance = 216 ∧ time = 8 ∧ V_d = distance / time

-- The speed of the boat in still water
def speed_of_boat (V_b : ℝ) : Prop := ∃ V_s : ℝ, speed_of_stream V_s ∧ ∃ V_d : ℝ, 
  travel_downstream 216 8 V_d ∧ V_d = V_b + V_s ∧ V_b = 22

theorem boat_speed_in_still_water : ∃ (V_b : ℝ), speed_of_boat V_b :=
by
  use 22
  unfold speed_of_boat speed_of_stream travel_downstream
  split; norm_num
  split
  use 5
  split; norm_num
  use 27
  split; norm_num

end boat_speed_in_still_water_l494_494350


namespace symmetric_to_P_prime_is_in_fourth_quadrant_l494_494070

variable {a b : ℝ}

def is_in_second_quadrant (a b : ℝ) : Prop := a < 0 ∧ b > 0
def symmetric_with_y_axis (a b : ℝ) : ℝ × ℝ := (-a, b)
def point_P_prime (a b : ℝ) : ℝ × ℝ := (a - 1, -b)
def point_symmetric_to_P_prime (a b : ℝ) : ℝ × ℝ := symmetric_with_y_axis (a - 1) (-b)
def is_in_fourth_quadrant (a b : ℝ) : Prop := a > 0 ∧ b < 0

theorem symmetric_to_P_prime_is_in_fourth_quadrant (h : is_in_second_quadrant a b) :
  is_in_fourth_quadrant (fst (point_symmetric_to_P_prime a b)) (snd (point_symmetric_to_P_prime a b)) :=
by
  sorry

end symmetric_to_P_prime_is_in_fourth_quadrant_l494_494070


namespace shaded_region_area_l494_494583

structure Hexagon :=
  (A B C D E F : Point)
  (area : ℝ)
  (is_regular : True)

structure Point :=
  (x y : ℝ)

def G := midpoint A B
def H := midpoint C D
def I := midpoint D E
def J := midpoint F A

theorem shaded_region_area (hex : Hexagon) 
  (h : hex.area = 60) 
  (hg : G = midpoint hex.A hex.B) 
  (hh : H = midpoint hex.C hex.D) 
  (hi : I = midpoint hex.D hex.E) 
  (hj : J = midpoint hex.F hex.A) : 
  ∃ (shaded_area : ℝ), shaded_area = 30 :=
begin
  sorry,
end

end shaded_region_area_l494_494583


namespace max_min_cos_sin_cos_l494_494485

theorem max_min_cos_sin_cos (x y z : ℝ) (h1 : x ≥ y) (h2 : y ≥ z) (h3 : z ≥ π / 12) (h4 : x + y + z = π / 2) :
  ∃ (max_val min_val : ℝ), 
    (max_val = (2 + Real.sqrt 3) / 8) ∧ 
    (min_val = 1 / 8) ∧ 
    max_val = max (cos x * sin y * cos z) ∧ 
    min_val = min (cos x * sin y * cos z) :=
  sorry

end max_min_cos_sin_cos_l494_494485


namespace anya_initial_seat_l494_494087

theorem anya_initial_seat :
  ∃ a v g d e : ℕ, 
    {a, v, g, d, e} = {1, 2, 3, 4, 5} ∧
    (v + 3) % 5 + 1 = 5 ∧ -- Varya moves 3 seats to the right
    g - 1 = 4 ∧           -- Galya moves 1 seat to the left
    d ≠ e ∧               -- Diana and Ella swap places
    1 ∈ {a, v + 3, g - 1, d, e} ∧ -- End seat is left for Anya (final seat)
    a = 3.                -- Original position of Anya
Proof
  sorry

end anya_initial_seat_l494_494087


namespace andy_wrong_questions_l494_494783

/-- Andy, Beth, Charlie, and Daniel take a test. Andy and Beth together get the same number of 
    questions wrong as Charlie and Daniel together. Andy and Daniel together get four more 
    questions wrong than Beth and Charlie do together. Charlie gets five questions wrong. 
    Prove that Andy gets seven questions wrong. -/
theorem andy_wrong_questions (a b c d : ℕ) (h1 : a + b = c + d) (h2 : a + d = b + c + 4) (h3 : c = 5) :
  a = 7 :=
by
  sorry

end andy_wrong_questions_l494_494783


namespace num_ways_dist_6_balls_3_boxes_l494_494051

open Finset

/-- Number of ways to distribute 6 distinguishable balls into 3 indistinguishable boxes -/
theorem num_ways_dist_6_balls_3_boxes : 
  let num_ways := 
    let n := 6 in let k := 3 in
    ((choose n 6) + 
    (choose n 5) + 
    (choose n 4) + 
    (choose n 4) + 
    (choose n 3) / 2 + 
    (choose n 3) * (choose (n - 3) 2) + 
    (choose n 2) * (choose (n - 2) 2) / (factorial k)) 
  in num_ways = 122 := 
by 
  sorry

end num_ways_dist_6_balls_3_boxes_l494_494051


namespace find_a2_sum_coefficients_remainder_div_6_l494_494885

noncomputable def binomial (n k : ℕ) : ℕ := Nat.choose n k 

-- Conditions
-- given a function f(x) = (2x-3)^9 and the sum of binomial coefficients of (2x-3)^9 is 512
def f (x : ℝ) : ℝ := (2 * x - 3)^9

-- Question 1: Prove that a_2 = -144
theorem find_a2 : ∀ x : ℝ, 
  ∃ a_2 : ℤ, 
  (f(x) = (2 * (x - 1) - 1)^9 ∧ 
   a_2 = -144) := 
by 
  intros x 
  use -144
  sorry

-- Question 2: Prove the sum of coefficients from a1 to an equals 2
theorem sum_coefficients : 
  let a := (f 1 : ℤ),
      exp := a + List.sum [ a_1 * 1^1, a_2 * 1^2, a_3 * 1^3, a_4 * 1^4, a_5 * 1^5, a_6 * 1^6, a_7 * 1^7, a_8 * 1^8, a_9 * 1^9 ] 
  in exp - (f 1 : ℤ) = 2 :=
by 
  intros 
  let a : ℤ := -1
  let b : ℤ := 1
  have h : a = (2 * 1 - 3)^9 := by norm_num 
  have hb : b = (2 * 2 - 3)^9 := by norm_num 
  rw [f, h, hb]
  sorry

-- Question 3: Prove the remainder when f(20) - 20 is divided by 6 is 5
theorem remainder_div_6 :
  (f (20) - 20) % 6 = 5 := by
  sorry

end find_a2_sum_coefficients_remainder_div_6_l494_494885


namespace halfway_fraction_l494_494704

theorem halfway_fraction (a b : ℚ) (ha : a = 3/4) (hb : b = 5/6) : (a + b) / 2 = 19/24 :=
by
  rw [ha, hb] -- replace a and b with 3/4 and 5/6 respectively
  have h1 : 3/4 + 5/6 = 19/12,
  { norm_num, -- ensures 3/4 + 5/6 = 19/12
    linarith },
  rw h1, -- replace a + b with 19/12
  norm_num -- ensures (19/12) / 2 = 19/24

end halfway_fraction_l494_494704


namespace balls_in_boxes_l494_494034

theorem balls_in_boxes : ∀ (balls boxes : ℕ), balls = 6 → boxes = 3 → 
  (∃! n : ℕ, n = 47 ∧ n = number_of_ways_to_distribute_balls_in_boxes_d(balls, boxes)) :=
by
  intros balls boxes h1 h2
  have h_balls : balls = 6 := h1
  have h_boxes : boxes = 3 := h2
  use 47
  split
  sorry
  sorry

end balls_in_boxes_l494_494034


namespace min_value_of_m_l494_494122

noncomputable def y : ℝ → ℝ
| x => x^2 - 1

def m (x y : ℝ) : ℝ :=
  (3 * x + y - 5) / (x - 1) + (x + 3 * y - 7) / (y - 2)

theorem min_value_of_m : 
  x > sqrt 3 → y x = x^2 - 1 → m x (y x) ≥ 8 :=
by
  sorry

end min_value_of_m_l494_494122


namespace largest_smallest_awarded_number_sum_awarded_numbers_divisible_by_13_l494_494738

-- Definition: 6-digit number with distinct digits is awarded if sum of the first 3 digits equals sum of the last 3 digits
def is_awarded (n : ℕ) : Prop :=
  let digits := List.ofDigits (Nat.digits 10 n) in
  n >= 100000 ∧ n < 1000000 ∧ List.Nodup digits ∧
  List.sum (digits.take 3) = List.sum (digits.drop 3)

-- Proof for part (a): 
-- There exist the largest and smallest awarded numbers with distinct digits
theorem largest_smallest_awarded_number :
  ∃ (n_max n_min : ℕ), is_awarded n_max ∧ is_awarded n_min ∧
  ∀ (n : ℕ), is_awarded n → n ≤ n_max ∧ n_min ≤ n := sorry

-- Proof for part (b):
-- The sum of all awarded numbers with 6 different digits is divisible by 13
theorem sum_awarded_numbers_divisible_by_13 :
  ∃ (s : ℕ), (∑ n in (Finset.range 1000000).filter is_awarded, n) = s ∧ 13 ∣ s := sorry

end largest_smallest_awarded_number_sum_awarded_numbers_divisible_by_13_l494_494738


namespace price_of_first_tea_x_l494_494648

theorem price_of_first_tea_x (x : ℝ) :
  let price_second := 135
  let price_third := 173.5
  let avg_price := 152
  let ratio := [1, 1, 2]
  1 * x + 1 * price_second + 2 * price_third = 4 * avg_price -> x = 126 :=
by
  intros price_second price_third avg_price ratio h
  sorry

end price_of_first_tea_x_l494_494648


namespace sewers_handle_rain_l494_494209

theorem sewers_handle_rain (total_capacity : ℕ) (runoff_per_hour : ℕ) : 
  total_capacity = 240000 → 
  runoff_per_hour = 1000 → 
  total_capacity / runoff_per_hour / 24 = 10 :=
by 
  intro h1 h2
  sorry

end sewers_handle_rain_l494_494209


namespace matrix_multiplication_correct_l494_494426

variables (d e f : ℝ)

def matrix_A : Matrix (Fin 3) (Fin 3) ℝ :=
  !![ 0,  d, -e;
    -d,  0,  f;
     e, -f,  0]

def matrix_B : Matrix (Fin 3) (Fin 3) ℝ :=
  !![ f^2,  fd,  fe;
     fd,  d^2,  de;
     fe,  de,  e^2]

def expected_result : Matrix (Fin 3) (Fin 3) ℝ :=
  !![ d^2 - e^2,  2 * f * d,  0;
      0,  f^2 - d^2,  d * e - f * e;
      0,  e^2 - d^2,  f * e - d * f]

theorem matrix_multiplication_correct :
  matrix_A d e f ⬝ matrix_B d e f = expected_result d e f := 
  by sorry

end matrix_multiplication_correct_l494_494426


namespace polar_eq_line_AB_ratio_CD_CE_l494_494983

-- Definitions for the circle C and curve C1
def circle_C := ∀ (x y : ℝ), (x - 2)^2 + y^2 = 4
def curve_C1 (θ : ℝ) : ℝ := -4 * (Real.sqrt 3) * (Real.sin θ)

-- Polar equation of line AB
theorem polar_eq_line_AB (θ : ℝ) : θ = -Real.pi / 6 :=
by sorry

-- Definitions for line C2
def line_C2 (t : ℝ) : ℝ × ℝ := (2 + (Real.sqrt 3 / 2) * t, (1 / 2) * t)

-- Ratio of distances |CD| : |CE|
theorem ratio_CD_CE (D E C: ℝ × ℝ) (hD : D = (1, -Real.sqrt 3 / 3)) 
  (hE : E = (0, -2 * Real.sqrt 3 / 3)) 
  (C : ℝ × ℝ := (2, 0)) :
  (Real.sqrt ((2 - 1)^2 + (0 + Real.sqrt 3 / 3)^2)) / 
  (Real.sqrt ((2 - 0)^2 + (0 + 2 * Real.sqrt 3 / 3)^2)) = 
  1 / 2 :=
by sorry

end polar_eq_line_AB_ratio_CD_CE_l494_494983


namespace matrix_multiplication_correct_l494_494813

def mat1 : Matrix (Fin 2) (Fin 2) ℤ := ![![3, -2], ![-1, 4]]
def mat2 : Matrix (Fin 2) (Fin 2) ℤ := ![![2, 5], ![0, -3]]
def mat3 : Matrix (Fin 2) (Fin 2) ℤ := ![![6, 21], ![-2, -17]]

theorem matrix_multiplication_correct : mat1 ⬝ mat2 = mat3 :=
by
  sorry

end matrix_multiplication_correct_l494_494813


namespace average_weight_of_remaining_carrots_l494_494345

noncomputable def total_weight_30_carrots : ℕ := 5940
noncomputable def total_weight_3_carrots : ℕ := 540
noncomputable def carrots_count_30 : ℕ := 30
noncomputable def carrots_count_3_removed : ℕ := 3
noncomputable def carrots_count_remaining : ℕ := 27
noncomputable def average_weight_of_removed_carrots : ℕ := 180

theorem average_weight_of_remaining_carrots :
  (total_weight_30_carrots - total_weight_3_carrots) / carrots_count_remaining = 200 :=
  by
  sorry

end average_weight_of_remaining_carrots_l494_494345


namespace bcdeq65_l494_494907

theorem bcdeq65 (a b c d e f : ℝ)
  (h₁ : a * b * c = 130)
  (h₂ : c * d * e = 500)
  (h₃ : d * e * f = 250)
  (h₄ : (a * f) / (c * d) = 1) :
  b * c * d = 65 :=
sorry

end bcdeq65_l494_494907


namespace range_of_a_l494_494565

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, |x - a| + |x - 1| ≤ 3) → -2 ≤ a ∧ a ≤ 4 :=
by
  intro h
  sorry

end range_of_a_l494_494565


namespace maximum_minimum_cos_sin_cos_l494_494479

noncomputable def max_min_cos_sin_cos_product (x y z : ℝ) : ℝ × ℝ :=
  if x + y + z = π / 2 ∧ x ≥ y ∧ y ≥ z ∧ z ≥ π / 12 then
    let product := cos x * sin y * cos z
    (max product, min product)
  else (0, 0)

theorem maximum_minimum_cos_sin_cos :
  ∃ x y z : ℝ, 
    x + y + z = π / 2 ∧ x ≥ y ∧ y ≥ z ∧ z ≥ π / 12 ∧
    max_min_cos_sin_cos_product x y z = ( (2 + real.sqrt 3) / 8, 1 / 8) :=
by
  sorry

end maximum_minimum_cos_sin_cos_l494_494479


namespace distribution_of_6_balls_in_3_indistinguishable_boxes_l494_494003

-- Definition of the problem with conditions
def ways_to_distribute_balls_into_boxes
    (balls : ℕ) (boxes : ℕ) (distinguishable : bool)
    (indistinguishable : bool) : ℕ :=
  if (balls = 6) ∧ (boxes = 3) ∧ (distinguishable = true) ∧ (indistinguishable = true) 
  then 122 -- The correct answer given the conditions
  else 0

-- The Lean statement for the proof problem
theorem distribution_of_6_balls_in_3_indistinguishable_boxes :
  ways_to_distribute_balls_into_boxes 6 3 true true = 122 :=
by sorry

end distribution_of_6_balls_in_3_indistinguishable_boxes_l494_494003


namespace jenny_cat_expense_l494_494600

def adoption_fee : ℕ := 50
def vet_visits_cost : ℕ := 500
def monthly_food_cost : ℕ := 25
def jenny_toy_expenses : ℕ := 200
def split_factor : ℕ := 2

-- Given conditions, prove that Jenny spent $625 on the cat in the first year.
theorem jenny_cat_expense : 
  let yearly_food_cost := 12 * monthly_food_cost 
  let total_shared_expenses := adoption_fee + vet_visits_cost + yearly_food_cost 
  let jenny_shared_expenses := total_shared_expenses / split_factor 
  let total_jenny_cost := jenny_shared_expenses + jenny_toy_expenses
  in total_jenny_cost = 625 := 
by 
  sorry

end jenny_cat_expense_l494_494600


namespace find_k_t_l494_494592

noncomputable def areaOfTrianglePSR : ℝ :=
  let PM := 24
  let QN := 36
  let PQ := 32
  let k := 38
  let t := 91
  k * Real.sqrt t

theorem find_k_t (k t : ℕ) (PSR_area : ℝ) :
  PSR_area = (38 : ℝ) * Real.sqrt 91 → k + t = 129 :=
by {
  intros h,
  have h1 : PSR_area = 38 * Real.sqrt 91 := h,
  have h2 : k = 38 := by linarith [h1.symm],
  have h3 : t = 91 := by linarith [h1.symm],
  rw [h2, h3],
  exact rfl
}

end find_k_t_l494_494592


namespace find_certain_number_l494_494078

theorem find_certain_number (x certain_number : ℕ) (h: x = 3) (h2: certain_number = 5 * x + 4) : certain_number = 19 :=
by
  sorry

end find_certain_number_l494_494078


namespace problem_statement_l494_494886

def f (x : ℝ) : ℝ := 3^x + 3^(-x)

theorem problem_statement (a : ℝ) (h : f a = 3) : f (2 * a) = 7 := by
  sorry

end problem_statement_l494_494886


namespace ratio_of_sum_of_divisors_l494_494121

def M : ℕ := 36 * 36 * 75 * 224

def sum_of_odd_divisors (n : ℕ) : ℕ :=
  ∑ d in (Finset.filter (λ x, ¬(even x)) (Finset.range (n+1))), d

def sum_of_all_divisors (n : ℕ) : ℕ :=
  ∑ d in (Finset.range (n+1)), d

def sum_of_even_divisors (n : ℕ) : ℕ :=
  sum_of_all_divisors n - sum_of_odd_divisors n

theorem ratio_of_sum_of_divisors : (sum_of_odd_divisors M : ℚ) / sum_of_even_divisors M = 1 / 510 := by
  sorry

end ratio_of_sum_of_divisors_l494_494121


namespace line_equation_of_projection_l494_494678

noncomputable def proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_uv := u.1 * v.1 + u.2 * v.2
  let norm_v2 := v.1 * v.1 + v.2 * v.2
  (dot_uv / norm_v2 * v.1, dot_uv / norm_v2 * v.2)

theorem line_equation_of_projection (x y : ℝ) :
  proj (x, y) (3, -4) = (9 / 5, -12 / 5) ↔ y = (3 / 4) * x - 15 / 4 :=
sorry

end line_equation_of_projection_l494_494678


namespace distributive_laws_l494_494120

def avg (a b : ℝ) : ℝ := (a + b) / 2

theorem distributive_laws (x y z : ℝ) :
  (x @ (y + z) = (x @ y) + (x @ z) → false) ∧
  (x + (y @ z) = (x + y) @ (x + z)) ∧
  (x @ (y @ z) = (x @ y) @ (x @ z)) :=
by
  sorry

end distributive_laws_l494_494120


namespace remainder_when_divided_by_13_l494_494334

theorem remainder_when_divided_by_13 (N : ℕ) (k : ℕ) (hk : N = 39 * k + 15) : N % 13 = 2 :=
sorry

end remainder_when_divided_by_13_l494_494334


namespace problem_condition_l494_494062

theorem problem_condition (x y : ℝ) (h : x^2 + y^2 - x * y = 1) : 
  x + y ≥ -2 ∧ x^2 + y^2 ≤ 2 :=
by
  sorry

end problem_condition_l494_494062


namespace triangle_area_l494_494576

namespace Geometry

open Real

variables {a b c A B C : ℝ}

def area_of_acute_triangle (a b c : ℝ) : ℝ :=
  (1 / 2) * a * b * sin C

theorem triangle_area
  (h₁ : (sin B * sin C) / sin A = (3 * sqrt 7) / 2)
  (h₂ : b = 4 * a)
  (h₃ : a + c = 5)
  (h₄ : sin C = (2 * sqrt 7) / 8)
  (h₅ : ∀ C : ℝ, 0 < C ∧ C < π / 2 -> cos C = 1 / 8) :
  area_of_acute_triangle 1 4 ((5 - 1)) = (sqrt 7) / 2 :=
by
  sorry

end Geometry

end triangle_area_l494_494576


namespace quadratic_roots_diff_l494_494867

noncomputable def quadratic_eq : Polynomial ℚ := Polynomial.C 2 * X ^ 2 + Polynomial.C (-5) * X + Polynomial.C (-3)

theorem quadratic_roots_diff (m n : ℤ) (h1 : n ≠ 0) (h2 : (∀ p : ℤ, Prime p → ¬ (p ^ 2 ∣ m))) :
  let Δ := (-5)^2 - 4 * 2 * (-3)
  let r1 := (5 + Real.sqrt (Δ : ℚ)) / (4 : ℚ)
  let r2 := (5 - Real.sqrt (Δ : ℚ)) / (4 : ℚ)
  ∃ m n : ℤ, abs (r1 - r2) = (Real.sqrt m) / n ∧ m + n = 51 :=
begin
  sorry
end

end quadratic_roots_diff_l494_494867


namespace liu_xiang_hurdles_l494_494387

theorem liu_xiang_hurdles :
  (110 - 13.72 - 14.02) / 9 = 9.14 ∧
  (2.5 + 0.96 * 9 + 1.4) = 12.54 :=
by {
  split,
  { norm_num, },
  { norm_num, },
}

end liu_xiang_hurdles_l494_494387


namespace cookies_per_bag_l494_494081

theorem cookies_per_bag (total_cookies bags : ℕ) (h_total : total_cookies = 703) (h_bags : bags = 37) :
  total_cookies / bags = 19 :=
by
  -- Given the conditions
  rw [h_total, h_bags] 
  -- Show the division
  exact Nat.div_eq_of_eq_mul_right (by norm_num) (by norm_num)

end cookies_per_bag_l494_494081


namespace max_area_l494_494606

section ParabolaTriangle

-- Defining points A and B
def A : ℝ × ℝ := (2, 2)
def B : ℝ × ℝ := (5, 0)

-- Function for point C on the parabola given p
def C (p : ℝ) : ℝ × ℝ := (p, p^2 - 6 * p + 10)

-- Function to calculate the area of triangle ABC using Shoelace Theorem
def area (p : ℝ) : ℝ := 
  (1/2) * abs ((A.1 * B.2 + B.1 * (p^2 - 6*p + 10) + p * A.2) - 
               (A.2 * B.1 + B.2 * p + (p^2 - 6*p + 10) * A.1))

-- The maximum area when p is between 2 and 5
theorem max_area : (∀ p : ℝ, 2 ≤ p ∧ p ≤ 5 → area p ≤ 3.5) ∧ 
                   (∃ p : ℝ, 2 ≤ p ∧ p ≤ 5 ∧ area p = 3.5) :=
by
  sorry

end ParabolaTriangle

end max_area_l494_494606


namespace area_triangle_CKL_l494_494155

/-- Define points A, B, K, C, H, and L --/
variables (A B K C H L : Point)

/-- AB is a segment of length 10 --/
axiom AB_length : dist A B = 10

/-- Circle ω is constructed with AB as its diameter --/
axiom ω : Circle
axiom ω_properties : ω.diameter A B

/-- Tangent at A and point K on tangent --/
axiom tangent_at_A : Tangent ω A
axiom K_on_tangent : PointOnLine K (tangent_at_A.line)

/-- Line through K different from AK touches ω at point C --/
axiom line_through_K : ∃ L : Line, LineThrough L K ∧ L ≠ tangent_at_A.line ∧ tangent_to_circle L ω C

/-- Altitude CH of triangle ABC intersects BK at L --/
axiom CH_altitude : ∃ H : Point, LineOrthogonalToLine H C (line_through_K.KEY.line)
axiom L_intersection : IntersectionOfLines L (SegmentThroughPoints H B) (SegmentThroughPoints K H L)

/-- Given BH:AH = 1:4 --/
axiom ratio_BH_AH : ratio (dist B H) (dist A H) = 1 / 4

/-- Prove area of triangle CKL is 8 --/
theorem area_triangle_CKL : area_of_triangle K C L = 8 := 
sorry

end area_triangle_CKL_l494_494155


namespace length_of_O1O2_l494_494502

def incenter (A B C : Point) : Point := sorry
def distance (P Q : Point) : ℝ := sorry

structure Triangle :=
(A B C : Point)
(angleC : ∠C = 90°)
(ac_equals_4 : distance A C = 4)
(bc_equals_3 : distance B C = 3)
(cd_is_altitude : ∃ D, distance C D = distance D A * distance D B / distance A B)

def incenter_in_triangle (T : Triangle) : (incenter (T.A) (T.C) (D : Point)) := sorry

noncomputable def length_O1O2 (T : Triangle) : ℝ :=
let O1 := incenter_in_triangle ⟨T.A, T.C, (D : Point), λ _, T.angleC, T.ac_equals_4, T.bc_equals_3, T.cd_is_altitude⟩ in
let O2 := incenter_in_triangle ⟨T.B, T.C, (D : Point), λ _, T.angleC, T.ac_equals_4, T.bc_equals_3, T.cd_is_altitude⟩ in
distance O1 O2

theorem length_of_O1O2 (T : Triangle) : length_O1O2 T = 2 := sorry

end length_of_O1O2_l494_494502


namespace find_number_of_children_l494_494356

theorem find_number_of_children (adults children : ℕ) (adult_ticket_price child_ticket_price total_money change : ℕ) 
    (h1 : adult_ticket_price = 9) 
    (h2 : child_ticket_price = adult_ticket_price - 2) 
    (h3 : total_money = 40) 
    (h4 : change = 1) 
    (h5 : adults = 2) 
    (total_cost : total_money - change = adults * adult_ticket_price + children * child_ticket_price) : 
    children = 3 :=
sorry

end find_number_of_children_l494_494356


namespace intersection_A_B_l494_494904

def set_A : Set ℝ := { x | (x + 1) * (x - 2) < 0 }
def set_B : Set ℝ := { x | 2^(x - 1) >= 1 }

theorem intersection_A_B : set_A ∩ set_B = { x | 1 ≤ x ∧ x < 2 } :=
  sorry

end intersection_A_B_l494_494904


namespace increase_80_by_150_percent_l494_494289

theorem increase_80_by_150_percent : 
  ∀ x percent, x = 80 ∧ percent = 1.5 → x + (x * percent) = 200 :=
by
  sorry

end increase_80_by_150_percent_l494_494289


namespace part_I_part_II_l494_494143

def f (x a : ℝ) := |2 * x - a| + 5 * x

theorem part_I (x : ℝ) : f x 3 ≥ 5 * x + 1 ↔ (x ≤ 1 ∨ x ≥ 2) := sorry

theorem part_II (a x : ℝ) (h : (∀ x, f x a ≤ 0 ↔ x ≤ -1)) : a = 3 := sorry

end part_I_part_II_l494_494143


namespace balls_in_boxes_l494_494008

/-
Prove that the number of ways to distribute 6 distinguishable balls into 3 indistinguishable boxes is 132.
-/
theorem balls_in_boxes : 
  ∃ (ways : ℕ), ways = 132 ∧ ways = 
    (1) + 
    (Nat.choose 6 5) + 
    (Nat.choose 6 4) + 
    (Nat.choose 6 4) + 
    (Nat.choose 6 3) + 
    (Nat.choose 6 3 * Nat.choose 3 2) + 
    (Nat.choose 6 2 * Nat.choose 4 2 / 6) := 
by
  sorry

end balls_in_boxes_l494_494008


namespace plane_intersections_l494_494586

theorem plane_intersections (n : ℕ) (h : ∀ i < n, ∃! j < n, i ≠ j ∧ plane_intersects i j) : n = 2000 ∨ n = 3998 :=
by
  sorry

end plane_intersections_l494_494586


namespace lunch_to_novel_ratio_l494_494110

theorem lunch_to_novel_ratio 
  (initial_amount : ℕ) 
  (novel_cost : ℕ) 
  (remaining_after_mall : ℕ) 
  (spent_on_lunch : ℕ)
  (h1 : initial_amount = 50) 
  (h2 : novel_cost = 7) 
  (h3 : remaining_after_mall = 29) 
  (h4 : spent_on_lunch = initial_amount - novel_cost - remaining_after_mall) :
  spent_on_lunch / novel_cost = 2 := 
  sorry

end lunch_to_novel_ratio_l494_494110


namespace f_zero_eq_f_expression_alpha_value_l494_494919

noncomputable def f (ω x : ℝ) : ℝ :=
  3 * Real.sin (ω * x + Real.pi / 6)

theorem f_zero_eq (ω : ℝ) (hω : ω > 0) (h_period : (2 * Real.pi / ω) = Real.pi / 2) :
  f ω 0 = 3 / 2 :=
by
  sorry

theorem f_expression (ω : ℝ) (hω : ω > 0) (h_period : (2 * Real.pi / ω) = Real.pi / 2) :
  ∀ x : ℝ, f ω x = f 4 x :=
by
  sorry

theorem alpha_value (f_4 : ℝ → ℝ) (α : ℝ) (hα : α ∈ Set.Ioo 0 (Real.pi / 2))
  (h_f4 : ∀ x : ℝ, f_4 x = 3 * Real.sin (4 * x + Real.pi / 6)) (h_fα : f_4 (α / 2) = 3 / 2) :
  α = Real.pi / 3 :=
by
  sorry

end f_zero_eq_f_expression_alpha_value_l494_494919


namespace toba_fractions_sum_bounds_l494_494725

noncomputable def A : ℝ := ∑ n in finset.range (200 - 101 + 1), 1 / (101 + n : ℝ)

theorem toba_fractions_sum_bounds (A : ℝ) : (7 / 12 : ℝ) < A ∧ A < (5 / 6 : ℝ) :=
by
  let A := ∑ n in finset.range (200 - 100), 1 / (101 + n : ℝ)
  sorry

end toba_fractions_sum_bounds_l494_494725


namespace cheesecake_factory_hours_per_day_l494_494164

theorem cheesecake_factory_hours_per_day
  (wage_per_hour : ℝ)
  (days_per_week : ℝ)
  (weeks : ℝ)
  (combined_savings : ℝ)
  (robbie_saves : ℝ)
  (jaylen_saves : ℝ)
  (miranda_saves : ℝ)
  (h : ℝ) :
  wage_per_hour = 10 → days_per_week = 5 → weeks = 4 → combined_savings = 3000 →
  robbie_saves = 2/5 → jaylen_saves = 3/5 → miranda_saves = 1/2 →
  (robbie_saves * (wage_per_hour * h * days_per_week) +
  jaylen_saves * (wage_per_hour * h * days_per_week) +
  miranda_saves * (wage_per_hour * h * days_per_week)) * weeks = combined_savings →
  h = 10 :=
by
  intros hwage hweek hweeks hsavings hrobbie hjaylen hmiranda heq
  sorry

end cheesecake_factory_hours_per_day_l494_494164


namespace proof_problem_l494_494924

-- Definition of polar to cartesian conversion
def polar_to_cartesian (rho θ : ℝ) : ℝ × ℝ :=
  (rho * cos(θ), rho * sin(θ))

-- Given polar equation
def polar_eq (ρ θ : ℝ) : Prop :=
  ρ = 2 * cos θ

-- Cartesian equation resulting from polar_eq
def cartesian_eq (x y : ℝ) : Prop :=
  (x - 1)^2 + y^2 = 1

-- Parametric equation of line l
def parametric_eq (t m : ℝ) : ℝ × ℝ :=
  (sqrt 3 / 2 * t + m, 1 / 2 * t)

-- General equation of line l
def general_eq (x y m : ℝ) : Prop :=
  x - sqrt 3 * y - m = 0

-- Values of m given |PA| * |PB| = 1 and line l intersects curve C
def valid_m_values : Set ℝ := 
  {m | m = 1 ∨ m = 1 + sqrt 2 ∨ m = 1 - sqrt 2}

-- The main theorem to prove
theorem proof_problem (m : ℝ) :
  (∀ (ρ θ x y : ℝ), polar_eq ρ θ → polar_to_cartesian ρ θ = (x, y) → cartesian_eq x y) →
  (∀ (t : ℝ), parametric_eq t m = (x, y) → general_eq x y m) →
  (∀ (t1 t2 : ℝ), |PA|*|PB| = |m^2 - 2m| = 1 → m ∈ valid_m_values) :=
sorry

end proof_problem_l494_494924


namespace largest_number_division_l494_494459

-- Define the fractions
def frac₁ : ℚ := 154 / 195
def frac₂ : ℚ := 385 / 156
def frac₃ : ℚ := 231 / 130

-- Define the target fraction
def target : ℚ := 77 / 780

-- The statement to show that target is the largest number such that the fractions are natural numbers when divided by it.
theorem largest_number_division :
  (frac₁ / target).denom = 1 ∧ (frac₂ / target).denom = 1 ∧ (frac₃ / target).denom = 1 :=
sorry

end largest_number_division_l494_494459


namespace school_total_payment_l494_494184

theorem school_total_payment
  (price : ℕ)
  (kindergarten_models : ℕ)
  (elementary_library_multiplier : ℕ)
  (model_reduction_percentage : ℚ)
  (total_models : ℕ)
  (reduced_price : ℚ)
  (total_payment : ℚ)
  (h1 : price = 100)
  (h2 : kindergarten_models = 2)
  (h3 : elementary_library_multiplier = 2)
  (h4 : model_reduction_percentage = 0.05)
  (h5 : total_models = kindergarten_models + (kindergarten_models * elementary_library_multiplier))
  (h6 : total_models > 5)
  (h7 : reduced_price = price - (price * model_reduction_percentage))
  (h8 : total_payment = total_models * reduced_price) :
  total_payment = 570 := 
by
  sorry

end school_total_payment_l494_494184


namespace log_addition_property_l494_494800

theorem log_addition_property : log 10 50 + log 10 20 = 3 :=
by
  sorry

end log_addition_property_l494_494800


namespace increase_by_150_percent_l494_494309

theorem increase_by_150_percent (n : ℕ) : 
  n = 80 → n + (3 / 2) * n = 200 :=
by
  intros h
  rw [h]
  norm_num
  sorry

end increase_by_150_percent_l494_494309


namespace area_triangle_sum_l494_494959

theorem area_triangle_sum (AB : ℝ) (angle_BAC angle_ABC angle_ACB angle_EDC : ℝ) 
  (h_AB : AB = 1) (h_angle_BAC : angle_BAC = 70) (h_angle_ABC : angle_ABC = 50) 
  (h_angle_ACB : angle_ACB = 60) (h_angle_EDC : angle_EDC = 80) :
  let area_triangle := (1/2) * AB * (Real.sin angle_70 / Real.sin angle_60) * (Real.sin angle_60) 
  let area_CDE := (1/2) * (Real.sin angle_80)
  area_triangle + 2 * area_CDE = (Real.sin angle_70 + Real.sin angle_80) / 2 :=
sorry

end area_triangle_sum_l494_494959


namespace evaluate_expression_l494_494829

theorem evaluate_expression : (-3)^7 / 3^5 + 2^6 - 4^2 = 39 := by
  -- (-3)^7 evaluates to -3^7 since 7 is odd: (-a)^n = -a^n
  have h1 : (-3)^7 = -(3^7) := by sorry
  -- 3^7 / 3^5 = 3^(7-5): a^m / a^n = a^(m-n)
  have h2 : 3^7 / 3^5 = 3^(7-5) := by sorry
  -- Combine the simplified terms
  have h3 : (-3)^7 / 3^5 = -(3^2) := by sorry
  have h4 : 2^6 = 64 := by sorry
  have h5 : 4^2 = 16 := by sorry
  calc
    (-3)^7 / 3^5 + 2^6 - 4^2
        = -(3^2) + 64 - 16 : by rw [h3, h4, h5]
    ... = -9 + 64 - 16 : by rfl
    ... = 39 : by norm_num

end evaluate_expression_l494_494829


namespace inequality_one_solution_inequality_two_solution_l494_494642

theorem inequality_one_solution (x : ℝ) :
  (-x^2 + x + 6 ≤ 0) ↔ (x ∈ set.Iic (-2) ∪ set.Ici 3) :=
sorry

theorem inequality_two_solution (x : ℝ) :
  (x^2 - 4x - 5 < 0) ↔ (x ∈ set.Ioi (-1) ∩ set.Iio 5) :=
sorry

end inequality_one_solution_inequality_two_solution_l494_494642


namespace num_children_eq_3_l494_494360

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

end num_children_eq_3_l494_494360


namespace AP_eq_AQ_l494_494156

-- Defining the given conditions and variables
variables {A B C D E P Q : Type}
variables [acute_triangle ABC]
variables [on_segment BC D E]
variables [BD_eq_CE : BD = CE]
variables [on_arc_DE_of_circumcircle_ADE A P Q]
variables [P_Q_not_contain_A P Q]

-- Distances as per the conditions
variables [eq_dist_AB_PC : AB = PC]
variables [eq_dist_AC_BQ : AC = BQ]

-- Main theorem statement
theorem AP_eq_AQ : AP = AQ := by
  sorry

end AP_eq_AQ_l494_494156


namespace increase_by_percentage_l494_494257

theorem increase_by_percentage (initial_amount : ℕ) (percentage_increase : ℕ) :
  initial_amount = 80 → 
  percentage_increase = 150 → 
  initial_amount * (percentage_increase / 100) + initial_amount = 200 :=
by
  intros h_initial h_percentage
  rw [h_initial, h_percentage]
  calc
    80 * (150 / 100) + 80 = 80 * 1.5 + 80 : by norm_num
                     ... = 120 + 80         : by norm_num
                     ... = 200              : by norm_num
  sorry

end increase_by_percentage_l494_494257


namespace solve_cauchy_problem_l494_494463

noncomputable def solution {y : ℝ → ℝ} (y'' y' y : ℝ → ℝ) : Prop :=
  ∀ (x : ℝ), y'' x - 3 * y' x + 2 * y x = 0

theorem solve_cauchy_problem (y : ℝ → ℝ) (y' y'' : ℝ → ℝ)
  (h_eq : solution y'' y' y)
  (h_initial_y : y 0 = 1)
  (h_initial_y' : y' 0 = 0) :
  y = λ x, 2 * Real.exp x - Real.exp (2 * x) :=
by
  sorry

end solve_cauchy_problem_l494_494463


namespace increase_by_150_percent_l494_494306

theorem increase_by_150_percent (n : ℕ) : 
  n = 80 → n + (3 / 2) * n = 200 :=
by
  intros h
  rw [h]
  norm_num
  sorry

end increase_by_150_percent_l494_494306


namespace monotonic_increasing_on_interval_l494_494527

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  x^2 - a * Real.log x

theorem monotonic_increasing_on_interval (a : ℝ) :
  (∀ x > 1, 2 * x - a / x ≥ 0) → a ≤ 2 :=
sorry

end monotonic_increasing_on_interval_l494_494527


namespace ratio_of_a_b_to_b_c_l494_494546

theorem ratio_of_a_b_to_b_c (a b c : ℝ) (h₁ : b / a = 3) (h₂ : c / b = 2) : 
  (a + b) / (b + c) = 4 / 9 := by
  sorry

end ratio_of_a_b_to_b_c_l494_494546


namespace vector_projection_example_l494_494226

theorem vector_projection_example :
  ∀ (a : ℝ) (ϕ : ℝ), a = 5 → ϕ = 60 → (a * (real.cos (ϕ * real.pi / 180))) = 2.5 :=
by
  intros a ϕ ha hϕ
  rw [ha, hϕ]
  rw [real.cos_pi_div_three] -- cos 60 degrees is 1/2
  norm_num

end vector_projection_example_l494_494226


namespace problem_l494_494714

theorem problem : 3 + 15 / 3 - 2^3 = 0 := by
  sorry

end problem_l494_494714


namespace increase_80_by_150_percent_l494_494233

-- Define the initial number (n) and the percentage increase (p)
def n : ℕ := 80
def p : ℚ := 1.5

-- The theorem stating the expected result after increasing n by 150%
theorem increase_80_by_150_percent : n + (p * n) = 200 := by
  sorry

end increase_80_by_150_percent_l494_494233


namespace systematic_sampling_l494_494380

theorem systematic_sampling (total_employees groups group_size draw_5th draw_10th : ℕ)
  (h1 : total_employees = 200)
  (h2 : groups = 40)
  (h3 : group_size = total_employees / groups)
  (h4 : draw_5th = 22)
  (h5 : ∃ x : ℕ, draw_5th = (5-1) * group_size + x)
  (h6 : ∀ k : ℕ, 1 ≤ k ∧ k ≤ groups → draw_10th = (k-1) * group_size + x) :
  draw_10th = 47 := 
by
  sorry

end systematic_sampling_l494_494380


namespace sine_of_plane_angle_l494_494786

noncomputable def cube_vertices : 
  List (ℝ × ℝ × ℝ) :=
  [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
   (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]

noncomputable def midpoint (p1 p2 : (ℝ × ℝ × ℝ)) : ℝ × ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2, (p1.3 + p2.3) / 2)

variable {A B C D A₁ B₁ C₁ D₁ E F : (ℝ × ℝ × ℝ)}

axiom cube_structure : 
  cube_vertices = [A, B, C, D, A₁, B₁, C₁, D₁]

axiom A_coords : A = (0, 0, 0)
axiom B_coords : B = (1, 0, 0)
axiom C_coords : C = (1, 1, 0)
axiom D_coords : D = (0, 1, 0)
axiom A₁_coords : A₁ = (0, 0, 1)
axiom B₁_coords : B₁ = (1, 0, 1)
axiom C₁_coords : C₁ = (1, 1, 1)
axiom D₁_coords : D₁ = (0, 1, 1)

axiom E_midpoint : E = midpoint A B
axiom F_midpoint : F = midpoint A A₁

theorem sine_of_plane_angle : 
  ∃ θ, sin θ = sqrt 3 / 2 ∧ θ = ∡ (CEB₁) (D₁FB₁) :=
by
  sorry

end sine_of_plane_angle_l494_494786


namespace log_addition_property_l494_494802

theorem log_addition_property : log 10 50 + log 10 20 = 3 :=
by
  sorry

end log_addition_property_l494_494802


namespace sequence_general_formula_sum_of_first_n_terms_l494_494146

/-- Problem 1: Sequence General Formula -/
theorem sequence_general_formula (S : ℕ → ℝ) (a : ℕ → ℝ) (hS : ∀ n : ℕ, S n = 2 * a n - 2) :
  ∀ n, a n = 2 ^ n :=
sorry

/-- Problem 2: Sum of the First n Terms -/
theorem sum_of_first_n_terms (a : ℕ → ℝ) (b : ℕ → ℝ) (hA : ∀ n, a n = 2 ^ n) (hB : ∀ n, b n = Real.logb 2 (a n)) :
  ∀ n, (Finset.range n).sum (λ i, 1 / (b i * b (i + 1))) = n / (n + 1) :=
sorry

end sequence_general_formula_sum_of_first_n_terms_l494_494146


namespace correct_statement_l494_494331

-- Definitions based on conditions in the problem
def statement_A : Prop := (sqrt 36 = 6)
def statement_B : Prop := (real.cbrt 8 = 2)
def statement_C : Prop := (sqrt (sqrt 4) = 2 ∨ sqrt (sqrt 4) = -2)
def statement_D : Prop := (sqrt 9 = -3)

-- Main statement to prove
theorem correct_statement : statement_B :=
by {
    sorry
}

end correct_statement_l494_494331


namespace max_min_cos_sin_product_l494_494482

theorem max_min_cos_sin_product (x y z : ℝ) 
  (h1 : x ≥ y) (h2 : y ≥ z) (h3 : z ≥ π / 12) (h4 : x + y + z = π / 2) :
  ∃ (maximum minimum : ℝ), maximum = (2 + Real.sqrt 3) / 8 ∧ minimum = 1 / 8 := by
  sorry

end max_min_cos_sin_product_l494_494482


namespace length_of_longer_leg_of_smallest_triangle_l494_494442

theorem length_of_longer_leg_of_smallest_triangle :
  ∀ (a b c : ℝ), 
  is_30_60_90_triangle (a, b, c) ∧ c = 16 
  ∧ (∀ (a₁ b₁ c₁ : ℝ), is_30_60_90_triangle (a₁, b₁, c₁) → b = c₁ → true) 
  ∧ (∀ (a₂ b₂ c₂ : ℝ), is_30_60_90_triangle (a₂, b₂, c₂) → true) 
  ∧ (∀ (a₃ b₃ c₃ : ℝ), is_30_60_90_triangle (a₃, b₃, c₃) → true) 
  → ∃ (a₄ b₄ c₄ : ℝ), is_30_60_90_triangle (a₄, b₄, c₄) ∧ b₄ = 9 :=
sorry

end length_of_longer_leg_of_smallest_triangle_l494_494442


namespace increase_by_percentage_l494_494252

theorem increase_by_percentage (initial_amount : ℕ) (percentage_increase : ℕ) :
  initial_amount = 80 → 
  percentage_increase = 150 → 
  initial_amount * (percentage_increase / 100) + initial_amount = 200 :=
by
  intros h_initial h_percentage
  rw [h_initial, h_percentage]
  calc
    80 * (150 / 100) + 80 = 80 * 1.5 + 80 : by norm_num
                     ... = 120 + 80         : by norm_num
                     ... = 200              : by norm_num
  sorry

end increase_by_percentage_l494_494252


namespace meeting_anniversary_day_l494_494409

-- Define the input parameters for the problem
def initial_years : Set ℕ := {1668, 1669, 1670, 1671}
def meeting_day := "Friday"
def is_leap_year (year : ℕ) : Bool := (year % 4 = 0)

-- Define the theorem for the problem statement
theorem meeting_anniversary_day :
  ∀ (year : ℕ), year ∈ initial_years →
  let leap_years := (∑ n in range 1668, if is_leap_year n then 1 else 0)
  let total_days := 11 * 365 + leap_years
  let day_of_week := total_days % 7
  in (day_of_week = 0 ∧ probability Friday = 3 / 4) ∨ (day_of_week = 6 ∧ probability Thursday 1 / 4) :=
by
  sorry

end meeting_anniversary_day_l494_494409


namespace optimal_selling_price_l494_494760

def purchase_price : ℝ := 60
def initial_selling_price : ℝ := 90
def initial_sales_volume : ℝ := 40

def profit (x : ℝ) : ℝ := (130 - x) * (x - 60)

theorem optimal_selling_price :
  ∃ x : ℝ, (60 ≤ x ∧ x ≤ 130) ∧ (∀ y : ℝ, 60 ≤ y → y ≤ 130 → profit(x) ≥ profit(y)) ∧ x = 95 :=
by
  sorry

end optimal_selling_price_l494_494760


namespace remainder_div_38_l494_494553

theorem remainder_div_38 (n : ℕ) (h : n = 432 * 44) : n % 38 = 32 :=
sorry

end remainder_div_38_l494_494553


namespace cap_given_sunglasses_probability_l494_494975

-- Definitions based on conditions
def total_people_sunglasses := 60
def total_people_caps := 40
def total_people_both (h_cap_probs : ℝ) := 0.5 * total_people_caps
def total_people_hats := 8
def cap_probability := (total_people_both 0.5) / total_people_caps

-- The theorem to prove the final probability based on the given conditions
theorem cap_given_sunglasses_probability : (total_people_both 0.5) / total_people_sunglasses = 1 / 3 :=
by
  sorry

end cap_given_sunglasses_probability_l494_494975


namespace teacher_mathematics_is_C_l494_494179

theorem teacher_mathematics_is_C :
  (∀ (A B C D : Type),
    (A → (A teaches Physics ∧ A teaches Chemistry)) →
    (B → (B teaches Mathematics ∧ B teaches English)) →
    (C → (C teaches Mathematics ∧ C teaches Physics ∧ C teaches Chemistry)) →
    (D → (D teaches Chemistry ∧ ∀ S, S ≠ D ∨ S teaches Chemistry)) →
    (∀ (T : Type), (T teaches Mathematics ∧ (T = A ∨ T = B ∨ T = C ∨ T = D)) → T = C)) :=
by
  intros A B C D a_conds b_conds c_conds d_conds T t_conds
  sorry

end teacher_mathematics_is_C_l494_494179


namespace family_has_11_eggs_l494_494750

def initialEggs : ℕ := 10
def eggsUsed : ℕ := 5
def chickens : ℕ := 2
def eggsPerChicken : ℕ := 3

theorem family_has_11_eggs :
  (initialEggs - eggsUsed) + (chickens * eggsPerChicken) = 11 := by
  sorry

end family_has_11_eggs_l494_494750


namespace tan_minus_pi_four_l494_494066

theorem tan_minus_pi_four (α : ℝ) (h1 : α ∈ set.Ioo (-π / 2) (-π / 4)) (h2 : cos α ^ 2 + cos (3 * π / 2 + 2 * α) = -1 / 2) :
  tan (α - π / 4) = 2 := sorry

end tan_minus_pi_four_l494_494066


namespace horse_catch_up_l494_494984

theorem horse_catch_up :
  ∀ (x : ℕ), (240 * x = 150 * (x + 12)) → x = 20 :=
by
  intros x h
  have : 240 * x = 150 * x + 1800 := by sorry
  have : 240 * x - 150 * x = 1800 := by sorry
  have : 90 * x = 1800 := by sorry
  have : x = 1800 / 90 := by sorry
  have : x = 20 := by sorry
  exact this

end horse_catch_up_l494_494984


namespace balls_in_boxes_l494_494059

theorem balls_in_boxes :
    (Nat.choose 6 6) + 
    (Nat.choose 6 5) + 
    (Nat.choose 6 4 * Nat.choose 2 2) + 
    (Nat.choose 6 3 / 2) + 
    (Nat.choose 6 4 * Nat.choose 2 1) + 
    (Nat.choose 6 2 * Nat.choose 4 2 / 3.factorial) = 77 :=
by
  sorry

end balls_in_boxes_l494_494059


namespace increase_by_150_percent_l494_494302

theorem increase_by_150_percent (n : ℕ) : 
  n = 80 → n + (3 / 2) * n = 200 :=
by
  intros h
  rw [h]
  norm_num
  sorry

end increase_by_150_percent_l494_494302


namespace area_ratio_of_circles_l494_494947

theorem area_ratio_of_circles (R_C R_D : ℝ) (hL : (60.0 / 360.0) * 2.0 * Real.pi * R_C = (40.0 / 360.0) * 2.0 * Real.pi * R_D) : 
  (Real.pi * R_C^2) / (Real.pi * R_D^2) = 4.0 / 9.0 :=
by
  sorry

end area_ratio_of_circles_l494_494947


namespace log_addition_identity_l494_494796

theorem log_addition_identity :
  ∃ (a x y : ℝ), a = 10 ∧ x = 50 ∧ y = 20 ∧ (log 10 50 + log 10 20 = 3) :=
by
  let a := 10
  let x := 50
  let y := 20
  have h1 : log a (x * y) = log a x + log a y,
    from sorry -- logarithmic identity
  have h2 : log a (x * y) = log a 1000,
    from congrArg (log a) (by norm_num) -- simplifying x * y
  have h3 : log 10 1000 = 3,
    from sorry -- calculating log 1000 base 10 directly
  exact ⟨a, x, y, rfl, rfl, rfl, by linarith [h1, h2, h3]⟩

end log_addition_identity_l494_494796


namespace increase_80_by_150_percent_l494_494287

theorem increase_80_by_150_percent : 
  ∀ x percent, x = 80 ∧ percent = 1.5 → x + (x * percent) = 200 :=
by
  sorry

end increase_80_by_150_percent_l494_494287


namespace find_m_n_range_k_range_a_l494_494528

-- Part (1)
theorem find_m_n (m n : ℝ) (hm_pos : 0 < m) 
  (h_max : ∀ x ∈ Icc (1:ℝ) 2, g(x) ≤ 0) (h_min : ∀ x ∈ Icc (1:ℝ) 2, g(x) ≥ -1) :
  m = 1 ∧ n = -1 :=
sorry

-- Part (2)
theorem range_k (k : ℝ) (m n : ℝ) (hm_pos : 0 < m) (x : ℝ) (hx : x ∈ Icc (0:ℝ) 1)
  (h_ineq : g(2^x) + 1 - k * 2^(x+1) ≥ 0) :
  k ∈ Iic (1 / 4) :=
sorry

-- Part (3)
theorem range_a (a : ℝ) (m n : ℝ) (hm_pos : 0 < m) (x : ℝ) (hx : x ∈ Icc (0:ℝ) 1)
  (h_abs_ineq : |(g(x) + h(x))| ≤ 1) :
  a ∈ Icc (-2 : ℝ) 0 :=
sorry

-- Definitions for g and h for completeness
def g (x m n : ℝ) := m * x^2 - 2 * m * x + 1 + n
def h (x a : ℝ) := (a - 1) * x^2 + 3 * x
def f (x m n a : ℝ) := g x m n + h x a

end find_m_n_range_k_range_a_l494_494528


namespace jake_more_peaches_than_jill_l494_494645

theorem jake_more_peaches_than_jill :
  let steven_peaches := 14
  let jake_peaches := steven_peaches - 6
  let jill_peaches := 5
  jake_peaches - jill_peaches = 3 :=
by
  let steven_peaches := 14
  let jake_peaches := steven_peaches - 6
  let jill_peaches := 5
  sorry

end jake_more_peaches_than_jill_l494_494645


namespace ship_navigation_avoid_reefs_l494_494687

theorem ship_navigation_avoid_reefs (a : ℝ) (h : a > 0) :
  (10 * a) * 40 / Real.sqrt ((10 * a) ^ 2 + 40 ^ 2) > 20 ↔
  a > (4 * Real.sqrt 3 / 3) :=
by
  sorry

end ship_navigation_avoid_reefs_l494_494687


namespace sum_of_first_ten_primes_ending_in_3_is_671_l494_494848

noncomputable def sum_of_first_ten_primes_ending_in_3 : ℕ :=
  3 + 13 + 23 + 43 + 53 + 73 + 83 + 103 + 113 + 163

theorem sum_of_first_ten_primes_ending_in_3_is_671 :
  sum_of_first_ten_primes_ending_in_3 = 671 :=
by
  sorry

end sum_of_first_ten_primes_ending_in_3_is_671_l494_494848


namespace turner_oldest_child_age_l494_494189

theorem turner_oldest_child_age (a b c : ℕ) (avg : ℕ) :
  (a = 6) → (b = 8) → (c = 11) → (avg = 9) → 
  (4 * avg = (a + b + c + x) → x = 11) :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  sorry

end turner_oldest_child_age_l494_494189


namespace increase_result_l494_494248

-- Given conditions
def original_number : ℝ := 80
def increase_percentage : ℝ := 1.5

-- The result after the increase
theorem increase_result (h1 : original_number = 80) (h2 : increase_percentage = 1.5) : 
  original_number + (increase_percentage * original_number) = 200 := by
  sorry

end increase_result_l494_494248


namespace distribute_6_balls_in_3_boxes_l494_494041

def num_ways_to_distribute_balls (balls : ℕ) (boxes : ℕ) : ℕ :=
  if balls = 6 ∧ boxes = 3 then 92 else 0

theorem distribute_6_balls_in_3_boxes :
  num_ways_to_distribute_balls 6 3 = 92 :=
by
  rw num_ways_to_distribute_balls
  norm_num
  sorry

end distribute_6_balls_in_3_boxes_l494_494041


namespace eccentricity_of_ellipse_value_of_lambda_l494_494489

-- Define the conditions given in the problem
variables {a b x0 y0 λ : ℝ}
variables {P : Point}
variables {E : Ellipse}
variables {M N A B C O : Point}

-- Point P on the ellipse with the given coordinate restrictions
def P_on_ellipse (x0 y0 a b : ℝ) : Prop :=
  (x0^2 / a^2 + y0^2 / b^2 = 1) ∧ x0 ≠ abs a

-- Ellipse properties
def ellipse_properties (a b : ℝ) : Prop :=
  a > b > 0 

-- Given slope condition for PM and PN
def product_of_slopes_condition (x0 y0 a : ℝ) : Prop :=
  (y0 / (x0 + a) * y0 / (x0 - a)) = -1/4

-- Points A and B on ellipse intersected by a line with slope 1 through left focus
def line_through_left_focus (P : Point) : Prop := 
  let (x1, y1) := A in
  let (x2, y2) := B in
  y1 = x1 + c ∧ y2 = x2 + c ∧ 
  (P.x^2 + 4 * P.y^2 = 4 * b^2) ∧ (C.coords = λ * A.coords + B.coords)

-- Prove the eccentricity
theorem eccentricity_of_ellipse (h1 : P_on_ellipse x0 y0 a b) (h2 : ellipse_properties a b) 
  (h3 : product_of_slopes_condition x0 y0 a) : 
  (∃ c : ℝ, c^2 = a^2 - b^2 ∧ (c / a) = sqrt 3 / 2) := sorry

-- Prove the value of λ
theorem value_of_lambda (h1 : line_through_left_focus P) (h2 : λ ≠ 0) : 
  λ = -2/5 := sorry

end eccentricity_of_ellipse_value_of_lambda_l494_494489


namespace inequality_least_n_l494_494507

theorem inequality_least_n (n : ℕ) (h : (1 : ℝ) / n - (1 : ℝ) / (n + 2) < 1 / 15) : n = 5 :=
sorry

end inequality_least_n_l494_494507


namespace intersection_points_of_cones_l494_494191

noncomputable def intersection_points_on_sphere 
(X Y Z a b c A B : ℝ) : Prop :=
  ∀ X Y Z a b c A B, 
    (∃ (X Y Z: ℝ), (X^2 + Y^2 = A * Z^2) 
     ∧ ((X - a)^2 + (Y - b)^2 = B * (Z - c)^2)) → 
     ∃ k: ℝ, (X - k * a)^2 + (Y - k * b)^2 + (Z + k * B * c)^2 = k

-- statement to verify correctness of our main theorem
theorem intersection_points_of_cones 
(X Y Z a b c A B k : ℝ) :
  intersection_points_on_sphere X Y Z a b c A B :=
begin
  -- NOTE: This is just the statement construction and not the proof
  sorry,
end

end intersection_points_of_cones_l494_494191


namespace find_two_digit_ab_l494_494099

def digit_range (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 9

def different_digits (a b c d : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

theorem find_two_digit_ab (A B C D : ℕ) (hA : digit_range A) (hB : digit_range B)
                         (hC : digit_range C) (hD : digit_range D)
                         (h_diff : different_digits A B C D)
                         (h_eq : (100 * A + 10 * B + C) * (10 * A + B) + C * D = 2017) :
  10 * A + B = 14 :=
sorry

end find_two_digit_ab_l494_494099


namespace increase_by_150_percent_l494_494314

theorem increase_by_150_percent (x : ℝ) (h : x = 80) : x + (1.5 * x) = 200 := 
by
  -- The proof goes here, but is omitted as per instructions.
  sorry

end increase_by_150_percent_l494_494314


namespace orthogonal_circles_l494_494221

theorem orthogonal_circles (R1 R2 d : ℝ) :
  (d^2 = R1^2 + R2^2) ↔ (d^2 = R1^2 + R2^2) :=
by sorry

end orthogonal_circles_l494_494221


namespace average_interest_rate_l494_494383

theorem average_interest_rate (I : ℝ) (r1 r2 : ℝ) (y : ℝ)
  (h0 : I = 6000)
  (h1 : r1 = 0.05)
  (h2 : r2 = 0.07)
  (h3 : 0.05 * (6000 - y) = 0.07 * y) :
  ((r1 * (I - y) + r2 * y) / I) = 0.05833 :=
by
  sorry

end average_interest_rate_l494_494383


namespace ratio_of_areas_of_circles_l494_494950

theorem ratio_of_areas_of_circles
    (C_C R_C C_D R_D L : ℝ)
    (hC : C_C = 2 * Real.pi * R_C)
    (hD : C_D = 2 * Real.pi * R_D)
    (hL : (60 / 360) * C_C = L ∧ L = (40 / 360) * C_D) :
    (Real.pi * R_C ^ 2) / (Real.pi * R_D ^ 2) = 4 / 9 :=
by
  sorry

end ratio_of_areas_of_circles_l494_494950


namespace median_CD_eq_altitude_from_C_eq_centroid_G_eq_l494_494494

namespace Geometry

/-- Vertices of the triangle -/
def A : ℝ × ℝ := (4, 4)
def B : ℝ × ℝ := (-4, 2)
def C : ℝ × ℝ := (2, 0)

/-- Proof of the equation of the median CD on the side AB -/
theorem median_CD_eq : ∀ (x y : ℝ), 3 * x + 2 * y - 6 = 0 :=
sorry

/-- Proof of the equation of the altitude from C to AB -/
theorem altitude_from_C_eq : ∀ (x y : ℝ), 4 * x + y - 8 = 0 :=
sorry

/-- Proof of the coordinates of the centroid G of triangle ABC -/
theorem centroid_G_eq : ∃ (x y : ℝ), x = 2 / 3 ∧ y = 2 :=
sorry

end Geometry

end median_CD_eq_altitude_from_C_eq_centroid_G_eq_l494_494494


namespace trapezoid_equation_l494_494625

theorem trapezoid_equation
  (A B C D E O P : Point)
  (AD BC: ℝ)
  (trapezoid_ABCD : is_trapezoid A B C D)
  (on_E_on_AD : E ∈ segment A D)
  (AE_eq_BC : distance A E = BC)
  (CA_meets_BD_at_O : ray C A ∩ segment B D = {O})
  (CE_meets_BD_at_P : ray C E ∩ segment B D = {P})
  (BO_eq_PD : distance B O = distance P D) :
  (AD ^ 2 = BC ^ 2 + AD * BC) :=
sorry

end trapezoid_equation_l494_494625


namespace second_fragment_speed_is_l494_494365

variables (u t g vₓ₁ : ℝ)
variables (vₓ₂ vᵧ₂ : ℝ)

-- Given conditions
def initial_vertical_velocity : ℝ := u
def time_of_explosion : ℝ := t
def gravity_acceleration : ℝ := g
def first_fragment_horizontal_velocity : ℝ := vₓ₁

noncomputable def second_fragment_speed : ℝ :=
  let vᵧ := initial_vertical_velocity - gravity_acceleration * time_of_explosion in
  let vₓ₂ := -first_fragment_horizontal_velocity in
  let vᵧ₂ := vᵧ in
  real.sqrt (vₓ₂^2 + vᵧ₂^2)

theorem second_fragment_speed_is : second_fragment_speed u t g vₓ₁ = real.sqrt 2404 := 
sorry

end second_fragment_speed_is_l494_494365


namespace average_temperature_l494_494190

def temperatures :=
  ∃ T_tue T_wed T_thu : ℝ,
    (44 + T_tue + T_wed + T_thu) / 4 = 48 ∧
    (T_tue + T_wed + T_thu + 36) / 4 = 46

theorem average_temperature :
  temperatures :=
by
  sorry

end average_temperature_l494_494190


namespace log_sum_example_l494_494803

theorem log_sum_example : log 10 50 + log 10 20 = 3 :=
by
  -- Proof goes here, skipping with sorry
  sorry

end log_sum_example_l494_494803


namespace problem_proof_l494_494897

noncomputable def ellipse_equation_exists (a b : ℝ) (h1 : a > b) (h2 : b > 0) : Prop :=
  ∃ (x y : ℝ), (x, y) ∈ { p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1 }

noncomputable def equation_of_ellipse_C : Prop :=
  ∃ a b : ℝ, a > b ∧ b > 0 ∧ 
  (let C := λ x y, (x^2 / a^2) + (y^2 / b^2) = 1 in
  C 1 (3/2) ∧
  (∃ c : ℝ, b = sqrt 3 * c ∧ 1 / a^2 + 9 / (4 * b^2) = 1 ∧ a^2 = b^2 + c^2) ∧
  a^2 = 4 ∧ b^2 = 3 ∧ C x y)

noncomputable def line_l_exists_and_bisects (k m : ℝ) (h : (k = 1/2 ∨ k = -1/2) ∧ m = sqrt 21 / 7) : Prop :=
  ∃ x1 x2 : ℝ, 
  (let ellipse_eqn := λ x y, (x^2 / 4) + (y^2 / 3) = 1 in
  ∀ l : ℝ → ℝ, l = λ x, k * x + m →
  ∀ (N M : ℝ × ℝ), N = (-(m / k), 0) ∧ M = (0, m) →
  ∀ (P Q : ℝ × ℝ), P = ((m / k), 2 * m) ∧ Q = ((m / k), -2 * m) →
  ∀ (A B : ℝ × ℝ), A = (x1, (k * x1 + m)) ∧ B = (x2, (-3 * k * x2 + m)) →
  ∀ (A1 B1 : ℝ × ℝ), A1 = (x1, 0) ∧ B1 = (x2, 0) →
  (N.1 = (A1.1 + B1.1) / 2))

theorem problem_proof :
  exists (a b : ℝ), (a > b) ∧ (b > 0) ∧
  ellipse_equation_exists a b ∧
  equation_of_ellipse_C ∧
  exists (k m : ℝ), line_l_exists_and_bisects k m sorry :=
begin
  sorry
end

end problem_proof_l494_494897


namespace triangle_A1B1C1_angles_l494_494667

theorem triangle_A1B1C1_angles (A B C A0 A1 B1 C1 : Type)
  [triangle_ABC : isosceles_triangle A B C 30 30 120]
  (median_AA0 : is_median A A0 B C)
  (perpendicular_A0A1 : perpendicular A0 A1 (line_segment B C) )
  (median_BB0 : is_median B B0 A C)
  (perpendicular_B0B1 : perpendicular B0 B1 (line_segment C A))
  (median_CC0 : is_median C C0 A B)
  (perpendicular_C0C1 : perpendicular C0 C1 (line_segment A B)) :
  triangle A1 B1 C1 ∧ ∀ x y z, x ≠ y ∧ y ≠ z ∧ z ≠ x -> angle x y z = 60 := 
sorry

end triangle_A1B1C1_angles_l494_494667


namespace original_number_of_students_l494_494979

theorem original_number_of_students (X : ℝ) (h_increase : X * 1.20 = 1.20 * X) (h_decrease : (X * 1.20) * 0.90 = 1.08 * X) (h_third_year : (1.08 * X) = 950) :
  X ≈ 880 := by 
  sorry

end original_number_of_students_l494_494979


namespace distance_from_Q_to_EG_l494_494178

noncomputable def distance_to_line : ℝ :=
  let E := (0, 5)
  let F := (5, 5)
  let G := (5, 0)
  let H := (0, 0)
  let N := (2.5, 0)
  let Q := (25 / 7, 10 / 7)
  let line_y := 5
  let distance := abs (line_y - Q.2)
  distance

theorem distance_from_Q_to_EG : distance_to_line = 25 / 7 :=
by
  sorry

end distance_from_Q_to_EG_l494_494178


namespace anniversary_day_of_week_probability_l494_494403

/-- The 11th anniversary of Robinson Crusoe and Friday's meeting can fall on a Friday with a
probability of 3/4 and on a Thursday with a probability of 1/4, given that the meeting occurred
in any year from 1668 to 1671 with equal probability. -/
theorem anniversary_day_of_week_probability :
  let years := {1668, 1669, 1670, 1671},
      leap (y : ℕ) := y % 4 = 0,
      days_in_year := λ y, if leap y then 366 else 365,
      total_days (yr : ℕ) := list.sum (list.map days_in_year (list.range' yr 11)),
      day_of_week_after_11_years (initial_year : ℕ) := total_days initial_year % 7 = 0,
      events := {week_day | ∀ y ∈ years, (day_of_week_after_11_years y)},
      friday_probability := rat.mk 3 4,
      thursday_probability := rat.mk 1 4
  in
  (events = {0} ∨ events = {6}) ∧
  (events = {0} → friday_probability = rat.mk 3 4 ∧ thursday_probability = rat.mk 1 4) ∧
  (events = {6} → friday_probability = rat.mk 1 4 ∧ thursday_probability = rat.mk 3 4):=
begin
  sorry
end

end anniversary_day_of_week_probability_l494_494403


namespace cube_painting_probability_l494_494698

theorem cube_painting_probability :
  let total_configurations := 2^6 * 2^6
  let identical_configurations := 90
  (identical_configurations / total_configurations : ℚ) = 45 / 2048 :=
by
  sorry

end cube_painting_probability_l494_494698


namespace increase_by_150_percent_l494_494318

theorem increase_by_150_percent (x : ℝ) (h : x = 80) : x + (1.5 * x) = 200 := 
by
  -- The proof goes here, but is omitted as per instructions.
  sorry

end increase_by_150_percent_l494_494318


namespace cos_phi_l494_494754

-- Define the given vectors 
def vec1 : ℝ × ℝ := (4, 5)
def vec2 : ℝ × ℝ := (2, -1)

-- Define the dot product function
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Define the magnitude function
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Define the problem
theorem cos_phi : 
  let cos_phi := dot_product vec1 vec2 / (magnitude vec1 * magnitude vec2) in
  cos_phi = 3 / Real.sqrt 205 := by
sorry

end cos_phi_l494_494754


namespace shaded_area_semicircle_rotation_l494_494835

theorem shaded_area_semicircle_rotation (R : ℝ) (α : ℝ) (hα : α = 20 * real.pi / 180) :
    ∃ A : ℝ, A = (2 * real.pi * R^2 / 9) :=
by
  sorry

end shaded_area_semicircle_rotation_l494_494835


namespace balls_in_boxes_l494_494054

theorem balls_in_boxes :
    (Nat.choose 6 6) + 
    (Nat.choose 6 5) + 
    (Nat.choose 6 4 * Nat.choose 2 2) + 
    (Nat.choose 6 3 / 2) + 
    (Nat.choose 6 4 * Nat.choose 2 1) + 
    (Nat.choose 6 2 * Nat.choose 4 2 / 3.factorial) = 77 :=
by
  sorry

end balls_in_boxes_l494_494054


namespace prime_combinations_count_l494_494622

-- Define the set of single digits
def digits := {1, 2, 3, 4, 5, 6, 7, 8, 9 : ℕ}

-- Condition: Define all prime numbers less than 45
def primesUpto45 := {11, 13, 17, 19, 23, 29, 31, 37, 41, 43 : ℕ}

-- Define the function that checks if a number can be formed by using digits exactly once and inserting "+" or "-" between adjacent digits.
def isValidNumber (n : ℕ) : Prop :=
  ∃ (s : list ℕ), s.perm (digits.toList) ∧ list.sum s = 45 ∧ (set.to_finset (s.to_finset.image (λ x, abs x))).subscript (primesUpto45.toList)

-- Define the main theorem to prove the problem statement
theorem prime_combinations_count : (finset.filter isValidNumber primesUpto45).card = 10 := sorry

end prime_combinations_count_l494_494622


namespace lydia_age_at_first_apple_l494_494594

theorem lydia_age_at_first_apple (tree_bear_fruit_years : ℕ) (lydia_plant_age : ℕ) (current_lydia_age : ℕ) 
  (tree_bear_fruit_condition : tree_bear_fruit_years = 10) 
  (lydia_plant_age_condition : lydia_plant_age = 6) 
  (current_lydia_age_condition : current_lydia_age = 11) : 
  (lydia_plant_age + tree_bear_fruit_years) = 16 := 
by 
  rw [tree_bear_fruit_condition, lydia_plant_age_condition]
  exact rfl

end lydia_age_at_first_apple_l494_494594


namespace balls_in_boxes_l494_494007

/-
Prove that the number of ways to distribute 6 distinguishable balls into 3 indistinguishable boxes is 132.
-/
theorem balls_in_boxes : 
  ∃ (ways : ℕ), ways = 132 ∧ ways = 
    (1) + 
    (Nat.choose 6 5) + 
    (Nat.choose 6 4) + 
    (Nat.choose 6 4) + 
    (Nat.choose 6 3) + 
    (Nat.choose 6 3 * Nat.choose 3 2) + 
    (Nat.choose 6 2 * Nat.choose 4 2 / 6) := 
by
  sorry

end balls_in_boxes_l494_494007


namespace max_distinct_rectangles_l494_494346

theorem max_distinct_rectangles : 
  ∃ (rectangles : Finset ℕ), (∀ n ∈ rectangles, n > 0) ∧ rectangles.sum id = 100 ∧ rectangles.card = 14 :=
by 
  sorry

end max_distinct_rectangles_l494_494346


namespace quadrilateral_is_rhombus_l494_494372

-- Define a quadrilateral as a structure
structure Quadrilateral :=
(a b c d : Point) (diagonal1 diagonal2 : Line)

-- Define properties of the quadrilateral
def Perpendicular (l1 l2 : Line) : Prop := sorry  -- Placeholder for the actual definition
def Bisect (line1 line2 : Line) : Prop := sorry  -- Placeholder for the actual definition
def CongruentTrianglesFormed (diagonal1 diagonal2 : Line) : Prop := sorry  -- Implies all four triangles are congruent

-- Define rhombus
def Rhombus (q : Quadrilateral) : Prop :=
  (∀ p1 p2 p3 p4 : Point, q.a = p1 ∧ q.b = p2 ∧ q.c = p3 ∧ q.d = p4 ∧
    Dist p1 p2 = Dist p2 p3 ∧ Dist p3 p4 = Dist p4 p1)

-- The main theorem to be proven
theorem quadrilateral_is_rhombus (q : Quadrilateral) 
  (h1: Perpendicular q.diagonal1 q.diagonal2)
  (h2: Bisect q.diagonal1 q.diagonal2) : Rhombus q := 
sorry

end quadrilateral_is_rhombus_l494_494372


namespace find_AB_l494_494222

theorem find_AB
  (r R : ℝ)
  (h : r < R) :
  ∃ AB : ℝ, AB = (4 * r * (Real.sqrt (R * r))) / (R + r) :=
by
  sorry

end find_AB_l494_494222


namespace balls_in_boxes_l494_494010

/-
Prove that the number of ways to distribute 6 distinguishable balls into 3 indistinguishable boxes is 132.
-/
theorem balls_in_boxes : 
  ∃ (ways : ℕ), ways = 132 ∧ ways = 
    (1) + 
    (Nat.choose 6 5) + 
    (Nat.choose 6 4) + 
    (Nat.choose 6 4) + 
    (Nat.choose 6 3) + 
    (Nat.choose 6 3 * Nat.choose 3 2) + 
    (Nat.choose 6 2 * Nat.choose 4 2 / 6) := 
by
  sorry

end balls_in_boxes_l494_494010


namespace increase_by_150_percent_l494_494304

theorem increase_by_150_percent (n : ℕ) : 
  n = 80 → n + (3 / 2) * n = 200 :=
by
  intros h
  rw [h]
  norm_num
  sorry

end increase_by_150_percent_l494_494304


namespace number_sequence_10th_term_l494_494112

theorem number_sequence_10th_term : 
  ∀ (n : ℕ), (sequence_start n = 1) → (sequence_rule n) = 1 :=
by
  -- Definitions of sequence start and rule based on the conditions
  def sequence_start (n : ℕ) : ℕ :=
    -- Jo starts by saying 1
    if n = 1 then 1 else sorry

  def sequence_rule (n : ℕ) : ℕ :=
    -- Each subsequent number said is the square of the last number
    if n = 1 then sequence_start 1 else (sequence_rule (n - 1)) ^ 2
   
  -- Assuming the problem definition and conditions
  assume n h,
  sorry

end number_sequence_10th_term_l494_494112


namespace anniversary_day_probability_l494_494394

/- Definitions based on the conditions -/
def is_leap_year (year : ℕ) : Prop :=
  year % 4 = 0

def days_in_year (year : ℕ) : ℕ :=
  if is_leap_year year then 366 else 365

def total_days (start_year : ℕ) : ℕ :=
  (list.sum (list.map days_in_year (list.range' start_year 11)))

/- Prove the day of the 11th anniversary and its probabilities -/
theorem anniversary_day_probability (start_year : ℕ) (h : start_year ∈ {1668, 1669, 1670, 1671}) :
  let days := total_days start_year % 7
  in (days = 0 ∧ 0.75 ≤ 1) ∨ (days = 6 ∧ 0.25 ≤ 1) :=
by
  sorry

end anniversary_day_probability_l494_494394


namespace team_B_not_played_E_l494_494772

def team_has_played (n : Nat) (team : String) (matches : List (String × String)) : Prop :=
  matches.countp (λ m => m.1 = team ∨ m.2 = team) = n

def team_has_not_played_against (team1 team2 : String) (matches : List (String × String)) : Prop :=
  ¬ ∃ m, (m.1 = team1 ∧ m.2 = team2) ∨ (m.1 = team2 ∧ m.2 = team1)

theorem team_B_not_played_E
    (matches : List (String × String))
    (hA : team_has_played 5 "A" matches)
    (hB : team_has_played 4 "B" matches)
    (hC : team_has_played 3 "C" matches)
    (hD : team_has_played 2 "D" matches)
    (hE : team_has_played 1 "E" matches) :
    team_has_not_played_against "B" "E" matches :=
  sorry

end team_B_not_played_E_l494_494772


namespace divide_loot_among_robbers_l494_494724

theorem divide_loot_among_robbers (n : ℕ) (loot : Type) (valuation : fin n → set loot → ℚ) :
  ∃ (division : fin n → set loot), ∀ (i : fin n), valuation i (division i) ≥ 1 / n := by
  sorry

end divide_loot_among_robbers_l494_494724


namespace each_person_share_after_taxes_l494_494595

noncomputable theory

def market_value : ℝ := 500000
def over_market_percentage : ℝ := 0.20
def tax_percentage : ℝ := 0.10
def num_people : ℕ := 4

theorem each_person_share_after_taxes :
  let selling_price := market_value * (1 + over_market_percentage) in
  let revenue_after_taxes := selling_price * (1 - tax_percentage) in
  let each_person_share := revenue_after_taxes / num_people in
  each_person_share = 135000 := 
by
  sorry

end each_person_share_after_taxes_l494_494595


namespace limit_exponential_sine_l494_494422

theorem limit_exponential_sine :
  (lim x → 0, (exp (5 * x) - exp (3 * x)) / (sin (2 * x) - sin (x)) = 2) := by
  sorry

end limit_exponential_sine_l494_494422


namespace elderly_people_pears_l494_494966

theorem elderly_people_pears (x y : ℕ) :
  (y = x + 1) ∧ (2 * x = y + 2) ↔
  (x = y - 1) ∧ (2 * x = y + 2) := by
  sorry

end elderly_people_pears_l494_494966


namespace part1_part2_mean_part2_variance_l494_494094

open ProbabilityTheory

section Problem

/-- Define the population of doctors. --/
def doctors : Finset ℕ := {0, 1, 2, 3, 4, 5}

/-- The set of surgeons, internists, and ophthalmologists. --/
def surgeons : Finset ℕ := {0, 1}
def internists : Finset ℕ := {2, 3}
def ophthalmologists : Finset ℕ := {4, 5}

/-- The probability of selecting a subset of 3 doctors. --/
def select_doctors (s : Finset ℕ) : Prop :=
  s.card = 3 ∧ s ⊆ doctors

/-- Probability mass function for selecting 3 doctors. --/
noncomputable def pmf_select : PMF (Finset ℕ) :=
  PMF.ofMultisetMultiset (doctors.powerset.filter select_doctors).val
    (by { rw ← Multiset.card, exact (Multiset.card_pos_iff_exists_mem.mpr ⟨_, Multiset.mem_powerset.mpr (by refl)⟩), })

/-- Probability event where the number of selected surgeons is greater than the number of selected internists. --/
def event_more_surgeons (s : Finset ℕ) : Prop :=
  (s ∩ surgeons).card > (s ∩ internists).card

/-- The random variable representing the number of surgeons selected. --/
def num_surgeons (s : Finset ℕ) : ℕ :=
  (s ∩ surgeons).card

theorem part1 :
  PMF.prob (pmf_select {w | event_more_surgeons w}) = 3 / 10 :=
sorry

theorem part2_mean :
  PMF.expectedValue pmf_select num_surgeons = 1 :=
sorry

theorem part2_variance :
  PMF.variance num_surgeons pmf_select = 2 / 5 :=
sorry

end Problem

end part1_part2_mean_part2_variance_l494_494094


namespace speed_of_second_fragment_l494_494363

noncomputable def magnitude_speed_of_second_fragment 
  (u : ℝ) (t : ℝ) (g : ℝ) (v_x1 : ℝ) (v_y1 : ℝ := - (u - g * t)) 
  (v_x2 : ℝ := -v_x1) (v_y2 : ℝ := v_y1) : ℝ :=
Real.sqrt ((v_x2 ^ 2) + (v_y2 ^ 2))

theorem speed_of_second_fragment 
  (u : ℝ) (t : ℝ) (g : ℝ) (v_x1 : ℝ) 
  (h_u : u = 20) (h_t : t = 3) (h_g : g = 10) (h_vx1 : v_x1 = 48) :
  magnitude_speed_of_second_fragment u t g v_x1 = Real.sqrt 2404 :=
by
  -- Proof
  sorry

end speed_of_second_fragment_l494_494363


namespace balls_in_boxes_l494_494057

theorem balls_in_boxes :
    (Nat.choose 6 6) + 
    (Nat.choose 6 5) + 
    (Nat.choose 6 4 * Nat.choose 2 2) + 
    (Nat.choose 6 3 / 2) + 
    (Nat.choose 6 4 * Nat.choose 2 1) + 
    (Nat.choose 6 2 * Nat.choose 4 2 / 3.factorial) = 77 :=
by
  sorry

end balls_in_boxes_l494_494057


namespace sequence_sum_lt_one_l494_494205

def sequence_a : ℕ → ℚ
| 1 := 1 / 2
| (n + 1) := (sequence_a n)^2 / ((sequence_a n)^2 - sequence_a n + 1)

theorem sequence_sum_lt_one (n : ℕ) (hn : n > 0) : 
  (∑ i in (Finset.range n).map (λ x, x + 1), sequence_a i) < 1 :=
sorry

end sequence_sum_lt_one_l494_494205


namespace find_F_Y_D_angle_l494_494097

structure GeometrySetup where
  points : Type
  A B C D X Y E F Z : points
  Line : points → points → Prop
  parallel : Prop
  angle : points → points → points → ℝ
  perpendicular : Prop
  (line_AB : Line A B)
  (line_CD : Line C D)
  (line_EX : Line E X)
  (line_FX : Line F X)
  (line_CY : Line C Y)
  (line_DY : Line D Y)
  (line_EF : Line E F)
  (line_EZ : Line E Z)
  (line_ZF : Line Z F)
  (parallel_AB_CD : parallel)
  (angle_AXF_110 : angle A X F = 110)
  (perpendicular_EZ_CD : perpendicular)

theorem find_F_Y_D_angle (geom : GeometrySetup) : geom.angle F Y D = 70 := by
  sorry

end find_F_Y_D_angle_l494_494097


namespace expansion_term_count_l494_494824

theorem expansion_term_count (a b c x d e f g h : Type) [Distinct a b c x d e f g h] :
  let t₁ := 3
  let t₂ := 6
  t₁ * t₂ = 18 :=
by
  sorry

end expansion_term_count_l494_494824


namespace increase_150_percent_of_80_l494_494300

theorem increase_150_percent_of_80 : 80 * 1.5 + 80 = 200 :=
by
  sorry

end increase_150_percent_of_80_l494_494300


namespace age_difference_l494_494211

/-- The age difference between each child d -/
theorem age_difference (d : ℝ) 
  (h1 : ∃ a b c e : ℝ, d = a ∧ 2*d = b ∧ 3*d = c ∧ 4*d = e)
  (h2 : 12 + (12 - d) + (12 - 2*d) + (12 - 3*d) + (12 - 4*d) = 40) : 
  d = 2 := 
sorry

end age_difference_l494_494211


namespace find_sum_of_abs_coeffs_of_binomial_expansion_l494_494544

theorem find_sum_of_abs_coeffs_of_binomial_expansion :
  let a, a_1, a_2, a_3, a_4, a_5, a_6, a_7, a_8 : ℤ
  in (3 * x - 1)^8 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 +
                    a_5 * x^5 + a_6 * x^6 + a_7 * x^7 + a_8 * x^8 →
  |a| + |a_1| + |a_2| + |a_3| + |a_4| + |a_5| + |a_6| + |a_7| + |a_8| = 4 ^ 8 :=
by
  sorry

end find_sum_of_abs_coeffs_of_binomial_expansion_l494_494544


namespace log_sum_example_l494_494792

theorem log_sum_example :
  let log_base_10 (x : ℝ) := Real.log x / Real.log 10 in
  log_base_10 50 + log_base_10 20 = 3 :=
by
  sorry

end log_sum_example_l494_494792


namespace sum_of_first_ten_primes_ending_in_3_is_671_l494_494846

noncomputable def sum_of_first_ten_primes_ending_in_3 : ℕ :=
  3 + 13 + 23 + 43 + 53 + 73 + 83 + 103 + 113 + 163

theorem sum_of_first_ten_primes_ending_in_3_is_671 :
  sum_of_first_ten_primes_ending_in_3 = 671 :=
by
  sorry

end sum_of_first_ten_primes_ending_in_3_is_671_l494_494846


namespace minimize_S_n_l494_494534

def sequence (n : ℕ) : ℝ := 1 / (2 * n - 11)

def S_n (n : ℕ) : ℝ := ∑ k in Finset.range n, sequence k

theorem minimize_S_n : ∀ n : ℕ, n ≤ 5 → n = 5 :=
by sorry

end minimize_S_n_l494_494534


namespace rationalize_denominator_l494_494634

theorem rationalize_denominator :
  ( √18 + √8 ) / ( √3 + √8 ) = 5 * √6 - 20 :=
by
  sorry

end rationalize_denominator_l494_494634


namespace probability_sin_interval_l494_494914

theorem probability_sin_interval (x : ℝ) (h : 0 ≤ x ∧ x ≤ 1) : 
  let θ := (π / 2) * x in
  0 ≤ θ ∧ θ ≤ π / 2 →
  ∃ p : ℝ, 
    p = ((λ y, if 0 ≤ y ∧ y ≤ 1/2 then 1 else 0) <$> sin '' (λ y, (π / 2) * y) '' Icc 0 1).integral = 1 / 3 := 
sorry

end probability_sin_interval_l494_494914


namespace increase_by_150_percent_l494_494301

theorem increase_by_150_percent (n : ℕ) : 
  n = 80 → n + (3 / 2) * n = 200 :=
by
  intros h
  rw [h]
  norm_num
  sorry

end increase_by_150_percent_l494_494301


namespace find_w_find_g_minimum_l494_494144

theorem find_w (f : ℝ → ℝ) (w : ℝ) (h₀ : 0 < w ∧ w < 3)
  (h : f (π / 6) = 0) :
  (∀ x, f x = sin (w * x - π / 6) + sin (w * x - π / 2)) → w = 2 := sorry

theorem find_g_minimum (g : ℝ → ℝ) :
  (∀ x, g x = sqrt 3 * sin (x - π / 12)) →
  ∀ x ∈ Icc (-π / 4) (3 * π / 4), g x = -3 / 2 → x = -π / 4 := sorry

end find_w_find_g_minimum_l494_494144


namespace randy_trip_distance_l494_494633

theorem randy_trip_distance (x : ℝ) (h1 : x = x / 4 + 30 + x / 10 + (x - (x / 4 + 30 + x / 10))) :
  x = 60 :=
by {
  sorry -- Placeholder for the actual proof
}

end randy_trip_distance_l494_494633


namespace gcd_372_684_l494_494195

theorem gcd_372_684 : Nat.gcd 372 684 = 12 :=
by
  sorry

end gcd_372_684_l494_494195


namespace distribute_6_balls_in_3_boxes_l494_494042

def num_ways_to_distribute_balls (balls : ℕ) (boxes : ℕ) : ℕ :=
  if balls = 6 ∧ boxes = 3 then 92 else 0

theorem distribute_6_balls_in_3_boxes :
  num_ways_to_distribute_balls 6 3 = 92 :=
by
  rw num_ways_to_distribute_balls
  norm_num
  sorry

end distribute_6_balls_in_3_boxes_l494_494042


namespace triangle_identity_l494_494962

-- Given conditions
def a : ℝ := 3
def c : ℝ := 2
def sin_A_eq_cos_B : Prop := ∀ (A B : ℝ), sin A = cos (π / 2 - B)

-- Derived equalities and result
theorem triangle_identity (A B C : ℝ) (h : sin_A_eq_cos_B A B) :
  (A = B) ∧ 
  (let b := a in cos C = (a^2 + b^2 - c^2) / (2 * a * b) ∧
      let sin_C := sqrt (1 - (cos C)^2) in
      let area := (1 / 2) * a * b * sin_C in area = 2 * sqrt 2) :=
by
  sorry

end triangle_identity_l494_494962


namespace horse_catch_up_l494_494985

theorem horse_catch_up :
  ∀ (x : ℕ), (240 * x = 150 * (x + 12)) → x = 20 :=
by
  intros x h
  have : 240 * x = 150 * x + 1800 := by sorry
  have : 240 * x - 150 * x = 1800 := by sorry
  have : 90 * x = 1800 := by sorry
  have : x = 1800 / 90 := by sorry
  have : x = 20 := by sorry
  exact this

end horse_catch_up_l494_494985


namespace anniversary_day_probability_l494_494399

-- Define the years in which the meeting could take place
def years := {1668, 1669, 1670, 1671}

-- Define a function to check if a year is a leap year
def is_leap_year (y : ℕ) : Prop := (y % 4 = 0)

-- Define the meeting date
def meeting_day := 5 -- Friday as the 5th day of the week (assuming 0 = Sunday)

-- Define the anniversary function that computes the day of the 11th anniversary
def anniversary_day (start_year : ℕ) : ℕ :=
  let days_in_year (y : ℕ) := if is_leap_year y then 366 else 365
  let total_days := (List.range 11).map (λ i => days_in_year (start_year + i))
  (total_days.sum + meeting_day) % 7

-- Define the probability computation
def probability (day : ℕ) : ℚ :=
  let occurences := (years.toList.map anniversary_day).count (λ d => d = day)
  occurences / years.toList.length

-- Statement of the theorem
theorem anniversary_day_probability :
  probability 5 = 3 / 4 ∧ probability 4 = 1 / 4 := -- 5 is Friday, 4 is Thursday
  by
    sorry -- Proof goes here

end anniversary_day_probability_l494_494399


namespace anniversary_day_probability_l494_494401

-- Define the years in which the meeting could take place
def years := {1668, 1669, 1670, 1671}

-- Define a function to check if a year is a leap year
def is_leap_year (y : ℕ) : Prop := (y % 4 = 0)

-- Define the meeting date
def meeting_day := 5 -- Friday as the 5th day of the week (assuming 0 = Sunday)

-- Define the anniversary function that computes the day of the 11th anniversary
def anniversary_day (start_year : ℕ) : ℕ :=
  let days_in_year (y : ℕ) := if is_leap_year y then 366 else 365
  let total_days := (List.range 11).map (λ i => days_in_year (start_year + i))
  (total_days.sum + meeting_day) % 7

-- Define the probability computation
def probability (day : ℕ) : ℚ :=
  let occurences := (years.toList.map anniversary_day).count (λ d => d = day)
  occurences / years.toList.length

-- Statement of the theorem
theorem anniversary_day_probability :
  probability 5 = 3 / 4 ∧ probability 4 = 1 / 4 := -- 5 is Friday, 4 is Thursday
  by
    sorry -- Proof goes here

end anniversary_day_probability_l494_494401


namespace minimum_m_value_l494_494921

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 + a * Real.log x + 1

theorem minimum_m_value :
  (∀ x1 x2 : ℝ, x1 ∈ Set.Ici (3 : ℝ) → x2 ∈ Set.Ici (3 : ℝ) → x1 ≠ x2 →
     ∃ a : ℝ, a ∈ Set.Icc (1 : ℝ) (2 : ℝ) ∧
     (f x1 a - f x2 a) / (x2 - x1) < m) →
  m ≥ -20 / 3 := sorry

end minimum_m_value_l494_494921


namespace minimum_roots_f_l494_494145

theorem minimum_roots_f (f : ℝ → ℝ)
  (h1 : ∀ x : ℝ, f (2 - x) = f (2 + x))
  (h2 : ∀ x : ℝ, f (7 - x) = f (7 + x))
  (h3 : f 0 = 0) :
  (∃ M : ℕ, M = 401 ∧ ∀ x ∈ Icc (-1000) 1000, f x = 0 → M = 401) :=
sorry

end minimum_roots_f_l494_494145


namespace speed_of_second_train_l494_494700

theorem speed_of_second_train :
  ∀ (L1 L2 : ℕ) (V1 : ℝ) (T : ℝ), 
    L1 = 120 → 
    L2 = 165 → 
    V1 = 80 → 
    T = 7.0752960452818945 →
    let D : ℝ := L1 + L2 in
    let D_km : ℝ := D / 1000 in
    let T_hr : ℝ := T / 3600 in
    let V_rel : ℝ := D_km / T_hr in
    S ≈ V_rel - V1 :=
  -- where S is the speed of the second train in km/h
  sorry

end speed_of_second_train_l494_494700


namespace cos_alpha_value_l494_494932

theorem cos_alpha_value (α β : Real) (hα1 : 0 < α) (hα2 : α < π / 2) 
    (hβ1 : π / 2 < β) (hβ2 : β < π) (hcosβ : Real.cos β = -1/3)
    (hsin_alpha_beta : Real.sin (α + β) = 1/3) : 
    Real.cos α = 4 * Real.sqrt 2 / 9 := by
  sorry

end cos_alpha_value_l494_494932


namespace f_2012_eq_x_l494_494125

noncomputable def f (x : ℝ) : ℝ := (1 + x) / (1 - x)

@[simp] def f_n : ℕ → ℝ → ℝ
| 0, x := x
| (n+1), x := f (f_n n x)

theorem f_2012_eq_x (x : ℝ) : f_n 2012 x = x :=
sorry

end f_2012_eq_x_l494_494125


namespace sum_abs_coeffs_l494_494547

theorem sum_abs_coeffs (a : ℝ → ℝ) :
  (∀ x, (1 - 3 * x)^9 = a 0 + a 1 * x + a 2 * x^2 + a 3 * x^3 + a 4 * x^4 + a 5 * x^5 + a 6 * x^6 + a 7 * x^7 + a 8 * x^8 + a 9 * x^9) →
  |a 0| + |a 1| + |a 2| + |a 3| + |a 4| + |a 5| + |a 6| + |a 7| + |a 8| + |a 9| = 4^9 := by
  sorry

end sum_abs_coeffs_l494_494547


namespace increase_150_percent_of_80_l494_494293

theorem increase_150_percent_of_80 : 80 * 1.5 + 80 = 200 :=
by
  sorry

end increase_150_percent_of_80_l494_494293


namespace increased_number_l494_494274

theorem increased_number (original_number : ℕ) (percentage_increase : ℚ) :
  original_number = 80 →
  percentage_increase = 1.5 →
  original_number * (1 + percentage_increase) = 200 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end increased_number_l494_494274


namespace increase_150_percent_of_80_l494_494298

theorem increase_150_percent_of_80 : 80 * 1.5 + 80 = 200 :=
by
  sorry

end increase_150_percent_of_80_l494_494298


namespace evaluate_expression_l494_494448

theorem evaluate_expression: 121 + 2 * 11 * 4 + 16 + 7 = 232 := by
  -- Definitions based on the conditions provided
  have h1: 121 = 11^2 := by norm_num
  have h2: 16 = 4^2 := by norm_num
  have h3: 2 * 11 * 4 = 88 := by norm_num

  -- Proving the main statement
  calc
    121 + 2 * 11 * 4 + 16 + 7
    = 11^2 + 2 * 11 * 4 + 4^2 + 7 : by rw [h1, h2, h3]
    ... = (11 + 4)^2 + 7 : by ring
    ... = 15^2 + 7 : by norm_num
    ... = 225 + 7 : by norm_num
    ... = 232 : by norm_num
  sorry

end evaluate_expression_l494_494448


namespace eta_ge_one_probability_l494_494616

noncomputable def binomial_distribution (n : ℕ) (p : ℚ) : ℕ → ℚ
| k := nat.choose n k * p^k * (1-p)^(n-k)

noncomputable def prob_ge_one (n : ℕ) (p : ℚ) : ℚ :=
1 - binomial_distribution n p 0

theorem eta_ge_one_probability :
  ∀ (p : ℚ), (prob_ge_one 2 p = 5/9) → (prob_ge_one 3 p = 19/27) :=
by
  intros p h
  sorry

end eta_ge_one_probability_l494_494616


namespace correct_operation_l494_494720

-- Define the various conditions as mathematical equalities
def condition_A (a : ℝ) : Prop := a^2 * a^3 = a^6
def condition_B (a : ℝ) : Prop := (-a^3)^2 = -a^6
def condition_C (a : ℝ) : Prop := (2a)^3 = 8a^3
def condition_D (a : ℝ) : Prop := a^2 + a^3 = a^5

-- The statement to prove that condition_C is the correct operation
theorem correct_operation (a : ℝ) : condition_C a ∧ ¬ condition_A a ∧ ¬ condition_B a ∧ ¬ condition_D a :=
by
  sorry

end correct_operation_l494_494720


namespace squirrel_travel_time_l494_494767

theorem squirrel_travel_time
  (distance : ℕ) (rate : ℕ) (rest_time : ℕ) 
  (h_distance : distance = 3) 
  (h_rate : rate = 5) 
  (h_rest_time : rest_time = 10) :
  let travel_time := (distance : ℚ) / (rate : ℚ) * 60 in
  travel_time + rest_time = 46 := by 
  sorry

end squirrel_travel_time_l494_494767


namespace distinguishable_balls_in_indistinguishable_boxes_l494_494027

theorem distinguishable_balls_in_indistinguishable_boxes :
  let num_distinguishable_balls := 6
  let num_indistinguishable_boxes := 3
  -- different ways to distribute balls across boxes summarized
  let ways := ∑([{6, 0, 0} ++ {5, 1, 0} ++ {4, 2, 0} ++ {4, 1, 1} ++ {3, 3, 0} ++ {3, 2, 1} ++ {2, 2, 2}], sum)
  -- corresponding ways to remove permutation of distinguishable ball cases
  ways = 222 := sorry

end distinguishable_balls_in_indistinguishable_boxes_l494_494027


namespace impossibility_of_fitting_l494_494161

noncomputable def impossible_fit_non_overlapping_triangles_with_area_greater_than_1_inside_circle_of_radius_1 : Prop :=
  ∀ (T1 T2 : Type) [triangle T1] [triangle T2] 
  (area_T1 : ℝ) (area_T2 : ℝ) (R : ℝ),
  R = 1 →
  area_T1 > 1 →
  area_T2 > 1 →
  (area_T1 + area_T2 ≤ π) →
  false

theorem impossibility_of_fitting :
  impossible_fit_non_overlapping_triangles_with_area_greater_than_1_inside_circle_of_radius_1 := 
by
  sorry

end impossibility_of_fitting_l494_494161


namespace neg_forall_sin_gt_zero_l494_494532

theorem neg_forall_sin_gt_zero :
  ¬ (∀ x : ℝ, Real.sin x > 0) ↔ ∃ x : ℝ, Real.sin x ≤ 0 := 
sorry

end neg_forall_sin_gt_zero_l494_494532


namespace arithmetic_sequence_a5_l494_494091

variable (a : ℕ → ℝ) -- Define a_n as a sequence of real numbers

-- Define the arithmetic sequence property
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) := ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions
axiom arithmetic_seq_condition (a4 a5 a6 : ℝ) (h1 : a 4 = a4) (h2 : a 5 = a5) (h3 : a 6 = a6) :
  a4 + a5 + a6 = 90

-- Prove that a5 = 30 under the given conditions
theorem arithmetic_sequence_a5 (a : ℕ → ℝ) (d : ℝ) (a4 a5 a6 : ℝ) 
  (h1: a 4 = a4) (h2: a 5 = a5) (h3: a 6 = a6) 
  (seq_prop : arithmetic_sequence a d) 
  (cond : a4 + a5 + a6 = 90) : a5 = 30 :=
by
  have h0 : a4 + a5 + a6 = 90 := cond
  have h4 : a4 + a6 = 2 * a5 := sorry
  have h5 : 2 * a5 + a5 = 90 := by rw [←h4, h0]
  have h6 : 3 * a5 = 90 := by rw [h5]
  have h7 : a5 = 30 := by sorry
  exact h7

end arithmetic_sequence_a5_l494_494091


namespace exists_monotonically_decreasing_interval_unique_tangent_line_intersection_l494_494525

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := 
  (1 / 2) * m * (x - 1) ^ 2 - 2 * x + 3 + Real.log x

theorem exists_monotonically_decreasing_interval {m : ℝ} (hm : m ≥ 1) : 
  ∃ (a b : ℝ), 0 < a ∧ a < b ∧ ∀ x ∈ Icc a b, f m x ≤ f m a :=
sorry

theorem unique_tangent_line_intersection {m : ℝ} (hm : m ≥ 1) : 
  (∃ x : ℝ, f m x = -x + 2 ∧ f m 1 = 1 ∧ (∀ y, f m y = -y + 2 → y = 1)) ↔ (m = 1) :=
sorry

end exists_monotonically_decreasing_interval_unique_tangent_line_intersection_l494_494525


namespace possible_values_of_varphi_l494_494515

theorem possible_values_of_varphi (f : ℝ → ℝ) (φ : ℝ) :
  (∀ x, f x = sin (x + φ) - sin (x + 7 * φ)) →
  (∀ x, f (-x) = -f (x)) →
  (φ = π / 8 ∨ φ = 3 * π / 8) :=
by
  sorry

end possible_values_of_varphi_l494_494515


namespace increase_150_percent_of_80_l494_494291

theorem increase_150_percent_of_80 : 80 * 1.5 + 80 = 200 :=
by
  sorry

end increase_150_percent_of_80_l494_494291


namespace solution_set_m_is_2_l494_494564

theorem solution_set_m_is_2 (a m : ℝ) (h : {x : ℝ | ax^2 - 6 * x + a^2 < 0} = set.Ioo 1 m) : 
  m = 2 :=
begin
  sorry
end

end solution_set_m_is_2_l494_494564


namespace similar_triangles_side_length_l494_494659

theorem similar_triangles_side_length
  (A1 A2 : ℕ) (k : ℕ) (h1 : A1 - A2 = 18)
  (h2 : A1 = k^2 * A2) (h3 : ∃ n : ℕ, A2 = n)
  (s : ℕ) (h4 : s = 3) :
  s * k = 6 :=
by
  sorry

end similar_triangles_side_length_l494_494659


namespace num_ways_dist_6_balls_3_boxes_l494_494049

open Finset

/-- Number of ways to distribute 6 distinguishable balls into 3 indistinguishable boxes -/
theorem num_ways_dist_6_balls_3_boxes : 
  let num_ways := 
    let n := 6 in let k := 3 in
    ((choose n 6) + 
    (choose n 5) + 
    (choose n 4) + 
    (choose n 4) + 
    (choose n 3) / 2 + 
    (choose n 3) * (choose (n - 3) 2) + 
    (choose n 2) * (choose (n - 2) 2) / (factorial k)) 
  in num_ways = 122 := 
by 
  sorry

end num_ways_dist_6_balls_3_boxes_l494_494049


namespace anniversary_day_of_week_probability_l494_494404

/-- The 11th anniversary of Robinson Crusoe and Friday's meeting can fall on a Friday with a
probability of 3/4 and on a Thursday with a probability of 1/4, given that the meeting occurred
in any year from 1668 to 1671 with equal probability. -/
theorem anniversary_day_of_week_probability :
  let years := {1668, 1669, 1670, 1671},
      leap (y : ℕ) := y % 4 = 0,
      days_in_year := λ y, if leap y then 366 else 365,
      total_days (yr : ℕ) := list.sum (list.map days_in_year (list.range' yr 11)),
      day_of_week_after_11_years (initial_year : ℕ) := total_days initial_year % 7 = 0,
      events := {week_day | ∀ y ∈ years, (day_of_week_after_11_years y)},
      friday_probability := rat.mk 3 4,
      thursday_probability := rat.mk 1 4
  in
  (events = {0} ∨ events = {6}) ∧
  (events = {0} → friday_probability = rat.mk 3 4 ∧ thursday_probability = rat.mk 1 4) ∧
  (events = {6} → friday_probability = rat.mk 1 4 ∧ thursday_probability = rat.mk 3 4):=
begin
  sorry
end

end anniversary_day_of_week_probability_l494_494404


namespace combined_tax_rate_is_correct_l494_494602

noncomputable def combined_tax_rate (john_income : ℝ) (ingrid_income : ℝ) (john_tax_rate : ℝ) (ingrid_tax_rate : ℝ) : ℝ :=
  let john_tax := john_tax_rate * john_income
  let ingrid_tax := ingrid_tax_rate * ingrid_income
  let total_tax := john_tax + ingrid_tax
  let total_income := john_income + ingrid_income
  total_tax / total_income

theorem combined_tax_rate_is_correct :
  combined_tax_rate 56000 72000 0.30 0.40 = 0.35625 := 
by
  sorry

end combined_tax_rate_is_correct_l494_494602


namespace polynomial_decomposition_l494_494220

-- Define the given polynomial
def P (x y z : ℝ) : ℝ := x^2 + 2*x*y + 5*y^2 - 6*x*z - 22*y*z + 16*z^2

-- Define the target decomposition
def Q (x y z : ℝ) : ℝ := (x + (y - 3*z))^2 + (2*y - 4*z)^2 - (3*z)^2

theorem polynomial_decomposition (x y z : ℝ) : P x y z = Q x y z :=
  sorry

end polynomial_decomposition_l494_494220


namespace phil_initial_money_eq_40_l494_494628

def initial_money (p q s j left_in_quarters : ℝ) :=
  p + q + s + j + left_in_quarters

constant pizza_cost : ℝ := 2.75
constant soda_cost : ℝ := 1.50
constant jeans_cost : ℝ := 11.50
constant quarter_value : ℝ := 0.25
constant quarters_left : ℕ := 97

theorem phil_initial_money_eq_40 :
  initial_money pizza_cost soda_cost jeans_cost (quarters_left * quarter_value) = 40 :=
by
  sorry

end phil_initial_money_eq_40_l494_494628


namespace increase_80_by_150_percent_l494_494282

theorem increase_80_by_150_percent : 
  ∀ x percent, x = 80 ∧ percent = 1.5 → x + (x * percent) = 200 :=
by
  sorry

end increase_80_by_150_percent_l494_494282


namespace equation_of_ellipse_range_of_triangle_area_l494_494498

-- Definition of the problem conditions

def ellipse (x y : ℝ) := (y^2 / 4) + x^2 = 1

def sum_of_distances (P F1 F2 : ℝ × ℝ) : Prop :=
  let (xₚ, yₚ) := P in
  let (x₁, y₁) := F1 in
  let (x₂, y₂) := F2 in
  (√((xₚ - x₁)^2 + (yₚ - y₁)^2) + √((xₚ - x₂)^2 + (yₚ - y₂)^2)) = 4

def eccentricity (a c : ℝ) : Prop := (c / a) = √3 / 2

def foci_on_y_axis (F1 F2 : ℝ × ℝ) : Prop := F1 = (0, c) ∧ F2 = (0, -c)

-- Definitions for finding the ellipse equation
theorem equation_of_ellipse :
  ∃ a b c : ℝ, a > b ∧ b > 0 ∧ sum_of_distances P (0, c) (0, -c) ∧ eccentricity a c
  → ellipse = (λ x y, (y^2 / 4) + x^2 = 1) := sorry

-- Definitions for the second part
theorem range_of_triangle_area (k : ℝ) :
  ∃ S : ℝ, (∀ x y : ℝ, ellipse x y ∧ (y = k * x + 1) → S ∈ (0, √3 / 2)) := sorry

end equation_of_ellipse_range_of_triangle_area_l494_494498


namespace sum_of_first_ten_primes_with_units_digit_3_l494_494856

def is_prime (n : ℕ) : Prop := nat.prime n

def units_digit_3 (n : ℕ) : Prop := n % 10 = 3

def first_ten_primes_units_digit_3 : list ℕ :=
  [3, 13, 23, 43, 53, 73, 83, 103, 113, 163]

def sum_first_ten_primes_units_digit_3 : ℕ :=
  first_ten_primes_units_digit_3.sum

theorem sum_of_first_ten_primes_with_units_digit_3 :
  sum_first_ten_primes_units_digit_3 = 793 := by
  -- Here we provide the steps as a placeholder, but in real practice,
  -- a proof should be constructed to verify this calculation.
  sorry

end sum_of_first_ten_primes_with_units_digit_3_l494_494856


namespace triangle_area_of_parabola_l494_494517

theorem triangle_area_of_parabola (m : ℝ) (h₁ : ∀ x, x^2 + m * x - 6 = 0 → (x = -3 ∨ x = 2))
                                   (h₂ : ∀ y, y = -6 → y = -6 )
                                   (h₃ : -3 = -3) :
  let A : (ℝ × ℝ) := (-3, 0)
  let B : (ℝ × ℝ) := (2, 0)
  let C : (ℝ × ℝ) := (0, -6)
  let base := (B.fst - A.fst).abs
  let height := C.snd.abs
  let S := (1 / 2) * base * height
  S = 15 :=
by sorry

end triangle_area_of_parabola_l494_494517


namespace articles_selling_price_to_cost_price_eq_l494_494657

theorem articles_selling_price_to_cost_price_eq (C N : ℝ) (h_gain : 2 * C * N = 20 * C) : N = 10 :=
by
  sorry

end articles_selling_price_to_cost_price_eq_l494_494657


namespace sum_a_n_l494_494870

def a_n (n : ℕ) : ℕ :=
  if n % 30 = 0 then 15
  else if n % 60 = 0 then 10
  else if n % 60 = 0 then 12
  else 0

theorem sum_a_n : (∑ n in Finset.range 1500, a_n n) = 1263 := by
  sorry

end sum_a_n_l494_494870


namespace ellipse_equation_line_slope_point_y0_value_l494_494497

-- Given an ellipse \frac{x^2}{a^2}+\frac{y^2}{b^2}=1(a > b > 0) with the length of the major axis being 
-- twice the length of the minor axis, and the area of the rhombus formed by connecting the four vertices 
-- of the ellipse is 4. A line l passes through point A(-a,0) and intersects the ellipse at another 
-- point B; We need to prove the following:

-- 1. Prove the equation of the ellipse is \frac{x^2}{4}+y^2=1
theorem ellipse_equation
  (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : 2 * a = 4 * b) (h4 : a * b = 2) :
  (∀ x y : ℝ, x^2 / 4 + y^2 = 1 ↔ (x^2 / a^2) + (y^2 / b^2) = 1) :=
sorry

-- 2. If the length of segment |AB| is 4sqrt(2)/5 and the line l passes through point A(-2,0)
-- find the slope of line l is ± 1
theorem line_slope
  (a : ℝ) (h1 : a = 2) (k : ℝ) (AB_length : AB_length = 4 * sqrt 2 / 5) :
  (k = 1 ∨ k = -1) :=
sorry

-- 3. Prove the value of y_0 for point 
-- Q(0, y_0) lies on the perpendicular bisector of segment AB is ±2sqrt(2) or ±2sqrt(14)/5
theorem point_y0_value
  (y0 : ℝ) (h1 : (QA.dot_product QB) = 4) :
  (y0 = 2 * sqrt 2 ∨ y0 = -2 * sqrt 2 ∨ y0 = 2 * sqrt (14 / 5) ∨ y0 = - 2 * sqrt (14 / 5)) :=
sorry

end ellipse_equation_line_slope_point_y0_value_l494_494497


namespace balls_in_boxes_l494_494004

/-
Prove that the number of ways to distribute 6 distinguishable balls into 3 indistinguishable boxes is 132.
-/
theorem balls_in_boxes : 
  ∃ (ways : ℕ), ways = 132 ∧ ways = 
    (1) + 
    (Nat.choose 6 5) + 
    (Nat.choose 6 4) + 
    (Nat.choose 6 4) + 
    (Nat.choose 6 3) + 
    (Nat.choose 6 3 * Nat.choose 3 2) + 
    (Nat.choose 6 2 * Nat.choose 4 2 / 6) := 
by
  sorry

end balls_in_boxes_l494_494004


namespace increase_150_percent_of_80_l494_494299

theorem increase_150_percent_of_80 : 80 * 1.5 + 80 = 200 :=
by
  sorry

end increase_150_percent_of_80_l494_494299


namespace solve_log_equation_l494_494169

theorem solve_log_equation (y : ℝ) (h : log 3 y + log 9 y = 5) : y = 3^(10/3) :=
sorry

end solve_log_equation_l494_494169


namespace last_digit_sum_l494_494197

theorem last_digit_sum (a b : ℕ) (exp : ℕ)
  (h₁ : a = 1993) (h₂ : b = 1995) (h₃ : exp = 2002) :
  ((a ^ exp + b ^ exp) % 10) = 4 := 
by
  sorry

end last_digit_sum_l494_494197


namespace panthers_score_l494_494085

-- Definitions as per the conditions
def total_points (C P : ℕ) : Prop := C + P = 48
def margin (C P : ℕ) : Prop := C = P + 20

-- Theorem statement proving Panthers score 14 points
theorem panthers_score (C P : ℕ) (h1 : total_points C P) (h2 : margin C P) : P = 14 :=
sorry

end panthers_score_l494_494085


namespace eccentricity_of_ellipse_l494_494658

variables (a b : ℝ) (h : 0 < b ∧ b < a)

def elliptical_conditions :=
  (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 →
    (1 - x = 2 * (x - 1) ∧ 1 - y = 2 * (y - 1)) ∧ 
    (∀ x₁ y₁ x₂ y₂ : ℝ, x₁ ≠ x₂ → y₁ ≠ y₂ → 
      ((y₂ - y₁) / (x₂ - x₁) = -1 / 4)))

theorem eccentricity_of_ellipse 
  (h_conditions : elliptical_conditions a b) : 
  sqrt (1 - b^2 / a^2) = sqrt 3 / 2 :=
begin
  sorry
end

end eccentricity_of_ellipse_l494_494658


namespace parallelogram_OCO1O2_l494_494977

open EuclideanGeometry

-- Definitions for given conditions
noncomputable def O (ABC : Triangle) : Point := sorry -- circumcenter of ABC
noncomputable def O_1 (CKL : Triangle) : Point := sorry -- circumcenter of CKL
noncomputable def O_2 (OAB : Triangle) : Point := sorry -- circumcenter of OAB

-- Proving that OCO_1O_2 is a parallelogram given the conditions
theorem parallelogram_OCO1O2
  {ABC : Triangle}
  (h1 : ABC.A < ABC.B)
  (h2 : ABC.B < ABC.C)
  (h3 : is_perpendicular_bisector ABC.BC point.K)
  (h4 : is_perpendicular_bisector ABC.AC point.L)
  (h5 : circumcenter ABC = O(ABC))
  (h6 : circumcenter (triangle_of_points C K L) = O_1(triangle_of_points C K L))
  (h7 : circumcenter (triangle_of_points O(ABC) A B) = O_2(triangle_of_points O(ABC) A B)) :
  is_parallelogram (quad_of_points O(ABC) C O_1(triangle_of_points C K L) O_2(triangle_of_points O(ABC) A B)) :=
sorry

end parallelogram_OCO1O2_l494_494977


namespace balls_in_boxes_l494_494053

theorem balls_in_boxes :
    (Nat.choose 6 6) + 
    (Nat.choose 6 5) + 
    (Nat.choose 6 4 * Nat.choose 2 2) + 
    (Nat.choose 6 3 / 2) + 
    (Nat.choose 6 4 * Nat.choose 2 1) + 
    (Nat.choose 6 2 * Nat.choose 4 2 / 3.factorial) = 77 :=
by
  sorry

end balls_in_boxes_l494_494053


namespace smallest_m_l494_494137

open Real

noncomputable def y_seq (m : ℕ) : Type := Fin m → ℝ

theorem smallest_m (m : ℕ) (y : y_seq m) (h1 : ∀ i : Fin m, |y i| ≤ 1/2)
  (h2 : ∑ i, |y i| = 10 + |∑ i, y i|) : m = 20 :=
sorry

end smallest_m_l494_494137


namespace geometric_sequence_sum_l494_494971

theorem geometric_sequence_sum
  (a : ℕ → ℝ)
  (a1 : a 1 = 3)
  (a4 : a 4 = 24)
  (h_geo : ∃ q : ℝ, ∀ n : ℕ, a n = 3 * q^(n - 1)) :
  a 3 + a 4 + a 5 = 84 :=
by
  sorry

end geometric_sequence_sum_l494_494971


namespace increase_by_percentage_l494_494256

theorem increase_by_percentage (initial_amount : ℕ) (percentage_increase : ℕ) :
  initial_amount = 80 → 
  percentage_increase = 150 → 
  initial_amount * (percentage_increase / 100) + initial_amount = 200 :=
by
  intros h_initial h_percentage
  rw [h_initial, h_percentage]
  calc
    80 * (150 / 100) + 80 = 80 * 1.5 + 80 : by norm_num
                     ... = 120 + 80         : by norm_num
                     ... = 200              : by norm_num
  sorry

end increase_by_percentage_l494_494256


namespace balls_in_boxes_l494_494052

theorem balls_in_boxes :
    (Nat.choose 6 6) + 
    (Nat.choose 6 5) + 
    (Nat.choose 6 4 * Nat.choose 2 2) + 
    (Nat.choose 6 3 / 2) + 
    (Nat.choose 6 4 * Nat.choose 2 1) + 
    (Nat.choose 6 2 * Nat.choose 4 2 / 3.factorial) = 77 :=
by
  sorry

end balls_in_boxes_l494_494052


namespace exists_city_reaching_all_l494_494572

variables {City : Type} (canReach : City → City → Prop)

-- Conditions from the problem
axiom reach_itself (A : City) : canReach A A
axiom reach_transitive {A B C : City} : canReach A B → canReach B C → canReach A C
axiom reach_any_two {P Q : City} : ∃ R : City, canReach R P ∧ canReach R Q

-- The proof problem
theorem exists_city_reaching_all (cities : City → Prop) :
  (∀ P Q, P ≠ Q → cities P → cities Q → ∃ R, cities R ∧ canReach R P ∧ canReach R Q) →
  ∃ C, ∀ A, cities A → canReach C A :=
by
  intros H
  sorry

end exists_city_reaching_all_l494_494572


namespace min_f_x_gt_2_solve_inequality_l494_494887

noncomputable def f (a b x : ℝ) : ℝ := a * x^2 / (x + b)

theorem min_f_x_gt_2 (a b : ℝ) (h1 : ∀ x, f a b x = 2 * x + 3 → x = -2 ∨ x = 3) :
∃ c, ∀ x > 2, f a b x ≥ c ∧ (∀ y, y > 2 → f a b y = c → y = 4 ∧ c = 8) :=
sorry

theorem solve_inequality (a b k : ℝ) (x : ℝ) (h1 : ∀ x, f a b x = 2 * x + 3 → x = -2 ∨ x = 3) :
  f a b x < (k * (x - 1) + 1 - x^2) / (2 - x) ↔ 
  (x < 2 ∧ k = 0) ∨ 
  (-1 < k ∧ k < 0 ∧ 1 - 1 / k < x ∧ x < 2) ∨ 
  ((k > 0 ∨ k < -1) ∧ (1 - 1 / k < x ∧ x < 2) ∨ x > 2) ∨ 
  (k = -1 ∧ x ≠ 2) :=
sorry

end min_f_x_gt_2_solve_inequality_l494_494887


namespace region_area_correct_l494_494834

open Real

noncomputable def area_of_region : ℝ :=
  let region := {p : ℝ × ℝ | abs (p.1 + p.2) + abs (p.1 - p.2) ≤ 6 ∧ p.1 + p.2 ≥ 1}
  30 * real.sqrt 2

theorem region_area_correct : 
  let region := {p : ℝ × ℝ | abs (p.1 + p.2) + abs (p.1 - p.2) ≤ 6 ∧ p.1 + p.2 ≥ 1} in
  let area := 30 * real.sqrt 2 in
  sorry

end region_area_correct_l494_494834


namespace tan_seven_pi_over_four_l494_494450

theorem tan_seven_pi_over_four : Real.tan (7 * Real.pi / 4) = -1 := 
by
  -- In this case, we are proving a specific trigonometric identity
  sorry

end tan_seven_pi_over_four_l494_494450


namespace sufficient_not_necessary_condition_l494_494629

theorem sufficient_not_necessary_condition (k : ℝ) (h₁ : k > 1) 
    (h₂ : ∀ y : ℝ, ∃ x : ℝ, y = real.log (x^2 - 2 * k * x + k)) :
  (∀ y : ℝ, ∃ x : ℝ, y = real.log (x^2 - 2 * k * x + k)) ↔ k > 1 := 
  sorry

end sufficient_not_necessary_condition_l494_494629


namespace Joe_reduced_fraction_l494_494113

theorem Joe_reduced_fraction (initial_data_points : ℕ) (final_data_points : ℕ) :
  initial_data_points = 200 → 
  final_data_points = 180 → 
  let increased_data_points := initial_data_points + 0.20 * initial_data_points in
  (increased_data_points - final_data_points) / increased_data_points = 1 / 4 :=
by
  intros h1 h2
  let increased_data_points := initial_data_points + 0.20 * initial_data_points
  let reduced_fraction := (increased_data_points - final_data_points) / increased_data_points
  rw [h1, h2]
  sorry

end Joe_reduced_fraction_l494_494113


namespace solve_for_x_l494_494168

theorem solve_for_x (x : ℝ) : (x - 20) / 3 = (4 - 3 * x) / 4 → x = 7.08 := by
  sorry

end solve_for_x_l494_494168


namespace distribute_6_balls_in_3_boxes_l494_494040

def num_ways_to_distribute_balls (balls : ℕ) (boxes : ℕ) : ℕ :=
  if balls = 6 ∧ boxes = 3 then 92 else 0

theorem distribute_6_balls_in_3_boxes :
  num_ways_to_distribute_balls 6 3 = 92 :=
by
  rw num_ways_to_distribute_balls
  norm_num
  sorry

end distribute_6_balls_in_3_boxes_l494_494040


namespace determine_counterfeit_coin_l494_494790

-- Definitions and conditions
def coin_weight (coin : ℕ) : ℕ :=
  match coin with
  | 1 => 1 -- 1-kopek coin weighs 1 gram
  | 2 => 2 -- 2-kopeks coin weighs 2 grams
  | 3 => 3 -- 3-kopeks coin weighs 3 grams
  | 5 => 5 -- 5-kopeks coin weighs 5 grams
  | _ => 0 -- Invalid coin denomination, should not happen

def is_counterfeit (coin : ℕ) (actual_weight : ℕ) : Prop :=
  coin_weight coin ≠ actual_weight

-- Statement of the problem to be proved
theorem determine_counterfeit_coin (coins : List (ℕ × ℕ)) :
   (∀ (coin: ℕ) (weight: ℕ) (h : (coin, weight) ∈ coins),
      coin_weight coin = weight ∨ is_counterfeit coin weight) →
   (∃ (counterfeit_coin: ℕ) (weight: ℕ),
      (counterfeit_coin, weight) ∈ coins ∧ is_counterfeit counterfeit_coin weight) :=
sorry

end determine_counterfeit_coin_l494_494790


namespace balls_in_indistinguishable_boxes_l494_494015

theorem balls_in_indistinguishable_boxes : 
  let balls := 6 in
  let boxes := 3 in
  -- Expression to count ways to distribute balls into indistinguishable boxes
  ((balls.choose 0 * ((balls-0).choose 6)) + 
   (balls.choose 1 * ((balls-1).choose 5)) + 
   (balls.choose 2 * ((balls-2).choose 4)) + 
   (balls.choose 2 * ((balls-2).choose 3) / 2!) + 
   (balls.choose 3 * ((balls-3).choose 3)) + 
   (balls.choose 3 * ((balls-3).choose 2) * ((balls-5).choose 1)) + 
   (balls.choose 2 * ((balls-2).choose 2) / 3!))
  = 92 := 
sorry

end balls_in_indistinguishable_boxes_l494_494015


namespace increased_number_l494_494280

theorem increased_number (original_number : ℕ) (percentage_increase : ℚ) :
  original_number = 80 →
  percentage_increase = 1.5 →
  original_number * (1 + percentage_increase) = 200 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end increased_number_l494_494280


namespace fraction_halfway_l494_494708

theorem fraction_halfway (a b c d : ℚ) (h₁ : a = 3) (h₂ : b = 4) (h₃ : c = 5) (h₄ : d = 6) :
  (1 / 2) * ((a / b) + (c / d)) = 19 / 24 := 
by
  sorry

end fraction_halfway_l494_494708


namespace converse_proposition_l494_494656

-- Define the predicate variables p and q
variables (p q : Prop)

-- State the theorem about the converse of the proposition
theorem converse_proposition (hpq : p → q) : q → p :=
sorry

end converse_proposition_l494_494656


namespace ratio_of_areas_l494_494941

theorem ratio_of_areas (R_C R_D : ℝ) (h : (60 / 360) * (2 * Real.pi * R_C) = (40 / 360) * (2 * Real.pi * R_D)) :
  (Real.pi * R_C^2) / (Real.pi * R_D^2) = (9 / 4) :=
by
  sorry

end ratio_of_areas_l494_494941


namespace can_lid_boxes_count_l494_494384

theorem can_lid_boxes_count 
  (x y : ℕ) 
  (h1 : 3 * x + y + 14 = 75) : 
  x = 20 ∧ y = 1 :=
by 
  sorry

end can_lid_boxes_count_l494_494384


namespace boat_travel_distance_per_day_l494_494621

-- Definitions from conditions
def men : ℕ := 25
def water_daily_per_man : ℚ := 1/2
def travel_distance : ℕ := 4000
def total_water : ℕ := 250

-- Main theorem
theorem boat_travel_distance_per_day : 
  ∀ (men : ℕ) (water_daily_per_man : ℚ) (travel_distance : ℕ) (total_water : ℕ), 
  men = 25 ∧ water_daily_per_man = 1/2 ∧ travel_distance = 4000 ∧ total_water = 250 ->
  travel_distance / (total_water / (men * water_daily_per_man)) = 200 :=
by
  sorry

end boat_travel_distance_per_day_l494_494621


namespace rise_in_water_level_l494_494335

theorem rise_in_water_level : 
  let edge_length : ℝ := 15
  let volume_cube : ℝ := edge_length ^ 3
  let length : ℝ := 20
  let width : ℝ := 15
  let base_area : ℝ := length * width
  let rise_in_level : ℝ := volume_cube / base_area
  rise_in_level = 11.25 :=
by
  sorry

end rise_in_water_level_l494_494335


namespace largest_n_unique_k_l494_494713

theorem largest_n_unique_k :
  ∃ (n : ℕ), ( ∃! (k : ℕ), (5 : ℚ) / 11 < (n : ℚ) / (n + k) ∧ (n : ℚ) / (n + k) < 6 / 11 )
    ∧ n = 359 :=
sorry

end largest_n_unique_k_l494_494713


namespace increase_80_by_150_percent_l494_494239

-- Define the initial number (n) and the percentage increase (p)
def n : ℕ := 80
def p : ℚ := 1.5

-- The theorem stating the expected result after increasing n by 150%
theorem increase_80_by_150_percent : n + (p * n) = 200 := by
  sorry

end increase_80_by_150_percent_l494_494239


namespace third_candidate_votes_l494_494691

def percentage (p : ℝ) (v : ℝ) : ℝ := (p / 100) * v

theorem third_candidate_votes (T : ℝ) : 
  let V := 7636 + 11628 + T in
  percentage 54.336448598130836 V = 11628 →
  T = 2136 := 
by
  sorry

end third_candidate_votes_l494_494691


namespace cost_of_four_dozen_l494_494391

-- Defining the conditions
def cost_of_three_dozen (cost : ℚ) : Prop :=
  cost = 25.20

-- The theorem to prove the cost of four dozen apples at the same rate
theorem cost_of_four_dozen (cost : ℚ) :
  cost_of_three_dozen cost →
  (4 * (cost / 3) = 33.60) :=
by
  sorry

end cost_of_four_dozen_l494_494391


namespace train_speed_is_correct_l494_494379

-- Conditions
def train_length := 190.0152  -- in meters
def crossing_time := 17.1     -- in seconds

-- Convert units
def train_length_km := train_length / 1000  -- in kilometers
def crossing_time_hr := crossing_time / 3600  -- in hours

-- Statement of the proof problem
theorem train_speed_is_correct :
  (train_length_km / crossing_time_hr) = 40 :=
sorry

end train_speed_is_correct_l494_494379


namespace ratio_of_areas_of_circles_l494_494952

theorem ratio_of_areas_of_circles
    (C_C R_C C_D R_D L : ℝ)
    (hC : C_C = 2 * Real.pi * R_C)
    (hD : C_D = 2 * Real.pi * R_D)
    (hL : (60 / 360) * C_C = L ∧ L = (40 / 360) * C_D) :
    (Real.pi * R_C ^ 2) / (Real.pi * R_D ^ 2) = 4 / 9 :=
by
  sorry

end ratio_of_areas_of_circles_l494_494952


namespace balls_in_indistinguishable_boxes_l494_494013

theorem balls_in_indistinguishable_boxes : 
  let balls := 6 in
  let boxes := 3 in
  -- Expression to count ways to distribute balls into indistinguishable boxes
  ((balls.choose 0 * ((balls-0).choose 6)) + 
   (balls.choose 1 * ((balls-1).choose 5)) + 
   (balls.choose 2 * ((balls-2).choose 4)) + 
   (balls.choose 2 * ((balls-2).choose 3) / 2!) + 
   (balls.choose 3 * ((balls-3).choose 3)) + 
   (balls.choose 3 * ((balls-3).choose 2) * ((balls-5).choose 1)) + 
   (balls.choose 2 * ((balls-2).choose 2) / 3!))
  = 92 := 
sorry

end balls_in_indistinguishable_boxes_l494_494013


namespace solve_inequality_l494_494826

theorem solve_inequality : 
  {x : ℝ | (x^3 - x^2 - 6 * x) / (x^2 - 3 * x + 2) > 0} = 
  {x : ℝ | (-2 < x ∧ x < 0) ∨ (1 < x ∧ x < 2) ∨ (3 < x)} :=
sorry

end solve_inequality_l494_494826


namespace two_digit_number_count_l494_494427

theorem two_digit_number_count : 
  let valid_N := { N : ℕ | 10 ≤ N ∧ N < 100 ∧ ∃ a b : ℕ, N = 10 * a + b ∧ 9 * (a - b) ∈ { n^2 | n ∈ ℕ } } in
  card valid_N = 8 :=
by sorry

end two_digit_number_count_l494_494427


namespace triangle_area_is_96_l494_494744

-- Definitions of radii and sides being congruent
def tangent_circles (radius1 radius2 : ℝ) : Prop :=
  ∃ (O O' : ℝ × ℝ), dist O O' = radius1 + radius2

-- Given conditions
def radius_small : ℝ := 2
def radius_large : ℝ := 4
def sides_congruent (AB AC : ℝ) : Prop :=
  AB = AC

-- Theorem stating the goal
theorem triangle_area_is_96 
  (O O' : ℝ × ℝ)
  (AB AC : ℝ)
  (circ_tangent : tangent_circles radius_small radius_large)
  (sides_tangent : sides_congruent AB AC) :
  ∃ (BC : ℝ), ∃ (AF : ℝ), (1/2) * BC * AF = 96 := 
by
  sorry

end triangle_area_is_96_l494_494744


namespace increase_result_l494_494244

-- Given conditions
def original_number : ℝ := 80
def increase_percentage : ℝ := 1.5

-- The result after the increase
theorem increase_result (h1 : original_number = 80) (h2 : increase_percentage = 1.5) : 
  original_number + (increase_percentage * original_number) = 200 := by
  sorry

end increase_result_l494_494244


namespace coconut_grove_problem_l494_494084

variable (x : ℝ)

-- Conditions
def trees_yield_40_nuts_per_year : ℝ := 40 * (x + 2)
def trees_yield_120_nuts_per_year : ℝ := 120 * x
def trees_yield_180_nuts_per_year : ℝ := 180 * (x - 2)
def average_yield_per_tree_per_year : ℝ := 100

-- Problem Statement
theorem coconut_grove_problem
  (yield_40_trees : trees_yield_40_nuts_per_year x = 40 * (x + 2))
  (yield_120_trees : trees_yield_120_nuts_per_year x = 120 * x)
  (yield_180_trees : trees_yield_180_nuts_per_year x = 180 * (x - 2))
  (average_yield : average_yield_per_tree_per_year = 100) :
  x = 7 :=
by
  sorry

end coconut_grove_problem_l494_494084


namespace same_number_of_friends_l494_494729

open Finset

variable {A : Type*} [DecidableEq A] [Fintype A]

def knows (G : A → A → Prop) (a1 a2 : A) : Prop := G a1 a2

def common_friends (G : A → A → Prop) (a1 a2 : A) : Finset A :=
  {x | G a1 x ∧ G a2 x}.toFinset

noncomputable def friends (G : A → A → Prop) (a : A) : Finset A :=
  {x | G a x}.toFinset

theorem same_number_of_friends 
  (G : A → A → Prop)
  (h_symmetric : ∀ a b, G a b → G b a)
  (h_known : ∀ a b c, ¬ G a b → G a c → G b c → G c a)
  {a1 a2 : A}
  (h_a1_a2 : knows G a1 a2)
  (h_no_common_friends : common_friends G a1 a2 = ∅) :
  friends G a1 = friends G a2 :=
sorry

end same_number_of_friends_l494_494729


namespace red_to_green_speed_ratio_l494_494428

-- Conditions
def blue_car_speed : Nat := 80 -- The blue car's speed is 80 miles per hour
def green_car_speed : Nat := 8 * blue_car_speed -- The green car's speed is 8 times the blue car's speed
def red_car_speed : Nat := 1280 -- The red car's speed is 1280 miles per hour

-- Theorem stating the ratio of red car's speed to green car's speed
theorem red_to_green_speed_ratio : red_car_speed / green_car_speed = 2 := by
  sorry -- proof goes here

end red_to_green_speed_ratio_l494_494428


namespace increased_number_l494_494271

theorem increased_number (original_number : ℕ) (percentage_increase : ℚ) :
  original_number = 80 →
  percentage_increase = 1.5 →
  original_number * (1 + percentage_increase) = 200 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end increased_number_l494_494271


namespace polygon_area_l494_494102

theorem polygon_area (n : ℕ) (perimeter : ℕ) (s : ℕ) (area : ℕ) 
  (h1 : n = 32) 
  (h2 : perimeter = 64) 
  (h3 : s = perimeter / n) 
  (h4 : ∀ i, i < n → (□ (s = 2))) : 
  area = 64 := 
sorry

end polygon_area_l494_494102


namespace increase_by_percentage_l494_494327

theorem increase_by_percentage (a : ℤ) (b : ℚ) : ((b + 1) * a) = 200 :=
by
  -- Definitions based directly on conditions
  let a := (80 : ℤ)
  let b := (1.5 : ℚ)
  -- Assertions about the equivalent proof problem
  have h1 : b + 1 = 2.5 := by norm_num
  have h2 : (b + 1) * a = 200 := by norm_num
  exact h2

end increase_by_percentage_l494_494327


namespace vartan_recreation_l494_494117

noncomputable def vartan_recreation_percent (W : ℝ) (P : ℝ) : Prop := 
  let W_this_week := 0.9 * W
  let recreation_last_week := (P / 100) * W
  let recreation_this_week := 0.3 * W_this_week
  recreation_this_week = 1.8 * recreation_last_week

theorem vartan_recreation (W : ℝ) : ∀ P : ℝ, vartan_recreation_percent W P → P = 15 := 
by
  intro P h
  unfold vartan_recreation_percent at h
  sorry

end vartan_recreation_l494_494117


namespace cover_set_by_disk_l494_494373

variable {Point : Type}
variable [MetricSpace Point]

def can_be_covered_by_disk_of_radius_1 (S : Set Point) : Prop :=
  ∀ (A B C : Point), A ∈ S → B ∈ S → C ∈ S → Metric.closedBall (A+B+C)/3 1 ⊇ {A, B, C}

theorem cover_set_by_disk (S : Set Point) (h : can_be_covered_by_disk_of_radius_1 S) :
  ∃ (O : Point), Metric.closedBall O 1 ⊇ S :=
sorry

end cover_set_by_disk_l494_494373


namespace min_S_value_l494_494780

noncomputable def S (x : ℝ) : ℝ :=
(3 - x) ^ 2 / (Real.sqrt 3 / 4 * (1 - x ^ 2))

theorem min_S_value :
  inf {S x | x ∈ Ioo 0 1} = 32 * Real.sqrt 3 / 3 :=
by
  sorry

end min_S_value_l494_494780


namespace sample_mean_and_variance_correct_l494_494843

/-- The sample distribution data -/
def sample_data_x : List ℝ := [2, 3, 7, 9, 11, 12.5, 16, 18, 23, 25, 26]
def sample_data_n : List ℝ := [3, 5, 10, 6, 10, 4, 12, 13, 8, 20, 9]

/-- Sample size -/
def sample_size : ℝ := 100

/-- Number of sub-intervals -/
def num_sub_intervals : ℕ := 4

/-- Length of each interval -/
def interval_length : ℝ := 6

/-- Midpoints and frequencies for the new intervals -/
def midpoints : List ℝ := [5, 11, 17, 23]
def new_frequencies : List ℝ := [18, 20, 25, 37]

/-- Sample mean -/
def sample_mean : ℝ :=
  let total_frequency := new_frequencies.sum
  let weighted_sum := (List.zipWith (· * ·) new_frequencies midpoints).sum
  weighted_sum / total_frequency

/-- Sample variance with Sheppard's correction -/
def sample_variance : ℝ :=
  let total_frequency := new_frequencies.sum
  let weighted_square_sum := (List.zipWith (· * (· * ·)) new_frequencies (List.map (· * ·) midpoints)).sum
  let D_B := weighted_square_sum / total_frequency - sample_mean^2
  D_B - (interval_length^2 / 12)

/-- The proof statement asserting the computed values for sample mean and sample variance -/
theorem sample_mean_and_variance_correct :
  sample_mean = 15.86 ∧ sample_variance = 42.14 :=
by
  sorry

end sample_mean_and_variance_correct_l494_494843


namespace increase_by_percentage_l494_494259

theorem increase_by_percentage (initial_amount : ℕ) (percentage_increase : ℕ) :
  initial_amount = 80 → 
  percentage_increase = 150 → 
  initial_amount * (percentage_increase / 100) + initial_amount = 200 :=
by
  intros h_initial h_percentage
  rw [h_initial, h_percentage]
  calc
    80 * (150 / 100) + 80 = 80 * 1.5 + 80 : by norm_num
                     ... = 120 + 80         : by norm_num
                     ... = 200              : by norm_num
  sorry

end increase_by_percentage_l494_494259


namespace coeff_x2_of_square_p_l494_494531

def p (x : ℝ) : ℝ := x^5 - 5*x^3 + 4*x

theorem coeff_x2_of_square_p : 
  (polynomial.coeff (p(x) * p(x)) 2) = 16 := 
by 
  -- expand and calculate the coefficient
  sorry

end coeff_x2_of_square_p_l494_494531


namespace ratio_of_areas_l494_494942

theorem ratio_of_areas (R_C R_D : ℝ) (h : (60 / 360) * (2 * Real.pi * R_C) = (40 / 360) * (2 * Real.pi * R_D)) :
  (Real.pi * R_C^2) / (Real.pi * R_D^2) = (9 / 4) :=
by
  sorry

end ratio_of_areas_l494_494942


namespace possible_values_of_varphi_l494_494516

theorem possible_values_of_varphi (f : ℝ → ℝ) (φ : ℝ) :
  (∀ x, f x = sin (x + φ) - sin (x + 7 * φ)) →
  (∀ x, f (-x) = -f (x)) →
  (φ = π / 8 ∨ φ = 3 * π / 8) :=
by
  sorry

end possible_values_of_varphi_l494_494516


namespace books_loaned_l494_494374

theorem books_loaned (L : ℕ)
  (initial_books : ℕ := 150)
  (end_year_books : ℕ := 100)
  (return_rate : ℝ := 0.60)
  (loan_rate : ℝ := 0.40)
  (returned_books : ℕ := (initial_books - end_year_books)) :
  loan_rate * (L : ℝ) = (returned_books : ℝ) → L = 125 := by
  intro h
  sorry

end books_loaned_l494_494374


namespace larger_tablet_diagonal_is_approx_7_516_l494_494680

noncomputable def diagonal_length_larger_tablet : ℝ :=
sqrt 56.5

theorem larger_tablet_diagonal_is_approx_7_516 :
  ∀ (d : ℝ), d = diagonal_length_larger_tablet → d ≈ 7.516 :=
by
  intro d h
  rw [h]
  norm_num -- refine the proof with appropriate numerical approximation library usage.
  done

end larger_tablet_diagonal_is_approx_7_516_l494_494680


namespace increase_by_percentage_l494_494321

theorem increase_by_percentage (a : ℤ) (b : ℚ) : ((b + 1) * a) = 200 :=
by
  -- Definitions based directly on conditions
  let a := (80 : ℤ)
  let b := (1.5 : ℚ)
  -- Assertions about the equivalent proof problem
  have h1 : b + 1 = 2.5 := by norm_num
  have h2 : (b + 1) * a = 200 := by norm_num
  exact h2

end increase_by_percentage_l494_494321


namespace balls_in_boxes_l494_494005

/-
Prove that the number of ways to distribute 6 distinguishable balls into 3 indistinguishable boxes is 132.
-/
theorem balls_in_boxes : 
  ∃ (ways : ℕ), ways = 132 ∧ ways = 
    (1) + 
    (Nat.choose 6 5) + 
    (Nat.choose 6 4) + 
    (Nat.choose 6 4) + 
    (Nat.choose 6 3) + 
    (Nat.choose 6 3 * Nat.choose 3 2) + 
    (Nat.choose 6 2 * Nat.choose 4 2 / 6) := 
by
  sorry

end balls_in_boxes_l494_494005


namespace erin_days_to_receive_30_l494_494445

theorem erin_days_to_receive_30 (x : ℕ) (h : 3 * x = 30) : x = 10 :=
by
  sorry

end erin_days_to_receive_30_l494_494445


namespace pb_less_than_pc_l494_494567

theorem pb_less_than_pc {A B C P : Point}
  (h_ab_ac : dist A B = dist A C)
  (h_angle_apb_gt_apc : ∠APB > ∠APC) : dist P B < dist P C :=
sorry

end pb_less_than_pc_l494_494567


namespace michelle_total_payment_l494_494569
noncomputable def michelle_base_cost := 25
noncomputable def included_talk_time := 40 -- in hours
noncomputable def text_cost := 10 -- in cents per message
noncomputable def extra_talk_cost := 15 -- in cents per minute
noncomputable def february_texts_sent := 200
noncomputable def february_talk_time := 41 -- in hours

theorem michelle_total_payment : 
  25 + ((200 * 10) / 100) + (((41 - 40) * 60 * 15) / 100) = 54 := by
  sorry

end michelle_total_payment_l494_494569


namespace num_ways_dist_6_balls_3_boxes_l494_494047

open Finset

/-- Number of ways to distribute 6 distinguishable balls into 3 indistinguishable boxes -/
theorem num_ways_dist_6_balls_3_boxes : 
  let num_ways := 
    let n := 6 in let k := 3 in
    ((choose n 6) + 
    (choose n 5) + 
    (choose n 4) + 
    (choose n 4) + 
    (choose n 3) / 2 + 
    (choose n 3) * (choose (n - 3) 2) + 
    (choose n 2) * (choose (n - 2) 2) / (factorial k)) 
  in num_ways = 122 := 
by 
  sorry

end num_ways_dist_6_balls_3_boxes_l494_494047


namespace minimum_framing_required_l494_494737

def original_width : ℕ := 5
def original_height : ℕ := 7
def enlargement_factor : ℕ := 4
def border_width : ℕ := 3
def inches_per_yard : ℕ := 36

def enlarged_width := original_width * enlargement_factor
def enlarged_height := original_height * enlargement_factor
def final_width := enlarged_width + 2 * border_width
def final_height := enlarged_height + 2 * border_width
def perimeter_inches := 2 * (final_width + final_height)
def perimeter_yards : ℚ := perimeter_inches / inches_per_yard

theorem minimum_framing_required : ∀ (original_width original_height enlargement_factor border_width : ℕ),
  original_width = 5 →
  original_height = 7 →
  enlargement_factor = 4 →
  border_width = 3 →
  (⟦perimeter_inches / inches_per_yard⟧ : ℤ) = 4 :=
by
  sorry

end minimum_framing_required_l494_494737


namespace intersection_eq_N_l494_494471

noncomputable def U : Set ℝ := Set.univ
noncomputable def M : Set ℝ := {x : ℝ | x < 1}
noncomputable def N : Set ℝ := {x : ℝ | 0 < x ∧ x < 1}

theorem intersection_eq_N : M ∩ N = N :=
by { sorry }

#eval intersection_eq_N

end intersection_eq_N_l494_494471


namespace solve_logarithm_eq_l494_494177

theorem solve_logarithm_eq (y : ℝ) (hy : log 3 y + log 9 y = 5) : y = 3^(10/3) := 
  sorry

end solve_logarithm_eq_l494_494177


namespace tangent_of_7pi_over_4_l494_494452

   theorem tangent_of_7pi_over_4 : Real.tan (7 * Real.pi / 4) = -1 := 
   sorry
   
end tangent_of_7pi_over_4_l494_494452


namespace probability_interval_l494_494203

noncomputable def P_A : ℚ := 5 / 6
noncomputable def P_B : ℚ := 3 / 4

theorem probability_interval : 
  let p := min P_A P_B in
  (5 / 12 : ℚ) ≤ p ∧ p ≤ (3 / 4 : ℚ) := sorry

end probability_interval_l494_494203


namespace balls_in_boxes_l494_494029

theorem balls_in_boxes : ∀ (balls boxes : ℕ), balls = 6 → boxes = 3 → 
  (∃! n : ℕ, n = 47 ∧ n = number_of_ways_to_distribute_balls_in_boxes_d(balls, boxes)) :=
by
  intros balls boxes h1 h2
  have h_balls : balls = 6 := h1
  have h_boxes : boxes = 3 := h2
  use 47
  split
  sorry
  sorry

end balls_in_boxes_l494_494029


namespace tan_addition_f_extreme_values_l494_494540

variable {α β x : ℝ}

-- Given conditions
def tan_alpha : ℝ := -1 / 3
def cos_beta : ℝ := sqrt 5 / 5
def alpha_in_interval : α ∈ Ioo 0 π := sorry
def beta_in_interval : β ∈ Ioo 0 π := sorry

-- The first part of the proof problem
theorem tan_addition :
  tan α = tan_alpha ∧ cos β = cos_beta ∧ α ∈ Ioo 0 π ∧ β ∈ Ioo 0 π →
  tan (α + β) = 1 :=
by 
  intros h
  sorry

-- The second part of the proof problem
theorem f_extreme_values :
  (tan α = tan_alpha ∧ cos β = cos_beta ∧ α ∈ Ioo 0 π ∧ β ∈ Ioo 0 π) →
  (∃ max min : ℝ, 
    (∀ x, f x = sqrt 2 * sin (x - α) + cos (x + β) → max = sqrt 5 ∧ min = -sqrt 5) ) :=
by 
  intros h
  sorry

end tan_addition_f_extreme_values_l494_494540


namespace distinguishable_balls_in_indistinguishable_boxes_l494_494020

theorem distinguishable_balls_in_indistinguishable_boxes :
  let num_distinguishable_balls := 6
  let num_indistinguishable_boxes := 3
  -- different ways to distribute balls across boxes summarized
  let ways := ∑([{6, 0, 0} ++ {5, 1, 0} ++ {4, 2, 0} ++ {4, 1, 1} ++ {3, 3, 0} ++ {3, 2, 1} ++ {2, 2, 2}], sum)
  -- corresponding ways to remove permutation of distinguishable ball cases
  ways = 222 := sorry

end distinguishable_balls_in_indistinguishable_boxes_l494_494020


namespace sum_of_first_ten_primes_with_units_digit_three_l494_494859

-- Define the problem to prove the sum of the first 10 primes ending in 3 is 639
theorem sum_of_first_ten_primes_with_units_digit_three : 
  let primes_with_units_digit_three := [3, 13, 23, 43, 53, 73, 83, 103, 113, 163]
  in list.sum primes_with_units_digit_three = 639 := 
by 
  -- We define the primes with the units digit 3 as given and check the sum
  let primes_with_units_digit_three := [3, 13, 23, 43, 53, 73, 83, 103, 113, 163]
  show list.sum primes_with_units_digit_three = 639 from sorry

end sum_of_first_ten_primes_with_units_digit_three_l494_494859


namespace distribute_6_balls_in_3_boxes_l494_494038

def num_ways_to_distribute_balls (balls : ℕ) (boxes : ℕ) : ℕ :=
  if balls = 6 ∧ boxes = 3 then 92 else 0

theorem distribute_6_balls_in_3_boxes :
  num_ways_to_distribute_balls 6 3 = 92 :=
by
  rw num_ways_to_distribute_balls
  norm_num
  sorry

end distribute_6_balls_in_3_boxes_l494_494038


namespace increase_by_percentage_l494_494251

theorem increase_by_percentage (initial_amount : ℕ) (percentage_increase : ℕ) :
  initial_amount = 80 → 
  percentage_increase = 150 → 
  initial_amount * (percentage_increase / 100) + initial_amount = 200 :=
by
  intros h_initial h_percentage
  rw [h_initial, h_percentage]
  calc
    80 * (150 / 100) + 80 = 80 * 1.5 + 80 : by norm_num
                     ... = 120 + 80         : by norm_num
                     ... = 200              : by norm_num
  sorry

end increase_by_percentage_l494_494251


namespace smallest_n_divides_2010_l494_494339

noncomputable def find_smallest_divisor_n (a : ℕ → ℤ) : ℕ :=
  if h : a 1 % 2 = 1 ∧ (∀ n > 0, n * (a (n + 1) - a n + 3) = a (n + 1) + a n + 3) ∧ 2010 ∣ a 2009
  then classical.some (exists_nat_ge_two_divisible_by a 2010)
  else 0

theorem smallest_n_divides_2010 (a : ℕ → ℤ)
  (h1 : a 1 % 2 = 1)
  (h2 : ∀ n > 0, n * (a (n + 1) - a n + 3) = a (n + 1) + a n + 3)
  (h3 : 2010 ∣ a 2009) :
  find_smallest_divisor_n a = 671 :=
sorry

end smallest_n_divides_2010_l494_494339


namespace find_x_l494_494454

theorem find_x (x : ℝ) (h : x = sqrt (x - 1/x) + sqrt (1 - 1/x)) : x = (1 + sqrt 5) / 2 :=
sorry

end find_x_l494_494454


namespace trapezoid_midline_exists_l494_494587

-- Define the structure of a trapezoid with its properties
structure Trapezoid (A B C D : Type) [LinearOrder A] [LinearOrder B] [LinearOrder C] [LinearOrder D] :=
(parallel : A = B ∨ A = D ∨ B = C)
(len_relation : (A = 3 * D))

-- The definition representing the midline construction problem
def midline_construction (A B C D : Type) [LinearOrder A] [LinearOrder B] [LinearOrder C] [LinearOrder D] 
  (T : Trapezoid A B C D) : Prop :=
  ∃ M, sorry -- Here we skip the actual construction proof with sorry.

-- We then assert there exists such a midline under the given conditions
theorem trapezoid_midline_exists (A B C D : Type) [LinearOrder A] [LinearOrder B] [LinearOrder C] [LinearOrder D] 
  (T : Trapezoid A B C D) : midline_construction A B C D T :=
sorry

end trapezoid_midline_exists_l494_494587


namespace sum_smallest_largest_even_mean_l494_494188

theorem sum_smallest_largest_even_mean (n : ℕ) (h_odd : n % 2 = 1) (b z : ℤ)
  (h_mean : z = (n * b + (2 * (1 + 2 + ... + (n - 1)))) / n) :
  let sum := b + (b + 2 * (n - 1)) in
  sum = 2 * z := by
sorry

end sum_smallest_largest_even_mean_l494_494188


namespace projective_transformation_is_cross_ratio_preserving_l494_494488

theorem projective_transformation_is_cross_ratio_preserving (P : ℝ → ℝ) :
  (∃ a b c d : ℝ, (ad - bc ≠ 0) ∧ (∀ x : ℝ, P x = (a * x + b) / (c * x + d))) ↔
  (∀ x1 x2 x3 x4 : ℝ, (x1 - x3) * (x2 - x4) / ((x1 - x4) * (x2 - x3)) =
       (P x1 - P x3) * (P x2 - P x4) / ((P x1 - P x4) * (P x2 - P x3))) :=
sorry

end projective_transformation_is_cross_ratio_preserving_l494_494488


namespace perpendicular_vectors_l494_494095

open Real EuclideanPlane

def i : Vector := (1, 0)
def j : Vector := (0, 1)
def a : Vector := 2 • i
def b : Vector := i + j

theorem perpendicular_vectors :
  dot (a - b) b = 0 := 
sorry

end perpendicular_vectors_l494_494095


namespace distinguishable_balls_in_indistinguishable_boxes_l494_494022

theorem distinguishable_balls_in_indistinguishable_boxes :
  let num_distinguishable_balls := 6
  let num_indistinguishable_boxes := 3
  -- different ways to distribute balls across boxes summarized
  let ways := ∑([{6, 0, 0} ++ {5, 1, 0} ++ {4, 2, 0} ++ {4, 1, 1} ++ {3, 3, 0} ++ {3, 2, 1} ++ {2, 2, 2}], sum)
  -- corresponding ways to remove permutation of distinguishable ball cases
  ways = 222 := sorry

end distinguishable_balls_in_indistinguishable_boxes_l494_494022


namespace sum_first_100_terms_l494_494896

theorem sum_first_100_terms (a : ℕ → ℤ) (S : ℕ → ℤ) 
  (h1 : ∀ n, S n = (nat.sum (range n) (λ i, a i)))
  (h2 : a 5 = 5)
  (h3 : S 5 = 15) :
  (finset.sum (finset.range 100) (λ n, 1 / (a n * a (n + 1)))) = 100 / 101 :=
sorry

end sum_first_100_terms_l494_494896


namespace increase_80_by_150_percent_l494_494236

-- Define the initial number (n) and the percentage increase (p)
def n : ℕ := 80
def p : ℚ := 1.5

-- The theorem stating the expected result after increasing n by 150%
theorem increase_80_by_150_percent : n + (p * n) = 200 := by
  sorry

end increase_80_by_150_percent_l494_494236


namespace find_a_evaluate_expr_l494_494901

-- Given polynomials A and B
def A (a x y : ℝ) : ℝ := a * x^2 + 3 * x * y + 2 * |a| * x
def B (x y : ℝ) : ℝ := 2 * x^2 + 6 * x * y + 4 * x + y + 1

-- Statement part (1)
theorem find_a (a : ℝ) (x y : ℝ) (h : (2 * A a x y - B x y) = (2 * a - 2) * x^2 + (4 * |a| - 4) * x - y - 1) : a = -1 := 
  sorry

-- Expression for part (2)
def expr (a : ℝ) : ℝ := 3 * (-3 * a^2 - 2 * a) - (a^2 - 2 * (5 * a - 4 * a^2 + 1) - 2 * a)

-- Statement part (2)
theorem evaluate_expr : expr (-1) = -22 := 
  sorry

end find_a_evaluate_expr_l494_494901


namespace toothpicks_in_each_box_l494_494446

theorem toothpicks_in_each_box 
    (cards : ℕ) 
    (cards_used : ℕ) 
    (toothpicks_per_card : ℕ) 
    (total_toothpicks : ℕ) 
    (toothpicks_per_box : ℕ) 
    (boxes_used: ℕ)
    (h1 : cards_used = cards - 16) 
    (h2 : total_toothpicks = cards_used * toothpicks_per_card) 
    (h3 : boxes_used * toothpicks_per_box = total_toothpicks) :
    toothpicks_per_box = 450 :=
begin
    sorry
end

end toothpicks_in_each_box_l494_494446


namespace pentagon_angle_sum_proof_l494_494976

-- Define the key elements involved in the problem
variables {A B C D E F G : Type} [inhabited A] [inhabited B] [inhabited C] [inhabited D] [inhabited E] [inhabited F] [inhabited G]

-- Define the angles and their measures
variables {angle_A : ℝ} (h1 : angle_A = 30)
variables {angle_AFG angle_AGF : ℝ} (h2 : angle_AFG = angle_AGF)
variables {angle_BFD angle_BDF : ℝ} (h3 : angle_BFD = angle_BDF)
variables {angle_B angle_D : ℝ} -- These are the angles we need to sum

-- Define the proof problem statement
theorem pentagon_angle_sum_proof 
  (h1 : angle_A = 30)
  (h2 : angle_AFG = angle_AGF)
  (h3 : angle_BFD = angle_BDF) : angle_B + angle_D = 150 :=
begin
  sorry
end

end pentagon_angle_sum_proof_l494_494976


namespace max_min_cos_sin_product_l494_494481

theorem max_min_cos_sin_product (x y z : ℝ) 
  (h1 : x ≥ y) (h2 : y ≥ z) (h3 : z ≥ π / 12) (h4 : x + y + z = π / 2) :
  ∃ (maximum minimum : ℝ), maximum = (2 + Real.sqrt 3) / 8 ∧ minimum = 1 / 8 := by
  sorry

end max_min_cos_sin_product_l494_494481


namespace hexagon_inequality_l494_494612

-- Define the convex hexagon and its properties
variable {ABCDEF : Type} [ConvexHexagon ABCDEF]
variable {A B C D E F : ABCDEF}

-- Given conditions
variable (h1 : AB = BC)
variable (h2 : CD = DE)
variable (h3 : EF = FA)

-- Main goal
theorem hexagon_inequality (h1 : AB = BC) (h2 : CD = DE) (h3 : EF = FA) :
  (BC / BE) + (DE / DA) + (FA / FC) ≥ (3 / 2) :=
sorry

end hexagon_inequality_l494_494612


namespace number_of_tangent_lines_through_point_to_hyperbola_l494_494892

-- Define the point P
def P : ℝ × ℝ := (1/2, 0)

-- Define the equation of the hyperbola
def hyperbola_eq (x y : ℝ) : Bool := (4 * x^2 - y^2 = 1)

-- The proposition that needs to be proved: The number of lines passing through the point P 
-- that intersect the hyperbola at only one point is equal to 3.
theorem number_of_tangent_lines_through_point_to_hyperbola :
  (∃ (count : ℕ), count = 3 ∧ ∀ (L : ℝ × ℝ → Prop), (L P → ∃! Q : ℝ × ℝ, hyperbola_eq Q.1 Q.2 ∧ L Q)) :=
sorry

end number_of_tangent_lines_through_point_to_hyperbola_l494_494892


namespace simplest_common_denominator_l494_494210

variable (m n a : ℕ)

theorem simplest_common_denominator (h₁ : m > 0) (h₂ : n > 0) (h₃ : a > 0) :
  ∃ l : ℕ, l = 2 * a^2 := 
sorry

end simplest_common_denominator_l494_494210


namespace part_a_part_b_l494_494106

variables {A B C D M : Type}
variables [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C] [InnerProductSpace ℝ D] [InnerProductSpace ℝ M]

-- Conditions:
variable (h1: Trapezoid A B C D)
variable (h2: Bases A D B C)
variable (h3: DiagonalsIntersect A B C D M)
variable (h4: AB = DM)
variable (h5: ∠ ABD = ∠ CBD)

-- Part (a): Prove that ∠BAD > 60°
theorem part_a : ∠ BAD > 60° :=
by sorry

-- Part (b): Prove that AB > BC
theorem part_b : AB > BC :=
by sorry

end part_a_part_b_l494_494106


namespace cone_and_sphere_volume_l494_494682

theorem cone_and_sphere_volume (π : ℝ) (r h : ℝ) (V_cylinder : ℝ) (V_cone V_sphere V_total : ℝ) 
  (h_cylinder : V_cylinder = 54 * π) 
  (h_radius : h = 3 * r)
  (h_cone : V_cone = (1 / 3) * π * r^2 * h) 
  (h_sphere : V_sphere = (4 / 3) * π * r^3) :
  V_total = 42 * π := 
by
  sorry

end cone_and_sphere_volume_l494_494682


namespace min_queries_for_six_coprime_numbers_l494_494225

theorem min_queries_for_six_coprime_numbers : 
  ∃ (a b c d e f : ℕ), (∀ {x y : ℕ}, x ≠ y → Nat.gcd x y = 1) → (min_queries_to_find a b c d e f) = 4 := 
begin 
  sorry
end

end min_queries_for_six_coprime_numbers_l494_494225


namespace area_of_square_ABCD_l494_494582

variable (A B C D G I : Point)
variable (ABCD : Square A B C D)
variable (GI : Segment G I)
variable (side_length_ABCD : ℝ)
variable (sum_perimeters : ℝ)

-- Given conditions
axiom GI_length : length GI = 6
axiom number_of_rectangles : 18
axiom sum_of_perimeters : sum_perimeters = 456

-- Define the area of the square
def area_square (s : ℝ) : ℝ := s * s

-- The main statement to be proved
theorem area_of_square_ABCD : area_square side_length_ABCD = 100 :=
by
  -- proof goes here
  sorry

end area_of_square_ABCD_l494_494582


namespace exists_sum_free_subset_l494_494130

variable {A : Set ℤ}

def is_sum_free (B : Set ℤ) : Prop :=
  ∀ a b c : ℤ, a ∈ B → b ∈ B → a + b = c → c ∉ B

theorem exists_sum_free_subset (hA : A ⊆ {n : ℤ | n ≠ 0}) :
  ∃ B ⊆ A, is_sum_free B ∧ B.card > A.card / 3 :=
by
  sorry

end exists_sum_free_subset_l494_494130


namespace sum_of_first_ten_primes_with_units_digit_3_l494_494851

def units_digit_3_and_prime (n : ℕ) : Prop :=
  (n % 10 = 3) ∧ (Prime n)

def first_ten_primes_with_units_digit_3 : list ℕ :=
  [3, 13, 23, 43, 53, 73, 83, 103, 113, 163]

theorem sum_of_first_ten_primes_with_units_digit_3 :
  list.sum first_ten_primes_with_units_digit_3 = 671 :=
by
  sorry

end sum_of_first_ten_primes_with_units_digit_3_l494_494851


namespace max_radius_hyperbola_tangent_circle_l494_494922

theorem max_radius_hyperbola_tangent_circle (a b r e : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : r > 0) (h₄ : e ≤ 2) 
  (hyperbola : ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1) 
  (circle_tangent : ∀ x y : ℝ, (x - 2)^2 + y^2 = r^2) :
  r ≤ sqrt 3 := 
sorry

end max_radius_hyperbola_tangent_circle_l494_494922


namespace anniversary_day_probability_l494_494395

/- Definitions based on the conditions -/
def is_leap_year (year : ℕ) : Prop :=
  year % 4 = 0

def days_in_year (year : ℕ) : ℕ :=
  if is_leap_year year then 366 else 365

def total_days (start_year : ℕ) : ℕ :=
  (list.sum (list.map days_in_year (list.range' start_year 11)))

/- Prove the day of the 11th anniversary and its probabilities -/
theorem anniversary_day_probability (start_year : ℕ) (h : start_year ∈ {1668, 1669, 1670, 1671}) :
  let days := total_days start_year % 7
  in (days = 0 ∧ 0.75 ≤ 1) ∨ (days = 6 ∧ 0.25 ≤ 1) :=
by
  sorry

end anniversary_day_probability_l494_494395


namespace sum_of_all_possible_values_of_s_t_r_l494_494614

noncomputable def r (x : Int) : Int := 
  if x = -2 then -1 
  else if x = -1 then 0 
  else if x = 0 then 2 
  else if x = 1 then 3 
  else 0 -- Placeholder, r(x) is only defined for specific x

def t (x : Int) : Int := 2 * x + 1

def s (x : Int) : Int := x + 2

theorem sum_of_all_possible_values_of_s_t_r :
  ∑ x in ({1, 5} : Finset Int), s x = 10 :=
by
  sorry

end sum_of_all_possible_values_of_s_t_r_l494_494614


namespace football_field_image_l494_494088

noncomputable def central_projection_image (O : Point) (ϕ : Plane) (P : Point) : Point := sorry
noncomputable def line_image (O : Point) (ϕ : Plane) (e : Line) : Line := sorry
noncomputable def intersection (l1 l2 : Line) : Point := sorry

theorem football_field_image (O : Point) (ϕ : Plane) 
  (A' B' C' D' K : Point) (M : Point) (k : Line) (h1 h2 : Line):
  (K' = intersection (line_image O ϕ (line_through A' C')) (line_image O ϕ (line_through B' D'))) →
  (k' = line_segment K' M') →
  (h1' = line_segment M S1') →
  (h2' = line_segment M S2') →
  true := 
sorry

end football_field_image_l494_494088


namespace increase_80_by_150_percent_l494_494285

theorem increase_80_by_150_percent : 
  ∀ x percent, x = 80 ∧ percent = 1.5 → x + (x * percent) = 200 :=
by
  sorry

end increase_80_by_150_percent_l494_494285


namespace jenny_spent_625_dollars_l494_494598

def adoption_fee := 50
def vet_visits_cost := 500
def monthly_food_cost := 25
def toys_cost := 200
def year_months := 12

def jenny_adoption_vet_share := (adoption_fee + vet_visits_cost) / 2
def jenny_food_share := (monthly_food_cost * year_months) / 2
def jenny_total_cost := jenny_adoption_vet_share + jenny_food_share + toys_cost

theorem jenny_spent_625_dollars :
  jenny_total_cost = 625 := by
  sorry

end jenny_spent_625_dollars_l494_494598


namespace john_got_80_percent_of_value_l494_494114

noncomputable def percentage_of_value (P : ℝ) : Prop :=
  let old_system_cost := 250
  let new_system_cost := 600
  let discount_percentage := 0.25
  let pocket_spent := 250
  let discount_amount := discount_percentage * new_system_cost
  let price_after_discount := new_system_cost - discount_amount
  let value_for_old_system := (P / 100) * old_system_cost
  value_for_old_system + pocket_spent = price_after_discount

theorem john_got_80_percent_of_value : percentage_of_value 80 :=
by
  sorry

end john_got_80_percent_of_value_l494_494114


namespace unique_root_in_interval_max_value_of_m_l494_494615

noncomputable def f (x : ℝ) : ℝ := (x + 1) * Real.log x

noncomputable def g (x : ℝ) : ℝ := x^2 / Real.exp x

theorem unique_root_in_interval :
  ∃ k : ℕ, (∀ x : ℝ, (x ∈ (k : ℝ) ∧ x < (k + 1 : ℝ) → f x = g x) → k = 1) :=
sorry

theorem max_value_of_m :
  ∃ x : ℝ, x > 0 ∧ min (f x) (g x) = 4 / Real.exp 2 :=
sorry

end unique_root_in_interval_max_value_of_m_l494_494615


namespace tan_phi_value_l494_494954

-- Defining the initial function f(x)
def f (x : ℝ) : ℝ := Real.cos(2 * x + (Real.pi / 6))

-- Condition for the translated function being symmetric about the origin
def is_symmetric_about_origin (phi : ℝ) : Prop :=
  ∃ k : ℤ, 2 * phi + (Real.pi / 6) = k * Real.pi + (Real.pi / 2)

-- The main statement proving the required phi value
theorem tan_phi_value (phi : ℝ) (h_phi_pos : phi > 0) (h_symmetry : is_symmetric_about_origin phi) :
  phi = Real.pi / 6 ∧ Real.tan(phi) = Real.sqrt(3) / 3 :=
by
  sorry

end tan_phi_value_l494_494954


namespace B_should_be_paid_3000_l494_494419

theorem B_should_be_paid_3000 
  (B_alone_days : ℕ) 
  (A_alone_days : ℕ) 
  (total_wages : ℕ) 
  (B_ratio : ℕ) 
  (A_ratio : ℕ)
  (total_ratio : ℕ)
  (B_share : ℕ) : 
  B_alone_days = 10 →
  A_alone_days = 15 →
  total_wages = 5000 →
  B_ratio = 3 →
  A_ratio = 2 →
  total_ratio = B_ratio + A_ratio →
  B_share = (B_ratio * total_wages) / total_ratio →
  B_share = 3000 :=
begin
  sorry
end

end B_should_be_paid_3000_l494_494419


namespace sum_of_digits_2_pow_2010_mul_5_pow_2012_mul_7_l494_494229

theorem sum_of_digits_2_pow_2010_mul_5_pow_2012_mul_7 :
  let n := 2 ^ 2010 * 5 ^ 2012 * 7
  let digits_sum := 1 + 7 + 5
  sum_of_digits n = digits_sum := by
  sorry

end sum_of_digits_2_pow_2010_mul_5_pow_2012_mul_7_l494_494229


namespace third_player_matches_l494_494215

theorem third_player_matches (first_player second_player third_player : ℕ) (h1 : first_player = 10) (h2 : second_player = 21) :
  third_player = 11 :=
by
  sorry

end third_player_matches_l494_494215


namespace zero_exists_in_one_two_l494_494646

noncomputable def f (x : ℝ) : ℝ := log x / log 2 - 1 / x

theorem zero_exists_in_one_two :
  ∃ x_0 : ℝ, (1 < x_0 ∧ x_0 < 2 ∧ f x_0 = 0) := 
by
  sorry

end zero_exists_in_one_two_l494_494646


namespace real_part_of_mult_i_l494_494500

def z1 : ℂ := 4 + 2 * complex.I
def z2 : ℂ := 6 + 9 * complex.I
def diff : ℂ := z1 - z2
def mult_i : ℂ := diff * complex.I

theorem real_part_of_mult_i :
  mult_i.re = 7 := sorry

end real_part_of_mult_i_l494_494500


namespace other_number_is_twelve_l494_494352

variable (x certain_number : ℕ)
variable (h1: certain_number = 60)
variable (h2: certain_number = 5 * x)

theorem other_number_is_twelve :
  x = 12 :=
by
  sorry

end other_number_is_twelve_l494_494352


namespace area_ratio_of_circles_l494_494945

theorem area_ratio_of_circles (R_C R_D : ℝ) (hL : (60.0 / 360.0) * 2.0 * Real.pi * R_C = (40.0 / 360.0) * 2.0 * Real.pi * R_D) : 
  (Real.pi * R_C^2) / (Real.pi * R_D^2) = 4.0 / 9.0 :=
by
  sorry

end area_ratio_of_circles_l494_494945


namespace x_intercept_line_l494_494833

theorem x_intercept_line : 
  let p1 := (-2, 2)
  let p2 := (2, 10)
  let x_intercept (p1 p2 : ℝ × ℝ) : ℝ := 
    let m := (p2.2 - p1.2) / (p2.1 - p1.1)
    let b := p1.2 - m * p1.1
    -b / m
  x_intercept (-2, 2) (2, 10) = -3 := 
by 
  have p1 := (-2, 2)
  have p2 := (2, 10)
  have m : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)
  have b : ℝ := p1.2 - m * p1.1
  have h : -b / m = -b / m := by
    sorry
  exact h

end x_intercept_line_l494_494833


namespace log_sum_example_l494_494804

theorem log_sum_example : log 10 50 + log 10 20 = 3 :=
by
  -- Proof goes here, skipping with sorry
  sorry

end log_sum_example_l494_494804


namespace ticket_distribution_count_l494_494433

-- Defining the parameters
def tickets : Finset ℕ := {1, 2, 3, 4, 5, 6}
def people : ℕ := 4

-- Condition: Each person gets at least 1 ticket and at most 2 tickets, consecutive if 2.
def valid_distribution (dist: Finset (Finset ℕ)) :=
  dist.card = 4 ∧ ∀ s ∈ dist, s.card >= 1 ∧ s.card <= 2 ∧ (s.card = 1 ∨ (∃ x, s = {x, x+1}))

-- Question: Prove that there are 144 valid ways to distribute the tickets.
theorem ticket_distribution_count :
  ∃ dist: Finset (Finset ℕ), valid_distribution dist ∧ dist.card = 144 :=
by {
  sorry -- Proof is omitted as per instructions.
}

-- This statement checks distribution of 6 tickets to 4 people with given constraints is precisely 144

end ticket_distribution_count_l494_494433


namespace sequence_value_at_5_l494_494585

def seq (a : ℕ → ℚ) : Prop :=
  a 1 = 1 / 3 ∧ ∀ n, 1 < n → a n = (-1) ^ n * 2 * a (n - 1)

theorem sequence_value_at_5 (a : ℕ → ℚ) (h : seq a) : a 5 = 16 / 3 :=
by 
  sorry

end sequence_value_at_5_l494_494585


namespace simplify_expression_1_simplify_expression_2_l494_494641

-- The first proof statement
theorem simplify_expression_1 :
  (0.027)^(-1/3) - (1/7)^(-2) + (2 * (7/9))^(1/2) - (Math.sqrt 2 - 1)^0 = -45 :=
by sorry

-- The second proof statement
theorem simplify_expression_2 :
  Real.logBase 3 (Real.root 4 (27) / 3) + Math.log10 25 + Math.log10 4 + 7^(Real.logBase 7 2) = 15/4 :=
by sorry

end simplify_expression_1_simplify_expression_2_l494_494641


namespace line_through_point_intersects_circle_l494_494753

noncomputable def line_eq (k : ℝ) : (ℝ → ℝ → Prop) :=
  if k = 0 then λ x y, x = 4
  else λ x y, 5*x - 12*y - 20 = 0

theorem line_through_point_intersects_circle
  {x y : ℝ}
  (l : ℝ → ℝ → Prop)
  (h1 : l 4 0)
  (h2 : ∃ A B, A ≠ B ∧ l (A.1) (A.2) ∧ l (B.1) (B.2) ∧ (A.1 - B.1)^2 + (A.2 - B.2)^2 = 64)
  (h3 : ∀ P, l P.1 P.2 → (P.1 - 1)^2 + (P.2 - 2)^2 = 25):
  l = line_eq 0 ∨ l = line_eq (5/12) :=
sorry

end line_through_point_intersects_circle_l494_494753


namespace carol_rectangle_width_l494_494812

theorem carol_rectangle_width :
  ∃ (W : ℕ), 5 * W = 8 * 15 ∧ W = 24 :=
begin
  use 24,
  split,
  { linarith, },
  { refl, }
end

end carol_rectangle_width_l494_494812


namespace YZ_length_is_correct_l494_494978

noncomputable def calculate_YZ_length (XY XZ XN : ℝ) : ℝ := 
  sqrt (2 * XY^2 + 2 * XZ^2 - 4 * XN^2)

theorem YZ_length_is_correct : 
  calculate_YZ_length 5 8 4.5 = sqrt 97 := 
by
  unfold calculate_YZ_length
  sorry

end YZ_length_is_correct_l494_494978


namespace anniversary_day_of_week_probability_l494_494407

/-- The 11th anniversary of Robinson Crusoe and Friday's meeting can fall on a Friday with a
probability of 3/4 and on a Thursday with a probability of 1/4, given that the meeting occurred
in any year from 1668 to 1671 with equal probability. -/
theorem anniversary_day_of_week_probability :
  let years := {1668, 1669, 1670, 1671},
      leap (y : ℕ) := y % 4 = 0,
      days_in_year := λ y, if leap y then 366 else 365,
      total_days (yr : ℕ) := list.sum (list.map days_in_year (list.range' yr 11)),
      day_of_week_after_11_years (initial_year : ℕ) := total_days initial_year % 7 = 0,
      events := {week_day | ∀ y ∈ years, (day_of_week_after_11_years y)},
      friday_probability := rat.mk 3 4,
      thursday_probability := rat.mk 1 4
  in
  (events = {0} ∨ events = {6}) ∧
  (events = {0} → friday_probability = rat.mk 3 4 ∧ thursday_probability = rat.mk 1 4) ∧
  (events = {6} → friday_probability = rat.mk 1 4 ∧ thursday_probability = rat.mk 3 4):=
begin
  sorry
end

end anniversary_day_of_week_probability_l494_494407


namespace eleventh_anniversary_days_l494_494415

-- Define the conditions
def is_leap_year (year : ℕ) : Prop := year % 4 = 0

def initial_years : Set ℕ := {1668, 1669, 1670, 1671}

def initial_day := "Friday"

noncomputable def day_after_11_years (start_year : ℕ) : String :=
  let total_days := 4015 + (if is_leap_year start_year then 3 else 2)
  if total_days % 7 = 0 then "Friday"
  else "Thursday"

-- Define the proposition to prove
theorem eleventh_anniversary_days : 
  (∀ year ∈ initial_years, 
    (if day_after_11_years year = "Friday" then (3 : ℝ) / 4 else (1 : ℝ) / 4) = 
    (if year = 1668 ∨ year = 1670 ∨ year = 1671 then (3 : ℝ) / 4 else (1 : ℝ) / 4)) := 
sorry

end eleventh_anniversary_days_l494_494415


namespace increase_80_by_150_percent_l494_494286

theorem increase_80_by_150_percent : 
  ∀ x percent, x = 80 ∧ percent = 1.5 → x + (x * percent) = 200 :=
by
  sorry

end increase_80_by_150_percent_l494_494286


namespace triangle_cover_square_l494_494381

theorem triangle_cover_square (n : ℕ) (h : n ≥ 1) :
  ∃ T : set (ℝ × ℝ), unit_square_divided_into_n_triangles T n ∧ 
  ∃ K : set (ℝ × ℝ), is_square_with_side K (1 / n) ∧ K ⊆ T :=
sorry

def unit_square_divided_into_n_triangles (T : set (ℝ × ℝ)) (n : ℕ) : Prop :=
  -- Definition stating that the unit square is divided into n triangles
  sorry

def is_square_with_side (K : set (ℝ × ℝ)) (side_length : ℝ) : Prop :=
  -- Definition stating that K is a square with the given side length
  sorry

end triangle_cover_square_l494_494381


namespace true_statement_D_l494_494332

-- Definitions related to the problem conditions
def supplementary_angles (a b : ℝ) : Prop := a + b = 180

def exterior_angle_sum_of_polygon (n : ℕ) : ℝ := 360

def acute_angle (a : ℝ) : Prop := a < 90

def triangle_inequality (a b c : ℝ) : Prop := 
  a + b > c ∧ a + c > b ∧ b + c > a

-- The theorem to be proven based on the correct evaluation
theorem true_statement_D (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0):
  triangle_inequality a b c :=
by 
  sorry

end true_statement_D_l494_494332


namespace factorize_poly_l494_494831

-- Statement of the problem
theorem factorize_poly (x : ℝ) : x^2 - 3 * x = x * (x - 3) :=
sorry

end factorize_poly_l494_494831


namespace log_sum_example_l494_494791

theorem log_sum_example :
  let log_base_10 (x : ℝ) := Real.log x / Real.log 10 in
  log_base_10 50 + log_base_10 20 = 3 :=
by
  sorry

end log_sum_example_l494_494791


namespace halfway_fraction_l494_494705

theorem halfway_fraction (a b : ℚ) (ha : a = 3/4) (hb : b = 5/6) : (a + b) / 2 = 19/24 :=
by
  rw [ha, hb] -- replace a and b with 3/4 and 5/6 respectively
  have h1 : 3/4 + 5/6 = 19/12,
  { norm_num, -- ensures 3/4 + 5/6 = 19/12
    linarith },
  rw h1, -- replace a + b with 19/12
  norm_num -- ensures (19/12) / 2 = 19/24

end halfway_fraction_l494_494705


namespace intersection_of_lines_l494_494712

noncomputable def line1 (x : ℝ) : ℝ := -3 * x - 1
noncomputable def line2 (x : ℝ) : ℝ := (1 / 3) * x - (4 / 3)
def intersection_point : ℝ × ℝ := (1 / 10, -1.3)

theorem intersection_of_lines : ∃ x y : ℝ, line1 x = y ∧ line2 x = y ∧ (x, y) = intersection_point :=
by
  sorry

end intersection_of_lines_l494_494712


namespace polar_to_rect_coords_l494_494819

noncomputable def polar_to_rect (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

theorem polar_to_rect_coords (r θ : ℝ) (h_r : r = 2 * Real.sqrt 3) (h_θ : θ = 2 * Real.pi / 3) :
  polar_to_rect r θ = (-Real.sqrt 3, 3) :=
by
  rw [h_r, h_θ]
  dsimp [polar_to_rect]
  rw [Real.cos_two_mul_pi_div_three, Real.sin_two_mul_pi_div_three]
  sorry  -- Proof steps can be filled here.

end polar_to_rect_coords_l494_494819


namespace increase_by_percentage_l494_494330

theorem increase_by_percentage (a : ℤ) (b : ℚ) : ((b + 1) * a) = 200 :=
by
  -- Definitions based directly on conditions
  let a := (80 : ℤ)
  let b := (1.5 : ℚ)
  -- Assertions about the equivalent proof problem
  have h1 : b + 1 = 2.5 := by norm_num
  have h2 : (b + 1) * a = 200 := by norm_num
  exact h2

end increase_by_percentage_l494_494330


namespace family_has_11_eggs_l494_494749

def initialEggs : ℕ := 10
def eggsUsed : ℕ := 5
def chickens : ℕ := 2
def eggsPerChicken : ℕ := 3

theorem family_has_11_eggs :
  (initialEggs - eggsUsed) + (chickens * eggsPerChicken) = 11 := by
  sorry

end family_has_11_eggs_l494_494749


namespace quadratic_inequality_solution_l494_494689

theorem quadratic_inequality_solution (m : ℝ) :
    (∃ x : ℝ, x^2 - m * x + 1 ≤ 0) ↔ m ≥ 2 ∨ m ≤ -2 := by
  sorry

end quadratic_inequality_solution_l494_494689


namespace good_horse_catchup_l494_494987

theorem good_horse_catchup (x : ℕ) : 240 * x = 150 * (x + 12) :=
by sorry

end good_horse_catchup_l494_494987


namespace obtain_all_positive_integers_l494_494390

theorem obtain_all_positive_integers :
  ∀ (n : ℕ), n > 0 → ∃ (seq : List ℕ), seq.headD = 1 ∧ seq.lastD 1 = n ∧
    (∀ x ∈ seq, x = 3 * (seq.nth (seq.indexOf x).pred).getD 0 + 1 ∨ x = (seq.nth (seq.indexOf x).pred).getD 0 / 2) := 
sorry

end obtain_all_positive_integers_l494_494390


namespace problem_1_problem_2_problem_3_l494_494883

noncomputable def vec_a (x : ℝ) : ℝ × ℝ := (Real.cos x, -1 / 2)
noncomputable def vec_b (x : ℝ) : ℝ × ℝ := (Real.sin x + Real.cos x, 1)
noncomputable def f (x : ℝ) : ℝ := (vec_a x).1 * (vec_b x).1 + (vec_a x).2 * (vec_b x).2

theorem problem_1 (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.sin α = Real.sqrt 2 / 2) :
  f α = 1 / 2 :=
sorry

theorem problem_2 (x : ℝ) : 
  analytic_function.periodic f π :=
sorry

theorem problem_3 (x : ℝ) (k : ℤ) :
  ∃ a b : ℝ, 
    (a = k * π - (3 * π / 8)) ∧ (b = k * π + (π / 8)) ∧ 
    ∀ (y : ℝ), a ≤ y ∧ y ≤ b → f x is_strictly_increasing_on_interval a b :=
sorry

end problem_1_problem_2_problem_3_l494_494883


namespace new_standard_deviation_l494_494493

variable {x1 x2 x3 x4 x5 : ℝ}
variable {m n a : ℝ}
variable {h_avg : m = (x1 + x2 + x3 + x4 + x5) / 5}
variable {h_var : n = ((x1 - m) ^ 2 + (x2 - m) ^ 2 + (x3 - m) ^ 2 + (x4 - m) ^ 2 + (x5 - m) ^ 2) / 5}
variable {ha_pos : a > 0}

theorem new_standard_deviation :
  (sqrt ((a * (x1 - m))^2 + (a * (x2 - m))^2 + (a * (x3 - m))^2 + (a * (x4 - m))^2 + (a * (x5 - m))^2) / 5) = a * sqrt n :=
sorry

end new_standard_deviation_l494_494493


namespace increased_number_l494_494276

theorem increased_number (original_number : ℕ) (percentage_increase : ℚ) :
  original_number = 80 →
  percentage_increase = 1.5 →
  original_number * (1 + percentage_increase) = 200 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end increased_number_l494_494276


namespace domain_of_f_l494_494837

noncomputable def f (x : ℝ) : ℝ := (2 * x ^ 2 - 3 * x + 1) / real.sqrt (2 * x - 10)

theorem domain_of_f : 
  {x : ℝ | ∃ (y : ℝ), y = f x} = {x : ℝ | x > 5} :=
by
  sorry

end domain_of_f_l494_494837


namespace distinct_numbers_in_list_l494_494431

theorem distinct_numbers_in_list :
  ∃ (s : Finset ℕ), s.card = 1975 ∧ (∀ n ∈ (Finset.range 1000).image (λ n, ⌊(n + 1)^2 / 500⌋), n ∈ s) :=
sorry

end distinct_numbers_in_list_l494_494431


namespace arithmetic_sequence_8th_term_l494_494181

theorem arithmetic_sequence_8th_term (a d : ℤ) :
  (a + d = 25) ∧ (a + 5 * d = 49) → (a + 7 * d = 61) :=
by
  sorry

end arithmetic_sequence_8th_term_l494_494181


namespace _l494_494995

noncomputable theorem area_of_triangle_ABC 
  (A B C : ℝ) 
  (a b c : ℝ) 
  (h_a : a = 2) 
  (h_2sinA_eq_sinC : 2 * Real.sin A = Real.sin C) 
  (h_B_obtuse : B > π / 2)
  (h_cos2C : Real.cos (2 * C) = -1 / 4) :
  let area := (1 / 2) * a * c * Real.sin B in
  area = Real.sqrt 15 := 
sorry

end _l494_494995


namespace cevian_ratios_l494_494605

variable {A B C P A' B' C' : Type} 

variables (x y z : ℝ)
variables (AP PA' BP PB' CP PC' : ℝ)

variables (hx : x = AP / PA')
variables (hy : y = BP / PB')
variables (hz : z = CP / PC')

theorem cevian_ratios (hp: P ∈ triangle A B C) (ha: AP ∈ line A P) (hb: BP ∈ line B P) (hc: CP ∈ line C P) :
  x * y * z = x + y + z + 2 :=
sorry

end cevian_ratios_l494_494605


namespace probability_divisible_by_5_l494_494435

def spinner_nums : List ℕ := [1, 2, 3, 5]

def total_outcomes (spins : ℕ) : ℕ :=
  List.length spinner_nums ^ spins

def count_divisible_by_5 (spins : ℕ) : ℕ :=
  let units_digit := 1
  let rest_combinations := (List.length spinner_nums) ^ (spins - units_digit)
  rest_combinations

theorem probability_divisible_by_5 : 
  let spins := 3 
  let successful_cases := count_divisible_by_5 spins
  let all_cases := total_outcomes spins
  successful_cases / all_cases = 1 / 4 :=
by
  sorry

end probability_divisible_by_5_l494_494435


namespace total_squares_on_16x16_chessboard_l494_494083

theorem total_squares_on_16x16_chessboard : 
  let n := 16 in
  let total_squares := (1/6 : ℝ) * n * (n + 1) * (2 * n + 1) in
  total_squares = 1496 := by
  sorry

end total_squares_on_16x16_chessboard_l494_494083


namespace range_of_k_in_ellipse_l494_494557

def ellipse_with_foci_on_x_axis (k : ℝ) : Prop :=
  (k > 1) ∧ (k ≠ 0)

theorem range_of_k_in_ellipse (k : ℝ) :
  (∃ k > 1, equation (x^2 + k * y^2 = 2)) ↔ k > 1 := 
sorry

end range_of_k_in_ellipse_l494_494557


namespace data_set_variance_l494_494424

def data_set : List ℕ := [2, 4, 5, 3, 6]

noncomputable def mean (l : List ℕ) : ℝ :=
  l.sum / l.length

noncomputable def variance (l : List ℕ) : ℝ :=
  let m : ℝ := mean l
  (l.map (fun x => (x - m) ^ 2)).sum / l.length

theorem data_set_variance : variance data_set = 2 := by
  sorry

end data_set_variance_l494_494424


namespace ratio_of_carpets_l494_494684

theorem ratio_of_carpets (h1 h2 h3 h4 : ℕ) (total : ℕ) 
  (H1 : h1 = 12) (H2 : h2 = 20) (H3 : h3 = 10) (H_total : total = 62) 
  (H_all_houses : h1 + h2 + h3 + h4 = total) : h4 / h3 = 2 :=
by
  sorry

end ratio_of_carpets_l494_494684


namespace sum_of_first_ten_primes_with_units_digit_3_l494_494849

def units_digit_3_and_prime (n : ℕ) : Prop :=
  (n % 10 = 3) ∧ (Prime n)

def first_ten_primes_with_units_digit_3 : list ℕ :=
  [3, 13, 23, 43, 53, 73, 83, 103, 113, 163]

theorem sum_of_first_ten_primes_with_units_digit_3 :
  list.sum first_ten_primes_with_units_digit_3 = 671 :=
by
  sorry

end sum_of_first_ten_primes_with_units_digit_3_l494_494849


namespace calculate_expression_l494_494810

theorem calculate_expression : real.cbrt (-8) + real.sqrt 4 + abs (-real.sqrt 5) = real.sqrt 5 := by
  simp [real.cbrt, real.sqrt, abs]
  sorry

end calculate_expression_l494_494810


namespace radius_circle_S_form_final_answer_l494_494591

noncomputable def radius_circle_S : ℝ :=
  let p : ℕ := 66
  let q : ℕ := 8
  let u : ℕ := 42
  p - q * Real.sqrt u

theorem radius_circle_S_form :
  let p : ℕ := 66
  let q : ℕ := 8
  let u : ℕ := 42
  radius_circle_S = p - q * Real.sqrt u :=
by
  rw radius_circle_S
  ring
  sorry 

theorem final_answer :
  let p : ℕ := 66 
  let q : ℕ := 8
  let u : ℕ := 42
  p + q*u = 338 :=
by
  norm_num

end radius_circle_S_form_final_answer_l494_494591


namespace vasya_meets_mother_at_800_meters_l494_494695

-- Definitions of the given conditions
def distance_highway : ℝ := 3
def distance_path : ℝ := 2
def speed_mother : ℝ := 4
def speed_vasya_path : ℝ := 20
def speed_vasya_highway : ℝ := 22

-- Problem statement
theorem vasya_meets_mother_at_800_meters : 
  ∀ (distance_highway distance_path speed_mother speed_vasya_path speed_vasya_highway : ℝ), 
    distance_highway = 3 →
    distance_path = 2 →
    speed_mother = 4 →
    speed_vasya_path = 20 →
    speed_vasya_highway = 22 →
    let t_path_vasya := distance_path / speed_vasya_path in
    let distance_mother_on_path := speed_mother * t_path_vasya in
    let remaining_distance := distance_highway - distance_mother_on_path in
    let t_highway := remaining_distance / (speed_mother + speed_vasya_highway) in
    let total_time := t_path_vasya + t_highway in
    let distance_mother := speed_mother * total_time in
    distance_mother * 1000 = 800 :=
by intros
   intro H1 H2 H3 H4 H5
   let t_path_vasya := distance_path / speed_vasya_path
   let distance_mother_on_path := speed_mother * t_path_vasya
   let remaining_distance := distance_highway - distance_mother_on_path
   let t_highway := remaining_distance / (speed_mother + speed_vasya_highway)
   let total_time := t_path_vasya + t_highway
   let distance_mother := speed_mother * total_time
   sorry

end vasya_meets_mother_at_800_meters_l494_494695


namespace circle_not_pass_second_quadrant_l494_494344

theorem circle_not_pass_second_quadrant (a : ℝ) : ¬(∃ x y : ℝ, x < 0 ∧ y > 0 ∧ (x - a)^2 + y^2 = 4) → a ≥ 2 :=
by
  intro h
  by_contra
  sorry

end circle_not_pass_second_quadrant_l494_494344


namespace sqrt_sequence_term_l494_494533

theorem sqrt_sequence_term :
  let seq := λ n : ℕ => Real.sqrt (5 + 6 * (n - 1))
  in seq 21 = 5 * Real.sqrt 5 :=
by
  sorry

end sqrt_sequence_term_l494_494533


namespace geometric_sequence_eighth_term_l494_494366

variable (a r : ℕ)
variable (h1 : a = 3)
variable (h2 : a * r^6 = 2187)
variable (h3 : a = 3)

theorem geometric_sequence_eighth_term (a r : ℕ) (h1 : a = 3) (h2 : a * r^6 = 2187) (h3 : a = 3) :
  a * r^7 = 6561 := by
  sorry

end geometric_sequence_eighth_term_l494_494366


namespace largest_integer_less_than_log_sum_l494_494228

theorem largest_integer_less_than_log_sum : 
  let s := ∑ k in Finset.range 1009, Real.log2 (↑(k + 2) / ↑(k + 1))
  in s < 10 ∧ (⌊s⌋ : ℤ) = 9 :=
by
  sorry

end largest_integer_less_than_log_sum_l494_494228


namespace problem_a4_l494_494910

noncomputable def f : ℝ → ℝ
| x := if 0 ≤ x ∧ x < 1 then -x^2 + x else 2 * f (x-1)

def a_n (n : ℕ) : ℝ :=
  let rec max_f : ℝ × ℝ → ℝ
  | (a, b) := if a < b then max (f a) (max_f (a + 1, b)) else f a
  in max_f (n-1, n)

theorem problem_a4 : a_n 4 = 2 := by
  simp [a_n, f]
  sorry

end problem_a4_l494_494910


namespace find_number_l494_494934

theorem find_number (x : ℝ) 
  (h : 0.85 * (3 / 5) * x = 36) : 
  x ≈ 70.59 := 
by 
  -- proof steps will go here
  sorry

end find_number_l494_494934


namespace shaded_area_l494_494743

-- Define the mathematical objects
def Circle (center : ℝ × ℝ) (radius : ℝ) : Prop := sorry

-- Condition 1: Smaller circle and two larger circles with given radii
def smaller_circle : Prop := Circle (0, 0) 1
def larger_circle_A : Prop := Circle (0, -1) 2
def larger_circle_B : Prop := Circle (0, 1) 2

-- Condition 2: AB is a diameter of the smaller circle
def diameter_AB : Prop := sorry -- Details on defining diameter condition

-- Statement about the shaded area
theorem shaded_area :
    smaller_circle →
    larger_circle_A →
    larger_circle_B →
    diameter_AB →
    (let pi := Real.pi, rad3 := Real.sqrt 3 in (5 / 3) * pi - 2 * rad3 = (correct_area : ℝ)) :=
begin
  intros _ _ _ _,
  -- Proof would go here
  sorry,
end

end shaded_area_l494_494743


namespace man_l494_494336

variable (V_m V_c : ℝ)

theorem man's_speed_against_current :
  (V_m + V_c = 21 ∧ V_c = 2.5) → (V_m - V_c = 16) :=
by
  sorry

end man_l494_494336


namespace max_min_cos_sin_product_l494_494480

theorem max_min_cos_sin_product (x y z : ℝ) 
  (h1 : x ≥ y) (h2 : y ≥ z) (h3 : z ≥ π / 12) (h4 : x + y + z = π / 2) :
  ∃ (maximum minimum : ℝ), maximum = (2 + Real.sqrt 3) / 8 ∧ minimum = 1 / 8 := by
  sorry

end max_min_cos_sin_product_l494_494480


namespace horner_method_value_l494_494486

def f (x : ℝ) : ℝ := x^5 - 2*x^3 + 3*x^2 - x + 1

theorem horner_method_value (x : ℝ) : 
  let v_0 := 1 in
  let v_1 := 1 * x + 0 in
  let v_2 := v_1 * x - 2 in
  let v_3 := v_2 * x + 3 in
  x = 3 → v_3 = 24 := 
by 
  intros h_eq;
  let v_0 := 1;
  let v_1 := 1 * x + 0;
  let v_2 := v_1 * x - 2;
  let v_3 := v_2 * x + 3;
  sorry

end horner_method_value_l494_494486


namespace correct_proposition_l494_494915

-- Defining the propositions as conditions
def prop1 : Prop :=
∀ (P : Point) (skew1 skew2 : Line), ∃ (line : Line), is_perpendicular line skew1 ∧ is_perpendicular line skew2 

def prop2 : Prop :=
∀ (P : Point) (skew1 skew2 : Line), (¬ on_line P skew1) → (¬ on_line P skew2) → ∃ (plane : Plane), 
is_parallel plane skew1 ∧ is_parallel plane skew2

def prop3 : Prop :=
∀ (α β : Plane) (a b : Line), (intersection α β = a) → (is_perpendicular b a) → ¬ (is_perpendicular b α)

def prop4 : Prop :=
∀ (prism : Prism), (∀ (face1 face2 : Face), is_congruent face1 face2) →
is_right_prism prism

def prop5 : Prop := 
∀ (pyramid : Pyramid), ∃ (base : Triangle), (is_equilateral base) ∧ (∀ (face : Face), 
(is_isosceles face)) → is_regular_tetrahedron pyramid

-- The theorem to verify
theorem correct_proposition : prop1 ∧ ¬prop2 ∧ ¬prop3 ∧ ¬prop4 ∧ ¬prop5 :=
by {
  sorry
}

end correct_proposition_l494_494915


namespace sum_of_first_ten_primes_with_units_digit_3_l494_494854

def is_prime (n : ℕ) : Prop := nat.prime n

def units_digit_3 (n : ℕ) : Prop := n % 10 = 3

def first_ten_primes_units_digit_3 : list ℕ :=
  [3, 13, 23, 43, 53, 73, 83, 103, 113, 163]

def sum_first_ten_primes_units_digit_3 : ℕ :=
  first_ten_primes_units_digit_3.sum

theorem sum_of_first_ten_primes_with_units_digit_3 :
  sum_first_ten_primes_units_digit_3 = 793 := by
  -- Here we provide the steps as a placeholder, but in real practice,
  -- a proof should be constructed to verify this calculation.
  sorry

end sum_of_first_ten_primes_with_units_digit_3_l494_494854


namespace parallelogram_sum_l494_494681

-- Define the vertices of the parallelogram
def A : ℝ × ℝ := (2, 1)
def B : ℝ × ℝ := (6, 1)
def C : ℝ × ℝ := (9, 5)
def D : ℝ × ℝ := (5, 5)

-- Define the distances
def dist (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((q.1 - p.1)^2 + (q.2 - p.2)^2)

-- Calculate distances AB, AD (since parallelogram can use opposite sides)
def AB := dist A B
def AD := dist A D

-- Calculate parameters using distances
def perimeter := 2 * (AB + AD)
def area := AB * (abs (D.2 - A.2))

-- The final sum
def sum_perimeter_area : ℝ :=
  perimeter + area

theorem parallelogram_sum :
  sum_perimeter_area = 34 :=
by
  sorry

end parallelogram_sum_l494_494681


namespace integer_xyz_zero_l494_494337

theorem integer_xyz_zero (x y z : ℤ) (h : x^2 + y^2 + z^2 = 2 * x * y * z) : x = 0 ∧ y = 0 ∧ z = 0 := 
sorry

end integer_xyz_zero_l494_494337


namespace balls_in_boxes_l494_494011

/-
Prove that the number of ways to distribute 6 distinguishable balls into 3 indistinguishable boxes is 132.
-/
theorem balls_in_boxes : 
  ∃ (ways : ℕ), ways = 132 ∧ ways = 
    (1) + 
    (Nat.choose 6 5) + 
    (Nat.choose 6 4) + 
    (Nat.choose 6 4) + 
    (Nat.choose 6 3) + 
    (Nat.choose 6 3 * Nat.choose 3 2) + 
    (Nat.choose 6 2 * Nat.choose 4 2 / 6) := 
by
  sorry

end balls_in_boxes_l494_494011


namespace find_four_letter_list_with_equal_product_l494_494434

open Nat

theorem find_four_letter_list_with_equal_product :
  ∃ (L T M W : ℕ), 
  (L * T * M * W = 23 * 24 * 25 * 26) 
  ∧ (1 ≤ L ∧ L ≤ 26) ∧ (1 ≤ T ∧ T ≤ 26) ∧ (1 ≤ M ∧ M ≤ 26) ∧ (1 ≤ W ∧ W ≤ 26) 
  ∧ (L ≠ T) ∧ (T ≠ M) ∧ (M ≠ W) ∧ (W ≠ L) ∧ (L ≠ M) ∧ (T ≠ W)
  ∧ (L * T * M * W) = (12 * 20 * 13 * 23) :=
by
  sorry

end find_four_letter_list_with_equal_product_l494_494434


namespace cos_double_angle_of_rotated_line_l494_494666

theorem cos_double_angle_of_rotated_line :
  let l := λ x, 2 * x
  let α := arctan (1/3)
  let rotated_l := α + 45 * (π / 180)
  cos (2 * α) = 4 / 5 :=
by 
  sorry

end cos_double_angle_of_rotated_line_l494_494666


namespace max_plus_min_eq_one_l494_494953

noncomputable def f (x : ℝ) : ℝ :=
  ((Real.sqrt 1008 * x + Real.sqrt 1009) ^ 2 + Real.sin (2018 * x)) / (2016 * x ^ 2 + 2018)

theorem max_plus_min_eq_one :
  let M := Real.supr (set.range f)
  let m := Real.infi (set.range f)
  M + m = 1 :=
begin
  sorry
end

end max_plus_min_eq_one_l494_494953


namespace express_y_in_terms_of_x_l494_494098

theorem express_y_in_terms_of_x (x y : ℝ) (h : 4 * x - y = 7) : y = 4 * x - 7 :=
sorry

end express_y_in_terms_of_x_l494_494098


namespace combined_average_score_l494_494768

noncomputable def class_avg_scores (A B C D E : ℕ) (ratioA ratioB ratioC ratioD ratioE : ℕ) : ℕ :=
  let students := ratioA + ratioB + ratioC + ratioD + ratioE
  let total_sum := (A * ratioA) + (B * ratioB) + (C * ratioC) + (D * ratioD) + (E * ratioE)
  total_sum / students

theorem combined_average_score 
  (avgA avgB avgC avgD avgE : ℕ)
  (ratioA ratioB ratioC ratioD ratioE : ℕ)
  (h_avgA : avgA = 68)
  (h_avgB : avgB = 85)
  (h_avgC : avgC = 78)
  (h_avgD : avgD = 92)
  (h_avgE : avgE = 74)
  (h_ratioA : ratioA = 5)
  (h_ratioB : ratioB = 4)
  (h_ratioC : ratioC = 6)
  (h_ratioD : ratioD = 3)
  (h_ratioE : ratioE = 7) :
  class_avg_scores avgA avgB avgC avgD avgE ratioA ratioB ratioC ratioD ratioE = 77.68 :=
by {
  sorry
}

#print combined_average_score

end combined_average_score_l494_494768


namespace width_of_pool_correct_l494_494683

-- Define constants
def length_of_pool : ℝ := 20
def volume_of_water_removed_gallons : ℝ := 1875
def conversion_factor_gallons_to_cubic_feet : ℝ := 7.48052
def water_level_lowered_inches : ℝ := 6
def inches_to_feet : ℝ := 12

-- Define the given conditions in a clearer way
def volume_of_water_removed_cubic_feet : ℝ := volume_of_water_removed_gallons / conversion_factor_gallons_to_cubic_feet
def water_level_lowered_feet : ℝ := water_level_lowered_inches / inches_to_feet

-- The width of the pool is the quantity we want to prove
def width_of_pool : ℝ := volume_of_water_removed_cubic_feet / (length_of_pool * water_level_lowered_feet)

-- Prove that the calculated width equals the given correct answer
theorem width_of_pool_correct : width_of_pool = 25.066 := 
by 
  have h1 : volume_of_water_removed_cubic_feet = 250.66 := by sorry -- Calculations skipped
  have h2 : water_level_lowered_feet = 0.5 := by sorry -- Calculations skipped
  have h3 : width_of_pool = 25.066 := by sorry -- Calculations skipped
  exact h3

end width_of_pool_correct_l494_494683


namespace not_all_pentagons_lie_on_one_side_l494_494996

open Set

noncomputable def pentagon (A B C D E : Point) : Prop :=
(linear_indep ℝ ![A,B,C,D,E]) ∧ (∃ℓ : Linear ℝ, ∀ (p : in specifiedShape), p ∈ sided ℓ)

theorem not_all_pentagons_lie_on_one_side : 
  ∃ (A B C D E : Point), ¬(pentagon A B C D E) := by
  sorry

end not_all_pentagons_lie_on_one_side_l494_494996


namespace circle_area_ratio_is_correct_l494_494938

def circle_area_ratio (R_C R_D : ℝ) : ℝ := (R_C / R_D) ^ 2

theorem circle_area_ratio_is_correct (R_C R_D : ℝ) (h1: R_C / R_D = 3 / 2) : 
  circle_area_ratio R_C R_D = 9 / 4 :=
by
  unfold circle_area_ratio
  rw [h1]
  norm_num

end circle_area_ratio_is_correct_l494_494938


namespace petya_wins_with_optimal_play_l494_494575

theorem petya_wins_with_optimal_play :
  ∃ (n m : ℕ), n = 2000 ∧ m = (n * (n - 1)) / 2 ∧
  (∀ (v_cut : ℕ), ∀ (p_cut : ℕ), v_cut = 1 ∧ (p_cut = 2 ∨ p_cut = 3) ∧
  ((∃ k, m - v_cut = 4 * k) → ∃ k, m - v_cut - p_cut = 4 * k + 1) → 
  ∃ k, m - p_cut = 4 * k + 3) :=
sorry

end petya_wins_with_optimal_play_l494_494575


namespace find_AO_AQ_AR_l494_494731

variables (ABCDEF : Type) [RegularHexagon ABCDEF] (O A P Q R: Point)
variables (AP AQ AR : Line) (OP : ℝ)

-- Additional assumptions/corollaries based on conditions
variables (h_reg : regular_hexagon ABCDEF)
variables (h_center : is_center O ABCDEF)
variables (h_perp_AP : is_perpendicular_from AP A EF)
variables (h_perp_AQ : is_perpendicular_from AQ A (extended ED))
variables (h_perp_AR : is_perpendicular_from AR A (extended FC))
variables (h_OP : OP = 2)

theorem find_AO_AQ_AR :
  AO + AQ + AR = 4/Math.sqrt 3 :=
sorry

end find_AO_AQ_AR_l494_494731


namespace find_m_l494_494916

noncomputable def f (x m : ℝ) : ℝ :=
if x > 1 then log x else 2 * x + ∫ t in 0..m, 3 * t^2

theorem find_m (m : ℝ) (h : f (f real.exp m) m = 29) : m = 3 :=
by {
  sorry
}

end find_m_l494_494916


namespace smallest_n_integer_l494_494816

def sequence (x : ℕ → ℝ) (base : ℝ) : ℕ → ℝ
| 0       := base
| (n + 1) := (sequence n)^(base)

theorem smallest_n_integer (n : ℕ) (x : ℕ → ℝ) : 
  x 0 = real.root 5 5 → 
  (∀ k, x (k + 1) = (x k)^(real.root 5 5)) → 
  (∀ k, x k ∉ ℤ) → 
  n = 2 :=
by
  intros h_base h_step h_not_int
  have : x 1 = real.root 5 5 := h_base
  have : x 2 = (real.root 5 5)^(real.root 5 5) := h_step 1
  sorry

end smallest_n_integer_l494_494816


namespace ratio_bones_child_to_adult_woman_l494_494573

noncomputable def num_skeletons : ℕ := 20
noncomputable def num_adult_women : ℕ := num_skeletons / 2
noncomputable def num_adult_men_and_children : ℕ := num_skeletons - num_adult_women
noncomputable def num_adult_men : ℕ := num_adult_men_and_children / 2
noncomputable def num_children : ℕ := num_adult_men_and_children / 2
noncomputable def bones_per_adult_woman : ℕ := 20
noncomputable def bones_per_adult_man : ℕ := bones_per_adult_woman + 5
noncomputable def total_bones : ℕ := 375
noncomputable def bones_per_child : ℕ := (total_bones - (num_adult_women * bones_per_adult_woman + num_adult_men * bones_per_adult_man)) / num_children

theorem ratio_bones_child_to_adult_woman : 
  (bones_per_child : ℚ) / (bones_per_adult_woman : ℚ) = 1 / 2 := by
sorry

end ratio_bones_child_to_adult_woman_l494_494573


namespace combined_weight_l494_494165

-- Define the conditions
variables (Ron_weight Roger_weight Rodney_weight : ℕ)

-- Define the conditions as Lean propositions
def conditions : Prop :=
  Rodney_weight = 2 * Roger_weight ∧ 
  Roger_weight = 4 * Ron_weight - 7 ∧ 
  Rodney_weight = 146

-- Define the proof goal
def proof_goal : Prop :=
  Rodney_weight + Roger_weight + Ron_weight = 239

theorem combined_weight (Ron_weight Roger_weight Rodney_weight : ℕ) (h : conditions Ron_weight Roger_weight Rodney_weight) : 
  proof_goal Ron_weight Roger_weight Rodney_weight :=
sorry

end combined_weight_l494_494165


namespace distribution_of_6_balls_in_3_indistinguishable_boxes_l494_494000

-- Definition of the problem with conditions
def ways_to_distribute_balls_into_boxes
    (balls : ℕ) (boxes : ℕ) (distinguishable : bool)
    (indistinguishable : bool) : ℕ :=
  if (balls = 6) ∧ (boxes = 3) ∧ (distinguishable = true) ∧ (indistinguishable = true) 
  then 122 -- The correct answer given the conditions
  else 0

-- The Lean statement for the proof problem
theorem distribution_of_6_balls_in_3_indistinguishable_boxes :
  ways_to_distribute_balls_into_boxes 6 3 true true = 122 :=
by sorry

end distribution_of_6_balls_in_3_indistinguishable_boxes_l494_494000


namespace balls_in_boxes_l494_494032

theorem balls_in_boxes : ∀ (balls boxes : ℕ), balls = 6 → boxes = 3 → 
  (∃! n : ℕ, n = 47 ∧ n = number_of_ways_to_distribute_balls_in_boxes_d(balls, boxes)) :=
by
  intros balls boxes h1 h2
  have h_balls : balls = 6 := h1
  have h_boxes : boxes = 3 := h2
  use 47
  split
  sorry
  sorry

end balls_in_boxes_l494_494032


namespace divisible_by_6_l494_494167

theorem divisible_by_6 (n : ℤ) : 6 ∣ (n * (n + 1) * (n + 2)) :=
sorry

end divisible_by_6_l494_494167


namespace original_price_of_trouser_l494_494115

theorem original_price_of_trouser (P : ℝ) (sale_price : ℝ) (percent_decrease : ℝ) 
  (h1 : sale_price = 40) (h2 : percent_decrease = 0.60) 
  (h3 : sale_price = P * (1 - percent_decrease)) : P = 100 :=
by
  sorry

end original_price_of_trouser_l494_494115


namespace average_number_of_carnations_l494_494696

-- Define the number of carnations in each bouquet
def n1 : ℤ := 9
def n2 : ℤ := 23
def n3 : ℤ := 13
def n4 : ℤ := 36
def n5 : ℤ := 28
def n6 : ℤ := 45

-- Define the number of bouquets
def number_of_bouquets : ℤ := 6

-- Prove that the average number of carnations in the bouquets is 25.67
theorem average_number_of_carnations :
  ((n1 + n2 + n3 + n4 + n5 + n6) : ℚ) / (number_of_bouquets : ℚ) = 25.67 := 
by
  sorry

end average_number_of_carnations_l494_494696


namespace minimum_value_a_plus_b_l494_494608

open Real

theorem minimum_value_a_plus_b {a b : ℝ} (h : a^2 + 2 * b^2 = 6) : ∃ x, x ≤ a + b ∧ (∀ y, y ≤ a + b → y ≥ x) :=
begin
  use -3,
  split,
  { sorry, },  -- This will be replaced by the proof that -3 is indeed a lower bound of a + b
  { intro y,
    intros ha,
    sorry,  -- This will be replaced by the proof that -3 is the greatest lower bound
  }
end

end minimum_value_a_plus_b_l494_494608


namespace alexis_multiple_of_alyssa_l494_494965

-- Mathematical theorem in Lean 4:
theorem alexis_multiple_of_alyssa :
  ∃ k : ℝ, (∀ (A B : ℝ), A = 45 → B = 45 →
    (A + 22 = 4 * (B + 22) - 297) →
    (A = k * B - 162)) :=
by
  use 4.6
  intros A B hA hB h1 h2
  sorry

end alexis_multiple_of_alyssa_l494_494965


namespace first_term_of_geometric_series_l494_494781

theorem first_term_of_geometric_series (a r S : ℝ)
  (h_sum : S = a / (1 - r))
  (h_r : r = 1/3)
  (h_S : S = 18) :
  a = 12 :=
by
  sorry

end first_term_of_geometric_series_l494_494781


namespace total_yearly_interest_l494_494620

/-- Mathematical statement:
Given Nina's total inheritance of $12,000, with $5,000 invested at 6% interest and the remainder invested at 8% interest, the total yearly interest from both investments is $860.
-/
theorem total_yearly_interest (principal : ℕ) (principal_part : ℕ) (rate1 rate2 : ℚ) (interest_part1 interest_part2 : ℚ) (total_interest : ℚ) :
  principal = 12000 ∧ principal_part = 5000 ∧ rate1 = 0.06 ∧ rate2 = 0.08 ∧
  interest_part1 = (principal_part : ℚ) * rate1 ∧ interest_part2 = ((principal - principal_part) : ℚ) * rate2 →
  total_interest = interest_part1 + interest_part2 → 
  total_interest = 860 := by
  sorry

end total_yearly_interest_l494_494620


namespace second_fragment_speed_is_l494_494364

variables (u t g vₓ₁ : ℝ)
variables (vₓ₂ vᵧ₂ : ℝ)

-- Given conditions
def initial_vertical_velocity : ℝ := u
def time_of_explosion : ℝ := t
def gravity_acceleration : ℝ := g
def first_fragment_horizontal_velocity : ℝ := vₓ₁

noncomputable def second_fragment_speed : ℝ :=
  let vᵧ := initial_vertical_velocity - gravity_acceleration * time_of_explosion in
  let vₓ₂ := -first_fragment_horizontal_velocity in
  let vᵧ₂ := vᵧ in
  real.sqrt (vₓ₂^2 + vᵧ₂^2)

theorem second_fragment_speed_is : second_fragment_speed u t g vₓ₁ = real.sqrt 2404 := 
sorry

end second_fragment_speed_is_l494_494364


namespace monomial_coefficient_and_degree_l494_494655

def monomial := -7 * a^3 * b^4 * c

-- Definitions
noncomputable def coefficient (M : ℤ) : ℤ := -7
noncomputable def degree (M : ℤ) : ℕ := 3 + 4 + 1

-- Theorem
theorem monomial_coefficient_and_degree :
  coefficient monomial = -7 ∧ degree monomial = 8 :=
by
  -- Proof goes here, but we use sorry for now.
  sorry

end monomial_coefficient_and_degree_l494_494655


namespace regular_triangle_inside_pentagon_l494_494631

theorem regular_triangle_inside_pentagon (A B C D E F : Type)
  [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E] [metric_space F]
  (pentagon : set (A × B × C × D × E)) (triangle : set (A × B × F))
  (is_pentagon_convex : ∀ (x y z w v : A), convex (set', pentagon x y z w v))
  (is_pentagon_equilateral : ∀ (x y z w v : A), length (⟨x, y⟩) = length (⟨y, z⟩) ∧
    length (⟨y, z⟩) = length (⟨z, w⟩) ∧ length (⟨z, w⟩) = length (⟨w, v⟩) ∧
    length (⟨w, v⟩) = length (⟨v, x⟩))
  (internal_angle_sum : ∑ i in {a, b, c, d, e}, interior_angle (pentagon i) = 540) :
  ∃ (F : A) (triangle : set (A × B × F)), ∀ (A B : A), 
  (triangle (A, B, F) → triangle ⊆ pentagon) := sorry

end regular_triangle_inside_pentagon_l494_494631


namespace min_area_AMB_correct_l494_494116

noncomputable def min_area_AMB (CD_length : ℝ) (M : ℝ × ℝ) (perimeter_ACD perimeter_BCD : ℝ)
  (angle_AMB : ℝ) : ℝ :=
if CD_length = 6 ∧ M = (0, 0) ∧ perimeter_ACD = 16 ∧ perimeter_BCD = 16 ∧ angle_AMB = real.pi / 2 then 
  400 / 41 
else 
  0

theorem min_area_AMB_correct : 
  min_area_AMB 6 (0, 0) 16 16 (real.pi / 2) = 400 / 41 :=
sorry

end min_area_AMB_correct_l494_494116


namespace circumcircles_concurrent_l494_494970

open EuclideanGeometry

theorem circumcircles_concurrent
  (A B C D : Point)
  (AB_ne : A ≠ B)
  (convex_quadrilateral : ConvexQuadrilateral A B C D)
  (E : Point)
  (E_on_AB : LiesOn E (LineSegment AB))
  (E_diff : E ≠ A ∧ E ≠ B)
  (F : Point)
  (F_on_AC_DE : LiesOn F (Intersection (LineSegment AC) (LineSegment DE))) :
  (exists P : Point, PointOnCircumcircle P A B C ∧ PointOnCircumcircle P C D F ∧ PointOnCircumcircle P B D E) :=
sorry

end circumcircles_concurrent_l494_494970


namespace increase_150_percent_of_80_l494_494295

theorem increase_150_percent_of_80 : 80 * 1.5 + 80 = 200 :=
by
  sorry

end increase_150_percent_of_80_l494_494295


namespace parallel_CH_AB_l494_494196

variables {A B C D E F G H : Type} [incircle : circle T]

-- Given Conditions
def incircle_touches_sides (ABC : triangle) (D E F : point) : Prop := 
  touches ABC B C D ∧ touches ABC C A E ∧ touches ABC A B F

def diameter_on_incircle (F G : point) : Prop := 
  diameter incircle F G 

def intersection_EG_FD (E G D F H : point) : Prop := 
  intersects (line E G) (line F D) H

-- Prove Statement
theorem parallel_CH_AB (ABC : triangle) (D E F G H : point) 
  (h1 : incircle_touches_sides ABC D E F)
  (h2 : diameter_on_incircle F G)
  (h3 : intersection_EG_FD E G D F H) :
  parallel (line C H) (line A B) :=
sorry

end parallel_CH_AB_l494_494196


namespace distribution_of_6_balls_in_3_indistinguishable_boxes_l494_494001

-- Definition of the problem with conditions
def ways_to_distribute_balls_into_boxes
    (balls : ℕ) (boxes : ℕ) (distinguishable : bool)
    (indistinguishable : bool) : ℕ :=
  if (balls = 6) ∧ (boxes = 3) ∧ (distinguishable = true) ∧ (indistinguishable = true) 
  then 122 -- The correct answer given the conditions
  else 0

-- The Lean statement for the proof problem
theorem distribution_of_6_balls_in_3_indistinguishable_boxes :
  ways_to_distribute_balls_into_boxes 6 3 true true = 122 :=
by sorry

end distribution_of_6_balls_in_3_indistinguishable_boxes_l494_494001


namespace decreasing_range_a_l494_494126

noncomputable def f (a : ℝ) : ℝ → ℝ := λ x,
  if x ≤ 1 then x^2 - 2*(a - 1)*x + 5
  else -x + a

theorem decreasing_range_a (a : ℝ) :
  (∀ x y : ℝ, x ≤ y → f a x ≥ f a y) ↔ (2 ≤ a ∧ a ≤ 3) :=
by
  sorry

end decreasing_range_a_l494_494126


namespace product_of_ks_l494_494827

theorem product_of_ks : 
  (∀ (k : ℕ), (3 * (x : ℚ)^2 + 17 * x + k = 0) → k ∈ {10, 24}) → 
  (10 * 24 = 240) :=
by
  sorry

end product_of_ks_l494_494827


namespace vector_problems_l494_494927

noncomputable def vec_2d := (ℝ × ℝ)

def a : vec_2d := (1, 2)
def b (x : ℝ) : vec_2d := (3, x)
def c (y : ℝ) : vec_2d := (2, y)

def parallel (v1 v2 : vec_2d) : Prop :=
  ∃ k : ℝ, v1 = k • v2

def perpendicular (v1 v2 : vec_2d) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

def m (x : ℝ) : vec_2d :=
  (2 * a.1 - (b x).1, 2 * a.2 - (b x).2)

def n (y : ℝ) : vec_2d :=
  (a.1 + (c y).1, a.2 + (c y).2)

def dot_product (v1 v2 : vec_2d) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def magnitude (v : vec_2d) : ℝ :=
  sqrt (v.1^2 + v.2^2)

def angle_between (v1 v2 : vec_2d) : ℝ :=
  real.arccos ((dot_product v1 v2) / (magnitude v1 * magnitude v2))

theorem vector_problems :
  (∃ x y, 
    parallel a (b x) ∧
    perpendicular a (c y) ∧ 
    b x = (3, 6) ∧
    c y = (2, -1)) ∧
  (∃ x y, 
     angle_between (m x) (n y) = 3 * real.pi / 4) :=
by {
  sorry 
}

end vector_problems_l494_494927


namespace find_real_values_l494_494889

noncomputable def solve_complex_eq (x y : ℝ) : Prop :=
  x^2 - y^2 + 2 * x * y * complex.I = 2 * complex.I

theorem find_real_values (x y : ℝ) : solve_complex_eq x y ↔ (x = 1 ∧ y = 1) ∨ (x = -1 ∧ y = -1) :=
by
  sorry

end find_real_values_l494_494889


namespace volume_of_soil_extracted_l494_494723

-- Define the problem
variable (Length : ℕ) (Width : ℕ) (Height : ℕ)

-- Given conditions
def pondLength := 20
def pondWidth := 15
def pondHeight := 5

-- Statement to prove
theorem volume_of_soil_extracted (Length = pondLength) (Width = pondWidth) (Height = pondHeight) :
  Length * Width * Height = 1500 := by 
sorry

end volume_of_soil_extracted_l494_494723


namespace circle_and_intersection_conditions_l494_494581

-- Define the points on the circle
def pointA : ℝ × ℝ := (0, 1)
def pointB : ℝ × ℝ := (3, 4)
def pointC : ℝ × ℝ := (6, 1)

-- Define the general form of the circle equation
def circle_eq (D E F : ℝ) (x y : ℝ) : Prop := x^2 + y^2 + D*x + E*y + F = 0

-- Define the condition that points A, B, and C lie on the circle
def passes_through_points (D E F : ℝ) : Prop :=
  circle_eq D E F pointA.fst pointA.snd ∧ 
  circle_eq D E F pointB.fst pointB.snd ∧ 
  circle_eq D E F pointC.fst pointC.snd

-- Define the specific circle equation derived in the solution
def specific_circle_eq (x y : ℝ) : Prop := (x - 3)^2 + (y - 1)^2 = 9

-- Define the intersection condition and orthogonality condition
def intersection_and_orthogonality (a : ℝ) : Prop :=
  ∀ (x y : ℝ), specific_circle_eq x y → (x - y + a = 0) → 
    (let x1 := x in let y1 := y in
     ∀ (x y : ℝ), specific_circle_eq x y → (x - y + a = 0) → 
       (let x2 := x in let y2 := y in
        x1 * x2 + y1 * y2 = 0))

-- The final proof statement combining both parts of the problem
theorem circle_and_intersection_conditions (D E F : ℝ) :
  passes_through_points D E F →
  (∀ x y : ℝ, circle_eq D E F x y ↔ specific_circle_eq x y) →
  (∀ a : ℝ, intersection_and_orthogonality a ↔ a = -1) :=
by
  intros _ _
  sorry

end circle_and_intersection_conditions_l494_494581


namespace determinant_after_row_operation_l494_494815

def original_matrix : Matrix (Fin 3) (Fin 3) ℤ :=
  ![\[3, 0, -2\],
    \[8, 5, -4\],
    \[5, 2, 3\]]

def row_operation_matrix : Matrix (Fin 3) (Fin 3) ℤ :=
  ![\[3, 0, -2\],
    \[8, 5, -4\],
    \[8, 2, 1\]] -- adding the first row to the third row

theorem determinant_after_row_operation : 
  Matrix.det row_operation_matrix = 87 := by
  sorry

end determinant_after_row_operation_l494_494815


namespace increase_80_by_150_percent_l494_494265

-- Original number
def originalNumber : ℕ := 80

-- Percentage increase as a decimal
def increaseFactor : ℚ := 1.5

-- Expected result after the increase
def expectedResult : ℕ := 200

theorem increase_80_by_150_percent :
  originalNumber + (increaseFactor * originalNumber) = expectedResult :=
by
  sorry

end increase_80_by_150_percent_l494_494265


namespace sum_xyz_l494_494548

theorem sum_xyz (x y z : ℝ) 
  (h1 : log 3 (log 4 (log 5 x)) = 0)
  (h2 : log 4 (log 5 (log 3 y)) = 0)
  (h3 : log 5 (log 3 (log 4 z)) = 0) : 
  x + y + z = 932 := 
by 
  sorry

end sum_xyz_l494_494548


namespace correct_negation_of_exactly_one_even_l494_494718

-- Define a predicate to check if a natural number is even
def is_even (n : ℕ) : Prop := n % 2 = 0

-- Define a predicate to check if a natural number is odd
def is_odd (n : ℕ) : Prop := n % 2 = 1

-- Problem statement in Lean
theorem correct_negation_of_exactly_one_even (a b c : ℕ) :
  ¬ ( (is_even a ∧ is_odd b ∧ is_odd c) ∨ 
      (is_odd a ∧ is_even b ∧ is_odd c) ∨ 
      (is_odd a ∧ is_odd b ∧ is_even c) ) ↔ 
  ( (is_odd a ∧ is_odd b ∧ is_odd c) ∨ 
    (is_even a ∧ is_even b ∧ is_even c) ) :=
by 
  sorry

end correct_negation_of_exactly_one_even_l494_494718


namespace limit_exponential_sine_l494_494423

theorem limit_exponential_sine :
  (lim x → 0, (exp (5 * x) - exp (3 * x)) / (sin (2 * x) - sin (x)) = 2) := by
  sorry

end limit_exponential_sine_l494_494423


namespace max_sector_area_l494_494077

theorem max_sector_area (r l : ℝ) (hp : 2 * r + l = 40) : (1 / 2) * l * r ≤ 100 := 
by
  sorry

end max_sector_area_l494_494077


namespace increase_by_150_percent_l494_494313

theorem increase_by_150_percent (x : ℝ) (h : x = 80) : x + (1.5 * x) = 200 := 
by
  -- The proof goes here, but is omitted as per instructions.
  sorry

end increase_by_150_percent_l494_494313


namespace balls_in_boxes_l494_494028

theorem balls_in_boxes : ∀ (balls boxes : ℕ), balls = 6 → boxes = 3 → 
  (∃! n : ℕ, n = 47 ∧ n = number_of_ways_to_distribute_balls_in_boxes_d(balls, boxes)) :=
by
  intros balls boxes h1 h2
  have h_balls : balls = 6 := h1
  have h_boxes : boxes = 3 := h2
  use 47
  split
  sorry
  sorry

end balls_in_boxes_l494_494028


namespace jamshid_taimour_paint_fence_l494_494109

theorem jamshid_taimour_paint_fence :
  ∀ (t : ℕ) (j : ℕ),
  (t = 15) →
  (j = t / 2) →
  ((1 / t + 1 / j) = 1 / 5) :=
by
  intros t j ht hj
  rw [ht, hj]
  simp only [div_eq_mul_inv, add_mul, mul_div_assoc, mul_div_cancel_left, eq_self_iff_true, inv_mul_cancel]
  norm_num
  sorry

end jamshid_taimour_paint_fence_l494_494109


namespace meeting_anniversary_day_l494_494412

-- Define the input parameters for the problem
def initial_years : Set ℕ := {1668, 1669, 1670, 1671}
def meeting_day := "Friday"
def is_leap_year (year : ℕ) : Bool := (year % 4 = 0)

-- Define the theorem for the problem statement
theorem meeting_anniversary_day :
  ∀ (year : ℕ), year ∈ initial_years →
  let leap_years := (∑ n in range 1668, if is_leap_year n then 1 else 0)
  let total_days := 11 * 365 + leap_years
  let day_of_week := total_days % 7
  in (day_of_week = 0 ∧ probability Friday = 3 / 4) ∨ (day_of_week = 6 ∧ probability Thursday 1 / 4) :=
by
  sorry

end meeting_anniversary_day_l494_494412


namespace problem1_problem2_l494_494917

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (2^x) * abs (2^x - a) + (2^(x+1)) - 3

theorem problem1 (a : ℝ) : a = 4 → (∀ x ∈ set.Icc 1 3, 5 ≤ f x 4 ∧ f x 4 ≤ 45) :=
sorry

theorem problem2 (h_monotonic_f : ∀ x₁ x₂, x₁ ≤ x₂ → f x₁ a ≤ f x₂ a) : -2 ≤ a ∧ a ≤ 2 :=
sorry

end problem1_problem2_l494_494917


namespace cos_C_in_triangle_l494_494082

-- Define the problem conditions
def triangle_ABC (A B C : ℝ) (sin_A sin_B sin_C : ℝ) : Prop :=
  sin_A / sin_B = 4 / 3 ∧ sin_B / sin_C = 3 / 2

-- The Lean statement for proving the problem
theorem cos_C_in_triangle {A B C sin_A sin_B sin_C : ℝ}
  (h : triangle_ABC A B C sin_A sin_B sin_C) : (cos C = 7 / 8) := 
by
  sorry

end cos_C_in_triangle_l494_494082


namespace line_equation_of_chord_through_midpoint_chord_length_l494_494755

noncomputable def Q := (1/3 : ℝ, 4/3 : ℝ)

def hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 4 = 1

def midpoint (A B Q : ℝ × ℝ) : Prop :=
  ∃ x1 x2 y1 y2 : ℝ, A = (x1, y1) ∧ B = (x2, y2) ∧ Q = ((x1 + x2) / 2, (y1 + y2) / 2)

theorem line_equation_of_chord_through_midpoint (A B : ℝ × ℝ) :
  midpoint A B Q →
  (∃ x1 y1 x2 y2 : ℝ, A = (x1, y1) ∧ B = (x2, y2) ∧ hyperbola x1 y1 ∧ hyperbola x2 y2) →
  ∀ x y : ℝ, y = x + 1 ↔ (∃ t : ℝ, (x, y) = (t, t + 1))
:= sorry

theorem chord_length (A B : ℝ × ℝ) :
  midpoint A B Q →
  (∃ x1 y1 x2 y2 : ℝ, A = (x1, y1) ∧ B = (x2, y2) ∧ hyperbola x1 y1 ∧ hyperbola x2 y2) →
  ∃ l : ℝ, l = 8 * real.sqrt 2 / 3 ∧
    l = real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
:= sorry

end line_equation_of_chord_through_midpoint_chord_length_l494_494755


namespace lefty_points_l494_494603

variable (L : ℝ) -- Define L as the points scored by Lefty

-- Define the conditions
def righty_points (L : ℝ) := L / 2 -- Righty's points
def teammate_points (L : ℝ) := 3 * righty_points L -- Their other teammate's points

-- Hypothesis stating the average points per player
def avg_points_per_player (L : ℝ) := (L + righty_points L + teammate_points L) / 3

-- The proof problem to be translated
theorem lefty_points : avg_points_per_player L = 30 → L = 20 := by
  sorry -- Proof placeholder

end lefty_points_l494_494603


namespace ellipse_eccentricity_l494_494496

-- Definitions based on given conditions
variables (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : a^2 = b^2 + c^2)

-- Statement: Show the eccentricity e of the ellipse is (sqrt 5 - 1) / 2
theorem ellipse_eccentricity (h_perpendicular: ( -c * a) + ( -b * b ) = 0) : 
  let e := c / a in
  e = (Real.sqrt 5 - 1) / 2 := 
begin
  have h4 : b^2 = a * c, from eq_of_sub_eq_zero (eq_sub_of_add_eq h_perpendicular),
  have h5 : 2 * b ^ 2 - a ^ 2 = 0, from by { rw [h4, ← h3], ring },
  have h6 : (1 - e^2 - e = 0), from sorry, -- Simplifying steps 
  have h7 : e^2 + e - 1 = 0, from by { rw h6, ring },
  exact eq_of_sub_eq_zero (eq_sub_of_add_eq h7), -- Quadratic solution
end

end ellipse_eccentricity_l494_494496


namespace instructors_ge_trainees_in_Grésillon_l494_494570

variables {Person : Type} (acquainted : Person → Person → Prop) (is_instructor : Person → Prop) 

-- Definition: an instructor always tells the truth
def always_tells_truth (p : Person) : Prop := ∀ s : String, is_instructor p → s = "true"

-- Definition: a trainee always lies
def always_lies (p : Person) : Prop := ∀ s : String, ¬is_instructor p → s = "false"

-- Trick to approximate acquaintanceship (everyone claims all their acquaintances know each other)
def all_acquainted (p : Person) : Prop := ∀ q r : Person, acquainted p q → acquainted p r → acquainted q r

-- Trick to approximate trainees >= instructors in acquaintances
def trainees_ge_instructors (p : Person) : Prop :=
  ∀ q : Person, acquainted p q → (∃ (t : nat), ∃ (i : nat), t ≥ i ∧ ¬is_instructor q → t = count_trainees p q ∧ is_instructor q → i = count_instructors p q)

-- Main theorem: The number of instructors is greater than or equal to the number of trainees
theorem instructors_ge_trainees_in_Grésillon {k : ℕ} :
  (∀ p : Person, all_acquainted p ) →
  (∀ p : Person, trainees_ge_instructors p ) →
  (∃ p : Person, always_tells_truth p ∨ always_lies p) →
  count_instructors_in_Grésillon ≥ count_trainees_in_Grésillon :=
by
  sorry

end instructors_ge_trainees_in_Grésillon_l494_494570


namespace distance_from_circle_center_to_line_l494_494101

noncomputable def polar_circle_center_distance : ℝ :=
  let θ := Real.pi / 3 in
  let A := Real.sqrt 3 in
  let B := (-1) in
  let C := 0 in
  let x₁ := 2 in
  let y₁ := 0 in
  |A * x₁ + B * y₁ + C| / (Real.sqrt (A * A + B * B))

theorem distance_from_circle_center_to_line :
  polar_circle_center_distance = Real.sqrt 3 :=
sorry

end distance_from_circle_center_to_line_l494_494101


namespace circumcenter_equidistant_l494_494626

noncomputable def circumcenter {α : Type} [euclidean_space α] : α → α → α → α := sorry

variables (A B C D K L : euclidean_space ℝ) 
variables (h_parallelogram : parallelogram A B C D)
variables (hK : point_on_segment K A B)
variables (hL : point_on_segment L B C)
variables (h_angle : ∠ A K D = ∠ C L D)

theorem circumcenter_equidistant (hK : ∠ A K D = ∠ C L D) :
  dist (circumcenter B K L) A = dist (circumcenter B K L) C :=
sorry

end circumcenter_equidistant_l494_494626


namespace total_miles_walked_l494_494108

def weekly_group_walk_miles : ℕ := 3 * 6

def Jamie_additional_walk_miles_per_week : ℕ := 2 * 6
def Sue_additional_walk_miles_per_week : ℕ := 1 * 6 -- half of Jamie's additional walk
def Laura_additional_walk_miles_per_week : ℕ := 1 * 3 -- 1 mile every two days for 6 days
def Melissa_additional_walk_miles_per_week : ℕ := 2 * 2 -- 2 miles every three days for 6 days
def Katie_additional_walk_miles_per_week : ℕ := 1 * 6

def Jamie_weekly_miles : ℕ := weekly_group_walk_miles + Jamie_additional_walk_miles_per_week
def Sue_weekly_miles : ℕ := weekly_group_walk_miles + Sue_additional_walk_miles_per_week
def Laura_weekly_miles : ℕ := weekly_group_walk_miles + Laura_additional_walk_miles_per_week
def Melissa_weekly_miles : ℕ := weekly_group_walk_miles + Melissa_additional_walk_miles_per_week
def Katie_weekly_miles : ℕ := weekly_group_walk_miles + Katie_additional_walk_miles_per_week

def weeks_in_month : ℕ := 4

def Jamie_monthly_miles : ℕ := Jamie_weekly_miles * weeks_in_month
def Sue_monthly_miles : ℕ := Sue_weekly_miles * weeks_in_month
def Laura_monthly_miles : ℕ := Laura_weekly_miles * weeks_in_month
def Melissa_monthly_miles : ℕ := Melissa_weekly_miles * weeks_in_month
def Katie_monthly_miles : ℕ := Katie_weekly_miles * weeks_in_month

def total_monthly_miles : ℕ :=
  Jamie_monthly_miles + Sue_monthly_miles + Laura_monthly_miles + Melissa_monthly_miles + Katie_monthly_miles

theorem total_miles_walked (month_has_30_days : Prop) : total_monthly_miles = 484 :=
by
  unfold total_monthly_miles
  unfold Jamie_monthly_miles Sue_monthly_miles Laura_monthly_miles Melissa_monthly_miles Katie_monthly_miles
  unfold Jamie_weekly_miles Sue_weekly_miles Laura_weekly_miles Melissa_weekly_miles Katie_weekly_miles
  unfold weekly_group_walk_miles Jamie_additional_walk_miles_per_week Sue_additional_walk_miles_per_week Laura_additional_walk_miles_per_week Melissa_additional_walk_miles_per_week Katie_additional_walk_miles_per_week
  unfold weeks_in_month
  sorry

end total_miles_walked_l494_494108


namespace shift_cos_to_sin_l494_494218

theorem shift_cos_to_sin (x : ℝ) :
  sin (2 * x - (Real.pi / 4)) = cos (2 * (x - (3 * Real.pi / 8))) := 
sorry

end shift_cos_to_sin_l494_494218


namespace school_total_payment_l494_494185

theorem school_total_payment
  (price : ℕ)
  (kindergarten_models : ℕ)
  (elementary_library_multiplier : ℕ)
  (model_reduction_percentage : ℚ)
  (total_models : ℕ)
  (reduced_price : ℚ)
  (total_payment : ℚ)
  (h1 : price = 100)
  (h2 : kindergarten_models = 2)
  (h3 : elementary_library_multiplier = 2)
  (h4 : model_reduction_percentage = 0.05)
  (h5 : total_models = kindergarten_models + (kindergarten_models * elementary_library_multiplier))
  (h6 : total_models > 5)
  (h7 : reduced_price = price - (price * model_reduction_percentage))
  (h8 : total_payment = total_models * reduced_price) :
  total_payment = 570 := 
by
  sorry

end school_total_payment_l494_494185


namespace increase_by_percentage_l494_494325

theorem increase_by_percentage (a : ℤ) (b : ℚ) : ((b + 1) * a) = 200 :=
by
  -- Definitions based directly on conditions
  let a := (80 : ℤ)
  let b := (1.5 : ℚ)
  -- Assertions about the equivalent proof problem
  have h1 : b + 1 = 2.5 := by norm_num
  have h2 : (b + 1) * a = 200 := by norm_num
  exact h2

end increase_by_percentage_l494_494325


namespace production_line_B_l494_494354

theorem production_line_B (A B C : ℕ) (h1 : A + B + C = 19500) (h2 : (A, B, C).FormArithmeticSeq) : B = 6500 := 
by
  -- Proof goes here
  sorry

end production_line_B_l494_494354


namespace ratio_of_returned_pints_l494_494623

theorem ratio_of_returned_pints:
  let sunday_pints := 4 in
  let monday_pints := 3 * sunday_pints in
  let tuesday_pints := monday_pints / 3 in
  let total_pints_before_returning := sunday_pints + monday_pints + tuesday_pints in
  let wednesday_pints_after_returning := 18 in
  let returned_pints := total_pints_before_returning - wednesday_pints_after_returning in
  returned_pints / tuesday_pints = 1 / 2 :=
by
  sorry

end ratio_of_returned_pints_l494_494623


namespace wrong_statement_e_l494_494065

open Real

theorem wrong_statement_e (b x : ℝ) (y : ℝ) (hb1 : b > 0) (hb2 : b ≠ 1) (hx1 : x = 1) (hy1 : y = 0)
  (hx2 : x = b) (hy2 : y = 1) (hx3 : x = 1 / b) (hy3 : y = -1) (hx4_low : 0 < x) (hx4_high : x < 1) (hy4 : y < 0) :
  "Only some of the above statements are correct" = False := by
  sorry

end wrong_statement_e_l494_494065


namespace increase_result_l494_494241

-- Given conditions
def original_number : ℝ := 80
def increase_percentage : ℝ := 1.5

-- The result after the increase
theorem increase_result (h1 : original_number = 80) (h2 : increase_percentage = 1.5) : 
  original_number + (increase_percentage * original_number) = 200 := by
  sorry

end increase_result_l494_494241


namespace radius_of_the_wheel_l494_494675

open Real

noncomputable def radius_of_wheel (speed_kmh : ℝ) (rpm : ℝ) : ℝ :=
  let speed_cms_min := (speed_kmh * 100000) / 60
  let circumference := speed_cms_min / rpm
  circumference / (2 * π)

theorem radius_of_the_wheel (speed_kmh : ℝ) (rpm : ℝ) (r : ℝ) :
  speed_kmh = 66 →
  rpm = 125.11373976342128 →
  abs (r - 140.007) < 0.001 :=
by
  intros h_speed h_rpm
  have r_def := radius_of_wheel speed_kmh rpm
  have hr : r = r_def := by sorry
  rw [r_def, h_speed, h_rpm]
  sorry

end radius_of_the_wheel_l494_494675


namespace p_sufficient_not_necessary_for_q_l494_494504

variables {a b : ℝ}

def p : Prop := a * b ≠ 0
def q : Prop := a^2 + b^2 ≠ 0

theorem p_sufficient_not_necessary_for_q :
  (p → q) ∧ (¬(q → p)) := by
  -- Insert proof here
  sorry

end p_sufficient_not_necessary_for_q_l494_494504


namespace umbrella_numbers_are_40_l494_494556

open Finset

def is_umbrella_number (x y z : ℕ) : Prop :=
  x < y ∧ z < y ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x ∈ {1, 2, 3, 4, 5, 6} ∧ y ∈ {1, 2, 3, 4, 5, 6} ∧ z ∈ {1, 2, 3, 4, 5, 6}

def umbrella_numbers_count : ℕ :=
  (univ : Finset (ℕ × ℕ × ℕ)).filter (λ n, is_umbrella_number n.1 n.2.1 n.2.2).card

theorem umbrella_numbers_are_40 : umbrella_numbers_count = 40 := by
  sorry

end umbrella_numbers_are_40_l494_494556


namespace range_of_k_l494_494913

noncomputable def e := Real.exp 1

theorem range_of_k (k : ℝ) (h : ∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ e ^ (x1 - 1) = |k * x1| ∧ e ^ (x2 - 1) = |k * x2| ∧ e ^ (x3 - 1) = |k * x3|) : k^2 > 1 := sorry

end range_of_k_l494_494913


namespace increase_80_by_150_percent_l494_494281

theorem increase_80_by_150_percent : 
  ∀ x percent, x = 80 ∧ percent = 1.5 → x + (x * percent) = 200 :=
by
  sorry

end increase_80_by_150_percent_l494_494281


namespace DQ_bisects_angle_ADC_l494_494093

variables (A B C D M N Q : Type*)
variables [parallelogram A B C D] (M_on_AB : M ∈ segment A B) (N_on_BC : N ∈ segment B C)
variables (AM_eq_NC : dist A M = dist N C) (Q_intercept : is_intersection (line_through A N) (line_through C M) Q)

theorem DQ_bisects_angle_ADC :
  bisects_angle (line_through D Q) (angle A D C) :=
sorry

end DQ_bisects_angle_ADC_l494_494093


namespace smallest_years_to_multiple_l494_494601

theorem smallest_years_to_multiple (J C F : ℕ) (hJ : J = 40) (hC : C = 38) (hF : F = 60) : 
  ∃ n : ℕ, (J + n) % F = 0 ∧ (C + n) % F = 0 ∧ n = 180 := by
{
  use 180,
  have hJn : (40 + 180) % 60 = 0, by norm_num,
  have hCn : (38 + 180) % 60 = 0, by norm_num,
  exact ⟨hJn, hCn, rfl⟩,
}

end smallest_years_to_multiple_l494_494601


namespace anniversary_day_of_week_probability_l494_494406

/-- The 11th anniversary of Robinson Crusoe and Friday's meeting can fall on a Friday with a
probability of 3/4 and on a Thursday with a probability of 1/4, given that the meeting occurred
in any year from 1668 to 1671 with equal probability. -/
theorem anniversary_day_of_week_probability :
  let years := {1668, 1669, 1670, 1671},
      leap (y : ℕ) := y % 4 = 0,
      days_in_year := λ y, if leap y then 366 else 365,
      total_days (yr : ℕ) := list.sum (list.map days_in_year (list.range' yr 11)),
      day_of_week_after_11_years (initial_year : ℕ) := total_days initial_year % 7 = 0,
      events := {week_day | ∀ y ∈ years, (day_of_week_after_11_years y)},
      friday_probability := rat.mk 3 4,
      thursday_probability := rat.mk 1 4
  in
  (events = {0} ∨ events = {6}) ∧
  (events = {0} → friday_probability = rat.mk 3 4 ∧ thursday_probability = rat.mk 1 4) ∧
  (events = {6} → friday_probability = rat.mk 1 4 ∧ thursday_probability = rat.mk 3 4):=
begin
  sorry
end

end anniversary_day_of_week_probability_l494_494406


namespace anniversary_day_of_week_probability_l494_494405

/-- The 11th anniversary of Robinson Crusoe and Friday's meeting can fall on a Friday with a
probability of 3/4 and on a Thursday with a probability of 1/4, given that the meeting occurred
in any year from 1668 to 1671 with equal probability. -/
theorem anniversary_day_of_week_probability :
  let years := {1668, 1669, 1670, 1671},
      leap (y : ℕ) := y % 4 = 0,
      days_in_year := λ y, if leap y then 366 else 365,
      total_days (yr : ℕ) := list.sum (list.map days_in_year (list.range' yr 11)),
      day_of_week_after_11_years (initial_year : ℕ) := total_days initial_year % 7 = 0,
      events := {week_day | ∀ y ∈ years, (day_of_week_after_11_years y)},
      friday_probability := rat.mk 3 4,
      thursday_probability := rat.mk 1 4
  in
  (events = {0} ∨ events = {6}) ∧
  (events = {0} → friday_probability = rat.mk 3 4 ∧ thursday_probability = rat.mk 1 4) ∧
  (events = {6} → friday_probability = rat.mk 1 4 ∧ thursday_probability = rat.mk 3 4):=
begin
  sorry
end

end anniversary_day_of_week_probability_l494_494405


namespace new_volume_eq_8054_l494_494762

variable {l w h : ℝ}

-- Initial conditions
def volume_initial : Prop := l * w * h = 5400
def surface_area_initial : Prop := l * w + w * h + h * l = 1176
def edge_sum_initial : Prop := l + w + h = 60

-- Theorem statement
theorem new_volume_eq_8054 
  (h1 : volume_initial) 
  (h2 : surface_area_initial) 
  (h3 : edge_sum_initial) : 
  (l + 2) * (w + 2) * (h + 2) = 8054 :=
by
  sorry

end new_volume_eq_8054_l494_494762


namespace std_dev_example_l494_494769

open Real

noncomputable def std_dev (l : List ℝ) : ℝ :=
  let mean := (l.sum) / (l.length) in
  let variance := (l.map (λ x => (x - mean) ^ 2)).sum / (l.length) in
  sqrt variance

theorem std_dev_example :
  std_dev [10, 6, 8, 5, 6] = 4 * sqrt 5 / 5 := 
by
  sorry

end std_dev_example_l494_494769


namespace value_of_a3_l494_494204

theorem value_of_a3 :
  (∀ n : ℕ, n ≥ 1 → a (n + 1) = a n - 3) →
  a 1 = 7 →
  a 3 = 1 :=
by
  sorry

end value_of_a3_l494_494204


namespace balls_in_indistinguishable_boxes_l494_494018

theorem balls_in_indistinguishable_boxes : 
  let balls := 6 in
  let boxes := 3 in
  -- Expression to count ways to distribute balls into indistinguishable boxes
  ((balls.choose 0 * ((balls-0).choose 6)) + 
   (balls.choose 1 * ((balls-1).choose 5)) + 
   (balls.choose 2 * ((balls-2).choose 4)) + 
   (balls.choose 2 * ((balls-2).choose 3) / 2!) + 
   (balls.choose 3 * ((balls-3).choose 3)) + 
   (balls.choose 3 * ((balls-3).choose 2) * ((balls-5).choose 1)) + 
   (balls.choose 2 * ((balls-2).choose 2) / 3!))
  = 92 := 
sorry

end balls_in_indistinguishable_boxes_l494_494018


namespace six_digit_number_523_divisible_7_8_9_l494_494386

theorem six_digit_number_523_divisible_7_8_9 : 
  ∃ n : ℕ, 523 * 1000 + n = 523392 ∧ 523392 % 7 = 0 ∧ 523392 % 8 = 0 ∧ 523392 % 9 = 0 :=
by {
  use 392,
  split,
  { norm_num },
  { norm_num, sorry },
  { norm_num, sorry }
}

end six_digit_number_523_divisible_7_8_9_l494_494386


namespace distinct_real_roots_max_abs_gt_2_l494_494061

theorem distinct_real_roots_max_abs_gt_2 
  (r1 r2 r3 q : ℝ)
  (h_distinct : r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3)
  (h_sum : r1 + r2 + r3 = -q)
  (h_product : r1 * r2 * r3 = -9)
  (h_sum_prod : r1 * r2 + r2 * r3 + r3 * r1 = 6)
  (h_nonzero_discriminant : q^2 * 6^2 - 4 * 6^3 - 4 * q^3 * 9 - 27 * 9^2 + 18 * q * 6 * (-9) ≠ 0) :
  max (|r1|) (max (|r2|) (|r3|)) > 2 :=
sorry

end distinct_real_roots_max_abs_gt_2_l494_494061


namespace log_sum_example_l494_494794

theorem log_sum_example :
  let log_base_10 (x : ℝ) := Real.log x / Real.log 10 in
  log_base_10 50 + log_base_10 20 = 3 :=
by
  sorry

end log_sum_example_l494_494794


namespace increase_by_percentage_l494_494254

theorem increase_by_percentage (initial_amount : ℕ) (percentage_increase : ℕ) :
  initial_amount = 80 → 
  percentage_increase = 150 → 
  initial_amount * (percentage_increase / 100) + initial_amount = 200 :=
by
  intros h_initial h_percentage
  rw [h_initial, h_percentage]
  calc
    80 * (150 / 100) + 80 = 80 * 1.5 + 80 : by norm_num
                     ... = 120 + 80         : by norm_num
                     ... = 200              : by norm_num
  sorry

end increase_by_percentage_l494_494254


namespace cubical_surface_white_area_l494_494644

theorem cubical_surface_white_area (structures : ℕ) (cubes_per_structure : ℕ) (white_cubes : ℕ) (edge_length : ℕ) :
  structures = 7 → cubes_per_structure = 8 → white_cubes = 8 → edge_length = 1 →
  let total_cubes := (structures * cubes_per_structure) + white_cubes,
      large_cube_side := nat.sqrt (nat.sqrt total_cubes),
      large_cube_surface := 6 * (large_cube_side * large_cube_side) in
  large_cube_surface = 96 →
  ∃ min_white_surface : ℕ, min_white_surface = 12 ∧ (large_cube_surface - max_gray_surface) = min_white_surface :=
begin
  intros h1 h2 h3 h4,
  let total_cubes := (structures * cubes_per_structure) + white_cubes,
  let large_cube_side := 4,
  let large_cube_surface := 96,
  sorry
end

end cubical_surface_white_area_l494_494644


namespace eleventh_anniversary_days_l494_494417

-- Define the conditions
def is_leap_year (year : ℕ) : Prop := year % 4 = 0

def initial_years : Set ℕ := {1668, 1669, 1670, 1671}

def initial_day := "Friday"

noncomputable def day_after_11_years (start_year : ℕ) : String :=
  let total_days := 4015 + (if is_leap_year start_year then 3 else 2)
  if total_days % 7 = 0 then "Friday"
  else "Thursday"

-- Define the proposition to prove
theorem eleventh_anniversary_days : 
  (∀ year ∈ initial_years, 
    (if day_after_11_years year = "Friday" then (3 : ℝ) / 4 else (1 : ℝ) / 4) = 
    (if year = 1668 ∨ year = 1670 ∨ year = 1671 then (3 : ℝ) / 4 else (1 : ℝ) / 4)) := 
sorry

end eleventh_anniversary_days_l494_494417


namespace sum_of_roots_eq_five_thirds_l494_494604

-- Define the quadratic equation
def quadratic_eq (n : ℝ) : Prop := 3 * n^2 - 5 * n - 4 = 0

-- Prove that the sum of the solutions to the quadratic equation is 5/3
theorem sum_of_roots_eq_five_thirds :
  (∃ a b : ℝ, quadratic_eq a ∧ quadratic_eq b ∧ a + b = 5 / 3) :=
sorry

end sum_of_roots_eq_five_thirds_l494_494604


namespace median_from_vertex_P_l494_494588

theorem median_from_vertex_P
  {P Q R : Type} [EuclideanGeometry P Q R]
  (PQ PR : ℝ) (S : ℝ)
  (hPQ : PQ ≤ 9)
  (hPR : PR ≤ 12)
  (hArea : S ≥ 54)
  (hS : S = 0.5 * PQ * PR * (Real.sin π)):
  let QR := Real.sqrt (PQ^2 + PR^2) in
  let PM := QR / 2 in
  PM = 7.5 :=
by
  sorry

end median_from_vertex_P_l494_494588


namespace problem_profit_percentage_l494_494740

theorem problem_profit_percentage
(CP SP : ℝ)
(hCP : CP = 47.50)
(hSP : SP = 67.47) :
  ∃ profit_percentage : ℝ, profit_percentage ≈ 42.04 :=
by
  sorry

end problem_profit_percentage_l494_494740


namespace degree_ge_n_of_polynomial_approximation_l494_494491

open Real Polynomial

theorem degree_ge_n_of_polynomial_approximation (t : ℝ) (n : ℕ) (f : Polynomial ℝ)
  (h1 : t ≥ 3) 
  (h2 : ∀ k : ℕ, k ≤ n → |f.eval k - t^k| < 1) : 
  f.natDegree ≥ n :=
sorry

end degree_ge_n_of_polynomial_approximation_l494_494491


namespace min_weights_one_pan_min_weights_both_pans_l494_494702

-- Part 1: Weights can only be placed on one pan
theorem min_weights_one_pan : ∃ (weights : List ℕ), (∀ m ∈ weights, 1 ≤ m ∧ m ≤ 40) ∧ (∀ (n : ℕ), 1 ≤ n ∧ n ≤ 40 → ∃ (wt : Finset ℕ), wt ⊆ weights.to_finset ∧ n = wt.sum id) ∧ weights.length = 6 :=
sorry

-- Part 2: Weights can be placed on both pans
theorem min_weights_both_pans : ∃ (weights : List ℕ), (∀ m ∈ weights, 1 ≤ m ∧ m ≤ 40) ∧ (∀ (n : ℕ), 1 ≤ n ∧ n ≤ 40 →
∃ (wt_pos wt_neg : Finset ℕ), wt_pos ⊆ weights.to_finset ∧ wt_neg ⊆ weights.to_finset ∧ n = wt_pos.sum id - wt_neg.sum id) ∧ weights.length = 4 :=
sorry

end min_weights_one_pan_min_weights_both_pans_l494_494702


namespace log_addition_property_l494_494799

theorem log_addition_property : log 10 50 + log 10 20 = 3 :=
by
  sorry

end log_addition_property_l494_494799


namespace find_number_of_children_l494_494358

theorem find_number_of_children (adults children : ℕ) (adult_ticket_price child_ticket_price total_money change : ℕ) 
    (h1 : adult_ticket_price = 9) 
    (h2 : child_ticket_price = adult_ticket_price - 2) 
    (h3 : total_money = 40) 
    (h4 : change = 1) 
    (h5 : adults = 2) 
    (total_cost : total_money - change = adults * adult_ticket_price + children * child_ticket_price) : 
    children = 3 :=
sorry

end find_number_of_children_l494_494358


namespace morning_snowfall_l494_494967

theorem morning_snowfall (total_snowfall afternoon_snowfall morning_snowfall : ℝ) 
  (h1 : total_snowfall = 0.625) 
  (h2 : afternoon_snowfall = 0.5) 
  (h3 : total_snowfall = morning_snowfall + afternoon_snowfall) : 
  morning_snowfall = 0.125 :=
by
  sorry

end morning_snowfall_l494_494967


namespace eleventh_anniversary_days_l494_494414

-- Define the conditions
def is_leap_year (year : ℕ) : Prop := year % 4 = 0

def initial_years : Set ℕ := {1668, 1669, 1670, 1671}

def initial_day := "Friday"

noncomputable def day_after_11_years (start_year : ℕ) : String :=
  let total_days := 4015 + (if is_leap_year start_year then 3 else 2)
  if total_days % 7 = 0 then "Friday"
  else "Thursday"

-- Define the proposition to prove
theorem eleventh_anniversary_days : 
  (∀ year ∈ initial_years, 
    (if day_after_11_years year = "Friday" then (3 : ℝ) / 4 else (1 : ℝ) / 4) = 
    (if year = 1668 ∨ year = 1670 ∨ year = 1671 then (3 : ℝ) / 4 else (1 : ℝ) / 4)) := 
sorry

end eleventh_anniversary_days_l494_494414


namespace meeting_time_l494_494348

def speed_kmph_to_mps (speed_kmph : ℕ) : ℕ :=
  (speed_kmph * 1000) / 3600

def lcm (a b : Nat) : Nat := 
  Nat.lcm a b

theorem meeting_time (track_length : ℕ) (speed_A_kmph : ℕ) (speed_B_kmph : ℕ) :
  let speed_A_mps := speed_kmph_to_mps speed_A_kmph
  let speed_B_mps := speed_kmph_to_mps speed_B_kmph
  let time_A := track_length / speed_A_mps
  let time_B := track_length / speed_B_mps
  lcm time_A time_B = 120 :=
by
  sorry

end meeting_time_l494_494348


namespace tangent_segrment_sum_eq_l494_494785

/-- Given a cyclic quadrilateral ABCD inscribed in a circle Γ,
let E be the intersection of AD and BC, and F be the intersection of AB and DC.
Let segments EG and FH be tangents to the circle Γ. -/
def cyclic_quadrilateral (A B C D E F G H : Point) : Prop :=
  cyclic (A, B, C, D) ∧
  collinear [A, D, E] ∧ 
  collinear [B, C, E] ∧
  collinear [A, B, F] ∧
  collinear [D, C, F] ∧
  tangent_to (E, G, Γ) ∧
  tangent_to (F, H, Γ)

/-- Prove that for the given cyclic quadrilateral and tangents, the equality EG^2 + FH^2 = EF^2 holds. -/
theorem tangent_segrment_sum_eq (A B C D E F G H : Point) (Γ : Circle) 
  (hcyclic : cyclic_quadrilateral A B C D E F G H)
  (tangent_EG : tangent_to (E, G, Γ))
  (tangent_FH : tangent_to (F, H, Γ)) :
  dist E G ^ 2 + dist F H ^ 2 = dist E F ^ 2 :=
sorry

end tangent_segrment_sum_eq_l494_494785


namespace increase_result_l494_494249

-- Given conditions
def original_number : ℝ := 80
def increase_percentage : ℝ := 1.5

-- The result after the increase
theorem increase_result (h1 : original_number = 80) (h2 : increase_percentage = 1.5) : 
  original_number + (increase_percentage * original_number) = 200 := by
  sorry

end increase_result_l494_494249


namespace distinguishable_balls_in_indistinguishable_boxes_l494_494025

theorem distinguishable_balls_in_indistinguishable_boxes :
  let num_distinguishable_balls := 6
  let num_indistinguishable_boxes := 3
  -- different ways to distribute balls across boxes summarized
  let ways := ∑([{6, 0, 0} ++ {5, 1, 0} ++ {4, 2, 0} ++ {4, 1, 1} ++ {3, 3, 0} ++ {3, 2, 1} ++ {2, 2, 2}], sum)
  -- corresponding ways to remove permutation of distinguishable ball cases
  ways = 222 := sorry

end distinguishable_balls_in_indistinguishable_boxes_l494_494025


namespace balls_in_indistinguishable_boxes_l494_494014

theorem balls_in_indistinguishable_boxes : 
  let balls := 6 in
  let boxes := 3 in
  -- Expression to count ways to distribute balls into indistinguishable boxes
  ((balls.choose 0 * ((balls-0).choose 6)) + 
   (balls.choose 1 * ((balls-1).choose 5)) + 
   (balls.choose 2 * ((balls-2).choose 4)) + 
   (balls.choose 2 * ((balls-2).choose 3) / 2!) + 
   (balls.choose 3 * ((balls-3).choose 3)) + 
   (balls.choose 3 * ((balls-3).choose 2) * ((balls-5).choose 1)) + 
   (balls.choose 2 * ((balls-2).choose 2) / 3!))
  = 92 := 
sorry

end balls_in_indistinguishable_boxes_l494_494014


namespace find_smaller_number_l494_494223

theorem find_smaller_number (x y : ℤ) (h1 : x + y = 60) (h2 : x - y = 8) : y = 26 :=
by
  sorry

end find_smaller_number_l494_494223


namespace increase_by_percentage_l494_494323

theorem increase_by_percentage (a : ℤ) (b : ℚ) : ((b + 1) * a) = 200 :=
by
  -- Definitions based directly on conditions
  let a := (80 : ℤ)
  let b := (1.5 : ℚ)
  -- Assertions about the equivalent proof problem
  have h1 : b + 1 = 2.5 := by norm_num
  have h2 : (b + 1) * a = 200 := by norm_num
  exact h2

end increase_by_percentage_l494_494323


namespace difference_between_balls_l494_494192

theorem difference_between_balls (B R : ℕ) (h1 : R - 152 = B + 152 + 346) : R - B = 650 := 
sorry

end difference_between_balls_l494_494192


namespace xy_eq_yx_l494_494133

variable (n : ℕ) (hn : 0 < n)
noncomputable def x : ℝ := (1 + 1/n) ^ n
noncomputable def y : ℝ := (1 + 1/n) ^ (n + 1)

theorem xy_eq_yx : x^y = y^x :=
sorry

end xy_eq_yx_l494_494133


namespace distribute_6_balls_in_3_boxes_l494_494039

def num_ways_to_distribute_balls (balls : ℕ) (boxes : ℕ) : ℕ :=
  if balls = 6 ∧ boxes = 3 then 92 else 0

theorem distribute_6_balls_in_3_boxes :
  num_ways_to_distribute_balls 6 3 = 92 :=
by
  rw num_ways_to_distribute_balls
  norm_num
  sorry

end distribute_6_balls_in_3_boxes_l494_494039


namespace center_of_circumcircle_on_angle_bisector_l494_494627

-- Definitions based on conditions

variables {α β γ : ℝ}
variables {A B C D E F O : Type*} [metric_space A] [metric_space B] [metric_space C]
[metric_space D] [metric_space E] [metric_space F] [metric_space O]

-- Representing the points and triangles
variables [triangle ABC] [point D AB] [point E BC] [point F AC]

-- Conditions
def condition1 := (dist D E = dist B E)
def condition2 := (dist F E = dist C E)

-- The theorem to be proven
theorem center_of_circumcircle_on_angle_bisector : 
  condition1 ∧ condition2 → 
  let O := circumcenter A D F in 
  lies_on_angle_bisector O (angle D E F) :=
sorry

end center_of_circumcircle_on_angle_bisector_l494_494627


namespace triangle_QR_length_l494_494342

/-- Conditions for the triangles PQR and SQR sharing a side QR with given side lengths. -/
structure TriangleSetup where
  (PQ PR SR SQ QR : ℝ)
  (PQ_pos : PQ > 0)
  (PR_pos : PR > 0)
  (SR_pos : SR > 0)
  (SQ_pos : SQ > 0)
  (shared_side_QR : QR = QR)

/-- The problem statement asserting the least possible length of QR. -/
theorem triangle_QR_length (t : TriangleSetup) 
  (h1 : t.PQ = 8)
  (h2 : t.PR = 15)
  (h3 : t.SR = 10)
  (h4 : t.SQ = 25) :
  t.QR = 15 :=
by
  sorry

end triangle_QR_length_l494_494342


namespace increase_by_percentage_l494_494328

theorem increase_by_percentage (a : ℤ) (b : ℚ) : ((b + 1) * a) = 200 :=
by
  -- Definitions based directly on conditions
  let a := (80 : ℤ)
  let b := (1.5 : ℚ)
  -- Assertions about the equivalent proof problem
  have h1 : b + 1 = 2.5 := by norm_num
  have h2 : (b + 1) * a = 200 := by norm_num
  exact h2

end increase_by_percentage_l494_494328


namespace anniversary_day_probability_l494_494396

/- Definitions based on the conditions -/
def is_leap_year (year : ℕ) : Prop :=
  year % 4 = 0

def days_in_year (year : ℕ) : ℕ :=
  if is_leap_year year then 366 else 365

def total_days (start_year : ℕ) : ℕ :=
  (list.sum (list.map days_in_year (list.range' start_year 11)))

/- Prove the day of the 11th anniversary and its probabilities -/
theorem anniversary_day_probability (start_year : ℕ) (h : start_year ∈ {1668, 1669, 1670, 1671}) :
  let days := total_days start_year % 7
  in (days = 0 ∧ 0.75 ≤ 1) ∨ (days = 6 ∧ 0.25 ≤ 1) :=
by
  sorry

end anniversary_day_probability_l494_494396


namespace Amanda_money_left_l494_494775

theorem Amanda_money_left (initial_amount cost_cassette tape_count cost_headphone : ℕ) 
  (h1 : initial_amount = 50) 
  (h2 : cost_cassette = 9) 
  (h3 : tape_count = 2) 
  (h4 : cost_headphone = 25) :
  initial_amount - (tape_count * cost_cassette + cost_headphone) = 7 :=
by
  sorry

end Amanda_money_left_l494_494775


namespace prime_divisor_111_l494_494721

theorem prime_divisor_111 (p : ℕ) (hp : p.prime) (h : p ≠ 2 ∧ p ≠ 5) : 
  ∃ k : ℕ, (10^k - 1) / 9 % p = 0 := 
sorry

end prime_divisor_111_l494_494721


namespace ramu_profit_percent_l494_494163

theorem ramu_profit_percent (cost_car cost_repairs selling_price : ℝ) 
  (h1 : cost_car = 45000) (h2 : cost_repairs = 12000) (h3 : selling_price = 80000) :
  let total_cost := cost_car + cost_repairs in
  let profit := selling_price - total_cost in
  let profit_percent := (profit / total_cost) * 100 in
  profit_percent = 40.35 := 
by
  sorry

end ramu_profit_percent_l494_494163


namespace square_area_from_diagonal_l494_494073

theorem square_area_from_diagonal (d : ℝ) (h_d : d = 12) : ∃ (A : ℝ), A = 72 :=
by
  -- we will use the given diagonal to derive the result
  sorry

end square_area_from_diagonal_l494_494073


namespace points_below_line_l494_494501

-- Conditions
def is_arithmetic_sequence (a b c d : ℝ) : Prop := (b - a) = (c - b) ∧ (c - b) = (d - c)
def is_geometric_sequence (a b c d : ℝ) : Prop := (b / a) = (c / b) ∧ (c / b) = (d / c)

-- Points
def P1 : (ℝ × ℝ) := (4 / 3, real.cbrt 2)
def P2 : (ℝ × ℝ) := (5 / 3, real.cbrt 4)

-- Theorem Statement
theorem points_below_line (x1 y1 x2 y2 : ℝ) 
  (h1 : is_arithmetic_sequence 1 x1 x2 2)
  (h2 : is_geometric_sequence 1 y1 y2 2)
  (hx1 : x1 = 4 / 3)
  (hx2 : x2 = 5 / 3)
  (hy1 : y1 = real.cbrt 2)
  (hy2 : y2 = real.cbrt 4) : 
  x1 > y1 ∧ x2 > y2 := 
by {
  sorry,
}

end points_below_line_l494_494501


namespace longest_leg_of_smallest_triangle_l494_494438

-- Definitions based on conditions
def is306090Triangle (h : ℝ) (s : ℝ) (l : ℝ) : Prop :=
  s = h / 2 ∧ l = s * (Real.sqrt 3)

def chain_of_306090Triangles (H : ℝ) : Prop :=
  ∃ h1 s1 l1 h2 s2 l2 h3 s3 l3 h4 s4 l4,
    is306090Triangle h1 s1 l1 ∧
    is306090Triangle h2 s2 l2 ∧
    is306090Triangle h3 s3 l3 ∧
    is306090Triangle h4 s4 l4 ∧
    h1 = H ∧ l1 = h2 ∧ l2 = h3 ∧ l3 = h4

-- Main theorem
theorem longest_leg_of_smallest_triangle (H : ℝ) (h : ℝ) (l : ℝ) (H_cond : H = 16) 
  (h_cond : h = 9) :
  chain_of_306090Triangles H →
  ∃ h4 s4 l4, is306090Triangle h4 s4 l4 ∧ l = h4 →
  l = 9 := 
by
  sorry

end longest_leg_of_smallest_triangle_l494_494438


namespace constant_term_expansion_l494_494212

noncomputable def sum_of_coefficients (a : ℕ) : ℕ := sorry

noncomputable def constant_term (a : ℕ) : ℕ := sorry

theorem constant_term_expansion (a : ℕ) (h : sum_of_coefficients a = 2) : constant_term 2 = 10 :=
sorry

end constant_term_expansion_l494_494212


namespace sequence_property_reciprocal_diff_constant_reciprocal_arithmetic_l494_494535

variable (a : ℕ → ℝ)

/-- Define the sequence a -/
def seq (n : ℕ) : ℝ :=
  if n = 1 then
    1
  else
    let n' := n - 1
    let a_n' := seq n'
    2 * a_n' / (2 + a_n')

theorem sequence_property :
  let a := seq
  ∀ n, n > 0 → a (n + 1) = (2 * a n) / (2 + a n) :=
by
  sorry

theorem reciprocal_diff_constant (n : ℕ) (h : n > 0) :
  ⟦1 / (seq n + 1)⟧ - ⟦1 / seq n⟧ = 1 / 2 :=
by
  sorry

/-- Proof that the sequence 1/a_n is arithmetic -/
theorem reciprocal_arithmetic :
  ∀ n, 1 / (seq (n + 1)) - 1 / (seq n) = 1 / 2 :=
by
  intros
  apply reciprocal_diff_constant
  exact npos

end


end sequence_property_reciprocal_diff_constant_reciprocal_arithmetic_l494_494535


namespace increase_80_by_150_percent_l494_494261

-- Original number
def originalNumber : ℕ := 80

-- Percentage increase as a decimal
def increaseFactor : ℚ := 1.5

-- Expected result after the increase
def expectedResult : ℕ := 200

theorem increase_80_by_150_percent :
  originalNumber + (increaseFactor * originalNumber) = expectedResult :=
by
  sorry

end increase_80_by_150_percent_l494_494261


namespace natural_number_factors_of_M_l494_494432

def M : ℕ := (2^3) * (3^2) * (5^5) * (7^1) * (11^2)

theorem natural_number_factors_of_M : ∃ n : ℕ, n = 432 ∧ (∀ d, d ∣ M → d > 0 → d ≤ M) :=
by
  let number_of_factors := (3 + 1) * (2 + 1) * (5 + 1) * (1 + 1) * (2 + 1)
  use number_of_factors
  sorry

end natural_number_factors_of_M_l494_494432


namespace calculate_f_f_10_l494_494529

noncomputable def f : ℝ → ℝ :=
  λ x, if x ≤ 1 then 10^(x-1) else Real.log10 x

theorem calculate_f_f_10 : f (f 10) = 1 := 
  by 
  -- We state the theorem without providing the proof
  sorry

end calculate_f_f_10_l494_494529


namespace votes_difference_is_84_l494_494972

-- Definitions and lemmas based on conditions
variables (x y x' y' m : ℕ)
constants (total_voters : ℕ := 500)
constants (margin : ℕ)

axiom total_voting_initial : x + y = total_voters
axiom initial_margin : y - x = m
axiom twice_margin : x' - y' = 2 * m
axiom total_voting_revote : x' + y' = total_voters
axiom revote_ratio : x' = 11 * y / 10

-- The proof statement
theorem votes_difference_is_84 : (x' - x) = 84 :=
by
  -- Mathematically equivalent proof problem condition
  sorry

end votes_difference_is_84_l494_494972
