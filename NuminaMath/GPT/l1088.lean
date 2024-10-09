import Mathlib

namespace min_value_l1088_108872

theorem min_value (a b c x y z : ℝ) (h1 : a + b + c = 1) (h2 : x + y + z = 1) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  ∃ val : ℝ, val = -1 / 4 ∧ ∀ a b c x y z : ℝ, 0 ≤ a → 0 ≤ b → 0 ≤ c → 0 ≤ x → 0 ≤ y → 0 ≤ z → a + b + c = 1 → x + y + z = 1 → (a - x^2) * (b - y^2) * (c - z^2) ≥ val :=
sorry

end min_value_l1088_108872


namespace proof_problem_l1088_108812

-- Declare x, y as real numbers
variables (x y : ℝ)

-- Define the condition given in the problem
def condition (k : ℝ) : Prop :=
  (x^2 + y^2) / (x^2 - y^2) + (x^2 - y^2) / (x^2 + y^2) = k

-- The main conclusion we need to prove given the condition
theorem proof_problem (k : ℝ) (h : condition x y k) :
  (x^8 + y^8) / (x^8 - y^8) + (x^8 - y^8) / (x^8 + y^8) = (k^4 + 24 * k^2 + 16) / (4 * k^3 + 16 * k) :=
sorry

end proof_problem_l1088_108812


namespace dagger_computation_l1088_108829

def dagger (m n p q : ℕ) (hn : n ≠ 0) (hm : m ≠ 0) : ℚ :=
  (m^2 * p * (q / n)) + ((p : ℚ) / m)

theorem dagger_computation :
  dagger 5 9 6 2 (by norm_num) (by norm_num) = 518 / 15 :=
sorry

end dagger_computation_l1088_108829


namespace problem_Z_value_l1088_108855

def Z (a b : ℕ) : ℕ := 3 * (a - b) ^ 2

theorem problem_Z_value : Z 5 3 = 12 := by
  sorry

end problem_Z_value_l1088_108855


namespace shoe_size_percentage_difference_l1088_108876

theorem shoe_size_percentage_difference :
  ∀ (size8_len size15_len size17_len : ℝ)
  (h1 : size8_len = size15_len - (7 * (1 / 5)))
  (h2 : size17_len = size15_len + (2 * (1 / 5)))
  (h3 : size15_len = 10.4),
  ((size17_len - size8_len) / size8_len) * 100 = 20 := by
  intros size8_len size15_len size17_len h1 h2 h3
  sorry

end shoe_size_percentage_difference_l1088_108876


namespace sqrt_fourth_root_l1088_108845

theorem sqrt_fourth_root (h : Real.sqrt (Real.sqrt (0.00000081)) = 0.1732) : Real.sqrt (Real.sqrt (0.00000081)) = 0.2 :=
by
  sorry

end sqrt_fourth_root_l1088_108845


namespace line_relation_with_plane_l1088_108850

variables {P : Type} [Infinite P] [MetricSpace P]

variables (a b : Line P) (α : Plane P)

-- Conditions
axiom intersecting_lines : ∃ p : P, p ∈ a ∧ p ∈ b
axiom line_parallel_plane : ∀ p : P, p ∈ a → p ∈ α

-- Theorem statement for the proof problem
theorem line_relation_with_plane : (∀ p : P, p ∈ b → p ∈ α) ∨ (∃ q : P, q ∈ α ∧ q ∈ b) :=
sorry

end line_relation_with_plane_l1088_108850


namespace range_of_a_l1088_108844

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a * Real.log (x + 1) - x^2

theorem range_of_a (a : ℝ) (p q : ℝ) (h₀ : p ≠ q) (h₁ : -1 < p ∧ p < 0) (h₂ : -1 < q ∧ q < 0) :
  (∀ p q : ℝ, -1 < p ∧ p < 0 → -1 < q ∧ q < 0 → p ≠ q → (f a (p + 1) - f a (q + 1)) / (p - q) > 1) ↔ (6 ≤ a) :=
by
  -- proof is omitted
  sorry

end range_of_a_l1088_108844


namespace age_of_B_present_l1088_108894

theorem age_of_B_present (A B C : ℕ) (h1 : A + B + C = 90)
  (h2 : (A - 10) * 2 = (B - 10))
  (h3 : (B - 10) * 3 = (C - 10) * 2) :
  B = 30 := 
sorry

end age_of_B_present_l1088_108894


namespace problem_statement_l1088_108808

theorem problem_statement (a : Fin 17 → ℕ)
  (h : ∀ i : Fin 17, a i ^ a (i + 1) = a (i + 1) ^ a (i + 2)): 
  a 0 = a 1 :=
sorry

end problem_statement_l1088_108808


namespace top_angle_degrees_l1088_108879

def isosceles_triangle_with_angle_ratio (x : ℝ) (a b c : ℝ) : Prop :=
  a = x ∧ b = 4 * x ∧ a + b + c = 180 ∧ (a = b ∨ a = c ∨ b = c)

theorem top_angle_degrees (x : ℝ) (a b c : ℝ) :
  isosceles_triangle_with_angle_ratio x a b c → c = 20 ∨ c = 120 :=
by
  sorry

end top_angle_degrees_l1088_108879


namespace quiz_score_of_dropped_student_l1088_108806

theorem quiz_score_of_dropped_student (avg16 : ℝ) (avg15 : ℝ) (num_students : ℝ) (dropped_students : ℝ) (x : ℝ)
  (h1 : avg16 = 60.5) (h2 : avg15 = 64) (h3 : num_students = 16) (h4 : dropped_students = 1) :
  x = 60.5 * 16 - 64 * 15 :=
by
  sorry

end quiz_score_of_dropped_student_l1088_108806


namespace solve_for_k_l1088_108835

-- Definition and conditions
def ellipse_eq (k : ℝ) : Prop := ∀ x y, k * x^2 + 5 * y^2 = 5

-- Problem: Prove k = 1 given the above definitions
theorem solve_for_k (k : ℝ) :
  (exists (x y : ℝ), ellipse_eq k ∧ x = 2 ∧ y = 0) -> k = 1 :=
sorry

end solve_for_k_l1088_108835


namespace johns_gym_time_l1088_108858

noncomputable def time_spent_at_gym (day : String) : ℝ :=
  match day with
  | "Monday" => 1 + 0.5
  | "Tuesday" => 40/60 + 20/60 + 15/60
  | "Thursday" => 40/60 + 20/60 + 15/60
  | "Saturday" => 1.5 + 0.75
  | "Sunday" => 10/60 + 50/60 + 10/60
  | _ => 0

noncomputable def total_hours_per_week : ℝ :=
  time_spent_at_gym "Monday" 
  + 2 * time_spent_at_gym "Tuesday" 
  + time_spent_at_gym "Saturday" 
  + time_spent_at_gym "Sunday"

theorem johns_gym_time : total_hours_per_week = 7.4167 := by
  sorry

end johns_gym_time_l1088_108858


namespace smallest_x_y_sum_l1088_108885

theorem smallest_x_y_sum (x y : ℕ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x ≠ y)
                        (h4 : (1 / (x : ℝ)) + (1 / (y : ℝ)) = (1 / 20)) :
    x + y = 81 :=
sorry

end smallest_x_y_sum_l1088_108885


namespace tom_spent_correct_amount_l1088_108814

-- Define the prices of the games
def batman_game_price : ℝ := 13.6
def superman_game_price : ℝ := 5.06

-- Define the total amount spent calculation
def total_spent := batman_game_price + superman_game_price

-- The main statement to prove
theorem tom_spent_correct_amount : total_spent = 18.66 := by
  -- Proof (intended)
  sorry

end tom_spent_correct_amount_l1088_108814


namespace sum_of_7_terms_arithmetic_seq_l1088_108809

variable {α : Type*} [LinearOrderedField α]

def arithmetic_seq (a : ℕ → α) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem sum_of_7_terms_arithmetic_seq (a : ℕ → α) (h_arith : arithmetic_seq a)
  (h_a4 : a 4 = 2) :
  (7 * (a 1 + a 7)) / 2 = 14 :=
sorry

end sum_of_7_terms_arithmetic_seq_l1088_108809


namespace find_a_l1088_108813

def A : Set ℝ := { x | x^2 - 5 * x + 6 = 0 }
def B (a : ℝ) : Set ℝ := { x | a * x - 1 = 0 }

theorem find_a (a : ℝ) : (A ∩ B a = B a) → (a = 0 ∨ a = 1 / 2 ∨ a = 1 / 3) := by
  sorry

end find_a_l1088_108813


namespace problem_1_problem_2_l1088_108857

noncomputable def f (x : ℝ) := Real.sin x + (x - 1) / Real.exp x

theorem problem_1 (x : ℝ) (h₀ : x ∈ Set.Icc (-Real.pi) (Real.pi / 2)) :
  MonotoneOn f (Set.Icc (-Real.pi) (Real.pi / 2)) :=
sorry

theorem problem_2 (k : ℝ) :
  ∀ x ∈ Set.Icc (-Real.pi) 0, ((f x - Real.sin x) * Real.exp x - Real.cos x) ≤ k * Real.sin x → 
  k ∈ Set.Iic (1 + Real.pi / 2) :=
sorry

end problem_1_problem_2_l1088_108857


namespace system_of_equations_solution_l1088_108898

theorem system_of_equations_solution (x y z : ℝ) 
  (h1 : x + 2 * y = 4)
  (h2 : 2 * x + 5 * y - 2 * z = 11)
  (h3 : 3 * x - 5 * y + 2 * z = -1) : 
  x = 2 ∧ y = 1 ∧ z = -1 :=
by {
  sorry
}

end system_of_equations_solution_l1088_108898


namespace num_solutions_20_l1088_108883

def num_solutions (n : ℕ) : ℕ :=
  4 * n

theorem num_solutions_20 : num_solutions 20 = 80 := by
  sorry

end num_solutions_20_l1088_108883


namespace relationship_between_exponents_l1088_108842

theorem relationship_between_exponents 
  (p r : ℝ) (u v s t m n : ℝ)
  (h1 : p^u = r^s)
  (h2 : r^v = p^t)
  (h3 : m = r^s)
  (h4 : n = r^v)
  (h5 : m^2 = n^3) :
  (s / u = v / t) ∧ (2 * s = 3 * v) :=
  by
  sorry

end relationship_between_exponents_l1088_108842


namespace regular_polygon_sides_l1088_108837

theorem regular_polygon_sides (n : ℕ) (h1 : 2 ≤ n) (h2 : (n - 2) * 180 / n = 120) : n = 6 :=
by
  sorry

end regular_polygon_sides_l1088_108837


namespace circle_tangent_to_parabola_directrix_l1088_108867

theorem circle_tangent_to_parabola_directrix (m : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + m * x - 1/4 = 0 → y^2 = 4 * x → x = -1) → m = 3/4 :=
by
  sorry

end circle_tangent_to_parabola_directrix_l1088_108867


namespace number_of_girls_l1088_108871

open Rat

theorem number_of_girls 
  (G B : ℕ) 
  (h1 : G / B = 5 / 8)
  (h2 : G + B = 300) 
  : G = 116 := 
by
  sorry

end number_of_girls_l1088_108871


namespace slices_eaten_l1088_108847

theorem slices_eaten (slices_cheese : ℕ) (slices_pepperoni : ℕ) (slices_left_per_person : ℕ) (phil_andre_slices_left : ℕ) :
  (slices_cheese + slices_pepperoni = 22) →
  (slices_left_per_person = 2) →
  (phil_andre_slices_left = 2 + 2) →
  (slices_cheese + slices_pepperoni - phil_andre_slices_left = 18) :=
by
  intros
  sorry

end slices_eaten_l1088_108847


namespace inverse_function_fixed_point_l1088_108878

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x-2) - 1

theorem inverse_function_fixed_point
  (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) :
  ∃ g : ℝ → ℝ, (∀ y : ℝ, g (f a y) = y) ∧ g 0 = 2 :=
sorry

end inverse_function_fixed_point_l1088_108878


namespace diff_not_equal_l1088_108831

variable (A B : Set ℕ)

def diff (A B : Set ℕ) : Set ℕ := {x | x ∈ A ∧ x ∉ B}

theorem diff_not_equal (A B : Set ℕ) :
  A ≠ ∅ ∧ B ≠ ∅ → (diff A B ≠ diff B A) :=
by
  sorry

end diff_not_equal_l1088_108831


namespace pet_store_cages_l1088_108848

theorem pet_store_cages (initial_puppies sold_puppies puppies_per_cage remaining_puppies num_cages : ℕ)
  (h1 : initial_puppies = 102) 
  (h2 : sold_puppies = 21) 
  (h3 : puppies_per_cage = 9) 
  (h4 : remaining_puppies = initial_puppies - sold_puppies)
  (h5 : num_cages = remaining_puppies / puppies_per_cage) : 
  num_cages = 9 := 
by
  sorry

end pet_store_cages_l1088_108848


namespace find_integers_l1088_108864

theorem find_integers (n : ℕ) (h1 : n < 10^100)
  (h2 : n ∣ 2^n) (h3 : n - 1 ∣ 2^n - 1) (h4 : n - 2 ∣ 2^n - 2) :
  n = 2^2 ∨ n = 2^4 ∨ n = 2^16 ∨ n = 2^256 := by
  sorry

end find_integers_l1088_108864


namespace sum_of_possible_values_of_N_l1088_108807

variable (N S : ℝ) (hN : N ≠ 0)

theorem sum_of_possible_values_of_N : 
  (3 * N + 5 / N = S) → 
  ∀ N1 N2 : ℝ, (3 * N1^2 - S * N1 + 5 = 0) ∧ (3 * N2^2 - S * N2 + 5 = 0) → 
  N1 + N2 = S / 3 :=
by 
  intro hS hRoots
  sorry

end sum_of_possible_values_of_N_l1088_108807


namespace simple_interest_years_l1088_108800

theorem simple_interest_years (P R : ℝ) (T : ℝ) :
  P = 2500 → (2500 * (R + 2) / 100 * T = 2500 * R / 100 * T + 250) → T = 5 :=
by
  intro hP h
  -- Note: Actual proof details would go here
  sorry

end simple_interest_years_l1088_108800


namespace wendy_full_face_time_l1088_108882

-- Define the constants based on the conditions
def num_products := 5
def wait_time := 5
def makeup_time := 30

-- Calculate the total time to put on "full face"
def total_time (products : ℕ) (wait_time : ℕ) (makeup_time : ℕ) : ℕ :=
  (products - 1) * wait_time + makeup_time

-- The theorem stating that Wendy's full face routine takes 50 minutes
theorem wendy_full_face_time : total_time num_products wait_time makeup_time = 50 :=
by {
  -- the proof would be provided here, for now we use sorry
  sorry
}

end wendy_full_face_time_l1088_108882


namespace largest_is_21_l1088_108804

theorem largest_is_21(a b c d : ℕ) 
  (h1 : (a + b + c) / 3 + d = 17)
  (h2 : (a + b + d) / 3 + c = 21)
  (h3 : (a + c + d) / 3 + b = 23)
  (h4 : (b + c + d) / 3 + a = 29):
  d = 21 := 
sorry

end largest_is_21_l1088_108804


namespace bowling_ball_weight_l1088_108823

theorem bowling_ball_weight (b c : ℝ) (h1 : c = 36) (h2 : 5 * b = 4 * c) : b = 28.8 := by
  sorry

end bowling_ball_weight_l1088_108823


namespace remainder_of_division_l1088_108888

theorem remainder_of_division : 
  ∀ (L x : ℕ), (L = 1430) → 
               (L - x = 1311) → 
               (L = 11 * x + (L % x)) → 
               (L % x = 121) :=
by
  intros L x L_value diff quotient
  sorry

end remainder_of_division_l1088_108888


namespace salary_january_l1088_108877

theorem salary_january
  (J F M A May : ℝ)  -- declare the salaries as real numbers
  (h1 : (J + F + M + A) / 4 = 8000)  -- condition 1
  (h2 : (F + M + A + May) / 4 = 9500)  -- condition 2
  (h3 : May = 6500) :  -- condition 3
  J = 500 := 
by
  sorry

end salary_january_l1088_108877


namespace bernardo_larger_probability_l1088_108860

-- Mathematical definitions
def bernardo_set : Finset ℕ := {1,2,3,4,5,6,7,8,10}
def silvia_set : Finset ℕ := {1,2,3,4,5,6}

-- Probability calculation function (you need to define the detailed implementation)
noncomputable def probability_bernardo_gt_silvia : ℚ := sorry

-- The proof statement
theorem bernardo_larger_probability : 
  probability_bernardo_gt_silvia = 13 / 20 :=
sorry

end bernardo_larger_probability_l1088_108860


namespace problem_1_problem_2_problem_3_l1088_108856

section MathProblems

variable (a b c m n x y : ℝ)
-- Problem 1
theorem problem_1 :
  (-6 * a^2 * b^5 * c) / (-2 * a * b^2)^2 = (3/2) * b * c := sorry

-- Problem 2
theorem problem_2 :
  (-3 * m - 2 * n) * (3 * m + 2 * n) = -9 * m^2 - 12 * m * n - 4 * n^2 := sorry

-- Problem 3
theorem problem_3 :
  ((x - 2 * y)^2 - (x - 2 * y) * (x + 2 * y)) / (2 * y) = -2 * x + 4 * y := sorry

end MathProblems

end problem_1_problem_2_problem_3_l1088_108856


namespace find_numbers_l1088_108830

-- Define the conditions
def condition_1 (L S : ℕ) : Prop := L - S = 8327
def condition_2 (L S : ℕ) : Prop := ∃ q r, L = q * S + r ∧ q = 21 ∧ r = 125

-- Define the math proof problem
theorem find_numbers (S L : ℕ) (h1 : condition_1 L S) (h2 : condition_2 L S) : S = 410 ∧ L = 8735 :=
by
  sorry

end find_numbers_l1088_108830


namespace intersection_of_M_and_N_is_correct_l1088_108824

-- Definitions according to conditions
def M : Set ℤ := {-4, -2, 0, 2, 4, 6}
def N : Set ℤ := {x | -3 ≤ x ∧ x ≤ 4}

-- Proof statement
theorem intersection_of_M_and_N_is_correct : (M ∩ N) = {-2, 0, 2, 4} := by
  sorry

end intersection_of_M_and_N_is_correct_l1088_108824


namespace valid_votes_per_candidate_l1088_108899

theorem valid_votes_per_candidate (total_votes : ℕ) (invalid_percentage valid_percentage_A valid_percentage_B : ℚ) 
                                  (A_votes B_votes C_votes valid_votes : ℕ) :
  total_votes = 1250000 →
  invalid_percentage = 20 →
  valid_percentage_A = 45 →
  valid_percentage_B = 35 →
  valid_votes = total_votes * (1 - invalid_percentage / 100) →
  A_votes = valid_votes * (valid_percentage_A / 100) →
  B_votes = valid_votes * (valid_percentage_B / 100) →
  C_votes = valid_votes - A_votes - B_votes →
  valid_votes = 1000000 ∧ A_votes = 450000 ∧ B_votes = 350000 ∧ C_votes = 200000 :=
by {
  sorry
}

end valid_votes_per_candidate_l1088_108899


namespace value_of_expression_l1088_108891

theorem value_of_expression (a b : ℝ) (h1 : ∃ x : ℝ, x^2 + 3 * x - 5 = 0)
  (h2 : ∃ y : ℝ, y^2 + 3 * y - 5 = 0)
  (h3 : a ≠ b)
  (h4 : ∀ r : ℝ, r^2 + 3 * r - 5 = 0 → r = a ∨ r = b) : a^2 + 3 * a * b + a - 2 * b = -4 :=
by
  sorry

end value_of_expression_l1088_108891


namespace proof_2d_minus_r_l1088_108854

theorem proof_2d_minus_r (d r: ℕ) (h1 : 1059 % d = r)
  (h2 : 1482 % d = r) (h3 : 2340 % d = r) (hd : d > 1) : 2 * d - r = 6 := 
by 
  sorry

end proof_2d_minus_r_l1088_108854


namespace eighty_percent_of_number_l1088_108849

theorem eighty_percent_of_number (x : ℝ) (h : 0.20 * x = 60) : 0.80 * x = 240 := 
by sorry

end eighty_percent_of_number_l1088_108849


namespace extra_cost_from_online_purchase_l1088_108874

-- Define the in-store price
def inStorePrice : ℝ := 150.00

-- Define the online payment and processing fee
def onlinePayment : ℝ := 35.00
def processingFee : ℝ := 12.00

-- Calculate the total online cost
def totalOnlineCost : ℝ := (4 * onlinePayment) + processingFee

-- Calculate the difference in cents
def differenceInCents : ℝ := (totalOnlineCost - inStorePrice) * 100

-- The proof statement
theorem extra_cost_from_online_purchase : differenceInCents = 200 :=
by
  -- Proof steps go here
  sorry

end extra_cost_from_online_purchase_l1088_108874


namespace minimum_value_a_2b_3c_l1088_108881

theorem minimum_value_a_2b_3c (a b c : ℝ)
  (h : ∀ x y : ℝ, x + 2*y - 3 ≤ a*x + b*y + c ∧ a*x + b*y + c ≤ x + 2*y + 3) :
  (a + 2*b - 3*c) = -4 :=
sorry

end minimum_value_a_2b_3c_l1088_108881


namespace train_ride_time_in_hours_l1088_108840

-- Definition of conditions
def lukes_total_trip_time_hours : ℕ := 8
def bus_ride_minutes : ℕ := 75
def walk_to_train_center_minutes : ℕ := 15
def wait_time_minutes : ℕ := 2 * walk_to_train_center_minutes

-- Convert total trip time to minutes
def lukes_total_trip_time_minutes : ℕ := lukes_total_trip_time_hours * 60

-- Calculate the total time spent on bus, walking, and waiting
def bus_walk_wait_time_minutes : ℕ :=
  bus_ride_minutes + walk_to_train_center_minutes + wait_time_minutes

-- Calculate the train ride time in minutes
def train_ride_time_minutes : ℕ :=
  lukes_total_trip_time_minutes - bus_walk_wait_time_minutes

-- Prove the train ride time in hours
theorem train_ride_time_in_hours : train_ride_time_minutes / 60 = 6 :=
by
  sorry

end train_ride_time_in_hours_l1088_108840


namespace max_age_l1088_108859

-- Definitions of the conditions
def born_same_day (max_birth luka_turn4 : ℕ) : Prop := max_birth = luka_turn4
def age_difference (luka_age aubrey_age : ℕ) : Prop := luka_age = aubrey_age + 2
def aubrey_age_on_birthday : ℕ := 8

-- Prove that Max's age is 6 years when Aubrey is 8 years old
theorem max_age (luka_birth aubrey_birth max_birth : ℕ) 
                (h1 : born_same_day max_birth luka_birth) 
                (h2 : age_difference luka_birth aubrey_birth) : 
                (aubrey_birth + 4 - luka_birth) = 6 :=
by
  sorry

end max_age_l1088_108859


namespace medieval_society_hierarchy_l1088_108836

-- Given conditions
def members := 12
def king_choices := members
def remaining_after_king := members - 1
def duke_choices : ℕ := remaining_after_king * (remaining_after_king - 1) * (remaining_after_king - 2)
def knight_choices : ℕ := Nat.choose (remaining_after_king - 2) 2 * Nat.choose (remaining_after_king - 4) 2 * Nat.choose (remaining_after_king - 6) 2

-- The number of ways to establish the hierarchy can be stated as:
def total_ways : ℕ := king_choices * duke_choices * knight_choices

-- Our main theorem
theorem medieval_society_hierarchy : total_ways = 907200 := by
  -- Proof would go here, we skip it with sorry
  sorry

end medieval_society_hierarchy_l1088_108836


namespace solve_for_s_l1088_108896

theorem solve_for_s :
  let numerator := Real.sqrt (7^2 + 24^2)
  let denominator := Real.sqrt (64 + 36)
  let s := numerator / denominator
  s = 5 / 2 :=
by
  sorry

end solve_for_s_l1088_108896


namespace eulers_formula_l1088_108834

structure PlanarGraph :=
(vertices : ℕ)
(edges : ℕ)
(faces : ℕ)
(connected : Prop)

theorem eulers_formula (G: PlanarGraph) (H_conn: G.connected) : G.vertices - G.edges + G.faces = 2 :=
sorry

end eulers_formula_l1088_108834


namespace sum_of_proper_divisors_less_than_100_of_780_l1088_108887

def is_divisor (n d : ℕ) : Bool :=
  d ∣ n

def proper_divisors (n : ℕ) : List ℕ :=
  (List.range n).filter (λ d => d ∣ n ∧ d < n)

def proper_divisors_less_than (n bound : ℕ) : List ℕ :=
  (proper_divisors n).filter (λ d => d < bound)

def sum_list (l : List ℕ) : ℕ :=
  l.foldl (λ acc x => acc + x) 0

theorem sum_of_proper_divisors_less_than_100_of_780 :
  sum_list (proper_divisors_less_than 780 100) = 428 :=
by
  sorry

end sum_of_proper_divisors_less_than_100_of_780_l1088_108887


namespace texts_sent_on_Tuesday_l1088_108817

theorem texts_sent_on_Tuesday (total_texts monday_texts : Nat) (texts_each_monday : Nat)
  (h_monday : texts_each_monday = 5)
  (h_total : total_texts = 40)
  (h_monday_total : monday_texts = 2 * texts_each_monday) :
  total_texts - monday_texts = 30 := by
  sorry

end texts_sent_on_Tuesday_l1088_108817


namespace find_c_in_triangle_l1088_108862

theorem find_c_in_triangle
  (A : Real) (a b S : Real) (c : Real)
  (hA : A = 60) 
  (ha : a = 6 * Real.sqrt 3)
  (hb : b = 12)
  (hS : S = 18 * Real.sqrt 3) :
  c = 6 := by
  sorry

end find_c_in_triangle_l1088_108862


namespace intersection_A_compl_B_subset_E_B_l1088_108838

namespace MathProof

-- Definitions
def A := {x : ℝ | (x + 3) * (x - 6) ≥ 0}
def B := {x : ℝ | (x + 2) / (x - 14) < 0}
def compl_R_B := {x : ℝ | x ≤ -2 ∨ x ≥ 14}
def E (a : ℝ) := {x : ℝ | 2 * a < x ∧ x < a + 1}

-- Theorem for intersection of A and complement of B
theorem intersection_A_compl_B : A ∩ compl_R_B = {x : ℝ | x ≤ -3 ∨ x ≥ 14} :=
by
  sorry

-- Theorem for subset relationship to determine range of a
theorem subset_E_B (a : ℝ) : (E a ⊆ B) → a ≥ -1 :=
by
  sorry

end MathProof

end intersection_A_compl_B_subset_E_B_l1088_108838


namespace square_window_side_length_l1088_108811

-- Definitions based on the conditions
def total_panes := 8
def rows := 2
def cols := 4
def height_ratio := 3
def width_ratio := 1
def border_width := 3

-- The statement to prove
theorem square_window_side_length :
  let height := 3 * (1 : ℝ)
  let width := 1 * (1 : ℝ)
  let total_width := cols * width + (cols + 1) * border_width
  let total_height := rows * height + (rows + 1) * border_width
  total_width = total_height → total_width = 27 :=
by
  sorry

end square_window_side_length_l1088_108811


namespace complex_imaginary_part_l1088_108828

theorem complex_imaginary_part (z : ℂ) (h : z + (3 - 4 * I) = 1) : z.im = 4 :=
  sorry

end complex_imaginary_part_l1088_108828


namespace find_a_plus_b_l1088_108869
-- Definition of the problem variables and conditions
variables (a b : ℝ)
def condition1 : Prop := a - b = 3
def condition2 : Prop := a^2 - b^2 = -12

-- Goal: Prove that a + b = -4 given the conditions
theorem find_a_plus_b (h1 : condition1 a b) (h2 : condition2 a b) : a + b = -4 :=
  sorry

end find_a_plus_b_l1088_108869


namespace intersection_A_B_l1088_108852

def A := {y : ℝ | ∃ x : ℝ, y = 2^x}
def B := {y : ℝ | ∃ x : ℝ, y = -x^2 + 2}
def Intersection := {y : ℝ | 0 < y ∧ y ≤ 2}

theorem intersection_A_B :
  (A ∩ B) = Intersection :=
by
  sorry

end intersection_A_B_l1088_108852


namespace spherical_caps_ratio_l1088_108802

theorem spherical_caps_ratio (r : ℝ) (m₁ m₂ : ℝ) (σ₁ σ₂ : ℝ)
  (h₁ : r = 1)
  (h₂ : σ₁ = 2 * π * m₁ + π * (1 - (1 - m₁)^2))
  (h₃ : σ₂ = 2 * π * m₂ + π * (1 - (1 - m₂)^2))
  (h₄ : σ₁ + σ₂ = 5 * π)
  (h₅ : m₁ + m₂ = 2) :
  (2 * m₁ + (1 - (1 - m₁)^2)) / (2 * m₂ + (1 - (1 - m₂)^2)) = 3.6 :=
sorry

end spherical_caps_ratio_l1088_108802


namespace circle_equation_l1088_108875

-- Definitions based on the conditions
def center_on_x_axis (a b r : ℝ) := b = 0
def tangent_at_point (a b r : ℝ) := (b - 1) / a = -1/2

-- Proof statement
theorem circle_equation (a b r : ℝ) (h1: center_on_x_axis a b r) (h2: tangent_at_point a b r) :
    ∃ (a b r : ℝ), (x - a)^2 + y^2 = r^2 ∧ a = 2 ∧ b = 0 ∧ r^2 = 5 :=
by 
  sorry

end circle_equation_l1088_108875


namespace marcia_average_cost_l1088_108810

theorem marcia_average_cost :
  let price_apples := 2
  let price_bananas := 1
  let price_oranges := 3
  let count_apples := 12
  let count_bananas := 4
  let count_oranges := 4
  let offer_apples_free := count_apples / 10 * 2
  let offer_oranges_free := count_oranges / 3
  let total_apples := count_apples + offer_apples_free
  let total_oranges := count_oranges + offer_oranges_free
  let total_fruits := total_apples + count_bananas + count_oranges
  let cost_apples := price_apples * (count_apples - offer_apples_free)
  let cost_bananas := price_bananas * count_bananas
  let cost_oranges := price_oranges * (count_oranges - offer_oranges_free)
  let total_cost := cost_apples + cost_bananas + cost_oranges
  let average_cost := total_cost / total_fruits
  average_cost = 1.85 :=
  sorry

end marcia_average_cost_l1088_108810


namespace find_minimum_value_of_quadratic_l1088_108821

theorem find_minimum_value_of_quadratic :
  ∀ (x : ℝ), (x = 5/2) -> (∀ y, y = 3 * x ^ 2 - 15 * x + 7 -> ∀ z, z ≥ y) := 
sorry

end find_minimum_value_of_quadratic_l1088_108821


namespace percentage_less_than_a_plus_d_l1088_108884

def symmetric_distribution (a d : ℝ) (p : ℝ) : Prop :=
  p = (68 / 100 : ℝ) ∧ 
  (p / 2) = (34 / 100 : ℝ)

theorem percentage_less_than_a_plus_d (a d : ℝ) 
  (symmetry : symmetric_distribution a d (68 / 100)) : 
  (0.5 + (34 / 100) : ℝ) = (84 / 100 : ℝ) :=
by
  sorry

end percentage_less_than_a_plus_d_l1088_108884


namespace population_growth_proof_l1088_108825

noncomputable def population_growth (P0 : ℕ) (P200 : ℕ) (t : ℕ) (x : ℝ) : Prop :=
  P200 = P0 * (1 + 1 / x)^t

theorem population_growth_proof :
  population_growth 6 1000000 200 16 :=
by
  -- Proof goes here
  sorry

end population_growth_proof_l1088_108825


namespace common_chord_and_length_l1088_108832

-- Define the two circles
def circle1 (x y : ℝ) := x^2 + y^2 + 2*x - 4*y - 5 = 0
def circle2 (x y : ℝ) := x^2 + y^2 + 2*x - 1 = 0

-- The theorem statement with the conditions and expected solutions
theorem common_chord_and_length :
  (∀ x y : ℝ, circle1 x y ∧ circle2 x y → y = -1)
  ∧
  (∃ A B : (ℝ × ℝ), (circle1 A.1 A.2 ∧ circle2 A.1 A.2) ∧ 
                    (circle1 B.1 B.2 ∧ circle2 B.1 B.2) ∧ 
                    (|A.1 - B.1|^2 + |A.2 - B.2|^2 = 4)) :=
by
  sorry

end common_chord_and_length_l1088_108832


namespace maximum_value_of_a_squared_b_l1088_108886

theorem maximum_value_of_a_squared_b {a b : ℝ} (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : a * (a + b) = 27) : 
  a^2 * b ≤ 54 :=
sorry

end maximum_value_of_a_squared_b_l1088_108886


namespace pipe_length_l1088_108897

theorem pipe_length (L x : ℝ) 
  (h1 : 20 = L - x)
  (h2 : 140 = L + 7 * x) : 
  L = 35 := by
  sorry

end pipe_length_l1088_108897


namespace mass_percentage_O_in_Al2_CO3_3_correct_l1088_108895

noncomputable def mass_percentage_O_in_Al2_CO3_3 : ℚ := 
  let mass_O := 9 * 16.00
  let molar_mass_Al2_CO3_3 := (2 * 26.98) + (3 * 12.01) + (9 * 16.00)
  (mass_O / molar_mass_Al2_CO3_3) * 100

theorem mass_percentage_O_in_Al2_CO3_3_correct :
  mass_percentage_O_in_Al2_CO3_3 = 61.54 :=
by
  unfold mass_percentage_O_in_Al2_CO3_3
  sorry

end mass_percentage_O_in_Al2_CO3_3_correct_l1088_108895


namespace rental_cost_l1088_108801

theorem rental_cost (total_cost gallons gas_price mile_cost miles : ℝ)
    (H1 : gallons = 8)
    (H2 : gas_price = 3.50)
    (H3 : mile_cost = 0.50)
    (H4 : miles = 320)
    (H5 : total_cost = 338) :
    total_cost - (gallons * gas_price + miles * mile_cost) = 150 := by
  sorry

end rental_cost_l1088_108801


namespace no_odd_total_given_ratio_l1088_108846

theorem no_odd_total_given_ratio (T : ℕ) (hT1 : 50 < T) (hT2 : T < 150) (hT3 : T % 2 = 1) : 
  ∀ (B : ℕ), T ≠ 8 * B + B / 4 :=
sorry

end no_odd_total_given_ratio_l1088_108846


namespace number_of_players_in_hockey_club_l1088_108816

-- Defining the problem parameters
def cost_of_gloves : ℕ := 6
def cost_of_helmet := cost_of_gloves + 7
def total_cost_per_set := cost_of_gloves + cost_of_helmet
def total_cost_per_player := 2 * total_cost_per_set
def total_expenditure : ℕ := 3120

-- Defining the target number of players
def num_players : ℕ := total_expenditure / total_cost_per_player

theorem number_of_players_in_hockey_club : num_players = 82 := by
  sorry

end number_of_players_in_hockey_club_l1088_108816


namespace smallest_n_for_two_distinct_tuples_l1088_108826

theorem smallest_n_for_two_distinct_tuples : ∃ (n : ℕ), n = 1729 ∧ 
  (∃ (x1 y1 x2 y2 : ℕ), x1 ≠ x2 ∧ y1 ≠ y2 ∧ n = x1^3 + y1^3 ∧ n = x2^3 + y2^3 ∧ 0 < x1 ∧ 0 < y1 ∧ 0 < x2 ∧ 0 < y2) := sorry

end smallest_n_for_two_distinct_tuples_l1088_108826


namespace sqrt_meaningful_l1088_108853

theorem sqrt_meaningful (x : ℝ) : (x - 2 ≥ 0) ↔ (x ≥ 2) := by
  sorry

end sqrt_meaningful_l1088_108853


namespace diamond_expression_calculation_l1088_108822

def diamond (a b : ℚ) : ℚ := a - 1 / b

theorem diamond_expression_calculation :
  (diamond (diamond 2 3) 5) - (diamond 2 (diamond 3 5)) = -37 / 210 :=
by
  sorry

end diamond_expression_calculation_l1088_108822


namespace clock_angle_at_3_40_l1088_108805

noncomputable def hour_hand_angle (h m : ℕ) : ℝ := (h % 12) * 30 + m * 0.5
noncomputable def minute_hand_angle (m : ℕ) : ℝ := m * 6
noncomputable def angle_between_hands (h m : ℕ) : ℝ := 
  let angle := |minute_hand_angle m - hour_hand_angle h m|
  if angle > 180 then 360 - angle else angle

theorem clock_angle_at_3_40 : angle_between_hands 3 40 = 130.0 := 
by
  sorry

end clock_angle_at_3_40_l1088_108805


namespace steiner_ellipse_equation_l1088_108873

theorem steiner_ellipse_equation
  (α β γ : ℝ) 
  (h : α + β + γ = 1) :
  β * γ + α * γ + α * β = 0 := 
sorry

end steiner_ellipse_equation_l1088_108873


namespace difference_is_20_l1088_108865

def x : ℕ := 10

def a : ℕ := 3 * x

def b : ℕ := 20 - x

theorem difference_is_20 : a - b = 20 := 
by 
  sorry

end difference_is_20_l1088_108865


namespace f_20_equals_97_l1088_108889

noncomputable def f_rec (f : ℕ → ℝ) (n : ℕ) := (2 * f n + n) / 2

theorem f_20_equals_97 (f : ℕ → ℝ) (h₁ : f 1 = 2)
    (h₂ : ∀ n : ℕ, f (n + 1) = f_rec f n) : 
    f 20 = 97 :=
sorry

end f_20_equals_97_l1088_108889


namespace problem_correct_l1088_108820

def is_four_digit (n : ℕ) : Prop := n >= 1000 ∧ n < 10000
def is_nat_lt_10 (n : ℕ) : Prop := n < 10
def not_zero (n : ℕ) : Prop := n ≠ 0

structure Matrix4x4 :=
  (a₀₀ a₀₁ a₀₂ a₀₃ : ℕ)
  (a₁₀ a₁₁ a₁₂ a₁₃ : ℕ)
  (a₂₀ a₂₁ a₂₂ a₂₃ : ℕ)
  (a₃₀ a₃₁ a₃₂ a₃₃ : ℕ)

def valid_matrix (M : Matrix4x4) : Prop :=
  -- Each cell must be a natural number less than 10
  is_nat_lt_10 M.a₀₀ ∧ is_nat_lt_10 M.a₀₁ ∧ is_nat_lt_10 M.a₀₂ ∧ is_nat_lt_10 M.a₀₃ ∧
  is_nat_lt_10 M.a₁₀ ∧ is_nat_lt_10 M.a₁₁ ∧ is_nat_lt_10 M.a₁₂ ∧ is_nat_lt_10 M.a₁₃ ∧
  is_nat_lt_10 M.a₂₀ ∧ is_nat_lt_10 M.a₂₁ ∧ is_nat_lt_10 M.a₂₂ ∧ is_nat_lt_10 M.a₂₃ ∧
  is_nat_lt_10 M.a₃₀ ∧ is_nat_lt_10 M.a₃₁ ∧ is_nat_lt_10 M.a₃₂ ∧ is_nat_lt_10 M.a₃₃ ∧

  -- Cells in the same region must contain the same number
  M.a₀₀ = M.a₁₀ ∧ M.a₀₀ = M.a₂₀ ∧ M.a₀₀ = M.a₃₀ ∧
  M.a₂₀ = M.a₂₁ ∧
  M.a₂₂ = M.a₂₃ ∧ M.a₂₂ = M.a₃₂ ∧ M.a₂₂ = M.a₃₃ ∧
  M.a₀₃ = M.a₁₃ ∧
  
  -- Cells in the leftmost column cannot contain the number 0
  not_zero M.a₀₀ ∧ not_zero M.a₁₀ ∧ not_zero M.a₂₀ ∧ not_zero M.a₃₀ ∧

  -- The four-digit number formed by the first row is 2187
  is_four_digit (M.a₀₀ * 1000 + M.a₀₁ * 100 + M.a₀₂ * 10 + M.a₀₃) ∧ 
  (M.a₀₀ * 1000 + M.a₀₁ * 100 + M.a₀₂ * 10 + M.a₀₃ = 2187) ∧
  
  -- The four-digit number formed by the second row is 7387
  is_four_digit (M.a₁₀ * 1000 + M.a₁₁ * 100 + M.a₁₂ * 10 + M.a₁₃) ∧ 
  (M.a₁₀ * 1000 + M.a₁₁ * 100 + M.a₁₂ * 10 + M.a₁₃ = 7387) ∧
  
  -- The four-digit number formed by the third row is 7744
  is_four_digit (M.a₂₀ * 1000 + M.a₂₁ * 100 + M.a₂₂ * 10 + M.a₂₃) ∧ 
  (M.a₂₀ * 1000 + M.a₂₁ * 100 + M.a₂₂ * 10 + M.a₂₃ = 7744) ∧
  
  -- The four-digit number formed by the fourth row is 7844
  is_four_digit (M.a₃₀ * 1000 + M.a₃₁ * 100 + M.a₃₂ * 10 + M.a₃₃) ∧ 
  (M.a₃₀ * 1000 + M.a₃₁ * 100 + M.a₃₂ * 10 + M.a₃₃ = 7844)

noncomputable def problem_solution : Matrix4x4 :=
{ a₀₀ := 2, a₀₁ := 1, a₀₂ := 8, a₀₃ := 7,
  a₁₀ := 7, a₁₁ := 3, a₁₂ := 8, a₁₃ := 7,
  a₂₀ := 7, a₂₁ := 7, a₂₂ := 4, a₂₃ := 4,
  a₃₀ := 7, a₃₁ := 8, a₃₂ := 4, a₃₃ := 4 }

theorem problem_correct : valid_matrix problem_solution :=
by
  -- The proof would go here to show that problem_solution meets valid_matrix
  sorry

end problem_correct_l1088_108820


namespace one_fourth_of_six_point_three_as_fraction_l1088_108851

noncomputable def one_fourth_of_six_point_three_is_simplified : ℚ :=
  6.3 / 4

theorem one_fourth_of_six_point_three_as_fraction :
  one_fourth_of_six_point_three_is_simplified = 63 / 40 :=
by
  sorry

end one_fourth_of_six_point_three_as_fraction_l1088_108851


namespace relationship_among_a_b_c_l1088_108827

theorem relationship_among_a_b_c 
  (a : ℝ) (b : ℝ) (c : ℝ)
  (ha : a = (1 / 2) ^ (3 / 2))
  (hb : b = Real.log pi)
  (hc : c = Real.logb 0.5 (3 / 2)) :
  c < a ∧ a < b :=
by 
  sorry

end relationship_among_a_b_c_l1088_108827


namespace jerry_trips_l1088_108890

-- Define the conditions
def trays_per_trip : Nat := 8
def trays_table1 : Nat := 9
def trays_table2 : Nat := 7

-- Define the proof problem
theorem jerry_trips :
  trays_table1 + trays_table2 = 16 →
  (16 / trays_per_trip) = 2 :=
by
  sorry

end jerry_trips_l1088_108890


namespace multiply_polynomials_l1088_108839

open Polynomial

variable {R : Type*} [CommRing R]

theorem multiply_polynomials (x : R) :
  (x^4 + 6*x^2 + 9) * (x^2 - 3) = x^4 + 6*x^2 :=
  sorry

end multiply_polynomials_l1088_108839


namespace max_sum_integers_differ_by_60_l1088_108870

theorem max_sum_integers_differ_by_60 (b : ℕ) (c : ℕ) (h_diff : 0 < b) (h_sqrt : (Nat.sqrt b : ℝ) + (Nat.sqrt (b + 60) : ℝ) = (Nat.sqrt c : ℝ)) (h_not_square : ¬ ∃ (k : ℕ), k * k = c) :
  ∃ (b : ℕ), b + (b + 60) = 156 := 
sorry

end max_sum_integers_differ_by_60_l1088_108870


namespace circle_passing_through_points_l1088_108880

noncomputable def parabola (x: ℝ) (a b: ℝ) : ℝ :=
  x^2 + a * x + b

theorem circle_passing_through_points (a b α β k: ℝ) :
  parabola 0 a b = b ∧ parabola α a b = 0 ∧ parabola β a b = 0 ∧
  ((0 - (α + β) / 2)^2 + (1 - k)^2 = ((α + β) / 2)^2 + (k - b)^2) →
  b = 1 :=
by
  sorry

end circle_passing_through_points_l1088_108880


namespace range_of_a_for_local_min_l1088_108803

noncomputable def f (a x : ℝ) : ℝ := (x - 2 * a) * (x^2 + a^2 * x + 2 * a^3)

theorem range_of_a_for_local_min :
  (∀ a : ℝ, (∃ δ > 0, ∀ ε ∈ Set.Ioo (-δ) δ, f a ε > f a 0) → a < 0 ∨ a > 2) :=
by
  sorry

end range_of_a_for_local_min_l1088_108803


namespace sum_of_squares_l1088_108818

theorem sum_of_squares (r b s : ℕ) 
  (h1 : 2 * r + 3 * b + s = 80) 
  (h2 : 4 * r + 2 * b + 3 * s = 98) : 
  r^2 + b^2 + s^2 = 485 := 
by {
  sorry
}

end sum_of_squares_l1088_108818


namespace no_five_integer_solutions_divisibility_condition_l1088_108866

variables (k : ℤ) 

-- Definition of equation
def equation (x y : ℤ) : Prop :=
  y^2 - k = x^3

-- Variables to capture the integer solutions
variables (x1 x2 x3 x4 x5 y1 : ℤ)

-- Prove that there do not exist five solutions satisfying the given forms
theorem no_five_integer_solutions :
  ¬(equation k x1 y1 ∧ 
    equation k x2 (y1 - 1) ∧ 
    equation k x3 (y1 - 2) ∧ 
    equation k x4 (y1 - 3) ∧ 
    equation k x5 (y1 - 4)) :=
sorry

-- Prove divisibility condition for the first four solutions
theorem divisibility_condition :
  (equation k x1 y1 ∧ 
   equation k x2 (y1 - 1) ∧ 
   equation k x3 (y1 - 2) ∧ 
   equation k x4 (y1 - 3)) → 
  63 ∣ (k - 17) :=
sorry

end no_five_integer_solutions_divisibility_condition_l1088_108866


namespace max_angle_line_plane_l1088_108861

theorem max_angle_line_plane (θ : ℝ) (h_angle : θ = 72) :
  ∃ φ : ℝ, φ = 90 ∧ (72 ≤ φ ∧ φ ≤ 90) :=
by sorry

end max_angle_line_plane_l1088_108861


namespace solve_quadratic_l1088_108863

theorem solve_quadratic (x : ℝ) (h : (9 / x^2) - (6 / x) + 1 = 0) : 2 / x = 2 / 3 :=
by
  sorry

end solve_quadratic_l1088_108863


namespace num_prime_factors_30_fact_l1088_108843

noncomputable def factorial (n : ℕ) : ℕ :=
  if h : n = 0 then 1 else n * factorial (n - 1)

def is_prime (n : ℕ) : Bool :=
  if h : n ≤ 1 then false else
    let divisors := List.range (n - 2) |>.map (· + 2)
    !divisors.any (· ∣ n)

def primes_upto (n : ℕ) : List ℕ :=
  List.range (n - 1) |>.map (· + 1) |>.filter is_prime

def count_primes_factorial_upto (n : ℕ) : ℕ :=
  (primes_upto n).length

theorem num_prime_factors_30_fact : count_primes_factorial_upto 30 = 10 := sorry

end num_prime_factors_30_fact_l1088_108843


namespace robert_cash_spent_as_percentage_l1088_108892

theorem robert_cash_spent_as_percentage 
  (raw_material_cost : ℤ) (machinery_cost : ℤ) (total_amount : ℤ) 
  (h_raw : raw_material_cost = 100) 
  (h_machinery : machinery_cost = 125) 
  (h_total : total_amount = 250) :
  ((total_amount - (raw_material_cost + machinery_cost)) * 100 / total_amount) = 10 := 
by 
  -- Proof will be filled here
  sorry

end robert_cash_spent_as_percentage_l1088_108892


namespace find_a4_l1088_108841

variable {α : Type*} [Field α] [Inhabited α]

-- Definitions of the geometric sequence conditions
def geometric_sequence_condition1 (a₁ q : α) : Prop :=
  a₁ * (1 + q) = -1

def geometric_sequence_condition2 (a₁ q : α) : Prop :=
  a₁ * (1 - q^2) = -3

-- Definition of the geometric sequence
def geometric_sequence (a₁ q : α) (n : ℕ) : α :=
  a₁ * q^n

-- The theorem to be proven
theorem find_a4 (a₁ q : α) (h₁ : geometric_sequence_condition1 a₁ q) (h₂ : geometric_sequence_condition2 a₁ q) :
  geometric_sequence a₁ q 3 = -8 :=
  sorry

end find_a4_l1088_108841


namespace initial_books_l1088_108819

theorem initial_books (B : ℕ) (h : B + 5 = 7) : B = 2 :=
by sorry

end initial_books_l1088_108819


namespace melissa_driving_time_l1088_108833

theorem melissa_driving_time
  (trips_per_month: ℕ)
  (months_per_year: ℕ)
  (total_hours_per_year: ℕ)
  (total_trips: ℕ)
  (hours_per_trip: ℕ) :
  trips_per_month = 2 ∧
  months_per_year = 12 ∧
  total_hours_per_year = 72 ∧
  total_trips = (trips_per_month * months_per_year) ∧
  hours_per_trip = (total_hours_per_year / total_trips) →
  hours_per_trip = 3 :=
by
  intro h
  obtain ⟨h1, h2, h3, h4, h5⟩ := h
  sorry

end melissa_driving_time_l1088_108833


namespace largest_n_for_divisibility_l1088_108868

theorem largest_n_for_divisibility (n : ℕ) (h : (n + 20) ∣ (n^3 + 1000)) : n ≤ 180 := 
sorry

example : ∃ n : ℕ, (n + 20) ∣ (n^3 + 1000) ∧ n = 180 :=
by
  use 180
  sorry

end largest_n_for_divisibility_l1088_108868


namespace number_of_tickets_l1088_108893

-- Define the given conditions
def initial_premium := 50 -- dollars per month
def premium_increase_accident (initial_premium : ℕ) := initial_premium / 10 -- 10% increase
def premium_increase_ticket := 5 -- dollars per month per ticket
def num_accidents := 1
def new_premium := 70 -- dollars per month

-- Define the target question
theorem number_of_tickets (tickets : ℕ) :
  initial_premium + premium_increase_accident initial_premium * num_accidents + premium_increase_ticket * tickets = new_premium → 
  tickets = 3 :=
by
   sorry

end number_of_tickets_l1088_108893


namespace correct_statement_l1088_108815

theorem correct_statement : 
  (∀ x : ℝ, (x < 0 → x^2 > x)) ∧
  (¬ ∀ x : ℝ, (x^2 > 0 → x > 0)) ∧
  (¬ ∀ x : ℝ, (x^2 > x → x > 0)) ∧
  (¬ ∀ x : ℝ, (x^2 > x → x < 0)) ∧
  (¬ ∀ x : ℝ, (x < 1 → x^2 < x)) :=
by
  sorry

end correct_statement_l1088_108815
