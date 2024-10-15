import Mathlib

namespace NUMINAMATH_GPT_inequality_solution_l1940_194067

theorem inequality_solution :
  {x : ℝ | (x^2 - 1) / (x - 3)^2 ≥ 0} = (Set.Iic (-1) ∪ Set.Ici 1) :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l1940_194067


namespace NUMINAMATH_GPT_inequality_problem_l1940_194074

theorem inequality_problem
  (a b c d : ℝ)
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d)
  (h_sum : a + b + c + d = 1) :
  (a^2 / (1 + a)) + (b^2 / (1 + b)) + (c^2 / (1 + c)) + (d^2 / (1 + d)) ≥ 1 / 5 :=
by
  sorry

end NUMINAMATH_GPT_inequality_problem_l1940_194074


namespace NUMINAMATH_GPT_special_collection_books_l1940_194023

theorem special_collection_books (initial_books loaned_books returned_percent: ℕ) (loaned_books_value: loaned_books = 55) (returned_percent_value: returned_percent = 80) (initial_books_value: initial_books = 75) :
  initial_books - (loaned_books - (returned_percent * loaned_books / 100)) = 64 := by
  sorry

end NUMINAMATH_GPT_special_collection_books_l1940_194023


namespace NUMINAMATH_GPT_arithmetic_geometric_ratio_l1940_194028

variables {a : ℕ → ℝ} {d : ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = d

def is_geometric_sequence (a b c : ℝ) : Prop :=
  b^2 = a * c

theorem arithmetic_geometric_ratio {a : ℕ → ℝ} {d : ℝ} (h1 : is_arithmetic_sequence a d)
  (h2 : a 9 ≠ a 3) (h3 : is_geometric_sequence (a 1) (a 3) (a 9)):
  (a 2 + a 4 + a 10) / (a 1 + a 3 + a 9) = 16 / 13 :=
sorry

end NUMINAMATH_GPT_arithmetic_geometric_ratio_l1940_194028


namespace NUMINAMATH_GPT_number_of_rows_of_desks_is_8_l1940_194021

-- Definitions for the conditions
def first_row_desks : ℕ := 10
def desks_increment : ℕ := 2
def total_desks : ℕ := 136

-- Definition for the sum of an arithmetic series
def arithmetic_series_sum (n a1 d : ℕ) : ℕ :=
  n * (2 * a1 + (n - 1) * d) / 2

-- The proof problem statement
theorem number_of_rows_of_desks_is_8 :
  ∃ n : ℕ, arithmetic_series_sum n first_row_desks desks_increment = total_desks ∧ n = 8 :=
by
  sorry

end NUMINAMATH_GPT_number_of_rows_of_desks_is_8_l1940_194021


namespace NUMINAMATH_GPT_smallest_vertical_distance_between_graphs_l1940_194053

noncomputable def f (x : ℝ) : ℝ := abs x
noncomputable def g (x : ℝ) : ℝ := -x^2 + 2 * x + 3

theorem smallest_vertical_distance_between_graphs :
  ∃ (d : ℝ), (∀ (x : ℝ), |f x - g x| ≥ d) ∧ (∀ (ε : ℝ), ε > 0 → ∃ (x : ℝ), |f x - g x| < d + ε) ∧ d = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_smallest_vertical_distance_between_graphs_l1940_194053


namespace NUMINAMATH_GPT_triangle_right_angled_l1940_194056

-- Define the variables and the condition of the problem
variables {a b c : ℝ}

-- Given condition of the problem
def triangle_condition (a b c : ℝ) : Prop :=
  2 * (a ^ 8 + b ^ 8 + c ^ 8) = (a ^ 4 + b ^ 4 + c ^ 4) ^ 2

-- The theorem to prove the triangle is right-angled
theorem triangle_right_angled (h : triangle_condition a b c) : a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2 :=
sorry

end NUMINAMATH_GPT_triangle_right_angled_l1940_194056


namespace NUMINAMATH_GPT_tangent_circle_given_r_l1940_194072

theorem tangent_circle_given_r (r : ℝ) (h_pos : 0 < r)
    (h_tangent : ∀ x y : ℝ, (2 * x + y = r) → (x^2 + y^2 = 2 * r))
  : r = 10 :=
sorry

end NUMINAMATH_GPT_tangent_circle_given_r_l1940_194072


namespace NUMINAMATH_GPT_total_volume_of_four_cubes_l1940_194017

theorem total_volume_of_four_cubes (s : ℝ) (h_s : s = 5) : 4 * s^3 = 500 :=
by
  sorry

end NUMINAMATH_GPT_total_volume_of_four_cubes_l1940_194017


namespace NUMINAMATH_GPT_decomposition_x_pqr_l1940_194082

-- Definitions of vectors x, p, q, r
def x : ℝ := sorry
def p : ℝ := sorry
def q : ℝ := sorry
def r : ℝ := sorry

-- The linear combination we want to prove
theorem decomposition_x_pqr : 
  (x = -1 • p + 4 • q + 3 • r) :=
sorry

end NUMINAMATH_GPT_decomposition_x_pqr_l1940_194082


namespace NUMINAMATH_GPT_find_x_l1940_194022

theorem find_x (x : ℝ) : 
  (1 + x) * 0.20 = x * 0.4 → x = 1 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_find_x_l1940_194022


namespace NUMINAMATH_GPT_total_surface_area_l1940_194026

theorem total_surface_area (a b c : ℝ) 
  (h1 : a + b + c = 45) 
  (h2 : a^2 + b^2 + c^2 = 625) : 
  2 * (a * b + b * c + c * a) = 1400 :=
sorry

end NUMINAMATH_GPT_total_surface_area_l1940_194026


namespace NUMINAMATH_GPT_real_solutions_of_equation_l1940_194052

theorem real_solutions_of_equation (x : ℝ) :
  (1 / ((x - 1) * (x - 3)) + 1 / ((x - 3) * (x - 5)) + 1 / ((x - 5) * (x - 7)) = 1 / 12) ↔ (x = 13 ∨ x = -5) :=
by
  sorry

end NUMINAMATH_GPT_real_solutions_of_equation_l1940_194052


namespace NUMINAMATH_GPT_calculate_number_of_models_l1940_194096

-- Define the constants and conditions
def time_per_set : ℕ := 2  -- time per set in minutes
def sets_bathing_suits : ℕ := 2  -- number of bathing suit sets each model wears
def sets_evening_wear : ℕ := 3  -- number of evening wear sets each model wears
def total_show_time : ℕ := 60  -- total show time in minutes

-- Calculate the total time each model takes
def model_time : ℕ := 
  (sets_bathing_suits + sets_evening_wear) * time_per_set

-- Proof problem statement
theorem calculate_number_of_models : 
  (total_show_time / model_time) = 6 := by
  sorry

end NUMINAMATH_GPT_calculate_number_of_models_l1940_194096


namespace NUMINAMATH_GPT_garden_width_is_14_l1940_194046

theorem garden_width_is_14 (w : ℝ) (h1 : ∃ (l : ℝ), l = 3 * w ∧ l * w = 588) : w = 14 :=
sorry

end NUMINAMATH_GPT_garden_width_is_14_l1940_194046


namespace NUMINAMATH_GPT_Nell_initial_cards_l1940_194095

theorem Nell_initial_cards 
  (cards_given : ℕ)
  (cards_left : ℕ)
  (cards_given_eq : cards_given = 301)
  (cards_left_eq : cards_left = 154) :
  cards_given + cards_left = 455 := by
sorry

end NUMINAMATH_GPT_Nell_initial_cards_l1940_194095


namespace NUMINAMATH_GPT_total_books_read_l1940_194000

-- Definitions based on the conditions
def books_per_month : ℕ := 4
def months_per_year : ℕ := 12
def books_per_year_per_student : ℕ := books_per_month * months_per_year

variables (c s : ℕ)

-- Main theorem statement
theorem total_books_read (c s : ℕ) : 
  (books_per_year_per_student * c * s) = 48 * c * s :=
by
  sorry

end NUMINAMATH_GPT_total_books_read_l1940_194000


namespace NUMINAMATH_GPT_tan_product_30_60_l1940_194078

theorem tan_product_30_60 : 
  (1 + Real.tan (30 * Real.pi / 180)) * (1 + Real.tan (60 * Real.pi / 180)) = 2 + (4 * Real.sqrt 3) / 3 := 
  sorry

end NUMINAMATH_GPT_tan_product_30_60_l1940_194078


namespace NUMINAMATH_GPT_problems_left_to_grade_l1940_194003

-- Definitions based on provided conditions
def problems_per_worksheet : ℕ := 4
def total_worksheets : ℕ := 16
def graded_worksheets : ℕ := 8

-- The statement for the required proof with the correct answer included
theorem problems_left_to_grade : 4 * (16 - 8) = 32 := by
  sorry

end NUMINAMATH_GPT_problems_left_to_grade_l1940_194003


namespace NUMINAMATH_GPT_monotonic_intervals_max_value_of_k_l1940_194002

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - a * x - 2
noncomputable def f_prime (x a : ℝ) : ℝ := Real.exp x - a

theorem monotonic_intervals (a : ℝ) :
  (a ≤ 0 → ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ a < f x₂ a) ∧
  (a > 0 → ∀ x₁ x₂ : ℝ,
    x₁ < x₂ → (x₁ < Real.log a → f x₁ a > f x₂ a) ∧ (x₁ > Real.log a → f x₁ a < f x₂ a)) :=
sorry

theorem max_value_of_k (x : ℝ) (k : ℤ) (a : ℝ) (h_a : a = 1)
  (h : ∀ x > 0, (x - k) * f_prime x a + x + 1 > 0) :
  k ≤ 2 :=
sorry

end NUMINAMATH_GPT_monotonic_intervals_max_value_of_k_l1940_194002


namespace NUMINAMATH_GPT_carrie_is_left_with_50_l1940_194084

-- Definitions for the conditions given in the problem
def amount_given : ℕ := 91
def cost_of_sweater : ℕ := 24
def cost_of_tshirt : ℕ := 6
def cost_of_shoes : ℕ := 11

-- Definition of the total amount spent
def total_spent : ℕ := cost_of_sweater + cost_of_tshirt + cost_of_shoes

-- Definition of the amount left
def amount_left : ℕ := amount_given - total_spent

-- The theorem we want to prove
theorem carrie_is_left_with_50 : amount_left = 50 :=
by
  have h1 : amount_given = 91 := rfl
  have h2 : total_spent = 41 := rfl
  have h3 : amount_left = 50 := rfl
  exact rfl

end NUMINAMATH_GPT_carrie_is_left_with_50_l1940_194084


namespace NUMINAMATH_GPT_problem1_problem2_l1940_194042

variable (a b : ℝ)

-- Proof problem for Question 1
theorem problem1 : 2 * a * (a^2 - 3 * a - 1) = 2 * a^3 - 6 * a^2 - 2 * a :=
by sorry

-- Proof problem for Question 2
theorem problem2 : (a^2 * b - 2 * a * b^2 + b^3) / b - (a + b)^2 = -4 * a * b :=
by sorry

end NUMINAMATH_GPT_problem1_problem2_l1940_194042


namespace NUMINAMATH_GPT_lawn_area_l1940_194009

theorem lawn_area (s l : ℕ) (hs: 5 * s = 10) (hl: 5 * l = 50) (hposts: 2 * (s + l) = 24) (hlen: l + 1 = 3 * (s + 1)) :
  s * l = 500 :=
by {
  sorry
}

end NUMINAMATH_GPT_lawn_area_l1940_194009


namespace NUMINAMATH_GPT_fraction_from_tips_l1940_194016

-- Define the waiter's salary and the conditions given in the problem
variables (S : ℕ) -- S is natural assuming salary is a non-negative integer
def tips := (4/5 : ℚ) * S
def bonus := 2 * (1/10 : ℚ) * S
def total_income := S + tips S + bonus S

-- The theorem to be proven
theorem fraction_from_tips (S : ℕ) :
  (tips S / total_income S) = (2/5 : ℚ) :=
sorry

end NUMINAMATH_GPT_fraction_from_tips_l1940_194016


namespace NUMINAMATH_GPT_roger_collected_nickels_l1940_194088

theorem roger_collected_nickels 
  (N : ℕ)
  (initial_pennies : ℕ := 42) 
  (initial_dimes : ℕ := 15)
  (donated_coins : ℕ := 66)
  (left_coins : ℕ := 27)
  (h_total_coins_initial : initial_pennies + N + initial_dimes - donated_coins = left_coins) :
  N = 36 := 
sorry

end NUMINAMATH_GPT_roger_collected_nickels_l1940_194088


namespace NUMINAMATH_GPT_michael_points_scored_l1940_194045

theorem michael_points_scored (team_points : ℕ) (other_players : ℕ) (average_points : ℕ) (michael_points : ℕ) :
  team_points = 72 → other_players = 8 → average_points = 9 → 
  michael_points = team_points - other_players * average_points → michael_points = 36 :=
by
  intro h_team_points h_other_players h_average_points h_calculation
  -- skip the actual proof for now
  sorry

end NUMINAMATH_GPT_michael_points_scored_l1940_194045


namespace NUMINAMATH_GPT_solve_for_n_l1940_194073

theorem solve_for_n : 
  (∃ n : ℤ, (1 / (n + 2) + 2 / (n + 2) + (n + 1) / (n + 2) = 3)) ↔ n = -1 :=
sorry

end NUMINAMATH_GPT_solve_for_n_l1940_194073


namespace NUMINAMATH_GPT_anna_age_when_married_l1940_194057

-- Define constants for the conditions
def j_married : ℕ := 22
def m : ℕ := 30
def combined_age_today : ℕ := 5 * j_married
def j_current : ℕ := j_married + m

-- Define Anna's current age based on the combined age today and Josh's current age
def a_current : ℕ := combined_age_today - j_current

-- Define Anna's age when married
def a_married : ℕ := a_current - m

-- Statement of the theorem to be proved
theorem anna_age_when_married : a_married = 28 :=
by
  sorry

end NUMINAMATH_GPT_anna_age_when_married_l1940_194057


namespace NUMINAMATH_GPT_endpoint_sum_l1940_194055

theorem endpoint_sum
  (x y : ℤ)
  (H_midpoint_x : (x + 15) / 2 = 10)
  (H_midpoint_y : (y - 8) / 2 = -3) :
  x + y = 7 :=
sorry

end NUMINAMATH_GPT_endpoint_sum_l1940_194055


namespace NUMINAMATH_GPT_problem_l1940_194032

variable (R S : Prop)

theorem problem (h1 : R → S) :
  ((¬S → ¬R) ∧ (¬R ∨ S)) :=
by
  sorry

end NUMINAMATH_GPT_problem_l1940_194032


namespace NUMINAMATH_GPT_arrangements_three_balls_four_boxes_l1940_194019

theorem arrangements_three_balls_four_boxes : 
  ∃ (f : Fin 4 → Fin 4), Function.Injective f :=
sorry

end NUMINAMATH_GPT_arrangements_three_balls_four_boxes_l1940_194019


namespace NUMINAMATH_GPT_m_leq_neg_one_l1940_194008

theorem m_leq_neg_one (m : ℝ) :
    (∀ x : ℝ, 2^(-x) + m > 0 → x ≤ 0) → m ≤ -1 :=
by
  sorry

end NUMINAMATH_GPT_m_leq_neg_one_l1940_194008


namespace NUMINAMATH_GPT_function_characterization_l1940_194049
noncomputable def f : ℕ → ℕ := sorry

theorem function_characterization (h : ∀ m n : ℕ, m^2 + f n ∣ m * f m + n) : 
  ∀ n : ℕ, f n = n :=
by
  intro n
  sorry

end NUMINAMATH_GPT_function_characterization_l1940_194049


namespace NUMINAMATH_GPT_problem_proof_l1940_194020

def P : Set ℝ := {x | x ≤ 3}

theorem problem_proof : {-1} ⊆ P := 
sorry

end NUMINAMATH_GPT_problem_proof_l1940_194020


namespace NUMINAMATH_GPT_train_average_speed_l1940_194035

open Real -- Assuming all required real number operations 

noncomputable def average_speed (distances : List ℝ) (times : List ℝ) : ℝ := 
  let total_distance := distances.sum
  let total_time := times.sum
  total_distance / total_time

theorem train_average_speed :
  average_speed [125, 270] [2.5, 3] = 71.82 := 
by 
  -- Details of the actual proof steps are omitted
  sorry

end NUMINAMATH_GPT_train_average_speed_l1940_194035


namespace NUMINAMATH_GPT_num_people_second_hour_l1940_194044

theorem num_people_second_hour 
  (n1_in n2_in n1_left n2_left : ℕ) 
  (rem_hour1 rem_hour2 : ℕ)
  (h1 : n1_in = 94)
  (h2 : n1_left = 27)
  (h3 : n2_left = 9)
  (h4 : rem_hour2 = 76)
  (h5 : rem_hour1 = n1_in - n1_left)
  (h6 : rem_hour2 = rem_hour1 + n2_in - n2_left) :
  n2_in = 18 := 
  by 
  sorry

end NUMINAMATH_GPT_num_people_second_hour_l1940_194044


namespace NUMINAMATH_GPT_common_ratio_is_two_l1940_194098

-- Define the geometric sequence
def geom_seq (a_1 : ℝ) (q : ℝ) (n : ℕ) : ℝ := a_1 * q^n

-- Define the conditions
variables (a_1 q : ℝ)
variables (h_inc : 1 < q) (h_pos : 0 < a_1)
variables (h_seq : ∀ n : ℕ, 2 * (geom_seq a_1 q n + geom_seq a_1 q (n+2)) = 5 * geom_seq a_1 q (n+1))

-- Statement to prove
theorem common_ratio_is_two : q = 2 :=
by
  sorry

end NUMINAMATH_GPT_common_ratio_is_two_l1940_194098


namespace NUMINAMATH_GPT_perimeter_of_shaded_shape_l1940_194058

noncomputable def shaded_perimeter (x : ℝ) : ℝ := 
  let l := 18 - 2 * x
  3 * l

theorem perimeter_of_shaded_shape (x : ℝ) (hx : x > 0) (h_sectors : 2 * x + (18 - 2 * x) = 18) : 
  shaded_perimeter x = 54 := 
by
  rw [shaded_perimeter]
  rw [← h_sectors]
  simp
  sorry

end NUMINAMATH_GPT_perimeter_of_shaded_shape_l1940_194058


namespace NUMINAMATH_GPT_Alexis_mangoes_l1940_194047

-- Define the variables for the number of mangoes each person has.
variable (A D Ash : ℕ)

-- Conditions given in the problem.
axiom h1 : A = 4 * (D + Ash)
axiom h2 : A + D + Ash = 75

-- The proof goal.
theorem Alexis_mangoes : A = 60 :=
sorry

end NUMINAMATH_GPT_Alexis_mangoes_l1940_194047


namespace NUMINAMATH_GPT_nat_numbers_square_minus_one_power_of_prime_l1940_194089

def is_power_of_prime (x : ℕ) : Prop :=
  ∃ (p : ℕ), Nat.Prime p ∧ ∃ (k : ℕ), x = p ^ k

theorem nat_numbers_square_minus_one_power_of_prime (n : ℕ) (hn : 1 ≤ n) :
  is_power_of_prime (n ^ 2 - 1) ↔ (n = 2 ∨ n = 3) := by
  sorry

end NUMINAMATH_GPT_nat_numbers_square_minus_one_power_of_prime_l1940_194089


namespace NUMINAMATH_GPT_some_number_proof_l1940_194030

def g (n : ℕ) : ℕ :=
  if n < 3 then 1 else 
  if n % 2 = 0 then g (n - 1) else 
    g (n - 2) * n

theorem some_number_proof : g 106 - g 103 = 105 :=
by sorry

end NUMINAMATH_GPT_some_number_proof_l1940_194030


namespace NUMINAMATH_GPT_find_number_l1940_194068

theorem find_number (x : ℝ) :
  (1.5 * 1265) / x = 271.07142857142856 → x = 7 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_number_l1940_194068


namespace NUMINAMATH_GPT_friends_reach_destinations_l1940_194077

noncomputable def travel_times (d : ℕ) := 
  let walking_speed := 6
  let cycling_speed := 18
  let meet_time := d / (walking_speed + cycling_speed)
  let remaining_time := d / cycling_speed
  let total_time_A := meet_time + (d - cycling_speed * meet_time) / walking_speed
  let total_time_B := (cycling_speed * meet_time) / walking_speed + (d - cycling_speed * meet_time) / walking_speed
  let total_time_C := remaining_time + meet_time
  (total_time_A, total_time_B, total_time_C)

theorem friends_reach_destinations (d : ℕ) (d_eq_24 : d = 24) : 
  let (total_time_A, total_time_B, total_time_C) := travel_times d
  total_time_A ≤ 160 / 60 ∧ total_time_B ≤ 160 / 60 ∧ total_time_C ≤ 160 / 60 :=
by 
  sorry

end NUMINAMATH_GPT_friends_reach_destinations_l1940_194077


namespace NUMINAMATH_GPT_distance_symmetric_reflection_l1940_194005

theorem distance_symmetric_reflection (x : ℝ) (y : ℝ) (B : (ℝ × ℝ)) 
  (hB : B = (-1, 4)) (A : (ℝ × ℝ)) (hA : A = (x, -y)) : 
  dist A B = 8 :=
by
  sorry

end NUMINAMATH_GPT_distance_symmetric_reflection_l1940_194005


namespace NUMINAMATH_GPT_shaded_fraction_eighth_triangle_l1940_194091

def triangular_number (n : Nat) : Nat := n * (n + 1) / 2
def square_number (n : Nat) : Nat := n * n

theorem shaded_fraction_eighth_triangle :
  let shaded_triangles := triangular_number 7
  let total_triangles := square_number 8
  shaded_triangles / total_triangles = 7 / 16 := 
by
  sorry

end NUMINAMATH_GPT_shaded_fraction_eighth_triangle_l1940_194091


namespace NUMINAMATH_GPT_polynomial_root_sum_l1940_194062

theorem polynomial_root_sum : 
  ∀ (r1 r2 r3 r4 : ℝ), 
  (r1^4 - r1 - 504 = 0) ∧ 
  (r2^4 - r2 - 504 = 0) ∧ 
  (r3^4 - r3 - 504 = 0) ∧ 
  (r4^4 - r4 - 504 = 0) → 
  r1^4 + r2^4 + r3^4 + r4^4 = 2016 := by
sorry

end NUMINAMATH_GPT_polynomial_root_sum_l1940_194062


namespace NUMINAMATH_GPT_tan_theta_of_obtuse_angle_l1940_194081

noncomputable def theta_expression (θ : Real) : Complex :=
  Complex.mk (3 * Real.sin θ) (Real.cos θ)

theorem tan_theta_of_obtuse_angle {θ : Real} (h_modulus : Complex.abs (theta_expression θ) = Real.sqrt 5) 
  (h_obtuse : π / 2 < θ ∧ θ < π) : Real.tan θ = -1 := 
  sorry

end NUMINAMATH_GPT_tan_theta_of_obtuse_angle_l1940_194081


namespace NUMINAMATH_GPT_problem_l1940_194036

noncomputable def f (x : ℝ) : ℝ := ((x + 1) ^ 2 + Real.sin x) / (x ^ 2 + 1)

noncomputable def f' (x : ℝ) : ℝ := ((2 + Real.cos x) * (x ^ 2 + 1) - (2 * x + Real.sin x) * (2 * x)) / (x ^ 2 + 1) ^ 2

theorem problem : f 2016 + f' 2016 + f (-2016) - f' (-2016) = 2 := by
  sorry

end NUMINAMATH_GPT_problem_l1940_194036


namespace NUMINAMATH_GPT_surface_area_of_cube_l1940_194063

theorem surface_area_of_cube (edge : ℝ) (h : edge = 5) : 6 * (edge * edge) = 150 := by
  have h_square : edge * edge = 25 := by
    rw [h]
    norm_num
  rw [h_square]
  norm_num

end NUMINAMATH_GPT_surface_area_of_cube_l1940_194063


namespace NUMINAMATH_GPT_factorization_property_l1940_194001

theorem factorization_property (a b : ℤ) (h1 : 25 * x ^ 2 - 160 * x - 144 = (5 * x + a) * (5 * x + b)) 
    (h2 : a + b = -32) (h3 : a * b = -144) : 
    a + 2 * b = -68 := 
sorry

end NUMINAMATH_GPT_factorization_property_l1940_194001


namespace NUMINAMATH_GPT_power_of_sqrt2_minus_1_l1940_194086

noncomputable def a (n : ℕ) : ℝ := (Real.sqrt 2 - 1) ^ n
noncomputable def b (n : ℕ) : ℝ := (Real.sqrt 2 + 1) ^ n
noncomputable def c (n : ℕ) : ℝ := (b n + a n) / 2
noncomputable def d (n : ℕ) : ℝ := (b n - a n) / 2

theorem power_of_sqrt2_minus_1 (n : ℕ) : a n = Real.sqrt (d n ^ 2 + 1) - Real.sqrt (d n ^ 2) :=
by
  sorry

end NUMINAMATH_GPT_power_of_sqrt2_minus_1_l1940_194086


namespace NUMINAMATH_GPT_find_f_l1940_194097

-- Define the function space and conditions
def func (f : ℕ+ → ℝ) :=
  (∀ m n : ℕ+, f (m * n) = f m + f n) ∧
  (∀ n : ℕ+, f (n + 1) ≥ f n)

-- Define the theorem statement
theorem find_f (f : ℕ+ → ℝ) (hf : func f) : ∀ n : ℕ+, f n = 0 :=
sorry

end NUMINAMATH_GPT_find_f_l1940_194097


namespace NUMINAMATH_GPT_probability_at_least_one_prize_proof_l1940_194027

noncomputable def probability_at_least_one_wins_prize
  (total_tickets : ℕ) (prize_tickets : ℕ) (people : ℕ) :
  ℚ :=
1 - ((@Nat.choose (total_tickets - prize_tickets) people) /
      (@Nat.choose total_tickets people))

theorem probability_at_least_one_prize_proof :
  probability_at_least_one_wins_prize 10 3 5 = 11 / 12 :=
by
  sorry

end NUMINAMATH_GPT_probability_at_least_one_prize_proof_l1940_194027


namespace NUMINAMATH_GPT_victor_percentage_of_marks_l1940_194065

theorem victor_percentage_of_marks (marks_obtained : ℝ) (maximum_marks : ℝ) (h1 : marks_obtained = 285) (h2 : maximum_marks = 300) : 
  (marks_obtained / maximum_marks) * 100 = 95 :=
by
  sorry

end NUMINAMATH_GPT_victor_percentage_of_marks_l1940_194065


namespace NUMINAMATH_GPT_prob_both_A_B_prob_exactly_one_l1940_194051

def prob_A : ℝ := 0.8
def prob_not_B : ℝ := 0.1
def prob_B : ℝ := 1 - prob_not_B

lemma prob_independent (a b : Prop) : Prop := -- Placeholder for actual independence definition
sorry

-- Given conditions
variables (P_A : ℝ := prob_A) (P_not_B : ℝ := prob_not_B) (P_B : ℝ := prob_B) (indep : ∀ A B, prob_independent A B)

-- Questions translated to Lean statements
theorem prob_both_A_B : P_A * P_B = 0.72 := sorry

theorem prob_exactly_one : (P_A * P_not_B) + ((1 - P_A) * P_B) = 0.26 := sorry

end NUMINAMATH_GPT_prob_both_A_B_prob_exactly_one_l1940_194051


namespace NUMINAMATH_GPT_common_card_cost_l1940_194070

def totalDeckCost (rareCost uncommonCost commonCost numRares numUncommons numCommons : ℝ) : ℝ :=
  (numRares * rareCost) + (numUncommons * uncommonCost) + (numCommons * commonCost)

theorem common_card_cost (numRares numUncommons numCommons : ℝ) (rareCost uncommonCost totalCost : ℝ) : 
  numRares = 19 → numUncommons = 11 → numCommons = 30 → 
  rareCost = 1 → uncommonCost = 0.5 → totalCost = 32 → 
  commonCost = 0.25 :=
by 
  intros 
  sorry

end NUMINAMATH_GPT_common_card_cost_l1940_194070


namespace NUMINAMATH_GPT_geoff_tuesday_multiple_l1940_194007

variable (monday_spending : ℝ) (tuesday_multiple : ℝ) (total_spending : ℝ)

-- Given conditions
def geoff_conditions (monday_spending tuesday_multiple total_spending : ℝ) : Prop :=
  monday_spending = 60 ∧
  (tuesday_multiple * monday_spending) + (5 * monday_spending) + monday_spending = total_spending ∧
  total_spending = 600

-- Proof goal
theorem geoff_tuesday_multiple (monday_spending tuesday_multiple total_spending : ℝ)
  (h : geoff_conditions monday_spending tuesday_multiple total_spending) : 
  tuesday_multiple = 4 :=
by
  sorry

end NUMINAMATH_GPT_geoff_tuesday_multiple_l1940_194007


namespace NUMINAMATH_GPT_find_x_l1940_194041

def infinite_sqrt (d : ℝ) : ℝ := sorry -- A placeholder since infinite nesting is non-trivial

def bowtie (c d : ℝ) : ℝ := c - infinite_sqrt d

theorem find_x (x : ℝ) (h : bowtie 7 x = 3) : x = 20 :=
sorry

end NUMINAMATH_GPT_find_x_l1940_194041


namespace NUMINAMATH_GPT_tan_alpha_not_unique_l1940_194033

theorem tan_alpha_not_unique (α : ℝ) (h1 : α > 0) (h2 : α < Real.pi) (h3 : (Real.sin α)^2 + Real.cos (2 * α) = 1) :
  ¬(∃ t : ℝ, Real.tan α = t) :=
by
  sorry

end NUMINAMATH_GPT_tan_alpha_not_unique_l1940_194033


namespace NUMINAMATH_GPT_negation_of_exists_l1940_194087

theorem negation_of_exists (p : Prop) : 
  (∃ (x₀ : ℝ), x₀ > 0 ∧ |x₀| ≤ 2018) ↔ 
  ¬(∀ (x : ℝ), x > 0 → |x| > 2018) :=
by sorry

end NUMINAMATH_GPT_negation_of_exists_l1940_194087


namespace NUMINAMATH_GPT_solve_for_a_l1940_194024

def i := Complex.I

theorem solve_for_a (a : ℝ) (h : (2 + i) / (1 + a * i) = i) : a = -2 := 
by 
  sorry

end NUMINAMATH_GPT_solve_for_a_l1940_194024


namespace NUMINAMATH_GPT_calories_in_250g_mixed_drink_l1940_194012

def calories_in_mixed_drink (grams_cranberry : ℕ) (grams_honey : ℕ) (grams_water : ℕ)
  (calories_per_100g_cranberry : ℕ) (calories_per_100g_honey : ℕ) (calories_per_100g_water : ℕ)
  (total_grams : ℕ) (portion_grams : ℕ) : ℚ :=
  ((grams_cranberry * calories_per_100g_cranberry + grams_honey * calories_per_100g_honey + grams_water * calories_per_100g_water) : ℚ)
  / (total_grams * portion_grams)

theorem calories_in_250g_mixed_drink :
  calories_in_mixed_drink 150 50 300 30 304 0 100 250 = 98.5 := by
  -- The proof will involve arithmetic operations
  sorry

end NUMINAMATH_GPT_calories_in_250g_mixed_drink_l1940_194012


namespace NUMINAMATH_GPT_bus_stop_time_l1940_194083

theorem bus_stop_time (speed_excl_stops speed_incl_stops : ℝ) (h1 : speed_excl_stops = 50) (h2 : speed_incl_stops = 45) : (60 * ((speed_excl_stops - speed_incl_stops) / speed_excl_stops)) = 6 := 
by
  sorry

end NUMINAMATH_GPT_bus_stop_time_l1940_194083


namespace NUMINAMATH_GPT_usual_time_to_school_l1940_194080

variables (R T : ℝ)

theorem usual_time_to_school (h₁ : T > 0) (h₂ : R > 0) (h₃ : R / T = (5 / 4 * R) / (T - 4)) :
  T = 20 :=
by
  sorry

end NUMINAMATH_GPT_usual_time_to_school_l1940_194080


namespace NUMINAMATH_GPT_fraction_equals_i_l1940_194039

theorem fraction_equals_i (m n : ℝ) (i : ℂ) (h : i * i = -1) (h_cond : m * (1 + i) = (11 + n * i)) :
  (m + n * i) / (m - n * i) = i :=
sorry

end NUMINAMATH_GPT_fraction_equals_i_l1940_194039


namespace NUMINAMATH_GPT_possible_third_side_l1940_194092

theorem possible_third_side (x : ℝ) : (3 + 4 > x) ∧ (abs (4 - 3) < x) → (x = 2) :=
by 
  sorry

end NUMINAMATH_GPT_possible_third_side_l1940_194092


namespace NUMINAMATH_GPT_doubled_radius_and_arc_length_invariant_l1940_194064

theorem doubled_radius_and_arc_length_invariant (r l : ℝ) : (l / r) = (2 * l / (2 * r)) :=
by
  sorry

end NUMINAMATH_GPT_doubled_radius_and_arc_length_invariant_l1940_194064


namespace NUMINAMATH_GPT_chooseOneFromEachCategory_chooseTwoDifferentTypes_l1940_194034

-- Define the number of different paintings in each category
def traditionalChinesePaintings : ℕ := 5
def oilPaintings : ℕ := 2
def watercolorPaintings : ℕ := 7

-- Part (1): Prove that the number of ways to choose one painting from each category is 70
theorem chooseOneFromEachCategory : traditionalChinesePaintings * oilPaintings * watercolorPaintings = 70 := by
  sorry

-- Part (2): Prove that the number of ways to choose two paintings of different types is 59
theorem chooseTwoDifferentTypes :
  (traditionalChinesePaintings * oilPaintings) + 
  (traditionalChinesePaintings * watercolorPaintings) + 
  (oilPaintings * watercolorPaintings) = 59 := by
  sorry

end NUMINAMATH_GPT_chooseOneFromEachCategory_chooseTwoDifferentTypes_l1940_194034


namespace NUMINAMATH_GPT_max_value_sin_sin2x_l1940_194085

open Real

/-- Given x is an acute angle, find the maximum value of the function y = sin x * sin (2 * x). -/
theorem max_value_sin_sin2x (x : ℝ) (hx : 0 < x ∧ x < π / 2) :
    ∃ max_y : ℝ, ∀ y : ℝ, y = sin x * sin (2 * x) -> y ≤ max_y ∧ max_y = 4 * sqrt 3 / 9 :=
by
  -- To be completed
  sorry

end NUMINAMATH_GPT_max_value_sin_sin2x_l1940_194085


namespace NUMINAMATH_GPT_calculate_expression_l1940_194054

theorem calculate_expression : 8 / 2 - 3 - 12 + 3 * (5^2 - 4) = 52 := 
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l1940_194054


namespace NUMINAMATH_GPT_find_ordered_pair_l1940_194038

theorem find_ordered_pair (x y : ℚ) 
  (h1 : 7 * x - 30 * y = 3) 
  (h2 : 3 * y - x = 5) : 
  x = -53 / 3 ∧ y = -38 / 9 :=
sorry

end NUMINAMATH_GPT_find_ordered_pair_l1940_194038


namespace NUMINAMATH_GPT_binary_computation_l1940_194006

theorem binary_computation :
  (0b101101 * 0b10101 + 0b1010 / 0b10) = 0b110111100000 := by
  sorry

end NUMINAMATH_GPT_binary_computation_l1940_194006


namespace NUMINAMATH_GPT_problem_1_problem_2_l1940_194015

open Real

def vec_parallel (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 = u.2 * v.1

def vec_perpendicular (u v : ℝ × ℝ) : Prop :=
  u.1 * v.1 + u.2 * v.2 = 0

theorem problem_1 (k : ℝ) : 
  let a := (3, 4)
  let b := (2, k)
  (vec_parallel (a.1 + 2 * b.1, a.2 + 2 * b.2) (a.1 - b.1, a.2 - b.2)) →
  k = 8 / 3 := sorry

theorem problem_2 (k : ℝ) : 
  let a := (3, 4)
  let b := (2, k)
  (vec_perpendicular (a.1 + b.1, a.2 + b.2) (a.1 - b.1, a.2 - b.2)) →
  k = sqrt 21 ∨ k = - sqrt 21 := sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1940_194015


namespace NUMINAMATH_GPT_least_possible_value_of_smallest_integer_l1940_194071

theorem least_possible_value_of_smallest_integer :
  ∀ (A B C D : ℤ), A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
  (A + B + C + D) / 4 = 76 ∧ D = 90 →
  A = 37 :=
by
  sorry

end NUMINAMATH_GPT_least_possible_value_of_smallest_integer_l1940_194071


namespace NUMINAMATH_GPT_compare_abc_l1940_194025

noncomputable def a : ℝ := 9 ^ (Real.log 4.1 / Real.log 2)
noncomputable def b : ℝ := 9 ^ (Real.log 2.7 / Real.log 2)
noncomputable def c : ℝ := (1 / 3 : ℝ) ^ (Real.log 0.1 / Real.log 2)

theorem compare_abc :
  a > c ∧ c > b := by
  sorry

end NUMINAMATH_GPT_compare_abc_l1940_194025


namespace NUMINAMATH_GPT_longest_interval_green_l1940_194090

-- Definitions for the conditions
def light_cycle_duration : ℕ := 180 -- total cycle duration in seconds
def green_duration : ℕ := 90 -- green light duration in seconds
def red_delay : ℕ := 10 -- red light delay between consecutive lights in seconds
def num_lights : ℕ := 8 -- number of lights

-- Theorem statement to be proved
theorem longest_interval_green (h1 : ∀ i : ℕ, i < num_lights → 
  ∃ t : ℕ, t < light_cycle_duration ∧ (∀ k : ℕ, i + k < num_lights → t + k * red_delay < light_cycle_duration ∧ t + k * red_delay + green_duration <= light_cycle_duration)):
  ∃ interval : ℕ, interval = 20 :=
sorry

end NUMINAMATH_GPT_longest_interval_green_l1940_194090


namespace NUMINAMATH_GPT_sum_of_coefficients_l1940_194014

theorem sum_of_coefficients (a a_1 a_2 a_3 a_4 a_5 a_6 a_7 : ℤ) (hx : (1 - 2 * x)^7 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6 + a_7 * x^7) :
  a_2 + a_3 + a_4 + a_5 + a_6 + a_7 = 12 :=
sorry

end NUMINAMATH_GPT_sum_of_coefficients_l1940_194014


namespace NUMINAMATH_GPT_question_a_plus_b_eq_11_b_plus_c_eq_9_c_plus_d_eq_3_a_plus_d_eq_neg1_l1940_194031

theorem question_a_plus_b_eq_11_b_plus_c_eq_9_c_plus_d_eq_3_a_plus_d_eq_neg1
    (a b c d : ℤ)
    (h1 : a + b = 11)
    (h2 : b + c = 9)
    (h3 : c + d = 3)
    : a + d = -1 :=
by
  sorry

end NUMINAMATH_GPT_question_a_plus_b_eq_11_b_plus_c_eq_9_c_plus_d_eq_3_a_plus_d_eq_neg1_l1940_194031


namespace NUMINAMATH_GPT_remainder_problem_l1940_194076

theorem remainder_problem (x y : ℤ) (k m : ℤ) 
  (hx : x = 126 * k + 11) 
  (hy : y = 126 * m + 25) :
  (x + y + 23) % 63 = 59 := 
by
  sorry

end NUMINAMATH_GPT_remainder_problem_l1940_194076


namespace NUMINAMATH_GPT_original_number_l1940_194029

theorem original_number (x : ℝ) (h : 1.10 * x = 550) : x = 500 :=
by
  sorry

end NUMINAMATH_GPT_original_number_l1940_194029


namespace NUMINAMATH_GPT_geometric_seq_a4_l1940_194037

variable {a : ℕ → ℝ}

theorem geometric_seq_a4 (h : ∀ n, a (n + 2) / a n = a 2 / a 0)
  (root_condition1 : a 2 * a 6 = 64)
  (root_condition2 : a 2 + a 6 = 34) :
  a 4 = 8 :=
by
  sorry

end NUMINAMATH_GPT_geometric_seq_a4_l1940_194037


namespace NUMINAMATH_GPT_pizza_slices_with_both_toppings_l1940_194040

theorem pizza_slices_with_both_toppings (total_slices pepperoni_slices mushroom_slices n : ℕ) 
    (h1 : total_slices = 14) 
    (h2 : pepperoni_slices = 8) 
    (h3 : mushroom_slices = 12) 
    (h4 : ∀ s, s = pepperoni_slices + mushroom_slices - n ∧ s = total_slices := by sorry) :
    n = 6 :=
sorry

end NUMINAMATH_GPT_pizza_slices_with_both_toppings_l1940_194040


namespace NUMINAMATH_GPT_chapters_in_first_book_l1940_194011

theorem chapters_in_first_book (x : ℕ) (h1 : 2 * 15 = 30) (h2 : (x + 30) / 2 + x + 30 = 75) : x = 20 :=
sorry

end NUMINAMATH_GPT_chapters_in_first_book_l1940_194011


namespace NUMINAMATH_GPT_combined_salaries_A_B_C_D_l1940_194018

-- To ensure the whole calculation is noncomputable due to ℝ
noncomputable section

-- Let's define the variables
def salary_E : ℝ := 9000
def average_salary_group : ℝ := 8400
def num_people : ℕ := 5

-- combined salary A + B + C + D represented as a definition
def combined_salaries : ℝ := (average_salary_group * num_people) - salary_E

-- We need to prove that the combined salaries equals 33000
theorem combined_salaries_A_B_C_D : combined_salaries = 33000 := by
  sorry

end NUMINAMATH_GPT_combined_salaries_A_B_C_D_l1940_194018


namespace NUMINAMATH_GPT_ratio_of_supply_to_demand_l1940_194059

def supply : ℕ := 1800000
def demand : ℕ := 2400000

theorem ratio_of_supply_to_demand : (supply / demand : ℚ) = 3 / 4 := by
  sorry

end NUMINAMATH_GPT_ratio_of_supply_to_demand_l1940_194059


namespace NUMINAMATH_GPT_line_through_fixed_point_and_parabola_l1940_194060

theorem line_through_fixed_point_and_parabola :
  (∀ (a : ℝ), ∃ (P : ℝ × ℝ), 
    (a - 1) * P.1 - P.2 + 2 * a + 1 = 0 ∧ 
    (∀ (x y : ℝ), (y^2 = - ((9:ℝ) / 2) * x ∧ x = -2 ∧ y = 3) ∨ (x^2 = (4:ℝ) / 3 * y ∧ x = -2 ∧ y = 3))) :=
by
  sorry

end NUMINAMATH_GPT_line_through_fixed_point_and_parabola_l1940_194060


namespace NUMINAMATH_GPT_product_with_a_equals_3_l1940_194075

theorem product_with_a_equals_3 (a : ℤ) (h : a = 3) : 
  (a - 12) * (a - 11) * (a - 10) * (a - 9) * (a - 8) * (a - 7) * (a - 6) * 
  (a - 5) * (a - 4) * (a - 3) * (a - 2) * (a - 1) * a * 3 = 0 :=
by
  sorry

end NUMINAMATH_GPT_product_with_a_equals_3_l1940_194075


namespace NUMINAMATH_GPT_Johnson_Carter_Tie_August_l1940_194069

structure MonthlyHomeRuns where
  March : Nat
  April : Nat
  May : Nat
  June : Nat
  July : Nat
  August : Nat
  September : Nat

def Johnson_runs : MonthlyHomeRuns := { March:= 2, April:= 11, May:= 15, June:= 9, July:= 7, August:= 9, September:= 0 }
def Carter_runs : MonthlyHomeRuns := { March:= 1, April:= 9, May:= 8, June:= 19, July:= 6, August:= 10, September:= 0 }

noncomputable def cumulative_runs (runs: MonthlyHomeRuns) (month: String) : Nat :=
  match month with
  | "March" => runs.March
  | "April" => runs.March + runs.April
  | "May" => runs.March + runs.April + runs.May
  | "June" => runs.March + runs.April + runs.May + runs.June
  | "July" => runs.March + runs.April + runs.May + runs.June + runs.July
  | "August" => runs.March + runs.April + runs.May + runs.June + runs.July + runs.August
  | _ => 0

theorem Johnson_Carter_Tie_August :
  cumulative_runs Johnson_runs "August" = cumulative_runs Carter_runs "August" := 
  by
  sorry

end NUMINAMATH_GPT_Johnson_Carter_Tie_August_l1940_194069


namespace NUMINAMATH_GPT_find_a_l1940_194043

theorem find_a (a : ℝ) (h_pos : a > 0) (h_ne_one : a ≠ 1) 
  (h_max : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → a^(2*x) + 2 * a^x - 1 ≤ 7) 
  (h_eq : ∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ a^(2*x) + 2 * a^x - 1 = 7) : 
  a = 2 ∨ a = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l1940_194043


namespace NUMINAMATH_GPT_length_of_larger_cuboid_l1940_194099

theorem length_of_larger_cuboid
  (n : ℕ)
  (l_small : ℝ) (w_small : ℝ) (h_small : ℝ)
  (w_large : ℝ) (h_large : ℝ)
  (V_large : ℝ)
  (n_eq : n = 56)
  (dim_small : l_small = 5 ∧ w_small = 3 ∧ h_small = 2)
  (dim_large : w_large = 14 ∧ h_large = 10)
  (V_large_eq : V_large = n * (l_small * w_small * h_small)) :
  ∃ l_large : ℝ, l_large = V_large / (w_large * h_large) ∧ l_large = 12 := by
  sorry

end NUMINAMATH_GPT_length_of_larger_cuboid_l1940_194099


namespace NUMINAMATH_GPT_evaluate_expression_l1940_194066

def numerator : ℤ :=
  (12 - 11) + (10 - 9) + (8 - 7) + (6 - 5) + (4 - 3) + (2 - 1)

def denominator : ℤ :=
  (2 - 3) + (4 - 5) + (6 - 7) + (8 - 9) + (10 - 11) + 12

theorem evaluate_expression : numerator / denominator = 6 / 7 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1940_194066


namespace NUMINAMATH_GPT_roots_quadratic_eq_l1940_194004

theorem roots_quadratic_eq (a b : ℝ) (h1 : a^2 + 3*a - 4 = 0) (h2 : b^2 + 3*b - 4 = 0) (h3 : a + b = -3) : a^2 + 4*a + b - 3 = -2 :=
by
  sorry

end NUMINAMATH_GPT_roots_quadratic_eq_l1940_194004


namespace NUMINAMATH_GPT_neither_sufficient_nor_necessary_l1940_194093

def M : Set ℝ := {x | 0 < x ∧ x ≤ 3}
def N : Set ℝ := {x | -1 < x ∧ x ≤ 2}

theorem neither_sufficient_nor_necessary (a : ℝ) :
  (a ∈ M → a ∈ N) = false ∧ (a ∈ N → a ∈ M) = false := by
  sorry

end NUMINAMATH_GPT_neither_sufficient_nor_necessary_l1940_194093


namespace NUMINAMATH_GPT_calculate_expression_l1940_194079

theorem calculate_expression : 64 + 5 * 12 / (180 / 3) = 65 := by
  sorry

end NUMINAMATH_GPT_calculate_expression_l1940_194079


namespace NUMINAMATH_GPT_angles_equal_l1940_194048

theorem angles_equal (A B C : ℝ) (h1 : A + B = 180) (h2 : B + C = 180) : A = C := sorry

end NUMINAMATH_GPT_angles_equal_l1940_194048


namespace NUMINAMATH_GPT_gum_pieces_in_each_packet_l1940_194013

theorem gum_pieces_in_each_packet
  (packets : ℕ) (chewed_pieces : ℕ) (remaining_pieces : ℕ) (total_pieces : ℕ)
  (h1 : packets = 8) (h2 : chewed_pieces = 54) (h3 : remaining_pieces = 2) (h4 : total_pieces = chewed_pieces + remaining_pieces)
  (h5 : total_pieces = packets * (total_pieces / packets)) :
  total_pieces / packets = 7 :=
by
  sorry

end NUMINAMATH_GPT_gum_pieces_in_each_packet_l1940_194013


namespace NUMINAMATH_GPT_simplify_expression_l1940_194010

theorem simplify_expression (a b c : ℝ) (ha : a = 7.4) (hb : b = 5 / 37) :
  1.6 * ((1 / a + 1 / b - 2 * c / (a * b)) * (a + b + 2 * c)) / 
  ((1 / a^2 + 1 / b^2 + 2 / (a * b) - 4 * c^2 / (a^2 * b^2))) = 1.6 :=
by 
  rw [ha, hb] 
  sorry

end NUMINAMATH_GPT_simplify_expression_l1940_194010


namespace NUMINAMATH_GPT_k_is_perfect_square_l1940_194094

theorem k_is_perfect_square (m n : ℤ) (hm : m > 0) (hn : n > 0) (k := ((m + n) ^ 2) / (4 * m * (m - n) ^ 2 + 4)) :
  ∃ (a : ℤ), k = a ^ 2 := by
  sorry

end NUMINAMATH_GPT_k_is_perfect_square_l1940_194094


namespace NUMINAMATH_GPT_trig_identity_l1940_194061

-- Proving the equality (we state the problem here)
theorem trig_identity :
  Real.sin (40 * Real.pi / 180) * (Real.tan (10 * Real.pi / 180) - Real.sqrt 3) = -8 / 3 :=
by
  sorry

end NUMINAMATH_GPT_trig_identity_l1940_194061


namespace NUMINAMATH_GPT_tv_price_change_l1940_194050

theorem tv_price_change (P : ℝ) :
  let decrease := 0.20
  let increase := 0.45
  let new_price := P * (1 - decrease)
  let final_price := new_price * (1 + increase)
  final_price - P = 0.16 * P := 
by
  sorry

end NUMINAMATH_GPT_tv_price_change_l1940_194050
