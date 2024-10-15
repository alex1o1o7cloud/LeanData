import Mathlib

namespace NUMINAMATH_GPT_find_m_values_l1070_107005

def is_solution (m : ℝ) : Prop :=
  let A : Set ℝ := {1, -2}
  let B : Set ℝ := {x | m * x + 1 = 0}
  B ⊆ A

theorem find_m_values :
  {m : ℝ | is_solution m} = {0, -1, 1 / 2} :=
by
  sorry

end NUMINAMATH_GPT_find_m_values_l1070_107005


namespace NUMINAMATH_GPT_sector_area_l1070_107090

theorem sector_area (C : ℝ) (θ : ℝ) (r : ℝ) (S : ℝ)
  (hC : C = (8 * Real.pi / 9) + 4)
  (hθ : θ = (80 * Real.pi / 180))
  (hne : θ * r / 2 + r = C) :
  S = (1 / 2) * θ * r^2 → S = 8 * Real.pi / 9 :=
by
  sorry

end NUMINAMATH_GPT_sector_area_l1070_107090


namespace NUMINAMATH_GPT_xiaozhi_needs_median_for_top_10_qualification_l1070_107067

-- Define a set of scores as a list of integers
def scores : List ℕ := sorry

-- Assume these scores are unique (this is a condition given in the problem)
axiom unique_scores : ∀ (a b : ℕ), a ∈ scores → b ∈ scores → a ≠ b → scores.indexOf a ≠ scores.indexOf b

-- Define the median function (in practice, you would implement this, but we're just outlining it here)
def median (scores: List ℕ) : ℕ := sorry

-- Define the position of Xiao Zhi's score
def xiaozhi_score : ℕ := sorry

-- Given that the top 10 scores are needed to advance
def top_10 (scores: List ℕ) : List ℕ := scores.take 10

-- Proposition that Xiao Zhi needs median to determine his rank in top 10
theorem xiaozhi_needs_median_for_top_10_qualification 
    (scores_median : ℕ) (zs_score : ℕ) : 
    (∀ (s: List ℕ), s = scores → scores_median = median s → zs_score ≤ scores_median → zs_score ∉ top_10 s) ∧ 
    (exists (s: List ℕ), s = scores → zs_score ∉ top_10 s → zs_score ≤ scores_median) := 
sorry

end NUMINAMATH_GPT_xiaozhi_needs_median_for_top_10_qualification_l1070_107067


namespace NUMINAMATH_GPT_machine_A_produces_7_sprockets_per_hour_l1070_107058

theorem machine_A_produces_7_sprockets_per_hour
    (A B : ℝ)
    (h1 : B = 1.10 * A)
    (h2 : ∃ t : ℝ, 770 = A * (t + 10) ∧ 770 = B * t) : 
    A = 7 := 
by 
    sorry

end NUMINAMATH_GPT_machine_A_produces_7_sprockets_per_hour_l1070_107058


namespace NUMINAMATH_GPT_items_counted_l1070_107012

def convert_counter (n : Nat) : Nat := sorry

theorem items_counted
  (counter_reading : Nat) 
  (condition_1 : ∀ d, d ∈ [5, 6, 7] → ¬(d ∈ [0, 1, 2, 3, 4, 8, 9]))
  (condition_2 : ∀ d1 d2, d1 = 4 → d2 = 8 → ¬(d2 = 5 ∨ d2 = 6 ∨ d2 = 7)) :
  convert_counter 388 = 151 :=
sorry

end NUMINAMATH_GPT_items_counted_l1070_107012


namespace NUMINAMATH_GPT_john_fouled_per_game_l1070_107048

theorem john_fouled_per_game
  (hit_rate : ℕ) (shots_per_foul : ℕ) (total_games : ℕ) (participation_rate : ℚ) (total_free_throws : ℕ) :
  hit_rate = 70 → shots_per_foul = 2 → total_games = 20 → participation_rate = 0.8 → total_free_throws = 112 →
  (total_free_throws / (participation_rate * total_games)) / shots_per_foul = 3.5 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_john_fouled_per_game_l1070_107048


namespace NUMINAMATH_GPT_movies_in_series_l1070_107047

theorem movies_in_series :
  -- conditions 
  let number_books := 10
  let books_read := 14
  let book_read_vs_movies_extra := 5
  (∀ number_movies : ℕ, 
  (books_read = number_movies + book_read_vs_movies_extra) →
  -- question
  number_movies = 9) := sorry

end NUMINAMATH_GPT_movies_in_series_l1070_107047


namespace NUMINAMATH_GPT_functions_from_M_to_N_l1070_107049

def M : Set ℤ := { -1, 1, 2, 3 }
def N : Set ℤ := { 0, 1, 2, 3, 4 }
def f2 (x : ℤ) := x + 1
def f4 (x : ℤ) := (x - 1)^2

theorem functions_from_M_to_N :
  (∀ x ∈ M, f2 x ∈ N) ∧ (∀ x ∈ M, f4 x ∈ N) :=
by
  sorry

end NUMINAMATH_GPT_functions_from_M_to_N_l1070_107049


namespace NUMINAMATH_GPT_geom_progression_lines_common_point_l1070_107081

theorem geom_progression_lines_common_point
  (a c b : ℝ) (r : ℝ)
  (h_geom_prog : c = a * r ∧ b = a * r^2) :
  ∃ (P : ℝ × ℝ), ∀ (a c b : ℝ), c = a * r ∧ b = a * r^2 → (P = (0, 0) ∧ a ≠ 0) :=
by
  sorry

end NUMINAMATH_GPT_geom_progression_lines_common_point_l1070_107081


namespace NUMINAMATH_GPT_total_distance_correct_l1070_107060

-- Given conditions
def fuel_efficiency_city : Float := 15
def fuel_efficiency_highway : Float := 25
def fuel_efficiency_gravel : Float := 18

def gallons_used_city : Float := 2.5
def gallons_used_highway : Float := 3.8
def gallons_used_gravel : Float := 1.7

-- Define distances
def distance_city := fuel_efficiency_city * gallons_used_city
def distance_highway := fuel_efficiency_highway * gallons_used_highway
def distance_gravel := fuel_efficiency_gravel * gallons_used_gravel

-- Define total distance
def total_distance := distance_city + distance_highway + distance_gravel

-- Prove the total distance traveled is 163.1 miles
theorem total_distance_correct : total_distance = 163.1 := by
  -- Proof to be filled in
  sorry

end NUMINAMATH_GPT_total_distance_correct_l1070_107060


namespace NUMINAMATH_GPT_intersect_complement_l1070_107018

-- Definition of the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Definition of set A
def A : Set ℕ := {1, 2, 3}

-- Definition of set B
def B : Set ℕ := {3, 4}

-- Definition of the complement of B in U
def CU (U : Set ℕ) (B : Set ℕ) : Set ℕ := {x | x ∈ U ∧ x ∉ B}

-- Expected result of the intersection
def result : Set ℕ := {1, 2}

-- The proof statement
theorem intersect_complement :
  A ∩ CU U B = result :=
sorry

end NUMINAMATH_GPT_intersect_complement_l1070_107018


namespace NUMINAMATH_GPT_inverse_proposition_l1070_107085

theorem inverse_proposition (a b : ℝ) (h : ab = 0) : (a = 0 → ab = 0) :=
by
  sorry

end NUMINAMATH_GPT_inverse_proposition_l1070_107085


namespace NUMINAMATH_GPT_three_students_received_A_l1070_107083

variables (A B C E D : Prop)
variables (h1 : A → B) (h2 : B → C) (h3 : C → E) (h4 : E → D)

theorem three_students_received_A :
  (A ∨ ¬A) ∧ (B ∨ ¬B) ∧ (C ∨ ¬C) ∧ (E ∨ ¬E) ∧ (D ∨ ¬D) ∧ (¬A ∧ ¬B) → (C ∧ E ∧ D) ∧ ¬A ∧ ¬B :=
by sorry

end NUMINAMATH_GPT_three_students_received_A_l1070_107083


namespace NUMINAMATH_GPT_integer_conditions_satisfy_eq_l1070_107087

theorem integer_conditions_satisfy_eq (
  a b c : ℤ 
) : (a > b ∧ b = c → (a * (a - b) + b * (b - c) + c * (c - a) = 2)) ∧
    (¬(a = b - 1 ∧ b = c - 2) → (a * (a - b) + b * (b - c) + c * (c - a) ≠ 2)) ∧
    (¬(a = c + 1 ∧ b = a + 2) → (a * (a - b) + b * (b - c) + c * (c - a) ≠ 2)) ∧
    (¬(a = c ∧ b - 2 = c) → (a * (a - b) + b * (b - c) + c * (c - a) ≠ 2)) ∧
    (¬(a + b + c = 2) → (a * (a - b) + b * (b - c) + c * (c - a) ≠ 2)) :=
by
sorry

end NUMINAMATH_GPT_integer_conditions_satisfy_eq_l1070_107087


namespace NUMINAMATH_GPT_original_fraction_eq_2_5_l1070_107035

theorem original_fraction_eq_2_5 (a b : ℤ) (h : (a + 4) * b = a * (b + 10)) : (a / b) = (2 / 5) := by
  sorry

end NUMINAMATH_GPT_original_fraction_eq_2_5_l1070_107035


namespace NUMINAMATH_GPT_constant_term_in_expansion_l1070_107063

theorem constant_term_in_expansion (n k : ℕ) (x : ℝ) (choose : ℕ → ℕ → ℕ):
  (choose 12 3) * (6 ^ 3) = 47520 :=
by
  sorry

end NUMINAMATH_GPT_constant_term_in_expansion_l1070_107063


namespace NUMINAMATH_GPT_divisibility_problem_l1070_107027

theorem divisibility_problem (n : ℕ) : 2016 ∣ ((n^2 + n)^2 - (n^2 - n)^2) * (n^6 - 1) := 
sorry

end NUMINAMATH_GPT_divisibility_problem_l1070_107027


namespace NUMINAMATH_GPT_matrix_solution_property_l1070_107074

theorem matrix_solution_property (N : Matrix (Fin 2) (Fin 2) ℝ) 
    (h : N = Matrix.of ![![2, 4], ![1, 4]]) :
    N ^ 4 - 5 * N ^ 3 + 9 * N ^ 2 - 5 * N = Matrix.of ![![6, 12], ![3, 6]] :=
by 
  sorry

end NUMINAMATH_GPT_matrix_solution_property_l1070_107074


namespace NUMINAMATH_GPT_bottles_sold_tuesday_l1070_107096

def initial_inventory : ℕ := 4500
def sold_monday : ℕ := 2445
def sold_days_wed_to_sun : ℕ := 50 * 5
def bottles_delivered_saturday : ℕ := 650
def final_inventory : ℕ := 1555

theorem bottles_sold_tuesday : 
  initial_inventory + bottles_delivered_saturday - sold_monday - sold_days_wed_to_sun - final_inventory = 900 := 
by
  sorry

end NUMINAMATH_GPT_bottles_sold_tuesday_l1070_107096


namespace NUMINAMATH_GPT_eval_product_eq_1093_l1070_107071

noncomputable def z : ℂ := Complex.exp (2 * Real.pi * Complex.I / 7)

theorem eval_product_eq_1093 : (3 - z) * (3 - z^2) * (3 - z^3) * (3 - z^4) * (3 - z^5) * (3 - z^6) = 1093 := by
  sorry

end NUMINAMATH_GPT_eval_product_eq_1093_l1070_107071


namespace NUMINAMATH_GPT_angle_parallel_result_l1070_107088

theorem angle_parallel_result (A B : ℝ) (h1 : A = 60) (h2 : (A = B ∨ A + B = 180)) : (B = 60 ∨ B = 120) :=
by
  sorry

end NUMINAMATH_GPT_angle_parallel_result_l1070_107088


namespace NUMINAMATH_GPT_total_ice_cubes_correct_l1070_107056

/-- Each tray holds 48 ice cubes -/
def cubes_per_tray : Nat := 48

/-- Billy has 24 trays -/
def number_of_trays : Nat := 24

/-- Calculate the total number of ice cubes -/
def total_ice_cubes (cubes_per_tray : Nat) (number_of_trays : Nat) : Nat :=
  cubes_per_tray * number_of_trays

/-- Proof that the total number of ice cubes is 1152 given the conditions -/
theorem total_ice_cubes_correct : total_ice_cubes cubes_per_tray number_of_trays = 1152 := by
  /- Here we state the main theorem, but we leave the proof as sorry per the instructions -/
  sorry

end NUMINAMATH_GPT_total_ice_cubes_correct_l1070_107056


namespace NUMINAMATH_GPT_rice_less_than_beans_by_30_l1070_107053

noncomputable def GB : ℝ := 60
noncomputable def S : ℝ := 50

theorem rice_less_than_beans_by_30 (R : ℝ) (x : ℝ) (h1 : R = 60 - x) (h2 : (2/3) * R + (4/5) * S + GB = 120) : 60 - R = 30 :=
by 
  -- Proof steps would go here, but they are not required for this task.
  sorry

end NUMINAMATH_GPT_rice_less_than_beans_by_30_l1070_107053


namespace NUMINAMATH_GPT_Maxim_is_correct_l1070_107099

-- Defining the parameters
def mortgage_rate := 0.125
def dividend_yield := 0.17

-- Theorem statement
theorem Maxim_is_correct : (dividend_yield - mortgage_rate > 0) := by 
    -- Dividing the proof's logical steps
    sorry

end NUMINAMATH_GPT_Maxim_is_correct_l1070_107099


namespace NUMINAMATH_GPT_power_difference_divisible_l1070_107003

-- Define the variables and conditions
variables {a b c : ℤ} {n : ℕ}

-- Condition: a - b is divisible by c
def is_divisible (a b c : ℤ) : Prop := ∃ k : ℤ, a - b = k * c

-- Lean proof statement
theorem power_difference_divisible {a b c : ℤ} {n : ℕ} (h : is_divisible a b c) : c ∣ (a^n - b^n) :=
  sorry

end NUMINAMATH_GPT_power_difference_divisible_l1070_107003


namespace NUMINAMATH_GPT_penniless_pete_dime_difference_l1070_107075

theorem penniless_pete_dime_difference :
  ∃ a b c : ℕ, 
  (a + b + c = 100) ∧ 
  (5 * a + 10 * b + 50 * c = 1350) ∧ 
  (b = 170 ∨ b = 8) ∧ 
  (b - 8 = 162 ∨ 170 - b = 162) :=
sorry

end NUMINAMATH_GPT_penniless_pete_dime_difference_l1070_107075


namespace NUMINAMATH_GPT_find_r_value_l1070_107080

theorem find_r_value (m : ℕ) (h_m : m = 3) (t : ℕ) (h_t : t = 3^m + 2) (r : ℕ) (h_r : r = 4^t - 2 * t) : r = 4^29 - 58 := by
  sorry

end NUMINAMATH_GPT_find_r_value_l1070_107080


namespace NUMINAMATH_GPT_find_f2_l1070_107052

def f (x : ℝ) : ℝ := sorry

theorem find_f2 : (∀ x, f (x-1) = x / (x-1)) → f 2 = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_f2_l1070_107052


namespace NUMINAMATH_GPT_mean_proportional_l1070_107002

theorem mean_proportional (a b c : ℝ) (ha : a = 1) (hb : b = 2) (h : c ^ 2 = a * b) : c = Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_mean_proportional_l1070_107002


namespace NUMINAMATH_GPT_trapezoid_circle_ratio_l1070_107030

variable (P R : ℝ)

def is_isosceles_trapezoid_inscribed_in_circle (P R : ℝ) : Prop :=
  ∃ m A, 
    m = P / 4 ∧
    A = m * 2 * R ∧
    A = (P * R) / 2

theorem trapezoid_circle_ratio (P R : ℝ) 
  (h : is_isosceles_trapezoid_inscribed_in_circle P R) :
  (P / 2 * π * R) = (P / 2 * π * R) :=
by
  -- Use the given condition to prove the statement
  sorry

end NUMINAMATH_GPT_trapezoid_circle_ratio_l1070_107030


namespace NUMINAMATH_GPT_solution_for_x_l1070_107008

theorem solution_for_x : ∀ (x : ℚ), (10 * x^2 + 9 * x - 2 = 0) ∧ (30 * x^2 + 59 * x - 6 = 0) → x = 1 / 5 :=
by
  sorry

end NUMINAMATH_GPT_solution_for_x_l1070_107008


namespace NUMINAMATH_GPT_induction_step_l1070_107040

theorem induction_step (k : ℕ) : ((k + 1 + k) * (k + 1 + k + 1) / (k + 1)) = 2 * (2 * k + 1) := by
  sorry

end NUMINAMATH_GPT_induction_step_l1070_107040


namespace NUMINAMATH_GPT_minimum_value_l1070_107016

noncomputable def min_value_of_expression (a b c : ℝ) : ℝ :=
  1/a + 2/b + 4/c

theorem minimum_value (a b c : ℝ) (h₀ : c > 0) (h₁ : a ≠ 0) (h₂ : b ≠ 0)
    (h₃ : 4 * a^2 - 2 * a * b + b^2 - c = 0)
    (h₄ : ∀ x y, 4*x^2 - 2*x*y + y^2 - c = 0 → |2*x + y| ≤ |2*a + b|)
    : min_value_of_expression a b c = -1 :=
sorry

end NUMINAMATH_GPT_minimum_value_l1070_107016


namespace NUMINAMATH_GPT_find_c_l1070_107009

theorem find_c (c : ℝ) :
  (∀ x : ℝ, -2 * x^2 + c * x - 8 < 0 ↔ x ∈ Set.Iio 2 ∪ Set.Ioi 6) → c = 16 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_find_c_l1070_107009


namespace NUMINAMATH_GPT_average_goal_l1070_107011

-- Define the list of initial rolls
def initial_rolls : List ℕ := [1, 3, 2, 4, 3, 5, 3, 4, 4, 2]

-- Define the next roll
def next_roll : ℕ := 2

-- Define the goal for the average
def goal_average : ℕ := 3

-- The theorem to prove that Ronald's goal for the average of all his rolls is 3
theorem average_goal : (List.sum (initial_rolls ++ [next_roll]) / (List.length (initial_rolls ++ [next_roll]))) = goal_average :=
by
  -- The proof will be provided later
  sorry

end NUMINAMATH_GPT_average_goal_l1070_107011


namespace NUMINAMATH_GPT_infinitely_many_H_points_l1070_107044

-- Define the curve C as (x^2 / 4) + y^2 = 1
def is_on_curve (x y : ℝ) : Prop :=
  (x^2 / 4) + y^2 = 1

-- Define point P on curve C
def is_H_point (P : ℝ × ℝ) : Prop :=
  is_on_curve P.1 P.2 ∧
  ∃ (A B : ℝ × ℝ), is_on_curve A.1 A.2 ∧ B.1 = 4 ∧
  (dist (P.1, P.2) (A.1, A.2) = dist (P.1, P.2) (B.1, B.2) ∨
   dist (P.1, P.2) (A.1, A.2) = dist (A.1, A.2) (B.1, B.2))

-- Theorem to prove the existence of infinitely many H points
theorem infinitely_many_H_points : ∃ (P : ℝ × ℝ), is_H_point P ∧ ∀ (Q : ℝ × ℝ), Q ≠ P → is_H_point Q :=
sorry


end NUMINAMATH_GPT_infinitely_many_H_points_l1070_107044


namespace NUMINAMATH_GPT_sin_triangle_sides_l1070_107065

theorem sin_triangle_sides (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0)
  (h₃ : a + b + c ≤ 2 * Real.pi) (h₄ : a + b > c) (h₅ : b + c > a) (h₆ : c + a > b) :
  ∃ x y z : ℝ, x = Real.sin a ∧ y = Real.sin b ∧ z = Real.sin c ∧ x + y > z ∧ y + z > x ∧ z + x > y := 
by
  sorry

end NUMINAMATH_GPT_sin_triangle_sides_l1070_107065


namespace NUMINAMATH_GPT_recurring_decimal_sum_l1070_107014

noncomputable def x : ℚ := 1 / 3

noncomputable def y : ℚ := 14 / 999

noncomputable def z : ℚ := 5 / 9999

theorem recurring_decimal_sum :
  x + y + z = 3478 / 9999 := by
  sorry

end NUMINAMATH_GPT_recurring_decimal_sum_l1070_107014


namespace NUMINAMATH_GPT_candy_left_proof_l1070_107062

def candy_left (d_candy : ℕ) (s_candy : ℕ) (eaten_candy : ℕ) : ℕ :=
  d_candy + s_candy - eaten_candy

theorem candy_left_proof :
  candy_left 32 42 35 = 39 :=
by
  sorry

end NUMINAMATH_GPT_candy_left_proof_l1070_107062


namespace NUMINAMATH_GPT_domain_of_function_l1070_107021

theorem domain_of_function :
  { x : ℝ // (6 - x - x^2) > 0 } = { x : ℝ // -3 < x ∧ x < 2 } :=
by
  sorry

end NUMINAMATH_GPT_domain_of_function_l1070_107021


namespace NUMINAMATH_GPT_range_of_m_l1070_107070

theorem range_of_m (f : ℝ → ℝ) (m : ℝ) 
  (h1 : ∀ x : ℝ, f x = x^2 + 4 * x + 5)
  (h2 : ∀ x : ℝ, f (-2 + x) = f (-2 - x))
  (h3 : ∀ x : ℝ, m ≤ x ∧ x ≤ 0 → 1 ≤ f x ∧ f x ≤ 5)
  : -4 ≤ m ∧ m ≤ -2 :=
  sorry

end NUMINAMATH_GPT_range_of_m_l1070_107070


namespace NUMINAMATH_GPT_scientific_notation_eq_l1070_107097

-- Define the number 82,600,000
def num : ℝ := 82600000

-- Define the scientific notation representation
def sci_not : ℝ := 8.26 * 10^7

-- The theorem to prove that the number is equal to its scientific notation
theorem scientific_notation_eq : num = sci_not :=
by 
  sorry

end NUMINAMATH_GPT_scientific_notation_eq_l1070_107097


namespace NUMINAMATH_GPT_prob_both_shoot_in_one_round_prob_specified_shots_in_two_rounds_l1070_107055

noncomputable def P_A := 4 / 5
noncomputable def P_B := 3 / 4

def independent (P_X P_Y : ℚ) := P_X * P_Y

theorem prob_both_shoot_in_one_round : independent P_A P_B = 3 / 5 := by
  sorry

noncomputable def P_A_1 := 2 * (4 / 5) * (1 / 5)
noncomputable def P_A_2 := (4 / 5) * (4 / 5)
noncomputable def P_B_1 := 2 * (3 / 4) * (1 / 4)
noncomputable def P_B_2 := (3 / 4) * (3 / 4)

def event_A (P_A_1 P_A_2 P_B_1 P_B_2 : ℚ) := (P_A_1 * P_B_2) + (P_A_2 * P_B_1)

theorem prob_specified_shots_in_two_rounds : event_A P_A_1 P_A_2 P_B_1 P_B_2 = 3 / 10 := by
  sorry

end NUMINAMATH_GPT_prob_both_shoot_in_one_round_prob_specified_shots_in_two_rounds_l1070_107055


namespace NUMINAMATH_GPT_time_spent_on_Type_A_problems_l1070_107037

theorem time_spent_on_Type_A_problems (t : ℝ) (h1 : 25 * (8 * t) + 100 * (2 * t) = 120) : 
  25 * (8 * t) = 60 := by
  sorry

-- Conditions
-- t is the time spent on a Type C problem in minutes
-- 25 * (8 * t) + 100 * (2 * t) = 120 (time spent on Type A and B problems combined equals 120 minutes)

end NUMINAMATH_GPT_time_spent_on_Type_A_problems_l1070_107037


namespace NUMINAMATH_GPT_innovation_contribution_l1070_107093

variable (material : String)
variable (contribution : String → Prop)
variable (A B C D : Prop)

-- Conditions
axiom condA : contribution material → A
axiom condB : contribution material → ¬B
axiom condC : contribution material → ¬C
axiom condD : contribution material → ¬D

-- The problem statement
theorem innovation_contribution :
  contribution material → A :=
by
  -- dummy proof as placeholder
  sorry

end NUMINAMATH_GPT_innovation_contribution_l1070_107093


namespace NUMINAMATH_GPT_measure_α_l1070_107017

noncomputable def measure_α_proof (AB BC : ℝ) (h1: AB = 1) (h2 : BC = 2) : ℝ :=
  let α := 120
  α

theorem measure_α (AB BC : ℝ) (h1: AB = 1) (h2 : BC = 2) : measure_α_proof AB BC h1 h2 = 120 :=
  sorry

end NUMINAMATH_GPT_measure_α_l1070_107017


namespace NUMINAMATH_GPT_total_uniform_cost_l1070_107022

theorem total_uniform_cost :
  let pants_cost := 20
  let shirt_cost := 2 * pants_cost
  let tie_cost := shirt_cost / 5
  let socks_cost := 3
  let uniform_cost := pants_cost + shirt_cost + tie_cost + socks_cost
  let total_cost := 5 * uniform_cost
  total_cost = 355 :=
by 
  let pants_cost := 20
  let shirt_cost := 2 * pants_cost
  let tie_cost := shirt_cost / 5
  let socks_cost := 3
  let uniform_cost := pants_cost + shirt_cost + tie_cost + socks_cost
  let total_cost := 5 * uniform_cost
  sorry

end NUMINAMATH_GPT_total_uniform_cost_l1070_107022


namespace NUMINAMATH_GPT_avg_remaining_two_l1070_107025

variables {A B C D E : ℝ}

-- Conditions
def avg_five (A B C D E : ℝ) : Prop := (A + B + C + D + E) / 5 = 10
def avg_three (A B C : ℝ) : Prop := (A + B + C) / 3 = 4

-- Theorem to prove
theorem avg_remaining_two (A B C D E : ℝ) (h1 : avg_five A B C D E) (h2 : avg_three A B C) : ((D + E) / 2) = 19 := 
sorry

end NUMINAMATH_GPT_avg_remaining_two_l1070_107025


namespace NUMINAMATH_GPT_f_800_value_l1070_107077

theorem f_800_value (f : ℝ → ℝ) (f_condition : ∀ x y : ℝ, 0 < x → 0 < y → f (x * y) = f x / y) (f_400 : f 400 = 4) : f 800 = 2 :=
  sorry

end NUMINAMATH_GPT_f_800_value_l1070_107077


namespace NUMINAMATH_GPT_smallest_possible_value_l1070_107020

theorem smallest_possible_value (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (⌊(a + b + c) / d⌋ + ⌊(a + b + d) / c⌋ + ⌊(a + c + d) / b⌋ + ⌊(b + c + d) / a⌋) ≥ 8 :=
sorry

end NUMINAMATH_GPT_smallest_possible_value_l1070_107020


namespace NUMINAMATH_GPT_Mark_average_speed_l1070_107015

theorem Mark_average_speed 
  (start_time : ℝ) (end_time : ℝ) (distance : ℝ)
  (h1 : start_time = 8.5) (h2 : end_time = 14.75) (h3 : distance = 210) :
  distance / (end_time - start_time) = 33.6 :=
by 
  sorry

end NUMINAMATH_GPT_Mark_average_speed_l1070_107015


namespace NUMINAMATH_GPT_perp_lines_of_parallel_planes_l1070_107010

variables {Line Plane : Type} 
variables (m n : Line) (α β : Plane)
variable (is_parallel : Line → Plane → Prop)
variable (is_perpendicular : Line → Plane → Prop)
variable (planes_parallel : Plane → Plane → Prop)
variable (lines_perpendicular : Line → Line → Prop)

-- Given Conditions
variables (h1 : planes_parallel α β) (h2 : is_perpendicular m α) (h3 : is_parallel n β)

-- Prove that
theorem perp_lines_of_parallel_planes (h1 : planes_parallel α β) (h2 : is_perpendicular m α) (h3 : is_parallel n β) : lines_perpendicular m n := 
sorry

end NUMINAMATH_GPT_perp_lines_of_parallel_planes_l1070_107010


namespace NUMINAMATH_GPT_smallest_b_factors_l1070_107057

theorem smallest_b_factors (b : ℕ) (m n : ℤ) (h : m * n = 2023 ∧ m + n = b) : b = 136 :=
sorry

end NUMINAMATH_GPT_smallest_b_factors_l1070_107057


namespace NUMINAMATH_GPT_milk_leftover_l1070_107023

def milk (milkshake_num : ℕ) := 4 * milkshake_num
def ice_cream (milkshake_num : ℕ) := 12 * milkshake_num
def possible_milkshakes (ice_cream_amount : ℕ) := ice_cream_amount / 12

theorem milk_leftover (total_milk total_ice_cream : ℕ) (h1 : total_milk = 72) (h2 : total_ice_cream = 192) :
  total_milk - milk (possible_milkshakes total_ice_cream) = 8 :=
by
  sorry

end NUMINAMATH_GPT_milk_leftover_l1070_107023


namespace NUMINAMATH_GPT_destroyed_cakes_l1070_107032

theorem destroyed_cakes (initial_cakes : ℕ) (half_falls : ℕ) (half_saved : ℕ)
  (h1 : initial_cakes = 12)
  (h2 : half_falls = initial_cakes / 2)
  (h3 : half_saved = half_falls / 2) :
  initial_cakes - half_falls / 2 = 3 :=
by
  sorry

end NUMINAMATH_GPT_destroyed_cakes_l1070_107032


namespace NUMINAMATH_GPT_min_distance_between_tracks_l1070_107041

noncomputable def min_distance : ℝ :=
  (Real.sqrt 163 - 6) / 3

theorem min_distance_between_tracks :
  let RationalManTrack := {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}
  let IrrationalManTrack := {p : ℝ × ℝ | (p.1 - 2)^2 / 9 + p.2^2 / 25 = 1}
  ∀ pA ∈ RationalManTrack, ∀ pB ∈ IrrationalManTrack,
  dist pA pB = min_distance :=
sorry

end NUMINAMATH_GPT_min_distance_between_tracks_l1070_107041


namespace NUMINAMATH_GPT_bus_stops_for_18_minutes_l1070_107086

-- Definitions based on conditions
def speed_without_stoppages : ℝ := 50 -- kmph
def speed_with_stoppages : ℝ := 35 -- kmph
def distance_reduced_due_to_stoppage_per_hour : ℝ := speed_without_stoppages - speed_with_stoppages

noncomputable def time_bus_stops_per_hour (distance : ℝ) (speed : ℝ) : ℝ := distance / speed

theorem bus_stops_for_18_minutes :
  time_bus_stops_per_hour distance_reduced_due_to_stoppage_per_hour (speed_without_stoppages / 60) = 18 := by
  sorry

end NUMINAMATH_GPT_bus_stops_for_18_minutes_l1070_107086


namespace NUMINAMATH_GPT_solve_for_x_l1070_107095

theorem solve_for_x (x : ℚ) : 
  (3 - x) / (x + 2) + (3 * x - 9) / (3 - x) = 2 ↔ x = -7 / 6 :=
by sorry

end NUMINAMATH_GPT_solve_for_x_l1070_107095


namespace NUMINAMATH_GPT_find_smaller_integer_l1070_107079

theorem find_smaller_integer (x : ℤ) (h1 : ∃ y : ℤ, y = 2 * x) (h2 : x + 2 * x = 96) : x = 32 :=
sorry

end NUMINAMATH_GPT_find_smaller_integer_l1070_107079


namespace NUMINAMATH_GPT_cone_lateral_surface_area_l1070_107024

noncomputable def lateralSurfaceArea (r l : ℝ) : ℝ := Real.pi * r * l

theorem cone_lateral_surface_area : 
  ∀ (r l : ℝ), r = 2 → l = 5 → lateralSurfaceArea r l = 10 * Real.pi :=
by 
  intros r l hr hl
  rw [hr, hl]
  unfold lateralSurfaceArea
  norm_num
  sorry

end NUMINAMATH_GPT_cone_lateral_surface_area_l1070_107024


namespace NUMINAMATH_GPT_stickers_left_correct_l1070_107006

-- Define the initial number of stickers and number of stickers given away
def n_initial : ℝ := 39.0
def n_given_away : ℝ := 22.0

-- Proof statement: The number of stickers left at the end is 17.0
theorem stickers_left_correct : n_initial - n_given_away = 17.0 := by
  sorry

end NUMINAMATH_GPT_stickers_left_correct_l1070_107006


namespace NUMINAMATH_GPT_find_x_squared_plus_y_squared_l1070_107031

open Real

theorem find_x_squared_plus_y_squared (x y : ℝ) (h1 : (x + y)^2 = 4) (h2 : x * y = -6) : x^2 + y^2 = 16 := by
  sorry

end NUMINAMATH_GPT_find_x_squared_plus_y_squared_l1070_107031


namespace NUMINAMATH_GPT_students_in_donnelly_class_l1070_107028

-- Conditions
def initial_cupcakes : ℕ := 40
def cupcakes_to_delmont_class : ℕ := 18
def cupcakes_to_staff : ℕ := 4
def leftover_cupcakes : ℕ := 2

-- Question: How many students are in Mrs. Donnelly's class?
theorem students_in_donnelly_class : 
  let cupcakes_given_to_students := initial_cupcakes - (cupcakes_to_delmont_class + cupcakes_to_staff)
  let cupcakes_given_to_donnelly_class := cupcakes_given_to_students - leftover_cupcakes
  cupcakes_given_to_donnelly_class = 16 :=
by
  sorry

end NUMINAMATH_GPT_students_in_donnelly_class_l1070_107028


namespace NUMINAMATH_GPT_find_fraction_l1070_107073

theorem find_fraction (c d : ℕ) (h1 : 435 = 2 * 100 + c * 10 + d) :
  (c + d) / 12 = 5 / 6 :=
by sorry

end NUMINAMATH_GPT_find_fraction_l1070_107073


namespace NUMINAMATH_GPT_expression_equals_4034_l1070_107078

theorem expression_equals_4034 : 6 * 2017 - 4 * 2017 = 4034 := by
  sorry

end NUMINAMATH_GPT_expression_equals_4034_l1070_107078


namespace NUMINAMATH_GPT_karen_cases_pickup_l1070_107076

theorem karen_cases_pickup (total_boxes cases_per_box: ℕ) (h1 : total_boxes = 36) (h2 : cases_per_box = 12):
  total_boxes / cases_per_box = 3 :=
by
  -- We insert a placeholder to skip the proof here
  sorry

end NUMINAMATH_GPT_karen_cases_pickup_l1070_107076


namespace NUMINAMATH_GPT_boys_planted_more_by_62_percent_girls_fraction_of_total_l1070_107043

-- Define the number of trees planted by boys and girls
def boys_trees : ℕ := 130
def girls_trees : ℕ := 80

-- Statement 1: Boys planted 62% more trees than girls
theorem boys_planted_more_by_62_percent : (boys_trees - girls_trees) * 100 / girls_trees = 62 := by
  sorry

-- Statement 2: The number of trees planted by girls represents 4/7 of the total number of trees
theorem girls_fraction_of_total : girls_trees * 7 = 4 * (boys_trees + girls_trees) := by
  sorry

end NUMINAMATH_GPT_boys_planted_more_by_62_percent_girls_fraction_of_total_l1070_107043


namespace NUMINAMATH_GPT_price_of_mixture_l1070_107046

theorem price_of_mixture :
  (1 * 64 + 1 * 74) / (1 + 1) = 69 :=
by
  sorry

end NUMINAMATH_GPT_price_of_mixture_l1070_107046


namespace NUMINAMATH_GPT_fixed_point_line_l1070_107036

theorem fixed_point_line (m x y : ℝ) (h : (2 * m + 1) * x + (m + 1) * y - 7 * m - 4 = 0) :
  x = 3 ∧ y = 1 :=
sorry

end NUMINAMATH_GPT_fixed_point_line_l1070_107036


namespace NUMINAMATH_GPT_december_sales_fraction_l1070_107059

theorem december_sales_fraction (A : ℚ) : 
  let sales_jan_to_nov := 11 * A
  let sales_dec := 5 * A
  let total_sales := sales_jan_to_nov + sales_dec
  (sales_dec / total_sales) = 5 / 16 :=
by
  sorry

end NUMINAMATH_GPT_december_sales_fraction_l1070_107059


namespace NUMINAMATH_GPT_red_team_score_l1070_107098

theorem red_team_score (C R : ℕ) (h1 : C = 95) (h2 : C - R = 19) : R = 76 :=
by
  sorry

end NUMINAMATH_GPT_red_team_score_l1070_107098


namespace NUMINAMATH_GPT_eccentricity_is_sqrt2_div2_l1070_107089

noncomputable def eccentricity_square_ellipse (a b c : ℝ) : ℝ :=
  c / (Real.sqrt (b ^ 2 + c ^ 2))

theorem eccentricity_is_sqrt2_div2 (a b c : ℝ) (h : b = c) : 
  eccentricity_square_ellipse a b c = Real.sqrt 2 / 2 :=
by
  -- The proof will show that the eccentricity calculation is correct given the conditions.
  sorry

end NUMINAMATH_GPT_eccentricity_is_sqrt2_div2_l1070_107089


namespace NUMINAMATH_GPT_crayons_per_child_l1070_107092

theorem crayons_per_child (total_crayons children : ℕ) (h_total : total_crayons = 56) (h_children : children = 7) : (total_crayons / children) = 8 := by
  -- proof will go here
  sorry

end NUMINAMATH_GPT_crayons_per_child_l1070_107092


namespace NUMINAMATH_GPT_gala_arrangements_l1070_107051

theorem gala_arrangements :
  let original_programs := 10
  let added_programs := 3
  let total_positions := original_programs + 1 - 2 -- Excluding first and last
  (total_positions * (total_positions - 1) * (total_positions - 2)) / 6 = 165 :=
by sorry

end NUMINAMATH_GPT_gala_arrangements_l1070_107051


namespace NUMINAMATH_GPT_intersection_points_area_l1070_107061

noncomputable def C (x : ℝ) : ℝ := (Real.log x)^2

noncomputable def L (α : ℝ) (x : ℝ) : ℝ :=
  (2 * Real.log α / α) * x - (Real.log α)^2

noncomputable def n (α : ℝ) : ℕ :=
  if α < 1 then 0 else if α = 1 then 1 else 2

noncomputable def S (α : ℝ) : ℝ :=
  2 - 2 * α - (1 / 2) * α * (Real.log α)^2 + 2 * α * Real.log α

theorem intersection_points (α : ℝ) (h : 0 < α) : n α = if α < 1 then 0 else if α = 1 then 1 else 2 := by
  sorry

theorem area (α : ℝ) (h : 0 < α ∧ α < 1) : S α = 2 - 2 * α - (1 / 2) * α * (Real.log α)^2 + 2 * α * Real.log α := by
  sorry

end NUMINAMATH_GPT_intersection_points_area_l1070_107061


namespace NUMINAMATH_GPT_max_sum_numbered_cells_max_zero_number_cell_l1070_107050

-- Part 1
theorem max_sum_numbered_cells (n : ℕ) (grid : Matrix (Fin (2*n+1)) (Fin (2*n+1)) Cell) (mines : Finset (Fin (2*n+1) × Fin (2*n+1))) 
  (h1 : mines.card = n^2 + 1) :
  ∃ sum : ℕ, sum = 8 * n^2 + 4 := sorry

-- Part 2
theorem max_zero_number_cell (n k : ℕ) (grid : Matrix (Fin n) (Fin n) Cell) (mines : Finset (Fin n × Fin n)) 
  (h1 : mines.card = k) :
  ∃ (k_max : ℕ), k_max = (Nat.floor ((n + 2) / 3) ^ 2) - 1 := sorry

end NUMINAMATH_GPT_max_sum_numbered_cells_max_zero_number_cell_l1070_107050


namespace NUMINAMATH_GPT_equation_of_line_l1070_107039

noncomputable def line_equation_parallel (x y : ℝ) : Prop :=
  ∃ (m : ℝ), (3 * x - 6 * y = 9) ∧ (m = 1/2)

theorem equation_of_line (m : ℝ) (b : ℝ) :
  line_equation_parallel 3 9 →
  (m = 1/2) →
  (∀ (x y : ℝ), (y = m * x + b) ↔ (y - (-1) = m * (x - 2))) →
  b = -2 :=
by
  intros h_eq h_m h_line
  sorry

end NUMINAMATH_GPT_equation_of_line_l1070_107039


namespace NUMINAMATH_GPT_lambda_phi_relation_l1070_107007

-- Define the context and conditions
variables (A B C D M N : Type) -- Points on the triangle with D being the midpoint of BC
variables (AB AC BC BN BM MN : ℝ) -- Lengths
variables (lambda phi : ℝ) -- Ratios given in the problem

-- Conditions
-- 1. M is a point on the median AD of triangle ABC
variable (h1 : M = D ∨ M = A ∨ M = D) -- Simplified condition stating M's location
-- 2. The line BM intersects the side AC at point N
variable (h2 : N = M ∧ N ≠ A ∧ N ≠ C) -- Defining the intersection point
-- 3. AB is tangent to the circumcircle of triangle NBC
variable (h3 : tangent AB (circumcircle N B C))
-- 4. BC = lambda BN
variable (h4 : BC = lambda * BN)
-- 5. BM = phi * MN
variable (h5 : BM = phi * MN)

-- Goal
theorem lambda_phi_relation : phi = lambda ^ 2 :=
sorry

end NUMINAMATH_GPT_lambda_phi_relation_l1070_107007


namespace NUMINAMATH_GPT_articles_selling_price_eq_cost_price_of_50_articles_l1070_107084

theorem articles_selling_price_eq_cost_price_of_50_articles (C S : ℝ) (N : ℕ) 
  (h1 : 50 * C = N * S) (h2 : S = 2 * C) : N = 25 := by
  sorry

end NUMINAMATH_GPT_articles_selling_price_eq_cost_price_of_50_articles_l1070_107084


namespace NUMINAMATH_GPT_sin_minus_cos_eq_neg_sqrt_10_over_5_l1070_107045

theorem sin_minus_cos_eq_neg_sqrt_10_over_5 (θ : ℝ) (h1 : 0 < θ ∧ θ < π / 2) (h2 : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = - ((Real.sqrt 10) / 5) :=
by
  sorry

end NUMINAMATH_GPT_sin_minus_cos_eq_neg_sqrt_10_over_5_l1070_107045


namespace NUMINAMATH_GPT_eggs_per_chicken_l1070_107034

theorem eggs_per_chicken (num_chickens : ℕ) (eggs_per_carton : ℕ) (num_cartons : ℕ) (total_eggs : ℕ) 
  (h1 : num_chickens = 20) (h2 : eggs_per_carton = 12) (h3 : num_cartons = 10) (h4 : total_eggs = num_cartons * eggs_per_carton) : 
  total_eggs / num_chickens = 6 :=
by
  sorry

end NUMINAMATH_GPT_eggs_per_chicken_l1070_107034


namespace NUMINAMATH_GPT_cube_side_length_l1070_107019

theorem cube_side_length (n : ℕ) (h1 : 6 * (n^2) = 1/3 * 6 * (n^3)) : n = 3 := 
sorry

end NUMINAMATH_GPT_cube_side_length_l1070_107019


namespace NUMINAMATH_GPT_convert_20121_base3_to_base10_l1070_107068

/- Define the base conversion function for base 3 to base 10 -/
def base3_to_base10 (d4 d3 d2 d1 d0 : ℕ) :=
  d4 * 3^4 + d3 * 3^3 + d2 * 3^2 + d1 * 3^1 + d0 * 3^0

/- Define the specific number in base 3 -/
def num20121_base3 := (2, 0, 1, 2, 1)

/- The theorem stating the equivalence of the base 3 number 20121_3 to its base 10 equivalent -/
theorem convert_20121_base3_to_base10 :
  base3_to_base10 2 0 1 2 1 = 178 :=
by
  sorry

end NUMINAMATH_GPT_convert_20121_base3_to_base10_l1070_107068


namespace NUMINAMATH_GPT_value_of_T_l1070_107072

variables {A M T E H : ℕ}

theorem value_of_T (H : ℕ) (MATH : ℕ) (MEET : ℕ) (TEAM : ℕ) (H_eq : H = 8) (MATH_eq : MATH = 47) (MEET_eq : MEET = 62) (TEAM_eq : TEAM = 58) :
  T = 9 :=
by
  sorry

end NUMINAMATH_GPT_value_of_T_l1070_107072


namespace NUMINAMATH_GPT_RiversideAcademy_statistics_l1070_107026

theorem RiversideAcademy_statistics (total_students physics_students both_subjects : ℕ)
  (h1 : total_students = 25)
  (h2 : physics_students = 10)
  (h3 : both_subjects = 6) :
  total_students - (physics_students - both_subjects) = 21 :=
by
  sorry

end NUMINAMATH_GPT_RiversideAcademy_statistics_l1070_107026


namespace NUMINAMATH_GPT_dance_contradiction_l1070_107000

variable {Boy Girl : Type}
variable {danced_with : Boy → Girl → Prop}

theorem dance_contradiction
    (H1 : ¬ ∃ g : Boy, ∀ f : Girl, danced_with g f)
    (H2 : ∀ f : Girl, ∃ g : Boy, danced_with g f) :
    ∃ (g g' : Boy) (f f' : Girl),
        danced_with g f ∧ ¬ danced_with g f' ∧
        danced_with g' f' ∧ ¬ danced_with g' f :=
by
  -- Proof will be inserted here
  sorry

end NUMINAMATH_GPT_dance_contradiction_l1070_107000


namespace NUMINAMATH_GPT_T_53_eq_38_l1070_107064

def T (x y : ℕ) : ℕ := 4 * x + 6 * y

theorem T_53_eq_38 : T 5 3 = 38 := by
  sorry

end NUMINAMATH_GPT_T_53_eq_38_l1070_107064


namespace NUMINAMATH_GPT_smallest_n_l1070_107033

def is_perfect_fourth (m : ℕ) : Prop := ∃ x : ℕ, m = x^4
def is_perfect_fifth (m : ℕ) : Prop := ∃ y : ℕ, m = y^5

theorem smallest_n :
  ∃ n : ℕ, n > 0 ∧ is_perfect_fourth (3 * n) ∧ is_perfect_fifth (2 * n) ∧ n = 6912 :=
by {
  sorry
}

end NUMINAMATH_GPT_smallest_n_l1070_107033


namespace NUMINAMATH_GPT_find_first_term_l1070_107066

theorem find_first_term
  (a r : ℝ)
  (h1 : a / (1 - r) = 20)
  (h2 : a^2 / (1 - r^2) = 80) :
  a = 20 / 3 :=
by
  -- Proof is omitted for brevity
  sorry

end NUMINAMATH_GPT_find_first_term_l1070_107066


namespace NUMINAMATH_GPT_inversely_proportional_y_ratio_l1070_107082

variable {k : ℝ}
variable {x₁ x₂ y₁ y₂ : ℝ}
variable (h_inv_prop : ∀ (x y : ℝ), x * y = k)
variable (hx₁x₂ : x₁ ≠ 0 ∧ x₂ ≠ 0)
variable (hy₁y₂ : y₁ ≠ 0 ∧ y₂ ≠ 0)
variable (hx_ratio : x₁ / x₂ = 3 / 4)

theorem inversely_proportional_y_ratio :
  y₁ / y₂ = 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_inversely_proportional_y_ratio_l1070_107082


namespace NUMINAMATH_GPT_maximize_product_minimize_product_l1070_107069

-- Define the numbers that need to be arranged
def numbers : List ℕ := [2, 4, 6, 8]

-- Prove that 82 * 64 is the maximum product arrangement
theorem maximize_product : ∃ a b c d : ℕ, (a = 8) ∧ (b = 2) ∧ (c = 6) ∧ (d = 4) ∧ 
  (a * 10 + b) * (c * 10 + d) = 5248 :=
by
  existsi 8, 2, 6, 4
  constructor; constructor
  repeat {assumption}
  sorry

-- Prove that 28 * 46 is the minimum product arrangement
theorem minimize_product : ∃ a b c d : ℕ, (a = 2) ∧ (b = 8) ∧ (c = 4) ∧ (d = 6) ∧ 
  (a * 10 + b) * (c * 10 + d) = 1288 :=
by
  existsi 2, 8, 4, 6
  constructor; constructor
  repeat {assumption}
  sorry

end NUMINAMATH_GPT_maximize_product_minimize_product_l1070_107069


namespace NUMINAMATH_GPT_area_difference_is_correct_l1070_107091

noncomputable def circumference_1 : ℝ := 264
noncomputable def circumference_2 : ℝ := 352

noncomputable def radius_1 : ℝ := circumference_1 / (2 * Real.pi)
noncomputable def radius_2 : ℝ := circumference_2 / (2 * Real.pi)

noncomputable def area_1 : ℝ := Real.pi * radius_1^2
noncomputable def area_2 : ℝ := Real.pi * radius_2^2

noncomputable def area_difference : ℝ := area_2 - area_1

theorem area_difference_is_correct :
  abs (area_difference - 4305.28) < 1e-2 :=
by
  sorry

end NUMINAMATH_GPT_area_difference_is_correct_l1070_107091


namespace NUMINAMATH_GPT_distance_between_trees_l1070_107038

theorem distance_between_trees (num_trees : ℕ) (length_yard : ℝ)
  (h1 : num_trees = 26) (h2 : length_yard = 800) : 
  (length_yard / (num_trees - 1)) = 32 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_trees_l1070_107038


namespace NUMINAMATH_GPT_smallest_number_of_three_integers_l1070_107001

theorem smallest_number_of_three_integers 
  (a b c : ℕ) 
  (hpos1 : 0 < a) (hpos2 : 0 < b) (hpos3 : 0 < c) 
  (hmean : (a + b + c) / 3 = 24)
  (hmed : b = 23)
  (hlargest : b + 4 = c) 
  : a = 22 :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_of_three_integers_l1070_107001


namespace NUMINAMATH_GPT_total_molecular_weight_is_1317_12_l1070_107013

def atomic_weight_Al : ℝ := 26.98
def atomic_weight_S : ℝ := 32.06
def atomic_weight_H : ℝ := 1.01
def atomic_weight_O : ℝ := 16.00
def atomic_weight_C : ℝ := 12.01

def molecular_weight_Al2S3 : ℝ := (2 * atomic_weight_Al) + (3 * atomic_weight_S)
def molecular_weight_H2O : ℝ := (2 * atomic_weight_H) + (1 * atomic_weight_O)
def molecular_weight_CO2 : ℝ := (1 * atomic_weight_C) + (2 * atomic_weight_O)

def total_weight_7_Al2S3 : ℝ := 7 * molecular_weight_Al2S3
def total_weight_5_H2O : ℝ := 5 * molecular_weight_H2O
def total_weight_4_CO2 : ℝ := 4 * molecular_weight_CO2

def total_molecular_weight : ℝ := total_weight_7_Al2S3 + total_weight_5_H2O + total_weight_4_CO2

theorem total_molecular_weight_is_1317_12 : total_molecular_weight = 1317.12 := by
  sorry

end NUMINAMATH_GPT_total_molecular_weight_is_1317_12_l1070_107013


namespace NUMINAMATH_GPT_value_of_a_l1070_107004

noncomputable def f (a x : ℝ) : ℝ := a^x + Real.logb a (x^2 + 1)

theorem value_of_a (a : ℝ) (h : f a 1 + f a 2 = a^2 + a + 2) : a = Real.sqrt 10 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_l1070_107004


namespace NUMINAMATH_GPT_cos_alpha_correct_l1070_107029

-- Define the point P
def P : ℝ × ℝ := (3, -4)

-- Define the hypotenuse using the Pythagorean theorem
noncomputable def r : ℝ :=
  Real.sqrt (P.1 * P.1 + P.2 * P.2)

-- Define x-coordinate of point P
def x : ℝ := P.1

-- Define the cosine of the angle
noncomputable def cos_alpha : ℝ :=
  x / r

-- Prove that cos_alpha equals 3/5 given the conditions
theorem cos_alpha_correct : cos_alpha = 3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_cos_alpha_correct_l1070_107029


namespace NUMINAMATH_GPT_factorization_roots_l1070_107094

theorem factorization_roots (x : ℂ) : 
  (x^3 - 2*x^2 - x + 2) * (x - 3) * (x + 1) = 0 ↔ (x = -1 ∨ x = 1 ∨ x = 2 ∨ x = 3) :=
by
  -- Note: Proof to be completed
  sorry

end NUMINAMATH_GPT_factorization_roots_l1070_107094


namespace NUMINAMATH_GPT_tank_fill_time_l1070_107042

theorem tank_fill_time (R L : ℝ) (h1 : (R - L) * 8 = 1) (h2 : L * 56 = 1) :
  (1 / R) = 7 :=
by
  sorry

end NUMINAMATH_GPT_tank_fill_time_l1070_107042


namespace NUMINAMATH_GPT_find_x_l1070_107054

theorem find_x (x : ℝ) 
  (h1 : x = (1 / x * -x) - 5) 
  (h2 : x^2 - 3 * x + 2 ≥ 0) : 
  x = -6 := 
sorry

end NUMINAMATH_GPT_find_x_l1070_107054
