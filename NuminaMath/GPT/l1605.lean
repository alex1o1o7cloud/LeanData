import Mathlib

namespace NUMINAMATH_GPT_cone_height_l1605_160591

theorem cone_height (r_sector : ℝ) (θ_sector : ℝ) :
  r_sector = 3 → θ_sector = (2 * Real.pi / 3) → 
  ∃ (h : ℝ), h = 2 * Real.sqrt 2 := 
by 
  intros r_sector_eq θ_sector_eq
  sorry

end NUMINAMATH_GPT_cone_height_l1605_160591


namespace NUMINAMATH_GPT_solve_for_x_l1605_160578

variables {x y : ℝ}

theorem solve_for_x (h : x / (x - 3) = (y^2 + 3 * y + 1) / (y^2 + 3 * y - 4)) : 
  x = (3 * y^2 + 9 * y + 3) / 5 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l1605_160578


namespace NUMINAMATH_GPT_b_horses_pasture_l1605_160587

theorem b_horses_pasture (H : ℕ) : (9 * H / (96 + 9 * H + 108)) * 870 = 360 → H = 6 :=
by
  -- Here we state the problem and skip the proof
  sorry

end NUMINAMATH_GPT_b_horses_pasture_l1605_160587


namespace NUMINAMATH_GPT_cylinder_area_ratio_l1605_160554

theorem cylinder_area_ratio (r h : ℝ) (h_eq : h = 2 * r * Real.sqrt π) :
  let S_lateral := 2 * π * r * h
  let S_total := S_lateral + 2 * π * r^2
  S_total / S_lateral = 1 + (1 / (2 * Real.sqrt π)) := by
sorry

end NUMINAMATH_GPT_cylinder_area_ratio_l1605_160554


namespace NUMINAMATH_GPT_find_d_l1605_160529

noncomputable def f (x : ℝ) (c : ℝ) : ℝ := 5 * x + c
noncomputable def g (x : ℝ) (c : ℝ) : ℝ := c * x - 3

theorem find_d (c d : ℝ) (h : ∀ x, f (g x c) c = 15 * x + d) : d = -12 :=
by
  have h1 : ∀ x, f (g x c) c = 5 * (c * x - 3) + c := by intros; simp [f, g]
  have h2 : ∀ x, 5 * (c * x - 3) + c = 5 * c * x + c - 15 := by intros; ring
  specialize h 0
  rw [h1, h2] at h
  sorry

end NUMINAMATH_GPT_find_d_l1605_160529


namespace NUMINAMATH_GPT_train_ride_duration_is_360_minutes_l1605_160504

-- Define the conditions given in the problem
def arrived_at_station_at_8 (t : ℕ) : Prop := t = 8 * 60
def train_departed_at_835 (t_depart : ℕ) : Prop := t_depart = 8 * 60 + 35
def train_arrived_at_215 (t_arrive : ℕ) : Prop := t_arrive = 14 * 60 + 15
def exited_station_at_3 (t_exit : ℕ) : Prop := t_exit = 15 * 60

-- Define the problem statement
theorem train_ride_duration_is_360_minutes (boarding alighting : ℕ) :
  arrived_at_station_at_8 boarding ∧ 
  train_departed_at_835 boarding ∧ 
  train_arrived_at_215 alighting ∧ 
  exited_station_at_3 alighting → 
  alighting - boarding = 360 := 
by
  sorry

end NUMINAMATH_GPT_train_ride_duration_is_360_minutes_l1605_160504


namespace NUMINAMATH_GPT_bob_coloring_l1605_160568

/-
  Problem:
  Find the number of ways to color five points in {(x, y) | 1 ≤ x, y ≤ 5} blue 
  such that the distance between any two blue points is not an integer.
-/

def is_integer_distance (p1 p2 : ℤ × ℤ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let d := Int.gcd ((x2 - x1)^2 + (y2 - y1)^2)
  d ≠ 1

def valid_coloring (points : List (ℤ × ℤ)) : Prop :=
  points.length = 5 ∧ 
  (∀ (p1 p2 : ℤ × ℤ), p1 ∈ points → p2 ∈ points → p1 ≠ p2 → ¬ is_integer_distance p1 p2)

theorem bob_coloring : ∃ (points : List (ℤ × ℤ)), valid_coloring points ∧ points.length = 80 :=
sorry

end NUMINAMATH_GPT_bob_coloring_l1605_160568


namespace NUMINAMATH_GPT_andrew_age_l1605_160590

theorem andrew_age (a g : ℕ) (h1 : g = 10 * a) (h2 : g - a = 63) : a = 7 := by
  sorry

end NUMINAMATH_GPT_andrew_age_l1605_160590


namespace NUMINAMATH_GPT_find_f_neg_two_l1605_160501

def is_even_function (f : ℝ → ℝ) (h : ℝ → ℝ) := ∀ x, h (-x) = h x

theorem find_f_neg_two (f : ℝ → ℝ) (h : ℝ → ℝ) (hx : ∀ x, h x = f (2*x) + x)
  (h_even : is_even_function f h) 
  (h_f_two : f 2 = 1) : 
  f (-2) = 3 :=
  by
    sorry

end NUMINAMATH_GPT_find_f_neg_two_l1605_160501


namespace NUMINAMATH_GPT_min_students_same_place_l1605_160524

-- Define the context of the problem
def classSize := 45
def numberOfChoices := 6

-- The proof statement
theorem min_students_same_place : 
  ∃ (n : ℕ), 8 ≤ n ∧ n = Nat.ceil (classSize / numberOfChoices) :=
by
  sorry

end NUMINAMATH_GPT_min_students_same_place_l1605_160524


namespace NUMINAMATH_GPT_symmetric_circle_eq_l1605_160571

theorem symmetric_circle_eq :
  (∃ x y : ℝ, (x + 2)^2 + (y - 1)^2 = 4) :=
by
  sorry

end NUMINAMATH_GPT_symmetric_circle_eq_l1605_160571


namespace NUMINAMATH_GPT_angle_perpendicular_vectors_l1605_160522

theorem angle_perpendicular_vectors (α : ℝ) (h1 : 0 < α) (h2 : α < π)
  (h3 : (1 : ℝ) * Real.sin α + Real.cos α * (1 : ℝ) = 0) : α = 3 * Real.pi / 4 :=
sorry

end NUMINAMATH_GPT_angle_perpendicular_vectors_l1605_160522


namespace NUMINAMATH_GPT_yasna_finish_books_in_two_weeks_l1605_160514

theorem yasna_finish_books_in_two_weeks (pages_book1 : ℕ) (pages_book2 : ℕ) (pages_per_day : ℕ) (days_per_week : ℕ) 
  (h1 : pages_book1 = 180) (h2 : pages_book2 = 100) (h3 : pages_per_day = 20) (h4 : days_per_week = 7) : 
  ((pages_book1 + pages_book2) / pages_per_day) / days_per_week = 2 := 
by
  sorry

end NUMINAMATH_GPT_yasna_finish_books_in_two_weeks_l1605_160514


namespace NUMINAMATH_GPT_leap_years_among_given_years_l1605_160503

-- Definitions for conditions
def is_divisible (a b : Nat) : Prop := b ≠ 0 ∧ a % b = 0

def is_leap_year (y : Nat) : Prop :=
  is_divisible y 4 ∧ (¬ is_divisible y 100 ∨ is_divisible y 400)

-- Statement of the problem
theorem leap_years_among_given_years :
  is_leap_year 1996 ∧ is_leap_year 2036 ∧ (¬ is_leap_year 1700) ∧ (¬ is_leap_year 1998) :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_leap_years_among_given_years_l1605_160503


namespace NUMINAMATH_GPT_min_value_a_plus_8b_min_value_a_plus_8b_min_l1605_160518

theorem min_value_a_plus_8b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a * b = a + 2 * b) :
  a + 8 * b ≥ 9 :=
by sorry

-- The minimum value is 9 (achievable at specific values of a and b)
theorem min_value_a_plus_8b_min (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a * b = a + 2 * b) :
  ∃ a b, a > 0 ∧ b > 0 ∧ 2 * a * b = a + 2 * b ∧ a + 8 * b = 9 :=
by sorry

end NUMINAMATH_GPT_min_value_a_plus_8b_min_value_a_plus_8b_min_l1605_160518


namespace NUMINAMATH_GPT_solve_inequality_and_find_positive_int_solutions_l1605_160579

theorem solve_inequality_and_find_positive_int_solutions :
  ∀ (x : ℝ), (2 * x + 1) / 3 - 1 ≤ (2 / 5) * x → x ≤ 2.5 ∧ ∃ (n : ℕ), n = 1 ∨ n = 2 :=
by
  intro x
  intro h
  sorry

end NUMINAMATH_GPT_solve_inequality_and_find_positive_int_solutions_l1605_160579


namespace NUMINAMATH_GPT_range_of_m_l1605_160526

theorem range_of_m (m: ℝ) : (∀ x : ℝ, x ∈ Set.Icc (-1 : ℝ) (1 : ℝ) → x^2 - x + 1 > 2*x + m) → m < -1 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_range_of_m_l1605_160526


namespace NUMINAMATH_GPT_remainder_division_l1605_160557

theorem remainder_division : ∃ (r : ℕ), 271 = 30 * 9 + r ∧ r = 1 :=
by
  -- Details of the proof would be filled here
  sorry

end NUMINAMATH_GPT_remainder_division_l1605_160557


namespace NUMINAMATH_GPT_units_digit_17_pow_2023_l1605_160562

theorem units_digit_17_pow_2023 : (17^2023 % 10) = 3 := sorry

end NUMINAMATH_GPT_units_digit_17_pow_2023_l1605_160562


namespace NUMINAMATH_GPT_solve_eq_simplify_expression_l1605_160534

-- Part 1: Prove the solution to the given equation

theorem solve_eq (x : ℚ) : (1 / (x - 1) + 1 = 3 / (2 * x - 2)) → x = 3 / 2 :=
sorry

-- Part 2: Prove the simplified value of the given expression when x=1/2

theorem simplify_expression : (x = 1/2) →
  ((x^2 / (1 + x) - x) / ((x^2 - 1) / (x^2 + 2 * x + 1)) = 1) :=
sorry

end NUMINAMATH_GPT_solve_eq_simplify_expression_l1605_160534


namespace NUMINAMATH_GPT_emily_final_lives_l1605_160550

/-- Initial number of lives Emily had. --/
def initialLives : ℕ := 42

/-- Number of lives Emily lost in the hard part of the game. --/
def livesLost : ℕ := 25

/-- Number of lives Emily gained in the next level. --/
def livesGained : ℕ := 24

/-- Final number of lives Emily should have after the changes. --/
def finalLives : ℕ := (initialLives - livesLost) + livesGained

theorem emily_final_lives : finalLives = 41 := by
  /-
  Proof is omitted as per instructions.
  Prove that the final number of lives Emily has is 41.
  -/
  sorry

end NUMINAMATH_GPT_emily_final_lives_l1605_160550


namespace NUMINAMATH_GPT_quadratic_roots_m_eq_2_quadratic_discriminant_pos_l1605_160597

theorem quadratic_roots_m_eq_2 (x : ℝ) (m : ℝ) (h1 : m = 2) : x^2 + 2 * x - 3 = 0 ↔ (x = -3 ∨ x = 1) :=
by sorry

theorem quadratic_discriminant_pos (m : ℝ) : m^2 + 12 > 0 :=
by sorry

end NUMINAMATH_GPT_quadratic_roots_m_eq_2_quadratic_discriminant_pos_l1605_160597


namespace NUMINAMATH_GPT_circle_radius_tangent_l1605_160581

theorem circle_radius_tangent (A B O M X : Type) (AB AM MB r : ℝ)
  (hL1 : AB = 2) (hL2 : AM = 1) (hL3 : MB = 1) (hMX : MX = 1/2)
  (hTangent1 : OX = 1/2 + r) (hTangent2 : OM = 1 - r)
  (hPythagorean : OM^2 + MX^2 = OX^2) :
  r = 1/3 :=
by
  sorry

end NUMINAMATH_GPT_circle_radius_tangent_l1605_160581


namespace NUMINAMATH_GPT_num_integers_between_sqrt10_sqrt100_l1605_160575

theorem num_integers_between_sqrt10_sqrt100 : 
  ∃ n : ℕ, n = 7 ∧ ∀ x : ℤ, (⌊Real.sqrt 10⌋ + 1 <= x) ∧ (x <= ⌈Real.sqrt 100⌉ - 1) ↔ (4 <= x ∧ x <= 10) := 
by 
  sorry

end NUMINAMATH_GPT_num_integers_between_sqrt10_sqrt100_l1605_160575


namespace NUMINAMATH_GPT_speed_of_second_train_l1605_160580

-- Definitions of given conditions
def length_first_train : ℝ := 60 
def length_second_train : ℝ := 280 
def speed_first_train : ℝ := 30 
def time_clear : ℝ := 16.998640108791296 

-- The Lean statement for the proof problem
theorem speed_of_second_train : 
  let relative_distance_km := (length_first_train + length_second_train) / 1000
  let time_clear_hr := time_clear / 3600
  (speed_first_train + (relative_distance_km / time_clear_hr)) = 72.00588235294118 → 
  ∃ V : ℝ, V = 42.00588235294118 :=
by 
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_speed_of_second_train_l1605_160580


namespace NUMINAMATH_GPT_range_of_a_l1605_160508

def discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

def no_real_roots (a b c : ℝ) : Prop :=
  discriminant a b c < 0

theorem range_of_a (a : ℝ) :
  no_real_roots 1 (2 * a - 1) 1 ↔ -1 / 2 < a ∧ a < 3 / 2 := 
by sorry

end NUMINAMATH_GPT_range_of_a_l1605_160508


namespace NUMINAMATH_GPT_fraction_exponentiation_l1605_160561

theorem fraction_exponentiation : (3 / 4) ^ 5 = 243 / 1024 := by
  sorry

end NUMINAMATH_GPT_fraction_exponentiation_l1605_160561


namespace NUMINAMATH_GPT_matrix_operation_correct_l1605_160552

open Matrix

def matrix1 : Matrix (Fin 2) (Fin 2) ℤ := ![![7, -3], ![2, 5]]
def matrix2 : Matrix (Fin 2) (Fin 2) ℤ := ![![1, 4], ![0, -3]]
def matrix3 : Matrix (Fin 2) (Fin 2) ℤ := ![![6, 0], ![-1, 8]]
def result : Matrix (Fin 2) (Fin 2) ℤ := ![![12, -7], ![1, 16]]

theorem matrix_operation_correct:
  matrix1 - matrix2 + matrix3 = result :=
by
  sorry

end NUMINAMATH_GPT_matrix_operation_correct_l1605_160552


namespace NUMINAMATH_GPT_sin_sum_angle_eq_sqrt15_div5_l1605_160577

variable {x : Real}
variable (h1 : 0 < x ∧ x < Real.pi) (h2 : Real.sin (2 * x) = 1 / 5)

theorem sin_sum_angle_eq_sqrt15_div5 : Real.sin (Real.pi / 4 + x) = Real.sqrt 15 / 5 := by
  -- The proof is omitted as instructed.
  sorry

end NUMINAMATH_GPT_sin_sum_angle_eq_sqrt15_div5_l1605_160577


namespace NUMINAMATH_GPT_no_common_solution_general_case_l1605_160592

-- Define the context: three linear equations in two variables
variables {a1 b1 c1 a2 b2 c2 a3 b3 c3 : ℝ}

-- Statement of the theorem
theorem no_common_solution_general_case :
  (∃ (x y : ℝ), a1 * x + b1 * y = c1 ∧ a2 * x + b2 * y = c2 ∧ a3 * x + b3 * y = c3) →
  (a1 * b2 ≠ a2 * b1 ∧ a1 * b3 ≠ a3 * b1 ∧ a2 * b3 ≠ a3 * b2) →
  false := 
sorry

end NUMINAMATH_GPT_no_common_solution_general_case_l1605_160592


namespace NUMINAMATH_GPT_mrs_hilt_more_l1605_160548

-- Define the values of the pennies, nickels, and dimes.
def value_penny : ℝ := 0.01
def value_nickel : ℝ := 0.05
def value_dime : ℝ := 0.10

-- Define the count of coins Mrs. Hilt has.
def mrs_hilt_pennies : ℕ := 2
def mrs_hilt_nickels : ℕ := 2
def mrs_hilt_dimes : ℕ := 2

-- Define the count of coins Jacob has.
def jacob_pennies : ℕ := 4
def jacob_nickels : ℕ := 1
def jacob_dimes : ℕ := 1

-- Calculate the total amount of money Mrs. Hilt has.
def mrs_hilt_total : ℝ :=
  mrs_hilt_pennies * value_penny
  + mrs_hilt_nickels * value_nickel
  + mrs_hilt_dimes * value_dime

-- Calculate the total amount of money Jacob has.
def jacob_total : ℝ :=
  jacob_pennies * value_penny
  + jacob_nickels * value_nickel
  + jacob_dimes * value_dime

-- Prove that Mrs. Hilt has $0.13 more than Jacob.
theorem mrs_hilt_more : mrs_hilt_total - jacob_total = 0.13 := by
  sorry

end NUMINAMATH_GPT_mrs_hilt_more_l1605_160548


namespace NUMINAMATH_GPT_right_regular_prism_impossible_sets_l1605_160511

-- Define a function to check if a given set of numbers {x, y, z} forms an invalid right regular prism
def not_possible (x y z : ℕ) : Prop := (x^2 + y^2 ≤ z^2)

-- Define individual propositions for the given sets of numbers
def set_a : Prop := not_possible 3 4 6
def set_b : Prop := not_possible 5 5 8
def set_e : Prop := not_possible 7 8 12

-- Define our overall proposition that these sets cannot be the lengths of the external diagonals of a right regular prism
theorem right_regular_prism_impossible_sets : 
  set_a ∧ set_b ∧ set_e :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_right_regular_prism_impossible_sets_l1605_160511


namespace NUMINAMATH_GPT_angles_in_order_l1605_160558

-- α1, α2, α3 are real numbers representing the angles of inclination of lines
variable (α1 α2 α3 : ℝ)

-- Conditions given in the problem
axiom tan_α1 : Real.tan α1 = 1
axiom tan_α2 : Real.tan α2 = -1
axiom tan_α3 : Real.tan α3 = -2

-- Theorem to prove
theorem angles_in_order : α1 < α3 ∧ α3 < α2 := 
by
  sorry

end NUMINAMATH_GPT_angles_in_order_l1605_160558


namespace NUMINAMATH_GPT_find_m_symmetry_l1605_160549

theorem find_m_symmetry (A B : ℝ × ℝ) (m : ℝ)
  (hA : A = (-3, m)) (hB : B = (3, 4)) (hy : A.2 = B.2) : m = 4 :=
sorry

end NUMINAMATH_GPT_find_m_symmetry_l1605_160549


namespace NUMINAMATH_GPT_Suzanne_runs_5_kilometers_l1605_160556

theorem Suzanne_runs_5_kilometers 
  (a : ℕ) 
  (r : ℕ) 
  (total_donation : ℕ) 
  (n : ℕ)
  (h1 : a = 10) 
  (h2 : r = 2) 
  (h3 : total_donation = 310) 
  (h4 : total_donation = a * (1 - r^n) / (1 - r)) 
  : n = 5 :=
by
  sorry

end NUMINAMATH_GPT_Suzanne_runs_5_kilometers_l1605_160556


namespace NUMINAMATH_GPT_number_of_pencils_l1605_160559

theorem number_of_pencils (E P : ℕ) (h1 : E + P = 8) (h2 : 300 * E + 500 * P = 3000) (hE : E ≥ 1) (hP : P ≥ 1) : P = 3 :=
by
  sorry

end NUMINAMATH_GPT_number_of_pencils_l1605_160559


namespace NUMINAMATH_GPT_maximum_value_frac_l1605_160589

-- Let x and y be positive real numbers. Prove that (x + y)^3 / (x^3 + y^3) ≤ 4.
theorem maximum_value_frac (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  (x + y)^3 / (x^3 + y^3) ≤ 4 := sorry

end NUMINAMATH_GPT_maximum_value_frac_l1605_160589


namespace NUMINAMATH_GPT_sum_of_areas_of_tangent_circles_l1605_160543

theorem sum_of_areas_of_tangent_circles
  (r s t : ℝ)
  (h1 : r + s = 6)
  (h2 : s + t = 8)
  (h3 : r + t = 10) :
  π * (r^2 + s^2 + t^2) = 56 * π :=
by
  sorry

end NUMINAMATH_GPT_sum_of_areas_of_tangent_circles_l1605_160543


namespace NUMINAMATH_GPT_ticket_distribution_count_l1605_160535

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

end NUMINAMATH_GPT_ticket_distribution_count_l1605_160535


namespace NUMINAMATH_GPT_evaluate_using_horners_method_l1605_160585

def f (x : ℝ) : ℝ := 3 * x^6 + 12 * x^5 + 8 * x^4 - 3.5 * x^3 + 7.2 * x^2 + 5 * x - 13

theorem evaluate_using_horners_method :
  f 6 = 243168.2 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_using_horners_method_l1605_160585


namespace NUMINAMATH_GPT_union_of_A_and_B_l1605_160588

open Set -- to use set notation and operations

def A : Set ℝ := { x | -1/2 < x ∧ x < 2 }

def B : Set ℝ := { x | x^2 ≤ 1 }

theorem union_of_A_and_B :
  A ∪ B = Ico (-1:ℝ) 2 := 
by
  -- proof steps would go here, but we skip these with sorry.
  sorry

end NUMINAMATH_GPT_union_of_A_and_B_l1605_160588


namespace NUMINAMATH_GPT_units_digits_no_match_l1605_160515

theorem units_digits_no_match : ∀ x : ℕ, 1 ≤ x ∧ x ≤ 100 → (x % 10 ≠ (101 - x) % 10) :=
by
  intro x hx
  sorry

end NUMINAMATH_GPT_units_digits_no_match_l1605_160515


namespace NUMINAMATH_GPT_total_people_on_hike_l1605_160598

theorem total_people_on_hike
  (cars : ℕ) (cars_people : ℕ)
  (taxis : ℕ) (taxis_people : ℕ)
  (vans : ℕ) (vans_people : ℕ)
  (buses : ℕ) (buses_people : ℕ)
  (minibuses : ℕ) (minibuses_people : ℕ)
  (h_cars : cars = 7) (h_cars_people : cars_people = 4)
  (h_taxis : taxis = 10) (h_taxis_people : taxis_people = 6)
  (h_vans : vans = 4) (h_vans_people : vans_people = 5)
  (h_buses : buses = 3) (h_buses_people : buses_people = 20)
  (h_minibuses : minibuses = 2) (h_minibuses_people : minibuses_people = 8) :
  cars * cars_people + taxis * taxis_people + vans * vans_people + buses * buses_people + minibuses * minibuses_people = 184 :=
by
  sorry

end NUMINAMATH_GPT_total_people_on_hike_l1605_160598


namespace NUMINAMATH_GPT_simplification_evaluation_l1605_160530

noncomputable def simplify_and_evaluate (x : ℤ) : ℚ :=
  (1 - 1 / (x - 1)) * ((x - 1) / ((x - 2) * (x - 2)))

theorem simplification_evaluation (x : ℤ) (h1 : x > 0) (h2 : 3 - x ≥ 0) : 
  simplify_and_evaluate x = 1 :=
by
  have h3 : x = 3 := sorry
  rw [simplify_and_evaluate, h3]
  simp [h3]
  sorry

end NUMINAMATH_GPT_simplification_evaluation_l1605_160530


namespace NUMINAMATH_GPT_sum_of_squares_l1605_160584

theorem sum_of_squares (x y : ℝ) (h1 : x + y = 18) (h2 : x * y = 52) : x^2 + y^2 = 220 :=
sorry

end NUMINAMATH_GPT_sum_of_squares_l1605_160584


namespace NUMINAMATH_GPT_greatest_price_book_l1605_160516

theorem greatest_price_book (p : ℕ) (B : ℕ) (D : ℕ) (F : ℕ) (T : ℚ) 
  (h1 : B = 20) 
  (h2 : D = 200) 
  (h3 : F = 5)
  (h4 : T = 0.07) 
  (h5 : ∀ p, 20 * p * (1 + T) ≤ (D - F)) : 
  p ≤ 9 :=
by
  sorry

end NUMINAMATH_GPT_greatest_price_book_l1605_160516


namespace NUMINAMATH_GPT_grains_in_one_tsp_l1605_160502

-- Definitions based on conditions
def grains_in_one_cup : Nat := 480
def half_cup_is_8_tbsp : Nat := 8
def one_tbsp_is_3_tsp : Nat := 3

-- Theorem statement
theorem grains_in_one_tsp :
  let grains_in_half_cup := grains_in_one_cup / 2
  let grains_in_one_tbsp := grains_in_half_cup / half_cup_is_8_tbsp
  grains_in_one_tbsp / one_tbsp_is_3_tsp = 10 :=
by
  sorry

end NUMINAMATH_GPT_grains_in_one_tsp_l1605_160502


namespace NUMINAMATH_GPT_suff_not_nec_condition_l1605_160531

/-- f is an even function --/
def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

/-- Condition x1 + x2 = 0 --/
def sum_eq_zero (x1 x2 : ℝ) : Prop := x1 + x2 = 0

/-- Prove: sufficient but not necessary condition --/
theorem suff_not_nec_condition (f : ℝ → ℝ) (h_even : is_even f) (x1 x2 : ℝ) :
  sum_eq_zero x1 x2 → f x1 - f x2 = 0 ∧ (f x1 - f x2 = 0 → ¬ sum_eq_zero x1 x2) :=
by
  sorry

end NUMINAMATH_GPT_suff_not_nec_condition_l1605_160531


namespace NUMINAMATH_GPT_shaniqua_styles_count_l1605_160538

variable (S : ℕ)

def shaniqua_haircuts (haircuts : ℕ) : ℕ := 12 * haircuts
def shaniqua_styles (styles : ℕ) : ℕ := 25 * styles

theorem shaniqua_styles_count (total_money haircuts : ℕ) (styles : ℕ) :
  total_money = shaniqua_haircuts haircuts + shaniqua_styles styles → haircuts = 8 → total_money = 221 → S = 5 :=
by
  sorry

end NUMINAMATH_GPT_shaniqua_styles_count_l1605_160538


namespace NUMINAMATH_GPT_quad_condition_l1605_160509

noncomputable def quadratic (a : ℝ) (x : ℝ) : ℝ := x^2 + a * x - 4 * a

theorem quad_condition (a : ℝ) : (-16 ≤ a ∧ a ≤ 0) → (∀ x : ℝ, quadratic a x > 0) ↔ (¬ ∃ x : ℝ, quadratic a x ≤ 0) := by
  sorry

end NUMINAMATH_GPT_quad_condition_l1605_160509


namespace NUMINAMATH_GPT_average_height_of_trees_l1605_160573

theorem average_height_of_trees :
  ∃ (h : ℕ → ℕ), (h 2 = 12) ∧ (∀ i, h i = 2 * h (i+1) ∨ h i = h (i+1) / 2) ∧ (h 1 * h 2 * h 3 * h 4 * h 5 * h 6 = 4608) →
  (h 1 + h 2 + h 3 + h 4 + h 5 + h 6) / 6 = 21 :=
sorry

end NUMINAMATH_GPT_average_height_of_trees_l1605_160573


namespace NUMINAMATH_GPT_number_of_integer_values_of_x_l1605_160560

theorem number_of_integer_values_of_x (x : ℕ) (hx1 : x + 15 > 40) (hx2 : x + 40 > 15) (hx3 : 15 + 40 > x) :
  ∃ n : ℕ, n = 29 ∧ ∀ y : ℕ, (26 ≤ y ∧ y ≤ 54) ↔ true :=
by
  sorry

end NUMINAMATH_GPT_number_of_integer_values_of_x_l1605_160560


namespace NUMINAMATH_GPT_functions_same_function_C_functions_same_function_D_l1605_160583

theorem functions_same_function_C (x : ℝ) : (x^2) = (x^6)^(1/3) :=
by sorry

theorem functions_same_function_D (x : ℝ) : x = (x^3)^(1/3) :=
by sorry

end NUMINAMATH_GPT_functions_same_function_C_functions_same_function_D_l1605_160583


namespace NUMINAMATH_GPT_cost_of_each_toy_l1605_160510

theorem cost_of_each_toy (initial_money spent_money remaining_money toys_count toy_cost : ℕ) 
  (h1 : initial_money = 57)
  (h2 : spent_money = 27)
  (h3 : remaining_money = initial_money - spent_money)
  (h4 : toys_count = 5)
  (h5 : remaining_money / toys_count = toy_cost) :
  toy_cost = 6 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_each_toy_l1605_160510


namespace NUMINAMATH_GPT_find_c_l1605_160500

theorem find_c (a b c d y1 y2 : ℝ) (h1 : y1 = a * 2^3 + b * 2^2 + c * 2 + d)
  (h2 : y2 = a * (-2)^3 + b * (-2)^2 + c * (-2) + d)
  (h3 : y1 - y2 = 12) : c = 3 - 4 * a := by
  sorry

end NUMINAMATH_GPT_find_c_l1605_160500


namespace NUMINAMATH_GPT_geometric_sequence_term_l1605_160574

theorem geometric_sequence_term 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h_seq : ∀ n, a (n+1) = a n * q)
  (h_a2 : a 2 = 8) 
  (h_a5 : a 5 = 64) : 
  a 3 = 16 := 
by 
  sorry

end NUMINAMATH_GPT_geometric_sequence_term_l1605_160574


namespace NUMINAMATH_GPT_zero_in_interval_l1605_160533

noncomputable def f (x : ℝ) : ℝ := Real.log (3 * x / 2) - 2 / x

theorem zero_in_interval :
  (Real.log (3 / 2) - 2 < 0) ∧ (Real.log 3 - 2 / 3 > 0) →
  ∃ x : ℝ, 1 < x ∧ x < 2 ∧ f x = 0 :=
by
  -- conditions from the problem statement
  intros h
  -- proving the result
  sorry

end NUMINAMATH_GPT_zero_in_interval_l1605_160533


namespace NUMINAMATH_GPT_inspection_probability_l1605_160527

noncomputable def defective_items : ℕ := 2
noncomputable def good_items : ℕ := 3
noncomputable def total_items : ℕ := defective_items + good_items

/-- Given 2 defective items and 3 good items mixed together,
the probability that the inspection stops exactly after
four inspections is 3/5 --/
theorem inspection_probability :
  (2 * (total_items - 1) * total_items / (total_items * (total_items - 1) * (total_items - 2) * (total_items - 3))) = (3 / 5) :=
by
  sorry

end NUMINAMATH_GPT_inspection_probability_l1605_160527


namespace NUMINAMATH_GPT_garden_length_l1605_160564

theorem garden_length (columns : ℕ) (distance_between_trees : ℕ) (boundary_distance : ℕ) (h_columns : columns = 12) (h_distance_between_trees : distance_between_trees = 2) (h_boundary_distance : boundary_distance = 5) : 
  ((columns - 1) * distance_between_trees + 2 * boundary_distance) = 32 :=
by 
  sorry

end NUMINAMATH_GPT_garden_length_l1605_160564


namespace NUMINAMATH_GPT_average_weight_of_girls_l1605_160512

theorem average_weight_of_girls :
  ∀ (total_students boys girls total_weight class_average_weight boys_average_weight girls_average_weight : ℝ),
  total_students = 25 →
  boys = 15 →
  girls = 10 →
  boys + girls = total_students →
  class_average_weight = 45 →
  boys_average_weight = 48 →
  total_weight = 1125 →
  girls_average_weight = (total_weight - (boys * boys_average_weight)) / girls →
  total_weight = class_average_weight * total_students →
  girls_average_weight = 40.5 :=
by
  intros total_students boys girls total_weight class_average_weight boys_average_weight girls_average_weight
  sorry

end NUMINAMATH_GPT_average_weight_of_girls_l1605_160512


namespace NUMINAMATH_GPT_abs_m_minus_n_eq_five_l1605_160523

theorem abs_m_minus_n_eq_five (m n : ℝ) (h₁ : m * n = 6) (h₂ : m + n = 7) : |m - n| = 5 :=
sorry

end NUMINAMATH_GPT_abs_m_minus_n_eq_five_l1605_160523


namespace NUMINAMATH_GPT_prove_students_second_and_third_l1605_160547

namespace MonicaClasses

def Monica := 
  let classes_per_day := 6
  let students_first_class := 20
  let students_fourth_class := students_first_class / 2
  let students_fifth_class := 28
  let students_sixth_class := 28
  let total_students := 136
  let known_students := students_first_class + students_fourth_class + students_fifth_class + students_sixth_class
  let students_second_and_third := total_students - known_students
  students_second_and_third = 50

theorem prove_students_second_and_third : Monica :=
  by
    sorry

end MonicaClasses

end NUMINAMATH_GPT_prove_students_second_and_third_l1605_160547


namespace NUMINAMATH_GPT_mary_finds_eggs_l1605_160520

theorem mary_finds_eggs (initial final found : ℕ) (h_initial : initial = 27) (h_final : final = 31) :
  found = final - initial → found = 4 :=
by
  intro h
  rw [h_initial, h_final] at h
  exact h

end NUMINAMATH_GPT_mary_finds_eggs_l1605_160520


namespace NUMINAMATH_GPT_radius_of_circle_l1605_160544

-- Definitions based on conditions
def center_in_first_quadrant (C : ℝ × ℝ) : Prop :=
  C.1 > 0 ∧ C.2 > 0

def intersects_x_axis (C : ℝ × ℝ) (r : ℝ) : Prop :=
  r = Real.sqrt ((C.1 - 1)^2 + (C.2)^2) ∧ r = Real.sqrt ((C.1 - 3)^2 + (C.2)^2)

def tangent_to_line (C : ℝ × ℝ) (r : ℝ) : Prop :=
  r = abs (C.1 - C.2 + 1) / Real.sqrt 2

-- Main statement
theorem radius_of_circle (C : ℝ × ℝ) (r : ℝ) 
  (h1 : center_in_first_quadrant C)
  (h2 : intersects_x_axis C r)
  (h3 : tangent_to_line C r) : 
  r = Real.sqrt 2 := 
sorry

end NUMINAMATH_GPT_radius_of_circle_l1605_160544


namespace NUMINAMATH_GPT_jump_rope_difference_l1605_160569

noncomputable def cindy_jump_time : ℕ := 12
noncomputable def betsy_jump_time : ℕ := cindy_jump_time / 2
noncomputable def tina_jump_time : ℕ := 3 * betsy_jump_time

theorem jump_rope_difference : tina_jump_time - cindy_jump_time = 6 :=
by
  -- proof steps would go here
  sorry

end NUMINAMATH_GPT_jump_rope_difference_l1605_160569


namespace NUMINAMATH_GPT_usual_time_eight_l1605_160540

/-- Define the parameters used in the problem -/
def usual_speed (S : ℝ) : ℝ := S
def usual_time (T : ℝ) : ℝ := T
def reduced_speed (S : ℝ) := 0.25 * S
def reduced_time (T : ℝ) := T + 24

/-- The main theorem that we need to prove -/
theorem usual_time_eight (S T : ℝ) 
  (h1 : usual_speed S = S)
  (h2 : usual_time T = T)
  (h3 : reduced_speed S = 0.25 * S)
  (h4 : reduced_time T = T + 24)
  (h5 : S / (0.25 * S) = (T + 24) / T) : T = 8 :=
by 
  sorry -- Proof omitted for brevity. Refers to the solution steps.


end NUMINAMATH_GPT_usual_time_eight_l1605_160540


namespace NUMINAMATH_GPT_range_of_k_l1605_160565

noncomputable def quadratic_has_real_roots (k : ℝ) :=
  ∃ (x : ℝ), (k - 3) * x^2 - 4 * x + 2 = 0

theorem range_of_k (k : ℝ) : quadratic_has_real_roots k ↔ k ≤ 5 := 
  sorry

end NUMINAMATH_GPT_range_of_k_l1605_160565


namespace NUMINAMATH_GPT_smaller_area_l1605_160551

theorem smaller_area (A B : ℝ) (total_area : A + B = 1800) (diff_condition : B - A = (A + B) / 6) :
  A = 750 := 
by
  sorry

end NUMINAMATH_GPT_smaller_area_l1605_160551


namespace NUMINAMATH_GPT_zero_a_and_b_l1605_160582

theorem zero_a_and_b (a b : ℝ) (h : a^2 + |b| = 0) : a = 0 ∧ b = 0 :=
by
  sorry

end NUMINAMATH_GPT_zero_a_and_b_l1605_160582


namespace NUMINAMATH_GPT_kernels_popped_in_final_bag_l1605_160595

/-- Parker wants to find out what the average percentage of kernels that pop in a bag is.
In the first bag he makes, 60 kernels pop and the bag has 75 kernels.
In the second bag, 42 kernels pop and there are 50 in the bag.
In the final bag, some kernels pop and the bag has 100 kernels.
The average percentage of kernels that pop in a bag is 82%.
How many kernels popped in the final bag?
We prove that given these conditions, the number of popped kernels in the final bag is 82.
-/
noncomputable def kernelsPoppedInFirstBag := 60
noncomputable def totalKernelsInFirstBag := 75
noncomputable def kernelsPoppedInSecondBag := 42
noncomputable def totalKernelsInSecondBag := 50
noncomputable def totalKernelsInFinalBag := 100
noncomputable def averagePoppedPercentage := 82

theorem kernels_popped_in_final_bag (x : ℕ) :
  (kernelsPoppedInFirstBag * 100 / totalKernelsInFirstBag +
   kernelsPoppedInSecondBag * 100 / totalKernelsInSecondBag +
   x * 100 / totalKernelsInFinalBag) / 3 = averagePoppedPercentage →
  x = 82 := 
by
  sorry

end NUMINAMATH_GPT_kernels_popped_in_final_bag_l1605_160595


namespace NUMINAMATH_GPT_domain_of_log_sqrt_l1605_160517

theorem domain_of_log_sqrt (x : ℝ) : (-1 < x ∧ x ≤ 3) ↔ (0 < x + 1 ∧ 3 - x ≥ 0) :=
by
  sorry

end NUMINAMATH_GPT_domain_of_log_sqrt_l1605_160517


namespace NUMINAMATH_GPT_floor_equation_solution_l1605_160521

open Int

theorem floor_equation_solution (x : ℝ) :
  (⌊ ⌊ 3 * x ⌋ - 1/2 ⌋ = ⌊ x + 4 ⌋) ↔ (7/3 ≤ x ∧ x < 3) := sorry

end NUMINAMATH_GPT_floor_equation_solution_l1605_160521


namespace NUMINAMATH_GPT_altered_solution_contains_60_liters_of_detergent_l1605_160528

-- Definitions corresponding to the conditions
def initial_ratio_bleach_to_detergent_to_water : ℚ := 2 / 40 / 100
def initial_ratio_bleach_to_detergent : ℚ := 1 / 20
def initial_ratio_detergent_to_water : ℚ := 1 / 5

def altered_ratio_bleach_to_detergent : ℚ := 3 / 20
def altered_ratio_detergent_to_water : ℚ := 1 / 5

def water_in_altered_solution : ℚ := 300

-- We need to find the amount of detergent in the altered solution
def amount_of_detergent_in_altered_solution : ℚ := 20

-- The proportion and the final amount calculation
theorem altered_solution_contains_60_liters_of_detergent :
  (300 / 100) * (20) = 60 :=
by
  sorry

end NUMINAMATH_GPT_altered_solution_contains_60_liters_of_detergent_l1605_160528


namespace NUMINAMATH_GPT_largest_three_digit_multiple_of_6_sum_15_l1605_160546

-- Statement of the problem in Lean
theorem largest_three_digit_multiple_of_6_sum_15 : 
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 6 = 0 ∧ (Nat.digits 10 n).sum = 15 ∧ 
  ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 6 = 0 ∧ (Nat.digits 10 m).sum = 15 → m ≤ n :=
by
  sorry -- proof not required

end NUMINAMATH_GPT_largest_three_digit_multiple_of_6_sum_15_l1605_160546


namespace NUMINAMATH_GPT_last_digit_7_powers_l1605_160599

theorem last_digit_7_powers :
  (∃ n : ℕ, (∀ k < 4004, k.mod 2002 == n))
  := sorry

end NUMINAMATH_GPT_last_digit_7_powers_l1605_160599


namespace NUMINAMATH_GPT_function_intersects_y_axis_at_0_neg4_l1605_160563

theorem function_intersects_y_axis_at_0_neg4 :
  (∃ x y : ℝ, y = 4 * x - 4 ∧ x = 0 ∧ y = -4) :=
sorry

end NUMINAMATH_GPT_function_intersects_y_axis_at_0_neg4_l1605_160563


namespace NUMINAMATH_GPT_find_a_5_in_arithmetic_sequence_l1605_160525

variable {a : ℕ → ℕ}

def arithmetic_sequence (a : ℕ → ℕ) (a1 d : ℕ) : Prop :=
  a 1 = a1 ∧ ∀ n, a (n + 1) = a n + d

theorem find_a_5_in_arithmetic_sequence (h : arithmetic_sequence a 1 2) : a 5 = 9 :=
sorry

end NUMINAMATH_GPT_find_a_5_in_arithmetic_sequence_l1605_160525


namespace NUMINAMATH_GPT_find_correct_day_l1605_160567

def tomorrow_is_not_September (d : String) : Prop :=
  d ≠ "September"

def in_a_week_is_September (d : String) : Prop :=
  d = "September"

def day_after_tomorrow_is_not_Wednesday (d : String) : Prop :=
  d ≠ "Wednesday"

theorem find_correct_day :
    ((∀ d, tomorrow_is_not_September d) ∧ 
    (∀ d, in_a_week_is_September d) ∧ 
    (∀ d, day_after_tomorrow_is_not_Wednesday d)) → 
    "Wednesday, August 25" = "Wednesday, August 25" :=
by
sorry

end NUMINAMATH_GPT_find_correct_day_l1605_160567


namespace NUMINAMATH_GPT_arithmetic_sequence_common_difference_l1605_160506

theorem arithmetic_sequence_common_difference (a : ℕ → ℤ) (d : ℤ) (h_arith_seq: ∀ n, a n = a 1 + (n - 1) * d) 
  (h_cond1 : a 3 + a 9 = 4 * a 5) (h_cond2 : a 2 = -8) : 
  d = 4 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_common_difference_l1605_160506


namespace NUMINAMATH_GPT_Beto_can_determine_xy_l1605_160539

theorem Beto_can_determine_xy (m n : ℤ) :
  (∃ k t : ℤ, 0 < t ∧ m = 2 * k + 1 ∧ n = 2 * t * (2 * k + 1)) ↔ 
  (∀ x y : ℝ, (∃ a b : ℝ, a ≠ b ∧ x = a ∧ y = b) →
    ∃ xy_val : ℝ, (x^m + y^m = xy_val) ∧ (x^n + y^n = xy_val)) := 
sorry

end NUMINAMATH_GPT_Beto_can_determine_xy_l1605_160539


namespace NUMINAMATH_GPT_james_writes_pages_per_hour_l1605_160555

theorem james_writes_pages_per_hour (hours_per_night : ℕ) (days_per_week : ℕ) (weeks : ℕ) (total_pages : ℕ) (total_hours : ℕ) :
  hours_per_night = 3 → 
  days_per_week = 7 → 
  weeks = 7 → 
  total_pages = 735 → 
  total_hours = 147 → 
  total_hours = hours_per_night * days_per_week * weeks → 
  total_pages / total_hours = 5 :=
by sorry

end NUMINAMATH_GPT_james_writes_pages_per_hour_l1605_160555


namespace NUMINAMATH_GPT_no_snow_three_days_l1605_160576

noncomputable def probability_no_snow_first_two_days : ℚ := 1 - 2/3
noncomputable def probability_no_snow_third_day : ℚ := 1 - 3/5

theorem no_snow_three_days : 
  (probability_no_snow_first_two_days) * 
  (probability_no_snow_first_two_days) * 
  (probability_no_snow_third_day) = 2/45 :=
by
  sorry

end NUMINAMATH_GPT_no_snow_three_days_l1605_160576


namespace NUMINAMATH_GPT_coin_heads_probability_l1605_160519

theorem coin_heads_probability
    (prob_tails : ℚ := 1/2)
    (prob_specific_sequence : ℚ := 0.0625)
    (flips : ℕ := 4)
    (ht : prob_tails = 1 / 2)
    (hs : prob_specific_sequence = (1 / 2) * (1 / 2) * (1 / 2) * (1 / 2)) 
    : ∀ (p_heads : ℚ), p_heads = 1 - prob_tails := by
  sorry

end NUMINAMATH_GPT_coin_heads_probability_l1605_160519


namespace NUMINAMATH_GPT_Jake_has_62_balls_l1605_160541

theorem Jake_has_62_balls 
  (C A J : ℕ)
  (h1 : C = 41 + 7)
  (h2 : A = 2 * C)
  (h3 : J = A - 34) : 
  J = 62 :=
by 
  sorry

end NUMINAMATH_GPT_Jake_has_62_balls_l1605_160541


namespace NUMINAMATH_GPT_county_population_percentage_l1605_160532

theorem county_population_percentage 
    (percent_less_than_20000 : ℝ)
    (percent_20000_to_49999 : ℝ) 
    (h1 : percent_less_than_20000 = 35) 
    (h2 : percent_20000_to_49999 = 40) : 
    percent_less_than_20000 + percent_20000_to_49999 = 75 := 
by
  sorry

end NUMINAMATH_GPT_county_population_percentage_l1605_160532


namespace NUMINAMATH_GPT_p_interval_satisfies_inequality_l1605_160572

theorem p_interval_satisfies_inequality :
  ∀ (p q : ℝ), 0 ≤ p ∧ p < 2.232 ∧ q > 0 ∧ p + q ≠ 0 →
    (4 * (p * q ^ 2 + p ^ 2 * q + 4 * q ^ 2 + 4 * p * q)) / (p + q) > 5 * p ^ 2 * q :=
by sorry

end NUMINAMATH_GPT_p_interval_satisfies_inequality_l1605_160572


namespace NUMINAMATH_GPT_measure_exactly_10_liters_l1605_160566

theorem measure_exactly_10_liters (A B : ℕ) (A_cap B_cap : ℕ) (hA : A_cap = 11) (hB : B_cap = 9) :
  ∃ (A B : ℕ), A + B = 10 ∧ A ≤ A_cap ∧ B ≤ B_cap := 
sorry

end NUMINAMATH_GPT_measure_exactly_10_liters_l1605_160566


namespace NUMINAMATH_GPT_abcd_mod_7_zero_l1605_160536

theorem abcd_mod_7_zero
  (a b c d : ℕ)
  (h1 : a + 2 * b + 3 * c + 4 * d ≡ 1 [MOD 7])
  (h2 : 2 * a + 3 * b + c + 2 * d ≡ 5 [MOD 7])
  (h3 : 3 * a + b + 2 * c + 3 * d ≡ 3 [MOD 7])
  (h4 : 4 * a + 2 * b + d + c ≡ 2 [MOD 7])
  (ha : a < 7) (hb : b < 7) (hc : c < 7) (hd : d < 7) :
  (a * b * c * d) % 7 = 0 :=
by sorry

end NUMINAMATH_GPT_abcd_mod_7_zero_l1605_160536


namespace NUMINAMATH_GPT_geometric_progression_ineq_l1605_160545

variable (q b₁ b₂ b₃ b₄ b₅ b₆ : ℝ)

-- Condition: \(b_n\) is an increasing positive geometric progression
-- \( q > 1 \) because the progression is increasing
variable (q_pos : q > 1) 

-- Recursive definitions for the geometric progression
variable (geom_b₂ : b₂ = b₁ * q)
variable (geom_b₃ : b₃ = b₁ * q^2)
variable (geom_b₄ : b₄ = b₁ * q^3)
variable (geom_b₅ : b₅ = b₁ * q^4)
variable (geom_b₆ : b₆ = b₁ * q^5)

-- Given condition from the problem
variable (condition : b₄ + b₃ - b₂ - b₁ = 5)

-- Statement to prove
theorem geometric_progression_ineq (q b₁ b₂ b₃ b₄ b₅ b₆ : ℝ) 
  (q_pos : q > 1) 
  (geom_b₂ : b₂ = b₁ * q)
  (geom_b₃ : b₃ = b₁ * q^2)
  (geom_b₄ : b₄ = b₁ * q^3)
  (geom_b₅ : b₅ = b₁ * q^4)
  (geom_b₆ : b₆ = b₁ * q^5)
  (condition : b₃ + b₄ - b₂ - b₁ = 5) : b₆ + b₅ ≥ 20 := by
    sorry

end NUMINAMATH_GPT_geometric_progression_ineq_l1605_160545


namespace NUMINAMATH_GPT_vertical_asymptotes_sum_l1605_160594

theorem vertical_asymptotes_sum (A B C : ℤ)
  (h : ∀ x : ℝ, x = -1 ∨ x = 2 ∨ x = 3 → x^3 + A * x^2 + B * x + C = 0)
  : A + B + C = -3 :=
sorry

end NUMINAMATH_GPT_vertical_asymptotes_sum_l1605_160594


namespace NUMINAMATH_GPT_rhombus_diagonal_l1605_160553

theorem rhombus_diagonal (d1 d2 : ℝ) (area : ℝ) (h1 : d1 = 14) (h2 : area = 126) (h3 : area = (d1 * d2) / 2) : d2 = 18 := 
by
  -- h1, h2, and h3 are the conditions
  sorry

end NUMINAMATH_GPT_rhombus_diagonal_l1605_160553


namespace NUMINAMATH_GPT_max_k_no_real_roots_l1605_160542

theorem max_k_no_real_roots : ∀ k : ℤ, (∀ x : ℝ, x^2 - 2 * x - (k : ℝ) ≠ 0) → k ≤ -2 :=
by
  sorry

end NUMINAMATH_GPT_max_k_no_real_roots_l1605_160542


namespace NUMINAMATH_GPT_percent_games_lost_l1605_160513

theorem percent_games_lost
  (w l t : ℕ)
  (h_ratio : 7 * l = 3 * w)
  (h_tied : t = 5) :
  (l : ℝ) / (w + l + t) * 100 = 20 :=
by
  sorry

end NUMINAMATH_GPT_percent_games_lost_l1605_160513


namespace NUMINAMATH_GPT_percentage_of_part_whole_l1605_160570

theorem percentage_of_part_whole (part whole : ℝ) (h_part : part = 75) (h_whole : whole = 125) : 
  (part / whole) * 100 = 60 :=
by
  rw [h_part, h_whole]
  -- Simplification steps would follow, but we substitute in the placeholders
  sorry

end NUMINAMATH_GPT_percentage_of_part_whole_l1605_160570


namespace NUMINAMATH_GPT_geometric_sequence_sum_l1605_160505

theorem geometric_sequence_sum 
  (a : ℕ → ℝ) 
  (h_geo : ∀ n, a (n + 1) = (3 : ℝ) * ((-2 : ℝ) ^ n))
  (h_first : a 1 = 3)
  (h_ratio_ne_1 : -2 ≠ 1)
  (h_arith : 2 * a 3 = a 4 + a 5) :
  a 1 + a 2 + a 3 + a 4 + a 5 = 33 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l1605_160505


namespace NUMINAMATH_GPT_interest_rate_calculation_l1605_160593

theorem interest_rate_calculation :
  let P := 1599.9999999999998
  let A := 1792
  let T := 2 + 2 / 5
  let I := A - P
  I / (P * T) = 0.05 :=
  sorry

end NUMINAMATH_GPT_interest_rate_calculation_l1605_160593


namespace NUMINAMATH_GPT_determine_x_l1605_160507

variables {m n x : ℝ}
variable (k : ℝ)
variable (Hmn : m ≠ 0 ∧ n ≠ 0)
variable (Hk : k = 5 * (m^2 - n^2))

theorem determine_x (H : (x + 2 * m)^2 - (x - 3 * n)^2 = k) : 
  x = (5 * m^2 - 9 * n^2) / (4 * m + 6 * n) := by
  sorry

end NUMINAMATH_GPT_determine_x_l1605_160507


namespace NUMINAMATH_GPT_determine_unique_row_weight_free_l1605_160596

theorem determine_unique_row_weight_free (t : ℝ) (rows : Fin 10 → ℝ) (unique_row : Fin 10)
  (h_weights_same : ∀ i : Fin 10, i ≠ unique_row → rows i = t) :
  0 = 0 := by
  sorry

end NUMINAMATH_GPT_determine_unique_row_weight_free_l1605_160596


namespace NUMINAMATH_GPT_reflected_ray_bisects_circle_circumference_l1605_160537

open Real

noncomputable def equation_of_line_reflected_ray : Prop :=
  ∃ (m b : ℝ), (m = 2 / (-3 + 1)) ∧ (b = (3/(-5 + 5)) + 1) ∧ ((-5, -3) = (-5, (-5*m + b))) ∧ ((1, 1) = (1, (1*m + b)))

theorem reflected_ray_bisects_circle_circumference :
  equation_of_line_reflected_ray ↔ ∃ a b c : ℝ, (a = 2) ∧ (b = -3) ∧ (c = 1) ∧ (a*x + b*y + c = 0) :=
by
  sorry

end NUMINAMATH_GPT_reflected_ray_bisects_circle_circumference_l1605_160537


namespace NUMINAMATH_GPT_perpendicular_vecs_l1605_160586

open Real

def vec_a : ℝ × ℝ := (1, 3)
def vec_b : ℝ × ℝ := (3, 4)
def lambda := 1 / 2

theorem perpendicular_vecs : 
  let vec_diff := (vec_a.1 - lambda * vec_b.1, vec_a.2 - lambda * vec_b.2)
  let vec_opp := (vec_b.1 - vec_a.1, vec_b.2 - vec_a.2)
  (vec_diff.1 * vec_opp.1 + vec_diff.2 * vec_opp.2) = 0 := 
by 
  let vec_diff := (vec_a.1 - lambda * vec_b.1, vec_a.2 - lambda * vec_b.2)
  let vec_opp := (vec_b.1 - vec_a.1, vec_b.2 - vec_a.2)
  show (vec_diff.1 * vec_opp.1 + vec_diff.2 * vec_opp.2) = 0
  sorry

end NUMINAMATH_GPT_perpendicular_vecs_l1605_160586
