import Mathlib

namespace output_correct_l731_73153

-- Definitions derived from the conditions
def initial_a : Nat := 3
def initial_b : Nat := 4

-- Proof that the final output of PRINT a, b is (4, 4)
theorem output_correct : 
  let a := initial_a;
  let b := initial_b;
  let a := b;
  let b := a;
  (a, b) = (4, 4) :=
by
  sorry

end output_correct_l731_73153


namespace isosceles_triangle_base_angle_l731_73148

theorem isosceles_triangle_base_angle (α : ℕ) (base_angle : ℕ) 
  (hα : α = 40) (hsum : α + 2 * base_angle = 180) : 
  base_angle = 70 :=
sorry

end isosceles_triangle_base_angle_l731_73148


namespace rectangle_circle_area_ratio_l731_73141

theorem rectangle_circle_area_ratio {d : ℝ} (h : d > 0) :
  let A_rectangle := 2 * d * d
  let A_circle := (π * d^2) / 4
  (A_rectangle / A_circle) = (8 / π) :=
by
  sorry

end rectangle_circle_area_ratio_l731_73141


namespace mean_temperature_correct_l731_73116

-- Define the list of temperatures
def temperatures : List ℤ := [-8, -5, -5, -6, 0, 4]

-- Define the mean temperature calculation
def mean_temperature (temps: List ℤ) : ℚ :=
  (temps.sum : ℚ) / temps.length

-- The theorem we want to prove
theorem mean_temperature_correct :
  mean_temperature temperatures = -10 / 3 :=
by
  sorry

end mean_temperature_correct_l731_73116


namespace rational_expression_nonnegative_l731_73159

theorem rational_expression_nonnegative (x : ℚ) : 2 * |x| + x ≥ 0 :=
  sorry

end rational_expression_nonnegative_l731_73159


namespace graph_movement_l731_73161

noncomputable def f (x : ℝ) : ℝ := -2 * (x - 1) ^ 2 + 3

noncomputable def g (x : ℝ) : ℝ := -2 * x ^ 2

theorem graph_movement :
  ∀ (x y : ℝ),
  y = f x →
  g x = y → 
  (∃ Δx Δy, Δx = -1 ∧ Δy = -3 ∧ g (x + Δx) = y + Δy) :=
by
  sorry

end graph_movement_l731_73161


namespace max_quarters_l731_73105

/-- Prove that given the conditions for the number of nickels, dimes, and quarters,
    the maximum number of quarters can be 20. --/
theorem max_quarters {a b c : ℕ} (h1 : a + b + c = 120) (h2 : 5 * a + 10 * b + 25 * c = 1000) :
  c ≤ 20 :=
sorry

end max_quarters_l731_73105


namespace value_of_b_l731_73154

theorem value_of_b (x b : ℝ) (h₁ : x = 0.3) 
  (h₂ : (10 * x + b) / 4 - (3 * x - 6) / 18 = (2 * x + 4) / 3) : 
  b = 2 :=
by
  sorry

end value_of_b_l731_73154


namespace average_xyz_l731_73193

theorem average_xyz (x y z : ℝ) (h1 : x = 3) (h2 : y = 2 * x) (h3 : z = 3 * y) : 
  (x + y + z) / 3 = 9 :=
by
  sorry

end average_xyz_l731_73193


namespace mahesh_worked_days_l731_73187

-- Definitions
def mahesh_work_days := 45
def rajesh_work_days := 30
def total_work_days := 54

-- Theorem statement
theorem mahesh_worked_days (maheshrate : ℕ := mahesh_work_days) (rajeshrate : ℕ := rajesh_work_days) (totaldays : ℕ := total_work_days) :
  ∃ x : ℕ, x = totaldays - rajesh_work_days := by
  apply Exists.intro (54 - 30)
  simp
  sorry

end mahesh_worked_days_l731_73187


namespace ratio_diagonals_to_sides_l731_73194

-- Definition of the number of diagonals in a polygon with n sides
def number_of_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

-- Definition of the condition
def n : ℕ := 5

-- Proof statement that the ratio of the number of diagonals to the number of sides is 1
theorem ratio_diagonals_to_sides (n_eq_5 : n = 5) : 
  (number_of_diagonals n) / n = 1 :=
by {
  -- Proof would go here, but is omitted
  sorry
}

end ratio_diagonals_to_sides_l731_73194


namespace proof_w3_u2_y2_l731_73176

variable (x y z w u d : ℤ)

def arithmetic_sequence := x = 1370 ∧ z = 1070 ∧ w = -180 ∧ u = -6430 ∧ (z = x + 2 * d) ∧ (y = x + d)

theorem proof_w3_u2_y2 (h : arithmetic_sequence x y z w u d) : w^3 - u^2 + y^2 = -44200100 :=
  by
    sorry

end proof_w3_u2_y2_l731_73176


namespace find_annual_compound_interest_rate_l731_73199

noncomputable def compound_interest_rate (P A : ℝ) (n t : ℕ) (r : ℝ) : Prop :=
  A = P * (1 + r / n) ^ (n * t)

theorem find_annual_compound_interest_rate :
  compound_interest_rate 10000 24882.50 1 7 0.125 :=
by sorry

end find_annual_compound_interest_rate_l731_73199


namespace inscribed_circle_area_l731_73167

/-- Defining the inscribed circle problem and its area. -/
theorem inscribed_circle_area (l : ℝ) (h₁ : 90 = 90) (h₂ : true) : 
  ∃ r : ℝ, (r = (2 * (Real.sqrt 2 - 1) * l / Real.pi)) ∧ ((Real.pi * r ^ 2) = (12 - 8 * Real.sqrt 2) * l ^ 2 / Real.pi) :=
  sorry

end inscribed_circle_area_l731_73167


namespace sequences_count_equals_fibonacci_n_21_l731_73132

noncomputable def increasing_sequences_count (n: ℕ) : ℕ := 
  -- Function to count the number of valid increasing sequences
  sorry

def fibonacci : ℕ → ℕ 
  | 0 => 0
  | 1 => 1
  | (n + 2) => fibonacci n + fibonacci (n + 1)

theorem sequences_count_equals_fibonacci_n_21 :
  increasing_sequences_count 20 = fibonacci 21 :=
sorry

end sequences_count_equals_fibonacci_n_21_l731_73132


namespace jenna_owes_amount_l731_73163

theorem jenna_owes_amount (initial_bill : ℝ) (rate : ℝ) (times : ℕ) : 
  initial_bill = 400 → rate = 0.02 → times = 3 → 
  owed_amount = (400 * (1 + 0.02)^3) := 
by
  intros
  sorry

end jenna_owes_amount_l731_73163


namespace prism_height_relation_l731_73127

theorem prism_height_relation (a b c h : ℝ) 
  (h_perp : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) 
  (h_height : 0 < h) 
  (h_right_angles : true) :
  1 / h^2 = 1 / a^2 + 1 / b^2 + 1 / c^2 :=
by 
  sorry 

end prism_height_relation_l731_73127


namespace oranges_and_apples_costs_l731_73142

theorem oranges_and_apples_costs :
  ∃ (x y : ℚ), 7 * x + 5 * y = 13 ∧ 3 * x + 4 * y = 8 ∧ 37 * x + 45 * y = 93 :=
by 
  sorry

end oranges_and_apples_costs_l731_73142


namespace snowboard_price_after_discounts_l731_73140

noncomputable def final_snowboard_price (P_original : ℝ) (d_Friday : ℝ) (d_Monday : ℝ) : ℝ :=
  P_original * (1 - d_Friday) * (1 - d_Monday)

theorem snowboard_price_after_discounts :
  final_snowboard_price 100 0.50 0.30 = 35 :=
by 
  sorry

end snowboard_price_after_discounts_l731_73140


namespace sum_of_ages_l731_73195

theorem sum_of_ages (a b c d : ℕ) 
  (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d)
  (h4 : b ≠ c) (h5 : b ≠ d) (h6 : c ≠ d)
  (h7 : a * b = 24 ∨ a * c = 24 ∨ a * d = 24 ∨ b * c = 24 ∨ b * d = 24 ∨ c * d = 24)
  (h8 : a * b = 35 ∨ a * c = 35 ∨ a * d = 35 ∨ b * c = 35 ∨ b * d = 35 ∨ c * d = 35)
  (h9 : a < 10) (h10 : b < 10) (h11 : c < 10) (h12 : d < 10)
  (h13 : 0 < a) (h14 : 0 < b) (h15 : 0 < c) (h16 : 0 < d) :
  a + b + c + d = 23 := sorry

end sum_of_ages_l731_73195


namespace sarah_books_check_out_l731_73115

theorem sarah_books_check_out
  (words_per_minute : ℕ)
  (words_per_page : ℕ)
  (pages_per_book : ℕ)
  (reading_hours : ℕ)
  (number_of_books : ℕ) :
  words_per_minute = 40 →
  words_per_page = 100 →
  pages_per_book = 80 →
  reading_hours = 20 →
  number_of_books = 6 :=
by
  intros h1 h2 h3 h4
  sorry

end sarah_books_check_out_l731_73115


namespace find_sum_of_variables_l731_73192

theorem find_sum_of_variables (x y : ℚ) (h1 : 5 * x - 3 * y = 17) (h2 : 3 * x + 5 * y = 1) : x + y = 21 / 17 := 
  sorry

end find_sum_of_variables_l731_73192


namespace combined_gravitational_force_l731_73145

theorem combined_gravitational_force 
    (d_E_surface : ℝ) (f_E_surface : ℝ) (d_M_surface : ℝ) (f_M_surface : ℝ) 
    (d_E_new : ℝ) (d_M_new : ℝ) 
    (k_E : ℝ) (k_M : ℝ) 
    (h1 : k_E = f_E_surface * d_E_surface^2)
    (h2 : k_M = f_M_surface * d_M_surface^2)
    (h3 : f_E_new = k_E / d_E_new^2)
    (h4 : f_M_new = k_M / d_M_new^2) : 
  f_E_new + f_M_new = 755.7696 :=
by
  sorry

end combined_gravitational_force_l731_73145


namespace westgate_high_school_chemistry_l731_73197

theorem westgate_high_school_chemistry :
  ∀ (total_players physics_both physics : ℕ),
    total_players = 15 →
    physics_both = 3 →
    physics = 8 →
    (total_players - (physics - physics_both)) - physics_both = 10 := by
  intros total_players physics_both physics h1 h2 h3
  sorry

end westgate_high_school_chemistry_l731_73197


namespace part1_part2_l731_73139

-- Define the system of equations
def system_eq (x y k : ℝ) : Prop := 
  3 * x + y = k + 1 ∧ x + 3 * y = 3

-- Part (1): x and y are opposite in sign implies k = -4
theorem part1 (x y k : ℝ) (h_eq : system_eq x y k) (h_sign : x * y < 0) : k = -4 := by
  sorry

-- Part (2): range of values for k given extra inequalities
theorem part2 (x y k : ℝ) (h_eq : system_eq x y k) 
  (h_ineq1 : x + y < 3) (h_ineq2 : x - y > 1) : 4 < k ∧ k < 8 := by
  sorry

end part1_part2_l731_73139


namespace find_x_l731_73186

theorem find_x (x : ℝ) (h : x - 2 * x + 3 * x = 100) : x = 50 := by
  sorry

end find_x_l731_73186


namespace sam_watermelons_second_batch_l731_73136

theorem sam_watermelons_second_batch
  (initial_watermelons : ℕ)
  (total_watermelons : ℕ)
  (second_batch_watermelons : ℕ) :
  initial_watermelons = 4 →
  total_watermelons = 7 →
  second_batch_watermelons = total_watermelons - initial_watermelons →
  second_batch_watermelons = 3 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end sam_watermelons_second_batch_l731_73136


namespace license_plate_palindrome_probability_l731_73184

-- Define the two-letter palindrome probability
def prob_two_letter_palindrome : ℚ := 1 / 26

-- Define the four-digit palindrome probability
def prob_four_digit_palindrome : ℚ := 1 / 100

-- Define the joint probability of both two-letter and four-digit palindrome
def prob_joint_palindrome : ℚ := prob_two_letter_palindrome * prob_four_digit_palindrome

-- Define the probability of at least one palindrome using Inclusion-Exclusion
def prob_at_least_one_palindrome : ℚ := prob_two_letter_palindrome + prob_four_digit_palindrome - prob_joint_palindrome

-- Convert the probability to the form of sum of two integers
def sum_of_integers : ℕ := 5 + 104

-- The final proof problem
theorem license_plate_palindrome_probability :
  (prob_at_least_one_palindrome = 5 / 104) ∧ (sum_of_integers = 109) := by
  sorry

end license_plate_palindrome_probability_l731_73184


namespace maurice_late_467th_trip_l731_73135

-- Define the recurrence relation
def p (n : ℕ) : ℚ := 
  if n = 0 then 0
  else 1 / 4 * (p (n - 1) + 1)

-- Define the steady-state probability
def steady_state_p : ℚ := 1 / 3

-- Define L_n as the probability Maurice is late on the nth day
def L (n : ℕ) : ℚ := 1 - p n

-- The main goal (probability Maurice is late on his 467th trip)
theorem maurice_late_467th_trip :
  L 467 = 2 / 3 :=
sorry

end maurice_late_467th_trip_l731_73135


namespace trajectory_of_moving_circle_l731_73108

noncomputable def trajectory_equation_of_moving_circle_center 
  (x y : Real) : Prop :=
  (∃ r : Real, 
    ((x + 5)^2 + y^2 = 16) ∧ 
    ((x - 5)^2 + y^2 = 16)
  ) → (x > 0 → x^2 / 16 - y^2 / 9 = 1)

-- here's the statement of the proof problem
theorem trajectory_of_moving_circle
  (h₁ : ∀ x y : Real, (x + 5)^2 + y^2 = 16)
  (h₂ : ∀ x y : Real, (x - 5)^2 + y^2 = 16) :
  ∀ x y : Real, trajectory_equation_of_moving_circle_center x y :=
sorry

end trajectory_of_moving_circle_l731_73108


namespace incorrect_major_premise_l731_73196

noncomputable def Line := Type
noncomputable def Plane := Type

-- Conditions: Definitions
variable (b a : Line) (α : Plane)

-- Assumption: Line b is parallel to Plane α
axiom parallel_to_plane (p : Line) (π : Plane) : Prop

-- Assumption: Line a is in Plane α
axiom line_in_plane (l : Line) (π : Plane) : Prop

-- Define theorem stating the incorrect major premise
theorem incorrect_major_premise 
  (hb_par_α : parallel_to_plane b α)
  (ha_in_α : line_in_plane a α) : ¬ (parallel_to_plane b α → ∀ l, line_in_plane l α → b = l) := 
sorry

end incorrect_major_premise_l731_73196


namespace find_triples_l731_73120

theorem find_triples (x y p : ℤ) (prime_p : Prime p) :
  x^2 - 3 * x * y + p^2 * y^2 = 12 * p ↔ 
  (p = 3 ∧ ( (x = 6 ∧ y = 0) ∨ (x = -6 ∧ y = 0) ∨ (x = 4 ∧ y = 2) ∨ (x = -2 ∧ y = 2) ∨ (x = 2 ∧ y = -2) ∨ (x = -4 ∧ y = -2) ) ) := 
by
  sorry

end find_triples_l731_73120


namespace algebra_identity_l731_73175

theorem algebra_identity (x y : ℝ) (h1 : x + y = 2) (h2 : x - y = 4) : x^2 - y^2 = 8 := by
  sorry

end algebra_identity_l731_73175


namespace students_at_end_l731_73102

def initial_students : ℝ := 42.0
def students_left : ℝ := 4.0
def students_transferred : ℝ := 10.0

theorem students_at_end : initial_students - students_left - students_transferred = 28.0 :=
by
  -- Proof omitted
  sorry

end students_at_end_l731_73102


namespace b_work_time_l731_73111

theorem b_work_time (W : ℝ) (days_A days_combined : ℝ)
  (hA : W / days_A = W / 16)
  (h_combined : W / days_combined = W / (16 / 3)) :
  ∃ days_B, days_B = 8 :=
by
  sorry

end b_work_time_l731_73111


namespace correct_option_B_l731_73117

-- Given conditions
variable (f : ℝ → ℝ)
variable (h_even : ∀ x : ℝ, f x = f (-x))
variable (h_mono_inc : ∀ (a b : ℝ), 0 ≤ a ∧ a ≤ b → f a ≤ f b)

-- Theorem statement
theorem correct_option_B : f (-2) > f (-1) ∧ f (-1) > f (0) :=
by
  sorry

end correct_option_B_l731_73117


namespace t_bounds_f_bounds_l731_73119

noncomputable def t (x : ℝ) : ℝ := 3^x

noncomputable def f (x : ℝ) : ℝ := 9^x - 2 * 3^x + 4

theorem t_bounds (x : ℝ) (hx : -1 ≤ x ∧ x ≤ 2) :
  (1/3 ≤ t x ∧ t x ≤ 9) :=
sorry

theorem f_bounds (x : ℝ) (hx : -1 ≤ x ∧ x ≤ 2) :
  (3 ≤ f x ∧ f x ≤ 67) :=
sorry

end t_bounds_f_bounds_l731_73119


namespace hyperbola_eccentricity_ratio_hyperbola_condition_l731_73182

-- Part (a)
theorem hyperbola_eccentricity_ratio
  (a b c : ℝ) (h1 : c^2 = a^2 + b^2)
  (x0 y0 : ℝ) 
  (P : ℝ × ℝ) (h2 : P = (x0, y0))
  (F : ℝ × ℝ) (h3 : F = (c, 0))
  (D : ℝ) (h4 : D = a^2 / c)
  (d_PF : ℝ) (h5 : d_PF = ( (x0 - c)^2 + y0^2 )^(1/2))
  (d_PD : ℝ) (h6 : d_PD = |x0 - a^2 / c|)
  (e : ℝ) (h7 : e = c / a) :
  d_PF / d_PD = e :=
sorry

-- Part (b)
theorem hyperbola_condition
  (F_l : ℝ × ℝ) (h1 : F_l = (0, k))
  (X_l : ℝ × ℝ) (h2 : X_l = (x, l))
  (d_XF : ℝ) (h3 : d_XF = (x^2 + y^2)^(1/2))
  (d_Xl : ℝ) (h4 : d_Xl = |x - k|)
  (e : ℝ) (h5 : e > 1)
  (h6 : d_XF / d_Xl = e) :
  ∃ a b : ℝ, (x / a)^2 - (y / b)^2 = 1 :=
sorry

end hyperbola_eccentricity_ratio_hyperbola_condition_l731_73182


namespace fraction_to_decimal_l731_73198

theorem fraction_to_decimal : (5 : ℚ) / 8 = 0.625 := by
  -- Prove that the fraction 5/8 equals the decimal 0.625
  sorry

end fraction_to_decimal_l731_73198


namespace number_added_after_division_is_5_l731_73169

noncomputable def number_thought_of : ℕ := 72
noncomputable def result_after_division (n : ℕ) : ℕ := n / 6
noncomputable def final_result (n x : ℕ) : ℕ := result_after_division n + x

theorem number_added_after_division_is_5 :
  ∃ x : ℕ, final_result number_thought_of x = 17 ∧ x = 5 :=
by
  sorry

end number_added_after_division_is_5_l731_73169


namespace combined_percentage_tennis_is_31_l731_73130

-- Define the number of students at North High School
def students_north : ℕ := 1800

-- Define the number of students at South Elementary School
def students_south : ℕ := 2200

-- Define the percentage of students who prefer tennis at North High School
def percentage_tennis_north : ℚ := 25/100

-- Define the percentage of students who prefer tennis at South Elementary School
def percentage_tennis_south : ℚ := 35/100

-- Calculate the number of students who prefer tennis at North High School
def tennis_students_north : ℚ := students_north * percentage_tennis_north

-- Calculate the number of students who prefer tennis at South Elementary School
def tennis_students_south : ℚ := students_south * percentage_tennis_south

-- Calculate the total number of students who prefer tennis in both schools
def total_tennis_students : ℚ := tennis_students_north + tennis_students_south

-- Calculate the total number of students in both schools
def total_students : ℚ := students_north + students_south

-- Calculate the combined percentage of students who prefer tennis
def combined_percentage_tennis : ℚ := (total_tennis_students / total_students) * 100

-- Main statement to prove
theorem combined_percentage_tennis_is_31 :
  round combined_percentage_tennis = 31 := by sorry

end combined_percentage_tennis_is_31_l731_73130


namespace roots_poly_cond_l731_73137

theorem roots_poly_cond (α β p q γ δ : ℝ) 
  (h1 : α ^ 2 + p * α - 1 = 0) 
  (h2 : β ^ 2 + p * β - 1 = 0) 
  (h3 : γ ^ 2 + q * γ - 1 = 0) 
  (h4 : δ ^ 2 + q * δ - 1 = 0)
  (h5 : γ * δ = -1) :
  (α - γ) * (β - γ) * (α - δ) * (β - δ) = -(p - q) ^ 2 := 
by 
  sorry

end roots_poly_cond_l731_73137


namespace teacher_allocation_l731_73189

theorem teacher_allocation :
  ∃ n : ℕ, n = 150 ∧ 
  (∀ t1 t2 t3 t4 t5 : Prop, -- represent the five teachers
    ∃ s1 s2 s3 : Prop, -- represent the three schools
      s1 ∧ s2 ∧ s3 ∧ -- each school receives at least one teacher
        ((t1 ∨ t2 ∨ t3 ∨ t4 ∨ t5) ∧ -- allocation condition
         (t1 ∨ t2 ∨ t3 ∨ t4 ∨ t5) ∧
         (t1 ∨ t2 ∨ t3 ∨ t4 ∨ t5))) := sorry

end teacher_allocation_l731_73189


namespace ab_gt_ac_l731_73172

variables {a b c : ℝ}

theorem ab_gt_ac (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) : a * b > a * c :=
sorry

end ab_gt_ac_l731_73172


namespace find_complex_number_l731_73143

-- Define the complex number z and the condition
variable (z : ℂ)
variable (h : (conj z) / (1 + I) = 1 - 2 * I)

-- State the theorem
theorem find_complex_number (hz : h) : z = 3 + I := 
sorry

end find_complex_number_l731_73143


namespace remainder_of_division_l731_73155

def dividend := 1234567
def divisor := 257

theorem remainder_of_division : dividend % divisor = 774 :=
by
  sorry

end remainder_of_division_l731_73155


namespace smaller_cuboid_length_l731_73149

theorem smaller_cuboid_length
  (L : ℝ)
  (h1 : 32 * (L * 4 * 3) = 16 * 10 * 12) :
  L = 5 :=
by
  sorry

end smaller_cuboid_length_l731_73149


namespace sum_of_altitudes_of_triangle_l731_73110

open Real

noncomputable def sum_of_altitudes (a b c : ℝ) : ℝ :=
  let inter_x := -c / a
  let inter_y := -c / b
  let vertex1 := (inter_x, 0)
  let vertex2 := (0, inter_y)
  let vertex3 := (0, 0)
  let area_triangle := (1 / 2) * abs (inter_x * inter_y)
  let altitude_x := abs inter_x
  let altitude_y := abs inter_y
  let altitude_line := abs c / sqrt (a ^ 2 + b ^ 2)
  altitude_x + altitude_y + altitude_line

theorem sum_of_altitudes_of_triangle :
  sum_of_altitudes 15 6 90 = 21 + 10 * sqrt (1 / 29) :=
by
  sorry

end sum_of_altitudes_of_triangle_l731_73110


namespace range_of_a_l731_73181

theorem range_of_a (a : ℝ) :
  (∀ θ : ℝ, 0 < θ ∧ θ < (π / 2) → a ≤ 1 / Real.sin θ + 1 / Real.cos θ) ↔ a ≤ 2 * Real.sqrt 2 :=
sorry

end range_of_a_l731_73181


namespace initial_bananas_per_child_l731_73122

theorem initial_bananas_per_child (B x : ℕ) (total_children : ℕ := 780) (absent_children : ℕ := 390) :
  390 * (x + 2) = total_children * x → x = 2 :=
by
  intros h
  sorry

end initial_bananas_per_child_l731_73122


namespace time_to_cross_stationary_train_l731_73123

theorem time_to_cross_stationary_train (t_pole : ℝ) (speed_train : ℝ) (length_stationary_train : ℝ) 
  (t_pole_eq : t_pole = 5) (speed_train_eq : speed_train = 64.8) (length_stationary_train_eq : length_stationary_train = 360) :
  (t_pole * speed_train + length_stationary_train) / speed_train = 10.56 := 
by
  rw [t_pole_eq, speed_train_eq, length_stationary_train_eq]
  norm_num
  sorry

end time_to_cross_stationary_train_l731_73123


namespace number_of_lists_l731_73170

theorem number_of_lists (n k : ℕ) (h_n : n = 15) (h_k : k = 4) : (n ^ k) = 50625 := by
  have : 15 ^ 4 = 50625 := by norm_num
  rwa [h_n, h_k]

end number_of_lists_l731_73170


namespace age_of_b_l731_73113

theorem age_of_b (a b c : ℕ) (h1 : a = b + 2) (h2 : b = 2 * c) (h3 : a + b + c = 47) : b = 18 := 
  sorry

end age_of_b_l731_73113


namespace cos_sum_identity_l731_73174

theorem cos_sum_identity (α : ℝ) (h_cos : Real.cos α = 3 / 5) (h_alpha : 0 < α ∧ α < Real.pi / 2) :
  Real.cos (α + Real.pi / 3) = (3 - 4 * Real.sqrt 3) / 10 :=
by
  sorry

end cos_sum_identity_l731_73174


namespace least_possible_product_of_distinct_primes_gt_50_l731_73164

-- Define a predicate to check if a number is a prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

-- Define the conditions: two distinct primes greater than 50
def distinct_primes_gt_50 (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p > 50 ∧ q > 50 ∧ p ≠ q

-- The least possible product of two distinct primes each greater than 50
theorem least_possible_product_of_distinct_primes_gt_50 :
  ∃ p q : ℕ, distinct_primes_gt_50 p q ∧ p * q = 3127 :=
by
  sorry

end least_possible_product_of_distinct_primes_gt_50_l731_73164


namespace angles_in_interval_l731_73101

open Real

theorem angles_in_interval
    (θ : ℝ)
    (hθ : 0 ≤ θ ∧ θ ≤ 2 * π)
    (h : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → x^2 * sin θ - x * (2 - x) + (2 - x)^2 * cos θ > 0) :
  π / 12 < θ ∧ θ < 5 * π / 12 :=
by
  sorry

end angles_in_interval_l731_73101


namespace intersecting_lines_l731_73134

theorem intersecting_lines (p : ℝ) :
    (∃ x y : ℝ, y = 3 * x - 6 ∧ y = -4 * x + 8 ∧ y = 7 * x + p) ↔ p = -14 :=
by {
    sorry
}

end intersecting_lines_l731_73134


namespace point_to_polar_coordinates_l731_73106

noncomputable def convert_to_polar_coordinates (x y : ℝ) (r θ : ℝ) : Prop :=
  r = Real.sqrt (x^2 + y^2) ∧ θ = Real.arctan (y / x)

theorem point_to_polar_coordinates :
  convert_to_polar_coordinates 8 (2 * Real.sqrt 6) 
    (2 * Real.sqrt 22) (Real.arctan (Real.sqrt 6 / 4)) :=
sorry

end point_to_polar_coordinates_l731_73106


namespace max_value_of_a_l731_73180

theorem max_value_of_a :
  ∃ b : ℤ, ∃ (a : ℝ), 
    (a = 30285) ∧
    (a * b^2 / (a + 2 * b) = 2019) :=
by
  sorry

end max_value_of_a_l731_73180


namespace point_distance_l731_73118

theorem point_distance (x : ℤ) : abs x = 2021 → (x = 2021 ∨ x = -2021) := 
sorry

end point_distance_l731_73118


namespace bus_stops_per_hour_l731_73146

theorem bus_stops_per_hour 
  (speed_without_stoppages : ℝ)
  (speed_with_stoppages : ℝ)
  (h₁ : speed_without_stoppages = 50)
  (h₂ : speed_with_stoppages = 40) :
  ∃ (minutes_stopped : ℝ), minutes_stopped = 12 :=
by
  sorry

end bus_stops_per_hour_l731_73146


namespace sequence_solution_exists_l731_73133

noncomputable def math_problem (a : ℕ → ℝ) : Prop :=
  ∀ n < 1990, a n > 0 ∧ a 1990 < 0

theorem sequence_solution_exists {a0 c : ℝ} (h_a0 : a0 > 0) (h_c : c > 0) :
  ∃ (a : ℕ → ℝ),
    a 0 = a0 ∧
    (∀ n, a (n + 1) = (a n + c) / (1 - a n * c)) ∧
    math_problem a :=
by
  sorry

end sequence_solution_exists_l731_73133


namespace bill_sunday_miles_l731_73129

-- Define the variables
variables (B S J : ℕ) -- B for miles Bill ran on Saturday, S for miles Bill ran on Sunday, J for miles Julia ran on Sunday

-- State the conditions
def condition1 (B S : ℕ) : Prop := S = B + 4
def condition2 (B S J : ℕ) : Prop := J = 2 * S
def condition3 (B S J : ℕ) : Prop := B + S + J = 20

-- The final theorem to prove the number of miles Bill ran on Sunday
theorem bill_sunday_miles (B S J : ℕ) 
  (h1 : condition1 B S)
  (h2 : condition2 B S J)
  (h3 : condition3 B S J) : 
  S = 6 := 
sorry

end bill_sunday_miles_l731_73129


namespace arithmetic_progression_cubic_eq_l731_73151

theorem arithmetic_progression_cubic_eq (x y z u : ℤ) (d : ℤ) :
  (x, y, z, u) = (3 * d, 4 * d, 5 * d, 6 * d) →
  x^3 + y^3 + z^3 = u^3 →
  ∃ d : ℤ, x = 3 * d ∧ y = 4 * d ∧ z = 5 * d ∧ u = 6 * d :=
by sorry

end arithmetic_progression_cubic_eq_l731_73151


namespace average_score_difference_l731_73138

theorem average_score_difference {A B : ℝ} (hA : (19 * A + 125) / 20 = A + 5) (hB : (17 * B + 145) / 18 = B + 6) :
  (B + 6) - (A + 5) = 13 :=
  sorry

end average_score_difference_l731_73138


namespace distinct_ordered_pairs_count_l731_73156

theorem distinct_ordered_pairs_count :
  ∃ S : Finset (ℕ × ℕ), 
    (∀ p ∈ S, 1 ≤ p.1 ∧ 1 ≤ p.2 ∧ (1 / (p.1 : ℚ) + 1 / (p.2 : ℚ) = 1 / 6)) ∧
    S.card = 9 := 
by
  sorry

end distinct_ordered_pairs_count_l731_73156


namespace balls_into_boxes_problem_l731_73128

theorem balls_into_boxes_problem :
  ∃ (n : ℕ), n = 144 ∧ ∃ (balls : Fin 4 → ℕ), 
  (∃ (boxes : Fin 4 → Fin 4), 
    (∀ (b : Fin 4), boxes b < 4 ∧ boxes b ≠ b) ∧ 
    (∃! (empty_box : Fin 4), ∀ (b : Fin 4), (boxes b = empty_box) → false)) := 
by
  sorry

end balls_into_boxes_problem_l731_73128


namespace circle_center_radius_l731_73107

theorem circle_center_radius :
  ∀ x y : ℝ,
  x^2 + y^2 + 4 * x - 6 * y - 3 = 0 →
  (∃ h k r : ℝ, (x + h)^2 + (y + k)^2 = r^2 ∧ h = -2 ∧ k = 3 ∧ r = 4) :=
by
  intros x y hxy
  sorry

end circle_center_radius_l731_73107


namespace platform_length_l731_73190

theorem platform_length (train_length : ℝ) (speed_kmph : ℝ) (time_sec : ℝ) (platform_length : ℝ) :
  train_length = 150 ∧ speed_kmph = 75 ∧ time_sec = 20 →
  platform_length = 1350 :=
by
  sorry

end platform_length_l731_73190


namespace percentage_problem_l731_73185

variable (N P : ℝ)

theorem percentage_problem (h1 : 0.3 * N = 120) (h2 : (P / 100) * N = 160) : P = 40 :=
by
  sorry

end percentage_problem_l731_73185


namespace determine_c_for_quadratic_eq_l731_73178

theorem determine_c_for_quadratic_eq (x1 x2 c : ℝ) 
  (h1 : x1 + x2 = 2)
  (h2 : x1 * x2 = c)
  (h3 : 7 * x2 - 4 * x1 = 47) : 
  c = -15 :=
sorry

end determine_c_for_quadratic_eq_l731_73178


namespace speed_upstream_l731_73183

-- Conditions definitions
def speed_of_boat_still_water : ℕ := 50
def speed_of_current : ℕ := 20

-- Theorem stating the problem
theorem speed_upstream : (speed_of_boat_still_water - speed_of_current = 30) :=
by
  -- Proof is omitted
  sorry

end speed_upstream_l731_73183


namespace valve_difference_l731_73157

theorem valve_difference (time_both : ℕ) (time_first : ℕ) (pool_capacity : ℕ) (V1 V2 diff : ℕ) :
  time_both = 48 → 
  time_first = 120 → 
  pool_capacity = 12000 → 
  V1 = pool_capacity / time_first → 
  V1 + V2 = pool_capacity / time_both → 
  diff = V2 - V1 → 
  diff = 50 :=
by sorry

end valve_difference_l731_73157


namespace Marty_combinations_l731_73191

def unique_combinations (colors techniques : ℕ) : ℕ :=
  colors * techniques

theorem Marty_combinations :
  unique_combinations 6 5 = 30 := by
  sorry

end Marty_combinations_l731_73191


namespace x_greater_than_y_l731_73152

theorem x_greater_than_y (x y z : ℝ) (h1 : x + y + z = 28) (h2 : 2 * x - y = 32) (h3 : 0 < x) (h4 : 0 < y) (h5 : 0 < z) : 
  x > y :=
by 
  sorry

end x_greater_than_y_l731_73152


namespace average_stoppage_time_l731_73177

def bus_a_speed_excluding_stoppages := 54 -- kmph
def bus_a_speed_including_stoppages := 45 -- kmph

def bus_b_speed_excluding_stoppages := 60 -- kmph
def bus_b_speed_including_stoppages := 50 -- kmph

def bus_c_speed_excluding_stoppages := 72 -- kmph
def bus_c_speed_including_stoppages := 60 -- kmph

theorem average_stoppage_time :
  (bus_a_speed_excluding_stoppages - bus_a_speed_including_stoppages) / bus_a_speed_excluding_stoppages * 60
  + (bus_b_speed_excluding_stoppages - bus_b_speed_including_stoppages) / bus_b_speed_excluding_stoppages * 60
  + (bus_c_speed_excluding_stoppages - bus_c_speed_including_stoppages) / bus_c_speed_excluding_stoppages * 60
  = 30 / 3 :=
  by sorry

end average_stoppage_time_l731_73177


namespace proof_goats_minus_pigs_l731_73158

noncomputable def number_of_goats : ℕ := 66
noncomputable def number_of_chickens : ℕ := 2 * number_of_goats - 10
noncomputable def number_of_ducks : ℕ := (number_of_goats + number_of_chickens) / 2
noncomputable def number_of_pigs : ℕ := number_of_ducks / 3
noncomputable def number_of_rabbits : ℕ := Nat.floor (Real.sqrt (2 * number_of_ducks - number_of_pigs))
noncomputable def number_of_cows : ℕ := number_of_rabbits ^ number_of_pigs / Nat.factorial (number_of_goats / 2)

theorem proof_goats_minus_pigs : number_of_goats - number_of_pigs = 35 := by
  sorry

end proof_goats_minus_pigs_l731_73158


namespace combined_teaching_experience_l731_73162

def james_teaching_years : ℕ := 40
def partner_teaching_years : ℕ := james_teaching_years - 10

theorem combined_teaching_experience : james_teaching_years + partner_teaching_years = 70 :=
by
  sorry

end combined_teaching_experience_l731_73162


namespace students_not_made_the_cut_l731_73112

-- Define the constants for the number of girls, boys, and students called back
def girls := 17
def boys := 32
def called_back := 10

-- Total number of students trying out for the team
def total_try_out := girls + boys

-- Number of students who didn't make the cut
def not_made_the_cut := total_try_out - called_back

-- The theorem to be proved
theorem students_not_made_the_cut : not_made_the_cut = 39 := by
  -- Adding the proof is not required, so we use sorry
  sorry

end students_not_made_the_cut_l731_73112


namespace prime_factors_1260_l731_73121

theorem prime_factors_1260 (w x y z : ℕ) (h : 2 ^ w * 3 ^ x * 5 ^ y * 7 ^ z = 1260) : 2 * w + 3 * x + 5 * y + 7 * z = 22 :=
by sorry

end prime_factors_1260_l731_73121


namespace lesser_fraction_solution_l731_73179

noncomputable def lesser_fraction (x y : ℚ) (h₁ : x + y = 7/8) (h₂ : x * y = 1/12) : ℚ :=
  if x ≤ y then x else y

theorem lesser_fraction_solution (x y : ℚ) (h₁ : x + y = 7/8) (h₂ : x * y = 1/12) :
  lesser_fraction x y h₁ h₂ = (7 - Real.sqrt 17) / 16 := by
  sorry

end lesser_fraction_solution_l731_73179


namespace number_is_580_l731_73104

noncomputable def find_number (x : ℝ) : Prop :=
  0.20 * x = 116

theorem number_is_580 (x : ℝ) (h : find_number x) : x = 580 :=
  by sorry

end number_is_580_l731_73104


namespace power_mod_l731_73168

theorem power_mod (n : ℕ) : 2^99 % 7 = 1 := 
by {
  sorry
}

end power_mod_l731_73168


namespace acute_angle_comparison_l731_73147

noncomputable def f : ℝ → ℝ := sorry

def even_function (f : ℝ → ℝ) :=
  ∀ x, f x = f (-x)

def periodic_function (f : ℝ → ℝ) (p : ℝ) :=
  ∀ x, f (x + p) = f x

def increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

theorem acute_angle_comparison (A B : ℝ)
  (hA : 0 < A ∧ A < π / 2)
  (hB : 0 < B ∧ B < π / 2)
  (f_even : even_function f)
  (f_periodic : ∀ x, f (x + 1) + f x = 0)
  (f_increasing : increasing_on_interval f 3 4) :
  f (Real.sin A) < f (Real.cos B) :=
sorry

end acute_angle_comparison_l731_73147


namespace trapezoid_area_correct_l731_73124

noncomputable def trapezoid_area : ℝ := 
  let base1 : ℝ := 8
  let base2 : ℝ := 4
  let height : ℝ := 2
  (1 / 2) * (base1 + base2) * height

theorem trapezoid_area_correct :
  trapezoid_area = 12.0 :=
by
  -- sorry is used here to skip the actual proof
  sorry

end trapezoid_area_correct_l731_73124


namespace probability_x_gt_3y_correct_l731_73165

noncomputable def probability_x_gt_3y : ℚ :=
  let rect_area := (2010 : ℚ) * 2011
  let triangle_area := (2010 : ℚ) * (2010 / 3) / 2
  (triangle_area / rect_area)

theorem probability_x_gt_3y_correct :
  probability_x_gt_3y = 670 / 2011 := 
by
  sorry

end probability_x_gt_3y_correct_l731_73165


namespace find_solutions_to_system_l731_73150

theorem find_solutions_to_system (x y z : ℝ) 
    (h1 : 3 * (x^2 + y^2 + z^2) = 1) 
    (h2 : x^2 * y^2 + y^2 * z^2 + z^2 * x^2 = x * y * z * (x + y + z)^3) : 
    x = y ∧ y = z ∧ (x = 1 / 3 ∨ x = -1 / 3) :=
by
  sorry

end find_solutions_to_system_l731_73150


namespace range_of_m_l731_73131

theorem range_of_m (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : (2 * x - y / Real.exp 1) * Real.log (y / x) ≤ x / (m * Real.exp 1)) :
  0 < m ∧ m ≤ 1 / Real.exp 1 :=
sorry

end range_of_m_l731_73131


namespace unique_solution_l731_73114

theorem unique_solution (a b : ℤ) (h1 : a > b) (h2 : b > 0) (h3 : a * b - a - b = 1) : (a, b) = (3, 2) :=
by
  sorry

end unique_solution_l731_73114


namespace symmetric_function_l731_73100

def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def symmetric_about_axis (f : ℤ → ℤ) (axis : ℤ) : Prop :=
  ∀ x : ℤ, f (axis - x) = f (axis + x)

theorem symmetric_function (a : ℕ → ℤ) (d : ℤ) (f : ℤ → ℤ) (a1 a2 : ℤ) (axis : ℤ) :
  (∀ x, f x = |x - a1| + |x - a2|) →
  arithmetic_sequence a d →
  d ≠ 0 →
  axis = (a1 + a2) / 2 →
  symmetric_about_axis f axis :=
by
  -- Proof goes here
  sorry

end symmetric_function_l731_73100


namespace not_proportional_x2_y2_l731_73173

def directly_proportional (x y : ℝ) : Prop :=
∃ k : ℝ, x = k * y

def inversely_proportional (x y : ℝ) : Prop :=
∃ k : ℝ, x * y = k

theorem not_proportional_x2_y2 (x y : ℝ) :
  x^2 + y^2 = 16 → ¬directly_proportional x y ∧ ¬inversely_proportional x y :=
by
  sorry

end not_proportional_x2_y2_l731_73173


namespace undefined_expression_l731_73125

theorem undefined_expression (y : ℝ) : (y^2 - 16 * y + 64 = 0) ↔ (y = 8) := by
  sorry

end undefined_expression_l731_73125


namespace minimum_value_of_f_l731_73166

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x^2 + 3 * x + 3) + Real.sqrt (x^2 - 3 * x + 3)

theorem minimum_value_of_f : (∃ x : ℝ, ∀ y : ℝ, f x ≤ f y) ∧ f 0 = 2 * Real.sqrt 3 :=
by
  sorry

end minimum_value_of_f_l731_73166


namespace birds_remaining_l731_73171

variable (initial_birds : ℝ) (birds_flew_away : ℝ)

theorem birds_remaining (h1 : initial_birds = 12.0) (h2 : birds_flew_away = 8.0) : initial_birds - birds_flew_away = 4.0 :=
by
  rw [h1, h2]
  norm_num

end birds_remaining_l731_73171


namespace original_population_l731_73188

-- Define the conditions
def population_increase (n : ℕ) : ℕ := n + 1200
def population_decrease (p : ℕ) : ℕ := (89 * p) / 100
def final_population (n : ℕ) : ℕ := population_decrease (population_increase n)

-- Claim that needs to be proven
theorem original_population (n : ℕ) (H : final_population n = n - 32) : n = 10000 :=
by
  sorry

end original_population_l731_73188


namespace arithmetic_series_remainder_l731_73103

noncomputable def arithmetic_series_sum_mod (a d n : ℕ) : ℕ :=
  (n * (2 * a + (n - 1) * d) / 2) % 10

theorem arithmetic_series_remainder :
  let a := 3
  let d := 5
  let n := 21
  arithmetic_series_sum_mod a d n = 3 :=
by
  sorry

end arithmetic_series_remainder_l731_73103


namespace f_2020_minus_f_2018_l731_73144

noncomputable def f : ℝ → ℝ := sorry

axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom periodic_f : ∀ x : ℝ, f (x + 5) = f x
axiom f_seven : f 7 = 9

theorem f_2020_minus_f_2018 : f 2020 - f 2018 = 9 := by
  sorry

end f_2020_minus_f_2018_l731_73144


namespace floor_expression_correct_l731_73109

theorem floor_expression_correct :
  (∃ x : ℝ, x = 2007 ^ 3 / (2005 * 2006) - 2005 ^ 3 / (2006 * 2007) ∧ ⌊x⌋ = 8) := 
sorry

end floor_expression_correct_l731_73109


namespace range_of_a_l731_73126

-- Definitions of the propositions in Lean terms
def proposition_p (a : ℝ) := 
  ∃ x : ℝ, x ∈ [-1, 1] ∧ x^2 - (2 + a) * x + 2 * a = 0

def proposition_q (a : ℝ) := 
  ∃ x : ℝ, x^2 + 2 * a * x + 2 * a ≤ 0

-- The main theorem to prove that the range of values for a is [-1, 0]
theorem range_of_a {a : ℝ} (h : proposition_p a ∧ proposition_q a) : 
  -1 ≤ a ∧ a ≤ 0 := sorry

end range_of_a_l731_73126


namespace closest_fraction_to_medals_won_l731_73160

theorem closest_fraction_to_medals_won :
  let won_ratio : ℚ := 35 / 225
  let choices : List ℚ := [1/5, 1/6, 1/7, 1/8, 1/9]
  (closest : ℚ) = 1 / 6 → 
  (closest_in_choices : closest ∈ choices) →
  ∀ choice ∈ choices, abs ((7 / 45) - (1 / 6)) ≤ abs ((7 / 45) - choice) :=
by
  let won_ratio := 7 / 45
  let choices := [1/5, 1/6, 1/7, 1/8, 1/9]
  let closest := 1 / 6
  have closest_in_choices : closest ∈ choices := sorry
  intro choice h_choice_in_choices
  sorry

end closest_fraction_to_medals_won_l731_73160
