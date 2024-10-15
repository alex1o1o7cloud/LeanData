import Mathlib

namespace NUMINAMATH_GPT_unique_integer_solution_l1694_169463

-- Define the problem statement and the conditions: integers x, y such that x^4 - 2y^2 = 1
theorem unique_integer_solution (x y: ℤ) (h: x^4 - 2 * y^2 = 1) : (x = 1 ∧ y = 0) :=
sorry

end NUMINAMATH_GPT_unique_integer_solution_l1694_169463


namespace NUMINAMATH_GPT_find_distance_CD_l1694_169493

noncomputable def distance_CD : ℝ :=
  let C : ℝ × ℝ := (0, 0)
  let D : ℝ × ℝ := (3, 6)
  Real.sqrt ((D.1 - C.1)^2 + (D.2 - C.2)^2)

theorem find_distance_CD :
  ∀ (C D : ℝ × ℝ), 
  (C = (0, 0) ∧ D = (3, 6)) ∧ 
  (∃ x y : ℝ, (y^2 = 12 * x ∧ (x^2 + y^2 - 4 * x - 6 * y = 0))) → 
  distance_CD = 3 * Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_find_distance_CD_l1694_169493


namespace NUMINAMATH_GPT_larger_integer_l1694_169489

theorem larger_integer (x y : ℕ) (h_diff : y - x = 8) (h_prod : x * y = 272) : y = 20 :=
by
  sorry

end NUMINAMATH_GPT_larger_integer_l1694_169489


namespace NUMINAMATH_GPT_cost_of_paving_floor_l1694_169400

-- Conditions
def length_of_room : ℝ := 8
def width_of_room : ℝ := 4.75
def rate_per_sq_metre : ℝ := 900

-- Statement to prove
theorem cost_of_paving_floor : (length_of_room * width_of_room * rate_per_sq_metre) = 34200 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_paving_floor_l1694_169400


namespace NUMINAMATH_GPT_cans_of_soda_l1694_169408

theorem cans_of_soda (S Q D : ℕ) : (4 * D * S) / Q = x :=
by
  sorry

end NUMINAMATH_GPT_cans_of_soda_l1694_169408


namespace NUMINAMATH_GPT_largest_number_is_B_l1694_169447

-- Define the numbers as constants
def A : ℝ := 0.989
def B : ℝ := 0.998
def C : ℝ := 0.981
def D : ℝ := 0.899
def E : ℝ := 0.9801

-- State the theorem that B is the largest number
theorem largest_number_is_B : B > A ∧ B > C ∧ B > D ∧ B > E := by
  -- By comparison
  sorry

end NUMINAMATH_GPT_largest_number_is_B_l1694_169447


namespace NUMINAMATH_GPT_greater_number_is_twenty_two_l1694_169472

theorem greater_number_is_twenty_two (x y : ℝ) (h1 : x + y = 40) (h2 : 3 * (x - y) = 12) : x = 22 :=
sorry

end NUMINAMATH_GPT_greater_number_is_twenty_two_l1694_169472


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1694_169450

-- Definitions used in the conditions
variable (a : ℕ → ℕ)
variable (n : ℕ)
variable (a_seq : Prop)
-- Declaring the arithmetic sequence condition
def is_arithmetic_sequence (a : ℕ → ℕ) : Prop := ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1

noncomputable def a_5_is_2 : Prop := a 5 = 2

-- The statement we need to prove
theorem arithmetic_sequence_sum (a : ℕ → ℕ) (h_arith_seq : is_arithmetic_sequence a) (h_a5 : a 5 = 2) :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 2 * 9 := by
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1694_169450


namespace NUMINAMATH_GPT_smallest_number_of_butterflies_l1694_169413

theorem smallest_number_of_butterflies 
  (identical_groups : ℕ) 
  (groups_of_butterflies : ℕ) 
  (groups_of_fireflies : ℕ) 
  (groups_of_ladybugs : ℕ)
  (h1 : groups_of_butterflies = 44)
  (h2 : groups_of_fireflies = 17)
  (h3 : groups_of_ladybugs = 25)
  (h4 : identical_groups * (groups_of_butterflies + groups_of_fireflies + groups_of_ladybugs) % 60 = 0) :
  identical_groups * groups_of_butterflies = 425 :=
sorry

end NUMINAMATH_GPT_smallest_number_of_butterflies_l1694_169413


namespace NUMINAMATH_GPT_find_number_l1694_169444

theorem find_number (x : ℝ) (h : 0.15 * 40 = 0.25 * x + 2) : x = 16 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l1694_169444


namespace NUMINAMATH_GPT_shape_of_constant_phi_l1694_169481

-- Define the spherical coordinates structure
structure SphericalCoordinates where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

-- Define the condition that φ is a constant c
def constant_phi (c : ℝ) (coords : SphericalCoordinates) : Prop :=
  coords.φ = c

-- Define the type for shapes
inductive Shape
  | Line : Shape
  | Circle : Shape
  | Plane : Shape
  | Sphere : Shape
  | Cylinder : Shape
  | Cone : Shape

-- The theorem statement
theorem shape_of_constant_phi (c : ℝ) (coords : SphericalCoordinates) 
  (h : constant_phi c coords) : Shape :=
  Shape.Cone

end NUMINAMATH_GPT_shape_of_constant_phi_l1694_169481


namespace NUMINAMATH_GPT_polynomial_division_l1694_169482

open Polynomial

-- Define the theorem statement
theorem polynomial_division (f g : ℤ[X])
  (h : ∀ n : ℤ, f.eval n ∣ g.eval n) :
  ∃ (h : ℤ[X]), g = f * h :=
sorry

end NUMINAMATH_GPT_polynomial_division_l1694_169482


namespace NUMINAMATH_GPT_geometric_series_sum_squares_l1694_169474

theorem geometric_series_sum_squares (a r : ℝ) (hr : -1 < r) (hr2 : r < 1) :
  (∑' n : ℕ, a^2 * r^(3 * n)) = a^2 / (1 - r^3) :=
by
  -- Note: Proof goes here
  sorry

end NUMINAMATH_GPT_geometric_series_sum_squares_l1694_169474


namespace NUMINAMATH_GPT_largest_divisor_of_prime_squares_l1694_169458

theorem largest_divisor_of_prime_squares (p q : ℕ) (hp : Prime p) (hq : Prime q) (h : q < p) : 
  ∃ d : ℕ, ∀ p q : ℕ, Prime p → Prime q → q < p → d ∣ (p^2 - q^2) ∧ ∀ k : ℕ, (∀ p q : ℕ, Prime p → Prime q → q < p → k ∣ (p^2 - q^2)) → k ≤ d :=
by 
  use 2
  {
    sorry
  }

end NUMINAMATH_GPT_largest_divisor_of_prime_squares_l1694_169458


namespace NUMINAMATH_GPT_complex_quadrant_l1694_169419

-- Define the imaginary unit
def i := Complex.I

-- Define the complex number z satisfying the given condition
variables (z : Complex)
axiom h : (3 - 2 * i) * z = 4 + 3 * i

-- Statement for the proof problem
theorem complex_quadrant (h : (3 - 2 * i) * z = 4 + 3 * i) : 
  (0 < z.re ∧ 0 < z.im) :=
sorry

end NUMINAMATH_GPT_complex_quadrant_l1694_169419


namespace NUMINAMATH_GPT_program_output_eq_l1694_169470

theorem program_output_eq : ∀ (n : ℤ), n^2 + 3 * n - (2 * n^2 - n) = -n^2 + 4 * n := by
  intro n
  sorry

end NUMINAMATH_GPT_program_output_eq_l1694_169470


namespace NUMINAMATH_GPT_factorize_3x_squared_minus_7x_minus_6_factorize_6x_squared_minus_7x_minus_5_l1694_169406

theorem factorize_3x_squared_minus_7x_minus_6 (x : ℝ) :
  3 * x^2 - 7 * x - 6 = (x - 3) * (3 * x + 2) :=
sorry

theorem factorize_6x_squared_minus_7x_minus_5 (x : ℝ) :
  6 * x^2 - 7 * x - 5 = (2 * x + 1) * (3 * x - 5) :=
sorry

end NUMINAMATH_GPT_factorize_3x_squared_minus_7x_minus_6_factorize_6x_squared_minus_7x_minus_5_l1694_169406


namespace NUMINAMATH_GPT_eighth_grade_students_l1694_169480

def avg_books (total_books : ℕ) (num_students : ℕ) : ℚ :=
  total_books / num_students

theorem eighth_grade_students (x : ℕ) (y : ℕ)
  (h1 : x + y = 1800)
  (h2 : y = x - 150)
  (h3 : avg_books x 1800 = 1.5 * avg_books (x - 150) 1800) :
  y = 450 :=
by {
  sorry
}

end NUMINAMATH_GPT_eighth_grade_students_l1694_169480


namespace NUMINAMATH_GPT_largest_perfect_square_factor_of_882_l1694_169417

theorem largest_perfect_square_factor_of_882 : ∃ n, n * n = 441 ∧ ∀ m, m * m ∣ 882 → m * m ≤ 441 := 
by 
 sorry

end NUMINAMATH_GPT_largest_perfect_square_factor_of_882_l1694_169417


namespace NUMINAMATH_GPT_heath_time_spent_l1694_169471

variables (rows_per_carrot : ℕ) (plants_per_row : ℕ) (carrots_per_hour : ℕ) (total_hours : ℕ)

def total_carrots (rows_per_carrot plants_per_row : ℕ) : ℕ :=
  rows_per_carrot * plants_per_row

def time_spent (total_carrots carrots_per_hour : ℕ) : ℕ :=
  total_carrots / carrots_per_hour

theorem heath_time_spent
  (h1 : rows_per_carrot = 400)
  (h2 : plants_per_row = 300)
  (h3 : carrots_per_hour = 6000)
  (h4 : total_hours = 20) :
  time_spent (total_carrots rows_per_carrot plants_per_row) carrots_per_hour = total_hours :=
by
  sorry

end NUMINAMATH_GPT_heath_time_spent_l1694_169471


namespace NUMINAMATH_GPT_exp_gt_one_iff_a_gt_one_l1694_169467

theorem exp_gt_one_iff_a_gt_one (a : ℝ) : 
  (∀ x : ℝ, 0 < x → a^x > 1) ↔ a > 1 :=
by
  sorry

end NUMINAMATH_GPT_exp_gt_one_iff_a_gt_one_l1694_169467


namespace NUMINAMATH_GPT_art_club_activity_l1694_169420

theorem art_club_activity (n p s b : ℕ) (h1 : n = 150) (h2 : p = 80) (h3 : s = 60) (h4 : b = 20) :
  (n - (p + s - b) = 30) :=
by
  sorry

end NUMINAMATH_GPT_art_club_activity_l1694_169420


namespace NUMINAMATH_GPT_number_of_moles_of_HCl_l1694_169488

-- Defining the chemical equation relationship
def reaction_relation (HCl NaHCO3 NaCl H2O CO2 : ℕ) : Prop :=
  H2O = HCl ∧ H2O = NaHCO3

-- Conditions
def conditions (HCl NaHCO3 H2O : ℕ) : Prop :=
  NaHCO3 = 3 ∧ H2O = 3

-- Theorem statement proving the number of moles of HCl given the conditions
theorem number_of_moles_of_HCl (HCl NaHCO3 NaCl H2O CO2 : ℕ) 
  (h1 : reaction_relation HCl NaHCO3 NaCl H2O CO2) 
  (h2 : conditions HCl NaHCO3 H2O) :
  HCl = 3 :=
sorry

end NUMINAMATH_GPT_number_of_moles_of_HCl_l1694_169488


namespace NUMINAMATH_GPT_combinations_count_l1694_169404

def colorChoices := 4
def decorationChoices := 3
def methodChoices := 3

theorem combinations_count : colorChoices * decorationChoices * methodChoices = 36 := by
  sorry

end NUMINAMATH_GPT_combinations_count_l1694_169404


namespace NUMINAMATH_GPT_units_digit_sum_l1694_169498

theorem units_digit_sum (n1 n2 : ℕ) (h1 : n1 % 10 = 1) (h2 : n2 % 10 = 3) : ((n1^3 + n2^3) % 10) = 8 := 
by
  sorry

end NUMINAMATH_GPT_units_digit_sum_l1694_169498


namespace NUMINAMATH_GPT_ice_cream_cones_sold_l1694_169424

theorem ice_cream_cones_sold (T W : ℕ) (h1 : W = 2 * T) (h2 : T + W = 36000) : T = 12000 :=
by
  sorry

end NUMINAMATH_GPT_ice_cream_cones_sold_l1694_169424


namespace NUMINAMATH_GPT_dhoni_savings_l1694_169429

theorem dhoni_savings :
  let earnings := 100
  let rent := 0.25 * earnings
  let dishwasher := rent - (0.10 * rent)
  let utilities := 0.15 * earnings
  let groceries := 0.20 * earnings
  let transportation := 0.12 * earnings
  let total_spent := rent + dishwasher + utilities + groceries + transportation
  earnings - total_spent = 0.055 * earnings :=
by
  sorry

end NUMINAMATH_GPT_dhoni_savings_l1694_169429


namespace NUMINAMATH_GPT_find_third_number_l1694_169486

theorem find_third_number (x y : ℕ) (h1 : x = 3)
  (h2 : (x + 1) / (x + 5) = (x + 5) / (x + y)) : y = 13 :=
by
  sorry

end NUMINAMATH_GPT_find_third_number_l1694_169486


namespace NUMINAMATH_GPT_books_in_bin_after_transactions_l1694_169455

def initial_books : ℕ := 4
def sold_books : ℕ := 3
def added_books : ℕ := 10

def final_books (initial_books sold_books added_books : ℕ) : ℕ :=
  initial_books - sold_books + added_books

theorem books_in_bin_after_transactions :
  final_books initial_books sold_books added_books = 11 := by
  sorry

end NUMINAMATH_GPT_books_in_bin_after_transactions_l1694_169455


namespace NUMINAMATH_GPT_greatest_int_with_gcd_of_24_eq_2_l1694_169490

theorem greatest_int_with_gcd_of_24_eq_2 (n : ℕ) (h1 : n < 200) (h2 : Int.gcd n 24 = 2) : n = 194 := 
sorry

end NUMINAMATH_GPT_greatest_int_with_gcd_of_24_eq_2_l1694_169490


namespace NUMINAMATH_GPT_money_lent_to_C_is_3000_l1694_169483

def principal_B : ℕ := 5000
def time_B : ℕ := 2
def time_C : ℕ := 4
def rate_of_interest : ℕ := 12
def total_interest : ℕ := 2640
def interest_rate : ℚ := (rate_of_interest : ℚ) / 100
def interest_B : ℚ := principal_B * interest_rate * time_B
def interest_C (P_C : ℚ) : ℚ := P_C * interest_rate * time_C

theorem money_lent_to_C_is_3000 :
  ∃ P_C : ℚ, interest_B + interest_C P_C = total_interest ∧ P_C = 3000 :=
by
  use 3000
  unfold interest_B interest_C interest_rate principal_B time_B time_C rate_of_interest total_interest
  sorry

end NUMINAMATH_GPT_money_lent_to_C_is_3000_l1694_169483


namespace NUMINAMATH_GPT_scientific_notation_of_21500000_l1694_169428

theorem scientific_notation_of_21500000 :
  21500000 = 2.15 * 10^7 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_of_21500000_l1694_169428


namespace NUMINAMATH_GPT_intersection_in_fourth_quadrant_l1694_169491

theorem intersection_in_fourth_quadrant (m : ℝ) :
  let x := (3 * m + 2) / 4
  let y := (-m - 2) / 8
  (x > 0) ∧ (y < 0) ↔ (m > -2 / 3) :=
by
  sorry

end NUMINAMATH_GPT_intersection_in_fourth_quadrant_l1694_169491


namespace NUMINAMATH_GPT_Steven_has_more_peaches_l1694_169466

variable (Steven_peaches : Nat) (Jill_peaches : Nat)
variable (h1 : Steven_peaches = 19) (h2 : Jill_peaches = 6)

theorem Steven_has_more_peaches : Steven_peaches - Jill_peaches = 13 :=
by
  sorry

end NUMINAMATH_GPT_Steven_has_more_peaches_l1694_169466


namespace NUMINAMATH_GPT_no_quadruples_solution_l1694_169476

theorem no_quadruples_solution (a b c d : ℝ) :
    a^3 + c^3 = 2 ∧
    a^2 * b + c^2 * d = 0 ∧
    b^3 + d^3 = 1 ∧
    a * b^2 + c * d^2 = -6 →
    false :=
by 
  intros h
  sorry

end NUMINAMATH_GPT_no_quadruples_solution_l1694_169476


namespace NUMINAMATH_GPT_cats_left_l1694_169469

theorem cats_left (siamese_cats : ℕ) (house_cats : ℕ) (cats_sold : ℕ) 
  (h1 : siamese_cats = 13) (h2 : house_cats = 5) (h3 : cats_sold = 10) : 
  siamese_cats + house_cats - cats_sold = 8 :=
by
  sorry

end NUMINAMATH_GPT_cats_left_l1694_169469


namespace NUMINAMATH_GPT_students_arrangement_l1694_169427

theorem students_arrangement (B1 B2 S1 S2 T1 T2 C1 C2 : ℕ) :
  (B1 = B2 ∧ S1 ≠ S2 ∧ T1 ≠ T2 ∧ C1 ≠ C2) →
  (C1 ≠ C2) →
  (arrangements = 7200) :=
by
  sorry

end NUMINAMATH_GPT_students_arrangement_l1694_169427


namespace NUMINAMATH_GPT_relationship_between_vars_l1694_169441

-- Define the variables a, b, c, d as real numbers
variables (a b c d : ℝ)

-- Define the initial condition
def initial_condition := (a + 2 * b) / (2 * b + c) = (c + 2 * d) / (2 * d + a)

-- State the theorem to be proved
theorem relationship_between_vars (h : initial_condition a b c d) : 
  a = c ∨ a + c + 2 * (b + d) = 0 :=
sorry

end NUMINAMATH_GPT_relationship_between_vars_l1694_169441


namespace NUMINAMATH_GPT_inverse_matrix_l1694_169418

-- Define the matrix A
def A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![7, -2], ![-3, 1]]

-- Define the supposed inverse B
def B : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![1, 2], ![3, 7]]

-- Define the condition that A * B should yield the identity matrix
theorem inverse_matrix :
  A * B = 1 :=
sorry

end NUMINAMATH_GPT_inverse_matrix_l1694_169418


namespace NUMINAMATH_GPT_remainder_of_power_mod_l1694_169459

noncomputable def carmichael (n : ℕ) : ℕ := sorry  -- Define Carmichael function (as a placeholder)

theorem remainder_of_power_mod :
  ∀ (n : ℕ), carmichael 1000 = 100 → carmichael 100 = 20 → 
    (5 ^ 5 ^ 5 ^ 5) % 1000 = 625 :=
by
  intros n h₁ h₂
  sorry

end NUMINAMATH_GPT_remainder_of_power_mod_l1694_169459


namespace NUMINAMATH_GPT_carter_students_received_grades_l1694_169449

theorem carter_students_received_grades
  (students_thompson : ℕ)
  (a_thompson : ℕ)
  (remaining_students_thompson : ℕ)
  (b_thompson : ℕ)
  (students_carter : ℕ)
  (ratio_A_thompson : ℚ)
  (ratio_B_thompson : ℚ)
  (A_carter : ℕ)
  (B_carter : ℕ) :
  students_thompson = 20 →
  a_thompson = 12 →
  remaining_students_thompson = 8 →
  b_thompson = 5 →
  students_carter = 30 →
  ratio_A_thompson = (a_thompson : ℚ) / students_thompson →
  ratio_B_thompson = (b_thompson : ℚ) / remaining_students_thompson →
  A_carter = ratio_A_thompson * students_carter →
  B_carter = (b_thompson : ℚ) / remaining_students_thompson * (students_carter - A_carter) →
  A_carter = 18 ∧ B_carter = 8 := 
by 
  intros;
  sorry

end NUMINAMATH_GPT_carter_students_received_grades_l1694_169449


namespace NUMINAMATH_GPT_find_r_l1694_169457

theorem find_r (r : ℝ) (h : ⌊r⌋ + r = 20.7) : r = 10.7 := 
by 
  sorry 

end NUMINAMATH_GPT_find_r_l1694_169457


namespace NUMINAMATH_GPT_max_abs_value_l1694_169475

open Complex Real

theorem max_abs_value (z : ℂ) (h : abs (z - 8) + abs (z + 6 * I) = 10) : abs z ≤ 8 :=
sorry

example : ∃ z : ℂ, abs (z - 8) + abs (z + 6 * I) = 10 ∧ abs z = 8 :=
sorry

end NUMINAMATH_GPT_max_abs_value_l1694_169475


namespace NUMINAMATH_GPT_cricket_bat_cost_l1694_169439

variable (CP_A : ℝ) (CP_B : ℝ) (CP_C : ℝ)

-- Conditions
def CP_B_def : Prop := CP_B = 1.20 * CP_A
def CP_C_def : Prop := CP_C = 1.25 * CP_B
def CP_C_val : Prop := CP_C = 234

-- Theorem statement
theorem cricket_bat_cost (h1 : CP_B_def CP_A CP_B) (h2 : CP_C_def CP_B CP_C) (h3 : CP_C_val CP_C) : CP_A = 156 :=by
  sorry

end NUMINAMATH_GPT_cricket_bat_cost_l1694_169439


namespace NUMINAMATH_GPT_calculate_expression_l1694_169412

theorem calculate_expression : 
  (2^1002 + 5^1003)^2 - (2^1002 - 5^1003)^2 = 20 * 10^1002 := 
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l1694_169412


namespace NUMINAMATH_GPT_dave_final_tickets_l1694_169422

variable (initial_tickets_set1_won : ℕ) (initial_tickets_set1_lost : ℕ)
variable (initial_tickets_set2_won : ℕ) (initial_tickets_set2_lost : ℕ)
variable (multiplier_set3 : ℕ)
variable (initial_tickets_set3_lost : ℕ)
variable (used_tickets : ℕ)
variable (additional_tickets : ℕ)

theorem dave_final_tickets :
  let net_gain_set1 := initial_tickets_set1_won - initial_tickets_set1_lost
  let net_gain_set2 := initial_tickets_set2_won - initial_tickets_set2_lost
  let net_gain_set3 := multiplier_set3 * net_gain_set1 - initial_tickets_set3_lost
  let total_tickets_after_sets := net_gain_set1 + net_gain_set2 + net_gain_set3
  let tickets_after_buying := total_tickets_after_sets - used_tickets
  let final_tickets := tickets_after_buying + additional_tickets
  initial_tickets_set1_won = 14 →
  initial_tickets_set1_lost = 2 →
  initial_tickets_set2_won = 8 →
  initial_tickets_set2_lost = 5 →
  multiplier_set3 = 3 →
  initial_tickets_set3_lost = 15 →
  used_tickets = 25 →
  additional_tickets = 7 →
  final_tickets = 18 :=
by
  intros
  sorry

end NUMINAMATH_GPT_dave_final_tickets_l1694_169422


namespace NUMINAMATH_GPT_area_of_new_shape_l1694_169438

noncomputable def unit_equilateral_triangle_area : ℝ :=
  (1 : ℝ)^2 * Real.sqrt 3 / 4

noncomputable def area_removed_each_step (k : ℕ) : ℝ :=
  3 * (4 ^ (k - 1)) * (Real.sqrt 3 / (4 * (9 ^ k)))

noncomputable def total_removed_area : ℝ :=
  ∑' k, 3 * (4 ^ (k - 1)) * (Real.sqrt 3 / (4 * (9 ^ k)))

noncomputable def final_area := unit_equilateral_triangle_area - total_removed_area

theorem area_of_new_shape :
  final_area = Real.sqrt 3 / 10 := sorry

end NUMINAMATH_GPT_area_of_new_shape_l1694_169438


namespace NUMINAMATH_GPT_mans_speed_against_current_l1694_169403

/-- Given the man's speed with the current and the speed of the current, prove the man's speed against the current. -/
theorem mans_speed_against_current
  (speed_with_current : ℝ) (speed_of_current : ℝ)
  (h1 : speed_with_current = 16)
  (h2 : speed_of_current = 3.2) :
  speed_with_current - 2 * speed_of_current = 9.6 :=
sorry

end NUMINAMATH_GPT_mans_speed_against_current_l1694_169403


namespace NUMINAMATH_GPT_pure_alcohol_to_add_l1694_169415

-- Variables and known values
variables (x : ℝ) -- amount of pure alcohol added
def initial_volume : ℝ := 6 -- initial solution volume in liters
def initial_concentration : ℝ := 0.35 -- initial alcohol concentration
def target_concentration : ℝ := 0.50 -- target alcohol concentration

-- Conditions
def initial_pure_alcohol : ℝ := initial_volume * initial_concentration

-- Statement of the problem
theorem pure_alcohol_to_add :
  (2.1 + x) / (initial_volume + x) = target_concentration ↔ x = 1.8 :=
by
  sorry

end NUMINAMATH_GPT_pure_alcohol_to_add_l1694_169415


namespace NUMINAMATH_GPT_centers_of_parallelograms_l1694_169492

def is_skew_lines (l1 l2 l3 l4 : Line) : Prop :=
  -- A function that checks if 4 lines are pairwise skew and no three of them are parallel to the same plane.
  sorry

def count_centers_of_parallelograms (l1 l2 l3 l4 : Line) : ℕ :=
  -- A function that counts the number of lines through which the centers of parallelograms formed by the intersections of the lines pass.
  sorry

theorem centers_of_parallelograms (l1 l2 l3 l4 : Line) (h_skew: is_skew_lines l1 l2 l3 l4) : count_centers_of_parallelograms l1 l2 l3 l4 = 3 :=
  sorry

end NUMINAMATH_GPT_centers_of_parallelograms_l1694_169492


namespace NUMINAMATH_GPT_part1_part2_l1694_169411

-- Part 1: Proving the value of a given f(x) = a/x + 1 and f(-2) = 0
theorem part1 (a : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = a / x + 1) (h2 : f (-2) = 0) : a = 2 := 
by 
-- Placeholder for the proof
sorry

-- Part 2: Proving the value of f(4) given f(x) = 6/x + 1
theorem part2 (f : ℝ → ℝ) (h1 : ∀ x, f x = 6 / x + 1) : f 4 = 5 / 2 := 
by 
-- Placeholder for the proof
sorry

end NUMINAMATH_GPT_part1_part2_l1694_169411


namespace NUMINAMATH_GPT_example_is_fraction_l1694_169453

def is_fraction (a b : ℚ) : Prop := ∃ x y : ℚ, a = x ∧ b = y ∧ y ≠ 0

-- Example condition relevant to the problem
theorem example_is_fraction (x : ℚ) : is_fraction x (x + 2) :=
by
  sorry

end NUMINAMATH_GPT_example_is_fraction_l1694_169453


namespace NUMINAMATH_GPT_oranges_sold_in_the_morning_eq_30_l1694_169446

variable (O : ℝ)  -- Denote the number of oranges Wendy sold in the morning

-- Conditions as assumptions
def price_per_apple : ℝ := 1.5
def price_per_orange : ℝ := 1
def morning_apples_sold : ℝ := 40
def afternoon_apples_sold : ℝ := 50
def afternoon_oranges_sold : ℝ := 40
def total_sales_for_day : ℝ := 205

-- Prove that O, satisfying the given conditions, equals 30
theorem oranges_sold_in_the_morning_eq_30 (h : 
    (morning_apples_sold * price_per_apple) +
    (O * price_per_orange) +
    (afternoon_apples_sold * price_per_apple) +
    (afternoon_oranges_sold * price_per_orange) = 
    total_sales_for_day
  ) : O = 30 :=
by
  sorry

end NUMINAMATH_GPT_oranges_sold_in_the_morning_eq_30_l1694_169446


namespace NUMINAMATH_GPT_inverse_ratio_l1694_169494

theorem inverse_ratio (a b c d : ℝ) :
  (∀ x, x ≠ -6 → (3 * x - 2) / (x + 6) = (a * x + b) / (c * x + d)) →
  a/c = -6 :=
by
  sorry

end NUMINAMATH_GPT_inverse_ratio_l1694_169494


namespace NUMINAMATH_GPT_carpet_area_l1694_169499

/-- A rectangular floor with a length of 15 feet and a width of 12 feet needs 20 square yards of carpet to cover it. -/
theorem carpet_area (length_feet : ℕ) (width_feet : ℕ) (feet_per_yard : ℕ) (length_yards : ℕ) (width_yards : ℕ) (area_sq_yards : ℕ) :
  length_feet = 15 ∧
  width_feet = 12 ∧
  feet_per_yard = 3 ∧
  length_yards = length_feet / feet_per_yard ∧
  width_yards = width_feet / feet_per_yard ∧
  area_sq_yards = length_yards * width_yards → 
  area_sq_yards = 20 :=
by
  sorry

end NUMINAMATH_GPT_carpet_area_l1694_169499


namespace NUMINAMATH_GPT_number_of_girls_l1694_169432

theorem number_of_girls
  (B G : ℕ)
  (ratio_condition : B * 8 = 5 * G)
  (total_condition : B + G = 260) :
  G = 160 :=
by
  sorry

end NUMINAMATH_GPT_number_of_girls_l1694_169432


namespace NUMINAMATH_GPT_part1_part2_l1694_169410

noncomputable def problem1 (x y: ℕ) : Prop := 
  (2 * x + 3 * y = 44) ∧ (4 * x = 5 * y)

noncomputable def solution1 (x y: ℕ) : Prop :=
  (x = 10) ∧ (y = 8)

theorem part1 : ∃ x y: ℕ, problem1 x y → solution1 x y :=
by sorry

noncomputable def problem2 (a b: ℕ) : Prop := 
  25 * (10 * a + 8 * b) = 3500

noncomputable def solution2 (a b: ℕ) : Prop :=
  ((a = 2 ∧ b = 15) ∨ (a = 6 ∧ b = 10) ∨ (a = 10 ∧ b = 5))

theorem part2 : ∃ a b: ℕ, problem2 a b → solution2 a b :=
by sorry

end NUMINAMATH_GPT_part1_part2_l1694_169410


namespace NUMINAMATH_GPT_cube_painting_l1694_169448

theorem cube_painting (n : ℕ) (h1 : n > 3) 
  (h2 : 2 * (n-2) * (n-2) = 4 * (n-2)) :
  n = 4 :=
sorry

end NUMINAMATH_GPT_cube_painting_l1694_169448


namespace NUMINAMATH_GPT_symmetric_line_eq_l1694_169484

theorem symmetric_line_eq (x y : ℝ) :
  (∀ x y, 2 * x - y + 1 = 0 → y = -x) → (∀ x y, x - 2 * y + 1 = 0) :=
by sorry

end NUMINAMATH_GPT_symmetric_line_eq_l1694_169484


namespace NUMINAMATH_GPT_average_loss_per_loot_box_l1694_169405

theorem average_loss_per_loot_box
  (cost_per_loot_box : ℝ := 5)
  (value_standard_item : ℝ := 3.5)
  (probability_rare_item_A : ℝ := 0.05)
  (value_rare_item_A : ℝ := 10)
  (probability_rare_item_B : ℝ := 0.03)
  (value_rare_item_B : ℝ := 15)
  (probability_rare_item_C : ℝ := 0.02)
  (value_rare_item_C : ℝ := 20) 
  : (cost_per_loot_box 
      - (0.90 * value_standard_item 
      + probability_rare_item_A * value_rare_item_A 
      + probability_rare_item_B * value_rare_item_B 
      + probability_rare_item_C * value_rare_item_C)) = 0.50 := by 
  sorry

end NUMINAMATH_GPT_average_loss_per_loot_box_l1694_169405


namespace NUMINAMATH_GPT_temperature_difference_l1694_169477

theorem temperature_difference (T_high T_low : ℝ) (h_high : T_high = 9) (h_low : T_low = -1) : 
  T_high - T_low = 10 :=
by
  rw [h_high, h_low]
  norm_num

end NUMINAMATH_GPT_temperature_difference_l1694_169477


namespace NUMINAMATH_GPT_square_of_105_l1694_169497

/-- Prove that 105^2 = 11025. -/
theorem square_of_105 : 105^2 = 11025 :=
by
  sorry

end NUMINAMATH_GPT_square_of_105_l1694_169497


namespace NUMINAMATH_GPT_larger_root_of_degree_11_l1694_169473

theorem larger_root_of_degree_11 {x : ℝ} :
  (∃ x₁, x₁ > 0 ∧ (x₁ + x₁^2 + x₁^3 + x₁^4 + x₁^5 + x₁^6 + x₁^7 + x₁^8 = 8 - 10 * x₁^9)) ∧
  (∃ x₂, x₂ > 0 ∧ (x₂ + x₂^2 + x₂^3 + x₂^4 + x₂^5 + x₂^6 + x₂^7 + x₂^8 + x₂^9 + x₂^10 = 8 - 10 * x₂^11)) →
  (∃ x₁ x₂, x₁ > 0 ∧ x₂ > 0 ∧
    (x₁ + x₁^2 + x₁^3 + x₁^4 + x₁^5 + x₁^6 + x₁^7 + x₁^8 = 8 - 10 * x₁^9) ∧
    (x₂ + x₂^2 + x₂^3 + x₂^4 + x₂^5 + x₂^6 + x₂^7 + x₂^8 + x₂^9 + x₂^10 = 8 - 10 * x₂^11) ∧
    x₁ < x₂) :=
by
  sorry

end NUMINAMATH_GPT_larger_root_of_degree_11_l1694_169473


namespace NUMINAMATH_GPT_dan_initial_money_l1694_169456

theorem dan_initial_money 
  (cost_chocolate : ℕ) 
  (cost_candy_bar : ℕ) 
  (h1 : cost_chocolate = 3) 
  (h2 : cost_candy_bar = 7)
  (h3 : cost_candy_bar - cost_chocolate = 4) : 
  cost_candy_bar + cost_chocolate = 10 := 
by
  sorry

end NUMINAMATH_GPT_dan_initial_money_l1694_169456


namespace NUMINAMATH_GPT_bruce_and_anne_clean_house_l1694_169495

theorem bruce_and_anne_clean_house 
  (A : ℚ) (B : ℚ) (H1 : A = 1/12) 
  (H2 : 3 * (B + 2 * A) = 1) :
  1 / (B + A) = 4 := 
sorry

end NUMINAMATH_GPT_bruce_and_anne_clean_house_l1694_169495


namespace NUMINAMATH_GPT_find_angle_l1694_169485

theorem find_angle (x : ℝ) (h : 90 - x = 2 * x + 15) : x = 25 :=
by
  sorry

end NUMINAMATH_GPT_find_angle_l1694_169485


namespace NUMINAMATH_GPT_factorize_expression_l1694_169409

-- Define the expression E
def E (x y z : ℝ) : ℝ := x^2 + x*y - x*z - y*z

-- State the theorem to prove \(E = (x + y)(x - z)\)
theorem factorize_expression (x y z : ℝ) : 
  E x y z = (x + y) * (x - z) := 
sorry

end NUMINAMATH_GPT_factorize_expression_l1694_169409


namespace NUMINAMATH_GPT_percentage_increase_in_consumption_l1694_169423

-- Define the conditions
variables {T C : ℝ}  -- T: original tax, C: original consumption
variables (P : ℝ)    -- P: percentage increase in consumption

-- Non-zero conditions
variables (hT : T ≠ 0) (hC : C ≠ 0)

-- Define the Lean theorem
theorem percentage_increase_in_consumption 
  (h : 0.8 * (1 + P / 100) = 0.96) : 
  P = 20 :=
by
  sorry

end NUMINAMATH_GPT_percentage_increase_in_consumption_l1694_169423


namespace NUMINAMATH_GPT_min_2x3y2z_l1694_169401

noncomputable def min_value (x y z : ℝ) : ℝ := 2 * (x^3) * (y^2) * z

theorem min_2x3y2z (x y z : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hz_pos : 0 < z) (h : (1/x) + (1/y) + (1/z) = 9) :
  min_value x y z = 2 / 675 :=
sorry

end NUMINAMATH_GPT_min_2x3y2z_l1694_169401


namespace NUMINAMATH_GPT_sphere_radius_volume_eq_surface_area_l1694_169437

theorem sphere_radius_volume_eq_surface_area (r : ℝ) (h₁ : (4 / 3) * π * r^3 = 4 * π * r^2) : r = 3 :=
by
  sorry

end NUMINAMATH_GPT_sphere_radius_volume_eq_surface_area_l1694_169437


namespace NUMINAMATH_GPT_g_is_odd_l1694_169425

noncomputable def g (x : ℝ) : ℝ := (1 / (3^x - 1)) + (1 / 3)

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x :=
by
  intro x
  sorry

end NUMINAMATH_GPT_g_is_odd_l1694_169425


namespace NUMINAMATH_GPT_find_t_l1694_169451

def vector (α : Type) : Type := (α × α)

def dot_product (v1 v2 : vector ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def orthogonal (v1 v2 : vector ℝ) : Prop :=
  dot_product v1 v2 = 0

theorem find_t (t : ℝ) :
  let a : vector ℝ := (1, -1)
  let b : vector ℝ := (2, t)
  orthogonal a b → t = 2 := by
  sorry

end NUMINAMATH_GPT_find_t_l1694_169451


namespace NUMINAMATH_GPT_sum_of_squares_divisible_by_sum_l1694_169454

theorem sum_of_squares_divisible_by_sum (a b c : ℕ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a) 
    (h_bound : a < 2017 ∧ b < 2017 ∧ c < 2017)
    (h_mod : (a^3 - b^3) % 2017 = 0 ∧ (b^3 - c^3) % 2017 = 0 ∧ (c^3 - a^3) % 2017 = 0) :
    (a^2 + b^2 + c^2) % (a + b + c) = 0 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_divisible_by_sum_l1694_169454


namespace NUMINAMATH_GPT_value_of_x_l1694_169416

theorem value_of_x (x : ℝ) (h : x = 88 + 0.3 * 88) : x = 114.4 :=
by
  sorry

end NUMINAMATH_GPT_value_of_x_l1694_169416


namespace NUMINAMATH_GPT_bears_in_shipment_l1694_169443

theorem bears_in_shipment (initial_bears shipment_bears bears_per_shelf total_shelves : ℕ)
  (h1 : initial_bears = 17)
  (h2 : bears_per_shelf = 9)
  (h3 : total_shelves = 3)
  (h4 : total_shelves * bears_per_shelf = 27) :
  shipment_bears = 10 :=
by
  sorry

end NUMINAMATH_GPT_bears_in_shipment_l1694_169443


namespace NUMINAMATH_GPT_meters_to_kilometers_kilograms_to_grams_centimeters_to_decimeters_hours_to_minutes_l1694_169460

theorem meters_to_kilometers (h : 1 = 1000) : 6000 / 1000 = 6 := by
  sorry

theorem kilograms_to_grams (h : 1 = 1000) : (5 + 2) * 1000 = 7000 := by
  sorry

theorem centimeters_to_decimeters (h : 10 = 1) : (58 + 32) / 10 = 9 := by
  sorry

theorem hours_to_minutes (h : 60 = 1) : 3 * 60 + 30 = 210 := by
  sorry

end NUMINAMATH_GPT_meters_to_kilometers_kilograms_to_grams_centimeters_to_decimeters_hours_to_minutes_l1694_169460


namespace NUMINAMATH_GPT_smallest_n_boxes_l1694_169445

theorem smallest_n_boxes (n : ℕ) : (15 * n - 1) % 11 = 0 ↔ n = 3 :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_boxes_l1694_169445


namespace NUMINAMATH_GPT_find_x_y_n_l1694_169414

def is_reverse_digit (x y : ℕ) : Prop := 
  x / 10 = y % 10 ∧ x % 10 = y / 10

def is_two_digit_nonzero (z : ℕ) : Prop := 
  10 ≤ z ∧ z < 100

theorem find_x_y_n : 
  ∃ (x y n : ℕ), is_two_digit_nonzero x ∧ is_two_digit_nonzero y ∧ is_reverse_digit x y ∧ (x^2 - y^2 = 44 * n) ∧ (x + y + n = 93) :=
sorry

end NUMINAMATH_GPT_find_x_y_n_l1694_169414


namespace NUMINAMATH_GPT_scientific_notation_example_l1694_169434

theorem scientific_notation_example :
  ∃ (a : ℝ) (n : ℤ), (1 ≤ |a| ∧ |a| < 10) ∧ (3650000 : ℝ) = a * 10 ^ n :=
sorry

end NUMINAMATH_GPT_scientific_notation_example_l1694_169434


namespace NUMINAMATH_GPT_length_of_bridge_l1694_169462

-- Definitions based on the conditions
def walking_speed_kmph : ℝ := 10 -- speed in km/hr
def time_minutes : ℝ := 24 -- crossing time in minutes
def conversion_factor_km_to_m : ℝ := 1000
def conversion_factor_hr_to_min : ℝ := 60

-- The main statement to prove
theorem length_of_bridge :
  let walking_speed_m_per_min := walking_speed_kmph * conversion_factor_km_to_m / conversion_factor_hr_to_min;
  walking_speed_m_per_min * time_minutes = 4000 := 
by
  let walking_speed_m_per_min := walking_speed_kmph * conversion_factor_km_to_m / conversion_factor_hr_to_min;
  sorry

end NUMINAMATH_GPT_length_of_bridge_l1694_169462


namespace NUMINAMATH_GPT_age_ratio_rahul_deepak_l1694_169496

/--
Prove that the ratio between Rahul and Deepak's current ages is 4:3 given the following conditions:
1. After 10 years, Rahul's age will be 26 years.
2. Deepak's current age is 12 years.
-/
theorem age_ratio_rahul_deepak (R D : ℕ) (h1 : R + 10 = 26) (h2 : D = 12) : R / D = 4 / 3 :=
by sorry

end NUMINAMATH_GPT_age_ratio_rahul_deepak_l1694_169496


namespace NUMINAMATH_GPT_total_cost_is_103_l1694_169435

-- Base cost of the plan is 20 dollars
def base_cost : ℝ := 20

-- Cost per text message in dollars
def cost_per_text : ℝ := 0.10

-- Cost per minute over 25 hours in dollars
def cost_per_minute_over_limit : ℝ := 0.15

-- Number of text messages sent
def text_messages : ℕ := 200

-- Total hours talked
def hours_talked : ℝ := 32

-- Free minutes (25 hours)
def free_minutes : ℝ := 25 * 60

-- Calculating the extra minutes talked
def extra_minutes : ℝ := (hours_talked * 60) - free_minutes

-- Total cost
def total_cost : ℝ :=
  base_cost +
  (text_messages * cost_per_text) +
  (extra_minutes * cost_per_minute_over_limit)

-- Proving that the total cost is 103 dollars
theorem total_cost_is_103 : total_cost = 103 := by
  sorry

end NUMINAMATH_GPT_total_cost_is_103_l1694_169435


namespace NUMINAMATH_GPT_bake_sale_earnings_eq_400_l1694_169407

/-
  The problem statement derived from the given bake sale problem.
  We are to verify that the bake sale earned 400 dollars.
-/

def total_donation (bake_sale_earnings : ℕ) :=
  ((bake_sale_earnings - 100) / 2) + 10

theorem bake_sale_earnings_eq_400 (X : ℕ) (h : total_donation X = 160) : X = 400 :=
by
  sorry

end NUMINAMATH_GPT_bake_sale_earnings_eq_400_l1694_169407


namespace NUMINAMATH_GPT_polygon_quadrilateral_l1694_169402

theorem polygon_quadrilateral {n : ℕ} (h : (n - 2) * 180 = 360) : n = 4 := by
  sorry

end NUMINAMATH_GPT_polygon_quadrilateral_l1694_169402


namespace NUMINAMATH_GPT_find_a7_l1694_169479

-- Definitions for geometric progression and conditions
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n, ∃ q : ℝ, a n = a 1 * q^(n-1)

def condition1 (a : ℕ → ℝ) : Prop :=
a 2 * a 4 * a 5 = a 3 * a 6

def condition2 (a : ℕ → ℝ) : Prop :=
a 9 * a 10 = -8

-- Theorem stating the equivalence given the conditions
theorem find_a7 (a : ℕ → ℝ) (h : is_geometric_sequence a) (h1 : condition1 a) (h2 : condition2 a) : 
    a 7 = -2 :=
sorry

end NUMINAMATH_GPT_find_a7_l1694_169479


namespace NUMINAMATH_GPT_smallest_number_greater_than_l1694_169430

theorem smallest_number_greater_than : 
  ∀ (S : Set ℝ), S = {0.8, 0.5, 0.3} → 
  (∃ x ∈ S, x > 0.4 ∧ (∀ y ∈ S, y > 0.4 → x ≤ y)) → 
  x = 0.5 :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_greater_than_l1694_169430


namespace NUMINAMATH_GPT_solution_set_of_abs_inequality_l1694_169431

theorem solution_set_of_abs_inequality (x : ℝ) : |x| - |x - 3| < 2 ↔ x < 2.5 :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_abs_inequality_l1694_169431


namespace NUMINAMATH_GPT_arithmetic_sequences_ratio_l1694_169433

theorem arithmetic_sequences_ratio (x y a1 a2 a3 b1 b2 b3 b4 : Real) (hxy : x ≠ y) 
  (h_arith1 : a1 = x + (y - x) / 4 ∧ a2 = x + 2 * (y - x) / 4 ∧ a3 = x + 3 * (y - x) / 4 ∧ y = x + 4 * (y - x) / 4)
  (h_arith2 : b1 = x - (y - x) / 2 ∧ b2 = x + (y - x) / 2 ∧ b3 = x + 2 * (y - x) / 2 ∧ y = x + 2 * (y - x) / 2 ∧ b4 = y + (y - x) / 2):
  (b4 - b3) / (a2 - a1) = 8 / 3 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequences_ratio_l1694_169433


namespace NUMINAMATH_GPT_water_tank_equilibrium_l1694_169421

theorem water_tank_equilibrium :
  (1 / 15 : ℝ) + (1 / 10 : ℝ) - (1 / 6 : ℝ) = 0 :=
by
  sorry

end NUMINAMATH_GPT_water_tank_equilibrium_l1694_169421


namespace NUMINAMATH_GPT_find_A_l1694_169461

theorem find_A : ∃ (A : ℕ), 
  (A > 0) ∧ (A ∣ (270 * 2 - 312)) ∧ (A ∣ (211 * 2 - 270)) ∧ 
  (∃ (rA rB rC : ℕ), 312 % A = rA ∧ 270 % A = rB ∧ 211 % A = rC ∧ 
                      rA = 2 * rB ∧ rB = 2 * rC ∧ A = 19) :=
by sorry

end NUMINAMATH_GPT_find_A_l1694_169461


namespace NUMINAMATH_GPT_total_distance_covered_l1694_169465

theorem total_distance_covered (h : ℝ) : (h > 0) → 
  ∑' n : ℕ, (h * (0.8 : ℝ) ^ n + h * (0.8 : ℝ) ^ (n + 1)) = 5 * h :=
  by
  sorry

end NUMINAMATH_GPT_total_distance_covered_l1694_169465


namespace NUMINAMATH_GPT_range_of_a_for_common_tangents_l1694_169426

theorem range_of_a_for_common_tangents :
  ∃ (a : ℝ), ∀ (x y : ℝ),
    ((x - 2)^2 + y^2 = 4) ∧ ((x - a)^2 + (y + 3)^2 = 9) →
    (-2 < a) ∧ (a < 6) := by
  sorry

end NUMINAMATH_GPT_range_of_a_for_common_tangents_l1694_169426


namespace NUMINAMATH_GPT_find_fraction_l1694_169452

theorem find_fraction :
  ∀ (t k : ℝ) (frac : ℝ),
    t = frac * (k - 32) →
    t = 20 → 
    k = 68 → 
    frac = 5 / 9 :=
by
  intro t k frac h_eq h_t h_k
  -- Start from the conditions and end up showing frac = 5/9
  sorry

end NUMINAMATH_GPT_find_fraction_l1694_169452


namespace NUMINAMATH_GPT_least_number_of_trees_l1694_169487

theorem least_number_of_trees :
  ∃ n : ℕ, (n % 5 = 0) ∧ (n % 6 = 0) ∧ (n % 7 = 0) ∧ n = 210 :=
by
  sorry

end NUMINAMATH_GPT_least_number_of_trees_l1694_169487


namespace NUMINAMATH_GPT_triangle_centers_exist_l1694_169440

structure Triangle (α : Type _) [OrderedCommSemiring α] :=
(A B C : α × α)

noncomputable def circumcenter {α : Type _} [OrderedCommSemiring α] (T : Triangle α) : α × α :=
sorry

noncomputable def incenter {α : Type _} [OrderedCommSemiring α] (T : Triangle α) : α × α :=
sorry

noncomputable def excenter {α : Type _} [OrderedCommSemiring α] (T : Triangle α) : α × α :=
sorry

noncomputable def centroid {α : Type _} [OrderedCommSemiring α] (T : Triangle α) : α × α :=
sorry

theorem triangle_centers_exist {α : Type _} [OrderedCommSemiring α] (T : Triangle α) :
  ∃ K O Oc S : α × α, K = circumcenter T ∧ O = incenter T ∧ Oc = excenter T ∧ S = centroid T :=
by
  refine ⟨circumcenter T, incenter T, excenter T, centroid T, ⟨rfl, rfl, rfl, rfl⟩⟩

end NUMINAMATH_GPT_triangle_centers_exist_l1694_169440


namespace NUMINAMATH_GPT_algebraic_expression_eval_l1694_169478

theorem algebraic_expression_eval (a b c : ℝ) (h : a * (-5:ℝ)^4 + b * (-5)^2 + c = 3): 
  a * (5:ℝ)^4 + b * (5)^2 + c = 3 :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_eval_l1694_169478


namespace NUMINAMATH_GPT_youngest_brother_age_difference_l1694_169468

def Rick_age : ℕ := 15
def Oldest_brother_age : ℕ := 2 * Rick_age
def Middle_brother_age : ℕ := Oldest_brother_age / 3
def Smallest_brother_age : ℕ := Middle_brother_age / 2
def Youngest_brother_age : ℕ := 3

theorem youngest_brother_age_difference :
  Smallest_brother_age - Youngest_brother_age = 2 :=
by
  -- sorry to skip the proof
  sorry

end NUMINAMATH_GPT_youngest_brother_age_difference_l1694_169468


namespace NUMINAMATH_GPT_line_circle_no_intersection_l1694_169464

theorem line_circle_no_intersection :
  ∀ (x y : ℝ), (3 * x + 4 * y = 12) ∧ (x^2 + y^2 = 4) → false :=
by
  -- Sorry to skip the actual proof
  sorry

end NUMINAMATH_GPT_line_circle_no_intersection_l1694_169464


namespace NUMINAMATH_GPT_trapezoid_height_l1694_169436

variables (a b h : ℝ)

def is_trapezoid (a b h : ℝ) (angle_diag : ℝ) (angle_ext : ℝ) : Prop :=
a < b ∧ angle_diag = 90 ∧ angle_ext = 45

theorem trapezoid_height
  (a b : ℝ) (ha : a < b)
  (angle_diag : ℝ) (h_angle_diag : angle_diag = 90)
  (angle_ext : ℝ) (h_angle_ext : angle_ext = 45)
  (h_def : is_trapezoid a b h angle_diag angle_ext) :
  h = a * b / (b - a) :=
sorry

end NUMINAMATH_GPT_trapezoid_height_l1694_169436


namespace NUMINAMATH_GPT_side_length_is_prime_l1694_169442

-- Define the integer side length of the square
variable (a : ℕ)

-- Define the conditions
def impossible_rectangle (m n : ℕ) : Prop :=
  m * n = a^2 ∧ m ≠ 1 ∧ n ≠ 1

-- Declare the theorem to be proved
theorem side_length_is_prime (h : ∀ m n : ℕ, impossible_rectangle a m n → false) : Nat.Prime a := sorry

end NUMINAMATH_GPT_side_length_is_prime_l1694_169442
