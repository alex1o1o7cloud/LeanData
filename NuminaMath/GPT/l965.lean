import Mathlib

namespace NUMINAMATH_GPT_smallest_n_l965_96590

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def condition_for_n (n : ℕ) : Prop :=
  ∀ k : ℕ, k ≥ n → ∀ x : ℕ, x ∈ M k → ∃ y : ℕ, y ∈ M k ∧ y ≠ x ∧ is_perfect_square (x + y)
  where M (k : ℕ) := { m : ℕ | m > 0 ∧ m ≤ k }

theorem smallest_n : ∃ n : ℕ, (condition_for_n n) ∧ (∀ m < n, ¬ condition_for_n m) :=
  sorry

end NUMINAMATH_GPT_smallest_n_l965_96590


namespace NUMINAMATH_GPT_jill_has_1_more_peach_than_jake_l965_96585

theorem jill_has_1_more_peach_than_jake
    (jill_peaches : ℕ)
    (steven_peaches : ℕ)
    (jake_peaches : ℕ)
    (h1 : jake_peaches = steven_peaches - 16)
    (h2 : steven_peaches = jill_peaches + 15)
    (h3 : jill_peaches = 12) :
    12 - (steven_peaches - 16) = 1 := 
sorry

end NUMINAMATH_GPT_jill_has_1_more_peach_than_jake_l965_96585


namespace NUMINAMATH_GPT_solve_quadratic_equation_1_solve_quadratic_equation_2_l965_96578

theorem solve_quadratic_equation_1 (x : ℝ) :
  3 * x^2 + 2 * x - 1 = 0 ↔ x = 1/3 ∨ x = -1 :=
by sorry

theorem solve_quadratic_equation_2 (x : ℝ) :
  (x + 2) * (x - 3) = 5 * x - 15 ↔ x = 3 :=
by sorry

end NUMINAMATH_GPT_solve_quadratic_equation_1_solve_quadratic_equation_2_l965_96578


namespace NUMINAMATH_GPT_boards_nailing_l965_96544

variables {x y a b : ℕ}

theorem boards_nailing (h1 : 2 * x + 3 * y = 87)
                       (h2 : 3 * a + 5 * b = 94) :
                       x + y = 30 ∧ a + b = 30 :=
sorry

end NUMINAMATH_GPT_boards_nailing_l965_96544


namespace NUMINAMATH_GPT_average_abc_l965_96502

theorem average_abc (A B C : ℚ) 
  (h1 : 2002 * C - 3003 * A = 6006) 
  (h2 : 2002 * B + 4004 * A = 8008) 
  (h3 : B - C = A + 1) :
  (A + B + C) / 3 = 7 / 3 := 
sorry

end NUMINAMATH_GPT_average_abc_l965_96502


namespace NUMINAMATH_GPT_series_sum_l965_96594

variable (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_gt : b < a)

noncomputable def infinite_series : ℝ := 
∑' n, 1 / ( ((n - 1) * a^2 - (n - 2) * b^2) * (n * a^2 - (n - 1) * b^2) )

theorem series_sum : infinite_series a b = 1 / ((a^2 - b^2) * b^2) := 
by 
  sorry

end NUMINAMATH_GPT_series_sum_l965_96594


namespace NUMINAMATH_GPT_no_solution_exists_l965_96549

theorem no_solution_exists : ¬ ∃ (n m : ℕ), (n + 1) * (2 * n + 1) = 2 * m^2 := by sorry

end NUMINAMATH_GPT_no_solution_exists_l965_96549


namespace NUMINAMATH_GPT_speed_of_water_l965_96565

theorem speed_of_water (v : ℝ) :
  (∀ (distance time : ℝ), distance = 16 ∧ time = 8 → distance = (4 - v) * time) → 
  v = 2 :=
by
  intro h
  have h1 : 16 = (4 - v) * 8 := h 16 8 (by simp)
  sorry

end NUMINAMATH_GPT_speed_of_water_l965_96565


namespace NUMINAMATH_GPT_frank_total_pages_read_l965_96598

-- Definitions of given conditions
def first_book_pages (pages_per_day : ℕ) (days : ℕ) := pages_per_day * days
def second_book_pages (pages_per_day : ℕ) (days : ℕ) := pages_per_day * days
def third_book_pages (pages_per_day : ℕ) (days : ℕ) := pages_per_day * days

-- Given values
def pages_first_book := first_book_pages 22 569
def pages_second_book := second_book_pages 35 315
def pages_third_book := third_book_pages 18 450

-- Total number of pages read by Frank
def total_pages := pages_first_book + pages_second_book + pages_third_book

-- Statement to prove
theorem frank_total_pages_read : total_pages = 31643 := by
  sorry

end NUMINAMATH_GPT_frank_total_pages_read_l965_96598


namespace NUMINAMATH_GPT_Sasha_added_digit_l965_96552

noncomputable def Kolya_number : Nat := 45 -- Sum of all digits 0 to 9

theorem Sasha_added_digit (d x : Nat) (h : 0 ≤ d ∧ d ≤ 9) (h1 : 0 ≤ x ∧ x ≤ 9) (condition : Kolya_number - d + x ≡ 0 [MOD 9]) : x = 0 ∨ x = 9 := 
sorry

end NUMINAMATH_GPT_Sasha_added_digit_l965_96552


namespace NUMINAMATH_GPT_tan_periodic_n_solution_l965_96516

open Real

theorem tan_periodic_n_solution :
  ∃ n : ℤ, -180 < n ∧ n < 180 ∧ tan (n * (π / 180)) = tan (1540 * (π / 180)) ∧ n = 40 :=
by
  sorry

end NUMINAMATH_GPT_tan_periodic_n_solution_l965_96516


namespace NUMINAMATH_GPT_point_in_at_least_15_circles_l965_96579

theorem point_in_at_least_15_circles
  (C : Fin 100 → Set (ℝ × ℝ))
  (h1 : ∀ i j, ∃ p, p ∈ C i ∧ p ∈ C j)
  : ∃ p, ∃ S : Finset (Fin 100), S.card ≥ 15 ∧ ∀ i ∈ S, p ∈ C i :=
sorry

end NUMINAMATH_GPT_point_in_at_least_15_circles_l965_96579


namespace NUMINAMATH_GPT_seventh_graders_count_l965_96559

-- Define the problem conditions
def total_students (T : ℝ) : Prop := 0.38 * T = 76
def seventh_grade_ratio : ℝ := 0.32
def seventh_graders (S : ℝ) (T : ℝ) : Prop := S = seventh_grade_ratio * T

-- The goal statement
theorem seventh_graders_count {T S : ℝ} (h : total_students T) : seventh_graders S T → S = 64 :=
by
  sorry

end NUMINAMATH_GPT_seventh_graders_count_l965_96559


namespace NUMINAMATH_GPT_exists_same_color_rectangle_l965_96517

open Finset

-- Define the grid size
def gridSize : ℕ := 12

-- Define the type of colors
inductive Color
| red
| white
| blue

-- Define a point in the grid
structure Point :=
(x : ℕ)
(y : ℕ)
(hx : x ≥ 1 ∧ x ≤ gridSize)
(hy : y ≥ 1 ∧ y ≤ gridSize)

-- Assume a coloring function
def color (p : Point) : Color := sorry

-- The theorem statement
theorem exists_same_color_rectangle :
  ∃ (p1 p2 p3 p4 : Point),
    p1.x = p2.x ∧ p3.x = p4.x ∧
    p1.y = p3.y ∧ p2.y = p4.y ∧
    color p1 = color p2 ∧
    color p1 = color p3 ∧
    color p1 = color p4 :=
sorry

end NUMINAMATH_GPT_exists_same_color_rectangle_l965_96517


namespace NUMINAMATH_GPT_third_smallest_abc_sum_l965_96545

-- Define the necessary conditions and properties
def isIntegerRoots (a b c : ℕ) : Prop :=
  ∃ r1 r2 r3 r4 : ℤ, 
    a * r1^2 + b * r1 + c = 0 ∧ a * r2^2 + b * r2 - c = 0 ∧ 
    a * r3^2 - b * r3 + c = 0 ∧ a * r4^2 - b * r4 - c = 0

-- State the main theorem
theorem third_smallest_abc_sum : ∃ a b c : ℕ, 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ isIntegerRoots a b c ∧ 
  (a + b + c = 35 ∧ a = 1 ∧ b = 10 ∧ c = 24) :=
by sorry

end NUMINAMATH_GPT_third_smallest_abc_sum_l965_96545


namespace NUMINAMATH_GPT_students_like_basketball_l965_96583

variable (B C B_inter_C B_union_C : ℕ)

theorem students_like_basketball (hC : C = 8) (hB_inter_C : B_inter_C = 3) (hB_union_C : B_union_C = 17) 
    (h_incl_excl : B_union_C = B + C - B_inter_C) : B = 12 := by 
  -- Given: 
  --   C = 8
  --   B_inter_C = 3
  --   B_union_C = 17
  --   B_union_C = B + C - B_inter_C
  -- Prove: 
  --   B = 12
  sorry

end NUMINAMATH_GPT_students_like_basketball_l965_96583


namespace NUMINAMATH_GPT_regular_polygon_sides_l965_96563

theorem regular_polygon_sides (n : ℕ) (h : ∀ (i : ℕ), i < n → (160 : ℝ) = (180 * (n - 2)) / n) : n = 18 := 
by 
  sorry

end NUMINAMATH_GPT_regular_polygon_sides_l965_96563


namespace NUMINAMATH_GPT_Pam_read_more_than_Harrison_l965_96518

theorem Pam_read_more_than_Harrison :
  ∀ (assigned : ℕ) (Harrison : ℕ) (Pam : ℕ) (Sam : ℕ),
    assigned = 25 →
    Harrison = assigned + 10 →
    Sam = 2 * Pam →
    Sam = 100 →
    Pam - Harrison = 15 :=
by
  intros assigned Harrison Pam Sam h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_Pam_read_more_than_Harrison_l965_96518


namespace NUMINAMATH_GPT_david_has_15_shells_l965_96537

-- Definitions from the conditions
def mia_shells (david_shells : ℕ) : ℕ := 4 * david_shells
def ava_shells (david_shells : ℕ) : ℕ := mia_shells david_shells + 20
def alice_shells (david_shells : ℕ) : ℕ := (ava_shells david_shells) / 2

-- Total number of shells
def total_shells (david_shells : ℕ) : ℕ := david_shells + mia_shells david_shells + ava_shells david_shells + alice_shells david_shells

-- Proving the number of shells David has is 15 given the total number of shells is 195
theorem david_has_15_shells : total_shells 15 = 195 :=
by
  sorry

end NUMINAMATH_GPT_david_has_15_shells_l965_96537


namespace NUMINAMATH_GPT_find_number_of_dimes_l965_96510

def total_value (pennies nickels dimes quarters half_dollars : Nat) : Nat :=
  pennies * 1 + nickels * 5 + dimes * 10 + quarters * 25 + half_dollars * 50

def number_of_coins (pennies nickels dimes quarters half_dollars : Nat) : Nat :=
  pennies + nickels + dimes + quarters + half_dollars

theorem find_number_of_dimes
  (pennies nickels dimes quarters half_dollars : Nat)
  (h_value : total_value pennies nickels dimes quarters half_dollars = 163)
  (h_coins : number_of_coins pennies nickels dimes quarters half_dollars = 13)
  (h_penny : 1 ≤ pennies)
  (h_nickel : 1 ≤ nickels)
  (h_dime : 1 ≤ dimes)
  (h_quarter : 1 ≤ quarters)
  (h_half_dollar : 1 ≤ half_dollars) :
  dimes = 3 :=
sorry

end NUMINAMATH_GPT_find_number_of_dimes_l965_96510


namespace NUMINAMATH_GPT_exists_N_binary_representation_l965_96508

theorem exists_N_binary_representation (n p : ℕ) (h_composite : ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0) (h_proper_divisor : p > 0 ∧ p < n ∧ n % p = 0) :
  ∃ N : ℕ, ((1 + 2^p + 2^(n-p)) * N) % 2^n = 1 % 2^n :=
by
  sorry

end NUMINAMATH_GPT_exists_N_binary_representation_l965_96508


namespace NUMINAMATH_GPT_incorrect_statement_C_l965_96591

theorem incorrect_statement_C :
  (∀ b h : ℕ, (2 * b) * h = 2 * (b * h)) ∧
  (∀ b h : ℕ, (1 / 2) * b * (2 * h) = 2 * ((1 / 2) * b * h)) ∧
  (∀ r : ℕ, (π * (2 * r) ^ 2 ≠ 2 * (π * r ^ 2))) ∧
  (∀ a b : ℕ, (a / 2) / (2 * b) ≠ a / b) ∧
  (∀ x : ℤ, x < 0 -> 2 * x < x) →
  false :=
by
  intros h
  sorry

end NUMINAMATH_GPT_incorrect_statement_C_l965_96591


namespace NUMINAMATH_GPT_eval_g_231_l965_96546

def g (a b c : ℤ) : ℚ :=
  (c ^ 2 + a ^ 2) / (c - b)

theorem eval_g_231 : g 2 (-3) 1 = 5 / 4 :=
by
  sorry

end NUMINAMATH_GPT_eval_g_231_l965_96546


namespace NUMINAMATH_GPT_proof_equivalence_l965_96582

variable {x y : ℝ}

theorem proof_equivalence (h : x - y = 1) : x^3 - 3 * x * y - y^3 = 1 := by
  sorry

end NUMINAMATH_GPT_proof_equivalence_l965_96582


namespace NUMINAMATH_GPT_mass_percentage_of_carbon_in_ccl4_l965_96573

-- Define the atomic masses
def atomic_mass_c : Float := 12.01
def atomic_mass_cl : Float := 35.45

-- Define the molecular composition of Carbon Tetrachloride (CCl4)
def mol_mass_ccl4 : Float := (1 * atomic_mass_c) + (4 * atomic_mass_cl)

-- Theorem to prove the mass percentage of carbon in Carbon Tetrachloride is 7.81%
theorem mass_percentage_of_carbon_in_ccl4 : 
  (atomic_mass_c / mol_mass_ccl4) * 100 = 7.81 := by
  sorry

end NUMINAMATH_GPT_mass_percentage_of_carbon_in_ccl4_l965_96573


namespace NUMINAMATH_GPT_ratio_of_volumes_l965_96547

theorem ratio_of_volumes (rC hC rD hD : ℝ) (h1 : rC = 10) (h2 : hC = 25) (h3 : rD = 25) (h4 : hD = 10) : 
  (1/3 * Real.pi * rC^2 * hC) / (1/3 * Real.pi * rD^2 * hD) = 2 / 5 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_volumes_l965_96547


namespace NUMINAMATH_GPT_valid_twenty_letter_words_l965_96575

noncomputable def number_of_valid_words : ℕ := sorry

theorem valid_twenty_letter_words :
  number_of_valid_words = 3 * 2^18 := sorry

end NUMINAMATH_GPT_valid_twenty_letter_words_l965_96575


namespace NUMINAMATH_GPT_problem_x_value_l965_96523

theorem problem_x_value (x : ℝ) (h : 0.25 * x = 0.15 * 1500 - 15) : x = 840 :=
by
  sorry

end NUMINAMATH_GPT_problem_x_value_l965_96523


namespace NUMINAMATH_GPT_plane_through_Ox_and_point_plane_parallel_Oz_and_points_l965_96566

-- Definitions for first plane problem
def plane1_through_Ox_axis (y z : ℝ) : Prop := 3 * y + 2 * z = 0

-- Definitions for second plane problem
def plane2_parallel_Oz (x y : ℝ) : Prop := x + 3 * y - 1 = 0

theorem plane_through_Ox_and_point : plane1_through_Ox_axis 2 (-3) := 
by {
  -- Hint: Prove that substituting y = 2 and z = -3 in the equation results in LHS equals RHS.
  -- proof
  sorry 
}

theorem plane_parallel_Oz_and_points : 
  plane2_parallel_Oz 1 0 ∧ plane2_parallel_Oz (-2) 1 :=
by {
  -- Hint: Prove that substituting the points (1, 0) and (-2, 1) in the equation results in LHS equals RHS.
  -- proof
  sorry
}

end NUMINAMATH_GPT_plane_through_Ox_and_point_plane_parallel_Oz_and_points_l965_96566


namespace NUMINAMATH_GPT_transport_tax_correct_l965_96572

def engine_power : ℕ := 250
def tax_rate : ℕ := 75
def months_owned : ℕ := 2
def months_in_year : ℕ := 12
def annual_tax : ℕ := engine_power * tax_rate
def adjusted_tax : ℕ := (annual_tax * months_owned) / months_in_year

theorem transport_tax_correct :
  adjusted_tax = 3125 :=
by
  sorry

end NUMINAMATH_GPT_transport_tax_correct_l965_96572


namespace NUMINAMATH_GPT_gain_percentage_is_66_67_l965_96548

variable (C S : ℝ)
variable (cost_price_eq : 20 * C = 12 * S)

theorem gain_percentage_is_66_67 (h : 20 * C = 12 * S) : (((5 / 3) * C - C) / C) * 100 = 66.67 := by
  sorry

end NUMINAMATH_GPT_gain_percentage_is_66_67_l965_96548


namespace NUMINAMATH_GPT_sum_of_three_quadratics_no_rot_l965_96571

def quad_poly_sum_no_root (p q : ℝ -> ℝ) : Prop :=
  ∀ x : ℝ, (p x + q x ≠ 0)

theorem sum_of_three_quadratics_no_rot (a b c d e f : ℝ)
    (h1 : quad_poly_sum_no_root (λ x => x^2 + a*x + b) (λ x => x^2 + c*x + d))
    (h2 : quad_poly_sum_no_root (λ x => x^2 + c*x + d) (λ x => x^2 + e*x + f))
    (h3 : quad_poly_sum_no_root (λ x => x^2 + e*x + f) (λ x => x^2 + a*x + b)) :
    quad_poly_sum_no_root (λ x => x^2 + a*x + b) 
                         (λ x => x^2 + c*x + d + x^2 + e*x + f) :=
sorry

end NUMINAMATH_GPT_sum_of_three_quadratics_no_rot_l965_96571


namespace NUMINAMATH_GPT_total_leaves_l965_96534

theorem total_leaves (ferns fronds leaves : ℕ) (h1 : ferns = 12) (h2 : fronds = 15) (h3 : leaves = 45) :
  ferns * fronds * leaves = 8100 :=
by
  sorry

end NUMINAMATH_GPT_total_leaves_l965_96534


namespace NUMINAMATH_GPT_complex_number_in_second_quadrant_l965_96558

open Complex

theorem complex_number_in_second_quadrant (z : ℂ) :
  (Complex.abs z = Real.sqrt 7) →
  (z.re < 0 ∧ z.im > 0) →
  z = -2 + Real.sqrt 3 * Complex.I :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_complex_number_in_second_quadrant_l965_96558


namespace NUMINAMATH_GPT_cylinder_has_no_triangular_cross_section_l965_96553

inductive GeometricSolid
  | cylinder
  | cone
  | triangularPrism
  | cube

open GeometricSolid

-- Define the cross section properties
def can_have_triangular_cross_section (s : GeometricSolid) : Prop :=
  s = cone ∨ s = triangularPrism ∨ s = cube

-- Define the property where a solid cannot have a triangular cross-section
def cannot_have_triangular_cross_section (s : GeometricSolid) : Prop :=
  s = cylinder

theorem cylinder_has_no_triangular_cross_section :
  cannot_have_triangular_cross_section cylinder ∧
  ¬ can_have_triangular_cross_section cylinder :=
by
  -- This is where we state the proof goal
  sorry

end NUMINAMATH_GPT_cylinder_has_no_triangular_cross_section_l965_96553


namespace NUMINAMATH_GPT_cubic_inequality_solution_l965_96511

theorem cubic_inequality_solution (x : ℝ) : x^3 - 12 * x^2 + 27 * x > 0 ↔ (0 < x ∧ x < 3) ∨ (9 < x) :=
by sorry

end NUMINAMATH_GPT_cubic_inequality_solution_l965_96511


namespace NUMINAMATH_GPT_multiplication_is_247_l965_96515

theorem multiplication_is_247 (a b : ℕ) (ha : 10 ≤ a ∧ a < 100) (hb : 10 ≤ b ∧ b < 100) (h : a * b = 247) : 
a = 13 ∧ b = 19 :=
by sorry

end NUMINAMATH_GPT_multiplication_is_247_l965_96515


namespace NUMINAMATH_GPT_total_amount_paid_correct_l965_96562

-- Definitions of quantities and rates
def quantity_grapes := 3
def rate_grapes := 70
def quantity_mangoes := 9
def rate_mangoes := 55

-- Total amount calculation
def total_amount_paid := quantity_grapes * rate_grapes + quantity_mangoes * rate_mangoes

-- Theorem to prove total amount paid is 705
theorem total_amount_paid_correct : total_amount_paid = 705 :=
by
  sorry

end NUMINAMATH_GPT_total_amount_paid_correct_l965_96562


namespace NUMINAMATH_GPT_solve_system_of_equations_l965_96520

theorem solve_system_of_equations :
  ∃ (x y : ℝ),
    (5 * x^2 - 14 * x * y + 10 * y^2 = 17) ∧ (4 * x^2 - 10 * x * y + 6 * y^2 = 8) ∧
    ((x = -1 ∧ y = -2) ∨ (x = 11 ∧ y = 7) ∨ (x = -11 ∧ y = -7) ∨ (x = 1 ∧ y = 2)) :=
by
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_l965_96520


namespace NUMINAMATH_GPT_find_pairs_l965_96507

-- Define predicative statements for the conditions
def is_integer (x : ℝ) : Prop :=
  ∃ (n : ℤ), x = n

def condition1 (m n : ℕ) : Prop := 
  (n^2 + 1) % (2 * m) = 0

def condition2 (m n : ℕ) : Prop := 
  is_integer (Real.sqrt (2^(n-1) + m + 4))

-- The goal is to find the pairs of positive integers
theorem find_pairs (m n : ℕ) (h1: condition1 m n) (h2: condition2 m n) : 
  (m = 61 ∧ n = 11) :=
sorry

end NUMINAMATH_GPT_find_pairs_l965_96507


namespace NUMINAMATH_GPT_pencils_difference_l965_96528

theorem pencils_difference
  (pencils_in_backpack : ℕ := 2)
  (pencils_at_home : ℕ := 15) :
  pencils_at_home - pencils_in_backpack = 13 := by
  sorry

end NUMINAMATH_GPT_pencils_difference_l965_96528


namespace NUMINAMATH_GPT_largest_of_five_consecutive_non_primes_under_40_l965_96555

noncomputable def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n 

theorem largest_of_five_consecutive_non_primes_under_40 :
  ∃ x, (x > 9) ∧ (x + 4 < 40) ∧ 
       (¬ is_prime x) ∧
       (¬ is_prime (x + 1)) ∧
       (¬ is_prime (x + 2)) ∧
       (¬ is_prime (x + 3)) ∧
       (¬ is_prime (x + 4)) ∧
       (x + 4 = 36) :=
sorry

end NUMINAMATH_GPT_largest_of_five_consecutive_non_primes_under_40_l965_96555


namespace NUMINAMATH_GPT_number_of_buckets_l965_96593

-- Defining the conditions
def total_mackerels : ℕ := 27
def mackerels_per_bucket : ℕ := 3

-- The theorem to prove
theorem number_of_buckets :
  total_mackerels / mackerels_per_bucket = 9 :=
sorry

end NUMINAMATH_GPT_number_of_buckets_l965_96593


namespace NUMINAMATH_GPT_total_books_correct_l965_96576

-- Define the number of books each person has
def joan_books : ℕ := 10
def tom_books : ℕ := 38
def lisa_books : ℕ := 27
def steve_books : ℕ := 45

-- Calculate the total number of books they have together
def total_books : ℕ := joan_books + tom_books + lisa_books + steve_books

-- State the theorem that needs to be proved
theorem total_books_correct : total_books = 120 :=
by
  sorry

end NUMINAMATH_GPT_total_books_correct_l965_96576


namespace NUMINAMATH_GPT_calculation_equality_l965_96570

theorem calculation_equality : ((8^5 / 8^2) * 4^4) = 2^17 := by
  sorry

end NUMINAMATH_GPT_calculation_equality_l965_96570


namespace NUMINAMATH_GPT_jesse_max_correct_answers_l965_96556

theorem jesse_max_correct_answers :
  ∃ a b c : ℕ, a + b + c = 60 ∧ 5 * a - 2 * c = 150 ∧ a ≤ 38 :=
sorry

end NUMINAMATH_GPT_jesse_max_correct_answers_l965_96556


namespace NUMINAMATH_GPT_original_number_increase_l965_96535

theorem original_number_increase (x : ℝ) (h : 1.20 * x = 1800) : x = 1500 :=
by
  sorry

end NUMINAMATH_GPT_original_number_increase_l965_96535


namespace NUMINAMATH_GPT_polynomial_has_real_root_l965_96514

theorem polynomial_has_real_root (b : ℝ) : ∃ x : ℝ, x^3 + b * x^2 - 4 * x + b = 0 := 
sorry

end NUMINAMATH_GPT_polynomial_has_real_root_l965_96514


namespace NUMINAMATH_GPT_integer_solution_inequalities_l965_96541

theorem integer_solution_inequalities (x : ℤ) (h1 : x + 12 > 14) (h2 : -3 * x > -9) : x = 2 :=
by
  sorry

end NUMINAMATH_GPT_integer_solution_inequalities_l965_96541


namespace NUMINAMATH_GPT_vectors_parallel_solution_l965_96581

theorem vectors_parallel_solution (x : ℝ) (a b : ℝ × ℝ) (h1 : a = (2, x)) (h2 : b = (x, 8)) (h3 : ∃ k, b = (k * 2, k * x)) : x = 4 ∨ x = -4 :=
by
  sorry

end NUMINAMATH_GPT_vectors_parallel_solution_l965_96581


namespace NUMINAMATH_GPT_find_a_l965_96536

noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 - 9 * Real.log x

def monotonically_decreasing (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ I → y ∈ I → x < y → f y ≤ f x

def valid_interval (a : ℝ) : Prop :=
  monotonically_decreasing f (Set.Icc (a-1) (a+1))

theorem find_a :
  {a : ℝ | valid_interval a} = {a : ℝ | 1 < a ∧ a ≤ 2} :=
by
  sorry

end NUMINAMATH_GPT_find_a_l965_96536


namespace NUMINAMATH_GPT_problem_a_l965_96504

def continuous (f : ℝ → ℝ) : Prop := sorry -- Assume this is properly defined somewhere in Mathlib
def monotonic (f : ℝ → ℝ) : Prop := sorry -- Assume this is properly defined somewhere in Mathlib

theorem problem_a :
  ¬ (∀ (f : ℝ → ℝ), continuous f ∧ (∀ y, ∃ x, f x = y) → monotonic f) := sorry

end NUMINAMATH_GPT_problem_a_l965_96504


namespace NUMINAMATH_GPT_determine_missing_digits_l965_96584

theorem determine_missing_digits :
  (237 * 0.31245 = 7430.65) := 
by 
  sorry

end NUMINAMATH_GPT_determine_missing_digits_l965_96584


namespace NUMINAMATH_GPT_solve_n_l965_96522

open Nat

def condition (n : ℕ) : Prop := 2^(n + 1) * 2^3 = 2^10

theorem solve_n (n : ℕ) (hn_pos : 0 < n) (h_cond : condition n) : n = 6 :=
by
  sorry

end NUMINAMATH_GPT_solve_n_l965_96522


namespace NUMINAMATH_GPT_find_constant_a_l965_96587

theorem find_constant_a (a : ℚ) (S : ℕ → ℚ) (hS : ∀ n, S n = (a - 2) * 3^(n + 1) + 2) : a = 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_constant_a_l965_96587


namespace NUMINAMATH_GPT_inequality_proof_l965_96550

open Real

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (ab / (a + b)^2 + bc / (b + c)^2 + ca / (c + a)^2) + (3 * (a^2 + b^2 + c^2)) / (a + b + c)^2 ≥ 7 / 4 := 
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l965_96550


namespace NUMINAMATH_GPT_value_of_g_at_2_l965_96588

def g (x : ℝ) : ℝ := x^2 - 4 * x + 4

theorem value_of_g_at_2 : g 2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_value_of_g_at_2_l965_96588


namespace NUMINAMATH_GPT_fraction_to_terminating_decimal_l965_96567

-- Lean statement for the mathematical problem
theorem fraction_to_terminating_decimal: (13 : ℚ) / 200 = 0.26 := 
sorry

end NUMINAMATH_GPT_fraction_to_terminating_decimal_l965_96567


namespace NUMINAMATH_GPT_find_m_value_l965_96538

theorem find_m_value (m : ℤ) (h1 : m - 2 ≠ 0) (h2 : |m| = 2) : m = -2 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_m_value_l965_96538


namespace NUMINAMATH_GPT_emma_bank_account_balance_l965_96503

theorem emma_bank_account_balance
  (initial_balance : ℕ)
  (daily_spend : ℕ)
  (days_in_week : ℕ)
  (unit_bill : ℕ) :
  initial_balance = 100 → daily_spend = 8 → days_in_week = 7 → unit_bill = 5 →
  (initial_balance - daily_spend * days_in_week) % unit_bill = 4 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_emma_bank_account_balance_l965_96503


namespace NUMINAMATH_GPT_geometric_sequence_common_ratio_l965_96501

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ)
    (h1 : a 2 = a 1 * q)
    (h2 : a 5 = a 1 * q ^ 4)
    (h3 : a 2 = 8)
    (h4 : a 5 = 64) :
    q = 2 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_common_ratio_l965_96501


namespace NUMINAMATH_GPT_speed_difference_l965_96592

theorem speed_difference :
  let distance : ℝ := 8
  let zoe_time_hours : ℝ := 2 / 3
  let john_time_hours : ℝ := 1
  let zoe_speed : ℝ := distance / zoe_time_hours
  let john_speed : ℝ := distance / john_time_hours
  zoe_speed - john_speed = 4 :=
by
  sorry

end NUMINAMATH_GPT_speed_difference_l965_96592


namespace NUMINAMATH_GPT_findPositiveRealSolutions_l965_96543

noncomputable def onlySolutions (a b c d : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
  (a^2 - b * d) / (b + 2 * c + d) +
  (b^2 - c * a) / (c + 2 * d + a) +
  (c^2 - d * b) / (d + 2 * a + b) +
  (d^2 - a * c) / (a + 2 * b + c) = 0

theorem findPositiveRealSolutions :
  ∀ a b c d : ℝ,
  onlySolutions a b c d →
  ∃ k m : ℝ, k > 0 ∧ m > 0 ∧ a = k ∧ b = m ∧ c = k ∧ d = m :=
by
  intros a b c d h
  -- proof steps (if required) go here
  sorry

end NUMINAMATH_GPT_findPositiveRealSolutions_l965_96543


namespace NUMINAMATH_GPT_sin_double_angle_identity_l965_96531

open Real

theorem sin_double_angle_identity (α : ℝ) (h : sin (α - π / 4) = 3 / 5) : sin (2 * α) = 7 / 25 :=
by
  sorry

end NUMINAMATH_GPT_sin_double_angle_identity_l965_96531


namespace NUMINAMATH_GPT_pow_mul_eq_add_l965_96500

theorem pow_mul_eq_add (a : ℝ) : a^3 * a^4 = a^7 :=
by
  -- This is where the proof would go.
  sorry

end NUMINAMATH_GPT_pow_mul_eq_add_l965_96500


namespace NUMINAMATH_GPT_james_speed_is_16_l965_96551

theorem james_speed_is_16
  (distance : ℝ)
  (time : ℝ)
  (distance_eq : distance = 80)
  (time_eq : time = 5) :
  (distance / time = 16) :=
by
  rw [distance_eq, time_eq]
  norm_num

end NUMINAMATH_GPT_james_speed_is_16_l965_96551


namespace NUMINAMATH_GPT_solve_equation1_solve_equation2_l965_96561

theorem solve_equation1 :
  ∀ x : ℝ, ((x-1) * (x-1) = 3 * (x-1)) ↔ (x = 1 ∨ x = 4) :=
by
  intro x
  sorry

theorem solve_equation2 :
  ∀ x : ℝ, (x^2 - 4 * x + 1 = 0) ↔ (x = 2 + Real.sqrt 3 ∨ x = 2 - Real.sqrt 3) :=
by
  intro x
  sorry

end NUMINAMATH_GPT_solve_equation1_solve_equation2_l965_96561


namespace NUMINAMATH_GPT_budget_spent_on_utilities_l965_96512

noncomputable def budget_is_correct : Prop :=
  let total_budget := 100
  let salaries := 60
  let r_and_d := 9
  let equipment := 4
  let supplies := 2
  let degrees_in_circle := 360
  let transportation_degrees := 72
  let transportation_percentage := (transportation_degrees * total_budget) / degrees_in_circle
  let known_percentages := salaries + r_and_d + equipment + supplies + transportation_percentage
  let utilities_percentage := total_budget - known_percentages
  utilities_percentage = 5

theorem budget_spent_on_utilities : budget_is_correct :=
  sorry

end NUMINAMATH_GPT_budget_spent_on_utilities_l965_96512


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_product_l965_96542

noncomputable def a := 13 / 2
def d := 3 / 2

theorem arithmetic_sequence_sum_product (a d : ℚ) (h1 : 4 * a = 26) (h2 : a^2 - d^2 = 40) :
  (a - 3 * d, a - d, a + d, a + 3 * d) = (2, 5, 8, 11) ∨
  (a - 3 * d, a - d, a + d, a + 3 * d) = (11, 8, 5, 2) :=
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_product_l965_96542


namespace NUMINAMATH_GPT_lemon_pie_degrees_l965_96519

noncomputable def num_students := 45
noncomputable def chocolate_pie_students := 15
noncomputable def apple_pie_students := 9
noncomputable def blueberry_pie_students := 9
noncomputable def other_pie_students := num_students - (chocolate_pie_students + apple_pie_students + blueberry_pie_students)
noncomputable def each_remaining_pie_students := other_pie_students / 3
noncomputable def fraction_lemon_pie := each_remaining_pie_students / num_students
noncomputable def degrees_lemon_pie := fraction_lemon_pie * 360

theorem lemon_pie_degrees : degrees_lemon_pie = 32 :=
sorry

end NUMINAMATH_GPT_lemon_pie_degrees_l965_96519


namespace NUMINAMATH_GPT_break_even_shirts_needed_l965_96530

-- Define the conditions
def initialInvestment : ℕ := 1500
def costPerShirt : ℕ := 3
def sellingPricePerShirt : ℕ := 20

-- Define the profit per T-shirt and the number of T-shirts to break even
def profitPerShirt (sellingPrice costPrice : ℕ) : ℕ := sellingPrice - costPrice

def shirtsToBreakEven (investment profit : ℕ) : ℕ :=
  (investment + profit - 1) / profit -- ceil division

-- The theorem to prove
theorem break_even_shirts_needed :
  shirtsToBreakEven initialInvestment (profitPerShirt sellingPricePerShirt costPerShirt) = 89 :=
by
  -- Calculation
  sorry

end NUMINAMATH_GPT_break_even_shirts_needed_l965_96530


namespace NUMINAMATH_GPT_max_min_values_of_function_l965_96521

theorem max_min_values_of_function :
  (∀ x, 0 ≤ 2 * Real.sin x + 2 ∧ 2 * Real.sin x + 2 ≤ 4) ↔ (∃ x, 2 * Real.sin x + 2 = 0) ∧ (∃ y, 2 * Real.sin y + 2 = 4) :=
by
  sorry

end NUMINAMATH_GPT_max_min_values_of_function_l965_96521


namespace NUMINAMATH_GPT_smallest_x_abs_eq_9_l965_96527

theorem smallest_x_abs_eq_9 : ∃ x : ℝ, |x - 4| = 9 ∧ ∀ y : ℝ, |y - 4| = 9 → x ≤ y :=
by
  -- Prove there exists an x such that |x - 4| = 9 and for all y satisfying |y - 4| = 9, x is the minimum.
  sorry

end NUMINAMATH_GPT_smallest_x_abs_eq_9_l965_96527


namespace NUMINAMATH_GPT_compute_expression_l965_96597

theorem compute_expression : 85 * 1500 + (1 / 2) * 1500 = 128250 :=
by
  sorry

end NUMINAMATH_GPT_compute_expression_l965_96597


namespace NUMINAMATH_GPT_find_x_given_y_l965_96580

theorem find_x_given_y (x y : ℤ) (h1 : 16 * (4 : ℝ)^x = 3^(y + 2)) (h2 : y = -2) : x = -2 := by
  sorry

end NUMINAMATH_GPT_find_x_given_y_l965_96580


namespace NUMINAMATH_GPT_combined_height_of_rockets_l965_96533

noncomputable def height_of_rocket (a t : ℝ) : ℝ := (1/2) * a * t^2

theorem combined_height_of_rockets
  (h_A_ft : ℝ)
  (fuel_type_B_coeff : ℝ)
  (g : ℝ)
  (ft_to_m : ℝ)
  (h_combined : ℝ) :
  h_A_ft = 850 →
  fuel_type_B_coeff = 1.7 →
  g = 9.81 →
  ft_to_m = 0.3048 →
  h_combined = 348.96 :=
by sorry

end NUMINAMATH_GPT_combined_height_of_rockets_l965_96533


namespace NUMINAMATH_GPT_initial_black_pieces_is_118_l965_96505

open Nat

-- Define the initial conditions and variables
variables (b w n : ℕ)

-- Hypotheses based on the conditions
axiom h1 : b = 2 * w
axiom h2 : w - 2 * n = 1
axiom h3 : b - 3 * n = 31

-- Goal to prove the initial number of black pieces were 118
theorem initial_black_pieces_is_118 : b = 118 :=
by 
  -- We only state the theorem, proof will be added as sorry
  sorry

end NUMINAMATH_GPT_initial_black_pieces_is_118_l965_96505


namespace NUMINAMATH_GPT_charity_delivered_100_plates_l965_96557

variables (cost_rice_per_plate cost_chicken_per_plate total_amount_spent : ℝ)
variable (P : ℝ)

-- Conditions provided
def rice_cost : ℝ := 0.10
def chicken_cost : ℝ := 0.40
def total_spent : ℝ := 50
def total_cost_per_plate : ℝ := rice_cost + chicken_cost

-- Lean 4 statement to prove:
theorem charity_delivered_100_plates :
  total_spent = 50 →
  total_cost_per_plate = rice_cost + chicken_cost →
  rice_cost = 0.10 →
  chicken_cost = 0.40 →
  P = total_spent / total_cost_per_plate →
  P = 100 :=
by
  sorry

end NUMINAMATH_GPT_charity_delivered_100_plates_l965_96557


namespace NUMINAMATH_GPT_evaluate_f_l965_96564

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x + 1 else if x = 0 then Real.pi else 0

theorem evaluate_f : f (f (f (-1))) = Real.pi + 1 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_evaluate_f_l965_96564


namespace NUMINAMATH_GPT_surface_area_is_726_l965_96569

def edge_length : ℝ := 11

def surface_area_of_cube (e : ℝ) : ℝ := 6 * (e * e)

theorem surface_area_is_726 (h : edge_length = 11) : surface_area_of_cube edge_length = 726 := by
  sorry

end NUMINAMATH_GPT_surface_area_is_726_l965_96569


namespace NUMINAMATH_GPT_new_train_distance_l965_96599

theorem new_train_distance (old_train_distance : ℕ) (additional_factor : ℕ) (h₀ : old_train_distance = 300) (h₁ : additional_factor = 50) :
  let new_train_distance := old_train_distance + (additional_factor * old_train_distance / 100)
  new_train_distance = 450 :=
by
  sorry

end NUMINAMATH_GPT_new_train_distance_l965_96599


namespace NUMINAMATH_GPT_change_combinations_l965_96539

def isValidCombination (nickels dimes quarters : ℕ) : Prop :=
  nickels * 5 + dimes * 10 + quarters * 25 = 50 ∧ quarters ≤ 1

theorem change_combinations : {n // ∃ (combinations : ℕ) (nickels dimes quarters : ℕ), 
  n = combinations ∧ isValidCombination nickels dimes quarters ∧ 
  ((nickels, dimes, quarters) = (10, 0, 0) ∨
   (nickels, dimes, quarters) = (8, 1, 0) ∨
   (nickels, dimes, quarters) = (6, 2, 0) ∨
   (nickels, dimes, quarters) = (4, 3, 0) ∨
   (nickels, dimes, quarters) = (2, 4, 0) ∨
   (nickels, dimes, quarters) = (0, 5, 0) ∨
   (nickels, dimes, quarters) = (5, 0, 1) ∨
   (nickels, dimes, quarters) = (3, 1, 1) ∨
   (nickels, dimes, quarters) = (1, 2, 1))}
  :=
  ⟨9, sorry⟩

end NUMINAMATH_GPT_change_combinations_l965_96539


namespace NUMINAMATH_GPT_find_foreign_language_score_l965_96513

variable (c m f : ℝ)

theorem find_foreign_language_score
  (h1 : (c + m + f) / 3 = 94)
  (h2 : (c + m) / 2 = 92) :
  f = 98 := by
  sorry

end NUMINAMATH_GPT_find_foreign_language_score_l965_96513


namespace NUMINAMATH_GPT_cisco_spots_difference_l965_96589

theorem cisco_spots_difference :
  ∃ C G R : ℕ, R = 46 ∧ G = 5 * C ∧ G + C = 108 ∧ (23 - C) = 5 :=
by
  sorry

end NUMINAMATH_GPT_cisco_spots_difference_l965_96589


namespace NUMINAMATH_GPT_compute_LM_length_l965_96574

-- Definitions of lengths and equidistant property
variables (GH JK LM : ℝ) 
variables (equidistant : GH * 2 = 120 ∧ JK * 2 = 80)
variables (parallel : GH = 120 ∧ JK = 80 ∧ GH = JK)

-- State the theorem to prove lengths
theorem compute_LM_length (GH JD LM : ℝ) (equidistant : GH * 2 = 120 ∧ JK * 2 = 80)
  (parallel : GH = 120 ∧ JK = 80 ∧ GH = JK) :
  LM = (2 / 3) * 80 := 
sorry

end NUMINAMATH_GPT_compute_LM_length_l965_96574


namespace NUMINAMATH_GPT_average_salary_l965_96529

theorem average_salary (R S T : ℝ) 
  (h1 : (R + S) / 2 = 4000) 
  (h2 : T = 7000) : 
  (R + S + T) / 3 = 5000 :=
by
  sorry

end NUMINAMATH_GPT_average_salary_l965_96529


namespace NUMINAMATH_GPT_vector_arithmetic_l965_96526

-- Define the vectors
def v1 : ℝ × ℝ := (3, -5)
def v2 : ℝ × ℝ := (2, -6)
def v3 : ℝ × ℝ := (-1, 4)

-- Define scalar multiplications
def scalar_mult1 : ℝ × ℝ := (12, -20)  -- 4 * v1
def scalar_mult2 : ℝ × ℝ := (6, -18)   -- 3 * v2

-- Define intermediate vector operations
def intermediate_vector1 : ℝ × ℝ := (6, -2)  -- (12, -20) - (6, -18)

-- Final operation
def final_vector : ℝ × ℝ := (5, 2)  -- (6, -2) + (-1, 4)

-- Prove the main statement
theorem vector_arithmetic : 
  (4 : ℝ) • v1 - (3 : ℝ) • v2 + v3 = final_vector := by
  sorry  -- proof placeholder

end NUMINAMATH_GPT_vector_arithmetic_l965_96526


namespace NUMINAMATH_GPT_problem_I_problem_II_l965_96568

-- Problem I statement
theorem problem_I (a b c : ℝ) (h : a + b + c = 1) : (a + 1)^2 + (b + 1)^2 + (c + 1)^2 ≥ 16 / 3 := 
by
  sorry

-- Problem II statement
theorem problem_II (a : ℝ) : 
  (∀ x : ℝ, abs (x - a) + abs (2 * x - 1) ≥ 2) →
  a ∈ Set.Iic (-3/2) ∪ Set.Ici (5/2) :=
by 
  sorry

end NUMINAMATH_GPT_problem_I_problem_II_l965_96568


namespace NUMINAMATH_GPT_sweatshirt_sales_l965_96509

variables (S H : ℝ)

theorem sweatshirt_sales (h1 : 13 * S + 9 * H = 370) (h2 : 9 * S + 2 * H = 180) :
  12 * S + 6 * H = 300 :=
sorry

end NUMINAMATH_GPT_sweatshirt_sales_l965_96509


namespace NUMINAMATH_GPT_sub_neg_eq_add_pos_l965_96586

theorem sub_neg_eq_add_pos : 0 - (-2) = 2 := 
by
  sorry

end NUMINAMATH_GPT_sub_neg_eq_add_pos_l965_96586


namespace NUMINAMATH_GPT_negation_of_P_l965_96577

open Classical

variable (x : ℝ)

def P (x : ℝ) : Prop :=
  x^2 + 2 > 2 * x

theorem negation_of_P : (¬ ∀ x : ℝ, P x) ↔ ∃ x : ℝ, ¬ P x := by
  sorry

end NUMINAMATH_GPT_negation_of_P_l965_96577


namespace NUMINAMATH_GPT_smallest_integer_b_gt_4_base_b_perfect_square_l965_96560

theorem smallest_integer_b_gt_4_base_b_perfect_square :
  ∃ b : ℕ, b > 4 ∧ ∃ n : ℕ, 2 * b + 5 = n^2 ∧ b = 10 :=
by
  sorry

end NUMINAMATH_GPT_smallest_integer_b_gt_4_base_b_perfect_square_l965_96560


namespace NUMINAMATH_GPT_number_of_students_taking_math_l965_96506

variable (totalPlayers physicsOnly physicsAndMath mathOnly : ℕ)
variable (h1 : totalPlayers = 15) (h2 : physicsOnly = 9) (h3 : physicsAndMath = 3)

theorem number_of_students_taking_math : mathOnly = 9 :=
by {
  sorry
}

end NUMINAMATH_GPT_number_of_students_taking_math_l965_96506


namespace NUMINAMATH_GPT_final_number_correct_l965_96525

noncomputable def initial_number : ℝ := 1256
noncomputable def first_increase_rate : ℝ := 3.25
noncomputable def second_increase_rate : ℝ := 1.47

theorem final_number_correct :
  initial_number * first_increase_rate * second_increase_rate = 6000.54 := 
by
  sorry

end NUMINAMATH_GPT_final_number_correct_l965_96525


namespace NUMINAMATH_GPT_geometric_series_sum_l965_96524

theorem geometric_series_sum :
  let a := 2
  let r := 3
  let n := 6
  S = a * (r ^ n - 1) / (r - 1) → S = 728 :=
by
  intros a r n h
  sorry

end NUMINAMATH_GPT_geometric_series_sum_l965_96524


namespace NUMINAMATH_GPT_michael_passes_donovan_l965_96554

noncomputable def track_length : ℕ := 600
noncomputable def donovan_lap_time : ℕ := 45
noncomputable def michael_lap_time : ℕ := 40

theorem michael_passes_donovan :
  ∃ n : ℕ, michael_lap_time * n > donovan_lap_time * (n - 1) ∧ n = 9 :=
by
  sorry

end NUMINAMATH_GPT_michael_passes_donovan_l965_96554


namespace NUMINAMATH_GPT_bill_cooking_time_l965_96596

def total_time_spent 
  (chop_pepper_time : ℕ) (chop_onion_time : ℕ)
  (grate_cheese_time : ℕ) (cook_omelet_time : ℕ)
  (num_peppers : ℕ) (num_onions : ℕ)
  (num_omelets : ℕ) : ℕ :=
num_peppers * chop_pepper_time + 
num_onions * chop_onion_time + 
num_omelets * grate_cheese_time + 
num_omelets * cook_omelet_time

theorem bill_cooking_time 
  (chop_pepper_time : ℕ) (chop_onion_time : ℕ)
  (grate_cheese_time : ℕ) (cook_omelet_time : ℕ)
  (num_peppers : ℕ) (num_onions : ℕ)
  (num_omelets : ℕ)
  (chop_pepper_time_eq : chop_pepper_time = 3)
  (chop_onion_time_eq : chop_onion_time = 4)
  (grate_cheese_time_eq : grate_cheese_time = 1)
  (cook_omelet_time_eq : cook_omelet_time = 5)
  (num_peppers_eq : num_peppers = 4)
  (num_onions_eq : num_onions = 2)
  (num_omelets_eq : num_omelets = 5) :
  total_time_spent chop_pepper_time chop_onion_time grate_cheese_time cook_omelet_time num_peppers num_onions num_omelets = 50 :=
by {
  sorry
}

end NUMINAMATH_GPT_bill_cooking_time_l965_96596


namespace NUMINAMATH_GPT_ratio_sheila_purity_l965_96595

theorem ratio_sheila_purity (rose_share : ℕ) (total_rent : ℕ) (purity_share : ℕ) (sheila_share : ℕ) 
  (h1 : rose_share = 1800) 
  (h2 : total_rent = 5400) 
  (h3 : rose_share = 3 * purity_share)
  (h4 : total_rent = purity_share + rose_share + sheila_share) : 
  sheila_share / purity_share = 5 :=
by
  -- Proof will be here
  sorry

end NUMINAMATH_GPT_ratio_sheila_purity_l965_96595


namespace NUMINAMATH_GPT_white_balls_count_l965_96532

theorem white_balls_count (a : ℕ) (h : 3 / (3 + a) = 3 / 7) : a = 4 :=
by sorry

end NUMINAMATH_GPT_white_balls_count_l965_96532


namespace NUMINAMATH_GPT_roots_solution_l965_96540

theorem roots_solution (p q : ℝ) (h1 : (∀ x : ℝ, (x - 3) * (3 * x + 8) = x^2 - 5 * x + 6 → (x = p ∨ x = q)))
  (h2 : p + q = 0) (h3 : p * q = -9) : (p + 4) * (q + 4) = 7 :=
by
  sorry

end NUMINAMATH_GPT_roots_solution_l965_96540
