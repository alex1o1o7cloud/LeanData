import Mathlib

namespace NUMINAMATH_GPT_find_chord_eq_l1637_163781

-- Given conditions 
def ellipse_eq (x y : ℝ) : Prop := 4 * x^2 + 9 * y^2 = 144
def point_p : (ℝ × ℝ) := (3, 2)
def midpoint_chord (p1 p2 p : (ℝ × ℝ)) : Prop := p.fst = (p1.fst + p2.fst) / 2 ∧ p.snd = (p1.snd + p2.snd) / 2

-- Conditions in Lean definition
def conditions (x1 y1 x2 y2 : ℝ) : Prop :=
  ellipse_eq x1 y1 ∧ ellipse_eq x2 y2 ∧ midpoint_chord (x1,y1) (x2,y2) point_p

-- The statement to prove
theorem find_chord_eq (x1 y1 x2 y2 : ℝ) (h : conditions x1 y1 x2 y2) :
  ∃ m b : ℝ, (m = -2 / 3) ∧ b = 2 - m * 3 ∧ (∀ x y : ℝ, y = m * x + b → 2 * x + 3 * y - 12 = 0) :=
by {
  sorry
}

end NUMINAMATH_GPT_find_chord_eq_l1637_163781


namespace NUMINAMATH_GPT_calculate_loss_percentage_l1637_163757

theorem calculate_loss_percentage
  (CP SP₁ SP₂ : ℝ)
  (h₁ : SP₁ = CP * 1.05)
  (h₂ : SP₂ = 1140) :
  (CP = 1200) → (SP₁ = 1260) → ((CP - SP₂) / CP * 100 = 5) :=
by
  intros h1 h2
  -- Here, we will eventually provide the actual proof steps.
  sorry

end NUMINAMATH_GPT_calculate_loss_percentage_l1637_163757


namespace NUMINAMATH_GPT_cubic_polynomials_l1637_163728

theorem cubic_polynomials (a b c a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) 
    (h1 : a - 1/b = r₁ ∧ b - 1/c = r₂ ∧ c - 1/a = r₃)
    (h2 : r₁ + r₂ + r₃ = 5)
    (h3 : r₁ * r₂ + r₂ * r₃ + r₃ * r₁ = -15)
    (h4 : r₁ * r₂ * r₃ = -3)
    (h5 : a₁ * b₁ * c₁ = 1 + Real.sqrt 2 ∨ a₁ * b₁ * c₁ = 1 - Real.sqrt 2)
    (h6 : a₂ * b₂ * c₂ = 1 + Real.sqrt 2 ∨ a₂ * b₂ * c₂ = 1 - Real.sqrt 2) :
    (-(a₁ * b₁ * c₁))^3 + (-(a₂ * b₂ * c₂))^3 = -14 := sorry

end NUMINAMATH_GPT_cubic_polynomials_l1637_163728


namespace NUMINAMATH_GPT_positive_integers_no_common_factor_l1637_163735

theorem positive_integers_no_common_factor (X Y Z : ℕ) 
    (X_pos : 0 < X) (Y_pos : 0 < Y) (Z_pos : 0 < Z)
    (coprime_XYZ : Nat.gcd (Nat.gcd X Y) Z = 1)
    (eqn : X * (Real.log 3 / Real.log 100) + Y * (Real.log 4 / Real.log 100) = Z^2) :
    X + Y + Z = 4 :=
sorry

end NUMINAMATH_GPT_positive_integers_no_common_factor_l1637_163735


namespace NUMINAMATH_GPT_lily_milk_amount_l1637_163763

def initial_milk : ℚ := 5
def milk_given_to_james : ℚ := 18 / 4
def milk_received_from_neighbor : ℚ := 7 / 4

theorem lily_milk_amount : (initial_milk - milk_given_to_james + milk_received_from_neighbor) = 9 / 4 :=
by
  sorry

end NUMINAMATH_GPT_lily_milk_amount_l1637_163763


namespace NUMINAMATH_GPT_swimmer_speed_in_still_water_l1637_163754

-- Define the various given conditions as constants in Lean
def swimmer_distance : ℝ := 3
def river_current_speed : ℝ := 1.7
def time_taken : ℝ := 2.3076923076923075

-- Define what we need to prove: the swimmer's speed in still water
theorem swimmer_speed_in_still_water (v : ℝ) :
  swimmer_distance = (v - river_current_speed) * time_taken → 
  v = 3 := by
  sorry

end NUMINAMATH_GPT_swimmer_speed_in_still_water_l1637_163754


namespace NUMINAMATH_GPT_average_books_per_student_l1637_163796

theorem average_books_per_student
  (total_students : ℕ)
  (students_0_books : ℕ)
  (students_1_book : ℕ)
  (students_2_books : ℕ)
  (students_at_least_3_books : ℕ)
  (total_students_eq : total_students = 38)
  (students_0_books_eq : students_0_books = 2)
  (students_1_book_eq : students_1_book = 12)
  (students_2_books_eq : students_2_books = 10)
  (students_at_least_3_books_eq : students_at_least_3_books = 14)
  (students_count_consistent : total_students = students_0_books + students_1_book + students_2_books + students_at_least_3_books) :
  (students_0_books * 0 + students_1_book * 1 + students_2_books * 2 + students_at_least_3_books * 3 : ℝ) / total_students = 1.947 :=
by
  sorry

end NUMINAMATH_GPT_average_books_per_student_l1637_163796


namespace NUMINAMATH_GPT_number_of_10_digit_integers_with_consecutive_twos_l1637_163791

open Nat

-- Define the total number of 10-digit integers using only '1' and '2's
def total_10_digit_numbers : ℕ := 2^10

-- Define the Fibonacci function
def fibonacci : ℕ → ℕ
| 0    => 1
| 1    => 2
| n+2  => fibonacci (n+1) + fibonacci n

-- Calculate the 10th Fibonacci number for the problem context
def F_10 : ℕ := fibonacci 9 + fibonacci 8

-- Prove that the number of 10-digit integers with at least one pair of consecutive '2's is 880
theorem number_of_10_digit_integers_with_consecutive_twos :
  total_10_digit_numbers - F_10 = 880 :=
by
  sorry

end NUMINAMATH_GPT_number_of_10_digit_integers_with_consecutive_twos_l1637_163791


namespace NUMINAMATH_GPT_unique_odd_number_between_500_and_1000_l1637_163760

theorem unique_odd_number_between_500_and_1000 :
  ∃! x : ℤ, 500 ≤ x ∧ x ≤ 1000 ∧ x % 25 = 6 ∧ x % 9 = 7 ∧ x % 2 = 1 :=
sorry

end NUMINAMATH_GPT_unique_odd_number_between_500_and_1000_l1637_163760


namespace NUMINAMATH_GPT_ratio_S15_S5_l1637_163771

variable {a : ℕ → ℝ}  -- The geometric sequence
variable {S : ℕ → ℝ}  -- The sum of the first n terms of the geometric sequence

-- Define the conditions:
axiom sum_of_first_n_terms (n : ℕ) : S n = a 0 * (1 - (a 1)^n) / (1 - a 1)
axiom ratio_S10_S5 : S 10 / S 5 = 1 / 2

-- Define the math proof problem:
theorem ratio_S15_S5 : S 15 / S 5 = 3 / 4 :=
  sorry

end NUMINAMATH_GPT_ratio_S15_S5_l1637_163771


namespace NUMINAMATH_GPT_units_digit_k_cube_plus_2_to_k_plus_1_mod_10_l1637_163707

def k : ℕ := 2012 ^ 2 + 2 ^ 2012

theorem units_digit_k_cube_plus_2_to_k_plus_1_mod_10 : (k ^ 3 + 2 ^ (k + 1)) % 10 = 2 := 
by sorry

end NUMINAMATH_GPT_units_digit_k_cube_plus_2_to_k_plus_1_mod_10_l1637_163707


namespace NUMINAMATH_GPT_find_side_c_and_area_S_find_sinA_plus_cosB_l1637_163711

-- Definitions for the conditions given
structure Triangle :=
  (a b c : ℝ)
  (angleA angleB angleC : ℝ)

noncomputable def givenTriangle : Triangle :=
  { a := 2, b := 4, c := 2 * Real.sqrt 3, angleA := 30, angleB := 90, angleC := 60 }

-- Prove the length of side c and the area S
theorem find_side_c_and_area_S (t : Triangle) (h : t = givenTriangle) :
  t.c = 2 * Real.sqrt 3 ∧ (1 / 2) * t.a * t.b * Real.sin (t.angleC * Real.pi / 180) = 2 * Real.sqrt 3 :=
by
  sorry

-- Prove the value of sin A + cos B
theorem find_sinA_plus_cosB (t : Triangle) (h : t = givenTriangle) :
  Real.sin (t.angleA * Real.pi / 180) + Real.cos (t.angleB * Real.pi / 180) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_side_c_and_area_S_find_sinA_plus_cosB_l1637_163711


namespace NUMINAMATH_GPT_first_three_digits_of_x_are_571_l1637_163798

noncomputable def x : ℝ := (10^2003 + 1)^(11/7)

theorem first_three_digits_of_x_are_571 : 
  ∃ d₁ d₂ d₃ : ℕ, 
  (d₁, d₂, d₃) = (5, 7, 1) ∧ 
  ∃ k : ℤ, 
  (x - k : ℝ) * 1000 = d₁ * 100 + d₂ * 10 + d₃ := 
by
  sorry

end NUMINAMATH_GPT_first_three_digits_of_x_are_571_l1637_163798


namespace NUMINAMATH_GPT_fraction_value_l1637_163756

def x : ℚ := 4 / 7
def y : ℚ := 8 / 11

theorem fraction_value : (7 * x + 11 * y) / (49 * x * y) = 231 / 56 := by
  sorry

end NUMINAMATH_GPT_fraction_value_l1637_163756


namespace NUMINAMATH_GPT_no_real_solution_l1637_163749

noncomputable def augmented_matrix (m : ℝ) : Matrix (Fin 2) (Fin 3) ℝ :=
  ![![m, 4, m+2], ![1, m, m]]

theorem no_real_solution (m : ℝ) :
  (∀ (a b : ℝ), ¬ ∃ (x y : ℝ), a * x + b * y = m ∧ a * x + b * y = 4 ∧ a * x + b * y = m + 2) ↔ m = 2 :=
by
sorry

end NUMINAMATH_GPT_no_real_solution_l1637_163749


namespace NUMINAMATH_GPT_john_draw_on_back_l1637_163772

theorem john_draw_on_back (total_pictures front_pictures : ℕ) (h1 : total_pictures = 15) (h2 : front_pictures = 6) : total_pictures - front_pictures = 9 :=
  by
  sorry

end NUMINAMATH_GPT_john_draw_on_back_l1637_163772


namespace NUMINAMATH_GPT_normals_intersect_at_single_point_l1637_163767

-- Definitions of points on the parabola and distinct condition
variables {a b c : ℝ}

-- Condition stating that A, B, C are distinct points
def distinct_points (a b c : ℝ) : Prop :=
  (a - b) ≠ 0 ∧ (b - c) ≠ 0 ∧ (c - a) ≠ 0

-- Statement to be proved
theorem normals_intersect_at_single_point (habc : distinct_points a b c) :
  a + b + c = 0 :=
sorry

end NUMINAMATH_GPT_normals_intersect_at_single_point_l1637_163767


namespace NUMINAMATH_GPT_B_alone_completion_l1637_163784

-- Define the conditions:
def A_efficiency_rel_to_B (A B: ℕ → Prop) : Prop :=
  ∀ (x: ℕ), B x → A (2 * x)

def together_job_completion (A B: ℕ → Prop) : Prop :=
  ∀ (t: ℕ), t = 20 → (∃ (x y : ℕ), B x ∧ A y ∧ (1/x + 1/y = 1/t))

-- Define the theorem:
theorem B_alone_completion (A B: ℕ → Prop) (h1 : A_efficiency_rel_to_B A B) (h2 : together_job_completion A B) :
  ∃ (x: ℕ), B x ∧ x = 30 :=
sorry

end NUMINAMATH_GPT_B_alone_completion_l1637_163784


namespace NUMINAMATH_GPT_cone_volume_proof_l1637_163727

noncomputable def slant_height := 21
noncomputable def horizontal_semi_axis := 10
noncomputable def vertical_semi_axis := 12
noncomputable def equivalent_radius :=
  Real.sqrt (horizontal_semi_axis * vertical_semi_axis)
noncomputable def cone_height :=
  Real.sqrt (slant_height ^ 2 - equivalent_radius ^ 2)

noncomputable def cone_volume :=
  (1 / 3) * Real.pi * horizontal_semi_axis * vertical_semi_axis * cone_height

theorem cone_volume_proof :
  cone_volume = 2250.24 * Real.pi := sorry

end NUMINAMATH_GPT_cone_volume_proof_l1637_163727


namespace NUMINAMATH_GPT_arithmetic_sqrt_sqrt_16_l1637_163744

theorem arithmetic_sqrt_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_sqrt_sqrt_16_l1637_163744


namespace NUMINAMATH_GPT_num_new_terms_in_sequence_l1637_163718

theorem num_new_terms_in_sequence (k : ℕ) (h : k ≥ 2) : 
  (2^(k+1) - 1) - (2^k - 1) = 2^k := by
  sorry

end NUMINAMATH_GPT_num_new_terms_in_sequence_l1637_163718


namespace NUMINAMATH_GPT_security_to_bag_ratio_l1637_163715

noncomputable def U_house : ℕ := 10
noncomputable def U_airport : ℕ := 5 * U_house
noncomputable def C_bag : ℕ := 15
noncomputable def W_boarding : ℕ := 20
noncomputable def W_takeoff : ℕ := 2 * W_boarding
noncomputable def T_total : ℕ := 180
noncomputable def T_known : ℕ := U_house + U_airport + C_bag + W_boarding + W_takeoff
noncomputable def T_security : ℕ := T_total - T_known

theorem security_to_bag_ratio : T_security / C_bag = 3 :=
by sorry

end NUMINAMATH_GPT_security_to_bag_ratio_l1637_163715


namespace NUMINAMATH_GPT_find_pairs_l1637_163799

theorem find_pairs (a b : ℕ) (h : a + b + a * b = 1000) : 
  (a = 6 ∧ b = 142) ∨ (a = 142 ∧ b = 6) ∨
  (a = 10 ∧ b = 90) ∨ (a = 90 ∧ b = 10) ∨
  (a = 12 ∧ b = 76) ∨ (a = 76 ∧ b = 12) :=
by sorry

end NUMINAMATH_GPT_find_pairs_l1637_163799


namespace NUMINAMATH_GPT_part1_part2_l1637_163764

noncomputable def f (x m : ℝ) : ℝ := x^2 - 2*m*x + 2 - m

theorem part1 (m : ℝ) : (∀ x : ℝ, f x m ≥ x - m*x) → -7 ≤ m ∧ m ≤ 1 :=
by
  sorry

theorem part2 (m : ℝ) : (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → 0 ≤ f x m) → m ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l1637_163764


namespace NUMINAMATH_GPT_provenance_of_positive_test_l1637_163783

noncomputable def pr_disease : ℚ := 1 / 200
noncomputable def pr_no_disease : ℚ := 1 - pr_disease
noncomputable def pr_test_given_disease : ℚ := 1
noncomputable def pr_test_given_no_disease : ℚ := 0.05
noncomputable def pr_test : ℚ := pr_test_given_disease * pr_disease + pr_test_given_no_disease * pr_no_disease
noncomputable def pr_disease_given_test : ℚ := 
  (pr_test_given_disease * pr_disease) / pr_test

theorem provenance_of_positive_test : pr_disease_given_test = 20 / 219 :=
by
  sorry

end NUMINAMATH_GPT_provenance_of_positive_test_l1637_163783


namespace NUMINAMATH_GPT_general_formula_an_geometric_sequence_bn_sum_of_geometric_sequence_Tn_l1637_163743

-- Definitions based on conditions
def a (n : ℕ) : ℕ := 2 * n + 1

def b (n : ℕ) : ℕ := 2 ^ (a n)

noncomputable def S (n : ℕ) : ℕ := (n * (2 * n + 2)) / 2

noncomputable def T (n : ℕ) : ℕ := (8 * (4 ^ n - 1)) / 3

-- Statements to be proved
theorem general_formula_an : ∀ n : ℕ, a n = 2 * n + 1 := sorry

theorem geometric_sequence_bn : ∀ n : ℕ, b n = 2 ^ (2 * n + 1) := sorry

theorem sum_of_geometric_sequence_Tn : ∀ n : ℕ, T n = (8 * (4 ^ n - 1)) / 3 := sorry

end NUMINAMATH_GPT_general_formula_an_geometric_sequence_bn_sum_of_geometric_sequence_Tn_l1637_163743


namespace NUMINAMATH_GPT_find_weight_of_b_l1637_163769

theorem find_weight_of_b (A B C : ℝ)
  (h1 : (A + B + C) / 3 = 45)
  (h2 : (A + B) / 2 = 40)
  (h3 : (B + C) / 2 = 43) : B = 31 :=
sorry

end NUMINAMATH_GPT_find_weight_of_b_l1637_163769


namespace NUMINAMATH_GPT_gcd_a_b_l1637_163725

def a : ℕ := 6666666
def b : ℕ := 999999999

theorem gcd_a_b : Nat.gcd a b = 3 := by
  sorry

end NUMINAMATH_GPT_gcd_a_b_l1637_163725


namespace NUMINAMATH_GPT_number_of_5_dollar_bills_l1637_163706

theorem number_of_5_dollar_bills (x y : ℝ) (h1 : x + y = 54) (h2 : 5 * x + 20 * y = 780) : x = 20 :=
sorry

end NUMINAMATH_GPT_number_of_5_dollar_bills_l1637_163706


namespace NUMINAMATH_GPT_president_and_committee_combination_l1637_163795

theorem president_and_committee_combination : 
  (∃ (n : ℕ), n = 10 * (Nat.choose 9 3)) := 
by
  use 840
  sorry

end NUMINAMATH_GPT_president_and_committee_combination_l1637_163795


namespace NUMINAMATH_GPT_ingrid_income_l1637_163770

theorem ingrid_income (combined_tax_rate : ℝ)
  (john_income : ℝ) (john_tax_rate : ℝ)
  (ingrid_tax_rate : ℝ)
  (combined_income : ℝ)
  (combined_tax : ℝ) :
  combined_tax_rate = 0.35581395348837205 →
  john_income = 57000 →
  john_tax_rate = 0.3 →
  ingrid_tax_rate = 0.4 →
  combined_income = john_income + (combined_income - john_income) →
  combined_tax = (john_tax_rate * john_income) + (ingrid_tax_rate * (combined_income - john_income)) →
  combined_tax_rate = combined_tax / combined_income →
  combined_income = 57000 + 72000 :=
by
  sorry

end NUMINAMATH_GPT_ingrid_income_l1637_163770


namespace NUMINAMATH_GPT_polygon_diagonals_eq_sum_sides_and_right_angles_l1637_163716

-- Define the number of sides of the polygon
variables (n : ℕ)

-- Definition of the number of diagonals in a convex n-sided polygon
def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- Definition of the sum of interior angles of an n-sided polygon
def sum_interior_angles (n : ℕ) : ℕ := (n - 2) * 180

-- Definition of equivalent right angles for interior angles
def num_right_angles (n : ℕ) : ℕ := 2 * (n - 2)

-- The proof statement: prove that the equation holds for n
theorem polygon_diagonals_eq_sum_sides_and_right_angles (h : 3 ≤ n) :
  num_diagonals n = n + num_right_angles n :=
sorry

end NUMINAMATH_GPT_polygon_diagonals_eq_sum_sides_and_right_angles_l1637_163716


namespace NUMINAMATH_GPT_sum_of_three_numbers_l1637_163721

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 156)
  (h2 : a * b + b * c + c * a = 50) :
  a + b + c = 16 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_three_numbers_l1637_163721


namespace NUMINAMATH_GPT_minimum_value_is_six_l1637_163708

noncomputable def minimum_value (m n : ℝ) (h : m > 2 * n) : ℝ :=
  m + (4 * n ^ 2 - 2 * m * n + 9) / (m - 2 * n)

theorem minimum_value_is_six (m n : ℝ) (h : m > 2 * n) : minimum_value m n h = 6 := 
sorry

end NUMINAMATH_GPT_minimum_value_is_six_l1637_163708


namespace NUMINAMATH_GPT_ac_bd_sum_l1637_163704

theorem ac_bd_sum (a b c d : ℝ) (h1 : a + b + c = 6) (h2 : a + b + d = -3) (h3 : a + c + d = 0) (h4 : b + c + d = -9) : 
  a * c + b * d = 23 := 
sorry

end NUMINAMATH_GPT_ac_bd_sum_l1637_163704


namespace NUMINAMATH_GPT_arcsin_arccos_eq_l1637_163773

theorem arcsin_arccos_eq (x : ℝ) (h : Real.arcsin x + Real.arcsin (2 * x - 1) = Real.arccos x) : x = 1 := by
  sorry

end NUMINAMATH_GPT_arcsin_arccos_eq_l1637_163773


namespace NUMINAMATH_GPT_river_current_speed_l1637_163761

theorem river_current_speed 
  (downstream_distance upstream_distance still_water_speed : ℝ)
  (H1 : still_water_speed = 20)
  (H2 : downstream_distance = 100)
  (H3 : upstream_distance = 60)
  (H4 : (downstream_distance / (still_water_speed + x)) = (upstream_distance / (still_water_speed - x)))
  : x = 5 :=
by
  sorry

end NUMINAMATH_GPT_river_current_speed_l1637_163761


namespace NUMINAMATH_GPT_value_of_a_plus_b_l1637_163758

theorem value_of_a_plus_b (a b : ℝ) (h : (2 * a + 2 * b - 1) * (2 * a + 2 * b + 1) = 99) :
  a + b = 5 ∨ a + b = -5 :=
sorry

end NUMINAMATH_GPT_value_of_a_plus_b_l1637_163758


namespace NUMINAMATH_GPT_monotonic_intervals_a_eq_1_range_of_a_no_zero_points_in_interval_l1637_163785

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (2 - a) * (x - 1) - 2 * Real.log x

theorem monotonic_intervals_a_eq_1 :
  ∀ x : ℝ, (0 < x ∧ x ≤ 2 → (f x 1) < (f 2 1)) ∧ 
           (2 ≤ x → (f x 1) > (f 2 1)) :=
by
  sorry

theorem range_of_a_no_zero_points_in_interval :
  ∀ a : ℝ, (∀ x : ℝ, (0 < x ∧ x < 1/3) → ((2 - a) * (x - 1) - 2 * Real.log x) > 0) ↔ 2 - 3 * Real.log 3 ≤ a :=
by
  sorry

end NUMINAMATH_GPT_monotonic_intervals_a_eq_1_range_of_a_no_zero_points_in_interval_l1637_163785


namespace NUMINAMATH_GPT_perfect_square_trinomial_iff_l1637_163710

theorem perfect_square_trinomial_iff (m : ℤ) :
  (∃ a b : ℤ, 4 = a^2 ∧ 121 = b^2 ∧ (4 = a^2 ∧ 121 = b^2) ∧ m = 2 * a * b ∨ m = -2 * a * b) ↔ (m = 44 ∨ m = -44) :=
by sorry

end NUMINAMATH_GPT_perfect_square_trinomial_iff_l1637_163710


namespace NUMINAMATH_GPT_area_enclosed_by_abs_eq_l1637_163750

theorem area_enclosed_by_abs_eq (x y : ℝ) : 
  (|x| + |3 * y| = 12) → (∃ area : ℝ, area = 96) :=
by
  sorry

end NUMINAMATH_GPT_area_enclosed_by_abs_eq_l1637_163750


namespace NUMINAMATH_GPT_inequality_must_hold_l1637_163747

theorem inequality_must_hold (a b c : ℝ) (h : (a / c^2) > (b / c^2)) (hc : c ≠ 0) : a^2 > b^2 :=
sorry

end NUMINAMATH_GPT_inequality_must_hold_l1637_163747


namespace NUMINAMATH_GPT_condition_neither_sufficient_nor_necessary_l1637_163740
-- Import necessary library

-- Define the function and conditions
def f (x a : ℝ) : ℝ := x^2 + a * x + 1

-- State the proof problem
theorem condition_neither_sufficient_nor_necessary :
  ∀ a : ℝ, (∀ x : ℝ, f x a = 0 -> x = 1/2) ↔ a^2 - 4 = 0 ∧ a ≤ -2 := sorry

end NUMINAMATH_GPT_condition_neither_sufficient_nor_necessary_l1637_163740


namespace NUMINAMATH_GPT_point_in_second_quadrant_range_l1637_163739

theorem point_in_second_quadrant_range (m : ℝ) :
  (m - 3 < 0 ∧ m + 1 > 0) ↔ (-1 < m ∧ m < 3) :=
by
  sorry

end NUMINAMATH_GPT_point_in_second_quadrant_range_l1637_163739


namespace NUMINAMATH_GPT_ratio_square_l1637_163713

theorem ratio_square (x y : ℕ) (h1 : x * (x + y) = 40) (h2 : y * (x + y) = 90) (h3 : 2 * y = 3 * x) : (x + y) ^ 2 = 100 := 
by 
  sorry

end NUMINAMATH_GPT_ratio_square_l1637_163713


namespace NUMINAMATH_GPT_jamies_mother_twice_age_l1637_163768

theorem jamies_mother_twice_age (y : ℕ) :
  ∀ (jamie_age_2010 mother_age_2010 : ℕ), 
  jamie_age_2010 = 10 → 
  mother_age_2010 = 5 * jamie_age_2010 → 
  mother_age_2010 + y = 2 * (jamie_age_2010 + y) → 
  2010 + y = 2040 :=
by
  intros jamie_age_2010 mother_age_2010 h_jamie h_mother h_eq
  sorry

end NUMINAMATH_GPT_jamies_mother_twice_age_l1637_163768


namespace NUMINAMATH_GPT_plumber_fix_cost_toilet_l1637_163738

noncomputable def fixCost_Sink : ℕ := 30
noncomputable def fixCost_Shower : ℕ := 40

theorem plumber_fix_cost_toilet
  (T : ℕ)
  (Earnings1 : ℕ := 3 * T + 3 * fixCost_Sink)
  (Earnings2 : ℕ := 2 * T + 5 * fixCost_Sink)
  (Earnings3 : ℕ := T + 2 * fixCost_Shower + 3 * fixCost_Sink)
  (MaxEarnings : ℕ := 250) :
  Earnings2 = MaxEarnings → T = 50 :=
by
  sorry

end NUMINAMATH_GPT_plumber_fix_cost_toilet_l1637_163738


namespace NUMINAMATH_GPT_range_of_a_l1637_163703

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 < x → a < x + (1 / x)) → a < 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1637_163703


namespace NUMINAMATH_GPT_no_real_ordered_triples_l1637_163794

theorem no_real_ordered_triples (x y z : ℝ) (h1 : x + y = 3) (h2 : xy - z^2 = 4) : false :=
sorry

end NUMINAMATH_GPT_no_real_ordered_triples_l1637_163794


namespace NUMINAMATH_GPT_percentage_increase_formula_l1637_163736

theorem percentage_increase_formula (A B C : ℝ) (h1 : A = 3 * B) (h2 : C = B - 30) :
  100 * ((A - C) / C) = 200 + 9000 / C := 
by 
  sorry

end NUMINAMATH_GPT_percentage_increase_formula_l1637_163736


namespace NUMINAMATH_GPT_range_of_m_l1637_163789

theorem range_of_m (m : ℝ) :
  let M := {x : ℝ | x ≤ m}
  let P := {x : ℝ | x ≥ -1}
  (M ∩ P = ∅) → m < -1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1637_163789


namespace NUMINAMATH_GPT_evaluate_powers_of_i_l1637_163712

-- Define complex number "i"
def i := Complex.I

-- Define the theorem to prove
theorem evaluate_powers_of_i : i^44 + i^444 + 3 = 5 := by
  -- use the cyclic property of i to simplify expressions
  sorry

end NUMINAMATH_GPT_evaluate_powers_of_i_l1637_163712


namespace NUMINAMATH_GPT_geometric_seq_xyz_eq_neg_two_l1637_163762

open Real

noncomputable def geometric_seq (a b c d e : ℝ) : Prop :=
  ∃ r : ℝ, b = a * r ∧ c = b * r ∧ d = c * r ∧ e = d * r

theorem geometric_seq_xyz_eq_neg_two (x y z : ℝ) :
  geometric_seq (-1) x y z (-2) → x * y * z = -2 :=
by
  intro h
  obtain ⟨r, hx, hy, hz, he⟩ := h
  rw [hx, hy, hz, he] at *
  sorry

end NUMINAMATH_GPT_geometric_seq_xyz_eq_neg_two_l1637_163762


namespace NUMINAMATH_GPT_continuous_implies_defined_defined_does_not_imply_continuous_l1637_163726

-- Define function continuity at a point x = a
def continuous_at (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - a) < δ → abs (f x - f a) < ε

-- Prove that if f is continuous at x = a, then f is defined at x = a
theorem continuous_implies_defined (f : ℝ → ℝ) (a : ℝ) : 
  continuous_at f a → ∃ y, f a = y :=
by
  sorry  -- Proof omitted

-- Prove that the definition of f at x = a does not guarantee continuity at x = a
theorem defined_does_not_imply_continuous (f : ℝ → ℝ) (a : ℝ) :
  (∃ y, f a = y) → ¬ continuous_at f a :=
by
  sorry  -- Proof omitted

end NUMINAMATH_GPT_continuous_implies_defined_defined_does_not_imply_continuous_l1637_163726


namespace NUMINAMATH_GPT_uncle_zhang_age_l1637_163737

theorem uncle_zhang_age (z l : ℕ) (h1 : z + l = 56) (h2 : z = l - (l / 2)) : z = 24 :=
by sorry

end NUMINAMATH_GPT_uncle_zhang_age_l1637_163737


namespace NUMINAMATH_GPT_FruitKeptForNextWeek_l1637_163746

/-- Define the variables and conditions -/
def total_fruit : ℕ := 10
def fruit_eaten : ℕ := 5
def fruit_brought_on_friday : ℕ := 3

/-- Define what we need to prove -/
theorem FruitKeptForNextWeek : 
  ∃ k, total_fruit - fruit_eaten - fruit_brought_on_friday = k ∧ k = 2 :=
by
  sorry

end NUMINAMATH_GPT_FruitKeptForNextWeek_l1637_163746


namespace NUMINAMATH_GPT_probability_white_ball_initial_probability_yellow_or_red_ball_initial_probability_white_ball_after_removal_l1637_163705

def total_balls := 20
def red_balls := 10
def yellow_balls := 6
def white_balls := 4
def initial_white_balls_probability := (white_balls : ℚ) / total_balls
def initial_yellow_or_red_balls_probability := (yellow_balls + red_balls : ℚ) / total_balls

def removed_red_balls := 2
def removed_white_balls := 2
def remaining_balls := total_balls - (removed_red_balls + removed_white_balls)
def remaining_white_balls := white_balls - removed_white_balls
def remaining_white_balls_probability := (remaining_white_balls : ℚ) / remaining_balls

theorem probability_white_ball_initial : initial_white_balls_probability = 1 / 5 := by sorry
theorem probability_yellow_or_red_ball_initial : initial_yellow_or_red_balls_probability = 4 / 5 := by sorry
theorem probability_white_ball_after_removal : remaining_white_balls_probability = 1 / 8 := by sorry

end NUMINAMATH_GPT_probability_white_ball_initial_probability_yellow_or_red_ball_initial_probability_white_ball_after_removal_l1637_163705


namespace NUMINAMATH_GPT_computer_cost_l1637_163779

theorem computer_cost (C : ℝ) (h1 : 0.10 * C = a) (h2 : 3 * C = b) (h3 : b - 1.10 * C = 2700) : 
  C = 2700 / 2.90 :=
by
  sorry

end NUMINAMATH_GPT_computer_cost_l1637_163779


namespace NUMINAMATH_GPT_value_of_y_l1637_163717

-- Problem: Prove that given the conditions \( x - y = 8 \) and \( x + y = 16 \),
-- the value of \( y \) is 4.
theorem value_of_y (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 16) : y = 4 := 
sorry

end NUMINAMATH_GPT_value_of_y_l1637_163717


namespace NUMINAMATH_GPT_Diana_additional_video_game_time_l1637_163720

theorem Diana_additional_video_game_time 
    (original_reward_per_hour : ℕ := 30)
    (raise_percentage : ℕ := 20)
    (hours_read : ℕ := 12)
    (minutes_per_hour : ℕ := 60) :
    let raise := (raise_percentage * original_reward_per_hour) / 100
    let new_reward_per_hour := original_reward_per_hour + raise
    let total_time_after_raise := new_reward_per_hour * hours_read
    let total_time_before_raise := original_reward_per_hour * hours_read
    let additional_minutes := total_time_after_raise - total_time_before_raise
    additional_minutes = 72 :=
by sorry

end NUMINAMATH_GPT_Diana_additional_video_game_time_l1637_163720


namespace NUMINAMATH_GPT_sum_proof_l1637_163709

theorem sum_proof (X Y : ℝ) (hX : 0.45 * X = 270) (hY : 0.35 * Y = 210) : 
  (0.75 * X) + (0.55 * Y) = 780 := by
  sorry

end NUMINAMATH_GPT_sum_proof_l1637_163709


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l1637_163702

theorem simplify_and_evaluate_expression (a b : ℝ) (h : (a + 1)^2 + |b + 1| = 0) : 
  1 - (a^2 + 2 * a * b + b^2) / (a^2 - a * b) / ((a + b) / (a - b)) = -1 := 
sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l1637_163702


namespace NUMINAMATH_GPT_find_g_eq_minus_x_l1637_163787

-- Define the function g and the given conditions.
def g (x : ℝ) : ℝ := sorry

axiom g0 : g 0 = 2
axiom g_xy : ∀ (x y : ℝ), g (x * y) = g ((x^2 + 2 * y^2) / 3) + 3 * (x - y)^2

-- State the problem: proving that g(x) = -x.
theorem find_g_eq_minus_x : ∀ (x : ℝ), g x = -x := by
  sorry

end NUMINAMATH_GPT_find_g_eq_minus_x_l1637_163787


namespace NUMINAMATH_GPT_reciprocal_of_fraction_sum_l1637_163723

theorem reciprocal_of_fraction_sum : 
  (1 / (1 / 3 + 1 / 4 - 1 / 12)) = 2 := sorry

end NUMINAMATH_GPT_reciprocal_of_fraction_sum_l1637_163723


namespace NUMINAMATH_GPT_tournament_ranking_sequences_l1637_163734

def total_fair_ranking_sequences (A B C D : Type) : Nat :=
  let saturday_outcomes := 2
  let sunday_outcomes := 4 -- 2 possibilities for (first, second) and 2 for (third, fourth)
  let tiebreaker_effect := 2 -- swap second and third
  saturday_outcomes * sunday_outcomes * tiebreaker_effect

theorem tournament_ranking_sequences (A B C D : Type) :
  total_fair_ranking_sequences A B C D = 32 := 
by
  sorry

end NUMINAMATH_GPT_tournament_ranking_sequences_l1637_163734


namespace NUMINAMATH_GPT_percentage_per_annum_is_correct_l1637_163792

-- Define the conditions of the problem
def banker_gain : ℝ := 24
def present_worth : ℝ := 600
def time : ℕ := 2

-- Define the formula for the amount due
def amount_due (r : ℝ) (t : ℕ) (PW : ℝ) : ℝ := PW * (1 + r * t)

-- Define the given conditions translated from the problem
def given_conditions (r : ℝ) : Prop :=
  amount_due r time present_worth = present_worth + banker_gain

-- Lean statement of the problem to be proved
theorem percentage_per_annum_is_correct :
  ∃ r : ℝ, given_conditions r ∧ r = 0.02 :=
by {
  sorry
}

end NUMINAMATH_GPT_percentage_per_annum_is_correct_l1637_163792


namespace NUMINAMATH_GPT_range_of_a_l1637_163755

noncomputable def f (x a : ℝ) : ℝ := x^2 * Real.exp x - a

theorem range_of_a 
 (h : ∃ a, (∀ x₀ x₁ x₂, x₀ ≠ x₁ ∧ x₀ ≠ x₂ ∧ x₁ ≠ x₂ ∧ f x₀ a = 0 ∧ f x₁ a = 0 ∧ f x₂ a = 0)) :
  ∃ a, 0 < a ∧ a < 4 / Real.exp 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1637_163755


namespace NUMINAMATH_GPT_right_triangle_wy_expression_l1637_163748

theorem right_triangle_wy_expression (α β : ℝ) (u v w y : ℝ)
    (h1 : (∀ x : ℝ, x^2 - u * x + v = 0 → x = Real.sin α ∨ x = Real.sin β))
    (h2 : (∀ x : ℝ, x^2 - w * x + y = 0 → x = Real.cos α ∨ x = Real.cos β))
    (h3 : α + β = Real.pi / 2) :
    w * y = u * v :=
sorry

end NUMINAMATH_GPT_right_triangle_wy_expression_l1637_163748


namespace NUMINAMATH_GPT_max_possible_value_of_C_l1637_163765

theorem max_possible_value_of_C (A B C D : ℕ) (h₁ : A + B + C + D = 200) (h₂ : A + B = 70) (h₃ : 0 < A) (h₄ : 0 < B) (h₅ : 0 < C) (h₆ : 0 < D) :
  C ≤ 129 :=
by
  sorry

end NUMINAMATH_GPT_max_possible_value_of_C_l1637_163765


namespace NUMINAMATH_GPT_product_polynomials_l1637_163776

theorem product_polynomials (x : ℝ) : 
  (1 + x^3) * (1 - 2 * x + x^4) = 1 - 2 * x + x^3 - x^4 + x^7 :=
by sorry

end NUMINAMATH_GPT_product_polynomials_l1637_163776


namespace NUMINAMATH_GPT_function_passes_through_point_l1637_163793

theorem function_passes_through_point (a : ℝ) (x y : ℝ) (h_pos : a > 0) (h_ne_one : a ≠ 1) :
  (x = 1 ∧ y = 4) ↔ (y = a^(x-1) + 3) :=
sorry

end NUMINAMATH_GPT_function_passes_through_point_l1637_163793


namespace NUMINAMATH_GPT_multiply_eq_four_l1637_163778

variables (a b c d : ℝ)

theorem multiply_eq_four (h1 : a = d) 
                         (h2 : b = c) 
                         (h3 : d + d = c * d) 
                         (h4 : b = d) 
                         (h5 : d + d = d * d) 
                         (h6 : c = 3) :
                         a * b = 4 := 
by 
  sorry

end NUMINAMATH_GPT_multiply_eq_four_l1637_163778


namespace NUMINAMATH_GPT_solve_for_x_l1637_163775

theorem solve_for_x (x : ℝ) (h : (1 / 2) * (1 / 7) * x = 14) : x = 196 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1637_163775


namespace NUMINAMATH_GPT_product_of_four_integers_l1637_163751

theorem product_of_four_integers:
  ∃ (A B C D : ℚ) (x : ℚ), 0 < A ∧ 0 < B ∧ 0 < C ∧ 0 < D ∧
  A + B + C + D = 40 ∧
  A - 3 = x ∧ B + 3 = x ∧ C / 2 = x ∧ D * 2 = x ∧
  A * B * C * D = (9089600 / 6561) := by
  sorry

end NUMINAMATH_GPT_product_of_four_integers_l1637_163751


namespace NUMINAMATH_GPT_function_passes_through_vertex_l1637_163782

theorem function_passes_through_vertex (a : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) : a^(2 - 2) + 1 = 2 :=
by
  sorry

end NUMINAMATH_GPT_function_passes_through_vertex_l1637_163782


namespace NUMINAMATH_GPT_good_permutation_exists_iff_power_of_two_l1637_163700

def is_good_permutation (n : ℕ) (a : ℕ → ℕ) : Prop :=
  ∀ i j k : ℕ, i < j → j < k → k < n → ¬ (↑n ∣ (a i + a k - 2 * a j))

theorem good_permutation_exists_iff_power_of_two (n : ℕ) (h : n ≥ 3) :
  (∃ a : ℕ → ℕ, (∀ i, i < n → a i < n) ∧ is_good_permutation n a) ↔ ∃ b : ℕ, 2 ^ b = n :=
sorry

end NUMINAMATH_GPT_good_permutation_exists_iff_power_of_two_l1637_163700


namespace NUMINAMATH_GPT_gcd_lcm_45_75_l1637_163742

theorem gcd_lcm_45_75 : gcd 45 75 = 15 ∧ lcm 45 75 = 1125 :=
by sorry

end NUMINAMATH_GPT_gcd_lcm_45_75_l1637_163742


namespace NUMINAMATH_GPT_g_difference_l1637_163759

def g (n : ℕ) : ℚ := (1 / 4) * n * (n + 3) * (n + 5) + 2

theorem g_difference (s : ℕ) : g s - g (s - 1) = (3 * s^2 + 9 * s + 8) / 4 :=
by
  -- skip the proof
  sorry

end NUMINAMATH_GPT_g_difference_l1637_163759


namespace NUMINAMATH_GPT_series_sum_l1637_163752

theorem series_sum :
  ∑' n : ℕ, (2 ^ n) / (3 ^ (2 ^ n) + 1) = 1 / 2 :=
sorry

end NUMINAMATH_GPT_series_sum_l1637_163752


namespace NUMINAMATH_GPT_area_of_rectangular_field_l1637_163777

theorem area_of_rectangular_field 
  (P L W : ℕ) 
  (hP : P = 120) 
  (hL : L = 3 * W) 
  (hPerimeter : 2 * L + 2 * W = P) : 
  (L * W = 675) :=
by 
  sorry

end NUMINAMATH_GPT_area_of_rectangular_field_l1637_163777


namespace NUMINAMATH_GPT_system_of_equations_l1637_163741

theorem system_of_equations (x y : ℝ) 
  (h1 : 2 * x + y = 5) 
  (h2 : x + 2 * y = 4) : 
  x + y = 3 :=
sorry

end NUMINAMATH_GPT_system_of_equations_l1637_163741


namespace NUMINAMATH_GPT_total_pages_read_is_785_l1637_163797

-- Definitions based on the conditions in the problem
def pages_read_first_five_days : ℕ := 5 * 52
def pages_read_next_five_days : ℕ := 5 * 63
def pages_read_last_three_days : ℕ := 3 * 70

-- The main statement to prove
theorem total_pages_read_is_785 :
  pages_read_first_five_days + pages_read_next_five_days + pages_read_last_three_days = 785 :=
by
  sorry

end NUMINAMATH_GPT_total_pages_read_is_785_l1637_163797


namespace NUMINAMATH_GPT_trigonometric_identity_l1637_163719

theorem trigonometric_identity :
  (Real.sin (18 * Real.pi / 180) * Real.sin (78 * Real.pi / 180)) -
  (Real.cos (162 * Real.pi / 180) * Real.cos (78 * Real.pi / 180)) = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l1637_163719


namespace NUMINAMATH_GPT_no_integer_points_between_A_and_B_on_line_l1637_163729

theorem no_integer_points_between_A_and_B_on_line
  (A : ℕ × ℕ) (B : ℕ × ℕ)
  (hA : A = (2, 3))
  (hB : B = (50, 500)) :
  ∀ (P : ℕ × ℕ), P.1 > 2 ∧ P.1 < 50 ∧ 
    (P.2 * 48 - P.1 * 497 = 2 * 497 - 3 * 48) →
    false := 
by
  sorry

end NUMINAMATH_GPT_no_integer_points_between_A_and_B_on_line_l1637_163729


namespace NUMINAMATH_GPT_decreasing_interval_l1637_163790

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (x - Real.pi / 4)

theorem decreasing_interval :
  ∀ x ∈ Set.Icc 0 (2 * Real.pi), x ∈ Set.Icc (3 * Real.pi / 4) (2 * Real.pi) ↔ (∀ ε > 0, f x > f (x + ε)) := 
sorry

end NUMINAMATH_GPT_decreasing_interval_l1637_163790


namespace NUMINAMATH_GPT_total_money_is_102_l1637_163786

-- Defining the amounts of money each person has
def Jack_money : ℕ := 26
def Ben_money : ℕ := Jack_money - 9
def Eric_money : ℕ := Ben_money - 10
def Anna_money : ℕ := Jack_money * 2

-- Defining the total amount of money
def total_money : ℕ := Eric_money + Ben_money + Jack_money + Anna_money

-- Proving the total money is 102
theorem total_money_is_102 : total_money = 102 :=
by
  -- this is where the proof would go
  sorry

end NUMINAMATH_GPT_total_money_is_102_l1637_163786


namespace NUMINAMATH_GPT_dilution_problem_l1637_163788
-- Definitions of the conditions
def volume_initial : ℝ := 15
def concentration_initial : ℝ := 0.60
def concentration_final : ℝ := 0.40
def amount_alcohol_initial : ℝ := volume_initial * concentration_initial

-- Proof problem statement in Lean 4
theorem dilution_problem : 
  ∃ (x : ℝ), x = 7.5 ∧ 
              amount_alcohol_initial = concentration_final * (volume_initial + x) :=
sorry

end NUMINAMATH_GPT_dilution_problem_l1637_163788


namespace NUMINAMATH_GPT_area_of_AFCH_l1637_163780

-- Define the sides of the rectangles ABCD and EFGH
def AB : ℝ := 9
def BC : ℝ := 5
def EF : ℝ := 3
def FG : ℝ := 10

-- Define the area of quadrilateral AFCH
def area_AFCH : ℝ := 52.5

-- The theorem we want to prove
theorem area_of_AFCH :
  AB = 9 ∧ BC = 5 ∧ EF = 3 ∧ FG = 10 → (area_AFCH = 52.5) :=
by
  sorry

end NUMINAMATH_GPT_area_of_AFCH_l1637_163780


namespace NUMINAMATH_GPT_find_number_of_students_l1637_163724

variables (n : ℕ)
variables (avg_A avg_B avg_C excl_avg_A excl_avg_B excl_avg_C : ℕ)
variables (new_avg_A new_avg_B new_avg_C : ℕ)
variables (excluded_students : ℕ)

theorem find_number_of_students :
  avg_A = 80 ∧ avg_B = 85 ∧ avg_C = 75 ∧
  excl_avg_A = 20 ∧ excl_avg_B = 25 ∧ excl_avg_C = 15 ∧
  excluded_students = 5 ∧
  new_avg_A = 90 ∧ new_avg_B = 95 ∧ new_avg_C = 85 →
  n = 35 :=
by
  sorry

end NUMINAMATH_GPT_find_number_of_students_l1637_163724


namespace NUMINAMATH_GPT_sum_of_first_10_bn_l1637_163714

def a (n : ℕ) : ℚ :=
  (2 / 5) * n + (3 / 5)

def b (n : ℕ) : ℤ :=
  ⌊a n⌋

def sum_first_10_b : ℤ :=
  (b 1) + (b 2) + (b 3) + (b 4) + (b 5) + (b 6) + (b 7) + (b 8) + (b 9) + (b 10)

theorem sum_of_first_10_bn : sum_first_10_b = 24 :=
  by sorry

end NUMINAMATH_GPT_sum_of_first_10_bn_l1637_163714


namespace NUMINAMATH_GPT_steve_more_than_wayne_first_time_at_2004_l1637_163733

def initial_steve_money (year: ℕ) := if year = 2000 then 100 else 0
def initial_wayne_money (year: ℕ) := if year = 2000 then 10000 else 0

def steve_money (year: ℕ) : ℕ :=
  if year < 2000 then 0
  else if year = 2000 then initial_steve_money year
  else 2 * steve_money (year - 1)

def wayne_money (year: ℕ) : ℕ :=
  if year < 2000 then 0
  else if year = 2000 then initial_wayne_money year
  else wayne_money (year - 1) / 2

theorem steve_more_than_wayne_first_time_at_2004 :
  ∃ (year: ℕ), year = 2004 ∧ steve_money year > wayne_money year := by
  sorry

end NUMINAMATH_GPT_steve_more_than_wayne_first_time_at_2004_l1637_163733


namespace NUMINAMATH_GPT_only_nonneg_int_solution_l1637_163701

theorem only_nonneg_int_solution (x y z : ℕ) (h : x^3 = 3 * y^3 + 9 * z^3) : x = 0 ∧ y = 0 ∧ z = 0 := 
sorry

end NUMINAMATH_GPT_only_nonneg_int_solution_l1637_163701


namespace NUMINAMATH_GPT_true_propositions_l1637_163722

noncomputable def discriminant_leq_zero : Prop :=
  let a := 1
  let b := -1
  let c := 2
  b^2 - 4 * a * c ≤ 0

def proposition_1 : Prop := discriminant_leq_zero

def proposition_2 (x : ℝ) : Prop :=
  abs x ≥ 0 → x ≥ 0

def proposition_3 : Prop :=
  5 > 2 ∧ 3 < 7

theorem true_propositions : proposition_1 ∧ proposition_3 ∧ ¬∀ x : ℝ, proposition_2 x :=
by
  sorry

end NUMINAMATH_GPT_true_propositions_l1637_163722


namespace NUMINAMATH_GPT_sum_of_possible_values_l1637_163753

theorem sum_of_possible_values (x y : ℝ) (h : x * y - (2 * x) / (y ^ 3) - (2 * y) / (x ^ 3) = 6) :
  ∃ (a b : ℝ), (a - 2) * (b - 2) = 4 ∧ (a - 2) * (b - 2) = 9 ∧ 4 + 9 = 13 :=
sorry

end NUMINAMATH_GPT_sum_of_possible_values_l1637_163753


namespace NUMINAMATH_GPT_area_of_trapezium_l1637_163730

-- Definitions for the given conditions
def parallel_side_a : ℝ := 18  -- in cm
def parallel_side_b : ℝ := 20  -- in cm
def distance_between_sides : ℝ := 5  -- in cm

-- Statement to prove the area is 95 cm²
theorem area_of_trapezium : 
  let a := parallel_side_a
  let b := parallel_side_b
  let h := distance_between_sides
  (1 / 2 * (a + b) * h = 95) :=
by
  sorry  -- Proof is not required here

end NUMINAMATH_GPT_area_of_trapezium_l1637_163730


namespace NUMINAMATH_GPT_carter_siblings_oldest_age_l1637_163732

theorem carter_siblings_oldest_age
    (avg_age : ℕ)
    (sibling1 : ℕ)
    (sibling2 : ℕ)
    (sibling3 : ℕ)
    (sibling4 : ℕ) :
    avg_age = 9 →
    sibling1 = 5 →
    sibling2 = 8 →
    sibling3 = 7 →
    ((sibling1 + sibling2 + sibling3 + sibling4) / 4) = avg_age →
    sibling4 = 16 := by
  intros
  sorry

end NUMINAMATH_GPT_carter_siblings_oldest_age_l1637_163732


namespace NUMINAMATH_GPT_Bill_has_39_dollars_l1637_163731

noncomputable def Frank_initial_money : ℕ := 42
noncomputable def pizza_cost : ℕ := 11
noncomputable def num_pizzas : ℕ := 3
noncomputable def Bill_initial_money : ℕ := 30

noncomputable def Frank_spent : ℕ := pizza_cost * num_pizzas
noncomputable def Frank_remaining_money : ℕ := Frank_initial_money - Frank_spent
noncomputable def Bill_final_money : ℕ := Bill_initial_money + Frank_remaining_money

theorem Bill_has_39_dollars :
  Bill_final_money = 39 :=
by
  sorry

end NUMINAMATH_GPT_Bill_has_39_dollars_l1637_163731


namespace NUMINAMATH_GPT_Problem_l1637_163766

def f (x : ℕ) : ℕ := x ^ 2 + 1
def g (x : ℕ) : ℕ := 2 * x - 1

theorem Problem : f (g (3 + 1)) = 50 := by
  sorry

end NUMINAMATH_GPT_Problem_l1637_163766


namespace NUMINAMATH_GPT_problem1_problem2_l1637_163745

noncomputable def f (x a b : ℝ) := |x + a^2| + |x - b^2|

theorem problem1 (a b x : ℝ) (h : a^2 + b^2 - 2 * a + 2 * b + 2 = 0) :
  f x a b >= 3 ↔ x <= -0.5 ∨ x >= 1.5 :=
sorry

theorem problem2 (a b x : ℝ) (h : a + b = 4) :
  f x a b >= 8 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l1637_163745


namespace NUMINAMATH_GPT_gcd_143_144_l1637_163774

def a : ℕ := 143
def b : ℕ := 144

theorem gcd_143_144 : Nat.gcd a b = 1 :=
by
  sorry

end NUMINAMATH_GPT_gcd_143_144_l1637_163774
