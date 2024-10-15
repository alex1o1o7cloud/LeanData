import Mathlib

namespace NUMINAMATH_GPT_no_valid_positive_x_l155_15563

theorem no_valid_positive_x
  (π : Real)
  (R H x : Real)
  (hR : R = 5)
  (hH : H = 10)
  (hx_pos : x > 0) :
  ¬π * (R + x) ^ 2 * H = π * R ^ 2 * (H + x) :=
by
  sorry

end NUMINAMATH_GPT_no_valid_positive_x_l155_15563


namespace NUMINAMATH_GPT_taxes_are_135_l155_15571

def gross_pay : ℕ := 450
def net_pay : ℕ := 315
def taxes_paid (G N: ℕ) : ℕ := G - N

theorem taxes_are_135 : taxes_paid gross_pay net_pay = 135 := by
  sorry

end NUMINAMATH_GPT_taxes_are_135_l155_15571


namespace NUMINAMATH_GPT_range_of_k_l155_15552

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 2 then 2 / x else (x - 1)^3

theorem range_of_k (k : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 - k = 0 ∧ f x2 - k = 0) ↔ (0 < k ∧ k < 1) := sorry

end NUMINAMATH_GPT_range_of_k_l155_15552


namespace NUMINAMATH_GPT_remainder_when_divided_by_8_l155_15595

theorem remainder_when_divided_by_8 (k : ℤ) : ((63 * k + 25) % 8) = 1 := 
by sorry

end NUMINAMATH_GPT_remainder_when_divided_by_8_l155_15595


namespace NUMINAMATH_GPT_reduced_price_l155_15575

open Real

noncomputable def original_price : ℝ := 33.33

variables (P R: ℝ) (Q : ℝ)

theorem reduced_price
  (h1 : R = 0.75 * P)
  (h2 : P * 500 / P = 500)
  (h3 : 0.75 * P * (Q + 5) = 500)
  (h4 : Q = 500 / P) :
  R = 25 :=
by
  -- The proof will be provided here
  sorry

end NUMINAMATH_GPT_reduced_price_l155_15575


namespace NUMINAMATH_GPT_boy_scouts_signed_slips_l155_15501

-- Definitions for the problem conditions have only been used; solution steps are excluded.

theorem boy_scouts_signed_slips (total_scouts : ℕ) (signed_slips : ℕ) (boy_scouts : ℕ) (girl_scouts : ℕ)
  (boy_scouts_signed : ℕ) (girl_scouts_signed : ℕ)
  (h1 : signed_slips = 4 * total_scouts / 5)  -- 80% of the scouts arrived with signed permission slips
  (h2 : boy_scouts = 2 * total_scouts / 5)  -- 40% of the scouts were boy scouts
  (h3 : girl_scouts = total_scouts - boy_scouts)  -- Rest are girl scouts
  (h4 : girl_scouts_signed = 8333 * girl_scouts / 10000)  -- 83.33% of girl scouts with permission slips
  (h5 : signed_slips = boy_scouts_signed + girl_scouts_signed)  -- Total signed slips by both boy and girl scouts
  : (boy_scouts_signed * 100 / boy_scouts = 75) :=    -- 75% of boy scouts with permission slips
by
  -- Proof to be filled in.
  sorry

end NUMINAMATH_GPT_boy_scouts_signed_slips_l155_15501


namespace NUMINAMATH_GPT_expression_value_l155_15505

theorem expression_value (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (h1 : x + y + z = 0) (h2 : xy + xz + yz ≠ 0) :
  (x^3 + y^3 + z^3) / (xyz * (xy + xz + yz)^2) = 3 / (x^2 + xy + y^2)^2 :=
by
  sorry

end NUMINAMATH_GPT_expression_value_l155_15505


namespace NUMINAMATH_GPT_sequence_b_10_eq_110_l155_15581

theorem sequence_b_10_eq_110 :
  (∃ (b : ℕ → ℕ), b 1 = 2 ∧ (∀ m n : ℕ, b (m + n) = b m + b n + 2 * m * n) ∧ b 10 = 110) :=
sorry

end NUMINAMATH_GPT_sequence_b_10_eq_110_l155_15581


namespace NUMINAMATH_GPT_tan_alpha_20_l155_15549

theorem tan_alpha_20 (α : ℝ) 
  (h : Real.tan (α + 80 * Real.pi / 180) = 4 * Real.sin (420 * Real.pi / 180)) : 
  Real.tan (α + 20 * Real.pi / 180) = Real.sqrt 3 / 7 := 
sorry

end NUMINAMATH_GPT_tan_alpha_20_l155_15549


namespace NUMINAMATH_GPT_largest_integer_n_l155_15524

theorem largest_integer_n (n : ℤ) :
  (n^2 - 11 * n + 24 < 0) → n ≤ 7 :=
by
  sorry

end NUMINAMATH_GPT_largest_integer_n_l155_15524


namespace NUMINAMATH_GPT_debby_photos_of_friends_l155_15574

theorem debby_photos_of_friends (F : ℕ) (h1 : 23 + F = 86) : F = 63 := by
  -- Proof steps will go here
  sorry

end NUMINAMATH_GPT_debby_photos_of_friends_l155_15574


namespace NUMINAMATH_GPT_BANANA_arrangements_l155_15540

theorem BANANA_arrangements : 
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  (n.factorial / (n1.factorial * n2.factorial * n3.factorial)) = 60 :=
by
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  have h : n.factorial / (n1.factorial * n2.factorial * n3.factorial) = 60 := sorry
  exact h

end NUMINAMATH_GPT_BANANA_arrangements_l155_15540


namespace NUMINAMATH_GPT_derivative_of_sin_squared_is_sin_2x_l155_15557

open Real

noncomputable def f (x : ℝ) : ℝ := sin x ^ 2

theorem derivative_of_sin_squared_is_sin_2x : 
  ∀ x : ℝ, deriv f x = sin (2 * x) :=
by
  sorry

end NUMINAMATH_GPT_derivative_of_sin_squared_is_sin_2x_l155_15557


namespace NUMINAMATH_GPT_volume_of_fuel_A_l155_15564

variables (V_A V_B : ℝ)

def condition1 := V_A + V_B = 212
def condition2 := 0.12 * V_A + 0.16 * V_B = 30

theorem volume_of_fuel_A :
  condition1 V_A V_B → condition2 V_A V_B → V_A = 98 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_volume_of_fuel_A_l155_15564


namespace NUMINAMATH_GPT_correct_operation_l155_15538

theorem correct_operation (a : ℝ) : 
    (a ^ 2 + a ^ 4 ≠ a ^ 6) ∧ 
    (a ^ 2 * a ^ 3 ≠ a ^ 6) ∧ 
    (a ^ 3 / a ^ 2 = a) ∧ 
    ((a ^ 2) ^ 3 ≠ a ^ 5) :=
by
  sorry

end NUMINAMATH_GPT_correct_operation_l155_15538


namespace NUMINAMATH_GPT_difference_of_roots_l155_15596

theorem difference_of_roots (r1 r2 : ℝ) 
    (h_eq : ∀ x : ℝ, x^2 - 9 * x + 4 = 0 ↔ x = r1 ∨ x = r2) : 
    abs (r1 - r2) = Real.sqrt 65 := 
sorry

end NUMINAMATH_GPT_difference_of_roots_l155_15596


namespace NUMINAMATH_GPT_focus_of_parabola_l155_15584

-- Definitions for the problem
def parabola_eq (x y : ℝ) : Prop := y = 2 * x^2

def general_parabola_form (x y h k p : ℝ) : Prop :=
  4 * p * (y - k) = (x - h)^2

def vertex_origin (h k : ℝ) : Prop := h = 0 ∧ k = 0

-- Lean statement asserting that the focus of the given parabola is (0, 1/8)
theorem focus_of_parabola : ∃ p : ℝ, parabola_eq x y → general_parabola_form x y 0 0 p ∧ p = 1/8 := by
  sorry

end NUMINAMATH_GPT_focus_of_parabola_l155_15584


namespace NUMINAMATH_GPT_root_range_of_f_eq_zero_solution_set_of_f_le_zero_l155_15592

variable (m : ℝ) (x : ℝ)

def f (x : ℝ) : ℝ := m * x^2 + (2 * m + 1) * x + 2

theorem root_range_of_f_eq_zero (h : ∃ r1 r2 : ℝ, r1 > 1 ∧ r2 < 1 ∧ f r1 = 0 ∧ f r2 = 0) : -1 < m ∧ m < 0 :=
sorry

theorem solution_set_of_f_le_zero : 
  (m = 0 -> ∀ x, f x ≤ 0 ↔ x ≤ - 2) ∧
  (m < 0 -> ∀ x, f x ≤ 0 ↔ -2 ≤ x ∧ x ≤ - (1/m)) ∧
  (0 < m ∧ m < 1/2 -> ∀ x, f x ≤ 0 ↔ - (1/m) ≤ x ∧ x ≤ - 2) ∧
  (m = 1/2 -> ∀ x, f x ≤ 0 ↔ x = - 2) ∧
  (m > 1/2 -> ∀ x, f x ≤ 0 ↔ -2 ≤ x ∧ x ≤ - (1/m)) :=
sorry

end NUMINAMATH_GPT_root_range_of_f_eq_zero_solution_set_of_f_le_zero_l155_15592


namespace NUMINAMATH_GPT_trivia_team_members_l155_15558

theorem trivia_team_members (x : ℕ) (h : 3 * (x - 6) = 27) : x = 15 := 
by
  sorry

end NUMINAMATH_GPT_trivia_team_members_l155_15558


namespace NUMINAMATH_GPT_pyramid_can_be_oblique_l155_15585

-- Define what it means for the pyramid to have a regular triangular base.
def regular_triangular_base (pyramid : Type) : Prop := sorry

-- Define what it means for each lateral face to be an isosceles triangle.
def isosceles_lateral_faces (pyramid : Type) : Prop := sorry

-- Define what it means for a pyramid to be oblique.
def can_be_oblique (pyramid : Type) : Prop := sorry

-- Defining pyramid as a type.
variable (pyramid : Type)

-- The theorem stating the problem's conclusion.
theorem pyramid_can_be_oblique 
  (h1 : regular_triangular_base pyramid) 
  (h2 : isosceles_lateral_faces pyramid) : 
  can_be_oblique pyramid :=
sorry

end NUMINAMATH_GPT_pyramid_can_be_oblique_l155_15585


namespace NUMINAMATH_GPT_range_of_k_l155_15529

theorem range_of_k (k : ℝ) :
  (∃ x y : ℝ, (x - 3)^2 + (y - 2)^2 = 4 ∧ y = k * x + 3) ∧ 
  (∃ M N : ℝ × ℝ, ((M.1 - N.1)^2 + (M.2 - N.2)^2)^(1/2) ≥ 2) →
  (k ≤ 0) :=
by
  sorry

end NUMINAMATH_GPT_range_of_k_l155_15529


namespace NUMINAMATH_GPT_num_ordered_pairs_l155_15542

theorem num_ordered_pairs : ∃! n : ℕ, n = 4 ∧ 
  ∃ (x y : ℤ), y = (x - 90)^2 - 4907 ∧ 
  (∃ m : ℕ, y = m^2) := 
sorry

end NUMINAMATH_GPT_num_ordered_pairs_l155_15542


namespace NUMINAMATH_GPT_lines_intersect_at_single_point_l155_15577

def line1 (a b x y: ℝ) := a * x + 2 * b * y + 3 * (a + b + 1) = 0
def line2 (a b x y: ℝ) := b * x + 2 * (a + b + 1) * y + 3 * a = 0
def line3 (a b x y: ℝ) := (a + b + 1) * x + 2 * a * y + 3 * b = 0

theorem lines_intersect_at_single_point (a b : ℝ) :
  (∃ x y : ℝ, line1 a b x y ∧ line2 a b x y ∧ line3 a b x y) ↔ a + b = -1/2 :=
by
  sorry

end NUMINAMATH_GPT_lines_intersect_at_single_point_l155_15577


namespace NUMINAMATH_GPT_stickers_distribution_l155_15579

theorem stickers_distribution : 
  (10 + 5 - 1).choose (5 - 1) = 1001 := 
by
  sorry

end NUMINAMATH_GPT_stickers_distribution_l155_15579


namespace NUMINAMATH_GPT_simplify_expression_1_simplify_expression_2_l155_15525

-- Define the algebraic simplification problem for the first expression
theorem simplify_expression_1 (x y : ℝ) : 5 * x - 3 * (2 * x - 3 * y) + x = 9 * y :=
by
  sorry

-- Define the algebraic simplification problem for the second expression
theorem simplify_expression_2 (a : ℝ) : 3 * a^2 + 5 - 2 * a^2 - 2 * a + 3 * a - 8 = a^2 + a - 3 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_1_simplify_expression_2_l155_15525


namespace NUMINAMATH_GPT_atomic_weight_chlorine_l155_15527

-- Define the given conditions and constants
def molecular_weight_compound : ℝ := 53
def atomic_weight_nitrogen : ℝ := 14.01
def atomic_weight_hydrogen : ℝ := 1.01
def number_of_hydrogen_atoms : ℝ := 4
def number_of_nitrogen_atoms : ℝ := 1

-- Define the total weight of nitrogen and hydrogen in the compound
def total_weight_nh : ℝ := (number_of_nitrogen_atoms * atomic_weight_nitrogen) + (number_of_hydrogen_atoms * atomic_weight_hydrogen)

-- Define the statement to be proved: the atomic weight of chlorine
theorem atomic_weight_chlorine : (molecular_weight_compound - total_weight_nh) = 34.95 := by
  sorry

end NUMINAMATH_GPT_atomic_weight_chlorine_l155_15527


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_l155_15587

theorem necessary_and_sufficient_condition (x : ℝ) : (0 < (1 / x) ∧ (1 / x) < 1) ↔ (1 < x) := sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_l155_15587


namespace NUMINAMATH_GPT_motorcyclists_speeds_l155_15517

theorem motorcyclists_speeds 
  (distance_AB : ℝ) (distance1 : ℝ) (distance2 : ℝ) (time_diff : ℝ) 
  (x y : ℝ) 
  (h1 : distance_AB = 600) 
  (h2 : distance1 = 250) 
  (h3 : distance2 = 200) 
  (h4 : time_diff = 3)
  (h5 : distance1 / x = distance2 / y)
  (h6 : distance_AB / x + time_diff = distance_AB / y) : 
  x = 50 ∧ y = 40 := 
sorry

end NUMINAMATH_GPT_motorcyclists_speeds_l155_15517


namespace NUMINAMATH_GPT_determine_k_l155_15588

-- Definitions of the vectors a and b.
variables (a b : ℝ)

-- Noncomputable definition of the scalar k.
noncomputable def k_value : ℝ :=
  (2 : ℚ) / 7

-- Definition of line through vectors a and b as a parametric equation.
def line_through (a b : ℝ) (t : ℝ) : ℝ :=
  a + t * (b - a)

-- Hypothesis: The vector k * a + (5/7) * b is on the line passing through a and b.
def vector_on_line (a b : ℝ) (k : ℝ) : Prop :=
  ∃ t : ℝ, k * a + (5/7) * b = line_through a b t

-- Proof that k must be 2/7 for the vector to be on the line.
theorem determine_k (a b : ℝ) : vector_on_line a b k_value :=
by sorry

end NUMINAMATH_GPT_determine_k_l155_15588


namespace NUMINAMATH_GPT_problem_ns_k_divisibility_l155_15533

theorem problem_ns_k_divisibility (n k : ℕ) (h1 : 0 < n) (h2 : 0 < k) :
  (∃ (a b : ℕ), (a = 1 ∨ a = 5) ∧ (b = 1 ∨ b = 5) ∧ a = n ∧ b = k) ↔ 
  n * k ∣ (2^(2^n) + 1) * (2^(2^k) + 1) := 
sorry

end NUMINAMATH_GPT_problem_ns_k_divisibility_l155_15533


namespace NUMINAMATH_GPT_canoe_trip_shorter_l155_15530

def lake_diameter : ℝ := 2
def pi_value : ℝ := 3.14

theorem canoe_trip_shorter : (2 * pi_value * (lake_diameter / 2) - lake_diameter) = 4.28 :=
by
  sorry

end NUMINAMATH_GPT_canoe_trip_shorter_l155_15530


namespace NUMINAMATH_GPT_quotient_real_iff_quotient_purely_imaginary_iff_l155_15508

variables {a b c d : ℝ} -- Declare real number variables

-- Problem 1: Proving the necessary and sufficient condition for the quotient to be a real number
theorem quotient_real_iff (a b c d : ℝ) : 
  (c ≠ 0 ∨ d ≠ 0) → 
  (∀ i : ℝ, ∃ r : ℝ, a/c = r ∧ b/d = 0) ↔ (a * d - b * c = 0) := 
by sorry -- Proof to be filled in

-- Problem 2: Proving the necessary and sufficient condition for the quotient to be a purely imaginary number
theorem quotient_purely_imaginary_iff (a b c d : ℝ) : 
  (c ≠ 0 ∨ d ≠ 0) → 
  (∀ r : ℝ, ∃ i : ℝ, a/c = 0 ∧ b/d = i) ↔ (a * c + b * d = 0) := 
by sorry -- Proof to be filled in

end NUMINAMATH_GPT_quotient_real_iff_quotient_purely_imaginary_iff_l155_15508


namespace NUMINAMATH_GPT_football_cost_l155_15591

-- Definitions derived from conditions
def marbles_cost : ℝ := 9.05
def baseball_cost : ℝ := 6.52
def total_spent : ℝ := 20.52

-- The statement to prove the cost of the football
theorem football_cost :
  ∃ (football_cost : ℝ), football_cost = total_spent - marbles_cost - baseball_cost :=
sorry

end NUMINAMATH_GPT_football_cost_l155_15591


namespace NUMINAMATH_GPT_sum_of_reciprocal_squares_of_roots_l155_15567

theorem sum_of_reciprocal_squares_of_roots (a b c : ℝ) 
    (h_roots : ∀ x : ℝ, x^3 - 6 * x^2 + 11 * x - 6 = 0 → x = a ∨ x = b ∨ x = c) :
    a + b + c = 6 ∧ ab + bc + ca = 11 ∧ abc = 6 → 
    (1 / a^2) + (1 / b^2) + (1 / c^2) = 49 / 36 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_reciprocal_squares_of_roots_l155_15567


namespace NUMINAMATH_GPT_article_filling_correct_l155_15590

-- definitions based on conditions provided
def Gottlieb_Daimler := "Gottlieb Daimler was a German engineer."
def Invented_Car := "Daimler is normally believed to have invented the car."

-- Statement we want to prove
theorem article_filling_correct : 
  (Gottlieb_Daimler = "Gottlieb Daimler was a German engineer.") ∧ 
  (Invented_Car = "Daimler is normally believed to have invented the car.") →
  ("Gottlieb Daimler, a German engineer, is normally believed to have invented the car." = 
   "Gottlieb Daimler, a German engineer, is normally believed to have invented the car.") :=
by
  sorry

end NUMINAMATH_GPT_article_filling_correct_l155_15590


namespace NUMINAMATH_GPT_find_n_for_arithmetic_sequence_l155_15583

variable {a : ℕ → ℤ}

-- Conditions
def is_arithmetic_sequence (a : ℕ → ℤ) (a₁ : ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a n = a₁ + n * d

theorem find_n_for_arithmetic_sequence (h_arith : is_arithmetic_sequence a (-1) 2)
  (h_nth_term : ∃ n : ℕ, a n = 15) : ∃ n : ℕ, n = 9 :=
by
  sorry

end NUMINAMATH_GPT_find_n_for_arithmetic_sequence_l155_15583


namespace NUMINAMATH_GPT_number_of_m_values_l155_15572

theorem number_of_m_values (m : ℕ) (h1 : 4 * m > 11) (h2 : m < 12) : 
  11 - 3 + 1 = 9 := 
sorry

end NUMINAMATH_GPT_number_of_m_values_l155_15572


namespace NUMINAMATH_GPT_value_of_a_l155_15548

theorem value_of_a (x y z a : ℤ) (k : ℤ) 
  (h1 : x = 4 * k) (h2 : y = 6 * k) (h3 : z = 10 * k) 
  (hy_eq : y^2 = 40 * a - 20) 
  (ha_int : ∃ m : ℤ, a = m) : a = 1 := 
  sorry

end NUMINAMATH_GPT_value_of_a_l155_15548


namespace NUMINAMATH_GPT_sum_of_distinct_prime_factors_of_462_l155_15528

-- Given a number n, define its prime factors.
def prime_factors (n : ℕ) : List ℕ :=
  if h : n = 462 then [2, 3, 7, 11] else []

-- Defines the sum of a list of natural numbers.
def sum_list (l : List ℕ) : ℕ :=
  l.foldr (· + ·) 0

-- The main theorem statement.
theorem sum_of_distinct_prime_factors_of_462 : sum_list (prime_factors 462) = 23 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_distinct_prime_factors_of_462_l155_15528


namespace NUMINAMATH_GPT_solution_set_of_inequality_l155_15594

theorem solution_set_of_inequality (a x : ℝ) (h : 1 < a) :
  (x - a) * (x - (1 / a)) > 0 ↔ x < 1 / a ∨ x > a :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l155_15594


namespace NUMINAMATH_GPT_base_conversion_min_sum_l155_15586

theorem base_conversion_min_sum (a b : ℕ) (h1 : 3 * a + 6 = 6 * b + 3) (h2 : 6 < a) (h3 : 6 < b) : a + b = 20 :=
sorry

end NUMINAMATH_GPT_base_conversion_min_sum_l155_15586


namespace NUMINAMATH_GPT_dress_hem_length_in_feet_l155_15597

def stitch_length_in_inches : ℚ := 1 / 4
def stitches_per_minute : ℕ := 24
def time_in_minutes : ℕ := 6

theorem dress_hem_length_in_feet :
  (stitch_length_in_inches * (stitches_per_minute * time_in_minutes)) / 12 = 3 :=
by
  sorry

end NUMINAMATH_GPT_dress_hem_length_in_feet_l155_15597


namespace NUMINAMATH_GPT_translate_line_up_l155_15544

-- Define the original line equation as a function
def original_line (x : ℝ) : ℝ := -2 * x

-- Define the transformed line equation as a function
def translated_line (x : ℝ) : ℝ := -2 * x + 1

-- Prove that translating the original line upward by 1 unit gives the translated line
theorem translate_line_up (x : ℝ) :
  original_line x + 1 = translated_line x :=
by
  unfold original_line translated_line
  simp

end NUMINAMATH_GPT_translate_line_up_l155_15544


namespace NUMINAMATH_GPT_find_range_a_l155_15502

def bounded_a (a : ℝ) : Prop :=
  ∀ x : ℝ, x ≤ 2 → a * (4 ^ x) + 2 ^ x + 1 ≥ 0

theorem find_range_a :
  ∃ (a : ℝ), bounded_a a ↔ a ≥ -5 / 16 :=
sorry

end NUMINAMATH_GPT_find_range_a_l155_15502


namespace NUMINAMATH_GPT_part1_part2_l155_15559

open Set

variable (a : ℝ)

def real_universe := @univ ℝ

def set_A : Set ℝ := {x | 3 ≤ x ∧ x ≤ 7}
def set_B : Set ℝ := {x | 2 < x ∧ x < 10}
def set_C (a : ℝ) : Set ℝ := {x | x ≤ a}

noncomputable def complement_A := (real_universe \ set_A)

theorem part1 : (complement_A ∩ set_B) = { x | (2 < x ∧ x < 3) ∨ (7 < x ∧ x < 10) } :=
by sorry

theorem part2 : set_A ⊆ set_C a → a > 7 :=
by sorry

end NUMINAMATH_GPT_part1_part2_l155_15559


namespace NUMINAMATH_GPT_multiply_same_exponents_l155_15534

theorem multiply_same_exponents (x : ℝ) : (x^3) * (x^3) = x^6 :=
by sorry

end NUMINAMATH_GPT_multiply_same_exponents_l155_15534


namespace NUMINAMATH_GPT_bus_stops_bound_l155_15507

-- Definitions based on conditions
variables (n x : ℕ)

-- Condition 1: Any bus stop is serviced by at most 3 bus lines
def at_most_three_bus_lines (bus_stops : ℕ) : Prop :=
  ∀ (stop : ℕ), stop < bus_stops → stop ≤ 3

-- Condition 2: Any bus line has at least two stops
def at_least_two_stops (bus_lines : ℕ) : Prop :=
  ∀ (line : ℕ), line < bus_lines → line ≥ 2

-- Condition 3: For any two specific bus lines, there is a third line such that passengers can transfer
def transfer_line_exists (bus_lines : ℕ) : Prop :=
  ∀ (line1 line2 : ℕ), line1 < bus_lines ∧ line2 < bus_lines →
  ∃ (line3 : ℕ), line3 < bus_lines

-- Theorem statement: The number of bus stops is at least 5/6 (n-5)
theorem bus_stops_bound (h1 : at_most_three_bus_lines x) (h2 : at_least_two_stops n)
  (h3 : transfer_line_exists n) : x ≥ (5 * (n - 5)) / 6 :=
sorry

end NUMINAMATH_GPT_bus_stops_bound_l155_15507


namespace NUMINAMATH_GPT_verify_drawn_numbers_when_x_is_24_possible_values_of_x_l155_15503

-- Population size and group division
def population_size := 1000
def number_of_groups := 10
def group_size := population_size / number_of_groups

-- Systematic sampling function
def systematic_sample (x : ℕ) (k : ℕ) : ℕ :=
  (x + 33 * k) % 1000

-- Prove the drawn 10 numbers when x = 24
theorem verify_drawn_numbers_when_x_is_24 :
  (∃ drawn_numbers, drawn_numbers = [24, 157, 290, 323, 456, 589, 622, 755, 888, 921]) :=
  sorry

-- Prove possible values of x given last two digits equal to 87
theorem possible_values_of_x (k : ℕ) (h : k < number_of_groups) :
  (∃ x_values, x_values = [87, 54, 21, 88, 55, 22, 89, 56, 23, 90]) :=
  sorry

end NUMINAMATH_GPT_verify_drawn_numbers_when_x_is_24_possible_values_of_x_l155_15503


namespace NUMINAMATH_GPT_num_of_arithmetic_sequences_l155_15513

-- Define the set of digits {1, 2, ..., 15}
def digits := {n : ℕ | 1 ≤ n ∧ n ≤ 15}

-- Define an arithmetic sequence condition 
def is_arithmetic_sequence (a b c : ℕ) (d : ℕ) : Prop :=
  b - a = d ∧ c - b = d

-- Define the count of valid sequences with a specific difference
def count_arithmetic_sequences_with_difference (d : ℕ) : ℕ :=
  if d = 1 then 13
  else if d = 5 then 6
  else 0

-- Define the total count of valid sequences
def total_arithmetic_sequences : ℕ :=
  count_arithmetic_sequences_with_difference 1 +
  count_arithmetic_sequences_with_difference 5

-- The final statement to prove
theorem num_of_arithmetic_sequences : total_arithmetic_sequences = 19 := 
  sorry

end NUMINAMATH_GPT_num_of_arithmetic_sequences_l155_15513


namespace NUMINAMATH_GPT_third_quadrant_angle_bisector_l155_15551

theorem third_quadrant_angle_bisector
  (a b : ℝ)
  (hA : A = (-4,a))
  (hB : B = (-2,b))
  (h_lineA : a = -4)
  (h_lineB : b = -2)
  : a + b + a * b = 2 :=
by
  sorry

end NUMINAMATH_GPT_third_quadrant_angle_bisector_l155_15551


namespace NUMINAMATH_GPT_min_sum_geometric_sequence_l155_15570

noncomputable def sequence_min_value (a : ℕ → ℝ) : ℝ :=
  a 4 + a 3 - 2 * a 2 - 2 * a 1

theorem min_sum_geometric_sequence (a : ℕ → ℝ)
  (h : sequence_min_value a = 6) :
  a 5 + a 6 = 48 := 
by
  sorry

end NUMINAMATH_GPT_min_sum_geometric_sequence_l155_15570


namespace NUMINAMATH_GPT_solve_for_q_l155_15547

theorem solve_for_q (k r q : ℕ) (h1 : 4 / 5 = k / 90) (h2 : 4 / 5 = (k + r) / 105) (h3 : 4 / 5 = (q - r) / 150) : q = 132 := 
  sorry

end NUMINAMATH_GPT_solve_for_q_l155_15547


namespace NUMINAMATH_GPT_correct_division_result_l155_15519

theorem correct_division_result (x : ℝ) (h : 4 * x = 166.08) : x / 4 = 10.38 :=
by
  sorry

end NUMINAMATH_GPT_correct_division_result_l155_15519


namespace NUMINAMATH_GPT_no_nat_solutions_m_sq_eq_n_sq_plus_2014_l155_15545

theorem no_nat_solutions_m_sq_eq_n_sq_plus_2014 :
  ¬ ∃ (m n : ℕ), m ^ 2 = n ^ 2 + 2014 := 
sorry

end NUMINAMATH_GPT_no_nat_solutions_m_sq_eq_n_sq_plus_2014_l155_15545


namespace NUMINAMATH_GPT_hex_product_l155_15562

def hex_to_dec (h : Char) : Nat :=
  match h with
  | 'A' => 10
  | 'B' => 11
  | 'C' => 12
  | 'D' => 13
  | 'E' => 14
  | 'F' => 15
  | c   => c.toNat - '0'.toNat

noncomputable def dec_to_hex (n : Nat) : String :=
  let q := n / 16
  let r := n % 16
  let r_hex := if r < 10 then Char.ofNat (r + '0'.toNat) else Char.ofNat (r - 10 + 'A'.toNat)
  (if q > 0 then toString q else "") ++ Char.toString r_hex

theorem hex_product :
  dec_to_hex (hex_to_dec 'A' * hex_to_dec 'B') = "6E" :=
by
  sorry

end NUMINAMATH_GPT_hex_product_l155_15562


namespace NUMINAMATH_GPT_max_choir_members_l155_15550

theorem max_choir_members : 
  ∃ (m : ℕ), 
    (∃ k : ℕ, m = k^2 + 11) ∧ 
    (∃ n : ℕ, m = n * (n + 5)) ∧ 
    (∀ m' : ℕ, 
      ((∃ k' : ℕ, m' = k' * k' + 11) ∧ 
       (∃ n' : ℕ, m' = n' * (n' + 5))) → 
      m' ≤ 266) ∧ 
    m = 266 :=
by sorry

end NUMINAMATH_GPT_max_choir_members_l155_15550


namespace NUMINAMATH_GPT_john_payment_l155_15536

def camera_value : ℝ := 5000
def weekly_rental_percentage : ℝ := 0.10
def rental_period : ℕ := 4
def friend_contribution_percentage : ℝ := 0.40

theorem john_payment :
  let weekly_rental_fee := camera_value * weekly_rental_percentage
  let total_rental_fee := weekly_rental_fee * rental_period
  let friend_contribution := total_rental_fee * friend_contribution_percentage
  let john_payment := total_rental_fee - friend_contribution
  john_payment = 1200 :=
by
  sorry

end NUMINAMATH_GPT_john_payment_l155_15536


namespace NUMINAMATH_GPT_melanie_books_bought_l155_15523

def books_before_yard_sale : ℝ := 41.0
def books_after_yard_sale : ℝ := 128
def books_bought : ℝ := books_after_yard_sale - books_before_yard_sale

theorem melanie_books_bought : books_bought = 87 := by
  sorry

end NUMINAMATH_GPT_melanie_books_bought_l155_15523


namespace NUMINAMATH_GPT_triangle_inequality_l155_15532

theorem triangle_inequality (a b c : ℝ) (h1 : a + b + c = 2)
  (h2 : a > 0) (h3 : b > 0) (h4 : c > 0)
  (h5 : a < b + c) (h6 : b < a + c) (h7 : c < a + b) :
  a^2 + b^2 + c^2 + 2 * a * b * c < 2 := 
sorry

end NUMINAMATH_GPT_triangle_inequality_l155_15532


namespace NUMINAMATH_GPT_round_trip_time_l155_15554

theorem round_trip_time (boat_speed : ℝ) (stream_speed : ℝ) (distance : ℝ) : 
  boat_speed = 8 → stream_speed = 2 → distance = 210 → 
  ((distance / (boat_speed - stream_speed)) + (distance / (boat_speed + stream_speed))) = 56 :=
by
  intros hb hs hd
  sorry

end NUMINAMATH_GPT_round_trip_time_l155_15554


namespace NUMINAMATH_GPT_exponential_function_passes_through_fixed_point_l155_15509

theorem exponential_function_passes_through_fixed_point {a : ℝ} (ha_pos : a > 0) (ha_ne_one : a ≠ 1) : 
  (a^(2 - 2) + 3) = 4 :=
by
  sorry

end NUMINAMATH_GPT_exponential_function_passes_through_fixed_point_l155_15509


namespace NUMINAMATH_GPT_arithmetic_sequence_seventh_term_l155_15515

theorem arithmetic_sequence_seventh_term (a d : ℚ) 
  (h1 : a + (a + d) + (a + 2 * d) + (a + 3 * d) + (a + 4 * d) = 15)
  (h2 : a + 5 * d = 8) :
  a + 6 * d = 29 / 3 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_seventh_term_l155_15515


namespace NUMINAMATH_GPT_sn_geq_mnplus1_l155_15589

namespace Polysticks

def n_stick (n : ℕ) : Type := sorry -- formalize the definition of n-stick
def n_mino (n : ℕ) : Type := sorry -- formalize the definition of n-mino

def S (n : ℕ) : ℕ := sorry -- define the number of n-sticks
def M (n : ℕ) : ℕ := sorry -- define the number of n-minos

theorem sn_geq_mnplus1 (n : ℕ) : S n ≥ M (n+1) := sorry

end Polysticks

end NUMINAMATH_GPT_sn_geq_mnplus1_l155_15589


namespace NUMINAMATH_GPT_amelia_money_left_l155_15546

theorem amelia_money_left :
  let first_course := 15
  let second_course := first_course + 5
  let dessert := 0.25 * second_course
  let total_first_three_courses := first_course + second_course + dessert
  let drink := 0.20 * total_first_three_courses
  let pre_tip_total := total_first_three_courses + drink
  let tip := 0.15 * pre_tip_total
  let total_bill := pre_tip_total + tip
  let initial_money := 60
  let money_left := initial_money - total_bill
  money_left = 4.8 :=
by
  sorry

end NUMINAMATH_GPT_amelia_money_left_l155_15546


namespace NUMINAMATH_GPT_cos_pi_div_3_l155_15522

theorem cos_pi_div_3 : Real.cos (π / 3) = 1 / 2 := 
by
  sorry

end NUMINAMATH_GPT_cos_pi_div_3_l155_15522


namespace NUMINAMATH_GPT_distance_from_pole_l155_15569

-- Define the structure for polar coordinates.
structure PolarCoordinates where
  r : ℝ
  θ : ℝ

-- Define point A with its polar coordinates.
def A : PolarCoordinates := { r := 3, θ := -4 }

-- State the problem to prove that the distance |OA| is 3.
theorem distance_from_pole (A : PolarCoordinates) : A.r = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_distance_from_pole_l155_15569


namespace NUMINAMATH_GPT_complex_division_l155_15573

noncomputable def imagine_unit : ℂ := Complex.I

theorem complex_division :
  (Complex.mk (-3) 1) / (Complex.mk 1 (-1)) = (Complex.mk (-2) 1) :=
by
sorry

end NUMINAMATH_GPT_complex_division_l155_15573


namespace NUMINAMATH_GPT_ratio_of_wire_lengths_l155_15582

theorem ratio_of_wire_lengths (b_pieces : ℕ) (b_piece_length : ℕ)
  (c_piece_length : ℕ) (cubes_volume : ℕ) :
  b_pieces = 12 →
  b_piece_length = 8 →
  c_piece_length = 2 →
  cubes_volume = (b_piece_length ^ 3) →
  b_pieces * b_piece_length * cubes_volume
    / (cubes_volume * (12 * c_piece_length)) = (1 / 128) :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_ratio_of_wire_lengths_l155_15582


namespace NUMINAMATH_GPT_parabola_equation_hyperbola_equation_l155_15566

-- Part 1: Prove the standard equation of the parabola given the directrix.
theorem parabola_equation (x y : ℝ) : x = -2 → y^2 = 8 * x := 
by
  -- Here we will include proof steps based on given conditions
  sorry

-- Part 2: Prove the standard equation of the hyperbola given center at origin, focus on the x-axis,
-- the given asymptotes, and its real axis length.
theorem hyperbola_equation (x y a b : ℝ) : 
  a = 1 → b = 2 → y = 2 * x ∨ y = -2 * x → x^2 - (y^2 / 4) = 1 :=
by
  -- Here we will include proof steps based on given conditions
  sorry

end NUMINAMATH_GPT_parabola_equation_hyperbola_equation_l155_15566


namespace NUMINAMATH_GPT_quadrilateral_area_l155_15593

theorem quadrilateral_area (EF FG EH HG : ℕ) (hEFH : EF * EF + FG * FG = 25)
(hEHG : EH * EH + HG * HG = 25) (h_distinct : EF ≠ EH ∧ FG ≠ HG) 
(h_greater_one : EF > 1 ∧ FG > 1 ∧ EH > 1 ∧ HG > 1) :
  (EF * FG) / 2 + (EH * HG) / 2 = 12 := 
sorry

end NUMINAMATH_GPT_quadrilateral_area_l155_15593


namespace NUMINAMATH_GPT_wage_difference_l155_15543

-- Definitions of the problem
variables (P Q h : ℝ)
axiom total_pay : P * h = 480
axiom wage_relation : P = 1.5 * Q
axiom time_relation : Q * (h + 10) = 480

-- Theorem to prove the hourly wage difference
theorem wage_difference : P - Q = 8 :=
by
  sorry

end NUMINAMATH_GPT_wage_difference_l155_15543


namespace NUMINAMATH_GPT_union_A_B_at_a_3_inter_B_compl_A_at_a_3_B_subset_A_imp_a_range_l155_15504

open Set

variables (U : Set ℝ) (A B : Set ℝ) (a : ℝ)

def A_def : Set ℝ := { x | 1 ≤ x ∧ x ≤ 4 }
def B_def (a : ℝ) : Set ℝ := { x | a ≤ x ∧ x ≤ a + 2 }
def comp_U_A : Set ℝ := { x | x < 1 ∨ x > 4 }

theorem union_A_B_at_a_3 (h : a = 3) :
  A_def ∪ B_def 3 = { x | 1 ≤ x ∧ x ≤ 5 } :=
sorry

theorem inter_B_compl_A_at_a_3 (h : a = 3) :
  B_def 3 ∩ comp_U_A = { x | 4 < x ∧ x ≤ 5 } :=
sorry

theorem B_subset_A_imp_a_range (h : B_def a ⊆ A_def) :
  1 ≤ a ∧ a ≤ 2 :=
sorry

end NUMINAMATH_GPT_union_A_B_at_a_3_inter_B_compl_A_at_a_3_B_subset_A_imp_a_range_l155_15504


namespace NUMINAMATH_GPT_fourth_term_of_geometric_sequence_is_320_l155_15520

theorem fourth_term_of_geometric_sequence_is_320
  (a : ℕ) (r : ℕ)
  (h_a : a = 5)
  (h_fifth_term : a * r^4 = 1280) :
  a * r^3 = 320 := 
by
  sorry

end NUMINAMATH_GPT_fourth_term_of_geometric_sequence_is_320_l155_15520


namespace NUMINAMATH_GPT_original_people_complete_work_in_four_days_l155_15510

noncomputable def original_people_work_days (P D : ℕ) :=
  (2 * P) * 2 = (1 / 2) * (P * D)

theorem original_people_complete_work_in_four_days (P D : ℕ) (h : original_people_work_days P D) : D = 4 :=
by
  sorry

end NUMINAMATH_GPT_original_people_complete_work_in_four_days_l155_15510


namespace NUMINAMATH_GPT_stockholm_uppsala_distance_l155_15568

variable (map_distance : ℝ) (scale_factor : ℝ)

def actual_distance (d : ℝ) (s : ℝ) : ℝ := d * s

theorem stockholm_uppsala_distance :
  actual_distance 65 20 = 1300 := by
  sorry

end NUMINAMATH_GPT_stockholm_uppsala_distance_l155_15568


namespace NUMINAMATH_GPT_train_speed_approx_900072_kmph_l155_15535

noncomputable def speed_of_train (train_length platform_length time_seconds : ℝ) : ℝ :=
  let total_distance := train_length + platform_length
  let speed_m_s := total_distance / time_seconds
  speed_m_s * 3.6

theorem train_speed_approx_900072_kmph :
  abs (speed_of_train 225 400.05 25 - 90.0072) < 0.001 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_approx_900072_kmph_l155_15535


namespace NUMINAMATH_GPT_brocard_inequalities_l155_15526

theorem brocard_inequalities (α β γ φ: ℝ) (h1: φ > 0) (h2: φ < π / 6)
  (h3: α > 0) (h4: β > 0) (h5: γ > 0) (h6: α + β + γ = π) : 
  (φ^3 ≤ (α - φ) * (β - φ) * (γ - φ)) ∧ (8 * φ^3 ≤ α * β * γ) := 
by 
  sorry

end NUMINAMATH_GPT_brocard_inequalities_l155_15526


namespace NUMINAMATH_GPT_part1_part2_l155_15541

-- Define what a double root equation is
def is_double_root_eq (a b c : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ * x₁ * a + x₁ * b + c = 0 ∧ x₂ = 2 * x₁ ∧ x₂ * x₂ * a + x₂ * b + c = 0

-- Statement for part 1: proving x^2 - 3x + 2 = 0 is a double root equation
theorem part1 : is_double_root_eq 1 (-3) 2 :=
sorry

-- Statement for part 2: finding correct values of a and b for ax^2 + bx - 6 = 0 to be a double root equation with one root 2
theorem part2 : (∃ a b : ℝ, is_double_root_eq a b (-6) ∧ (a = -3 ∧ b = 9) ∨ (a = -3/4 ∧ b = 9/2)) :=
sorry

end NUMINAMATH_GPT_part1_part2_l155_15541


namespace NUMINAMATH_GPT_Sam_distance_l155_15565

theorem Sam_distance (miles_Marguerite: ℝ) (hours_Marguerite: ℕ) (hours_Sam: ℕ) (speed_factor: ℝ) 
  (h1: miles_Marguerite = 150) 
  (h2: hours_Marguerite = 3) 
  (h3: hours_Sam = 4)
  (h4: speed_factor = 1.2) :
  let average_speed_Marguerite := miles_Marguerite / hours_Marguerite
  let average_speed_Sam := speed_factor * average_speed_Marguerite
  let distance_Sam := average_speed_Sam * hours_Sam
  distance_Sam = 240 := 
by 
  sorry

end NUMINAMATH_GPT_Sam_distance_l155_15565


namespace NUMINAMATH_GPT_pastries_made_l155_15556

theorem pastries_made (P cakes_sold pastries_sold extra_pastries : ℕ)
  (h1 : cakes_sold = 78)
  (h2 : pastries_sold = 154)
  (h3 : extra_pastries = 76)
  (h4 : pastries_sold = cakes_sold + extra_pastries) :
  P = 154 := sorry

end NUMINAMATH_GPT_pastries_made_l155_15556


namespace NUMINAMATH_GPT_intersection_sets_l155_15521

def universal_set : Set ℝ := Set.univ
def set_A : Set ℝ := {x | (x + 2) * (x - 5) < 0}
def set_B : Set ℝ := {x | -3 < x ∧ x < 4}

theorem intersection_sets (x : ℝ) : 
  (x ∈ set_A ∩ set_B) ↔ (-2 < x ∧ x < 4) :=
by sorry

end NUMINAMATH_GPT_intersection_sets_l155_15521


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l155_15512

theorem arithmetic_sequence_sum {a b : ℤ} (h : ∀ n : ℕ, 3 + n * 6 = if n = 2 then a else if n = 3 then b else 33) : a + b = 48 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l155_15512


namespace NUMINAMATH_GPT_remaining_volume_l155_15580

-- Given
variables (a d : ℚ) 
-- Define the volumes of sections as arithmetic sequence terms
def volume (n : ℕ) := a + n*d

-- Define total volume of bottom three sections
def bottomThreeVolume := volume a 0 + volume a d + volume a (2 * d) = 4

-- Define total volume of top four sections
def topFourVolume := volume a (5 * d) + volume a (6 * d) + volume a (7 * d) + volume a (8 * d) = 3

-- Define the volumes of the two middle sections
def middleTwoVolume := volume a (3 * d) + volume a (4 * d) = 2 + 3 / 22

-- Prove that the total volume of the remaining two sections is 2 3/22
theorem remaining_volume : bottomThreeVolume a d ∧ topFourVolume a d → middleTwoVolume a d :=
sorry  -- Placeholder for the actual proof

end NUMINAMATH_GPT_remaining_volume_l155_15580


namespace NUMINAMATH_GPT_fred_initial_sheets_l155_15578

theorem fred_initial_sheets (X : ℕ) (h1 : X + 307 - 156 = 363) : X = 212 :=
by
  sorry

end NUMINAMATH_GPT_fred_initial_sheets_l155_15578


namespace NUMINAMATH_GPT_total_cats_l155_15500

theorem total_cats (a b c d : ℝ) (ht : a = 15.5) (hs : b = 11.6) (hg : c = 24.2) (hr : d = 18.3) :
  a + b + c + d = 69.6 :=
by
  sorry

end NUMINAMATH_GPT_total_cats_l155_15500


namespace NUMINAMATH_GPT_product_of_local_and_absolute_value_l155_15518

def localValue (n : ℕ) (digit : ℕ) : ℕ :=
  match n with
  | 564823 =>
    match digit with
    | 4 => 4000
    | _ => 0 -- only defining for digit 4 as per problem
  | _ => 0 -- only case for 564823 is considered

def absoluteValue (x : ℤ) : ℤ := if x < 0 then -x else x

theorem product_of_local_and_absolute_value:
  localValue 564823 4 * absoluteValue 4 = 16000 :=
by
  sorry

end NUMINAMATH_GPT_product_of_local_and_absolute_value_l155_15518


namespace NUMINAMATH_GPT_intersection_P_Q_l155_15514

def P : Set ℝ := { x | x^2 - 9 < 0 }
def Q : Set ℤ := { x | -1 ≤ x ∧ x ≤ 3 }

theorem intersection_P_Q : (P ∩ (coe '' Q)) = { x : ℝ | x = -1 ∨ x = 0 ∨ x = 1 ∨ x = 2 } :=
by sorry

end NUMINAMATH_GPT_intersection_P_Q_l155_15514


namespace NUMINAMATH_GPT_min_surface_area_of_sphere_l155_15539

theorem min_surface_area_of_sphere (a b c : ℝ) (volume : ℝ) (height : ℝ) 
  (h_volume : a * b * c = volume) (h_height : c = height) 
  (volume_val : volume = 12) (height_val : height = 4) : 
  ∃ r : ℝ, 4 * π * r^2 = 22 * π := 
by
  sorry

end NUMINAMATH_GPT_min_surface_area_of_sphere_l155_15539


namespace NUMINAMATH_GPT_distance_between_X_and_Y_l155_15537

theorem distance_between_X_and_Y 
  (b_walked_distance : ℕ) 
  (time_difference : ℕ) 
  (yolanda_rate : ℕ) 
  (bob_rate : ℕ) 
  (time_bob_walked : ℕ) 
  (distance_when_met : ℕ) 
  (bob_walked_8_miles : b_walked_distance = 8) 
  (one_hour_time_difference : time_difference = 1) 
  (yolanda_3_mph : yolanda_rate = 3) 
  (bob_4_mph : bob_rate = 4) 
  (time_bob_2_hours : time_bob_walked = b_walked_distance / bob_rate)
  : 
  distance_when_met = yolanda_rate * (time_bob_walked + time_difference) + bob_rate * time_bob_walked :=
by
  sorry  -- proof steps

end NUMINAMATH_GPT_distance_between_X_and_Y_l155_15537


namespace NUMINAMATH_GPT_sin_240_eq_neg_sqrt3_div_2_l155_15561

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = - Real.sqrt 3 / 2 := 
  sorry

end NUMINAMATH_GPT_sin_240_eq_neg_sqrt3_div_2_l155_15561


namespace NUMINAMATH_GPT_divisibility_by_2880_l155_15576

theorem divisibility_by_2880 (n : ℕ) : 
  (∃ t u : ℕ, (n = 16 * t - 2 ∨ n = 16 * t + 2 ∨ n = 8 * u - 1 ∨ n = 8 * u + 1) ∧ ¬(n % 3 = 0) ∧ ¬(n % 5 = 0)) ↔
  2880 ∣ (n^2 - 4) * (n^2 - 1) * (n^2 + 3) :=
sorry

end NUMINAMATH_GPT_divisibility_by_2880_l155_15576


namespace NUMINAMATH_GPT_total_people_waiting_l155_15599

theorem total_people_waiting 
  (initial_first_line : ℕ := 7)
  (left_first_line : ℕ := 4)
  (joined_first_line : ℕ := 8)
  (initial_second_line : ℕ := 12)
  (left_second_line : ℕ := 3)
  (joined_second_line : ℕ := 10)
  (initial_third_line : ℕ := 15)
  (left_third_line : ℕ := 5)
  (joined_third_line : ℕ := 7) :
  (initial_first_line - left_first_line + joined_first_line) +
  (initial_second_line - left_second_line + joined_second_line) +
  (initial_third_line - left_third_line + joined_third_line) = 47 :=
by
  sorry

end NUMINAMATH_GPT_total_people_waiting_l155_15599


namespace NUMINAMATH_GPT_equilateral_triangle_of_condition_l155_15598

theorem equilateral_triangle_of_condition (a b c : ℝ) (h : a^2 + 2 * b^2 + c^2 - 2 * b * (a + c) = 0) : a = b ∧ b = c :=
by
  /- Proof goes here -/
  sorry

end NUMINAMATH_GPT_equilateral_triangle_of_condition_l155_15598


namespace NUMINAMATH_GPT_sqrt_meaningful_range_l155_15553

theorem sqrt_meaningful_range {x : ℝ} (h : x - 1 ≥ 0) : x ≥ 1 :=
sorry

end NUMINAMATH_GPT_sqrt_meaningful_range_l155_15553


namespace NUMINAMATH_GPT_remainder_91_pow_91_mod_100_l155_15555

theorem remainder_91_pow_91_mod_100 : Nat.mod (91 ^ 91) 100 = 91 :=
by
  sorry

end NUMINAMATH_GPT_remainder_91_pow_91_mod_100_l155_15555


namespace NUMINAMATH_GPT_vines_painted_l155_15531

-- Definitions based on the conditions in the problem statement
def time_per_lily : ℕ := 5
def time_per_rose : ℕ := 7
def time_per_orchid : ℕ := 3
def time_per_vine : ℕ := 2
def total_time_spent : ℕ := 213
def lilies_painted : ℕ := 17
def roses_painted : ℕ := 10
def orchids_painted : ℕ := 6

-- The theorem to prove the number of vines painted
theorem vines_painted (vines_painted : ℕ) : 
  213 = (17 * 5) + (10 * 7) + (6 * 3) + (vines_painted * 2) → 
  vines_painted = 20 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_vines_painted_l155_15531


namespace NUMINAMATH_GPT_passing_marks_l155_15516

theorem passing_marks (T P : ℝ) 
  (h1 : 0.30 * T = P - 60) 
  (h2 : 0.45 * T = P + 30) : 
  P = 240 := 
by
  sorry

end NUMINAMATH_GPT_passing_marks_l155_15516


namespace NUMINAMATH_GPT_trigonometric_expression_l155_15511

theorem trigonometric_expression
  (α : ℝ)
  (h1 : Real.tan α = 3) : 
  (Real.sin α + 3 * Real.cos α) / (Real.cos α - 3 * Real.sin α) = -3/4 := 
by
  sorry

end NUMINAMATH_GPT_trigonometric_expression_l155_15511


namespace NUMINAMATH_GPT_machine_minutes_worked_l155_15560

theorem machine_minutes_worked {x : ℕ} 
  (h_rate : ∀ y : ℕ, 6 * y = number_of_shirts_machine_makes_yesterday)
  (h_today : 14 = number_of_shirts_machine_makes_today)
  (h_total : number_of_shirts_machine_makes_yesterday + number_of_shirts_machine_makes_today = 156) : 
  x = 23 :=
by
  sorry

end NUMINAMATH_GPT_machine_minutes_worked_l155_15560


namespace NUMINAMATH_GPT_additional_workers_needed_l155_15506

theorem additional_workers_needed :
  let initial_workers := 4
  let initial_parts := 108
  let initial_hours := 3
  let target_parts := 504
  let target_hours := 8
  (target_parts / target_hours) / (initial_parts / (initial_hours * initial_workers)) - initial_workers = 3 := by
  sorry

end NUMINAMATH_GPT_additional_workers_needed_l155_15506
