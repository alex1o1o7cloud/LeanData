import Mathlib

namespace NUMINAMATH_GPT_modified_full_house_probability_l1944_194467

def total_choices : ℕ := Nat.choose 52 6

def ways_rank1 : ℕ := 13
def ways_3_cards : ℕ := Nat.choose 4 3
def ways_rank2 : ℕ := 12
def ways_2_cards : ℕ := Nat.choose 4 2
def ways_additional_card : ℕ := 11 * 4

def ways_modified_full_house : ℕ := ways_rank1 * ways_3_cards * ways_rank2 * ways_2_cards * ways_additional_card

def probability_modified_full_house : ℚ := ways_modified_full_house / total_choices

theorem modified_full_house_probability : probability_modified_full_house = 24 / 2977 := 
by sorry

end NUMINAMATH_GPT_modified_full_house_probability_l1944_194467


namespace NUMINAMATH_GPT_find_a_and_b_find_monotonic_intervals_and_extreme_values_l1944_194415

-- Definitions and conditions
def f (x : ℝ) (a b c : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

def takes_extreme_values (f : ℝ → ℝ) (a b c : ℝ) : Prop := 
  ∃ x₁ x₂, x₁ = 1 ∧ x₂ = -2/3 ∧ 3*x₁^2 + 2*a*x₁ + b = 0 ∧ 3*x₂^2 + 2*a*x₂ + b = 0

def f_at_specific_point (f : ℝ → ℝ) (x v : ℝ) : Prop :=
  f x = v

theorem find_a_and_b (a b c : ℝ) :
  takes_extreme_values (f a b c) a b c →
  a = -1/2 ∧ b = -2 :=
sorry

theorem find_monotonic_intervals_and_extreme_values (a b c : ℝ) :
  takes_extreme_values (f a b c) a b c →
  f_at_specific_point (f a b c) (-1) (3/2) →
  c = 1 ∧ 
  (∀ x, x < -2/3 ∨ x > 1 → deriv (f a b c) x > 0) ∧
  (∀ x, -2/3 < x ∧ x < 1 → deriv (f a b c) x < 0) ∧
  f a b c (-2/3) = 49/27 ∧ 
  f a b c 1 = -1/2 :=
sorry

end NUMINAMATH_GPT_find_a_and_b_find_monotonic_intervals_and_extreme_values_l1944_194415


namespace NUMINAMATH_GPT_multiplication_difference_is_1242_l1944_194494

theorem multiplication_difference_is_1242 (a b c : ℕ) (h1 : a = 138) (h2 : b = 43) (h3 : c = 34) :
  a * b - a * c = 1242 :=
by
  sorry

end NUMINAMATH_GPT_multiplication_difference_is_1242_l1944_194494


namespace NUMINAMATH_GPT_largest_final_number_l1944_194477

-- Define the sequence and conditions
def initial_number := List.replicate 40 [3, 1, 1, 2, 3] |> List.join

-- The transformation rule
def valid_transform (a b : ℕ) : ℕ := if a + b <= 9 then a + b else 0

-- Sum of digits of a number
def sum_digits : List ℕ → ℕ := List.foldr (· + ·) 0

-- Define the final valid number pattern
def valid_final_pattern (n : ℕ) : Prop := n = 77

-- The main theorem statement
theorem largest_final_number (seq : List ℕ) (h_seq : seq = initial_number) :
  valid_final_pattern (sum_digits seq) := sorry

end NUMINAMATH_GPT_largest_final_number_l1944_194477


namespace NUMINAMATH_GPT_radius_of_circumscribed_sphere_l1944_194410

-- Condition: SA = 2
def SA : ℝ := 2

-- Condition: SB = 4
def SB : ℝ := 4

-- Condition: SC = 4
def SC : ℝ := 4

-- Condition: The three side edges are pairwise perpendicular.
def pairwise_perpendicular : Prop := true -- This condition is described but would require geometric definition.

-- To prove: Radius of circumscribed sphere is 3
theorem radius_of_circumscribed_sphere : 
  ∀ (SA SB SC : ℝ) (pairwise_perpendicular : Prop), SA = 2 → SB = 4 → SC = 4 → pairwise_perpendicular → 
  (3 : ℝ) = 3 := by 
  intros SA SB SC pairwise_perpendicular h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_radius_of_circumscribed_sphere_l1944_194410


namespace NUMINAMATH_GPT_num_games_last_year_l1944_194487

-- Definitions from conditions
def num_games_this_year : ℕ := 14
def total_num_games : ℕ := 43

-- Theorem to prove
theorem num_games_last_year (num_games_last_year : ℕ) : 
  total_num_games - num_games_this_year = num_games_last_year ↔ num_games_last_year = 29 :=
by
  sorry

end NUMINAMATH_GPT_num_games_last_year_l1944_194487


namespace NUMINAMATH_GPT_lcm_second_factor_l1944_194474

theorem lcm_second_factor (A B : ℕ) (hcf : ℕ) (f1 f2 : ℕ) 
  (h₁ : hcf = 25) 
  (h₂ : A = 350) 
  (h₃ : Nat.gcd A B = hcf) 
  (h₄ : Nat.lcm A B = hcf * f1 * f2) 
  (h₅ : f1 = 13)
  : f2 = 14 := 
sorry

end NUMINAMATH_GPT_lcm_second_factor_l1944_194474


namespace NUMINAMATH_GPT_equation_of_rotated_translated_line_l1944_194403

theorem equation_of_rotated_translated_line (x y : ℝ) :
  (∀ x, y = 3 * x → y = x / -3 + 1 / -3) →
  (∀ x, y = -1/3 * (x - 1)) →
  y = -1/3 * x + 1/3 :=
sorry

end NUMINAMATH_GPT_equation_of_rotated_translated_line_l1944_194403


namespace NUMINAMATH_GPT_perpendicular_slope_l1944_194456

-- Conditions
def slope_of_given_line : ℚ := 5 / 2

-- The statement
theorem perpendicular_slope (slope_of_given_line : ℚ) : (-1 / slope_of_given_line = -2 / 5) :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_slope_l1944_194456


namespace NUMINAMATH_GPT_smallest_M_satisfying_conditions_l1944_194419

theorem smallest_M_satisfying_conditions :
  ∃ M : ℕ, M > 0 ∧ M = 250 ∧
    ( (M % 125 = 0 ∧ ((M + 1) % 8 = 0 ∧ (M + 2) % 9 = 0) ∨ ((M + 1) % 9 = 0 ∧ (M + 2) % 8 = 0)) ∨
      (M % 8 = 0 ∧ ((M + 1) % 125 = 0 ∧ (M + 2) % 9 = 0) ∨ ((M + 1) % 9 = 0 ∧ (M + 2) % 125 = 0)) ∨
      (M % 9 = 0 ∧ ((M + 1) % 8 = 0 ∧ (M + 2) % 125 = 0) ∨ ((M + 1) % 125 = 0 ∧ (M + 2) % 8 = 0)) ) :=
by
  sorry

end NUMINAMATH_GPT_smallest_M_satisfying_conditions_l1944_194419


namespace NUMINAMATH_GPT_socks_expected_value_l1944_194469

noncomputable def expected_socks_pairs (p : ℕ) : ℕ :=
2 * p

theorem socks_expected_value (p : ℕ) : 
  expected_socks_pairs p = 2 * p := 
by sorry

end NUMINAMATH_GPT_socks_expected_value_l1944_194469


namespace NUMINAMATH_GPT_sugar_amount_l1944_194461

noncomputable def mixed_to_improper (a : ℕ) (b c : ℕ) : ℚ :=
  a + b / c

theorem sugar_amount (a : ℚ) (h : a = mixed_to_improper 7 3 4) : 1 / 3 * a = 2 + 7 / 12 :=
by
  rw [h]
  simp
  sorry

end NUMINAMATH_GPT_sugar_amount_l1944_194461


namespace NUMINAMATH_GPT_correct_option_l1944_194411

def monomial_structure_same (m1 m2 : ℕ → ℕ) : Prop :=
  ∀ i, m1 i = m2 i

def monomial1 : ℕ → ℕ
| 0 => 1 -- Exponent of a in 3ab^2
| 1 => 2 -- Exponent of b in 3ab^2
| _ => 0

def monomial2 : ℕ → ℕ
| 0 => 1 -- Exponent of a in 4ab^2
| 1 => 2 -- Exponent of b in 4ab^2
| _ => 0

theorem correct_option :
  monomial_structure_same monomial1 monomial2 := sorry

end NUMINAMATH_GPT_correct_option_l1944_194411


namespace NUMINAMATH_GPT_bicycle_total_distance_l1944_194457

noncomputable def front_wheel_circumference : ℚ := 4/3
noncomputable def rear_wheel_circumference : ℚ := 3/2
noncomputable def extra_revolutions : ℕ := 25

theorem bicycle_total_distance :
  (front_wheel_circumference * extra_revolutions + (rear_wheel_circumference * 
  ((front_wheel_circumference * extra_revolutions) / (rear_wheel_circumference - front_wheel_circumference))) = 300) := sorry

end NUMINAMATH_GPT_bicycle_total_distance_l1944_194457


namespace NUMINAMATH_GPT_total_games_played_l1944_194455

-- Definition of the number of teams
def num_teams : ℕ := 20

-- Definition of the number of games each pair plays
def games_per_pair : ℕ := 10

-- Theorem stating the total number of games played
theorem total_games_played : (num_teams * (num_teams - 1) / 2) * games_per_pair = 1900 :=
by sorry

end NUMINAMATH_GPT_total_games_played_l1944_194455


namespace NUMINAMATH_GPT_dividend_rate_is_16_l1944_194432

noncomputable def dividend_rate_of_shares : ℝ :=
  let share_value := 48
  let interest_rate := 0.12
  let market_value := 36.00000000000001
  (interest_rate * share_value) / market_value * 100

theorem dividend_rate_is_16 :
  dividend_rate_of_shares = 16 := by
  sorry

end NUMINAMATH_GPT_dividend_rate_is_16_l1944_194432


namespace NUMINAMATH_GPT_distribute_teachers_l1944_194499

theorem distribute_teachers :
  let math_teachers := 3
  let lang_teachers := 3 
  let schools := 2
  let teachers_each_school := 3
  let distribution_plans := 
    (math_teachers.choose 2) * (lang_teachers.choose 1) + 
    (math_teachers.choose 1) * (lang_teachers.choose 2)
  distribution_plans = 18 := 
by
  sorry

end NUMINAMATH_GPT_distribute_teachers_l1944_194499


namespace NUMINAMATH_GPT_reciprocal_square_inequality_l1944_194433

variable (x y : ℝ)
variable (hx_pos : 0 < x) (hy_pos : 0 < y) (hxy : x ≤ y)

theorem reciprocal_square_inequality :
  (1 / y^2) ≤ (1 / x^2) :=
sorry

end NUMINAMATH_GPT_reciprocal_square_inequality_l1944_194433


namespace NUMINAMATH_GPT_reciprocals_expression_value_l1944_194442

theorem reciprocals_expression_value (a b : ℝ) (h : a * b = 1) : a^2 * b - (a - 2023) = 2023 := 
by 
  sorry

end NUMINAMATH_GPT_reciprocals_expression_value_l1944_194442


namespace NUMINAMATH_GPT_reciprocal_of_negative_2023_l1944_194440

theorem reciprocal_of_negative_2023 : (-2023) * (-1 / 2023) = 1 :=
by
  sorry

end NUMINAMATH_GPT_reciprocal_of_negative_2023_l1944_194440


namespace NUMINAMATH_GPT_towel_decrease_percentage_l1944_194488

variable (L B : ℝ)
variable (h1 : 0.70 * L = L - (0.30 * L))
variable (h2 : 0.60 * B = B - (0.40 * B))

theorem towel_decrease_percentage (L B : ℝ) 
  (h1 : 0.70 * L = L - (0.30 * L))
  (h2 : 0.60 * B = B - (0.40 * B)) :
  ((L * B - (0.70 * L) * (0.60 * B)) / (L * B)) * 100 = 58 := 
by
  sorry

end NUMINAMATH_GPT_towel_decrease_percentage_l1944_194488


namespace NUMINAMATH_GPT_total_CDs_in_stores_l1944_194443

def shelvesA := 5
def racksPerShelfA := 7
def cdsPerRackA := 8

def shelvesB := 4
def racksPerShelfB := 6
def cdsPerRackB := 7

def totalCDsA := shelvesA * racksPerShelfA * cdsPerRackA
def totalCDsB := shelvesB * racksPerShelfB * cdsPerRackB

def totalCDs := totalCDsA + totalCDsB

theorem total_CDs_in_stores :
  totalCDs = 448 := 
by 
  sorry

end NUMINAMATH_GPT_total_CDs_in_stores_l1944_194443


namespace NUMINAMATH_GPT_evaluate_polynomial_at_6_l1944_194435

def polynomial (x : ℝ) : ℝ := 2 * x^4 + 5 * x^3 - x^2 + 3 * x + 4

theorem evaluate_polynomial_at_6 : polynomial 6 = 3658 :=
by 
  sorry

end NUMINAMATH_GPT_evaluate_polynomial_at_6_l1944_194435


namespace NUMINAMATH_GPT_non_congruent_non_square_rectangles_count_l1944_194441

theorem non_congruent_non_square_rectangles_count :
  ∃ (S : Finset (ℕ × ℕ)), 
    (∀ (x : ℕ × ℕ), x ∈ S → 2 * (x.1 + x.2) = 80) ∧
    S.card = 19 ∧
    (∀ (x : ℕ × ℕ), x ∈ S → x.1 ≠ x.2) ∧
    (∀ (x y : ℕ × ℕ), x ∈ S → y ∈ S → x ≠ y → x.1 = y.2 → x.2 = y.1) :=
sorry

end NUMINAMATH_GPT_non_congruent_non_square_rectangles_count_l1944_194441


namespace NUMINAMATH_GPT_sum_of_proper_divisors_30_is_42_l1944_194490

def is_proper_divisor (n d : ℕ) : Prop := d ∣ n ∧ d ≠ n

-- The set of proper divisors of 30.
def proper_divisors_30 : Finset ℕ := {1, 2, 3, 5, 6, 10, 15}

-- The sum of all proper divisors of 30.
def sum_proper_divisors_30 : ℕ := proper_divisors_30.sum id

theorem sum_of_proper_divisors_30_is_42 : sum_proper_divisors_30 = 42 := 
by
  -- Proof can be filled in here
  sorry

end NUMINAMATH_GPT_sum_of_proper_divisors_30_is_42_l1944_194490


namespace NUMINAMATH_GPT_difference_face_local_value_8_l1944_194464

theorem difference_face_local_value_8 :
  let numeral := 96348621
  let digit := 8
  let face_value := digit
  let position := 3  -- 0-indexed place for thousands
  let local_value := digit * 10^position
  local_value - face_value = 7992 :=
by
  let numeral := 96348621
  let digit := 8
  let face_value := digit
  let position := 3
  let local_value := digit * 10^position
  show local_value - face_value = 7992
  sorry

end NUMINAMATH_GPT_difference_face_local_value_8_l1944_194464


namespace NUMINAMATH_GPT_M_inter_N_l1944_194437

def M : Set ℝ := {x | x^2 - x = 0}
def N : Set ℝ := {-1, 0}

theorem M_inter_N :
  M ∩ N = {0} :=
by
  sorry

end NUMINAMATH_GPT_M_inter_N_l1944_194437


namespace NUMINAMATH_GPT_derivative_at_pi_over_six_l1944_194446

-- Define the function f(x) = cos(x)
noncomputable def f (x : ℝ) : ℝ := Real.cos x

-- State the theorem: the derivative of f at π/6 is -1/2
theorem derivative_at_pi_over_six : deriv f (Real.pi / 6) = -1 / 2 :=
by sorry

end NUMINAMATH_GPT_derivative_at_pi_over_six_l1944_194446


namespace NUMINAMATH_GPT_waiters_hired_l1944_194416

theorem waiters_hired (W H : ℕ) (h1 : 3 * W = 90) (h2 : 3 * (W + H) = 126) : H = 12 :=
sorry

end NUMINAMATH_GPT_waiters_hired_l1944_194416


namespace NUMINAMATH_GPT_all_values_are_equal_l1944_194402

theorem all_values_are_equal
  (f : ℤ × ℤ → ℕ)
  (h : ∀ x y : ℤ, f (x, y) * 4 = f (x + 1, y) + f (x - 1, y) + f (x, y + 1) + f (x, y - 1))
  (hf_pos : ∀ x y : ℤ, 0 < f (x, y)) : 
  ∀ x y x' y' : ℤ, f (x, y) = f (x', y') :=
by
  sorry

end NUMINAMATH_GPT_all_values_are_equal_l1944_194402


namespace NUMINAMATH_GPT_two_digit_number_count_four_digit_number_count_l1944_194423

-- Defining the set of digits
def digits : Finset ℕ := {1, 2, 3, 4}

-- Problem 1 condition and question
def two_digit_count := Nat.choose 4 2 * 2

-- Problem 2 condition and question
def four_digit_count := Nat.choose 4 4 * 24

-- Theorem statement for Problem 1
theorem two_digit_number_count : two_digit_count = 12 :=
sorry

-- Theorem statement for Problem 2
theorem four_digit_number_count : four_digit_count = 24 :=
sorry

end NUMINAMATH_GPT_two_digit_number_count_four_digit_number_count_l1944_194423


namespace NUMINAMATH_GPT_number_of_tangent_lines_l1944_194495

def f (a x : ℝ) : ℝ := x^3 - 3 * x^2 + a

def on_line (a x y : ℝ) : Prop := 3 * x + y = a + 1

theorem number_of_tangent_lines (a m : ℝ) (h1 : on_line a m (a + 1 - 3 * m)) :
  ∃ n : ℤ, n = 1 ∨ n = 2 :=
sorry

end NUMINAMATH_GPT_number_of_tangent_lines_l1944_194495


namespace NUMINAMATH_GPT_hyperbola_standard_equation_l1944_194484

theorem hyperbola_standard_equation (a b : ℝ) :
  (∃ (P Q : ℝ × ℝ), P = (-3, 2 * Real.sqrt 7) ∧ Q = (-6 * Real.sqrt 2, -7) ∧
    (∀ x y b, y^2 / b^2 - x^2 / a^2 = 1 ∧ (2 * Real.sqrt 7)^2 / b^2 - (-3)^2 / a^2 = 1
    ∧ (-7)^2 / b^2 - (-6 * Real.sqrt 2)^2 / a^2 = 1)) →
  b^2 = 25 → a^2 = 75 →
  (∀ x y, y^2 / (25:ℝ) - x^2 / (75:ℝ) = 1) :=
sorry

end NUMINAMATH_GPT_hyperbola_standard_equation_l1944_194484


namespace NUMINAMATH_GPT_correct_calculation_l1944_194429

theorem correct_calculation (N : ℤ) (h : 41 - N = 12) : 41 + N = 70 := 
by 
  sorry

end NUMINAMATH_GPT_correct_calculation_l1944_194429


namespace NUMINAMATH_GPT_solve_system_l1944_194436

theorem solve_system (X Y : ℝ) : 
  (X + (X + 2 * Y) / (X^2 + Y^2) = 2 ∧ Y + (2 * X - Y) / (X^2 + Y^2) = 0) ↔ (X = 0 ∧ Y = 1) ∨ (X = 2 ∧ Y = -1) :=
by
  sorry

end NUMINAMATH_GPT_solve_system_l1944_194436


namespace NUMINAMATH_GPT_problem_1_problem_2_problem_3_l1944_194434

def range_1 : Set ℝ :=
  { y | ∃ x : ℝ, y = 1 / (x - 1) ∧ x ≠ 1 }

def range_2 : Set ℝ :=
  { y | ∃ x : ℝ, y = x^2 + 4 * x - 1 }

def range_3 : Set ℝ :=
  { y | ∃ x : ℝ, y = x + Real.sqrt (x + 1) ∧ x ≥ 0 }

theorem problem_1 : range_1 = {y | y < 0 ∨ y > 0} :=
by 
  sorry

theorem problem_2 : range_2 = {y | y ≥ -5} :=
by 
  sorry

theorem problem_3 : range_3 = {y | y ≥ -1} :=
by 
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_problem_3_l1944_194434


namespace NUMINAMATH_GPT_rectangle_ratio_l1944_194450

-- Given conditions
variable (w : ℕ) -- width is a natural number

-- Definitions based on conditions 
def length := 10
def perimeter := 30

-- Theorem to prove
theorem rectangle_ratio (h : 2 * length + 2 * w = perimeter) : w = 5 ∧ 1 = 1 ∧ 2 = 2 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_ratio_l1944_194450


namespace NUMINAMATH_GPT_solve_equation_l1944_194445

theorem solve_equation :
  ∀ x : ℝ, 4 * x * (6 * x - 1) = 1 - 6 * x ↔ (x = 1/6 ∨ x = -1/4) := 
by
  sorry

end NUMINAMATH_GPT_solve_equation_l1944_194445


namespace NUMINAMATH_GPT_trees_total_count_l1944_194428

theorem trees_total_count (D P : ℕ) 
  (h1 : D = 350 ∨ P = 350)
  (h2 : 300 * D + 225 * P = 217500) :
  D + P = 850 :=
by
  sorry

end NUMINAMATH_GPT_trees_total_count_l1944_194428


namespace NUMINAMATH_GPT_measure_of_angle_C_l1944_194470

variable (A B C : ℕ)

theorem measure_of_angle_C :
  (A = B - 20) →
  (C = A + 40) →
  (A + B + C = 180) →
  C = 80 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_measure_of_angle_C_l1944_194470


namespace NUMINAMATH_GPT_fractions_sum_equals_one_l1944_194412

variable {a b c x y z : ℝ}

variables (h1 : 17 * x + b * y + c * z = 0)
variables (h2 : a * x + 29 * y + c * z = 0)
variables (h3 : a * x + b * y + 53 * z = 0)
variables (ha : a ≠ 17)
variables (hx : x ≠ 0)

theorem fractions_sum_equals_one (a b c x y z : ℝ) 
  (h1 : 17 * x + b * y + c * z = 0)
  (h2 : a * x + 29 * y + c * z = 0)
  (h3 : a * x + b * y + 53 * z = 0)
  (ha : a ≠ 17)
  (hx : x ≠ 0) :
  (a / (a - 17)) + (b / (b - 29)) + (c / (c - 53)) = 1 := by 
  sorry

end NUMINAMATH_GPT_fractions_sum_equals_one_l1944_194412


namespace NUMINAMATH_GPT_perimeter_of_triangle_l1944_194486

theorem perimeter_of_triangle (r a : ℝ) (p : ℝ) (h1 : r = 3.5) (h2 : a = 56) :
  p = 32 :=
by
  sorry

end NUMINAMATH_GPT_perimeter_of_triangle_l1944_194486


namespace NUMINAMATH_GPT_sock_pairs_count_l1944_194497

theorem sock_pairs_count :
  let white_socks := 5
  let brown_socks := 3
  let blue_socks := 4
  let blue_white_pairs := blue_socks * white_socks
  let blue_brown_pairs := blue_socks * brown_socks
  let total_pairs := blue_white_pairs + blue_brown_pairs
  total_pairs = 32 :=
by
  sorry

end NUMINAMATH_GPT_sock_pairs_count_l1944_194497


namespace NUMINAMATH_GPT_proof_problem_l1944_194408

def otimes (a b : ℕ) : ℕ := (a^2 - b) / (a - b)

theorem proof_problem : otimes (otimes 7 5) 2 = 24 := by
  sorry

end NUMINAMATH_GPT_proof_problem_l1944_194408


namespace NUMINAMATH_GPT_sum_excluding_multiples_l1944_194454

theorem sum_excluding_multiples (S_total S_2 S_3 S_6 : ℕ) 
  (hS_total : S_total = (100 * (1 + 100)) / 2) 
  (hS_2 : S_2 = (50 * (2 + 100)) / 2) 
  (hS_3 : S_3 = (33 * (3 + 99)) / 2) 
  (hS_6 : S_6 = (16 * (6 + 96)) / 2) :
  S_total - S_2 - S_3 + S_6 = 1633 :=
by
  sorry

end NUMINAMATH_GPT_sum_excluding_multiples_l1944_194454


namespace NUMINAMATH_GPT_puppies_per_female_dog_l1944_194466

theorem puppies_per_female_dog
  (number_of_dogs : ℕ)
  (percent_female : ℝ)
  (fraction_female_giving_birth : ℝ)
  (remaining_puppies : ℕ)
  (donated_puppies : ℕ)
  (total_puppies : ℕ)
  (number_of_female_dogs : ℕ)
  (number_female_giving_birth : ℕ)
  (puppies_per_dog : ℕ) :
  number_of_dogs = 40 →
  percent_female = 0.60 →
  fraction_female_giving_birth = 0.75 →
  remaining_puppies = 50 →
  donated_puppies = 130 →
  total_puppies = remaining_puppies + donated_puppies →
  number_of_female_dogs = percent_female * number_of_dogs →
  number_female_giving_birth = fraction_female_giving_birth * number_of_female_dogs →
  puppies_per_dog = total_puppies / number_female_giving_birth →
  puppies_per_dog = 10 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end NUMINAMATH_GPT_puppies_per_female_dog_l1944_194466


namespace NUMINAMATH_GPT_problem_solution_l1944_194447

-- Definitions of sets A and B
def A : Set ℝ := {y | ∃ x : ℝ, y = x^2 }
def B : Set ℝ := {-2, -1, 1, 2}

-- Complement of set A in reals
def C_A : Set ℝ := {x | x < 0}

-- Lean theorem statement
theorem problem_solution : (C_A ∩ B) = {-2, -1} :=
by sorry

end NUMINAMATH_GPT_problem_solution_l1944_194447


namespace NUMINAMATH_GPT_inequality_proof_l1944_194485

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 / (a * b * (b + 1) * (c + 1))) + 
  (1 / (b * c * (c + 1) * (a + 1))) + 
  (1 / (c * a * (a + 1) * (b + 1))) ≥ 
  (3 / (1 + a * b * c)^2) :=
sorry

end NUMINAMATH_GPT_inequality_proof_l1944_194485


namespace NUMINAMATH_GPT_no_integer_solutions_l1944_194449

theorem no_integer_solutions (a b : ℤ) : ¬ (3 * a ^ 2 = b ^ 2 + 1) :=
by {
  sorry
}

end NUMINAMATH_GPT_no_integer_solutions_l1944_194449


namespace NUMINAMATH_GPT_num_pos_four_digit_integers_l1944_194417

theorem num_pos_four_digit_integers : 
  ∃ (n : ℕ), n = (Nat.factorial 4) / ((Nat.factorial 3) * (Nat.factorial 1)) ∧ n = 4 := 
by
  sorry

end NUMINAMATH_GPT_num_pos_four_digit_integers_l1944_194417


namespace NUMINAMATH_GPT_store_credit_percentage_l1944_194482

theorem store_credit_percentage (SN NES cash_given change_back game_value : ℕ) (P : ℚ)
  (hSN : SN = 150)
  (hNES : NES = 160)
  (hcash_given : cash_given = 80)
  (hchange_back : change_back = 10)
  (hgame_value : game_value = 30)
  (hP_def : NES = P * SN + (cash_given - change_back) + game_value) :
  P = 0.4 :=
  sorry

end NUMINAMATH_GPT_store_credit_percentage_l1944_194482


namespace NUMINAMATH_GPT_min_value_reciprocals_l1944_194427

open Real

theorem min_value_reciprocals (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : a + 3 * b = 1) :
  ∃ m : ℝ, (∀ x y : ℝ, 0 < x → 0 < y → x + 3 * y = 1 → (1 / x + 1 / y) ≥ m) ∧ m = 4 + 2 * sqrt 3 :=
sorry

end NUMINAMATH_GPT_min_value_reciprocals_l1944_194427


namespace NUMINAMATH_GPT_merchant_profit_percentage_l1944_194418

theorem merchant_profit_percentage 
    (cost_price : ℝ) 
    (markup_percentage : ℝ) 
    (discount_percentage : ℝ) 
    (h1 : cost_price = 100) 
    (h2 : markup_percentage = 0.20) 
    (h3 : discount_percentage = 0.05) 
    : ((cost_price * (1 + markup_percentage) * (1 - discount_percentage) - cost_price) / cost_price * 100) = 14 := 
by 
    sorry

end NUMINAMATH_GPT_merchant_profit_percentage_l1944_194418


namespace NUMINAMATH_GPT_parabola_b_value_l1944_194465

variable (a b c p : ℝ)
variable (h1 : p ≠ 0)
variable (h2 : ∀ x, y = a*x^2 + b*x + c)
variable (h3 : vertex' y = (p, -p))
variable (h4 : y-intercept' y = (0, p))

theorem parabola_b_value : b = -4 :=
sorry

end NUMINAMATH_GPT_parabola_b_value_l1944_194465


namespace NUMINAMATH_GPT_range_of_m_if_p_and_q_true_range_of_t_if_q_necessary_for_s_l1944_194451

/-- There exists a real number x such that 2x^2 + (m-1)x + 1/2 ≤ 0 -/
def proposition_p (m : ℝ) : Prop :=
  ∃ x : ℝ, 2 * x^2 + (m - 1) * x + 1 / 2 ≤ 0

/-- The curve C1: x^2/m^2 + y^2/(2m+8) = 1 represents an ellipse with foci on the x-axis -/
def proposition_q (m : ℝ) : Prop :=
  m^2 > 2 * m + 8 ∧ 2 * m + 8 > 0

/-- The curve C2: x^2/(m-t) + y^2/(m-t-1) = 1 represents a hyperbola -/
def proposition_s (m t : ℝ) : Prop :=
  (m - t) * (m - t - 1) < 0

/-- Find the range of values for m if p and q are true -/
theorem range_of_m_if_p_and_q_true (m : ℝ) :
  proposition_p m ∧ proposition_q m ↔ (-4 < m ∧ m < -2) ∨ m > 4 :=
  sorry

/-- Find the range of values for t if q is a necessary but not sufficient condition for s -/
theorem range_of_t_if_q_necessary_for_s (m t : ℝ) :
  (∀ m, proposition_q m → proposition_s m t) ∧ ¬(proposition_s m t → proposition_q m) ↔ 
  (-4 ≤ t ∧ t ≤ -3) ∨ t ≥ 4 :=
  sorry

end NUMINAMATH_GPT_range_of_m_if_p_and_q_true_range_of_t_if_q_necessary_for_s_l1944_194451


namespace NUMINAMATH_GPT_number_of_groups_is_correct_l1944_194478

-- Defining the conditions
def new_players : Nat := 48
def returning_players : Nat := 6
def players_per_group : Nat := 6
def total_players : Nat := new_players + returning_players

-- Theorem to prove the number of groups
theorem number_of_groups_is_correct : total_players / players_per_group = 9 := by
  sorry

end NUMINAMATH_GPT_number_of_groups_is_correct_l1944_194478


namespace NUMINAMATH_GPT_circle_equation_l1944_194404

def circle_center : (ℝ × ℝ) := (1, 2)
def radius : ℝ := 3

theorem circle_equation : 
  (∀ x y : ℝ, (x - circle_center.1) ^ 2 + (y - circle_center.2) ^ 2 = radius ^ 2 ↔ 
  (x - 1) ^ 2 + (y - 2) ^ 2 = 9) := 
by
  sorry

end NUMINAMATH_GPT_circle_equation_l1944_194404


namespace NUMINAMATH_GPT_range_of_g_l1944_194492

noncomputable def g (x : ℝ) : ℝ :=
  (Real.arccos (x / 3))^2 + 2 * Real.pi * Real.arcsin (x / 3) -
  (Real.arcsin (x / 3))^2 + (Real.pi^2 / 18) * (x^2 + 12 * x + 27)

lemma arccos_arcsin_identity (x : ℝ) (h : -1 ≤ x ∧ x ≤ 1) : 
  Real.arccos x + Real.arcsin x = Real.pi / 2 := sorry

theorem range_of_g : ∀ (x : ℝ), -3 ≤ x ∧ x ≤ 3 → ∃ y : ℝ, g x = y ∧ y ∈ Set.Icc (Real.pi^2 / 4) (5 * Real.pi^2 / 2) :=
sorry

end NUMINAMATH_GPT_range_of_g_l1944_194492


namespace NUMINAMATH_GPT_courtyard_length_proof_l1944_194425

noncomputable def paving_stone_area (length width : ℝ) : ℝ := length * width

noncomputable def total_area_stones (stone_area : ℝ) (num_stones : ℝ) : ℝ := stone_area * num_stones

noncomputable def courtyard_length (total_area width : ℝ) : ℝ := total_area / width

theorem courtyard_length_proof :
  let stone_length := 2.5
  let stone_width := 2
  let courtyard_width := 16.5
  let num_stones := 99
  let stone_area := paving_stone_area stone_length stone_width
  let total_area := total_area_stones stone_area num_stones
  courtyard_length total_area courtyard_width = 30 :=
by
  sorry

end NUMINAMATH_GPT_courtyard_length_proof_l1944_194425


namespace NUMINAMATH_GPT_odd_function_value_l1944_194409

theorem odd_function_value (f : ℝ → ℝ)
  (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_fx : ∀ x : ℝ, x ≤ 0 → f x = 2 * x ^ 2 - x) :
  f 1 = -3 := 
sorry

end NUMINAMATH_GPT_odd_function_value_l1944_194409


namespace NUMINAMATH_GPT_largest_digit_B_divisible_by_3_l1944_194407

-- Define the six-digit number form and the known digits sum.
def isIntegerDivisibleBy3 (b : ℕ) : Prop :=
  b < 10 ∧ (b + 30) % 3 = 0

-- The main theorem: Find the largest digit B such that the number 4B5,894 is divisible by 3.
theorem largest_digit_B_divisible_by_3 : ∃ (B : ℕ), isIntegerDivisibleBy3 B ∧ ∀ (b' : ℕ), isIntegerDivisibleBy3 b' → b' ≤ B := by
  -- Notice the existential and universal quantifiers involved in finding the largest B.
  sorry

end NUMINAMATH_GPT_largest_digit_B_divisible_by_3_l1944_194407


namespace NUMINAMATH_GPT_ratio_of_selling_to_buying_l1944_194471

noncomputable def natasha_has_3_times_carla (N C : ℕ) : Prop :=
  N = 3 * C

noncomputable def carla_has_2_times_cosima (C S : ℕ) : Prop :=
  C = 2 * S

noncomputable def total_buying_price (N C S : ℕ) : ℕ :=
  N + C + S

noncomputable def total_selling_price (buying_price profit : ℕ) : ℕ :=
  buying_price + profit

theorem ratio_of_selling_to_buying (N C S buying_price selling_price ratio : ℕ) 
  (h1 : natasha_has_3_times_carla N C)
  (h2 : carla_has_2_times_cosima C S)
  (h3 : N = 60)
  (h4 : buying_price = total_buying_price N C S)
  (h5 : total_selling_price buying_price 36 = selling_price)
  (h6 : 18 * ratio = selling_price * 5): ratio = 7 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_selling_to_buying_l1944_194471


namespace NUMINAMATH_GPT_total_accessories_correct_l1944_194406

-- Definitions
def dresses_first_period := 10 * 4
def dresses_second_period := 3 * 5
def total_dresses := dresses_first_period + dresses_second_period
def accessories_per_dress := 3 + 2 + 1
def total_accessories := total_dresses * accessories_per_dress

-- Theorem statement
theorem total_accessories_correct : total_accessories = 330 := by
  sorry

end NUMINAMATH_GPT_total_accessories_correct_l1944_194406


namespace NUMINAMATH_GPT_max_min_values_of_f_l1944_194458

noncomputable def f (x : ℝ) : ℝ :=
  4^x - 2^(x+1) - 3

theorem max_min_values_of_f :
  ∀ x, 0 ≤ x ∧ x ≤ 2 → (∀ y, y = f x → y ≤ 5) ∧ (∃ y, y = f 2 ∧ y = 5) ∧ (∀ y, y = f x → y ≥ -4) ∧ (∃ y, y = f 0 ∧ y = -4) :=
by
  sorry

end NUMINAMATH_GPT_max_min_values_of_f_l1944_194458


namespace NUMINAMATH_GPT_even_function_zeros_l1944_194420

noncomputable def f (x m : ℝ) : ℝ := (x - 1) * (x + m)

theorem even_function_zeros (m : ℝ) (h : ∀ x : ℝ, f x m = f (-x) m ) : 
  m = 1 ∧ (∀ x : ℝ, f x m = 0 → (x = 1 ∨ x = -1)) := by
  sorry

end NUMINAMATH_GPT_even_function_zeros_l1944_194420


namespace NUMINAMATH_GPT_find_ABC_l1944_194414

variables (A B C D : ℕ)

-- Conditions
def non_zero_distinct_digits_less_than_7 : Prop :=
  A > 0 ∧ B > 0 ∧ C > 0 ∧ A < 7 ∧ B < 7 ∧ C < 7 ∧ A ≠ B ∧ B ≠ C ∧ A ≠ C

def ab_c_seven : Prop := 
  (A * 7 + B) + C = C * 7

def ab_ba_dc_seven : Prop :=
  (A * 7 + B) + (B * 7 + A) = D * 7 + C

-- Theorem to prove
theorem find_ABC 
  (h1 : non_zero_distinct_digits_less_than_7 A B C) 
  (h2 : ab_c_seven A B C) 
  (h3 : ab_ba_dc_seven A B C D) : 
  A * 100 + B * 10 + C = 516 :=
sorry

end NUMINAMATH_GPT_find_ABC_l1944_194414


namespace NUMINAMATH_GPT_range_of_m_l1944_194472

noncomputable def f (a x: ℝ) := 2 * x^3 - 3 * (a + 1) * x^2 + 6 * a * x

theorem range_of_m (a m x₁ x₂: ℝ) (h₁: a ∈ Set.Icc (-3) (0)) (h₂: x₁ ∈ Set.Icc (0) (2)) (h₃: x₂ ∈ Set.Icc (0) (2)) : m ∈ Set.Ici (5) → m - a * m^2 ≥ |f a x₁ - f a x₂| :=
sorry

end NUMINAMATH_GPT_range_of_m_l1944_194472


namespace NUMINAMATH_GPT_value_of_abcg_defh_l1944_194430

theorem value_of_abcg_defh
  (a b c d e f g h: ℝ)
  (h1 : a / b = 1 / 3)
  (h2 : b / c = 2)
  (h3 : c / d = 1 / 2)
  (h4 : d / e = 3)
  (h5 : e / f = 1 / 6)
  (h6 : f / g = 5 / 2)
  (h7 : g / h = 3 / 4) :
  abcg / defh = 5 / 48 :=
by
  sorry

end NUMINAMATH_GPT_value_of_abcg_defh_l1944_194430


namespace NUMINAMATH_GPT_xyz_inequality_l1944_194448

theorem xyz_inequality {x y z : ℝ} (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x * y * z = 1) :
  (x^3 / ((1 + y) * (1 + z))) + (y^3 / ((1 + z) * (1 + x))) + (z^3 / ((1 + x) * (1 + y))) ≥ (3/4) :=
sorry

end NUMINAMATH_GPT_xyz_inequality_l1944_194448


namespace NUMINAMATH_GPT_composite_numbers_with_same_main_divisors_are_equal_l1944_194463

theorem composite_numbers_with_same_main_divisors_are_equal (a b : ℕ) 
  (h_a_not_prime : ¬ Prime a)
  (h_b_not_prime : ¬ Prime b)
  (h_a_comp : 1 < a ∧ ∃ p, p ∣ a ∧ p ≠ a)
  (h_b_comp : 1 < b ∧ ∃ p, p ∣ b ∧ p ≠ b)
  (main_divisors : {d : ℕ // d ∣ a ∧ d ≠ a} = {d : ℕ // d ∣ b ∧ d ≠ b}) :
  a = b := 
sorry

end NUMINAMATH_GPT_composite_numbers_with_same_main_divisors_are_equal_l1944_194463


namespace NUMINAMATH_GPT_time_to_cover_length_l1944_194439

def escalator_rate : ℝ := 12 -- rate of the escalator in feet per second
def person_rate : ℝ := 8 -- rate of the person in feet per second
def escalator_length : ℝ := 160 -- length of the escalator in feet

theorem time_to_cover_length : escalator_length / (escalator_rate + person_rate) = 8 := by
  sorry

end NUMINAMATH_GPT_time_to_cover_length_l1944_194439


namespace NUMINAMATH_GPT_trapezium_side_length_l1944_194476

variable (length1 length2 height area : ℕ)

theorem trapezium_side_length
  (h1 : length1 = 20)
  (h2 : height = 15)
  (h3 : area = 270)
  (h4 : area = (length1 + length2) * height / 2) :
  length2 = 16 :=
by
  sorry

end NUMINAMATH_GPT_trapezium_side_length_l1944_194476


namespace NUMINAMATH_GPT_rose_tom_profit_difference_l1944_194462

def investment_months (amount: ℕ) (months: ℕ) : ℕ :=
  amount * months

def total_investment_months (john_inv: ℕ) (rose_inv: ℕ) (tom_inv: ℕ) : ℕ :=
  john_inv + rose_inv + tom_inv

def profit_share (investment: ℕ) (total_investment: ℕ) (total_profit: ℕ) : ℤ :=
  (investment * total_profit) / total_investment

theorem rose_tom_profit_difference
  (john_inv rs_per_year: ℕ := 18000 * 12)
  (rose_inv rs_per_9_months: ℕ := 12000 * 9)
  (tom_inv rs_per_8_months: ℕ := 9000 * 8)
  (total_profit: ℕ := 4070):
  profit_share rose_inv (total_investment_months john_inv rose_inv tom_inv) total_profit -
  profit_share tom_inv (total_investment_months john_inv rose_inv tom_inv) total_profit = 370 := 
by
  sorry

end NUMINAMATH_GPT_rose_tom_profit_difference_l1944_194462


namespace NUMINAMATH_GPT_toothpick_count_l1944_194473

theorem toothpick_count (length width : ℕ) (h_len : length = 20) (h_width : width = 10) : 
  2 * (length * (width + 1) + width * (length + 1)) = 430 :=
by
  sorry

end NUMINAMATH_GPT_toothpick_count_l1944_194473


namespace NUMINAMATH_GPT_total_monthly_cost_l1944_194498

theorem total_monthly_cost (volume_per_box : ℕ := 1800) 
                          (total_volume : ℕ := 1080000)
                          (cost_per_box_per_month : ℝ := 0.8) 
                          (expected_cost : ℝ := 480) : 
                          (total_volume / volume_per_box) * cost_per_box_per_month = expected_cost :=
by
  sorry

end NUMINAMATH_GPT_total_monthly_cost_l1944_194498


namespace NUMINAMATH_GPT_friend_decks_l1944_194493

noncomputable def cost_per_deck : ℕ := 8
noncomputable def victor_decks : ℕ := 6
noncomputable def total_amount_spent : ℕ := 64

theorem friend_decks :
  ∃ x : ℕ, (victor_decks * cost_per_deck) + (x * cost_per_deck) = total_amount_spent ∧ x = 2 :=
by
  sorry

end NUMINAMATH_GPT_friend_decks_l1944_194493


namespace NUMINAMATH_GPT_simplify_expression_correct_l1944_194413

-- Defining the problem conditions and required proof
def simplify_expression (x : ℝ) (h : x ≠ 2) : Prop :=
  (x / (x - 2) + 2 / (2 - x) = 1)

-- Stating the theorem
theorem simplify_expression_correct (x : ℝ) (h : x ≠ 2) : simplify_expression x h :=
  by sorry

end NUMINAMATH_GPT_simplify_expression_correct_l1944_194413


namespace NUMINAMATH_GPT_unique_number_l1944_194480

theorem unique_number (a : ℕ) (h1 : 1 < a) 
  (h2 : ∀ p : ℕ, Prime p → p ∣ a^6 - 1 → p ∣ a^3 - 1 ∨ p ∣ a^2 - 1) : a = 2 :=
by
  sorry

end NUMINAMATH_GPT_unique_number_l1944_194480


namespace NUMINAMATH_GPT_insulation_cost_per_sq_ft_l1944_194426

theorem insulation_cost_per_sq_ft 
  (l w h : ℤ) 
  (surface_area : ℤ := (2 * l * w) + (2 * l * h) + (2 * w * h))
  (total_cost : ℤ)
  (cost_per_sq_ft : ℤ := total_cost / surface_area)
  (h_l : l = 3)
  (h_w : w = 5)
  (h_h : h = 2)
  (h_total_cost : total_cost = 1240) :
  cost_per_sq_ft = 20 := 
by
  sorry

end NUMINAMATH_GPT_insulation_cost_per_sq_ft_l1944_194426


namespace NUMINAMATH_GPT_model_height_l1944_194444

noncomputable def H_actual : ℝ := 50
noncomputable def A_actual : ℝ := 25
noncomputable def A_model : ℝ := 0.025

theorem model_height : 
  let ratio := (A_actual / A_model)
  ∃ h : ℝ, h = H_actual / (Real.sqrt ratio) ∧ h = 5 * Real.sqrt 10 := 
by 
  sorry

end NUMINAMATH_GPT_model_height_l1944_194444


namespace NUMINAMATH_GPT_four_consecutive_numbers_l1944_194400

theorem four_consecutive_numbers (numbers : List ℝ) (h_distinct : numbers.Nodup) (h_length : numbers.length = 100) :
  ∃ (a b c d : ℝ) (h_seq : ([a, b, c, d] ∈ numbers.cyclicPermutations)), b + c < a + d :=
by
  sorry

end NUMINAMATH_GPT_four_consecutive_numbers_l1944_194400


namespace NUMINAMATH_GPT_william_time_on_road_l1944_194424

-- Define departure and arrival times
def departure_time := 7 -- 7:00 AM
def arrival_time := 20 -- 8:00 PM in 24-hour format

-- Define stop times in minutes
def stop1 := 25
def stop2 := 10
def stop3 := 25

-- Define total journey time in hours
def total_travel_time := arrival_time - departure_time

-- Define total stop time in hours
def total_stop_time := (stop1 + stop2 + stop3) / 60

-- Define time spent on the road
def time_on_road := total_travel_time - total_stop_time

-- The theorem to prove
theorem william_time_on_road : time_on_road = 12 := by
  sorry

end NUMINAMATH_GPT_william_time_on_road_l1944_194424


namespace NUMINAMATH_GPT_F_at_2_eq_minus_22_l1944_194489

variable (a b c d : ℝ)

def f (x : ℝ) : ℝ := a * x^7 + b * x^5 + c * x^3 + d * x

def F (x : ℝ) : ℝ := f a b c d x - 6

theorem F_at_2_eq_minus_22 (h : F a b c d (-2) = 10) : F a b c d 2 = -22 :=
by
  sorry

end NUMINAMATH_GPT_F_at_2_eq_minus_22_l1944_194489


namespace NUMINAMATH_GPT_pyramid_base_side_length_l1944_194438

theorem pyramid_base_side_length
  (area_lateral_face : ℝ)
  (slant_height : ℝ)
  (h1 : area_lateral_face = 120)
  (h2 : slant_height = 40) :
  ∃ s : ℝ, (120 = 0.5 * s * 40) ∧ s = 6 := by
  sorry

end NUMINAMATH_GPT_pyramid_base_side_length_l1944_194438


namespace NUMINAMATH_GPT_find_first_part_l1944_194483

variable (x y : ℕ)

theorem find_first_part (h₁ : x + y = 24) (h₂ : 7 * x + 5 * y = 146) : x = 13 :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_find_first_part_l1944_194483


namespace NUMINAMATH_GPT_anthony_ate_total_l1944_194401

def slices := 16

def ate_alone := 1 / slices
def shared_with_ben := (1 / 2) * (1 / slices)
def shared_with_chris := (1 / 2) * (1 / slices)

theorem anthony_ate_total :
  ate_alone + shared_with_ben + shared_with_chris = 1 / 8 :=
by
  sorry

end NUMINAMATH_GPT_anthony_ate_total_l1944_194401


namespace NUMINAMATH_GPT_min_time_to_shoe_horses_l1944_194405

-- Definitions based on the conditions
def n_blacksmiths : ℕ := 48
def n_horses : ℕ := 60
def t_hoof : ℕ := 5 -- minutes per hoof
def n_hooves : ℕ := n_horses * 4
def total_time : ℕ := n_hooves * t_hoof
def t_min : ℕ := total_time / n_blacksmiths

-- The theorem states that the minimal time required is 25 minutes
theorem min_time_to_shoe_horses : t_min = 25 := by
  sorry

end NUMINAMATH_GPT_min_time_to_shoe_horses_l1944_194405


namespace NUMINAMATH_GPT_solve_for_x_l1944_194431

theorem solve_for_x : 
  (∃ x : ℝ, (x^2 - 6 * x + 8) / (x^2 - 9 * x + 14) = (x^2 - 3 * x - 18) / (x^2 - 4 * x - 21) 
  ∧ x = 4.5) := by
{
  sorry
}

end NUMINAMATH_GPT_solve_for_x_l1944_194431


namespace NUMINAMATH_GPT_initial_units_of_phones_l1944_194496

theorem initial_units_of_phones
  (X : ℕ) 
  (h1 : 5 = 5) 
  (h2 : X - 5 = 3 + 5 + 7) : 
  X = 20 := 
by
  sorry

end NUMINAMATH_GPT_initial_units_of_phones_l1944_194496


namespace NUMINAMATH_GPT_find_m_if_perpendicular_l1944_194491

theorem find_m_if_perpendicular 
  (m : ℝ)
  (h : ∀ m (slope1 : ℝ) (slope2 : ℝ), 
    (slope1 = -m) → 
    (slope2 = (-1) / (3 - 2 * m)) → 
    slope1 * slope2 = -1)
  : m = 3 := 
by
  sorry

end NUMINAMATH_GPT_find_m_if_perpendicular_l1944_194491


namespace NUMINAMATH_GPT_percentage_of_nine_hundred_l1944_194452

theorem percentage_of_nine_hundred : (45 * 8 = 360) ∧ ((360 / 900) * 100 = 40) :=
by
  have h1 : 45 * 8 = 360 := by sorry
  have h2 : (360 / 900) * 100 = 40 := by sorry
  exact ⟨h1, h2⟩

end NUMINAMATH_GPT_percentage_of_nine_hundred_l1944_194452


namespace NUMINAMATH_GPT_average_marks_l1944_194453

theorem average_marks (D I T : ℕ) 
  (hD : D = 90)
  (hI : I = (3 * D) / 5)
  (hT : T = 2 * I) : 
  (D + I + T) / 3 = 84 :=
by
  sorry

end NUMINAMATH_GPT_average_marks_l1944_194453


namespace NUMINAMATH_GPT_find_ab_l1944_194481
-- Import the necessary Lean libraries 

-- Define the statement for the proof problem
theorem find_ab (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 27) : ab = 9 :=
by {
    sorry
}

end NUMINAMATH_GPT_find_ab_l1944_194481


namespace NUMINAMATH_GPT_find_first_number_in_list_l1944_194460

theorem find_first_number_in_list
  (x : ℕ)
  (h1 : x < 10)
  (h2 : ∃ n : ℕ, 2012 = x + 9 * n)
  : x = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_first_number_in_list_l1944_194460


namespace NUMINAMATH_GPT_fraction_difference_in_simplest_form_l1944_194459

noncomputable def difference_fraction : ℚ := (5 / 19) - (2 / 23)

theorem fraction_difference_in_simplest_form :
  difference_fraction = 77 / 437 := by sorry

end NUMINAMATH_GPT_fraction_difference_in_simplest_form_l1944_194459


namespace NUMINAMATH_GPT_rabbits_initially_bought_l1944_194475

theorem rabbits_initially_bought (R : ℕ) (h : ∃ (k : ℕ), R + 6 = 17 * k) : R = 28 :=
sorry

end NUMINAMATH_GPT_rabbits_initially_bought_l1944_194475


namespace NUMINAMATH_GPT_min_sine_difference_l1944_194422

theorem min_sine_difference (N : ℕ) (hN : 0 < N) :
  ∃ (n k : ℕ), (1 ≤ n ∧ n ≤ N + 1) ∧ (1 ≤ k ∧ k ≤ N + 1) ∧ (n ≠ k) ∧ 
    (|Real.sin n - Real.sin k| < 2 / N) := 
sorry

end NUMINAMATH_GPT_min_sine_difference_l1944_194422


namespace NUMINAMATH_GPT_expression_value_l1944_194468

theorem expression_value
  (x y : ℝ) 
  (h : x - 3 * y = 4) : 
  (x - 3 * y)^2 + 2 * x - 6 * y - 10 = 14 :=
by
  sorry

end NUMINAMATH_GPT_expression_value_l1944_194468


namespace NUMINAMATH_GPT_numbers_with_special_remainder_property_l1944_194479

theorem numbers_with_special_remainder_property (n : ℕ) :
  (∀ q : ℕ, q > 0 → n % (q ^ 2) < (q ^ 2) / 2) ↔ (n = 1 ∨ n = 4) := 
by
  sorry

end NUMINAMATH_GPT_numbers_with_special_remainder_property_l1944_194479


namespace NUMINAMATH_GPT_magician_guarantee_success_l1944_194421

-- Definitions based on the conditions in part a).
def deck_size : ℕ := 52

def is_edge_position (position : ℕ) : Prop :=
  position = 0 ∨ position = deck_size - 1

-- Statement of the proof problem in part c).
theorem magician_guarantee_success (position : ℕ) : is_edge_position position ↔ 
  forall spectator_strategy : ℕ → ℕ, 
  exists magician_strategy : (ℕ → ℕ → ℕ), 
  forall t : ℕ, t = position →
  (∃ k : ℕ, t = magician_strategy k (spectator_strategy k)) :=
sorry

end NUMINAMATH_GPT_magician_guarantee_success_l1944_194421
