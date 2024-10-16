import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_fractions_inequality_l615_61516

theorem sum_of_fractions_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x / (y + z) + y / (z + x) + z / (x + y) ≥ (3 : ℝ) / 2 := by sorry

end NUMINAMATH_CALUDE_sum_of_fractions_inequality_l615_61516


namespace NUMINAMATH_CALUDE_target_is_largest_in_column_and_smallest_in_row_l615_61581

/-- The matrix represented as a 4x4 array of integers -/
def matrix : Matrix (Fin 4) (Fin 4) ℤ :=
  ![![5, -2, 3, 7],
    ![8, 0, 2, -1],
    ![1, -3, 6, 0],
    ![9, 1, 4, 2]]

/-- The element we're proving to be both largest in column and smallest in row -/
def target_element : ℤ := 1

/-- The position of the target element in the matrix -/
def target_position : Fin 4 × Fin 4 := (3, 1)

theorem target_is_largest_in_column_and_smallest_in_row :
  (∀ i : Fin 4, matrix i (target_position.2) ≤ target_element) ∧
  (∀ j : Fin 4, target_element ≤ matrix (target_position.1) j) := by
  sorry

#check target_is_largest_in_column_and_smallest_in_row

end NUMINAMATH_CALUDE_target_is_largest_in_column_and_smallest_in_row_l615_61581


namespace NUMINAMATH_CALUDE_bacteria_growth_days_l615_61554

def initial_bacteria : ℕ := 5
def growth_rate : ℕ := 3
def target_bacteria : ℕ := 200

def bacteria_count (days : ℕ) : ℕ :=
  initial_bacteria * growth_rate ^ days

theorem bacteria_growth_days :
  (∀ k : ℕ, k < 4 → bacteria_count k ≤ target_bacteria) ∧
  bacteria_count 4 > target_bacteria :=
sorry

end NUMINAMATH_CALUDE_bacteria_growth_days_l615_61554


namespace NUMINAMATH_CALUDE_petrol_expenses_l615_61540

def monthly_salary : ℕ := 23000
def savings_percentage : ℚ := 1/10
def savings : ℕ := 2300
def known_expenses : ℕ := 18700

theorem petrol_expenses : 
  monthly_salary * savings_percentage = savings →
  monthly_salary - savings - known_expenses = 2000 := by
sorry

end NUMINAMATH_CALUDE_petrol_expenses_l615_61540


namespace NUMINAMATH_CALUDE_soap_brand_usage_l615_61515

theorem soap_brand_usage (total : ℕ) (neither : ℕ) (both : ℕ) :
  total = 180 →
  neither = 80 →
  both = 10 →
  ∃ (only_A only_B : ℕ),
    total = only_A + only_B + both + neither ∧
    only_B = 3 * both ∧
    only_A = 60 := by
  sorry

end NUMINAMATH_CALUDE_soap_brand_usage_l615_61515


namespace NUMINAMATH_CALUDE_tan_sum_equals_three_l615_61574

theorem tan_sum_equals_three (α β : Real) 
  (h1 : α + β = π/3)
  (h2 : Real.sin α * Real.sin β = (Real.sqrt 3 - 3)/6) :
  Real.tan α + Real.tan β = 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_equals_three_l615_61574


namespace NUMINAMATH_CALUDE_bug_path_length_l615_61523

theorem bug_path_length (a b c : ℝ) (h1 : a = 120) (h2 : b = 90) (h3 : c = 150) : 
  ∃ (d : ℝ), (a^2 + b^2 = c^2) ∧ (c + c + d = 390) ∧ (d = a ∨ d = b) :=
sorry

end NUMINAMATH_CALUDE_bug_path_length_l615_61523


namespace NUMINAMATH_CALUDE_coefficient_x5_in_binomial_expansion_l615_61546

theorem coefficient_x5_in_binomial_expansion :
  (Finset.range 9).sum (fun k => (Nat.choose 8 k) * (1 ^ (8 - k)) * (1 ^ k)) = 256 ∧
  (Finset.range 9).sum (fun k => if k = 3 then (Nat.choose 8 k) else 0) = 56 :=
by sorry

end NUMINAMATH_CALUDE_coefficient_x5_in_binomial_expansion_l615_61546


namespace NUMINAMATH_CALUDE_real_part_of_z_l615_61564

theorem real_part_of_z (z : ℂ) (h : (3 + 4*I)*z = 5*(1 - I)) : 
  z.re = -1/5 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_z_l615_61564


namespace NUMINAMATH_CALUDE_solve_for_B_l615_61598

theorem solve_for_B : ∃ B : ℝ, (4 * B + 4 - 3 = 33) ∧ (B = 8) := by sorry

end NUMINAMATH_CALUDE_solve_for_B_l615_61598


namespace NUMINAMATH_CALUDE_zoo_visitors_l615_61573

/-- The number of adults who went to the zoo on Monday -/
def adults_monday : ℕ := sorry

/-- The theorem stating the number of adults who went to the zoo on Monday -/
theorem zoo_visitors :
  (7 * 3 + adults_monday * 4) + (4 * 3 + 2 * 4) = 61 →
  adults_monday = 5 := by
  sorry

end NUMINAMATH_CALUDE_zoo_visitors_l615_61573


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l615_61597

theorem algebraic_expression_value (a b : ℝ) (h : a - 3 * b = 0) :
  (a - (2 * a * b - b^2) / a) / ((a^2 - b^2) / a) = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l615_61597


namespace NUMINAMATH_CALUDE_basketball_team_combinations_l615_61577

theorem basketball_team_combinations :
  let total_players : ℕ := 15
  let team_size : ℕ := 6
  let must_include : ℕ := 2
  let remaining_slots : ℕ := team_size - must_include
  let remaining_players : ℕ := total_players - must_include
  Nat.choose remaining_players remaining_slots = 715 :=
by sorry

end NUMINAMATH_CALUDE_basketball_team_combinations_l615_61577


namespace NUMINAMATH_CALUDE_expected_closest_distance_five_points_l615_61551

/-- The expected distance between the closest pair of points when five points are chosen uniformly at random on a segment of length 1 -/
theorem expected_closest_distance_five_points (segment_length : ℝ) 
  (h_segment : segment_length = 1) : ℝ :=
by
  sorry

end NUMINAMATH_CALUDE_expected_closest_distance_five_points_l615_61551


namespace NUMINAMATH_CALUDE_conference_duration_l615_61571

def minutes_in_hour : ℕ := 60

def day1_hours : ℕ := 7
def day1_minutes : ℕ := 15

def day2_hours : ℕ := 8
def day2_minutes : ℕ := 45

def total_conference_minutes : ℕ := 
  (day1_hours * minutes_in_hour + day1_minutes) +
  (day2_hours * minutes_in_hour + day2_minutes)

theorem conference_duration :
  total_conference_minutes = 960 := by
  sorry

end NUMINAMATH_CALUDE_conference_duration_l615_61571


namespace NUMINAMATH_CALUDE_unique_number_satisfying_conditions_l615_61586

/-- Checks if three numbers form a geometric progression -/
def is_geometric_progression (a b c : ℕ) : Prop :=
  b * b = a * c

/-- Checks if three numbers form an arithmetic progression -/
def is_arithmetic_progression (a b c : ℕ) : Prop :=
  b - a = c - b

/-- Reverses a three-digit number -/
def reverse_number (n : ℕ) : ℕ :=
  (n % 10) * 100 + ((n / 10) % 10) * 10 + (n / 100)

theorem unique_number_satisfying_conditions : ∃! n : ℕ,
  100 ≤ n ∧ n < 1000 ∧
  is_geometric_progression (n / 100) ((n / 10) % 10) (n % 10) ∧
  n - 792 = reverse_number n ∧
  is_arithmetic_progression ((n / 100) - 4) ((n / 10) % 10) (n % 10) ∧
  n = 931 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_satisfying_conditions_l615_61586


namespace NUMINAMATH_CALUDE_elberta_amount_l615_61543

/-- The amount of money Granny Smith has -/
def granny_smith : ℚ := 75

/-- The amount of money Anjou has -/
def anjou : ℚ := granny_smith / 4

/-- The amount of money Elberta has -/
def elberta : ℚ := anjou + 3

/-- Theorem stating that Elberta has $21.75 -/
theorem elberta_amount : elberta = 21.75 := by
  sorry

end NUMINAMATH_CALUDE_elberta_amount_l615_61543


namespace NUMINAMATH_CALUDE_word_game_possible_l615_61501

structure WordDistribution where
  anya_only : ℕ
  borya_only : ℕ
  vasya_only : ℕ
  anya_borya : ℕ
  anya_vasya : ℕ
  borya_vasya : ℕ

def total_words (d : WordDistribution) : ℕ :=
  d.anya_only + d.borya_only + d.vasya_only + d.anya_borya + d.anya_vasya + d.borya_vasya

def anya_words (d : WordDistribution) : ℕ :=
  d.anya_only + d.anya_borya + d.anya_vasya

def borya_words (d : WordDistribution) : ℕ :=
  d.borya_only + d.anya_borya + d.borya_vasya

def vasya_words (d : WordDistribution) : ℕ :=
  d.vasya_only + d.anya_vasya + d.borya_vasya

def anya_score (d : WordDistribution) : ℕ :=
  2 * d.anya_only + d.anya_borya + d.anya_vasya

def borya_score (d : WordDistribution) : ℕ :=
  2 * d.borya_only + d.anya_borya + d.borya_vasya

def vasya_score (d : WordDistribution) : ℕ :=
  2 * d.vasya_only + d.anya_vasya + d.borya_vasya

theorem word_game_possible : ∃ d : WordDistribution,
  anya_words d > borya_words d ∧
  borya_words d > vasya_words d ∧
  vasya_score d > borya_score d ∧
  borya_score d > anya_score d :=
sorry

end NUMINAMATH_CALUDE_word_game_possible_l615_61501


namespace NUMINAMATH_CALUDE_kims_test_probability_l615_61504

theorem kims_test_probability (p_english : ℝ) (p_history : ℝ) 
  (h_english : p_english = 5/9)
  (h_history : p_history = 1/3)
  (h_independent : True) -- We don't need to explicitly define independence in this statement
  : (1 - p_english) * p_history = 4/27 := by
  sorry

end NUMINAMATH_CALUDE_kims_test_probability_l615_61504


namespace NUMINAMATH_CALUDE_cube_volume_scaling_l615_61526

theorem cube_volume_scaling (V : ℝ) (V_pos : V > 0) :
  let original_side := V ^ (1/3)
  let new_side := 2 * original_side
  let new_volume := new_side ^ 3
  new_volume = 8 * V := by sorry

end NUMINAMATH_CALUDE_cube_volume_scaling_l615_61526


namespace NUMINAMATH_CALUDE_milk_for_six_cookies_l615_61528

/-- Represents the number of cups of milk required for a given number of cookies -/
def milkRequired (cookies : ℕ) : ℚ :=
  sorry

theorem milk_for_six_cookies :
  let cookies_per_quart : ℕ := 24 / 4
  let pints_per_quart : ℕ := 2
  let cups_per_pint : ℕ := 2
  milkRequired 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_milk_for_six_cookies_l615_61528


namespace NUMINAMATH_CALUDE_gunny_bag_capacity_l615_61502

/-- The capacity of a gunny bag filled with wheat packets -/
theorem gunny_bag_capacity
  (pounds_per_ton : ℕ)
  (ounces_per_pound : ℕ)
  (num_packets : ℕ)
  (packet_weight_pounds : ℕ)
  (packet_weight_ounces : ℕ)
  (h1 : pounds_per_ton = 2200)
  (h2 : ounces_per_pound = 16)
  (h3 : num_packets = 1760)
  (h4 : packet_weight_pounds = 16)
  (h5 : packet_weight_ounces = 4) :
  (num_packets * (packet_weight_pounds + packet_weight_ounces / ounces_per_pound : ℚ)) / pounds_per_ton = 13 := by
  sorry


end NUMINAMATH_CALUDE_gunny_bag_capacity_l615_61502


namespace NUMINAMATH_CALUDE_banana_permutations_proof_l615_61547

def banana_permutations : ℕ := 60

theorem banana_permutations_proof :
  let total_letters : ℕ := 6
  let b_count : ℕ := 1
  let a_count : ℕ := 3
  let n_count : ℕ := 2
  banana_permutations = (Nat.factorial total_letters) / (Nat.factorial b_count * Nat.factorial a_count * Nat.factorial n_count) :=
by sorry

end NUMINAMATH_CALUDE_banana_permutations_proof_l615_61547


namespace NUMINAMATH_CALUDE_merchant_profit_percentage_l615_61534

/-- The profit percentage for a merchant who marks up goods by 75% and then offers a 10% discount -/
theorem merchant_profit_percentage : 
  let markup_percentage : ℝ := 75
  let discount_percentage : ℝ := 10
  let cost_price : ℝ := 100
  let marked_price : ℝ := cost_price * (1 + markup_percentage / 100)
  let selling_price : ℝ := marked_price * (1 - discount_percentage / 100)
  let profit : ℝ := selling_price - cost_price
  let profit_percentage : ℝ := (profit / cost_price) * 100
  profit_percentage = 57.5 := by sorry

end NUMINAMATH_CALUDE_merchant_profit_percentage_l615_61534


namespace NUMINAMATH_CALUDE_integer_solutions_yk_eq_x2_plus_x_l615_61582

theorem integer_solutions_yk_eq_x2_plus_x (k : ℕ) (h : k > 1) :
  ∀ x y : ℤ, y^k = x^2 + x ↔ (x = 0 ∧ y = 0) ∨ (x = -1 ∧ y = 0) := by
  sorry

end NUMINAMATH_CALUDE_integer_solutions_yk_eq_x2_plus_x_l615_61582


namespace NUMINAMATH_CALUDE_jenny_sleep_duration_l615_61566

/-- Calculates the total minutes of sleep given the number of hours and minutes per hour. -/
def total_minutes_of_sleep (hours : ℕ) (minutes_per_hour : ℕ) : ℕ :=
  hours * minutes_per_hour

/-- Proves that 8 hours of sleep is equivalent to 480 minutes. -/
theorem jenny_sleep_duration :
  total_minutes_of_sleep 8 60 = 480 := by
  sorry

end NUMINAMATH_CALUDE_jenny_sleep_duration_l615_61566


namespace NUMINAMATH_CALUDE_same_heads_probability_l615_61591

def num_pennies_keiko : ℕ := 2
def num_pennies_ephraim : ℕ := 3

def total_outcomes : ℕ := 2^num_pennies_keiko * 2^num_pennies_ephraim

def favorable_outcomes : ℕ := 6

theorem same_heads_probability :
  (favorable_outcomes : ℚ) / total_outcomes = 3 / 16 := by sorry

end NUMINAMATH_CALUDE_same_heads_probability_l615_61591


namespace NUMINAMATH_CALUDE_smallest_divisible_by_10_and_24_l615_61576

theorem smallest_divisible_by_10_and_24 : ∃ n : ℕ, n > 0 ∧ n % 10 = 0 ∧ n % 24 = 0 ∧ ∀ m : ℕ, m > 0 → m % 10 = 0 → m % 24 = 0 → n ≤ m := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_10_and_24_l615_61576


namespace NUMINAMATH_CALUDE_fraction_problem_l615_61579

theorem fraction_problem (N : ℝ) (f : ℝ) :
  N = 24 →
  N * f - 10 = 0.25 * N →
  f = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_fraction_problem_l615_61579


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l615_61518

theorem consecutive_integers_sum (n : ℚ) : 
  (n - 1) + (n + 1) + (n + 2) = 175 → n = 57 + 2/3 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l615_61518


namespace NUMINAMATH_CALUDE_gcd_6051_10085_l615_61570

theorem gcd_6051_10085 : Nat.gcd 6051 10085 = 2017 := by
  sorry

end NUMINAMATH_CALUDE_gcd_6051_10085_l615_61570


namespace NUMINAMATH_CALUDE_function_value_at_alpha_l615_61561

theorem function_value_at_alpha (α : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ Real.cos x ^ 4 + Real.sin x ^ 4
  Real.sin (2 * α) = 2 / 3 →
  f α = 7 / 9 := by
sorry

end NUMINAMATH_CALUDE_function_value_at_alpha_l615_61561


namespace NUMINAMATH_CALUDE_line_segment_ratio_l615_61575

/-- Given five points P, Q, R, S, T on a line in that order, with specified distances between them,
    prove that the ratio of PR to ST is 9/10. -/
theorem line_segment_ratio (P Q R S T : ℝ) : 
  P < Q ∧ Q < R ∧ R < S ∧ S < T →  -- Points are in order
  Q - P = 3 →                      -- PQ = 3
  R - Q = 6 →                      -- QR = 6
  S - R = 4 →                      -- RS = 4
  T - S = 10 →                     -- ST = 10
  T - P = 30 →                     -- Total distance PT = 30
  (R - P) / (T - S) = 9 / 10 :=    -- Ratio of PR to ST
by
  sorry

end NUMINAMATH_CALUDE_line_segment_ratio_l615_61575


namespace NUMINAMATH_CALUDE_even_digits_512_base5_l615_61525

/-- Converts a natural number to its base-5 representation --/
def toBase5 (n : ℕ) : List ℕ :=
  sorry

/-- Counts the number of even digits in a list of natural numbers --/
def countEvenDigits (digits : List ℕ) : ℕ :=
  sorry

/-- The number of even digits in the base-5 representation of 512 is 3 --/
theorem even_digits_512_base5 : countEvenDigits (toBase5 512) = 3 := by
  sorry

end NUMINAMATH_CALUDE_even_digits_512_base5_l615_61525


namespace NUMINAMATH_CALUDE_fraction_equality_l615_61508

theorem fraction_equality (b : ℕ+) : 
  (b : ℚ) / ((b : ℚ) + 35) = 869 / 1000 → b = 232 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l615_61508


namespace NUMINAMATH_CALUDE_adjacent_edge_angle_is_45_degrees_l615_61595

/-- A regular tetrahedron with coinciding centers of inscribed and circumscribed spheres -/
structure RegularTetrahedron where
  -- The tetrahedron is regular
  is_regular : Bool
  -- The center of the circumscribed sphere coincides with the center of the inscribed sphere
  centers_coincide : Bool

/-- The angle between two adjacent edges of a regular tetrahedron -/
def adjacent_edge_angle (t : RegularTetrahedron) : ℝ := sorry

/-- Theorem: The angle between two adjacent edges of a regular tetrahedron 
    with coinciding sphere centers is 45 degrees -/
theorem adjacent_edge_angle_is_45_degrees (t : RegularTetrahedron) 
  (h1 : t.is_regular = true) 
  (h2 : t.centers_coincide = true) : 
  adjacent_edge_angle t = 45 * (π / 180) := by sorry

end NUMINAMATH_CALUDE_adjacent_edge_angle_is_45_degrees_l615_61595


namespace NUMINAMATH_CALUDE_number_of_coverings_number_of_coverings_eq_coverings_order_invariant_l615_61505

/-- The number of coverings of a finite set -/
theorem number_of_coverings (n : ℕ) : ℕ := 
  2^(2^n - 1)

/-- The number of coverings of a finite set X with n elements is 2^(2^n - 1) -/
theorem number_of_coverings_eq (X : Finset ℕ) (h : X.card = n) :
  (Finset.powerset X).card = number_of_coverings n := by
  sorry

/-- The order of covering sets does not affect the total number of coverings -/
theorem coverings_order_invariant (X : Finset ℕ) (C₁ C₂ : Finset (Finset ℕ)) 
  (h₁ : ∀ x ∈ X, ∃ S ∈ C₁, x ∈ S) (h₂ : ∀ x ∈ X, ∃ S ∈ C₂, x ∈ S) :
  C₁.card = C₂.card := by
  sorry

end NUMINAMATH_CALUDE_number_of_coverings_number_of_coverings_eq_coverings_order_invariant_l615_61505


namespace NUMINAMATH_CALUDE_sum_of_digits_in_repeating_decimal_l615_61560

/-- The repeating decimal representation of 3/11 -/
def repeating_decimal : ℚ := 3 / 11

/-- The first digit in the repeating part of the decimal -/
def a : ℕ := 2

/-- The second digit in the repeating part of the decimal -/
def b : ℕ := 7

/-- Theorem stating that the sum of a and b is 9 -/
theorem sum_of_digits_in_repeating_decimal : a + b = 9 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_in_repeating_decimal_l615_61560


namespace NUMINAMATH_CALUDE_project_hours_difference_l615_61538

theorem project_hours_difference (total_pay : ℝ) (wage_p wage_q : ℝ) :
  total_pay = 420 ∧ 
  wage_p = wage_q * 1.5 ∧ 
  wage_p = wage_q + 7 →
  (total_pay / wage_q) - (total_pay / wage_p) = 10 := by
sorry

end NUMINAMATH_CALUDE_project_hours_difference_l615_61538


namespace NUMINAMATH_CALUDE_min_value_of_f_l615_61531

/-- The function f(x) = x^2 + 16x + 20 -/
def f (x : ℝ) : ℝ := x^2 + 16*x + 20

/-- The minimum value of f(x) is -44 -/
theorem min_value_of_f :
  ∃ (m : ℝ), (∀ x, f x ≥ m) ∧ (∃ x₀, f x₀ = m) ∧ (m = -44) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l615_61531


namespace NUMINAMATH_CALUDE_photo_voting_total_l615_61512

/-- Represents a photo voting system with applauds and boos -/
structure PhotoVoting where
  total_votes : ℕ
  applaud_ratio : ℚ
  score : ℤ

/-- Theorem: Given the conditions, the total votes cast is 300 -/
theorem photo_voting_total (pv : PhotoVoting) 
  (h1 : pv.applaud_ratio = 3/4)
  (h2 : pv.score = 150) :
  pv.total_votes = 300 := by
  sorry

end NUMINAMATH_CALUDE_photo_voting_total_l615_61512


namespace NUMINAMATH_CALUDE_triangle_count_in_square_with_inscribed_circle_l615_61541

structure SquareWithInscribedCircle where
  square : Set (ℝ × ℝ)
  circle : Set (ℝ × ℝ)
  midpoints : Set (ℝ × ℝ)
  diagonals : Set (Set (ℝ × ℝ))
  midpoint_segments : Set (Set (ℝ × ℝ))

/-- Given a square with an inscribed circle touching the midpoints of each side,
    with diagonals and segments joining midpoints of opposite sides drawn,
    the total number of triangles formed is 16. -/
theorem triangle_count_in_square_with_inscribed_circle
  (config : SquareWithInscribedCircle) : Nat :=
  16

#check triangle_count_in_square_with_inscribed_circle

end NUMINAMATH_CALUDE_triangle_count_in_square_with_inscribed_circle_l615_61541


namespace NUMINAMATH_CALUDE_value_of_A_l615_61527

/-- Given the value assignments for letters and words, prove the value of A -/
theorem value_of_A (H M A T E : ℤ)
  (h1 : H = 10)
  (h2 : M + A + T + H = 35)
  (h3 : T + E + A + M = 42)
  (h4 : M + E + E + T = 38) :
  A = 21 := by
  sorry

end NUMINAMATH_CALUDE_value_of_A_l615_61527


namespace NUMINAMATH_CALUDE_polynomial_factors_l615_61521

-- Define the polynomial P(x)
def P (x m n : ℝ) : ℝ := x^3 - m*x^2 + n*x - 42

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem polynomial_factors (m n a b : ℝ) : 
  (∃ (x : ℝ), P x m n = 0 ∧ x = -6) ∧ 
  (∃ (z : ℂ), P z.re m n = 0 ∧ z = a - b*i) ∧ 
  (b ≠ 0) →
  False :=
sorry

end NUMINAMATH_CALUDE_polynomial_factors_l615_61521


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_sum_squares_l615_61565

theorem arithmetic_geometric_mean_sum_squares (x y : ℝ) :
  (x + y) / 2 = 20 →
  Real.sqrt (x * y) = Real.sqrt 110 →
  x^2 + y^2 = 1380 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_sum_squares_l615_61565


namespace NUMINAMATH_CALUDE_mass_percentage_Cl_is_66_04_l615_61590

/-- The mass percentage of Cl in a certain compound -/
def mass_percentage_Cl : ℝ := 66.04

/-- Theorem stating that the mass percentage of Cl is 66.04% -/
theorem mass_percentage_Cl_is_66_04 : mass_percentage_Cl = 66.04 := by
  sorry

end NUMINAMATH_CALUDE_mass_percentage_Cl_is_66_04_l615_61590


namespace NUMINAMATH_CALUDE_geometric_sum_seven_terms_l615_61557

theorem geometric_sum_seven_terms : 
  let a : ℚ := 1/4  -- first term
  let r : ℚ := 1/4  -- common ratio
  let n : ℕ := 7    -- number of terms
  let S := a * (1 - r^n) / (1 - r)  -- formula for sum of geometric series
  S = 16383/49152 := by sorry

end NUMINAMATH_CALUDE_geometric_sum_seven_terms_l615_61557


namespace NUMINAMATH_CALUDE_a_fourth_plus_reciprocal_l615_61511

theorem a_fourth_plus_reciprocal (a : ℝ) (h : (a + 1/a)^2 = 5) : a^4 + 1/a^4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_a_fourth_plus_reciprocal_l615_61511


namespace NUMINAMATH_CALUDE_difference_of_place_values_l615_61583

def numeral : ℕ := 7669

def place_value (digit : ℕ) (position : ℕ) : ℕ :=
  digit * (10 ^ position)

theorem difference_of_place_values : 
  place_value 6 2 - place_value 6 1 = 540 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_place_values_l615_61583


namespace NUMINAMATH_CALUDE_cos_alpha_plus_7pi_12_l615_61558

theorem cos_alpha_plus_7pi_12 (α : ℝ) (h : Real.sin (α + π/12) = 1/3) :
  Real.cos (α + 7*π/12) = -(1 + Real.sqrt 24) / 6 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_plus_7pi_12_l615_61558


namespace NUMINAMATH_CALUDE_no_integer_solutions_l615_61594

theorem no_integer_solutions : ¬∃ (m n : ℤ), m^3 + 10*m^2 + 11*m + 2 = 81*n^3 + 27*n^2 + 3*n - 8 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l615_61594


namespace NUMINAMATH_CALUDE_ten_lines_intersection_points_l615_61559

/-- The number of intersection points of n lines in a plane, where no lines are parallel
    and exactly two lines pass through each intersection point. -/
def intersection_points (n : ℕ) : ℕ :=
  Nat.choose n 2

/-- Given 10 lines in a plane where no lines are parallel and exactly two lines pass through
    each intersection point, the number of intersection points is 45. -/
theorem ten_lines_intersection_points :
  intersection_points 10 = 45 := by
  sorry

end NUMINAMATH_CALUDE_ten_lines_intersection_points_l615_61559


namespace NUMINAMATH_CALUDE_inequality_proof_l615_61533

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a^2 + b^2 + c^2 = 1/2) : 
  (1 - a^2 + c^2) / (c * (a + 2*b)) + 
  (1 - b^2 + a^2) / (a * (b + 2*c)) + 
  (1 - c^2 + b^2) / (b * (c + 2*a)) ≥ 6 := by
sorry


end NUMINAMATH_CALUDE_inequality_proof_l615_61533


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l615_61569

theorem complex_fraction_equality : (1 + 3*Complex.I) / (Complex.I - 1) = 1 - 2*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l615_61569


namespace NUMINAMATH_CALUDE_open_box_volume_l615_61506

/-- The volume of an open box constructed from a rectangular sheet -/
def box_volume (x : ℝ) : ℝ :=
  (16 - 2*x) * (12 - 2*x) * x

/-- Theorem stating the volume of the open box -/
theorem open_box_volume (x : ℝ) : 
  box_volume x = 4*x^3 - 56*x^2 + 192*x :=
by sorry

end NUMINAMATH_CALUDE_open_box_volume_l615_61506


namespace NUMINAMATH_CALUDE_no_n_exists_for_combination_equality_l615_61599

theorem no_n_exists_for_combination_equality :
  ¬ ∃ (n : ℕ), n > 0 ∧ (Nat.choose n 3 = Nat.choose (n-1) 3 + Nat.choose (n-1) 4) := by
  sorry

end NUMINAMATH_CALUDE_no_n_exists_for_combination_equality_l615_61599


namespace NUMINAMATH_CALUDE_leaves_blown_away_proof_l615_61520

/-- The number of leaves that blew away -/
def leaves_blown_away (initial_leaves remaining_leaves : ℕ) : ℕ :=
  initial_leaves - remaining_leaves

/-- Proof that the number of leaves blown away is the difference between initial and remaining leaves -/
theorem leaves_blown_away_proof (initial_leaves remaining_leaves : ℕ) 
  (h : initial_leaves ≥ remaining_leaves) :
  leaves_blown_away initial_leaves remaining_leaves = initial_leaves - remaining_leaves :=
by
  sorry

#eval leaves_blown_away 356 112  -- Should evaluate to 244

end NUMINAMATH_CALUDE_leaves_blown_away_proof_l615_61520


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l615_61596

open Set

-- Define the universal set U as the set of real numbers
def U : Set ℝ := univ

-- Define set A
def A : Set ℝ := {x | x^2 - 2*x < 0}

-- Define set B
def B : Set ℝ := {x | x ≥ 1}

-- Theorem statement
theorem intersection_A_complement_B :
  A ∩ (U \ B) = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l615_61596


namespace NUMINAMATH_CALUDE_correct_committee_count_l615_61589

/-- Represents a department in the division of mathematical sciences --/
inductive Department
| Mathematics
| Statistics
| ComputerScience

/-- Represents the gender of a professor --/
inductive Gender
| Male
| Female

/-- Represents the composition of professors in a department --/
structure DepartmentComposition where
  department : Department
  maleCount : Nat
  femaleCount : Nat

/-- Represents the requirements for forming a committee --/
structure CommitteeRequirements where
  totalSize : Nat
  femaleCount : Nat
  maleCount : Nat
  mathDepartmentCount : Nat
  minDepartmentsRepresented : Nat

def divisionComposition : List DepartmentComposition := [
  ⟨Department.Mathematics, 3, 3⟩,
  ⟨Department.Statistics, 2, 3⟩,
  ⟨Department.ComputerScience, 2, 3⟩
]

def committeeReqs : CommitteeRequirements := {
  totalSize := 7,
  femaleCount := 4,
  maleCount := 3,
  mathDepartmentCount := 2,
  minDepartmentsRepresented := 3
}

/-- Calculates the number of possible committees given the division composition and requirements --/
def countPossibleCommittees (composition : List DepartmentComposition) (reqs : CommitteeRequirements) : Nat :=
  sorry

theorem correct_committee_count :
  countPossibleCommittees divisionComposition committeeReqs = 1050 :=
sorry

end NUMINAMATH_CALUDE_correct_committee_count_l615_61589


namespace NUMINAMATH_CALUDE_horses_equal_to_four_oxen_l615_61580

/-- The cost of animals in Rupees --/
structure AnimalCosts where
  camel : ℝ
  horse : ℝ
  ox : ℝ
  elephant : ℝ

/-- The conditions of the problem --/
def problem_conditions (costs : AnimalCosts) : Prop :=
  10 * costs.camel = 24 * costs.horse ∧
  costs.horse = 4 * costs.ox ∧
  6 * costs.ox = 4 * costs.elephant ∧
  10 * costs.elephant = 170000 ∧
  costs.camel = 4184.615384615385

/-- The theorem to prove --/
theorem horses_equal_to_four_oxen (costs : AnimalCosts) 
  (h : problem_conditions costs) : 
  costs.horse = 4 * costs.ox := by
  sorry

#check horses_equal_to_four_oxen

end NUMINAMATH_CALUDE_horses_equal_to_four_oxen_l615_61580


namespace NUMINAMATH_CALUDE_smallest_base_for_fourth_power_l615_61544

theorem smallest_base_for_fourth_power (b : ℕ) : 
  b > 0 ∧ 
  (∃ (x : ℕ), 7 * b^2 + 7 * b + 7 = x^4) ∧
  (∀ (c : ℕ), 0 < c ∧ c < b → ¬∃ (y : ℕ), 7 * c^2 + 7 * c + 7 = y^4) → 
  b = 18 := by
sorry

end NUMINAMATH_CALUDE_smallest_base_for_fourth_power_l615_61544


namespace NUMINAMATH_CALUDE_triangle_side_length_l615_61524

theorem triangle_side_length (a b c : ℝ) (h1 : a + b + c = 180) 
  (h2 : a = 120) (h3 : b = 45) (h4 : c = 15) 
  (side_b : ℝ) (h5 : side_b = 4 * Real.sqrt 6) : 
  side_b * Real.sin a / Real.sin b = 12 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l615_61524


namespace NUMINAMATH_CALUDE_sequence_general_term_l615_61536

def S (n : ℕ+) : ℚ := 2 * n.val ^ 2 + n.val

def a (n : ℕ+) : ℚ := 4 * n.val - 1

theorem sequence_general_term (n : ℕ+) : 
  (∀ k : ℕ+, S k - S (k - 1) = a k) ∧ S 1 = a 1 := by sorry

end NUMINAMATH_CALUDE_sequence_general_term_l615_61536


namespace NUMINAMATH_CALUDE_fraction_chain_l615_61587

theorem fraction_chain (a b c d e : ℝ) 
  (h1 : a / b = 5)
  (h2 : b / c = 1 / 2)
  (h3 : c / d = 3)
  (h4 : d / e = 1 / 4)
  : e / a = 8 / 15 := by
  sorry

end NUMINAMATH_CALUDE_fraction_chain_l615_61587


namespace NUMINAMATH_CALUDE_napoleon_has_17_beans_l615_61542

/-- The number of jelly beans Napoleon has -/
def napoleon_beans : ℕ := sorry

/-- The number of jelly beans Sedrich has -/
def sedrich_beans : ℕ := napoleon_beans + 4

/-- The number of jelly beans Mikey has -/
def mikey_beans : ℕ := 19

theorem napoleon_has_17_beans : napoleon_beans = 17 := by
  have h1 : sedrich_beans = napoleon_beans + 4 := rfl
  have h2 : 2 * (napoleon_beans + sedrich_beans) = 4 * mikey_beans := sorry
  have h3 : mikey_beans = 19 := rfl
  sorry

end NUMINAMATH_CALUDE_napoleon_has_17_beans_l615_61542


namespace NUMINAMATH_CALUDE_point_M_coordinates_midpoint_E_points_P₁_P₂_l615_61578

noncomputable section

-- Define the ellipse
def ellipse (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define vertices
def A (b : ℝ) : ℝ × ℝ := (0, b)
def B (b : ℝ) : ℝ × ℝ := (0, -b)
def Q (a : ℝ) : ℝ × ℝ := (a, 0)

-- Define vector operations
def vec_add (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)
def vec_scale (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)

-- Theorem statements
theorem point_M_coordinates (a b : ℝ) (h : 0 < b ∧ b < a) :
  ∃ M : ℝ × ℝ, vec_add (A b) M = vec_scale (1/2) (vec_add (vec_add (A b) (Q a)) (vec_add (A b) (B b))) →
  M = (a/2, -b/2) := sorry

theorem midpoint_E (a b k₁ k₂ : ℝ) (h : k₁ * k₂ = -b^2 / a^2) :
  ∃ C D E : ℝ × ℝ, ellipse a b C.1 C.2 ∧ ellipse a b D.1 D.2 ∧
  C.2 = k₁ * C.1 + p ∧ D.2 = k₁ * D.1 + p ∧ E.2 = k₂ * E.1 ∧ E.2 = k₁ * E.1 + p →
  E = ((C.1 + D.1) / 2, (C.2 + D.2) / 2) := sorry

theorem points_P₁_P₂ (a b : ℝ) (P P₁ P₂ : ℝ × ℝ) (h₁ : a = 10 ∧ b = 5) (h₂ : P = (-8, -1)) :
  ellipse a b P₁.1 P₁.2 ∧ ellipse a b P₂.1 P₂.2 ∧
  vec_add (vec_add P P₁) (vec_add P P₂) = vec_add P (Q a) →
  (P₁ = (-6, -4) ∧ P₂ = (8, 3)) ∨ (P₁ = (8, 3) ∧ P₂ = (-6, -4)) := sorry

end NUMINAMATH_CALUDE_point_M_coordinates_midpoint_E_points_P₁_P₂_l615_61578


namespace NUMINAMATH_CALUDE_scientific_notation_of_population_l615_61584

theorem scientific_notation_of_population (population : ℝ) : 
  population = 2184.3 * 1000000 → 
  ∃ (a : ℝ) (n : ℤ), population = a * 10^n ∧ 1 ≤ a ∧ a < 10 ∧ a = 2.1843 ∧ n = 7 :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_population_l615_61584


namespace NUMINAMATH_CALUDE_violets_to_carnations_ratio_l615_61572

/-- Represents the number of each type of flower in the shop -/
structure FlowerShop where
  violets : ℕ
  carnations : ℕ
  tulips : ℕ
  roses : ℕ

/-- The conditions of the flower shop -/
def FlowerShopConditions (shop : FlowerShop) : Prop :=
  shop.tulips = shop.violets / 4 ∧
  shop.roses = shop.tulips ∧
  shop.carnations = (2 * (shop.violets + shop.carnations + shop.tulips + shop.roses)) / 3

/-- The theorem stating the ratio of violets to carnations -/
theorem violets_to_carnations_ratio (shop : FlowerShop) 
  (h : FlowerShopConditions shop) : 
  shop.violets = shop.carnations / 3 := by
  sorry

#check violets_to_carnations_ratio

end NUMINAMATH_CALUDE_violets_to_carnations_ratio_l615_61572


namespace NUMINAMATH_CALUDE_min_value_fraction_l615_61563

theorem min_value_fraction (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (4 / a + 9 / b) ≥ 25 := by
  sorry

end NUMINAMATH_CALUDE_min_value_fraction_l615_61563


namespace NUMINAMATH_CALUDE_simplify_fraction_l615_61530

theorem simplify_fraction : (140 : ℚ) / 210 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l615_61530


namespace NUMINAMATH_CALUDE_log_inequality_l615_61556

theorem log_inequality : ∃ (a b : ℝ), 
  a = Real.log 0.8 / Real.log 0.7 ∧ 
  b = Real.log 0.9 / Real.log 1.1 ∧ 
  a > 0 ∧ 0 > b := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l615_61556


namespace NUMINAMATH_CALUDE_triangle_count_is_twenty_l615_61552

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a square with diagonals and midpoint segments -/
structure SquareWithDiagonalsAndMidpoints :=
  (vertices : Fin 4 → Point)
  (diagonals : Fin 2 → Point × Point)
  (midpoints : Fin 4 → Point)
  (cross : Point × Point)

/-- Counts the number of triangles in the figure -/
def countTriangles (square : SquareWithDiagonalsAndMidpoints) : ℕ :=
  sorry

/-- Theorem stating that the number of triangles in the figure is 20 -/
theorem triangle_count_is_twenty (square : SquareWithDiagonalsAndMidpoints) :
  countTriangles square = 20 :=
sorry

end NUMINAMATH_CALUDE_triangle_count_is_twenty_l615_61552


namespace NUMINAMATH_CALUDE_task_pages_l615_61510

/-- Represents the number of pages in the printing task -/
def P : ℕ := 480

/-- Represents the rate of Printer A in pages per minute -/
def rate_A : ℚ := P / 60

/-- Represents the rate of Printer B in pages per minute -/
def rate_B : ℚ := rate_A + 4

/-- Theorem stating that the number of pages in the task is 480 -/
theorem task_pages : P = 480 := by
  have h1 : rate_A + rate_B = P / 40 := by sorry
  have h2 : rate_A = P / 60 := by sorry
  have h3 : rate_B = rate_A + 4 := by sorry
  sorry

#check task_pages

end NUMINAMATH_CALUDE_task_pages_l615_61510


namespace NUMINAMATH_CALUDE_amoeba_count_after_10_days_l615_61522

def amoeba_count (day : ℕ) : ℕ :=
  if day = 0 then 3
  else if (day % 3 = 0) ∧ (day ≥ 3) then
    amoeba_count (day - 1)
  else
    2 * amoeba_count (day - 1)

theorem amoeba_count_after_10_days :
  amoeba_count 10 = 384 :=
sorry

end NUMINAMATH_CALUDE_amoeba_count_after_10_days_l615_61522


namespace NUMINAMATH_CALUDE_equal_savings_after_820_weeks_l615_61513

/-- Represents the number of weeks it takes for Jim and Sara to have saved the same amount -/
def weeks_to_equal_savings : ℕ :=
  820

/-- Sara's initial savings in dollars -/
def sara_initial_savings : ℕ :=
  4100

/-- Sara's weekly savings in dollars -/
def sara_weekly_savings : ℕ :=
  10

/-- Jim's weekly savings in dollars -/
def jim_weekly_savings : ℕ :=
  15

theorem equal_savings_after_820_weeks :
  sara_initial_savings + sara_weekly_savings * weeks_to_equal_savings =
  jim_weekly_savings * weeks_to_equal_savings :=
by
  sorry

#check equal_savings_after_820_weeks

end NUMINAMATH_CALUDE_equal_savings_after_820_weeks_l615_61513


namespace NUMINAMATH_CALUDE_motorcycle_price_l615_61548

theorem motorcycle_price (upfront_payment : ℝ) (upfront_percentage : ℝ) (total_price : ℝ) : 
  upfront_payment = 400 → 
  upfront_percentage = 20 → 
  upfront_payment = (upfront_percentage / 100) * total_price →
  total_price = 2000 := by
sorry

end NUMINAMATH_CALUDE_motorcycle_price_l615_61548


namespace NUMINAMATH_CALUDE_base_conversion_537_8_to_7_l615_61567

def base_8_to_10 (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds * 8^2 + tens * 8^1 + ones * 8^0

def base_10_to_7 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 7) ((m % 7) :: acc)
    aux n []

theorem base_conversion_537_8_to_7 :
  base_10_to_7 (base_8_to_10 537) = [1, 1, 0, 1, 1] := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_537_8_to_7_l615_61567


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l615_61535

theorem arithmetic_mean_of_fractions (x b c : ℝ) (hx : x ≠ 0) (hc : c ≠ 0) :
  ((x + b) / (c * x) + (x - b) / (c * x)) / 2 = 1 / c :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l615_61535


namespace NUMINAMATH_CALUDE_wooden_planks_weight_l615_61545

theorem wooden_planks_weight
  (crate_capacity : ℕ)
  (num_crates : ℕ)
  (num_nail_bags : ℕ)
  (nail_bag_weight : ℕ)
  (num_hammer_bags : ℕ)
  (hammer_bag_weight : ℕ)
  (num_plank_bags : ℕ)
  (weight_to_leave_out : ℕ)
  (h1 : crate_capacity = 20)
  (h2 : num_crates = 15)
  (h3 : num_nail_bags = 4)
  (h4 : nail_bag_weight = 5)
  (h5 : num_hammer_bags = 12)
  (h6 : hammer_bag_weight = 5)
  (h7 : num_plank_bags = 10)
  (h8 : weight_to_leave_out = 80) :
  (num_crates * crate_capacity - weight_to_leave_out
    - (num_nail_bags * nail_bag_weight + num_hammer_bags * hammer_bag_weight))
  / num_plank_bags = 14 := by
sorry

end NUMINAMATH_CALUDE_wooden_planks_weight_l615_61545


namespace NUMINAMATH_CALUDE_zero_in_interval_l615_61592

def f (x : ℝ) := 2*x + 3*x

theorem zero_in_interval :
  ∃ x ∈ Set.Ioo (-1 : ℝ) 0, f x = 0 :=
by sorry

end NUMINAMATH_CALUDE_zero_in_interval_l615_61592


namespace NUMINAMATH_CALUDE_point_P_properties_l615_61529

-- Define the point P
def P (a : ℝ) : ℝ × ℝ := (-3*a - 4, 2 + a)

-- Define the point Q
def Q : ℝ × ℝ := (5, 8)

theorem point_P_properties (a : ℝ) :
  -- Case 1: P lies on x-axis
  (P a).1 = 2 ∧ (P a).2 = 0 → a = -2
  ∧
  -- Case 2: PQ is parallel to y-axis
  (P a).1 = Q.1 → a = -3
  ∧
  -- Case 3: P is in second quadrant and equidistant from axes
  (P a).1 < 0 ∧ (P a).2 > 0 ∧ |(P a).1| = |(P a).2| → 
    a = -1 ∧ (-1 : ℝ)^2023 + 2023 = 2022 :=
by sorry

end NUMINAMATH_CALUDE_point_P_properties_l615_61529


namespace NUMINAMATH_CALUDE_nonzero_terms_count_l615_61507

-- Define the polynomials
def p (x : ℝ) : ℝ := 2*x + 3
def q (x : ℝ) : ℝ := x^2 + 4*x + 5
def r (x : ℝ) : ℝ := x^3 - x^2 + 2*x + 1

-- Define the expanded expression
def expanded_expr (x : ℝ) : ℝ := p x * q x - 4 * r x

-- Theorem statement
theorem nonzero_terms_count :
  ∃ (a b c d : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
  ∀ x, expanded_expr x = a*x^3 + b*x^2 + c*x + d :=
by sorry

end NUMINAMATH_CALUDE_nonzero_terms_count_l615_61507


namespace NUMINAMATH_CALUDE_bills_equal_at_100_minutes_l615_61593

/-- United Telephone's base rate in dollars -/
def united_base : ℚ := 7

/-- United Telephone's per-minute charge in dollars -/
def united_per_minute : ℚ := 1/4

/-- Atlantic Call's base rate in dollars -/
def atlantic_base : ℚ := 12

/-- Atlantic Call's per-minute charge in dollars -/
def atlantic_per_minute : ℚ := 1/5

/-- The number of minutes at which the bills are equal -/
def equal_minutes : ℚ := 100

theorem bills_equal_at_100_minutes :
  united_base + united_per_minute * equal_minutes =
  atlantic_base + atlantic_per_minute * equal_minutes :=
sorry

end NUMINAMATH_CALUDE_bills_equal_at_100_minutes_l615_61593


namespace NUMINAMATH_CALUDE_tangent_line_x_intercept_l615_61517

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + 4*x + 5

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 + 4

-- Theorem statement
theorem tangent_line_x_intercept :
  let tangent_slope : ℝ := f' 1
  let tangent_point : ℝ × ℝ := (1, f 1)
  let x_intercept : ℝ := tangent_point.1 - tangent_point.2 / tangent_slope
  x_intercept = -3/7 := by sorry

end NUMINAMATH_CALUDE_tangent_line_x_intercept_l615_61517


namespace NUMINAMATH_CALUDE_c_value_is_one_l615_61568

/-- The quadratic function f(x) = -x^2 + cx + 12 is positive only on (-∞, -3) ∪ (4, ∞) -/
def is_positive_on_intervals (c : ℝ) : Prop :=
  ∀ x : ℝ, (-x^2 + c*x + 12 > 0) ↔ (x < -3 ∨ x > 4)

/-- The value of c for which f(x) = -x^2 + cx + 12 is positive only on (-∞, -3) ∪ (4, ∞) is 1 -/
theorem c_value_is_one :
  ∃! c : ℝ, is_positive_on_intervals c ∧ c = 1 :=
by sorry

end NUMINAMATH_CALUDE_c_value_is_one_l615_61568


namespace NUMINAMATH_CALUDE_bubble_theorem_l615_61509

/-- The number of bubbles appearing each minute -/
def k : ℕ := 36

/-- The number of minutes after which bubbles start bursting -/
def m : ℕ := 80

/-- The maximum number of bubbles on the screen -/
def max_bubbles : ℕ := k * (k + 21) / 2

theorem bubble_theorem :
  (∀ n : ℕ, n ≤ 10 + m → n * k = n * k) ∧  -- Bubbles appear every minute
  ((10 + m) * k = m * (m + 1) / 2) ∧  -- All bubbles eventually burst
  (∀ n : ℕ, n ≤ m → n * (n + 1) / 2 ≤ (10 + n) * k) ∧  -- Bursting pattern
  (k * (k + 21) / 2 = 1026) →  -- Definition of max_bubbles
  max_bubbles = 1026 := by sorry

#eval max_bubbles  -- Should output 1026

end NUMINAMATH_CALUDE_bubble_theorem_l615_61509


namespace NUMINAMATH_CALUDE_opposite_of_negative_fraction_l615_61549

theorem opposite_of_negative_fraction :
  -(-(7 : ℚ) / 3) = 7 / 3 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_fraction_l615_61549


namespace NUMINAMATH_CALUDE_hyperbola_proof_l615_61539

/-- Given hyperbola -/
def given_hyperbola (x y : ℝ) : Prop := x^2 - y^2/4 = 1

/-- Hyperbola to prove -/
def target_hyperbola (x y : ℝ) : Prop := x^2/3 - y^2/12 = 1

/-- Point that the target hyperbola passes through -/
def point : ℝ × ℝ := (2, 2)

theorem hyperbola_proof :
  (∀ x y : ℝ, given_hyperbola x y ↔ ∃ k : ℝ, x^2 - y^2/4 = k) ∧
  target_hyperbola point.1 point.2 ∧
  (∀ x y : ℝ, given_hyperbola x y ↔ target_hyperbola x y) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_proof_l615_61539


namespace NUMINAMATH_CALUDE_parentheses_equivalence_l615_61532

theorem parentheses_equivalence (a b c : ℝ) : a + 2*b - 3*c = a + (2*b - 3*c) := by
  sorry

end NUMINAMATH_CALUDE_parentheses_equivalence_l615_61532


namespace NUMINAMATH_CALUDE_tan_thirteen_pi_thirds_l615_61553

theorem tan_thirteen_pi_thirds : Real.tan (13 * Real.pi / 3) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_thirteen_pi_thirds_l615_61553


namespace NUMINAMATH_CALUDE_composite_product_ratio_l615_61519

def first_six_composites : List Nat := [4, 6, 8, 9, 10, 12]
def next_six_composites : List Nat := [14, 15, 16, 18, 20, 21]

def product (l : List Nat) : Nat := l.foldl (· * ·) 1

theorem composite_product_ratio :
  (product first_six_composites : ℚ) / (product next_six_composites) = 1 / 49 := by
  sorry

end NUMINAMATH_CALUDE_composite_product_ratio_l615_61519


namespace NUMINAMATH_CALUDE_chicken_coop_max_area_l615_61585

/-- The maximum area of a rectangular chicken coop with one side against a wall --/
theorem chicken_coop_max_area :
  let wall_length : ℝ := 15
  let fence_length : ℝ := 24
  let area (x : ℝ) : ℝ := x * (fence_length - x) / 2
  let max_area : ℝ := 72
  ∀ x, 0 < x ∧ x ≤ wall_length → area x ≤ max_area :=
by sorry

end NUMINAMATH_CALUDE_chicken_coop_max_area_l615_61585


namespace NUMINAMATH_CALUDE_largest_n_with_lcm_property_l615_61550

theorem largest_n_with_lcm_property : 
  ∃ (m : ℕ+), Nat.lcm m.val 972 = 3 * m.val * Nat.gcd m.val 972 ∧ 
  ∀ (n : ℕ) (m : ℕ+), n > 972 → n < 1000 → 
    Nat.lcm m.val n ≠ 3 * m.val * Nat.gcd m.val n := by
  sorry

end NUMINAMATH_CALUDE_largest_n_with_lcm_property_l615_61550


namespace NUMINAMATH_CALUDE_triangle_side_length_l615_61500

noncomputable def f (x : ℝ) : ℝ := 
  Real.sqrt 3 * Real.sin (x / 2) * Real.cos (x / 2) - Real.cos (2 * x / 2)^2 + 1/2

theorem triangle_side_length 
  (A B C : ℝ) 
  (hA : 0 < A ∧ A < π) 
  (hB : 0 < B ∧ B < π) 
  (hC : 0 < C ∧ C < π) 
  (hABC : A + B + C = π) 
  (hf : f A = 1/2) 
  (ha : Real.sqrt 3 = (Real.sin B / Real.sin A)) 
  (hB : Real.sin B = 2 * Real.sin C) : 
  Real.sin C / Real.sin A = 1 := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l615_61500


namespace NUMINAMATH_CALUDE_ellipse_focus_distance_l615_61562

/-- Definition of the ellipse -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 16 = 1

/-- Distance from a point to a focus -/
noncomputable def distance_to_focus (x y : ℝ) (fx fy : ℝ) : ℝ :=
  Real.sqrt ((x - fx)^2 + (y - fy)^2)

/-- The statement to prove -/
theorem ellipse_focus_distance 
  (x y : ℝ) 
  (h_on_ellipse : is_on_ellipse x y) 
  (f1x f1y f2x f2y : ℝ) 
  (h_focus1 : distance_to_focus x y f1x f1y = 7) :
  distance_to_focus x y f2x f2y = 3 :=
sorry

end NUMINAMATH_CALUDE_ellipse_focus_distance_l615_61562


namespace NUMINAMATH_CALUDE_mod_equivalence_l615_61537

theorem mod_equivalence (n : ℕ) : 
  185 * 944 ≡ n [ZMOD 60] → 0 ≤ n → n < 60 → n = 40 := by
sorry

end NUMINAMATH_CALUDE_mod_equivalence_l615_61537


namespace NUMINAMATH_CALUDE_equal_squares_sum_l615_61555

theorem equal_squares_sum (a b c : ℝ) :
  a^2 + b^2 + c^2 - a*b - b*c - a*c = 0 → a = b ∧ b = c := by
  sorry

end NUMINAMATH_CALUDE_equal_squares_sum_l615_61555


namespace NUMINAMATH_CALUDE_termite_ridden_homes_l615_61514

theorem termite_ridden_homes (total_homes : ℝ) (termite_ridden_homes : ℝ) 
  (h1 : termite_ridden_homes > 0)
  (h2 : (4 : ℝ) / 7 * termite_ridden_homes = termite_ridden_homes - (1 : ℝ) / 7 * total_homes) :
  termite_ridden_homes = (1 : ℝ) / 3 * total_homes := by
sorry

end NUMINAMATH_CALUDE_termite_ridden_homes_l615_61514


namespace NUMINAMATH_CALUDE_sum_of_digits_l615_61588

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def is_single_digit (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 9

def divisible_by_11 (n : ℕ) : Prop := ∃ k : ℕ, n = 11 * k

theorem sum_of_digits (a b : ℕ) : 
  is_single_digit a → 
  is_single_digit b → 
  is_three_digit (700 + 10 * a + 1) →
  is_three_digit (100 * b + 60 + 5) →
  (700 + 10 * a + 1) + 184 = (100 * b + 60 + 5) →
  divisible_by_11 (100 * b + 60 + 5) →
  a + b = 9 := by
sorry

end NUMINAMATH_CALUDE_sum_of_digits_l615_61588


namespace NUMINAMATH_CALUDE_age_difference_proof_l615_61503

/-- The age difference between Mandy and Sarah --/
def age_difference : ℕ := by sorry

theorem age_difference_proof (mandy_age tom_age julia_age max_age sarah_age : ℕ) 
  (h1 : mandy_age = 3)
  (h2 : tom_age = 4 * mandy_age)
  (h3 : julia_age = tom_age - 5)
  (h4 : max_age = 2 * julia_age)
  (h5 : sarah_age = 3 * max_age - 1) :
  sarah_age - mandy_age = age_difference := by sorry

end NUMINAMATH_CALUDE_age_difference_proof_l615_61503
