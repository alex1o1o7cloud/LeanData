import Mathlib

namespace NUMINAMATH_CALUDE_three_coin_outcomes_l3104_310437

/-- The number of possible outcomes when throwing a single coin -/
def coin_outcomes : Nat := 2

/-- The number of coins being thrown -/
def num_coins : Nat := 3

/-- Calculates the total number of outcomes when throwing multiple coins -/
def total_outcomes (n : Nat) : Nat := coin_outcomes ^ n

/-- Theorem: The number of possible outcomes when throwing three distinguishable coins is 8 -/
theorem three_coin_outcomes : total_outcomes num_coins = 8 := by
  sorry

end NUMINAMATH_CALUDE_three_coin_outcomes_l3104_310437


namespace NUMINAMATH_CALUDE_opposite_of_negative_seven_l3104_310471

theorem opposite_of_negative_seven :
  ∃ x : ℤ, ((-7 : ℤ) + x = 0) ∧ x = 7 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_seven_l3104_310471


namespace NUMINAMATH_CALUDE_unique_solution_implies_k_zero_l3104_310493

theorem unique_solution_implies_k_zero (a b k : ℤ) : 
  (∃! p : ℝ × ℝ, 
    (p.1 = a ∧ p.2 = b) ∧ 
    Real.sqrt (↑a - 1) + Real.sqrt (↑b - 1) = Real.sqrt (↑(a * b + k))) → 
  k = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_implies_k_zero_l3104_310493


namespace NUMINAMATH_CALUDE_factor_expression_l3104_310492

theorem factor_expression (x : ℝ) : 12 * x^2 - 6 * x = 6 * x * (2 * x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3104_310492


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3104_310478

theorem complex_equation_solution (z : ℂ) : z * (1 - Complex.I) = 2 → z = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3104_310478


namespace NUMINAMATH_CALUDE_equal_area_rectangles_l3104_310422

theorem equal_area_rectangles (carol_width jordan_length jordan_width : ℝ) 
  (h1 : carol_width = 15)
  (h2 : jordan_length = 9)
  (h3 : jordan_width = 20)
  (h4 : carol_width * carol_length = jordan_length * jordan_width) :
  carol_length = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_equal_area_rectangles_l3104_310422


namespace NUMINAMATH_CALUDE_range_of_a_for_positive_solutions_l3104_310431

theorem range_of_a_for_positive_solutions (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ (1/4)^x + (1/2)^(x-1) + a = 0) ↔ -3 < a ∧ a < 0 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_for_positive_solutions_l3104_310431


namespace NUMINAMATH_CALUDE_divisible_by_seven_l3104_310410

theorem divisible_by_seven (n : ℕ) : 7 ∣ (3^(2*n + 1) + 2^(n + 2)) := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_seven_l3104_310410


namespace NUMINAMATH_CALUDE_arithmetic_sequence_8th_term_l3104_310445

/-- Given an arithmetic sequence where the 4th term is 23 and the 6th term is 41, the 8th term is 59. -/
theorem arithmetic_sequence_8th_term 
  (a : ℝ) (d : ℝ) -- first term and common difference
  (h1 : a + 3*d = 23) -- 4th term condition
  (h2 : a + 5*d = 41) -- 6th term condition
  : a + 7*d = 59 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_8th_term_l3104_310445


namespace NUMINAMATH_CALUDE_matrix_inverse_proof_l3104_310400

def A : Matrix (Fin 2) (Fin 2) ℝ := !![5, -3; -2, 1]
def A_inv : Matrix (Fin 2) (Fin 2) ℝ := !![-1, -3; -2, -5]

theorem matrix_inverse_proof :
  A * A_inv = 1 ∧ A_inv * A = 1 := by sorry

end NUMINAMATH_CALUDE_matrix_inverse_proof_l3104_310400


namespace NUMINAMATH_CALUDE_factorial_sum_ratio_l3104_310474

theorem factorial_sum_ratio : 
  (1 * 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9 * 10) / 
  (1 * 2 + 3 * 4 + 5 * 6 + 7 * 8 + 9 * 10) = 19120 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_ratio_l3104_310474


namespace NUMINAMATH_CALUDE_consecutive_product_divisible_by_two_l3104_310433

theorem consecutive_product_divisible_by_two (n : ℕ) : 
  2 ∣ (n * (n + 1)) := by
sorry

end NUMINAMATH_CALUDE_consecutive_product_divisible_by_two_l3104_310433


namespace NUMINAMATH_CALUDE_division_problem_l3104_310414

theorem division_problem (total : ℕ) (p q r : ℕ) : 
  total = 1210 →
  p * 4 = q * 5 →
  q * 10 = r * 9 →
  p + q + r = total →
  r = 400 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l3104_310414


namespace NUMINAMATH_CALUDE_invest_in_good_B_l3104_310465

def expected_profit (p1 p2 p3 : ℝ) (v1 v2 v3 : ℝ) : ℝ :=
  p1 * v1 + p2 * v2 + p3 * v3

theorem invest_in_good_B (capital : ℝ) 
  (a_p1 a_p2 a_p3 : ℝ) (a_v1 a_v2 a_v3 : ℝ)
  (b_p1 b_p2 b_p3 : ℝ) (b_v1 b_v2 b_v3 : ℝ)
  (ha1 : a_p1 = 0.4) (ha2 : a_p2 = 0.3) (ha3 : a_p3 = 0.3)
  (ha4 : a_v1 = 20000) (ha5 : a_v2 = 30000) (ha6 : a_v3 = -10000)
  (hb1 : b_p1 = 0.6) (hb2 : b_p2 = 0.2) (hb3 : b_p3 = 0.2)
  (hb4 : b_v1 = 20000) (hb5 : b_v2 = 40000) (hb6 : b_v3 = -20000)
  (hcap : capital = 100000) :
  expected_profit b_p1 b_p2 b_p3 b_v1 b_v2 b_v3 > 
  expected_profit a_p1 a_p2 a_p3 a_v1 a_v2 a_v3 := by
  sorry

#check invest_in_good_B

end NUMINAMATH_CALUDE_invest_in_good_B_l3104_310465


namespace NUMINAMATH_CALUDE_skylar_donation_l3104_310485

/-- Represents the donation scenario for Skylar -/
structure DonationScenario where
  start_age : ℕ
  current_age : ℕ
  annual_donation : ℕ

/-- Calculates the total amount donated given a DonationScenario -/
def total_donated (scenario : DonationScenario) : ℕ :=
  (scenario.current_age - scenario.start_age) * scenario.annual_donation

/-- Theorem stating that Skylar's total donation is $432,000 -/
theorem skylar_donation :
  let scenario : DonationScenario := {
    start_age := 17,
    current_age := 71,
    annual_donation := 8000
  }
  total_donated scenario = 432000 := by
  sorry

end NUMINAMATH_CALUDE_skylar_donation_l3104_310485


namespace NUMINAMATH_CALUDE_vessel_capacity_proof_l3104_310415

/-- Proves that the capacity of the first vessel is 2 liters given the problem conditions -/
theorem vessel_capacity_proof (
  first_vessel_alcohol_percentage : ℝ)
  (second_vessel_capacity : ℝ)
  (second_vessel_alcohol_percentage : ℝ)
  (total_liquid_poured : ℝ)
  (new_vessel_capacity : ℝ)
  (new_mixture_alcohol_percentage : ℝ)
  (h1 : first_vessel_alcohol_percentage = 0.20)
  (h2 : second_vessel_capacity = 6)
  (h3 : second_vessel_alcohol_percentage = 0.55)
  (h4 : total_liquid_poured = 8)
  (h5 : new_vessel_capacity = 10)
  (h6 : new_mixture_alcohol_percentage = 0.37)
  : ∃ (first_vessel_capacity : ℝ),
    first_vessel_capacity = 2 ∧
    first_vessel_capacity * first_vessel_alcohol_percentage +
    second_vessel_capacity * second_vessel_alcohol_percentage =
    new_vessel_capacity * new_mixture_alcohol_percentage :=
by sorry

end NUMINAMATH_CALUDE_vessel_capacity_proof_l3104_310415


namespace NUMINAMATH_CALUDE_sum_of_fractions_l3104_310499

theorem sum_of_fractions : (1 : ℚ) / 12 + (1 : ℚ) / 15 = (3 : ℚ) / 20 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l3104_310499


namespace NUMINAMATH_CALUDE_modular_inverse_13_mod_997_l3104_310473

theorem modular_inverse_13_mod_997 :
  ∃ x : ℕ, x < 997 ∧ (13 * x) % 997 = 1 :=
by
  use 767
  sorry

end NUMINAMATH_CALUDE_modular_inverse_13_mod_997_l3104_310473


namespace NUMINAMATH_CALUDE_patio_rearrangement_l3104_310491

theorem patio_rearrangement (total_tiles : ℕ) (initial_rows : ℕ) (added_rows : ℕ) :
  total_tiles = 126 →
  initial_rows = 9 →
  added_rows = 4 →
  ∃ (initial_columns final_columns : ℕ),
    initial_columns * initial_rows = total_tiles ∧
    final_columns * (initial_rows + added_rows) = total_tiles ∧
    initial_columns - final_columns = 5 :=
by sorry

end NUMINAMATH_CALUDE_patio_rearrangement_l3104_310491


namespace NUMINAMATH_CALUDE_total_art_pieces_l3104_310426

theorem total_art_pieces (asian_art : ℕ) (egyptian_art : ℕ) 
  (h1 : asian_art = 465) (h2 : egyptian_art = 527) :
  asian_art + egyptian_art = 992 := by
  sorry

end NUMINAMATH_CALUDE_total_art_pieces_l3104_310426


namespace NUMINAMATH_CALUDE_must_divide_five_l3104_310443

theorem must_divide_five (a b c d : ℕ+) 
  (h1 : Nat.gcd a b = 40)
  (h2 : Nat.gcd b c = 45)
  (h3 : Nat.gcd c d = 75)
  (h4 : 120 < Nat.gcd d a ∧ Nat.gcd d a < 150) :
  5 ∣ a := by
  sorry

end NUMINAMATH_CALUDE_must_divide_five_l3104_310443


namespace NUMINAMATH_CALUDE_reciprocal_sum_theorem_l3104_310416

def sum_of_reciprocals (a b : ℕ+) : ℚ := (a⁻¹ : ℚ) + (b⁻¹ : ℚ)

theorem reciprocal_sum_theorem (a b : ℕ+) 
  (sum_cond : a + b = 45)
  (lcm_cond : Nat.lcm a b = 120)
  (hcf_cond : Nat.gcd a b = 5) :
  sum_of_reciprocals a b = 3/40 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_sum_theorem_l3104_310416


namespace NUMINAMATH_CALUDE_triangle_properties_l3104_310450

/-- Given a triangle ABC with sides a, b, c and angles A, B, C, prove properties about angle A and area. -/
theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  b^2 + c^2 - a^2 + b*c = 0 →
  Real.sin C = Real.sqrt 2 / 2 →
  a = Real.sqrt 3 →
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ A < Real.pi →
  0 < B ∧ B < Real.pi →
  0 < C ∧ C < Real.pi →
  A + B + C = Real.pi →
  a * Real.sin B = b * Real.sin A →
  a * Real.sin C = c * Real.sin A →
  b * Real.sin C = c * Real.sin B →
  A = 2 * Real.pi / 3 ∧
  (1/2 * a * c * Real.sin B) = (3 - Real.sqrt 3) / 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l3104_310450


namespace NUMINAMATH_CALUDE_stagecoach_encounter_l3104_310407

/-- The number of stagecoaches traveling daily from Bratislava to Brașov -/
def daily_coaches_bratislava_to_brasov : ℕ := 2

/-- The number of stagecoaches traveling daily from Brașov to Bratislava -/
def daily_coaches_brasov_to_bratislava : ℕ := 2

/-- The number of days the journey takes -/
def journey_duration : ℕ := 10

/-- The number of stagecoaches encountered when traveling from Bratislava to Brașov -/
def encountered_coaches : ℕ := daily_coaches_brasov_to_bratislava * journey_duration

theorem stagecoach_encounter :
  encountered_coaches = 20 :=
sorry

end NUMINAMATH_CALUDE_stagecoach_encounter_l3104_310407


namespace NUMINAMATH_CALUDE_range_of_m_range_of_a_l3104_310458

-- Part I
theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, 1/3 < x ∧ x < 1/2 → |x - m| < 1) → 
  -1/2 ≤ m ∧ m ≤ 4/3 := by
sorry

-- Part II
theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, |x - 3| + |x - 5| < a) →
  a > 2 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_range_of_a_l3104_310458


namespace NUMINAMATH_CALUDE_range_of_m_l3104_310440

theorem range_of_m (m x : ℝ) : 
  (((m + 3) / (x - 1) = 1) ∧ (x > 0)) → (m > -4 ∧ m ≠ -3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l3104_310440


namespace NUMINAMATH_CALUDE_large_paintings_sold_is_five_l3104_310453

/-- Represents the sale of paintings at an art show -/
structure PaintingSale where
  large_price : ℕ
  small_price : ℕ
  small_count : ℕ
  total_earnings : ℕ

/-- Calculates the number of large paintings sold -/
def large_paintings_sold (sale : PaintingSale) : ℕ :=
  (sale.total_earnings - sale.small_price * sale.small_count) / sale.large_price

/-- Theorem stating that the number of large paintings sold is 5 -/
theorem large_paintings_sold_is_five (sale : PaintingSale)
  (h1 : sale.large_price = 100)
  (h2 : sale.small_price = 80)
  (h3 : sale.small_count = 8)
  (h4 : sale.total_earnings = 1140) :
  large_paintings_sold sale = 5 := by
  sorry

end NUMINAMATH_CALUDE_large_paintings_sold_is_five_l3104_310453


namespace NUMINAMATH_CALUDE_simplify_expression_l3104_310404

theorem simplify_expression (x : ℝ) (hx : x > 0) :
  2 / (3 * x) * Real.sqrt (9 * x^3) + 6 * Real.sqrt (x / 4) - 2 * x * Real.sqrt (1 / x) = 3 * Real.sqrt x :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l3104_310404


namespace NUMINAMATH_CALUDE_least_integer_absolute_value_l3104_310469

theorem least_integer_absolute_value (x : ℤ) :
  (∀ y : ℤ, y < x → ¬(|3 * y - 5| ≤ 22)) ∧ (|3 * x - 5| ≤ 22) → x = -5 := by
  sorry

end NUMINAMATH_CALUDE_least_integer_absolute_value_l3104_310469


namespace NUMINAMATH_CALUDE_fudge_pan_dimensions_l3104_310490

theorem fudge_pan_dimensions (side1 : ℝ) (area : ℝ) : 
  side1 = 29 → area = 522 → (area / side1) = 18 := by
  sorry

end NUMINAMATH_CALUDE_fudge_pan_dimensions_l3104_310490


namespace NUMINAMATH_CALUDE_water_bottle_pricing_l3104_310466

theorem water_bottle_pricing (x : ℝ) 
  (h1 : x > 0) -- Ensure x is positive (price can't be negative or zero)
  (h2 : x > 10) -- Ensure x-10 is positive (price of type B can't be negative)
  : 700 / x = 500 / (x - 10) := by
  sorry

end NUMINAMATH_CALUDE_water_bottle_pricing_l3104_310466


namespace NUMINAMATH_CALUDE_distance_to_reflection_over_y_axis_l3104_310417

/-- Given a point F with coordinates (-4, 3), prove that the distance between F
    and its reflection over the y-axis is 8. -/
theorem distance_to_reflection_over_y_axis :
  let F : ℝ × ℝ := (-4, 3)
  let F' : ℝ × ℝ := (4, 3)  -- Reflection of F over y-axis
  dist F F' = 8 := by
  sorry

#check distance_to_reflection_over_y_axis

end NUMINAMATH_CALUDE_distance_to_reflection_over_y_axis_l3104_310417


namespace NUMINAMATH_CALUDE_complex_power_magnitude_l3104_310487

theorem complex_power_magnitude : Complex.abs ((2 + 2*Complex.I)^6) = 512 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_magnitude_l3104_310487


namespace NUMINAMATH_CALUDE_find_b_l3104_310454

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A (b : ℝ) : Set ℝ := {x | 1 ≤ x ∧ x < b}

-- Define the complement of A in U
def complement_A (b : ℝ) : Set ℝ := {x | x < 1 ∨ x ≥ 2}

-- Theorem statement
theorem find_b : ∃ b : ℝ, A b = Set.compl (complement_A b) := by sorry

end NUMINAMATH_CALUDE_find_b_l3104_310454


namespace NUMINAMATH_CALUDE_consecutive_digits_sum_divisibility_l3104_310409

/-- Given three consecutive digits p, q, and r, the sum of the three-digit 
    numbers pqr and rqp is always divisible by 212. -/
theorem consecutive_digits_sum_divisibility (p : ℕ) (hp : p < 8) : ∃ (k : ℕ),
  (100 * p + 10 * (p + 1) + (p + 2)) + (100 * (p + 2) + 10 * (p + 1) + p) = 212 * k :=
by
  sorry

#check consecutive_digits_sum_divisibility

end NUMINAMATH_CALUDE_consecutive_digits_sum_divisibility_l3104_310409


namespace NUMINAMATH_CALUDE_perfect_square_divisibility_l3104_310470

theorem perfect_square_divisibility (m n : ℕ) (h : m * n ∣ m^2 + n^2 + m) :
  ∃ k : ℕ, m = k^2 :=
sorry

end NUMINAMATH_CALUDE_perfect_square_divisibility_l3104_310470


namespace NUMINAMATH_CALUDE_chord_dot_product_l3104_310449

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a chord passing through the focus
def chord_through_focus (A B : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, A = (1 - t, -2*t) ∧ B = (1 + t, 2*t)

-- Theorem statement
theorem chord_dot_product (A B : ℝ × ℝ) :
  parabola A.1 A.2 → parabola B.1 B.2 → chord_through_focus A B →
  A.1 * B.1 + A.2 * B.2 = -3 :=
by sorry

end NUMINAMATH_CALUDE_chord_dot_product_l3104_310449


namespace NUMINAMATH_CALUDE_percentage_increase_l3104_310451

theorem percentage_increase (initial : ℝ) (final : ℝ) : 
  initial = 600 → final = 660 → (final - initial) / initial * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_l3104_310451


namespace NUMINAMATH_CALUDE_lottery_probability_l3104_310420

/-- The probability of exactly one person winning a prize in a lottery. -/
theorem lottery_probability : 
  let total_tickets : ℕ := 3
  let winning_tickets : ℕ := 2
  let people_drawing : ℕ := 2
  -- Probability of exactly one person winning
  (1 : ℚ) - (winning_tickets : ℚ) / (total_tickets : ℚ) * ((winning_tickets - 1) : ℚ) / ((total_tickets - 1) : ℚ) = 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_lottery_probability_l3104_310420


namespace NUMINAMATH_CALUDE_three_lines_intersection_l3104_310467

/-- Three lines intersect at a single point if and only if m = 9 -/
theorem three_lines_intersection (m : ℝ) : 
  (∃ (x y : ℝ), y = 2*x ∧ x + y = 3 ∧ m*x - 2*y - 5 = 0) ↔ m = 9 :=
by sorry

end NUMINAMATH_CALUDE_three_lines_intersection_l3104_310467


namespace NUMINAMATH_CALUDE_first_winner_of_both_prizes_l3104_310477

theorem first_winner_of_both_prizes (n : ℕ) : 
  (n % 5 = 0 ∧ n % 7 = 0) → n ≥ 35 :=
by sorry

end NUMINAMATH_CALUDE_first_winner_of_both_prizes_l3104_310477


namespace NUMINAMATH_CALUDE_sqrt_fraction_equality_l3104_310405

theorem sqrt_fraction_equality : Real.sqrt (25 / 121) = 5 / 11 := by sorry

end NUMINAMATH_CALUDE_sqrt_fraction_equality_l3104_310405


namespace NUMINAMATH_CALUDE_train_passing_pole_l3104_310495

theorem train_passing_pole (train_length platform_length : ℝ) 
  (platform_passing_time : ℝ) : 
  train_length = 120 →
  platform_length = 120 →
  platform_passing_time = 22 →
  (∃ (pole_passing_time : ℝ), 
    pole_passing_time = train_length / (train_length + platform_length) * platform_passing_time ∧
    pole_passing_time = 11) :=
by sorry

end NUMINAMATH_CALUDE_train_passing_pole_l3104_310495


namespace NUMINAMATH_CALUDE_police_emergency_number_prime_factor_l3104_310401

/-- A police emergency number is a positive integer that ends with 133 in decimal representation. -/
def PoliceEmergencyNumber (n : ℕ+) : Prop :=
  ∃ k : ℕ, n = k * 1000 + 133

/-- Theorem: Every police emergency number has a prime factor greater than 7. -/
theorem police_emergency_number_prime_factor
  (n : ℕ+) (h : PoliceEmergencyNumber n) :
  ∃ p : ℕ, p.Prime ∧ p > 7 ∧ p ∣ n.val :=
by sorry

end NUMINAMATH_CALUDE_police_emergency_number_prime_factor_l3104_310401


namespace NUMINAMATH_CALUDE_probability_of_two_specific_stamps_l3104_310419

/-- Represents a set of four distinct stamps -/
def Stamps : Type := Fin 4

/-- The number of ways to choose 2 stamps from 4 stamps -/
def total_combinations : ℕ := Nat.choose 4 2

/-- The number of ways to choose the specific 2 stamps we want -/
def favorable_combinations : ℕ := 1

/-- Theorem: The probability of drawing exactly two specific stamps from a set of four stamps is 1/6 -/
theorem probability_of_two_specific_stamps : 
  (favorable_combinations : ℚ) / total_combinations = 1 / 6 := by
  sorry


end NUMINAMATH_CALUDE_probability_of_two_specific_stamps_l3104_310419


namespace NUMINAMATH_CALUDE_general_term_max_sum_l3104_310481

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- General term of the sequence
  S : ℕ → ℤ  -- Sum function of the sequence
  sum_3 : S 3 = 42
  sum_6 : S 6 = 57

/-- The general term of the sequence is 20 - 3n -/
theorem general_term (seq : ArithmeticSequence) : 
  ∀ n : ℕ, seq.a n = 20 - 3 * n := by sorry

/-- The sum S_n is maximized when n = 6 -/
theorem max_sum (seq : ArithmeticSequence) : 
  ∃ n : ℕ, ∀ m : ℕ, seq.S n ≥ seq.S m ∧ n = 6 := by sorry

end NUMINAMATH_CALUDE_general_term_max_sum_l3104_310481


namespace NUMINAMATH_CALUDE_arithmetic_sequence_log_problem_l3104_310464

-- Define the logarithm function
noncomputable def log : ℝ → ℝ := Real.log

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0

-- Define the theorem
theorem arithmetic_sequence_log_problem 
  (a b : ℝ) 
  (h_pos_a : a > 0) 
  (h_pos_b : b > 0) 
  (h_arithmetic : arithmetic_sequence (λ n => 
    if n = 0 then log (a^5 * b^4)
    else if n = 1 then log (a^7 * b^9)
    else if n = 2 then log (a^10 * b^13)
    else if n = 9 then log (b^72)
    else 0)) : 
  ∃ n : ℕ, log (b^n) = (λ k => 
    if k = 0 then log (a^5 * b^4)
    else if k = 1 then log (a^7 * b^9)
    else if k = 2 then log (a^10 * b^13)
    else if k = 9 then log (b^72)
    else 0) 9 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_log_problem_l3104_310464


namespace NUMINAMATH_CALUDE_caricatures_sold_on_sunday_l3104_310421

/-- Proves the number of caricatures sold on Sunday given the conditions of the problem -/
theorem caricatures_sold_on_sunday 
  (price_per_caricature : ℚ)
  (saturday_sales : ℕ)
  (total_revenue : ℚ)
  (h1 : price_per_caricature = 20)
  (h2 : saturday_sales = 24)
  (h3 : total_revenue = 800) :
  (total_revenue - (↑saturday_sales * price_per_caricature)) / price_per_caricature = 16 := by
  sorry

end NUMINAMATH_CALUDE_caricatures_sold_on_sunday_l3104_310421


namespace NUMINAMATH_CALUDE_gain_percent_is_112_5_l3104_310406

/-- Represents the ratio of selling price to cost price -/
def price_ratio : ℚ := 5 / 2

/-- Represents the discount factor applied to the selling price -/
def discount_factor : ℚ := 85 / 100

/-- Calculates the gain percent based on the given conditions -/
def gain_percent : ℚ := (price_ratio * discount_factor - 1) * 100

/-- Theorem stating the gain percent under the given conditions -/
theorem gain_percent_is_112_5 : gain_percent = 112.5 := by
  sorry

end NUMINAMATH_CALUDE_gain_percent_is_112_5_l3104_310406


namespace NUMINAMATH_CALUDE_min_S6_arithmetic_sequence_l3104_310402

/-- Given an arithmetic sequence with common ratio q > 1, where S_n denotes the sum of first n terms,
    and S_4 = 2S_2 + 1, the minimum value of S_6 is 2√3 + 3. -/
theorem min_S6_arithmetic_sequence (q : ℝ) (a : ℕ → ℝ) (S : ℕ → ℝ) :
  q > 1 →
  (∀ n, a (n + 1) = a n + q) →
  (∀ n, S n = (n : ℝ) * (2 * a 1 + (n - 1) * q) / 2) →
  S 4 = 2 * S 2 + 1 →
  (∀ s : ℝ, s = S 6 → s ≥ 2 * Real.sqrt 3 + 3) ∧
  ∃ s : ℝ, s = S 6 ∧ s = 2 * Real.sqrt 3 + 3 :=
by sorry


end NUMINAMATH_CALUDE_min_S6_arithmetic_sequence_l3104_310402


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3104_310442

theorem imaginary_part_of_z (z : ℂ) : z = (Complex.I : ℂ) / (1 - Complex.I) → z.im = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3104_310442


namespace NUMINAMATH_CALUDE_hyperbola_transverse_axis_length_l3104_310498

/-- The length of the transverse axis of a hyperbola with equation x²/9 - y²/16 = 1 is 6 -/
theorem hyperbola_transverse_axis_length :
  ∀ (x y : ℝ), x^2 / 9 - y^2 / 16 = 1 →
  ∃ (a : ℝ), a > 0 ∧ a^2 = 9 ∧ 2 * a = 6 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_transverse_axis_length_l3104_310498


namespace NUMINAMATH_CALUDE_algebraic_simplification_l3104_310462

/-- Proves the simplification of two algebraic expressions -/
theorem algebraic_simplification 
  (a b m n : ℝ) : 
  ((a - 2*b) - (2*b - 5*a) = 6*a - 4*b) ∧ 
  (-m^2*n + (4*m*n^2 - 3*m*n) - 2*(m*n^2 - 3*m^2*n) = 5*m^2*n + 2*m*n^2 - 3*m*n) :=
by sorry

end NUMINAMATH_CALUDE_algebraic_simplification_l3104_310462


namespace NUMINAMATH_CALUDE_largest_four_digit_divisible_by_sum_of_digits_l3104_310411

/-- The sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- Predicate for a number being divisible by the sum of its digits -/
def isDivisibleBySumOfDigits (n : ℕ) : Prop :=
  n % sumOfDigits n = 0

/-- Theorem: 9990 is the largest four-digit number divisible by the sum of its digits -/
theorem largest_four_digit_divisible_by_sum_of_digits :
  ∀ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ isDivisibleBySumOfDigits n → n ≤ 9990 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_divisible_by_sum_of_digits_l3104_310411


namespace NUMINAMATH_CALUDE_two_digit_number_proof_l3104_310483

theorem two_digit_number_proof : ∃! n : ℕ, 
  10 ≤ n ∧ n < 100 ∧
  n % 16 = 0 ∧
  (n / 10 + n % 10 = 15) ∧
  n % 27 ≠ 0 ∧
  n % 36 ≠ 0 := by sorry

end NUMINAMATH_CALUDE_two_digit_number_proof_l3104_310483


namespace NUMINAMATH_CALUDE_student_guinea_pig_difference_l3104_310436

/-- The number of fourth-grade classrooms -/
def num_classrooms : ℕ := 5

/-- The number of students in each fourth-grade classroom -/
def students_per_classroom : ℕ := 24

/-- The number of guinea pigs in each fourth-grade classroom -/
def guinea_pigs_per_classroom : ℕ := 2

/-- The difference between the total number of students and guinea pigs in all classrooms -/
theorem student_guinea_pig_difference :
  num_classrooms * students_per_classroom - num_classrooms * guinea_pigs_per_classroom = 110 := by
  sorry

end NUMINAMATH_CALUDE_student_guinea_pig_difference_l3104_310436


namespace NUMINAMATH_CALUDE_dan_marbles_remaining_l3104_310447

/-- The number of marbles Dan has after giving some to Mary -/
def marbles_remaining (initial : ℕ) (given : ℕ) : ℕ :=
  initial - given

/-- Theorem stating that Dan has 50 marbles after giving 14 to Mary -/
theorem dan_marbles_remaining : marbles_remaining 64 14 = 50 := by
  sorry

end NUMINAMATH_CALUDE_dan_marbles_remaining_l3104_310447


namespace NUMINAMATH_CALUDE_max_integer_value_x_l3104_310489

theorem max_integer_value_x (x : ℤ) : 
  (3 : ℚ) * x - 1/4 ≤ 1/3 * x - 2 → x ≤ -1 :=
by sorry

end NUMINAMATH_CALUDE_max_integer_value_x_l3104_310489


namespace NUMINAMATH_CALUDE_rational_sum_and_square_sum_integer_implies_integer_l3104_310452

theorem rational_sum_and_square_sum_integer_implies_integer (a b : ℚ) 
  (h1 : ∃ n : ℤ, (a + b : ℚ) = n)
  (h2 : ∃ m : ℤ, (a^2 + b^2 : ℚ) = m) :
  ∃ (x y : ℤ), (a = x ∧ b = y) :=
sorry

end NUMINAMATH_CALUDE_rational_sum_and_square_sum_integer_implies_integer_l3104_310452


namespace NUMINAMATH_CALUDE_liter_equals_cubic_decimeter_l3104_310456

-- Define the conversion factor between liters and cubic decimeters
def liter_to_cubic_decimeter : ℝ := 1

-- Theorem statement
theorem liter_equals_cubic_decimeter :
  1.5 * liter_to_cubic_decimeter = 1.5 := by sorry

end NUMINAMATH_CALUDE_liter_equals_cubic_decimeter_l3104_310456


namespace NUMINAMATH_CALUDE_smallest_y_for_square_l3104_310496

theorem smallest_y_for_square (y : ℕ) : y = 10 ↔ 
  (y > 0 ∧ 
   ∃ n : ℕ, 4410 * y = n^2 ∧
   ∀ z < y, z > 0 → ¬∃ m : ℕ, 4410 * z = m^2) := by
sorry

end NUMINAMATH_CALUDE_smallest_y_for_square_l3104_310496


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l3104_310484

/-- The ratio of area to perimeter for an equilateral triangle with side length 6 -/
theorem equilateral_triangle_area_perimeter_ratio :
  let side_length : ℝ := 6
  let area : ℝ := (side_length^2 * Real.sqrt 3) / 4
  let perimeter : ℝ := 3 * side_length
  area / perimeter = Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l3104_310484


namespace NUMINAMATH_CALUDE_weight_replacement_l3104_310475

theorem weight_replacement (n : ℕ) (new_weight avg_increase : ℝ) :
  n = 8 →
  new_weight = 93 →
  avg_increase = 3.5 →
  new_weight - n * avg_increase = 65 := by
  sorry

end NUMINAMATH_CALUDE_weight_replacement_l3104_310475


namespace NUMINAMATH_CALUDE_unique_solutions_l3104_310408

def system_solution (x y : ℝ) : Prop :=
  x > 0 ∧ x ≠ 1 ∧ y > 0 ∧ y ≠ 1 ∧
  x + y = 12 ∧
  2 * (2 * (Real.log x / Real.log (y^2)) - Real.log y / Real.log (1/x)) = 5

theorem unique_solutions :
  ∀ x y : ℝ, system_solution x y ↔ ((x = 9 ∧ y = 3) ∨ (x = 3 ∧ y = 9)) :=
by sorry

end NUMINAMATH_CALUDE_unique_solutions_l3104_310408


namespace NUMINAMATH_CALUDE_incorrect_permutations_hello_l3104_310435

def word_length : ℕ := 5
def repeated_letter_count : ℕ := 2

theorem incorrect_permutations_hello :
  (word_length.factorial / repeated_letter_count.factorial) - 1 = 119 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_permutations_hello_l3104_310435


namespace NUMINAMATH_CALUDE_ratio_equality_l3104_310444

theorem ratio_equality (a b c x y z : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (sum_squares_abc : a^2 + b^2 + c^2 = 49)
  (sum_squares_xyz : x^2 + y^2 + z^2 = 64)
  (dot_product : a*x + b*y + c*z = 56) :
  (a + b + c) / (x + y + z) = 7/8 := by
sorry

end NUMINAMATH_CALUDE_ratio_equality_l3104_310444


namespace NUMINAMATH_CALUDE_sean_whistle_count_l3104_310418

/-- Given that Charles has 128 whistles and Sean has 95 more whistles than Charles,
    prove that Sean has 223 whistles. -/
theorem sean_whistle_count :
  let charles_whistles : ℕ := 128
  let sean_extra_whistles : ℕ := 95
  let sean_whistles : ℕ := charles_whistles + sean_extra_whistles
  sean_whistles = 223 := by
  sorry

end NUMINAMATH_CALUDE_sean_whistle_count_l3104_310418


namespace NUMINAMATH_CALUDE_rhombus_fourth_vertex_area_l3104_310497

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A square defined by its four vertices -/
structure Square where
  a : Point
  b : Point
  c : Point
  d : Point

/-- A rhombus defined by its four vertices -/
structure Rhombus where
  p : Point
  q : Point
  r : Point
  s : Point

/-- Predicate to check if a square is a unit square -/
def isUnitSquare (sq : Square) : Prop := sorry

/-- Predicate to check if a point lies on a side of the square -/
def pointOnSide (p : Point) (sq : Square) : Prop := sorry

/-- Function to calculate the area of a set of points -/
def areaOfSet (s : Set Point) : ℝ := sorry

/-- The set of all possible locations for the fourth vertex of the rhombus -/
def fourthVertexSet (sq : Square) : Set Point := sorry

/-- Main theorem -/
theorem rhombus_fourth_vertex_area (sq : Square) :
  isUnitSquare sq →
  (∃ (r : Rhombus), 
    pointOnSide r.p sq ∧ 
    pointOnSide r.q sq ∧ 
    pointOnSide r.r sq) →
  areaOfSet (fourthVertexSet sq) = 7/3 := sorry

end NUMINAMATH_CALUDE_rhombus_fourth_vertex_area_l3104_310497


namespace NUMINAMATH_CALUDE_custom_mult_theorem_l3104_310480

/-- Custom multiplication operation -/
def custom_mult (m n : ℝ) : ℝ := 2 * m - 3 * n

/-- Theorem stating that if x satisfies the given condition, then x = 7 -/
theorem custom_mult_theorem (x : ℝ) : 
  (∀ m n : ℝ, custom_mult m n = 2 * m - 3 * n) → 
  custom_mult x 7 = custom_mult 7 x → 
  x = 7 := by
  sorry

end NUMINAMATH_CALUDE_custom_mult_theorem_l3104_310480


namespace NUMINAMATH_CALUDE_solution_set_implies_a_values_l3104_310460

theorem solution_set_implies_a_values (a : ℕ) 
  (h : ∀ x : ℝ, (a - 2 : ℝ) * x > (a - 2 : ℝ) ↔ x < 1) : 
  a = 0 ∨ a = 1 := by
sorry

end NUMINAMATH_CALUDE_solution_set_implies_a_values_l3104_310460


namespace NUMINAMATH_CALUDE_volume_rotated_square_l3104_310472

/-- The volume of a solid formed by rotating a square around its diagonal -/
theorem volume_rotated_square (area : ℝ) (volume : ℝ) : 
  area = 4 → volume = (4 * Real.sqrt 2 * Real.pi) / 3 := by
  sorry

end NUMINAMATH_CALUDE_volume_rotated_square_l3104_310472


namespace NUMINAMATH_CALUDE_tan_value_for_special_condition_l3104_310412

theorem tan_value_for_special_condition (α : Real) 
  (h1 : α > 0) (h2 : α < Real.pi / 2) 
  (h3 : Real.sin α ^ 2 + Real.cos (2 * α) = 1 / 4) : 
  Real.tan α = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_tan_value_for_special_condition_l3104_310412


namespace NUMINAMATH_CALUDE_two_transformations_preserve_pattern_l3104_310476

/-- Represents the pattern of squares on the infinite line -/
structure SquarePattern where
  s : ℝ  -- side length of each square
  ℓ : Line2  -- the infinite line

/-- Enumeration of the four transformations -/
inductive Transformation
  | rotation180 : Point → Transformation
  | translation4s : Transformation
  | reflectionAcrossL : Transformation
  | reflectionPerpendicular : Point → Transformation

/-- Predicate to check if a transformation maps the pattern onto itself -/
def mapsOntoItself (t : Transformation) (p : SquarePattern) : Prop :=
  sorry

theorem two_transformations_preserve_pattern (p : SquarePattern) :
  ∃! (ts : Finset Transformation), ts.card = 2 ∧
    ∀ t ∈ ts, mapsOntoItself t p ∧
    ∀ t, mapsOntoItself t p → t ∈ ts :=
  sorry

end NUMINAMATH_CALUDE_two_transformations_preserve_pattern_l3104_310476


namespace NUMINAMATH_CALUDE_power_division_nineteen_l3104_310434

theorem power_division_nineteen : 19^11 / 19^8 = 6859 := by sorry

end NUMINAMATH_CALUDE_power_division_nineteen_l3104_310434


namespace NUMINAMATH_CALUDE_regular_polygon_interior_exterior_angle_relation_l3104_310455

theorem regular_polygon_interior_exterior_angle_relation (n : ℕ) :
  (n ≥ 3) →
  ((n - 2) * 180 : ℝ) = 2 * 360 →
  n = 6 :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_interior_exterior_angle_relation_l3104_310455


namespace NUMINAMATH_CALUDE_simplified_fourth_root_sum_l3104_310479

theorem simplified_fourth_root_sum (a b : ℕ+) :
  (2^6 * 5^2 : ℝ)^(1/4) = a * b^(1/4) → a + b = 102 := by
  sorry

end NUMINAMATH_CALUDE_simplified_fourth_root_sum_l3104_310479


namespace NUMINAMATH_CALUDE_fifty_cent_items_count_l3104_310425

theorem fifty_cent_items_count (a b c : ℕ) : 
  a + b + c = 50 →
  50 * a + 400 * b + 500 * c = 10000 →
  a = 40 :=
by sorry

end NUMINAMATH_CALUDE_fifty_cent_items_count_l3104_310425


namespace NUMINAMATH_CALUDE_magic_square_x_value_l3104_310430

/-- Represents a 3x3 multiplicative magic square --/
structure MagicSquare where
  a11 : ℝ
  a12 : ℝ
  a13 : ℝ
  a21 : ℝ
  a22 : ℝ
  a23 : ℝ
  a31 : ℝ
  a32 : ℝ
  a33 : ℝ
  positive : ∀ i j, (i, j) ∈ [(1,1), (1,2), (1,3), (2,1), (2,2), (2,3), (3,1), (3,2), (3,3)] → 
    match (i, j) with
    | (1, 1) => a11 > 0
    | (1, 2) => a12 > 0
    | (1, 3) => a13 > 0
    | (2, 1) => a21 > 0
    | (2, 2) => a22 > 0
    | (2, 3) => a23 > 0
    | (3, 1) => a31 > 0
    | (3, 2) => a32 > 0
    | (3, 3) => a33 > 0
    | _ => False
  magic : a11 * a12 * a13 = a21 * a22 * a23 ∧
          a11 * a12 * a13 = a31 * a32 * a33 ∧
          a11 * a12 * a13 = a11 * a21 * a31 ∧
          a11 * a12 * a13 = a12 * a22 * a32 ∧
          a11 * a12 * a13 = a13 * a23 * a33 ∧
          a11 * a12 * a13 = a11 * a22 * a33 ∧
          a11 * a12 * a13 = a13 * a22 * a31

theorem magic_square_x_value (ms : MagicSquare) 
  (h1 : ms.a11 = 5)
  (h2 : ms.a21 = 4)
  (h3 : ms.a33 = 20) :
  ms.a12 = 100 := by
  sorry

end NUMINAMATH_CALUDE_magic_square_x_value_l3104_310430


namespace NUMINAMATH_CALUDE_fourth_number_proof_l3104_310413

theorem fourth_number_proof (x : ℝ) : 
  3 + 33 + 333 + x = 399.6 → x = 30.6 := by
sorry

end NUMINAMATH_CALUDE_fourth_number_proof_l3104_310413


namespace NUMINAMATH_CALUDE_ladder_cost_theorem_l3104_310438

/-- Calculates the total cost of ladders given the number of ladders, rungs per ladder, and cost per rung for three different types of ladders. -/
def total_ladder_cost (ladders1 rungs1 cost1 ladders2 rungs2 cost2 ladders3 rungs3 cost3 : ℕ) : ℕ :=
  ladders1 * rungs1 * cost1 + ladders2 * rungs2 * cost2 + ladders3 * rungs3 * cost3

/-- Proves that the total cost of ladders for the given specifications is $14200. -/
theorem ladder_cost_theorem :
  total_ladder_cost 10 50 2 20 60 3 30 80 4 = 14200 := by
  sorry

end NUMINAMATH_CALUDE_ladder_cost_theorem_l3104_310438


namespace NUMINAMATH_CALUDE_triangle_ratio_l3104_310446

/-- In a triangle ABC, given that a * sin(A) * sin(B) + b * cos²(A) = √3 * a, 
    prove that b/a = √3 -/
theorem triangle_ratio (a b c : ℝ) (A B C : ℝ) : 
  (a > 0) → 
  (b > 0) → 
  (c > 0) → 
  (A > 0) → (A < π) →
  (B > 0) → (B < π) →
  (C > 0) → (C < π) →
  (A + B + C = π) →
  (a * Real.sin A * Real.sin B + b * (Real.cos A)^2 = Real.sqrt 3 * a) →
  (b / a = Real.sqrt 3) := by
sorry

end NUMINAMATH_CALUDE_triangle_ratio_l3104_310446


namespace NUMINAMATH_CALUDE_d_72_eq_22_l3104_310423

/-- D(n) is the number of ways to write n as a product of integers greater than 1,
    where the order of factors matters. -/
def D (n : ℕ+) : ℕ :=
  sorry

/-- Theorem: The number of ways to express 72 as a product of integers greater than 1,
    considering the order, is 22. -/
theorem d_72_eq_22 : D 72 = 22 := by
  sorry

end NUMINAMATH_CALUDE_d_72_eq_22_l3104_310423


namespace NUMINAMATH_CALUDE_shaded_area_between_circles_l3104_310427

theorem shaded_area_between_circles (d_small : ℝ) (r_large : ℝ) :
  d_small = 6 →
  r_large = 5 * (d_small / 2) →
  let r_small := d_small / 2
  let area_large := π * r_large^2
  let area_small := π * r_small^2
  area_large - area_small = 216 * π := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_between_circles_l3104_310427


namespace NUMINAMATH_CALUDE_money_division_l3104_310424

theorem money_division (a b c : ℕ) (h1 : a = b / 2) (h2 : b = c / 2) (h3 : c = 224) :
  a + b + c = 392 := by
  sorry

end NUMINAMATH_CALUDE_money_division_l3104_310424


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l3104_310463

theorem quadratic_no_real_roots 
  (p q a b c : ℝ) 
  (hp : p > 0) (hq : q > 0) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hpq : p ≠ q)
  (hgeom : a^2 = p * q)  -- p, a, q form a geometric sequence
  (harith : b + c = p + q)  -- p, b, c, q form an arithmetic sequence
  : ∀ x : ℝ, b * x^2 - 2 * a * x + c ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l3104_310463


namespace NUMINAMATH_CALUDE_complex_square_root_l3104_310486

theorem complex_square_root : ∃ (z : ℂ),
  let a : ℝ := Real.sqrt ((-81 + Real.sqrt 8865) / 2)
  let b : ℝ := -24 / a
  z = Complex.mk a b ∧ z^2 = Complex.mk (-81) (-48) := by
  sorry

end NUMINAMATH_CALUDE_complex_square_root_l3104_310486


namespace NUMINAMATH_CALUDE_swimmers_speed_l3104_310429

/-- Proves that a person's swimming speed in still water is 4 km/h given the conditions -/
theorem swimmers_speed (water_speed : ℝ) (distance : ℝ) (time : ℝ) (h1 : water_speed = 2)
  (h2 : distance = 16) (h3 : time = 8) : ∃ v : ℝ, v = 4 ∧ distance = (v - water_speed) * time :=
by
  sorry

end NUMINAMATH_CALUDE_swimmers_speed_l3104_310429


namespace NUMINAMATH_CALUDE_first_zero_position_l3104_310448

open Real

-- Define the decimal expansion of √2
def sqrt2_expansion : ℕ → ℕ
  | 0 => 1  -- integer part
  | (n+1) => sorry  -- n-th decimal digit, implementation details omitted

-- Define a function to check if there's a sequence of k zeroes starting at position n
def has_k_zeroes (k n : ℕ) : Prop :=
  ∀ i, i ∈ Finset.range k → sqrt2_expansion (n + i) = 0

-- Main theorem
theorem first_zero_position (k : ℕ) (h : k > 0) :
  ∀ n, has_k_zeroes k n → n ≥ k :=
sorry

end NUMINAMATH_CALUDE_first_zero_position_l3104_310448


namespace NUMINAMATH_CALUDE_max_y_over_x_l3104_310459

theorem max_y_over_x (x y : ℝ) (h : x^2 + y^2 - 6*x - 6*y + 12 = 0) :
  ∃ (k : ℝ), k = 3 + 2 * Real.sqrt 2 ∧ ∀ (x' y' : ℝ), x'^2 + y'^2 - 6*x' - 6*y' + 12 = 0 → y' / x' ≤ k := by
  sorry

end NUMINAMATH_CALUDE_max_y_over_x_l3104_310459


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l3104_310468

/-- The discriminant of a quadratic equation ax^2 + bx + c is b^2 - 4ac -/
def discriminant (a b c : ℚ) : ℚ := b^2 - 4*a*c

theorem quadratic_discriminant :
  let a : ℚ := 5
  let b : ℚ := 5 + 1/2
  let c : ℚ := -2
  discriminant a b c = 281/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l3104_310468


namespace NUMINAMATH_CALUDE_frog_jump_distance_l3104_310461

/-- The jumping contest problem -/
theorem frog_jump_distance 
  (grasshopper_jump : ℕ) 
  (frog_extra_jump : ℕ) 
  (mouse_less_jump : ℕ) 
  (h1 : grasshopper_jump = 19)
  (h2 : frog_extra_jump = 39)
  (h3 : mouse_less_jump = 94) :
  grasshopper_jump + frog_extra_jump = 58 :=
by sorry

end NUMINAMATH_CALUDE_frog_jump_distance_l3104_310461


namespace NUMINAMATH_CALUDE_fourth_root_l3104_310494

/-- The polynomial function defined by the given coefficients -/
def f (b c x : ℝ) : ℝ := b*x^4 + (b + 3*c)*x^3 + (c - 4*b)*x^2 + (19 - b)*x - 2

theorem fourth_root (b c : ℝ) 
  (h1 : f b c (-3) = 0)
  (h2 : f b c 4 = 0)
  (h3 : f b c 2 = 0) :
  ∃ x, x ≠ -3 ∧ x ≠ 4 ∧ x ≠ 2 ∧ f b c x = 0 ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_l3104_310494


namespace NUMINAMATH_CALUDE_power_of_power_of_three_l3104_310441

theorem power_of_power_of_three : (3 : ℕ) ^ ((3 : ℕ) ^ (3 : ℕ)) = (3 : ℕ) ^ (27 : ℕ) := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_of_three_l3104_310441


namespace NUMINAMATH_CALUDE_modern_model_leads_to_older_structure_l3104_310428

/-- Represents the population growth model -/
structure PopulationGrowthModel where
  -- Add necessary fields here

/-- Represents the age structure of a population -/
inductive AgeStructure
  | Younger
  | Older

/-- The modern population growth model -/
def modernPopulationGrowthModel : PopulationGrowthModel :=
  sorry

/-- The consequence of a population growth model on age structure -/
def consequenceOnAgeStructure (model : PopulationGrowthModel) : AgeStructure :=
  sorry

/-- Theorem stating that the modern population growth model leads to an older age structure -/
theorem modern_model_leads_to_older_structure :
  consequenceOnAgeStructure modernPopulationGrowthModel = AgeStructure.Older :=
sorry

end NUMINAMATH_CALUDE_modern_model_leads_to_older_structure_l3104_310428


namespace NUMINAMATH_CALUDE_product_and_sum_of_integers_l3104_310482

theorem product_and_sum_of_integers : ∃ (a b c d e : ℤ),
  (b = a + 1) ∧
  (d = c + 1) ∧
  (e = d + 1) ∧
  (a > 0) ∧
  (a * b = 336) ∧
  (c * d * e = 336) ∧
  (a + b + c + d + e = 51) := by
sorry

end NUMINAMATH_CALUDE_product_and_sum_of_integers_l3104_310482


namespace NUMINAMATH_CALUDE_projectile_max_height_l3104_310488

/-- The height function of the projectile --/
def h (t : ℝ) : ℝ := -12 * t^2 + 48 * t + 25

/-- The maximum height reached by the projectile --/
theorem projectile_max_height : 
  ∃ (t_max : ℝ), ∀ (t : ℝ), h t ≤ h t_max ∧ h t_max = 73 :=
sorry

end NUMINAMATH_CALUDE_projectile_max_height_l3104_310488


namespace NUMINAMATH_CALUDE_sin_negative_nineteen_pi_sixths_l3104_310457

theorem sin_negative_nineteen_pi_sixths (π : Real) : 
  let sine_is_odd : ∀ x, Real.sin (-x) = -Real.sin x := by sorry
  let sine_period : ∀ x, Real.sin (x + 2 * π) = Real.sin x := by sorry
  let sine_cofunction : ∀ θ, Real.sin (π + θ) = -Real.sin θ := by sorry
  let sin_pi_sixth : Real.sin (π / 6) = 1 / 2 := by sorry
  Real.sin (-19 * π / 6) = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_negative_nineteen_pi_sixths_l3104_310457


namespace NUMINAMATH_CALUDE_inequality_and_equality_conditions_l3104_310439

theorem inequality_and_equality_conditions (a b : ℝ) (h : a * b > 0) :
  (((a^2 * b^2 * (a + b)^2) / 4)^(1/3) ≤ (a^2 + 10*a*b + b^2) / 12) ∧
  (((a^2 * b^2 * (a + b)^2) / 4)^(1/3) = (a^2 + 10*a*b + b^2) / 12 ↔ a = b) ∧
  (((a^2 * b^2 * (a + b)^2) / 4)^(1/3) ≤ (a^2 + a*b + b^2) / 3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_conditions_l3104_310439


namespace NUMINAMATH_CALUDE_axb_equals_bxa_l3104_310403

open Matrix

variable {n : ℕ}
variable (A B X : Matrix (Fin n) (Fin n) ℝ)

theorem axb_equals_bxa (h : A * X * B + A + B = 0) : A * X * B = B * X * A := by
  sorry

end NUMINAMATH_CALUDE_axb_equals_bxa_l3104_310403


namespace NUMINAMATH_CALUDE_james_soda_packs_l3104_310432

/-- The number of packs of sodas James bought -/
def packs_bought : ℕ := 5

/-- The number of sodas in each pack -/
def sodas_per_pack : ℕ := 12

/-- The number of sodas James already had -/
def initial_sodas : ℕ := 10

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of sodas James drinks per day -/
def sodas_per_day : ℕ := 10

theorem james_soda_packs :
  packs_bought * sodas_per_pack + initial_sodas = sodas_per_day * days_in_week := by
  sorry

end NUMINAMATH_CALUDE_james_soda_packs_l3104_310432
