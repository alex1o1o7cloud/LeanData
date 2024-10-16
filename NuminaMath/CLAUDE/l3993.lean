import Mathlib

namespace NUMINAMATH_CALUDE_simplify_fraction_l3993_399315

theorem simplify_fraction (x y : ℝ) (hx : x = 3) (hy : y = 4) :
  (15 * x^2 * y^3) / (9 * x * y^2) = 20 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3993_399315


namespace NUMINAMATH_CALUDE_tomatoes_sold_to_wilson_l3993_399351

def total_harvest : Float := 245.5
def sold_to_maxwell : Float := 125.5
def not_sold : Float := 42.0

theorem tomatoes_sold_to_wilson :
  total_harvest - sold_to_maxwell - not_sold = 78 := by
  sorry

end NUMINAMATH_CALUDE_tomatoes_sold_to_wilson_l3993_399351


namespace NUMINAMATH_CALUDE_equal_cost_at_60_messages_l3993_399337

/-- The cost per text message for Plan A -/
def plan_a_cost_per_text : ℚ := 25 / 100

/-- The monthly fee for Plan A -/
def plan_a_monthly_fee : ℚ := 9

/-- The cost per text message for Plan B -/
def plan_b_cost_per_text : ℚ := 40 / 100

/-- The number of text messages at which both plans cost the same -/
def equal_cost_messages : ℕ := 60

theorem equal_cost_at_60_messages :
  plan_a_cost_per_text * equal_cost_messages + plan_a_monthly_fee =
  plan_b_cost_per_text * equal_cost_messages :=
by sorry

end NUMINAMATH_CALUDE_equal_cost_at_60_messages_l3993_399337


namespace NUMINAMATH_CALUDE_sector_angle_l3993_399334

/-- Given a circular sector with arc length and area both equal to 3,
    prove that the central angle in radians is 3/2. -/
theorem sector_angle (r : ℝ) (θ : ℝ) 
  (arc_length : θ * r = 3)
  (area : 1/2 * θ * r^2 = 3) :
  θ = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_sector_angle_l3993_399334


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l3993_399385

def U : Set ℤ := {-2, -1, 0, 1, 2}

def A : Set ℤ := {x : ℤ | 0 < |x| ∧ |x| < 2}

theorem complement_of_A_in_U :
  U \ A = {-2, 0, 2} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l3993_399385


namespace NUMINAMATH_CALUDE_unique_solution_inequality_l3993_399323

theorem unique_solution_inequality (a : ℝ) : 
  (∃! x : ℝ, |x^2 + 2*a*x + 3*a| ≤ 2) ↔ (a = 1 ∨ a = 2) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_inequality_l3993_399323


namespace NUMINAMATH_CALUDE_sum_of_integers_with_product_seven_cubed_l3993_399316

theorem sum_of_integers_with_product_seven_cubed :
  ∃ (a b c : ℕ+),
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (a * b * c : ℕ) = 7^3 →
    (a : ℕ) + (b : ℕ) + (c : ℕ) = 57 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_with_product_seven_cubed_l3993_399316


namespace NUMINAMATH_CALUDE_salary_change_l3993_399300

theorem salary_change (original_salary : ℝ) (h : original_salary > 0) :
  let increased_salary := original_salary * 1.3
  let final_salary := increased_salary * 0.7
  (final_salary - original_salary) / original_salary = -0.09 := by
sorry

end NUMINAMATH_CALUDE_salary_change_l3993_399300


namespace NUMINAMATH_CALUDE_cubic_root_sum_l3993_399390

theorem cubic_root_sum (a b : ℝ) : 
  (Complex.I : ℂ) ^ 2 = -1 →
  (2 + Complex.I : ℂ) ^ 3 + a * (2 + Complex.I : ℂ) + b = 0 →
  a + b = 9 := by sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l3993_399390


namespace NUMINAMATH_CALUDE_table_cost_l3993_399312

/-- The cost of furniture items and payment details --/
structure FurniturePurchase where
  couch_cost : ℕ
  lamp_cost : ℕ
  initial_payment : ℕ
  remaining_balance : ℕ

/-- Theorem stating the cost of the table --/
theorem table_cost (purchase : FurniturePurchase)
  (h1 : purchase.couch_cost = 750)
  (h2 : purchase.lamp_cost = 50)
  (h3 : purchase.initial_payment = 500)
  (h4 : purchase.remaining_balance = 400) :
  ∃ (table_cost : ℕ), 
    purchase.couch_cost + table_cost + purchase.lamp_cost - purchase.initial_payment = purchase.remaining_balance ∧
    table_cost = 100 :=
sorry

end NUMINAMATH_CALUDE_table_cost_l3993_399312


namespace NUMINAMATH_CALUDE_count_valid_pairs_l3993_399305

def has_two_distinct_real_solutions (a b c : ℤ) : Prop :=
  b^2 - 4*a*c > 0

def valid_pair (b c : ℕ+) : Prop :=
  ¬(has_two_distinct_real_solutions 1 b c) ∧
  ¬(has_two_distinct_real_solutions 1 c b)

theorem count_valid_pairs :
  ∃ (S : Finset (ℕ+ × ℕ+)), 
    (∀ (p : ℕ+ × ℕ+), p ∈ S ↔ valid_pair p.1 p.2) ∧
    Finset.card S = 6 := by sorry

end NUMINAMATH_CALUDE_count_valid_pairs_l3993_399305


namespace NUMINAMATH_CALUDE_min_value_problem_l3993_399352

theorem min_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_sum : x + 2*y = 1) :
  (x^2 + y^2 + x) / (x*y) ≥ 2*Real.sqrt 2 + 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_problem_l3993_399352


namespace NUMINAMATH_CALUDE_min_value_expression_l3993_399391

theorem min_value_expression (x y z : ℝ) (h : z = Real.sin x) :
  ∃ (m : ℝ), (∀ (x' y' z' : ℝ), z' = Real.sin x' →
    (y' * Real.cos x' - 2)^2 + (y' + z' + 1)^2 ≥ m) ∧
  m = (1/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3993_399391


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l3993_399329

theorem product_of_three_numbers (a b c : ℝ) 
  (sum_condition : a + b + c = 24)
  (sum_squares_condition : a^2 + b^2 + c^2 = 392)
  (sum_cubes_condition : a^3 + b^3 + c^3 = 2760) :
  a * b * c = 1844 := by sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l3993_399329


namespace NUMINAMATH_CALUDE_f_neg_five_eq_one_l3993_399371

/-- A polynomial function of degree 5 with a constant term of 5 -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^5 + b * x^3 + c * x + 5

/-- Theorem stating that if f(5) = 9, then f(-5) = 1 -/
theorem f_neg_five_eq_one (a b c : ℝ) (h : f a b c 5 = 9) : f a b c (-5) = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_neg_five_eq_one_l3993_399371


namespace NUMINAMATH_CALUDE_i_to_2016_l3993_399367

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem i_to_2016 : i^2016 = 1 := by
  sorry

end NUMINAMATH_CALUDE_i_to_2016_l3993_399367


namespace NUMINAMATH_CALUDE_unique_solution_proof_l3993_399373

/-- The value of q for which the quadratic equation qx^2 - 16x + 8 = 0 has exactly one solution -/
def unique_solution_q : ℝ := 8

/-- The quadratic equation qx^2 - 16x + 8 = 0 -/
def quadratic_equation (q x : ℝ) : ℝ := q * x^2 - 16 * x + 8

theorem unique_solution_proof :
  ∀ q : ℝ, q ≠ 0 →
  (∃! x : ℝ, quadratic_equation q x = 0) ↔ q = unique_solution_q :=
sorry

end NUMINAMATH_CALUDE_unique_solution_proof_l3993_399373


namespace NUMINAMATH_CALUDE_taxi_ride_cost_l3993_399303

def base_fee : ℚ := 1.5
def cost_per_mile : ℚ := 0.25
def ride1_distance : ℕ := 5
def ride2_distance : ℕ := 8
def ride3_distance : ℕ := 3

theorem taxi_ride_cost : 
  (base_fee + cost_per_mile * ride1_distance) + 
  (base_fee + cost_per_mile * ride2_distance) + 
  (base_fee + cost_per_mile * ride3_distance) = 8.5 := by
sorry

end NUMINAMATH_CALUDE_taxi_ride_cost_l3993_399303


namespace NUMINAMATH_CALUDE_product_divisible_by_ten_l3993_399394

theorem product_divisible_by_ten : ∃ k : ℤ, 1265 * 4233 * 254 * 1729 = 10 * k := by sorry

end NUMINAMATH_CALUDE_product_divisible_by_ten_l3993_399394


namespace NUMINAMATH_CALUDE_curve_circle_intersection_perpendicular_l3993_399344

-- Define the curve C
def curve_C (x y m : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y + m = 0

-- Define the line
def line (x y : ℝ) : Prop := x + 2*y - 3 = 0

-- Define the perpendicularity condition
def perpendicular (x1 y1 x2 y2 : ℝ) : Prop := x1*x2 + y1*y2 = 0

theorem curve_circle_intersection_perpendicular (m : ℝ) :
  (∃ x1 y1 x2 y2 : ℝ, 
    curve_C x1 y1 m ∧ 
    curve_C x2 y2 m ∧ 
    line x1 y1 ∧ 
    line x2 y2 ∧ 
    perpendicular x1 y1 x2 y2) →
  m = 12/5 :=
by sorry

end NUMINAMATH_CALUDE_curve_circle_intersection_perpendicular_l3993_399344


namespace NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l3993_399362

-- Problem 1
theorem factorization_problem_1 (a b : ℝ) :
  a^3*b - 2*a^2*b^2 + a*b^3 = a*b*(a-b)^2 := by sorry

-- Problem 2
theorem factorization_problem_2 (x : ℝ) :
  (x^2 + 4)^2 - 16*x^2 = (x+2)^2 * (x-2)^2 := by sorry

end NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l3993_399362


namespace NUMINAMATH_CALUDE_min_distance_to_line_l3993_399326

theorem min_distance_to_line (A B : ℝ × ℝ) (l : ℝ → ℝ → Prop) :
  A = (-2, 0) →
  B = (0, 3) →
  (∀ x y, l x y ↔ x - y + 1 = 0) →
  ∃ P : ℝ × ℝ, l P.1 P.2 ∧
    (∀ Q : ℝ × ℝ, l Q.1 Q.2 → Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2) +
                               Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2) ≤
                               Real.sqrt ((Q.1 - A.1)^2 + (Q.2 - A.2)^2) +
                               Real.sqrt ((Q.1 - B.1)^2 + (Q.2 - B.2)^2)) ∧
    Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2) + Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2) = Real.sqrt 17 :=
by sorry


end NUMINAMATH_CALUDE_min_distance_to_line_l3993_399326


namespace NUMINAMATH_CALUDE_sum_of_digits_of_difference_gcd_l3993_399384

def difference_gcd (a b c : ℕ) : ℕ :=
  Nat.gcd (Nat.gcd (b - a) (c - b)) (c - a)

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem sum_of_digits_of_difference_gcd :
  sum_of_digits (difference_gcd 1305 4665 6905) = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_difference_gcd_l3993_399384


namespace NUMINAMATH_CALUDE_four_digit_sum_l3993_399360

theorem four_digit_sum (a b c d : ℕ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 →
  (6 * (a + b + c + d) * 1111 = 73326) →
  ({a, b, c, d} : Set ℕ) = {1, 2, 3, 5} :=
by sorry

end NUMINAMATH_CALUDE_four_digit_sum_l3993_399360


namespace NUMINAMATH_CALUDE_stability_comparison_A_more_stable_than_B_l3993_399342

/-- Represents a set of data with its variance -/
structure DataSet where
  variance : ℝ
  variance_nonneg : 0 ≤ variance

/-- Defines stability comparison between two DataSets -/
def more_stable (a b : DataSet) : Prop :=
  a.variance < b.variance

theorem stability_comparison (a b : DataSet) :
  a.variance < b.variance → more_stable a b :=
by
  sorry

/-- Example datasets A and B -/
def A : DataSet := ⟨0.03, by norm_num⟩
def B : DataSet := ⟨0.13, by norm_num⟩

/-- Theorem stating that A is more stable than B -/
theorem A_more_stable_than_B : more_stable A B :=
by
  sorry

end NUMINAMATH_CALUDE_stability_comparison_A_more_stable_than_B_l3993_399342


namespace NUMINAMATH_CALUDE_range_of_m_l3993_399396

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 3*x - 10 ≤ 0}
def B (m : ℝ) : Set ℝ := {x : ℝ | m + 1 ≤ x ∧ x ≤ 2*m - 1}

-- State the theorem
theorem range_of_m (m : ℝ) : A ∩ B m = B m → m ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l3993_399396


namespace NUMINAMATH_CALUDE_other_number_proof_l3993_399365

theorem other_number_proof (a b : ℕ+) 
  (h1 : Nat.lcm a b = 2310)
  (h2 : Nat.gcd a b = 61)
  (h3 : a = 210) : 
  b = 671 := by
  sorry

end NUMINAMATH_CALUDE_other_number_proof_l3993_399365


namespace NUMINAMATH_CALUDE_largest_divisor_of_n_squared_div_72_l3993_399397

theorem largest_divisor_of_n_squared_div_72 (n : ℕ) (h : n > 0) (h_div : 72 ∣ n^2) :
  ∀ k : ℕ, k > 12 → ¬(∀ m : ℕ, m > 0 ∧ 72 ∣ m^2 → k ∣ m) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_n_squared_div_72_l3993_399397


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3993_399340

/-- Given an arithmetic sequence {a_n} where a_5 + a_6 + a_7 = 15, 
    prove that a_3 + a_4 + ... + a_9 equals 35. -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) : 
  (∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  a 5 + a 6 + a 7 = 15 →                                -- given condition
  a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 35 :=        -- conclusion to prove
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3993_399340


namespace NUMINAMATH_CALUDE_floor_paving_cost_l3993_399349

/-- The cost of paving a rectangular floor -/
theorem floor_paving_cost 
  (length : ℝ) 
  (width : ℝ) 
  (rate : ℝ) 
  (h1 : length = 10) 
  (h2 : width = 4.75) 
  (h3 : rate = 900) : 
  length * width * rate = 42750 := by
  sorry

end NUMINAMATH_CALUDE_floor_paving_cost_l3993_399349


namespace NUMINAMATH_CALUDE_gas_cost_proof_l3993_399343

/-- The original total cost of gas for a group of friends -/
def original_cost : ℝ := 200

/-- The number of friends initially -/
def initial_friends : ℕ := 5

/-- The number of additional friends who joined -/
def additional_friends : ℕ := 3

/-- The decrease in cost per person for the original friends -/
def cost_decrease : ℝ := 15

theorem gas_cost_proof :
  let total_friends := initial_friends + additional_friends
  let initial_cost_per_person := original_cost / initial_friends
  let final_cost_per_person := original_cost / total_friends
  initial_cost_per_person - final_cost_per_person = cost_decrease :=
by sorry

end NUMINAMATH_CALUDE_gas_cost_proof_l3993_399343


namespace NUMINAMATH_CALUDE_card_ratio_l3993_399304

theorem card_ratio (total : ℕ) (difference : ℕ) (ellis : ℕ) (orion : ℕ) : 
  total = 500 → 
  difference = 50 → 
  ellis = orion + difference → 
  total = ellis + orion → 
  (ellis : ℚ) / (orion : ℚ) = 11 / 9 := by
sorry

end NUMINAMATH_CALUDE_card_ratio_l3993_399304


namespace NUMINAMATH_CALUDE_complement_A_inter_B_l3993_399324

open Set Real

-- Define the universal set U as ℝ
def U : Set ℝ := univ

-- Define set A
def A : Set ℝ := {y | ∃ x, y = 3^x + 1}

-- Define set B
def B : Set ℝ := {x | log x < 0}

-- Statement to prove
theorem complement_A_inter_B : 
  (U \ A) ∩ B = {x | 0 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_complement_A_inter_B_l3993_399324


namespace NUMINAMATH_CALUDE_green_apples_count_l3993_399395

theorem green_apples_count (total : ℕ) (red_to_green_ratio : ℕ) 
  (h1 : total = 496) 
  (h2 : red_to_green_ratio = 3) : 
  ∃ green : ℕ, green = 124 ∧ total = green * (red_to_green_ratio + 1) :=
by sorry

end NUMINAMATH_CALUDE_green_apples_count_l3993_399395


namespace NUMINAMATH_CALUDE_arithmetic_sequences_equal_sum_l3993_399311

/-- Sum of the first n terms of an arithmetic sequence with first term a and common difference d -/
def arithmetic_sum (a d n : ℤ) : ℤ := n * (2 * a + (n - 1) * d) / 2

theorem arithmetic_sequences_equal_sum :
  ∃! (n : ℕ), n > 0 ∧ arithmetic_sum 5 4 n = arithmetic_sum 12 3 n :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequences_equal_sum_l3993_399311


namespace NUMINAMATH_CALUDE_interest_problem_l3993_399386

/-- Given a principal amount and an interest rate, prove that they satisfy the conditions for simple and compound interest over 2 years -/
theorem interest_problem (P R : ℝ) : 
  (P * R * 2 / 100 = 20) →  -- Simple interest condition
  (P * ((1 + R/100)^2 - 1) = 22) →  -- Compound interest condition
  (P = 50 ∧ R = 20) := by
sorry

end NUMINAMATH_CALUDE_interest_problem_l3993_399386


namespace NUMINAMATH_CALUDE_digit_puzzle_l3993_399392

theorem digit_puzzle :
  ∀ (A B C D E F G H M : ℕ),
    A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ D ≠ 0 ∧ E ≠ 0 ∧ F ≠ 0 ∧ G ≠ 0 ∧ H ≠ 0 ∧ M ≠ 0 →
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ A ≠ G ∧ A ≠ H ∧ A ≠ M →
    B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ B ≠ G ∧ B ≠ H ∧ B ≠ M →
    C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ C ≠ G ∧ C ≠ H ∧ C ≠ M →
    D ≠ E ∧ D ≠ F ∧ D ≠ G ∧ D ≠ H ∧ D ≠ M →
    E ≠ F ∧ E ≠ G ∧ E ≠ H ∧ E ≠ M →
    F ≠ G ∧ F ≠ H ∧ F ≠ M →
    G ≠ H ∧ G ≠ M →
    H ≠ M →
    A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧ E < 10 ∧ F < 10 ∧ G < 10 ∧ H < 10 ∧ M < 10 →
    A + B = 14 →
    M / G = M - F ∧ M - F = H - C →
    D * F = 24 →
    B + E = 16 →
    H = 4 :=
by sorry

end NUMINAMATH_CALUDE_digit_puzzle_l3993_399392


namespace NUMINAMATH_CALUDE_tangent_line_problem_l3993_399364

-- Define f as a real-valued function
variable (f : ℝ → ℝ)

-- State the theorem
theorem tangent_line_problem (h : ∀ y, y = f 2 → y = 2 + 4) : 
  f 2 + deriv f 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_problem_l3993_399364


namespace NUMINAMATH_CALUDE_polynomial_division_problem_l3993_399350

theorem polynomial_division_problem (x : ℝ) :
  let quotient := 2 * x + 6
  let divisor := x - 5
  let remainder := 2
  let polynomial := 2 * x^2 - 4 * x - 28
  polynomial = quotient * divisor + remainder := by sorry

end NUMINAMATH_CALUDE_polynomial_division_problem_l3993_399350


namespace NUMINAMATH_CALUDE_cube_difference_l3993_399333

theorem cube_difference (a b : ℝ) (h1 : a - b = 7) (h2 : a^2 + b^2 = 50) : 
  a^3 - b^3 = 353.5 := by sorry

end NUMINAMATH_CALUDE_cube_difference_l3993_399333


namespace NUMINAMATH_CALUDE_congruence_2023_mod_10_l3993_399314

theorem congruence_2023_mod_10 :
  ∀ n : ℤ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -2023 [ZMOD 10] → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_congruence_2023_mod_10_l3993_399314


namespace NUMINAMATH_CALUDE_perpendicular_condition_l3993_399387

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Condition for two lines to be perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- The first line: 2x - y - 1 = 0 -/
def line1 : Line := { a := 2, b := -1, c := -1 }

/-- The second line: mx + y + 1 = 0 -/
def line2 (m : ℝ) : Line := { a := m, b := 1, c := 1 }

/-- Theorem: The necessary and sufficient condition for perpendicularity -/
theorem perpendicular_condition :
  ∀ m : ℝ, perpendicular line1 (line2 m) ↔ m = 1/2 := by sorry

end NUMINAMATH_CALUDE_perpendicular_condition_l3993_399387


namespace NUMINAMATH_CALUDE_distance_between_points_l3993_399366

def point1 : ℝ × ℝ := (-3, 1)
def point2 : ℝ × ℝ := (4, -9)

theorem distance_between_points : 
  Real.sqrt ((point2.1 - point1.1)^2 + (point2.2 - point1.2)^2) = Real.sqrt 149 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l3993_399366


namespace NUMINAMATH_CALUDE_function_identity_l3993_399388

def is_divisible (a b : ℕ) : Prop := ∃ k : ℕ, b = k * a

theorem function_identity (f : ℕ+ → ℕ+) : 
  (∀ m n : ℕ+, is_divisible (m^2 + f n) (m * f m + n)) → 
  (∀ n : ℕ+, f n = n) :=
by sorry

end NUMINAMATH_CALUDE_function_identity_l3993_399388


namespace NUMINAMATH_CALUDE_diameter_difference_explanation_l3993_399336

/-- Represents the system of a polar bear walking on an ice floe -/
structure BearIceSystem where
  bear_mass : ℝ
  ice_mass : ℝ
  instrument_diameter : ℝ
  photo_diameter : ℝ

/-- The observed diameters differ due to relative motion -/
theorem diameter_difference_explanation (system : BearIceSystem)
  (h1 : system.instrument_diameter = 8.5)
  (h2 : system.photo_diameter = 9)
  (h3 : system.ice_mass > system.bear_mass)
  (h4 : system.ice_mass ≤ 100 * system.bear_mass) :
  ∃ (center_of_mass_shift : ℝ),
    center_of_mass_shift > 0 ∧
    center_of_mass_shift < 0.5 ∧
    system.photo_diameter = system.instrument_diameter + 2 * center_of_mass_shift :=
by sorry

#check diameter_difference_explanation

end NUMINAMATH_CALUDE_diameter_difference_explanation_l3993_399336


namespace NUMINAMATH_CALUDE_fraction_problem_l3993_399354

theorem fraction_problem : ∃ x : ℚ, x * 1206 = 3 * 134 ∧ x = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l3993_399354


namespace NUMINAMATH_CALUDE_ellipse_foci_coordinates_l3993_399377

theorem ellipse_foci_coordinates :
  let ellipse := fun (x y : ℝ) => x^2 / 25 + y^2 / 169 = 1
  let a := Real.sqrt 169
  let b := 5
  let c := Real.sqrt (a^2 - b^2)
  (∀ x y, ellipse x y ↔ x^2 / a^2 + y^2 / b^2 = 1) →
  (∀ x y, ellipse x y → x^2 / a^2 + y^2 / b^2 ≤ 1) →
  ({(0, c), (0, -c)} : Set (ℝ × ℝ)) = {p | ∃ x y, ellipse x y ∧ (x - p.1)^2 + (y - p.2)^2 = a^2} :=
by sorry

end NUMINAMATH_CALUDE_ellipse_foci_coordinates_l3993_399377


namespace NUMINAMATH_CALUDE_binomial_12_choose_10_l3993_399380

theorem binomial_12_choose_10 : Nat.choose 12 10 = 66 := by sorry

end NUMINAMATH_CALUDE_binomial_12_choose_10_l3993_399380


namespace NUMINAMATH_CALUDE_fraction_division_calculate_fraction_l3993_399378

theorem fraction_division (a b c d : ℚ) (hb : b ≠ 0) (hd : d ≠ 0) :
  a / (c / d) = (a * d) / c := by
  sorry

theorem calculate_fraction :
  7 / (9 / 14) = 98 / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_calculate_fraction_l3993_399378


namespace NUMINAMATH_CALUDE_problem_statement_l3993_399322

theorem problem_statement (x : ℝ) (h : x + 1/x = 7) : 
  (x - 3)^2 + 49/(x - 3)^2 = 23 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3993_399322


namespace NUMINAMATH_CALUDE_sqrt_nine_factorial_over_ninety_l3993_399389

theorem sqrt_nine_factorial_over_ninety : 
  Real.sqrt (Nat.factorial 9 / 90) = 24 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_nine_factorial_over_ninety_l3993_399389


namespace NUMINAMATH_CALUDE_fraction_simplification_l3993_399356

theorem fraction_simplification (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -3) :
  (3*x^2 + x) / ((x - 1) * (x + 3)) + (5 - x) / ((x - 1) * (x + 3)) =
  (3*x^2 + 4) / ((x - 1) * (x + 3)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3993_399356


namespace NUMINAMATH_CALUDE_equal_expressions_imply_abs_difference_l3993_399302

theorem equal_expressions_imply_abs_difference (x y : ℝ) :
  ((x + y = x - y ∧ x + y = x / y) ∨
   (x + y = x - y ∧ x + y = x * y) ∨
   (x + y = x / y ∧ x + y = x * y) ∨
   (x - y = x / y ∧ x - y = x * y) ∨
   (x - y = x / y ∧ x * y = x / y) ∨
   (x + y = x / y ∧ x - y = x / y)) →
  |y| - |x| = 1/2 := by
sorry

end NUMINAMATH_CALUDE_equal_expressions_imply_abs_difference_l3993_399302


namespace NUMINAMATH_CALUDE_f_positive_before_zero_point_l3993_399393

noncomputable def f (x : ℝ) : ℝ := (1/3)^x + Real.log x / Real.log (1/3)

theorem f_positive_before_zero_point (a x₀ : ℝ) 
  (h_zero : f a = 0) 
  (h_decreasing : ∀ x y, 0 < x → x < y → f y < f x) 
  (h_range : 0 < x₀ ∧ x₀ < a) : 
  f x₀ > 0 := by
  sorry

end NUMINAMATH_CALUDE_f_positive_before_zero_point_l3993_399393


namespace NUMINAMATH_CALUDE_escalator_length_is_210_l3993_399313

/-- The length of an escalator given its speed, a person's walking speed, and the time taken to cover the entire length. -/
def escalator_length (escalator_speed : ℝ) (person_speed : ℝ) (time : ℝ) : ℝ :=
  (escalator_speed + person_speed) * time

/-- Theorem stating that under the given conditions, the escalator length is 210 feet. -/
theorem escalator_length_is_210 :
  escalator_length 12 2 15 = 210 := by
  sorry

#eval escalator_length 12 2 15

end NUMINAMATH_CALUDE_escalator_length_is_210_l3993_399313


namespace NUMINAMATH_CALUDE_painting_theorem_l3993_399309

/-- Represents the portion of a wall painted in a given time -/
def paint_portion (rate : ℚ) (time : ℚ) : ℚ := rate * time

/-- The combined painting rate of two painters -/
def combined_rate (rate1 : ℚ) (rate2 : ℚ) : ℚ := rate1 + rate2

theorem painting_theorem (heidi_rate liam_rate : ℚ) 
  (h1 : heidi_rate = 1 / 60)
  (h2 : liam_rate = 1 / 90)
  (time : ℚ)
  (h3 : time = 15) :
  paint_portion (combined_rate heidi_rate liam_rate) time = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_painting_theorem_l3993_399309


namespace NUMINAMATH_CALUDE_intersection_product_range_l3993_399353

/-- Sphere S centered at origin with radius √6 -/
def S (x y z : ℝ) : Prop := x^2 + y^2 + z^2 = 6

/-- Plane α passing through (4, 0, 0), (0, 4, 0), (0, 0, 4) -/
def α (x y z : ℝ) : Prop := x + y + z = 4

theorem intersection_product_range :
  ∀ x y z : ℝ, S x y z → α x y z → 50/27 ≤ x*y*z ∧ x*y*z ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_product_range_l3993_399353


namespace NUMINAMATH_CALUDE_robotics_team_combinations_l3993_399375

def girls : ℕ := 4
def boys : ℕ := 7
def team_size : ℕ := 5
def min_girls : ℕ := 2

theorem robotics_team_combinations : 
  (Finset.sum (Finset.range (girls - min_girls + 1))
    (λ k => Nat.choose girls (k + min_girls) * Nat.choose boys (team_size - (k + min_girls)))) = 301 := by
  sorry

end NUMINAMATH_CALUDE_robotics_team_combinations_l3993_399375


namespace NUMINAMATH_CALUDE_triangle_side_length_l3993_399325

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  A = π / 4 →
  Real.sin A + Real.sin (B - C) = 2 * Real.sqrt 2 * Real.sin (2 * C) →
  (1 / 2) * b * c * Real.sin A = 1 →
  a = Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3993_399325


namespace NUMINAMATH_CALUDE_profit_percentage_previous_year_l3993_399361

theorem profit_percentage_previous_year 
  (revenue_previous : ℝ) 
  (profit_previous : ℝ) 
  (revenue_fall_rate : ℝ) 
  (profit_rate_current : ℝ) 
  (profit_increase_rate : ℝ)
  (h1 : revenue_fall_rate = 0.2)
  (h2 : profit_rate_current = 0.14)
  (h3 : profit_increase_rate = 1.1200000000000001)
  (h4 : profit_rate_current * (1 - revenue_fall_rate) * revenue_previous = profit_increase_rate * profit_previous) :
  profit_previous / revenue_previous = 0.1 := by
sorry

end NUMINAMATH_CALUDE_profit_percentage_previous_year_l3993_399361


namespace NUMINAMATH_CALUDE_fraction_multiplication_result_l3993_399317

theorem fraction_multiplication_result : 
  (5 / 8 : ℚ) * (7 / 12 : ℚ) * (3 / 7 : ℚ) * 1350 = 210.9375 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_result_l3993_399317


namespace NUMINAMATH_CALUDE_bernardo_wins_l3993_399331

def game_winner (M : ℕ) : Prop :=
  M ≤ 999 ∧
  3 * M < 1000 ∧
  3 * M + 100 < 1000 ∧
  3 * (3 * M + 100) < 1000 ∧
  3 * (3 * M + 100) + 100 ≥ 1000

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem bernardo_wins :
  ∃ M : ℕ, game_winner M ∧
    (∀ N : ℕ, N < M → ¬game_winner N) ∧
    M = 67 ∧
    sum_of_digits M = 13 := by
  sorry

end NUMINAMATH_CALUDE_bernardo_wins_l3993_399331


namespace NUMINAMATH_CALUDE_employee_hire_year_l3993_399372

/-- Rule of 70 retirement provision -/
def rule_of_70 (age : ℕ) (years_employed : ℕ) : Prop :=
  age + years_employed ≥ 70

/-- The year an employee was hired -/
def hire_year : ℕ := 1970

/-- The age at which the employee was hired -/
def hire_age : ℕ := 32

/-- The year the employee becomes eligible to retire -/
def retirement_year : ℕ := 2008

theorem employee_hire_year :
  rule_of_70 (hire_age + (retirement_year - hire_year)) (retirement_year - hire_year) ∧
  ∀ y, y > hire_year →
    ¬rule_of_70 (hire_age + (retirement_year - y)) (retirement_year - y) :=
sorry

end NUMINAMATH_CALUDE_employee_hire_year_l3993_399372


namespace NUMINAMATH_CALUDE_sphere_volume_ratio_l3993_399318

theorem sphere_volume_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) :
  (4 * π * r₁^2) / (4 * π * r₂^2) = 4 / 9 →
  ((4 / 3) * π * r₁^3) / ((4 / 3) * π * r₂^3) = 8 / 27 := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_ratio_l3993_399318


namespace NUMINAMATH_CALUDE_even_quadratic_function_l3993_399307

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem even_quadratic_function (a b : ℝ) :
  let f : ℝ → ℝ := fun x ↦ a * x^2 + (b - 3) * x + 3
  IsEven f ∧ (∀ x, x ∈ Set.Icc (a^2 - 2) a → f x ∈ Set.range f) →
  a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_even_quadratic_function_l3993_399307


namespace NUMINAMATH_CALUDE_least_integer_absolute_value_l3993_399346

theorem least_integer_absolute_value (x : ℤ) : 
  (∀ y : ℤ, |3 * y + 4| ≤ 21 → y ≥ x) → x = -8 ∧ |3 * x + 4| ≤ 21 := by
  sorry

end NUMINAMATH_CALUDE_least_integer_absolute_value_l3993_399346


namespace NUMINAMATH_CALUDE_subset_properties_l3993_399355

variable {α : Type*}
variable (A B : Set α)

theorem subset_properties (hAB : A ⊆ B) (hA : A.Nonempty) (hB : B.Nonempty) :
  (∀ x, x ∈ A → x ∈ B) ∧
  (∃ x, x ∈ B ∧ x ∉ A) ∧
  (∀ x, x ∉ B → x ∉ A) :=
by sorry

end NUMINAMATH_CALUDE_subset_properties_l3993_399355


namespace NUMINAMATH_CALUDE_odd_number_bound_l3993_399376

/-- Sum of digits in base 2 -/
def S₂ (n : ℕ) : ℕ := sorry

theorem odd_number_bound (K a b l m : ℕ) (hK_odd : Odd K) (hS₂K : S₂ K = 2)
  (hK_factor : K = a * b) (ha_pos : a > 1) (hb_pos : b > 1)
  (hl_pos : l > 2) (hm_pos : m > 2)
  (hS₂a : S₂ a < l) (hS₂b : S₂ b < m) : K ≤ 2^(l*m - 6) + 1 := by
  sorry

end NUMINAMATH_CALUDE_odd_number_bound_l3993_399376


namespace NUMINAMATH_CALUDE_problem_solution_l3993_399399

theorem problem_solution (p q : ℝ) (hp : p > 0) (hq : q > 0) 
  (h1 : 1/p + 1/q = 2) (h2 : p*q = 1) : p = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3993_399399


namespace NUMINAMATH_CALUDE_partial_fraction_sum_l3993_399330

theorem partial_fraction_sum (x : ℝ) (A B C D E F : ℝ) :
  (1 : ℝ) / (x * (x + 1) * (x + 2) * (x + 3) * (x + 4) * (x + 5)) =
  A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 4) + F / (x + 5) →
  A + B + C + D + E + F = 0 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_sum_l3993_399330


namespace NUMINAMATH_CALUDE_teachers_separation_probability_l3993_399381

/-- The number of students in the group photo arrangement. -/
def num_students : ℕ := 5

/-- The number of teachers in the group photo arrangement. -/
def num_teachers : ℕ := 2

/-- The total number of people in the group photo arrangement. -/
def total_people : ℕ := num_students + num_teachers

/-- The probability of arranging the group such that the two teachers
    are not at the ends and not adjacent to each other. -/
def probability_teachers_separated : ℚ :=
  (num_students.factorial * (num_students + 1).choose 2) / total_people.factorial

theorem teachers_separation_probability :
  probability_teachers_separated = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_teachers_separation_probability_l3993_399381


namespace NUMINAMATH_CALUDE_prism_with_18_edges_has_8_faces_l3993_399357

/-- A prism is a polyhedron with two congruent parallel faces (bases) and all other faces (lateral faces) are parallelograms. -/
structure Prism where
  edges : ℕ

/-- The number of faces in a prism. -/
def num_faces (p : Prism) : ℕ :=
  let lateral_faces := p.edges / 3
  lateral_faces + 2

theorem prism_with_18_edges_has_8_faces (p : Prism) (h : p.edges = 18) : num_faces p = 8 := by
  sorry

#check prism_with_18_edges_has_8_faces

end NUMINAMATH_CALUDE_prism_with_18_edges_has_8_faces_l3993_399357


namespace NUMINAMATH_CALUDE_quadratic_root_sum_product_l3993_399370

theorem quadratic_root_sum_product (p q : ℝ) : 
  (∃ x y : ℝ, 3 * x^2 - p * x + q = 0 ∧ 3 * y^2 - p * y + q = 0 ∧ x + y = 8 ∧ x * y = 12) → 
  p + q = 60 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_product_l3993_399370


namespace NUMINAMATH_CALUDE_calculate_daily_fine_l3993_399320

/-- Calculates the daily fine for absence given contract details -/
theorem calculate_daily_fine (total_days : ℕ) (daily_pay : ℚ) (absent_days : ℕ) (total_payment : ℚ) : 
  total_days = 30 →
  daily_pay = 25 →
  absent_days = 10 →
  total_payment = 425 →
  (total_days - absent_days) * daily_pay - absent_days * (daily_pay - total_payment / (total_days - absent_days)) = total_payment →
  daily_pay - total_payment / (total_days - absent_days) = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_calculate_daily_fine_l3993_399320


namespace NUMINAMATH_CALUDE_parabola_focus_point_slope_l3993_399368

/-- The slope of line AF for a parabola y² = 4x with focus F(1,0) and point A on the parabola -/
theorem parabola_focus_point_slope (A : ℝ × ℝ) : 
  A.1 > 0 → -- A is in the first quadrant
  A.2 > 0 →
  A.1 + 1 = 5 → -- distance from A to directrix x = -1 is 5
  A.2^2 = 4 * A.1 → -- A is on the parabola y² = 4x
  (A.2 - 0) / (A.1 - 1) = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_parabola_focus_point_slope_l3993_399368


namespace NUMINAMATH_CALUDE_alex_jane_pen_difference_l3993_399341

/-- Calculates the number of pens Alex has after a given number of weeks -/
def alex_pens (initial_pens : ℕ) (weeks : ℕ) : ℕ :=
  initial_pens * (2 ^ weeks)

/-- The number of pens Jane has after a month -/
def jane_pens : ℕ := 16

/-- The number of weeks in a month -/
def weeks_in_month : ℕ := 4

/-- The initial number of pens Alex has -/
def alex_initial_pens : ℕ := 4

theorem alex_jane_pen_difference :
  alex_pens alex_initial_pens weeks_in_month - jane_pens = 16 := by
  sorry


end NUMINAMATH_CALUDE_alex_jane_pen_difference_l3993_399341


namespace NUMINAMATH_CALUDE_carla_project_days_l3993_399363

/-- The number of days needed to complete a project given the number of items to collect and items collected per day. -/
def daysNeeded (leaves : ℕ) (bugs : ℕ) (itemsPerDay : ℕ) : ℕ :=
  (leaves + bugs) / itemsPerDay

/-- Theorem: Carla needs 10 days to complete the project. -/
theorem carla_project_days : daysNeeded 30 20 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_carla_project_days_l3993_399363


namespace NUMINAMATH_CALUDE_first_cube_weight_l3993_399359

/-- Given two cubical blocks of the same metal, where the sides of the second cube
    are twice as long as the first cube, and the second cube weighs 24 pounds,
    prove that the weight of the first cubical block is 3 pounds. -/
theorem first_cube_weight (s : ℝ) (weight : ℝ → ℝ) :
  (∀ x, weight (8 * x) = 8 * weight x) →  -- Weight is proportional to volume
  weight (8 * s^3) = 24 →                 -- Second cube weighs 24 pounds
  weight (s^3) = 3 :=
by sorry

end NUMINAMATH_CALUDE_first_cube_weight_l3993_399359


namespace NUMINAMATH_CALUDE_power_of_five_l3993_399321

theorem power_of_five (x : ℕ) : 121 * (5^x) = 75625 ↔ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_of_five_l3993_399321


namespace NUMINAMATH_CALUDE_jane_well_days_l3993_399382

/-- Represents Jane's performance levels --/
inductive Performance
  | Poor
  | Well
  | Excellent

/-- Returns the daily earnings based on performance --/
def dailyEarnings (p : Performance) : ℕ :=
  match p with
  | Performance.Poor => 2
  | Performance.Well => 4
  | Performance.Excellent => 6

/-- Represents Jane's work record over 15 days --/
structure WorkRecord :=
  (poorDays : ℕ)
  (wellDays : ℕ)
  (excellentDays : ℕ)
  (total_days : poorDays + wellDays + excellentDays = 15)
  (excellent_poor_relation : excellentDays = poorDays + 4)
  (total_earnings : poorDays * 2 + wellDays * 4 + excellentDays * 6 = 66)

/-- Theorem stating that Jane performed well for 11 days --/
theorem jane_well_days (record : WorkRecord) : record.wellDays = 11 := by
  sorry

end NUMINAMATH_CALUDE_jane_well_days_l3993_399382


namespace NUMINAMATH_CALUDE_single_elimination_tournament_matches_l3993_399310

/-- Represents a single-elimination tournament. -/
structure Tournament where
  teams : ℕ
  matches_played : ℕ

/-- The number of teams eliminated in a single-elimination tournament. -/
def eliminated_teams (t : Tournament) : ℕ := t.matches_played

/-- A tournament is complete when there is only one team remaining. -/
def is_complete (t : Tournament) : Prop := t.teams - eliminated_teams t = 1

theorem single_elimination_tournament_matches (t : Tournament) :
  t.teams = 128 → is_complete t → t.matches_played = 127 := by
  sorry

end NUMINAMATH_CALUDE_single_elimination_tournament_matches_l3993_399310


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3993_399308

theorem sufficient_not_necessary : 
  (∀ x : ℝ, x > 2 → x^2 + 2*x - 8 > 0) ∧ 
  (∃ x : ℝ, x ≤ 2 ∧ x^2 + 2*x - 8 > 0) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3993_399308


namespace NUMINAMATH_CALUDE_divisibility_proof_l3993_399327

theorem divisibility_proof (n : ℕ) : 
  n = 6268440 → n % 8 = 0 ∧ n % 66570 = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_proof_l3993_399327


namespace NUMINAMATH_CALUDE_gcd_count_for_product_360_l3993_399369

theorem gcd_count_for_product_360 (a b : ℕ+) : 
  (Nat.gcd a b * Nat.lcm a b = 360) → 
  (∃ (S : Finset ℕ+), S.card = 11 ∧ (∀ x, x ∈ S ↔ ∃ c d : ℕ+, (Nat.gcd c d * Nat.lcm c d = 360) ∧ Nat.gcd c d = x)) :=
sorry

end NUMINAMATH_CALUDE_gcd_count_for_product_360_l3993_399369


namespace NUMINAMATH_CALUDE_student_weight_replacement_l3993_399339

theorem student_weight_replacement (W : ℝ) :
  (W - 12) / 5 = 12 →
  W = 72 := by
sorry

end NUMINAMATH_CALUDE_student_weight_replacement_l3993_399339


namespace NUMINAMATH_CALUDE_slips_with_three_l3993_399328

/-- Given a bag of 20 slips with numbers 3 or 8, prove the number of 3s when expected value is 6 -/
theorem slips_with_three (total : ℕ) (value_one value_two : ℕ) (expected_value : ℚ) : 
  total = 20 →
  value_one = 3 →
  value_two = 8 →
  expected_value = 6 →
  ∃ (num_value_one : ℕ),
    num_value_one ≤ total ∧
    (num_value_one : ℚ) / total * value_one + (total - num_value_one : ℚ) / total * value_two = expected_value ∧
    num_value_one = 8 :=
by sorry

end NUMINAMATH_CALUDE_slips_with_three_l3993_399328


namespace NUMINAMATH_CALUDE_cube_diff_even_iff_sum_even_l3993_399301

theorem cube_diff_even_iff_sum_even (p q : ℕ) : 
  Even (p^3 - q^3) ↔ Even (p + q) :=
by sorry

end NUMINAMATH_CALUDE_cube_diff_even_iff_sum_even_l3993_399301


namespace NUMINAMATH_CALUDE_hexadecagon_diagonals_l3993_399383

/-- Number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A hexadecagon is a 16-sided polygon -/
def hexadecagon_sides : ℕ := 16

theorem hexadecagon_diagonals :
  num_diagonals hexadecagon_sides = 104 := by sorry

end NUMINAMATH_CALUDE_hexadecagon_diagonals_l3993_399383


namespace NUMINAMATH_CALUDE_conditional_statements_requirement_l3993_399335

-- Define a type for the problems
inductive Problem
| AbsoluteValue
| CubeVolume
| PiecewiseFunction

-- Define a function to check if a problem requires conditional statements
def requiresConditionalStatements (p : Problem) : Prop :=
  match p with
  | Problem.AbsoluteValue => true
  | Problem.CubeVolume => false
  | Problem.PiecewiseFunction => true

-- Theorem statement
theorem conditional_statements_requirement :
  (requiresConditionalStatements Problem.AbsoluteValue ∧
   requiresConditionalStatements Problem.PiecewiseFunction) ∧
  ¬requiresConditionalStatements Problem.CubeVolume := by
  sorry


end NUMINAMATH_CALUDE_conditional_statements_requirement_l3993_399335


namespace NUMINAMATH_CALUDE_units_digit_of_7_cubed_l3993_399345

theorem units_digit_of_7_cubed (n : ℕ) : n = 7^3 → n % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_7_cubed_l3993_399345


namespace NUMINAMATH_CALUDE_roses_unchanged_l3993_399332

/-- Represents the number of flowers in a vase -/
structure FlowerVase where
  roses : ℕ
  orchids : ℕ

/-- The initial state of the flower vase -/
def initial_vase : FlowerVase := { roses := 13, orchids := 84 }

/-- The final state of the flower vase -/
def final_vase : FlowerVase := { roses := 13, orchids := 91 }

/-- Theorem stating that the number of roses remains unchanged -/
theorem roses_unchanged (initial : FlowerVase) (final : FlowerVase) 
  (h_initial : initial = initial_vase) 
  (h_final_orchids : final.orchids = 91) :
  final.roses = initial.roses := by sorry

end NUMINAMATH_CALUDE_roses_unchanged_l3993_399332


namespace NUMINAMATH_CALUDE_largest_area_is_16_l3993_399398

/-- Represents a polygon made of squares and right triangles -/
structure Polygon where
  num_squares : Nat
  num_triangles : Nat

/-- Calculates the area of a polygon -/
def area (p : Polygon) : ℝ :=
  4 * p.num_squares + 2 * p.num_triangles

/-- The set of all possible polygons in our problem -/
def polygon_set : Set Polygon :=
  { p | p.num_squares + p.num_triangles ≤ 4 }

theorem largest_area_is_16 :
  ∃ (p : Polygon), p ∈ polygon_set ∧ area p = 16 ∧ ∀ (q : Polygon), q ∈ polygon_set → area q ≤ 16 := by
  sorry

end NUMINAMATH_CALUDE_largest_area_is_16_l3993_399398


namespace NUMINAMATH_CALUDE_tangent_line_equation_l3993_399374

/-- The equation of the tangent line to y = (3x - 2x^3) / 3 at x = 1 is y = -x + 4/3 -/
theorem tangent_line_equation (x y : ℝ) :
  let f : ℝ → ℝ := λ x => (3*x - 2*x^3) / 3
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let f' : ℝ → ℝ := λ x => 1 - 2*x^2
  y = -x + 4/3 ↔ y - y₀ = f' x₀ * (x - x₀) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l3993_399374


namespace NUMINAMATH_CALUDE_m_geq_1_necessary_not_sufficient_l3993_399358

def p (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 4*x + 2*m ≥ 0

theorem m_geq_1_necessary_not_sufficient :
  (∀ m : ℝ, p m → m ≥ 1) ∧
  ¬(∀ m : ℝ, m ≥ 1 → p m) :=
sorry

end NUMINAMATH_CALUDE_m_geq_1_necessary_not_sufficient_l3993_399358


namespace NUMINAMATH_CALUDE_job_completion_time_l3993_399348

/-- The number of days it takes for a given number of machines to complete a job -/
def days_to_complete (num_machines : ℕ) : ℝ := sorry

/-- The rate at which each machine works (jobs per day) -/
def machine_rate : ℝ := sorry

theorem job_completion_time :
  -- Five machines working at the same rate
  (days_to_complete 5 * 5 * machine_rate = 1) →
  -- Ten machines can complete the job in 10 days
  (10 * 10 * machine_rate = 1) →
  -- The initial five machines take 20 days to complete the job
  days_to_complete 5 = 20 := by sorry

end NUMINAMATH_CALUDE_job_completion_time_l3993_399348


namespace NUMINAMATH_CALUDE_inequality_relationship_l3993_399319

theorem inequality_relationship (a : ℝ) (h : a^2 + a < 0) :
  -a > a^2 ∧ a^2 > -a^2 ∧ -a^2 > a := by sorry

end NUMINAMATH_CALUDE_inequality_relationship_l3993_399319


namespace NUMINAMATH_CALUDE_power_of_two_equality_unique_exponent_l3993_399338

theorem power_of_two_equality : 32^3 * 4^3 = 2^21 := by sorry

theorem unique_exponent (h : 32^3 * 4^3 = 2^J) : J = 21 := by sorry

end NUMINAMATH_CALUDE_power_of_two_equality_unique_exponent_l3993_399338


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l3993_399347

theorem decimal_to_fraction : 
  (3.56 : ℚ) = 89 / 25 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l3993_399347


namespace NUMINAMATH_CALUDE_symmetric_circle_correct_l3993_399379

-- Define the original circle
def original_circle (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 3)^2 = 1

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop :=
  x + y - 1 = 0

-- Define the symmetric circle
def symmetric_circle (x y : ℝ) : Prop :=
  (x + 2)^2 + (y + 1)^2 = 1

-- Theorem stating that the symmetric circle is correct
theorem symmetric_circle_correct :
  ∀ (x y : ℝ), 
    (∃ (x₀ y₀ : ℝ), original_circle x₀ y₀ ∧ 
      (∀ (x' y' : ℝ), symmetry_line ((x + x₀)/2) ((y + y₀)/2) → 
        (x - x')^2 + (y - y')^2 = (x₀ - x')^2 + (y₀ - y')^2)) →
    symmetric_circle x y :=
by sorry

end NUMINAMATH_CALUDE_symmetric_circle_correct_l3993_399379


namespace NUMINAMATH_CALUDE_part_one_part_two_l3993_399306

-- Define polynomials A and B
def A (x y : ℝ) : ℝ := x^2 + x*y + 3*y
def B (x y : ℝ) : ℝ := x^2 - x*y

-- Part 1
theorem part_one (x y : ℝ) : (x - 2)^2 + |y + 5| = 0 → 2 * A x y - B x y = -56 := by
  sorry

-- Part 2
theorem part_two (x : ℝ) : (∀ y : ℝ, ∃ c : ℝ, 2 * A x y - B x y = c) ↔ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3993_399306
