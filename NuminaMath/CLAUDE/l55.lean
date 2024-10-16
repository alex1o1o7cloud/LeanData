import Mathlib

namespace NUMINAMATH_CALUDE_annual_increase_y_l55_5549

/-- The annual increase in price of commodity Y -/
def y : ℝ := sorry

/-- The price of commodity X in a given year -/
def price_x (year : ℕ) : ℝ :=
  4.20 + 0.30 * (year - 2001)

/-- The price of commodity Y in a given year -/
def price_y (year : ℕ) : ℝ :=
  4.40 + y * (year - 2001)

theorem annual_increase_y : y = 0.20 :=
  have h1 : price_x 2010 = price_y 2010 + 0.70 := by sorry
  sorry

end NUMINAMATH_CALUDE_annual_increase_y_l55_5549


namespace NUMINAMATH_CALUDE_jeremys_songs_l55_5528

theorem jeremys_songs (songs_yesterday songs_today total_songs : ℕ) : 
  songs_yesterday < songs_today →
  songs_yesterday = 9 →
  total_songs = 23 →
  songs_yesterday + songs_today = total_songs →
  songs_today = 14 := by
  sorry

end NUMINAMATH_CALUDE_jeremys_songs_l55_5528


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l55_5544

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x => x * (x - 3) - (x - 3)
  ∃ x₁ x₂ : ℝ, x₁ = 3 ∧ x₂ = 1 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l55_5544


namespace NUMINAMATH_CALUDE_power_difference_evaluation_l55_5578

theorem power_difference_evaluation : (3^4)^3 - (4^3)^4 = -16245775 := by sorry

end NUMINAMATH_CALUDE_power_difference_evaluation_l55_5578


namespace NUMINAMATH_CALUDE_dans_minimum_speed_l55_5545

/-- Proves that Dan must travel at a speed greater than 48 miles per hour to arrive in city B before Cara. -/
theorem dans_minimum_speed (distance : ℝ) (cara_speed : ℝ) (dan_delay : ℝ) : 
  distance = 120 → 
  cara_speed = 30 → 
  dan_delay = 1.5 → 
  ∀ dan_speed : ℝ, dan_speed > 48 → distance / dan_speed < distance / cara_speed - dan_delay := by
  sorry

#check dans_minimum_speed

end NUMINAMATH_CALUDE_dans_minimum_speed_l55_5545


namespace NUMINAMATH_CALUDE_remaining_eggs_eggs_after_three_days_l55_5503

/-- Calculates the remaining eggs after consumption --/
theorem remaining_eggs (initial : ℕ) (consumed : ℕ) (h : initial ≥ consumed) : 
  initial - consumed = 75 - 49 → initial - consumed = 26 := by
  sorry

/-- Proves that 26 eggs remain after 3 days --/
theorem eggs_after_three_days : 
  ∃ (initial consumed : ℕ), initial = 75 ∧ consumed = 49 ∧ initial - consumed = 26 := by
  sorry

end NUMINAMATH_CALUDE_remaining_eggs_eggs_after_three_days_l55_5503


namespace NUMINAMATH_CALUDE_sqrt6_custom_op_approx_l55_5535

/-- Custom binary operation ¤ -/
def custom_op (x y : ℝ) : ℝ := x^2 + y^2 + 12

/-- Theorem stating that √6 ¤ √6 ≈ 23.999999999999996 -/
theorem sqrt6_custom_op_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1e-14 ∧ |custom_op (Real.sqrt 6) (Real.sqrt 6) - 23.999999999999996| < ε :=
sorry

end NUMINAMATH_CALUDE_sqrt6_custom_op_approx_l55_5535


namespace NUMINAMATH_CALUDE_no_prime_multiple_chain_l55_5534

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def primes_1_to_12 : Set ℕ := {n : ℕ | is_prime n ∧ n ≤ 12}

theorem no_prime_multiple_chain :
  ∀ a b c : ℕ, a ∈ primes_1_to_12 → b ∈ primes_1_to_12 → c ∈ primes_1_to_12 →
  a ≠ b → b ≠ c → a ≠ c →
  ¬(a ∣ b ∧ b ∣ c) :=
sorry

end NUMINAMATH_CALUDE_no_prime_multiple_chain_l55_5534


namespace NUMINAMATH_CALUDE_shipping_cost_for_five_pounds_l55_5567

/-- Calculates the shipping cost based on weight and rates -/
def shipping_cost (flat_fee : ℝ) (per_pound_rate : ℝ) (weight : ℝ) : ℝ :=
  flat_fee + per_pound_rate * weight

/-- Proves that the shipping cost for a 5-pound package is $9.00 -/
theorem shipping_cost_for_five_pounds :
  shipping_cost 5 0.8 5 = 9 := by
  sorry

end NUMINAMATH_CALUDE_shipping_cost_for_five_pounds_l55_5567


namespace NUMINAMATH_CALUDE_critical_points_product_bound_l55_5518

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x * Real.log x - (1/2) * m * x^2 - x

theorem critical_points_product_bound (m : ℝ) (x₁ x₂ : ℝ) :
  x₁ < x₂ →
  (∃ (y : ℝ), y ∈ Set.Icc x₁ x₂ ∧ (deriv (f m)) y = 0) →
  (deriv (f m)) x₁ = 0 →
  (deriv (f m)) x₂ = 0 →
  x₁ * x₂ > Real.exp 2 := by
sorry

end NUMINAMATH_CALUDE_critical_points_product_bound_l55_5518


namespace NUMINAMATH_CALUDE_skew_lines_iff_b_neq_two_sevenths_l55_5566

def line1 (b t : ℝ) : ℝ × ℝ × ℝ := (2 + 3*t, 1 + 4*t, b + 2*t)
def line2 (u : ℝ) : ℝ × ℝ × ℝ := (5 + u, 3 - u, 2 + 2*u)

def are_skew (b : ℝ) : Prop :=
  ∀ t u : ℝ, line1 b t ≠ line2 u

theorem skew_lines_iff_b_neq_two_sevenths (b : ℝ) :
  are_skew b ↔ b ≠ 2/7 :=
sorry

end NUMINAMATH_CALUDE_skew_lines_iff_b_neq_two_sevenths_l55_5566


namespace NUMINAMATH_CALUDE_negation_of_proposition_l55_5511

theorem negation_of_proposition (x y : ℝ) : 
  ¬(x > 0 ∧ y > 0 → x * y > 0) ↔ ((x ≤ 0 ∨ y ≤ 0) → x * y ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l55_5511


namespace NUMINAMATH_CALUDE_trig_identity_l55_5577

theorem trig_identity : (Real.cos (10 * π / 180) - 2 * Real.sin (20 * π / 180)) / Real.sin (10 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l55_5577


namespace NUMINAMATH_CALUDE_complement_intersection_problem_l55_5572

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {3, 4}

theorem complement_intersection_problem :
  (U \ A) ∩ B = {4} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_problem_l55_5572


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_eight_l55_5556

theorem sqrt_sum_equals_eight :
  Real.sqrt (18 - 8 * Real.sqrt 2) + Real.sqrt (18 + 8 * Real.sqrt 2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_eight_l55_5556


namespace NUMINAMATH_CALUDE_wilmas_garden_red_flowers_l55_5504

/-- Wilma's Garden Flower Count Theorem -/
theorem wilmas_garden_red_flowers :
  let total_flowers : ℕ := 6 * 13
  let yellow_flowers : ℕ := 12
  let green_flowers : ℕ := 2 * yellow_flowers
  let red_flowers : ℕ := total_flowers - (yellow_flowers + green_flowers)
  red_flowers = 42 := by sorry

end NUMINAMATH_CALUDE_wilmas_garden_red_flowers_l55_5504


namespace NUMINAMATH_CALUDE_four_Z_three_l55_5587

def Z (a b : ℤ) : ℤ := a^2 - 3*a*b + b^2

theorem four_Z_three : Z 4 3 = -11 := by
  sorry

end NUMINAMATH_CALUDE_four_Z_three_l55_5587


namespace NUMINAMATH_CALUDE_base5_324_equals_binary_1011001_l55_5569

/-- Converts a base-5 number to decimal --/
def base5ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Converts a decimal number to binary --/
def decimalToBinary (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec toBinary (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc
    else toBinary (m / 2) ((m % 2) :: acc)
  toBinary n []

/-- Theorem: The base-5 number 324₍₅₎ is equal to the binary number 1011001₍₂₎ --/
theorem base5_324_equals_binary_1011001 :
  decimalToBinary (base5ToDecimal [4, 2, 3]) = [1, 0, 1, 1, 0, 0, 1] := by
  sorry


end NUMINAMATH_CALUDE_base5_324_equals_binary_1011001_l55_5569


namespace NUMINAMATH_CALUDE_solution_values_solution_set_when_a_negative_l55_5582

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + x - a

-- Define the solution set condition
def hasSolutionSet (a b : ℝ) : Prop :=
  ∀ x, f a x < b ↔ x < -1 ∨ x > 3

-- Theorem statement
theorem solution_values (a b : ℝ) (h : hasSolutionSet a b) : a = -1/2 ∧ b = -1 := by
  sorry

-- Additional theorem for part 2
theorem solution_set_when_a_negative (a : ℝ) (h : a < 0) :
  (∀ x, f a x > 1 ↔ 
    (a < -1/2 ∧ -((a+1)/a) < x ∧ x < 1) ∨
    (a = -1/2 ∧ False) ∨
    (-1/2 < a ∧ a < 0 ∧ 1 < x ∧ x < -((a+1)/a))) := by
  sorry

end NUMINAMATH_CALUDE_solution_values_solution_set_when_a_negative_l55_5582


namespace NUMINAMATH_CALUDE_sqrt_x_minus_one_meaningful_l55_5575

theorem sqrt_x_minus_one_meaningful (x : ℝ) : x = 2 → ∃ y : ℝ, y ^ 2 = x - 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_one_meaningful_l55_5575


namespace NUMINAMATH_CALUDE_A_intersect_B_l55_5597

def A : Set ℕ := {1, 2, 4, 6, 8}

def B : Set ℕ := {x | ∃ k ∈ A, x = 2 * k}

theorem A_intersect_B : A ∩ B = {2, 4, 8} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_l55_5597


namespace NUMINAMATH_CALUDE_certification_cost_coverage_percentage_l55_5593

/-- Calculates the percentage of certification cost covered by insurance for a seeing-eye dog. -/
theorem certification_cost_coverage_percentage
  (adoption_fee : ℕ)
  (training_cost_per_week : ℕ)
  (training_weeks : ℕ)
  (certification_cost : ℕ)
  (total_out_of_pocket : ℕ)
  (h1 : adoption_fee = 150)
  (h2 : training_cost_per_week = 250)
  (h3 : training_weeks = 12)
  (h4 : certification_cost = 3000)
  (h5 : total_out_of_pocket = 3450) :
  (100 * (certification_cost - (total_out_of_pocket - adoption_fee - training_cost_per_week * training_weeks))) / certification_cost = 90 :=
by sorry

end NUMINAMATH_CALUDE_certification_cost_coverage_percentage_l55_5593


namespace NUMINAMATH_CALUDE_merchant_articles_count_l55_5554

theorem merchant_articles_count (N : ℕ) (CP SP : ℝ) : 
  N > 0 → 
  CP > 0 →
  N * CP = 15 * SP → 
  SP = CP * (1 + 33.33 / 100) → 
  N = 20 := by
sorry

end NUMINAMATH_CALUDE_merchant_articles_count_l55_5554


namespace NUMINAMATH_CALUDE_simplify_exponential_expression_l55_5509

theorem simplify_exponential_expression (a : ℝ) (h : a ≠ 0) :
  (a^9 * a^15) / a^3 = a^21 := by
  sorry

end NUMINAMATH_CALUDE_simplify_exponential_expression_l55_5509


namespace NUMINAMATH_CALUDE_sqrt_two_times_sqrt_eight_equals_four_l55_5515

theorem sqrt_two_times_sqrt_eight_equals_four :
  Real.sqrt 2 * Real.sqrt 8 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_times_sqrt_eight_equals_four_l55_5515


namespace NUMINAMATH_CALUDE_set_problem_l55_5574

theorem set_problem (U A B : Finset ℕ) (h1 : U.card = 190) (h2 : B.card = 49)
  (h3 : (U \ (A ∪ B)).card = 59) (h4 : (A ∩ B).card = 23) :
  A.card = 105 := by
  sorry

end NUMINAMATH_CALUDE_set_problem_l55_5574


namespace NUMINAMATH_CALUDE_no_geometric_sequence_sqrt235_l55_5580

theorem no_geometric_sequence_sqrt235 :
  ¬∃ (m n : ℕ) (q : ℝ), m > n ∧ n > 1 ∧ q > 0 ∧
    Real.sqrt 3 = q ^ n * Real.sqrt 2 ∧
    Real.sqrt 5 = q ^ m * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_no_geometric_sequence_sqrt235_l55_5580


namespace NUMINAMATH_CALUDE_sequence_sum_l55_5547

/-- Given a sequence {a_n} with a₁ = 1 and S_{n+1} = ((n+1)a_n)/n + S_n, 
    prove that S_n = n(n+1)/2 for all positive integers n. -/
theorem sequence_sum (a : ℕ → ℕ) (S : ℕ → ℕ) : 
  a 1 = 1 → 
  (∀ n : ℕ, n > 0 → S (n + 1) = ((n + 1) * a n) / n + S n) → 
  (∀ n : ℕ, n > 0 → S n = n * (n + 1) / 2) := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_l55_5547


namespace NUMINAMATH_CALUDE_tailor_trim_problem_l55_5565

theorem tailor_trim_problem (original_side : ℝ) (trimmed_other_side : ℝ) (remaining_area : ℝ) 
  (h1 : original_side = 18)
  (h2 : trimmed_other_side = 3)
  (h3 : remaining_area = 120) :
  ∃ x : ℝ, x = 10 ∧ (original_side - x) * (original_side - trimmed_other_side) = remaining_area :=
by
  sorry

end NUMINAMATH_CALUDE_tailor_trim_problem_l55_5565


namespace NUMINAMATH_CALUDE_charitable_gentleman_proof_l55_5573

def charitable_donation (initial : ℕ) : Prop :=
  let after_first := initial - (initial / 2 + 1)
  let after_second := after_first - (after_first / 2 + 2)
  let after_third := after_second - (after_second / 2 + 3)
  after_third = 1

theorem charitable_gentleman_proof :
  ∃ (initial : ℕ), charitable_donation initial ∧ initial = 42 := by
  sorry

end NUMINAMATH_CALUDE_charitable_gentleman_proof_l55_5573


namespace NUMINAMATH_CALUDE_divisor_sum_totient_inequality_divisor_sum_totient_equality_l55_5568

/-- Sum of divisors function -/
def sigma (n : ℕ) : ℕ := sorry

/-- Euler's totient function -/
def phi (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem divisor_sum_totient_inequality (n : ℕ) :
  1 / (phi n : ℝ) + 1 / (sigma n : ℝ) ≥ 2 / n :=
sorry

/-- Characterization of the equality case -/
theorem divisor_sum_totient_equality (n : ℕ) :
  (1 / (phi n : ℝ) + 1 / (sigma n : ℝ) = 2 / n) ↔ n = 1 :=
sorry

end NUMINAMATH_CALUDE_divisor_sum_totient_inequality_divisor_sum_totient_equality_l55_5568


namespace NUMINAMATH_CALUDE_white_balls_count_l55_5502

theorem white_balls_count (total : ℕ) (yellow_probability : ℚ) : 
  total = 20 → yellow_probability = 3/5 → total - (total * yellow_probability).num = 8 := by
  sorry

end NUMINAMATH_CALUDE_white_balls_count_l55_5502


namespace NUMINAMATH_CALUDE_total_participants_grandmasters_top_positions_l55_5507

/-- A round-robin chess tournament with grandmasters and masters -/
structure ChessTournament where
  num_grandmasters : ℕ
  num_masters : ℕ
  total_points_grandmasters : ℕ
  total_points_masters : ℕ

/-- The conditions of the tournament -/
def tournament_conditions (t : ChessTournament) : Prop :=
  t.num_masters = 3 * t.num_grandmasters ∧
  t.total_points_masters = (12 * t.total_points_grandmasters) / 10 ∧
  t.total_points_grandmasters + t.total_points_masters = (t.num_grandmasters + t.num_masters) * (t.num_grandmasters + t.num_masters - 1)

/-- The theorem stating the total number of participants -/
theorem total_participants (t : ChessTournament) (h : tournament_conditions t) : 
  t.num_grandmasters + t.num_masters = 12 := by
  sorry

/-- The theorem stating that grandmasters took the top positions -/
theorem grandmasters_top_positions (t : ChessTournament) (h : tournament_conditions t) : 
  t.num_grandmasters ≤ 3 ∧ t.num_grandmasters > 0 := by
  sorry

end NUMINAMATH_CALUDE_total_participants_grandmasters_top_positions_l55_5507


namespace NUMINAMATH_CALUDE_negative_three_times_inequality_l55_5559

theorem negative_three_times_inequality (a b : ℝ) (h : a < b) : -3 * a > -3 * b := by
  sorry

end NUMINAMATH_CALUDE_negative_three_times_inequality_l55_5559


namespace NUMINAMATH_CALUDE_min_value_expression_l55_5550

theorem min_value_expression (x y : ℝ) (hx : x > 1) (hy : y > 1) (hxy : x * y = 4) :
  (x^3 / (y - 1)) + (y^3 / (x - 1)) ≥ 16 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l55_5550


namespace NUMINAMATH_CALUDE_area_under_curve_l55_5523

-- Define the curve
def f (x : ℝ) : ℝ := x^2 + 1

-- Define the bounds of integration
def a : ℝ := 0
def b : ℝ := 1

-- State the theorem
theorem area_under_curve : ∫ x in a..b, f x = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_area_under_curve_l55_5523


namespace NUMINAMATH_CALUDE_joyce_apples_l55_5590

theorem joyce_apples (initial : ℕ) (given : ℕ) (remaining : ℕ) : 
  initial = 75 → given = 52 → remaining = initial - given → remaining = 23 := by
sorry

end NUMINAMATH_CALUDE_joyce_apples_l55_5590


namespace NUMINAMATH_CALUDE_average_marks_proof_l55_5589

theorem average_marks_proof (M P C : ℕ) (h1 : M + P = 60) (h2 : C = P + 20) :
  (M + C) / 2 = 40 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_proof_l55_5589


namespace NUMINAMATH_CALUDE_union_when_m_is_neg_one_subset_iff_m_leq_neg_two_disjoint_iff_m_geq_zero_l55_5543

-- Define the sets A and B
def A : Set ℝ := {x | 1 < x ∧ x < 3}
def B (m : ℝ) : Set ℝ := {x | 2*m < x ∧ x < 1-m}

-- Statement 1
theorem union_when_m_is_neg_one : 
  A ∪ B (-1) = {x : ℝ | -2 < x ∧ x < 3} := by sorry

-- Statement 2
theorem subset_iff_m_leq_neg_two :
  ∀ m : ℝ, A ⊆ B m ↔ m ≤ -2 := by sorry

-- Statement 3
theorem disjoint_iff_m_geq_zero :
  ∀ m : ℝ, A ∩ B m = ∅ ↔ m ≥ 0 := by sorry

end NUMINAMATH_CALUDE_union_when_m_is_neg_one_subset_iff_m_leq_neg_two_disjoint_iff_m_geq_zero_l55_5543


namespace NUMINAMATH_CALUDE_maple_trees_planted_proof_l55_5592

/-- The number of maple trees planted in a park -/
def maple_trees_planted (initial : ℕ) (final : ℕ) : ℕ := final - initial

/-- Theorem stating that 11 maple trees were planted -/
theorem maple_trees_planted_proof :
  let initial_trees : ℕ := 53
  let final_trees : ℕ := 64
  maple_trees_planted initial_trees final_trees = 11 := by
  sorry

end NUMINAMATH_CALUDE_maple_trees_planted_proof_l55_5592


namespace NUMINAMATH_CALUDE_distance_equation_l55_5546

/-- The distance between the boy's house and school -/
def D : ℝ := sorry

/-- The speed from house to library (km/hr) -/
def speed_to_library : ℝ := 3

/-- The speed from library to school (km/hr) -/
def speed_library_to_school : ℝ := 2.5

/-- The speed from school to house (km/hr) -/
def speed_return : ℝ := 2

/-- The time spent at the library (hours) -/
def library_time : ℝ := 0.5

/-- The total trip time (hours) -/
def total_time : ℝ := 5.5

theorem distance_equation : 
  (D / 2) / speed_to_library + library_time + 
  (D / 2) / speed_library_to_school + 
  D / speed_return = total_time := by sorry

end NUMINAMATH_CALUDE_distance_equation_l55_5546


namespace NUMINAMATH_CALUDE_product_zero_l55_5570

theorem product_zero (b : ℤ) (h : b = 3) : 
  (b - 12) * (b - 11) * (b - 10) * (b - 9) * (b - 8) * (b - 7) * (b - 6) * (b - 5) * (b - 4) * (b - 3) * (b - 2) * (b - 1) * b = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_zero_l55_5570


namespace NUMINAMATH_CALUDE_ellipse_tangent_product_l55_5533

/-- An ellipse with its key points -/
structure Ellipse where
  A : ℝ × ℝ  -- Major axis endpoint
  B : ℝ × ℝ  -- Minor axis endpoint
  F₁ : ℝ × ℝ  -- Focus 1
  F₂ : ℝ × ℝ  -- Focus 2

/-- Vector dot product -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- Vector from point to point -/
def vector (p q : ℝ × ℝ) : ℝ × ℝ := (q.1 - p.1, q.2 - p.2)

/-- Tangent of angle between three points -/
noncomputable def tan_angle (p q r : ℝ × ℝ) : ℝ :=
  let v1 := vector q p
  let v2 := vector q r
  (v1.2 * v2.1 - v1.1 * v2.2) / (v1.1 * v2.1 + v1.2 * v2.2)

/-- Main theorem -/
theorem ellipse_tangent_product (Γ : Ellipse) 
  (h : dot_product (vector Γ.A Γ.F₁) (vector Γ.A Γ.F₂) + 
       dot_product (vector Γ.B Γ.F₁) (vector Γ.B Γ.F₂) = 0) : 
  tan_angle Γ.A Γ.B Γ.F₁ * tan_angle Γ.A Γ.B Γ.F₂ = -1/5 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_tangent_product_l55_5533


namespace NUMINAMATH_CALUDE_intersection_segment_length_l55_5525

/-- The length of the line segment formed by the intersection of a line and an ellipse -/
theorem intersection_segment_length 
  (a b : ℝ) 
  (h_ab : a > b ∧ b > 0) 
  (h_ecc : (a^2 - b^2) / a^2 = 1/2) 
  (h_focal : 2 * Real.sqrt (a^2 - b^2) = 2) : 
  ∃ (A B : ℝ × ℝ), 
    (A.2 = -A.1 + 1 ∧ A.1^2 / a^2 + A.2^2 / b^2 = 1) ∧ 
    (B.2 = -B.1 + 1 ∧ B.1^2 / a^2 + B.2^2 / b^2 = 1) ∧ 
    Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 4 * Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_segment_length_l55_5525


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l55_5510

theorem quadratic_two_distinct_roots (m : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x₁^2 + m*x₁ - 8 = 0) ∧ (x₂^2 + m*x₂ - 8 = 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l55_5510


namespace NUMINAMATH_CALUDE_exponential_function_properties_l55_5514

theorem exponential_function_properties (x₁ x₂ : ℝ) (h : x₁ ≠ x₂) :
  let f : ℝ → ℝ := fun x ↦ 2^x
  (f (x₁ + x₂) = f x₁ * f x₂) ∧
  (f (-x₁) = 1 / f x₁) ∧
  ((f x₁ - f x₂) / (x₁ - x₂) > 0) :=
by sorry

end NUMINAMATH_CALUDE_exponential_function_properties_l55_5514


namespace NUMINAMATH_CALUDE_parallel_lines_distance_l55_5500

/-- Given two lines l₁ and l₂ in the form x + ay = 1 and ax + y = 1 respectively,
    if they are parallel, then the distance between them is √2. -/
theorem parallel_lines_distance (a : ℝ) :
  let l₁ := {(x, y) : ℝ × ℝ | x + a * y = 1}
  let l₂ := {(x, y) : ℝ × ℝ | a * x + y = 1}
  (∀ (x₁ y₁ x₂ y₂ : ℝ), (x₁, y₁) ∈ l₁ → (x₂, y₂) ∈ l₂ → 
    (y₂ - y₁) / (x₂ - x₁) = (y₁ - y₂) / (x₁ - x₂)) →
  (∃ (p₁ : ℝ × ℝ) (p₂ : ℝ × ℝ), p₁ ∈ l₁ ∧ p₂ ∈ l₂ ∧ 
    Real.sqrt ((p₂.1 - p₁.1)^2 + (p₂.2 - p₁.2)^2) = Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_distance_l55_5500


namespace NUMINAMATH_CALUDE_property_transaction_outcome_l55_5508

def initial_value : ℝ := 15000
def profit_percentage : ℝ := 0.15
def loss_percentage : ℝ := 0.05

theorem property_transaction_outcome :
  let first_sale := initial_value * (1 + profit_percentage)
  let second_sale := first_sale * (1 - loss_percentage)
  first_sale - second_sale = 862.50 := by sorry

end NUMINAMATH_CALUDE_property_transaction_outcome_l55_5508


namespace NUMINAMATH_CALUDE_min_swaps_100_l55_5537

/-- The type representing a permutation of the first 100 natural numbers. -/
def Perm100 := Fin 100 → Fin 100

/-- The identity permutation. -/
def id_perm : Perm100 := fun i => i

/-- The target permutation we want to achieve. -/
def target_perm : Perm100 := fun i =>
  if i = 99 then 0 else i + 1

/-- A swap operation on a permutation. -/
def swap (p : Perm100) (i j : Fin 100) : Perm100 := fun k =>
  if k = i then p j
  else if k = j then p i
  else p k

/-- The number of swaps needed to transform one permutation into another. -/
def num_swaps (p q : Perm100) : ℕ := sorry

theorem min_swaps_100 :
  num_swaps id_perm target_perm = 99 := by sorry

end NUMINAMATH_CALUDE_min_swaps_100_l55_5537


namespace NUMINAMATH_CALUDE_fraction_sum_l55_5591

theorem fraction_sum : 3/8 + 9/12 = 9/8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l55_5591


namespace NUMINAMATH_CALUDE_ratio_change_after_addition_l55_5594

theorem ratio_change_after_addition : 
  ∀ (a b : ℕ), 
    (a : ℚ) / b = 2 / 3 →
    b - a = 8 →
    (a + 4 : ℚ) / (b + 4) = 5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ratio_change_after_addition_l55_5594


namespace NUMINAMATH_CALUDE_min_workers_to_complete_job_l55_5527

theorem min_workers_to_complete_job
  (total_days : ℕ)
  (days_worked : ℕ)
  (initial_workers : ℕ)
  (job_fraction_completed : ℚ)
  (h1 : total_days = 30)
  (h2 : days_worked = 6)
  (h3 : initial_workers = 8)
  (h4 : job_fraction_completed = 1/3)
  (h5 : days_worked < total_days) :
  ∃ (min_workers : ℕ),
    min_workers ≤ initial_workers ∧
    (min_workers : ℚ) * (total_days - days_worked : ℚ) * job_fraction_completed / days_worked ≥ 1 - job_fraction_completed ∧
    ∀ (w : ℕ), w < min_workers →
      (w : ℚ) * (total_days - days_worked : ℚ) * job_fraction_completed / days_worked < 1 - job_fraction_completed ∧
    min_workers = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_workers_to_complete_job_l55_5527


namespace NUMINAMATH_CALUDE_stream_rate_proof_l55_5557

/-- The speed of the man rowing in still water -/
def still_water_speed : ℝ := 24

/-- The rate of the stream -/
def stream_rate : ℝ := 12

/-- The ratio of time taken to row upstream vs downstream -/
def time_ratio : ℝ := 3

theorem stream_rate_proof :
  (1 / (still_water_speed - stream_rate) = time_ratio * (1 / (still_water_speed + stream_rate))) →
  stream_rate = 12 := by
  sorry

end NUMINAMATH_CALUDE_stream_rate_proof_l55_5557


namespace NUMINAMATH_CALUDE_parallel_vectors_sum_l55_5505

/-- Given two vectors a and b in R³, if they are parallel and have specific components,
    then the sum of their unknown components is -7. -/
theorem parallel_vectors_sum (x y : ℝ) :
  let a : ℝ × ℝ × ℝ := (2, x, 3)
  let b : ℝ × ℝ × ℝ := (-4, 2, y)
  (∃ (k : ℝ), a = k • b) →
  x + y = -7 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_sum_l55_5505


namespace NUMINAMATH_CALUDE_integral_x_sin_ax_over_x2_plus_k2_l55_5542

/-- The integral of x*sin(ax)/(x^2 + k^2) from 0 to infinity equals (π/2)*e^(-ak) for positive a and k -/
theorem integral_x_sin_ax_over_x2_plus_k2 (a k : ℝ) (ha : a > 0) (hk : k > 0) :
  ∫ (x : ℝ) in Set.Ici 0, (x * Real.sin (a * x)) / (x^2 + k^2) = (Real.pi / 2) * Real.exp (-a * k) := by
  sorry

end NUMINAMATH_CALUDE_integral_x_sin_ax_over_x2_plus_k2_l55_5542


namespace NUMINAMATH_CALUDE_bowling_ball_weight_is_correct_l55_5521

/-- The weight of a single bowling ball in pounds -/
def bowling_ball_weight : ℝ := 18.75

/-- The weight of a single canoe in pounds -/
def canoe_weight : ℝ := 30

/-- Theorem stating that the weight of one bowling ball is 18.75 pounds -/
theorem bowling_ball_weight_is_correct :
  (8 * bowling_ball_weight = 5 * canoe_weight) ∧
  (4 * canoe_weight = 120) →
  bowling_ball_weight = 18.75 :=
by
  sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_is_correct_l55_5521


namespace NUMINAMATH_CALUDE_metal_disc_weight_expectation_l55_5555

/-- The nominal radius of a metal disc in meters -/
def nominal_radius : ℝ := 0.5

/-- The standard deviation of the radius in meters -/
def radius_std_dev : ℝ := 0.01

/-- The weight of a disc with exactly 1 m diameter in kilograms -/
def nominal_weight : ℝ := 100

/-- The number of discs in the stack -/
def num_discs : ℕ := 100

/-- The expected weight of the stack of discs in kilograms -/
def expected_stack_weight : ℝ := 10004

theorem metal_disc_weight_expectation :
  let expected_area := π * (nominal_radius^2 + radius_std_dev^2)
  let expected_single_weight := nominal_weight * expected_area / (π * nominal_radius^2)
  expected_single_weight * num_discs = expected_stack_weight :=
sorry

end NUMINAMATH_CALUDE_metal_disc_weight_expectation_l55_5555


namespace NUMINAMATH_CALUDE_quadratic_inequalities_l55_5506

theorem quadratic_inequalities :
  (∀ y : ℝ, y^2 + 4*y + 8 ≥ 4) ∧
  (∀ m : ℝ, m^2 + 2*m + 3 ≥ 2) ∧
  (∀ m : ℝ, -m^2 + 2*m + 3 ≤ 4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequalities_l55_5506


namespace NUMINAMATH_CALUDE_no_roots_of_composition_if_no_roots_l55_5531

/-- A quadratic polynomial -/
def QuadraticPolynomial (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

/-- Theorem: If p(x) = x has no real roots, then p(p(x)) = x has no real roots -/
theorem no_roots_of_composition_if_no_roots (a b c : ℝ) :
  (∀ x : ℝ, QuadraticPolynomial a b c x ≠ x) →
  (∀ x : ℝ, QuadraticPolynomial a b c (QuadraticPolynomial a b c x) ≠ x) := by
  sorry


end NUMINAMATH_CALUDE_no_roots_of_composition_if_no_roots_l55_5531


namespace NUMINAMATH_CALUDE_spencer_walk_distance_l55_5561

theorem spencer_walk_distance (total_distance house_to_library post_office_to_home : ℝ) 
  (h1 : total_distance = 0.8)
  (h2 : house_to_library = 0.3)
  (h3 : post_office_to_home = 0.4) :
  total_distance - house_to_library - post_office_to_home = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_spencer_walk_distance_l55_5561


namespace NUMINAMATH_CALUDE_frank_second_half_correct_l55_5564

/-- Represents the number of questions Frank answered correctly in the first half -/
def first_half_correct : ℕ := 3

/-- Represents the points awarded for each correct answer -/
def points_per_question : ℕ := 3

/-- Represents Frank's final score -/
def final_score : ℕ := 15

/-- Calculates the number of questions Frank answered correctly in the second half -/
def second_half_correct : ℕ :=
  (final_score - first_half_correct * points_per_question) / points_per_question

theorem frank_second_half_correct :
  second_half_correct = 2 := by sorry

end NUMINAMATH_CALUDE_frank_second_half_correct_l55_5564


namespace NUMINAMATH_CALUDE_joshua_borrowed_amount_l55_5576

/-- The cost of the pen in dollars -/
def pen_cost : ℚ := 6

/-- The amount Joshua has in dollars -/
def joshua_has : ℚ := 5

/-- The additional amount Joshua needs in cents -/
def additional_cents : ℚ := 32 / 100

/-- The amount Joshua borrowed in cents -/
def borrowed_amount : ℚ := 132 / 100

theorem joshua_borrowed_amount :
  borrowed_amount = (pen_cost - joshua_has) * 100 + additional_cents := by
  sorry

end NUMINAMATH_CALUDE_joshua_borrowed_amount_l55_5576


namespace NUMINAMATH_CALUDE_square_root_sum_equals_six_l55_5585

theorem square_root_sum_equals_six : 
  Real.sqrt (15 - 6 * Real.sqrt 6) + Real.sqrt (15 + 6 * Real.sqrt 6) = 6 := by
  sorry

end NUMINAMATH_CALUDE_square_root_sum_equals_six_l55_5585


namespace NUMINAMATH_CALUDE_streetlight_shadow_indeterminate_l55_5599

-- Define persons A and B
def Person : Type := String

-- Define shadow length under sunlight
def sunShadowLength (p : Person) : ℝ := sorry

-- Define shadow length under streetlight
def streetShadowLength (p : Person) (distance : ℝ) : ℝ := sorry

-- Define the problem conditions
axiom longer_sun_shadow (A B : Person) : sunShadowLength A > sunShadowLength B

-- Theorem stating that the relative shadow lengths under streetlight cannot be determined
theorem streetlight_shadow_indeterminate (A B : Person) :
  ∃ (d1 d2 : ℝ), 
    (streetShadowLength A d1 > streetShadowLength B d2) ∧
    (∃ (d3 d4 : ℝ), streetShadowLength A d3 < streetShadowLength B d4) ∧
    (∃ (d5 d6 : ℝ), streetShadowLength A d5 = streetShadowLength B d6) :=
sorry

end NUMINAMATH_CALUDE_streetlight_shadow_indeterminate_l55_5599


namespace NUMINAMATH_CALUDE_allyson_age_l55_5563

theorem allyson_age (hiram_age : ℕ) (allyson_age : ℕ) 
  (h1 : hiram_age = 40)
  (h2 : hiram_age + 12 = 2 * allyson_age - 4) :
  allyson_age = 28 := by
  sorry

end NUMINAMATH_CALUDE_allyson_age_l55_5563


namespace NUMINAMATH_CALUDE_car_speed_proof_l55_5583

/-- Proves that a car traveling at speed v km/h takes 15 seconds longer to travel 1 kilometer
    than it would at 48 km/h if and only if v = 40 km/h. -/
theorem car_speed_proof (v : ℝ) : v > 0 →
  (1 / v) * 3600 = (1 / 48) * 3600 + 15 ↔ v = 40 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_proof_l55_5583


namespace NUMINAMATH_CALUDE_victor_trays_l55_5519

/-- The number of trays Victor can carry per trip -/
def tray_capacity : ℕ := 7

/-- The number of trips Victor made -/
def num_trips : ℕ := 4

/-- The number of trays picked up from the second table -/
def trays_second_table : ℕ := 5

/-- The number of trays picked up from the first table -/
def trays_first_table : ℕ := tray_capacity * num_trips - trays_second_table

theorem victor_trays : trays_first_table = 23 := by
  sorry

end NUMINAMATH_CALUDE_victor_trays_l55_5519


namespace NUMINAMATH_CALUDE_area_of_rectangle_l55_5501

/-- A square with two points on its sides forming a rectangle --/
structure SquareWithRectangle where
  -- Side length of the square
  side : ℝ
  -- Ratio of PT to PQ
  pt_ratio : ℝ
  -- Ratio of SU to SR
  su_ratio : ℝ
  -- Assumptions
  side_pos : 0 < side
  pt_ratio_pos : 0 < pt_ratio
  pt_ratio_lt_one : pt_ratio < 1
  su_ratio_pos : 0 < su_ratio
  su_ratio_lt_one : su_ratio < 1

/-- The perimeter of the rectangle PTUS --/
def rectangle_perimeter (s : SquareWithRectangle) : ℝ :=
  2 * (s.side * s.pt_ratio + s.side * s.su_ratio)

/-- The area of the rectangle PTUS --/
def rectangle_area (s : SquareWithRectangle) : ℝ :=
  (s.side * s.pt_ratio) * (s.side * s.su_ratio)

/-- Theorem: If PQRS is a square, T on PQ with PT:TQ = 1:2, U on SR with SU:UR = 1:2,
    and the perimeter of PTUS is 40 cm, then the area of PTUS is 75 cm² --/
theorem area_of_rectangle (s : SquareWithRectangle)
    (h_pt : s.pt_ratio = 1/3)
    (h_su : s.su_ratio = 1/3)
    (h_perimeter : rectangle_perimeter s = 40) :
    rectangle_area s = 75 := by
  sorry

end NUMINAMATH_CALUDE_area_of_rectangle_l55_5501


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l55_5562

-- Define an isosceles triangle with side lengths 6 and 14
def IsoscelesTriangle (a b c : ℝ) : Prop :=
  (a = 14 ∧ b = 14 ∧ c = 6) ∨ (a = 6 ∧ b = 6 ∧ c = 14)

-- Define the perimeter of a triangle
def Perimeter (a b c : ℝ) : ℝ := a + b + c

-- Theorem statement
theorem isosceles_triangle_perimeter :
  ∀ a b c : ℝ, IsoscelesTriangle a b c → Perimeter a b c = 34 :=
by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l55_5562


namespace NUMINAMATH_CALUDE_sqrt_sum_equality_l55_5584

theorem sqrt_sum_equality (a b c k : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hk : k > 0)
  (h : 2 * a * b * c + k * (a^2 + b^2 + c^2) = k^3) :
  Real.sqrt ((k - a) * (k - b) / ((k + a) * (k + b))) +
  Real.sqrt ((k - b) * (k - c) / ((k + b) * (k + c))) +
  Real.sqrt ((k - c) * (k - a) / ((k + c) * (k + a))) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equality_l55_5584


namespace NUMINAMATH_CALUDE_max_profit_theorem_additional_cost_range_l55_5548

/-- Represents the monthly sales and profit model for a product. -/
structure SalesModel where
  cost_price : ℝ
  initial_price : ℝ
  initial_sales : ℝ
  price_sensitivity : ℝ
  max_price : ℝ

/-- Calculates the monthly sales volume given a price increase. -/
def sales_volume (model : SalesModel) (price_increase : ℝ) : ℝ :=
  model.initial_sales - model.price_sensitivity * price_increase

/-- Calculates the monthly profit given a price increase. -/
def monthly_profit (model : SalesModel) (price_increase : ℝ) : ℝ :=
  (sales_volume model price_increase) * (model.initial_price + price_increase - model.cost_price)

/-- Theorem stating the maximum monthly profit and optimal selling price. -/
theorem max_profit_theorem (model : SalesModel) 
  (h_cost : model.cost_price = 40)
  (h_initial_price : model.initial_price = 50)
  (h_initial_sales : model.initial_sales = 210)
  (h_price_sensitivity : model.price_sensitivity = 10)
  (h_max_price : model.max_price = 65) :
  ∃ (x : ℝ), x ∈ Set.Icc 5 6 ∧ 
  ∀ (y : ℝ), y > 0 ∧ y ≤ 15 → monthly_profit model x ≥ monthly_profit model y ∧
  monthly_profit model x = 2400 := by sorry

/-- Theorem stating the range of additional costs. -/
theorem additional_cost_range (model : SalesModel) (a : ℝ)
  (h_cost : model.cost_price = 40)
  (h_initial_price : model.initial_price = 50)
  (h_initial_sales : model.initial_sales = 210)
  (h_price_sensitivity : model.price_sensitivity = 10)
  (h_max_price : model.max_price = 65) :
  (∀ (x y : ℝ), 8 ≤ x ∧ x < y ∧ y ≤ 15 → 
    monthly_profit model x - (sales_volume model x * a) > 
    monthly_profit model y - (sales_volume model y * a)) 
  ↔ 0 < a ∧ a < 6 := by sorry

end NUMINAMATH_CALUDE_max_profit_theorem_additional_cost_range_l55_5548


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l55_5526

theorem inequality_and_equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (1 / (a * (1 + b)) + 1 / (b * (1 + c)) + 1 / (c * (1 + a)) ≥ 3 / (1 + a * b * c)) ∧
  (1 / (a * (1 + b)) + 1 / (b * (1 + c)) + 1 / (c * (1 + a)) = 3 / (1 + a * b * c) ↔ a = 1 ∧ b = 1 ∧ c = 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l55_5526


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l55_5598

/-- Given two vectors a and b in ℝ², where a = (4, 8) and b = (x, 4),
    if a is perpendicular to b, then x = -8. -/
theorem perpendicular_vectors_x_value :
  let a : ℝ × ℝ := (4, 8)
  let b : ℝ × ℝ := (x, 4)
  (a.1 * b.1 + a.2 * b.2 = 0) → x = -8 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l55_5598


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l55_5512

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h : ∀ n : ℕ, a n = 3 * 2^(n - 1)) : 
  (∃ q : ℝ, ∀ n : ℕ, n ≥ 1 → a (n + 1) = q * a n) ∧ 
  (∀ q : ℝ, (∀ n : ℕ, n ≥ 1 → a (n + 1) = q * a n) → q = 2) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l55_5512


namespace NUMINAMATH_CALUDE_expression_equality_l55_5517

theorem expression_equality (x : ℝ) (h : x > 0) : 
  x^x - x^x = 0 ∧ (x - 1)^x = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l55_5517


namespace NUMINAMATH_CALUDE_problem_solution_l55_5530

/-- The probability that person A can solve the problem within half an hour -/
def prob_A : ℚ := 1/2

/-- The probability that person B can solve the problem within half an hour -/
def prob_B : ℚ := 1/3

/-- The probability that neither A nor B solves the problem -/
def prob_neither_solves : ℚ := (1 - prob_A) * (1 - prob_B)

/-- The probability that the problem is solved -/
def prob_problem_solved : ℚ := 1 - prob_neither_solves

theorem problem_solution :
  prob_neither_solves = 1/3 ∧ prob_problem_solved = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l55_5530


namespace NUMINAMATH_CALUDE_exponential_problem_l55_5536

theorem exponential_problem (a x y : ℝ) (h1 : a^x = 2) (h2 : a^y = 3) :
  a^(2*x + 3*y) = 108 := by
  sorry

end NUMINAMATH_CALUDE_exponential_problem_l55_5536


namespace NUMINAMATH_CALUDE_average_speed_calculation_l55_5586

theorem average_speed_calculation (total_distance : ℝ) (first_half_speed : ℝ) (second_half_time_factor : ℝ) :
  total_distance = 640 →
  first_half_speed = 80 →
  second_half_time_factor = 3 →
  let first_half_distance := total_distance / 2
  let first_half_time := first_half_distance / first_half_speed
  let second_half_time := first_half_time * second_half_time_factor
  let total_time := first_half_time + second_half_time
  (total_distance / total_time) = 40 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l55_5586


namespace NUMINAMATH_CALUDE_equation_holds_for_all_x_l55_5558

theorem equation_holds_for_all_x (m : ℝ) : 
  (∀ x : ℝ, (m - 5) * x = 0) → m = 5 := by
  sorry

end NUMINAMATH_CALUDE_equation_holds_for_all_x_l55_5558


namespace NUMINAMATH_CALUDE_smallest_three_digit_divisible_by_3_and_6_l55_5524

theorem smallest_three_digit_divisible_by_3_and_6 :
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 3 = 0 ∧ n % 6 = 0 → n ≥ 102 := by
  sorry

end NUMINAMATH_CALUDE_smallest_three_digit_divisible_by_3_and_6_l55_5524


namespace NUMINAMATH_CALUDE_tv_sales_decrease_l55_5551

theorem tv_sales_decrease (original_price original_quantity : ℝ) 
  (price_increase : ℝ) (revenue_increase : ℝ) (sales_decrease : ℝ) :
  price_increase = 0.4 →
  revenue_increase = 0.12 →
  (1 + price_increase) * (1 - sales_decrease) = 1 + revenue_increase →
  sales_decrease = 0.2 := by
sorry

end NUMINAMATH_CALUDE_tv_sales_decrease_l55_5551


namespace NUMINAMATH_CALUDE_projection_of_vectors_l55_5571

/-- Given two vectors in ℝ², prove that the projection of one onto the other is as specified. -/
theorem projection_of_vectors (a b : ℝ × ℝ) (h1 : a = (0, 1)) (h2 : b = (1, Real.sqrt 3)) :
  (a • b / (b • b)) • b = (Real.sqrt 3 / 4) • b :=
sorry

end NUMINAMATH_CALUDE_projection_of_vectors_l55_5571


namespace NUMINAMATH_CALUDE_evaluate_nested_square_roots_l55_5513

theorem evaluate_nested_square_roots : 
  Real.sqrt (25 * Real.sqrt (15 * Real.sqrt 9)) = 5 * Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_nested_square_roots_l55_5513


namespace NUMINAMATH_CALUDE_surrounding_circles_radius_l55_5539

theorem surrounding_circles_radius (r : ℝ) : 
  (∃ (A B C D : ℝ × ℝ),
    -- The centers of the surrounding circles form a square
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = (4*r)^2 ∧
    (B.1 - C.1)^2 + (B.2 - C.2)^2 = (4*r)^2 ∧
    (C.1 - D.1)^2 + (C.2 - D.2)^2 = (4*r)^2 ∧
    (D.1 - A.1)^2 + (D.2 - A.2)^2 = (4*r)^2 ∧
    -- The diagonal of the square
    (A.1 - C.1)^2 + (A.2 - C.2)^2 = (2 + 2*r)^2 ∧
    -- The surrounding circles touch the central circle
    ∃ (O : ℝ × ℝ), (A.1 - O.1)^2 + (A.2 - O.2)^2 = (r + 1)^2) →
  r = 1 + Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_surrounding_circles_radius_l55_5539


namespace NUMINAMATH_CALUDE_power_mod_45_l55_5579

theorem power_mod_45 : 14^100 % 45 = 31 := by
  sorry

end NUMINAMATH_CALUDE_power_mod_45_l55_5579


namespace NUMINAMATH_CALUDE_no_consecutive_product_l55_5541

theorem no_consecutive_product (n : ℕ) : ¬ ∃ k : ℕ, n^2 + 7*n + 8 = k * (k + 1) := by
  sorry

end NUMINAMATH_CALUDE_no_consecutive_product_l55_5541


namespace NUMINAMATH_CALUDE_one_third_of_recipe_l55_5595

theorem one_third_of_recipe (original_amount : ℚ) (reduced_amount : ℚ) : 
  original_amount = 5 + 3/4 → reduced_amount = (1/3) * original_amount → 
  reduced_amount = 1 + 11/12 := by
sorry

end NUMINAMATH_CALUDE_one_third_of_recipe_l55_5595


namespace NUMINAMATH_CALUDE_min_cars_with_racing_stripes_l55_5552

theorem min_cars_with_racing_stripes 
  (total_cars : ℕ) 
  (cars_without_ac : ℕ) 
  (max_ac_no_stripes : ℕ) 
  (h1 : total_cars = 100)
  (h2 : cars_without_ac = 37)
  (h3 : max_ac_no_stripes = 59) :
  ∃ (min_cars_with_stripes : ℕ), 
    min_cars_with_stripes = 4 ∧ 
    min_cars_with_stripes ≤ total_cars - cars_without_ac ∧
    min_cars_with_stripes = total_cars - cars_without_ac - max_ac_no_stripes :=
by
  sorry

#check min_cars_with_racing_stripes

end NUMINAMATH_CALUDE_min_cars_with_racing_stripes_l55_5552


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l55_5522

theorem simplify_and_evaluate (a : ℝ) (h : a = 3) :
  (1 + 1 / (a + 1)) / ((a^2 - 4) / (2 * a + 2)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l55_5522


namespace NUMINAMATH_CALUDE_injective_function_inequality_l55_5520

theorem injective_function_inequality (f : ℕ → ℕ) 
  (h_inj : Function.Injective f) 
  (h_ineq : ∀ n : ℕ, f (f n) ≤ (n + f n) / 2) : 
  ∀ n : ℕ, f n = n := by
  sorry

end NUMINAMATH_CALUDE_injective_function_inequality_l55_5520


namespace NUMINAMATH_CALUDE_light_flashes_l55_5596

/-- A light flashes every 15 seconds. This theorem proves that it will flash 180 times in ¾ of an hour. -/
theorem light_flashes (flash_interval : ℕ) (hour_fraction : ℚ) (flashes : ℕ) : 
  flash_interval = 15 → hour_fraction = 3/4 → flashes = 180 → 
  (hour_fraction * 3600) / flash_interval = flashes := by
  sorry

end NUMINAMATH_CALUDE_light_flashes_l55_5596


namespace NUMINAMATH_CALUDE_curve_tangent_parallel_l55_5538

theorem curve_tangent_parallel (k : ℝ) : 
  let f := fun x : ℝ => k * x + Real.log x
  let f' := fun x : ℝ => k + 1 / x
  (f' 1 = 2) → k = 1 := by
sorry

end NUMINAMATH_CALUDE_curve_tangent_parallel_l55_5538


namespace NUMINAMATH_CALUDE_median_less_than_half_sum_of_sides_l55_5529

/-- Given a triangle ABC with sides a, b, and c, and median CM₃ to side c,
    prove that CM₃ < (a + b) / 2 -/
theorem median_less_than_half_sum_of_sides 
  {a b c : ℝ} 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (triangle_inequality : c < a + b) :
  let CM₃ := Real.sqrt ((2 * a^2 + 2 * b^2 - c^2) / 4)
  CM₃ < (a + b) / 2 := by
sorry

end NUMINAMATH_CALUDE_median_less_than_half_sum_of_sides_l55_5529


namespace NUMINAMATH_CALUDE_coal_shoveling_time_l55_5532

/-- Given that 10 people can shovel 10,000 pounds of coal in 10 days,
    prove that 5 people will take 80 days to shovel 40,000 pounds of coal. -/
theorem coal_shoveling_time 
  (people : ℕ) 
  (days : ℕ) 
  (coal : ℕ) 
  (h1 : people = 10) 
  (h2 : days = 10) 
  (h3 : coal = 10000) :
  (people / 2) * (coal * 4 / (people * days)) = 80 := by
  sorry

#check coal_shoveling_time

end NUMINAMATH_CALUDE_coal_shoveling_time_l55_5532


namespace NUMINAMATH_CALUDE_cat_speed_l55_5516

/-- Proves that given a rabbit running at 25 miles per hour, a cat with a 15-minute head start,
    and the rabbit taking 1 hour to catch up, the cat's speed is 20 miles per hour. -/
theorem cat_speed (rabbit_speed : ℝ) (head_start : ℝ) (catch_up_time : ℝ) :
  rabbit_speed = 25 →
  head_start = 0.25 →
  catch_up_time = 1 →
  ∃ cat_speed : ℝ,
    cat_speed * (head_start + catch_up_time) = rabbit_speed * catch_up_time ∧
    cat_speed = 20 :=
by sorry

end NUMINAMATH_CALUDE_cat_speed_l55_5516


namespace NUMINAMATH_CALUDE_sqrt_a_squared_plus_a_equals_two_thirds_l55_5540

theorem sqrt_a_squared_plus_a_equals_two_thirds (a : ℝ) :
  a > 0 ∧ Real.sqrt (a^2 + a) = 2/3 ↔ a = 1/3 := by sorry

end NUMINAMATH_CALUDE_sqrt_a_squared_plus_a_equals_two_thirds_l55_5540


namespace NUMINAMATH_CALUDE_car_clock_accuracy_l55_5588

def actual_time (start_time : ℕ) (elapsed_time : ℕ) (gain_rate : ℚ) : ℚ :=
  start_time + elapsed_time / gain_rate

theorem car_clock_accuracy (start_time : ℕ) (elapsed_time : ℕ) : 
  start_time = 8 * 60 →  -- 8:00 AM in minutes
  elapsed_time = 14 * 60 →  -- 14 hours from 8:00 AM to 10:00 PM in minutes
  actual_time start_time elapsed_time (37/36) = 21 * 60 + 47  -- 9:47 PM in minutes
  := by sorry

end NUMINAMATH_CALUDE_car_clock_accuracy_l55_5588


namespace NUMINAMATH_CALUDE_stratified_sample_theorem_l55_5560

/-- Calculates the number of units to be selected from a stratum in stratified sampling -/
def stratified_sample_size (total_population : ℕ) (total_sample_size : ℕ) (stratum_size : ℕ) : ℕ :=
  (stratum_size * total_sample_size) / total_population

/-- Theorem: In a stratified sampling method with a total population of 1000 units
    and a sample size of 60 units, the number of units to be selected from
    a stratum of 300 units is 18. -/
theorem stratified_sample_theorem :
  stratified_sample_size 1000 60 300 = 18 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_theorem_l55_5560


namespace NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l55_5553

theorem at_least_one_not_less_than_two
  (a b c : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (hc : c > 0)
  (sum_eq_three : a + b + c = 3) :
  ¬(a + 1/b < 2 ∧ b + 1/c < 2 ∧ c + 1/a < 2) :=
by sorry

end NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l55_5553


namespace NUMINAMATH_CALUDE_rectangular_prism_problem_l55_5581

/-- The number of valid triples (a, b, c) for the rectangular prism problem -/
def valid_triples : Nat :=
  (Finset.filter (fun a => a < 1995 ∧ (1995 * 1995) % a = 0)
    (Finset.range 1995)).card

/-- The theorem stating that there are exactly 40 valid triples -/
theorem rectangular_prism_problem :
  valid_triples = 40 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_problem_l55_5581
