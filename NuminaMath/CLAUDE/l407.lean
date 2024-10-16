import Mathlib

namespace NUMINAMATH_CALUDE_distance_p_to_y_axis_l407_40765

/-- The distance from a point to the y-axis is the absolute value of its x-coordinate -/
def distance_to_y_axis (x : ℝ) : ℝ := |x|

/-- Given a point P(-3, 2) in the second quadrant, its distance to the y-axis is 3 -/
theorem distance_p_to_y_axis :
  let P : ℝ × ℝ := (-3, 2)
  distance_to_y_axis P.1 = 3 := by sorry

end NUMINAMATH_CALUDE_distance_p_to_y_axis_l407_40765


namespace NUMINAMATH_CALUDE_inequality_solution_set_l407_40756

theorem inequality_solution_set (x : ℝ) : 
  (x - 2) * (x + 2) < 5 ↔ -3 < x ∧ x < 3 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l407_40756


namespace NUMINAMATH_CALUDE_perfect_square_from_48_numbers_l407_40726

theorem perfect_square_from_48_numbers (S : Finset ℕ) 
  (h1 : S.card = 48)
  (h2 : (S.prod id).factors.toFinset.card = 10) :
  ∃ (a b c d : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧
  ∃ (m : ℕ), a * b * c * d = m ^ 2 :=
sorry

end NUMINAMATH_CALUDE_perfect_square_from_48_numbers_l407_40726


namespace NUMINAMATH_CALUDE_injective_function_inequality_l407_40729

theorem injective_function_inequality (f : Set.Icc 0 1 → ℝ) 
  (h_inj : Function.Injective f) (h_sum : f 0 + f 1 = 1) :
  ∃ (x₁ x₂ : Set.Icc 0 1), x₁ ≠ x₂ ∧ 2 * f x₁ < f x₂ + 1/2 := by
  sorry

end NUMINAMATH_CALUDE_injective_function_inequality_l407_40729


namespace NUMINAMATH_CALUDE_gold_coin_distribution_l407_40777

theorem gold_coin_distribution (x y : ℕ) (h1 : x + y = 16) (h2 : x ≠ y) :
  ∃ k : ℕ, x^2 - y^2 = k * (x - y) → k = 16 := by
  sorry

end NUMINAMATH_CALUDE_gold_coin_distribution_l407_40777


namespace NUMINAMATH_CALUDE_sqrt_inequality_l407_40712

theorem sqrt_inequality (m : ℝ) (h : m > 1) :
  Real.sqrt (m + 1) - Real.sqrt m < Real.sqrt m - Real.sqrt (m - 1) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l407_40712


namespace NUMINAMATH_CALUDE_isoelectronic_pairs_l407_40748

/-- Represents a molecule with its composition and valence electron count -/
structure Molecule where
  composition : List (Nat × Nat)  -- List of (atomic number, count) pairs
  valence_electrons : Nat

/-- Calculates the total number of valence electrons for a molecule -/
def calculate_valence_electrons (composition : List (Nat × Nat)) : Nat :=
  composition.foldl (fun acc (atomic_number, count) => 
    acc + count * match atomic_number with
      | 6 => 4  -- Carbon
      | 7 => 5  -- Nitrogen
      | 8 => 6  -- Oxygen
      | 16 => 6 -- Sulfur
      | _ => 0
    ) 0

/-- Determines if two molecules are isoelectronic -/
def are_isoelectronic (m1 m2 : Molecule) : Prop :=
  m1.valence_electrons = m2.valence_electrons

/-- N2 molecule -/
def N2 : Molecule := ⟨[(7, 2)], calculate_valence_electrons [(7, 2)]⟩

/-- CO molecule -/
def CO : Molecule := ⟨[(6, 1), (8, 1)], calculate_valence_electrons [(6, 1), (8, 1)]⟩

/-- N2O molecule -/
def N2O : Molecule := ⟨[(7, 2), (8, 1)], calculate_valence_electrons [(7, 2), (8, 1)]⟩

/-- CO2 molecule -/
def CO2 : Molecule := ⟨[(6, 1), (8, 2)], calculate_valence_electrons [(6, 1), (8, 2)]⟩

/-- NO2⁻ ion -/
def NO2_minus : Molecule := ⟨[(7, 1), (8, 2)], calculate_valence_electrons [(7, 1), (8, 2)] + 1⟩

/-- SO2 molecule -/
def SO2 : Molecule := ⟨[(16, 1), (8, 2)], calculate_valence_electrons [(16, 1), (8, 2)]⟩

/-- O3 molecule -/
def O3 : Molecule := ⟨[(8, 3)], calculate_valence_electrons [(8, 3)]⟩

theorem isoelectronic_pairs : 
  (are_isoelectronic N2 CO) ∧ 
  (are_isoelectronic N2O CO2) ∧ 
  (are_isoelectronic NO2_minus SO2) ∧ 
  (are_isoelectronic NO2_minus O3) := by
  sorry

end NUMINAMATH_CALUDE_isoelectronic_pairs_l407_40748


namespace NUMINAMATH_CALUDE_max_value_of_a_l407_40799

theorem max_value_of_a (a b c : ℝ) (sum_zero : a + b + c = 0) (sum_squares_six : a^2 + b^2 + c^2 = 6) :
  ∃ (max_a : ℝ), max_a = 2 ∧ ∀ x, (∃ y z, x + y + z = 0 ∧ x^2 + y^2 + z^2 = 6) → x ≤ max_a :=
sorry

end NUMINAMATH_CALUDE_max_value_of_a_l407_40799


namespace NUMINAMATH_CALUDE_no_solution_double_inequality_l407_40751

theorem no_solution_double_inequality :
  ¬∃ x : ℝ, (3 * x - 2 < (x + 2)^2) ∧ ((x + 2)^2 < 9 * x - 5) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_double_inequality_l407_40751


namespace NUMINAMATH_CALUDE_gcd_4536_8721_l407_40711

theorem gcd_4536_8721 : Nat.gcd 4536 8721 = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_4536_8721_l407_40711


namespace NUMINAMATH_CALUDE_gold_coin_percentage_is_45_5_percent_l407_40731

/-- Represents the composition of items in an urn --/
structure UrnComposition where
  beadPercentage : ℝ
  bronzeCoinPercentage : ℝ

/-- Calculates the percentage of gold coins in the urn --/
def goldCoinPercentage (urn : UrnComposition) : ℝ :=
  (1 - urn.beadPercentage) * (1 - urn.bronzeCoinPercentage)

/-- Theorem: The percentage of gold coins in the urn is 45.5% --/
theorem gold_coin_percentage_is_45_5_percent (urn : UrnComposition)
  (h1 : urn.beadPercentage = 0.35)
  (h2 : urn.bronzeCoinPercentage = 0.30) :
  goldCoinPercentage urn = 0.455 := by
  sorry

#eval goldCoinPercentage { beadPercentage := 0.35, bronzeCoinPercentage := 0.30 }

end NUMINAMATH_CALUDE_gold_coin_percentage_is_45_5_percent_l407_40731


namespace NUMINAMATH_CALUDE_largest_divisor_of_difference_of_squares_l407_40791

theorem largest_divisor_of_difference_of_squares (a b : ℤ) :
  let m : ℤ := 2*a + 3
  let n : ℤ := 2*b + 1
  (n < m) →
  (∃ k : ℤ, m^2 - n^2 = 4*k) ∧
  (∀ d : ℤ, d > 4 → ∃ a' b' : ℤ, 
    let m' : ℤ := 2*a' + 3
    let n' : ℤ := 2*b' + 1
    (n' < m') ∧ (m'^2 - n'^2) % d ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_difference_of_squares_l407_40791


namespace NUMINAMATH_CALUDE_representation_theorem_l407_40714

theorem representation_theorem (a b : ℕ+) :
  (∃ (S : Finset ℕ), ∀ (n : ℕ), ∃ (x y : ℕ) (s : ℕ), s ∈ S ∧ n = x^(a:ℕ) + y^(b:ℕ) + s) ↔
  (a = 1 ∨ b = 1) :=
sorry

end NUMINAMATH_CALUDE_representation_theorem_l407_40714


namespace NUMINAMATH_CALUDE_quadratic_problem_l407_40747

-- Define the quadratic equation
def quadratic (p q x : ℝ) : ℝ := x^2 + p*x + q

-- Theorem statement
theorem quadratic_problem (p q : ℝ) 
  (h1 : quadratic p (q + 1) 2 = 0) : 
  -- 1. Relationship between q and p
  q = -2*p - 5 ∧ 
  -- 2. Two distinct real roots
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ quadratic p q x₁ = 0 ∧ quadratic p q x₂ = 0 ∧
  -- 3. If equal roots in original equation, roots of modified equation
  (∃ (r : ℝ), ∀ x, quadratic p (q + 1) x = 0 → x = r) → 
    quadratic p q 1 = 0 ∧ quadratic p q 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_problem_l407_40747


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l407_40709

theorem quadratic_inequality_solution (a b : ℝ) :
  (∀ x, a * x^2 - 3*x + b > 4 ↔ x < 1 ∨ x > 2) →
  (a = 1 ∧ b = 6) ∧
  (∀ c : ℝ,
    (c > 2 → ∀ x, x^2 - (c+2)*x + 2*c < 0 ↔ 2 < x ∧ x < c) ∧
    (c = 2 → ∀ x, ¬(x^2 - (c+2)*x + 2*c < 0)) ∧
    (c < 2 → ∀ x, x^2 - (c+2)*x + 2*c < 0 ↔ c < x ∧ x < 2)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l407_40709


namespace NUMINAMATH_CALUDE_sixth_angle_measure_l407_40703

/-- The sum of interior angles of a hexagon is 720 degrees -/
def hexagon_angle_sum : ℝ := 720

/-- The given measures of five angles in the hexagon -/
def given_angles : List ℝ := [134, 108, 122, 99, 87]

/-- Theorem: In a hexagon where five of the interior angles measure 134°, 108°, 122°, 99°, and 87°, 
    the measure of the sixth angle is 170°. -/
theorem sixth_angle_measure :
  let sum_given_angles := given_angles.sum
  hexagon_angle_sum - sum_given_angles = 170 := by sorry

end NUMINAMATH_CALUDE_sixth_angle_measure_l407_40703


namespace NUMINAMATH_CALUDE_function_properties_l407_40761

/-- Given a function f(x) = x + m/x where f(1) = 5, this theorem proves:
    1. The value of m
    2. The parity of f
    3. The monotonicity of f on (2, +∞) -/
theorem function_properties (f : ℝ → ℝ) (m : ℝ) 
    (h1 : ∀ x ≠ 0, f x = x + m / x)
    (h2 : f 1 = 5) :
    (m = 4) ∧ 
    (∀ x ≠ 0, f (-x) = -f x) ∧
    (∀ x₁ x₂, 2 < x₁ → x₁ < x₂ → f x₁ < f x₂) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l407_40761


namespace NUMINAMATH_CALUDE_bag_original_price_l407_40701

theorem bag_original_price (sale_price : ℝ) (discount_percent : ℝ) (original_price : ℝ) : 
  sale_price = 120 → 
  discount_percent = 50 → 
  sale_price = original_price * (1 - discount_percent / 100) → 
  original_price = 240 := by
sorry

end NUMINAMATH_CALUDE_bag_original_price_l407_40701


namespace NUMINAMATH_CALUDE_contest_probabilities_l407_40774

/-- Represents the total number of questions -/
def total_questions : ℕ := 8

/-- Represents the number of listening questions -/
def listening_questions : ℕ := 3

/-- Represents the number of written response questions -/
def written_questions : ℕ := 5

/-- Calculates the probability of the first student drawing a listening question
    and the second student drawing a written response question -/
def prob_listening_written : ℚ :=
  (listening_questions * written_questions : ℚ) / (total_questions * (total_questions - 1))

/-- Calculates the probability of at least one student drawing a listening question -/
def prob_at_least_one_listening : ℚ :=
  1 - (written_questions * (written_questions - 1) : ℚ) / (total_questions * (total_questions - 1))

theorem contest_probabilities :
  prob_listening_written = 15 / 56 ∧ prob_at_least_one_listening = 9 / 14 := by
  sorry

end NUMINAMATH_CALUDE_contest_probabilities_l407_40774


namespace NUMINAMATH_CALUDE_sum_positive_if_greater_than_abs_l407_40736

theorem sum_positive_if_greater_than_abs (a b : ℝ) (h : a - |b| > 0) : a + b > 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_positive_if_greater_than_abs_l407_40736


namespace NUMINAMATH_CALUDE_complex_modulus_one_l407_40710

theorem complex_modulus_one (z : ℂ) (h : z * (1 + Complex.I) = (1 - Complex.I)) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_one_l407_40710


namespace NUMINAMATH_CALUDE_train_speed_calculation_l407_40702

/-- Calculates the speed of a train crossing a bridge -/
theorem train_speed_calculation (train_length bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 110 →
  bridge_length = 265 →
  crossing_time = 30 →
  (train_length + bridge_length) / crossing_time * 3.6 = 45 := by
  sorry

#check train_speed_calculation

end NUMINAMATH_CALUDE_train_speed_calculation_l407_40702


namespace NUMINAMATH_CALUDE_expected_winnings_unique_coin_l407_40793

/-- A unique weighted coin with four possible outcomes -/
structure WeightedCoin where
  prob_heads : ℚ
  prob_tails : ℚ
  prob_edge : ℚ
  prob_other : ℚ
  winnings_heads : ℚ
  winnings_tails : ℚ
  winnings_edge : ℚ
  winnings_other : ℚ

/-- The expected winnings from flipping the coin -/
def expected_winnings (c : WeightedCoin) : ℚ :=
  c.prob_heads * c.winnings_heads +
  c.prob_tails * c.winnings_tails +
  c.prob_edge * c.winnings_edge +
  c.prob_other * c.winnings_other

/-- The specific coin described in the problem -/
def unique_coin : WeightedCoin :=
  { prob_heads := 3/7
  , prob_tails := 1/4
  , prob_edge := 1/7
  , prob_other := 2/7
  , winnings_heads := 2
  , winnings_tails := 4
  , winnings_edge := -6
  , winnings_other := -2 }

theorem expected_winnings_unique_coin :
  expected_winnings unique_coin = 3/7 := by sorry

end NUMINAMATH_CALUDE_expected_winnings_unique_coin_l407_40793


namespace NUMINAMATH_CALUDE_value_of_b_minus_d_squared_l407_40794

theorem value_of_b_minus_d_squared (a b c d : ℤ) 
  (eq1 : a - b - c + d = 13)
  (eq2 : a + b - c - d = 5) : 
  (b - d)^2 = 16 := by sorry

end NUMINAMATH_CALUDE_value_of_b_minus_d_squared_l407_40794


namespace NUMINAMATH_CALUDE_right_triangle_7_24_25_l407_40770

theorem right_triangle_7_24_25 (a b c : ℝ) :
  a = 7 ∧ b = 24 ∧ c = 25 → a^2 + b^2 = c^2 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_7_24_25_l407_40770


namespace NUMINAMATH_CALUDE_quadratic_equation_completion_square_l407_40700

theorem quadratic_equation_completion_square (x : ℝ) : 
  16 * x^2 - 32 * x - 512 = 0 → ∃ r s : ℝ, (x + r)^2 = s ∧ s = 33 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_completion_square_l407_40700


namespace NUMINAMATH_CALUDE_probability_defective_from_A_l407_40724

-- Define the probabilities
def prob_factory_A : ℝ := 0.45
def prob_factory_B : ℝ := 0.55
def defect_rate_A : ℝ := 0.06
def defect_rate_B : ℝ := 0.05

-- Theorem statement
theorem probability_defective_from_A : 
  let prob_defective := prob_factory_A * defect_rate_A + prob_factory_B * defect_rate_B
  prob_factory_A * defect_rate_A / prob_defective = 54 / 109 := by
sorry

end NUMINAMATH_CALUDE_probability_defective_from_A_l407_40724


namespace NUMINAMATH_CALUDE_sequence_convergence_bound_l407_40725

def x : ℕ → ℚ
  | 0 => 6
  | n + 1 => (x n ^ 2 + 6 * x n + 7) / (x n + 7)

theorem sequence_convergence_bound :
  ∃ m : ℕ, m ∈ Set.Icc 151 300 ∧
    x m ≤ 4 + 1 / (2^25) ∧
    ∀ k : ℕ, k < m → x k > 4 + 1 / (2^25) :=
by sorry

end NUMINAMATH_CALUDE_sequence_convergence_bound_l407_40725


namespace NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l407_40722

theorem least_positive_integer_with_remainders : ∃! x : ℕ,
  x > 0 ∧
  x % 4 = 3 ∧
  x % 5 = 4 ∧
  x % 7 = 6 ∧
  ∀ y : ℕ, y > 0 ∧ y % 4 = 3 ∧ y % 5 = 4 ∧ y % 7 = 6 → x ≤ y :=
by
  use 139
  sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l407_40722


namespace NUMINAMATH_CALUDE_expression_simplification_l407_40787

theorem expression_simplification (x : ℝ) : 
  2*x - 3*(2 - x) + 5*(2 + x) - 7*(1 - 3*x) = 31*x - 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l407_40787


namespace NUMINAMATH_CALUDE_second_child_birth_year_l407_40744

theorem second_child_birth_year (first_child_age : ℕ) (fourth_child_age : ℕ) 
  (h1 : first_child_age = 15)
  (h2 : fourth_child_age = 8)
  (h3 : ∃ (third_child_age : ℕ), third_child_age = fourth_child_age + 2)
  (h4 : ∃ (second_child_age : ℕ), second_child_age + 4 = third_child_age) :
  first_child_age - (fourth_child_age + 6) = 1 := by
sorry

end NUMINAMATH_CALUDE_second_child_birth_year_l407_40744


namespace NUMINAMATH_CALUDE_compare_powers_l407_40740

theorem compare_powers (x m n : ℝ) (hx_pos : x > 0) (hx_neq_one : x ≠ 1) (hm_gt_n : m > n) (hn_pos : n > 0) :
  x^m + 1/x^m > x^n + 1/x^n := by
  sorry

end NUMINAMATH_CALUDE_compare_powers_l407_40740


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l407_40716

theorem sufficient_but_not_necessary (x : ℝ) : 
  (∀ x, x > 1 → x > 0) ∧ (∃ x, x > 0 ∧ ¬(x > 1)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l407_40716


namespace NUMINAMATH_CALUDE_smallest_sum_arithmetic_cubic_sequence_l407_40766

theorem smallest_sum_arithmetic_cubic_sequence (A B C D : ℕ+) : 
  (∃ r : ℚ, B = A + r ∧ C = B + r) →  -- A, B, C form an arithmetic sequence
  (D - C = (C - B)^2) →  -- B, C, D form a cubic sequence
  (C : ℚ) / B = 4 / 3 →  -- C/B = 4/3
  (∀ A' B' C' D' : ℕ+, 
    (∃ r' : ℚ, B' = A' + r' ∧ C' = B' + r') → 
    (D' - C' = (C' - B')^2) → 
    (C' : ℚ) / B' = 4 / 3 → 
    A + B + C + D ≤ A' + B' + C' + D') →
  A + B + C + D = 14 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_arithmetic_cubic_sequence_l407_40766


namespace NUMINAMATH_CALUDE_star_properties_l407_40760

-- Define the * operation
def star (x y : ℝ) : ℝ := (x + 1) * (y + 1) - 1

-- State the theorem
theorem star_properties :
  ∀ x y : ℝ,
  (star x y = star y x) ∧
  (star (x + 1) (x - 1) = x * x - 1) := by
  sorry

end NUMINAMATH_CALUDE_star_properties_l407_40760


namespace NUMINAMATH_CALUDE_original_price_proof_l407_40762

/-- Given an item sold at a 20% loss with a selling price of 1040, 
    prove that the original price of the item was 1300. -/
theorem original_price_proof (selling_price : ℝ) (loss_percentage : ℝ) 
  (h1 : selling_price = 1040)
  (h2 : loss_percentage = 20) : 
  ∃ (original_price : ℝ), 
    selling_price = original_price * (1 - loss_percentage / 100) ∧ 
    original_price = 1300 :=
by
  sorry

end NUMINAMATH_CALUDE_original_price_proof_l407_40762


namespace NUMINAMATH_CALUDE_spectators_count_l407_40704

/-- The number of wristbands given to each spectator -/
def wristbands_per_person : ℕ := 2

/-- The total number of wristbands distributed -/
def total_wristbands : ℕ := 250

/-- The number of people who watched the game -/
def spectators : ℕ := total_wristbands / wristbands_per_person

theorem spectators_count : spectators = 125 := by
  sorry

end NUMINAMATH_CALUDE_spectators_count_l407_40704


namespace NUMINAMATH_CALUDE_other_number_proof_l407_40737

theorem other_number_proof (A B : ℕ+) (hcf lcm : ℕ+) : 
  hcf = Nat.gcd A B →
  lcm = Nat.lcm A B →
  hcf = 12 →
  lcm = 396 →
  A = 36 →
  B = 132 := by
sorry

end NUMINAMATH_CALUDE_other_number_proof_l407_40737


namespace NUMINAMATH_CALUDE_right_triangle_tan_b_l407_40735

theorem right_triangle_tan_b (A B C : Real) (h1 : 0 < A ∧ A < π/2) (h2 : 0 < B ∧ B < π/2) : 
  A + B + Real.pi/2 = Real.pi → Real.sin A = 2/3 → Real.tan B = Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_tan_b_l407_40735


namespace NUMINAMATH_CALUDE_lcm_12_15_18_l407_40730

theorem lcm_12_15_18 : Nat.lcm (Nat.lcm 12 15) 18 = 180 := by
  sorry

end NUMINAMATH_CALUDE_lcm_12_15_18_l407_40730


namespace NUMINAMATH_CALUDE_circle_area_increase_l407_40786

theorem circle_area_increase (r : ℝ) (h : r > 0) : 
  (π * (2 * r)^2 - π * r^2) / (π * r^2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_increase_l407_40786


namespace NUMINAMATH_CALUDE_least_n_proof_l407_40768

/-- The number of distinct positive divisors of n -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- Check if n can be expressed as m * 12^k where 12 is not a divisor of m -/
def is_valid_form (n : ℕ) : Prop :=
  ∃ (m k : ℕ), n = m * (12^k) ∧ ¬(12 ∣ m)

/-- Find the least n satisfying the conditions -/
def least_n : ℕ := sorry

/-- Find m and k for the least n -/
def find_m_k (n : ℕ) : ℕ × ℕ := sorry

theorem least_n_proof :
  num_divisors least_n = 2023 ∧
  is_valid_form least_n ∧
  (let (m, k) := find_m_k least_n
   m + k = 6569) :=
sorry

end NUMINAMATH_CALUDE_least_n_proof_l407_40768


namespace NUMINAMATH_CALUDE_car_dealership_problem_l407_40728

/- Define the prices of models A and B -/
def price_A : ℝ := 20
def price_B : ℝ := 15

/- Define the sales data for two weeks -/
def week1_sales : ℝ := 65
def week1_units_A : ℕ := 1
def week1_units_B : ℕ := 3

def week2_sales : ℝ := 155
def week2_units_A : ℕ := 4
def week2_units_B : ℕ := 5

/- Define the company's purchase constraints -/
def total_units : ℕ := 8
def min_cost : ℝ := 145
def max_cost : ℝ := 153

/- Define a function to calculate the cost of a purchase plan -/
def purchase_cost (units_A : ℕ) : ℝ :=
  price_A * units_A + price_B * (total_units - units_A)

/- Define a function to check if a purchase plan is valid -/
def is_valid_plan (units_A : ℕ) : Prop :=
  units_A ≤ total_units ∧ 
  min_cost ≤ purchase_cost units_A ∧ 
  purchase_cost units_A ≤ max_cost

/- Theorem statement -/
theorem car_dealership_problem :
  /- Prices satisfy the sales data -/
  (price_A * week1_units_A + price_B * week1_units_B = week1_sales) ∧
  (price_A * week2_units_A + price_B * week2_units_B = week2_sales) ∧
  /- Exactly two valid purchase plans exist -/
  (∃ (plan1 plan2 : ℕ), 
    plan1 ≠ plan2 ∧ 
    is_valid_plan plan1 ∧ 
    is_valid_plan plan2 ∧
    (∀ (plan : ℕ), is_valid_plan plan → plan = plan1 ∨ plan = plan2)) ∧
  /- The most cost-effective plan is 5 units of A and 3 units of B -/
  (∀ (plan : ℕ), is_valid_plan plan → purchase_cost 5 ≤ purchase_cost plan) :=
by sorry

end NUMINAMATH_CALUDE_car_dealership_problem_l407_40728


namespace NUMINAMATH_CALUDE_unique_two_digit_beprisque_l407_40784

/-- A number is Beprisque if it's the only natural number between a prime number and a perfect square. -/
def isBeprisque (n : ℕ) : Prop :=
  ∃ (p q : ℕ), Nat.Prime p ∧ (∃ k, q = k * k) ∧
  ((p < n ∧ n < q) ∨ (q < n ∧ n < p)) ∧
  ∀ m, m ≠ n → ¬((p < m ∧ m < q) ∨ (q < m ∧ m < p))

/-- A number is a two-digit number if it's between 10 and 99, inclusive. -/
def isTwoDigit (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

/-- There is exactly one two-digit Beprisque number. -/
theorem unique_two_digit_beprisque :
  ∃! n, isTwoDigit n ∧ isBeprisque n :=
sorry

end NUMINAMATH_CALUDE_unique_two_digit_beprisque_l407_40784


namespace NUMINAMATH_CALUDE_hexagon_square_side_ratio_l407_40789

/-- Given a regular hexagon and a square with the same perimeter,
    this theorem proves that the ratio of the side length of the hexagon
    to the side length of the square is 2/3. -/
theorem hexagon_square_side_ratio (perimeter : ℝ) (h s : ℝ)
  (hexagon_perimeter : 6 * h = perimeter)
  (square_perimeter : 4 * s = perimeter)
  (positive_perimeter : perimeter > 0) :
  h / s = 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_hexagon_square_side_ratio_l407_40789


namespace NUMINAMATH_CALUDE_bush_height_after_two_years_l407_40734

/-- The height of a bush after a given number of years -/
def bush_height (initial_height : ℝ) (years : ℕ) : ℝ :=
  initial_height * 4^years

/-- Theorem stating the height of the bush after 2 years -/
theorem bush_height_after_two_years
  (h : bush_height (bush_height 1 0) 4 = 64) :
  bush_height (bush_height 1 0) 2 = 4 :=
by
  sorry

#check bush_height_after_two_years

end NUMINAMATH_CALUDE_bush_height_after_two_years_l407_40734


namespace NUMINAMATH_CALUDE_sum_of_squares_roots_l407_40753

theorem sum_of_squares_roots (x : ℝ) :
  Real.sqrt (64 - x^2) - Real.sqrt (36 - x^2) = 4 →
  Real.sqrt (64 - x^2) + Real.sqrt (36 - x^2) = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_roots_l407_40753


namespace NUMINAMATH_CALUDE_linear_polynomial_impossibility_l407_40708

theorem linear_polynomial_impossibility (a b : ℝ) : 
  ¬(∃ (f : ℝ → ℝ), 
    (∀ x, f x = a * x + b) ∧ 
    (|f 0 - 1| < 1) ∧ 
    (|f 1 - 3| < 1) ∧ 
    (|f 2 - 9| < 1)) := by
  sorry

end NUMINAMATH_CALUDE_linear_polynomial_impossibility_l407_40708


namespace NUMINAMATH_CALUDE_M_properties_M_remainder_l407_40795

/-- The greatest integer multiple of 16 with no repeated digits -/
def M : ℕ :=
  sorry

/-- Predicate to check if a natural number has no repeated digits -/
def has_no_repeated_digits (n : ℕ) : Prop :=
  sorry

theorem M_properties :
  M % 16 = 0 ∧
  has_no_repeated_digits M ∧
  ∀ n : ℕ, n % 16 = 0 → has_no_repeated_digits n → n ≤ M :=
sorry

theorem M_remainder :
  M % 1000 = 864 :=
sorry

end NUMINAMATH_CALUDE_M_properties_M_remainder_l407_40795


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l407_40767

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum : a 3 + a 8 = 22)
  (h_sixth : a 6 = 7) :
  a 5 = 15 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l407_40767


namespace NUMINAMATH_CALUDE_xy_sum_product_l407_40741

theorem xy_sum_product (x y : ℝ) (h1 : x * y = 3) (h2 : x + y = 5) :
  x^2 * y + x * y^2 = 15 := by
sorry

end NUMINAMATH_CALUDE_xy_sum_product_l407_40741


namespace NUMINAMATH_CALUDE_seventh_grade_rooms_l407_40713

/-- The number of rooms on the first floor where seventh-grade boys live -/
def num_rooms : ℕ := sorry

/-- The total number of students -/
def total_students : ℕ := sorry

theorem seventh_grade_rooms :
  (6 * (num_rooms - 1) = total_students) ∧
  (5 * num_rooms + 4 = total_students) →
  num_rooms = 10 := by
sorry

end NUMINAMATH_CALUDE_seventh_grade_rooms_l407_40713


namespace NUMINAMATH_CALUDE_fib_div_three_iff_index_div_four_l407_40764

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

theorem fib_div_three_iff_index_div_four (n : ℕ) : 
  3 ∣ fib n ↔ 4 ∣ n := by sorry

end NUMINAMATH_CALUDE_fib_div_three_iff_index_div_four_l407_40764


namespace NUMINAMATH_CALUDE_triangle_inequality_l407_40721

/-- The length of the shortest altitude of a triangle, or 0 if the points are collinear -/
noncomputable def m (P Q R : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

/-- For any four points on a plane, the inequality m(ABC) ≤ m(ABX) + m(AXC) + m(XBC) holds -/
theorem triangle_inequality (A B C X : EuclideanSpace ℝ (Fin 2)) :
  m A B C ≤ m A B X + m A X C + m X B C := by sorry

end NUMINAMATH_CALUDE_triangle_inequality_l407_40721


namespace NUMINAMATH_CALUDE_gcd_221_195_l407_40743

theorem gcd_221_195 : Nat.gcd 221 195 = 13 := by
  sorry

end NUMINAMATH_CALUDE_gcd_221_195_l407_40743


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l407_40750

theorem necessary_not_sufficient_condition (a : ℝ) :
  (((a - 1) * (a - 2) = 0) → (a = 2)) ∧
  ¬(∀ a : ℝ, ((a - 1) * (a - 2) = 0) ↔ (a = 2)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l407_40750


namespace NUMINAMATH_CALUDE_min_value_theorem_l407_40739

theorem min_value_theorem (a b x : ℝ) (ha : a > 0) (hb : b > 0) (hx : 0 < x ∧ x < 1) :
  a^2 / x + b^2 / (1 - x) ≥ (a + b)^2 ∧ 
  ∃ y, 0 < y ∧ y < 1 ∧ a^2 / y + b^2 / (1 - y) = (a + b)^2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l407_40739


namespace NUMINAMATH_CALUDE_large_monkey_doll_cost_l407_40732

/-- The cost of a large monkey doll satisfies the given conditions -/
theorem large_monkey_doll_cost : ∃ (L : ℚ), 
  (L > 0) ∧ 
  (320 / (L - 4) = 320 / L + 40) ∧ 
  L = 8 := by
  sorry

end NUMINAMATH_CALUDE_large_monkey_doll_cost_l407_40732


namespace NUMINAMATH_CALUDE_third_month_sale_l407_40782

theorem third_month_sale
  (average : ℕ)
  (month1 month2 month4 month5 month6 : ℕ)
  (h1 : average = 6800)
  (h2 : month1 = 6435)
  (h3 : month2 = 6927)
  (h4 : month4 = 7230)
  (h5 : month5 = 6562)
  (h6 : month6 = 6791)
  : ∃ month3 : ℕ, 
    month3 = 6855 ∧ 
    (month1 + month2 + month3 + month4 + month5 + month6) / 6 = average :=
by sorry

end NUMINAMATH_CALUDE_third_month_sale_l407_40782


namespace NUMINAMATH_CALUDE_marks_initial_friends_l407_40758

/-- Calculates the initial number of friends Mark had -/
def initial_friends (kept_percentage : ℚ) (contacted_percentage : ℚ) (response_rate : ℚ) (final_friends : ℕ) : ℚ :=
  final_friends / (kept_percentage + contacted_percentage * response_rate)

/-- Proves that Mark initially had 100 friends -/
theorem marks_initial_friends :
  let kept_percentage : ℚ := 2/5
  let contacted_percentage : ℚ := 3/5
  let response_rate : ℚ := 1/2
  let final_friends : ℕ := 70
  initial_friends kept_percentage contacted_percentage response_rate final_friends = 100 := by
  sorry

#eval initial_friends (2/5) (3/5) (1/2) 70

end NUMINAMATH_CALUDE_marks_initial_friends_l407_40758


namespace NUMINAMATH_CALUDE_unique_five_digit_number_l407_40757

def is_valid_digit (d : ℕ) : Prop := d ≥ 1 ∧ d ≤ 5

def digits_to_number (a b c : ℕ) : ℕ := 100 * a + 10 * b + c

theorem unique_five_digit_number 
  (P Q R S T : ℕ) 
  (h_valid : ∀ d ∈ [P, Q, R, S, T], is_valid_digit d)
  (h_distinct : P ≠ Q ∧ P ≠ R ∧ P ≠ S ∧ P ≠ T ∧ Q ≠ R ∧ Q ≠ S ∧ Q ≠ T ∧ R ≠ S ∧ R ≠ T ∧ S ≠ T)
  (h_div_4 : digits_to_number P Q R % 4 = 0)
  (h_div_5 : digits_to_number Q R S % 5 = 0)
  (h_div_3 : digits_to_number R S T % 3 = 0) :
  P = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_five_digit_number_l407_40757


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l407_40773

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Theorem statement
theorem sufficient_not_necessary
  (f : ℝ → ℝ) (h : OddFunction f) :
  (∀ x₁ x₂ : ℝ, x₁ + x₂ = 0 → f x₁ + f x₂ = 0) ∧
  (∃ x₁ x₂ : ℝ, f x₁ + f x₂ = 0 ∧ x₁ + x₂ ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l407_40773


namespace NUMINAMATH_CALUDE_division_problem_l407_40785

theorem division_problem (dividend quotient remainder divisor : ℕ) 
  (h1 : dividend = 161)
  (h2 : quotient = 10)
  (h3 : remainder = 1)
  (h4 : dividend = divisor * quotient + remainder) :
  divisor = 16 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l407_40785


namespace NUMINAMATH_CALUDE_sum_negative_forty_to_sixty_l407_40763

def sum_range (a b : Int) : Int :=
  (b - a + 1) * (a + b) / 2

theorem sum_negative_forty_to_sixty :
  sum_range (-40) 60 = 1010 := by
  sorry

end NUMINAMATH_CALUDE_sum_negative_forty_to_sixty_l407_40763


namespace NUMINAMATH_CALUDE_max_revenue_at_50_10_l407_40790

/-- Represents the parking lot problem -/
structure ParkingLot where
  carSpace : ℝ
  busSpace : ℝ
  carFee : ℝ
  busFee : ℝ
  totalArea : ℝ
  maxVehicles : ℕ

/-- Revenue function for the parking lot -/
def revenue (p : ParkingLot) (x y : ℝ) : ℝ :=
  p.carFee * x + p.busFee * y

/-- Theorem stating that (50, 10) maximizes revenue for the given parking lot problem -/
theorem max_revenue_at_50_10 (p : ParkingLot)
  (h1 : p.carSpace = 6)
  (h2 : p.busSpace = 30)
  (h3 : p.carFee = 2.5)
  (h4 : p.busFee = 7.5)
  (h5 : p.totalArea = 600)
  (h6 : p.maxVehicles = 60) :
  ∀ x y : ℝ,
  x ≥ 0 → y ≥ 0 →
  x + y ≤ p.maxVehicles →
  p.carSpace * x + p.busSpace * y ≤ p.totalArea →
  revenue p x y ≤ revenue p 50 10 := by
  sorry


end NUMINAMATH_CALUDE_max_revenue_at_50_10_l407_40790


namespace NUMINAMATH_CALUDE_gcd_digits_bound_l407_40781

theorem gcd_digits_bound (a b : ℕ) : 
  (1000000 ≤ a ∧ a < 10000000) →
  (1000000 ≤ b ∧ b < 10000000) →
  (100000000000 ≤ Nat.lcm a b ∧ Nat.lcm a b < 1000000000000) →
  Nat.gcd a b < 1000 :=
by sorry

end NUMINAMATH_CALUDE_gcd_digits_bound_l407_40781


namespace NUMINAMATH_CALUDE_debate_club_committee_compositions_l407_40769

def total_candidates : ℕ := 20
def past_members : ℕ := 10
def committee_size : ℕ := 5
def min_past_members : ℕ := 3

theorem debate_club_committee_compositions :
  (Nat.choose past_members min_past_members * Nat.choose (total_candidates - past_members) (committee_size - min_past_members)) +
  (Nat.choose past_members (min_past_members + 1) * Nat.choose (total_candidates - past_members) (committee_size - (min_past_members + 1))) +
  (Nat.choose past_members committee_size) = 7752 := by
  sorry

end NUMINAMATH_CALUDE_debate_club_committee_compositions_l407_40769


namespace NUMINAMATH_CALUDE_third_studio_students_l407_40796

theorem third_studio_students (total : ℕ) (first : ℕ) (second : ℕ) 
  (h_total : total = 376)
  (h_first : first = 110)
  (h_second : second = 135) :
  total - first - second = 131 := by
  sorry

end NUMINAMATH_CALUDE_third_studio_students_l407_40796


namespace NUMINAMATH_CALUDE_expression_evaluation_l407_40715

theorem expression_evaluation :
  let a : ℚ := 1/2
  let b : ℚ := -1/3
  5 * (3 * a^2 * b - a * b^2) - 4 * (-a * b^2 + 3 * a^2 * b) = -11/36 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l407_40715


namespace NUMINAMATH_CALUDE_taxi_charge_calculation_l407_40752

/-- Taxi service charge calculation -/
theorem taxi_charge_calculation 
  (initial_fee : ℝ) 
  (total_charge : ℝ) 
  (trip_distance : ℝ) 
  (segment_length : ℝ) : 
  initial_fee = 2.25 →
  total_charge = 4.5 →
  trip_distance = 3.6 →
  segment_length = 2/5 →
  (total_charge - initial_fee) / (trip_distance / segment_length) = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_taxi_charge_calculation_l407_40752


namespace NUMINAMATH_CALUDE_inequality_solution_implies_n_range_l407_40759

theorem inequality_solution_implies_n_range (n : ℝ) : 
  (∀ x : ℝ, ((n - 3) * x > 2) ↔ (x < 2 / (n - 3))) → n < 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_implies_n_range_l407_40759


namespace NUMINAMATH_CALUDE_parabola_properties_l407_40798

/-- Parabola C with vertex at origin and focus on y-axis -/
structure Parabola where
  focus : ℝ
  equation : ℝ → ℝ → Prop

/-- Point on the parabola -/
structure PointOnParabola (C : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : C.equation x y

/-- Line segment on the parabola -/
structure LineSegmentOnParabola (C : Parabola) where
  A : PointOnParabola C
  B : PointOnParabola C

/-- Triangle on the parabola -/
structure TriangleOnParabola (C : Parabola) where
  A : PointOnParabola C
  B : PointOnParabola C
  D : PointOnParabola C

theorem parabola_properties (C : Parabola) (Q : PointOnParabola C) 
    (AB : LineSegmentOnParabola C) (M : ℝ) (ABD : TriangleOnParabola C) :
  Q.x = Real.sqrt 8 ∧ Q.y = 2 ∧ (Q.x - C.focus)^2 + Q.y^2 = 9 →
  (∃ (m : ℝ), m > 0 ∧ 
    (∃ (k : ℝ), AB.A.y = k * AB.A.x + m ∧ AB.B.y = k * AB.B.x + m) ∧
    AB.A.x * AB.B.x + AB.A.y * AB.B.y = 0) →
  ABD.D.x < ABD.A.x ∧ ABD.A.x < ABD.B.x ∧
  (ABD.B.x - ABD.A.x)^2 + (ABD.B.y - ABD.A.y)^2 = 
    (ABD.D.x - ABD.A.x)^2 + (ABD.D.y - ABD.A.y)^2 ∧
  (ABD.B.x - ABD.A.x) * (ABD.D.x - ABD.A.x) + 
    (ABD.B.y - ABD.A.y) * (ABD.D.y - ABD.A.y) = 0 →
  C.equation = (fun x y => x^2 = 4*y) ∧ 
  M = 4 ∧
  (∀ (ABD' : TriangleOnParabola C), 
    ABD'.D.x < ABD'.A.x ∧ ABD'.A.x < ABD'.B.x ∧
    (ABD'.B.x - ABD'.A.x)^2 + (ABD'.B.y - ABD'.A.y)^2 = 
      (ABD'.D.x - ABD'.A.x)^2 + (ABD'.D.y - ABD'.A.y)^2 ∧
    (ABD'.B.x - ABD'.A.x) * (ABD'.D.x - ABD'.A.x) + 
      (ABD'.B.y - ABD'.A.y) * (ABD'.D.y - ABD'.A.y) = 0 →
    (ABD'.B.x - ABD'.A.x) * (ABD'.D.y - ABD'.A.y) -
      (ABD'.B.y - ABD'.A.y) * (ABD'.D.x - ABD'.A.x) ≥ 8) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l407_40798


namespace NUMINAMATH_CALUDE_currency_exchange_problem_l407_40772

def exchange_rate : ℚ := 9 / 6

def spent_amount : ℕ := 45

theorem currency_exchange_problem (d : ℕ) :
  (d : ℚ) * exchange_rate - spent_amount = d →
  (d / 10 + d % 10 : ℕ) = 9 := by
  sorry

end NUMINAMATH_CALUDE_currency_exchange_problem_l407_40772


namespace NUMINAMATH_CALUDE_rice_purchase_l407_40771

theorem rice_purchase (rice_price lentil_price total_weight total_cost : ℚ)
  (h1 : rice_price = 105/100)
  (h2 : lentil_price = 33/100)
  (h3 : total_weight = 30)
  (h4 : total_cost = 2340/100) :
  ∃ (rice_weight : ℚ),
    rice_weight + (total_weight - rice_weight) = total_weight ∧
    rice_price * rice_weight + lentil_price * (total_weight - rice_weight) = total_cost ∧
    rice_weight = 75/4 := by
  sorry

end NUMINAMATH_CALUDE_rice_purchase_l407_40771


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l407_40720

/-- Two points are symmetric about a line if their midpoint lies on that line -/
def symmetric_points (x₁ y₁ x₂ y₂ k b : ℝ) : Prop :=
  let mx := (x₁ + x₂) / 2
  let my := (y₁ + y₂) / 2
  k * mx - my + b = 0

/-- The theorem statement -/
theorem symmetric_points_sum (m k : ℝ) :
  symmetric_points 1 2 (-1) m k 3 → m + k = 5 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l407_40720


namespace NUMINAMATH_CALUDE_parabola_properties_l407_40779

theorem parabola_properties (a b c : ℝ) (h1 : a < 0) 
  (h2 : a * (-3)^2 + b * (-3) + c = 0) 
  (h3 : a * 1^2 + b * 1 + c = 0) : 
  (b^2 - 4*a*c > 0) ∧ (3*b + 2*c = 0) := by sorry

end NUMINAMATH_CALUDE_parabola_properties_l407_40779


namespace NUMINAMATH_CALUDE_max_value_ab_l407_40707

theorem max_value_ab (a b : ℝ) (h : ∀ x : ℝ, Real.exp (x + 1) ≥ a * x + b) :
  a * b ≤ Real.exp 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_ab_l407_40707


namespace NUMINAMATH_CALUDE_probability_eight_distinct_rolls_l407_40792

/-- The number of sides on each die -/
def num_sides : ℕ := 8

/-- The number of dice rolled -/
def num_dice : ℕ := 8

/-- The probability of rolling eight standard, eight-sided dice and getting eight distinct numbers -/
def probability_distinct_rolls : ℚ :=
  (Nat.factorial num_dice) / (num_sides ^ num_dice)

theorem probability_eight_distinct_rolls :
  probability_distinct_rolls = 5 / 1296 := by
  sorry

end NUMINAMATH_CALUDE_probability_eight_distinct_rolls_l407_40792


namespace NUMINAMATH_CALUDE_existence_of_integers_l407_40719

theorem existence_of_integers : ∃ (a b c d : ℤ), 
  d ≥ 1 ∧ 
  b % d = c % d ∧ 
  a ∣ b ∧ a ∣ c ∧ 
  (b / a) % d ≠ (c / a) % d := by
  sorry

end NUMINAMATH_CALUDE_existence_of_integers_l407_40719


namespace NUMINAMATH_CALUDE_wrong_to_right_exists_l407_40742

-- Define a type for single-digit numbers (1-9)
def Digit := {n : ℕ | 1 ≤ n ∧ n ≤ 9}

-- Define a function to convert a 5-digit number to its numerical value
def to_number (a b c d e : Digit) : ℕ :=
  10000 * a + 1000 * b + 100 * c + 10 * d + e

-- State the theorem
theorem wrong_to_right_exists :
  ∃ (W R O N G I H T : Digit),
    (W ≠ R) ∧ (W ≠ O) ∧ (W ≠ N) ∧ (W ≠ G) ∧ (W ≠ I) ∧ (W ≠ H) ∧ (W ≠ T) ∧
    (R ≠ O) ∧ (R ≠ N) ∧ (R ≠ G) ∧ (R ≠ I) ∧ (R ≠ H) ∧ (R ≠ T) ∧
    (O ≠ N) ∧ (O ≠ G) ∧ (O ≠ I) ∧ (O ≠ H) ∧ (O ≠ T) ∧
    (N ≠ G) ∧ (N ≠ I) ∧ (N ≠ H) ∧ (N ≠ T) ∧
    (G ≠ I) ∧ (G ≠ H) ∧ (G ≠ T) ∧
    (I ≠ H) ∧ (I ≠ T) ∧
    (H ≠ T) ∧
    to_number W R O N G + to_number W R O N G = to_number R I G H T :=
by sorry

end NUMINAMATH_CALUDE_wrong_to_right_exists_l407_40742


namespace NUMINAMATH_CALUDE_fraction_equality_l407_40745

theorem fraction_equality : (1992^2 - 1985^2) / (2001^2 - 1976^2) = 7 / 25 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l407_40745


namespace NUMINAMATH_CALUDE_pitcher_distribution_l407_40775

theorem pitcher_distribution (C : ℝ) (h : C > 0) : 
  let juice_amount : ℝ := (2/3) * C
  let cups : ℕ := 6
  let juice_per_cup : ℝ := juice_amount / cups
  juice_per_cup / C = 1/9 := by sorry

end NUMINAMATH_CALUDE_pitcher_distribution_l407_40775


namespace NUMINAMATH_CALUDE_markup_rate_proof_l407_40718

theorem markup_rate_proof (S : ℝ) (h_positive : S > 0) : 
  let profit_rate : ℝ := 0.20
  let expense_rate : ℝ := 0.10
  let C : ℝ := S * (1 - profit_rate - expense_rate)
  ((S - C) / C) * 100 = 42.857 := by
sorry

end NUMINAMATH_CALUDE_markup_rate_proof_l407_40718


namespace NUMINAMATH_CALUDE_book_reading_time_l407_40705

/-- Calculates the number of weeks needed to read a book given the total pages and pages read per week. -/
def weeks_to_read (total_pages : ℕ) (pages_per_day : ℕ) (reading_days_per_week : ℕ) : ℕ :=
  total_pages / (pages_per_day * reading_days_per_week)

/-- Proves that it takes 7 weeks to read a 2100-page book when reading 100 pages on 3 days per week. -/
theorem book_reading_time : weeks_to_read 2100 100 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_book_reading_time_l407_40705


namespace NUMINAMATH_CALUDE_translation_theorem_l407_40755

/-- A translation in the complex plane that moves 1 - 3i to 5 + 2i also moves 3 - 4i to 7 + i -/
theorem translation_theorem (t : ℂ → ℂ) :
  (t (1 - 3*I) = 5 + 2*I) →
  (∃ w : ℂ, ∀ z : ℂ, t z = z + w) →
  t (3 - 4*I) = 7 + I :=
by sorry

end NUMINAMATH_CALUDE_translation_theorem_l407_40755


namespace NUMINAMATH_CALUDE_triangle_properties_l407_40754

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  0 < a ∧ 0 < b ∧ 0 < c ∧
  A + B + C = Real.pi ∧
  (Real.cos A - 2 * Real.cos C) / Real.cos B = (2 * c - a) / b ∧
  Real.cos B = 1/4 ∧
  1/2 * a * c * Real.sin B = Real.sqrt 15 / 4 →
  Real.sin C / Real.sin A = 2 ∧
  a + b + c = 5 := by sorry

end NUMINAMATH_CALUDE_triangle_properties_l407_40754


namespace NUMINAMATH_CALUDE_difference_of_squares_l407_40778

theorem difference_of_squares (m : ℝ) : m^2 - 16 = (m + 4) * (m - 4) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l407_40778


namespace NUMINAMATH_CALUDE_equation_solutions_l407_40727

-- Define the equation
def equation (r p : ℤ) : Prop := r^2 - r*(p + 6) + p^2 + 5*p + 6 = 0

-- Define the set of solution pairs
def solution_set : Set (ℤ × ℤ) := {(3,1), (4,1), (0,-2), (4,-2), (0,-3), (3,-3)}

-- Theorem statement
theorem equation_solutions :
  ∀ (r p : ℤ), equation r p ↔ (r, p) ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l407_40727


namespace NUMINAMATH_CALUDE_books_for_sale_l407_40706

/-- The total number of books for sale is the sum of initial books and additional books found. -/
theorem books_for_sale (initial_books additional_books : ℕ) :
  initial_books = 33 → additional_books = 26 →
  initial_books + additional_books = 59 := by
  sorry

end NUMINAMATH_CALUDE_books_for_sale_l407_40706


namespace NUMINAMATH_CALUDE_race_time_calculation_race_problem_l407_40797

theorem race_time_calculation (race_distance : ℝ) (a_time : ℝ) (beat_distance : ℝ) : ℝ :=
  let a_speed := race_distance / a_time
  let b_distance_when_a_finishes := race_distance - beat_distance
  let b_speed := b_distance_when_a_finishes / a_time
  let b_time := race_distance / b_speed
  b_time

theorem race_problem : 
  race_time_calculation 130 20 26 = 25 := by sorry

end NUMINAMATH_CALUDE_race_time_calculation_race_problem_l407_40797


namespace NUMINAMATH_CALUDE_max_area_triangle_AOB_l407_40749

/-- The maximum area of triangle AOB formed by the intersection points of
    two lines and a curve in polar coordinates. -/
theorem max_area_triangle_AOB :
  ∀ α : Real,
  0 < α →
  α < π / 2 →
  let C₁ := {θ : Real | θ = α}
  let C₂ := {θ : Real | θ = α + π / 2}
  let C₃ := {(ρ, θ) : Real × Real | ρ = 8 * Real.sin θ}
  let A := (8 * Real.sin α, α)
  let B := (8 * Real.cos α, α + π / 2)
  A.1 ≠ 0 ∨ A.2 ≠ 0 →
  B.1 ≠ 0 ∨ B.2 ≠ 0 →
  (∃ (S : Real → Real),
    (∀ α, S α = (1/2) * 8 * Real.sin α * 8 * Real.cos α) ∧
    (∀ α, S α ≤ 16)) :=
by sorry

end NUMINAMATH_CALUDE_max_area_triangle_AOB_l407_40749


namespace NUMINAMATH_CALUDE_amys_haircut_l407_40780

/-- Amy's haircut problem -/
theorem amys_haircut (initial_length : ℝ) (final_length : ℝ) (cut_length : ℝ)
  (h1 : initial_length = 11)
  (h2 : final_length = 7)
  (h3 : cut_length = initial_length - final_length) :
  cut_length = 4 := by sorry

end NUMINAMATH_CALUDE_amys_haircut_l407_40780


namespace NUMINAMATH_CALUDE_central_angle_nairobi_lima_l407_40723

/-- Represents a point on Earth's surface -/
structure EarthPoint where
  latitude : Real
  longitude : Real

/-- Calculates the central angle between two points on Earth -/
def centralAngle (p1 p2 : EarthPoint) : Real :=
  |p1.longitude - p2.longitude|

theorem central_angle_nairobi_lima :
  let nairobi : EarthPoint := { latitude := -1, longitude := 36 }
  let lima : EarthPoint := { latitude := -12, longitude := -77 }
  centralAngle nairobi lima = 113 := by
  sorry

end NUMINAMATH_CALUDE_central_angle_nairobi_lima_l407_40723


namespace NUMINAMATH_CALUDE_dan_final_marbles_l407_40733

/-- The number of marbles Dan has after giving some away and receiving more. -/
def final_marbles (initial : ℕ) (given_mary : ℕ) (given_peter : ℕ) (received : ℕ) : ℕ :=
  initial - given_mary - given_peter + received

/-- Theorem stating that Dan has 98 marbles at the end. -/
theorem dan_final_marbles :
  final_marbles 128 24 16 10 = 98 := by
  sorry

end NUMINAMATH_CALUDE_dan_final_marbles_l407_40733


namespace NUMINAMATH_CALUDE_max_sum_constrained_max_sum_constrained_attained_l407_40738

theorem max_sum_constrained (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 16 * x * y * z = (x + y)^2 * (x + z)^2) :
  x + y + z ≤ 4 := by
sorry

theorem max_sum_constrained_attained :
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧
  16 * x * y * z = (x + y)^2 * (x + z)^2 ∧
  x + y + z = 4 := by
sorry

end NUMINAMATH_CALUDE_max_sum_constrained_max_sum_constrained_attained_l407_40738


namespace NUMINAMATH_CALUDE_female_population_count_l407_40746

def total_population : ℕ := 5000
def male_population : ℕ := 2000
def females_with_glasses : ℕ := 900
def female_glasses_percentage : ℚ := 30 / 100

theorem female_population_count : 
  ∃ (female_population : ℕ), 
    female_population = total_population - male_population ∧
    female_population = females_with_glasses / female_glasses_percentage :=
by sorry

end NUMINAMATH_CALUDE_female_population_count_l407_40746


namespace NUMINAMATH_CALUDE_cone_base_radius_l407_40717

/-- Proves that a cone with a lateral surface made from a sector of a circle
    with radius 9 cm and central angle 240° has a circular base with radius 6 cm. -/
theorem cone_base_radius (r : ℝ) (θ : ℝ) (h1 : r = 9) (h2 : θ = 240 * π / 180) :
  r * θ / (2 * π) = 6 := by
  sorry

end NUMINAMATH_CALUDE_cone_base_radius_l407_40717


namespace NUMINAMATH_CALUDE_rahul_share_l407_40788

/-- Calculates the share of payment for a worker given the total payment and the time taken by each worker --/
def calculate_share (total_payment : ℚ) (time_worker1 time_worker2 : ℚ) : ℚ :=
  let work_rate1 := 1 / time_worker1
  let work_rate2 := 1 / time_worker2
  let combined_rate := work_rate1 + work_rate2
  let share_ratio := work_rate1 / combined_rate
  total_payment * share_ratio

/-- Proves that Rahul's share of the payment is 900 given the conditions --/
theorem rahul_share :
  let total_payment : ℚ := 2250
  let rahul_time : ℚ := 3
  let rajesh_time : ℚ := 2
  calculate_share total_payment rahul_time rajesh_time = 900 := by
  sorry

#eval calculate_share 2250 3 2

end NUMINAMATH_CALUDE_rahul_share_l407_40788


namespace NUMINAMATH_CALUDE_negative_sixty_four_to_four_thirds_l407_40783

theorem negative_sixty_four_to_four_thirds (x : ℝ) : x = (-64)^(4/3) → x = 256 := by
  sorry

end NUMINAMATH_CALUDE_negative_sixty_four_to_four_thirds_l407_40783


namespace NUMINAMATH_CALUDE_bargaining_range_l407_40776

def marked_price : ℝ := 100

def min_markup_percent : ℝ := 50
def max_markup_percent : ℝ := 100

def min_profit_percent : ℝ := 20

def lower_bound : ℝ := 60
def upper_bound : ℝ := 80

theorem bargaining_range :
  ∀ (cost_price : ℝ),
    (cost_price * (1 + min_markup_percent / 100) ≤ marked_price) →
    (cost_price * (1 + max_markup_percent / 100) ≥ marked_price) →
    (lower_bound ≥ cost_price * (1 + min_profit_percent / 100)) ∧
    (upper_bound ≤ marked_price) ∧
    (lower_bound ≤ upper_bound) :=
by sorry

end NUMINAMATH_CALUDE_bargaining_range_l407_40776
