import Mathlib

namespace NUMINAMATH_CALUDE_largest_prime_factor_l1117_111701

theorem largest_prime_factor (n : ℕ) : 
  (∃ p : ℕ, Nat.Prime p ∧ p ∣ n ∧ ∀ q : ℕ, Nat.Prime q → q ∣ n → q ≤ p) →
  (∃ p : ℕ, Nat.Prime p ∧ p ∣ (16^4 + 3*16^2 + 2 - 17^4) ∧ 
    ∀ q : ℕ, Nat.Prime q → q ∣ (16^4 + 3*16^2 + 2 - 17^4) → q ≤ p) →
  (∃ p : ℕ, p = 547 ∧ Nat.Prime p ∧ p ∣ (16^4 + 3*16^2 + 2 - 17^4) ∧ 
    ∀ q : ℕ, Nat.Prime q → q ∣ (16^4 + 3*16^2 + 2 - 17^4) → q ≤ p) :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_l1117_111701


namespace NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l1117_111702

/-- Given two vectors a and b in R², prove that if they are perpendicular
    and a = (1, -1) and b = (m+1, 2m-4), then m = 5. -/
theorem perpendicular_vectors_m_value (m : ℝ) :
  let a : Fin 2 → ℝ := ![1, -1]
  let b : Fin 2 → ℝ := ![m+1, 2*m-4]
  (∀ i, a i * b i = 0) → m = 5 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l1117_111702


namespace NUMINAMATH_CALUDE_solution_value_l1117_111791

theorem solution_value (x a : ℝ) (h : 5 * 3 - a = 8) : a = 7 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l1117_111791


namespace NUMINAMATH_CALUDE_second_polygon_sides_l1117_111766

/-- Given two regular polygons with the same perimeter, where one has 50 sides
    and a side length three times as long as the other, prove that the number
    of sides of the second polygon is 150. -/
theorem second_polygon_sides (s : ℝ) (n : ℕ) :
  s > 0 →
  50 * (3 * s) = n * s →
  n = 150 := by
  sorry

end NUMINAMATH_CALUDE_second_polygon_sides_l1117_111766


namespace NUMINAMATH_CALUDE_implication_q_not_p_l1117_111771

theorem implication_q_not_p (x : ℝ) : x^2 - x - 2 > 0 → x ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_implication_q_not_p_l1117_111771


namespace NUMINAMATH_CALUDE_profit_division_ratio_l1117_111733

/-- Represents the capital contribution and duration for a business partner -/
structure Contribution where
  capital : ℕ
  months : ℕ

/-- Calculates the total capital contribution over time -/
def totalContribution (c : Contribution) : ℕ := c.capital * c.months

/-- Simplifies a ratio by dividing both numbers by their GCD -/
def simplifyRatio (a b : ℕ) : ℕ × ℕ :=
  let gcd := Nat.gcd a b
  (a / gcd, b / gcd)

/-- The main theorem stating the profit division ratio -/
theorem profit_division_ratio 
  (a_initial : ℕ) 
  (b_capital : ℕ) 
  (total_months : ℕ) 
  (b_join_month : ℕ) 
  (h1 : a_initial = 3500)
  (h2 : b_capital = 31500)
  (h3 : total_months = 12)
  (h4 : b_join_month = 10) :
  simplifyRatio 
    (totalContribution { capital := a_initial, months := total_months })
    (totalContribution { capital := b_capital, months := total_months - b_join_month }) 
  = (2, 3) := by
  sorry

end NUMINAMATH_CALUDE_profit_division_ratio_l1117_111733


namespace NUMINAMATH_CALUDE_min_value_polynomial_l1117_111793

theorem min_value_polynomial (x y : ℝ) : 
  x^2 + y^2 - 6*x + 8*y + 7 ≥ -18 :=
by sorry

end NUMINAMATH_CALUDE_min_value_polynomial_l1117_111793


namespace NUMINAMATH_CALUDE_smallest_permutation_number_is_1089_l1117_111779

/-- A function that returns true if two natural numbers are permutations of each other's digits -/
def is_digit_permutation (a b : ℕ) : Prop := sorry

/-- A function that returns the smallest natural number satisfying the permutation condition when multiplied by 9 -/
noncomputable def smallest_permutation_number : ℕ := sorry

theorem smallest_permutation_number_is_1089 :
  smallest_permutation_number = 1089 ∧
  is_digit_permutation smallest_permutation_number (9 * smallest_permutation_number) ∧
  ∀ n < smallest_permutation_number, ¬is_digit_permutation n (9 * n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_permutation_number_is_1089_l1117_111779


namespace NUMINAMATH_CALUDE_area_of_triangle_FYG_l1117_111717

theorem area_of_triangle_FYG (EF GH : ℝ) (area_EFGH : ℝ) (angle_E : ℝ) :
  EF = 15 →
  GH = 25 →
  area_EFGH = 400 →
  angle_E = 30 * π / 180 →
  ∃ (area_FYG : ℝ), area_FYG = 240 - 45 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_area_of_triangle_FYG_l1117_111717


namespace NUMINAMATH_CALUDE_marias_number_problem_l1117_111705

theorem marias_number_problem (x : ℝ) : 
  (((x - 3) * 3 + 3) / 3 = 10) → x = 12 := by
  sorry

end NUMINAMATH_CALUDE_marias_number_problem_l1117_111705


namespace NUMINAMATH_CALUDE_extremum_implies_a_equals_one_l1117_111744

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*x*a

-- State the theorem
theorem extremum_implies_a_equals_one (a : ℝ) : 
  (∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), x ≠ 1 ∧ |x - 1| < ε → f a x ≤ f a 1) →
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_extremum_implies_a_equals_one_l1117_111744


namespace NUMINAMATH_CALUDE_min_values_theorem_l1117_111703

theorem min_values_theorem (r s : ℝ) (hr : r > 0) (hs : s > 0) 
  (h : (r + s - r * s) * (r + s + r * s) = r * s) : 
  (∃ (r' s' : ℝ), r' > 0 ∧ s' > 0 ∧ 
    (r' + s' - r' * s') * (r' + s' + r' * s') = r' * s' ∧
    r + s - r * s ≥ -3 + 2 * Real.sqrt 3 ∧
    r + s + r * s ≥ 3 + 2 * Real.sqrt 3) ∧
  (r + s - r * s = -3 + 2 * Real.sqrt 3 ∨ r + s + r * s = 3 + 2 * Real.sqrt 3 → 
    r = Real.sqrt 3 ∧ s = Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_min_values_theorem_l1117_111703


namespace NUMINAMATH_CALUDE_arithmetic_sequence_max_product_l1117_111773

theorem arithmetic_sequence_max_product (a : ℕ → ℝ) (d : ℝ) :
  (∀ n : ℕ, a (n + 1) = a n + d) →  -- arithmetic sequence condition
  a 8 + a 9 + a 10 = 24 →           -- given sum condition
  ∃ m : ℝ, m = 2 ∧ ∀ d' : ℝ, a 1 * d' ≤ m := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_max_product_l1117_111773


namespace NUMINAMATH_CALUDE_games_planned_this_month_l1117_111770

theorem games_planned_this_month
  (total_attended : ℕ)
  (planned_last_month : ℕ)
  (missed : ℕ)
  (h1 : total_attended = 12)
  (h2 : planned_last_month = 17)
  (h3 : missed = 16)
  : ℕ := by
  sorry

end NUMINAMATH_CALUDE_games_planned_this_month_l1117_111770


namespace NUMINAMATH_CALUDE_not_sufficient_nor_necessary_l1117_111747

theorem not_sufficient_nor_necessary (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ¬(∀ a b : ℝ, a > b → (1 / a) < (1 / b)) ∧
  ¬(∀ a b : ℝ, (1 / a) < (1 / b) → a > b) :=
sorry

end NUMINAMATH_CALUDE_not_sufficient_nor_necessary_l1117_111747


namespace NUMINAMATH_CALUDE_smallest_integer_divisible_l1117_111724

theorem smallest_integer_divisible (n : ℕ) : n = 43179 ↔ 
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 21) = 48 * k)) ∧
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 21) = 64 * k)) ∧
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 21) = 75 * k)) ∧
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 21) = 108 * k)) ∧
  (∃ k₁ k₂ k₃ k₄ : ℕ, (n + 21) = 48 * k₁ ∧ (n + 21) = 64 * k₂ ∧ (n + 21) = 75 * k₃ ∧ (n + 21) = 108 * k₄) := by
sorry

end NUMINAMATH_CALUDE_smallest_integer_divisible_l1117_111724


namespace NUMINAMATH_CALUDE_responses_needed_l1117_111711

/-- Given a 65% response rate and 461.54 questionnaires mailed, prove that 300 responses are needed -/
theorem responses_needed (response_rate : ℝ) (questionnaires_mailed : ℝ) : 
  response_rate = 0.65 → 
  questionnaires_mailed = 461.54 → 
  ⌊response_rate * questionnaires_mailed⌋ = 300 := by
sorry

end NUMINAMATH_CALUDE_responses_needed_l1117_111711


namespace NUMINAMATH_CALUDE_interval_equivalence_l1117_111789

-- Define the intervals as sets
def openRightInf (a : ℝ) : Set ℝ := {x | x > a}
def closedRightInf (a : ℝ) : Set ℝ := {x | x ≥ a}
def openLeftInf (b : ℝ) : Set ℝ := {x | x < b}
def closedLeftInf (b : ℝ) : Set ℝ := {x | x ≤ b}

-- State the theorem
theorem interval_equivalence (a b : ℝ) :
  (∀ x, x ∈ openRightInf a ↔ x > a) ∧
  (∀ x, x ∈ closedRightInf a ↔ x ≥ a) ∧
  (∀ x, x ∈ openLeftInf b ↔ x < b) ∧
  (∀ x, x ∈ closedLeftInf b ↔ x ≤ b) :=
by sorry

end NUMINAMATH_CALUDE_interval_equivalence_l1117_111789


namespace NUMINAMATH_CALUDE_last_nonzero_digit_aperiodic_l1117_111737

/-- d_n is the last nonzero digit of n! -/
def last_nonzero_digit (n : ℕ) : ℕ :=
  sorry

/-- The sequence d_n is aperiodic -/
theorem last_nonzero_digit_aperiodic :
  ∀ T n₀ : ℕ, ∃ n : ℕ, n ≥ n₀ ∧ last_nonzero_digit (n + T) ≠ last_nonzero_digit n :=
sorry

end NUMINAMATH_CALUDE_last_nonzero_digit_aperiodic_l1117_111737


namespace NUMINAMATH_CALUDE_two_digit_number_problem_l1117_111722

theorem two_digit_number_problem : ∃ x : ℕ, 
  10 ≤ x ∧ x < 100 ∧ 10 * x + 6 = x + 474 → x = 52 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_problem_l1117_111722


namespace NUMINAMATH_CALUDE_substitution_remainder_l1117_111761

/-- Calculates the number of substitution combinations for a given number of substitutions -/
def substitutionCombinations (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | k + 1 => 11 * (23 - k) * substitutionCombinations k

/-- The total number of substitution combinations for up to 5 substitutions -/
def totalCombinations : ℕ :=
  (List.range 6).map substitutionCombinations |>.sum

/-- Theorem stating the remainder when dividing the total number of substitution combinations by 1000 -/
theorem substitution_remainder :
  totalCombinations % 1000 = 586 := by
  sorry

end NUMINAMATH_CALUDE_substitution_remainder_l1117_111761


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l1117_111715

theorem polynomial_evaluation : 
  ∃ (x : ℝ), x > 0 ∧ x^2 - 3*x - 10 = 0 ∧ x^4 - 3*x^3 + 2*x^2 + 5*x - 7 = 318 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l1117_111715


namespace NUMINAMATH_CALUDE_reach_64_from_2_cannot_reach_2_2011_from_2_l1117_111795

def cube (x : ℚ) : ℚ := x^3

def div_by_8 (x : ℚ) : ℚ := x / 8

inductive Operation
| Cube
| DivBy8

def apply_operation (x : ℚ) (op : Operation) : ℚ :=
  match op with
  | Operation.Cube => cube x
  | Operation.DivBy8 => div_by_8 x

def can_reach (start : ℚ) (target : ℚ) : Prop :=
  ∃ (ops : List Operation), target = ops.foldl apply_operation start

theorem reach_64_from_2 : can_reach 2 64 := by sorry

theorem cannot_reach_2_2011_from_2 : ¬ can_reach 2 (2^2011) := by sorry

end NUMINAMATH_CALUDE_reach_64_from_2_cannot_reach_2_2011_from_2_l1117_111795


namespace NUMINAMATH_CALUDE_negation_existence_statement_l1117_111727

theorem negation_existence_statement (A : Set ℝ) :
  (¬ ∃ x ∈ A, x^2 - 2*x - 3 > 0) ↔ (∀ x ∈ A, x^2 - 2*x - 3 ≤ 0) :=
sorry

end NUMINAMATH_CALUDE_negation_existence_statement_l1117_111727


namespace NUMINAMATH_CALUDE_no_solution_exists_l1117_111748

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem no_solution_exists : ¬ ∃ n : ℕ, n * sum_of_digits n = 20222022 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l1117_111748


namespace NUMINAMATH_CALUDE_koolaid_percentage_l1117_111768

/-- Calculates the percentage of Kool-Aid powder in a mixture after evaporation and water addition -/
theorem koolaid_percentage
  (initial_powder : ℚ)
  (initial_water : ℚ)
  (evaporation_rate : ℚ)
  (water_increase_factor : ℚ)
  (h1 : initial_powder = 3)
  (h2 : initial_water = 20)
  (h3 : evaporation_rate = 1/4)
  (h4 : water_increase_factor = 5) :
  let remaining_water := initial_water * (1 - evaporation_rate)
  let final_water := remaining_water * water_increase_factor
  let final_mixture := initial_powder + final_water
  initial_powder / final_mixture = 1/26 :=
sorry

end NUMINAMATH_CALUDE_koolaid_percentage_l1117_111768


namespace NUMINAMATH_CALUDE_rice_cost_is_ten_cents_l1117_111713

/-- The cost of rice per plate in cents -/
def rice_cost_per_plate (total_plates : ℕ) (chicken_cost_per_plate : ℚ) (total_spent : ℚ) : ℚ :=
  (total_spent - (chicken_cost_per_plate * total_plates)) / total_plates * 100

/-- Theorem: The cost of rice per plate is 10 cents -/
theorem rice_cost_is_ten_cents :
  rice_cost_per_plate 100 0.40 50 = 10 := by
  sorry

end NUMINAMATH_CALUDE_rice_cost_is_ten_cents_l1117_111713


namespace NUMINAMATH_CALUDE_invitations_per_pack_is_four_l1117_111742

/-- A structure representing the invitation problem --/
structure InvitationProblem where
  total_invitations : ℕ
  num_packs : ℕ
  invitations_per_pack : ℕ
  h1 : total_invitations = num_packs * invitations_per_pack

/-- Theorem stating that given the conditions of the problem, the number of invitations per pack is 4 --/
theorem invitations_per_pack_is_four (problem : InvitationProblem)
  (h2 : problem.total_invitations = 12)
  (h3 : problem.num_packs = 3) :
  problem.invitations_per_pack = 4 := by
  sorry

end NUMINAMATH_CALUDE_invitations_per_pack_is_four_l1117_111742


namespace NUMINAMATH_CALUDE_floor_sum_product_l1117_111714

theorem floor_sum_product : 3 * (⌊(12.7 : ℝ)⌋ + ⌊(-12.7 : ℝ)⌋) = -3 := by sorry

end NUMINAMATH_CALUDE_floor_sum_product_l1117_111714


namespace NUMINAMATH_CALUDE_water_amount_in_sport_formulation_l1117_111763

/-- Represents the ratio of ingredients in a drink formulation -/
structure DrinkRatio where
  flavoring : ℚ
  corn_syrup : ℚ
  water : ℚ

/-- The standard formulation ratio -/
def standard_ratio : DrinkRatio := ⟨1, 12, 30⟩

/-- The sport formulation ratio -/
def sport_ratio : DrinkRatio := 
  let f := standard_ratio.flavoring
  let c := standard_ratio.corn_syrup
  let w := standard_ratio.water
  ⟨f, f * 4, f * 15⟩

/-- Calculates the amount of water given the amount of corn syrup -/
def water_amount (corn_syrup_amount : ℚ) : ℚ :=
  (corn_syrup_amount * sport_ratio.water) / sport_ratio.corn_syrup

/-- Theorem: The amount of water in the sport formulation is 7.5 ounces when there are 2 ounces of corn syrup -/
theorem water_amount_in_sport_formulation :
  water_amount 2 = 7.5 := by sorry

end NUMINAMATH_CALUDE_water_amount_in_sport_formulation_l1117_111763


namespace NUMINAMATH_CALUDE_largest_band_size_l1117_111732

theorem largest_band_size : ∃ (m r x : ℕ),
  m < 150 ∧
  r * x + 3 = m ∧
  (r - 3) * (x + 2) = m ∧
  ∀ (m' r' x' : ℕ),
    m' < 150 →
    r' * x' + 3 = m' →
    (r' - 3) * (x' + 2) = m' →
    m' ≤ m ∧
  m = 107 := by
sorry

end NUMINAMATH_CALUDE_largest_band_size_l1117_111732


namespace NUMINAMATH_CALUDE_fifth_day_temperature_l1117_111769

/-- Given the average temperatures and ratio of temperatures for specific days,
    prove that the temperature on the fifth day is 32 degrees. -/
theorem fifth_day_temperature
  (avg_first_four : ℝ)
  (avg_second_to_fifth : ℝ)
  (temp_first : ℝ)
  (temp_fifth : ℝ)
  (h1 : avg_first_four = 58)
  (h2 : avg_second_to_fifth = 59)
  (h3 : temp_fifth = (8 / 7) * temp_first)
  (h4 : temp_first + (avg_first_four * 4 - temp_first) = avg_first_four * 4)
  (h5 : (avg_first_four * 4 - temp_first) + temp_fifth = avg_second_to_fifth * 4) :
  temp_fifth = 32 :=
sorry

end NUMINAMATH_CALUDE_fifth_day_temperature_l1117_111769


namespace NUMINAMATH_CALUDE_smallest_lcm_with_gcd_11_l1117_111725

theorem smallest_lcm_with_gcd_11 :
  ∃ (k l : ℕ), 
    1000 ≤ k ∧ k < 10000 ∧
    1000 ≤ l ∧ l < 10000 ∧
    Nat.gcd k l = 11 ∧
    Nat.lcm k l = 92092 ∧
    ∀ (m n : ℕ), 
      1000 ≤ m ∧ m < 10000 ∧
      1000 ≤ n ∧ n < 10000 ∧
      Nat.gcd m n = 11 →
      Nat.lcm m n ≥ 92092 :=
by sorry

end NUMINAMATH_CALUDE_smallest_lcm_with_gcd_11_l1117_111725


namespace NUMINAMATH_CALUDE_other_divisor_problem_l1117_111772

theorem other_divisor_problem (n : ℕ) (h1 : n = 266) (h2 : n % 33 = 2) : 
  ∃ (x : ℕ), x ≠ 33 ∧ n % x = 2 ∧ x = 132 ∧ ∀ y : ℕ, y ≠ 33 → n % y = 2 → y ≤ x :=
by sorry

end NUMINAMATH_CALUDE_other_divisor_problem_l1117_111772


namespace NUMINAMATH_CALUDE_cycle_selling_price_l1117_111739

/-- Calculates the selling price of an item given its cost price and gain percentage. -/
def selling_price (cost : ℕ) (gain_percent : ℕ) : ℕ :=
  cost + (cost * gain_percent) / 100

/-- Theorem: If a cycle is bought for Rs. 1000 and sold with a 100% gain, the selling price is Rs. 2000. -/
theorem cycle_selling_price :
  selling_price 1000 100 = 2000 := by
  sorry

#eval selling_price 1000 100

end NUMINAMATH_CALUDE_cycle_selling_price_l1117_111739


namespace NUMINAMATH_CALUDE_fraction_ordering_l1117_111712

theorem fraction_ordering : 
  let a := (3 : ℚ) / 8
  let b := (5 : ℚ) / 14
  let c := (6 : ℚ) / 17
  let d := b - (1 : ℚ) / 56
  d < c ∧ c < a :=
by sorry

end NUMINAMATH_CALUDE_fraction_ordering_l1117_111712


namespace NUMINAMATH_CALUDE_parabola_properties_l1117_111708

/-- A parabola with equation y = x² - 2mx + m² - 9 where m is a constant -/
def parabola (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*m*x + m^2 - 9

/-- The x-coordinates of the intersection points of the parabola with the x-axis -/
def roots (m : ℝ) : Set ℝ := {x : ℝ | parabola m x = 0}

/-- The y-coordinate of a point on the parabola given its x-coordinate -/
def y_coord (m : ℝ) (x : ℝ) : ℝ := parabola m x

theorem parabola_properties (m : ℝ) :
  (∃ (A B : ℝ), A ∈ roots m ∧ B ∈ roots m ∧ A ≠ B) →
  (∀ x, y_coord m x ≥ -9) ∧
  (∃ (A B : ℝ), A ∈ roots m ∧ B ∈ roots m ∧ |A - B| = 6) ∧
  (∀ x₁ x₂, x₁ < x₂ ∧ x₂ < m - 1 → y_coord m x₁ > y_coord m x₂) ∧
  (y_coord m (m + 1) < y_coord m (m - 3)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l1117_111708


namespace NUMINAMATH_CALUDE_derivative_at_one_l1117_111798

open Real

theorem derivative_at_one (f : ℝ → ℝ) (f' : ℝ → ℝ) :
  (∀ x, deriv f x = f' x) →
  (∀ x, f x = 2 * x * f' 1 + log x) →
  f' 1 = -1 := by
sorry

end NUMINAMATH_CALUDE_derivative_at_one_l1117_111798


namespace NUMINAMATH_CALUDE_geometric_sequence_12th_term_l1117_111718

/-- A geometric sequence is defined by its first term and common ratio -/
def GeometricSequence (a₁ : ℝ) (r : ℝ) := fun n : ℕ => a₁ * r ^ (n - 1)

/-- The nth term of a geometric sequence -/
def nthTerm (seq : ℕ → ℝ) (n : ℕ) : ℝ := seq n

theorem geometric_sequence_12th_term
  (seq : ℕ → ℝ)
  (h_geometric : ∃ a₁ r, seq = GeometricSequence a₁ r)
  (h_4th : nthTerm seq 4 = 4)
  (h_7th : nthTerm seq 7 = 32) :
  nthTerm seq 12 = 1024 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_12th_term_l1117_111718


namespace NUMINAMATH_CALUDE_water_for_lemonade_l1117_111757

/-- Represents the ratio of water to lemon juice in the lemonade mixture -/
def water_to_juice_ratio : ℚ := 7 / 1

/-- Represents the number of quarts in a gallon -/
def quarts_per_gallon : ℚ := 4

/-- Calculates the amount of water needed to make one gallon of lemonade -/
def water_needed (ratio : ℚ) (quarts_in_gallon : ℚ) : ℚ :=
  (ratio * quarts_in_gallon) / (ratio + 1)

/-- Theorem stating that the amount of water needed for one gallon of lemonade is 7/2 quarts -/
theorem water_for_lemonade :
  water_needed water_to_juice_ratio quarts_per_gallon = 7 / 2 := by
  sorry

end NUMINAMATH_CALUDE_water_for_lemonade_l1117_111757


namespace NUMINAMATH_CALUDE_notebook_packages_l1117_111723

theorem notebook_packages (L : ℕ) : L > 4 →
  (∃ a b : ℕ, a > 0 ∧ a * L + 4 * b = 69) →
  L = 23 := by sorry

end NUMINAMATH_CALUDE_notebook_packages_l1117_111723


namespace NUMINAMATH_CALUDE_six_digit_pin_probability_six_digit_pin_probability_value_l1117_111797

/-- The probability of randomly selecting a 6-digit PIN with a non-zero first digit, 
    such that the first two digits are both 6 -/
theorem six_digit_pin_probability : ℝ :=
  let total_pins := 9 * 10^5  -- 9 choices for first digit, 10 choices each for other 5 digits
  let favorable_pins := 10^4  -- 4 digits can be any number from 0 to 9
  favorable_pins / total_pins

/-- The probability is equal to 1/90 -/
theorem six_digit_pin_probability_value : six_digit_pin_probability = 1 / 90 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_pin_probability_six_digit_pin_probability_value_l1117_111797


namespace NUMINAMATH_CALUDE_negative_three_plus_four_equals_one_l1117_111750

theorem negative_three_plus_four_equals_one : -3 + 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_negative_three_plus_four_equals_one_l1117_111750


namespace NUMINAMATH_CALUDE_purchase_equivalence_l1117_111726

/-- Proves that if a person can buy exactly 6 items at price x and exactly 8 items at price (x - 1.5),
    then the total amount of money the person has is 36. -/
theorem purchase_equivalence (x : ℝ) :
  (6 * x = 8 * (x - 1.5)) → 6 * x = 36 := by
  sorry

end NUMINAMATH_CALUDE_purchase_equivalence_l1117_111726


namespace NUMINAMATH_CALUDE_fourth_vertex_of_rectangle_l1117_111721

/-- A rectangle in a 2D plane --/
structure Rectangle where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ
  v4 : ℝ × ℝ

/-- Predicate to check if four points form a rectangle --/
def is_rectangle (r : Rectangle) : Prop :=
  let (x1, y1) := r.v1
  let (x2, y2) := r.v2
  let (x3, y3) := r.v3
  let (x4, y4) := r.v4
  ((x1 = x3 ∧ x2 = x4) ∨ (x1 = x2 ∧ x3 = x4)) ∧
  ((y1 = y2 ∧ y3 = y4) ∨ (y1 = y4 ∧ y2 = y3))

/-- Theorem stating that given three vertices of a rectangle, the fourth vertex is determined --/
theorem fourth_vertex_of_rectangle (x1 y1 x2 y2 x3 y3 : ℝ) 
  (h1 : x1 = 2 ∧ y1 = 1)
  (h2 : x2 = 4 ∧ y2 = 1)
  (h3 : x3 = 2 ∧ y3 = 5) :
  ∃ (r : Rectangle), is_rectangle r ∧ 
    r.v1 = (x1, y1) ∧ r.v2 = (x2, y2) ∧ r.v3 = (x3, y3) ∧ r.v4 = (4, 5) := by
  sorry

#check fourth_vertex_of_rectangle

end NUMINAMATH_CALUDE_fourth_vertex_of_rectangle_l1117_111721


namespace NUMINAMATH_CALUDE_cave_door_weight_theorem_l1117_111709

/-- The weight already on the switch, in pounds. -/
def weight_on_switch : ℕ := 234

/-- The total weight needed, in pounds. -/
def total_weight_needed : ℕ := 712

/-- The additional weight needed to open the cave doors, in pounds. -/
def additional_weight_needed : ℕ := total_weight_needed - weight_on_switch

theorem cave_door_weight_theorem : additional_weight_needed = 478 := by
  sorry

end NUMINAMATH_CALUDE_cave_door_weight_theorem_l1117_111709


namespace NUMINAMATH_CALUDE_minimum_apples_l1117_111746

theorem minimum_apples (n : ℕ) (total_apples : ℕ) : 
  (∃ k : ℕ, total_apples = 25 * k + 24) →   -- Condition 1 and 2
  total_apples > 300 →                      -- Condition 3
  total_apples ≥ 324 :=                     -- Minimum number of apples
by
  sorry

#check minimum_apples

end NUMINAMATH_CALUDE_minimum_apples_l1117_111746


namespace NUMINAMATH_CALUDE_teacher_assignment_count_l1117_111765

/-- The number of ways to assign teachers to classes -/
def assign_teachers (n : ℕ) (m : ℕ) : ℕ :=
  -- The actual implementation is not provided here
  sorry

/-- The number of intern teachers -/
def num_teachers : ℕ := 4

/-- The number of classes -/
def num_classes : ℕ := 3

/-- Theorem stating that the number of ways to assign 4 teachers to 3 classes,
    with each class having at least 1 teacher, is 36 -/
theorem teacher_assignment_count :
  assign_teachers num_teachers num_classes = 36 :=
sorry

end NUMINAMATH_CALUDE_teacher_assignment_count_l1117_111765


namespace NUMINAMATH_CALUDE_initial_breads_count_l1117_111735

/-- The number of thieves -/
def num_thieves : ℕ := 5

/-- The number of breads remaining after all thieves -/
def remaining_breads : ℕ := 3

/-- Function to calculate the number of breads after a thief takes their share -/
def breads_after_thief (x : ℚ) : ℚ := x / 2 - 1 / 2

/-- Function to calculate the number of breads after n thieves -/
def breads_after_n_thieves : ℕ → ℚ → ℚ
  | 0, x => x
  | n + 1, x => breads_after_n_thieves n (breads_after_thief x)

/-- Theorem stating that the initial number of breads was 127 -/
theorem initial_breads_count : 
  breads_after_n_thieves num_thieves 127 = remaining_breads := by sorry

end NUMINAMATH_CALUDE_initial_breads_count_l1117_111735


namespace NUMINAMATH_CALUDE_point_on_axis_l1117_111741

-- Define a point P in 2D space
def P (m : ℝ) : ℝ × ℝ := (m, 2 - m)

-- Define what it means for a point to lie on the coordinate axis
def lies_on_coordinate_axis (p : ℝ × ℝ) : Prop :=
  p.1 = 0 ∨ p.2 = 0

-- Theorem statement
theorem point_on_axis (m : ℝ) : 
  lies_on_coordinate_axis (P m) → m = 0 ∨ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_point_on_axis_l1117_111741


namespace NUMINAMATH_CALUDE_no_integer_solution_l1117_111790

theorem no_integer_solution : ¬∃ (a b : ℕ+), 
  (Real.sqrt a.val + Real.sqrt b.val = 10) ∧ 
  (Real.sqrt a.val * Real.sqrt b.val = 18) := by
sorry

end NUMINAMATH_CALUDE_no_integer_solution_l1117_111790


namespace NUMINAMATH_CALUDE_smallest_factor_difference_l1117_111720

theorem smallest_factor_difference (n : ℕ) (hn : n = 2310) :
  ∃ (a b : ℕ), a * b = n ∧ 
    (∀ (x y : ℕ), x * y = n → x ≤ y → y - x ≥ (b - a)) ∧
    b - a = 13 :=
  sorry

end NUMINAMATH_CALUDE_smallest_factor_difference_l1117_111720


namespace NUMINAMATH_CALUDE_angle_trisection_l1117_111730

theorem angle_trisection (α : ℝ) (h : α = 54) :
  ∃ β : ℝ, β * 3 = α ∧ β = 18 := by
  sorry

end NUMINAMATH_CALUDE_angle_trisection_l1117_111730


namespace NUMINAMATH_CALUDE_consecutive_primes_in_sequence_l1117_111762

theorem consecutive_primes_in_sequence (a b : ℕ) (h1 : a > b) (h2 : b > 1) :
  ∃ n : ℕ, n ≥ 2 → 
    ¬(Nat.Prime ((a^n - 1) / (b^n - 1)) ∧ Nat.Prime ((a^(n+1) - 1) / (b^(n+1) - 1))) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_primes_in_sequence_l1117_111762


namespace NUMINAMATH_CALUDE_arithmetic_sequence_difference_l1117_111704

/-- An arithmetic sequence with given first four terms -/
def arithmetic_sequence (x y : ℝ) : ℕ → ℝ
  | 0 => x
  | 1 => y
  | 2 => 3*x + y
  | 3 => x + 2*y + 2
  | n + 4 => arithmetic_sequence x y 3 + (n + 1) * (arithmetic_sequence x y 1 - arithmetic_sequence x y 0)

/-- The theorem stating that y - x = 2 for the given arithmetic sequence -/
theorem arithmetic_sequence_difference (x y : ℝ) :
  let a := arithmetic_sequence x y
  (∀ n : ℕ, a (n + 1) - a n = a 1 - a 0) →
  y - x = 2 := by
  sorry

#check arithmetic_sequence_difference

end NUMINAMATH_CALUDE_arithmetic_sequence_difference_l1117_111704


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1117_111778

variable (a : ℝ)

theorem polynomial_simplification :
  ((-a^3)^2 * a^3 - 4*a^2 * a^7 = -3*a^9) ∧
  ((2*a + 1) * (-2*a + 1) = 4*a^2 - 1) := by
sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1117_111778


namespace NUMINAMATH_CALUDE_alice_baked_five_more_l1117_111729

/-- The number of additional chocolate chip cookies Alice baked after the accident -/
def additional_cookies (alice_initial bob_initial thrown_away bob_additional final_count : ℕ) : ℕ :=
  final_count - (alice_initial + bob_initial - thrown_away + bob_additional)

/-- Theorem stating that Alice baked 5 more chocolate chip cookies after the accident -/
theorem alice_baked_five_more : additional_cookies 74 7 29 36 93 = 5 := by
  sorry

end NUMINAMATH_CALUDE_alice_baked_five_more_l1117_111729


namespace NUMINAMATH_CALUDE_solve_for_a_l1117_111700

theorem solve_for_a : 
  ∀ a : ℝ, (∃ x : ℝ, x = 3 ∧ a * x - 5 = x + 1) → a = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l1117_111700


namespace NUMINAMATH_CALUDE_largest_divisor_of_m_l1117_111764

theorem largest_divisor_of_m (m : ℕ) (h1 : m > 0) (h2 : 72 ∣ m^3) : 
  ∀ k : ℕ, k ∣ m → k ≤ 6 ∧ 6 ∣ m :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_m_l1117_111764


namespace NUMINAMATH_CALUDE_simplify_expression_l1117_111787

theorem simplify_expression (x y : ℝ) (h : x ≠ 0) :
  y * (x⁻¹ - 2) = (y * (1 - 2*x)) / x := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l1117_111787


namespace NUMINAMATH_CALUDE_geometric_sequence_a6_l1117_111758

/-- Given a geometric sequence {aₙ} where a₁ = 1 and a₄ = 8, prove that a₆ = 32 -/
theorem geometric_sequence_a6 (a : ℕ → ℝ) :
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r) →  -- geometric sequence condition
  a 1 = 1 →                                  -- a₁ = 1
  a 4 = 8 →                                  -- a₄ = 8
  a 6 = 32 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_a6_l1117_111758


namespace NUMINAMATH_CALUDE_product_evaluation_l1117_111780

theorem product_evaluation (a : ℕ) (h : a = 7) : 
  (a - 5) * (a - 4) * (a - 3) * (a - 2) * (a - 1) * a = 5040 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l1117_111780


namespace NUMINAMATH_CALUDE_apples_left_l1117_111781

/-- Given that Mike picked 7.0 apples, Nancy ate 3.0 apples, and Keith picked 6.0 apples,
    prove that the number of apples left is 10.0. -/
theorem apples_left (mike_picked : ℝ) (nancy_ate : ℝ) (keith_picked : ℝ)
    (h1 : mike_picked = 7.0)
    (h2 : nancy_ate = 3.0)
    (h3 : keith_picked = 6.0) :
    mike_picked + keith_picked - nancy_ate = 10.0 := by
  sorry

end NUMINAMATH_CALUDE_apples_left_l1117_111781


namespace NUMINAMATH_CALUDE_min_value_of_expression_min_value_attained_l1117_111734

theorem min_value_of_expression (x : ℝ) : 
  (15 - x) * (13 - x) * (15 + x) * (13 + x) ≥ -961 :=
by sorry

theorem min_value_attained (x : ℝ) : 
  (15 - x) * (13 - x) * (15 + x) * (13 + x) = -961 ↔ x = Real.sqrt 197 ∨ x = -Real.sqrt 197 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_min_value_attained_l1117_111734


namespace NUMINAMATH_CALUDE_x_over_y_is_negative_one_l1117_111740

theorem x_over_y_is_negative_one (x y : ℝ) 
  (h1 : 3 < (x - y) / (x + y)) 
  (h2 : (x - y) / (x + y) < 8) 
  (h3 : ∃ (n : ℤ), x / y = n) : 
  x / y = -1 := by sorry

end NUMINAMATH_CALUDE_x_over_y_is_negative_one_l1117_111740


namespace NUMINAMATH_CALUDE_find_number_l1117_111759

theorem find_number : ∃ (x : ℝ), 5 + x * (8 - 3) = 15 ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l1117_111759


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l1117_111786

theorem consecutive_integers_sum (x : ℕ) (h1 : x > 0) (h2 : x * (x + 1) = 812) : x + (x + 1) = 57 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l1117_111786


namespace NUMINAMATH_CALUDE_repeating_decimal_proof_l1117_111783

theorem repeating_decimal_proof : ∃ (n : ℕ), n ≥ 10 ∧ n < 100 ∧ 
  (48 * (n / 99 : ℚ) - 48 * (n / 100 : ℚ) = 1 / 5) ∧
  (100 * (n / 99 : ℚ) - (n / 99 : ℚ) = n) :=
by sorry

end NUMINAMATH_CALUDE_repeating_decimal_proof_l1117_111783


namespace NUMINAMATH_CALUDE_largest_inscribed_circle_radius_l1117_111774

-- Define the quadrilateral ABCD
def Quadrilateral (A B C D : ℝ × ℝ) : Prop :=
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let CD := Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2)
  let DA := Real.sqrt ((D.1 - A.1)^2 + (D.2 - A.2)^2)
  AB = 13 ∧ BC = 10 ∧ CD = 8 ∧ DA = 11

-- Define the inscribed circle
def InscribedCircle (A B C D : ℝ × ℝ) (O : ℝ × ℝ) (r : ℝ) : Prop :=
  Quadrilateral A B C D ∧
  ∀ P : ℝ × ℝ, (P ∈ Set.range (fun t => (t * B.1 + (1 - t) * A.1, t * B.2 + (1 - t) * A.2)) ∨
                 P ∈ Set.range (fun t => (t * C.1 + (1 - t) * B.1, t * C.2 + (1 - t) * B.2)) ∨
                 P ∈ Set.range (fun t => (t * D.1 + (1 - t) * C.1, t * D.2 + (1 - t) * C.2)) ∨
                 P ∈ Set.range (fun t => (t * A.1 + (1 - t) * D.1, t * A.2 + (1 - t) * D.2))) →
                Real.sqrt ((P.1 - O.1)^2 + (P.2 - O.2)^2) ≥ r

-- Theorem statement
theorem largest_inscribed_circle_radius :
  ∀ A B C D O : ℝ × ℝ,
  ∀ r : ℝ,
  InscribedCircle A B C D O r →
  r ≤ 2 * Real.sqrt 6 :=
sorry

end NUMINAMATH_CALUDE_largest_inscribed_circle_radius_l1117_111774


namespace NUMINAMATH_CALUDE_regular_polygon_area_l1117_111767

theorem regular_polygon_area (n : ℕ) (R : ℝ) : 
  n > 2 → 
  R > 0 → 
  (1 / 2 : ℝ) * n * R^2 * Real.sin ((2 * Real.pi) / n) = 2 * R^2 → 
  n = 12 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_area_l1117_111767


namespace NUMINAMATH_CALUDE_school_fair_revenue_l1117_111719

/-- Calculates the total revenue from sales at a school fair -/
theorem school_fair_revenue (chips_sold : ℕ) (chips_price : ℚ)
  (hot_dogs_sold : ℕ) (hot_dogs_price : ℚ)
  (drinks_sold : ℕ) (drinks_price : ℚ) :
  chips_sold = 27 →
  chips_price = 3/2 →
  hot_dogs_sold = chips_sold - 8 →
  hot_dogs_price = 3 →
  drinks_sold = hot_dogs_sold + 12 →
  drinks_price = 2 →
  chips_sold * chips_price + hot_dogs_sold * hot_dogs_price + drinks_sold * drinks_price = 159.5 := by
  sorry

#eval (27 : ℕ) * (3/2 : ℚ) + (27 - 8 : ℕ) * (3 : ℚ) + ((27 - 8 : ℕ) + 12) * (2 : ℚ)

end NUMINAMATH_CALUDE_school_fair_revenue_l1117_111719


namespace NUMINAMATH_CALUDE_solution_of_equation_l1117_111749

theorem solution_of_equation :
  ∃! y : ℚ, 6 * (4 * y - 1) - 3 = 3 * (2 - 5 * y) ∧ y = 5 / 13 := by
  sorry

end NUMINAMATH_CALUDE_solution_of_equation_l1117_111749


namespace NUMINAMATH_CALUDE_zero_point_not_implies_product_negative_l1117_111728

-- Define a continuous function on a closed interval
def ContinuousOnInterval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ContinuousOn f (Set.Icc a b)

-- Define the existence of a zero point in an open interval
def HasZeroInOpenInterval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x, a < x ∧ x < b ∧ f x = 0

-- Theorem statement
theorem zero_point_not_implies_product_negative
  (f : ℝ → ℝ) (a b : ℝ) (h1 : a < b) (h2 : ContinuousOnInterval f a b) :
  HasZeroInOpenInterval f a b → (f a) * (f b) < 0 → False :=
sorry

end NUMINAMATH_CALUDE_zero_point_not_implies_product_negative_l1117_111728


namespace NUMINAMATH_CALUDE_caleb_spent_correct_amount_l1117_111716

-- Define the given conditions
def total_burgers : ℕ := 50
def single_burger_cost : ℚ := 1
def double_burger_cost : ℚ := 1.5
def double_burgers_bought : ℕ := 37

-- Define the function to calculate the total cost
def total_cost : ℚ :=
  (double_burgers_bought * double_burger_cost) +
  ((total_burgers - double_burgers_bought) * single_burger_cost)

-- Theorem to prove
theorem caleb_spent_correct_amount :
  total_cost = 68.5 := by sorry

end NUMINAMATH_CALUDE_caleb_spent_correct_amount_l1117_111716


namespace NUMINAMATH_CALUDE_constant_variance_properties_l1117_111785

/-- A sequence is constant variance if the sequence of its squares is arithmetic -/
def ConstantVariance (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 2)^2 - a (n + 1)^2 = a (n + 1)^2 - a n^2

/-- A sequence is constant if all its terms are equal -/
def ConstantSequence (a : ℕ → ℝ) : Prop :=
  ∃ c : ℝ, ∀ n : ℕ, a n = c

/-- A sequence is arithmetic if the difference between consecutive terms is constant -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem constant_variance_properties (a : ℕ → ℝ) :
  (ConstantSequence a → ConstantVariance a) ∧
  (ConstantVariance a → ArithmeticSequence (λ n => (a n)^2)) ∧
  (ConstantVariance a → ConstantVariance (λ n => a (2*n))) :=
sorry

end NUMINAMATH_CALUDE_constant_variance_properties_l1117_111785


namespace NUMINAMATH_CALUDE_coins_in_pockets_l1117_111707

/-- The number of ways to place n identical objects into k distinct containers -/
def stars_and_bars (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The problem of placing 5 identical coins into 3 different pockets -/
theorem coins_in_pockets : stars_and_bars 5 3 = 21 := by sorry

end NUMINAMATH_CALUDE_coins_in_pockets_l1117_111707


namespace NUMINAMATH_CALUDE_largest_square_area_l1117_111775

theorem largest_square_area (x y z : ℝ) (h1 : x^2 + y^2 = z^2) 
  (h2 : x^2 + y^2 + 2*z^2 = 722) : z^2 = 722/3 := by
  sorry

end NUMINAMATH_CALUDE_largest_square_area_l1117_111775


namespace NUMINAMATH_CALUDE_sum_of_sqrt_greater_than_one_l1117_111736

theorem sum_of_sqrt_greater_than_one 
  (x y z t : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (ht : t > 0)
  (hxy : x ≠ y) (hxz : x ≠ z) (hxt : x ≠ t)
  (hyz : y ≠ z) (hyt : y ≠ t) (hzt : z ≠ t)
  (hsum : x + y + z + t = 1) :
  (Real.sqrt x + Real.sqrt y > 1) ∨
  (Real.sqrt x + Real.sqrt z > 1) ∨
  (Real.sqrt x + t > 1) ∨
  (Real.sqrt y + Real.sqrt z > 1) ∨
  (Real.sqrt y + Real.sqrt t > 1) ∨
  (Real.sqrt z + Real.sqrt t > 1) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_sqrt_greater_than_one_l1117_111736


namespace NUMINAMATH_CALUDE_equation_solution_l1117_111745

theorem equation_solution : ∃ x : ℝ, 9 - 3 / (1/3) + x = 3 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1117_111745


namespace NUMINAMATH_CALUDE_probability_of_sum_22_l1117_111751

/-- A function representing the probability of rolling a specific sum with four standard 6-faced dice -/
def probability_of_sum (sum : ℕ) : ℚ :=
  sorry

/-- The number of faces on each die -/
def num_faces : ℕ := 6

/-- The number of dice rolled -/
def num_dice : ℕ := 4

/-- Theorem stating the probability of rolling a sum of 22 with four standard 6-faced dice -/
theorem probability_of_sum_22 : probability_of_sum 22 = 5 / 648 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_sum_22_l1117_111751


namespace NUMINAMATH_CALUDE_cubic_inequality_l1117_111796

theorem cubic_inequality (x : ℝ) (h : x ≥ 0) : 3 * x^3 - 6 * x^2 + 4 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_l1117_111796


namespace NUMINAMATH_CALUDE_orchestra_members_count_l1117_111756

theorem orchestra_members_count :
  ∃! n : ℕ, 200 < n ∧ n < 300 ∧
    n % 4 = 2 ∧
    n % 5 = 3 ∧
    n % 7 = 4 ∧
    n = 158 :=
by sorry

end NUMINAMATH_CALUDE_orchestra_members_count_l1117_111756


namespace NUMINAMATH_CALUDE_income_distribution_l1117_111776

theorem income_distribution (total_income : ℝ) 
  (h_total : total_income = 100) 
  (food_percent : ℝ) (h_food : food_percent = 35)
  (education_percent : ℝ) (h_education : education_percent = 25)
  (rent_percent : ℝ) (h_rent : rent_percent = 80) : 
  (total_income - (food_percent + education_percent) * total_income / 100 - 
   rent_percent * (total_income - (food_percent + education_percent) * total_income / 100) / 100) / 
  total_income * 100 = 8 := by
  sorry

end NUMINAMATH_CALUDE_income_distribution_l1117_111776


namespace NUMINAMATH_CALUDE_smallest_n_proof_l1117_111788

/-- The number of boxes -/
def num_boxes : ℕ := 2010

/-- The probability of stopping after drawing exactly n marbles -/
def P (n : ℕ) : ℚ := 1 / (n * (n + 1))

/-- The threshold probability -/
def threshold : ℚ := 1 / num_boxes

/-- The smallest n for which P(n) < 1/2010 -/
def smallest_n : ℕ := 45

theorem smallest_n_proof :
  (∀ k < smallest_n, P k ≥ threshold) ∧
  P smallest_n < threshold :=
sorry

#check smallest_n_proof

end NUMINAMATH_CALUDE_smallest_n_proof_l1117_111788


namespace NUMINAMATH_CALUDE_triangle_construction_theorem_l1117_111731

/-- A line in 2D space -/
structure Line where
  -- Define a line using two points
  p1 : ℝ × ℝ
  p2 : ℝ × ℝ

/-- A triangle in 2D space -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Check if a point lies on a line -/
def point_on_line (p : ℝ × ℝ) (l : Line) : Prop := sorry

/-- Check if a line is a perpendicular bisector of a triangle side -/
def is_perp_bisector (l : Line) (t : Triangle) : Prop := sorry

/-- The main theorem -/
theorem triangle_construction_theorem 
  (e f g : Line) -- Three given lines (perpendicular bisectors)
  (P : ℝ × ℝ)    -- Given point
  (h : point_on_line P e ∨ point_on_line P f ∨ point_on_line P g) -- P is on one of the lines
  : ∃ (t : Triangle), 
    (point_on_line P e ∧ is_perp_bisector e t) ∨ 
    (point_on_line P f ∧ is_perp_bisector f t) ∨ 
    (point_on_line P g ∧ is_perp_bisector g t) :=
sorry

end NUMINAMATH_CALUDE_triangle_construction_theorem_l1117_111731


namespace NUMINAMATH_CALUDE_nut_mixture_price_l1117_111752

/-- Calculates the total selling price of a nut mixture -/
def total_selling_price (total_weight : ℝ) (cashew_weight : ℝ) (cashew_price : ℝ) (peanut_price : ℝ) : ℝ :=
  cashew_weight * cashew_price + (total_weight - cashew_weight) * peanut_price

/-- The total selling price of the nut mixture is $83.00 -/
theorem nut_mixture_price : total_selling_price 25 11 5 2 = 83 := by
  sorry

end NUMINAMATH_CALUDE_nut_mixture_price_l1117_111752


namespace NUMINAMATH_CALUDE_morning_shells_count_l1117_111799

/-- The number of shells Lino picked up in the afternoon -/
def afternoon_shells : ℕ := 324

/-- The total number of shells Lino picked up -/
def total_shells : ℕ := 616

/-- The number of shells Lino picked up in the morning -/
def morning_shells : ℕ := total_shells - afternoon_shells

theorem morning_shells_count : morning_shells = 292 := by
  sorry

end NUMINAMATH_CALUDE_morning_shells_count_l1117_111799


namespace NUMINAMATH_CALUDE_inequality_property_l1117_111794

theorem inequality_property (a b c d : ℝ) : a > b → c > d → a - d > b - c := by
  sorry

end NUMINAMATH_CALUDE_inequality_property_l1117_111794


namespace NUMINAMATH_CALUDE_no_real_solution_for_A_only_l1117_111738

theorem no_real_solution_for_A_only : 
  (¬ ∃ x : ℝ, (x - 3)^2 = -1) ∧ 
  (∃ x : ℝ, |x/2| - 6 = 0) ∧ 
  (∃ x : ℝ, x^2 + 8*x + 16 = 0) ∧ 
  (∃ x : ℝ, x + Real.sqrt (x - 5) = 0) ∧ 
  (∃ x : ℝ, Real.sqrt (-2*x - 10) = 3) :=
by sorry

end NUMINAMATH_CALUDE_no_real_solution_for_A_only_l1117_111738


namespace NUMINAMATH_CALUDE_all_subtracting_not_purple_not_all_happy_are_purple_some_happy_cant_subtract_l1117_111753

-- Define the universe of snakes
variable (Snake : Type)

-- Define properties of snakes
variable (purple happy can_add can_subtract : Snake → Prop)

-- Define the number of snakes
variable (total_snakes : ℕ)
variable (purple_snakes : ℕ)
variable (happy_snakes : ℕ)

-- State the given conditions
variable (h1 : total_snakes = 20)
variable (h2 : purple_snakes = 6)
variable (h3 : happy_snakes = 8)
variable (h4 : ∃ s, happy s ∧ can_add s)
variable (h5 : ∀ s, purple s → ¬can_subtract s)
variable (h6 : ∀ s, ¬can_subtract s → ¬can_add s)

-- State the theorems to be proved
theorem all_subtracting_not_purple : ∀ s, can_subtract s → ¬purple s := by sorry

theorem not_all_happy_are_purple : ¬(∀ s, happy s → purple s) := by sorry

theorem some_happy_cant_subtract : ∃ s, happy s ∧ ¬can_subtract s := by sorry

end NUMINAMATH_CALUDE_all_subtracting_not_purple_not_all_happy_are_purple_some_happy_cant_subtract_l1117_111753


namespace NUMINAMATH_CALUDE_candy_bar_ratio_l1117_111743

/-- Proves the ratio of candy bars given the second time to the first time -/
theorem candy_bar_ratio (initial_bars : ℕ) (initial_given : ℕ) (bought_bars : ℕ) (kept_bars : ℕ) :
  initial_bars = 7 →
  initial_given = 3 →
  bought_bars = 30 →
  kept_bars = 22 →
  ∃ (second_given : ℕ), 
    second_given = initial_bars + bought_bars - kept_bars - initial_given ∧
    second_given = 4 * initial_given :=
by sorry

end NUMINAMATH_CALUDE_candy_bar_ratio_l1117_111743


namespace NUMINAMATH_CALUDE_turtle_theorem_l1117_111777

def turtle_problem (initial : ℕ) : ℕ :=
  let additional := 3 * initial - 2
  let total := initial + additional
  total / 2

theorem turtle_theorem : turtle_problem 9 = 17 := by
  sorry

end NUMINAMATH_CALUDE_turtle_theorem_l1117_111777


namespace NUMINAMATH_CALUDE_dennis_initial_money_l1117_111782

-- Define the sale discount
def sale_discount : ℚ := 25 / 100

-- Define the original price of the shirts
def original_price : ℚ := 125

-- Define the amount Dennis paid
def amount_paid : ℚ := 100 + 50 + 4 * 5

-- Define the change Dennis received
def change_received : ℚ := 3 * 20 + 10 + 2 * 5 + 4

-- Theorem statement
theorem dennis_initial_money :
  let discounted_price := original_price * (1 - sale_discount)
  let initial_money := discounted_price + change_received
  initial_money = 177.75 := by
  sorry

end NUMINAMATH_CALUDE_dennis_initial_money_l1117_111782


namespace NUMINAMATH_CALUDE_forgotten_item_distance_l1117_111706

/-- Calculates the total distance walked when a person forgets an item halfway to school -/
def total_distance_walked (home_to_school : ℕ) : ℕ :=
  let halfway := home_to_school / 2
  halfway + halfway + home_to_school

/-- Proves that the total distance walked is 1500 meters given the conditions -/
theorem forgotten_item_distance :
  total_distance_walked 750 = 1500 := by
  sorry

#eval total_distance_walked 750

end NUMINAMATH_CALUDE_forgotten_item_distance_l1117_111706


namespace NUMINAMATH_CALUDE_absolute_value_implies_inequality_l1117_111755

theorem absolute_value_implies_inequality (x : ℝ) : 
  |x - 1| = 1 - x → x ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_implies_inequality_l1117_111755


namespace NUMINAMATH_CALUDE_james_writing_pages_l1117_111760

/-- James' writing scenario -/
structure WritingScenario where
  pages_per_hour : ℕ
  hours_per_week : ℕ
  people_per_day : ℕ

/-- Calculate pages written daily per person -/
def pages_per_person_daily (scenario : WritingScenario) : ℚ :=
  (scenario.pages_per_hour * scenario.hours_per_week : ℚ) / (7 * scenario.people_per_day)

/-- Theorem: James writes 5 pages daily to each person -/
theorem james_writing_pages (james : WritingScenario) 
  (h1 : james.pages_per_hour = 10)
  (h2 : james.hours_per_week = 7)
  (h3 : james.people_per_day = 2) :
  pages_per_person_daily james = 5 := by
  sorry

end NUMINAMATH_CALUDE_james_writing_pages_l1117_111760


namespace NUMINAMATH_CALUDE_event_arrangements_l1117_111754

def number_of_arrangements (n : ℕ) (k : ℕ) : ℕ :=
  (Nat.choose n k) * k * k

theorem event_arrangements : number_of_arrangements 6 3 = 180 := by
  sorry

end NUMINAMATH_CALUDE_event_arrangements_l1117_111754


namespace NUMINAMATH_CALUDE_range_of_x_l1117_111784

theorem range_of_x (a b x : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (∀ a b, a ≠ 0 → b ≠ 0 → |2*a - b| + |a + b| ≥ |a| * (|x - 1| + |x + 1|)) →
  x ∈ Set.Icc (-3/2) (3/2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_x_l1117_111784


namespace NUMINAMATH_CALUDE_luna_pink_crayons_percentage_l1117_111792

/-- Given information about Mara's and Luna's crayons, prove that 20% of Luna's crayons are pink -/
theorem luna_pink_crayons_percentage
  (mara_total : ℕ)
  (mara_pink_percent : ℚ)
  (luna_total : ℕ)
  (total_pink : ℕ)
  (h1 : mara_total = 40)
  (h2 : mara_pink_percent = 1/10)
  (h3 : luna_total = 50)
  (h4 : total_pink = 14)
  : (luna_total - (mara_pink_percent * mara_total).floor) / luna_total = 1/5 := by
  sorry

#eval (50 : ℚ) / 5  -- Expected output: 10

end NUMINAMATH_CALUDE_luna_pink_crayons_percentage_l1117_111792


namespace NUMINAMATH_CALUDE_number_with_given_division_properties_l1117_111710

theorem number_with_given_division_properties : ∃ n : ℕ, n / 9 = 80 ∧ n % 9 = 4 ∧ n = 724 := by
  sorry

end NUMINAMATH_CALUDE_number_with_given_division_properties_l1117_111710
