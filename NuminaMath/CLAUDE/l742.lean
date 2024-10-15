import Mathlib

namespace NUMINAMATH_CALUDE_min_value_of_expression_l742_74233

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_perp : (2 : ℝ) * b - (b - 3) * a = 0) : 
  ∀ x y : ℝ, 2 * a + 3 * b ≥ 25 := by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l742_74233


namespace NUMINAMATH_CALUDE_consecutive_integers_average_l742_74299

theorem consecutive_integers_average (a b : ℤ) : 
  (a > 0) →
  (b = (a + (a + 1) + (a + 2) + (a + 3) + (a + 4)) / 5) →
  ((b + (b + 1) + (b + 2) + (b + 3) + (b + 4)) / 5 = a + 4) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_average_l742_74299


namespace NUMINAMATH_CALUDE_number_operation_l742_74231

theorem number_operation (x : ℝ) : (x - 5) / 7 = 7 → (x - 24) / 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_number_operation_l742_74231


namespace NUMINAMATH_CALUDE_correct_statements_l742_74266

theorem correct_statements :
  (∀ a : ℝ, ¬(- a < 0) → a ≤ 0) ∧
  (∀ a : ℝ, |-(a^2)| = (-a)^2) ∧
  (∀ a b : ℝ, a ≠ 0 → b ≠ 0 → a / |a| + b / |b| = 0 → a * b / |a * b| = -1) ∧
  (∀ a b : ℝ, |a| = -b → |b| = b → a = b) :=
by sorry

end NUMINAMATH_CALUDE_correct_statements_l742_74266


namespace NUMINAMATH_CALUDE_exists_grid_with_partitions_l742_74265

/-- A cell in the grid --/
structure Cell :=
  (x : Nat) (y : Nat)

/-- A shape in the grid --/
structure Shape :=
  (cells : List Cell)

/-- The grid --/
def Grid := List Cell

/-- Predicate to check if a shape is valid (contains 5 cells) --/
def isValidShape5 (s : Shape) : Prop :=
  s.cells.length = 5

/-- Predicate to check if a shape is valid (contains 4 cells) --/
def isValidShape4 (s : Shape) : Prop :=
  s.cells.length = 4

/-- Predicate to check if shapes are equal (up to rotation and flipping) --/
def areShapesEqual (s1 s2 : Shape) : Prop :=
  sorry  -- Implementation of shape equality check

/-- Theorem stating the existence of a grid with the required properties --/
theorem exists_grid_with_partitions :
  ∃ (g : Grid) (partition1 partition2 : List Shape),
    g.length = 20 ∧
    partition1.length = 4 ∧
    (∀ s ∈ partition1, isValidShape5 s) ∧
    (∀ i j, i < partition1.length → j < partition1.length → i ≠ j →
      areShapesEqual (partition1.get ⟨i, sorry⟩) (partition1.get ⟨j, sorry⟩)) ∧
    partition2.length = 5 ∧
    (∀ s ∈ partition2, isValidShape4 s) ∧
    (∀ i j, i < partition2.length → j < partition2.length → i ≠ j →
      areShapesEqual (partition2.get ⟨i, sorry⟩) (partition2.get ⟨j, sorry⟩)) :=
by
  sorry


end NUMINAMATH_CALUDE_exists_grid_with_partitions_l742_74265


namespace NUMINAMATH_CALUDE_min_quotient_three_digit_number_l742_74249

theorem min_quotient_three_digit_number : 
  ∀ a b c : ℕ, 
  1 ≤ a ∧ a ≤ 9 ∧ 
  1 ≤ b ∧ b ≤ 9 ∧ 
  1 ≤ c ∧ c ≤ 9 ∧ 
  a ≠ b ∧ b ≠ c ∧ a ≠ c → 
  (100 * a + 10 * b + c : ℚ) / (a + b + c) ≥ 10.5 :=
by sorry

end NUMINAMATH_CALUDE_min_quotient_three_digit_number_l742_74249


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l742_74275

theorem arithmetic_calculation : (-3 + 2) * 3 - (-4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l742_74275


namespace NUMINAMATH_CALUDE_smallest_b_value_l742_74290

theorem smallest_b_value (a b : ℕ+) (h1 : a - b = 4) 
  (h2 : Nat.gcd ((a^3 + b^3) / (a + b)) (a * b) = 4) : 
  b ≥ 2 ∧ ∃ (a' b' : ℕ+), b' = 2 ∧ a' - b' = 4 ∧ 
    Nat.gcd ((a'^3 + b'^3) / (a' + b')) (a' * b') = 4 :=
sorry

end NUMINAMATH_CALUDE_smallest_b_value_l742_74290


namespace NUMINAMATH_CALUDE_intersection_sum_l742_74282

def M : Set ℝ := {x | |x - 4| + |x - 1| < 5}
def N (a : ℝ) : Set ℝ := {x | a < x ∧ x < 6}

theorem intersection_sum (a b : ℝ) : 
  M ∩ N a = {2, b} → a + b = 7 := by
  sorry

end NUMINAMATH_CALUDE_intersection_sum_l742_74282


namespace NUMINAMATH_CALUDE_scenario_contradiction_characteristics_l742_74258

/-- Represents a person's reaction to a statement --/
inductive Reaction
  | Cry
  | Laugh

/-- Represents a family member --/
inductive FamilyMember
  | Mother
  | Father

/-- Represents the characteristics of a contradiction --/
structure ContradictionCharacteristics where
  interpenetrating : Bool
  specific : Bool

/-- Given scenario where a child's "I love you" causes different reactions --/
def scenario : FamilyMember → Reaction
  | FamilyMember.Mother => Reaction.Cry
  | FamilyMember.Father => Reaction.Laugh

/-- Theorem stating that the contradiction in the scenario exhibits both 
    interpenetration of contradictory sides and specificity --/
theorem scenario_contradiction_characteristics :
  ∃ (c : ContradictionCharacteristics), 
    c.interpenetrating ∧ c.specific := by
  sorry

end NUMINAMATH_CALUDE_scenario_contradiction_characteristics_l742_74258


namespace NUMINAMATH_CALUDE_quadratic_equation_condition_l742_74225

theorem quadratic_equation_condition (x : ℝ) :
  (x^2 + 2*x - 3 = 0 ↔ (x = -3 ∨ x = 1)) →
  (x = 1 → x^2 + 2*x - 3 = 0) ∧
  ¬(x^2 + 2*x - 3 = 0 → x = 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_condition_l742_74225


namespace NUMINAMATH_CALUDE_six_eight_ten_pythagorean_l742_74209

/-- A function to check if three positive integers form a Pythagorean triple -/
def isPythagoreanTriple (a b c : ℕ+) : Prop :=
  a * a + b * b = c * c

/-- Theorem stating that 6, 8, and 10 form a Pythagorean triple -/
theorem six_eight_ten_pythagorean :
  isPythagoreanTriple 6 8 10 := by
  sorry

#check six_eight_ten_pythagorean

end NUMINAMATH_CALUDE_six_eight_ten_pythagorean_l742_74209


namespace NUMINAMATH_CALUDE_student_count_l742_74252

theorem student_count (W : ℝ) (N : ℕ) (h1 : N > 0) :
  W / N - 12 = (W - 72 + 12) / N → N = 5 := by
sorry

end NUMINAMATH_CALUDE_student_count_l742_74252


namespace NUMINAMATH_CALUDE_nh3_formation_l742_74223

-- Define the chemical reaction
structure Reaction where
  nh4no3 : ℕ
  naoh : ℕ
  nh3 : ℕ

-- Define the stoichiometric relationship
def stoichiometric (r : Reaction) : Prop :=
  r.nh4no3 = r.naoh ∧ r.nh3 = r.nh4no3

-- Theorem statement
theorem nh3_formation (r : Reaction) (h : stoichiometric r) :
  r.nh3 = r.nh4no3 := by
  sorry

end NUMINAMATH_CALUDE_nh3_formation_l742_74223


namespace NUMINAMATH_CALUDE_unique_solution_l742_74298

/-- Represents the number of photos taken by each person -/
structure PhotoCounts where
  C : ℕ  -- Claire
  L : ℕ  -- Lisa
  R : ℕ  -- Robert
  D : ℕ  -- David
  E : ℕ  -- Emma

/-- Checks if the given photo counts satisfy all the conditions -/
def satisfiesConditions (p : PhotoCounts) : Prop :=
  p.L = 3 * p.C ∧
  p.R = p.C + 10 ∧
  p.D = 2 * p.C - 5 ∧
  p.E = 2 * p.R ∧
  p.L + p.R + p.C + p.D + p.E = 350

/-- The unique solution to the photo counting problem -/
def solution : PhotoCounts :=
  { C := 36, L := 108, R := 46, D := 67, E := 93 }

/-- Theorem stating that the solution is unique and satisfies all conditions -/
theorem unique_solution :
  satisfiesConditions solution ∧
  ∀ p : PhotoCounts, satisfiesConditions p → p = solution :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_l742_74298


namespace NUMINAMATH_CALUDE_sum_of_digits_greatest_prime_divisor_18447_l742_74295

def greatest_prime_divisor (n : ℕ) : ℕ := sorry

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem sum_of_digits_greatest_prime_divisor_18447 :
  sum_of_digits (greatest_prime_divisor 18447) = 20 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_greatest_prime_divisor_18447_l742_74295


namespace NUMINAMATH_CALUDE_monday_temp_is_43_l742_74268

/-- Represents the temperatures for each day of the week --/
structure WeekTemperatures where
  monday : ℝ
  tuesday : ℝ
  wednesday : ℝ
  thursday : ℝ
  friday : ℝ

/-- The theorem stating that Monday's temperature is 43 degrees --/
theorem monday_temp_is_43 (w : WeekTemperatures) 
  (avg_mon_to_thu : (w.monday + w.tuesday + w.wednesday + w.thursday) / 4 = 48)
  (avg_tue_to_fri : (w.tuesday + w.wednesday + w.thursday + w.friday) / 4 = 46)
  (one_day_43 : w.monday = 43 ∨ w.tuesday = 43 ∨ w.wednesday = 43 ∨ w.thursday = 43 ∨ w.friday = 43)
  (friday_35 : w.friday = 35) : 
  w.monday = 43 := by
  sorry


end NUMINAMATH_CALUDE_monday_temp_is_43_l742_74268


namespace NUMINAMATH_CALUDE_radio_selling_price_l742_74242

/-- Calculates the selling price of a radio given the purchase price, overhead expenses, and profit percentage. -/
def calculate_selling_price (purchase_price : ℚ) (overhead : ℚ) (profit_percent : ℚ) : ℚ :=
  let total_cost := purchase_price + overhead
  let profit := (profit_percent / 100) * total_cost
  total_cost + profit

/-- Theorem stating that the selling price of the radio is 300 given the specified conditions. -/
theorem radio_selling_price :
  calculate_selling_price 225 28 (18577075098814234 / 1000000000) = 300 := by
  sorry

end NUMINAMATH_CALUDE_radio_selling_price_l742_74242


namespace NUMINAMATH_CALUDE_divisibility_property_l742_74255

theorem divisibility_property (q : ℕ) (h_prime : Nat.Prime q) (h_odd : Odd q) :
  ∃ k : ℤ, (q - 1 : ℤ) ^ (q - 2) + 1 = k * q :=
sorry

end NUMINAMATH_CALUDE_divisibility_property_l742_74255


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_2sqrt14_l742_74222

theorem sqrt_sum_equals_2sqrt14 :
  Real.sqrt (20 - 8 * Real.sqrt 5) + Real.sqrt (20 + 8 * Real.sqrt 5) = 2 * Real.sqrt 14 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_2sqrt14_l742_74222


namespace NUMINAMATH_CALUDE_highest_power_of_three_dividing_M_l742_74279

def M : ℕ := sorry

theorem highest_power_of_three_dividing_M :
  ∃ (j : ℕ), (3^j ∣ M) ∧ ¬(3^(j+1) ∣ M) ∧ j = 1 := by sorry

end NUMINAMATH_CALUDE_highest_power_of_three_dividing_M_l742_74279


namespace NUMINAMATH_CALUDE_max_sum_at_15_l742_74238

/-- An arithmetic sequence with first term 29 and S_10 = S_20 -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  first_term : a 1 = 29
  sum_equal : (Finset.range 10).sum a = (Finset.range 20).sum a

/-- Sum of the first n terms of the arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  (Finset.range n).sum seq.a

/-- The maximum value of S_n occurs when n = 15 -/
theorem max_sum_at_15 (seq : ArithmeticSequence) :
  ∀ n : ℕ, S seq n ≤ S seq 15 :=
sorry

end NUMINAMATH_CALUDE_max_sum_at_15_l742_74238


namespace NUMINAMATH_CALUDE_f_nonnegative_implies_a_range_l742_74273

-- Define the function f
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + 1

-- State the theorem
theorem f_nonnegative_implies_a_range (a b : ℝ) :
  (∀ x ≥ 2, f a b x ≥ 0) → a ∈ Set.Ioo (-9 : ℝ) (-3 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_f_nonnegative_implies_a_range_l742_74273


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l742_74257

theorem quadratic_roots_condition (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ (k - 1) * x^2 - 2 * x + 1 = 0 ∧ (k - 1) * y^2 - 2 * y + 1 = 0) ↔
  (k ≤ 2 ∧ k ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l742_74257


namespace NUMINAMATH_CALUDE_unique_solution_g_equals_g_inv_l742_74241

-- Define the function g
def g (x : ℝ) : ℝ := 4 * x - 9

-- Define the inverse function of g
noncomputable def g_inv (x : ℝ) : ℝ := (x + 9) / 4

-- Theorem statement
theorem unique_solution_g_equals_g_inv :
  ∃! x : ℝ, g x = g_inv x :=
sorry

end NUMINAMATH_CALUDE_unique_solution_g_equals_g_inv_l742_74241


namespace NUMINAMATH_CALUDE_cos_2013pi_l742_74213

theorem cos_2013pi : Real.cos (2013 * Real.pi) = -1 := by
  sorry

end NUMINAMATH_CALUDE_cos_2013pi_l742_74213


namespace NUMINAMATH_CALUDE_claire_pets_l742_74207

theorem claire_pets (total_pets : ℕ) (total_males : ℕ) : 
  total_pets = 92 →
  total_males = 25 →
  ∃ (gerbils hamsters : ℕ),
    gerbils + hamsters = total_pets ∧
    (gerbils / 4 : ℚ) + (hamsters / 3 : ℚ) = total_males ∧
    gerbils = 68 := by
  sorry

end NUMINAMATH_CALUDE_claire_pets_l742_74207


namespace NUMINAMATH_CALUDE_sandwiches_left_for_others_l742_74248

def total_sandwiches : ℕ := 20
def sandwiches_for_coworker : ℕ := 4
def sandwiches_for_self : ℕ := 2 * sandwiches_for_coworker

theorem sandwiches_left_for_others : 
  total_sandwiches - sandwiches_for_coworker - sandwiches_for_self = 8 := by
  sorry

end NUMINAMATH_CALUDE_sandwiches_left_for_others_l742_74248


namespace NUMINAMATH_CALUDE_rice_price_reduction_l742_74274

theorem rice_price_reduction (x : ℝ) (h : x > 0) :
  let original_amount := 30
  let price_reduction_factor := 0.75
  let new_amount := original_amount / price_reduction_factor
  new_amount = 40 := by
sorry

end NUMINAMATH_CALUDE_rice_price_reduction_l742_74274


namespace NUMINAMATH_CALUDE_smallest_prime_dividing_sum_l742_74293

theorem smallest_prime_dividing_sum : 
  ∀ p : Nat, Prime p → p ∣ (2^14 + 7^9) → p ≥ 7 :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_dividing_sum_l742_74293


namespace NUMINAMATH_CALUDE_grinder_loss_percentage_l742_74218

theorem grinder_loss_percentage (grinder_cp mobile_cp total_profit mobile_profit_percent : ℝ)
  (h1 : grinder_cp = 15000)
  (h2 : mobile_cp = 10000)
  (h3 : total_profit = 400)
  (h4 : mobile_profit_percent = 10) :
  let mobile_sp := mobile_cp * (1 + mobile_profit_percent / 100)
  let total_sp := grinder_cp + mobile_cp + total_profit
  let grinder_sp := total_sp - mobile_sp
  let loss_amount := grinder_cp - grinder_sp
  loss_amount / grinder_cp * 100 = 4 := by sorry

end NUMINAMATH_CALUDE_grinder_loss_percentage_l742_74218


namespace NUMINAMATH_CALUDE_f_two_zero_l742_74284

/-- A mapping f that takes a point (x,y) to (x+y, x-y) -/
def f (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + p.2, p.1 - p.2)

/-- The theorem stating that f(2,0) = (2,2) -/
theorem f_two_zero : f (2, 0) = (2, 2) := by
  sorry

end NUMINAMATH_CALUDE_f_two_zero_l742_74284


namespace NUMINAMATH_CALUDE_custom_op_four_six_l742_74288

def custom_op (a b : ℤ) : ℤ := 4*a - 2*b + a*b

theorem custom_op_four_six : custom_op 4 6 = 28 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_four_six_l742_74288


namespace NUMINAMATH_CALUDE_simple_interest_rate_l742_74264

/-- Given a principal amount P and a time period of 10 years,
    prove that the rate of simple interest is 6% per annum
    when the simple interest is 3/5 of the principal amount. -/
theorem simple_interest_rate (P : ℝ) (P_pos : P > 0) :
  let SI := (3/5) * P  -- Simple interest is 3/5 of principal
  let T := 10  -- Time period in years
  let r := 6  -- Rate percent per annum
  SI = (P * r * T) / 100  -- Simple interest formula
  := by sorry

end NUMINAMATH_CALUDE_simple_interest_rate_l742_74264


namespace NUMINAMATH_CALUDE_fraction_simplification_l742_74227

theorem fraction_simplification (a b : ℝ) (h1 : a ≠ b) (h2 : a + b ≠ 0) :
  ((a + b)^2 * (a^3 - b^3)) / ((a^2 - b^2)^2) = (a^2 + a*b + b^2) / (a - b) ∧
  (6*a^2*b^2 - 3*a^3*b - 3*a*b^3) / (a*b^3 - a^3*b) = 3 * (a - b) / (a + b) :=
sorry

end NUMINAMATH_CALUDE_fraction_simplification_l742_74227


namespace NUMINAMATH_CALUDE_function_symmetry_l742_74247

/-- Given a function f(x) = ax³ + bx + c*sin(x) - 2 where f(-2) = 8, prove that f(2) = -12 -/
theorem function_symmetry (a b c : ℝ) :
  let f : ℝ → ℝ := λ x => a * x^3 + b * x + c * Real.sin x - 2
  f (-2) = 8 → f 2 = -12 := by
  sorry

end NUMINAMATH_CALUDE_function_symmetry_l742_74247


namespace NUMINAMATH_CALUDE_inequality_proofs_l742_74286

theorem inequality_proofs :
  (∀ a b : ℝ, a^2 + b^2 + 3 ≥ a*b + Real.sqrt 3 * (a + b)) ∧
  (Real.sqrt 6 + Real.sqrt 7 > 2 * Real.sqrt 2 + Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proofs_l742_74286


namespace NUMINAMATH_CALUDE_oil_leak_total_l742_74216

theorem oil_leak_total (leaked_before_fixing leaked_while_fixing : ℕ) 
  (h1 : leaked_before_fixing = 2475)
  (h2 : leaked_while_fixing = 3731) :
  leaked_before_fixing + leaked_while_fixing = 6206 :=
by sorry

end NUMINAMATH_CALUDE_oil_leak_total_l742_74216


namespace NUMINAMATH_CALUDE_train_length_proof_l742_74280

/-- The length of a train in meters -/
def train_length : ℝ := 1200

/-- The time in seconds it takes for the train to cross a tree -/
def tree_crossing_time : ℝ := 120

/-- The time in seconds it takes for the train to pass a platform -/
def platform_passing_time : ℝ := 150

/-- The length of the platform in meters -/
def platform_length : ℝ := 300

theorem train_length_proof :
  (train_length / tree_crossing_time = (train_length + platform_length) / platform_passing_time) →
  train_length = 1200 := by
sorry

end NUMINAMATH_CALUDE_train_length_proof_l742_74280


namespace NUMINAMATH_CALUDE_goose_egg_count_l742_74215

-- Define the number of goose eggs laid at the pond
def total_eggs : ℕ := 2000

-- Define the fraction of eggs that hatched
def hatch_rate : ℚ := 2/3

-- Define the fraction of hatched geese that survived the first month
def first_month_survival_rate : ℚ := 3/4

-- Define the fraction of geese that survived the first month but did not survive the first year
def first_year_mortality_rate : ℚ := 3/5

-- Define the number of geese that survived the first year
def survived_first_year : ℕ := 100

-- Theorem statement
theorem goose_egg_count :
  (total_eggs : ℚ) * hatch_rate * first_month_survival_rate * (1 - first_year_mortality_rate) = survived_first_year :=
sorry

end NUMINAMATH_CALUDE_goose_egg_count_l742_74215


namespace NUMINAMATH_CALUDE_smallest_positive_period_of_f_l742_74278

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 2 * (Real.cos x)^2 - 1

def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

theorem smallest_positive_period_of_f :
  ∃ T > 0, is_periodic f T ∧ ∀ S, 0 < S → S < T → ¬ is_periodic f S :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_period_of_f_l742_74278


namespace NUMINAMATH_CALUDE_emma_wrapping_time_l742_74201

/-- Represents the time (in hours) it takes for Emma to wrap presents individually -/
def emma_time : ℝ := 6

/-- Represents the time (in hours) it takes for Troy to wrap presents individually -/
def troy_time : ℝ := 8

/-- Represents the time (in hours) Emma and Troy work together -/
def together_time : ℝ := 2

/-- Represents the additional time (in hours) Emma works alone after Troy leaves -/
def emma_extra_time : ℝ := 2.5

theorem emma_wrapping_time :
  emma_time = 6 ∧
  (together_time * (1 / emma_time + 1 / troy_time) + emma_extra_time / emma_time = 1) :=
sorry

end NUMINAMATH_CALUDE_emma_wrapping_time_l742_74201


namespace NUMINAMATH_CALUDE_max_value_of_expression_l742_74294

theorem max_value_of_expression (m n t : ℝ) (hm : m > 0) (hn : n > 0) (ht : t > 0)
  (heq : m^2 - 3*m*n + 4*n^2 - t = 0) :
  ∃ (m₀ n₀ t₀ : ℝ), m₀ > 0 ∧ n₀ > 0 ∧ t₀ > 0 ∧
    m₀^2 - 3*m₀*n₀ + 4*n₀^2 - t₀ = 0 ∧
    (∀ m' n' t' : ℝ, m' > 0 → n' > 0 → t' > 0 → m'^2 - 3*m'*n' + 4*n'^2 - t' = 0 →
      t₀/(m₀*n₀) ≤ t'/(m'*n')) ∧
    (∀ m' n' t' : ℝ, m' > 0 → n' > 0 → t' > 0 → m'^2 - 3*m'*n' + 4*n'^2 - t' = 0 →
      m' + 2*n' - t' ≤ 2) ∧
    m₀ + 2*n₀ - t₀ = 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l742_74294


namespace NUMINAMATH_CALUDE_distinct_sets_count_l742_74234

def A : Finset ℕ := {1, 2, 3, 4}
def B : Finset ℕ := {5, 6, 7}
def C : Finset ℕ := {8, 9}

def form_sets (X Y : Finset ℕ) : Finset (Finset ℕ) :=
  (X.product Y).image (λ (x, y) => {x, y})

theorem distinct_sets_count :
  (form_sets A B ∪ form_sets A C ∪ form_sets B C).card = 26 := by
  sorry

end NUMINAMATH_CALUDE_distinct_sets_count_l742_74234


namespace NUMINAMATH_CALUDE_max_leap_years_in_period_l742_74226

/-- Represents the number of years in a period -/
def period : ℕ := 200

/-- Represents the frequency of leap years -/
def leap_year_frequency : ℕ := 5

/-- Calculates the maximum number of leap years in the given period -/
def max_leap_years : ℕ := period / leap_year_frequency

/-- Theorem stating that the maximum number of leap years in a 200-year period is 40,
    given that leap years occur every 5 years -/
theorem max_leap_years_in_period :
  max_leap_years = 40 :=
by sorry

end NUMINAMATH_CALUDE_max_leap_years_in_period_l742_74226


namespace NUMINAMATH_CALUDE_sin_plus_power_cos_pi_third_l742_74289

theorem sin_plus_power_cos_pi_third :
  Real.sin 3 + 2^(8-3) * Real.cos (π/3) = Real.sin 3 + 16 := by
  sorry

end NUMINAMATH_CALUDE_sin_plus_power_cos_pi_third_l742_74289


namespace NUMINAMATH_CALUDE_union_when_m_is_4_intersection_condition_l742_74267

-- Define sets A and B
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

-- Theorem 1: When m = 4, A ∪ B = {x | -2 ≤ x ≤ 7}
theorem union_when_m_is_4 :
  A ∪ B 4 = {x : ℝ | -2 ≤ x ∧ x ≤ 7} := by sorry

-- Theorem 2: B ∩ A = B if and only if m ∈ (-∞, 3]
theorem intersection_condition :
  ∀ m : ℝ, B m ∩ A = B m ↔ m ≤ 3 := by sorry

end NUMINAMATH_CALUDE_union_when_m_is_4_intersection_condition_l742_74267


namespace NUMINAMATH_CALUDE_path_length_for_73_l742_74212

/-- The length of a path along squares constructed on subdivisions of a segment --/
def path_length (segment_length : ℝ) : ℝ :=
  3 * segment_length

theorem path_length_for_73 :
  path_length 73 = 219 := by
  sorry

end NUMINAMATH_CALUDE_path_length_for_73_l742_74212


namespace NUMINAMATH_CALUDE_john_trees_chopped_l742_74253

/-- Represents the number of trees John chopped down -/
def num_trees : ℕ := 30

/-- Represents the number of planks that can be made from each tree -/
def planks_per_tree : ℕ := 25

/-- Represents the number of planks needed to make one table -/
def planks_per_table : ℕ := 15

/-- Represents the selling price of each table in dollars -/
def price_per_table : ℕ := 300

/-- Represents the total labor cost in dollars -/
def labor_cost : ℕ := 3000

/-- Represents the profit John made in dollars -/
def profit : ℕ := 12000

theorem john_trees_chopped :
  num_trees * planks_per_tree / planks_per_table * price_per_table - labor_cost = profit :=
sorry

end NUMINAMATH_CALUDE_john_trees_chopped_l742_74253


namespace NUMINAMATH_CALUDE_root_product_theorem_l742_74217

-- Define the polynomial f(x) = x^6 + x^3 + 1
def f (x : ℂ) : ℂ := x^6 + x^3 + 1

-- Define the function g(x) = x^2 - 3
def g (x : ℂ) : ℂ := x^2 - 3

-- State the theorem
theorem root_product_theorem (x₁ x₂ x₃ x₄ x₅ x₆ : ℂ) 
  (hroots : (X - x₁) * (X - x₂) * (X - x₃) * (X - x₄) * (X - x₅) * (X - x₆) = f X) :
  g x₁ * g x₂ * g x₃ * g x₄ * g x₅ * g x₆ = 757 := by
  sorry

end NUMINAMATH_CALUDE_root_product_theorem_l742_74217


namespace NUMINAMATH_CALUDE_f_composition_value_l742_74229

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then 2 * x^3
  else if 0 ≤ x ∧ x < Real.pi / 2 then -Real.sin x
  else 0  -- undefined for x ≥ π/2, but we need to cover all cases

theorem f_composition_value : f (f (Real.pi / 6)) = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_value_l742_74229


namespace NUMINAMATH_CALUDE_malar_completion_time_l742_74205

/-- The number of days Malar takes to complete the task alone -/
def M : ℝ := 60

/-- The number of days Roja takes to complete the task alone -/
def R : ℝ := 84

/-- The number of days Malar and Roja take to complete the task together -/
def T : ℝ := 35

theorem malar_completion_time :
  (1 / M + 1 / R = 1 / T) → M = 60 := by
  sorry

end NUMINAMATH_CALUDE_malar_completion_time_l742_74205


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l742_74251

def set_A : Set ℝ := {x | x^2 + 2*x = 0}
def set_B : Set ℝ := {x | x^2 - 2*x ≤ 0}

theorem intersection_of_A_and_B : set_A ∩ set_B = {0} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l742_74251


namespace NUMINAMATH_CALUDE_range_of_m_l742_74261

theorem range_of_m (a b m : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b - a * b = 0)
  (h_log : ∀ m : ℝ, Real.log ((m^2) / (a + b)) ≤ 0) :
  -2 ≤ m ∧ m ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l742_74261


namespace NUMINAMATH_CALUDE_martha_black_butterflies_l742_74206

def butterfly_collection (total blue yellow black : ℕ) : Prop :=
  total = blue + yellow + black ∧ blue = 2 * yellow

theorem martha_black_butterflies :
  ∀ total blue yellow black : ℕ,
  butterfly_collection total blue yellow black →
  total = 19 →
  blue = 6 →
  black = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_martha_black_butterflies_l742_74206


namespace NUMINAMATH_CALUDE_same_route_probability_l742_74211

theorem same_route_probability (num_routes : ℕ) (num_students : ℕ) : 
  num_routes = 3 → num_students = 2 → 
  (num_routes : ℝ) / (num_routes * num_routes : ℝ) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_same_route_probability_l742_74211


namespace NUMINAMATH_CALUDE_monica_study_ratio_l742_74240

/-- Monica's study schedule problem -/
theorem monica_study_ratio : 
  ∀ (thursday_hours : ℝ),
  thursday_hours > 0 →
  2 + thursday_hours + (thursday_hours / 2) + (2 + thursday_hours + (thursday_hours / 2)) = 22 →
  thursday_hours / 2 = 3 / 1 := by
sorry

end NUMINAMATH_CALUDE_monica_study_ratio_l742_74240


namespace NUMINAMATH_CALUDE_composite_number_l742_74259

theorem composite_number : ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ (2^17 + 2^5 - 1 = a * b) := by
  sorry

end NUMINAMATH_CALUDE_composite_number_l742_74259


namespace NUMINAMATH_CALUDE_johnsRemainingMoneyTheorem_l742_74263

/-- The amount of money John has left after purchasing pizzas and drinks -/
def johnsRemainingMoney (d : ℝ) : ℝ :=
  let drinkCost := d
  let mediumPizzaCost := 3 * d
  let largePizzaCost := 4 * d
  let totalCost := 5 * drinkCost + mediumPizzaCost + 2 * largePizzaCost
  50 - totalCost

/-- Theorem stating that John's remaining money is 50 - 16d -/
theorem johnsRemainingMoneyTheorem (d : ℝ) :
  johnsRemainingMoney d = 50 - 16 * d :=
by sorry

end NUMINAMATH_CALUDE_johnsRemainingMoneyTheorem_l742_74263


namespace NUMINAMATH_CALUDE_trig_identity_l742_74269

theorem trig_identity (α : ℝ) (h : Real.sin α + 3 * Real.cos α = 0) : 
  2 * Real.sin (2 * α) - (Real.cos α)^2 = -13/10 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l742_74269


namespace NUMINAMATH_CALUDE_average_marks_of_passed_boys_l742_74287

theorem average_marks_of_passed_boys
  (total_boys : ℕ)
  (overall_average : ℚ)
  (passed_boys : ℕ)
  (failed_average : ℚ)
  (h1 : total_boys = 120)
  (h2 : overall_average = 38)
  (h3 : passed_boys = 115)
  (h4 : failed_average = 15)
  : ∃ (passed_average : ℚ), passed_average = 39 ∧
    overall_average * total_boys = passed_average * passed_boys + failed_average * (total_boys - passed_boys) := by
  sorry

end NUMINAMATH_CALUDE_average_marks_of_passed_boys_l742_74287


namespace NUMINAMATH_CALUDE_hyperbola_axis_ratio_implies_m_l742_74281

/-- Represents a hyperbola with equation mx^2 + y^2 = 1 -/
structure Hyperbola (m : ℝ) where
  equation : ∀ (x y : ℝ), m * x^2 + y^2 = 1

/-- The length of the imaginary axis of the hyperbola -/
def imaginary_axis_length (h : Hyperbola m) : ℝ := sorry

/-- The length of the real axis of the hyperbola -/
def real_axis_length (h : Hyperbola m) : ℝ := sorry

/-- 
  Theorem: For a hyperbola with equation mx^2 + y^2 = 1, 
  if the length of the imaginary axis is twice the length of the real axis, 
  then m = -1/4
-/
theorem hyperbola_axis_ratio_implies_m (m : ℝ) (h : Hyperbola m) 
  (axis_ratio : imaginary_axis_length h = 2 * real_axis_length h) : 
  m = -1/4 := by sorry

end NUMINAMATH_CALUDE_hyperbola_axis_ratio_implies_m_l742_74281


namespace NUMINAMATH_CALUDE_consecutive_integers_deduction_l742_74245

theorem consecutive_integers_deduction (n : ℕ) (avg : ℚ) (new_avg : ℚ) : 
  n = 30 → 
  avg = 50 → 
  new_avg = 34.3 → 
  let sum := n * avg
  let first_deduction := 29
  let last_deduction := 1
  let deduction_sum := n.pred / 2 * (first_deduction + last_deduction)
  let final_deduction := 6 + 12 + 18
  let new_sum := sum - deduction_sum - final_deduction
  new_avg = new_sum / n := by sorry

end NUMINAMATH_CALUDE_consecutive_integers_deduction_l742_74245


namespace NUMINAMATH_CALUDE_dice_trick_existence_l742_74296

def DicePair : Type := { p : ℕ × ℕ // p.1 ≤ p.2 ∧ p.1 ≥ 1 ∧ p.2 ≤ 6 }

theorem dice_trick_existence :
  ∃ f : DicePair → ℕ,
    Function.Bijective f ∧
    (∀ p : DicePair, 3 ≤ f p ∧ f p ≤ 21) :=
sorry

end NUMINAMATH_CALUDE_dice_trick_existence_l742_74296


namespace NUMINAMATH_CALUDE_triangle_area_doubles_l742_74285

theorem triangle_area_doubles (a b θ : ℝ) (ha : a > 0) (hb : b > 0) (hθ : 0 < θ ∧ θ < π) :
  let area := (1 / 2) * a * b * Real.sin θ
  let new_area := (1 / 2) * (2 * a) * b * Real.sin θ
  new_area = 2 * area := by sorry

end NUMINAMATH_CALUDE_triangle_area_doubles_l742_74285


namespace NUMINAMATH_CALUDE_floor_neg_seven_fourths_l742_74232

theorem floor_neg_seven_fourths : ⌊(-7 : ℚ) / 4⌋ = -2 := by
  sorry

end NUMINAMATH_CALUDE_floor_neg_seven_fourths_l742_74232


namespace NUMINAMATH_CALUDE_sum_of_three_pentagons_l742_74220

/-- The value of a square -/
def square_value : ℚ := sorry

/-- The value of a pentagon -/
def pentagon_value : ℚ := sorry

/-- First equation: 3 squares + 2 pentagons = 27 -/
axiom eq1 : 3 * square_value + 2 * pentagon_value = 27

/-- Second equation: 2 squares + 3 pentagons = 25 -/
axiom eq2 : 2 * square_value + 3 * pentagon_value = 25

/-- Theorem: The sum of three pentagons equals 63/5 -/
theorem sum_of_three_pentagons : 3 * pentagon_value = 63 / 5 := by sorry

end NUMINAMATH_CALUDE_sum_of_three_pentagons_l742_74220


namespace NUMINAMATH_CALUDE_max_value_expression_max_value_achievable_l742_74244

theorem max_value_expression (x : ℝ) (hx : x > 0) :
  (3 * x^2 + 2 - Real.sqrt (9 * x^4 + 4)) / x ≤ 12 / (5 + 3 * Real.sqrt 3) :=
sorry

theorem max_value_achievable :
  ∃ x : ℝ, x > 0 ∧ (3 * x^2 + 2 - Real.sqrt (9 * x^4 + 4)) / x = 12 / (5 + 3 * Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_max_value_expression_max_value_achievable_l742_74244


namespace NUMINAMATH_CALUDE_investment_sum_l742_74250

/-- Proves that if a sum P is invested at 15% p.a. for two years instead of 12% p.a. for two years, 
    and the difference in interest is Rs. 840, then P = Rs. 14,000. -/
theorem investment_sum (P : ℝ) : 
  (P * 0.15 * 2 - P * 0.12 * 2 = 840) → P = 14000 := by
  sorry

end NUMINAMATH_CALUDE_investment_sum_l742_74250


namespace NUMINAMATH_CALUDE_complex_equation_solution_l742_74256

theorem complex_equation_solution (z : ℂ) : (1 - Complex.I)^2 * z = 3 + 2 * Complex.I → z = -1 + (3/2) * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l742_74256


namespace NUMINAMATH_CALUDE_extremum_value_theorem_l742_74246

/-- The function f(x) = x sin x achieves an extremum at x₀ -/
def has_extremum (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ ε > 0, ∀ x, |x - x₀| < ε → f x ≤ f x₀ ∨ f x ≥ f x₀

theorem extremum_value_theorem (x₀ : ℝ) :
  has_extremum (fun x => x * Real.sin x) x₀ →
  (1 + x₀^2) * (1 + Real.cos (2 * x₀)) = 2 :=
by sorry

end NUMINAMATH_CALUDE_extremum_value_theorem_l742_74246


namespace NUMINAMATH_CALUDE_solve_equation_l742_74276

theorem solve_equation (y : ℚ) (h : 1/3 - 1/4 = 1/y) : y = 12 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l742_74276


namespace NUMINAMATH_CALUDE_treasure_value_l742_74203

theorem treasure_value (fonzie_investment aunt_bee_investment lapis_investment lapis_share : ℚ)
  (h1 : fonzie_investment = 7000)
  (h2 : aunt_bee_investment = 8000)
  (h3 : lapis_investment = 9000)
  (h4 : lapis_share = 337500) :
  let total_investment := fonzie_investment + aunt_bee_investment + lapis_investment
  let lapis_proportion := lapis_investment / total_investment
  lapis_proportion * (lapis_share / lapis_proportion) = 1125000 := by
sorry

end NUMINAMATH_CALUDE_treasure_value_l742_74203


namespace NUMINAMATH_CALUDE_equation_solution_l742_74221

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem equation_solution :
  ∃ (z : ℂ), (1 - i * z = -1 + i * z) ∧ (z = -i) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l742_74221


namespace NUMINAMATH_CALUDE_factorization_identities_l742_74291

theorem factorization_identities :
  (∀ m : ℝ, m^3 - 16*m = m*(m+4)*(m-4)) ∧
  (∀ a x : ℝ, -4*a^2*x + 12*a*x - 9*x = -x*(2*a-3)^2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_identities_l742_74291


namespace NUMINAMATH_CALUDE_archimedes_academy_students_l742_74260

/-- The number of distinct students preparing for AMC 8 at Archimedes Academy -/
def distinct_students (algebra_students calculus_students statistics_students overlap : ℕ) : ℕ :=
  algebra_students + calculus_students + statistics_students - overlap

/-- Theorem stating the number of distinct students preparing for AMC 8 at Archimedes Academy -/
theorem archimedes_academy_students :
  distinct_students 13 10 12 3 = 32 := by
  sorry

end NUMINAMATH_CALUDE_archimedes_academy_students_l742_74260


namespace NUMINAMATH_CALUDE_n_in_interval_l742_74208

def is_repeating_decimal (q : ℚ) (period : ℕ) : Prop :=
  ∃ (k : ℕ), q * (10^period - 1) = k ∧ k < 10^period

theorem n_in_interval (n : ℕ) :
  n < 500 →
  is_repeating_decimal (1 / n) 4 →
  is_repeating_decimal (1 / (n + 4)) 2 →
  n ∈ Set.Icc 1 125 :=
sorry

end NUMINAMATH_CALUDE_n_in_interval_l742_74208


namespace NUMINAMATH_CALUDE_greatest_possible_k_l742_74210

theorem greatest_possible_k (k : ℝ) : 
  (∃ x y : ℝ, x^2 + k*x + 8 = 0 ∧ y^2 + k*y + 8 = 0 ∧ |x - y| = Real.sqrt 73) →
  k ≤ Real.sqrt 105 :=
by sorry

end NUMINAMATH_CALUDE_greatest_possible_k_l742_74210


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l742_74230

/-- Two arithmetic sequences and their sums -/
def arithmetic_sequences (a b : ℕ → ℝ) (A B : ℕ → ℝ) : Prop :=
  (∀ n, A n = (n * (a 1 + a n)) / 2) ∧
  (∀ n, B n = (n * (b 1 + b n)) / 2) ∧
  (∀ n, a (n + 1) - a n = a 2 - a 1) ∧
  (∀ n, b (n + 1) - b n = b 2 - b 1)

/-- The main theorem -/
theorem arithmetic_sequence_ratio 
  (a b : ℕ → ℝ) (A B : ℕ → ℝ) 
  (h : arithmetic_sequences a b A B) 
  (h_ratio : ∀ n : ℕ, A n / B n = (2 * n - 1) / (3 * n + 1)) :
  ∀ n : ℕ, a n / b n = (4 * n - 3) / (6 * n - 2) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l742_74230


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l742_74204

def set_A : Set ℝ := {x | x^2 - 3*x - 4 < 0}
def set_B : Set ℝ := {-4, 1, 3, 5}

theorem intersection_of_A_and_B :
  set_A ∩ set_B = {1, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l742_74204


namespace NUMINAMATH_CALUDE_quartic_polynomial_unique_l742_74214

/-- A monic quartic polynomial with real coefficients -/
def QuarticPolynomial (a b c d : ℝ) : ℂ → ℂ := fun x ↦ x^4 + a*x^3 + b*x^2 + c*x + d

theorem quartic_polynomial_unique 
  (q : ℂ → ℂ) 
  (h_monic : q = fun x ↦ x^4 + (q 1 - 1)*x^3 + (q 2 - q 1 + 2)*x^2 + (q 3 - q 2 + q 1 - 3)*x + q 0)
  (h_real : ∀ x : ℝ, ∃ y : ℝ, q x = y)
  (h_root : q (2 + I) = 0)
  (h_value : q 0 = -120) :
  q = QuarticPolynomial 1 (-19) (-116) 120 := by
  sorry

end NUMINAMATH_CALUDE_quartic_polynomial_unique_l742_74214


namespace NUMINAMATH_CALUDE_math_paper_probability_l742_74235

theorem math_paper_probability (total_pages : ℕ) (math_pages : ℕ) (prob : ℚ) :
  total_pages = 12 →
  math_pages = 2 →
  prob = math_pages / total_pages →
  prob = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_math_paper_probability_l742_74235


namespace NUMINAMATH_CALUDE_P_speed_is_8_l742_74219

/-- Represents the cycling speed of P in kmph -/
def P_speed : ℝ := 8

/-- J's walking speed in kmph -/
def J_speed : ℝ := 6

/-- Time (in hours) J walks before P starts -/
def time_before_P_starts : ℝ := 1.5

/-- Total time (in hours) from J's start to the point where J is 3 km behind P -/
def total_time : ℝ := 7.5

/-- Distance (in km) J is behind P at the end -/
def distance_behind : ℝ := 3

theorem P_speed_is_8 :
  P_speed = 8 :=
sorry

#check P_speed_is_8

end NUMINAMATH_CALUDE_P_speed_is_8_l742_74219


namespace NUMINAMATH_CALUDE_trihedral_angle_relations_l742_74262

/-- Represents a trihedral angle -/
structure TrihedralAngle where
  /-- Plane angles of the trihedral angle -/
  plane_angles : Fin 3 → ℝ
  /-- Dihedral angles of the trihedral angle -/
  dihedral_angles : Fin 3 → ℝ

/-- Theorem about the relationship between plane angles and dihedral angles in a trihedral angle -/
theorem trihedral_angle_relations (t : TrihedralAngle) :
  (∀ i : Fin 3, t.plane_angles i > Real.pi / 2 → ∀ j : Fin 3, t.dihedral_angles j > Real.pi / 2) ∧
  (∀ i : Fin 3, t.dihedral_angles i < Real.pi / 2 → ∀ j : Fin 3, t.plane_angles j < Real.pi / 2) :=
sorry

end NUMINAMATH_CALUDE_trihedral_angle_relations_l742_74262


namespace NUMINAMATH_CALUDE_first_day_exceeding_150_fungi_l742_74237

def fungi_growth (n : ℕ) : ℕ := 4 * 2^n

theorem first_day_exceeding_150_fungi : 
  (∃ n : ℕ, fungi_growth n > 150) ∧ 
  (∀ m : ℕ, m < 6 → fungi_growth m ≤ 150) ∧
  (fungi_growth 6 > 150) :=
by sorry

end NUMINAMATH_CALUDE_first_day_exceeding_150_fungi_l742_74237


namespace NUMINAMATH_CALUDE_shirts_arrangement_l742_74272

/-- The number of ways to arrange shirts -/
def arrange_shirts (red : Nat) (green : Nat) : Nat :=
  Nat.factorial (red + green) / (Nat.factorial red * Nat.factorial green)

/-- The number of ways to arrange shirts with green shirts together -/
def arrange_shirts_green_together (red : Nat) (green : Nat) : Nat :=
  arrange_shirts red 1

theorem shirts_arrangement :
  arrange_shirts 3 2 - arrange_shirts_green_together 3 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_shirts_arrangement_l742_74272


namespace NUMINAMATH_CALUDE_davis_oldest_child_age_l742_74236

/-- The age of the oldest Davis child given the conditions -/
def oldest_child_age (avg_age : ℕ) (younger_child1 : ℕ) (younger_child2 : ℕ) : ℕ :=
  3 * avg_age - younger_child1 - younger_child2

/-- Theorem stating the age of the oldest Davis child -/
theorem davis_oldest_child_age :
  oldest_child_age 10 7 9 = 14 := by
  sorry

end NUMINAMATH_CALUDE_davis_oldest_child_age_l742_74236


namespace NUMINAMATH_CALUDE_symmetry_axes_intersection_l742_74228

-- Define a polygon as a set of points in 2D space
def Polygon := Set (ℝ × ℝ)

-- Define an axis of symmetry for a polygon
def IsAxisOfSymmetry (p : Polygon) (axis : Set (ℝ × ℝ)) : Prop := sorry

-- Define the center of mass for a set of points
def CenterOfMass (points : Set (ℝ × ℝ)) : ℝ × ℝ := sorry

-- Define a property that a point lies on a line
def PointOnLine (point : ℝ × ℝ) (line : Set (ℝ × ℝ)) : Prop := sorry

-- Theorem statement
theorem symmetry_axes_intersection (p : Polygon) 
  (h_multiple_axes : ∃ (axis1 axis2 : Set (ℝ × ℝ)), axis1 ≠ axis2 ∧ IsAxisOfSymmetry p axis1 ∧ IsAxisOfSymmetry p axis2) :
  ∀ (axis : Set (ℝ × ℝ)), IsAxisOfSymmetry p axis → 
    PointOnLine (CenterOfMass p) axis :=
sorry

end NUMINAMATH_CALUDE_symmetry_axes_intersection_l742_74228


namespace NUMINAMATH_CALUDE_cube_sum_ge_mixed_product_l742_74243

theorem cube_sum_ge_mixed_product {a b : ℝ} (ha : a > 0) (hb : b > 0) :
  a^3 + b^3 ≥ a^2 * b + a * b^2 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_ge_mixed_product_l742_74243


namespace NUMINAMATH_CALUDE_f_max_value_l742_74254

def f (x : ℝ) := |x| - |x - 3|

theorem f_max_value :
  (∀ x, f x ≤ 3) ∧ (∃ x, f x = 3) := by sorry

end NUMINAMATH_CALUDE_f_max_value_l742_74254


namespace NUMINAMATH_CALUDE_log_difference_equals_one_l742_74270

-- Define the logarithm function
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- State the theorem
theorem log_difference_equals_one (a : ℝ) 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (h3 : log a 3 > log a 2) : 
  (log a (2 * a) - log a a = 1) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_difference_equals_one_l742_74270


namespace NUMINAMATH_CALUDE_complete_square_sum_l742_74202

theorem complete_square_sum (x : ℝ) : 
  (x^2 - 10*x + 15 = 0) → 
  ∃ (d e : ℤ), ((x + d : ℝ)^2 = e) ∧ (d + e = 5) :=
by sorry

end NUMINAMATH_CALUDE_complete_square_sum_l742_74202


namespace NUMINAMATH_CALUDE_bridge_length_bridge_length_proof_l742_74239

/-- The length of a bridge that a train can cross, given the train's length, speed, and time to cross. -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let total_distance := train_speed_ms * crossing_time
  total_distance - train_length

/-- Proves that the length of the bridge is 230 meters given the specified conditions. -/
theorem bridge_length_proof :
  bridge_length 145 45 30 = 230 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_bridge_length_proof_l742_74239


namespace NUMINAMATH_CALUDE_quadratic_b_value_l742_74224

-- Define the quadratic function
def f (b c : ℝ) (x : ℝ) : ℝ := 2 * x^2 + b * x + c

-- Define the theorem
theorem quadratic_b_value :
  ∀ (b c y₁ y₂ : ℝ),
  (f b c 2 = y₁) →
  (f b c (-2) = y₂) →
  (y₁ - y₂ = 12) →
  b = 3 := by
sorry


end NUMINAMATH_CALUDE_quadratic_b_value_l742_74224


namespace NUMINAMATH_CALUDE_rectangle_area_change_l742_74292

theorem rectangle_area_change (L W : ℝ) (h1 : L > 0) (h2 : W > 0) : 
  let new_length := 1.4 * L
  let new_width := W / 2
  let original_area := L * W
  let new_area := new_length * new_width
  (new_area / original_area) = 0.7 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_change_l742_74292


namespace NUMINAMATH_CALUDE_meeting_time_calculation_l742_74200

/-- Two people moving towards each other -/
structure TwoPersonMovement where
  v₁ : ℝ  -- Speed of person 1
  v₂ : ℝ  -- Speed of person 2
  t₂ : ℝ  -- Waiting time after turning around

/-- The theorem statement -/
theorem meeting_time_calculation (m : TwoPersonMovement) 
  (h₁ : m.v₁ = 6)   -- Speed of person 1 is 6 m/s
  (h₂ : m.v₂ = 4)   -- Speed of person 2 is 4 m/s
  (h₃ : m.t₂ = 600) -- Waiting time is 10 minutes (600 seconds)
  : ∃ t₁ : ℝ, t₁ = 1200 ∧ (m.v₁ * t₁ + m.v₂ * t₁ = 2 * m.v₂ * t₁ + m.v₂ * m.t₂) := by
  sorry

end NUMINAMATH_CALUDE_meeting_time_calculation_l742_74200


namespace NUMINAMATH_CALUDE_cuboid_dimensions_l742_74271

theorem cuboid_dimensions (x y v : ℕ) 
  (h1 : x * y * v - v = 602)
  (h2 : x * y * v - x = 605)
  (h3 : v = x + 3)
  (hx : x > 0)
  (hy : y > 0)
  (hv : v > 0) :
  x = 11 ∧ y = 4 ∧ v = 14 := by
sorry

end NUMINAMATH_CALUDE_cuboid_dimensions_l742_74271


namespace NUMINAMATH_CALUDE_remainder_theorem_l742_74277

theorem remainder_theorem (n : ℤ) : 
  (2 * n) % 11 = 2 → n % 22 = 1 :=
by sorry

end NUMINAMATH_CALUDE_remainder_theorem_l742_74277


namespace NUMINAMATH_CALUDE_partition_six_into_three_l742_74283

/-- The number of ways to partition a set of n elements into k disjoint subsets -/
def partitionWays (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to partition a set of 6 elements into 3 disjoint subsets is 15 -/
theorem partition_six_into_three : partitionWays 6 3 = 15 := by sorry

end NUMINAMATH_CALUDE_partition_six_into_three_l742_74283


namespace NUMINAMATH_CALUDE_max_b_minus_a_l742_74297

theorem max_b_minus_a (a b : ℝ) (ha : a < 0)
  (h : ∀ x : ℝ, (3 * x^2 + a) * (2 * x + b) ≥ 0) :
  ∃ (max : ℝ), max = 1/3 ∧ b - a ≤ max ∧
  ∀ (a' b' : ℝ), a' < 0 → (∀ x : ℝ, (3 * x^2 + a') * (2 * x + b') ≥ 0) →
  b' - a' ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_b_minus_a_l742_74297
