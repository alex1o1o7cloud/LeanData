import Mathlib

namespace NUMINAMATH_CALUDE_minimum_cubes_for_valid_assembly_l3254_325420

/-- Represents a cube with either one or two snaps -/
inductive Cube
  | SingleSnap
  | DoubleSnap

/-- An assembly of cubes -/
def Assembly := List Cube

/-- Checks if an assembly is valid (all snaps covered, only receptacles exposed) -/
def isValidAssembly : Assembly → Bool := sorry

/-- Counts the number of cubes in an assembly -/
def countCubes : Assembly → Nat := sorry

theorem minimum_cubes_for_valid_assembly :
  ∃ (a : Assembly), isValidAssembly a ∧ countCubes a = 6 ∧
  ∀ (b : Assembly), isValidAssembly b → countCubes b ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_minimum_cubes_for_valid_assembly_l3254_325420


namespace NUMINAMATH_CALUDE_arc_length_for_72_degrees_l3254_325430

theorem arc_length_for_72_degrees (d : ℝ) (θ_deg : ℝ) (l : ℝ) : 
  d = 4 →  -- diameter is 4 cm
  θ_deg = 72 →  -- central angle is 72°
  l = d / 2 * (θ_deg * π / 180) →  -- arc length formula
  l = 4 * π / 5 :=  -- arc length is 4π/5 cm
by sorry

end NUMINAMATH_CALUDE_arc_length_for_72_degrees_l3254_325430


namespace NUMINAMATH_CALUDE_last_four_average_l3254_325431

theorem last_four_average (list : List ℝ) : 
  list.length = 7 →
  list.sum / 7 = 62 →
  (list.take 3).sum / 3 = 55 →
  (list.drop 3).sum / 4 = 67.25 :=
by sorry

end NUMINAMATH_CALUDE_last_four_average_l3254_325431


namespace NUMINAMATH_CALUDE_green_peaches_count_l3254_325438

/-- Given a basket of peaches with a total of 10 peaches and 4 red peaches,
    prove that there are 6 green peaches in the basket. -/
theorem green_peaches_count (total_peaches : ℕ) (red_peaches : ℕ) (baskets : ℕ) :
  total_peaches = 10 → red_peaches = 4 → baskets = 1 →
  total_peaches - red_peaches = 6 := by
  sorry

end NUMINAMATH_CALUDE_green_peaches_count_l3254_325438


namespace NUMINAMATH_CALUDE_ratio_a_c_l3254_325425

-- Define the ratios
def ratio_a_b : ℚ := 5 / 3
def ratio_b_c : ℚ := 1 / 5

-- Theorem statement
theorem ratio_a_c (a b c : ℚ) (h1 : a / b = ratio_a_b) (h2 : b / c = ratio_b_c) : 
  a / c = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_a_c_l3254_325425


namespace NUMINAMATH_CALUDE_advance_tickets_sold_l3254_325452

theorem advance_tickets_sold (advance_cost same_day_cost total_tickets total_receipts : ℕ) 
  (h1 : advance_cost = 20)
  (h2 : same_day_cost = 30)
  (h3 : total_tickets = 60)
  (h4 : total_receipts = 1600) :
  ∃ (advance_sold : ℕ), 
    advance_sold * advance_cost + (total_tickets - advance_sold) * same_day_cost = total_receipts ∧ 
    advance_sold = 20 :=
by sorry

end NUMINAMATH_CALUDE_advance_tickets_sold_l3254_325452


namespace NUMINAMATH_CALUDE_quadrilateral_side_length_l3254_325429

/-- Represents a quadrilateral with sides a, b, c, d --/
structure Quadrilateral :=
  (a b c d : ℝ)

/-- Represents the properties of the specific quadrilateral in the problem --/
def ProblemQuadrilateral (q : Quadrilateral) (x y : ℕ) : Prop :=
  q.a = 20 ∧ 
  q.a = x^2 + y^2 ∧ 
  q.b = x ∧ 
  q.c = y ∧ 
  x > 0 ∧ 
  y > 0 ∧ 
  q.d ≥ q.a ∧ 
  q.d ≥ q.b ∧ 
  q.d ≥ q.c

theorem quadrilateral_side_length 
  (q : Quadrilateral) 
  (x y : ℕ) 
  (h : ProblemQuadrilateral q x y) : 
  q.d = 4 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_side_length_l3254_325429


namespace NUMINAMATH_CALUDE_divisor_problem_l3254_325494

theorem divisor_problem (n : ℕ) (d : ℕ) (h1 : n > 0) (h2 : ∃ k : ℕ, n + 1 = k * d + 4) (h3 : ∃ m : ℕ, n = 2 * m + 1) : d = 6 := by
  sorry

end NUMINAMATH_CALUDE_divisor_problem_l3254_325494


namespace NUMINAMATH_CALUDE_probability_james_and_david_chosen_l3254_325474

-- Define the total number of workers
def total_workers : ℕ := 14

-- Define the number of workers to be chosen
def chosen_workers : ℕ := 2

-- Define the function to calculate combinations
def combination (n k : ℕ) : ℕ := 
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Theorem statement
theorem probability_james_and_david_chosen :
  (1 : ℚ) / (combination total_workers chosen_workers) = 1 / 91 :=
sorry

end NUMINAMATH_CALUDE_probability_james_and_david_chosen_l3254_325474


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l3254_325421

/-- 
For a quadratic equation x^2 + 8x + q = 0 to have two distinct real roots,
q must be less than 16.
-/
theorem quadratic_roots_condition (q : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + 8*x + q = 0 ∧ y^2 + 8*y + q = 0) ↔ q < 16 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l3254_325421


namespace NUMINAMATH_CALUDE_journey_distance_l3254_325464

theorem journey_distance (speed : ℝ) (time : ℝ) 
  (h1 : (speed + 1/2) * (3/4 * time) = speed * time)
  (h2 : (speed - 1/2) * (time + 3) = speed * time)
  : speed * time = 9 := by
  sorry

end NUMINAMATH_CALUDE_journey_distance_l3254_325464


namespace NUMINAMATH_CALUDE_profit_sharing_multiple_l3254_325451

/-- Given the conditions of a profit-sharing scenario, prove that the multiple of R's capital is 10. -/
theorem profit_sharing_multiple (P Q R k : ℚ) (total_profit : ℚ) : 
  4 * P = 6 * Q ∧ 
  4 * P = k * R ∧ 
  total_profit = 4340 ∧ 
  R * (total_profit / (P + Q + R)) = 840 →
  k = 10 := by
  sorry

end NUMINAMATH_CALUDE_profit_sharing_multiple_l3254_325451


namespace NUMINAMATH_CALUDE_postage_calculation_l3254_325402

/-- Calculates the postage for a letter given the base fee, additional fee per ounce, and weight -/
def calculatePostage (baseFee : ℚ) (additionalFeePerOunce : ℚ) (weight : ℚ) : ℚ :=
  baseFee + additionalFeePerOunce * (weight - 1)

/-- Theorem stating that the postage for a 5.3 ounce letter is $1.425 under the given fee structure -/
theorem postage_calculation :
  let baseFee : ℚ := 35 / 100  -- 35 cents in dollars
  let additionalFeePerOunce : ℚ := 25 / 100  -- 25 cents in dollars
  let weight : ℚ := 53 / 10  -- 5.3 ounces
  calculatePostage baseFee additionalFeePerOunce weight = 1425 / 1000 := by
  sorry

#eval calculatePostage (35/100) (25/100) (53/10)

end NUMINAMATH_CALUDE_postage_calculation_l3254_325402


namespace NUMINAMATH_CALUDE_perfect_square_implies_zero_l3254_325455

theorem perfect_square_implies_zero (a b : ℤ) :
  (∀ n : ℕ, ∃ k : ℤ, a * 2013^n + b = k^2) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_implies_zero_l3254_325455


namespace NUMINAMATH_CALUDE_arithmetic_mean_after_removal_l3254_325432

theorem arithmetic_mean_after_removal (S : Finset ℝ) (x y : ℝ) :
  S.card = 60 →
  x ∈ S →
  y ∈ S →
  x = 60 →
  y = 75 →
  (S.sum id) / S.card = 45 →
  ((S.sum id - (x + y)) / (S.card - 2) : ℝ) = 465 / 106 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_after_removal_l3254_325432


namespace NUMINAMATH_CALUDE_negative_square_cubed_l3254_325410

theorem negative_square_cubed (a : ℝ) : (-a^2)^3 = -a^6 := by
  sorry

end NUMINAMATH_CALUDE_negative_square_cubed_l3254_325410


namespace NUMINAMATH_CALUDE_same_parity_min_max_l3254_325470

/-- A set of elements related to positioning in a function or polynomial -/
def A_P : Set ℤ := sorry

/-- The smallest element of A_P -/
def min_element (A : Set ℤ) : ℤ := sorry

/-- The largest element of A_P -/
def max_element (A : Set ℤ) : ℤ := sorry

/-- A predicate to check if a number is even -/
def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

theorem same_parity_min_max : 
  is_even (min_element A_P) ↔ is_even (max_element A_P) := by sorry

end NUMINAMATH_CALUDE_same_parity_min_max_l3254_325470


namespace NUMINAMATH_CALUDE_unique_odd_k_for_sum_1372_l3254_325411

theorem unique_odd_k_for_sum_1372 :
  ∃! (k : ℤ), ∃ (m : ℕ), 
    (k % 2 = 1) ∧ 
    (m > 0) ∧ 
    (k * m + 5 * (m * (m - 1) / 2) = 1372) ∧ 
    (k = 211) := by
  sorry

end NUMINAMATH_CALUDE_unique_odd_k_for_sum_1372_l3254_325411


namespace NUMINAMATH_CALUDE_three_numbers_sum_6_product_4_l3254_325467

theorem three_numbers_sum_6_product_4 :
  ∀ a b c : ℕ,
  a + b + c = 6 →
  a * b * c = 4 →
  ((a = 1 ∧ b = 1 ∧ c = 4) ∨
   (a = 1 ∧ b = 4 ∧ c = 1) ∨
   (a = 4 ∧ b = 1 ∧ c = 1)) :=
by sorry

end NUMINAMATH_CALUDE_three_numbers_sum_6_product_4_l3254_325467


namespace NUMINAMATH_CALUDE_system_solution_l3254_325439

theorem system_solution : ∃ (x y : ℚ), 
  (x + 4*y = 14) ∧ 
  ((x - 3) / 4 - (y - 3) / 3 = 1 / 12) ∧ 
  (x = 3) ∧ 
  (y = 11 / 4) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3254_325439


namespace NUMINAMATH_CALUDE_power_fraction_simplification_l3254_325481

theorem power_fraction_simplification :
  (3^2020 - 3^2018) / (3^2020 + 3^2018) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_simplification_l3254_325481


namespace NUMINAMATH_CALUDE_final_balance_is_correct_l3254_325422

/-- Represents a bank account with transactions and interest --/
structure BankAccount where
  initialBalance : ℝ
  annualInterestRate : ℝ
  monthlyInterestRate : ℝ
  shoeWithdrawalPercent : ℝ
  shoeDepositPercent : ℝ
  paycheckDepositPercent : ℝ
  giftWithdrawalPercent : ℝ

/-- Calculates the final balance after all transactions and interest --/
def finalBalance (account : BankAccount) : ℝ :=
  let shoeWithdrawal := account.initialBalance * account.shoeWithdrawalPercent
  let balanceAfterShoes := account.initialBalance - shoeWithdrawal
  let shoeDeposit := shoeWithdrawal * account.shoeDepositPercent
  let balanceAfterShoeDeposit := balanceAfterShoes + shoeDeposit
  let januaryInterest := balanceAfterShoeDeposit * account.monthlyInterestRate
  let balanceAfterJanuary := balanceAfterShoeDeposit + januaryInterest
  let paycheckDeposit := shoeWithdrawal * account.paycheckDepositPercent
  let balanceAfterPaycheck := balanceAfterJanuary + paycheckDeposit
  let februaryInterest := balanceAfterPaycheck * account.monthlyInterestRate
  let balanceAfterFebruary := balanceAfterPaycheck + februaryInterest
  let giftWithdrawal := balanceAfterFebruary * account.giftWithdrawalPercent
  let balanceAfterGift := balanceAfterFebruary - giftWithdrawal
  let marchInterest := balanceAfterGift * account.monthlyInterestRate
  balanceAfterGift + marchInterest

/-- Theorem stating that the final balance is correct --/
theorem final_balance_is_correct (account : BankAccount) : 
  account.initialBalance = 1200 ∧
  account.annualInterestRate = 0.03 ∧
  account.monthlyInterestRate = account.annualInterestRate / 12 ∧
  account.shoeWithdrawalPercent = 0.08 ∧
  account.shoeDepositPercent = 0.25 ∧
  account.paycheckDepositPercent = 1.5 ∧
  account.giftWithdrawalPercent = 0.05 →
  finalBalance account = 1217.15 := by
  sorry


end NUMINAMATH_CALUDE_final_balance_is_correct_l3254_325422


namespace NUMINAMATH_CALUDE_coinciding_rest_days_l3254_325495

/-- Chris's schedule cycle length -/
def chris_cycle : ℕ := 6

/-- Dana's schedule cycle length -/
def dana_cycle : ℕ := 7

/-- Total number of days -/
def total_days : ℕ := 1000

/-- Chris's rest days within his cycle -/
def chris_rest_days : List ℕ := [5, 6]

/-- Dana's rest days within her cycle -/
def dana_rest_days : List ℕ := [6, 7]

/-- The number of coinciding rest days for Chris and Dana in the first 1000 days -/
theorem coinciding_rest_days : 
  (List.filter (λ d : ℕ => 
    (d % chris_cycle ∈ chris_rest_days) ∧ 
    (d % dana_cycle ∈ dana_rest_days)) 
    (List.range total_days)).length = 23 := by
  sorry

end NUMINAMATH_CALUDE_coinciding_rest_days_l3254_325495


namespace NUMINAMATH_CALUDE_outflow_symmetry_outflow_ratio_replacement_time_ratio_l3254_325499

/-- Represents the structure of a sewage purification tower -/
structure SewageTower where
  layers : Nat
  outlets : Nat
  flow_distribution : List (List Rat)

/-- Calculates the outflow for a given outlet -/
def outflow (tower : SewageTower) (outlet : Nat) : Rat :=
  sorry

/-- Theorem stating that outflows of outlet 2 and 4 are equal -/
theorem outflow_symmetry (tower : SewageTower) :
  tower.outlets = 5 → outflow tower 2 = outflow tower 4 :=
  sorry

/-- Theorem stating the ratio of outflows for outlets 1, 2, and 3 -/
theorem outflow_ratio (tower : SewageTower) :
  tower.outlets = 5 →
  ∃ (k : Rat), outflow tower 1 = k ∧ outflow tower 2 = 4*k ∧ outflow tower 3 = 6*k :=
  sorry

/-- Calculates the wear rate for a given triangle in the tower -/
def wear_rate (tower : SewageTower) (triangle : Nat) : Rat :=
  sorry

/-- Theorem stating the replacement time ratio for slowest and fastest wearing triangles -/
theorem replacement_time_ratio (tower : SewageTower) :
  ∃ (slow fast : Nat),
    wear_rate tower slow = (1/8 : Rat) * wear_rate tower fast ∧
    ∀ t, wear_rate tower t ≥ wear_rate tower slow ∧
         wear_rate tower t ≤ wear_rate tower fast :=
  sorry

end NUMINAMATH_CALUDE_outflow_symmetry_outflow_ratio_replacement_time_ratio_l3254_325499


namespace NUMINAMATH_CALUDE_nuts_problem_l3254_325407

theorem nuts_problem (x y : ℕ) : 
  (70 ≤ x + y ∧ x + y ≤ 80) ∧ 
  (3 * x + 5 * y + x = 20 * x + 20) →
  x = 36 ∧ y = 41 :=
sorry

end NUMINAMATH_CALUDE_nuts_problem_l3254_325407


namespace NUMINAMATH_CALUDE_new_person_weight_l3254_325448

/-- Given a group of 10 people, if replacing one person weighing 65 kg
    with a new person increases the average weight by 7.2 kg,
    then the weight of the new person is 137 kg. -/
theorem new_person_weight
  (n : ℕ)
  (initial_weight : ℝ)
  (weight_increase : ℝ)
  (replaced_weight : ℝ)
  (h1 : n = 10)
  (h2 : weight_increase = 7.2)
  (h3 : replaced_weight = 65)
  : initial_weight + n * weight_increase = initial_weight - replaced_weight + 137 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l3254_325448


namespace NUMINAMATH_CALUDE_outdoor_section_length_l3254_325414

/-- The length of a rectangular outdoor section, given its width and area -/
theorem outdoor_section_length (width : ℝ) (area : ℝ) (h1 : width = 4) (h2 : area = 24) :
  area / width = 6 := by
  sorry

end NUMINAMATH_CALUDE_outdoor_section_length_l3254_325414


namespace NUMINAMATH_CALUDE_normal_distribution_standard_deviations_l3254_325458

/-- Proves that for a normal distribution with mean 14.0 and standard deviation 1.5,
    the value 11 is exactly 2 standard deviations less than the mean. -/
theorem normal_distribution_standard_deviations (μ σ x : ℝ) 
  (h_mean : μ = 14.0)
  (h_std_dev : σ = 1.5)
  (h_value : x = 11.0) :
  (μ - x) / σ = 2 := by
  sorry

end NUMINAMATH_CALUDE_normal_distribution_standard_deviations_l3254_325458


namespace NUMINAMATH_CALUDE_extreme_value_condition_l3254_325456

-- Define the function f(x)
def f (m n : ℝ) (x : ℝ) : ℝ := x^3 + 3*m*x^2 + n*x + m^2

-- Define the derivative of f(x)
def f_prime (m n : ℝ) (x : ℝ) : ℝ := 3*x^2 + 6*m*x + n

-- Theorem statement
theorem extreme_value_condition (m n : ℝ) :
  f m n (-1) = 0 ∧ f_prime m n (-1) = 0 → m + n = 11 := by
  sorry

end NUMINAMATH_CALUDE_extreme_value_condition_l3254_325456


namespace NUMINAMATH_CALUDE_parabola_ellipse_focus_coincide_l3254_325418

/-- The value of p for which the focus of the parabola y^2 = -2px coincides with the left focus of the ellipse (x^2/16) + (y^2/12) = 1 -/
theorem parabola_ellipse_focus_coincide : ∃ p : ℝ,
  (∀ x y : ℝ, y^2 = -2*p*x → (x^2/16 + y^2/12 = 1 → x = -2)) →
  p = 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_ellipse_focus_coincide_l3254_325418


namespace NUMINAMATH_CALUDE_sqrt_inequality_l3254_325449

theorem sqrt_inequality (a b c d : ℝ) 
  (h1 : a > b) (h2 : b > c) (h3 : c > d) (h4 : d > 0) 
  (h5 : a + d = b + c) : 
  Real.sqrt d + Real.sqrt a < Real.sqrt b + Real.sqrt c := by
sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l3254_325449


namespace NUMINAMATH_CALUDE_students_in_no_subjects_l3254_325486

theorem students_in_no_subjects (total : ℕ) (math chem bio : ℕ) (math_chem chem_bio math_bio : ℕ) (all_three : ℕ) : 
  total = 120 →
  math = 70 →
  chem = 50 →
  bio = 40 →
  math_chem = 30 →
  chem_bio = 20 →
  math_bio = 10 →
  all_three = 5 →
  total - (math + chem + bio - math_chem - chem_bio - math_bio + all_three) = 20 :=
by sorry

end NUMINAMATH_CALUDE_students_in_no_subjects_l3254_325486


namespace NUMINAMATH_CALUDE_inverse_function_intersection_l3254_325447

def f (x : ℝ) : ℝ := 3 * x^2 - 8

theorem inverse_function_intersection (x : ℝ) : 
  f x = x ↔ x = (1 + Real.sqrt 97) / 6 ∨ x = (1 - Real.sqrt 97) / 6 := by
  sorry

end NUMINAMATH_CALUDE_inverse_function_intersection_l3254_325447


namespace NUMINAMATH_CALUDE_age_difference_decade_difference_l3254_325427

/-- Given that the sum of ages of x and y is 10 years greater than the sum of ages of y and z,
    prove that x is 1 decade older than z. -/
theorem age_difference (x y z : ℕ) (h : x + y = y + z + 10) : x = z + 10 := by
  sorry

/-- A decade is defined as 10 years. -/
def decade : ℕ := 10

/-- Given that x is 10 years older than z, prove that x is 1 decade older than z. -/
theorem decade_difference (x z : ℕ) (h : x = z + 10) : x = z + decade := by
  sorry

end NUMINAMATH_CALUDE_age_difference_decade_difference_l3254_325427


namespace NUMINAMATH_CALUDE_max_k_value_l3254_325478

theorem max_k_value (x y k : ℝ) (hx : x > 0) (hy : y > 0) (hk : k > 0)
  (heq : 5 = k^2 * (x^2 / y^2 + y^2 / x^2) + k * (x / y + y / x)) :
  k ≤ (-4 + Real.sqrt 29) / 13 := by
  sorry

end NUMINAMATH_CALUDE_max_k_value_l3254_325478


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3254_325490

/-- An isosceles triangle with side lengths 3 and 6 has a perimeter of 15. -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ),
  a = 3 ∧ b = 6 ∧ c = 6 →  -- Two sides are 6, one side is 3
  a + b + c = 15 :=        -- The perimeter is 15
by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3254_325490


namespace NUMINAMATH_CALUDE_largest_partition_size_l3254_325401

/-- A partition of the positive integers into k subsets -/
def Partition (k : ℕ) := Fin k → Set ℕ

/-- The property that every integer ≥ 15 can be represented as a sum of two distinct elements from each subset -/
def HasPropertyForAll (P : Partition k) : Prop :=
  ∀ (n : ℕ) (i : Fin k), n ≥ 15 → ∃ (x y : ℕ), x ≠ y ∧ x ∈ P i ∧ y ∈ P i ∧ x + y = n

/-- The main theorem statement -/
theorem largest_partition_size :
  ∃ (k : ℕ), k > 0 ∧ 
    (∃ (P : Partition k), HasPropertyForAll P) ∧ 
    (∀ (m : ℕ), m > k → ¬∃ (Q : Partition m), HasPropertyForAll Q) ∧
    k = 3 := by
  sorry

end NUMINAMATH_CALUDE_largest_partition_size_l3254_325401


namespace NUMINAMATH_CALUDE_fahrenheit_to_celsius_l3254_325475

theorem fahrenheit_to_celsius (F C : ℝ) : F = 1.8 * C + 32 → F = 68 → C = 20 := by
  sorry

end NUMINAMATH_CALUDE_fahrenheit_to_celsius_l3254_325475


namespace NUMINAMATH_CALUDE_seed_calculation_total_seed_gallons_l3254_325460

/-- Calculates the total gallons of seed used for a football field given the specified conditions -/
theorem seed_calculation (field_area : ℝ) (seed_ratio : ℝ) (combined_gallons : ℝ) (combined_area : ℝ) : ℝ :=
  let total_parts := seed_ratio + 1
  let seed_fraction := seed_ratio / total_parts
  let seed_per_combined_area := seed_fraction * combined_gallons
  let field_coverage_factor := field_area / combined_area
  field_coverage_factor * seed_per_combined_area

/-- Proves that the total gallons of seed used for the entire football field is 768 gallons -/
theorem total_seed_gallons :
  seed_calculation 8000 4 240 2000 = 768 := by
  sorry

end NUMINAMATH_CALUDE_seed_calculation_total_seed_gallons_l3254_325460


namespace NUMINAMATH_CALUDE_factorable_implies_even_l3254_325416

-- Define the quadratic expression
def quadratic (a : ℤ) (x : ℝ) : ℝ := 21 * x^2 + a * x + 21

-- Define what it means for the quadratic to be factorable into linear binomials with integer coefficients
def is_factorable (a : ℤ) : Prop :=
  ∃ (m n p q : ℤ), 
    ∀ (x : ℝ), quadratic a x = (m * x + n) * (p * x + q)

-- The theorem to prove
theorem factorable_implies_even (a : ℤ) : 
  is_factorable a → ∃ k : ℤ, a = 2 * k :=
sorry

end NUMINAMATH_CALUDE_factorable_implies_even_l3254_325416


namespace NUMINAMATH_CALUDE_simple_interest_rate_percent_l3254_325480

/-- Given simple interest conditions, prove the rate percent -/
theorem simple_interest_rate_percent 
  (P : ℝ) (SI : ℝ) (T : ℝ) 
  (h_P : P = 900) 
  (h_SI : SI = 160) 
  (h_T : T = 4) 
  (h_formula : SI = (P * R * T) / 100) : 
  R = 400 / 90 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_percent_l3254_325480


namespace NUMINAMATH_CALUDE_total_worth_is_14000_l3254_325409

/-- The cost of the ring John gave to his fiancee -/
def ring_cost : ℕ := 4000

/-- The cost of the car John gave to his fiancee -/
def car_cost : ℕ := 2000

/-- The cost of the diamond brace John gave to his fiancee -/
def brace_cost : ℕ := 2 * ring_cost

/-- The total worth of the presents John gave to his fiancee -/
def total_worth : ℕ := ring_cost + car_cost + brace_cost

theorem total_worth_is_14000 : total_worth = 14000 := by
  sorry

end NUMINAMATH_CALUDE_total_worth_is_14000_l3254_325409


namespace NUMINAMATH_CALUDE_percentage_of_percentage_l3254_325471

theorem percentage_of_percentage (x : ℝ) : (10 / 100) * ((50 / 100) * 500) = 25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_percentage_l3254_325471


namespace NUMINAMATH_CALUDE_smaller_rectangle_area_l3254_325466

theorem smaller_rectangle_area (length width : ℝ) (h1 : length = 40) (h2 : width = 20) :
  (length / 2) * (width / 2) = 200 :=
by
  sorry

end NUMINAMATH_CALUDE_smaller_rectangle_area_l3254_325466


namespace NUMINAMATH_CALUDE_lollipops_remaining_l3254_325462

def raspberry_lollipops : ℕ := 57
def mint_lollipops : ℕ := 98
def blueberry_lollipops : ℕ := 13
def cola_lollipops : ℕ := 167
def num_friends : ℕ := 13

theorem lollipops_remaining :
  (raspberry_lollipops + mint_lollipops + blueberry_lollipops + cola_lollipops) % num_friends = 10 :=
by sorry

end NUMINAMATH_CALUDE_lollipops_remaining_l3254_325462


namespace NUMINAMATH_CALUDE_range_of_a_l3254_325445

def P (a : ℝ) : Set ℝ := {x | |x - a| < 4}
def Q : Set ℝ := {x | x^2 - 4*x + 3 < 0}

theorem range_of_a (a : ℝ) : 
  (∀ x, x ∈ Q → x ∈ P a) → -1 ≤ a ∧ a ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3254_325445


namespace NUMINAMATH_CALUDE_infinite_series_sum_l3254_325459

theorem infinite_series_sum : 
  let r := (1 : ℝ) / 1950
  let S := ∑' n, n * r^(n-1)
  S = 3802500 / 3802601 := by sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l3254_325459


namespace NUMINAMATH_CALUDE_march_birth_percentage_l3254_325496

def total_people : ℕ := 100
def march_births : ℕ := 8

theorem march_birth_percentage :
  (march_births : ℚ) / (total_people : ℚ) * 100 = 8 := by
  sorry

end NUMINAMATH_CALUDE_march_birth_percentage_l3254_325496


namespace NUMINAMATH_CALUDE_quiz_logic_l3254_325482

theorem quiz_logic (x y z w u v : ℝ) : 
  (x > y → z < w) → 
  (z > w → u < v) → 
  ¬((x < y → u < v) ∨ 
    (u < v → x < y) ∨ 
    (u > v → x > y) ∨ 
    (x > y → u > v)) := by
  sorry

end NUMINAMATH_CALUDE_quiz_logic_l3254_325482


namespace NUMINAMATH_CALUDE_special_triangle_sides_l3254_325485

/-- A triangle with an inscribed circle that passes through trisection points of a median -/
structure SpecialTriangle where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The circle passes through trisection points of a median -/
  trisects_median : Bool
  /-- The sides of the triangle -/
  side_a : ℝ
  side_b : ℝ
  side_c : ℝ

/-- The theorem about the special triangle -/
theorem special_triangle_sides (t : SpecialTriangle) 
  (h_radius : t.r = 3 * Real.sqrt 2)
  (h_trisects : t.trisects_median = true) :
  t.side_a = 5 * Real.sqrt 7 ∧ 
  t.side_b = 13 * Real.sqrt 7 ∧ 
  t.side_c = 10 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_sides_l3254_325485


namespace NUMINAMATH_CALUDE_trigonometric_identities_l3254_325417

theorem trigonometric_identities :
  let a := Real.sqrt 2 / 2 * (Real.cos (15 * π / 180) - Real.sin (15 * π / 180))
  let b := Real.cos (π / 12) ^ 2 - Real.sin (π / 12) ^ 2
  let c := Real.tan (22.5 * π / 180) / (1 - Real.tan (22.5 * π / 180) ^ 2)
  let d := Real.sin (15 * π / 180) * Real.cos (15 * π / 180)
  (a = 1/2) ∧ 
  (c = 1/2) ∧ 
  (b ≠ 1/2) ∧ 
  (d ≠ 1/2) := by
sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l3254_325417


namespace NUMINAMATH_CALUDE_two_numbers_difference_l3254_325498

theorem two_numbers_difference (x y : ℝ) : x + y = 55 ∧ x = 35 → x - y = 15 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l3254_325498


namespace NUMINAMATH_CALUDE_farmer_land_usage_l3254_325446

/-- Represents the ratio of land used for beans, wheat, and corn -/
def land_ratio : Fin 3 → ℕ
  | 0 => 5  -- beans
  | 1 => 2  -- wheat
  | 2 => 4  -- corn
  | _ => 0

/-- The amount of land used for corn in acres -/
def corn_land : ℕ := 376

/-- The total amount of land used by the farmer in acres -/
def total_land : ℕ := 1034

/-- Theorem stating that given the land ratio and corn land usage, 
    the total land used by the farmer is 1034 acres -/
theorem farmer_land_usage : 
  (land_ratio 2 : ℚ) / (land_ratio 0 + land_ratio 1 + land_ratio 2 : ℚ) * total_land = corn_land :=
by sorry

end NUMINAMATH_CALUDE_farmer_land_usage_l3254_325446


namespace NUMINAMATH_CALUDE_smallest_n_for_odd_ratio_l3254_325437

def concatenate_decimal_expansions (k : ℕ) : ℕ :=
  sorry

def X (n : ℕ) : ℕ := concatenate_decimal_expansions n

theorem smallest_n_for_odd_ratio :
  (∀ n : ℕ, n ≥ 2 → n < 5 → ¬(Odd (X n / 1024^n))) ∧
  (Odd (X 5 / 1024^5)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_odd_ratio_l3254_325437


namespace NUMINAMATH_CALUDE_ratio_of_squares_l3254_325468

theorem ratio_of_squares (x y : ℝ) (h : x^2 = 8*y^2 - 224) :
  x/y = Real.sqrt (8 - 224/y^2) :=
by sorry

end NUMINAMATH_CALUDE_ratio_of_squares_l3254_325468


namespace NUMINAMATH_CALUDE_employee_pay_l3254_325493

/-- Given two employees A and B with a total weekly pay of 450 and A's pay being 150% of B's,
    prove that B's weekly pay is 180. -/
theorem employee_pay (total_pay : ℝ) (a_pay : ℝ) (b_pay : ℝ) 
  (h1 : total_pay = 450)
  (h2 : a_pay = 1.5 * b_pay)
  (h3 : total_pay = a_pay + b_pay) :
  b_pay = 180 := by
  sorry

end NUMINAMATH_CALUDE_employee_pay_l3254_325493


namespace NUMINAMATH_CALUDE_bicycle_stand_stability_l3254_325403

/-- A triangle is a geometric shape with three sides and three angles. -/
structure Triangle where
  -- We don't need to define the specifics of a triangle for this problem

/-- A bicycle stand is a device that supports a bicycle. -/
structure BicycleStand where
  -- We don't need to define the specifics of a bicycle stand for this problem

/-- Stability is a property that allows an object to remain balanced and resist toppling. -/
def Stability : Prop := sorry

/-- A property that allows an object to stand firmly on the ground. -/
def AllowsToStandFirmly (prop : Prop) : Prop := sorry

theorem bicycle_stand_stability (t : Triangle) (s : BicycleStand) :
  AllowsToStandFirmly Stability := by sorry

end NUMINAMATH_CALUDE_bicycle_stand_stability_l3254_325403


namespace NUMINAMATH_CALUDE_odd_function_values_and_monotonicity_and_inequality_l3254_325400

noncomputable def f (a b : ℝ) : ℝ → ℝ := λ x ↦ (2^x + a) / (2^x + b)

theorem odd_function_values_and_monotonicity_and_inequality
  (h_odd : ∀ x, f a b (-x) = -(f a b x)) :
  (a = -1 ∧ b = 1) ∧
  (∀ x y, x < y → f a b x < f a b y) ∧
  (∀ x, f a b x + f a b (6 - x^2) ≤ 0 ↔ x ≤ -2 ∨ x ≥ 3) :=
sorry

end NUMINAMATH_CALUDE_odd_function_values_and_monotonicity_and_inequality_l3254_325400


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l3254_325492

/-- Proves that the annual interest rate is 0.1 given the initial investment,
    final amount, and time period. -/
theorem interest_rate_calculation (initial_investment : ℝ) (final_amount : ℝ) (years : ℕ) :
  initial_investment = 3000 →
  final_amount = 3630.0000000000005 →
  years = 2 →
  ∃ (r : ℝ), r = 0.1 ∧ final_amount = initial_investment * (1 + r) ^ years :=
by sorry


end NUMINAMATH_CALUDE_interest_rate_calculation_l3254_325492


namespace NUMINAMATH_CALUDE_game_show_probability_l3254_325483

theorem game_show_probability (total_doors : ℕ) (prize_doors : ℕ) 
  (opened_doors : ℕ) (opened_prize_doors : ℕ) :
  total_doors = 7 →
  prize_doors = 2 →
  opened_doors = 3 →
  opened_prize_doors = 1 →
  (total_doors - opened_doors - 1 : ℚ) / (total_doors - opened_doors) * 
  (prize_doors - opened_prize_doors) / (total_doors - opened_doors - 1) = 4 / 7 :=
by sorry

end NUMINAMATH_CALUDE_game_show_probability_l3254_325483


namespace NUMINAMATH_CALUDE_position_of_negative_three_l3254_325428

theorem position_of_negative_three : 
  ∀ (x : ℝ), (x = 1 - 4) → (x = -3) :=
by
  sorry

end NUMINAMATH_CALUDE_position_of_negative_three_l3254_325428


namespace NUMINAMATH_CALUDE_marble_difference_l3254_325440

/-- Given information about marbles owned by Amanda, Katrina, and Mabel -/
theorem marble_difference (amanda katrina mabel : ℕ) 
  (h1 : amanda + 12 = 2 * katrina)
  (h2 : mabel = 5 * katrina)
  (h3 : mabel = 85) :
  mabel - amanda = 63 := by
  sorry

end NUMINAMATH_CALUDE_marble_difference_l3254_325440


namespace NUMINAMATH_CALUDE_smallest_number_in_set_l3254_325479

def number_set : Set ℤ := {0, -2, 1, 5}

theorem smallest_number_in_set : 
  ∃ x ∈ number_set, ∀ y ∈ number_set, x ≤ y ∧ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_in_set_l3254_325479


namespace NUMINAMATH_CALUDE_cup_production_decrease_rate_l3254_325442

theorem cup_production_decrease_rate 
  (initial_production : ℝ) 
  (final_production : ℝ) 
  (months : ℕ) 
  (h1 : initial_production = 1.6) 
  (h2 : final_production = 0.9) 
  (h3 : months = 2) :
  ∃ (rate : ℝ), 
    rate = 0.25 ∧ 
    final_production = initial_production * (1 - rate) ^ months :=
by sorry

end NUMINAMATH_CALUDE_cup_production_decrease_rate_l3254_325442


namespace NUMINAMATH_CALUDE_intersection_constraint_l3254_325453

theorem intersection_constraint (a : ℝ) : 
  let A : Set ℝ := {-1, 1, 3}
  let B : Set ℝ := {a + 2, a^2 + 4}
  (A ∩ B = {3}) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_constraint_l3254_325453


namespace NUMINAMATH_CALUDE_production_cost_reduction_l3254_325408

/-- Represents the equation for production cost reduction over two years -/
theorem production_cost_reduction (initial_cost target_cost : ℝ) (x : ℝ) :
  initial_cost = 200000 →
  target_cost = 150000 →
  initial_cost * (1 - x)^2 = target_cost :=
by
  sorry

end NUMINAMATH_CALUDE_production_cost_reduction_l3254_325408


namespace NUMINAMATH_CALUDE_f_nonnegative_iff_a_in_range_l3254_325413

noncomputable def f (a x : ℝ) : ℝ := Real.exp x - (1/2) * (x - a)^2 + 4

theorem f_nonnegative_iff_a_in_range (a : ℝ) :
  (∀ x ≥ 0, f a x ≥ 0) ↔ a ∈ Set.Icc (Real.log 4 - 4) (Real.sqrt 10) :=
sorry

end NUMINAMATH_CALUDE_f_nonnegative_iff_a_in_range_l3254_325413


namespace NUMINAMATH_CALUDE_star_two_three_l3254_325434

-- Define the star operation
def star (a b : ℝ) : ℝ := a^2 * b^2 - a + 2

-- Theorem statement
theorem star_two_three : star 2 3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_star_two_three_l3254_325434


namespace NUMINAMATH_CALUDE_odd_function_property_l3254_325406

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- State the theorem
theorem odd_function_property (f : ℝ → ℝ) (h1 : OddFunction f) (h2 : f (-3) = -2) :
  f 3 + f 0 = 2 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_property_l3254_325406


namespace NUMINAMATH_CALUDE_proportional_segments_l3254_325497

theorem proportional_segments (a b c d : ℝ) : 
  a = 3 ∧ d = 4 ∧ c = 6 ∧ (a / b = c / d) → b = 2 := by
  sorry

end NUMINAMATH_CALUDE_proportional_segments_l3254_325497


namespace NUMINAMATH_CALUDE_polynomial_remainder_l3254_325489

def p (x : ℝ) : ℝ := 4*x^5 - 3*x^4 + 5*x^3 - 7*x^2 + 3*x - 10

theorem polynomial_remainder : p 2 = 88 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l3254_325489


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3254_325473

theorem right_triangle_hypotenuse (a b c : ℝ) :
  -- Right triangle condition
  c^2 = a^2 + b^2 →
  -- Area condition
  (1/2) * a * b = 48 →
  -- Geometric mean condition
  (a * b)^(1/2) = 8 →
  -- Conclusion: hypotenuse length
  c = 4 * (13 : ℝ)^(1/2) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3254_325473


namespace NUMINAMATH_CALUDE_tower_heights_theorem_l3254_325484

/-- Represents the dimensions of a brick -/
structure BrickDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents the possible height contributions of a brick -/
def HeightContributions : List ℕ := [4, 10, 19]

/-- The total number of bricks -/
def TotalBricks : ℕ := 94

/-- Calculates the number of distinct tower heights -/
def distinctTowerHeights (brickDims : BrickDimensions) (contributions : List ℕ) (totalBricks : ℕ) : ℕ :=
  sorry

theorem tower_heights_theorem (brickDims : BrickDimensions) 
    (h1 : brickDims.length = 4 ∧ brickDims.width = 10 ∧ brickDims.height = 19) :
    distinctTowerHeights brickDims HeightContributions TotalBricks = 465 := by
  sorry

end NUMINAMATH_CALUDE_tower_heights_theorem_l3254_325484


namespace NUMINAMATH_CALUDE_abs_geq_ax_implies_abs_a_leq_one_l3254_325469

theorem abs_geq_ax_implies_abs_a_leq_one (a : ℝ) :
  (∀ x : ℝ, |x| ≥ a * x) → |a| ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_abs_geq_ax_implies_abs_a_leq_one_l3254_325469


namespace NUMINAMATH_CALUDE_divisibility_problem_l3254_325419

theorem divisibility_problem (a b c d : ℕ+) 
  (h1 : Nat.gcd a b = 18)
  (h2 : Nat.gcd b c = 45)
  (h3 : Nat.gcd c d = 75)
  (h4 : 80 < Nat.gcd d a)
  (h5 : Nat.gcd d a < 120) :
  7 ∣ a.val := by
  sorry

end NUMINAMATH_CALUDE_divisibility_problem_l3254_325419


namespace NUMINAMATH_CALUDE_reciprocal_problem_l3254_325450

theorem reciprocal_problem (x : ℚ) : (10 : ℚ) / 3 = 1 / x + 1 → x = 3 / 7 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_problem_l3254_325450


namespace NUMINAMATH_CALUDE_motorcyclist_meets_cyclist1_l3254_325443

/-- Represents the time in minutes for two entities to meet or overtake each other. -/
structure MeetingTime where
  time : ℝ
  time_positive : time > 0

/-- Represents an entity moving on the circular highway. -/
structure Entity where
  speed : ℝ
  direction : Bool  -- True for one direction, False for the opposite

/-- The circular highway setup with four entities. -/
structure CircularHighway where
  runner : Entity
  cyclist1 : Entity
  cyclist2 : Entity
  motorcyclist : Entity
  runner_cyclist2_meeting : MeetingTime
  runner_cyclist1_overtake : MeetingTime
  motorcyclist_cyclist2_overtake : MeetingTime
  highway_length : ℝ
  highway_length_positive : highway_length > 0

  runner_direction : runner.direction = true
  cyclist1_direction : cyclist1.direction = true
  cyclist2_direction : cyclist2.direction = false
  motorcyclist_direction : motorcyclist.direction = false

  runner_cyclist2_meeting_time : runner_cyclist2_meeting.time = 12
  runner_cyclist1_overtake_time : runner_cyclist1_overtake.time = 20
  motorcyclist_cyclist2_overtake_time : motorcyclist_cyclist2_overtake.time = 5

/-- The theorem stating that the motorcyclist meets the first cyclist every 3 minutes. -/
theorem motorcyclist_meets_cyclist1 (h : CircularHighway) :
  ∃ (t : MeetingTime), t.time = 3 ∧
    h.highway_length / t.time = h.motorcyclist.speed + h.cyclist1.speed :=
sorry

end NUMINAMATH_CALUDE_motorcyclist_meets_cyclist1_l3254_325443


namespace NUMINAMATH_CALUDE_fifth_term_value_l3254_325463

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem fifth_term_value (a : ℕ → ℝ) :
  geometric_sequence a →
  a 3 = 6 →
  a 3 + a 5 + a 7 = 78 →
  a 5 = 18 := by
sorry

end NUMINAMATH_CALUDE_fifth_term_value_l3254_325463


namespace NUMINAMATH_CALUDE_f_intersects_x_axis_iff_l3254_325435

/-- A function that represents (k-3)x^2+2x+1 --/
def f (k : ℝ) (x : ℝ) : ℝ := (k - 3) * x^2 + 2 * x + 1

/-- Predicate to check if a function intersects the x-axis --/
def intersects_x_axis (g : ℝ → ℝ) : Prop :=
  ∃ x, g x = 0

/-- Theorem stating that f intersects the x-axis iff k ≤ 4 --/
theorem f_intersects_x_axis_iff (k : ℝ) :
  intersects_x_axis (f k) ↔ k ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_f_intersects_x_axis_iff_l3254_325435


namespace NUMINAMATH_CALUDE_residual_plot_vertical_axis_l3254_325436

/-- A residual plot used in residual analysis. -/
structure ResidualPlot where
  vertical_axis : Set ℝ
  horizontal_axis : Set ℝ

/-- The definition of a residual in the context of residual analysis. -/
def Residual : Type := ℝ

/-- Theorem stating that the vertical axis of a residual plot represents the residuals. -/
theorem residual_plot_vertical_axis (plot : ResidualPlot) :
  plot.vertical_axis = Set.range (λ r : Residual => r) :=
sorry

end NUMINAMATH_CALUDE_residual_plot_vertical_axis_l3254_325436


namespace NUMINAMATH_CALUDE_angle_complement_l3254_325491

theorem angle_complement (A : ℝ) : 
  A = 45 → 90 - A = 45 := by
sorry

end NUMINAMATH_CALUDE_angle_complement_l3254_325491


namespace NUMINAMATH_CALUDE_homework_time_theorem_l3254_325405

/-- The total time left for homework completion --/
def total_time (jacob_time greg_time patrick_time : ℕ) : ℕ :=
  jacob_time + greg_time + patrick_time

/-- Theorem stating the total time left for homework completion --/
theorem homework_time_theorem (jacob_time greg_time patrick_time : ℕ) 
  (h1 : jacob_time = 18)
  (h2 : greg_time = jacob_time - 6)
  (h3 : patrick_time = 2 * greg_time - 4) :
  total_time jacob_time greg_time patrick_time = 50 := by
  sorry

end NUMINAMATH_CALUDE_homework_time_theorem_l3254_325405


namespace NUMINAMATH_CALUDE_inscribed_octagon_area_l3254_325477

theorem inscribed_octagon_area (circle_area : ℝ) (octagon_area : ℝ) : 
  circle_area = 64 * Real.pi →
  octagon_area = 8 * (1 / 2 * (circle_area / Real.pi) * Real.sin (Real.pi / 4)) →
  octagon_area = 128 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_octagon_area_l3254_325477


namespace NUMINAMATH_CALUDE_permutation_combination_sum_l3254_325476

/-- Given that A(n,m) = 272 and C(n,m) = 136, prove that m + n = 19 -/
theorem permutation_combination_sum (m n : ℕ) : 
  (m.factorial * (n.choose m) = 272) → (n.choose m = 136) → m + n = 19 := by
  sorry

end NUMINAMATH_CALUDE_permutation_combination_sum_l3254_325476


namespace NUMINAMATH_CALUDE_oatmeal_cookies_count_l3254_325424

/-- Represents the number of cookies in each baggie -/
def cookies_per_baggie : ℕ := 3

/-- Represents the number of chocolate chip cookies Maria had -/
def chocolate_chip_cookies : ℕ := 2

/-- Represents the number of baggies Maria could make -/
def total_baggies : ℕ := 6

/-- Theorem stating the number of oatmeal cookies Maria had -/
theorem oatmeal_cookies_count : 
  (total_baggies * cookies_per_baggie) - chocolate_chip_cookies = 16 := by
  sorry

end NUMINAMATH_CALUDE_oatmeal_cookies_count_l3254_325424


namespace NUMINAMATH_CALUDE_length_breadth_difference_is_ten_l3254_325454

/-- Represents a rectangular plot with given dimensions and fencing costs. -/
structure RectangularPlot where
  length : ℝ
  breadth : ℝ
  fenceCostPerMeter : ℝ
  totalFenceCost : ℝ

/-- Calculates the difference between length and breadth of the plot. -/
def lengthBreadthDifference (plot : RectangularPlot) : ℝ :=
  plot.length - plot.breadth

/-- Theorem stating that for a rectangular plot with length 55 meters,
    where the cost of fencing at Rs. 26.50 per meter totals Rs. 5300,
    the length is 10 meters more than the breadth. -/
theorem length_breadth_difference_is_ten
  (plot : RectangularPlot)
  (h1 : plot.length = 55)
  (h2 : plot.fenceCostPerMeter = 26.5)
  (h3 : plot.totalFenceCost = 5300)
  (h4 : plot.totalFenceCost = plot.fenceCostPerMeter * (2 * (plot.length + plot.breadth))) :
  lengthBreadthDifference plot = 10 := by
  sorry

#eval lengthBreadthDifference { length := 55, breadth := 45, fenceCostPerMeter := 26.5, totalFenceCost := 5300 }

end NUMINAMATH_CALUDE_length_breadth_difference_is_ten_l3254_325454


namespace NUMINAMATH_CALUDE_quadratic_point_condition_l3254_325487

/-- The quadratic function y = -(x-1)² + n -/
def f (x n : ℝ) : ℝ := -(x - 1)^2 + n

theorem quadratic_point_condition (m y₁ y₂ n : ℝ) :
  f m n = y₁ →
  f (m + 1) n = y₂ →
  y₁ > y₂ →
  m > 1/2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_point_condition_l3254_325487


namespace NUMINAMATH_CALUDE_krishans_money_l3254_325423

/-- Proves that Krishan has Rs. 4335 given the conditions of the problem -/
theorem krishans_money (ram gopal krishan : ℕ) : 
  (ram : ℚ) / gopal = 7 / 17 →
  (gopal : ℚ) / krishan = 7 / 17 →
  ram = 735 →
  krishan = 4335 := by
  sorry

end NUMINAMATH_CALUDE_krishans_money_l3254_325423


namespace NUMINAMATH_CALUDE_two_digit_multiplication_l3254_325404

theorem two_digit_multiplication (a b c d : ℕ) :
  (a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10) →
  ((b = d ∧ a + c = 10) ∨ (a = c ∧ b + d = 10) ∨ (c = d ∧ a + b = 10)) →
  (10 * a + b) * (10 * c + d) = 
    (if b = d ∧ a + c = 10 then 100 * (a^2 + a) + b * d
     else if a = c ∧ b + d = 10 then 100 * a * c + 100 * b + b^2
     else 100 * a * c + 100 * c + b * c) :=
by sorry

end NUMINAMATH_CALUDE_two_digit_multiplication_l3254_325404


namespace NUMINAMATH_CALUDE_decreasing_linear_function_k_range_l3254_325461

-- Define the linear function
def f (k : ℝ) (x : ℝ) : ℝ := (2*k - 4)*x - 1

-- State the theorem
theorem decreasing_linear_function_k_range (k : ℝ) :
  (∀ x y : ℝ, x < y → f k x > f k y) → k < 2 := by sorry

end NUMINAMATH_CALUDE_decreasing_linear_function_k_range_l3254_325461


namespace NUMINAMATH_CALUDE_walking_delay_l3254_325441

/-- Proves that walking at 3/4 of normal speed results in an 8-minute delay -/
theorem walking_delay (normal_speed : ℝ) (distance : ℝ) : 
  normal_speed > 0 → distance > 0 → 
  (distance / normal_speed = 24) → 
  (distance / (3/4 * normal_speed) - 24 = 8) := by
  sorry

end NUMINAMATH_CALUDE_walking_delay_l3254_325441


namespace NUMINAMATH_CALUDE_matthew_cake_division_l3254_325415

/-- Given that Matthew has 30 cakes and 2 friends, prove that each friend receives 15 cakes when the cakes are divided equally. -/
theorem matthew_cake_division (total_cakes : ℕ) (num_friends : ℕ) (cakes_per_friend : ℕ) :
  total_cakes = 30 →
  num_friends = 2 →
  cakes_per_friend = total_cakes / num_friends →
  cakes_per_friend = 15 := by
  sorry

end NUMINAMATH_CALUDE_matthew_cake_division_l3254_325415


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3254_325433

theorem inequality_solution_set (x : ℝ) :
  (-2 < (x^2 - 16*x + 15) / (x^2 - 4*x + 5) ∧ (x^2 - 16*x + 15) / (x^2 - 4*x + 5) < 2) ↔
  (x < -13.041 ∨ x > -0.959) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3254_325433


namespace NUMINAMATH_CALUDE_fourth_term_of_sequence_l3254_325472

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem fourth_term_of_sequence (a : ℕ → ℝ) :
  geometric_sequence a →
  a 1 = (16 : ℝ) ^ (1/4) →
  a 2 = (16 : ℝ) ^ (1/6) →
  a 3 = (16 : ℝ) ^ (1/8) →
  a 4 = (2 : ℝ) ^ (1/3) :=
sorry

end NUMINAMATH_CALUDE_fourth_term_of_sequence_l3254_325472


namespace NUMINAMATH_CALUDE_temperature_difference_l3254_325465

theorem temperature_difference 
  (highest_temp lowest_temp : ℝ) 
  (h_highest : highest_temp = 27) 
  (h_lowest : lowest_temp = 17) : 
  highest_temp - lowest_temp = 10 := by
sorry

end NUMINAMATH_CALUDE_temperature_difference_l3254_325465


namespace NUMINAMATH_CALUDE_alison_money_l3254_325412

def money_problem (kent_original brittany brooke kent alison : ℚ) : Prop :=
  let kent_after_lending := kent_original - 200
  alison = brittany / 2 ∧
  brittany = 4 * brooke ∧
  brooke = 2 * kent ∧
  kent = kent_after_lending ∧
  kent_original = 1000

theorem alison_money :
  ∀ kent_original brittany brooke kent alison,
    money_problem kent_original brittany brooke kent alison →
    alison = 3200 := by
  sorry

end NUMINAMATH_CALUDE_alison_money_l3254_325412


namespace NUMINAMATH_CALUDE_nina_tomato_harvest_l3254_325488

/-- Represents the dimensions of a rectangular garden -/
structure GardenDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular garden -/
def gardenArea (d : GardenDimensions) : ℝ :=
  d.length * d.width

/-- Represents the planting and yield information for tomatoes -/
structure TomatoInfo where
  plantsPerSquareFoot : ℝ
  tomatoesPerPlant : ℝ

/-- Calculates the total number of tomatoes expected from a garden -/
def expectedTomatoes (d : GardenDimensions) (t : TomatoInfo) : ℝ :=
  gardenArea d * t.plantsPerSquareFoot * t.tomatoesPerPlant

/-- Theorem stating the expected tomato harvest for Nina's garden -/
theorem nina_tomato_harvest :
  let garden := GardenDimensions.mk 10 20
  let tomato := TomatoInfo.mk 5 10
  expectedTomatoes garden tomato = 10000 := by
  sorry


end NUMINAMATH_CALUDE_nina_tomato_harvest_l3254_325488


namespace NUMINAMATH_CALUDE_inequality_not_always_holds_l3254_325457

theorem inequality_not_always_holds (a b : ℝ) (h : a > b) :
  ¬ ∀ c : ℝ, a * c > b * c := by
  sorry

end NUMINAMATH_CALUDE_inequality_not_always_holds_l3254_325457


namespace NUMINAMATH_CALUDE_millet_majority_on_friday_l3254_325426

/-- Represents the amount of millet in the feeder on a given day -/
def millet_amount (day : ℕ) : ℚ :=
  0.5 * (1 - (0.7 ^ day))

/-- Represents the total amount of seeds in the feeder on a given day -/
def total_seeds (day : ℕ) : ℚ :=
  0.5 * day

/-- Theorem stating that on the 5th day, more than two-thirds of the seeds are millet -/
theorem millet_majority_on_friday :
  (millet_amount 5) / (total_seeds 5) > 2/3 ∧
  ∀ d : ℕ, d < 5 → (millet_amount d) / (total_seeds d) ≤ 2/3 :=
sorry

end NUMINAMATH_CALUDE_millet_majority_on_friday_l3254_325426


namespace NUMINAMATH_CALUDE_max_product_l3254_325444

def digits : List Nat := [1, 3, 5, 8, 9]

def is_valid_combination (a b c d e : Nat) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧ e ∈ digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e

def product (a b c d e : Nat) : Nat :=
  (100 * a + 10 * b + c) * (10 * d + e)

theorem max_product :
  ∀ a b c d e : Nat,
    is_valid_combination a b c d e →
    product a b c d e ≤ product 8 5 1 9 3 :=
by sorry

end NUMINAMATH_CALUDE_max_product_l3254_325444
