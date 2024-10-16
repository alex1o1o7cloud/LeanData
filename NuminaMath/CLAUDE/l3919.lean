import Mathlib

namespace NUMINAMATH_CALUDE_min_horizontal_distance_l3919_391907

/-- The parabola function -/
def f (x : ℝ) : ℝ := x^2 - x - 6

/-- Theorem stating the minimum horizontal distance between points P and Q -/
theorem min_horizontal_distance :
  ∃ (xp xq : ℝ),
    f xp = 6 ∧
    f xq = -6 ∧
    ∀ (yp yq : ℝ),
      f yp = 6 → f yq = -6 →
      |xp - xq| ≤ |yp - yq| ∧
      |xp - xq| = 4 :=
sorry

end NUMINAMATH_CALUDE_min_horizontal_distance_l3919_391907


namespace NUMINAMATH_CALUDE_rohans_savings_l3919_391911

/-- Rohan's monthly budget and savings calculation -/
theorem rohans_savings (salary : ℕ) (food_percent house_percent entertainment_percent conveyance_percent : ℚ) : 
  salary = 5000 →
  food_percent = 40 / 100 →
  house_percent = 20 / 100 →
  entertainment_percent = 10 / 100 →
  conveyance_percent = 10 / 100 →
  salary - (salary * (food_percent + house_percent + entertainment_percent + conveyance_percent)).floor = 1000 := by
  sorry

#check rohans_savings

end NUMINAMATH_CALUDE_rohans_savings_l3919_391911


namespace NUMINAMATH_CALUDE_lcm_1584_1188_l3919_391936

theorem lcm_1584_1188 : Nat.lcm 1584 1188 = 4752 := by
  sorry

end NUMINAMATH_CALUDE_lcm_1584_1188_l3919_391936


namespace NUMINAMATH_CALUDE_ac_unit_final_price_l3919_391965

/-- Calculates the final price of an air-conditioning unit after multiple price changes -/
def finalPrice (initialPrice : ℝ) (christmasDiscount additionalDiscount priceIncrease1 priceIncrease2 seasonalDiscount : ℝ) : ℝ :=
  let priceAfterChristmas := initialPrice * (1 - christmasDiscount)
  let priceAfterAdditional := priceAfterChristmas * (1 - additionalDiscount)
  let priceAfterIncrease1 := priceAfterAdditional * (1 + priceIncrease1)
  let priceAfterIncrease2 := priceAfterIncrease1 * (1 + priceIncrease2)
  priceAfterIncrease2 * (1 - seasonalDiscount)

/-- Theorem stating the final price of the air-conditioning unit -/
theorem ac_unit_final_price :
  finalPrice 470 0.16 0.07 0.12 0.08 0.10 = 399.71 := by
  sorry


end NUMINAMATH_CALUDE_ac_unit_final_price_l3919_391965


namespace NUMINAMATH_CALUDE_benny_total_spent_l3919_391928

def soft_drink_quantity : ℕ := 2
def soft_drink_price : ℕ := 4
def candy_bar_quantity : ℕ := 5
def candy_bar_price : ℕ := 4

theorem benny_total_spent :
  soft_drink_quantity * soft_drink_price + candy_bar_quantity * candy_bar_price = 28 := by
  sorry

end NUMINAMATH_CALUDE_benny_total_spent_l3919_391928


namespace NUMINAMATH_CALUDE_max_rooms_needed_l3919_391950

/-- Represents a group of football fans -/
structure FanGroup where
  team : Fin 3
  gender : Bool
  count : Nat

/-- The maximum number of fans that can be accommodated in one room -/
def max_fans_per_room : Nat := 3

/-- The total number of fans -/
def total_fans : Nat := 100

/-- Calculate the number of rooms needed for a group of fans -/
def rooms_needed (group : FanGroup) : Nat :=
  (group.count + max_fans_per_room - 1) / max_fans_per_room

/-- The main theorem stating the maximum number of rooms needed -/
theorem max_rooms_needed (fans : List FanGroup) 
  (h1 : fans.length = 6)
  (h2 : fans.all (λ g ↦ g.count > 0))
  (h3 : (fans.map FanGroup.count).sum = total_fans) :
  (fans.map rooms_needed).sum ≤ 37 := by
  sorry

end NUMINAMATH_CALUDE_max_rooms_needed_l3919_391950


namespace NUMINAMATH_CALUDE_smallest_visible_sum_l3919_391909

/-- Represents a die with 6 faces -/
structure Die :=
  (faces : Fin 6 → ℕ)
  (opposite_sum : ∀ i : Fin 3, faces i + faces (i + 3) = 7)

/-- Represents a 4x4x4 cube made of dice -/
def LargeCube := Fin 4 → Fin 4 → Fin 4 → Die

/-- The sum of visible face values on the large cube -/
def visible_sum (cube : LargeCube) : ℕ :=
  sorry

theorem smallest_visible_sum (cube : LargeCube) :
  visible_sum cube ≥ 144 :=
sorry

end NUMINAMATH_CALUDE_smallest_visible_sum_l3919_391909


namespace NUMINAMATH_CALUDE_part_one_part_two_l3919_391916

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | (x - (a - 1)) * (x - (a + 1)) < 0}
def B : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

-- Part 1: Prove that when a = 2, A ∪ B = B
theorem part_one : A 2 ∪ B = B := by sorry

-- Part 2: Prove that x ∈ A ⇔ x ∈ B holds if and only if 0 ≤ a ≤ 2
theorem part_two : (∀ x, x ∈ A a ↔ x ∈ B) ↔ 0 ≤ a ∧ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3919_391916


namespace NUMINAMATH_CALUDE_optimal_price_reduction_l3919_391968

/-- Represents the price reduction of thermal shirts -/
def price_reduction : ℝ := 20

/-- Initial average daily sales -/
def initial_sales : ℝ := 20

/-- Initial profit per piece -/
def initial_profit_per_piece : ℝ := 40

/-- Sales increase per dollar of price reduction -/
def sales_increase_rate : ℝ := 2

/-- Target daily profit -/
def target_profit : ℝ := 1200

/-- New daily sales after price reduction -/
def new_sales (x : ℝ) : ℝ := initial_sales + sales_increase_rate * x

/-- New profit per piece after price reduction -/
def new_profit_per_piece (x : ℝ) : ℝ := initial_profit_per_piece - x

/-- Daily profit function -/
def daily_profit (x : ℝ) : ℝ := new_sales x * new_profit_per_piece x

theorem optimal_price_reduction :
  daily_profit price_reduction = target_profit ∧
  ∀ y, y ≠ price_reduction → daily_profit y ≤ daily_profit price_reduction :=
by sorry

end NUMINAMATH_CALUDE_optimal_price_reduction_l3919_391968


namespace NUMINAMATH_CALUDE_parallel_planes_sufficient_not_necessary_l3919_391923

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation for planes and lines
variable (parallelPlanes : Plane → Plane → Prop)
variable (parallelLinePlane : Line → Plane → Prop)

-- Define a predicate for a line being in a plane
variable (lineInPlane : Line → Plane → Prop)

variable (α β : Plane)
variable (l : Line)

-- State the theorem
theorem parallel_planes_sufficient_not_necessary :
  (α ≠ β) →
  (lineInPlane l α) →
  (parallelPlanes α β → parallelLinePlane l β) ∧
  ∃ γ : Plane, (parallelLinePlane l γ ∧ ¬parallelPlanes α γ) :=
by sorry

end NUMINAMATH_CALUDE_parallel_planes_sufficient_not_necessary_l3919_391923


namespace NUMINAMATH_CALUDE_sandwich_combinations_l3919_391905

theorem sandwich_combinations :
  let meat_types : ℕ := 12
  let cheese_types : ℕ := 12
  let spread_types : ℕ := 5
  let meat_selection : ℕ := meat_types
  let cheese_selection : ℕ := cheese_types.choose 2
  let spread_selection : ℕ := spread_types
  meat_selection * cheese_selection * spread_selection = 3960 :=
by sorry

end NUMINAMATH_CALUDE_sandwich_combinations_l3919_391905


namespace NUMINAMATH_CALUDE_two_number_problem_l3919_391966

theorem two_number_problem (x y : ℚ) : 
  x + y = 40 →
  3 * y - 4 * x = 10 →
  |y - x| = 60 / 7 := by
sorry

end NUMINAMATH_CALUDE_two_number_problem_l3919_391966


namespace NUMINAMATH_CALUDE_coat_price_proof_l3919_391927

/-- Proves that the original price of a coat is $500 given the specified conditions -/
theorem coat_price_proof (P : ℝ) 
  (h1 : 0.70 * P = 350) : P = 500 := by
  sorry

end NUMINAMATH_CALUDE_coat_price_proof_l3919_391927


namespace NUMINAMATH_CALUDE_friend_balloon_count_l3919_391949

theorem friend_balloon_count (my_balloons : ℕ) (difference : ℕ) : my_balloons = 7 → difference = 2 → my_balloons - difference = 5 := by
  sorry

end NUMINAMATH_CALUDE_friend_balloon_count_l3919_391949


namespace NUMINAMATH_CALUDE_steves_emails_l3919_391920

theorem steves_emails (initial_emails : ℕ) : 
  (initial_emails / 2 : ℚ) * (1 - 0.4) = 120 → initial_emails = 400 :=
by
  sorry

end NUMINAMATH_CALUDE_steves_emails_l3919_391920


namespace NUMINAMATH_CALUDE_k_range_for_single_extremum_l3919_391976

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := Real.exp x / x + k * (Real.log x - x)

theorem k_range_for_single_extremum (k : ℝ) :
  (∀ x > 0, x ≠ 1 → (deriv (f k)) x ≠ 0) →
  (deriv (f k)) 1 = 0 →
  k ≤ Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_k_range_for_single_extremum_l3919_391976


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l3919_391986

theorem negation_of_existence (P : ℝ → Prop) :
  (¬∃ x, P x) ↔ (∀ x, ¬P x) :=
by sorry

theorem negation_of_proposition :
  (¬∃ x : ℝ, Real.exp x - x - 1 ≤ 0) ↔ (∀ x : ℝ, Real.exp x - x - 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l3919_391986


namespace NUMINAMATH_CALUDE_fahrenheit_to_celsius_l3919_391979

theorem fahrenheit_to_celsius (C F : ℝ) : C = (5 / 9) * (F - 32) → C = 40 → F = 104 := by
  sorry

end NUMINAMATH_CALUDE_fahrenheit_to_celsius_l3919_391979


namespace NUMINAMATH_CALUDE_intersection_when_a_is_two_complement_union_when_a_is_two_union_equals_B_iff_l3919_391952

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < 2 * a + 3}
def B : Set ℝ := {x | -2 ≤ x ∧ x ≤ 4}

-- Theorem 1: When a = 2, A ∩ B = {x | 1 < x ≤ 4}
theorem intersection_when_a_is_two :
  A 2 ∩ B = {x : ℝ | 1 < x ∧ x ≤ 4} := by sorry

-- Theorem 2: When a = 2, (Uᶜ A) ∪ (Uᶜ B) = {x | x ≤ 1 or x > 4}
theorem complement_union_when_a_is_two :
  (Set.univ \ A 2) ∪ (Set.univ \ B) = {x : ℝ | x ≤ 1 ∨ x > 4} := by sorry

-- Theorem 3: A ∪ B = B if and only if a ≤ -4 or -1 ≤ a ≤ 1/2
theorem union_equals_B_iff :
  ∀ a : ℝ, A a ∪ B = B ↔ a ≤ -4 ∨ (-1 ≤ a ∧ a ≤ 1/2) := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_two_complement_union_when_a_is_two_union_equals_B_iff_l3919_391952


namespace NUMINAMATH_CALUDE_mixed_sample_2_prob_is_0_19_expected_tests_plan3_is_2_3756_l3919_391971

-- Define the probability of an animal having the disease
def disease_prob : ℝ := 0.1

-- Define the probability of a mixed sample of 2 animals testing positive
def mixed_sample_2_prob : ℝ := 1 - (1 - disease_prob)^2

-- Define the probability of a mixed sample of 4 animals testing negative
def mixed_sample_4_neg_prob : ℝ := (1 - disease_prob)^4

-- Define the expected number of tests for Plan 3 (mixing all 4 samples)
def expected_tests_plan3 : ℝ := 1 * mixed_sample_4_neg_prob + 5 * (1 - mixed_sample_4_neg_prob)

-- Theorem 1: Probability of positive test for mixed sample of 2 animals
theorem mixed_sample_2_prob_is_0_19 : mixed_sample_2_prob = 0.19 := by sorry

-- Theorem 2: Expected number of tests for Plan 3
theorem expected_tests_plan3_is_2_3756 : expected_tests_plan3 = 2.3756 := by sorry

end NUMINAMATH_CALUDE_mixed_sample_2_prob_is_0_19_expected_tests_plan3_is_2_3756_l3919_391971


namespace NUMINAMATH_CALUDE_birds_landed_l3919_391942

/-- Given an initial number of birds on a fence and a final number of birds on the fence,
    this theorem proves that the number of birds that landed is equal to
    the difference between the final and initial numbers. -/
theorem birds_landed (initial final : ℕ) (h : initial ≤ final) :
  final - initial = final - initial :=
by sorry

end NUMINAMATH_CALUDE_birds_landed_l3919_391942


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l3919_391924

theorem quadratic_roots_property (x₁ x₂ c : ℝ) : 
  (x₁^2 + x₁ + c = 0) →
  (x₂^2 + x₂ + c = 0) →
  (x₁^2 * x₂ + x₂^2 * x₁ = 3) →
  c = -3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l3919_391924


namespace NUMINAMATH_CALUDE_inequality_proof_l3919_391931

theorem inequality_proof (x : ℝ) (h1 : x ≥ 5) (h2 : x ≠ 2) :
  (x - 5) / (x^2 + x + 3) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3919_391931


namespace NUMINAMATH_CALUDE_class_composition_l3919_391900

theorem class_composition (total : ℕ) (girls : ℕ) (boys : ℕ) 
  (h1 : total = 70) 
  (h2 : 4 * boys = 3 * girls) 
  (h3 : girls + boys = total) : 
  girls = 40 ∧ boys = 30 := by
  sorry

end NUMINAMATH_CALUDE_class_composition_l3919_391900


namespace NUMINAMATH_CALUDE_negation_equivalence_l3919_391972

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x^4 - x^3 + x^2 + 5 ≤ 0) ↔ (∃ x₀ : ℝ, x₀^4 - x₀^3 + x₀^2 + 5 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3919_391972


namespace NUMINAMATH_CALUDE_number_of_cows_l3919_391947

/-- The number of cows in a field with a total of 200 animals, 56 sheep, and 104 goats. -/
theorem number_of_cows (total : ℕ) (sheep : ℕ) (goats : ℕ) (h1 : total = 200) (h2 : sheep = 56) (h3 : goats = 104) :
  total - sheep - goats = 40 := by
  sorry

end NUMINAMATH_CALUDE_number_of_cows_l3919_391947


namespace NUMINAMATH_CALUDE_infinite_series_sum_l3919_391977

/-- The sum of the infinite series Σ(n=1 to ∞) [(3n - 2) / (n(n+1)(n+3))] is equal to 2/21 -/
theorem infinite_series_sum : 
  (∑' n : ℕ, (3 * n - 2) / (n * (n + 1) * (n + 3))) = 2 / 21 := by
  sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l3919_391977


namespace NUMINAMATH_CALUDE_product_upper_bound_l3919_391939

theorem product_upper_bound (x y z : ℝ) (h1 : x ≥ y) (h2 : y ≥ z) (h3 : x * y + y * z + z * x = 1) : 
  x * z < 1/2 ∧ ∀ ε > 0, ∃ x' y' z' : ℝ, x' ≥ y' ∧ y' ≥ z' ∧ x' * y' + y' * z' + z' * x' = 1 ∧ x' * z' > 1/2 - ε :=
sorry

end NUMINAMATH_CALUDE_product_upper_bound_l3919_391939


namespace NUMINAMATH_CALUDE_prime_pair_product_l3919_391906

theorem prime_pair_product : ∃ p q : ℕ, 
  Nat.Prime p ∧ 
  Nat.Prime q ∧ 
  Odd (p + q) ∧ 
  p + q < 100 ∧ 
  (∃ k : ℕ, p + q = 17 * k) ∧ 
  p * q = 166 := by
  sorry

end NUMINAMATH_CALUDE_prime_pair_product_l3919_391906


namespace NUMINAMATH_CALUDE_complement_intersection_equals_specific_set_l3919_391994

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def M : Set Nat := {1, 4}
def N : Set Nat := {2, 3}

theorem complement_intersection_equals_specific_set :
  (U \ M) ∩ (U \ N) = {5, 6} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_equals_specific_set_l3919_391994


namespace NUMINAMATH_CALUDE_total_skips_theorem_l3919_391955

/-- Represents the number of skips a person can do with one rock -/
structure SkipAbility :=
  (skips : ℕ)

/-- Represents the number of rocks a person skipped -/
structure RocksSkipped :=
  (rocks : ℕ)

/-- Calculates the total skips for a person -/
def totalSkips (ability : SkipAbility) (skipped : RocksSkipped) : ℕ :=
  ability.skips * skipped.rocks

theorem total_skips_theorem 
  (bob_ability : SkipAbility)
  (jim_ability : SkipAbility)
  (sally_ability : SkipAbility)
  (bob_skipped : RocksSkipped)
  (jim_skipped : RocksSkipped)
  (sally_skipped : RocksSkipped)
  (h1 : bob_ability.skips = 12)
  (h2 : jim_ability.skips = 15)
  (h3 : sally_ability.skips = 18)
  (h4 : bob_skipped.rocks = 10)
  (h5 : jim_skipped.rocks = 8)
  (h6 : sally_skipped.rocks = 12) :
  totalSkips bob_ability bob_skipped + 
  totalSkips jim_ability jim_skipped + 
  totalSkips sally_ability sally_skipped = 456 := by
  sorry

#check total_skips_theorem

end NUMINAMATH_CALUDE_total_skips_theorem_l3919_391955


namespace NUMINAMATH_CALUDE_units_digit_of_7_power_1000_l3919_391912

theorem units_digit_of_7_power_1000 : (7^(10^3)) % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_7_power_1000_l3919_391912


namespace NUMINAMATH_CALUDE_point_on_line_l3919_391922

theorem point_on_line (m n p : ℝ) : 
  (m = n / 3 - 2 / 5) → 
  (m + p = (n + 9) / 3 - 2 / 5) → 
  p = 3 := by sorry

end NUMINAMATH_CALUDE_point_on_line_l3919_391922


namespace NUMINAMATH_CALUDE_minimize_difference_product_l3919_391959

theorem minimize_difference_product (x y : ℤ) : 
  (20 * x + 19 * y = 2019) →
  (∀ (a b : ℤ), 20 * a + 19 * b = 2019 → |x - y| ≤ |a - b|) →
  x * y = 2623 := by
sorry

end NUMINAMATH_CALUDE_minimize_difference_product_l3919_391959


namespace NUMINAMATH_CALUDE_joyce_initial_eggs_l3919_391956

/-- 
Given that Joyce has an initial number of eggs, receives 6 more eggs from Marie,
and ends up with 14 eggs in total, prove that Joyce initially had 8 eggs.
-/
theorem joyce_initial_eggs : ℕ → Prop :=
  fun initial_eggs =>
    initial_eggs + 6 = 14 → initial_eggs = 8

/-- Proof of the theorem -/
lemma joyce_initial_eggs_proof : joyce_initial_eggs 8 := by
  sorry

end NUMINAMATH_CALUDE_joyce_initial_eggs_l3919_391956


namespace NUMINAMATH_CALUDE_perfect_square_digit_sum_l3919_391987

def is_valid_number (N : ℕ) : Prop :=
  ∃ k : ℕ,
    10000 ≤ N ∧ N < 100000 ∧
    N = 10000 * k + 1000 * (k + 1) + 100 * (k + 2) + 10 * (3 * k) + (k + 3)

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem perfect_square_digit_sum :
  ∀ N : ℕ, is_valid_number N →
    ∃ m : ℕ, m * m = N → sum_of_digits m = 15 :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_digit_sum_l3919_391987


namespace NUMINAMATH_CALUDE_prime_factors_of_factorial_30_l3919_391978

theorem prime_factors_of_factorial_30 : 
  (Finset.filter Nat.Prime (Finset.range 31)).card = 10 := by
  sorry

end NUMINAMATH_CALUDE_prime_factors_of_factorial_30_l3919_391978


namespace NUMINAMATH_CALUDE_log_30_8_l3919_391970

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Define the variables a and b
variable (a b : ℝ)

-- Define the conditions
axiom lg_5 : lg 5 = a
axiom lg_3 : lg 3 = b

-- State the theorem
theorem log_30_8 : (Real.log 8) / (Real.log 30) = 3 * (1 - a) / (b + 1) :=
sorry

end NUMINAMATH_CALUDE_log_30_8_l3919_391970


namespace NUMINAMATH_CALUDE_greatest_three_digit_number_multiple_condition_l3919_391989

theorem greatest_three_digit_number_multiple_condition : ∃ n : ℕ,
  (n ≤ 999) ∧ 
  (n ≥ 100) ∧
  (∃ k : ℕ, n = 9 * k + 2) ∧
  (∃ m : ℕ, n = 7 * m + 4) ∧
  (∀ x : ℕ, x ≤ 999 ∧ x ≥ 100 ∧ (∃ k : ℕ, x = 9 * k + 2) ∧ (∃ m : ℕ, x = 7 * m + 4) → x ≤ n) ∧
  n = 956 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_three_digit_number_multiple_condition_l3919_391989


namespace NUMINAMATH_CALUDE_fractional_sum_equality_l3919_391908

theorem fractional_sum_equality (n : ℕ) (h : n > 1) :
  ∃ i j : ℕ, (1 : ℚ) / n = 
    Finset.sum (Finset.range (j - i + 1)) (λ k => 1 / ((i + k) * (i + k + 1))) := by
  sorry

end NUMINAMATH_CALUDE_fractional_sum_equality_l3919_391908


namespace NUMINAMATH_CALUDE_num_shortest_paths_A_to_B_l3919_391967

/-- Represents a point on the grid --/
structure Point where
  x : ℕ
  y : ℕ

/-- Represents the road network --/
def RoadNetwork : Type := Set (Point × Point)

/-- Calculates the number of shortest paths between two points on a given road network --/
def numShortestPaths (start finish : Point) (network : RoadNetwork) : ℕ :=
  sorry

/-- The specific road network described in the problem --/
def specificNetwork : RoadNetwork :=
  sorry

/-- The start point A --/
def pointA : Point :=
  ⟨0, 0⟩

/-- The end point B --/
def pointB : Point :=
  ⟨11, 8⟩

/-- Theorem stating that the number of shortest paths from A to B on the specific network is 22023 --/
theorem num_shortest_paths_A_to_B :
  numShortestPaths pointA pointB specificNetwork = 22023 :=
by sorry

end NUMINAMATH_CALUDE_num_shortest_paths_A_to_B_l3919_391967


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l3919_391963

theorem completing_square_equivalence :
  ∀ x : ℝ, x^2 - 6*x + 5 = 0 ↔ (x - 3)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l3919_391963


namespace NUMINAMATH_CALUDE_highway_extension_l3919_391991

theorem highway_extension (current_length : ℕ) (target_length : ℕ) (first_day : ℕ) : 
  current_length = 200 →
  target_length = 650 →
  first_day = 50 →
  target_length - current_length - (first_day + 3 * first_day) = 250 := by
sorry

end NUMINAMATH_CALUDE_highway_extension_l3919_391991


namespace NUMINAMATH_CALUDE_polynomial_bound_l3919_391901

theorem polynomial_bound (z : ℂ) (h : Complex.abs z = 1) :
  ∃ p : Polynomial ℂ, (∀ i : Fin 1996, p.coeff i = 1 ∨ p.coeff i = -1) ∧
    p.degree = 1995 ∧ Complex.abs (p.eval z) ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_bound_l3919_391901


namespace NUMINAMATH_CALUDE_fraction_power_simplification_l3919_391921

theorem fraction_power_simplification :
  (77777 ^ 6 : ℕ) / (11111 ^ 6 : ℕ) = 117649 := by
  sorry

end NUMINAMATH_CALUDE_fraction_power_simplification_l3919_391921


namespace NUMINAMATH_CALUDE_sandys_shopping_money_l3919_391973

theorem sandys_shopping_money (initial_amount : ℝ) : 
  (initial_amount * 0.7 = 217) → initial_amount = 310 := by
  sorry

end NUMINAMATH_CALUDE_sandys_shopping_money_l3919_391973


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l3919_391993

/-- Given two vectors a and b in ℝ², prove that if k*a + b is parallel to a - 3*b, 
    then k = -1/3 -/
theorem parallel_vectors_k_value (a b : ℝ × ℝ) (k : ℝ) 
    (h1 : a = (1, 2)) 
    (h2 : b = (-3, 2)) 
    (h_parallel : ∃ (c : ℝ), c • (k • a + b) = a - 3 • b) : 
  k = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l3919_391993


namespace NUMINAMATH_CALUDE_fraction_of_satisfactory_grades_l3919_391913

def grade_counts : List ℕ := [7, 5, 4, 4, 6]

def satisfactory_grades : List ℕ := grade_counts.take 4

theorem fraction_of_satisfactory_grades :
  (satisfactory_grades.sum : ℚ) / grade_counts.sum = 10 / 13 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_satisfactory_grades_l3919_391913


namespace NUMINAMATH_CALUDE_vector_operation_result_l3919_391914

def a : ℝ × ℝ := (-1, 1)
def b : ℝ × ℝ := (2, -3)

theorem vector_operation_result :
  (2 * a.1 - b.1, 2 * a.2 - b.2) = (-4, 5) := by
  sorry

end NUMINAMATH_CALUDE_vector_operation_result_l3919_391914


namespace NUMINAMATH_CALUDE_linear_systems_solutions_l3919_391969

/-- Prove that the solutions to the given systems of linear equations are correct -/
theorem linear_systems_solutions :
  -- System 1
  (∃ x y : ℝ, 3 * x - 2 * y = -1 ∧ 2 * x + 3 * y = 8 ∧ x = 1 ∧ y = 2) ∧
  -- System 2
  (∃ x y : ℝ, 2 * x + y = 1 ∧ 2 * x - y = 7 ∧ x = 2 ∧ y = -3) :=
by sorry

end NUMINAMATH_CALUDE_linear_systems_solutions_l3919_391969


namespace NUMINAMATH_CALUDE_value_of_x_l3919_391948

theorem value_of_x (x y z : ℝ) 
  (h1 : x = (1/2) * y) 
  (h2 : y = (1/4) * z) 
  (h3 : z = 80) : 
  x = 10 := by
sorry

end NUMINAMATH_CALUDE_value_of_x_l3919_391948


namespace NUMINAMATH_CALUDE_contestant_final_score_l3919_391995

/-- Calculates the final score of a contestant given their individual scores and weightings -/
def final_score (etiquette_score language_score behavior_score : ℝ)
  (etiquette_weight language_weight behavior_weight : ℝ) : ℝ :=
  etiquette_score * etiquette_weight +
  language_score * language_weight +
  behavior_score * behavior_weight

/-- Theorem stating that the contestant's final score is 89 points -/
theorem contestant_final_score :
  final_score 95 92 80 0.4 0.25 0.35 = 89 := by
  sorry

end NUMINAMATH_CALUDE_contestant_final_score_l3919_391995


namespace NUMINAMATH_CALUDE_solution_is_correct_l3919_391985

def is_valid_triple (a m n : ℕ) : Prop :=
  a ≥ 2 ∧ m ≥ 2 ∧ ∃ k : ℕ, a^n + 203 = k * (a^m + 1)

def solution_set : Set (ℕ × ℕ × ℕ) :=
  {(a, m, n) | 
    (∃ k : ℕ, a = 2 ∧ m = 2 ∧ n = 4*k + 1) ∨
    (∃ k : ℕ, a = 2 ∧ m = 3 ∧ n = 6*k + 2) ∨
    (∃ k : ℕ, a = 2 ∧ m = 4 ∧ n = 8*k + 8) ∨
    (∃ k : ℕ, a = 2 ∧ m = 6 ∧ n = 12*k + 9) ∨
    (∃ k : ℕ, a = 3 ∧ m = 2 ∧ n = 4*k + 3) ∨
    (∃ k : ℕ, a = 4 ∧ m = 2 ∧ n = 4*k + 4) ∨
    (∃ k : ℕ, a = 5 ∧ m = 2 ∧ n = 4*k + 1) ∨
    (∃ k : ℕ, a = 8 ∧ m = 2 ∧ n = 4*k + 3) ∨
    (∃ k : ℕ, a = 10 ∧ m = 2 ∧ n = 4*k + 2) ∨
    (∃ k m : ℕ, a = 203 ∧ n = (2*k + 1)*m + 1)}

theorem solution_is_correct :
  ∀ a m n : ℕ, is_valid_triple a m n ↔ (a, m, n) ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_solution_is_correct_l3919_391985


namespace NUMINAMATH_CALUDE_lottery_jackpot_probability_l3919_391934

def megaBallCount : ℕ := 30
def winnerBallCount : ℕ := 50
def bonusBallCount : ℕ := 15
def winnerBallsDrawn : ℕ := 5

def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

def megaBallProb : ℚ := 1 / megaBallCount
def winnerBallsProb : ℚ := 1 / (binomial winnerBallCount winnerBallsDrawn)
def bonusBallProb : ℚ := 1 / bonusBallCount

theorem lottery_jackpot_probability : 
  megaBallProb * winnerBallsProb * bonusBallProb = 1 / 954594900 := by
  sorry

end NUMINAMATH_CALUDE_lottery_jackpot_probability_l3919_391934


namespace NUMINAMATH_CALUDE_frog_jump_distance_l3919_391982

theorem frog_jump_distance (grasshopper_jump : ℕ) (total_jump : ℕ) (frog_jump : ℕ) : 
  grasshopper_jump = 31 → total_jump = 66 → frog_jump = total_jump - grasshopper_jump → frog_jump = 35 := by
  sorry

end NUMINAMATH_CALUDE_frog_jump_distance_l3919_391982


namespace NUMINAMATH_CALUDE_root_existence_l3919_391999

theorem root_existence (h1 : Real.log 1.5 < 4/11) (h2 : Real.log 2 > 2/7) :
  ∃ x : ℝ, 1/4 < x ∧ x < 1/2 ∧ Real.log (2*x + 1) = 1 / (3*x + 2) := by
  sorry

end NUMINAMATH_CALUDE_root_existence_l3919_391999


namespace NUMINAMATH_CALUDE_least_number_divisible_by_all_l3919_391918

def divisors : List Nat := [24, 32, 36, 54, 72, 81, 100]

theorem least_number_divisible_by_all (n : Nat) :
  (∀ d ∈ divisors, (n + 21) % d = 0) →
  (∀ m < n, ∃ d ∈ divisors, (m + 21) % d ≠ 0) →
  n = 64779 := by
  sorry

end NUMINAMATH_CALUDE_least_number_divisible_by_all_l3919_391918


namespace NUMINAMATH_CALUDE_rohan_house_rent_percentage_l3919_391958

/-- Rohan's monthly expenses and savings -/
structure RohanFinances where
  salary : ℝ
  food_percent : ℝ
  entertainment_percent : ℝ
  conveyance_percent : ℝ
  savings : ℝ
  house_rent_percent : ℝ

/-- Theorem stating that Rohan spends 20% of his salary on house rent -/
theorem rohan_house_rent_percentage 
  (rf : RohanFinances)
  (h_salary : rf.salary = 7500)
  (h_food : rf.food_percent = 40)
  (h_entertainment : rf.entertainment_percent = 10)
  (h_conveyance : rf.conveyance_percent = 10)
  (h_savings : rf.savings = 1500)
  (h_total : rf.food_percent + rf.entertainment_percent + rf.conveyance_percent + 
             rf.house_rent_percent + (rf.savings / rf.salary * 100) = 100) :
  rf.house_rent_percent = 20 := by
  sorry

end NUMINAMATH_CALUDE_rohan_house_rent_percentage_l3919_391958


namespace NUMINAMATH_CALUDE_f_value_at_7_minus_a_l3919_391951

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2^x - 2 else -Real.log x / Real.log 3

-- Define a as the value where f(a) = -2
noncomputable def a : ℝ :=
  Real.exp (-2 * Real.log 3)

-- Theorem statement
theorem f_value_at_7_minus_a :
  f (7 - a) = -7/4 :=
sorry

end NUMINAMATH_CALUDE_f_value_at_7_minus_a_l3919_391951


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3919_391937

theorem complex_fraction_simplification :
  (2 : ℂ) / (Complex.I * (3 - Complex.I)) = (1 - 3 * Complex.I) / 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3919_391937


namespace NUMINAMATH_CALUDE_smallest_n_for_coprime_subset_l3919_391998

def S : Set Nat := {n | 1 ≤ n ∧ n ≤ 100}

theorem smallest_n_for_coprime_subset : 
  ∃ (n : Nat), n = 75 ∧ 
  (∀ (A : Set Nat), A ⊆ S → A.Finite → A.ncard ≥ n → 
    ∃ (a b c d : Nat), a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ d ∈ A ∧ 
    Nat.Coprime a b ∧ Nat.Coprime a c ∧ Nat.Coprime a d ∧ 
    Nat.Coprime b c ∧ Nat.Coprime b d ∧ Nat.Coprime c d) ∧
  (∀ (m : Nat), m < 75 → 
    ∃ (B : Set Nat), B ⊆ S ∧ B.Finite ∧ B.ncard = m ∧
    ¬(∃ (a b c d : Nat), a ∈ B ∧ b ∈ B ∧ c ∈ B ∧ d ∈ B ∧ 
      Nat.Coprime a b ∧ Nat.Coprime a c ∧ Nat.Coprime a d ∧ 
      Nat.Coprime b c ∧ Nat.Coprime b d ∧ Nat.Coprime c d)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_coprime_subset_l3919_391998


namespace NUMINAMATH_CALUDE_common_divisors_of_36_and_60_l3919_391981

/-- The number of positive integers that are divisors of both 36 and 60 -/
def common_divisors_count : ℕ := 
  (Finset.filter (fun d => 36 % d = 0 ∧ 60 % d = 0) (Finset.range 61)).card

/-- Theorem stating that the number of common divisors of 36 and 60 is 6 -/
theorem common_divisors_of_36_and_60 : common_divisors_count = 6 := by
  sorry

end NUMINAMATH_CALUDE_common_divisors_of_36_and_60_l3919_391981


namespace NUMINAMATH_CALUDE_function_value_l3919_391962

-- Define the function f
noncomputable def f (b : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 + b else Real.log x / Real.log 2

-- State the theorem
theorem function_value (b : ℝ) : f b (f b (1/2)) = 3 → b = 2 := by
  sorry

end NUMINAMATH_CALUDE_function_value_l3919_391962


namespace NUMINAMATH_CALUDE_max_college_courses_l3919_391902

theorem max_college_courses (max_courses sid_courses : ℕ) : 
  sid_courses = 4 * max_courses →
  max_courses + sid_courses = 200 →
  max_courses = 40 := by
sorry

end NUMINAMATH_CALUDE_max_college_courses_l3919_391902


namespace NUMINAMATH_CALUDE_mikeys_leaves_l3919_391903

/-- Given an initial number of leaves and a number of leaves that blew away,
    calculate the remaining number of leaves. -/
def remaining_leaves (initial : ℕ) (blew_away : ℕ) : ℕ :=
  initial - blew_away

/-- Theorem stating that for Mikey's specific case, the remaining leaves
    calculation yields the correct result. -/
theorem mikeys_leaves :
  remaining_leaves 356 244 = 112 := by
  sorry

#eval remaining_leaves 356 244

end NUMINAMATH_CALUDE_mikeys_leaves_l3919_391903


namespace NUMINAMATH_CALUDE_john_good_games_l3919_391984

/-- 
Given:
- John bought 21 games from a friend
- John bought 8 games at a garage sale
- 23 of the games didn't work

Prove that John ended up with 6 good games.
-/
theorem john_good_games : 
  let games_from_friend : ℕ := 21
  let games_from_garage_sale : ℕ := 8
  let non_working_games : ℕ := 23
  let total_games := games_from_friend + games_from_garage_sale
  let good_games := total_games - non_working_games
  good_games = 6 := by
  sorry

end NUMINAMATH_CALUDE_john_good_games_l3919_391984


namespace NUMINAMATH_CALUDE_spade_problem_l3919_391943

def spade (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem spade_problem : spade 5 (spade 3 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_spade_problem_l3919_391943


namespace NUMINAMATH_CALUDE_convex_polygon_diagonals_l3919_391935

theorem convex_polygon_diagonals (n : ℕ) (h : n = 49) : 
  (n * (n - 3)) / 2 = 23 * n := by
  sorry

end NUMINAMATH_CALUDE_convex_polygon_diagonals_l3919_391935


namespace NUMINAMATH_CALUDE_extra_planks_count_l3919_391925

/-- The number of planks Charlie got -/
def charlie_planks : ℕ := 10

/-- The number of planks Charlie's father got -/
def father_planks : ℕ := 10

/-- The total number of wood pieces they have -/
def total_wood : ℕ := 35

/-- The number of extra planks initially in the house -/
def extra_planks : ℕ := total_wood - (charlie_planks + father_planks)

theorem extra_planks_count : extra_planks = 15 := by
  sorry

end NUMINAMATH_CALUDE_extra_planks_count_l3919_391925


namespace NUMINAMATH_CALUDE_two_digit_addition_puzzle_l3919_391974

theorem two_digit_addition_puzzle :
  ∀ (A B : ℕ),
    A ≠ B →
    A < 10 →
    B < 10 →
    10 * A + B + 25 = 10 * B + 3 →
    B = 8 :=
by sorry

end NUMINAMATH_CALUDE_two_digit_addition_puzzle_l3919_391974


namespace NUMINAMATH_CALUDE_fraction_irreducible_l3919_391932

theorem fraction_irreducible (n : ℤ) : Int.gcd (39 * n + 4) (26 * n + 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_irreducible_l3919_391932


namespace NUMINAMATH_CALUDE_sparklers_to_crackers_value_comparison_l3919_391946

-- Define the exchange rates
def ornament_to_cracker : ℚ := 2
def sparkler_to_garland : ℚ := 2/5
def ornament_to_garland : ℚ := 1/4

-- Define the conversion function
def convert (item : String) (quantity : ℚ) : ℚ :=
  match item with
  | "sparkler" => quantity * sparkler_to_garland * (1 / ornament_to_garland) * ornament_to_cracker
  | "ornament" => quantity * ornament_to_cracker
  | _ => 0

-- Theorem 1: 10 sparklers are equivalent to 32 crackers
theorem sparklers_to_crackers :
  convert "sparkler" 10 = 32 :=
sorry

-- Theorem 2: 5 Christmas ornaments and 1 cracker are more valuable than 2 sparklers
theorem value_comparison :
  convert "ornament" 5 + 1 > convert "sparkler" 2 :=
sorry

end NUMINAMATH_CALUDE_sparklers_to_crackers_value_comparison_l3919_391946


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l3919_391957

theorem simplify_trig_expression :
  (Real.sin (58 * π / 180) - Real.sin (28 * π / 180) * Real.cos (30 * π / 180)) / Real.cos (28 * π / 180) = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l3919_391957


namespace NUMINAMATH_CALUDE_part_one_part_two_l3919_391917

-- Define the function f
def f (a x : ℝ) : ℝ := (a + 1) * x^2 + 4 * a * x - 3

-- Part I
theorem part_one (a : ℝ) :
  a > 0 →
  (∃ x₁ x₂ : ℝ, f a x₁ = 0 ∧ f a x₂ = 0 ∧ x₁ > 1 ∧ x₂ < 1) ↔
  (0 < a ∧ a < 2/5) :=
sorry

-- Part II
theorem part_two (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc 0 2 → f a x ≤ f a 2) ↔
  a ≥ -1/3 :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3919_391917


namespace NUMINAMATH_CALUDE_acme_profit_calculation_l3919_391964

/-- Calculates the profit for Acme's horseshoe manufacturing business -/
def calculate_profit (initial_outlay : ℝ) (cost_per_set : ℝ) (price_per_set : ℝ) (num_sets : ℕ) : ℝ :=
  let total_revenue := price_per_set * num_sets
  let total_cost := initial_outlay + cost_per_set * num_sets
  total_revenue - total_cost

/-- Proves that the profit for Acme's horseshoe manufacturing business is $15,337.50 -/
theorem acme_profit_calculation :
  calculate_profit 12450 20.75 50 950 = 15337.50 := by
  sorry

end NUMINAMATH_CALUDE_acme_profit_calculation_l3919_391964


namespace NUMINAMATH_CALUDE_hunting_ratio_l3919_391988

theorem hunting_ratio : 
  ∀ (sam rob mark peter total : ℕ) (mark_fraction : ℚ),
    sam = 6 →
    rob = sam / 2 →
    mark = mark_fraction * (sam + rob) →
    peter = 3 * mark →
    sam + rob + mark + peter = 21 →
    mark_fraction = 1 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_hunting_ratio_l3919_391988


namespace NUMINAMATH_CALUDE_compute_fraction_power_l3919_391904

theorem compute_fraction_power : 8 * (1 / 3)^4 = 8 / 81 := by
  sorry

end NUMINAMATH_CALUDE_compute_fraction_power_l3919_391904


namespace NUMINAMATH_CALUDE_additive_function_characterization_l3919_391919

/-- A function satisfying the given functional equation -/
def AdditiveFunctionQ (f : ℚ → ℚ) : Prop :=
  ∀ x y : ℚ, f (x + y) = f x + f y

/-- The main theorem characterizing additive functions on rationals -/
theorem additive_function_characterization :
  ∀ f : ℚ → ℚ, AdditiveFunctionQ f →
  ∃ a : ℚ, ∀ x : ℚ, f x = a * x := by
  sorry

end NUMINAMATH_CALUDE_additive_function_characterization_l3919_391919


namespace NUMINAMATH_CALUDE_existence_of_x_y_l3919_391953

theorem existence_of_x_y (p : ℝ) : (∃ x y : ℝ, p = x + y ∧ x^2 + 4*y^2 + 8*y + 4 ≤ 4*x) ↔ 
  -3 - Real.sqrt 5 ≤ p ∧ p ≤ -3 + Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_existence_of_x_y_l3919_391953


namespace NUMINAMATH_CALUDE_conditional_probability_same_color_given_first_red_l3919_391930

def total_balls : ℕ := 5
def red_balls : ℕ := 2
def black_balls : ℕ := 3

def P_A : ℚ := red_balls / total_balls
def P_AB : ℚ := (red_balls / total_balls) * ((red_balls - 1) / (total_balls - 1))

theorem conditional_probability_same_color_given_first_red :
  P_AB / P_A = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_conditional_probability_same_color_given_first_red_l3919_391930


namespace NUMINAMATH_CALUDE_cauchy_schwarz_inequality_l3919_391996

theorem cauchy_schwarz_inequality (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : 
  a * c + b * d ≤ Real.sqrt ((a^2 + b^2) * (c^2 + d^2)) := by
  sorry

end NUMINAMATH_CALUDE_cauchy_schwarz_inequality_l3919_391996


namespace NUMINAMATH_CALUDE_coplanar_iff_k_eq_neg_two_l3919_391945

/-- Two lines in 3D space -/
structure Line3D where
  parameterization : ℝ → ℝ × ℝ × ℝ

/-- Checks if two lines are coplanar -/
def are_coplanar (l1 l2 : Line3D) : Prop :=
  ∃ (a b c d : ℝ), a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0 ∧
    ∀ (t s : ℝ),
      let (x1, y1, z1) := l1.parameterization s
      let (x2, y2, z2) := l2.parameterization t
      a * x1 + b * y1 + c * z1 + d =
      a * x2 + b * y2 + c * z2 + d

theorem coplanar_iff_k_eq_neg_two :
  ∀ (k : ℝ),
    let l1 : Line3D := ⟨λ s => (-1 + s, 3 - k*s, 1 + k*s)⟩
    let l2 : Line3D := ⟨λ t => (t/2, 1 + t, 2 - t)⟩
    are_coplanar l1 l2 ↔ k = -2 := by
  sorry

end NUMINAMATH_CALUDE_coplanar_iff_k_eq_neg_two_l3919_391945


namespace NUMINAMATH_CALUDE_euler_property_division_l3919_391980

theorem euler_property_division (x : ℝ) : 
  (x > 0) →
  (1/2 * x - 3000 + 1/3 * x - 1000 + 1/4 * x + 1/5 * x + 600 = x) →
  (x = 12000 ∧ 
   1/2 * x - 3000 = 3000 ∧
   1/3 * x - 1000 = 3000 ∧
   1/4 * x = 3000 ∧
   1/5 * x + 600 = 3000) :=
by sorry

end NUMINAMATH_CALUDE_euler_property_division_l3919_391980


namespace NUMINAMATH_CALUDE_min_games_to_dominate_leaderboard_l3919_391910

/-- Represents the game with a leaderboard of 30 scores -/
structure Game where
  leaderboard_size : Nat
  leaderboard_size_eq : leaderboard_size = 30

/-- Calculates the number of games needed to achieve all scores -/
def games_needed (game : Game) : Nat :=
  game.leaderboard_size + (game.leaderboard_size * (game.leaderboard_size - 1)) / 2

/-- Theorem stating the minimum number of games required -/
theorem min_games_to_dominate_leaderboard (game : Game) :
  games_needed game = 465 := by
  sorry

#check min_games_to_dominate_leaderboard

end NUMINAMATH_CALUDE_min_games_to_dominate_leaderboard_l3919_391910


namespace NUMINAMATH_CALUDE_farmer_ploughing_problem_l3919_391983

/-- Represents the problem of determining the planned daily ploughing area for a farmer --/
theorem farmer_ploughing_problem 
  (total_area : ℝ) 
  (actual_daily_area : ℝ) 
  (extra_days : ℕ) 
  (area_left : ℝ) 
  (h1 : total_area = 448) 
  (h2 : actual_daily_area = 85) 
  (h3 : extra_days = 2) 
  (h4 : area_left = 40) : 
  ∃ planned_daily_area : ℝ, 
    planned_daily_area = 188.5 ∧ 
    (total_area / planned_daily_area + extra_days) * actual_daily_area = total_area - area_left :=
sorry

end NUMINAMATH_CALUDE_farmer_ploughing_problem_l3919_391983


namespace NUMINAMATH_CALUDE_blackboard_operation_theorem_l3919_391961

/-- Operation that replaces a number with two new numbers -/
def replace_operation (r : ℝ) : ℝ × ℝ :=
  let a := r
  let b := 2 * r
  (a, b)

/-- Applies the replace_operation n times to an initial set of numbers -/
def apply_operations (initial : Set ℝ) (n : ℕ) : Set ℝ :=
  sorry

theorem blackboard_operation_theorem (r : ℝ) (k : ℕ) (h_r : r > 0) :
  ∃ s ∈ apply_operations {r} (k^2 - 1), s ≤ k * r :=
sorry

end NUMINAMATH_CALUDE_blackboard_operation_theorem_l3919_391961


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3919_391915

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, 1 + x^5 = a₀ + a₁*(x-1) + a₂*(x-1)^2 + a₃*(x-1)^3 + a₄*(x-1)^4 + a₅*(x-1)^5) →
  a₁ + a₂ + a₃ + a₄ + a₅ = 31 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3919_391915


namespace NUMINAMATH_CALUDE_final_balance_is_96_l3919_391954

/-- Calculates the final balance on a credit card given initial balance, interest rate, additional charge, and number of interest applications. -/
def calculateFinalBalance (initialBalance : ℚ) (interestRate : ℚ) (additionalCharge : ℚ) (interestApplications : ℕ) : ℚ :=
  let balanceAfterFirstInterest := initialBalance * (1 + interestRate)
  let balanceBeforeSecondInterest := balanceAfterFirstInterest + additionalCharge
  balanceBeforeSecondInterest * (1 + interestRate)

/-- Theorem stating that the final balance is $96.00 given the specific conditions. -/
theorem final_balance_is_96 :
  calculateFinalBalance 50 (1/5) 20 2 = 96 := by
  sorry

#eval calculateFinalBalance 50 (1/5) 20 2

end NUMINAMATH_CALUDE_final_balance_is_96_l3919_391954


namespace NUMINAMATH_CALUDE_polyhedron_sum_l3919_391941

/-- A convex polyhedron with triangular, pentagonal, and hexagonal faces. -/
structure Polyhedron where
  T : ℕ  -- Number of triangular faces
  P : ℕ  -- Number of pentagonal faces
  H : ℕ  -- Number of hexagonal faces
  V : ℕ  -- Number of vertices
  E : ℕ  -- Number of edges

/-- Properties of the polyhedron -/
def is_valid_polyhedron (poly : Polyhedron) : Prop :=
  -- Total number of faces is 42
  poly.T + poly.P + poly.H = 42 ∧
  -- At each vertex, 3 triangular, 2 pentagonal, and 1 hexagonal face meet
  6 * poly.V = 3 * poly.T + 2 * poly.P + poly.H ∧
  -- Edge count
  2 * poly.E = 3 * poly.T + 5 * poly.P + 6 * poly.H ∧
  -- Euler's formula
  poly.V - poly.E + (poly.T + poly.P + poly.H) = 2

/-- Theorem statement -/
theorem polyhedron_sum (poly : Polyhedron) 
  (h : is_valid_polyhedron poly) : 
  100 * poly.H + 10 * poly.P + poly.T + poly.V = 714 := by
  sorry


end NUMINAMATH_CALUDE_polyhedron_sum_l3919_391941


namespace NUMINAMATH_CALUDE_grants_test_score_l3919_391944

theorem grants_test_score (hunter_score john_score grant_score : ℕ) :
  hunter_score = 45 →
  john_score = 2 * hunter_score →
  grant_score = john_score + 10 →
  grant_score = 100 := by
sorry

end NUMINAMATH_CALUDE_grants_test_score_l3919_391944


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3919_391929

theorem arithmetic_sequence_problem (a₁ d : ℝ) : 
  let a := fun n => a₁ + (n - 1) * d
  (a 9) / (a 2) = 5 ∧ (a 13) = 2 * (a 6) + 5 → a₁ = 3 ∧ d = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3919_391929


namespace NUMINAMATH_CALUDE_average_marks_second_class_l3919_391975

theorem average_marks_second_class 
  (students1 : ℕ) 
  (students2 : ℕ) 
  (avg1 : ℝ) 
  (avg_combined : ℝ) :
  students1 = 25 →
  students2 = 40 →
  avg1 = 50 →
  avg_combined = 59.23076923076923 →
  let total_students := students1 + students2
  let avg2 := (avg_combined * total_students - avg1 * students1) / students2
  avg2 = 65 := by sorry

end NUMINAMATH_CALUDE_average_marks_second_class_l3919_391975


namespace NUMINAMATH_CALUDE_ceiling_fraction_evaluation_l3919_391938

theorem ceiling_fraction_evaluation : 
  (⌈(19 / 8 : ℚ) - ⌈(35 / 19 : ℚ)⌉⌉ : ℚ) / 
  (⌈(35 / 8 : ℚ) + ⌈(8 * 19 / 35 : ℚ)⌉⌉ : ℚ) = 1 / 10 := by sorry

end NUMINAMATH_CALUDE_ceiling_fraction_evaluation_l3919_391938


namespace NUMINAMATH_CALUDE_cube_decomposition_smallest_term_l3919_391960

theorem cube_decomposition_smallest_term (m : ℕ) (h : m > 0) :
  m^2 - m + 1 = 73 → m = 9 := by
  sorry

end NUMINAMATH_CALUDE_cube_decomposition_smallest_term_l3919_391960


namespace NUMINAMATH_CALUDE_negative_two_cubed_equality_l3919_391933

theorem negative_two_cubed_equality : (-2)^3 = -2^3 := by
  sorry

end NUMINAMATH_CALUDE_negative_two_cubed_equality_l3919_391933


namespace NUMINAMATH_CALUDE_waiter_tables_l3919_391926

theorem waiter_tables (initial_customers : ℕ) (left_customers : ℕ) (people_per_table : ℕ) 
  (h1 : initial_customers = 22)
  (h2 : left_customers = 14)
  (h3 : people_per_table = 4) :
  (initial_customers - left_customers) / people_per_table = 2 := by
  sorry

end NUMINAMATH_CALUDE_waiter_tables_l3919_391926


namespace NUMINAMATH_CALUDE_rectangular_prism_dimensions_l3919_391997

theorem rectangular_prism_dimensions :
  ∀ (l b h : ℝ),
    l = 3 * b →
    l = 2 * h →
    l * b * h = 12168 →
    l = 42 ∧ b = 14 ∧ h = 21 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_dimensions_l3919_391997


namespace NUMINAMATH_CALUDE_unique_pair_l3919_391940

-- Define the properties of N and X
def is_valid_pair (N : ℕ) (X : ℚ) : Prop :=
  -- N is a two-digit natural number
  10 ≤ N ∧ N < 100 ∧
  -- X is a two-digit decimal number
  3 ≤ X ∧ X < 10 ∧
  -- N becomes 56.7 smaller when a decimal point is inserted between its digits
  (N : ℚ) = (N / 10 : ℚ) + 56.7 ∧
  -- X becomes twice as close to N after this change
  (N : ℚ) - X = 2 * ((N : ℚ) - (N / 10 : ℚ))

-- Theorem statement
theorem unique_pair : ∃! (N : ℕ) (X : ℚ), is_valid_pair N X ∧ N = 63 ∧ X = 3.6 := by
  sorry

end NUMINAMATH_CALUDE_unique_pair_l3919_391940


namespace NUMINAMATH_CALUDE_union_of_sets_l3919_391992

theorem union_of_sets : 
  let P : Set ℕ := {1, 2, 3, 4}
  let Q : Set ℕ := {3, 4, 5, 6}
  P ∪ Q = {1, 2, 3, 4, 5, 6} := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l3919_391992


namespace NUMINAMATH_CALUDE_money_split_l3919_391990

theorem money_split (donna_share : ℚ) (donna_amount : ℕ) (total : ℕ) : 
  donna_share = 5 / 17 →
  donna_amount = 35 →
  donna_share * total = donna_amount →
  total = 119 := by
sorry

end NUMINAMATH_CALUDE_money_split_l3919_391990
