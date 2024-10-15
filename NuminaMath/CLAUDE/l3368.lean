import Mathlib

namespace NUMINAMATH_CALUDE_quick_customer_sale_l3368_336827

def chicken_problem (initial_chickens neighbor_sale remaining_chickens : ℕ) : ℕ :=
  initial_chickens - neighbor_sale - remaining_chickens

theorem quick_customer_sale :
  chicken_problem 80 12 43 = 25 := by
  sorry

end NUMINAMATH_CALUDE_quick_customer_sale_l3368_336827


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_bound_l3368_336824

theorem quadratic_roots_sum_bound (p : ℝ) (r₁ r₂ : ℝ) : 
  r₁^2 + p*r₁ + 15 = 0 → r₂^2 + p*r₂ + 15 = 0 → |r₁ + r₂| > 2 * Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_bound_l3368_336824


namespace NUMINAMATH_CALUDE_consumption_wage_ratio_l3368_336874

/-- Linear regression equation parameters -/
def a : ℝ := 0.6
def b : ℝ := 1.5

/-- Average consumption per capita -/
def y : ℝ := 7.5

/-- Theorem stating the ratio of average consumption to average wage -/
theorem consumption_wage_ratio :
  ∃ x : ℝ, y = a * x + b ∧ y / x = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_consumption_wage_ratio_l3368_336874


namespace NUMINAMATH_CALUDE_quadratic_solution_difference_l3368_336829

theorem quadratic_solution_difference : ∃ (x₁ x₂ : ℝ), 
  (x₁^2 - 5*x₁ + 15 = x₁ + 55) ∧ 
  (x₂^2 - 5*x₂ + 15 = x₂ + 55) ∧ 
  (x₁ ≠ x₂) ∧
  (|x₁ - x₂| = 14) := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_difference_l3368_336829


namespace NUMINAMATH_CALUDE_zeros_of_composite_function_l3368_336826

noncomputable section

-- Define the function f
def f (k : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then k * x + 1 else Real.log x

-- Define the composite function g
def g (k : ℝ) (x : ℝ) : ℝ := f k (f k x + 1)

-- Theorem statement
theorem zeros_of_composite_function (k : ℝ) :
  (k > 0 → (∃ x₁ x₂ x₃ x₄ : ℝ, g k x₁ = 0 ∧ g k x₂ = 0 ∧ g k x₃ = 0 ∧ g k x₄ = 0 ∧
    x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄)) ∧
  (k < 0 → (∃! x : ℝ, g k x = 0)) :=
sorry

end

end NUMINAMATH_CALUDE_zeros_of_composite_function_l3368_336826


namespace NUMINAMATH_CALUDE_ice_cream_sales_l3368_336822

theorem ice_cream_sales (sales : List ℝ) (mean : ℝ) : 
  sales.length = 6 →
  sales = [100, 92, 109, 96, 103, 105] →
  mean = 100.1 →
  (sales.sum + (7 * mean - sales.sum)) / 7 = mean →
  7 * mean - sales.sum = 95.7 :=
by sorry

end NUMINAMATH_CALUDE_ice_cream_sales_l3368_336822


namespace NUMINAMATH_CALUDE_count_odd_rank_subsets_l3368_336869

/-- The number of cards in the deck -/
def total_cards : ℕ := 8056

/-- The number of ranks in the deck -/
def total_ranks : ℕ := 2014

/-- The number of suits per rank -/
def suits_per_rank : ℕ := 4

/-- The number of subsets with cards from an odd number of distinct ranks -/
def odd_rank_subsets : ℕ := (16^total_ranks - 14^total_ranks) / 2

/-- Theorem stating the number of subsets with cards from an odd number of distinct ranks -/
theorem count_odd_rank_subsets :
  total_cards = total_ranks * suits_per_rank →
  odd_rank_subsets = (16^total_ranks - 14^total_ranks) / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_count_odd_rank_subsets_l3368_336869


namespace NUMINAMATH_CALUDE_sum_reciprocals_l3368_336835

theorem sum_reciprocals (a b c d : ℝ) (ω : ℂ) : 
  a ≠ -1 → b ≠ -1 → c ≠ -1 → d ≠ -1 →
  ω^4 = 1 → ω ≠ 1 →
  1 / (a + ω) + 1 / (b + ω) + 1 / (c + ω) + 1 / (d + ω) = 3 / ω^2 →
  1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) + 1 / (d + 1) = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_reciprocals_l3368_336835


namespace NUMINAMATH_CALUDE_basketball_probability_l3368_336858

theorem basketball_probability (p_at_least_one p_hs3 p_pro3 : ℝ) :
  p_at_least_one = 0.9333333333333333 →
  p_hs3 = 1/2 →
  p_pro3 = 1/3 →
  1 - (1 - p_hs3) * (1 - p_pro3) * (1 - 0.8) = p_at_least_one :=
by sorry

end NUMINAMATH_CALUDE_basketball_probability_l3368_336858


namespace NUMINAMATH_CALUDE_total_voters_l3368_336813

/-- Given information about voters in three districts, prove the total number of voters. -/
theorem total_voters (d1 d2 d3 : ℕ) : 
  d1 = 322 →
  d3 = 2 * d1 →
  d2 = d3 - 19 →
  d1 + d2 + d3 = 1591 := by
sorry

end NUMINAMATH_CALUDE_total_voters_l3368_336813


namespace NUMINAMATH_CALUDE_grid_squares_count_l3368_336831

/-- Represents a square grid --/
structure Grid :=
  (size : Nat)

/-- Counts the number of squares of a given size in the grid --/
def countSquares (g : Grid) (squareSize : Nat) : Nat :=
  (g.size + 1 - squareSize) * (g.size + 1 - squareSize)

/-- Calculates the total number of squares in the grid --/
def totalSquares (g : Grid) : Nat :=
  (countSquares g 1) + (countSquares g 2) + (countSquares g 3) + (countSquares g 4)

/-- Theorem stating that the total number of squares in a 5x5 grid is 54 --/
theorem grid_squares_count :
  let g : Grid := ⟨5⟩
  totalSquares g = 54 := by
  sorry

end NUMINAMATH_CALUDE_grid_squares_count_l3368_336831


namespace NUMINAMATH_CALUDE_correct_factorization_l3368_336807

theorem correct_factorization (x : ℝ) : -x^2 + 2*x - 1 = -(x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_correct_factorization_l3368_336807


namespace NUMINAMATH_CALUDE_batsman_matches_proof_l3368_336843

theorem batsman_matches_proof (first_matches : Nat) (second_matches : Nat) 
  (first_average : Nat) (second_average : Nat) (overall_average : Nat) :
  first_matches = 30 ∧ 
  second_matches = 15 ∧ 
  first_average = 50 ∧ 
  second_average = 26 ∧ 
  overall_average = 42 →
  first_matches + second_matches = 45 := by
  sorry

end NUMINAMATH_CALUDE_batsman_matches_proof_l3368_336843


namespace NUMINAMATH_CALUDE_tom_peeled_24_potatoes_l3368_336891

/-- Represents the potato peeling scenario -/
structure PotatoPeeling where
  initialPile : ℕ
  maryRate : ℕ
  tomRate : ℕ
  maryAloneTime : ℕ

/-- Calculates the number of potatoes Tom peeled -/
def potatoesPeeledByTom (scenario : PotatoPeeling) : ℕ :=
  let potatoesPeeledByMaryAlone := scenario.maryRate * scenario.maryAloneTime
  let remainingPotatoes := scenario.initialPile - potatoesPeeledByMaryAlone
  let combinedRate := scenario.maryRate + scenario.tomRate
  let timeToFinish := remainingPotatoes / combinedRate
  scenario.tomRate * timeToFinish

/-- Theorem stating that Tom peeled 24 potatoes -/
theorem tom_peeled_24_potatoes :
  let scenario : PotatoPeeling := {
    initialPile := 60,
    maryRate := 4,
    tomRate := 6,
    maryAloneTime := 5
  }
  potatoesPeeledByTom scenario = 24 := by sorry

end NUMINAMATH_CALUDE_tom_peeled_24_potatoes_l3368_336891


namespace NUMINAMATH_CALUDE_min_sum_is_two_l3368_336893

/-- Represents a sequence of five digits -/
def DigitSequence := Fin 5 → Nat

/-- Ensures all digits in the sequence are between 1 and 9 -/
def valid_sequence (s : DigitSequence) : Prop :=
  ∀ i, 1 ≤ s i ∧ s i ≤ 9

/-- Computes the sum of the last four digits in the sequence -/
def sum_last_four (s : DigitSequence) : Nat :=
  (s 1) + (s 2) + (s 3) + (s 4)

/-- Represents the evolution rule for the sequence -/
def evolve (s : DigitSequence) : DigitSequence :=
  fun i => match i with
    | 0 => s 1
    | 1 => s 2
    | 2 => s 3
    | 3 => s 4
    | 4 => sum_last_four s % 10

/-- Represents the sum of all digits in the sequence -/
def sequence_sum (s : DigitSequence) : Nat :=
  (s 0) + (s 1) + (s 2) + (s 3) + (s 4)

/-- The main theorem stating that the minimum sum is 2 -/
theorem min_sum_is_two :
  ∃ (s : DigitSequence), valid_sequence s ∧
  ∀ (n : Nat), sequence_sum (Nat.iterate evolve n s) ≥ 2 ∧
  ∃ (m : Nat), sequence_sum (Nat.iterate evolve m s) = 2 :=
sorry

end NUMINAMATH_CALUDE_min_sum_is_two_l3368_336893


namespace NUMINAMATH_CALUDE_divisors_of_eight_factorial_greater_than_seven_factorial_l3368_336879

theorem divisors_of_eight_factorial_greater_than_seven_factorial :
  (Finset.filter (fun d => d > Nat.factorial 7 ∧ Nat.factorial 8 % d = 0) (Finset.range (Nat.factorial 8 + 1))).card = 7 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_eight_factorial_greater_than_seven_factorial_l3368_336879


namespace NUMINAMATH_CALUDE_twice_largest_two_digit_is_190_l3368_336817

def largest_two_digit (a b c : Nat) : Nat :=
  max (10 * max a (max b c) + min (max a b) (max b c))
      (10 * min (max a b) (max b c) + max a (max b c))

theorem twice_largest_two_digit_is_190 :
  largest_two_digit 3 5 9 * 2 = 190 := by
  sorry

end NUMINAMATH_CALUDE_twice_largest_two_digit_is_190_l3368_336817


namespace NUMINAMATH_CALUDE_kola_solution_water_percentage_l3368_336894

/-- Proves that the initial water percentage in a kola solution was 64% -/
theorem kola_solution_water_percentage :
  let initial_volume : ℝ := 340
  let initial_kola_percentage : ℝ := 9
  let added_sugar : ℝ := 3.2
  let added_water : ℝ := 8
  let added_kola : ℝ := 6.8
  let final_sugar_percentage : ℝ := 26.536312849162012
  let initial_water_percentage : ℝ := 64
  let initial_sugar_percentage : ℝ := 91 - initial_water_percentage - initial_kola_percentage
  let final_volume : ℝ := initial_volume + added_sugar + added_water + added_kola
  let final_sugar_volume : ℝ := (initial_sugar_percentage / 100) * initial_volume + added_sugar
  final_sugar_volume / final_volume * 100 = final_sugar_percentage :=
by
  sorry


end NUMINAMATH_CALUDE_kola_solution_water_percentage_l3368_336894


namespace NUMINAMATH_CALUDE_dice_probability_l3368_336816

def num_dice : ℕ := 5
def num_faces : ℕ := 12
def num_divisible_by_three : ℕ := 4  -- 3, 6, 9, 12 are divisible by 3

def prob_divisible_by_three : ℚ := num_divisible_by_three / num_faces
def prob_not_divisible_by_three : ℚ := 1 - prob_divisible_by_three

def exactly_three_divisible_probability : ℚ :=
  (Nat.choose num_dice 3 : ℚ) * 
  (prob_divisible_by_three ^ 3) * 
  (prob_not_divisible_by_three ^ 2)

theorem dice_probability : 
  exactly_three_divisible_probability = 40 / 243 := by
  sorry

end NUMINAMATH_CALUDE_dice_probability_l3368_336816


namespace NUMINAMATH_CALUDE_odot_inequality_range_l3368_336855

-- Define the operation ⊙
def odot (x y : ℝ) : ℝ := x * (1 - y)

-- State the theorem
theorem odot_inequality_range (a : ℝ) :
  (∀ x : ℝ, odot (x - a) (x + a) < 1) ↔ -1/2 < a ∧ a < 3/2 :=
by sorry

end NUMINAMATH_CALUDE_odot_inequality_range_l3368_336855


namespace NUMINAMATH_CALUDE_prop_one_prop_three_prop_five_l3368_336802

-- Proposition 1
theorem prop_one (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : (a + 1) / (b + 1) > a / b) : a < b := by
  sorry

-- Proposition 3
theorem prop_three : ∀ x : ℝ, x^2 - 2*x + 1 ≥ 0 := by
  sorry

-- Proposition 5
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.enum.foldl (λ acc (i, b) => acc + if b then 2^i else 0) 0

theorem prop_five : binary_to_decimal [true, false, true, true, true] = 23 := by
  sorry

end NUMINAMATH_CALUDE_prop_one_prop_three_prop_five_l3368_336802


namespace NUMINAMATH_CALUDE_min_wire_length_for_specific_parallelepiped_l3368_336899

/-- The minimum length of wire needed to construct a rectangular parallelepiped -/
def wire_length (width length height : ℝ) : ℝ :=
  4 * (width + length + height)

/-- Theorem stating the minimum wire length for a specific rectangular parallelepiped -/
theorem min_wire_length_for_specific_parallelepiped :
  wire_length 10 8 5 = 92 := by
  sorry

end NUMINAMATH_CALUDE_min_wire_length_for_specific_parallelepiped_l3368_336899


namespace NUMINAMATH_CALUDE_marcel_shopping_cost_l3368_336853

def pen_cost : ℝ := 4

def briefcase_cost : ℝ := 5 * pen_cost

def notebook_cost : ℝ := 2 * pen_cost

def calculator_cost : ℝ := 3 * notebook_cost

def total_cost_before_tax : ℝ := pen_cost + briefcase_cost + notebook_cost + calculator_cost

def tax_rate : ℝ := 0.1

def tax_amount : ℝ := tax_rate * total_cost_before_tax

def total_cost_with_tax : ℝ := total_cost_before_tax + tax_amount

theorem marcel_shopping_cost : total_cost_with_tax = 61.60 := by sorry

end NUMINAMATH_CALUDE_marcel_shopping_cost_l3368_336853


namespace NUMINAMATH_CALUDE_count_six_digit_permutations_l3368_336867

/-- The number of different positive six-digit integers that can be formed using the digits 2, 2, 5, 5, 9, and 9 -/
def six_digit_permutations : ℕ :=
  Nat.factorial 6 / (Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 2)

/-- Theorem stating that the number of different positive six-digit integers
    that can be formed using the digits 2, 2, 5, 5, 9, and 9 is equal to 90 -/
theorem count_six_digit_permutations :
  six_digit_permutations = 90 := by
  sorry

end NUMINAMATH_CALUDE_count_six_digit_permutations_l3368_336867


namespace NUMINAMATH_CALUDE_rational_product_sum_integer_sum_product_max_rational_product_negative_sum_comparison_l3368_336851

theorem rational_product_sum (a b : ℚ) : 
  a * b = 6 → a + b ≠ 0 := by sorry

theorem integer_sum_product_max (a b : ℤ) : 
  a + b = -5 → a * b ≤ 6 := by sorry

theorem rational_product_negative_sum_comparison (a b : ℚ) : 
  a * b < 0 → 
  (∃ (x y : ℚ), x * y < 0 ∧ x + y < 0) ∧ 
  (∃ (x y : ℚ), x * y < 0 ∧ x + y = 0) ∧ 
  (∃ (x y : ℚ), x * y < 0 ∧ x + y > 0) := by sorry

end NUMINAMATH_CALUDE_rational_product_sum_integer_sum_product_max_rational_product_negative_sum_comparison_l3368_336851


namespace NUMINAMATH_CALUDE_pastry_distribution_combinations_l3368_336840

/-- The number of ways to distribute additional items among a subset of groups,
    given that each group already has one item. -/
def distribute_additional_items (total_items : ℕ) (total_groups : ℕ) (subset_groups : ℕ) : ℕ :=
  Nat.choose (subset_groups + (total_items - total_groups) - 1) (subset_groups - 1)

/-- Theorem stating that distributing 3 additional items among 4 groups,
    given that 5 items have already been distributed among 5 groups, results in 20 combinations. -/
theorem pastry_distribution_combinations :
  distribute_additional_items 8 5 4 = 20 := by
  sorry

end NUMINAMATH_CALUDE_pastry_distribution_combinations_l3368_336840


namespace NUMINAMATH_CALUDE_fraction_addition_l3368_336845

theorem fraction_addition : (1 : ℚ) / 3 + (-1 / 2) = -1 / 6 := by sorry

end NUMINAMATH_CALUDE_fraction_addition_l3368_336845


namespace NUMINAMATH_CALUDE_lcm_of_ratio_and_hcf_l3368_336834

theorem lcm_of_ratio_and_hcf (a b : ℕ) (h_ratio : a * 3 = b * 2) (h_hcf : Nat.gcd a b = 6) : Nat.lcm a b = 36 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_ratio_and_hcf_l3368_336834


namespace NUMINAMATH_CALUDE_amount_ratio_l3368_336803

/-- Given three amounts a, b, and c in rupees, prove that the ratio of a to b is 3:1 -/
theorem amount_ratio (a b c : ℕ) : 
  a + b + c = 645 →
  b = c + 25 →
  b = 134 →
  a / b = 3 := by
  sorry

end NUMINAMATH_CALUDE_amount_ratio_l3368_336803


namespace NUMINAMATH_CALUDE_inequality_proof_l3368_336854

theorem inequality_proof (a b c : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) (h4 : a * b * c = 1) :
  (a^2 + 1) * (b^3 + 2) * (c^6 + 5) ≥ 36 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3368_336854


namespace NUMINAMATH_CALUDE_ones_digit_sum_powers_2011_l3368_336878

theorem ones_digit_sum_powers_2011 : ∃ n : ℕ, n < 10 ∧ (1^2011 + 2^2011 + 3^2011 + 4^2011 + 5^2011 + 6^2011 + 7^2011 + 8^2011 + 9^2011 + 10^2011) % 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ones_digit_sum_powers_2011_l3368_336878


namespace NUMINAMATH_CALUDE_water_mixture_percentage_l3368_336857

/-- Given a mixture of wine and water, calculate the new percentage of water after adding more water. -/
theorem water_mixture_percentage
  (total_volume : ℝ)
  (initial_water_percentage : ℝ)
  (added_water : ℝ)
  (h1 : total_volume = 120)
  (h2 : initial_water_percentage = 20)
  (h3 : added_water = 8) :
  let initial_water := total_volume * (initial_water_percentage / 100)
  let new_water := initial_water + added_water
  let new_total := total_volume + added_water
  new_water / new_total * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_water_mixture_percentage_l3368_336857


namespace NUMINAMATH_CALUDE_perfect_squares_problem_l3368_336844

theorem perfect_squares_problem :
  ¬∃ (x : ℝ), x^2 = 5^2025 ∧
  ∃ (a : ℝ), a^2 = 3^2024 ∧
  ∃ (b : ℝ), b^2 = 7^2026 ∧
  ∃ (c : ℝ), c^2 = 8^2027 ∧
  ∃ (d : ℝ), d^2 = 9^2028 :=
by sorry

end NUMINAMATH_CALUDE_perfect_squares_problem_l3368_336844


namespace NUMINAMATH_CALUDE_problem_solution_l3368_336880

theorem problem_solution : 
  (100.2 * 99.8 = 9999.96) ∧ (103^2 = 10609) := by sorry

end NUMINAMATH_CALUDE_problem_solution_l3368_336880


namespace NUMINAMATH_CALUDE_complex_pure_imaginary_l3368_336884

theorem complex_pure_imaginary (a : ℝ) : 
  (Complex.I * (a - 2 * Complex.I) + (2 : ℂ) * (a - 2 * Complex.I)).re = 0 → a = -1 :=
by sorry

end NUMINAMATH_CALUDE_complex_pure_imaginary_l3368_336884


namespace NUMINAMATH_CALUDE_both_selected_probability_l3368_336846

theorem both_selected_probability 
  (p_ram : ℝ) (p_ravi : ℝ) 
  (h_ram : p_ram = 1 / 7) 
  (h_ravi : p_ravi = 1 / 5) 
  (h_independent : True) -- Assuming independence
  : p_ram * p_ravi = 1 / 35 := by
  sorry

end NUMINAMATH_CALUDE_both_selected_probability_l3368_336846


namespace NUMINAMATH_CALUDE_firm_employee_count_l3368_336821

-- Define the initial number of Democrats and Republicans
def initial_democrats : ℕ := sorry
def initial_republicans : ℕ := sorry

-- Define the conditions
axiom condition1 : initial_democrats + 1 = initial_republicans - 1
axiom condition2 : initial_democrats + 4 = 2 * (initial_republicans - 4)

-- Define the total number of employees
def total_employees : ℕ := initial_democrats + initial_republicans

-- Theorem to prove
theorem firm_employee_count : total_employees = 18 := by
  sorry

end NUMINAMATH_CALUDE_firm_employee_count_l3368_336821


namespace NUMINAMATH_CALUDE_jackson_earned_thirty_dollars_l3368_336898

-- Define the rate of pay per hour
def pay_rate : ℝ := 5

-- Define the time spent on each chore
def vacuuming_time : ℝ := 2
def dish_washing_time : ℝ := 0.5
def bathroom_cleaning_time : ℝ := 3 * dish_washing_time

-- Define the number of times vacuuming is done
def vacuuming_repetitions : ℕ := 2

-- Calculate total chore time
def total_chore_time : ℝ :=
  vacuuming_time * vacuuming_repetitions + dish_washing_time + bathroom_cleaning_time

-- Calculate earned money
def earned_money : ℝ := total_chore_time * pay_rate

-- Theorem statement
theorem jackson_earned_thirty_dollars :
  earned_money = 30 :=
by sorry

end NUMINAMATH_CALUDE_jackson_earned_thirty_dollars_l3368_336898


namespace NUMINAMATH_CALUDE_average_marks_combined_classes_l3368_336804

theorem average_marks_combined_classes (n1 n2 : ℕ) (avg1 avg2 : ℚ) 
  (h1 : n1 = 30) (h2 : n2 = 50) (h3 : avg1 = 40) (h4 : avg2 = 90) :
  (n1 * avg1 + n2 * avg2) / (n1 + n2 : ℚ) = 71.25 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_combined_classes_l3368_336804


namespace NUMINAMATH_CALUDE_intersection_A_B_l3368_336823

def A : Set ℕ := {1, 2, 3, 4}

def B : Set ℕ := {y | ∃ x ∈ A, y = 3 * x - 2}

theorem intersection_A_B : A ∩ B = {1, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3368_336823


namespace NUMINAMATH_CALUDE_difference_of_squares_l3368_336890

theorem difference_of_squares (x y : ℝ) (h1 : x + y = 10) (h2 : x - y = 19) : x^2 - y^2 = 190 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3368_336890


namespace NUMINAMATH_CALUDE_window_area_ratio_l3368_336805

/-- Proves that for a rectangle with semicircles at either end, where the ratio of the length (AD)
    to the width (AB) is 3:2 and the width is 30 inches, the ratio of the area of the rectangle
    to the combined area of the semicircles is 6:π. -/
theorem window_area_ratio :
  let AB : ℝ := 30
  let AD : ℝ := (3/2) * AB
  let rectangle_area : ℝ := AD * AB
  let semicircle_radius : ℝ := AB / 2
  let semicircles_area : ℝ := π * semicircle_radius^2
  rectangle_area / semicircles_area = 6 / π :=
by sorry

end NUMINAMATH_CALUDE_window_area_ratio_l3368_336805


namespace NUMINAMATH_CALUDE_problem_solution_l3368_336876

theorem problem_solution (a b : ℝ) 
  (h1 : a^2 + 2*b = 0) 
  (h2 : |a^2 - 2*b| = 8) : 
  b + 2023 = 2021 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3368_336876


namespace NUMINAMATH_CALUDE_book_words_per_page_l3368_336814

theorem book_words_per_page :
  ∀ (words_per_page : ℕ),
    words_per_page ≤ 120 →
    (150 * words_per_page) % 221 = 210 →
    words_per_page = 48 := by
  sorry

end NUMINAMATH_CALUDE_book_words_per_page_l3368_336814


namespace NUMINAMATH_CALUDE_squared_sum_ge_double_product_l3368_336863

theorem squared_sum_ge_double_product (a b : ℝ) : a^2 + b^2 ≥ 2*a*b ∧ (a^2 + b^2 = 2*a*b ↔ a = b) := by
  sorry

end NUMINAMATH_CALUDE_squared_sum_ge_double_product_l3368_336863


namespace NUMINAMATH_CALUDE_divisibility_pairs_l3368_336848

theorem divisibility_pairs : 
  ∀ m n : ℕ+, 
    (∀ k : ℕ+, k ≤ n → m.val % k = 0) ∧ 
    (m.val % (n + 1) ≠ 0) ∧ 
    (m.val % (n + 2) ≠ 0) ∧ 
    (m.val % (n + 3) ≠ 0) →
    ((n = 1 ∧ Nat.gcd m.val 6 = 1) ∨ 
     (n = 2 ∧ Nat.gcd m.val 12 = 1)) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_pairs_l3368_336848


namespace NUMINAMATH_CALUDE_max_x_in_grid_l3368_336882

/-- Represents a 5x5 grid with X placements -/
def Grid := Fin 5 → Fin 5 → Bool

/-- Checks if three X's are in a row (horizontally, vertically, or diagonally) -/
def has_three_in_row (g : Grid) : Prop := sorry

/-- Checks if each row has at least one X -/
def each_row_has_x (g : Grid) : Prop := sorry

/-- Counts the number of X's in the grid -/
def count_x (g : Grid) : Nat := sorry

/-- Theorem: The maximum number of X's in a 5x5 grid without three in a row and at least one X per row is 10 -/
theorem max_x_in_grid : 
  ∀ g : Grid, 
  ¬has_three_in_row g → 
  each_row_has_x g → 
  count_x g ≤ 10 ∧ 
  ∃ g' : Grid, ¬has_three_in_row g' ∧ each_row_has_x g' ∧ count_x g' = 10 := by
  sorry

end NUMINAMATH_CALUDE_max_x_in_grid_l3368_336882


namespace NUMINAMATH_CALUDE_no_integer_solutions_specific_solution_l3368_336800

theorem no_integer_solutions : ¬∃ (m n : ℤ), m^3 + 6*m^2 + 5*m = 27*n^3 + 9*n^2 + 9*n + 1 := by
  sorry

-- Additional fact mentioned in the problem
theorem specific_solution : (31 * 26)^3 + 6*(31*26)^2 + 5*(31*26) = 27*n^3 + 9*n^2 + 9*n + 1 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_specific_solution_l3368_336800


namespace NUMINAMATH_CALUDE_factors_imply_absolute_value_l3368_336833

-- Define the polynomial
def p (h k x : ℝ) : ℝ := 3 * x^3 - h * x - 3 * k

-- Define the factors
def f₁ (x : ℝ) : ℝ := x + 3
def f₂ (x : ℝ) : ℝ := x - 2

-- Theorem statement
theorem factors_imply_absolute_value (h k : ℝ) :
  (∀ x, p h k x = 0 → f₁ x = 0 ∨ f₂ x = 0) →
  |3 * h - 4 * k| = 615 := by
  sorry

end NUMINAMATH_CALUDE_factors_imply_absolute_value_l3368_336833


namespace NUMINAMATH_CALUDE_determinant_equals_four_l3368_336842

/-- The determinant of a 2x2 matrix [[a, b], [c, d]] is ad - bc. -/
def det2x2 (a b c d : ℝ) : ℝ := a * d - b * c

/-- The matrix in question, parameterized by x. -/
def matrix (x : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![3*x, 2],
    ![x, 2*x]]

theorem determinant_equals_four (x : ℝ) : 
  det2x2 (3*x) 2 x (2*x) = 4 ↔ x = -2/3 ∨ x = 1 := by sorry

end NUMINAMATH_CALUDE_determinant_equals_four_l3368_336842


namespace NUMINAMATH_CALUDE_parabola_one_x_intercept_l3368_336895

/-- The parabola defined by x = -3y^2 + 2y + 3 has exactly one x-intercept. -/
theorem parabola_one_x_intercept : 
  ∃! x : ℝ, ∃ y : ℝ, x = -3 * y^2 + 2 * y + 3 ∧ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_parabola_one_x_intercept_l3368_336895


namespace NUMINAMATH_CALUDE_molecular_weight_N2O5H3P_l3368_336852

-- Define atomic weights
def atomic_weight_N : ℝ := 14.01
def atomic_weight_O : ℝ := 16.00
def atomic_weight_H : ℝ := 1.01
def atomic_weight_P : ℝ := 30.97

-- Define the molecular formula
def N_count : ℕ := 2
def O_count : ℕ := 5
def H_count : ℕ := 3
def P_count : ℕ := 1

-- Theorem statement
theorem molecular_weight_N2O5H3P :
  N_count * atomic_weight_N +
  O_count * atomic_weight_O +
  H_count * atomic_weight_H +
  P_count * atomic_weight_P = 142.02 := by
  sorry

end NUMINAMATH_CALUDE_molecular_weight_N2O5H3P_l3368_336852


namespace NUMINAMATH_CALUDE_kelly_head_start_l3368_336897

/-- The length of the race in meters -/
def race_length : ℝ := 100

/-- The distance by which Abel lost to Kelly in meters -/
def losing_distance : ℝ := 0.5

/-- The additional distance Abel needs to run to overtake Kelly in meters -/
def overtake_distance : ℝ := 19.9

/-- Kelly's head start in meters -/
def head_start : ℝ := race_length - (race_length - losing_distance - overtake_distance)

theorem kelly_head_start :
  head_start = 20.4 :=
by sorry

end NUMINAMATH_CALUDE_kelly_head_start_l3368_336897


namespace NUMINAMATH_CALUDE_painting_cost_cny_l3368_336825

/-- Exchange rate from US dollars to Namibian dollars -/
def usd_to_nad : ℚ := 8

/-- Exchange rate from US dollars to Chinese yuan -/
def usd_to_cny : ℚ := 7

/-- Cost of the painting in Namibian dollars -/
def painting_cost_nad : ℚ := 160

/-- Theorem stating the cost of the painting in Chinese yuan -/
theorem painting_cost_cny : 
  (painting_cost_nad / usd_to_nad) * usd_to_cny = 140 := by
  sorry

end NUMINAMATH_CALUDE_painting_cost_cny_l3368_336825


namespace NUMINAMATH_CALUDE_add_and_convert_to_base7_37_45_l3368_336830

/-- Converts a number from base 10 to base 7 -/
def toBase7 (n : ℕ) : ℕ := sorry

/-- Adds two numbers in base 10 and returns the result in base 7 -/
def addAndConvertToBase7 (a b : ℕ) : ℕ :=
  toBase7 (a + b)

theorem add_and_convert_to_base7_37_45 :
  addAndConvertToBase7 37 45 = 145 := by sorry

end NUMINAMATH_CALUDE_add_and_convert_to_base7_37_45_l3368_336830


namespace NUMINAMATH_CALUDE_proportional_function_and_point_l3368_336812

-- Define the function f
def f (x : ℝ) : ℝ := 4 * x + 2

-- State the theorem
theorem proportional_function_and_point :
  -- Conditions
  (∃ k : ℝ, ∀ x y : ℝ, y - 2 = k * x) →  -- y-2 is directly proportional to x
  (f 1 = 6) →                           -- When x=1, y=6
  (f (-3/4) = -1) →                     -- Point P(a, -1) is on the graph of the function
  -- Conclusions
  ((∀ x : ℝ, f x = 4 * x + 2) ∧         -- The function expression is y = 4x + 2
   (-3/4 : ℝ) = -3/4)                   -- The value of a is -3/4
  := by sorry

end NUMINAMATH_CALUDE_proportional_function_and_point_l3368_336812


namespace NUMINAMATH_CALUDE_max_value_of_sin_cos_product_l3368_336850

theorem max_value_of_sin_cos_product (f : ℝ → ℝ) (α : ℝ) :
  (∀ x, f x = Real.sin (x + α) * Real.cos (x + α)) →
  (∀ x, f x ≤ f 1) →
  ∃ k : ℤ, α = Real.pi / 4 + k * Real.pi / 2 - 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_sin_cos_product_l3368_336850


namespace NUMINAMATH_CALUDE_sequence_length_730_l3368_336839

/-- Given a sequence of real numbers satisfying certain conditions, prove that the length of the sequence is 730. -/
theorem sequence_length_730 (n : ℕ+) (b : ℕ → ℝ) : 
  b 0 = 45 → 
  b 1 = 81 → 
  b n = 0 → 
  (∀ k : ℕ, 1 ≤ k ∧ k < n → b (k + 1) = b (k - 1) - 5 / b k) → 
  n = 730 := by
  sorry

end NUMINAMATH_CALUDE_sequence_length_730_l3368_336839


namespace NUMINAMATH_CALUDE_existsShapeWithCircularTopView_multipleShapesWithCircularTopView_l3368_336841

-- Define a type for 3D shapes
inductive Shape3D
  | Sphere
  | Cylinder
  | Cone
  | Frustum
  | Other

-- Define a property for having a circular top view
def hasCircularTopView (s : Shape3D) : Prop :=
  match s with
  | Shape3D.Sphere => True
  | Shape3D.Cylinder => True
  | Shape3D.Cone => True
  | Shape3D.Frustum => True
  | Shape3D.Other => False

-- Theorem stating that there exist shapes with circular top views
theorem existsShapeWithCircularTopView : ∃ (s : Shape3D), hasCircularTopView s :=
  sorry

-- Theorem stating that multiple shapes can have circular top views
theorem multipleShapesWithCircularTopView :
  ∃ (s1 s2 : Shape3D), s1 ≠ s2 ∧ hasCircularTopView s1 ∧ hasCircularTopView s2 :=
  sorry

end NUMINAMATH_CALUDE_existsShapeWithCircularTopView_multipleShapesWithCircularTopView_l3368_336841


namespace NUMINAMATH_CALUDE_y_investment_l3368_336809

/-- Represents the investment and profit share of a person in a business. -/
structure Investor where
  investment : ℕ
  profitShare : ℕ

/-- Represents the business with three investors. -/
structure Business where
  x : Investor
  y : Investor
  z : Investor

/-- The theorem stating that given the conditions, y's investment is 15000 rupees. -/
theorem y_investment (b : Business) : 
  b.x.investment = 5000 ∧ 
  b.z.investment = 7000 ∧ 
  b.x.profitShare = 2 ∧ 
  b.y.profitShare = 6 ∧ 
  b.z.profitShare = 7 → 
  b.y.investment = 15000 :=
by sorry

end NUMINAMATH_CALUDE_y_investment_l3368_336809


namespace NUMINAMATH_CALUDE_double_up_polynomial_properties_l3368_336810

/-- A double-up polynomial is a quadratic polynomial with two real roots, one of which is twice the other. -/
def DoubleUpPolynomial (p q : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ (k^2 + p*k + q = 0) ∧ ((2*k)^2 + p*(2*k) + q = 0)

theorem double_up_polynomial_properties :
  (∀ p q : ℝ, DoubleUpPolynomial p q →
    (p = -15 → q = 50) ∧
    (∃ k : ℝ, (k = 4 ∨ k = 2) → p + q = 20 ∨ p + q = 2) ∧
    (p + q = 9 → ∃ k : ℝ, k = 3 ∨ k = -3/2)) := by
  sorry

end NUMINAMATH_CALUDE_double_up_polynomial_properties_l3368_336810


namespace NUMINAMATH_CALUDE_min_links_remove_10x10_grid_l3368_336871

/-- Represents a grid with horizontal and vertical lines -/
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)
  (horizontal_lines : ℕ)
  (vertical_lines : ℕ)

/-- Calculates the total number of links in the grid -/
def total_links (g : Grid) : ℕ :=
  (g.rows * g.vertical_lines) + (g.cols * g.horizontal_lines)

/-- Calculates the number of interior nodes in the grid -/
def interior_nodes (g : Grid) : ℕ :=
  (g.rows - 1) * (g.cols - 1)

/-- The minimum number of links to remove -/
def min_links_to_remove (g : Grid) : ℕ := 41

/-- Theorem stating the minimum number of links to remove for a 10x10 grid -/
theorem min_links_remove_10x10_grid :
  let g : Grid := { rows := 10, cols := 10, horizontal_lines := 11, vertical_lines := 11 }
  min_links_to_remove g = 41 :=
by sorry

end NUMINAMATH_CALUDE_min_links_remove_10x10_grid_l3368_336871


namespace NUMINAMATH_CALUDE_cos_double_angle_given_tan_l3368_336837

theorem cos_double_angle_given_tan (α : Real) (h : Real.tan α = 3) : 
  Real.cos (2 * α) = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_double_angle_given_tan_l3368_336837


namespace NUMINAMATH_CALUDE_cylinder_radius_l3368_336888

/-- 
Theorem: For a cylinder with an original height of 3 inches, 
if increasing either the radius or the height by 7 inches results in the same volume, 
then the original radius must be 7 inches.
-/
theorem cylinder_radius (r : ℝ) : 
  r > 0 →  -- radius must be positive
  3 * π * (r + 7)^2 = 10 * π * r^2 → -- volumes are equal
  r = 7 := by
sorry

end NUMINAMATH_CALUDE_cylinder_radius_l3368_336888


namespace NUMINAMATH_CALUDE_jen_buys_50_candy_bars_l3368_336883

/-- The number of candy bars Jen buys -/
def num_candy_bars : ℕ := 50

/-- The cost of buying each candy bar in cents -/
def buy_price : ℕ := 80

/-- The selling price of each candy bar in cents -/
def sell_price : ℕ := 100

/-- The number of candy bars Jen sells -/
def num_sold : ℕ := 48

/-- Jen's profit in cents -/
def profit : ℕ := 800

/-- Theorem stating that given the conditions, Jen buys 50 candy bars -/
theorem jen_buys_50_candy_bars :
  (sell_price * num_sold) - (buy_price * num_candy_bars) = profit :=
by sorry

end NUMINAMATH_CALUDE_jen_buys_50_candy_bars_l3368_336883


namespace NUMINAMATH_CALUDE_segment_length_unit_circle_l3368_336889

/-- The length of the segment cut by a unit circle from the line y - x = 1 is √2. -/
theorem segment_length_unit_circle : ∃ (L : ℝ), L = Real.sqrt 2 ∧ 
  ∀ (x y : ℝ), x^2 + y^2 = 1 ∧ y - x = 1 → 
  ∃ (x' y' : ℝ), x'^2 + y'^2 = 1 ∧ y' - x' = 1 ∧ 
  Real.sqrt ((x - x')^2 + (y - y')^2) = L :=
sorry

end NUMINAMATH_CALUDE_segment_length_unit_circle_l3368_336889


namespace NUMINAMATH_CALUDE_solve_cab_driver_problem_l3368_336892

def cab_driver_problem (day1 day2 day4 day5 average : ℕ) : Prop :=
  let total := 5 * average
  let known_sum := day1 + day2 + day4 + day5
  let day3 := total - known_sum
  (day1 = 300) ∧ (day2 = 150) ∧ (day4 = 400) ∧ (day5 = 500) ∧ (average = 420) → day3 = 750

theorem solve_cab_driver_problem :
  cab_driver_problem 300 150 400 500 420 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_cab_driver_problem_l3368_336892


namespace NUMINAMATH_CALUDE_average_transformation_l3368_336860

theorem average_transformation (a₁ a₂ a₃ a₄ a₅ : ℝ) 
  (h : (a₁ + a₂ + a₃ + a₄ + a₅) / 5 = 8) : 
  ((a₁ + 10) + (a₂ - 10) + (a₃ + 10) + (a₄ - 10) + (a₅ + 10)) / 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_average_transformation_l3368_336860


namespace NUMINAMATH_CALUDE_min_value_theorem_l3368_336859

theorem min_value_theorem (p q r s t u : ℝ) 
  (pos_p : p > 0) (pos_q : q > 0) (pos_r : r > 0) 
  (pos_s : s > 0) (pos_t : t > 0) (pos_u : u > 0)
  (sum_eq : p + q + r + s + t + u = 11) :
  (3/p + 12/q + 27/r + 48/s + 75/t + 108/u) ≥ 819/11 ∧ 
  ∃ (p' q' r' s' t' u' : ℝ), 
    p' > 0 ∧ q' > 0 ∧ r' > 0 ∧ s' > 0 ∧ t' > 0 ∧ u' > 0 ∧
    p' + q' + r' + s' + t' + u' = 11 ∧
    (3/p' + 12/q' + 27/r' + 48/s' + 75/t' + 108/u') = 819/11 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3368_336859


namespace NUMINAMATH_CALUDE_sqrt_simplification_l3368_336811

theorem sqrt_simplification (a : ℝ) (ha : a > 0) : a^2 * Real.sqrt a = a^(5/2) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_simplification_l3368_336811


namespace NUMINAMATH_CALUDE_problem_statement_l3368_336875

theorem problem_statement (n : ℝ) : 
  (n - 2009)^2 + (2008 - n)^2 = 1 → (n - 2009) * (2008 - n) = 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3368_336875


namespace NUMINAMATH_CALUDE_square_sum_given_diff_and_product_l3368_336828

theorem square_sum_given_diff_and_product (a b : ℝ) 
  (h1 : a - b = 2) 
  (h2 : a * b = 3) : 
  (a + b)^2 = 16 := by
sorry

end NUMINAMATH_CALUDE_square_sum_given_diff_and_product_l3368_336828


namespace NUMINAMATH_CALUDE_crayons_in_drawer_l3368_336872

/-- The number of crayons remaining in a drawer after some are removed. -/
def crayons_remaining (initial : ℕ) (removed : ℕ) : ℕ :=
  initial - removed

/-- Theorem: Given 7 initial crayons and 3 removed, 4 crayons remain. -/
theorem crayons_in_drawer : crayons_remaining 7 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_crayons_in_drawer_l3368_336872


namespace NUMINAMATH_CALUDE_percentage_problem_l3368_336832

theorem percentage_problem (x : ℝ) : 200 = 4 * x → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3368_336832


namespace NUMINAMATH_CALUDE_cookies_left_l3368_336885

/-- The number of cookies in a dozen -/
def cookies_per_dozen : ℕ := 12

/-- The number of dozens of cookies Meena bakes -/
def dozens_baked : ℕ := 5

/-- The number of dozens of cookies Mr. Stone buys -/
def dozens_sold_to_stone : ℕ := 2

/-- The number of cookies Brock buys -/
def cookies_sold_to_brock : ℕ := 7

/-- Calculates the total number of cookies Meena bakes -/
def total_cookies_baked : ℕ := dozens_baked * cookies_per_dozen

/-- Calculates the number of cookies sold to Mr. Stone -/
def cookies_sold_to_stone : ℕ := dozens_sold_to_stone * cookies_per_dozen

/-- Calculates the number of cookies sold to Katy -/
def cookies_sold_to_katy : ℕ := 2 * cookies_sold_to_brock

/-- Calculates the total number of cookies sold -/
def total_cookies_sold : ℕ := cookies_sold_to_stone + cookies_sold_to_brock + cookies_sold_to_katy

/-- Theorem: Meena has 15 cookies left after selling to Mr. Stone, Brock, and Katy -/
theorem cookies_left : total_cookies_baked - total_cookies_sold = 15 := by
  sorry

end NUMINAMATH_CALUDE_cookies_left_l3368_336885


namespace NUMINAMATH_CALUDE_sarah_walked_4_6_miles_l3368_336815

/-- The distance Sarah walked in miles -/
def sarah_distance (blocks_west : ℕ) (blocks_south : ℕ) (miles_per_block : ℚ) : ℚ :=
  (blocks_west + blocks_south : ℚ) * miles_per_block

/-- Theorem stating that Sarah walked 4.6 miles -/
theorem sarah_walked_4_6_miles :
  sarah_distance 8 15 (1/5) = 46/10 := by
  sorry

end NUMINAMATH_CALUDE_sarah_walked_4_6_miles_l3368_336815


namespace NUMINAMATH_CALUDE_house_painting_cost_l3368_336881

/-- Calculates the cost of painting a house given its area and price per square foot. -/
def paintingCost (area : ℝ) (pricePerSqFt : ℝ) : ℝ :=
  area * pricePerSqFt

/-- Proves that the cost of painting a house with an area of 484 sq ft
    at a rate of Rs. 20 per sq ft is equal to Rs. 9,680. -/
theorem house_painting_cost :
  paintingCost 484 20 = 9680 := by
  sorry

end NUMINAMATH_CALUDE_house_painting_cost_l3368_336881


namespace NUMINAMATH_CALUDE_real_part_of_z_l3368_336868

theorem real_part_of_z (z : ℂ) (h : z * (1 + Complex.I) = 2) : 
  Complex.re z = 1 := by sorry

end NUMINAMATH_CALUDE_real_part_of_z_l3368_336868


namespace NUMINAMATH_CALUDE_equation_solution_l3368_336818

theorem equation_solution :
  let f : ℝ → ℝ := λ x => (20 / (x^2 - 9)) + (5 / (x - 3))
  ∀ x : ℝ, f x = 2 ↔ x = (5 + Real.sqrt 449) / 4 ∨ x = (5 - Real.sqrt 449) / 4 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3368_336818


namespace NUMINAMATH_CALUDE_mens_wages_are_fifty_l3368_336819

/-- Represents the wages of a group given the number of individuals and their equality relationships -/
def group_wages (men women boys : ℕ) (total_earnings : ℚ) : ℚ :=
  total_earnings / (men + women + boys : ℚ) * men

/-- Theorem stating that under given conditions, the men's wages are 50 -/
theorem mens_wages_are_fifty
  (men : ℕ) (women : ℕ) (boys : ℕ) (total_earnings : ℚ)
  (h1 : men = 5)
  (h2 : boys = 8)
  (h3 : men = women)  -- 5 men are equal to W women
  (h4 : women = boys) -- W women are equal to 8 boys
  (h5 : total_earnings = 150) :
  group_wages men women boys total_earnings = 50 := by
  sorry

#eval group_wages 5 5 8 150

end NUMINAMATH_CALUDE_mens_wages_are_fifty_l3368_336819


namespace NUMINAMATH_CALUDE_cycle_price_calculation_l3368_336862

/-- Proves that a cycle sold at a 25% loss for 1050 had an original price of 1400 -/
theorem cycle_price_calculation (selling_price : ℝ) (loss_percentage : ℝ) 
  (h1 : selling_price = 1050)
  (h2 : loss_percentage = 25) : 
  ∃ original_price : ℝ, 
    original_price * (1 - loss_percentage / 100) = selling_price ∧ 
    original_price = 1400 := by
  sorry

end NUMINAMATH_CALUDE_cycle_price_calculation_l3368_336862


namespace NUMINAMATH_CALUDE_preimage_of_3_1_l3368_336838

/-- The mapping f: ℝ² → ℝ² defined by f(x,y) = (x+y, x-y) -/
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + p.2, p.1 - p.2)

/-- Theorem stating that (2,1) is the pre-image of (3,1) under the mapping f -/
theorem preimage_of_3_1 : f (2, 1) = (3, 1) := by sorry

end NUMINAMATH_CALUDE_preimage_of_3_1_l3368_336838


namespace NUMINAMATH_CALUDE_new_alloy_aluminum_bounds_l3368_336866

/-- Represents the composition of an alloy -/
structure AlloyComposition where
  aluminum : ℝ
  copper : ℝ
  magnesium : ℝ

/-- Given three alloys and their compositions, proves that a new alloy with 20% copper
    made from these alloys will have an aluminum percentage between 15% and 40% -/
theorem new_alloy_aluminum_bounds 
  (alloy1 : AlloyComposition)
  (alloy2 : AlloyComposition)
  (alloy3 : AlloyComposition)
  (h1 : alloy1.aluminum = 0.6 ∧ alloy1.copper = 0.15 ∧ alloy1.magnesium = 0.25)
  (h2 : alloy2.aluminum = 0 ∧ alloy2.copper = 0.3 ∧ alloy2.magnesium = 0.7)
  (h3 : alloy3.aluminum = 0.45 ∧ alloy3.copper = 0 ∧ alloy3.magnesium = 0.55)
  : ∃ (x1 x2 x3 : ℝ), 
    x1 + x2 + x3 = 1 ∧
    0.15 * x1 + 0.3 * x2 = 0.2 ∧
    0.15 ≤ 0.6 * x1 + 0.45 * x3 ∧
    0.6 * x1 + 0.45 * x3 ≤ 0.4 :=
by sorry

end NUMINAMATH_CALUDE_new_alloy_aluminum_bounds_l3368_336866


namespace NUMINAMATH_CALUDE_bike_shop_wheels_l3368_336806

/-- The number of wheels on all vehicles in a bike shop -/
def total_wheels (num_bicycles num_tricycles : ℕ) : ℕ :=
  2 * num_bicycles + 3 * num_tricycles

/-- Theorem stating that the total number of wheels from 50 bicycles and 20 tricycles is 160 -/
theorem bike_shop_wheels :
  total_wheels 50 20 = 160 := by
  sorry

end NUMINAMATH_CALUDE_bike_shop_wheels_l3368_336806


namespace NUMINAMATH_CALUDE_magnitude_of_vector_sum_l3368_336896

/-- The magnitude of the sum of vectors (1, √3) and (-2, 0) is 2 -/
theorem magnitude_of_vector_sum : 
  let a : Fin 2 → ℝ := ![1, Real.sqrt 3]
  let b : Fin 2 → ℝ := ![-2, 0]
  Real.sqrt ((a 0 + b 0)^2 + (a 1 + b 1)^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_vector_sum_l3368_336896


namespace NUMINAMATH_CALUDE_item_value_proof_l3368_336870

/-- Proves that the total value of an item is $2,590 given the import tax conditions -/
theorem item_value_proof (tax_rate : ℝ) (tax_threshold : ℝ) (tax_paid : ℝ) :
  tax_rate = 0.07 →
  tax_threshold = 1000 →
  tax_paid = 111.30 →
  ∃ (total_value : ℝ), 
    tax_rate * (total_value - tax_threshold) = tax_paid ∧
    total_value = 2590 := by
  sorry

end NUMINAMATH_CALUDE_item_value_proof_l3368_336870


namespace NUMINAMATH_CALUDE_system_of_equations_sum_l3368_336886

theorem system_of_equations_sum (a b c x y z : ℝ) 
  (eq1 : 17 * x + b * y + c * z = 0)
  (eq2 : a * x + 29 * y + c * z = 0)
  (eq3 : a * x + b * y + 53 * z = 0)
  (ha : a ≠ 17)
  (hx : x ≠ 0) :
  a / (a - 17) + b / (b - 29) + c / (c - 53) = 1 := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_sum_l3368_336886


namespace NUMINAMATH_CALUDE_union_A_B_when_a_is_one_value_set_of_a_when_intersection_empty_l3368_336856

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | 0 < a * x - 1 ∧ a * x - 1 ≤ 5}
def B : Set ℝ := {x | -1/2 < x ∧ x ≤ 2}

-- Theorem for part I
theorem union_A_B_when_a_is_one :
  A 1 ∪ B = {x : ℝ | -1/2 < x ∧ x ≤ 6} := by sorry

-- Theorem for part II
theorem value_set_of_a_when_intersection_empty :
  ∀ a : ℝ, a ≥ 0 → (A a ∩ B = ∅ ↔ 0 ≤ a ∧ a ≤ 1/2) := by sorry

end NUMINAMATH_CALUDE_union_A_B_when_a_is_one_value_set_of_a_when_intersection_empty_l3368_336856


namespace NUMINAMATH_CALUDE_robin_gum_total_l3368_336847

theorem robin_gum_total (initial : ℕ) (additional : ℕ) (total : ℕ) : 
  initial = 18 → additional = 26 → total = initial + additional → total = 44 := by
  sorry

end NUMINAMATH_CALUDE_robin_gum_total_l3368_336847


namespace NUMINAMATH_CALUDE_triangle_side_values_l3368_336865

theorem triangle_side_values (A B C : ℝ) (a b c : ℝ) : 
  A = 30 * π / 180 →  -- Convert 30° to radians
  a = 1 →
  c = Real.sqrt 3 →
  (a^2 = b^2 + c^2 - 2*b*c*(Real.cos A)) →  -- Law of Cosines
  (b = 1 ∨ b = 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_values_l3368_336865


namespace NUMINAMATH_CALUDE_eighth_diagram_shaded_fraction_l3368_336887

/-- The number of shaded triangles in the nth diagram (n ≥ 1) -/
def shaded (n : ℕ) : ℕ := (n - 1) * n / 2

/-- The total number of small triangles in the nth diagram -/
def total (n : ℕ) : ℕ := n ^ 2

/-- The fraction of shaded triangles in the nth diagram -/
def shaded_fraction (n : ℕ) : ℚ := shaded n / total n

theorem eighth_diagram_shaded_fraction :
  shaded_fraction 8 = 7 / 16 := by sorry

end NUMINAMATH_CALUDE_eighth_diagram_shaded_fraction_l3368_336887


namespace NUMINAMATH_CALUDE_three_hundredth_term_of_sequence_l3368_336861

def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₁ * r^(n - 1)

theorem three_hundredth_term_of_sequence (a₁ a₂ : ℝ) (h₁ : a₁ = 8) (h₂ : a₂ = -8) :
  geometric_sequence a₁ (a₂ / a₁) 300 = -8 := by
  sorry

end NUMINAMATH_CALUDE_three_hundredth_term_of_sequence_l3368_336861


namespace NUMINAMATH_CALUDE_average_apples_picked_l3368_336836

theorem average_apples_picked (maggie kelsey layla : ℕ) (h1 : maggie = 40) (h2 : kelsey = 28) (h3 : layla = 22) :
  (maggie + kelsey + layla) / 3 = 30 := by
  sorry

end NUMINAMATH_CALUDE_average_apples_picked_l3368_336836


namespace NUMINAMATH_CALUDE_gcd_problem_l3368_336820

theorem gcd_problem (a b c : ℕ) : 
  a * b * c = 2^4 * 3^2 * 5^3 →
  Nat.gcd a b = 15 →
  Nat.gcd a c = 5 →
  Nat.gcd b c = 20 →
  (a = 15 ∧ b = 60 ∧ c = 20) := by
sorry

end NUMINAMATH_CALUDE_gcd_problem_l3368_336820


namespace NUMINAMATH_CALUDE_sector_max_area_l3368_336877

/-- A sector is defined by its radius and central angle. -/
structure Sector where
  radius : ℝ
  angle : ℝ

/-- The perimeter of a sector. -/
def perimeter (s : Sector) : ℝ := s.radius * s.angle + 2 * s.radius

/-- The area of a sector. -/
def area (s : Sector) : ℝ := 0.5 * s.radius^2 * s.angle

/-- Theorem: For a sector with perimeter 4, the area is maximized when the central angle is 2 radians. -/
theorem sector_max_area (s : Sector) (h : perimeter s = 4) :
  area s ≤ area { radius := 1, angle := 2 } := by
  sorry

#check sector_max_area

end NUMINAMATH_CALUDE_sector_max_area_l3368_336877


namespace NUMINAMATH_CALUDE_cards_nell_has_left_nell_remaining_cards_l3368_336801

/-- Calculates the number of cards Nell has left after giving some to Jeff -/
theorem cards_nell_has_left (nell_initial : ℕ) (jeff_initial : ℕ) (jeff_final : ℕ) : ℕ :=
  let cards_transferred := jeff_final - jeff_initial
  nell_initial - cards_transferred

/-- Proves that Nell has 252 cards left after giving some to Jeff -/
theorem nell_remaining_cards : 
  cards_nell_has_left 528 11 287 = 252 := by
  sorry


end NUMINAMATH_CALUDE_cards_nell_has_left_nell_remaining_cards_l3368_336801


namespace NUMINAMATH_CALUDE_company_growth_rate_l3368_336864

/-- Represents the yearly capital growth rate as a real number between 0 and 1 -/
def yearly_growth_rate : ℝ := sorry

/-- The initial loan amount in ten thousands of yuan -/
def initial_loan : ℝ := 200

/-- The loan duration in years -/
def loan_duration : ℕ := 2

/-- The annual interest rate as a real number between 0 and 1 -/
def interest_rate : ℝ := 0.08

/-- The surplus after repayment in ten thousands of yuan -/
def surplus : ℝ := 72

theorem company_growth_rate :
  (initial_loan * (1 + yearly_growth_rate) ^ loan_duration) =
  (initial_loan * (1 + interest_rate) ^ loan_duration + surplus) ∧
  yearly_growth_rate = 0.2 := by sorry

end NUMINAMATH_CALUDE_company_growth_rate_l3368_336864


namespace NUMINAMATH_CALUDE_max_value_g_in_unit_interval_l3368_336808

-- Define the function g(x)
def g (x : ℝ) : ℝ := x * (x^2 - 1)

-- State the theorem
theorem max_value_g_in_unit_interval :
  ∃ (M : ℝ), M = 0 ∧ ∀ x, x ∈ Set.Icc 0 1 → g x ≤ M :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_g_in_unit_interval_l3368_336808


namespace NUMINAMATH_CALUDE_rectangle_area_is_72_l3368_336849

/-- Represents a circle with a center point and radius -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents a rectangle with four vertices -/
structure Rectangle where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Checks if three circles are congruent -/
def areCongruent (c1 c2 c3 : Circle) : Prop :=
  c1.radius = c2.radius ∧ c2.radius = c3.radius

/-- Checks if a circle is tangent to all sides of a rectangle -/
def isTangentToRectangle (c : Circle) (r : Rectangle) : Prop :=
  sorry

/-- Checks if a circle passes through two points -/
def passesThrough (c : Circle) (p1 p2 : Point) : Prop :=
  sorry

/-- Calculates the area of a rectangle -/
def rectangleArea (r : Rectangle) : ℝ :=
  sorry

theorem rectangle_area_is_72 
  (ABCD : Rectangle) (P Q R : Point) (circleP circleQ circleR : Circle) :
  circleP.center = P →
  circleQ.center = Q →
  circleR.center = R →
  areCongruent circleP circleQ circleR →
  isTangentToRectangle circleP ABCD →
  isTangentToRectangle circleQ ABCD →
  isTangentToRectangle circleR ABCD →
  circleQ.radius = 3 →
  passesThrough circleQ P R →
  rectangleArea ABCD = 72 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_is_72_l3368_336849


namespace NUMINAMATH_CALUDE_comic_book_stacking_l3368_336873

theorem comic_book_stacking (batman : ℕ) (xmen : ℕ) (calvin_hobbes : ℕ) :
  batman = 5 →
  xmen = 4 →
  calvin_hobbes = 3 →
  (Nat.factorial batman * Nat.factorial xmen * Nat.factorial calvin_hobbes) *
  Nat.factorial 3 = 103680 :=
by sorry

end NUMINAMATH_CALUDE_comic_book_stacking_l3368_336873
