import Mathlib

namespace ratio_problem_l2921_292104

theorem ratio_problem (a b c : ℝ) (h1 : a / b = 6 / 5) (h2 : b / c = 8 / 7) :
  (7 * a + 6 * b + 5 * c) / (7 * a - 6 * b + 5 * c) = 751 / 271 := by
  sorry

end ratio_problem_l2921_292104


namespace imaginary_part_of_z_over_one_plus_i_l2921_292144

theorem imaginary_part_of_z_over_one_plus_i :
  ∀ (z : ℂ), z = 1 - 2 * I →
  (z / (1 + I)).im = -3/2 := by
sorry

end imaginary_part_of_z_over_one_plus_i_l2921_292144


namespace standard_deviation_calculation_l2921_292165

/-- A normal distribution with mean μ and standard deviation σ -/
structure NormalDistribution where
  μ : ℝ
  σ : ℝ

/-- The value that is exactly k standard deviations away from the mean -/
def value_k_std_dev_from_mean (d : NormalDistribution) (k : ℝ) : ℝ :=
  d.μ - k * d.σ

theorem standard_deviation_calculation (d : NormalDistribution) 
  (h1 : d.μ = 16.2)
  (h2 : value_k_std_dev_from_mean d 2 = 11.6) : 
  d.σ = 2.3 := by
sorry

end standard_deviation_calculation_l2921_292165


namespace base_conversion_1987_to_base5_l2921_292174

/-- Converts a list of digits in base 5 to a natural number in base 10 -/
def fromBase5 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Theorem: 1987 in base 10 is equal to 30422 in base 5 -/
theorem base_conversion_1987_to_base5 :
  1987 = fromBase5 [2, 2, 4, 0, 3] := by sorry

end base_conversion_1987_to_base5_l2921_292174


namespace zongzi_purchase_problem_l2921_292146

/-- Represents the cost and quantity information for zongzi purchases -/
structure ZongziPurchase where
  cost_A : ℝ  -- Cost per bag of brand A zongzi
  cost_B : ℝ  -- Cost per bag of brand B zongzi
  quantity_A : ℕ  -- Quantity of brand A zongzi
  quantity_B : ℕ  -- Quantity of brand B zongzi
  total_cost : ℝ  -- Total cost of the purchase

/-- Theorem representing the zongzi purchase problem -/
theorem zongzi_purchase_problem 
  (purchase1 : ZongziPurchase)
  (purchase2 : ZongziPurchase)
  (h1 : purchase1.quantity_A = 100 ∧ purchase1.quantity_B = 150 ∧ purchase1.total_cost = 7000)
  (h2 : purchase2.quantity_A = 180 ∧ purchase2.quantity_B = 120 ∧ purchase2.total_cost = 8100)
  (h3 : purchase1.cost_A = purchase2.cost_A ∧ purchase1.cost_B = purchase2.cost_B) :
  ∃ (optimal_purchase : ZongziPurchase),
    purchase1.cost_A = 25 ∧
    purchase1.cost_B = 30 ∧
    optimal_purchase.quantity_A = 200 ∧
    optimal_purchase.quantity_B = 100 ∧
    optimal_purchase.total_cost = 8000 ∧
    optimal_purchase.quantity_A + optimal_purchase.quantity_B = 300 ∧
    optimal_purchase.quantity_A ≤ 2 * optimal_purchase.quantity_B ∧
    ∀ (other_purchase : ZongziPurchase),
      other_purchase.quantity_A + other_purchase.quantity_B = 300 →
      other_purchase.quantity_A ≤ 2 * other_purchase.quantity_B →
      other_purchase.total_cost ≥ optimal_purchase.total_cost := by
  sorry


end zongzi_purchase_problem_l2921_292146


namespace sin_390_degrees_l2921_292129

theorem sin_390_degrees : Real.sin (390 * π / 180) = 1 / 2 := by
  sorry

end sin_390_degrees_l2921_292129


namespace least_positive_integer_divisible_by_four_primes_l2921_292130

theorem least_positive_integer_divisible_by_four_primes : 
  ∃ (p₁ p₂ p₃ p₄ : Nat), Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ 
  p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄ ∧
  (∀ n : Nat, n > 0 → (∃ (q₁ q₂ q₃ q₄ : Nat), Prime q₁ ∧ Prime q₂ ∧ Prime q₃ ∧ Prime q₄ ∧
    q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₁ ≠ q₄ ∧ q₂ ≠ q₃ ∧ q₂ ≠ q₄ ∧ q₃ ≠ q₄ ∧
    n % q₁ = 0 ∧ n % q₂ = 0 ∧ n % q₃ = 0 ∧ n % q₄ = 0) → n ≥ 210) ∧
  210 % p₁ = 0 ∧ 210 % p₂ = 0 ∧ 210 % p₃ = 0 ∧ 210 % p₄ = 0 :=
by sorry

end least_positive_integer_divisible_by_four_primes_l2921_292130


namespace count_less_equal_04_l2921_292163

def count_less_equal (threshold : ℚ) (numbers : List ℚ) : ℕ :=
  (numbers.filter (λ x => x ≤ threshold)).length

theorem count_less_equal_04 : count_less_equal (4/10) [8/10, 1/2, 3/10] = 1 := by
  sorry

end count_less_equal_04_l2921_292163


namespace equation_represents_statement_l2921_292198

/-- Represents an unknown number -/
def n : ℤ := sorry

/-- The statement "a number increased by five equals 15" -/
def statement : Prop := n + 5 = 15

/-- Theorem stating that the equation correctly represents the given statement -/
theorem equation_represents_statement : statement ↔ n + 5 = 15 := by sorry

end equation_represents_statement_l2921_292198


namespace polynomial_sum_theorem_l2921_292111

theorem polynomial_sum_theorem (d : ℝ) :
  let f : ℝ → ℝ := λ x => 15 * x^3 + 17 * x + 18 + 19 * x^2
  let g : ℝ → ℝ := λ x => 3 * x^3 + 4 * x + 2
  ∃ (p q r s : ℤ),
    (∀ x, f x + g x = p * x^3 + q * x^2 + r * x + s) ∧
    p + q + r + s = 78 :=
by
  sorry

end polynomial_sum_theorem_l2921_292111


namespace reptiles_per_swamp_l2921_292138

theorem reptiles_per_swamp (total_reptiles : ℕ) (num_swamps : ℕ) 
  (h1 : total_reptiles = 1424) (h2 : num_swamps = 4) :
  total_reptiles / num_swamps = 356 := by
  sorry

end reptiles_per_swamp_l2921_292138


namespace bridge_length_l2921_292186

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length 
  (train_length : ℝ) 
  (train_speed_kmh : ℝ) 
  (crossing_time_s : ℝ) : 
  train_length = 160 →
  train_speed_kmh = 45 →
  crossing_time_s = 30 →
  (train_speed_kmh * 1000 / 3600 * crossing_time_s) - train_length = 215 := by
  sorry

end bridge_length_l2921_292186


namespace wine_pouring_equivalence_l2921_292101

/-- Represents the state of the four glasses --/
structure GlassState :=
  (glass1 : ℕ)
  (glass2 : ℕ)
  (glass3 : ℕ)
  (glass4 : ℕ)

/-- Represents a single pouring operation --/
inductive PourOperation
  | pour1to2
  | pour1to3
  | pour1to4
  | pour2to1
  | pour2to3
  | pour2to4
  | pour3to1
  | pour3to2
  | pour3to4
  | pour4to1
  | pour4to2
  | pour4to3

/-- Applies a single pouring operation to a glass state --/
def applyOperation (state : GlassState) (op : PourOperation) (m n k : ℕ) : GlassState :=
  sorry

/-- Checks if a specific amount can be achieved in any glass --/
def canAchieveAmount (m n k s : ℕ) : Prop :=
  ∃ (operations : List PourOperation),
    let finalState := operations.foldl (λ state op => applyOperation state op m n k)
                        (GlassState.mk 0 0 0 (m + n + k))
    finalState.glass1 = s ∨ finalState.glass2 = s ∨ finalState.glass3 = s ∨ finalState.glass4 = s

/-- The main theorem stating the equivalence --/
theorem wine_pouring_equivalence (m n k : ℕ) :
  (∀ s : ℕ, s < m + n + k → canAchieveAmount m n k s) ↔ Nat.gcd m (Nat.gcd n k) = 1 :=
sorry

end wine_pouring_equivalence_l2921_292101


namespace subset_condition_empty_intersection_condition_l2921_292191

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 2}
def B (a : ℝ) : Set ℝ := {x | 2*a - 1 < x ∧ x < 2*a + 3}

-- Theorem for the subset condition
theorem subset_condition (a : ℝ) : 
  A ⊆ B a ↔ a ∈ Set.Icc (-1/2) 0 :=
sorry

-- Theorem for the empty intersection condition
theorem empty_intersection_condition (a : ℝ) :
  A ∩ B a = ∅ ↔ a ∈ Set.Iic (-2) ∪ Set.Ici (3/2) :=
sorry

end subset_condition_empty_intersection_condition_l2921_292191


namespace train_speed_l2921_292183

/-- The speed of a train passing a bridge -/
theorem train_speed (train_length bridge_length : ℝ) (time : ℝ) (h1 : train_length = 360) 
  (h2 : bridge_length = 140) (h3 : time = 40) : 
  (train_length + bridge_length) / time * 3.6 = 45 := by
  sorry

end train_speed_l2921_292183


namespace max_min_S_l2921_292155

theorem max_min_S (x y z : ℚ) 
  (non_neg_x : x ≥ 0) (non_neg_y : y ≥ 0) (non_neg_z : z ≥ 0)
  (eq1 : 3 * x + 2 * y + z = 5)
  (eq2 : x + y - z = 2)
  (S : ℚ := 2 * x + y - z) :
  (∀ s : ℚ, S ≤ s → s ≤ 3) ∧ (∀ s : ℚ, 2 ≤ s → s ≤ S) :=
by sorry

end max_min_S_l2921_292155


namespace unique_number_with_equal_sums_l2921_292139

theorem unique_number_with_equal_sums : 
  ∃! n : ℕ, 
    (n ≥ 10000) ∧ 
    (n % 10000 = 9876) ∧ 
    (n / 10000 + 9876 = n / 1000 + 876) ∧
    (n = 9999876) := by
  sorry

end unique_number_with_equal_sums_l2921_292139


namespace factory_output_increase_l2921_292158

theorem factory_output_increase (P : ℝ) : 
  (1 + P / 100) * 1.30 * (1 - 30.07 / 100) = 1 → P = 10 := by
sorry

end factory_output_increase_l2921_292158


namespace exam_candidates_girls_l2921_292109

theorem exam_candidates_girls (total : ℕ) (boys_pass_rate girls_pass_rate fail_rate : ℚ) :
  total = 2000 ∧
  boys_pass_rate = 28/100 ∧
  girls_pass_rate = 32/100 ∧
  fail_rate = 702/1000 →
  ∃ (girls : ℕ), 
    girls + (total - girls) = total ∧
    (girls_pass_rate * girls + boys_pass_rate * (total - girls)) / total = 1 - fail_rate ∧
    girls = 900 := by
  sorry

end exam_candidates_girls_l2921_292109


namespace log_10_7_in_terms_of_r_and_s_l2921_292196

theorem log_10_7_in_terms_of_r_and_s (r s : ℝ) 
  (hr : Real.log 2 / Real.log 5 = r) 
  (hs : Real.log 7 / Real.log 2 = s) : 
  Real.log 7 / Real.log 10 = s * r / (r + 1) := by
  sorry

end log_10_7_in_terms_of_r_and_s_l2921_292196


namespace max_non_managers_l2921_292166

theorem max_non_managers (managers : ℕ) (non_managers : ℕ) : 
  managers = 8 → 
  (managers : ℚ) / non_managers > 7 / 32 → 
  non_managers ≤ 36 :=
by
  sorry

end max_non_managers_l2921_292166


namespace first_question_percentage_l2921_292123

theorem first_question_percentage
  (second_correct : Real)
  (neither_correct : Real)
  (both_correct : Real)
  (h1 : second_correct = 0.3)
  (h2 : neither_correct = 0.2)
  (h3 : both_correct = 0.25) :
  ∃ (first_correct : Real),
    first_correct = 0.75 ∧
    first_correct + second_correct - both_correct = 1 - neither_correct :=
by sorry

end first_question_percentage_l2921_292123


namespace equation_solutions_l2921_292120

theorem equation_solutions :
  (∃ y₁ y₂ : ℝ, (2 * y₁^2 + 3 * y₁ - 1 = 0 ∧ 
                 2 * y₂^2 + 3 * y₂ - 1 = 0 ∧
                 y₁ = (-3 + Real.sqrt 17) / 4 ∧
                 y₂ = (-3 - Real.sqrt 17) / 4)) ∧
  (∃ x : ℝ, x * (x - 4) = -4 ∧ x = 2) := by
  sorry

end equation_solutions_l2921_292120


namespace min_value_problem_l2921_292169

theorem min_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 2) :
  (1 / x^2) + (1 / y^2) + (1 / (x * y)) ≥ 3 ∧
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + b = 2 ∧ (1 / a^2) + (1 / b^2) + (1 / (a * b)) = 3 :=
by sorry

end min_value_problem_l2921_292169


namespace netGoalsForTimesMiddleSchool_l2921_292190

/-- Calculates the net goals for a single match -/
def netGoals (goalsFor goalsAgainst : ℤ) : ℤ := goalsFor - goalsAgainst

/-- Represents the scores of three soccer matches -/
structure ThreeMatches where
  match1 : (ℤ × ℤ)
  match2 : (ℤ × ℤ)
  match3 : (ℤ × ℤ)

/-- The specific scores for the Times Middle School soccer team -/
def timesMiddleSchoolScores : ThreeMatches := {
  match1 := (5, 3)
  match2 := (2, 6)
  match3 := (2, 2)
}

/-- Theorem stating that the net number of goals for the given scores is -2 -/
theorem netGoalsForTimesMiddleSchool :
  (netGoals timesMiddleSchoolScores.match1.1 timesMiddleSchoolScores.match1.2) +
  (netGoals timesMiddleSchoolScores.match2.1 timesMiddleSchoolScores.match2.2) +
  (netGoals timesMiddleSchoolScores.match3.1 timesMiddleSchoolScores.match3.2) = -2 := by
  sorry

end netGoalsForTimesMiddleSchool_l2921_292190


namespace a_in_open_interval_l2921_292160

/-- The set A defined as {x | |x-a| ≤ 1} -/
def A (a : ℝ) : Set ℝ := {x | |x - a| ≤ 1}

/-- The set B defined as {x | x^2-5x+4 ≥ 0} -/
def B : Set ℝ := {x | x^2 - 5*x + 4 ≥ 0}

/-- Theorem stating that if A intersect B is empty, then a is in the open interval (2, 3) -/
theorem a_in_open_interval (a : ℝ) (h : A a ∩ B = ∅) : a ∈ Set.Ioo 2 3 := by
  sorry

#check a_in_open_interval

end a_in_open_interval_l2921_292160


namespace sum_of_distinct_roots_is_zero_l2921_292179

/-- Given real numbers a and b, if x^2 + ax + b and x^2 + bx + a each have two distinct real roots,
    and the product of their roots results in exactly three distinct real roots,
    then the sum of these three distinct roots is 0. -/
theorem sum_of_distinct_roots_is_zero (a b : ℝ) 
    (h1 : ∃ r1 r2 : ℝ, r1 ≠ r2 ∧ ∀ x : ℝ, x^2 + a*x + b = 0 ↔ x = r1 ∨ x = r2)
    (h2 : ∃ s1 s2 : ℝ, s1 ≠ s2 ∧ ∀ x : ℝ, x^2 + b*x + a = 0 ↔ x = s1 ∨ x = s2)
    (h3 : ∃ t1 t2 t3 : ℝ, t1 ≠ t2 ∧ t1 ≠ t3 ∧ t2 ≠ t3 ∧ 
          ∀ x : ℝ, (x = t1 ∨ x = t2 ∨ x = t3) ↔ (x^2 + a*x + b = 0 ∨ x^2 + b*x + a = 0)) :
    t1 + t2 + t3 = 0 :=
by sorry

end sum_of_distinct_roots_is_zero_l2921_292179


namespace coffee_mixture_proof_l2921_292141

/-- Given a total mixture and a ratio of coffee to milk, calculate the amount of coffee needed. -/
def coffee_amount (total_mixture : ℕ) (coffee_ratio milk_ratio : ℕ) : ℕ :=
  (total_mixture * coffee_ratio) / (coffee_ratio + milk_ratio)

/-- Theorem stating that for a 4400g mixture with a 2:9 coffee to milk ratio, 800g of coffee is needed. -/
theorem coffee_mixture_proof :
  coffee_amount 4400 2 9 = 800 := by
  sorry

end coffee_mixture_proof_l2921_292141


namespace true_propositions_l2921_292199

-- Define the propositions
def p₁ : Prop := ∀ a b : ℝ, a < b → a^2 < b^2
def p₂ : Prop := ∀ x : ℝ, x > 0 → Real.sin x < x
def p₃ : Prop := ∀ f : ℝ → ℝ, (∀ x : ℝ, f x / f (-x) = -1) ↔ (∀ x : ℝ, f (-x) = -f x)
def p₄ : Prop := ∀ a : ℕ → ℝ, (∀ n : ℕ, a (n+1) = a n * (a 2 / a 1)) →
  ((a 1 > a 2 ∧ a 2 > a 3) ↔ (∀ n : ℕ, a (n+1) < a n))

-- Theorem stating which propositions are true
theorem true_propositions :
  ¬p₁ ∧ p₂ ∧ ¬p₃ ∧ p₄ :=
sorry

end true_propositions_l2921_292199


namespace tan_four_greater_than_tan_three_l2921_292117

theorem tan_four_greater_than_tan_three :
  π / 2 < 3 ∧ 3 < π ∧ π < 4 ∧ 4 < 3 * π / 2 →
  Real.tan 4 > Real.tan 3 := by
  sorry

end tan_four_greater_than_tan_three_l2921_292117


namespace smallest_multiple_eighty_is_solution_eighty_is_smallest_l2921_292161

theorem smallest_multiple (x : ℕ) : x > 0 ∧ 540 * x % 800 = 0 → x ≥ 80 := by
  sorry

theorem eighty_is_solution : 540 * 80 % 800 = 0 := by
  sorry

theorem eighty_is_smallest : ∀ y : ℕ, y > 0 ∧ 540 * y % 800 = 0 → y ≥ 80 := by
  sorry

end smallest_multiple_eighty_is_solution_eighty_is_smallest_l2921_292161


namespace orange_juice_mixture_fraction_l2921_292133

/-- Represents the fraction of orange juice in a mixture -/
def orange_juice_fraction (juice_volume : ℚ) (total_volume : ℚ) : ℚ :=
  juice_volume / total_volume

/-- The volume of the first pitcher in mL -/
def pitcher1_volume : ℚ := 500

/-- The volume of the second pitcher in mL -/
def pitcher2_volume : ℚ := 700

/-- The fraction of orange juice in the first pitcher -/
def pitcher1_juice_fraction : ℚ := 1/2

/-- The fraction of orange juice in the second pitcher -/
def pitcher2_juice_fraction : ℚ := 3/5

/-- Theorem stating that the fraction of orange juice in the final mixture is 67/120 -/
theorem orange_juice_mixture_fraction : 
  orange_juice_fraction 
    (pitcher1_volume * pitcher1_juice_fraction + pitcher2_volume * pitcher2_juice_fraction)
    (pitcher1_volume + pitcher2_volume) = 67/120 := by
  sorry

end orange_juice_mixture_fraction_l2921_292133


namespace firefighter_remaining_money_l2921_292184

-- Define the firefighter's financial parameters
def hourly_rate : ℚ := 30
def weekly_hours : ℚ := 48
def food_expense : ℚ := 500
def tax_expense : ℚ := 1000
def weeks_per_month : ℚ := 4

-- Calculate weekly and monthly earnings
def weekly_earnings : ℚ := hourly_rate * weekly_hours
def monthly_earnings : ℚ := weekly_earnings * weeks_per_month

-- Calculate monthly rent
def monthly_rent : ℚ := monthly_earnings / 3

-- Calculate total monthly expenses
def total_monthly_expenses : ℚ := monthly_rent + food_expense + tax_expense

-- Calculate remaining money after expenses
def remaining_money : ℚ := monthly_earnings - total_monthly_expenses

-- Theorem to prove
theorem firefighter_remaining_money :
  remaining_money = 2340 := by
  sorry

end firefighter_remaining_money_l2921_292184


namespace smallest_possible_value_l2921_292157

theorem smallest_possible_value (x : ℕ+) (a b : ℕ+) : 
  (Nat.gcd a b = x + 2) →
  (Nat.lcm a b = x * (x + 2)) →
  (a = 24) →
  (∀ c : ℕ+, c < b → ¬(Nat.gcd 24 c = x + 2 ∧ Nat.lcm 24 c = x * (x + 2))) →
  b = 6 := by
sorry

end smallest_possible_value_l2921_292157


namespace right_triangle_hypotenuse_l2921_292159

theorem right_triangle_hypotenuse : ∀ (a b c : ℝ),
  -- Right triangle condition
  a^2 + b^2 = c^2 →
  -- Perimeter condition
  a + b + c = 32 →
  -- Area condition
  (1/2) * a * b = 20 →
  -- Conclusion: hypotenuse length
  c = 59/4 := by
  sorry

end right_triangle_hypotenuse_l2921_292159


namespace minimum_value_implies_a_l2921_292124

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a / x

theorem minimum_value_implies_a (a : ℝ) :
  (∀ x ∈ Set.Icc 1 (Real.exp 1), f a x ≥ 3/2) ∧
  (∃ x ∈ Set.Icc 1 (Real.exp 1), f a x = 3/2) →
  a = -Real.sqrt (Real.exp 1) :=
by sorry

end minimum_value_implies_a_l2921_292124


namespace product_of_1101_base2_and_102_base3_l2921_292115

def base2_to_dec (n : List Nat) : Nat :=
  n.enum.foldr (λ (i, b) acc => acc + b * 2^i) 0

def base3_to_dec (n : List Nat) : Nat :=
  n.enum.foldr (λ (i, b) acc => acc + b * 3^i) 0

theorem product_of_1101_base2_and_102_base3 :
  let n1 := base2_to_dec [1, 0, 1, 1]
  let n2 := base3_to_dec [2, 0, 1]
  n1 * n2 = 143 := by
  sorry

end product_of_1101_base2_and_102_base3_l2921_292115


namespace range_of_a_for_R_solution_set_l2921_292103

-- Define the quadratic function
def f (a x : ℝ) : ℝ := (a - 2) * x^2 + 4 * (a - 2) * x - 4

-- Define the property that the solution set is ℝ
def solution_set_is_R (a : ℝ) : Prop := ∀ x, f a x < 0

-- Theorem statement
theorem range_of_a_for_R_solution_set :
  {a : ℝ | solution_set_is_R a} = Set.Ioo 1 2 ∪ {2} :=
sorry

end range_of_a_for_R_solution_set_l2921_292103


namespace part_one_part_two_l2921_292151

-- Define the propositions p and q
def p (a x : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

def q (x : ℝ) : Prop := abs (x - 1) ≤ 2 ∧ (x + 3) / (x - 2) ≥ 0

-- Part 1
theorem part_one : 
  ∀ x : ℝ, p 1 x ∧ q x → 2 < x ∧ x < 3 :=
sorry

-- Part 2
theorem part_two :
  (∀ x : ℝ, q x → (∀ a : ℝ, a > 0 → p a x)) ∧ 
  (∃ x a : ℝ, a > 0 ∧ p a x ∧ ¬q x) → 
  ∀ a : ℝ, 1 < a ∧ a ≤ 2 :=
sorry

end part_one_part_two_l2921_292151


namespace frieda_hop_probability_l2921_292180

/-- Represents a position on the 4x4 grid -/
structure Position :=
  (x : Fin 4)
  (y : Fin 4)

/-- Checks if a position is a corner -/
def is_corner (p : Position) : Bool :=
  (p.x = 0 ∧ p.y = 0) ∨ (p.x = 0 ∧ p.y = 3) ∨
  (p.x = 3 ∧ p.y = 0) ∨ (p.x = 3 ∧ p.y = 3)

/-- Represents a single hop direction -/
inductive Direction
  | Up
  | Down
  | Left
  | Right

/-- Applies a hop in the given direction, considering wrap-around rules -/
def apply_hop (p : Position) (d : Direction) : Position :=
  match d with
  | Direction.Up => ⟨p.x, (p.y + 1) % 4⟩
  | Direction.Down => ⟨p.x, (p.y - 1 + 4) % 4⟩
  | Direction.Left => ⟨(p.x - 1 + 4) % 4, p.y⟩
  | Direction.Right => ⟨(p.x + 1) % 4, p.y⟩

/-- Calculates the probability of reaching a corner in at most n hops -/
def prob_reach_corner (start : Position) (n : Nat) : ℚ :=
  sorry  -- Proof implementation goes here

/-- The main theorem to prove -/
theorem frieda_hop_probability :
  prob_reach_corner ⟨0, 1⟩ 3 = 21 / 32 :=
by sorry  -- Proof goes here

end frieda_hop_probability_l2921_292180


namespace breanna_books_count_l2921_292128

theorem breanna_books_count (tony_total : ℕ) (dean_total : ℕ) (tony_dean_shared : ℕ) (all_shared : ℕ) (total_different : ℕ) :
  tony_total = 23 →
  dean_total = 12 →
  tony_dean_shared = 3 →
  all_shared = 1 →
  total_different = 47 →
  ∃ breanna_total : ℕ,
    tony_total - tony_dean_shared - all_shared +
    dean_total - tony_dean_shared - all_shared +
    breanna_total - all_shared = total_different ∧
    breanna_total = 20 :=
by sorry

end breanna_books_count_l2921_292128


namespace count_pairs_eq_32_l2921_292170

/-- The number of pairs of positive integers (m,n) satisfying m^2 + n^2 < 50 -/
def count_pairs : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => 
    p.1 > 0 ∧ p.2 > 0 ∧ p.1^2 + p.2^2 < 50) (Finset.product (Finset.range 50) (Finset.range 50))).card

theorem count_pairs_eq_32 : count_pairs = 32 := by
  sorry

end count_pairs_eq_32_l2921_292170


namespace inequality_solution_range_l2921_292156

theorem inequality_solution_range (a : ℝ) : 
  (∃ x ∈ Set.Icc 1 4, x^2 + a*x - 2 < 0) ↔ a < 1 := by sorry

end inequality_solution_range_l2921_292156


namespace geometric_sequence_ratio_l2921_292100

/-- A geometric sequence with first term a₁ and common ratio q -/
def geometric_sequence (a₁ q : ℝ) : ℕ → ℝ := fun n ↦ a₁ * q^(n - 1)

/-- The sequence is increasing -/
def is_increasing (a : ℕ → ℝ) : Prop := ∀ n, a n < a (n + 1)

theorem geometric_sequence_ratio 
  (a : ℕ → ℝ) 
  (h_geom : ∃ a₁ q, a = geometric_sequence a₁ q) 
  (h_a₁ : a 1 = -2)
  (h_inc : is_increasing a)
  (h_eq : ∀ n, 3 * (a n + a (n + 2)) = 10 * a (n + 1)) :
  ∃ q, a = geometric_sequence (-2) q ∧ q = 1/3 := by sorry

end geometric_sequence_ratio_l2921_292100


namespace square_sum_formula_l2921_292192

theorem square_sum_formula (x y z a b c d : ℝ) 
  (h1 : x * y = a) 
  (h2 : x * z = b) 
  (h3 : y * z = c) 
  (h4 : (x + y + z)^2 = d) : 
  x^2 + y^2 + z^2 = d - 2*(a + b + c) := by
sorry

end square_sum_formula_l2921_292192


namespace number_equals_two_thirds_a_l2921_292106

/-- Given a and n are real numbers satisfying certain conditions, 
    prove that n equals 2a/3 -/
theorem number_equals_two_thirds_a (a n : ℝ) 
  (h1 : 2 * a = 3 * n) 
  (h2 : a * n ≠ 0) 
  (h3 : (a / 3) / (n / 2) = 1) : 
  n = 2 * a / 3 := by
  sorry

end number_equals_two_thirds_a_l2921_292106


namespace fraction_equality_l2921_292119

theorem fraction_equality : (1^4 + 2009^4 + 2010^4) / (1^2 + 2009^2 + 2010^2) = 4038091 := by
  sorry

end fraction_equality_l2921_292119


namespace rectangular_plot_fence_l2921_292136

theorem rectangular_plot_fence (short_side : ℝ) : 
  short_side > 0 →
  2 * short_side + 2 * (3 * short_side) = 640 →
  short_side = 80 := by
  sorry

end rectangular_plot_fence_l2921_292136


namespace complex_division_equality_l2921_292172

theorem complex_division_equality : Complex.I = (3 + 2 * Complex.I) / (2 - 3 * Complex.I) := by
  sorry

end complex_division_equality_l2921_292172


namespace unit_price_ratio_of_quantity_and_price_difference_l2921_292140

/-- Given two products A and B, where A offers 30% more quantity and costs 15% less than B,
    this theorem proves that the ratio of unit prices (A to B) is 17/26. -/
theorem unit_price_ratio_of_quantity_and_price_difference 
  (quantity_A quantity_B : ℝ) 
  (price_A price_B : ℝ) 
  (h_quantity : quantity_A = 1.3 * quantity_B)
  (h_price : price_A = 0.85 * price_B)
  (h_positive_quantity : quantity_B > 0)
  (h_positive_price : price_B > 0) :
  (price_A / quantity_A) / (price_B / quantity_B) = 17 / 26 := by
  sorry

end unit_price_ratio_of_quantity_and_price_difference_l2921_292140


namespace cylinder_to_cone_base_area_l2921_292127

theorem cylinder_to_cone_base_area (cylinder_radius : Real) (cylinder_height : Real)
  (cone_height : Real) (h1 : cylinder_radius = 1) (h2 : cylinder_height = 1)
  (h3 : cone_height = cylinder_height)
  (h4 : π * cylinder_radius^2 * cylinder_height = (1/3) * π * cone_base_radius^2 * cone_height) :
  π * cone_base_radius^2 = 3 * π :=
by sorry

end cylinder_to_cone_base_area_l2921_292127


namespace optimal_bus_rental_l2921_292135

/-- Represents the rental problem for buses -/
structure BusRental where
  cost_a : ℕ  -- Cost of renting one bus A
  cost_b : ℕ  -- Cost of renting one bus B
  capacity_a : ℕ  -- Capacity of bus A
  capacity_b : ℕ  -- Capacity of bus B
  total_people : ℕ  -- Total number of people to transport
  total_buses : ℕ  -- Total number of buses to rent

/-- Calculates the total cost for a given number of buses A and B -/
def total_cost (br : BusRental) (num_a : ℕ) (num_b : ℕ) : ℕ :=
  num_a * br.cost_a + num_b * br.cost_b

/-- Calculates the total capacity for a given number of buses A and B -/
def total_capacity (br : BusRental) (num_a : ℕ) (num_b : ℕ) : ℕ :=
  num_a * br.capacity_a + num_b * br.capacity_b

/-- Theorem stating that renting 2 buses A and 6 buses B minimizes the cost -/
theorem optimal_bus_rental (br : BusRental) 
  (h1 : br.cost_a + br.cost_b = 500)
  (h2 : 2 * br.cost_a + 3 * br.cost_b = 1300)
  (h3 : br.capacity_a = 15)
  (h4 : br.capacity_b = 25)
  (h5 : br.total_people = 180)
  (h6 : br.total_buses = 8) :
  ∀ (num_a num_b : ℕ), 
    num_a + num_b = br.total_buses →
    total_capacity br num_a num_b ≥ br.total_people →
    total_cost br 2 6 ≤ total_cost br num_a num_b :=
sorry

end optimal_bus_rental_l2921_292135


namespace max_triangle_area_l2921_292164

-- Define the ellipse C₁
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the parabola C₂
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Define a point on the parabola (excluding origin)
def point_on_parabola (P : ℝ × ℝ) : Prop :=
  parabola P.1 P.2 ∧ P ≠ (0, 0)

-- Define the tangent line from a point on the parabola
def tangent_line (P : ℝ × ℝ) (l : ℝ → ℝ) : Prop :=
  point_on_parabola P ∧ ∃ (m b : ℝ), ∀ x, l x = m * x + b

-- Define the intersection points of the tangent with the ellipse
def intersection_points (A B : ℝ × ℝ) (l : ℝ → ℝ) : Prop :=
  ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧
  l A.1 = A.2 ∧ l B.1 = B.2

-- Theorem statement
theorem max_triangle_area 
  (P : ℝ × ℝ) (l : ℝ → ℝ) (A B : ℝ × ℝ) :
  tangent_line P l → intersection_points A B l →
  ∃ (S : ℝ), S ≤ 8 * Real.sqrt 3 ∧ 
  (∃ (P' : ℝ × ℝ) (l' : ℝ → ℝ) (A' B' : ℝ × ℝ),
    tangent_line P' l' ∧ intersection_points A' B' l' ∧
    S = 8 * Real.sqrt 3) :=
sorry

end max_triangle_area_l2921_292164


namespace john_total_calories_l2921_292143

/-- Calculates the total calories consumed by John given the following conditions:
  * John eats 15 potato chips with a total of 90 calories
  * He eats 10 cheezits, each with 2/5 more calories than a chip
  * He eats 8 pretzels, each with 25% fewer calories than a cheezit
-/
theorem john_total_calories : ℝ := by
  -- Define the number of each item eaten
  let num_chips : ℕ := 15
  let num_cheezits : ℕ := 10
  let num_pretzels : ℕ := 8

  -- Define the total calories from chips
  let total_chip_calories : ℝ := 90

  -- Define the calorie increase ratio for cheezits compared to chips
  let cheezit_increase_ratio : ℝ := 2 / 5

  -- Define the calorie decrease ratio for pretzels compared to cheezits
  let pretzel_decrease_ratio : ℝ := 1 / 4

  -- Calculate the total calories
  have h : ∃ (total_calories : ℝ), total_calories = 224.4 := by sorry

  exact h.choose

end john_total_calories_l2921_292143


namespace friend_age_problem_l2921_292116

theorem friend_age_problem (A B C : ℕ) 
  (h1 : A - B = 2)
  (h2 : A - C = 5)
  (h3 : A + B + C = 110) :
  A = 39 := by
sorry

end friend_age_problem_l2921_292116


namespace sarah_score_l2921_292105

theorem sarah_score (s g : ℕ) (h1 : s = g + 30) (h2 : (s + g) / 2 = 108) : s = 123 := by
  sorry

end sarah_score_l2921_292105


namespace sustainable_tree_planting_l2921_292108

theorem sustainable_tree_planting (trees_first_half trees_second_half trees_to_plant : ℕ) :
  trees_first_half = 200 →
  trees_second_half = 300 →
  trees_to_plant = 1500 →
  (trees_to_plant : ℚ) / (trees_first_half + trees_second_half : ℚ) = 3 := by
sorry

end sustainable_tree_planting_l2921_292108


namespace public_transportation_users_l2921_292126

/-- Calculates the number of employees using public transportation -/
theorem public_transportation_users
  (total_employees : ℕ)
  (drive_percentage : ℚ)
  (public_transport_fraction : ℚ)
  (h1 : total_employees = 100)
  (h2 : drive_percentage = 60 / 100)
  (h3 : public_transport_fraction = 1 / 2) :
  ⌊(total_employees : ℚ) * (1 - drive_percentage) * public_transport_fraction⌋ = 20 := by
  sorry

end public_transportation_users_l2921_292126


namespace angle_sum_is_pi_over_two_l2921_292142

theorem angle_sum_is_pi_over_two (α β : Real) (h_acute_α : 0 < α ∧ α < π/2) (h_acute_β : 0 < β ∧ β < π/2) 
  (h_condition : (Real.sin α)^4 / (Real.cos β)^2 + (Real.cos α)^4 / (Real.sin β)^2 = 1) : 
  α + β = π/2 := by
sorry

end angle_sum_is_pi_over_two_l2921_292142


namespace q_subset_p_intersect_q_iff_a_in_range_l2921_292113

-- Define sets P and Q
def P : Set ℝ := {x | 3 < x ∧ x ≤ 22}
def Q (a : ℝ) : Set ℝ := {x | 2 * a + 1 ≤ x ∧ x < 3 * a - 5}

-- State the theorem
theorem q_subset_p_intersect_q_iff_a_in_range :
  ∀ a : ℝ, (Q a).Nonempty → (Q a ⊆ (P ∩ Q a) ↔ 6 < a ∧ a ≤ 9) := by
  sorry

end q_subset_p_intersect_q_iff_a_in_range_l2921_292113


namespace equation_one_solutions_equation_two_solutions_equation_three_solution_l2921_292152

-- Equation 1
theorem equation_one_solutions (x : ℝ) :
  9 * x^2 - (x - 1)^2 = 0 ↔ x = -0.5 ∨ x = 0.25 := by sorry

-- Equation 2
theorem equation_two_solutions (x : ℝ) :
  x * (x - 3) = 10 ↔ x = 5 ∨ x = -2 := by sorry

-- Equation 3
theorem equation_three_solution (x : ℝ) :
  (x + 3)^2 = 2 * x + 5 ↔ x = -2 := by sorry

end equation_one_solutions_equation_two_solutions_equation_three_solution_l2921_292152


namespace unique_numbers_with_lcm_conditions_l2921_292182

theorem unique_numbers_with_lcm_conditions :
  ∃! (x y z : ℕ),
    x > y ∧ x > z ∧
    Nat.lcm x y = 200 ∧
    Nat.lcm y z = 300 ∧
    Nat.lcm x z = 120 ∧
    x = 40 ∧ y = 25 ∧ z = 12 := by
  sorry

end unique_numbers_with_lcm_conditions_l2921_292182


namespace lilias_peaches_l2921_292118

/-- Represents the problem of calculating how many peaches Lilia sold to her friends. -/
theorem lilias_peaches (total_peaches : ℕ) (friends_price : ℚ) (relatives_peaches : ℕ) (relatives_price : ℚ) (kept_peaches : ℕ) (total_earned : ℚ) (total_sold : ℕ) :
  total_peaches = 15 →
  friends_price = 2 →
  relatives_peaches = 4 →
  relatives_price = 5/4 →
  kept_peaches = 1 →
  total_earned = 25 →
  total_sold = 14 →
  ∃ (friends_peaches : ℕ), 
    friends_peaches + relatives_peaches + kept_peaches = total_peaches ∧
    friends_peaches * friends_price + relatives_peaches * relatives_price = total_earned ∧
    friends_peaches = 10 :=
by sorry

end lilias_peaches_l2921_292118


namespace checkerboard_tiling_l2921_292171

/-- The size of the checkerboard -/
def boardSize : Nat := 8

/-- The length of a trimino -/
def triminoLength : Nat := 3

/-- The width of a trimino -/
def triminoWidth : Nat := 1

/-- The area of the checkerboard -/
def boardArea : Nat := boardSize * boardSize

/-- The area of a trimino -/
def triminoArea : Nat := triminoLength * triminoWidth

theorem checkerboard_tiling (boardSize triminoLength triminoWidth : Nat) :
  ¬(boardArea % triminoArea = 0) ∧
  ((boardArea - 1) % triminoArea = 0) := by
  sorry

#check checkerboard_tiling

end checkerboard_tiling_l2921_292171


namespace consecutive_integers_sqrt_3_l2921_292145

theorem consecutive_integers_sqrt_3 (a b : ℤ) : 
  (b = a + 1) → (↑a < Real.sqrt 3 ∧ Real.sqrt 3 < ↑b) → a + b = 3 := by
  sorry

end consecutive_integers_sqrt_3_l2921_292145


namespace odd_prime_square_root_l2921_292154

theorem odd_prime_square_root (p k : ℕ) : 
  Prime p → 
  Odd p → 
  k > 0 → 
  ∃ n : ℕ, n > 0 ∧ n * n = k * k - p * k → 
  k = (p + 1)^2 / 4 := by
sorry

end odd_prime_square_root_l2921_292154


namespace range_of_a_characterize_solution_set_l2921_292189

-- Define the function f
def f (a x : ℝ) : ℝ := a * x^2 + x - a

-- Part 1
theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, f a x > x^2 + a*x - 1 - a) → a ∈ Set.Icc 1 5 := by sorry

-- Part 2
-- We define a function that characterizes the solution set based on 'a'
noncomputable def solution_set (a : ℝ) : Set ℝ :=
  if a > 0 then {x | x < -(a+1)/a ∨ x > 1}
  else if a = 0 then {x | x > 1}
  else if -1/2 < a ∧ a < 0 then {x | 1 < x ∧ x < -(a+1)/a}
  else if a = -1/2 then ∅
  else {x | -(a+1)/a < x ∧ x < 1}

theorem characterize_solution_set (a : ℝ) :
  {x : ℝ | f a x > 1} = solution_set a := by sorry

end range_of_a_characterize_solution_set_l2921_292189


namespace race_people_count_l2921_292148

theorem race_people_count (num_cars : ℕ) (initial_people_per_car : ℕ) (people_gained_halfway : ℕ) :
  num_cars = 20 →
  initial_people_per_car = 3 →
  people_gained_halfway = 1 →
  num_cars * (initial_people_per_car + people_gained_halfway) = 80 := by
sorry

end race_people_count_l2921_292148


namespace fraction_addition_l2921_292121

theorem fraction_addition (a : ℝ) (ha : a ≠ 0) : 3 / a + 2 / a = 5 / a := by
  sorry

end fraction_addition_l2921_292121


namespace wasted_fruits_and_vegetables_is_15_l2921_292181

/-- Calculates the amount of wasted fruits and vegetables in pounds -/
def wasted_fruits_and_vegetables (meat_pounds : ℕ) (meat_price : ℚ) 
  (bread_pounds : ℕ) (bread_price : ℚ) (janitor_hours : ℕ) (janitor_normal_wage : ℚ)
  (minimum_wage : ℚ) (work_hours : ℕ) (fruit_veg_price : ℚ) : ℚ :=
  let meat_cost := meat_pounds * meat_price
  let bread_cost := bread_pounds * bread_price
  let janitor_cost := janitor_hours * (janitor_normal_wage * 1.5)
  let total_earnings := work_hours * minimum_wage
  let remaining_cost := total_earnings - (meat_cost + bread_cost + janitor_cost)
  remaining_cost / fruit_veg_price

theorem wasted_fruits_and_vegetables_is_15 :
  wasted_fruits_and_vegetables 20 5 60 (3/2) 10 10 8 50 4 = 15 := by
  sorry

end wasted_fruits_and_vegetables_is_15_l2921_292181


namespace soda_machine_leak_time_difference_l2921_292194

/-- 
Given a machine that normally fills a barrel of soda in 3 minutes, 
but takes 5 minutes when leaking, prove that it will take 2n minutes 
longer to fill n barrels when leaking, given that it takes 24 minutes 
longer for 12 barrels.
-/
theorem soda_machine_leak_time_difference (n : ℕ) : 
  (3 : ℝ) = normal_fill_time_per_barrel →
  (5 : ℝ) = leaking_fill_time_per_barrel →
  24 = 12 * (leaking_fill_time_per_barrel - normal_fill_time_per_barrel) →
  2 * n = n * (leaking_fill_time_per_barrel - normal_fill_time_per_barrel) :=
by sorry


end soda_machine_leak_time_difference_l2921_292194


namespace column_addition_sum_l2921_292187

theorem column_addition_sum : ∀ (w x y z : ℕ),
  w ≤ 9 ∧ x ≤ 9 ∧ y ≤ 9 ∧ z ≤ 9 →  -- digits are between 0 and 9
  w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z →  -- digits are distinct
  y + w = 10 →  -- rightmost column
  x + y + 1 = 10 →  -- middle column
  w + z + 1 = 11 →  -- leftmost column
  w + x + y + z = 20 :=
by sorry

end column_addition_sum_l2921_292187


namespace complex_multiplication_l2921_292134

theorem complex_multiplication (a : ℝ) (h1 : a > 0) (h2 : Complex.abs (1 + a * Complex.I) = Real.sqrt 5) :
  (1 + a * Complex.I) * (1 + Complex.I) = -1 + 3 * Complex.I := by
  sorry

end complex_multiplication_l2921_292134


namespace complex_magnitude_one_l2921_292107

theorem complex_magnitude_one (z : ℂ) (h : 3 * z^6 + 2 * Complex.I * z^5 - 2 * z - 3 * Complex.I = 0) : 
  Complex.abs z = 1 := by
sorry

end complex_magnitude_one_l2921_292107


namespace trig_fraction_value_l2921_292178

theorem trig_fraction_value (θ : Real) (h : Real.tan θ = -2) :
  (7 * Real.sin θ - 3 * Real.cos θ) / (4 * Real.sin θ + 5 * Real.cos θ) = 17 / 3 := by
  sorry

end trig_fraction_value_l2921_292178


namespace f_negative_a_l2921_292175

theorem f_negative_a (a : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^3 + Real.sin x + 1
  f a = 2 → f (-a) = -2 := by
sorry

end f_negative_a_l2921_292175


namespace circle_chord_tangent_relation_l2921_292195

/-- Given a circle with radius r, a chord FG extending to meet the tangent at F at point H,
    and a point I on FH such that FI = GH, prove that v^2 = u^3 / (r + u),
    where u is the distance of I from the tangent through G
    and v is the distance of I from the line through chord FG. -/
theorem circle_chord_tangent_relation (r : ℝ) (u v : ℝ) 
  (h_positive : r > 0) 
  (h_u_positive : u > 0) 
  (h_v_positive : v > 0) 
  (h_v_eq_r : v = r) : 
  v^2 = u^3 / (r + u) := by
  sorry

end circle_chord_tangent_relation_l2921_292195


namespace max_npm_value_l2921_292168

/-- Represents a two-digit number with equal even digits -/
structure EvenTwoDigit where
  digit : Nat
  h1 : digit % 2 = 0
  h2 : digit < 10

/-- Represents a three-digit number of the form NPM -/
structure ThreeDigitNPM where
  n : Nat
  p : Nat
  m : Nat
  h1 : n > 0
  h2 : n < 10
  h3 : p < 10
  h4 : m < 10

/-- The main theorem stating the maximum value of NPM -/
theorem max_npm_value (mm : EvenTwoDigit) (m : Nat) (npm : ThreeDigitNPM) 
    (h1 : m < 10)
    (h2 : m = mm.digit)
    (h3 : m = npm.m)
    (h4 : (mm.digit * 10 + mm.digit) * m = npm.n * 100 + npm.p * 10 + npm.m) :
  npm.n * 100 + npm.p * 10 + npm.m ≤ 396 := by
  sorry

end max_npm_value_l2921_292168


namespace pizza_distribution_l2921_292131

/-- Given 12 coworkers sharing 3 pizzas equally, where each pizza is cut into 8 slices,
    prove that each coworker will receive 2 slices. -/
theorem pizza_distribution (coworkers : ℕ) (pizzas : ℕ) (slices_per_pizza : ℕ) 
    (h1 : coworkers = 12)
    (h2 : pizzas = 3)
    (h3 : slices_per_pizza = 8) :
    (pizzas * slices_per_pizza) / coworkers = 2 := by
  sorry

end pizza_distribution_l2921_292131


namespace town_neighborhoods_count_l2921_292197

/-- Represents a town with neighborhoods and street lights -/
structure Town where
  total_lights : ℕ
  lights_per_road : ℕ
  roads_per_neighborhood : ℕ

/-- Calculates the number of neighborhoods in the town -/
def number_of_neighborhoods (t : Town) : ℕ :=
  (t.total_lights / t.lights_per_road) / t.roads_per_neighborhood

/-- Theorem: The number of neighborhoods in the given town is 10 -/
theorem town_neighborhoods_count :
  let t : Town := {
    total_lights := 20000,
    lights_per_road := 500,
    roads_per_neighborhood := 4
  }
  number_of_neighborhoods t = 10 := by
  sorry

end town_neighborhoods_count_l2921_292197


namespace container_filling_l2921_292150

theorem container_filling (initial_percentage : Real) (added_amount : Real) (capacity : Real) :
  initial_percentage = 0.4 →
  added_amount = 14 →
  capacity = 40 →
  (initial_percentage * capacity + added_amount) / capacity = 3/4 := by
  sorry

end container_filling_l2921_292150


namespace ellipse_dot_product_constant_l2921_292102

/-- The ellipse with semi-major axis 2 and semi-minor axis √2 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p | (p.1^2 / 4) + (p.2^2 / 2) = 1}

/-- The line x = 2 -/
def Line : Set (ℝ × ℝ) :=
  {p | p.1 = 2}

/-- Left vertex of the ellipse -/
def A₁ : ℝ × ℝ := (-2, 0)

/-- Theorem: For any point C on the ellipse and D on the line x = 2,
    if A₁C = 2CD, then OC · OD = 4 -/
theorem ellipse_dot_product_constant
    (C : ℝ × ℝ) (hC : C ∈ Ellipse)
    (D : ℝ × ℝ) (hD : D ∈ Line)
    (h : dist A₁ C = 2 * dist C D) :
  C.1 * D.1 + C.2 * D.2 = 4 := by
  sorry


end ellipse_dot_product_constant_l2921_292102


namespace original_number_proof_l2921_292112

theorem original_number_proof (numbers : Finset ℕ) (original_sum : ℕ) (changed_sum : ℕ) : 
  numbers.card = 7 →
  original_sum / numbers.card = 7 →
  changed_sum / numbers.card = 8 →
  changed_sum = original_sum - (original_sum / numbers.card) + 9 →
  original_sum / numbers.card = 2 := by
sorry

end original_number_proof_l2921_292112


namespace unique_assignment_exists_l2921_292147

-- Define the types for our images
inductive Image : Type
| cat
| chicken
| crab
| bear
| goat

-- Define a function type that assigns digits to images
def ImageAssignment := Image → Nat

-- Define the conditions for the row and column sums
def satisfiesRowSums (assignment : ImageAssignment) : Prop :=
  assignment Image.cat + assignment Image.chicken + assignment Image.crab + assignment Image.bear + assignment Image.goat = 15 ∧
  assignment Image.goat + assignment Image.goat + assignment Image.crab + assignment Image.bear + assignment Image.bear = 16 ∧
  assignment Image.chicken + assignment Image.chicken + assignment Image.goat + assignment Image.cat + assignment Image.cat = 15 ∧
  assignment Image.crab + assignment Image.crab + assignment Image.crab + assignment Image.crab + assignment Image.crab = 10 ∧
  assignment Image.bear + assignment Image.bear + assignment Image.chicken + assignment Image.chicken + assignment Image.goat = 21

def satisfiesColumnSums (assignment : ImageAssignment) : Prop :=
  assignment Image.cat + assignment Image.goat + assignment Image.chicken + assignment Image.crab + assignment Image.bear = 15 ∧
  assignment Image.chicken + assignment Image.bear + assignment Image.goat + assignment Image.crab + assignment Image.bear = 13 ∧
  assignment Image.crab + assignment Image.crab + assignment Image.chicken + assignment Image.chicken + assignment Image.goat = 17 ∧
  assignment Image.bear + assignment Image.bear + assignment Image.goat + assignment Image.cat + assignment Image.chicken = 20 ∧
  assignment Image.goat + assignment Image.bear + assignment Image.cat + assignment Image.crab + assignment Image.crab = 11

-- Define the condition for different images having different digits
def differentImagesHaveDifferentDigits (assignment : ImageAssignment) : Prop :=
  assignment Image.cat ≠ assignment Image.chicken ∧
  assignment Image.cat ≠ assignment Image.crab ∧
  assignment Image.cat ≠ assignment Image.bear ∧
  assignment Image.cat ≠ assignment Image.goat ∧
  assignment Image.chicken ≠ assignment Image.crab ∧
  assignment Image.chicken ≠ assignment Image.bear ∧
  assignment Image.chicken ≠ assignment Image.goat ∧
  assignment Image.crab ≠ assignment Image.bear ∧
  assignment Image.crab ≠ assignment Image.goat ∧
  assignment Image.bear ≠ assignment Image.goat

-- The main theorem
theorem unique_assignment_exists : 
  ∃! assignment : ImageAssignment, 
    satisfiesRowSums assignment ∧ 
    satisfiesColumnSums assignment ∧ 
    differentImagesHaveDifferentDigits assignment ∧
    assignment Image.cat = 1 ∧
    assignment Image.chicken = 5 ∧
    assignment Image.crab = 2 ∧
    assignment Image.bear = 4 ∧
    assignment Image.goat = 3 :=
  sorry


end unique_assignment_exists_l2921_292147


namespace belle_rawhide_bones_l2921_292193

/-- The number of dog biscuits Belle eats every evening -/
def dog_biscuits : ℕ := 4

/-- The cost of one dog biscuit in dollars -/
def dog_biscuit_cost : ℚ := 1/4

/-- The cost of one rawhide bone in dollars -/
def rawhide_bone_cost : ℚ := 1

/-- The total cost of Belle's treats for a week in dollars -/
def weekly_treat_cost : ℚ := 21

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of rawhide bones Belle eats every evening -/
def rawhide_bones : ℕ := 2

theorem belle_rawhide_bones :
  (dog_biscuits : ℚ) * dog_biscuit_cost * (days_in_week : ℚ) +
  (rawhide_bones : ℚ) * rawhide_bone_cost * (days_in_week : ℚ) =
  weekly_treat_cost :=
sorry

end belle_rawhide_bones_l2921_292193


namespace intersection_A_B_l2921_292110

def A : Set ℝ := {x : ℝ | |x - 2| < 1}
def B : Set ℝ := Set.range (Int.cast : ℤ → ℝ)

theorem intersection_A_B : A ∩ B = {2} := by sorry

end intersection_A_B_l2921_292110


namespace not_divisible_by_81_l2921_292176

theorem not_divisible_by_81 (n : ℤ) : ¬ (81 ∣ (n^3 - 9*n + 27)) := by
  sorry

end not_divisible_by_81_l2921_292176


namespace balloon_multiple_l2921_292153

def nancy_balloons : ℝ := 7.0
def mary_balloons : ℝ := 1.75

theorem balloon_multiple : nancy_balloons / mary_balloons = 4 := by
  sorry

end balloon_multiple_l2921_292153


namespace quadratic_properties_l2921_292188

def f (x : ℝ) := x^2 + 6*x + 5

theorem quadratic_properties :
  (f 0 = 5) ∧
  (∃ v : ℝ × ℝ, v = (-3, -4) ∧ ∀ x : ℝ, f x ≥ f v.1) ∧
  (∀ x : ℝ, f (x + (-3)) = f ((-3) - x)) ∧
  (∀ p : ℝ, f p ≠ -p^2) := by
  sorry

end quadratic_properties_l2921_292188


namespace prime_sum_problem_l2921_292185

theorem prime_sum_problem (p q r s : ℕ) : 
  Prime p → Prime q → Prime r → Prime s →
  p < q → q < r → r < s →
  p * q * r * s + 1 = 4^(p + q) →
  r + s = 274 := by
sorry

end prime_sum_problem_l2921_292185


namespace ratio_proof_l2921_292177

theorem ratio_proof (a b c d : ℝ) 
  (h1 : b / c = 13 / 9)
  (h2 : c / d = 5 / 13)
  (h3 : a / d = 1 / 7.2) :
  a / b = 1 / 4 := by
sorry

end ratio_proof_l2921_292177


namespace eight_to_one_l2921_292122

theorem eight_to_one : (8/8)^(8/8) * 8/8 = 1 := by
  sorry

end eight_to_one_l2921_292122


namespace base9_3671_equals_base10_2737_l2921_292125

def base9_to_base10 (n : Nat) : Nat :=
  (n / 1000) * (9^3) + ((n / 100) % 10) * (9^2) + ((n / 10) % 10) * 9 + (n % 10)

theorem base9_3671_equals_base10_2737 :
  base9_to_base10 3671 = 2737 := by
  sorry

end base9_3671_equals_base10_2737_l2921_292125


namespace sphere_volume_l2921_292173

theorem sphere_volume (r : ℝ) (h : 4 * π * r^2 = 2 * Real.sqrt 3 * π * (2 * r)) :
  (4 / 3) * π * r^3 = 4 * Real.sqrt 3 * π := by
  sorry

end sphere_volume_l2921_292173


namespace factor_x12_minus_729_l2921_292162

theorem factor_x12_minus_729 (x : ℝ) :
  x^12 - 729 = (x^6 + 27) * (x^3 + 3 * Real.sqrt 3) * (x^3 - 3 * Real.sqrt 3) :=
by
  have h : 729 = 3^6 := by norm_num
  sorry

end factor_x12_minus_729_l2921_292162


namespace euclidean_division_remainder_l2921_292149

theorem euclidean_division_remainder 
  (P : Polynomial ℝ) 
  (D : Polynomial ℝ) 
  (h1 : P = X^100 - 2*X^51 + 1)
  (h2 : D = X^2 - 1) :
  ∃ (Q R : Polynomial ℝ), 
    P = D * Q + R ∧ 
    R.degree < D.degree ∧ 
    R = -2*X + 2 := by
sorry

end euclidean_division_remainder_l2921_292149


namespace average_of_numbers_l2921_292132

def numbers : List ℤ := [54, 55, 57, 58, 59, 62, 62, 63, 65, 65]

theorem average_of_numbers : (numbers.sum : ℚ) / numbers.length = 60 := by
  sorry

end average_of_numbers_l2921_292132


namespace fifth_term_of_geometric_sequence_l2921_292137

-- Define a positive geometric sequence
def is_positive_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n > 0 ∧ ∃ r : ℝ, r > 0 ∧ ∀ k : ℕ, a (k + 1) = r * a k

-- State the theorem
theorem fifth_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geom : is_positive_geometric_sequence a)
  (h_roots : a 4 * a 6 = 6 ∧ a 4 + a 6 = 5) :
  a 5 = Real.sqrt 6 := by
  sorry

end fifth_term_of_geometric_sequence_l2921_292137


namespace garden_ratio_l2921_292114

/-- A rectangular garden with given perimeter and length has a specific length-to-width ratio -/
theorem garden_ratio (perimeter : ℝ) (length : ℝ) (width : ℝ) 
  (h_perimeter : perimeter = 900)
  (h_length : length = 300)
  (h_rectangle : perimeter = 2 * (length + width)) : 
  length / width = 2 := by
  sorry

end garden_ratio_l2921_292114


namespace max_value_fraction_l2921_292167

theorem max_value_fraction (x y : ℝ) (hx : 10 ≤ x ∧ x ≤ 20) (hy : 40 ≤ y ∧ y ≤ 60) :
  (x^2 / (2 * y)) ≤ 5 :=
by sorry

end max_value_fraction_l2921_292167
