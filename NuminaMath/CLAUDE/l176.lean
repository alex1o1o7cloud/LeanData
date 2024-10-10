import Mathlib

namespace complex_sum_equals_minus_ten_i_l176_17652

theorem complex_sum_equals_minus_ten_i :
  (5 - 5 * Complex.I) + (-2 - Complex.I) - (3 + 4 * Complex.I) = -10 * Complex.I :=
by sorry

end complex_sum_equals_minus_ten_i_l176_17652


namespace geometric_increasing_condition_l176_17692

/-- A geometric sequence with common ratio q -/
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

/-- An increasing sequence -/
def increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

/-- The condition "q > 1" is neither sufficient nor necessary for a geometric sequence to be increasing -/
theorem geometric_increasing_condition (a : ℕ → ℝ) (q : ℝ) :
  ¬(((q > 1) → increasing_sequence a) ∧ (increasing_sequence a → (q > 1))) :=
sorry

end geometric_increasing_condition_l176_17692


namespace max_visibility_score_l176_17681

/-- Represents a configuration of towers --/
structure TowerConfig where
  height1 : ℕ  -- Number of towers with height 1
  height2 : ℕ  -- Number of towers with height 2

/-- The total height of all towers is 30 --/
def validConfig (config : TowerConfig) : Prop :=
  config.height1 + 2 * config.height2 = 30

/-- Calculate the visibility score for a given configuration --/
def visibilityScore (config : TowerConfig) : ℕ :=
  config.height1 * config.height2

/-- Theorem: The maximum visibility score is 112 and is achieved
    when all towers are either height 1 or 2 --/
theorem max_visibility_score :
  ∃ (config : TowerConfig), validConfig config ∧
    visibilityScore config = 112 ∧
    (∀ (other : TowerConfig), validConfig other →
      visibilityScore other ≤ visibilityScore config) :=
by sorry

end max_visibility_score_l176_17681


namespace fifth_term_is_five_l176_17612

/-- An arithmetic sequence with specific conditions -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence
  S : ℕ → ℚ  -- Sum function
  h1 : S 6 = 3
  h2 : a 4 = 2

/-- The fifth term of the arithmetic sequence is 5 -/
theorem fifth_term_is_five (seq : ArithmeticSequence) : seq.a 5 = 5 := by
  sorry

end fifth_term_is_five_l176_17612


namespace total_legs_of_bokyungs_animals_l176_17668

-- Define the number of legs for puppies and chicks
def puppy_legs : ℕ := 4
def chick_legs : ℕ := 2

-- Define the number of puppies and chicks Bokyung has
def num_puppies : ℕ := 3
def num_chicks : ℕ := 7

-- Theorem to prove
theorem total_legs_of_bokyungs_animals : 
  num_puppies * puppy_legs + num_chicks * chick_legs = 26 := by
  sorry

end total_legs_of_bokyungs_animals_l176_17668


namespace inequality_range_l176_17602

theorem inequality_range (a : ℝ) : 
  (∀ x > 0, 2 * x + 1 / x - a > 0) → a < 2 * Real.sqrt 2 := by
sorry

end inequality_range_l176_17602


namespace phi_value_for_symmetric_sine_l176_17694

theorem phi_value_for_symmetric_sine (f : ℝ → ℝ) (φ : ℝ) :
  (∀ x, f x = Real.sin (2 * x + φ)) →
  f (π / 3) = 1 / 2 →
  f (-π / 3) = 1 / 2 →
  ∃ k : ℤ, φ = 2 * k * π - π / 2 := by
  sorry

end phi_value_for_symmetric_sine_l176_17694


namespace smallest_n_congruence_l176_17689

theorem smallest_n_congruence (n : ℕ) : 
  (n > 0 ∧ 23 * n ≡ 789 [ZMOD 11] ∧ ∀ m : ℕ, m > 0 ∧ m < n → ¬(23 * m ≡ 789 [ZMOD 11])) → n = 9 := by
  sorry

end smallest_n_congruence_l176_17689


namespace existence_of_multiple_1984_l176_17698

theorem existence_of_multiple_1984 (a : Fin 97 → ℕ+) (h_distinct : ∀ i j, i ≠ j → a i ≠ a j) :
  ∃ i j k l : Fin 97, i ≠ j ∧ k ≠ l ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧
    1984 ∣ ((a i).val - (a j).val) * ((a k).val - (a l).val) :=
by sorry

end existence_of_multiple_1984_l176_17698


namespace unique_triplet_l176_17616

theorem unique_triplet :
  ∃! (a b c : ℕ), 1 < a ∧ a < b ∧ b < c ∧
  (c ∣ a * b + 1) ∧
  (b ∣ a * c + 1) ∧
  (a ∣ b * c + 1) ∧
  a = 2 ∧ b = 3 ∧ c = 7 :=
by sorry

end unique_triplet_l176_17616


namespace tv_cost_is_250_l176_17618

/-- The cost of the TV given Linda's savings and furniture expenditure -/
def tv_cost (savings : ℚ) (furniture_fraction : ℚ) : ℚ :=
  savings * (1 - furniture_fraction)

/-- Theorem stating that the TV cost is $250 given the problem conditions -/
theorem tv_cost_is_250 :
  tv_cost 1000 (3/4) = 250 := by
  sorry

end tv_cost_is_250_l176_17618


namespace line_vector_proof_l176_17607

def line_vector (t : ℝ) : ℝ × ℝ × ℝ := sorry

theorem line_vector_proof :
  (line_vector 0 = (2, 5, 9)) →
  (line_vector 1 = (3, 3, 5)) →
  (line_vector (-1) = (1, 7, 13)) := by sorry

end line_vector_proof_l176_17607


namespace inequality_range_l176_17636

theorem inequality_range (a : ℝ) : 
  (∀ x ∈ Set.Icc (-5 : ℝ) 0, x^2 + 2*x - 3 + a ≤ 0) ↔ a ∈ Set.Iic (-12 : ℝ) :=
by sorry

end inequality_range_l176_17636


namespace new_shoes_lifespan_l176_17604

/-- Proves that the lifespan of new shoes is 2 years given the costs and conditions -/
theorem new_shoes_lifespan (repair_cost : ℝ) (repair_lifespan : ℝ) (new_cost : ℝ) (cost_increase_percentage : ℝ) :
  repair_cost = 14.50 →
  repair_lifespan = 1 →
  new_cost = 32.00 →
  cost_increase_percentage = 10.344827586206897 →
  let new_lifespan := new_cost / (repair_cost * (1 + cost_increase_percentage / 100))
  new_lifespan = 2 := by
  sorry

end new_shoes_lifespan_l176_17604


namespace count_perfect_square_factors_l176_17638

def prime_factors : List (Nat × Nat) := [(2, 12), (3, 16), (7, 18), (11, 7)]

def count_square_factors (p : Nat) (e : Nat) : Nat :=
  (e / 2) + 1

theorem count_perfect_square_factors :
  (prime_factors.map (fun (p, e) => count_square_factors p e)).prod = 2520 := by
  sorry

end count_perfect_square_factors_l176_17638


namespace solve_beef_problem_l176_17631

def beef_problem (pounds_per_pack : ℝ) (price_per_pound : ℝ) (total_paid : ℝ) : Prop :=
  let price_per_pack := pounds_per_pack * price_per_pound
  let num_packs := total_paid / price_per_pack
  num_packs = 5

theorem solve_beef_problem :
  beef_problem 4 5.50 110 := by
  sorry

end solve_beef_problem_l176_17631


namespace christmas_to_january_10_l176_17623

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

-- Define a function to get the next day
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

-- Define a function to advance a day by n days
def advanceDay (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => advanceDay (nextDay d) n

-- Theorem statement
theorem christmas_to_january_10 :
  advanceDay DayOfWeek.Wednesday 16 = DayOfWeek.Friday := by
  sorry

end christmas_to_january_10_l176_17623


namespace twenty_one_numbers_inequality_l176_17606

theorem twenty_one_numbers_inequality (S : Finset ℕ) : 
  S ⊆ Finset.range 2047 →
  S.card = 21 →
  ∃ a b c : ℕ, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ 
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (b * c : ℝ) < 2 * (a^2 : ℝ) ∧ 2 * (a^2 : ℝ) < 4 * (b * c : ℝ) := by
  sorry

end twenty_one_numbers_inequality_l176_17606


namespace smallest_n_not_prime_l176_17696

theorem smallest_n_not_prime (n : ℕ) : 
  (∀ k < n, Nat.Prime (2^k + 1)) ∧ ¬(Nat.Prime (2^n + 1)) ↔ n = 3 := by
  sorry

end smallest_n_not_prime_l176_17696


namespace square_of_binomial_exclusion_l176_17643

theorem square_of_binomial_exclusion (a b x m : ℝ) : 
  (∃ p q : ℝ, (x + a) * (x - a) = p^2 - q^2) ∧ 
  (∃ p q : ℝ, (-x - b) * (x - b) = -(p^2 - q^2)) ∧ 
  (∃ p q : ℝ, (b + m) * (m - b) = p^2 - q^2) ∧ 
  ¬(∃ p : ℝ, (a + b) * (-a - b) = p^2) :=
by sorry

end square_of_binomial_exclusion_l176_17643


namespace f_increasing_iff_l176_17633

/-- Definition of the piecewise function f(x) --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < -1 then (-a + 4) * x - 3 * a
  else x^2 + a * x - 8

/-- Theorem stating the condition for f to be increasing --/
theorem f_increasing_iff (a : ℝ) :
  (∀ x y, x < y → f a x < f a y) ↔ 3 ≤ a ∧ a < 4 := by
  sorry

end f_increasing_iff_l176_17633


namespace unique_consecutive_sum_18_l176_17685

/-- A function that returns the sum of n consecutive integers starting from a -/
def consecutive_sum (a n : ℕ) : ℕ := n * (2 * a + n - 1) / 2

/-- A predicate that checks if a set of consecutive integers sums to 18 -/
def is_valid_set (a n : ℕ) : Prop :=
  n ≥ 2 ∧ consecutive_sum a n = 18

theorem unique_consecutive_sum_18 :
  ∃! p : ℕ × ℕ, is_valid_set p.1 p.2 :=
sorry

end unique_consecutive_sum_18_l176_17685


namespace negation_equivalence_l176_17620

theorem negation_equivalence :
  (¬ ∀ n : ℕ, 3^n > 500^n) ↔ (∃ n₀ : ℕ, 3^n₀ ≤ 500) :=
sorry

end negation_equivalence_l176_17620


namespace selection_methods_count_l176_17682

/-- The number of teachers in each department -/
def teachers_per_dept : ℕ := 4

/-- The total number of departments -/
def total_depts : ℕ := 4

/-- The number of leaders to be selected -/
def leaders_to_select : ℕ := 3

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of ways to select leaders satisfying the given conditions -/
def selection_methods : ℕ :=
  -- One from admin, two from same other dept
  choose teachers_per_dept 1 * choose (total_depts - 1) 1 * choose teachers_per_dept 2 +
  -- One from admin, two from different other depts
  choose teachers_per_dept 1 * choose (total_depts - 1) 2 * choose teachers_per_dept 1 * choose teachers_per_dept 1 +
  -- Two from admin, one from any other dept
  choose teachers_per_dept 2 * choose (total_depts - 1) 1 * choose teachers_per_dept 1

theorem selection_methods_count :
  selection_methods = 336 :=
by sorry

end selection_methods_count_l176_17682


namespace keychain_arrangements_l176_17621

def number_of_keychains : ℕ := 5

def total_permutations (n : ℕ) : ℕ := n.factorial

def adjacent_permutations (n : ℕ) : ℕ := 2 * (n - 1).factorial

theorem keychain_arrangements :
  total_permutations number_of_keychains - adjacent_permutations number_of_keychains = 72 :=
sorry

end keychain_arrangements_l176_17621


namespace roses_planted_over_three_days_l176_17611

/-- Represents the number of roses planted by each person on a given day -/
structure PlantingData where
  susan : ℕ
  maria : ℕ
  john : ℕ

/-- Calculates the total roses planted on a given day -/
def total_roses (data : PlantingData) : ℕ :=
  data.susan + data.maria + data.john

/-- Represents the planting data for all three days -/
structure ThreeDayPlanting where
  day1 : PlantingData
  day2 : PlantingData
  day3 : PlantingData

theorem roses_planted_over_three_days :
  ∀ (planting : ThreeDayPlanting),
    (planting.day1.susan + planting.day1.maria + planting.day1.john = 50) →
    (planting.day1.maria = 2 * planting.day1.susan) →
    (planting.day1.john = planting.day1.susan + 10) →
    (total_roses planting.day2 = total_roses planting.day1 + 20) →
    (planting.day2.susan * 5 = planting.day1.susan * 7) →
    (planting.day2.maria * 5 = planting.day1.maria * 7) →
    (planting.day2.john * 5 = planting.day1.john * 7) →
    (total_roses planting.day3 = 2 * total_roses planting.day1) →
    (planting.day3.susan = planting.day1.susan) →
    (planting.day3.maria = planting.day1.maria + (planting.day1.maria / 4)) →
    (planting.day3.john = planting.day1.john - (planting.day1.john / 10)) →
    (total_roses planting.day1 + total_roses planting.day2 + total_roses planting.day3 = 173) :=
by sorry

end roses_planted_over_three_days_l176_17611


namespace reciprocal_equals_self_l176_17669

theorem reciprocal_equals_self (q : ℚ) : q⁻¹ = q → q = 1 ∨ q = -1 := by
  sorry

end reciprocal_equals_self_l176_17669


namespace square_root_squared_specific_square_root_squared_l176_17608

theorem square_root_squared (n : ℝ) (h : 0 ≤ n) : (Real.sqrt n) ^ 2 = n := by sorry

theorem specific_square_root_squared : (Real.sqrt 529441) ^ 2 = 529441 := by
  apply square_root_squared
  norm_num

end square_root_squared_specific_square_root_squared_l176_17608


namespace geometric_sequence_sum_l176_17690

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_geometric_sequence a →
  a 1 > 0 →
  a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 36 →
  a 3 + a 5 = 6 :=
by
  sorry

end geometric_sequence_sum_l176_17690


namespace certain_number_proof_l176_17644

theorem certain_number_proof (x : ℤ) (N : ℝ) 
  (h1 : N * (10 : ℝ)^(x : ℝ) < 220000)
  (h2 : ∀ y : ℤ, y > 5 → N * (10 : ℝ)^(y : ℝ) ≥ 220000) :
  N = 2 := by
  sorry

end certain_number_proof_l176_17644


namespace root_two_implies_a_and_other_root_always_real_roots_l176_17605

-- Define the equation
def equation (x a : ℝ) : Prop := x^2 + a*x + a - 1 = 0

-- Theorem 1: If 2 is a root, then a = -1 and the other root is -1
theorem root_two_implies_a_and_other_root (a : ℝ) :
  equation 2 a → a = -1 ∧ equation (-1) a := by sorry

-- Theorem 2: The equation always has real roots
theorem always_real_roots (a : ℝ) :
  ∃ x : ℝ, equation x a := by sorry

end root_two_implies_a_and_other_root_always_real_roots_l176_17605


namespace monotone_decreasing_implies_a_geq_3_l176_17661

/-- A cubic function with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x - 1

/-- The derivative of f with respect to x -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - a

/-- f is monotonically decreasing on (-1, 1) -/
def is_monotone_decreasing (a : ℝ) : Prop :=
  ∀ x ∈ Set.Ioo (-1) 1, f_deriv a x ≤ 0

theorem monotone_decreasing_implies_a_geq_3 :
  ∀ a : ℝ, is_monotone_decreasing a → a ≥ 3 :=
sorry

end monotone_decreasing_implies_a_geq_3_l176_17661


namespace cups_per_serving_l176_17659

/-- Given a recipe that requires 18.0 servings of cereal and 36 cups in total,
    prove that each serving consists of 2 cups. -/
theorem cups_per_serving (servings : Real) (total_cups : Nat) 
    (h1 : servings = 18.0) (h2 : total_cups = 36) : 
    (total_cups : Real) / servings = 2 := by
  sorry

end cups_per_serving_l176_17659


namespace power_of_three_plus_five_mod_ten_l176_17667

theorem power_of_three_plus_five_mod_ten : (3^108 + 5) % 10 = 6 := by
  sorry

end power_of_three_plus_five_mod_ten_l176_17667


namespace two_times_zero_times_one_plus_one_l176_17651

theorem two_times_zero_times_one_plus_one : 2 * 0 * 1 + 1 = 1 := by
  sorry

end two_times_zero_times_one_plus_one_l176_17651


namespace equation_solution_l176_17695

theorem equation_solution : ∃ m : ℝ, (243 : ℝ) ^ (1/5) = 3^m ∧ m = 1 := by
  sorry

end equation_solution_l176_17695


namespace simplify_expression_l176_17654

theorem simplify_expression (s : ℝ) : 180 * s - 88 * s = 92 * s := by
  sorry

end simplify_expression_l176_17654


namespace systematic_sampling_validity_l176_17641

/-- Represents a set of student numbers -/
def StudentSet : Type := List Nat

/-- Checks if a list of natural numbers is arithmetic progression with common difference d -/
def isArithmeticProgression (l : List Nat) (d : Nat) : Prop :=
  l.zipWith (· - ·) (l.tail) = List.replicate (l.length - 1) d

/-- Checks if a set of student numbers is a valid systematic sample -/
def isValidSystematicSample (s : StudentSet) (totalStudents numSelected : Nat) : Prop :=
  s.length = numSelected ∧
  s.all (· ≤ totalStudents) ∧
  isArithmeticProgression s (totalStudents / numSelected)

theorem systematic_sampling_validity :
  let totalStudents : Nat := 50
  let numSelected : Nat := 5
  let sampleSet : StudentSet := [6, 16, 26, 36, 46]
  isValidSystematicSample sampleSet totalStudents numSelected := by
  sorry

end systematic_sampling_validity_l176_17641


namespace max_sum_is_38_l176_17679

/-- Represents the setup of numbers in the grid -/
structure Grid :=
  (a b c d e : ℕ)

/-- The set of available numbers -/
def availableNumbers : Finset ℕ := {2, 3, 8, 9, 14, 15}

/-- Checks if the grid satisfies the equality condition -/
def isValidGrid (g : Grid) : Prop :=
  g.a + g.b + g.e = g.a + g.c + g.e ∧
  g.a + g.c + g.e = g.b + g.d + g.e

/-- Checks if the grid uses numbers from the available set -/
def usesAvailableNumbers (g : Grid) : Prop :=
  g.a ∈ availableNumbers ∧
  g.b ∈ availableNumbers ∧
  g.c ∈ availableNumbers ∧
  g.d ∈ availableNumbers ∧
  g.e ∈ availableNumbers

/-- Calculates the sum of the grid -/
def gridSum (g : Grid) : ℕ := g.a + g.b + g.e

/-- Theorem: The maximum sum of a valid grid using the available numbers is 38 -/
theorem max_sum_is_38 :
  ∃ (g : Grid), isValidGrid g ∧ usesAvailableNumbers g ∧
  (∀ (h : Grid), isValidGrid h ∧ usesAvailableNumbers h → gridSum h ≤ gridSum g) ∧
  gridSum g = 38 := by
  sorry

end max_sum_is_38_l176_17679


namespace profit_percent_when_cost_is_40_percent_of_selling_price_l176_17674

theorem profit_percent_when_cost_is_40_percent_of_selling_price :
  ∀ (selling_price : ℝ), selling_price > 0 →
  let cost_price := 0.4 * selling_price
  let profit := selling_price - cost_price
  let profit_percent := (profit / cost_price) * 100
  profit_percent = 150 := by
sorry

end profit_percent_when_cost_is_40_percent_of_selling_price_l176_17674


namespace complex_real_condition_l176_17615

theorem complex_real_condition (m : ℝ) : 
  (Complex.I : ℂ) * (1 + m * Complex.I) + (m^2 : ℂ) * (1 + m * Complex.I) ∈ Set.range (Complex.ofReal) → 
  m = -1 := by
sorry

end complex_real_condition_l176_17615


namespace gcd_204_85_f_at_2_l176_17632

-- Part 1: GCD of 204 and 85
theorem gcd_204_85 : Nat.gcd 204 85 = 17 := by sorry

-- Part 2: Value of polynomial at x = 2
def f (x : ℝ) : ℝ := 2*x^4 + 3*x^3 + 5*x - 4

theorem f_at_2 : f 2 = 62 := by sorry

end gcd_204_85_f_at_2_l176_17632


namespace number_difference_l176_17640

theorem number_difference (a b : ℕ) (h1 : a + b = 56) (h2 : a < b) (h3 : a = 22) (h4 : b = 34) :
  b - a = 12 := by
  sorry

end number_difference_l176_17640


namespace paige_folders_proof_l176_17680

def number_of_folders (initial_files deleted_files files_per_folder : ℕ) : ℕ :=
  (initial_files - deleted_files) / files_per_folder

theorem paige_folders_proof (initial_files deleted_files files_per_folder : ℕ) 
  (h1 : initial_files = 27)
  (h2 : deleted_files = 9)
  (h3 : files_per_folder = 6)
  : number_of_folders initial_files deleted_files files_per_folder = 3 := by
  sorry

#eval number_of_folders 27 9 6

end paige_folders_proof_l176_17680


namespace symmetric_points_m_value_l176_17676

/-- Two points are symmetric about the origin if their coordinates are negatives of each other -/
def symmetric_about_origin (A B : ℝ × ℝ) : Prop :=
  A.1 = -B.1 ∧ A.2 = -B.2

/-- Given that point A(2, -1) is symmetric with point B(-2, m) about the origin, prove that m = 1 -/
theorem symmetric_points_m_value :
  let A : ℝ × ℝ := (2, -1)
  let B : ℝ × ℝ := (-2, m)
  symmetric_about_origin A B → m = 1 := by
sorry

end symmetric_points_m_value_l176_17676


namespace minimum_point_of_translated_parabola_l176_17648

def f (x : ℝ) : ℝ := (x - 3)^2 - 8

theorem minimum_point_of_translated_parabola :
  ∃ (x₀ : ℝ), ∀ (x : ℝ), f x₀ ≤ f x ∧ x₀ = 3 ∧ f x₀ = -8 :=
sorry

end minimum_point_of_translated_parabola_l176_17648


namespace convex_pentagon_inner_lattice_point_l176_17649

/-- A point in the 2D Cartesian plane -/
structure Point where
  x : ℤ
  y : ℤ

/-- A pentagon defined by five points -/
structure Pentagon where
  A : Point
  B : Point
  C : Point
  D : Point
  E : Point

/-- Checks if a pentagon is convex -/
def isConvex (p : Pentagon) : Prop :=
  sorry

/-- Checks if a point is a lattice point -/
def isLatticePoint (p : Point) : Prop :=
  sorry

/-- Constructs the inner pentagon formed by the intersection of diagonals -/
def innerPentagon (p : Pentagon) : Pentagon :=
  sorry

/-- Checks if a point is inside or on the boundary of a pentagon -/
def isInOrOnPentagon (point : Point) (p : Pentagon) : Prop :=
  sorry

/-- The main theorem -/
theorem convex_pentagon_inner_lattice_point (p : Pentagon) :
  isConvex p →
  isLatticePoint p.A ∧ isLatticePoint p.B ∧ isLatticePoint p.C ∧ isLatticePoint p.D ∧ isLatticePoint p.E →
  ∃ (point : Point), isLatticePoint point ∧ isInOrOnPentagon point (innerPentagon p) :=
sorry

end convex_pentagon_inner_lattice_point_l176_17649


namespace unique_lcm_triple_l176_17673

theorem unique_lcm_triple : ∃! (x y z : ℕ+), 
  (Nat.lcm x.val y.val = 108) ∧ 
  (Nat.lcm x.val z.val = 400) ∧ 
  (Nat.lcm y.val z.val = 450) := by
  sorry

end unique_lcm_triple_l176_17673


namespace simplify_expression_l176_17687

theorem simplify_expression (y : ℝ) : 3*y + 4*y^2 + 2 - (5 - (3*y + 4*y^2) - 8) = 8*y^2 + 6*y + 5 := by
  sorry

end simplify_expression_l176_17687


namespace spurs_rockets_basketballs_l176_17614

/-- The number of basketballs for two teams given their player counts and basketballs per player -/
def combined_basketballs (x y z : ℕ) : ℕ := x * z + y * z

/-- Theorem: The combined number of basketballs for the Spurs and Rockets is 440 -/
theorem spurs_rockets_basketballs :
  let x : ℕ := 22  -- number of Spurs players
  let y : ℕ := 18  -- number of Rockets players
  let z : ℕ := 11  -- number of basketballs per player
  combined_basketballs x y z = 440 := by
  sorry

end spurs_rockets_basketballs_l176_17614


namespace leopards_count_l176_17626

def zoo_problem (leopards : ℕ) : Prop :=
  let snakes := 100
  let arctic_foxes := 80
  let bee_eaters := 10 * leopards
  let cheetahs := snakes / 2
  let alligators := 2 * (arctic_foxes + leopards)
  let total_animals := 670
  snakes + arctic_foxes + leopards + bee_eaters + cheetahs + alligators = total_animals

theorem leopards_count : ∃ (l : ℕ), zoo_problem l ∧ l = 20 := by
  sorry

end leopards_count_l176_17626


namespace reciprocal_of_sum_divided_l176_17671

theorem reciprocal_of_sum_divided : 
  (((1 : ℚ) / 4 + (1 : ℚ) / 5) / ((1 : ℚ) / 3))⁻¹ = 20 / 27 := by
  sorry

end reciprocal_of_sum_divided_l176_17671


namespace union_of_sets_l176_17645

theorem union_of_sets (p q : ℝ) :
  let A := {x : ℝ | x^2 + p*x + q = 0}
  let B := {x : ℝ | x^2 - p*x - 2*q = 0}
  (A ∩ B = {-1}) →
  (A ∪ B = {-1, -2, 4}) := by
sorry

end union_of_sets_l176_17645


namespace double_value_points_range_l176_17642

/-- A point (k, 2k) is a double value point -/
def DoubleValuePoint (k : ℝ) : ℝ × ℝ := (k, 2 * k)

/-- The quadratic function -/
def QuadraticFunction (t s : ℝ) (x : ℝ) : ℝ := (t + 1) * x^2 + (t + 2) * x + s

theorem double_value_points_range (t s : ℝ) (h : t ≠ -1) :
  (∀ k₁ k₂ : ℝ, k₁ ≠ k₂ → 
    ∃ (p₁ p₂ : ℝ × ℝ), p₁ = DoubleValuePoint k₁ ∧ p₂ = DoubleValuePoint k₂ ∧
    QuadraticFunction t s (p₁.1) = p₁.2 ∧ QuadraticFunction t s (p₂.1) = p₂.2) →
  -1 < s ∧ s < 0 := by
  sorry

end double_value_points_range_l176_17642


namespace stadium_entrance_exit_ways_l176_17688

/-- The number of gates on the south side of the stadium -/
def south_gates : ℕ := 4

/-- The number of gates on the north side of the stadium -/
def north_gates : ℕ := 3

/-- The total number of gates in the stadium -/
def total_gates : ℕ := south_gates + north_gates

/-- The number of different ways to enter and exit the stadium -/
def entrance_exit_ways : ℕ := total_gates * total_gates

theorem stadium_entrance_exit_ways :
  entrance_exit_ways = 49 := by sorry

end stadium_entrance_exit_ways_l176_17688


namespace apple_count_in_second_group_l176_17660

/-- The cost of an apple in dollars -/
def apple_cost : ℚ := 21/100

/-- The cost of an orange in dollars -/
def orange_cost : ℚ := 17/100

/-- The number of apples in the second group -/
def x : ℕ := 2

theorem apple_count_in_second_group :
  (6 * apple_cost + 3 * orange_cost = 177/100) →
  (↑x * apple_cost + 5 * orange_cost = 127/100) →
  (apple_cost = 21/100) →
  x = 2 := by
sorry

end apple_count_in_second_group_l176_17660


namespace vectors_perpendicular_l176_17622

def a : ℝ × ℝ := (-1, 2)
def b : ℝ × ℝ := (1, 3)

theorem vectors_perpendicular : 
  let c := (a.1 - b.1, a.2 - b.2)
  (a.1 * c.1 + a.2 * c.2 = 0) := by sorry

end vectors_perpendicular_l176_17622


namespace simplify_fraction_1_l176_17635

theorem simplify_fraction_1 (x : ℝ) (h : x ≠ 0 ∧ x ≠ 1) :
  1 / x + 1 / (x * (x - 1)) = 1 / (x - 1) := by
  sorry


end simplify_fraction_1_l176_17635


namespace julians_initial_debt_l176_17625

/-- Given that Julian will owe Jenny 28 dollars if he borrows 8 dollars more,
    prove that Julian's initial debt to Jenny is 20 dollars. -/
theorem julians_initial_debt (current_debt additional_borrow total_after_borrow : ℕ) :
  additional_borrow = 8 →
  total_after_borrow = 28 →
  total_after_borrow = current_debt + additional_borrow →
  current_debt = 20 :=
by
  sorry

end julians_initial_debt_l176_17625


namespace cake_price_problem_l176_17627

theorem cake_price_problem (original_price : ℝ) : 
  (8 * original_price = 320) → 
  (10 * (0.8 * original_price) = 320) → 
  original_price = 40 := by
sorry

end cake_price_problem_l176_17627


namespace tuesday_kids_count_l176_17653

def monday_kids : ℕ := 12
def total_kids : ℕ := 19

theorem tuesday_kids_count : total_kids - monday_kids = 7 := by
  sorry

end tuesday_kids_count_l176_17653


namespace triangle_angle_sum_l176_17637

theorem triangle_angle_sum (A B C : ℝ) (h1 : 0 < A) (h2 : 0 < B) (h3 : 0 < C) 
  (h4 : A + B + C = 180) (h5 : A ≤ B) (h6 : A ≤ C) (h7 : C ≤ B) (h8 : 2 * B = 5 * A) : 
  ∃ (m n : ℝ), m = max B C ∧ n = min B C ∧ m + n = 175 := by
sorry

end triangle_angle_sum_l176_17637


namespace sufficient_not_necessary_condition_l176_17697

theorem sufficient_not_necessary_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ a b, a > 0 → b > 0 → a > b → a^3 + b^3 > a^2*b + a*b^2) ∧
  (∃ a b, a > 0 ∧ b > 0 ∧ a^3 + b^3 > a^2*b + a*b^2 ∧ a ≤ b) :=
by sorry

end sufficient_not_necessary_condition_l176_17697


namespace petes_walking_distance_closest_to_2800_l176_17678

/-- Represents a pedometer with a maximum step count --/
structure Pedometer :=
  (max_count : ℕ)

/-- Represents Pete's walking data for a year --/
structure YearlyWalkingData :=
  (pedometer : Pedometer)
  (flips : ℕ)
  (final_reading : ℕ)
  (steps_per_mile : ℕ)

/-- Calculates the total steps walked in a year --/
def total_steps (data : YearlyWalkingData) : ℕ :=
  data.flips * (data.pedometer.max_count + 1) + data.final_reading

/-- Calculates the total miles walked in a year --/
def total_miles (data : YearlyWalkingData) : ℚ :=
  (total_steps data : ℚ) / data.steps_per_mile

/-- Theorem stating that Pete's walking distance is closest to 2800 miles --/
theorem petes_walking_distance_closest_to_2800 (data : YearlyWalkingData) 
  (h1 : data.pedometer.max_count = 89999)
  (h2 : data.flips = 55)
  (h3 : data.final_reading = 30000)
  (h4 : data.steps_per_mile = 1800) :
  ∃ (n : ℕ), n ≤ 50 ∧ |total_miles data - 2800| < |total_miles data - (2800 - n)| ∧ 
             |total_miles data - 2800| < |total_miles data - (2800 + n)| :=
  sorry

#eval total_miles { pedometer := { max_count := 89999 }, flips := 55, final_reading := 30000, steps_per_mile := 1800 }

end petes_walking_distance_closest_to_2800_l176_17678


namespace min_value_x_plus_2y_l176_17686

theorem min_value_x_plus_2y (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (heq : x + 2*y + 2*x*y = 8) : 
  ∀ z, z = x + 2*y → z ≥ 4 :=
by sorry

end min_value_x_plus_2y_l176_17686


namespace tank_capacity_correct_l176_17684

/-- The capacity of a tank in gallons -/
def tank_capacity : ℝ := 32

/-- The total amount of oil in gallons -/
def total_oil : ℝ := 728

/-- The number of tanks needed -/
def num_tanks : ℕ := 23

/-- Theorem stating that the tank capacity is approximately correct -/
theorem tank_capacity_correct : 
  ∃ ε > 0, ε < 1 ∧ |tank_capacity - total_oil / num_tanks| < ε :=
sorry

end tank_capacity_correct_l176_17684


namespace fraction_equality_l176_17658

theorem fraction_equality (x y : ℝ) (h : (1/x + 1/y)/(1/x - 1/y) = 101) : 
  (x - y)/(x + y) = -1/5101 := by sorry

end fraction_equality_l176_17658


namespace gcf_72_90_l176_17677

theorem gcf_72_90 : Nat.gcd 72 90 = 18 := by
  sorry

end gcf_72_90_l176_17677


namespace unique_integer_pair_existence_l176_17699

theorem unique_integer_pair_existence (a b : ℤ) :
  ∃! (x y : ℤ), (x + 2*y - a)^2 + (2*x - y - b)^2 ≤ 1 := by
  sorry

end unique_integer_pair_existence_l176_17699


namespace expectation_of_linear_combination_l176_17666

variable (ξ η : ℝ → ℝ)
variable (E : (ℝ → ℝ) → ℝ)

axiom linearity_of_expectation : ∀ (a b : ℝ) (X Y : ℝ → ℝ), E (λ ω => a * X ω + b * Y ω) = a * E X + b * E Y

theorem expectation_of_linear_combination
  (h1 : E ξ = 10)
  (h2 : E η = 3) :
  E (λ ω => 3 * ξ ω + 5 * η ω) = 45 := by
sorry

end expectation_of_linear_combination_l176_17666


namespace sum_first_6_primes_l176_17630

-- Define a function that returns the nth prime number
def nthPrime (n : ℕ) : ℕ := sorry

-- Define a function that sums the first n prime numbers
def sumFirstNPrimes (n : ℕ) : ℕ :=
  (List.range n).map (fun i => nthPrime (i + 1)) |>.sum

-- Theorem stating that the sum of the first 6 prime numbers is 41
theorem sum_first_6_primes : sumFirstNPrimes 6 = 41 := by sorry

end sum_first_6_primes_l176_17630


namespace sum_of_repeating_decimals_l176_17693

/-- Represents a repeating decimal with a single digit repeating infinitely. -/
def RepeatingDecimal (n : Nat) : ℚ := n / 9

theorem sum_of_repeating_decimals :
  RepeatingDecimal 6 + RepeatingDecimal 4 = 10 / 9 := by
  sorry

end sum_of_repeating_decimals_l176_17693


namespace equality_condition_l176_17639

theorem equality_condition (x y z : ℝ) : 
  x + y * z = (x + y) * (x + z) ↔ x + y + z = 0 := by
  sorry

end equality_condition_l176_17639


namespace alex_savings_l176_17657

def car_cost : ℝ := 14600
def trip_charge : ℝ := 1.5
def grocery_percentage : ℝ := 0.05
def num_trips : ℕ := 40
def grocery_value : ℝ := 800

theorem alex_savings (initial_savings : ℝ) : 
  initial_savings + 
  (num_trips : ℝ) * trip_charge + 
  grocery_percentage * grocery_value = 
  car_cost :=
by sorry

end alex_savings_l176_17657


namespace initial_walnut_trees_l176_17655

/-- The number of walnut trees initially in the park -/
def initial_trees : ℕ := sorry

/-- The number of walnut trees planted today -/
def planted_trees : ℕ := 33

/-- The total number of walnut trees after planting -/
def final_trees : ℕ := 55

/-- Theorem stating that the initial number of walnut trees is 22 -/
theorem initial_walnut_trees : initial_trees = 22 := by
  sorry

end initial_walnut_trees_l176_17655


namespace simplify_and_rationalize_l176_17647

theorem simplify_and_rationalize :
  (Real.sqrt 3 / Real.sqrt 4) * (Real.sqrt 5 / Real.sqrt 6) * (Real.sqrt 7 / Real.sqrt 8) = Real.sqrt 70 / 8 := by
  sorry

end simplify_and_rationalize_l176_17647


namespace quadratic_root_relation_l176_17665

theorem quadratic_root_relation (p q : ℝ) : 
  (∃ x y : ℝ, x^2 + p*x + q = 0 ∧ y^2 + p*y + q = 0 ∧ x - y = 2 ∧ x = 2*y) → p = -6 := by
  sorry

end quadratic_root_relation_l176_17665


namespace product_of_numbers_l176_17656

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 48) : x * y = 7 := by
  sorry

end product_of_numbers_l176_17656


namespace distribute_six_balls_two_boxes_limit_four_l176_17664

/-- The number of ways to distribute n distinguishable balls into 2 indistinguishable boxes,
    where no box can hold more than m balls. -/
def distributeWithLimit (n : ℕ) (m : ℕ) : ℕ := sorry

/-- The theorem stating that there are 25 ways to distribute 6 distinguishable balls
    into 2 indistinguishable boxes, where no box can hold more than 4 balls. -/
theorem distribute_six_balls_two_boxes_limit_four :
  distributeWithLimit 6 4 = 25 := by sorry

end distribute_six_balls_two_boxes_limit_four_l176_17664


namespace percent_of_360_l176_17609

theorem percent_of_360 : (35 / 100) * 360 = 126 := by
  sorry

end percent_of_360_l176_17609


namespace arithmetic_sequence_sum_l176_17675

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℕ) :
  arithmetic_sequence a →
  a 1 = 3 →
  a 2 = 9 →
  a 3 = 15 →
  a 7 = 33 →
  a 4 + a 5 + a 6 = 81 := by
sorry

end arithmetic_sequence_sum_l176_17675


namespace giorgios_class_size_l176_17601

theorem giorgios_class_size (cookies_per_student : ℕ) 
  (oatmeal_raisin_percentage : ℚ) (oatmeal_raisin_cookies : ℕ) :
  cookies_per_student = 2 →
  oatmeal_raisin_percentage = 1/10 →
  oatmeal_raisin_cookies = 8 →
  ∃ (total_students : ℕ), 
    total_students = 40 ∧
    (oatmeal_raisin_cookies : ℚ) / cookies_per_student = oatmeal_raisin_percentage * total_students :=
by
  sorry

#check giorgios_class_size

end giorgios_class_size_l176_17601


namespace expression_evaluation_l176_17662

theorem expression_evaluation (x y z : ℝ) 
  (hz : z = y - 11)
  (hy : y = x + 3)
  (hx : x = 5)
  (hd1 : x + 2 ≠ 0)
  (hd2 : y - 3 ≠ 0)
  (hd3 : z + 7 ≠ 0) :
  ((x + 3) / (x + 2)) * ((y - 2) / (y - 3)) * ((z + 9) / (z + 7)) = 72 / 35 := by
  sorry

end expression_evaluation_l176_17662


namespace cubic_polynomial_relation_l176_17617

def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 5*x - 7

theorem cubic_polynomial_relation (h : ℝ → ℝ) 
  (h_cubic : ∃ a b c d : ℝ, ∀ x, h x = a*x^3 + b*x^2 + c*x + d)
  (h_zero : h 0 = 7)
  (h_roots : ∀ r : ℝ, f r = 0 → ∃ s : ℝ, h s = 0 ∧ s = r^3) :
  h (-8) = -1813 := by
sorry

end cubic_polynomial_relation_l176_17617


namespace incorrect_proposition_statement_l176_17634

theorem incorrect_proposition_statement : 
  ¬(∀ (p q : Prop), (p ∧ q = False) → (p = False ∧ q = False)) := by
  sorry

end incorrect_proposition_statement_l176_17634


namespace intersection_of_M_and_N_l176_17624

def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {0, 1, 2}

theorem intersection_of_M_and_N : M ∩ N = {1, 2} := by
  sorry

end intersection_of_M_and_N_l176_17624


namespace square_perimeter_count_l176_17672

/-- The number of students on each side of the square arrangement -/
def side_length : ℕ := 10

/-- The number of students on the perimeter of a square arrangement -/
def perimeter_count (n : ℕ) : ℕ := 4 * n - 4

theorem square_perimeter_count :
  perimeter_count side_length = 36 := by
  sorry


end square_perimeter_count_l176_17672


namespace midpoint_coordinates_l176_17646

/-- Given two points P and Q in a plane, where Q is the midpoint of PR, 
    prove that R has specific coordinates. -/
theorem midpoint_coordinates (P Q : ℝ × ℝ) (h1 : P = (1, 3)) (h2 : Q = (4, 7)) 
    (h3 : Q = ((P.1 + R.1) / 2, (P.2 + R.2) / 2)) : R = (7, 11) :=
by sorry

end midpoint_coordinates_l176_17646


namespace trig_inequality_l176_17613

theorem trig_inequality (α β : Real) (h_α : 0 < α ∧ α < π / 2) (h_β : 0 < β ∧ β < π / 2) :
  Real.sin α ^ 3 * Real.cos β ^ 3 + Real.sin α ^ 3 * Real.sin β ^ 3 + Real.cos α ^ 3 ≥ Real.sqrt 3 / 3 := by
  sorry

end trig_inequality_l176_17613


namespace isosceles_triangle_relationship_l176_17683

-- Define the isosceles triangle
structure IsoscelesTriangle where
  x : ℝ  -- leg length
  y : ℝ  -- base length

-- Define the properties of the isosceles triangle
def validIsoscelesTriangle (t : IsoscelesTriangle) : Prop :=
  t.x > 0 ∧ t.y > 0 ∧ 2 * t.x > t.y ∧ t.x + t.y > t.x

-- Define the perimeter constraint
def hasPerimeter30 (t : IsoscelesTriangle) : Prop :=
  2 * t.x + t.y = 30

-- Define the relationship between x and y
def relationshipXY (t : IsoscelesTriangle) : Prop :=
  t.y = 30 - 2 * t.x

-- Define the constraints on x
def xConstraints (t : IsoscelesTriangle) : Prop :=
  15 / 2 < t.x ∧ t.x < 15

-- Theorem stating the relationship between x and y for the isosceles triangle
theorem isosceles_triangle_relationship (t : IsoscelesTriangle) :
  validIsoscelesTriangle t → hasPerimeter30 t → relationshipXY t ∧ xConstraints t :=
by sorry

end isosceles_triangle_relationship_l176_17683


namespace outfits_count_l176_17628

/-- The number of different outfits that can be made with a given number of shirts, pants, and ties. -/
def number_of_outfits (shirts : ℕ) (pants : ℕ) (ties : ℕ) : ℕ :=
  shirts * pants * (ties + 1)

/-- Theorem stating that the number of outfits with 7 shirts, 5 pants, and 4 ties (plus the option of no tie) is 175. -/
theorem outfits_count : number_of_outfits 7 5 4 = 175 := by
  sorry

end outfits_count_l176_17628


namespace power_five_hundred_mod_eighteen_l176_17619

theorem power_five_hundred_mod_eighteen : 
  (5 : ℤ) ^ 100 % 18 = 13 := by
  sorry

end power_five_hundred_mod_eighteen_l176_17619


namespace train_passing_time_l176_17629

/-- The time taken for a train to pass a man moving in the same direction -/
theorem train_passing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) : 
  train_length = 150 ∧ 
  train_speed = 62 * (1000 / 3600) ∧ 
  man_speed = 8 * (1000 / 3600) →
  train_length / (train_speed - man_speed) = 10 := by
  sorry


end train_passing_time_l176_17629


namespace smallest_number_l176_17650

/-- Converts a number from base b to decimal --/
def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldl (fun acc d => acc * base + d) 0

/-- The binary representation of 1111 --/
def binary_1111 : List Nat := [1, 1, 1, 1]

/-- The base-6 representation of 210 --/
def base6_210 : List Nat := [2, 1, 0]

/-- The base-4 representation of 1000 --/
def base4_1000 : List Nat := [1, 0, 0, 0]

/-- The octal representation of 101 --/
def octal_101 : List Nat := [1, 0, 1]

theorem smallest_number :
  to_decimal binary_1111 2 < to_decimal base6_210 6 ∧
  to_decimal binary_1111 2 < to_decimal base4_1000 4 ∧
  to_decimal binary_1111 2 < to_decimal octal_101 8 :=
by sorry

end smallest_number_l176_17650


namespace solve_equation_for_x_l176_17610

theorem solve_equation_for_x (x y : ℤ) 
  (h1 : x > y) 
  (h2 : y > 0) 
  (h3 : x + y + x * y = 80) : 
  x = 26 := by
sorry

end solve_equation_for_x_l176_17610


namespace prob_one_defective_is_half_l176_17600

/-- Represents the total number of items -/
def total_items : ℕ := 4

/-- Represents the number of genuine items -/
def genuine_items : ℕ := 3

/-- Represents the number of defective items -/
def defective_items : ℕ := 1

/-- Represents the number of items selected -/
def items_selected : ℕ := 2

/-- Calculates the number of ways to select k items from n items -/
def combinations (n k : ℕ) : ℕ := Nat.choose n k

/-- Calculates the probability of selecting exactly one defective item -/
def prob_one_defective : ℚ :=
  (combinations defective_items 1 * combinations genuine_items 1) /
  combinations total_items items_selected

theorem prob_one_defective_is_half :
  prob_one_defective = 1 / 2 := by sorry

end prob_one_defective_is_half_l176_17600


namespace unique_solution_l176_17670

/-- Calculates the cost per person based on the number of participants -/
def costPerPerson (n : ℕ) : ℕ :=
  if n ≤ 30 then 80
  else max 50 (80 - (n - 30))

/-- Calculates the total cost for a given number of participants -/
def totalCost (n : ℕ) : ℕ :=
  n * costPerPerson n

/-- States that there exists a unique number of employees that satisfies the problem conditions -/
theorem unique_solution : ∃! n : ℕ, n > 30 ∧ totalCost n = 2800 := by sorry

end unique_solution_l176_17670


namespace inequality_equivalence_l176_17603

theorem inequality_equivalence (x : ℝ) : 3 - 2 / (3 * x + 2) < 5 ↔ x > -2/3 :=
by sorry

end inequality_equivalence_l176_17603


namespace symmetric_point_theorem_l176_17691

-- Define the line of symmetry
def line_of_symmetry (x y : ℝ) : Prop := x + y + 1 = 0

-- Define the symmetric point relation
def symmetric_point (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  -- Midpoint of (x₁, y₁) and (x₂, y₂) lies on the line of symmetry
  line_of_symmetry ((x₁ + x₂) / 2) ((y₁ + y₂) / 2) ∧
  -- The line connecting (x₁, y₁) and (x₂, y₂) is perpendicular to the line of symmetry
  (y₂ - y₁) = (x₂ - x₁)

-- Theorem statement
theorem symmetric_point_theorem :
  symmetric_point 2 1 (-2) (-3) :=
sorry

end symmetric_point_theorem_l176_17691


namespace discount_calculation_l176_17663

theorem discount_calculation (original_price discount_percentage : ℝ) 
  (h1 : original_price = 80)
  (h2 : discount_percentage = 15) : 
  original_price * (1 - discount_percentage / 100) = 68 := by
sorry

end discount_calculation_l176_17663
