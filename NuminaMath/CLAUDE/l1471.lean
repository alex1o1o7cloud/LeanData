import Mathlib

namespace power_72_in_terms_of_m_and_n_l1471_147126

theorem power_72_in_terms_of_m_and_n (a m n : ℝ) 
  (h1 : 2^a = m) (h2 : 3^a = n) : 72^a = m^3 * n^2 := by
  sorry

end power_72_in_terms_of_m_and_n_l1471_147126


namespace quadratic_root_sqrt2_minus3_l1471_147134

theorem quadratic_root_sqrt2_minus3 :
  let f : ℝ → ℝ := λ x => x^2 + 6*x + 7
  f (Real.sqrt 2 - 3) = 0 := by sorry

end quadratic_root_sqrt2_minus3_l1471_147134


namespace sqrt_expressions_equality_l1471_147141

theorem sqrt_expressions_equality : 
  (2 * Real.sqrt (2/3) - 3 * Real.sqrt (3/2) + Real.sqrt 24 = (7 * Real.sqrt 6) / 6) ∧
  (Real.sqrt (25/2) + Real.sqrt 32 - Real.sqrt 18 - (Real.sqrt 2 - 1)^2 = (11 * Real.sqrt 2) / 2 - 3) := by
  sorry

end sqrt_expressions_equality_l1471_147141


namespace simplify_expression_l1471_147148

-- Define the trigonometric identity
axiom trig_identity (θ : Real) : Real.sin θ ^ 2 + Real.cos θ ^ 2 = 1

-- Define the theorem
theorem simplify_expression : 
  2 - Real.sin (21 * π / 180) ^ 2 - Real.cos (21 * π / 180) ^ 2 
  + Real.sin (17 * π / 180) ^ 4 + Real.sin (17 * π / 180) ^ 2 * Real.cos (17 * π / 180) ^ 2 
  + Real.cos (17 * π / 180) ^ 2 = 2 := by
  sorry

end simplify_expression_l1471_147148


namespace meaningful_expression_l1471_147115

theorem meaningful_expression (x : ℝ) : 
  (∃ y : ℝ, y = (Real.sqrt (x + 1)) / x) ↔ x ≥ -1 ∧ x ≠ 0 := by
  sorry

end meaningful_expression_l1471_147115


namespace function_satisfies_differential_equation_l1471_147117

/-- Prove that the function y = x(c - ln x) satisfies the differential equation (x - y) dx + x · dy = 0 -/
theorem function_satisfies_differential_equation (x : ℝ) (c : ℝ) :
  let y := x * (c - Real.log x)
  (x - y) * 1 + x * (c - Real.log x - 1) = 0 := by
  sorry

end function_satisfies_differential_equation_l1471_147117


namespace car_speed_proof_l1471_147127

/-- Proves that the speed of a car is 60 miles per hour given specific conditions -/
theorem car_speed_proof (
  fuel_efficiency : Real
) (
  tank_capacity : Real
) (
  travel_time : Real
) (
  fuel_used_ratio : Real
) (
  h1 : fuel_efficiency = 30 -- miles per gallon
) (
  h2 : tank_capacity = 12 -- gallons
) (
  h3 : travel_time = 5 -- hours
) (
  h4 : fuel_used_ratio = 0.8333333333333334 -- ratio of full tank
) : Real := by
  sorry

end car_speed_proof_l1471_147127


namespace largest_n_binomial_equality_l1471_147197

theorem largest_n_binomial_equality : ∃ (n : ℕ), (Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 n) ∧ 
  (∀ (m : ℕ), Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 m → m ≤ n) :=
by sorry

end largest_n_binomial_equality_l1471_147197


namespace max_value_F_H_surjective_implies_s_value_l1471_147175

noncomputable section

def f (x : ℝ) : ℝ := (Real.log x) / x

def F (x : ℝ) : ℝ := x^2 - x * f x

def H (s : ℝ) (x : ℝ) : ℝ :=
  if x ≥ s then x / (2 * Real.exp 1) else f x

theorem max_value_F :
  ∃ (x : ℝ), x ∈ Set.Icc (1/2) 2 ∧
  ∀ (y : ℝ), y ∈ Set.Icc (1/2) 2 → F x ≥ F y ∧
  F x = 4 - Real.log 2 := by sorry

theorem H_surjective_implies_s_value (s : ℝ) :
  (∀ (k : ℝ), ∃ (x : ℝ), H s x = k) →
  s = Real.sqrt (Real.exp 1) := by sorry

end max_value_F_H_surjective_implies_s_value_l1471_147175


namespace stick_markings_l1471_147147

/-- The number of unique markings on a one-foot stick marked in both 1/4 and 1/5 portions -/
def num_markings : ℕ := 9

/-- The set of markings for 1/4 portions -/
def quarter_markings : Set ℚ :=
  {0, 1/4, 1/2, 3/4, 1}

/-- The set of markings for 1/5 portions -/
def fifth_markings : Set ℚ :=
  {0, 1/5, 2/5, 3/5, 4/5, 1}

/-- The theorem stating that the number of unique markings is 9 -/
theorem stick_markings :
  (quarter_markings ∪ fifth_markings).ncard = num_markings :=
sorry

end stick_markings_l1471_147147


namespace equation_solution_l1471_147187

/-- Given two equations with the same solution for x, prove the value of an expression -/
theorem equation_solution (m n x : ℝ) : 
  (m + 3) * x^(|m| - 2) + 6 * m = 0 →  -- First equation
  n * x - 5 = x * (3 - n) →            -- Second equation
  (|m| - 2 = 0) →                      -- Condition for first-degree equation
  (m + x)^2000 * (-m^2 * n + x * n^2) + 1 = 1 := by
  sorry

end equation_solution_l1471_147187


namespace triangle_abc_properties_l1471_147184

theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) :
  a = 2 →
  Real.sqrt 3 * Real.sin B - Real.cos B = 1 →
  b^2 = a * c →
  B = π / 3 ∧ (1/2) * a * c * Real.sin B = Real.sqrt 3 :=
by sorry

end triangle_abc_properties_l1471_147184


namespace cherry_tomato_jars_l1471_147167

theorem cherry_tomato_jars (total_tomatoes : ℕ) (tomatoes_per_jar : ℕ) (h1 : total_tomatoes = 550) (h2 : tomatoes_per_jar = 14) :
  ∃ (jars : ℕ), jars = ((total_tomatoes + tomatoes_per_jar - 1) / tomatoes_per_jar) ∧ jars = 40 :=
by sorry

end cherry_tomato_jars_l1471_147167


namespace trigonometric_calculations_l1471_147146

theorem trigonometric_calculations :
  (2 * Real.cos (60 * π / 180) + |1 - 2 * Real.sin (45 * π / 180)| + (1/2)^0 = Real.sqrt 2 + 1) ∧
  (Real.sqrt (1 - 2 * Real.tan (60 * π / 180) + Real.tan (60 * π / 180)^2) - Real.tan (60 * π / 180) = -1) := by
  sorry

end trigonometric_calculations_l1471_147146


namespace center_is_three_l1471_147160

/-- Represents a 3x3 grid with numbers from 1 to 9 -/
def Grid := Fin 3 → Fin 3 → Fin 9

/-- The sum of elements on the main diagonal (top-left to bottom-right) -/
def mainDiagonalSum (g : Grid) : ℕ :=
  (g 0 0).val + (g 1 1).val + (g 2 2).val

/-- The sum of elements on the other diagonal (top-right to bottom-left) -/
def otherDiagonalSum (g : Grid) : ℕ :=
  (g 0 2).val + (g 1 1).val + (g 2 0).val

/-- All numbers in the grid are distinct and from 1 to 9 -/
def isValidGrid (g : Grid) : Prop :=
  ∀ i j, 1 ≤ (g i j).val ∧ (g i j).val ≤ 9 ∧
  ∀ i' j', (i ≠ i' ∨ j ≠ j') → g i j ≠ g i' j'

theorem center_is_three (g : Grid) 
  (h1 : isValidGrid g)
  (h2 : mainDiagonalSum g = 6)
  (h3 : otherDiagonalSum g = 20) :
  (g 1 1).val = 3 := by
  sorry

end center_is_three_l1471_147160


namespace pages_left_to_read_l1471_147105

/-- Calculates the number of pages left to read in a storybook -/
theorem pages_left_to_read (a b : ℕ) : a - 8 * b = a - 8 * b :=
by
  sorry

#check pages_left_to_read

end pages_left_to_read_l1471_147105


namespace inequality_proof_l1471_147159

theorem inequality_proof (a b : ℝ) (h1 : a ≠ b) (h2 : a < 0) :
  a < 2*b - b^2/a := by
  sorry

end inequality_proof_l1471_147159


namespace exponent_simplification_l1471_147170

theorem exponent_simplification (a b : ℝ) (m n : ℤ) 
    (ha : a > 0) (hb : b > 0) (hm : m ≠ 0) (hn : n ≠ 0) :
  (a^m)^(1/n) = a^(m/n) ∧
  (a^(1/n))^(n/m) = a^(1/m) ∧
  (a^n * b)^(1/n) = a * b^(1/n) ∧
  (a^n * b^m)^(1/(m*n)) = a^(1/m) * b^(1/n) ∧
  (a^n / b^m)^(1/(m*n)) = (a^(1/m)) / (b^(1/n)) :=
by sorry

end exponent_simplification_l1471_147170


namespace halloween_candy_division_l1471_147130

/-- Represents the fraction of candy taken by each person --/
def candy_fraction (total : ℚ) (remaining : ℚ) (ratio : ℚ) : ℚ :=
  ratio * remaining / total

/-- The problem of dividing Halloween candy --/
theorem halloween_candy_division :
  let total := 1
  let al_ratio := 4 / 10
  let bert_ratio := 3 / 10
  let carl_ratio := 2 / 10
  let dana_ratio := 1 / 10
  
  let al_takes := candy_fraction total total al_ratio
  let bert_takes := candy_fraction total (total - al_takes) bert_ratio
  let carl_takes := candy_fraction total (total - al_takes - bert_takes) carl_ratio
  let dana_takes := candy_fraction total (total - al_takes - bert_takes - carl_takes) dana_ratio
  
  total - (al_takes + bert_takes + carl_takes + dana_takes) = 27 / 125 :=
by sorry

end halloween_candy_division_l1471_147130


namespace sandbox_length_l1471_147168

/-- The length of a rectangular sandbox given its width and area -/
theorem sandbox_length (width : ℝ) (area : ℝ) (h1 : width = 146) (h2 : area = 45552) :
  area / width = 312 := by
  sorry

end sandbox_length_l1471_147168


namespace p_shape_points_l1471_147124

/-- Represents a "P" shape formed from a square -/
structure PShape :=
  (side_length : ℕ)

/-- Counts the number of distinct points on a "P" shape -/
def count_points (p : PShape) : ℕ :=
  3 * (p.side_length + 1) - 2

/-- Theorem stating the number of points on a "P" shape with side length 10 -/
theorem p_shape_points :
  let p : PShape := { side_length := 10 }
  count_points p = 31 := by sorry

end p_shape_points_l1471_147124


namespace unique_root_between_zero_and_e_l1471_147190

/-- The natural logarithm function -/
noncomputable def ln : ℝ → ℝ := Real.log

/-- The mathematical constant e -/
noncomputable def e : ℝ := Real.exp 1

theorem unique_root_between_zero_and_e (a : ℝ) (h1 : 0 < a) (h2 : a < e) :
  ∃! x : ℝ, x = ln (a * x) := by sorry

end unique_root_between_zero_and_e_l1471_147190


namespace same_price_at_12_sheets_l1471_147174

/-- The price per sheet for John's Photo World -/
def johns_price_per_sheet : ℚ := 275/100

/-- The sitting fee for John's Photo World -/
def johns_sitting_fee : ℚ := 125

/-- The price per sheet for Sam's Picture Emporium -/
def sams_price_per_sheet : ℚ := 150/100

/-- The sitting fee for Sam's Picture Emporium -/
def sams_sitting_fee : ℚ := 140

/-- The total cost for John's Photo World given a number of sheets -/
def johns_total_cost (sheets : ℚ) : ℚ := johns_price_per_sheet * sheets + johns_sitting_fee

/-- The total cost for Sam's Picture Emporium given a number of sheets -/
def sams_total_cost (sheets : ℚ) : ℚ := sams_price_per_sheet * sheets + sams_sitting_fee

theorem same_price_at_12_sheets :
  ∃ (sheets : ℚ), sheets = 12 ∧ johns_total_cost sheets = sams_total_cost sheets :=
by sorry

end same_price_at_12_sheets_l1471_147174


namespace population_growth_proof_l1471_147156

/-- The annual population growth rate -/
def annual_growth_rate : ℝ := 0.10

/-- The population after 2 years -/
def final_population : ℝ := 18150

/-- The present population of the town -/
def present_population : ℝ := 15000

/-- Theorem stating that the present population results in the final population after 2 years of growth -/
theorem population_growth_proof :
  present_population * (1 + annual_growth_rate)^2 = final_population :=
by sorry

end population_growth_proof_l1471_147156


namespace dolphin_count_l1471_147123

theorem dolphin_count (initial : ℕ) (joining_factor : ℕ) (h1 : initial = 65) (h2 : joining_factor = 3) :
  initial + joining_factor * initial = 260 := by
  sorry

end dolphin_count_l1471_147123


namespace num_different_results_is_1024_l1471_147195

/-- The expression as a list of integers -/
def expression : List Int := [1, -2, -4, -8, -16, -32, -64, -128, -256, -512, -1024]

/-- The number of terms in the expression that can have their sign changed -/
def num_changeable_terms : Nat := expression.length - 1

/-- The number of different results obtainable by placing parentheses in the expression -/
def num_different_results : Nat := 2^num_changeable_terms

theorem num_different_results_is_1024 : num_different_results = 1024 := by
  sorry

end num_different_results_is_1024_l1471_147195


namespace sum_of_max_and_min_g_l1471_147111

-- Define the function g(x)
def g (x : ℝ) : ℝ := |x - 3| + |x - 5| - |2*x - 8|

-- Define the interval [3, 10]
def I : Set ℝ := {x | 3 ≤ x ∧ x ≤ 10}

-- Theorem statement
theorem sum_of_max_and_min_g :
  ∃ (max_g min_g : ℝ),
    (∀ x ∈ I, g x ≤ max_g) ∧
    (∃ x ∈ I, g x = max_g) ∧
    (∀ x ∈ I, min_g ≤ g x) ∧
    (∃ x ∈ I, g x = min_g) ∧
    max_g + min_g = 2 := by
  sorry

end sum_of_max_and_min_g_l1471_147111


namespace simple_interest_calculation_l1471_147158

theorem simple_interest_calculation (principal : ℝ) (rate : ℝ) (time : ℝ) :
  principal = 2323 → rate = 8 → time = 5 →
  (principal * rate * time) / 100 = 1861.84 := by
  sorry

end simple_interest_calculation_l1471_147158


namespace triangle_perimeter_l1471_147199

theorem triangle_perimeter (a b c : ℝ) (A B C : ℝ) : 
  A = π / 3 →  -- 60 degrees in radians
  (1 / 2) * b * c * Real.sin A = (15 * Real.sqrt 3) / 4 →  -- Area formula
  5 * Real.sin B = 3 * Real.sin C →
  a + b + c = 8 + Real.sqrt 19 := by
sorry

end triangle_perimeter_l1471_147199


namespace jones_pants_count_l1471_147191

/-- Represents the number of pants Mr. Jones has -/
def num_pants : ℕ := 40

/-- Represents the number of shirts Mr. Jones has for each pair of pants -/
def shirts_per_pants : ℕ := 6

/-- Represents the total number of pieces of clothes Mr. Jones owns -/
def total_clothes : ℕ := 280

/-- Theorem stating that the number of pants Mr. Jones has is 40 -/
theorem jones_pants_count :
  num_pants * (shirts_per_pants + 1) = total_clothes :=
by sorry

end jones_pants_count_l1471_147191


namespace correct_observation_value_l1471_147139

theorem correct_observation_value (n : ℕ) (initial_mean corrected_mean wrong_value : ℝ) 
  (h1 : n = 50)
  (h2 : initial_mean = 36)
  (h3 : corrected_mean = 36.5)
  (h4 : wrong_value = 23) :
  let total_sum := n * initial_mean
  let corrected_sum := n * corrected_mean
  corrected_sum = total_sum - wrong_value + (total_sum - wrong_value + corrected_sum - total_sum) := by
  sorry

end correct_observation_value_l1471_147139


namespace modular_congruence_unique_solution_l1471_147176

theorem modular_congruence_unique_solution : ∃! m : ℤ, 0 ≤ m ∧ m < 31 ∧ 79453 ≡ m [ZMOD 31] := by
  sorry

end modular_congruence_unique_solution_l1471_147176


namespace leg_head_difference_l1471_147193

/-- Represents the number of legs for a buffalo -/
def buffalo_legs : ℕ := 4

/-- Represents the number of legs for a duck -/
def duck_legs : ℕ := 2

/-- Represents the number of heads for any animal -/
def animal_head : ℕ := 1

/-- The number of buffaloes in the group -/
def num_buffaloes : ℕ := 12

theorem leg_head_difference (num_ducks : ℕ) :
  (num_buffaloes * buffalo_legs + num_ducks * duck_legs) -
  2 * (num_buffaloes * animal_head + num_ducks * animal_head) = 24 :=
by sorry

end leg_head_difference_l1471_147193


namespace quadratic_factorization_l1471_147178

theorem quadratic_factorization (c d : ℕ) (hc : c > d) :
  (∀ x, x^2 - 18*x + 72 = (x - c) * (x - d)) →
  4*d - c = 12 := by
  sorry

end quadratic_factorization_l1471_147178


namespace two_adults_in_group_l1471_147103

/-- Represents the restaurant bill problem --/
def restaurant_bill_problem (num_children : ℕ) (meal_cost : ℕ) (total_bill : ℕ) : Prop :=
  ∃ (num_adults : ℕ), 
    num_adults * meal_cost + num_children * meal_cost = total_bill

/-- Proves that there are 2 adults in the group --/
theorem two_adults_in_group : 
  restaurant_bill_problem 5 3 21 → 
  ∃ (num_adults : ℕ), num_adults = 2 ∧ restaurant_bill_problem 5 3 21 :=
by sorry

end two_adults_in_group_l1471_147103


namespace simplify_fraction_multiplication_l1471_147128

theorem simplify_fraction_multiplication : (123 : ℚ) / 9999 * 41 = 1681 / 3333 := by
  sorry

end simplify_fraction_multiplication_l1471_147128


namespace team_games_count_l1471_147155

/-- Proves that a team with the given win percentages played 175 games in total -/
theorem team_games_count (first_hundred_win_rate : Real) 
                          (remaining_win_rate : Real)
                          (total_win_rate : Real)
                          (h1 : first_hundred_win_rate = 0.85)
                          (h2 : remaining_win_rate = 0.5)
                          (h3 : total_win_rate = 0.7) : 
  ∃ (total_games : ℕ), total_games = 175 ∧ 
    (first_hundred_win_rate * 100 + remaining_win_rate * (total_games - 100)) / total_games = total_win_rate :=
by
  sorry


end team_games_count_l1471_147155


namespace union_of_A_and_B_l1471_147196

def A : Set ℕ := {0, 1}
def B : Set ℕ := {1, 2}

theorem union_of_A_and_B : A ∪ B = {0, 1, 2} := by
  sorry

end union_of_A_and_B_l1471_147196


namespace c_share_is_36_l1471_147180

/-- Represents the usage of the pasture by a person -/
structure Usage where
  oxen : ℕ
  months : ℕ

/-- Calculates the total ox-months for a usage -/
def oxMonths (u : Usage) : ℕ := u.oxen * u.months

/-- Represents the rental situation -/
structure RentalSituation where
  usageA : Usage
  usageB : Usage
  usageC : Usage
  totalRent : ℚ

/-- The specific rental situation from the problem -/
def problemSituation : RentalSituation := {
  usageA := { oxen := 10, months := 7 }
  usageB := { oxen := 12, months := 5 }
  usageC := { oxen := 15, months := 3 }
  totalRent := 140
}

/-- Calculates C's share of the rent -/
def cShare (s : RentalSituation) : ℚ :=
  let totalUsage := oxMonths s.usageA + oxMonths s.usageB + oxMonths s.usageC
  let costPerOxMonth := s.totalRent / totalUsage
  (oxMonths s.usageC : ℚ) * costPerOxMonth

/-- Theorem stating that C's share in the problem situation is 36 -/
theorem c_share_is_36 : cShare problemSituation = 36 := by
  sorry

end c_share_is_36_l1471_147180


namespace functional_equation_solution_l1471_147182

/-- A polynomial of degree 2015 -/
def Polynomial2015 := Polynomial ℝ

/-- An odd polynomial of degree 2015 -/
def OddPolynomial2015 := {Q : Polynomial2015 // ∀ x, Q.eval (-x) = -Q.eval x}

/-- The functional equation P(x) + P(1-x) = 1 -/
def SatisfiesFunctionalEquation (P : Polynomial2015) : Prop :=
  ∀ x, P.eval x + P.eval (1 - x) = 1

theorem functional_equation_solution :
  ∀ P : Polynomial2015, SatisfiesFunctionalEquation P →
  ∃ Q : OddPolynomial2015, ∀ x, P.eval x = Q.val.eval (1/2 - x) + 1/2 :=
sorry

end functional_equation_solution_l1471_147182


namespace base_6_to_base_3_conversion_l1471_147153

def base_6_to_decimal (n : ℕ) : ℕ :=
  2 * 6^2 + 1 * 6^1 + 0 * 6^0

def base_3_to_decimal (n : ℕ) : ℕ :=
  2 * 3^3 + 2 * 3^2 + 2 * 3^1 + 0 * 3^0

theorem base_6_to_base_3_conversion :
  base_6_to_decimal 210 = base_3_to_decimal 2220 := by
  sorry

end base_6_to_base_3_conversion_l1471_147153


namespace recurrence_2004_values_l1471_147102

/-- A sequence of real numbers satisfying the given recurrence relation -/
def RecurrenceSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * (a n + 2)

/-- The set of possible values for the 2004th term of the sequence -/
def PossibleValues (a : ℕ → ℝ) : Set ℝ :=
  {x : ℝ | ∃ (seq : ℕ → ℝ), RecurrenceSequence seq ∧ seq 2004 = x}

/-- The theorem stating that the set of possible values for a₂₀₀₄ is [-1, ∞) -/
theorem recurrence_2004_values :
  ∀ a : ℕ → ℝ, RecurrenceSequence a →
  PossibleValues a = Set.Ici (-1) :=
sorry

end recurrence_2004_values_l1471_147102


namespace quadratic_equation_roots_l1471_147113

theorem quadratic_equation_roots (p q : ℤ) (h1 : p + q = 28) : 
  ∃ (x₁ x₂ : ℤ), x₁ > 0 ∧ x₂ > 0 ∧ x₁^2 + p*x₁ + q = 0 ∧ x₂^2 + p*x₂ + q = 0 ∧ 
  ((x₁ = 30 ∧ x₂ = 2) ∨ (x₁ = 2 ∧ x₂ = 30)) := by
  sorry

end quadratic_equation_roots_l1471_147113


namespace min_sum_for_product_4410_l1471_147108

theorem min_sum_for_product_4410 (a b c d : ℕ+) 
  (h : a * b * c * d = 4410) : 
  (∀ w x y z : ℕ+, w * x * y * z = 4410 → a + b + c + d ≤ w + x + y + z) ∧ 
  (∃ w x y z : ℕ+, w * x * y * z = 4410 ∧ w + x + y + z = 69) :=
by sorry

end min_sum_for_product_4410_l1471_147108


namespace a_range_l1471_147133

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 7 then (3 - a) * x - 3 else a^(x - 6)

def is_increasing (seq : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n < m → seq n < seq m

theorem a_range (a : ℝ) :
  (∀ n : ℕ+, is_increasing (λ n => f a n)) →
  a ∈ Set.Ioo 2 3 :=
sorry

end a_range_l1471_147133


namespace lawn_length_is_four_l1471_147169

-- Define the lawn's properties
def lawn_area : ℝ := 20
def lawn_width : ℝ := 5

-- Theorem statement
theorem lawn_length_is_four :
  ∃ (length : ℝ), length * lawn_width = lawn_area ∧ length = 4 :=
by
  sorry

end lawn_length_is_four_l1471_147169


namespace children_count_proof_l1471_147164

theorem children_count_proof :
  ∃! n : ℕ, 100 ≤ n ∧ n ≤ 150 ∧ n % 8 = 5 ∧ n % 10 = 7 ∧ n = 125 := by
  sorry

end children_count_proof_l1471_147164


namespace trigonometric_expression_equals_sqrt_three_l1471_147118

theorem trigonometric_expression_equals_sqrt_three (α : Real) (h : α = -35 * Real.pi / 6) :
  (2 * Real.sin (Real.pi + α) * Real.cos (Real.pi - α) - Real.cos (Real.pi + α)) /
  (1 + Real.sin α ^ 2 + Real.sin (Real.pi - α) - Real.cos (Real.pi + α) ^ 2) = Real.sqrt 3 := by
  sorry

end trigonometric_expression_equals_sqrt_three_l1471_147118


namespace students_count_l1471_147198

/-- The total number of students in a rectangular arrangement -/
def total_students (left right front back : ℕ) : ℕ :=
  (left + right - 1) * (front + back - 1)

/-- Theorem stating that the total number of students is 399 -/
theorem students_count (left right front back : ℕ) 
  (h1 : left = 7)
  (h2 : right = 13)
  (h3 : front = 8)
  (h4 : back = 14) :
  total_students left right front back = 399 := by
  sorry

#eval total_students 7 13 8 14

end students_count_l1471_147198


namespace circular_garden_area_l1471_147120

-- Define the radius of the garden
def radius : ℝ := 16

-- Define the relationship between circumference and area
def fence_area_relation (circumference area : ℝ) : Prop :=
  circumference = (1/8) * area

-- Theorem statement
theorem circular_garden_area :
  let circumference := 2 * Real.pi * radius
  let area := Real.pi * radius^2
  fence_area_relation circumference area →
  area = 256 * Real.pi := by
sorry

end circular_garden_area_l1471_147120


namespace smallest_bound_is_two_l1471_147135

/-- A function satisfying the given conditions -/
def SatisfyingFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, x ∈ Set.Icc 0 1 → f x ≥ 0) ∧
  (∀ x₁ x₂, x₁ ≥ 0 → x₂ ≥ 0 → x₁ + x₂ ≤ 1 → f (x₁ + x₂) ≥ f x₁ + f x₂) ∧
  f 0 = 0 ∧ f 1 = 1

/-- The theorem stating that 2 is the smallest positive number c such that f(x) ≤ cx for all x ∈ [0,1] -/
theorem smallest_bound_is_two (f : ℝ → ℝ) (h : SatisfyingFunction f) :
  (∀ c > 0, (∀ x ∈ Set.Icc 0 1, f x ≤ c * x) → c ≥ 2) ∧
  (∀ x ∈ Set.Icc 0 1, f x ≤ 2 * x) :=
sorry

end smallest_bound_is_two_l1471_147135


namespace equation_solution_l1471_147144

theorem equation_solution :
  let f : ℝ → ℝ := λ x => 1 / (x + 3) + 3 * x / (x + 3) - 5 / (x + 3) + 2 / (x - 1)
  ∀ x : ℝ, f x = 5 ↔ x = (-11 + Real.sqrt 257) / 4 ∨ x = (-11 - Real.sqrt 257) / 4 := by
  sorry

end equation_solution_l1471_147144


namespace point_transformation_l1471_147122

/-- Rotation of a point (x, y) by 90° counterclockwise around (h, k) -/
def rotate90 (x y h k : ℝ) : ℝ × ℝ :=
  (k - (y - h) + h, h + (x - h) + k)

/-- Reflection of a point (x, y) about the line y = x -/
def reflectYEqualsX (x y : ℝ) : ℝ × ℝ :=
  (y, x)

theorem point_transformation (a b : ℝ) :
  let (x₁, y₁) := rotate90 a b 2 3
  let (x₂, y₂) := reflectYEqualsX x₁ y₁
  (x₂ = -3 ∧ y₂ = 1) → b - a = -6 := by
  sorry

end point_transformation_l1471_147122


namespace perimeter_of_C_l1471_147112

-- Define squares A, B, and C
def square_A : Real → Real := λ s ↦ 4 * s
def square_B : Real → Real := λ s ↦ 4 * s
def square_C : Real → Real := λ s ↦ 4 * s

-- Define the conditions
def perimeter_A : Real := 20
def perimeter_B : Real := 40

-- Define the relationship between side lengths
def side_C (side_A side_B : Real) : Real := 2 * (side_A + side_B)

-- Theorem to prove
theorem perimeter_of_C (side_A side_B : Real) 
  (h1 : square_A side_A = perimeter_A)
  (h2 : square_B side_B = perimeter_B)
  : square_C (side_C side_A side_B) = 120 := by
  sorry


end perimeter_of_C_l1471_147112


namespace problem_statement_l1471_147181

theorem problem_statement : 
  |1 - Real.sqrt 3| + 2 * Real.cos (30 * π / 180) - Real.sqrt 12 - 2023 = -2024 := by
  sorry

end problem_statement_l1471_147181


namespace inequality_proof_l1471_147173

theorem inequality_proof (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  (y*z + z*x + x*y)^2 * (x + y + z) ≥ 4 * x*y*z * (x^2 + y^2 + z^2) := by
  sorry

end inequality_proof_l1471_147173


namespace subset_iff_positive_l1471_147132

def A : Set ℝ := {2, 0, 1, 6}
def B (a : ℝ) : Set ℝ := {x : ℝ | x + a > 0}

theorem subset_iff_positive (a : ℝ) : A ⊆ B a ↔ a > 0 := by
  sorry

end subset_iff_positive_l1471_147132


namespace rectangles_on_grid_l1471_147121

/-- The number of rectangles on a 4x4 grid with 5 points in each direction -/
def num_rectangles : ℕ := 100

/-- The number of points in each direction of the grid -/
def points_per_direction : ℕ := 5

/-- Theorem stating the number of rectangles on the grid -/
theorem rectangles_on_grid : 
  (Nat.choose points_per_direction 2) * (Nat.choose points_per_direction 2) = num_rectangles := by
  sorry

end rectangles_on_grid_l1471_147121


namespace round_85960_to_three_sig_figs_l1471_147107

/-- Rounds a number to a specified number of significant figures using the round-half-up method -/
def roundToSigFigs (x : ℝ) (sigFigs : ℕ) : ℝ :=
  sorry

/-- Theorem: Rounding 85960 to three significant figures using the round-half-up method results in 8.60 × 10^4 -/
theorem round_85960_to_three_sig_figs :
  roundToSigFigs 85960 3 = 8.60 * (10 : ℝ)^4 :=
sorry

end round_85960_to_three_sig_figs_l1471_147107


namespace remainder_sum_l1471_147185

theorem remainder_sum (n : ℤ) (h : n % 21 = 13) : (n % 3 + n % 7 = 7) := by
  sorry

end remainder_sum_l1471_147185


namespace roots_of_equation_l1471_147110

theorem roots_of_equation : 
  ∀ x : ℝ, (x^3 - 2*x^2 - x + 2)*(x - 5) = 0 ↔ x = -1 ∨ x = 1 ∨ x = 2 ∨ x = 5 := by
  sorry

end roots_of_equation_l1471_147110


namespace min_bird_species_l1471_147106

theorem min_bird_species (total_birds : ℕ) (h_total : total_birds = 2021) :
  let min_species := (total_birds + 1) / 2
  ∀ (num_species : ℕ),
    (∀ (i j : ℕ) (species : ℕ → ℕ),
      i < j ∧ j < total_birds ∧ species i = species j →
      ∃ (k : ℕ), k ∈ Finset.range (j - i - 1) ∧ species (i + k + 1) ≠ species i) →
    num_species ≥ min_species :=
by sorry

end min_bird_species_l1471_147106


namespace bottle_production_time_l1471_147152

/-- Given that 6 identical machines produce 270 bottles per minute at a constant rate,
    prove that 5 such machines will take 4 minutes to produce 900 bottles. -/
theorem bottle_production_time (rate : ℕ) (h1 : 6 * rate = 270) : 
  (900 : ℕ) / (5 * rate) = 4 := by
  sorry

end bottle_production_time_l1471_147152


namespace max_area_triangle_l1471_147194

/-- Given points A and B, and a circle with two symmetric points M and N,
    prove that the maximum area of triangle PAB is 3 + √2 --/
theorem max_area_triangle (k : ℝ) : 
  let A : ℝ × ℝ := (-2, 0)
  let B : ℝ × ℝ := (0, 2)
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 + k*x = 0}
  let symmetry_line := {(x, y) : ℝ × ℝ | x - y - 1 = 0}
  ∃ (M N : ℝ × ℝ), M ∈ circle ∧ N ∈ circle ∧ M ≠ N ∧
    (∃ (c : ℝ × ℝ), c ∈ symmetry_line ∧ 
      (M.1 - c.1)^2 + (M.2 - c.2)^2 = (N.1 - c.1)^2 + (N.2 - c.2)^2) →
  (⨆ (P : ℝ × ℝ) (h : P ∈ circle), 
    abs ((P.1 - A.1) * (B.2 - A.2) - (P.2 - A.2) * (B.1 - A.1)) / 2) = 3 + Real.sqrt 2 :=
by sorry

end max_area_triangle_l1471_147194


namespace yellow_ball_probability_l1471_147188

/-- The probability of drawing a yellow ball from a bag with yellow, red, and white balls -/
theorem yellow_ball_probability (yellow red white : ℕ) : 
  yellow = 5 → red = 8 → white = 7 → 
  (yellow : ℚ) / (yellow + red + white) = 1 / 4 := by
  sorry

end yellow_ball_probability_l1471_147188


namespace min_x_plus_y_l1471_147171

def is_median (x : ℝ) : Prop := 
  x ≥ 2 ∧ x ≤ 4

def average_condition (x y : ℝ) : Prop :=
  (-1 + 5 + (-1/x) + y) / 4 = 3

theorem min_x_plus_y (x y : ℝ) 
  (h1 : is_median x) 
  (h2 : average_condition x y) : 
  x + y ≥ 21/2 := by
  sorry

end min_x_plus_y_l1471_147171


namespace absolute_value_four_l1471_147165

theorem absolute_value_four (x : ℝ) : |x| = 4 → x = 4 ∨ x = -4 := by
  sorry

end absolute_value_four_l1471_147165


namespace dons_average_speed_l1471_147192

theorem dons_average_speed 
  (ambulance_speed : ℝ) 
  (ambulance_time : ℝ) 
  (don_time : ℝ) 
  (h1 : ambulance_speed = 60) 
  (h2 : ambulance_time = 1/4) 
  (h3 : don_time = 1/2) : 
  (ambulance_speed * ambulance_time) / don_time = 30 := by
sorry

end dons_average_speed_l1471_147192


namespace probability_of_one_in_first_20_rows_l1471_147104

/-- Calculates the number of elements in the first n rows of Pascal's Triangle. -/
def elementsInRows (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Calculates the number of ones in the first n rows of Pascal's Triangle. -/
def onesInRows (n : ℕ) : ℕ := if n = 0 then 1 else 2 * n - 1

/-- The probability of selecting a 1 from the first n rows of Pascal's Triangle. -/
def probabilityOfOne (n : ℕ) : ℚ :=
  (onesInRows n) / (elementsInRows n)

theorem probability_of_one_in_first_20_rows :
  probabilityOfOne 20 = 13 / 70 := by
  sorry

end probability_of_one_in_first_20_rows_l1471_147104


namespace opposite_def_opposite_of_point_one_l1471_147157

/-- The opposite of a real number -/
def opposite (x : ℝ) : ℝ := -x

/-- The property that defines the opposite of a number -/
theorem opposite_def (x : ℝ) : x + opposite x = 0 := by sorry

/-- Proof that the opposite of 0.1 is -0.1 -/
theorem opposite_of_point_one : opposite 0.1 = -0.1 := by sorry

end opposite_def_opposite_of_point_one_l1471_147157


namespace bee_return_theorem_l1471_147149

/-- Represents a position on the hexagonal grid -/
structure HexPosition where
  x : ℤ
  y : ℤ

/-- Represents a move on the hexagonal grid -/
structure HexMove where
  direction : Fin 6
  length : ℕ

/-- Applies a move to a position -/
def applyMove (pos : HexPosition) (move : HexMove) : HexPosition :=
  sorry

/-- Applies a sequence of moves to a position -/
def applyMoves (pos : HexPosition) (moves : List HexMove) : HexPosition :=
  sorry

/-- Generates a sequence of moves for a given N -/
def generateMoves (N : ℕ) : List HexMove :=
  sorry

theorem bee_return_theorem (N : ℕ) (h : N ≥ 3) :
  ∃ (startPos : HexPosition),
    applyMoves startPos (generateMoves N) = startPos :=
  sorry

end bee_return_theorem_l1471_147149


namespace average_fishes_is_45_2_l1471_147131

-- Define the number of lakes
def num_lakes : ℕ := 5

-- Define the number of fishes caught in each lake
def lake_marion : ℕ := 38
def lake_norman : ℕ := 52
def lake_wateree : ℕ := 27
def lake_wylie : ℕ := 45
def lake_keowee : ℕ := 64

-- Define the total number of fishes caught
def total_fishes : ℕ := lake_marion + lake_norman + lake_wateree + lake_wylie + lake_keowee

-- Define the average number of fishes caught per lake
def average_fishes : ℚ := total_fishes / num_lakes

-- Theorem statement
theorem average_fishes_is_45_2 : average_fishes = 45.2 := by
  sorry

end average_fishes_is_45_2_l1471_147131


namespace solve_equation_l1471_147136

theorem solve_equation (x : ℝ) : ((17.28 / x) / (3.6 * 0.2) = 2) → x = 12 := by
  sorry

end solve_equation_l1471_147136


namespace simplify_product_l1471_147140

theorem simplify_product : (625 : ℝ) ^ (1/4) * (343 : ℝ) ^ (1/3) = 35 := by
  sorry

end simplify_product_l1471_147140


namespace rhombus_side_length_l1471_147143

theorem rhombus_side_length 
  (diag1 diag2 : ℝ) 
  (m : ℝ) 
  (h1 : diag1^2 - 10*diag1 + m = 0)
  (h2 : diag2^2 - 10*diag2 + m = 0)
  (h3 : diag1 * diag2 / 2 = 11) :
  ∃ (side : ℝ), side^2 = 14 ∧ 
    side = Real.sqrt ((diag1/2)^2 + (diag2/2)^2) :=
by sorry

end rhombus_side_length_l1471_147143


namespace function_monotonicity_l1471_147119

def is_monotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y ∨ f y ≤ f x

theorem function_monotonicity (f : ℝ → ℝ) 
  (h : ∀ a b x, a < x ∧ x < b → min (f a) (f b) < f x ∧ f x < max (f a) (f b)) :
  is_monotonic f := by
  sorry

end function_monotonicity_l1471_147119


namespace rectangle_area_l1471_147150

/-- Theorem: Area of a rectangle with specific properties -/
theorem rectangle_area (length : ℝ) (width : ℝ) : 
  length = 12 →
  width * 1.2 = length →
  length * width = 120 :=
by
  sorry

end rectangle_area_l1471_147150


namespace three_tangent_planes_l1471_147162

/-- Represents a sphere in 3D space -/
structure Sphere where
  center : ℝ × ℝ × ℝ
  radius : ℝ

/-- Represents an equilateral triangle in 3D space -/
structure EquilateralTriangle where
  vertices : List (ℝ × ℝ × ℝ)
  side_length : ℝ

/-- Configuration of three spheres whose centers form an equilateral triangle -/
structure SphereConfiguration where
  spheres : List Sphere
  triangle : EquilateralTriangle

/-- Returns the number of planes tangent to all spheres in the configuration -/
def count_tangent_planes (config : SphereConfiguration) : ℕ :=
  sorry

theorem three_tangent_planes (config : SphereConfiguration) :
  (config.spheres.length = 3) →
  (config.triangle.side_length = 11) →
  (config.spheres.map Sphere.radius = [3, 4, 6]) →
  (count_tangent_planes config = 3) :=
sorry

end three_tangent_planes_l1471_147162


namespace a_power_b_equals_sixteen_l1471_147189

theorem a_power_b_equals_sixteen (a b : ℝ) : (a - 4)^2 + |2 - b| = 0 → a^b = 16 := by
  sorry

end a_power_b_equals_sixteen_l1471_147189


namespace problem_statement_l1471_147183

/-- A complex number is purely imaginary if its real part is zero -/
def isPurelyImaginary (z : ℂ) : Prop := z.re = 0

/-- The imaginary unit -/
def i : ℂ := Complex.I

/-- The equation given in the problem -/
def equation (z a : ℂ) : Prop := (2 + i) * z = 1 + a * i^3

/-- A complex number is in Quadrant IV if its real part is positive and imaginary part is negative -/
def inQuadrantIV (z : ℂ) : Prop := z.re > 0 ∧ z.im < 0

theorem problem_statement (z a : ℂ) :
  isPurelyImaginary z → equation z a → inQuadrantIV (a + z) :=
by sorry

end problem_statement_l1471_147183


namespace geometric_sequence_problem_l1471_147145

/-- A geometric sequence is a sequence where the ratio between any two consecutive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Given a geometric sequence a with a₁ = -16 and a₄ = 8, prove that a₇ = -4 -/
theorem geometric_sequence_problem (a : ℕ → ℝ) 
    (h_geom : IsGeometricSequence a) 
    (h_a1 : a 1 = -16) 
    (h_a4 : a 4 = 8) : 
  a 7 = -4 := by
sorry

end geometric_sequence_problem_l1471_147145


namespace complex_sum_powers_l1471_147172

theorem complex_sum_powers (x : ℂ) (h : x^2 + x + 1 = 0) :
  x^49 + x^50 + x^51 + x^52 + x^53 = -1 := by
  sorry

end complex_sum_powers_l1471_147172


namespace contrapositive_equivalence_l1471_147161

theorem contrapositive_equivalence :
  (∀ a : ℝ, a > 0 → a^2 > 0) ↔ (∀ a : ℝ, a^2 ≤ 0 → a ≤ 0) :=
by sorry

end contrapositive_equivalence_l1471_147161


namespace smallest_box_for_vase_l1471_147186

/-- Represents a cylindrical vase -/
structure Vase where
  height : ℝ
  baseDiameter : ℝ

/-- Represents a cube-shaped box -/
structure CubeBox where
  sideLength : ℝ

/-- The volume of a cube-shaped box -/
def boxVolume (box : CubeBox) : ℝ := box.sideLength ^ 3

/-- Predicate to check if a vase fits upright in a box -/
def fitsUpright (v : Vase) (b : CubeBox) : Prop :=
  v.height ≤ b.sideLength ∧ v.baseDiameter ≤ b.sideLength

theorem smallest_box_for_vase (v : Vase) (h1 : v.height = 15) (h2 : v.baseDiameter = 8) :
  ∃ (b : CubeBox), fitsUpright v b ∧
    (∀ (b' : CubeBox), fitsUpright v b' → boxVolume b ≤ boxVolume b') ∧
    boxVolume b = 3375 := by
  sorry

end smallest_box_for_vase_l1471_147186


namespace students_without_A_l1471_147151

theorem students_without_A (total : ℕ) (english_A : ℕ) (math_A : ℕ) (both_A : ℕ) : 
  total = 40 →
  english_A = 10 →
  math_A = 18 →
  both_A = 6 →
  total - (english_A + math_A - both_A) = 18 := by
sorry

end students_without_A_l1471_147151


namespace cistern_initial_water_fraction_l1471_147142

theorem cistern_initial_water_fraction 
  (pipe_a_fill_time : ℝ) 
  (pipe_b_fill_time : ℝ) 
  (combined_fill_time : ℝ) 
  (h1 : pipe_a_fill_time = 12) 
  (h2 : pipe_b_fill_time = 8) 
  (h3 : combined_fill_time = 14.4) : 
  ∃ x : ℝ, x = 2/3 ∧ 
    (1 / combined_fill_time = (1 - x) / pipe_a_fill_time + (1 - x) / pipe_b_fill_time) :=
by sorry

end cistern_initial_water_fraction_l1471_147142


namespace pudding_weight_l1471_147125

theorem pudding_weight (w : ℝ) 
  (h1 : 9/11 * w + 4 = w - (w - (9/11 * w + 4)))
  (h2 : 9/11 * w + 52 = w + (w - (9/11 * w + 4))) :
  w = 154 := by sorry

end pudding_weight_l1471_147125


namespace amandas_quiz_average_l1471_147138

theorem amandas_quiz_average :
  ∀ (num_quizzes : ℕ) (final_quiz_score : ℝ) (required_average : ℝ),
    num_quizzes = 4 →
    final_quiz_score = 97 →
    required_average = 93 →
    ∃ (current_average : ℝ),
      current_average = 92 ∧
      (num_quizzes : ℝ) * current_average + final_quiz_score = (num_quizzes + 1 : ℝ) * required_average :=
by
  sorry

end amandas_quiz_average_l1471_147138


namespace helga_usual_work_hours_l1471_147100

/-- Helga's work schedule and article writing capacity -/
structure HelgaWork where
  articles_per_30min : ℕ
  days_per_week : ℕ
  extra_hours_thursday : ℕ
  extra_hours_friday : ℕ
  total_articles_this_week : ℕ

/-- Calculate Helga's usual daily work hours -/
def usual_daily_hours (hw : HelgaWork) : ℚ :=
  let articles_per_hour : ℚ := (hw.articles_per_30min : ℚ) * 2
  let total_hours_this_week : ℚ := (hw.total_articles_this_week : ℚ) / articles_per_hour
  let usual_hours_this_week : ℚ := total_hours_this_week - (hw.extra_hours_thursday + hw.extra_hours_friday)
  usual_hours_this_week / (hw.days_per_week : ℚ)

/-- Theorem: Helga usually works 4 hours each day -/
theorem helga_usual_work_hours (hw : HelgaWork)
  (h1 : hw.articles_per_30min = 5)
  (h2 : hw.days_per_week = 5)
  (h3 : hw.extra_hours_thursday = 2)
  (h4 : hw.extra_hours_friday = 3)
  (h5 : hw.total_articles_this_week = 250) :
  usual_daily_hours hw = 4 := by
  sorry

end helga_usual_work_hours_l1471_147100


namespace single_elimination_tournament_games_l1471_147179

/-- The number of games in a single-elimination tournament -/
def num_games (n : ℕ) : ℕ := n - 1

/-- Theorem: In a single-elimination tournament with 17 teams and no ties, 
    the number of games played is 16 -/
theorem single_elimination_tournament_games : 
  num_games 17 = 16 := by sorry

end single_elimination_tournament_games_l1471_147179


namespace cleaner_solution_calculation_l1471_147177

/-- Represents the amount of cleaner solution needed for each type of stain -/
structure StainCleaner where
  dog : Nat
  cat : Nat
  bird : Nat
  rabbit : Nat
  fish : Nat

/-- Represents the number of stains for each type -/
structure StainCount where
  dog : Nat
  cat : Nat
  bird : Nat
  rabbit : Nat
  fish : Nat

def cleaner : StainCleaner :=
  { dog := 6, cat := 4, bird := 3, rabbit := 1, fish := 2 }

def weeklyStains : StainCount :=
  { dog := 10, cat := 8, bird := 5, rabbit := 1, fish := 3 }

def bottleSize : Nat := 64

/-- Calculates the total amount of cleaner solution needed -/
def totalSolutionNeeded (c : StainCleaner) (s : StainCount) : Nat :=
  c.dog * s.dog + c.cat * s.cat + c.bird * s.bird + c.rabbit * s.rabbit + c.fish * s.fish

/-- Calculates the additional amount of cleaner solution needed -/
def additionalSolutionNeeded (total : Nat) (bottleSize : Nat) : Nat :=
  if total > bottleSize then total - bottleSize else 0

theorem cleaner_solution_calculation :
  totalSolutionNeeded cleaner weeklyStains = 114 ∧
  additionalSolutionNeeded (totalSolutionNeeded cleaner weeklyStains) bottleSize = 50 := by
  sorry

end cleaner_solution_calculation_l1471_147177


namespace square_circle_relation_l1471_147154

theorem square_circle_relation (s r : ℝ) (h : s > 0) :
  4 * s = π * r^2 → r = 2 * Real.sqrt 2 / π := by
  sorry

end square_circle_relation_l1471_147154


namespace smallest_fraction_l1471_147163

theorem smallest_fraction (a x y : ℕ) (ha : a > 100) (hx : x > 100) (hy : y > 100)
  (h : y^2 - 1 = a^2 * (x^2 - 1)) : a / x ≥ 2 := by
  sorry

end smallest_fraction_l1471_147163


namespace expression_evaluations_l1471_147101

theorem expression_evaluations :
  -- Part 1
  (25 ^ (1/3) - 125 ^ (1/2)) / (5 ^ (1/4)) = 5 ^ (5/12) - 5 * (5 ^ (1/4)) ∧
  -- Part 2
  ∀ a : ℝ, a > 0 → a^2 / (a^(1/2) * a^(2/3)) = a^(5/6) := by
  sorry

end expression_evaluations_l1471_147101


namespace johns_recycling_money_l1471_147109

/-- The weight of a Monday-Saturday newspaper in ounces -/
def weekdayPaperWeight : ℕ := 8

/-- The weight of a Sunday newspaper in ounces -/
def sundayPaperWeight : ℕ := 2 * weekdayPaperWeight

/-- The number of papers John is supposed to deliver daily -/
def dailyPapers : ℕ := 250

/-- The number of weeks John steals the papers -/
def stolenWeeks : ℕ := 10

/-- The recycling value of one ton of paper in dollars -/
def recyclingValuePerTon : ℕ := 20

/-- The number of ounces in a ton -/
def ouncesPerTon : ℕ := 32000

/-- Calculate the total money John makes from recycling stolen newspapers -/
def johnsMoney : ℚ :=
  let totalWeekdayWeight := 6 * stolenWeeks * dailyPapers * weekdayPaperWeight
  let totalSundayWeight := stolenWeeks * dailyPapers * sundayPaperWeight
  let totalWeight := totalWeekdayWeight + totalSundayWeight
  let weightInTons := totalWeight / ouncesPerTon
  weightInTons * recyclingValuePerTon

/-- Theorem stating that John makes $100 from recycling the stolen newspapers -/
theorem johns_recycling_money : johnsMoney = 100 := by
  sorry

end johns_recycling_money_l1471_147109


namespace sector_radius_l1471_147116

theorem sector_radius (A : ℝ) (θ : ℝ) (r : ℝ) : 
  A = 6 * Real.pi → θ = (4 * Real.pi) / 3 → A = (1/2) * r^2 * θ → r = 3 := by
  sorry

end sector_radius_l1471_147116


namespace no_solution_implies_m_equals_two_l1471_147114

theorem no_solution_implies_m_equals_two :
  (∀ x : ℝ, (2 - m) / (1 - x) ≠ 1) → m = 2 := by
  sorry

end no_solution_implies_m_equals_two_l1471_147114


namespace prob_998th_toss_heads_l1471_147166

/-- A fair coin is a coin where the probability of getting heads is 1/2. -/
def fair_coin (coin : Type) : Prop :=
  ∃ (p : coin → ℝ), (∀ c, p c = 1/2) ∧ (∀ c, 0 ≤ p c ∧ p c ≤ 1)

/-- An independent event is an event whose probability is not affected by other events. -/
def independent_event (event : Type) (p : event → ℝ) : Prop :=
  ∀ (e₁ e₂ : event), p e₁ = p e₂

/-- The probability of getting heads on the 998th toss of a fair coin in a sequence of 1000 tosses. -/
theorem prob_998th_toss_heads (coin : Type) (toss : ℕ → coin) :
  fair_coin coin →
  independent_event coin (λ c => 1/2) →
  (λ c => 1/2) (toss 998) = 1/2 := by
  sorry

end prob_998th_toss_heads_l1471_147166


namespace pascal_triangle_47_l1471_147129

/-- Pascal's Triangle contains the number 47 in exactly one row -/
theorem pascal_triangle_47 (p : ℕ) (h_prime : Nat.Prime p) (h_p : p = 47) : 
  (∃! n : ℕ, ∃ k : ℕ, Nat.choose n k = p) :=
sorry

end pascal_triangle_47_l1471_147129


namespace point_c_coordinates_l1471_147137

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- The area of a triangle given three points -/
def triangleArea (a b c : Point2D) : ℝ := sorry

/-- Theorem: Given the conditions, point C has coordinates (0,4) or (0,-4) -/
theorem point_c_coordinates :
  let a : Point2D := ⟨-2, 0⟩
  let b : Point2D := ⟨3, 0⟩
  ∀ c : Point2D,
    c.x = 0 →  -- C lies on the y-axis
    triangleArea a b c = 10 →
    (c.y = 4 ∨ c.y = -4) :=
by sorry

end point_c_coordinates_l1471_147137
