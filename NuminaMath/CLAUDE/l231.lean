import Mathlib

namespace NUMINAMATH_CALUDE_incorrect_calculation_l231_23179

theorem incorrect_calculation (x : ℝ) : 
  25 * ((1/25) * x^2 - (1/10) * x + 1) ≠ x^2 - (5/2) * x + 25 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_calculation_l231_23179


namespace NUMINAMATH_CALUDE_max_weight_proof_l231_23104

/-- The maximum number of crates the trailer can carry on a single trip -/
def max_crates : ℕ := 5

/-- The minimum weight of each crate in kilograms -/
def min_crate_weight : ℕ := 150

/-- The maximum weight of crates on a single trip in kilograms -/
def max_total_weight : ℕ := max_crates * min_crate_weight

theorem max_weight_proof :
  max_total_weight = 750 := by
  sorry

end NUMINAMATH_CALUDE_max_weight_proof_l231_23104


namespace NUMINAMATH_CALUDE_soccer_league_games_l231_23115

/-- The number of teams in the soccer league -/
def num_teams : ℕ := 10

/-- The number of games each team plays with every other team -/
def games_per_pair : ℕ := 2

/-- The total number of games played in the season -/
def total_games : ℕ := num_teams * (num_teams - 1) * games_per_pair / 2

theorem soccer_league_games :
  total_games = 90 :=
sorry

end NUMINAMATH_CALUDE_soccer_league_games_l231_23115


namespace NUMINAMATH_CALUDE_prime_roots_integer_l231_23112

theorem prime_roots_integer (p : ℕ) : 
  Prime p ∧ 
  (∃ x y : ℤ, x ≠ y ∧ 
    x^2 + 2*p*x - 240*p = 0 ∧ 
    y^2 + 2*p*y - 240*p = 0) ↔ 
  p = 5 := by
sorry

end NUMINAMATH_CALUDE_prime_roots_integer_l231_23112


namespace NUMINAMATH_CALUDE_three_eighths_count_l231_23134

theorem three_eighths_count : (8 + 5/3 - 3) / (3/8) = 160/9 := by sorry

end NUMINAMATH_CALUDE_three_eighths_count_l231_23134


namespace NUMINAMATH_CALUDE_roots_sum_cubes_fourth_powers_l231_23155

theorem roots_sum_cubes_fourth_powers (α β : ℝ) : 
  α^2 - 3*α - 2 = 0 → β^2 - 3*β - 2 = 0 → 3*α^3 + 8*β^4 = 1229 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_cubes_fourth_powers_l231_23155


namespace NUMINAMATH_CALUDE_flight_passengers_l231_23106

theorem flight_passengers :
  ∀ (total_passengers : ℕ),
    (total_passengers : ℝ) * 0.4 = total_passengers * 0.4 →
    (total_passengers : ℝ) * 0.1 = total_passengers * 0.1 →
    (total_passengers : ℝ) * 0.9 = total_passengers - total_passengers * 0.1 →
    (total_passengers * 0.1 : ℝ) * (2/3) = total_passengers * 0.1 * (2/3) →
    (total_passengers : ℝ) * 0.4 - total_passengers * 0.1 * (2/3) = 40 →
    total_passengers = 120 :=
by sorry

end NUMINAMATH_CALUDE_flight_passengers_l231_23106


namespace NUMINAMATH_CALUDE_female_students_count_l231_23131

/-- Represents a school with male and female students. -/
structure School where
  total_students : ℕ
  sample_size : ℕ
  sample_boys_girls_diff : ℕ

/-- Calculates the number of female students in the school based on the given parameters. -/
def female_students (s : School) : ℕ :=
  let sampled_girls := (s.sample_size - s.sample_boys_girls_diff) / 2
  let ratio := s.total_students / s.sample_size
  sampled_girls * ratio

/-- Theorem stating that for the given school parameters, the number of female students is 760. -/
theorem female_students_count (s : School) 
  (h1 : s.total_students = 1600)
  (h2 : s.sample_size = 200)
  (h3 : s.sample_boys_girls_diff = 10) :
  female_students s = 760 := by
  sorry

end NUMINAMATH_CALUDE_female_students_count_l231_23131


namespace NUMINAMATH_CALUDE_polynomial_coefficients_l231_23197

theorem polynomial_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, x^5 = a₀ + a₁*(1-x) + a₂*(1-x)^2 + a₃*(1-x)^3 + a₄*(1-x)^4 + a₅*(1-x)^5) →
  (a₃ = -10 ∧ a₁ + a₃ + a₅ = -16) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_coefficients_l231_23197


namespace NUMINAMATH_CALUDE_company_size_proof_l231_23118

/-- The total number of employees in the company -/
def total_employees : ℕ := 100

/-- The number of employees in group C -/
def group_C_employees : ℕ := 10

/-- The ratio of employees in levels A:B:C -/
def employee_ratio : Fin 3 → ℕ
| 0 => 5  -- Level A
| 1 => 4  -- Level B
| 2 => 1  -- Level C

/-- The size of the stratified sample -/
def sample_size : ℕ := 20

/-- The probability of selecting both people from group C in the sample -/
def prob_both_from_C : ℚ := 1 / 45

theorem company_size_proof :
  (total_employees = 100) ∧
  (group_C_employees = 10) ∧
  (∀ i : Fin 3, employee_ratio i = [5, 4, 1].get i) ∧
  (sample_size = 20) ∧
  (prob_both_from_C = 1 / 45) ∧
  (group_C_employees.choose 2 = prob_both_from_C * total_employees.choose 2) ∧
  (group_C_employees * (employee_ratio 0 + employee_ratio 1 + employee_ratio 2) = total_employees) :=
by sorry

#check company_size_proof

end NUMINAMATH_CALUDE_company_size_proof_l231_23118


namespace NUMINAMATH_CALUDE_triangle_height_problem_l231_23163

theorem triangle_height_problem (base1 height1 base2 : ℝ) 
  (h_base1 : base1 = 15)
  (h_height1 : height1 = 12)
  (h_base2 : base2 = 20)
  (h_area_relation : base2 * (base1 * height1) = 2 * base1 * (base2 * height1)) :
  ∃ height2 : ℝ, height2 = 18 ∧ base2 * height2 = 2 * (base1 * height1) := by
sorry

end NUMINAMATH_CALUDE_triangle_height_problem_l231_23163


namespace NUMINAMATH_CALUDE_circumscribed_circle_area_relation_l231_23182

/-- Given a right triangle with sides 15, 36, and 39, and a circumscribed circle,
    where an altitude from the right angle divides one non-triangular region into
    areas A and B, and C is the largest non-triangular region, prove that A + B + 270 = C -/
theorem circumscribed_circle_area_relation (A B C : ℝ) : 
  A > 0 → B > 0 → C > 0 →
  (15 : ℝ) ^ 2 + 36 ^ 2 = 39 ^ 2 →
  A < B →
  B < C →
  A + B + 270 = C := by
  sorry

end NUMINAMATH_CALUDE_circumscribed_circle_area_relation_l231_23182


namespace NUMINAMATH_CALUDE_puppies_given_away_l231_23117

theorem puppies_given_away (initial_puppies : ℝ) (current_puppies : ℕ) : 
  initial_puppies = 6.0 →
  current_puppies = 4 →
  initial_puppies - current_puppies = 2 := by
sorry

end NUMINAMATH_CALUDE_puppies_given_away_l231_23117


namespace NUMINAMATH_CALUDE_speed_conversion_l231_23120

theorem speed_conversion (speed_kmh : ℝ) (speed_ms : ℝ) : 
  speed_kmh = 1.2 → speed_ms = 1/3 → speed_kmh * (1000 / 3600) = speed_ms :=
by
  sorry

end NUMINAMATH_CALUDE_speed_conversion_l231_23120


namespace NUMINAMATH_CALUDE_abs_le_two_necessary_not_sufficient_for_zero_le_x_le_two_l231_23172

theorem abs_le_two_necessary_not_sufficient_for_zero_le_x_le_two :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → |x| ≤ 2) ∧
  ¬(∀ x : ℝ, |x| ≤ 2 → 0 ≤ x ∧ x ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_abs_le_two_necessary_not_sufficient_for_zero_le_x_le_two_l231_23172


namespace NUMINAMATH_CALUDE_five_dozen_apples_cost_l231_23183

/-- The cost of a given number of dozens of apples, given the cost of two dozens. -/
def apple_cost (two_dozen_cost : ℚ) (dozens : ℚ) : ℚ :=
  (dozens / 2) * two_dozen_cost

/-- Theorem: If two dozen apples cost $15.60, then five dozen apples cost $39.00 -/
theorem five_dozen_apples_cost (two_dozen_cost : ℚ) 
  (h : two_dozen_cost = 15.6) : 
  apple_cost two_dozen_cost 5 = 39 := by
  sorry

end NUMINAMATH_CALUDE_five_dozen_apples_cost_l231_23183


namespace NUMINAMATH_CALUDE_square_area_after_cut_l231_23124

theorem square_area_after_cut (side : ℝ) (h1 : side > 0) : 
  side * (side - 3) = 40 → side * side = 64 := by sorry

end NUMINAMATH_CALUDE_square_area_after_cut_l231_23124


namespace NUMINAMATH_CALUDE_fraction_equality_l231_23164

theorem fraction_equality (p q : ℝ) (h : p / q - q / p = 21 / 10) :
  4 * p / q + 4 * q / p = 16.8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l231_23164


namespace NUMINAMATH_CALUDE_chinese_english_total_score_l231_23171

theorem chinese_english_total_score 
  (average_score : ℝ) 
  (math_score : ℝ) 
  (num_subjects : ℕ) 
  (h1 : average_score = 97) 
  (h2 : math_score = 100) 
  (h3 : num_subjects = 3) :
  average_score * num_subjects - math_score = 191 :=
by sorry

end NUMINAMATH_CALUDE_chinese_english_total_score_l231_23171


namespace NUMINAMATH_CALUDE_smallest_sum_of_primes_with_digit_conditions_l231_23123

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def digit_count (n : ℕ) (d : ℕ) : ℕ := 
  (n.digits 10).count d

def satisfies_conditions (primes : List ℕ) : Prop :=
  primes.length = 5 ∧
  (∀ p ∈ primes, is_prime p) ∧
  (primes.map (digit_count · 3)).sum = 2 ∧
  (primes.map (digit_count · 7)).sum = 2 ∧
  (primes.map (digit_count · 8)).sum = 2 ∧
  (∀ d ∈ [1, 2, 4, 5, 6, 9], (primes.map (digit_count · d)).sum = 1)

theorem smallest_sum_of_primes_with_digit_conditions :
  ∃ (primes : List ℕ),
    satisfies_conditions primes ∧
    primes.sum = 2063 ∧
    (∀ other_primes : List ℕ, satisfies_conditions other_primes → other_primes.sum ≥ 2063) :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_primes_with_digit_conditions_l231_23123


namespace NUMINAMATH_CALUDE_derivative_f_at_zero_l231_23166

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then 3^(x^2 * Real.sin (2/x)) - 1 + 2*x else 0

theorem derivative_f_at_zero : 
  deriv f 0 = -2 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_zero_l231_23166


namespace NUMINAMATH_CALUDE_roots_of_quadratic_equation_l231_23176

theorem roots_of_quadratic_equation : 
  ∀ x : ℝ, x^2 = 2*x ↔ x = 0 ∨ x = 2 := by sorry

end NUMINAMATH_CALUDE_roots_of_quadratic_equation_l231_23176


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l231_23110

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1
  sum_1_5 : a 1 + a 5 = -20
  sum_3_8 : a 3 + a 8 = -10

/-- The general term of the sequence -/
def general_term (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  2 * n - 16

/-- The sum of the first n terms of the sequence -/
def sum_n_terms (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  n * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n : ℕ, seq.a n = general_term seq n) ∧
  (∃ n : ℕ, sum_n_terms seq n = -56 ∧ (n = 7 ∨ n = 8)) ∧
  (∀ m : ℕ, sum_n_terms seq m ≥ -56) :=
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l231_23110


namespace NUMINAMATH_CALUDE_z_in_first_quadrant_l231_23151

/-- The complex number i -/
def i : ℂ := Complex.I

/-- The given equation (1-2i)z = 5 -/
def given_equation (z : ℂ) : Prop := (1 - 2*i) * z = 5

/-- A complex number is in the first quadrant if its real and imaginary parts are both positive -/
def in_first_quadrant (z : ℂ) : Prop := 0 < z.re ∧ 0 < z.im

/-- 
If (1-2i)z = 5, then z is in the first quadrant of the complex plane
-/
theorem z_in_first_quadrant (z : ℂ) (h : given_equation z) : in_first_quadrant z :=
sorry

end NUMINAMATH_CALUDE_z_in_first_quadrant_l231_23151


namespace NUMINAMATH_CALUDE_water_donation_difference_l231_23121

/-- The number of food items donated by five food companies to a local food bank. -/
def food_bank_donation : ℕ := 375

/-- The number of dressed chickens donated by Foster Farms. -/
def foster_farms_chickens : ℕ := 45

/-- The number of bottles of water donated by American Summits. -/
def american_summits_water : ℕ := 2 * foster_farms_chickens

/-- The number of dressed chickens donated by Hormel. -/
def hormel_chickens : ℕ := 3 * foster_farms_chickens

/-- The number of dressed chickens donated by Boudin Butchers. -/
def boudin_butchers_chickens : ℕ := hormel_chickens / 3

/-- The number of bottles of water donated by Del Monte Foods. -/
def del_monte_water : ℕ := food_bank_donation - (foster_farms_chickens + american_summits_water + hormel_chickens + boudin_butchers_chickens)

/-- Theorem stating the difference in water bottles donated between American Summits and Del Monte Foods. -/
theorem water_donation_difference :
  american_summits_water - del_monte_water = 30 := by
  sorry

end NUMINAMATH_CALUDE_water_donation_difference_l231_23121


namespace NUMINAMATH_CALUDE_grocery_solution_l231_23100

/-- Represents the grocery shopping problem --/
def grocery_problem (initial_money : ℝ) (mustard_oil_price : ℝ) (mustard_oil_quantity : ℝ)
  (pasta_price : ℝ) (pasta_quantity : ℝ) (sauce_price : ℝ) (money_left : ℝ) : Prop :=
  let total_spent := initial_money - money_left
  let mustard_oil_cost := mustard_oil_price * mustard_oil_quantity
  let pasta_cost := pasta_price * pasta_quantity
  let sauce_cost := total_spent - mustard_oil_cost - pasta_cost
  sauce_cost / sauce_price = 1

/-- Theorem stating the solution to the grocery problem --/
theorem grocery_solution :
  grocery_problem 50 13 2 4 3 5 7 := by
  sorry

#check grocery_solution

end NUMINAMATH_CALUDE_grocery_solution_l231_23100


namespace NUMINAMATH_CALUDE_max_sum_property_terms_l231_23139

/-- A quadratic polynomial -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Evaluation of a quadratic polynomial at a given point -/
def evaluate (P : QuadraticPolynomial) (x : ℝ) : ℝ :=
  P.a * x^2 + P.b * x + P.c

/-- The property that P(n+1) = P(n) + P(n-1) -/
def hasSumProperty (P : QuadraticPolynomial) (n : ℕ) : Prop :=
  evaluate P (n + 1 : ℝ) = evaluate P n + evaluate P (n - 1 : ℝ)

/-- The main theorem: maximum number of terms with sum property is 2 -/
theorem max_sum_property_terms (P : QuadraticPolynomial) :
  (∃ (S : Finset ℕ), (∀ n ∈ S, n ≥ 2 ∧ hasSumProperty P n) ∧ S.card > 2) → False :=
sorry

end NUMINAMATH_CALUDE_max_sum_property_terms_l231_23139


namespace NUMINAMATH_CALUDE_intersecting_lines_theorem_l231_23157

/-- Given two lines that intersect at (-7, 9), prove that the line passing through their coefficients as points has the equation -7x + 9y = 1 -/
theorem intersecting_lines_theorem (A₁ B₁ A₂ B₂ : ℝ) : 
  (A₁ * (-7) + B₁ * 9 = 1) →  -- First line passes through (-7, 9)
  (A₂ * (-7) + B₂ * 9 = 1) →  -- Second line passes through (-7, 9)
  ∃ (k : ℝ), k * (-7) * (A₂ - A₁) = 9 * (B₂ - B₁) ∧   -- Points (A₁, B₁) and (A₂, B₂) satisfy -7x + 9y = k
             k = 1 :=
by sorry

end NUMINAMATH_CALUDE_intersecting_lines_theorem_l231_23157


namespace NUMINAMATH_CALUDE_lcm_gcd_product_l231_23109

theorem lcm_gcd_product (a b : ℕ) (ha : a = 12) (hb : b = 9) :
  Nat.lcm a b * Nat.gcd a b = 108 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_l231_23109


namespace NUMINAMATH_CALUDE_parabola_x_intercepts_l231_23145

theorem parabola_x_intercepts :
  ∃! x : ℝ, ∃ y : ℝ, x = -3 * y^2 + 2 * y + 3 ∧ y = 0 :=
sorry

end NUMINAMATH_CALUDE_parabola_x_intercepts_l231_23145


namespace NUMINAMATH_CALUDE_parallelepiped_covering_l231_23154

/-- A parallelepiped constructed from 4 identical unit cubes stacked vertically -/
structure Parallelepiped :=
  (height : ℕ)
  (width : ℕ)
  (depth : ℕ)
  (is_valid : height = 4 ∧ width = 1 ∧ depth = 1)

/-- A square with side length n -/
structure Square (n : ℕ) :=
  (side_length : ℕ)
  (is_valid : side_length = n)

/-- The covering of the parallelepiped -/
structure Covering (p : Parallelepiped) :=
  (vertical_square : Square 4)
  (top_square : Square 1)
  (bottom_square : Square 1)

/-- The theorem stating that the parallelepiped can be covered by three squares -/
theorem parallelepiped_covering (p : Parallelepiped) :
  ∃ (c : Covering p),
    (c.vertical_square.side_length ^ 2 = 2 * p.height * p.width + 2 * p.height * p.depth) ∧
    (c.top_square.side_length ^ 2 = p.width * p.depth) ∧
    (c.bottom_square.side_length ^ 2 = p.width * p.depth) :=
  sorry

end NUMINAMATH_CALUDE_parallelepiped_covering_l231_23154


namespace NUMINAMATH_CALUDE_samantha_birth_year_l231_23138

-- Define the year of the first AMC 8
def first_amc8_year : ℕ := 1985

-- Define the frequency of AMC 8 (every 2 years)
def amc8_frequency : ℕ := 2

-- Define Samantha's age when she took the fourth AMC 8
def samantha_age_fourth_amc8 : ℕ := 12

-- Function to calculate the year of the nth AMC 8
def nth_amc8_year (n : ℕ) : ℕ :=
  first_amc8_year + (n - 1) * amc8_frequency

-- Theorem to prove Samantha's birth year
theorem samantha_birth_year :
  nth_amc8_year 4 - samantha_age_fourth_amc8 = 1981 :=
by sorry

end NUMINAMATH_CALUDE_samantha_birth_year_l231_23138


namespace NUMINAMATH_CALUDE_triangle_property_l231_23125

theorem triangle_property (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  (a - c) / (a - b) = (Real.sin A + Real.sin B) / Real.sin (A + B) ∧
  2 = b / Real.sin (π / 3) →
  B = π / 3 ∧
  ∀ (S : ℝ), S = (1 / 2) * a * c * Real.sin (π / 3) → S ≤ 3 * Real.sqrt 3 / 4 :=
by sorry


end NUMINAMATH_CALUDE_triangle_property_l231_23125


namespace NUMINAMATH_CALUDE_infinite_solutions_l231_23173

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := 3 * x - 4 * y = 10
def equation2 (x y : ℝ) : Prop := 6 * x - 8 * y = 20

-- Theorem stating that the system has infinitely many solutions
theorem infinite_solutions :
  ∃ (f : ℝ → ℝ × ℝ), ∀ t : ℝ,
    let (x, y) := f t
    equation1 x y ∧ equation2 x y ∧
    (∀ s : ℝ, s ≠ t → f s ≠ f t) :=
sorry

end NUMINAMATH_CALUDE_infinite_solutions_l231_23173


namespace NUMINAMATH_CALUDE_condo_cats_l231_23192

theorem condo_cats (total_families : ℕ) 
  (one_cat_families : ℕ) (three_cat_families : ℕ) (five_cat_families : ℕ) : 
  total_families = 29 →
  total_families = one_cat_families + three_cat_families + five_cat_families →
  one_cat_families = five_cat_families →
  one_cat_families * 1 + three_cat_families * 3 + five_cat_families * 5 = 87 := by
sorry

end NUMINAMATH_CALUDE_condo_cats_l231_23192


namespace NUMINAMATH_CALUDE_journey_matches_graph_characteristics_l231_23149

/-- Represents a point on the speed-time graph -/
structure SpeedTimePoint where
  time : ℝ
  speed : ℝ

/-- Represents a section of the speed-time graph -/
inductive GraphSection
  | Increasing : GraphSection
  | Flat : GraphSection
  | Decreasing : GraphSection

/-- Represents Mike's journey -/
structure Journey where
  cityTraffic : Bool
  highway : Bool
  workplace : Bool
  coffeeBreak : Bool
  workDuration : ℝ
  breakDuration : ℝ

/-- Defines the characteristics of the correct graph -/
def correctGraphCharacteristics : List GraphSection :=
  [GraphSection.Increasing, GraphSection.Flat, GraphSection.Increasing, 
   GraphSection.Flat, GraphSection.Decreasing]

/-- Theorem stating that Mike's journey matches the correct graph characteristics -/
theorem journey_matches_graph_characteristics (j : Journey) :
  j.cityTraffic = true →
  j.highway = true →
  j.workplace = true →
  j.coffeeBreak = true →
  j.workDuration = 2 →
  j.breakDuration = 0.5 →
  ∃ (graph : List GraphSection), graph = correctGraphCharacteristics := by
  sorry

#check journey_matches_graph_characteristics

end NUMINAMATH_CALUDE_journey_matches_graph_characteristics_l231_23149


namespace NUMINAMATH_CALUDE_subset_condition_1_subset_condition_2_l231_23114

-- Define the sets M, N1, and N2
def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def N1 (m : ℝ) : Set ℝ := {x | m - 6 ≤ x ∧ x ≤ 2*m - 1}
def N2 (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

-- Theorem for part 1
theorem subset_condition_1 :
  ∀ m : ℝ, (M ⊆ N1 m) ↔ (2 ≤ m ∧ m ≤ 3) :=
by sorry

-- Theorem for part 2
theorem subset_condition_2 :
  ∀ m : ℝ, (N2 m ⊆ M) ↔ (m ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_subset_condition_1_subset_condition_2_l231_23114


namespace NUMINAMATH_CALUDE_complex_modulus_l231_23191

theorem complex_modulus (z : ℂ) (h : (z - I) / (2 - I) = I) : Complex.abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l231_23191


namespace NUMINAMATH_CALUDE_gcd_2_exp_1020_minus_1_2_exp_1031_minus_1_l231_23168

theorem gcd_2_exp_1020_minus_1_2_exp_1031_minus_1 :
  Nat.gcd (2^1020 - 1) (2^1031 - 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_2_exp_1020_minus_1_2_exp_1031_minus_1_l231_23168


namespace NUMINAMATH_CALUDE_brown_dogs_count_l231_23105

/-- Proves the number of brown dogs in a kennel with specific conditions -/
theorem brown_dogs_count (total : ℕ) (long_fur : ℕ) (neither : ℕ) (long_fur_brown : ℕ)
  (h1 : total = 45)
  (h2 : long_fur = 26)
  (h3 : neither = 8)
  (h4 : long_fur_brown = 19) :
  total - long_fur + long_fur_brown = 30 := by
  sorry

#check brown_dogs_count

end NUMINAMATH_CALUDE_brown_dogs_count_l231_23105


namespace NUMINAMATH_CALUDE_b_minus_d_squared_l231_23144

theorem b_minus_d_squared (a b c d e : ℝ) 
  (eq1 : a - b - c + d = 13)
  (eq2 : a + b - c - d = 9)
  (eq3 : a - b + c + e = 11) : 
  (b - d)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_b_minus_d_squared_l231_23144


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l231_23161

theorem sufficient_not_necessary_condition (x : ℝ) :
  (∀ x, x > 0 → x * (x + 1) > 0) ∧
  (∃ x, x * (x + 1) > 0 ∧ ¬(x > 0)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l231_23161


namespace NUMINAMATH_CALUDE_steps_to_school_l231_23107

/-- The number of steps Raine takes walking to and from school in five days -/
def total_steps : ℕ := 1500

/-- The number of days Raine walks to and from school -/
def days : ℕ := 5

/-- Proves that the number of steps Raine takes to walk to school is 150 -/
theorem steps_to_school : (total_steps / (2 * days) : ℕ) = 150 := by
  sorry

end NUMINAMATH_CALUDE_steps_to_school_l231_23107


namespace NUMINAMATH_CALUDE_median_and_mean_of_set_l231_23150

theorem median_and_mean_of_set (m : ℝ) (h : m + 4 = 16) :
  let S : Finset ℝ := {m, m + 2, m + 4, m + 11, m + 18}
  (S.sum id) / S.card = 19 := by
sorry

end NUMINAMATH_CALUDE_median_and_mean_of_set_l231_23150


namespace NUMINAMATH_CALUDE_negation_equivalence_l231_23108

-- Define a triangle
structure Triangle where
  angles : Fin 3 → ℝ
  sum_180 : angles 0 + angles 1 + angles 2 = 180

-- Define the original proposition
def has_angle_le_60 (t : Triangle) : Prop :=
  ∃ i : Fin 3, t.angles i ≤ 60

-- Define the negation (assumption for proof by contradiction)
def all_angles_gt_60 (t : Triangle) : Prop :=
  ∀ i : Fin 3, t.angles i > 60

-- The theorem to prove
theorem negation_equivalence :
  ∀ t : Triangle, ¬(has_angle_le_60 t) ↔ all_angles_gt_60 t :=
sorry

end NUMINAMATH_CALUDE_negation_equivalence_l231_23108


namespace NUMINAMATH_CALUDE_vector_same_direction_l231_23162

open Real

/-- Given two vectors a and b in ℝ², prove that if they have the same direction,
    a = (1, -√3), and |b| = 1, then b = (1/2, -√3/2) -/
theorem vector_same_direction (a b : ℝ × ℝ) :
  (∃ k : ℝ, b = k • a) →  -- same direction
  a = (1, -Real.sqrt 3) →
  Real.sqrt ((b.1)^2 + (b.2)^2) = 1 →
  b = (1/2, -(Real.sqrt 3)/2) := by
sorry

end NUMINAMATH_CALUDE_vector_same_direction_l231_23162


namespace NUMINAMATH_CALUDE_price_difference_l231_23133

theorem price_difference (P : ℝ) (h : P > 0) : 
  let P' := 1.25 * P
  (P' - P) / P' * 100 = 20 := by sorry

end NUMINAMATH_CALUDE_price_difference_l231_23133


namespace NUMINAMATH_CALUDE_simplify_expression_l231_23140

theorem simplify_expression : 
  (Real.sqrt (Real.sqrt 81) - Real.sqrt (8 + 1/2))^2 = (70 - 12 * Real.sqrt 34) / 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l231_23140


namespace NUMINAMATH_CALUDE_positive_distinct_solution_condition_l231_23185

theorem positive_distinct_solution_condition 
  (a b x y z : ℝ) 
  (eq1 : x + y + z = a) 
  (eq2 : x^2 + y^2 + z^2 = b^2) 
  (eq3 : x * y = z^2) 
  (pos_x : x > 0) 
  (pos_y : y > 0) 
  (pos_z : z > 0) 
  (distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z) : 
  b^2 ≥ a^2 / 2 := by
sorry

end NUMINAMATH_CALUDE_positive_distinct_solution_condition_l231_23185


namespace NUMINAMATH_CALUDE_quadratic_vertex_form_l231_23169

theorem quadratic_vertex_form (a b c : ℝ) (h k : ℝ) :
  (∀ x, a * x^2 + b * x + c = a * (x - h)^2 + k) →
  (a = 3 ∧ b = 9 ∧ c = 20) →
  h = -1.5 := by sorry

end NUMINAMATH_CALUDE_quadratic_vertex_form_l231_23169


namespace NUMINAMATH_CALUDE_triangle_max_area_l231_23143

noncomputable def triangle_area (x : Real) : Real :=
  4 * Real.sqrt 3 * Real.sin x * Real.sin ((2 * Real.pi / 3) - x)

theorem triangle_max_area :
  ∀ x : Real, 0 < x → x < 2 * Real.pi / 3 →
    triangle_area x ≤ triangle_area (Real.pi / 3) ∧
    triangle_area (Real.pi / 3) = 3 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_area_l231_23143


namespace NUMINAMATH_CALUDE_investment_split_l231_23190

theorem investment_split (alice_share bob_share total : ℕ) : 
  alice_share = 5 →
  bob_share = 3 * (total / bob_share) →
  bob_share = 3 * alice_share + 3 →
  total = bob_share * (total / bob_share) + alice_share →
  total = 113 := by
sorry

end NUMINAMATH_CALUDE_investment_split_l231_23190


namespace NUMINAMATH_CALUDE_evaluate_expression_l231_23178

theorem evaluate_expression : 15 * 30 + 45 * 15 - 15 * 10 = 975 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l231_23178


namespace NUMINAMATH_CALUDE_lamp_probability_l231_23129

/-- The number of red lamps -/
def num_red_lamps : ℕ := 4

/-- The number of blue lamps -/
def num_blue_lamps : ℕ := 4

/-- The total number of lamps -/
def total_lamps : ℕ := num_red_lamps + num_blue_lamps

/-- The number of lamps turned on -/
def num_on_lamps : ℕ := 4

/-- The probability of the leftmost lamp being blue and on, and the rightmost lamp being red and off -/
theorem lamp_probability : 
  (num_red_lamps : ℚ) * (num_blue_lamps : ℚ) * (Nat.choose (total_lamps - 2) (num_on_lamps - 1)) / 
  ((Nat.choose total_lamps num_red_lamps) * (Nat.choose total_lamps num_on_lamps)) = 5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_lamp_probability_l231_23129


namespace NUMINAMATH_CALUDE_adjacency_probability_correct_l231_23132

/-- The probability of A being adjacent to both B and C in a random lineup of 4 people --/
def adjacency_probability : ℚ := 1 / 6

/-- The total number of people in the group --/
def total_people : ℕ := 4

/-- The number of ways to arrange ABC as a unit with the fourth person --/
def favorable_arrangements : ℕ := 4

/-- The total number of possible arrangements of 4 people --/
def total_arrangements : ℕ := 24

theorem adjacency_probability_correct :
  adjacency_probability = (favorable_arrangements : ℚ) / total_arrangements := by
  sorry

end NUMINAMATH_CALUDE_adjacency_probability_correct_l231_23132


namespace NUMINAMATH_CALUDE_max_NPMK_is_8010_l231_23180

/-- Represents a three-digit number MMK where M and K are digits and M = K + 1 -/
def MMK (M K : ℕ) : Prop :=
  M ≥ 1 ∧ M ≤ 9 ∧ K ≥ 0 ∧ K ≤ 8 ∧ M = K + 1

/-- Represents the result of multiplying MMK by M -/
def NPMK (M K : ℕ) : ℕ := (100 * M + 10 * M + K) * M

/-- The theorem stating that the maximum value of NPMK is 8010 -/
theorem max_NPMK_is_8010 :
  ∀ M K : ℕ, MMK M K → NPMK M K ≤ 8010 ∧ ∃ M K : ℕ, MMK M K ∧ NPMK M K = 8010 := by
  sorry

end NUMINAMATH_CALUDE_max_NPMK_is_8010_l231_23180


namespace NUMINAMATH_CALUDE_function_inequality_implies_parameter_bound_l231_23153

open Real

theorem function_inequality_implies_parameter_bound (a : ℝ) :
  (∃ x : ℝ, x > 1 ∧ -a * x^2 + log x > -a) → a < 1/2 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_implies_parameter_bound_l231_23153


namespace NUMINAMATH_CALUDE_room_calculation_correct_l231_23141

/-- Given a total number of paintings and paintings per room, calculates the number of rooms -/
def calculate_rooms (total_paintings : ℕ) (paintings_per_room : ℕ) : ℕ :=
  total_paintings / paintings_per_room

theorem room_calculation_correct :
  let total_paintings : ℕ := 32
  let paintings_per_room : ℕ := 8
  calculate_rooms total_paintings paintings_per_room = 4 := by
sorry

end NUMINAMATH_CALUDE_room_calculation_correct_l231_23141


namespace NUMINAMATH_CALUDE_red_grapes_count_l231_23142

/-- Represents the composition of a fruit salad -/
structure FruitSalad where
  greenGrapes : ℕ
  redGrapes : ℕ
  raspberries : ℕ

/-- Checks if a fruit salad satisfies the given conditions -/
def isValidFruitSalad (fs : FruitSalad) : Prop :=
  fs.redGrapes = 3 * fs.greenGrapes + 7 ∧
  fs.raspberries = fs.greenGrapes - 5 ∧
  fs.greenGrapes + fs.redGrapes + fs.raspberries = 102

/-- Theorem: The number of red grapes in the fruit salad is 67 -/
theorem red_grapes_count (fs : FruitSalad) (h : isValidFruitSalad fs) : fs.redGrapes = 67 := by
  sorry

#check red_grapes_count

end NUMINAMATH_CALUDE_red_grapes_count_l231_23142


namespace NUMINAMATH_CALUDE_optimal_strategy_is_down_l231_23152

/-- Represents the direction of movement on the escalator -/
inductive Direction
  | Up
  | Down

/-- Represents the state of Petya and his hat on the escalators -/
structure EscalatorState where
  petyaPosition : ℝ  -- Position of Petya (0 = bottom, 1 = top)
  hatPosition : ℝ    -- Position of the hat (0 = bottom, 1 = top)
  petyaSpeed : ℝ     -- Petya's movement speed
  escalatorSpeed : ℝ  -- Speed of the escalator

/-- Calculates the time for Petya to reach his hat -/
def timeToReachHat (state : EscalatorState) (direction : Direction) : ℝ :=
  sorry

/-- Theorem stating that moving downwards is the optimal strategy -/
theorem optimal_strategy_is_down (state : EscalatorState) :
  state.petyaPosition = 0.5 →
  state.hatPosition = 1 →
  state.petyaSpeed > state.escalatorSpeed →
  state.petyaSpeed < 2 * state.escalatorSpeed →
  timeToReachHat state Direction.Down < timeToReachHat state Direction.Up :=
sorry

#check optimal_strategy_is_down

end NUMINAMATH_CALUDE_optimal_strategy_is_down_l231_23152


namespace NUMINAMATH_CALUDE_girls_tried_out_l231_23194

/-- The number of girls who tried out for the basketball team -/
def girls : ℕ := 39

/-- The number of boys who tried out for the basketball team -/
def boys : ℕ := 4

/-- The number of students who got called back -/
def called_back : ℕ := 26

/-- The number of students who didn't make the cut -/
def didnt_make_cut : ℕ := 17

/-- The total number of students who tried out -/
def total_students : ℕ := called_back + didnt_make_cut

theorem girls_tried_out : girls = total_students - boys := by
  sorry

end NUMINAMATH_CALUDE_girls_tried_out_l231_23194


namespace NUMINAMATH_CALUDE_product_five_cubed_sum_l231_23184

theorem product_five_cubed_sum (a b c : ℕ+) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a * b * c = 5^3 →
  (a : ℕ) + (b : ℕ) + (c : ℕ) = 31 := by sorry

end NUMINAMATH_CALUDE_product_five_cubed_sum_l231_23184


namespace NUMINAMATH_CALUDE_jenny_jellybeans_proof_l231_23127

/-- The original number of jellybeans in Jenny's jar -/
def original_jellybeans : ℝ := 85

/-- The fraction of jellybeans remaining after each day -/
def daily_remaining_fraction : ℝ := 0.7

/-- The number of days Jenny eats jellybeans -/
def days : ℕ := 3

/-- The number of jellybeans remaining after 'days' days -/
def remaining_jellybeans : ℝ := 29.16

/-- Theorem stating that the original number of jellybeans is correct -/
theorem jenny_jellybeans_proof :
  original_jellybeans * daily_remaining_fraction ^ days = remaining_jellybeans := by
  sorry

#eval original_jellybeans -- Should output 85

end NUMINAMATH_CALUDE_jenny_jellybeans_proof_l231_23127


namespace NUMINAMATH_CALUDE_truck_calculation_l231_23189

/-- The number of trucks initially requested to transport 60 tons of cargo,
    where reducing each truck's capacity by 0.5 tons required 4 additional trucks. -/
def initial_trucks : ℕ := 20

/-- The total cargo to be transported in tons. -/
def total_cargo : ℝ := 60

/-- The reduction in capacity per truck in tons. -/
def capacity_reduction : ℝ := 0.5

/-- The number of additional trucks required after capacity reduction. -/
def additional_trucks : ℕ := 4

theorem truck_calculation :
  initial_trucks * (total_cargo / initial_trucks - capacity_reduction) = 
  (initial_trucks + additional_trucks) * ((total_cargo / initial_trucks) - capacity_reduction) ∧
  (initial_trucks + additional_trucks) * ((total_cargo / initial_trucks) - capacity_reduction) = total_cargo :=
sorry

end NUMINAMATH_CALUDE_truck_calculation_l231_23189


namespace NUMINAMATH_CALUDE_prob_combined_event_l231_23160

/-- The number of sides on each die -/
def sides : ℕ := 8

/-- The probability of rolling a number less than four on an eight-sided die -/
def prob_less_than_four : ℚ := 3 / 8

/-- The probability of rolling a number greater than four on an eight-sided die -/
def prob_greater_than_four : ℚ := 1 / 2

/-- The theorem stating the probability of the combined event -/
theorem prob_combined_event : 
  prob_less_than_four * prob_greater_than_four = 3 / 16 := by
  sorry

end NUMINAMATH_CALUDE_prob_combined_event_l231_23160


namespace NUMINAMATH_CALUDE_chess_game_most_likely_outcome_l231_23187

theorem chess_game_most_likely_outcome
  (prob_A_win : ℝ)
  (prob_A_not_lose : ℝ)
  (h1 : prob_A_win = 0.3)
  (h2 : prob_A_not_lose = 0.7)
  (h3 : 0 ≤ prob_A_win ∧ prob_A_win ≤ 1)
  (h4 : 0 ≤ prob_A_not_lose ∧ prob_A_not_lose ≤ 1) :
  let prob_draw := prob_A_not_lose - prob_A_win
  let prob_B_win := 1 - prob_A_not_lose
  prob_draw > prob_A_win ∧ prob_draw > prob_B_win :=
by sorry

end NUMINAMATH_CALUDE_chess_game_most_likely_outcome_l231_23187


namespace NUMINAMATH_CALUDE_f_bounds_l231_23119

def a : ℤ := 2001

def A : Set (ℤ × ℤ) :=
  {p | p.2 ≠ 0 ∧ 
       p.1 < 2 * a ∧ 
       (2 * p.2) ∣ (2 * a * p.1 - p.1^2 + p.2^2) ∧ 
       p.2^2 - p.1^2 + 2 * p.1 * p.2 ≤ 2 * a * (p.2 - p.1)}

def f (p : ℤ × ℤ) : ℚ :=
  (2 * a * p.1 - p.1^2 - p.1 * p.2) / p.2

theorem f_bounds :
  ∃ (min max : ℚ), min = 2 ∧ max = 3750 ∧
  ∀ p ∈ A, min ≤ f p ∧ f p ≤ max :=
sorry

end NUMINAMATH_CALUDE_f_bounds_l231_23119


namespace NUMINAMATH_CALUDE_beavers_swimming_l231_23188

theorem beavers_swimming (initial_beavers : ℕ) (remaining_beavers : ℕ) : 
  initial_beavers = 2 → remaining_beavers = 1 → initial_beavers - remaining_beavers = 1 := by
  sorry

end NUMINAMATH_CALUDE_beavers_swimming_l231_23188


namespace NUMINAMATH_CALUDE_arithmetic_sequence_contains_powers_of_four_l231_23113

theorem arithmetic_sequence_contains_powers_of_four (k : ℕ) :
  ∃ n : ℕ, 3 + 9 * (n - 1) = 3 * 4^k := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_contains_powers_of_four_l231_23113


namespace NUMINAMATH_CALUDE_volleyball_lineup_count_l231_23136

def choose (n k : ℕ) : ℕ := Nat.choose n k

def volleyball_lineups (total_players triplets : ℕ) (max_triplets : ℕ) : ℕ :=
  let non_triplets := total_players - triplets - 1  -- Subtract 1 for the captain
  let case0 := choose non_triplets 5
  let case1 := triplets * choose non_triplets 4
  let case2 := choose triplets 2 * choose non_triplets 3
  case0 + case1 + case2

theorem volleyball_lineup_count :
  volleyball_lineups 15 4 2 = 1812 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_lineup_count_l231_23136


namespace NUMINAMATH_CALUDE_original_list_size_l231_23199

theorem original_list_size (n : ℕ) (m : ℚ) : 
  (m + 3) * (n + 1) = m * n + 20 →
  (m + 1) * (n + 2) = (m + 3) * (n + 1) + 2 →
  n = 7 := by
sorry

end NUMINAMATH_CALUDE_original_list_size_l231_23199


namespace NUMINAMATH_CALUDE_joe_haircut_time_l231_23122

/-- Represents the time taken for different types of haircuts and the number of each type performed --/
structure HaircutData where
  womenTime : ℕ  -- Time to cut a woman's hair
  menTime : ℕ    -- Time to cut a man's hair
  kidsTime : ℕ   -- Time to cut a kid's hair
  womenCount : ℕ -- Number of women's haircuts
  menCount : ℕ   -- Number of men's haircuts
  kidsCount : ℕ  -- Number of kids' haircuts

/-- Calculates the total time spent cutting hair --/
def totalHaircutTime (data : HaircutData) : ℕ :=
  data.womenTime * data.womenCount +
  data.menTime * data.menCount +
  data.kidsTime * data.kidsCount

/-- Theorem stating that Joe's total haircut time is 255 minutes --/
theorem joe_haircut_time :
  let data : HaircutData := {
    womenTime := 50,
    menTime := 15,
    kidsTime := 25,
    womenCount := 3,
    menCount := 2,
    kidsCount := 3
  }
  totalHaircutTime data = 255 := by
  sorry

end NUMINAMATH_CALUDE_joe_haircut_time_l231_23122


namespace NUMINAMATH_CALUDE_stock_percentage_return_l231_23156

def stock_yield : ℝ := 0.08
def market_value : ℝ := 137.5

theorem stock_percentage_return :
  (stock_yield * market_value) / market_value * 100 = stock_yield * 100 := by
  sorry

end NUMINAMATH_CALUDE_stock_percentage_return_l231_23156


namespace NUMINAMATH_CALUDE_spot_horn_proportion_is_half_l231_23177

/-- Represents the proportion of spotted females and horned males -/
def spot_horn_proportion (total_cows : ℕ) (female_to_male_ratio : ℕ) (spotted_horned_difference : ℕ) : ℚ :=
  let male_cows := total_cows / (female_to_male_ratio + 1)
  let female_cows := female_to_male_ratio * male_cows
  (spotted_horned_difference : ℚ) / (female_cows - male_cows)

/-- Theorem stating the proportion of spotted females and horned males -/
theorem spot_horn_proportion_is_half :
  spot_horn_proportion 300 2 50 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_spot_horn_proportion_is_half_l231_23177


namespace NUMINAMATH_CALUDE_tan_beta_minus_2alpha_l231_23102

theorem tan_beta_minus_2alpha (α β : Real) 
  (h1 : Real.tan α = 1 / 2) 
  (h2 : Real.tan (α - β) = -1 / 3) : 
  Real.tan (β - 2 * α) = -1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_tan_beta_minus_2alpha_l231_23102


namespace NUMINAMATH_CALUDE_equation_solution_l231_23146

theorem equation_solution : 
  ∃! x : ℝ, x^2 + 6*x + 6*x * Real.sqrt (x + 5) = 24 ∧ 
  x = (-17 + Real.sqrt 277) / 2 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l231_23146


namespace NUMINAMATH_CALUDE_range_of_a_l231_23148

theorem range_of_a (a : ℝ) : 
  (∃! (s : Finset ℤ), s.card = 5 ∧ ∀ x ∈ s, (1 + a ≤ x ∧ x < 2)) → 
  -5 < a ∧ a ≤ -4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l231_23148


namespace NUMINAMATH_CALUDE_special_octagon_regions_l231_23165

/-- Represents an octagon with specific properties -/
structure SpecialOctagon where
  angles : Fin 8 → ℝ
  sides : Fin 8 → ℝ
  all_angles_135 : ∀ i, angles i = 135
  alternating_sides : ∀ i, sides i = if i % 2 = 0 then 1 else Real.sqrt 2

/-- Counts the regions formed by drawing all sides and diagonals of the octagon -/
def count_regions (o : SpecialOctagon) : ℕ :=
  84

/-- Theorem stating that the special octagon is divided into 84 regions -/
theorem special_octagon_regions (o : SpecialOctagon) : 
  count_regions o = 84 := by sorry

end NUMINAMATH_CALUDE_special_octagon_regions_l231_23165


namespace NUMINAMATH_CALUDE_w_value_l231_23158

def cubic_poly (x : ℝ) := x^3 - 4*x^2 + 2*x + 1

def second_poly (x u v w : ℝ) := x^3 + u*x^2 + v*x + w

theorem w_value (p q r u v w : ℝ) :
  cubic_poly p = 0 ∧ cubic_poly q = 0 ∧ cubic_poly r = 0 →
  second_poly (p + q) u v w = 0 ∧ second_poly (q + r) u v w = 0 ∧ second_poly (r + p) u v w = 0 →
  w = 15 := by sorry

end NUMINAMATH_CALUDE_w_value_l231_23158


namespace NUMINAMATH_CALUDE_section_4_eight_times_section_1_l231_23170

/-- Represents a circular target divided into sections -/
structure CircularTarget where
  r₁ : ℝ
  r₂ : ℝ
  r₃ : ℝ
  α : ℝ
  β : ℝ
  (r₁_pos : 0 < r₁)
  (r₂_pos : 0 < r₂)
  (r₃_pos : 0 < r₃)
  (r₁_lt_r₂ : r₁ < r₂)
  (r₂_lt_r₃ : r₂ < r₃)
  (α_pos : 0 < α)
  (β_pos : 0 < β)
  (section_equality : r₁^2 * β = α * (r₂^2 - r₁^2))
  (section_2_half_3 : β * (r₂^2 - r₁^2) = 2 * r₁^2 * β)

/-- The theorem stating that the area of section 4 is 8 times the area of section 1 -/
theorem section_4_eight_times_section_1 (t : CircularTarget) : 
  (t.β * (t.r₃^2 - t.r₂^2)) / (t.α * t.r₁^2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_section_4_eight_times_section_1_l231_23170


namespace NUMINAMATH_CALUDE_triangle_problem_l231_23128

/-- Given an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ A < π/2 →   -- A is acute
  0 < B ∧ B < π/2 →   -- B is acute
  0 < C ∧ C < π/2 →   -- C is acute
  Real.sqrt 3 * c = 2 * a * Real.sin C →  -- √3c = 2a sin C
  a = Real.sqrt 7 →  -- a = √7
  (1/2) * b * c * Real.sin A = (3 * Real.sqrt 3) / 2 →  -- Area of triangle ABC
  (A = π/3) ∧ (a + b + c = Real.sqrt 7 + 5) := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l231_23128


namespace NUMINAMATH_CALUDE_parallel_transitivity_l231_23126

-- Define the type for planes
def Plane : Type := Unit

-- Define the parallel relation between planes
def parallel (p q : Plane) : Prop := sorry

-- State the theorem
theorem parallel_transitivity (α β γ : Plane) 
  (h1 : α ≠ β) (h2 : β ≠ γ) (h3 : α ≠ γ)
  (h4 : parallel α β) (h5 : parallel β γ) : 
  parallel α γ := by sorry

end NUMINAMATH_CALUDE_parallel_transitivity_l231_23126


namespace NUMINAMATH_CALUDE_inequality_proof_l231_23103

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 / (a^3 + b^3 + a*b*c) + 1 / (b^3 + c^3 + a*b*c) + 1 / (c^3 + a^3 + a*b*c) ≤ 1 / (a*b*c) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l231_23103


namespace NUMINAMATH_CALUDE_M_definition_sum_of_digits_M_l231_23198

def M : ℕ := sorry

-- M is the smallest positive integer divisible by every positive integer less than 8
theorem M_definition : 
  M > 0 ∧ 
  (∀ k : ℕ, k > 0 → k < 8 → M % k = 0) ∧
  (∀ n : ℕ, n > 0 → (∀ k : ℕ, k > 0 → k < 8 → n % k = 0) → n ≥ M) :=
sorry

-- Function to calculate the sum of digits
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

-- Theorem stating that the sum of digits of M is 6
theorem sum_of_digits_M : sum_of_digits M = 6 :=
sorry

end NUMINAMATH_CALUDE_M_definition_sum_of_digits_M_l231_23198


namespace NUMINAMATH_CALUDE_geraldine_dolls_count_l231_23174

theorem geraldine_dolls_count (jazmin_dolls total_dolls : ℕ) 
  (h1 : jazmin_dolls = 1209)
  (h2 : total_dolls = 3395) :
  total_dolls - jazmin_dolls = 2186 :=
by sorry

end NUMINAMATH_CALUDE_geraldine_dolls_count_l231_23174


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l231_23135

/-- The quadratic equation (a-2)x^2 + x + a^2 - 4 = 0 has 0 as one of its roots -/
def has_zero_root (a : ℝ) : Prop :=
  ∃ x : ℝ, (a - 2) * x^2 + x + a^2 - 4 = 0 ∧ x = 0

/-- The value of a that satisfies the condition -/
def solution : ℝ := -2

theorem quadratic_equation_solution :
  ∀ a : ℝ, has_zero_root a → a = solution :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l231_23135


namespace NUMINAMATH_CALUDE_quartic_root_product_l231_23101

theorem quartic_root_product (k : ℝ) : 
  (∃ a b c d : ℝ, 
    (a^4 - 18*a^3 + k*a^2 + 200*a - 1984 = 0) ∧
    (b^4 - 18*b^3 + k*b^2 + 200*b - 1984 = 0) ∧
    (c^4 - 18*c^3 + k*c^2 + 200*c - 1984 = 0) ∧
    (d^4 - 18*d^3 + k*d^2 + 200*d - 1984 = 0) ∧
    (a * b = -32 ∨ a * c = -32 ∨ a * d = -32 ∨ b * c = -32 ∨ b * d = -32 ∨ c * d = -32)) →
  k = 86 := by
sorry

end NUMINAMATH_CALUDE_quartic_root_product_l231_23101


namespace NUMINAMATH_CALUDE_modulo_thirteen_residue_l231_23137

theorem modulo_thirteen_residue : (247 + 5 * 39 + 7 * 143 + 4 * 15) % 13 = 8 := by
  sorry

end NUMINAMATH_CALUDE_modulo_thirteen_residue_l231_23137


namespace NUMINAMATH_CALUDE_complex_square_pure_imaginary_l231_23147

theorem complex_square_pure_imaginary (a : ℝ) : 
  let z : ℂ := a + 3*I
  (∃ b : ℝ, z^2 = b*I ∧ b ≠ 0) → (a = 3 ∨ a = -3) :=
by sorry

end NUMINAMATH_CALUDE_complex_square_pure_imaginary_l231_23147


namespace NUMINAMATH_CALUDE_speed_ratio_is_three_fourths_l231_23195

/-- Represents the motion of objects A and B -/
structure Motion where
  vA : ℝ  -- Speed of A
  vB : ℝ  -- Speed of B

/-- The conditions of the problem -/
def satisfiesConditions (m : Motion) : Prop :=
  let distanceB := 800  -- Initial distance of B from O
  let t1 := 3           -- Time of first equidistance (in minutes)
  let t2 := 15          -- Time of second equidistance (in minutes)
  (t1 * m.vA = |distanceB - t1 * m.vB|) ∧   -- Equidistance at t1
  (t2 * m.vA = |distanceB - t2 * m.vB|)     -- Equidistance at t2

/-- The theorem to be proved -/
theorem speed_ratio_is_three_fourths :
  ∃ m : Motion, satisfiesConditions m ∧ m.vA / m.vB = 3 / 4 := by
  sorry


end NUMINAMATH_CALUDE_speed_ratio_is_three_fourths_l231_23195


namespace NUMINAMATH_CALUDE_fraction_equality_l231_23186

theorem fraction_equality (a b : ℝ) (h : b ≠ 0) : a / b = (a * b) / (b ^ 2) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l231_23186


namespace NUMINAMATH_CALUDE_van_distance_theorem_l231_23167

theorem van_distance_theorem (initial_time : ℝ) (speed : ℝ) :
  initial_time = 6 →
  speed = 30 →
  (3 / 2 : ℝ) * initial_time * speed = 270 :=
by
  sorry

end NUMINAMATH_CALUDE_van_distance_theorem_l231_23167


namespace NUMINAMATH_CALUDE_cosine_shift_equals_sine_l231_23175

open Real

theorem cosine_shift_equals_sine (m : ℝ) : (∀ x, cos (x + m) = sin x) → m = 3 * π / 2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_shift_equals_sine_l231_23175


namespace NUMINAMATH_CALUDE_tom_payment_is_nine_l231_23181

/-- The original price of the rare robot in dollars -/
def original_price : ℝ := 3

/-- The multiplier for the selling price -/
def price_multiplier : ℝ := 3

/-- The amount Tom should pay in dollars -/
def tom_payment : ℝ := original_price * price_multiplier

/-- Theorem stating that Tom should pay $9.00 for the rare robot -/
theorem tom_payment_is_nine : tom_payment = 9 := by
  sorry

end NUMINAMATH_CALUDE_tom_payment_is_nine_l231_23181


namespace NUMINAMATH_CALUDE_brick_length_satisfies_wall_requirements_l231_23159

/-- The length of a brick in centimeters -/
def brick_length : ℝ := 25

/-- The width of a brick in centimeters -/
def brick_width : ℝ := 11.25

/-- The height of a brick in centimeters -/
def brick_height : ℝ := 6

/-- The length of the wall in centimeters -/
def wall_length : ℝ := 900

/-- The height of the wall in centimeters -/
def wall_height : ℝ := 600

/-- The thickness of the wall in centimeters -/
def wall_thickness : ℝ := 22.5

/-- The number of bricks needed to build the wall -/
def num_bricks : ℕ := 7200

/-- Theorem stating that the given brick length satisfies the wall building requirements -/
theorem brick_length_satisfies_wall_requirements :
  wall_length * wall_height * wall_thickness = 
  (brick_length * brick_width * brick_height) * num_bricks := by
  sorry

#check brick_length_satisfies_wall_requirements

end NUMINAMATH_CALUDE_brick_length_satisfies_wall_requirements_l231_23159


namespace NUMINAMATH_CALUDE_original_class_count_original_class_count_is_seven_l231_23193

theorem original_class_count : ℕ → Prop :=
  fun x : ℕ =>
    (280 % x = 0) ∧
    (585 % (x + 6) = 0) ∧
    x > 0 ∧
    (∀ y : ℕ, y ≠ x → (280 % y = 0 ∧ 585 % (y + 6) = 0 ∧ y > 0) → False) →
    x = 7

/-- The original number of classes in the grade is 7. -/
theorem original_class_count_is_seven : original_class_count 7 := by
  sorry

end NUMINAMATH_CALUDE_original_class_count_original_class_count_is_seven_l231_23193


namespace NUMINAMATH_CALUDE_total_problems_l231_23116

def daily_record : List Int := [-3, 5, -4, 2, -1, 1, 0, -3, 8, 7]

theorem total_problems (record : List Int) (h : record = daily_record) :
  (List.sum record + 60 : Int) = 72 := by
  sorry

end NUMINAMATH_CALUDE_total_problems_l231_23116


namespace NUMINAMATH_CALUDE_rosy_fish_count_l231_23111

/-- Given that Lilly has 10 fish and the total number of fish is 22,
    prove that Rosy has 12 fish. -/
theorem rosy_fish_count (lilly_fish : ℕ) (total_fish : ℕ) (h1 : lilly_fish = 10) (h2 : total_fish = 22) :
  total_fish - lilly_fish = 12 := by
  sorry

end NUMINAMATH_CALUDE_rosy_fish_count_l231_23111


namespace NUMINAMATH_CALUDE_integer_roots_of_polynomial_l231_23196

/-- The set of all possible integer roots for the polynomial x^4 + 4x^3 + a_2 x^2 + a_1 x - 60 = 0 -/
def possible_roots : Set ℤ := {1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 10, -10, 12, -12, 15, -15, 20, -20, 30, -30, 60, -60}

/-- The polynomial x^4 + 4x^3 + a_2 x^2 + a_1 x - 60 -/
def polynomial (a₂ a₁ x : ℤ) : ℤ := x^4 + 4*x^3 + a₂*x^2 + a₁*x - 60

theorem integer_roots_of_polynomial (a₂ a₁ : ℤ) :
  ∀ x : ℤ, polynomial a₂ a₁ x = 0 → x ∈ possible_roots :=
sorry

end NUMINAMATH_CALUDE_integer_roots_of_polynomial_l231_23196


namespace NUMINAMATH_CALUDE_alloy_mix_solvable_l231_23130

/-- Represents an alloy of copper and tin -/
structure Alloy where
  mass : ℝ
  copper_percentage : ℝ

/-- Represents the problem of mixing two alloys -/
def AlloyMixProblem (alloy1 alloy2 : Alloy) (target_mass : ℝ) (target_percentage : ℝ) :=
  alloy1.mass ≥ 0 ∧
  alloy2.mass ≥ 0 ∧
  alloy1.copper_percentage ≥ 0 ∧ alloy1.copper_percentage ≤ 100 ∧
  alloy2.copper_percentage ≥ 0 ∧ alloy2.copper_percentage ≤ 100 ∧
  target_mass > 0 ∧
  target_percentage ≥ 0 ∧ target_percentage ≤ 100

theorem alloy_mix_solvable (alloy1 alloy2 : Alloy) (target_mass : ℝ) (p : ℝ) :
  AlloyMixProblem alloy1 alloy2 target_mass p →
  (alloy1.mass = 3 ∧ 
   alloy2.mass = 7 ∧ 
   alloy1.copper_percentage = 40 ∧ 
   alloy2.copper_percentage = 30 ∧
   target_mass = 8) →
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ alloy1.mass ∧ 
            0 ≤ target_mass - x ∧ target_mass - x ≤ alloy2.mass ∧
            alloy1.copper_percentage * x / 100 + alloy2.copper_percentage * (target_mass - x) / 100 = target_mass * p / 100) ↔
  (31.25 ≤ p ∧ p ≤ 33.75) :=
by sorry

end NUMINAMATH_CALUDE_alloy_mix_solvable_l231_23130
