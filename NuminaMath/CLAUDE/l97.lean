import Mathlib

namespace ali_spending_ratio_l97_9784

theorem ali_spending_ratio :
  ∀ (initial_amount food_cost glasses_cost remaining : ℕ),
  initial_amount = 480 →
  glasses_cost = (initial_amount - food_cost) / 3 →
  remaining = initial_amount - food_cost - glasses_cost →
  remaining = 160 →
  food_cost * 2 = initial_amount :=
λ initial_amount food_cost glasses_cost remaining
  h_initial h_glasses h_remaining h_final =>
sorry

end ali_spending_ratio_l97_9784


namespace parallel_vectors_problem_l97_9793

/-- Given vectors a, b, and c in ℝ², prove that if a is parallel to c and c = a + 3b, then x = 4 -/
theorem parallel_vectors_problem (x : ℝ) : 
  let a : Fin 2 → ℝ := ![1, -4]
  let b : Fin 2 → ℝ := ![-1, x]
  let c : Fin 2 → ℝ := a + 3 • b
  (∃ (k : ℝ), c = k • a) → x = 4 := by
  sorry

end parallel_vectors_problem_l97_9793


namespace negation_equivalence_l97_9766

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∃ x : ℝ, x^3 - x^2 + 1 > 0) := by
  sorry

end negation_equivalence_l97_9766


namespace triangle_altitude_l97_9778

theorem triangle_altitude (area : ℝ) (base : ℝ) (altitude : ℝ) :
  area = 960 →
  base = 48 →
  area = (1 / 2) * base * altitude →
  altitude = 40 := by
sorry

end triangle_altitude_l97_9778


namespace large_ball_uses_300_rubber_bands_l97_9710

/-- Calculates the number of rubber bands used in a large ball -/
def large_ball_rubber_bands (total_rubber_bands : ℕ) (small_balls : ℕ) (rubber_bands_per_small : ℕ) (large_balls : ℕ) : ℕ :=
  (total_rubber_bands - small_balls * rubber_bands_per_small) / large_balls

/-- Proves that a large ball uses 300 rubber bands given the problem conditions -/
theorem large_ball_uses_300_rubber_bands :
  large_ball_rubber_bands 5000 22 50 13 = 300 := by
  sorry

end large_ball_uses_300_rubber_bands_l97_9710


namespace mascot_purchase_equations_l97_9731

/-- Represents the purchase of mascot dolls and keychains --/
structure MascotPurchase where
  dolls : ℕ
  keychains : ℕ
  total_cost : ℕ
  doll_price : ℕ
  keychain_price : ℕ

/-- The correct system of equations for the mascot purchase --/
def correct_equations (p : MascotPurchase) : Prop :=
  p.keychains = 2 * p.dolls ∧ 
  p.total_cost = p.doll_price * p.dolls + p.keychain_price * p.keychains

/-- Theorem stating the correct system of equations for the given conditions --/
theorem mascot_purchase_equations :
  ∀ (p : MascotPurchase), 
    p.total_cost = 5000 ∧ 
    p.doll_price = 60 ∧ 
    p.keychain_price = 20 →
    correct_equations p :=
by
  sorry


end mascot_purchase_equations_l97_9731


namespace susan_bob_cat_difference_l97_9722

/-- Proves that Susan has 6 more cats than Bob after all exchanges --/
theorem susan_bob_cat_difference : 
  let susan_initial : ℕ := 21
  let bob_initial : ℕ := 3
  let emma_initial : ℕ := 8
  let neighbor_to_susan : ℕ := 12
  let neighbor_to_bob : ℕ := 14
  let neighbor_to_emma : ℕ := 6
  let susan_to_bob : ℕ := 6
  let emma_to_susan : ℕ := 5
  let emma_to_bob : ℕ := 3

  let susan_final := susan_initial + neighbor_to_susan - susan_to_bob + emma_to_susan
  let bob_final := bob_initial + neighbor_to_bob + susan_to_bob + emma_to_bob

  susan_final - bob_final = 6 :=
by sorry

end susan_bob_cat_difference_l97_9722


namespace factorization_proof_l97_9753

theorem factorization_proof (z : ℝ) : 
  75 * z^12 + 162 * z^24 + 27 = 3 * (9 + z^12 * (25 + 54 * z^12)) := by
  sorry

end factorization_proof_l97_9753


namespace work_completion_time_l97_9782

/-- Represents the work rate of one person per hour -/
structure WorkRate where
  man : ℝ
  woman : ℝ

/-- Represents a work scenario -/
structure WorkScenario where
  men : ℕ
  women : ℕ
  hours_per_day : ℝ
  days : ℝ

def total_work (rate : WorkRate) (scenario : WorkScenario) : ℝ :=
  (scenario.men * rate.man + scenario.women * rate.woman) * scenario.hours_per_day * scenario.days

theorem work_completion_time 
  (rate : WorkRate)
  (scenario1 : WorkScenario)
  (scenario2 : WorkScenario)
  (scenario3 : WorkScenario) :
  scenario1.men = 1 →
  scenario1.women = 3 →
  scenario1.hours_per_day = 7 →
  scenario1.days = 5 →
  scenario2.men = 4 →
  scenario2.women = 4 →
  scenario2.hours_per_day = 3 →
  scenario3.men = 7 →
  scenario3.women = 0 →
  scenario3.hours_per_day = 4 →
  scenario3.days = 5.000000000000001 →
  total_work rate scenario1 = total_work rate scenario3 →
  scenario2.days = 2.5 := by
  sorry

end work_completion_time_l97_9782


namespace simplify_and_evaluate_l97_9780

theorem simplify_and_evaluate (m : ℝ) (h : m = 4 * Real.sqrt 3) :
  (1 - m / (m - 3)) / ((m^2 - 3*m) / (m^2 - 6*m + 9)) = -(Real.sqrt 3) / 4 := by
  sorry

end simplify_and_evaluate_l97_9780


namespace power_relationship_l97_9738

theorem power_relationship (a b c : ℝ) 
  (ha : a = Real.rpow 0.8 5.2)
  (hb : b = Real.rpow 0.8 5.5)
  (hc : c = Real.rpow 5.2 0.1) : 
  b < a ∧ a < c := by
  sorry

end power_relationship_l97_9738


namespace weight_of_three_liters_l97_9754

/-- Given that 1 liter weighs 2.2 pounds, prove that 3 liters weigh 6.6 pounds. -/
theorem weight_of_three_liters (weight_per_liter : ℝ) (h : weight_per_liter = 2.2) :
  3 * weight_per_liter = 6.6 := by
  sorry

end weight_of_three_liters_l97_9754


namespace max_value_of_expression_l97_9716

theorem max_value_of_expression (a b c d : ℝ) 
  (nonneg_a : 0 ≤ a) (nonneg_b : 0 ≤ b) (nonneg_c : 0 ≤ c) (nonneg_d : 0 ≤ d)
  (sum_one : a + b + c + d = 1) :
  (∀ x y z w : ℝ, x ≥ 0 → y ≥ 0 → z ≥ 0 → w ≥ 0 → x + y + z + w = 1 →
    (x*y)/(x+y) + (x*z)/(x+z) + (x*w)/(x+w) + 
    (y*z)/(y+z) + (y*w)/(y+w) + (z*w)/(z+w) ≤ 1/2) ∧
  ((a*b)/(a+b) + (a*c)/(a+c) + (a*d)/(a+d) + 
   (b*c)/(b+c) + (b*d)/(b+d) + (c*d)/(c+d) = 1/2) :=
by sorry

end max_value_of_expression_l97_9716


namespace existence_of_additive_approximation_l97_9774

/-- Given a function f: ℝ → ℝ satisfying |f(x+y) - f(x) - f(y)| ≤ 1 for all x, y ∈ ℝ,
    there exists an additive function g: ℝ → ℝ such that |f(x) - g(x)| ≤ 1 for all x ∈ ℝ. -/
theorem existence_of_additive_approximation (f : ℝ → ℝ) 
    (h : ∀ x y : ℝ, |f (x + y) - f x - f y| ≤ 1) :
  ∃ g : ℝ → ℝ, (∀ x y : ℝ, g (x + y) = g x + g y) ∧ 
    (∀ x : ℝ, |f x - g x| ≤ 1) := by
  sorry

end existence_of_additive_approximation_l97_9774


namespace roots_relation_l97_9783

-- Define the polynomials f and g
def f (x : ℝ) : ℝ := x^3 + 2*x^2 + 3*x + 4
def g (x b c d : ℝ) : ℝ := x^3 + b*x^2 + c*x + d

-- State the theorem
theorem roots_relation (b c d : ℝ) : 
  (∀ x y : ℝ, x ≠ y → f x = 0 → f y = 0 → x ≠ y) →  -- f has distinct roots
  (∀ r : ℝ, f r = 0 → g (r^3) b c d = 0) →          -- roots of g are cubes of roots of f
  b = -8 ∧ c = -36 ∧ d = -64 := by
  sorry

end roots_relation_l97_9783


namespace age_difference_l97_9741

/-- Given information about Lexie and her siblings' ages, prove the age difference between her brother and sister. -/
theorem age_difference (lexie_age : ℕ) (brother_age_diff : ℕ) (sister_age_factor : ℕ) 
  (h1 : lexie_age = 8)
  (h2 : lexie_age = brother_age_diff + lexie_age - 6)
  (h3 : sister_age_factor * lexie_age = 2 * lexie_age) :
  2 * lexie_age - (lexie_age - 6) = 14 := by
  sorry

end age_difference_l97_9741


namespace homologous_functions_count_l97_9785

-- Define the function
def f (x : ℝ) : ℝ := x^2 + 1

-- Define the range
def range : Set ℝ := {1, 3}

-- Define a valid domain
def is_valid_domain (D : Set ℝ) : Prop :=
  (∀ x ∈ D, f x ∈ range) ∧ (∀ y ∈ range, ∃ x ∈ D, f x = y)

-- Theorem statement
theorem homologous_functions_count :
  ∃! (domains : Finset (Set ℝ)), 
    domains.card = 3 ∧ 
    (∀ D ∈ domains, is_valid_domain D) ∧
    (∀ D : Set ℝ, is_valid_domain D → D ∈ domains) :=
sorry

end homologous_functions_count_l97_9785


namespace square_triangle_ratio_l97_9756

theorem square_triangle_ratio (a : ℝ) (h : a > 0) :
  let square_side := a
  let triangle_leg := a * Real.sqrt 2
  let triangle_hypotenuse := triangle_leg * Real.sqrt 2
  triangle_hypotenuse / square_side = 2 := by
sorry

end square_triangle_ratio_l97_9756


namespace box_dimensions_l97_9733

theorem box_dimensions (a b c : ℝ) 
  (h1 : a + c = 17)
  (h2 : a + b = 13)
  (h3 : b + c = 20)
  (h4 : a < b)
  (h5 : b < c) :
  a = 5 ∧ b = 8 ∧ c = 12 := by
  sorry

end box_dimensions_l97_9733


namespace income_comparison_l97_9736

theorem income_comparison (juan tim mary : ℝ) 
  (h1 : tim = juan * (1 - 0.4))
  (h2 : mary = tim * (1 + 0.7)) :
  mary = juan * 1.02 := by
sorry

end income_comparison_l97_9736


namespace exam_analysis_theorem_l97_9791

structure StatisticalAnalysis where
  population_size : ℕ
  sample_size : ℕ
  sample_is_subset : sample_size ≤ population_size

def is_population (sa : StatisticalAnalysis) (n : ℕ) : Prop :=
  n = sa.population_size

def is_sample (sa : StatisticalAnalysis) (n : ℕ) : Prop :=
  n = sa.sample_size

def is_sample_size (sa : StatisticalAnalysis) (n : ℕ) : Prop :=
  n = sa.sample_size

-- The statement we want to prove incorrect
def each_examinee_is_individual_unit (sa : StatisticalAnalysis) : Prop :=
  False  -- This is set to False to represent that the statement is incorrect

theorem exam_analysis_theorem (sa : StatisticalAnalysis) 
  (h_pop : sa.population_size = 13000)
  (h_sample : sa.sample_size = 500) :
  is_population sa 13000 ∧ 
  is_sample sa 500 ∧ 
  is_sample_size sa 500 ∧ 
  ¬(each_examinee_is_individual_unit sa) := by
  sorry

#check exam_analysis_theorem

end exam_analysis_theorem_l97_9791


namespace exactly_one_even_l97_9781

theorem exactly_one_even (a b c : ℕ) : 
  (a % 2 = 0 ∧ b % 2 ≠ 0 ∧ c % 2 ≠ 0) ∨ 
  (a % 2 ≠ 0 ∧ b % 2 = 0 ∧ c % 2 ≠ 0) ∨ 
  (a % 2 ≠ 0 ∧ b % 2 ≠ 0 ∧ c % 2 = 0) :=
by
  sorry

#check exactly_one_even

end exactly_one_even_l97_9781


namespace solution_count_l97_9768

def is_solution (a b : ℕ+) : Prop :=
  (1 : ℚ) / a.val - (1 : ℚ) / b.val = (1 : ℚ) / 2018

theorem solution_count :
  ∃! (s : Finset (ℕ+ × ℕ+)), 
    (∀ (p : ℕ+ × ℕ+), p ∈ s ↔ is_solution p.1 p.2) ∧ 
    s.card = 4 :=
by sorry

end solution_count_l97_9768


namespace infinitely_many_special_numbers_l97_9702

/-- A natural number n such that n^2 + 1 has no divisors of the form k^2 + 1 except 1 and itself. -/
def SpecialNumber (n : ℕ) : Prop :=
  ∀ k : ℕ, k^2 + 1 ∣ n^2 + 1 → k^2 + 1 = 1 ∨ k^2 + 1 = n^2 + 1

/-- The set of SpecialNumbers is infinite. -/
theorem infinitely_many_special_numbers : Set.Infinite {n : ℕ | SpecialNumber n} := by
  sorry


end infinitely_many_special_numbers_l97_9702


namespace expression_simplification_l97_9734

theorem expression_simplification (x y z : ℝ) (h : y ≠ 0) :
  (6 * x^3 * y^4 * z - 4 * x^2 * y^3 * z + 2 * x * y^3) / (2 * x * y^3) = 3 * x^2 * y * z - 2 * x * z + 1 :=
by sorry

end expression_simplification_l97_9734


namespace twenty_percent_women_without_plan_l97_9705

/-- Represents a company with workers and their retirement plan status -/
structure Company where
  total_workers : ℕ
  workers_without_plan : ℕ
  men_with_plan : ℕ
  total_men : ℕ
  total_women : ℕ

/-- Conditions for the company -/
def company_conditions (c : Company) : Prop :=
  c.workers_without_plan = c.total_workers / 3 ∧
  c.men_with_plan = (c.total_workers - c.workers_without_plan) * 2 / 5 ∧
  c.total_men = 112 ∧
  c.total_women = 98 ∧
  c.total_workers = c.total_men + c.total_women

/-- The percentage of women without a retirement plan -/
def women_without_plan_percentage (c : Company) : ℚ :=
  let women_without_plan := c.workers_without_plan - (c.total_men - c.men_with_plan)
  (women_without_plan : ℚ) / c.workers_without_plan * 100

/-- Theorem stating that 20% of workers without a retirement plan are women -/
theorem twenty_percent_women_without_plan (c : Company) 
  (h : company_conditions c) : women_without_plan_percentage c = 20 := by
  sorry


end twenty_percent_women_without_plan_l97_9705


namespace integer_solution_problem_l97_9708

theorem integer_solution_problem :
  let S : Set (ℤ × ℤ × ℤ) := {(a, b, c) | a + b + c = 15 ∧ (a - 3)^3 + (b - 5)^3 + (c - 7)^3 = 540}
  S = {(12, 0, 3), (-2, 14, 3), (-1, 0, 16), (-2, 1, 16)} := by
  sorry

end integer_solution_problem_l97_9708


namespace smallest_valid_number_last_digits_l97_9759

def is_valid_number (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 3 ∨ d = 6

def contains_both (n : ℕ) : Prop :=
  3 ∈ n.digits 10 ∧ 6 ∈ n.digits 10

def last_four_digits (n : ℕ) : ℕ :=
  n % 10000

theorem smallest_valid_number_last_digits :
  ∃ m : ℕ,
    m > 0 ∧
    m % 3 = 0 ∧
    m % 6 = 0 ∧
    is_valid_number m ∧
    contains_both m ∧
    (∀ k : ℕ, k > 0 ∧ k % 3 = 0 ∧ k % 6 = 0 ∧ is_valid_number k ∧ contains_both k → m ≤ k) ∧
    last_four_digits m = 3630 :=
  sorry

end smallest_valid_number_last_digits_l97_9759


namespace tomorrow_is_saturday_l97_9727

-- Define the days of the week
inductive Day : Type
  | Monday : Day
  | Tuesday : Day
  | Wednesday : Day
  | Thursday : Day
  | Friday : Day
  | Saturday : Day
  | Sunday : Day

-- Define a function to get the next day
def nextDay (d : Day) : Day :=
  match d with
  | Day.Monday => Day.Tuesday
  | Day.Tuesday => Day.Wednesday
  | Day.Wednesday => Day.Thursday
  | Day.Thursday => Day.Friday
  | Day.Friday => Day.Saturday
  | Day.Saturday => Day.Sunday
  | Day.Sunday => Day.Monday

-- Define a function to add days
def addDays (d : Day) (n : Nat) : Day :=
  match n with
  | 0 => d
  | Nat.succ m => nextDay (addDays d m)

-- Theorem statement
theorem tomorrow_is_saturday (dayBeforeYesterday : Day) :
  addDays dayBeforeYesterday 5 = Day.Monday →
  addDays dayBeforeYesterday 7 = Day.Saturday :=
by
  sorry


end tomorrow_is_saturday_l97_9727


namespace chocolate_distribution_l97_9796

theorem chocolate_distribution (total_chocolate : ℚ) (num_packages : ℕ) (neighbor_packages : ℕ) :
  total_chocolate = 72 / 7 →
  num_packages = 6 →
  neighbor_packages = 2 →
  (total_chocolate / num_packages) * neighbor_packages = 24 / 7 := by
  sorry

end chocolate_distribution_l97_9796


namespace total_crayons_l97_9792

/-- Given a box of crayons where there are four times as many red crayons as blue crayons,
    and there are 3 blue crayons, prove that the total number of crayons is 15. -/
theorem total_crayons (blue : ℕ) (red : ℕ) (h1 : blue = 3) (h2 : red = 4 * blue) :
  blue + red = 15 := by
  sorry

end total_crayons_l97_9792


namespace baseball_cost_value_l97_9795

/-- The amount Mike spent on toys -/
def total_spent : ℚ := 20.52

/-- The cost of marbles -/
def marbles_cost : ℚ := 9.05

/-- The cost of the football -/
def football_cost : ℚ := 4.95

/-- The cost of the baseball -/
def baseball_cost : ℚ := total_spent - (marbles_cost + football_cost)

theorem baseball_cost_value : baseball_cost = 6.52 := by
  sorry

end baseball_cost_value_l97_9795


namespace even_number_less_than_square_l97_9713

theorem even_number_less_than_square (m : ℕ) (h1 : m > 1) (h2 : Even m) : m < m^2 := by
  sorry

end even_number_less_than_square_l97_9713


namespace min_difference_ln2_l97_9747

/-- Given functions f and g, prove that the minimum value of x₁ - x₂ is ln(2) when f(x₁) = g(x₂) -/
theorem min_difference_ln2 (f g : ℝ → ℝ) (x₁ x₂ : ℝ) 
  (hf : f = fun x ↦ Real.log (x / 2) + 1 / 2)
  (hg : g = fun x ↦ Real.exp (x - 2))
  (hx₁ : x₁ > 0)
  (hequal : f x₁ = g x₂) :
  ∃ (min : ℝ), min = Real.log 2 ∧ ∀ y₁ y₂, f y₁ = g y₂ → y₁ - y₂ ≥ min :=
by sorry

end min_difference_ln2_l97_9747


namespace z_purely_imaginary_z_in_fourth_quadrant_l97_9706

/-- Definition of the complex number z in terms of m -/
def z (m : ℝ) : ℂ := Complex.mk (3*m - 2) (m - 1)

/-- z is purely imaginary if and only if m = 2/3 -/
theorem z_purely_imaginary (m : ℝ) : z m = Complex.I * Complex.im (z m) ↔ m = 2/3 := by
  sorry

/-- z lies in the fourth quadrant if and only if 2/3 < m < 1 -/
theorem z_in_fourth_quadrant (m : ℝ) : 
  (Complex.re (z m) > 0 ∧ Complex.im (z m) < 0) ↔ (2/3 < m ∧ m < 1) := by
  sorry

end z_purely_imaginary_z_in_fourth_quadrant_l97_9706


namespace fraction_equality_solution_l97_9701

theorem fraction_equality_solution (x : ℚ) :
  (x + 11) / (x - 4) = (x - 1) / (x + 6) → x = -31/11 := by
  sorry

end fraction_equality_solution_l97_9701


namespace smallest_linear_combination_l97_9707

theorem smallest_linear_combination (m n : ℤ) : 
  ∃ (k : ℕ), k > 0 ∧ (∃ (a b : ℤ), k = 2024 * a + 48048 * b) ∧ 
  (∀ (l : ℕ) (c d : ℤ), l > 0 ∧ l = 2024 * c + 48048 * d → k ≤ l) ∧ 
  k = 88 := by
sorry

end smallest_linear_combination_l97_9707


namespace dark_tiles_fraction_is_three_fourths_l97_9703

/-- Represents a square tiling pattern -/
structure TilingPattern where
  size : ℕ
  dark_tiles_per_corner : ℕ
  is_symmetrical : Bool

/-- Calculates the fraction of dark tiles in a tiling pattern -/
def fraction_of_dark_tiles (pattern : TilingPattern) : ℚ :=
  if pattern.is_symmetrical
  then (4 * pattern.dark_tiles_per_corner : ℚ) / (pattern.size * pattern.size : ℚ)
  else 0

/-- Theorem stating that a 4x4 symmetrical pattern with 3 dark tiles per corner 
    has 3/4 of its tiles dark -/
theorem dark_tiles_fraction_is_three_fourths 
  (pattern : TilingPattern) 
  (h1 : pattern.size = 4) 
  (h2 : pattern.dark_tiles_per_corner = 3) 
  (h3 : pattern.is_symmetrical = true) : 
  fraction_of_dark_tiles pattern = 3/4 := by
  sorry

end dark_tiles_fraction_is_three_fourths_l97_9703


namespace shekar_average_marks_l97_9770

def shekar_marks : List ℕ := [76, 65, 82, 67, 85]

theorem shekar_average_marks :
  (shekar_marks.sum / shekar_marks.length : ℚ) = 75 := by
  sorry

end shekar_average_marks_l97_9770


namespace longest_altitudes_sum_is_14_l97_9744

/-- A triangle with sides 6, 8, and 10 -/
structure Triangle :=
  (a : ℝ) (b : ℝ) (c : ℝ)
  (ha : a = 6)
  (hb : b = 8)
  (hc : c = 10)

/-- The sum of the lengths of the two longest altitudes in the triangle -/
def longest_altitudes_sum (t : Triangle) : ℝ := sorry

/-- Theorem stating that the sum of the lengths of the two longest altitudes is 14 -/
theorem longest_altitudes_sum_is_14 (t : Triangle) : longest_altitudes_sum t = 14 := by
  sorry

end longest_altitudes_sum_is_14_l97_9744


namespace angle_sum_is_420_l97_9718

/-- A geometric configuration with six angles A, B, C, D, E, and F -/
structure GeometricConfiguration where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  E : ℝ
  F : ℝ

/-- The theorem stating that if E = 30°, then the sum of all angles is 420° -/
theorem angle_sum_is_420 (config : GeometricConfiguration) 
  (h_E : config.E = 30) : 
  config.A + config.B + config.C + config.D + config.E + config.F = 420 := by
  sorry

#check angle_sum_is_420

end angle_sum_is_420_l97_9718


namespace equation_solution_l97_9789

theorem equation_solution :
  let f : ℝ → ℝ := λ x => x + 36 / (x - 3)
  {x : ℝ | f x = -12} = {0, -9} := by sorry

end equation_solution_l97_9789


namespace quadratic_equation_unique_solution_l97_9764

theorem quadratic_equation_unique_solution (a c : ℝ) : 
  (∃! x, 2 * a * x^2 + 10 * x + c = 0) →
  a + c = 12 →
  a < c →
  a = 1.15 ∧ c = 10.85 := by
sorry

end quadratic_equation_unique_solution_l97_9764


namespace cubic_root_transformation_l97_9714

theorem cubic_root_transformation (p : ℝ) : 
  p^3 + p - 3 = 0 → (p^2)^3 + 2*(p^2)^2 + p^2 - 9 = 0 := by
  sorry

end cubic_root_transformation_l97_9714


namespace profit_function_and_maximum_profit_constraint_and_price_l97_9794

/-- Weekly profit function -/
def W (x : ℝ) : ℝ := -10 * x^2 + 200 * x + 15000

/-- Initial cost per box in yuan -/
def initial_cost : ℝ := 70

/-- Initial selling price per box in yuan -/
def initial_price : ℝ := 120

/-- Initial weekly sales volume in boxes -/
def initial_sales : ℝ := 300

/-- Sales increase per yuan of price reduction -/
def sales_increase_rate : ℝ := 10

theorem profit_function_and_maximum (x : ℝ) :
  W x = -10 * x^2 + 200 * x + 15000 ∧
  (∀ y : ℝ, W y ≤ W 10) ∧
  W 10 = 16000 := by sorry

theorem profit_constraint_and_price (x : ℝ) :
  W x = 15960 →
  x ≤ 12 →
  initial_price - 12 = 108 := by sorry

end profit_function_and_maximum_profit_constraint_and_price_l97_9794


namespace last_nonzero_digit_aperiodic_l97_9720

/-- d_n is the last nonzero digit of n! -/
def last_nonzero_digit (n : ℕ) : ℕ :=
  sorry

/-- The sequence d_n is aperiodic -/
theorem last_nonzero_digit_aperiodic :
  ∀ T n₀ : ℕ, ∃ n : ℕ, n ≥ n₀ ∧ last_nonzero_digit (n + T) ≠ last_nonzero_digit n :=
sorry

end last_nonzero_digit_aperiodic_l97_9720


namespace sum_le_product_plus_two_l97_9740

theorem sum_le_product_plus_two (x y z : ℝ) (h : x^2 + y^2 + z^2 = 2) :
  x + y + z ≤ x * y * z + 2 := by
sorry

end sum_le_product_plus_two_l97_9740


namespace game_points_sequence_l97_9745

theorem game_points_sequence (a : ℕ → ℕ) : 
  a 1 = 2 ∧ 
  a 3 = 5 ∧ 
  a 4 = 8 ∧ 
  a 5 = 12 ∧ 
  a 6 = 17 ∧ 
  (∀ n : ℕ, n > 1 → (a (n + 1) - a n) - (a n - a (n - 1)) = 1) →
  a 2 = 3 := by
sorry

end game_points_sequence_l97_9745


namespace minimum_time_is_110_l97_9776

/-- Represents the time taken by each teacher to examine one student -/
structure TeacherTime where
  time : ℕ

/-- Represents the problem of finding the minimum examination time -/
structure ExaminationProblem where
  teacher1 : TeacherTime
  teacher2 : TeacherTime
  totalStudents : ℕ

/-- Calculates the minimum examination time for the given problem -/
def minimumExaminationTime (problem : ExaminationProblem) : ℕ :=
  sorry

/-- Theorem stating that the minimum examination time for the given problem is 110 minutes -/
theorem minimum_time_is_110 (problem : ExaminationProblem) 
  (h1 : problem.teacher1.time = 12)
  (h2 : problem.teacher2.time = 7)
  (h3 : problem.totalStudents = 25) :
  minimumExaminationTime problem = 110 := by
  sorry

end minimum_time_is_110_l97_9776


namespace intersection_probability_l97_9760

def U : Finset Nat := {1, 2, 3, 4, 5}
def I : Finset (Finset Nat) := Finset.powerset U

def favorable_pairs : Nat :=
  (Finset.powerset U).card * (Finset.filter (fun s => s.card = 3) (Finset.powerset U)).card

def total_pairs : Nat := (I.card.choose 2)

theorem intersection_probability :
  (favorable_pairs : ℚ) / total_pairs = 5 / 62 := by sorry

end intersection_probability_l97_9760


namespace brownie_calories_l97_9779

-- Define the parameters of the problem
def cake_slices : ℕ := 8
def calories_per_cake_slice : ℕ := 347
def brownies : ℕ := 6
def calorie_difference : ℕ := 526

-- Define the function to calculate calories per brownie
def calories_per_brownie : ℕ :=
  ((cake_slices * calories_per_cake_slice - calorie_difference) / brownies : ℕ)

-- Theorem statement
theorem brownie_calories :
  calories_per_brownie = 375 := by
  sorry

end brownie_calories_l97_9779


namespace james_writing_pages_l97_9757

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

end james_writing_pages_l97_9757


namespace origin_is_solution_l97_9730

/-- The equation defining the set of points -/
def equation (x y : ℝ) : Prop :=
  x^2 * (y + y^2) = y^3 + x^4

/-- Theorem stating that (0, 0) is a solution to the equation -/
theorem origin_is_solution : equation 0 0 := by
  sorry

end origin_is_solution_l97_9730


namespace complex_equation_solution_l97_9761

theorem complex_equation_solution :
  let z : ℂ := (1 - I)^2 + 1 + 3*I
  ∀ a b : ℝ, z^2 + a*z + b = 1 - I → a = -3 ∧ b = 4 := by
  sorry

end complex_equation_solution_l97_9761


namespace find_A_minus_C_l97_9746

theorem find_A_minus_C (A B C : ℕ) 
  (h1 : A + B = 84)
  (h2 : B + C = 60)
  (h3 : A = B + B + B + B + B + B)
  (h4 : A ≠ B ∧ B ≠ C ∧ A ≠ C) :
  A - C = 24 := by
sorry

end find_A_minus_C_l97_9746


namespace complex_number_subtraction_l97_9755

theorem complex_number_subtraction (z : ℂ) (a b : ℝ) : 
  z = 2 + I → Complex.re z = a → Complex.im z = b → a - b = 1 := by
  sorry

end complex_number_subtraction_l97_9755


namespace rod_solution_l97_9715

/-- Represents a rod divided into parts -/
structure Rod where
  m : ℕ  -- number of red parts
  n : ℕ  -- number of black parts
  x : ℕ  -- number of coinciding lines
  total_segments : ℕ  -- total number of segments after cutting
  longest_segments : ℕ  -- number of longest segments

/-- Conditions for the rod problem -/
def rod_conditions (r : Rod) : Prop :=
  r.m > r.n ∧
  r.total_segments = 170 ∧
  r.longest_segments = 100

/-- Theorem stating the solution to the rod problem -/
theorem rod_solution (r : Rod) (h : rod_conditions r) : r.m = 13 ∧ r.n = 156 :=
sorry

/-- Lemma stating that x + 1 is a common divisor of m and n -/
lemma common_divisor (r : Rod) : (r.x + 1) ∣ r.m ∧ (r.x + 1) ∣ r.n :=
sorry

end rod_solution_l97_9715


namespace substitution_remainder_l97_9758

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

end substitution_remainder_l97_9758


namespace pythagorean_triple_with_primes_l97_9767

theorem pythagorean_triple_with_primes (x y z : ℤ) :
  x^2 + y^2 = z^2 →
  (Prime y ∧ y > 5 ∧ Prime z ∧ z > 5) ∨
  (Prime x ∧ x > 5 ∧ Prime z ∧ z > 5) ∨
  (Prime x ∧ x > 5 ∧ Prime y ∧ y > 5) →
  60 ∣ x ∨ 60 ∣ y ∨ 60 ∣ z :=
by sorry

end pythagorean_triple_with_primes_l97_9767


namespace rocket_heights_l97_9728

theorem rocket_heights (h1 : ℝ) (h2 : ℝ) (height1 : h1 = 500) (height2 : h2 = 2 * h1) :
  h1 + h2 = 1500 := by
  sorry

end rocket_heights_l97_9728


namespace shawn_red_pebbles_l97_9765

/-- The number of red pebbles in Shawn's collection -/
def red_pebbles (total blue yellow : ℕ) : ℕ :=
  total - (blue + 3 * yellow)

/-- Theorem stating the number of red pebbles Shawn painted -/
theorem shawn_red_pebbles :
  ∃ (yellow : ℕ),
    red_pebbles 40 13 yellow = 9 ∧
    13 - yellow = 7 :=
by sorry

end shawn_red_pebbles_l97_9765


namespace systematic_sampling_interval_72_8_l97_9799

/-- Calculates the interval between segments for systematic sampling -/
def systematicSamplingInterval (populationSize : ℕ) (sampleSize : ℕ) : ℕ :=
  populationSize / sampleSize

/-- Theorem: The systematic sampling interval for 72 students with a sample size of 8 is 9 -/
theorem systematic_sampling_interval_72_8 :
  systematicSamplingInterval 72 8 = 9 := by
  sorry

end systematic_sampling_interval_72_8_l97_9799


namespace students_in_school_l97_9725

theorem students_in_school (total_students : ℕ) (trip_fraction : ℚ) (home_fraction : ℚ) : 
  total_students = 1000 →
  trip_fraction = 1/2 →
  home_fraction = 1/2 →
  (total_students - (trip_fraction * total_students).floor - 
   (home_fraction * (total_students - (trip_fraction * total_students).floor)).floor) = 250 := by
  sorry

#check students_in_school

end students_in_school_l97_9725


namespace smallest_positive_solution_l97_9798

theorem smallest_positive_solution :
  ∀ x : ℝ, x > 0 ∧ Real.sqrt x = 9 * x^2 → x ≥ 1/81 :=
by sorry

end smallest_positive_solution_l97_9798


namespace equality_from_fraction_l97_9704

theorem equality_from_fraction (a b : ℝ) (n : ℕ+) 
  (h_pos_a : a > 0) (h_pos_b : b > 0)
  (h_eq : ((a + b)^(n : ℕ) - (a - b)^(n : ℕ)) / ((a + b)^(n : ℕ) + (a - b)^(n : ℕ)) = a / b) :
  a = b := by sorry

end equality_from_fraction_l97_9704


namespace angle_bisector_length_bound_l97_9751

theorem angle_bisector_length_bound (a b : ℝ) (θ : ℝ) (h1 : a = 10) (h2 : b = 15) (h3 : 0 < θ) (h4 : θ < π) :
  (2 * a * b * Real.cos (θ / 2)) / (a + b) < 12 := by
  sorry

end angle_bisector_length_bound_l97_9751


namespace triangle_properties_l97_9748

noncomputable section

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : Real
  B : Real
  C : Real

/-- The main theorem -/
theorem triangle_properties (t : Triangle) 
  (h1 : t.c * Real.sin t.B = Real.sqrt 3 * t.b * Real.cos t.C)
  (h2 : t.a^2 - t.c^2 = 2 * t.b^2)
  (h3 : (1/2) * t.a * t.b * Real.sin t.C = 21 * Real.sqrt 3) :
  t.C = π/3 ∧ t.b = 2 * Real.sqrt 7 := by
  sorry

end

end triangle_properties_l97_9748


namespace find_t_l97_9743

theorem find_t : ∃ t : ℤ, (∃ s : ℤ, 12 * s + 7 * t = 173 ∧ s = t - 3) → t = 11 := by
  sorry

end find_t_l97_9743


namespace solution_quadratic_equation_l97_9777

theorem solution_quadratic_equation : 
  ∀ x : ℝ, (x - 2) * (x + 3) = 0 ↔ x = 2 ∨ x = -3 := by sorry

end solution_quadratic_equation_l97_9777


namespace susan_chairs_l97_9752

theorem susan_chairs (total : ℕ) (red : ℕ) (yellow : ℕ) (blue : ℕ) : 
  total = 43 →
  red = 5 →
  yellow = 4 * red →
  total = red + yellow + blue →
  blue = 18 := by
sorry

end susan_chairs_l97_9752


namespace supplement_of_complement_of_35_degree_angle_l97_9790

/-- The degree measure of the supplement of the complement of a 35-degree angle is 125°. -/
theorem supplement_of_complement_of_35_degree_angle : 
  let angle : ℝ := 35
  let complement := 90 - angle
  let supplement := 180 - complement
  supplement = 125 := by sorry

end supplement_of_complement_of_35_degree_angle_l97_9790


namespace rahul_work_time_l97_9797

-- Define the work completion time for Rajesh
def rajesh_time : ℝ := 2

-- Define the total payment
def total_payment : ℝ := 170

-- Define Rahul's share
def rahul_share : ℝ := 68

-- Define Rahul's work completion time (to be proved)
def rahul_time : ℝ := 3

-- Theorem statement
theorem rahul_work_time :
  -- Given conditions
  (rajesh_time = 2) →
  (total_payment = 170) →
  (rahul_share = 68) →
  -- Proof goal
  (rahul_time = 3) := by
    sorry -- Proof is omitted as per instructions

end rahul_work_time_l97_9797


namespace cos_150_degrees_l97_9749

theorem cos_150_degrees : Real.cos (150 * π / 180) = -1 / 2 := by sorry

end cos_150_degrees_l97_9749


namespace parabola_segment_length_l97_9724

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the focus and directrix
def focus : ℝ × ℝ := (2, 0)
def directrix : ℝ → Prop := λ x => x = -2

-- Define a point on the directrix
def point_on_directrix (P : ℝ × ℝ) : Prop :=
  directrix P.1

-- Define points on the parabola
def point_on_parabola (M : ℝ × ℝ) : Prop :=
  parabola M.1 M.2

-- Define the condition PF = 3MF
def vector_condition (P M : ℝ × ℝ) : Prop :=
  (P.1 - focus.1)^2 + (P.2 - focus.2)^2 = 
  9 * ((M.1 - focus.1)^2 + (M.2 - focus.2)^2)

-- State the theorem
theorem parabola_segment_length 
  (P M N : ℝ × ℝ) 
  (h1 : point_on_directrix P) 
  (h2 : point_on_parabola M) 
  (h3 : point_on_parabola N) 
  (h4 : vector_condition P M) :
  let MN_length := Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2)
  MN_length = 32/3 := by sorry

end parabola_segment_length_l97_9724


namespace irreducible_fractions_count_l97_9762

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

theorem irreducible_fractions_count : 
  (∃! (count : ℕ), ∃ (S : Finset ℕ), 
    S.card = count ∧ 
    (∀ n ∈ S, (1 : ℚ) / 16 < (n : ℚ) / 15 ∧ (n : ℚ) / 15 < (1 : ℚ) / 15) ∧
    (∀ n ∈ S, is_coprime n 15) ∧
    (∀ n : ℕ, (1 : ℚ) / 16 < (n : ℚ) / 15 ∧ (n : ℚ) / 15 < (1 : ℚ) / 15 → 
      is_coprime n 15 → n ∈ S)) ∧
  count = 8 :=
sorry

end irreducible_fractions_count_l97_9762


namespace expression_simplification_l97_9712

theorem expression_simplification (x : ℝ) : (x + 1)^2 + 2*(x + 1)*(5 - x) + (5 - x)^2 = 36 := by
  sorry

end expression_simplification_l97_9712


namespace five_three_bar_equals_sixteen_thirds_l97_9775

/-- Represents a repeating decimal with an integer part and a repeating fractional part. -/
structure RepeatingDecimal where
  integerPart : ℤ
  repeatingPart : ℕ

/-- Converts a RepeatingDecimal to a rational number. -/
def repeatingDecimalToRational (x : RepeatingDecimal) : ℚ :=
  x.integerPart + (x.repeatingPart : ℚ) / 9

/-- The repeating decimal 5.3̄ -/
def five_three_bar : RepeatingDecimal :=
  { integerPart := 5, repeatingPart := 3 }

/-- Theorem: The repeating decimal 5.3̄ is equal to 16/3 -/
theorem five_three_bar_equals_sixteen_thirds :
  repeatingDecimalToRational five_three_bar = 16 / 3 := by
  sorry

end five_three_bar_equals_sixteen_thirds_l97_9775


namespace miser_knight_theorem_l97_9769

theorem miser_knight_theorem (n : ℕ) (h : n > 0) :
  (∀ k : ℕ, 2 ≤ k ∧ k ≤ 76 → ∃ m : ℕ, n = m * k) →
  ∃ m : ℕ, n = m * 77 :=
by sorry

end miser_knight_theorem_l97_9769


namespace triangle_perimeter_from_excircle_radii_l97_9773

theorem triangle_perimeter_from_excircle_radii (a b c : ℝ) (ra rb rc : ℝ) :
  ra = 3 ∧ rb = 10 ∧ rc = 15 →
  ra > 0 ∧ rb > 0 ∧ rc > 0 →
  a > 0 ∧ b > 0 ∧ c > 0 →
  (b + c - a) / 2 = ra ∧ (a + c - b) / 2 = rb ∧ (a + b - c) / 2 = rc →
  a + b + c = 30 := by
sorry

end triangle_perimeter_from_excircle_radii_l97_9773


namespace first_prize_tickets_characterization_l97_9735

def is_valid_ticket (n : Nat) : Prop := n ≥ 0 ∧ n ≤ 9999

def is_first_prize (n : Nat) : Prop := n % 1000 = 418

def first_prize_tickets : Set Nat :=
  {n : Nat | is_valid_ticket n ∧ is_first_prize n}

theorem first_prize_tickets_characterization :
  first_prize_tickets = {0418, 1418, 2418, 3418, 4418, 5418, 6418, 7418, 8418, 9418} :=
by sorry

end first_prize_tickets_characterization_l97_9735


namespace counterfeit_weight_equals_net_profit_l97_9772

/-- Represents a dishonest dealer's selling strategy -/
structure DishonestDealer where
  /-- The percentage of impurities added to the product -/
  impurities : ℝ
  /-- The net profit percentage achieved by the dealer -/
  net_profit : ℝ

/-- Calculates the percentage by which the counterfeit weight is less than the real weight -/
def counterfeit_weight_percentage (dealer : DishonestDealer) : ℝ :=
  dealer.net_profit

/-- Theorem stating that under specific conditions, the counterfeit weight percentage
    equals the net profit percentage -/
theorem counterfeit_weight_equals_net_profit 
  (dealer : DishonestDealer) 
  (h1 : dealer.impurities = 35)
  (h2 : dealer.net_profit = 68.75) :
  counterfeit_weight_percentage dealer = 68.75 := by
  sorry

end counterfeit_weight_equals_net_profit_l97_9772


namespace complex_exponential_sum_l97_9739

theorem complex_exponential_sum (θ φ : ℝ) :
  Complex.exp (Complex.I * θ) + Complex.exp (Complex.I * φ) = (1/3 : ℂ) + (2/5 : ℂ) * Complex.I →
  Complex.exp (-Complex.I * θ) + Complex.exp (-Complex.I * φ) = (1/3 : ℂ) - (2/5 : ℂ) * Complex.I := by
  sorry

end complex_exponential_sum_l97_9739


namespace train_length_l97_9723

/-- The length of a train given its speed, time to cross a bridge, and the bridge's length -/
theorem train_length (train_speed : ℝ) (crossing_time : ℝ) (bridge_length : ℝ) :
  train_speed = 45 * (1000 / 3600) →
  crossing_time = 30 →
  bridge_length = 225 →
  train_speed * crossing_time - bridge_length = 150 := by
  sorry

end train_length_l97_9723


namespace teacher_number_game_l97_9742

theorem teacher_number_game (x : ℤ) : 
  let ben_result := ((x + 3) * 2) - 2
  let sue_result := ((ben_result + 1) * 2) + 4
  sue_result = 2 * x + 30 := by
sorry

end teacher_number_game_l97_9742


namespace unused_sector_angle_l97_9750

/-- Given a cone with radius 10 cm and volume 250π cm³, 
    prove that the angle of the sector not used to form the cone is 72°. -/
theorem unused_sector_angle (r h : ℝ) (volume : ℝ) : 
  r = 10 → 
  volume = 250 * Real.pi → 
  (1/3) * Real.pi * r^2 * h = volume → 
  Real.sqrt (r^2 + h^2) = 12.5 → 
  360 - (360 * ((2 * Real.pi * r) / (2 * Real.pi * 12.5))) = 72 := by
  sorry

#check unused_sector_angle

end unused_sector_angle_l97_9750


namespace perimeter_area_bisector_coincide_l97_9763

/-- An isosceles triangle with side lengths 5, 5, and 6 -/
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  isIsosceles : a = b ∧ a = 5 ∧ c = 6

/-- A line bisecting the perimeter of the triangle -/
def perimeterBisector (t : IsoscelesTriangle) : Set (ℝ × ℝ) :=
  sorry

/-- A line bisecting the area of the triangle -/
def areaBisector (t : IsoscelesTriangle) : Set (ℝ × ℝ) :=
  sorry

/-- Theorem stating that the perimeter bisector coincides with the area bisector -/
theorem perimeter_area_bisector_coincide (t : IsoscelesTriangle) :
  perimeterBisector t = areaBisector t :=
sorry

end perimeter_area_bisector_coincide_l97_9763


namespace limit_point_theorem_l97_9787

def is_limit_point (X : Set ℝ) (x₀ : ℝ) : Prop :=
  ∀ a > 0, ∃ x ∈ X, 0 < |x - x₀| ∧ |x - x₀| < a

def set1 : Set ℝ := {x | ∃ n : ℤ, n ≥ 0 ∧ x = n / (n + 1)}
def set2 : Set ℝ := {x : ℝ | x ≠ 0}
def set3 : Set ℝ := {x | ∃ n : ℤ, n ≠ 0 ∧ x = 1 / n}
def set4 : Set ℝ := {x : ℝ | ∃ n : ℤ, x = n}

theorem limit_point_theorem :
  ¬(is_limit_point set1 0) ∧
  (is_limit_point set2 0) ∧
  (is_limit_point set3 0) ∧
  ¬(is_limit_point set4 0) := by sorry

end limit_point_theorem_l97_9787


namespace cubic_equation_solvable_l97_9737

theorem cubic_equation_solvable (a b c d : ℝ) (ha : a ≠ 0) :
  ∃ x : ℝ, a * x^3 + b * x^2 + c * x + d = 0 ∧
  (∃ (e f g h i j k l m n : ℝ),
    x = (e * (f^(1/3)) + g * (h^(1/3)) + i * (j^(1/3)) + k) /
        (l * (m^(1/2)) + n)) := by
  sorry

end cubic_equation_solvable_l97_9737


namespace triangle_side_length_l97_9726

/-- In a triangle ABC, given angle A, angle B, and side AC, prove the length of AB --/
theorem triangle_side_length (A B C : Real) (angleA angleB : Real) (sideAC : Real) :
  -- Conditions
  angleA = 105 * Real.pi / 180 →
  angleB = 45 * Real.pi / 180 →
  sideAC = 2 →
  -- Triangle angle sum property
  angleA + angleB + C = Real.pi →
  -- Sine rule
  sideAC / Real.sin angleB = A / Real.sin C →
  -- Conclusion
  A = Real.sqrt 2 := by
  sorry

end triangle_side_length_l97_9726


namespace function_equality_l97_9711

theorem function_equality (f g : ℕ → ℕ) 
  (h_surj : Function.Surjective f)
  (h_inj : Function.Injective g)
  (h_ge : ∀ n : ℕ, f n ≥ g n) :
  ∀ n : ℕ, f n = g n := by
  sorry

end function_equality_l97_9711


namespace remainder_problem_l97_9717

theorem remainder_problem (n : ℤ) (h : n % 6 = 1) : (3 * (n + 1812)) % 6 = 3 := by
  sorry

end remainder_problem_l97_9717


namespace no_real_solution_for_A_only_l97_9721

theorem no_real_solution_for_A_only : 
  (¬ ∃ x : ℝ, (x - 3)^2 = -1) ∧ 
  (∃ x : ℝ, |x/2| - 6 = 0) ∧ 
  (∃ x : ℝ, x^2 + 8*x + 16 = 0) ∧ 
  (∃ x : ℝ, x + Real.sqrt (x - 5) = 0) ∧ 
  (∃ x : ℝ, Real.sqrt (-2*x - 10) = 3) :=
by sorry

end no_real_solution_for_A_only_l97_9721


namespace inverse_proportionality_l97_9700

theorem inverse_proportionality (x y : ℝ) (k : ℝ) (h1 : x * y = k) (h2 : 5 * (-10) = k) :
  x = 10 / 3 → y = -15 := by
  sorry

end inverse_proportionality_l97_9700


namespace store_pricing_l97_9729

/-- Represents the price structure of a store selling chairs, tables, and shelves. -/
structure StorePrice where
  chair : ℝ
  table : ℝ
  shelf : ℝ

/-- Defines the properties of the store's pricing and discount policy. -/
def ValidStorePrice (p : StorePrice) : Prop :=
  p.chair + p.table = 72 ∧
  2 * p.chair + p.table = 0.6 * (p.chair + 2 * p.table) ∧
  p.chair + p.table + p.shelf = 95

/-- Calculates the discounted price for a combination of items. -/
def DiscountedPrice (p : StorePrice) : ℝ :=
  0.9 * (p.chair + 2 * p.table + p.shelf)

/-- Theorem stating the correct prices for the store items and the discounted combination. -/
theorem store_pricing (p : StorePrice) (h : ValidStorePrice p) :
  p.table = 63 ∧ DiscountedPrice p = 142.2 := by
  sorry

end store_pricing_l97_9729


namespace intersection_of_three_lines_l97_9719

theorem intersection_of_three_lines (k : ℚ) : 
  (∃ (x y : ℚ), y = 6*x + 5 ∧ y = -3*x - 30 ∧ y = 4*x + k) → 
  k = -25/9 := by
sorry

end intersection_of_three_lines_l97_9719


namespace symmetry_of_lines_l97_9788

-- Define the line l₁
def l₁ (x y : ℝ) : Prop := 2 * x - y + 1 = 0

-- Define the line of symmetry
def line_of_symmetry (x y : ℝ) : Prop := y = -x

-- Define the symmetric line l₂
def l₂ (x y : ℝ) : Prop := x - 2 * y + 1 = 0

-- Theorem statement
theorem symmetry_of_lines :
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
    l₁ x₁ y₁ →
    line_of_symmetry ((x₁ + x₂) / 2) ((y₁ + y₂) / 2) →
    (x₂ = 2 * ((x₁ + x₂) / 2) - x₁ ∧ y₂ = 2 * ((y₁ + y₂) / 2) - y₁) →
    l₂ x₂ y₂ :=
sorry

end symmetry_of_lines_l97_9788


namespace digit_2023_of_7_18_l97_9732

/-- The 2023rd digit past the decimal point in the decimal expansion of 7/18 is 3 -/
theorem digit_2023_of_7_18 : ∃ (d : ℕ), d = 3 ∧ 
  (∃ (a b : ℕ+) (s : Finset ℕ), 
    (7 : ℚ) / 18 = (a : ℚ) / b ∧ 
    s.card = 2023 ∧ 
    (∀ n ∈ s, (10 ^ n * ((7 : ℚ) / 18) % 1).floor % 10 = d) ∧
    (∀ m < 2023, m ∉ s)) :=
by sorry

end digit_2023_of_7_18_l97_9732


namespace lauryn_company_employees_l97_9786

/-- The number of men working for Lauryn's company -/
def num_men : ℕ := 80

/-- The difference between the number of women and men -/
def women_men_diff : ℕ := 20

/-- The total number of people working for Lauryn's company -/
def total_employees : ℕ := num_men + (num_men + women_men_diff)

theorem lauryn_company_employees :
  total_employees = 180 :=
by sorry

end lauryn_company_employees_l97_9786


namespace combination_equation_solution_l97_9709

theorem combination_equation_solution (n : ℕ) : n ≥ 2 → (Nat.choose n 2 = Nat.choose (n - 1) 2 + Nat.choose (n - 1) 3) → n = 5 := by
  sorry

end combination_equation_solution_l97_9709


namespace grandpa_to_uncle_ratio_l97_9771

/-- Represents the number of toy cars in various scenarios --/
structure ToyCars where
  initial : ℕ
  final : ℕ
  fromDad : ℕ
  fromMum : ℕ
  fromAuntie : ℕ
  fromUncle : ℕ
  fromGrandpa : ℕ

/-- Theorem stating the ratio of Grandpa's gift to Uncle's gift --/
theorem grandpa_to_uncle_ratio (cars : ToyCars)
  (h1 : cars.initial = 150)
  (h2 : cars.final = 196)
  (h3 : cars.fromDad = 10)
  (h4 : cars.fromMum = cars.fromDad + 5)
  (h5 : cars.fromAuntie = 6)
  (h6 : cars.fromUncle = cars.fromAuntie - 1)
  (h7 : cars.final = cars.initial + cars.fromDad + cars.fromMum + cars.fromAuntie + cars.fromUncle + cars.fromGrandpa) :
  cars.fromGrandpa = 2 * cars.fromUncle := by
  sorry

#check grandpa_to_uncle_ratio

end grandpa_to_uncle_ratio_l97_9771
