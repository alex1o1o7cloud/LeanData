import Mathlib

namespace NUMINAMATH_CALUDE_z₁z₂_value_a_value_when_sum_real_l2378_237819

-- Define complex numbers z₁ and z₂
def z₁ (a : ℝ) : ℂ := 2 + a * Complex.I
def z₂ : ℂ := 3 - 4 * Complex.I

-- Theorem 1: When a = 1, z₁z₂ = 10 - 5i
theorem z₁z₂_value : z₁ 1 * z₂ = 10 - 5 * Complex.I := by sorry

-- Theorem 2: When z₁ + z₂ is a real number, a = 4
theorem a_value_when_sum_real : 
  ∃ (a : ℝ), (z₁ a + z₂).im = 0 → a = 4 := by sorry

end NUMINAMATH_CALUDE_z₁z₂_value_a_value_when_sum_real_l2378_237819


namespace NUMINAMATH_CALUDE_range_of_a_for_false_proposition_l2378_237833

theorem range_of_a_for_false_proposition :
  (¬ ∃ x : ℝ, x^2 - a*x + 1 ≤ 0) ↔ -2 < a ∧ a < 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_for_false_proposition_l2378_237833


namespace NUMINAMATH_CALUDE_system_no_solution_l2378_237844

-- Define the system of equations
def system (n : ℝ) (x y z : ℝ) : Prop :=
  2*n*x + 3*y = 2 ∧ 3*n*y + 4*z = 3 ∧ 4*x + 2*n*z = 4

-- Theorem statement
theorem system_no_solution (n : ℝ) :
  (∀ x y z : ℝ, ¬ system n x y z) ↔ n = -Real.rpow 4 (1/3) :=
sorry

end NUMINAMATH_CALUDE_system_no_solution_l2378_237844


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2378_237824

theorem quadratic_inequality_solution_set :
  {x : ℝ | -x^2 + 2*x + 3 > 0} = Set.Ioo (-1 : ℝ) 3 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2378_237824


namespace NUMINAMATH_CALUDE_rain_probability_tel_aviv_l2378_237851

def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

theorem rain_probability_tel_aviv :
  let n : ℕ := 6
  let k : ℕ := 4
  let p : ℝ := 0.5
  binomial_probability n k p = 0.234375 := by
sorry

end NUMINAMATH_CALUDE_rain_probability_tel_aviv_l2378_237851


namespace NUMINAMATH_CALUDE_exists_special_function_l2378_237874

theorem exists_special_function : ∃ (s : ℚ → Int), 
  (∀ x, s x = 1 ∨ s x = -1) ∧ 
  (∀ x y, x ≠ y → (x * y = 1 ∨ x + y = 0 ∨ x + y = 1) → s x * s y = -1) := by
  sorry

end NUMINAMATH_CALUDE_exists_special_function_l2378_237874


namespace NUMINAMATH_CALUDE_tiffany_bag_difference_l2378_237807

/-- The number of bags Tiffany had on Monday -/
def monday_bags : ℕ := 8

/-- The number of bags Tiffany found the next day -/
def next_day_bags : ℕ := 7

/-- The difference between the number of bags on Monday and the next day -/
def bag_difference : ℕ := monday_bags - next_day_bags

theorem tiffany_bag_difference : bag_difference = 1 := by
  sorry

end NUMINAMATH_CALUDE_tiffany_bag_difference_l2378_237807


namespace NUMINAMATH_CALUDE_tan_105_degrees_l2378_237893

theorem tan_105_degrees : Real.tan (105 * π / 180) = -2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_105_degrees_l2378_237893


namespace NUMINAMATH_CALUDE_common_point_implies_c_equals_d_l2378_237806

/-- Given three linear functions with a common point, prove that c = d -/
theorem common_point_implies_c_equals_d 
  (a b c d : ℝ) 
  (h_neq : a ≠ b) 
  (h_common : ∃ (x y : ℝ), 
    y = a * x + a ∧ 
    y = b * x + b ∧ 
    y = c * x + d) : 
  c = d := by
sorry


end NUMINAMATH_CALUDE_common_point_implies_c_equals_d_l2378_237806


namespace NUMINAMATH_CALUDE_quadrilateral_perimeter_l2378_237888

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the perimeter function
def perimeter (q : Quadrilateral) : ℝ :=
  sorry

-- Define the perpendicular property
def perpendicular (v w : ℝ × ℝ) : Prop :=
  sorry

-- Theorem statement
theorem quadrilateral_perimeter :
  ∀ (q : Quadrilateral),
    perpendicular (q.B - q.A) (q.C - q.B) →
    perpendicular (q.C - q.D) (q.C - q.B) →
    ‖q.B - q.A‖ = 9 →
    ‖q.D - q.C‖ = 4 →
    ‖q.C - q.B‖ = 12 →
    perimeter q = 38 :=
by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_perimeter_l2378_237888


namespace NUMINAMATH_CALUDE_product_of_odd_primes_below_16_mod_32_l2378_237895

def odd_primes_below_16 : List Nat := [3, 5, 7, 11, 13]

theorem product_of_odd_primes_below_16_mod_32 :
  (List.prod odd_primes_below_16) % (2^5) = 7 := by
  sorry

end NUMINAMATH_CALUDE_product_of_odd_primes_below_16_mod_32_l2378_237895


namespace NUMINAMATH_CALUDE_factor_x_10_minus_1024_l2378_237827

theorem factor_x_10_minus_1024 (x : ℝ) :
  x^10 - 1024 = (x^5 + 32) * (x - 2) * (x^4 + 2*x^3 + 4*x^2 + 8*x + 16) := by
  sorry

end NUMINAMATH_CALUDE_factor_x_10_minus_1024_l2378_237827


namespace NUMINAMATH_CALUDE_min_value_a_l2378_237877

theorem min_value_a (a : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 1 3 ∧ x^2 - 2 ≤ a) → 
  (∀ b : ℝ, (∃ x : ℝ, x ∈ Set.Icc 1 3 ∧ x^2 - 2 ≤ b) → a ≤ b) → 
  a = -1 := by
sorry

end NUMINAMATH_CALUDE_min_value_a_l2378_237877


namespace NUMINAMATH_CALUDE_olympic_mascot_problem_l2378_237876

/-- Olympic Mascot Problem -/
theorem olympic_mascot_problem (m : ℝ) : 
  -- Conditions
  (3000 / m = 2400 / (m - 30)) →
  -- Definitions
  let bing_price := m
  let shuey_price := m - 30
  let bing_sell := 190
  let shuey_sell := 140
  let total_mascots := 200
  let profit (x : ℝ) := (bing_sell - bing_price) * x + (shuey_sell - shuey_price) * (total_mascots - x)
  -- Theorem statements
  (m = 150 ∧ 
   ∀ x : ℝ, 0 ≤ x ∧ x ≤ total_mascots ∧ (total_mascots - x ≥ (2/3) * x) →
     profit x ≤ profit 120) := by sorry

end NUMINAMATH_CALUDE_olympic_mascot_problem_l2378_237876


namespace NUMINAMATH_CALUDE_smallest_prime_after_four_nonprimes_l2378_237890

/-- A function that checks if a natural number is prime -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- A function that checks if four consecutive natural numbers are all nonprime -/
def fourConsecutiveNonPrime (n : ℕ) : Prop :=
  ¬(isPrime n) ∧ ¬(isPrime (n + 1)) ∧ ¬(isPrime (n + 2)) ∧ ¬(isPrime (n + 3))

/-- The theorem stating that 29 is the smallest prime after four consecutive nonprimes -/
theorem smallest_prime_after_four_nonprimes :
  ∃ n : ℕ, fourConsecutiveNonPrime n ∧ isPrime (n + 4) ∧
  ∀ m : ℕ, m < n → ¬(fourConsecutiveNonPrime m ∧ isPrime (m + 4)) :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_after_four_nonprimes_l2378_237890


namespace NUMINAMATH_CALUDE_prob_same_number_four_dice_l2378_237845

/-- The number of sides on a standard die -/
def standard_die_sides : ℕ := 6

/-- The number of dice being tossed -/
def num_dice : ℕ := 4

/-- The probability of getting the same number on all dice -/
def prob_same_number : ℚ := 1 / (standard_die_sides ^ (num_dice - 1))

/-- Theorem: The probability of getting the same number on all four standard six-sided dice is 1/216 -/
theorem prob_same_number_four_dice : 
  prob_same_number = 1 / 216 := by sorry

end NUMINAMATH_CALUDE_prob_same_number_four_dice_l2378_237845


namespace NUMINAMATH_CALUDE_toms_dimes_calculation_l2378_237837

/-- Calculates the final number of dimes Tom has after receiving and spending some. -/
def final_dimes (initial : ℕ) (received : ℕ) (spent : ℕ) : ℕ :=
  initial + received - spent

/-- Proves that Tom's final number of dimes is correct given the initial amount, 
    the amount received, and the amount spent. -/
theorem toms_dimes_calculation (initial : ℕ) (received : ℕ) (spent : ℕ) :
  final_dimes initial received spent = initial + received - spent :=
by sorry

end NUMINAMATH_CALUDE_toms_dimes_calculation_l2378_237837


namespace NUMINAMATH_CALUDE_geoff_total_spending_l2378_237879

/-- Geoff's spending on sneakers over three days -/
def sneaker_spending (monday_spend : ℝ) : ℝ :=
  let tuesday_spend := 4 * monday_spend
  let wednesday_spend := 5 * monday_spend
  monday_spend + tuesday_spend + wednesday_spend

/-- Theorem stating that Geoff's total spending over three days is $600 -/
theorem geoff_total_spending :
  sneaker_spending 60 = 600 := by
  sorry

end NUMINAMATH_CALUDE_geoff_total_spending_l2378_237879


namespace NUMINAMATH_CALUDE_rhombus_area_l2378_237805

/-- The area of a rhombus with vertices at (0, 3.5), (7, 0), (0, -3.5), and (-7, 0) is 49 square units. -/
theorem rhombus_area : 
  let v1 : ℝ × ℝ := (0, 3.5)
  let v2 : ℝ × ℝ := (7, 0)
  let v3 : ℝ × ℝ := (0, -3.5)
  let v4 : ℝ × ℝ := (-7, 0)
  let d1 : ℝ := v1.2 - v3.2  -- Vertical diagonal
  let d2 : ℝ := v2.1 - v4.1  -- Horizontal diagonal
  let area : ℝ := (d1 * d2) / 2
  area = 49 := by sorry

end NUMINAMATH_CALUDE_rhombus_area_l2378_237805


namespace NUMINAMATH_CALUDE_investment_difference_l2378_237866

def initial_investment : ℕ := 2000

def alice_multiplier : ℕ := 2
def bob_multiplier : ℕ := 5

def alice_final (initial : ℕ) : ℕ := initial * alice_multiplier
def bob_final (initial : ℕ) : ℕ := initial * bob_multiplier

theorem investment_difference : 
  bob_final initial_investment - alice_final initial_investment = 6000 := by
  sorry

end NUMINAMATH_CALUDE_investment_difference_l2378_237866


namespace NUMINAMATH_CALUDE_product_pure_imaginary_implies_a_eq_neg_one_l2378_237862

/-- Given complex numbers z₁ and z₂, prove that if z₁ · z₂ is purely imaginary, then a = -1 -/
theorem product_pure_imaginary_implies_a_eq_neg_one (a : ℝ) :
  let z₁ : ℂ := a - Complex.I
  let z₂ : ℂ := 1 + Complex.I
  (∃ (b : ℝ), z₁ * z₂ = b * Complex.I) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_product_pure_imaginary_implies_a_eq_neg_one_l2378_237862


namespace NUMINAMATH_CALUDE_sqrt_nine_factorial_over_108_l2378_237886

theorem sqrt_nine_factorial_over_108 : 
  Real.sqrt (Nat.factorial 9 / 108) = 8 * Real.sqrt 35 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_nine_factorial_over_108_l2378_237886


namespace NUMINAMATH_CALUDE_bottle_caps_eaten_l2378_237830

theorem bottle_caps_eaten (initial : ℕ) (remaining : ℕ) (eaten : ℕ) : 
  initial = 34 → remaining = 26 → eaten = initial - remaining → eaten = 8 := by
sorry

end NUMINAMATH_CALUDE_bottle_caps_eaten_l2378_237830


namespace NUMINAMATH_CALUDE_unique_solution_cubic_system_l2378_237891

theorem unique_solution_cubic_system :
  ∃! (x y z : ℝ),
    x = y^3 + y - 8 ∧
    y = z^3 + z - 8 ∧
    z = x^3 + x - 8 ∧
    x = 2 ∧ y = 2 ∧ z = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_cubic_system_l2378_237891


namespace NUMINAMATH_CALUDE_four_positive_integers_sum_l2378_237823

theorem four_positive_integers_sum (a b c d : ℕ+) 
  (sum1 : a + b + c = 6)
  (sum2 : a + b + d = 7)
  (sum3 : a + c + d = 8)
  (sum4 : b + c + d = 9) :
  a = 1 ∧ b = 2 ∧ c = 3 ∧ d = 4 := by
  sorry

end NUMINAMATH_CALUDE_four_positive_integers_sum_l2378_237823


namespace NUMINAMATH_CALUDE_percentage_difference_l2378_237881

theorem percentage_difference (x : ℝ) : 
  (60 / 100 * 50 = 30) →
  (30 = x / 100 * 30 + 17.4) →
  x = 42 := by
sorry

end NUMINAMATH_CALUDE_percentage_difference_l2378_237881


namespace NUMINAMATH_CALUDE_pear_mango_weight_equivalence_l2378_237899

/-- Given that 9 pears weigh the same as 6 mangoes, 
    prove that 36 pears weigh the same as 24 mangoes. -/
theorem pear_mango_weight_equivalence 
  (pear_weight mango_weight : ℝ) 
  (h : 9 * pear_weight = 6 * mango_weight) : 
  36 * pear_weight = 24 * mango_weight := by
  sorry

end NUMINAMATH_CALUDE_pear_mango_weight_equivalence_l2378_237899


namespace NUMINAMATH_CALUDE_dodecahedron_interior_diagonals_l2378_237849

/-- A dodecahedron is a 3-dimensional figure with 20 vertices -/
structure Dodecahedron where
  vertices : Finset ℕ
  vertex_count : vertices.card = 20

/-- Each vertex in a dodecahedron is connected to 3 other vertices by edges -/
def connected_vertices (d : Dodecahedron) (v : ℕ) : Finset ℕ :=
  sorry

axiom connected_vertices_count (d : Dodecahedron) (v : ℕ) (h : v ∈ d.vertices) :
  (connected_vertices d v).card = 3

/-- An interior diagonal is a segment connecting two vertices which do not share an edge -/
def interior_diagonals (d : Dodecahedron) : Finset (ℕ × ℕ) :=
  sorry

/-- The main theorem: a dodecahedron has 160 interior diagonals -/
theorem dodecahedron_interior_diagonals (d : Dodecahedron) :
  (interior_diagonals d).card = 160 :=
sorry

end NUMINAMATH_CALUDE_dodecahedron_interior_diagonals_l2378_237849


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l2378_237865

theorem complex_magnitude_problem (z : ℂ) (h : (Complex.I / (1 + Complex.I)) * z = 1) :
  Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l2378_237865


namespace NUMINAMATH_CALUDE_add_9876_seconds_to_2_45_pm_l2378_237838

/-- Represents a time of day in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat
  deriving Repr

/-- Converts seconds to a Time structure -/
def secondsToTime (totalSeconds : Nat) : Time :=
  let hours := totalSeconds / 3600
  let remainingSeconds := totalSeconds % 3600
  let minutes := remainingSeconds / 60
  let seconds := remainingSeconds % 60
  { hours := hours, minutes := minutes, seconds := seconds }

/-- Adds two Time structures -/
def addTime (t1 t2 : Time) : Time :=
  let totalSeconds := t1.hours * 3600 + t1.minutes * 60 + t1.seconds +
                      t2.hours * 3600 + t2.minutes * 60 + t2.seconds
  secondsToTime totalSeconds

/-- The main theorem to prove -/
theorem add_9876_seconds_to_2_45_pm (startTime : Time) 
  (h1 : startTime.hours = 14) 
  (h2 : startTime.minutes = 45) 
  (h3 : startTime.seconds = 0) : 
  addTime startTime (secondsToTime 9876) = { hours := 17, minutes := 29, seconds := 36 } := by
  sorry

end NUMINAMATH_CALUDE_add_9876_seconds_to_2_45_pm_l2378_237838


namespace NUMINAMATH_CALUDE_range_of_a_l2378_237878

-- Define the condition that x > 2 is sufficient but not necessary for x^2 > a
def sufficient_not_necessary (a : ℝ) : Prop :=
  (∀ x : ℝ, x > 2 → x^2 > a) ∧ 
  ¬(∀ x : ℝ, x^2 > a → x > 2)

-- Theorem statement
theorem range_of_a (a : ℝ) :
  sufficient_not_necessary a ↔ a ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2378_237878


namespace NUMINAMATH_CALUDE_lcm_of_150_and_490_l2378_237809

theorem lcm_of_150_and_490 : Nat.lcm 150 490 = 7350 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_150_and_490_l2378_237809


namespace NUMINAMATH_CALUDE_square_plus_reciprocal_square_l2378_237803

theorem square_plus_reciprocal_square (a : ℝ) (h : a + 1/a = 3) : a^2 + 1/a^2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_reciprocal_square_l2378_237803


namespace NUMINAMATH_CALUDE_expression_has_four_terms_l2378_237822

/-- The expression with the asterisk replaced by a monomial -/
def expression (x : ℝ) : ℝ := (x^4 - 3)^2 + (x^3 + 3*x)^2

/-- The result after expanding and combining like terms -/
def expanded_result (x : ℝ) : ℝ := x^8 + x^6 + 9*x^2 + 9

/-- Theorem stating that the expanded result has exactly four terms -/
theorem expression_has_four_terms :
  ∃ (a b c d : ℝ → ℝ),
    (∀ x, expanded_result x = a x + b x + c x + d x) ∧
    (∀ x, a x ≠ 0 ∧ b x ≠ 0 ∧ c x ≠ 0 ∧ d x ≠ 0) ∧
    (∀ x, expression x = expanded_result x) :=
sorry

end NUMINAMATH_CALUDE_expression_has_four_terms_l2378_237822


namespace NUMINAMATH_CALUDE_max_potential_salary_is_440000_l2378_237817

/-- Represents a soccer team with its payroll constraints -/
structure SoccerTeam where
  numPlayers : ℕ
  minSalary : ℕ
  maxPayroll : ℕ

/-- Calculates the maximum potential salary for an individual player on a team -/
def maxPotentialSalary (team : SoccerTeam) : ℕ :=
  team.maxPayroll - (team.numPlayers - 1) * team.minSalary

/-- Theorem stating the maximum potential salary for an individual player -/
theorem max_potential_salary_is_440000 :
  let team : SoccerTeam := ⟨19, 20000, 800000⟩
  maxPotentialSalary team = 440000 := by
  sorry

#eval maxPotentialSalary ⟨19, 20000, 800000⟩

end NUMINAMATH_CALUDE_max_potential_salary_is_440000_l2378_237817


namespace NUMINAMATH_CALUDE_min_value_hyperbola_ellipse_foci_l2378_237832

/-- The minimum value of (4/m + 1/n) given the conditions of the problem -/
theorem min_value_hyperbola_ellipse_foci (m n : ℝ) : 
  m > 0 → n > 0 → 
  (∃ x y : ℝ, x^2/m - y^2/n = 1) → 
  (∃ x y : ℝ, x^2/5 + y^2/2 = 1) → 
  (∀ x y : ℝ, x^2/m - y^2/n = 1 ↔ x^2/5 + y^2/2 = 1) → 
  (4/m + 1/n ≥ 3 ∧ ∃ m₀ n₀ : ℝ, m₀ > 0 ∧ n₀ > 0 ∧ 4/m₀ + 1/n₀ = 3) :=
sorry

end NUMINAMATH_CALUDE_min_value_hyperbola_ellipse_foci_l2378_237832


namespace NUMINAMATH_CALUDE_rationalize_denominator_l2378_237816

-- Define the original expression
def original_expr := (4 : ℚ) / (3 * (7 : ℚ)^(1/3))

-- Define the rationalized expression
def rationalized_expr := (4 * (49 : ℚ)^(1/3)) / 21

-- Define the property that 49 is not divisible by the cube of any prime
def not_cube_divisible (n : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → ¬(p^3 ∣ n)

-- Theorem statement
theorem rationalize_denominator :
  original_expr = rationalized_expr ∧ not_cube_divisible 49 := by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l2378_237816


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2378_237863

theorem inequality_solution_set (a : ℝ) : 
  (∀ x : ℝ, |x - 3| + |x - 4| ≥ a) ↔ a ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2378_237863


namespace NUMINAMATH_CALUDE_adam_shelf_count_l2378_237818

/-- The number of shelves in Adam's room -/
def num_shelves : ℕ := sorry

/-- The number of action figures that can fit on each shelf -/
def figures_per_shelf : ℕ := 9

/-- The total number of action figures that can be held by all shelves -/
def total_figures : ℕ := 27

/-- Theorem stating that the number of shelves is 3 -/
theorem adam_shelf_count : num_shelves = 3 := by sorry

end NUMINAMATH_CALUDE_adam_shelf_count_l2378_237818


namespace NUMINAMATH_CALUDE_sum_real_imag_parts_3_minus_4i_l2378_237842

theorem sum_real_imag_parts_3_minus_4i :
  let z : ℂ := 3 - 4*I
  (z.re + z.im : ℝ) = -1 := by sorry

end NUMINAMATH_CALUDE_sum_real_imag_parts_3_minus_4i_l2378_237842


namespace NUMINAMATH_CALUDE_soda_price_calculation_soda_price_proof_l2378_237815

/-- Given the cost of sandwiches and total cost, calculate the price of each soda -/
theorem soda_price_calculation (sandwich_price : ℚ) (num_sandwiches : ℕ) (num_sodas : ℕ) (total_cost : ℚ) : ℚ :=
  let sandwich_total := sandwich_price * num_sandwiches
  let soda_total := total_cost - sandwich_total
  soda_total / num_sodas

/-- Prove that the price of each soda is $1.87 given the problem conditions -/
theorem soda_price_proof :
  soda_price_calculation 2.49 2 4 12.46 = 1.87 := by
  sorry

end NUMINAMATH_CALUDE_soda_price_calculation_soda_price_proof_l2378_237815


namespace NUMINAMATH_CALUDE_characteristic_equation_of_A_l2378_237800

def A : Matrix (Fin 3) (Fin 3) ℝ := !![1, 2, 3; 3, 1, 2; 2, 3, 1]

theorem characteristic_equation_of_A :
  ∃ (p q r : ℝ), A^3 + p • A^2 + q • A + r • (1 : Matrix (Fin 3) (Fin 3) ℝ) = 0 ∧
  p = -3 ∧ q = -9 ∧ r = -2 := by
  sorry

end NUMINAMATH_CALUDE_characteristic_equation_of_A_l2378_237800


namespace NUMINAMATH_CALUDE_friend_walking_problem_l2378_237896

/-- 
Given two friends walking towards each other on a trail:
- The trail length is 33 km
- They start at opposite ends at the same time
- One friend's speed is 20% faster than the other's
Prove that the faster friend will have walked 18 km when they meet.
-/
theorem friend_walking_problem (v : ℝ) (h_v_pos : v > 0) :
  let trail_length : ℝ := 33
  let speed_ratio : ℝ := 1.2
  let t : ℝ := trail_length / (v * (1 + speed_ratio))
  speed_ratio * v * t = 18 := by sorry

end NUMINAMATH_CALUDE_friend_walking_problem_l2378_237896


namespace NUMINAMATH_CALUDE_probability_at_least_one_from_A_l2378_237894

/-- Represents the number of classes in each school -/
structure SchoolClasses where
  A : ℕ
  B : ℕ
  C : ℕ

/-- Represents the number of classes sampled from each school -/
structure SampledClasses where
  A : ℕ
  B : ℕ
  C : ℕ

/-- The total number of classes to be sampled -/
def totalSampled : ℕ := 6

/-- The number of classes to be randomly selected for comparison -/
def comparisonClasses : ℕ := 2

/-- Calculate the probability of selecting at least one class from school A 
    when randomly choosing 2 out of 6 sampled classes -/
def probabilityAtLeastOneFromA (classes : SchoolClasses) (sampled : SampledClasses) : ℚ :=
  sorry

/-- Theorem stating the probability is 3/5 given the specific conditions -/
theorem probability_at_least_one_from_A : 
  let classes : SchoolClasses := ⟨12, 6, 18⟩
  let sampled : SampledClasses := ⟨2, 1, 3⟩
  probabilityAtLeastOneFromA classes sampled = 3/5 :=
sorry

end NUMINAMATH_CALUDE_probability_at_least_one_from_A_l2378_237894


namespace NUMINAMATH_CALUDE_sum_of_intersection_coordinates_l2378_237808

-- Define the two parabolas
def parabola1 (x y : ℝ) : Prop := y = (x - 1)^2
def parabola2 (x y : ℝ) : Prop := x - 6 = (y + 1)^2

-- Define the intersection points
def intersection_points : Prop :=
  ∃ x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ,
    (parabola1 x₁ y₁ ∧ parabola2 x₁ y₁) ∧
    (parabola1 x₂ y₂ ∧ parabola2 x₂ y₂) ∧
    (parabola1 x₃ y₃ ∧ parabola2 x₃ y₃) ∧
    (parabola1 x₄ y₄ ∧ parabola2 x₄ y₄)

-- Theorem statement
theorem sum_of_intersection_coordinates :
  intersection_points →
  ∃ x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ,
    (parabola1 x₁ y₁ ∧ parabola2 x₁ y₁) ∧
    (parabola1 x₂ y₂ ∧ parabola2 x₂ y₂) ∧
    (parabola1 x₃ y₃ ∧ parabola2 x₃ y₃) ∧
    (parabola1 x₄ y₄ ∧ parabola2 x₄ y₄) ∧
    x₁ + x₂ + x₃ + x₄ + y₁ + y₂ + y₃ + y₄ = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_intersection_coordinates_l2378_237808


namespace NUMINAMATH_CALUDE_squares_below_line_l2378_237869

/-- Represents a line in 2D space --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Counts the number of integer squares below a line in the first quadrant --/
def countSquaresBelowLine (l : Line) : ℕ :=
  sorry

/-- The specific line from the problem --/
def problemLine : Line := { a := 5, b := 152, c := 1520 }

/-- The theorem to be proved --/
theorem squares_below_line : countSquaresBelowLine problemLine = 1363 := by
  sorry

end NUMINAMATH_CALUDE_squares_below_line_l2378_237869


namespace NUMINAMATH_CALUDE_point_on_translated_line_l2378_237887

/-- The original line -/
def original_line (x : ℝ) : ℝ := x

/-- The translated line -/
def translated_line (x : ℝ) : ℝ := x + 2

/-- Theorem stating that (2, 4) lies on the translated line -/
theorem point_on_translated_line : translated_line 2 = 4 := by sorry

end NUMINAMATH_CALUDE_point_on_translated_line_l2378_237887


namespace NUMINAMATH_CALUDE_jacobs_gift_budget_l2378_237892

theorem jacobs_gift_budget (total_budget : ℕ) (num_friends : ℕ) (friend_gift_cost : ℕ) (num_parents : ℕ) :
  total_budget = 100 →
  num_friends = 8 →
  friend_gift_cost = 9 →
  num_parents = 2 →
  (total_budget - num_friends * friend_gift_cost) / num_parents = 14 :=
by sorry

end NUMINAMATH_CALUDE_jacobs_gift_budget_l2378_237892


namespace NUMINAMATH_CALUDE_fourth_person_height_l2378_237814

/-- Proves that the height of the 4th person is 85 inches given the conditions of the problem -/
theorem fourth_person_height :
  ∀ (h₁ h₂ h₃ h₄ : ℝ),
    h₁ < h₂ ∧ h₂ < h₃ ∧ h₃ < h₄ →  -- Heights are in increasing order
    h₂ - h₁ = 2 →  -- Difference between 1st and 2nd person
    h₃ - h₂ = 2 →  -- Difference between 2nd and 3rd person
    h₄ - h₃ = 6 →  -- Difference between 3rd and 4th person
    (h₁ + h₂ + h₃ + h₄) / 4 = 79 →  -- Average height
    h₄ = 85 := by
  sorry

end NUMINAMATH_CALUDE_fourth_person_height_l2378_237814


namespace NUMINAMATH_CALUDE_divisibility_of_expression_l2378_237873

theorem divisibility_of_expression (a b : ℤ) : 
  ∃ k : ℤ, (2*a + 3)^2 - (2*b + 1)^2 = 8 * k := by
sorry

end NUMINAMATH_CALUDE_divisibility_of_expression_l2378_237873


namespace NUMINAMATH_CALUDE_initial_deposit_calculation_l2378_237871

-- Define the initial deposit
variable (P : ℝ)

-- Define the interest rates
def first_year_rate : ℝ := 0.20
def second_year_rate : ℝ := 0.15

-- Define the final amount
def final_amount : ℝ := 690

-- Theorem statement
theorem initial_deposit_calculation :
  (P * (1 + first_year_rate) / 2) * (1 + second_year_rate) = final_amount →
  P = 1000 := by
  sorry


end NUMINAMATH_CALUDE_initial_deposit_calculation_l2378_237871


namespace NUMINAMATH_CALUDE_count_divisors_of_M_l2378_237870

/-- The number of positive divisors of M, where M = 2^3 * 3^4 * 5^3 * 7^1 -/
def num_divisors : ℕ :=
  (3 + 1) * (4 + 1) * (3 + 1) * (1 + 1)

/-- M is defined as 2^3 * 3^4 * 5^3 * 7^1 -/
def M : ℕ := 2^3 * 3^4 * 5^3 * 7^1

theorem count_divisors_of_M :
  num_divisors = 160 ∧ num_divisors = (Finset.filter (· ∣ M) (Finset.range (M + 1))).card :=
sorry

end NUMINAMATH_CALUDE_count_divisors_of_M_l2378_237870


namespace NUMINAMATH_CALUDE_subset_iff_elements_l2378_237829

theorem subset_iff_elements (A B : Set α) : A ⊆ B ↔ ∀ x, x ∈ A → x ∈ B := by sorry

end NUMINAMATH_CALUDE_subset_iff_elements_l2378_237829


namespace NUMINAMATH_CALUDE_tan_alpha_minus_beta_l2378_237861

theorem tan_alpha_minus_beta (α β : ℝ) 
  (h1 : Real.tan (α + π / 5) = 2) 
  (h2 : Real.tan (β - 4 * π / 5) = -3) : 
  Real.tan (α - β) = -1 := by
sorry

end NUMINAMATH_CALUDE_tan_alpha_minus_beta_l2378_237861


namespace NUMINAMATH_CALUDE_gondor_repair_earnings_l2378_237831

/-- Gondor's repair earnings problem -/
theorem gondor_repair_earnings (phone_repair_price laptop_repair_price : ℕ)
  (monday_phones wednesday_laptops thursday_laptops : ℕ)
  (total_earnings : ℕ) :
  phone_repair_price = 10 →
  laptop_repair_price = 20 →
  monday_phones = 3 →
  wednesday_laptops = 2 →
  thursday_laptops = 4 →
  total_earnings = 200 →
  ∃ tuesday_phones : ℕ,
    total_earnings = 
      phone_repair_price * (monday_phones + tuesday_phones) +
      laptop_repair_price * (wednesday_laptops + thursday_laptops) ∧
    tuesday_phones = 5 :=
by sorry

end NUMINAMATH_CALUDE_gondor_repair_earnings_l2378_237831


namespace NUMINAMATH_CALUDE_fraction_subtraction_result_l2378_237864

theorem fraction_subtraction_result : 
  (3 * 5 + 5 * 7 + 7 * 9) / (2 * 4 + 4 * 6 + 6 * 8) - 
  (2 * 4 + 4 * 6 + 6 * 8) / (3 * 5 + 5 * 7 + 7 * 9) = 74 / 119 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_result_l2378_237864


namespace NUMINAMATH_CALUDE_prob_two_qualified_bottles_l2378_237834

/-- The probability of a single bottle of beverage being qualified -/
def qualified_rate : ℝ := 0.8

/-- The probability of two bottles both being qualified -/
def both_qualified_prob : ℝ := qualified_rate * qualified_rate

/-- Theorem: The probability of drinking two qualified bottles is 0.64 -/
theorem prob_two_qualified_bottles : both_qualified_prob = 0.64 := by sorry

end NUMINAMATH_CALUDE_prob_two_qualified_bottles_l2378_237834


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l2378_237811

theorem arithmetic_sequence_formula (a : ℕ → ℝ) :
  (∀ n, a (n + 1) < a n) →  -- decreasing sequence
  a 2 * a 4 * a 6 = 45 →
  a 2 + a 4 + a 6 = 15 →
  ∃ d : ℝ, d < 0 ∧ ∀ n, a n = a 1 + (n - 1) * d ∧ a n = -2 * n + 13 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l2378_237811


namespace NUMINAMATH_CALUDE_max_points_theorem_l2378_237858

/-- Represents a football tournament with the given conditions -/
structure Tournament where
  teams : Nat
  total_points : Nat
  draw_points : Nat
  win_points : Nat

/-- Calculates the total number of matches in the tournament -/
def total_matches (t : Tournament) : Nat :=
  t.teams * (t.teams - 1) / 2

/-- Represents the result of solving the tournament equations -/
structure TournamentResult where
  draws : Nat
  wins : Nat

/-- Solves the tournament equations to find the number of draws and wins -/
def solve_tournament (t : Tournament) : TournamentResult :=
  { draws := 23, wins := 5 }

/-- Calculates the maximum points a single team can obtain -/
def max_points (t : Tournament) (result : TournamentResult) : Nat :=
  (result.wins * t.win_points) + (t.teams - 1 - result.wins) * t.draw_points

/-- The main theorem stating the maximum points obtainable by a single team -/
theorem max_points_theorem (t : Tournament) 
  (h1 : t.teams = 8)
  (h2 : t.total_points = 61)
  (h3 : t.draw_points = 1)
  (h4 : t.win_points = 3) :
  max_points t (solve_tournament t) = 17 := by
  sorry

#eval max_points 
  { teams := 8, total_points := 61, draw_points := 1, win_points := 3 } 
  (solve_tournament { teams := 8, total_points := 61, draw_points := 1, win_points := 3 })

end NUMINAMATH_CALUDE_max_points_theorem_l2378_237858


namespace NUMINAMATH_CALUDE_compare_sizes_l2378_237857

theorem compare_sizes (a b : ℝ) (ha : a = 0.2^(1/2)) (hb : b = 0.5^(1/5)) :
  0 < a ∧ a < b ∧ b < 1 := by sorry

end NUMINAMATH_CALUDE_compare_sizes_l2378_237857


namespace NUMINAMATH_CALUDE_three_times_relationship_l2378_237855

theorem three_times_relationship (M₁ M₂ M₃ M₄ : ℝ) 
  (h₁ : M₁ = 2.02e-6)
  (h₂ : M₂ = 0.0000202)
  (h₃ : M₃ = 0.00000202)
  (h₄ : M₄ = 6.06e-5) :
  (M₄ = 3 * M₂ ∧ 
   M₄ ≠ 3 * M₁ ∧ 
   M₄ ≠ 3 * M₃ ∧ 
   M₃ ≠ 3 * M₁ ∧ 
   M₃ ≠ 3 * M₂ ∧ 
   M₂ ≠ 3 * M₁) :=
by sorry

end NUMINAMATH_CALUDE_three_times_relationship_l2378_237855


namespace NUMINAMATH_CALUDE_event_probability_l2378_237825

-- Define the probability of the event occurring in a single trial
def p : ℝ := sorry

-- Define the probability of the event not occurring in a single trial
def q : ℝ := 1 - p

-- Define the number of trials
def n : ℕ := 3

-- State the theorem
theorem event_probability :
  (1 - q^n = 0.973) → p = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_event_probability_l2378_237825


namespace NUMINAMATH_CALUDE_machine_year_production_l2378_237898

/-- A machine that produces items at a constant rate. -/
structure Machine where
  production_rate : ℕ  -- Items produced per hour

/-- Represents a year with a fixed number of days. -/
structure Year where
  days : ℕ

/-- Calculates the total number of units produced by a machine in a year. -/
def units_produced (m : Machine) (y : Year) : ℕ :=
  m.production_rate * y.days * 24

/-- Theorem stating that a machine producing one item per hour will make 8760 units in a year of 365 days. -/
theorem machine_year_production :
  ∀ (m : Machine) (y : Year),
    m.production_rate = 1 →
    y.days = 365 →
    units_produced m y = 8760 :=
by
  sorry


end NUMINAMATH_CALUDE_machine_year_production_l2378_237898


namespace NUMINAMATH_CALUDE_find_number_l2378_237847

theorem find_number : ∃ x : ℝ, (0.15 * 40 = 0.25 * x + 2) ∧ x = 16 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l2378_237847


namespace NUMINAMATH_CALUDE_solve_linear_equation_l2378_237821

theorem solve_linear_equation :
  ∃ y : ℝ, (7 * y - 10 = 4 * y + 5) ∧ y = 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l2378_237821


namespace NUMINAMATH_CALUDE_special_function_properties_l2378_237840

/-- A function satisfying certain properties -/
structure SpecialFunction where
  g : ℝ → ℝ
  pos : ∀ x, g x > 0
  mult : ∀ a b, g a * g b = g (a * b)

/-- Properties of the special function -/
theorem special_function_properties (f : SpecialFunction) :
  (f.g 1 = 1) ∧
  (∀ a ≠ 0, f.g (a⁻¹) = (f.g a)⁻¹) ∧
  (∀ a, f.g (a^2) = f.g a * f.g a) := by
  sorry

end NUMINAMATH_CALUDE_special_function_properties_l2378_237840


namespace NUMINAMATH_CALUDE_prize_distribution_and_cost_l2378_237812

/-- Represents the prize distribution and cost calculation for a school event --/
theorem prize_distribution_and_cost 
  (x : ℕ) -- number of first prize items
  (h1 : x + (3*x - 2) + (52 - 4*x) = 50) -- total prizes constraint
  (h2 : x > 0) -- ensure positive number of first prize items
  (h3 : 3*x - 2 ≥ 0) -- ensure non-negative number of second prize items
  : 
  (20*x + 14*(3*x - 2) + 8*(52 - 4*x) = 30*x + 388) ∧ 
  (3*x - 2 = 22 → 20*x + 14*(3*x - 2) + 8*(52 - 4*x) = 628)
  := by sorry


end NUMINAMATH_CALUDE_prize_distribution_and_cost_l2378_237812


namespace NUMINAMATH_CALUDE_base8_to_base7_conversion_l2378_237883

-- Define a function to convert from base 8 to base 10
def base8ToBase10 (n : Nat) : Nat :=
  (n / 100) * 64 + ((n / 10) % 10) * 8 + (n % 10)

-- Define a function to convert from base 10 to base 7
def base10ToBase7 (n : Nat) : Nat :=
  (n / 343) * 1000 + ((n / 49) % 7) * 100 + ((n / 7) % 7) * 10 + (n % 7)

theorem base8_to_base7_conversion :
  base10ToBase7 (base8ToBase10 563) = 1162 := by
  sorry

end NUMINAMATH_CALUDE_base8_to_base7_conversion_l2378_237883


namespace NUMINAMATH_CALUDE_final_turtle_count_l2378_237841

def turtle_statues : ℕ → ℕ
| 0 => 4  -- Initial number of statues
| 1 => turtle_statues 0 * 4  -- Second year: quadrupled
| 2 => turtle_statues 1 + 12 - 3  -- Third year: added 12, removed 3
| 3 => turtle_statues 2 + 2 * 3  -- Fourth year: added twice the number broken in year 3
| _ => 0  -- We only care about the first 4 years

theorem final_turtle_count : turtle_statues 3 = 31 := by
  sorry

#eval turtle_statues 3

end NUMINAMATH_CALUDE_final_turtle_count_l2378_237841


namespace NUMINAMATH_CALUDE_three_card_draw_different_colors_l2378_237813

def total_cards : ℕ := 16
def cards_per_color : ℕ := 4
def num_colors : ℕ := 4
def cards_drawn : ℕ := 3

theorem three_card_draw_different_colors : 
  (Nat.choose total_cards cards_drawn) - (num_colors * Nat.choose cards_per_color cards_drawn) = 544 := by
  sorry

end NUMINAMATH_CALUDE_three_card_draw_different_colors_l2378_237813


namespace NUMINAMATH_CALUDE_max_cos_sum_l2378_237843

theorem max_cos_sum (x y : ℝ) 
  (h1 : Real.sin y + Real.sin x + Real.cos (3 * x) = 0)
  (h2 : Real.sin (2 * y) - Real.sin (2 * x) = Real.cos (4 * x) + Real.cos (2 * x)) :
  ∃ (max : ℝ), max = 1 + (Real.sqrt (2 + Real.sqrt 2)) / 2 ∧ 
    ∀ (x' y' : ℝ), 
      Real.sin y' + Real.sin x' + Real.cos (3 * x') = 0 →
      Real.sin (2 * y') - Real.sin (2 * x') = Real.cos (4 * x') + Real.cos (2 * x') →
      Real.cos y' + Real.cos x' ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_cos_sum_l2378_237843


namespace NUMINAMATH_CALUDE_lucy_money_problem_l2378_237885

theorem lucy_money_problem (initial_amount : ℚ) : 
  (initial_amount * (2/3) * (3/4) = 15) → initial_amount = 30 := by
  sorry

end NUMINAMATH_CALUDE_lucy_money_problem_l2378_237885


namespace NUMINAMATH_CALUDE_hyperbola_parameter_value_l2378_237897

/-- Represents a hyperbola with parameter a -/
structure Hyperbola (a : ℝ) :=
  (equation : ∀ (x y : ℝ), x^2 / (a - 3) + y^2 / (1 - a) = 1)

/-- Condition that the foci lie on the x-axis -/
def foci_on_x_axis (h : Hyperbola a) : Prop :=
  a > 1 ∧ a > 3

/-- Condition that the focal distance is 4 -/
def focal_distance_is_4 (h : Hyperbola a) : Prop :=
  ∃ (c : ℝ), c^2 = (a - 3) - (1 - a) ∧ 2 * c = 4

/-- Theorem stating that for a hyperbola with the given conditions, a = 4 -/
theorem hyperbola_parameter_value
  (a : ℝ)
  (h : Hyperbola a)
  (h_foci : foci_on_x_axis h)
  (h_focal : focal_distance_is_4 h) :
  a = 4 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_parameter_value_l2378_237897


namespace NUMINAMATH_CALUDE_triangle_angle_solution_l2378_237839

-- Define the angles in degrees
def angle_PQR : ℝ := 90
def angle_PQS (x : ℝ) : ℝ := 3 * x
def angle_SQR (y : ℝ) : ℝ := y

-- State the theorem
theorem triangle_angle_solution :
  ∃ (x y : ℝ),
    angle_PQS x + angle_SQR y = angle_PQR ∧
    x = 18 ∧
    y = 36 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_solution_l2378_237839


namespace NUMINAMATH_CALUDE_equation_is_parabola_l2378_237846

/-- Represents a point in a 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Defines the equation |y - 3| = √((x+4)² + (y-1)²) -/
def equation (p : Point2D) : Prop :=
  |p.y - 3| = Real.sqrt ((p.x + 4)^2 + (p.y - 1)^2)

/-- Defines a parabola as a set of points satisfying a quadratic equation -/
def isParabola (S : Set Point2D) : Prop :=
  ∃ (a b c d e : ℝ), a ≠ 0 ∧ 
    ∀ p ∈ S, a * p.x^2 + b * p.x * p.y + c * p.y^2 + d * p.x + e * p.y = 0

/-- Theorem stating that the given equation represents a parabola -/
theorem equation_is_parabola :
  isParabola {p : Point2D | equation p} :=
sorry

end NUMINAMATH_CALUDE_equation_is_parabola_l2378_237846


namespace NUMINAMATH_CALUDE_regular_polygon_140_deg_interior_angle_l2378_237801

/-- A regular polygon with interior angles of 140 degrees has 9 sides -/
theorem regular_polygon_140_deg_interior_angle (n : ℕ) : 
  (n ≥ 3) → 
  (∀ θ : ℝ, θ = 140 → (180 * (n - 2) : ℝ) = n * θ) → 
  n = 9 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_140_deg_interior_angle_l2378_237801


namespace NUMINAMATH_CALUDE_least_number_divisible_by_five_primes_l2378_237884

theorem least_number_divisible_by_five_primes : ∃ n : ℕ, 
  (∃ p₁ p₂ p₃ p₄ p₅ : ℕ, 
    Nat.Prime p₁ ∧ Nat.Prime p₂ ∧ Nat.Prime p₃ ∧ Nat.Prime p₄ ∧ Nat.Prime p₅ ∧
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₁ ≠ p₅ ∧ 
    p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₂ ≠ p₅ ∧
    p₃ ≠ p₄ ∧ p₃ ≠ p₅ ∧
    p₄ ≠ p₅ ∧
    p₁ ∣ n ∧ p₂ ∣ n ∧ p₃ ∣ n ∧ p₄ ∣ n ∧ p₅ ∣ n) ∧
  (∀ m : ℕ, m < n → 
    ¬(∃ q₁ q₂ q₃ q₄ q₅ : ℕ, 
      Nat.Prime q₁ ∧ Nat.Prime q₂ ∧ Nat.Prime q₃ ∧ Nat.Prime q₄ ∧ Nat.Prime q₅ ∧
      q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₁ ≠ q₄ ∧ q₁ ≠ q₅ ∧ 
      q₂ ≠ q₃ ∧ q₂ ≠ q₄ ∧ q₂ ≠ q₅ ∧
      q₃ ≠ q₄ ∧ q₃ ≠ q₅ ∧
      q₄ ≠ q₅ ∧
      q₁ ∣ m ∧ q₂ ∣ m ∧ q₃ ∣ m ∧ q₄ ∣ m ∧ q₅ ∣ m)) ∧
  n = 2310 :=
by sorry

end NUMINAMATH_CALUDE_least_number_divisible_by_five_primes_l2378_237884


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2378_237860

theorem sum_of_roots_quadratic (x : ℝ) : 
  (x^2 - 6*x + 8 = 0) → (∃ α β : ℝ, (α + β = 6) ∧ (α * β = 8) ∧ (α ≠ β → (α - β)^2 = 36 - 4*8)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2378_237860


namespace NUMINAMATH_CALUDE_calculate_flat_fee_l2378_237848

/-- Calculates the flat fee for shipping given the total cost, cost per pound, and weight. -/
theorem calculate_flat_fee (C : ℝ) (cost_per_pound : ℝ) (weight : ℝ) (h1 : C = 9) (h2 : cost_per_pound = 0.8) (h3 : weight = 5) :
  ∃ F : ℝ, C = F + cost_per_pound * weight ∧ F = 5 := by
  sorry

end NUMINAMATH_CALUDE_calculate_flat_fee_l2378_237848


namespace NUMINAMATH_CALUDE_captain_selection_theorem_l2378_237836

/-- The number of ways to choose 4 captains from a team of 12 people,
    where two of the captains must be from a subset of 4 specific players. -/
def captain_selection_ways (total_team : ℕ) (total_captains : ℕ) (specific_subset : ℕ) (required_from_subset : ℕ) : ℕ :=
  (Nat.choose specific_subset required_from_subset) * 
  (Nat.choose (total_team - specific_subset) (total_captains - required_from_subset))

/-- Theorem stating that the number of ways to choose 4 captains from a team of 12 people,
    where two of the captains must be from a subset of 4 specific players, is 168. -/
theorem captain_selection_theorem : 
  captain_selection_ways 12 4 4 2 = 168 := by
  sorry

end NUMINAMATH_CALUDE_captain_selection_theorem_l2378_237836


namespace NUMINAMATH_CALUDE_smallest_n_value_l2378_237828

theorem smallest_n_value (o y m : ℕ+) (n : ℕ+) : 
  (10 * o = 16 * y) ∧ (16 * y = 18 * m) ∧ (18 * m = 18 * n) →
  n ≥ 40 ∧ ∃ (o' y' m' : ℕ+), 10 * o' = 16 * y' ∧ 16 * y' = 18 * m' ∧ 18 * m' = 18 * 40 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_value_l2378_237828


namespace NUMINAMATH_CALUDE_crabapple_sequences_l2378_237882

/-- The number of students in Mrs. Crabapple's class -/
def num_students : ℕ := 12

/-- The number of times the class meets in a week -/
def meetings_per_week : ℕ := 5

/-- The number of different sequences of crabapple recipients possible in a week -/
def num_sequences : ℕ := num_students ^ meetings_per_week

theorem crabapple_sequences :
  num_sequences = 248832 := by
  sorry

end NUMINAMATH_CALUDE_crabapple_sequences_l2378_237882


namespace NUMINAMATH_CALUDE_harry_potter_book_price_l2378_237850

theorem harry_potter_book_price : 
  ∀ (wang_money li_money book_price : ℕ),
  wang_money + 6 = 2 * book_price →
  li_money + 31 = 2 * book_price →
  wang_money + li_money = 3 * book_price →
  book_price = 37 := by
sorry

end NUMINAMATH_CALUDE_harry_potter_book_price_l2378_237850


namespace NUMINAMATH_CALUDE_leftover_coins_value_l2378_237854

def quarters_per_roll : ℕ := 30
def dimes_per_roll : ℕ := 60
def michael_quarters : ℕ := 94
def michael_dimes : ℕ := 184
def sara_quarters : ℕ := 137
def sara_dimes : ℕ := 312

def quarter_value : ℚ := 0.25
def dime_value : ℚ := 0.10

def total_quarters : ℕ := michael_quarters + sara_quarters
def total_dimes : ℕ := michael_dimes + sara_dimes

def leftover_quarters : ℕ := total_quarters % quarters_per_roll
def leftover_dimes : ℕ := total_dimes % dimes_per_roll

def leftover_value : ℚ := 
  (leftover_quarters : ℚ) * quarter_value + (leftover_dimes : ℚ) * dime_value

theorem leftover_coins_value : leftover_value = 6.85 := by
  sorry

end NUMINAMATH_CALUDE_leftover_coins_value_l2378_237854


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2378_237859

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  (x₁ = 1 + Real.sqrt 3 ∧ x₁^2 - 2*x₁ = 2) ∧ 
  (x₂ = 1 - Real.sqrt 3 ∧ x₂^2 - 2*x₂ = 2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2378_237859


namespace NUMINAMATH_CALUDE_garden_area_l2378_237835

theorem garden_area (width : ℝ) (length : ℝ) :
  length = 3 * width + 30 →
  2 * (length + width) = 800 →
  width * length = 28443.75 := by
sorry

end NUMINAMATH_CALUDE_garden_area_l2378_237835


namespace NUMINAMATH_CALUDE_integer_sum_l2378_237868

theorem integer_sum (x y : ℕ+) : x - y = 14 → x * y = 48 → x + y = 20 := by
  sorry

end NUMINAMATH_CALUDE_integer_sum_l2378_237868


namespace NUMINAMATH_CALUDE_caterpillar_final_position_l2378_237889

/-- Represents a point in 2D space -/
structure Point where
  x : Int
  y : Int

/-- Represents a direction as a unit vector -/
inductive Direction
  | West
  | North
  | East
  | South

/-- Represents the state of the caterpillar -/
structure CaterpillarState where
  position : Point
  direction : Direction
  moveDistance : Nat

/-- Performs a single move and turn -/
def move (state : CaterpillarState) : CaterpillarState :=
  sorry

/-- Performs n moves and turns -/
def moveNTimes (initialState : CaterpillarState) (n : Nat) : CaterpillarState :=
  sorry

/-- The main theorem to prove -/
theorem caterpillar_final_position :
  let initialState : CaterpillarState := {
    position := { x := 15, y := -15 },
    direction := Direction.West,
    moveDistance := 1
  }
  let finalState := moveNTimes initialState 1010
  finalState.position = { x := -491, y := 489 } :=
sorry

end NUMINAMATH_CALUDE_caterpillar_final_position_l2378_237889


namespace NUMINAMATH_CALUDE_roots_of_polynomials_l2378_237820

theorem roots_of_polynomials (α : ℝ) : 
  α^2 = 2*α + 2 → α^5 = 44*α + 32 := by sorry

end NUMINAMATH_CALUDE_roots_of_polynomials_l2378_237820


namespace NUMINAMATH_CALUDE_a_fourth_minus_four_a_cubed_minus_four_a_plus_seven_equals_eight_l2378_237852

theorem a_fourth_minus_four_a_cubed_minus_four_a_plus_seven_equals_eight :
  ∀ a : ℝ, a = 1 / (Real.sqrt 5 - 2) → a^4 - 4*a^3 - 4*a + 7 = 8 := by
sorry

end NUMINAMATH_CALUDE_a_fourth_minus_four_a_cubed_minus_four_a_plus_seven_equals_eight_l2378_237852


namespace NUMINAMATH_CALUDE_charles_pictures_before_work_l2378_237872

/-- The number of pictures Charles drew before going to work yesterday -/
def pictures_before_work : ℕ → ℕ → ℕ → ℕ → ℕ
  | total_papers, papers_left, pictures_today, pictures_after_work =>
    total_papers - papers_left - pictures_today - pictures_after_work

theorem charles_pictures_before_work :
  pictures_before_work 20 2 6 6 = 6 := by
  sorry

end NUMINAMATH_CALUDE_charles_pictures_before_work_l2378_237872


namespace NUMINAMATH_CALUDE_divide_and_add_l2378_237802

theorem divide_and_add (x : ℝ) : x = 72 → (x / 6) + 5 = 17 := by
  sorry

end NUMINAMATH_CALUDE_divide_and_add_l2378_237802


namespace NUMINAMATH_CALUDE_samuel_remaining_amount_samuel_remaining_amount_proof_l2378_237880

theorem samuel_remaining_amount 
  (total : ℕ) 
  (samuel_fraction : ℚ) 
  (spent_fraction : ℚ) 
  (h1 : total = 240) 
  (h2 : samuel_fraction = 3/4) 
  (h3 : spent_fraction = 1/5) : 
  ℕ :=
  let samuel_received : ℚ := total * samuel_fraction
  let samuel_spent : ℚ := total * spent_fraction
  let samuel_remaining : ℚ := samuel_received - samuel_spent
  132

theorem samuel_remaining_amount_proof 
  (total : ℕ) 
  (samuel_fraction : ℚ) 
  (spent_fraction : ℚ) 
  (h1 : total = 240) 
  (h2 : samuel_fraction = 3/4) 
  (h3 : spent_fraction = 1/5) : 
  samuel_remaining_amount total samuel_fraction spent_fraction h1 h2 h3 = 132 := by
  sorry

end NUMINAMATH_CALUDE_samuel_remaining_amount_samuel_remaining_amount_proof_l2378_237880


namespace NUMINAMATH_CALUDE_largest_negative_and_smallest_absolute_l2378_237875

theorem largest_negative_and_smallest_absolute : ∃ (a b : ℤ),
  (∀ x : ℤ, x < 0 → x ≤ a) ∧
  (∀ y : ℤ, |b| ≤ |y|) ∧
  b - 4*a = 4 :=
sorry

end NUMINAMATH_CALUDE_largest_negative_and_smallest_absolute_l2378_237875


namespace NUMINAMATH_CALUDE_min_value_theorem_l2378_237856

theorem min_value_theorem (x : ℝ) (hx : x > 0) :
  4 * x^2 + 1 / x^3 ≥ 5 ∧
  (4 * x^2 + 1 / x^3 = 5 ↔ x = 1) := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2378_237856


namespace NUMINAMATH_CALUDE_mathematical_induction_l2378_237810

theorem mathematical_induction (P : ℕ → Prop) (base : ℕ) 
  (base_case : P base)
  (inductive_step : ∀ k : ℕ, k ≥ base → P k → P (k + 1)) :
  ∀ n : ℕ, n ≥ base → P n :=
by
  sorry


end NUMINAMATH_CALUDE_mathematical_induction_l2378_237810


namespace NUMINAMATH_CALUDE_no_solution_exists_l2378_237853

theorem no_solution_exists : ¬∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ 1 / a^2 + 1 / b^2 = 1 / (a^2 + b^2) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l2378_237853


namespace NUMINAMATH_CALUDE_sequence_formula_l2378_237867

theorem sequence_formula (a : ℕ+ → ℝ) (S : ℕ+ → ℝ) 
  (h : ∀ n : ℕ+, S n = 2 * a n - 1) : 
  ∀ n : ℕ+, a n = 2^(n.val - 1) := by
  sorry

end NUMINAMATH_CALUDE_sequence_formula_l2378_237867


namespace NUMINAMATH_CALUDE_min_value_fraction_l2378_237804

theorem min_value_fraction (a b : ℝ) (h1 : a > b) (h2 : a * b = 1) :
  (a^2 + b^2) / (a - b) ≥ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_fraction_l2378_237804


namespace NUMINAMATH_CALUDE_function_properties_l2378_237826

def FunctionProperties (f : ℝ → ℝ) : Prop :=
  (∀ x y, f x + f y = f (x + y)) ∧ (∀ x, x > 0 → f x < 0)

theorem function_properties (f : ℝ → ℝ) (h : FunctionProperties f) :
  (∀ x, f (-x) = -f x) ∧ (∀ x y, x < y → f x > f y) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l2378_237826
