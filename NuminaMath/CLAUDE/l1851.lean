import Mathlib

namespace larger_square_side_length_l1851_185195

theorem larger_square_side_length 
  (small_square_side : ℝ) 
  (larger_square_perimeter : ℝ) 
  (h1 : small_square_side = 20) 
  (h2 : larger_square_perimeter = 4 * small_square_side + 20) : 
  larger_square_perimeter / 4 = 25 := by
  sorry

end larger_square_side_length_l1851_185195


namespace train_length_proof_l1851_185193

/-- Given a train crossing two platforms with constant speed, prove its length is 70 meters. -/
theorem train_length_proof (
  platform1_length : ℝ)
  (platform2_length : ℝ)
  (time1 : ℝ)
  (time2 : ℝ)
  (h1 : platform1_length = 170)
  (h2 : platform2_length = 250)
  (h3 : time1 = 15)
  (h4 : time2 = 20)
  (h5 : (platform1_length + train_length) / time1 = (platform2_length + train_length) / time2)
  : train_length = 70 := by
  sorry

end train_length_proof_l1851_185193


namespace xy_value_l1851_185115

theorem xy_value (x y : ℝ) (h : x * (x + y) = x^2 + 18) : x * y = 18 := by
  sorry

end xy_value_l1851_185115


namespace weightlifter_fourth_minute_l1851_185106

/-- Calculates the total weight a weightlifter can lift in the 4th minute given initial weights,
    weight increments, and fatigue factor. -/
def weightLifterFourthMinute (leftInitial rightInitial leftIncrement rightIncrement fatigueDecline : ℕ) : ℕ :=
  let leftAfterThree := leftInitial + 3 * leftIncrement
  let rightAfterThree := rightInitial + 3 * rightIncrement
  let totalAfterThree := leftAfterThree + rightAfterThree
  totalAfterThree - fatigueDecline

/-- Theorem stating that the weightlifter can lift 55 pounds in the 4th minute under given conditions. -/
theorem weightlifter_fourth_minute :
  weightLifterFourthMinute 12 18 4 6 5 = 55 := by
  sorry

end weightlifter_fourth_minute_l1851_185106


namespace maximize_product_l1851_185103

theorem maximize_product (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_sum : x + y = 50) :
  x^4 * y^3 ≤ (200/7)^4 * (150/7)^3 ∧
  (x = 200/7 ∧ y = 150/7) → x^4 * y^3 = (200/7)^4 * (150/7)^3 :=
by sorry

end maximize_product_l1851_185103


namespace line_touches_x_axis_twice_l1851_185196

/-- Represents the equation d = x^2 - x^3 -/
def d (x : ℝ) : ℝ := x^2 - x^3

/-- The line touches the x-axis when d(x) = 0 -/
def touches_x_axis (x : ℝ) : Prop := d x = 0

theorem line_touches_x_axis_twice :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ touches_x_axis x₁ ∧ touches_x_axis x₂ ∧
  ∀ (x : ℝ), touches_x_axis x → (x = x₁ ∨ x = x₂) :=
sorry

end line_touches_x_axis_twice_l1851_185196


namespace ages_when_john_is_50_l1851_185174

/- Define the initial ages and relationships -/
def john_initial_age : ℕ := 10
def alice_initial_age : ℕ := 2 * john_initial_age
def mike_initial_age : ℕ := alice_initial_age - 4

/- Define John's future age -/
def john_future_age : ℕ := 50

/- Define the theorem to prove -/
theorem ages_when_john_is_50 :
  (john_future_age + (alice_initial_age - john_initial_age) = 60) ∧
  (john_future_age + (mike_initial_age - john_initial_age) = 56) := by
  sorry

end ages_when_john_is_50_l1851_185174


namespace science_fair_participants_l1851_185127

theorem science_fair_participants (total_girls : ℕ) (total_boys : ℕ)
  (girls_participation_rate : ℚ) (boys_participation_rate : ℚ)
  (h1 : total_girls = 150)
  (h2 : total_boys = 100)
  (h3 : girls_participation_rate = 4 / 5)
  (h4 : boys_participation_rate = 3 / 4) :
  let participating_girls : ℚ := girls_participation_rate * total_girls
  let participating_boys : ℚ := boys_participation_rate * total_boys
  let total_participants : ℚ := participating_girls + participating_boys
  participating_girls / total_participants = 8 / 13 := by
sorry

end science_fair_participants_l1851_185127


namespace cd_ratio_l1851_185155

/-- Represents the number of CDs Tyler has at different stages --/
structure CDCount where
  initial : ℕ
  given_away : ℕ
  bought : ℕ
  final : ℕ

/-- Theorem stating the ratio of CDs given away to initial CDs --/
theorem cd_ratio (c : CDCount) 
  (h1 : c.initial = 21)
  (h2 : c.bought = 8)
  (h3 : c.final = 22)
  (h4 : c.initial - c.given_away + c.bought = c.final) :
  (c.given_away : ℚ) / c.initial = 1 / 3 := by
  sorry

#check cd_ratio

end cd_ratio_l1851_185155


namespace rationalize_and_product_l1851_185179

theorem rationalize_and_product : ∃ (A B C : ℚ),
  (2 + Real.sqrt 5) / (3 - Real.sqrt 5) = A + B * Real.sqrt C ∧
  A = 11/4 ∧ B = 5/4 ∧ C = 5 ∧ A * B * C = 275/16 := by
  sorry

end rationalize_and_product_l1851_185179


namespace solve_inequality_find_a_range_l1851_185129

-- Define the function f
def f (x : ℝ) : ℝ := |3*x + 2|

-- Part I
theorem solve_inequality :
  {x : ℝ | f x < 4 - |x - 1|} = {x : ℝ | -5/4 < x ∧ x < 1/2} :=
sorry

-- Part II
theorem find_a_range (m n : ℝ) (hm : m > 0) (hn : n > 0) (hmn : m + n = 1) :
  (∀ x : ℝ, |x - a| - f x ≤ 1/m + 1/n) ↔ (0 < a ∧ a ≤ 10/3) :=
sorry

end solve_inequality_find_a_range_l1851_185129


namespace solution_set_l1851_185175

def is_valid (n : ℕ) : Prop :=
  n ≥ 6 ∧ n.choose 4 * 24 ≤ n.factorial / ((n - 4).factorial)

theorem solution_set :
  {n : ℕ | is_valid n} = {6, 7, 8, 9} := by sorry

end solution_set_l1851_185175


namespace axis_of_symmetry_sin_l1851_185132

open Real

theorem axis_of_symmetry_sin (φ : ℝ) :
  (∀ x, |sin (2*x + φ)| ≤ |sin (π/3 + φ)|) →
  ∃ k : ℤ, 2*(2*π/3) + φ = k*π + π/2 :=
by sorry

end axis_of_symmetry_sin_l1851_185132


namespace min_value_expression_l1851_185143

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2 * b = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + 2 * y = 1 → a^2 + 4 * b^2 + 1 / (a * b) ≤ x^2 + 4 * y^2 + 1 / (x * y)) ∧
  a^2 + 4 * b^2 + 1 / (a * b) = 17 / 2 :=
sorry

end min_value_expression_l1851_185143


namespace factor_polynomial_l1851_185165

theorem factor_polynomial (x : ℝ) : 66 * x^6 - 231 * x^12 = 33 * x^6 * (2 - 7 * x^6) := by
  sorry

end factor_polynomial_l1851_185165


namespace subsets_without_consecutive_eq_fib_l1851_185121

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Number of subsets without consecutive elements -/
def subsets_without_consecutive (n : ℕ) : ℕ :=
  fib (n + 2)

/-- Theorem: The number of subsets of {1, 2, 3, ..., n} that do not contain
    two consecutive numbers is equal to the (n+2)th Fibonacci number -/
theorem subsets_without_consecutive_eq_fib (n : ℕ) :
  subsets_without_consecutive n = fib (n + 2) := by
  sorry


end subsets_without_consecutive_eq_fib_l1851_185121


namespace quadratic_no_real_roots_l1851_185157

theorem quadratic_no_real_roots (a b c : ℝ) (h : b^2 = a*c) :
  ∀ x : ℝ, a*x^2 + b*x + c ≠ 0 :=
sorry

end quadratic_no_real_roots_l1851_185157


namespace f_odd_and_increasing_l1851_185120

-- Define the function f(x) = x|x|
def f (x : ℝ) : ℝ := x * abs x

-- Theorem statement
theorem f_odd_and_increasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f x < f y) :=
sorry

end f_odd_and_increasing_l1851_185120


namespace gilda_marbles_l1851_185162

theorem gilda_marbles (M : ℝ) (h : M > 0) : 
  let remaining_after_pedro : ℝ := M * (1 - 0.3)
  let remaining_after_ebony : ℝ := remaining_after_pedro * (1 - 0.1)
  let remaining_after_lisa : ℝ := remaining_after_ebony * (1 - 0.4)
  remaining_after_lisa / M = 0.378 := by
  sorry

end gilda_marbles_l1851_185162


namespace cost_of_horse_l1851_185159

/-- Proves that the cost of a horse is 2000 given the problem conditions --/
theorem cost_of_horse (total_cost : ℝ) (num_horses : ℕ) (num_cows : ℕ) 
  (horse_profit_rate : ℝ) (cow_profit_rate : ℝ) (total_profit : ℝ) :
  total_cost = 13400 ∧ 
  num_horses = 4 ∧ 
  num_cows = 9 ∧ 
  horse_profit_rate = 0.1 ∧ 
  cow_profit_rate = 0.2 ∧ 
  total_profit = 1880 →
  ∃ (horse_cost cow_cost : ℝ),
    num_horses * horse_cost + num_cows * cow_cost = total_cost ∧
    num_horses * horse_cost * horse_profit_rate + num_cows * cow_cost * cow_profit_rate = total_profit ∧
    horse_cost = 2000 := by
  sorry

end cost_of_horse_l1851_185159


namespace number_puzzle_l1851_185131

theorem number_puzzle : ∃ x : ℤ, x - (28 - (37 - (15 - 19))) = 58 ∧ x = 45 := by
  sorry

end number_puzzle_l1851_185131


namespace percent_increase_decrease_l1851_185156

theorem percent_increase_decrease (p q M : ℝ) 
  (hp : p > 0) (hq : q > 0) (hq_bound : q < 50) (hM : M > 0) :
  M * (1 + p / 100) * (1 - q / 100) < M ↔ p < 100 * q / (100 - q) := by
  sorry

end percent_increase_decrease_l1851_185156


namespace inequality_equivalence_l1851_185134

theorem inequality_equivalence (x : ℝ) : (x - 3) / 2 ≥ 1 ↔ x ≥ 5 := by
  sorry

end inequality_equivalence_l1851_185134


namespace ad_fraction_of_page_l1851_185167

theorem ad_fraction_of_page 
  (page_width : ℝ) 
  (page_height : ℝ) 
  (cost_per_sq_inch : ℝ) 
  (total_cost : ℝ) : 
  page_width = 9 → 
  page_height = 12 → 
  cost_per_sq_inch = 8 → 
  total_cost = 432 → 
  (total_cost / cost_per_sq_inch) / (page_width * page_height) = 1 / 2 := by
sorry

end ad_fraction_of_page_l1851_185167


namespace sufficient_not_necessary_condition_l1851_185163

theorem sufficient_not_necessary_condition (x y : ℝ) : 
  (∀ y, x = 0 → x * y = 0) ∧ (∃ x y, x * y = 0 ∧ x ≠ 0) := by sorry

end sufficient_not_necessary_condition_l1851_185163


namespace dot_product_specific_vectors_l1851_185158

theorem dot_product_specific_vectors :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (1, -1)
  (a.1 * b.1 + a.2 * b.2) = -1 := by
  sorry

end dot_product_specific_vectors_l1851_185158


namespace kanul_cash_proof_l1851_185137

/-- The total amount of cash Kanul had -/
def total_cash : ℝ := 1000

/-- The amount spent on raw materials -/
def raw_materials : ℝ := 500

/-- The amount spent on machinery -/
def machinery : ℝ := 400

/-- The percentage of total cash spent as cash -/
def cash_percentage : ℝ := 0.1

theorem kanul_cash_proof :
  total_cash = raw_materials + machinery + cash_percentage * total_cash :=
by sorry

end kanul_cash_proof_l1851_185137


namespace no_four_polynomials_exist_l1851_185192

-- Define a type for polynomials with real coefficients
def RealPolynomial := ℝ → ℝ

-- Define a predicate to check if a polynomial has a real root
def has_real_root (p : RealPolynomial) : Prop :=
  ∃ x : ℝ, p x = 0

-- Define a predicate to check if a polynomial has no real root
def has_no_real_root (p : RealPolynomial) : Prop :=
  ∀ x : ℝ, p x ≠ 0

theorem no_four_polynomials_exist :
  ¬ ∃ (P₁ P₂ P₃ P₄ : RealPolynomial),
    (has_real_root (λ x => P₁ x + P₂ x + P₃ x)) ∧
    (has_real_root (λ x => P₁ x + P₂ x + P₄ x)) ∧
    (has_real_root (λ x => P₁ x + P₃ x + P₄ x)) ∧
    (has_real_root (λ x => P₂ x + P₃ x + P₄ x)) ∧
    (has_no_real_root (λ x => P₁ x + P₂ x)) ∧
    (has_no_real_root (λ x => P₁ x + P₃ x)) ∧
    (has_no_real_root (λ x => P₁ x + P₄ x)) ∧
    (has_no_real_root (λ x => P₂ x + P₃ x)) ∧
    (has_no_real_root (λ x => P₂ x + P₄ x)) ∧
    (has_no_real_root (λ x => P₃ x + P₄ x)) :=
by
  sorry

end no_four_polynomials_exist_l1851_185192


namespace intersection_points_theorem_l1851_185136

-- Define the functions
def y₁ (x : ℝ) : ℝ := x^2 + 2*x + 1
def y₂ (x b : ℝ) : ℝ := x^2 + b*x + 2
def y₃ (x c : ℝ) : ℝ := x^2 + c*x + 3

-- Define the number of roots for each function
def M₁ : ℕ := 1
def M₂ : ℕ := 1
def M₃ : ℕ := 2

-- Theorem statement
theorem intersection_points_theorem 
  (b c : ℝ) 
  (hb : b > 0) 
  (hc : c > 0) 
  (h_bc : b^2 = 2*c) 
  (h_M₁ : ∃! x, y₁ x = 0) 
  (h_M₂ : ∃! x, y₂ x b = 0) : 
  ∃ x₁ x₂, x₁ ≠ x₂ ∧ y₃ x₁ c = 0 ∧ y₃ x₂ c = 0 ∧ ∀ x, y₃ x c = 0 → x = x₁ ∨ x = x₂ :=
sorry

end intersection_points_theorem_l1851_185136


namespace complex_modulus_problem_l1851_185182

theorem complex_modulus_problem (Z : ℂ) (a : ℝ) :
  Z = 3 + a * I ∧ Complex.abs Z = 5 → a = 4 ∨ a = -4 := by
  sorry

end complex_modulus_problem_l1851_185182


namespace ellipse_condition_l1851_185122

def is_ellipse_with_y_axis_foci (m : ℝ) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ b > a ∧
  ∀ (x y : ℝ), x^2 / (5 - m) + y^2 / (m - 1) = 1 ↔ 
    x^2 / a^2 + y^2 / b^2 = 1

theorem ellipse_condition (m : ℝ) : 
  is_ellipse_with_y_axis_foci m ↔ 3 < m ∧ m < 5 :=
sorry

end ellipse_condition_l1851_185122


namespace right_triangle_area_l1851_185100

/-- The area of a right triangle with vertices at (0, 0), (0, 7), and (-7, 0) is 24.5 square units. -/
theorem right_triangle_area : 
  let vertex1 : ℝ × ℝ := (0, 0)
  let vertex2 : ℝ × ℝ := (0, 7)
  let vertex3 : ℝ × ℝ := (-7, 0)
  let base : ℝ := 7
  let height : ℝ := 7
  let area : ℝ := (1 / 2) * base * height
  area = 24.5 := by sorry

end right_triangle_area_l1851_185100


namespace tan_20_plus_4sin_20_equals_sqrt_3_l1851_185114

theorem tan_20_plus_4sin_20_equals_sqrt_3 :
  Real.tan (20 * π / 180) + 4 * Real.sin (20 * π / 180) = Real.sqrt 3 := by
  sorry

end tan_20_plus_4sin_20_equals_sqrt_3_l1851_185114


namespace smallest_number_of_ducks_l1851_185176

def duck_flock_size : ℕ := 18
def seagull_flock_size : ℕ := 10

theorem smallest_number_of_ducks (total_ducks total_seagulls : ℕ) : 
  total_ducks = total_seagulls → 
  total_ducks % duck_flock_size = 0 →
  total_seagulls % seagull_flock_size = 0 →
  total_ducks ≥ 90 :=
by sorry

end smallest_number_of_ducks_l1851_185176


namespace crane_sling_diameter_l1851_185180

/-- Represents the problem of finding the smallest suitable rope diameter for a crane sling --/
theorem crane_sling_diameter
  (M : ℝ)  -- Mass of the load in tons
  (n : ℕ)  -- Number of slings
  (α : ℝ)  -- Angle of each sling with vertical in radians
  (k : ℝ)  -- Safety factor
  (q : ℝ)  -- Maximum load per thread in N/mm²
  (g : ℝ)  -- Free fall acceleration in m/s²
  (h₁ : M = 20)
  (h₂ : n = 3)
  (h₃ : α = Real.pi / 6)  -- 30° in radians
  (h₄ : k = 6)
  (h₅ : q = 1000)
  (h₆ : g = 10)
  : ∃ (D : ℕ), D = 26 ∧ 
    (∀ (D' : ℕ), D' < D → 
      (Real.pi * D'^2 / 4) * q * 10^6 < 
      k * M * g * 1000 / (n * Real.cos α)) ∧
    (Real.pi * D^2 / 4) * q * 10^6 ≥ 
    k * M * g * 1000 / (n * Real.cos α) :=
sorry

end crane_sling_diameter_l1851_185180


namespace quadratic_real_roots_condition_l1851_185198

theorem quadratic_real_roots_condition (m : ℝ) :
  (∃ x : ℝ, x^2 - 2*x + m = 0) → m ≤ 1 := by
  sorry

end quadratic_real_roots_condition_l1851_185198


namespace string_length_problem_l1851_185117

theorem string_length_problem (num_strings : ℕ) (total_length : ℝ) (h1 : num_strings = 7) (h2 : total_length = 98) :
  total_length / num_strings = 14 := by
  sorry

end string_length_problem_l1851_185117


namespace function_with_two_integer_solutions_l1851_185168

theorem function_with_two_integer_solutions (a : ℝ) : 
  (∃ x y : ℤ, x ≠ y ∧ 
   (∀ z : ℤ, (Real.log (↑z) - a * (↑z)^2 - (a - 2) * ↑z > 0) ↔ (z = x ∨ z = y))) →
  (1 < a ∧ a ≤ (4 + Real.log 2) / 6) :=
sorry

end function_with_two_integer_solutions_l1851_185168


namespace number_problem_l1851_185116

theorem number_problem : ∃ x : ℝ, 0.65 * x = 0.05 * 60 + 23 ∧ x = 40 := by
  sorry

end number_problem_l1851_185116


namespace quadratic_solution_and_gcd_sum_l1851_185160

theorem quadratic_solution_and_gcd_sum : ∃ m n p : ℕ,
  (∀ x : ℝ, x * (4 * x - 5) = 7 ↔ x = (m + Real.sqrt n) / p ∨ x = (m - Real.sqrt n) / p) ∧
  Nat.gcd m (Nat.gcd n p) = 1 ∧
  m + n + p = 150 := by
  sorry

end quadratic_solution_and_gcd_sum_l1851_185160


namespace pool_fill_time_l1851_185144

/-- The time required to fill a pool, given its capacity and the water supply rate. -/
def fillTime (poolCapacity : ℚ) (numHoses : ℕ) (flowRatePerHose : ℚ) : ℚ :=
  poolCapacity / (numHoses * flowRatePerHose * 60)

/-- Theorem stating that the time to fill the pool is 100/3 hours. -/
theorem pool_fill_time :
  fillTime 36000 6 3 = 100 / 3 := by
  sorry

end pool_fill_time_l1851_185144


namespace parentheses_value_l1851_185152

theorem parentheses_value (x : ℚ) : x * (-2/3) = 2 → x = -3 := by
  sorry

end parentheses_value_l1851_185152


namespace smallest_hot_dog_packages_l1851_185148

theorem smallest_hot_dog_packages : ∃ n : ℕ, n > 0 ∧ (∀ m : ℕ, m > 0 → 5 * m % 7 = 0 → m ≥ n) ∧ 5 * n % 7 = 0 := by
  sorry

end smallest_hot_dog_packages_l1851_185148


namespace complex_equation_solution_l1851_185173

theorem complex_equation_solution (z : ℂ) : 
  z^2 + 2*Complex.I*z + 3 = 0 ↔ z = Complex.I ∨ z = -3*Complex.I :=
sorry

end complex_equation_solution_l1851_185173


namespace existence_of_close_pairs_l1851_185197

theorem existence_of_close_pairs :
  ∀ (a b : Fin 7 → ℝ),
  (∀ i, 0 ≤ a i) →
  (∀ i, 0 ≤ b i) →
  (∀ i, a i + b i ≤ 2) →
  ∃ k m, k ≠ m ∧ |a k - a m| + |b k - b m| ≤ 1 :=
by sorry

end existence_of_close_pairs_l1851_185197


namespace flash_interval_value_l1851_185170

/-- The number of flashes in ¾ of an hour -/
def flashes : ℕ := 240

/-- The duration in hours -/
def duration : ℚ := 3/4

/-- The number of seconds in an hour -/
def seconds_per_hour : ℕ := 3600

/-- The time interval between flashes in seconds -/
def flash_interval : ℚ := (duration * seconds_per_hour) / flashes

theorem flash_interval_value : flash_interval = 45/4 := by sorry

end flash_interval_value_l1851_185170


namespace xiao_liang_score_l1851_185161

/-- Calculates the comprehensive score for a speech contest given the weights and scores for each aspect. -/
def comprehensive_score (content_weight delivery_weight effectiveness_weight : ℚ)
                        (content_score delivery_score effectiveness_score : ℚ) : ℚ :=
  content_weight * content_score + delivery_weight * delivery_score + effectiveness_weight * effectiveness_score

/-- Theorem stating that Xiao Liang's comprehensive score is 91 points. -/
theorem xiao_liang_score :
  let content_weight : ℚ := 1/2
  let delivery_weight : ℚ := 2/5
  let effectiveness_weight : ℚ := 1/10
  let content_score : ℚ := 88
  let delivery_score : ℚ := 95
  let effectiveness_score : ℚ := 90
  comprehensive_score content_weight delivery_weight effectiveness_weight
                      content_score delivery_score effectiveness_score = 91 := by
  sorry


end xiao_liang_score_l1851_185161


namespace tan_alpha_plus_pi_sixth_l1851_185169

theorem tan_alpha_plus_pi_sixth (α : ℝ) 
  (h : Real.cos (3 * Real.pi / 2 - α) = 2 * Real.sin (α + Real.pi / 3)) : 
  Real.tan (α + Real.pi / 6) = -Real.sqrt 3 / 9 := by
sorry

end tan_alpha_plus_pi_sixth_l1851_185169


namespace isosceles_trapezoid_rotation_l1851_185107

-- Define an isosceles trapezoid
structure IsoscelesTrapezoid where
  base1 : ℝ
  base2 : ℝ
  height : ℝ
  is_isosceles : True
  base1_longer : base1 > base2

-- Define the rotation of the trapezoid
def rotate_trapezoid (t : IsoscelesTrapezoid) : Solid :=
  sorry

-- Define the components of a solid
inductive SolidComponent
  | Cylinder
  | Cone
  | Frustum

-- Define a solid as a collection of components
def Solid := List SolidComponent

-- Theorem statement
theorem isosceles_trapezoid_rotation 
  (t : IsoscelesTrapezoid) : 
  rotate_trapezoid t = [SolidComponent.Cylinder, SolidComponent.Cone, SolidComponent.Cone] :=
sorry

end isosceles_trapezoid_rotation_l1851_185107


namespace correct_multiplication_result_l1851_185101

theorem correct_multiplication_result : ∃ (n : ℕ), 
  (987 * n = 559989) ∧ 
  (∃ (a b : ℕ), 559981 = 550000 + a * 100 + b * 10 + 1 ∧ a ≠ 9 ∧ b ≠ 8) :=
by sorry

end correct_multiplication_result_l1851_185101


namespace union_empty_iff_both_empty_union_eq_diff_iff_empty_diff_eq_inter_iff_empty_l1851_185135

-- Define the universe set
variable {U : Type}

-- Define sets A, B, C as subsets of U
variable (A B C : Set U)

-- Theorem 1
theorem union_empty_iff_both_empty :
  A ∪ B = ∅ ↔ A = ∅ ∧ B = ∅ := by sorry

-- Theorem 2
theorem union_eq_diff_iff_empty :
  A ∪ B = A \ B ↔ B = ∅ := by sorry

-- Theorem 3
theorem diff_eq_inter_iff_empty :
  A \ B = A ∩ B ↔ A = ∅ := by sorry

-- Additional theorems can be added similarly for the remaining equivalences

end union_empty_iff_both_empty_union_eq_diff_iff_empty_diff_eq_inter_iff_empty_l1851_185135


namespace equivalent_discount_l1851_185133

theorem equivalent_discount (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) (equivalent_discount : ℝ) : 
  original_price = 50 →
  discount1 = 0.15 →
  discount2 = 0.10 →
  equivalent_discount = 0.235 →
  original_price * (1 - equivalent_discount) = 
  original_price * (1 - discount1) * (1 - discount2) := by
sorry

end equivalent_discount_l1851_185133


namespace expression_evaluation_l1851_185178

theorem expression_evaluation : 
  20 * ((150 / 3) + (40 / 5) + (16 / 25) + 2) = 1212.8 := by
  sorry

end expression_evaluation_l1851_185178


namespace smallest_sum_of_consecutive_integers_l1851_185141

theorem smallest_sum_of_consecutive_integers : ∃ n : ℕ,
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, 20 * m + 190 = 2 * k^2)) ∧
  (∃ k : ℕ, 20 * n + 190 = 2 * k^2) ∧
  20 * n + 190 = 450 :=
by sorry

end smallest_sum_of_consecutive_integers_l1851_185141


namespace candles_remaining_l1851_185185

def total_candles : ℕ := 40
def alyssa_fraction : ℚ := 1/2
def chelsea_fraction : ℚ := 7/10

theorem candles_remaining (total : ℕ) (alyssa_frac chelsea_frac : ℚ) :
  total = total_candles →
  alyssa_frac = alyssa_fraction →
  chelsea_frac = chelsea_fraction →
  ↑total * (1 - alyssa_frac) * (1 - chelsea_frac) = 6 :=
by sorry

end candles_remaining_l1851_185185


namespace club_officer_selection_l1851_185150

/-- Represents the number of ways to choose officers in a club --/
def choose_officers (total_members boys girls : ℕ) : ℕ :=
  let boy_president := boys * (boys - 1) * girls
  let girl_president := girls * (girls - 1) * boys
  boy_president + girl_president

/-- The main theorem stating the number of ways to choose officers --/
theorem club_officer_selection :
  choose_officers 24 14 10 = 3080 :=
by sorry

end club_officer_selection_l1851_185150


namespace jane_ice_cream_purchase_l1851_185125

/-- The number of ice cream cones Jane purchased -/
def num_ice_cream_cones : ℕ := 15

/-- The number of pudding cups Jane purchased -/
def num_pudding_cups : ℕ := 5

/-- The cost of one ice cream cone in dollars -/
def ice_cream_cost : ℕ := 5

/-- The cost of one pudding cup in dollars -/
def pudding_cost : ℕ := 2

/-- The difference in dollars between ice cream and pudding expenses -/
def expense_difference : ℕ := 65

theorem jane_ice_cream_purchase :
  num_ice_cream_cones * ice_cream_cost = num_pudding_cups * pudding_cost + expense_difference :=
by sorry

end jane_ice_cream_purchase_l1851_185125


namespace race_tie_l1851_185199

/-- A race between two runners A and B -/
structure Race where
  length : ℝ
  speed_ratio : ℝ
  head_start : ℝ

/-- The race conditions -/
def race_conditions : Race where
  length := 100
  speed_ratio := 4
  head_start := 75

/-- Theorem stating that the given head start results in a tie -/
theorem race_tie (race : Race) (h1 : race.length = 100) (h2 : race.speed_ratio = 4) 
    (h3 : race.head_start = 75) : 
  race.length / race.speed_ratio = (race.length - race.head_start) / 1 := by
  sorry

#check race_tie race_conditions rfl rfl rfl

end race_tie_l1851_185199


namespace cone_lateral_surface_area_l1851_185177

/-- The lateral surface area of a cone with base radius 6 and volume 30π is 39π -/
theorem cone_lateral_surface_area (r h l : ℝ) : 
  r = 6 → 
  (1/3) * π * r^2 * h = 30 * π → 
  l^2 = r^2 + h^2 → 
  π * r * l = 39 * π := by
sorry

end cone_lateral_surface_area_l1851_185177


namespace necklace_divisibility_l1851_185153

/-- The number of ways to make an even number of necklaces -/
def D₀ (n : ℕ) : ℕ := sorry

/-- The number of ways to make an odd number of necklaces -/
def D₁ (n : ℕ) : ℕ := sorry

/-- Theorem: n - 1 divides D₁(n) - D₀(n) for all n ≥ 2 -/
theorem necklace_divisibility (n : ℕ) (h : n ≥ 2) :
  ∃ k : ℤ, D₁ n - D₀ n = k * (n - 1) := by sorry

end necklace_divisibility_l1851_185153


namespace waiter_tables_count_l1851_185142

/-- Calculates the number of tables a waiter has based on customer information -/
def waiterTables (initialCustomers leavingCustomers peoplePerTable : ℕ) : ℕ :=
  (initialCustomers - leavingCustomers) / peoplePerTable

/-- Theorem stating that under the given conditions, the waiter had 5 tables -/
theorem waiter_tables_count :
  waiterTables 62 17 9 = 5 := by
  sorry

end waiter_tables_count_l1851_185142


namespace sum_of_roots_of_quadratic_l1851_185149

theorem sum_of_roots_of_quadratic : ∃ (x₁ x₂ : ℝ), 
  x₁^2 - 6*x₁ + 8 = 0 ∧ 
  x₂^2 - 6*x₂ + 8 = 0 ∧ 
  x₁ ≠ x₂ ∧
  x₁ + x₂ = 6 := by
  sorry

end sum_of_roots_of_quadratic_l1851_185149


namespace system_solutions_l1851_185184

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Define the system of equations
def system (x y : ℝ) : Prop :=
  lg (x^2 + y^2) = 2 - lg 5 ∧
  lg (x + y) + lg (x - y) = lg 1.2 + 1 ∧
  x + y > 0 ∧
  x - y > 0

-- Theorem statement
theorem system_solutions :
  ∀ x y : ℝ, system x y ↔ ((x = 4 ∧ y = 2) ∨ (x = 4 ∧ y = -2)) :=
by sorry

end system_solutions_l1851_185184


namespace rectangle_area_l1851_185146

theorem rectangle_area (b : ℝ) (h1 : b > 0) : 
  let l := 3 * b
  let perimeter := 2 * (l + b)
  perimeter = 64 → l * b = 192 := by
sorry

end rectangle_area_l1851_185146


namespace unique_perfect_square_solution_l1851_185190

theorem unique_perfect_square_solution : 
  ∃! (n : ℕ), n > 0 ∧ ∃ (m : ℕ), n^4 - n^3 + 3*n^2 + 5 = m^2 ∧ n = 2 := by
  sorry

end unique_perfect_square_solution_l1851_185190


namespace pizza_problem_l1851_185186

theorem pizza_problem (slices_per_pizza : ℕ) (games_played : ℕ) (avg_goals_per_game : ℕ) :
  slices_per_pizza = 12 →
  games_played = 8 →
  avg_goals_per_game = 9 →
  (games_played * avg_goals_per_game) / slices_per_pizza = 6 := by
  sorry

end pizza_problem_l1851_185186


namespace sara_oranges_l1851_185113

theorem sara_oranges (joan_oranges : ℕ) (total_oranges : ℕ) (h1 : joan_oranges = 37) (h2 : total_oranges = 47) :
  total_oranges - joan_oranges = 10 := by
  sorry

end sara_oranges_l1851_185113


namespace mashed_potatoes_count_l1851_185171

theorem mashed_potatoes_count :
  let bacon_count : ℕ := 42
  let difference : ℕ := 366
  let mashed_potatoes_count : ℕ := bacon_count + difference
  mashed_potatoes_count = 408 := by
sorry

end mashed_potatoes_count_l1851_185171


namespace four_in_B_iff_m_in_range_B_subset_A_iff_m_in_range_l1851_185104

-- Define the sets A and B
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

-- Part 1: 4 ∈ B iff m ∈ [5/2, 3]
theorem four_in_B_iff_m_in_range (m : ℝ) : 
  (4 ∈ B m) ↔ (5/2 ≤ m ∧ m ≤ 3) :=
sorry

-- Part 2: B ⊂ A iff m ∈ (-∞, 3]
theorem B_subset_A_iff_m_in_range (m : ℝ) :
  (B m ⊂ A) ↔ (m ≤ 3) :=
sorry

end four_in_B_iff_m_in_range_B_subset_A_iff_m_in_range_l1851_185104


namespace problem_statement_l1851_185172

theorem problem_statement (a b : ℝ) : 
  |a - 2| + (b + 1)^2 = 0 → 3*b - a = -5 := by sorry

end problem_statement_l1851_185172


namespace incorrect_multiplication_l1851_185139

theorem incorrect_multiplication (correct_multiplier : ℕ) (number_to_multiply : ℕ) (difference : ℕ) 
  (h1 : correct_multiplier = 43)
  (h2 : number_to_multiply = 134)
  (h3 : difference = 1206) :
  ∃ (incorrect_multiplier : ℕ), 
    number_to_multiply * correct_multiplier - number_to_multiply * incorrect_multiplier = difference ∧
    incorrect_multiplier = 34 :=
by sorry

end incorrect_multiplication_l1851_185139


namespace smallest_n_congruence_l1851_185130

theorem smallest_n_congruence (n : ℕ) : 
  (∀ k : ℕ, k > 0 ∧ k < n → ¬(6^k ≡ k^6 [ZMOD 3])) ∧ 
  (6^n ≡ n^6 [ZMOD 3]) → 
  n = 3 :=
sorry

end smallest_n_congruence_l1851_185130


namespace poster_placement_l1851_185110

/-- Given a wall of width 25 feet and a centrally placed poster of width 4 feet,
    the distance from the end of the wall to the nearest edge of the poster is 10.5 feet. -/
theorem poster_placement (wall_width : ℝ) (poster_width : ℝ) 
    (h1 : wall_width = 25) 
    (h2 : poster_width = 4) :
  (wall_width - poster_width) / 2 = 10.5 := by
  sorry

end poster_placement_l1851_185110


namespace fraction_power_and_multiply_l1851_185128

theorem fraction_power_and_multiply :
  (2 / 3 : ℚ) ^ 3 * (1 / 4 : ℚ) = 2 / 27 := by sorry

end fraction_power_and_multiply_l1851_185128


namespace harmonic_mean_closest_integer_l1851_185111

theorem harmonic_mean_closest_integer :
  ∃ (h : ℝ), 
    (h = 2 / ((1 : ℝ)⁻¹ + (2023 : ℝ)⁻¹)) ∧ 
    (∀ n : ℤ, n ≠ 2 → |h - 2| < |h - (n : ℝ)|) := by
  sorry

end harmonic_mean_closest_integer_l1851_185111


namespace louise_needs_30_boxes_l1851_185166

/-- Represents the number of pencils each box can hold for different colors --/
structure BoxCapacity where
  red : ℕ
  blue : ℕ
  yellow : ℕ
  green : ℕ

/-- Represents the number of pencils Louise has for each color --/
structure PencilCount where
  red : ℕ
  blue : ℕ
  yellow : ℕ
  green : ℕ

/-- Calculates the number of boxes needed for a given color --/
def boxesNeeded (capacity : ℕ) (count : ℕ) : ℕ :=
  (count + capacity - 1) / capacity

/-- Calculates the total number of boxes Louise needs --/
def totalBoxesNeeded (capacity : BoxCapacity) (count : PencilCount) : ℕ :=
  boxesNeeded capacity.red count.red +
  boxesNeeded capacity.blue count.blue +
  boxesNeeded capacity.yellow count.yellow +
  boxesNeeded capacity.green count.green

/-- The main theorem stating that Louise needs 30 boxes --/
theorem louise_needs_30_boxes :
  let capacity := BoxCapacity.mk 15 25 10 30
  let redCount := 45
  let blueCount := 3 * redCount + 6
  let yellowCount := 80
  let greenCount := 2 * (redCount + blueCount)
  let count := PencilCount.mk redCount blueCount yellowCount greenCount
  totalBoxesNeeded capacity count = 30 := by
  sorry


end louise_needs_30_boxes_l1851_185166


namespace polynomial_equality_l1851_185181

theorem polynomial_equality (a k n : ℚ) :
  (∀ x : ℚ, (3 * x + 2) * (2 * x - 3) = a * x^2 + k * x + n) →
  a - n + k = 7 := by
  sorry

end polynomial_equality_l1851_185181


namespace smallest_product_l1851_185183

def digits : List Nat := [1, 2, 3, 4]

def is_valid_arrangement (a b c d : Nat) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def product (a b c d : Nat) : Nat :=
  (10 * a + b) * (10 * c + d)

theorem smallest_product :
  ∀ a b c d : Nat,
    is_valid_arrangement a b c d →
    product a b c d ≥ 312 :=
by sorry

end smallest_product_l1851_185183


namespace cos_178_plus_theta_l1851_185187

theorem cos_178_plus_theta (θ : ℝ) (h : Real.sin (88 * π / 180 + θ) = 2/3) :
  Real.cos (178 * π / 180 + θ) = -2/3 := by
  sorry

end cos_178_plus_theta_l1851_185187


namespace prize_probability_l1851_185140

/-- The probability of at least one person winning a prize when 5 people each buy 1 ticket
    from a pool of 10 tickets, where 3 tickets have prizes. -/
theorem prize_probability (total_tickets : ℕ) (prize_tickets : ℕ) (buyers : ℕ) :
  total_tickets = 10 →
  prize_tickets = 3 →
  buyers = 5 →
  (1 : ℚ) - (Nat.choose (total_tickets - prize_tickets) buyers : ℚ) / (Nat.choose total_tickets buyers : ℚ) = 11/12 :=
by sorry

end prize_probability_l1851_185140


namespace x_value_l1851_185124

theorem x_value (x : ℝ) (h1 : x > 0) (h2 : (2 * x / 100) * x = 10) : x = 10 * Real.sqrt 5 := by
  sorry

end x_value_l1851_185124


namespace probability_eliminate_six_eq_seven_twentysix_l1851_185126

/-- Represents a team in the tournament -/
structure Team :=
  (players : ℕ)

/-- Represents the tournament structure -/
structure Tournament :=
  (teamA : Team)
  (teamB : Team)

/-- Calculates the binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- Calculates the probability of one team eliminating exactly 6 players before winning -/
def probability_eliminate_six (t : Tournament) : ℚ :=
  if t.teamA.players = 7 ∧ t.teamB.players = 7 then
    (binomial 12 6 : ℚ) / (2 * (binomial 13 7 : ℚ))
  else
    0

/-- Theorem stating the probability of eliminating 6 players before winning -/
theorem probability_eliminate_six_eq_seven_twentysix (t : Tournament) :
  t.teamA.players = 7 ∧ t.teamB.players = 7 →
  probability_eliminate_six t = 7 / 26 :=
sorry

end probability_eliminate_six_eq_seven_twentysix_l1851_185126


namespace incircle_radius_of_special_triangle_l1851_185102

/-- The radius of the incircle of a triangle with sides 5, 12, and 13 units is 2 units. -/
theorem incircle_radius_of_special_triangle : 
  ∀ (a b c r : ℝ), 
  a = 5 → b = 12 → c = 13 →
  r = (a * b) / (a + b + c) →
  r = 2 := by sorry

end incircle_radius_of_special_triangle_l1851_185102


namespace range_of_m_l1851_185188

theorem range_of_m (m : ℝ) : 
  (¬ (∃ m : ℝ, m + 1 ≤ 0) ∨ ¬ (∀ x : ℝ, x^2 + m*x + 1 > 0)) → 
  (m ≤ -2 ∨ m > -1) := by
  sorry

end range_of_m_l1851_185188


namespace rachels_homework_l1851_185194

theorem rachels_homework (total_pages math_pages : ℕ) 
  (h1 : total_pages = 7) 
  (h2 : math_pages = 5) : 
  total_pages - math_pages = 2 := by
sorry

end rachels_homework_l1851_185194


namespace AC_length_l1851_185151

/-- Right triangle ABC with altitude AH and circle through A and H -/
structure RightTriangleWithCircle where
  -- Points
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  H : ℝ × ℝ
  X : ℝ × ℝ
  Y : ℝ × ℝ
  -- ABC is a right triangle with right angle at A
  right_angle_at_A : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0
  -- AH is an altitude
  AH_perpendicular_BC : (H.1 - A.1) * (C.1 - B.1) + (H.2 - A.2) * (C.2 - B.2) = 0
  -- Circle passes through A, H, X, and Y
  circle_through_AHXY : ∃ (center : ℝ × ℝ) (radius : ℝ),
    (A.1 - center.1)^2 + (A.2 - center.2)^2 = radius^2 ∧
    (H.1 - center.1)^2 + (H.2 - center.2)^2 = radius^2 ∧
    (X.1 - center.1)^2 + (X.2 - center.2)^2 = radius^2 ∧
    (Y.1 - center.1)^2 + (Y.2 - center.2)^2 = radius^2
  -- X is on AB
  X_on_AB : ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ X = (A.1 + t * (B.1 - A.1), A.2 + t * (B.2 - A.2))
  -- Y is on AC
  Y_on_AC : ∃ (s : ℝ), 0 ≤ s ∧ s ≤ 1 ∧ Y = (A.1 + s * (C.1 - A.1), A.2 + s * (C.2 - A.2))
  -- Given lengths
  AX_length : ((X.1 - A.1)^2 + (X.2 - A.2)^2)^(1/2 : ℝ) = 5
  AY_length : ((Y.1 - A.1)^2 + (Y.2 - A.2)^2)^(1/2 : ℝ) = 6
  AB_length : ((B.1 - A.1)^2 + (B.2 - A.2)^2)^(1/2 : ℝ) = 9

/-- Theorem: AC length is 13.5 -/
theorem AC_length (t : RightTriangleWithCircle) : 
  ((t.C.1 - t.A.1)^2 + (t.C.2 - t.A.2)^2)^(1/2 : ℝ) = 13.5 := by
  sorry

end AC_length_l1851_185151


namespace total_nuts_is_half_cup_l1851_185189

/-- The amount of walnuts Karen added to the trail mix in cups -/
def walnuts : ℚ := 0.25

/-- The amount of almonds Karen added to the trail mix in cups -/
def almonds : ℚ := 0.25

/-- The total amount of nuts Karen added to the trail mix in cups -/
def total_nuts : ℚ := walnuts + almonds

/-- Theorem stating that the total amount of nuts Karen added is 0.50 cups -/
theorem total_nuts_is_half_cup : total_nuts = 0.50 := by sorry

end total_nuts_is_half_cup_l1851_185189


namespace trays_from_second_table_l1851_185147

theorem trays_from_second_table
  (trays_per_trip : ℕ)
  (num_trips : ℕ)
  (trays_from_first_table : ℕ)
  (h1 : trays_per_trip = 4)
  (h2 : num_trips = 3)
  (h3 : trays_from_first_table = 10) :
  trays_per_trip * num_trips - trays_from_first_table = 2 :=
by sorry

end trays_from_second_table_l1851_185147


namespace sum_lent_problem_l1851_185108

/-- Proves that given the conditions of the problem, the sum lent is 450 Rs. -/
theorem sum_lent_problem (P : ℝ) : 
  (P * 0.04 * 8 = P - 306) → P = 450 := by
  sorry

end sum_lent_problem_l1851_185108


namespace quadratic_roots_difference_l1851_185164

theorem quadratic_roots_difference (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    x₁^2 + p*x₁ + q = 0 ∧
    x₂^2 + p*x₂ + q = 0 ∧
    |x₁ - x₂| = 2) →
  p = 2 * Real.sqrt (q + 1) :=
by sorry

end quadratic_roots_difference_l1851_185164


namespace building_height_ratio_l1851_185138

/-- Proves the ratio of building heights given specific conditions -/
theorem building_height_ratio :
  let h₁ : ℝ := 600  -- Height of first building
  let h₂ : ℝ := 2 * h₁  -- Height of second building
  let h_total : ℝ := 7200  -- Total height of all three buildings
  let h₃ : ℝ := h_total - (h₁ + h₂)  -- Height of third building
  (h₃ / (h₁ + h₂) = 3) :=
by sorry

end building_height_ratio_l1851_185138


namespace anniversary_number_is_counting_l1851_185109

/-- Represents the categories of numbers in context --/
inductive NumberCategory
  | Label
  | MeasurementResult
  | Counting

/-- Represents the context in which the number is used --/
structure AnniversaryContext where
  years : ℕ

/-- Determines the category of a number in the anniversary context --/
def categorizeAnniversaryNumber (context : AnniversaryContext) : NumberCategory :=
  NumberCategory.Counting

/-- Theorem stating that the number used for anniversary years is a counting number --/
theorem anniversary_number_is_counting (context : AnniversaryContext) :
  categorizeAnniversaryNumber context = NumberCategory.Counting :=
by sorry

end anniversary_number_is_counting_l1851_185109


namespace books_read_l1851_185123

theorem books_read (total : ℕ) (remaining : ℕ) (read : ℕ) : 
  total = 14 → remaining = 6 → read = total - remaining → read = 8 := by
sorry

end books_read_l1851_185123


namespace pyramid_frustum_problem_l1851_185118

noncomputable def pyramid_frustum_theorem (AB BC height : ℝ) : Prop :=
  AB > 0 ∧ BC > 0 ∧ height > 0 →
  let ABCD := AB * BC
  let P_volume := (1/3) * ABCD * height
  let P'_volume := (1/8) * P_volume
  let F_height := height / 2
  let A'B' := AB / 2
  let B'C' := BC / 2
  let AC := Real.sqrt (AB^2 + BC^2)
  let A'C' := AC / 2
  let h := (73/8 : ℝ)
  let XT := h + F_height
  XT = 169/8

theorem pyramid_frustum_problem :
  pyramid_frustum_theorem 12 16 24 := by sorry

end pyramid_frustum_problem_l1851_185118


namespace first_train_speed_l1851_185154

/-- Proves that the speed of the first train is 45 kmph given the problem conditions --/
theorem first_train_speed (v : ℝ) : 
  v > 0 → -- The speed of the first train is positive
  (∃ t : ℝ, t > 0 ∧ v * (1 + t) = 90 ∧ 90 * t = 90) → -- Equations from the problem
  v = 45 := by
  sorry


end first_train_speed_l1851_185154


namespace unique_odd_number_with_eight_multiples_l1851_185191

theorem unique_odd_number_with_eight_multiples : 
  ∃! x : ℕ, 
    x % 2 = 1 ∧ 
    x > 0 ∧
    (∃ S : Finset ℕ, 
      S.card = 8 ∧
      (∀ y ∈ S, 
        y < 80 ∧ 
        y % 2 = 1 ∧
        ∃ k m : ℕ, k > 0 ∧ m % 2 = 1 ∧ y = k * x * m) ∧
      (∀ y : ℕ, 
        y < 80 → 
        y % 2 = 1 → 
        (∃ k m : ℕ, k > 0 ∧ m % 2 = 1 ∧ y = k * x * m) → 
        y ∈ S)) :=
sorry

end unique_odd_number_with_eight_multiples_l1851_185191


namespace max_product_divisible_by_55_l1851_185112

/-- Represents a four-digit number in the form 11,0ab -/
structure Number11_0ab where
  a : Nat
  b : Nat
  a_single_digit : a < 10
  b_single_digit : b < 10

/-- Check if a number in the form 11,0ab is divisible by 55 -/
def isDivisibleBy55 (n : Number11_0ab) : Prop :=
  (11000 + 100 * n.a + n.b) % 55 = 0

/-- The maximum product of a and b for numbers divisible by 55 -/
def maxProduct : Nat :=
  25

theorem max_product_divisible_by_55 :
  ∀ n : Number11_0ab, isDivisibleBy55 n → n.a * n.b ≤ maxProduct :=
by sorry

end max_product_divisible_by_55_l1851_185112


namespace algebraic_expression_value_l1851_185145

theorem algebraic_expression_value (x y : ℝ) 
  (eq1 : x + y = 0.2) 
  (eq2 : x + 3*y = 1) : 
  x^2 + 4*x*y + 4*y^2 = 0.36 := by
sorry

end algebraic_expression_value_l1851_185145


namespace min_value_shifted_l1851_185105

/-- The function f(x) = x^2 + 4x + 5 - c -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 + 4*x + 5 - c

/-- The shifted function f(x-2009) -/
def f_shifted (c : ℝ) (x : ℝ) : ℝ := f c (x - 2009)

theorem min_value_shifted (c : ℝ) :
  (∃ (m : ℝ), ∀ (x : ℝ), f c x ≥ m ∧ (∃ (x_0 : ℝ), f c x_0 = m) ∧ m = 2) →
  (∃ (m : ℝ), ∀ (x : ℝ), f_shifted c x ≥ m ∧ (∃ (x_0 : ℝ), f_shifted c x_0 = m) ∧ m = 2) :=
by sorry

end min_value_shifted_l1851_185105


namespace exam_score_problem_l1851_185119

theorem exam_score_problem (total_questions : ℕ) (correct_marks : ℕ) (wrong_marks : ℕ) (total_score : ℤ) :
  total_questions = 100 →
  correct_marks = 5 →
  wrong_marks = 2 →
  total_score = 210 →
  ∃ (correct_answers : ℕ),
    correct_answers ≤ total_questions ∧
    (correct_marks * correct_answers : ℤ) - (wrong_marks * (total_questions - correct_answers) : ℤ) = total_score ∧
    correct_answers = 58 :=
by sorry

end exam_score_problem_l1851_185119
