import Mathlib

namespace NUMINAMATH_CALUDE_line_intersects_parabola_once_l2390_239052

-- Define the point A
def A : ℝ × ℝ := (1, 2)

-- Define the parabola C
def C (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line L
def L (y : ℝ) : Prop := y = 2

-- Theorem statement
theorem line_intersects_parabola_once :
  L (A.2) ∧ 
  (∃! p : ℝ × ℝ, C p.1 p.2 ∧ L p.2) ∧
  (∀ y : ℝ, L y → ∃ x : ℝ, (x, y) = A ∨ C x y) :=
sorry

end NUMINAMATH_CALUDE_line_intersects_parabola_once_l2390_239052


namespace NUMINAMATH_CALUDE_smallest_period_scaled_function_l2390_239042

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem smallest_period_scaled_function
  (f : ℝ → ℝ) (h : is_periodic f 10) :
  ∃ b : ℝ, b > 0 ∧ (∀ x, f ((x - b) / 2) = f (x / 2)) ∧
    ∀ b' : ℝ, 0 < b' → (∀ x, f ((x - b') / 2) = f (x / 2)) → b ≤ b' :=
sorry

end NUMINAMATH_CALUDE_smallest_period_scaled_function_l2390_239042


namespace NUMINAMATH_CALUDE_consecutive_integers_square_sum_l2390_239021

theorem consecutive_integers_square_sum (a b c d e : ℕ) : 
  a > 0 → 
  b = a + 1 → 
  c = a + 2 → 
  d = a + 3 → 
  e = a + 4 → 
  a^2 + b^2 + c^2 = d^2 + e^2 → 
  a = 10 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_square_sum_l2390_239021


namespace NUMINAMATH_CALUDE_car_catch_up_time_l2390_239014

/-- The time it takes for a car to catch up with a truck, given their speeds and the truck's head start -/
theorem car_catch_up_time (truck_speed car_speed : ℝ) (head_start : ℝ) : 
  truck_speed = 45 →
  car_speed = 60 →
  head_start = 1 →
  (car_speed * t - truck_speed * t = truck_speed * head_start) →
  t = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_car_catch_up_time_l2390_239014


namespace NUMINAMATH_CALUDE_intersection_midpoint_l2390_239095

theorem intersection_midpoint (k : ℝ) : 
  (∃ A B : ℝ × ℝ, 
    (A.2 = A.1 - k ∧ A.1^2 = A.2) ∧ 
    (B.2 = B.1 - k ∧ B.1^2 = B.2) ∧ 
    A ≠ B ∧
    (A.2 + B.2) / 2 = 1) →
  k = -1/2 := by sorry

end NUMINAMATH_CALUDE_intersection_midpoint_l2390_239095


namespace NUMINAMATH_CALUDE_inequality_solution_l2390_239043

theorem inequality_solution (x : ℝ) : 
  (4 * x + 2 < (x - 1)^2 ∧ (x - 1)^2 < 5 * x + 5) ↔ 
  (x > 3 + Real.sqrt 10 ∧ x < (7 + Real.sqrt 65) / 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2390_239043


namespace NUMINAMATH_CALUDE_common_solution_y_values_l2390_239023

theorem common_solution_y_values : 
  ∀ x y : ℝ, 
  (x^2 + y^2 - 9 = 0 ∧ x^2 + 2*y - 7 = 0) ↔ 
  (y = 1 + Real.sqrt 3 ∨ y = 1 - Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_common_solution_y_values_l2390_239023


namespace NUMINAMATH_CALUDE_original_number_proof_l2390_239007

theorem original_number_proof : ∃! n : ℕ, 
  (n ≥ 1000 ∧ n < 10000) ∧ 
  (n / 1000 = 6) ∧
  (1000 * (n % 1000) + 6 = n - 1152) ∧
  n = 6538 := by
sorry

end NUMINAMATH_CALUDE_original_number_proof_l2390_239007


namespace NUMINAMATH_CALUDE_extreme_values_of_f_max_min_on_interval_parallel_tangents_midpoint_l2390_239017

/-- The function f(x) = x^3 - 12x + 12 --/
def f (x : ℝ) : ℝ := x^3 - 12*x + 12

/-- The derivative of f(x) --/
def f' (x : ℝ) : ℝ := 3*x^2 - 12

theorem extreme_values_of_f :
  (∃ x₁ x₂ : ℝ, x₁ = -2 ∧ x₂ = 2 ∧ 
   f x₁ = 28 ∧ f x₂ = -4 ∧
   ∀ x : ℝ, f x ≤ f x₁ ∧ f x₂ ≤ f x) :=
sorry

theorem max_min_on_interval :
  (∀ x : ℝ, -3 ≤ x ∧ x ≤ 4 → f x ≤ 28) ∧
  (∀ x : ℝ, -3 ≤ x ∧ x ≤ 4 → -4 ≤ f x) ∧
  (∃ x₁ x₂ : ℝ, -3 ≤ x₁ ∧ x₁ ≤ 4 ∧ -3 ≤ x₂ ∧ x₂ ≤ 4 ∧ f x₁ = 28 ∧ f x₂ = -4) :=
sorry

theorem parallel_tangents_midpoint :
  ∀ a b : ℝ, f' a = f' b →
  (a + b) / 2 = 0 ∧ (f a + f b) / 2 = 12 :=
sorry

end NUMINAMATH_CALUDE_extreme_values_of_f_max_min_on_interval_parallel_tangents_midpoint_l2390_239017


namespace NUMINAMATH_CALUDE_equal_selection_probability_l2390_239063

/-- Represents a sampling method -/
inductive SamplingMethod
  | SimpleRandom
  | Stratified
  | Systematic

/-- Represents the population and sample characteristics -/
structure Population where
  total_items : ℕ
  first_grade : ℕ
  second_grade : ℕ
  third_grade : ℕ
  fourth_grade : ℕ
  sample_size : ℕ

/-- Calculates the probability of an item being selected for a given sampling method -/
def selection_probability (pop : Population) (method : SamplingMethod) : ℚ :=
  pop.sample_size / pop.total_items

/-- The main theorem stating that all sampling methods have the same selection probability -/
theorem equal_selection_probability (pop : Population) 
  (h1 : pop.total_items = 160)
  (h2 : pop.first_grade = 48)
  (h3 : pop.second_grade = 64)
  (h4 : pop.third_grade = 32)
  (h5 : pop.fourth_grade = 16)
  (h6 : pop.sample_size = 20)
  (h7 : pop.total_items = pop.first_grade + pop.second_grade + pop.third_grade + pop.fourth_grade) :
  ∀ m : SamplingMethod, selection_probability pop m = 1/8 := by
  sorry

#check equal_selection_probability

end NUMINAMATH_CALUDE_equal_selection_probability_l2390_239063


namespace NUMINAMATH_CALUDE_maggie_bouncy_balls_l2390_239027

/-- The number of packs of red bouncy balls -/
def red_packs : ℕ := 6

/-- The number of balls in each pack of red bouncy balls -/
def red_balls_per_pack : ℕ := 12

/-- The number of packs of yellow bouncy balls -/
def yellow_packs : ℕ := 10

/-- The number of balls in each pack of yellow bouncy balls -/
def yellow_balls_per_pack : ℕ := 8

/-- The number of packs of green bouncy balls -/
def green_packs : ℕ := 4

/-- The number of balls in each pack of green bouncy balls -/
def green_balls_per_pack : ℕ := 15

/-- The number of packs of blue bouncy balls -/
def blue_packs : ℕ := 3

/-- The number of balls in each pack of blue bouncy balls -/
def blue_balls_per_pack : ℕ := 20

/-- The total number of bouncy balls Maggie bought -/
def total_bouncy_balls : ℕ := 
  red_packs * red_balls_per_pack + 
  yellow_packs * yellow_balls_per_pack + 
  green_packs * green_balls_per_pack + 
  blue_packs * blue_balls_per_pack

theorem maggie_bouncy_balls : total_bouncy_balls = 272 := by
  sorry

end NUMINAMATH_CALUDE_maggie_bouncy_balls_l2390_239027


namespace NUMINAMATH_CALUDE_assignment_effect_l2390_239065

/-- Represents the effect of the assignment statement M = M + 3 --/
theorem assignment_effect (M : ℤ) : 
  let M' := M + 3
  M' = M + 3 := by sorry

end NUMINAMATH_CALUDE_assignment_effect_l2390_239065


namespace NUMINAMATH_CALUDE_rowing_distance_l2390_239002

/-- Calculates the total distance traveled by a man rowing in a river --/
theorem rowing_distance (v_man : ℝ) (v_river : ℝ) (total_time : ℝ) : 
  v_man = 7 →
  v_river = 1.2 →
  total_time = 1 →
  let d := (v_man^2 - v_river^2) * total_time / (2 * v_man)
  2 * d = 7 := by sorry

end NUMINAMATH_CALUDE_rowing_distance_l2390_239002


namespace NUMINAMATH_CALUDE_ratio_equality_l2390_239087

theorem ratio_equality (x : ℝ) :
  (0.75 / x = 5 / 8) → x = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l2390_239087


namespace NUMINAMATH_CALUDE_intersection_point_l2390_239066

-- Define the curves C₁ and C₂
def C₁ (x y : ℝ) : Prop := x + y + 2 = 0

def C₂ (x y : ℝ) : Prop := y^2 = 8*x

-- State the theorem
theorem intersection_point :
  ∃! p : ℝ × ℝ, C₁ p.1 p.2 ∧ C₂ p.1 p.2 ∧ p = (2, -4) := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_l2390_239066


namespace NUMINAMATH_CALUDE_tetrahedron_volume_l2390_239099

/-- The volume of a regular tetrahedron with given base side length and angle between lateral face and base. -/
theorem tetrahedron_volume 
  (base_side : ℝ) 
  (lateral_angle : ℝ) 
  (h : base_side = Real.sqrt 3) 
  (θ : lateral_angle = π / 3) : 
  (1 / 3 : ℝ) * base_side ^ 2 * (base_side / 2) / Real.tan lateral_angle = 1 / 2 := by
  sorry

#check tetrahedron_volume

end NUMINAMATH_CALUDE_tetrahedron_volume_l2390_239099


namespace NUMINAMATH_CALUDE_expression_equality_l2390_239034

theorem expression_equality (x : ℝ) (h : 3 * x^2 - 6 * x + 4 = 7) : 
  x^2 - 2 * x + 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2390_239034


namespace NUMINAMATH_CALUDE_quadratic_inequality_and_distance_comparison_l2390_239005

theorem quadratic_inequality_and_distance_comparison :
  (∀ (k : ℝ), (∀ (x : ℝ), 2 * k * x^2 + k * x - 3/8 < 0) ↔ (k > -3 ∧ k ≤ 0)) ∧
  (∀ (a b : ℝ), a ≠ b → |(a^2 + b^2)/2 - ((a+b)/2)^2| > |a*b - ((a+b)/2)^2|) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_and_distance_comparison_l2390_239005


namespace NUMINAMATH_CALUDE_no_prime_sum_10003_l2390_239025

theorem no_prime_sum_10003 : ¬∃ p q : ℕ, Prime p ∧ Prime q ∧ p + q = 10003 := by
  sorry

end NUMINAMATH_CALUDE_no_prime_sum_10003_l2390_239025


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l2390_239040

/-- Given a hyperbola with equation x²/a² - y² = 1 and an asymptote √3x + y = 0,
    prove that a = √3/3 -/
theorem hyperbola_asymptote (a : ℝ) : 
  (∃ x y : ℝ, x^2/a^2 - y^2 = 1) ∧ 
  (∃ x y : ℝ, Real.sqrt 3 * x + y = 0) → 
  a = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l2390_239040


namespace NUMINAMATH_CALUDE_negation_equivalence_l2390_239092

theorem negation_equivalence :
  (¬ ∀ (a b : ℝ), ab > 0 → a > 0) ↔ (∀ (a b : ℝ), ab ≤ 0 → a ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2390_239092


namespace NUMINAMATH_CALUDE_paper_towel_package_rolls_l2390_239045

/-- Given a package of paper towels with the following properties:
  * The package price is $9
  * The individual roll price is $1
  * The savings per roll in the package is 25% compared to individual purchase
  Prove that the number of rolls in the package is 12 -/
theorem paper_towel_package_rolls : 
  ∀ (package_price individual_price : ℚ) (savings_percent : ℚ) (num_rolls : ℕ),
  package_price = 9 →
  individual_price = 1 →
  savings_percent = 25 / 100 →
  package_price = num_rolls * (individual_price * (1 - savings_percent)) →
  num_rolls = 12 := by
sorry

end NUMINAMATH_CALUDE_paper_towel_package_rolls_l2390_239045


namespace NUMINAMATH_CALUDE_prob_three_common_books_l2390_239053

/-- The number of books in Mr. Johnson's list -/
def total_books : ℕ := 12

/-- The number of books each student must choose -/
def books_to_choose : ℕ := 5

/-- The number of common books we're interested in -/
def common_books : ℕ := 3

/-- The probability of Alice and Bob selecting exactly 3 common books -/
def prob_common_books : ℚ := 55 / 209

theorem prob_three_common_books :
  (Nat.choose total_books common_books *
   Nat.choose (total_books - common_books) (books_to_choose - common_books) *
   Nat.choose (total_books - books_to_choose) (books_to_choose - common_books)) /
  (Nat.choose total_books books_to_choose)^2 = prob_common_books :=
sorry

end NUMINAMATH_CALUDE_prob_three_common_books_l2390_239053


namespace NUMINAMATH_CALUDE_symmetry_wrt_origin_l2390_239004

/-- Given a point P(3, 2) in the Cartesian coordinate system, 
    its symmetrical point P' with respect to the origin has coordinates (-3, -2). -/
theorem symmetry_wrt_origin :
  let P : ℝ × ℝ := (3, 2)
  let P' : ℝ × ℝ := (-P.1, -P.2)
  P' = (-3, -2) := by sorry

end NUMINAMATH_CALUDE_symmetry_wrt_origin_l2390_239004


namespace NUMINAMATH_CALUDE_bricklayers_theorem_l2390_239055

/-- Represents the problem of two bricklayers building a wall --/
structure BricklayersProblem where
  total_bricks : ℕ
  time_first : ℕ
  time_second : ℕ
  joint_decrease : ℕ
  joint_time : ℕ

/-- The solution to the bricklayers problem --/
def solve_bricklayers_problem (p : BricklayersProblem) : Prop :=
  p.total_bricks = 288 ∧
  p.time_first = 8 ∧
  p.time_second = 12 ∧
  p.joint_decrease = 12 ∧
  p.joint_time = 6 ∧
  (p.total_bricks / p.time_first + p.total_bricks / p.time_second - p.joint_decrease) * p.joint_time = p.total_bricks

theorem bricklayers_theorem (p : BricklayersProblem) :
  solve_bricklayers_problem p :=
sorry

end NUMINAMATH_CALUDE_bricklayers_theorem_l2390_239055


namespace NUMINAMATH_CALUDE_modulo_equivalence_unique_solution_l2390_239060

theorem modulo_equivalence_unique_solution : ∃! n : ℤ, 0 ≤ n ∧ n ≤ 11 ∧ n ≡ 15827 [ZMOD 12] := by
  sorry

end NUMINAMATH_CALUDE_modulo_equivalence_unique_solution_l2390_239060


namespace NUMINAMATH_CALUDE_line_equation_l2390_239077

/-- The equation of a line passing through the intersection of two given lines and parallel to a third line -/
theorem line_equation (x y : ℝ) : 
  (2 * x - 3 * y - 3 = 0) →   -- First given line
  (x + y + 2 = 0) →           -- Second given line
  (∃ k : ℝ, 3 * x + y - k = 0) →  -- Parallel line condition
  (15 * x + 5 * y + 16 = 0) := by  -- Equation to prove
sorry

end NUMINAMATH_CALUDE_line_equation_l2390_239077


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2390_239084

/-- Hyperbola eccentricity theorem -/
theorem hyperbola_eccentricity 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (h_equation : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1)
  (h_asymptote : ∀ x, ∃ y, y = Real.sqrt 3 / 3 * x ∨ y = -Real.sqrt 3 / 3 * x) :
  let c := Real.sqrt (a^2 + b^2)
  c / a = 2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2390_239084


namespace NUMINAMATH_CALUDE_inequality_problem_l2390_239038

theorem inequality_problem (m : ℝ) (h : ∀ x : ℝ, |x - 2| + |x - 3| ≥ m) :
  (∃ k : ℝ, k = 1 ∧ (∀ m' : ℝ, (∀ x : ℝ, |x - 2| + |x - 3| ≥ m') → m' ≤ k)) ∧
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → 1/a + 1/(2*b) + 1/(3*c) = 1 → a + 2*b + 3*c ≥ 9) :=
by sorry

end NUMINAMATH_CALUDE_inequality_problem_l2390_239038


namespace NUMINAMATH_CALUDE_race_track_inner_circumference_l2390_239029

/-- Given a circular race track with an outer radius of 140.0563499208679 m and a width of 18 m, 
    the inner circumference is approximately 767.145882893066 m. -/
theorem race_track_inner_circumference :
  let outer_radius : ℝ := 140.0563499208679
  let track_width : ℝ := 18
  let inner_radius : ℝ := outer_radius - track_width
  let inner_circumference : ℝ := 2 * Real.pi * inner_radius
  ∃ ε > 0, abs (inner_circumference - 767.145882893066) < ε :=
by sorry

end NUMINAMATH_CALUDE_race_track_inner_circumference_l2390_239029


namespace NUMINAMATH_CALUDE_emilys_necklaces_l2390_239097

/-- Emily's necklace-making problem -/
theorem emilys_necklaces (necklaces : ℕ) (beads_per_necklace : ℕ) (total_beads : ℕ) 
  (h1 : necklaces = 26)
  (h2 : beads_per_necklace = 2)
  (h3 : total_beads = 52)
  (h4 : necklaces * beads_per_necklace = total_beads) :
  necklaces = total_beads / beads_per_necklace :=
by sorry

end NUMINAMATH_CALUDE_emilys_necklaces_l2390_239097


namespace NUMINAMATH_CALUDE_count_k_eq_1006_l2390_239062

/-- The number of positive integers k such that (k/2013)(a+b) = lcm(a,b) has a solution in positive integers (a,b) -/
def count_k : ℕ := sorry

/-- The equation (k/2013)(a+b) = lcm(a,b) has a solution in positive integers (a,b) -/
def has_solution (k : ℕ+) : Prop :=
  ∃ (a b : ℕ+), (k : ℚ) / 2013 * (a + b) = Nat.lcm a b

theorem count_k_eq_1006 : count_k = 1006 := by sorry

end NUMINAMATH_CALUDE_count_k_eq_1006_l2390_239062


namespace NUMINAMATH_CALUDE_unique_k_for_prime_roots_l2390_239015

-- Define a function to check if a number is prime
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

-- Define the quadratic equation
def quadraticEquation (x k : ℤ) : Prop := x^2 - 99*x + k = 0

-- Define a function to check if both roots are prime
def bothRootsPrime (k : ℤ) : Prop :=
  ∃ p q : ℤ, 
    quadraticEquation p k ∧ 
    quadraticEquation q k ∧ 
    p ≠ q ∧ 
    isPrime p.natAbs ∧ 
    isPrime q.natAbs

-- Theorem statement
theorem unique_k_for_prime_roots : 
  ∃! k : ℤ, bothRootsPrime k :=
sorry

end NUMINAMATH_CALUDE_unique_k_for_prime_roots_l2390_239015


namespace NUMINAMATH_CALUDE_domain_of_function_l2390_239088

/-- The domain of the function f(x) = √(x - 1) + ∛(8 - x) is [1, 8] -/
theorem domain_of_function (f : ℝ → ℝ) (h : f = fun x ↦ Real.sqrt (x - 1) + (8 - x) ^ (1/3)) :
  Set.Icc 1 8 = {x : ℝ | ∃ y, f x = y} := by
  sorry

end NUMINAMATH_CALUDE_domain_of_function_l2390_239088


namespace NUMINAMATH_CALUDE_matrix_determinant_equals_four_l2390_239082

def A (y : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := ![![3*y, 2], ![3, y]]

theorem matrix_determinant_equals_four (y : ℝ) :
  Matrix.det (A y) = 4 ↔ y = Real.sqrt (10/3) ∨ y = -Real.sqrt (10/3) := by
  sorry

end NUMINAMATH_CALUDE_matrix_determinant_equals_four_l2390_239082


namespace NUMINAMATH_CALUDE_cab_driver_average_income_l2390_239058

/-- Calculates the average daily income of a cab driver over 5 days --/
theorem cab_driver_average_income :
  let day1_earnings := 250
  let day1_commission_rate := 0.1
  let day2_earnings := 400
  let day2_expense := 50
  let day3_earnings := 750
  let day3_commission_rate := 0.15
  let day4_earnings := 400
  let day4_expense := 40
  let day5_earnings := 500
  let day5_commission_rate := 0.2
  let total_days := 5
  let total_net_income := 
    (day1_earnings * (1 - day1_commission_rate)) +
    (day2_earnings - day2_expense) +
    (day3_earnings * (1 - day3_commission_rate)) +
    (day4_earnings - day4_expense) +
    (day5_earnings * (1 - day5_commission_rate))
  let average_daily_income := total_net_income / total_days
  average_daily_income = 394.50 := by
sorry


end NUMINAMATH_CALUDE_cab_driver_average_income_l2390_239058


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l2390_239022

theorem solution_set_quadratic_inequality (x : ℝ) :
  x * (x - 1) > 0 ↔ x < 0 ∨ x > 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l2390_239022


namespace NUMINAMATH_CALUDE_jacks_change_jacks_change_is_five_l2390_239093

/-- Given Jack's sandwich order and payment, calculate his change -/
theorem jacks_change (num_sandwiches : ℕ) (price_per_sandwich : ℕ) (payment : ℕ) : ℕ :=
  let total_cost := num_sandwiches * price_per_sandwich
  payment - total_cost

/-- Prove that Jack's change is $5 given the problem conditions -/
theorem jacks_change_is_five : 
  jacks_change 3 5 20 = 5 := by
  sorry

end NUMINAMATH_CALUDE_jacks_change_jacks_change_is_five_l2390_239093


namespace NUMINAMATH_CALUDE_league_matches_count_l2390_239079

/-- The number of teams in the league -/
def num_teams : ℕ := 14

/-- The number of matches played in the league -/
def total_matches : ℕ := num_teams * (num_teams - 1)

/-- Theorem stating that the total number of matches in the league is 182 -/
theorem league_matches_count :
  total_matches = 182 :=
sorry

end NUMINAMATH_CALUDE_league_matches_count_l2390_239079


namespace NUMINAMATH_CALUDE_factorization_proof_l2390_239086

theorem factorization_proof (a b x y m : ℝ) : 
  (3 * a^2 - 6 * a * b + 3 * b^2 = 3 * (a - b)^2) ∧ 
  (x^2 * (m - 2) + y^2 * (2 - m) = (m - 2) * (x + y) * (x - y)) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l2390_239086


namespace NUMINAMATH_CALUDE_find_divisor_l2390_239013

theorem find_divisor (n : ℕ) (d : ℕ) : 
  n % 44 = 0 ∧ n / 44 = 432 ∧ n % d = 8 → d = 50 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l2390_239013


namespace NUMINAMATH_CALUDE_inequality_proof_l2390_239037

theorem inequality_proof (a b c d e f : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : 0 ≤ d) (h5 : 0 ≤ e) (h6 : 0 ≤ f)
  (h7 : a + b ≤ e) (h8 : c + d ≤ f) : 
  Real.sqrt (a * c) + Real.sqrt (b * d) ≤ Real.sqrt (e * f) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2390_239037


namespace NUMINAMATH_CALUDE_penny_to_nickel_ratio_l2390_239041

/-- Represents the number of coins of each type -/
structure CoinCounts where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ
  quarters : ℕ

/-- Calculates the total value of coins in cents -/
def totalValue (coins : CoinCounts) : ℕ :=
  coins.pennies + 5 * coins.nickels + 10 * coins.dimes + 25 * coins.quarters

/-- The main theorem stating the ratio of pennies to nickels -/
theorem penny_to_nickel_ratio (coins : CoinCounts) :
  coins.pennies = 120 ∧
  coins.nickels = 5 * coins.dimes ∧
  coins.quarters = 2 * coins.dimes ∧
  totalValue coins = 800 →
  coins.pennies / coins.nickels = 3 :=
by sorry

end NUMINAMATH_CALUDE_penny_to_nickel_ratio_l2390_239041


namespace NUMINAMATH_CALUDE_magnitude_of_complex_square_l2390_239000

theorem magnitude_of_complex_square : Complex.abs ((3 - 4*Complex.I)^2) = 25 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_square_l2390_239000


namespace NUMINAMATH_CALUDE_subset_with_fourth_power_product_l2390_239020

theorem subset_with_fourth_power_product 
  (M : Finset ℕ+) 
  (distinct : M.card = 1985) 
  (prime_bound : ∀ n ∈ M, ∀ p : ℕ, p.Prime → p ∣ n → p ≤ 23) :
  ∃ (a b c d : ℕ+), a ∈ M ∧ b ∈ M ∧ c ∈ M ∧ d ∈ M ∧ 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  ∃ (m : ℕ+), a * b * c * d = m ^ 4 :=
sorry

end NUMINAMATH_CALUDE_subset_with_fourth_power_product_l2390_239020


namespace NUMINAMATH_CALUDE_correct_quadratic_equation_l2390_239064

-- Define the quadratic equation
def quadratic_equation (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c

-- Define the roots from the first student's mistake
def root1 : ℝ := 3
def root2 : ℝ := 7

-- Define the roots from the second student's mistake
def root3 : ℝ := 5
def root4 : ℝ := -1

-- Theorem statement
theorem correct_quadratic_equation :
  ∃ (b c : ℝ),
    (root1 + root2 = -b) ∧
    (root3 * root4 = c) ∧
    (∀ x, quadratic_equation b c x = x^2 - 10*x - 5) :=
sorry

end NUMINAMATH_CALUDE_correct_quadratic_equation_l2390_239064


namespace NUMINAMATH_CALUDE_ball_cost_l2390_239059

/-- Given that Kyoko paid $4.62 for 3 balls, prove that each ball costs $1.54. -/
theorem ball_cost (total_paid : ℝ) (num_balls : ℕ) (h1 : total_paid = 4.62) (h2 : num_balls = 3) :
  total_paid / num_balls = 1.54 := by
sorry

end NUMINAMATH_CALUDE_ball_cost_l2390_239059


namespace NUMINAMATH_CALUDE_revenue_maximized_at_064_l2390_239054

/-- Revenue function for electricity pricing -/
def revenue (x : ℝ) : ℝ := (1 + 50 * (x - 0.8)^2) * (x - 0.5)

/-- The domain of the revenue function -/
def price_range (x : ℝ) : Prop := 0.5 < x ∧ x < 0.8

theorem revenue_maximized_at_064 :
  ∃ (x : ℝ), price_range x ∧
    (∀ (y : ℝ), price_range y → revenue y ≤ revenue x) ∧
    x = 0.64 :=
sorry

end NUMINAMATH_CALUDE_revenue_maximized_at_064_l2390_239054


namespace NUMINAMATH_CALUDE_jose_peanut_count_l2390_239039

-- Define the number of peanuts Kenya has
def kenya_peanuts : ℕ := 133

-- Define the difference between Kenya's and Jose's peanuts
def peanut_difference : ℕ := 48

-- Define Jose's peanuts
def jose_peanuts : ℕ := kenya_peanuts - peanut_difference

-- Theorem statement
theorem jose_peanut_count : jose_peanuts = 85 := by sorry

end NUMINAMATH_CALUDE_jose_peanut_count_l2390_239039


namespace NUMINAMATH_CALUDE_smallest_resolvable_debt_l2390_239019

/-- The value of a pig in dollars -/
def pig_value : ℕ := 500

/-- The value of a goat in dollars -/
def goat_value : ℕ := 350

/-- The smallest positive debt that can be resolved -/
def smallest_debt : ℕ := 50

theorem smallest_resolvable_debt :
  smallest_debt = Nat.gcd pig_value goat_value ∧
  ∃ (p g : ℤ), smallest_debt = p * pig_value + g * goat_value :=
sorry

end NUMINAMATH_CALUDE_smallest_resolvable_debt_l2390_239019


namespace NUMINAMATH_CALUDE_price_reduction_effect_l2390_239009

theorem price_reduction_effect (price_reduction : ℝ) (revenue_increase : ℝ) (sales_increase : ℝ) : 
  price_reduction = 30 →
  revenue_increase = 26 →
  (1 - price_reduction / 100) * (1 + sales_increase / 100) = 1 + revenue_increase / 100 →
  sales_increase = 80 := by
sorry

end NUMINAMATH_CALUDE_price_reduction_effect_l2390_239009


namespace NUMINAMATH_CALUDE_wrong_multiplication_correction_l2390_239028

theorem wrong_multiplication_correction (x : ℝ) (h : x * 2.4 = 288) : (x / 2.4) / 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_wrong_multiplication_correction_l2390_239028


namespace NUMINAMATH_CALUDE_integer_sqrt_divisibility_l2390_239018

theorem integer_sqrt_divisibility (n : ℕ) (h1 : n ≥ 4) :
  (Int.floor (Real.sqrt n) + 1 ∣ n - 1) ∧
  (Int.floor (Real.sqrt n) - 1 ∣ n + 1) →
  n = 4 ∨ n = 7 ∨ n = 9 ∨ n = 13 ∨ n = 31 := by
  sorry

end NUMINAMATH_CALUDE_integer_sqrt_divisibility_l2390_239018


namespace NUMINAMATH_CALUDE_parallelogram_area_18_16_l2390_239031

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 18 cm and height 16 cm is 288 square centimeters -/
theorem parallelogram_area_18_16 : parallelogram_area 18 16 = 288 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_18_16_l2390_239031


namespace NUMINAMATH_CALUDE_age_difference_l2390_239094

/-- Given three people a, b, and c, with their ages satisfying certain conditions,
    prove that a is 2 years older than b. -/
theorem age_difference (a b c : ℕ) : 
  b = 28 →                  -- b is 28 years old
  b = 2 * c →               -- b is twice as old as c
  a + b + c = 72 →          -- The total of the ages of a, b, and c is 72
  a = b + 2 :=              -- a is 2 years older than b
by
  sorry

end NUMINAMATH_CALUDE_age_difference_l2390_239094


namespace NUMINAMATH_CALUDE_shopkeeper_loss_percent_l2390_239089

theorem shopkeeper_loss_percent
  (initial_value : ℝ)
  (profit_margin : ℝ)
  (theft_percentage : ℝ)
  (h_profit : profit_margin = 0.1)
  (h_theft : theft_percentage = 0.6)
  (h_initial_positive : initial_value > 0) :
  let selling_price := initial_value * (1 + profit_margin)
  let remaining_value := initial_value * (1 - theft_percentage)
  let remaining_selling_price := selling_price * (1 - theft_percentage)
  let loss := initial_value - remaining_selling_price
  let loss_percent := (loss / initial_value) * 100
  loss_percent = 56 := by
sorry

end NUMINAMATH_CALUDE_shopkeeper_loss_percent_l2390_239089


namespace NUMINAMATH_CALUDE_unique_solution_l2390_239080

/-- Represents the number of children in each family and the house number -/
structure FamilyData where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  N : ℕ

/-- The conditions given in the problem -/
def satisfiesConditions (fd : FamilyData) : Prop :=
  fd.a > fd.b ∧ fd.b > fd.c ∧ fd.c > fd.d ∧
  fd.a + fd.b + fd.c + fd.d < 18 ∧
  fd.a * fd.b * fd.c * fd.d = fd.N

/-- The theorem statement -/
theorem unique_solution :
  ∃! fd : FamilyData, satisfiesConditions fd ∧ fd.N = 120 ∧
    fd.a = 5 ∧ fd.b = 4 ∧ fd.c = 3 ∧ fd.d = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l2390_239080


namespace NUMINAMATH_CALUDE_square_to_rectangle_area_increase_l2390_239033

theorem square_to_rectangle_area_increase : 
  ∀ (a : ℝ), a > 0 →
  let original_area := a * a
  let new_length := a * 1.4
  let new_breadth := a * 1.3
  let new_area := new_length * new_breadth
  (new_area - original_area) / original_area = 0.82 := by
sorry

end NUMINAMATH_CALUDE_square_to_rectangle_area_increase_l2390_239033


namespace NUMINAMATH_CALUDE_solve_equation1_solve_equation2_l2390_239006

-- Define the quadratic equations
def equation1 (x : ℝ) : Prop := x^2 - 2*x - 8 = 0
def equation2 (x : ℝ) : Prop := x^2 - 2*x - 5 = 0

-- Theorem for the first equation
theorem solve_equation1 : 
  ∃ x1 x2 : ℝ, x1 = 4 ∧ x2 = -2 ∧ equation1 x1 ∧ equation1 x2 ∧
  ∀ x : ℝ, equation1 x → x = x1 ∨ x = x2 :=
sorry

-- Theorem for the second equation
theorem solve_equation2 : 
  ∃ x1 x2 : ℝ, x1 = 1 + Real.sqrt 6 ∧ x2 = 1 - Real.sqrt 6 ∧ 
  equation2 x1 ∧ equation2 x2 ∧
  ∀ x : ℝ, equation2 x → x = x1 ∨ x = x2 :=
sorry

end NUMINAMATH_CALUDE_solve_equation1_solve_equation2_l2390_239006


namespace NUMINAMATH_CALUDE_bus_departure_interval_l2390_239008

/-- Represents the speed of individual B -/
def speed_B : ℝ := 1

/-- Represents the speed of individual A -/
def speed_A : ℝ := 3 * speed_B

/-- Represents the time interval (in minutes) at which buses overtake A -/
def overtake_time_A : ℝ := 10

/-- Represents the time interval (in minutes) at which buses overtake B -/
def overtake_time_B : ℝ := 6

/-- Represents the speed of the buses -/
def speed_bus : ℝ := speed_A + speed_B

theorem bus_departure_interval (t : ℝ) :
  (t = overtake_time_A ∧ speed_bus * t = speed_A * overtake_time_A + speed_bus * overtake_time_A) ∧
  (t = overtake_time_B ∧ speed_bus * t = speed_B * overtake_time_B + speed_bus * overtake_time_B) →
  t = 5 := by sorry

end NUMINAMATH_CALUDE_bus_departure_interval_l2390_239008


namespace NUMINAMATH_CALUDE_inequality_proof_l2390_239070

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x ≥ y + z) :
  (x + y) / z + (y + z) / x + (z + x) / y ≥ 7 ∧
  ((x + y) / z + (y + z) / x + (z + x) / y = 7 ↔ x = 2 * y ∧ y = z) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2390_239070


namespace NUMINAMATH_CALUDE_sqrt_product_simplification_l2390_239073

theorem sqrt_product_simplification (p : ℝ) :
  Real.sqrt (8 * p^2) * Real.sqrt (12 * p^3) * Real.sqrt (18 * p^5) = 24 * p^5 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_simplification_l2390_239073


namespace NUMINAMATH_CALUDE_min_coins_for_dollar_l2390_239090

/-- Represents the different types of coins available --/
inductive Coin
  | Penny
  | Nickel
  | Dime
  | Quarter
  | HalfDollar

/-- Returns the value of a coin in cents --/
def coinValue (c : Coin) : ℕ :=
  match c with
  | Coin.Penny => 1
  | Coin.Nickel => 5
  | Coin.Dime => 10
  | Coin.Quarter => 25
  | Coin.HalfDollar => 50

/-- Calculates the total value of a list of coins in cents --/
def totalValue (coins : List Coin) : ℕ :=
  coins.foldl (fun acc c => acc + coinValue c) 0

/-- Theorem: The minimum number of coins to make one dollar is 3 --/
theorem min_coins_for_dollar :
  ∃ (coins : List Coin), totalValue coins = 100 ∧
    (∀ (other_coins : List Coin), totalValue other_coins = 100 →
      coins.length ≤ other_coins.length) ∧
    coins.length = 3 :=
  sorry

#check min_coins_for_dollar

end NUMINAMATH_CALUDE_min_coins_for_dollar_l2390_239090


namespace NUMINAMATH_CALUDE_negation_of_existence_quadratic_inequality_negation_l2390_239061

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) :=
by sorry

theorem quadratic_inequality_negation :
  (¬ ∃ x : ℝ, x^2 - 2*x - 3 < 0) ↔ (∀ x : ℝ, x^2 - 2*x - 3 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_quadratic_inequality_negation_l2390_239061


namespace NUMINAMATH_CALUDE_square_root_of_1024_l2390_239071

theorem square_root_of_1024 (x : ℝ) (h1 : x > 0) (h2 : x^2 = 1024) : x = 32 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_1024_l2390_239071


namespace NUMINAMATH_CALUDE_amount_owed_after_one_year_l2390_239067

/-- Calculates the total amount owed after applying simple interest --/
def total_amount_owed (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Theorem stating the total amount owed after one year --/
theorem amount_owed_after_one_year :
  let principal : ℝ := 54
  let rate : ℝ := 0.05
  let time : ℝ := 1
  total_amount_owed principal rate time = 56.70 := by
sorry

end NUMINAMATH_CALUDE_amount_owed_after_one_year_l2390_239067


namespace NUMINAMATH_CALUDE_cost_of_five_basketballs_l2390_239035

/-- The cost of buying multiple basketballs -/
def cost_of_basketballs (price_per_ball : ℝ) (num_balls : ℕ) : ℝ :=
  price_per_ball * num_balls

/-- Theorem: The cost of 5 basketballs is 5a yuan, given that one basketball costs a yuan -/
theorem cost_of_five_basketballs (a : ℝ) :
  cost_of_basketballs a 5 = 5 * a := by
  sorry

end NUMINAMATH_CALUDE_cost_of_five_basketballs_l2390_239035


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l2390_239046

-- Define the vectors a and b
def a (x : ℝ) : Fin 2 → ℝ := λ i => if i = 0 then x else 3
def b : Fin 2 → ℝ := λ i => if i = 0 then 3 else 1

-- Define the dot product of two 2D vectors
def dot_product (v w : Fin 2 → ℝ) : ℝ := (v 0) * (w 0) + (v 1) * (w 1)

-- State the theorem
theorem perpendicular_vectors_x_value :
  ∀ x : ℝ, dot_product (a x) b = 0 → x = -1 := by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l2390_239046


namespace NUMINAMATH_CALUDE_two_over_x_values_l2390_239032

theorem two_over_x_values (x : ℝ) (h : 1 - 9/x + 20/x^2 = 0) :
  2/x = 1/2 ∨ 2/x = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_two_over_x_values_l2390_239032


namespace NUMINAMATH_CALUDE_smallest_sum_of_digits_plus_one_l2390_239044

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: The smallest possible sum of digits of n+1 is 2, given that the sum of digits of n is 2017 -/
theorem smallest_sum_of_digits_plus_one (n : ℕ) (h : sum_of_digits n = 2017) :
  ∃ m : ℕ, sum_of_digits (n + 1) = 2 ∧ ∀ k : ℕ, sum_of_digits (k + 1) < 2 → sum_of_digits k ≠ 2017 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_digits_plus_one_l2390_239044


namespace NUMINAMATH_CALUDE_triangle_to_pentagon_area_ratio_l2390_239050

/-- Given a pentagon formed by placing an equilateral triangle atop a square,
    where the side length of the square equals the height of the triangle,
    prove that the ratio of the triangle's area to the pentagon's area is (3(√3 - 1))/6 -/
theorem triangle_to_pentagon_area_ratio :
  ∀ s : ℝ, s > 0 →
  let h := s * (Real.sqrt 3 / 2)
  let triangle_area := (Real.sqrt 3 / 4) * s^2
  let square_area := h^2
  let pentagon_area := triangle_area + square_area
  triangle_area / pentagon_area = (3 * (Real.sqrt 3 - 1)) / 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_to_pentagon_area_ratio_l2390_239050


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2390_239085

theorem polynomial_simplification (x : ℝ) :
  (3 * x^3 + 4 * x^2 + 6 * x - 5) - (2 * x^3 + 2 * x^2 + 4 * x - 15) =
  x^3 + 2 * x^2 + 2 * x + 10 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2390_239085


namespace NUMINAMATH_CALUDE_pencils_per_row_l2390_239051

/-- Theorem: Number of pencils in each row when 6 pencils are equally distributed into 2 rows -/
theorem pencils_per_row (total_pencils : ℕ) (num_rows : ℕ) (h1 : total_pencils = 6) (h2 : num_rows = 2) :
  total_pencils / num_rows = 3 := by
  sorry

end NUMINAMATH_CALUDE_pencils_per_row_l2390_239051


namespace NUMINAMATH_CALUDE_work_completed_in_three_days_l2390_239075

-- Define the work rates for A, B, and C
def work_rate_A : ℚ := 1 / 4
def work_rate_B : ℚ := 1 / 14
def work_rate_C : ℚ := 1 / 7

-- Define the total work to be done
def total_work : ℚ := 1

-- Define the work done in the first two days by A and B
def work_done_first_two_days : ℚ := 2 * (work_rate_A + work_rate_B)

-- Define the work done on the third day by A, B, and C
def work_done_third_day : ℚ := work_rate_A + work_rate_B + work_rate_C

-- Theorem to prove
theorem work_completed_in_three_days :
  work_done_first_two_days + work_done_third_day ≥ total_work :=
by sorry

end NUMINAMATH_CALUDE_work_completed_in_three_days_l2390_239075


namespace NUMINAMATH_CALUDE_no_real_solutions_l2390_239012

theorem no_real_solutions : ¬∃ x : ℝ, (2*x - 3*x + 7)^2 + 2 = -|2*x| := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l2390_239012


namespace NUMINAMATH_CALUDE_angle_C_is_84_l2390_239098

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_180 : A + B + C = 180
  positive : 0 < A ∧ 0 < B ∧ 0 < C

-- Define the ratio condition
def ratio_condition (t : Triangle) : Prop :=
  ∃ (k : ℝ), t.A = 4*k ∧ t.B = 4*k ∧ t.C = 7*k

-- Theorem statement
theorem angle_C_is_84 (t : Triangle) (h : ratio_condition t) : t.C = 84 :=
  sorry

end NUMINAMATH_CALUDE_angle_C_is_84_l2390_239098


namespace NUMINAMATH_CALUDE_square_sum_from_difference_and_product_l2390_239083

theorem square_sum_from_difference_and_product (x y : ℝ) 
  (h1 : x - y = 18) (h2 : x * y = 16) : x^2 + y^2 = 356 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_from_difference_and_product_l2390_239083


namespace NUMINAMATH_CALUDE_polynomial_with_three_equal_roots_l2390_239096

theorem polynomial_with_three_equal_roots (a b : ℤ) : 
  (∃ r : ℤ, (∀ x : ℝ, x^4 + x^3 - 18*x^2 + a*x + b = 0 ↔ 
    (x = r ∨ x = r ∨ x = r ∨ x = ((-1 : ℝ) - 3*r)))) → 
  (a = -52 ∧ b = -40) := by
sorry

end NUMINAMATH_CALUDE_polynomial_with_three_equal_roots_l2390_239096


namespace NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l2390_239069

/-- Given a geometric sequence {aₙ} with a₁ = 3 and common ratio q = √2, prove that a₇ = 24 -/
theorem geometric_sequence_seventh_term :
  ∀ (a : ℕ → ℝ), 
    (a 1 = 3) →
    (∀ n : ℕ, a (n + 1) = a n * Real.sqrt 2) →
    (a 7 = 24) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l2390_239069


namespace NUMINAMATH_CALUDE_max_area_of_rectangle_with_constraints_l2390_239024

/-- Represents a rectangle with sides x and y -/
structure Rectangle where
  x : ℝ
  y : ℝ

/-- The perimeter of the rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.x + r.y)

/-- The area of the rectangle -/
def area (r : Rectangle) : ℝ := r.x * r.y

/-- The condition that one side is at least twice as long as the other -/
def oneSideAtLeastTwiceOther (r : Rectangle) : Prop := r.x ≥ 2 * r.y

theorem max_area_of_rectangle_with_constraints :
  ∃ (r : Rectangle),
    perimeter r = 60 ∧
    oneSideAtLeastTwiceOther r ∧
    area r = 200 ∧
    ∀ (s : Rectangle),
      perimeter s = 60 →
      oneSideAtLeastTwiceOther s →
      area s ≤ area r :=
by sorry

end NUMINAMATH_CALUDE_max_area_of_rectangle_with_constraints_l2390_239024


namespace NUMINAMATH_CALUDE_average_of_remaining_digits_l2390_239003

theorem average_of_remaining_digits 
  (total_digits : Nat) 
  (subset_digits : Nat)
  (total_average : ℚ) 
  (subset_average : ℚ) :
  total_digits = 9 →
  subset_digits = 4 →
  total_average = 18 →
  subset_average = 8 →
  (total_digits * total_average - subset_digits * subset_average) / (total_digits - subset_digits) = 26 :=
by sorry

end NUMINAMATH_CALUDE_average_of_remaining_digits_l2390_239003


namespace NUMINAMATH_CALUDE_prime_divides_repunit_iff_l2390_239011

/-- A number of the form 111...1 (consisting entirely of the digit '1') -/
def repunit (n : ℕ) : ℕ := (10^n - 1) / 9

/-- Theorem stating that a prime number p is a divisor of some repunit if and only if p ≠ 2 and p ≠ 5 -/
theorem prime_divides_repunit_iff (p : ℕ) (hp : Prime p) :
  (∃ n : ℕ, p ∣ repunit n) ↔ p ≠ 2 ∧ p ≠ 5 := by
  sorry

end NUMINAMATH_CALUDE_prime_divides_repunit_iff_l2390_239011


namespace NUMINAMATH_CALUDE_fraction_value_l2390_239001

theorem fraction_value : (1 : ℚ) / (4 * 5) = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l2390_239001


namespace NUMINAMATH_CALUDE_coefficient_x6_in_expansion_l2390_239068

theorem coefficient_x6_in_expansion : 
  (Finset.range 5).sum (fun k => 
    (Nat.choose 4 k : ℝ) * (1 : ℝ)^(4 - k) * (3 : ℝ)^k * 
    if k = 2 then 1 else 0) = 54 := by sorry

end NUMINAMATH_CALUDE_coefficient_x6_in_expansion_l2390_239068


namespace NUMINAMATH_CALUDE_expression_bounds_l2390_239049

theorem expression_bounds (x y : ℝ) (h : x^2 + y^2 = 4) :
  1 ≤ 4*(x - 1/2)^2 + (y - 1)^2 + 4*x*y ∧ 
  4*(x - 1/2)^2 + (y - 1)^2 + 4*x*y ≤ 22 + 4*Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_bounds_l2390_239049


namespace NUMINAMATH_CALUDE_max_value_of_s_l2390_239036

theorem max_value_of_s (p q r s : ℝ) 
  (sum_condition : p + q + r + s = 10)
  (product_condition : p * q + p * r + p * s + q * r + q * s + r * s = 20) :
  s ≤ (5 + Real.sqrt 105) / 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_s_l2390_239036


namespace NUMINAMATH_CALUDE_vector_magnitude_proof_l2390_239016

def a : Fin 2 → ℝ := ![0, 1]
def b : Fin 2 → ℝ := ![2, -1]

theorem vector_magnitude_proof : 
  ‖(2 • (a : Fin 2 → ℝ)) + (b : Fin 2 → ℝ)‖ = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_proof_l2390_239016


namespace NUMINAMATH_CALUDE_count_valid_pairs_l2390_239072

-- Define Ω as a nonreal root of z^4 = 1
def Ω : ℂ := Complex.I

-- Define the condition for valid pairs
def isValidPair (a b : ℤ) : Prop := Complex.abs (a • Ω + b) = 2

-- Theorem statement
theorem count_valid_pairs : 
  (∃! (n : ℕ), ∃ (s : Finset (ℤ × ℤ)), s.card = n ∧ 
    (∀ (p : ℤ × ℤ), p ∈ s ↔ isValidPair p.1 p.2) ∧ n = 4) :=
sorry

end NUMINAMATH_CALUDE_count_valid_pairs_l2390_239072


namespace NUMINAMATH_CALUDE_women_equal_to_five_men_l2390_239078

/-- Represents the amount of work one person can do in a day -/
structure WorkPerDay (α : Type) where
  amount : ℝ

/-- Represents the total amount of work for a job -/
def Job : Type := ℝ

variable (men_work : WorkPerDay Unit) (women_work : WorkPerDay Unit)

/-- The amount of work 5 men do in a day equals the amount of work x women do in a day -/
def men_women_equal (x : ℝ) : Prop :=
  5 * men_work.amount = x * women_work.amount

/-- 3 men and 5 women finish the job in 10 days -/
def job_condition1 (job : Job) : Prop :=
  (3 * men_work.amount + 5 * women_work.amount) * 10 = job

/-- 7 women finish the job in 14 days -/
def job_condition2 (job : Job) : Prop :=
  7 * women_work.amount * 14 = job

/-- The main theorem: prove that 8 women do the same amount of work in a day as 5 men -/
theorem women_equal_to_five_men
  (job : Job)
  (h1 : job_condition1 men_work women_work job)
  (h2 : job_condition2 women_work job) :
  men_women_equal men_work women_work 8 := by
  sorry


end NUMINAMATH_CALUDE_women_equal_to_five_men_l2390_239078


namespace NUMINAMATH_CALUDE_expand_and_simplify_solve_equation_l2390_239010

-- Problem 1
theorem expand_and_simplify (x y : ℝ) : 
  (x + 3*y)^2 - (x + 3*y)*(x - 3*y) = 6*x*y + 18*y^2 := by sorry

-- Problem 2
theorem solve_equation : 
  ∃ (x : ℝ), x / (2*x - 1) = 2 - 3 / (1 - 2*x) ∧ x = -1/3 := by sorry

end NUMINAMATH_CALUDE_expand_and_simplify_solve_equation_l2390_239010


namespace NUMINAMATH_CALUDE_machines_count_l2390_239048

theorem machines_count (x : ℝ) (N : ℕ) (R : ℝ) : 
  N * R = x / 3 →
  45 * R = 5 * x / 10 →
  N = 30 := by
  sorry

end NUMINAMATH_CALUDE_machines_count_l2390_239048


namespace NUMINAMATH_CALUDE_is_center_of_ellipse_l2390_239057

/-- The equation of the ellipse -/
def ellipse_equation (x y : ℝ) : Prop :=
  2 * x^2 + 2 * x * y + y^2 + 2 * x + 2 * y - 4 = 0

/-- The center of the ellipse -/
def ellipse_center : ℝ × ℝ := (0, -1)

/-- Theorem stating that the given point is the center of the ellipse -/
theorem is_center_of_ellipse :
  ∀ (x y : ℝ), ellipse_equation x y →
  ellipse_center = (0, -1) := by sorry

end NUMINAMATH_CALUDE_is_center_of_ellipse_l2390_239057


namespace NUMINAMATH_CALUDE_circle_larger_than_unit_circle_in_larger_square_circle_may_not_touch_diamond_l2390_239081

/-- A circle defined by two inequalities -/
def SpecialCircle (x y : ℝ) : Prop :=
  (abs x + abs y ≤ (3/2) * Real.sqrt (2 * (x^2 + y^2))) ∧
  (Real.sqrt (2 * (x^2 + y^2)) ≤ 3 * max (abs x) (abs y))

/-- The circle is larger than a standard unit circle -/
theorem circle_larger_than_unit : ∃ (x y : ℝ), SpecialCircle x y ∧ x^2 + y^2 > 1 := by sorry

/-- The circle is contained within a square larger than the standard unit square -/
theorem circle_in_larger_square : ∃ (s : ℝ), s > 1 ∧ ∀ (x y : ℝ), SpecialCircle x y → max (abs x) (abs y) ≤ s := by sorry

/-- The circle may not touch all points of a diamond inscribed in the square -/
theorem circle_may_not_touch_diamond : ∃ (x y : ℝ), abs x + abs y = 1 ∧ ¬(SpecialCircle x y) := by sorry

end NUMINAMATH_CALUDE_circle_larger_than_unit_circle_in_larger_square_circle_may_not_touch_diamond_l2390_239081


namespace NUMINAMATH_CALUDE_burger_cost_l2390_239030

theorem burger_cost (alice_burgers alice_sodas alice_total bill_burgers bill_sodas bill_total : ℕ)
  (h_alice : alice_burgers = 4 ∧ alice_sodas = 3 ∧ alice_total = 420)
  (h_bill : bill_burgers = 3 ∧ bill_sodas = 2 ∧ bill_total = 310) :
  ∃ (burger_cost soda_cost : ℕ),
    alice_burgers * burger_cost + alice_sodas * soda_cost = alice_total ∧
    bill_burgers * burger_cost + bill_sodas * soda_cost = bill_total ∧
    burger_cost = 90 := by
  sorry

end NUMINAMATH_CALUDE_burger_cost_l2390_239030


namespace NUMINAMATH_CALUDE_average_leaves_theorem_l2390_239047

/-- The number of leaves that fell in the first hour -/
def leaves_first_hour : ℕ := 7

/-- The rate of leaves falling per hour for the second and third hour -/
def leaves_rate_later : ℕ := 4

/-- The total number of hours of observation -/
def total_hours : ℕ := 3

/-- The total number of leaves that fell during the observation period -/
def total_leaves : ℕ := leaves_first_hour + leaves_rate_later * (total_hours - 1)

/-- The average number of leaves falling per hour -/
def average_leaves_per_hour : ℚ := total_leaves / total_hours

theorem average_leaves_theorem : average_leaves_per_hour = 5 := by
  sorry

end NUMINAMATH_CALUDE_average_leaves_theorem_l2390_239047


namespace NUMINAMATH_CALUDE_midpoint_distance_theorem_l2390_239091

theorem midpoint_distance_theorem (s : ℝ) : 
  let P : ℝ × ℝ := (s - 3, 2)
  let Q : ℝ × ℝ := (1, s + 2)
  let M : ℝ × ℝ := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)
  (M.1 - P.1)^2 + (M.2 - P.2)^2 = 3 * s^2 / 4 →
  s = -5 - 5 * Real.sqrt 2 ∨ s = -5 + 5 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_midpoint_distance_theorem_l2390_239091


namespace NUMINAMATH_CALUDE_eve_hit_ten_l2390_239056

-- Define the set of possible scores
def ScoreSet : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}

-- Define a type for players
inductive Player : Type
| Alex | Becca | Carli | Dan | Eve | Fiona

-- Define a function that returns a player's score
def player_score : Player → ℕ
| Player.Alex => 20
| Player.Becca => 5
| Player.Carli => 13
| Player.Dan => 15
| Player.Eve => 21
| Player.Fiona => 6

-- Define a function that returns a pair of scores for a player
def player_throws (p : Player) : ℕ × ℕ := sorry

-- State the theorem
theorem eve_hit_ten :
  ∀ (p : Player),
    (∀ (q : Player), p ≠ q → player_throws p ≠ player_throws q) ∧
    (∀ (p : Player), (player_throws p).1 ∈ ScoreSet ∧ (player_throws p).2 ∈ ScoreSet) ∧
    (∀ (p : Player), (player_throws p).1 + (player_throws p).2 = player_score p) →
    (player_throws Player.Eve).1 = 10 ∨ (player_throws Player.Eve).2 = 10 :=
by sorry

end NUMINAMATH_CALUDE_eve_hit_ten_l2390_239056


namespace NUMINAMATH_CALUDE_cone_volume_over_pi_l2390_239026

/-- The volume of a cone formed from a 300-degree sector of a circle with radius 18, divided by π, is equal to 225√11 -/
theorem cone_volume_over_pi (r : ℝ) (sector_angle : ℝ) :
  r = 18 →
  sector_angle = 300 →
  let base_radius := sector_angle / 360 * r
  let height := Real.sqrt (r^2 - base_radius^2)
  let volume := (1/3) * π * base_radius^2 * height
  volume / π = 225 * Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_over_pi_l2390_239026


namespace NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l2390_239074

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_third_term 
  (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_sum : a 1 + a 2 + a 3 + a 4 + a 5 = 20) : 
  a 3 = 4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l2390_239074


namespace NUMINAMATH_CALUDE_permutations_with_four_transpositions_l2390_239076

/-- The number of elements in the permutation -/
def n : ℕ := 6

/-- The total number of permutations of n elements -/
def total_permutations : ℕ := n.factorial

/-- The number of even permutations -/
def even_permutations : ℕ := total_permutations / 2

/-- The number of permutations that require i transpositions to become the identity permutation -/
def num_permutations (i : ℕ) : ℕ := sorry

/-- The theorem stating that the number of permutations requiring 4 transpositions is 304 -/
theorem permutations_with_four_transpositions :
  num_permutations 4 = 304 :=
sorry

end NUMINAMATH_CALUDE_permutations_with_four_transpositions_l2390_239076
