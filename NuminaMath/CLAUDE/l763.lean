import Mathlib

namespace NUMINAMATH_CALUDE_kates_average_speed_l763_76381

theorem kates_average_speed (bike_speed : ℝ) (bike_time : ℝ) (walk_speed : ℝ) (walk_time : ℝ) 
  (h1 : bike_speed = 20) 
  (h2 : bike_time = 45 / 60) 
  (h3 : walk_speed = 3) 
  (h4 : walk_time = 60 / 60) : 
  (bike_speed * bike_time + walk_speed * walk_time) / (bike_time + walk_time) = 10 := by
  sorry

#check kates_average_speed

end NUMINAMATH_CALUDE_kates_average_speed_l763_76381


namespace NUMINAMATH_CALUDE_planted_fraction_is_404_841_l763_76333

/-- Represents a rectangular field with a planted right triangular area and an unplanted square --/
structure PlantedField where
  length : ℝ
  width : ℝ
  square_side : ℝ
  hypotenuse_distance : ℝ

/-- The fraction of the field that is planted --/
def planted_fraction (field : PlantedField) : ℚ :=
  sorry

/-- Theorem stating the planted fraction for the given field configuration --/
theorem planted_fraction_is_404_841 :
  let field : PlantedField := {
    length := 5,
    width := 6,
    square_side := 33 / 41,
    hypotenuse_distance := 3
  }
  planted_fraction field = 404 / 841 := by sorry

end NUMINAMATH_CALUDE_planted_fraction_is_404_841_l763_76333


namespace NUMINAMATH_CALUDE_power_product_rule_l763_76313

theorem power_product_rule (x : ℝ) : x^2 * x^3 = x^5 := by
  sorry

end NUMINAMATH_CALUDE_power_product_rule_l763_76313


namespace NUMINAMATH_CALUDE_marble_remainder_l763_76310

theorem marble_remainder (r p : ℤ) 
  (hr : r % 8 = 5) 
  (hp : p % 8 = 6) : 
  (r + p) % 8 = 3 := by
sorry

end NUMINAMATH_CALUDE_marble_remainder_l763_76310


namespace NUMINAMATH_CALUDE_unique_three_digit_number_square_equals_sum_of_digits_power_five_l763_76321

/-- A function that returns the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem stating that 243 is the only three-digit number whose square 
    is equal to the sum of its digits raised to the power of 5 -/
theorem unique_three_digit_number_square_equals_sum_of_digits_power_five : 
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n^2 = (sum_of_digits n)^5 := by sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_square_equals_sum_of_digits_power_five_l763_76321


namespace NUMINAMATH_CALUDE_exists_strategy_to_find_genuine_coin_l763_76363

/-- Represents a coin, which can be either genuine or counterfeit. -/
inductive Coin
| genuine
| counterfeit

/-- Represents the result of weighing two coins. -/
inductive WeighResult
| equal
| left_heavier
| right_heavier

/-- A function that simulates weighing two coins. -/
def weigh : Coin → Coin → WeighResult := sorry

/-- The total number of coins. -/
def total_coins : Nat := 100

/-- A function that represents the distribution of coins. -/
def coin_distribution : Fin total_coins → Coin := sorry

/-- The number of counterfeit coins. -/
def counterfeit_count : Nat := sorry

/-- Assumption that there are more than 0 but less than 99 counterfeit coins. -/
axiom counterfeit_range : 0 < counterfeit_count ∧ counterfeit_count < 99

/-- A strategy is a function that takes the current state and returns the next pair of coins to weigh. -/
def Strategy := List WeighResult → Fin total_coins × Fin total_coins

/-- The theorem stating that there exists a strategy to find a genuine coin. -/
theorem exists_strategy_to_find_genuine_coin :
  ∃ (s : Strategy), ∀ (coin_dist : Fin total_coins → Coin),
    ∃ (n : Nat), n ≤ 99 ∧
      (∃ (i : Fin total_coins), coin_dist i = Coin.genuine ∧
        (∀ (j : Fin total_coins), j ≠ i → coin_dist j = Coin.counterfeit)) :=
sorry

end NUMINAMATH_CALUDE_exists_strategy_to_find_genuine_coin_l763_76363


namespace NUMINAMATH_CALUDE_selling_price_ratio_l763_76327

/-- Proves that the ratio of selling prices is 2:1 given different profit percentages -/
theorem selling_price_ratio (cost : ℝ) (profit1 profit2 : ℝ) 
  (h1 : profit1 = 0.20)
  (h2 : profit2 = 1.40) :
  (cost + profit2 * cost) / (cost + profit1 * cost) = 2 := by
  sorry

end NUMINAMATH_CALUDE_selling_price_ratio_l763_76327


namespace NUMINAMATH_CALUDE_contest_scores_mode_and_median_l763_76397

def scores : List ℕ := [91, 95, 89, 93, 88, 94, 95]

def mode (l : List ℕ) : ℕ := sorry

def median (l : List ℕ) : ℕ := sorry

theorem contest_scores_mode_and_median :
  mode scores = 95 ∧ median scores = 93 := by sorry

end NUMINAMATH_CALUDE_contest_scores_mode_and_median_l763_76397


namespace NUMINAMATH_CALUDE_gcd_175_100_65_l763_76380

theorem gcd_175_100_65 : Nat.gcd 175 (Nat.gcd 100 65) = 5 := by
  sorry

end NUMINAMATH_CALUDE_gcd_175_100_65_l763_76380


namespace NUMINAMATH_CALUDE_expression_equality_l763_76346

theorem expression_equality : (45 + 15)^2 - 3 * (45^2 + 15^2 - 2 * 45 * 15) = 900 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l763_76346


namespace NUMINAMATH_CALUDE_scholarship_theorem_l763_76386

def scholarship_problem (wendy_last_year : ℝ) : Prop :=
  let kelly_last_year := 2 * wendy_last_year
  let nina_last_year := kelly_last_year - 8000
  let jason_last_year := 3/4 * kelly_last_year
  let wendy_this_year := wendy_last_year * 1.1
  let kelly_this_year := kelly_last_year * 1.08
  let nina_this_year := nina_last_year * 1.15
  let jason_this_year := jason_last_year * 1.12
  let total_this_year := wendy_this_year + kelly_this_year + nina_this_year + jason_this_year
  wendy_last_year = 20000 → total_this_year = 135600

theorem scholarship_theorem : scholarship_problem 20000 := by
  sorry

end NUMINAMATH_CALUDE_scholarship_theorem_l763_76386


namespace NUMINAMATH_CALUDE_exists_construction_with_1001_free_endpoints_l763_76377

/-- Represents a point in the construction --/
structure Point :=
  (depth : ℕ)
  (branches : Fin 5)

/-- Represents the construction of line segments --/
def Construction := List Point

/-- Counts the number of free endpoints in a construction --/
def count_free_endpoints (c : Construction) : ℕ := sorry

/-- Theorem: There exists a construction with 1001 free endpoints --/
theorem exists_construction_with_1001_free_endpoints :
  ∃ (c : Construction), count_free_endpoints c = 1001 := by sorry

end NUMINAMATH_CALUDE_exists_construction_with_1001_free_endpoints_l763_76377


namespace NUMINAMATH_CALUDE_function_inequality_l763_76394

/-- Given functions f and g, prove that a ≤ 1 -/
theorem function_inequality (f g : ℝ → ℝ) (a : ℝ) : 
  (∀ x, f x = x + 4 / x) →
  (∀ x, g x = 2^x + a) →
  (∀ x₁ ∈ Set.Icc (1/2) 1, ∃ x₂ ∈ Set.Icc 2 3, f x₁ ≥ g x₂) →
  a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_function_inequality_l763_76394


namespace NUMINAMATH_CALUDE_two_marbles_in_two_boxes_proof_l763_76306

/-- The number of ways to choose 2 marbles out of 3 distinct marbles 
    and place them in 2 indistinguishable boxes -/
def two_marbles_in_two_boxes : ℕ := 3

/-- The number of distinct marbles -/
def total_marbles : ℕ := 3

/-- The number of marbles to be chosen -/
def chosen_marbles : ℕ := 2

/-- The number of boxes -/
def num_boxes : ℕ := 2

/-- Boxes are indistinguishable -/
def boxes_indistinguishable : Prop := True

theorem two_marbles_in_two_boxes_proof :
  two_marbles_in_two_boxes = (total_marbles.choose chosen_marbles) :=
by sorry

end NUMINAMATH_CALUDE_two_marbles_in_two_boxes_proof_l763_76306


namespace NUMINAMATH_CALUDE_m_arithmetic_pascal_triangle_structure_l763_76319

/-- m-arithmetic Pascal triangle with s-th row zeros except extremes -/
structure MArithmeticPascalTriangle where
  m : ℕ
  s : ℕ
  Δ : ℕ → ℕ → ℕ
  P : ℕ → ℕ → ℕ

/-- Properties of the m-arithmetic Pascal triangle -/
class MArithmeticPascalTriangleProps (T : MArithmeticPascalTriangle) where
  zero_middle : ∀ (i k : ℕ), i % T.s = 0 → 0 < k → k < i → T.Δ i k = 0
  relation_a : ∀ (n k : ℕ), T.Δ n (k-1) + T.Δ n k = T.Δ (n+1) k
  relation_b : ∀ (n k : ℕ), T.Δ n k = T.P n k * T.Δ 0 0

/-- Theorem: The structure of the m-arithmetic Pascal triangle is correct -/
theorem m_arithmetic_pascal_triangle_structure 
  (T : MArithmeticPascalTriangle) 
  [MArithmeticPascalTriangleProps T] : 
  ∀ (n k : ℕ), T.Δ n k = T.P n k * T.Δ 0 0 := by
  sorry

end NUMINAMATH_CALUDE_m_arithmetic_pascal_triangle_structure_l763_76319


namespace NUMINAMATH_CALUDE_min_value_of_expression_min_value_achievable_l763_76301

theorem min_value_of_expression (x y : ℝ) : (x*y - 1)^2 + (x + y)^2 ≥ 1 := by
  sorry

theorem min_value_achievable : ∃ x y : ℝ, (x*y - 1)^2 + (x + y)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_min_value_achievable_l763_76301


namespace NUMINAMATH_CALUDE_garlic_cloves_used_l763_76391

/-- Proves that the number of garlic cloves used for cooking is the difference between
    the initial number and the remaining number of cloves -/
theorem garlic_cloves_used (initial : ℕ) (remaining : ℕ) (used : ℕ) 
    (h1 : initial = 93)
    (h2 : remaining = 7)
    (h3 : used = initial - remaining) :
  used = 86 := by
  sorry

end NUMINAMATH_CALUDE_garlic_cloves_used_l763_76391


namespace NUMINAMATH_CALUDE_find_other_number_l763_76358

theorem find_other_number (A B : ℕ+) (h1 : A = 500) (h2 : Nat.lcm A B = 3000) (h3 : Nat.gcd A B = 100) :
  B = 600 := by
  sorry

end NUMINAMATH_CALUDE_find_other_number_l763_76358


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l763_76302

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the solution set of f(x) < 0
def solution_set_f_neg (x : ℝ) : Prop := x < -1 ∨ x > 1/2

-- Define the solution set of f(10^x) > 0
def solution_set_f_exp (x : ℝ) : Prop := x < -Real.log 2 / Real.log 10

-- Theorem statement
theorem solution_set_equivalence :
  (∀ x, f x < 0 ↔ solution_set_f_neg x) →
  (∀ x, f (10^x) > 0 ↔ solution_set_f_exp x) :=
sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l763_76302


namespace NUMINAMATH_CALUDE_find_number_l763_76398

theorem find_number : ∃ x : ℝ, 4 * x - 23 = 33 ∧ x = 14 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l763_76398


namespace NUMINAMATH_CALUDE_lives_lost_l763_76339

theorem lives_lost (initial_lives gained_lives final_lives : ℕ) : 
  initial_lives = 43 → gained_lives = 27 → final_lives = 56 → 
  ∃ (lost_lives : ℕ), initial_lives - lost_lives + gained_lives = final_lives ∧ lost_lives = 14 := by
sorry

end NUMINAMATH_CALUDE_lives_lost_l763_76339


namespace NUMINAMATH_CALUDE_triangle_side_expression_l763_76340

/-- Given a triangle with sides a, b, and c, 
    prove that |a-b-c| + |b-c-a| + |c+a-b| = 3c + a - b -/
theorem triangle_side_expression 
  (a b c : ℝ) 
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0)
  (h_ineq1 : a + b > c) 
  (h_ineq2 : b + c > a) 
  (h_ineq3 : c + a > b) : 
  |a - b - c| + |b - c - a| + |c + a - b| = 3 * c + a - b :=
sorry

end NUMINAMATH_CALUDE_triangle_side_expression_l763_76340


namespace NUMINAMATH_CALUDE_art_students_count_l763_76304

/-- Given a high school with the following student enrollment:
  * 500 total students
  * 50 students taking music
  * 10 students taking both music and art
  * 440 students taking neither music nor art
  Prove that the number of students taking art is 20. -/
theorem art_students_count (total : ℕ) (music : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : total = 500)
  (h2 : music = 50)
  (h3 : both = 10)
  (h4 : neither = 440) :
  total - music - neither + both = 20 := by
  sorry

#check art_students_count

end NUMINAMATH_CALUDE_art_students_count_l763_76304


namespace NUMINAMATH_CALUDE_madeline_max_distance_difference_l763_76314

-- Define the speeds and durations
def madeline_speed : ℝ := 12
def madeline_time : ℝ := 3
def max_speed : ℝ := 15
def max_time : ℝ := 2

-- Define the distance function
def distance (speed time : ℝ) : ℝ := speed * time

-- Theorem statement
theorem madeline_max_distance_difference :
  distance madeline_speed madeline_time - distance max_speed max_time = 6 := by
  sorry

end NUMINAMATH_CALUDE_madeline_max_distance_difference_l763_76314


namespace NUMINAMATH_CALUDE_subway_ride_time_l763_76342

theorem subway_ride_time (total_time subway_time train_time bike_time : ℝ) : 
  total_time = 38 →
  train_time = 2 * subway_time →
  bike_time = 8 →
  total_time = subway_time + train_time + bike_time →
  subway_time = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_subway_ride_time_l763_76342


namespace NUMINAMATH_CALUDE_linear_system_fraction_sum_l763_76354

theorem linear_system_fraction_sum (a b c u v w : ℝ) 
  (eq1 : 17 * u + b * v + c * w = 0)
  (eq2 : a * u + 29 * v + c * w = 0)
  (eq3 : a * u + b * v + 56 * w = 0)
  (ha : a ≠ 17)
  (hu : u ≠ 0) :
  a / (a - 17) + b / (b - 29) + c / (c - 56) = 1 := by
  sorry

end NUMINAMATH_CALUDE_linear_system_fraction_sum_l763_76354


namespace NUMINAMATH_CALUDE_problem_1_l763_76324

theorem problem_1 : Real.sqrt (3/2) * Real.sqrt (21/4) / Real.sqrt (7/2) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l763_76324


namespace NUMINAMATH_CALUDE_simplify_polynomial_l763_76352

theorem simplify_polynomial (x : ℝ) : (3 * x^2 + 6 * x - 5) - (2 * x^2 + 4 * x - 8) = x^2 + 2 * x + 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l763_76352


namespace NUMINAMATH_CALUDE_real_part_of_complex_product_l763_76334

theorem real_part_of_complex_product : 
  let i : ℂ := Complex.I
  let z : ℂ := (1 + 3*i) * i
  Complex.re z = -3 := by sorry

end NUMINAMATH_CALUDE_real_part_of_complex_product_l763_76334


namespace NUMINAMATH_CALUDE_batsman_average_after_12th_inning_l763_76325

/-- Represents a batsman's performance -/
structure Batsman where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the new average after an inning -/
def newAverage (b : Batsman) (runsInLastInning : ℕ) : ℚ :=
  (b.totalRuns + runsInLastInning : ℚ) / (b.innings + 1)

theorem batsman_average_after_12th_inning 
  (b : Batsman) 
  (h1 : b.innings = 11)
  (h2 : newAverage b 60 = b.average + 4) :
  newAverage b 60 = 16 := by
sorry

end NUMINAMATH_CALUDE_batsman_average_after_12th_inning_l763_76325


namespace NUMINAMATH_CALUDE_product_of_sum_and_difference_l763_76364

theorem product_of_sum_and_difference (a b : ℝ) 
  (sum_eq : a + b = 3) 
  (diff_eq : a - b = 7) : 
  a * b = -10 := by sorry

end NUMINAMATH_CALUDE_product_of_sum_and_difference_l763_76364


namespace NUMINAMATH_CALUDE_jack_morning_emails_l763_76361

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := sorry

/-- The number of emails Jack received in the afternoon -/
def afternoon_emails : ℕ := 10

/-- The number of emails Jack received in the evening -/
def evening_emails : ℕ := 7

/-- Theorem stating that Jack received 9 emails in the morning -/
theorem jack_morning_emails : 
  morning_emails = evening_emails + 2 → morning_emails = 9 :=
by sorry

end NUMINAMATH_CALUDE_jack_morning_emails_l763_76361


namespace NUMINAMATH_CALUDE_final_answer_is_67_l763_76372

def ben_calculation (x : ℕ) : ℕ :=
  ((x + 2) * 3) + 5

def sue_calculation (x : ℕ) : ℕ :=
  ((x - 3) * 3) + 7

theorem final_answer_is_67 :
  sue_calculation (ben_calculation 4) = 67 := by
  sorry

end NUMINAMATH_CALUDE_final_answer_is_67_l763_76372


namespace NUMINAMATH_CALUDE_trigonometric_expression_simplification_l763_76360

theorem trigonometric_expression_simplification :
  let expr := (Real.sin (10 * π / 180) + Real.sin (20 * π / 180) + 
               Real.sin (30 * π / 180) + Real.sin (40 * π / 180) + 
               Real.sin (50 * π / 180) + Real.sin (60 * π / 180) + 
               Real.sin (70 * π / 180) + Real.sin (80 * π / 180)) / 
              (Real.cos (5 * π / 180) * Real.cos (10 * π / 180) * Real.cos (20 * π / 180))
  expr = 4 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_expression_simplification_l763_76360


namespace NUMINAMATH_CALUDE_a_less_than_b_l763_76388

-- Define the function f
def f (x m : ℝ) : ℝ := -4 * x^2 + 8 * x + m

-- State the theorem
theorem a_less_than_b (m : ℝ) (a b : ℝ) 
  (h1 : f (-2) m = a) 
  (h2 : f 3 m = b) : 
  a < b := by
  sorry

end NUMINAMATH_CALUDE_a_less_than_b_l763_76388


namespace NUMINAMATH_CALUDE_koala_fiber_consumption_l763_76320

/-- Given that koalas absorb 30% of the fiber they eat, 
    prove that if a koala absorbed 12 ounces of fiber, 
    it ate 40 ounces of fiber. -/
theorem koala_fiber_consumption 
  (absorption_rate : ℝ) 
  (absorbed_amount : ℝ) 
  (h1 : absorption_rate = 0.3)
  (h2 : absorbed_amount = 12) :
  absorbed_amount / absorption_rate = 40 := by
  sorry

end NUMINAMATH_CALUDE_koala_fiber_consumption_l763_76320


namespace NUMINAMATH_CALUDE_area_SUVR_area_SUVR_is_141_44_l763_76317

/-- Triangle PQR with given properties and points S, T, U, V as described -/
structure TrianglePQR where
  /-- Side length PR -/
  pr : ℝ
  /-- Side length PQ -/
  pq : ℝ
  /-- Area of triangle PQR -/
  area : ℝ
  /-- Point S on PR such that PS = 1/3 * PR -/
  s : ℝ
  /-- Point T on PQ such that PT = 1/3 * PQ -/
  t : ℝ
  /-- Point U on ST -/
  u : ℝ
  /-- Point V on QR -/
  v : ℝ
  /-- PR equals 60 -/
  h_pr : pr = 60
  /-- PQ equals 15 -/
  h_pq : pq = 15
  /-- Area of triangle PQR equals 180 -/
  h_area : area = 180
  /-- PS equals 1/3 of PR -/
  h_s : s = 1/3 * pr
  /-- PT equals 1/3 of PQ -/
  h_t : t = 1/3 * pq
  /-- U is on the angle bisector of angle PQR -/
  h_u_bisector : True  -- Placeholder for the angle bisector condition
  /-- V is on the angle bisector of angle PQR -/
  h_v_bisector : True  -- Placeholder for the angle bisector condition

/-- The area of quadrilateral SUVR in the given triangle PQR is 141.44 -/
theorem area_SUVR (tri : TrianglePQR) : ℝ := 141.44

/-- The main theorem: The area of quadrilateral SUVR is 141.44 -/
theorem area_SUVR_is_141_44 (tri : TrianglePQR) : area_SUVR tri = 141.44 := by
  sorry

end NUMINAMATH_CALUDE_area_SUVR_area_SUVR_is_141_44_l763_76317


namespace NUMINAMATH_CALUDE_max_value_of_ab_l763_76330

theorem max_value_of_ab (a b : ℝ) (h1 : b > 0) (h2 : 3 * a + 4 * b = 2) :
  a * b ≤ 1 / 12 ∧ ∃ (a₀ b₀ : ℝ), b₀ > 0 ∧ 3 * a₀ + 4 * b₀ = 2 ∧ a₀ * b₀ = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_ab_l763_76330


namespace NUMINAMATH_CALUDE_negation_of_existential_proposition_l763_76343

theorem negation_of_existential_proposition :
  ¬(∃ x : ℝ, x ≥ 0 ∧ x^2 > 3) ↔ ∀ x : ℝ, x ≥ 0 → x^2 ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existential_proposition_l763_76343


namespace NUMINAMATH_CALUDE_right_triangle_revolution_is_cone_l763_76353

/-- A right-angled triangle -/
structure RightTriangle where
  base : ℝ
  height : ℝ

/-- A cone -/
structure Cone where
  radius : ℝ
  height : ℝ

/-- Solid of revolution generated by rotating a right-angled triangle about one of its legs -/
def solidOfRevolution (t : RightTriangle) : Cone :=
  { radius := t.base, height := t.height }

/-- Theorem: The solid of revolution generated by rotating a right-angled triangle
    about one of its legs is a cone -/
theorem right_triangle_revolution_is_cone (t : RightTriangle) :
  ∃ (c : Cone), solidOfRevolution t = c :=
sorry

end NUMINAMATH_CALUDE_right_triangle_revolution_is_cone_l763_76353


namespace NUMINAMATH_CALUDE_coin_worth_l763_76308

def total_coins : ℕ := 20
def nickel_value : ℚ := 5 / 100
def dime_value : ℚ := 10 / 100
def swap_difference : ℚ := 70 / 100

theorem coin_worth (n : ℕ) (h1 : n ≤ total_coins) :
  (n : ℚ) * nickel_value + (total_coins - n : ℚ) * dime_value + swap_difference = 
  (n : ℚ) * dime_value + (total_coins - n : ℚ) * nickel_value →
  (n : ℚ) * nickel_value + (total_coins - n : ℚ) * dime_value = 115 / 100 := by
  sorry

end NUMINAMATH_CALUDE_coin_worth_l763_76308


namespace NUMINAMATH_CALUDE_power_function_odd_condition_l763_76300

def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x ^ b

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

theorem power_function_odd_condition (m : ℝ) :
  let f : ℝ → ℝ := λ x ↦ (m^2 - 5*m + 7) * x^(m-2)
  is_power_function f ∧ is_odd_function f → m = 3 :=
by sorry

end NUMINAMATH_CALUDE_power_function_odd_condition_l763_76300


namespace NUMINAMATH_CALUDE_lifeguard_swimming_test_l763_76341

/-- Lifeguard swimming test problem -/
theorem lifeguard_swimming_test
  (total_distance : ℝ)
  (front_crawl_speed : ℝ)
  (total_time : ℝ)
  (front_crawl_time : ℝ)
  (h1 : total_distance = 500)
  (h2 : front_crawl_speed = 45)
  (h3 : total_time = 12)
  (h4 : front_crawl_time = 8) :
  let front_crawl_distance := front_crawl_speed * front_crawl_time
  let breaststroke_distance := total_distance - front_crawl_distance
  let breaststroke_time := total_time - front_crawl_time
  let breaststroke_speed := breaststroke_distance / breaststroke_time
  breaststroke_speed = 35 := by
sorry


end NUMINAMATH_CALUDE_lifeguard_swimming_test_l763_76341


namespace NUMINAMATH_CALUDE_sum_abcd_l763_76337

theorem sum_abcd (a b c d : ℚ) 
  (h : a + 2 = b + 3 ∧ b + 3 = c + 4 ∧ c + 4 = d + 5 ∧ d + 5 = a + b + c + d + 7) : 
  a + b + c + d = -14/3 := by
  sorry

end NUMINAMATH_CALUDE_sum_abcd_l763_76337


namespace NUMINAMATH_CALUDE_unique_solution_l763_76392

/-- The function f(x) = x^3 + 3x^2 + 1 -/
def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 1

/-- Theorem stating the unique solution for a and b -/
theorem unique_solution (a b : ℝ) : 
  a ≠ 0 ∧ 
  (∀ x : ℝ, f x - f a = (x - b) * (x - a)^2) → 
  a = -2 ∧ b = 1 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l763_76392


namespace NUMINAMATH_CALUDE_area_of_triangle_ABC_l763_76395

-- Define the curve
def f (x : ℝ) : ℝ := x^2 - 7*x + 12

-- Define the points A, B, and C
def A : ℝ × ℝ := (3, 0)
def B : ℝ × ℝ := (4, 0)
def C : ℝ × ℝ := (0, 12)

-- Theorem statement
theorem area_of_triangle_ABC : 
  let triangle_area := (1/2) * |A.1 - B.1| * C.2
  (f A.1 = 0) ∧ (f B.1 = 0) ∧ (f 0 = C.2) → triangle_area = 6 := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_ABC_l763_76395


namespace NUMINAMATH_CALUDE_hundred_with_fewer_threes_l763_76389

/-- An arithmetic expression using only the number 3 and basic operations -/
inductive Expr
  | three : Expr
  | add : Expr → Expr → Expr
  | sub : Expr → Expr → Expr
  | mul : Expr → Expr → Expr
  | div : Expr → Expr → Expr

/-- Evaluate an expression to a rational number -/
def eval : Expr → ℚ
  | Expr.three => 3
  | Expr.add e1 e2 => eval e1 + eval e2
  | Expr.sub e1 e2 => eval e1 - eval e2
  | Expr.mul e1 e2 => eval e1 * eval e2
  | Expr.div e1 e2 => eval e1 / eval e2

/-- Count the number of 3's in an expression -/
def count_threes : Expr → Nat
  | Expr.three => 1
  | Expr.add e1 e2 => count_threes e1 + count_threes e2
  | Expr.sub e1 e2 => count_threes e1 + count_threes e2
  | Expr.mul e1 e2 => count_threes e1 + count_threes e2
  | Expr.div e1 e2 => count_threes e1 + count_threes e2

/-- Theorem: There exists an expression that evaluates to 100 using fewer than 10 threes -/
theorem hundred_with_fewer_threes : ∃ e : Expr, eval e = 100 ∧ count_threes e < 10 := by
  sorry

end NUMINAMATH_CALUDE_hundred_with_fewer_threes_l763_76389


namespace NUMINAMATH_CALUDE_cherry_pits_correct_l763_76331

/-- The number of cherry pits Kim planted -/
def cherry_pits : ℕ := 80

/-- The fraction of cherry pits that sprout -/
def sprout_rate : ℚ := 1/4

/-- The number of saplings Kim sold -/
def saplings_sold : ℕ := 6

/-- The number of saplings left after selling -/
def saplings_left : ℕ := 14

/-- Theorem stating that the number of cherry pits is correct given the conditions -/
theorem cherry_pits_correct : 
  (↑cherry_pits * sprout_rate : ℚ) - saplings_sold = saplings_left := by sorry

end NUMINAMATH_CALUDE_cherry_pits_correct_l763_76331


namespace NUMINAMATH_CALUDE_circular_path_length_l763_76383

/-- The length of a circular path given specific walking conditions -/
theorem circular_path_length
  (step_length_1 : ℝ)
  (step_length_2 : ℝ)
  (total_footprints : ℕ)
  (h1 : step_length_1 = 0.54)  -- 54 cm in meters
  (h2 : step_length_2 = 0.72)  -- 72 cm in meters
  (h3 : total_footprints = 60)
  (h4 : ∃ (n m : ℕ), n * step_length_1 = m * step_length_2) -- Both complete one lap
  : ∃ (path_length : ℝ), path_length = 21.6 :=
by
  sorry

end NUMINAMATH_CALUDE_circular_path_length_l763_76383


namespace NUMINAMATH_CALUDE_quadratic_inequality_problem_l763_76309

-- Define the quadratic function
def f (a c x : ℝ) := a * x^2 + x + c

-- Define the solution set of the original inequality
def S := {x : ℝ | 1 < x ∧ x < 3}

-- Define the solution set A
def A (a c : ℝ) := {x : ℝ | a * x^2 + 2*x + 4*c > 0}

-- Define the solution set B
def B (a c m : ℝ) := {x : ℝ | 3*a*x + c*m < 0}

-- State the theorem
theorem quadratic_inequality_problem 
  (h1 : ∀ x, x ∈ S ↔ f a c x > 0)
  (h2 : A a c ⊆ B a c m) :
  a = -1/4 ∧ c = -3/4 ∧ m ≥ -2 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_problem_l763_76309


namespace NUMINAMATH_CALUDE_parallel_line_distance_is_twelve_l763_76305

/-- Represents a circle with three equally spaced parallel lines intersecting it. -/
structure CircleWithParallelLines where
  /-- The radius of the circle -/
  radius : ℝ
  /-- The distance between adjacent parallel lines -/
  line_distance : ℝ
  /-- The length of the first chord -/
  chord1 : ℝ
  /-- The length of the second chord -/
  chord2 : ℝ
  /-- The length of the third chord -/
  chord3 : ℝ
  /-- The first chord has length 40 -/
  chord1_length : chord1 = 40
  /-- The second chord has length 36 -/
  chord2_length : chord2 = 36
  /-- The third chord has length 40 -/
  chord3_length : chord3 = 40

/-- Theorem stating that the distance between adjacent parallel lines is 12 -/
theorem parallel_line_distance_is_twelve (c : CircleWithParallelLines) : c.line_distance = 12 := by
  sorry


end NUMINAMATH_CALUDE_parallel_line_distance_is_twelve_l763_76305


namespace NUMINAMATH_CALUDE_adam_new_books_l763_76355

theorem adam_new_books (initial_books sold_books final_books : ℕ) 
  (h1 : initial_books = 33) 
  (h2 : sold_books = 11)
  (h3 : final_books = 45) :
  final_books - (initial_books - sold_books) = 23 := by
  sorry

end NUMINAMATH_CALUDE_adam_new_books_l763_76355


namespace NUMINAMATH_CALUDE_jogging_track_circumference_jogging_track_circumference_value_l763_76350

/-- The circumference of a circular jogging track where two people walking in opposite directions meet. -/
theorem jogging_track_circumference (deepak_speed wife_speed : ℝ) (meeting_time : ℝ) : ℝ :=
  let deepak_speed := 4.5
  let wife_speed := 3.75
  let meeting_time := 3.84 / 60
  2 * (deepak_speed + wife_speed) * meeting_time

/-- The circumference of the jogging track is 1.056 km. -/
theorem jogging_track_circumference_value :
  jogging_track_circumference 4.5 3.75 (3.84 / 60) = 1.056 := by
  sorry

end NUMINAMATH_CALUDE_jogging_track_circumference_jogging_track_circumference_value_l763_76350


namespace NUMINAMATH_CALUDE_total_value_correct_l763_76382

/-- The total value of an imported item -/
def total_value : ℝ := 2580

/-- The import tax rate -/
def tax_rate : ℝ := 0.07

/-- The tax-free threshold -/
def tax_free_threshold : ℝ := 1000

/-- The amount of import tax paid -/
def tax_paid : ℝ := 110.60

/-- Theorem stating that the total value is correct given the conditions -/
theorem total_value_correct : 
  tax_rate * (total_value - tax_free_threshold) = tax_paid := by sorry

end NUMINAMATH_CALUDE_total_value_correct_l763_76382


namespace NUMINAMATH_CALUDE_square_area_relation_l763_76370

theorem square_area_relation (a b : ℝ) : 
  let diagonal_I := 2*a + 2*b
  let area_I := (diagonal_I / Real.sqrt 2)^2
  let area_II := 3 * area_I
  area_II = 6 * (a + b)^2 := by
sorry

end NUMINAMATH_CALUDE_square_area_relation_l763_76370


namespace NUMINAMATH_CALUDE_divisible_by_nine_l763_76367

/-- Sum of digits function -/
def sum_of_digits (n : ℤ) : ℤ := sorry

theorem divisible_by_nine (x : ℤ) 
  (h : sum_of_digits x = sum_of_digits (3 * x)) : 
  9 ∣ x := by sorry

end NUMINAMATH_CALUDE_divisible_by_nine_l763_76367


namespace NUMINAMATH_CALUDE_remainder_17_pow_63_mod_7_l763_76315

theorem remainder_17_pow_63_mod_7 : 17^63 % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_17_pow_63_mod_7_l763_76315


namespace NUMINAMATH_CALUDE_three_fish_added_l763_76316

/-- The number of fish added to a barrel -/
def fish_added (initial_a initial_b final_total : ℕ) : ℕ :=
  final_total - (initial_a + initial_b)

/-- Theorem: Given the initial numbers of fish and the final total, prove that 3 fish were added -/
theorem three_fish_added : fish_added 4 3 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_fish_added_l763_76316


namespace NUMINAMATH_CALUDE_condition_p_sufficient_not_necessary_for_q_l763_76359

theorem condition_p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, |x + 1| ≤ 2 → -3 ≤ x ∧ x ≤ 2) ∧
  (∃ x : ℝ, -3 ≤ x ∧ x ≤ 2 ∧ ¬(|x + 1| ≤ 2)) := by
  sorry

end NUMINAMATH_CALUDE_condition_p_sufficient_not_necessary_for_q_l763_76359


namespace NUMINAMATH_CALUDE_least_third_side_length_l763_76356

theorem least_third_side_length (a b c : ℝ) : 
  a = 8 → b = 6 → c > 0 → a^2 + b^2 ≤ c^2 → c ≥ 10 := by
  sorry

end NUMINAMATH_CALUDE_least_third_side_length_l763_76356


namespace NUMINAMATH_CALUDE_choose_leaders_count_l763_76345

/-- A club with members divided by gender and class -/
structure Club where
  total_members : ℕ
  boys : ℕ
  girls : ℕ
  classes : ℕ
  boys_per_class : ℕ
  girls_per_class : ℕ

/-- The number of ways to choose a president and vice-president -/
def choose_leaders (c : Club) : ℕ := sorry

/-- The specific club configuration -/
def my_club : Club := {
  total_members := 24,
  boys := 12,
  girls := 12,
  classes := 2,
  boys_per_class := 6,
  girls_per_class := 6
}

/-- Theorem stating the number of ways to choose leaders for the given club -/
theorem choose_leaders_count : choose_leaders my_club = 144 := by sorry

end NUMINAMATH_CALUDE_choose_leaders_count_l763_76345


namespace NUMINAMATH_CALUDE_floor_equation_solution_l763_76374

theorem floor_equation_solution (n : ℤ) : (Int.floor (n^2 / 4 : ℚ) - Int.floor (n / 2 : ℚ)^2 = 5) ↔ n = 11 := by
  sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l763_76374


namespace NUMINAMATH_CALUDE_board_game_investment_l763_76323

/-- Proves that the investment in equipment for a board game business is $10,410 -/
theorem board_game_investment
  (manufacture_cost : ℝ)
  (selling_price : ℝ)
  (break_even_quantity : ℕ)
  (h1 : manufacture_cost = 2.65)
  (h2 : selling_price = 20)
  (h3 : break_even_quantity = 600)
  (h4 : selling_price * break_even_quantity = 
        manufacture_cost * break_even_quantity + investment) :
  investment = 10410 := by
  sorry

end NUMINAMATH_CALUDE_board_game_investment_l763_76323


namespace NUMINAMATH_CALUDE_arithmetic_sequence_terms_l763_76312

theorem arithmetic_sequence_terms (a₁ a₂ aₙ : ℕ) (h1 : a₁ = 6) (h2 : a₂ = 9) (h3 : aₙ = 300) :
  ∃ n : ℕ, n = 99 ∧ aₙ = a₁ + (n - 1) * (a₂ - a₁) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_terms_l763_76312


namespace NUMINAMATH_CALUDE_nested_fraction_equals_five_thirds_l763_76332

theorem nested_fraction_equals_five_thirds :
  1 + (1 / (1 + (1 / (1 + 1)))) = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_equals_five_thirds_l763_76332


namespace NUMINAMATH_CALUDE_range_of_4a_minus_2b_l763_76379

theorem range_of_4a_minus_2b (a b : ℝ) 
  (h1 : -1 ≤ a - b ∧ a - b ≤ 1) 
  (h2 : 2 ≤ a + b ∧ a + b ≤ 4) : 
  ∃ (x : ℝ), -1 ≤ x ∧ x ≤ 7 ∧ x = 4*a - 2*b :=
by sorry

end NUMINAMATH_CALUDE_range_of_4a_minus_2b_l763_76379


namespace NUMINAMATH_CALUDE_parabola_vertex_l763_76387

/-- Given a quadratic function f(x) = -x^2 + cx + d where the solution to f(x) ≤ 0
    is (-∞, -5] ∪ [1, ∞), the vertex of the parabola defined by f(x) is (3, 4). -/
theorem parabola_vertex (c d : ℝ) :
  let f : ℝ → ℝ := λ x => -x^2 + c*x + d
  (∀ x, f x ≤ 0 ↔ x ≤ -5 ∨ x ≥ 1) →
  (∃! p : ℝ × ℝ, p.1 = 3 ∧ p.2 = 4 ∧ ∀ x, f x ≤ f p.1) :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l763_76387


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l763_76371

theorem arithmetic_sequence_length :
  ∀ (a₁ d l : ℝ) (n : ℕ),
    a₁ = 3.5 →
    d = 4 →
    l = 55.5 →
    l = a₁ + (n - 1) * d →
    n = 14 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l763_76371


namespace NUMINAMATH_CALUDE_plate_selection_probability_l763_76348

def red_plates : ℕ := 6
def light_blue_plates : ℕ := 3
def dark_blue_plates : ℕ := 3

def total_plates : ℕ := red_plates + light_blue_plates + dark_blue_plates

def favorable_outcomes : ℕ := 
  (red_plates.choose 2) + 
  (light_blue_plates.choose 2) + 
  (dark_blue_plates.choose 2) + 
  (light_blue_plates * dark_blue_plates)

def total_outcomes : ℕ := total_plates.choose 2

theorem plate_selection_probability : 
  (favorable_outcomes : ℚ) / total_outcomes = 5 / 11 := by sorry

end NUMINAMATH_CALUDE_plate_selection_probability_l763_76348


namespace NUMINAMATH_CALUDE_power_two_2005_mod_7_l763_76366

theorem power_two_2005_mod_7 : 2^2005 % 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_two_2005_mod_7_l763_76366


namespace NUMINAMATH_CALUDE_smallest_m_with_divisibility_l763_76369

def is_divisible (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem smallest_m_with_divisibility : 
  (∀ M : ℕ, M > 0 ∧ M < 250 → 
    ¬(is_divisible M (5^3) ∧ is_divisible (M+1) (2^3) ∧ is_divisible (M+2) (3^2) ∨
      is_divisible M (5^3) ∧ is_divisible (M+1) (3^2) ∧ is_divisible (M+2) (2^3) ∨
      is_divisible M (2^3) ∧ is_divisible (M+1) (5^3) ∧ is_divisible (M+2) (3^2) ∨
      is_divisible M (2^3) ∧ is_divisible (M+1) (3^2) ∧ is_divisible (M+2) (5^3) ∨
      is_divisible M (3^2) ∧ is_divisible (M+1) (5^3) ∧ is_divisible (M+2) (2^3) ∨
      is_divisible M (3^2) ∧ is_divisible (M+1) (2^3) ∧ is_divisible (M+2) (5^3))) ∧
  (is_divisible 250 (5^3) ∧ is_divisible 252 (2^3) ∧ is_divisible 252 (3^2)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_m_with_divisibility_l763_76369


namespace NUMINAMATH_CALUDE_admission_probability_l763_76365

theorem admission_probability (P_A P_B : ℝ) (h_A : P_A = 0.6) (h_B : P_B = 0.7) 
  (h_indep : P_A + P_B - P_A * P_B = P_A + P_B - (P_A * P_B)) : 
  P_A + P_B - P_A * P_B = 0.88 := by
sorry

end NUMINAMATH_CALUDE_admission_probability_l763_76365


namespace NUMINAMATH_CALUDE_students_left_fourth_grade_students_left_l763_76326

theorem students_left (initial_students : ℝ) (final_students : ℝ) (transferred_students : ℝ) :
  initial_students ≥ final_students + transferred_students →
  initial_students - (final_students + transferred_students) =
  initial_students - final_students - transferred_students :=
by sorry

def calculate_students_left (initial_students : ℝ) (final_students : ℝ) (transferred_students : ℝ) : ℝ :=
  initial_students - final_students - transferred_students

theorem fourth_grade_students_left :
  let initial_students : ℝ := 42.0
  let final_students : ℝ := 28.0
  let transferred_students : ℝ := 10.0
  calculate_students_left initial_students final_students transferred_students = 4 :=
by sorry

end NUMINAMATH_CALUDE_students_left_fourth_grade_students_left_l763_76326


namespace NUMINAMATH_CALUDE_real_roots_necessary_condition_l763_76375

theorem real_roots_necessary_condition (a : ℝ) :
  (∃ x : ℝ, x^2 - a*x + 1 = 0) →
  (a ≥ 1 ∨ a ≤ -2) :=
by sorry

end NUMINAMATH_CALUDE_real_roots_necessary_condition_l763_76375


namespace NUMINAMATH_CALUDE_root_bounds_l763_76378

theorem root_bounds (x : ℝ) : 
  x^2014 - 100*x + 1 = 0 → 1/100 ≤ x ∧ x ≤ 100^(1/2013) := by
  sorry

end NUMINAMATH_CALUDE_root_bounds_l763_76378


namespace NUMINAMATH_CALUDE_probability_ace_king_queen_standard_deck_l763_76351

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (num_aces : ℕ)
  (num_kings : ℕ)
  (num_queens : ℕ)

/-- Calculates the probability of drawing an Ace, then a King, then a Queen from a standard deck without replacement -/
def probability_ace_king_queen (d : Deck) : ℚ :=
  (d.num_aces : ℚ) / d.total_cards *
  (d.num_kings : ℚ) / (d.total_cards - 1) *
  (d.num_queens : ℚ) / (d.total_cards - 2)

/-- Theorem stating the probability of drawing an Ace, then a King, then a Queen from a standard 52-card deck -/
theorem probability_ace_king_queen_standard_deck :
  probability_ace_king_queen ⟨52, 4, 4, 4⟩ = 8 / 16575 := by
  sorry

end NUMINAMATH_CALUDE_probability_ace_king_queen_standard_deck_l763_76351


namespace NUMINAMATH_CALUDE_square_mod_five_not_three_l763_76368

theorem square_mod_five_not_three (n : ℕ) : (n ^ 2) % 5 ≠ 3 := by
  sorry

end NUMINAMATH_CALUDE_square_mod_five_not_three_l763_76368


namespace NUMINAMATH_CALUDE_cube_octahedron_surface_area_ratio_l763_76318

/-- The ratio of the surface area of a cube to the surface area of an inscribed regular octahedron --/
theorem cube_octahedron_surface_area_ratio :
  ∀ (cube_side_length : ℝ) (octahedron_side_length : ℝ),
    cube_side_length = 2 →
    octahedron_side_length = Real.sqrt 2 →
    (6 * cube_side_length^2) / (2 * Real.sqrt 3 * octahedron_side_length^2) = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_octahedron_surface_area_ratio_l763_76318


namespace NUMINAMATH_CALUDE_max_abs_z_l763_76322

theorem max_abs_z (z : ℂ) : 
  Complex.abs (z + 3 + 4*I) ≤ 2 → 
  ∃ (w : ℂ), Complex.abs (w + 3 + 4*I) ≤ 2 ∧ 
             ∀ (u : ℂ), Complex.abs (u + 3 + 4*I) ≤ 2 → Complex.abs u ≤ Complex.abs w ∧
             Complex.abs w = 7 :=
by sorry

end NUMINAMATH_CALUDE_max_abs_z_l763_76322


namespace NUMINAMATH_CALUDE_negation_of_zero_product_property_l763_76385

theorem negation_of_zero_product_property :
  (¬ ∀ (x y : ℝ), x * y = 0 → x = 0 ∨ y = 0) ↔
  (∃ (x y : ℝ), x * y = 0 ∧ x ≠ 0 ∧ y ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_zero_product_property_l763_76385


namespace NUMINAMATH_CALUDE_error_clock_correct_time_fraction_l763_76335

/-- Represents a 24-hour digital clock with specific display errors -/
structure ErrorClock where
  /-- The clock displays 9 instead of 1 -/
  error_one : Nat
  /-- The clock displays 5 instead of 2 -/
  error_two : Nat

/-- Calculates the fraction of the day an ErrorClock shows the correct time -/
def correct_time_fraction (clock : ErrorClock) : Rat :=
  sorry

/-- Theorem stating that the ErrorClock shows the correct time for 7/36 of the day -/
theorem error_clock_correct_time_fraction :
  ∀ (clock : ErrorClock),
  clock.error_one = 9 ∧ clock.error_two = 5 →
  correct_time_fraction clock = 7 / 36 := by
  sorry

end NUMINAMATH_CALUDE_error_clock_correct_time_fraction_l763_76335


namespace NUMINAMATH_CALUDE_root_exists_in_interval_l763_76396

def f (x : ℝ) := x^3 - 9

theorem root_exists_in_interval :
  ∃ (x : ℝ), x ∈ Set.Ioo 2 3 ∧ f x = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_root_exists_in_interval_l763_76396


namespace NUMINAMATH_CALUDE_complement_hit_at_least_once_l763_76362

-- Define the sample space
def Ω : Type := Bool × Bool

-- Define the event of hitting the target at least once
def hit_at_least_once (ω : Ω) : Prop :=
  ω.1 ∨ ω.2

-- Define the event of missing the target both times
def miss_both_times (ω : Ω) : Prop :=
  ¬ω.1 ∧ ¬ω.2

-- Theorem stating that the complement of hitting at least once
-- is equivalent to missing both times
theorem complement_hit_at_least_once (ω : Ω) :
  ¬(hit_at_least_once ω) ↔ miss_both_times ω :=
sorry

end NUMINAMATH_CALUDE_complement_hit_at_least_once_l763_76362


namespace NUMINAMATH_CALUDE_cow_count_l763_76329

/-- Given a group of ducks and cows, where the total number of legs is 36 more than
    twice the number of heads, prove that the number of cows is 18. -/
theorem cow_count (ducks cows : ℕ) : 
  (2 * ducks + 4 * cows = 2 * (ducks + cows) + 36) → cows = 18 := by
  sorry

end NUMINAMATH_CALUDE_cow_count_l763_76329


namespace NUMINAMATH_CALUDE_baker_a_remaining_pastries_l763_76328

/-- Baker A's initial number of cakes -/
def baker_a_initial_cakes : ℕ := 7

/-- Baker A's initial number of pastries -/
def baker_a_initial_pastries : ℕ := 148

/-- Baker B's initial number of cakes -/
def baker_b_initial_cakes : ℕ := 10

/-- Baker B's initial number of pastries -/
def baker_b_initial_pastries : ℕ := 200

/-- Number of pastries Baker A sold -/
def baker_a_sold_pastries : ℕ := 103

/-- Theorem: Baker A will have 71 pastries left after redistribution and selling -/
theorem baker_a_remaining_pastries :
  (baker_a_initial_pastries + baker_b_initial_pastries) / 2 - baker_a_sold_pastries = 71 := by
  sorry

end NUMINAMATH_CALUDE_baker_a_remaining_pastries_l763_76328


namespace NUMINAMATH_CALUDE_triangle_stack_sum_impossible_l763_76384

theorem triangle_stack_sum_impossible : ¬ ∃ k : ℕ+, 6 * k = 165 := by
  sorry

end NUMINAMATH_CALUDE_triangle_stack_sum_impossible_l763_76384


namespace NUMINAMATH_CALUDE_sequence_inequality_l763_76349

/-- The sum of the first n terms of the sequence -/
def S (n : ℕ+) : ℤ := -2 * n.val ^ 2 + 3 * n.val

/-- The n-th term of the sequence -/
def a (n : ℕ+) : ℤ := -4 * n.val + 5

/-- Theorem stating the relationship between na_n, S_n, and na_1 for n ≥ 2 -/
theorem sequence_inequality (n : ℕ+) (h : 2 ≤ n.val) :
  (n.val : ℤ) * a n < S n ∧ S n < (n.val : ℤ) * a 1 :=
sorry

end NUMINAMATH_CALUDE_sequence_inequality_l763_76349


namespace NUMINAMATH_CALUDE_inverse_contrapositive_equivalence_l763_76303

theorem inverse_contrapositive_equivalence (a b : ℝ) :
  (¬(a = 0 ∧ b = 0) → a^2 + b^2 ≠ 0) ↔ (a^2 + b^2 = 0 → a = 0 ∧ b = 0) :=
by sorry

end NUMINAMATH_CALUDE_inverse_contrapositive_equivalence_l763_76303


namespace NUMINAMATH_CALUDE_february_first_is_friday_l763_76373

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a date in February -/
structure FebruaryDate where
  day : Nat
  dayOfWeek : DayOfWeek

/-- Given that February 13th is a Wednesday, prove that February 1st is a Friday -/
theorem february_first_is_friday 
  (feb13 : FebruaryDate)
  (h13 : feb13.day = 13)
  (hWed : feb13.dayOfWeek = DayOfWeek.Wednesday) :
  ∃ (feb1 : FebruaryDate), feb1.day = 1 ∧ feb1.dayOfWeek = DayOfWeek.Friday :=
sorry

end NUMINAMATH_CALUDE_february_first_is_friday_l763_76373


namespace NUMINAMATH_CALUDE_y1_value_l763_76393

theorem y1_value (y1 y2 y3 : Real) 
  (h1 : 0 ≤ y3 ∧ y3 ≤ y2 ∧ y2 ≤ y1 ∧ y1 ≤ 1)
  (h2 : (1-y1)^2 + 2*(y1-y2)^2 + 2*(y2-y3)^2 + y3^2 = 1/2) :
  y1 = 3/4 := by
sorry

end NUMINAMATH_CALUDE_y1_value_l763_76393


namespace NUMINAMATH_CALUDE_cookie_ratio_proof_l763_76347

theorem cookie_ratio_proof (raisin_cookies oatmeal_cookies : ℕ) : 
  raisin_cookies = 42 → 
  raisin_cookies + oatmeal_cookies = 49 → 
  raisin_cookies / oatmeal_cookies = 6 := by
sorry

end NUMINAMATH_CALUDE_cookie_ratio_proof_l763_76347


namespace NUMINAMATH_CALUDE_circle_symmetry_l763_76344

-- Define the original circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the line l
def line_l (x y : ℝ) : Prop := x + y = 1

-- Define the symmetric circle
def symmetric_circle (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 1

-- Theorem statement
theorem circle_symmetry :
  ∀ (x y : ℝ), circle_C x y ∧ line_l x y → symmetric_circle x y :=
by sorry

end NUMINAMATH_CALUDE_circle_symmetry_l763_76344


namespace NUMINAMATH_CALUDE_semicircle_area_ratio_l763_76336

/-- The ratio of the combined areas of two semicircles to the area of their circumscribing circle -/
theorem semicircle_area_ratio (r : ℝ) (h : r > 0) :
  let semicircle_area := π * (r / 2)^2 / 2
  let circle_area := π * r^2
  2 * semicircle_area / circle_area = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_semicircle_area_ratio_l763_76336


namespace NUMINAMATH_CALUDE_student_answer_difference_l763_76311

theorem student_answer_difference (number : ℕ) (h : number = 288) : 
  (5 : ℚ) / 6 * number - (5 : ℚ) / 16 * number = 150 := by
  sorry

end NUMINAMATH_CALUDE_student_answer_difference_l763_76311


namespace NUMINAMATH_CALUDE_circle_equation_k_l763_76390

/-- The equation of a circle with center (h, k) and radius r is (x - h)^2 + (y - k)^2 = r^2 -/
theorem circle_equation_k (k : ℝ) : 
  (∀ x y : ℝ, x^2 + 8*x + y^2 + 10*y - k = 0 ↔ (x + 4)^2 + (y + 5)^2 = 25) → 
  k = -16 := by
sorry

end NUMINAMATH_CALUDE_circle_equation_k_l763_76390


namespace NUMINAMATH_CALUDE_salem_population_decrease_l763_76376

def salem_leesburg_ratio : ℕ := 15
def leesburg_population : ℕ := 58940
def salem_women_population : ℕ := 377050

def salem_original_population : ℕ := salem_leesburg_ratio * leesburg_population
def salem_current_population : ℕ := 2 * salem_women_population

theorem salem_population_decrease :
  salem_original_population - salem_current_population = 130000 :=
by sorry

end NUMINAMATH_CALUDE_salem_population_decrease_l763_76376


namespace NUMINAMATH_CALUDE_function_and_monotonicity_l763_76399

-- Define the function f(x)
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x + 1

-- Define the derivative of f(x)
def f' (a b : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + b

theorem function_and_monotonicity 
  (a b : ℝ) 
  (h1 : f a b 1 = -3)  -- f(1) = -3
  (h2 : f' a b 1 = 0)  -- f'(1) = 0
  : 
  (∀ x, f a b x = 2 * x^3 - 6 * x + 1) ∧  -- Explicit formula
  (∀ x, x < -1 → (f' a b x > 0)) ∧       -- Monotonically increasing for x < -1
  (∀ x, x > 1 → (f' a b x > 0))          -- Monotonically increasing for x > 1
  := by sorry

end NUMINAMATH_CALUDE_function_and_monotonicity_l763_76399


namespace NUMINAMATH_CALUDE_quadratic_properties_l763_76357

theorem quadratic_properties (a b c : ℝ) (ha : a ≠ 0) :
  (∃ m n : ℝ, m ≠ n ∧ a * m^2 + b * m + c = a * n^2 + b * n + c) ∧
  (a * c < 0 → ∃ m n : ℝ, m > n ∧ a * m^2 + b * m + c < 0 ∧ 0 < a * n^2 + b * n + c) ∧
  (a * b > 0 → ∃ p q : ℝ, p ≠ q ∧ a * p^2 + b * p + c = a * q^2 + b * q + c ∧ p + q < 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l763_76357


namespace NUMINAMATH_CALUDE_calculation_proof_l763_76307

theorem calculation_proof : (3752 / (39 * 2) + 5030 / (39 * 10) : ℚ) = 61 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l763_76307


namespace NUMINAMATH_CALUDE_calculation_proof_l763_76338

theorem calculation_proof : (3.25 - 1.57) * 2 = 3.36 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l763_76338
