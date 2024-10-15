import Mathlib

namespace NUMINAMATH_CALUDE_inequality_theorem_l3784_378458

/-- A function satisfying the given conditions -/
def SatisfiesConditions (f : ℝ → ℝ) : Prop :=
  (∀ x, DifferentiableAt ℝ f x) ∧
  (∀ x, DifferentiableAt ℝ (deriv f) x) ∧
  f 0 = 1 ∧
  deriv f 0 = 0 ∧
  ∀ x ≥ 0, deriv (deriv f) x - 5 * deriv f x + 6 * f x ≥ 0

/-- The main theorem -/
theorem inequality_theorem (f : ℝ → ℝ) (h : SatisfiesConditions f) :
  ∀ x ≥ 0, f x ≥ 3 * Real.exp (2 * x) - 2 * Real.exp (3 * x) := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_l3784_378458


namespace NUMINAMATH_CALUDE_winter_sales_proof_l3784_378405

/-- Represents the sales of hamburgers in millions for each season and the total year -/
structure HamburgerSales where
  spring_summer : ℝ
  fall : ℝ
  winter : ℝ
  total : ℝ

/-- Given the conditions of the hamburger sales, prove that winter sales are 4 million -/
theorem winter_sales_proof (sales : HamburgerSales) 
  (h1 : sales.total = 20)
  (h2 : sales.spring_summer = 0.6 * sales.total)
  (h3 : sales.fall = 0.2 * sales.total)
  (h4 : sales.total = sales.spring_summer + sales.fall + sales.winter) :
  sales.winter = 4 := by
  sorry

#check winter_sales_proof

end NUMINAMATH_CALUDE_winter_sales_proof_l3784_378405


namespace NUMINAMATH_CALUDE_lcm_gcd_problem_l3784_378472

theorem lcm_gcd_problem (a b : ℕ+) 
  (h1 : Nat.lcm a b = 2520)
  (h2 : Nat.gcd a b = 30)
  (h3 : a = 150) :
  b = 504 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_problem_l3784_378472


namespace NUMINAMATH_CALUDE_jake_peaches_l3784_378473

/-- Given the number of peaches Steven, Jill, and Jake have, prove that Jake has 8 peaches. -/
theorem jake_peaches (steven jill jake : ℕ) 
  (h1 : steven = 15)
  (h2 : steven = jill + 14)
  (h3 : jake + 7 = steven) : 
  jake = 8 := by
sorry

end NUMINAMATH_CALUDE_jake_peaches_l3784_378473


namespace NUMINAMATH_CALUDE_time_to_destination_l3784_378435

/-- The time it takes to reach a destination given relative speeds and distances -/
theorem time_to_destination
  (your_speed : ℝ)
  (harris_speed : ℝ)
  (harris_time : ℝ)
  (distance_ratio : ℝ)
  (h1 : your_speed = 3 * harris_speed)
  (h2 : harris_time = 3)
  (h3 : distance_ratio = 5) :
  your_speed * (distance_ratio * harris_time / your_speed) = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_time_to_destination_l3784_378435


namespace NUMINAMATH_CALUDE_pie_baking_difference_l3784_378429

def alice_bake_time : ℕ := 5
def bob_bake_time : ℕ := 6
def charlie_bake_time : ℕ := 7
def total_time : ℕ := 90

def pies_baked (bake_time : ℕ) : ℕ := total_time / bake_time

theorem pie_baking_difference :
  (pies_baked alice_bake_time) - (pies_baked bob_bake_time) + 
  (pies_baked alice_bake_time) - (pies_baked charlie_bake_time) = 9 := by
  sorry

end NUMINAMATH_CALUDE_pie_baking_difference_l3784_378429


namespace NUMINAMATH_CALUDE_our_parabola_properties_l3784_378450

/-- A parabola with specific properties -/
structure Parabola where
  -- The equation of the parabola in the form ax^2 + bxy + cy^2 + dx + ey + f = 0
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ
  f : ℤ
  c_pos : c > 0
  gcd_one : Nat.gcd (Int.natAbs a) (Nat.gcd (Int.natAbs b) (Nat.gcd (Int.natAbs c) (Nat.gcd (Int.natAbs d) (Nat.gcd (Int.natAbs e) (Int.natAbs f))))) = 1

/-- The specific parabola we are interested in -/
def our_parabola : Parabola where
  a := 0
  b := 0
  c := 1
  d := -8
  e := -8
  f := 16
  c_pos := by sorry
  gcd_one := by sorry

/-- Theorem stating that our_parabola satisfies all the required properties -/
theorem our_parabola_properties :
  -- Passes through (2,8)
  (2 : ℝ)^2 * our_parabola.a + 2 * 8 * our_parabola.b + 8^2 * our_parabola.c + 2 * our_parabola.d + 8 * our_parabola.e + our_parabola.f = 0 ∧
  -- Vertex lies on y-axis (x-coordinate of vertex is 0)
  our_parabola.b^2 - 4 * our_parabola.a * our_parabola.c = 0 ∧
  -- y-coordinate of focus is 4
  (our_parabola.e^2 - 4 * our_parabola.c * our_parabola.f) / (4 * our_parabola.c^2) = 4 ∧
  -- Axis of symmetry is parallel to x-axis
  our_parabola.b = 0 ∧ our_parabola.a = 0 := by
  sorry

end NUMINAMATH_CALUDE_our_parabola_properties_l3784_378450


namespace NUMINAMATH_CALUDE_number_times_one_sixth_squared_l3784_378483

theorem number_times_one_sixth_squared (x : ℝ) : x * (1/6)^2 = 6^3 ↔ x = 7776 := by
  sorry

end NUMINAMATH_CALUDE_number_times_one_sixth_squared_l3784_378483


namespace NUMINAMATH_CALUDE_divisibility_condition_l3784_378434

theorem divisibility_condition (a b : ℕ+) (h : b ≥ 2) :
  (2^a.val + 1) % (2^b.val - 1) = 0 ↔ b = 2 ∧ a.val % 2 = 1 := by
sorry

end NUMINAMATH_CALUDE_divisibility_condition_l3784_378434


namespace NUMINAMATH_CALUDE_remainder_2013_div_85_l3784_378441

theorem remainder_2013_div_85 : 2013 % 85 = 58 := by
  sorry

end NUMINAMATH_CALUDE_remainder_2013_div_85_l3784_378441


namespace NUMINAMATH_CALUDE_collinearity_condition_l3784_378495

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define points A₁, B₁, C₁
variables (A₁ B₁ C₁ : ℝ × ℝ)

-- Define the R value as in Problem 191
def R (ABC : Triangle) (A₁ B₁ C₁ : ℝ × ℝ) : ℝ := sorry

-- Define a function to check if a point is on a side of the triangle
def onSide (ABC : Triangle) (P : ℝ × ℝ) : Bool := sorry

-- Define a function to count how many points are on the sides of the triangle
def countOnSides (ABC : Triangle) (A₁ B₁ C₁ : ℝ × ℝ) : Nat := sorry

-- Define collinearity
def collinear (A₁ B₁ C₁ : ℝ × ℝ) : Prop := sorry

-- The main theorem
theorem collinearity_condition (ABC : Triangle) (A₁ B₁ C₁ : ℝ × ℝ) :
  collinear A₁ B₁ C₁ ↔ R ABC A₁ B₁ C₁ = 1 ∧ Even (countOnSides ABC A₁ B₁ C₁) :=
sorry

end NUMINAMATH_CALUDE_collinearity_condition_l3784_378495


namespace NUMINAMATH_CALUDE_coin_value_calculation_l3784_378439

/-- The value of a penny in dollars -/
def penny_value : ℚ := 0.01

/-- The value of a nickel in dollars -/
def nickel_value : ℚ := 0.05

/-- The value of a dime in dollars -/
def dime_value : ℚ := 0.10

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 0.25

/-- The value of a half-dollar in dollars -/
def half_dollar_value : ℚ := 0.50

/-- The number of pennies -/
def num_pennies : ℕ := 9

/-- The number of nickels -/
def num_nickels : ℕ := 4

/-- The number of dimes -/
def num_dimes : ℕ := 3

/-- The number of quarters -/
def num_quarters : ℕ := 7

/-- The number of half-dollars -/
def num_half_dollars : ℕ := 5

/-- The total value of the coins in dollars -/
def total_value : ℚ :=
  num_pennies * penny_value +
  num_nickels * nickel_value +
  num_dimes * dime_value +
  num_quarters * quarter_value +
  num_half_dollars * half_dollar_value

theorem coin_value_calculation :
  total_value = 4.84 := by sorry

end NUMINAMATH_CALUDE_coin_value_calculation_l3784_378439


namespace NUMINAMATH_CALUDE_polynomial_equality_l3784_378428

-- Define the polynomials P and Q
variable (P Q : ℝ → ℝ)

-- Define the property of being a nonconstant polynomial
def IsNonconstantPolynomial (f : ℝ → ℝ) : Prop := sorry

-- Define the theorem
theorem polynomial_equality
  (hP : IsNonconstantPolynomial P)
  (hQ : IsNonconstantPolynomial Q)
  (h : ∀ y : ℝ, ⌊P y⌋ = ⌊Q y⌋) :
  ∀ x : ℝ, P x = Q x := by sorry

end NUMINAMATH_CALUDE_polynomial_equality_l3784_378428


namespace NUMINAMATH_CALUDE_mans_rowing_speed_l3784_378411

/-- The rowing speed of a man in still water, given his speeds with and against the stream -/
theorem mans_rowing_speed (speed_with_stream speed_against_stream : ℝ) 
  (h1 : speed_with_stream = 18)
  (h2 : speed_against_stream = 8) :
  (speed_with_stream + speed_against_stream) / 2 = 13 :=
by sorry

end NUMINAMATH_CALUDE_mans_rowing_speed_l3784_378411


namespace NUMINAMATH_CALUDE_josh_marbles_l3784_378449

theorem josh_marbles (lost : ℕ) (left : ℕ) (initial : ℕ) : 
  lost = 7 → left = 9 → initial = lost + left → initial = 16 := by
  sorry

end NUMINAMATH_CALUDE_josh_marbles_l3784_378449


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3784_378492

theorem quadratic_inequality_range (a : ℝ) :
  (∀ x : ℝ, a * x^2 + 2 * x + 3 > 0) ↔ a > 1/3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3784_378492


namespace NUMINAMATH_CALUDE_max_value_of_p_l3784_378414

theorem max_value_of_p (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h : a * b * c + a + c = b) :
  ∃ (p : ℝ), p = (2 / (a^2 + 1)) - (2 / (b^2 + 1)) + (3 / (c^2 + 1)) ∧ 
  p ≤ 10/3 ∧ 
  (∃ (a' b' c' : ℝ) (ha' : 0 < a') (hb' : 0 < b') (hc' : 0 < c') 
    (h' : a' * b' * c' + a' + c' = b'), 
    (2 / (a'^2 + 1)) - (2 / (b'^2 + 1)) + (3 / (c'^2 + 1)) = 10/3) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_p_l3784_378414


namespace NUMINAMATH_CALUDE_tourists_distribution_theorem_l3784_378484

/-- The number of ways to distribute tourists among guides -/
def distribute_tourists (num_tourists : ℕ) (num_guides : ℕ) : ℕ :=
  num_guides ^ num_tourists

/-- The number of ways to distribute tourists among guides, excluding cases where some guides have no tourists -/
def distribute_tourists_with_restriction (num_tourists : ℕ) (num_guides : ℕ) : ℕ :=
  distribute_tourists num_tourists num_guides - 
  (num_guides.choose 1 * (num_guides - 1) ^ num_tourists) +
  (num_guides.choose 2 * (num_guides - 2) ^ num_tourists)

/-- The theorem stating that distributing 8 tourists among 3 guides, with each guide having at least one tourist, results in 5796 possible groupings -/
theorem tourists_distribution_theorem : 
  distribute_tourists_with_restriction 8 3 = 5796 := by
  sorry

end NUMINAMATH_CALUDE_tourists_distribution_theorem_l3784_378484


namespace NUMINAMATH_CALUDE_max_value_of_z_l3784_378453

theorem max_value_of_z (x y k : ℝ) (h1 : x + 2*y - 1 ≥ 0) (h2 : x - y ≥ 0) 
  (h3 : 0 ≤ x) (h4 : x ≤ k) (h5 : ∃ (x_min y_min : ℝ), x_min + k*y_min = -2 ∧ 
  x_min + 2*y_min - 1 ≥ 0 ∧ x_min - y_min ≥ 0 ∧ 0 ≤ x_min ∧ x_min ≤ k ∧
  ∀ (x' y' : ℝ), x' + 2*y' - 1 ≥ 0 → x' - y' ≥ 0 → 0 ≤ x' → x' ≤ k → x' + k*y' ≥ -2) :
  ∃ (x_max y_max : ℝ), x_max + k*y_max = 20 ∧ 
  x_max + 2*y_max - 1 ≥ 0 ∧ x_max - y_max ≥ 0 ∧ 0 ≤ x_max ∧ x_max ≤ k ∧
  ∀ (x' y' : ℝ), x' + 2*y' - 1 ≥ 0 → x' - y' ≥ 0 → 0 ≤ x' → x' ≤ k → x' + k*y' ≤ 20 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_z_l3784_378453


namespace NUMINAMATH_CALUDE_not_divisible_by_five_l3784_378415

theorem not_divisible_by_five (n : ℤ) : ¬ (5 ∣ (n^3 + 2*n - 1)) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_five_l3784_378415


namespace NUMINAMATH_CALUDE_coin_problem_l3784_378416

theorem coin_problem (total : ℕ) (difference : ℕ) (heads : ℕ) : 
  total = 128 → difference = 12 → heads = (total + difference) / 2 → heads = 70 := by
  sorry

end NUMINAMATH_CALUDE_coin_problem_l3784_378416


namespace NUMINAMATH_CALUDE_polynomial_B_value_l3784_378430

def polynomial (z A B C D : ℤ) : ℤ := z^6 - 9*z^5 + A*z^4 + B*z^3 + C*z^2 + D*z + 81

theorem polynomial_B_value (A B C D : ℤ) :
  (∀ r : ℤ, polynomial r A B C D = 0 → r > 0) →
  B = -46 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_B_value_l3784_378430


namespace NUMINAMATH_CALUDE_tv_cost_l3784_378410

theorem tv_cost (savings : ℝ) (furniture_fraction : ℝ) (tv_cost : ℝ) : 
  savings = 840 → 
  furniture_fraction = 3/4 → 
  tv_cost = savings * (1 - furniture_fraction) → 
  tv_cost = 210 :=
by
  sorry

end NUMINAMATH_CALUDE_tv_cost_l3784_378410


namespace NUMINAMATH_CALUDE_fraction_equality_l3784_378498

theorem fraction_equality (u v : ℝ) (h : (1/u + 1/v) / (1/u - 1/v) = 2024) :
  (u + v) / (u - v) = 2024 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3784_378498


namespace NUMINAMATH_CALUDE_five_student_committees_with_two_fixed_l3784_378413

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of different five-student committees that can be chosen from a group of 8 students,
    where two specific students must always be included -/
theorem five_student_committees_with_two_fixed (total_students : ℕ) (committee_size : ℕ) (fixed_students : ℕ) :
  total_students = 8 →
  committee_size = 5 →
  fixed_students = 2 →
  choose (total_students - fixed_students) (committee_size - fixed_students) = 20 := by
  sorry


end NUMINAMATH_CALUDE_five_student_committees_with_two_fixed_l3784_378413


namespace NUMINAMATH_CALUDE_prob_all_players_five_coins_l3784_378417

/-- Represents a player in the coin game -/
inductive Player : Type
| Abby : Player
| Bernardo : Player
| Carl : Player
| Debra : Player

/-- Represents a ball color in the game -/
inductive BallColor : Type
| Green : BallColor
| Red : BallColor
| Blue : BallColor
| White : BallColor

/-- Represents the state of the game after each round -/
structure GameState :=
(coins : Player → ℕ)
(round : ℕ)

/-- Represents a single round of the game -/
def play_round (state : GameState) : GameState :=
sorry

/-- The probability of drawing both green and blue balls by the same player in a single round -/
def prob_green_blue_same_player : ℚ :=
1 / 20

/-- The number of rounds in the game -/
def num_rounds : ℕ := 5

/-- The initial number of coins for each player -/
def initial_coins : ℕ := 5

/-- The probability that each player has exactly 5 coins after 5 rounds -/
theorem prob_all_players_five_coins :
  (prob_green_blue_same_player ^ num_rounds : ℚ) = 1 / 3200000 :=
sorry

end NUMINAMATH_CALUDE_prob_all_players_five_coins_l3784_378417


namespace NUMINAMATH_CALUDE_symmetric_line_equation_l3784_378436

/-- Given a line l symmetric to the line 2x - 3y + 4 = 0 with respect to x = 1,
    prove that the equation of l is 2x + 3y - 8 = 0 -/
theorem symmetric_line_equation :
  ∀ (l : Set (ℝ × ℝ)),
  (∀ (x y : ℝ), (x, y) ∈ l ↔ (2 - x, y) ∈ {(x, y) | 2*x - 3*y + 4 = 0}) →
  l = {(x, y) | 2*x + 3*y - 8 = 0} :=
by sorry

end NUMINAMATH_CALUDE_symmetric_line_equation_l3784_378436


namespace NUMINAMATH_CALUDE_bottles_per_case_l3784_378404

theorem bottles_per_case (april_cases : ℕ) (may_cases : ℕ) (total_bottles : ℕ) : 
  april_cases = 20 → may_cases = 30 → total_bottles = 1000 →
  ∃ (bottles_per_case : ℕ), bottles_per_case * (april_cases + may_cases) = total_bottles ∧ bottles_per_case = 20 :=
by sorry

end NUMINAMATH_CALUDE_bottles_per_case_l3784_378404


namespace NUMINAMATH_CALUDE_percentage_difference_l3784_378427

theorem percentage_difference (p t j : ℝ) 
  (h1 : j = 0.75 * p) 
  (h2 : j = 0.8 * t) : 
  t = (15/16) * p := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l3784_378427


namespace NUMINAMATH_CALUDE_equation_implies_fraction_value_l3784_378422

theorem equation_implies_fraction_value
  (a x y : ℝ)
  (h : x * Real.sqrt (a * (x - a)) + y * Real.sqrt (a * (y - a)) = Real.sqrt (abs (Real.log (x - a) - Real.log (a - y)))) :
  (3 * x^2 + x * y - y^2) / (x^2 - x * y + y^2) = 1/3 := by
sorry

end NUMINAMATH_CALUDE_equation_implies_fraction_value_l3784_378422


namespace NUMINAMATH_CALUDE_roots_of_quartic_equation_l3784_378497

theorem roots_of_quartic_equation :
  let f : ℝ → ℝ := λ x => 7 * x^4 - 44 * x^3 + 78 * x^2 - 44 * x + 7
  ∃ (a b c d : ℝ), 
    (∀ x : ℝ, f x = 0 ↔ x = a ∨ x = b ∨ x = c ∨ x = d) ∧
    a = 2 ∧ 
    b = 1/2 ∧ 
    c = (8 + Real.sqrt 15) / 7 ∧ 
    d = (8 - Real.sqrt 15) / 7 :=
by sorry

end NUMINAMATH_CALUDE_roots_of_quartic_equation_l3784_378497


namespace NUMINAMATH_CALUDE_evaluate_expression_l3784_378466

theorem evaluate_expression : (-3)^4 + (-3)^3 + (-3)^2 + 3^2 + 3^3 + 3^4 = 180 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3784_378466


namespace NUMINAMATH_CALUDE_mean_proportion_of_3_and_4_l3784_378448

theorem mean_proportion_of_3_and_4 :
  ∃ x : ℝ, (3 : ℝ) / x = x / 4 ∧ (x = 2 * Real.sqrt 3 ∨ x = -2 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_mean_proportion_of_3_and_4_l3784_378448


namespace NUMINAMATH_CALUDE_closest_product_l3784_378433

def options : List ℝ := [1600, 1800, 2000, 2200, 2400]

theorem closest_product : 
  let product := 0.000625 * 3142857
  ∀ x ∈ options, x ≠ 1800 → |product - 1800| < |product - x| :=
by sorry

end NUMINAMATH_CALUDE_closest_product_l3784_378433


namespace NUMINAMATH_CALUDE_solve_quadratic_equation_l3784_378440

theorem solve_quadratic_equation : 
  ∃ x : ℚ, (10 - 2*x)^2 = 4*x^2 + 20*x ∧ x = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_solve_quadratic_equation_l3784_378440


namespace NUMINAMATH_CALUDE_polynomial_sum_l3784_378451

theorem polynomial_sum (a b c d : ℤ) : 
  (∀ x, (x^2 + a*x + b) * (x^2 + c*x + d) = x^4 + 2*x^3 - 5*x^2 + 8*x - 12) → 
  a + b + c + d = 6 := by
sorry

end NUMINAMATH_CALUDE_polynomial_sum_l3784_378451


namespace NUMINAMATH_CALUDE_windows_preference_l3784_378485

theorem windows_preference (total : ℕ) (mac : ℕ) (no_pref : ℕ) : 
  total = 210 → 
  mac = 60 → 
  no_pref = 90 → 
  ∃ (windows : ℕ), windows = total - (mac + no_pref + mac / 3) ∧ windows = 40 := by
  sorry

#check windows_preference

end NUMINAMATH_CALUDE_windows_preference_l3784_378485


namespace NUMINAMATH_CALUDE_laylas_score_l3784_378482

theorem laylas_score (total : ℕ) (difference : ℕ) (laylas_score : ℕ) : 
  total = 112 → difference = 28 → laylas_score = 70 →
  ∃ (nahimas_score : ℕ), 
    nahimas_score + laylas_score = total ∧ 
    laylas_score = nahimas_score + difference :=
by
  sorry

end NUMINAMATH_CALUDE_laylas_score_l3784_378482


namespace NUMINAMATH_CALUDE_arithmetic_sequence_difference_l3784_378478

/-- The arithmetic sequence with first term a and common difference d -/
def arithmeticSequence (a d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

/-- The absolute difference between two integers -/
def absDiff (a b : ℤ) : ℕ := (a - b).natAbs

theorem arithmetic_sequence_difference :
  let a := -10  -- First term of the sequence
  let d := 11   -- Common difference of the sequence
  absDiff (arithmeticSequence a d 2025) (arithmeticSequence a d 2010) = 165 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_difference_l3784_378478


namespace NUMINAMATH_CALUDE_simple_interest_problem_l3784_378425

theorem simple_interest_problem (P : ℝ) : 
  P * 0.08 * 3 = 0.5 * 4000 * ((1 + 0.10)^2 - 1) ↔ P = 1750 :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l3784_378425


namespace NUMINAMATH_CALUDE_johns_remaining_money_l3784_378471

/-- The amount of money John has left after buying pizzas and sodas -/
theorem johns_remaining_money (q : ℝ) : 
  (3 : ℝ) * (2 : ℝ) * q + -- cost of small pizzas
  (2 : ℝ) * (3 : ℝ) * q + -- cost of medium pizzas
  (4 : ℝ) * q             -- cost of sodas
  ≤ (50 : ℝ) →
  (50 : ℝ) - ((3 : ℝ) * (2 : ℝ) * q + (2 : ℝ) * (3 : ℝ) * q + (4 : ℝ) * q) = 50 - 16 * q :=
by sorry

end NUMINAMATH_CALUDE_johns_remaining_money_l3784_378471


namespace NUMINAMATH_CALUDE_stating_n_gon_triangulation_l3784_378423

/-- 
A polygon with n sides (n-gon) can be divided into triangles by non-intersecting diagonals. 
This function represents the number of such triangles.
-/
def num_triangles (n : ℕ) : ℕ := n - 2

/-- 
Theorem stating that the number of triangles into which non-intersecting diagonals 
divide an n-gon is equal to n-2, for any n ≥ 3.
-/
theorem n_gon_triangulation (n : ℕ) (h : n ≥ 3) : 
  num_triangles n = n - 2 := by
  sorry

end NUMINAMATH_CALUDE_stating_n_gon_triangulation_l3784_378423


namespace NUMINAMATH_CALUDE_farmers_market_sales_l3784_378418

theorem farmers_market_sales (total_earnings broccoli_sales cauliflower_sales : ℕ) 
  (h1 : total_earnings = 380)
  (h2 : broccoli_sales = 57)
  (h3 : cauliflower_sales = 136) :
  ∃ (spinach_sales : ℕ), 
    spinach_sales = 73 ∧ 
    spinach_sales > (2 * broccoli_sales) / 2 ∧
    total_earnings = broccoli_sales + (2 * broccoli_sales) + cauliflower_sales + spinach_sales :=
by
  sorry


end NUMINAMATH_CALUDE_farmers_market_sales_l3784_378418


namespace NUMINAMATH_CALUDE_factorial_ratio_l3784_378488

theorem factorial_ratio : Nat.factorial 50 / Nat.factorial 48 = 2450 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l3784_378488


namespace NUMINAMATH_CALUDE_cleanup_drive_total_l3784_378420

/-- The total amount of garbage collected by two groups, given the amount collected by one group
    and the difference between the two groups' collections. -/
def totalGarbageCollected (group1Amount : ℕ) (difference : ℕ) : ℕ :=
  group1Amount + (group1Amount - difference)

/-- Theorem stating that given Lizzie's group collected 387 pounds of garbage and another group
    collected 39 pounds less, the total amount of garbage collected by both groups is 735 pounds. -/
theorem cleanup_drive_total :
  totalGarbageCollected 387 39 = 735 := by
  sorry

end NUMINAMATH_CALUDE_cleanup_drive_total_l3784_378420


namespace NUMINAMATH_CALUDE_absolute_sum_zero_implies_sum_l3784_378467

theorem absolute_sum_zero_implies_sum (a b : ℝ) : 
  |a - 5| + |b + 8| = 0 → a + b = -3 := by
  sorry

end NUMINAMATH_CALUDE_absolute_sum_zero_implies_sum_l3784_378467


namespace NUMINAMATH_CALUDE_movie_marathon_end_time_correct_l3784_378494

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Represents a duration in hours and minutes -/
structure Duration where
  hours : Nat
  minutes : Nat
  deriving Repr

def addDurationToTime (t : Time) (d : Duration) : Time :=
  let totalMinutes := t.minutes + d.minutes + d.hours * 60
  { hours := t.hours + totalMinutes / 60,
    minutes := totalMinutes % 60 }

def movie_marathon_end_time (start : Time) 
  (movie1 : Duration) (break1 : Duration) 
  (movie2 : Duration) (break2 : Duration) 
  (movie3 : Duration) : Time :=
  let t1 := addDurationToTime start movie1
  let t2 := addDurationToTime t1 break1
  let t3 := addDurationToTime t2 movie2
  let t4 := addDurationToTime t3 break2
  addDurationToTime t4 movie3

theorem movie_marathon_end_time_correct :
  let start := Time.mk 13 0  -- 1:00 p.m.
  let movie1 := Duration.mk 2 20
  let break1 := Duration.mk 0 20
  let movie2 := Duration.mk 1 45
  let break2 := Duration.mk 0 20
  let movie3 := Duration.mk 2 10
  movie_marathon_end_time start movie1 break1 movie2 break2 movie3 = Time.mk 19 55  -- 7:55 p.m.
  := by sorry

end NUMINAMATH_CALUDE_movie_marathon_end_time_correct_l3784_378494


namespace NUMINAMATH_CALUDE_twins_ratios_l3784_378407

/-- Represents the family composition before and after the birth of twins -/
structure Family where
  initial_boys : ℕ
  initial_girls : ℕ
  k : ℚ
  t : ℚ

/-- The ratio of brothers to sisters for boys after the birth of twins -/
def boys_ratio (f : Family) : ℚ :=
  f.initial_boys / (f.initial_girls + 1)

/-- The ratio of brothers to sisters for girls after the birth of twins -/
def girls_ratio (f : Family) : ℚ :=
  (f.initial_boys + 1) / f.initial_girls

/-- Theorem stating the ratios after the birth of twins -/
theorem twins_ratios (f : Family) 
  (h1 : (f.initial_boys + 2) / f.initial_girls = f.k)
  (h2 : f.initial_boys / (f.initial_girls + 2) = f.t) :
  boys_ratio f = f.t ∧ girls_ratio f = f.k := by
  sorry

#check twins_ratios

end NUMINAMATH_CALUDE_twins_ratios_l3784_378407


namespace NUMINAMATH_CALUDE_total_seashells_l3784_378480

-- Define the number of seashells found by Sam
def sam_shells : ℕ := 18

-- Define the number of seashells found by Mary
def mary_shells : ℕ := 47

-- Theorem stating the total number of seashells found
theorem total_seashells : sam_shells + mary_shells = 65 := by
  sorry

end NUMINAMATH_CALUDE_total_seashells_l3784_378480


namespace NUMINAMATH_CALUDE_intersection_and_union_of_sets_l3784_378464

theorem intersection_and_union_of_sets (a : ℝ) :
  let A : Set ℝ := {-3, a + 1}
  let B : Set ℝ := {2 * a - 1, a^2 + 1}
  (A ∩ B = {3}) →
  (a = 2 ∧ A ∪ B = {-3, 3, 5}) := by
sorry

end NUMINAMATH_CALUDE_intersection_and_union_of_sets_l3784_378464


namespace NUMINAMATH_CALUDE_star_sqrt_11_l3784_378465

/-- Custom binary operation ¤ -/
def star (x y z : ℝ) : ℝ := (x + y)^2 - z^2

theorem star_sqrt_11 (z : ℝ) :
  star (Real.sqrt 11) (Real.sqrt 11) z = 44 → z = 0 := by
  sorry

end NUMINAMATH_CALUDE_star_sqrt_11_l3784_378465


namespace NUMINAMATH_CALUDE_parking_space_savings_l3784_378424

/-- Proves the yearly savings when renting a parking space monthly instead of weekly -/
theorem parking_space_savings (weekly_rate : ℕ) (monthly_rate : ℕ) (weeks_per_year : ℕ) (months_per_year : ℕ) :
  weekly_rate = 10 →
  monthly_rate = 42 →
  weeks_per_year = 52 →
  months_per_year = 12 →
  weekly_rate * weeks_per_year - monthly_rate * months_per_year = 16 := by
sorry

end NUMINAMATH_CALUDE_parking_space_savings_l3784_378424


namespace NUMINAMATH_CALUDE_employed_males_percentage_l3784_378426

theorem employed_males_percentage
  (total_population : ℝ)
  (employed_percentage : ℝ)
  (employed_females_percentage : ℝ)
  (h1 : employed_percentage = 70)
  (h2 : employed_females_percentage = 70)
  (h3 : total_population > 0) :
  (employed_percentage / 100 * (1 - employed_females_percentage / 100) * 100) = 21 := by
  sorry

end NUMINAMATH_CALUDE_employed_males_percentage_l3784_378426


namespace NUMINAMATH_CALUDE_smallest_cube_root_with_small_remainder_l3784_378475

theorem smallest_cube_root_with_small_remainder : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ) (r : ℝ), 0 < r ∧ r < 1/1000 ∧ (m : ℝ)^(1/3) = n + r →
    ∀ (k : ℕ) (s : ℝ), 0 < s ∧ s < 1/1000 ∧ (k : ℝ)^(1/3) = (n-1) + s → k > m) ∧
  n = 19 :=
sorry

end NUMINAMATH_CALUDE_smallest_cube_root_with_small_remainder_l3784_378475


namespace NUMINAMATH_CALUDE_trapezoid_perimeter_l3784_378481

/-- Represents a trapezoid ABCD with given properties -/
structure Trapezoid where
  BC : ℝ
  AP : ℝ
  DQ : ℝ
  AB : ℝ
  CD : ℝ
  bc_length : BC = 32
  ap_length : AP = 24
  dq_length : DQ = 18
  ab_length : AB = 29
  cd_length : CD = 35

/-- Calculates the perimeter of the trapezoid -/
def perimeter (t : Trapezoid) : ℝ :=
  t.AB + t.BC + t.CD + (t.AP + t.BC + t.DQ)

/-- Theorem: The perimeter of the trapezoid is 170 units -/
theorem trapezoid_perimeter (t : Trapezoid) : perimeter t = 170 := by
  sorry

#check trapezoid_perimeter

end NUMINAMATH_CALUDE_trapezoid_perimeter_l3784_378481


namespace NUMINAMATH_CALUDE_hiking_rate_up_l3784_378409

/-- Hiking problem statement -/
theorem hiking_rate_up (total_time : ℝ) (time_up : ℝ) (rate_down : ℝ) :
  total_time = 3 →
  time_up = 1.2 →
  rate_down = 6 →
  ∃ (rate_up : ℝ), rate_up = 9 ∧ rate_up * time_up = rate_down * (total_time - time_up) :=
by
  sorry

end NUMINAMATH_CALUDE_hiking_rate_up_l3784_378409


namespace NUMINAMATH_CALUDE_cube_root_of_square_l3784_378474

theorem cube_root_of_square (x : ℝ) : (x^2)^(1/3) = x^(2/3) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_square_l3784_378474


namespace NUMINAMATH_CALUDE_thursday_five_times_in_july_l3784_378447

/-- Represents the days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a specific date in a month -/
structure Date :=
  (day : Nat)
  (dayOfWeek : DayOfWeek)

/-- A month with its number of days and list of dates -/
structure Month :=
  (numDays : Nat)
  (dates : List Date)

def june : Month := sorry
def july : Month := sorry

/-- Counts the number of occurrences of a specific day in a month -/
def countDayInMonth (m : Month) (d : DayOfWeek) : Nat := sorry

theorem thursday_five_times_in_july 
  (h1 : june.numDays = 30)
  (h2 : july.numDays = 31)
  (h3 : countDayInMonth june DayOfWeek.Tuesday = 5) :
  countDayInMonth july DayOfWeek.Thursday = 5 := by
  sorry

end NUMINAMATH_CALUDE_thursday_five_times_in_july_l3784_378447


namespace NUMINAMATH_CALUDE_ratio_of_divisor_sums_l3784_378446

def M : ℕ := 36 * 36 * 65 * 275

def sum_odd_divisors (n : ℕ) : ℕ := sorry
def sum_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_of_divisor_sums :
  (sum_odd_divisors M : ℚ) / (sum_even_divisors M : ℚ) = 1 / 30 := by sorry

end NUMINAMATH_CALUDE_ratio_of_divisor_sums_l3784_378446


namespace NUMINAMATH_CALUDE_tangent_circles_bc_length_l3784_378468

/-- Two externally tangent circles with centers A and B -/
structure TangentCircles where
  A : ℝ × ℝ  -- Center of first circle
  B : ℝ × ℝ  -- Center of second circle
  radius_A : ℝ  -- Radius of first circle
  radius_B : ℝ  -- Radius of second circle
  externally_tangent : ‖A - B‖ = radius_A + radius_B

/-- A line tangent to both circles intersecting ray AB at point C -/
def tangent_line (tc : TangentCircles) (C : ℝ × ℝ) : Prop :=
  ∃ D E : ℝ × ℝ,
    ‖D - tc.A‖ = tc.radius_A ∧
    ‖E - tc.B‖ = tc.radius_B ∧
    (D - C) • (tc.A - C) = 0 ∧
    (E - C) • (tc.B - C) = 0 ∧
    (C - tc.A) • (tc.B - tc.A) ≥ 0

/-- The main theorem -/
theorem tangent_circles_bc_length 
  (tc : TangentCircles) 
  (hA : tc.radius_A = 7)
  (hB : tc.radius_B = 4)
  (C : ℝ × ℝ)
  (h_tangent : tangent_line tc C) :
  ‖C - tc.B‖ = 44 / 3 :=
sorry

end NUMINAMATH_CALUDE_tangent_circles_bc_length_l3784_378468


namespace NUMINAMATH_CALUDE_ninety_eight_squared_l3784_378470

theorem ninety_eight_squared : (100 - 2)^2 = 9604 := by
  sorry

end NUMINAMATH_CALUDE_ninety_eight_squared_l3784_378470


namespace NUMINAMATH_CALUDE_max_leftover_stickers_l3784_378491

theorem max_leftover_stickers (y : ℕ+) : 
  ∃ (q r : ℕ), y = 12 * q + r ∧ r ≤ 11 ∧ 
  ∀ (q' r' : ℕ), y = 12 * q' + r' → r' ≤ r :=
sorry

end NUMINAMATH_CALUDE_max_leftover_stickers_l3784_378491


namespace NUMINAMATH_CALUDE_cafe_bill_difference_l3784_378421

theorem cafe_bill_difference (amy_tip beth_tip : ℝ) 
  (amy_percentage beth_percentage : ℝ) : 
  amy_tip = 4 →
  beth_tip = 5 →
  amy_percentage = 0.08 →
  beth_percentage = 0.10 →
  amy_tip = amy_percentage * (amy_tip / amy_percentage) →
  beth_tip = beth_percentage * (beth_tip / beth_percentage) →
  (amy_tip / amy_percentage) - (beth_tip / beth_percentage) = 0 := by
sorry

end NUMINAMATH_CALUDE_cafe_bill_difference_l3784_378421


namespace NUMINAMATH_CALUDE_final_price_is_12_l3784_378406

/-- The price of a set consisting of one cup of coffee and one piece of cheesecake,
    with a 25% discount when bought together. -/
def discounted_set_price (coffee_price cheesecake_price : ℚ) (discount_rate : ℚ) : ℚ :=
  (coffee_price + cheesecake_price) * (1 - discount_rate)

/-- Theorem stating that the final price of the set is $12 -/
theorem final_price_is_12 :
  discounted_set_price 6 10 (1/4) = 12 := by
  sorry

end NUMINAMATH_CALUDE_final_price_is_12_l3784_378406


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l3784_378400

theorem necessary_not_sufficient_condition (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (∀ x y, x > 0 → y > 0 → x + y < 4 → x * y < 4) ∧
  (∃ x y, x > 0 ∧ y > 0 ∧ x * y < 4 ∧ x + y ≥ 4) := by
  sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l3784_378400


namespace NUMINAMATH_CALUDE_production_rates_correct_l3784_378499

/-- Represents the production data for a company in November and March --/
structure ProductionData where
  nov_production : ℕ
  mar_production : ℕ
  time_difference : ℕ
  efficiency_ratio : Rat

/-- Calculates the production rates given the production data --/
def calculate_production_rates (data : ProductionData) : ℚ × ℚ :=
  let nov_rate := 2 * data.efficiency_ratio
  let mar_rate := 3 * data.efficiency_ratio
  (nov_rate, mar_rate)

theorem production_rates_correct (data : ProductionData) 
  (h1 : data.nov_production = 1400)
  (h2 : data.mar_production = 2400)
  (h3 : data.time_difference = 50)
  (h4 : data.efficiency_ratio = 2/3) :
  calculate_production_rates data = (4, 6) := by
  sorry

#eval calculate_production_rates {
  nov_production := 1400,
  mar_production := 2400,
  time_difference := 50,
  efficiency_ratio := 2/3
}

end NUMINAMATH_CALUDE_production_rates_correct_l3784_378499


namespace NUMINAMATH_CALUDE_smallest_prime_factor_in_C_l3784_378490

def C : Set Nat := {33, 35, 36, 39, 41}

theorem smallest_prime_factor_in_C :
  ∃ (n : Nat), n ∈ C ∧ (∀ (m : Nat), m ∈ C → ∀ (p q : Nat), Prime p → Prime q → p ∣ n → q ∣ m → p ≤ q) ∧ n = 36 := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_in_C_l3784_378490


namespace NUMINAMATH_CALUDE_last_s_replacement_l3784_378419

-- Define the alphabet size
def alphabet_size : ℕ := 26

-- Define the function to calculate the shift for the nth occurrence
def shift (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define the function to apply the shift modulo alphabet size
def apply_shift (shift : ℕ) : ℕ := shift % alphabet_size

-- Theorem statement
theorem last_s_replacement (occurrences : ℕ) (h : occurrences = 12) :
  apply_shift (shift occurrences) = 0 := by sorry

end NUMINAMATH_CALUDE_last_s_replacement_l3784_378419


namespace NUMINAMATH_CALUDE_stephanie_oranges_l3784_378452

/-- The number of times Stephanie went to the store last month -/
def store_visits : ℕ := 8

/-- The number of oranges Stephanie buys each time she goes to the store -/
def oranges_per_visit : ℕ := 2

/-- The total number of oranges Stephanie bought last month -/
def total_oranges : ℕ := store_visits * oranges_per_visit

theorem stephanie_oranges : total_oranges = 16 := by
  sorry

end NUMINAMATH_CALUDE_stephanie_oranges_l3784_378452


namespace NUMINAMATH_CALUDE_large_triangle_toothpicks_l3784_378487

/-- The number of small triangles in the base row of the large triangle -/
def base_triangles : ℕ := 1001

/-- The total number of small triangles in the large triangle -/
def total_triangles : ℕ := base_triangles * (base_triangles + 1) / 2

/-- The number of toothpicks needed to construct the large triangle -/
def toothpicks : ℕ := (3 * total_triangles) / 2

theorem large_triangle_toothpicks :
  toothpicks = 752252 :=
sorry

end NUMINAMATH_CALUDE_large_triangle_toothpicks_l3784_378487


namespace NUMINAMATH_CALUDE_monday_attendance_l3784_378479

theorem monday_attendance (tuesday : ℕ) (wed_to_fri : ℕ) (average : ℕ) (days : ℕ)
  (h1 : tuesday = 15)
  (h2 : wed_to_fri = 10)
  (h3 : average = 11)
  (h4 : days = 5) :
  ∃ (monday : ℕ), monday + tuesday + 3 * wed_to_fri = average * days ∧ monday = 10 := by
  sorry

end NUMINAMATH_CALUDE_monday_attendance_l3784_378479


namespace NUMINAMATH_CALUDE_range_of_a_l3784_378456

-- Define the conditions
def sufficient_condition (x : ℝ) : Prop := -2 < x ∧ x < 4

def necessary_condition (x a : ℝ) : Prop := (x + 2) * (x - a) < 0

-- Define the theorem
theorem range_of_a : 
  (∀ x a : ℝ, sufficient_condition x → necessary_condition x a) ∧ 
  (∃ x a : ℝ, ¬sufficient_condition x ∧ necessary_condition x a) → 
  ∀ a : ℝ, (a ∈ Set.Ioi 4) ↔ (∃ x : ℝ, necessary_condition x a) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3784_378456


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_l3784_378432

theorem right_triangle_perimeter (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a * b / 2 = 150 → 
  a = 30 →
  a^2 + b^2 = c^2 →
  a + b + c = 40 + 10 * Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_l3784_378432


namespace NUMINAMATH_CALUDE_apple_distribution_l3784_378408

theorem apple_distribution (total_apples : ℕ) (min_apples : ℕ) (num_people : ℕ) : 
  total_apples = 30 → min_apples = 3 → num_people = 3 →
  (Nat.choose (total_apples - num_people * min_apples + num_people - 1) (num_people - 1)) = 253 := by
sorry

end NUMINAMATH_CALUDE_apple_distribution_l3784_378408


namespace NUMINAMATH_CALUDE_perpendicular_construction_l3784_378486

-- Define the basic geometric elements
structure Point := (x y : ℝ)
structure Line := (a b c : ℝ)

-- Define the concept of a point being on a line
def Point.on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Define parallel lines
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a ∧ l1.a * l2.c ≠ l1.c * l2.a

-- Define perpendicular lines
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

-- Define the construction process
def construct_perpendicular (l : Line) (A : Point) (m l1 l2 lm : Line) (M1 M2 B : Point) : Prop :=
  A.on_line l ∧
  A.on_line m ∧
  parallel l1 m ∧
  parallel l2 m ∧
  M1.on_line l ∧
  M1.on_line l1 ∧
  M2.on_line l ∧
  M2.on_line l2 ∧
  parallel lm (Line.mk (M1.x - A.x) (M1.y - A.y) 0) ∧
  B.on_line l2 ∧
  B.on_line lm

-- State the theorem
theorem perpendicular_construction (l : Line) (A : Point) :
  ∃ (m l1 l2 lm : Line) (M1 M2 B : Point),
    construct_perpendicular l A m l1 l2 lm M1 M2 B →
    perpendicular l (Line.mk (B.x - A.x) (B.y - A.y) 0) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_construction_l3784_378486


namespace NUMINAMATH_CALUDE_four_equal_area_volume_prisms_l3784_378459

/-- A square prism with integer edge lengths where the surface area equals the volume. -/
structure EqualAreaVolumePrism where
  a : ℕ  -- length of the base
  b : ℕ  -- height of the prism
  h : 2 * a^2 + 4 * a * b = a^2 * b

/-- The set of all square prisms with integer edge lengths where the surface area equals the volume. -/
def allEqualAreaVolumePrisms : Set EqualAreaVolumePrism :=
  {p : EqualAreaVolumePrism | True}

/-- The theorem stating that there are only four square prisms with integer edge lengths
    where the surface area equals the volume. -/
theorem four_equal_area_volume_prisms :
  allEqualAreaVolumePrisms = {
    ⟨12, 3, by sorry⟩,
    ⟨8, 4, by sorry⟩,
    ⟨6, 6, by sorry⟩,
    ⟨5, 10, by sorry⟩
  } := by sorry

end NUMINAMATH_CALUDE_four_equal_area_volume_prisms_l3784_378459


namespace NUMINAMATH_CALUDE_calculation_result_l3784_378454

theorem calculation_result : (101 * 2012 * 121) / 1111 / 503 = 44 := by
  sorry

end NUMINAMATH_CALUDE_calculation_result_l3784_378454


namespace NUMINAMATH_CALUDE_strawberry_jam_earnings_l3784_378489

/-- Represents the number of strawberries picked by each person and the jam-making process. -/
structure StrawberryPicking where
  betty : ℕ
  matthew : ℕ
  natalie : ℕ
  strawberries_per_jar : ℕ
  price_per_jar : ℕ

/-- Calculates the total money earned from selling jam made from picked strawberries. -/
def total_money_earned (sp : StrawberryPicking) : ℕ :=
  let total_strawberries := sp.betty + sp.matthew + sp.natalie
  let jars_of_jam := total_strawberries / sp.strawberries_per_jar
  jars_of_jam * sp.price_per_jar

/-- Theorem stating that under the given conditions, the total money earned is $40. -/
theorem strawberry_jam_earnings : ∀ (sp : StrawberryPicking),
  sp.betty = 16 ∧
  sp.matthew = sp.betty + 20 ∧
  sp.matthew = 2 * sp.natalie ∧
  sp.strawberries_per_jar = 7 ∧
  sp.price_per_jar = 4 →
  total_money_earned sp = 40 :=
by
  sorry


end NUMINAMATH_CALUDE_strawberry_jam_earnings_l3784_378489


namespace NUMINAMATH_CALUDE_polygon_interior_angles_l3784_378461

theorem polygon_interior_angles (n : ℕ) : (n - 2) * 180 = 360 → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_polygon_interior_angles_l3784_378461


namespace NUMINAMATH_CALUDE_rope_cut_theorem_l3784_378403

/-- Given a rope of 60 meters cut into two pieces, where the longer piece is twice
    the length of the shorter piece, prove that the length of the shorter piece is 20 meters. -/
theorem rope_cut_theorem (total_length : ℝ) (short_piece : ℝ) (long_piece : ℝ) : 
  total_length = 60 →
  long_piece = 2 * short_piece →
  total_length = short_piece + long_piece →
  short_piece = 20 := by
  sorry

end NUMINAMATH_CALUDE_rope_cut_theorem_l3784_378403


namespace NUMINAMATH_CALUDE_no_solution_with_vasyas_correction_l3784_378469

theorem no_solution_with_vasyas_correction (r : ℝ) : ¬ ∃ (a h : ℝ),
  (0 < r) ∧                           -- radius is positive
  (0 < a) ∧ (0 < h) ∧                 -- base and height are positive
  (a ≤ 2*r) ∧                         -- base is at most diameter
  (h < 2*r) ∧                         -- height is less than diameter
  (a + h = 2*Real.pi*r) :=            -- sum equals circumference (Vasya's condition)
by
  sorry

end NUMINAMATH_CALUDE_no_solution_with_vasyas_correction_l3784_378469


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_is_one_seventh_l3784_378401

noncomputable def ellipse_eccentricity (a : ℝ) : ℝ :=
  let b := Real.sqrt 3
  let c := (1 : ℝ) / 4
  c / a

theorem ellipse_eccentricity_is_one_seventh :
  ∃ a : ℝ, (a > 0) ∧ 
  ((1 : ℝ) / 4)^2 / a^2 + (0 : ℝ)^2 / 3 = 1 ∧
  ellipse_eccentricity a = (1 : ℝ) / 7 := by
sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_is_one_seventh_l3784_378401


namespace NUMINAMATH_CALUDE_positive_integer_pairs_eq_enumerated_set_l3784_378412

def positive_integer_pairs : Set (ℕ × ℕ) :=
  {p | p.1 > 0 ∧ p.2 > 0 ∧ p.1 + p.2 = 4}

def enumerated_set : Set (ℕ × ℕ) :=
  {(1, 3), (2, 2), (3, 1)}

theorem positive_integer_pairs_eq_enumerated_set :
  positive_integer_pairs = enumerated_set := by
  sorry

end NUMINAMATH_CALUDE_positive_integer_pairs_eq_enumerated_set_l3784_378412


namespace NUMINAMATH_CALUDE_spinner_direction_l3784_378442

-- Define the directions
inductive Direction
| North
| East
| South
| West

-- Define the rotation
def rotate (initial : Direction) (revolutions : ℚ) : Direction :=
  sorry

-- Theorem statement
theorem spinner_direction (initial : Direction) 
  (clockwise : ℚ) (counterclockwise : ℚ) :
  initial = Direction.North ∧ 
  clockwise = 7/2 ∧ 
  counterclockwise = 17/4 →
  rotate (rotate initial clockwise) (-counterclockwise) = Direction.East :=
sorry

end NUMINAMATH_CALUDE_spinner_direction_l3784_378442


namespace NUMINAMATH_CALUDE_quadratic_general_form_l3784_378493

/-- Given a quadratic equation 3x² + 1 = 7x, its general form is 3x² - 7x + 1 = 0 -/
theorem quadratic_general_form : 
  ∀ x : ℝ, 3 * x^2 + 1 = 7 * x ↔ 3 * x^2 - 7 * x + 1 = 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_general_form_l3784_378493


namespace NUMINAMATH_CALUDE_domain_of_f_l3784_378431

open Real Set

noncomputable def f (x : ℝ) : ℝ := log (2 * sin x - 1) + sqrt (1 - 2 * cos x)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = ⋃ (k : ℤ), Ico (2 * k * π + π / 3) (2 * k * π + 5 * π / 6) :=
by sorry

end NUMINAMATH_CALUDE_domain_of_f_l3784_378431


namespace NUMINAMATH_CALUDE_square_root_fraction_equals_one_l3784_378463

theorem square_root_fraction_equals_one : 
  Real.sqrt (3^2 + 4^2) / Real.sqrt (20 + 5) = 1 := by sorry

end NUMINAMATH_CALUDE_square_root_fraction_equals_one_l3784_378463


namespace NUMINAMATH_CALUDE_pentagon_area_l3784_378457

/-- The area of a pentagon formed by an equilateral triangle sharing a side with a square -/
theorem pentagon_area (s : ℝ) (h_perimeter : 5 * s = 20) : 
  s^2 + (s^2 * Real.sqrt 3) / 4 = 16 + 4 * Real.sqrt 3 := by
  sorry

#check pentagon_area

end NUMINAMATH_CALUDE_pentagon_area_l3784_378457


namespace NUMINAMATH_CALUDE_fewer_bees_than_flowers_l3784_378437

theorem fewer_bees_than_flowers : 
  let flowers : ℕ := 5
  let bees : ℕ := 3
  flowers - bees = 2 := by sorry

end NUMINAMATH_CALUDE_fewer_bees_than_flowers_l3784_378437


namespace NUMINAMATH_CALUDE_matthew_crackers_l3784_378496

theorem matthew_crackers (initial : ℕ) (friends : ℕ) (given_each : ℕ) (left : ℕ) : 
  friends = 3 → given_each = 7 → left = 17 → 
  initial = friends * given_each + left → initial = 38 :=
by sorry

end NUMINAMATH_CALUDE_matthew_crackers_l3784_378496


namespace NUMINAMATH_CALUDE_system_I_solution_system_II_solution_l3784_378476

-- System I
theorem system_I_solution :
  ∃ (x y : ℝ), (y = x + 3 ∧ x - 2*y + 12 = 0) → (x = 6 ∧ y = 9) := by sorry

-- System II
theorem system_II_solution :
  ∃ (x y : ℝ), (4*(x - y - 1) = 3*(1 - y) - 2 ∧ x/2 + y/3 = 2) → (x = 2 ∧ y = 3) := by sorry

end NUMINAMATH_CALUDE_system_I_solution_system_II_solution_l3784_378476


namespace NUMINAMATH_CALUDE_odd_quadruple_composition_l3784_378460

/-- A function g is odd if g(-x) = -g(x) for all x -/
def IsOdd (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = -g x

theorem odd_quadruple_composition (g : ℝ → ℝ) (h : IsOdd g) :
  IsOdd (fun x ↦ g (g (g (g x)))) := by
  sorry

end NUMINAMATH_CALUDE_odd_quadruple_composition_l3784_378460


namespace NUMINAMATH_CALUDE_solution_difference_l3784_378477

theorem solution_difference (r s : ℝ) : 
  ((5 * r - 20) / (r^2 + 3*r - 18) = r + 3) →
  ((5 * s - 20) / (s^2 + 3*s - 18) = s + 3) →
  (r ≠ s) →
  (r > s) →
  (r - s = Real.sqrt 29) :=
by sorry

end NUMINAMATH_CALUDE_solution_difference_l3784_378477


namespace NUMINAMATH_CALUDE_rope_length_ratio_l3784_378443

/-- Given three ropes with lengths A, B, and C, where A is the longest, B is the middle, and C is the shortest,
    if A + C = B + 100 and C = 80, then the ratio of their lengths is (B + 20):B:80. -/
theorem rope_length_ratio (A B C : ℕ) (h1 : A ≥ B) (h2 : B ≥ C) (h3 : A + C = B + 100) (h4 : C = 80) :
  ∃ (k : ℕ), k > 0 ∧ A = k * (B + 20) ∧ B = k * B ∧ C = k * 80 :=
sorry

end NUMINAMATH_CALUDE_rope_length_ratio_l3784_378443


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3784_378462

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a → a 3 + a 8 = 10 → 3 * a 5 + a 7 = 20 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3784_378462


namespace NUMINAMATH_CALUDE_joyce_apples_l3784_378402

theorem joyce_apples (initial : Real) (received : Real) : 
  initial = 75.0 → received = 52.0 → initial + received = 127.0 := by
  sorry

end NUMINAMATH_CALUDE_joyce_apples_l3784_378402


namespace NUMINAMATH_CALUDE_solve_equation_l3784_378455

theorem solve_equation : ∃ x : ℚ, (3 * x + 5) / 7 = 13 ∧ x = 86 / 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3784_378455


namespace NUMINAMATH_CALUDE_chess_club_mixed_groups_l3784_378445

theorem chess_club_mixed_groups 
  (total_children : Nat) 
  (total_groups : Nat) 
  (group_size : Nat)
  (boy_games : Nat)
  (girl_games : Nat) :
  total_children = 90 →
  total_groups = 30 →
  group_size = 3 →
  boy_games = 30 →
  girl_games = 14 →
  (∃ (mixed_groups : Nat), 
    mixed_groups = 23 ∧ 
    mixed_groups * 2 = total_children - boy_games - girl_games) := by
  sorry

end NUMINAMATH_CALUDE_chess_club_mixed_groups_l3784_378445


namespace NUMINAMATH_CALUDE_rectangular_field_diagonal_shortcut_l3784_378438

theorem rectangular_field_diagonal_shortcut (x y : ℝ) (hxy : 0 < x ∧ x < y) :
  x + y - Real.sqrt (x^2 + y^2) = (1/3) * y →
  x / y = 5/12 := by
sorry

end NUMINAMATH_CALUDE_rectangular_field_diagonal_shortcut_l3784_378438


namespace NUMINAMATH_CALUDE_nested_root_equality_l3784_378444

theorem nested_root_equality (x : ℝ) (h : x ≥ 0) :
  Real.sqrt (x * Real.sqrt (x * Real.sqrt x)) = (x ^ 7) ^ (1 / 8) := by
  sorry

end NUMINAMATH_CALUDE_nested_root_equality_l3784_378444
