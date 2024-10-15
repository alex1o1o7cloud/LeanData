import Mathlib

namespace NUMINAMATH_CALUDE_stock_price_increase_l10_1053

theorem stock_price_increase (x : ℝ) : 
  (1 + x / 100) * 0.75 * 1.25 = 1.125 → x = 20 := by
sorry

end NUMINAMATH_CALUDE_stock_price_increase_l10_1053


namespace NUMINAMATH_CALUDE_triangle_line_equation_l10_1006

/-- A line with slope 3/4 that forms a triangle with the coordinate axes -/
structure TriangleLine where
  /-- The y-intercept of the line -/
  b : ℝ
  /-- The perimeter of the triangle formed by the line and the coordinate axes is 12 -/
  perimeter_eq : |b| + |-(4/3)*b| + Real.sqrt (b^2 + (-(4/3)*b)^2) = 12

/-- The equation of a TriangleLine is either 3x-4y+12=0 or 3x-4y-12=0 -/
theorem triangle_line_equation (l : TriangleLine) :
  (3 : ℝ) * l.b = 12 ∨ (3 : ℝ) * l.b = -12 := by sorry

end NUMINAMATH_CALUDE_triangle_line_equation_l10_1006


namespace NUMINAMATH_CALUDE_distribute_five_items_to_fifteen_recipients_l10_1022

/-- The number of ways to distribute distinct items to recipients -/
def distribute_items (num_items : ℕ) (num_recipients : ℕ) : ℕ :=
  num_recipients ^ num_items

/-- Theorem: Distributing 5 distinct items to 15 recipients results in 759,375 possible ways -/
theorem distribute_five_items_to_fifteen_recipients :
  distribute_items 5 15 = 759375 := by
  sorry

end NUMINAMATH_CALUDE_distribute_five_items_to_fifteen_recipients_l10_1022


namespace NUMINAMATH_CALUDE_londolozi_lion_population_l10_1045

/-- Calculates the lion population after a given number of months -/
def lionPopulation (initialPopulation birthRate deathRate months : ℕ) : ℕ :=
  initialPopulation + birthRate * months - deathRate * months

/-- Theorem: The lion population in Londolozi after 12 months -/
theorem londolozi_lion_population :
  lionPopulation 100 5 1 12 = 148 := by
  sorry

#eval lionPopulation 100 5 1 12

end NUMINAMATH_CALUDE_londolozi_lion_population_l10_1045


namespace NUMINAMATH_CALUDE_complex_equation_solution_l10_1054

theorem complex_equation_solution (z : ℂ) (h : z * (1 + Complex.I) = 1 - Complex.I) : 
  z = -Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l10_1054


namespace NUMINAMATH_CALUDE_car_wash_earnings_difference_l10_1089

theorem car_wash_earnings_difference :
  ∀ (total : ℝ) (lisa_earnings : ℝ) (tommy_earnings : ℝ),
  total = 60 →
  lisa_earnings = total / 2 →
  tommy_earnings = lisa_earnings / 2 →
  lisa_earnings - tommy_earnings = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_car_wash_earnings_difference_l10_1089


namespace NUMINAMATH_CALUDE_johny_total_distance_l10_1057

def johny_journey (south_distance : ℕ) : ℕ :=
  let east_distance := south_distance + 20
  let north_distance := 2 * east_distance
  south_distance + east_distance + north_distance

theorem johny_total_distance :
  johny_journey 40 = 220 := by
  sorry

end NUMINAMATH_CALUDE_johny_total_distance_l10_1057


namespace NUMINAMATH_CALUDE_product_binary_ternary_l10_1055

/-- Converts a binary number represented as a list of digits to its decimal value -/
def binary_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * 2^i) 0

/-- Converts a ternary number represented as a list of digits to its decimal value -/
def ternary_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * 3^i) 0

/-- The main theorem stating that the product of 1011 (base 2) and 1021 (base 3) is 374 (base 10) -/
theorem product_binary_ternary :
  (binary_to_decimal [1, 1, 0, 1]) * (ternary_to_decimal [1, 2, 0, 1]) = 374 := by
  sorry

end NUMINAMATH_CALUDE_product_binary_ternary_l10_1055


namespace NUMINAMATH_CALUDE_work_scaling_l10_1083

/-- Given that 3 people can do 3 times of a particular work in 3 days,
    prove that 6 people can do 6 times of that work in the same number of days. -/
theorem work_scaling (work : ℕ → ℕ → ℕ → Prop) : 
  work 3 3 3 → work 6 6 3 :=
by sorry

end NUMINAMATH_CALUDE_work_scaling_l10_1083


namespace NUMINAMATH_CALUDE_charlie_coins_l10_1084

/-- The number of coins Alice and Charlie have satisfy the given conditions -/
def satisfy_conditions (a c : ℕ) : Prop :=
  (c + 2 = 5 * (a - 2)) ∧ (c - 2 = 4 * (a + 2))

/-- Charlie has 98 coins given the conditions -/
theorem charlie_coins : ∃ a : ℕ, satisfy_conditions a 98 := by
  sorry

end NUMINAMATH_CALUDE_charlie_coins_l10_1084


namespace NUMINAMATH_CALUDE_remainder_55_57_mod_7_l10_1004

theorem remainder_55_57_mod_7 : (55 * 57) % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_55_57_mod_7_l10_1004


namespace NUMINAMATH_CALUDE_partnership_investment_ratio_l10_1074

/-- Partnership investment problem -/
theorem partnership_investment_ratio :
  ∀ (n : ℚ) (c : ℚ),
  c > 0 →  -- C's investment is positive
  (2 / 3 * c) / (n * (2 / 3 * c) + (2 / 3 * c) + c) = 800 / 4400 →
  n = 3 :=
by sorry

end NUMINAMATH_CALUDE_partnership_investment_ratio_l10_1074


namespace NUMINAMATH_CALUDE_quiz_answer_key_l10_1030

theorem quiz_answer_key (n : ℕ) : 
  (2^5 - 2) * 4^n = 480 → n = 2 := by
  sorry

end NUMINAMATH_CALUDE_quiz_answer_key_l10_1030


namespace NUMINAMATH_CALUDE_kendras_goal_is_sixty_l10_1047

/-- Kendra's goal for new words to learn before her eighth birthday -/
def kendras_goal (words_learned : ℕ) (words_needed : ℕ) : ℕ :=
  words_learned + words_needed

/-- Theorem: Kendra's goal is 60 words -/
theorem kendras_goal_is_sixty :
  kendras_goal 36 24 = 60 := by
  sorry

end NUMINAMATH_CALUDE_kendras_goal_is_sixty_l10_1047


namespace NUMINAMATH_CALUDE_inequality_proof_l10_1090

theorem inequality_proof (a b : ℝ) (n : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : 1/a + 1/b = 1) :
  (a + b)^n - a^n - b^n ≥ 2^(2*n) - 2^(n+1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l10_1090


namespace NUMINAMATH_CALUDE_expand_and_simplify_l10_1035

theorem expand_and_simplify (x : ℝ) : (2*x - 3)^2 - (x + 3)*(x - 2) = 3*x^2 - 13*x + 15 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l10_1035


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l10_1073

theorem greatest_three_digit_multiple_of_17 :
  ∀ n : ℕ, n < 1000 → n % 17 = 0 → n ≤ 986 :=
by sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l10_1073


namespace NUMINAMATH_CALUDE_pizza_toppings_combinations_l10_1049

theorem pizza_toppings_combinations : Nat.choose 9 3 = 84 := by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_combinations_l10_1049


namespace NUMINAMATH_CALUDE_random_walk_properties_l10_1007

/-- A random walk on a line with forward probability 3/4 and backward probability 1/4 -/
structure RandomWalk where
  forwardProb : ℝ
  backwardProb : ℝ
  forwardProbEq : forwardProb = 3/4
  backwardProbEq : backwardProb = 1/4
  probSum : forwardProb + backwardProb = 1

/-- The probability of returning to the starting point after n steps -/
def returnProbability (rw : RandomWalk) (n : ℕ) : ℝ :=
  sorry

/-- The probability distribution of the distance from the starting point after n steps -/
def distanceProbability (rw : RandomWalk) (n : ℕ) (d : ℕ) : ℝ :=
  sorry

/-- The expected value of the distance from the starting point after n steps -/
def expectedDistance (rw : RandomWalk) (n : ℕ) : ℝ :=
  sorry

theorem random_walk_properties (rw : RandomWalk) :
  returnProbability rw 4 = 27/128 ∧
  distanceProbability rw 5 1 = 45/128 ∧
  distanceProbability rw 5 3 = 105/256 ∧
  distanceProbability rw 5 5 = 61/256 ∧
  expectedDistance rw 5 = 355/128 := by
  sorry

end NUMINAMATH_CALUDE_random_walk_properties_l10_1007


namespace NUMINAMATH_CALUDE_min_sum_of_mn_l10_1087

theorem min_sum_of_mn (m n : ℕ+) (h : m.val * n.val - 2 * m.val - 3 * n.val - 20 = 0) :
  ∃ (m' n' : ℕ+), m'.val * n'.val - 2 * m'.val - 3 * n'.val - 20 = 0 ∧ 
  m'.val + n'.val = 20 ∧ 
  ∀ (a b : ℕ+), a.val * b.val - 2 * a.val - 3 * b.val - 20 = 0 → a.val + b.val ≥ 20 :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_mn_l10_1087


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l10_1050

def P : Set ℝ := {0, 1, 2}
def N : Set ℝ := {x | x^2 - 3*x + 2 = 0}

theorem intersection_complement_theorem : P ∩ (Set.univ \ N) = {0} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l10_1050


namespace NUMINAMATH_CALUDE_greatest_perimeter_of_special_triangle_l10_1067

theorem greatest_perimeter_of_special_triangle :
  ∀ (a b c : ℕ),
    a > 0 ∧ b > 0 ∧ c > 0 →
    b = 4 * a →
    c = 20 →
    a + b > c ∧ a + c > b ∧ b + c > a →
    a + b + c ≤ 50 :=
by sorry

end NUMINAMATH_CALUDE_greatest_perimeter_of_special_triangle_l10_1067


namespace NUMINAMATH_CALUDE_ratio_of_divisor_sums_l10_1036

def M : ℕ := 45 * 45 * 98 * 340

def sum_of_even_divisors (n : ℕ) : ℕ := (List.filter (λ x => x % 2 = 0) (List.range (n + 1))).sum

def sum_of_odd_divisors (n : ℕ) : ℕ := (List.filter (λ x => x % 2 ≠ 0) (List.range (n + 1))).sum

theorem ratio_of_divisor_sums :
  (sum_of_even_divisors M) / (sum_of_odd_divisors M) = 14 := by sorry

end NUMINAMATH_CALUDE_ratio_of_divisor_sums_l10_1036


namespace NUMINAMATH_CALUDE_intersection_line_equation_l10_1014

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 4*y - 1 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 13 = 0

-- Define the line
def line (x y : ℝ) : Prop := x - 2*y + 6 = 0

-- Theorem statement
theorem intersection_line_equation :
  ∀ (P Q : ℝ × ℝ),
  circle1 P.1 P.2 ∧ circle1 Q.1 Q.2 ∧
  circle2 P.1 P.2 ∧ circle2 Q.1 Q.2 ∧
  P ≠ Q →
  line P.1 P.2 ∧ line Q.1 Q.2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_line_equation_l10_1014


namespace NUMINAMATH_CALUDE_building_floors_l10_1028

/-- Represents a staircase in the building -/
structure Staircase where
  steps : ℕ

/-- Represents the building with three staircases -/
structure Building where
  staircase_a : Staircase
  staircase_b : Staircase
  staircase_c : Staircase

/-- The number of floors in the building is equal to the GCD of the number of steps in each staircase -/
theorem building_floors (b : Building) 
  (h1 : b.staircase_a.steps = 104)
  (h2 : b.staircase_b.steps = 117)
  (h3 : b.staircase_c.steps = 156) : 
  ∃ (floors : ℕ), floors = Nat.gcd (Nat.gcd b.staircase_a.steps b.staircase_b.steps) b.staircase_c.steps ∧ 
    floors = 13 := by
  sorry

end NUMINAMATH_CALUDE_building_floors_l10_1028


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l10_1071

theorem partial_fraction_decomposition :
  ∃ (A B : ℝ),
    (∀ x : ℝ, x ≠ 12 ∧ x ≠ -3 →
      (6 * x + 3) / (x^2 - 9 * x - 36) = A / (x - 12) + B / (x + 3)) ∧
    A = 5 ∧
    B = 1 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l10_1071


namespace NUMINAMATH_CALUDE_factorial_sum_equation_l10_1085

theorem factorial_sum_equation : ∃ n m : ℕ, n * n.factorial + m * m.factorial = 4032 ∧ n = 7 ∧ m = 6 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_equation_l10_1085


namespace NUMINAMATH_CALUDE_steves_commute_l10_1099

/-- The distance from Steve's house to work -/
def distance : ℝ := sorry

/-- Steve's speed on the way to work -/
def speed_to_work : ℝ := sorry

/-- Steve's speed on the way back from work -/
def speed_from_work : ℝ := 10

/-- Total time Steve spends on the road daily -/
def total_time : ℝ := 6

theorem steves_commute :
  (speed_from_work = 2 * speed_to_work) →
  (distance / speed_to_work + distance / speed_from_work = total_time) →
  distance = 20 := by sorry

end NUMINAMATH_CALUDE_steves_commute_l10_1099


namespace NUMINAMATH_CALUDE_opposite_signs_sum_zero_l10_1042

theorem opposite_signs_sum_zero (a b : ℝ) : a * b < 0 → a + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_opposite_signs_sum_zero_l10_1042


namespace NUMINAMATH_CALUDE_system_solutions_l10_1052

def system (x y z : ℝ) : Prop :=
  x + y + z = 8 ∧ x * y * z = 8 ∧ 1/x - 1/y - 1/z = 1/8

def solution_set : Set (ℝ × ℝ × ℝ) :=
  { (1, (7 + Real.sqrt 17)/2, (7 - Real.sqrt 17)/2),
    (1, (7 - Real.sqrt 17)/2, (7 + Real.sqrt 17)/2),
    (-1, (9 + Real.sqrt 113)/2, (9 - Real.sqrt 113)/2),
    (-1, (9 - Real.sqrt 113)/2, (9 + Real.sqrt 113)/2) }

theorem system_solutions :
  ∀ x y z : ℝ, system x y z ↔ (x, y, z) ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_system_solutions_l10_1052


namespace NUMINAMATH_CALUDE_round_robin_tournament_probability_l10_1008

def num_teams : ℕ := 5

-- Define the type for tournament outcomes
def TournamentOutcome := Fin num_teams → Fin num_teams

-- Function to check if an outcome has unique win counts
def has_unique_win_counts (outcome : TournamentOutcome) : Prop :=
  ∀ i j, i ≠ j → outcome i ≠ outcome j

-- Total number of possible outcomes
def total_outcomes : ℕ := 2^(num_teams * (num_teams - 1) / 2)

-- Number of favorable outcomes (where no two teams have the same number of wins)
def favorable_outcomes : ℕ := Nat.factorial num_teams

-- The probability we want to prove
def target_probability : ℚ := favorable_outcomes / total_outcomes

theorem round_robin_tournament_probability :
  target_probability = 15 / 128 := by sorry

end NUMINAMATH_CALUDE_round_robin_tournament_probability_l10_1008


namespace NUMINAMATH_CALUDE_distance_between_points_l10_1051

theorem distance_between_points :
  let p1 : ℝ × ℝ := (5, 5)
  let p2 : ℝ × ℝ := (0, 0)
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2) = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l10_1051


namespace NUMINAMATH_CALUDE_ten_boys_handshakes_l10_1075

/-- The number of handshakes in a group of boys with special conditions -/
def specialHandshakes (n : ℕ) : ℕ :=
  n * (n - 1) / 2 - 2

/-- Theorem: In a group of 10 boys with the given handshake conditions, 
    the total number of handshakes is 43 -/
theorem ten_boys_handshakes : specialHandshakes 10 = 43 := by
  sorry

end NUMINAMATH_CALUDE_ten_boys_handshakes_l10_1075


namespace NUMINAMATH_CALUDE_solution_characterization_l10_1092

/-- The set of ordered pairs (m, n) that satisfy the given condition -/
def SolutionSet : Set (ℕ × ℕ) :=
  {(2, 2), (2, 1), (3, 1), (1, 2), (1, 3), (5, 2), (5, 3), (2, 5), (3, 5)}

/-- Predicate to check if a pair (m, n) satisfies the condition -/
def SatisfiesCondition (p : ℕ × ℕ) : Prop :=
  let m := p.1
  let n := p.2
  m > 0 ∧ n > 0 ∧ ∃ k : ℤ, (n^3 + 1 : ℤ) = k * (m * n - 1)

theorem solution_characterization :
  ∀ p : ℕ × ℕ, p ∈ SolutionSet ↔ SatisfiesCondition p :=
sorry

end NUMINAMATH_CALUDE_solution_characterization_l10_1092


namespace NUMINAMATH_CALUDE_marble_count_l10_1063

theorem marble_count (blue : ℕ) (yellow : ℕ) (p_yellow : ℚ) (red : ℕ) :
  blue = 7 →
  yellow = 6 →
  p_yellow = 1/4 →
  red = blue + yellow + red →
  yellow = p_yellow * (blue + yellow + red) →
  red = 11 := by sorry

end NUMINAMATH_CALUDE_marble_count_l10_1063


namespace NUMINAMATH_CALUDE_minimum_k_value_l10_1078

theorem minimum_k_value (k : ℝ) : 
  (∀ x y : ℝ, Real.sqrt x + Real.sqrt y ≤ k * Real.sqrt (x + y)) → 
  k ≥ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_minimum_k_value_l10_1078


namespace NUMINAMATH_CALUDE_rotten_tomatoes_solution_l10_1061

/-- Represents the problem of calculating rotten tomatoes --/
def RottenTomatoesProblem (crate_capacity : ℕ) (num_crates : ℕ) (total_cost : ℕ) (selling_price : ℕ) (profit : ℕ) : Prop :=
  let total_capacity := crate_capacity * num_crates
  let revenue := total_cost + profit
  let sold_kg := revenue / selling_price
  total_capacity - sold_kg = 3

/-- Theorem stating the solution to the rotten tomatoes problem --/
theorem rotten_tomatoes_solution :
  RottenTomatoesProblem 20 3 330 6 12 := by
  sorry

#check rotten_tomatoes_solution

end NUMINAMATH_CALUDE_rotten_tomatoes_solution_l10_1061


namespace NUMINAMATH_CALUDE_pentagon_y_coordinate_of_C_l10_1010

/-- Pentagon with vertices A, B, C, D, E in 2D space -/
structure Pentagon where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ

/-- Calculate the area of a triangle given three vertices -/
def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

/-- Calculate the area of a pentagon -/
def pentagonArea (p : Pentagon) : ℝ := sorry

/-- Check if a pentagon has a vertical line of symmetry -/
def hasVerticalSymmetry (p : Pentagon) : Prop := sorry

/-- Main theorem -/
theorem pentagon_y_coordinate_of_C (p : Pentagon) 
  (h1 : p.A = (0, 0))
  (h2 : p.B = (0, 4))
  (h3 : p.D = (4, 4))
  (h4 : p.E = (4, 0))
  (h5 : ∃ y, p.C = (2, y))
  (h6 : hasVerticalSymmetry p)
  (h7 : pentagonArea p = 40) :
  ∃ y, p.C = (2, y) ∧ y = 16 := by sorry

end NUMINAMATH_CALUDE_pentagon_y_coordinate_of_C_l10_1010


namespace NUMINAMATH_CALUDE_max_sum_ab_l10_1033

/-- Represents a digit from 0 to 9 -/
def Digit := Fin 10

/-- Given four distinct digits A, B, C, D, where (A+B)/(C+D) is an integer
    and C+D > 1, the maximum value of A+B is 15 -/
theorem max_sum_ab (A B C D : Digit) 
  (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (h_integer : ∃ k : ℕ+, k * (C.val + D.val) = A.val + B.val)
  (h_cd_gt_one : C.val + D.val > 1) :
  A.val + B.val ≤ 15 :=
sorry

end NUMINAMATH_CALUDE_max_sum_ab_l10_1033


namespace NUMINAMATH_CALUDE_arrival_time_difference_l10_1098

-- Define the constants
def distance : ℝ := 2
def jenna_speed : ℝ := 12
def jamie_speed : ℝ := 6

-- Define the theorem
theorem arrival_time_difference : 
  (distance / jenna_speed * 60 - distance / jamie_speed * 60) = 10 := by
  sorry

end NUMINAMATH_CALUDE_arrival_time_difference_l10_1098


namespace NUMINAMATH_CALUDE_tourist_tax_calculation_l10_1012

/-- Represents the tax system in Country B -/
structure TaxSystem where
  taxFreeLimit : ℝ
  bracket1Rate : ℝ
  bracket1Limit : ℝ
  bracket2Rate : ℝ
  bracket2Limit : ℝ
  bracket3Rate : ℝ
  electronicsRate : ℝ
  luxuryRate : ℝ
  studentDiscount : ℝ

/-- Represents a purchase made by a tourist -/
structure Purchase where
  totalValue : ℝ
  electronicsValue : ℝ
  luxuryValue : ℝ
  educationalValue : ℝ
  hasStudentID : Bool

def calculateTax (system : TaxSystem) (purchase : Purchase) : ℝ :=
  sorry

theorem tourist_tax_calculation (system : TaxSystem) (purchase : Purchase) :
  system.taxFreeLimit = 600 ∧
  system.bracket1Rate = 0.12 ∧
  system.bracket1Limit = 1000 ∧
  system.bracket2Rate = 0.18 ∧
  system.bracket2Limit = 1500 ∧
  system.bracket3Rate = 0.25 ∧
  system.electronicsRate = 0.05 ∧
  system.luxuryRate = 0.10 ∧
  system.studentDiscount = 0.05 ∧
  purchase.totalValue = 2100 ∧
  purchase.electronicsValue = 900 ∧
  purchase.luxuryValue = 820 ∧
  purchase.educationalValue = 380 ∧
  purchase.hasStudentID = true
  →
  calculateTax system purchase = 304 :=
by sorry

end NUMINAMATH_CALUDE_tourist_tax_calculation_l10_1012


namespace NUMINAMATH_CALUDE_consecutive_odd_squares_difference_l10_1064

theorem consecutive_odd_squares_difference (n : ℤ) : 
  ∃ k : ℤ, (2*n + 1)^2 - (2*n - 1)^2 = 8*k := by sorry

end NUMINAMATH_CALUDE_consecutive_odd_squares_difference_l10_1064


namespace NUMINAMATH_CALUDE_wire_length_around_square_field_l10_1068

/-- The length of wire required to go 15 times round a square field with area 69696 m^2 is 15840 meters. -/
theorem wire_length_around_square_field : 
  let field_area : ℝ := 69696
  let side_length : ℝ := (field_area) ^ (1/2 : ℝ)
  let perimeter : ℝ := 4 * side_length
  let wire_length : ℝ := 15 * perimeter
  wire_length = 15840 := by sorry

end NUMINAMATH_CALUDE_wire_length_around_square_field_l10_1068


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l10_1040

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) := by sorry

theorem negation_of_proposition :
  (¬ ∃ x : ℝ, Real.exp x - x - 1 ≤ 0) ↔ (∀ x : ℝ, Real.exp x - x - 1 > 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l10_1040


namespace NUMINAMATH_CALUDE_parity_of_S_l10_1009

theorem parity_of_S (a b c n : ℤ) 
  (h1 : (a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 0) ∨ 
        (a % 2 = 1 ∧ b % 2 = 0 ∧ c % 2 = 1) ∨ 
        (a % 2 = 0 ∧ b % 2 = 1 ∧ c % 2 = 1)) :
  let S := (a + 2*n + 1) * (b + 2*n + 2) * (c + 2*n + 3)
  S % 2 = 0 := by
sorry

end NUMINAMATH_CALUDE_parity_of_S_l10_1009


namespace NUMINAMATH_CALUDE_percentage_seniors_in_statistics_l10_1058

theorem percentage_seniors_in_statistics :
  ∀ (total_students : ℕ) (seniors_in_statistics : ℕ),
    total_students = 120 →
    seniors_in_statistics = 54 →
    (seniors_in_statistics : ℚ) / ((total_students : ℚ) / 2) * 100 = 90 := by
  sorry

end NUMINAMATH_CALUDE_percentage_seniors_in_statistics_l10_1058


namespace NUMINAMATH_CALUDE_right_triangles_with_increasing_sides_l10_1097

theorem right_triangles_with_increasing_sides (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_pyth1 : a^2 + (b-100)^2 = (c-30)^2)
  (h_pyth2 : a^2 + b^2 = c^2)
  (h_pyth3 : a^2 + (b+100)^2 = (c+40)^2) :
  a = 819 ∧ b = 308 ∧ c = 875 := by
  sorry

end NUMINAMATH_CALUDE_right_triangles_with_increasing_sides_l10_1097


namespace NUMINAMATH_CALUDE_haunted_castle_problem_l10_1041

/-- Represents a castle with windows -/
structure Castle where
  totalWindows : Nat
  forbiddenExitWindows : Nat

/-- Calculates the number of ways to enter and exit the castle -/
def waysToEnterAndExit (castle : Castle) : Nat :=
  castle.totalWindows * (castle.totalWindows - 1 - castle.forbiddenExitWindows)

/-- The haunted castle problem -/
theorem haunted_castle_problem :
  let castle : Castle := { totalWindows := 8, forbiddenExitWindows := 2 }
  waysToEnterAndExit castle = 40 := by
  sorry

end NUMINAMATH_CALUDE_haunted_castle_problem_l10_1041


namespace NUMINAMATH_CALUDE_family_money_difference_l10_1034

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 25 / 100

/-- The value of a dime in dollars -/
def dime_value : ℚ := 10 / 100

/-- The value of a nickel in dollars -/
def nickel_value : ℚ := 5 / 100

/-- Calculate the total value of coins for a person -/
def total_value (quarters dimes nickels : ℕ) : ℚ :=
  quarters * quarter_value + dimes * dime_value + nickels * nickel_value

/-- Karen's total value -/
def karen_value : ℚ := total_value 32 0 0

/-- Christopher's total value -/
def christopher_value : ℚ := total_value 64 0 0

/-- Emily's total value -/
def emily_value : ℚ := total_value 20 15 0

/-- Michael's total value -/
def michael_value : ℚ := total_value 12 10 25

/-- Sophia's total value -/
def sophia_value : ℚ := total_value 0 50 40

/-- Alex's total value -/
def alex_value : ℚ := total_value 0 25 100

/-- Total value for Karen and Christopher's family -/
def family1_value : ℚ := karen_value + christopher_value + emily_value + michael_value

/-- Total value for Sophia and Alex's family -/
def family2_value : ℚ := sophia_value + alex_value

theorem family_money_difference :
  family1_value - family2_value = 85/4 := by sorry

end NUMINAMATH_CALUDE_family_money_difference_l10_1034


namespace NUMINAMATH_CALUDE_geometric_series_sum_l10_1001

theorem geometric_series_sum : 
  let a : ℚ := 2
  let r : ℚ := -2/5
  let series : ℕ → ℚ := λ n => a * r^n
  ∑' n, series n = 10/7 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l10_1001


namespace NUMINAMATH_CALUDE_squirrel_acorns_l10_1082

theorem squirrel_acorns (total_acorns : ℕ) (winter_months : ℕ) (spring_acorns : ℕ) 
  (h1 : total_acorns = 210)
  (h2 : winter_months = 3)
  (h3 : spring_acorns = 30) :
  (total_acorns / winter_months) - (spring_acorns / winter_months) = 60 := by
  sorry

end NUMINAMATH_CALUDE_squirrel_acorns_l10_1082


namespace NUMINAMATH_CALUDE_mary_candy_count_l10_1043

theorem mary_candy_count (megan_candy : ℕ) (mary_multiplier : ℕ) (mary_additional : ℕ)
  (h1 : megan_candy = 5)
  (h2 : mary_multiplier = 3)
  (h3 : mary_additional = 10) :
  megan_candy * mary_multiplier + mary_additional = 25 :=
by sorry

end NUMINAMATH_CALUDE_mary_candy_count_l10_1043


namespace NUMINAMATH_CALUDE_inverse_proportion_k_value_l10_1015

/-- Given an inverse proportion function y = k/x passing through the point (2, -1), 
    prove that k = -2 -/
theorem inverse_proportion_k_value : 
  ∀ k : ℝ, (∀ x : ℝ, x ≠ 0 → -1 = k / 2) → k = -2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_k_value_l10_1015


namespace NUMINAMATH_CALUDE_abs_inequality_solution_set_l10_1005

theorem abs_inequality_solution_set (x : ℝ) : 
  |x - 3| < 1 ↔ 2 < x ∧ x < 4 :=
sorry

end NUMINAMATH_CALUDE_abs_inequality_solution_set_l10_1005


namespace NUMINAMATH_CALUDE_multiple_choice_probabilities_l10_1059

/-- Represents the scoring rules for multiple-choice questions -/
structure ScoringRules where
  all_correct : Nat
  some_correct : Nat
  incorrect_or_none : Nat

/-- Represents the probabilities of selecting different numbers of options -/
structure SelectionProbabilities where
  one_option : Real
  two_options : Real
  three_options : Real

/-- Represents a multiple-choice question with its correct answer -/
structure MultipleChoiceQuestion where
  correct_options : Nat

/-- Theorem stating the probabilities for the given scenario -/
theorem multiple_choice_probabilities 
  (rules : ScoringRules)
  (probs : SelectionProbabilities)
  (q11 q12 : MultipleChoiceQuestion)
  (h1 : rules.all_correct = 5 ∧ rules.some_correct = 2 ∧ rules.incorrect_or_none = 0)
  (h2 : probs.one_option = 1/3 ∧ probs.two_options = 1/3 ∧ probs.three_options = 1/3)
  (h3 : q11.correct_options = 2 ∧ q12.correct_options = 2) :
  (∃ (p1 p2 : Real),
    -- Probability of getting 2 points for question 11
    p1 = 1/6 ∧
    -- Probability of scoring a total of 7 points for questions 11 and 12
    p2 = 1/54) :=
  sorry

end NUMINAMATH_CALUDE_multiple_choice_probabilities_l10_1059


namespace NUMINAMATH_CALUDE_min_stamps_for_33_cents_l10_1062

def is_valid_combination (c f : ℕ) : Prop :=
  3 * c + 4 * f = 33

def total_stamps (c f : ℕ) : ℕ :=
  c + f

theorem min_stamps_for_33_cents :
  ∃ (c f : ℕ), is_valid_combination c f ∧
    total_stamps c f = 9 ∧
    ∀ (c' f' : ℕ), is_valid_combination c' f' →
      total_stamps c' f' ≥ 9 :=
sorry

end NUMINAMATH_CALUDE_min_stamps_for_33_cents_l10_1062


namespace NUMINAMATH_CALUDE_range_of_a_l10_1046

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x > a^2 - a - 3) → 
  a > -1 ∧ a < 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l10_1046


namespace NUMINAMATH_CALUDE_peters_cucumbers_l10_1044

/-- The problem of Peter's grocery shopping -/
theorem peters_cucumbers 
  (initial_amount : ℕ)
  (potato_kilos potato_price : ℕ)
  (tomato_kilos tomato_price : ℕ)
  (banana_kilos banana_price : ℕ)
  (cucumber_price : ℕ)
  (remaining_amount : ℕ)
  (h1 : initial_amount = 500)
  (h2 : potato_kilos = 6)
  (h3 : potato_price = 2)
  (h4 : tomato_kilos = 9)
  (h5 : tomato_price = 3)
  (h6 : banana_kilos = 3)
  (h7 : banana_price = 5)
  (h8 : cucumber_price = 4)
  (h9 : remaining_amount = 426)
  : ∃ (cucumber_kilos : ℕ), 
    initial_amount - 
    (potato_kilos * potato_price + 
     tomato_kilos * tomato_price + 
     banana_kilos * banana_price + 
     cucumber_kilos * cucumber_price) = remaining_amount ∧ 
    cucumber_kilos = 5 := by
  sorry

end NUMINAMATH_CALUDE_peters_cucumbers_l10_1044


namespace NUMINAMATH_CALUDE_ratio_change_problem_l10_1025

theorem ratio_change_problem (x y z : ℝ) : 
  y / x = 3 / 2 →  -- Initial ratio
  y - x = 8 →  -- Difference between numbers
  (y + z) / (x + z) = 7 / 5 →  -- New ratio after adding z
  z = 4 :=  -- The number added to both
by sorry

end NUMINAMATH_CALUDE_ratio_change_problem_l10_1025


namespace NUMINAMATH_CALUDE_lcm_factor_problem_l10_1020

theorem lcm_factor_problem (A B : ℕ+) (h : Nat.gcd A B = 25) (hA : A = 350) 
  (hlcm : ∃ x : ℕ+, Nat.lcm A B = 25 * 13 * x) : 
  ∃ x : ℕ+, Nat.lcm A B = 25 * 13 * x ∧ x = 14 := by
  sorry

end NUMINAMATH_CALUDE_lcm_factor_problem_l10_1020


namespace NUMINAMATH_CALUDE_stuffed_animals_sales_difference_l10_1094

/-- Given the sales of stuffed animals by Jake, Thor, and Quincy, prove that Quincy sold 170 more than Jake. -/
theorem stuffed_animals_sales_difference :
  ∀ (jake_sales thor_sales quincy_sales : ℕ),
  jake_sales = thor_sales + 10 →
  quincy_sales = 10 * thor_sales →
  quincy_sales = 200 →
  quincy_sales - jake_sales = 170 := by
sorry

end NUMINAMATH_CALUDE_stuffed_animals_sales_difference_l10_1094


namespace NUMINAMATH_CALUDE_notebook_notepad_pen_cost_l10_1029

theorem notebook_notepad_pen_cost (x y z : ℤ) : 
  x + 3*y + 2*z = 98 →
  3*x + y = 5*z - 36 →
  Even x →
  x = 4 ∧ y = 22 ∧ z = 14 := by
sorry

end NUMINAMATH_CALUDE_notebook_notepad_pen_cost_l10_1029


namespace NUMINAMATH_CALUDE_larger_number_problem_l10_1079

theorem larger_number_problem (x y : ℝ) (h1 : x + y = 40) (h2 : x - y = 10) : 
  max x y = 25 := by
sorry

end NUMINAMATH_CALUDE_larger_number_problem_l10_1079


namespace NUMINAMATH_CALUDE_seats_per_bus_correct_l10_1019

/-- Represents a school with classrooms, students, and buses for a field trip. -/
structure School where
  classrooms : ℕ
  students_per_classroom : ℕ
  seats_per_bus : ℕ

/-- Calculates the total number of students in the school. -/
def total_students (s : School) : ℕ :=
  s.classrooms * s.students_per_classroom

/-- Calculates the number of buses needed for the field trip. -/
def buses_needed (s : School) : ℕ :=
  (total_students s + s.seats_per_bus - 1) / s.seats_per_bus

/-- Theorem stating that for a school with 87 classrooms, 58 students per classroom,
    and buses with 29 seats each, the number of seats on each school bus is 29. -/
theorem seats_per_bus_correct (s : School) 
  (h1 : s.classrooms = 87)
  (h2 : s.students_per_classroom = 58)
  (h3 : s.seats_per_bus = 29) :
  s.seats_per_bus = 29 := by
  sorry

#eval buses_needed { classrooms := 87, students_per_classroom := 58, seats_per_bus := 29 }

end NUMINAMATH_CALUDE_seats_per_bus_correct_l10_1019


namespace NUMINAMATH_CALUDE_power_sum_equals_four_sqrt_three_over_three_logarithm_equation_solution_l10_1065

-- Define a as log_4(3)
noncomputable def a : ℝ := Real.log 3 / Real.log 4

-- Theorem 1: 2^a + 2^(-a) = (4 * sqrt(3)) / 3
theorem power_sum_equals_four_sqrt_three_over_three :
  2^a + 2^(-a) = (4 * Real.sqrt 3) / 3 := by sorry

-- Theorem 2: The solution to log_2(9^(x-1) - 5) = log_2(3^(x-1) - 2) + 2 is x = 2
theorem logarithm_equation_solution :
  ∃! x : ℝ, (x > 1 ∧ Real.log (9^(x-1) - 5) / Real.log 2 = Real.log (3^(x-1) - 2) / Real.log 2 + 2) ∧ x = 2 := by sorry

end NUMINAMATH_CALUDE_power_sum_equals_four_sqrt_three_over_three_logarithm_equation_solution_l10_1065


namespace NUMINAMATH_CALUDE_supplementary_angles_ratio_l10_1060

theorem supplementary_angles_ratio (a b : ℝ) : 
  a + b = 180 →  -- angles are supplementary
  a = 4 * b →    -- angles are in ratio 4:1
  b = 36 :=      -- smaller angle is 36°
by sorry

end NUMINAMATH_CALUDE_supplementary_angles_ratio_l10_1060


namespace NUMINAMATH_CALUDE_minimum_race_distance_proof_l10_1076

/-- The minimum distance a runner must travel in the race setup -/
def minimum_race_distance : ℝ := 1011

/-- Point A's vertical distance from the wall -/
def distance_A_to_wall : ℝ := 400

/-- Point B's vertical distance above the wall -/
def distance_B_above_wall : ℝ := 600

/-- Point B's horizontal distance to the right of point A -/
def horizontal_distance_A_to_B : ℝ := 150

/-- Theorem stating the minimum distance a runner must travel -/
theorem minimum_race_distance_proof :
  let total_vertical_distance := distance_A_to_wall + distance_B_above_wall
  let squared_distance := horizontal_distance_A_to_B ^ 2 + total_vertical_distance ^ 2
  Real.sqrt squared_distance = minimum_race_distance := by sorry

end NUMINAMATH_CALUDE_minimum_race_distance_proof_l10_1076


namespace NUMINAMATH_CALUDE_triangle_area_is_one_third_of_square_l10_1032

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square in 2D space -/
structure Square where
  bottomLeft : Point
  topRight : Point

/-- Calculates the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : Point) : ℝ :=
  0.5 * abs ((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y))

/-- Calculates the area of a square -/
def squareArea (s : Square) : ℝ :=
  (s.topRight.x - s.bottomLeft.x) * (s.topRight.y - s.bottomLeft.y)

/-- Main theorem: The area of the triangle formed by the line and the bottom of the square
    is 1/3 of the total square area -/
theorem triangle_area_is_one_third_of_square (s : Square)
  (p1 p2 : Point)
  (h1 : s.bottomLeft = ⟨2, 1⟩)
  (h2 : s.topRight = ⟨5, 4⟩)
  (h3 : p1 = ⟨2, 3⟩)
  (h4 : p2 = ⟨5, 1⟩) :
  triangleArea p1 p2 s.bottomLeft / squareArea s = 1/3 := by
  sorry


end NUMINAMATH_CALUDE_triangle_area_is_one_third_of_square_l10_1032


namespace NUMINAMATH_CALUDE_puppies_per_dog_l10_1017

/-- Given information about Chuck's dog breeding operation -/
structure DogBreeding where
  num_pregnant_dogs : ℕ
  shots_per_puppy : ℕ
  cost_per_shot : ℕ
  total_shot_cost : ℕ

/-- Theorem stating the number of puppies per pregnant dog -/
theorem puppies_per_dog (d : DogBreeding)
  (h1 : d.num_pregnant_dogs = 3)
  (h2 : d.shots_per_puppy = 2)
  (h3 : d.cost_per_shot = 5)
  (h4 : d.total_shot_cost = 120) :
  d.total_shot_cost / (d.num_pregnant_dogs * d.shots_per_puppy * d.cost_per_shot) = 4 := by
  sorry

end NUMINAMATH_CALUDE_puppies_per_dog_l10_1017


namespace NUMINAMATH_CALUDE_toms_deck_cost_l10_1088

/-- Represents the cost of a deck of cards -/
def deck_cost (rare_count : ℕ) (uncommon_count : ℕ) (common_count : ℕ) 
              (rare_price : ℚ) (uncommon_price : ℚ) (common_price : ℚ) : ℚ :=
  rare_count * rare_price + uncommon_count * uncommon_price + common_count * common_price

/-- Theorem stating that the cost of Tom's deck is $32 -/
theorem toms_deck_cost : 
  deck_cost 19 11 30 1 (1/2) (1/4) = 32 := by
  sorry

end NUMINAMATH_CALUDE_toms_deck_cost_l10_1088


namespace NUMINAMATH_CALUDE_union_necessary_not_sufficient_for_complement_l10_1013

universe u

variable {U : Type u}
variable (A B : Set U)

theorem union_necessary_not_sufficient_for_complement :
  (∀ (A B : Set U), B = Aᶜ → A ∪ B = Set.univ) ∧
  (∃ (A B : Set U), A ∪ B = Set.univ ∧ B ≠ Aᶜ) :=
sorry

end NUMINAMATH_CALUDE_union_necessary_not_sufficient_for_complement_l10_1013


namespace NUMINAMATH_CALUDE_work_completion_time_l10_1081

theorem work_completion_time (x_time y_time : ℝ) (hx : x_time = 30) (hy : y_time = 45) :
  (1 / x_time + 1 / y_time)⁻¹ = 18 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l10_1081


namespace NUMINAMATH_CALUDE_bella_roses_l10_1000

def dozen : ℕ := 12

def roses_from_parents : ℕ := 2 * dozen

def number_of_friends : ℕ := 10

def roses_per_friend : ℕ := 2

def total_roses : ℕ := roses_from_parents + number_of_friends * roses_per_friend

theorem bella_roses : total_roses = 44 := by sorry

end NUMINAMATH_CALUDE_bella_roses_l10_1000


namespace NUMINAMATH_CALUDE_compare_squares_l10_1077

theorem compare_squares (a : ℝ) : (a + 1)^2 > a^2 + 2*a := by
  sorry

end NUMINAMATH_CALUDE_compare_squares_l10_1077


namespace NUMINAMATH_CALUDE_rotation_180_maps_points_l10_1037

def rotate180 (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

theorem rotation_180_maps_points :
  let C : ℝ × ℝ := (-1, 2)
  let D : ℝ × ℝ := (3, 2)
  let C' : ℝ × ℝ := (1, -2)
  let D' : ℝ × ℝ := (-3, -2)
  rotate180 C = C' ∧ rotate180 D = D' := by sorry

end NUMINAMATH_CALUDE_rotation_180_maps_points_l10_1037


namespace NUMINAMATH_CALUDE_polynomial_nonnegative_l10_1070

theorem polynomial_nonnegative (a : ℝ) : a^2 * (a^2 - 1) - a^2 + 1 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_nonnegative_l10_1070


namespace NUMINAMATH_CALUDE_willys_age_proof_l10_1048

theorem willys_age_proof :
  ∃ (P : ℤ → ℤ) (A : ℤ),
    (∀ x, ∃ (a₀ a₁ a₂ a₃ : ℤ), P x = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3) ∧
    P 7 = 77 ∧
    P 8 = 85 ∧
    A > 8 ∧
    P A = 0 ∧
    A = 14 := by
  sorry

end NUMINAMATH_CALUDE_willys_age_proof_l10_1048


namespace NUMINAMATH_CALUDE_slope_range_for_given_inclination_l10_1091

theorem slope_range_for_given_inclination (α : Real) (h : α ∈ Set.Icc (π / 4) (3 * π / 4)) :
  let k := Real.tan α
  k ∈ Set.Iic (-1) ∪ Set.Ici 1 :=
sorry

end NUMINAMATH_CALUDE_slope_range_for_given_inclination_l10_1091


namespace NUMINAMATH_CALUDE_three_digit_subtraction_convergence_l10_1038

-- Define a three-digit number type
def ThreeDigitNumber := { n : ℕ // n ≥ 100 ∧ n ≤ 999 }

-- Function to reverse a three-digit number
def reverse (n : ThreeDigitNumber) : ThreeDigitNumber := sorry

-- Function to perform one step of the operation
def step (n : ThreeDigitNumber) : ThreeDigitNumber := sorry

-- Define the set of possible results
def ResultSet : Set ℕ := {0, 495}

-- Theorem statement
theorem three_digit_subtraction_convergence (start : ThreeDigitNumber) :
  ∃ (k : ℕ), (step^[k] start).val ∈ ResultSet := sorry

end NUMINAMATH_CALUDE_three_digit_subtraction_convergence_l10_1038


namespace NUMINAMATH_CALUDE_greater_than_negative_one_by_two_l10_1026

theorem greater_than_negative_one_by_two : 
  (fun x => x > -1 ∧ x - (-1) = 2) 1 := by sorry

end NUMINAMATH_CALUDE_greater_than_negative_one_by_two_l10_1026


namespace NUMINAMATH_CALUDE_no_equal_shards_l10_1086

theorem no_equal_shards : ¬∃ (x y : ℕ), 17 * x + 18 * (35 - y) = 17 * (25 - x) + 18 * y := by
  sorry

end NUMINAMATH_CALUDE_no_equal_shards_l10_1086


namespace NUMINAMATH_CALUDE_perpendicular_vector_with_sum_condition_l10_1021

/-- Given two parallel lines l and m with direction vector (4, 3),
    prove that (-6, 8) is perpendicular to their direction vector
    and its components sum to 2. -/
theorem perpendicular_vector_with_sum_condition :
  let direction_vector : ℝ × ℝ := (4, 3)
  let perpendicular_vector : ℝ × ℝ := (-6, 8)
  (direction_vector.1 * perpendicular_vector.1 + direction_vector.2 * perpendicular_vector.2 = 0) ∧
  (perpendicular_vector.1 + perpendicular_vector.2 = 2) := by
  sorry


end NUMINAMATH_CALUDE_perpendicular_vector_with_sum_condition_l10_1021


namespace NUMINAMATH_CALUDE_max_product_constraint_l10_1016

theorem max_product_constraint (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_sum : m + n = 3) :
  m * n ≤ 9 / 4 ∧ ∃ m₀ n₀ : ℝ, m₀ > 0 ∧ n₀ > 0 ∧ m₀ + n₀ = 3 ∧ m₀ * n₀ = 9 / 4 :=
by sorry

end NUMINAMATH_CALUDE_max_product_constraint_l10_1016


namespace NUMINAMATH_CALUDE_infinitely_many_divisible_by_prime_l10_1011

theorem infinitely_many_divisible_by_prime (p : ℕ) (hp : Prime p) :
  ∃ (N : Set ℕ), Set.Infinite N ∧ ∀ n ∈ N, p ∣ (2^n - n) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_divisible_by_prime_l10_1011


namespace NUMINAMATH_CALUDE_coordinate_problem_l10_1093

/-- Represents a point in the coordinate system -/
structure Point where
  x : ℕ
  y : ℕ

/-- The problem statement -/
theorem coordinate_problem (A B : Point) : 
  (A.x < A.y) →  -- Angle OA > 45°
  (B.x > B.y) →  -- Angle OB < 45°
  (B.x * B.y - A.x * A.y = 67) →  -- Area difference
  (A.x * 1000 + B.x * 100 + B.y * 10 + A.y = 1985) := by
  sorry

end NUMINAMATH_CALUDE_coordinate_problem_l10_1093


namespace NUMINAMATH_CALUDE_unique_solution_product_l10_1069

theorem unique_solution_product (r : ℝ) : 
  (∃! x : ℝ, (1 : ℝ) / (3 * x) = (r - 2 * x) / 10) → 
  (∃ r₁ r₂ : ℝ, r = r₁ ∨ r = r₂) ∧ r₁ * r₂ = -80 / 3 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_product_l10_1069


namespace NUMINAMATH_CALUDE_day_305_is_thursday_l10_1039

/-- Represents days of the week -/
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

/-- Represents the number of days after Wednesday -/
def daysAfterWednesday (n : Nat) : DayOfWeek :=
  match n % 7 with
  | 0 => DayOfWeek.Wednesday
  | 1 => DayOfWeek.Thursday
  | 2 => DayOfWeek.Friday
  | 3 => DayOfWeek.Saturday
  | 4 => DayOfWeek.Sunday
  | 5 => DayOfWeek.Monday
  | _ => DayOfWeek.Tuesday

theorem day_305_is_thursday :
  daysAfterWednesday (305 - 17) = DayOfWeek.Thursday := by
  sorry

#check day_305_is_thursday

end NUMINAMATH_CALUDE_day_305_is_thursday_l10_1039


namespace NUMINAMATH_CALUDE_jessica_remaining_money_l10_1027

/-- The remaining money after a purchase --/
def remaining_money (initial : ℚ) (spent : ℚ) : ℚ :=
  initial - spent

/-- Proof that Jessica's remaining money is $1.51 --/
theorem jessica_remaining_money :
  remaining_money 11.73 10.22 = 1.51 := by
  sorry

end NUMINAMATH_CALUDE_jessica_remaining_money_l10_1027


namespace NUMINAMATH_CALUDE_stratified_sampling_female_count_stratified_sampling_female_count_correct_l10_1031

theorem stratified_sampling_female_count 
  (total_employees : ℕ) 
  (male_employees : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_employees = 75) 
  (h2 : male_employees = 30) 
  (h3 : sample_size = 20) : ℕ :=
let female_employees := total_employees - male_employees
let sample_ratio := sample_size / total_employees
let female_sample := (female_employees : ℚ) * sample_ratio
12

theorem stratified_sampling_female_count_correct 
  (total_employees : ℕ) 
  (male_employees : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_employees = 75) 
  (h2 : male_employees = 30) 
  (h3 : sample_size = 20) : 
  stratified_sampling_female_count total_employees male_employees sample_size h1 h2 h3 = 12 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_female_count_stratified_sampling_female_count_correct_l10_1031


namespace NUMINAMATH_CALUDE_distribute_six_balls_three_boxes_l10_1003

/-- The number of ways to distribute indistinguishable balls into indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 7 ways to distribute 6 indistinguishable balls into 3 indistinguishable boxes -/
theorem distribute_six_balls_three_boxes : distribute_balls 6 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_distribute_six_balls_three_boxes_l10_1003


namespace NUMINAMATH_CALUDE_perpendicular_slope_l10_1096

theorem perpendicular_slope (x y : ℝ) :
  let original_line := {(x, y) | 4 * x - 6 * y = 12}
  let original_slope := 2 / 3
  let perpendicular_slope := -1 / original_slope
  perpendicular_slope = -3 / 2 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_slope_l10_1096


namespace NUMINAMATH_CALUDE_sons_age_l10_1002

/-- Proves that the son's age is 7.5 years given the conditions of the problem -/
theorem sons_age (son_age man_age : ℝ) : 
  man_age = son_age + 25 →
  man_age + 5 = 3 * (son_age + 5) →
  son_age = 7.5 := by
sorry

end NUMINAMATH_CALUDE_sons_age_l10_1002


namespace NUMINAMATH_CALUDE_fraction_reduction_l10_1095

theorem fraction_reduction (b y : ℝ) : 
  (Real.sqrt (b^2 + y^2) - (y^2 - b^2) / Real.sqrt (b^2 + y^2)) / (b^2 + y^2) = 
  2 * b^2 / (b^2 + y^2)^(3/2) := by sorry

end NUMINAMATH_CALUDE_fraction_reduction_l10_1095


namespace NUMINAMATH_CALUDE_expression_evaluation_l10_1072

theorem expression_evaluation (a b : ℝ) (h1 : a = 2) (h2 : b = -1) :
  ((2*a + 3*b) * (2*a - 3*b) - (2*a - b)^2 - 2*a*b) / (-2*b) = -7 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l10_1072


namespace NUMINAMATH_CALUDE_range_of_a_l10_1023

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 0 1, a ≥ Real.exp x

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 4*x + a = 0

-- Define the set of valid a values
def valid_a : Set ℝ := {a | a < Real.exp 1 ∨ a > 4}

-- Theorem statement
theorem range_of_a (a : ℝ) : ¬(p a ∧ q a) → a ∈ valid_a := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l10_1023


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l10_1066

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + x + 1 ≥ 0) ↔ (∃ x : ℝ, x^2 + x + 1 < 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l10_1066


namespace NUMINAMATH_CALUDE_spherical_coordinate_transformation_l10_1024

theorem spherical_coordinate_transformation (x y z ρ θ φ : ℝ) :
  x = ρ * Real.sin φ * Real.cos θ →
  y = ρ * Real.sin φ * Real.sin θ →
  z = ρ * Real.cos φ →
  x^2 + y^2 + z^2 = ρ^2 →
  x = 4 →
  y = -3 →
  z = -2 →
  ∃ (x' y' z' : ℝ),
    x' = ρ * Real.sin (-φ) * Real.cos (θ + π) ∧
    y' = ρ * Real.sin (-φ) * Real.sin (θ + π) ∧
    z' = ρ * Real.cos (-φ) ∧
    x' = -4 ∧
    y' = 3 ∧
    z' = -2 :=
by sorry

end NUMINAMATH_CALUDE_spherical_coordinate_transformation_l10_1024


namespace NUMINAMATH_CALUDE_quadratic_transformation_l10_1080

theorem quadratic_transformation (a b : ℝ) :
  (∀ x : ℝ, x^2 - 10*x + b = (x - a)^2 - 1) → b - a = 19 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_transformation_l10_1080


namespace NUMINAMATH_CALUDE_frost_90_cupcakes_l10_1056

/-- The number of cupcakes frosted by three people working together --/
def cupcakes_frosted (rate1 rate2 rate3 time : ℚ) : ℚ :=
  time / (1 / rate1 + 1 / rate2 + 1 / rate3)

/-- Theorem stating that Cagney, Lacey, and Jamie can frost 90 cupcakes in 10 minutes --/
theorem frost_90_cupcakes :
  cupcakes_frosted (1/20) (1/30) (1/15) 600 = 90 := by
  sorry

#eval cupcakes_frosted (1/20) (1/30) (1/15) 600

end NUMINAMATH_CALUDE_frost_90_cupcakes_l10_1056


namespace NUMINAMATH_CALUDE_inverse_proposition_l10_1018

theorem inverse_proposition : 
  (∀ a b : ℝ, a > b → b - a < 0) ↔ (∀ a b : ℝ, b - a < 0 → a > b) :=
sorry

end NUMINAMATH_CALUDE_inverse_proposition_l10_1018
