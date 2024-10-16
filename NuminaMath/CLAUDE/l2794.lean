import Mathlib

namespace NUMINAMATH_CALUDE_bacterial_eradication_l2794_279458

/-- Represents the state of the bacterial culture at a given minute -/
structure BacterialState where
  minute : ℕ
  infected : ℕ
  nonInfected : ℕ

/-- Models the evolution of the bacterial culture over time -/
def evolve (n : ℕ) : ℕ → BacterialState
  | 0 => ⟨0, 1, n - 1⟩
  | t + 1 => 
    let prev := evolve n t
    ⟨t + 1, 2 * prev.infected, (prev.nonInfected * 2) - (2 * prev.infected)⟩

/-- Theorem stating that the bacterial culture will be eradicated in n minutes -/
theorem bacterial_eradication (n : ℕ) (h : n > 0) : 
  (evolve n (n - 1)).nonInfected = 0 ∧ (evolve n n).infected = 0 := by
  sorry


end NUMINAMATH_CALUDE_bacterial_eradication_l2794_279458


namespace NUMINAMATH_CALUDE_total_spectators_l2794_279462

theorem total_spectators (men : ℕ) (children : ℕ) (women : ℕ) 
  (h1 : men = 7000)
  (h2 : children = 2500)
  (h3 : children = 5 * women) :
  men + children + women = 10000 := by
  sorry

end NUMINAMATH_CALUDE_total_spectators_l2794_279462


namespace NUMINAMATH_CALUDE_cryptarithm_multiplication_l2794_279477

theorem cryptarithm_multiplication :
  ∃! n : ℕ, ∃ m : ℕ,
    100 ≤ n ∧ n < 1000 ∧
    10000 ≤ m ∧ m < 100000 ∧
    n * n = m ∧
    ∃ k : ℕ, 100 ≤ k ∧ k < 1000 ∧ m = k * 1000 + k :=
by sorry

end NUMINAMATH_CALUDE_cryptarithm_multiplication_l2794_279477


namespace NUMINAMATH_CALUDE_negation_equivalence_l2794_279406

theorem negation_equivalence : 
  (¬ ∀ x : ℝ, x^2 + Real.sin x + 1 < 0) ↔ (∃ x : ℝ, x^2 + Real.sin x + 1 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2794_279406


namespace NUMINAMATH_CALUDE_greatest_five_digit_with_product_120_sum_18_l2794_279473

/-- Represents a five-digit number -/
def FiveDigitNumber := Fin 100000

/-- Returns true if the number is five digits -/
def isFiveDigit (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

/-- Returns the product of digits of a natural number -/
def digitProduct (n : ℕ) : ℕ := sorry

/-- Returns the sum of digits of a natural number -/
def digitSum (n : ℕ) : ℕ := sorry

/-- The greatest five-digit number whose digits have a product of 120 -/
def N : FiveDigitNumber := sorry

theorem greatest_five_digit_with_product_120_sum_18 :
  isFiveDigit N.val ∧ 
  digitProduct N.val = 120 ∧ 
  (∀ m : FiveDigitNumber, digitProduct m.val = 120 → m.val ≤ N.val) →
  digitSum N.val = 18 := by sorry

end NUMINAMATH_CALUDE_greatest_five_digit_with_product_120_sum_18_l2794_279473


namespace NUMINAMATH_CALUDE_order_of_cube_roots_l2794_279485

theorem order_of_cube_roots (a : ℝ) (x y z : ℝ) 
  (hx : x = (1 + 991 * a) ^ (1/3))
  (hy : y = (1 + 992 * a) ^ (1/3))
  (hz : z = (1 + 993 * a) ^ (1/3))
  (ha : a ≤ 0) : 
  z ≤ y ∧ y ≤ x := by
sorry

end NUMINAMATH_CALUDE_order_of_cube_roots_l2794_279485


namespace NUMINAMATH_CALUDE_people_in_room_l2794_279465

/-- Proves that given the conditions in the problem, the number of people in the room is 67 -/
theorem people_in_room (chairs : ℕ) (people : ℕ) : 
  (3 : ℚ) / 5 * people = (4 : ℚ) / 5 * chairs →  -- Three-fifths of people are seated in four-fifths of chairs
  chairs - (4 : ℚ) / 5 * chairs = 10 →           -- 10 chairs are empty
  people = 67 := by
sorry


end NUMINAMATH_CALUDE_people_in_room_l2794_279465


namespace NUMINAMATH_CALUDE_possible_values_of_a_l2794_279426

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 4 = 0}
def B (a : ℝ) : Set ℝ := {x | a * x - 2 = 0}

-- State the theorem
theorem possible_values_of_a (a : ℝ) : B a ⊆ A → a ∈ ({1, -1, 0} : Set ℝ) := by
  sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l2794_279426


namespace NUMINAMATH_CALUDE_arithmetic_sequence_seventh_term_l2794_279411

/-- An arithmetic sequence with given properties -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_seventh_term
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_first : a 1 = 2)
  (h_sum : a 3 + a 5 = 10) :
  a 7 = 8 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_seventh_term_l2794_279411


namespace NUMINAMATH_CALUDE_cube_sum_eq_triple_product_l2794_279490

theorem cube_sum_eq_triple_product (a b c : ℝ) (h : a + b + c = 0) :
  a^3 + b^3 + c^3 = 3*a*b*c := by sorry

end NUMINAMATH_CALUDE_cube_sum_eq_triple_product_l2794_279490


namespace NUMINAMATH_CALUDE_joan_apples_l2794_279491

/-- The number of apples Joan has after picking and giving some away -/
def apples_remaining (picked : ℕ) (given_away : ℕ) : ℕ :=
  picked - given_away

/-- Theorem: Joan has 16 apples after picking 43 and giving away 27 -/
theorem joan_apples : apples_remaining 43 27 = 16 := by
  sorry

end NUMINAMATH_CALUDE_joan_apples_l2794_279491


namespace NUMINAMATH_CALUDE_emiliano_consumption_theorem_l2794_279489

/-- Represents the number of fruits in a basket -/
structure FruitBasket where
  apples : ℕ
  oranges : ℕ
  bananas : ℕ

/-- Calculates the number of fruits Emiliano consumes -/
def emilianoConsumption (basket : FruitBasket) : ℕ :=
  (3 * basket.apples / 5) + (2 * basket.oranges / 3) + (4 * basket.bananas / 7)

/-- Theorem: Given the conditions, Emiliano consumes 16 fruits -/
theorem emiliano_consumption_theorem (basket : FruitBasket) 
  (h1 : basket.apples = 15)
  (h2 : basket.apples = 4 * basket.oranges)
  (h3 : basket.bananas = 3 * basket.oranges) :
  emilianoConsumption basket = 16 := by
  sorry


end NUMINAMATH_CALUDE_emiliano_consumption_theorem_l2794_279489


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l2794_279417

def U : Set Nat := {1, 2, 3, 4, 5}
def M : Set Nat := {1, 2, 3}
def N : Set Nat := {3, 4, 5}

theorem intersection_complement_equality : M ∩ (U \ N) = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l2794_279417


namespace NUMINAMATH_CALUDE_fraction_equality_implies_four_l2794_279436

theorem fraction_equality_implies_four (k n m : ℕ+) :
  (1 : ℚ) / n^2 + (1 : ℚ) / m^2 = (k : ℚ) / (n^2 + m^2) →
  k = 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_four_l2794_279436


namespace NUMINAMATH_CALUDE_solution_l2794_279459

def problem (B : ℕ) (A : ℕ) (X : ℕ) : Prop :=
  B = 38 ∧ 
  A = B + 8 ∧ 
  A + 10 = 2 * (B - X)

theorem solution : ∃ X, problem 38 (38 + 8) X ∧ X = 10 := by
  sorry

end NUMINAMATH_CALUDE_solution_l2794_279459


namespace NUMINAMATH_CALUDE_one_ounce_in_gallons_l2794_279432

/-- The number of ounces in one gallon of water -/
def ounces_per_gallon : ℚ := 128

/-- The number of ounces Jimmy drinks each time -/
def ounces_per_serving : ℚ := 8

/-- The number of times Jimmy drinks water per day -/
def servings_per_day : ℚ := 8

/-- The number of gallons Jimmy prepares for 5 days -/
def gallons_for_five_days : ℚ := 5/2

/-- The number of days Jimmy prepares water for -/
def days_prepared : ℚ := 5

/-- Theorem stating that 1 ounce of water is equal to 1/128 gallons -/
theorem one_ounce_in_gallons :
  1 / ounces_per_gallon = 
    gallons_for_five_days / (ounces_per_serving * servings_per_day * days_prepared) :=
by sorry

end NUMINAMATH_CALUDE_one_ounce_in_gallons_l2794_279432


namespace NUMINAMATH_CALUDE_complement_A_inter_B_a_range_l2794_279476

open Set

-- Define the sets A and B
def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 6}
def B : Set ℝ := {x | x ≥ 3}

-- Define the complement of the intersection of A and B
def complement_intersection : Set ℝ := {x | x < 3 ∨ x > 6}

-- Define set C
def C (a : ℝ) : Set ℝ := {x | x ≤ a}

-- Theorem 1: The complement of A ∩ B is equal to the defined complement_intersection
theorem complement_A_inter_B : (A ∩ B)ᶜ = complement_intersection := by sorry

-- Theorem 2: If A is a subset of C, then a is greater than or equal to 6
theorem a_range (a : ℝ) (h : A ⊆ C a) : a ≥ 6 := by sorry

end NUMINAMATH_CALUDE_complement_A_inter_B_a_range_l2794_279476


namespace NUMINAMATH_CALUDE_n_pointed_star_degree_sum_l2794_279408

/-- An n-pointed star formed from a convex polygon -/
structure NPointedStar where
  n : ℕ
  h_n_ge_7 : n ≥ 7

/-- The degree sum of interior angles of an n-pointed star -/
def degree_sum (star : NPointedStar) : ℝ :=
  180 * (star.n - 2)

/-- Theorem: The degree sum of interior angles of an n-pointed star is 180(n-2) -/
theorem n_pointed_star_degree_sum (star : NPointedStar) :
  degree_sum star = 180 * (star.n - 2) := by
  sorry

end NUMINAMATH_CALUDE_n_pointed_star_degree_sum_l2794_279408


namespace NUMINAMATH_CALUDE_gcd_power_three_l2794_279498

theorem gcd_power_three : Nat.gcd (3^600 - 1) (3^612 - 1) = 3^12 - 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_power_three_l2794_279498


namespace NUMINAMATH_CALUDE_P_necessary_not_sufficient_for_Q_l2794_279463

theorem P_necessary_not_sufficient_for_Q :
  (∀ x : ℝ, (x + 2) * (x - 1) < 0 → x < 1) ∧
  (∃ x : ℝ, x < 1 ∧ (x + 2) * (x - 1) ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_P_necessary_not_sufficient_for_Q_l2794_279463


namespace NUMINAMATH_CALUDE_constant_point_of_quadratic_l2794_279487

/-- The quadratic function f(x) that depends on a parameter m -/
def f (m : ℝ) (x : ℝ) : ℝ := 3 * x^2 - m * x + 2 * m + 1

/-- The theorem stating that (2, 13) is the unique constant point for f(x) -/
theorem constant_point_of_quadratic :
  ∃! p : ℝ × ℝ, ∀ m : ℝ, f m p.1 = p.2 ∧ p = (2, 13) :=
sorry

end NUMINAMATH_CALUDE_constant_point_of_quadratic_l2794_279487


namespace NUMINAMATH_CALUDE_gcd_143_100_l2794_279499

theorem gcd_143_100 : Nat.gcd 143 100 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_143_100_l2794_279499


namespace NUMINAMATH_CALUDE_puzzle_solution_l2794_279446

theorem puzzle_solution (a b c d : ℕ+) 
  (h1 : a^6 = b^5) 
  (h2 : c^4 = d^3) 
  (h3 : c - a = 25) : 
  d - b = 561 := by
  sorry

end NUMINAMATH_CALUDE_puzzle_solution_l2794_279446


namespace NUMINAMATH_CALUDE_chairs_to_remove_l2794_279407

theorem chairs_to_remove (initial_chairs : ℕ) (chairs_per_row : ℕ) (expected_students : ℕ)
  (h1 : initial_chairs = 156)
  (h2 : chairs_per_row = 13)
  (h3 : expected_students = 95)
  (h4 : initial_chairs % chairs_per_row = 0) -- All rows are initially completely filled
  : ∃ (removed_chairs : ℕ),
    removed_chairs = 52 ∧
    (initial_chairs - removed_chairs) % chairs_per_row = 0 ∧ -- Remaining rows are completely filled
    (initial_chairs - removed_chairs) ≥ expected_students ∧ -- Can accommodate all students
    ∀ (x : ℕ), x < removed_chairs →
      (initial_chairs - x < expected_students ∨ (initial_chairs - x) % chairs_per_row ≠ 0) -- Minimizes empty seats
    := by sorry

end NUMINAMATH_CALUDE_chairs_to_remove_l2794_279407


namespace NUMINAMATH_CALUDE_fixed_point_theorem_l2794_279457

-- Define the line equation
def line_equation (k x y : ℝ) : Prop := k * x + y - 2 = 3 * k

-- State the theorem
theorem fixed_point_theorem :
  ∀ k : ℝ, line_equation k 3 2 := by sorry

end NUMINAMATH_CALUDE_fixed_point_theorem_l2794_279457


namespace NUMINAMATH_CALUDE_soccer_practice_probability_l2794_279431

theorem soccer_practice_probability (p : ℚ) (h : p = 5/8) :
  1 - p = 3/8 := by sorry

end NUMINAMATH_CALUDE_soccer_practice_probability_l2794_279431


namespace NUMINAMATH_CALUDE_potatoes_for_dinner_l2794_279420

def potatoes_for_lunch : ℕ := 5
def total_potatoes : ℕ := 7

theorem potatoes_for_dinner : total_potatoes - potatoes_for_lunch = 2 := by
  sorry

end NUMINAMATH_CALUDE_potatoes_for_dinner_l2794_279420


namespace NUMINAMATH_CALUDE_half_day_percentage_l2794_279454

def total_students : ℕ := 80
def full_day_students : ℕ := 60

theorem half_day_percentage :
  (total_students - full_day_students) / total_students * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_half_day_percentage_l2794_279454


namespace NUMINAMATH_CALUDE_x_difference_is_fourteen_l2794_279403

theorem x_difference_is_fourteen (x : ℝ) :
  (x + 3)^2 / (3*x + 29) = 2 → ∃ y : ℝ, (y + 3)^2 / (3*y + 29) = 2 ∧ |x - y| = 14 :=
by sorry

end NUMINAMATH_CALUDE_x_difference_is_fourteen_l2794_279403


namespace NUMINAMATH_CALUDE_max_value_implies_a_values_exactly_two_a_values_l2794_279467

/-- The function f for a given real number a -/
def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + 2*a*x + 1 - a

/-- The theorem stating the possible values of a -/
theorem max_value_implies_a_values (a : ℝ) : 
  (∀ x ∈ Set.Icc 0 1, f a x ≤ 2) ∧ 
  (∃ x ∈ Set.Icc 0 1, f a x = 2) → 
  a = -1 ∨ a = 2 := by
sorry

/-- The main theorem stating that there are exactly two possible values for a -/
theorem exactly_two_a_values : 
  ∃! s : Set ℝ, s = {-1, 2} ∧ 
  ∀ a : ℝ, (∀ x ∈ Set.Icc 0 1, f a x ≤ 2) ∧ 
           (∃ x ∈ Set.Icc 0 1, f a x = 2) → 
           a ∈ s := by
sorry

end NUMINAMATH_CALUDE_max_value_implies_a_values_exactly_two_a_values_l2794_279467


namespace NUMINAMATH_CALUDE_unknown_number_proof_l2794_279433

theorem unknown_number_proof (x : ℝ) : 
  (0.15 * 25 + 0.12 * x = 9.15) → x = 45 :=
by sorry

end NUMINAMATH_CALUDE_unknown_number_proof_l2794_279433


namespace NUMINAMATH_CALUDE_exponent_manipulation_l2794_279442

theorem exponent_manipulation (x y : ℝ) :
  (x - y)^4 * (y - x)^3 / (y - x)^2 = (x - y)^5 :=
by sorry

end NUMINAMATH_CALUDE_exponent_manipulation_l2794_279442


namespace NUMINAMATH_CALUDE_card_distribution_exists_iff_odd_l2794_279437

/-- A magic pair is a pair of consecutive numbers or the pair (1, n(n-1)/2) -/
def is_magic_pair (a b : Nat) (n : Nat) : Prop :=
  (a + 1 = b ∨ b + 1 = a) ∨ (a = 1 ∧ b = n * (n - 1) / 2) ∨ (b = 1 ∧ a = n * (n - 1) / 2)

/-- A valid distribution of cards into stacks -/
def valid_distribution (n : Nat) (stacks : Fin n → Finset Nat) : Prop :=
  (∀ i : Fin n, ∀ x ∈ stacks i, x ≤ n * (n - 1) / 2) ∧
  (∀ i j : Fin n, i ≠ j → ∃! (a b : Nat), a ∈ stacks i ∧ b ∈ stacks j ∧ is_magic_pair a b n)

theorem card_distribution_exists_iff_odd (n : Nat) (h : n > 2) :
  (∃ stacks : Fin n → Finset Nat, valid_distribution n stacks) ↔ Odd n :=
sorry

end NUMINAMATH_CALUDE_card_distribution_exists_iff_odd_l2794_279437


namespace NUMINAMATH_CALUDE_max_intersections_eight_l2794_279434

/-- Represents a tiled floor with equilateral triangles -/
structure TriangularFloor where
  side_length : ℝ
  side_length_positive : side_length > 0

/-- Represents a needle -/
structure Needle where
  length : ℝ
  length_positive : length > 0

/-- Counts the maximum number of triangles intersected by a needle -/
def max_intersected_triangles (floor : TriangularFloor) (needle : Needle) : ℕ :=
  sorry

/-- Theorem stating the maximum number of intersected triangles -/
theorem max_intersections_eight
  (floor : TriangularFloor)
  (needle : Needle)
  (h_floor : floor.side_length = 1)
  (h_needle : needle.length = 2) :
  max_intersected_triangles floor needle = 8 :=
sorry

end NUMINAMATH_CALUDE_max_intersections_eight_l2794_279434


namespace NUMINAMATH_CALUDE_green_blue_difference_l2794_279480

/-- Represents the number of disks of each color -/
structure DiskCounts where
  blue : ℕ
  yellow : ℕ
  green : ℕ
  red : ℕ
  purple : ℕ

/-- The ratio of disks for each color -/
def diskRatio : DiskCounts := {
  blue := 3,
  yellow := 7,
  green := 8,
  red := 4,
  purple := 5
}

/-- The total number of disks in the bag -/
def totalDisks : ℕ := 360

/-- Calculates the total ratio parts -/
def totalRatioParts (ratio : DiskCounts) : ℕ :=
  ratio.blue + ratio.yellow + ratio.green + ratio.red + ratio.purple

/-- Calculates the number of disks for each ratio part -/
def disksPerPart (total : ℕ) (ratioParts : ℕ) : ℕ :=
  total / ratioParts

/-- Calculates the actual disk counts based on the ratio and total disks -/
def actualDiskCounts (ratio : DiskCounts) (total : ℕ) : DiskCounts :=
  let parts := totalRatioParts ratio
  let perPart := disksPerPart total parts
  {
    blue := ratio.blue * perPart,
    yellow := ratio.yellow * perPart,
    green := ratio.green * perPart,
    red := ratio.red * perPart,
    purple := ratio.purple * perPart
  }

theorem green_blue_difference :
  let counts := actualDiskCounts diskRatio totalDisks
  counts.green - counts.blue = 65 := by sorry

end NUMINAMATH_CALUDE_green_blue_difference_l2794_279480


namespace NUMINAMATH_CALUDE_vector_colinearity_l2794_279450

def a : ℝ × ℝ := (1, 3)
def b : ℝ × ℝ := (-2, -1)
def c : ℝ × ℝ := (2, 4)

def colinear (v w : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, v.1 * w.2 = t * v.2 * w.1

theorem vector_colinearity (k : ℝ) :
  colinear (a.1 + k * b.1, a.2 + k * b.2) c →
  k = -1/3 :=
by sorry

end NUMINAMATH_CALUDE_vector_colinearity_l2794_279450


namespace NUMINAMATH_CALUDE_tournament_games_theorem_l2794_279474

/-- A single-elimination tournament with no ties. -/
structure Tournament :=
  (num_teams : ℕ)
  (no_ties : Bool)

/-- The number of games needed to declare a winner in a single-elimination tournament. -/
def games_to_winner (t : Tournament) : ℕ :=
  t.num_teams - 1

/-- Theorem: In a single-elimination tournament with 23 teams and no ties, 
    the number of games played to declare a winner is 22. -/
theorem tournament_games_theorem (t : Tournament) 
  (h1 : t.num_teams = 23) 
  (h2 : t.no_ties = true) : 
  games_to_winner t = 22 := by
  sorry


end NUMINAMATH_CALUDE_tournament_games_theorem_l2794_279474


namespace NUMINAMATH_CALUDE_least_integer_with_1323_divisors_l2794_279472

/-- The number of distinct positive divisors of n -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- Checks if n can be expressed as m * 30^k where 30 is not a divisor of m -/
def is_valid_form (n m k : ℕ) : Prop :=
  n = m * (30 ^ k) ∧ ¬(30 ∣ m)

theorem least_integer_with_1323_divisors :
  ∃ (n m k : ℕ),
    (∀ i < n, num_divisors i ≠ 1323) ∧
    num_divisors n = 1323 ∧
    is_valid_form n m k ∧
    m + k = 83 :=
sorry

end NUMINAMATH_CALUDE_least_integer_with_1323_divisors_l2794_279472


namespace NUMINAMATH_CALUDE_union_equal_iff_a_geq_one_l2794_279471

/-- The set A defined as {x | 2 ≤ x ≤ 6} -/
def A : Set ℝ := {x : ℝ | 2 ≤ x ∧ x ≤ 6}

/-- The set B defined as {x | 2a ≤ x ≤ a+3} -/
def B (a : ℝ) : Set ℝ := {x : ℝ | 2*a ≤ x ∧ x ≤ a+3}

/-- Theorem stating that A ∪ B = A if and only if a ≥ 1 -/
theorem union_equal_iff_a_geq_one (a : ℝ) : A ∪ B a = A ↔ a ≥ 1 := by sorry

end NUMINAMATH_CALUDE_union_equal_iff_a_geq_one_l2794_279471


namespace NUMINAMATH_CALUDE_greatest_three_digit_divisible_by_4_and_9_l2794_279404

theorem greatest_three_digit_divisible_by_4_and_9 :
  ∃ n : ℕ, n = 972 ∧ 
  100 ≤ n ∧ n ≤ 999 ∧ 
  n % 4 = 0 ∧ n % 9 = 0 ∧
  ∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 ∧ m % 4 = 0 ∧ m % 9 = 0 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_three_digit_divisible_by_4_and_9_l2794_279404


namespace NUMINAMATH_CALUDE_adult_ticket_price_l2794_279423

theorem adult_ticket_price 
  (total_tickets : ℕ) 
  (total_profit : ℕ) 
  (kid_tickets : ℕ) 
  (kid_price : ℕ) :
  total_tickets = 175 →
  total_profit = 750 →
  kid_tickets = 75 →
  kid_price = 2 →
  (total_tickets - kid_tickets) * 
    ((total_profit - kid_tickets * kid_price) / (total_tickets - kid_tickets)) = 600 :=
by sorry

end NUMINAMATH_CALUDE_adult_ticket_price_l2794_279423


namespace NUMINAMATH_CALUDE_complex_modulus_l2794_279495

theorem complex_modulus (z : ℂ) (h : z + Complex.I = 3) : Complex.abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l2794_279495


namespace NUMINAMATH_CALUDE_max_value_fraction_l2794_279486

theorem max_value_fraction (x : ℝ) :
  (4 * x^2 + 8 * x + 19) / (4 * x^2 + 8 * x + 5) ≤ 15 ∧
  ∃ y : ℝ, (4 * y^2 + 8 * y + 19) / (4 * y^2 + 8 * y + 5) = 15 := by
  sorry

end NUMINAMATH_CALUDE_max_value_fraction_l2794_279486


namespace NUMINAMATH_CALUDE_perpendicular_unit_vectors_l2794_279451

def vector_a : ℝ × ℝ := (2, -2)

def is_unit_vector (v : ℝ × ℝ) : Prop :=
  v.1 ^ 2 + v.2 ^ 2 = 1

def is_perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

theorem perpendicular_unit_vectors :
  ∀ v : ℝ × ℝ, is_unit_vector v ∧ is_perpendicular v vector_a →
    v = (Real.sqrt 2 / 2, Real.sqrt 2 / 2) ∨ v = (-Real.sqrt 2 / 2, -Real.sqrt 2 / 2) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_unit_vectors_l2794_279451


namespace NUMINAMATH_CALUDE_electronics_store_theorem_l2794_279427

theorem electronics_store_theorem (total : ℕ) (tv : ℕ) (computer : ℕ) (both : ℕ) 
  (h1 : total = 15)
  (h2 : tv = 9)
  (h3 : computer = 7)
  (h4 : both = 3)
  : total - (tv + computer - both) = 2 :=
by sorry

end NUMINAMATH_CALUDE_electronics_store_theorem_l2794_279427


namespace NUMINAMATH_CALUDE_no_integer_solution_for_z_l2794_279444

theorem no_integer_solution_for_z :
  ¬ ∃ (z : ℤ), (2 : ℚ) / z = 2 / (z + 1) + 2 / (z + 25) :=
by sorry

end NUMINAMATH_CALUDE_no_integer_solution_for_z_l2794_279444


namespace NUMINAMATH_CALUDE_constant_sum_reciprocal_distances_l2794_279424

noncomputable section

-- Define the hyperbola C
def hyperbola (x y a b : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the focal length
def focal_length (c : ℝ) : Prop := c = 3 * Real.sqrt 2

-- Define one asymptote of C
def asymptote (x y : ℝ) : Prop := x - Real.sqrt 2 * y = 0

-- Define the ellipse E
def ellipse (x y : ℝ) : Prop := x^2 / 3 + 2 * y^2 / 3 = 1

-- Define point P as the left vertex of E
def P : ℝ × ℝ := (-Real.sqrt 3, 0)

-- Define point G
def G : ℝ × ℝ := (-Real.sqrt 3 / 3, 0)

-- Define the theorem
theorem constant_sum_reciprocal_distances 
  (O A B : ℝ × ℝ) 
  (hE : ellipse A.1 A.2 ∧ ellipse B.1 B.2) 
  (hP : P.1^2 + P.2^2 = (A.1 - P.1)^2 + (A.2 - P.2)^2) :
  1 / (A.1^2 + A.2^2) + 1 / (B.1^2 + B.2^2) + 2 / (P.1^2 + P.2^2) = 2 :=
sorry

end NUMINAMATH_CALUDE_constant_sum_reciprocal_distances_l2794_279424


namespace NUMINAMATH_CALUDE_associated_points_theorem_l2794_279428

/-- Definition of k times associated point -/
def k_times_associated_point (P M : ℝ × ℝ) (k : ℤ) : Prop :=
  let d_PM := Real.sqrt ((M.1 - P.1)^2 + (M.2 - P.2)^2)
  let d_PO := Real.sqrt (P.1^2 + P.2^2)
  d_PM = k * d_PO

/-- Main theorem -/
theorem associated_points_theorem :
  let P₁ : ℝ × ℝ := (-1.5, 0)
  let P₂ : ℝ × ℝ := (-1, 0)
  ∀ (b : ℝ),
  (∃ (M : ℝ × ℝ), k_times_associated_point P₁ M 2 ∧ M.2 = 0 →
    (M = (1.5, 0) ∨ M = (-4.5, 0))) ∧
  (∀ (M : ℝ × ℝ) (k : ℤ),
    k_times_associated_point P₁ M k ∧ M.1 = -1.5 ∧ -3 ≤ M.2 ∧ M.2 ≤ 5 →
    k ≤ 3) ∧
  (∃ (A B C : ℝ × ℝ),
    A = (b, 0) ∧ B = (b + 1, 0) ∧
    Real.sqrt ((C.1 - A.1)^2 + C.2^2) = Real.sqrt ((B.1 - A.1)^2 + (C.2 - B.2)^2) ∧
    C.2 / (C.1 - A.1) = Real.sqrt 3 / 3 →
    (∃ (Q : ℝ × ℝ), k_times_associated_point P₂ Q 2 ∧
      (∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ Q = (A.1 + t * (B.1 - A.1), A.2 + t * (B.2 - A.2))) →
      (-4 ≤ b ∧ b ≤ -3) ∨ (-1 ≤ b ∧ b ≤ 1))) :=
by
  sorry

end NUMINAMATH_CALUDE_associated_points_theorem_l2794_279428


namespace NUMINAMATH_CALUDE_not_sufficient_not_necessary_l2794_279410

/-- Two lines ax + by + c = 0 and dx + ey + f = 0 are perpendicular if and only if ad + be = 0 -/
def are_perpendicular (a b c d e f : ℝ) : Prop :=
  a * d + b * e = 0

/-- The proposition "a = 2" is neither sufficient nor necessary for the line ax + 3y - 1 = 0
    to be perpendicular to the line 6x + 4y - 3 = 0 -/
theorem not_sufficient_not_necessary : 
  (∃ a : ℝ, a = 2 ∧ ¬(are_perpendicular a 3 (-1) 6 4 (-3))) ∧ 
  (∃ a : ℝ, are_perpendicular a 3 (-1) 6 4 (-3) ∧ a ≠ 2) :=
sorry

end NUMINAMATH_CALUDE_not_sufficient_not_necessary_l2794_279410


namespace NUMINAMATH_CALUDE_equation_solution_l2794_279468

theorem equation_solution (x : ℝ) : 
  (∀ y : ℝ, 12 * x * y - 18 * y + 3 * x - 9 / 2 = 0) ↔ x = 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2794_279468


namespace NUMINAMATH_CALUDE_expected_girls_left_10_7_l2794_279466

/-- The expected number of girls standing to the left of all boys in a random lineup -/
def expected_girls_left (num_boys num_girls : ℕ) : ℚ :=
  (num_girls : ℚ) / (num_boys + num_girls + 1 : ℚ)

/-- Theorem: In a random lineup of 10 boys and 7 girls, the expected number of girls 
    standing to the left of all boys is 7/11 -/
theorem expected_girls_left_10_7 :
  expected_girls_left 10 7 = 7 / 11 := by
  sorry

end NUMINAMATH_CALUDE_expected_girls_left_10_7_l2794_279466


namespace NUMINAMATH_CALUDE_historian_writing_speed_l2794_279470

/-- Given a historian who wrote 60,000 words in 150 hours,
    prove that the average number of words written per hour is 400. -/
theorem historian_writing_speed :
  let total_words : ℕ := 60000
  let total_hours : ℕ := 150
  let average_words_per_hour : ℚ := total_words / total_hours
  average_words_per_hour = 400 := by
  sorry

end NUMINAMATH_CALUDE_historian_writing_speed_l2794_279470


namespace NUMINAMATH_CALUDE_container_water_problem_l2794_279430

theorem container_water_problem (x y : ℝ) : 
  x > 0 ∧ y > 0 → -- Containers and total masses are positive
  (4 / 5 * y - x) + (y - x) = 8 * x → -- Pouring water from B to A
  y - x - (4 / 5 * y - x) = 50 → -- B has 50g more water than A
  x = 50 ∧ 4 / 5 * y - x = 150 ∧ y - x = 200 := by
sorry

end NUMINAMATH_CALUDE_container_water_problem_l2794_279430


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2794_279483

/-- Given an arithmetic sequence where:
    - n is a positive integer
    - The sum of the first n terms is 48
    - The sum of the first 2n terms is 60
    This theorem states that the sum of the first 3n terms is 36 -/
theorem arithmetic_sequence_sum (n : ℕ+) 
  (sum_n : ℕ) (sum_2n : ℕ) (h1 : sum_n = 48) (h2 : sum_2n = 60) :
  ∃ (sum_3n : ℕ), sum_3n = 36 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2794_279483


namespace NUMINAMATH_CALUDE_cubic_identity_l2794_279405

theorem cubic_identity (x : ℝ) : 
  (2*x - 1)^3 = 5*x^3 + (3*x + 1)*(x^2 - x - 1) - 10*x^2 + 10*x := by
  sorry

end NUMINAMATH_CALUDE_cubic_identity_l2794_279405


namespace NUMINAMATH_CALUDE_gcd_204_85_l2794_279416

theorem gcd_204_85 : Nat.gcd 204 85 = 57 := by
  sorry

end NUMINAMATH_CALUDE_gcd_204_85_l2794_279416


namespace NUMINAMATH_CALUDE_red_blue_difference_after_border_l2794_279479

/-- Represents a hexagonal figure with blue and red tiles -/
structure HexFigure where
  blue_tiles : ℕ
  red_tiles : ℕ

/-- Adds a border to a hexagonal figure, alternating between blue and red tiles -/
def add_border (fig : HexFigure) : HexFigure :=
  { blue_tiles := fig.blue_tiles + 12,
    red_tiles := fig.red_tiles + 12 }

/-- The initial hexagonal figure -/
def initial_figure : HexFigure :=
  { blue_tiles := 10,
    red_tiles := 20 }

theorem red_blue_difference_after_border :
  (add_border initial_figure).red_tiles - (add_border initial_figure).blue_tiles = 10 := by
  sorry

end NUMINAMATH_CALUDE_red_blue_difference_after_border_l2794_279479


namespace NUMINAMATH_CALUDE_hunting_company_composition_l2794_279421

theorem hunting_company_composition :
  ∃ (foxes wolves bears : ℕ),
    foxes + wolves + bears = 45 ∧
    59 * foxes + 41 * wolves + 40 * bears = 2008 ∧
    foxes = 10 ∧ wolves = 18 ∧ bears = 17 := by
  sorry

end NUMINAMATH_CALUDE_hunting_company_composition_l2794_279421


namespace NUMINAMATH_CALUDE_train_fraction_is_four_fifths_l2794_279402

def journey (D : ℝ) (train_speed car_speed avg_speed : ℝ) (x : ℝ) : Prop :=
  D > 0 ∧ 
  train_speed = 80 ∧ 
  car_speed = 20 ∧ 
  avg_speed = 50 ∧ 
  0 ≤ x ∧ 
  x ≤ 1 ∧
  D / ((x * D / train_speed) + ((1 - x) * D / car_speed)) = avg_speed

theorem train_fraction_is_four_fifths (D : ℝ) (train_speed car_speed avg_speed x : ℝ) :
  journey D train_speed car_speed avg_speed x → x = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_train_fraction_is_four_fifths_l2794_279402


namespace NUMINAMATH_CALUDE_yoongi_multiplication_l2794_279418

theorem yoongi_multiplication (n : ℚ) : n * 15 = 45 → n - 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_yoongi_multiplication_l2794_279418


namespace NUMINAMATH_CALUDE_sum_of_coefficients_quadratic_l2794_279449

theorem sum_of_coefficients_quadratic (x : ℝ) : 
  (∃ a b c : ℝ, a * x^2 + b * x + c = 0 ∧ x * (x + 1) = 4) → 
  (∃ a b c : ℝ, a * x^2 + b * x + c = 0 ∧ x * (x + 1) = 4 ∧ a + b + c = -2) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_quadratic_l2794_279449


namespace NUMINAMATH_CALUDE_expression_evaluation_l2794_279413

theorem expression_evaluation :
  let m : ℚ := -1/2
  let f (x : ℚ) := (5 / (x - 2) - x - 2) * ((2 * x - 4) / (3 - x))
  f m = 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2794_279413


namespace NUMINAMATH_CALUDE_mens_wages_l2794_279447

/-- Proves that the total wages of men is Rs. 30 given the problem conditions -/
theorem mens_wages (W : ℕ) (total_earnings : ℕ) : 
  (5 : ℕ) = W →  -- 5 men are equal to W women
  W = 8 →        -- W women are equal to 8 boys
  total_earnings = 90 →  -- Total earnings of all people is Rs. 90
  (5 : ℕ) * (total_earnings / 15) = 30 := by
sorry

end NUMINAMATH_CALUDE_mens_wages_l2794_279447


namespace NUMINAMATH_CALUDE_new_rectangle_area_l2794_279492

theorem new_rectangle_area (x y : ℝ) (h : 0 < x ∧ x ≤ y) :
  let base := Real.sqrt (x^2 + y^2) + y
  let altitude := Real.sqrt (x^2 + y^2) - y
  base * altitude = x^2 := by
  sorry

end NUMINAMATH_CALUDE_new_rectangle_area_l2794_279492


namespace NUMINAMATH_CALUDE_thirtieth_term_is_292_l2794_279464

/-- A function that checks if a natural number contains the digit 2 --/
def containsTwo (n : ℕ) : Bool :=
  sorry

/-- A function that generates the sequence of positive multiples of 4 containing at least one digit 2 --/
def sequenceOfFoursWithTwo : List ℕ :=
  sorry

/-- The 30th term of the sequence --/
def thirtiethTerm : ℕ := sorry

theorem thirtieth_term_is_292 : thirtiethTerm = 292 := by
  sorry

end NUMINAMATH_CALUDE_thirtieth_term_is_292_l2794_279464


namespace NUMINAMATH_CALUDE_num_ways_to_select_is_186_l2794_279461

def num_red_balls : ℕ := 4
def num_white_balls : ℕ := 6
def red_ball_score : ℕ := 2
def white_ball_score : ℕ := 1
def total_balls_to_take : ℕ := 5
def min_total_score : ℕ := 7

def score (red white : ℕ) : ℕ :=
  red * red_ball_score + white * white_ball_score

def valid_selection (red white : ℕ) : Prop :=
  red + white = total_balls_to_take ∧ 
  red ≤ num_red_balls ∧ 
  white ≤ num_white_balls ∧ 
  score red white ≥ min_total_score

def num_ways_to_select : ℕ := 
  (Nat.choose num_red_balls 4 * Nat.choose num_white_balls 1) +
  (Nat.choose num_red_balls 3 * Nat.choose num_white_balls 2) +
  (Nat.choose num_red_balls 2 * Nat.choose num_white_balls 3)

theorem num_ways_to_select_is_186 : num_ways_to_select = 186 := by
  sorry

end NUMINAMATH_CALUDE_num_ways_to_select_is_186_l2794_279461


namespace NUMINAMATH_CALUDE_semicircle_perimeter_l2794_279455

/-- The perimeter of a semi-circle with radius 7 cm is 7π + 14 cm. -/
theorem semicircle_perimeter : 
  ∀ (r : ℝ), r = 7 → (π * r + 2 * r) = 7 * π + 14 := by
  sorry

end NUMINAMATH_CALUDE_semicircle_perimeter_l2794_279455


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l2794_279422

theorem arithmetic_mean_problem (y b : ℝ) (h : y ≠ 0) :
  (((y + b) / y + (2 * y - b) / y) / 2) = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l2794_279422


namespace NUMINAMATH_CALUDE_village_population_l2794_279448

theorem village_population (partial_population : ℕ) (percentage : ℚ) (total_population : ℕ) : 
  percentage = 60 / 100 →
  partial_population = 23040 →
  (percentage : ℚ) * total_population = partial_population →
  total_population = 38400 := by
sorry

end NUMINAMATH_CALUDE_village_population_l2794_279448


namespace NUMINAMATH_CALUDE_product_of_squares_l2794_279412

theorem product_of_squares (x : ℝ) :
  (Real.sqrt (7 + x) + Real.sqrt (28 - x) = 9) →
  (7 + x) * (28 - x) = 529 := by
sorry

end NUMINAMATH_CALUDE_product_of_squares_l2794_279412


namespace NUMINAMATH_CALUDE_zero_is_self_opposite_l2794_279478

/-- Two real numbers are opposite if they have the same magnitude but opposite signs, or both are zero. -/
def are_opposite (a b : ℝ) : Prop := (a = -b) ∨ (a = 0 ∧ b = 0)

/-- Zero is its own opposite number. -/
theorem zero_is_self_opposite : are_opposite 0 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_is_self_opposite_l2794_279478


namespace NUMINAMATH_CALUDE_largest_multiple_of_15_less_than_500_l2794_279445

theorem largest_multiple_of_15_less_than_500 :
  ∀ n : ℕ, n * 15 < 500 → n * 15 ≤ 495 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_15_less_than_500_l2794_279445


namespace NUMINAMATH_CALUDE_sally_balloons_l2794_279440

theorem sally_balloons (initial : ℕ) (lost : ℕ) (final : ℕ) : 
  initial = 20 → lost = 5 → final = (initial - lost) * 2 → final = 30 := by
sorry

end NUMINAMATH_CALUDE_sally_balloons_l2794_279440


namespace NUMINAMATH_CALUDE_pot_holds_three_liters_l2794_279438

/-- Represents the volume of a pot in liters -/
def pot_volume (drops_per_minute : ℕ) (ml_per_drop : ℕ) (minutes_to_fill : ℕ) : ℚ :=
  (drops_per_minute * ml_per_drop * minutes_to_fill : ℚ) / 1000

/-- Theorem stating that a pot filled by a leak with given parameters holds 3 liters -/
theorem pot_holds_three_liters :
  pot_volume 3 20 50 = 3 := by
  sorry

end NUMINAMATH_CALUDE_pot_holds_three_liters_l2794_279438


namespace NUMINAMATH_CALUDE_subtraction_problem_l2794_279482

theorem subtraction_problem (x : ℤ) : 
  (x - 48 = 22) → (x - 32 = 38) := by
  sorry

end NUMINAMATH_CALUDE_subtraction_problem_l2794_279482


namespace NUMINAMATH_CALUDE_right_triangle_side_lengths_l2794_279452

theorem right_triangle_side_lengths (a : ℝ) : 
  (∃ (x y z : ℝ), x = a + 1 ∧ y = a + 2 ∧ z = a + 3 ∧ 
  x^2 + y^2 = z^2) → a = 2 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_side_lengths_l2794_279452


namespace NUMINAMATH_CALUDE_cistern_fill_time_l2794_279414

-- Define the rates of the pipes
def rateA : ℚ := 1 / 12
def rateB : ℚ := 1 / 18
def rateC : ℚ := -(1 / 15)

-- Define the combined rate
def combinedRate : ℚ := rateA + rateB + rateC

-- Define the time to fill the cistern
def timeToFill : ℚ := 1 / combinedRate

-- Theorem statement
theorem cistern_fill_time :
  timeToFill = 180 / 13 :=
sorry

end NUMINAMATH_CALUDE_cistern_fill_time_l2794_279414


namespace NUMINAMATH_CALUDE_num_syt_54321_l2794_279435

/-- A partition is a non-increasing sequence of natural numbers. -/
def Partition : Type := List Nat

/-- A Standard Young Tableau is a filling of a partition shape with integers
    such that rows and columns are strictly increasing. -/
def StandardYoungTableau (p : Partition) : Type := sorry

/-- Hook length of a cell in a partition -/
def hookLength (p : Partition) (i j : Nat) : Nat := sorry

/-- Number of Standard Young Tableaux for a given partition -/
def numSYT (p : Partition) : Nat := sorry

/-- The main theorem: number of Standard Young Tableaux for shape (5,4,3,2,1) -/
theorem num_syt_54321 :
  numSYT [5, 4, 3, 2, 1] = 292864 := by sorry

end NUMINAMATH_CALUDE_num_syt_54321_l2794_279435


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2794_279496

/-- Two vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (x - 1, 2)
  let b : ℝ × ℝ := (x, 1)
  parallel a b → x = -1 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2794_279496


namespace NUMINAMATH_CALUDE_calculation_proof_l2794_279453

theorem calculation_proof : -1^4 + (-1/2)^2 * |(-5) + 3| / (-1/2)^3 = -5 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2794_279453


namespace NUMINAMATH_CALUDE_correct_mixture_ratio_l2794_279429

/-- Represents a salt solution with a given concentration and amount -/
structure SaltSolution :=
  (concentration : ℚ)
  (amount : ℚ)

/-- Represents a mixture of two salt solutions -/
def mix (s1 s2 : SaltSolution) (r1 r2 : ℚ) : SaltSolution :=
  { concentration := (s1.concentration * r1 + s2.concentration * r2) / (r1 + r2),
    amount := r1 + r2 }

theorem correct_mixture_ratio :
  let solutionA : SaltSolution := ⟨2/5, 30⟩
  let solutionB : SaltSolution := ⟨4/5, 60⟩
  let mixedSolution := mix solutionA solutionB 3 1
  mixedSolution.concentration = 1/2 ∧ mixedSolution.amount = 50 :=
by sorry


end NUMINAMATH_CALUDE_correct_mixture_ratio_l2794_279429


namespace NUMINAMATH_CALUDE_total_bushels_is_65_l2794_279419

/-- The number of bushels needed for all animals for a day on Dany's farm -/
def total_bushels : ℕ :=
  let cow_count : ℕ := 5
  let cow_consumption : ℕ := 3
  let sheep_count : ℕ := 4
  let sheep_consumption : ℕ := 2
  let chicken_count : ℕ := 8
  let chicken_consumption : ℕ := 1
  let pig_count : ℕ := 6
  let pig_consumption : ℕ := 4
  let horse_count : ℕ := 2
  let horse_consumption : ℕ := 5
  cow_count * cow_consumption +
  sheep_count * sheep_consumption +
  chicken_count * chicken_consumption +
  pig_count * pig_consumption +
  horse_count * horse_consumption

theorem total_bushels_is_65 : total_bushels = 65 := by
  sorry

end NUMINAMATH_CALUDE_total_bushels_is_65_l2794_279419


namespace NUMINAMATH_CALUDE_rikshaw_charge_theorem_l2794_279475

/-- Represents the rikshaw charging system in Mumbai -/
structure RikshawCharge where
  base_charge : ℝ  -- Charge for the first 1 km
  rate_1_5 : ℝ     -- Rate per km for 1-5 km
  rate_5_10 : ℝ    -- Rate per 1/3 km for 5-10 km
  rate_10_plus : ℝ -- Rate per 1/3 km beyond 10 km
  wait_rate : ℝ    -- Waiting charge per hour after first 10 minutes

/-- Calculates the total charge for a rikshaw ride -/
def calculate_charge (c : RikshawCharge) (distance : ℝ) (wait_time : ℝ) : ℝ :=
  sorry

/-- The theorem stating the total charge for the given ride -/
theorem rikshaw_charge_theorem (c : RikshawCharge) 
  (h1 : c.base_charge = 18.5)
  (h2 : c.rate_1_5 = 3)
  (h3 : c.rate_5_10 = 2.5)
  (h4 : c.rate_10_plus = 4)
  (h5 : c.wait_rate = 20) :
  calculate_charge c 16 1.5 = 170 :=
sorry

end NUMINAMATH_CALUDE_rikshaw_charge_theorem_l2794_279475


namespace NUMINAMATH_CALUDE_symmetric_point_in_third_quadrant_l2794_279425

def point_symmetric_to_x_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

def in_third_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 < 0

theorem symmetric_point_in_third_quadrant :
  let p := (-2, 1)
  let symmetric_p := point_symmetric_to_x_axis p
  in_third_quadrant symmetric_p :=
by sorry

end NUMINAMATH_CALUDE_symmetric_point_in_third_quadrant_l2794_279425


namespace NUMINAMATH_CALUDE_overall_gain_percentage_l2794_279400

def item_a_cost : ℚ := 700
def item_b_cost : ℚ := 500
def item_c_cost : ℚ := 300
def item_a_gain : ℚ := 70
def item_b_gain : ℚ := 50
def item_c_gain : ℚ := 30

def total_cost : ℚ := item_a_cost + item_b_cost + item_c_cost
def total_gain : ℚ := item_a_gain + item_b_gain + item_c_gain

theorem overall_gain_percentage :
  (total_gain / total_cost) * 100 = 10 := by sorry

end NUMINAMATH_CALUDE_overall_gain_percentage_l2794_279400


namespace NUMINAMATH_CALUDE_expansion_coefficient_l2794_279460

/-- The coefficient of x^2 in the expansion of (1-ax)(1+x)^5 -/
def coefficient_x_squared (a : ℝ) : ℝ := 10 - 5*a

theorem expansion_coefficient (a : ℝ) : coefficient_x_squared a = 5 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_expansion_coefficient_l2794_279460


namespace NUMINAMATH_CALUDE_joes_notebooks_l2794_279441

theorem joes_notebooks (initial_amount : ℕ) (notebook_cost : ℕ) (book_cost : ℕ) 
  (books_bought : ℕ) (amount_left : ℕ) : 
  initial_amount = 56 → 
  notebook_cost = 4 → 
  book_cost = 7 → 
  books_bought = 2 → 
  amount_left = 14 → 
  ∃ (notebooks_bought : ℕ), 
    notebooks_bought = 7 ∧ 
    initial_amount = notebook_cost * notebooks_bought + book_cost * books_bought + amount_left :=
by sorry

end NUMINAMATH_CALUDE_joes_notebooks_l2794_279441


namespace NUMINAMATH_CALUDE_zanders_stickers_l2794_279401

theorem zanders_stickers (S : ℚ) : 
  (1/5 : ℚ) * S + (3/10 : ℚ) * (S - (1/5 : ℚ) * S) = 44 → S = 100 := by
sorry

end NUMINAMATH_CALUDE_zanders_stickers_l2794_279401


namespace NUMINAMATH_CALUDE_jessica_purchases_total_cost_l2794_279409

/-- The cost of Jessica's cat toy in dollars -/
def cat_toy_cost : ℚ := 10.22

/-- The cost of Jessica's cage in dollars -/
def cage_cost : ℚ := 11.73

/-- The total cost of Jessica's purchases in dollars -/
def total_cost : ℚ := cat_toy_cost + cage_cost

/-- Theorem stating that the total cost of Jessica's purchases is $21.95 -/
theorem jessica_purchases_total_cost : total_cost = 21.95 := by
  sorry

end NUMINAMATH_CALUDE_jessica_purchases_total_cost_l2794_279409


namespace NUMINAMATH_CALUDE_smallest_valid_m_l2794_279443

def is_valid_partition (m : ℕ) (partition : Fin 14 → Set ℕ) : Prop :=
  (∀ i, partition i ⊆ Finset.range (m + 1)) ∧
  (∀ x, x ∈ Finset.range (m + 1) → ∃ i, x ∈ partition i) ∧
  (∀ i j, i ≠ j → partition i ∩ partition j = ∅)

def has_valid_subset (m : ℕ) (partition : Fin 14 → Set ℕ) : Prop :=
  ∃ i : Fin 14, 1 < i.val ∧ i.val < 14 ∧
    ∃ a b : ℕ, a ∈ partition i ∧ b ∈ partition i ∧
      b < a ∧ (a : ℚ) ≤ 4/3 * (b : ℚ)

theorem smallest_valid_m :
  (∀ m < 56, ∃ partition : Fin 14 → Set ℕ,
    is_valid_partition m partition ∧ ¬has_valid_subset m partition) ∧
  (∀ partition : Fin 14 → Set ℕ,
    is_valid_partition 56 partition → has_valid_subset 56 partition) := by
  sorry

end NUMINAMATH_CALUDE_smallest_valid_m_l2794_279443


namespace NUMINAMATH_CALUDE_field_length_difference_l2794_279469

/-- 
Given a rectangular field with length 24 meters and width 13.5 meters,
prove that the difference between twice the width and the length is 3 meters.
-/
theorem field_length_difference (length width : ℝ) 
  (h1 : length = 24)
  (h2 : width = 13.5) :
  2 * width - length = 3 := by
  sorry

end NUMINAMATH_CALUDE_field_length_difference_l2794_279469


namespace NUMINAMATH_CALUDE_simplify_fraction_l2794_279493

theorem simplify_fraction : (180 : ℚ) / 270 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2794_279493


namespace NUMINAMATH_CALUDE_repeating_decimal_denominator_l2794_279488

theorem repeating_decimal_denominator : ∃ (n d : ℕ), d > 0 ∧ (n / d : ℚ) = 2 / 3 ∧ 
  (∀ (n' d' : ℕ), d' > 0 → (n' / d' : ℚ) = 2 / 3 → d ≤ d') := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_denominator_l2794_279488


namespace NUMINAMATH_CALUDE_range_of_a_l2794_279415

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, 4 * x^2 + 4 * (a - 2) * x + 1 ≠ 0

def q (a : ℝ) : Prop := ∀ x : ℝ, x^2 + a * x + 1 > 0

-- Define the theorem
theorem range_of_a (a : ℝ) (h : p a ∨ q a) : -2 < a ∧ a < 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2794_279415


namespace NUMINAMATH_CALUDE_largest_n_for_factorization_l2794_279497

theorem largest_n_for_factorization : 
  ∀ n : ℤ, 
  (∃ A B : ℤ, ∀ x : ℤ, 3 * x^2 + n * x + 72 = (3 * x + A) * (x + B)) →
  n ≤ 217 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_factorization_l2794_279497


namespace NUMINAMATH_CALUDE_pie_crust_flour_usage_l2794_279484

/-- Given that 30 pie crusts each use 1/6 cup of flour, and 25 new pie crusts use
    the same total amount of flour, prove that each new pie crust uses 1/5 cup of flour. -/
theorem pie_crust_flour_usage
  (original_crusts : ℕ)
  (original_flour_per_crust : ℚ)
  (new_crusts : ℕ)
  (h1 : original_crusts = 30)
  (h2 : original_flour_per_crust = 1/6)
  (h3 : new_crusts = 25)
  (h4 : original_crusts * original_flour_per_crust = new_crusts * new_flour_per_crust) :
  new_flour_per_crust = 1/5 :=
sorry

end NUMINAMATH_CALUDE_pie_crust_flour_usage_l2794_279484


namespace NUMINAMATH_CALUDE_spherical_segment_angle_l2794_279456

theorem spherical_segment_angle (r : ℝ) (α : ℝ) (h : r > 0) :
  (2 * π * r * (r * (1 - Real.cos (α / 2))) + π * (r * Real.sin (α / 2))^2 = π * r^2) →
  (Real.cos (α / 2))^2 + 2 * Real.cos (α / 2) - 2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_spherical_segment_angle_l2794_279456


namespace NUMINAMATH_CALUDE_bombardment_percentage_l2794_279481

/-- Proves that the percentage of people who died by bombardment is 5% --/
theorem bombardment_percentage (initial_population : ℕ) (final_population : ℕ) 
  (h1 : initial_population = 4675)
  (h2 : final_population = 3553) :
  ∃ (x : ℝ), x = 5 ∧ 
  (initial_population : ℝ) * ((100 - x) / 100) * 0.8 = final_population := by
  sorry

end NUMINAMATH_CALUDE_bombardment_percentage_l2794_279481


namespace NUMINAMATH_CALUDE_solve_system_of_equations_l2794_279439

theorem solve_system_of_equations (a b : ℝ) 
  (eq1 : 3 * a + 2 = 2) 
  (eq2 : 2 * b - 3 * a = 4) : 
  b = 2 := by
sorry

end NUMINAMATH_CALUDE_solve_system_of_equations_l2794_279439


namespace NUMINAMATH_CALUDE_equation_solution_l2794_279494

theorem equation_solution : ∃! x : ℝ, 4 * x + 9 * x = 430 - 10 * (x + 4) :=
  by
    use 17
    constructor
    · -- Prove that 17 satisfies the equation
      sorry
    · -- Prove that 17 is the unique solution
      sorry

end NUMINAMATH_CALUDE_equation_solution_l2794_279494
