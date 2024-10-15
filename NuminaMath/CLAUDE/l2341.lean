import Mathlib

namespace NUMINAMATH_CALUDE_gcd_count_for_product_360_l2341_234150

theorem gcd_count_for_product_360 : 
  ∃! (s : Finset ℕ), 
    (∀ d ∈ s, d > 0 ∧ ∃ a b : ℕ, a > 0 ∧ b > 0 ∧ Nat.gcd a b = d ∧ Nat.gcd a b * Nat.lcm a b = 360) ∧
    s.card = 12 := by
  sorry

end NUMINAMATH_CALUDE_gcd_count_for_product_360_l2341_234150


namespace NUMINAMATH_CALUDE_f_monotonicity_and_zero_point_l2341_234127

def f (t : ℝ) (x : ℝ) : ℝ := 4 * x^3 + 3 * t * x^2 - 6 * t^2 * x + t - 1

theorem f_monotonicity_and_zero_point :
  ∀ t : ℝ,
  (t > 0 →
    (∀ x y : ℝ, ((x < y ∧ y < -t) ∨ (t/2 < x ∧ x < y)) → f t x < f t y) ∧
    (∀ x y : ℝ, -t < x ∧ x < y ∧ y < t/2 → f t x > f t y)) ∧
  (t < 0 →
    (∀ x y : ℝ, ((x < y ∧ y < t/2) ∨ (-t < x ∧ x < y)) → f t x < f t y) ∧
    (∀ x y : ℝ, t/2 < x ∧ x < y ∧ y < -t → f t x > f t y)) ∧
  (t > 0 → ∃ x : ℝ, 0 < x ∧ x < 1 ∧ f t x = 0) :=
by sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_zero_point_l2341_234127


namespace NUMINAMATH_CALUDE_find_n_l2341_234197

def quarter_value : ℕ := 25
def nickel_value : ℕ := 5

theorem find_n : ∃ n : ℕ, 
  15 * quarter_value + 20 * nickel_value = 10 * quarter_value + n * nickel_value ∧ n = 45 := by
  sorry

end NUMINAMATH_CALUDE_find_n_l2341_234197


namespace NUMINAMATH_CALUDE_opposite_of_one_half_l2341_234194

theorem opposite_of_one_half : 
  (1 / 2 : ℚ) + (-1 / 2 : ℚ) = 0 := by sorry

end NUMINAMATH_CALUDE_opposite_of_one_half_l2341_234194


namespace NUMINAMATH_CALUDE_binomial_probabilities_l2341_234124

/-- The probability of success in a single trial -/
def p : ℝ := 0.7

/-- The number of trials -/
def n : ℕ := 5

/-- The probability of failure in a single trial -/
def q : ℝ := 1 - p

/-- Binomial probability mass function -/
def binomialPMF (k : ℕ) : ℝ :=
  (n.choose k) * p^k * q^(n-k)

/-- The probability of at most 3 successes in 5 trials -/
def probAtMost3 : ℝ :=
  binomialPMF 0 + binomialPMF 1 + binomialPMF 2 + binomialPMF 3

/-- The probability of at least 4 successes in 5 trials -/
def probAtLeast4 : ℝ :=
  binomialPMF 4 + binomialPMF 5

theorem binomial_probabilities :
  probAtMost3 = 0.4718 ∧ probAtLeast4 = 0.5282 := by
  sorry

#eval probAtMost3
#eval probAtLeast4

end NUMINAMATH_CALUDE_binomial_probabilities_l2341_234124


namespace NUMINAMATH_CALUDE_min_ratio_of_valid_partition_l2341_234147

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def is_valid_partition (partition : List ℕ × List ℕ) : Prop :=
  let (group1, group2) := partition
  (group1 ++ group2).toFinset = Finset.range 30
  ∧ (group1.prod % group2.prod = 0)

def ratio (partition : List ℕ × List ℕ) : ℚ :=
  let (group1, group2) := partition
  (group1.prod : ℚ) / (group2.prod : ℚ)

theorem min_ratio_of_valid_partition :
  ∀ partition : List ℕ × List ℕ,
    is_valid_partition partition →
    ratio partition ≥ 1077205 :=
sorry

end NUMINAMATH_CALUDE_min_ratio_of_valid_partition_l2341_234147


namespace NUMINAMATH_CALUDE_square_diagonal_point_theorem_l2341_234118

/-- Square with side length 12 -/
structure Square :=
  (side : ℝ)
  (is_twelve : side = 12)

/-- Point on the diagonal of the square -/
structure DiagonalPoint (s : Square) :=
  (x : ℝ)
  (y : ℝ)
  (on_diagonal : y = x)
  (in_square : 0 < x ∧ x < s.side)

/-- Circumcenter of a right triangle -/
def circumcenter (a b c : ℝ × ℝ) : ℝ × ℝ := sorry

/-- Angle between three points -/
def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

theorem square_diagonal_point_theorem (s : Square) (p : DiagonalPoint s)
  (o1 : ℝ × ℝ) (o2 : ℝ × ℝ)
  (h1 : o1 = circumcenter (0, 0) (s.side, 0) (p.x, p.y))
  (h2 : o2 = circumcenter (s.side, s.side) (0, s.side) (p.x, p.y))
  (h3 : angle o1 (p.x, p.y) o2 = 120)
  : ∃ (a b : ℕ), (p.x : ℝ) = Real.sqrt a + Real.sqrt b ∧ a + b = 96 := by
  sorry

end NUMINAMATH_CALUDE_square_diagonal_point_theorem_l2341_234118


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l2341_234144

theorem quadratic_distinct_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 9 = 0 ∧ y^2 + m*y + 9 = 0) ↔ 
  (m < -6 ∨ m > 6) :=
sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l2341_234144


namespace NUMINAMATH_CALUDE_pencils_per_pack_l2341_234114

theorem pencils_per_pack (num_packs : ℕ) (num_rows : ℕ) (pencils_per_row : ℕ) : 
  num_packs = 35 → num_rows = 70 → pencils_per_row = 2 → 
  (num_rows * pencils_per_row) / num_packs = 4 := by
sorry

end NUMINAMATH_CALUDE_pencils_per_pack_l2341_234114


namespace NUMINAMATH_CALUDE_largest_increase_1998_l2341_234131

def sales : ℕ → ℕ
| 0 => 3000    -- 1994
| 1 => 4500    -- 1995
| 2 => 6000    -- 1996
| 3 => 6750    -- 1997
| 4 => 8400    -- 1998
| 5 => 9000    -- 1999
| 6 => 9600    -- 2000
| 7 => 10400   -- 2001
| 8 => 9500    -- 2002
| 9 => 6500    -- 2003
| _ => 0       -- undefined for other years

def salesIncrease (year : ℕ) : ℤ :=
  (sales (year + 1) : ℤ) - (sales year : ℤ)

theorem largest_increase_1998 :
  ∀ y : ℕ, y ≥ 0 ∧ y < 9 → salesIncrease 4 ≥ salesIncrease y :=
by sorry

end NUMINAMATH_CALUDE_largest_increase_1998_l2341_234131


namespace NUMINAMATH_CALUDE_onion_sale_earnings_is_66_l2341_234155

/-- Calculates the money earned from selling onions given the initial quantities and conditions --/
def onion_sale_earnings (sally_onions fred_onions : ℕ) 
  (sara_plant_multiplier sara_harvest_multiplier : ℕ) 
  (onions_given_to_sara total_after_giving remaining_onions price_per_onion : ℕ) : ℕ :=
  let sara_planted := sara_plant_multiplier * sally_onions
  let sara_harvested := sara_harvest_multiplier * fred_onions
  let total_before_giving := sally_onions + fred_onions + sara_harvested
  let total_after_giving := total_before_giving - onions_given_to_sara
  let onions_sold := total_after_giving - remaining_onions
  onions_sold * price_per_onion

/-- Theorem stating that given the problem conditions, the earnings from selling onions is $66 --/
theorem onion_sale_earnings_is_66 : 
  onion_sale_earnings 5 9 3 2 4 24 6 3 = 66 := by
  sorry

end NUMINAMATH_CALUDE_onion_sale_earnings_is_66_l2341_234155


namespace NUMINAMATH_CALUDE_absolute_prime_at_most_three_digits_l2341_234135

/-- A function that returns true if a positive integer is prime -/
def IsPrime (n : ℕ) : Prop := sorry

/-- A function that returns the set of distinct digits in a positive integer's decimal representation -/
def DistinctDigits (n : ℕ) : Finset ℕ := sorry

/-- A function that returns true if all permutations of a positive integer's digits are prime -/
def AllDigitPermutationsPrime (n : ℕ) : Prop := sorry

/-- Definition of an absolute prime -/
def IsAbsolutePrime (n : ℕ) : Prop :=
  n > 0 ∧ IsPrime n ∧ AllDigitPermutationsPrime n

theorem absolute_prime_at_most_three_digits (n : ℕ) :
  IsAbsolutePrime n → Finset.card (DistinctDigits n) ≤ 3 := by sorry

end NUMINAMATH_CALUDE_absolute_prime_at_most_three_digits_l2341_234135


namespace NUMINAMATH_CALUDE_complement_union_theorem_l2341_234186

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 2}
def B : Set Nat := {2, 3, 4}

theorem complement_union_theorem : 
  (U \ A) ∪ B = {2, 3, 4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l2341_234186


namespace NUMINAMATH_CALUDE_centroid_maximizes_dist_product_l2341_234166

/-- A triangle in a 2D plane --/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- A point in a 2D plane --/
def Point := ℝ × ℝ

/-- Distance from a point to a line defined by two points --/
def distToLine (P : Point) (A B : Point) : ℝ := sorry

/-- The centroid of a triangle --/
def centroid (t : Triangle) : Point := sorry

/-- Product of distances from a point to the sides of a triangle --/
def distProduct (P : Point) (t : Triangle) : ℝ := 
  distToLine P t.A t.B * distToLine P t.B t.C * distToLine P t.C t.A

/-- Predicate to check if a point is inside a triangle --/
def isInside (P : Point) (t : Triangle) : Prop := sorry

theorem centroid_maximizes_dist_product (t : Triangle) :
  ∀ P, isInside P t → distProduct P t ≤ distProduct (centroid t) t :=
sorry

end NUMINAMATH_CALUDE_centroid_maximizes_dist_product_l2341_234166


namespace NUMINAMATH_CALUDE_line_points_comparison_l2341_234149

theorem line_points_comparison (m n b : ℝ) : 
  (m = -3 * (-2) + b) → (n = -3 * 3 + b) → m > n := by
  sorry

end NUMINAMATH_CALUDE_line_points_comparison_l2341_234149


namespace NUMINAMATH_CALUDE_expression_evaluation_l2341_234179

theorem expression_evaluation : (2023 - 1910 + 5)^2 / 121 = 114 + 70 / 121 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2341_234179


namespace NUMINAMATH_CALUDE_max_value_of_expression_l2341_234146

theorem max_value_of_expression (a b c d : ℝ) 
  (nonneg_a : 0 ≤ a) (nonneg_b : 0 ≤ b) (nonneg_c : 0 ≤ c) (nonneg_d : 0 ≤ d)
  (sum_condition : a + b + c + d = 200)
  (a_condition : a = 2 * d) :
  a * b + b * c + c * d ≤ 42500 / 3 := by
sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l2341_234146


namespace NUMINAMATH_CALUDE_house_cost_proof_l2341_234128

def land_acres : ℕ := 30
def land_cost_per_acre : ℕ := 20
def num_cows : ℕ := 20
def cost_per_cow : ℕ := 1000
def num_chickens : ℕ := 100
def cost_per_chicken : ℕ := 5
def solar_install_hours : ℕ := 6
def solar_install_cost_per_hour : ℕ := 100
def solar_equipment_cost : ℕ := 6000
def total_cost : ℕ := 147700

def land_cost : ℕ := land_acres * land_cost_per_acre
def cows_cost : ℕ := num_cows * cost_per_cow
def chickens_cost : ℕ := num_chickens * cost_per_chicken
def solar_install_cost : ℕ := solar_install_hours * solar_install_cost_per_hour
def total_solar_cost : ℕ := solar_install_cost + solar_equipment_cost

theorem house_cost_proof :
  total_cost - (land_cost + cows_cost + chickens_cost + total_solar_cost) = 120000 := by
  sorry

end NUMINAMATH_CALUDE_house_cost_proof_l2341_234128


namespace NUMINAMATH_CALUDE_solution_triplets_l2341_234100

theorem solution_triplets (x y z : ℝ) : 
  x^2 * y + y^2 * z = 1040 ∧
  x^2 * z + z^2 * y = 260 ∧
  (x - y) * (y - z) * (z - x) = -540 →
  (x = 16 ∧ y = 4 ∧ z = 1) ∨ (x = 1 ∧ y = 16 ∧ z = 4) :=
by sorry

end NUMINAMATH_CALUDE_solution_triplets_l2341_234100


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l2341_234138

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5, 6, 7, 8}

-- Define set A
def A : Set Nat := {2, 3, 5, 6}

-- Define set B
def B : Set Nat := {1, 3, 4, 6, 7}

-- Theorem statement
theorem intersection_A_complement_B : A ∩ (U \ B) = {2, 5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l2341_234138


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l2341_234136

theorem smallest_n_congruence : ∃! n : ℕ+, 
  (∀ m : ℕ+, 5 * m ≡ 2023 [MOD 26] → n ≤ m) ∧ 
  5 * n ≡ 2023 [MOD 26] := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l2341_234136


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2341_234182

-- Define the solution sets
def solution_set_1 : Set ℝ := {x : ℝ | (x + 3) * (x - 1) = 0}
def solution_set_2 : Set ℝ := {x : ℝ | x - 1 = 0}

-- State the theorem
theorem necessary_but_not_sufficient :
  (solution_set_2 ⊆ solution_set_1) ∧ (solution_set_2 ≠ solution_set_1) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2341_234182


namespace NUMINAMATH_CALUDE_johns_final_weight_l2341_234110

/-- Calculates the final weight after a series of weight changes --/
def final_weight (initial_weight : ℝ) : ℝ :=
  let weight1 := initial_weight * 0.9  -- 10% loss
  let weight2 := weight1 + 5           -- 5 pounds gain
  let weight3 := weight2 * 0.85        -- 15% loss
  let weight4 := weight3 + 8           -- 8 pounds gain
  weight4 * 0.8                        -- 20% loss

/-- Theorem stating that John's final weight is approximately 144.44 pounds --/
theorem johns_final_weight :
  ∃ ε > 0, |final_weight 220 - 144.44| < ε :=
sorry

end NUMINAMATH_CALUDE_johns_final_weight_l2341_234110


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l2341_234183

def P : Set ℝ := {1, 2}
def Q : Set ℝ := {x | |x| < 2}

theorem intersection_of_P_and_Q : P ∩ Q = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l2341_234183


namespace NUMINAMATH_CALUDE_cube_edge_is_nine_l2341_234117

-- Define the dimensions of the cuboid
def cuboid_base : Real := 10
def cuboid_height : Real := 73

-- Define the volume difference between the cuboid and the cube
def volume_difference : Real := 1

-- Define the function to calculate the edge length of the cube
def cube_edge_length : Real :=
  (cuboid_base * cuboid_height - volume_difference) ^ (1/3)

-- Theorem statement
theorem cube_edge_is_nine :
  cube_edge_length = 9 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_is_nine_l2341_234117


namespace NUMINAMATH_CALUDE_infinitely_many_powers_of_five_with_consecutive_zeros_l2341_234107

theorem infinitely_many_powers_of_five_with_consecutive_zeros : 
  ∀ k : ℕ, ∃ S : Set ℕ, (Set.Infinite S) ∧ 
  (∀ m ∈ S, (5^m : ℕ) ≡ 1 [MOD 2^k]) ∧
  (∃ N : ℕ, ∀ n ≥ N, ∃ m ∈ S, 
    (∃ i : ℕ, (5^m : ℕ) / 10^i % 10^1976 = 0)) :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_powers_of_five_with_consecutive_zeros_l2341_234107


namespace NUMINAMATH_CALUDE_marble_count_l2341_234158

/-- Given a bag of marbles with blue, red, and white marbles, 
    prove that the total number of marbles is 50 -/
theorem marble_count (blue red white : ℕ) (total : ℕ) 
    (h1 : blue = 5)
    (h2 : red = 9)
    (h3 : total = blue + red + white)
    (h4 : (red + white : ℚ) / total = 9/10) :
  total = 50 := by
  sorry

end NUMINAMATH_CALUDE_marble_count_l2341_234158


namespace NUMINAMATH_CALUDE_partition_into_three_exists_partition_into_four_not_exists_l2341_234176

-- Define a partition of positive integers into three sets
def PartitionIntoThree : (ℕ → Fin 3) → Prop :=
  λ f => ∀ n, n > 0 → ∃ i, f n = i

-- Define a partition of positive integers into four sets
def PartitionIntoFour : (ℕ → Fin 4) → Prop :=
  λ f => ∀ n, n > 0 → ∃ i, f n = i

-- Statement 1
theorem partition_into_three_exists :
  ∃ f : ℕ → Fin 3, PartitionIntoThree f ∧
    ∀ n ≥ 15, ∀ i : Fin 3,
      ∃ a b : ℕ, a ≠ b ∧ f a = i ∧ f b = i ∧ a + b = n :=
sorry

-- Statement 2
theorem partition_into_four_not_exists :
  ∀ f : ℕ → Fin 4, PartitionIntoFour f →
    ∃ n ≥ 15, ∃ i : Fin 4,
      ∀ a b : ℕ, a ≠ b → f a = i → f b = i → a + b ≠ n :=
sorry

end NUMINAMATH_CALUDE_partition_into_three_exists_partition_into_four_not_exists_l2341_234176


namespace NUMINAMATH_CALUDE_f_intersects_axes_l2341_234122

-- Define the function
def f (x : ℝ) : ℝ := x + 1

-- Theorem statement
theorem f_intersects_axes : 
  (∃ x : ℝ, x < 0 ∧ f x = 0) ∧ 
  (∃ y : ℝ, y > 0 ∧ f 0 = y) := by
sorry

end NUMINAMATH_CALUDE_f_intersects_axes_l2341_234122


namespace NUMINAMATH_CALUDE_single_digit_between_4_and_9_less_than_6_l2341_234106

theorem single_digit_between_4_and_9_less_than_6 (n : ℕ) 
  (h1 : n ≤ 9)
  (h2 : 4 < n)
  (h3 : n < 9)
  (h4 : n < 6) : 
  n = 5 := by
sorry

end NUMINAMATH_CALUDE_single_digit_between_4_and_9_less_than_6_l2341_234106


namespace NUMINAMATH_CALUDE_games_missed_l2341_234120

theorem games_missed (total_games attended_games : ℕ) 
  (h1 : total_games = 89) 
  (h2 : attended_games = 47) : 
  total_games - attended_games = 42 := by
  sorry

end NUMINAMATH_CALUDE_games_missed_l2341_234120


namespace NUMINAMATH_CALUDE_three_numbers_sum_l2341_234161

theorem three_numbers_sum (a b c : ℝ) 
  (sum_ab : a + b = 37)
  (sum_bc : b + c = 58)
  (sum_ca : c + a = 72) :
  a + b + c - 10 = 73.5 := by sorry

end NUMINAMATH_CALUDE_three_numbers_sum_l2341_234161


namespace NUMINAMATH_CALUDE_range_of_f_l2341_234101

/-- The function f defined on real numbers. -/
def f (x : ℝ) : ℝ := x^2 + 1

/-- The range of f is [1, +∞) -/
theorem range_of_f : Set.range f = {y : ℝ | y ≥ 1} := by
  sorry

end NUMINAMATH_CALUDE_range_of_f_l2341_234101


namespace NUMINAMATH_CALUDE_r_l2341_234109

/-- r'(n) is the sum of distinct primes in the prime factorization of n -/
noncomputable def r' (n : ℕ) : ℕ :=
  (Nat.factors n).toFinset.sum id

/-- The set of composite positive integers -/
def CompositeSet : Set ℕ :=
  {n : ℕ | n > 1 ∧ ¬Nat.Prime n}

/-- The set of integers that can be expressed as sums of two or more distinct primes -/
def SumOfDistinctPrimesSet : Set ℕ :=
  {n : ℕ | ∃ (s : Finset ℕ), s.card ≥ 2 ∧ (∀ p ∈ s, Nat.Prime p) ∧ s.sum id = n}

/-- The range of r' is equal to the set of integers that can be expressed as sums of two or more distinct primes -/
theorem r'_range_eq_sum_of_distinct_primes :
  (CompositeSet.image r') = SumOfDistinctPrimesSet :=
sorry

end NUMINAMATH_CALUDE_r_l2341_234109


namespace NUMINAMATH_CALUDE_three_right_angles_implies_rectangle_l2341_234160

/-- A quadrilateral is a polygon with four sides and four vertices. -/
structure Quadrilateral where
  vertices : Fin 4 → ℝ × ℝ

/-- An angle is right if it measures 90 degrees or π/2 radians. -/
def is_right_angle (a b c : ℝ × ℝ) : Prop := sorry

/-- A quadrilateral is a rectangle if all its angles are right angles. -/
def is_rectangle (q : Quadrilateral) : Prop :=
  ∀ i : Fin 4, is_right_angle (q.vertices i) (q.vertices (i + 1)) (q.vertices (i + 2))

/-- If a quadrilateral has three right angles, then it is a rectangle. -/
theorem three_right_angles_implies_rectangle (q : Quadrilateral) :
  (∃ i j k : Fin 4, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    is_right_angle (q.vertices i) (q.vertices (i + 1)) (q.vertices (i + 2)) ∧
    is_right_angle (q.vertices j) (q.vertices (j + 1)) (q.vertices (j + 2)) ∧
    is_right_angle (q.vertices k) (q.vertices (k + 1)) (q.vertices (k + 2)))
  → is_rectangle q :=
sorry

end NUMINAMATH_CALUDE_three_right_angles_implies_rectangle_l2341_234160


namespace NUMINAMATH_CALUDE_at_most_two_sides_equal_longest_diagonal_l2341_234169

/-- A convex polygon -/
structure ConvexPolygon where
  -- We don't need to define the full structure, just declare it exists
  mk :: 

/-- The longest diagonal of a convex polygon -/
def longest_diagonal (p : ConvexPolygon) : ℝ := sorry

/-- A side of a convex polygon -/
def side (p : ConvexPolygon) : ℝ := sorry

/-- The number of sides in a convex polygon that are equal to the longest diagonal -/
def num_sides_equal_to_longest_diagonal (p : ConvexPolygon) : ℕ := sorry

/-- Theorem: At most two sides of a convex polygon can be equal to its longest diagonal -/
theorem at_most_two_sides_equal_longest_diagonal (p : ConvexPolygon) :
  num_sides_equal_to_longest_diagonal p ≤ 2 := by sorry

end NUMINAMATH_CALUDE_at_most_two_sides_equal_longest_diagonal_l2341_234169


namespace NUMINAMATH_CALUDE_isabel_sold_three_bead_necklaces_total_cost_equals_earnings_l2341_234119

/-- The number of bead necklaces sold by Isabel -/
def bead_necklaces : ℕ := sorry

/-- The number of gem stone necklaces sold by Isabel -/
def gem_necklaces : ℕ := 3

/-- The cost of each necklace in dollars -/
def necklace_cost : ℕ := 6

/-- The total earnings from all necklaces in dollars -/
def total_earnings : ℕ := 36

/-- Theorem stating that Isabel sold 3 bead necklaces -/
theorem isabel_sold_three_bead_necklaces :
  bead_necklaces = 3 :=
by
  sorry

/-- The total number of necklaces sold -/
def total_necklaces : ℕ := bead_necklaces + gem_necklaces

/-- The total cost of all necklaces sold -/
def total_cost : ℕ := total_necklaces * necklace_cost

/-- Assertion that the total cost equals the total earnings -/
theorem total_cost_equals_earnings :
  total_cost = total_earnings :=
by
  sorry

end NUMINAMATH_CALUDE_isabel_sold_three_bead_necklaces_total_cost_equals_earnings_l2341_234119


namespace NUMINAMATH_CALUDE_area_of_S_l2341_234167

/-- A regular octagon in the complex plane -/
structure RegularOctagon where
  center : ℂ
  side_distance : ℝ
  parallel_to_real_axis : Prop

/-- The region outside the octagon -/
def R (octagon : RegularOctagon) : Set ℂ :=
  sorry

/-- The set S defined by the inversion of R -/
def S (octagon : RegularOctagon) : Set ℂ :=
  {w | ∃ z ∈ R octagon, w = 1 / z}

/-- The area of a set in the complex plane -/
noncomputable def area (s : Set ℂ) : ℝ :=
  sorry

theorem area_of_S (octagon : RegularOctagon) 
    (h1 : octagon.center = 0)
    (h2 : octagon.side_distance = 1.5)
    (h3 : octagon.parallel_to_real_axis) :
  area (S octagon) = 9 * Real.pi + 4.5 * Real.sqrt 2 * Real.pi :=
sorry

end NUMINAMATH_CALUDE_area_of_S_l2341_234167


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2341_234111

/-- Represents a hyperbola with center O and focus F -/
structure Hyperbola where
  O : ℝ × ℝ  -- Center of the hyperbola
  F : ℝ × ℝ  -- Focus of the hyperbola

/-- Represents a point on the asymptote of the hyperbola -/
def AsymptoticPoint (h : Hyperbola) := ℝ × ℝ

/-- Checks if a triangle is isosceles right -/
def IsIsoscelesRight (A B C : ℝ × ℝ) : Prop := sorry

/-- Calculates the eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ := sorry

/-- Main theorem: If a point P on the asymptote of a hyperbola forms an isosceles right triangle
    with the center O and focus F, then the eccentricity of the hyperbola is √2 -/
theorem hyperbola_eccentricity 
  (h : Hyperbola) 
  (P : AsymptoticPoint h) 
  (h_isosceles : IsIsoscelesRight h.O h.F P) : 
  eccentricity h = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2341_234111


namespace NUMINAMATH_CALUDE_intersection_equality_l2341_234189

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.log (1 - x) ∧ 1 - x > 0}
def B (m : ℝ) : Set ℝ := {y | ∃ x, y = -x^2 + 2*x + m}

-- State the theorem
theorem intersection_equality (m : ℝ) : A ∩ B m = A ↔ m ≥ 0 := by sorry

end NUMINAMATH_CALUDE_intersection_equality_l2341_234189


namespace NUMINAMATH_CALUDE_solution_replacement_fraction_l2341_234191

theorem solution_replacement_fraction 
  (initial_concentration : Real)
  (replacement_concentration : Real)
  (final_concentration : Real)
  (h1 : initial_concentration = 0.40)
  (h2 : replacement_concentration = 0.25)
  (h3 : final_concentration = 0.35)
  : ∃ x : Real, x = 1/3 ∧ 
    final_concentration * 1 = 
    (initial_concentration * (1 - x)) + (replacement_concentration * x) := by
  sorry

end NUMINAMATH_CALUDE_solution_replacement_fraction_l2341_234191


namespace NUMINAMATH_CALUDE_square_area_from_adjacent_points_l2341_234184

/-- The area of a square given two adjacent points on a Cartesian plane -/
theorem square_area_from_adjacent_points (x₁ y₁ x₂ y₂ : ℝ) :
  x₁ = 1 ∧ y₁ = 2 ∧ x₂ = 4 ∧ y₂ = 6 →
  (∃ (area : ℝ), area = 25 ∧ 
    area = ((x₂ - x₁)^2 + (y₂ - y₁)^2)) :=
by sorry


end NUMINAMATH_CALUDE_square_area_from_adjacent_points_l2341_234184


namespace NUMINAMATH_CALUDE_wall_area_2_by_4_l2341_234129

/-- The area of a rectangular wall -/
def wall_area (width height : ℝ) : ℝ := width * height

/-- Theorem: The area of a wall that is 2 feet wide and 4 feet tall is 8 square feet -/
theorem wall_area_2_by_4 : wall_area 2 4 = 8 := by
  sorry

end NUMINAMATH_CALUDE_wall_area_2_by_4_l2341_234129


namespace NUMINAMATH_CALUDE_max_trees_on_road_l2341_234103

theorem max_trees_on_road (road_length : ℕ) (interval : ℕ) (h1 : road_length = 28) (h2 : interval = 4) :
  (road_length / interval) + 1 = 8 := by
  sorry

end NUMINAMATH_CALUDE_max_trees_on_road_l2341_234103


namespace NUMINAMATH_CALUDE_excluded_students_average_mark_l2341_234170

/-- Proves that given a class of 35 students with an average mark of 80,
    if 5 students are excluded and the remaining students have an average mark of 90,
    then the average mark of the excluded students is 20. -/
theorem excluded_students_average_mark
  (total_students : ℕ)
  (class_average : ℚ)
  (remaining_students : ℕ)
  (remaining_average : ℚ)
  (h1 : total_students = 35)
  (h2 : class_average = 80)
  (h3 : remaining_students = 30)
  (h4 : remaining_average = 90) :
  let excluded_students := total_students - remaining_students
  let excluded_average := (total_students * class_average - remaining_students * remaining_average) / excluded_students
  excluded_average = 20 := by
  sorry

#check excluded_students_average_mark

end NUMINAMATH_CALUDE_excluded_students_average_mark_l2341_234170


namespace NUMINAMATH_CALUDE_hidden_primes_average_l2341_234116

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def card_sum (a b : ℕ) : ℕ := a + b

theorem hidden_primes_average (p₁ p₂ p₃ : ℕ) 
  (h₁ : is_prime p₁) (h₂ : is_prime p₂) (h₃ : is_prime p₃)
  (h₄ : card_sum p₁ 51 = card_sum p₂ 72)
  (h₅ : card_sum p₂ 72 = card_sum p₃ 43)
  (h₆ : p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₂ ≠ p₃)
  (h₇ : p₁ ≠ 51 ∧ p₂ ≠ 72 ∧ p₃ ≠ 43) :
  (p₁ + p₂ + p₃) / 3 = 56 / 3 :=
by sorry

end NUMINAMATH_CALUDE_hidden_primes_average_l2341_234116


namespace NUMINAMATH_CALUDE_quadratic_expression_k_value_l2341_234171

theorem quadratic_expression_k_value :
  ∀ a h k : ℝ, (∀ x : ℝ, x^2 - 8*x = a*(x - h)^2 + k) → k = -16 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_k_value_l2341_234171


namespace NUMINAMATH_CALUDE_inverse_proportion_ordering_l2341_234188

/-- Represents a point on the inverse proportion function -/
structure InversePoint where
  x : ℝ
  y : ℝ
  k : ℝ
  h : y = k / x

/-- The theorem statement -/
theorem inverse_proportion_ordering
  (p₁ : InversePoint)
  (p₂ : InversePoint)
  (p₃ : InversePoint)
  (h₁ : p₁.x = -1)
  (h₂ : p₂.x = 2)
  (h₃ : p₃.x = 3)
  (hk : p₁.k = p₂.k ∧ p₂.k = p₃.k ∧ p₁.k < 0) :
  p₁.y > p₃.y ∧ p₃.y > p₂.y :=
by sorry

end NUMINAMATH_CALUDE_inverse_proportion_ordering_l2341_234188


namespace NUMINAMATH_CALUDE_circle_center_coordinates_l2341_234173

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in the form ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y = l.c

/-- Check if a circle with center p is tangent to two parallel lines -/
def circleTangentToParallelLines (p : Point) (l1 l2 : Line) : Prop :=
  l1.a = l2.a ∧ l1.b = l2.b ∧ 
  abs (l1.a * p.x + l1.b * p.y - l1.c) = abs (l2.a * p.x + l2.b * p.y - l2.c)

theorem circle_center_coordinates : 
  ∃ (p : Point),
    circleTangentToParallelLines p (Line.mk 3 4 40) (Line.mk 3 4 (-20)) ∧
    pointOnLine p (Line.mk 1 (-2) 0) ∧
    p.x = 2 ∧ p.y = 1 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_coordinates_l2341_234173


namespace NUMINAMATH_CALUDE_cube_inequality_l2341_234199

theorem cube_inequality (a b : ℝ) (h : a > b) : a^3 > b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_inequality_l2341_234199


namespace NUMINAMATH_CALUDE_construction_problem_l2341_234181

/-- Represents the construction plan for a quarter --/
structure ConstructionPlan where
  ordinary : ℝ
  elevated : ℝ
  tunnel : ℝ

/-- Represents the cost per kilometer for each type of construction --/
structure CostPerKm where
  ordinary : ℝ
  elevated : ℝ
  tunnel : ℝ

/-- Calculates the total cost of a construction plan given the cost per kilometer --/
def totalCost (plan : ConstructionPlan) (cost : CostPerKm) : ℝ :=
  plan.ordinary * cost.ordinary + plan.elevated * cost.elevated + plan.tunnel * cost.tunnel

theorem construction_problem (a : ℝ) :
  let q1_plan : ConstructionPlan := { ordinary := 32, elevated := 21, tunnel := 3 }
  let q1_cost : CostPerKm := { ordinary := 1, elevated := 2, tunnel := 4 }
  let q2_plan : ConstructionPlan := { ordinary := 32 - 9*a, elevated := 21 - 2*a, tunnel := 3 + a }
  let q2_cost : CostPerKm := { ordinary := 1, elevated := 2 + 0.5*a, tunnel := 4 }
  
  (∀ x, x ≤ 3 → 56 - 32 - x ≥ 7*x) ∧ 
  (totalCost q1_plan q1_cost = totalCost q2_plan q2_cost) →
  a = 3/2 := by sorry

end NUMINAMATH_CALUDE_construction_problem_l2341_234181


namespace NUMINAMATH_CALUDE_total_animals_seen_l2341_234148

theorem total_animals_seen (initial_beavers initial_chipmunks : ℕ) : 
  initial_beavers = 35 →
  initial_chipmunks = 60 →
  (initial_beavers + initial_chipmunks) + (3 * initial_beavers + (initial_chipmunks - 15)) = 245 := by
  sorry

end NUMINAMATH_CALUDE_total_animals_seen_l2341_234148


namespace NUMINAMATH_CALUDE_roots_equation_sum_l2341_234104

theorem roots_equation_sum (α β : ℝ) : 
  α^2 - 3*α + 1 = 0 → β^2 - 3*β + 1 = 0 → 3*α^5 + 7*β^4 = 817 := by
  sorry

end NUMINAMATH_CALUDE_roots_equation_sum_l2341_234104


namespace NUMINAMATH_CALUDE_triangle_side_length_l2341_234133

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides opposite to A, B, C respectively

-- State the theorem
theorem triangle_side_length 
  (ABC : Triangle) 
  (h1 : 2 * (ABC.b * Real.cos ABC.A + ABC.a * Real.cos ABC.B) = ABC.c ^ 2)
  (h2 : ABC.b = 3)
  (h3 : 3 * Real.cos ABC.A = 1) :
  ABC.a = 3 := by
  sorry


end NUMINAMATH_CALUDE_triangle_side_length_l2341_234133


namespace NUMINAMATH_CALUDE_smallest_five_digit_divisible_by_3_and_4_l2341_234175

theorem smallest_five_digit_divisible_by_3_and_4 : ∃ n : ℕ, 
  (n ≥ 10000 ∧ n < 100000) ∧ 
  n % 3 = 0 ∧ 
  n % 4 = 0 ∧
  (∀ m : ℕ, m ≥ 10000 ∧ m < 100000 ∧ m % 3 = 0 ∧ m % 4 = 0 → m ≥ n) ∧
  n = 10008 := by
sorry

end NUMINAMATH_CALUDE_smallest_five_digit_divisible_by_3_and_4_l2341_234175


namespace NUMINAMATH_CALUDE_power_equation_l2341_234163

theorem power_equation (x y z : ℕ) : 
  3^x * 4^y = z → x - y = 9 → x = 9 → z = 19683 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_l2341_234163


namespace NUMINAMATH_CALUDE_fruit_store_total_weight_l2341_234123

theorem fruit_store_total_weight 
  (boxes_sold : ℕ) 
  (weight_per_box : ℕ) 
  (remaining_weight : ℕ) 
  (h1 : boxes_sold = 14)
  (h2 : weight_per_box = 30)
  (h3 : remaining_weight = 80) :
  boxes_sold * weight_per_box + remaining_weight = 500 := by
sorry

end NUMINAMATH_CALUDE_fruit_store_total_weight_l2341_234123


namespace NUMINAMATH_CALUDE_sum_of_coefficients_10_11_l2341_234178

/-- Given that (x-1)^21 = a + a₁x + a₂x² + ... + a₂₁x²¹, prove that a₁₀ + a₁₁ = 0 -/
theorem sum_of_coefficients_10_11 (a : ℝ) (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ a₁₂ a₁₃ a₁₄ a₁₅ a₁₆ a₁₇ a₁₈ a₁₉ a₂₀ a₂₁ : ℝ) :
  (∀ x : ℝ, (x - 1)^21 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9 + 
             a₁₀*x^10 + a₁₁*x^11 + a₁₂*x^12 + a₁₃*x^13 + a₁₄*x^14 + a₁₅*x^15 + a₁₆*x^16 + a₁₇*x^17 + a₁₈*x^18 + 
             a₁₉*x^19 + a₂₀*x^20 + a₂₁*x^21) →
  a₁₀ + a₁₁ = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_10_11_l2341_234178


namespace NUMINAMATH_CALUDE_jane_ribbons_per_dress_l2341_234115

/-- The number of ribbons Jane adds to each dress --/
def ribbons_per_dress (dresses_first_week : ℕ) (dresses_second_week : ℕ) (total_ribbons : ℕ) : ℚ :=
  total_ribbons / (dresses_first_week + dresses_second_week)

/-- Theorem stating that Jane adds 2 ribbons to each dress --/
theorem jane_ribbons_per_dress :
  ribbons_per_dress (7 * 2) (2 * 3) 40 = 2 := by
  sorry

end NUMINAMATH_CALUDE_jane_ribbons_per_dress_l2341_234115


namespace NUMINAMATH_CALUDE_min_value_theorem_l2341_234139

theorem min_value_theorem (x : ℝ) (h : x > 0) :
  x^2 + 6*x + 100/x^3 ≥ 3 * 50^(2/5) + 6 * 50^(1/5) ∧
  ∃ y > 0, y^2 + 6*y + 100/y^3 = 3 * 50^(2/5) + 6 * 50^(1/5) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2341_234139


namespace NUMINAMATH_CALUDE_key_lime_requirement_l2341_234165

/-- The number of tablespoons in one cup -/
def tablespoons_per_cup : ℕ := 16

/-- The original amount of key lime juice in cups -/
def original_juice_cups : ℚ := 1/4

/-- The multiplication factor for the juice amount -/
def juice_multiplier : ℕ := 3

/-- The minimum amount of juice (in tablespoons) that a key lime can yield -/
def min_juice_per_lime : ℕ := 1

/-- The maximum amount of juice (in tablespoons) that a key lime can yield -/
def max_juice_per_lime : ℕ := 2

/-- The number of key limes needed to ensure enough juice for the recipe -/
def key_limes_needed : ℕ := 12

theorem key_lime_requirement :
  key_limes_needed * min_juice_per_lime ≥ 
  juice_multiplier * (original_juice_cups * tablespoons_per_cup) ∧
  key_limes_needed * max_juice_per_lime ≥
  juice_multiplier * (original_juice_cups * tablespoons_per_cup) ∧
  ∀ n : ℕ, n < key_limes_needed →
    n * min_juice_per_lime < juice_multiplier * (original_juice_cups * tablespoons_per_cup) :=
by sorry

end NUMINAMATH_CALUDE_key_lime_requirement_l2341_234165


namespace NUMINAMATH_CALUDE_saree_stripes_l2341_234151

theorem saree_stripes (brown gold blue : ℕ) : 
  gold = 3 * brown → 
  blue = 5 * gold → 
  brown = 4 → 
  blue = 60 := by
  sorry

end NUMINAMATH_CALUDE_saree_stripes_l2341_234151


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_inequality_negation_of_proposition_l2341_234159

theorem negation_of_existence (p : ℝ → Prop) : 
  (¬∃ x, p x) ↔ (∀ x, ¬p x) :=
by sorry

theorem negation_of_inequality : 
  (¬(x + 2 ≤ 0)) ↔ (x + 2 > 0) :=
by sorry

theorem negation_of_proposition : 
  (¬∃ x : ℝ, x + 2 ≤ 0) ↔ (∀ x : ℝ, x + 2 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_inequality_negation_of_proposition_l2341_234159


namespace NUMINAMATH_CALUDE_solve_system_l2341_234142

theorem solve_system (x y : ℝ) (h1 : x - 2*y = 10) (h2 : x * y = 40) : y = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l2341_234142


namespace NUMINAMATH_CALUDE_parabola_ratio_l2341_234190

/-- Given a parabola y = ax² + bx + c passing through points (-1, 1) and (3, 1),
    prove that a/b = -2 -/
theorem parabola_ratio (a b c : ℝ) : 
  (∀ x, a * x^2 + b * x + c = 1 → x = -1 ∨ x = 3) →
  a / b = -2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_ratio_l2341_234190


namespace NUMINAMATH_CALUDE_min_sum_squares_l2341_234141

theorem min_sum_squares (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3*x*y*z = 8) :
  ∃ (m : ℝ), m = 4 ∧ (∀ a b c : ℝ, a^3 + b^3 + c^3 - 3*a*b*c = 8 → a^2 + b^2 + c^2 ≥ m) ∧
  (∃ p q r : ℝ, p^3 + q^3 + r^3 - 3*p*q*r = 8 ∧ p^2 + q^2 + r^2 = m) :=
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l2341_234141


namespace NUMINAMATH_CALUDE_computer_operations_per_hour_l2341_234102

theorem computer_operations_per_hour :
  let additions_per_second : ℕ := 12000
  let multiplications_per_second : ℕ := 8000
  let seconds_per_hour : ℕ := 3600
  let total_operations_per_second : ℕ := additions_per_second + multiplications_per_second
  let operations_per_hour : ℕ := total_operations_per_second * seconds_per_hour
  operations_per_hour = 72000000 := by
sorry

end NUMINAMATH_CALUDE_computer_operations_per_hour_l2341_234102


namespace NUMINAMATH_CALUDE_max_value_on_sphere_l2341_234195

theorem max_value_on_sphere (x y z : ℝ) (h : x^2 + y^2 + 4*z^2 = 1) :
  ∃ (max : ℝ), max = Real.sqrt 6 ∧ ∀ (a b c : ℝ), a^2 + b^2 + 4*c^2 = 1 → a + b + 4*c ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_on_sphere_l2341_234195


namespace NUMINAMATH_CALUDE_quadratic_function_coefficient_l2341_234145

theorem quadratic_function_coefficient (a b c : ℤ) :
  (∀ x y : ℝ, y = a * x^2 + b * x + c) →
  (2, -3) = (-(b / (2 * a)), -(b^2 - 4 * a * c) / (4 * a)) →
  1 = a * 0^2 + b * 0 + c →
  6 = a * 5^2 + b * 5 + c →
  a = 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_function_coefficient_l2341_234145


namespace NUMINAMATH_CALUDE_line_equation_through_points_l2341_234126

/-- The line passing through points A(0, -5) and B(1, 0) has the equation y = 5x - 5 -/
theorem line_equation_through_points (x y : ℝ) : 
  (x = 0 ∧ y = -5) ∨ (x = 1 ∧ y = 0) → y = 5*x - 5 :=
by sorry

end NUMINAMATH_CALUDE_line_equation_through_points_l2341_234126


namespace NUMINAMATH_CALUDE_maintenance_interval_doubled_l2341_234157

/-- 
Given an original maintenance check interval and a percentage increase,
this function calculates the new maintenance check interval.
-/
def new_maintenance_interval (original : ℕ) (percent_increase : ℕ) : ℕ :=
  original * (100 + percent_increase) / 100

/-- 
Theorem: If the original maintenance check interval is 30 days and 
the interval is increased by 100%, then the new interval is 60 days.
-/
theorem maintenance_interval_doubled :
  new_maintenance_interval 30 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_maintenance_interval_doubled_l2341_234157


namespace NUMINAMATH_CALUDE_compound_formula_l2341_234154

-- Define atomic weights
def atomic_weight_Al : ℝ := 26.98
def atomic_weight_O : ℝ := 16.00

-- Define the number of oxygen atoms
def num_O : ℕ := 3

-- Define the total molecular weight
def total_molecular_weight : ℝ := 102

-- Define the molecular formula
structure MolecularFormula where
  num_Al : ℕ
  num_O : ℕ

-- Theorem to prove
theorem compound_formula :
  ∃ (formula : MolecularFormula),
    formula.num_O = num_O ∧
    formula.num_Al * atomic_weight_Al + formula.num_O * atomic_weight_O = total_molecular_weight ∧
    formula = MolecularFormula.mk 2 3 := by
  sorry


end NUMINAMATH_CALUDE_compound_formula_l2341_234154


namespace NUMINAMATH_CALUDE_bus_average_speed_l2341_234132

/-- Proves that given a bicycle traveling at 15 km/h and a bus starting 195 km behind it,
    if the bus catches up to the bicycle in 3 hours, then the average speed of the bus is 80 km/h. -/
theorem bus_average_speed
  (bicycle_speed : ℝ)
  (initial_distance : ℝ)
  (catch_up_time : ℝ)
  (h1 : bicycle_speed = 15)
  (h2 : initial_distance = 195)
  (h3 : catch_up_time = 3)
  : (initial_distance + bicycle_speed * catch_up_time) / catch_up_time = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_bus_average_speed_l2341_234132


namespace NUMINAMATH_CALUDE_symmetric_function_is_odd_and_periodic_l2341_234121

/-- A function satisfying specific symmetry properties -/
def SymmetricFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (10 + x) = f (10 - x)) ∧ 
  (∀ x, f (20 - x) = -f (20 + x))

/-- A function is odd -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function is periodic with period T -/
def PeriodicFunction (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

/-- Theorem: A function satisfying specific symmetry properties is odd and periodic with period 40 -/
theorem symmetric_function_is_odd_and_periodic (f : ℝ → ℝ) 
  (h : SymmetricFunction f) : 
  OddFunction f ∧ PeriodicFunction f 40 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_function_is_odd_and_periodic_l2341_234121


namespace NUMINAMATH_CALUDE_internally_tangent_circles_l2341_234140

/-- Given two circles, where one is internally tangent to the other, prove the possible values of m -/
theorem internally_tangent_circles (m : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 = m) ∧ 
  (∃ (x y : ℝ), x^2 + y^2 + 6*x - 8*y - 11 = 0) ∧ 
  (∃ (x y : ℝ), x^2 + y^2 = m ∧ x^2 + y^2 + 6*x - 8*y - 11 = 0) →
  m = 1 ∨ m = 121 :=
by sorry

end NUMINAMATH_CALUDE_internally_tangent_circles_l2341_234140


namespace NUMINAMATH_CALUDE_tailor_cuts_difference_l2341_234198

theorem tailor_cuts_difference : 
  (7/8 + 11/12) - (5/6 + 3/4) = 5/24 := by
  sorry

end NUMINAMATH_CALUDE_tailor_cuts_difference_l2341_234198


namespace NUMINAMATH_CALUDE_product_zero_from_sum_and_cube_sum_l2341_234196

theorem product_zero_from_sum_and_cube_sum (a b : ℝ) 
  (h1 : a + b = 5) 
  (h2 : a^3 + b^3 = 125) : 
  a * b = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_zero_from_sum_and_cube_sum_l2341_234196


namespace NUMINAMATH_CALUDE_monotonicity_of_g_minimum_a_for_negative_f_l2341_234164

noncomputable section

def f (a x : ℝ) : ℝ := x * Real.log (x + 1) + (1/2 - a) * x + 2 - a

def g (a x : ℝ) : ℝ := f a x + Real.log (x + 1) + (1/2) * x

theorem monotonicity_of_g (a : ℝ) :
  (a ≤ 2 → StrictMono (g a)) ∧
  (a > 2 → StrictAntiOn (g a) (Set.Ioo 0 (Real.exp (a - 2) - 1)) ∧
           StrictMono (g a ∘ (λ x => x + Real.exp (a - 2) - 1))) :=
sorry

theorem minimum_a_for_negative_f :
  (∃ (a : ℤ), ∃ (x : ℝ), x ≥ 0 ∧ f a x < 0) ∧
  (∀ (a : ℤ), a < 3 → ∀ (x : ℝ), x ≥ 0 → f a x ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_monotonicity_of_g_minimum_a_for_negative_f_l2341_234164


namespace NUMINAMATH_CALUDE_number_problem_l2341_234174

theorem number_problem (n : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * n = 15 → (40/100 : ℝ) * n = 180 :=
by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2341_234174


namespace NUMINAMATH_CALUDE_roque_bike_time_l2341_234152

/-- Represents the time in hours for Roque's commute -/
structure CommuteTime where
  walk_one_way : ℝ
  bike_one_way : ℝ
  walk_trips_per_week : ℕ
  bike_trips_per_week : ℕ
  total_time_per_week : ℝ

/-- Theorem stating that given the conditions, Roque's bike ride to work takes 1 hour -/
theorem roque_bike_time (c : CommuteTime)
  (h1 : c.walk_one_way = 2)
  (h2 : c.walk_trips_per_week = 3)
  (h3 : c.bike_trips_per_week = 2)
  (h4 : c.total_time_per_week = 16)
  (h5 : c.total_time_per_week = 2 * c.walk_one_way * c.walk_trips_per_week + 2 * c.bike_one_way * c.bike_trips_per_week) :
  c.bike_one_way = 1 := by
  sorry

end NUMINAMATH_CALUDE_roque_bike_time_l2341_234152


namespace NUMINAMATH_CALUDE_units_digit_of_F_F7_l2341_234113

-- Define the modified Fibonacci sequence
def modifiedFib : ℕ → ℕ
  | 0 => 3
  | 1 => 5
  | (n + 2) => modifiedFib (n + 1) + modifiedFib n

-- Function to get the units digit
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Theorem statement
theorem units_digit_of_F_F7 :
  unitsDigit (modifiedFib (modifiedFib 7)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_F_F7_l2341_234113


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l2341_234177

theorem perfect_square_trinomial (a : ℚ) : 
  (∃ r s : ℚ, a * x^2 + 20 * x + 9 = (r * x + s)^2) → a = 100 / 9 :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l2341_234177


namespace NUMINAMATH_CALUDE_perpendicular_condition_l2341_234192

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between planes and between a line and a plane
variable (perp_planes : Plane → Plane → Prop)
variable (perp_line_plane : Line → Plane → Prop)

-- Define the relation of a line being within a plane
variable (line_in_plane : Line → Plane → Prop)

theorem perpendicular_condition (α β : Plane) (m : Line) 
  (h1 : α ≠ β) 
  (h2 : line_in_plane m α) : 
  (∀ m, line_in_plane m α → perp_line_plane m β → perp_planes α β) ∧ 
  (∃ m, line_in_plane m α ∧ perp_planes α β ∧ ¬perp_line_plane m β) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_condition_l2341_234192


namespace NUMINAMATH_CALUDE_pigeon_hole_theorem_l2341_234172

/-- The number of pigeons -/
def num_pigeons : ℕ := 160

/-- The function that determines which hole a pigeon flies to -/
def pigeon_hole (i n : ℕ) : ℕ := i^2 % n

/-- Predicate to check if all pigeons fly to unique holes -/
def all_unique_holes (n : ℕ) : Prop :=
  ∀ i j, i ≤ num_pigeons → j ≤ num_pigeons → i ≠ j → pigeon_hole i n ≠ pigeon_hole j n

/-- The minimum number of holes needed -/
def min_holes : ℕ := 326

theorem pigeon_hole_theorem :
  (∀ k, k < min_holes → ¬(all_unique_holes k)) ∧ all_unique_holes min_holes :=
by sorry

end NUMINAMATH_CALUDE_pigeon_hole_theorem_l2341_234172


namespace NUMINAMATH_CALUDE_five_topping_pizzas_l2341_234108

theorem five_topping_pizzas (n : ℕ) (k : ℕ) (h1 : n = 8) (h2 : k = 5) :
  Nat.choose n k = 56 := by
  sorry

end NUMINAMATH_CALUDE_five_topping_pizzas_l2341_234108


namespace NUMINAMATH_CALUDE_states_fraction_proof_l2341_234193

theorem states_fraction_proof (total_states : ℕ) (decade_states : ℕ) :
  total_states = 22 →
  decade_states = 8 →
  (decade_states : ℚ) / total_states = 4 / 11 := by
sorry

end NUMINAMATH_CALUDE_states_fraction_proof_l2341_234193


namespace NUMINAMATH_CALUDE_solve_for_b_l2341_234162

theorem solve_for_b (a b : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 25 * 315 * b) : b = 7 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_b_l2341_234162


namespace NUMINAMATH_CALUDE_katie_soccer_granola_l2341_234112

/-- The number of boxes of granola bars needed for a soccer game --/
def granola_boxes_needed (num_kids : ℕ) (bars_per_kid : ℕ) (bars_per_box : ℕ) : ℕ :=
  (num_kids * bars_per_kid + bars_per_box - 1) / bars_per_box

theorem katie_soccer_granola : granola_boxes_needed 30 2 12 = 5 := by
  sorry

end NUMINAMATH_CALUDE_katie_soccer_granola_l2341_234112


namespace NUMINAMATH_CALUDE_clock_hands_coincide_l2341_234137

/-- The rate at which the hour hand moves, in degrees per minute -/
def hour_hand_rate : ℝ := 0.5

/-- The rate at which the minute hand moves, in degrees per minute -/
def minute_hand_rate : ℝ := 6

/-- The position of the hour hand at 7:00, in degrees -/
def initial_hour_hand_position : ℝ := 210

/-- The time interval in which we're checking for coincidence -/
def time_interval : Set ℝ := {t | 30 ≤ t ∧ t ≤ 45}

/-- The theorem stating that the clock hands coincide once in the given interval -/
theorem clock_hands_coincide : ∃ t ∈ time_interval, 
  initial_hour_hand_position + hour_hand_rate * t = minute_hand_rate * t :=
sorry

end NUMINAMATH_CALUDE_clock_hands_coincide_l2341_234137


namespace NUMINAMATH_CALUDE_mrs_hilt_friends_l2341_234185

/-- The number of friends Mrs. Hilt met who were carrying pears -/
def friends_with_pears : ℕ := 9

/-- The number of friends Mrs. Hilt met who were carrying oranges -/
def friends_with_oranges : ℕ := 6

/-- The total number of friends Mrs. Hilt met -/
def total_friends : ℕ := friends_with_pears + friends_with_oranges

theorem mrs_hilt_friends :
  total_friends = 15 :=
by sorry

end NUMINAMATH_CALUDE_mrs_hilt_friends_l2341_234185


namespace NUMINAMATH_CALUDE_initial_capital_calculation_l2341_234125

def profit_distribution_ratio : ℚ := 2/3
def income_increase : ℕ := 200
def initial_profit_rate : ℚ := 5/100
def final_profit_rate : ℚ := 7/100

theorem initial_capital_calculation (P : ℚ) : 
  P * final_profit_rate * profit_distribution_ratio - 
  P * initial_profit_rate * profit_distribution_ratio = income_increase →
  P = 15000 := by
sorry

end NUMINAMATH_CALUDE_initial_capital_calculation_l2341_234125


namespace NUMINAMATH_CALUDE_digit_sum_puzzle_l2341_234156

theorem digit_sum_puzzle : ∀ (a b c d e f g : ℕ),
  a ∈ ({0, 1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) →
  b ∈ ({0, 1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) →
  c ∈ ({0, 1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) →
  d ∈ ({0, 1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) →
  e ∈ ({0, 1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) →
  f ∈ ({0, 1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) →
  g ∈ ({0, 1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) →
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧
  e ≠ f ∧ e ≠ g ∧
  f ≠ g →
  a + b + c = 24 →
  d + e + f + g = 14 →
  (b = e ∨ a = e ∨ c = e) →
  a + b + c + d + f + g = 30 :=
by sorry

end NUMINAMATH_CALUDE_digit_sum_puzzle_l2341_234156


namespace NUMINAMATH_CALUDE_gmat_question_percentage_l2341_234134

/-- The percentage of test takers who answered the second question correctly -/
def second_correct : ℝ := 75

/-- The percentage of test takers who answered neither question correctly -/
def neither_correct : ℝ := 5

/-- The percentage of test takers who answered both questions correctly -/
def both_correct : ℝ := 60

/-- The percentage of test takers who answered the first question correctly -/
def first_correct : ℝ := 80

theorem gmat_question_percentage :
  first_correct = 80 :=
sorry

end NUMINAMATH_CALUDE_gmat_question_percentage_l2341_234134


namespace NUMINAMATH_CALUDE_candies_per_house_l2341_234130

/-- Proves that the number of candies received from each house is 7,
    given that there are 5 houses in a block and 35 candies are received from each block. -/
theorem candies_per_house
  (houses_per_block : ℕ)
  (candies_per_block : ℕ)
  (h1 : houses_per_block = 5)
  (h2 : candies_per_block = 35) :
  candies_per_block / houses_per_block = 7 :=
by sorry

end NUMINAMATH_CALUDE_candies_per_house_l2341_234130


namespace NUMINAMATH_CALUDE_salary_change_l2341_234143

theorem salary_change (S : ℝ) : 
  let increase := S * 1.2
  let decrease := increase * 0.8
  decrease = S * 0.96 := by sorry

end NUMINAMATH_CALUDE_salary_change_l2341_234143


namespace NUMINAMATH_CALUDE_find_k_l2341_234187

theorem find_k (k : ℝ) (h1 : k ≠ 0) : 
  (∀ x : ℝ, (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 2)) → k = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_k_l2341_234187


namespace NUMINAMATH_CALUDE_right_rectangular_prism_x_value_l2341_234180

theorem right_rectangular_prism_x_value 
  (x : ℝ) 
  (h1 : x > 0) -- Ensure x is positive for valid logarithms
  (edge1 : ℝ := Real.log x / Real.log 5)
  (edge2 : ℝ := Real.log x / Real.log 6)
  (edge3 : ℝ := Real.log x / Real.log 10)
  (surface_area : ℝ := 2 * (edge1 * edge2 + edge1 * edge3 + edge2 * edge3))
  (volume : ℝ := edge1 * edge2 * edge3)
  (h2 : surface_area = 3 * volume) :
  x = 300^(2/3) :=
by sorry

end NUMINAMATH_CALUDE_right_rectangular_prism_x_value_l2341_234180


namespace NUMINAMATH_CALUDE_total_value_calculation_l2341_234168

/-- Calculates the total value of coins and paper money with a certificate bonus --/
def totalValue (goldWorth silverWorth bronzeWorth titaniumWorth : ℝ)
                (banknoteWorth couponWorth voucherWorth : ℝ)
                (goldCount silverCount bronzeCount titaniumCount : ℕ)
                (banknoteCount couponCount voucherCount : ℕ)
                (certificateBonus : ℝ) : ℝ :=
  let goldValue := goldWorth * goldCount
  let silverValue := silverWorth * silverCount
  let bronzeValue := bronzeWorth * bronzeCount
  let titaniumValue := titaniumWorth * titaniumCount
  let banknoteValue := banknoteWorth * banknoteCount
  let couponValue := couponWorth * couponCount
  let voucherValue := voucherWorth * voucherCount
  let baseTotal := goldValue + silverValue + bronzeValue + titaniumValue +
                   banknoteValue + couponValue + voucherValue
  let bonusAmount := certificateBonus * (goldValue + silverValue)
  baseTotal + bonusAmount

theorem total_value_calculation :
  totalValue 80 45 25 10 50 10 20 7 9 12 5 3 6 4 0.05 = 1653.25 := by
  sorry

end NUMINAMATH_CALUDE_total_value_calculation_l2341_234168


namespace NUMINAMATH_CALUDE_math_test_problem_l2341_234153

theorem math_test_problem (total_questions word_problems steve_answers difference : ℕ)
  (h1 : total_questions = 45)
  (h2 : word_problems = 17)
  (h3 : steve_answers = 38)
  (h4 : difference = total_questions - steve_answers)
  (h5 : difference = 7) :
  total_questions - word_problems - steve_answers = 21 :=
by sorry

end NUMINAMATH_CALUDE_math_test_problem_l2341_234153


namespace NUMINAMATH_CALUDE_star_power_equality_l2341_234105

/-- The k-th smallest positive integer not in X -/
def f_X (X : Finset ℕ+) (k : ℕ+) : ℕ+ := sorry

/-- The * operation on finite sets of positive integers -/
def star (X Y : Finset ℕ+) : Finset ℕ+ :=
  X ∪ (Y.image (f_X X))

/-- Repeated application of star operation n times -/
def star_power (X : Finset ℕ+) : ℕ → Finset ℕ+
  | 0 => X
  | n + 1 => star X (star_power X n)

theorem star_power_equality {A B : Finset ℕ+} (hA : A.Nonempty) (hB : B.Nonempty)
    (h : star A B = star B A) :
    star_power A B.card = star_power B A.card := by sorry

end NUMINAMATH_CALUDE_star_power_equality_l2341_234105
