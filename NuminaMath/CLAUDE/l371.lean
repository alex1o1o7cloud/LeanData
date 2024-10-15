import Mathlib

namespace NUMINAMATH_CALUDE_mickeys_jaydens_difference_l371_37133

theorem mickeys_jaydens_difference (mickey jayden coraline : ℕ) : 
  (∃ d : ℕ, mickey = jayden + d) →
  jayden = coraline - 40 →
  coraline = 80 →
  mickey + jayden + coraline = 180 →
  ∃ d : ℕ, mickey = jayden + d ∧ d = 20 := by
  sorry

end NUMINAMATH_CALUDE_mickeys_jaydens_difference_l371_37133


namespace NUMINAMATH_CALUDE_quadratic_roots_nature_l371_37199

theorem quadratic_roots_nature (x : ℝ) : 
  let a : ℝ := 1
  let b : ℝ := -4 * Real.sqrt 5
  let c : ℝ := 20
  let discriminant := b^2 - 4*a*c
  discriminant = 0 ∧ ∃ (root : ℝ), x^2 - 4*x*(Real.sqrt 5) + 20 = 0 → x = root :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_nature_l371_37199


namespace NUMINAMATH_CALUDE_backyard_fence_problem_l371_37188

theorem backyard_fence_problem (back_length : ℝ) (fence_cost_per_foot : ℝ) 
  (owner_back_fraction : ℝ) (owner_left_fraction : ℝ) (owner_total_cost : ℝ) :
  back_length = 18 →
  fence_cost_per_foot = 3 →
  owner_back_fraction = 1/2 →
  owner_left_fraction = 2/3 →
  owner_total_cost = 72 →
  ∃ side_length : ℝ,
    side_length * fence_cost_per_foot * owner_left_fraction + 
    side_length * fence_cost_per_foot +
    back_length * fence_cost_per_foot * owner_back_fraction = owner_total_cost ∧
    side_length = 9 := by
  sorry

end NUMINAMATH_CALUDE_backyard_fence_problem_l371_37188


namespace NUMINAMATH_CALUDE_frog_arrangement_problem_l371_37104

theorem frog_arrangement_problem :
  ∃! (N : ℕ), 
    N > 0 ∧
    N % 2 = 1 ∧
    N % 3 = 1 ∧
    N % 4 = 1 ∧
    N % 5 = 0 ∧
    N < 50 ∧
    N = 25 := by sorry

end NUMINAMATH_CALUDE_frog_arrangement_problem_l371_37104


namespace NUMINAMATH_CALUDE_cube_sum_equals_linear_sum_l371_37152

theorem cube_sum_equals_linear_sum (a b : ℝ) 
  (h : (a / (1 + b)) + (b / (1 + a)) = 1) : 
  a^3 + b^3 = a + b := by sorry

end NUMINAMATH_CALUDE_cube_sum_equals_linear_sum_l371_37152


namespace NUMINAMATH_CALUDE_patrol_impossibility_l371_37132

/-- Represents the number of people in the group -/
def n : ℕ := 100

/-- Represents the number of people on duty each evening -/
def k : ℕ := 3

/-- Represents the total number of possible pairs of people -/
def totalPairs : ℕ := n.choose 2

/-- Represents the number of pairs formed each evening -/
def pairsPerEvening : ℕ := k.choose 2

theorem patrol_impossibility : ¬ ∃ (m : ℕ), m * pairsPerEvening = totalPairs ∧ 
  ∃ (f : Fin n → Fin m → Bool), 
    (∀ i j, i ≠ j → (∃! t, f i t ∧ f j t)) ∧
    (∀ t, ∃! (s : Fin k → Fin n), (∀ i, f (s i) t)) :=
sorry

end NUMINAMATH_CALUDE_patrol_impossibility_l371_37132


namespace NUMINAMATH_CALUDE_power_sums_l371_37145

variable (x y p q : ℝ)

def sum_condition : Prop := x + y = -p
def product_condition : Prop := x * y = q

theorem power_sums (h1 : sum_condition x y p) (h2 : product_condition x y q) :
  (x^2 + y^2 = p^2 - 2*q) ∧
  (x^3 + y^3 = -p^3 + 3*p*q) ∧
  (x^4 + y^4 = p^4 - 4*p^2*q + 2*q^2) := by
  sorry

end NUMINAMATH_CALUDE_power_sums_l371_37145


namespace NUMINAMATH_CALUDE_quadratic_rewrite_l371_37123

theorem quadratic_rewrite (d e f : ℤ) : 
  (∀ x : ℝ, 4 * x^2 - 16 * x + 2 = (d * x + e)^2 + f) → d * e = -8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_l371_37123


namespace NUMINAMATH_CALUDE_sue_answer_for_ben_partner_answer_formula_l371_37128

/-- Given an initial number, calculate the partner's final answer according to the instructions -/
def partnerAnswer (x : ℤ) : ℤ :=
  (((x + 2) * 3 - 2) * 3)

/-- Theorem stating that for Ben's initial number 6, Sue's answer should be 66 -/
theorem sue_answer_for_ben :
  partnerAnswer 6 = 66 := by sorry

/-- Theorem proving the general formula for the partner's answer -/
theorem partner_answer_formula (x : ℤ) :
  partnerAnswer x = (((x + 2) * 3 - 2) * 3) := by sorry

end NUMINAMATH_CALUDE_sue_answer_for_ben_partner_answer_formula_l371_37128


namespace NUMINAMATH_CALUDE_taobao_villages_growth_l371_37163

/-- 
Given an arithmetic sequence with first term 1311 and common difference 1000,
prove that the 8th term of this sequence is 8311.
-/
theorem taobao_villages_growth (a₁ : ℕ) (d : ℕ) (n : ℕ) :
  a₁ = 1311 → d = 1000 → n = 8 →
  a₁ + (n - 1) * d = 8311 :=
by sorry

end NUMINAMATH_CALUDE_taobao_villages_growth_l371_37163


namespace NUMINAMATH_CALUDE_smallest_non_prime_non_square_no_small_factors_l371_37148

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def is_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def has_prime_factor_less_than_60 (n : ℕ) : Prop :=
  ∃ p : ℕ, is_prime p ∧ p < 60 ∧ p ∣ n

theorem smallest_non_prime_non_square_no_small_factors : 
  (∀ n : ℕ, n < 4087 → is_prime n ∨ is_square n ∨ has_prime_factor_less_than_60 n) ∧
  ¬is_prime 4087 ∧
  ¬is_square 4087 ∧
  ¬has_prime_factor_less_than_60 4087 :=
sorry

end NUMINAMATH_CALUDE_smallest_non_prime_non_square_no_small_factors_l371_37148


namespace NUMINAMATH_CALUDE_scholarship_problem_l371_37194

theorem scholarship_problem (total_students : ℕ) 
  (full_merit_percent half_merit_percent sports_percent need_based_percent : ℚ)
  (full_merit_and_sports_percent half_merit_and_need_based_percent : ℚ)
  (h1 : total_students = 300)
  (h2 : full_merit_percent = 5 / 100)
  (h3 : half_merit_percent = 10 / 100)
  (h4 : sports_percent = 3 / 100)
  (h5 : need_based_percent = 7 / 100)
  (h6 : full_merit_and_sports_percent = 1 / 100)
  (h7 : half_merit_and_need_based_percent = 2 / 100) :
  ↑total_students - 
  (↑total_students * (full_merit_percent + half_merit_percent + sports_percent + need_based_percent) -
   ↑total_students * (full_merit_and_sports_percent + half_merit_and_need_based_percent)) = 234 := by
  sorry

end NUMINAMATH_CALUDE_scholarship_problem_l371_37194


namespace NUMINAMATH_CALUDE_circular_arrangement_problem_l371_37147

/-- Represents a circular arrangement of 6 numbers -/
structure CircularArrangement where
  numbers : Fin 6 → ℕ
  sum_rule : ∀ i : Fin 6, numbers i + numbers (i + 1) = 2 * numbers ((i + 2) % 6)

theorem circular_arrangement_problem 
  (arr : CircularArrangement)
  (h1 : ∃ i : Fin 6, arr.numbers i = 15 ∧ arr.numbers ((i + 1) % 6) + arr.numbers ((i + 5) % 6) = 16)
  (h2 : ∃ j : Fin 6, arr.numbers j + arr.numbers ((j + 2) % 6) = 10) :
  ∃ k : Fin 6, arr.numbers k = 7 ∧ arr.numbers ((k + 1) % 6) + arr.numbers ((k + 5) % 6) = 10 :=
sorry

end NUMINAMATH_CALUDE_circular_arrangement_problem_l371_37147


namespace NUMINAMATH_CALUDE_zeros_not_adjacent_probability_l371_37165

/-- The number of ones in the arrangement -/
def num_ones : ℕ := 4

/-- The number of zeros in the arrangement -/
def num_zeros : ℕ := 2

/-- The total number of spaces where zeros can be placed -/
def total_spaces : ℕ := num_ones + 1

/-- The probability that two zeros are not adjacent when randomly arranged with four ones -/
theorem zeros_not_adjacent_probability : 
  (Nat.choose total_spaces num_zeros : ℚ) / 
  (Nat.choose total_spaces 1 + Nat.choose total_spaces num_zeros : ℚ) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_zeros_not_adjacent_probability_l371_37165


namespace NUMINAMATH_CALUDE_total_ladybugs_count_l371_37131

/-- The number of ladybugs with spots -/
def ladybugs_with_spots : ℕ := 12170

/-- The number of ladybugs without spots -/
def ladybugs_without_spots : ℕ := 54912

/-- The total number of ladybugs -/
def total_ladybugs : ℕ := ladybugs_with_spots + ladybugs_without_spots

theorem total_ladybugs_count : total_ladybugs = 67082 := by
  sorry

end NUMINAMATH_CALUDE_total_ladybugs_count_l371_37131


namespace NUMINAMATH_CALUDE_divisors_121_divisors_1000_divisors_1000000000_l371_37107

-- Define a function to calculate the number of divisors given prime factorization
def num_divisors (factorization : List (Nat × Nat)) : Nat :=
  factorization.foldl (fun acc (_, exp) => acc * (exp + 1)) 1

-- Theorem for 121
theorem divisors_121 :
  num_divisors [(11, 2)] = 3 := by sorry

-- Theorem for 1000
theorem divisors_1000 :
  num_divisors [(2, 3), (5, 3)] = 16 := by sorry

-- Theorem for 1000000000
theorem divisors_1000000000 :
  num_divisors [(2, 9), (5, 9)] = 100 := by sorry

end NUMINAMATH_CALUDE_divisors_121_divisors_1000_divisors_1000000000_l371_37107


namespace NUMINAMATH_CALUDE_shirt_price_change_l371_37109

theorem shirt_price_change (original_price : ℝ) (decrease_percent : ℝ) : 
  original_price > 0 →
  decrease_percent ≥ 0 →
  (1.15 * original_price) * (1 - decrease_percent / 100) = 97.75 →
  decrease_percent = 0 := by
sorry

end NUMINAMATH_CALUDE_shirt_price_change_l371_37109


namespace NUMINAMATH_CALUDE_regular_triangular_pyramid_volume_regular_triangular_pyramid_volume_is_correct_l371_37181

/-- The volume of a regular triangular pyramid with specific properties -/
theorem regular_triangular_pyramid_volume 
  (r : ℝ) -- Length of the perpendicular from the base of the height to a lateral edge
  (α : ℝ) -- Dihedral angle between the lateral face and the base of the pyramid
  (h1 : 0 < r) -- r is positive
  (h2 : 0 < α) -- α is positive
  (h3 : α < π / 2) -- α is less than 90 degrees
  : ℝ :=
  let volume := (Real.sqrt 3 * r^3 * Real.sqrt ((4 + Real.tan α ^ 2) ^ 3)) / (8 * Real.tan α ^ 2)
  volume

#check regular_triangular_pyramid_volume

theorem regular_triangular_pyramid_volume_is_correct
  (r : ℝ) -- Length of the perpendicular from the base of the height to a lateral edge
  (α : ℝ) -- Dihedral angle between the lateral face and the base of the pyramid
  (h1 : 0 < r) -- r is positive
  (h2 : 0 < α) -- α is positive
  (h3 : α < π / 2) -- α is less than 90 degrees
  : regular_triangular_pyramid_volume r α h1 h2 h3 = 
    (Real.sqrt 3 * r^3 * Real.sqrt ((4 + Real.tan α ^ 2) ^ 3)) / (8 * Real.tan α ^ 2) :=
by sorry

end NUMINAMATH_CALUDE_regular_triangular_pyramid_volume_regular_triangular_pyramid_volume_is_correct_l371_37181


namespace NUMINAMATH_CALUDE_cube_of_sqrt_three_l371_37170

theorem cube_of_sqrt_three (x : ℝ) : 
  Real.sqrt (x + 3) = 3 → (x + 3)^3 = 729 := by
sorry

end NUMINAMATH_CALUDE_cube_of_sqrt_three_l371_37170


namespace NUMINAMATH_CALUDE_balloon_count_l371_37110

theorem balloon_count (initial_balloons : Real) (friend_balloons : Real) 
  (h1 : initial_balloons = 7.0) 
  (h2 : friend_balloons = 5.0) : 
  initial_balloons + friend_balloons = 12.0 := by
  sorry

end NUMINAMATH_CALUDE_balloon_count_l371_37110


namespace NUMINAMATH_CALUDE_total_wings_is_14_l371_37179

/-- Represents the types of birds available for purchase. -/
inductive BirdType
| Parrot
| Pigeon
| Canary

/-- Represents the money received from each grandparent. -/
def grandparentMoney : List ℕ := [45, 60, 55, 50]

/-- Represents the cost of each bird type. -/
def birdCost : BirdType → ℕ
| BirdType.Parrot => 35
| BirdType.Pigeon => 25
| BirdType.Canary => 20

/-- Represents the number of birds in a discounted set for each bird type. -/
def discountSet : BirdType → ℕ
| BirdType.Parrot => 3
| BirdType.Pigeon => 4
| BirdType.Canary => 5

/-- Represents the cost of a discounted set for each bird type. -/
def discountSetCost : BirdType → ℕ
| BirdType.Parrot => 35 * 2 + 35 / 2
| BirdType.Pigeon => 25 * 3
| BirdType.Canary => 20 * 4

/-- Represents the number of wings each bird has. -/
def wingsPerBird : ℕ := 2

/-- Represents the total money John has to spend. -/
def totalMoney : ℕ := grandparentMoney.sum

/-- Theorem stating that the total number of wings of all birds John bought is 14. -/
theorem total_wings_is_14 :
  ∃ (parrot pigeon canary : ℕ),
    parrot > 0 ∧ pigeon > 0 ∧ canary > 0 ∧
    parrot * birdCost BirdType.Parrot +
    pigeon * birdCost BirdType.Pigeon +
    canary * birdCost BirdType.Canary = totalMoney ∧
    (parrot + pigeon + canary) * wingsPerBird = 14 :=
  sorry

end NUMINAMATH_CALUDE_total_wings_is_14_l371_37179


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l371_37175

theorem algebraic_expression_value (a b : ℝ) (h : a - 2*b = -1) :
  1 - 2*a + 4*b = 3 := by sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l371_37175


namespace NUMINAMATH_CALUDE_angle_sixty_degrees_l371_37106

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- State the theorem
theorem angle_sixty_degrees (t : Triangle) 
  (h : t.b^2 + t.c^2 - t.a^2 = t.b * t.c) : 
  t.A = 60 * π / 180 := by
  sorry


end NUMINAMATH_CALUDE_angle_sixty_degrees_l371_37106


namespace NUMINAMATH_CALUDE_stratified_sampling_problem_l371_37178

theorem stratified_sampling_problem (total_students : ℕ) 
  (group1_students : ℕ) (selected_from_group1 : ℕ) (n : ℕ) : 
  total_students = 1230 → 
  group1_students = 480 → 
  selected_from_group1 = 16 → 
  (n : ℚ) / total_students = selected_from_group1 / group1_students → 
  n = 41 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_problem_l371_37178


namespace NUMINAMATH_CALUDE_fair_coin_same_side_four_times_l371_37166

theorem fair_coin_same_side_four_times (p : ℝ) :
  (p = 1 / 2) →                        -- The coin is fair (equal probability for each side)
  (p ^ 4 : ℝ) = 1 / 16 := by            -- Probability of same side 4 times is 1/16
sorry


end NUMINAMATH_CALUDE_fair_coin_same_side_four_times_l371_37166


namespace NUMINAMATH_CALUDE_seed_cost_calculation_l371_37162

def seed_cost_2lb : ℝ := 44.68
def seed_amount : ℝ := 6

theorem seed_cost_calculation : 
  seed_amount * (seed_cost_2lb / 2) = 134.04 := by
  sorry

end NUMINAMATH_CALUDE_seed_cost_calculation_l371_37162


namespace NUMINAMATH_CALUDE_cos_angle_POQ_l371_37153

/-- Given two points P and Q on the unit circle centered at the origin O,
    where P is in the first quadrant with x-coordinate 4/5,
    and Q is in the fourth quadrant with x-coordinate 5/13,
    prove that the cosine of angle POQ is 56/65. -/
theorem cos_angle_POQ (P Q : ℝ × ℝ) : 
  (P.1^2 + P.2^2 = 1) →  -- P is on the unit circle
  (Q.1^2 + Q.2^2 = 1) →  -- Q is on the unit circle
  (P.1 = 4/5) →          -- x-coordinate of P is 4/5
  (P.2 ≥ 0) →            -- P is in the first quadrant
  (Q.1 = 5/13) →         -- x-coordinate of Q is 5/13
  (Q.2 ≤ 0) →            -- Q is in the fourth quadrant
  Real.cos (Real.arccos P.1 + Real.arccos Q.1) = 56/65 := by
  sorry


end NUMINAMATH_CALUDE_cos_angle_POQ_l371_37153


namespace NUMINAMATH_CALUDE_orange_eaters_ratio_l371_37142

/-- Represents a family gathering with a specific number of people and orange eaters. -/
structure FamilyGathering where
  total_people : ℕ
  orange_eaters : ℕ
  h_orange_eaters : orange_eaters = total_people - 10

/-- The ratio of orange eaters to total people in a family gathering is 1:2. -/
theorem orange_eaters_ratio (gathering : FamilyGathering) 
    (h_total : gathering.total_people = 20) : 
    (gathering.orange_eaters : ℚ) / gathering.total_people = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_orange_eaters_ratio_l371_37142


namespace NUMINAMATH_CALUDE_sasha_guessing_game_l371_37182

theorem sasha_guessing_game (X : ℕ) (hX : X ≤ 100) :
  ∃ (questions : List (ℕ × ℕ)),
    questions.length ≤ 7 ∧
    (∀ (M N : ℕ), (M, N) ∈ questions → M < 100 ∧ N < 100) ∧
    ∀ (Y : ℕ), Y ≤ 100 →
      (∀ (M N : ℕ), (M, N) ∈ questions →
        Nat.gcd (X + M) N = Nat.gcd (Y + M) N) →
      X = Y :=
by sorry

end NUMINAMATH_CALUDE_sasha_guessing_game_l371_37182


namespace NUMINAMATH_CALUDE_max_intersections_four_circles_l371_37103

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a plane, represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Predicate to check if a line intersects a circle -/
def intersects (l : Line) (c : Circle) : Prop :=
  sorry

/-- The number of intersection points between a line and a circle -/
def numIntersections (l : Line) (c : Circle) : ℕ :=
  sorry

/-- Predicate to check if four circles are coplanar -/
def coplanar (c1 c2 c3 c4 : Circle) : Prop :=
  sorry

/-- Theorem: The maximum number of intersection points between a line and four coplanar circles is 8 -/
theorem max_intersections_four_circles (c1 c2 c3 c4 : Circle) (l : Line) :
  coplanar c1 c2 c3 c4 →
  intersects l c1 →
  intersects l c2 →
  intersects l c3 →
  intersects l c4 →
  numIntersections l c1 + numIntersections l c2 + numIntersections l c3 + numIntersections l c4 ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_max_intersections_four_circles_l371_37103


namespace NUMINAMATH_CALUDE_abs_value_condition_l371_37102

theorem abs_value_condition (a b : ℝ) (h : ((1 + a * b) / (a + b))^2 < 1) :
  (abs a > 1 ∧ abs b < 1) ∨ (abs a < 1 ∧ abs b > 1) := by
  sorry

end NUMINAMATH_CALUDE_abs_value_condition_l371_37102


namespace NUMINAMATH_CALUDE_train_length_calculation_l371_37159

/-- Given a train passing a bridge, calculate its length -/
theorem train_length_calculation (train_speed : ℝ) (bridge_length : ℝ) (time_to_pass : ℝ) : 
  train_speed = 45 * 1000 / 3600 →
  bridge_length = 140 →
  time_to_pass = 52 →
  (train_speed * time_to_pass - bridge_length) = 510 := by
  sorry

#check train_length_calculation

end NUMINAMATH_CALUDE_train_length_calculation_l371_37159


namespace NUMINAMATH_CALUDE_fifteen_factorial_base_eight_zeroes_l371_37116

/-- The number of trailing zeroes in n! when written in base b -/
def trailingZeroes (n : ℕ) (b : ℕ) : ℕ :=
  sorry

/-- Theorem: 15! ends with 3 zeroes when written in base 8 -/
theorem fifteen_factorial_base_eight_zeroes :
  trailingZeroes 15 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_factorial_base_eight_zeroes_l371_37116


namespace NUMINAMATH_CALUDE_maximal_regions_quadrilaterals_l371_37122

/-- The maximal number of regions created by n convex quadrilaterals in a plane -/
def maxRegions (n : ℕ) : ℕ := 4*n^2 - 4*n + 2

/-- Theorem stating that maxRegions gives the maximal number of regions -/
theorem maximal_regions_quadrilaterals (n : ℕ) :
  ∀ (regions : ℕ), regions ≤ maxRegions n :=
by sorry

end NUMINAMATH_CALUDE_maximal_regions_quadrilaterals_l371_37122


namespace NUMINAMATH_CALUDE_intersection_condition_l371_37185

-- Define the sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x < 2}
def B (a : ℝ) : Set ℝ := {x | x < a}

-- State the theorem
theorem intersection_condition (a : ℝ) : (A ∩ B a).Nonempty → a > -1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_condition_l371_37185


namespace NUMINAMATH_CALUDE_probability_of_double_l371_37150

-- Define the range of integers for the mini-domino set
def dominoRange : ℕ := 7

-- Define a function to calculate the total number of pairings
def totalPairings (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define the number of doubles in the set
def numDoubles : ℕ := dominoRange

-- Theorem statement
theorem probability_of_double :
  (numDoubles : ℚ) / (totalPairings dominoRange : ℚ) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_double_l371_37150


namespace NUMINAMATH_CALUDE_quadratic_real_roots_range_l371_37139

theorem quadratic_real_roots_range (m : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x + m = 0) → m ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_range_l371_37139


namespace NUMINAMATH_CALUDE_min_value_of_3a_plus_2_l371_37117

theorem min_value_of_3a_plus_2 (a : ℝ) (h : 4 * a^2 + 7 * a + 3 = 2) :
  ∃ (m : ℝ), m = 3 * a + 2 ∧ ∀ (x : ℝ), (4 * x^2 + 7 * x + 3 = 2) → m ≤ 3 * x + 2 ∧ m = -1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_3a_plus_2_l371_37117


namespace NUMINAMATH_CALUDE_diner_menu_problem_l371_37134

theorem diner_menu_problem (n : ℕ) (h1 : n > 0) : 
  let vegan_dishes : ℕ := 6
  let vegan_fraction : ℚ := 1 / 6
  let nut_containing_vegan : ℕ := 5
  (vegan_dishes : ℚ) / n = vegan_fraction →
  (vegan_dishes - nut_containing_vegan : ℚ) / n = 1 / 36 := by
  sorry

end NUMINAMATH_CALUDE_diner_menu_problem_l371_37134


namespace NUMINAMATH_CALUDE_f_negation_l371_37118

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x - 2

-- State the theorem
theorem f_negation (a b : ℝ) :
  f a b 2011 = 10 → f a b (-2011) = -14 := by
  sorry

end NUMINAMATH_CALUDE_f_negation_l371_37118


namespace NUMINAMATH_CALUDE_victorias_initial_money_l371_37144

/-- Theorem: Victoria's Initial Money --/
theorem victorias_initial_money (rice_price wheat_price soda_price : ℕ)
  (rice_quantity wheat_quantity : ℕ) (remaining_balance : ℕ) :
  rice_price = 20 →
  wheat_price = 25 →
  soda_price = 150 →
  rice_quantity = 2 →
  wheat_quantity = 3 →
  remaining_balance = 235 →
  rice_quantity * rice_price + wheat_quantity * wheat_price + soda_price + remaining_balance = 500 :=
by
  sorry

#check victorias_initial_money

end NUMINAMATH_CALUDE_victorias_initial_money_l371_37144


namespace NUMINAMATH_CALUDE_bob_age_proof_l371_37127

theorem bob_age_proof (alice_age bob_age charlie_age : ℕ) : 
  (alice_age + 10 = 2 * (bob_age - 10)) →
  (alice_age = bob_age + 7) →
  (charlie_age = (alice_age + bob_age) / 2) →
  bob_age = 37 := by
  sorry

end NUMINAMATH_CALUDE_bob_age_proof_l371_37127


namespace NUMINAMATH_CALUDE_open_box_volume_l371_37126

/-- The volume of an open box formed by cutting squares from the corners of a rectangular sheet. -/
theorem open_box_volume 
  (sheet_length : ℝ) 
  (sheet_width : ℝ) 
  (cut_length : ℝ) 
  (h1 : sheet_length = 48) 
  (h2 : sheet_width = 36) 
  (h3 : cut_length = 8) : 
  (sheet_length - 2 * cut_length) * (sheet_width - 2 * cut_length) * cut_length = 5120 := by
  sorry

#check open_box_volume

end NUMINAMATH_CALUDE_open_box_volume_l371_37126


namespace NUMINAMATH_CALUDE_cake_recipe_flour_l371_37167

/-- The number of cups of flour in a cake recipe -/
def recipe_flour (sugar_cups : ℕ) (flour_added : ℕ) (flour_remaining : ℕ) : ℕ :=
  flour_added + flour_remaining

theorem cake_recipe_flour :
  ∀ (sugar_cups : ℕ) (flour_added : ℕ) (flour_remaining : ℕ),
    sugar_cups = 2 →
    flour_added = 7 →
    flour_remaining = sugar_cups + 1 →
    recipe_flour sugar_cups flour_added flour_remaining = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_cake_recipe_flour_l371_37167


namespace NUMINAMATH_CALUDE_inequality_proof_l371_37191

theorem inequality_proof (a b c d : ℝ) :
  (a^8 + b^3 + c^8 + d^3)^2 ≤ 4 * (a^4 + b^8 + c^8 + d^4) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l371_37191


namespace NUMINAMATH_CALUDE_lcm_1320_924_l371_37186

theorem lcm_1320_924 : Nat.lcm 1320 924 = 9240 := by
  sorry

end NUMINAMATH_CALUDE_lcm_1320_924_l371_37186


namespace NUMINAMATH_CALUDE_A_intersect_B_range_of_a_l371_37198

-- Define the sets A, B, and C
def A : Set ℝ := {x | x < -2 ∨ (3 < x ∧ x < 4)}
def B : Set ℝ := {x | x^2 - 2*x - 15 ≤ 0}
def C (a : ℝ) : Set ℝ := {x | x ≥ a}

-- Theorem 1: Intersection of A and B
theorem A_intersect_B : A ∩ B = {x : ℝ | 3 < x ∧ x ≤ 5} := by sorry

-- Theorem 2: Range of a given B ∩ C = B
theorem range_of_a (h : B ∩ C a = B) : a ≤ -3 := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_range_of_a_l371_37198


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l371_37168

theorem cyclic_sum_inequality (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) : 
  (a / (b + 2*c + 3*d) + b / (c + 2*d + 3*a) + 
   c / (d + 2*a + 3*b) + d / (a + 2*b + 3*c)) ≥ 2/3 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l371_37168


namespace NUMINAMATH_CALUDE_odd_function_condition_l371_37121

def f (x a b : ℝ) : ℝ := x * abs (x + a) + b

theorem odd_function_condition (a b : ℝ) :
  (∀ x, f (-x) a b = -f x a b) ↔ a^2 + b^2 = 0 := by sorry

end NUMINAMATH_CALUDE_odd_function_condition_l371_37121


namespace NUMINAMATH_CALUDE_cubic_square_fraction_inequality_l371_37195

theorem cubic_square_fraction_inequality (s r : ℝ) (hs : s > 0) (hr : r > 0) (hsr : r < s) :
  (s^3 - r^3) / (s^3 + r^3) > (s^2 - r^2) / (s^2 + r^2) := by
  sorry

end NUMINAMATH_CALUDE_cubic_square_fraction_inequality_l371_37195


namespace NUMINAMATH_CALUDE_average_difference_l371_37143

def average (a b : Int) : ℚ := (a + b) / 2

theorem average_difference : 
  average 500 1000 - average 100 500 = 450 := by sorry

end NUMINAMATH_CALUDE_average_difference_l371_37143


namespace NUMINAMATH_CALUDE_total_turnips_l371_37140

def keith_turnips : ℕ := 6
def alyssa_turnips : ℕ := 9

theorem total_turnips : keith_turnips + alyssa_turnips = 15 := by
  sorry

end NUMINAMATH_CALUDE_total_turnips_l371_37140


namespace NUMINAMATH_CALUDE_max_integer_a_for_real_roots_l371_37124

theorem max_integer_a_for_real_roots : 
  ∀ a : ℤ, 
  (a ≠ 1) → 
  (∃ x : ℝ, (a - 1) * x^2 - 2 * x + 3 = 0) → 
  a ≤ 0 ∧ 
  ∀ b : ℤ, b > 0 → ¬(∃ x : ℝ, (b - 1) * x^2 - 2 * x + 3 = 0) :=
by sorry

end NUMINAMATH_CALUDE_max_integer_a_for_real_roots_l371_37124


namespace NUMINAMATH_CALUDE_euler_totient_equation_solutions_l371_37169

/-- Euler's totient function -/
def phi (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem euler_totient_equation_solutions (a b : ℕ) :
  (a > 0 ∧ b > 0 ∧ 14 * (phi a)^2 - phi (a * b) + 22 * (phi b)^2 = a^2 + b^2) ↔
  (∃ x y : ℕ, a = 30 * 2^x * 3^y ∧ b = 6 * 2^x * 3^y) :=
sorry

end NUMINAMATH_CALUDE_euler_totient_equation_solutions_l371_37169


namespace NUMINAMATH_CALUDE_line_in_plane_if_points_in_plane_l371_37154

-- Define the types for our geometric objects
variable (α : Type) [LinearOrderedField α]
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the relationships between geometric objects
variable (on_line : Point → Line → Prop)
variable (on_plane : Point → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)

-- State the theorem
theorem line_in_plane_if_points_in_plane 
  (a b l : Line) (α : Plane) (M N : Point) :
  line_in_plane a α →
  line_in_plane b α →
  on_line M a →
  on_line N b →
  on_line M l →
  on_line N l →
  line_in_plane l α :=
sorry

end NUMINAMATH_CALUDE_line_in_plane_if_points_in_plane_l371_37154


namespace NUMINAMATH_CALUDE_cistern_filling_time_l371_37161

-- Define the filling rates of pipes p and q
def fill_rate_p : ℚ := 1 / 10
def fill_rate_q : ℚ := 1 / 15

-- Define the time both pipes are open together
def initial_time : ℚ := 4

-- Define the total capacity of the cistern
def total_capacity : ℚ := 1

-- Theorem statement
theorem cistern_filling_time :
  let filled_initially := (fill_rate_p + fill_rate_q) * initial_time
  let remaining_to_fill := total_capacity - filled_initially
  let remaining_time := remaining_to_fill / fill_rate_q
  remaining_time = 5 := by sorry

end NUMINAMATH_CALUDE_cistern_filling_time_l371_37161


namespace NUMINAMATH_CALUDE_total_cds_on_shelf_l371_37197

/-- The number of CDs that a single rack can hold -/
def cds_per_rack : ℕ := 8

/-- The number of racks that can fit on a shelf -/
def racks_per_shelf : ℕ := 4

/-- Theorem: The total number of CDs that can fit on a shelf is 32 -/
theorem total_cds_on_shelf : cds_per_rack * racks_per_shelf = 32 := by
  sorry

end NUMINAMATH_CALUDE_total_cds_on_shelf_l371_37197


namespace NUMINAMATH_CALUDE_find_divisor_l371_37192

theorem find_divisor : ∃ (d : ℕ), d > 0 ∧ 
  (13603 - 31) % d = 0 ∧
  (∀ (n : ℕ), n < 31 → (13603 - n) % d ≠ 0) ∧
  d = 13572 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l371_37192


namespace NUMINAMATH_CALUDE_union_complement_equality_l371_37184

def U : Finset Nat := {1, 2, 3, 4, 5}
def M : Finset Nat := {1, 4}
def N : Finset Nat := {2, 5}

theorem union_complement_equality : N ∪ (U \ M) = {2, 3, 5} := by
  sorry

end NUMINAMATH_CALUDE_union_complement_equality_l371_37184


namespace NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l371_37137

theorem least_positive_integer_with_remainders : ∃ (M : ℕ), 
  (M > 0) ∧
  (M % 6 = 5) ∧
  (M % 8 = 7) ∧
  (M % 9 = 8) ∧
  (M % 11 = 10) ∧
  (M % 12 = 11) ∧
  (M % 13 = 12) ∧
  (∀ (N : ℕ), N > 0 ∧ 
    N % 6 = 5 ∧
    N % 8 = 7 ∧
    N % 9 = 8 ∧
    N % 11 = 10 ∧
    N % 12 = 11 ∧
    N % 13 = 12 → M ≤ N) ∧
  M = 10163 :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l371_37137


namespace NUMINAMATH_CALUDE_right_triangle_third_side_l371_37101

theorem right_triangle_third_side : ∀ a b c : ℝ,
  a = 3 → b = 4 →
  (a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) →
  c = Real.sqrt 7 ∨ c = 5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_l371_37101


namespace NUMINAMATH_CALUDE_gift_arrangement_count_gift_arrangement_proof_l371_37160

theorem gift_arrangement_count : ℕ → ℕ → ℕ
  | 5, 4 => 120
  | _, _ => 0

theorem gift_arrangement_proof (n m : ℕ) (hn : n = 5) (hm : m = 4) :
  gift_arrangement_count n m = (n.choose 1) * m.factorial :=
by sorry

end NUMINAMATH_CALUDE_gift_arrangement_count_gift_arrangement_proof_l371_37160


namespace NUMINAMATH_CALUDE_infinite_sum_solution_l371_37149

theorem infinite_sum_solution (k : ℝ) (h1 : k > 2) 
  (h2 : (∑' n, (6 * n + 2) / k^n) = 15) : 
  k = (38 + 2 * Real.sqrt 46) / 30 := by
sorry

end NUMINAMATH_CALUDE_infinite_sum_solution_l371_37149


namespace NUMINAMATH_CALUDE_QR_distance_l371_37196

-- Define the right triangle DEF
structure RightTriangle where
  DE : ℝ
  EF : ℝ
  DF : ℝ
  is_right_triangle : DE^2 + EF^2 = DF^2

-- Define the circles
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the problem setup
def problem_setup (t : RightTriangle) (Q R : Circle) : Prop :=
  t.DE = 9 ∧ t.EF = 12 ∧ t.DF = 15 ∧
  Q.center.2 = t.EF ∧ 
  R.center.1 = 0 ∧
  Q.radius = t.DE ∧
  R.radius = t.EF

-- Theorem statement
theorem QR_distance (t : RightTriangle) (Q R : Circle) 
  (h : problem_setup t Q R) : 
  Real.sqrt ((Q.center.1 - R.center.1)^2 + (Q.center.2 - R.center.2)^2) = 15 :=
sorry

end NUMINAMATH_CALUDE_QR_distance_l371_37196


namespace NUMINAMATH_CALUDE_equilateral_triangle_intersections_l371_37120

/-- Represents a point on a side of the triangle -/
structure DivisionPoint where
  side : Fin 3
  position : Fin 11

/-- Represents a line segment from a vertex to a division point -/
structure Segment where
  vertex : Fin 3
  endpoint : DivisionPoint

/-- The number of intersection points in the described configuration -/
def intersection_points : ℕ := 301

/-- States that the number of intersection points in the described triangle configuration is 301 -/
theorem equilateral_triangle_intersections :
  ∀ (triangle : Type) (is_equilateral : triangle → Prop) 
    (divide_sides : triangle → Fin 3 → Fin 12 → DivisionPoint)
    (connect_vertices : triangle → Segment → Prop),
  (∃ (t : triangle), is_equilateral t ∧ 
    (∀ (s : Fin 3) (p : Fin 12), ∃ (dp : DivisionPoint), divide_sides t s p = dp) ∧
    (∀ (v : Fin 3) (dp : DivisionPoint), v ≠ dp.side → connect_vertices t ⟨v, dp⟩)) →
  (∃ (intersection_count : ℕ), intersection_count = intersection_points) :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_intersections_l371_37120


namespace NUMINAMATH_CALUDE_ellipse_intersection_theorem_l371_37180

/-- Definition of the ellipse C -/
def C (x y : ℝ) : Prop := x^2 + y^2/4 = 1

/-- Definition of the line that intersects C -/
def L (k : ℝ) (x y : ℝ) : Prop := y = k*x + 1

/-- Definition of points A and B as intersections of C and L -/
def intersectionPoints (k : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), C x₁ y₁ ∧ C x₂ y₂ ∧ L k x₁ y₁ ∧ L k x₂ y₂

/-- Condition for OA ⊥ OB -/
def perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁*x₂ + y₁*y₂ = 0

/-- Theorem stating the conditions for perpendicularity and the length of AB -/
theorem ellipse_intersection_theorem :
  ∀ k : ℝ, intersectionPoints k →
    (k = 1/2 ∨ k = -1/2) ↔
      (∃ x₁ y₁ x₂ y₂ : ℝ, C x₁ y₁ ∧ C x₂ y₂ ∧ L k x₁ y₁ ∧ L k x₂ y₂ ∧
        perpendicular x₁ y₁ x₂ y₂ ∧
        ((x₁ - x₂)^2 + (y₁ - y₂)^2)^(1/2) = 4*(65^(1/2))/17) :=
by sorry


end NUMINAMATH_CALUDE_ellipse_intersection_theorem_l371_37180


namespace NUMINAMATH_CALUDE_division_problem_l371_37156

theorem division_problem : ∃ (d r : ℕ), d > 0 ∧ 1270 = 74 * d + r ∧ r < d := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_division_problem_l371_37156


namespace NUMINAMATH_CALUDE_set_intersection_difference_l371_37193

theorem set_intersection_difference (S T : Set ℕ) (a b : ℕ) :
  S = {1, 2, a} →
  T = {2, 3, 4, b} →
  S ∩ T = {1, 2, 3} →
  a - b = 2 := by
sorry

end NUMINAMATH_CALUDE_set_intersection_difference_l371_37193


namespace NUMINAMATH_CALUDE_croissant_distribution_l371_37155

theorem croissant_distribution (total : Nat) (neighbors : Nat) (h1 : total = 59) (h2 : neighbors = 8) :
  total - (neighbors * (total / neighbors)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_croissant_distribution_l371_37155


namespace NUMINAMATH_CALUDE_geometric_sequence_proof_l371_37172

theorem geometric_sequence_proof :
  let a : ℚ := 3
  let r : ℚ := 8 / 27
  let sequence : ℕ → ℚ := λ n => a * r ^ (n - 1)
  (sequence 1 = 3) ∧ 
  (sequence 2 = 8 / 9) ∧ 
  (sequence 3 = 32 / 81) :=
by
  sorry

#check geometric_sequence_proof

end NUMINAMATH_CALUDE_geometric_sequence_proof_l371_37172


namespace NUMINAMATH_CALUDE_means_inequality_l371_37112

theorem means_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c)
  (h_max : max b c ≥ (a + b) / 2) : 
  Real.sqrt ((b^2 + c^2) / 2) > (a + b) / 2 ∧ 
  (a + b) / 2 > Real.sqrt (a * b) ∧ 
  Real.sqrt (a * b) > 2 * a * b / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_means_inequality_l371_37112


namespace NUMINAMATH_CALUDE_least_number_divisible_by_five_primes_l371_37164

theorem least_number_divisible_by_five_primes : ℕ := by
  -- Define the property of being divisible by five different primes
  let divisible_by_five_primes (n : ℕ) :=
    ∃ (p₁ p₂ p₃ p₄ p₅ : ℕ),
      Nat.Prime p₁ ∧ Nat.Prime p₂ ∧ Nat.Prime p₃ ∧ Nat.Prime p₄ ∧ Nat.Prime p₅ ∧
      p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₁ ≠ p₅ ∧
      p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₂ ≠ p₅ ∧
      p₃ ≠ p₄ ∧ p₃ ≠ p₅ ∧
      p₄ ≠ p₅ ∧
      n % p₁ = 0 ∧ n % p₂ = 0 ∧ n % p₃ = 0 ∧ n % p₄ = 0 ∧ n % p₅ = 0

  -- State that 2310 is divisible by five different primes
  have h1 : divisible_by_five_primes 2310 := by sorry

  -- State that 2310 is the least such number
  have h2 : ∀ m : ℕ, m < 2310 → ¬(divisible_by_five_primes m) := by sorry

  -- Conclude that 2310 is the answer
  exact 2310

end NUMINAMATH_CALUDE_least_number_divisible_by_five_primes_l371_37164


namespace NUMINAMATH_CALUDE_binomial_coefficient_problem_l371_37176

theorem binomial_coefficient_problem (a : ℝ) : 
  (Finset.range 11).sum (λ k => Nat.choose 10 k * a^(10 - k) * (if k = 3 then 1 else 0)) = 15 → 
  a = 1/2 := by
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_problem_l371_37176


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l371_37190

theorem intersection_implies_a_value (P Q : Set ℕ) (a : ℕ) :
  P = {0, a} →
  Q = {1, 2} →
  (P ∩ Q).Nonempty →
  a = 1 ∨ a = 2 := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l371_37190


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l371_37187

/-- Given an arithmetic sequence {a_n} where a₁₀ = 30 and a₂₀ = 50,
    the general term is a_n = 2n + 10 -/
theorem arithmetic_sequence_general_term
  (a : ℕ → ℝ)  -- The sequence
  (h_arith : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))  -- Arithmetic sequence condition
  (h_10 : a 10 = 30)  -- Given condition
  (h_20 : a 20 = 50)  -- Given condition
  : ∀ n : ℕ, a n = 2 * n + 10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l371_37187


namespace NUMINAMATH_CALUDE_toms_climbing_time_l371_37114

/-- Proves that Tom's climbing time is 2 hours given the conditions -/
theorem toms_climbing_time (elizabeth_time : ℕ) (tom_factor : ℕ) :
  elizabeth_time = 30 →
  tom_factor = 4 →
  (elizabeth_time * tom_factor : ℚ) / 60 = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_toms_climbing_time_l371_37114


namespace NUMINAMATH_CALUDE_least_multiple_first_ten_gt_1000_l371_37111

theorem least_multiple_first_ten_gt_1000 : ∃ n : ℕ,
  n > 1000 ∧
  (∀ k : ℕ, k ≤ 10 → k > 0 → n % k = 0) ∧
  (∀ m : ℕ, m > 1000 ∧ (∀ k : ℕ, k ≤ 10 → k > 0 → m % k = 0) → m ≥ n) ∧
  n = 2520 := by
sorry

end NUMINAMATH_CALUDE_least_multiple_first_ten_gt_1000_l371_37111


namespace NUMINAMATH_CALUDE_peter_has_winning_strategy_l371_37136

/-- Represents the possible moves in the game -/
inductive Move
  | Single : Nat → Nat → Move  -- 1x1
  | HorizontalRect : Nat → Nat → Move  -- 1x2
  | VerticalRect : Nat → Nat → Move  -- 2x1
  | Square : Nat → Nat → Move  -- 2x2

/-- Represents the game state -/
structure GameState where
  board : Matrix (Fin 8) (Fin 8) Bool
  currentPlayer : Bool  -- true for Peter, false for Victor

/-- Checks if a move is valid in the current game state -/
def isValidMove (state : GameState) (move : Move) : Bool :=
  sorry

/-- Applies a move to the current game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  sorry

/-- Checks if the game is over (no valid moves left) -/
def isGameOver (state : GameState) : Bool :=
  sorry

/-- Represents a strategy for playing the game -/
def Strategy := GameState → Move

/-- The symmetry strategy for Peter -/
def symmetryStrategy : Strategy :=
  sorry

/-- Theorem: Peter has a winning strategy -/
theorem peter_has_winning_strategy :
  ∃ (strategy : Strategy),
    ∀ (game : GameState),
      game.currentPlayer = true →  -- Peter's turn
      ¬(isGameOver game) →
      ∃ (move : Move),
        isValidMove game move ∧
        ¬(isGameOver (applyMove game move)) ∧
        ∀ (victor_move : Move),
          isValidMove (applyMove game move) victor_move →
          ¬(isGameOver (applyMove (applyMove game move) victor_move)) →
          ∃ (peter_response : Move),
            isValidMove (applyMove (applyMove game move) victor_move) peter_response ∧
            strategy (applyMove (applyMove game move) victor_move) = peter_response :=
  sorry

end NUMINAMATH_CALUDE_peter_has_winning_strategy_l371_37136


namespace NUMINAMATH_CALUDE_income_comparison_l371_37141

theorem income_comparison (juan tim mary : ℝ) 
  (h1 : tim = juan * 0.7)
  (h2 : mary = juan * 1.12) :
  (mary - tim) / tim = 0.6 :=
sorry

end NUMINAMATH_CALUDE_income_comparison_l371_37141


namespace NUMINAMATH_CALUDE_coefficient_of_3x2y_l371_37157

/-- The coefficient of a monomial is the numerical factor that multiplies the variable part. -/
def coefficient (m : ℕ → ℕ → ℚ) : ℚ := m 0 0

/-- A monomial is represented as a function from ℕ × ℕ to ℚ, where m i j represents the coefficient of x^i * y^j. -/
def monomial_3x2y : ℕ → ℕ → ℚ := fun i j => if i = 2 ∧ j = 1 then 3 else 0

theorem coefficient_of_3x2y :
  coefficient monomial_3x2y = 3 := by sorry

end NUMINAMATH_CALUDE_coefficient_of_3x2y_l371_37157


namespace NUMINAMATH_CALUDE_overlapping_squares_area_l371_37135

/-- Represents a square sheet of paper -/
structure Square :=
  (side : ℝ)

/-- Represents the rotation of a square -/
inductive Rotation
  | NoRotation
  | Rotation45
  | Rotation90

/-- Represents a stack of rotated squares -/
structure RotatedSquares :=
  (bottom : Square)
  (middle : Square)
  (top : Square)
  (middleRotation : Rotation)
  (topRotation : Rotation)

/-- Calculates the area of the resulting shape formed by overlapping rotated squares -/
def resultingArea (rs : RotatedSquares) : ℝ :=
  sorry

theorem overlapping_squares_area :
  ∀ (rs : RotatedSquares),
    rs.bottom.side = 8 ∧
    rs.middle.side = 8 ∧
    rs.top.side = 8 ∧
    rs.middleRotation = Rotation.Rotation45 ∧
    rs.topRotation = Rotation.Rotation90 →
    resultingArea rs = 192 :=
  sorry

end NUMINAMATH_CALUDE_overlapping_squares_area_l371_37135


namespace NUMINAMATH_CALUDE_family_trip_eggs_l371_37158

/-- Calculates the total number of boiled eggs prepared for a family trip -/
def total_eggs (num_adults num_girls num_boys : ℕ) (eggs_per_adult : ℕ) (eggs_per_girl : ℕ) : ℕ :=
  num_adults * eggs_per_adult + num_girls * eggs_per_girl + num_boys * (eggs_per_girl + 1)

/-- Theorem stating that the total number of boiled eggs for the given family trip is 36 -/
theorem family_trip_eggs :
  total_eggs 3 7 10 3 1 = 36 := by
  sorry

end NUMINAMATH_CALUDE_family_trip_eggs_l371_37158


namespace NUMINAMATH_CALUDE_system_solution_unique_l371_37130

theorem system_solution_unique : 
  ∃! (x y : ℝ), x + y = 2 ∧ x + 2*y = 3 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l371_37130


namespace NUMINAMATH_CALUDE_quadratic_root_ratio_l371_37189

theorem quadratic_root_ratio (k : ℝ) : 
  (∃ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ x / y = 3 ∧ 
   x^2 + 10*x + k = 0 ∧ y^2 + 10*y + k = 0) →
  k = 18.75 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_ratio_l371_37189


namespace NUMINAMATH_CALUDE_square_difference_65_35_l371_37174

theorem square_difference_65_35 : 65^2 - 35^2 = 3000 := by sorry

end NUMINAMATH_CALUDE_square_difference_65_35_l371_37174


namespace NUMINAMATH_CALUDE_trapezoid_mn_length_l371_37113

/-- Represents a trapezoid ABCD with point M on diagonal AC and point N on diagonal BD -/
structure Trapezoid where
  /-- Length of base AD -/
  ad : ℝ
  /-- Length of base BC -/
  bc : ℝ
  /-- Ratio of AM to MC on diagonal AC -/
  am_mc_ratio : ℝ × ℝ
  /-- Length of segment MN -/
  mn : ℝ

/-- Theorem stating the length of MN in the given trapezoid configuration -/
theorem trapezoid_mn_length (t : Trapezoid) :
  t.ad = 3 ∧ t.bc = 18 ∧ t.am_mc_ratio = (1, 2) → t.mn = 4 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_mn_length_l371_37113


namespace NUMINAMATH_CALUDE_weighted_average_price_approximation_l371_37177

def large_bottles : ℕ := 1365
def small_bottles : ℕ := 720
def medium_bottles : ℕ := 450
def extra_large_bottles : ℕ := 275

def large_price : ℚ := 189 / 100
def small_price : ℚ := 142 / 100
def medium_price : ℚ := 162 / 100
def extra_large_price : ℚ := 209 / 100

def total_bottles : ℕ := large_bottles + small_bottles + medium_bottles + extra_large_bottles

def total_cost : ℚ := 
  large_bottles * large_price + 
  small_bottles * small_price + 
  medium_bottles * medium_price + 
  extra_large_bottles * extra_large_price

def weighted_average_price : ℚ := total_cost / total_bottles

theorem weighted_average_price_approximation : 
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/100 ∧ |weighted_average_price - 175/100| < ε :=
sorry

end NUMINAMATH_CALUDE_weighted_average_price_approximation_l371_37177


namespace NUMINAMATH_CALUDE_extra_workers_for_deeper_hole_extra_workers_needed_l371_37183

/-- Represents the number of workers needed for a digging task. -/
def workers_needed (initial_workers : ℕ) (initial_depth : ℕ) (initial_hours : ℕ) 
                   (target_depth : ℕ) (target_hours : ℕ) : ℕ :=
  (initial_workers * initial_hours * target_depth) / (initial_depth * target_hours)

/-- Theorem stating the number of workers needed for the new digging task. -/
theorem extra_workers_for_deeper_hole 
  (initial_workers : ℕ) (initial_depth : ℕ) (initial_hours : ℕ)
  (target_depth : ℕ) (target_hours : ℕ) :
  initial_workers = 45 → 
  initial_depth = 30 → 
  initial_hours = 8 → 
  target_depth = 70 → 
  target_hours = 5 → 
  workers_needed initial_workers initial_depth initial_hours target_depth target_hours = 168 :=
by
  sorry

/-- Calculates the extra workers needed based on the initial and required number of workers. -/
def extra_workers (initial : ℕ) (required : ℕ) : ℕ :=
  required - initial

/-- Theorem stating the number of extra workers needed for the new digging task. -/
theorem extra_workers_needed 
  (initial_workers : ℕ) (required_workers : ℕ) :
  initial_workers = 45 →
  required_workers = 168 →
  extra_workers initial_workers required_workers = 123 :=
by
  sorry

end NUMINAMATH_CALUDE_extra_workers_for_deeper_hole_extra_workers_needed_l371_37183


namespace NUMINAMATH_CALUDE_sum_abc_is_zero_l371_37171

theorem sum_abc_is_zero (a b c : ℝ) 
  (h1 : (a + b) / c = (b + c) / a) 
  (h2 : (b + c) / a = (a + c) / b) 
  (h3 : b ≠ c) : 
  a + b + c = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_abc_is_zero_l371_37171


namespace NUMINAMATH_CALUDE_rectangle_ratio_l371_37129

/-- A rectangle with a circle passing through two vertices and touching one side. -/
structure RectangleWithCircle where
  /-- Length of the longer side of the rectangle -/
  x : ℝ
  /-- Length of the shorter side of the rectangle -/
  y : ℝ
  /-- Radius of the circle -/
  R : ℝ
  /-- The perimeter of the rectangle is 4 times the radius of the circle -/
  h_perimeter : x + y = 2 * R
  /-- The circle passes through two vertices and touches one side -/
  h_circle_touch : y = R + Real.sqrt (R^2 - (x/2)^2)
  /-- The sides are positive -/
  h_positive : x > 0 ∧ y > 0 ∧ R > 0

/-- The ratio of the sides of the rectangle is 4:1 -/
theorem rectangle_ratio (rect : RectangleWithCircle) : rect.x / rect.y = 4 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l371_37129


namespace NUMINAMATH_CALUDE_isosceles_triangle_l371_37173

/-- A triangle with sides a, b, c exists -/
def triangle_exists (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- Triangle PQR with sides p, q, r -/
structure Triangle (p q r : ℝ) : Type :=
  (exists_triangle : triangle_exists p q r)

/-- For any positive integer n, a triangle with sides p^n, q^n, r^n exists -/
def power_triangle_exists (p q r : ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → triangle_exists (p^n) (q^n) (r^n)

/-- Main theorem: If a triangle PQR with sides p, q, r exists, and for any positive integer n,
    a triangle with sides p^n, q^n, r^n also exists, then at least two sides of triangle PQR are equal -/
theorem isosceles_triangle (p q r : ℝ) (tr : Triangle p q r) 
    (h : power_triangle_exists p q r) : 
    p = q ∨ q = r ∨ r = p :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_l371_37173


namespace NUMINAMATH_CALUDE_distilled_water_remaining_l371_37151

/-- Represents a mixed number as a pair of integers (whole, fraction) -/
structure MixedNumber where
  whole : Int
  numerator : Int
  denominator : Int
  denom_pos : denominator > 0

/-- Converts a MixedNumber to a rational number -/
def mixedToRational (m : MixedNumber) : Rat :=
  m.whole + (m.numerator : Rat) / (m.denominator : Rat)

theorem distilled_water_remaining
  (initial : MixedNumber)
  (used : MixedNumber)
  (h_initial : initial = ⟨3, 1, 2, by norm_num⟩)
  (h_used : used = ⟨1, 3, 4, by norm_num⟩) :
  mixedToRational initial - mixedToRational used = 7/4 := by
  sorry

#check distilled_water_remaining

end NUMINAMATH_CALUDE_distilled_water_remaining_l371_37151


namespace NUMINAMATH_CALUDE_prime_pairs_satisfying_equation_l371_37138

theorem prime_pairs_satisfying_equation : 
  ∀ x y : ℕ, 
    Prime x → Prime y → 
    (x^2 - y^2 = x * y^2 - 19) ↔ 
    ((x = 2 ∧ y = 3) ∨ (x = 2 ∧ y = 7)) :=
by sorry

end NUMINAMATH_CALUDE_prime_pairs_satisfying_equation_l371_37138


namespace NUMINAMATH_CALUDE_males_in_band_not_in_orchestra_l371_37105

/-- Given the information about band and orchestra membership, prove that the number of males in the band who are not in the orchestra is 10. -/
theorem males_in_band_not_in_orchestra : 
  ∀ (female_band male_band female_orch male_orch female_both total_students : ℕ),
    female_band = 100 →
    male_band = 80 →
    female_orch = 80 →
    male_orch = 100 →
    female_both = 60 →
    total_students = 230 →
    ∃ (male_both : ℕ),
      female_band + female_orch - female_both + male_band + male_orch - male_both = total_students ∧
      male_band - male_both = 10 :=
by sorry

end NUMINAMATH_CALUDE_males_in_band_not_in_orchestra_l371_37105


namespace NUMINAMATH_CALUDE_cube_sum_equals_linear_sum_l371_37108

theorem cube_sum_equals_linear_sum (a b : ℝ) 
  (h : a / (1 + b) + b / (1 + a) = 1) : 
  a^3 + b^3 = a + b := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_equals_linear_sum_l371_37108


namespace NUMINAMATH_CALUDE_three_roles_four_people_l371_37125

def number_of_assignments (n : ℕ) (k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.factorial n / Nat.factorial (n - k)

theorem three_roles_four_people :
  number_of_assignments 4 3 = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_three_roles_four_people_l371_37125


namespace NUMINAMATH_CALUDE_unique_solution_for_x_l371_37100

theorem unique_solution_for_x (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x + 1 / y = 15)
  (h2 : y + 1 / x = 7 / 20)
  (h3 : x * y = 2) :
  x = 10 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_x_l371_37100


namespace NUMINAMATH_CALUDE_sum_first_23_equals_11_l371_37119

def repeatingSequence : List Int := [4, -3, 2, -1, 0]

def sumFirstN (n : Nat) : Int :=
  let fullCycles := n / repeatingSequence.length
  let remainder := n % repeatingSequence.length
  fullCycles * repeatingSequence.sum +
    (repeatingSequence.take remainder).sum

theorem sum_first_23_equals_11 : sumFirstN 23 = 11 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_23_equals_11_l371_37119


namespace NUMINAMATH_CALUDE_max_value_f_on_interval_l371_37115

def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 2

theorem max_value_f_on_interval :
  ∃ (c : ℝ), c ∈ Set.Icc (-1) 1 ∧
  ∀ (x : ℝ), x ∈ Set.Icc (-1) 1 → f x ≤ f c ∧
  f c = 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_f_on_interval_l371_37115


namespace NUMINAMATH_CALUDE_problem_statement_l371_37146

theorem problem_statement (a b c : ℝ) (h : a^3 + a*b + a*c < 0) : b^5 - 4*a*c > 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l371_37146
