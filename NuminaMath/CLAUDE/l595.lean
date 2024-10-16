import Mathlib

namespace NUMINAMATH_CALUDE_two_boxes_in_case_l595_59503

/-- The number of boxes in a case, given the total number of blocks and blocks per box -/
def boxes_in_case (total_blocks : ℕ) (blocks_per_box : ℕ) : ℕ :=
  total_blocks / blocks_per_box

/-- Theorem: There are 2 boxes in a case when there are 12 blocks in total and 6 blocks per box -/
theorem two_boxes_in_case :
  boxes_in_case 12 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_boxes_in_case_l595_59503


namespace NUMINAMATH_CALUDE_twenty_four_point_game_l595_59587

theorem twenty_four_point_game (Q : ℕ) (h : Q = 12) : 
  (Q * 9) - (Q * 7) = 24 := by
  sorry

end NUMINAMATH_CALUDE_twenty_four_point_game_l595_59587


namespace NUMINAMATH_CALUDE_sin_330_degrees_l595_59512

theorem sin_330_degrees : Real.sin (330 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_330_degrees_l595_59512


namespace NUMINAMATH_CALUDE_souvenir_distribution_solution_l595_59582

/-- Represents the souvenir distribution problem -/
structure SouvenirDistribution where
  total_items : ℕ
  total_cost : ℕ
  type_a_cost : ℕ
  type_a_price : ℕ
  type_b_cost : ℕ
  type_b_price : ℕ

/-- Theorem stating the solution to the souvenir distribution problem -/
theorem souvenir_distribution_solution (sd : SouvenirDistribution)
  (h1 : sd.total_items = 100)
  (h2 : sd.total_cost = 6200)
  (h3 : sd.type_a_cost = 50)
  (h4 : sd.type_a_price = 100)
  (h5 : sd.type_b_cost = 70)
  (h6 : sd.type_b_price = 90) :
  ∃ (type_a type_b : ℕ),
    type_a + type_b = sd.total_items ∧
    type_a * sd.type_a_cost + type_b * sd.type_b_cost = sd.total_cost ∧
    type_a = 40 ∧
    type_b = 60 ∧
    (type_a * (sd.type_a_price - sd.type_a_cost) + type_b * (sd.type_b_price - sd.type_b_cost)) = 3200 :=
by
  sorry

end NUMINAMATH_CALUDE_souvenir_distribution_solution_l595_59582


namespace NUMINAMATH_CALUDE_P_k_at_neg_half_is_zero_l595_59563

/-- The unique polynomial P_k such that P_k(n) = 1^k + 2^k + 3^k + ... + n^k for each positive integer n -/
noncomputable def P_k (k : ℕ+) : ℝ → ℝ :=
  sorry

/-- For any positive integer k, P_k(-1/2) = 0 -/
theorem P_k_at_neg_half_is_zero (k : ℕ+) : P_k k (-1/2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_P_k_at_neg_half_is_zero_l595_59563


namespace NUMINAMATH_CALUDE_lawnmower_blade_cost_l595_59540

/-- The cost of a single lawnmower blade -/
def blade_cost : ℝ := sorry

/-- The number of lawnmower blades purchased -/
def num_blades : ℕ := 4

/-- The cost of the weed eater string -/
def string_cost : ℝ := 7

/-- The total cost of supplies -/
def total_cost : ℝ := 39

/-- Theorem stating that the cost of each lawnmower blade is $8 -/
theorem lawnmower_blade_cost : 
  blade_cost = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_lawnmower_blade_cost_l595_59540


namespace NUMINAMATH_CALUDE_max_k_value_l595_59569

theorem max_k_value (k : ℝ) : 
  (∃ x y : ℝ, x^2 + k*x + 17 = 0 ∧ y^2 + k*y + 17 = 0 ∧ |x - y| = Real.sqrt 85) →
  k ≤ Real.sqrt 153 :=
sorry

end NUMINAMATH_CALUDE_max_k_value_l595_59569


namespace NUMINAMATH_CALUDE_square_side_length_l595_59527

theorem square_side_length (area : ℚ) (side : ℚ) : 
  area = 9/16 → side^2 = area → side = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l595_59527


namespace NUMINAMATH_CALUDE_product_of_reciprocal_differences_l595_59558

theorem product_of_reciprocal_differences (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a + b + c = 1) : 
  (1/a - 1) * (1/b - 1) * (1/c - 1) ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_product_of_reciprocal_differences_l595_59558


namespace NUMINAMATH_CALUDE_rainfall_volume_calculation_l595_59526

/-- Calculates the total rainfall volume given rainfall rates and area --/
def total_rainfall_volume (rate1 rate2 : ℝ) (area : ℝ) : ℝ :=
  (rate1 * area + rate2 * area) * 0.001

theorem rainfall_volume_calculation :
  let rate1 : ℝ := 5  -- mm/hour
  let rate2 : ℝ := 10 -- mm/hour
  let area : ℝ := 100 -- square meters
  total_rainfall_volume rate1 rate2 area = 1.5 := by
sorry

end NUMINAMATH_CALUDE_rainfall_volume_calculation_l595_59526


namespace NUMINAMATH_CALUDE_no_positive_integer_solutions_l595_59500

theorem no_positive_integer_solutions : 
  ¬∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x^4 * y^4 - 14 * x^2 * y^2 + 49 = 0 :=
by sorry

end NUMINAMATH_CALUDE_no_positive_integer_solutions_l595_59500


namespace NUMINAMATH_CALUDE_magicians_marbles_l595_59592

/-- The number of marbles left after the magician's trick --/
def marbles_left (red_initial blue_initial green_initial yellow_initial : ℕ) : ℕ :=
  let red_removed := red_initial / 4
  let blue_removed := 3 * (green_initial / 5)
  let green_removed := (green_initial * 3) / 10  -- 30% rounded down
  let yellow_removed := 25

  let red_left := red_initial - red_removed
  let blue_left := blue_initial - blue_removed
  let green_left := green_initial - green_removed
  let yellow_left := yellow_initial - yellow_removed

  red_left + blue_left + green_left + yellow_left

/-- Theorem stating that given the initial conditions, the number of marbles left is 213 --/
theorem magicians_marbles :
  marbles_left 80 120 75 50 = 213 :=
by sorry

end NUMINAMATH_CALUDE_magicians_marbles_l595_59592


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l595_59596

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  (X^3 + 3*X^2 - 4 : Polynomial ℝ) = (X^2 + X - 2 : Polynomial ℝ) * q + 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l595_59596


namespace NUMINAMATH_CALUDE_farm_animals_after_addition_l595_59589

/-- Represents the farm with its animals -/
structure Farm :=
  (cows : ℕ)
  (pigs : ℕ)
  (goats : ℕ)

/-- Calculates the total number of animals on the farm -/
def Farm.total (f : Farm) : ℕ := f.cows + f.pigs + f.goats

/-- Adds new animals to the farm -/
def Farm.add (f : Farm) (new_cows new_pigs new_goats : ℕ) : Farm :=
  { cows := f.cows + new_cows,
    pigs := f.pigs + new_pigs,
    goats := f.goats + new_goats }

/-- Theorem: The farm will have 21 animals after adding the new ones -/
theorem farm_animals_after_addition :
  let initial_farm := Farm.mk 2 3 6
  let final_farm := initial_farm.add 3 5 2
  final_farm.total = 21 := by sorry

end NUMINAMATH_CALUDE_farm_animals_after_addition_l595_59589


namespace NUMINAMATH_CALUDE_arrangement_count_l595_59561

theorem arrangement_count :
  let total_men : ℕ := 4
  let total_women : ℕ := 5
  let group_of_four_men : ℕ := 2
  let group_of_four_women : ℕ := 2
  let remaining_men : ℕ := total_men - group_of_four_men
  let remaining_women : ℕ := total_women - group_of_four_women
  (Nat.choose total_men group_of_four_men) *
  (Nat.choose total_women group_of_four_women) *
  (Nat.choose remaining_women remaining_men) = 180 :=
by sorry

end NUMINAMATH_CALUDE_arrangement_count_l595_59561


namespace NUMINAMATH_CALUDE_sets_equality_implies_sum_l595_59556

-- Define the sets A and B
def A (x y : ℝ) : Set ℝ := {x, y/x, 1}
def B (x y : ℝ) : Set ℝ := {x^2, x+y, 0}

-- State the theorem
theorem sets_equality_implies_sum (x y : ℝ) (h : A x y = B x y) : x^2014 + y^2015 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sets_equality_implies_sum_l595_59556


namespace NUMINAMATH_CALUDE_cone_sphere_ratio_l595_59544

/-- Given a right circular cone and a sphere with the same radius,
    if the volume of the cone is two-fifths that of the sphere,
    then the ratio of the cone's altitude to twice its base radius is 4/5. -/
theorem cone_sphere_ratio (r h : ℝ) (hr : r > 0) : 
  (1 / 3 * π * r^2 * h) = (2 / 5 * (4 / 3 * π * r^3)) → 
  h / (2 * r) = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cone_sphere_ratio_l595_59544


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l595_59567

theorem imaginary_part_of_z (z : ℂ) (h : (1 + Complex.I) * z = Complex.I) : 
  z.im = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l595_59567


namespace NUMINAMATH_CALUDE_ping_pong_probabilities_l595_59552

/-- Represents the probability of player A winning a point -/
def prob_A_wins (serving : Bool) : ℝ :=
  if serving then 0.5 else 0.4

/-- Represents the probability of player A winning the k-th point after a 10:10 tie -/
def prob_A_k (k : ℕ) : ℝ :=
  prob_A_wins (k % 2 = 1)

/-- The probability of the game ending in exactly 2 points after a 10:10 tie -/
def prob_X_2 : ℝ :=
  prob_A_k 1 * prob_A_k 2 + (1 - prob_A_k 1) * (1 - prob_A_k 2)

/-- The probability of the game ending in exactly 4 points after a 10:10 tie with A winning -/
def prob_X_4_A_wins : ℝ :=
  (1 - prob_A_k 1) * prob_A_k 2 * prob_A_k 3 * prob_A_k 4 +
  prob_A_k 1 * (1 - prob_A_k 2) * prob_A_k 3 * prob_A_k 4

theorem ping_pong_probabilities :
  (prob_X_2 = prob_A_k 1 * prob_A_k 2 + (1 - prob_A_k 1) * (1 - prob_A_k 2)) ∧
  (prob_X_4_A_wins = (1 - prob_A_k 1) * prob_A_k 2 * prob_A_k 3 * prob_A_k 4 +
                     prob_A_k 1 * (1 - prob_A_k 2) * prob_A_k 3 * prob_A_k 4) := by
  sorry

end NUMINAMATH_CALUDE_ping_pong_probabilities_l595_59552


namespace NUMINAMATH_CALUDE_intersection_N_complement_M_l595_59586

def M : Set ℝ := {x | x^2 > 4}
def N : Set ℝ := {x | 1 < x ∧ x < 3}

theorem intersection_N_complement_M :
  N ∩ (Set.univ \ M) = {x | 1 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_N_complement_M_l595_59586


namespace NUMINAMATH_CALUDE_total_people_l595_59570

/-- Calculates the total number of people in two tribes of soldiers -/
theorem total_people (cannoneers : ℕ) : 
  cannoneers = 63 → 
  (let women := 2 * cannoneers
   let men := cannoneers + 2 * women
   women + men) = 441 := by
sorry

end NUMINAMATH_CALUDE_total_people_l595_59570


namespace NUMINAMATH_CALUDE_wire_length_around_square_field_l595_59545

theorem wire_length_around_square_field (area : ℝ) (num_rounds : ℕ) : 
  area = 27889 ∧ num_rounds = 11 → 
  Real.sqrt area * 4 * num_rounds = 7348 := by
sorry

end NUMINAMATH_CALUDE_wire_length_around_square_field_l595_59545


namespace NUMINAMATH_CALUDE_power_of_two_difference_l595_59575

theorem power_of_two_difference (n : ℕ) (h : n > 0) : 2^n - 2^(n-1) = 2^(n-1) := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_difference_l595_59575


namespace NUMINAMATH_CALUDE_complex_equality_l595_59542

theorem complex_equality (z : ℂ) : 
  Complex.abs (1 + Complex.I * z) = Complex.abs (3 + 4 * Complex.I) →
  Complex.abs (z - Complex.I) = 5 := by
sorry

end NUMINAMATH_CALUDE_complex_equality_l595_59542


namespace NUMINAMATH_CALUDE_max_blocks_fit_l595_59593

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- The dimensions of the large box -/
def largeBox : BoxDimensions := ⟨3, 3, 2⟩

/-- The dimensions of the small block -/
def smallBlock : BoxDimensions := ⟨1, 2, 2⟩

/-- Calculates the volume of a box given its dimensions -/
def volume (box : BoxDimensions) : ℝ :=
  box.length * box.width * box.height

/-- Represents the number of small blocks that can fit in the large box -/
def maxBlocks : ℕ := 3

/-- Theorem stating that the maximum number of small blocks that can fit in the large box is 3 -/
theorem max_blocks_fit :
  maxBlocks = 3 ∧
  maxBlocks * volume smallBlock ≤ volume largeBox ∧
  ∀ n : ℕ, n > maxBlocks → n * volume smallBlock > volume largeBox :=
by sorry

end NUMINAMATH_CALUDE_max_blocks_fit_l595_59593


namespace NUMINAMATH_CALUDE_ranking_sequences_count_l595_59524

/-- Represents a player in the chess tournament -/
inductive Player : Type
| A : Player
| B : Player
| C : Player
| D : Player

/-- Represents a match between two players -/
structure Match :=
(player1 : Player)
(player2 : Player)

/-- Represents the tournament structure -/
structure Tournament :=
(initial_match1 : Match)
(initial_match2 : Match)
(winners_match : Match)
(losers_match : Match)
(third_place_match : Match)

/-- A function to calculate the number of possible ranking sequences -/
def count_ranking_sequences (t : Tournament) : Nat :=
  sorry

/-- The theorem stating that the number of possible ranking sequences is 8 -/
theorem ranking_sequences_count :
  ∀ t : Tournament, count_ranking_sequences t = 8 :=
sorry

end NUMINAMATH_CALUDE_ranking_sequences_count_l595_59524


namespace NUMINAMATH_CALUDE_sarahs_coin_box_l595_59513

/-- The number of pennies in Sarah's box --/
def num_coins : ℕ := 36

/-- The total value of coins in cents --/
def total_value : ℕ := 2000

/-- Theorem stating that the number of each type of coin in Sarah's box is 36,
    given that the total value is $20 (2000 cents) and there are equal numbers
    of pennies, nickels, and half-dollars. --/
theorem sarahs_coin_box :
  (num_coins : ℚ) * (1 + 5 + 50) = total_value :=
sorry

end NUMINAMATH_CALUDE_sarahs_coin_box_l595_59513


namespace NUMINAMATH_CALUDE_michaels_bunnies_l595_59598

theorem michaels_bunnies (total_pets : ℕ) (dog_percent : ℚ) (cat_percent : ℚ) 
  (h1 : total_pets = 36)
  (h2 : dog_percent = 25 / 100)
  (h3 : cat_percent = 50 / 100)
  (h4 : dog_percent + cat_percent < 1) :
  (1 - dog_percent - cat_percent) * total_pets = 9 := by
  sorry

end NUMINAMATH_CALUDE_michaels_bunnies_l595_59598


namespace NUMINAMATH_CALUDE_root_sum_ratio_l595_59595

theorem root_sum_ratio (x₁ x₂ : ℝ) : 
  (2 * x₁^2 - 4 * x₁ + 1 = 0) → 
  (2 * x₂^2 - 4 * x₂ + 1 = 0) → 
  (x₁ ≠ x₂) →
  (x₁ / x₂ + x₂ / x₁ = 6) := by
sorry

end NUMINAMATH_CALUDE_root_sum_ratio_l595_59595


namespace NUMINAMATH_CALUDE_solve_exponent_equation_l595_59520

theorem solve_exponent_equation (n : ℕ) : 2 * 2^2 * 2^n = 2^10 → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponent_equation_l595_59520


namespace NUMINAMATH_CALUDE_linear_function_solution_l595_59578

/-- Represents a linear function y = ax + b -/
structure LinearFunction where
  a : ℝ
  b : ℝ

/-- Represents a point (x, y) on the linear function -/
structure Point where
  x : ℝ
  y : ℝ

/-- Given data points for the linear function -/
def dataPoints : List Point := [
  { x := -3, y := -4 },
  { x := -2, y := -2 },
  { x := -1, y := 0 },
  { x := 0, y := 2 },
  { x := 1, y := 4 },
  { x := 2, y := 6 }
]

/-- The linear function satisfies all given data points -/
def satisfiesDataPoints (f : LinearFunction) : Prop :=
  ∀ p ∈ dataPoints, f.a * p.x + f.b = p.y

theorem linear_function_solution (f : LinearFunction) 
  (h : satisfiesDataPoints f) : 
  f.a * 1 + f.b = 4 := by sorry

end NUMINAMATH_CALUDE_linear_function_solution_l595_59578


namespace NUMINAMATH_CALUDE_greatest_odd_integer_below_sqrt_50_l595_59514

theorem greatest_odd_integer_below_sqrt_50 :
  ∀ x : ℕ, x % 2 = 1 → x^2 < 50 → x ≤ 7 :=
by sorry

end NUMINAMATH_CALUDE_greatest_odd_integer_below_sqrt_50_l595_59514


namespace NUMINAMATH_CALUDE_maintenance_check_increase_l595_59584

theorem maintenance_check_increase (original : ℝ) (new : ℝ) 
  (h1 : original = 30)
  (h2 : new = 60) :
  (new - original) / original * 100 = 100 := by
  sorry

end NUMINAMATH_CALUDE_maintenance_check_increase_l595_59584


namespace NUMINAMATH_CALUDE_triangle_side_lengths_l595_59565

/-- Given a triangle with area t, angle α, and angle β, prove that the sides a, b, and c have the specified lengths. -/
theorem triangle_side_lengths 
  (t : ℝ) 
  (α β : Real) 
  (h_t : t = 4920)
  (h_α : α = 43 + 36 / 60 + 10 / 3600)
  (h_β : β = 72 + 23 / 60 + 11 / 3600) :
  ∃ (a b c : ℝ), 
    (abs (a - 89) < 1) ∧ 
    (abs (b - 123) < 1) ∧ 
    (abs (c - 116) < 1) ∧
    (a > 0) ∧ (b > 0) ∧ (c > 0) :=
by
  sorry


end NUMINAMATH_CALUDE_triangle_side_lengths_l595_59565


namespace NUMINAMATH_CALUDE_coffee_blend_price_l595_59515

/-- Given two coffee blends, this theorem proves the price of the second blend
    given the conditions of the problem. -/
theorem coffee_blend_price
  (price_blend1 : ℝ)
  (total_weight : ℝ)
  (total_price_per_pound : ℝ)
  (weight_blend2 : ℝ)
  (h1 : price_blend1 = 9)
  (h2 : total_weight = 20)
  (h3 : total_price_per_pound = 8.4)
  (h4 : weight_blend2 = 12)
  : ∃ (price_blend2 : ℝ),
    price_blend2 * weight_blend2 + price_blend1 * (total_weight - weight_blend2) =
    total_price_per_pound * total_weight ∧
    price_blend2 = 8 :=
by sorry

end NUMINAMATH_CALUDE_coffee_blend_price_l595_59515


namespace NUMINAMATH_CALUDE_perpendicular_vector_scalar_l595_59580

/-- Given two vectors a and b in ℝ², prove that if a + xb is perpendicular to b, then x = -2/5 -/
theorem perpendicular_vector_scalar (a b : ℝ × ℝ) (x : ℝ) 
    (h1 : a = (3, 4))
    (h2 : b = (2, -1))
    (h3 : (a.1 + x * b.1, a.2 + x * b.2) • b = 0) :
  x = -2/5 := by
  sorry

#check perpendicular_vector_scalar

end NUMINAMATH_CALUDE_perpendicular_vector_scalar_l595_59580


namespace NUMINAMATH_CALUDE_trajectory_equation_l595_59518

/-- Given m ∈ ℝ, vector a = (mx, y+1), vector b = (x, y-1), and a ⊥ b,
    the equation of trajectory E for moving point M(x,y) is mx² + y² = 1 -/
theorem trajectory_equation (m : ℝ) (x y : ℝ) 
    (h : (m * x) * x + (y + 1) * (y - 1) = 0) : 
  m * x^2 + y^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_trajectory_equation_l595_59518


namespace NUMINAMATH_CALUDE_quadratic_function_min_value_l595_59532

theorem quadratic_function_min_value 
  (f : ℝ → ℝ) 
  (a : ℝ) 
  (h1 : ∀ x, f x = x^2 + x + a) 
  (h2 : ∃ x ∈ Set.Icc (-1 : ℝ) 1, ∀ y ∈ Set.Icc (-1 : ℝ) 1, f y ≤ f x) 
  (h3 : ∃ x ∈ Set.Icc (-1 : ℝ) 1, f x = 2) :
  ∃ x ∈ Set.Icc (-1 : ℝ) 1, ∀ y ∈ Set.Icc (-1 : ℝ) 1, f x ≤ f y ∧ f x = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_min_value_l595_59532


namespace NUMINAMATH_CALUDE_max_added_value_max_at_two_thirds_verify_half_a_l595_59539

/-- The added value function for a car factory's production line -/
def f (a : ℝ) (x : ℝ) : ℝ := 8 * (a - x) * x^2

/-- Theorem stating the maximum value of the added value function -/
theorem max_added_value (a : ℝ) (h_a : a > 0) :
  ∃ (x : ℝ), x ∈ Set.Ioo 0 (4 * a / 5) ∧
    f a x = (32 / 27) * a^3 ∧
    ∀ (y : ℝ), y ∈ Set.Ioo 0 (4 * a / 5) → f a y ≤ (32 / 27) * a^3 :=
by sorry

/-- Theorem stating that the maximum occurs at x = 2a/3 -/
theorem max_at_two_thirds (a : ℝ) (h_a : a > 0) :
  f a (2 * a / 3) = (32 / 27) * a^3 :=
by sorry

/-- Theorem verifying that f(a/2) = a^3 -/
theorem verify_half_a (a : ℝ) :
  f a (a / 2) = a^3 :=
by sorry

end NUMINAMATH_CALUDE_max_added_value_max_at_two_thirds_verify_half_a_l595_59539


namespace NUMINAMATH_CALUDE_exists_x_y_for_a_l595_59588

def a : ℕ → ℤ
  | 0 => 4
  | 1 => 22
  | (n + 2) => 6 * a (n + 1) - a n

def b : ℕ → ℤ
  | 0 => 2
  | 1 => 1
  | (n + 2) => 2 * b (n + 1) + b n

theorem exists_x_y_for_a : ∃ (x y : ℕ → ℕ), ∀ n, 
  (y n)^2 + 7 = (x n - y n) * a n :=
sorry

end NUMINAMATH_CALUDE_exists_x_y_for_a_l595_59588


namespace NUMINAMATH_CALUDE_quadratic_function_general_form_l595_59572

/-- A quadratic function with the same shape as y = 5x² and vertex at (3, 7) -/
def quadratic_function (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, 
    (a = 5 ∨ a = -5) ∧
    (∀ x : ℝ, f x = a * (x - 3)^2 + 7)

theorem quadratic_function_general_form (f : ℝ → ℝ) 
  (h : quadratic_function f) :
  (∀ x : ℝ, f x = 5 * x^2 - 30 * x + 52) ∨
  (∀ x : ℝ, f x = -5 * x^2 + 30 * x - 38) :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_general_form_l595_59572


namespace NUMINAMATH_CALUDE_subtraction_addition_equality_l595_59551

theorem subtraction_addition_equality : -32 - (-14) + 4 = -14 := by sorry

end NUMINAMATH_CALUDE_subtraction_addition_equality_l595_59551


namespace NUMINAMATH_CALUDE_bracelet_cost_calculation_josh_bracelet_cost_l595_59521

theorem bracelet_cost_calculation (bracelet_price : ℝ) (num_bracelets : ℕ) (cookie_cost : ℝ) (money_left : ℝ) : ℝ :=
  let total_earned := bracelet_price * num_bracelets
  let total_after_cookies := cookie_cost + money_left
  let supply_cost := (total_earned - total_after_cookies) / num_bracelets
  supply_cost

theorem josh_bracelet_cost :
  bracelet_cost_calculation 1.5 12 3 3 = 1 := by sorry

end NUMINAMATH_CALUDE_bracelet_cost_calculation_josh_bracelet_cost_l595_59521


namespace NUMINAMATH_CALUDE_class_size_problem_l595_59530

theorem class_size_problem (x y : ℕ) : 
  y = x / 6 →  -- Initial condition: absent = 1/6 of present
  y = (x - 1) / 5 →  -- Condition after one student leaves
  x + y = 7  -- Total number of students
  := by sorry

end NUMINAMATH_CALUDE_class_size_problem_l595_59530


namespace NUMINAMATH_CALUDE_probability_point_between_C_and_E_l595_59505

/-- Given a line segment AB with points C, D, and E, where AB = 4AD, AB = 5BC, 
    and E is the midpoint of CD, the probability that a randomly selected point 
    on AB falls between C and E is 1/4. -/
theorem probability_point_between_C_and_E 
  (A B C D E : ℝ) 
  (h1 : A < C) (h2 : C < D) (h3 : D < B)
  (h4 : B - A = 4 * (D - A))
  (h5 : B - A = 5 * (C - B))
  (h6 : E = (C + D) / 2) :
  (E - C) / (B - A) = 1 / 4 := by
  sorry

#check probability_point_between_C_and_E

end NUMINAMATH_CALUDE_probability_point_between_C_and_E_l595_59505


namespace NUMINAMATH_CALUDE_top_triangle_number_l595_59557

/-- Represents the shape of a cell in the diagram -/
inductive Shape
| Circle
| Triangle
| Hexagon

/-- The sum of numbers in each shape -/
def sum_of_shape (s : Shape) : ℕ :=
  match s with
  | Shape.Circle => 10
  | Shape.Triangle => 15
  | Shape.Hexagon => 30

/-- The total number of cells in the diagram -/
def total_cells : ℕ := 9

/-- The set of numbers used in the diagram -/
def number_set : Finset ℕ := Finset.range 9

/-- The theorem stating the possible numbers in the top triangle -/
theorem top_triangle_number :
  ∃ (n : ℕ), n ∈ number_set ∧ n ≥ 8 ∧ n ≤ 9 ∧
  (∃ (a b : ℕ), a ∈ number_set ∧ b ∈ number_set ∧ a + b + n = sum_of_shape Shape.Triangle) :=
sorry

end NUMINAMATH_CALUDE_top_triangle_number_l595_59557


namespace NUMINAMATH_CALUDE_sequence_sum_l595_59522

theorem sequence_sum (A B C D E F G H : ℤ) : 
  C = 3 ∧ 
  A + B + C = 27 ∧
  B + C + D = 27 ∧
  C + D + E = 27 ∧
  D + E + F = 27 ∧
  E + F + G = 27 ∧
  F + G + H = 27 →
  A + H = 27 := by sorry

end NUMINAMATH_CALUDE_sequence_sum_l595_59522


namespace NUMINAMATH_CALUDE_max_product_sum_300_l595_59549

theorem max_product_sum_300 : 
  ∀ x y : ℤ, x + y = 300 → x * y ≤ 22500 :=
by sorry

end NUMINAMATH_CALUDE_max_product_sum_300_l595_59549


namespace NUMINAMATH_CALUDE_perpendicular_tangent_line_l595_59576

/-- The equation of a line perpendicular to x + 4y - 4 = 0 and tangent to y = 2x² --/
theorem perpendicular_tangent_line : 
  ∃ (a b c : ℝ), 
    (∀ x y : ℝ, a*x + b*y + c = 0) ∧ 
    (∀ x y : ℝ, x + 4*y - 4 = 0 → (a*1 + b*4 = 0)) ∧
    (∃ x₀ : ℝ, a*x₀ + b*(2*x₀^2) + c = 0 ∧ 
              ∀ x : ℝ, a*x + b*(2*x^2) + c ≥ 0) ∧
    (a = 4 ∧ b = -1 ∧ c = -2) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_tangent_line_l595_59576


namespace NUMINAMATH_CALUDE_solution_satisfies_equation_l595_59538

theorem solution_satisfies_equation (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  let x := (a^2 - b^2) / (2*a)
  (x^2 + b^2 + c^2) = ((a - x)^2 + c^2) :=
by sorry

end NUMINAMATH_CALUDE_solution_satisfies_equation_l595_59538


namespace NUMINAMATH_CALUDE_gcd_1729_867_l595_59574

theorem gcd_1729_867 : Nat.gcd 1729 867 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1729_867_l595_59574


namespace NUMINAMATH_CALUDE_sqrt_product_plus_one_l595_59535

theorem sqrt_product_plus_one : 
  Real.sqrt ((25 : ℝ) * 24 * 23 * 22 + 1) = 551 := by sorry

end NUMINAMATH_CALUDE_sqrt_product_plus_one_l595_59535


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l595_59504

-- Define the quadratic function
def f (b : ℝ) (x : ℝ) : ℝ := x^2 - b*x + 1

-- State the theorem
theorem quadratic_inequality_solution_range (b : ℝ) (x₁ x₂ : ℝ) 
  (h1 : ∀ x, f b x > 0 ↔ x < x₁ ∨ x > x₂)
  (h2 : x₁ < 1)
  (h3 : x₂ > 1) :
  b > 2 ∧ b ∈ Set.Ioi 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l595_59504


namespace NUMINAMATH_CALUDE_twins_age_problem_l595_59550

theorem twins_age_problem (age : ℕ) : 
  (age + 1) * (age + 1) = age * age + 9 → age = 4 := by
  sorry

end NUMINAMATH_CALUDE_twins_age_problem_l595_59550


namespace NUMINAMATH_CALUDE_yoongi_hoseok_age_sum_l595_59507

/-- Given the ages and relationships between Yoongi, Hoseok, and Yoongi's aunt,
    prove that the sum of Yoongi's and Hoseok's ages is 26 years. -/
theorem yoongi_hoseok_age_sum :
  ∀ (yoongi_age hoseok_age aunt_age : ℕ),
  aunt_age = 38 →
  aunt_age = yoongi_age + 23 →
  yoongi_age = hoseok_age + 4 →
  yoongi_age + hoseok_age = 26 :=
by
  sorry

end NUMINAMATH_CALUDE_yoongi_hoseok_age_sum_l595_59507


namespace NUMINAMATH_CALUDE_cubic_inequality_solution_l595_59501

theorem cubic_inequality_solution (x : ℝ) :
  x^3 - 10*x^2 + 28*x > 0 ↔ (x > 0 ∧ x < 4) ∨ x > 6 := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_solution_l595_59501


namespace NUMINAMATH_CALUDE_total_blocks_is_2250_l595_59517

/-- Represents the size of a dog -/
inductive DogSize
  | Small
  | Medium
  | Large

/-- Represents the walking speed of a dog in blocks per 10 minutes -/
def walkingSpeed (size : DogSize) : ℕ :=
  match size with
  | .Small => 3
  | .Medium => 4
  | .Large => 2

/-- Represents the number of dogs of each size -/
def dogCounts : DogSize → ℕ
  | .Small => 10
  | .Medium => 8
  | .Large => 7

/-- The total vacation cost in dollars -/
def vacationCost : ℕ := 1200

/-- The number of family members -/
def familyMembers : ℕ := 5

/-- The total available time in minutes -/
def totalAvailableTime : ℕ := 8 * 60

/-- The break time in minutes -/
def breakTime : ℕ := 30

/-- Calculates the total number of blocks Jules has to walk -/
def totalBlocks : ℕ :=
  let availableTime := totalAvailableTime - breakTime
  let slowestSpeed := walkingSpeed DogSize.Large
  let blocksPerDog := (availableTime / 10) * slowestSpeed
  (dogCounts DogSize.Small + dogCounts DogSize.Medium + dogCounts DogSize.Large) * blocksPerDog

theorem total_blocks_is_2250 : totalBlocks = 2250 := by
  sorry

#eval totalBlocks

end NUMINAMATH_CALUDE_total_blocks_is_2250_l595_59517


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l595_59536

/-- Given a line L1 with equation x + y - 5 = 0 and a point P (2, -1),
    prove that the line L2 passing through P and perpendicular to L1
    has the equation x - y - 3 = 0 -/
theorem perpendicular_line_equation (L1 : Set (ℝ × ℝ)) (P : ℝ × ℝ) :
  L1 = {(x, y) | x + y - 5 = 0} →
  P = (2, -1) →
  ∃ L2 : Set (ℝ × ℝ),
    (P ∈ L2) ∧
    (∀ (A B : ℝ × ℝ), A ∈ L1 → B ∈ L1 → A ≠ B →
      ∀ (C D : ℝ × ℝ), C ∈ L2 → D ∈ L2 → C ≠ D →
        ((A.1 - B.1) * (C.1 - D.1) + (A.2 - B.2) * (C.2 - D.2) = 0)) ∧
    L2 = {(x, y) | x - y - 3 = 0} :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l595_59536


namespace NUMINAMATH_CALUDE_benny_eggs_count_l595_59560

def dozen : ℕ := 12

def eggs_bought (num_dozens : ℕ) : ℕ := num_dozens * dozen

theorem benny_eggs_count : eggs_bought 7 = 84 := by sorry

end NUMINAMATH_CALUDE_benny_eggs_count_l595_59560


namespace NUMINAMATH_CALUDE_trivia_game_score_l595_59581

/-- The final score of a trivia game given the scores of three rounds -/
def final_score (round1 : Int) (round2 : Int) (round3 : Int) : Int :=
  round1 + round2 + round3

/-- Theorem: Given the scores from three rounds of a trivia game (16, 33, and -48),
    the final score is equal to 1. -/
theorem trivia_game_score :
  final_score 16 33 (-48) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trivia_game_score_l595_59581


namespace NUMINAMATH_CALUDE_manager_percentage_after_leaving_l595_59553

/-- Calculates the new percentage of managers after some leave the room -/
def new_manager_percentage (initial_employees : ℕ) (initial_manager_percentage : ℚ) 
  (managers_leaving : ℚ) : ℚ :=
  let initial_managers : ℚ := (initial_manager_percentage / 100) * initial_employees
  let remaining_managers : ℚ := initial_managers - managers_leaving
  let remaining_employees : ℚ := initial_employees - managers_leaving
  (remaining_managers / remaining_employees) * 100

/-- Theorem stating that given the initial conditions and managers leaving, 
    the new percentage of managers is 98% -/
theorem manager_percentage_after_leaving :
  new_manager_percentage 200 99 99.99999999999991 = 98 := by
  sorry

end NUMINAMATH_CALUDE_manager_percentage_after_leaving_l595_59553


namespace NUMINAMATH_CALUDE_tan_100_degrees_l595_59506

theorem tan_100_degrees (k : ℝ) (h : Real.sin (-(80 * π / 180)) = k) :
  Real.tan ((100 * π / 180)) = k / Real.sqrt (1 - k^2) := by
  sorry

end NUMINAMATH_CALUDE_tan_100_degrees_l595_59506


namespace NUMINAMATH_CALUDE_largest_divisor_of_n_squared_divisible_by_72_l595_59591

theorem largest_divisor_of_n_squared_divisible_by_72 (n : ℕ) (h1 : n > 0) (h2 : 72 ∣ n^2) :
  ∃ q : ℕ, q > 0 ∧ q ∣ n ∧ ∀ m : ℕ, m > 0 ∧ m ∣ n → m ≤ q ∧ q = 12 :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_n_squared_divisible_by_72_l595_59591


namespace NUMINAMATH_CALUDE_range_of_m_l595_59547

theorem range_of_m (x m : ℝ) : 
  (∀ x, (1/3 < x ∧ x < 1/2) → |x - m| < 1) ∧ 
  (∃ x, |x - m| < 1 ∧ ¬(1/3 < x ∧ x < 1/2)) → 
  -1/2 ≤ m ∧ m ≤ 4/3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l595_59547


namespace NUMINAMATH_CALUDE_intersection_empty_iff_b_in_range_l595_59537

def set_A : Set ℝ := {x | -2 < x ∧ x < 1/3}

def set_B (b : ℝ) : Set ℝ := {x | x^2 - 4*b*x + 3*b^2 < 0}

theorem intersection_empty_iff_b_in_range (b : ℝ) :
  set_A ∩ set_B b = ∅ ↔ b ≥ 1/3 ∨ b ≤ -2 ∨ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_intersection_empty_iff_b_in_range_l595_59537


namespace NUMINAMATH_CALUDE_symmetry_axis_of_translated_sine_function_l595_59546

/-- Given a function f(x) = 2sin(2x + π/6), g(x) is obtained by translating
    the graph of f(x) to the right by π/6 units. This theorem states that
    x = π/3 is an equation of one symmetry axis of g(x). -/
theorem symmetry_axis_of_translated_sine_function :
  ∀ (f g : ℝ → ℝ),
  (∀ x, f x = 2 * Real.sin (2 * x + π / 6)) →
  (∀ x, g x = f (x - π / 6)) →
  (∃ k : ℤ, 2 * (π / 3) - π / 6 = π / 2 + k * π) :=
by sorry

end NUMINAMATH_CALUDE_symmetry_axis_of_translated_sine_function_l595_59546


namespace NUMINAMATH_CALUDE_set_operations_l595_59554

-- Define the sets A and B
def A : Set ℝ := {x | x ≥ 2}
def B : Set ℝ := {x | x < 5}

-- State the theorem
theorem set_operations :
  (A ∪ B = Set.univ) ∧
  (Aᶜ ∩ B = {x | x < 2}) := by sorry

end NUMINAMATH_CALUDE_set_operations_l595_59554


namespace NUMINAMATH_CALUDE_equation_solution_l595_59533

theorem equation_solution : 
  ∃! x : ℚ, (x + 1) / 3 - 1 = (5 * x - 1) / 6 :=
by
  use -1
  constructor
  · -- Prove that -1 satisfies the equation
    sorry
  · -- Prove uniqueness
    sorry

#check equation_solution

end NUMINAMATH_CALUDE_equation_solution_l595_59533


namespace NUMINAMATH_CALUDE_problem_solution_inequality_proof_l595_59502

-- Define the functions f and g
def f (m : ℝ) (x : ℝ) : ℝ := |x - m|
def g (m : ℝ) (x : ℝ) : ℝ := 2 * f m x - f m (x + m)

-- Theorem statement
theorem problem_solution (m : ℝ) (h_m : m > 0) :
  (∃ (x : ℝ), g m x = -1 ∧ ∀ (y : ℝ), g m y ≥ -1) ↔ m = 1 :=
sorry

theorem inequality_proof (m : ℝ) (h_m : m > 0) 
  (a b : ℝ) (h_a : |a| < m) (h_b : |b| < m) (h_a_neq_0 : a ≠ 0) :
  |a * b - m| > |a| * |b / a - m| :=
sorry

end NUMINAMATH_CALUDE_problem_solution_inequality_proof_l595_59502


namespace NUMINAMATH_CALUDE_stationery_box_sheets_l595_59564

theorem stationery_box_sheets : ∀ (S E : ℕ),
  S - E = 30 →  -- Ann's condition
  2 * E = S →   -- Bob's condition
  3 * E = S - 10 →  -- Sue's condition
  S = 40 := by
sorry

end NUMINAMATH_CALUDE_stationery_box_sheets_l595_59564


namespace NUMINAMATH_CALUDE_neg_i_cubed_l595_59559

theorem neg_i_cubed (i : ℂ) (h : i^2 = -1) : (-i)^3 = -i := by
  sorry

end NUMINAMATH_CALUDE_neg_i_cubed_l595_59559


namespace NUMINAMATH_CALUDE_walking_speed_problem_l595_59528

theorem walking_speed_problem (slower_speed : ℝ) (faster_speed : ℝ) 
  (actual_distance : ℝ) (total_distance : ℝ) :
  faster_speed = 20 →
  actual_distance = 20 →
  total_distance = actual_distance + 20 →
  actual_distance / slower_speed = total_distance / faster_speed →
  slower_speed = 10 := by
  sorry

end NUMINAMATH_CALUDE_walking_speed_problem_l595_59528


namespace NUMINAMATH_CALUDE_two_colonies_growth_time_l595_59594

/-- Represents the number of days it takes for a bacteria colony to reach its habitat limit -/
def habitatLimitDays : ℕ := 25

/-- Represents the daily growth factor of a bacteria colony -/
def dailyGrowthFactor : ℕ := 2

/-- Theorem stating that two simultaneously growing bacteria colonies 
    will reach the habitat limit in the same number of days as a single colony -/
theorem two_colonies_growth_time (initialSize : ℕ) (habitatLimit : ℕ) :
  initialSize > 0 →
  habitatLimit > 0 →
  habitatLimit = initialSize * dailyGrowthFactor ^ habitatLimitDays →
  (2 * initialSize) * dailyGrowthFactor ^ habitatLimitDays = 2 * habitatLimit :=
by
  sorry

end NUMINAMATH_CALUDE_two_colonies_growth_time_l595_59594


namespace NUMINAMATH_CALUDE_corner_sum_is_164_l595_59566

/-- Represents a 9x9 checkerboard filled with numbers 1 through 81 -/
def Checkerboard := Fin 9 → Fin 9 → Nat

/-- The number at position (i, j) on the checkerboard -/
def number_at (board : Checkerboard) (i j : Fin 9) : Nat :=
  9 * i.val + j.val + 1

/-- The sum of numbers in the four corners of the checkerboard -/
def corner_sum (board : Checkerboard) : Nat :=
  number_at board 0 0 + number_at board 0 8 + 
  number_at board 8 0 + number_at board 8 8

/-- Theorem stating that the sum of numbers in the four corners is 164 -/
theorem corner_sum_is_164 (board : Checkerboard) : corner_sum board = 164 := by
  sorry

end NUMINAMATH_CALUDE_corner_sum_is_164_l595_59566


namespace NUMINAMATH_CALUDE_dilation_rotation_composition_l595_59555

def dilation_matrix : Matrix (Fin 2) (Fin 2) ℝ := !![2, 0; 0, 2]
def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ := !![0, 1; -1, 0]

theorem dilation_rotation_composition :
  rotation_matrix * dilation_matrix = !![0, 2; -2, 0] := by sorry

end NUMINAMATH_CALUDE_dilation_rotation_composition_l595_59555


namespace NUMINAMATH_CALUDE_y_intercept_of_l_l595_59583

/-- The y-intercept of a line is the y-coordinate of the point where the line intersects the y-axis. -/
def y_intercept (a b : ℝ) : ℝ := b

/-- The line l is defined by the equation y = 3x - 2 -/
def l (x : ℝ) : ℝ := 3 * x - 2

theorem y_intercept_of_l :
  y_intercept 3 (-2) = -2 := by sorry

end NUMINAMATH_CALUDE_y_intercept_of_l_l595_59583


namespace NUMINAMATH_CALUDE_algebraic_expression_equality_l595_59531

theorem algebraic_expression_equality (x y : ℝ) (h : 2 * x - 3 * y = 1) :
  6 * y - 4 * x + 8 = 6 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_equality_l595_59531


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l595_59534

theorem simplify_fraction_product : 16 * (-24 / 5) * (45 / 56) = -2160 / 7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l595_59534


namespace NUMINAMATH_CALUDE_athletes_count_is_ten_l595_59516

-- Define the types for our counts
def TotalLegs : ℕ := 108
def TotalHeads : ℕ := 32

-- Define a structure to represent the counts of each animal type
structure AnimalCounts where
  athletes : ℕ
  elephants : ℕ
  monkeys : ℕ

-- Define the property that the counts satisfy the given conditions
def satisfiesConditions (counts : AnimalCounts) : Prop :=
  2 * counts.athletes + 4 * counts.elephants + 2 * counts.monkeys = TotalLegs ∧
  counts.athletes + counts.elephants + counts.monkeys = TotalHeads

-- The theorem to prove
theorem athletes_count_is_ten :
  ∃ (counts : AnimalCounts), satisfiesConditions counts ∧ counts.athletes = 10 :=
by sorry

end NUMINAMATH_CALUDE_athletes_count_is_ten_l595_59516


namespace NUMINAMATH_CALUDE_room_breadth_is_five_meters_l595_59510

/-- Given a building with 5 equal-area rooms, prove that the breadth of each room is 5 meters. -/
theorem room_breadth_is_five_meters 
  (num_rooms : ℕ) 
  (room_length : ℝ) 
  (room_height : ℝ) 
  (bricks_per_sqm : ℕ) 
  (bricks_for_floor : ℕ) :
  num_rooms = 5 →
  room_length = 4 →
  room_height = 2 →
  bricks_per_sqm = 17 →
  bricks_for_floor = 340 →
  ∃ (room_breadth : ℝ), room_breadth = 5 :=
by sorry

end NUMINAMATH_CALUDE_room_breadth_is_five_meters_l595_59510


namespace NUMINAMATH_CALUDE_three_heads_with_tail_probability_l595_59585

/-- A fair coin flip sequence that ends when either three heads in a row or two tails in a row occur -/
inductive CoinFlipSequence
  | Incomplete : List Bool → CoinFlipSequence
  | ThreeHeads : List Bool → CoinFlipSequence
  | TwoTails : List Bool → CoinFlipSequence

/-- The probability of getting three heads in a row with at least one tail before the third head -/
def probability_three_heads_with_tail : ℚ :=
  5 / 64

/-- The main theorem stating that the calculated probability is correct -/
theorem three_heads_with_tail_probability :
  probability_three_heads_with_tail = 5 / 64 := by
  sorry

end NUMINAMATH_CALUDE_three_heads_with_tail_probability_l595_59585


namespace NUMINAMATH_CALUDE_prob_different_suits_78_card_deck_l595_59543

/-- A custom deck of cards -/
structure CustomDeck where
  total_cards : ℕ
  num_suits : ℕ
  cards_per_suit : ℕ
  total_cards_eq : total_cards = num_suits * cards_per_suit

/-- The probability of drawing two cards of different suits from a custom deck -/
def prob_different_suits (deck : CustomDeck) : ℚ :=
  let remaining_cards := deck.total_cards - 1
  let cards_different_suit := (deck.num_suits - 1) * deck.cards_per_suit
  cards_different_suit / remaining_cards

/-- The main theorem stating the probability for the specific deck -/
theorem prob_different_suits_78_card_deck :
  ∃ (deck : CustomDeck),
    deck.total_cards = 78 ∧
    deck.num_suits = 6 ∧
    deck.cards_per_suit = 13 ∧
    prob_different_suits deck = 65 / 77 := by
  sorry

end NUMINAMATH_CALUDE_prob_different_suits_78_card_deck_l595_59543


namespace NUMINAMATH_CALUDE_line_perp_theorem_l595_59508

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation
variable (perp : Line → Plane → Prop)
variable (perpLine : Line → Line → Prop)
variable (perpPlane : Plane → Plane → Prop)

-- State the theorem
theorem line_perp_theorem 
  (m n : Line) (a β : Plane) 
  (hm : perp m a) (hn : perp n β) (hab : perpPlane a β) :
  perpLine m n :=
sorry

end NUMINAMATH_CALUDE_line_perp_theorem_l595_59508


namespace NUMINAMATH_CALUDE_order_of_powers_l595_59579

theorem order_of_powers : 5^56 < 31^28 ∧ 31^28 < 17^35 ∧ 17^35 < 10^51 := by
  sorry

end NUMINAMATH_CALUDE_order_of_powers_l595_59579


namespace NUMINAMATH_CALUDE_smallest_n_for_candy_purchase_l595_59523

theorem smallest_n_for_candy_purchase : ∃ n : ℕ+, 
  (∀ m : ℕ+, (15 * m).gcd 10 = 10 ∧ (15 * m).gcd 16 = 16 ∧ (15 * m).gcd 18 = 18 → n ≤ m) ∧
  (15 * n).gcd 10 = 10 ∧ (15 * n).gcd 16 = 16 ∧ (15 * n).gcd 18 = 18 ∧
  n = 48 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_candy_purchase_l595_59523


namespace NUMINAMATH_CALUDE_percentage_of_red_non_honda_cars_l595_59568

theorem percentage_of_red_non_honda_cars 
  (total_cars : ℕ) 
  (honda_cars : ℕ) 
  (honda_red_ratio : ℚ) 
  (total_red_ratio : ℚ) 
  (h1 : total_cars = 900) 
  (h2 : honda_cars = 500) 
  (h3 : honda_red_ratio = 90 / 100) 
  (h4 : total_red_ratio = 60 / 100) :
  (((total_red_ratio * total_cars) - (honda_red_ratio * honda_cars)) / (total_cars - honda_cars)) = 225 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_red_non_honda_cars_l595_59568


namespace NUMINAMATH_CALUDE_arithmetic_sequence_12th_term_l595_59562

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_12th_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum : a 7 + a 9 = 16)
  (h_4th : a 4 = 1) :
  a 12 = 15 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_12th_term_l595_59562


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l595_59571

theorem sqrt_equation_solution (y : ℝ) : Real.sqrt (y + 5) = 7 → y = 44 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l595_59571


namespace NUMINAMATH_CALUDE_inscribed_triangles_equal_area_different_shape_l595_59511

/-- A circle in which triangles can be inscribed -/
structure Circle where
  radius : ℝ
  radius_pos : radius > 0

/-- A triangle inscribed in a circle -/
structure InscribedTriangle (c : Circle) where
  vertices : Fin 3 → ℝ × ℝ
  inscribed : ∀ i, (vertices i).1^2 + (vertices i).2^2 = c.radius^2

/-- The area of an inscribed triangle -/
def area (c : Circle) (t : InscribedTriangle c) : ℝ :=
  sorry

/-- Two triangles are non-congruent if they have different shapes -/
def non_congruent (c : Circle) (t1 t2 : InscribedTriangle c) : Prop :=
  sorry

theorem inscribed_triangles_equal_area_different_shape (c : Circle) :
  ∃ (t1 t2 : InscribedTriangle c), area c t1 = area c t2 ∧ non_congruent c t1 t2 :=
sorry

end NUMINAMATH_CALUDE_inscribed_triangles_equal_area_different_shape_l595_59511


namespace NUMINAMATH_CALUDE_cube_sum_from_sum_and_square_sum_l595_59529

theorem cube_sum_from_sum_and_square_sum (x y : ℝ) 
  (h1 : x + y = 5) 
  (h2 : x^2 + y^2 = 13) : 
  x^3 + y^3 = 35 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_from_sum_and_square_sum_l595_59529


namespace NUMINAMATH_CALUDE_arthurs_spending_l595_59597

/-- The cost of Arthur's purchase on the first day -/
def arthurs_first_day_cost (hamburger_price hot_dog_price : ℝ) : ℝ :=
  3 * hamburger_price + 4 * hot_dog_price

/-- The cost of Arthur's purchase on the second day -/
def arthurs_second_day_cost (hamburger_price hot_dog_price : ℝ) : ℝ :=
  2 * hamburger_price + 3 * hot_dog_price

theorem arthurs_spending : 
  ∀ (hamburger_price : ℝ),
    arthurs_second_day_cost hamburger_price 1 = 7 →
    arthurs_first_day_cost hamburger_price 1 = 10 := by
  sorry

end NUMINAMATH_CALUDE_arthurs_spending_l595_59597


namespace NUMINAMATH_CALUDE_even_function_sum_l595_59599

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

theorem even_function_sum (f : ℝ → ℝ) (h_even : is_even_function f) (h_value : f (-1) = 2) :
  f (-1) + f 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_even_function_sum_l595_59599


namespace NUMINAMATH_CALUDE_tree_growth_rate_consistency_l595_59519

theorem tree_growth_rate_consistency :
  ∃ (a b : ℝ), 
    (a + b) / 2 = 0.15 ∧ 
    (1 + a) * (1 + b) = 0.9 := by
  sorry

end NUMINAMATH_CALUDE_tree_growth_rate_consistency_l595_59519


namespace NUMINAMATH_CALUDE_ellipse_equation_for_given_conditions_l595_59573

/-- Represents an ellipse with center at the origin -/
structure Ellipse where
  a : ℝ  -- Semi-major axis
  b : ℝ  -- Semi-minor axis
  c : ℝ  -- Semi-focal distance

/-- The standard equation of an ellipse -/
def Ellipse.equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

theorem ellipse_equation_for_given_conditions :
  ∀ e : Ellipse,
  e.a = 6 →                  -- Major axis is 12 (2a = 12)
  e.c / e.a = 1 / 3 →        -- Eccentricity is 1/3
  e.c = 2 →                  -- Derived from eccentricity and semi-major axis
  e.b^2 = e.a^2 - e.c^2 →    -- Relationship between a, b, and c
  ∀ x y : ℝ,
  e.equation x y ↔ x^2 / 36 + y^2 / 32 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_for_given_conditions_l595_59573


namespace NUMINAMATH_CALUDE_proposition_logic_l595_59509

theorem proposition_logic (p q : Prop) 
  (h_p_false : ¬p) 
  (h_q_true : q) : 
  (¬(p ∧ q)) ∧ 
  (p ∨ q) ∧ 
  (¬p) ∧ 
  (¬(¬q)) := by
  sorry

end NUMINAMATH_CALUDE_proposition_logic_l595_59509


namespace NUMINAMATH_CALUDE_trig_identity_l595_59548

theorem trig_identity (x y : ℝ) : 
  Real.sin (x - y) * Real.sin x + Real.cos (x - y) * Real.cos x = Real.cos y := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l595_59548


namespace NUMINAMATH_CALUDE_number_problem_l595_59590

theorem number_problem (N : ℝ) : 
  (1/6 : ℝ) * (2/3 : ℝ) * (3/4 : ℝ) * (5/7 : ℝ) * N = 25 → 
  (60/100 : ℝ) * N = 252 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l595_59590


namespace NUMINAMATH_CALUDE_exam_score_theorem_l595_59525

/-- Represents an examination scoring system and a student's performance --/
structure ExamScore where
  correct_score : ℕ      -- Marks awarded for each correct answer
  wrong_penalty : ℕ      -- Marks deducted for each wrong answer
  total_score : ℤ        -- Total score achieved
  correct_answers : ℕ    -- Number of correct answers
  total_questions : ℕ    -- Total number of questions attempted

/-- Theorem stating that given the exam conditions, the total questions attempted is 75 --/
theorem exam_score_theorem (exam : ExamScore) 
  (h1 : exam.correct_score = 4)
  (h2 : exam.wrong_penalty = 1)
  (h3 : exam.total_score = 125)
  (h4 : exam.correct_answers = 40) :
  exam.total_questions = 75 := by
  sorry


end NUMINAMATH_CALUDE_exam_score_theorem_l595_59525


namespace NUMINAMATH_CALUDE_mr_a_net_gain_l595_59577

def initial_value : ℚ := 12000
def first_sale_profit : ℚ := 20 / 100
def second_sale_loss : ℚ := 15 / 100
def third_sale_profit : ℚ := 10 / 100

theorem mr_a_net_gain : 
  let first_sale := initial_value * (1 + first_sale_profit)
  let second_sale := first_sale * (1 - second_sale_loss)
  let third_sale := second_sale * (1 + third_sale_profit)
  first_sale - second_sale + third_sale - initial_value = 3384 := by
sorry

end NUMINAMATH_CALUDE_mr_a_net_gain_l595_59577


namespace NUMINAMATH_CALUDE_snooker_tournament_tickets_l595_59541

theorem snooker_tournament_tickets (total_tickets : ℕ) (vip_price gen_price : ℚ) 
  (total_revenue : ℚ) (h1 : total_tickets = 320) (h2 : vip_price = 40) 
  (h3 : gen_price = 15) (h4 : total_revenue = 7500) : 
  ∃ (vip_tickets gen_tickets : ℕ), 
    vip_tickets + gen_tickets = total_tickets ∧ 
    vip_price * vip_tickets + gen_price * gen_tickets = total_revenue ∧ 
    gen_tickets - vip_tickets = 104 :=
by sorry

end NUMINAMATH_CALUDE_snooker_tournament_tickets_l595_59541
