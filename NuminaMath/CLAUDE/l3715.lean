import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_root_implies_k_l3715_371525

theorem quadratic_root_implies_k (k : ℝ) : 
  (∃ x : ℝ, x^2 - 5*x + k = 0 ∧ x = 2) → k = 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_k_l3715_371525


namespace NUMINAMATH_CALUDE_alvin_friend_gave_wood_l3715_371534

/-- The number of pieces of wood Alvin needs in total -/
def total_needed : ℕ := 376

/-- The number of pieces of wood Alvin's brother gave him -/
def brother_gave : ℕ := 136

/-- The number of pieces of wood Alvin still needs to gather -/
def still_needed : ℕ := 117

/-- The number of pieces of wood Alvin's friend gave him -/
def friend_gave : ℕ := total_needed - brother_gave - still_needed

theorem alvin_friend_gave_wood : friend_gave = 123 := by
  sorry

end NUMINAMATH_CALUDE_alvin_friend_gave_wood_l3715_371534


namespace NUMINAMATH_CALUDE_largest_angle_in_consecutive_angle_hexagon_l3715_371504

/-- The largest angle in a convex hexagon with six consecutive integer angles -/
def largest_hexagon_angle : ℝ := 122.5

/-- A convex hexagon with six consecutive integer angles -/
structure ConsecutiveAngleHexagon where
  angles : Fin 6 → ℤ
  is_consecutive : ∀ i : Fin 5, angles i.succ = angles i + 1
  is_convex : ∀ i : Fin 6, 0 < angles i ∧ angles i < 180

theorem largest_angle_in_consecutive_angle_hexagon (h : ConsecutiveAngleHexagon) :
  (h.angles 5 : ℝ) = largest_hexagon_angle :=
sorry

end NUMINAMATH_CALUDE_largest_angle_in_consecutive_angle_hexagon_l3715_371504


namespace NUMINAMATH_CALUDE_sugar_salt_price_l3715_371505

/-- Given the price of 2 kg sugar and 5 kg salt, and the price of 1 kg sugar,
    prove the price of 3 kg sugar and 1 kg salt. -/
theorem sugar_salt_price
  (total_price : ℝ)
  (sugar_price : ℝ)
  (h1 : total_price = 5.5)
  (h2 : sugar_price = 1.5)
  (h3 : 2 * sugar_price + 5 * ((total_price - 2 * sugar_price) / 5) = total_price) :
  3 * sugar_price + ((total_price - 2 * sugar_price) / 5) = 5 :=
by sorry

end NUMINAMATH_CALUDE_sugar_salt_price_l3715_371505


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3715_371554

/-- An arithmetic sequence -/
def arithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h_arithmetic : arithmeticSequence a) 
  (h_sum : a 2 + a 12 = 32) : 
  2 * a 3 + a 15 = 48 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3715_371554


namespace NUMINAMATH_CALUDE_exists_square_with_1983_nines_l3715_371589

theorem exists_square_with_1983_nines : ∃ n : ℕ, ∃ m : ℕ, n^2 = 10^3968 - 10^1985 + m ∧ m < 10^1985 := by
  sorry

end NUMINAMATH_CALUDE_exists_square_with_1983_nines_l3715_371589


namespace NUMINAMATH_CALUDE_election_majority_l3715_371555

theorem election_majority (total_votes : ℕ) (winning_percentage : ℚ) : 
  total_votes = 6500 →
  winning_percentage = 60 / 100 →
  (winning_percentage * total_votes : ℚ).num - ((1 - winning_percentage) * total_votes : ℚ).num = 1300 := by
  sorry

end NUMINAMATH_CALUDE_election_majority_l3715_371555


namespace NUMINAMATH_CALUDE_problem_statement_l3715_371572

theorem problem_statement : (-4 : ℝ)^2007 * (-0.25 : ℝ)^2008 = -0.25 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3715_371572


namespace NUMINAMATH_CALUDE_M_enumeration_l3715_371596

def M : Set ℕ := {a | a > 0 ∧ ∃ k : ℤ, 4 / (1 - a) = k}

theorem M_enumeration : M = {2, 3, 5} := by sorry

end NUMINAMATH_CALUDE_M_enumeration_l3715_371596


namespace NUMINAMATH_CALUDE_coin_flip_sequences_l3715_371562

/-- The number of flips in the sequence -/
def num_flips : ℕ := 10

/-- The number of fixed flips (fifth and sixth must be heads) -/
def fixed_flips : ℕ := 2

/-- The number of possible outcomes for each flip -/
def outcomes_per_flip : ℕ := 2

/-- 
Theorem: The number of distinct sequences of coin flips, 
where two specific flips are fixed, is equal to 2^(total flips - fixed flips)
-/
theorem coin_flip_sequences : 
  outcomes_per_flip ^ (num_flips - fixed_flips) = 256 := by
  sorry

end NUMINAMATH_CALUDE_coin_flip_sequences_l3715_371562


namespace NUMINAMATH_CALUDE_sixth_fibonacci_is_eight_l3715_371577

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci n + fibonacci (n + 1)

theorem sixth_fibonacci_is_eight :
  ∃ x, (fibonacci 0 = 1) ∧ 
       (fibonacci 1 = 1) ∧ 
       (fibonacci 2 = 2) ∧ 
       (fibonacci 3 = 3) ∧ 
       (fibonacci 4 = 5) ∧ 
       (fibonacci 5 = x) ∧ 
       (fibonacci 6 = 13) ∧ 
       (x = 8) := by
  sorry

end NUMINAMATH_CALUDE_sixth_fibonacci_is_eight_l3715_371577


namespace NUMINAMATH_CALUDE_alex_bike_trip_l3715_371521

/-- Alex's bike trip problem -/
theorem alex_bike_trip (v : ℝ) 
  (h1 : 4.5 * v + 2.5 * 12 + 1.5 * 24 + 8 = 164) : v = 20 := by
  sorry

end NUMINAMATH_CALUDE_alex_bike_trip_l3715_371521


namespace NUMINAMATH_CALUDE_quadratic_sum_l3715_371509

/-- A quadratic function f(x) = ax^2 + bx + c with integer coefficients -/
def QuadraticFunction (a b c : ℤ) : ℝ → ℝ := fun x ↦ (a : ℝ) * x^2 + (b : ℝ) * x + (c : ℝ)

/-- The theorem stating that for a quadratic function with given properties, a + b - c = -7 -/
theorem quadratic_sum (a b c : ℤ) :
  let f := QuadraticFunction a b c
  (f 2 = 5) →  -- The graph passes through (2, 5)
  (∀ x, f x ≥ f 1) →  -- The vertex is at x = 1
  (f 1 = 3) →  -- The y-coordinate of the vertex is 3
  a + b - c = -7 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l3715_371509


namespace NUMINAMATH_CALUDE_ice_cream_sundaes_l3715_371574

theorem ice_cream_sundaes (n : ℕ) (h : n = 8) : 
  n + n.choose 2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_sundaes_l3715_371574


namespace NUMINAMATH_CALUDE_rattlesnake_tail_difference_l3715_371579

/-- The number of tail segments in an Eastern rattlesnake -/
def eastern_segments : ℕ := 6

/-- The number of tail segments in a Western rattlesnake -/
def western_segments : ℕ := 8

/-- The percentage difference in tail size between Eastern and Western rattlesnakes,
    expressed as a percentage of the Western rattlesnake's tail size -/
def percentage_difference : ℚ :=
  (western_segments - eastern_segments : ℚ) / western_segments * 100

/-- Theorem stating that the percentage difference in tail size between
    Eastern and Western rattlesnakes is 25% -/
theorem rattlesnake_tail_difference :
  percentage_difference = 25 := by sorry

end NUMINAMATH_CALUDE_rattlesnake_tail_difference_l3715_371579


namespace NUMINAMATH_CALUDE_gcd_9011_2147_l3715_371503

theorem gcd_9011_2147 : Nat.gcd 9011 2147 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_9011_2147_l3715_371503


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3715_371585

theorem inequality_solution_set 
  (a : ℝ) 
  (h1 : 0 < a) 
  (h2 : a < 1) 
  (h3 : ∀ x : ℝ, x^2 - 2*a*x + a > 0) : 
  {x : ℝ | a^(x^2 - 3) < a^(2*x) ∧ a^(2*x) < 1} = {x : ℝ | x > 3} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3715_371585


namespace NUMINAMATH_CALUDE_solution_set_for_m_equals_one_m_range_for_inequality_l3715_371526

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := |x + m| + |2*x - 1|

-- Theorem 1
theorem solution_set_for_m_equals_one (x : ℝ) :
  f 1 x ≥ 3 ↔ x ≤ -1 ∨ x ≥ 1 := by sorry

-- Theorem 2
theorem m_range_for_inequality (m : ℝ) (h1 : m > 0) :
  (∀ x ∈ Set.Icc m (2*m^2), (1/2) * f m x ≤ |x + 1|) →
  1/2 < m ∧ m ≤ 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_for_m_equals_one_m_range_for_inequality_l3715_371526


namespace NUMINAMATH_CALUDE_wrong_operation_correction_l3715_371598

theorem wrong_operation_correction (x : ℕ) : 
  x - 46 = 27 → x * 46 = 3358 := by
  sorry

end NUMINAMATH_CALUDE_wrong_operation_correction_l3715_371598


namespace NUMINAMATH_CALUDE_line_passes_through_intersection_and_perpendicular_l3715_371532

-- Define the lines
def line1 (x y : ℝ) : Prop := 2 * x + y + 2 = 0
def line2 (x y : ℝ) : Prop := 3 * x + 4 * y - 2 = 0
def line3 (x y : ℝ) : Prop := 3 * x - 2 * y + 4 = 0
def line4 (x y : ℝ) : Prop := 2 * x + 3 * y - 2 = 0

-- Define the intersection point
def intersection_point (x y : ℝ) : Prop := line1 x y ∧ line2 x y

-- Define perpendicularity
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

-- Theorem statement
theorem line_passes_through_intersection_and_perpendicular :
  ∃ (x y : ℝ),
    intersection_point x y ∧
    line4 x y ∧
    perpendicular 
      ((3 : ℝ) / 2) -- slope of line3
      (-(2 : ℝ) / 3) -- slope of line4
  := by sorry

end NUMINAMATH_CALUDE_line_passes_through_intersection_and_perpendicular_l3715_371532


namespace NUMINAMATH_CALUDE_function_passes_through_point_l3715_371599

theorem function_passes_through_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^(x - 2)
  f 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_function_passes_through_point_l3715_371599


namespace NUMINAMATH_CALUDE_probability_not_touching_center_2x2_l3715_371537

/-- Represents a square checkerboard -/
structure Checkerboard :=
  (size : ℕ)

/-- Represents a square region on the checkerboard -/
structure Square :=
  (size : ℕ)
  (position : ℕ × ℕ)

/-- Calculates the number of unit squares touching a given square -/
def touching_squares (board : Checkerboard) (square : Square) : ℕ :=
  sorry

/-- Calculates the probability of a randomly chosen unit square not touching a given square -/
def probability_not_touching (board : Checkerboard) (square : Square) : ℚ :=
  sorry

theorem probability_not_touching_center_2x2 :
  let board := Checkerboard.mk 6
  let center_square := Square.mk 2 (2, 2)
  probability_not_touching board center_square = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_probability_not_touching_center_2x2_l3715_371537


namespace NUMINAMATH_CALUDE_mia_chocolate_amount_l3715_371528

/-- 
Given that Liam has 72/7 pounds of chocolate and divides it into 6 equal piles,
this theorem proves that if he gives 2 piles to Mia, she will receive 24/7 pounds of chocolate.
-/
theorem mia_chocolate_amount 
  (total_chocolate : ℚ) 
  (num_piles : ℕ) 
  (piles_to_mia : ℕ) 
  (h1 : total_chocolate = 72 / 7)
  (h2 : num_piles = 6)
  (h3 : piles_to_mia = 2) : 
  piles_to_mia * (total_chocolate / num_piles) = 24 / 7 := by
  sorry

end NUMINAMATH_CALUDE_mia_chocolate_amount_l3715_371528


namespace NUMINAMATH_CALUDE_regular_fish_price_l3715_371561

/-- The regular price of fish per pound, given a 50% discount and half-pound package price -/
theorem regular_fish_price (discount_percent : ℚ) (discounted_half_pound_price : ℚ) : 
  discount_percent = 50 →
  discounted_half_pound_price = 3 →
  12 = (2 * discounted_half_pound_price) / (1 - discount_percent / 100) :=
by sorry

end NUMINAMATH_CALUDE_regular_fish_price_l3715_371561


namespace NUMINAMATH_CALUDE_correct_division_l3715_371530

theorem correct_division (dividend : ℕ) : 
  (dividend / 47 = 5 ∧ dividend % 47 = 8) → 
  (dividend / 74 = 3 ∧ dividend % 74 = 21) :=
by
  sorry

end NUMINAMATH_CALUDE_correct_division_l3715_371530


namespace NUMINAMATH_CALUDE_function_identity_implies_constant_relation_l3715_371512

-- Define the functions and constants
variable (f g : ℝ → ℝ)
variable (a b c : ℝ)

-- State the theorem
theorem function_identity_implies_constant_relation 
  (h : ∀ (x y : ℝ), f x * g y = a * x * y + b * x + c * y + 1) : 
  a = b * c := by sorry

end NUMINAMATH_CALUDE_function_identity_implies_constant_relation_l3715_371512


namespace NUMINAMATH_CALUDE_minimum_implies_a_range_l3715_371531

/-- The function f(x) = x^3 - 3x -/
def f (x : ℝ) : ℝ := x^3 - 3*x

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3*x^2 - 3

/-- Theorem: If f has a minimum value in the interval (a, 6-a^2), then a ∈ [-2, 1) -/
theorem minimum_implies_a_range (a : ℝ) 
  (h_min : ∃ (x : ℝ), a < x ∧ x < 6 - a^2 ∧ ∀ (y : ℝ), a < y ∧ y < 6 - a^2 → f y ≥ f x) :
  a ≥ -2 ∧ a < 1 := by sorry

end NUMINAMATH_CALUDE_minimum_implies_a_range_l3715_371531


namespace NUMINAMATH_CALUDE_fraction_value_l3715_371544

theorem fraction_value : (3000 - 2883)^2 / 121 = 106.36 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l3715_371544


namespace NUMINAMATH_CALUDE_sector_area_sixty_degrees_radius_six_l3715_371563

/-- The area of a circular sector with central angle π/3 and radius 6 is 6π -/
theorem sector_area_sixty_degrees_radius_six : 
  let r : ℝ := 6
  let α : ℝ := π / 3
  let sector_area := (1 / 2) * r^2 * α
  sector_area = 6 * π := by sorry

end NUMINAMATH_CALUDE_sector_area_sixty_degrees_radius_six_l3715_371563


namespace NUMINAMATH_CALUDE_equation_value_l3715_371548

theorem equation_value (a b c : ℝ) 
  (eq1 : 3 * a - 2 * b - 2 * c = 30)
  (eq2 : a + b + c = 10) :
  Real.sqrt (3 * a) - Real.sqrt (2 * b + 2 * c) = Real.sqrt 30 := by
  sorry

end NUMINAMATH_CALUDE_equation_value_l3715_371548


namespace NUMINAMATH_CALUDE_evaluate_expression_l3715_371524

theorem evaluate_expression : (9 ^ 9) * (3 ^ 3) / (3 ^ 30) = 1 / 19683 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3715_371524


namespace NUMINAMATH_CALUDE_quadratic_equation_no_equal_roots_l3715_371538

theorem quadratic_equation_no_equal_roots :
  ¬ ∃ x : ℝ, (x^2 + x + 3 = 0 ∧ (∀ y : ℝ, y^2 + y + 3 = 0 → y = x)) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_no_equal_roots_l3715_371538


namespace NUMINAMATH_CALUDE_prob_three_odd_less_than_one_eighth_l3715_371507

def n : ℕ := 2016

def odd_count : ℕ := n / 2

def prob_three_odd : ℚ :=
  (odd_count : ℚ) / n *
  ((odd_count - 1) : ℚ) / (n - 1) *
  ((odd_count - 2) : ℚ) / (n - 2)

theorem prob_three_odd_less_than_one_eighth :
  prob_three_odd < 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_odd_less_than_one_eighth_l3715_371507


namespace NUMINAMATH_CALUDE_sum_and_reciprocal_value_l3715_371565

theorem sum_and_reciprocal_value (x : ℝ) (h1 : x ≠ 0) (h2 : x^2 + (1/x)^2 = 23) : 
  x + (1/x) = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_reciprocal_value_l3715_371565


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l3715_371511

theorem quadratic_inequality_solution_range (k : ℝ) : 
  (∃ x : ℝ, k * x^2 - 2 * x + 6 * k < 0) ↔ k < -Real.sqrt 6 / 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l3715_371511


namespace NUMINAMATH_CALUDE_points_three_units_from_negative_three_l3715_371551

def distance (x y : ℝ) : ℝ := |x - y|

theorem points_three_units_from_negative_three :
  ∀ x : ℝ, distance x (-3) = 3 ↔ x = 0 ∨ x = -6 := by
  sorry

end NUMINAMATH_CALUDE_points_three_units_from_negative_three_l3715_371551


namespace NUMINAMATH_CALUDE_rain_probability_l3715_371547

/-- The probability of rain on three consecutive days --/
theorem rain_probability (p_sat p_sun p_mon_given_sat : ℝ) 
  (h_sat : p_sat = 0.7)
  (h_sun : p_sun = 0.5)
  (h_mon_given_sat : p_mon_given_sat = 0.4) :
  p_sat * p_sun * p_mon_given_sat = 0.14 := by
  sorry

end NUMINAMATH_CALUDE_rain_probability_l3715_371547


namespace NUMINAMATH_CALUDE_rectangle_tileability_l3715_371527

/-- A rectangle can be tiled with 1 × b tiles -/
def IsTileable (m n b : ℕ) : Prop := sorry

/-- For an even b, there exists M such that for all m, n > M with mn even, 
    an m × n rectangle is (1, b)-tileable -/
theorem rectangle_tileability (b : ℕ) (h_even : Even b) : 
  ∃ M : ℕ, ∀ m n : ℕ, m > M → n > M → Even (m * n) → IsTileable m n b := by sorry

end NUMINAMATH_CALUDE_rectangle_tileability_l3715_371527


namespace NUMINAMATH_CALUDE_overlap_number_l3715_371578

theorem overlap_number (numbers : List ℝ) : 
  numbers.length = 9 ∧ 
  (numbers.take 5).sum / 5 = 7 ∧ 
  (numbers.drop 4).sum / 5 = 10 ∧ 
  numbers.sum / 9 = 74 / 9 → 
  ∃ x ∈ numbers, x = 11 ∧ x ∈ numbers.take 5 ∧ x ∈ numbers.drop 4 := by
sorry

end NUMINAMATH_CALUDE_overlap_number_l3715_371578


namespace NUMINAMATH_CALUDE_joans_kittens_l3715_371588

theorem joans_kittens (given_away : ℕ) (remaining : ℕ) (original : ℕ) : 
  given_away = 2 → remaining = 6 → original = given_away + remaining :=
by
  sorry

end NUMINAMATH_CALUDE_joans_kittens_l3715_371588


namespace NUMINAMATH_CALUDE_final_antifreeze_ratio_l3715_371545

/-- Calculates the fraction of antifreeze in a tank after multiple replacements --/
def antifreezeRatio (tankCapacity : ℚ) (initialRatio : ℚ) (replacementAmount : ℚ) (replacements : ℕ) : ℚ :=
  let initialAntifreeze := tankCapacity * initialRatio
  let remainingRatio := (tankCapacity - replacementAmount) / tankCapacity
  initialAntifreeze * remainingRatio ^ replacements / tankCapacity

/-- Theorem stating the final antifreeze ratio after 4 replacements --/
theorem final_antifreeze_ratio :
  antifreezeRatio 20 (1/4) 4 4 = 1024/5000 := by
  sorry

#eval antifreezeRatio 20 (1/4) 4 4

end NUMINAMATH_CALUDE_final_antifreeze_ratio_l3715_371545


namespace NUMINAMATH_CALUDE_circle_radius_l3715_371566

theorem circle_radius (A : ℝ) (h : A = 81 * Real.pi) : 
  ∃ r : ℝ, r > 0 ∧ A = Real.pi * r^2 ∧ r = 9 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_l3715_371566


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3715_371522

theorem quadratic_factorization (C D : ℤ) :
  (∀ y, 20 * y^2 - 117 * y + 72 = (C * y - 8) * (D * y - 9)) →
  C * D + C = 25 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3715_371522


namespace NUMINAMATH_CALUDE_function_not_in_first_quadrant_l3715_371556

theorem function_not_in_first_quadrant (a b : ℝ) (ha : 0 < a ∧ a < 1) (hb : b < -1) :
  ∀ x y : ℝ, x > 0 → y > 0 → a^x + b < y :=
by sorry

end NUMINAMATH_CALUDE_function_not_in_first_quadrant_l3715_371556


namespace NUMINAMATH_CALUDE_min_sum_p_q_l3715_371594

theorem min_sum_p_q (p q : ℕ) : 
  p > 1 → q > 1 → 17 * (p + 1) = 28 * (q + 1) → 
  ∀ (p' q' : ℕ), p' > 1 → q' > 1 → 17 * (p' + 1) = 28 * (q' + 1) → 
  p + q ≤ p' + q' → p + q = 135 := by
sorry

end NUMINAMATH_CALUDE_min_sum_p_q_l3715_371594


namespace NUMINAMATH_CALUDE_cube_volume_from_face_perimeter_l3715_371506

theorem cube_volume_from_face_perimeter (face_perimeter : ℝ) (h : face_perimeter = 40) :
  let side_length := face_perimeter / 4
  let volume := side_length ^ 3
  volume = 1000 := by sorry

end NUMINAMATH_CALUDE_cube_volume_from_face_perimeter_l3715_371506


namespace NUMINAMATH_CALUDE_supplementary_angles_ratio_l3715_371541

theorem supplementary_angles_ratio (a b : ℝ) : 
  a + b = 180 → -- angles are supplementary
  a / b = 5 / 3 → -- ratio of angles is 5:3
  b = 67.5 -- smaller angle is 67.5°
  := by sorry

end NUMINAMATH_CALUDE_supplementary_angles_ratio_l3715_371541


namespace NUMINAMATH_CALUDE_sqrt_one_hundredth_l3715_371553

theorem sqrt_one_hundredth : Real.sqrt (1 / 100) = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_one_hundredth_l3715_371553


namespace NUMINAMATH_CALUDE_cone_height_from_lateral_surface_l3715_371543

/-- Given a cone whose lateral surface is a semicircle with radius a,
    prove that the height of the cone is (√3/2)a. -/
theorem cone_height_from_lateral_surface (a : ℝ) (h : a > 0) :
  let l := a  -- slant height
  let r := a / 2  -- radius of the base
  let h := Real.sqrt ((l ^ 2) - (r ^ 2))  -- height of the cone
  h = (Real.sqrt 3 / 2) * a :=
by sorry

end NUMINAMATH_CALUDE_cone_height_from_lateral_surface_l3715_371543


namespace NUMINAMATH_CALUDE_jovanas_shells_l3715_371583

/-- The problem of finding Jovana's initial amount of shells. -/
theorem jovanas_shells (initial final added : ℕ) : 
  (added = 23) → (final = 28) → (final = initial + added) → (initial = 5) := by
  sorry

end NUMINAMATH_CALUDE_jovanas_shells_l3715_371583


namespace NUMINAMATH_CALUDE_sphere_cylinder_volume_ratio_l3715_371533

/-- The ratio of the volume of a sphere inscribed in a right circular cylinder
    to the volume of the cylinder. -/
theorem sphere_cylinder_volume_ratio :
  ∀ (r : ℝ), r > 0 →
  (4 / 3 * π * r^3) / (π * r^2 * (2 * r)) = 2 * Real.sqrt 3 * π / 27 := by
  sorry

end NUMINAMATH_CALUDE_sphere_cylinder_volume_ratio_l3715_371533


namespace NUMINAMATH_CALUDE_square_of_quadratic_condition_l3715_371523

/-- 
If a polynomial x^4 + ax^3 + bx^2 + cx + d is the square of a quadratic polynomial,
then ac^2 - 4abd + 8cd = 0.
-/
theorem square_of_quadratic_condition (a b c d : ℝ) : 
  (∃ p q : ℝ, ∀ x : ℝ, x^4 + a*x^3 + b*x^2 + c*x + d = (x^2 + p*x + q)^2) →
  a*c^2 - 4*a*b*d + 8*c*d = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_of_quadratic_condition_l3715_371523


namespace NUMINAMATH_CALUDE_opposite_numbers_solution_l3715_371587

theorem opposite_numbers_solution (x : ℚ) : (2 * x - 3 = -(1 - 4 * x)) → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_opposite_numbers_solution_l3715_371587


namespace NUMINAMATH_CALUDE_step_increase_proof_l3715_371573

def daily_steps (x : ℕ) (week : ℕ) : ℕ :=
  1000 + (week - 1) * x

def weekly_steps (x : ℕ) (week : ℕ) : ℕ :=
  7 * daily_steps x week

def total_steps (x : ℕ) : ℕ :=
  weekly_steps x 1 + weekly_steps x 2 + weekly_steps x 3 + weekly_steps x 4

theorem step_increase_proof :
  ∃ x : ℕ, total_steps x = 70000 ∧ x = 1000 :=
by sorry

end NUMINAMATH_CALUDE_step_increase_proof_l3715_371573


namespace NUMINAMATH_CALUDE_one_fourth_of_7_2_l3715_371514

theorem one_fourth_of_7_2 : (7.2 : ℚ) / 4 = 9 / 5 := by
  sorry

end NUMINAMATH_CALUDE_one_fourth_of_7_2_l3715_371514


namespace NUMINAMATH_CALUDE_quadratic_solution_l3715_371592

theorem quadratic_solution (x : ℝ) (h1 : x > 0) (h2 : 3 * x^2 + 11 * x - 20 = 0) : x = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_l3715_371592


namespace NUMINAMATH_CALUDE_ellipse_dot_product_range_slope_product_constant_parallelogram_condition_l3715_371501

-- Define the ellipse
def ellipse (m : ℝ) (x y : ℝ) : Prop := 9 * x^2 + y^2 = m^2

-- Define the line
def line (k b : ℝ) (x y : ℝ) : Prop := y = k * x + b

-- Define the dot product
def dot_product (x1 y1 x2 y2 : ℝ) : ℝ := x1 * x2 + y1 * y2

theorem ellipse_dot_product_range :
  ∀ (x y : ℝ), ellipse 3 x y →
  ∃ (f1x f1y f2x f2y : ℝ),
    f1x = 0 ∧ f1y = 2 * Real.sqrt 2 ∧
    f2x = 0 ∧ f2y = -2 * Real.sqrt 2 ∧
    -7 ≤ dot_product (x - f1x) (y - f1y) (x - f2x) (y - f2y) ∧
    dot_product (x - f1x) (y - f1y) (x - f2x) (y - f2y) ≤ 1 :=
sorry

theorem slope_product_constant (m k b : ℝ) :
  k ≠ 0 → b ≠ 0 →
  ∃ (x1 y1 x2 y2 : ℝ),
    ellipse m x1 y1 ∧ ellipse m x2 y2 ∧
    line k b x1 y1 ∧ line k b x2 y2 ∧
    let x0 := (x1 + x2) / 2
    let y0 := (y1 + y2) / 2
    (y0 / x0) * k = -9 :=
sorry

theorem parallelogram_condition (m k : ℝ) :
  ellipse m (m/3) m →
  line k ((3-k)*m/3) (m/3) m →
  (∃ (x y : ℝ),
    ellipse m x y ∧
    line k ((3-k)*m/3) x y ∧
    x ≠ m/3 ∧ y ≠ m ∧
    (∃ (xp yp : ℝ),
      ellipse m xp yp ∧
      yp / xp = -9 / k ∧
      2 * (-(m - k*m/3)*k / (k^2 + 9)) = xp)) ↔
  (k = 4 + Real.sqrt 7 ∨ k = 4 - Real.sqrt 7) :=
sorry

end NUMINAMATH_CALUDE_ellipse_dot_product_range_slope_product_constant_parallelogram_condition_l3715_371501


namespace NUMINAMATH_CALUDE_tangent_equation_solution_l3715_371540

theorem tangent_equation_solution :
  ∃! x : Real, 0 ≤ x ∧ x ≤ 180 ∧
  Real.tan ((150 - x) * π / 180) = 
    (Real.sin (150 * π / 180) - Real.sin (x * π / 180)) /
    (Real.cos (150 * π / 180) - Real.cos (x * π / 180)) ∧
  x = 30 := by
sorry

end NUMINAMATH_CALUDE_tangent_equation_solution_l3715_371540


namespace NUMINAMATH_CALUDE_x_eighth_equals_one_l3715_371536

theorem x_eighth_equals_one (x : ℝ) (h : x + 1/x = Real.sqrt 2) : x^8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_eighth_equals_one_l3715_371536


namespace NUMINAMATH_CALUDE_school_network_connections_l3715_371593

/-- The number of connections in a network of switches where each switch connects to a fixed number of others -/
def connections (n : ℕ) (k : ℕ) : ℕ := n * k / 2

/-- Theorem: In a network of 30 switches, where each switch connects to exactly 4 others, there are 60 connections -/
theorem school_network_connections :
  connections 30 4 = 60 := by
  sorry

end NUMINAMATH_CALUDE_school_network_connections_l3715_371593


namespace NUMINAMATH_CALUDE_sphere_volume_ratio_l3715_371546

theorem sphere_volume_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) :
  (4 * Real.pi * r₁^2) / (4 * Real.pi * r₂^2) = 1 / 4 →
  ((4 / 3) * Real.pi * r₁^3) / ((4 / 3) * Real.pi * r₂^3) = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_ratio_l3715_371546


namespace NUMINAMATH_CALUDE_airplane_flight_problem_l3715_371576

/-- Airplane flight problem -/
theorem airplane_flight_problem 
  (wind_speed : ℝ) 
  (time_with_wind : ℝ) 
  (time_against_wind : ℝ) 
  (h1 : wind_speed = 24)
  (h2 : time_with_wind = 2.8)
  (h3 : time_against_wind = 3) :
  ∃ (airplane_speed : ℝ) (distance : ℝ),
    airplane_speed = 696 ∧ 
    distance = 2016 ∧
    time_with_wind * (airplane_speed + wind_speed) = distance ∧
    time_against_wind * (airplane_speed - wind_speed) = distance :=
by
  sorry


end NUMINAMATH_CALUDE_airplane_flight_problem_l3715_371576


namespace NUMINAMATH_CALUDE_annual_production_exceeds_plan_l3715_371581

/-- Represents the annual car production plan and actual quarterly production --/
structure CarProduction where
  annual_plan : ℝ
  first_quarter : ℝ
  second_quarter : ℝ
  third_quarter : ℝ
  fourth_quarter : ℝ

/-- Conditions for car production --/
def production_conditions (p : CarProduction) : Prop :=
  p.first_quarter = 0.25 * p.annual_plan ∧
  p.second_quarter = 1.08 * p.first_quarter ∧
  ∃ (k : ℝ), p.second_quarter = 11.25 * k ∧
              p.third_quarter = 12 * k ∧
              p.fourth_quarter = 13.5 * k

/-- Theorem stating that the annual production exceeds the plan by 13.2% --/
theorem annual_production_exceeds_plan (p : CarProduction) 
  (h : production_conditions p) : 
  (p.first_quarter + p.second_quarter + p.third_quarter + p.fourth_quarter) / p.annual_plan = 1.132 :=
sorry

end NUMINAMATH_CALUDE_annual_production_exceeds_plan_l3715_371581


namespace NUMINAMATH_CALUDE_representation_of_2021_l3715_371520

theorem representation_of_2021 : ∃ (a b c : ℤ), 2021 = a^2 - b^2 + c^2 := by
  -- We need to prove that there exist integers a, b, and c such that
  -- 2021 = a^2 - b^2 + c^2
  sorry

end NUMINAMATH_CALUDE_representation_of_2021_l3715_371520


namespace NUMINAMATH_CALUDE_range_of_a_l3715_371570

/-- The function f(x) = x^2 - 2x --/
def f (x : ℝ) : ℝ := x^2 - 2*x

/-- The function g(x) = ax + 2, where a > 0 --/
def g (a : ℝ) (x : ℝ) : ℝ := a*x + 2

/-- The closed interval [-1, 2] --/
def I : Set ℝ := Set.Icc (-1) 2

theorem range_of_a :
  ∀ a : ℝ, (a > 0 ∧
    (∀ x₁ ∈ I, ∃ x₀ ∈ I, g a x₁ = f x₀)) ↔
    (a ∈ Set.Ioo 0 (1/2)) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3715_371570


namespace NUMINAMATH_CALUDE_vendor_profit_l3715_371515

/-- Vendor's profit calculation --/
theorem vendor_profit : 
  let apple_buy_price : ℚ := 3 / 2
  let apple_sell_price : ℚ := 2
  let orange_buy_price : ℚ := 2.7 / 3
  let orange_sell_price : ℚ := 1
  let apple_discount_rate : ℚ := 1 / 10
  let orange_discount_rate : ℚ := 3 / 20
  let num_apples : ℕ := 5
  let num_oranges : ℕ := 5

  let discounted_apple_price := apple_sell_price * (1 - apple_discount_rate)
  let discounted_orange_price := orange_sell_price * (1 - orange_discount_rate)

  let total_cost := num_apples * apple_buy_price + num_oranges * orange_buy_price
  let total_revenue := num_apples * discounted_apple_price + num_oranges * discounted_orange_price

  total_revenue - total_cost = 1.25 := by sorry

end NUMINAMATH_CALUDE_vendor_profit_l3715_371515


namespace NUMINAMATH_CALUDE_baseball_team_average_l3715_371558

theorem baseball_team_average (total_score : ℕ) (total_players : ℕ) (high_scorers : ℕ) (high_average : ℕ) (remaining_average : ℕ) : 
  total_score = 270 →
  total_players = 9 →
  high_scorers = 5 →
  high_average = 50 →
  high_scorers * high_average + (total_players - high_scorers) * remaining_average = total_score →
  remaining_average = 5 := by
sorry

end NUMINAMATH_CALUDE_baseball_team_average_l3715_371558


namespace NUMINAMATH_CALUDE_upstream_speed_is_25_l3715_371535

/-- Represents the speed of a man rowing in different conditions -/
structure RowingSpeed where
  stillWater : ℝ  -- Speed in still water
  downstream : ℝ  -- Speed downstream

/-- Calculates the speed of the man rowing upstream -/
def upstreamSpeed (s : RowingSpeed) : ℝ :=
  2 * s.stillWater - s.downstream

/-- Theorem: Given the conditions, the upstream speed is 25 kmph -/
theorem upstream_speed_is_25 (s : RowingSpeed) 
  (h1 : s.stillWater = 32) 
  (h2 : s.downstream = 39) : 
  upstreamSpeed s = 25 := by
  sorry

#eval upstreamSpeed { stillWater := 32, downstream := 39 }

end NUMINAMATH_CALUDE_upstream_speed_is_25_l3715_371535


namespace NUMINAMATH_CALUDE_machine_output_percentage_l3715_371564

theorem machine_output_percentage :
  let prob_defect_A : ℝ := 9 / 1000
  let prob_defect_B : ℝ := 1 / 50
  let total_prob_defect : ℝ := 0.0156
  ∃ p : ℝ, 
    0 ≤ p ∧ p ≤ 1 ∧
    total_prob_defect = p * prob_defect_A + (1 - p) * prob_defect_B ∧
    p = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_machine_output_percentage_l3715_371564


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_l3715_371575

theorem closest_integer_to_cube_root : 
  ∃ (n : ℤ), n = 10 ∧ ∀ (m : ℤ), |n - (7^3 + 9^3)^(1/3)| ≤ |m - (7^3 + 9^3)^(1/3)| :=
by sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_l3715_371575


namespace NUMINAMATH_CALUDE_xyz_problem_l3715_371549

theorem xyz_problem (x y z : ℝ) 
  (h1 : x ≥ 1) (h2 : y ≥ 1) (h3 : z ≥ 1)
  (h4 : x * y * z = 10)
  (h5 : (x ^ (Real.log x)) * (y ^ (Real.log y)) * (z ^ (Real.log z)) = 10) :
  ((x = 1 ∧ y = 1 ∧ z = 10) ∨ 
   (x = 10 ∧ y = 1 ∧ z = 1) ∨ 
   (x = 1 ∧ y = 10 ∧ z = 1)) :=
by sorry

end NUMINAMATH_CALUDE_xyz_problem_l3715_371549


namespace NUMINAMATH_CALUDE_imaginary_part_of_3_minus_2i_l3715_371567

theorem imaginary_part_of_3_minus_2i :
  Complex.im (3 - 2 * Complex.I) = -2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_3_minus_2i_l3715_371567


namespace NUMINAMATH_CALUDE_complex_magnitude_equation_l3715_371590

theorem complex_magnitude_equation (t : ℝ) (h : t > 0) :
  Complex.abs (8 + 2 * t * Complex.I) = 12 → t = 2 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_equation_l3715_371590


namespace NUMINAMATH_CALUDE_circle_diameter_from_area_l3715_371559

theorem circle_diameter_from_area :
  ∀ (A r d : ℝ),
  A = 81 * Real.pi →
  A = Real.pi * r^2 →
  d = 2 * r →
  d = 18 :=
by sorry

end NUMINAMATH_CALUDE_circle_diameter_from_area_l3715_371559


namespace NUMINAMATH_CALUDE_average_age_of_group_l3715_371508

/-- The average age of a group of seventh-graders and their guardians -/
def average_age (num_students : ℕ) (student_avg_age : ℚ) (num_guardians : ℕ) (guardian_avg_age : ℚ) : ℚ :=
  ((num_students : ℚ) * student_avg_age + (num_guardians : ℚ) * guardian_avg_age) / ((num_students + num_guardians) : ℚ)

/-- Theorem stating that the average age of 40 seventh-graders (average age 13) and 60 guardians (average age 40) is 29.2 -/
theorem average_age_of_group : average_age 40 13 60 40 = 29.2 := by
  sorry

end NUMINAMATH_CALUDE_average_age_of_group_l3715_371508


namespace NUMINAMATH_CALUDE_father_son_age_ratio_l3715_371542

theorem father_son_age_ratio : 
  ∀ (father_age son_age : ℕ),
  father_age * son_age = 756 →
  (father_age + 6) / (son_age + 6) = 2 →
  father_age / son_age = 7 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_father_son_age_ratio_l3715_371542


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l3715_371529

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, a (n + 1) = 2 * a n

theorem geometric_sequence_product (a : ℕ → ℝ) 
  (h : geometric_sequence a) : a 3 * a 5 = 64 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l3715_371529


namespace NUMINAMATH_CALUDE_cow_count_l3715_371519

/-- Represents a group of ducks and cows -/
structure AnimalGroup where
  ducks : ℕ
  cows : ℕ

/-- The total number of legs in the group -/
def totalLegs (group : AnimalGroup) : ℕ :=
  2 * group.ducks + 4 * group.cows

/-- The total number of heads in the group -/
def totalHeads (group : AnimalGroup) : ℕ :=
  group.ducks + group.cows

/-- Theorem: In a group where the total number of legs is 12 more than twice 
    the number of heads, the number of cows is 6 -/
theorem cow_count (group : AnimalGroup) 
    (h : totalLegs group = 2 * totalHeads group + 12) : 
    group.cows = 6 := by
  sorry


end NUMINAMATH_CALUDE_cow_count_l3715_371519


namespace NUMINAMATH_CALUDE_solution_for_all_polynomials_l3715_371539

/-- A polynomial of degree 3 in x and y -/
def q (b₁ b₂ b₄ b₇ b₈ : ℝ) (x y : ℝ) : ℝ :=
  b₁ * x * (1 - x^2) + b₂ * y * (1 - y^2) + b₄ * (x * y - x^2 * y) + b₇ * x^2 * y + b₈ * x * y^2

/-- The theorem stating that (√(3/2), √(3/2)) is a solution for all such polynomials -/
theorem solution_for_all_polynomials (b₁ b₂ b₄ b₇ b₈ : ℝ) :
  let q := q b₁ b₂ b₄ b₇ b₈
  (q 0 0 = 0) →
  (q 1 0 = 0) →
  (q (-1) 0 = 0) →
  (q 0 1 = 0) →
  (q 0 (-1) = 0) →
  (q 1 1 = 0) →
  (q (-1) (-1) = 0) →
  (q 2 2 = 0) →
  (deriv (fun x => q x 1) 1 = 0) →
  (deriv (fun y => q 1 y) 1 = 0) →
  q (Real.sqrt (3/2)) (Real.sqrt (3/2)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_solution_for_all_polynomials_l3715_371539


namespace NUMINAMATH_CALUDE_break_room_vacant_seats_l3715_371571

theorem break_room_vacant_seats :
  let total_tables : ℕ := 5
  let seats_per_table : ℕ := 8
  let occupied_tables : ℕ := 2
  let people_per_occupied_table : ℕ := 3
  let unusable_tables : ℕ := 1

  let usable_tables : ℕ := total_tables - unusable_tables
  let total_seats : ℕ := usable_tables * seats_per_table
  let occupied_seats : ℕ := occupied_tables * people_per_occupied_table

  total_seats - occupied_seats = 26 :=
by sorry

end NUMINAMATH_CALUDE_break_room_vacant_seats_l3715_371571


namespace NUMINAMATH_CALUDE_diamond_equal_is_three_lines_l3715_371582

/-- The diamond operation -/
def diamond (a b : ℝ) : ℝ := a^3 * b - a * b^3

/-- The set of points (x, y) where x ◇ y = y ◇ x -/
def diamond_equal_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | diamond p.1 p.2 = diamond p.2 p.1}

/-- The union of three lines: x = 0, y = 0, and y = -x -/
def three_lines : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = 0 ∨ p.2 = 0 ∨ p.2 = -p.1}

theorem diamond_equal_is_three_lines :
  diamond_equal_set = three_lines :=
sorry

end NUMINAMATH_CALUDE_diamond_equal_is_three_lines_l3715_371582


namespace NUMINAMATH_CALUDE_max_min_difference_l3715_371586

theorem max_min_difference (a b : ℝ) : 
  a^2 + b^2 - 2*a - 4 = 0 → 
  (∃ (t_max t_min : ℝ), 
    (∀ t : ℝ, (∃ a' b' : ℝ, a'^2 + b'^2 - 2*a' - 4 = 0 ∧ t = 2*a' - b') → t_min ≤ t ∧ t ≤ t_max) ∧
    t_max - t_min = 10) :=
by sorry

end NUMINAMATH_CALUDE_max_min_difference_l3715_371586


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3715_371584

-- Problem 1
theorem problem_1 : (-5) * (-7) + 20 / (-4) = 30 := by sorry

-- Problem 2
theorem problem_2 : (1/9 + 1/6 - 1/4) * (-36) = -1 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3715_371584


namespace NUMINAMATH_CALUDE_exists_valid_configuration_l3715_371595

/-- A configuration of 9 numbers placed in circles -/
def Configuration := Fin 9 → Nat

/-- The 6 lines connecting the circles -/
def Lines := Fin 6 → Fin 3 → Fin 9

/-- Check if a configuration is valid -/
def is_valid_configuration (config : Configuration) (lines : Lines) : Prop :=
  (∀ i : Fin 9, config i ∈ Finset.range 10 \ {0}) ∧  -- Numbers are from 1 to 9
  (∃ i : Fin 9, config i = 6) ∧                      -- 6 is included
  (∀ i j : Fin 9, i ≠ j → config i ≠ config j) ∧     -- All numbers are different
  (∀ l : Fin 6, (config (lines l 0) + config (lines l 1) + config (lines l 2) = 23))  -- Sum on each line is 23

theorem exists_valid_configuration (lines : Lines) : 
  ∃ (config : Configuration), is_valid_configuration config lines :=
sorry

end NUMINAMATH_CALUDE_exists_valid_configuration_l3715_371595


namespace NUMINAMATH_CALUDE_product_expansion_l3715_371500

theorem product_expansion (x : ℝ) : 2 * (x + 3) * (x + 4) = 2 * x^2 + 14 * x + 24 := by
  sorry

end NUMINAMATH_CALUDE_product_expansion_l3715_371500


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l3715_371568

theorem solve_exponential_equation :
  ∃ n : ℕ, 8^n * 8^n * 8^n = 64^3 ∧ n = 2 := by sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l3715_371568


namespace NUMINAMATH_CALUDE_smallest_n_property_l3715_371517

/-- The smallest positive integer N such that N and N^2 end in the same three-digit sequence abc in base 10, where a is not zero -/
def smallest_n : ℕ := 876

theorem smallest_n_property : 
  ∀ n : ℕ, n > 0 → 
  (n % 1000 = smallest_n % 1000 ∧ n^2 % 1000 = smallest_n % 1000 ∧ (smallest_n % 1000) ≥ 100) → 
  n ≥ smallest_n := by
  sorry

#eval smallest_n

end NUMINAMATH_CALUDE_smallest_n_property_l3715_371517


namespace NUMINAMATH_CALUDE_soap_packing_problem_l3715_371510

theorem soap_packing_problem :
  ∃! N : ℕ, 200 < N ∧ N < 300 ∧ 2007 % N = 5 := by
  sorry

end NUMINAMATH_CALUDE_soap_packing_problem_l3715_371510


namespace NUMINAMATH_CALUDE_smallest_cube_side_length_is_four_l3715_371550

/-- A cube that can contain two non-overlapping spheres of radius 1 -/
structure Cube :=
  (side_length : ℝ)
  (contains_spheres : side_length ≥ 4)

/-- The smallest side length of a cube that can contain two non-overlapping spheres of radius 1 -/
def smallest_cube_side_length : ℝ := 4

/-- Theorem: The smallest side length of a cube that can contain two non-overlapping spheres of radius 1 is 4 -/
theorem smallest_cube_side_length_is_four :
  ∀ (c : Cube), c.side_length ≥ smallest_cube_side_length :=
sorry

end NUMINAMATH_CALUDE_smallest_cube_side_length_is_four_l3715_371550


namespace NUMINAMATH_CALUDE_theater_ticket_difference_l3715_371513

theorem theater_ticket_difference :
  ∀ (x y : ℕ),
    x + y = 350 →
    12 * x + 8 * y = 3320 →
    y - x = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_theater_ticket_difference_l3715_371513


namespace NUMINAMATH_CALUDE_initial_stamp_ratio_l3715_371560

theorem initial_stamp_ratio (p q : ℕ) : 
  (p - 8 : ℚ) / (q + 8 : ℚ) = 6 / 5 →
  p - 8 = q + 8 →
  (p : ℚ) / q = 6 / 5 := by
sorry

end NUMINAMATH_CALUDE_initial_stamp_ratio_l3715_371560


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3715_371502

def i : ℂ := Complex.I

theorem imaginary_part_of_z (z : ℂ) (h : (i - 1) * z = i) : 
  z.im = -1/2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3715_371502


namespace NUMINAMATH_CALUDE_tournament_winner_percentage_l3715_371518

theorem tournament_winner_percentage (n : ℕ) (total_games : ℕ) 
  (top_player_advantage : ℝ) (least_successful_percentage : ℝ) 
  (remaining_players_percentage : ℝ) :
  n = 8 →
  total_games = 560 →
  top_player_advantage = 0.15 →
  least_successful_percentage = 0.08 →
  remaining_players_percentage = 0.35 →
  ∃ (top_player_percentage : ℝ),
    top_player_percentage = 0.395 ∧
    top_player_percentage = 
      (1 - (2 * least_successful_percentage + remaining_players_percentage)) / 2 + 
      top_player_advantage :=
by sorry

end NUMINAMATH_CALUDE_tournament_winner_percentage_l3715_371518


namespace NUMINAMATH_CALUDE_arcsin_sqrt2_over_2_l3715_371569

theorem arcsin_sqrt2_over_2 : Real.arcsin (Real.sqrt 2 / 2) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_sqrt2_over_2_l3715_371569


namespace NUMINAMATH_CALUDE_fraction_simplification_l3715_371591

theorem fraction_simplification (x : ℝ) : (x + 1) / 3 + (2 - 3 * x) / 2 = (8 - 7 * x) / 6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3715_371591


namespace NUMINAMATH_CALUDE_M_minimum_l3715_371557

/-- The function M to be minimized -/
def M (x y : ℝ) : ℝ := 4*x^2 - 12*x*y + 10*y^2 + 4*y + 9

/-- The theorem stating the minimum value of M and where it occurs -/
theorem M_minimum :
  (∀ x y : ℝ, M x y ≥ 5) ∧ M (-3) (-2) = 5 := by sorry

end NUMINAMATH_CALUDE_M_minimum_l3715_371557


namespace NUMINAMATH_CALUDE_riding_to_total_ratio_l3715_371580

/-- Given a group of horses and men with specific conditions, 
    prove the ratio of riding owners to total owners --/
theorem riding_to_total_ratio 
  (total_horses : ℕ) 
  (total_men : ℕ) 
  (legs_on_ground : ℕ) 
  (h1 : total_horses = 16)
  (h2 : total_men = total_horses)
  (h3 : legs_on_ground = 80) : 
  (total_horses - (legs_on_ground - 4 * total_horses) / 2) / total_horses = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_riding_to_total_ratio_l3715_371580


namespace NUMINAMATH_CALUDE_inequality_system_equivalence_l3715_371597

theorem inequality_system_equivalence :
  ∀ x : ℝ, (x + 1 ≥ 2 ∧ x > 0) ↔ x ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_equivalence_l3715_371597


namespace NUMINAMATH_CALUDE_play_roles_assignment_l3715_371516

/-- The number of ways to assign roles in a play -/
def assignRoles (numMen numWomen numMaleRoles numFemaleRoles numEitherRoles : ℕ) : ℕ :=
  let remainingActors := numMen + numWomen - numMaleRoles - numFemaleRoles
  (numMen.choose numMaleRoles) * 
  (numWomen.choose numFemaleRoles) * 
  (remainingActors.choose numEitherRoles)

theorem play_roles_assignment :
  assignRoles 6 7 3 3 3 = 5292000 := by
  sorry

end NUMINAMATH_CALUDE_play_roles_assignment_l3715_371516


namespace NUMINAMATH_CALUDE_never_exceeds_100_l3715_371552

def repeated_square (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | m + 1 => (repeated_square m) ^ 2

theorem never_exceeds_100 (n : ℕ) : repeated_square n ≤ 100 := by
  sorry

#check never_exceeds_100

end NUMINAMATH_CALUDE_never_exceeds_100_l3715_371552
