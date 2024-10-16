import Mathlib

namespace NUMINAMATH_CALUDE_product_of_A_and_B_l2217_221754

theorem product_of_A_and_B (A B : ℝ) (h1 : 3/9 = 6/A) (h2 : 6/A = B/63) : A * B = 378 := by
  sorry

end NUMINAMATH_CALUDE_product_of_A_and_B_l2217_221754


namespace NUMINAMATH_CALUDE_clock_rotation_proof_l2217_221748

/-- The number of large divisions on a clock face -/
def clock_divisions : ℕ := 12

/-- The number of degrees in one large division -/
def degrees_per_division : ℝ := 30

/-- The number of hours between 3 o'clock and 6 o'clock -/
def hours_elapsed : ℕ := 3

/-- The degree of rotation of the hour hand from 3 o'clock to 6 o'clock -/
def hour_hand_rotation : ℝ := hours_elapsed * degrees_per_division

theorem clock_rotation_proof :
  hour_hand_rotation = 90 :=
by sorry

end NUMINAMATH_CALUDE_clock_rotation_proof_l2217_221748


namespace NUMINAMATH_CALUDE_harry_apples_l2217_221769

/-- Proves that Harry has 19 apples given the conditions of the problem -/
theorem harry_apples :
  ∀ (martha_apples tim_apples harry_apples jane_apples : ℕ),
  martha_apples = 68 →
  tim_apples = martha_apples - 30 →
  harry_apples = tim_apples / 2 →
  jane_apples = ((martha_apples + tim_apples) * 25) / 100 →
  harry_apples = 19 := by
  sorry

#check harry_apples

end NUMINAMATH_CALUDE_harry_apples_l2217_221769


namespace NUMINAMATH_CALUDE_fence_painting_combinations_l2217_221731

theorem fence_painting_combinations :
  let color_choices : ℕ := 6
  let method_choices : ℕ := 3
  let finish_choices : ℕ := 2
  color_choices * method_choices * finish_choices = 36 :=
by sorry

end NUMINAMATH_CALUDE_fence_painting_combinations_l2217_221731


namespace NUMINAMATH_CALUDE_distance_between_points_l2217_221717

/-- The curve equation -/
def curve_equation (x y : ℝ) : Prop := y^2 + x^4 = 2*x^2*y + 1

/-- Theorem stating that for any real number e, if (e, a) and (e, b) are points on the curve y^2 + x^4 = 2x^2y + 1, then |a-b| = 2 -/
theorem distance_between_points (e a b : ℝ) 
  (ha : curve_equation e a) 
  (hb : curve_equation e b) : 
  |a - b| = 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l2217_221717


namespace NUMINAMATH_CALUDE_trigonometric_expression_value_l2217_221724

theorem trigonometric_expression_value (α : Real) (h : Real.tan α = -3/2) :
  (Real.cos (π/2 + α) * Real.sin (π + α)) / (Real.cos (π - α) * Real.sin (3*π - α)) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_value_l2217_221724


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2217_221702

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 > x) ↔ (∃ x : ℝ, x^2 ≤ x) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2217_221702


namespace NUMINAMATH_CALUDE_olympiad_score_problem_l2217_221734

theorem olympiad_score_problem :
  ∀ (x y : ℕ),
    x + y = 14 →
    7 * x - 12 * y = 60 →
    x = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_olympiad_score_problem_l2217_221734


namespace NUMINAMATH_CALUDE_compare_exponential_and_quadratic_l2217_221762

theorem compare_exponential_and_quadratic (n : ℕ) :
  (n ≥ 3 → 2^(2*n) > (2*n + 1)^2) ∧
  ((n = 1 ∨ n = 2) → 2^(2*n) < (2*n + 1)^2) := by
  sorry

end NUMINAMATH_CALUDE_compare_exponential_and_quadratic_l2217_221762


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2217_221720

theorem quadratic_inequality_solution (x : ℝ) :
  (-x^2 - 2*x + 3 ≤ 0) ↔ (x ≤ -3 ∨ x ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2217_221720


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l2217_221799

theorem circle_center_and_radius :
  let equation := (fun (x y : ℝ) => x^2 + y^2 - 2*x - 5 = 0)
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (1, 0) ∧ 
    radius = Real.sqrt 6 ∧
    ∀ (x y : ℝ), equation x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l2217_221799


namespace NUMINAMATH_CALUDE_milkshake_fraction_l2217_221735

theorem milkshake_fraction (total : ℚ) (milkshake_fraction : ℚ) 
  (lost : ℚ) (remaining : ℚ) : 
  total = 28 →
  lost = 11 →
  remaining = 1 →
  (1 - milkshake_fraction) * total / 2 = lost + remaining →
  milkshake_fraction = 1 / 7 := by
sorry

end NUMINAMATH_CALUDE_milkshake_fraction_l2217_221735


namespace NUMINAMATH_CALUDE_divisible_by_four_l2217_221719

theorem divisible_by_four (x : Nat) : 
  x < 10 → (3280 + x).mod 4 = 0 ↔ x = 0 ∨ x = 2 ∨ x = 4 ∨ x = 6 ∨ x = 8 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_four_l2217_221719


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l2217_221712

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l2217_221712


namespace NUMINAMATH_CALUDE_P_identity_l2217_221710

def P (n : ℕ) : ℕ := (n + 1).factorial / n.factorial

def oddProduct (n : ℕ) : ℕ :=
  Finset.prod (Finset.range n) (fun i => 2 * i + 1)

theorem P_identity (n : ℕ) : P n = 2^n * oddProduct n := by
  sorry

end NUMINAMATH_CALUDE_P_identity_l2217_221710


namespace NUMINAMATH_CALUDE_train_length_l2217_221741

theorem train_length (v : ℝ) (L : ℝ) : 
  v > 0 → -- The train's speed is positive
  (L + 120) / 60 = v → -- It takes 60 seconds to pass through a 120m tunnel
  L / 20 = v → -- It takes 20 seconds to be completely inside the tunnel
  L = 60 := by
sorry

end NUMINAMATH_CALUDE_train_length_l2217_221741


namespace NUMINAMATH_CALUDE_negation_of_proposition_l2217_221747

theorem negation_of_proposition (m : ℝ) :
  (¬(m > 0 → ∃ x : ℝ, x^2 + x - m = 0)) ↔ (m ≤ 0 → ∀ x : ℝ, x^2 + x - m ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l2217_221747


namespace NUMINAMATH_CALUDE_max_value_implies_m_equals_four_l2217_221709

theorem max_value_implies_m_equals_four (x y m : ℝ) : 
  x > 1 →
  y ≥ x →
  y ≤ 2 * x →
  x + y ≤ 1 →
  (∀ x' y' : ℝ, y' ≥ x' → y' ≤ 2 * x' → x' + y' ≤ 1 → x' + m * y' ≤ x + m * y) →
  x + m * y = 3 →
  m = 4 := by
sorry

end NUMINAMATH_CALUDE_max_value_implies_m_equals_four_l2217_221709


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l2217_221703

theorem polynomial_division_theorem (x : ℝ) :
  let dividend := x^5 - 20*x^3 + 15*x^2 - 18*x + 12
  let divisor := x - 2
  let quotient := x^4 + 2*x^3 - 16*x^2 - 17*x - 52
  let remainder := -92
  dividend = divisor * quotient + remainder := by sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l2217_221703


namespace NUMINAMATH_CALUDE_combined_weight_calculation_l2217_221781

/-- The combined weight of Leo and Kendra -/
def combinedWeight (leoWeight kenWeight : ℝ) : ℝ := leoWeight + kenWeight

/-- Leo's weight after gaining 10 pounds -/
def leoWeightGained (leoWeight : ℝ) : ℝ := leoWeight + 10

/-- Condition that Leo's weight after gaining 10 pounds is 50% more than Kendra's weight -/
def weightCondition (leoWeight kenWeight : ℝ) : Prop :=
  leoWeightGained leoWeight = kenWeight * 1.5

theorem combined_weight_calculation (leoWeight kenWeight : ℝ) 
  (h1 : leoWeight = 98) 
  (h2 : weightCondition leoWeight kenWeight) : 
  combinedWeight leoWeight kenWeight = 170 := by
  sorry

end NUMINAMATH_CALUDE_combined_weight_calculation_l2217_221781


namespace NUMINAMATH_CALUDE_inequality_proof_l2217_221783

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 1) :
  (x^3 / ((1+y)*(1+z))) + (y^3 / ((1+z)*(1+x))) + (z^3 / ((1+x)*(1+y))) ≥ 3/4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2217_221783


namespace NUMINAMATH_CALUDE_frog_edge_probability_l2217_221786

/-- Represents a position on the 4x4 grid -/
inductive Position
| Center : Position
| Edge : Position

/-- Represents the number of hops -/
def MaxHops : ℕ := 3

/-- The probability of reaching an edge from the center in n hops -/
def probability_reach_edge (n : ℕ) : ℚ :=
  sorry

/-- The 4x4 grid with wrapping and movement rules -/
structure Grid :=
  (size : ℕ := 4)
  (wrap : Bool := true)
  (diagonal_moves : Bool := false)

/-- Theorem: The probability of reaching an edge within 3 hops is 13/16 -/
theorem frog_edge_probability (g : Grid) : 
  probability_reach_edge MaxHops = 13/16 :=
sorry

end NUMINAMATH_CALUDE_frog_edge_probability_l2217_221786


namespace NUMINAMATH_CALUDE_student_count_l2217_221728

theorem student_count (rank_from_right rank_from_left : ℕ) 
  (h1 : rank_from_right = 21)
  (h2 : rank_from_left = 11) :
  rank_from_right + rank_from_left - 1 = 31 := by
  sorry

end NUMINAMATH_CALUDE_student_count_l2217_221728


namespace NUMINAMATH_CALUDE_least_integer_with_divisibility_conditions_l2217_221721

def is_prime (n : ℕ) : Prop := sorry

def is_consecutive (a b : ℕ) : Prop := b = a + 1 ∨ a = b + 1

theorem least_integer_with_divisibility_conditions (N : ℕ) : 
  (∀ k ∈ Finset.range 31, k ≠ 0 → ∃ (a b : ℕ), a ≠ b ∧ is_consecutive a b ∧ 
    (is_prime a ∨ is_prime b) ∧ 
    (∀ i ∈ Finset.range 31, i ≠ 0 ∧ i ≠ a ∧ i ≠ b → N % i = 0) ∧
    N % a ≠ 0 ∧ N % b ≠ 0) →
  N ≥ 8923714800 :=
sorry

end NUMINAMATH_CALUDE_least_integer_with_divisibility_conditions_l2217_221721


namespace NUMINAMATH_CALUDE_ratio_fraction_l2217_221752

theorem ratio_fraction (x y : ℚ) (h : x / y = 4 / 5) : (x + y) / (x - y) = -9 := by
  sorry

end NUMINAMATH_CALUDE_ratio_fraction_l2217_221752


namespace NUMINAMATH_CALUDE_q_satisfies_conditions_l2217_221753

/-- The quadratic polynomial q(x) that satisfies the given conditions -/
def q (x : ℚ) : ℚ := (17 * x^2 - 8 * x + 21) / 15

/-- Theorem stating that q(x) satisfies the required conditions -/
theorem q_satisfies_conditions :
  q (-2) = 7 ∧ q 1 = 2 ∧ q 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_q_satisfies_conditions_l2217_221753


namespace NUMINAMATH_CALUDE_color_film_fraction_l2217_221776

theorem color_film_fraction (x y : ℝ) (h1 : x > 0) (h2 : y > 0) : 
  let total_bw := 20 * x
  let total_color := 4 * y
  let selected_bw := y / (5 * x) * total_bw
  let selected_color := total_color
  let total_selected := selected_bw + selected_color
  (selected_color / total_selected) = 20 / 21 := by
  sorry

end NUMINAMATH_CALUDE_color_film_fraction_l2217_221776


namespace NUMINAMATH_CALUDE_max_condition_l2217_221732

/-- Given a function f with derivative f' and a parameter a, 
    proves that if f has a maximum at x = a and a < 0, then -1 < a < 0 -/
theorem max_condition (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, HasDerivAt f (a * (x + 1) * (x - a)) x) →
  a < 0 →
  (∀ x, f x ≤ f a) →
  -1 < a ∧ a < 0 :=
sorry

end NUMINAMATH_CALUDE_max_condition_l2217_221732


namespace NUMINAMATH_CALUDE_equation_solution_l2217_221707

theorem equation_solution : ∃! x : ℝ, 
  Real.sqrt x + Real.sqrt (x + 9) + 3 * Real.sqrt (x^2 + 9*x) + Real.sqrt (3*x + 27) = 45 - 3*x ∧ 
  x = 729/144 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2217_221707


namespace NUMINAMATH_CALUDE_regular_octagon_interior_angle_measure_l2217_221751

/-- The measure of an interior angle of a regular octagon in degrees -/
def regular_octagon_interior_angle : ℝ := 135

/-- A regular octagon has 8 sides -/
def regular_octagon_sides : ℕ := 8

theorem regular_octagon_interior_angle_measure :
  regular_octagon_interior_angle = (((regular_octagon_sides - 2) * 180) : ℝ) / regular_octagon_sides :=
sorry

end NUMINAMATH_CALUDE_regular_octagon_interior_angle_measure_l2217_221751


namespace NUMINAMATH_CALUDE_oil_for_rest_of_bike_l2217_221749

/-- Proves the amount of oil needed for the rest of the bike --/
theorem oil_for_rest_of_bike 
  (oil_per_wheel : ℝ) 
  (num_wheels : ℕ) 
  (total_oil : ℝ) 
  (h1 : oil_per_wheel = 10)
  (h2 : num_wheels = 2)
  (h3 : total_oil = 25) :
  total_oil - (oil_per_wheel * num_wheels) = 5 := by
sorry

end NUMINAMATH_CALUDE_oil_for_rest_of_bike_l2217_221749


namespace NUMINAMATH_CALUDE_sum_equals_221_2357_l2217_221742

theorem sum_equals_221_2357 : 217 + 2.017 + 0.217 + 2.0017 = 221.2357 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_221_2357_l2217_221742


namespace NUMINAMATH_CALUDE_simplify_expression_l2217_221788

theorem simplify_expression : (4^7 + 2^6) * (1^5 - (-1)^5)^10 * (2^3 + 4^2) = 404225648 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2217_221788


namespace NUMINAMATH_CALUDE_red_balls_count_l2217_221766

theorem red_balls_count (total : Nat) (white green yellow purple : Nat) (prob : Real) :
  total = 100 ∧ 
  white = 50 ∧ 
  green = 20 ∧ 
  yellow = 10 ∧ 
  purple = 3 ∧ 
  prob = 0.8 ∧ 
  prob = (white + green + yellow : Real) / total →
  total - (white + green + yellow + purple) = 17 := by
  sorry

end NUMINAMATH_CALUDE_red_balls_count_l2217_221766


namespace NUMINAMATH_CALUDE_rain_probability_l2217_221737

theorem rain_probability (M T N : ℝ) 
  (hM : M = 0.6)  -- 60% of counties received rain on Monday
  (hT : T = 0.55) -- 55% of counties received rain on Tuesday
  (hN : N = 0.25) -- 25% of counties received no rain on either day
  : M + T - N - 1 = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_rain_probability_l2217_221737


namespace NUMINAMATH_CALUDE_f_equals_g_l2217_221750

theorem f_equals_g (f g : ℝ → ℝ) 
  (hf_cont : Continuous f)
  (hg_mono : Monotone g)
  (h_seq : ∀ a b c : ℝ, a < b → b < c → 
    ∃ (x : ℕ → ℝ), (∀ ε > 0, ∃ N, ∀ n ≥ N, |x n - b| < ε) ∧ 
    (∃ L : ℝ, ∀ ε > 0, ∃ N, ∀ n ≥ N, |g (x n) - L| < ε) ∧
    f a < L ∧ L < f c) :
  f = g := by
sorry

end NUMINAMATH_CALUDE_f_equals_g_l2217_221750


namespace NUMINAMATH_CALUDE_n_pointed_star_interior_angle_sum_l2217_221716

/-- An n-pointed star where n is a multiple of 3 and n ≥ 6 -/
structure NPointedStar where
  n : ℕ
  n_multiple_of_3 : 3 ∣ n
  n_ge_6 : n ≥ 6

/-- The sum of interior angles of an n-pointed star -/
def interior_angle_sum (star : NPointedStar) : ℝ :=
  180 * (star.n - 4)

/-- Theorem: The sum of interior angles of an n-pointed star is 180° (n-4) -/
theorem n_pointed_star_interior_angle_sum (star : NPointedStar) :
  interior_angle_sum star = 180 * (star.n - 4) := by
  sorry

end NUMINAMATH_CALUDE_n_pointed_star_interior_angle_sum_l2217_221716


namespace NUMINAMATH_CALUDE_twelve_pointed_stars_count_l2217_221746

/-- Counts the number of non-similar regular n-pointed stars -/
def count_non_similar_stars (n : ℕ) : ℕ :=
  let valid_m := (Finset.range (n - 1)).filter (λ m => m > 1 ∧ m < n - 1 ∧ Nat.gcd m n = 1)
  (valid_m.card + 1) / 2

/-- The number of non-similar regular 12-pointed stars is 1 -/
theorem twelve_pointed_stars_count :
  count_non_similar_stars 12 = 1 := by
  sorry

end NUMINAMATH_CALUDE_twelve_pointed_stars_count_l2217_221746


namespace NUMINAMATH_CALUDE_junk_mail_for_block_l2217_221761

/-- Given a block with houses and junk mail distribution, calculate the total junk mail for the block. -/
def total_junk_mail (num_houses : ℕ) (pieces_per_house : ℕ) : ℕ :=
  num_houses * pieces_per_house

/-- Theorem: The total junk mail for a block with 6 houses, each receiving 4 pieces, is 24. -/
theorem junk_mail_for_block :
  total_junk_mail 6 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_junk_mail_for_block_l2217_221761


namespace NUMINAMATH_CALUDE_remaining_pepper_l2217_221745

/-- Calculates the remaining amount of pepper after usage and addition -/
theorem remaining_pepper (initial : ℝ) (used : ℝ) (added : ℝ) (remaining : ℝ) :
  initial = 0.25 →
  used = 0.16 →
  remaining = initial - used + added →
  remaining = 0.09 + added :=
by sorry

end NUMINAMATH_CALUDE_remaining_pepper_l2217_221745


namespace NUMINAMATH_CALUDE_identity_function_satisfies_equation_l2217_221778

theorem identity_function_satisfies_equation (f : ℕ → ℕ) :
  (∀ m n : ℕ, f (f m + f n) = m + n) → (∀ n : ℕ, f n = n) := by
  sorry

end NUMINAMATH_CALUDE_identity_function_satisfies_equation_l2217_221778


namespace NUMINAMATH_CALUDE_minor_premise_identification_l2217_221792

-- Define the type for functions
def Function := Type → Type

-- Define properties
def IsTrigonometric (f : Function) : Prop := sorry
def IsPeriodic (f : Function) : Prop := sorry

-- Define tan function
def tan : Function := sorry

-- Theorem statement
theorem minor_premise_identification :
  (∀ f : Function, IsTrigonometric f → IsPeriodic f) →  -- major premise
  (IsTrigonometric tan) →                               -- minor premise
  (IsPeriodic tan) →                                    -- conclusion
  (IsTrigonometric tan)                                 -- proves minor premise
  := by sorry

end NUMINAMATH_CALUDE_minor_premise_identification_l2217_221792


namespace NUMINAMATH_CALUDE_bf_length_l2217_221760

-- Define the points
variable (A B C D E F : ℝ × ℝ)

-- Define the conditions
variable (h1 : (A.1 = C.1 ∧ A.2 = D.2) ∧ (C.1 = D.1 ∧ C.2 = B.2))  -- right angles at A and C
variable (h2 : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ E = (1 - t) • A + t • C)  -- E is on AC
variable (h3 : ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ F = (1 - s) • A + s • C)  -- F is on AC
variable (h4 : (D.1 - E.1) * (C.1 - A.1) + (D.2 - E.2) * (C.2 - A.2) = 0)  -- DE perpendicular to AC
variable (h5 : (B.1 - F.1) * (C.1 - A.1) + (B.2 - F.2) * (C.2 - A.2) = 0)  -- BF perpendicular to AC
variable (h6 : Real.sqrt ((A.1 - E.1)^2 + (A.2 - E.2)^2) = 4)  -- AE = 4
variable (h7 : Real.sqrt ((D.1 - E.1)^2 + (D.2 - E.2)^2) = 4)  -- DE = 4
variable (h8 : Real.sqrt ((C.1 - E.1)^2 + (C.2 - E.2)^2) = 6)  -- CE = 6

-- Theorem statement
theorem bf_length : Real.sqrt ((B.1 - F.1)^2 + (B.2 - F.2)^2) = 4 :=
sorry

end NUMINAMATH_CALUDE_bf_length_l2217_221760


namespace NUMINAMATH_CALUDE_math_city_intersections_l2217_221795

/-- The number of intersections for n non-parallel streets where no three streets meet at a single point -/
def intersections (n : ℕ) : ℕ := (n - 1) * n / 2

/-- The number of streets in Math City -/
def num_streets : ℕ := 10

theorem math_city_intersections :
  intersections num_streets = 45 :=
by sorry

end NUMINAMATH_CALUDE_math_city_intersections_l2217_221795


namespace NUMINAMATH_CALUDE_cricket_average_increase_l2217_221706

theorem cricket_average_increase 
  (innings : ℕ) 
  (current_average : ℚ) 
  (next_innings_score : ℕ) 
  (average_increase : ℚ) : 
  innings = 20 → 
  current_average = 36 → 
  next_innings_score = 120 → 
  (innings : ℚ) * current_average + next_innings_score = (innings + 1) * (current_average + average_increase) → 
  average_increase = 4 := by
sorry

end NUMINAMATH_CALUDE_cricket_average_increase_l2217_221706


namespace NUMINAMATH_CALUDE_arithmetic_mean_difference_l2217_221725

theorem arithmetic_mean_difference (p q r : ℝ) 
  (h1 : (p + q) / 2 = 10) 
  (h2 : (q + r) / 2 = 22) : 
  r - p = 24 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_difference_l2217_221725


namespace NUMINAMATH_CALUDE_bench_and_student_count_l2217_221700

theorem bench_and_student_count :
  ∃ (a b s : ℕ), 
    (s = a * b + 5 ∧ s = 8 * b - 4) →
    ((b = 9 ∧ s = 68) ∨ (b = 3 ∧ s = 20)) := by
  sorry

end NUMINAMATH_CALUDE_bench_and_student_count_l2217_221700


namespace NUMINAMATH_CALUDE_aaron_can_lids_l2217_221740

theorem aaron_can_lids (num_boxes : ℕ) (lids_per_box : ℕ) (total_lids : ℕ) :
  num_boxes = 3 →
  lids_per_box = 13 →
  total_lids = 53 →
  total_lids - (num_boxes * lids_per_box) = 14 := by
  sorry

end NUMINAMATH_CALUDE_aaron_can_lids_l2217_221740


namespace NUMINAMATH_CALUDE_height_difference_l2217_221798

/-- The height difference between the tallest and shortest players on a basketball team. -/
theorem height_difference (tallest_height shortest_height : ℝ) 
  (h_tallest : tallest_height = 77.75)
  (h_shortest : shortest_height = 68.25) : 
  tallest_height - shortest_height = 9.5 := by
  sorry

end NUMINAMATH_CALUDE_height_difference_l2217_221798


namespace NUMINAMATH_CALUDE_f_is_even_l2217_221768

-- Define g as an odd function
def g_odd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

-- Define f in terms of g
def f (g : ℝ → ℝ) (x : ℝ) : ℝ := |g (x^3)|

-- Theorem statement
theorem f_is_even (g : ℝ → ℝ) (h : g_odd g) : ∀ x, f g (-x) = f g x := by sorry

end NUMINAMATH_CALUDE_f_is_even_l2217_221768


namespace NUMINAMATH_CALUDE_line_intercepts_sum_l2217_221743

theorem line_intercepts_sum (d : ℚ) : 
  (∃ (x y : ℚ), 6 * x + 5 * y + d = 0 ∧ x + y = 15) → d = -450 / 11 := by
  sorry

end NUMINAMATH_CALUDE_line_intercepts_sum_l2217_221743


namespace NUMINAMATH_CALUDE_systematic_sample_selection_l2217_221784

/-- Represents a systematic sampling scheme -/
structure SystematicSample where
  total : ℕ
  sample_size : ℕ
  start : ℕ
  interval : ℕ

/-- Checks if a number is selected in a systematic sample -/
def is_selected (s : SystematicSample) (n : ℕ) : Prop :=
  ∃ k : ℕ, n = s.start + k * s.interval ∧ n ≤ s.total

theorem systematic_sample_selection 
  (s : SystematicSample)
  (h_total : s.total = 900)
  (h_size : s.sample_size = 150)
  (h_start : s.start = 15)
  (h_interval : s.interval = s.total / s.sample_size)
  (h_15_selected : is_selected s 15)
  : is_selected s 81 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sample_selection_l2217_221784


namespace NUMINAMATH_CALUDE_find_boys_in_first_group_l2217_221779

/-- Represents the daily work done by a single person -/
structure WorkRate :=
  (amount : ℝ)

/-- Represents a group of workers -/
structure WorkGroup :=
  (men : ℕ)
  (boys : ℕ)

/-- Represents the time taken to complete a job -/
def completeJob (g : WorkGroup) (d : ℕ) (m : WorkRate) (b : WorkRate) : ℝ :=
  d * (g.men * m.amount + g.boys * b.amount)

theorem find_boys_in_first_group :
  ∀ (m b : WorkRate) (x : ℕ),
    m.amount = 2 * b.amount →
    completeJob ⟨12, x⟩ 5 m b = completeJob ⟨13, 24⟩ 4 m b →
    x = 16 := by
  sorry

end NUMINAMATH_CALUDE_find_boys_in_first_group_l2217_221779


namespace NUMINAMATH_CALUDE_sum_and_multiply_l2217_221771

theorem sum_and_multiply : (57.6 + 1.4) * 3 = 177 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_multiply_l2217_221771


namespace NUMINAMATH_CALUDE_first_three_seeds_l2217_221759

/-- Represents a random number table --/
def RandomNumberTable := List (List Nat)

/-- Checks if a number is a valid seed number --/
def isValidSeedNumber (n : Nat) : Bool :=
  1 ≤ n ∧ n ≤ 850

/-- Extracts numbers from the random number table --/
def extractNumbers (table : RandomNumberTable) (startRow : Nat) (startCol : Nat) (count : Nat) : List Nat :=
  sorry

/-- Selects valid seed numbers from a list of numbers --/
def selectValidSeedNumbers (numbers : List Nat) (count : Nat) : List Nat :=
  sorry

theorem first_three_seeds (table : RandomNumberTable) :
  let extractedNumbers := extractNumbers table 8 7 10
  let selectedSeeds := selectValidSeedNumbers extractedNumbers 3
  selectedSeeds = [785, 567, 199] := by
  sorry

end NUMINAMATH_CALUDE_first_three_seeds_l2217_221759


namespace NUMINAMATH_CALUDE_limit_of_a_l2217_221777

def a (n : ℕ) : ℚ := (2 * n + 1) / (5 * n - 1)

theorem limit_of_a :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |a n - 2/5| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_limit_of_a_l2217_221777


namespace NUMINAMATH_CALUDE_sum_in_D_l2217_221704

-- Define the sets A, B, C, and D
def A : Set Int := {x | ∃ k, x = 4 * k}
def B : Set Int := {x | ∃ m, x = 4 * m + 1}
def C : Set Int := {x | ∃ n, x = 4 * n + 2}
def D : Set Int := {x | ∃ t, x = 4 * t + 3}

-- State the theorem
theorem sum_in_D (a b : Int) (ha : a ∈ B) (hb : b ∈ C) : a + b ∈ D := by
  sorry

end NUMINAMATH_CALUDE_sum_in_D_l2217_221704


namespace NUMINAMATH_CALUDE_point_line_plane_membership_l2217_221773

-- Define the types for points, lines, and planes
variable (Point Line Plane : Type)

-- Define the relations for a point being on a line and within a plane
variable (on_line : Point → Line → Prop)
variable (in_plane : Point → Plane → Prop)

-- Define specific points, line, and plane
variable (A E F : Point)
variable (l : Line)
variable (ABC : Plane)

-- State the theorem
theorem point_line_plane_membership :
  (on_line A l) ∧ (in_plane E ABC) ∧ (in_plane F ABC) :=
sorry

end NUMINAMATH_CALUDE_point_line_plane_membership_l2217_221773


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l2217_221763

theorem sqrt_meaningful_range (x : ℝ) :
  (∃ y : ℝ, y ^ 2 = x - 3) ↔ x ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l2217_221763


namespace NUMINAMATH_CALUDE_f_geq_one_for_a_eq_two_g_min_value_g_min_value_exists_l2217_221789

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |2*x - a| + |x + 2/a|

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := f a x + f a (-x)

-- Theorem for part (1)
theorem f_geq_one_for_a_eq_two (x : ℝ) : f 2 x ≥ 1 := by sorry

-- Theorem for part (2)
theorem g_min_value (a : ℝ) : ∀ x : ℝ, g a x ≥ 4 * Real.sqrt 2 := by sorry

-- Theorem for existence of x that achieves the minimum value
theorem g_min_value_exists (a : ℝ) : ∃ x : ℝ, g a x = 4 * Real.sqrt 2 := by sorry

end

end NUMINAMATH_CALUDE_f_geq_one_for_a_eq_two_g_min_value_g_min_value_exists_l2217_221789


namespace NUMINAMATH_CALUDE_scale_model_height_l2217_221739

/-- The scale ratio of the model -/
def scale_ratio : ℚ := 1 / 25

/-- The actual height of the Eiffel Tower in feet -/
def actual_height : ℕ := 1063

/-- The height of the scale model before rounding -/
def model_height : ℚ := actual_height * scale_ratio

/-- Rounds a rational number to the nearest integer -/
def round_to_nearest (q : ℚ) : ℤ :=
  ⌊q + 1/2⌋

theorem scale_model_height :
  round_to_nearest model_height = 43 := by sorry

end NUMINAMATH_CALUDE_scale_model_height_l2217_221739


namespace NUMINAMATH_CALUDE_beth_coin_ratio_l2217_221775

/-- Proves that the ratio of coins Beth sold to her total coins after receiving Carl's gift is 1:2 -/
theorem beth_coin_ratio :
  let initial_coins : ℕ := 125
  let gift_coins : ℕ := 35
  let sold_coins : ℕ := 80
  let total_coins : ℕ := initial_coins + gift_coins
  (sold_coins : ℚ) / total_coins = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_beth_coin_ratio_l2217_221775


namespace NUMINAMATH_CALUDE_shaded_region_perimeter_l2217_221774

/-- The perimeter of a region consisting of two radii of length 5 and a 3/4 circular arc of a circle with radius 5 is equal to 10 + (15π/2). -/
theorem shaded_region_perimeter (r : ℝ) (h : r = 5) :
  2 * r + (3/4) * (2 * π * r) = 10 + (15 * π) / 2 :=
by sorry

end NUMINAMATH_CALUDE_shaded_region_perimeter_l2217_221774


namespace NUMINAMATH_CALUDE_right_triangle_with_incircle_legs_l2217_221738

/-- A right-angled triangle with an incircle touching the hypotenuse -/
structure RightTriangleWithIncircle where
  -- The lengths of the sides
  a : ℝ
  b : ℝ
  c : ℝ
  -- The lengths of the segments of the hypotenuse
  ap : ℝ
  bp : ℝ
  -- Conditions
  right_angle : a^2 + b^2 = c^2
  hypotenuse : c = ap + bp
  incircle_property : ap = (a + b + c) / 2 - a ∧ bp = (a + b + c) / 2 - b
  -- Given values
  ap_value : ap = 12
  bp_value : bp = 5

/-- The main theorem -/
theorem right_triangle_with_incircle_legs 
  (triangle : RightTriangleWithIncircle) : 
  triangle.a = 8 ∧ triangle.b = 15 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_with_incircle_legs_l2217_221738


namespace NUMINAMATH_CALUDE_function_minimum_value_l2217_221770

theorem function_minimum_value (x : ℝ) (h : x > -1) :
  (x^2 + 7*x + 10) / (x + 1) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_function_minimum_value_l2217_221770


namespace NUMINAMATH_CALUDE_fraction_simplification_l2217_221790

theorem fraction_simplification (x : ℝ) : (3*x + 2) / 4 + (x - 4) / 3 = (13*x - 10) / 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2217_221790


namespace NUMINAMATH_CALUDE_remainder_mod_11_l2217_221782

theorem remainder_mod_11 : (7 * 10^20 + 2^20) % 11 = 8 := by
  sorry

end NUMINAMATH_CALUDE_remainder_mod_11_l2217_221782


namespace NUMINAMATH_CALUDE_distance_calculation_l2217_221715

def speed : Real := 20
def time : Real := 8
def distance : Real := speed * time

theorem distance_calculation : distance = 160 := by
  sorry

end NUMINAMATH_CALUDE_distance_calculation_l2217_221715


namespace NUMINAMATH_CALUDE_polynomial_factor_theorem_l2217_221794

theorem polynomial_factor_theorem (a : ℝ) : 
  (∃ b : ℝ, ∀ y : ℝ, y^2 + 3*y - a = (y - 3) * (y + b)) → a = 18 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factor_theorem_l2217_221794


namespace NUMINAMATH_CALUDE_certain_number_proof_l2217_221701

theorem certain_number_proof (p q : ℚ) 
  (h1 : 3 / p = 8)
  (h2 : 3 / q = 18)
  (h3 : p - q = 0.20833333333333334) : 
  q = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l2217_221701


namespace NUMINAMATH_CALUDE_total_money_l2217_221756

theorem total_money (mark : ℚ) (carolyn : ℚ) (david : ℚ)
  (h1 : mark = 5 / 6)
  (h2 : carolyn = 4 / 9)
  (h3 : david = 7 / 12) :
  mark + carolyn + david = 67 / 36 := by
  sorry

end NUMINAMATH_CALUDE_total_money_l2217_221756


namespace NUMINAMATH_CALUDE_man_walking_speed_percentage_l2217_221785

/-- Proves that if a man's usual time to cover a distance is 24 minutes, and he takes 24 minutes more 
    when walking at a reduced speed, then the reduced speed is 50% of his usual speed. -/
theorem man_walking_speed_percentage (usual_time reduced_time : ℝ) 
    (h1 : usual_time = 24)
    (h2 : reduced_time = usual_time + 24) :
    reduced_time / usual_time = 2 := by
  sorry

#check man_walking_speed_percentage

end NUMINAMATH_CALUDE_man_walking_speed_percentage_l2217_221785


namespace NUMINAMATH_CALUDE_sin_squared_sum_6_to_174_l2217_221729

theorem sin_squared_sum_6_to_174 : 
  (Finset.range 29).sum (fun k => Real.sin ((6 * k + 6 : ℕ) * π / 180) ^ 2) = 31 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_squared_sum_6_to_174_l2217_221729


namespace NUMINAMATH_CALUDE_village_panic_percentage_l2217_221780

theorem village_panic_percentage (original_population : ℕ) 
  (initial_disappearance_rate : ℚ) (final_population : ℕ) :
  original_population = 7200 →
  initial_disappearance_rate = 1/10 →
  final_population = 4860 →
  (1 - (final_population : ℚ) / ((1 - initial_disappearance_rate) * original_population)) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_village_panic_percentage_l2217_221780


namespace NUMINAMATH_CALUDE_eggs_remaining_proof_l2217_221723

/-- Calculates the number of eggs remaining after eating some eggs --/
def remaining_eggs (initial : ℕ) (morning : ℕ) (afternoon : ℕ) : ℕ :=
  initial - (morning + afternoon)

/-- Proves that given 20 initial eggs, after eating 4 in the morning and 3 in the afternoon, 13 eggs remain --/
theorem eggs_remaining_proof :
  remaining_eggs 20 4 3 = 13 := by
  sorry

end NUMINAMATH_CALUDE_eggs_remaining_proof_l2217_221723


namespace NUMINAMATH_CALUDE_pauline_snow_shoveling_l2217_221744

/-- Calculates the total volume of snow shoveled up to a given hour -/
def snowShoveled (hour : ℕ) : ℕ :=
  (20 * hour) - (hour * (hour - 1) / 2)

/-- Represents Pauline's snow shoveling problem -/
theorem pauline_snow_shoveling (drivewayWidth drivewayLength snowDepth : ℕ) 
  (h1 : drivewayWidth = 5)
  (h2 : drivewayLength = 10)
  (h3 : snowDepth = 4) :
  ∃ (hour : ℕ), hour = 13 ∧ snowShoveled hour ≥ drivewayWidth * drivewayLength * snowDepth ∧ 
  snowShoveled (hour - 1) < drivewayWidth * drivewayLength * snowDepth :=
by
  sorry


end NUMINAMATH_CALUDE_pauline_snow_shoveling_l2217_221744


namespace NUMINAMATH_CALUDE_intersection_theorem_l2217_221705

/-- The intersection point of two lines in 2D space -/
def intersection_point : ℚ × ℚ := (1/3, 0)

/-- First line equation: y = -3x + 1 -/
def line1 (x y : ℚ) : Prop := y = -3 * x + 1

/-- Second line equation: y + 5 = 15x - 2 -/
def line2 (x y : ℚ) : Prop := y + 5 = 15 * x - 2

/-- Theorem stating that the intersection_point is the unique intersection of line1 and line2 -/
theorem intersection_theorem : 
  (∃! p : ℚ × ℚ, line1 p.1 p.2 ∧ line2 p.1 p.2) ∧ 
  (line1 intersection_point.1 intersection_point.2) ∧ 
  (line2 intersection_point.1 intersection_point.2) := by
  sorry

end NUMINAMATH_CALUDE_intersection_theorem_l2217_221705


namespace NUMINAMATH_CALUDE_quadratic_roots_ratio_l2217_221713

theorem quadratic_roots_ratio (p q α β : ℝ) (h1 : α + β = p) (h2 : α * β = 6) 
  (h3 : x^2 - p*x + q = 0 → x = α ∨ x = β) (h4 : p^2 ≠ 12) : 
  (α + β) / (α^2 + β^2) = p / (p^2 - 12) := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_ratio_l2217_221713


namespace NUMINAMATH_CALUDE_union_complement_equals_set_l2217_221757

universe u

def U : Finset ℕ := {0,1,2,4,6,8}
def M : Finset ℕ := {0,4,6}
def N : Finset ℕ := {0,1,6}

theorem union_complement_equals_set : M ∪ (U \ N) = {0,2,4,6,8} := by sorry

end NUMINAMATH_CALUDE_union_complement_equals_set_l2217_221757


namespace NUMINAMATH_CALUDE_existence_of_special_numbers_l2217_221772

theorem existence_of_special_numbers :
  ∃ (a b c : ℕ), 
    a > 10^10 ∧ b > 10^10 ∧ c > 10^10 ∧
    (∃ (k₁ k₂ k₃ : ℕ), 
      a * b * c = k₁ * (a + 2012) ∧
      a * b * c = k₂ * (b + 2012) ∧
      a * b * c = k₃ * (c + 2012)) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_special_numbers_l2217_221772


namespace NUMINAMATH_CALUDE_two_points_theorem_l2217_221791

/-- Represents the three possible states of a point in the bun -/
inductive PointState
  | Type1
  | Type2
  | NoRaisin

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The bun as a bounded 3D space -/
def Bun : Set Point3D :=
  sorry

/-- Function that determines the state of a point in the bun -/
def pointState : Point3D → PointState :=
  sorry

/-- Distance between two points in 3D space -/
def distance (p q : Point3D) : ℝ :=
  sorry

theorem two_points_theorem :
  ∃ (p q : Point3D), p ∈ Bun ∧ q ∈ Bun ∧ distance p q = 1 ∧
    (pointState p = pointState q ∨ (pointState p = PointState.NoRaisin ∧ pointState q = PointState.NoRaisin)) :=
  sorry

end NUMINAMATH_CALUDE_two_points_theorem_l2217_221791


namespace NUMINAMATH_CALUDE_ratio_B_to_C_l2217_221797

def total_amount : ℕ := 578
def share_A : ℕ := 408
def share_B : ℕ := 102
def share_C : ℕ := 68

theorem ratio_B_to_C :
  (share_B : ℚ) / share_C = 3 / 2 := by sorry

end NUMINAMATH_CALUDE_ratio_B_to_C_l2217_221797


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2217_221722

theorem absolute_value_inequality (x : ℝ) : 3 ≤ |x + 2| ∧ |x + 2| ≤ 7 ↔ (1 ≤ x ∧ x ≤ 5) ∨ (-9 ≤ x ∧ x ≤ -5) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2217_221722


namespace NUMINAMATH_CALUDE_solution_set_f_geq_1_range_of_m_l2217_221764

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 1| - |x - 2|

-- Theorem for the solution set of f(x) ≥ 1
theorem solution_set_f_geq_1 :
  {x : ℝ | f x ≥ 1} = {x : ℝ | x ≥ 1} :=
sorry

-- Theorem for the range of m
theorem range_of_m :
  {m : ℝ | ∀ x, |m - 2| ≥ |f x|} = {m : ℝ | m ≥ 5 ∨ m ≤ -1} :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_geq_1_range_of_m_l2217_221764


namespace NUMINAMATH_CALUDE_sum_of_ages_l2217_221727

/-- The sum of Jed and Matt's present ages given their age relationship and Jed's future age -/
theorem sum_of_ages (jed_age matt_age : ℕ) : 
  jed_age = matt_age + 10 →  -- Jed is 10 years older than Matt
  jed_age + 10 = 25 →        -- In 10 years, Jed will be 25 years old
  jed_age + matt_age = 20 :=  -- The sum of their present ages is 20
by sorry

end NUMINAMATH_CALUDE_sum_of_ages_l2217_221727


namespace NUMINAMATH_CALUDE_fan_ratio_theorem_l2217_221733

/-- Represents the number of fans for each team -/
structure FanCounts where
  yankees : ℕ
  mets : ℕ
  red_sox : ℕ

/-- The ratio of NY Yankees fans to NY Mets fans is 3:2 -/
def yankees_mets_ratio (fc : FanCounts) : Prop :=
  fc.yankees * 2 = fc.mets * 3

/-- The total number of fans is 390 -/
def total_fans (fc : FanCounts) : Prop :=
  fc.yankees + fc.mets + fc.red_sox = 390

/-- There are 104 NY Mets fans -/
def mets_fans_count (fc : FanCounts) : Prop :=
  fc.mets = 104

/-- The ratio of NY Mets fans to Boston Red Sox fans is 4:5 -/
def mets_red_sox_ratio (fc : FanCounts) : Prop :=
  fc.mets * 5 = fc.red_sox * 4

theorem fan_ratio_theorem (fc : FanCounts) :
  yankees_mets_ratio fc → total_fans fc → mets_fans_count fc → mets_red_sox_ratio fc := by
  sorry

end NUMINAMATH_CALUDE_fan_ratio_theorem_l2217_221733


namespace NUMINAMATH_CALUDE_derivative_of_f_l2217_221758

noncomputable def f (x : ℝ) : ℝ := x + Real.sin x

theorem derivative_of_f (x : ℝ) : 
  deriv f x = 1 + Real.cos x := by sorry

end NUMINAMATH_CALUDE_derivative_of_f_l2217_221758


namespace NUMINAMATH_CALUDE_alice_bob_number_sum_l2217_221730

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

theorem alice_bob_number_sum :
  ∀ (A B : ℕ),
    A ∈ Finset.range 50 →
    B ∈ Finset.range 50 →
    (∀ x ∈ Finset.range 50, x ≠ A → ¬(A > x ↔ B > x)) →
    (∀ y ∈ Finset.range 50, y ≠ B → (B > y ↔ A < y)) →
    is_prime B →
    B % 2 = 0 →
    is_perfect_square (90 * B + A) →
    A + B = 18 :=
by sorry

end NUMINAMATH_CALUDE_alice_bob_number_sum_l2217_221730


namespace NUMINAMATH_CALUDE_calculation_difference_l2217_221755

theorem calculation_difference : 
  (0.70 * 120 - ((6/9) * 150 / (0.80 * 250))) - (0.18 * 180 * (5/7) * 210) = -4776.5 := by
  sorry

end NUMINAMATH_CALUDE_calculation_difference_l2217_221755


namespace NUMINAMATH_CALUDE_sams_remaining_marbles_l2217_221736

/-- Given Sam's initial yellow marble count and the number of yellow marbles Joan took,
    prove that Sam's remaining yellow marble count is the difference between the two. -/
theorem sams_remaining_marbles (initial_count : ℕ) (marbles_taken : ℕ) 
    (h : marbles_taken ≤ initial_count) :
  initial_count - marbles_taken = initial_count - marbles_taken :=
by
  sorry

#check sams_remaining_marbles 86 25

end NUMINAMATH_CALUDE_sams_remaining_marbles_l2217_221736


namespace NUMINAMATH_CALUDE_square_sum_from_difference_and_product_l2217_221726

theorem square_sum_from_difference_and_product (x y : ℝ) 
  (h1 : (x - y)^2 = 49) 
  (h2 : x * y = -12) : 
  x^2 + y^2 = 25 := by
sorry

end NUMINAMATH_CALUDE_square_sum_from_difference_and_product_l2217_221726


namespace NUMINAMATH_CALUDE_square_area_ratio_l2217_221767

/-- Given three squares where each square's side is the diagonal of the next,
    the ratio of the largest square's area to the smallest square's area is 4 -/
theorem square_area_ratio (s₁ s₂ s₃ : ℝ) (h₁ : s₁ = s₂ * Real.sqrt 2) (h₂ : s₂ = s₃ * Real.sqrt 2) :
  s₁^2 / s₃^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l2217_221767


namespace NUMINAMATH_CALUDE_sphere_radius_and_area_l2217_221708

/-- A sphere with a chord creating a hollow on its surface -/
structure SphereWithHollow where
  radius : ℝ
  hollowDiameter : ℝ
  hollowDepth : ℝ

/-- The theorem about the sphere's radius and surface area given the hollow dimensions -/
theorem sphere_radius_and_area (s : SphereWithHollow) 
  (h1 : s.hollowDiameter = 12)
  (h2 : s.hollowDepth = 2) :
  s.radius = 10 ∧ 4 * Real.pi * s.radius^2 = 400 * Real.pi := by
  sorry

#check sphere_radius_and_area

end NUMINAMATH_CALUDE_sphere_radius_and_area_l2217_221708


namespace NUMINAMATH_CALUDE_jakes_motorcycle_purchase_l2217_221711

theorem jakes_motorcycle_purchase (initial_amount : ℝ) (motorcycle_cost : ℝ) (final_amount : ℝ) :
  initial_amount = 5000 ∧
  final_amount = 825 ∧
  final_amount = (initial_amount - motorcycle_cost) / 2 * 3 / 4 →
  motorcycle_cost = 2800 := by
sorry

end NUMINAMATH_CALUDE_jakes_motorcycle_purchase_l2217_221711


namespace NUMINAMATH_CALUDE_exactly_three_solutions_l2217_221796

-- Define S(n) as the sum of digits of n
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

-- Define the main equation
def satisfiesEquation (n : ℕ) : Prop :=
  n + sumOfDigits n + sumOfDigits (sumOfDigits n) = 2023

-- Theorem statement
theorem exactly_three_solutions :
  ∃! (s : Finset ℕ), s.card = 3 ∧ ∀ n, n ∈ s ↔ satisfiesEquation n :=
sorry

end NUMINAMATH_CALUDE_exactly_three_solutions_l2217_221796


namespace NUMINAMATH_CALUDE_min_y_over_x_on_ellipse_l2217_221718

theorem min_y_over_x_on_ellipse :
  ∀ x y : ℝ, 4 * (x - 2)^2 + y^2 = 4 →
  ∃ k : ℝ, k = -2/3 * Real.sqrt 3 ∧ ∀ z : ℝ, z = y / x → z ≥ k := by
  sorry

end NUMINAMATH_CALUDE_min_y_over_x_on_ellipse_l2217_221718


namespace NUMINAMATH_CALUDE_average_of_multiples_of_four_l2217_221714

theorem average_of_multiples_of_four : 
  let numbers := (Finset.range 33).filter (fun n => (n + 8) % 4 = 0)
  let sum := numbers.sum (fun n => n + 8)
  let count := numbers.card
  sum / count = 22 := by sorry

end NUMINAMATH_CALUDE_average_of_multiples_of_four_l2217_221714


namespace NUMINAMATH_CALUDE_equation_roots_problem_l2217_221765

/-- Given two equations with specific root conditions, prove the value of 100c + d -/
theorem equation_roots_problem (c d : ℝ) : 
  (∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    ∀ (x : ℝ), (x + c) * (x + d) * (x + 15) = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃) →
  (∀ (x : ℝ), x ≠ -4 → (x + c) * (x + d) * (x + 15) ≠ 0) →
  (∃! (x : ℝ), (x + 3*c) * (x + 4) * (x + 9) = 0 ∧ (x + d) * (x + 15) ≠ 0) →
  100 * c + d = -291 := by
sorry

end NUMINAMATH_CALUDE_equation_roots_problem_l2217_221765


namespace NUMINAMATH_CALUDE_problem_solution_l2217_221793

theorem problem_solution (x a : ℝ) :
  (a > 0) →
  (∀ x, (x^2 - 4*x + 3 < 0 ∧ x^2 - x - 12 ≤ 0 ∧ x^2 + 2*x - 8 > 0) → (2 < x ∧ x < 3)) ∧
  ((∀ x, (x^2 - 4*a*x + 3*a^2 ≥ 0) → (x^2 - x - 12 > 0 ∨ x^2 + 2*x - 8 ≤ 0)) ∧
   (∃ x, (x^2 - x - 12 > 0 ∨ x^2 + 2*x - 8 ≤ 0) ∧ x^2 - 4*a*x + 3*a^2 < 0) →
   (1 ≤ a ∧ a ≤ 2)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2217_221793


namespace NUMINAMATH_CALUDE_gerbils_sold_l2217_221787

theorem gerbils_sold (initial_gerbils : ℕ) (difference : ℕ) (h1 : initial_gerbils = 68) (h2 : difference = 54) :
  initial_gerbils - difference = 14 := by
  sorry

end NUMINAMATH_CALUDE_gerbils_sold_l2217_221787
